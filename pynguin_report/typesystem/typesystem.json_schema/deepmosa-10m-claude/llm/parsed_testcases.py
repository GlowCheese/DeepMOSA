####################################################################
#    TEST GENERATION BEGINS (DEEPMOSA + claude-haiku-4-5 t=0.8)    #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_to_json_schema_with_definitions. Retrieved 3/7 statements.
# Partially parsed test_to_json_schema_with_reference_field. Retrieved 2/5 statements.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True

import typesystem.composites as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.NeverMatch(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    assert var_2 is False

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['string', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minLength']
    assert var_4 == 5

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maxLength']
    assert var_4 == 10

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.String(allow_blank=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minLength']
    assert var_4 == 1

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'integer'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Integer(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['integer', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = {}
    var_2 = module_0.Integer(minimum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minimum']
    assert var_4 == 0

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Integer(maximum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maximum']
    assert var_4 == 100

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Float(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'number'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Decimal(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'number'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'boolean'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['boolean', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Array(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'array'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Array(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['array', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Array(min_items=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minItems']
    assert var_4 == 1

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Array(max_items=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maxItems']
    assert var_4 == 10

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['items']['type']
    assert var_5 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Array(unique_items=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['uniqueItems']
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'object'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Object(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['object', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = 'properties'
    var_8 = bool('properties' in var_6)
    assert var_8 is True
    var_9 = 'name'
    var_10 = bool('name' in var_6['properties'])
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_0]
    var_5 = {}
    var_6 = module_0.Object(properties=var_3, required=var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = var_7['required']
    var_9 = bool(var_7['required'] == ['name'])
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

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
    var_9 = module_1.to_json_schema(var_8)
    var_10 = var_9['enum']
    var_11 = bool(var_9['enum'] == ['a', 'b'])
    assert var_11 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'constant_value'
    var_1 = {}
    var_2 = module_0.Const(var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['const']
    assert var_4 == 'constant_value'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = 'anyOf'
    var_9 = bool('anyOf' in var_7)
    assert var_9 is True
    var_10 = 'anyOf'
    var_11 = var_7[var_10]
    var_12 = len(var_11)
    assert var_12 == 2

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'default_value'
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['default']
    assert var_5 == 'default_value'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'MyString'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_3]
    var_5 = 'components'
    var_6 = 'schemas'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 'MyString'
    var_3 = 'target'
    var_4 = {var_3: var_1}
    var_5 = 'components'
    var_6 = 'MyString'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_to_json_schema_with_definitions. Retrieved 3/8 statements.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True

import typesystem.composites as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.NeverMatch(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    assert var_2 is False

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['string', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'test_default'
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['default']
    assert var_5 == 'test_default'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minLength']
    assert var_4 == 5

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maxLength']
    assert var_4 == 10

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.String(allow_blank=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minLength']
    assert var_4 == 1

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'integer'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Integer(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['integer', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Float(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'number'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'boolean'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['boolean', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    assert var_5 == 'array'
    var_6 = var_4['items']['type']
    assert var_6 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 2
    var_3 = {}
    var_4 = module_0.Array(var_1, min_items=var_2, **var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = var_5['minItems']
    assert var_6 == 2

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 5
    var_3 = {}
    var_4 = module_0.Array(var_1, max_items=var_2, **var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = var_5['maxItems']
    assert var_6 == 5

import typesystem.fields as module_0
import typesystem.json_schema as module_1

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
    var_9 = module_1.to_json_schema(var_8)
    var_10 = var_9['enum']
    var_11 = bool(var_9['enum'] == ['a', 'b'])
    assert var_11 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = var_1 | var_3
    var_5 = module_1.to_json_schema(var_4)
    var_6 = 'anyOf'
    var_7 = bool('anyOf' in var_5)
    assert var_7 is True
    var_8 = 'anyOf'
    var_9 = var_5[var_8]
    var_10 = len(var_9)
    assert var_10 == 2

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_3]
    var_5 = 'components'
    var_6 = 'schemas'
    var_7 = 'name'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = var_6['type']
    assert var_7 == 'object'
    var_8 = 'properties'
    var_9 = bool('properties' in var_6)
    assert var_9 is True
    var_10 = 'name'
    var_11 = bool('name' in var_6['properties'])
    assert var_11 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_0]
    var_5 = {}
    var_6 = module_0.Object(properties=var_3, required=var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = var_7['required']
    var_9 = bool(var_7['required'] == ['name'])
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'fixed_value'
    var_1 = {}
    var_2 = module_0.Const(var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['const']
    assert var_4 == 'fixed_value'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_to_json_schema_with_definitions. Retrieved 3/7 statements.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True

import typesystem.composites as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.NeverMatch(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    assert var_2 is False

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['string', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['default']
    assert var_5 == 'test'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minLength']
    assert var_4 == 5

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maxLength']
    assert var_4 == 10

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.String(allow_blank=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minLength']
    assert var_4 == 1

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'integer'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Integer(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['integer', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = {}
    var_2 = module_0.Integer(minimum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minimum']
    assert var_4 == 0

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Integer(maximum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maximum']
    assert var_4 == 100

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Float(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'number'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'boolean'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['boolean', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Array(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'array'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    assert var_5 == 'array'
    var_6 = var_4['items']['type']
    assert var_6 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Array(min_items=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minItems']
    assert var_4 == 1

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Array(max_items=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maxItems']
    assert var_4 == 10

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'object'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = var_6['type']
    assert var_7 == 'object'
    var_8 = 'properties'
    var_9 = bool('properties' in var_6)
    assert var_9 is True
    var_10 = var_6['properties']['name']['type']
    assert var_10 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

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
    var_9 = module_1.to_json_schema(var_8)
    var_10 = var_9['enum']
    var_11 = bool(var_9['enum'] == ['a', 'b'])
    assert var_11 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'fixed_value'
    var_1 = {}
    var_2 = module_0.Const(var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['const']
    assert var_4 == 'fixed_value'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = var_1 | var_3
    var_5 = module_1.to_json_schema(var_4)
    var_6 = 'anyOf'
    var_7 = bool('anyOf' in var_5)
    assert var_7 is True
    var_8 = 'anyOf'
    var_9 = var_5[var_8]
    var_10 = len(var_9)
    assert var_10 == 2

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'user'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_3]



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_from_json_schema_type_number. Retrieved 15/21 statements.
# Partially parsed test_from_json_schema_type_integer. Retrieved 9/15 statements.
# Partially parsed test_from_json_schema_type_string. Retrieved 13/19 statements.
# Partially parsed test_from_json_schema_type_string_allow_blank. Retrieved 5/11 statements.
# Partially parsed test_from_json_schema_type_boolean. Retrieved 5/11 statements.
# Partially parsed test_from_json_schema_type_array_with_items_object. Retrieved 17/25 statements.
# Partially parsed test_from_json_schema_type_array_with_items_list. Retrieved 12/26 statements.
# Partially parsed test_from_json_schema_type_array_no_items. Retrieved 3/9 statements.
# Partially parsed test_from_json_schema_type_array_with_additional_items_field. Retrieved 10/18 statements.
# Partially parsed test_from_json_schema_type_object_with_properties. Retrieved 18/28 statements.
# Partially parsed test_from_json_schema_type_object_with_pattern_properties. Retrieved 9/17 statements.


def test_case_0():
    var_0 = []
    var_1 = 'minimum'
    var_2 = 'maximum'
    var_3 = 'exclusiveMinimum'
    var_4 = 'exclusiveMaximum'
    var_5 = 'multipleOf'
    var_6 = 'default'
    var_7 = 0
    var_8 = 100
    var_9 = -1
    var_10 = 101
    var_11 = 5
    var_12 = 50
    var_13 = {var_1: var_7, var_2: var_8, var_3: var_9, var_4: var_10, var_5: var_11, var_6: var_12}
    var_14 = 'number'
    var_15 = False

def test_case_0():
    var_0 = []
    var_1 = 'minimum'
    var_2 = 'maximum'
    var_3 = 'default'
    var_4 = 1
    var_5 = 10
    var_6 = 5
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = 'integer'
    var_9 = True

def test_case_0():
    var_0 = []
    var_1 = 'minLength'
    var_2 = 'maxLength'
    var_3 = 'pattern'
    var_4 = 'format'
    var_5 = 'default'
    var_6 = 2
    var_7 = 50
    var_8 = '^[a-z]+$'
    var_9 = 'email'
    var_10 = 'test@example.com'
    var_11 = {var_1: var_6, var_2: var_7, var_3: var_8, var_4: var_9, var_5: var_10}
    var_12 = 'string'
    var_13 = False

def test_case_0():
    var_0 = []
    var_1 = 'minLength'
    var_2 = 0
    var_3 = {var_1: var_2}
    var_4 = 'string'
    var_5 = False

def test_case_0():
    var_0 = []
    var_1 = 'default'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = 'boolean'
    var_5 = False

def test_case_0():
    var_0 = []
    var_1 = 'items'
    var_2 = 'minItems'
    var_3 = 'maxItems'
    var_4 = 'uniqueItems'
    var_5 = 'default'
    var_6 = 'type'
    var_7 = 'string'
    var_8 = {var_6: var_7}
    var_9 = 1
    var_10 = 10
    var_11 = True
    var_12 = 'a'
    var_13 = 'b'
    var_14 = [var_12, var_13]
    var_15 = {var_1: var_8, var_2: var_9, var_3: var_10, var_4: var_11, var_5: var_14}
    var_16 = 'array'
    var_17 = False

def test_case_0():
    var_0 = []
    var_1 = 'items'
    var_2 = 'additionalItems'
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = 'integer'
    var_7 = {var_3: var_6}
    var_8 = [var_5, var_7]
    var_9 = False
    var_10 = {var_1: var_8, var_2: var_9}
    var_11 = 'array'
    var_12 = 1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = 'array'
    var_3 = False

def test_case_0():
    var_0 = []
    var_1 = 'items'
    var_2 = 'additionalItems'
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = [var_5]
    var_7 = {var_3: var_4}
    var_8 = {var_1: var_6, var_2: var_7}
    var_9 = 'array'
    var_10 = False

def test_case_0():
    var_0 = []
    var_1 = 'properties'
    var_2 = 'required'
    var_3 = 'minProperties'
    var_4 = 'maxProperties'
    var_5 = 'name'
    var_6 = 'age'
    var_7 = 'type'
    var_8 = 'string'
    var_9 = {var_7: var_8}
    var_10 = 'integer'
    var_11 = {var_7: var_10}
    var_12 = {var_5: var_9, var_6: var_11}
    var_13 = [var_5]
    var_14 = 1
    var_15 = 5
    var_16 = {var_1: var_12, var_2: var_13, var_3: var_14, var_4: var_15}
    var_17 = 'object'
    var_18 = False
    var_19 = 'name'
    var_20 = 'age'

def test_case_0():
    var_0 = []
    var_1 = 'patternProperties'
    var_2 = '^S_'
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = 'object'
    var_9 = False
    var_10 = '^S_'

def test_case_0():
    pass



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_predicate_at_line_70_evaluates_to_false. Retrieved 7/11 statements.


def test_case_0():
    var_0 = 'additionalItems'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'array'
    var_6 = False
    var_7 = []



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_to_json_schema_with_definitions. Retrieved 3/7 statements.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True

import typesystem.composites as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.NeverMatch(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    assert var_2 is False

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['string', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minLength']
    assert var_4 == 5

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maxLength']
    assert var_4 == 10

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.String(allow_blank=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minLength']
    assert var_4 == 1

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'integer'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Integer(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['integer', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Float(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'number'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Decimal(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'number'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'boolean'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['boolean', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    assert var_5 == 'array'
    var_6 = var_4['items']['type']
    assert var_6 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 2
    var_3 = {}
    var_4 = module_0.Array(var_1, min_items=var_2, **var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = var_5['minItems']
    assert var_6 == 2

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 5
    var_3 = {}
    var_4 = module_0.Array(var_1, max_items=var_2, **var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = var_5['maxItems']
    assert var_6 == 5

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = True
    var_3 = {}
    var_4 = module_0.Array(var_1, unique_items=var_2, **var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = var_5['uniqueItems']
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = var_6['type']
    assert var_7 == 'object'
    var_8 = 'properties'
    var_9 = bool('properties' in var_6)
    assert var_9 is True
    var_10 = var_6['properties']['name']['type']
    assert var_10 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_0]
    var_5 = {}
    var_6 = module_0.Object(properties=var_3, required=var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = var_7['required']
    var_9 = bool(var_7['required'] == ['name'])
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = 'anyOf'
    var_9 = bool('anyOf' in var_7)
    assert var_9 is True
    var_10 = 'anyOf'
    var_11 = var_7[var_10]
    var_12 = len(var_11)
    assert var_12 == 2

import typesystem.fields as module_0
import typesystem.json_schema as module_1

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
    var_9 = module_1.to_json_schema(var_8)
    var_10 = var_9['enum']
    var_11 = bool(var_9['enum'] == ['a', 'b'])
    assert var_11 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'fixed_value'
    var_1 = {}
    var_2 = module_0.Const(var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['const']
    assert var_4 == 'fixed_value'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'default_value'
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['default']
    assert var_5 == 'default_value'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'StringType'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_3]
    var_5 = 'components'
    var_6 = 'schemas'
    var_7 = 'StringType'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = {}
    var_2 = module_0.Integer(minimum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minimum']
    assert var_4 == 0

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Integer(maximum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maximum']
    assert var_4 == 100

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = {}
    var_2 = module_0.Integer(exclusive_minimum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['exclusiveMinimum']
    assert var_4 == 0

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Integer(exclusive_maximum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['exclusiveMaximum']
    assert var_4 == 100

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.Integer(multiple_of=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['multipleOf']
    assert var_4 == 5



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_predicate_at_line_19_evaluates_to_true. Retrieved 4/5 statements.


import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'value1'
    var_2 = {var_0: var_1}
    var_3 = module_0.to_json_schema(var_2)
    var_4 = 'components'
    var_5 = bool('components' in var_3)
    assert var_5 is True
    var_6 = 'schemas'
    var_7 = bool('schemas' in var_3['components'])
    assert var_7 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_property_names_predicate_false. Retrieved 7/11 statements.


def test_case_0():
    var_0 = 'propertyNames'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'object'
    var_6 = False
    var_7 = []



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_from_json_schema_type_number. Retrieved 9/15 statements.
# Partially parsed test_from_json_schema_type_integer. Retrieved 7/13 statements.
# Partially parsed test_from_json_schema_type_string. Retrieved 9/15 statements.
# Partially parsed test_from_json_schema_type_string_allow_blank. Retrieved 5/11 statements.
# Partially parsed test_from_json_schema_type_boolean. Retrieved 3/9 statements.
# Partially parsed test_from_json_schema_type_array_simple. Retrieved 11/19 statements.
# Partially parsed test_from_json_schema_type_array_no_items. Retrieved 3/9 statements.
# Partially parsed test_from_json_schema_type_array_tuple_validation. Retrieved 11/25 statements.
# Partially parsed test_from_json_schema_type_array_additional_items_bool. Retrieved 4/10 statements.
# Partially parsed test_from_json_schema_type_array_additional_items_schema. Retrieved 7/15 statements.
# Partially parsed test_from_json_schema_type_array_unique_items. Retrieved 5/11 statements.
# Partially parsed test_from_json_schema_type_object_simple. Retrieved 9/17 statements.
# Partially parsed test_from_json_schema_type_object_no_properties. Retrieved 3/9 statements.
# Partially parsed test_from_json_schema_type_object_pattern_properties. Retrieved 9/17 statements.
# Partially parsed test_from_json_schema_type_object_additional_properties_bool. Retrieved 4/10 statements.


def test_case_0():
    var_0 = []
    var_1 = 'minimum'
    var_2 = 'maximum'
    var_3 = 'multipleOf'
    var_4 = 0
    var_5 = 100
    var_6 = 5
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = 'number'
    var_9 = False

def test_case_0():
    var_0 = []
    var_1 = 'minimum'
    var_2 = 'maximum'
    var_3 = 1
    var_4 = 10
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'integer'
    var_7 = True

def test_case_0():
    var_0 = []
    var_1 = 'minLength'
    var_2 = 'maxLength'
    var_3 = 'pattern'
    var_4 = 2
    var_5 = 50
    var_6 = '^[a-z]+$'
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = 'string'
    var_9 = False

def test_case_0():
    var_0 = []
    var_1 = 'minLength'
    var_2 = 0
    var_3 = {var_1: var_2}
    var_4 = 'string'
    var_5 = False

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = 'boolean'
    var_3 = True

def test_case_0():
    var_0 = []
    var_1 = 'items'
    var_2 = 'minItems'
    var_3 = 'maxItems'
    var_4 = 'type'
    var_5 = 'string'
    var_6 = {var_4: var_5}
    var_7 = 1
    var_8 = 10
    var_9 = {var_1: var_6, var_2: var_7, var_3: var_8}
    var_10 = 'array'
    var_11 = False

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = 'array'
    var_3 = False

def test_case_0():
    var_0 = []
    var_1 = 'items'
    var_2 = 'type'
    var_3 = 'string'
    var_4 = {var_2: var_3}
    var_5 = 'integer'
    var_6 = {var_2: var_5}
    var_7 = [var_4, var_6]
    var_8 = {var_1: var_7}
    var_9 = 'array'
    var_10 = False
    var_11 = 1

def test_case_0():
    var_0 = []
    var_1 = 'additionalItems'
    var_2 = False
    var_3 = {var_1: var_2}
    var_4 = 'array'

def test_case_0():
    var_0 = []
    var_1 = 'additionalItems'
    var_2 = 'type'
    var_3 = 'string'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'array'
    var_7 = False

def test_case_0():
    var_0 = []
    var_1 = 'uniqueItems'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = 'array'
    var_5 = False

def test_case_0():
    var_0 = []
    var_1 = 'properties'
    var_2 = 'name'
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = 'object'
    var_9 = False
    var_10 = 'name'

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = 'object'
    var_3 = False

def test_case_0():
    var_0 = []
    var_1 = 'patternProperties'
    var_2 = '^S_'
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = 'object'
    var_9 = False
    var_10 = '^S_'

def test_case_0():
    var_0 = []
    var_1 = 'additionalProperties'
    var_2 = False
    var_3 = {var_1: var_2}
    var_4 = 'object'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_predicate_at_line_172_evaluates_to_true. Retrieved 20/25 statements.


def test_case_0():
    var_0 = 'TestSchema'
    var_1 = 'type'
    var_2 = 'object'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = True
    var_6 = var_5 and var_4
    assert var_6 is True
    var_7 = True
    var_8 = 'key'
    var_9 = 'value'
    var_10 = {var_8: var_9}
    var_11 = var_7 and var_10
    assert var_11 is True
    var_12 = True
    var_13 = 'Schema1'
    var_14 = 'Schema2'
    var_15 = {var_1: var_2}
    var_16 = 'string'
    var_17 = {var_1: var_16}
    var_18 = {var_13: var_15, var_14: var_17}
    var_19 = var_12 and var_18
    assert var_19 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_from_json_schema_type_number. Retrieved 9/15 statements.
# Partially parsed test_from_json_schema_type_number_with_null. Retrieved 3/9 statements.
# Partially parsed test_from_json_schema_type_integer. Retrieved 11/17 statements.
# Partially parsed test_from_json_schema_type_string. Retrieved 11/17 statements.
# Partially parsed test_from_json_schema_type_string_with_minlength_zero. Retrieved 5/11 statements.
# Partially parsed test_from_json_schema_type_string_with_minlength_one. Retrieved 5/11 statements.
# Partially parsed test_from_json_schema_type_boolean. Retrieved 3/9 statements.
# Partially parsed test_from_json_schema_type_boolean_with_null. Retrieved 4/10 statements.
# Partially parsed test_from_json_schema_type_array_simple. Retrieved 11/19 statements.
# Partially parsed test_from_json_schema_type_array_no_items. Retrieved 3/9 statements.
# Partially parsed test_from_json_schema_type_array_tuple_items. Retrieved 11/25 statements.
# Partially parsed test_from_json_schema_type_array_with_additional_items_bool. Retrieved 4/10 statements.
# Partially parsed test_from_json_schema_type_array_with_additional_items_schema. Retrieved 7/15 statements.
# Partially parsed test_from_json_schema_type_object_simple. Retrieved 15/23 statements.


def test_case_0():
    var_0 = 'minimum'
    var_1 = 'maximum'
    var_2 = 'multipleOf'
    var_3 = 0
    var_4 = 100
    var_5 = 5
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = []
    var_8 = 'number'
    var_9 = False

def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = 'number'
    var_3 = True

def test_case_0():
    var_0 = 'minimum'
    var_1 = 'maximum'
    var_2 = 'exclusiveMinimum'
    var_3 = 'exclusiveMaximum'
    var_4 = 1
    var_5 = 10
    var_6 = 0
    var_7 = 11
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = []
    var_10 = 'integer'
    var_11 = False

def test_case_0():
    var_0 = 'minLength'
    var_1 = 'maxLength'
    var_2 = 'pattern'
    var_3 = 'format'
    var_4 = 2
    var_5 = 50
    var_6 = '^[a-z]+$'
    var_7 = 'email'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = []
    var_10 = 'string'
    var_11 = False

def test_case_0():
    var_0 = 'minLength'
    var_1 = 0
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = 'string'
    var_5 = False

def test_case_0():
    var_0 = 'minLength'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = 'string'
    var_5 = False

def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = 'boolean'
    var_3 = False

def test_case_0():
    var_0 = 'default'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = 'boolean'

def test_case_0():
    var_0 = 'items'
    var_1 = 'minItems'
    var_2 = 'maxItems'
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = 1
    var_7 = 10
    var_8 = {var_0: var_5, var_1: var_6, var_2: var_7}
    var_9 = []
    var_10 = 'array'
    var_11 = False

def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = 'array'
    var_3 = False

def test_case_0():
    var_0 = 'items'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = 'integer'
    var_5 = {var_1: var_4}
    var_6 = [var_3, var_5]
    var_7 = {var_0: var_6}
    var_8 = []
    var_9 = 'array'
    var_10 = False
    var_11 = 1

def test_case_0():
    var_0 = 'additionalItems'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = 'array'

def test_case_0():
    var_0 = 'additionalItems'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = []
    var_6 = 'array'
    var_7 = False

def test_case_0():
    var_0 = 'properties'
    var_1 = 'required'
    var_2 = 'minProperties'
    var_3 = 'maxProperties'
    var_4 = 'name'
    var_5 = 'type'
    var_6 = 'string'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = [var_4]
    var_10 = 1
    var_11 = 5
    var_12 = {var_0: var_8, var_1: var_9, var_2: var_10, var_3: var_11}
    var_13 = []
    var_14 = 'object'
    var_15 = False
    var_16 = 'name'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_all_of_from_json_schema. Retrieved 11/24 statements.
# Partially parsed test_all_of_from_json_schema_with_default. Retrieved 11/19 statements.
# Partially parsed test_all_of_from_json_schema_multiple_constraints. Retrieved 14/22 statements.
# Partially parsed test_all_of_from_json_schema_empty. Retrieved 3/11 statements.
# Partially parsed test_all_of_from_json_schema_with_definitions. Retrieved 10/20 statements.


def test_case_0():
    var_0 = []
    var_1 = 'allOf'
    var_2 = 'type'
    var_3 = 'string'
    var_4 = {var_2: var_3}
    var_5 = 'minLength'
    var_6 = 5
    var_7 = {var_5: var_6}
    var_8 = [var_4, var_7]
    var_9 = {var_1: var_8}
    var_10 = 0
    var_11 = 1

def test_case_0():
    var_0 = []
    var_1 = 'allOf'
    var_2 = 'default'
    var_3 = 'type'
    var_4 = 'integer'
    var_5 = {var_3: var_4}
    var_6 = 'minimum'
    var_7 = 0
    var_8 = {var_6: var_7}
    var_9 = [var_5, var_8]
    var_10 = 42
    var_11 = {var_1: var_9, var_2: var_10}

def test_case_0():
    var_0 = []
    var_1 = 'allOf'
    var_2 = 'type'
    var_3 = 'maxLength'
    var_4 = 'string'
    var_5 = 100
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 'minLength'
    var_8 = 1
    var_9 = {var_7: var_8}
    var_10 = 'pattern'
    var_11 = '^[a-z]+$'
    var_12 = {var_10: var_11}
    var_13 = [var_6, var_9, var_12]
    var_14 = {var_1: var_13}

def test_case_0():
    var_0 = []
    var_1 = 'allOf'
    var_2 = []
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = {}
    var_3 = 'allOf'
    var_4 = 'type'
    var_5 = 'string'
    var_6 = {var_4: var_5}
    var_7 = '$ref'
    var_8 = '#/components/schemas/Name'
    var_9 = {var_7: var_8}
    var_10 = [var_6, var_9]
    var_11 = {var_3: var_10}



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_if_then_else_from_json_schema. Retrieved 30/44 statements.


def test_case_0():
    var_0 = []
    var_1 = 'if'
    var_2 = 'then'
    var_3 = 'else'
    var_4 = 'default'
    var_5 = 'type'
    var_6 = 'string'
    var_7 = {var_5: var_6}
    var_8 = 'integer'
    var_9 = {var_5: var_8}
    var_10 = 'boolean'
    var_11 = {var_5: var_10}
    var_12 = 'test_default'
    var_13 = {var_1: var_7, var_2: var_9, var_3: var_11, var_4: var_12}
    var_14 = {var_5: var_6}
    var_15 = 'default_value'
    var_16 = {var_1: var_14, var_4: var_15}
    var_17 = 'number'
    var_18 = {var_5: var_17}
    var_19 = {var_5: var_6}
    var_20 = 'another_default'
    var_21 = {var_1: var_18, var_2: var_19, var_4: var_20}
    var_22 = {var_5: var_10}
    var_23 = 'array'
    var_24 = {var_5: var_23}
    var_25 = 'yet_another_default'
    var_26 = {var_1: var_22, var_3: var_24, var_4: var_25}
    var_27 = {var_5: var_6}
    var_28 = {var_5: var_8}
    var_29 = {var_5: var_10}
    var_30 = {var_1: var_27, var_2: var_28, var_3: var_29}



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_min_length_predicate_line_39. Retrieved 11/21 statements.


def test_case_0():
    var_0 = 'minLength'
    var_1 = 2
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = 'string'
    var_5 = False
    var_6 = 1
    var_7 = {var_0: var_6}
    var_8 = []
    var_9 = {}
    var_10 = []
    var_11 = 5
    var_12 = {var_0: var_11}
    var_13 = []
    var_14 = True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_from_json_schema_type_number. Retrieved 11/14 statements.
# Partially parsed test_from_json_schema_type_integer. Retrieved 13/16 statements.
# Partially parsed test_from_json_schema_type_string. Retrieved 13/16 statements.
# Partially parsed test_from_json_schema_type_string_allow_blank. Retrieved 9/12 statements.
# Partially parsed test_from_json_schema_type_boolean. Retrieved 7/10 statements.
# Partially parsed test_from_json_schema_type_boolean_allow_null. Retrieved 5/8 statements.
# Partially parsed test_from_json_schema_type_array_with_items. Retrieved 15/18 statements.
# Partially parsed test_from_json_schema_type_array_no_items. Retrieved 5/8 statements.
# Partially parsed test_from_json_schema_type_array_with_additional_items. Retrieved 10/13 statements.
# Partially parsed test_from_json_schema_type_object_with_properties. Retrieved 20/23 statements.
# Partially parsed test_from_json_schema_type_object_no_properties. Retrieved 5/8 statements.
# Partially parsed test_from_json_schema_type_object_with_pattern_properties. Retrieved 11/14 statements.
# Partially parsed test_from_json_schema_type_object_with_additional_properties_false. Retrieved 6/9 statements.
# Partially parsed test_from_json_schema_type_object_with_additional_properties_schema. Retrieved 9/12 statements.
# Partially parsed test_from_json_schema_type_object_with_property_names. Retrieved 9/12 statements.
# Partially parsed test_from_json_schema_type_with_default. Retrieved 8/11 statements.


import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'minimum'
    var_1 = 'maximum'
    var_2 = 'multipleOf'
    var_3 = 0
    var_4 = 100
    var_5 = 5
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 'number'
    var_8 = False
    var_9 = {}
    var_10 = module_0.from_json_schema_type(var_6, var_7, var_8, var_9)
    var_11 = var_10.allow_null
    assert var_11 is False
    var_12 = var_10.minimum
    assert var_12 == 0
    var_13 = var_10.maximum
    assert var_13 == 100
    var_14 = var_10.multiple_of
    assert var_14 == 5
    var_15 = var_10.coerce_types
    assert var_15 is False

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'minimum'
    var_1 = 'maximum'
    var_2 = 'exclusiveMinimum'
    var_3 = 'exclusiveMaximum'
    var_4 = 1
    var_5 = 10
    var_6 = 0
    var_7 = 11
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = 'integer'
    var_10 = True
    var_11 = {}
    var_12 = module_0.from_json_schema_type(var_8, var_9, var_10, var_11)
    var_13 = var_12.allow_null
    assert var_13 is True
    var_14 = var_12.minimum
    assert var_14 == 1
    var_15 = var_12.maximum
    assert var_15 == 10
    var_16 = var_12.exclusive_minimum
    assert var_16 == 0
    var_17 = var_12.exclusive_maximum
    assert var_17 == 11
    var_18 = var_12.coerce_types
    assert var_18 is False

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'minLength'
    var_1 = 'maxLength'
    var_2 = 'pattern'
    var_3 = 'format'
    var_4 = 2
    var_5 = 50
    var_6 = '^[a-z]+$'
    var_7 = 'email'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = 'string'
    var_10 = False
    var_11 = {}
    var_12 = module_0.from_json_schema_type(var_8, var_9, var_10, var_11)
    var_13 = var_12.allow_null
    assert var_13 is False
    var_14 = var_12.min_length
    assert var_14 == 2
    var_15 = var_12.max_length
    assert var_15 == 50
    var_16 = var_12.pattern
    assert var_16 == '^[a-z]+$'
    var_17 = var_12.format
    assert var_17 == 'email'
    var_18 = var_12.coerce_types
    assert var_18 is False

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'minLength'
    var_1 = 'maxLength'
    var_2 = 0
    var_3 = 100
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'string'
    var_6 = False
    var_7 = {}
    var_8 = module_0.from_json_schema_type(var_4, var_5, var_6, var_7)
    var_9 = var_8.allow_blank
    assert var_9 is True
    var_10 = var_8.min_length
    assert var_10 is None

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'default'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'boolean'
    var_4 = False
    var_5 = {}
    var_6 = module_0.from_json_schema_type(var_2, var_3, var_4, var_5)
    var_7 = var_6.allow_null
    assert var_7 is False
    var_8 = var_6.coerce_types
    assert var_8 is False

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'boolean'
    var_2 = True
    var_3 = {}
    var_4 = module_0.from_json_schema_type(var_0, var_1, var_2, var_3)
    var_5 = var_4.allow_null
    assert var_5 is True

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'items'
    var_1 = 'minItems'
    var_2 = 'maxItems'
    var_3 = 'uniqueItems'
    var_4 = 'type'
    var_5 = 'integer'
    var_6 = {var_4: var_5}
    var_7 = 1
    var_8 = 10
    var_9 = True
    var_10 = {var_0: var_6, var_1: var_7, var_2: var_8, var_3: var_9}
    var_11 = 'array'
    var_12 = False
    var_13 = {}
    var_14 = module_0.from_json_schema_type(var_10, var_11, var_12, var_13)
    var_15 = var_14.allow_null
    assert var_15 is False
    var_16 = var_14.min_items
    assert var_16 == 1
    var_17 = var_14.max_items
    assert var_17 == 10
    var_18 = var_14.unique_items
    assert var_18 is True

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'array'
    var_2 = False
    var_3 = {}
    var_4 = module_0.from_json_schema_type(var_0, var_1, var_2, var_3)
    var_5 = var_4.items
    assert var_5 is None

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'items'
    var_1 = 'additionalItems'
    var_2 = 'type'
    var_3 = 'string'
    var_4 = {var_2: var_3}
    var_5 = False
    var_6 = {var_0: var_4, var_1: var_5}
    var_7 = 'array'
    var_8 = {}
    var_9 = module_0.from_json_schema_type(var_6, var_7, var_5, var_8)
    var_10 = var_9.additional_items
    assert var_10 is False

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'properties'
    var_1 = 'required'
    var_2 = 'minProperties'
    var_3 = 'maxProperties'
    var_4 = 'name'
    var_5 = 'age'
    var_6 = 'type'
    var_7 = 'string'
    var_8 = {var_6: var_7}
    var_9 = 'integer'
    var_10 = {var_6: var_9}
    var_11 = {var_4: var_8, var_5: var_10}
    var_12 = [var_4]
    var_13 = 1
    var_14 = 10
    var_15 = {var_0: var_11, var_1: var_12, var_2: var_13, var_3: var_14}
    var_16 = 'object'
    var_17 = False
    var_18 = {}
    var_19 = module_0.from_json_schema_type(var_15, var_16, var_17, var_18)
    var_20 = var_19.allow_null
    assert var_20 is False
    var_21 = 'name'
    var_22 = bool('name' in var_19.properties)
    assert var_22 is True
    var_23 = 'age'
    var_24 = bool('age' in var_19.properties)
    assert var_24 is True
    var_25 = var_19.required
    var_26 = bool(var_19.required == ['name'])
    assert var_26 is True
    var_27 = var_19.min_properties
    assert var_27 == 1
    var_28 = var_19.max_properties
    assert var_28 == 10

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'object'
    var_2 = True
    var_3 = {}
    var_4 = module_0.from_json_schema_type(var_0, var_1, var_2, var_3)
    var_5 = var_4.allow_null
    assert var_5 is True
    var_6 = var_4.properties
    var_7 = bool(var_4.properties == {})
    assert var_7 is True

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'patternProperties'
    var_1 = '^S_'
    var_2 = 'type'
    var_3 = 'string'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'object'
    var_8 = False
    var_9 = {}
    var_10 = module_0.from_json_schema_type(var_6, var_7, var_8, var_9)
    var_11 = '^S_'
    var_12 = bool('^S_' in var_10.pattern_properties)
    assert var_12 is True

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'additionalProperties'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = 'object'
    var_4 = {}
    var_5 = module_0.from_json_schema_type(var_2, var_3, var_1, var_4)
    var_6 = var_5.additional_properties
    assert var_6 is False

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'additionalProperties'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'object'
    var_6 = False
    var_7 = {}
    var_8 = module_0.from_json_schema_type(var_4, var_5, var_6, var_7)
    var_9 = var_8.additional_properties
    var_10 = bool(var_8.additional_properties is not None)
    assert var_10 is True

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'propertyNames'
    var_1 = 'pattern'
    var_2 = '^[a-z]+$'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'object'
    var_6 = False
    var_7 = {}
    var_8 = module_0.from_json_schema_type(var_4, var_5, var_6, var_7)
    var_9 = var_8.property_names
    var_10 = bool(var_8.property_names is not None)
    assert var_10 is True

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'default'
    var_1 = 'hello'
    var_2 = {var_0: var_1}
    var_3 = 'string'
    var_4 = False
    var_5 = {}
    var_6 = module_0.from_json_schema_type(var_2, var_3, var_4, var_5)
    var_7 = var_6.get_default_value()
    assert var_7 == 'hello'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_from_json_schema_boolean_true. Retrieved 2/3 statements.
# Partially parsed test_from_json_schema_boolean_false. Retrieved 2/3 statements.
# Partially parsed test_from_json_schema_with_type_string. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_with_type_integer. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_with_type_number. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_with_type_boolean. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_with_type_array. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_with_type_object. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_with_type_null. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_with_multiple_types. Retrieved 6/7 statements.
# Partially parsed test_from_json_schema_with_enum. Retrieved 7/8 statements.
# Partially parsed test_from_json_schema_with_const. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_with_all_of. Retrieved 10/11 statements.
# Partially parsed test_from_json_schema_with_any_of. Retrieved 9/10 statements.
# Partially parsed test_from_json_schema_with_one_of. Retrieved 9/10 statements.
# Partially parsed test_from_json_schema_with_not. Retrieved 6/7 statements.
# Partially parsed test_from_json_schema_with_if_then_else. Retrieved 13/14 statements.
# Partially parsed test_from_json_schema_with_ref. Retrieved 3/7 statements.
# Partially parsed test_from_json_schema_with_components. Retrieved 10/11 statements.
# Partially parsed test_from_json_schema_with_null_type. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_with_type_and_null. Retrieved 6/7 statements.
# Partially parsed test_from_json_schema_empty_object. Retrieved 2/3 statements.
# Partially parsed test_from_json_schema_multiple_constraints. Retrieved 8/9 statements.
# Partially parsed test_from_json_schema_with_string_properties. Retrieved 12/13 statements.
# Partially parsed test_from_json_schema_with_array_items. Retrieved 7/8 statements.
# Partially parsed test_from_json_schema_with_min_max_string. Retrieved 8/9 statements.
# Partially parsed test_from_json_schema_with_pattern. Retrieved 6/7 statements.


import typesystem.json_schema as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.from_json_schema(var_0)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.from_json_schema(var_0)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'string'
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'integer'
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'number'
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'boolean'
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'array'
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'object'
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'null'
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'string'
    var_2 = 'integer'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = module_0.from_json_schema(var_4)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'enum'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = module_0.from_json_schema(var_5)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'const'
    var_1 = 42
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'allOf'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = 'minLength'
    var_5 = 1
    var_6 = {var_4: var_5}
    var_7 = [var_3, var_6]
    var_8 = {var_0: var_7}
    var_9 = module_0.from_json_schema(var_8)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'anyOf'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = 'integer'
    var_5 = {var_1: var_4}
    var_6 = [var_3, var_5]
    var_7 = {var_0: var_6}
    var_8 = module_0.from_json_schema(var_7)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'oneOf'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = 'integer'
    var_5 = {var_1: var_4}
    var_6 = [var_3, var_5]
    var_7 = {var_0: var_6}
    var_8 = module_0.from_json_schema(var_7)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'not'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = module_0.from_json_schema(var_4)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'if'
    var_1 = 'then'
    var_2 = 'else'
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = 'minLength'
    var_7 = 1
    var_8 = {var_6: var_7}
    var_9 = 'integer'
    var_10 = {var_3: var_9}
    var_11 = {var_0: var_5, var_1: var_8, var_2: var_10}
    var_12 = module_0.from_json_schema(var_11)

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = '$ref'
    var_3 = '#/components/schemas/MySchema'
    var_4 = {var_2: var_3}

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'components'
    var_1 = 'schemas'
    var_2 = 'MySchema'
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = module_0.from_json_schema(var_8)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'null'
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'string'
    var_2 = 'null'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = module_0.from_json_schema(var_4)
    var_6 = var_5.allow_null
    assert var_6 is True

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.from_json_schema(var_0)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'enum'
    var_2 = 'string'
    var_3 = 'a'
    var_4 = 'b'
    var_5 = [var_3, var_4]
    var_6 = {var_0: var_2, var_1: var_5}
    var_7 = module_0.from_json_schema(var_6)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'properties'
    var_2 = 'object'
    var_3 = 'name'
    var_4 = 'age'
    var_5 = 'string'
    var_6 = {var_0: var_5}
    var_7 = 'integer'
    var_8 = {var_0: var_7}
    var_9 = {var_3: var_6, var_4: var_8}
    var_10 = {var_0: var_2, var_1: var_9}
    var_11 = module_0.from_json_schema(var_10)
    var_12 = 'name'
    var_13 = bool('name' in var_11.properties)
    assert var_13 is True
    var_14 = 'age'
    var_15 = bool('age' in var_11.properties)
    assert var_15 is True

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'items'
    var_2 = 'array'
    var_3 = 'string'
    var_4 = {var_0: var_3}
    var_5 = {var_0: var_2, var_1: var_4}
    var_6 = module_0.from_json_schema(var_5)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'minLength'
    var_2 = 'maxLength'
    var_3 = 'string'
    var_4 = 5
    var_5 = 10
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = module_0.from_json_schema(var_6)
    var_8 = var_7.min_length
    assert var_8 == 5
    var_9 = var_7.max_length
    assert var_9 == 10

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'pattern'
    var_2 = 'string'
    var_3 = '^[a-z]+$'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.from_json_schema(var_4)
    var_6 = var_5.pattern
    assert var_6 == '^[a-z]+$'



# Parsed testcases at query #17
#--------------------------




def test_case_0():
    var_0 = '$ref'
    var_1 = '#/components/schemas/SomeSchema'
    var_2 = {var_0: var_1}
    var_3 = '$ref'
    var_4 = bool('$ref' in var_2)
    assert var_4 is True
    var_5 = 'type'
    var_6 = 'string'
    var_7 = {var_5: var_6}
    var_8 = '$ref'
    var_9 = bool('$ref' not in var_7)
    assert var_9 is True
    var_10 = {}
    var_11 = '$ref'
    var_12 = bool('$ref' not in var_10)
    assert var_12 is True
    var_13 = '#/components/schemas/Test'
    var_14 = 'object'
    var_15 = {var_0: var_13, var_5: var_14}
    var_16 = '$ref'
    var_17 = bool('$ref' in var_15)
    assert var_17 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_pattern_properties_predicate_evaluates_to_true. Retrieved 1/14 statements.


def test_case_0():
    var_0 = '^S_'



# Parsed testcases at query #19
#--------------------------




import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_schema_with_allow_null_false. Retrieved 9/32 statements.


def test_case_0():
    var_0 = 'Schema'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = False
    var_6 = var_0 is var_5
    var_7 = 'object'
    var_8 = 'null'
    var_9 = [var_7, var_8]
    var_10 = var_7 if var_6 else var_9
    assert var_10 == 'object'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_to_json_schema_with_string_field. Retrieved 2/4 statements.
# Partially parsed test_to_json_schema_with_schema_field. Retrieved 1/6 statements.
# Partially parsed test_to_json_schema_with_schema_field_nullable. Retrieved 2/7 statements.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True

import typesystem.composites as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.NeverMatch(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    assert var_2 is False

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['string', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minLength']
    assert var_4 == 5

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maxLength']
    assert var_4 == 10

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.String(allow_blank=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minLength']
    assert var_4 == 1

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'integer'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Integer(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['integer', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Float(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'number'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Decimal(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'number'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = {}
    var_2 = module_0.Integer(minimum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minimum']
    assert var_4 == 0

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Integer(maximum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maximum']
    assert var_4 == 100

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = {}
    var_2 = module_0.Integer(exclusive_minimum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['exclusiveMinimum']
    assert var_4 == 0

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Integer(exclusive_maximum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['exclusiveMaximum']
    assert var_4 == 100

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.Integer(multiple_of=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['multipleOf']
    assert var_4 == 5

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'boolean'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['boolean', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    assert var_5 == 'array'
    var_6 = 'items'
    var_7 = bool('items' in var_4)
    assert var_7 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = True
    var_3 = 'allow_null'
    var_4 = {var_3: var_2}
    var_5 = module_0.Array(var_1, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = var_6['type']
    var_8 = bool(var_6['type'] == ['array', 'null'])
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 1
    var_3 = {}
    var_4 = module_0.Array(var_1, min_items=var_2, **var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = var_5['minItems']
    assert var_6 == 1

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 10
    var_3 = {}
    var_4 = module_0.Array(var_1, max_items=var_2, **var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = var_5['maxItems']
    assert var_6 == 10

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = True
    var_3 = {}
    var_4 = module_0.Array(var_1, unique_items=var_2, **var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = var_5['uniqueItems']
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = var_6['type']
    assert var_7 == 'object'
    var_8 = 'properties'
    var_9 = bool('properties' in var_6)
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = True
    var_5 = 'allow_null'
    var_6 = {var_5: var_4}
    var_7 = module_0.Object(properties=var_3, **var_6)
    var_8 = module_1.to_json_schema(var_7)
    var_9 = var_8['type']
    var_10 = bool(var_8['type'] == ['object', 'null'])
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_0]
    var_5 = {}
    var_6 = module_0.Object(properties=var_3, required=var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = var_7['required']
    var_9 = bool(var_7['required'] == ['name'])
    assert var_9 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 'properties'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

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
    var_9 = module_1.to_json_schema(var_8)
    var_10 = var_9['enum']
    var_11 = bool(var_9['enum'] == ['a', 'b'])
    assert var_11 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'fixed_value'
    var_1 = {}
    var_2 = module_0.Const(var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['const']
    assert var_4 == 'fixed_value'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_string_field_with_format_not_none. Retrieved 7/25 statements.


def test_case_0():
    var_0 = 'allow_null'
    var_1 = 'min_length'
    var_2 = 'allow_blank'
    var_3 = 'max_length'
    var_4 = 'pattern_regex'
    var_5 = 'format'
    var_6 = [var_0, var_1, var_2, var_3, var_4, var_5]
    var_7 = 'format'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_if_then_else_from_json_schema_without_else_clause. Retrieved 8/15 statements.


def test_case_0():
    var_0 = []
    var_1 = 'if'
    var_2 = 'then'
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = 'integer'
    var_7 = {var_3: var_6}
    var_8 = {var_1: var_5, var_2: var_7}



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_schema_with_allow_null_true. Retrieved 3/11 statements.
# Partially parsed test_schema_with_allow_null_false. Retrieved 3/11 statements.


def test_case_0():
    var_0 = 'object'
    var_1 = 'null'
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 'object'
    var_1 = 'null'
    var_2 = [var_0, var_1]



# Parsed testcases at query #25
#--------------------------

# Failed to parse test_predicate_line_71_evaluates_to_true.




# Parsed testcases at query #26
#--------------------------

# Partially parsed test_to_json_schema_with_definitions. Retrieved 3/7 statements.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True

import typesystem.composites as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.NeverMatch(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    assert var_2 is False

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['string', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minLength']
    assert var_4 == 5

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maxLength']
    assert var_4 == 10

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.String(allow_blank=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minLength']
    assert var_4 == 1

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'integer'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Integer(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['integer', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = {}
    var_2 = module_0.Integer(minimum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minimum']
    assert var_4 == 0

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Integer(maximum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maximum']
    assert var_4 == 100

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Float(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'number'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Decimal(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'number'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'boolean'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['boolean', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Array(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'array'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Array(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['array', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Array(min_items=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minItems']
    assert var_4 == 1

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Array(max_items=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maxItems']
    assert var_4 == 10

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['items']['type']
    assert var_5 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Array(var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = 'items'
    var_9 = var_7[var_8]
    var_10 = len(var_9)
    assert var_10 == 2
    var_11 = var_7['items'][0]['type']
    assert var_11 == 'string'
    var_12 = var_7['items'][1]['type']
    assert var_12 == 'integer'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Array(unique_items=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['uniqueItems']
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'object'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Object(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['object', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = 'properties'
    var_8 = bool('properties' in var_6)
    assert var_8 is True
    var_9 = 'name'
    var_10 = bool('name' in var_6['properties'])
    assert var_10 is True
    var_11 = var_6['properties']['name']['type']
    assert var_11 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Object(required=var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['required']
    var_6 = bool(var_4['required'] == ['name'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 1
    var_1 = 'One'
    var_2 = (var_0, var_1)
    var_3 = 2
    var_4 = 'Two'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = {}
    var_8 = module_0.Choice(choices=var_6, **var_7)
    var_9 = module_1.to_json_schema(var_8)
    var_10 = var_9['enum']
    var_11 = bool(var_9['enum'] == [1, 2])
    assert var_11 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'constant_value'
    var_1 = {}
    var_2 = module_0.Const(var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['const']
    assert var_4 == 'constant_value'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = 'anyOf'
    var_9 = bool('anyOf' in var_7)
    assert var_9 is True
    var_10 = 'anyOf'
    var_11 = var_7[var_10]
    var_12 = len(var_11)
    assert var_12 == 2

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'default_value'
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['default']
    assert var_5 == 'default_value'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_3]
    var_5 = 'components'
    var_6 = 'schemas'
    var_7 = 'Name'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_if_then_else_from_json_schema_without_then_clause. Retrieved 9/14 statements.


def test_case_0():
    var_0 = 'if'
    var_1 = 'else'
    var_2 = 'type'
    var_3 = 'string'
    var_4 = {var_2: var_3}
    var_5 = 'integer'
    var_6 = {var_2: var_5}
    var_7 = {var_0: var_4, var_1: var_6}
    var_8 = []
    var_9 = 'test'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_from_json_schema_with_ref_in_data. Retrieved 3/9 statements.


def test_case_0():
    var_0 = []
    var_1 = '#/components/schemas/User'
    var_2 = '$ref'
    var_3 = {var_2: var_1}



# Parsed testcases at query #29
#--------------------------

# Failed to parse test_predicate_at_line_112_evaluates_to_true.




# Parsed testcases at query #30
#--------------------------

# Partially parsed test_line_60_isinstance_items_list_predicate. Retrieved 9/14 statements.


def test_case_0():
    var_0 = 'items'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = 'integer'
    var_5 = {var_1: var_4}
    var_6 = [var_3, var_5]
    var_7 = {var_0: var_6}
    var_8 = []
    var_9 = None



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_then_clause_is_none. Retrieved 1/12 statements.


def test_case_0():
    var_0 = None



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_else_clause_is_none. Retrieved 1/13 statements.


def test_case_0():
    var_0 = None



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_to_json_schema_with_string_field. Retrieved 2/5 statements.
# Partially parsed test_to_json_schema_with_union_field. Retrieved 4/7 statements.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True

import typesystem.composites as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.NeverMatch(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    assert var_2 is False

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['string', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['default']
    assert var_5 == 'test'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minLength']
    assert var_4 == 5

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maxLength']
    assert var_4 == 10

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.String(allow_blank=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minLength']
    assert var_4 == 1

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'integer'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Integer(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['integer', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = {}
    var_2 = module_0.Integer(minimum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minimum']
    assert var_4 == 0

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Integer(maximum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maximum']
    assert var_4 == 100

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Float(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'number'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Decimal(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'number'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'boolean'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['boolean', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    assert var_5 == 'array'
    var_6 = 'items'
    var_7 = bool('items' in var_4)
    assert var_7 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 1
    var_3 = {}
    var_4 = module_0.Array(var_1, min_items=var_2, **var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = var_5['minItems']
    assert var_6 == 1

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 10
    var_3 = {}
    var_4 = module_0.Array(var_1, max_items=var_2, **var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = var_5['maxItems']
    assert var_6 == 10

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = True
    var_3 = {}
    var_4 = module_0.Array(var_1, unique_items=var_2, **var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = var_5['uniqueItems']
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = var_6['type']
    assert var_7 == 'object'
    var_8 = 'properties'
    var_9 = bool('properties' in var_6)
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_0]
    var_5 = {}
    var_6 = module_0.Object(properties=var_3, required=var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = var_7['required']
    var_9 = bool(var_7['required'] == ['name'])
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

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
    var_9 = module_1.to_json_schema(var_8)
    var_10 = var_9['enum']
    var_11 = bool(var_9['enum'] == ['a', 'b'])
    assert var_11 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'fixed_value'
    var_1 = {}
    var_2 = module_0.Const(var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['const']
    assert var_4 == 'fixed_value'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)



# Parsed testcases at query #34
#--------------------------

# Failed to parse test_predicate_line_158_evaluates_to_false.




# Parsed testcases at query #35
#--------------------------

# Partially parsed test_from_json_schema_type_array_with_list_items. Retrieved 10/18 statements.


def test_case_0():
    var_0 = []
    var_1 = 'items'
    var_2 = 'type'
    var_3 = 'string'
    var_4 = {var_2: var_3}
    var_5 = 'integer'
    var_6 = {var_2: var_5}
    var_7 = [var_4, var_6]
    var_8 = {var_1: var_7}
    var_9 = 'array'
    var_10 = False



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_if_then_else_from_json_schema_without_else_clause. Retrieved 8/12 statements.


def test_case_0():
    var_0 = []
    var_1 = 'if'
    var_2 = 'then'
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = 'integer'
    var_7 = {var_3: var_6}
    var_8 = {var_1: var_5, var_2: var_7}



# Parsed testcases at query #37
#--------------------------




import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True

import typesystem.composites as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.NeverMatch(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    assert var_2 is False

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['string', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['default']
    assert var_5 == 'test'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minLength']
    assert var_4 == 5

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maxLength']
    assert var_4 == 10

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.String(allow_blank=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minLength']
    assert var_4 == 1

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'integer'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Integer(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['integer', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Float(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'number'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Decimal(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'number'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = {}
    var_2 = module_0.Integer(minimum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minimum']
    assert var_4 == 0

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Integer(maximum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maximum']
    assert var_4 == 100

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'boolean'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['boolean', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Array(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'array'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Array(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['array', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Array(min_items=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minItems']
    assert var_4 == 1

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Array(max_items=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maxItems']
    assert var_4 == 10

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['items']['type']
    assert var_5 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Array(unique_items=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['uniqueItems']
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'object'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Object(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['object', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = 'properties'
    var_8 = bool('properties' in var_6)
    assert var_8 is True
    var_9 = 'name'
    var_10 = bool('name' in var_6['properties'])
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.Object(additional_properties=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['additionalProperties']
    assert var_4 is False

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Object(required=var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['required']
    var_6 = bool(var_4['required'] == ['name'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

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
    var_9 = module_1.to_json_schema(var_8)
    var_10 = var_9['enum']
    var_11 = bool(var_9['enum'] == ['a', 'b'])
    assert var_11 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'constant_value'
    var_1 = {}
    var_2 = module_0.Const(var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['const']
    assert var_4 == 'constant_value'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = 'anyOf'
    var_9 = bool('anyOf' in var_7)
    assert var_9 is True
    var_10 = 'anyOf'
    var_11 = var_7[var_10]
    var_12 = len(var_11)
    assert var_12 == 2

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_1.OneOf(var_4, **var_5)
    var_7 = module_2.to_json_schema(var_6)
    var_8 = 'oneOf'
    var_9 = bool('oneOf' in var_7)
    assert var_9 is True
    var_10 = 'oneOf'
    var_11 = var_7[var_10]
    var_12 = len(var_11)
    assert var_12 == 2

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_1.AllOf(var_2, **var_3)
    var_5 = module_2.to_json_schema(var_4)
    var_6 = 'allOf'
    var_7 = bool('allOf' in var_5)
    assert var_7 is True
    var_8 = 'allOf'
    var_9 = var_5[var_8]
    var_10 = len(var_9)
    var_11 = bool(var_10)
    assert var_11 is True



# Parsed testcases at query #38
#--------------------------




import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'email'
    var_1 = {}
    var_2 = module_0.String(format=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['format']
    assert var_4 == 'email'
    var_5 = var_3['type']
    assert var_5 == 'string'



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_to_json_schema_with_array_field_items_list. Retrieved 9/11 statements.
# Partially parsed test_to_json_schema_with_definitions. Retrieved 3/7 statements.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True

import typesystem.composites as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.NeverMatch(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    assert var_2 is False

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['string', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minLength']
    assert var_4 == 5

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maxLength']
    assert var_4 == 10

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.String(allow_blank=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minLength']
    assert var_4 == 1

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'integer'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Integer(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['integer', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = {}
    var_2 = module_0.Integer(minimum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minimum']
    assert var_4 == 0

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Integer(maximum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maximum']
    assert var_4 == 100

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Float(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'number'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Decimal(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'number'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'boolean'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['boolean', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Array(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'array'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Array(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['array', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Array(min_items=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minItems']
    assert var_4 == 1

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Array(max_items=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maxItems']
    assert var_4 == 10

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Array(unique_items=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['uniqueItems']
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['items']['type']
    assert var_5 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Array(var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = 'items'
    var_9 = var_7[var_8]
    var_10 = var_7[var_8]
    var_11 = len(var_10)
    assert var_11 == 2

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'object'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Object(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['object', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = 'properties'
    var_8 = bool('properties' in var_6)
    assert var_8 is True
    var_9 = 'name'
    var_10 = bool('name' in var_6['properties'])
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_0]
    var_5 = {}
    var_6 = module_0.Object(properties=var_3, required=var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = var_7['required']
    var_9 = bool(var_7['required'] == ['name'])
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

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
    var_9 = module_1.to_json_schema(var_8)
    var_10 = var_9['enum']
    var_11 = bool(var_9['enum'] == ['a', 'b'])
    assert var_11 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'fixed_value'
    var_1 = {}
    var_2 = module_0.Const(var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['const']
    assert var_4 == 'fixed_value'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = 'anyOf'
    var_9 = bool('anyOf' in var_7)
    assert var_9 is True
    var_10 = 'anyOf'
    var_11 = var_7[var_10]
    var_12 = len(var_11)
    assert var_12 == 2

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'default_value'
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['default']
    assert var_5 == 'default_value'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_3]
    var_5 = 'components'
    var_6 = 'schemas'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_ref_from_json_schema_valid_reference. Retrieved 3/8 statements.
# Partially parsed test_ref_from_json_schema_with_nested_path. Retrieved 3/8 statements.
# Partially parsed test_ref_from_json_schema_invalid_ref_style. Retrieved 3/8 statements.
# Partially parsed test_ref_from_json_schema_with_absolute_url. Retrieved 3/8 statements.
# Partially parsed test_ref_from_json_schema_preserves_definitions_reference. Retrieved 3/8 statements.


def test_case_0():
    var_0 = []
    var_1 = '$ref'
    var_2 = '#/definitions/User'
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = []
    var_1 = '$ref'
    var_2 = '#/components/schemas/Product'
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = []
    var_1 = '$ref'
    var_2 = 'definitions/User'
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Unsupported $ref style in document.'

def test_case_0():
    var_0 = []
    var_1 = '$ref'
    var_2 = 'http://example.com/schema'
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Unsupported $ref style in document.'

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = '$ref'
    var_3 = '#/User'
    var_4 = {var_2: var_3}



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_if_then_else_from_json_schema_without_then_clause. Retrieved 8/12 statements.


def test_case_0():
    var_0 = []
    var_1 = 'if'
    var_2 = 'else'
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = 'number'
    var_7 = {var_3: var_6}
    var_8 = {var_1: var_5, var_2: var_7}



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_to_json_schema_with_definitions. Retrieved 3/8 statements.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True

import typesystem.composites as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.NeverMatch(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    assert var_2 is False

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['string', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minLength']
    assert var_4 == 5

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maxLength']
    assert var_4 == 10

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'integer'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Integer(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['integer', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = {}
    var_2 = module_0.Integer(minimum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minimum']
    assert var_4 == 0

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Integer(maximum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maximum']
    assert var_4 == 100

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Float(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'number'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'boolean'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['boolean', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    assert var_5 == 'array'
    var_6 = var_4['items']['type']
    assert var_6 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 1
    var_3 = {}
    var_4 = module_0.Array(var_1, min_items=var_2, **var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = var_5['minItems']
    assert var_6 == 1

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 10
    var_3 = {}
    var_4 = module_0.Array(var_1, max_items=var_2, **var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = var_5['maxItems']
    assert var_6 == 10

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = var_1 | var_3
    var_5 = module_1.to_json_schema(var_4)
    var_6 = 'anyOf'
    var_7 = bool('anyOf' in var_5)
    assert var_7 is True
    var_8 = 'anyOf'
    var_9 = var_5[var_8]
    var_10 = len(var_9)
    assert var_10 == 2

import typesystem.fields as module_0
import typesystem.json_schema as module_1

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
    var_9 = module_1.to_json_schema(var_8)
    var_10 = var_9['enum']
    var_11 = bool(var_9['enum'] == ['a', 'b'])
    assert var_11 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'default_value'
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['default']
    assert var_5 == 'default_value'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.String(allow_blank=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minLength']
    assert var_4 == 1

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = var_6['type']
    assert var_7 == 'object'
    var_8 = 'properties'
    var_9 = bool('properties' in var_6)
    assert var_9 is True
    var_10 = 'name'
    var_11 = bool('name' in var_6['properties'])
    assert var_11 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_3]
    var_5 = 'components'
    var_6 = 'schemas'
    var_7 = 'name'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'constant_value'
    var_1 = {}
    var_2 = module_0.Const(var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['const']
    assert var_4 == 'constant_value'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.Integer(multiple_of=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['multipleOf']
    assert var_4 == 5

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = True
    var_3 = {}
    var_4 = module_0.Array(var_1, unique_items=var_2, **var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = var_5['uniqueItems']
    assert var_6 is True



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_predicate_at_line_158_evaluates_to_false. Retrieved 1/14 statements.


def test_case_0():
    var_0 = None



# Parsed testcases at query #44
#--------------------------

# Failed to parse test_to_json_schema_with_definitions.
# Partially parsed test_to_json_schema_with_reference_field. Retrieved 2/6 statements.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True

import typesystem.composites as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.NeverMatch(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    assert var_2 is False

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['string', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minLength']
    assert var_4 == 5

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maxLength']
    assert var_4 == 10

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.String(allow_blank=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minLength']
    assert var_4 == 1

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'integer'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Integer(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['integer', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = {}
    var_2 = module_0.Integer(minimum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minimum']
    assert var_4 == 0

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Integer(maximum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maximum']
    assert var_4 == 100

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Float(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'number'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'boolean'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['boolean', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    assert var_5 == 'array'
    var_6 = var_4['items']['type']
    assert var_6 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 1
    var_3 = {}
    var_4 = module_0.Array(var_1, min_items=var_2, **var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = var_5['minItems']
    assert var_6 == 1

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 10
    var_3 = {}
    var_4 = module_0.Array(var_1, max_items=var_2, **var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = var_5['maxItems']
    assert var_6 == 10

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = var_6['type']
    assert var_7 == 'object'
    var_8 = 'name'
    var_9 = bool('name' in var_6['properties'])
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_0]
    var_5 = {}
    var_6 = module_0.Object(properties=var_3, required=var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = var_7['required']
    var_9 = bool(var_7['required'] == ['name'])
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

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
    var_9 = module_1.to_json_schema(var_8)
    var_10 = var_9['enum']
    var_11 = bool(var_9['enum'] == ['a', 'b'])
    assert var_11 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = var_1 | var_3
    var_5 = module_1.to_json_schema(var_4)
    var_6 = 'anyOf'
    var_7 = bool('anyOf' in var_5)
    assert var_7 is True
    var_8 = 'anyOf'
    var_9 = var_5[var_8]
    var_10 = len(var_9)
    assert var_10 == 2

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['default']
    assert var_5 == 'test'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 'User'
    var_3 = 'target'
    var_4 = {var_3: var_1}
    var_5 = 'components'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'fixed_value'
    var_1 = {}
    var_2 = module_0.Const(var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['const']
    assert var_4 == 'fixed_value'



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_predicate_at_line_78_evaluates_to_false. Retrieved 1/6 statements.


def test_case_0():
    var_0 = None



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_schema_field_with_allow_null_false. Retrieved 1/3 statements.


def test_case_0():
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_predicate_at_line_78_evaluates_to_false. Retrieved 1/6 statements.


def test_case_0():
    var_0 = None



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_ref_from_json_schema_unsupported_ref_style. Retrieved 5/9 statements.


def test_case_0():
    var_0 = []
    var_1 = '$ref'
    var_2 = 'external/schema'
    var_3 = {var_1: var_2}
    var_4 = False
    var_5 = True
    var_6 = bool(var_5)
    assert var_6 is True
    var_7 = 'Unsupported $ref style in document.'



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_from_json_schema_with_ref_in_data. Retrieved 3/8 statements.


def test_case_0():
    var_0 = []
    var_1 = '$ref'
    var_2 = '#/components/schemas/TestSchema'
    var_3 = {var_1: var_2}



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_to_json_schema_with_definitions. Retrieved 3/8 statements.
# Partially parsed test_to_json_schema_with_reference_field. Retrieved 2/6 statements.
# Partially parsed test_to_json_schema_with_ifthenelse_field. Retrieved 2/5 statements.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True

import typesystem.composites as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.NeverMatch(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    assert var_2 is False

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['string', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = {}
    var_3 = module_0.String(max_length=var_1, min_length=var_0, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    assert var_5 == 'string'
    var_6 = var_4['minLength']
    assert var_6 == 1
    var_7 = var_4['maxLength']
    assert var_7 == 10

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'integer'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = 100
    var_2 = {}
    var_3 = module_0.Integer(minimum=var_0, maximum=var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    assert var_5 == 'integer'
    var_6 = var_4['minimum']
    assert var_6 == 0
    var_7 = var_4['maximum']
    assert var_7 == 100

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Float(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'number'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'boolean'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    assert var_5 == 'array'
    var_6 = 'items'
    var_7 = bool('items' in var_4)
    assert var_7 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 1
    var_3 = 5
    var_4 = {}
    var_5 = module_0.Array(var_1, min_items=var_2, max_items=var_3, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = var_6['type']
    assert var_7 == 'array'
    var_8 = var_6['minItems']
    assert var_8 == 1
    var_9 = var_6['maxItems']
    assert var_9 == 5

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = var_6['type']
    assert var_7 == 'object'
    var_8 = 'properties'
    var_9 = bool('properties' in var_6)
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_0]
    var_5 = {}
    var_6 = module_0.Object(properties=var_3, required=var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = var_7['type']
    assert var_8 == 'object'
    var_9 = var_7['required']
    var_10 = bool(var_7['required'] == ['name'])
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

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
    var_9 = module_1.to_json_schema(var_8)
    var_10 = var_9['enum']
    var_11 = bool(var_9['enum'] == ['a', 'b'])
    assert var_11 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = 'anyOf'
    var_9 = bool('anyOf' in var_7)
    assert var_9 is True
    var_10 = 'anyOf'
    var_11 = var_7[var_10]
    var_12 = len(var_11)
    assert var_12 == 2

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['default']
    assert var_5 == 'hello'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_3]
    var_5 = 'components'
    var_6 = 'schemas'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 'StringType'
    var_3 = 'target'
    var_4 = {var_3: var_1}
    var_5 = 'components'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'fixed_value'
    var_1 = {}
    var_2 = module_0.Const(var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['const']
    assert var_4 == 'fixed_value'

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_1.OneOf(var_4, **var_5)
    var_7 = module_2.to_json_schema(var_6)
    var_8 = 'oneOf'
    var_9 = bool('oneOf' in var_7)
    assert var_9 is True
    var_10 = 'oneOf'
    var_11 = var_7[var_10]
    var_12 = len(var_11)
    assert var_12 == 2

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_1.AllOf(var_2, **var_3)
    var_5 = module_2.to_json_schema(var_4)
    var_6 = 'allOf'
    var_7 = bool('allOf' in var_5)
    assert var_7 is True
    var_8 = 'allOf'
    var_9 = var_5[var_8]
    var_10 = len(var_9)
    assert var_10 == 1

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_1.Not(var_1, **var_2)
    var_4 = module_2.to_json_schema(var_3)
    var_5 = 'not'
    var_6 = bool('not' in var_4)
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_1.IfThenElse(var_1, **var_2)



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_to_json_schema_with_definitions. Retrieved 3/7 statements.
# Partially parsed test_to_json_schema_with_reference_field. Retrieved 2/5 statements.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True

import typesystem.composites as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.NeverMatch(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    assert var_2 is False

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['string', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minLength']
    assert var_4 == 5

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maxLength']
    assert var_4 == 10

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'integer'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Integer(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['integer', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = {}
    var_2 = module_0.Integer(minimum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minimum']
    assert var_4 == 0

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Integer(maximum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maximum']
    assert var_4 == 100

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Float(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'number'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'boolean'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['boolean', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    assert var_5 == 'array'
    var_6 = var_4['items']['type']
    assert var_6 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 1
    var_3 = {}
    var_4 = module_0.Array(var_1, min_items=var_2, **var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = var_5['minItems']
    assert var_6 == 1

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 10
    var_3 = {}
    var_4 = module_0.Array(var_1, max_items=var_2, **var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = var_5['maxItems']
    assert var_6 == 10

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = var_6['type']
    assert var_7 == 'object'
    var_8 = 'name'
    var_9 = bool('name' in var_6['properties'])
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

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
    var_9 = module_1.to_json_schema(var_8)
    var_10 = var_9['enum']
    var_11 = bool(var_9['enum'] == ['a', 'b'])
    assert var_11 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = 'anyOf'
    var_9 = bool('anyOf' in var_7)
    assert var_9 is True
    var_10 = 'anyOf'
    var_11 = var_7[var_10]
    var_12 = len(var_11)
    assert var_12 == 2

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['default']
    assert var_5 == 'test'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_3]
    var_5 = 'components'
    var_6 = 'schemas'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 'TestRef'
    var_3 = 'target'
    var_4 = {var_3: var_1}
    var_5 = 'components'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.String(allow_blank=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minLength']
    assert var_4 == 1

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.Integer(exclusive_minimum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['exclusiveMinimum']
    assert var_4 == 5

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Integer(exclusive_maximum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['exclusiveMaximum']
    assert var_4 == 100

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.Integer(multiple_of=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['multipleOf']
    assert var_4 == 5

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = True
    var_3 = 'allow_null'
    var_4 = {var_3: var_2}
    var_5 = module_0.Array(var_1, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = var_6['type']
    var_8 = bool(var_6['type'] == ['array', 'null'])
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = True
    var_5 = 'allow_null'
    var_6 = {var_5: var_4}
    var_7 = module_0.Object(properties=var_3, **var_6)
    var_8 = module_1.to_json_schema(var_7)
    var_9 = var_8['type']
    var_10 = bool(var_8['type'] == ['object', 'null'])
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_0]
    var_5 = {}
    var_6 = module_0.Object(properties=var_3, required=var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = var_7['required']
    var_9 = bool(var_7['required'] == ['name'])
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = {}
    var_5 = module_0.Array(var_3, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = var_6['type']
    assert var_7 == 'array'
    var_8 = var_6['items']['type']
    assert var_8 == 'array'
    var_9 = var_6['items']['items']['type']
    assert var_9 == 'string'



# Parsed testcases at query #52
#--------------------------

# Failed to parse test_then_clause_is_none.




# Parsed testcases at query #53
#--------------------------

# Partially parsed test_predicate_at_line_33_evaluates_to_true. Retrieved 2/22 statements.


def test_case_0():
    var_0 = 'test_pattern'
    var_1 = None



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_else_clause_is_none. Retrieved 1/9 statements.


def test_case_0():
    var_0 = None



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_from_json_schema_type_number. Retrieved 11/14 statements.
# Partially parsed test_from_json_schema_type_number_with_null. Retrieved 5/8 statements.
# Partially parsed test_from_json_schema_type_integer. Retrieved 13/16 statements.
# Partially parsed test_from_json_schema_type_string. Retrieved 13/16 statements.
# Partially parsed test_from_json_schema_type_string_min_length_zero. Retrieved 7/10 statements.
# Partially parsed test_from_json_schema_type_string_min_length_one. Retrieved 7/10 statements.
# Partially parsed test_from_json_schema_type_boolean. Retrieved 7/10 statements.
# Partially parsed test_from_json_schema_type_boolean_with_null. Retrieved 5/8 statements.
# Partially parsed test_from_json_schema_type_array. Retrieved 15/18 statements.
# Partially parsed test_from_json_schema_type_array_with_additional_items. Retrieved 6/9 statements.
# Partially parsed test_from_json_schema_type_array_items_list. Retrieved 15/19 statements.
# Partially parsed test_from_json_schema_type_object. Retrieved 20/23 statements.
# Partially parsed test_from_json_schema_type_object_with_pattern_properties. Retrieved 11/14 statements.
# Partially parsed test_from_json_schema_type_object_with_additional_properties_bool. Retrieved 6/9 statements.
# Partially parsed test_from_json_schema_type_object_with_property_names. Retrieved 9/12 statements.


import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'minimum'
    var_1 = 'maximum'
    var_2 = 'multipleOf'
    var_3 = 0
    var_4 = 100
    var_5 = 5
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 'number'
    var_8 = False
    var_9 = {}
    var_10 = module_0.from_json_schema_type(var_6, var_7, var_8, var_9)
    var_11 = var_10.allow_null
    assert var_11 is False
    var_12 = var_10.minimum
    assert var_12 == 0
    var_13 = var_10.maximum
    assert var_13 == 100
    var_14 = var_10.multiple_of
    assert var_14 == 5
    var_15 = var_10.coerce_types
    assert var_15 is False

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'number'
    var_2 = True
    var_3 = {}
    var_4 = module_0.from_json_schema_type(var_0, var_1, var_2, var_3)
    var_5 = var_4.allow_null
    assert var_5 is True

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'minimum'
    var_1 = 'maximum'
    var_2 = 'exclusiveMinimum'
    var_3 = 'exclusiveMaximum'
    var_4 = 1
    var_5 = 50
    var_6 = 0
    var_7 = 51
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = 'integer'
    var_10 = False
    var_11 = {}
    var_12 = module_0.from_json_schema_type(var_8, var_9, var_10, var_11)
    var_13 = var_12.allow_null
    assert var_13 is False
    var_14 = var_12.minimum
    assert var_14 == 1
    var_15 = var_12.maximum
    assert var_15 == 50
    var_16 = var_12.exclusive_minimum
    assert var_16 == 0
    var_17 = var_12.exclusive_maximum
    assert var_17 == 51

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'minLength'
    var_1 = 'maxLength'
    var_2 = 'pattern'
    var_3 = 'format'
    var_4 = 2
    var_5 = 100
    var_6 = '^[a-z]+$'
    var_7 = 'email'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = 'string'
    var_10 = False
    var_11 = {}
    var_12 = module_0.from_json_schema_type(var_8, var_9, var_10, var_11)
    var_13 = var_12.allow_null
    assert var_13 is False
    var_14 = var_12.min_length
    assert var_14 == 2
    var_15 = var_12.max_length
    assert var_15 == 100
    var_16 = var_12.pattern
    assert var_16 == '^[a-z]+$'
    var_17 = var_12.format
    assert var_17 == 'email'
    var_18 = var_12.allow_blank
    assert var_18 is False

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'minLength'
    var_1 = 0
    var_2 = {var_0: var_1}
    var_3 = 'string'
    var_4 = False
    var_5 = {}
    var_6 = module_0.from_json_schema_type(var_2, var_3, var_4, var_5)
    var_7 = var_6.allow_blank
    assert var_7 is True
    var_8 = var_6.min_length
    assert var_8 is None

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'minLength'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'string'
    var_4 = False
    var_5 = {}
    var_6 = module_0.from_json_schema_type(var_2, var_3, var_4, var_5)
    var_7 = var_6.min_length
    assert var_7 is None

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'default'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'boolean'
    var_4 = False
    var_5 = {}
    var_6 = module_0.from_json_schema_type(var_2, var_3, var_4, var_5)
    var_7 = var_6.allow_null
    assert var_7 is False
    var_8 = var_6.coerce_types
    assert var_8 is False

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'boolean'
    var_2 = True
    var_3 = {}
    var_4 = module_0.from_json_schema_type(var_0, var_1, var_2, var_3)
    var_5 = var_4.allow_null
    assert var_5 is True

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'items'
    var_1 = 'minItems'
    var_2 = 'maxItems'
    var_3 = 'uniqueItems'
    var_4 = 'type'
    var_5 = 'integer'
    var_6 = {var_4: var_5}
    var_7 = 1
    var_8 = 10
    var_9 = True
    var_10 = {var_0: var_6, var_1: var_7, var_2: var_8, var_3: var_9}
    var_11 = 'array'
    var_12 = False
    var_13 = {}
    var_14 = module_0.from_json_schema_type(var_10, var_11, var_12, var_13)
    var_15 = var_14.allow_null
    assert var_15 is False
    var_16 = var_14.min_items
    assert var_16 == 1
    var_17 = var_14.max_items
    assert var_17 == 10
    var_18 = var_14.unique_items
    assert var_18 is True

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'additionalItems'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = 'array'
    var_4 = {}
    var_5 = module_0.from_json_schema_type(var_2, var_3, var_1, var_4)
    var_6 = var_5.additional_items
    assert var_6 is False

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'items'
    var_1 = 'type'
    var_2 = 'integer'
    var_3 = {var_1: var_2}
    var_4 = 'string'
    var_5 = {var_1: var_4}
    var_6 = [var_3, var_5]
    var_7 = {var_0: var_6}
    var_8 = 'array'
    var_9 = False
    var_10 = {}
    var_11 = module_0.from_json_schema_type(var_7, var_8, var_9, var_10)
    var_12 = var_11.items
    var_13 = var_11.items
    var_14 = len(var_13)
    assert var_14 == 2

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'properties'
    var_1 = 'required'
    var_2 = 'minProperties'
    var_3 = 'maxProperties'
    var_4 = 'name'
    var_5 = 'age'
    var_6 = 'type'
    var_7 = 'string'
    var_8 = {var_6: var_7}
    var_9 = 'integer'
    var_10 = {var_6: var_9}
    var_11 = {var_4: var_8, var_5: var_10}
    var_12 = [var_4]
    var_13 = 1
    var_14 = 10
    var_15 = {var_0: var_11, var_1: var_12, var_2: var_13, var_3: var_14}
    var_16 = 'object'
    var_17 = False
    var_18 = {}
    var_19 = module_0.from_json_schema_type(var_15, var_16, var_17, var_18)
    var_20 = var_19.allow_null
    assert var_20 is False
    var_21 = 'name'
    var_22 = bool('name' in var_19.properties)
    assert var_22 is True
    var_23 = 'age'
    var_24 = bool('age' in var_19.properties)
    assert var_24 is True
    var_25 = var_19.required
    var_26 = bool(var_19.required == ['name'])
    assert var_26 is True
    var_27 = var_19.min_properties
    assert var_27 == 1
    var_28 = var_19.max_properties
    assert var_28 == 10

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'patternProperties'
    var_1 = '^[a-z]+$'
    var_2 = 'type'
    var_3 = 'string'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'object'
    var_8 = False
    var_9 = {}
    var_10 = module_0.from_json_schema_type(var_6, var_7, var_8, var_9)
    var_11 = '^[a-z]+$'
    var_12 = bool('^[a-z]+$' in var_10.pattern_properties)
    assert var_12 is True

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'additionalProperties'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = 'object'
    var_4 = {}
    var_5 = module_0.from_json_schema_type(var_2, var_3, var_1, var_4)
    var_6 = var_5.additional_properties
    assert var_6 is False

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'propertyNames'
    var_1 = 'pattern'
    var_2 = '^[a-z]+$'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'object'
    var_6 = False
    var_7 = {}
    var_8 = module_0.from_json_schema_type(var_4, var_5, var_6, var_7)
    var_9 = var_8.property_names
    var_10 = bool(var_8.property_names is not None)
    assert var_10 is True



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_else_clause_is_none. Retrieved 1/13 statements.


def test_case_0():
    var_0 = None



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_to_json_schema_with_definitions. Retrieved 3/7 statements.
# Partially parsed test_to_json_schema_with_reference_field. Retrieved 2/5 statements.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True

import typesystem.composites as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.NeverMatch(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    assert var_2 is False

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'string'
    var_4 = 'default'
    var_5 = bool('default' not in var_2)
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['string', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['default']
    assert var_5 == 'hello'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minLength']
    assert var_4 == 5

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maxLength']
    assert var_4 == 10

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.String(allow_blank=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minLength']
    assert var_4 == 1

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'integer'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Integer(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['integer', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = {}
    var_2 = module_0.Integer(minimum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minimum']
    assert var_4 == 0

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Integer(maximum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maximum']
    assert var_4 == 100

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Float(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'number'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Decimal(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'number'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'boolean'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['boolean', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Array(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'array'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Array(min_items=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minItems']
    assert var_4 == 1

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Array(max_items=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maxItems']
    assert var_4 == 10

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Array(unique_items=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['uniqueItems']
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['items']['type']
    assert var_5 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Array(var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = 'items'
    var_9 = var_7[var_8]
    var_10 = len(var_9)
    assert var_10 == 2
    var_11 = var_7['items'][0]['type']
    assert var_11 == 'string'
    var_12 = var_7['items'][1]['type']
    assert var_12 == 'integer'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'object'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = 'properties'
    var_8 = bool('properties' in var_6)
    assert var_8 is True
    var_9 = var_6['properties']['name']['type']
    assert var_9 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_0]
    var_5 = {}
    var_6 = module_0.Object(properties=var_3, required=var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = var_7['required']
    var_9 = bool(var_7['required'] == ['name'])
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

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
    var_9 = module_1.to_json_schema(var_8)
    var_10 = var_9['enum']
    var_11 = bool(var_9['enum'] == ['a', 'b'])
    assert var_11 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'constant_value'
    var_1 = {}
    var_2 = module_0.Const(var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['const']
    assert var_4 == 'constant_value'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = 'anyOf'
    var_9 = bool('anyOf' in var_7)
    assert var_9 is True
    var_10 = 'anyOf'
    var_11 = var_7[var_10]
    var_12 = len(var_11)
    assert var_12 == 2

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = var_1 | var_3
    var_5 = module_1.to_json_schema(var_4)
    var_6 = 'anyOf'
    var_7 = bool('anyOf' in var_5)
    assert var_7 is True
    var_8 = 'anyOf'
    var_9 = var_5[var_8]
    var_10 = len(var_9)
    assert var_10 == 2

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_3]
    var_5 = 'components'
    var_6 = 'schemas'
    var_7 = 'name'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 'MyString'
    var_3 = 'target'
    var_4 = {var_3: var_1}



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_predicate_at_line_78_evaluates_to_false. Retrieved 1/14 statements.


def test_case_0():
    var_0 = None



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_if_then_else_from_json_schema_without_then_clause. Retrieved 8/12 statements.


def test_case_0():
    var_0 = []
    var_1 = 'if'
    var_2 = 'else'
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = 'number'
    var_7 = {var_3: var_6}
    var_8 = {var_1: var_5, var_2: var_7}



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_from_json_schema_with_ref. Retrieved 6/12 statements.


def test_case_0():
    var_0 = []
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = '$ref'
    var_5 = '#/components/schemas/MyType'
    var_6 = {var_4: var_5}



# Parsed testcases at query #61
#--------------------------

# Failed to parse test_to_json_schema_with_definitions.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True

import typesystem.composites as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.NeverMatch(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    assert var_2 is False

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['string', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minLength']
    assert var_4 == 5

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maxLength']
    assert var_4 == 10

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'integer'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Integer(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['integer', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = {}
    var_2 = module_0.Integer(minimum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minimum']
    assert var_4 == 0

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Integer(maximum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maximum']
    assert var_4 == 100

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Float(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'number'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Decimal(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'number'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'boolean'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['boolean', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    assert var_5 == 'array'
    var_6 = var_4['items']['type']
    assert var_6 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = True
    var_3 = 'allow_null'
    var_4 = {var_3: var_2}
    var_5 = module_0.Array(var_1, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = var_6['type']
    var_8 = bool(var_6['type'] == ['array', 'null'])
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 1
    var_3 = {}
    var_4 = module_0.Array(var_1, min_items=var_2, **var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = var_5['minItems']
    assert var_6 == 1

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 10
    var_3 = {}
    var_4 = module_0.Array(var_1, max_items=var_2, **var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = var_5['maxItems']
    assert var_6 == 10

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = var_6['type']
    assert var_7 == 'object'
    var_8 = 'properties'
    var_9 = bool('properties' in var_6)
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = True
    var_5 = 'allow_null'
    var_6 = {var_5: var_4}
    var_7 = module_0.Object(properties=var_3, **var_6)
    var_8 = module_1.to_json_schema(var_7)
    var_9 = var_8['type']
    var_10 = bool(var_8['type'] == ['object', 'null'])
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

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
    var_9 = module_1.to_json_schema(var_8)
    var_10 = var_9['enum']
    var_11 = bool(var_9['enum'] == ['a', 'b'])
    assert var_11 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'fixed_value'
    var_1 = {}
    var_2 = module_0.Const(var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['const']
    assert var_4 == 'fixed_value'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = 'anyOf'
    var_9 = bool('anyOf' in var_7)
    assert var_9 is True
    var_10 = 'anyOf'
    var_11 = var_7[var_10]
    var_12 = len(var_11)
    assert var_12 == 2

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'default_value'
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['default']
    assert var_5 == 'default_value'



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_if_then_else_from_json_schema_without_then_clause. Retrieved 8/12 statements.


def test_case_0():
    var_0 = []
    var_1 = 'if'
    var_2 = 'else'
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = 'number'
    var_7 = {var_3: var_6}
    var_8 = {var_1: var_5, var_2: var_7}



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_ref_from_json_schema_valid_reference. Retrieved 3/8 statements.
# Partially parsed test_ref_from_json_schema_with_multiple_path_segments. Retrieved 3/8 statements.
# Partially parsed test_ref_from_json_schema_invalid_ref_style. Retrieved 3/8 statements.
# Partially parsed test_ref_from_json_schema_with_external_url_ref. Retrieved 3/8 statements.
# Partially parsed test_ref_from_json_schema_preserves_definitions_reference. Retrieved 6/10 statements.


def test_case_0():
    var_0 = []
    var_1 = '$ref'
    var_2 = '#/definitions/User'
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = []
    var_1 = '$ref'
    var_2 = '#/components/schemas/Product'
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = []
    var_1 = '$ref'
    var_2 = 'definitions/User'
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Unsupported $ref style'

def test_case_0():
    var_0 = []
    var_1 = '$ref'
    var_2 = 'http://example.com/schema.json'
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Unsupported $ref style'

def test_case_0():
    var_0 = 'User'
    var_1 = 'mock_field'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = '$ref'
    var_5 = '#/definitions/User'
    var_6 = {var_4: var_5}



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_ref_from_json_schema_valid_reference_format. Retrieved 3/8 statements.


def test_case_0():
    var_0 = []
    var_1 = '$ref'
    var_2 = '#/components/schemas/MySchema'
    var_3 = {var_1: var_2}



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_ref_from_json_schema_unsupported_ref_style. Retrieved 3/8 statements.


def test_case_0():
    var_0 = []
    var_1 = '$ref'
    var_2 = 'http://example.com/schema'
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Unsupported $ref style in document.'



####################################################################
#    TEST GENERATION BEGINS (DEEPMOSA + claude-haiku-4-5 t=0.8)    #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_to_json_schema_with_definitions. Retrieved 3/7 statements.
# Partially parsed test_to_json_schema_with_reference_field. Retrieved 2/5 statements.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True

import typesystem.composites as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.NeverMatch(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    assert var_2 is False

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['string', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minLength']
    assert var_4 == 5

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maxLength']
    assert var_4 == 10

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.String(allow_blank=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minLength']
    assert var_4 == 1

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'integer'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Integer(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['integer', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = {}
    var_2 = module_0.Integer(minimum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minimum']
    assert var_4 == 0

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Integer(maximum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maximum']
    assert var_4 == 100

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Float(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'number'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Decimal(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'number'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'boolean'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['boolean', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Array(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'array'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Array(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['array', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Array(min_items=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minItems']
    assert var_4 == 2

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Array(max_items=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maxItems']
    assert var_4 == 10

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Array(unique_items=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['uniqueItems']
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'object'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Object(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['object', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = 'properties'
    var_8 = bool('properties' in var_6)
    assert var_8 is True
    var_9 = 'name'
    var_10 = bool('name' in var_6['properties'])
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_0]
    var_5 = {}
    var_6 = module_0.Object(properties=var_3, required=var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = var_7['required']
    var_9 = bool(var_7['required'] == ['name'])
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 1
    var_1 = 'One'
    var_2 = (var_0, var_1)
    var_3 = 2
    var_4 = 'Two'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = {}
    var_8 = module_0.Choice(choices=var_6, **var_7)
    var_9 = module_1.to_json_schema(var_8)
    var_10 = var_9['enum']
    var_11 = bool(var_9['enum'] == [1, 2])
    assert var_11 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'constant_value'
    var_1 = {}
    var_2 = module_0.Const(var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['const']
    assert var_4 == 'constant_value'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = 'anyOf'
    var_9 = bool('anyOf' in var_7)
    assert var_9 is True
    var_10 = 'anyOf'
    var_11 = var_7[var_10]
    var_12 = len(var_11)
    assert var_12 == 2

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'default_value'
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['default']
    assert var_5 == 'default_value'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'UserName'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_3]
    var_5 = 'components'
    var_6 = 'schemas'
    var_7 = 'UserName'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 'UserName'
    var_3 = 'target'
    var_4 = {var_3: var_1}
    var_5 = 'components'

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_1.OneOf(var_4, **var_5)
    var_7 = module_2.to_json_schema(var_6)
    var_8 = 'oneOf'
    var_9 = bool('oneOf' in var_7)
    assert var_9 is True
    var_10 = 'oneOf'
    var_11 = var_7[var_10]
    var_12 = len(var_11)
    assert var_12 == 2



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_to_json_schema_with_definitions. Retrieved 3/7 statements.
# Partially parsed test_to_json_schema_with_schema_field. Retrieved 1/5 statements.
# Failed to parse test_to_json_schema_with_unsupported_field_type.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True

import typesystem.composites as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.NeverMatch(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    assert var_2 is False

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['string', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minLength']
    assert var_4 == 5

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maxLength']
    assert var_4 == 10

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'integer'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = {}
    var_2 = module_0.Integer(minimum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minimum']
    assert var_4 == 0

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Integer(maximum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maximum']
    assert var_4 == 100

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Float(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'number'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Decimal(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'number'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'boolean'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['boolean', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    assert var_5 == 'array'
    var_6 = 'items'
    var_7 = bool('items' in var_4)
    assert var_7 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 1
    var_3 = {}
    var_4 = module_0.Array(var_1, min_items=var_2, **var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = var_5['minItems']
    assert var_6 == 1

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 10
    var_3 = {}
    var_4 = module_0.Array(var_1, max_items=var_2, **var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = var_5['maxItems']
    assert var_6 == 10

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = var_6['type']
    assert var_7 == 'object'
    var_8 = 'properties'
    var_9 = bool('properties' in var_6)
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

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
    var_9 = module_1.to_json_schema(var_8)
    var_10 = var_9['enum']
    var_11 = bool(var_9['enum'] == ['a', 'b'])
    assert var_11 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'fixed_value'
    var_1 = {}
    var_2 = module_0.Const(var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['const']
    assert var_4 == 'fixed_value'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = 'anyOf'
    var_9 = bool('anyOf' in var_7)
    assert var_9 is True
    var_10 = 'anyOf'
    var_11 = var_7[var_10]
    var_12 = len(var_11)
    assert var_12 == 2

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'test_default'
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['default']
    assert var_5 == 'test_default'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_3]
    var_5 = 'components'
    var_6 = 'schemas'
    var_7 = 'name'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 'properties'

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_1.OneOf(var_4, **var_5)
    var_7 = module_2.to_json_schema(var_6)
    var_8 = 'oneOf'
    var_9 = bool('oneOf' in var_7)
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_1.AllOf(var_2, **var_3)
    var_5 = module_2.to_json_schema(var_4)
    var_6 = 'allOf'
    var_7 = bool('allOf' in var_5)
    assert var_7 is True

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_1.Not(var_1, **var_2)
    var_4 = module_2.to_json_schema(var_3)
    var_5 = 'not'
    var_6 = bool('not' in var_4)
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = {}
    var_5 = module_1.IfThenElse(var_1, var_3, **var_4)
    var_6 = module_2.to_json_schema(var_5)
    var_7 = 'if'
    var_8 = bool('if' in var_6)
    assert var_8 is True
    var_9 = 'then'
    var_10 = bool('then' in var_6)
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = '^[a-z]+$'
    var_1 = {}
    var_2 = module_0.String(pattern=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = 'pattern'
    var_5 = bool('pattern' in var_3)
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'email'
    var_1 = {}
    var_2 = module_0.String(format=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['format']
    assert var_4 == 'email'



# Parsed testcases at query #3
#--------------------------




import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = 'field1'
    var_3 = [var_2]
    var_4 = 'allow_null'
    var_5 = 'required'
    var_6 = {var_4: var_0, var_5: var_3}
    var_7 = module_0.Schema(var_1, **var_6)
    var_8 = module_1.to_json_schema(var_7)
    var_9 = var_8['required']
    var_10 = bool(var_8['required'] == ['field1'])
    assert var_10 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_from_json_schema_type_number. Retrieved 15/21 statements.
# Partially parsed test_from_json_schema_type_integer. Retrieved 11/17 statements.
# Partially parsed test_from_json_schema_type_string. Retrieved 13/19 statements.
# Partially parsed test_from_json_schema_type_string_allow_blank. Retrieved 5/11 statements.
# Partially parsed test_from_json_schema_type_boolean. Retrieved 5/11 statements.
# Partially parsed test_from_json_schema_type_boolean_allow_null. Retrieved 3/9 statements.
# Partially parsed test_from_json_schema_type_array_simple. Retrieved 17/25 statements.
# Partially parsed test_from_json_schema_type_array_no_items. Retrieved 5/11 statements.
# Partially parsed test_from_json_schema_type_array_tuple_items. Retrieved 12/26 statements.
# Partially parsed test_from_json_schema_type_array_additional_items_field. Retrieved 9/17 statements.
# Partially parsed test_from_json_schema_type_object_simple. Retrieved 21/31 statements.
# Failed to parse test_from_json_schema_type_object_pattern_properties.


def test_case_0():
    var_0 = []
    var_1 = 'minimum'
    var_2 = 'maximum'
    var_3 = 'exclusiveMinimum'
    var_4 = 'exclusiveMaximum'
    var_5 = 'multipleOf'
    var_6 = 'default'
    var_7 = 0
    var_8 = 100
    var_9 = -1
    var_10 = 101
    var_11 = 5
    var_12 = 50
    var_13 = {var_1: var_7, var_2: var_8, var_3: var_9, var_4: var_10, var_5: var_11, var_6: var_12}
    var_14 = 'number'
    var_15 = False

def test_case_0():
    var_0 = []
    var_1 = 'minimum'
    var_2 = 'maximum'
    var_3 = 'multipleOf'
    var_4 = 'default'
    var_5 = 1
    var_6 = 10
    var_7 = 2
    var_8 = 4
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_7, var_4: var_8}
    var_10 = 'integer'
    var_11 = True

def test_case_0():
    var_0 = []
    var_1 = 'minLength'
    var_2 = 'maxLength'
    var_3 = 'pattern'
    var_4 = 'format'
    var_5 = 'default'
    var_6 = 5
    var_7 = 20
    var_8 = '^[a-z]+$'
    var_9 = 'email'
    var_10 = 'test'
    var_11 = {var_1: var_6, var_2: var_7, var_3: var_8, var_4: var_9, var_5: var_10}
    var_12 = 'string'
    var_13 = False

def test_case_0():
    var_0 = []
    var_1 = 'minLength'
    var_2 = 0
    var_3 = {var_1: var_2}
    var_4 = 'string'
    var_5 = False

def test_case_0():
    var_0 = []
    var_1 = 'default'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = 'boolean'
    var_5 = False

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = 'boolean'
    var_3 = True

def test_case_0():
    var_0 = []
    var_1 = 'items'
    var_2 = 'minItems'
    var_3 = 'maxItems'
    var_4 = 'uniqueItems'
    var_5 = 'default'
    var_6 = 'type'
    var_7 = 'string'
    var_8 = {var_6: var_7}
    var_9 = 1
    var_10 = 10
    var_11 = True
    var_12 = 'a'
    var_13 = 'b'
    var_14 = [var_12, var_13]
    var_15 = {var_1: var_8, var_2: var_9, var_3: var_10, var_4: var_11, var_5: var_14}
    var_16 = 'array'
    var_17 = False

def test_case_0():
    var_0 = []
    var_1 = 'minItems'
    var_2 = 0
    var_3 = {var_1: var_2}
    var_4 = 'array'
    var_5 = False

def test_case_0():
    var_0 = []
    var_1 = 'items'
    var_2 = 'additionalItems'
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = 'integer'
    var_7 = {var_3: var_6}
    var_8 = [var_5, var_7]
    var_9 = False
    var_10 = {var_1: var_8, var_2: var_9}
    var_11 = 'array'
    var_12 = 1

def test_case_0():
    var_0 = []
    var_1 = 'items'
    var_2 = 'additionalItems'
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = {var_3: var_4}
    var_7 = {var_1: var_5, var_2: var_6}
    var_8 = 'array'
    var_9 = False

def test_case_0():
    var_0 = []
    var_1 = 'properties'
    var_2 = 'required'
    var_3 = 'minProperties'
    var_4 = 'maxProperties'
    var_5 = 'default'
    var_6 = 'name'
    var_7 = 'age'
    var_8 = 'type'
    var_9 = 'string'
    var_10 = {var_8: var_9}
    var_11 = 'integer'
    var_12 = {var_8: var_11}
    var_13 = {var_6: var_10, var_7: var_12}
    var_14 = [var_6]
    var_15 = 1
    var_16 = 10
    var_17 = 'John'
    var_18 = {var_6: var_17}
    var_19 = {var_1: var_13, var_2: var_14, var_3: var_15, var_4: var_16, var_5: var_18}
    var_20 = 'object'
    var_21 = False
    var_22 = 'name'
    var_23 = 'age'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_to_json_schema_with_definitions. Retrieved 3/7 statements.
# Partially parsed test_to_json_schema_with_reference_field. Retrieved 2/5 statements.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True

import typesystem.composites as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.NeverMatch(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    assert var_2 is False

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['string', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minLength']
    assert var_4 == 5

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maxLength']
    assert var_4 == 10

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.String(allow_blank=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minLength']
    assert var_4 == 1

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'integer'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Integer(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['integer', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = {}
    var_2 = module_0.Integer(minimum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minimum']
    assert var_4 == 0

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Integer(maximum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maximum']
    assert var_4 == 100

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Float(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'number'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Decimal(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'number'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'boolean'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['boolean', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Array(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'array'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Array(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['array', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Array(min_items=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minItems']
    assert var_4 == 1

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Array(max_items=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maxItems']
    assert var_4 == 10

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['items']['type']
    assert var_5 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Array(unique_items=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['uniqueItems']
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'object'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Object(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['object', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = 'properties'
    var_8 = bool('properties' in var_6)
    assert var_8 is True
    var_9 = 'name'
    var_10 = bool('name' in var_6['properties'])
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Object(required=var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['required']
    var_6 = bool(var_4['required'] == ['name'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

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
    var_9 = module_1.to_json_schema(var_8)
    var_10 = var_9['enum']
    var_11 = bool(var_9['enum'] == ['a', 'b'])
    assert var_11 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'constant_value'
    var_1 = {}
    var_2 = module_0.Const(var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['const']
    assert var_4 == 'constant_value'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = 'anyOf'
    var_9 = bool('anyOf' in var_7)
    assert var_9 is True
    var_10 = 'anyOf'
    var_11 = var_7[var_10]
    var_12 = len(var_11)
    assert var_12 == 2

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'default_value'
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['default']
    assert var_5 == 'default_value'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_3]
    var_5 = 'components'
    var_6 = 'schemas'
    var_7 = 'name'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 'MySchema'
    var_3 = 'target'
    var_4 = {var_3: var_1}
    var_5 = 'components'

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_1.OneOf(var_4, **var_5)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_from_json_schema_with_bool_true. Retrieved 2/3 statements.
# Partially parsed test_from_json_schema_with_bool_false. Retrieved 2/3 statements.
# Partially parsed test_from_json_schema_with_empty_dict. Retrieved 2/3 statements.
# Partially parsed test_from_json_schema_with_type_string. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_with_type_number. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_with_type_integer. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_with_type_boolean. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_with_type_array. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_with_type_object. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_with_type_null. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_with_enum. Retrieved 7/8 statements.
# Partially parsed test_from_json_schema_with_const. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_with_any_of. Retrieved 9/10 statements.
# Partially parsed test_from_json_schema_with_one_of. Retrieved 9/10 statements.
# Partially parsed test_from_json_schema_with_all_of. Retrieved 10/11 statements.
# Partially parsed test_from_json_schema_with_not. Retrieved 6/7 statements.
# Partially parsed test_from_json_schema_with_if_then_else. Retrieved 13/14 statements.
# Partially parsed test_from_json_schema_with_if_only. Retrieved 6/7 statements.
# Partially parsed test_from_json_schema_with_type_and_enum. Retrieved 8/9 statements.
# Partially parsed test_from_json_schema_with_multiple_constraints. Retrieved 9/10 statements.
# Partially parsed test_from_json_schema_with_ref. Retrieved 3/7 statements.
# Partially parsed test_from_json_schema_with_components_schemas. Retrieved 10/11 statements.
# Partially parsed test_from_json_schema_with_type_and_null. Retrieved 6/7 statements.
# Partially parsed test_from_json_schema_with_string_properties. Retrieved 12/13 statements.
# Partially parsed test_from_json_schema_with_array_items. Retrieved 7/8 statements.
# Partially parsed test_from_json_schema_with_default_value. Retrieved 6/7 statements.
# Partially parsed test_from_json_schema_with_string_constraints. Retrieved 10/11 statements.
# Partially parsed test_from_json_schema_with_number_constraints. Retrieved 10/11 statements.
# Partially parsed test_from_json_schema_with_integer_constraints. Retrieved 10/11 statements.
# Partially parsed test_from_json_schema_with_array_constraints. Retrieved 10/11 statements.
# Partially parsed test_from_json_schema_with_object_constraints. Retrieved 11/12 statements.
# Partially parsed test_from_json_schema_with_object_pattern_properties. Retrieved 9/10 statements.
# Partially parsed test_from_json_schema_with_object_additional_properties_bool. Retrieved 6/7 statements.
# Partially parsed test_from_json_schema_with_object_additional_properties_schema. Retrieved 7/8 statements.
# Partially parsed test_from_json_schema_with_object_property_names. Retrieved 8/9 statements.
# Partially parsed test_from_json_schema_with_array_additional_items_bool. Retrieved 10/11 statements.


import typesystem.json_schema as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.from_json_schema(var_0)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.from_json_schema(var_0)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.from_json_schema(var_0)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'string'
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'number'
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'integer'
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'boolean'
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'array'
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'object'
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'null'
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'enum'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = module_0.from_json_schema(var_5)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'const'
    var_1 = 42
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'anyOf'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = 'number'
    var_5 = {var_1: var_4}
    var_6 = [var_3, var_5]
    var_7 = {var_0: var_6}
    var_8 = module_0.from_json_schema(var_7)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'oneOf'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = 'number'
    var_5 = {var_1: var_4}
    var_6 = [var_3, var_5]
    var_7 = {var_0: var_6}
    var_8 = module_0.from_json_schema(var_7)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'allOf'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = 'minLength'
    var_5 = 1
    var_6 = {var_4: var_5}
    var_7 = [var_3, var_6]
    var_8 = {var_0: var_7}
    var_9 = module_0.from_json_schema(var_8)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'not'
    var_1 = 'type'
    var_2 = 'null'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = module_0.from_json_schema(var_4)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'if'
    var_1 = 'then'
    var_2 = 'else'
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = 'minLength'
    var_7 = 1
    var_8 = {var_6: var_7}
    var_9 = 'number'
    var_10 = {var_3: var_9}
    var_11 = {var_0: var_5, var_1: var_8, var_2: var_10}
    var_12 = module_0.from_json_schema(var_11)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'if'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = module_0.from_json_schema(var_4)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'enum'
    var_2 = 'string'
    var_3 = 'a'
    var_4 = 'b'
    var_5 = [var_3, var_4]
    var_6 = {var_0: var_2, var_1: var_5}
    var_7 = module_0.from_json_schema(var_6)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'enum'
    var_2 = 'const'
    var_3 = 'string'
    var_4 = 'a'
    var_5 = 'b'
    var_6 = [var_4, var_5]
    var_7 = {var_0: var_3, var_1: var_6, var_2: var_4}
    var_8 = module_0.from_json_schema(var_7)

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = '$ref'
    var_3 = '#/components/schemas/Test'
    var_4 = {var_2: var_3}

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'components'
    var_1 = 'schemas'
    var_2 = 'Test'
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = module_0.from_json_schema(var_8)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'string'
    var_2 = 'null'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = module_0.from_json_schema(var_4)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'properties'
    var_2 = 'object'
    var_3 = 'name'
    var_4 = 'age'
    var_5 = 'string'
    var_6 = {var_0: var_5}
    var_7 = 'integer'
    var_8 = {var_0: var_7}
    var_9 = {var_3: var_6, var_4: var_8}
    var_10 = {var_0: var_2, var_1: var_9}
    var_11 = module_0.from_json_schema(var_10)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'items'
    var_2 = 'array'
    var_3 = 'string'
    var_4 = {var_0: var_3}
    var_5 = {var_0: var_2, var_1: var_4}
    var_6 = module_0.from_json_schema(var_5)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'default'
    var_2 = 'string'
    var_3 = 'test'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.from_json_schema(var_4)
    var_6 = var_5.default
    assert var_6 == 'test'

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'minLength'
    var_2 = 'maxLength'
    var_3 = 'pattern'
    var_4 = 'string'
    var_5 = 1
    var_6 = 10
    var_7 = '^[a-z]+$'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.from_json_schema(var_8)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'minimum'
    var_2 = 'maximum'
    var_3 = 'multipleOf'
    var_4 = 'number'
    var_5 = 0
    var_6 = 100
    var_7 = 5
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.from_json_schema(var_8)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'minimum'
    var_2 = 'maximum'
    var_3 = 'exclusiveMinimum'
    var_4 = 'exclusiveMaximum'
    var_5 = 'integer'
    var_6 = 0
    var_7 = 100
    var_8 = {var_0: var_5, var_1: var_6, var_2: var_7, var_3: var_6, var_4: var_7}
    var_9 = module_0.from_json_schema(var_8)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'minItems'
    var_2 = 'maxItems'
    var_3 = 'uniqueItems'
    var_4 = 'array'
    var_5 = 1
    var_6 = 10
    var_7 = True
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.from_json_schema(var_8)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'minProperties'
    var_2 = 'maxProperties'
    var_3 = 'required'
    var_4 = 'object'
    var_5 = 1
    var_6 = 10
    var_7 = 'name'
    var_8 = [var_7]
    var_9 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_8}
    var_10 = module_0.from_json_schema(var_9)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'patternProperties'
    var_2 = 'object'
    var_3 = '^S_'
    var_4 = 'string'
    var_5 = {var_0: var_4}
    var_6 = {var_3: var_5}
    var_7 = {var_0: var_2, var_1: var_6}
    var_8 = module_0.from_json_schema(var_7)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'additionalProperties'
    var_2 = 'object'
    var_3 = False
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.from_json_schema(var_4)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'additionalProperties'
    var_2 = 'object'
    var_3 = 'string'
    var_4 = {var_0: var_3}
    var_5 = {var_0: var_2, var_1: var_4}
    var_6 = module_0.from_json_schema(var_5)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'propertyNames'
    var_2 = 'object'
    var_3 = 'pattern'
    var_4 = '^[a-z]+$'
    var_5 = {var_3: var_4}
    var_6 = {var_0: var_2, var_1: var_5}
    var_7 = module_0.from_json_schema(var_6)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'items'
    var_2 = 'additionalItems'
    var_3 = 'array'
    var_4 = 'string'
    var_5 = {var_0: var_4}
    var_6 = [var_5]
    var_7 = False
    var_8 = {var_0: var_3, var_1: var_6, var_2: var_7}
    var_9 = module_0.from_json_schema(var_8)



# Parsed testcases at query #7
#--------------------------




import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Array(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_1.additional_items
    assert var_3 is None
    var_4 = bool(not var_1.additional_items is not None)
    assert var_4 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_to_json_schema_with_string_field. Retrieved 2/5 statements.
# Partially parsed test_to_json_schema_with_array_field. Retrieved 5/8 statements.
# Failed to parse test_to_json_schema_with_definitions.
# Partially parsed test_to_json_schema_with_reference_field. Retrieved 2/6 statements.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True

import typesystem.composites as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.NeverMatch(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    assert var_2 is False

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['string', 'null'])
    assert var_6 is True

import re as module_0
import typesystem.fields as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = '^[a-z]+$'
    var_1 = module_0.compile(var_0)
    var_2 = 'pattern_regex'
    var_3 = {var_2: var_1}
    var_4 = module_1.String(**var_3)
    var_5 = module_2.to_json_schema(var_4)
    var_6 = var_5['pattern']
    assert var_6 == '^[a-z]+$'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'integer'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = 100
    var_2 = {}
    var_3 = module_0.Integer(minimum=var_0, maximum=var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['minimum']
    assert var_5 == 0
    var_6 = var_4['maximum']
    assert var_6 == 100

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Float(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'number'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Decimal(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'number'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'boolean'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    assert var_5 == 'array'
    var_6 = 'items'
    var_7 = var_4[var_6]

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 1
    var_3 = 10
    var_4 = {}
    var_5 = module_0.Array(var_1, min_items=var_2, max_items=var_3, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = var_6['minItems']
    assert var_7 == 1
    var_8 = var_6['maxItems']
    assert var_8 == 10

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'object'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = 'properties'
    var_8 = bool('properties' in var_6)
    assert var_8 is True
    var_9 = 'name'
    var_10 = bool('name' in var_6['properties'])
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

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
    var_9 = module_1.to_json_schema(var_8)
    var_10 = var_9['enum']
    var_11 = bool(var_9['enum'] == ['a', 'b'])
    assert var_11 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'fixed_value'
    var_1 = {}
    var_2 = module_0.Const(var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['const']
    assert var_4 == 'fixed_value'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = 'anyOf'
    var_9 = bool('anyOf' in var_7)
    assert var_9 is True
    var_10 = 'anyOf'
    var_11 = var_7[var_10]
    var_12 = len(var_11)
    assert var_12 == 2

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'default_value'
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['default']
    assert var_5 == 'default_value'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 'StringDef'
    var_3 = 'target'
    var_4 = {var_3: var_1}
    var_5 = 'components'

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_1.OneOf(var_4, **var_5)
    var_7 = module_2.to_json_schema(var_6)
    var_8 = 'oneOf'
    var_9 = bool('oneOf' in var_7)
    assert var_9 is True
    var_10 = 'oneOf'
    var_11 = var_7[var_10]
    var_12 = len(var_11)
    assert var_12 == 2

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_1.AllOf(var_2, **var_3)
    var_5 = module_2.to_json_schema(var_4)
    var_6 = 'allOf'
    var_7 = bool('allOf' in var_5)
    assert var_7 is True
    var_8 = 'allOf'
    var_9 = var_5[var_8]
    var_10 = len(var_9)
    assert var_10 == 1

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_1.Not(var_1, **var_2)
    var_4 = module_2.to_json_schema(var_3)
    var_5 = 'not'
    var_6 = bool('not' in var_4)
    assert var_6 is True



# Parsed testcases at query #9
#--------------------------

# Failed to parse test_to_json_schema_with_definitions_argument.




# Parsed testcases at query #10
#--------------------------

# Partially parsed test_predicate_at_line_71_evaluates_to_true. Retrieved 5/8 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Array(var_4, **var_5)
    var_7 = var_6.items



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_to_json_schema_with_string_field. Retrieved 2/5 statements.
# Partially parsed test_to_json_schema_with_definitions. Retrieved 3/8 statements.
# Partially parsed test_to_json_schema_with_reference_field. Retrieved 2/6 statements.
# Partially parsed test_to_json_schema_with_array_multiple_items. Retrieved 9/12 statements.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True

import typesystem.composites as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.NeverMatch(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    assert var_2 is False

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['string', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = 10
    var_2 = {}
    var_3 = module_0.String(max_length=var_1, min_length=var_0, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['minLength']
    assert var_5 == 5
    var_6 = var_4['maxLength']
    assert var_6 == 10

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'integer'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = 100
    var_2 = {}
    var_3 = module_0.Integer(minimum=var_0, maximum=var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['minimum']
    assert var_5 == 0
    var_6 = var_4['maximum']
    assert var_6 == 100

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'boolean'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    assert var_5 == 'array'
    var_6 = 'items'
    var_7 = bool('items' in var_4)
    assert var_7 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 1
    var_3 = 10
    var_4 = {}
    var_5 = module_0.Array(var_1, min_items=var_2, max_items=var_3, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = var_6['minItems']
    assert var_7 == 1
    var_8 = var_6['maxItems']
    assert var_8 == 10

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = var_6['type']
    assert var_7 == 'object'
    var_8 = 'properties'
    var_9 = bool('properties' in var_6)
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

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
    var_9 = module_1.to_json_schema(var_8)
    var_10 = var_9['enum']
    var_11 = bool(var_9['enum'] == ['a', 'b'])
    assert var_11 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = var_1 | var_3
    var_5 = module_1.to_json_schema(var_4)
    var_6 = 'anyOf'
    var_7 = bool('anyOf' in var_5)
    assert var_7 is True
    var_8 = 'anyOf'
    var_9 = var_5[var_8]
    var_10 = len(var_9)
    assert var_10 == 2

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['default']
    assert var_5 == 'hello'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_3]
    var_5 = 'components'
    var_6 = 'schemas'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 'MySchema'
    var_3 = 'target'
    var_4 = {var_3: var_1}
    var_5 = 'components'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Float(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'number'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Decimal(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'number'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'fixed_value'
    var_1 = {}
    var_2 = module_0.Const(var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['const']
    assert var_4 == 'fixed_value'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = {}
    var_2 = module_0.String(pattern=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = 'pattern'
    var_5 = bool('pattern' in var_3)
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'email'
    var_1 = {}
    var_2 = module_0.String(format=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['format']
    assert var_4 == 'email'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_0]
    var_5 = {}
    var_6 = module_0.Object(properties=var_3, required=var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = var_7['required']
    var_9 = bool(var_7['required'] == ['name'])
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Array(var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = 'items'
    var_9 = var_7[var_8]
    var_10 = var_7[var_8]
    var_11 = len(var_10)
    assert var_11 == 2

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Object(additional_properties=var_0, **var_1)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_predicate_at_line_79_evaluates_to_false. Retrieved 5/7 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'string'
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = module_0.Array(additional_items=var_2, **var_3)
    var_5 = var_4.additional_items



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_to_json_schema_with_definitions. Retrieved 3/7 statements.
# Partially parsed test_to_json_schema_with_reference_field. Retrieved 2/5 statements.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True

import typesystem.composites as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.NeverMatch(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    assert var_2 is False

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['string', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['default']
    assert var_5 == 'test'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minLength']
    assert var_4 == 5

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maxLength']
    assert var_4 == 10

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.String(allow_blank=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minLength']
    assert var_4 == 1

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'integer'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Integer(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['integer', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = {}
    var_2 = module_0.Integer(minimum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minimum']
    assert var_4 == 0

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Integer(maximum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maximum']
    assert var_4 == 100

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Float(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'number'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'boolean'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['boolean', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    assert var_5 == 'array'
    var_6 = var_4['items']['type']
    assert var_6 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = True
    var_3 = 'allow_null'
    var_4 = {var_3: var_2}
    var_5 = module_0.Array(var_1, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = var_6['type']
    var_8 = bool(var_6['type'] == ['array', 'null'])
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 1
    var_3 = {}
    var_4 = module_0.Array(var_1, min_items=var_2, **var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = var_5['minItems']
    assert var_6 == 1

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 10
    var_3 = {}
    var_4 = module_0.Array(var_1, max_items=var_2, **var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = var_5['maxItems']
    assert var_6 == 10

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = True
    var_3 = {}
    var_4 = module_0.Array(var_1, unique_items=var_2, **var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = var_5['uniqueItems']
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = var_6['type']
    assert var_7 == 'object'
    var_8 = 'name'
    var_9 = bool('name' in var_6['properties'])
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = True
    var_5 = 'allow_null'
    var_6 = {var_5: var_4}
    var_7 = module_0.Object(properties=var_3, **var_6)
    var_8 = module_1.to_json_schema(var_7)
    var_9 = var_8['type']
    var_10 = bool(var_8['type'] == ['object', 'null'])
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_0]
    var_5 = {}
    var_6 = module_0.Object(properties=var_3, required=var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = var_7['required']
    var_9 = bool(var_7['required'] == ['name'])
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = 'anyOf'
    var_9 = bool('anyOf' in var_7)
    assert var_9 is True
    var_10 = 'anyOf'
    var_11 = var_7[var_10]
    var_12 = len(var_11)
    assert var_12 == 2

import typesystem.fields as module_0
import typesystem.json_schema as module_1

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
    var_9 = module_1.to_json_schema(var_8)
    var_10 = var_9['enum']
    var_11 = bool(var_9['enum'] == ['a', 'b'])
    assert var_11 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'fixed_value'
    var_1 = {}
    var_2 = module_0.Const(var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['const']
    assert var_4 == 'fixed_value'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'StringField'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_3]
    var_5 = 'components'
    var_6 = 'schemas'
    var_7 = 'StringField'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

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
    var_10 = module_1.to_json_schema(var_9)
    var_11 = var_10['type']
    assert var_11 == 'object'
    var_12 = 'outer'
    var_13 = bool('outer' in var_10['properties'])
    assert var_13 is True
    var_14 = var_10['properties']['outer']['type']
    assert var_14 == 'object'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 'StringTarget'
    var_3 = 'target'
    var_4 = {var_3: var_1}
    var_5 = 'components'
    var_6 = 'StringTarget'

def test_case_0():
    pass



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_predicate_at_line_112_evaluates_to_true. Retrieved 4/17 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = var_1.max_properties
    var_3 = None
    var_4 = var_2 is not var_3
    assert var_4 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_predicate_at_line_158_evaluates_to_false. Retrieved 1/31 statements.


def test_case_0():
    var_0 = None



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_exclusive_maximum_predicate_evaluates_to_true. Retrieved 1/13 statements.


def test_case_0():
    var_0 = None



# Parsed testcases at query #17
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.Integer(multiple_of=var_0, **var_1)
    var_3 = var_2.multiple_of
    var_4 = bool(var_2.multiple_of is not None)
    assert var_4 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_predicate_at_line_108_evaluates_to_true. Retrieved 1/6 statements.


def test_case_0():
    var_0 = None



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_to_json_schema_with_string_field. Retrieved 2/4 statements.
# Partially parsed test_to_json_schema_with_array_field_tuple_items. Retrieved 9/11 statements.
# Partially parsed test_to_json_schema_with_object_field_additional_properties_field. Retrieved 8/10 statements.
# Partially parsed test_to_json_schema_with_schema_field. Retrieved 1/5 statements.
# Failed to parse test_to_json_schema_with_definitions.
# Partially parsed test_to_json_schema_with_reference_field. Retrieved 2/5 statements.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True

import typesystem.composites as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.NeverMatch(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    assert var_2 is False

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['string', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 2
    var_1 = 10
    var_2 = {}
    var_3 = module_0.String(max_length=var_1, min_length=var_0, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['minLength']
    assert var_5 == 2
    var_6 = var_4['maxLength']
    assert var_6 == 10

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.String(allow_blank=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minLength']
    assert var_4 == 1

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'integer'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Integer(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['integer', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = 100
    var_2 = {}
    var_3 = module_0.Integer(minimum=var_0, maximum=var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['minimum']
    assert var_5 == 0
    var_6 = var_4['maximum']
    assert var_6 == 100

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Float(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'number'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Decimal(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'number'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'boolean'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['boolean', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    assert var_5 == 'array'
    var_6 = 'items'
    var_7 = bool('items' in var_4)
    assert var_7 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 1
    var_3 = 5
    var_4 = {}
    var_5 = module_0.Array(var_1, min_items=var_2, max_items=var_3, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = var_6['minItems']
    assert var_7 == 1
    var_8 = var_6['maxItems']
    assert var_8 == 5
    var_9 = var_6['uniqueItems']
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Array(var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = 'items'
    var_9 = var_7[var_8]
    var_10 = var_7[var_8]
    var_11 = len(var_10)
    assert var_11 == 2

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = var_6['type']
    assert var_7 == 'object'
    var_8 = 'properties'
    var_9 = bool('properties' in var_6)
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_0]
    var_5 = {}
    var_6 = module_0.Object(properties=var_3, required=var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = var_7['required']
    var_9 = bool(var_7['required'] == ['name'])
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = False
    var_5 = {}
    var_6 = module_0.Object(properties=var_3, additional_properties=var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = var_7['additionalProperties']
    assert var_8 is False

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.String(**var_4)
    var_6 = {}
    var_7 = module_0.Object(properties=var_3, additional_properties=var_5, **var_6)
    var_8 = module_1.to_json_schema(var_7)
    var_9 = 'additionalProperties'
    var_10 = var_8[var_9]

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 'properties'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

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
    var_9 = module_1.to_json_schema(var_8)
    var_10 = var_9['enum']
    var_11 = bool(var_9['enum'] == ['a', 'b'])
    assert var_11 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'constant_value'
    var_1 = {}
    var_2 = module_0.Const(var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['const']
    assert var_4 == 'constant_value'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = var_1 | var_3
    var_5 = module_1.to_json_schema(var_4)
    var_6 = 'anyOf'
    var_7 = bool('anyOf' in var_5)
    assert var_7 is True
    var_8 = 'anyOf'
    var_9 = var_5[var_8]
    var_10 = len(var_9)
    assert var_10 == 2

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'default_value'
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['default']
    assert var_5 == 'default_value'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 'MyString'
    var_3 = 'target'
    var_4 = {var_3: var_1}
    var_5 = 'components'

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_1.OneOf(var_4, **var_5)
    var_7 = module_2.to_json_schema(var_6)
    var_8 = 'oneOf'
    var_9 = bool('oneOf' in var_7)
    assert var_9 is True
    var_10 = 'oneOf'
    var_11 = var_7[var_10]
    var_12 = len(var_11)
    assert var_12 == 2

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_1.AllOf(var_2, **var_3)
    var_5 = module_2.to_json_schema(var_4)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_exclusive_maximum_predicate. Retrieved 8/18 statements.


def test_case_0():
    var_0 = 'exclusive_maximum'
    var_1 = 'allow_null'
    var_2 = 'minimum'
    var_3 = 'maximum'
    var_4 = 'exclusive_minimum'
    var_5 = 'multiple_of'
    var_6 = [var_0, var_1, var_2, var_3, var_4, var_5]
    var_7 = None



# Parsed testcases at query #21
#--------------------------






# Parsed testcases at query #22
#--------------------------

# Partially parsed test_to_json_schema_with_definitions. Retrieved 3/7 statements.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True

import typesystem.composites as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.NeverMatch(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    assert var_2 is False

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['string', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minLength']
    assert var_4 == 5

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maxLength']
    assert var_4 == 10

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'integer'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Integer(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['integer', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Float(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'number'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'boolean'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['boolean', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    assert var_5 == 'array'
    var_6 = var_4['items']['type']
    assert var_6 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 2
    var_3 = {}
    var_4 = module_0.Array(var_1, min_items=var_2, **var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = var_5['minItems']
    assert var_6 == 2

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 5
    var_3 = {}
    var_4 = module_0.Array(var_1, max_items=var_2, **var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = var_5['maxItems']
    assert var_6 == 5

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = var_6['type']
    assert var_7 == 'object'
    var_8 = 'properties'
    var_9 = bool('properties' in var_6)
    assert var_9 is True
    var_10 = var_6['properties']['name']['type']
    assert var_10 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = var_1 | var_3
    var_5 = module_1.to_json_schema(var_4)
    var_6 = 'anyOf'
    var_7 = bool('anyOf' in var_5)
    assert var_7 is True
    var_8 = 'anyOf'
    var_9 = var_5[var_8]
    var_10 = len(var_9)
    assert var_10 == 2

import typesystem.fields as module_0
import typesystem.json_schema as module_1

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
    var_9 = module_1.to_json_schema(var_8)
    var_10 = var_9['enum']
    var_11 = bool(var_9['enum'] == ['a', 'b'])
    assert var_11 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['default']
    assert var_5 == 'test'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_3]
    var_5 = 'components'
    var_6 = 'schemas'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = {}
    var_2 = module_0.Integer(minimum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minimum']
    assert var_4 == 0

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Integer(maximum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maximum']
    assert var_4 == 100

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = '^[a-z]+$'
    var_1 = {}
    var_2 = module_0.String(pattern=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = 'pattern'
    var_5 = bool('pattern' in var_3)
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'age'
    var_1 = {}
    var_2 = module_0.Integer(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = 'name'
    var_7 = 'person'
    var_8 = {}
    var_9 = module_0.String(**var_8)
    var_10 = {var_6: var_9, var_7: var_5}
    var_11 = {}
    var_12 = module_0.Object(properties=var_10, **var_11)
    var_13 = module_1.to_json_schema(var_12)
    var_14 = var_13['properties']['person']['type']
    assert var_14 == 'object'
    var_15 = var_13['properties']['person']['properties']['age']['type']
    assert var_15 == 'integer'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_from_json_schema_with_bool_true. Retrieved 2/3 statements.
# Partially parsed test_from_json_schema_with_bool_false. Retrieved 2/3 statements.
# Partially parsed test_from_json_schema_with_type_constraint_only. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_with_enum_only. Retrieved 7/8 statements.
# Partially parsed test_from_json_schema_with_const_only. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_with_allOf_only. Retrieved 10/11 statements.
# Partially parsed test_from_json_schema_with_anyOf_only. Retrieved 9/10 statements.
# Partially parsed test_from_json_schema_with_oneOf_only. Retrieved 9/10 statements.
# Partially parsed test_from_json_schema_with_not_only. Retrieved 6/7 statements.
# Partially parsed test_from_json_schema_with_if_then_else. Retrieved 13/14 statements.
# Partially parsed test_from_json_schema_with_ref. Retrieved 3/7 statements.
# Partially parsed test_from_json_schema_with_multiple_constraints. Retrieved 8/9 statements.
# Partially parsed test_from_json_schema_with_no_constraints. Retrieved 2/3 statements.
# Partially parsed test_from_json_schema_with_components_schemas. Retrieved 17/18 statements.
# Partially parsed test_from_json_schema_string_with_format. Retrieved 6/7 statements.
# Partially parsed test_from_json_schema_integer_with_constraints. Retrieved 8/9 statements.
# Partially parsed test_from_json_schema_array_with_items. Retrieved 7/8 statements.
# Partially parsed test_from_json_schema_object_with_properties. Retrieved 12/13 statements.
# Partially parsed test_from_json_schema_number_type. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_boolean_type. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_with_if_only. Retrieved 6/7 statements.
# Partially parsed test_from_json_schema_null_type_only. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_multiple_types. Retrieved 6/7 statements.


import typesystem.json_schema as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.from_json_schema(var_0)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.from_json_schema(var_0)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'string'
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'enum'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = module_0.from_json_schema(var_5)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'const'
    var_1 = 42
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)
    var_4 = var_3.const
    assert var_4 == 42

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'allOf'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = 'minLength'
    var_5 = 1
    var_6 = {var_4: var_5}
    var_7 = [var_3, var_6]
    var_8 = {var_0: var_7}
    var_9 = module_0.from_json_schema(var_8)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'anyOf'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = 'number'
    var_5 = {var_1: var_4}
    var_6 = [var_3, var_5]
    var_7 = {var_0: var_6}
    var_8 = module_0.from_json_schema(var_7)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'oneOf'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = 'number'
    var_5 = {var_1: var_4}
    var_6 = [var_3, var_5]
    var_7 = {var_0: var_6}
    var_8 = module_0.from_json_schema(var_7)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'not'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = module_0.from_json_schema(var_4)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'if'
    var_1 = 'then'
    var_2 = 'else'
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = 'minLength'
    var_7 = 1
    var_8 = {var_6: var_7}
    var_9 = 'number'
    var_10 = {var_3: var_9}
    var_11 = {var_0: var_5, var_1: var_8, var_2: var_10}
    var_12 = module_0.from_json_schema(var_11)

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = '$ref'
    var_3 = '#/components/schemas/MyType'
    var_4 = {var_2: var_3}

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'enum'
    var_2 = 'string'
    var_3 = 'a'
    var_4 = 'b'
    var_5 = [var_3, var_4]
    var_6 = {var_0: var_2, var_1: var_5}
    var_7 = module_0.from_json_schema(var_6)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.from_json_schema(var_0)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'properties'
    var_2 = 'components'
    var_3 = 'object'
    var_4 = 'name'
    var_5 = '$ref'
    var_6 = '#/components/schemas/Name'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = 'schemas'
    var_10 = 'Name'
    var_11 = 'string'
    var_12 = {var_0: var_11}
    var_13 = {var_10: var_12}
    var_14 = {var_9: var_13}
    var_15 = {var_0: var_3, var_1: var_8, var_2: var_14}
    var_16 = module_0.from_json_schema(var_15)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'format'
    var_2 = 'string'
    var_3 = 'email'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.from_json_schema(var_4)
    var_6 = var_5.format
    assert var_6 == 'email'

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'minimum'
    var_2 = 'maximum'
    var_3 = 'integer'
    var_4 = 0
    var_5 = 100
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = module_0.from_json_schema(var_6)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'items'
    var_2 = 'array'
    var_3 = 'string'
    var_4 = {var_0: var_3}
    var_5 = {var_0: var_2, var_1: var_4}
    var_6 = module_0.from_json_schema(var_5)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'properties'
    var_2 = 'object'
    var_3 = 'name'
    var_4 = 'age'
    var_5 = 'string'
    var_6 = {var_0: var_5}
    var_7 = 'integer'
    var_8 = {var_0: var_7}
    var_9 = {var_3: var_6, var_4: var_8}
    var_10 = {var_0: var_2, var_1: var_9}
    var_11 = module_0.from_json_schema(var_10)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'default'
    var_2 = 'string'
    var_3 = 'hello'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.from_json_schema(var_4)
    var_6 = var_5.default
    assert var_6 == 'hello'

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'string'
    var_2 = 'null'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = module_0.from_json_schema(var_4)
    var_6 = var_5.allow_null
    assert var_6 is True

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'number'
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'boolean'
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'if'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = module_0.from_json_schema(var_4)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'null'
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)
    var_4 = var_3.const
    assert var_4 is None

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'string'
    var_2 = 'number'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = module_0.from_json_schema(var_4)



# Parsed testcases at query #24
#--------------------------

# Failed to parse test_schema_with_fields_predicate_at_line_122.




# Parsed testcases at query #25
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = {}
    var_3 = {}
    var_4 = var_1.max_properties
    var_5 = None
    var_6 = var_4 is not var_5
    assert var_6 is True



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_predicate_at_line_56_evaluates_to_true. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 'Integer'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]
    var_4 = None



# Parsed testcases at query #27
#--------------------------




import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True

import typesystem.composites as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.NeverMatch(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    assert var_2 is False

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['string', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['default']
    assert var_5 == 'hello'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minLength']
    assert var_4 == 5

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maxLength']
    assert var_4 == 10

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.String(allow_blank=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minLength']
    assert var_4 == 1

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'integer'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Integer(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['integer', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = {}
    var_2 = module_0.Integer(minimum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minimum']
    assert var_4 == 0

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Integer(maximum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maximum']
    assert var_4 == 100

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Float(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'number'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Decimal(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'number'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'boolean'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['boolean', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    assert var_5 == 'array'
    var_6 = var_4['items']['type']
    assert var_6 == 'integer'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = 1
    var_3 = {}
    var_4 = module_0.Array(var_1, min_items=var_2, **var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = var_5['minItems']
    assert var_6 == 1

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = 10
    var_3 = {}
    var_4 = module_0.Array(var_1, max_items=var_2, **var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = var_5['maxItems']
    assert var_6 == 10

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = True
    var_3 = {}
    var_4 = module_0.Array(var_1, unique_items=var_2, **var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = var_5['uniqueItems']
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = var_6['type']
    assert var_7 == 'object'
    var_8 = 'name'
    var_9 = bool('name' in var_6['properties'])
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_0]
    var_5 = {}
    var_6 = module_0.Object(properties=var_3, required=var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = var_7['required']
    var_9 = bool(var_7['required'] == ['name'])
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

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
    var_9 = module_1.to_json_schema(var_8)
    var_10 = var_9['enum']
    var_11 = bool(var_9['enum'] == ['a', 'b'])
    assert var_11 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'constant_value'
    var_1 = {}
    var_2 = module_0.Const(var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['const']
    assert var_4 == 'constant_value'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = 'anyOf'
    var_9 = bool('anyOf' in var_7)
    assert var_9 is True
    var_10 = 'anyOf'
    var_11 = var_7[var_10]
    var_12 = len(var_11)
    assert var_12 == 2

def test_case_0():
    pass



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_exclusive_minimum_predicate. Retrieved 1/11 statements.


def test_case_0():
    var_0 = None



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_exclusive_minimum_predicate. Retrieved 1/11 statements.


def test_case_0():
    var_0 = None



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_to_json_schema_with_definitions. Retrieved 1/6 statements.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True

import typesystem.composites as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.NeverMatch(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    assert var_2 is False

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['string', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['default']
    assert var_5 == 'test'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minLength']
    assert var_4 == 5

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maxLength']
    assert var_4 == 10

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.String(allow_blank=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minLength']
    assert var_4 == 1

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'integer'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Integer(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['integer', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Float(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'number'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Decimal(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'number'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'boolean'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['boolean', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Array(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'array'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    assert var_5 == 'array'
    var_6 = var_4['items']['type']
    assert var_6 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Array(min_items=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minItems']
    assert var_4 == 2

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.Array(max_items=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maxItems']
    assert var_4 == 5

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'object'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = var_6['type']
    assert var_7 == 'object'
    var_8 = 'name'
    var_9 = bool('name' in var_6['properties'])
    assert var_9 is True
    var_10 = var_6['properties']['name']['type']
    assert var_10 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 1
    var_1 = 'One'
    var_2 = (var_0, var_1)
    var_3 = 2
    var_4 = 'Two'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = {}
    var_8 = module_0.Choice(choices=var_6, **var_7)
    var_9 = module_1.to_json_schema(var_8)
    var_10 = var_9['enum']
    var_11 = bool(var_9['enum'] == [1, 2])
    assert var_11 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'fixed_value'
    var_1 = {}
    var_2 = module_0.Const(var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['const']
    assert var_4 == 'fixed_value'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = var_1 | var_3
    var_5 = module_1.to_json_schema(var_4)
    var_6 = 'anyOf'
    var_7 = bool('anyOf' in var_5)
    assert var_7 is True
    var_8 = 'anyOf'
    var_9 = var_5[var_8]
    var_10 = len(var_9)
    assert var_10 == 2

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = []
    var_3 = 'components'
    var_4 = 'schemas'
    var_5 = 'name'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)



# Parsed testcases at query #31
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.Integer(exclusive_minimum=var_0, **var_1)
    var_3 = var_2.exclusive_minimum
    var_4 = None
    var_5 = var_3 is not var_4
    assert var_5 is True



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_multiple_of_predicate_evaluates_to_true. Retrieved 1/6 statements.


def test_case_0():
    var_0 = None



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_predicate_at_line_79_evaluates_to_false. Retrieved 2/13 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Array(**var_0)
    var_2 = var_1.additional_items



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_predicate_at_line_114_evaluates_to_true. Retrieved 1/6 statements.


def test_case_0():
    var_0 = None



# Parsed testcases at query #35
#--------------------------




import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True

import typesystem.composites as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.NeverMatch(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    assert var_2 is False

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['string', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minLength']
    assert var_4 == 5

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maxLength']
    assert var_4 == 10

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.String(allow_blank=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minLength']
    assert var_4 == 1

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'integer'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Integer(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['integer', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = {}
    var_2 = module_0.Integer(minimum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minimum']
    assert var_4 == 0

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Integer(maximum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maximum']
    assert var_4 == 100

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Float(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'number'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Decimal(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'number'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'boolean'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['boolean', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Array(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'array'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Array(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['array', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Array(min_items=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minItems']
    assert var_4 == 1

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Array(max_items=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maxItems']
    assert var_4 == 10

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['items']['type']
    assert var_5 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Array(var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = 'items'
    var_9 = var_7[var_8]
    var_10 = len(var_9)
    assert var_10 == 2
    var_11 = var_7['items'][0]['type']
    assert var_11 == 'string'
    var_12 = var_7['items'][1]['type']
    assert var_12 == 'integer'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Array(unique_items=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['uniqueItems']
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'object'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Object(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['object', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = 'properties'
    var_8 = bool('properties' in var_6)
    assert var_8 is True
    var_9 = 'name'
    var_10 = bool('name' in var_6['properties'])
    assert var_10 is True
    var_11 = var_6['properties']['name']['type']
    assert var_11 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Object(required=var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['required']
    var_6 = bool(var_4['required'] == ['name'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Object(max_properties=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maxProperties']
    assert var_4 == 10

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Object(min_properties=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minProperties']
    assert var_4 == 1

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 1
    var_1 = 'One'
    var_2 = (var_0, var_1)
    var_3 = 2
    var_4 = 'Two'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = {}
    var_8 = module_0.Choice(choices=var_6, **var_7)
    var_9 = module_1.to_json_schema(var_8)
    var_10 = var_9['enum']
    var_11 = bool(var_9['enum'] == [1, 2])
    assert var_11 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'value'
    var_1 = {}
    var_2 = module_0.Const(var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['const']
    assert var_4 == 'value'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = 'anyOf'
    var_9 = bool('anyOf' in var_7)
    assert var_9 is True
    var_10 = 'anyOf'
    var_11 = var_7[var_10]
    var_12 = len(var_11)
    assert var_12 == 2



# Parsed testcases at query #36
#--------------------------




import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True

import typesystem.composites as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.NeverMatch(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    assert var_2 is False

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['string', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['default']
    assert var_5 == 'hello'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minLength']
    assert var_4 == 5

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maxLength']
    assert var_4 == 10

import re as module_0
import typesystem.fields as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = '^[a-z]+$'
    var_1 = module_0.compile(var_0)
    var_2 = {}
    var_3 = module_1.String(pattern=var_1, **var_2)
    var_4 = module_2.to_json_schema(var_3)
    var_5 = var_4['pattern']
    assert var_5 == '^[a-z]+$'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'email'
    var_1 = {}
    var_2 = module_0.String(format=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['format']
    assert var_4 == 'email'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'integer'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Integer(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['integer', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = {}
    var_2 = module_0.Integer(minimum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minimum']
    assert var_4 == 0

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Integer(maximum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maximum']
    assert var_4 == 100

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Float(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'number'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Decimal(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'number'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'boolean'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['boolean', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    assert var_5 == 'array'
    var_6 = var_4['items']['type']
    assert var_6 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 1
    var_3 = {}
    var_4 = module_0.Array(var_1, min_items=var_2, **var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = var_5['minItems']
    assert var_6 == 1

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 10
    var_3 = {}
    var_4 = module_0.Array(var_1, max_items=var_2, **var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = var_5['maxItems']
    assert var_6 == 10

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = True
    var_3 = {}
    var_4 = module_0.Array(var_1, unique_items=var_2, **var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = var_5['uniqueItems']
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = var_6['type']
    assert var_7 == 'object'
    var_8 = 'properties'
    var_9 = bool('properties' in var_6)
    assert var_9 is True
    var_10 = 'name'
    var_11 = bool('name' in var_6['properties'])
    assert var_11 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = True
    var_5 = 'allow_null'
    var_6 = {var_5: var_4}
    var_7 = module_0.Object(properties=var_3, **var_6)
    var_8 = module_1.to_json_schema(var_7)
    var_9 = var_8['type']
    var_10 = bool(var_8['type'] == ['object', 'null'])
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_0]
    var_5 = {}
    var_6 = module_0.Object(properties=var_3, required=var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = var_7['required']
    var_9 = bool(var_7['required'] == ['name'])
    assert var_9 is True

def test_case_0():
    pass



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_to_json_schema_with_definitions. Retrieved 3/7 statements.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True

import typesystem.composites as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.NeverMatch(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    assert var_2 is False

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['string', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minLength']
    assert var_4 == 5

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maxLength']
    assert var_4 == 10

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.String(allow_blank=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minLength']
    assert var_4 == 1

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'integer'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Integer(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['integer', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Float(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'number'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Decimal(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'number'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'boolean'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['boolean', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    assert var_5 == 'array'
    var_6 = var_4['items']['type']
    assert var_6 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = True
    var_3 = 'allow_null'
    var_4 = {var_3: var_2}
    var_5 = module_0.Array(var_1, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = var_6['type']
    var_8 = bool(var_6['type'] == ['array', 'null'])
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 2
    var_3 = {}
    var_4 = module_0.Array(var_1, min_items=var_2, **var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = var_5['minItems']
    assert var_6 == 2

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 5
    var_3 = {}
    var_4 = module_0.Array(var_1, max_items=var_2, **var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = var_5['maxItems']
    assert var_6 == 5

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = var_6['type']
    assert var_7 == 'object'
    var_8 = 'properties'
    var_9 = bool('properties' in var_6)
    assert var_9 is True
    var_10 = var_6['properties']['name']['type']
    assert var_10 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = var_1 | var_3
    var_5 = module_1.to_json_schema(var_4)
    var_6 = 'anyOf'
    var_7 = bool('anyOf' in var_5)
    assert var_7 is True
    var_8 = 'anyOf'
    var_9 = var_5[var_8]
    var_10 = len(var_9)
    assert var_10 == 2

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['default']
    assert var_5 == 'hello'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

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
    var_9 = module_1.to_json_schema(var_8)
    var_10 = var_9['enum']
    var_11 = bool(var_9['enum'] == ['a', 'b'])
    assert var_11 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'fixed_value'
    var_1 = {}
    var_2 = module_0.Const(var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['const']
    assert var_4 == 'fixed_value'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_3]
    var_5 = 'components'
    var_6 = 'schemas'
    var_7 = 'name'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = {}
    var_2 = module_0.Integer(minimum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minimum']
    assert var_4 == 0

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Integer(maximum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maximum']
    assert var_4 == 100

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = {}
    var_2 = module_0.Integer(exclusive_minimum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['exclusiveMinimum']
    assert var_4 == 0

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Integer(exclusive_maximum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['exclusiveMaximum']
    assert var_4 == 100

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.Integer(multiple_of=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['multipleOf']
    assert var_4 == 5



# Parsed testcases at query #38
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = None
    var_1 = {}
    var_2 = module_0.Array(additional_items=var_0, **var_1)
    var_3 = var_2.additional_items
    var_4 = var_3 is not var_0
    assert var_4 is False



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_to_json_schema_with_definitions. Retrieved 3/8 statements.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True

import typesystem.composites as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.NeverMatch(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    assert var_2 is False

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['string', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minLength']
    assert var_4 == 5

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maxLength']
    assert var_4 == 10

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'integer'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Integer(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['integer', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Float(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'number'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'boolean'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['boolean', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    assert var_5 == 'array'
    var_6 = 'items'
    var_7 = bool('items' in var_4)
    assert var_7 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 2
    var_3 = {}
    var_4 = module_0.Array(var_1, min_items=var_2, **var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = var_5['minItems']
    assert var_6 == 2

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 5
    var_3 = {}
    var_4 = module_0.Array(var_1, max_items=var_2, **var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = var_5['maxItems']
    assert var_6 == 5

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = var_6['type']
    assert var_7 == 'object'
    var_8 = 'properties'
    var_9 = bool('properties' in var_6)
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

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
    var_9 = module_1.to_json_schema(var_8)
    var_10 = var_9['enum']
    var_11 = bool(var_9['enum'] == ['a', 'b'])
    assert var_11 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'fixed_value'
    var_1 = {}
    var_2 = module_0.Const(var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['const']
    assert var_4 == 'fixed_value'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = var_1 | var_3
    var_5 = module_1.to_json_schema(var_4)
    var_6 = 'anyOf'
    var_7 = bool('anyOf' in var_5)
    assert var_7 is True
    var_8 = 'anyOf'
    var_9 = var_5[var_8]
    var_10 = len(var_9)
    assert var_10 == 2

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['default']
    assert var_5 == 'hello'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_3]
    var_5 = 'components'
    var_6 = 'schemas'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = {}
    var_2 = module_0.Integer(minimum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minimum']
    assert var_4 == 0

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Integer(maximum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maximum']
    assert var_4 == 100

import re as module_0
import typesystem.fields as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.compile(var_0)
    var_2 = {}
    var_3 = module_1.String(pattern=var_1, **var_2)
    var_4 = module_2.to_json_schema(var_3)
    var_5 = var_4['pattern']
    assert var_5 == '^\\d+$'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'email'
    var_1 = {}
    var_2 = module_0.String(format=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['format']
    assert var_4 == 'email'



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_predicate_at_line_172_evaluates_to_true. Retrieved 4/17 statements.


import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'allow_null'
    var_1 = [var_0]
    var_2 = []
    var_3 = module_0.to_json_schema(var_1)
    var_4 = 'components'
    var_5 = bool('components' in var_3)
    assert var_5 is True
    var_6 = 'schemas'
    var_7 = bool('schemas' in var_3['components'])
    assert var_7 is True



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_to_json_schema_with_definitions. Retrieved 3/7 statements.
# Partially parsed test_to_json_schema_with_reference_field. Retrieved 2/5 statements.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True

import typesystem.composites as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.NeverMatch(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    assert var_2 is False

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['string', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minLength']
    assert var_4 == 5

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maxLength']
    assert var_4 == 10

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'integer'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Integer(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['integer', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = {}
    var_2 = module_0.Integer(minimum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minimum']
    assert var_4 == 0

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Integer(maximum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maximum']
    assert var_4 == 100

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Float(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'number'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Decimal(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'number'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'boolean'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['boolean', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    assert var_5 == 'array'
    var_6 = var_4['items']['type']
    assert var_6 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 1
    var_3 = {}
    var_4 = module_0.Array(var_1, min_items=var_2, **var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = var_5['minItems']
    assert var_6 == 1

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 10
    var_3 = {}
    var_4 = module_0.Array(var_1, max_items=var_2, **var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = var_5['maxItems']
    assert var_6 == 10

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = var_6['type']
    assert var_7 == 'object'
    var_8 = 'name'
    var_9 = bool('name' in var_6['properties'])
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 1
    var_1 = 'one'
    var_2 = (var_0, var_1)
    var_3 = 2
    var_4 = 'two'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = {}
    var_8 = module_0.Choice(choices=var_6, **var_7)
    var_9 = module_1.to_json_schema(var_8)
    var_10 = var_9['enum']
    var_11 = bool(var_9['enum'] == [1, 2])
    assert var_11 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'fixed_value'
    var_1 = {}
    var_2 = module_0.Const(var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['const']
    assert var_4 == 'fixed_value'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = 'anyOf'
    var_9 = bool('anyOf' in var_7)
    assert var_9 is True
    var_10 = 'anyOf'
    var_11 = var_7[var_10]
    var_12 = len(var_11)
    assert var_12 == 2

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'default_value'
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['default']
    assert var_5 == 'default_value'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_3]
    var_5 = 'components'
    var_6 = 'schemas'
    var_7 = 'name'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = {}
    var_2 = module_0.String(pattern=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = 'pattern'
    var_5 = bool('pattern' in var_3)
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'email'
    var_1 = {}
    var_2 = module_0.String(format=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['format']
    assert var_4 == 'email'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.String(allow_blank=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minLength']
    assert var_4 == 1

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 'MySchema'
    var_3 = 'target'
    var_4 = {var_3: var_1}
    var_5 = 'components'

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_1.OneOf(var_4, **var_5)
    var_7 = module_2.to_json_schema(var_6)
    var_8 = 'oneOf'
    var_9 = bool('oneOf' in var_7)
    assert var_9 is True
    var_10 = 'oneOf'
    var_11 = var_7[var_10]
    var_12 = len(var_11)
    assert var_12 == 2

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_1.AllOf(var_2, **var_3)
    var_5 = module_2.to_json_schema(var_4)
    var_6 = 'allOf'
    var_7 = bool('allOf' in var_5)
    assert var_7 is True
    var_8 = 'allOf'
    var_9 = var_5[var_8]
    var_10 = len(var_9)
    assert var_10 == 1

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_1.Not(var_1, **var_2)
    var_4 = module_2.to_json_schema(var_3)
    var_5 = 'not'
    var_6 = bool('not' in var_4)
    assert var_6 is True

def test_case_0():
    pass



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_ref_from_json_schema_valid_reference. Retrieved 3/8 statements.
# Partially parsed test_ref_from_json_schema_with_different_path. Retrieved 3/8 statements.
# Partially parsed test_ref_from_json_schema_invalid_ref_style. Retrieved 3/8 statements.
# Partially parsed test_ref_from_json_schema_with_external_ref. Retrieved 3/8 statements.
# Partially parsed test_ref_from_json_schema_preserves_definitions_reference. Retrieved 6/10 statements.


def test_case_0():
    var_0 = []
    var_1 = '$ref'
    var_2 = '#/definitions/MyType'
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = []
    var_1 = '$ref'
    var_2 = '#/components/schemas/User'
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = []
    var_1 = '$ref'
    var_2 = 'definitions/MyType'
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Unsupported $ref style in document.'

def test_case_0():
    var_0 = []
    var_1 = '$ref'
    var_2 = 'other.json#/definitions/Type'
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Unsupported $ref style in document.'

def test_case_0():
    var_0 = 'Type1'
    var_1 = 'value1'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = '$ref'
    var_5 = '#/definitions/Type1'
    var_6 = {var_4: var_5}



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_predicate_line_108_evaluates_to_true. Retrieved 4/17 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = var_1.property_names
    var_3 = None
    var_4 = var_2 is not var_3
    assert var_4 is True



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_predicate_at_line_158_evaluates_to_false. Retrieved 1/12 statements.


def test_case_0():
    var_0 = None



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_else_clause_is_none. Retrieved 1/15 statements.


def test_case_0():
    var_0 = None



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_ref_from_json_schema_unsupported_ref_style. Retrieved 3/8 statements.


def test_case_0():
    var_0 = []
    var_1 = '$ref'
    var_2 = 'https://example.com/schema'
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_schema_with_allow_null_true. Retrieved 3/14 statements.
# Partially parsed test_schema_with_allow_null_false. Retrieved 3/12 statements.


def test_case_0():
    var_0 = {}
    var_1 = 'object'
    var_2 = 'null'
    var_3 = [var_1, var_2]

def test_case_0():
    var_0 = {}
    var_1 = 'object'
    var_2 = 'null'
    var_3 = [var_1, var_2]



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_schema_field_with_allow_null_true. Retrieved 6/14 statements.


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = None
    var_3 = 'object'
    var_4 = 'null'
    var_5 = [var_3, var_4]



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_from_json_schema_type_number. Retrieved 9/15 statements.
# Partially parsed test_from_json_schema_type_integer. Retrieved 9/15 statements.
# Partially parsed test_from_json_schema_type_string. Retrieved 11/17 statements.
# Partially parsed test_from_json_schema_type_string_allow_blank. Retrieved 5/11 statements.
# Partially parsed test_from_json_schema_type_boolean. Retrieved 5/11 statements.
# Partially parsed test_from_json_schema_type_array_simple. Retrieved 13/21 statements.
# Partially parsed test_from_json_schema_type_array_no_items. Retrieved 3/9 statements.
# Partially parsed test_from_json_schema_type_array_additional_items_bool. Retrieved 4/10 statements.
# Partially parsed test_from_json_schema_type_object_simple. Retrieved 15/23 statements.
# Partially parsed test_from_json_schema_type_object_no_properties. Retrieved 3/9 statements.
# Partially parsed test_from_json_schema_type_object_pattern_properties. Retrieved 12/18 statements.
# Partially parsed test_from_json_schema_type_object_additional_properties_bool. Retrieved 4/10 statements.
# Partially parsed test_from_json_schema_type_object_property_names. Retrieved 7/15 statements.
# Partially parsed test_from_json_schema_type_with_default. Retrieved 5/11 statements.


def test_case_0():
    var_0 = []
    var_1 = 'minimum'
    var_2 = 'maximum'
    var_3 = 'multipleOf'
    var_4 = 0
    var_5 = 100
    var_6 = 5
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = 'number'
    var_9 = False

def test_case_0():
    var_0 = []
    var_1 = 'minimum'
    var_2 = 'maximum'
    var_3 = 'exclusiveMinimum'
    var_4 = 1
    var_5 = 10
    var_6 = 0
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = 'integer'
    var_9 = True

def test_case_0():
    var_0 = []
    var_1 = 'minLength'
    var_2 = 'maxLength'
    var_3 = 'pattern'
    var_4 = 'format'
    var_5 = 2
    var_6 = 50
    var_7 = '^[a-z]+$'
    var_8 = 'email'
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_7, var_4: var_8}
    var_10 = 'string'
    var_11 = False

def test_case_0():
    var_0 = []
    var_1 = 'minLength'
    var_2 = 0
    var_3 = {var_1: var_2}
    var_4 = 'string'
    var_5 = False

def test_case_0():
    var_0 = []
    var_1 = 'default'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = 'boolean'
    var_5 = False

def test_case_0():
    var_0 = []
    var_1 = 'items'
    var_2 = 'minItems'
    var_3 = 'maxItems'
    var_4 = 'uniqueItems'
    var_5 = 'type'
    var_6 = 'string'
    var_7 = {var_5: var_6}
    var_8 = 1
    var_9 = 10
    var_10 = True
    var_11 = {var_1: var_7, var_2: var_8, var_3: var_9, var_4: var_10}
    var_12 = 'array'
    var_13 = False

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = 'array'
    var_3 = True

def test_case_0():
    var_0 = []
    var_1 = 'additionalItems'
    var_2 = False
    var_3 = {var_1: var_2}
    var_4 = 'array'

def test_case_0():
    var_0 = []
    var_1 = 'properties'
    var_2 = 'required'
    var_3 = 'minProperties'
    var_4 = 'maxProperties'
    var_5 = 'name'
    var_6 = 'type'
    var_7 = 'string'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = [var_5]
    var_11 = 1
    var_12 = 5
    var_13 = {var_1: var_9, var_2: var_10, var_3: var_11, var_4: var_12}
    var_14 = 'object'
    var_15 = False
    var_16 = 'name'

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = 'object'
    var_3 = True

def test_case_0():
    var_0 = []
    var_1 = 'patternProperties'
    var_2 = '^S_'
    var_3 = '^I_'
    var_4 = 'type'
    var_5 = 'string'
    var_6 = {var_4: var_5}
    var_7 = 'integer'
    var_8 = {var_4: var_7}
    var_9 = {var_2: var_6, var_3: var_8}
    var_10 = {var_1: var_9}
    var_11 = 'object'
    var_12 = False
    var_13 = '^S_'
    var_14 = '^I_'

def test_case_0():
    var_0 = []
    var_1 = 'additionalProperties'
    var_2 = False
    var_3 = {var_1: var_2}
    var_4 = 'object'

def test_case_0():
    var_0 = []
    var_1 = 'propertyNames'
    var_2 = 'pattern'
    var_3 = '^[a-z]+$'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'object'
    var_7 = False

def test_case_0():
    var_0 = []
    var_1 = 'default'
    var_2 = 42
    var_3 = {var_1: var_2}
    var_4 = 'integer'
    var_5 = False



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_predicate_at_line_172_evaluates_to_true. Retrieved 6/11 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 'TestSchema'
    var_3 = {}
    var_4 = module_0.String(**var_3)
    var_5 = {var_2: var_4}
    var_6 = [var_5]
    var_7 = 'components'
    var_8 = 'schemas'
    var_9 = 'schemas'
    var_10 = 'components'



# Parsed testcases at query #51
#--------------------------

# Failed to parse test_pattern_regex_flags_unicode.




# Parsed testcases at query #52
#--------------------------




import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'additionalItems'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'array'
    var_6 = False
    var_7 = {}
    var_8 = module_0.from_json_schema_type(var_4, var_5, var_6, var_7)
    var_9 = bool(var_8 is not None)
    assert var_9 is True



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_to_json_schema_with_definitions. Retrieved 3/8 statements.
# Partially parsed test_to_json_schema_with_reference_field. Retrieved 2/6 statements.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True

import typesystem.composites as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.NeverMatch(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    assert var_2 is False

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['string', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'test_value'
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['default']
    assert var_5 == 'test_value'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 2
    var_1 = 10
    var_2 = {}
    var_3 = module_0.String(max_length=var_1, min_length=var_0, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['minLength']
    assert var_5 == 2
    var_6 = var_4['maxLength']
    assert var_6 == 10

import re as module_0
import typesystem.fields as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = '^[a-z]+$'
    var_1 = module_0.compile(var_0)
    var_2 = {}
    var_3 = module_1.String(pattern=var_1, **var_2)
    var_4 = module_2.to_json_schema(var_3)
    var_5 = var_4['pattern']
    assert var_5 == '^[a-z]+$'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'integer'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = 100
    var_2 = {}
    var_3 = module_0.Integer(minimum=var_0, maximum=var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['minimum']
    assert var_5 == 0
    var_6 = var_4['maximum']
    assert var_6 == 100

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Float(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'number'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'boolean'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    assert var_5 == 'array'
    var_6 = var_4['items']['type']
    assert var_6 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 1
    var_3 = 10
    var_4 = {}
    var_5 = module_0.Array(var_1, min_items=var_2, max_items=var_3, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = var_6['minItems']
    assert var_7 == 1
    var_8 = var_6['maxItems']
    assert var_8 == 10

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = var_6['type']
    assert var_7 == 'object'
    var_8 = 'properties'
    var_9 = bool('properties' in var_6)
    assert var_9 is True
    var_10 = var_6['properties']['name']['type']
    assert var_10 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = var_1 | var_3
    var_5 = module_1.to_json_schema(var_4)
    var_6 = 'anyOf'
    var_7 = bool('anyOf' in var_5)
    assert var_7 is True
    var_8 = 'anyOf'
    var_9 = var_5[var_8]
    var_10 = len(var_9)
    assert var_10 == 2

import typesystem.fields as module_0
import typesystem.json_schema as module_1

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
    var_9 = module_1.to_json_schema(var_8)
    var_10 = var_9['enum']
    var_11 = bool(var_9['enum'] == ['a', 'b'])
    assert var_11 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'fixed_value'
    var_1 = {}
    var_2 = module_0.Const(var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['const']
    assert var_4 == 'fixed_value'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_3]
    var_5 = 'components'
    var_6 = 'schemas'
    var_7 = 'name'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 'MyType'
    var_3 = 'target'
    var_4 = {var_3: var_1}
    var_5 = 'components'

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_1.OneOf(var_4, **var_5)
    var_7 = module_2.to_json_schema(var_6)
    var_8 = 'oneOf'
    var_9 = bool('oneOf' in var_7)
    assert var_9 is True
    var_10 = 'oneOf'
    var_11 = var_7[var_10]
    var_12 = len(var_11)
    assert var_12 == 2

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_1.AllOf(var_2, **var_3)
    var_5 = module_2.to_json_schema(var_4)
    var_6 = 'allOf'
    var_7 = bool('allOf' in var_5)
    assert var_7 is True

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_1.Not(var_1, **var_2)
    var_4 = module_2.to_json_schema(var_3)
    var_5 = 'not'
    var_6 = bool('not' in var_4)
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = {}
    var_5 = module_1.IfThenElse(var_1, var_3, **var_4)
    var_6 = module_2.to_json_schema(var_5)



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_then_clause_is_none. Retrieved 1/13 statements.


def test_case_0():
    var_0 = None



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_predicate_at_line_108_evaluates_to_true. Retrieved 1/6 statements.


def test_case_0():
    var_0 = None



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_else_clause_is_none. Retrieved 1/9 statements.


def test_case_0():
    var_0 = None



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_pattern_regex_flags_equal_unicode. Retrieved 1/16 statements.


def test_case_0():
    var_0 = 'test'



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_pattern_regex_flags_equals_unicode. Retrieved 3/22 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = 'String'
    var_2 = {}



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_to_json_schema_with_definitions. Retrieved 3/7 statements.
# Partially parsed test_to_json_schema_with_reference_field. Retrieved 2/5 statements.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True

import typesystem.composites as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.NeverMatch(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    assert var_2 is False

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['string', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minLength']
    assert var_4 == 5

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maxLength']
    assert var_4 == 10

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.String(allow_blank=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minLength']
    assert var_4 == 1

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'integer'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Integer(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['integer', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = {}
    var_2 = module_0.Integer(minimum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minimum']
    assert var_4 == 0

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Integer(maximum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maximum']
    assert var_4 == 100

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Float(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'number'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Decimal(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'number'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'boolean'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['boolean', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Array(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'array'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Array(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['array', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Array(min_items=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minItems']
    assert var_4 == 1

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Array(max_items=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maxItems']
    assert var_4 == 10

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Array(unique_items=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['uniqueItems']
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'object'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Object(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['object', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = 'properties'
    var_8 = bool('properties' in var_6)
    assert var_8 is True
    var_9 = 'name'
    var_10 = bool('name' in var_6['properties'])
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Object(required=var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['required']
    var_6 = bool(var_4['required'] == ['name'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

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
    var_9 = module_1.to_json_schema(var_8)
    var_10 = var_9['enum']
    var_11 = bool(var_9['enum'] == ['a', 'b'])
    assert var_11 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'value'
    var_1 = {}
    var_2 = module_0.Const(var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['const']
    assert var_4 == 'value'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = 'anyOf'
    var_9 = bool('anyOf' in var_7)
    assert var_9 is True
    var_10 = 'anyOf'
    var_11 = var_7[var_10]
    var_12 = len(var_11)
    assert var_12 == 2

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'default_value'
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['default']
    assert var_5 == 'default_value'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_3]
    var_5 = 'components'
    var_6 = 'schemas'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'User'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = 'target'
    var_4 = {var_3: var_2}

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_1.OneOf(var_4, **var_5)
    var_7 = module_2.to_json_schema(var_6)
    var_8 = 'oneOf'
    var_9 = bool('oneOf' in var_7)
    assert var_9 is True
    var_10 = 'oneOf'
    var_11 = var_7[var_10]
    var_12 = len(var_11)
    assert var_12 == 2

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_1.AllOf(var_4, **var_5)
    var_7 = module_2.to_json_schema(var_6)



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_schema_with_allow_null_false_generates_object_type. Retrieved 4/14 statements.
# Partially parsed test_schema_with_allow_null_true_generates_object_null_type. Retrieved 4/14 statements.


def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = 'object'
    var_3 = 'null'
    var_4 = [var_2, var_3]
    var_5 = var_1['type']
    assert var_5 == 'object'

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = 'object'
    var_3 = 'null'
    var_4 = [var_2, var_3]
    var_5 = var_1['type']
    var_6 = bool(var_1['type'] == ['object', 'null'])
    assert var_6 is True



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_pattern_properties_predicate_evaluates_to_true. Retrieved 1/14 statements.


def test_case_0():
    var_0 = '^S_'



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_predicate_at_line_78_evaluates_to_false. Retrieved 1/6 statements.


def test_case_0():
    var_0 = None



# Parsed testcases at query #63
#--------------------------

# Failed to parse test_to_json_schema_with_any_instance.




# Parsed testcases at query #64
#--------------------------

# Partially parsed test_pattern_properties_predicate_evaluates_to_true. Retrieved 1/7 statements.


def test_case_0():
    var_0 = '^[a-z]+$'



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_to_json_schema_with_definitions. Retrieved 3/7 statements.
# Partially parsed test_to_json_schema_with_reference_field. Retrieved 2/5 statements.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True

import typesystem.composites as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.NeverMatch(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    assert var_2 is False

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['string', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minLength']
    assert var_4 == 5

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maxLength']
    assert var_4 == 10

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['default']
    assert var_5 == 'hello'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'integer'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Integer(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['integer', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = {}
    var_2 = module_0.Integer(minimum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minimum']
    assert var_4 == 0

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Integer(maximum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maximum']
    assert var_4 == 100

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Float(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'number'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Decimal(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'number'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'boolean'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['boolean', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    assert var_5 == 'array'
    var_6 = var_4['items']['type']
    assert var_6 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = True
    var_3 = 'allow_null'
    var_4 = {var_3: var_2}
    var_5 = module_0.Array(var_1, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = var_6['type']
    var_8 = bool(var_6['type'] == ['array', 'null'])
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 1
    var_3 = {}
    var_4 = module_0.Array(var_1, min_items=var_2, **var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = var_5['minItems']
    assert var_6 == 1

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 10
    var_3 = {}
    var_4 = module_0.Array(var_1, max_items=var_2, **var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = var_5['maxItems']
    assert var_6 == 10

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = var_6['type']
    assert var_7 == 'object'
    var_8 = 'properties'
    var_9 = bool('properties' in var_6)
    assert var_9 is True
    var_10 = var_6['properties']['name']['type']
    assert var_10 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = True
    var_5 = 'allow_null'
    var_6 = {var_5: var_4}
    var_7 = module_0.Object(properties=var_3, **var_6)
    var_8 = module_1.to_json_schema(var_7)
    var_9 = var_8['type']
    var_10 = bool(var_8['type'] == ['object', 'null'])
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = 'anyOf'
    var_9 = bool('anyOf' in var_7)
    assert var_9 is True
    var_10 = 'anyOf'
    var_11 = var_7[var_10]
    var_12 = len(var_11)
    assert var_12 == 2

import typesystem.fields as module_0
import typesystem.json_schema as module_1

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
    var_9 = module_1.to_json_schema(var_8)
    var_10 = var_9['enum']
    var_11 = bool(var_9['enum'] == ['a', 'b'])
    assert var_11 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'fixed_value'
    var_1 = {}
    var_2 = module_0.Const(var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['const']
    assert var_4 == 'fixed_value'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_3]
    var_5 = 'components'
    var_6 = 'schemas'
    var_7 = 'name'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 'StringType'
    var_3 = 'target'
    var_4 = {var_3: var_1}
    var_5 = 'components'
    var_6 = 'StringType'

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_1.OneOf(var_4, **var_5)
    var_7 = module_2.to_json_schema(var_6)
    var_8 = 'oneOf'
    var_9 = bool('oneOf' in var_7)
    assert var_9 is True
    var_10 = 'oneOf'
    var_11 = var_7[var_10]
    var_12 = len(var_11)
    assert var_12 == 2

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_1.AllOf(var_2, **var_3)
    var_5 = module_2.to_json_schema(var_4)
    var_6 = 'allOf'
    var_7 = bool('allOf' in var_5)
    assert var_7 is True
    var_8 = 'allOf'
    var_9 = var_5[var_8]
    var_10 = len(var_9)
    assert var_10 == 1

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_1.Not(var_1, **var_2)
    var_4 = module_2.to_json_schema(var_3)
    var_5 = 'not'
    var_6 = bool('not' in var_4)
    assert var_6 is True
    var_7 = var_4['not']['type']
    assert var_7 == 'string'

def test_case_0():
    pass



# Parsed testcases at query #66
#--------------------------

# Partially parsed test_to_json_schema_predicate_line_1_evaluates_to_false. Retrieved 1/4 statements.


def test_case_0():
    var_0 = []



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_to_json_schema_with_definitions. Retrieved 3/7 statements.
# Partially parsed test_to_json_schema_with_reference_field. Retrieved 2/5 statements.
# Partially parsed test_to_json_schema_with_schema_field. Retrieved 1/6 statements.
# Failed to parse test_to_json_schema_with_one_of_field.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True

import typesystem.composites as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.NeverMatch(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    assert var_2 is False

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['string', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minLength']
    assert var_4 == 5

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maxLength']
    assert var_4 == 10

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.String(allow_blank=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minLength']
    assert var_4 == 1

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'integer'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Integer(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['integer', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = {}
    var_2 = module_0.Integer(minimum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minimum']
    assert var_4 == 0

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Integer(maximum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maximum']
    assert var_4 == 100

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Float(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'number'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Decimal(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'number'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'boolean'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['boolean', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    assert var_5 == 'array'
    var_6 = var_4['items']['type']
    assert var_6 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = True
    var_3 = 'allow_null'
    var_4 = {var_3: var_2}
    var_5 = module_0.Array(var_1, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = var_6['type']
    var_8 = bool(var_6['type'] == ['array', 'null'])
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 1
    var_3 = {}
    var_4 = module_0.Array(var_1, min_items=var_2, **var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = var_5['minItems']
    assert var_6 == 1

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 10
    var_3 = {}
    var_4 = module_0.Array(var_1, max_items=var_2, **var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = var_5['maxItems']
    assert var_6 == 10

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = True
    var_3 = {}
    var_4 = module_0.Array(var_1, unique_items=var_2, **var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = var_5['uniqueItems']
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = var_6['type']
    assert var_7 == 'object'
    var_8 = 'name'
    var_9 = bool('name' in var_6['properties'])
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = True
    var_5 = 'allow_null'
    var_6 = {var_5: var_4}
    var_7 = module_0.Object(properties=var_3, **var_6)
    var_8 = module_1.to_json_schema(var_7)
    var_9 = var_8['type']
    var_10 = bool(var_8['type'] == ['object', 'null'])
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_0]
    var_5 = {}
    var_6 = module_0.Object(properties=var_3, required=var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = var_7['required']
    var_9 = bool(var_7['required'] == ['name'])
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = 'anyOf'
    var_9 = bool('anyOf' in var_7)
    assert var_9 is True
    var_10 = 'anyOf'
    var_11 = var_7[var_10]
    var_12 = len(var_11)
    assert var_12 == 2

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['default']
    assert var_5 == 'hello'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

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
    var_9 = module_1.to_json_schema(var_8)
    var_10 = var_9['enum']
    var_11 = bool(var_9['enum'] == ['a', 'b'])
    assert var_11 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'fixed'
    var_1 = {}
    var_2 = module_0.Const(var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['const']
    assert var_4 == 'fixed'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_3]
    var_5 = 'components'
    var_6 = 'schemas'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 'MyString'
    var_3 = 'target'
    var_4 = {var_3: var_1}
    var_5 = 'components'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 'properties'



# Parsed testcases at query #68
#--------------------------

# Partially parsed test_to_json_schema_with_not_field. Retrieved 2/4 statements.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True

import typesystem.composites as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.NeverMatch(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    assert var_2 is False

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['string', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minLength']
    assert var_4 == 5

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maxLength']
    assert var_4 == 10

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.String(allow_blank=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minLength']
    assert var_4 == 1

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'integer'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Integer(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['integer', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = {}
    var_2 = module_0.Integer(minimum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minimum']
    assert var_4 == 0

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Integer(maximum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maximum']
    assert var_4 == 100

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Float(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'number'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Decimal(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'number'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'boolean'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['boolean', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Array(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'array'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Array(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['array', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Array(min_items=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minItems']
    assert var_4 == 1

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Array(max_items=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maxItems']
    assert var_4 == 10

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['items']['type']
    assert var_5 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Array(unique_items=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['uniqueItems']
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'object'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Object(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['object', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = 'properties'
    var_8 = bool('properties' in var_6)
    assert var_8 is True
    var_9 = 'name'
    var_10 = bool('name' in var_6['properties'])
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Object(required=var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['required']
    var_6 = bool(var_4['required'] == ['name'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

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
    var_9 = module_1.to_json_schema(var_8)
    var_10 = var_9['enum']
    var_11 = bool(var_9['enum'] == ['a', 'b'])
    assert var_11 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'value'
    var_1 = {}
    var_2 = module_0.Const(var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['const']
    assert var_4 == 'value'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = 'anyOf'
    var_9 = bool('anyOf' in var_7)
    assert var_9 is True
    var_10 = 'anyOf'
    var_11 = var_7[var_10]
    var_12 = len(var_11)
    assert var_12 == 2

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_1.OneOf(var_4, **var_5)
    var_7 = module_2.to_json_schema(var_6)
    var_8 = 'oneOf'
    var_9 = bool('oneOf' in var_7)
    assert var_9 is True
    var_10 = 'oneOf'
    var_11 = var_7[var_10]
    var_12 = len(var_11)
    assert var_12 == 2

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_1.AllOf(var_2, **var_3)
    var_5 = module_2.to_json_schema(var_4)
    var_6 = 'allOf'
    var_7 = bool('allOf' in var_5)
    assert var_7 is True
    var_8 = 'allOf'
    var_9 = var_5[var_8]
    var_10 = len(var_9)
    assert var_10 == 1

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = {}
    var_5 = module_1.IfThenElse(var_1, var_3, **var_4)
    var_6 = module_2.to_json_schema(var_5)
    var_7 = 'if'
    var_8 = bool('if' in var_6)
    assert var_8 is True
    var_9 = 'then'
    var_10 = bool('then' in var_6)
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_1.Not(var_1, **var_2)



# Parsed testcases at query #69
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_isinstance_arg_any_evaluates_to_false. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 'NotAny'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]
    var_4 = None
    var_5 = [var_4]



# Parsed testcases at query #71
#--------------------------

# Failed to parse test_to_json_schema_with_definitions.
# Partially parsed test_to_json_schema_with_reference_field. Retrieved 2/6 statements.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True

import typesystem.composites as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.NeverMatch(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    assert var_2 is False

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['string', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minLength']
    assert var_4 == 5

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maxLength']
    assert var_4 == 10

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.String(allow_blank=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minLength']
    assert var_4 == 1

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'integer'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Integer(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['integer', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Float(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'number'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Decimal(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'number'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'boolean'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['boolean', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Array(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'array'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Array(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['array', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Array(min_items=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minItems']
    assert var_4 == 2

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.Array(max_items=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maxItems']
    assert var_4 == 5

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'object'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Object(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['object', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

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
    var_9 = module_1.to_json_schema(var_8)
    var_10 = var_9['enum']
    var_11 = bool(var_9['enum'] == ['a', 'b'])
    assert var_11 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'fixed_value'
    var_1 = {}
    var_2 = module_0.Const(var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['const']
    assert var_4 == 'fixed_value'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = 'anyOf'
    var_9 = bool('anyOf' in var_7)
    assert var_9 is True
    var_10 = 'anyOf'
    var_11 = var_7[var_10]
    var_12 = len(var_11)
    assert var_12 == 2

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'default_value'
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['default']
    assert var_5 == 'default_value'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 'StringDef'
    var_3 = 'target'
    var_4 = {var_3: var_1}
    var_5 = 'components'
    var_6 = 'StringDef'



# Parsed testcases at query #72
#--------------------------

# Partially parsed test_to_json_schema_with_definitions. Retrieved 3/8 statements.
# Partially parsed test_to_json_schema_with_array_tuple_items. Retrieved 9/12 statements.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True

import typesystem.composites as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.NeverMatch(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    assert var_2 is False

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['string', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minLength']
    assert var_4 == 5

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maxLength']
    assert var_4 == 10

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'integer'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Integer(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['integer', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Float(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'number'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'boolean'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['boolean', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    assert var_5 == 'array'
    var_6 = var_4['items']['type']
    assert var_6 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 2
    var_3 = {}
    var_4 = module_0.Array(var_1, min_items=var_2, **var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = var_5['minItems']
    assert var_6 == 2

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 5
    var_3 = {}
    var_4 = module_0.Array(var_1, max_items=var_2, **var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = var_5['maxItems']
    assert var_6 == 5

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = var_1 | var_3
    var_5 = module_1.to_json_schema(var_4)
    var_6 = 'anyOf'
    var_7 = bool('anyOf' in var_5)
    assert var_7 is True
    var_8 = 'anyOf'
    var_9 = var_5[var_8]
    var_10 = len(var_9)
    assert var_10 == 2

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['default']
    assert var_5 == 'hello'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

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
    var_9 = module_1.to_json_schema(var_8)
    var_10 = var_9['enum']
    var_11 = bool(var_9['enum'] == ['a', 'b'])
    assert var_11 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = var_6['type']
    assert var_7 == 'object'
    var_8 = 'properties'
    var_9 = bool('properties' in var_6)
    assert var_9 is True
    var_10 = 'name'
    var_11 = bool('name' in var_6['properties'])
    assert var_11 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_3]
    var_5 = 'components'
    var_6 = 'schemas'
    var_7 = 'Name'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = '^[a-z]+$'
    var_1 = {}
    var_2 = module_0.String(pattern=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = 'pattern'
    var_5 = bool('pattern' in var_3)
    assert var_5 is True
    var_6 = var_3['pattern']
    assert var_6 == '^[a-z]+$'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'email'
    var_1 = {}
    var_2 = module_0.String(format=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['format']
    assert var_4 == 'email'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = {}
    var_2 = module_0.Integer(minimum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minimum']
    assert var_4 == 0

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Integer(maximum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maximum']
    assert var_4 == 100

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = (var_1, var_3)
    var_5 = {}
    var_6 = module_0.Array(var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = 'items'
    var_9 = var_7[var_8]
    var_10 = var_7[var_8]
    var_11 = len(var_10)
    assert var_11 == 2



# Parsed testcases at query #73
#--------------------------




import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 'test_key'
    var_3 = 'test_value'
    var_4 = {var_2: var_3}
    var_5 = module_1.to_json_schema(var_1, var_4)
    var_6 = var_5['components']['schemas']
    var_7 = bool(var_5['components']['schemas'] == var_4)
    assert var_7 is True



# Parsed testcases at query #74
#--------------------------

# Partially parsed test_to_json_schema_with_definitions. Retrieved 3/7 statements.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True

import typesystem.composites as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.NeverMatch(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    assert var_2 is False

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['string', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minLength']
    assert var_4 == 5

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maxLength']
    assert var_4 == 10

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'integer'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Integer(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['integer', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Float(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'number'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Decimal(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'number'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'boolean'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['boolean', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    assert var_5 == 'array'
    var_6 = 'items'
    var_7 = bool('items' in var_4)
    assert var_7 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = True
    var_3 = 'allow_null'
    var_4 = {var_3: var_2}
    var_5 = module_0.Array(var_1, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = var_6['type']
    var_8 = bool(var_6['type'] == ['array', 'null'])
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'object'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Object(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['object', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = var_1 | var_3
    var_5 = module_1.to_json_schema(var_4)
    var_6 = 'anyOf'
    var_7 = bool('anyOf' in var_5)
    assert var_7 is True
    var_8 = 'anyOf'
    var_9 = var_5[var_8]
    var_10 = len(var_9)
    assert var_10 == 2

import typesystem.fields as module_0
import typesystem.json_schema as module_1

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
    var_9 = module_1.to_json_schema(var_8)
    var_10 = 'enum'
    var_11 = bool('enum' in var_9)
    assert var_11 is True
    var_12 = var_9['enum']
    var_13 = bool(var_9['enum'] == ['a', 'b'])
    assert var_13 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['default']
    assert var_5 == 'test'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.String(allow_blank=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minLength']
    assert var_4 == 1

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = {}
    var_2 = module_0.Integer(minimum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minimum']
    assert var_4 == 0

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Integer(maximum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maximum']
    assert var_4 == 100

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 1
    var_3 = {}
    var_4 = module_0.Array(var_1, min_items=var_2, **var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = var_5['minItems']
    assert var_6 == 1

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 10
    var_3 = {}
    var_4 = module_0.Array(var_1, max_items=var_2, **var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = var_5['maxItems']
    assert var_6 == 10

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = True
    var_3 = {}
    var_4 = module_0.Array(var_1, unique_items=var_2, **var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = var_5['uniqueItems']
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = 'properties'
    var_8 = bool('properties' in var_6)
    assert var_8 is True
    var_9 = 'name'
    var_10 = bool('name' in var_6['properties'])
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_0]
    var_5 = {}
    var_6 = module_0.Object(properties=var_3, required=var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = var_7['required']
    var_9 = bool(var_7['required'] == ['name'])
    assert var_9 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_3]
    var_5 = 'components'
    var_6 = 'schemas'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'constant_value'
    var_1 = {}
    var_2 = module_0.Const(var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['const']
    assert var_4 == 'constant_value'

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_1.OneOf(var_4, **var_5)
    var_7 = module_2.to_json_schema(var_6)
    var_8 = 'oneOf'
    var_9 = bool('oneOf' in var_7)
    assert var_9 is True
    var_10 = 'oneOf'
    var_11 = var_7[var_10]
    var_12 = len(var_11)
    assert var_12 == 2

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_1.AllOf(var_4, **var_5)
    var_7 = module_2.to_json_schema(var_6)
    var_8 = 'allOf'
    var_9 = bool('allOf' in var_7)
    assert var_9 is True
    var_10 = 'allOf'
    var_11 = var_7[var_10]
    var_12 = len(var_11)
    assert var_12 == 2



# Parsed testcases at query #75
#--------------------------

# Partially parsed test_pattern_properties_predicate_evaluates_to_true. Retrieved 1/14 statements.


def test_case_0():
    var_0 = '^S_'



# Parsed testcases at query #76
#--------------------------

# Partially parsed test_predicate_at_line_108_evaluates_to_true. Retrieved 1/6 statements.


def test_case_0():
    var_0 = None



