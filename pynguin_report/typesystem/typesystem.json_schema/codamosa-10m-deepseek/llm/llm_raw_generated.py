####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

# Unit test for function ref_from_json_schema
def test_ref_from_json_schema():<｜end▁of▁thinking｜>```python
import pytest
from typesystem.fields import Reference
from typesystem.schemas import Definitions



# LLM-generated content at query #2
#--------------------------

# Unit test for function ref_from_json_schema
def test_ref_from_json_schema():


# LLM-generated content at query #3
#--------------------------

# Unit test for function any_of_from_json_schema
def test_any_of_from_json_schema(): 
    # Test case 1: Basic functionality
    data = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
    definitions = Definitions()
    result = any_of_from_json_schema(data, definitions)
    assert isinstance(result, Union)
    assert len(result.any_of) == 2
    assert isinstance(result.any_of[0], String)
    assert isinstance(result.any_of[1], Integer)

    # Test case 2: With default value
    data = {"anyOf": [{"type": "string"}, {"type": "integer"}], "default": "test"}
    result = any_of_from_json_schema(data, definitions)
    assert result.default == "test"

    # Test case 3: Nested anyOf
    data = {"anyOf": [{"anyOf": [{"type": "string"}]}, {"type": "integer"}]}
    result = any_of_from_json_schema(data, definitions)
    assert isinstance(result, Union)
    assert len(result.any_of) == 2
    assert isinstance(result.any_of[0], Union)
    assert isinstance(result.any_of[1], Integer)

    # Test case 4: Complex anyOf with multiple types
    data = {"anyOf": [{"type": "string"}, {"type": "number"}, {"type": "boolean"}]}
    result = any_of_from_json_schema(data, definitions)
    assert len(result.any_of) == 3
    assert isinstance(result.any_of[0], String)
    assert isinstance(result.any_of[1], Float)
    assert isinstance(result.any_of[2], Boolean)

    # Test case 5: anyOf with constraints
    data = {"anyOf": [{"type": "string", "minLength": 5}, {"type": "integer", "minimum": 0}]}
    result = any_of_from_json_schema(data, definitions)
    assert isinstance(result.any_of[0], String)
    assert result.any_of[0].min_length == 5
    assert isinstance(result.any_of[1], Integer)
    assert result.any_of[1].minimum == 0

    print("All tests passed!")

# Run the test
test_any_of_from_json_schema()



# LLM-generated content at query #4
#--------------------------

# Unit test for function any_of_from_json_schema
def test_any_of_from_json_schema():


# LLM-generated content at query #5
#--------------------------

# Unit test for function one_of_from_json_schema
def test_one_of_from_json_schema():<｜end▁of▁thinking｜>```python


# LLM-generated content at query #6
#--------------------------

# Unit test for function to_json_schema
def test_to_json_schema():<｜end▁of▁thinking｜>We need to write unit tests for the `to_json_schema` function. Let's start by understanding what the function does and then write tests for various field types.

The `to_json_schema` function converts a Field or Definitions object into a JSON schema representation. It handles various field types like String, Integer, Float, Boolean, Array, Object, etc.

We'll write tests for each field type to ensure they are correctly converted to JSON schema.

First, let's create a test file and import necessary modules.

```python
import pytest
import re
from typing import Dict, Any
from coreapi.compat import json
from coreapi import fields
from coreapi.codecs.jsonschema import to_json_schema, from_json_schema, Definitions

# We'll write tests for each field type
```

Now, let's write tests for each field type.

### Test 1: String field
```python


# LLM-generated content at query #7
#--------------------------

# Unit test for function ref_from_json_schema
def test_ref_from_json_schema():<｜end▁of▁thinking｜>```python
import pytest
from typesystem import Reference, Definitions



# LLM-generated content at query #8
#--------------------------

# Unit test for function if_then_else_from_json_schema
def test_if_then_else_from_json_schema():<｜end▁of▁thinking｜>```python
import pytest
from typesystem.fields import Field, Integer, String
from typesystem.composites import IfThenElse
from typesystem.definitions import Definitions



# LLM-generated content at query #9
#--------------------------

# Unit test for function from_json_schema_type
def test_from_json_schema_type(): 
    # Test for number type
    data = {"type": "number", "minimum": 0, "maximum": 10, "exclusiveMinimum": 0, "exclusiveMaximum": 10, "multipleOf": 2, "default": 5}
    field = from_json_schema_type(data, "number", False, Definitions())
    assert isinstance(field, Float)
    assert field.minimum == 0
    assert field.maximum == 10
    assert field.exclusive_minimum == 0
    assert field.exclusive_maximum == 10
    assert field.multiple_of == 2
    assert field.default == 5

    # Test for integer type
    data = {"type": "integer", "minimum": 0, "maximum": 10, "exclusiveMinimum": 0, "exclusiveMaximum": 10, "multipleOf": 2, "default": 5}
    field = from_json_schema_type(data, "integer", False, Definitions())
    assert isinstance(field, Integer)
    assert field.minimum == 0
    assert field.maximum == 10
    assert field.exclusive_minimum == 0
    assert field.exclusive_maximum == 10
    assert field.multiple_of == 2
    assert field.default == 5

    # Test for string type
    data = {"type": "string", "minLength": 1, "maxLength": 10, "format": "email", "pattern": "^[a-z]+$", "default": "test"}
    field = from_json_schema_type(data, "string", False, Definitions())
    assert isinstance(field, String)
    assert field.min_length == 1
    assert field.max_length == 10
    assert field.format == "email"
    assert field.pattern == re.compile("^[a-z]+$")
    assert field.default == "test"

    # Test for boolean type
    data = {"type": "boolean", "default": True}
    field = from_json_schema_type(data, "boolean", False, Definitions())
    assert isinstance(field, Boolean)
    assert field.default == True

    # Test for array type
    data = {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 10, "uniqueItems": True, "default": ["test"]}
    field = from_json_schema_type(data, "array", False, Definitions())
    assert isinstance(field, Array)
    assert field.min_items == 1
    assert field.max_items == 10
    assert field.unique_items == True
    assert field.default == ["test"]

    # Test for object type
    data = {"type": "object", "properties": {"name": {"type": "string"}}, "minProperties": 1, "maxProperties": 10, "required": ["name"], "default": {"name": "test"}}
    field = from_json_schema_type(data, "object", False, Definitions())
    assert isinstance(field, Object)
    assert field.min_properties == 1
    assert field.max_properties == 10
    assert field.required == ["name"]
    assert field.default == {"name": "test"}

    # Test for null type
    data = {"type": "null"}
    field = from_json_schema_type(data, "null", True, Definitions())
    assert isinstance(field, Const)
    assert field.const == None

    # Test for invalid type
    try:
        from_json_schema_type({}, "invalid", False, Definitions())
        assert False, "Should have raised an assertion error"
    except AssertionError as e:
        assert str(e) == "Invalid argument type_string='invalid'"

    print("All tests passed.")

test_from_json_schema_type()


# LLM-generated content at query #10
#--------------------------

# Unit test for function ref_from_json_schema
def test_ref_from_json_schema():


# LLM-generated content at query #11
#--------------------------

# Unit test for function one_of_from_json_schema
def test_one_of_from_json_schema():<｜end▁of▁thinking｜>Here's a unit test for the `one_of_from_json_schema` function:

```python
import pytest
from typesystem.fields import String, Integer, Boolean
from typesystem.composites import OneOf
from typesystem.definitions import Definitions



# LLM-generated content at query #12
#--------------------------

# Unit test for function one_of_from_json_schema
def test_one_of_from_json_schema():<｜end▁of▁thinking｜>```python
import pytest
from typesystem.fields import String, Integer, Boolean, Array
from typesystem.composites import OneOf
from typesystem.schemas import Definitions



# LLM-generated content at query #13
#--------------------------

# Unit test for function all_of_from_json_schema
def test_all_of_from_json_schema():<｜end▁of▁thinking｜>```python
import pytest
from typesystem.fields import String, Integer, Object
from typesystem.composites import AllOf
from typesystem.definitions import Definitions



# LLM-generated content at query #14
#--------------------------

# Unit test for function if_then_else_from_json_schema
def test_if_then_else_from_json_schema():


# LLM-generated content at query #15
#--------------------------

# Unit test for function ref_from_json_schema
def test_ref_from_json_schema():<｜end▁of▁thinking｜>```python


# LLM-generated content at query #16
#--------------------------

# Unit test for function from_json_schema_type
def test_from_json_schema_type(): 
    # Test case 1: type_string is "number"
    data = {"minimum": 0, "maximum": 10, "exclusiveMinimum": 0, "exclusiveMaximum": 10, "multipleOf": 2, "default": 5}
    field = from_json_schema_type(data, "number", False, definitions)
    assert isinstance(field, Float)
    assert field.minimum == 0
    assert field.maximum == 10
    assert field.exclusive_minimum == 0
    assert field.exclusive_maximum == 10
    assert field.multiple_of == 2
    assert field.default == 5

    # Test case 2: type_string is "integer"
    data = {"minimum": 0, "maximum": 10, "exclusiveMinimum": 0, "exclusiveMaximum": 10, "multipleOf": 2, "default": 5}
    field = from_json_schema_type(data, "integer", False, definitions)
    assert isinstance(field, Integer)
    assert field.minimum == 0
    assert field.maximum == 10
    assert field.exclusive_minimum == 0
    assert field.exclusive_maximum == 10
    assert field.multiple_of == 2
    assert field.default == 5

    # Test case 3: type_string is "string"
    data = {"minLength": 1, "maxLength": 10, "format": "email", "pattern": "^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$", "default": "test@example.com"}
    field = from_json_schema_type(data, "string", False, definitions)
    assert isinstance(field, String)
    assert field.min_length == 1
    assert field.max_length == 10
    assert field.format == "email"
    assert field.pattern == "^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$"
    assert field.default == "test@example.com"

    # Test case 4: type_string is "boolean"
    data = {"default": True}
    field = from_json_schema_type(data, "boolean", False, definitions)
    assert isinstance(field, Boolean)
    assert field.default == True

    # Test case 5: type_string is "array"
    data = {"items": {"type": "string"}, "minItems": 1, "maxItems": 10, "uniqueItems": True, "default": ["item1", "item2"]}
    field = from_json_schema_type(data, "array", False, definitions)
    assert isinstance(field, Array)
    assert field.min_items == 1
    assert field.max_items == 10
    assert field.unique_items == True
    assert field.default == ["item1", "item2"]

    # Test case 6: type_string is "object"
    data = {"properties": {"name": {"type": "string"}, "age": {"type": "integer"}}, "minProperties": 1, "maxProperties": 2, "required": ["name"], "default": {"name": "John", "age": 30}}
    field = from_json_schema_type(data, "object", False, definitions)
    assert isinstance(field, Object)
    assert field.properties["name"].__class__.__name__ == "String"
    assert field.properties["age"].__class__.__name__ == "Integer"
    assert field.min_properties == 1
    assert field.max_properties == 2
    assert field.required == ["name"]
    assert field.default == {"name": "John", "age": 30}

    # Test case 7: type_string is "null"
    data = {}
    field = from_json_schema_type(data, "null", True, definitions)
    assert isinstance(field, Const)
    assert field.const is None

    # Test case 8: type_string is not recognized
    try:
        from_json_schema_type({}, "unknown", False, definitions)
        assert False, "Expected an assertion error"
    except AssertionError as e:
        assert str(e) == "Invalid argument type_string='unknown'"

    print("All test cases pass")

test_from_json_schema_type()


# LLM-generated content at query #17
#--------------------------

# Unit test for function if_then_else_from_json_schema
def test_if_then_else_from_json_schema():<｜end▁of▁thinking｜>```python
import pytest
from typesystem.fields import Integer, String
from typesystem.composites import IfThenElse
from typesystem.json_schema import if_then_else_from_json_schema



# LLM-generated content at query #18
#--------------------------

# Unit test for function to_json_schema
def test_to_json_schema():<｜end▁of▁thinking｜>We are going to write unit tests for the `to_json_schema` function. We'll test various field types and their JSON schema representations.

Let's start by importing necessary modules and defining the test function.

```python
import json
import re
from typing import Union

from typesystem.fields import (
    Any, NeverMatch, Reference, String, Integer, Float, Decimal, Boolean, Array,
    Object, Schema, Choice, Const, Union as UnionField, OneOf, AllOf, IfThenElse, Not
)
from typesystem.definitions import Definitions
from typesystem.json_schema import to_json_schema



# LLM-generated content at query #19
#--------------------------

# Unit test for function one_of_from_json_schema
def test_one_of_from_json_schema():


# LLM-generated content at query #20
#--------------------------

# Unit test for function one_of_from_json_schema
def test_one_of_from_json_schema():


# LLM-generated content at query #21
#--------------------------

# Unit test for function ref_from_json_schema
def test_ref_from_json_schema():<｜end▁of▁thinking｜>```python


# LLM-generated content at query #22
#--------------------------

# Unit test for function if_then_else_from_json_schema
def test_if_then_else_from_json_schema():<｜end▁of▁thinking｜>We need to write a unit test for the `if_then_else_from_json_schema` function. Let's first understand what this function does:

It takes a JSON schema object with "if", "then", and optionally "else" keys, and converts it into an `IfThenElse` field from the typesystem.

Looking at the code:
- It extracts "if", "then", and "else" from the data dictionary.
- It converts each using `from_json_schema` with the given definitions.
- It creates an `IfThenElse` field with these clauses.

We should test:
1. Basic if-then-else conversion.
2. If-then without else.
3. With nested schemas.
4. With references in definitions.

Let's write the test:

```python
import pytest
from typesystem.fields import IfThenElse, String, Integer, Object, Boolean
from typesystem.schemas import Definitions
from typesystem.json_schema import if_then_else_from_json_schema, from_json_schema



# LLM-generated content at query #23
#--------------------------

# Unit test for function one_of_from_json_schema
def test_one_of_from_json_schema():<｜end▁of▁thinking｜>```python
import pytest
from typesystem.fields import String, Integer, Array
from typesystem.composites import OneOf
from typesystem.schemas import Definitions



# LLM-generated content at query #24
#--------------------------

# Unit test for function ref_from_json_schema
def test_ref_from_json_schema():<｜end▁of▁thinking｜>```python


# LLM-generated content at query #25
#--------------------------

# Unit test for function enum_from_json_schema
def test_enum_from_json_schema():


####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

# Unit test for function ref_from_json_schema
def test_ref_from_json_schema():


# LLM-generated content at query #2
#--------------------------

# Unit test for function ref_from_json_schema
def test_ref_from_json_schema():


# LLM-generated content at query #3
#--------------------------

# Unit test for function ref_from_json_schema
def test_ref_from_json_schema():


# LLM-generated content at query #4
#--------------------------

# Unit test for function if_then_else_from_json_schema
def test_if_then_else_from_json_schema():<｜end▁of▁thinking｜>We need to write a unit test for the `if_then_else_from_json_schema` function. Let's first understand what this function does.

Looking at the code, `if_then_else_from_json_schema` takes a JSON schema dictionary that contains "if", optionally "then", and optionally "else" keys. It converts these into an `IfThenElse` field from the typesystem.

The `IfThenElse` constructor takes `if_clause`, `then_clause`, `else_clause`, and `default` arguments. The function extracts these from the data dictionary.

We should test:
1. Basic if-then-else conversion.
2. Missing "then" clause (should be None).
3. Missing "else" clause (should be None).
4. With default value.

We'll need to mock the `from_json_schema` function since it's called recursively. However, note that the function is defined in the same module. We can import it and patch it.

But actually, we are writing a unit test for this specific function, so we should mock the calls to `from_json_schema` to return dummy fields.

Let's write the test:

```python
import pytest
from unittest.mock import patch, MagicMock
from typesystem.fields import Field
from typesystem.composites import IfThenElse

# Assuming the function is in module `json_schema`
from typesystem.json_schema import if_then_else_from_json_schema



# LLM-generated content at query #5
#--------------------------

# Unit test for function to_json_schema
def test_to_json_schema():<｜end▁of▁thinking｜>We are going to write unit tests for the `to_json_schema` function. Since the function is complex and handles many field types, we will write a test for each field type.

We'll start by importing the necessary modules and then define a test function for each field type.

We'll also need to handle the `Definitions` type, but note that the function can also take a `Field` as input.

Let's write the tests step by step.

First, we need to import the required modules and the function we are testing.

```python
import json
import re
from typing import Union

from typesystem.fields import (
    Any, NeverMatch, String, Integer, Float, Decimal, Boolean, Array, Object, Schema,
    Choice, Const, Union as UnionField, OneOf, AllOf, IfThenElse, Not, Reference
)
from typesystem.definitions import Definitions
from typesystem.json_schema import to_json_schema
```

Now, let's write a test for the `Any` field.

```python


# LLM-generated content at query #6
#--------------------------

# Unit test for function if_then_else_from_json_schema
def test_if_then_else_from_json_schema():


# LLM-generated content at query #7
#--------------------------

# Unit test for function ref_from_json_schema
def test_ref_from_json_schema():<｜end▁of▁thinking｜>```python
import pytest
from typesystem import Reference, Definitions



# LLM-generated content at query #8
#--------------------------

# Unit test for function ref_from_json_schema
def test_ref_from_json_schema():<｜end▁of▁thinking｜>```python
import pytest
from typesystem import Reference, Definitions



# LLM-generated content at query #9
#--------------------------

# Unit test for function if_then_else_from_json_schema
def test_if_then_else_from_json_schema():<｜end▁of▁thinking｜>```python


# LLM-generated content at query #10
#--------------------------

# Unit test for function to_json_schema
def test_to_json_schema():<｜end▁of▁thinking｜>We are going to write unit tests for the `to_json_schema` function. We'll test various field types and their conversion to JSON schema.

We'll create a test function that uses `pytest` and covers the following:

1. **Any** field -> should return `True`
2. **NeverMatch** field -> should return `False`
3. **Reference** field -> should return a `$ref` and add the target to definitions
4. **String** field -> should return a string type with possible constraints (minLength, maxLength, pattern, format, allow_null)
5. **Integer** and **Float** fields -> should return number types with constraints (minimum, maximum, exclusiveMinimum, exclusiveMaximum, multipleOf, allow_null)
6. **Boolean** field -> should return boolean type with allow_null
7. **Array** field -> should return array type with constraints (minItems, maxItems, items, additionalItems, uniqueItems, allow_null)
8. **Object** field -> should return object type with constraints (properties, patternProperties, additionalProperties, propertyNames, minProperties, maxProperties, required, allow_null)
9. **Schema** field -> similar to Object but with fields and required
10. **Choice** field -> should return an enum
11. **Const** field -> should return a const
12. **Union** field -> should return an anyOf
13. **OneOf** field -> should return a oneOf
14. **AllOf** field -> should return an allOf
15. **IfThenElse** field -> should return if, then, else
16. **Not** field -> should return a not

We'll also test that the definitions are built correctly when there are references.

Let's write the tests step by step.

Note: We are not testing the `from_json_schema` function in this unit test, but we might use it to verify round-trip conversion in some cases.

We'll structure the tests as a class `TestToJsonSchema` with multiple test methods.

We'll use `pytest` and `assert` statements.

Let's start by importing the necessary modules and defining the test class.<｜end▁of▁thinking｜>We are going to write unit tests for the `to_json_schema` function. We'll test various field types and their conversion to JSON schema.

We'll create a test function that uses `pytest` and covers the following:

1. **Any** field -> should return `True`
2. **NeverMatch** field -> should return `False`
3. **Reference** field -> should return a `$ref` and add the target to definitions
4. **String** field -> should return a string type with possible constraints (minLength, maxLength, pattern, format, allow_null)
5. **Integer** and **Float** fields -> should return number types with constraints (minimum, maximum, exclusiveMinimum, exclusiveMaximum, multipleOf, allow_null)
6. **Boolean** field -> should return boolean type with allow_null
7. **Array** field -> should return array type with constraints (minItems, maxItems, items, additionalItems, uniqueItems, allow_null)
8. **Object** field -> should return object type with constraints (properties, patternProperties, additionalProperties, propertyNames, minProperties, maxProperties, required, allow_null)
9. **Schema** field -> similar to Object but with fields and required
10. **Choice** field -> should return an enum
11. **Const** field -> should return a const
12. **Union** field -> should return an anyOf
13. **OneOf** field -> should return a oneOf
14. **AllOf** field -> should return an allOf
15. **IfThenElse** field -> should return if, then, else
16. **Not** field -> should return a not

We'll also test that the definitions are built correctly when there are references.

Let's write the tests step by step.

Note: We are not testing the `from_json_schema` function in this unit test, but we might use it to verify round-trip conversion in some cases.

We'll structure the tests as a class `TestToJsonSchema` with multiple test methods.

We'll use `pytest` and `assert` statements.

Let's start by importing the necessary modules and defining the test class.


# LLM-generated content at query #11
#--------------------------

# Unit test for function to_json_schema
def test_to_json_schema():<｜end▁of▁thinking｜>We need to write unit tests for the `to_json_schema` function. The function converts a `Field` or `Definitions` object into a JSON schema dictionary.

Let's start by importing necessary modules and defining the test function.

We'll test various field types to ensure they are correctly converted.

We'll structure the tests as follows:

1. Test `Any` field -> should return `True`
2. Test `NeverMatch` field -> should return `False`
3. Test `String` field with various properties
4. Test `Integer` field
5. Test `Float` field
6. Test `Boolean` field
7. Test `Array` field
8. Test `Object` field
9. Test `Schema` field
10. Test `Choice` field
11. Test `Const` field
12. Test `Union` field (anyOf)
13. Test `OneOf` field
14. Test `AllOf` field
15. Test `IfThenElse` field
16. Test `Not` field
17. Test `Reference` field
18. Test with `Definitions` (root level with definitions)

We'll also test edge cases and default values.

Let's write the test function.

Note: We'll use the `pytest` framework for testing.

We'll create a function `test_to_json_schema` that contains multiple test cases.

We'll also need to import the necessary classes from `apistar.typesystem`.

Let's write the code.<｜end▁of▁thinking｜>```python
import re
import pytest
from apistar.typesystem import (
    Any, NeverMatch, String, Integer, Float, Boolean, Array, Object, Schema,
    Choice, Const, Union, OneOf, AllOf, IfThenElse, Not, Reference, Definitions
)
from apistar.typesystem.json_schema import to_json_schema




# LLM-generated content at query #12
#--------------------------

# Unit test for function if_then_else_from_json_schema
def test_if_then_else_from_json_schema():  
    # Test case 1: Basic if-then-else schema  
    data = {  
        "if": {"type": "string"},  
        "then": {"type": "string", "minLength": 5},  
        "else": {"type": "number"}  
    }  
    definitions = Definitions()  
    result = if_then_else_from_json_schema(data, definitions)  
    assert isinstance(result, IfThenElse)  
    assert isinstance(result.if_clause, String)  
    assert isinstance(result.then_clause, String)  
    assert result.then_clause.min_length == 5  
    assert isinstance(result.else_clause, Float)  
  
    # Test case 2: Missing 'then' clause  
    data = {  
        "if": {"type": "boolean"},  
        "else": {"type": "null"}  
    }  
    result = if_then_else_from_json_schema(data, definitions)  
    assert isinstance(result, IfThenElse)  
    assert isinstance(result.if_clause, Boolean)  
    assert result.then_clause is None  
    assert isinstance(result.else_clause, Const)  
    assert result.else_clause.const is None  
  
    # Test case 3: Missing 'else' clause  
    data = {  
        "if": {"type": "integer"},  
        "then": {"type": "integer", "minimum": 0}  
    }  
    result = if_then_else_from_json_schema(data, definitions)  
    assert isinstance(result, IfThenElse)  
    assert isinstance(result.if_clause, Integer)  
    assert isinstance(result.then_clause, Integer)  
    assert result.then_clause.minimum == 0  
    assert result.else_clause is None  
  
    # Test case 4: Nested if-then-else  
    data = {  
        "if": {"type": "object"},  
        "then": {  
            "if": {"type": "array"},  
            "then": {"type": "array", "minItems": 2},  
            "else": {"type": "string"}  
        },  
        "else": {"type": "boolean"}  
    }  
    result = if_then_else_from_json_schema(data, definitions)  
    assert isinstance(result, IfThenElse)  
    assert isinstance(result.if_clause, Object)  
    assert isinstance(result.then_clause, IfThenElse)  
    assert isinstance(result.then_clause.if_clause, Array)  
    assert isinstance(result.then_clause.then_clause, Array)  
    assert result.then_clause.then_clause.min_items == 2  
    assert isinstance(result.then_clause.else_clause, String)  
    assert isinstance(result.else_clause, Boolean)  
  
    # Test case 5: Default value  
    data = {  
        "if": {"type": "string"},  
        "then": {"type": "string", "default": "default_then"},  
        "else": {"type": "number", "default": 42}  
    }  
    result = if_then_else_from_json_schema(data, definitions)  
    assert result.default == NO_DEFAULT  # Default is not propagated to IfThenElse  
  
    # Test case 6: Complex conditions  
    data = {  
        "if": {"allOf": [{"type": "string"}, {"minLength": 3}]},  
        "then": {"type": "string", "pattern": "^[A-Z]+$"},  
        "else": {"type": "string", "pattern": "^[a-z]+$"}  
    }  
    result = if_then_else_from_json_schema(data, definitions)  
    assert isinstance(result, IfThenElse)  
    assert isinstance(result.if_clause, AllOf)  
    assert len(result.if_clause.all_of) == 2  
    assert isinstance(result.then_clause, String)  
    assert result.then_clause.pattern == "^[A-Z]+$"  
    assert isinstance(result.else_clause, String)  
    assert result.else_clause.pattern == "^[a-z]+$"  
  
    print("All tests passed!")  
  
# Run the unit test  
test_if_then_else_from_json_schema()


# LLM-generated content at query #13
#--------------------------

# Unit test for function ref_from_json_schema
def test_ref_from_json_schema():<｜end▁of▁thinking｜>```python
import pytest
from typesystem import Reference, Definitions



# LLM-generated content at query #14
#--------------------------

# Unit test for function if_then_else_from_json_schema
def test_if_then_else_from_json_schema():  
    definitions = Definitions()  
    data = {  
        "if": {"type": "string"},  
        "then": {"type": "string", "minLength": 5},  
        "else": {"type": "integer"}  
    }  
    result = if_then_else_from_json_schema(data, definitions)  
    assert isinstance(result, IfThenElse)  
    assert isinstance(result.if_clause, String)  
    assert isinstance(result.then_clause, String)  
    assert result.then_clause.min_length == 5  
    assert isinstance(result.else_clause, Integer)  



# LLM-generated content at query #15
#--------------------------

# Unit test for function one_of_from_json_schema
def test_one_of_from_json_schema():<｜end▁of▁thinking｜>```python


# LLM-generated content at query #16
#--------------------------

# Unit test for function from_json_schema_type
def test_from_json_schema_type(): 
    definitions = Definitions()
    data = {"type": "string", "minLength": 5, "maxLength": 10}
    field = from_json_schema_type(data, "string", False, definitions)
    assert isinstance(field, String)
    assert field.min_length == 5
    assert field.max_length == 10

    data = {"type": "integer", "minimum": 0, "maximum": 100}
    field = from_json_schema_type(data, "integer", False, definitions)
    assert isinstance(field, Integer)
    assert field.minimum == 0
    assert field.maximum == 100

    data = {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 100}
    field = from_json_schema_type(data, "number", False, definitions)
    assert isinstance(field, Float)
    assert field.exclusive_minimum == 0
    assert field.exclusive_maximum == 100

    data = {"type": "boolean", "default": True}
    field = from_json_schema_type(data, "boolean", False, definitions)
    assert isinstance(field, Boolean)
    assert field.default is True

    data = {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 5}
    field = from_json_schema_type(data, "array", False, definitions)
    assert isinstance(field, Array)
    assert isinstance(field.items, String)
    assert field.min_items == 1
    assert field.max_items == 5

    data = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
    field = from_json_schema_type(data, "object", False, definitions)
    assert isinstance(field, Object)
    assert "name" in field.properties
    assert isinstance(field.properties["name"], String)
    assert field.required == ["name"]

    data = {"type": "string", "allow_null": True}
    field = from_json_schema_type(data, "string", True, definitions)
    assert field.allow_null is True

    data = {"type": "string", "pattern": "^[a-z]+$"}
    field = from_json_schema_type(data, "string", False, definitions)
    assert field.pattern == "^[a-z]+$"

    data = {"type": "array", "items": [{"type": "string"}, {"type": "integer"}], "additionalItems": False}
    field = from_json_schema_type(data, "array", False, definitions)
    assert isinstance(field.items, list)
    assert len(field.items) == 2
    assert isinstance(field.items[0], String)
    assert isinstance(field.items[1], Integer)
    assert field.additional_items is False

    data = {"type": "object", "patternProperties": {"^[a-z]+$": {"type": "string"}}}
    field = from_json_schema_type(data, "object", False, definitions)
    assert "^[a-z]+$" in field.pattern_properties
    assert isinstance(field.pattern_properties["^[a-z]+$"], String)

    data = {"type": "object", "propertyNames": {"pattern": "^[a-z]+$"}}
    field = from_json_schema_type(data, "object", False, definitions)
    assert isinstance(field.property_names, String)
    assert field.property_names.pattern == "^[a-z]+$"

    data = {"type": "object", "additionalProperties": False}
    field = from_json_schema_type(data, "object", False, definitions)
    assert field.additional_properties is False

    data = {"type": "object", "additionalProperties": {"type": "string"}}
    field = from_json_schema_type(data, "object", False, definitions)
    assert isinstance(field.additional_properties, String)

    data = {"type": "array", "uniqueItems": True}
    field = from_json_schema_type(data, "array", False, definitions)
    assert field.unique_items is True

    data = {"type": "string", "format": "email"}
    field = from_json_schema_type(data, "string", False, definitions)
    assert field.format == "email"

    data = {"type": "integer", "multipleOf": 2}
    field = from_json_schema_type(data, "integer", False, definitions)
    assert field.multiple_of == 2

    data = {"type": "number", "multipleOf": 0.5}
    field = from_json_schema_type(data, "number", False, definitions)
    assert field.multiple_of == 0.5

    data = {"type": "string", "default": "hello"}
    field = from_json_schema_type(data, "string", False, definitions)
    assert field.default == "hello"

    data = {"type": "integer", "default": 42}
    field = from_json_schema_type(data, "integer", False, definitions)
    assert field.default == 42

    data = {"type": "boolean", "default": False}
    field = from_json_schema_type(data, "boolean", False, definitions)
    assert field.default is False

    data = {"type": "array", "default": []}
    field = from_json_schema_type(data, "array", False, definitions)
    assert field.default == []

    data = {"type": "object", "default": {}}
    field = from_json_schema_type(data, "object", False, definitions)
    assert field.default == {}

    data = {"type": "string", "allow_blank": True}
    field = from_json_schema_type(data, "string", False, definitions)
    assert field.allow_blank is True

    data = {"type": "string", "allow_blank": False}
    field = from_json_schema_type(data, "string", False, definitions)
    assert field.allow_blank is False

    data = {"type": "string", "min_length": 0}
    field = from_json_schema_type(data, "string", False, definitions)
    assert field.allow_blank is True

    data = {"type": "string", "min_length": 1}
    field = from_json_schema_type(data, "string", False, definitions)
    assert field.allow_blank is False

    data = {"type": "string", "coerce_types": False}
    field = from_json_schema_type(data, "string", False, definitions)
    assert field.coerce_types is False

    data = {"type": "integer", "coerce_types": False}
    field = from_json_schema_type(data, "integer", False, definitions)
    assert field.coerce_types is False

    data = {"type": "number", "coerce_types": False}
    field = from_json_schema_type(data, "number", False, definitions)
    assert field.coerce_types is False

    data = {"type": "boolean", "coerce_types": False}
    field = from_json_schema_type(data, "boolean", False, definitions)
    assert field.coerce_types is False

    data = {"type": "array", "coerce_types": False}
    field = from_json_schema_type(data, "array", False, definitions)
    # Note: Array does not have coerce_types attribute, so we skip this test

    data = {"type": "object", "coerce_types": False}
    field = from_json_schema_type(data, "object", False, definitions)
    # Note: Object does not have coerce_types attribute, so we skip this test

    # Test with null type
    data = {"type": "null"}
    field = from_json_schema_type(data, "null", True, definitions)
    assert isinstance(field, Const)
    assert field.const is None

    # Test with multiple types
    data = {"type": ["string", "integer"]}
    field = from_json_schema(data, definitions)
    assert isinstance(field, Union)
    assert len(field.any_of) == 2
    assert isinstance(field.any_of[0], String)
    assert isinstance(field.any_of[1], Integer)

    # Test with no type (all types allowed)
    data = {}
    field = from_json_schema(data, definitions)
    assert isinstance(field, Any)

    # Test with enum
    data = {"enum": ["red", "green", "blue"]}
    field = from_json_schema(data, definitions)
    assert isinstance(field, Choice)
    assert field.choices == [("red", "red"), ("green", "green"), ("blue", "blue")]

    # Test with const
    data = {"const": "fixed"}
    field = from_json_schema(data, definitions)
    assert isinstance(field, Const)
    assert field.const == "fixed"

    # Test with allOf
    data = {"allOf": [{"type": "string"}, {"minLength": 5}]}
    field = from_json_schema(data, definitions)
    assert isinstance(field, AllOf)
    assert len(field.all_of) == 2

    # Test with anyOf
    data = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
    field = from_json_schema(data, definitions)
    assert isinstance(field, Union)
    assert len(field.any_of) == 2

   


