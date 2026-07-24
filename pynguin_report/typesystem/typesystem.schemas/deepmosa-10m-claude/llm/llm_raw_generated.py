####################################################################
#    TEST GENERATION BEGINS (DEEPMOSA + claude-haiku-4-5 t=0.8)    #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_validate_with_none_and_allow_null():
    from typesystem import Schema, String
    schema = Schema(fields={"name": String()}, allow_null=True)
    result = schema.validate(None)
    assert result is None


def test_validate_with_none_and_disallow_null():
    from typesystem import Schema, String, ValidationError
    schema = Schema(fields={"name": String()})
    try:
        schema.validate(None)
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert len(e.messages()) > 0


def test_validate_with_non_dict_value():
    from typesystem import Schema, String, ValidationError
    schema = Schema(fields={"name": String()})
    try:
        schema.validate("not a dict")
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert len(e.messages()) > 0


def test_validate_with_non_string_keys():
    from typesystem import Schema, String, ValidationError
    schema = Schema(fields={"name": String()})
    try:
        schema.validate({1: "value"})
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) > 0


def test_validate_with_missing_required_field():
    from typesystem import Schema, String, ValidationError
    schema = Schema(fields={"name": String()})
    try:
        schema.validate({})
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) > 0


def test_validate_with_valid_data():
    from typesystem import Schema, String
    schema = Schema(fields={"name": String()})
    result = schema.validate({"name": "John"})
    assert result == {"name": "John"}


def test_validate_with_extra_fields():
    from typesystem import Schema, String
    schema = Schema(fields={"name": String()})
    result = schema.validate({"name": "John", "extra": "field"})
    assert result == {"name": "John"}


def test_validate_with_default_value():
    from typesystem import Schema, String
    schema = Schema(fields={"name": String(default="Unknown")})
    result = schema.validate({})
    assert result == {"name": "Unknown"}


def test_validate_with_read_only_field():
    from typesystem import Schema, String
    schema = Schema(fields={"name": String(), "id": String(read_only=True)})
    result = schema.validate({"name": "John", "id": "123"})
    assert result == {"name": "John"}


def test_validate_with_nested_validation_error():
    from typesystem import Schema, String, Integer, ValidationError
    schema = Schema(fields={"name": String(), "age": Integer()})
    try:
        schema.validate({"name": "John", "age": "not an integer"})
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) > 0


def test_validate_with_multiple_errors():
    from typesystem import Schema, String, ValidationError
    schema = Schema(fields={"name": String(), "email": String()})
    try:
        schema.validate({1: "value"})
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) > 0


def test_validate_with_mapping_type():
    from typesystem import Schema, String
    from collections import OrderedDict
    schema = Schema(fields={"name": String()})
    mapping = OrderedDict([("name", "John")])
    result = schema.validate(mapping)
    assert result == {"name": "John"}


def test_validate_with_optional_field_missing():
    from typesystem import Schema, String
    schema = Schema(fields={"name": String(), "nickname": String(allow_null=True)})
    result = schema.validate({"name": "John"})
    assert "nickname" not in result


# LLM-generated content at query #2
#--------------------------

```python
def test_serialize_with_dict():
    from typesystem import Schema, String, Integer
    
    schema = Schema(fields={"name": String(), "age": Integer()})
    obj = {"name": "John", "age": 30}
    result = schema.serialize(obj)
    assert result == {"name": "John", "age": 30}


def test_serialize_with_none():
    from typesystem import Schema, String
    
    schema = Schema(fields={"name": String()})
    result = schema.serialize(None)
    assert result is None


def test_serialize_with_object():
    from typesystem import Schema, String, Integer
    
    class Person:
        def __init__(self, name, age):
            self.name = name
            self.age = age
    
    schema = Schema(fields={"name": String(), "age": Integer()})
    obj = Person("Jane", 25)
    result = schema.serialize(obj)
    assert result == {"name": "Jane", "age": 25}


def test_serialize_missing_keys():
    from typesystem import Schema, String, Integer
    
    schema = Schema(fields={"name": String(), "age": Integer(), "email": String()})
    obj = {"name": "Bob", "age": 35}
    result = schema.serialize(obj)
    assert result == {"name": "Bob", "age": 35}


def test_serialize_with_missing_attributes():
    from typesystem import Schema, String, Integer
    
    class Person:
        def __init__(self, name):
            self.name = name
    
    schema = Schema(fields={"name": String(), "age": Integer()})
    obj = Person("Alice")
    result = schema.serialize(obj)
    assert result == {"name": "Alice"}


def test_serialize_empty_schema():
    from typesystem import Schema
    
    schema = Schema(fields={})
    obj = {"name": "Test"}
    result = schema.serialize(obj)
    assert result == {}


def test_serialize_with_nested_schema():
    from typesystem import Schema, String
    
    address_schema = Schema(fields={"street": String(), "city": String()})
    person_schema = Schema(fields={"name": String(), "address": address_schema})
    
    obj = {"name": "John", "address": {"street": "123 Main St", "city": "NYC"}}
    result = person_schema.serialize(obj)
    assert result == {"name": "John", "address": {"street": "123 Main St", "city": "NYC"}}


# LLM-generated content at query #3
#--------------------------

```python
def test_serialize_exception_handler_attribute_error():
    class MockField(Field):
        def serialize(self, value):
            return value
    
    class MockObject:
        def __init__(self):
            self.existing_attr = "value"
    
    schema = Schema(fields={"existing_attr": MockField(), "missing_attr": MockField()})
    obj = MockObject()
    result = schema.serialize(obj)
    
    assert result == {"existing_attr": "value"}


# LLM-generated content at query #4
#--------------------------

```python
def test_serialize_with_dict():
    from typesystem import Schema, String, Integer

    schema = Schema(fields={
        "name": String(),
        "age": Integer(),
    })
    obj = {"name": "John", "age": 30}
    result = schema.serialize(obj)
    assert result == {"name": "John", "age": 30}


def test_serialize_with_none():
    from typesystem import Schema, String

    schema = Schema(fields={"name": String()})
    result = schema.serialize(None)
    assert result is None


def test_serialize_with_object():
    from typesystem import Schema, String, Integer

    class Person:
        def __init__(self, name, age):
            self.name = name
            self.age = age

    schema = Schema(fields={
        "name": String(),
        "age": Integer(),
    })
    obj = Person("Jane", 25)
    result = schema.serialize(obj)
    assert result == {"name": "Jane", "age": 25}


def test_serialize_missing_keys():
    from typesystem import Schema, String, Integer

    schema = Schema(fields={
        "name": String(),
        "age": Integer(),
        "email": String(),
    })
    obj = {"name": "Bob", "age": 35}
    result = schema.serialize(obj)
    assert result == {"name": "Bob", "age": 35}


def test_serialize_missing_attributes():
    from typesystem import Schema, String, Integer

    class Person:
        def __init__(self, name):
            self.name = name

    schema = Schema(fields={
        "name": String(),
        "age": Integer(),
    })
    obj = Person("Alice")
    result = schema.serialize(obj)
    assert result == {"name": "Alice"}


def test_serialize_empty_schema():
    from typesystem import Schema

    schema = Schema(fields={})
    obj = {"name": "Tom", "age": 40}
    result = schema.serialize(obj)
    assert result == {}


def test_serialize_with_nested_fields():
    from typesystem import Schema, String, Integer

    inner_schema = Schema(fields={
        "street": String(),
        "city": String(),
    })
    outer_schema = Schema(fields={
        "name": String(),
        "address": inner_schema,
    })
    obj = {
        "name": "Charlie",
        "address": {"street": "Main St", "city": "NYC"},
    }
    result = outer_schema.serialize(obj)
    assert result["name"] == "Charlie"
    assert result["address"] == {"street": "Main St", "city": "NYC"}


# LLM-generated content at query #5
#--------------------------

```python
def test_serialize_attribute_error_exception_handling():
    class MockField(Field):
        def serialize(self, value):
            return value
    
    class MockObject:
        def __init__(self):
            self.existing_attr = "value"
    
    schema = Schema(fields={"existing_attr": MockField(), "missing_attr": MockField()})
    obj = MockObject()
    
    result = schema.serialize(obj)
    
    assert result == {"existing_attr": "value"}
    assert "missing_attr" not in result


# LLM-generated content at query #6
#--------------------------

```python
def test_reference_validate_with_none_and_allow_null_true():
    definitions = {"User": None}
    reference = Reference(to="User", definitions=definitions, allow_null=True)
    result = reference.validate(None)
    assert result is None


def test_reference_validate_with_none_and_allow_null_false():
    definitions = {"User": None}
    reference = Reference(to="User", definitions=definitions, allow_null=False)
    try:
        reference.validate(None)
        assert False, "Expected validation_error to be raised"
    except Exception as e:
        assert "May not be null" in str(e)


def test_reference_validate_with_valid_value():
    mock_target = type('MockTarget', (), {'validate': lambda self, v: v * 2})()
    definitions = {"User": mock_target}
    reference = Reference(to="User", definitions=definitions, allow_null=False)
    result = reference.validate(5)
    assert result == 10


def test_reference_validate_with_valid_value_and_target_validation():
    mock_target = type('MockTarget', (), {'validate': lambda self, v: {"id": v}})()
    definitions = {"User": mock_target}
    reference = Reference(to="User", definitions=definitions, allow_null=False)
    result = reference.validate(123)
    assert result == {"id": 123}


def test_reference_validate_delegates_to_target():
    mock_target = type('MockTarget', (), {'validate': lambda self, v: v.upper()})()
    definitions = {"Status": mock_target}
    reference = Reference(to="Status", definitions=definitions, allow_null=False)
    result = reference.validate("active")
    assert result == "ACTIVE"


# LLM-generated content at query #7
#--------------------------

```python
def test_serialize_predicate_line_13_evaluates_to_false():
    class MockField(Field):
        def serialize(self, value):
            return value
    
    schema = Schema(fields={"name": MockField()})
    
    class TestObject:
        def __init__(self):
            self.name = "test_value"
    
    obj = TestObject()
    result = schema.serialize(obj)
    
    assert result == {"name": "test_value"}
    assert "name" in result


# LLM-generated content at query #8
#--------------------------

```python
def test_serialize_with_dict():
    from typesystem import Field, Schema
    
    field1 = Field()
    field2 = Field()
    schema = Schema(fields={"key1": field1, "key2": field2})
    obj = {"key1": "value1", "key2": "value2"}
    
    result = schema.serialize(obj)
    
    assert result == {"key1": "value1", "key2": "value2"}


def test_serialize_with_none():
    from typesystem import Field, Schema
    
    field1 = Field()
    schema = Schema(fields={"key1": field1})
    
    result = schema.serialize(None)
    
    assert result is None


def test_serialize_with_object():
    from typesystem import Field, Schema
    
    field1 = Field()
    field2 = Field()
    schema = Schema(fields={"key1": field1, "key2": field2})
    
    class TestObj:
        def __init__(self):
            self.key1 = "value1"
            self.key2 = "value2"
    
    obj = TestObj()
    result = schema.serialize(obj)
    
    assert result == {"key1": "value1", "key2": "value2"}


def test_serialize_missing_keys():
    from typesystem import Field, Schema
    
    field1 = Field()
    field2 = Field()
    schema = Schema(fields={"key1": field1, "key2": field2})
    obj = {"key1": "value1"}
    
    result = schema.serialize(obj)
    
    assert result == {"key1": "value1"}


def test_serialize_missing_attributes():
    from typesystem import Field, Schema
    
    field1 = Field()
    field2 = Field()
    schema = Schema(fields={"key1": field1, "key2": field2})
    
    class TestObj:
        def __init__(self):
            self.key1 = "value1"
    
    obj = TestObj()
    result = schema.serialize(obj)
    
    assert result == {"key1": "value1"}


def test_serialize_empty_dict():
    from typesystem import Field, Schema
    
    field1 = Field()
    schema = Schema(fields={"key1": field1})
    obj = {}
    
    result = schema.serialize(obj)
    
    assert result == {}


def test_serialize_with_custom_field_serialization():
    from typesystem import Field, Schema
    
    class CustomField(Field):
        def serialize(self, obj):
            return str(obj).upper()
    
    field1 = CustomField()
    schema = Schema(fields={"key1": field1})
    obj = {"key1": "hello"}
    
    result = schema.serialize(obj)
    
    assert result == {"key1": "HELLO"}


# LLM-generated content at query #9
#--------------------------

```python
def test_serialize_with_dict():
    from typesystem import Schema, String, Integer
    
    schema = Schema(fields={
        "name": String(),
        "age": Integer()
    })
    
    obj = {"name": "John", "age": 30}
    result = schema.serialize(obj)
    
    assert result == {"name": "John", "age": 30}


def test_serialize_with_none():
    from typesystem import Schema, String
    
    schema = Schema(fields={"name": String()})
    result = schema.serialize(None)
    
    assert result is None


def test_serialize_with_object():
    from typesystem import Schema, String, Integer
    
    class Person:
        def __init__(self, name, age):
            self.name = name
            self.age = age
    
    schema = Schema(fields={
        "name": String(),
        "age": Integer()
    })
    
    obj = Person("Jane", 25)
    result = schema.serialize(obj)
    
    assert result == {"name": "Jane", "age": 25}


def test_serialize_missing_key():
    from typesystem import Schema, String, Integer
    
    schema = Schema(fields={
        "name": String(),
        "age": Integer(),
        "email": String()
    })
    
    obj = {"name": "Bob", "age": 35}
    result = schema.serialize(obj)
    
    assert result == {"name": "Bob", "age": 35}
    assert "email" not in result


def test_serialize_missing_attribute():
    from typesystem import Schema, String, Integer
    
    class Person:
        def __init__(self, name):
            self.name = name
    
    schema = Schema(fields={
        "name": String(),
        "age": Integer()
    })
    
    obj = Person("Alice")
    result = schema.serialize(obj)
    
    assert result == {"name": "Alice"}
    assert "age" not in result


def test_serialize_empty_dict():
    from typesystem import Schema, String
    
    schema = Schema(fields={"name": String()})
    obj = {}
    result = schema.serialize(obj)
    
    assert result == {}


def test_serialize_with_nested_field_serialization():
    from typesystem import Schema, String, Integer
    
    schema = Schema(fields={
        "name": String(),
        "count": Integer()
    })
    
    obj = {"name": "test", "count": 42}
    result = schema.serialize(obj)
    
    assert result["name"] == "test"
    assert result["count"] == 42


# LLM-generated content at query #10
#--------------------------

```python
def test_serialize_attribute_error_exception_handling():
    class MockField(Field):
        def serialize(self, value):
            return value
    
    class TestObject:
        def __init__(self):
            self.existing_attr = "value"
    
    schema = Schema(fields={"existing_attr": MockField(), "missing_attr": MockField()})
    obj = TestObject()
    
    result = schema.serialize(obj)
    
    assert result == {"existing_attr": "value"}
    assert "missing_attr" not in result


# LLM-generated content at query #11
#--------------------------

```python
def test_serialize_missing_attribute_continues():
    class MockField(Field):
        def serialize(self, value):
            return value
    
    class MockObject:
        def __init__(self):
            self.field1 = "value1"
    
    schema = Schema(fields={"field1": MockField(), "field2": MockField()})
    obj = MockObject()
    
    result = schema.serialize(obj)
    
    assert result == {"field1": "value1"}
    assert "field2" not in result


####################################################################
#    TEST GENERATION BEGINS (DEEPMOSA + claude-haiku-4-5 t=0.8)    #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_validate_with_none_and_allow_null_true():
    from typesystem import Schema, Field, String
    schema = Schema(fields={"name": String()}, allow_null=True)
    result = schema.validate(None)
    assert result is None


def test_validate_with_none_and_allow_null_false():
    from typesystem import Schema, Field, String, ValidationError
    schema = Schema(fields={"name": String()})
    try:
        schema.validate(None)
        assert False
    except ValidationError:
        assert True


def test_validate_with_non_dict_value():
    from typesystem import Schema, String, ValidationError
    schema = Schema(fields={"name": String()})
    try:
        schema.validate("not a dict")
        assert False
    except ValidationError:
        assert True


def test_validate_with_non_string_keys():
    from typesystem import Schema, String, ValidationError
    schema = Schema(fields={"name": String()})
    try:
        schema.validate({1: "value"})
        assert False
    except ValidationError:
        assert True


def test_validate_with_missing_required_field():
    from typesystem import Schema, String, ValidationError
    schema = Schema(fields={"name": String()})
    try:
        schema.validate({})
        assert False
    except ValidationError:
        assert True


def test_validate_with_valid_data():
    from typesystem import Schema, String
    schema = Schema(fields={"name": String()})
    result = schema.validate({"name": "John"})
    assert result == {"name": "John"}


def test_validate_with_optional_field_missing():
    from typesystem import Schema, String
    schema = Schema(fields={"name": String(allow_null=True)})
    result = schema.validate({})
    assert result == {}


def test_validate_with_default_value():
    from typesystem import Schema, String
    schema = Schema(fields={"name": String(default="DefaultName")})
    result = schema.validate({})
    assert result == {"name": "DefaultName"}


def test_validate_with_read_only_field():
    from typesystem import Schema, String
    schema = Schema(fields={"name": String(), "id": String(read_only=True)})
    result = schema.validate({"name": "John", "id": "123"})
    assert result == {"name": "John"}


def test_validate_with_extra_fields():
    from typesystem import Schema, String
    schema = Schema(fields={"name": String()})
    result = schema.validate({"name": "John", "extra": "field"})
    assert result == {"name": "John"}


def test_validate_with_nested_schema():
    from typesystem import Schema, String
    nested_schema = Schema(fields={"age": String()})
    schema = Schema(fields={"name": String(), "details": nested_schema})
    result = schema.validate({"name": "John", "details": {"age": "30"}})
    assert result == {"name": "John", "details": {"age": "30"}}


def test_validate_with_mapping_object():
    from typesystem import Schema, String
    from collections import UserDict
    schema = Schema(fields={"name": String()})
    mapping = UserDict({"name": "John"})
    result = schema.validate(mapping)
    assert result == {"name": "John"}


def test_validate_with_invalid_nested_field():
    from typesystem import Schema, String, Integer, ValidationError
    nested_schema = Schema(fields={"age": Integer()})
    schema = Schema(fields={"name": String(), "details": nested_schema})
    try:
        schema.validate({"name": "John", "details": {"age": "not_an_int"}})
        assert False
    except ValidationError:
        assert True


def test_validate_multiple_required_fields():
    from typesystem import Schema, String
    schema = Schema(fields={"name": String(), "email": String()})
    result = schema.validate({"name": "John", "email": "john@example.com"})
    assert result == {"name": "John", "email": "john@example.com"}


def test_validate_with_empty_fields():
    from typesystem import Schema
    schema = Schema(fields={})
    result = schema.validate({})
    assert result == {}


# LLM-generated content at query #2
#--------------------------

```python
def test_validate_with_none_and_allow_null():
    from typesystem import Schema, String
    schema = Schema(fields={"name": String()}, allow_null=True)
    result = schema.validate(None)
    assert result is None


def test_validate_with_none_and_disallow_null():
    from typesystem import Schema, String, ValidationError
    schema = Schema(fields={"name": String()})
    try:
        schema.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError:
        pass


def test_validate_with_non_dict_value():
    from typesystem import Schema, String, ValidationError
    schema = Schema(fields={"name": String()})
    try:
        schema.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError:
        pass


def test_validate_with_non_string_keys():
    from typesystem import Schema, String, ValidationError
    schema = Schema(fields={"name": String()})
    try:
        schema.validate({1: "value"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert any(msg.code == "invalid_key" for msg in e.messages())


def test_validate_with_missing_required_field():
    from typesystem import Schema, String, ValidationError
    schema = Schema(fields={"name": String()})
    try:
        schema.validate({})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert any(msg.code == "required" for msg in e.messages())


def test_validate_with_valid_data():
    from typesystem import Schema, String
    schema = Schema(fields={"name": String()})
    result = schema.validate({"name": "John"})
    assert result == {"name": "John"}


def test_validate_with_read_only_field():
    from typesystem import Schema, String
    schema = Schema(fields={"name": String(), "id": String(read_only=True)})
    result = schema.validate({"name": "John", "id": "123"})
    assert result == {"name": "John"}
    assert "id" not in result


def test_validate_with_field_default_value():
    from typesystem import Schema, String
    schema = Schema(fields={"name": String(), "status": String(default="active")})
    result = schema.validate({"name": "John"})
    assert result == {"name": "John", "status": "active"}


def test_validate_with_invalid_field_value():
    from typesystem import Schema, Integer, ValidationError
    schema = Schema(fields={"age": Integer()})
    try:
        schema.validate({"age": "not an integer"})
        assert False, "Expected ValidationError"
    except ValidationError:
        pass


def test_validate_with_mapping_type():
    from typesystem import Schema, String
    from collections import OrderedDict
    schema = Schema(fields={"name": String()})
    result = schema.validate(OrderedDict([("name", "John")]))
    assert result == {"name": "John"}


def test_validate_with_multiple_errors():
    from typesystem import Schema, String, ValidationError
    schema = Schema(fields={"name": String(), "email": String()})
    try:
        schema.validate({})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) >= 2


def test_validate_with_extra_fields():
    from typesystem import Schema, String
    schema = Schema(fields={"name": String()})
    result = schema.validate({"name": "John", "extra": "field"})
    assert result == {"name": "John"}
    assert "extra" not in result


# LLM-generated content at query #3
#--------------------------

```python
def test_validate_with_none_and_allow_null_true():
    definitions = {"User": None}
    reference = Reference(to="User", definitions=definitions, allow_null=True)
    result = reference.validate(None)
    assert result is None


def test_validate_with_none_and_allow_null_false():
    definitions = {"User": None}
    reference = Reference(to="User", definitions=definitions, allow_null=False)
    try:
        reference.validate(None)
        assert False, "Expected validation_error to be raised"
    except Exception as e:
        assert "May not be null" in str(e)


def test_validate_with_valid_value():
    class MockField:
        def validate(self, value):
            return value.upper()
    
    definitions = {"User": MockField()}
    reference = Reference(to="User", definitions=definitions)
    result = reference.validate("test")
    assert result == "TEST"


def test_validate_delegates_to_target():
    class MockField:
        def validate(self, value):
            if value < 0:
                raise ValueError("Negative value")
            return value * 2
    
    definitions = {"Number": MockField()}
    reference = Reference(to="Number", definitions=definitions)
    result = reference.validate(5)
    assert result == 10


def test_validate_with_complex_object():
    class MockField:
        def validate(self, value):
            return {"id": value.get("id"), "name": value.get("name")}
    
    definitions = {"User": MockField()}
    reference = Reference(to="User", definitions=definitions)
    test_obj = {"id": 1, "name": "John"}
    result = reference.validate(test_obj)
    assert result == {"id": 1, "name": "John"}


# LLM-generated content at query #4
#--------------------------

```python
def test_schema_serialize_with_dict():
    from typesystem import Schema, String, Integer
    
    fields = {
        "name": String(),
        "age": Integer(),
    }
    schema = Schema(fields=fields)
    
    obj = {"name": "John", "age": 30}
    result = schema.serialize(obj)
    
    assert result == {"name": "John", "age": 30}


def test_schema_serialize_with_none():
    from typesystem import Schema, String
    
    fields = {"name": String()}
    schema = Schema(fields=fields)
    
    result = schema.serialize(None)
    
    assert result is None


def test_schema_serialize_with_object():
    from typesystem import Schema, String, Integer
    
    class Person:
        def __init__(self, name, age):
            self.name = name
            self.age = age
    
    fields = {
        "name": String(),
        "age": Integer(),
    }
    schema = Schema(fields=fields)
    
    obj = Person("Alice", 25)
    result = schema.serialize(obj)
    
    assert result == {"name": "Alice", "age": 25}


def test_schema_serialize_missing_keys():
    from typesystem import Schema, String, Integer
    
    fields = {
        "name": String(),
        "age": Integer(),
        "email": String(),
    }
    schema = Schema(fields=fields)
    
    obj = {"name": "Bob", "age": 35}
    result = schema.serialize(obj)
    
    assert result == {"name": "Bob", "age": 35}


def test_schema_serialize_missing_attributes():
    from typesystem import Schema, String, Integer
    
    class Person:
        def __init__(self, name):
            self.name = name
    
    fields = {
        "name": String(),
        "age": Integer(),
    }
    schema = Schema(fields=fields)
    
    obj = Person("Charlie")
    result = schema.serialize(obj)
    
    assert result == {"name": "Charlie"}


def test_schema_serialize_empty_dict():
    from typesystem import Schema, String
    
    fields = {"name": String()}
    schema = Schema(fields=fields)
    
    obj = {}
    result = schema.serialize(obj)
    
    assert result == {}


def test_schema_serialize_with_nested_schema():
    from typesystem import Schema, String, Integer
    
    address_schema = Schema(fields={"street": String(), "city": String()})
    fields = {
        "name": String(),
        "address": address_schema,
    }
    schema = Schema(fields=fields)
    
    obj = {"name": "David", "address": {"street": "123 Main St", "city": "NYC"}}
    result = schema.serialize(obj)
    
    assert result == {"name": "David", "address": {"street": "123 Main St", "city": "NYC"}}


# LLM-generated content at query #5
#--------------------------

```python
def test_serialize_attribute_error_handling():
    class MockField(Field):
        def serialize(self, value):
            return value
    
    class MockObject:
        def __init__(self):
            self.existing_field = "value"
    
    schema = Schema(fields={"existing_field": MockField(), "missing_field": MockField()})
    obj = MockObject()
    
    result = schema.serialize(obj)
    
    assert result == {"existing_field": "value"}
    assert "missing_field" not in result


# LLM-generated content at query #6
#--------------------------

```python
def test_serialize_predicate_line_13_evaluates_to_false():
    class MockField(Field):
        def serialize(self, value):
            return value
        
        def has_default(self):
            return False
    
    class MockObject:
        def __init__(self):
            self.field1 = "value1"
            self.field2 = "value2"
    
    fields = {
        "field1": MockField(),
        "field2": MockField(),
    }
    
    schema = Schema(fields=fields)
    obj = MockObject()
    result = schema.serialize(obj)
    
    assert result == {"field1": "value1", "field2": "value2"}
    assert "field1" in result
    assert "field2" in result


