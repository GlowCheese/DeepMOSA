####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_Schema_serialize():
    from typesystem.fields import String, Integer, Boolean
    
    # Test 1: Serialize None returns None
    schema = Schema(fields={})
    assert schema.serialize(None) is None
    
    # Test 2: Serialize dict with simple fields
    schema = Schema(fields={
        "name": String(),
        "age": Integer(),
        "active": Boolean()
    })
    
    obj_dict = {"name": "John", "age": 30, "active": True}
    result = schema.serialize(obj_dict)
    assert result == {"name": "John", "age": 30, "active": True}
    
    # Test 3: Serialize object with attributes
    class Person:
        def __init__(self, name, age, active):
            self.name = name
            self.age = age
            self.active = active
    
    person = Person("Jane", 25, False)
    result = schema.serialize(person)
    assert result == {"name": "Jane", "age": 25, "active": False}
    
    # Test 4: Missing keys in dict are skipped
    obj_dict = {"name": "Bob"}
    result = schema.serialize(obj_dict)
    assert result == {"name": "Bob"}
    
    # Test 5: Missing attributes in object are skipped
    class PartialPerson:
        def __init__(self, name):
            self.name = name
    
    partial_person = PartialPerson("Alice")
    result = schema.serialize(partial_person)
    assert result == {"name": "Alice"}
    
    # Test 6: Nested schema serialization
    address_schema = Schema(fields={
        "street": String(),
        "city": String()
    })
    
    user_schema = Schema(fields={
        "name": String(),
        "address": address_schema
    })
    
    user_dict = {
        "name": "Test",
        "address": {"street": "Main St", "city": "Metropolis"}
    }
    result = user_schema.serialize(user_dict)
    assert result == {"name": "Test", "address": {"street": "Main St", "city": "Metropolis"}}
    
    # Test 7: Field serialization with custom serialize method
    class CustomField(Field):
        def serialize(self, obj):
            return f"custom_{obj}"
    
    schema = Schema(fields={"data": CustomField()})
    result = schema.serialize({"data": "value"})
    assert result == {"data": "custom_value"}
    
    # Test 8: Empty dict with no fields
    schema = Schema(fields={})
    result = schema.serialize({})
    assert result == {}
    
    # Test 9: Mixed dict and object-like behavior
    class MixedObject:
        def __init__(self):
            self.name = "Mixed"
    
    mixed = MixedObject()
    schema = Schema(fields={"name": String(), "extra": String()})
    result = schema.serialize(mixed)
    assert result == {"name": "Mixed"}


# LLM-generated content at query #2
#--------------------------

```python
def test_Schema():
    # Test basic initialization with fields
    field1 = Field()
    field2 = Field(allow_null=True)
    fields = {"name": field1, "age": field2}
    schema = Schema(fields)
    
    assert schema.fields == fields
    assert schema.required == ["name"]
    assert not hasattr(schema, "allow_null") or schema.allow_null == False
    
    # Test with read_only field
    field3 = Field(read_only=True)
    fields2 = {"id": field3, "title": Field()}
    schema2 = Schema(fields2)
    
    assert schema2.required == ["title"]
    
    # Test with field that has default value
    field4 = Field(default="default_value")
    fields3 = {"name": field4, "email": Field()}
    schema3 = Schema(fields3)
    
    assert schema3.required == ["email"]
    
    # Test with all optional fields
    field5 = Field(default="default1")
    field6 = Field(allow_null=True)
    fields4 = {"field1": field5, "field2": field6}
    schema4 = Schema(fields4)
    
    assert schema4.required == []
    
    # Test with empty fields dict
    schema5 = Schema({})
    
    assert schema5.fields == {}
    assert schema5.required == []
    
    # Test with additional kwargs passed to parent Field class
    schema6 = Schema(fields, allow_null=True, description="Test schema")
    
    assert schema6.allow_null == True
    assert schema6.description == "Test schema"


# LLM-generated content at query #3
#--------------------------

```python
def test_Schema_validate():
    # Test basic validation with simple fields
    from typesystem.fields import String, Integer
    
    schema = Schema(fields={
        "name": String(max_length=10),
        "age": Integer(minimum=0, maximum=150)
    })
    
    # Test valid input
    result = schema.validate({"name": "John", "age": 30})
    assert result == {"name": "John", "age": 30}
    
    # Test null value when not allowed
    try:
        schema.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "null"
    
    # Test null value when allowed
    schema_with_null = Schema(
        fields={"name": String()},
        allow_null=True
    )
    result = schema_with_null.validate(None)
    assert result is None
    
    # Test non-dict input
    try:
        schema.validate("not a dict")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.messages()[0].code == "type"
    
    # Test invalid key type
    try:
        schema.validate({1: "value", "name": "John"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert any(msg.code == "invalid_key" for msg in e.messages())
    
    # Test missing required field
    try:
        schema.validate({"name": "John"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert any(msg.code == "required" for msg in e.messages())
        assert any("age" in str(msg.index) for msg in e.messages())
    
    # Test field validation errors
    try:
        schema.validate({"name": "VeryLongNameExceedsLimit", "age": 30})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert any(msg.code == "max_length" for msg in e.messages())
    
    # Test read_only field
    schema_with_readonly = Schema(fields={
        "id": Integer(read_only=True),
        "name": String()
    })
    result = schema_with_readonly.validate({"name": "Alice"})
    assert "id" not in result
    assert result["name"] == "Alice"
    
    # Test field with default value
    schema_with_default = Schema(fields={
        "name": String(),
        "active": Field(default=True)
    })
    result = schema_with_default.validate({"name": "Bob"})
    assert result["name"] == "Bob"
    assert result["active"] is True
    
    # Test multiple validation errors
    try:
        schema.validate({})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) >= 2
        codes = [msg.code for msg in messages]
        assert "required" in codes
    
    # Test nested validation
    nested_schema = Schema(fields={
        "address": Schema(fields={
            "street": String(),
            "city": String()
        })
    })
    
    result = nested_schema.validate({
        "address": {"street": "123 Main", "city": "Boston"}
    })
    assert result["address"]["street"] == "123 Main"
    assert result["address"]["city"] == "Boston"
    
    # Test nested validation error
    try:
        nested_schema.validate({
            "address": {"street": 123}
        })
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert any("city" in str(msg.index) for msg in e.messages())


# LLM-generated content at query #4
#--------------------------

```python
def test_Schema():
    # Test basic initialization with fields
    field1 = Field()
    field2 = Field()
    fields = {"name": field1, "age": field2}
    schema = Schema(fields)
    
    assert schema.fields == fields
    assert schema.required == ["name", "age"]
    assert schema.allow_null == False
    
    # Test with read_only field
    read_only_field = Field(read_only=True)
    fields_with_readonly = {"name": field1, "id": read_only_field}
    schema2 = Schema(fields_with_readonly)
    
    assert "id" not in schema2.required
    
    # Test with field that has default value
    field_with_default = Field(default="default_value")
    fields_with_default = {"name": field1, "optional": field_with_default}
    schema3 = Schema(fields_with_default)
    
    assert "optional" not in schema3.required
    
    # Test with mixed fields
    mixed_fields = {
        "required_field": field1,
        "read_only_field": read_only_field,
        "default_field": field_with_default
    }
    schema4 = Schema(mixed_fields)
    
    assert schema4.required == ["required_field"]
    
    # Test with allow_null parameter
    schema5 = Schema(fields, allow_null=True)
    assert schema5.allow_null == True
    
    # Test with empty fields
    schema6 = Schema({})
    assert schema6.fields == {}
    assert schema6.required == []
    
    # Test that parent class initialization works
    schema7 = Schema(fields, description="Test schema")
    assert hasattr(schema7, "description")


# LLM-generated content at query #5
#--------------------------

```python
def test_Schema():
    # Test basic initialization with fields
    field1 = Field()
    field2 = Field()
    fields = {"name": field1, "age": field2}
    schema = Schema(fields)
    
    assert schema.fields == fields
    assert schema.required == ["name", "age"]
    
    # Test with read_only field
    read_only_field = Field(read_only=True)
    fields_with_readonly = {"name": field1, "id": read_only_field}
    schema2 = Schema(fields_with_readonly)
    
    assert "id" not in schema2.required
    
    # Test with field that has default value
    field_with_default = Field(default="default_value")
    fields_with_default = {"name": field1, "optional": field_with_default}
    schema3 = Schema(fields_with_default)
    
    assert "optional" not in schema3.required
    
    # Test with mixed fields
    mixed_fields = {
        "required_field": field1,
        "read_only_field": Field(read_only=True),
        "default_field": Field(default=0)
    }
    schema4 = Schema(mixed_fields)
    
    assert schema4.required == ["required_field"]
    
    # Test inheritance from Field
    assert isinstance(schema, Field)
    
    # Test with additional kwargs passed to parent
    schema_with_kwargs = Schema(fields, allow_null=True, description="Test schema")
    assert schema_with_kwargs.allow_null == True
    assert schema_with_kwargs.description == "Test schema"


# LLM-generated content at query #6
#--------------------------

```python
def test_Reference_validate():
    # Test 1: Valid value with non-null target
    definitions = Definitions()
    field = Field()
    definitions["target_field"] = field
    reference = Reference(to="target_field", definitions=definitions)
    
    # Mock the target field's validate method
    mock_value = {"key": "value"}
    field.validate = lambda x: x if x == mock_value else None
    
    result = reference.validate(mock_value)
    assert result == mock_value

    # Test 2: Null value with allow_null=True
    reference_with_null = Reference(to="target_field", definitions=definitions, allow_null=True)
    result = reference_with_null.validate(None)
    assert result is None

    # Test 3: Null value with allow_null=False (default)
    reference_without_null = Reference(to="target_field", definitions=definitions)
    try:
        reference_without_null.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "null"

    # Test 4: Target field raises ValidationError
    definitions2 = Definitions()
    field2 = Field()
    definitions2["error_field"] = field2
    reference2 = Reference(to="error_field", definitions=definitions2)
    
    error_message = Message(text="Invalid", code="invalid")
    field2.validate = lambda x: (_ for _ in ()).throw(ValidationError(messages=[error_message]))
    
    try:
        reference2.validate("some_value")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0] == error_message

    # Test 5: Non-existent target in definitions
    empty_definitions = Definitions()
    reference3 = Reference(to="missing", definitions=empty_definitions)
    
    try:
        reference3.validate("value")
        assert False, "Should have raised KeyError"
    except KeyError:
        pass


# LLM-generated content at query #7
#--------------------------

```python
def test_Schema_validate():
    # Test basic validation with valid data
    field = Field()
    schema = Schema(fields={"name": field})
    result = schema.validate({"name": "test"})
    assert result == {"name": "test"}

    # Test null value when allow_null is True
    schema = Schema(fields={"name": field}, allow_null=True)
    result = schema.validate(None)
    assert result is None

    # Test null value when allow_null is False
    schema = Schema(fields={"name": field}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "null"

    # Test non-dict value
    schema = Schema(fields={"name": field})
    try:
        schema.validate("not a dict")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "type"

    # Test invalid key type (non-string key)
    schema = Schema(fields={"name": field})
    try:
        schema.validate({123: "value"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "invalid_key"
        assert e.messages[0].index == [123]

    # Test missing required field
    schema = Schema(fields={"name": field})
    try:
        schema.validate({})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "required"
        assert e.messages[0].index == ["name"]

    # Test field with default value
    field_with_default = Field(default="default_value")
    schema = Schema(fields={"name": field_with_default})
    result = schema.validate({})
    assert result == {"name": "default_value"}

    # Test read_only field
    read_only_field = Field(read_only=True)
    schema = Schema(fields={"name": read_only_field})
    result = schema.validate({"name": "ignored"})
    assert result == {}

    # Test nested validation error
    nested_field = Field()
    schema = Schema(fields={"nested": nested_field})
    try:
        schema.validate({"nested": None})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].index == ["nested"]

    # Test multiple validation errors
    field1 = Field()
    field2 = Field()
    schema = Schema(fields={"field1": field1, "field2": field2})
    try:
        schema.validate({123: "bad key", "extra": "unexpected"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        codes = {msg.code for msg in e.messages}
        assert "invalid_key" in codes
        assert "required" in codes

    # Test with allow_null=True and valid data
    schema = Schema(fields={"name": field}, allow_null=True)
    result = schema.validate({"name": "test"})
    assert result == {"name": "test"}

    # Test with Mapping type (not just dict)
    from collections.abc import Mapping

    class TestMapping(Mapping):
        def __init__(self, data):
            self._data = data
        
        def __getitem__(self, key):
            return self._data[key]
        
        def __iter__(self):
            return iter(self._data)
        
        def __len__(self):
            return len(self._data)

    schema = Schema(fields={"name": field})
    mapping = TestMapping({"name": "test"})
    result = schema.validate(mapping)
    assert result == {"name": "test"}


# LLM-generated content at query #8
#--------------------------

```python
def test_Schema_validate():
    # Test 1: Null value with allow_null=True
    field = Field(allow_null=True)
    schema = Schema(fields={}, allow_null=True)
    assert schema.validate(None) is None

    # Test 2: Null value with allow_null=False
    schema = Schema(fields={}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "null"

    # Test 3: Non-dict value
    schema = Schema(fields={})
    try:
        schema.validate("not a dict")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "type"

    # Test 4: Invalid key type (non-string)
    schema = Schema(fields={})
    try:
        schema.validate({1: "value"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "invalid_key"
        assert e.messages()[0].index == [1]

    # Test 5: Missing required field
    field = Field()
    schema = Schema(fields={"name": field})
    try:
        schema.validate({})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "required"
        assert e.messages()[0].index == ["name"]

    # Test 6: Field with read_only should be ignored during validation
    read_only_field = Field(read_only=True)
    schema = Schema(fields={"id": read_only_field})
    result = schema.validate({"id": 123})
    assert "id" not in result

    # Test 7: Field with default value when missing
    field_with_default = Field(default="default_value")
    schema = Schema(fields={"optional": field_with_default})
    result = schema.validate({})
    assert result["optional"] == "default_value"

    # Test 8: Valid field validation
    string_field = Field()
    schema = Schema(fields={"name": string_field})
    result = schema.validate({"name": "John"})
    assert result["name"] == "John"

    # Test 9: Nested field validation error
    nested_field = Field()
    nested_field.validate = lambda x: (_ for _ in ()).throw(
        ValidationError(messages=[Message(text="Invalid", code="invalid", index=[])])
    )
    schema = Schema(fields={"nested": nested_field})
    try:
        schema.validate({"nested": "value"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "invalid"

    # Test 10: Multiple validation errors
    required_field = Field()
    schema = Schema(fields={"field1": required_field, "field2": required_field})
    try:
        schema.validate({2: "invalid key"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 3  # invalid_key + 2 required fields
        codes = [msg.code for msg in messages]
        assert "invalid_key" in codes
        assert "required" in codes

    # Test 11: Field with allow_null in nested context
    nullable_field = Field(allow_null=True)
    schema = Schema(fields={"data": nullable_field})
    result = schema.validate({"data": None})
    assert result["data"] is None

    # Test 12: Complex nested structure
    inner_schema = Schema(fields={"age": Field()})
    outer_schema = Schema(fields={"person": inner_schema})
    result = outer_schema.validate({"person": {"age": 30}})
    assert result["person"]["age"] == 30

    # Test 13: Field validation returns error with prefix
    error_field = Field()
    error_field.validate_or_error = lambda x: (
        None,
        ValidationError(messages=[Message(text="Error", code="error", index=[])]),
    )
    schema = Schema(fields={"test": error_field})
    try:
        schema.validate({"test": "value"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "error"


# LLM-generated content at query #9
#--------------------------

```python
def test_Reference_validate():
    # Test 1: Valid value passes through to target validation
    definitions = Definitions()
    target_field = Field()
    target_field.validate = lambda x: x * 2
    definitions["target"] = target_field
    
    ref = Reference(to="target", definitions=definitions)
    assert ref.validate(5) == 10
    
    # Test 2: None value with allow_null=True
    ref = Reference(to="target", definitions=definitions, allow_null=True)
    assert ref.validate(None) is None
    
    # Test 3: None value with allow_null=False raises error
    ref = Reference(to="target", definitions=definitions, allow_null=False)
    try:
        ref.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "null"
    
    # Test 4: Target validation errors are propagated
    definitions2 = Definitions()
    target_field2 = Field()
    target_field2.validate = lambda x: (_ for _ in ()).throw(
        ValidationError(messages=[Message(text="Target error", code="target_error")])
    )
    definitions2["target2"] = target_field2
    
    ref2 = Reference(to="target2", definitions=definitions2)
    try:
        ref2.validate("bad_value")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "target_error"
    
    # Test 5: Non-existent target raises KeyError
    definitions3 = Definitions()
    ref3 = Reference(to="missing", definitions=definitions3)
    try:
        ref3.validate("value")
        assert False, "Should have raised KeyError"
    except KeyError:
        pass


# LLM-generated content at query #10
#--------------------------

```python
def test_Reference_validate():
    # Test 1: Valid value with non-null target
    definitions = Definitions()
    field = Field()
    definitions["target_field"] = field
    reference = Reference(to="target_field", definitions=definitions)
    
    # Mock target field's validate method
    mock_value = {"key": "value"}
    field.validate = lambda x: x if x == mock_value else None
    
    result = reference.validate(mock_value)
    assert result == mock_value
    
    # Test 2: Null value with allow_null=True
    reference_with_null = Reference(to="target_field", definitions=definitions, allow_null=True)
    result = reference_with_null.validate(None)
    assert result is None
    
    # Test 3: Null value with allow_null=False (default)
    reference_without_null = Reference(to="target_field", definitions=definitions)
    try:
        reference_without_null.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "null"
    
    # Test 4: Valid value with nested validation
    nested_field = Field()
    nested_field.validate = lambda x: x.upper() if isinstance(x, str) else x
    definitions["nested"] = nested_field
    nested_reference = Reference(to="nested", definitions=definitions)
    
    result = nested_reference.validate("test")
    assert result == "TEST"
    
    # Test 5: Target field raises ValidationError
    error_field = Field()
    error_field.validate = lambda x: (_ for _ in ()).throw(
        ValidationError(messages=[Message(text="Invalid", code="invalid")])
    )
    definitions["error_field"] = error_field
    error_reference = Reference(to="error_field", definitions=definitions)
    
    try:
        error_reference.validate("bad_value")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "invalid"
    
    # Test 6: Reference to non-existent definition
    missing_reference = Reference(to="missing", definitions=definitions)
    try:
        missing_reference.validate("value")
        assert False, "Should have raised KeyError when accessing target"
    except KeyError:
        pass


# LLM-generated content at query #11
#--------------------------

```python
def test_Schema_validate():
    # Test 1: Null value with allow_null=True
    field = Field(allow_null=True)
    schema = Schema(fields={"name": field})
    assert schema.validate(None) is None

    # Test 2: Null value with allow_null=False
    field = Field(allow_null=False)
    schema = Schema(fields={"name": field})
    try:
        schema.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "null"

    # Test 3: Non-dict value
    field = Field()
    schema = Schema(fields={"name": field})
    try:
        schema.validate("not a dict")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "type"

    # Test 4: Invalid non-string key
    field = Field()
    schema = Schema(fields={"name": field})
    try:
        schema.validate({123: "value"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "invalid_key"
        assert e.messages()[0].index == [123]

    # Test 5: Missing required field
    field = Field()
    schema = Schema(fields={"name": field})
    try:
        schema.validate({})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "required"
        assert e.messages()[0].index == ["name"]

    # Test 6: Field with default value when missing
    field = Field(default="default_name")
    schema = Schema(fields={"name": field})
    result = schema.validate({})
    assert result == {"name": "default_name"}

    # Test 7: Read-only field should be ignored during validation
    field = Field(read_only=True)
    schema = Schema(fields={"name": field})
    result = schema.validate({"name": "test"})
    assert result == {}

    # Test 8: Valid field validation
    field = Field()
    schema = Schema(fields={"name": field})
    result = schema.validate({"name": "John"})
    assert result == {"name": "John"}

    # Test 9: Invalid field value
    field = Field()
    schema = Schema(fields={"age": field})
    try:
        schema.validate({"age": "not a number"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert "age" in str(e.messages()[0])

    # Test 10: Multiple errors
    field1 = Field()
    field2 = Field()
    schema = Schema(fields={"field1": field1, "field2": field2})
    try:
        schema.validate({123: "invalid", "extra": "field"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) >= 3
        error_codes = [msg.code for msg in messages]
        assert "invalid_key" in error_codes
        assert "required" in error_codes

    # Test 11: Nested validation with valid data
    nested_field = Field()
    nested_schema = Schema(fields={"nested": nested_field})
    schema = Schema(fields={"data": nested_schema})
    result = schema.validate({"data": {"nested": "value"}})
    assert result == {"data": {"nested": "value"}}

    # Test 12: Complex scenario with multiple field types
    field1 = Field(default="default1")
    field2 = Field(read_only=True)
    field3 = Field()
    schema = Schema(fields={"f1": field1, "f2": field2, "f3": field3})
    
    result = schema.validate({"f3": "value3"})
    assert result == {"f1": "default1", "f3": "value3"}


# LLM-generated content at query #12
#--------------------------

```python
def test_Reference_validate():
    # Test 1: Valid value with non-null target
    definitions = Definitions()
    field = Field()
    definitions["target_field"] = field
    reference = Reference(to="target_field", definitions=definitions)
    
    # Mock the target field's validate method
    mock_value = {"key": "value"}
    field.validate = lambda x: x if x == mock_value else None
    result = reference.validate(mock_value)
    assert result == mock_value

    # Test 2: Null value with allow_null=True
    reference = Reference(to="target_field", definitions=definitions, allow_null=True)
    result = reference.validate(None)
    assert result is None

    # Test 3: Null value with allow_null=False (default)
    reference = Reference(to="target_field", definitions=definitions)
    try:
        reference.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test 4: Value passes through target validation
    definitions = Definitions()
    target_schema = Schema(fields={"name": Field()})
    definitions["person"] = target_schema
    reference = Reference(to="person", definitions=definitions)
    
    valid_data = {"name": "John"}
    result = reference.validate(valid_data)
    assert result == valid_data

    # Test 5: Target validation raises error
    definitions = Definitions()
    target_field = Field(allow_null=False)
    definitions["strict_field"] = target_field
    reference = Reference(to="strict_field", definitions=definitions)
    
    try:
        reference.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError:
        pass

    # Test 6: Target is another Reference (chaining)
    definitions = Definitions()
    inner_field = Field()
    definitions["inner"] = inner_field
    middle_reference = Reference(to="inner", definitions=definitions)
    definitions["middle"] = middle_reference
    outer_reference = Reference(to="middle", definitions=definitions)
    
    test_value = "test"
    inner_field.validate = lambda x: x if x == test_value else None
    result = outer_reference.validate(test_value)
    assert result == test_value

    # Test 7: Target not in definitions (should raise KeyError on access)
    definitions = Definitions()
    reference = Reference(to="missing", definitions=definitions)
    try:
        reference.validate("any")
        assert False, "Should have raised KeyError"
    except KeyError:
        pass


# LLM-generated content at query #13
#--------------------------

```python
def test_Schema_validate():
    # Test 1: Null value with allow_null=True
    field = Field(allow_null=True)
    schema = Schema(fields={}, allow_null=True)
    assert schema.validate(None) is None

    # Test 2: Null value with allow_null=False
    schema = Schema(fields={}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "null"

    # Test 3: Non-dict value
    schema = Schema(fields={})
    try:
        schema.validate("not a dict")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "type"

    # Test 4: Invalid non-string keys
    schema = Schema(fields={})
    try:
        schema.validate({1: "value", "valid": "value"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "invalid_key"
        assert e.messages()[0].index == [1]

    # Test 5: Missing required field
    field = Field()
    schema = Schema(fields={"required_field": field})
    try:
        schema.validate({})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "required"
        assert e.messages()[0].index == ["required_field"]

    # Test 6: Read-only field should be ignored
    read_only_field = Field(read_only=True)
    schema = Schema(fields={"read_only": read_only_field})
    result = schema.validate({"other": "value"})
    assert "read_only" not in result

    # Test 7: Field with default value
    field_with_default = Field(default="default_value")
    schema = Schema(fields={"with_default": field_with_default})
    result = schema.validate({})
    assert result["with_default"] == "default_value"

    # Test 8: Valid field validation
    field = Field()
    schema = Schema(fields={"name": field})
    result = schema.validate({"name": "John"})
    assert result["name"] == "John"

    # Test 9: Field validation error
    field = Field()
    schema = Schema(fields={"age": field})
    try:
        schema.validate({"age": None})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert "age" in str(e.messages()[0])

    # Test 10: Multiple validation errors
    required_field = Field()
    schema = Schema(fields={"required": required_field})
    try:
        schema.validate({1: "invalid", "other": "value"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 2
        codes = {msg.code for msg in messages}
        assert "invalid_key" in codes
        assert "required" in codes

    # Test 11: Complex nested validation
    nested_field = Field()
    nested_schema = Schema(fields={"nested": nested_field})
    schema = Schema(fields={"parent": nested_schema})
    result = schema.validate({"parent": {"nested": "value"}})
    assert result["parent"]["nested"] == "value"

    # Test 12: Empty schema with valid input
    schema = Schema(fields={})
    result = schema.validate({"extra": "ignored"})
    assert result == {}

    # Test 13: Field with allow_null in nested context
    nullable_field = Field(allow_null=True)
    schema = Schema(fields={"nullable": nullable_field})
    result = schema.validate({"nullable": None})
    assert result["nullable"] is None


# LLM-generated content at query #14
#--------------------------

```python
def test_Reference_validate():
    from typesystem.fields import Field
    from typesystem.base import ValidationError
    
    # Test 1: Valid value with non-null target
    definitions = Definitions()
    field = Field()
    field.validate = lambda x: x * 2
    definitions["target"] = field
    ref = Reference(to="target", definitions=definitions)
    assert ref.validate(5) == 10
    
    # Test 2: Null value with allow_null=True
    ref = Reference(to="target", definitions=definitions, allow_null=True)
    assert ref.validate(None) is None
    
    # Test 3: Null value with allow_null=False (default)
    ref = Reference(to="target", definitions=definitions)
    try:
        ref.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.messages()[0].code == "null"
    
    # Test 4: Target validation error propagation
    definitions = Definitions()
    field = Field()
    field.validate = lambda x: (_ for _ in ()).throw(ValidationError(messages=[]))
    definitions["target"] = field
    ref = Reference(to="target", definitions=definitions)
    try:
        ref.validate("invalid")
        assert False, "Should have raised ValidationError"
    except ValidationError:
        pass
    
    # Test 5: Non-null value with target returning same value
    definitions = Definitions()
    field = Field()
    field.validate = lambda x: x
    definitions["target"] = field
    ref = Reference(to="target", definitions=definitions)
    assert ref.validate("test") == "test"
    
    # Test 6: Missing target in definitions
    definitions = Definitions()
    ref = Reference(to="missing", definitions=definitions)
    try:
        ref.validate("value")
        assert False, "Should have raised KeyError"
    except KeyError:
        pass


# LLM-generated content at query #15
#--------------------------

```python
def test_Schema_validate():
    # Test basic validation with valid data
    field1 = Field()
    field2 = Field()
    schema = Schema(fields={"name": field1, "age": field2})
    value = {"name": "John", "age": 30}
    result = schema.validate(value)
    assert result == {"name": "John", "age": 30}

    # Test null value with allow_null=False (default)
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "null"

    # Test null value with allow_null=True
    schema = Schema(fields={"name": Field()}, allow_null=True)
    result = schema.validate(None)
    assert result is None

    # Test non-dict value
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate("not a dict")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "type"

    # Test invalid key type (non-string)
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate({1: "value"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "invalid_key"
        assert e.messages()[0].index == [1]

    # Test required field missing
    field = Field()
    schema = Schema(fields={"required_field": field})
    try:
        schema.validate({})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "required"
        assert e.messages()[0].index == ["required_field"]

    # Test read_only field is ignored during validation
    read_only_field = Field(read_only=True)
    regular_field = Field()
    schema = Schema(fields={"read_only": read_only_field, "regular": regular_field})
    value = {"regular": "value"}
    result = schema.validate(value)
    assert result == {"regular": "value"}

    # Test field with default value when missing
    field_with_default = Field(default="default_value")
    schema = Schema(fields={"field": field_with_default})
    result = schema.validate({})
    assert result == {"field": "default_value"}

    # Test nested validation errors
    nested_field = Field()
    nested_field.validate = lambda x: (_ for _ in ()).throw(
        ValidationError(messages=[Message(text="Nested error", code="nested", index=[])])
    )
    schema = Schema(fields={"nested": nested_field})
    try:
        schema.validate({"nested": "value"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "nested"
        assert e.messages()[0].index == ["nested"]

    # Test multiple validation errors
    field1 = Field()
    field2 = Field()
    schema = Schema(fields={"field1": field1, "field2": field2})
    try:
        schema.validate({})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 2
        codes = {msg.code for msg in e.messages()}
        assert codes == {"required"}

    # Test field validation with valid nested data
    field = Field()
    field.validate = lambda x: x.upper() if isinstance(x, str) else x
    schema = Schema(fields={"name": field})
    result = schema.validate({"name": "john"})
    assert result == {"name": "JOHN"}

    # Test with Mapping type (not just dict)
    from collections.abc import Mapping

    class TestMapping(Mapping):
        def __init__(self, data):
            self._data = data
        
        def __getitem__(self, key):
            return self._data[key]
        
        def __iter__(self):
            return iter(self._data)
        
        def __len__(self):
            return len(self._data)

    field = Field()
    schema = Schema(fields={"key": field})
    mapping = TestMapping({"key": "value"})
    result = schema.validate(mapping)
    assert result == {"key": "value"}


# LLM-generated content at query #16
#--------------------------

```python
def test_Reference_validate():
    # Test 1: Valid value with non-null target
    definitions = Definitions()
    field = Field()
    definitions["target_field"] = field
    reference = Reference(to="target_field", definitions=definitions)
    
    # Mock target field's validate method
    mock_value = {"key": "value"}
    field.validate = lambda x: x if x == mock_value else None
    result = reference.validate(mock_value)
    assert result == mock_value

    # Test 2: Null value with allow_null=True
    reference_with_null = Reference(to="target_field", definitions=definitions, allow_null=True)
    result = reference_with_null.validate(None)
    assert result is None

    # Test 3: Null value with allow_null=False (default)
    reference_without_null = Reference(to="target_field", definitions=definitions)
    try:
        reference_without_null.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "null"

    # Test 4: Target field raises ValidationError
    definitions2 = Definitions()
    field2 = Field()
    definitions2["error_field"] = field2
    reference2 = Reference(to="error_field", definitions=definitions2)
    
    error_message = Message(text="Invalid", code="custom", index=[])
    field2.validate = lambda x: (_ for _ in ()).throw(
        ValidationError(messages=[error_message])
    )
    
    try:
        reference2.validate("any_value")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "custom"

    # Test 5: Non-existent target in definitions
    empty_definitions = Definitions()
    reference3 = Reference(to="missing", definitions=empty_definitions)
    try:
        reference3.validate("value")
        assert False, "Should have raised KeyError"
    except KeyError:
        pass

    # Test 6: Valid value with nested structure
    definitions3 = Definitions()
    schema_field = Schema(fields={"nested": Field()})
    definitions3["schema_target"] = schema_field
    reference4 = Reference(to="schema_target", definitions=definitions3)
    
    test_data = {"nested": "test"}
    schema_field.validate = lambda x: x if x == test_data else None
    result = reference4.validate(test_data)
    assert result == test_data


# LLM-generated content at query #17
#--------------------------

```python
def test_Reference_validate():
    from typesystem.fields import Field
    from typesystem.base import ValidationError
    
    # Test 1: Valid value with non-null target
    definitions = Definitions()
    field = Field()
    field.validate = lambda x: x * 2
    definitions["target"] = field
    ref = Reference(to="target", definitions=definitions)
    assert ref.validate(5) == 10
    
    # Test 2: Null value with allow_null=True
    ref_allow_null = Reference(to="target", definitions=definitions, allow_null=True)
    assert ref_allow_null.validate(None) is None
    
    # Test 3: Null value with allow_null=False (default)
    ref_no_null = Reference(to="target", definitions=definitions)
    try:
        ref_no_null.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.messages()[0].code == "null"
    
    # Test 4: Target validation error propagation
    error_field = Field()
    error_field.validate = lambda x: (_ for _ in ()).throw(
        ValidationError(messages=[Message(text="Invalid", code="invalid")])
    )
    definitions["error_target"] = error_field
    ref_error = Reference(to="error_target", definitions=definitions)
    try:
        ref_error.validate("bad_value")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.messages()[0].code == "invalid"
    
    # Test 5: Non-existent target (should raise KeyError on access)
    ref_missing = Reference(to="missing", definitions=definitions)
    try:
        ref_missing.validate(1)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass


# LLM-generated content at query #18
#--------------------------

```python
def test_Schema_validate():
    # Test basic validation with valid data
    field1 = Field()
    field2 = Field()
    schema = Schema(fields={"name": field1, "age": field2})
    
    value = {"name": "John", "age": 30}
    result = schema.validate(value)
    assert result == {"name": "John", "age": 30}
    
    # Test null value when allow_null is False (default)
    schema = Schema(fields={"name": field1})
    try:
        schema.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "null"
    
    # Test null value when allow_null is True
    schema = Schema(fields={"name": field1}, allow_null=True)
    result = schema.validate(None)
    assert result is None
    
    # Test non-dict value
    schema = Schema(fields={"name": field1})
    try:
        schema.validate("not a dict")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "type"
    
    # Test invalid key type (non-string key)
    schema = Schema(fields={"name": field1})
    try:
        schema.validate({1: "value"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "invalid_key"
        assert e.messages()[0].index == [1]
    
    # Test missing required field
    field_required = Field()
    schema = Schema(fields={"name": field_required})
    try:
        schema.validate({})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "required"
        assert e.messages()[0].index == ["name"]
    
    # Test read_only field is ignored during validation
    field_readonly = Field(read_only=True)
    field_normal = Field()
    schema = Schema(fields={"id": field_readonly, "name": field_normal})
    value = {"name": "John"}
    result = schema.validate(value)
    assert result == {"name": "John"}
    
    # Test field with default value when missing
    field_with_default = Field(default="default_value")
    schema = Schema(fields={"name": field_with_default})
    value = {}
    result = schema.validate(value)
    assert result == {"name": "default_value"}
    
    # Test field validation error
    field_error = Field()
    # Mock the validate_or_error to return an error
    def mock_validate_or_error(val):
        return None, ValidationError(messages=[Message(text="Invalid", code="invalid", index=[])])
    field_error.validate_or_error = mock_validate_or_error
    
    schema = Schema(fields={"name": field_error})
    try:
        schema.validate({"name": "value"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "invalid"
    
    # Test multiple validation errors
    field1_error = Field()
    field2_error = Field()
    
    def mock_validate_or_error1(val):
        return None, ValidationError(messages=[Message(text="Error1", code="error1", index=[])])
    def mock_validate_or_error2(val):
        return None, ValidationError(messages=[Message(text="Error2", code="error2", index=[])])
    
    field1_error.validate_or_error = mock_validate_or_error1
    field2_error.validate_or_error = mock_validate_or_error2
    
    schema = Schema(fields={"field1": field1_error, "field2": field2_error})
    try:
        schema.validate({"field1": "val1", "field2": "val2"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 2
        codes = {msg.code for msg in e.messages()}
        assert codes == {"error1", "error2"}
    
    # Test combined errors (invalid key + missing required + field error)
    field_error = Field()
    field_error.validate_or_error = mock_validate_or_error1
    
    schema = Schema(fields={"valid_field": field_error})
    try:
        schema.validate({1: "invalid", "other": "value"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        # Should have invalid_key error and required error
        assert len(e.messages()) >= 2
    
    # Test nested validation with correct data
    nested_field = Field()
    schema = Schema(fields={"nested": nested_field})
    value = {"nested": "value"}
    result = schema.validate(value)
    assert result == {"nested": "value"}


# LLM-generated content at query #19
#--------------------------

```python
def test_Reference_validate():
    from typesystem.fields import String, Integer
    from typesystem.base import ValidationError

    # Test 1: Basic validation with valid value
    definitions = Definitions()
    schema = Schema(fields={"name": String(), "age": Integer()})
    definitions["Person"] = schema
    
    reference = Reference(to="Person", definitions=definitions)
    result = reference.validate({"name": "John", "age": 30})
    assert result == {"name": "John", "age": 30}

    # Test 2: Validation with null when allow_null=True
    reference = Reference(to="Person", definitions=definitions, allow_null=True)
    result = reference.validate(None)
    assert result is None

    # Test 3: Validation with null when allow_null=False (default)
    reference = Reference(to="Person", definitions=definitions)
    try:
        reference.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "null"

    # Test 4: Validation with invalid data (should propagate target's validation error)
    reference = Reference(to="Person", definitions=definitions)
    try:
        reference.validate({"name": "John", "age": "not_a_number"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "type"

    # Test 5: Validation with missing required field
    reference = Reference(to="Person", definitions=definitions)
    try:
        reference.validate({"name": "John"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "required"

    # Test 6: Validation with nested schema
    definitions = Definitions()
    address_schema = Schema(fields={"street": String(), "city": String()})
    person_schema = Schema(fields={
        "name": String(),
        "address": Reference(to="Address", definitions=definitions)
    })
    definitions["Address"] = address_schema
    definitions["Person"] = person_schema
    
    reference = Reference(to="Person", definitions=definitions)
    result = reference.validate({
        "name": "Alice",
        "address": {"street": "123 Main St", "city": "Boston"}
    })
    assert result == {
        "name": "Alice",
        "address": {"street": "123 Main St", "city": "Boston"}
    }

    # Test 7: Validation with invalid nested data
    reference = Reference(to="Person", definitions=definitions)
    try:
        reference.validate({
            "name": "Alice",
            "address": {"street": "123 Main St"}
        })
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "required"

    # Test 8: Validation with non-existent definition
    definitions = Definitions()
    reference = Reference(to="NonExistent", definitions=definitions)
    try:
        reference.validate({"name": "John"})
        assert False, "Should have raised KeyError"
    except KeyError:
        pass


# LLM-generated content at query #20
#--------------------------

```python
def test_Reference_validate():
    # Test 1: Valid value with non-null target
    definitions = Definitions()
    field = Field()
    definitions["target_field"] = field
    reference = Reference(to="target_field", definitions=definitions)
    
    # Mock the target's validate method
    mock_value = {"key": "value"}
    field.validate = lambda x: x if x == mock_value else None
    assert reference.validate(mock_value) == mock_value
    
    # Test 2: Null value with allow_null=True
    reference_allow_null = Reference(to="target_field", definitions=definitions, allow_null=True)
    assert reference_allow_null.validate(None) is None
    
    # Test 3: Null value with allow_null=False (default)
    reference_no_null = Reference(to="target_field", definitions=definitions)
    try:
        reference_no_null.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"
    
    # Test 4: Target validation error propagation
    definitions2 = Definitions()
    field2 = Field()
    definitions2["target_field2"] = field2
    reference2 = Reference(to="target_field2", definitions=definitions2)
    
    # Make target field raise ValidationError
    field2.validate = lambda x: (_ for _ in ()).throw(
        ValidationError(messages=[Message(text="Target error", code="target_error")])
    )
    try:
        reference2.validate("some_value")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "target_error"
    
    # Test 5: Non-existent target (should raise KeyError when accessing target property)
    definitions3 = Definitions()
    reference3 = Reference(to="non_existent", definitions=definitions3)
    try:
        reference3.validate("value")
        assert False, "Should have raised KeyError"
    except KeyError:
        pass  # Expected behavior
    
    # Test 6: Valid value with complex target schema
    definitions4 = Definitions()
    schema = Schema(fields={"name": Field()})
    definitions4["person_schema"] = schema
    reference4 = Reference(to="person_schema", definitions=definitions4)
    
    valid_person = {"name": "John"}
    assert reference4.validate(valid_person) == valid_person


# LLM-generated content at query #21
#--------------------------

```python
def test_Reference_validate():
    # Test 1: Valid value passes through to target validation
    definitions = Definitions()
    field = Field()
    definitions["target_field"] = field
    reference = Reference(to="target_field", definitions=definitions)
    
    # Mock the target's validate method
    mock_value = {"test": "data"}
    field.validate = lambda x: x
    result = reference.validate(mock_value)
    assert result == mock_value

    # Test 2: None value with allow_null=True
    reference_allow_null = Reference(to="target_field", definitions=definitions, allow_null=True)
    result = reference_allow_null.validate(None)
    assert result is None

    # Test 3: None value with allow_null=False raises error
    reference_no_null = Reference(to="target_field", definitions=definitions, allow_null=False)
    try:
        reference_no_null.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "null"

    # Test 4: Target validation error is propagated
    definitions2 = Definitions()
    field2 = Field()
    definitions2["target_field2"] = field2
    reference2 = Reference(to="target_field2", definitions=definitions2)
    
    # Make target field raise ValidationError
    def raise_validation_error(value):
        raise ValidationError(messages=[Message(text="Invalid", code="invalid")])
    
    field2.validate = raise_validation_error
    
    try:
        reference2.validate("some_value")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "invalid"

    # Test 5: Non-existent target raises KeyError on access
    definitions3 = Definitions()
    reference3 = Reference(to="non_existent", definitions=definitions3)
    
    try:
        reference3.validate("value")
        assert False, "Should have raised KeyError"
    except KeyError:
        pass


# LLM-generated content at query #22
#--------------------------

```python
def test_Schema_validate():
    # Test 1: Null value with allow_null=True
    field = Field(allow_null=True)
    schema = Schema(fields={}, allow_null=True)
    assert schema.validate(None) is None

    # Test 2: Null value with allow_null=False
    schema = Schema(fields={}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "null"

    # Test 3: Non-dict value
    schema = Schema(fields={})
    try:
        schema.validate("not a dict")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "type"

    # Test 4: Invalid key type (non-string)
    schema = Schema(fields={})
    try:
        schema.validate({1: "value"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "invalid_key"
        assert e.messages()[0].index == [1]

    # Test 5: Missing required field
    field = Field()
    schema = Schema(fields={"required_field": field})
    try:
        schema.validate({})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "required"
        assert e.messages()[0].index == ["required_field"]

    # Test 6: Read-only field should be ignored during validation
    read_only_field = Field(read_only=True)
    schema = Schema(fields={"read_only": read_only_field})
    result = schema.validate({"read_only": "value"})
    assert "read_only" not in result

    # Test 7: Field with default value when missing
    field_with_default = Field(default="default_value")
    schema = Schema(fields={"with_default": field_with_default})
    result = schema.validate({})
    assert result["with_default"] == "default_value"

    # Test 8: Field validation error
    failing_field = Field()
    def failing_validate(value):
        raise ValidationError(messages=[Message(text="Invalid", code="invalid")])
    failing_field.validate = failing_validate
    schema = Schema(fields={"failing": failing_field})
    try:
        schema.validate({"failing": "value"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "invalid"

    # Test 9: Multiple errors combined
    required_field = Field()
    schema = Schema(fields={"required": required_field})
    try:
        schema.validate({2: "invalid key"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 2
        codes = {msg.code for msg in e.messages()}
        assert codes == {"invalid_key", "required"}

    # Test 10: Successful validation with nested fields
    nested_field = Field()
    schema = Schema(fields={"nested": nested_field})
    result = schema.validate({"nested": "valid_value"})
    assert result == {"nested": "valid_value"}

    # Test 11: Field with has_default() returning True
    class DefaultField(Field):
        def has_default(self):
            return True
        def get_default_value(self):
            return "computed_default"
    
    default_field = DefaultField()
    schema = Schema(fields={"defaulted": default_field})
    result = schema.validate({})
    assert result["defaulted"] == "computed_default"

    # Test 12: Field that's both read_only and has default
    complex_field = Field(read_only=True, default="ignored")
    schema = Schema(fields={"complex": complex_field})
    result = schema.validate({})
    assert "complex" not in result


# LLM-generated content at query #23
#--------------------------

```python
def test_Schema_validate():
    # Test 1: Null value with allow_null=True
    field = Field(allow_null=True)
    schema = Schema(fields={}, allow_null=True)
    assert schema.validate(None) is None

    # Test 2: Null value with allow_null=False
    schema = Schema(fields={}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "null"

    # Test 3: Non-dict value
    schema = Schema(fields={})
    try:
        schema.validate("not a dict")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "type"

    # Test 4: Invalid key type (non-string)
    schema = Schema(fields={})
    try:
        schema.validate({1: "value"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "invalid_key"
        assert e.messages()[0].index == [1]

    # Test 5: Missing required field
    field = Field()
    schema = Schema(fields={"required_field": field})
    try:
        schema.validate({})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "required"
        assert e.messages()[0].index == ["required_field"]

    # Test 6: Read-only field should be ignored
    read_only_field = Field(read_only=True)
    schema = Schema(fields={"read_only": read_only_field})
    result = schema.validate({"other": "value"})
    assert "read_only" not in result

    # Test 7: Field with default value
    field_with_default = Field(default="default_value")
    schema = Schema(fields={"with_default": field_with_default})
    result = schema.validate({})
    assert result["with_default"] == "default_value"

    # Test 8: Valid field validation
    field = Field()
    schema = Schema(fields={"name": field})
    result = schema.validate({"name": "John"})
    assert result["name"] == "John"

    # Test 9: Field validation error
    def always_fail(value):
        raise ValidationError(text="Always fails", code="custom")

    failing_field = Field(validators=[always_fail])
    schema = Schema(fields={"failing": failing_field})
    try:
        schema.validate({"failing": "any value"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "custom"

    # Test 10: Multiple errors
    required_field = Field()
    schema = Schema(fields={"required": required_field})
    try:
        schema.validate({1: "invalid key"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 2
        codes = {msg.code for msg in messages}
        assert "invalid_key" in codes
        assert "required" in codes

    # Test 11: Complex nested validation
    nested_field = Field()
    nested_schema = Schema(fields={"nested": nested_field})
    schema = Schema(fields={"parent": nested_schema})
    result = schema.validate({"parent": {"nested": "value"}})
    assert result["parent"]["nested"] == "value"

    # Test 12: Field with allow_null=True
    nullable_field = Field(allow_null=True)
    schema = Schema(fields={"nullable": nullable_field})
    result = schema.validate({"nullable": None})
    assert result["nullable"] is None

    # Test 13: Field with allow_null=False
    non_nullable_field = Field(allow_null=False)
    schema = Schema(fields={"non_nullable": non_nullable_field})
    try:
        schema.validate({"non_nullable": None})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "null"


# LLM-generated content at query #24
#--------------------------

```python
def test_Reference_validate():
    from typesystem.fields import String, Integer
    from typesystem.base import ValidationError
    
    # Test 1: Valid value with non-null reference
    definitions = Definitions()
    schema = Schema(fields={"name": String(), "age": Integer()})
    definitions["Person"] = schema
    reference = Reference(to="Person", definitions=definitions)
    
    valid_data = {"name": "John", "age": 30}
    result = reference.validate(valid_data)
    assert result == valid_data
    
    # Test 2: Null value with allow_null=False (default)
    reference_no_null = Reference(to="Person", definitions=definitions)
    try:
        reference_no_null.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.messages()[0].code == "null"
        assert "May not be null" in str(e)
    
    # Test 3: Null value with allow_null=True
    reference_with_null = Reference(to="Person", definitions=definitions, allow_null=True)
    result = reference_with_null.validate(None)
    assert result is None
    
    # Test 4: Invalid value according to referenced schema
    invalid_data = {"name": "John", "age": "not_an_integer"}
    try:
        reference.validate(invalid_data)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) > 0
    
    # Test 5: Missing required field in referenced schema
    required_schema = Schema(fields={"name": String()})
    definitions["RequiredPerson"] = required_schema
    required_reference = Reference(to="RequiredPerson", definitions=definitions)
    
    try:
        required_reference.validate({})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert any(msg.code == "required" for msg in e.messages())
    
    # Test 6: Valid nested reference
    nested_definitions = Definitions()
    inner_schema = Schema(fields={"id": Integer()})
    outer_schema = Schema(fields={"data": Reference(to="Inner", definitions=nested_definitions)})
    nested_definitions["Inner"] = inner_schema
    nested_definitions["Outer"] = outer_schema
    
    outer_reference = Reference(to="Outer", definitions=nested_definitions)
    nested_data = {"data": {"id": 123}}
    result = outer_reference.validate(nested_data)
    assert result == nested_data
    
    # Test 7: Non-dict value passed to schema validation
    try:
        reference.validate("not_a_dict")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.messages()[0].code == "type"
    
    # Test 8: Reference to non-existent definition
    empty_definitions = Definitions()
    bad_reference = Reference(to="NonExistent", definitions=empty_definitions)
    try:
        bad_reference.validate({"test": "data"})
        assert False, "Should have raised KeyError"
    except KeyError:
        pass


# LLM-generated content at query #25
#--------------------------

```python
def test_Schema_validate():
    from typesystem.fields import String, Integer, Boolean, Array
    from typesystem.base import ValidationError

    # Test basic validation with simple fields
    schema = Schema({
        "name": String(max_length=10),
        "age": Integer(minimum=0, maximum=150),
        "active": Boolean()
    })

    # Test valid input
    valid_data = {"name": "John", "age": 30, "active": True}
    result = schema.validate(valid_data)
    assert result == valid_data

    # Test null value when not allowed
    try:
        schema.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "null"

    # Test null value when allowed
    nullable_schema = Schema({
        "name": String(max_length=10)
    }, allow_null=True)
    assert nullable_schema.validate(None) is None

    # Test non-dict input
    try:
        schema.validate("not a dict")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.messages()[0].code == "type"

    # Test invalid key type
    try:
        schema.validate({1: "value"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.messages()[0].code == "invalid_key"

    # Test missing required field
    try:
        schema.validate({"name": "John"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.messages()[0].code == "required"
        assert e.messages()[0].index == ["age"]

    # Test field validation errors
    try:
        schema.validate({"name": "VeryLongName", "age": 200, "active": True})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 2
        codes = {msg.code for msg in messages}
        assert "max_length" in codes
        assert "maximum" in codes

    # Test read_only field
    schema_with_readonly = Schema({
        "id": Integer(read_only=True),
        "name": String()
    })
    result = schema_with_readonly.validate({"name": "Alice"})
    assert "id" not in result
    assert result["name"] == "Alice"

    # Test field with default value
    schema_with_default = Schema({
        "name": String(),
        "count": Integer(default=0)
    })
    result = schema_with_default.validate({"name": "Bob"})
    assert result["name"] == "Bob"
    assert result["count"] == 0

    # Test nested validation with Array field
    nested_schema = Schema({
        "tags": Array(items=String()),
        "metadata": Schema({"key": String(), "value": String()})
    })
    
    valid_nested = {
        "tags": ["tag1", "tag2"],
        "metadata": {"key": "color", "value": "blue"}
    }
    result = nested_schema.validate(valid_nested)
    assert result == valid_nested

    # Test nested validation error
    try:
        nested_schema.validate({
            "tags": ["tag1", 123],
            "metadata": {"key": "color", "value": "blue"}
        })
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 1
        assert messages[0].code == "type"

    # Test multiple error accumulation
    try:
        schema.validate({
            "name": "VeryLongNameThatExceedsLimit",
            "age": -5
        })
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 3  # missing required, max_length, minimum

    # Test with allow_null on child fields
    schema_with_nullable_child = Schema({
        "name": String(allow_null=True),
        "age": Integer()
    })
    
    result = schema_with_nullable_child.validate({"name": None, "age": 25})
    assert result["name"] is None
    assert result["age"] == 25

    # Test empty schema
    empty_schema = Schema({})
    result = empty_schema.validate({})
    assert result == {}

    # Test with non-string mapping
    from collections import OrderedDict
    ordered_data = OrderedDict([("name", "John"), ("age", 30)])
    result = schema.validate(ordered_data)
    assert dict(result) == {"name": "John", "age": 30, "active": True}


# LLM-generated content at query #26
#--------------------------

```python
def test_Schema_validate():
    # Test basic validation with simple fields
    from typesystem.fields import String, Integer
    
    schema = Schema(fields={
        "name": String(max_length=10),
        "age": Integer(minimum=0, maximum=150)
    })
    
    # Test valid input
    result = schema.validate({"name": "John", "age": 30})
    assert result == {"name": "John", "age": 30}
    
    # Test null value when not allowed
    try:
        schema.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "null"
    
    # Test null value when allowed
    schema_with_null = Schema(
        fields={"name": String()},
        allow_null=True
    )
    result = schema_with_null.validate(None)
    assert result is None
    
    # Test non-dict input
    try:
        schema.validate("not a dict")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.messages()[0].code == "type"
    
    # Test invalid key type
    try:
        schema.validate({1: "value", "name": "John"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 1
        assert messages[0].code == "invalid_key"
        assert messages[0].index == [1]
    
    # Test required fields
    try:
        schema.validate({"name": "John"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 1
        assert messages[0].code == "required"
        assert messages[0].index == ["age"]
    
    # Test field validation errors
    try:
        schema.validate({"name": "John" * 5, "age": 200})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 2
        codes = {msg.code for msg in messages}
        assert "max_length" in codes
        assert "maximum" in codes
    
    # Test read_only field
    schema_with_readonly = Schema(fields={
        "id": Integer(read_only=True),
        "name": String()
    })
    result = schema_with_readonly.validate({"name": "Alice"})
    assert result == {"name": "Alice"}
    assert "id" not in result
    
    # Test field with default value
    from typesystem.fields import Boolean
    
    schema_with_default = Schema(fields={
        "name": String(),
        "active": Boolean(default=True)
    })
    result = schema_with_default.validate({"name": "Bob"})
    assert result == {"name": "Bob", "active": True}
    
    # Test nested validation errors with prefixes
    nested_schema = Schema(fields={
        "profile": Schema(fields={
            "email": String(format="email")
        })
    })
    try:
        nested_schema.validate({"profile": {"email": "invalid-email"}})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 1
        assert messages[0].code == "format"
        assert messages[0].index == ["profile", "email"]
    
    # Test multiple error accumulation
    complex_schema = Schema(fields={
        "required1": String(),
        "required2": Integer(),
        "email": String(format="email")
    })
    try:
        complex_schema.validate({
            "email": "not-an-email",
            "invalid_key": 123
        })
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) >= 3
        codes = {msg.code for msg in messages}
        assert "required" in codes
        assert "invalid_key" in codes
        assert "format" in codes
    
    # Test with Mapping type (not just dict)
    from collections import OrderedDict
    
    mapping_value = OrderedDict([("name", "John"), ("age", 25)])
    result = schema.validate(mapping_value)
    assert result == {"name": "John", "age": 25}


# LLM-generated content at query #27
#--------------------------

```python
def test_Reference_validate():
    from typesystem.fields import Field
    from typesystem.base import ValidationError
    
    # Test 1: Valid value with non-null target
    definitions = Definitions()
    field = Field()
    field.validate = lambda x: x * 2
    definitions["target"] = field
    ref = Reference(to="target", definitions=definitions)
    assert ref.validate(5) == 10
    
    # Test 2: Null value with allow_null=True
    ref = Reference(to="target", definitions=definitions, allow_null=True)
    assert ref.validate(None) is None
    
    # Test 3: Null value with allow_null=False (default)
    ref = Reference(to="target", definitions=definitions)
    try:
        ref.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.messages()[0].code == "null"
    
    # Test 4: Target validation error propagation
    class FailingField(Field):
        def validate(self, value):
            raise ValidationError("Invalid value")
    
    failing_field = FailingField()
    definitions["failing"] = failing_field
    ref = Reference(to="failing", definitions=definitions)
    try:
        ref.validate("anything")
        assert False, "Should have raised ValidationError"
    except ValidationError:
        pass
    
    # Test 5: Target not in definitions (should raise KeyError)
    ref = Reference(to="nonexistent", definitions=definitions)
    try:
        ref.validate("value")
        assert False, "Should have raised KeyError"
    except KeyError:
        pass
    
    # Test 6: Complex target validation
    class ComplexField(Field):
        def validate(self, value):
            if not isinstance(value, dict):
                raise ValidationError("Must be dict")
            return {"validated": value}
    
    complex_field = ComplexField()
    definitions["complex"] = complex_field
    ref = Reference(to="complex", definitions=definitions)
    result = ref.validate({"key": "value"})
    assert result == {"validated": {"key": "value"}}


# LLM-generated content at query #28
#--------------------------

```python
def test_Reference_validate():
    # Test 1: Valid value with non-null reference
    definitions = Definitions()
    field = Field()
    definitions["test_ref"] = field
    ref = Reference(to="test_ref", definitions=definitions)
    
    # Mock the target's validate method
    mock_value = {"key": "value"}
    field.validate = lambda x: x
    assert ref.validate(mock_value) == mock_value
    
    # Test 2: Null value with allow_null=True
    ref_allow_null = Reference(to="test_ref", definitions=definitions, allow_null=True)
    assert ref_allow_null.validate(None) is None
    
    # Test 3: Null value with allow_null=False (default)
    ref_no_null = Reference(to="test_ref", definitions=definitions)
    try:
        ref_no_null.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "null"
    
    # Test 4: Target validation error propagation
    definitions2 = Definitions()
    field2 = Field()
    definitions2["error_ref"] = field2
    ref2 = Reference(to="error_ref", definitions=definitions2)
    
    # Make target field raise validation error
    def raise_validation_error(value):
        raise ValidationError(messages=[Message(text="Invalid", code="invalid")])
    
    field2.validate = raise_validation_error
    
    try:
        ref2.validate("invalid_value")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "invalid"
    
    # Test 5: Reference to non-existent definition
    definitions3 = Definitions()
    ref3 = Reference(to="missing_ref", definitions=definitions3)
    
    try:
        ref3.validate({"key": "value"})
        assert False, "Should have raised KeyError"
    except KeyError:
        pass
    
    # Test 6: Valid value with nested validation
    definitions4 = Definitions()
    schema = Schema(fields={"name": Field()})
    definitions4["person_ref"] = schema
    ref4 = Reference(to="person_ref", definitions=definitions4)
    
    # Mock schema validation
    validated_result = {"name": "John"}
    schema.validate = lambda x: validated_result
    assert ref4.validate({"name": "John"}) == validated_result


# LLM-generated content at query #29
#--------------------------

```python
def test_Reference_validate():
    from typesystem.fields import Field
    from typesystem.base import ValidationError
    
    # Test 1: Valid value with non-null target
    definitions = Definitions()
    field = Field()
    field.validate = lambda x: x * 2
    definitions["target"] = field
    ref = Reference(to="target", definitions=definitions)
    assert ref.validate(5) == 10
    
    # Test 2: Null value with allow_null=True
    ref = Reference(to="target", definitions=definitions, allow_null=True)
    assert ref.validate(None) is None
    
    # Test 3: Null value with allow_null=False (default)
    ref = Reference(to="target", definitions=definitions)
    try:
        ref.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.messages()[0].code == "null"
    
    # Test 4: Target validation error propagation
    class ErrorField(Field):
        def validate(self, value):
            raise ValidationError("Invalid value")
    
    error_field = ErrorField()
    definitions["error_target"] = error_field
    ref = Reference(to="error_target", definitions=definitions)
    try:
        ref.validate("anything")
        assert False, "Should have raised ValidationError"
    except ValidationError:
        pass
    
    # Test 5: Target not in definitions
    ref = Reference(to="nonexistent", definitions=definitions)
    try:
        ref.validate(5)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass
    
    # Test 6: Complex target validation
    class ComplexField(Field):
        def validate(self, value):
            if not isinstance(value, dict):
                raise ValidationError("Must be dict")
            return {"validated": value}
    
    complex_field = ComplexField()
    definitions["complex"] = complex_field
    ref = Reference(to="complex", definitions=definitions)
    result = ref.validate({"key": "value"})
    assert result == {"validated": {"key": "value"}}


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_Schema_serialize():
    from typesystem.fields import String, Integer, Boolean
    
    # Test 1: Serialize None object
    schema = Schema(fields={})
    assert schema.serialize(None) is None
    
    # Test 2: Serialize dict object
    schema = Schema(fields={
        "name": String(),
        "age": Integer(),
        "active": Boolean()
    })
    
    obj_dict = {
        "name": "John",
        "age": 30,
        "active": True
    }
    
    result = schema.serialize(obj_dict)
    assert result == {"name": "John", "age": 30, "active": True}
    
    # Test 3: Serialize object with attributes (non-dict)
    class Person:
        def __init__(self, name, age, active):
            self.name = name
            self.age = age
            self.active = active
    
    person = Person(name="Alice", age=25, active=False)
    result = schema.serialize(person)
    assert result == {"name": "Alice", "age": 25, "active": False}
    
    # Test 4: Serialize with missing fields in object
    obj_dict = {"name": "Bob"}
    result = schema.serialize(obj_dict)
    assert result == {"name": "Bob"}
    
    # Test 5: Serialize with nested field serialization
    nested_schema = Schema(fields={
        "nested": String()
    })
    
    obj_dict = {"nested": "value"}
    result = nested_schema.serialize(obj_dict)
    assert result == {"nested": "value"}
    
    # Test 6: Serialize with read_only field
    schema = Schema(fields={
        "name": String(),
        "id": String(read_only=True)
    })
    
    obj_dict = {"name": "Test", "id": "123"}
    result = schema.serialize(obj_dict)
    assert result == {"name": "Test"}
    
    # Test 7: Serialize empty dict
    schema = Schema(fields={})
    assert schema.serialize({}) == {}
    
    # Test 8: Serialize with field that has custom serialize behavior
    class CustomField(Field):
        def serialize(self, obj):
            return f"custom_{obj}"
    
    schema = Schema(fields={
        "data": CustomField()
    })
    
    obj_dict = {"data": "test"}
    result = schema.serialize(obj_dict)
    assert result == {"data": "custom_test"}


# LLM-generated content at query #2
#--------------------------

```python
def test_Schema():
    # Test basic initialization with fields
    field1 = Field()
    field2 = Field()
    fields = {"name": field1, "age": field2}
    schema = Schema(fields)
    
    assert schema.fields == fields
    assert schema.required == ["name", "age"]
    
    # Test with read_only field
    read_only_field = Field(read_only=True)
    fields_with_readonly = {"name": field1, "id": read_only_field}
    schema2 = Schema(fields_with_readonly)
    
    assert "id" not in schema2.required
    
    # Test with field having default value
    field_with_default = Field(default="default_value")
    fields_with_default = {"name": field1, "optional": field_with_default}
    schema3 = Schema(fields_with_default)
    
    assert "optional" not in schema3.required
    
    # Test with mixed fields
    mixed_fields = {
        "required_field": field1,
        "read_only_field": Field(read_only=True),
        "default_field": Field(default=0)
    }
    schema4 = Schema(mixed_fields)
    
    assert schema4.required == ["required_field"]
    
    # Test inheritance from Field
    assert isinstance(schema, Field)
    
    # Test with additional kwargs passed to parent Field
    schema5 = Schema(fields, allow_null=True, description="Test schema")
    assert schema5.allow_null == True
    assert schema5.description == "Test schema"


# LLM-generated content at query #3
#--------------------------

```python
def test_Schema_validate():
    # Test 1: Basic validation with valid data
    field1 = Field()
    field2 = Field()
    schema = Schema({"name": field1, "age": field2})
    
    value = {"name": "John", "age": 30}
    result = schema.validate(value)
    assert result == {"name": "John", "age": 30}
    
    # Test 2: Null value when allow_null is False (default)
    schema = Schema({"name": Field()})
    try:
        schema.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "null"
    
    # Test 3: Null value when allow_null is True
    schema = Schema({"name": Field()}, allow_null=True)
    result = schema.validate(None)
    assert result is None
    
    # Test 4: Non-dict value
    schema = Schema({"name": Field()})
    try:
        schema.validate("not a dict")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "type"
    
    # Test 5: Invalid key type (non-string keys)
    schema = Schema({"name": Field()})
    try:
        schema.validate({1: "value"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "invalid_key"
        assert e.messages()[0].index == [1]
    
    # Test 6: Missing required field
    field = Field()
    schema = Schema({"name": field})
    try:
        schema.validate({})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "required"
        assert e.messages()[0].index == ["name"]
    
    # Test 7: Field with default value when missing
    field_with_default = Field(default="default_value")
    schema = Schema({"name": field_with_default})
    result = schema.validate({})
    assert result == {"name": "default_value"}
    
    # Test 8: Read-only field should be ignored during validation
    read_only_field = Field(read_only=True)
    schema = Schema({"name": Field(), "id": read_only_field})
    value = {"name": "John", "id": 123}
    result = schema.validate(value)
    assert result == {"name": "John"}
    
    # Test 9: Field validation error
    field_with_validation = Field()
    def failing_validate(x):
        raise ValidationError(messages=[Message(text="Invalid", code="invalid")])
    field_with_validation.validate = failing_validate
    schema = Schema({"data": field_with_validation})
    
    try:
        schema.validate({"data": "some value"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "invalid"
    
    # Test 10: Multiple validation errors
    field1 = Field()
    field2 = Field()
    schema = Schema({"field1": field1, "field2": field2})
    
    try:
        schema.validate({1: "invalid key"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 3  # invalid_key + 2 required fields
        codes = [msg.code for msg in messages]
        assert "invalid_key" in codes
        assert "required" in codes
    
    # Test 11: Nested validation with child field errors
    child_field = Field()
    def child_validate(x):
        if x != "valid":
            raise ValidationError(messages=[Message(text="Child error", code="child_error")])
        return x
    child_field.validate = child_validate
    
    schema = Schema({"child": child_field})
    
    try:
        schema.validate({"child": "invalid"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "child_error"
    
    # Test 12: Valid data with multiple fields
    name_field = Field()
    age_field = Field()
    email_field = Field()
    schema = Schema({"name": name_field, "age": age_field, "email": email_field})
    
    value = {"name": "Alice", "age": 25, "email": "alice@example.com"}
    result = schema.validate(value)
    assert result == value
    
    # Test 13: Field with get_default_value method
    class CustomField(Field):
        def get_default_value(self):
            return "custom_default"
    
    custom_field = CustomField()
    schema = Schema({"field": custom_field})
    result = schema.validate({})
    assert result == {"field": "custom_default"}
    
    # Test 14: Mixed valid and invalid data
    valid_field = Field()
    invalid_field = Field()
    def invalid_validate(x):
        raise ValidationError(messages=[Message(text="Always invalid", code="always_invalid")])
    invalid_field.validate = invalid_validate
    
    schema = Schema({"valid": valid_field, "invalid": invalid_field})
    
    try:
        schema.validate({"valid": "ok", "invalid": "anything"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "always_invalid"


# LLM-generated content at query #4
#--------------------------

```python
def test_Schema_serialize():
    from typesystem.fields import String, Integer, Boolean
    
    # Test 1: Serialize None object
    schema = Schema(fields={})
    result = schema.serialize(None)
    assert result is None
    
    # Test 2: Serialize empty dict
    schema = Schema(fields={})
    result = schema.serialize({})
    assert result == {}
    
    # Test 3: Serialize dict with simple fields
    schema = Schema(fields={
        "name": String(),
        "age": Integer(),
        "active": Boolean()
    })
    obj = {"name": "John", "age": 30, "active": True}
    result = schema.serialize(obj)
    assert result == {"name": "John", "age": 30, "active": True}
    
    # Test 4: Serialize dict with missing fields
    schema = Schema(fields={
        "name": String(),
        "age": Integer(),
        "active": Boolean()
    })
    obj = {"name": "Jane"}
    result = schema.serialize(obj)
    assert result == {"name": "Jane"}
    
    # Test 5: Serialize dict with extra fields (should be ignored)
    schema = Schema(fields={
        "name": String(),
        "age": Integer()
    })
    obj = {"name": "Bob", "age": 25, "extra": "ignored"}
    result = schema.serialize(obj)
    assert result == {"name": "Bob", "age": 25}
    assert "extra" not in result
    
    # Test 6: Serialize object with attributes (not dict)
    class Person:
        def __init__(self, name, age):
            self.name = name
            self.age = age
    
    schema = Schema(fields={
        "name": String(),
        "age": Integer()
    })
    person = Person("Alice", 28)
    result = schema.serialize(person)
    assert result == {"name": "Alice", "age": 28}
    
    # Test 7: Serialize object with missing attributes
    class PartialPerson:
        def __init__(self, name):
            self.name = name
    
    schema = Schema(fields={
        "name": String(),
        "age": Integer()
    })
    person = PartialPerson("Charlie")
    result = schema.serialize(person)
    assert result == {"name": "Charlie"}
    
    # Test 8: Serialize with nested schema
    address_schema = Schema(fields={
        "street": String(),
        "city": String()
    })
    user_schema = Schema(fields={
        "name": String(),
        "address": address_schema
    })
    obj = {
        "name": "David",
        "address": {"street": "123 Main", "city": "Boston"}
    }
    result = user_schema.serialize(obj)
    assert result == {
        "name": "David",
        "address": {"street": "123 Main", "city": "Boston"}
    }
    
    # Test 9: Serialize with read_only field
    schema = Schema(fields={
        "name": String(),
        "id": String(read_only=True)
    })
    obj = {"name": "Eve", "id": "12345"}
    result = schema.serialize(obj)
    assert result == {"name": "Eve"}
    
    # Test 10: Serialize with field that has serialize method
    class CustomField(Field):
        def serialize(self, obj):
            return f"custom_{obj}"
    
    schema = Schema(fields={
        "data": CustomField()
    })
    obj = {"data": "value"}
    result = schema.serialize(obj)
    assert result == {"data": "custom_value"}


# LLM-generated content at query #5
#--------------------------

```python
def test_Schema_validate():
    # Test 1: Null value with allow_null=True
    field = Field(allow_null=True)
    schema = Schema(fields={}, allow_null=True)
    assert schema.validate(None) is None

    # Test 2: Null value with allow_null=False
    schema = Schema(fields={}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "null"

    # Test 3: Non-dict value
    schema = Schema(fields={})
    try:
        schema.validate("not a dict")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "type"

    # Test 4: Invalid key type (non-string)
    schema = Schema(fields={})
    try:
        schema.validate({1: "value"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "invalid_key"
        assert e.messages()[0].index == [1]

    # Test 5: Missing required field
    field = Field()
    schema = Schema(fields={"required_field": field})
    try:
        schema.validate({})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "required"
        assert e.messages()[0].index == ["required_field"]

    # Test 6: Read-only field should be ignored
    read_only_field = Field(read_only=True)
    schema = Schema(fields={"read_only": read_only_field})
    result = schema.validate({"other": "value"})
    assert "read_only" not in result

    # Test 7: Field with default value when missing
    default_field = Field(default="default_value")
    schema = Schema(fields={"with_default": default_field})
    result = schema.validate({})
    assert result["with_default"] == "default_value"

    # Test 8: Valid field validation
    string_field = Field()
    schema = Schema(fields={"name": string_field})
    result = schema.validate({"name": "John"})
    assert result["name"] == "John"

    # Test 9: Nested field validation error
    nested_field = Field()
    schema = Schema(fields={"nested": nested_field})
    try:
        schema.validate({"nested": None})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "null"

    # Test 10: Multiple validation errors
    required_field = Field()
    schema = Schema(fields={"req1": required_field, "req2": required_field})
    try:
        schema.validate({2: "invalid key"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 3
        codes = {msg.code for msg in messages}
        assert "invalid_key" in codes
        assert "required" in codes

    # Test 11: Field with default and read_only combination
    complex_field = Field(default="default", read_only=True)
    schema = Schema(fields={"complex": complex_field})
    result = schema.validate({})
    assert "complex" not in result

    # Test 12: Valid complex schema
    name_field = Field()
    age_field = Field()
    schema = Schema(fields={"name": name_field, "age": age_field})
    result = schema.validate({"name": "Alice", "age": 30})
    assert result["name"] == "Alice"
    assert result["age"] == 30

    # Test 13: Field validation passes with correct value
    int_field = Field()
    schema = Schema(fields={"count": int_field})
    result = schema.validate({"count": 42})
    assert result["count"] == 42


# LLM-generated content at query #6
#--------------------------

```python
def test_Schema_serialize():
    from typesystem.fields import String, Integer, Boolean
    
    # Test 1: Serialize None object
    schema = Schema(fields={})
    assert schema.serialize(None) is None
    
    # Test 2: Serialize empty dict
    schema = Schema(fields={})
    assert schema.serialize({}) == {}
    
    # Test 3: Serialize dict with simple fields
    schema = Schema(fields={
        "name": String(),
        "age": Integer(),
        "active": Boolean()
    })
    obj = {"name": "John", "age": 30, "active": True}
    result = schema.serialize(obj)
    assert result == {"name": "John", "age": 30, "active": True}
    
    # Test 4: Serialize dict with missing fields
    schema = Schema(fields={
        "name": String(),
        "age": Integer(),
        "active": Boolean()
    })
    obj = {"name": "John", "age": 30}
    result = schema.serialize(obj)
    assert result == {"name": "John", "age": 30}
    
    # Test 5: Serialize object with attributes (non-dict)
    class Person:
        def __init__(self, name, age, active):
            self.name = name
            self.age = age
            self.active = active
    
    schema = Schema(fields={
        "name": String(),
        "age": Integer(),
        "active": Boolean()
    })
    person = Person("Alice", 25, False)
    result = schema.serialize(person)
    assert result == {"name": "Alice", "age": 25, "active": False}
    
    # Test 6: Serialize with nested Schema field
    address_schema = Schema(fields={
        "street": String(),
        "city": String()
    })
    schema = Schema(fields={
        "name": String(),
        "address": address_schema
    })
    obj = {
        "name": "Bob",
        "address": {"street": "123 Main St", "city": "Boston"}
    }
    result = schema.serialize(obj)
    assert result == {
        "name": "Bob",
        "address": {"street": "123 Main St", "city": "Boston"}
    }
    
    # Test 7: Serialize with read_only field
    schema = Schema(fields={
        "name": String(),
        "id": Integer(read_only=True),
        "age": Integer()
    })
    obj = {"name": "Charlie", "id": 123, "age": 40}
    result = schema.serialize(obj)
    assert result == {"name": "Charlie", "age": 40}
    
    # Test 8: Serialize with field that has serialize method
    class CustomField(Field):
        def serialize(self, obj):
            return obj.upper() if obj else None
    
    schema = Schema(fields={
        "name": CustomField(),
        "code": String()
    })
    obj = {"name": "test", "code": "abc"}
    result = schema.serialize(obj)
    assert result == {"name": "TEST", "code": "abc"}
    
    # Test 9: Serialize with missing attribute on object
    class PartialPerson:
        def __init__(self, name):
            self.name = name
    
    schema = Schema(fields={
        "name": String(),
        "age": Integer()
    })
    person = PartialPerson("David")
    result = schema.serialize(person)
    assert result == {"name": "David"}
    
    # Test 10: Serialize empty object with fields
    schema = Schema(fields={
        "name": String(),
        "age": Integer()
    })
    result = schema.serialize({})
    assert result == {}


# LLM-generated content at query #7
#--------------------------

```python
def test_Reference_validate():
    from typesystem.fields import Integer, String
    
    # Test 1: Valid value with non-null reference
    definitions = Definitions()
    definitions["Person"] = Schema({"name": String(), "age": Integer()})
    ref = Reference(to="Person", definitions=definitions)
    
    valid_data = {"name": "John", "age": 30}
    result = ref.validate(valid_data)
    assert result == valid_data
    
    # Test 2: Null value with allow_null=False (default)
    ref_no_null = Reference(to="Person", definitions=definitions)
    try:
        ref_no_null.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.messages()[0].code == "null"
    
    # Test 3: Null value with allow_null=True
    ref_with_null = Reference(to="Person", definitions=definitions, allow_null=True)
    result = ref_with_null.validate(None)
    assert result is None
    
    # Test 4: Invalid value according to target schema
    invalid_data = {"name": "John", "age": "not_an_integer"}
    try:
        ref.validate(invalid_data)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) > 0
    
    # Test 5: Valid value with nested schema reference
    definitions["Address"] = Schema({"street": String(), "city": String()})
    address_ref = Reference(to="Address", definitions=definitions)
    
    address_data = {"street": "123 Main St", "city": "Anytown"}
    result = address_ref.validate(address_data)
    assert result == address_data
    
    # Test 6: Reference to non-existent definition
    bad_ref = Reference(to="NonExistent", definitions=definitions)
    try:
        bad_ref.validate({"test": "data"})
        assert False, "Should have raised KeyError when accessing target"
    except KeyError:
        pass
    
    # Test 7: Test that target property returns correct schema
    assert ref.target is definitions["Person"]
    
    # Test 8: Valid value with complex nested schema
    definitions["Company"] = Schema({
        "name": String(),
        "employees": Integer(minimum=1)
    })
    company_ref = Reference(to="Company", definitions=definitions)
    
    company_data = {"name": "Tech Corp", "employees": 100}
    result = company_ref.validate(company_data)
    assert result == company_data


# LLM-generated content at query #8
#--------------------------

```python
def test_Schema_validate():
    # Test 1: Null value with allow_null=True
    field = Field(allow_null=True)
    schema = Schema(fields={}, allow_null=True)
    assert schema.validate(None) is None

    # Test 2: Null value with allow_null=False
    schema = Schema(fields={}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "null"

    # Test 3: Non-dict value
    schema = Schema(fields={})
    try:
        schema.validate("not a dict")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "type"

    # Test 4: Invalid key type (non-string)
    schema = Schema(fields={})
    try:
        schema.validate({1: "value"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "invalid_key"
        assert e.messages()[0].index == [1]

    # Test 5: Missing required field
    field = Field()
    schema = Schema(fields={"required_field": field})
    try:
        schema.validate({})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "required"
        assert e.messages()[0].index == ["required_field"]

    # Test 6: Required field with read_only
    field = Field(read_only=True)
    schema = Schema(fields={"read_only_field": field})
    assert schema.validate({}) == {}

    # Test 7: Field with default value
    field = Field(default="default_value")
    schema = Schema(fields={"field_with_default": field})
    result = schema.validate({})
    assert result == {"field_with_default": "default_value"}

    # Test 8: Valid nested validation
    nested_field = Field()
    schema = Schema(fields={"nested": nested_field})
    result = schema.validate({"nested": "value"})
    assert result == {"nested": "value"}

    # Test 9: Invalid nested validation
    nested_field = Field()
    schema = Schema(fields={"nested": nested_field})
    try:
        schema.validate({"nested": None})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert "null" in e.messages()[0].code

    # Test 10: Multiple errors
    field1 = Field()
    field2 = Field()
    schema = Schema(fields={"field1": field1, "field2": field2})
    try:
        schema.validate({3: "invalid", "field1": None})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 3
        codes = {msg.code for msg in messages}
        assert "invalid_key" in codes
        assert "required" in codes
        assert any("null" in msg.code for msg in messages)

    # Test 11: Complex nested structure
    inner_schema = Schema(fields={"inner_field": Field()})
    outer_schema = Schema(fields={"outer_field": inner_schema})
    result = outer_schema.validate({"outer_field": {"inner_field": "value"}})
    assert result == {"outer_field": {"inner_field": "value"}}

    # Test 12: Skip read_only fields during validation
    read_only_field = Field(read_only=True)
    regular_field = Field()
    schema = Schema(fields={"read_only": read_only_field, "regular": regular_field})
    result = schema.validate({"regular": "value"})
    assert result == {"regular": "value"}
    assert "read_only" not in result

    # Test 13: Mixed valid and invalid
    field1 = Field(default="default1")
    field2 = Field()
    field3 = Field()
    schema = Schema(fields={"field1": field1, "field2": field2, "field3": field3})
    
    try:
        schema.validate({"field2": "valid", "field3": None})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 1
        assert any("null" in msg.code for msg in messages)


# LLM-generated content at query #9
#--------------------------

```python
def test_Schema_validate():
    # Test 1: Basic validation with valid data
    from typesystem.fields import String, Integer
    from typesystem.base import ValidationError
    
    schema = Schema({
        "name": String(max_length=10),
        "age": Integer(minimum=0, maximum=150)
    })
    
    result = schema.validate({"name": "John", "age": 30})
    assert result == {"name": "John", "age": 30}
    
    # Test 2: Null value with allow_null=False (default)
    try:
        schema.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.messages()[0].code == "null"
    
    # Test 3: Null value with allow_null=True
    schema_nullable = Schema(
        {"name": String()},
        allow_null=True
    )
    assert schema_nullable.validate(None) is None
    
    # Test 4: Non-dict value
    try:
        schema.validate("not a dict")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.messages()[0].code == "type"
    
    # Test 5: Invalid key type (non-string)
    try:
        schema.validate({1: "value", "name": "John"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert any(msg.code == "invalid_key" for msg in e.messages())
    
    # Test 6: Missing required field
    try:
        schema.validate({"name": "John"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert any(msg.code == "required" and msg.index == ["age"] for msg in e.messages())
    
    # Test 7: Field validation error
    try:
        schema.validate({"name": "John", "age": -5})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert any("minimum" in msg.code for msg in e.messages())
    
    # Test 8: Field with default value
    schema_with_default = Schema({
        "name": String(),
        "age": Integer(default=25)
    })
    result = schema_with_default.validate({"name": "John"})
    assert result == {"name": "John", "age": 25}
    
    # Test 9: Read-only field should be ignored during validation
    read_only_field = String(read_only=True)
    schema_readonly = Schema({
        "name": String(),
        "id": read_only_field
    })
    result = schema_readonly.validate({"name": "John", "id": "123"})
    assert result == {"name": "John"}
    
    # Test 10: Multiple validation errors
    try:
        schema.validate({})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 2
        codes = {msg.code for msg in messages}
        assert "required" in codes
    
    # Test 11: Nested validation with child schema errors
    nested_schema = Schema({
        "person": Schema({
            "name": String(required=True),
            "age": Integer()
        })
    })
    
    try:
        nested_schema.validate({"person": {"age": "not a number"}})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        messages = e.messages()
        assert any(msg.code == "required" for msg in messages)
    
    # Test 12: Valid nested data
    result = nested_schema.validate({"person": {"name": "Alice", "age": 25}})
    assert result == {"person": {"name": "Alice", "age": 25}}
    
    # Test 13: Field with get_default_value method
    class CustomField(Field):
        def get_default_value(self):
            return "default_value"
    
    schema_custom_default = Schema({
        "field": CustomField()
    })
    result = schema_custom_default.validate({})
    assert result == {"field": "default_value"}
    
    # Test 14: Empty schema
    empty_schema = Schema({})
    result = empty_schema.validate({"extra": "field"})
    assert result == {}
    
    # Test 15: Complex error message aggregation
    complex_schema = Schema({
        "a": Integer(minimum=10),
        "b": String(required=True),
        "c": Integer(maximum=5)
    })
    
    try:
        complex_schema.validate({"a": 5, "c": 10})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) >= 3


# LLM-generated content at query #10
#--------------------------

```python
def test_Schema_serialize():
    from typesystem.fields import String, Integer, Boolean
    
    # Test 1: Serialize None returns None
    schema = Schema(fields={})
    assert schema.serialize(None) is None
    
    # Test 2: Serialize dict with simple fields
    schema = Schema(fields={
        "name": String(),
        "age": Integer(),
        "active": Boolean()
    })
    obj = {"name": "John", "age": 30, "active": True}
    result = schema.serialize(obj)
    assert result == {"name": "John", "age": 30, "active": True}
    
    # Test 3: Serialize object with attributes
    class Person:
        def __init__(self, name, age, active):
            self.name = name
            self.age = age
            self.active = active
    
    person = Person("Alice", 25, False)
    result = schema.serialize(person)
    assert result == {"name": "Alice", "age": 25, "active": False}
    
    # Test 4: Missing fields in dict are skipped
    schema = Schema(fields={
        "name": String(),
        "age": Integer(),
        "city": String()
    })
    obj = {"name": "Bob", "age": 35}
    result = schema.serialize(obj)
    assert result == {"name": "Bob", "age": 35}
    assert "city" not in result
    
    # Test 5: Missing attributes in object are skipped
    class PartialPerson:
        def __init__(self, name):
            self.name = name
    
    partial_person = PartialPerson("Charlie")
    result = schema.serialize(partial_person)
    assert result == {"name": "Charlie"}
    assert "age" not in result
    assert "city" not in result
    
    # Test 6: Nested serialization with Schema fields
    address_schema = Schema(fields={
        "street": String(),
        "zipcode": String()
    })
    person_schema = Schema(fields={
        "name": String(),
        "address": address_schema
    })
    
    obj = {
        "name": "David",
        "address": {"street": "Main St", "zipcode": "12345"}
    }
    result = person_schema.serialize(obj)
    assert result == {
        "name": "David",
        "address": {"street": "Main St", "zipcode": "12345"}
    }
    
    # Test 7: Serialize with read_only field (should be excluded)
    schema = Schema(fields={
        "id": Integer(read_only=True),
        "name": String()
    })
    obj = {"id": 1, "name": "Eve"}
    result = schema.serialize(obj)
    assert result == {"name": "Eve"}
    assert "id" not in result
    
    # Test 8: Empty dict with no fields
    schema = Schema(fields={})
    obj = {}
    result = schema.serialize(obj)
    assert result == {}
    
    # Test 9: Dict with extra fields not in schema are ignored
    schema = Schema(fields={"name": String()})
    obj = {"name": "Frank", "extra": "field"}
    result = schema.serialize(obj)
    assert result == {"name": "Frank"}
    assert "extra" not in result


# LLM-generated content at query #11
#--------------------------

```python
def test_Schema_validate():
    # Test 1: Null value with allow_null=True
    field = Field(allow_null=True)
    schema = Schema(fields={}, allow_null=True)
    assert schema.validate(None) is None

    # Test 2: Null value with allow_null=False
    schema = Schema(fields={}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "null"

    # Test 3: Non-dict value
    schema = Schema(fields={})
    try:
        schema.validate("not a dict")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "type"

    # Test 4: Invalid key type (non-string)
    schema = Schema(fields={})
    try:
        schema.validate({1: "value"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "invalid_key"
        assert e.messages()[0].index == [1]

    # Test 5: Missing required field
    field = Field()
    schema = Schema(fields={"required_field": field})
    try:
        schema.validate({})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "required"
        assert e.messages()[0].index == ["required_field"]

    # Test 6: Read-only field should be ignored during validation
    read_only_field = Field(read_only=True)
    schema = Schema(fields={"read_only": read_only_field})
    result = schema.validate({"read_only": "value"})
    assert "read_only" not in result

    # Test 7: Field with default value when missing
    field_with_default = Field(default="default_value")
    schema = Schema(fields={"field_with_default": field_with_default})
    result = schema.validate({})
    assert result["field_with_default"] == "default_value"

    # Test 8: Valid field validation
    field = Field()
    schema = Schema(fields={"name": field})
    result = schema.validate({"name": "John"})
    assert result["name"] == "John"

    # Test 9: Field validation error
    def failing_validate(value):
        raise ValidationError(messages=[Message(text="Invalid", code="invalid")])

    failing_field = Field()
    failing_field.validate_or_error = lambda x: (None, ValidationError(messages=[Message(text="Invalid", code="invalid")]))
    schema = Schema(fields={"failing": failing_field})
    try:
        schema.validate({"failing": "value"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "invalid"
        assert e.messages()[0].index == ["failing"]

    # Test 10: Multiple errors combined
    required_field = Field()
    another_field = Field()
    schema = Schema(fields={"required": required_field, "another": another_field})
    try:
        schema.validate({"another": 123})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1  # Only required error, another field is valid
        assert e.messages()[0].code == "required"

    # Test 11: Complex nested validation
    nested_field = Field()
    nested_schema = Schema(fields={"nested": nested_field})
    schema = Schema(fields={"outer": nested_schema})
    result = schema.validate({"outer": {"nested": "value"}})
    assert result["outer"]["nested"] == "value"

    # Test 12: Field with allow_null in nested context
    nullable_field = Field(allow_null=True)
    schema = Schema(fields={"nullable": nullable_field})
    result = schema.validate({"nullable": None})
    assert result["nullable"] is None

    # Test 13: Mixed valid and invalid scenarios
    field1 = Field()
    field2 = Field()
    schema = Schema(fields={"field1": field1, "field2": field2})
    
    # This should succeed with both fields
    result = schema.validate({"field1": "value1", "field2": "value2"})
    assert result["field1"] == "value1"
    assert result["field2"] == "value2"


# LLM-generated content at query #12
#--------------------------

```python
def test_Reference_validate():
    # Test basic validation with valid value
    definitions = Definitions()
    field = Field()
    definitions["test_ref"] = field
    reference = Reference(to="test_ref", definitions=definitions)
    
    # Test that validation delegates to target field
    mock_value = {"key": "value"}
    field.validate = lambda x: x
    result = reference.validate(mock_value)
    assert result == mock_value
    
    # Test null value with allow_null=True
    reference_allow_null = Reference(to="test_ref", definitions=definitions, allow_null=True)
    assert reference_allow_null.validate(None) is None
    
    # Test null value with allow_null=False (default)
    with pytest.raises(ValidationError) as exc_info:
        reference.validate(None)
    assert exc_info.value.messages[0].code == "null"
    
    # Test that validation error from target field is propagated
    definitions2 = Definitions()
    failing_field = Field()
    definitions2["failing_ref"] = failing_field
    reference2 = Reference(to="failing_ref", definitions=definitions2)
    
    def failing_validate(value):
        raise ValidationError(messages=[Message(text="Invalid", code="invalid")])
    
    failing_field.validate = failing_validate
    
    with pytest.raises(ValidationError) as exc_info:
        reference2.validate({"key": "value"})
    assert exc_info.value.messages[0].code == "invalid"
    
    # Test with non-existent reference (should raise KeyError when accessing target)
    reference3 = Reference(to="non_existent", definitions=definitions)
    with pytest.raises(KeyError):
        reference3.validate({"key": "value"})


# LLM-generated content at query #13
#--------------------------

```python
def test_Schema_validate():
    # Test basic validation with simple fields
    from typesystem.fields import String, Integer
    
    schema = Schema(fields={"name": String(), "age": Integer()})
    result = schema.validate({"name": "John", "age": 30})
    assert result == {"name": "John", "age": 30}
    
    # Test null handling with allow_null=False (default)
    try:
        schema.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.messages()[0].code == "null"
    
    # Test null handling with allow_null=True
    schema_with_null = Schema(fields={"name": String()}, allow_null=True)
    result = schema_with_null.validate(None)
    assert result is None
    
    # Test type validation - non-dict input
    try:
        schema.validate("not a dict")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.messages()[0].code == "type"
    
    # Test invalid key type (non-string keys)
    try:
        schema.validate({1: "value", "name": "John"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert any(msg.code == "invalid_key" for msg in e.messages())
    
    # Test required fields
    required_schema = Schema(fields={"name": String(), "age": Integer()})
    try:
        required_schema.validate({"name": "John"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert any(msg.code == "required" and msg.index == ["age"] for msg in e.messages())
    
    # Test field with default value
    from typesystem.fields import String, Integer
    
    field_with_default = String(default="default_name")
    schema_with_default = Schema(fields={"name": field_with_default, "age": Integer()})
    result = schema_with_default.validate({"age": 30})
    assert result == {"name": "default_name", "age": 30}
    
    # Test read_only field
    read_only_field = String(read_only=True)
    schema_read_only = Schema(fields={"name": read_only_field, "age": Integer()})
    result = schema_read_only.validate({"age": 30})
    assert result == {"age": 30}
    assert "name" not in result
    
    # Test nested validation errors
    nested_schema = Schema(fields={
        "name": String(max_length=5),
        "age": Integer(minimum=0)
    })
    
    try:
        nested_schema.validate({"name": "Too Long Name", "age": -5})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 2
        assert any("max_length" in msg.code for msg in messages)
        assert any("minimum" in msg.code for msg in messages)
    
    # Test complex nested structure
    address_schema = Schema(fields={
        "street": String(),
        "city": String()
    })
    
    person_schema = Schema(fields={
        "name": String(),
        "address": address_schema
    })
    
    result = person_schema.validate({
        "name": "John",
        "address": {"street": "123 Main St", "city": "Anytown"}
    })
    assert result["name"] == "John"
    assert result["address"]["street"] == "123 Main St"
    assert result["address"]["city"] == "Anytown"
    
    # Test with empty dict when all fields have defaults
    all_defaults_schema = Schema(fields={
        "name": String(default="unknown"),
        "age": Integer(default=0)
    })
    result = all_defaults_schema.validate({})
    assert result == {"name": "unknown", "age": 0}
    
    # Test error message indices for nested errors
    try:
        person_schema.validate({
            "name": "John",
            "address": {"street": 123, "city": "Anytown"}
        })
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        messages = e.messages()
        assert any(msg.index == ["address", "street"] for msg in messages)


# LLM-generated content at query #14
#--------------------------

```python
def test_Reference_validate():
    from typesystem.fields import String, Integer
    from typesystem.base import ValidationError
    
    # Test 1: Basic validation with valid value
    definitions = Definitions()
    schema = Schema({"name": String(), "age": Integer()})
    definitions["Person"] = schema
    reference = Reference(to="Person", definitions=definitions)
    
    result = reference.validate({"name": "John", "age": 30})
    assert result == {"name": "John", "age": 30}
    
    # Test 2: Validation with null when allow_null=True
    reference_allow_null = Reference(to="Person", definitions=definitions, allow_null=True)
    result = reference_allow_null.validate(None)
    assert result is None
    
    # Test 3: Validation with null when allow_null=False (default)
    reference_no_null = Reference(to="Person", definitions=definitions)
    try:
        reference_no_null.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "null"
    
    # Test 4: Validation with invalid data (should propagate target's validation error)
    try:
        reference.validate({"name": "John", "age": "not_a_number"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert "age" in str(e.messages()[0].index)
    
    # Test 5: Validation with missing required field
    try:
        reference.validate({"name": "John"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "required"
    
    # Test 6: Validation with nested schema
    definitions2 = Definitions()
    nested_schema = Schema({
        "address": String(),
        "city": String()
    })
    definitions2["Address"] = nested_schema
    person_schema = Schema({
        "name": String(),
        "address": Reference(to="Address", definitions=definitions2)
    })
    definitions2["Person"] = person_schema
    reference2 = Reference(to="Person", definitions=definitions2)
    
    result = reference2.validate({
        "name": "Alice",
        "address": {"address": "123 Main St", "city": "Boston"}
    })
    assert result["name"] == "Alice"
    assert result["address"] == {"address": "123 Main St", "city": "Boston"}
    
    # Test 7: Validation with non-existent definition
    empty_definitions = Definitions()
    bad_reference = Reference(to="NonExistent", definitions=empty_definitions)
    try:
        bad_reference.validate({"name": "John"})
        assert False, "Should have raised KeyError"
    except KeyError:
        pass


# LLM-generated content at query #15
#--------------------------

```python
def test_Schema_validate():
    # Test basic validation with simple fields
    from typesystem.fields import String, Integer
    
    schema = Schema(fields={
        "name": String(max_length=10),
        "age": Integer(minimum=0, maximum=150)
    })
    
    # Test valid input
    result = schema.validate({"name": "John", "age": 30})
    assert result == {"name": "John", "age": 30}
    
    # Test null value when not allowed
    try:
        schema.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "null"
    
    # Test null value when allowed
    schema_with_null = Schema(
        fields={"name": String()},
        allow_null=True
    )
    assert schema_with_null.validate(None) is None
    
    # Test non-dict input
    try:
        schema.validate("not a dict")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.messages()[0].code == "type"
    
    # Test invalid key type
    try:
        schema.validate({1: "value", "name": "John"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert any(msg.code == "invalid_key" for msg in e.messages())
        assert any(msg.index == [1] for msg in e.messages())
    
    # Test required fields
    try:
        schema.validate({"name": "John"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert any(msg.code == "required" for msg in e.messages())
        assert any(msg.index == ["age"] for msg in e.messages())
    
    # Test field validation errors
    try:
        schema.validate({"name": "VeryLongName", "age": 200})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 2
        assert any("max_length" in msg.code for msg in messages)
        assert any("maximum" in msg.code for msg in messages)
    
    # Test read_only field
    schema_with_readonly = Schema(fields={
        "id": Integer(read_only=True),
        "name": String()
    })
    result = schema_with_readonly.validate({"name": "Alice"})
    assert "id" not in result
    assert result["name"] == "Alice"
    
    # Test field with default value
    schema_with_default = Schema(fields={
        "name": String(default="Unknown"),
        "age": Integer()
    })
    result = schema_with_default.validate({"age": 25})
    assert result["name"] == "Unknown"
    assert result["age"] == 25
    
    # Test nested validation errors with prefixes
    nested_schema = Schema(fields={
        "address": Schema(fields={
            "street": String(required=True),
            "city": String()
        })
    })
    
    try:
        nested_schema.validate({"address": {"city": "NYC"}})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 1
        assert messages[0].code == "required"
        assert messages[0].index == ["address", "street"]
    
    # Test multiple validation errors
    complex_schema = Schema(fields={
        "name": String(required=True),
        "email": String(required=True),
        "age": Integer(minimum=18, required=True)
    })
    
    try:
        complex_schema.validate({"age": 15})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 3
        codes = [msg.code for msg in messages]
        assert "required" in codes
        assert "minimum" in codes
    
    # Test with empty dict when no required fields
    optional_schema = Schema(fields={
        "optional": String(allow_null=True)
    })
    result = optional_schema.validate({})
    assert result == {}
    
    # Test with Mapping type (not just dict)
    from collections.abc import Mapping
    
    class CustomMapping(Mapping):
        def __init__(self, data):
            self._data = data
        
        def __getitem__(self, key):
            return self._data[key]
        
        def __iter__(self):
            return iter(self._data)
        
        def __len__(self):
            return len(self._data)
    
    mapping_data = CustomMapping({"name": "Bob", "age": 40})
    result = schema.validate(mapping_data)
    assert result == {"name": "Bob", "age": 40}


# LLM-generated content at query #16
#--------------------------

```python
def test_Reference_validate():
    from typesystem.fields import Field
    
    # Test 1: Valid value with non-null target
    definitions = Definitions()
    target_field = Field()
    target_field.validate = lambda x: x * 2
    definitions["target"] = target_field
    
    ref = Reference(to="target", definitions=definitions)
    result = ref.validate(5)
    assert result == 10
    
    # Test 2: Null value with allow_null=True
    ref = Reference(to="target", definitions=definitions, allow_null=True)
    result = ref.validate(None)
    assert result is None
    
    # Test 3: Null value with allow_null=False (default)
    ref = Reference(to="target", definitions=definitions)
    try:
        ref.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "null"
    
    # Test 4: Target validation error propagation
    definitions = Definitions()
    target_field = Field()
    target_field.validate = lambda x: (_ for _ in ()).throw(
        ValidationError(messages=[Message(text="Invalid", code="invalid")])
    )
    definitions["target"] = target_field
    
    ref = Reference(to="target", definitions=definitions)
    try:
        ref.validate("bad_value")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "invalid"
    
    # Test 5: Non-existent target definition
    definitions = Definitions()
    ref = Reference(to="missing", definitions=definitions)
    try:
        ref.validate(42)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass


# LLM-generated content at query #17
#--------------------------

```python
def test_Reference_validate():
    # Test 1: Valid value with non-null target
    definitions = Definitions()
    field = Field()
    definitions["target_field"] = field
    reference = Reference(to="target_field", definitions=definitions)
    
    # Mock target field's validate method
    mock_value = {"key": "value"}
    field.validate = lambda x: x if x == mock_value else None
    result = reference.validate(mock_value)
    assert result == mock_value

    # Test 2: Null value with allow_null=True
    reference = Reference(to="target_field", definitions=definitions, allow_null=True)
    result = reference.validate(None)
    assert result is None

    # Test 3: Null value with allow_null=False (default)
    reference = Reference(to="target_field", definitions=definitions)
    try:
        reference.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "null"

    # Test 4: Valid value with nested schema target
    schema = Schema(fields={"name": Field()})
    definitions["person"] = schema
    reference = Reference(to="person", definitions=definitions)
    
    valid_data = {"name": "John"}
    schema.validate = lambda x: x if x == valid_data else None
    result = reference.validate(valid_data)
    assert result == valid_data

    # Test 5: Invalid value (target raises ValidationError)
    definitions["strict_field"] = Field()
    reference = Reference(to="strict_field", definitions=definitions)
    
    def raising_validate(value):
        raise ValidationError(messages=[Message(text="Invalid", code="invalid")])
    
    definitions["strict_field"].validate = raising_validate
    
    try:
        reference.validate("invalid")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "invalid"

    # Test 6: Reference to non-existent definition
    reference = Reference(to="missing", definitions=definitions)
    try:
        reference.validate("anything")
        assert False, "Should have raised KeyError"
    except KeyError:
        pass

    # Test 7: allow_null with non-null value
    reference = Reference(to="target_field", definitions=definitions, allow_null=True)
    result = reference.validate("not null")
    assert result == "not null"


# LLM-generated content at query #18
#--------------------------

```python
def test_Schema_validate():
    # Test 1: Null value with allow_null=True
    field = Field(allow_null=True)
    schema = Schema(fields={}, allow_null=True)
    assert schema.validate(None) is None

    # Test 2: Null value with allow_null=False
    schema = Schema(fields={}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "null"

    # Test 3: Non-dict value
    schema = Schema(fields={})
    try:
        schema.validate("not a dict")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "type"

    # Test 4: Invalid key type (non-string)
    schema = Schema(fields={})
    try:
        schema.validate({1: "value"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "invalid_key"
        assert e.messages()[0].index == [1]

    # Test 5: Missing required field
    field = Field()
    schema = Schema(fields={"required_field": field})
    try:
        schema.validate({})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "required"
        assert e.messages()[0].index == ["required_field"]

    # Test 6: Read-only field should be ignored
    read_only_field = Field(read_only=True)
    schema = Schema(fields={"read_only": read_only_field})
    result = schema.validate({"other": "value"})
    assert "read_only" not in result

    # Test 7: Field with default value when missing
    field_with_default = Field(default="default_value")
    schema = Schema(fields={"field_with_default": field_with_default})
    result = schema.validate({})
    assert result["field_with_default"] == "default_value"

    # Test 8: Valid field validation
    field = Field()
    schema = Schema(fields={"name": field})
    result = schema.validate({"name": "John"})
    assert result["name"] == "John"

    # Test 9: Field validation error
    field = Field()
    schema = Schema(fields={"age": field})
    try:
        schema.validate({"age": None})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert "age" in str(e.messages()[0].index)

    # Test 10: Multiple validation errors
    required_field = Field()
    schema = Schema(fields={"required": required_field})
    try:
        schema.validate({1: "invalid key"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        codes = [msg.code for msg in e.messages()]
        assert "invalid_key" in codes
        assert "required" in codes

    # Test 11: Nested validation with valid data
    nested_field = Field()
    nested_schema = Schema(fields={"nested": nested_field})
    schema = Schema(fields={"outer": nested_schema})
    result = schema.validate({"outer": {"nested": "value"}})
    assert result["outer"]["nested"] == "value"

    # Test 12: Complex scenario with mixed valid/invalid
    field1 = Field()
    field2 = Field(default="default")
    field3 = Field(read_only=True)
    schema = Schema(fields={
        "required": field1,
        "with_default": field2,
        "read_only": field3
    })
    
    result = schema.validate({"required": "value"})
    assert result["required"] == "value"
    assert result["with_default"] == "default"
    assert "read_only" not in result

    # Test 13: Empty schema with valid dict
    schema = Schema(fields={})
    result = schema.validate({"extra": "should be ignored"})
    assert result == {}

    # Test 14: Field with allow_null=True
    nullable_field = Field(allow_null=True)
    schema = Schema(fields={"nullable": nullable_field})
    result = schema.validate({"nullable": None})
    assert result["nullable"] is None


# LLM-generated content at query #19
#--------------------------

```python
def test_Schema_validate():
    # Test basic validation with simple fields
    from typesystem.fields import String, Integer
    
    schema = Schema({
        "name": String(max_length=10),
        "age": Integer(minimum=0, maximum=150)
    })
    
    # Test valid input
    result = schema.validate({"name": "John", "age": 30})
    assert result == {"name": "John", "age": 30}
    
    # Test null handling with allow_null=False (default)
    try:
        schema.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "null"
    
    # Test null handling with allow_null=True
    schema_with_null = Schema({
        "name": String(max_length=10),
        "age": Integer(minimum=0, maximum=150)
    }, allow_null=True)
    
    assert schema_with_null.validate(None) is None
    
    # Test type validation - not a dict/mapping
    try:
        schema.validate("not an object")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "type"
    
    # Test invalid key type (non-string keys)
    try:
        schema.validate({1: "value", "name": "John"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "invalid_key"
        assert e.messages()[0].index == [1]
    
    # Test required fields
    try:
        schema.validate({"name": "John"})  # Missing age
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "required"
        assert e.messages()[0].index == ["age"]
    
    # Test field validation errors
    try:
        schema.validate({"name": "John" * 10, "age": 200})  # Name too long, age too high
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 2
    
    # Test read_only fields are ignored during validation
    from typesystem.fields import String, Integer
    
    schema_with_readonly = Schema({
        "name": String(max_length=10),
        "id": Integer(read_only=True),
        "age": Integer(minimum=0)
    })
    
    # id should be ignored even if present
    result = schema_with_readonly.validate({"name": "John", "id": 123, "age": 30})
    assert result == {"name": "John", "age": 30}
    assert "id" not in result
    
    # Test fields with default values
    from typesystem.fields import String, Integer
    
    schema_with_defaults = Schema({
        "name": String(max_length=10),
        "age": Integer(default=25),
        "active": String(default="yes", allow_null=True)
    })
    
    # age and active should get default values when not provided
    result = schema_with_defaults.validate({"name": "John"})
    assert result == {"name": "John", "age": 25, "active": "yes"}
    
    # Test nested validation with child schema errors
    nested_schema = Schema({
        "name": String(max_length=10),
        "address": Schema({
            "street": String(),
            "city": String()
        })
    })
    
    try:
        nested_schema.validate({
            "name": "John",
            "address": {"street": "Main St"}  # Missing city
        })
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        # Should have error for missing city in address
        messages = e.messages()
        assert any(m.code == "required" and m.index == ["address", "city"] for m in messages)
    
    # Test successful nested validation
    result = nested_schema.validate({
        "name": "John",
        "address": {"street": "Main St", "city": "Springfield"}
    })
    assert result["name"] == "John"
    assert result["address"]["street"] == "Main St"
    assert result["address"]["city"] == "Springfield"
    
    # Test with Mapping type (not just dict)
    from collections.abc import Mapping
    
    class CustomMapping(Mapping):
        def __init__(self, data):
            self._data = data
        
        def __getitem__(self, key):
            return self._data[key]
        
        def __iter__(self):
            return iter(self._data)
        
        def __len__(self):
            return len(self._data)
    
    custom_data = CustomMapping({"name": "Jane", "age": 25})
    result = schema.validate(custom_data)
    assert result == {"name": "Jane", "age": 25}
    
    # Test multiple invalid keys
    try:
        schema.validate({1: "value", 2.5: "value2", "name": "John"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 2  # Two invalid keys
    
    # Test complex error message aggregation
    complex_schema = Schema({
        "field1": String(required=True),
        "field2": Integer(minimum=10, maximum=20),
        "field3": String(max_length=5)
    })
    
    try:
        complex_schema.validate({
            "field2": 5,  # Too low
            "field3": "too long value"  # Too long
        })
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 3  # Missing field1, field2 too low, field3 too long


# LLM-generated content at query #20
#--------------------------

```python
def test_Reference_validate():
    from typesystem.fields import Field
    
    class MockField(Field):
        def __init__(self, allow_null=False):
            super().__init__(allow_null=allow_null)
            
        def validate(self, value):
            if value is None and self.allow_null:
                return None
            elif value is None:
                raise self.validation_error("null")
            return value
    
    definitions = Definitions()
    mock_field = MockField()
    definitions["test_ref"] = mock_field
    
    reference = Reference(to="test_ref", definitions=definitions)
    
    assert reference.validate("test_value") == "test_value"
    
    reference_nullable = Reference(to="test_ref", definitions=definitions, allow_null=True)
    assert reference_nullable.validate(None) is None
    
    try:
        reference.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "null"
    
    definitions["other_ref"] = MockField(allow_null=True)
    reference_other = Reference(to="other_ref", definitions=definitions)
    assert reference_other.validate(None) is None
    
    try:
        Reference(to="missing_ref", definitions=definitions).validate("value")
        assert False, "Should have raised KeyError"
    except KeyError:
        pass


# LLM-generated content at query #21
#--------------------------

```python
def test_Schema_validate():
    # Test basic validation with valid data
    field1 = Field()
    field2 = Field()
    schema = Schema(fields={"name": field1, "age": field2})
    
    result = schema.validate({"name": "John", "age": 30})
    assert result == {"name": "John", "age": 30}
    
    # Test null value with allow_null=False (default)
    try:
        schema.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "null"
    
    # Test null value with allow_null=True
    schema_allow_null = Schema(fields={"name": field1, "age": field2}, allow_null=True)
    result = schema_allow_null.validate(None)
    assert result is None
    
    # Test non-dict value
    try:
        schema.validate("not a dict")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "type"
    
    # Test invalid key type (non-string key)
    try:
        schema.validate({1: "value", "name": "John"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "invalid_key"
        assert e.messages[0].index == [1]
    
    # Test required fields
    required_field = Field()
    schema_with_required = Schema(fields={"required_field": required_field})
    
    try:
        schema_with_required.validate({})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "required"
        assert e.messages[0].index == ["required_field"]
    
    # Test read_only fields are ignored during validation
    read_only_field = Field(read_only=True)
    schema_with_readonly = Schema(fields={"read_only": read_only_field, "normal": field1})
    
    result = schema_with_readonly.validate({"normal": "value"})
    assert result == {"normal": "value"}
    assert "read_only" not in result
    
    # Test fields with default values
    field_with_default = Field(default="default_value")
    schema_with_default = Schema(fields={"field": field_with_default})
    
    result = schema_with_default.validate({})
    assert result == {"field": "default_value"}
    
    # Test nested validation errors
    nested_field = Field()
    nested_schema = Schema(fields={"nested": nested_field})
    
    # Mock nested field to raise validation error
    class ErrorField(Field):
        def validate(self, value):
            raise ValidationError(messages=[Message(text="Invalid", code="invalid", index=[])])
    
    error_field = ErrorField()
    schema_with_error = Schema(fields={"error_field": error_field})
    
    try:
        schema_with_error.validate({"error_field": "bad_value"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "invalid"
    
    # Test multiple validation errors
    required_field1 = Field()
    required_field2 = Field()
    schema_multiple_errors = Schema(fields={
        "field1": required_field1,
        "field2": required_field2
    })
    
    try:
        schema_multiple_errors.validate({})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 2
        codes = {msg.code for msg in e.messages}
        assert codes == {"required"}
    
    # Test with Mapping type (not just dict)
    from collections.abc import Mapping
    
    class TestMapping(Mapping):
        def __init__(self, data):
            self._data = data
        
        def __getitem__(self, key):
            return self._data[key]
        
        def __iter__(self):
            return iter(self._data)
        
        def __len__(self):
            return len(self._data)
    
    mapping_data = TestMapping({"name": "John", "age": 30})
    result = schema.validate(mapping_data)
    assert result == {"name": "John", "age": 30}


# LLM-generated content at query #22
#--------------------------

```python
def test_Reference_validate():
    from typesystem.fields import Field
    
    # Create a mock field for testing
    class MockField(Field):
        def validate(self, value):
            if value == "invalid":
                raise ValidationError("Invalid value")
            return f"validated_{value}"
    
    # Test 1: Basic validation with non-null value
    definitions = Definitions()
    mock_field = MockField()
    definitions["test_ref"] = mock_field
    reference = Reference(to="test_ref", definitions=definitions)
    
    result = reference.validate("test_value")
    assert result == "validated_test_value"
    
    # Test 2: Null value with allow_null=True
    reference_allow_null = Reference(to="test_ref", definitions=definitions, allow_null=True)
    result = reference_allow_null.validate(None)
    assert result is None
    
    # Test 3: Null value with allow_null=False (default)
    reference_no_null = Reference(to="test_ref", definitions=definitions)
    try:
        reference_no_null.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"
    
    # Test 4: Validation error propagation from target field
    try:
        reference.validate("invalid")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert str(e) == "Invalid value"
    
    # Test 5: Non-existent reference
    reference_bad = Reference(to="nonexistent", definitions=definitions)
    try:
        reference_bad.validate("test")
        assert False, "Should have raised KeyError"
    except KeyError:
        pass
    
    # Test 6: Complex nested validation
    class NestedSchema(Schema):
        pass
    
    nested_field = NestedSchema(fields={"name": Field()})
    definitions["nested"] = nested_field
    reference_nested = Reference(to="nested", definitions=definitions)
    
    result = reference_nested.validate({"name": "test"})
    assert result == {"name": "test"}
    
    # Test 7: Validate with target that has its own validation logic
    class CustomField(Field):
        def validate(self, value):
            if not isinstance(value, int):
                raise ValidationError("Must be integer")
            return value * 2
    
    definitions["custom"] = CustomField()
    reference_custom = Reference(to="custom", definitions=definitions)
    
    result = reference_custom.validate(5)
    assert result == 10
    
    try:
        reference_custom.validate("not_int")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert "Must be integer" in str(e)


# LLM-generated content at query #23
#--------------------------

```python
def test_Reference_validate():
    from typesystem.fields import String, Integer
    from typesystem.base import ValidationError
    
    # Test 1: Valid value with non-null reference
    definitions = Definitions()
    schema = Schema({"name": String(), "age": Integer()})
    definitions["Person"] = schema
    reference = Reference(to="Person", definitions=definitions)
    
    valid_data = {"name": "John", "age": 30}
    result = reference.validate(valid_data)
    assert result == valid_data
    
    # Test 2: Null value with allow_null=False (default)
    reference_no_null = Reference(to="Person", definitions=definitions)
    try:
        reference_no_null.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.messages()[0].code == "null"
    
    # Test 3: Null value with allow_null=True
    reference_with_null = Reference(to="Person", definitions=definitions, allow_null=True)
    result = reference_with_null.validate(None)
    assert result is None
    
    # Test 4: Invalid value according to target schema
    invalid_data = {"name": 123, "age": "thirty"}
    try:
        reference.validate(invalid_data)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) > 0
    
    # Test 5: Valid value with nested schema
    address_schema = Schema({"street": String(), "city": String()})
    definitions["Address"] = address_schema
    address_reference = Reference(to="Address", definitions=definitions)
    
    valid_address = {"street": "Main St", "city": "New York"}
    result = address_reference.validate(valid_address)
    assert result == valid_address
    
    # Test 6: Reference to non-existent definition
    bad_reference = Reference(to="NonExistent", definitions=definitions)
    try:
        bad_reference.validate({"test": "data"})
        assert False, "Should have raised KeyError when accessing target"
    except KeyError:
        pass
    
    # Test 7: Valid value with complex nested structure
    complex_schema = Schema({
        "id": Integer(),
        "person": Reference(to="Person", definitions=definitions)
    })
    definitions["Complex"] = complex_schema
    complex_reference = Reference(to="Complex", definitions=definitions)
    
    complex_data = {
        "id": 1,
        "person": {"name": "Alice", "age": 25}
    }
    result = complex_reference.validate(complex_data)
    assert result == complex_data


# LLM-generated content at query #24
#--------------------------

```python
def test_Schema_validate():
    # Test basic validation with valid data
    field1 = Field()
    field2 = Field()
    schema = Schema({"name": field1, "age": field2})
    
    result = schema.validate({"name": "John", "age": 30})
    assert result == {"name": "John", "age": 30}
    
    # Test with null value when allow_null is False (default)
    schema = Schema({"name": field1})
    try:
        schema.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "null"
    
    # Test with null value when allow_null is True
    schema = Schema({"name": field1}, allow_null=True)
    result = schema.validate(None)
    assert result is None
    
    # Test with non-dict value
    schema = Schema({"name": field1})
    try:
        schema.validate("not a dict")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "type"
    
    # Test with non-string keys
    schema = Schema({"name": field1})
    try:
        schema.validate({1: "value"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "invalid_key"
        assert e.messages[0].index == [1]
    
    # Test required fields
    field_required = Field()
    schema = Schema({"name": field_required, "optional": Field()})
    try:
        schema.validate({"optional": "value"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "required"
        assert e.messages[0].index == ["name"]
    
    # Test read_only fields are ignored during validation
    read_only_field = Field(read_only=True)
    schema = Schema({"name": Field(), "id": read_only_field})
    result = schema.validate({"name": "John", "id": 123})
    assert result == {"name": "John"}
    
    # Test fields with default values
    field_with_default = Field(default="default_value")
    schema = Schema({"name": field_with_default})
    result = schema.validate({})
    assert result == {"name": "default_value"}
    
    # Test nested validation errors
    nested_field = Field()
    nested_field.validate = lambda x: (_ for _ in ()).throw(
        ValidationError(messages=[Message(text="Invalid", code="invalid", index=[])])
    )
    schema = Schema({"nested": nested_field})
    try:
        schema.validate({"nested": "value"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "invalid"
        assert e.messages[0].index == ["nested"]
    
    # Test multiple validation errors
    field1 = Field()
    field2 = Field()
    schema = Schema({"field1": field1, "field2": field2})
    
    # Create fields that will fail validation
    field1.validate = lambda x: (_ for _ in ()).throw(
        ValidationError(messages=[Message(text="Error 1", code="error1", index=[])])
    )
    field2.validate = lambda x: (_ for _ in ()).throw(
        ValidationError(messages=[Message(text="Error 2", code="error2", index=[])])
    )
    
    try:
        schema.validate({"field1": "val1", "field2": "val2"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 2
        codes = {msg.code for msg in e.messages}
        assert codes == {"error1", "error2"}
    
    # Test with Mapping type (not just dict)
    from collections.abc import Mapping
    
    class TestMapping(Mapping):
        def __init__(self, data):
            self._data = data
        
        def __getitem__(self, key):
            return self._data[key]
        
        def __len__(self):
            return len(self._data)
        
        def __iter__(self):
            return iter(self._data)
    
    schema = Schema({"name": Field()})
    mapping = TestMapping({"name": "John"})
    result = schema.validate(mapping)
    assert result == {"name": "John"}


# LLM-generated content at query #25
#--------------------------

```python
def test_Reference_validate():
    from typesystem.fields import String, Integer
    
    # Test 1: Valid value with non-null target
    definitions = Definitions()
    definitions["Person"] = Schema({"name": String(), "age": Integer()})
    ref = Reference(to="Person", definitions=definitions)
    
    valid_data = {"name": "John", "age": 30}
    result = ref.validate(valid_data)
    assert result == valid_data
    
    # Test 2: Null value with allow_null=False (default)
    ref_no_null = Reference(to="Person", definitions=definitions)
    try:
        ref_no_null.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"
    
    # Test 3: Null value with allow_null=True
    ref_with_null = Reference(to="Person", definitions=definitions, allow_null=True)
    result = ref_with_null.validate(None)
    assert result is None
    
    # Test 4: Invalid value according to target schema
    invalid_data = {"name": "John", "age": "not_an_integer"}
    try:
        ref.validate(invalid_data)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages) > 0
    
    # Test 5: Valid value with nested schema target
    definitions["Address"] = Schema({
        "street": String(),
        "city": String(),
        "zip": String()
    })
    address_ref = Reference(to="Address", definitions=definitions)
    
    address_data = {"street": "123 Main", "city": "Anytown", "zip": "12345"}
    result = address_ref.validate(address_data)
    assert result == address_data
    
    # Test 6: Reference to non-existent definition
    bad_ref = Reference(to="NonExistent", definitions=definitions)
    try:
        bad_ref.validate({"test": "data"})
        assert False, "Should have raised KeyError when accessing target"
    except KeyError:
        pass
    
    # Test 7: Valid value with required fields in target
    definitions["RequiredFields"] = Schema({
        "id": Integer(),
        "name": String(allow_null=True)
    })
    required_ref = Reference(to="RequiredFields", definitions=definitions)
    
    valid_required = {"id": 1, "name": None}
    result = required_ref.validate(valid_required)
    assert result == valid_required
    
    # Test 8: Invalid value missing required field
    missing_required = {"name": "Test"}
    try:
        required_ref.validate(missing_required)
        assert False, "Should have raised ValidationError for missing required field"
    except ValidationError as e:
        assert any(msg.code == "required" for msg in e.messages)


# LLM-generated content at query #26
#--------------------------

```python
def test_Reference_validate():
    from typesystem.fields import Field
    from typesystem.base import ValidationError
    
    # Test 1: Valid value with non-null target
    definitions = Definitions()
    field = Field()
    field.validate = lambda x: x * 2
    definitions["target"] = field
    ref = Reference(to="target", definitions=definitions)
    assert ref.validate(5) == 10
    
    # Test 2: Null value with allow_null=True
    ref = Reference(to="target", definitions=definitions, allow_null=True)
    assert ref.validate(None) is None
    
    # Test 3: Null value with allow_null=False (default)
    ref = Reference(to="target", definitions=definitions)
    try:
        ref.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.messages()[0].code == "null"
    
    # Test 4: Target validation error propagation
    class FailingField(Field):
        def validate(self, value):
            raise ValidationError("Invalid value")
    
    failing_field = FailingField()
    definitions["failing"] = failing_field
    ref = Reference(to="failing", definitions=definitions)
    try:
        ref.validate("anything")
        assert False, "Should have raised ValidationError"
    except ValidationError:
        pass
    
    # Test 5: Non-null value with target returning same value
    class IdentityField(Field):
        def validate(self, value):
            return value
    
    identity_field = IdentityField()
    definitions["identity"] = identity_field
    ref = Reference(to="identity", definitions=definitions)
    test_data = {"key": "value"}
    assert ref.validate(test_data) == test_data
    
    # Test 6: Missing definition
    ref = Reference(to="missing", definitions=definitions)
    try:
        ref.validate("test")
        assert False, "Should have raised KeyError"
    except KeyError:
        pass


# LLM-generated content at query #27
#--------------------------

```python
def test_Reference_validate():
    from typesystem.fields import String, Integer
    from typesystem.base import ValidationError
    
    # Test 1: Valid value with non-null reference
    definitions = Definitions()
    schema = Schema({"name": String(), "age": Integer()})
    definitions["Person"] = schema
    reference = Reference(to="Person", definitions=definitions)
    
    value = {"name": "John", "age": 30}
    result = reference.validate(value)
    assert result == {"name": "John", "age": 30}
    
    # Test 2: Null value with allow_null=False (default)
    reference_no_null = Reference(to="Person", definitions=definitions)
    try:
        reference_no_null.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.messages()[0].code == "null"
    
    # Test 3: Null value with allow_null=True
    reference_with_null = Reference(to="Person", definitions=definitions, allow_null=True)
    result = reference_with_null.validate(None)
    assert result is None
    
    # Test 4: Invalid value (fails target validation)
    try:
        reference.validate({"name": "John", "age": "not_a_number"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) > 0
    
    # Test 5: Reference to non-existent definition
    bad_reference = Reference(to="NonExistent", definitions=definitions)
    try:
        bad_reference.validate({"name": "John"})
        assert False, "Should have raised KeyError"
    except KeyError:
        pass
    
    # Test 6: Valid value with nested schema
    address_schema = Schema({"street": String(), "city": String()})
    definitions["Address"] = address_schema
    address_ref = Reference(to="Address", definitions=definitions)
    
    address_value = {"street": "123 Main St", "city": "Springfield"}
    result = address_ref.validate(address_value)
    assert result == {"street": "123 Main St", "city": "Springfield"}


# LLM-generated content at query #28
#--------------------------

```python
def test_Reference_validate():
    from typesystem.fields import Integer, String
    
    # Test 1: Valid value with non-null reference
    definitions = Definitions()
    definitions["Person"] = Schema({"name": String(), "age": Integer()})
    ref = Reference(to="Person", definitions=definitions)
    
    result = ref.validate({"name": "John", "age": 30})
    assert result == {"name": "John", "age": 30}
    
    # Test 2: Null value with allow_null=False (default)
    ref_no_null = Reference(to="Person", definitions=definitions)
    try:
        ref_no_null.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.messages()[0].code == "null"
    
    # Test 3: Null value with allow_null=True
    ref_with_null = Reference(to="Person", definitions=definitions, allow_null=True)
    result = ref_with_null.validate(None)
    assert result is None
    
    # Test 4: Invalid value according to target schema
    try:
        ref.validate({"name": "John", "age": "thirty"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages()) > 0
    
    # Test 5: Valid value with nested schema reference
    definitions["Address"] = Schema({"street": String(), "city": String()})
    address_ref = Reference(to="Address", definitions=definitions)
    
    result = address_ref.validate({"street": "Main St", "city": "Boston"})
    assert result == {"street": "Main St", "city": "Boston"}
    
    # Test 6: Reference to non-existent definition
    bad_ref = Reference(to="NonExistent", definitions=definitions)
    try:
        bad_ref.validate({"test": "value"})
        assert False, "Should have raised KeyError when accessing target"
    except KeyError:
        pass
    
    # Test 7: Valid value with complex nested reference
    definitions["Profile"] = Schema({
        "person": Reference(to="Person", definitions=definitions),
        "address": Reference(to="Address", definitions=definitions)
    })
    profile_ref = Reference(to="Profile", definitions=definitions)
    
    valid_data = {
        "person": {"name": "Alice", "age": 25},
        "address": {"street": "Oak Ave", "city": "Seattle"}
    }
    result = profile_ref.validate(valid_data)
    assert result == valid_data


