####################################################################
#    TEST GENERATION BEGINS (CODAMOSA + claude-haiku-4-5 t=0.8)    #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_Schema_serialize():
    # Test serialize with None value
    field1 = Field()
    schema = Schema(fields={"name": field1})
    assert schema.serialize(None) is None

    # Test serialize with dict object
    field1 = Field()
    field2 = Field()
    schema = Schema(fields={"name": field1, "age": field2})
    obj = {"name": "John", "age": 30}
    result = schema.serialize(obj)
    assert result == {"name": "John", "age": 30}

    # Test serialize with object attributes
    class Person:
        def __init__(self, name, age):
            self.name = name
            self.age = age
    
    field1 = Field()
    field2 = Field()
    schema = Schema(fields={"name": field1, "age": field2})
    obj = Person("Jane", 25)
    result = schema.serialize(obj)
    assert result == {"name": "Jane", "age": 25}

    # Test serialize with missing keys/attributes
    field1 = Field()
    field2 = Field()
    schema = Schema(fields={"name": field1, "age": field2, "email": field1})
    obj = {"name": "Bob", "age": 35}
    result = schema.serialize(obj)
    assert result == {"name": "Bob", "age": 35}
    assert "email" not in result

    # Test serialize with object missing attributes
    class Person2:
        def __init__(self, name):
            self.name = name
    
    field1 = Field()
    field2 = Field()
    schema = Schema(fields={"name": field1, "age": field2})
    obj = Person2("Alice")
    result = schema.serialize(obj)
    assert result == {"name": "Alice"}

    # Test serialize calls field.serialize on each field
    class CustomField(Field):
        def serialize(self, obj):
            return str(obj).upper() if obj else None
    
    custom_field = CustomField()
    schema = Schema(fields={"message": custom_field})
    obj = {"message": "hello"}
    result = schema.serialize(obj)
    assert result == {"message": "HELLO"}

    # Test serialize with empty dict
    schema = Schema(fields={})
    obj = {"name": "John"}
    result = schema.serialize(obj)
    assert result == {}

    # Test serialize with nested structures
    class NestedField(Field):
        def serialize(self, obj):
            return {"nested": obj}
    
    nested_field = NestedField()
    schema = Schema(fields={"data": nested_field})
    obj = {"data": "value"}
    result = schema.serialize(obj)
    assert result == {"data": {"nested": "value"}}


# LLM-generated content at query #2
#--------------------------

```python
def test_Schema_serialize():
    # Test serialize with None
    field1 = Field()
    schema = Schema(fields={"key1": field1})
    assert schema.serialize(None) is None

    # Test serialize with dict
    field1 = Field()
    field2 = Field()
    schema = Schema(fields={"key1": field1, "key2": field2})
    obj = {"key1": "value1", "key2": "value2"}
    result = schema.serialize(obj)
    assert result == {"key1": "value1", "key2": "value2"}

    # Test serialize with object (non-dict)
    class TestObj:
        def __init__(self):
            self.key1 = "value1"
            self.key2 = "value2"

    field1 = Field()
    field2 = Field()
    schema = Schema(fields={"key1": field1, "key2": field2})
    obj = TestObj()
    result = schema.serialize(obj)
    assert result == {"key1": "value1", "key2": "value2"}

    # Test serialize with missing keys
    field1 = Field()
    field2 = Field()
    schema = Schema(fields={"key1": field1, "key2": field2})
    obj = {"key1": "value1"}
    result = schema.serialize(obj)
    assert result == {"key1": "value1"}

    # Test serialize with extra keys (should be ignored)
    field1 = Field()
    schema = Schema(fields={"key1": field1})
    obj = {"key1": "value1", "extra_key": "extra_value"}
    result = schema.serialize(obj)
    assert result == {"key1": "value1"}

    # Test serialize calls field.serialize on each field
    class MockField(Field):
        def serialize(self, obj):
            return f"serialized_{obj}"

    field1 = MockField()
    schema = Schema(fields={"key1": field1})
    obj = {"key1": "value1"}
    result = schema.serialize(obj)
    assert result == {"key1": "serialized_value1"}

    # Test serialize with AttributeError on object
    class TestObjPartial:
        def __init__(self):
            self.key1 = "value1"

    field1 = Field()
    field2 = Field()
    schema = Schema(fields={"key1": field1, "key2": field2})
    obj = TestObjPartial()
    result = schema.serialize(obj)
    assert result == {"key1": "value1"}

    # Test serialize with KeyError on dict
    field1 = Field()
    field2 = Field()
    schema = Schema(fields={"key1": field1, "key2": field2})
    obj = {"key1": "value1"}
    result = schema.serialize(obj)
    assert result == {"key1": "value1"}


# LLM-generated content at query #3
#--------------------------

```python
def test_Schema_serialize():
    # Test with None value
    field1 = Field()
    schema = Schema(fields={"name": field1})
    assert schema.serialize(None) is None

    # Test with dict object
    name_field = Field()
    age_field = Field()
    schema = Schema(fields={"name": name_field, "age": age_field})
    obj = {"name": "John", "age": 30}
    result = schema.serialize(obj)
    assert result == {"name": "John", "age": 30}

    # Test with object attributes
    class Person:
        def __init__(self, name, age):
            self.name = name
            self.age = age

    schema = Schema(fields={"name": name_field, "age": age_field})
    person = Person("Alice", 25)
    result = schema.serialize(person)
    assert result == {"name": "Alice", "age": 25}

    # Test with missing keys/attributes
    obj_partial = {"name": "Bob"}
    result = schema.serialize(obj_partial)
    assert result == {"name": "Bob"}

    # Test with read_only fields
    name_field_ro = Field(read_only=True)
    age_field_normal = Field()
    schema = Schema(fields={"name": name_field_ro, "age": age_field_normal})
    obj = {"name": "Charlie", "age": 35}
    result = schema.serialize(obj)
    assert result == {"name": "Charlie", "age": 35}

    # Test with empty fields
    schema = Schema(fields={})
    obj = {"name": "Dave", "age": 40}
    result = schema.serialize(obj)
    assert result == {}

    # Test with KeyError and AttributeError handling
    class IncompleteObject:
        def __init__(self):
            self.name = "Eve"

    schema = Schema(fields={"name": Field(), "age": Field()})
    obj = IncompleteObject()
    result = schema.serialize(obj)
    assert result == {"name": "Eve"}

    # Test serialization with field serialize method called
    class CustomField(Field):
        def serialize(self, obj):
            return str(obj).upper() if obj else None

    schema = Schema(fields={"name": CustomField(), "status": Field()})
    obj = {"name": "test", "status": "active"}
    result = schema.serialize(obj)
    assert result == {"name": "TEST", "status": "active"}


# LLM-generated content at query #4
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    schema = Schema(fields={"name": Field()})
    definitions["TestSchema"] = schema
    
    reference = Reference(to="TestSchema", definitions=definitions)
    
    # Test 1: Valid value passes through to target validation
    test_value = {"name": "test"}
    result = reference.validate(test_value)
    assert result == {"name": "test"}
    
    # Test 2: None value with allow_null=True returns None
    reference_nullable = Reference(to="TestSchema", definitions=definitions, allow_null=True)
    result = reference_nullable.validate(None)
    assert result is None
    
    # Test 3: None value with allow_null=False raises validation error
    reference_non_nullable = Reference(to="TestSchema", definitions=definitions, allow_null=False)
    with pytest.raises(ValidationError) as exc_info:
        reference_non_nullable.validate(None)
    assert exc_info.value.code == "null"
    
    # Test 4: Invalid value raises validation error from target schema
    with pytest.raises(ValidationError):
        reference.validate(None)
    
    # Test 5: Reference resolves correct target schema
    definitions2 = Definitions()
    schema2 = Schema(fields={"id": Field(), "value": Field()})
    definitions2["AnotherSchema"] = schema2
    reference2 = Reference(to="AnotherSchema", definitions=definitions2)
    
    test_value2 = {"id": 1, "value": "test"}
    result2 = reference2.validate(test_value2)
    assert result2 == {"id": 1, "value": "test"}


# LLM-generated content at query #5
#--------------------------

def test_Schema_validate():
    # Test with None when allow_null is True
    schema = Schema(fields={}, allow_null=True)
    assert schema.validate(None) is None
    
    # Test with None when allow_null is False
    schema = Schema(fields={}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert e.code() == "null"
    
    # Test with non-dict/mapping type
    schema = Schema(fields={})
    try:
        schema.validate("not a dict")
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert e.code() == "type"
    
    # Test with valid dict
    field1 = Field()
    schema = Schema(fields={"key1": field1})
    result = schema.validate({"key1": "value1"})
    assert result == {"key1": "value1"}
    
    # Test with non-string keys
    schema = Schema(fields={})
    try:
        schema.validate({1: "value"})
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        messages = e.messages()
        assert any(msg.code == "invalid_key" for msg in messages)
    
    # Test with required field missing
    field1 = Field(default=None)
    field2 = Field()  # required
    schema = Schema(fields={"field1": field1, "field2": field2})
    try:
        schema.validate({"field1": "value1"})
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        messages = e.messages()
        assert any(msg.code == "required" for msg in messages)
    
    # Test with default values
    field1 = Field(default="default_value")
    schema = Schema(fields={"field1": field1})
    result = schema.validate({})
    assert result == {"field1": "default_value"}
    
    # Test with read_only field
    field1 = Field(read_only=True)
    field2 = Field()
    schema = Schema(fields={"field1": field1, "field2": field2})
    result = schema.validate({"field1": "ignored", "field2": "value2"})
    assert "field1" not in result
    assert result == {"field2": "value2"}
    
    # Test with mapping instead of dict
    field1 = Field()
    schema = Schema(fields={"key1": field1})
    from collections import OrderedDict
    mapping = OrderedDict([("key1", "value1")])
    result = schema.validate(mapping)
    assert result == {"key1": "value1"}
    
    # Test with nested validation error
    from typesystem.fields import Integer
    field1 = Integer()
    schema = Schema(fields={"field1": field1})
    try:
        schema.validate({"field1": "not_an_integer"})
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) > 0
    
    # Test with multiple fields, some valid some invalid
    field1 = Field()
    field2 = Field()
    schema = Schema(fields={"field1": field1, "field2": field2})
    result = schema.validate({"field1": "value1", "field2": "value2"})
    assert result == {"field1": "value1", "field2": "value2"}


# LLM-generated content at query #6
#--------------------------

```python
def test_Schema():
    from typesystem.fields import String, Integer, Boolean
    
    # Test basic initialization
    fields = {
        "name": String(),
        "age": Integer(),
    }
    schema = Schema(fields=fields)
    assert schema.fields == fields
    assert schema.required == ["name", "age"]
    
    # Test with read_only field
    fields_with_readonly = {
        "name": String(),
        "id": Integer(read_only=True),
    }
    schema = Schema(fields=fields_with_readonly)
    assert "id" not in schema.required
    assert "name" in schema.required
    
    # Test with default values
    fields_with_defaults = {
        "name": String(),
        "active": Boolean(default=True),
    }
    schema = Schema(fields=fields_with_defaults)
    assert "active" not in schema.required
    assert "name" in schema.required
    
    # Test with both read_only and default
    fields_mixed = {
        "name": String(),
        "id": Integer(read_only=True),
        "active": Boolean(default=False),
    }
    schema = Schema(fields=fields_mixed)
    assert schema.required == ["name"]
    
    # Test with empty fields
    schema = Schema(fields={})
    assert schema.fields == {}
    assert schema.required == []
    
    # Test with allow_null kwarg
    schema = Schema(fields=fields, allow_null=True)
    assert schema.allow_null is True
    assert schema.fields == fields


# LLM-generated content at query #7
#--------------------------

```python
def test_Schema():
    # Test basic initialization
    fields = {
        "name": Field(),
        "age": Field(),
    }
    schema = Schema(fields=fields)
    assert schema.fields == fields
    assert schema.required == ["name", "age"]

    # Test with read_only field
    read_only_field = Field(read_only=True)
    fields_with_readonly = {
        "id": read_only_field,
        "name": Field(),
    }
    schema = Schema(fields=fields_with_readonly)
    assert "id" not in schema.required
    assert "name" in schema.required

    # Test with field that has default
    field_with_default = Field(default="default_value")
    fields_with_default = {
        "status": field_with_default,
        "name": Field(),
    }
    schema = Schema(fields=fields_with_default)
    assert "status" not in schema.required
    assert "name" in schema.required

    # Test with mix of read_only and default fields
    fields_mixed = {
        "id": Field(read_only=True),
        "status": Field(default="active"),
        "name": Field(),
        "email": Field(),
    }
    schema = Schema(fields=fields_mixed)
    assert schema.required == ["name", "email"]

    # Test with empty fields
    schema = Schema(fields={})
    assert schema.fields == {}
    assert schema.required == []

    # Test that kwargs are passed to parent
    schema = Schema(fields={}, allow_null=True)
    assert schema.allow_null is True

    # Test with all fields having defaults or read_only
    fields_all_optional = {
        "id": Field(read_only=True),
        "status": Field(default="pending"),
    }
    schema = Schema(fields=fields_all_optional)
    assert schema.required == []


# LLM-generated content at query #8
#--------------------------

```python
def test_Schema_validate():
    # Test with None value when allow_null is True
    schema = Schema(fields={}, allow_null=True)
    assert schema.validate(None) is None

    # Test with None value when allow_null is False
    schema = Schema(fields={}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Should raise ValidationError"
    except ValidationError:
        pass

    # Test with non-dict/mapping value
    schema = Schema(fields={})
    try:
        schema.validate("invalid")
        assert False, "Should raise ValidationError"
    except ValidationError:
        pass

    # Test with valid empty dict
    schema = Schema(fields={})
    result = schema.validate({})
    assert result == {}

    # Test with non-string keys
    schema = Schema(fields={})
    try:
        schema.validate({1: "value"})
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert any(msg.code == "invalid_key" for msg in e.messages())

    # Test with required field missing
    required_field = Field()
    schema = Schema(fields={"name": required_field})
    try:
        schema.validate({})
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert any(msg.code == "required" for msg in e.messages())

    # Test with valid required field
    required_field = Field()
    schema = Schema(fields={"name": required_field})
    result = schema.validate({"name": "test"})
    assert result == {"name": "test"}

    # Test with read_only field
    read_only_field = Field(read_only=True)
    schema = Schema(fields={"id": read_only_field})
    result = schema.validate({"id": "123"})
    assert result == {}

    # Test with field having default value
    default_field = Field(default="default_value")
    schema = Schema(fields={"status": default_field})
    result = schema.validate({})
    assert result == {"status": "default_value"}

    # Test with field validation error
    from typesystem.fields import Integer
    int_field = Integer()
    schema = Schema(fields={"age": int_field})
    try:
        schema.validate({"age": "not_an_int"})
        assert False, "Should raise ValidationError"
    except ValidationError:
        pass

    # Test with multiple fields, some valid and some invalid
    from typesystem.fields import Integer, String
    schema = Schema(fields={
        "name": String(),
        "age": Integer()
    })
    result = schema.validate({"name": "John", "age": 30})
    assert result == {"name": "John", "age": 30}

    # Test with extra fields not in schema
    schema = Schema(fields={"name": String()})
    result = schema.validate({"name": "John", "extra": "field"})
    assert result == {"name": "John"}

    # Test with Mapping type instead of dict
    from collections import UserDict
    mapping = UserDict({"name": "test"})
    schema = Schema(fields={"name": String()})
    result = schema.validate(mapping)
    assert result == {"name": "test"}

    # Test with multiple validation errors
    schema = Schema(fields={
        "name": String(allow_null=False),
        "age": Integer()
    })
    try:
        schema.validate({})
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) >= 1


# LLM-generated content at query #9
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    schema = Schema(fields={"name": Field()})
    definitions["TestSchema"] = schema
    
    # Test 1: Valid value passes through to target validation
    reference = Reference(to="TestSchema", definitions=definitions)
    result = reference.validate({"name": "test"})
    assert result == {"name": "test"}
    
    # Test 2: None value when allow_null is True
    reference_nullable = Reference(to="TestSchema", definitions=definitions, allow_null=True)
    result = reference_nullable.validate(None)
    assert result is None
    
    # Test 3: None value when allow_null is False raises validation error
    reference = Reference(to="TestSchema", definitions=definitions, allow_null=False)
    try:
        reference.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code() == "null"
    
    # Test 4: Invalid value propagates target validation error
    reference = Reference(to="TestSchema", definitions=definitions)
    try:
        reference.validate("invalid")
        assert False, "Should have raised ValidationError"
    except ValidationError:
        pass
    
    # Test 5: Reference resolves to correct target
    reference = Reference(to="TestSchema", definitions=definitions)
    assert reference.target == schema
    
    # Test 6: Complex nested validation
    complex_schema = Schema(fields={
        "id": Field(),
        "value": Field()
    })
    definitions["ComplexSchema"] = complex_schema
    reference = Reference(to="ComplexSchema", definitions=definitions)
    result = reference.validate({"id": 1, "value": "test"})
    assert result == {"id": 1, "value": "test"}


# LLM-generated content at query #10
#--------------------------

```python
def test_Reference_validate():
    # Test 1: validate with None when allow_null is True
    definitions = Definitions()
    schema = Schema(fields={})
    definitions["TestSchema"] = schema
    ref = Reference(to="TestSchema", definitions=definitions, allow_null=True)
    result = ref.validate(None)
    assert result is None

    # Test 2: validate with None when allow_null is False
    ref_not_null = Reference(to="TestSchema", definitions=definitions, allow_null=False)
    try:
        ref_not_null.validate(None)
        assert False, "Should raise validation error"
    except ValidationError as e:
        assert e.code == "null"

    # Test 3: validate with valid value
    test_field = Field()
    schema_with_field = Schema(fields={"name": test_field})
    definitions["UserSchema"] = schema_with_field
    ref_user = Reference(to="UserSchema", definitions=definitions)
    result = ref_user.validate({"name": "John"})
    assert result == {"name": "John"}

    # Test 4: validate with invalid value that target schema rejects
    ref_user = Reference(to="UserSchema", definitions=definitions)
    try:
        ref_user.validate(None)
        assert False, "Should raise validation error"
    except ValidationError:
        pass

    # Test 5: target property returns correct definition
    definitions2 = Definitions()
    target_schema = Schema(fields={})
    definitions2["MyDef"] = target_schema
    ref = Reference(to="MyDef", definitions=definitions2)
    assert ref.target is target_schema

    # Test 6: validate with complex nested structure
    inner_field = Field()
    complex_schema = Schema(fields={"id": inner_field, "data": inner_field})
    definitions["ComplexSchema"] = complex_schema
    ref_complex = Reference(to="ComplexSchema", definitions=definitions)
    result = ref_complex.validate({"id": 1, "data": "test"})
    assert result == {"id": 1, "data": "test"}


# LLM-generated content at query #11
#--------------------------

```python
def test_Reference_validate():
    # Test 1: validate with None when allow_null is True
    definitions = Definitions()
    schema = Schema(fields={"name": Field()})
    definitions["TestSchema"] = schema
    ref = Reference(to="TestSchema", definitions=definitions, allow_null=True)
    result = ref.validate(None)
    assert result is None

    # Test 2: validate with None when allow_null is False (default)
    ref_no_null = Reference(to="TestSchema", definitions=definitions)
    with pytest.raises(ValidationError) as exc_info:
        ref_no_null.validate(None)
    assert exc_info.value.code == "null"

    # Test 3: validate with valid value delegates to target schema
    definitions2 = Definitions()
    schema2 = Schema(fields={"name": Field(), "age": Field()})
    definitions2["Person"] = schema2
    ref2 = Reference(to="Person", definitions=definitions2)
    
    valid_data = {"name": "John", "age": 30}
    result = ref2.validate(valid_data)
    assert result == valid_data

    # Test 4: validate with invalid data - delegates error to target schema
    definitions3 = Definitions()
    schema3 = Schema(fields={"name": Field(required=True)})
    definitions3["RequiredSchema"] = schema3
    ref3 = Reference(to="RequiredSchema", definitions=definitions3)
    
    invalid_data = {}
    with pytest.raises(ValidationError):
        ref3.validate(invalid_data)

    # Test 5: validate with nested schema
    definitions4 = Definitions()
    inner_schema = Schema(fields={"id": Field()})
    outer_schema = Schema(fields={"data": inner_schema})
    definitions4["Nested"] = outer_schema
    ref4 = Reference(to="Nested", definitions=definitions4)
    
    nested_data = {"data": {"id": 123}}
    result = ref4.validate(nested_data)
    assert result == nested_data

    # Test 6: validate error message is correct
    ref_error = Reference(to="TestSchema", definitions=definitions)
    with pytest.raises(ValidationError) as exc_info:
        ref_error.validate(None)
    assert "May not be null" in str(exc_info.value)


# LLM-generated content at query #12
#--------------------------

```python
def test_Reference_validate():
    # Test 1: validate with allow_null=True and value=None returns None
    definitions = Definitions()
    schema = Schema(fields={"name": Field()})
    definitions["TestSchema"] = schema
    ref = Reference(to="TestSchema", definitions=definitions, allow_null=True)
    assert ref.validate(None) is None

    # Test 2: validate with allow_null=False and value=None raises validation error
    ref_no_null = Reference(to="TestSchema", definitions=definitions, allow_null=False)
    try:
        ref_no_null.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code() == "null"

    # Test 3: validate with valid value delegates to target schema
    test_value = {"name": "test"}
    result = ref_no_null.validate(test_value)
    assert result == {"name": "test"}

    # Test 4: validate with invalid value from target schema raises error
    schema_strict = Schema(fields={"name": Field(allow_null=False)})
    definitions["StrictSchema"] = schema_strict
    ref_strict = Reference(to="StrictSchema", definitions=definitions, allow_null=False)
    
    try:
        ref_strict.validate({"name": None})
        assert False, "Should have raised ValidationError"
    except ValidationError:
        pass

    # Test 5: validate passes through target schema validation result
    definitions["EmptySchema"] = Schema(fields={})
    ref_empty = Reference(to="EmptySchema", definitions=definitions)
    result = ref_empty.validate({"extra": "field"})
    assert result == {}


# LLM-generated content at query #13
#--------------------------

```python
def test_Schema_validate():
    # Test with None value when allow_null is True
    schema = Schema(fields={}, allow_null=True)
    assert schema.validate(None) is None

    # Test with None value when allow_null is False
    schema = Schema(fields={}, allow_null=False)
    with pytest.raises(ValidationError) as exc_info:
        schema.validate(None)
    assert exc_info.value.code() == "null"

    # Test with non-dict/non-mapping value
    schema = Schema(fields={})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate("invalid")
    assert exc_info.value.code() == "type"

    # Test with non-string keys
    schema = Schema(fields={})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({1: "value"})
    assert any(msg.code == "invalid_key" for msg in exc_info.value.messages())

    # Test with required field missing
    required_field = Field(allow_null=False)
    schema = Schema(fields={"name": required_field})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({})
    assert any(msg.code == "required" for msg in exc_info.value.messages())

    # Test with valid data
    field = Field()
    schema = Schema(fields={"name": field})
    result = schema.validate({"name": "John"})
    assert result == {"name": "John"}

    # Test with read_only field
    read_only_field = Field(read_only=True)
    schema = Schema(fields={"id": read_only_field})
    result = schema.validate({"id": "123"})
    assert "id" not in result

    # Test with field having default value
    default_field = Field(default="default_value")
    schema = Schema(fields={"status": default_field})
    result = schema.validate({})
    assert result == {"status": "default_value"}

    # Test with nested validation error
    child_field = Field()
    child_field.validate_or_error = lambda x: (None, ValidationError(messages=[Message(text="Invalid", code="invalid", index=[])]))
    schema = Schema(fields={"nested": child_field})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({"nested": "value"})
    assert len(exc_info.value.messages()) > 0

    # Test with multiple fields
    field1 = Field()
    field2 = Field()
    schema = Schema(fields={"field1": field1, "field2": field2})
    result = schema.validate({"field1": "value1", "field2": "value2"})
    assert result == {"field1": "value1", "field2": "value2"}

    # Test with mapping object instead of dict
    from collections import UserDict
    user_dict = UserDict({"key": "value"})
    schema = Schema(fields={"key": Field()})
    result = schema.validate(user_dict)
    assert result == {"key": "value"}

    # Test with extra fields not in schema
    schema = Schema(fields={"name": Field()})
    result = schema.validate({"name": "John", "extra": "field"})
    assert "extra" not in result
    assert result == {"name": "John"}

    # Test with multiple validation errors
    required_field1 = Field(allow_null=False)
    required_field2 = Field(allow_null=False)
    schema = Schema(fields={"field1": required_field1, "field2": required_field2})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({})
    messages = exc_info.value.messages()
    assert len(messages) == 2
    assert all(msg.code == "required" for msg in messages)


# LLM-generated content at query #14
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    schema = Schema(fields={"name": Field()})
    definitions["TestSchema"] = schema
    
    # Test 1: validate with allow_null=True and None value
    ref = Reference(to="TestSchema", definitions=definitions, allow_null=True)
    result = ref.validate(None)
    assert result is None
    
    # Test 2: validate with allow_null=False and None value raises error
    ref = Reference(to="TestSchema", definitions=definitions, allow_null=False)
    try:
        ref.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.code() == "null"
    
    # Test 3: validate with valid dict value
    ref = Reference(to="TestSchema", definitions=definitions, allow_null=False)
    test_value = {"name": "test"}
    result = ref.validate(test_value)
    assert result == {"name": "test"}
    
    # Test 4: validate with invalid value (non-dict) should raise error from target schema
    ref = Reference(to="TestSchema", definitions=definitions, allow_null=False)
    try:
        ref.validate("invalid")
        assert False, "Expected ValidationError"
    except ValidationError:
        pass
    
    # Test 5: validate with complex nested schema
    nested_schema = Schema(fields={"id": Field(), "value": Field()})
    definitions["NestedSchema"] = nested_schema
    ref = Reference(to="NestedSchema", definitions=definitions, allow_null=False)
    test_value = {"id": "1", "value": "data"}
    result = ref.validate(test_value)
    assert result == {"id": "1", "value": "data"}
    
    # Test 6: target property returns correct schema from definitions
    ref = Reference(to="TestSchema", definitions=definitions)
    assert ref.target is schema


# LLM-generated content at query #15
#--------------------------

```python
def test_Reference_validate():
    # Test case 1: validate with allow_null=True and None value
    definitions = Definitions()
    schema = Schema(fields={"name": Field()})
    definitions["TestSchema"] = schema
    
    ref = Reference(to="TestSchema", definitions=definitions, allow_null=True)
    result = ref.validate(None)
    assert result is None
    
    # Test case 2: validate with allow_null=False and None value should raise error
    ref_no_null = Reference(to="TestSchema", definitions=definitions, allow_null=False)
    try:
        ref_no_null.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "null"
    
    # Test case 3: validate with valid value
    test_data = {"name": "John"}
    result = ref.validate(test_data)
    assert result == {"name": "John"}
    
    # Test case 4: validate with schema that has validation rules
    from typesystem.fields import String
    schema_with_rules = Schema(fields={"name": String(max_length=10)})
    definitions["SchemaWithRules"] = schema_with_rules
    
    ref_with_rules = Reference(to="SchemaWithRules", definitions=definitions)
    result = ref_with_rules.validate({"name": "Alice"})
    assert result == {"name": "Alice"}
    
    # Test case 5: validate with schema validation error
    try:
        ref_with_rules.validate({"name": "ThisNameIsTooLongForTheRule"})
        assert False, "Should have raised ValidationError"
    except ValidationError:
        pass
    
    # Test case 6: validate empty dict
    result = ref.validate({})
    assert result == {}


# LLM-generated content at query #16
#--------------------------

```python
def test_Reference_validate():
    # Test 1: validate with None when allow_null is True
    definitions = Definitions()
    schema = Schema(fields={"name": Field()})
    definitions["TestSchema"] = schema
    
    reference = Reference(to="TestSchema", definitions=definitions, allow_null=True)
    result = reference.validate(None)
    assert result is None
    
    # Test 2: validate with None when allow_null is False (default)
    reference = Reference(to="TestSchema", definitions=definitions)
    with pytest.raises(ValidationError) as exc_info:
        reference.validate(None)
    assert exc_info.value.code == "null"
    
    # Test 3: validate with valid value - dict
    test_data = {"name": "John"}
    result = reference.validate(test_data)
    assert result == {"name": "John"}
    
    # Test 4: validate with valid value - mapping object
    class Mapping:
        def __init__(self):
            self.name = "Jane"
    
    mapping_obj = Mapping()
    result = reference.validate(mapping_obj)
    # Result depends on schema validation
    assert isinstance(result, dict)
    
    # Test 5: validate delegates to target schema
    name_field = Field()
    schema_with_validation = Schema(fields={"name": name_field})
    definitions["SchemaWithField"] = schema_with_validation
    
    reference = Reference(to="SchemaWithField", definitions=definitions)
    test_data = {"name": "Test"}
    result = reference.validate(test_data)
    assert result == {"name": "Test"}
    
    # Test 6: target property returns correct schema
    definitions_new = Definitions()
    target_schema = Schema(fields={"id": Field()})
    definitions_new["TargetSchema"] = target_schema
    
    reference = Reference(to="TargetSchema", definitions=definitions_new)
    assert reference.target is target_schema


# LLM-generated content at query #17
#--------------------------

```python
def test_Reference_validate():
    # Test case 1: validate with None when allow_null is True
    definitions = Definitions()
    schema = Schema(fields={"name": Field()})
    definitions["TestSchema"] = schema
    ref = Reference(to="TestSchema", definitions=definitions, allow_null=True)
    assert ref.validate(None) is None

    # Test case 2: validate with None when allow_null is False
    ref_not_null = Reference(to="TestSchema", definitions=definitions, allow_null=False)
    with ValidationError as exc_info:
        ref_not_null.validate(None)
    assert exc_info.value.code() == "null"

    # Test case 3: validate with valid value
    definitions2 = Definitions()
    name_field = Field()
    schema2 = Schema(fields={"name": name_field})
    definitions2["Person"] = schema2
    ref2 = Reference(to="Person", definitions=definitions2)
    
    result = ref2.validate({"name": "John"})
    assert result == {"name": "John"}

    # Test case 4: validate delegates to target schema
    definitions3 = Definitions()
    int_field = Field()
    schema3 = Schema(fields={"age": int_field})
    definitions3["AgeSchema"] = schema3
    ref3 = Reference(to="AgeSchema", definitions=definitions3)
    
    result = ref3.validate({"age": 25})
    assert result == {"age": 25}

    # Test case 5: validate with invalid data propagates target schema errors
    definitions4 = Definitions()
    required_field = Field()
    schema4 = Schema(fields={"required_field": required_field})
    definitions4["RequiredSchema"] = schema4
    ref4 = Reference(to="RequiredSchema", definitions=definitions4)
    
    with ValidationError:
        ref4.validate({})

    # Test case 6: validate with None and allow_null False raises validation error
    ref_none = Reference(to="TestSchema", definitions=definitions, allow_null=False)
    try:
        ref_none.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert "null" in str(e)


# LLM-generated content at query #18
#--------------------------

```python
def test_Schema_validate():
    # Test with None and allow_null=True
    field1 = Field()
    schema = Schema(fields={"name": field1}, allow_null=True)
    assert schema.validate(None) is None

    # Test with None and allow_null=False
    schema = Schema(fields={"name": field1}, allow_null=False)
    with pytest.raises(ValidationError) as exc_info:
        schema.validate(None)
    assert exc_info.value.code == "null"

    # Test with non-dict/Mapping type
    schema = Schema(fields={"name": field1})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate("invalid")
    assert exc_info.value.code == "type"

    # Test with invalid key type (non-string)
    schema = Schema(fields={"name": field1})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({1: "value"})
    messages = exc_info.value.messages()
    assert any(msg.code == "invalid_key" for msg in messages)

    # Test with required field missing
    required_field = Field()
    schema = Schema(fields={"name": required_field})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({})
    messages = exc_info.value.messages()
    assert any(msg.code == "required" for msg in messages)

    # Test with read_only field (should be skipped)
    read_only_field = Field(read_only=True)
    schema = Schema(fields={"id": read_only_field})
    result = schema.validate({})
    assert "id" not in result

    # Test with field having default value
    default_field = Field(default="default_value")
    schema = Schema(fields={"status": default_field})
    result = schema.validate({})
    assert result["status"] == "default_value"

    # Test valid data
    name_field = Field()
    name_field.validate_or_error = lambda x: (x, None)
    schema = Schema(fields={"name": name_field})
    result = schema.validate({"name": "John"})
    assert result == {"name": "John"}

    # Test with nested validation error
    mock_field = Field()
    mock_field.validate_or_error = lambda x: (None, ValidationError(messages=[Message(text="Invalid", code="invalid", index=[])]))
    schema = Schema(fields={"data": mock_field})
    with pytest.raises(ValidationError):
        schema.validate({"data": "value"})

    # Test with Mapping type (not just dict)
    from collections import OrderedDict
    field_item = Field()
    field_item.validate_or_error = lambda x: (x, None)
    schema = Schema(fields={"key": field_item})
    mapping_input = OrderedDict([("key", "value")])
    result = schema.validate(mapping_input)
    assert result == {"key": "value"}

    # Test with multiple validation errors
    field_a = Field()
    field_b = Field()
    schema = Schema(fields={"a": field_a, "b": field_b})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({})
    messages = exc_info.value.messages()
    assert len(messages) == 2
    assert all(msg.code == "required" for msg in messages)

    # Test with mixed valid and invalid fields
    valid_field = Field()
    valid_field.validate_or_error = lambda x: (x, None)
    invalid_field = Field()
    invalid_field.validate_or_error = lambda x: (None, ValidationError(messages=[Message(text="Error", code="error", index=[])]))
    schema = Schema(fields={"valid": valid_field, "invalid": invalid_field})
    with pytest.raises(ValidationError):
        schema.validate({"valid": "data", "invalid": "data"})

    # Test empty dict with only read_only and default fields
    read_only = Field(read_only=True)
    with_default = Field(default="default")
    schema = Schema(fields={"readonly": read_only, "withdefault": with_default})
    result = schema.validate({})
    assert result == {"withdefault": "default"}


# LLM-generated content at query #19
#--------------------------

```python
def test_Reference_validate():
    # Test 1: validate with None when allow_null is True
    definitions = Definitions()
    schema = Schema(fields={"name": Field()})
    definitions["TestSchema"] = schema
    
    ref = Reference(to="TestSchema", definitions=definitions, allow_null=True)
    result = ref.validate(None)
    assert result is None
    
    # Test 2: validate with None when allow_null is False (default)
    ref_not_null = Reference(to="TestSchema", definitions=definitions)
    try:
        ref_not_null.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "null"
    
    # Test 3: validate with valid value
    ref = Reference(to="TestSchema", definitions=definitions)
    test_value = {"name": "test"}
    result = ref.validate(test_value)
    assert result == {"name": "test"}
    
    # Test 4: validate with invalid value that target schema rejects
    from typesystem.fields import String
    schema_with_string = Schema(fields={"name": String(max_length=5)})
    definitions["StringSchema"] = schema_with_string
    
    ref = Reference(to="StringSchema", definitions=definitions)
    try:
        ref.validate({"name": "this is too long"})
        assert False, "Should have raised ValidationError"
    except ValidationError:
        pass
    
    # Test 5: validate delegates to target schema
    simple_schema = Schema(fields={"id": Field()})
    definitions["SimpleSchema"] = simple_schema
    
    ref = Reference(to="SimpleSchema", definitions=definitions)
    result = ref.validate({"id": 123})
    assert result == {"id": 123}
    
    # Test 6: target property returns correct definition
    ref = Reference(to="TestSchema", definitions=definitions)
    assert ref.target is schema


# LLM-generated content at query #20
#--------------------------

```python
def test_Schema_validate():
    # Test with None and allow_null=True
    schema = Schema(fields={}, allow_null=True)
    assert schema.validate(None) is None

    # Test with None and allow_null=False
    schema = Schema(fields={}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert e.messages()[0].code == "null"

    # Test with non-dict/mapping type
    schema = Schema(fields={})
    try:
        schema.validate("invalid")
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert e.messages()[0].code == "type"

    # Test with invalid key type (non-string)
    schema = Schema(fields={})
    try:
        schema.validate({1: "value"})
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert e.messages()[0].code == "invalid_key"

    # Test with required field missing
    required_field = Field()
    schema = Schema(fields={"name": required_field})
    try:
        schema.validate({})
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert e.messages()[0].code == "required"

    # Test with valid data
    field = Field()
    schema = Schema(fields={"name": field})
    result = schema.validate({"name": "John"})
    assert result == {"name": "John"}

    # Test with read_only field
    read_only_field = Field(read_only=True)
    schema = Schema(fields={"id": read_only_field})
    result = schema.validate({})
    assert result == {}

    # Test with field having default value
    default_field = Field(default="default_value")
    schema = Schema(fields={"status": default_field})
    result = schema.validate({})
    assert result == {"status": "default_value"}

    # Test with field validation error
    from typesystem.fields import String
    string_field = String(max_length=5)
    schema = Schema(fields={"name": string_field})
    try:
        schema.validate({"name": "toolongname"})
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert len(e.messages()) > 0

    # Test with multiple fields, some missing, some with errors
    field1 = Field()
    field2 = Field()
    schema = Schema(fields={"field1": field1, "field2": field2})
    try:
        schema.validate({"field1": "value"})
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        messages = e.messages()
        assert any(m.code == "required" for m in messages)

    # Test with Mapping type (not just dict)
    from collections import OrderedDict
    field = Field()
    schema = Schema(fields={"name": field})
    result = schema.validate(OrderedDict([("name", "John")]))
    assert result == {"name": "John"}

    # Test with valid dict containing extra keys (should only include schema fields)
    field = Field()
    schema = Schema(fields={"name": field})
    result = schema.validate({"name": "John", "extra": "ignored"})
    assert result == {"name": "John"}
    assert "extra" not in result


# LLM-generated content at query #21
#--------------------------

```python
def test_Schema_validate():
    # Test with None value when allow_null is True
    schema = Schema(fields={}, allow_null=True)
    assert schema.validate(None) is None

    # Test with None value when allow_null is False
    schema = Schema(fields={}, allow_null=False)
    with pytest.raises(ValidationError) as exc_info:
        schema.validate(None)
    assert exc_info.value.code == "null"

    # Test with non-dict/mapping value
    schema = Schema(fields={})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate("not a dict")
    assert exc_info.value.code == "type"

    # Test with non-string keys
    schema = Schema(fields={})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({1: "value", "key": "value"})
    messages = exc_info.value.messages()
    assert any(msg.code == "invalid_key" for msg in messages)

    # Test with required field missing
    field = Field()
    schema = Schema(fields={"required_field": field})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({})
    messages = exc_info.value.messages()
    assert any(msg.code == "required" for msg in messages)

    # Test with valid data
    field = Field()
    schema = Schema(fields={"key": field})
    result = schema.validate({"key": "value"})
    assert result == {"key": "value"}

    # Test with read_only field
    read_only_field = Field(read_only=True)
    schema = Schema(fields={"read_only": read_only_field})
    result = schema.validate({"read_only": "value"})
    assert "read_only" not in result

    # Test with default value
    default_field = Field(default="default_value")
    schema = Schema(fields={"with_default": default_field})
    result = schema.validate({})
    assert result == {"with_default": "default_value"}

    # Test with child field validation error
    child_field = Field()
    child_field.validate = lambda x: (_ for _ in ()).throw(ValidationError(code="child_error"))
    schema = Schema(fields={"child": child_field})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({"child": "value"})
    messages = exc_info.value.messages()
    assert any(msg.code == "child_error" for msg in messages)

    # Test with mapping type instead of dict
    schema = Schema(fields={"key": Field()})
    result = schema.validate({"key": "value"})
    assert result == {"key": "value"}

    # Test multiple errors
    field1 = Field()
    field2 = Field()
    schema = Schema(fields={"field1": field1, "field2": field2})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({1: "value"})
    messages = exc_info.value.messages()
    assert len(messages) >= 2

    # Test with empty schema and empty dict
    schema = Schema(fields={})
    result = schema.validate({})
    assert result == {}


# LLM-generated content at query #22
#--------------------------

def test_Schema_validate():
    from typesystem.fields import String, Integer
    
    # Test: valid object
    schema = Schema(fields={"name": String(), "age": Integer()})
    result = schema.validate({"name": "John", "age": 30})
    assert result == {"name": "John", "age": 30}
    
    # Test: null value with allow_null=True
    schema = Schema(fields={"name": String()}, allow_null=True)
    result = schema.validate(None)
    assert result is None
    
    # Test: null value with allow_null=False
    schema = Schema(fields={"name": String()})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate(None)
    assert exc_info.value.messages()[0].code == "null"
    
    # Test: non-dict/mapping value
    schema = Schema(fields={"name": String()})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate("not a dict")
    assert exc_info.value.messages()[0].code == "type"
    
    # Test: non-string key
    schema = Schema(fields={"name": String()})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({1: "value"})
    assert exc_info.value.messages()[0].code == "invalid_key"
    
    # Test: missing required field
    schema = Schema(fields={"name": String(), "age": Integer()})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({"name": "John"})
    errors = exc_info.value.messages()
    assert any(msg.code == "required" for msg in errors)
    
    # Test: field with default value
    schema = Schema(fields={"name": String(), "status": String(default="active")})
    result = schema.validate({"name": "John"})
    assert result == {"name": "John", "status": "active"}
    
    # Test: read-only field ignored
    schema = Schema(fields={"name": String(), "id": Integer(read_only=True)})
    result = schema.validate({"name": "John", "id": 123})
    assert result == {"name": "John"}
    
    # Test: child field validation error
    schema = Schema(fields={"name": String(), "age": Integer()})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({"name": "John", "age": "not an integer"})
    assert len(exc_info.value.messages()) > 0
    
    # Test: mapping input (not dict)
    from collections import OrderedDict
    schema = Schema(fields={"name": String()})
    result = schema.validate(OrderedDict([("name", "John")]))
    assert result == {"name": "John"}
    
    # Test: multiple validation errors
    schema = Schema(fields={"name": String(), "age": Integer()})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({1: "value", "age": "invalid"})
    errors = exc_info.value.messages()
    assert len(errors) >= 2


# LLM-generated content at query #23
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    schema = Schema(fields={"name": Field()})
    definitions["TestSchema"] = schema
    
    reference = Reference(to="TestSchema", definitions=definitions)
    
    # Test valid value
    test_value = {"name": "test"}
    result = reference.validate(test_value)
    assert result == {"name": "test"}
    
    # Test None with allow_null=False (default)
    with pytest.raises(ValidationError) as exc_info:
        reference.validate(None)
    assert exc_info.value.code() == "null"
    
    # Test None with allow_null=True
    reference_nullable = Reference(to="TestSchema", definitions=definitions, allow_null=True)
    result = reference_nullable.validate(None)
    assert result is None
    
    # Test that validation is delegated to target schema
    reference2 = Reference(to="TestSchema", definitions=definitions)
    test_value2 = {"name": "another_test"}
    result2 = reference2.validate(test_value2)
    assert result2 == {"name": "another_test"}
    
    # Test with different schema definition
    string_field = Field()
    definitions["StringField"] = string_field
    reference3 = Reference(to="StringField", definitions=definitions)
    result3 = reference3.validate("test_string")
    assert result3 == "test_string"


# LLM-generated content at query #24
#--------------------------

```python
def test_Schema_validate():
    # Test valid object
    schema = Schema(fields={
        "name": Field(),
        "age": Field(),
    })
    result = schema.validate({"name": "John", "age": 30})
    assert result == {"name": "John", "age": 30}

    # Test with None when allow_null is True
    schema = Schema(fields={"name": Field()}, allow_null=True)
    result = schema.validate(None)
    assert result is None

    # Test with None when allow_null is False
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate(None)
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert any(msg.code == "null" for msg in e.messages())

    # Test with non-dict/mapping type
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate("invalid")
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert any(msg.code == "type" for msg in e.messages())

    # Test with non-string keys
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate({1: "value"})
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert any(msg.code == "invalid_key" for msg in e.messages())

    # Test required field missing
    name_field = Field()
    name_field.required = True
    schema = Schema(fields={"name": name_field})
    try:
        schema.validate({})
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert any(msg.code == "required" for msg in e.messages())

    # Test with default value
    default_field = Field(default="default_value")
    schema = Schema(fields={"name": default_field})
    result = schema.validate({})
    assert result == {"name": "default_value"}

    # Test with read_only field
    read_only_field = Field(read_only=True)
    schema = Schema(fields={"id": read_only_field})
    result = schema.validate({"id": 123})
    assert "id" not in result

    # Test child field validation error
    from typesystem.fields import Integer
    schema = Schema(fields={"age": Integer()})
    try:
        schema.validate({"age": "not_an_int"})
        assert False, "Should raise ValidationError"
    except ValidationError:
        pass

    # Test with mapping object (not dict)
    from collections import OrderedDict
    schema = Schema(fields={"name": Field()})
    result = schema.validate(OrderedDict([("name", "John")]))
    assert result == {"name": "John"}

    # Test empty object
    schema = Schema(fields={})
    result = schema.validate({})
    assert result == {}

    # Test extra fields are ignored
    schema = Schema(fields={"name": Field()})
    result = schema.validate({"name": "John", "extra": "field"})
    assert result == {"name": "John"}

    # Test multiple validation errors
    name_field = Field()
    name_field.required = True
    age_field = Field()
    age_field.required = True
    schema = Schema(fields={"name": name_field, "age": age_field})
    try:
        schema.validate({})
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 2
        assert any(msg.code == "required" for msg in messages)


# LLM-generated content at query #25
#--------------------------

```python
def test_Schema_validate():
    # Test with None when allow_null is True
    schema = Schema(fields={"name": Field()}, allow_null=True)
    assert schema.validate(None) is None

    # Test with None when allow_null is False
    schema = Schema(fields={"name": Field()}, allow_null=False)
    with pytest.raises(ValidationError) as exc_info:
        schema.validate(None)
    assert exc_info.value.messages()[0].code == "null"

    # Test with non-dict/non-mapping type
    schema = Schema(fields={"name": Field()})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate("not a dict")
    assert exc_info.value.messages()[0].code == "type"

    # Test with valid dict
    schema = Schema(fields={"name": Field()})
    result = schema.validate({"name": "John"})
    assert result == {"name": "John"}

    # Test with non-string keys
    schema = Schema(fields={"name": Field()})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({1: "value"})
    messages = exc_info.value.messages()
    assert any(msg.code == "invalid_key" for msg in messages)

    # Test with required field missing
    name_field = Field(allow_null=False)
    schema = Schema(fields={"name": name_field})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({})
    messages = exc_info.value.messages()
    assert any(msg.code == "required" for msg in messages)

    # Test with field having default value
    name_field = Field(default="Unknown")
    schema = Schema(fields={"name": name_field})
    result = schema.validate({})
    assert result == {"name": "Unknown"}

    # Test with read_only field
    name_field = Field(read_only=True)
    schema = Schema(fields={"name": name_field})
    result = schema.validate({"name": "John"})
    assert "name" not in result

    # Test with nested validation error
    nested_field = Field()
    schema = Schema(fields={"value": nested_field})
    result = schema.validate({"value": 123})
    assert result == {"value": 123}

    # Test with Mapping type instead of dict
    from collections import OrderedDict
    mapping = OrderedDict([("name", "Alice")])
    schema = Schema(fields={"name": Field()})
    result = schema.validate(mapping)
    assert result == {"name": "Alice"}

    # Test multiple required fields missing
    field1 = Field()
    field2 = Field()
    schema = Schema(fields={"field1": field1, "field2": field2})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({})
    messages = exc_info.value.messages()
    required_errors = [msg for msg in messages if msg.code == "required"]
    assert len(required_errors) == 2

    # Test with extra fields in value (should be ignored)
    schema = Schema(fields={"name": Field()})
    result = schema.validate({"name": "John", "extra": "field"})
    assert result == {"name": "John"}
    assert "extra" not in result


# LLM-generated content at query #26
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    schema = Schema(fields={"name": Field()})
    definitions["TestSchema"] = schema
    
    reference = Reference(to="TestSchema", definitions=definitions)
    
    # Test: Valid value passes through to target validation
    test_value = {"name": "test"}
    result = reference.validate(test_value)
    assert result == {"name": "test"}
    
    # Test: None with allow_null=True returns None
    reference_nullable = Reference(to="TestSchema", definitions=definitions, allow_null=True)
    result = reference_nullable.validate(None)
    assert result is None
    
    # Test: None with allow_null=False raises validation error
    reference_not_nullable = Reference(to="TestSchema", definitions=definitions, allow_null=False)
    with pytest.raises(ValidationError):
        reference_not_nullable.validate(None)
    
    # Test: Invalid value raises validation error from target
    with pytest.raises(ValidationError):
        reference.validate(None)
    
    # Test: Reference resolves correct target schema
    definitions2 = Definitions()
    schema1 = Schema(fields={"id": Field()})
    schema2 = Schema(fields={"name": Field()})
    definitions2["Schema1"] = schema1
    definitions2["Schema2"] = schema2
    
    ref1 = Reference(to="Schema1", definitions=definitions2)
    ref2 = Reference(to="Schema2", definitions=definitions2)
    
    assert ref1.target == schema1
    assert ref2.target == schema2


# LLM-generated content at query #27
#--------------------------

```python
def test_Reference_validate():
    # Test setup
    definitions = Definitions()
    schema = Schema(fields={"name": Field()})
    definitions["TestSchema"] = schema
    
    # Test 1: Valid value validation
    reference = Reference(to="TestSchema", definitions=definitions)
    test_data = {"name": "test"}
    result = reference.validate(test_data)
    assert result == {"name": "test"}
    
    # Test 2: None value with allow_null=True
    reference_nullable = Reference(to="TestSchema", definitions=definitions, allow_null=True)
    result = reference_nullable.validate(None)
    assert result is None
    
    # Test 3: None value with allow_null=False (default)
    reference_not_nullable = Reference(to="TestSchema", definitions=definitions)
    with pytest.raises(ValidationError) as exc_info:
        reference_not_nullable.validate(None)
    assert exc_info.value.messages()[0].code == "null"
    
    # Test 4: Delegates to target schema validation
    reference = Reference(to="TestSchema", definitions=definitions)
    invalid_data = None
    with pytest.raises(ValidationError):
        reference.validate(invalid_data)
    
    # Test 5: Complex schema validation through reference
    complex_schema = Schema(fields={
        "id": Field(),
        "name": Field(),
        "email": Field()
    })
    definitions["ComplexSchema"] = complex_schema
    reference = Reference(to="ComplexSchema", definitions=definitions)
    
    complex_data = {"id": 1, "name": "John", "email": "john@example.com"}
    result = reference.validate(complex_data)
    assert result == complex_data


# LLM-generated content at query #28
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    schema = Schema(fields={"name": Field()})
    definitions["TestSchema"] = schema
    
    reference = Reference(to="TestSchema", definitions=definitions)
    
    # Test: validate with valid value
    test_value = {"name": "test"}
    result = reference.validate(test_value)
    assert result == {"name": "test"}
    
    # Test: validate with None when allow_null is False (default)
    with pytest.raises(ValidationError) as exc_info:
        reference.validate(None)
    assert exc_info.value.code == "null"
    
    # Test: validate with None when allow_null is True
    reference_nullable = Reference(to="TestSchema", definitions=definitions, allow_null=True)
    result = reference_nullable.validate(None)
    assert result is None
    
    # Test: validate delegates to target schema
    reference_with_schema = Reference(to="TestSchema", definitions=definitions)
    result = reference_with_schema.validate({"name": "John"})
    assert result == {"name": "John"}
    
    # Test: validate with invalid data according to target schema
    invalid_schema = Schema(fields={"age": Field()})
    definitions["InvalidSchema"] = invalid_schema
    reference_invalid = Reference(to="InvalidSchema", definitions=definitions)
    
    with pytest.raises(ValidationError):
        reference_invalid.validate("not a dict")


# LLM-generated content at query #29
#--------------------------

```python
def test_Schema_validate():
    # Test with None when allow_null is True
    schema = Schema(fields={}, allow_null=True)
    assert schema.validate(None) is None

    # Test with None when allow_null is False
    schema = Schema(fields={}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert any(msg.code == "null" for msg in e.messages())

    # Test with non-dict/non-mapping type
    schema = Schema(fields={})
    try:
        schema.validate("invalid")
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert any(msg.code == "type" for msg in e.messages())

    # Test with non-string keys
    schema = Schema(fields={})
    try:
        schema.validate({1: "value"})
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert any(msg.code == "invalid_key" for msg in e.messages())

    # Test with missing required field
    required_field = Field()
    schema = Schema(fields={"required_key": required_field})
    try:
        schema.validate({})
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert any(msg.code == "required" for msg in e.messages())

    # Test with valid data
    field = Field()
    schema = Schema(fields={"key": field})
    result = schema.validate({"key": "value"})
    assert result == {"key": "value"}

    # Test with default value for missing field
    field_with_default = Field(default="default_value")
    schema = Schema(fields={"key": field_with_default})
    result = schema.validate({})
    assert result == {"key": "default_value"}

    # Test with read_only field
    read_only_field = Field(read_only=True)
    schema = Schema(fields={"readonly": read_only_field})
    result = schema.validate({"readonly": "should_be_ignored"})
    assert "readonly" not in result

    # Test with multiple fields and mixed validation
    field1 = Field()
    field2 = Field(default="default")
    schema = Schema(fields={"field1": field1, "field2": field2})
    result = schema.validate({"field1": "value1"})
    assert result == {"field1": "value1", "field2": "default"}

    # Test with child schema validation error
    child_schema = Schema(fields={"nested": Field()})
    parent_schema = Schema(fields={"child": child_schema})
    try:
        parent_schema.validate({"child": "invalid"})
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert len(e.messages()) > 0

    # Test with empty dict and no required fields
    schema = Schema(fields={})
    result = schema.validate({})
    assert result == {}

    # Test with mapping type instead of dict
    from collections import OrderedDict
    field = Field()
    schema = Schema(fields={"key": field})
    mapping = OrderedDict([("key", "value")])
    result = schema.validate(mapping)
    assert result == {"key": "value"}


# LLM-generated content at query #30
#--------------------------

```python
def test_Reference_validate():
    # Test 1: validate returns None when value is None and allow_null is True
    definitions = Definitions()
    schema = Schema(fields={})
    definitions["TestSchema"] = schema
    ref = Reference(to="TestSchema", definitions=definitions, allow_null=True)
    assert ref.validate(None) is None

    # Test 2: validate raises ValidationError when value is None and allow_null is False
    ref_not_nullable = Reference(to="TestSchema", definitions=definitions, allow_null=False)
    with pytest.raises(ValidationError):
        ref_not_nullable.validate(None)

    # Test 3: validate delegates to target schema for valid value
    test_field = Field()
    definitions["SimpleField"] = test_field
    ref = Reference(to="SimpleField", definitions=definitions)
    result = ref.validate("test_value")
    assert result == "test_value"

    # Test 4: validate with complex schema
    complex_schema = Schema(fields={
        "name": Field(),
        "age": Field(),
    })
    definitions["ComplexSchema"] = complex_schema
    ref = Reference(to="ComplexSchema", definitions=definitions)
    result = ref.validate({"name": "John", "age": 30})
    assert result == {"name": "John", "age": 30}

    # Test 5: validate raises ValidationError from target schema
    strict_schema = Schema(fields={
        "required_field": Field(),
    })
    definitions["StrictSchema"] = strict_schema
    ref = Reference(to="StrictSchema", definitions=definitions)
    with pytest.raises(ValidationError):
        ref.validate({})

    # Test 6: validate with nested reference
    inner_schema = Schema(fields={"value": Field()})
    definitions["InnerSchema"] = inner_schema
    outer_ref = Reference(to="InnerSchema", definitions=definitions)
    result = outer_ref.validate({"value": "test"})
    assert result == {"value": "test"}


# LLM-generated content at query #31
#--------------------------

```python
def test_Schema_validate():
    # Test with None value when allow_null is True
    schema = Schema(fields={}, allow_null=True)
    assert schema.validate(None) is None

    # Test with None value when allow_null is False
    schema = Schema(fields={}, allow_null=False)
    with pytest.raises(ValidationError) as exc_info:
        schema.validate(None)
    assert exc_info.value.code == "null"

    # Test with non-dict/mapping value
    schema = Schema(fields={})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate("not a dict")
    assert exc_info.value.code == "type"

    # Test with non-string keys
    schema = Schema(fields={})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({1: "value"})
    messages = exc_info.value.messages()
    assert any(msg.code == "invalid_key" for msg in messages)

    # Test with missing required field
    schema = Schema(fields={"name": Field(allow_null=False)})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({})
    messages = exc_info.value.messages()
    assert any(msg.code == "required" for msg in messages)

    # Test with valid data
    schema = Schema(fields={"name": Field()})
    result = schema.validate({"name": "John"})
    assert result == {"name": "John"}

    # Test with field having default value
    default_field = Field()
    default_field.default = "default_value"
    schema = Schema(fields={"name": default_field})
    result = schema.validate({})
    assert result == {"name": "default_value"}

    # Test with read_only field
    read_only_field = Field(read_only=True)
    schema = Schema(fields={"id": read_only_field})
    result = schema.validate({"id": "123"})
    assert "id" not in result

    # Test with child field validation error
    child_field = Field()
    schema = Schema(fields={"age": child_field})
    with pytest.raises(ValidationError):
        schema.validate({"age": "invalid"})

    # Test with multiple fields
    schema = Schema(
        fields={
            "name": Field(),
            "email": Field(),
        }
    )
    result = schema.validate({"name": "John", "email": "john@example.com"})
    assert result == {"name": "John", "email": "john@example.com"}

    # Test with Mapping type instead of dict
    from collections import OrderedDict
    schema = Schema(fields={"name": Field()})
    result = schema.validate(OrderedDict([("name", "John")]))
    assert result == {"name": "John"}

    # Test with multiple errors
    schema = Schema(
        fields={
            "required_field": Field(),
            "another_required": Field(),
        }
    )
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({})
    messages = exc_info.value.messages()
    assert len([m for m in messages if m.code == "required"]) == 2

    # Test with extra fields not in schema
    schema = Schema(fields={"name": Field()})
    result = schema.validate({"name": "John", "extra": "field"})
    assert result == {"name": "John"}


# LLM-generated content at query #32
#--------------------------

```python
def test_Reference_validate():
    # Test case 1: validate with None when allow_null is True
    definitions = Definitions()
    schema = Schema(fields={"name": Field()})
    definitions["TestSchema"] = schema
    ref = Reference(to="TestSchema", definitions=definitions, allow_null=True)
    assert ref.validate(None) is None

    # Test case 2: validate with None when allow_null is False (default)
    ref_no_null = Reference(to="TestSchema", definitions=definitions)
    with pytest.raises(ValidationError) as exc_info:
        ref_no_null.validate(None)
    assert exc_info.value.code() == "null"

    # Test case 3: validate with valid value that passes target validation
    definitions2 = Definitions()
    schema2 = Schema(fields={"name": Field()})
    definitions2["TestSchema"] = schema2
    ref2 = Reference(to="TestSchema", definitions=definitions2)
    test_value = {"name": "test"}
    result = ref2.validate(test_value)
    assert result == {"name": "test"}

    # Test case 4: validate with invalid value that fails target validation
    definitions3 = Definitions()
    from typesystem.fields import String
    schema3 = Schema(fields={"name": String(max_length=5)})
    definitions3["TestSchema"] = schema3
    ref3 = Reference(to="TestSchema", definitions=definitions3)
    with pytest.raises(ValidationError):
        ref3.validate({"name": "this is a very long string"})

    # Test case 5: validate with value missing required field
    definitions4 = Definitions()
    name_field = Field()
    schema4 = Schema(fields={"name": name_field})
    definitions4["TestSchema"] = schema4
    ref4 = Reference(to="TestSchema", definitions=definitions4)
    with pytest.raises(ValidationError):
        ref4.validate({})

    # Test case 6: validate with non-dict value that target schema rejects
    definitions5 = Definitions()
    schema5 = Schema(fields={"name": Field()})
    definitions5["TestSchema"] = schema5
    ref5 = Reference(to="TestSchema", definitions=definitions5)
    with pytest.raises(ValidationError) as exc_info:
        ref5.validate("not a dict")
    assert exc_info.value.code() == "type"


# LLM-generated content at query #33
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    schema = Schema(fields={"name": Field()})
    definitions["TestSchema"] = schema
    
    reference = Reference(to="TestSchema", definitions=definitions)
    
    # Test 1: Valid value passes through to target validation
    test_value = {"name": "test"}
    result = reference.validate(test_value)
    assert result == test_value
    
    # Test 2: None value with allow_null=True returns None
    reference_nullable = Reference(to="TestSchema", definitions=definitions, allow_null=True)
    result = reference_nullable.validate(None)
    assert result is None
    
    # Test 3: None value with allow_null=False raises validation error
    reference_not_nullable = Reference(to="TestSchema", definitions=definitions, allow_null=False)
    with pytest.raises(ValidationError) as exc_info:
        reference_not_nullable.validate(None)
    assert exc_info.value.code() == "null"
    
    # Test 4: Invalid value raises validation error from target
    with pytest.raises(ValidationError):
        reference.validate(None)
    
    # Test 5: Complex nested validation
    complex_schema = Schema(fields={
        "id": Field(),
        "data": Field()
    })
    definitions["ComplexSchema"] = complex_schema
    reference_complex = Reference(to="ComplexSchema", definitions=definitions)
    
    test_data = {"id": 1, "data": "value"}
    result = reference_complex.validate(test_data)
    assert result == test_data


# LLM-generated content at query #34
#--------------------------

```python
def test_Schema_validate():
    # Test with None value when allow_null is True
    schema = Schema(fields={"name": Field()}, allow_null=True)
    assert schema.validate(None) is None

    # Test with None value when allow_null is False
    schema = Schema(fields={"name": Field()}, allow_null=False)
    with pytest.raises(ValidationError) as exc_info:
        schema.validate(None)
    assert exc_info.value.code() == "null"

    # Test with non-dict/mapping type
    schema = Schema(fields={"name": Field()})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate("not a dict")
    assert exc_info.value.code() == "type"

    # Test with non-string keys
    schema = Schema(fields={"name": Field()})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({1: "value"})
    messages = exc_info.value.messages()
    assert any(msg.code == "invalid_key" for msg in messages)

    # Test with missing required field
    schema = Schema(fields={"name": Field()})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({})
    messages = exc_info.value.messages()
    assert any(msg.code == "required" for msg in messages)

    # Test with valid data
    schema = Schema(fields={"name": Field()})
    result = schema.validate({"name": "John"})
    assert result == {"name": "John"}

    # Test with field that has default value
    default_field = Field(default="default_value")
    schema = Schema(fields={"name": default_field})
    result = schema.validate({})
    assert result == {"name": "default_value"}

    # Test with read_only field
    read_only_field = Field(read_only=True)
    schema = Schema(fields={"id": read_only_field, "name": Field()})
    result = schema.validate({"id": 1, "name": "John"})
    assert "id" not in result
    assert result == {"name": "John"}

    # Test with nested field validation error
    nested_field = Field()
    nested_field.validate_or_error = lambda x: (None, ValidationError(
        messages=[Message(text="Invalid", code="invalid", index=[])]
    ))
    schema = Schema(fields={"nested": nested_field})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({"nested": "invalid"})
    assert len(exc_info.value.messages()) > 0

    # Test with mapping object (not dict)
    from collections import OrderedDict
    schema = Schema(fields={"name": Field()})
    result = schema.validate(OrderedDict([("name", "John")]))
    assert result == {"name": "John"}

    # Test with multiple fields, some required, some optional
    required_field = Field()
    optional_field = Field(allow_null=True)
    schema = Schema(fields={"required": required_field, "optional": optional_field})
    result = schema.validate({"required": "value"})
    assert result == {"required": "value"}

    # Test that read_only fields with defaults are not required
    read_only_with_default = Field(read_only=True, default="default")
    schema = Schema(fields={"id": read_only_with_default})
    result = schema.validate({})
    assert "id" not in result


# LLM-generated content at query #35
#--------------------------

```python
import pytest
from typesystem.base import ValidationError
from typesystem.fields import String, Integer, Boolean


def test_Schema_validate():
    # Test 1: Valid object
    schema = Schema(fields={
        "name": String(),
        "age": Integer(),
    })
    result = schema.validate({"name": "John", "age": 30})
    assert result == {"name": "John", "age": 30}

    # Test 2: None value with allow_null=True
    schema = Schema(fields={"name": String()}, allow_null=True)
    result = schema.validate(None)
    assert result is None

    # Test 3: None value with allow_null=False (default)
    schema = Schema(fields={"name": String()})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate(None)
    assert exc_info.value.code == "null"

    # Test 4: Non-dict/mapping type
    schema = Schema(fields={"name": String()})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate("not a dict")
    assert exc_info.value.code == "type"

    # Test 5: Non-string keys
    schema = Schema(fields={"name": String()})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({1: "value"})
    errors = exc_info.value.messages()
    assert any(msg.code == "invalid_key" for msg in errors)

    # Test 6: Missing required field
    schema = Schema(fields={"name": String()})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({})
    errors = exc_info.value.messages()
    assert any(msg.code == "required" and msg.index == ["name"] for msg in errors)

    # Test 7: Field with default value
    schema = Schema(fields={
        "name": String(),
        "age": Integer(default=0),
    })
    result = schema.validate({"name": "John"})
    assert result == {"name": "John", "age": 0}

    # Test 8: Read-only field is ignored
    name_field = String()
    name_field.read_only = True
    schema = Schema(fields={"name": name_field})
    result = schema.validate({})
    assert result == {}

    # Test 9: Child field validation error
    schema = Schema(fields={
        "name": String(),
        "age": Integer(),
    })
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({"name": "John", "age": "not_an_int"})
    errors = exc_info.value.messages()
    assert any(msg.index == ["age"] for msg in errors)

    # Test 10: Multiple errors
    schema = Schema(fields={
        "name": String(),
        "age": Integer(),
    })
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({1: "value", "age": "invalid"})
    errors = exc_info.value.messages()
    assert len(errors) >= 2

    # Test 11: Mapping type (not dict)
    from collections import UserDict
    schema = Schema(fields={"name": String()})
    result = schema.validate(UserDict({"name": "John"}))
    assert result == {"name": "John"}

    # Test 12: Extra fields are ignored
    schema = Schema(fields={"name": String()})
    result = schema.validate({"name": "John", "extra": "field"})
    assert result == {"name": "John"}
    assert "extra" not in result


# LLM-generated content at query #36
#--------------------------

```python
def test_Schema_validate():
    # Test with None when allow_null is True
    schema = Schema(fields={}, allow_null=True)
    assert schema.validate(None) is None

    # Test with None when allow_null is False
    schema = Schema(fields={}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert any(msg.code == "null" for msg in e.messages())

    # Test with non-dict/mapping type
    schema = Schema(fields={})
    try:
        schema.validate("not a dict")
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert any(msg.code == "type" for msg in e.messages())

    # Test with non-string keys
    schema = Schema(fields={})
    try:
        schema.validate({1: "value"})
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert any(msg.code == "invalid_key" for msg in e.messages())

    # Test with required field missing
    required_field = Field()
    schema = Schema(fields={"required_field": required_field})
    try:
        schema.validate({})
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert any(msg.code == "required" for msg in e.messages())

    # Test with valid data
    field = Field()
    schema = Schema(fields={"field": field})
    result = schema.validate({"field": "value"})
    assert result == {"field": "value"}

    # Test with field having default value
    field_with_default = Field(default="default_value")
    schema = Schema(fields={"field": field_with_default})
    result = schema.validate({})
    assert result == {"field": "default_value"}

    # Test with read_only field
    read_only_field = Field(read_only=True)
    schema = Schema(fields={"readonly": read_only_field})
    result = schema.validate({"readonly": "should_be_ignored"})
    assert "readonly" not in result

    # Test with nested validation error
    inner_field = Field()
    inner_field.validate = lambda x: (_ for _ in ()).throw(ValidationError(
        messages=[Message(text="Inner error", code="inner", index=[])]
    ))
    schema = Schema(fields={"nested": inner_field})
    try:
        schema.validate({"nested": "value"})
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert any(msg.code == "inner" for msg in e.messages())

    # Test with empty dict and no required fields
    schema = Schema(fields={})
    result = schema.validate({})
    assert result == {}

    # Test with mapping type (not dict)
    from collections import OrderedDict
    schema = Schema(fields={"field": Field()})
    mapping = OrderedDict([("field", "value")])
    result = schema.validate(mapping)
    assert result == {"field": "value"}

    # Test with multiple required fields
    field1 = Field()
    field2 = Field()
    schema = Schema(fields={"field1": field1, "field2": field2})
    try:
        schema.validate({"field1": "value"})
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        required_errors = [msg for msg in e.messages() if msg.code == "required"]
        assert len(required_errors) == 1


# LLM-generated content at query #37
#--------------------------

```python
def test_Schema_validate():
    # Test with None when allow_null is True
    schema = Schema(fields={}, allow_null=True)
    assert schema.validate(None) is None

    # Test with None when allow_null is False
    schema = Schema(fields={}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert e.code() == "null"

    # Test with non-dict/mapping type
    schema = Schema(fields={})
    try:
        schema.validate("invalid")
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert e.code() == "type"

    # Test with non-string keys
    schema = Schema(fields={})
    try:
        schema.validate({1: "value"})
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        messages = e.messages()
        assert any(msg.code == "invalid_key" for msg in messages)

    # Test with required field missing
    required_field = Field()
    schema = Schema(fields={"required_field": required_field})
    try:
        schema.validate({})
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        messages = e.messages()
        assert any(msg.code == "required" for msg in messages)

    # Test with valid data
    field = Field()
    schema = Schema(fields={"key": field})
    result = schema.validate({"key": "value"})
    assert result == {"key": "value"}

    # Test with field having default value
    field_with_default = Field(default="default_value")
    schema = Schema(fields={"field": field_with_default})
    result = schema.validate({})
    assert result == {"field": "default_value"}

    # Test with read_only field
    read_only_field = Field(read_only=True)
    schema = Schema(fields={"read_only": read_only_field})
    result = schema.validate({"read_only": "value"})
    assert "read_only" not in result

    # Test with mapping object (not dict)
    from collections import OrderedDict
    field = Field()
    schema = Schema(fields={"key": field})
    mapping = OrderedDict([("key", "value")])
    result = schema.validate(mapping)
    assert result == {"key": "value"}

    # Test with child field validation error
    class StrictField(Field):
        errors = {"invalid": "Invalid value"}
        def validate(self, value):
            if value != "valid":
                raise self.validation_error("invalid")
            return value

    strict_field = StrictField()
    schema = Schema(fields={"strict": strict_field})
    try:
        schema.validate({"strict": "invalid"})
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) > 0

    # Test with multiple fields, some valid, some invalid
    field1 = Field()
    strict_field = StrictField()
    schema = Schema(fields={"field1": field1, "strict": strict_field})
    try:
        schema.validate({"field1": "value1", "strict": "invalid"})
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) > 0

    # Test with empty valid object
    schema = Schema(fields={})
    result = schema.validate({})
    assert result == {}

    # Test with extra fields (should be ignored)
    field = Field()
    schema = Schema(fields={"key": field})
    result = schema.validate({"key": "value", "extra": "extra_value"})
    assert result == {"key": "value"}
    assert "extra" not in result


# LLM-generated content at query #38
#--------------------------

def test_Schema_validate():
    from typesystem.fields import String, Integer
    
    # Test 1: Valid object with required fields
    schema = Schema(fields={
        "name": String(),
        "age": Integer()
    })
    result = schema.validate({"name": "John", "age": 30})
    assert result == {"name": "John", "age": 30}
    
    # Test 2: None value with allow_null=True
    schema = Schema(fields={"name": String()}, allow_null=True)
    result = schema.validate(None)
    assert result is None
    
    # Test 3: None value with allow_null=False raises error
    schema = Schema(fields={"name": String()})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate(None)
    assert exc_info.value.code() == "null"
    
    # Test 4: Non-dict/mapping value raises type error
    schema = Schema(fields={"name": String()})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate("not a dict")
    assert exc_info.value.code() == "type"
    
    # Test 5: Non-string keys raise invalid_key error
    schema = Schema(fields={"name": String()})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({1: "value"})
    errors = exc_info.value.messages()
    assert any(msg.code == "invalid_key" for msg in errors)
    
    # Test 6: Missing required field raises required error
    schema = Schema(fields={"name": String(), "age": Integer()})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({"name": "John"})
    errors = exc_info.value.messages()
    assert any(msg.code == "required" for msg in errors)
    
    # Test 7: Field with default value fills in missing field
    schema = Schema(fields={
        "name": String(),
        "status": String(default="active")
    })
    result = schema.validate({"name": "John"})
    assert result == {"name": "John", "status": "active"}
    
    # Test 8: Read-only fields are skipped
    schema = Schema(fields={
        "name": String(),
        "id": String(read_only=True)
    })
    result = schema.validate({"name": "John", "id": "123"})
    assert result == {"name": "John"}
    assert "id" not in result
    
    # Test 9: Child field validation error is propagated
    schema = Schema(fields={
        "name": String(),
        "age": Integer()
    })
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({"name": "John", "age": "not_an_int"})
    errors = exc_info.value.messages()
    assert len(errors) > 0
    
    # Test 10: Multiple validation errors collected
    schema = Schema(fields={
        "name": String(),
        "age": Integer(),
        "email": String()
    })
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({1: "value", "age": "invalid"})
    errors = exc_info.value.messages()
    assert len(errors) >= 2
    
    # Test 11: Empty object with no required fields
    schema = Schema(fields={
        "optional_field": String(default="default")
    })
    result = schema.validate({})
    assert result == {"optional_field": "default"}
    
    # Test 12: Mapping object (not just dict)
    from collections import OrderedDict
    schema = Schema(fields={"name": String()})
    mapping = OrderedDict([("name", "John")])
    result = schema.validate(mapping)
    assert result == {"name": "John"}


# LLM-generated content at query #39
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    schema = Schema(fields={"name": Field()})
    definitions["TestSchema"] = schema
    
    # Test 1: validate with None when allow_null is True
    ref = Reference(to="TestSchema", definitions=definitions, allow_null=True)
    result = ref.validate(None)
    assert result is None
    
    # Test 2: validate with None when allow_null is False (default)
    ref = Reference(to="TestSchema", definitions=definitions)
    try:
        ref.validate(None)
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert any(msg.code == "null" for msg in e.messages())
    
    # Test 3: validate with valid dict value
    ref = Reference(to="TestSchema", definitions=definitions)
    test_value = {"name": "test"}
    result = ref.validate(test_value)
    assert result == {"name": "test"}
    
    # Test 4: validate with invalid value (non-dict) should delegate to target
    ref = Reference(to="TestSchema", definitions=definitions)
    try:
        ref.validate("invalid")
        assert False, "Should raise ValidationError"
    except ValidationError:
        pass
    
    # Test 5: target property returns correct definition
    ref = Reference(to="TestSchema", definitions=definitions)
    assert ref.target is schema
    
    # Test 6: validate with complex nested data
    complex_schema = Schema(fields={
        "id": Field(),
        "name": Field(),
    })
    definitions["ComplexSchema"] = complex_schema
    ref = Reference(to="ComplexSchema", definitions=definitions)
    test_data = {"id": 1, "name": "test"}
    result = ref.validate(test_data)
    assert result == {"id": 1, "name": "test"}


# LLM-generated content at query #40
#--------------------------

```python
def test_Reference_validate():
    # Test case 1: validate with None when allow_null is True
    definitions = Definitions()
    schema = Schema(fields={"name": Field()})
    definitions["TestSchema"] = schema
    ref = Reference(to="TestSchema", definitions=definitions, allow_null=True)
    
    result = ref.validate(None)
    assert result is None
    
    # Test case 2: validate with None when allow_null is False
    ref_not_null = Reference(to="TestSchema", definitions=definitions, allow_null=False)
    with pytest.raises(ValidationError) as exc_info:
        ref_not_null.validate(None)
    assert exc_info.value.code() == "null"
    
    # Test case 3: validate with valid value
    definitions2 = Definitions()
    name_field = Field()
    schema2 = Schema(fields={"name": name_field})
    definitions2["Person"] = schema2
    ref2 = Reference(to="Person", definitions=definitions2)
    
    test_data = {"name": "John"}
    result = ref2.validate(test_data)
    assert result == {"name": "John"}
    
    # Test case 4: validate with invalid data that target schema rejects
    definitions3 = Definitions()
    required_field = Field(allow_null=False)
    schema3 = Schema(fields={"required_field": required_field})
    definitions3["StrictSchema"] = schema3
    ref3 = Reference(to="StrictSchema", definitions=definitions3)
    
    invalid_data = {"required_field": None}
    with pytest.raises(ValidationError):
        ref3.validate(invalid_data)
    
    # Test case 5: validate delegates to target schema's validate method
    definitions4 = Definitions()
    inner_schema = Schema(fields={})
    definitions4["EmptySchema"] = inner_schema
    ref4 = Reference(to="EmptySchema", definitions=definitions4)
    
    result = ref4.validate({})
    assert result == {}


# LLM-generated content at query #41
#--------------------------

```python
import pytest
from typesystem.base import ValidationError
from typesystem.fields import Field, String, Integer, Boolean


def test_Schema_validate():
    # Test with None and allow_null=True
    schema = Schema(fields={"name": String()}, allow_null=True)
    assert schema.validate(None) is None

    # Test with None and allow_null=False
    schema = Schema(fields={"name": String()})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate(None)
    assert exc_info.value.code == "null"

    # Test with non-dict/non-Mapping type
    schema = Schema(fields={"name": String()})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate("invalid")
    assert exc_info.value.code == "type"

    # Test with non-string keys
    schema = Schema(fields={"name": String()})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({1: "value"})
    assert any(msg.code == "invalid_key" for msg in exc_info.value.messages())

    # Test with required field missing
    schema = Schema(fields={"name": String()})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({})
    assert any(msg.code == "required" for msg in exc_info.value.messages())

    # Test with valid data
    schema = Schema(fields={"name": String(), "age": Integer()})
    result = schema.validate({"name": "John", "age": 30})
    assert result == {"name": "John", "age": 30}

    # Test with optional field not provided
    optional_field = String(allow_null=True)
    schema = Schema(fields={"name": String(), "email": optional_field})
    result = schema.validate({"name": "John"})
    assert result == {"name": "John"}

    # Test with field having default value
    schema = Schema(fields={"name": String(), "active": Boolean(default=True)})
    result = schema.validate({"name": "John"})
    assert result == {"name": "John", "active": True}

    # Test with read_only field
    schema = Schema(fields={"name": String(), "id": String(read_only=True)})
    result = schema.validate({"name": "John", "id": "123"})
    assert result == {"name": "John"}
    assert "id" not in result

    # Test with nested validation error
    schema = Schema(fields={"name": String(), "age": Integer()})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({"name": "John", "age": "not_an_int"})
    assert len(exc_info.value.messages()) > 0

    # Test with empty dict and no required fields
    schema = Schema(
        fields={"name": String(default="Unknown"), "age": Integer(default=0)}
    )
    result = schema.validate({})
    assert result == {"name": "Unknown", "age": 0}

    # Test with extra fields (should be ignored)
    schema = Schema(fields={"name": String()})
    result = schema.validate({"name": "John", "extra": "field"})
    assert result == {"name": "John"}

    # Test with Mapping type instead of dict
    from collections import OrderedDict
    schema = Schema(fields={"name": String()})
    result = schema.validate(OrderedDict([("name", "John")]))
    assert result == {"name": "John"}

    # Test with multiple validation errors
    schema = Schema(fields={"name": String(), "age": Integer()})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({})
    messages = exc_info.value.messages()
    assert len(messages) == 2
    assert any(msg.code == "required" for msg in messages)


# LLM-generated content at query #42
#--------------------------

```python
def test_Schema_validate():
    # Test with None value and allow_null=True
    schema = Schema(fields={}, allow_null=True)
    assert schema.validate(None) is None

    # Test with None value and allow_null=False
    schema = Schema(fields={}, allow_null=False)
    with pytest.raises(ValidationError) as exc_info:
        schema.validate(None)
    assert exc_info.value.code == "null"

    # Test with non-dict/non-mapping value
    schema = Schema(fields={})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate("invalid")
    assert exc_info.value.code == "type"

    # Test with list (not a mapping)
    with pytest.raises(ValidationError) as exc_info:
        schema.validate([1, 2, 3])
    assert exc_info.value.code == "type"

    # Test with valid empty dict
    schema = Schema(fields={})
    assert schema.validate({}) == {}

    # Test with non-string keys
    schema = Schema(fields={})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({1: "value"})
    messages = exc_info.value.messages()
    assert any(msg.code == "invalid_key" for msg in messages)

    # Test with required field missing
    required_field = Field()
    schema = Schema(fields={"name": required_field})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({})
    messages = exc_info.value.messages()
    assert any(msg.code == "required" for msg in messages)

    # Test with valid required field
    schema = Schema(fields={"name": Field()})
    result = schema.validate({"name": "John"})
    assert result == {"name": "John"}

    # Test with field having default value
    default_field = Field(default="default_value")
    schema = Schema(fields={"status": default_field})
    result = schema.validate({})
    assert result == {"status": "default_value"}

    # Test with read_only field (should be skipped)
    read_only_field = Field(read_only=True)
    schema = Schema(fields={"id": read_only_field})
    result = schema.validate({"id": 123})
    assert "id" not in result

    # Test with nested field validation error
    inner_field = Field()
    schema = Schema(fields={"nested": inner_field})
    with pytest.raises(ValidationError):
        schema.validate({"nested": None})

    # Test with multiple fields
    schema = Schema(fields={
        "name": Field(),
        "age": Field()
    })
    result = schema.validate({"name": "John", "age": 30})
    assert result == {"name": "John", "age": 30}

    # Test with extra fields in input (should be ignored)
    schema = Schema(fields={"name": Field()})
    result = schema.validate({"name": "John", "extra": "field"})
    assert result == {"name": "John"}

    # Test with Mapping instead of dict
    from collections import OrderedDict
    mapping = OrderedDict([("name", "Jane")])
    schema = Schema(fields={"name": Field()})
    result = schema.validate(mapping)
    assert result == {"name": "Jane"}

    # Test with multiple validation errors
    schema = Schema(fields={
        "field1": Field(),
        "field2": Field()
    })
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({"field1": None})
    messages = exc_info.value.messages()
    assert len(messages) > 0


# LLM-generated content at query #43
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    schema = Schema(fields={"name": Field()})
    definitions["TestSchema"] = schema
    
    reference = Reference(to="TestSchema", definitions=definitions)
    
    # Test valid value
    test_value = {"name": "test"}
    result = reference.validate(test_value)
    assert result == {"name": "test"}
    
    # Test with allow_null=True and None value
    reference_nullable = Reference(to="TestSchema", definitions=definitions, allow_null=True)
    result = reference_nullable.validate(None)
    assert result is None
    
    # Test with allow_null=False (default) and None value
    reference_non_nullable = Reference(to="TestSchema", definitions=definitions, allow_null=False)
    with pytest.raises(ValidationError) as exc_info:
        reference_non_nullable.validate(None)
    assert exc_info.value.messages()[0].code == "null"
    
    # Test that validation is delegated to target schema
    reference_with_schema = Reference(to="TestSchema", definitions=definitions)
    valid_input = {"name": "John"}
    result = reference_with_schema.validate(valid_input)
    assert result == {"name": "John"}
    
    # Test invalid input raises error from target schema
    with pytest.raises(ValidationError):
        reference_with_schema.validate("not a dict")


# LLM-generated content at query #44
#--------------------------

```python
def test_Reference_validate():
    # Test setup
    definitions = Definitions()
    schema = Schema(fields={"name": Field()})
    definitions["TestSchema"] = schema
    
    reference = Reference(to="TestSchema", definitions=definitions)
    
    # Test: validate with None when allow_null is False (default)
    try:
        reference.validate(None)
        assert False, "Should raise validation error"
    except ValidationError as e:
        assert e.code == "null"
    
    # Test: validate with None when allow_null is True
    reference_nullable = Reference(to="TestSchema", definitions=definitions, allow_null=True)
    result = reference_nullable.validate(None)
    assert result is None
    
    # Test: validate with valid dict value
    valid_data = {"name": "test"}
    result = reference.validate(valid_data)
    assert result == valid_data
    
    # Test: validate delegates to target schema
    definitions_with_field = Definitions()
    schema_with_field = Schema(fields={"name": Field(allow_null=False)})
    definitions_with_field["RequiredSchema"] = schema_with_field
    reference_with_field = Reference(to="RequiredSchema", definitions=definitions_with_field)
    
    # Valid data should pass through
    valid_result = reference_with_field.validate({"name": "John"})
    assert valid_result == {"name": "John"}
    
    # Test: validate with empty dict
    empty_result = reference.validate({})
    assert empty_result == {}


# LLM-generated content at query #45
#--------------------------

```python
def test_Reference_validate():
    # Test with null value when allow_null is True
    definitions = Definitions()
    schema = Schema(fields={"name": Field()})
    definitions["TestSchema"] = schema
    ref = Reference(to="TestSchema", definitions=definitions, allow_null=True)
    
    result = ref.validate(None)
    assert result is None
    
    # Test with null value when allow_null is False (default)
    ref_not_null = Reference(to="TestSchema", definitions=definitions)
    with pytest.raises(ValidationError) as exc_info:
        ref_not_null.validate(None)
    assert exc_info.value.code == "null"
    
    # Test with valid value that passes target validation
    test_data = {"name": "test"}
    result = ref.validate(test_data)
    assert result == test_data
    
    # Test with invalid value that fails target validation
    invalid_data = None
    with pytest.raises(ValidationError):
        ref_not_null.validate(invalid_data)
    
    # Test reference to different schema types
    definitions2 = Definitions()
    simple_field = Field()
    definitions2["SimpleField"] = simple_field
    ref2 = Reference(to="SimpleField", definitions=definitions2)
    
    result2 = ref2.validate("test_value")
    assert result2 == "test_value"


####################################################################
#    TEST GENERATION BEGINS (CODAMOSA + claude-haiku-4-5 t=0.8)    #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_Definitions___setitem__():
    """Test __setitem__ method of Definitions class."""
    definitions = Definitions()
    
    # Test setting a new item
    definitions["schema1"] = "value1"
    assert definitions["schema1"] == "value1"
    
    # Test setting multiple items
    definitions["schema2"] = "value2"
    definitions["schema3"] = "value3"
    assert definitions["schema2"] == "value2"
    assert definitions["schema3"] == "value3"
    
    # Test that setting an existing key raises AssertionError
    with pytest.raises(AssertionError):
        definitions["schema1"] = "new_value"
    
    # Test with different value types
    definitions["schema_dict"] = {"key": "value"}
    assert definitions["schema_dict"] == {"key": "value"}
    
    definitions["schema_list"] = [1, 2, 3]
    assert definitions["schema_list"] == [1, 2, 3]
    
    definitions["schema_none"] = None
    assert definitions["schema_none"] is None
    
    # Test initialization with items and then setting new items
    definitions2 = Definitions({"initial": "value"})
    assert definitions2["initial"] == "value"
    
    with pytest.raises(AssertionError):
        definitions2["initial"] = "new_value"
    
    definitions2["new_key"] = "new_value"
    assert definitions2["new_key"] == "new_value"


# LLM-generated content at query #2
#--------------------------

```python
def test_Definitions___setitem__():
    # Test basic setitem functionality
    definitions = Definitions()
    definitions["schema1"] = "value1"
    assert definitions["schema1"] == "value1"
    
    # Test multiple items
    definitions["schema2"] = "value2"
    assert definitions["schema2"] == "value2"
    assert definitions["schema1"] == "value1"
    
    # Test that duplicate key raises AssertionError
    with pytest.raises(AssertionError):
        definitions["schema1"] = "new_value"
    
    # Test with different value types
    definitions["schema3"] = {"nested": "dict"}
    assert definitions["schema3"] == {"nested": "dict"}
    
    definitions["schema4"] = 42
    assert definitions["schema4"] == 42
    
    definitions["schema5"] = None
    assert definitions["schema5"] is None
    
    # Test that original value is unchanged after failed duplicate attempt
    with pytest.raises(AssertionError):
        definitions["schema3"] = "different_value"
    assert definitions["schema3"] == {"nested": "dict"}


# LLM-generated content at query #3
#--------------------------

```python
def test_Schema():
    # Test basic schema initialization
    fields = {
        "name": Field(),
        "age": Field(),
    }
    schema = Schema(fields=fields)
    assert schema.fields == fields
    assert schema.required == ["name", "age"]

    # Test schema with read_only fields
    read_only_field = Field(read_only=True)
    fields_with_readonly = {
        "id": read_only_field,
        "name": Field(),
    }
    schema = Schema(fields=fields_with_readonly)
    assert schema.required == ["name"]

    # Test schema with default values
    field_with_default = Field(default="default_value")
    fields_with_default = {
        "name": Field(),
        "status": field_with_default,
    }
    schema = Schema(fields=fields_with_default)
    assert schema.required == ["name"]

    # Test schema with both read_only and default fields
    fields_mixed = {
        "id": Field(read_only=True),
        "name": Field(),
        "status": Field(default="active"),
        "email": Field(),
    }
    schema = Schema(fields=fields_mixed)
    assert set(schema.required) == {"name", "email"}

    # Test schema with all fields having defaults or read_only
    fields_all_optional = {
        "id": Field(read_only=True),
        "status": Field(default="active"),
    }
    schema = Schema(fields=fields_all_optional)
    assert schema.required == []

    # Test schema with allow_null parameter
    schema_nullable = Schema(fields=fields, allow_null=True)
    assert schema_nullable.allow_null is True
    assert schema_nullable.required == ["name", "age"]

    # Test schema with empty fields
    empty_schema = Schema(fields={})
    assert empty_schema.fields == {}
    assert empty_schema.required == []

    # Test that kwargs are passed to parent Field class
    schema_with_kwargs = Schema(fields=fields, allow_null=True)
    assert schema_with_kwargs.allow_null is True


# LLM-generated content at query #4
#--------------------------

```python
def test_Definitions___setitem__():
    """Test __setitem__ method of Definitions class."""
    definitions = Definitions()
    
    # Test setting a new item
    definitions["schema1"] = "value1"
    assert definitions["schema1"] == "value1"
    
    # Test setting another new item
    definitions["schema2"] = "value2"
    assert definitions["schema2"] == "value2"
    assert len(definitions) == 2
    
    # Test that setting an already existing key raises AssertionError
    with pytest.raises(AssertionError):
        definitions["schema1"] = "new_value"
    
    # Verify the original value is unchanged
    assert definitions["schema1"] == "value1"
    
    # Test with different types of values
    definitions["list_key"] = [1, 2, 3]
    assert definitions["list_key"] == [1, 2, 3]
    
    definitions["dict_key"] = {"nested": "dict"}
    assert definitions["dict_key"] == {"nested": "dict"}
    
    # Test that duplicate keys still raise AssertionError
    with pytest.raises(AssertionError):
        definitions["list_key"] = [4, 5, 6]
    
    # Test initialization with items
    definitions_with_init = Definitions({"initial_key": "initial_value"})
    assert definitions_with_init["initial_key"] == "initial_value"
    
    # Test that we cannot override initial items
    with pytest.raises(AssertionError):
        definitions_with_init["initial_key"] = "new_value"
    
    # But can add new items
    definitions_with_init["new_key"] = "new_value"
    assert definitions_with_init["new_key"] == "new_value"


# LLM-generated content at query #5
#--------------------------

```python
def test_Schema_serialize():
    from typesystem.fields import String, Integer, Boolean
    
    # Test basic serialization with dict
    schema = Schema(fields={
        "name": String(),
        "age": Integer(),
        "active": Boolean()
    })
    
    obj = {"name": "John", "age": 30, "active": True}
    result = schema.serialize(obj)
    assert result == {"name": "John", "age": 30, "active": True}
    
    # Test serialization with None
    result = schema.serialize(None)
    assert result is None
    
    # Test serialization with object (using getattr)
    class Person:
        def __init__(self, name, age, active):
            self.name = name
            self.age = age
            self.active = active
    
    person = Person("Jane", 25, False)
    result = schema.serialize(person)
    assert result == {"name": "Jane", "age": 25, "active": False}
    
    # Test serialization with missing keys in dict
    obj_partial = {"name": "Bob"}
    result = schema.serialize(obj_partial)
    assert result == {"name": "Bob"}
    
    # Test serialization with missing attributes in object
    class PartialPerson:
        def __init__(self, name):
            self.name = name
    
    partial_person = PartialPerson("Alice")
    result = schema.serialize(partial_person)
    assert result == {"name": "Alice"}
    
    # Test serialization with read_only fields (should be skipped)
    schema_with_readonly = Schema(fields={
        "name": String(),
        "id": Integer(read_only=True)
    })
    
    obj_with_readonly = {"name": "Charlie", "id": 123}
    result = schema_with_readonly.serialize(obj_with_readonly)
    assert result == {"name": "Charlie"}
    
    # Test serialization with nested fields
    nested_schema = Schema(fields={
        "username": String(),
        "details": Schema(fields={
            "email": String(),
            "phone": String()
        })
    })
    
    nested_obj = {
        "username": "user1",
        "details": {"email": "user@example.com", "phone": "555-1234"}
    }
    result = nested_schema.serialize(nested_obj)
    assert result["username"] == "user1"
    assert result["details"]["email"] == "user@example.com"
    assert result["details"]["phone"] == "555-1234"


# LLM-generated content at query #6
#--------------------------

```python
def test_Schema_validate():
    # Test with None value when allow_null is True
    schema = Schema(fields={}, allow_null=True)
    assert schema.validate(None) is None

    # Test with None value when allow_null is False
    schema = Schema(fields={}, allow_null=False)
    with pytest.raises(ValidationError) as exc_info:
        schema.validate(None)
    assert exc_info.value.code == "null"

    # Test with non-dict/mapping type
    schema = Schema(fields={})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate("not a dict")
    assert exc_info.value.code == "type"

    # Test with list
    schema = Schema(fields={})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate([1, 2, 3])
    assert exc_info.value.code == "type"

    # Test with non-string keys
    field = Field()
    schema = Schema(fields={"name": field})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({1: "value"})
    messages = exc_info.value.messages()
    assert any(msg.code == "invalid_key" for msg in messages)

    # Test with required field missing
    field = Field()
    schema = Schema(fields={"name": field})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({})
    messages = exc_info.value.messages()
    assert any(msg.code == "required" for msg in messages)

    # Test with valid data
    field = Field()
    schema = Schema(fields={"name": field})
    result = schema.validate({"name": "John"})
    assert result == {"name": "John"}

    # Test with read_only field (should be ignored)
    read_only_field = Field(read_only=True)
    regular_field = Field()
    schema = Schema(fields={"id": read_only_field, "name": regular_field})
    result = schema.validate({"name": "John", "id": 123})
    assert result == {"name": "John"}
    assert "id" not in result

    # Test with field having default value
    field_with_default = Field(default="default_value")
    schema = Schema(fields={"status": field_with_default})
    result = schema.validate({})
    assert result == {"status": "default_value"}

    # Test with nested validation error
    child_field = Field()
    schema = Schema(fields={"nested": child_field})
    with pytest.raises(ValidationError):
        schema.validate({"nested": "value"})

    # Test with empty dict
    schema = Schema(fields={})
    result = schema.validate({})
    assert result == {}

    # Test with Mapping type instead of dict
    from collections import OrderedDict
    field = Field()
    schema = Schema(fields={"name": field})
    result = schema.validate(OrderedDict([("name", "John")]))
    assert result == {"name": "John"}

    # Test with multiple fields
    field1 = Field()
    field2 = Field()
    schema = Schema(fields={"name": field1, "age": field2})
    result = schema.validate({"name": "John", "age": 30})
    assert result == {"name": "John", "age": 30}

    # Test with extra fields not in schema (should be ignored)
    field = Field()
    schema = Schema(fields={"name": field})
    result = schema.validate({"name": "John", "extra": "field"})
    assert result == {"name": "John"}
    assert "extra" not in result


# LLM-generated content at query #7
#--------------------------

```python
def test_Schema_serialize():
    # Test serialize with None value
    field1 = Field()
    schema = Schema(fields={"key1": field1})
    assert schema.serialize(None) is None

    # Test serialize with dict object
    field1 = Field()
    field2 = Field()
    schema = Schema(fields={"key1": field1, "key2": field2})
    obj = {"key1": "value1", "key2": "value2"}
    result = schema.serialize(obj)
    assert result == {"key1": "value1", "key2": "value2"}

    # Test serialize with object attributes
    class TestObj:
        def __init__(self):
            self.key1 = "value1"
            self.key2 = "value2"

    field1 = Field()
    field2 = Field()
    schema = Schema(fields={"key1": field1, "key2": field2})
    obj = TestObj()
    result = schema.serialize(obj)
    assert result == {"key1": "value1", "key2": "value2"}

    # Test serialize with missing keys
    field1 = Field()
    field2 = Field()
    schema = Schema(fields={"key1": field1, "key2": field2})
    obj = {"key1": "value1"}
    result = schema.serialize(obj)
    assert result == {"key1": "value1"}

    # Test serialize with missing attributes on object
    class TestObj2:
        def __init__(self):
            self.key1 = "value1"

    field1 = Field()
    field2 = Field()
    schema = Schema(fields={"key1": field1, "key2": field2})
    obj = TestObj2()
    result = schema.serialize(obj)
    assert result == {"key1": "value1"}

    # Test serialize with nested fields
    inner_field = Field()
    outer_field = Field()
    schema = Schema(fields={"outer": outer_field, "inner": inner_field})
    obj = {"outer": "outer_value", "inner": "inner_value"}
    result = schema.serialize(obj)
    assert result == {"outer": "outer_value", "inner": "inner_value"}

    # Test serialize with empty dict
    schema = Schema(fields={})
    obj = {"key1": "value1"}
    result = schema.serialize(obj)
    assert result == {}

    # Test serialize with MutableMapping
    from collections import OrderedDict
    field1 = Field()
    field2 = Field()
    schema = Schema(fields={"key1": field1, "key2": field2})
    obj = OrderedDict([("key1", "value1"), ("key2", "value2")])
    result = schema.serialize(obj)
    assert result == {"key1": "value1", "key2": "value2"}


# LLM-generated content at query #8
#--------------------------

```python
def test_Schema_serialize():
    # Test with None value
    from typesystem.fields import String, Integer
    
    schema = Schema(fields={
        "name": String(),
        "age": Integer(),
    })
    
    assert schema.serialize(None) is None
    
    # Test with dict object
    obj = {"name": "John", "age": 30}
    result = schema.serialize(obj)
    assert result == {"name": "John", "age": 30}
    
    # Test with object having attributes
    class Person:
        def __init__(self, name, age):
            self.name = name
            self.age = age
    
    person = Person("Jane", 25)
    result = schema.serialize(person)
    assert result == {"name": "Jane", "age": 25}
    
    # Test with missing keys in dict
    obj_partial = {"name": "Bob"}
    result = schema.serialize(obj_partial)
    assert result == {"name": "Bob"}
    
    # Test with missing attributes in object
    class PartialPerson:
        def __init__(self, name):
            self.name = name
    
    partial_person = PartialPerson("Alice")
    result = schema.serialize(partial_person)
    assert result == {"name": "Alice"}
    
    # Test with read_only field
    schema_readonly = Schema(fields={
        "name": String(),
        "id": Integer(read_only=True),
    })
    obj_with_readonly = {"name": "Charlie", "id": 1}
    result = schema_readonly.serialize(obj_with_readonly)
    assert result == {"name": "Charlie"}
    assert "id" not in result
    
    # Test with nested field serialization
    from typesystem.fields import Boolean
    
    schema_complex = Schema(fields={
        "name": String(),
        "active": Boolean(),
    })
    obj_complex = {"name": "David", "active": True}
    result = schema_complex.serialize(obj_complex)
    assert result == {"name": "David", "active": True}


# LLM-generated content at query #9
#--------------------------

```python
def test_Schema_serialize():
    """Test Schema.serialize() method with various inputs."""
    
    # Test with None value
    string_field = Field()
    schema = Schema(fields={"name": string_field})
    assert schema.serialize(None) is None
    
    # Test with dict object
    schema = Schema(fields={"name": Field(), "age": Field()})
    obj = {"name": "John", "age": 30}
    result = schema.serialize(obj)
    assert result == {"name": "John", "age": 30}
    
    # Test with object (non-dict) with attributes
    class Person:
        def __init__(self, name, age):
            self.name = name
            self.age = age
    
    person = Person("Jane", 25)
    result = schema.serialize(person)
    assert result == {"name": "Jane", "age": 25}
    
    # Test with missing keys/attributes
    schema = Schema(fields={"name": Field(), "age": Field(), "email": Field()})
    obj = {"name": "Bob", "age": 35}
    result = schema.serialize(obj)
    assert result == {"name": "Bob", "age": 35}
    assert "email" not in result
    
    # Test with missing attributes on object
    person = Person("Alice", 28)
    result = schema.serialize(person)
    assert result == {"name": "Alice", "age": 28}
    
    # Test with empty dict
    schema = Schema(fields={"name": Field()})
    result = schema.serialize({})
    assert result == {}
    
    # Test with nested field serialization
    mock_field = Field()
    mock_field.serialize = lambda x: str(x).upper()
    schema = Schema(fields={"name": mock_field})
    result = schema.serialize({"name": "test"})
    assert result == {"name": "TEST"}
    
    # Test with read_only fields (should be included in output)
    read_only_field = Field(read_only=True)
    regular_field = Field()
    schema = Schema(fields={"id": read_only_field, "name": regular_field})
    obj = {"id": 1, "name": "John"}
    result = schema.serialize(obj)
    assert result == {"id": 1, "name": "John"}
    
    # Test with KeyError on dict access
    schema = Schema(fields={"name": Field(), "age": Field()})
    obj = {"name": "John"}
    result = schema.serialize(obj)
    assert result == {"name": "John"}
    
    # Test with AttributeError on object access
    person = Person("John", 30)
    schema = Schema(fields={"name": Field(), "age": Field(), "email": Field()})
    result = schema.serialize(person)
    assert result == {"name": "John", "age": 30}


# LLM-generated content at query #10
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    schema = Schema(fields={"name": Field(), "age": Field()})
    definitions["Person"] = schema
    
    reference = Reference(to="Person", definitions=definitions)
    
    # Test 1: Valid value passes through to target validation
    valid_data = {"name": "John", "age": 30}
    result = reference.validate(valid_data)
    assert result == valid_data
    
    # Test 2: None with allow_null=True returns None
    reference_nullable = Reference(to="Person", definitions=definitions, allow_null=True)
    assert reference_nullable.validate(None) is None
    
    # Test 3: None with allow_null=False raises validation error
    with pytest.raises(ValidationError) as exc_info:
        reference.validate(None)
    assert exc_info.value.code == "null"
    
    # Test 4: Invalid data raises validation error from target
    invalid_data = None
    reference_non_nullable = Reference(to="Person", definitions=definitions, allow_null=False)
    with pytest.raises(ValidationError):
        reference_non_nullable.validate(invalid_data)
    
    # Test 5: Complex nested validation through reference
    complex_schema = Schema(fields={
        "id": Field(),
        "data": Field()
    })
    definitions["ComplexType"] = complex_schema
    complex_ref = Reference(to="ComplexType", definitions=definitions)
    
    complex_data = {"id": 1, "data": "test"}
    result = complex_ref.validate(complex_data)
    assert result == complex_data
    
    # Test 6: Target property returns correct schema
    assert reference.target == schema


# LLM-generated content at query #11
#--------------------------

```python
def test_Reference_validate():
    # Test setup
    definitions = Definitions()
    schema = Schema(fields={"name": Field()})
    definitions["TestSchema"] = schema
    
    reference = Reference(to="TestSchema", definitions=definitions)
    
    # Test 1: Valid value passes through to target validation
    test_value = {"name": "test"}
    result = reference.validate(test_value)
    assert result == {"name": "test"}
    
    # Test 2: None with allow_null=True returns None
    reference_nullable = Reference(to="TestSchema", definitions=definitions, allow_null=True)
    result = reference_nullable.validate(None)
    assert result is None
    
    # Test 3: None with allow_null=False raises validation error
    reference_not_nullable = Reference(to="TestSchema", definitions=definitions, allow_null=False)
    with pytest.raises(ValidationError) as exc_info:
        reference_not_nullable.validate(None)
    assert exc_info.value.code == "null"
    
    # Test 4: Invalid value raises validation error from target schema
    with pytest.raises(ValidationError):
        reference.validate(None)
    
    # Test 5: Reference resolves to correct target schema
    assert reference.target == schema
    
    # Test 6: Multiple references to same definition
    reference2 = Reference(to="TestSchema", definitions=definitions)
    assert reference.target is reference2.target


# LLM-generated content at query #12
#--------------------------

```python
def test_Reference_validate():
    # Test case 1: validate with None when allow_null is True
    definitions = Definitions()
    schema = Schema(fields={"name": Field()})
    definitions["TestSchema"] = schema
    ref = Reference(to="TestSchema", definitions=definitions, allow_null=True)
    assert ref.validate(None) is None

    # Test case 2: validate with None when allow_null is False (should raise error)
    ref_not_null = Reference(to="TestSchema", definitions=definitions, allow_null=False)
    try:
        ref_not_null.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code() == "null"

    # Test case 3: validate with valid data
    test_data = {"name": "John"}
    result = ref.validate(test_data)
    assert result == {"name": "John"}

    # Test case 4: validate delegates to target schema
    definitions2 = Definitions()
    schema2 = Schema(fields={"age": Field()})
    definitions2["AgeSchema"] = schema2
    ref2 = Reference(to="AgeSchema", definitions=definitions2, allow_null=False)
    test_data2 = {"age": 25}
    result2 = ref2.validate(test_data2)
    assert result2 == {"age": 25}

    # Test case 5: validate with invalid data should propagate target errors
    definitions3 = Definitions()
    schema3 = Schema(fields={})
    definitions3["EmptySchema"] = schema3
    ref3 = Reference(to="EmptySchema", definitions=definitions3, allow_null=False)
    try:
        ref3.validate("not a dict")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code() == "type"


# LLM-generated content at query #13
#--------------------------

```python
def test_Schema_validate():
    # Test with None value and allow_null=True
    schema = Schema(fields={}, allow_null=True)
    assert schema.validate(None) is None

    # Test with None value and allow_null=False
    schema = Schema(fields={}, allow_null=False)
    with pytest.raises(ValidationError) as exc_info:
        schema.validate(None)
    assert exc_info.value.messages()[0].code == "null"

    # Test with non-dict/mapping value
    schema = Schema(fields={})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate("invalid")
    assert exc_info.value.messages()[0].code == "type"

    # Test with valid empty object
    schema = Schema(fields={})
    assert schema.validate({}) == {}

    # Test with non-string keys
    schema = Schema(fields={})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({1: "value"})
    assert any(msg.code == "invalid_key" for msg in exc_info.value.messages())

    # Test with required field missing
    required_field = Field()
    schema = Schema(fields={"name": required_field})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({})
    assert any(msg.code == "required" for msg in exc_info.value.messages())

    # Test with valid required field
    schema = Schema(fields={"name": Field()})
    result = schema.validate({"name": "John"})
    assert result == {"name": "John"}

    # Test with read_only field
    read_only_field = Field(read_only=True)
    schema = Schema(fields={"id": read_only_field})
    result = schema.validate({"id": 123})
    assert result == {}

    # Test with field having default value
    default_field = Field(default="default_value")
    schema = Schema(fields={"status": default_field})
    result = schema.validate({})
    assert result == {"status": "default_value"}

    # Test with multiple fields and validation errors
    from typesystem.fields import String, Integer
    schema = Schema(
        fields={
            "name": String(max_length=5),
            "age": Integer(),
        }
    )
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({"name": "VeryLongName", "age": "not_an_int"})
    messages = exc_info.value.messages()
    assert len(messages) > 0

    # Test with valid mapping object
    schema = Schema(fields={"key": Field()})
    mapping = {"key": "value"}
    result = schema.validate(mapping)
    assert result == {"key": "value"}

    # Test with extra fields not in schema
    schema = Schema(fields={"name": Field()})
    result = schema.validate({"name": "John", "extra": "field"})
    assert result == {"name": "John"}
    assert "extra" not in result


# LLM-generated content at query #14
#--------------------------

```python
def test_Schema_serialize():
    # Test with None
    field1 = Field()
    schema = Schema(fields={"key1": field1})
    assert schema.serialize(None) is None

    # Test with dict mapping
    field1 = Field()
    field2 = Field()
    schema = Schema(fields={"key1": field1, "key2": field2})
    obj = {"key1": "value1", "key2": "value2"}
    result = schema.serialize(obj)
    assert result == {"key1": "value1", "key2": "value2"}

    # Test with object (non-dict)
    class TestObj:
        def __init__(self):
            self.key1 = "value1"
            self.key2 = "value2"

    field1 = Field()
    field2 = Field()
    schema = Schema(fields={"key1": field1, "key2": field2})
    obj = TestObj()
    result = schema.serialize(obj)
    assert result == {"key1": "value1", "key2": "value2"}

    # Test with missing keys
    field1 = Field()
    field2 = Field()
    schema = Schema(fields={"key1": field1, "key2": field2})
    obj = {"key1": "value1"}
    result = schema.serialize(obj)
    assert result == {"key1": "value1"}

    # Test with extra keys in dict (should be ignored)
    field1 = Field()
    schema = Schema(fields={"key1": field1})
    obj = {"key1": "value1", "extra": "extra_value"}
    result = schema.serialize(obj)
    assert result == {"key1": "value1"}

    # Test with object missing attribute
    class TestObjMissing:
        def __init__(self):
            self.key1 = "value1"

    field1 = Field()
    field2 = Field()
    schema = Schema(fields={"key1": field1, "key2": field2})
    obj = TestObjMissing()
    result = schema.serialize(obj)
    assert result == {"key1": "value1"}

    # Test with nested field serialization
    class CustomField(Field):
        def serialize(self, obj):
            return str(obj).upper() if obj else None

    custom_field = CustomField()
    field2 = Field()
    schema = Schema(fields={"key1": custom_field, "key2": field2})
    obj = {"key1": "value1", "key2": "value2"}
    result = schema.serialize(obj)
    assert result == {"key1": "VALUE1", "key2": "value2"}

    # Test with empty fields
    schema = Schema(fields={})
    obj = {"key1": "value1"}
    result = schema.serialize(obj)
    assert result == {}

    # Test with dict having None values
    field1 = Field()
    field2 = Field()
    schema = Schema(fields={"key1": field1, "key2": field2})
    obj = {"key1": None, "key2": "value2"}
    result = schema.serialize(obj)
    assert result == {"key1": None, "key2": "value2"}


# LLM-generated content at query #15
#--------------------------

```python
def test_Schema_serialize():
    # Test serialize with None
    field1 = Field()
    schema = Schema(fields={"key1": field1})
    assert schema.serialize(None) is None

    # Test serialize with dict
    field1 = Field()
    field2 = Field()
    schema = Schema(fields={"key1": field1, "key2": field2})
    obj = {"key1": "value1", "key2": "value2"}
    result = schema.serialize(obj)
    assert result == {"key1": "value1", "key2": "value2"}

    # Test serialize with object attributes
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

    # Test serialize with missing keys
    field1 = Field()
    field2 = Field()
    schema = Schema(fields={"key1": field1, "key2": field2})
    obj = {"key1": "value1"}
    result = schema.serialize(obj)
    assert result == {"key1": "value1"}

    # Test serialize with missing attributes
    field1 = Field()
    field2 = Field()
    schema = Schema(fields={"key1": field1, "key2": field2})
    
    class TestObj2:
        def __init__(self):
            self.key1 = "value1"
    
    obj = TestObj2()
    result = schema.serialize(obj)
    assert result == {"key1": "value1"}

    # Test serialize with custom field serialization
    class CustomField(Field):
        def serialize(self, obj):
            return str(obj).upper() if obj else None
    
    field1 = CustomField()
    field2 = Field()
    schema = Schema(fields={"key1": field1, "key2": field2})
    obj = {"key1": "value1", "key2": "value2"}
    result = schema.serialize(obj)
    assert result == {"key1": "VALUE1", "key2": "value2"}

    # Test serialize with empty dict
    schema = Schema(fields={})
    obj = {"key1": "value1"}
    result = schema.serialize(obj)
    assert result == {}

    # Test serialize with dict containing extra keys not in schema
    field1 = Field()
    schema = Schema(fields={"key1": field1})
    obj = {"key1": "value1", "key2": "value2"}
    result = schema.serialize(obj)
    assert result == {"key1": "value1"}


# LLM-generated content at query #16
#--------------------------

```python
def test_Reference_validate():
    """Test Reference.validate() method."""
    import pytest
    from typesystem.base import ValidationError

    # Create a simple schema to reference
    definitions = Definitions()
    test_schema = Schema({"name": Field()})
    definitions["TestSchema"] = test_schema

    # Test 1: Valid value passes through to target schema
    reference = Reference(to="TestSchema", definitions=definitions)
    result = reference.validate({"name": "test"})
    assert result == {"name": "test"}

    # Test 2: None value with allow_null=True returns None
    reference_nullable = Reference(
        to="TestSchema", definitions=definitions, allow_null=True
    )
    result = reference_nullable.validate(None)
    assert result is None

    # Test 3: None value with allow_null=False raises validation error
    reference = Reference(to="TestSchema", definitions=definitions, allow_null=False)
    with pytest.raises(ValidationError) as exc_info:
        reference.validate(None)
    assert exc_info.value.code() == "null"

    # Test 4: Invalid value according to target schema raises validation error
    reference = Reference(to="TestSchema", definitions=definitions)
    with pytest.raises(ValidationError):
        reference.validate("not a dict")

    # Test 5: Empty dict validates successfully
    reference = Reference(to="TestSchema", definitions=definitions)
    result = reference.validate({})
    assert result == {}

    # Test 6: Reference with complex nested schema
    nested_definitions = Definitions()
    nested_schema = Schema({"id": Field(), "value": Field()})
    nested_definitions["ComplexSchema"] = nested_schema
    reference = Reference(to="ComplexSchema", definitions=nested_definitions)
    result = reference.validate({"id": 1, "value": "test"})
    assert result == {"id": 1, "value": "test"}


# LLM-generated content at query #17
#--------------------------

```python
def test_Reference_validate():
    # Test 1: validate with None when allow_null is True
    definitions = Definitions()
    schema = Schema(fields={"name": Field()})
    definitions["TestSchema"] = schema
    ref = Reference(to="TestSchema", definitions=definitions, allow_null=True)
    assert ref.validate(None) is None

    # Test 2: validate with None when allow_null is False (default)
    ref_not_null = Reference(to="TestSchema", definitions=definitions)
    with pytest.raises(ValidationError) as exc_info:
        ref_not_null.validate(None)
    assert exc_info.value.code == "null"

    # Test 3: validate with valid value delegates to target schema
    definitions2 = Definitions()
    schema2 = Schema(fields={"id": Field(), "name": Field()})
    definitions2["Person"] = schema2
    ref2 = Reference(to="Person", definitions=definitions2)
    result = ref2.validate({"id": 1, "name": "John"})
    assert result == {"id": 1, "name": "John"}

    # Test 4: validate with invalid value raises error from target schema
    definitions3 = Definitions()
    schema3 = Schema(fields={"email": Field()})
    definitions3["User"] = schema3
    ref3 = Reference(to="User", definitions=definitions3)
    with pytest.raises(ValidationError):
        ref3.validate(None)

    # Test 5: validate passes through target schema validation errors
    definitions4 = Definitions()
    inner_field = Field()
    schema4 = Schema(fields={"required_field": inner_field})
    definitions4["RequiredSchema"] = schema4
    ref4 = Reference(to="RequiredSchema", definitions=definitions4)
    with pytest.raises(ValidationError):
        ref4.validate({})

    # Test 6: validate with nested reference
    definitions5 = Definitions()
    inner_schema = Schema(fields={"value": Field()})
    definitions5["Inner"] = inner_schema
    outer_ref = Reference(to="Inner", definitions=definitions5)
    result = outer_ref.validate({"value": "test"})
    assert result == {"value": "test"}


# LLM-generated content at query #18
#--------------------------

```python
import pytest
from typesystem.base import ValidationError
from typesystem.fields import Field, String, Integer, Boolean


def test_Schema_validate():
    # Test basic valid schema validation
    fields = {
        "name": String(),
        "age": Integer(),
    }
    schema = Schema(fields=fields)
    result = schema.validate({"name": "John", "age": 30})
    assert result == {"name": "John", "age": 30}

    # Test null value when allow_null is True
    schema_nullable = Schema(fields=fields, allow_null=True)
    result = schema_nullable.validate(None)
    assert result is None

    # Test null value when allow_null is False
    with pytest.raises(ValidationError) as exc_info:
        schema.validate(None)
    assert exc_info.value.code == "null"

    # Test non-dict/mapping type
    with pytest.raises(ValidationError) as exc_info:
        schema.validate("invalid")
    assert exc_info.value.code == "type"

    # Test with list
    with pytest.raises(ValidationError) as exc_info:
        schema.validate([1, 2, 3])
    assert exc_info.value.code == "type"

    # Test invalid key type (non-string)
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({1: "value", "name": "John", "age": 30})
    messages = exc_info.value.messages()
    assert any(msg.code == "invalid_key" for msg in messages)

    # Test required field missing
    fields_with_required = {
        "name": String(),
        "age": Integer(),
    }
    schema_required = Schema(fields=fields_with_required)
    with pytest.raises(ValidationError) as exc_info:
        schema_required.validate({"name": "John"})
    messages = exc_info.value.messages()
    assert any(msg.code == "required" for msg in messages)

    # Test field with default value
    fields_with_default = {
        "name": String(),
        "status": String(default="active"),
    }
    schema_default = Schema(fields=fields_with_default)
    result = schema_default.validate({"name": "John"})
    assert result == {"name": "John", "status": "active"}

    # Test read_only field is ignored
    fields_with_readonly = {
        "name": String(),
        "id": Integer(read_only=True),
    }
    schema_readonly = Schema(fields=fields_with_readonly)
    result = schema_readonly.validate({"name": "John", "id": 123})
    assert result == {"name": "John"}

    # Test child field validation error
    fields_with_validation = {
        "name": String(),
        "age": Integer(),
    }
    schema_validation = Schema(fields=fields_with_validation)
    with pytest.raises(ValidationError) as exc_info:
        schema_validation.validate({"name": "John", "age": "not_an_int"})
    assert exc_info.value.messages()

    # Test with Mapping type
    from collections import UserDict
    mapping_obj = UserDict({"name": "John", "age": 30})
    result = schema.validate(mapping_obj)
    assert result == {"name": "John", "age": 30}

    # Test extra fields not in schema are ignored
    result = schema.validate({"name": "John", "age": 30, "extra": "field"})
    assert result == {"name": "John", "age": 30}

    # Test multiple validation errors
    with pytest.raises(ValidationError) as exc_info:
        schema_required.validate({1: "value", "age": "invalid"})
    messages = exc_info.value.messages()
    assert len(messages) > 1

    # Test empty dict
    fields_all_optional = {
        "name": String(default="Unknown"),
        "age": Integer(default=0),
    }
    schema_all_optional = Schema(fields=fields_all_optional)
    result = schema_all_optional.validate({})
    assert result == {"name": "Unknown", "age": 0}


# LLM-generated content at query #19
#--------------------------

```python
def test_Reference_validate():
    """Unit tests for Reference.validate() method."""
    
    # Setup
    definitions = Definitions()
    schema = Schema(fields={"name": Field()})
    definitions["TestSchema"] = schema
    
    # Test 1: validate with allow_null=True and None value returns None
    ref = Reference(to="TestSchema", definitions=definitions, allow_null=True)
    assert ref.validate(None) is None
    
    # Test 2: validate with allow_null=False and None value raises error
    ref = Reference(to="TestSchema", definitions=definitions, allow_null=False)
    try:
        ref.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code() == "null"
    
    # Test 3: validate delegates to target schema
    ref = Reference(to="TestSchema", definitions=definitions, allow_null=False)
    test_value = {"name": "test"}
    result = ref.validate(test_value)
    assert result == {"name": "test"}
    
    # Test 4: validate with invalid data from target schema raises error
    ref = Reference(to="TestSchema", definitions=definitions, allow_null=False)
    try:
        ref.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError:
        pass
    
    # Test 5: target property returns correct definition
    ref = Reference(to="TestSchema", definitions=definitions)
    assert ref.target is schema
    
    # Test 6: validate with complex nested schema
    nested_fields = {"id": Field(), "data": Field()}
    nested_schema = Schema(fields=nested_fields)
    definitions["NestedSchema"] = nested_schema
    ref = Reference(to="NestedSchema", definitions=definitions)
    test_data = {"id": 1, "data": "value"}
    result = ref.validate(test_data)
    assert result == {"id": 1, "data": "value"}


# LLM-generated content at query #20
#--------------------------

```python
def test_Reference_validate():
    # Test 1: validate with None when allow_null is True
    definitions = Definitions()
    schema = Schema(fields={"name": Field()})
    definitions["TestSchema"] = schema
    ref = Reference(to="TestSchema", definitions=definitions, allow_null=True)
    assert ref.validate(None) is None

    # Test 2: validate with None when allow_null is False (default)
    ref_not_null = Reference(to="TestSchema", definitions=definitions)
    with pytest.raises(ValidationError) as exc_info:
        ref_not_null.validate(None)
    assert exc_info.value.code == "null"

    # Test 3: validate with valid data delegates to target schema
    test_data = {"name": "John"}
    result = ref_not_null.validate(test_data)
    assert result == {"name": "John"}

    # Test 4: validate with invalid data from target schema raises error
    schema_with_required = Schema(fields={"required_field": Field(allow_null=False)})
    definitions["StrictSchema"] = schema_with_required
    ref_strict = Reference(to="StrictSchema", definitions=definitions)
    
    with pytest.raises(ValidationError):
        ref_strict.validate({})

    # Test 5: validate passes through target schema validation
    from typesystem.fields import Integer
    int_schema = Schema(fields={"age": Integer()})
    definitions["IntSchema"] = int_schema
    ref_int = Reference(to="IntSchema", definitions=definitions)
    
    result = ref_int.validate({"age": 25})
    assert result == {"age": 25}

    # Test 6: validate with target schema validation error
    with pytest.raises(ValidationError):
        ref_int.validate({"age": "not_an_int"})

    # Test 7: target property returns correct definition
    assert ref_not_null.target == schema


# LLM-generated content at query #21
#--------------------------

```python
import pytest
from typesystem.base import ValidationError


def test_Reference_validate():
    # Setup
    definitions = Definitions()
    schema = Schema(fields={"name": Field()})
    definitions["TestSchema"] = schema
    
    # Test 1: Valid value passes through to target validation
    reference = Reference(to="TestSchema", definitions=definitions)
    value = {"name": "test"}
    result = reference.validate(value)
    assert result == {"name": "test"}
    
    # Test 2: None value with allow_null=True returns None
    reference_nullable = Reference(to="TestSchema", definitions=definitions, allow_null=True)
    result = reference_nullable.validate(None)
    assert result is None
    
    # Test 3: None value with allow_null=False raises validation error
    reference = Reference(to="TestSchema", definitions=definitions, allow_null=False)
    with pytest.raises(ValidationError) as exc_info:
        reference.validate(None)
    assert exc_info.value.code == "null"
    
    # Test 4: Invalid value according to target schema raises validation error
    reference = Reference(to="TestSchema", definitions=definitions)
    with pytest.raises(ValidationError):
        reference.validate(None)
    
    # Test 5: Complex nested schema validation
    nested_schema = Schema(fields={"id": Field(), "data": Field()})
    definitions["ComplexSchema"] = nested_schema
    reference = Reference(to="ComplexSchema", definitions=definitions)
    value = {"id": 1, "data": "test"}
    result = reference.validate(value)
    assert result == {"id": 1, "data": "test"}
    
    # Test 6: Reference delegates to target's validate method
    definitions2 = Definitions()
    mock_schema = Schema(fields={"field1": Field()})
    definitions2["MockSchema"] = mock_schema
    reference = Reference(to="MockSchema", definitions=definitions2)
    test_value = {"field1": "value1"}
    result = reference.validate(test_value)
    assert isinstance(result, dict)


# LLM-generated content at query #22
#--------------------------

```python
def test_Reference_validate():
    # Test with null value when allow_null is True
    definitions = Definitions()
    schema = Schema(fields={"name": Field()})
    definitions["test_schema"] = schema
    
    reference = Reference(to="test_schema", definitions=definitions, allow_null=True)
    result = reference.validate(None)
    assert result is None
    
    # Test with null value when allow_null is False
    reference_not_null = Reference(to="test_schema", definitions=definitions, allow_null=False)
    with pytest.raises(ValidationError) as exc_info:
        reference_not_null.validate(None)
    assert exc_info.value.code == "null"
    
    # Test with valid value that gets validated by target schema
    reference = Reference(to="test_schema", definitions=definitions, allow_null=False)
    test_value = {"name": "test"}
    result = reference.validate(test_value)
    assert result == {"name": "test"}
    
    # Test with invalid value according to target schema
    invalid_schema = Schema(fields={"name": Field(required=True)})
    definitions["invalid_schema"] = invalid_schema
    reference = Reference(to="invalid_schema", definitions=definitions, allow_null=False)
    
    with pytest.raises(ValidationError):
        reference.validate({})
    
    # Test that validation delegates to target schema properly
    target_called = False
    original_validate = schema.validate
    
    def mock_validate(value):
        nonlocal target_called
        target_called = True
        return original_validate(value)
    
    schema.validate = mock_validate
    reference = Reference(to="test_schema", definitions=definitions)
    reference.validate({"name": "value"})
    assert target_called


# LLM-generated content at query #23
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    schema = Schema(fields={"name": Field()})
    definitions["TestSchema"] = schema
    
    reference = Reference(to="TestSchema", definitions=definitions)
    
    # Test 1: Valid value passes through to target validation
    test_value = {"name": "test"}
    result = reference.validate(test_value)
    assert result == test_value
    
    # Test 2: None value with allow_null=True returns None
    reference_nullable = Reference(to="TestSchema", definitions=definitions, allow_null=True)
    result = reference_nullable.validate(None)
    assert result is None
    
    # Test 3: None value with allow_null=False raises ValidationError
    reference_not_nullable = Reference(to="TestSchema", definitions=definitions, allow_null=False)
    try:
        reference_not_nullable.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "null"
    
    # Test 4: Invalid value raises ValidationError from target validation
    reference_strict = Reference(to="TestSchema", definitions=definitions)
    try:
        reference_strict.validate("not a dict")
        assert False, "Should have raised ValidationError"
    except ValidationError:
        pass
    
    # Test 5: Empty dict validates successfully
    result = reference.validate({})
    assert result == {}


# LLM-generated content at query #24
#--------------------------

```python
def test_Reference_validate():
    # Test with null value when allow_null is True
    definitions = Definitions()
    schema = Schema(fields={"name": Field()})
    definitions["TestSchema"] = schema
    
    ref = Reference(to="TestSchema", definitions=definitions, allow_null=True)
    assert ref.validate(None) is None
    
    # Test with null value when allow_null is False
    ref_no_null = Reference(to="TestSchema", definitions=definitions, allow_null=False)
    with pytest.raises(ValidationError) as exc_info:
        ref_no_null.validate(None)
    assert exc_info.value.code == "null"
    
    # Test with valid value
    test_dict = {"name": "test"}
    result = ref.validate(test_dict)
    assert result == test_dict
    
    # Test with invalid value (non-dict)
    with pytest.raises(ValidationError):
        ref.validate("invalid")
    
    # Test with complex nested schema
    nested_schema = Schema(fields={
        "id": Field(),
        "value": Field()
    })
    definitions["NestedSchema"] = nested_schema
    ref_nested = Reference(to="NestedSchema", definitions=definitions)
    
    valid_nested = {"id": 1, "value": "test"}
    result = ref_nested.validate(valid_nested)
    assert result == valid_nested
    
    # Test that target property resolves correctly
    assert ref.target is schema
    assert ref_nested.target is nested_schema


# LLM-generated content at query #25
#--------------------------

```python
def test_Reference_validate():
    # Test 1: Validate with None when allow_null is True
    definitions = Definitions()
    schema = Schema(fields={"name": Field()})
    definitions["TestSchema"] = schema
    
    ref = Reference(to="TestSchema", definitions=definitions, allow_null=True)
    result = ref.validate(None)
    assert result is None
    
    # Test 2: Validate with None when allow_null is False (default)
    ref_no_null = Reference(to="TestSchema", definitions=definitions)
    try:
        ref_no_null.validate(None)
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert e.code() == "null"
    
    # Test 3: Validate with valid value
    definitions2 = Definitions()
    simple_field = Field()
    definitions2["SimpleField"] = simple_field
    
    ref2 = Reference(to="SimpleField", definitions=definitions2)
    result = ref2.validate("test_value")
    assert result == "test_value"
    
    # Test 4: Validate with dict value against Schema reference
    definitions3 = Definitions()
    name_field = Field()
    schema_with_fields = Schema(fields={"name": name_field})
    definitions3["PersonSchema"] = schema_with_fields
    
    ref3 = Reference(to="PersonSchema", definitions=definitions3)
    result = ref3.validate({"name": "John"})
    assert result == {"name": "John"}
    
    # Test 5: Validate with invalid value propagates target's validation error
    definitions4 = Definitions()
    strict_schema = Schema(fields={"age": Field()})
    definitions4["StrictSchema"] = strict_schema
    
    ref4 = Reference(to="StrictSchema", definitions=definitions4)
    try:
        ref4.validate(None)
        assert False, "Should raise ValidationError"
    except ValidationError:
        pass
    
    # Test 6: Target property returns correct definition
    definitions5 = Definitions()
    target_schema = Schema(fields={"id": Field()})
    definitions5["TargetRef"] = target_schema
    
    ref5 = Reference(to="TargetRef", definitions=definitions5)
    assert ref5.target is target_schema


# LLM-generated content at query #26
#--------------------------

```python
def test_Reference_validate():
    # Test case 1: validate with None when allow_null is True
    definitions = Definitions()
    schema = Schema(fields={"name": Field()})
    definitions["TestSchema"] = schema
    
    ref = Reference(to="TestSchema", definitions=definitions, allow_null=True)
    result = ref.validate(None)
    assert result is None

    # Test case 2: validate with None when allow_null is False (default)
    ref_no_null = Reference(to="TestSchema", definitions=definitions)
    try:
        ref_no_null.validate(None)
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert e.code() == "null"

    # Test case 3: validate with valid value
    ref = Reference(to="TestSchema", definitions=definitions)
    test_data = {"name": "John"}
    result = ref.validate(test_data)
    assert result == {"name": "John"}

    # Test case 4: validate with complex nested schema
    inner_schema = Schema(fields={"age": Field()})
    definitions["InnerSchema"] = inner_schema
    
    ref = Reference(to="InnerSchema", definitions=definitions)
    test_data = {"age": 25}
    result = ref.validate(test_data)
    assert result == {"age": 25}

    # Test case 5: validate delegates to target schema's validate method
    definitions2 = Definitions()
    strict_schema = Schema(fields={"id": Field()})
    definitions2["StrictSchema"] = strict_schema
    
    ref = Reference(to="StrictSchema", definitions=definitions2)
    # Valid data should pass through
    result = ref.validate({"id": 123})
    assert result == {"id": 123}


# LLM-generated content at query #27
#--------------------------

```python
def test_Reference_validate():
    # Test case 1: validate with None when allow_null is True
    definitions = Definitions()
    schema = Schema(fields={"name": Field()})
    definitions["TestSchema"] = schema
    
    ref = Reference(to="TestSchema", definitions=definitions, allow_null=True)
    result = ref.validate(None)
    assert result is None
    
    # Test case 2: validate with None when allow_null is False (default)
    ref_not_null = Reference(to="TestSchema", definitions=definitions)
    with pytest.raises(ValidationError) as exc_info:
        ref_not_null.validate(None)
    assert exc_info.value.code == "null"
    
    # Test case 3: validate with valid value delegates to target schema
    definitions2 = Definitions()
    schema2 = Schema(fields={"name": Field()})
    definitions2["TestSchema2"] = schema2
    
    ref2 = Reference(to="TestSchema2", definitions=definitions2)
    valid_data = {"name": "test"}
    result = ref2.validate(valid_data)
    assert result == {"name": "test"}
    
    # Test case 4: validate with invalid value raises ValidationError from target
    definitions3 = Definitions()
    schema3 = Schema(fields={"name": Field(allow_null=False)})
    definitions3["TestSchema3"] = schema3
    
    ref3 = Reference(to="TestSchema3", definitions=definitions3)
    with pytest.raises(ValidationError):
        ref3.validate({"name": None})
    
    # Test case 5: validate with complex nested schema
    definitions4 = Definitions()
    schema4 = Schema(fields={"id": Field(), "value": Field()})
    definitions4["ComplexSchema"] = schema4
    
    ref4 = Reference(to="ComplexSchema", definitions=definitions4)
    complex_data = {"id": 1, "value": "test"}
    result = ref4.validate(complex_data)
    assert result == {"id": 1, "value": "test"}


# LLM-generated content at query #28
#--------------------------

```python
def test_Reference_validate():
    """Test Reference.validate method"""
    # Setup
    definitions = Definitions()
    schema = Schema(fields={"name": Field()})
    definitions["TestSchema"] = schema
    
    reference = Reference(to="TestSchema", definitions=definitions)
    
    # Test 1: Valid value passes through to target validation
    test_value = {"name": "test"}
    result = reference.validate(test_value)
    assert result == {"name": "test"}
    
    # Test 2: None value when allow_null is False raises validation error
    reference_no_null = Reference(to="TestSchema", definitions=definitions, allow_null=False)
    with pytest.raises(ValidationError) as exc_info:
        reference_no_null.validate(None)
    assert exc_info.value.code == "null"
    
    # Test 3: None value when allow_null is True returns None
    reference_with_null = Reference(to="TestSchema", definitions=definitions, allow_null=True)
    result = reference_with_null.validate(None)
    assert result is None
    
    # Test 4: Invalid value raises validation error from target schema
    reference_strict = Reference(to="TestSchema", definitions=definitions, allow_null=False)
    with pytest.raises(ValidationError):
        reference_strict.validate(None)
    
    # Test 5: Target schema validation is called correctly
    complex_schema = Schema(fields={"id": Field(), "name": Field()})
    definitions["ComplexSchema"] = complex_schema
    reference_complex = Reference(to="ComplexSchema", definitions=definitions)
    
    test_data = {"id": 1, "name": "example"}
    result = reference_complex.validate(test_data)
    assert result == {"id": 1, "name": "example"}


# LLM-generated content at query #29
#--------------------------

```python
def test_Reference_validate():
    # Test 1: validate with None when allow_null is True
    definitions = Definitions()
    definitions["test_schema"] = Schema(fields={"name": Field()})
    ref = Reference(to="test_schema", definitions=definitions, allow_null=True)
    assert ref.validate(None) is None

    # Test 2: validate with None when allow_null is False should raise error
    ref_no_null = Reference(to="test_schema", definitions=definitions, allow_null=False)
    with pytest.raises(ValidationError) as exc_info:
        ref_no_null.validate(None)
    assert exc_info.value.code == "null"

    # Test 3: validate with valid value delegates to target schema
    test_data = {"name": "John"}
    result = ref.validate(test_data)
    assert result == {"name": "John"}

    # Test 4: validate with invalid value from target schema raises error
    schema_with_field = Schema(fields={"age": Field(allow_null=False)})
    definitions["schema_with_field"] = schema_with_field
    ref_with_field = Reference(to="schema_with_field", definitions=definitions)
    
    invalid_data = {"age": None}
    with pytest.raises(ValidationError):
        ref_with_field.validate(invalid_data)

    # Test 5: validate with empty dict
    result = ref.validate({})
    assert result == {}

    # Test 6: validate with complex nested data
    nested_schema = Schema(fields={"id": Field(), "name": Field()})
    definitions["nested"] = nested_schema
    ref_nested = Reference(to="nested", definitions=definitions)
    
    complex_data = {"id": 123, "name": "Test"}
    result = ref_nested.validate(complex_data)
    assert result == {"id": 123, "name": "Test"}


# LLM-generated content at query #30
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    schema = Schema(fields={"name": Field()})
    definitions["TestSchema"] = schema
    
    reference = Reference(to="TestSchema", definitions=definitions)
    
    # Test: validate with None when allow_null is False
    try:
        reference.validate(None)
        assert False, "Should raise ValidationError"
    except ValidationError:
        pass
    
    # Test: validate with None when allow_null is True
    reference_nullable = Reference(to="TestSchema", definitions=definitions, allow_null=True)
    result = reference_nullable.validate(None)
    assert result is None
    
    # Test: validate with valid dict
    valid_value = {"name": "test"}
    result = reference.validate(valid_value)
    assert result == {"name": "test"}
    
    # Test: validate with invalid value (non-dict)
    try:
        reference.validate("invalid")
        assert False, "Should raise ValidationError"
    except ValidationError:
        pass
    
    # Test: validate delegates to target schema
    schema_with_validation = Schema(fields={"age": Field()})
    definitions["SchemaWithAge"] = schema_with_validation
    reference_with_age = Reference(to="SchemaWithAge", definitions=definitions)
    
    result = reference_with_age.validate({"age": 25})
    assert result == {"age": 25}


# LLM-generated content at query #31
#--------------------------

```python
def test_Schema_validate():
    # Test with None value when allow_null is True
    schema = Schema(fields={}, allow_null=True)
    assert schema.validate(None) is None

    # Test with None value when allow_null is False
    schema = Schema(fields={}, allow_null=False)
    with pytest.raises(ValidationError) as exc_info:
        schema.validate(None)
    assert exc_info.value.code == "null"

    # Test with non-dict/non-mapping value
    schema = Schema(fields={})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate("not a dict")
    assert exc_info.value.code == "type"

    # Test with list (not a mapping)
    with pytest.raises(ValidationError) as exc_info:
        schema.validate([1, 2, 3])
    assert exc_info.value.code == "type"

    # Test with non-string keys
    schema = Schema(fields={})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({1: "value", "key": "value"})
    assert any(msg.code == "invalid_key" for msg in exc_info.value.messages())

    # Test with missing required fields
    field1 = Field()
    field2 = Field()
    schema = Schema(fields={"required_field": field1, "optional_field": field2})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({})
    assert any(msg.code == "required" for msg in exc_info.value.messages())

    # Test with valid data
    field1 = Field()
    field2 = Field()
    schema = Schema(fields={"field1": field1, "field2": field2})
    result = schema.validate({"field1": "value1", "field2": "value2"})
    assert result == {"field1": "value1", "field2": "value2"}

    # Test with read_only fields
    field1 = Field(read_only=True)
    field2 = Field()
    schema = Schema(fields={"read_only_field": field1, "field2": field2})
    result = schema.validate({"read_only_field": "ignored", "field2": "value2"})
    assert "read_only_field" not in result
    assert result == {"field2": "value2"}

    # Test with fields having defaults
    field1 = Field(default="default_value")
    field2 = Field()
    schema = Schema(fields={"field_with_default": field1, "field2": field2})
    result = schema.validate({"field2": "value2"})
    assert result == {"field_with_default": "default_value", "field2": "value2"}

    # Test with mapping object (not dict)
    from collections import OrderedDict
    field1 = Field()
    schema = Schema(fields={"field1": field1})
    mapping = OrderedDict([("field1", "value1")])
    result = schema.validate(mapping)
    assert result == {"field1": "value1"}

    # Test with nested validation errors
    child_field = Field()
    child_field.validate = lambda x: (_ for _ in ()).throw(ValidationError(code="child_error"))
    schema = Schema(fields={"child": child_field})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({"child": "invalid"})
    assert len(exc_info.value.messages()) > 0

    # Test with multiple errors
    field1 = Field()
    field2 = Field()
    schema = Schema(fields={"field1": field1, "field2": field2})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({1: "value", "extra": "data"})
    messages = exc_info.value.messages()
    assert any(msg.code == "invalid_key" for msg in messages)
    assert any(msg.code == "required" for msg in messages)

    # Test with empty schema and empty dict
    schema = Schema(fields={})
    result = schema.validate({})
    assert result == {}

    # Test with extra keys not in schema
    field1 = Field()
    schema = Schema(fields={"field1": field1})
    result = schema.validate({"field1": "value1", "extra_key": "extra_value"})
    assert result == {"field1": "value1"}
    assert "extra_key" not in result


# LLM-generated content at query #32
#--------------------------

def test_Schema_validate():
    from typesystem.fields import String, Integer
    
    # Test with valid data
    schema = Schema(fields={
        "name": String(),
        "age": Integer(),
    })
    result = schema.validate({"name": "John", "age": 30})
    assert result == {"name": "John", "age": 30}
    
    # Test with None and allow_null=True
    schema_nullable = Schema(fields={"name": String()}, allow_null=True)
    result = schema_nullable.validate(None)
    assert result is None
    
    # Test with None and allow_null=False
    schema_not_nullable = Schema(fields={"name": String()})
    with pytest.raises(ValidationError) as exc_info:
        schema_not_nullable.validate(None)
    assert exc_info.value.code == "null"
    
    # Test with non-dict/mapping type
    schema = Schema(fields={"name": String()})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate("not a dict")
    assert exc_info.value.code == "type"
    
    # Test with non-string keys
    schema = Schema(fields={"name": String()})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({1: "value"})
    messages = exc_info.value.messages()
    assert any(msg.code == "invalid_key" for msg in messages)
    
    # Test with missing required field
    schema = Schema(fields={"name": String(), "age": Integer()})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({"name": "John"})
    messages = exc_info.value.messages()
    assert any(msg.code == "required" for msg in messages)
    
    # Test with field having default value
    schema = Schema(fields={"name": String(), "age": Integer(default=0)})
    result = schema.validate({"name": "John"})
    assert result == {"name": "John", "age": 0}
    
    # Test with read_only field
    schema = Schema(fields={"name": String(), "id": String(read_only=True)})
    result = schema.validate({"name": "John", "id": "123"})
    assert result == {"name": "John"}
    assert "id" not in result
    
    # Test with nested field validation error
    schema = Schema(fields={"name": String(), "age": Integer()})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({"name": "John", "age": "not_an_int"})
    messages = exc_info.value.messages()
    assert len(messages) > 0
    
    # Test with empty dict
    schema = Schema(fields={})
    result = schema.validate({})
    assert result == {}
    
    # Test with extra fields (should be ignored)
    schema = Schema(fields={"name": String()})
    result = schema.validate({"name": "John", "extra": "field"})
    assert result == {"name": "John"}
    assert "extra" not in result
    
    # Test with Mapping type instead of dict
    from collections import OrderedDict
    schema = Schema(fields={"name": String()})
    result = schema.validate(OrderedDict([("name", "John")]))
    assert result == {"name": "John"}


# LLM-generated content at query #33
#--------------------------

```python
def test_Schema_validate():
    # Test with None value and allow_null=True
    field1 = Field()
    schema = Schema(fields={"name": field1}, allow_null=True)
    assert schema.validate(None) is None

    # Test with None value and allow_null=False
    schema = Schema(fields={"name": field1}, allow_null=False)
    with pytest.raises(ValidationError):
        schema.validate(None)

    # Test with non-dict/non-mapping value
    schema = Schema(fields={"name": field1})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate("invalid")
    assert exc_info.value.code() == "type"

    # Test with valid dict
    field2 = Field(allow_null=True)
    schema = Schema(fields={"name": field2})
    result = schema.validate({"name": "test"})
    assert result == {"name": "test"}

    # Test with missing required field
    field3 = Field(allow_null=False)
    schema = Schema(fields={"name": field3})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({})
    error = exc_info.value
    assert any(msg.code == "required" for msg in error.messages())

    # Test with invalid key type (non-string)
    field4 = Field()
    schema = Schema(fields={"name": field4})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({123: "value"})
    error = exc_info.value
    assert any(msg.code == "invalid_key" for msg in error.messages())

    # Test with field having default value
    field5 = Field(default="default_name")
    schema = Schema(fields={"name": field5})
    result = schema.validate({})
    assert result == {"name": "default_name"}

    # Test with read_only field (should be skipped)
    field6 = Field(read_only=True)
    field7 = Field()
    schema = Schema(fields={"id": field6, "name": field7})
    result = schema.validate({"id": 123, "name": "test"})
    assert "id" not in result
    assert result == {"name": "test"}

    # Test with mapping instead of dict
    field8 = Field()
    schema = Schema(fields={"name": field8})
    mapping = {"name": "test"}
    result = schema.validate(mapping)
    assert result == {"name": "test"}

    # Test with child field validation error
    field9 = Field()
    schema = Schema(fields={"name": field9})
    with pytest.raises(ValidationError):
        schema.validate({"name": "test"})

    # Test with multiple fields
    field10 = Field(allow_null=True)
    field11 = Field(allow_null=True)
    schema = Schema(fields={"name": field10, "age": field11})
    result = schema.validate({"name": "John", "age": 30})
    assert result == {"name": "John", "age": 30}

    # Test with extra fields not in schema (should be ignored)
    field12 = Field(allow_null=True)
    schema = Schema(fields={"name": field12})
    result = schema.validate({"name": "test", "extra": "field"})
    assert result == {"name": "test"}


# LLM-generated content at query #34
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    schema = Schema(fields={"name": Field()})
    definitions["TestSchema"] = schema
    
    reference = Reference(to="TestSchema", definitions=definitions)
    
    # Test: validate with valid data
    test_data = {"name": "test"}
    result = reference.validate(test_data)
    assert result == {"name": "test"}
    
    # Test: validate with None when allow_null is False (default)
    with pytest.raises(ValidationError) as exc_info:
        reference.validate(None)
    assert exc_info.value.code() == "null"
    
    # Test: validate with None when allow_null is True
    reference_nullable = Reference(to="TestSchema", definitions=definitions, allow_null=True)
    result = reference_nullable.validate(None)
    assert result is None
    
    # Test: validate delegates to target schema
    reference_with_schema = Reference(to="TestSchema", definitions=definitions)
    valid_data = {"name": "valid"}
    result = reference_with_schema.validate(valid_data)
    assert result == {"name": "valid"}
    
    # Test: validate with invalid data that target schema rejects
    invalid_schema = Schema(fields={"required_field": Field(allow_null=False)})
    definitions["InvalidSchema"] = invalid_schema
    reference_invalid = Reference(to="InvalidSchema", definitions=definitions)
    
    with pytest.raises(ValidationError):
        reference_invalid.validate({})


# LLM-generated content at query #35
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    schema = Schema(fields={"name": Field()})
    definitions["TestSchema"] = schema
    
    # Test 1: Valid value passes through to target validation
    reference = Reference(to="TestSchema", definitions=definitions)
    result = reference.validate({"name": "test"})
    assert result == {"name": "test"}
    
    # Test 2: None value with allow_null=True returns None
    reference_nullable = Reference(to="TestSchema", definitions=definitions, allow_null=True)
    result = reference_nullable.validate(None)
    assert result is None
    
    # Test 3: None value with allow_null=False raises validation error
    reference = Reference(to="TestSchema", definitions=definitions, allow_null=False)
    with pytest.raises(ValidationError) as exc_info:
        reference.validate(None)
    assert exc_info.value.code == "null"
    
    # Test 4: Invalid value raises validation error from target
    reference = Reference(to="TestSchema", definitions=definitions)
    with pytest.raises(ValidationError):
        reference.validate("not a dict")
    
    # Test 5: Complex nested validation
    nested_schema = Schema(fields={"id": Field(), "data": Field()})
    definitions["NestedSchema"] = nested_schema
    reference = Reference(to="NestedSchema", definitions=definitions)
    result = reference.validate({"id": 1, "data": "value"})
    assert result == {"id": 1, "data": "value"}


# LLM-generated content at query #36
#--------------------------

```python
def test_Reference_validate():
    # Test case 1: validate with None and allow_null=True
    definitions = Definitions()
    schema = Schema(fields={"name": Field()})
    definitions["TestSchema"] = schema
    ref = Reference(to="TestSchema", definitions=definitions, allow_null=True)
    result = ref.validate(None)
    assert result is None

    # Test case 2: validate with None and allow_null=False (default)
    ref_no_null = Reference(to="TestSchema", definitions=definitions)
    with pytest.raises(ValidationError) as exc_info:
        ref_no_null.validate(None)
    assert exc_info.value.code == "null"

    # Test case 3: validate with valid dictionary
    definitions2 = Definitions()
    schema2 = Schema(fields={"id": Field(), "name": Field()})
    definitions2["User"] = schema2
    ref2 = Reference(to="User", definitions=definitions2)
    test_data = {"id": 1, "name": "John"}
    result = ref2.validate(test_data)
    assert result == {"id": 1, "name": "John"}

    # Test case 4: validate with invalid data (should propagate target's validation error)
    definitions3 = Definitions()
    from typesystem.fields import Integer
    schema3 = Schema(fields={"age": Integer(allow_null=False)})
    definitions3["Person"] = schema3
    ref3 = Reference(to="Person", definitions=definitions3)
    with pytest.raises(ValidationError):
        ref3.validate({"age": None})

    # Test case 5: validate with empty dictionary
    definitions4 = Definitions()
    schema4 = Schema(fields={"optional_field": Field()})
    definitions4["EmptySchema"] = schema4
    ref4 = Reference(to="EmptySchema", definitions=definitions4)
    result = ref4.validate({})
    assert result == {}

    # Test case 6: validate delegates to target schema
    definitions5 = Definitions()
    schema5 = Schema(fields={"value": Field()})
    definitions5["ValueSchema"] = schema5
    ref5 = Reference(to="ValueSchema", definitions=definitions5)
    test_value = {"value": "test"}
    result = ref5.validate(test_value)
    assert result == {"value": "test"}


# LLM-generated content at query #37
#--------------------------

```python
def test_Reference_validate():
    # Test 1: validate with None when allow_null is True
    definitions = Definitions()
    schema = Schema(fields={"name": Field()})
    definitions["TestSchema"] = schema
    ref = Reference(to="TestSchema", definitions=definitions, allow_null=True)
    assert ref.validate(None) is None

    # Test 2: validate with None when allow_null is False (default)
    ref_no_null = Reference(to="TestSchema", definitions=definitions)
    with pytest.raises(ValidationError) as exc_info:
        ref_no_null.validate(None)
    assert exc_info.value.code == "null"

    # Test 3: validate with valid data
    definitions2 = Definitions()
    schema2 = Schema(fields={"name": Field()})
    definitions2["TestSchema2"] = schema2
    ref2 = Reference(to="TestSchema2", definitions=definitions2)
    result = ref2.validate({"name": "test"})
    assert result == {"name": "test"}

    # Test 4: validate delegates to target schema
    definitions3 = Definitions()
    from typesystem.fields import String
    schema3 = Schema(fields={"email": String()})
    definitions3["UserSchema"] = schema3
    ref3 = Reference(to="UserSchema", definitions=definitions3)
    result = ref3.validate({"email": "test@example.com"})
    assert result == {"email": "test@example.com"}

    # Test 5: validate with invalid data propagates target schema errors
    definitions4 = Definitions()
    schema4 = Schema(fields={"age": Field()})
    definitions4["PersonSchema"] = schema4
    ref4 = Reference(to="PersonSchema", definitions=definitions4)
    # Valid dict structure but Schema will handle validation
    result = ref4.validate({"age": 25})
    assert result == {"age": 25}

    # Test 6: target property returns correct schema
    definitions5 = Definitions()
    schema5 = Schema(fields={"id": Field()})
    definitions5["ItemSchema"] = schema5
    ref5 = Reference(to="ItemSchema", definitions=definitions5)
    assert ref5.target is schema5


# LLM-generated content at query #38
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    schema = Schema(fields={"name": Field()})
    definitions["TestSchema"] = schema
    
    reference = Reference(to="TestSchema", definitions=definitions)
    
    # Test validate with valid value
    test_value = {"name": "test"}
    result = reference.validate(test_value)
    assert result == {"name": "test"}
    
    # Test validate with None when allow_null is False (default)
    try:
        reference.validate(None)
        assert False, "Should raise ValidationError"
    except ValidationError:
        pass
    
    # Test validate with None when allow_null is True
    reference_nullable = Reference(to="TestSchema", definitions=definitions, allow_null=True)
    result = reference_nullable.validate(None)
    assert result is None
    
    # Test validate delegates to target schema
    reference2 = Reference(to="TestSchema", definitions=definitions)
    result = reference2.validate({"name": "another_test"})
    assert result == {"name": "another_test"}


# LLM-generated content at query #39
#--------------------------

```python
def test_Schema_validate():
    # Test with None and allow_null=True
    schema = Schema(fields={}, allow_null=True)
    assert schema.validate(None) is None

    # Test with None and allow_null=False
    schema = Schema(fields={}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert any(msg.code == "null" for msg in e.messages())

    # Test with non-dict/mapping type
    schema = Schema(fields={})
    try:
        schema.validate("invalid")
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert any(msg.code == "type" for msg in e.messages())

    # Test with non-string keys
    schema = Schema(fields={})
    try:
        schema.validate({1: "value"})
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert any(msg.code == "invalid_key" for msg in e.messages())

    # Test with required field missing
    required_field = Field()
    schema = Schema(fields={"name": required_field})
    try:
        schema.validate({})
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert any(msg.code == "required" for msg in e.messages())

    # Test with valid data
    field = Field()
    schema = Schema(fields={"name": field})
    result = schema.validate({"name": "test"})
    assert result == {"name": "test"}

    # Test with read_only field (should be skipped)
    read_only_field = Field(read_only=True)
    schema = Schema(fields={"id": read_only_field})
    result = schema.validate({"id": "123"})
    assert "id" not in result

    # Test with field having default value
    default_field = Field(default="default_value")
    schema = Schema(fields={"status": default_field})
    result = schema.validate({})
    assert result == {"status": "default_value"}

    # Test with child field validation error
    child_field = Field()
    child_field.validate = lambda x: (_ for _ in ()).throw(ValidationError(messages=[Message(text="Invalid", code="invalid", index=[])]))
    schema = Schema(fields={"nested": child_field})
    try:
        schema.validate({"nested": "value"})
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert len(e.messages()) > 0

    # Test with valid mapping object (not dict)
    from collections import OrderedDict
    schema = Schema(fields={"key": Field()})
    result = schema.validate(OrderedDict([("key", "value")]))
    assert result == {"key": "value"}

    # Test with multiple fields and mixed valid/invalid
    field1 = Field()
    field2 = Field()
    schema = Schema(fields={"field1": field1, "field2": field2})
    result = schema.validate({"field1": "value1", "field2": "value2"})
    assert result == {"field1": "value1", "field2": "value2"}

    # Test with empty schema and empty data
    schema = Schema(fields={})
    result = schema.validate({})
    assert result == {}


# LLM-generated content at query #40
#--------------------------

```python
def test_Reference_validate():
    # Test with None value when allow_null is True
    definitions = Definitions()
    definitions["TestSchema"] = Schema(fields={"name": Field()})
    ref = Reference(to="TestSchema", definitions=definitions, allow_null=True)
    assert ref.validate(None) is None

    # Test with None value when allow_null is False (default)
    ref_not_null = Reference(to="TestSchema", definitions=definitions)
    with pytest.raises(ValidationError) as exc_info:
        ref_not_null.validate(None)
    assert exc_info.value.code() == "null"

    # Test with valid dict value
    test_schema = Schema(fields={"name": Field(), "age": Field()})
    definitions["PersonSchema"] = test_schema
    ref_person = Reference(to="PersonSchema", definitions=definitions)
    result = ref_person.validate({"name": "John", "age": 30})
    assert result == {"name": "John", "age": 30}

    # Test with simple Field target
    definitions["StringField"] = Field()
    ref_string = Reference(to="StringField", definitions=definitions)
    result = ref_string.validate("test_value")
    assert result == "test_value"

    # Test that validation delegates to target schema
    schema_with_validation = Schema(fields={"email": Field()})
    definitions["EmailSchema"] = schema_with_validation
    ref_email = Reference(to="EmailSchema", definitions=definitions)
    result = ref_email.validate({"email": "test@example.com"})
    assert result == {"email": "test@example.com"}

    # Test with invalid input type passed to target schema
    ref_invalid = Reference(to="TestSchema", definitions=definitions)
    with pytest.raises(ValidationError):
        ref_invalid.validate("not_a_dict")


# LLM-generated content at query #41
#--------------------------

def test_Schema_validate():
    from typesystem.fields import String, Integer
    
    # Test with valid data
    schema = Schema(fields={
        "name": String(),
        "age": Integer(),
    })
    result = schema.validate({"name": "John", "age": 30})
    assert result == {"name": "John", "age": 30}
    
    # Test with None and allow_null=True
    schema = Schema(fields={"name": String()}, allow_null=True)
    result = schema.validate(None)
    assert result is None
    
    # Test with None and allow_null=False (default)
    schema = Schema(fields={"name": String()})
    with pytest.raises(ValidationError):
        schema.validate(None)
    
    # Test with non-dict/mapping type
    schema = Schema(fields={"name": String()})
    with pytest.raises(ValidationError):
        schema.validate("not a dict")
    
    # Test with non-string key
    schema = Schema(fields={"name": String()})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({123: "value"})
    assert any(msg.code == "invalid_key" for msg in exc_info.value.messages())
    
    # Test with missing required field
    schema = Schema(fields={"name": String(), "age": Integer()})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({"name": "John"})
    assert any(msg.code == "required" for msg in exc_info.value.messages())
    
    # Test with read_only field (should be ignored)
    schema = Schema(fields={
        "name": String(),
        "id": String(read_only=True),
    })
    result = schema.validate({"name": "John", "id": "123"})
    assert result == {"name": "John"}
    
    # Test with field having default value
    schema = Schema(fields={
        "name": String(),
        "status": String(default="active"),
    })
    result = schema.validate({"name": "John"})
    assert result == {"name": "John", "status": "active"}
    
    # Test with field validation error
    schema = Schema(fields={
        "name": String(),
        "age": Integer(),
    })
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({"name": "John", "age": "not an integer"})
    errors = exc_info.value.messages()
    assert any(msg.index == ["age"] for msg in errors)
    
    # Test with multiple validation errors
    schema = Schema(fields={
        "name": String(),
        "age": Integer(),
    })
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({123: "value", "age": "not an integer"})
    errors = exc_info.value.messages()
    assert len(errors) >= 2
    
    # Test with Mapping type instead of dict
    from collections import OrderedDict
    schema = Schema(fields={"name": String()})
    result = schema.validate(OrderedDict([("name", "John")]))
    assert result == {"name": "John"}
    
    # Test with extra fields (should be ignored)
    schema = Schema(fields={"name": String()})
    result = schema.validate({"name": "John", "extra": "field"})
    assert result == {"name": "John"}


# LLM-generated content at query #42
#--------------------------

```python
def test_Reference_validate():
    # Test 1: Validate with None when allow_null is True
    definitions = Definitions()
    schema = Schema(fields={"name": Field()})
    definitions["TestSchema"] = schema
    ref = Reference(to="TestSchema", definitions=definitions, allow_null=True)
    
    result = ref.validate(None)
    assert result is None
    
    # Test 2: Validate with None when allow_null is False (default)
    ref_no_null = Reference(to="TestSchema", definitions=definitions)
    try:
        ref_no_null.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.code() == "null"
    
    # Test 3: Validate with valid data
    definitions2 = Definitions()
    name_field = Field()
    schema2 = Schema(fields={"name": name_field})
    definitions2["Person"] = schema2
    ref2 = Reference(to="Person", definitions=definitions2)
    
    data = {"name": "John"}
    result = ref2.validate(data)
    assert result == {"name": "John"}
    
    # Test 4: Validate with invalid data
    definitions3 = Definitions()
    required_field = Field(allow_null=False)
    schema3 = Schema(fields={"email": required_field})
    definitions3["User"] = schema3
    ref3 = Reference(to="User", definitions=definitions3)
    
    invalid_data = {"email": None}
    try:
        ref3.validate(invalid_data)
        assert False, "Expected ValidationError"
    except ValidationError:
        pass
    
    # Test 5: Validate with empty dict
    definitions4 = Definitions()
    optional_field = Field(allow_null=True)
    schema4 = Schema(fields={"optional": optional_field})
    definitions4["Optional"] = schema4
    ref4 = Reference(to="Optional", definitions=definitions4)
    
    result = ref4.validate({})
    assert result == {}
    
    # Test 6: Target resolution
    definitions5 = Definitions()
    target_schema = Schema(fields={"id": Field()})
    definitions5["TargetType"] = target_schema
    ref5 = Reference(to="TargetType", definitions=definitions5)
    
    assert ref5.target is target_schema


# LLM-generated content at query #43
#--------------------------

```python
import pytest
from typesystem.base import ValidationError
from typesystem.fields import Field, String, Integer, Boolean


def test_Schema_validate():
    # Test with None and allow_null=True
    schema = Schema(fields={"name": String()}, allow_null=True)
    assert schema.validate(None) is None

    # Test with None and allow_null=False
    schema = Schema(fields={"name": String()})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate(None)
    assert exc_info.value.code == "null"

    # Test with non-dict/non-mapping type
    schema = Schema(fields={"name": String()})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate("not a dict")
    assert exc_info.value.code == "type"

    # Test with list
    with pytest.raises(ValidationError) as exc_info:
        schema.validate([1, 2, 3])
    assert exc_info.value.code == "type"

    # Test with valid dict
    schema = Schema(fields={"name": String(), "age": Integer()})
    result = schema.validate({"name": "John", "age": 30})
    assert result == {"name": "John", "age": 30}

    # Test with required field missing
    schema = Schema(fields={"name": String(), "age": Integer()})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({"name": "John"})
    error = exc_info.value
    assert any(msg.code == "required" for msg in error.messages())

    # Test with non-string keys
    schema = Schema(fields={"name": String()})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({1: "value"})
    error = exc_info.value
    assert any(msg.code == "invalid_key" for msg in error.messages())

    # Test with field validation error
    schema = Schema(fields={"age": Integer()})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({"age": "not an integer"})
    error = exc_info.value
    assert len(error.messages()) > 0

    # Test with read_only field
    read_only_field = String(read_only=True)
    schema = Schema(fields={"name": String(), "id": read_only_field})
    result = schema.validate({"name": "John", "id": "123"})
    assert "id" not in result
    assert result == {"name": "John"}

    # Test with field having default value
    schema = Schema(fields={"name": String(), "status": String(default="active")})
    result = schema.validate({"name": "John"})
    assert result == {"name": "John", "status": "active"}

    # Test with extra fields not in schema
    schema = Schema(fields={"name": String()})
    result = schema.validate({"name": "John", "extra": "field"})
    assert result == {"name": "John"}
    assert "extra" not in result

    # Test with empty dict and all optional fields
    schema = Schema(fields={"name": String(default="Unknown"), "age": Integer(default=0)})
    result = schema.validate({})
    assert result == {"name": "Unknown", "age": 0}

    # Test with multiple validation errors
    schema = Schema(fields={"name": String(), "age": Integer(), "active": Boolean()})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({"age": "invalid", "active": "invalid"})
    error = exc_info.value
    messages = error.messages()
    assert len(messages) >= 2

    # Test with Mapping type (not dict)
    from collections import OrderedDict
    schema = Schema(fields={"name": String()})
    mapping = OrderedDict([("name", "John")])
    result = schema.validate(mapping)
    assert result == {"name": "John"}


# LLM-generated content at query #44
#--------------------------

```python
def test_Schema_validate():
    # Test with None and allow_null=True
    schema = Schema(fields={}, allow_null=True)
    assert schema.validate(None) is None

    # Test with None and allow_null=False
    schema = Schema(fields={}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert any(msg.code == "null" for msg in e.messages())

    # Test with non-dict/mapping type
    schema = Schema(fields={})
    try:
        schema.validate("invalid")
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert any(msg.code == "type" for msg in e.messages())

    # Test with non-string keys
    schema = Schema(fields={})
    try:
        schema.validate({1: "value"})
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert any(msg.code == "invalid_key" for msg in e.messages())

    # Test with missing required field
    field1 = Field()
    schema = Schema(fields={"field1": field1})
    try:
        schema.validate({})
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert any(msg.code == "required" for msg in e.messages())

    # Test with valid data
    field1 = Field()
    schema = Schema(fields={"field1": field1})
    result = schema.validate({"field1": "value"})
    assert result == {"field1": "value"}

    # Test with read_only field
    field1 = Field(read_only=True)
    schema = Schema(fields={"field1": field1})
    result = schema.validate({})
    assert result == {}

    # Test with field having default value
    field1 = Field(default="default_value")
    schema = Schema(fields={"field1": field1})
    result = schema.validate({})
    assert result == {"field1": "default_value"}

    # Test with child field validation error
    field1 = Field()
    field1.validate = lambda x: (_ for _ in ()).throw(ValidationError(messages=[Message(text="error", code="error")]))
    schema = Schema(fields={"field1": field1})
    try:
        schema.validate({"field1": "value"})
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert len(e.messages()) > 0

    # Test with mapping type (not dict)
    from collections import OrderedDict
    field1 = Field()
    schema = Schema(fields={"field1": field1})
    result = schema.validate(OrderedDict([("field1", "value")]))
    assert result == {"field1": "value"}

    # Test with multiple fields
    field1 = Field()
    field2 = Field()
    schema = Schema(fields={"field1": field1, "field2": field2})
    result = schema.validate({"field1": "value1", "field2": "value2"})
    assert result == {"field1": "value1", "field2": "value2"}

    # Test skipping read_only field during validation
    field1 = Field()
    field2 = Field(read_only=True)
    schema = Schema(fields={"field1": field1, "field2": field2})
    result = schema.validate({"field1": "value1", "field2": "should_be_ignored"})
    assert result == {"field1": "value1"}


# LLM-generated content at query #45
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    schema = Schema(fields={"name": Field()})
    definitions["TestSchema"] = schema
    
    reference = Reference(to="TestSchema", definitions=definitions)
    
    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None
    
    # Test with None and allow_null=False (default)
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Should raise validation_error"
    except ValidationError:
        pass
    
    # Test with valid value
    test_value = {"name": "test"}
    result = reference.validate(test_value)
    assert result == {"name": "test"}
    
    # Test that it delegates to target schema
    reference2 = Reference(to="TestSchema", definitions=definitions)
    result2 = reference2.validate({"name": "another"})
    assert result2 == {"name": "another"}


# LLM-generated content at query #46
#--------------------------

```python
def test_Reference_validate():
    # Test 1: validate with None when allow_null is True
    definitions = Definitions()
    schema = Schema(fields={"name": Field()})
    definitions["TestSchema"] = schema
    ref = Reference(to="TestSchema", definitions=definitions, allow_null=True)
    
    result = ref.validate(None)
    assert result is None
    
    # Test 2: validate with None when allow_null is False (default)
    ref_not_null = Reference(to="TestSchema", definitions=definitions)
    with pytest.raises(ValidationError) as exc_info:
        ref_not_null.validate(None)
    assert exc_info.value.code == "null"
    
    # Test 3: validate with valid dict value
    ref = Reference(to="TestSchema", definitions=definitions)
    test_value = {"name": "test"}
    result = ref.validate(test_value)
    assert result == {"name": "test"}
    
    # Test 4: validate delegates to target schema
    name_field = Field()
    schema_with_fields = Schema(fields={"name": name_field})
    definitions2 = Definitions()
    definitions2["NameSchema"] = schema_with_fields
    ref2 = Reference(to="NameSchema", definitions=definitions2)
    
    result = ref2.validate({"name": "John"})
    assert result == {"name": "John"}
    
    # Test 5: validate with invalid data according to target schema
    from typesystem.fields import String
    strict_schema = Schema(fields={"id": String()})
    definitions3 = Definitions()
    definitions3["StrictSchema"] = strict_schema
    ref3 = Reference(to="StrictSchema", definitions=definitions3)
    
    # Valid string should pass
    result = ref3.validate({"id": "123"})
    assert result == {"id": "123"}
    
    # Test 6: target property returns correct definition
    ref = Reference(to="TestSchema", definitions=definitions)
    assert ref.target == schema


# LLM-generated content at query #47
#--------------------------

def test_Schema_validate():
    # Test with None and allow_null=True
    schema = Schema(fields={}, allow_null=True)
    assert schema.validate(None) is None

    # Test with None and allow_null=False
    schema = Schema(fields={}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert any(msg.code == "null" for msg in e.messages())

    # Test with non-dict/mapping type
    schema = Schema(fields={})
    try:
        schema.validate("not a dict")
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert any(msg.code == "type" for msg in e.messages())

    # Test with non-string keys
    schema = Schema(fields={})
    try:
        schema.validate({1: "value"})
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert any(msg.code == "invalid_key" for msg in e.messages())

    # Test with required field missing
    field = Field()
    schema = Schema(fields={"name": field})
    try:
        schema.validate({})
        assert False, "Should raise ValidationError"
    except ValidationError as e:
        assert any(msg.code == "required" for msg in e.messages())

    # Test with valid data
    field = Field(allow_null=True)
    schema = Schema(fields={"name": field})
    result = schema.validate({"name": "John"})
    assert result == {"name": "John"}

    # Test with read_only field
    field1 = Field(read_only=True)
    field2 = Field()
    schema = Schema(fields={"id": field1, "name": field2})
    result = schema.validate({"id": 1, "name": "John"})
    assert "id" not in result
    assert result == {"name": "John"}

    # Test with field having default value
    field1 = Field(default="default_name")
    schema = Schema(fields={"name": field1})
    result = schema.validate({})
    assert result == {"name": "default_name"}

    # Test with valid mapping object
    from collections import OrderedDict
    mapping = OrderedDict([("key", "value")])
    field = Field()
    schema = Schema(fields={"key": field})
    result = schema.validate(mapping)
    assert result == {"key": "value"}

    # Test with child field validation error
    field = Field()
    schema = Schema(fields={"name": field})
    try:
        # Assuming Field.validate_or_error returns error for certain inputs
        schema.validate({"name": None})
        # Result depends on field's validation logic
    except ValidationError:
        pass

    # Test with empty schema and valid dict
    schema = Schema(fields={})
    result = schema.validate({})
    assert result == {}

    # Test with multiple fields
    field1 = Field()
    field2 = Field()
    schema = Schema(fields={"field1": field1, "field2": field2})
    try:
        schema.validate({"field1": "value1"})
        assert False, "Should raise ValidationError for missing required field2"
    except ValidationError as e:
        assert any(msg.code == "required" for msg in e.messages())


# LLM-generated content at query #48
#--------------------------

```python
def test_Reference_validate():
    # Test 1: validate with None when allow_null is True
    definitions = Definitions()
    definitions["test_schema"] = Schema(fields={"name": Field()})
    ref = Reference(to="test_schema", definitions=definitions, allow_null=True)
    assert ref.validate(None) is None

    # Test 2: validate with None when allow_null is False (default)
    ref_not_null = Reference(to="test_schema", definitions=definitions)
    with pytest.raises(ValidationError) as exc_info:
        ref_not_null.validate(None)
    assert exc_info.value.code() == "null"

    # Test 3: validate with valid value - delegates to target schema
    definitions2 = Definitions()
    name_field = Field()
    schema = Schema(fields={"name": name_field})
    definitions2["user"] = schema
    ref2 = Reference(to="user", definitions=definitions2)
    
    test_data = {"name": "John"}
    result = ref2.validate(test_data)
    assert isinstance(result, dict)

    # Test 4: validate with invalid value - target schema raises error
    definitions3 = Definitions()
    string_field = Field()
    schema3 = Schema(fields={"age": string_field})
    definitions3["person"] = schema3
    ref3 = Reference(to="person", definitions=definitions3)
    
    # Assuming the target schema will validate the input
    result = ref3.validate({"age": 25})
    assert result == {"age": 25}

    # Test 5: validate accesses correct target from definitions
    definitions4 = Definitions()
    schema_a = Schema(fields={"id": Field()})
    schema_b = Schema(fields={"name": Field()})
    definitions4["schema_a"] = schema_a
    definitions4["schema_b"] = schema_b
    
    ref_a = Reference(to="schema_a", definitions=definitions4)
    ref_b = Reference(to="schema_b", definitions=definitions4)
    
    assert ref_a.target is schema_a
    assert ref_b.target is schema_b
    
    result_a = ref_a.validate({"id": 1})
    result_b = ref_b.validate({"name": "test"})
    assert result_a == {"id": 1}
    assert result_b == {"name": "test"}

    # Test 6: validate with allow_null False explicitly
    ref_explicit = Reference(to="test_schema", definitions=definitions, allow_null=False)
    with pytest.raises(ValidationError):
        ref_explicit.validate(None)


# LLM-generated content at query #49
#--------------------------

```python
def test_Reference_validate():
    # Test case 1: validate with None when allow_null is True
    definitions = Definitions()
    schema = Schema(fields={"name": Field()})
    definitions["TestSchema"] = schema
    
    reference = Reference(to="TestSchema", definitions=definitions, allow_null=True)
    result = reference.validate(None)
    assert result is None
    
    # Test case 2: validate with None when allow_null is False should raise error
    reference_not_null = Reference(to="TestSchema", definitions=definitions, allow_null=False)
    try:
        reference_not_null.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.code == "null"
    
    # Test case 3: validate with valid data
    reference = Reference(to="TestSchema", definitions=definitions)
    test_data = {"name": "John"}
    result = reference.validate(test_data)
    assert result == {"name": "John"}
    
    # Test case 4: validate with invalid data against target schema
    definitions2 = Definitions()
    required_field = Field(allow_null=False)
    schema2 = Schema(fields={"required_field": required_field})
    definitions2["StrictSchema"] = schema2
    
    reference2 = Reference(to="StrictSchema", definitions=definitions2)
    invalid_data = {}
    try:
        reference2.validate(invalid_data)
        assert False, "Expected ValidationError"
    except ValidationError:
        pass
    
    # Test case 5: validate delegates to target schema validation
    definitions3 = Definitions()
    nested_schema = Schema(fields={"id": Field(), "value": Field()})
    definitions3["NestedSchema"] = nested_schema
    
    reference3 = Reference(to="NestedSchema", definitions=definitions3)
    valid_data = {"id": 1, "value": "test"}
    result = reference3.validate(valid_data)
    assert result == {"id": 1, "value": "test"}


# LLM-generated content at query #50
#--------------------------

```python
import pytest
from typesystem.base import ValidationError
from typesystem.fields import Field, String, Integer, Boolean


def test_Schema_validate():
    # Test with None and allow_null=True
    schema = Schema(fields={"name": String()}, allow_null=True)
    assert schema.validate(None) is None

    # Test with None and allow_null=False
    schema = Schema(fields={"name": String()})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate(None)
    assert exc_info.value.code() == "null"

    # Test with non-dict/non-mapping type
    schema = Schema(fields={"name": String()})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate("invalid")
    assert exc_info.value.code() == "type"

    # Test with non-string keys
    schema = Schema(fields={"name": String()})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({1: "value"})
    assert any(msg.code == "invalid_key" for msg in exc_info.value.messages())

    # Test with missing required field
    schema = Schema(fields={"name": String()})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({})
    assert any(msg.code == "required" for msg in exc_info.value.messages())

    # Test with valid data
    schema = Schema(fields={"name": String(), "age": Integer()})
    result = schema.validate({"name": "John", "age": 30})
    assert result == {"name": "John", "age": 30}

    # Test with field having default value
    default_field = String(default="default_name")
    schema = Schema(fields={"name": default_field})
    result = schema.validate({})
    assert result == {"name": "default_name"}

    # Test with read_only field
    read_only_field = String(read_only=True)
    schema = Schema(fields={"id": read_only_field, "name": String()})
    result = schema.validate({"id": "123", "name": "John"})
    assert "id" not in result
    assert result == {"name": "John"}

    # Test with extra fields not in schema
    schema = Schema(fields={"name": String()})
    result = schema.validate({"name": "John", "extra": "field"})
    assert result == {"name": "John"}

    # Test with child field validation error
    schema = Schema(fields={"age": Integer()})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({"age": "not_an_int"})
    assert len(exc_info.value.messages()) > 0

    # Test with mapping type instead of dict
    from collections import OrderedDict
    schema = Schema(fields={"name": String()})
    result = schema.validate(OrderedDict([("name", "John")]))
    assert result == {"name": "John"}

    # Test with multiple field validation errors
    schema = Schema(fields={"name": String(), "age": Integer()})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({"name": 123, "age": "invalid"})
    assert len(exc_info.value.messages()) >= 2

    # Test with optional field not provided
    optional_field = String(allow_null=True)
    schema = Schema(fields={"name": String(), "nickname": optional_field})
    result = schema.validate({"name": "John"})
    assert result == {"name": "John"}


