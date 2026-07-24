####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    fields = {
        "name": Field(allow_null=False),
        "age": Field(allow_null=True),
    }
    schema = Schema(fields=fields)
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test null input with allow_null=True
    schema_allow_null = Schema(fields=fields, allow_null=True)
    assert schema_allow_null.validate(None) is None

    # Test null input with allow_null=False
    schema_no_null = Schema(fields=fields, allow_null=False)
    try:
        schema_no_null.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test non-dict input
    try:
        schema.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"

    # Test invalid key type
    try:
        schema.validate({123: "invalid key"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

    # Test missing required field
    try:
        schema.validate({"age": 30})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"

    # Test field validation error
    fields_with_validation = {
        "name": Field(allow_null=False, validators=[lambda x: x != "invalid"]),
    }
    schema_with_validation = Schema(fields=fields_with_validation)
    try:
        schema_with_validation.validate({"name": "invalid"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) > 0

    # Test default value
    fields_with_default = {
        "name": Field(allow_null=False, default="default_name"),
    }
    schema_with_default = Schema(fields=fields_with_default)
    assert schema_with_default.validate({}) == {"name": "default_name"}

    # Test read-only field
    fields_with_readonly = {
        "name": Field(allow_null=False, read_only=True),
    }
    schema_with_readonly = Schema(fields=fields_with_readonly)
    assert schema_with_readonly.validate({"name": "John"}) == {}


# LLM-generated content at query #2
#--------------------------

```python
def test_Schema_serialize():
    # Test serialization of a simple schema
    schema = Schema(fields={"name": Field(), "age": Field()})
    obj = {"name": "John", "age": 30}
    assert schema.serialize(obj) == {"name": "John", "age": 30}

    # Test serialization with None
    assert schema.serialize(None) is None

    # Test serialization with missing fields
    obj = {"name": "John"}
    assert schema.serialize(obj) == {"name": "John"}

    # Test serialization with nested schema
    nested_schema = Schema(fields={"address": Schema(fields={"city": Field(), "zip": Field()})})
    obj = {"address": {"city": "NYC", "zip": "10001"}}
    assert nested_schema.serialize(obj) == {"address": {"city": "NYC", "zip": "10001"}}

    # Test serialization with object attributes
    class Person:
        def __init__(self, name, age):
            self.name = name
            self.age = age

    person = Person("Jane", 25)
    assert schema.serialize(person) == {"name": "Jane", "age": 25}


# LLM-generated content at query #3
#--------------------------

```python
def test_Schema():
    # Test initialization with fields and kwargs
    fields = {"name": Field(), "age": Field()}
    schema = Schema(fields=fields, allow_null=True)
    assert schema.fields == fields
    assert schema.allow_null is True
    assert schema.required == ["name", "age"]

    # Test initialization with read_only field
    fields_with_readonly = {"name": Field(read_only=True), "age": Field()}
    schema_with_readonly = Schema(fields=fields_with_readonly)
    assert schema_with_readonly.required == ["age"]

    # Test initialization with default value
    fields_with_default = {"name": Field(default="default_name"), "age": Field()}
    schema_with_default = Schema(fields=fields_with_default)
    assert schema_with_default.required == ["age"]

    # Test initialization with empty fields
    empty_schema = Schema(fields={})
    assert empty_schema.fields == {}
    assert empty_schema.required == []


# LLM-generated content at query #4
#--------------------------

```python
def test_Schema_serialize():
    # Test with a simple schema
    schema = Schema(fields={"name": Field(), "age": Field()})

    # Test with a dictionary input
    input_dict = {"name": "John", "age": 30}
    assert schema.serialize(input_dict) == {"name": "John", "age": 30}

    # Test with an object input
    class Person:
        def __init__(self, name, age):
            self.name = name
            self.age = age

    person = Person("Jane", 25)
    assert schema.serialize(person) == {"name": "Jane", "age": 25}

    # Test with None input
    assert schema.serialize(None) is None

    # Test with missing fields
    input_dict_missing = {"name": "Bob"}
    assert schema.serialize(input_dict_missing) == {"name": "Bob"}

    # Test with nested schema
    nested_schema = Schema(fields={
        "address": Schema(fields={
            "street": Field(),
            "city": Field()
        })
    })
    input_nested = {"address": {"street": "123 Main St", "city": "New York"}}
    assert nested_schema.serialize(input_nested) == {"address": {"street": "123 Main St", "city": "New York"}}

    # Test with read-only fields
    schema_readonly = Schema(fields={"id": Field(read_only=True), "name": Field()})
    input_readonly = {"id": 1, "name": "Alice"}
    assert schema_readonly.serialize(input_readonly) == {"name": "Alice"}


# LLM-generated content at query #5
#--------------------------

```python
def test_Schema_serialize():
    # Test with a simple schema
    schema = Schema(fields={
        'name': Field(),
        'age': Field()
    })

    # Test with a dictionary input
    input_dict = {'name': 'John', 'age': 30}
    assert schema.serialize(input_dict) == {'name': 'John', 'age': 30}

    # Test with an object input
    class Person:
        def __init__(self, name, age):
            self.name = name
            self.age = age

    person = Person('Jane', 25)
    assert schema.serialize(person) == {'name': 'Jane', 'age': 25}

    # Test with None input
    assert schema.serialize(None) is None

    # Test with missing fields
    input_dict_missing = {'name': 'Bob'}
    assert schema.serialize(input_dict_missing) == {'name': 'Bob'}

    # Test with nested schema
    nested_schema = Schema(fields={
        'address': Schema(fields={
            'street': Field(),
            'city': Field()
        })
    })

    input_nested = {
        'address': {
            'street': '123 Main St',
            'city': 'New York'
        }
    }
    assert nested_schema.serialize(input_nested) == {
        'address': {
            'street': '123 Main St',
            'city': 'New York'
        }
    }

    # Test with object having missing nested fields
    class Address:
        def __init__(self, street, city):
            self.street = street
            self.city = city

    class PersonWithAddress:
        def __init__(self, address):
            self.address = address

    person_with_address = PersonWithAddress(Address('456 Oak Ave', 'Boston'))
    assert nested_schema.serialize(person_with_address) == {
        'address': {
            'street': '456 Oak Ave',
            'city': 'Boston'
        }
    }


# LLM-generated content at query #6
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    field = Field()
    definitions["test_field"] = field
    reference = Reference(to="test_field", definitions=definitions)

    # Test valid input
    assert reference.validate("valid_input") == field.validate("valid_input")

    # Test null input with allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test null input with allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test that validate calls target's validate
    with patch.object(field, 'validate', return_value="mocked_value") as mock_validate:
        assert reference.validate("test_input") == "mocked_value"
        mock_validate.assert_called_once_with("test_input")


# LLM-generated content at query #7
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    schema = Schema(fields={"name": Field(), "age": Field()})
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test null input with allow_null=True
    schema = Schema(fields={"name": Field()}, allow_null=True)
    assert schema.validate(None) is None

    # Test null input with allow_null=False
    schema = Schema(fields={"name": Field()}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test non-dict input
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"

    # Test non-string keys
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate({123: "value"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

    # Test missing required field
    schema = Schema(fields={"name": Field(), "age": Field()})
    try:
        schema.validate({"name": "John"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"

    # Test field with default value
    schema = Schema(fields={"name": Field(), "age": Field(default=25)})
    assert schema.validate({"name": "John"}) == {"name": "John", "age": 25}

    # Test read-only field
    schema = Schema(fields={"name": Field(), "id": Field(read_only=True)})
    assert schema.validate({"name": "John"}) == {"name": "John"}

    # Test nested validation errors
    inner_schema = Schema(fields={"street": Field()})
    schema = Schema(fields={"name": Field(), "address": inner_schema})
    try:
        schema.validate({"name": "John", "address": {"city": "NYC"}})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"
        assert e.messages[0].index == ["address", "street"]


# LLM-generated content at query #8
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    schema = Schema(fields={"name": Field(), "age": Field()})
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test null input with allow_null=True
    schema = Schema(fields={"name": Field()}, allow_null=True)
    assert schema.validate(None) is None

    # Test null input with allow_null=False
    schema = Schema(fields={"name": Field()}, allow_null=False)
    with pytest.raises(ValidationError):
        schema.validate(None)

    # Test non-dict input
    schema = Schema(fields={"name": Field()})
    with pytest.raises(ValidationError):
        schema.validate("not a dict")

    # Test invalid key type
    schema = Schema(fields={"name": Field()})
    with pytest.raises(ValidationError):
        schema.validate({1: "invalid key"})

    # Test required field missing
    schema = Schema(fields={"name": Field(), "age": Field()})
    with pytest.raises(ValidationError):
        schema.validate({"name": "John"})

    # Test field with default value
    schema = Schema(fields={"name": Field(), "age": Field(default=25)})
    assert schema.validate({"name": "John"}) == {"name": "John", "age": 25}

    # Test read-only field
    schema = Schema(fields={"name": Field(), "id": Field(read_only=True)})
    assert schema.validate({"name": "John"}) == {"name": "John"}

    # Test nested validation error
    child_schema = Schema(fields={"street": Field(), "city": Field()})
    schema = Schema(fields={"name": Field(), "address": child_schema})
    with pytest.raises(ValidationError):
        schema.validate({"name": "John", "address": {"street": "123 Main St"}})


# LLM-generated content at query #9
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    fields = {
        "name": Field(allow_null=False),
        "age": Field(allow_null=True),
    }
    schema = Schema(fields=fields)
    result = schema.validate({"name": "John", "age": 30})
    assert result == {"name": "John", "age": 30}

    # Test null input with allow_null=True
    schema = Schema(fields=fields, allow_null=True)
    result = schema.validate(None)
    assert result is None

    # Test null input with allow_null=False
    schema = Schema(fields=fields, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test non-dict input
    try:
        schema.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"

    # Test invalid key type
    try:
        schema.validate({1: "value"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

    # Test missing required field
    try:
        schema.validate({"age": 30})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"

    # Test field validation error
    fields = {
        "name": Field(allow_null=False),
        "age": Field(allow_null=False, validators=[lambda x: x > 0]),
    }
    schema = Schema(fields=fields)
    try:
        schema.validate({"name": "John", "age": -1})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) > 0

    # Test default value
    fields = {
        "name": Field(allow_null=False),
        "age": Field(allow_null=True, default=25),
    }
    schema = Schema(fields=fields)
    result = schema.validate({"name": "John"})
    assert result == {"name": "John", "age": 25}

    # Test read-only field
    fields = {
        "name": Field(allow_null=False),
        "id": Field(allow_null=False, read_only=True),
    }
    schema = Schema(fields=fields)
    result = schema.validate({"name": "John"})
    assert result == {"name": "John"}


# LLM-generated content at query #10
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    field = Field()
    definitions['test_field'] = field
    reference = Reference(to='test_field', definitions=definitions)

    # Test with valid value
    assert reference.validate("valid_value") == "valid_value"

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    with pytest.raises(ValidationError):
        reference.validate(None)


# LLM-generated content at query #11
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    schema = Schema(fields={"name": Field(), "age": Field()})
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test with None and allow_null=True
    schema = Schema(fields={"name": Field()}, allow_null=True)
    assert schema.validate(None) is None

    # Test with None and allow_null=False
    schema = Schema(fields={"name": Field()})
    with pytest.raises(ValidationError):
        schema.validate(None)

    # Test with non-dict input
    schema = Schema(fields={"name": Field()})
    with pytest.raises(ValidationError):
        schema.validate("not a dict")

    # Test with non-string keys
    schema = Schema(fields={"name": Field()})
    with pytest.raises(ValidationError):
        schema.validate({123: "value"})

    # Test required fields
    schema = Schema(fields={"name": Field(), "age": Field()})
    with pytest.raises(ValidationError):
        schema.validate({"name": "John"})

    # Test with default values
    schema = Schema(fields={"name": Field(), "age": Field(default=25)})
    assert schema.validate({"name": "John"}) == {"name": "John", "age": 25}

    # Test with read-only fields
    schema = Schema(fields={"name": Field(), "id": Field(read_only=True)})
    assert schema.validate({"name": "John"}) == {"name": "John"}
    assert schema.validate({"name": "John", "id": 123}) == {"name": "John"}

    # Test nested validation errors
    inner_schema = Schema(fields={"city": Field()})
    schema = Schema(fields={"name": Field(), "address": inner_schema})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({"name": "John", "address": {"street": "Main St"}})
    assert "address.city" in str(exc_info.value)


# LLM-generated content at query #12
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    schema = Schema(fields={"name": Field()})
    definitions["test_schema"] = schema
    reference = Reference(to="test_schema", definitions=definitions)

    # Test valid input
    assert reference.validate({"name": "value"}) == {"name": "value"}

    # Test null input with allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test null input with allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test invalid input (passed to target schema)
    try:
        reference.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"


# LLM-generated content at query #13
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    schema = Schema({"name": Field(), "age": Field()})
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test null input with allow_null=True
    schema = Schema({"name": Field()}, allow_null=True)
    assert schema.validate(None) is None

    # Test null input with allow_null=False
    schema = Schema({"name": Field()}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test non-dict input
    schema = Schema({"name": Field()})
    try:
        schema.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"

    # Test invalid key type
    schema = Schema({"name": Field()})
    try:
        schema.validate({1: "value"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

    # Test missing required field
    schema = Schema({"name": Field(), "age": Field()})
    try:
        schema.validate({"name": "John"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"

    # Test field validation error
    class PositiveField(Field):
        def validate(self, value):
            if value <= 0:
                raise self.validation_error("positive")
            return value

    schema = Schema({"age": PositiveField()})
    try:
        schema.validate({"age": -5})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "positive"

    # Test default value
    class DefaultField(Field):
        def __init__(self, default=10, **kwargs):
            super().__init__(**kwargs)
            self._default = default

        def has_default(self):
            return True

        def get_default_value(self):
            return self._default

    schema = Schema({"age": DefaultField()})
    assert schema.validate({}) == {"age": 10}

    # Test read-only field
    schema = Schema({"id": Field(read_only=True), "name": Field()})
    assert schema.validate({"name": "John"}) == {"name": "John"}

    # Test nested validation errors
    schema = Schema({
        "user": Schema({
            "name": Field(),
            "age": PositiveField()
        })
    })
    try:
        schema.validate({"user": {"name": "John", "age": -5}})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].index == ["user", "age"]
        assert e.messages[0].code == "positive"


# LLM-generated content at query #14
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    schema = Schema(fields={"name": Field(), "age": Field()})
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test null input with allow_null=True
    schema = Schema(fields={"name": Field()}, allow_null=True)
    assert schema.validate(None) is None

    # Test null input with allow_null=False
    schema = Schema(fields={"name": Field()}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test non-dict input
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"

    # Test invalid key type
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate({123: "invalid key"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

    # Test missing required field
    schema = Schema(fields={"name": Field(), "age": Field()})
    try:
        schema.validate({"name": "John"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"

    # Test field with default value
    schema = Schema(fields={"name": Field(), "age": Field(default=25)})
    assert schema.validate({"name": "John"}) == {"name": "John", "age": 25}

    # Test read-only field
    schema = Schema(fields={"name": Field(), "id": Field(read_only=True)})
    assert schema.validate({"name": "John"}) == {"name": "John"}

    # Test nested validation error
    child_schema = Field()
    child_schema.validate = lambda x: (None, ValidationError(messages=[Message(text="Invalid", code="invalid")]))
    schema = Schema(fields={"name": child_schema})
    try:
        schema.validate({"name": "invalid"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid"


# LLM-generated content at query #15
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    schema = Schema(fields={"name": Field(), "age": Field()})
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test null input with allow_null=True
    schema = Schema(fields={"name": Field()}, allow_null=True)
    assert schema.validate(None) is None

    # Test null input with allow_null=False
    schema = Schema(fields={"name": Field()}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test non-dict input
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"

    # Test invalid key type
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate({123: "invalid key"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

    # Test missing required field
    schema = Schema(fields={"name": Field(), "age": Field()})
    try:
        schema.validate({"name": "John"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"

    # Test field with default value
    schema = Schema(fields={"name": Field(), "age": Field(default=25)})
    assert schema.validate({"name": "John"}) == {"name": "John", "age": 25}

    # Test read-only field
    schema = Schema(fields={"name": Field(), "id": Field(read_only=True)})
    assert schema.validate({"name": "John"}) == {"name": "John"}

    # Test nested validation error
    child_schema = Field()
    child_schema.validate = lambda x: (None, ValidationError(messages=[Message(text="Invalid", code="invalid")]))
    schema = Schema(fields={"name": child_schema})
    try:
        schema.validate({"name": "invalid"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid"


# LLM-generated content at query #16
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_schema = Schema(fields={"name": Field()})
    definitions["test_schema"] = target_schema
    reference = Reference(to="test_schema", definitions=definitions)

    # Test valid input
    valid_input = {"name": "test"}
    assert reference.validate(valid_input) == valid_input

    # Test null input with allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test null input with allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test invalid input (passed to target schema)
    try:
        reference.validate("invalid")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"


# LLM-generated content at query #17
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    field = Field()
    definitions['test_field'] = field
    reference = Reference(to='test_field', definitions=definitions)

    # Test with valid value
    assert reference.validate("valid_value") == "valid_value"

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    with pytest.raises(ValidationError):
        reference.validate(None)

    # Test with invalid value (assuming field.validate raises ValidationError)
    with pytest.raises(ValidationError):
        reference.validate("invalid_value")


# LLM-generated content at query #18
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    field = Field()
    definitions['test_field'] = field
    reference = Reference(to='test_field', definitions=definitions)

    # Test valid value
    assert reference.validate("valid_value") == "valid_value"

    # Test allow_null with None
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test null not allowed
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test target validation error
    field_with_error = Field()
    field_with_error.validate = lambda x: 1 / 0  # Force error
    definitions['error_field'] = field_with_error
    error_reference = Reference(to='error_field', definitions=definitions)
    try:
        error_reference.validate("any_value")
        assert False, "Expected ValidationError"
    except ValidationError:
        pass


# LLM-generated content at query #19
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    schema = Schema(fields={"name": Field(), "age": Field()})
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test null input with allow_null=True
    schema = Schema(fields={"name": Field()}, allow_null=True)
    assert schema.validate(None) is None

    # Test null input with allow_null=False
    schema = Schema(fields={"name": Field()}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test non-dict input
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"

    # Test invalid key type
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate({1: "value"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

    # Test required field missing
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate({})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"

    # Test field with default value
    schema = Schema(fields={"name": Field(default="default_name")})
    assert schema.validate({}) == {"name": "default_name"}

    # Test read-only field
    schema = Schema(fields={"name": Field(read_only=True)})
    assert schema.validate({}) == {}

    # Test nested validation error
    child_schema = Field()
    child_schema.validate = lambda x: (None, ValidationError(messages=[Message(text="error", code="error")]))
    schema = Schema(fields={"name": child_schema})
    try:
        schema.validate({"name": "value"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "error"


# LLM-generated content at query #20
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field(allow_null=True)
    definitions['test_ref'] = target_field
    ref = Reference(to='test_ref', definitions=definitions, allow_null=True)

    # Test with valid value
    assert ref.validate("valid_value") == "valid_value"

    # Test with None and allow_null=True
    assert ref.validate(None) is None

    # Test with None and allow_null=False
    ref_no_null = Reference(to='test_ref', definitions=definitions, allow_null=False)
    try:
        ref_no_null.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with target field validation error
    target_field_with_error = Field()
    target_field_with_error.validate = lambda x: 1/0  # Force error
    definitions['error_ref'] = target_field_with_error
    ref_error = Reference(to='error_ref', definitions=definitions)
    try:
        ref_error.validate("any_value")
        assert False, "Expected ValidationError"
    except ValidationError:
        pass


# LLM-generated content at query #21
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    schema = Schema(fields={"name": Field(), "age": Field()})
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test null input with allow_null=True
    schema = Schema(fields={"name": Field()}, allow_null=True)
    assert schema.validate(None) is None

    # Test null input with allow_null=False
    schema = Schema(fields={"name": Field()}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test non-dict input
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"

    # Test non-string keys
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate({1: "value"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

    # Test missing required field
    schema = Schema(fields={"name": Field(), "age": Field()})
    try:
        schema.validate({"name": "John"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"

    # Test field with default value
    schema = Schema(fields={"name": Field(), "age": Field(default=25)})
    assert schema.validate({"name": "John"}) == {"name": "John", "age": 25}

    # Test read-only field
    schema = Schema(fields={"name": Field(), "id": Field(read_only=True)})
    assert schema.validate({"name": "John"}) == {"name": "John"}


# LLM-generated content at query #22
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    fields = {
        "name": Field(allow_null=False),
        "age": Field(allow_null=True),
    }
    schema = Schema(fields=fields)
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test with default value
    fields_with_default = {
        "name": Field(allow_null=False, default="Unknown"),
        "age": Field(allow_null=True),
    }
    schema_with_default = Schema(fields=fields_with_default)
    assert schema_with_default.validate({"age": 30}) == {"name": "Unknown", "age": 30}

    # Test null input with allow_null=True
    schema_allow_null = Schema(fields=fields, allow_null=True)
    assert schema_allow_null.validate(None) is None

    # Test null input with allow_null=False
    schema_no_null = Schema(fields=fields, allow_null=False)
    try:
        schema_no_null.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test non-dict input
    try:
        schema.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"

    # Test invalid key type
    try:
        schema.validate({1: "value"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

    # Test missing required field
    try:
        schema.validate({"age": 30})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"

    # Test child field validation error
    fields_with_child_validation = {
        "name": Field(allow_null=False),
        "age": Field(allow_null=False, validators=[lambda x: x > 0]),
    }
    schema_with_child_validation = Schema(fields=fields_with_child_validation)
    try:
        schema_with_child_validation.validate({"name": "John", "age": -5})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid"

    # Test read-only field
    fields_with_readonly = {
        "name": Field(allow_null=False),
        "id": Field(allow_null=False, read_only=True),
    }
    schema_with_readonly = Schema(fields=fields_with_readonly)
    assert schema_with_readonly.validate({"name": "John"}) == {"name": "John"}


# LLM-generated content at query #23
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    schema = Schema(fields={"name": Field(), "age": Field()})
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test null input with allow_null=True
    schema = Schema(fields={"name": Field()}, allow_null=True)
    assert schema.validate(None) is None

    # Test null input with allow_null=False
    schema = Schema(fields={"name": Field()}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test non-dict input
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"

    # Test invalid key type
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate({123: "invalid key"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

    # Test missing required field
    schema = Schema(fields={"name": Field(), "age": Field()})
    try:
        schema.validate({"name": "John"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"

    # Test field with default value
    schema = Schema(fields={"name": Field(), "age": Field(default=25)})
    assert schema.validate({"name": "John"}) == {"name": "John", "age": 25}

    # Test read-only field
    schema = Schema(fields={"name": Field(), "id": Field(read_only=True)})
    assert schema.validate({"name": "John"}) == {"name": "John"}


# LLM-generated content at query #24
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    field = Field()
    definitions['test_field'] = field
    reference = Reference(to='test_field', definitions=definitions)

    # Test with valid value
    assert reference.validate("valid_value") == "valid_value"

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with invalid value
    field.validate = lambda x: 1/0  # Force an error
    try:
        reference.validate("invalid_value")
        assert False, "Expected ValidationError"
    except ValidationError:
        pass


# LLM-generated content at query #25
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field()
    definitions['test_ref'] = target_field
    ref = Reference(to='test_ref', definitions=definitions)

    # Test allow_null with None
    ref.allow_null = True
    assert ref.validate(None) is None

    # Test null error when allow_null is False
    ref.allow_null = False
    try:
        ref.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test successful validation
    test_value = "test"
    target_field.validate = lambda x: x  # Mock validate to return input
    assert ref.validate(test_value) == test_value

    # Test ValidationError from target
    target_field.validate = lambda x: ValidationError(messages=[Message(text="Error", code="error")])
    try:
        ref.validate(test_value)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "error"


# LLM-generated content at query #26
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    field = Field()
    definitions['test_field'] = field
    reference = Reference(to='test_field', definitions=definitions)

    # Test with valid value
    assert reference.validate("valid_value") == "valid_value"

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with invalid value (assuming field.validate raises ValidationError)
    try:
        reference.validate("invalid_value")
        assert False, "Expected ValidationError"
    except ValidationError:
        pass


# LLM-generated content at query #27
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field()
    definitions["test_ref"] = target_field
    ref = Reference(to="test_ref", definitions=definitions)

    # Test with valid value
    assert ref.validate("valid_value") == "valid_value"

    # Test with None and allow_null=True
    ref.allow_null = True
    assert ref.validate(None) is None

    # Test with None and allow_null=False
    ref.allow_null = False
    with pytest.raises(ValidationError) as exc_info:
        ref.validate(None)
    assert exc_info.value.messages[0].code == "null"

    # Test with target field raising ValidationError
    target_field_with_error = Field()
    target_field_with_error.validate = lambda x: (_ for _ in ()).throw(ValidationError(messages=[Message(text="error", code="error")]))
    definitions["error_ref"] = target_field_with_error
    ref_error = Reference(to="error_ref", definitions=definitions)
    with pytest.raises(ValidationError) as exc_info:
        ref_error.validate("any_value")
    assert exc_info.value.messages[0].code == "error"


# LLM-generated content at query #28
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    schema = Schema(fields={"name": Field(), "age": Field()})
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test null input with allow_null=True
    schema = Schema(fields={"name": Field()}, allow_null=True)
    assert schema.validate(None) is None

    # Test null input with allow_null=False
    schema = Schema(fields={"name": Field()}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test non-dict input
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"

    # Test non-string keys
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate({1: "value"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

    # Test required fields
    schema = Schema(fields={"name": Field(), "age": Field()})
    try:
        schema.validate({"name": "John"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"

    # Test default values
    schema = Schema(fields={"name": Field(), "age": Field(default=25)})
    assert schema.validate({"name": "John"}) == {"name": "John", "age": 25}

    # Test child field validation
    schema = Schema(fields={"name": Field(min_length=3)})
    try:
        schema.validate({"name": "Jo"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "min_length"


# LLM-generated content at query #29
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    schema = Schema(fields={"name": Field(), "age": Field()})
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test null input with allow_null=True
    schema = Schema(fields={"name": Field()}, allow_null=True)
    assert schema.validate(None) is None

    # Test null input with allow_null=False
    schema = Schema(fields={"name": Field()}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test non-dict input
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"

    # Test non-string keys
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate({123: "value"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

    # Test missing required field
    schema = Schema(fields={"name": Field(), "age": Field()})
    try:
        schema.validate({"name": "John"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"

    # Test field with default value
    schema = Schema(fields={"name": Field(), "age": Field(default=25)})
    assert schema.validate({"name": "John"}) == {"name": "John", "age": 25}

    # Test read-only field
    schema = Schema(fields={"name": Field(), "id": Field(read_only=True)})
    assert schema.validate({"name": "John"}) == {"name": "John"}

    # Test nested validation error
    child_schema = Field()
    child_schema.validate = lambda x: (None, ValidationError(messages=[Message(text="error", code="error")]))
    schema = Schema(fields={"name": child_schema})
    try:
        schema.validate({"name": "John"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "error"


# LLM-generated content at query #30
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field()
    definitions['test_field'] = target_field
    reference = Reference(to='test_field', definitions=definitions)

    # Test with valid value
    assert reference.validate("valid_value") == "valid_value"

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with target field raising ValidationError
    target_field_with_error = Field()
    target_field_with_error.validate = lambda x: (_ for _ in ()).throw(ValidationError(messages=[Message(text="error", code="error")]))
    definitions['error_field'] = target_field_with_error
    error_reference = Reference(to='error_field', definitions=definitions)
    try:
        error_reference.validate("any_value")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "error"


# LLM-generated content at query #31
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    field = Field()
    definitions['test_field'] = field
    reference = Reference(to='test_field', definitions=definitions)

    # Test with valid value
    with patch.object(field, 'validate', return_value='validated_value'):
        result = reference.validate('test_value')
        assert result == 'validated_value'
        field.validate.assert_called_once_with('test_value')

    # Test with None and allow_null=True
    reference.allow_null = True
    result = reference.validate(None)
    assert result is None

    # Test with None and allow_null=False
    reference.allow_null = False
    with pytest.raises(ValidationError) as excinfo:
        reference.validate(None)
    assert excinfo.value.messages[0].code == 'null'


# LLM-generated content at query #32
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    schema = Schema(fields={"name": Field(), "age": Field()})
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test null input with allow_null=True
    schema = Schema(fields={"name": Field()}, allow_null=True)
    assert schema.validate(None) is None

    # Test null input with allow_null=False
    schema = Schema(fields={"name": Field()}, allow_null=False)
    with pytest.raises(ValidationError):
        schema.validate(None)

    # Test non-dict input
    schema = Schema(fields={"name": Field()})
    with pytest.raises(ValidationError):
        schema.validate("not a dict")

    # Test invalid key type
    schema = Schema(fields={"name": Field()})
    with pytest.raises(ValidationError):
        schema.validate({1: "value"})

    # Test missing required field
    schema = Schema(fields={"name": Field(), "age": Field()})
    with pytest.raises(ValidationError):
        schema.validate({"name": "John"})

    # Test field with default value
    schema = Schema(fields={"name": Field(), "age": Field(default=25)})
    assert schema.validate({"name": "John"}) == {"name": "John", "age": 25}

    # Test read-only field
    schema = Schema(fields={"name": Field(), "id": Field(read_only=True)})
    assert schema.validate({"name": "John"}) == {"name": "John"}

    # Test child field validation error
    def validate_age(value):
        if value < 0:
            raise ValidationError(messages=[Message(text="Age must be positive", code="invalid")])
        return value

    schema = Schema(fields={"name": Field(), "age": Field(validate=validate_age)})
    with pytest.raises(ValidationError):
        schema.validate({"name": "John", "age": -5})


# LLM-generated content at query #33
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    schema = Schema(fields={"name": Field(), "age": Field()})
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test allow_null with None
    schema = Schema(fields={"name": Field()}, allow_null=True)
    assert schema.validate(None) is None

    # Test allow_null=False with None
    schema = Schema(fields={"name": Field()}, allow_null=False)
    with pytest.raises(ValidationError):
        schema.validate(None)

    # Test non-dict input
    schema = Schema(fields={"name": Field()})
    with pytest.raises(ValidationError):
        schema.validate("not a dict")

    # Test invalid key type
    schema = Schema(fields={"name": Field()})
    with pytest.raises(ValidationError):
        schema.validate({123: "invalid key"})

    # Test required field missing
    schema = Schema(fields={"name": Field(), "age": Field()})
    with pytest.raises(ValidationError):
        schema.validate({"name": "John"})

    # Test field with default value
    schema = Schema(fields={"name": Field(), "age": Field(default=25)})
    assert schema.validate({"name": "John"}) == {"name": "John", "age": 25}

    # Test read_only field
    schema = Schema(fields={"name": Field(), "id": Field(read_only=True)})
    assert schema.validate({"name": "John"}) == {"name": "John"}
    assert schema.validate({"name": "John", "id": 123}) == {"name": "John"}

    # Test nested validation error
    nested_field = Field()
    nested_field.validate = lambda x: (None, ValidationError(messages=[Message(text="Nested error", code="nested")]))
    schema = Schema(fields={"nested": nested_field})
    with pytest.raises(ValidationError):
        schema.validate({"nested": "invalid"})


# LLM-generated content at query #34
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    field = Field()
    definitions['test_field'] = field
    reference = Reference(to='test_field', definitions=definitions)

    # Test with valid value
    assert reference.validate("valid_value") == "valid_value"

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with invalid value (assuming field.validate raises ValidationError)
    try:
        reference.validate("invalid_value")
        assert False, "Expected ValidationError"
    except ValidationError:
        pass


# LLM-generated content at query #35
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    schema = Schema(fields={"name": Field(), "age": Field()})
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test null input with allow_null=True
    schema = Schema(fields={"name": Field()}, allow_null=True)
    assert schema.validate(None) is None

    # Test null input with allow_null=False
    schema = Schema(fields={"name": Field()}, allow_null=False)
    with pytest.raises(ValidationError):
        schema.validate(None)

    # Test non-dict input
    schema = Schema(fields={"name": Field()})
    with pytest.raises(ValidationError):
        schema.validate("not a dict")

    # Test invalid key type
    schema = Schema(fields={"name": Field()})
    with pytest.raises(ValidationError):
        schema.validate({123: "invalid key"})

    # Test missing required field
    schema = Schema(fields={"name": Field(), "age": Field()})
    with pytest.raises(ValidationError):
        schema.validate({"name": "John"})

    # Test field with default value
    schema = Schema(fields={"name": Field(), "age": Field(default=25)})
    assert schema.validate({"name": "John"}) == {"name": "John", "age": 25}

    # Test read-only field
    schema = Schema(fields={"name": Field(), "id": Field(read_only=True)})
    assert schema.validate({"name": "John"}) == {"name": "John"}
    assert schema.validate({"name": "John", "id": 1}) == {"name": "John"}

    # Test nested field validation error
    child_schema = Field()
    child_schema.validate = lambda x: (None, ValidationError(messages=[Message(text="error", code="error")]))
    schema = Schema(fields={"name": child_schema})
    with pytest.raises(ValidationError):
        schema.validate({"name": "invalid"})


# LLM-generated content at query #36
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    field = Field()
    definitions['test_field'] = field
    reference = Reference(to='test_field', definitions=definitions)

    # Test with valid value
    with patch.object(field, 'validate', return_value='validated_value'):
        result = reference.validate('test_value')
        assert result == 'validated_value'
        field.validate.assert_called_once_with('test_value')

    # Test with None value and allow_null=True
    reference.allow_null = True
    result = reference.validate(None)
    assert result is None

    # Test with None value and allow_null=False
    reference.allow_null = False
    with pytest.raises(ValidationError) as exc_info:
        reference.validate(None)
    assert exc_info.value.messages[0].code == 'null'


# LLM-generated content at query #37
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_schema = Schema(fields={"name": Field()})
    definitions["test_schema"] = target_schema
    reference = Reference(to="test_schema", definitions=definitions)

    # Test valid input
    valid_input = {"name": "test"}
    assert reference.validate(valid_input) == valid_input

    # Test null input with allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test null input with allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test invalid input (passed to target schema)
    try:
        reference.validate("invalid")
        assert False, "Expected ValidationError"
    except ValidationError:
        pass


# LLM-generated content at query #38
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    schema = Schema(fields={"name": Field(), "age": Field()})
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test allow_null with None input
    schema = Schema(fields={"name": Field()}, allow_null=True)
    assert schema.validate(None) is None

    # Test allow_null=False with None input
    schema = Schema(fields={"name": Field()}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test non-dict input
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"

    # Test invalid key type
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate({123: "invalid key"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

    # Test required field missing
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate({})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"

    # Test field with default value
    schema = Schema(fields={"name": Field(default="default_name")})
    assert schema.validate({}) == {"name": "default_name"}

    # Test nested field validation
    nested_field = Field()
    nested_field.validate = lambda x: (x.upper(), None) if isinstance(x, str) else (None, ValidationError(messages=[Message(text="Not a string", code="invalid")]))
    schema = Schema(fields={"name": nested_field})
    assert schema.validate({"name": "john"}) == {"name": "JOHN"}
    try:
        schema.validate({"name": 123})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid"


# LLM-generated content at query #39
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    definitions['test_schema'] = Schema(fields={'name': Field()})
    reference = Reference(to='test_schema', definitions=definitions)

    # Test valid input
    assert reference.validate({'name': 'value'}) == {'name': 'value'}

    # Test null input with allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test null input with allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test invalid input (passed to target schema)
    try:
        reference.validate('invalid')
        assert False, "Expected ValidationError"
    except ValidationError:
        pass


# LLM-generated content at query #40
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    definitions["test_schema"] = Schema(fields={"name": Field()})
    reference = Reference(to="test_schema", definitions=definitions)

    # Test valid input
    assert reference.validate({"name": "value"}) == {"name": "value"}

    # Test null input with allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test null input with allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test invalid input (passed to target schema)
    try:
        reference.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"


# LLM-generated content at query #41
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    schema = Schema(fields={"name": Field(), "age": Field()})
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test null input with allow_null=True
    schema = Schema(fields={"name": Field()}, allow_null=True)
    assert schema.validate(None) is None

    # Test null input with allow_null=False
    schema = Schema(fields={"name": Field()}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test non-dict input
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"

    # Test non-string keys
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate({123: "value"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

    # Test missing required field
    schema = Schema(fields={"name": Field(), "age": Field()})
    try:
        schema.validate({"name": "John"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"

    # Test field with default value
    schema = Schema(fields={"name": Field(), "age": Field(default=25)})
    assert schema.validate({"name": "John"}) == {"name": "John", "age": 25}

    # Test read-only field
    schema = Schema(fields={"name": Field(), "id": Field(read_only=True)})
    assert schema.validate({"name": "John"}) == {"name": "John"}

    # Test nested validation error
    child_schema = Schema(fields={"street": Field(), "city": Field()})
    schema = Schema(fields={"name": Field(), "address": child_schema})
    try:
        schema.validate({"name": "John", "address": {"street": "123 Main St"}})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"
        assert e.messages[0].index == ["address", "city"]


# LLM-generated content at query #42
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    fields = {
        "name": Field(allow_null=False),
        "age": Field(allow_null=True),
    }
    schema = Schema(fields=fields)

    # Test valid input with all required fields
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test valid input with missing optional field
    assert schema.validate({"name": "John"}) == {"name": "John"}

    # Test valid input with None value for nullable field
    assert schema.validate({"name": "John", "age": None}) == {"name": "John", "age": None}

    # Test invalid input with missing required field
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({"age": 30})
    assert exc_info.value.messages[0].code == "required"

    # Test invalid input with non-string key
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({123: "John"})
    assert exc_info.value.messages[0].code == "invalid_key"

    # Test invalid input with non-dict type
    with pytest.raises(ValidationError) as exc_info:
        schema.validate("not a dict")
    assert exc_info.value.messages[0].code == "type"

    # Test invalid input with None value for non-nullable field
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({"name": None})
    assert exc_info.value.messages[0].code == "null"

    # Test valid input with default value
    fields_with_default = {
        "name": Field(allow_null=False, default="DefaultName"),
        "age": Field(allow_null=True),
    }
    schema_with_default = Schema(fields=fields_with_default)
    assert schema_with_default.validate({"age": 30}) == {"name": "DefaultName", "age": 30}

    # Test valid input with read-only field
    fields_with_readonly = {
        "name": Field(allow_null=False),
        "age": Field(allow_null=True, read_only=True),
    }
    schema_with_readonly = Schema(fields=fields_with_readonly)
    assert schema_with_readonly.validate({"name": "John"}) == {"name": "John"}


# LLM-generated content at query #43
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    schema = Schema(fields={"name": Field(), "age": Field()})
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test None input with allow_null=True
    schema = Schema(fields={"name": Field()}, allow_null=True)
    assert schema.validate(None) is None

    # Test None input with allow_null=False
    schema = Schema(fields={"name": Field()}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test non-dict input
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"

    # Test invalid key type
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate({123: "invalid key"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

    # Test missing required field
    schema = Schema(fields={"name": Field(), "age": Field()})
    try:
        schema.validate({"name": "John"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"

    # Test field with default value
    schema = Schema(fields={"name": Field(), "age": Field(default=25)})
    assert schema.validate({"name": "John"}) == {"name": "John", "age": 25}

    # Test read-only field
    schema = Schema(fields={"name": Field(), "id": Field(read_only=True)})
    assert schema.validate({"name": "John"}) == {"name": "John"}

    # Test nested validation error
    inner_schema = Schema(fields={"value": Field()})
    schema = Schema(fields={"data": inner_schema})
    try:
        schema.validate({"data": {"invalid": "key"}})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"


# LLM-generated content at query #44
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field()
    definitions['test_field'] = target_field
    reference = Reference(to='test_field', definitions=definitions)

    # Test with valid value
    assert reference.validate("valid_value") == "valid_value"

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with target field raising ValidationError
    target_field_with_error = Field()
    target_field_with_error.validate = lambda x: (_ for _ in ()).throw(ValidationError(messages=[Message(text="test", code="test")]))
    definitions['error_field'] = target_field_with_error
    error_reference = Reference(to='error_field', definitions=definitions)
    try:
        error_reference.validate("any_value")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "test"


# LLM-generated content at query #45
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field()
    definitions["test_field"] = target_field

    reference = Reference(to="test_field", definitions=definitions)

    # Test with valid value
    assert reference.validate("valid_value") == "valid_value"

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with target field raising ValidationError
    target_field.validate = lambda value: (_ for _ in ()).throw(ValidationError(messages=[Message(text="test_error", code="test_code")]))
    try:
        reference.validate("invalid_value")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "test_code"


# LLM-generated content at query #46
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    schema = Schema(fields={"name": Field(), "age": Field()})
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test null input with allow_null=True
    schema = Schema(fields={"name": Field()}, allow_null=True)
    assert schema.validate(None) is None

    # Test null input with allow_null=False
    schema = Schema(fields={"name": Field()}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test non-dict input
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"

    # Test non-string keys
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate({1: "value"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

    # Test missing required field
    schema = Schema(fields={"name": Field(), "age": Field()})
    try:
        schema.validate({"name": "John"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"

    # Test field with default value
    schema = Schema(fields={"name": Field(), "age": Field(default=25)})
    assert schema.validate({"name": "John"}) == {"name": "John", "age": 25}

    # Test read-only field
    schema = Schema(fields={"name": Field(), "id": Field(read_only=True)})
    assert schema.validate({"name": "John"}) == {"name": "John"}


# LLM-generated content at query #47
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    schema = Schema(fields={"name": Field(), "age": Field()})
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test with None and allow_null=True
    schema = Schema(fields={"name": Field()}, allow_null=True)
    assert schema.validate(None) is None

    # Test with None and allow_null=False
    schema = Schema(fields={"name": Field()}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with non-dict input
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"

    # Test with non-string keys
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate({123: "invalid key"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

    # Test with missing required field
    schema = Schema(fields={"name": Field(), "age": Field()})
    try:
        schema.validate({"name": "John"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"

    # Test with field having default value
    schema = Schema(fields={"name": Field(), "age": Field(default=25)})
    assert schema.validate({"name": "John"}) == {"name": "John", "age": 25}

    # Test with read-only field
    schema = Schema(fields={"name": Field(), "id": Field(read_only=True)})
    assert schema.validate({"name": "John"}) == {"name": "John"}

    # Test with nested validation errors
    child_schema = Field()
    child_schema.validate = lambda x: (None, ValidationError(messages=[Message(text="Invalid", code="invalid")]))
    schema = Schema(fields={"name": child_schema})
    try:
        schema.validate({"name": "invalid_value"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid"
        assert e.messages[0].index == ["name"]


# LLM-generated content at query #48
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    fields = {
        "name": Field(allow_null=False),
        "age": Field(allow_null=True),
    }
    schema = Schema(fields=fields)

    # Test with valid input
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test with missing required field
    try:
        schema.validate({"age": 30})
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "required"

    # Test with null input and allow_null=False
    try:
        schema.validate(None)
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "null"

    # Test with non-dict input
    try:
        schema.validate("not a dict")
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "type"

    # Test with non-string keys
    try:
        schema.validate({1: "value"})
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "invalid_key"

    # Test with default values
    fields_with_defaults = {
        "name": Field(allow_null=False, default="default_name"),
        "age": Field(allow_null=True, default=25),
    }
    schema_with_defaults = Schema(fields=fields_with_defaults)
    assert schema_with_defaults.validate({}) == {"name": "default_name", "age": 25}

    # Test with read-only fields
    fields_with_readonly = {
        "name": Field(allow_null=False),
        "id": Field(allow_null=False, read_only=True),
    }
    schema_with_readonly = Schema(fields=fields_with_readonly)
    assert schema_with_readonly.validate({"name": "John"}) == {"name": "John"}

    # Test with nested validation errors
    nested_fields = {
        "user": Schema(fields={"name": Field(allow_null=False)}),
    }
    nested_schema = Schema(fields=nested_fields)
    try:
        nested_schema.validate({"user": {}})
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "required"
        assert e.messages[0].index == ["user", "name"]


# LLM-generated content at query #49
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    schema = Schema(fields={"name": Field(), "age": Field()})
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test None input with allow_null=True
    schema = Schema(fields={"name": Field()}, allow_null=True)
    assert schema.validate(None) is None

    # Test None input with allow_null=False
    schema = Schema(fields={"name": Field()}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test non-dict input
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"

    # Test invalid key type
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate({1: "value"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

    # Test missing required field
    schema = Schema(fields={"name": Field(), "age": Field()})
    try:
        schema.validate({"name": "John"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"

    # Test field with default value
    schema = Schema(fields={"name": Field(), "age": Field(default=25)})
    assert schema.validate({"name": "John"}) == {"name": "John", "age": 25}

    # Test read-only field
    schema = Schema(fields={"name": Field(), "id": Field(read_only=True)})
    assert schema.validate({"name": "John"}) == {"name": "John"}


# LLM-generated content at query #50
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    fields = {
        "name": Field(),
        "age": Field(),
    }
    schema = Schema(fields=fields)
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test null input with allow_null=True
    schema_allow_null = Schema(fields=fields, allow_null=True)
    assert schema_allow_null.validate(None) is None

    # Test null input with allow_null=False
    schema_no_null = Schema(fields=fields, allow_null=False)
    try:
        schema_no_null.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test non-dict input
    try:
        schema.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"

    # Test invalid key type
    try:
        schema.validate({1: "value"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

    # Test missing required field
    required_fields = {
        "name": Field(),
        "age": Field(),
    }
    schema_required = Schema(fields=required_fields)
    try:
        schema_required.validate({"name": "John"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"

    # Test field with default value
    fields_with_default = {
        "name": Field(),
        "age": Field(default=25),
    }
    schema_with_default = Schema(fields=fields_with_default)
    assert schema_with_default.validate({"name": "John"}) == {"name": "John", "age": 25}

    # Test read-only field
    fields_with_readonly = {
        "name": Field(),
        "id": Field(read_only=True),
    }
    schema_with_readonly = Schema(fields=fields_with_readonly)
    assert schema_with_readonly.validate({"name": "John"}) == {"name": "John"}

    # Test nested validation error
    nested_fields = {
        "name": Field(),
        "details": Schema(fields={"age": Field()}),
    }
    schema_nested = Schema(fields=nested_fields)
    try:
        schema_nested.validate({"name": "John", "details": {"age": "invalid"}})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].index == ["details", "age"]


# LLM-generated content at query #51
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    fields = {
        "name": Field(allow_null=False),
        "age": Field(allow_null=True),
    }
    schema = Schema(fields=fields)
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test with default value
    fields_with_default = {
        "name": Field(allow_null=False, default="Unknown"),
        "age": Field(allow_null=True),
    }
    schema_with_default = Schema(fields=fields_with_default)
    assert schema_with_default.validate({"age": 30}) == {"name": "Unknown", "age": 30}

    # Test null input with allow_null=True
    schema_allow_null = Schema(fields=fields, allow_null=True)
    assert schema_allow_null.validate(None) is None

    # Test null input with allow_null=False
    schema_no_null = Schema(fields=fields, allow_null=False)
    with pytest.raises(ValidationError) as excinfo:
        schema_no_null.validate(None)
    assert excinfo.value.messages[0].code == "null"

    # Test non-dict input
    with pytest.raises(ValidationError) as excinfo:
        schema.validate("not a dict")
    assert excinfo.value.messages[0].code == "type"

    # Test invalid key type
    with pytest.raises(ValidationError) as excinfo:
        schema.validate({123: "value"})
    assert excinfo.value.messages[0].code == "invalid_key"

    # Test missing required field
    with pytest.raises(ValidationError) as excinfo:
        schema.validate({"age": 30})
    assert excinfo.value.messages[0].code == "required"

    # Test child field validation error
    fields_with_child_validation = {
        "name": Field(allow_null=False, min_length=3),
    }
    schema_child_validation = Schema(fields=fields_with_child_validation)
    with pytest.raises(ValidationError) as excinfo:
        schema_child_validation.validate({"name": "Jo"})
    assert excinfo.value.messages[0].code == "min_length"


# LLM-generated content at query #52
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    field = Field()
    definitions['test_field'] = field
    reference = Reference(to='test_field', definitions=definitions)

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with valid value
    with patch.object(field, 'validate', return_value='validated_value'):
        assert reference.validate('test_value') == 'validated_value'


# LLM-generated content at query #53
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field()
    definitions["test_field"] = target_field
    reference = Reference(to="test_field", definitions=definitions)

    # Test valid value
    assert reference.validate("valid_value") == "valid_value"

    # Test allow_null with None
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test not allow_null with None
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test target validation error
    target_field_with_error = Field()
    target_field_with_error.validate = lambda x: 1/0  # Force error
    definitions["error_field"] = target_field_with_error
    error_reference = Reference(to="error_field", definitions=definitions)
    try:
        error_reference.validate("any_value")
        assert False, "Expected ValidationError"
    except ValidationError:
        pass


# LLM-generated content at query #54
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    fields = {
        "name": Field(allow_null=False),
        "age": Field(allow_null=True),
    }
    schema = Schema(fields=fields)

    # Test valid input with all required fields
    valid_input = {"name": "John", "age": 30}
    assert schema.validate(valid_input) == valid_input

    # Test valid input with missing optional field
    valid_input_optional_missing = {"name": "John"}
    assert schema.validate(valid_input_optional_missing) == {"name": "John", "age": None}

    # Test invalid input with missing required field
    invalid_input_missing_required = {"age": 30}
    try:
        schema.validate(invalid_input_missing_required)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "required"
        assert e.messages[0].index == ["name"]

    # Test invalid input with non-string key
    invalid_input_non_string_key = {123: "John"}
    try:
        schema.validate(invalid_input_non_string_key)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "invalid_key"
        assert e.messages[0].index == [123]

    # Test invalid input with null value when not allowed
    invalid_input_null = None
    try:
        schema.validate(invalid_input_null)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "null"

    # Test invalid input with non-dict value
    invalid_input_non_dict = "not a dict"
    try:
        schema.validate(invalid_input_non_dict)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "type"

    # Test valid input with null value when allowed
    schema_allow_null = Schema(fields=fields, allow_null=True)
    assert schema_allow_null.validate(None) is None


# LLM-generated content at query #55
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    definitions['test_schema'] = Schema(fields={'name': Field()})

    reference = Reference(to='test_schema', definitions=definitions)

    # Test valid input
    valid_input = {'name': 'test'}
    assert reference.validate(valid_input) == valid_input

    # Test null input with allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test null input with allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test invalid input
    try:
        reference.validate("invalid")
        assert False, "Expected ValidationError"
    except ValidationError:
        pass


# LLM-generated content at query #56
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    fields = {
        "name": Field(),
        "age": Field(),
    }
    schema = Schema(fields=fields)
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test null input with allow_null=True
    schema = Schema(fields=fields, allow_null=True)
    assert schema.validate(None) is None

    # Test null input with allow_null=False
    schema = Schema(fields=fields, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test non-dict input
    try:
        schema.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"

    # Test invalid key type
    try:
        schema.validate({1: "value"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

    # Test missing required field
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate({})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"

    # Test field with default value
    schema = Schema(fields={"name": Field(default="default_name")})
    assert schema.validate({}) == {"name": "default_name"}

    # Test read-only field
    schema = Schema(fields={"name": Field(read_only=True)})
    assert schema.validate({}) == {}

    # Test nested field validation
    nested_field = Field()
    nested_field.validate = lambda x: (x, None) if x == "valid" else (None, ValidationError(messages=[Message(text="invalid", code="invalid")]))
    schema = Schema(fields={"nested": nested_field})
    assert schema.validate({"nested": "valid"}) == {"nested": "valid"}
    try:
        schema.validate({"nested": "invalid"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid"


# LLM-generated content at query #57
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field()
    definitions["test_field"] = target_field
    reference = Reference(to="test_field", definitions=definitions)

    # Test valid value
    assert reference.validate("valid_value") == "valid_value"

    # Test null value with allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test null value with allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test that target field's validate is called
    target_field.validate = lambda x: x.upper() if isinstance(x, str) else x
    assert reference.validate("test") == "TEST"


# LLM-generated content at query #58
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    field = Field()
    definitions['test_field'] = field
    reference = Reference(to='test_field', definitions=definitions)

    # Test with valid value
    with patch.object(field, 'validate', return_value='validated_value') as mock_validate:
        result = reference.validate('test_value')
        mock_validate.assert_called_once_with('test_value')
        assert result == 'validated_value'

    # Test with None and allow_null=True
    reference.allow_null = True
    result = reference.validate(None)
    assert result is None

    # Test with None and allow_null=False
    reference.allow_null = False
    with pytest.raises(ValidationError) as exc_info:
        reference.validate(None)
    assert exc_info.value.messages[0].code == 'null'


# LLM-generated content at query #59
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field()
    definitions["test_field"] = target_field
    reference = Reference(to="test_field", definitions=definitions)

    # Test with valid value
    assert reference.validate("valid_value") == "valid_value"

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with target field validation error
    target_field_with_error = Field()
    target_field_with_error.validate = lambda x: 1 / 0  # Force an error
    definitions["error_field"] = target_field_with_error
    error_reference = Reference(to="error_field", definitions=definitions)
    try:
        error_reference.validate("any_value")
        assert False, "Expected ValidationError"
    except ValidationError:
        pass


# LLM-generated content at query #60
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    field = Field()
    definitions['test_field'] = field
    reference = Reference(to='test_field', definitions=definitions)

    # Test with valid value
    assert reference.validate("valid_value") == "valid_value"

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    with pytest.raises(ValidationError):
        reference.validate(None)


# LLM-generated content at query #61
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field(allow_null=False)
    definitions["test_field"] = target_field
    reference = Reference(to="test_field", definitions=definitions, allow_null=False)

    # Test valid value
    assert reference.validate("valid_value") == "valid_value"

    # Test null value with allow_null=True
    reference_allow_null = Reference(to="test_field", definitions=definitions, allow_null=True)
    assert reference_allow_null.validate(None) is None

    # Test null value with allow_null=False
    with pytest.raises(ValidationError) as exc_info:
        reference.validate(None)
    assert exc_info.value.messages[0].code == "null"

    # Test validation error from target field
    target_field_with_validation = Field(allow_null=False)
    target_field_with_validation.validate = lambda x: 1/0 if x == "invalid" else x
    definitions["test_field_validation"] = target_field_with_validation
    reference_validation = Reference(to="test_field_validation", definitions=definitions, allow_null=False)

    with pytest.raises(ValidationError):
        reference_validation.validate("invalid")


# LLM-generated content at query #62
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field()
    definitions['test_ref'] = target_field
    reference = Reference(to='test_ref', definitions=definitions)

    # Test with valid value
    target_field.validate = lambda value: value
    assert reference.validate("valid_value") == "valid_value"

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with target validation error
    target_field.validate = lambda value: 1 / 0  # Will raise an exception
    try:
        reference.validate("invalid_value")
        assert False, "Expected ValidationError"
    except ValidationError:
        pass


# LLM-generated content at query #63
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field()
    definitions['test_ref'] = target_field
    ref = Reference(to='test_ref', definitions=definitions)

    # Test with valid value
    with patch.object(target_field, 'validate', return_value='validated_value'):
        result = ref.validate('test_value')
        assert result == 'validated_value'
        target_field.validate.assert_called_once_with('test_value')

    # Test with None and allow_null=True
    ref.allow_null = True
    result = ref.validate(None)
    assert result is None

    # Test with None and allow_null=False
    ref.allow_null = False
    with pytest.raises(ValidationError) as exc_info:
        ref.validate(None)
    assert exc_info.value.messages[0].code == 'null'


# LLM-generated content at query #64
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    schema = Schema(fields={"name": Field(), "age": Field()})
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test null input with allow_null=True
    schema = Schema(fields={"name": Field()}, allow_null=True)
    assert schema.validate(None) is None

    # Test null input with allow_null=False
    schema = Schema(fields={"name": Field()}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test non-dict input
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"

    # Test invalid key type
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate({1: "value"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

    # Test missing required field
    schema = Schema(fields={"name": Field(), "age": Field()})
    try:
        schema.validate({"name": "John"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"

    # Test field with default value
    schema = Schema(fields={"name": Field(), "age": Field(default=25)})
    assert schema.validate({"name": "John"}) == {"name": "John", "age": 25}

    # Test read-only field
    schema = Schema(fields={"name": Field(), "id": Field(read_only=True)})
    assert schema.validate({"name": "John"}) == {"name": "John"}

    # Test nested validation error
    inner_schema = Schema(fields={"street": Field(), "city": Field()})
    schema = Schema(fields={"name": Field(), "address": inner_schema})
    try:
        schema.validate({"name": "John", "address": {"street": "123 Main St"}})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"
        assert e.messages[0].index == ["address", "city"]


# LLM-generated content at query #65
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    field = Field()
    definitions['test_field'] = field
    reference = Reference(to='test_field', definitions=definitions)

    # Test with valid value
    with patch.object(field, 'validate', return_value='validated_value'):
        assert reference.validate('test_value') == 'validated_value'

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    with pytest.raises(ValidationError) as exc_info:
        reference.validate(None)
    assert exc_info.value.messages[0].code == 'null'

    # Test with validation error from target field
    with patch.object(field, 'validate', side_effect=ValidationError(messages=[Message(text='error', code='error')])):
        with pytest.raises(ValidationError) as exc_info:
            reference.validate('test_value')
        assert exc_info.value.messages[0].code == 'error'


# LLM-generated content at query #66
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field()
    definitions['test_field'] = target_field
    reference = Reference(to='test_field', definitions=definitions)

    # Test with valid value
    assert reference.validate("valid_value") == "valid_value"

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with target field validation error
    target_field_with_error = Field()
    target_field_with_error.validate = lambda x: 1/0  # Force an error
    definitions['error_field'] = target_field_with_error
    error_reference = Reference(to='error_field', definitions=definitions)
    try:
        error_reference.validate("any_value")
        assert False, "Expected ValidationError"
    except ValidationError:
        pass


# LLM-generated content at query #67
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    schema = Schema(fields={"name": Field(), "age": Field()})
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test null input with allow_null=True
    schema = Schema(fields={"name": Field()}, allow_null=True)
    assert schema.validate(None) is None

    # Test null input with allow_null=False
    schema = Schema(fields={"name": Field()}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test non-dict input
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"

    # Test non-string key
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate({123: "value"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

    # Test missing required field
    schema = Schema(fields={"name": Field(), "age": Field()})
    try:
        schema.validate({"name": "John"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"

    # Test field with default value
    schema = Schema(fields={"name": Field(), "age": Field(default=25)})
    assert schema.validate({"name": "John"}) == {"name": "John", "age": 25}

    # Test read-only field
    schema = Schema(fields={"name": Field(), "id": Field(read_only=True)})
    assert schema.validate({"name": "John"}) == {"name": "John"}

    # Test nested validation error
    child_schema = Field()
    child_schema.validate = lambda x: (None, ValidationError(messages=[Message(text="Invalid", code="invalid")]))
    schema = Schema(fields={"name": child_schema})
    try:
        schema.validate({"name": "invalid"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid"


# LLM-generated content at query #68
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field(allow_null=True)
    definitions['test_field'] = target_field
    reference = Reference(to='test_field', definitions=definitions)

    # Test with valid value
    assert reference.validate("valid_value") == "valid_value"

    # Test with None and allow_null=True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with invalid value
    target_field.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"


# LLM-generated content at query #69
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field(allow_null=True)
    definitions["test_field"] = target_field
    reference = Reference(to="test_field", definitions=definitions, allow_null=True)

    # Test valid value
    assert reference.validate("valid_value") == "valid_value"

    # Test null value with allow_null=True
    assert reference.validate(None) is None

    # Test null value with allow_null=False
    reference_no_null = Reference(to="test_field", definitions=definitions, allow_null=False)
    with pytest.raises(ValidationError) as exc_info:
        reference_no_null.validate(None)
    assert exc_info.value.messages[0].code == "null"

    # Test invalid value (assuming target_field raises ValidationError for invalid values)
    target_field_with_validation = Field(allow_null=True)
    target_field_with_validation.validate = lambda x: 1/0 if x == "invalid" else x
    definitions["test_field_validation"] = target_field_with_validation
    reference_validation = Reference(to="test_field_validation", definitions=definitions, allow_null=True)
    with pytest.raises(ValidationError):
        reference_validation.validate("invalid")


# LLM-generated content at query #70
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    field = Field()
    definitions['test_field'] = field
    reference = Reference(to='test_field', definitions=definitions)

    # Test with valid input
    assert reference.validate('valid_value') == 'valid_value'

    # Test with None when allow_null is True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None when allow_null is False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with invalid input
    field.validate = lambda x: 1/0  # Simulate validation error
    try:
        reference.validate('invalid_value')
        assert False, "Expected ValidationError"
    except ValidationError:
        pass


# LLM-generated content at query #71
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    schema = Schema(fields={"name": Field(), "age": Field()})
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test null input with allow_null=True
    schema = Schema(fields={"name": Field()}, allow_null=True)
    assert schema.validate(None) is None

    # Test null input with allow_null=False
    schema = Schema(fields={"name": Field()}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test non-dict input
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"

    # Test invalid key type
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate({1: "invalid key"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

    # Test missing required field
    schema = Schema(fields={"name": Field(), "age": Field()})
    try:
        schema.validate({"name": "John"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"

    # Test field with default value
    schema = Schema(fields={"name": Field(), "age": Field(default=25)})
    assert schema.validate({"name": "John"}) == {"name": "John", "age": 25}

    # Test read-only field
    schema = Schema(fields={"name": Field(), "id": Field(read_only=True)})
    assert schema.validate({"name": "John"}) == {"name": "John"}


# LLM-generated content at query #72
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    schema = Schema(fields={"name": Field(), "age": Field()})
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test null input with allow_null=True
    schema = Schema(fields={"name": Field()}, allow_null=True)
    assert schema.validate(None) is None

    # Test null input with allow_null=False
    schema = Schema(fields={"name": Field()}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test non-dict input
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"

    # Test invalid key type
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate({1: "value"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

    # Test missing required field
    schema = Schema(fields={"name": Field(), "age": Field()})
    try:
        schema.validate({"name": "John"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"

    # Test field with default value
    schema = Schema(fields={"name": Field(), "age": Field(default=25)})
    assert schema.validate({"name": "John"}) == {"name": "John", "age": 25}

    # Test read-only field
    schema = Schema(fields={"name": Field(), "id": Field(read_only=True)})
    assert schema.validate({"name": "John"}) == {"name": "John"}

    # Test nested validation error
    inner_schema = Schema(fields={"value": Field()})
    schema = Schema(fields={"inner": inner_schema})
    try:
        schema.validate({"inner": {"invalid": "data"}})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].index == ["inner", "required"]


# LLM-generated content at query #73
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    field = Field()
    definitions['test_field'] = field
    reference = Reference(to='test_field', definitions=definitions)

    # Test with valid value
    assert reference.validate("valid_value") == "valid_value"

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"


# LLM-generated content at query #74
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field()
    definitions['test_ref'] = target_field
    ref = Reference(to='test_ref', definitions=definitions)

    # Test with valid value
    assert ref.validate("valid_value") == target_field.validate("valid_value")

    # Test with None and allow_null=True
    ref.allow_null = True
    assert ref.validate(None) is None

    # Test with None and allow_null=False
    ref.allow_null = False
    try:
        ref.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test that validate delegates to target's validate
    with patch.object(target_field, 'validate', return_value="mocked_value") as mock_validate:
        assert ref.validate("test_value") == "mocked_value"
        mock_validate.assert_called_once_with("test_value")


# LLM-generated content at query #75
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field()
    definitions['test_field'] = target_field
    reference = Reference(to='test_field', definitions=definitions)

    # Test with valid value
    assert reference.validate("valid_value") == "valid_value"

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with target field validation error
    target_field_with_error = Field()
    target_field_with_error.validate = lambda x: 1 / 0  # Force error
    definitions['error_field'] = target_field_with_error
    error_reference = Reference(to='error_field', definitions=definitions)
    try:
        error_reference.validate("any_value")
        assert False, "Expected ValidationError"
    except ValidationError:
        pass


# LLM-generated content at query #76
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    schema = Schema(fields={"name": Field(), "age": Field()})
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test null input with allow_null=True
    schema = Schema(fields={"name": Field()}, allow_null=True)
    assert schema.validate(None) is None

    # Test null input with allow_null=False
    schema = Schema(fields={"name": Field()}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test non-dict input
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"

    # Test invalid key type
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate({123: "invalid key"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

    # Test missing required field
    schema = Schema(fields={"name": Field(), "age": Field()})
    try:
        schema.validate({"name": "John"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"

    # Test field with default value
    schema = Schema(fields={"name": Field(), "age": Field(default=25)})
    assert schema.validate({"name": "John"}) == {"name": "John", "age": 25}

    # Test field validation error
    schema = Schema(fields={"name": Field(), "age": Field()})
    try:
        schema.validate({"name": "John", "age": "invalid"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"


# LLM-generated content at query #77
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    fields = {
        "name": Field(allow_null=False),
        "age": Field(allow_null=True),
    }
    schema = Schema(fields=fields)
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test null input with allow_null=True
    schema_with_null = Schema(fields=fields, allow_null=True)
    assert schema_with_null.validate(None) is None

    # Test null input with allow_null=False
    with pytest.raises(ValidationError) as exc_info:
        schema.validate(None)
    assert exc_info.value.messages[0].code == "null"

    # Test non-dict input
    with pytest.raises(ValidationError) as exc_info:
        schema.validate("not a dict")
    assert exc_info.value.messages[0].code == "type"

    # Test invalid key type
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({123: "invalid key"})
    assert exc_info.value.messages[0].code == "invalid_key"

    # Test missing required field
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({"age": 30})
    assert exc_info.value.messages[0].code == "required"

    # Test field validation error
    fields_with_validation = {
        "name": Field(allow_null=False, min_length=3),
    }
    schema_with_validation = Schema(fields=fields_with_validation)
    with pytest.raises(ValidationError) as exc_info:
        schema_with_validation.validate({"name": "Jo"})
    assert exc_info.value.messages[0].code == "min_length"

    # Test default value
    fields_with_default = {
        "name": Field(allow_null=False, default="DefaultName"),
    }
    schema_with_default = Schema(fields=fields_with_default)
    assert schema_with_default.validate({}) == {"name": "DefaultName"}

    # Test read-only field
    fields_with_readonly = {
        "name": Field(allow_null=False, read_only=True),
    }
    schema_with_readonly = Schema(fields=fields_with_readonly)
    assert schema_with_readonly.validate({"other": "value"}) == {}


# LLM-generated content at query #78
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field()
    definitions["test_field"] = target_field
    reference = Reference(to="test_field", definitions=definitions)

    # Test with valid value
    assert reference.validate("valid_value") == "valid_value"

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with target field validation error
    target_field_with_error = Field()
    target_field_with_error.validate = lambda x: 1 / 0  # Force an error
    definitions["error_field"] = target_field_with_error
    error_reference = Reference(to="error_field", definitions=definitions)
    try:
        error_reference.validate("any_value")
        assert False, "Expected ValidationError"
    except ValidationError:
        pass


# LLM-generated content at query #79
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field()
    definitions["test_field"] = target_field

    reference = Reference(to="test_field", definitions=definitions)

    # Test with valid value
    assert reference.validate("valid_value") == target_field.validate("valid_value")

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with target field raising ValidationError
    target_field_with_error = Field()
    target_field_with_error.validate = lambda x: (_ for _ in ()).throw(ValidationError(messages=[Message(text="error", code="error")]))
    definitions["error_field"] = target_field_with_error

    error_reference = Reference(to="error_field", definitions=definitions)
    try:
        error_reference.validate("any_value")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "error"


# LLM-generated content at query #80
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    schema = Schema(fields={"name": Field(), "age": Field()})
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test null input with allow_null=True
    schema = Schema(fields={"name": Field()}, allow_null=True)
    assert schema.validate(None) is None

    # Test null input with allow_null=False
    schema = Schema(fields={"name": Field()}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test non-dict input
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"

    # Test invalid key type
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate({123: "invalid key"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

    # Test missing required field
    schema = Schema(fields={"name": Field(), "age": Field()})
    try:
        schema.validate({"name": "John"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"

    # Test field with default value
    schema = Schema(fields={"name": Field(), "age": Field(default=25)})
    assert schema.validate({"name": "John"}) == {"name": "John", "age": 25}

    # Test read-only field
    schema = Schema(fields={"name": Field(), "id": Field(read_only=True)})
    assert schema.validate({"name": "John"}) == {"name": "John"}

    # Test nested field validation
    nested_schema = Schema(fields={"street": Field(), "city": Field()})
    schema = Schema(fields={"name": Field(), "address": nested_schema})
    try:
        schema.validate({"name": "John", "address": {"street": "Main St"}})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"


# LLM-generated content at query #81
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    field = Field()
    definitions['test_field'] = field
    reference = Reference(to='test_field', definitions=definitions)

    # Test with valid value
    assert reference.validate("valid_value") == "valid_value"

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with invalid value
    field.validate = lambda x: 1/0  # Force an error
    try:
        reference.validate("invalid_value")
        assert False, "Expected ValidationError"
    except ValidationError:
        pass


# LLM-generated content at query #82
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    schema = Schema(fields={"name": Field(), "age": Field()})
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test null input with allow_null=True
    schema = Schema(fields={"name": Field()}, allow_null=True)
    assert schema.validate(None) is None

    # Test null input with allow_null=False
    schema = Schema(fields={"name": Field()}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test non-dict input
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"

    # Test invalid key type
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate({1: "value"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

    # Test missing required field
    schema = Schema(fields={"name": Field(), "age": Field()})
    try:
        schema.validate({"name": "John"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"

    # Test field with default value
    schema = Schema(fields={"name": Field(), "age": Field(default=25)})
    assert schema.validate({"name": "John"}) == {"name": "John", "age": 25}

    # Test read-only field
    schema = Schema(fields={"name": Field(), "id": Field(read_only=True)})
    assert schema.validate({"name": "John"}) == {"name": "John"}

    # Test nested field validation error
    child_schema = Field()
    child_schema.validate = lambda x: (None, ValidationError(messages=[Message(text="Invalid", code="invalid")]))
    schema = Schema(fields={"name": child_schema})
    try:
        schema.validate({"name": "invalid_value"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid"


# LLM-generated content at query #83
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    definitions["test_schema"] = Schema(fields={"name": Field()})

    reference = Reference(to="test_schema", definitions=definitions)

    # Test valid input
    valid_input = {"name": "test"}
    assert reference.validate(valid_input) == {"name": "test"}

    # Test null input with allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test null input with allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test invalid input
    try:
        reference.validate("invalid")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) > 0


# LLM-generated content at query #84
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field()
    definitions['test_field'] = target_field
    reference = Reference(to='test_field', definitions=definitions)

    # Test with valid value
    assert reference.validate("valid_value") is None

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with target field validation
    target_field.validate = lambda x: x if x == "expected" else None
    assert reference.validate("expected") == "expected"
    try:
        reference.validate("unexpected")
        assert False, "Expected ValidationError"
    except ValidationError:
        pass


# LLM-generated content at query #85
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    schema = Schema(fields={"name": Field(), "age": Field()})
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test null input with allow_null=True
    schema = Schema(fields={"name": Field()}, allow_null=True)
    assert schema.validate(None) is None

    # Test null input with allow_null=False
    schema = Schema(fields={"name": Field()}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test non-dict input
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"

    # Test invalid key type
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate({123: "invalid key"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

    # Test missing required field
    schema = Schema(fields={"name": Field(), "age": Field()})
    try:
        schema.validate({"name": "John"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"

    # Test field with default value
    schema = Schema(fields={"name": Field(), "age": Field(default=25)})
    assert schema.validate({"name": "John"}) == {"name": "John", "age": 25}

    # Test read-only field
    schema = Schema(fields={"name": Field(), "id": Field(read_only=True)})
    assert schema.validate({"name": "John"}) == {"name": "John"}

    # Test nested validation error
    inner_field = Field()
    inner_field.validate = lambda x: (None, ValidationError(messages=[Message(text="error", code="test")]))
    schema = Schema(fields={"name": inner_field})
    try:
        schema.validate({"name": "invalid"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "test"


# LLM-generated content at query #86
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_schema = Schema(fields={"name": Field()})
    definitions["test_schema"] = target_schema
    reference = Reference(to="test_schema", definitions=definitions)

    # Test valid input
    valid_input = {"name": "test"}
    assert reference.validate(valid_input) == {"name": "test"}

    # Test null input with allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test null input with allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test invalid input (passed to target schema)
    try:
        reference.validate("invalid")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"


# LLM-generated content at query #87
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    schema = Schema(fields={"name": Field(), "age": Field()})
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test null input with allow_null=True
    schema = Schema(fields={"name": Field()}, allow_null=True)
    assert schema.validate(None) is None

    # Test null input with allow_null=False
    schema = Schema(fields={"name": Field()}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test non-dict input
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"

    # Test invalid key type
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate({123: "invalid key"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

    # Test missing required field
    schema = Schema(fields={"name": Field(), "age": Field()})
    try:
        schema.validate({"name": "John"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"

    # Test field with default value
    schema = Schema(fields={"name": Field(), "age": Field(default=25)})
    assert schema.validate({"name": "John"}) == {"name": "John", "age": 25}

    # Test read-only field
    schema = Schema(fields={"name": Field(), "id": Field(read_only=True)})
    assert schema.validate({"name": "John"}) == {"name": "John"}


# LLM-generated content at query #88
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field()
    definitions['test_ref'] = target_field
    reference = Reference(to='test_ref', definitions=definitions)

    # Test with valid value
    target_field.validate = lambda x: x  # Mock validate to return input
    assert reference.validate("valid_value") == "valid_value"

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with target validation error
    target_field.validate = lambda x: 1 / 0  # Mock validate to raise error
    try:
        reference.validate("invalid_value")
        assert False, "Expected ValidationError"
    except ValidationError:
        pass


# LLM-generated content at query #89
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    fields = {
        "name": Field(allow_null=False),
        "age": Field(allow_null=True),
    }
    schema = Schema(fields=fields)
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test with default value
    fields_with_default = {
        "name": Field(allow_null=False, default="DefaultName"),
        "age": Field(allow_null=True),
    }
    schema_with_default = Schema(fields=fields_with_default)
    assert schema_with_default.validate({"age": 30}) == {"name": "DefaultName", "age": 30}

    # Test with read-only field
    fields_with_readonly = {
        "name": Field(allow_null=False, read_only=True),
        "age": Field(allow_null=True),
    }
    schema_with_readonly = Schema(fields=fields_with_readonly)
    assert schema_with_readonly.validate({"age": 30}) == {"age": 30}

    # Test with null input and allow_null=True
    schema_allow_null = Schema(fields=fields, allow_null=True)
    assert schema_allow_null.validate(None) is None

    # Test with null input and allow_null=False
    schema_no_null = Schema(fields=fields, allow_null=False)
    try:
        schema_no_null.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with non-dict input
    try:
        schema.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"

    # Test with non-string key
    try:
        schema.validate({1: "value"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

    # Test with missing required field
    try:
        schema.validate({"age": 30})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"


# LLM-generated content at query #90
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field()
    definitions["test_field"] = target_field
    reference = Reference(to="test_field", definitions=definitions)

    # Test with valid value
    assert reference.validate("valid_value") is None

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError for None when allow_null=False"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with target validation error
    target_field_with_error = Field()
    target_field_with_error.validate = lambda x: (_ for _ in ()).throw(ValidationError(messages=[Message(text="error", code="error")]))
    definitions["error_field"] = target_field_with_error
    error_reference = Reference(to="error_field", definitions=definitions)
    try:
        error_reference.validate("any_value")
        assert False, "Expected ValidationError from target field"
    except ValidationError as e:
        assert e.messages[0].code == "error"


# LLM-generated content at query #91
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field()
    definitions['test_ref'] = target_field
    ref = Reference(to='test_ref', definitions=definitions)

    # Test with valid value
    assert ref.validate("valid_value") == "valid_value"

    # Test with None and allow_null=True
    ref.allow_null = True
    assert ref.validate(None) is None

    # Test with None and allow_null=False
    ref.allow_null = False
    try:
        ref.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].text == "May not be null."
        assert e.messages[0].code == "null"

    # Test that it delegates validation to target
    target_field.validate = lambda x: x.upper() if isinstance(x, str) else None
    assert ref.validate("test") == "TEST"
    assert ref.validate(123) is None


# LLM-generated content at query #92
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field()
    definitions['test_field'] = target_field
    reference = Reference(to='test_field', definitions=definitions)

    # Test with valid value
    assert reference.validate("valid_value") == "valid_value"

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with target field raising ValidationError
    target_field.validate = lambda value: (_ for _ in ()).throw(ValidationError(messages=[Message(text="test error", code="test")]))
    try:
        reference.validate("invalid_value")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "test"


# LLM-generated content at query #93
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    definitions["test_schema"] = Schema(fields={"name": Field()})
    reference = Reference(to="test_schema", definitions=definitions)

    # Test valid input
    valid_input = {"name": "value"}
    assert reference.validate(valid_input) == valid_input

    # Test null input with allow_null=True
    reference_with_null = Reference(to="test_schema", definitions=definitions, allow_null=True)
    assert reference_with_null.validate(None) is None

    # Test null input with allow_null=False
    with pytest.raises(ValidationError) as exc_info:
        reference.validate(None)
    assert exc_info.value.messages[0].code == "null"

    # Test invalid input (non-dict)
    with pytest.raises(ValidationError) as exc_info:
        reference.validate("invalid")
    assert exc_info.value.messages[0].code == "type"


# LLM-generated content at query #94
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    schema = Schema(fields={"name": Field(), "age": Field()})
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test None input with allow_null=True
    schema = Schema(fields={"name": Field()}, allow_null=True)
    assert schema.validate(None) is None

    # Test None input with allow_null=False
    schema = Schema(fields={"name": Field()}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test non-dict input
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"

    # Test invalid key type
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate({123: "value"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

    # Test missing required field
    schema = Schema(fields={"name": Field(), "age": Field()})
    try:
        schema.validate({"name": "John"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"

    # Test field with default value
    schema = Schema(fields={"name": Field(), "age": Field(default=25)})
    assert schema.validate({"name": "John"}) == {"name": "John", "age": 25}

    # Test read-only field
    schema = Schema(fields={"name": Field(), "id": Field(read_only=True)})
    assert schema.validate({"name": "John"}) == {"name": "John"}

    # Test nested validation errors
    child_schema = Field()
    child_schema.validate = lambda x: (None, ValidationError(messages=[Message(text="Invalid", code="invalid")]))
    schema = Schema(fields={"child": child_schema})
    try:
        schema.validate({"child": "value"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid"
        assert e.messages[0].index == ["child"]


# LLM-generated content at query #95
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field()
    definitions['test_field'] = target_field
    reference = Reference(to='test_field', definitions=definitions)

    # Test valid value
    assert reference.validate("valid_value") == "valid_value"

    # Test null value with allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test null value with allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test that validate calls target's validate
    target_field.validate = lambda x: x.upper()
    assert reference.validate("test") == "TEST"


# LLM-generated content at query #96
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    field = Field()
    definitions['test_field'] = field
    reference = Reference(to='test_field', definitions=definitions)

    # Test with valid value
    with patch.object(field, 'validate', return_value='validated_value'):
        assert reference.validate('test_value') == 'validated_value'
        field.validate.assert_called_once_with('test_value')

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    with pytest.raises(ValidationError) as exc_info:
        reference.validate(None)
    assert exc_info.value.messages[0].code == 'null'

    # Test with validation error
    with patch.object(field, 'validate', side_effect=ValidationError(messages=[Message(text='error', code='error')])):
        with pytest.raises(ValidationError) as exc_info:
            reference.validate('test_value')
        assert exc_info.value.messages[0].code == 'error'


# LLM-generated content at query #97
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    fields = {
        "name": Field(required=True),
        "age": Field(required=False),
    }
    schema = Schema(fields=fields)
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test missing required field
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({"age": 30})
    assert exc_info.value.messages[0].code == "required"
    assert exc_info.value.messages[0].index == ["name"]

    # Test invalid key type
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({1: "John", "age": 30})
    assert exc_info.value.messages[0].code == "invalid_key"
    assert exc_info.value.messages[0].index == [1]

    # Test null value with allow_null=False
    with pytest.raises(ValidationError) as exc_info:
        schema.validate(None)
    assert exc_info.value.messages[0].code == "null"

    # Test null value with allow_null=True
    schema_allow_null = Schema(fields=fields, allow_null=True)
    assert schema_allow_null.validate(None) is None

    # Test non-dict input
    with pytest.raises(ValidationError) as exc_info:
        schema.validate("not a dict")
    assert exc_info.value.messages[0].code == "type"

    # Test default value
    fields_with_default = {
        "name": Field(required=True),
        "age": Field(required=False, default=25),
    }
    schema_with_default = Schema(fields=fields_with_default)
    assert schema_with_default.validate({"name": "John"}) == {"name": "John", "age": 25}

    # Test read-only field
    fields_with_readonly = {
        "name": Field(required=True),
        "id": Field(required=False, read_only=True),
    }
    schema_with_readonly = Schema(fields=fields_with_readonly)
    assert schema_with_readonly.validate({"name": "John"}) == {"name": "John"}


# LLM-generated content at query #98
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    fields = {
        "name": Field(allow_null=False),
        "age": Field(allow_null=True),
    }
    schema = Schema(fields=fields)

    # Valid input
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test with default value
    fields_with_default = {
        "name": Field(allow_null=False, default="default_name"),
        "age": Field(allow_null=True, default=0),
    }
    schema_with_default = Schema(fields=fields_with_default)
    assert schema_with_default.validate({}) == {"name": "default_name", "age": 0}

    # Test with missing required field
    try:
        schema.validate({"age": 30})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "required"
        assert e.messages[0].index == ["name"]

    # Test with null input and allow_null=False
    try:
        schema.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "null"

    # Test with non-dict input
    try:
        schema.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "type"

    # Test with invalid key type
    try:
        schema.validate({123: "invalid key"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "invalid_key"
        assert e.messages[0].index == [123]

    # Test with nested field validation error
    nested_fields = {
        "name": Field(allow_null=False),
        "details": Schema(fields={"age": Field(allow_null=False)}),
    }
    nested_schema = Schema(fields=nested_fields)
    try:
        nested_schema.validate({"name": "John", "details": {"age": "invalid"}})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "type"
        assert e.messages[0].index == ["details", "age"]


# LLM-generated content at query #99
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    schema = Schema(fields={"name": Field(), "age": Field()})
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test allow_null
    schema = Schema(fields={"name": Field()}, allow_null=True)
    assert schema.validate(None) is None

    # Test null not allowed
    schema = Schema(fields={"name": Field()})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate(None)
    assert exc_info.value.messages[0].code == "null"

    # Test non-dict input
    schema = Schema(fields={"name": Field()})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate("not a dict")
    assert exc_info.value.messages[0].code == "type"

    # Test invalid key type
    schema = Schema(fields={"name": Field()})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({123: "value"})
    assert exc_info.value.messages[0].code == "invalid_key"

    # Test required field missing
    schema = Schema(fields={"name": Field(), "age": Field()})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({"name": "John"})
    assert exc_info.value.messages[0].code == "required"
    assert exc_info.value.messages[0].index == ["age"]

    # Test field with default value
    schema = Schema(fields={"name": Field(), "age": Field(default=25)})
    assert schema.validate({"name": "John"}) == {"name": "John", "age": 25}

    # Test read-only field
    schema = Schema(fields={"name": Field(), "id": Field(read_only=True)})
    assert schema.validate({"name": "John"}) == {"name": "John"}

    # Test nested validation error
    child_schema = Field()
    child_schema.validate = lambda x: (None, ValidationError(messages=[Message(text="Invalid", code="invalid")]))
    schema = Schema(fields={"name": child_schema})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({"name": "invalid_value"})
    assert exc_info.value.messages[0].code == "invalid"
    assert exc_info.value.messages[0].index == ["name"]


# LLM-generated content at query #100
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    fields = {
        "name": Field(allow_null=False),
        "age": Field(allow_null=True, default=0),
    }
    schema = Schema(fields=fields)
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test with default value
    assert schema.validate({"name": "John"}) == {"name": "John", "age": 0}

    # Test with None value and allow_null=True
    schema_with_null = Schema(fields=fields, allow_null=True)
    assert schema_with_null.validate(None) is None

    # Test with None value and allow_null=False
    with pytest.raises(ValidationError) as exc_info:
        schema.validate(None)
    assert exc_info.value.messages[0].code == "null"

    # Test with non-dict input
    with pytest.raises(ValidationError) as exc_info:
        schema.validate("not a dict")
    assert exc_info.value.messages[0].code == "type"

    # Test with non-string key
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({123: "invalid key"})
    assert exc_info.value.messages[0].code == "invalid_key"

    # Test with missing required field
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({"age": 30})
    assert exc_info.value.messages[0].code == "required"

    # Test with nested field validation error
    nested_field = Field(allow_null=False)
    nested_field.validate = lambda x: (None, ValidationError(messages=[Message(text="nested error", code="nested")])) if x == "error" else (x, None)
    fields_with_nested = {"nested": nested_field}
    schema_with_nested = Schema(fields=fields_with_nested)
    with pytest.raises(ValidationError) as exc_info:
        schema_with_nested.validate({"nested": "error"})
    assert exc_info.value.messages[0].code == "nested"


# LLM-generated content at query #101
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    schema = Schema(fields={"name": Field()})
    definitions["test_schema"] = schema
    reference = Reference(to="test_schema", definitions=definitions)

    # Test valid input
    assert reference.validate({"name": "test"}) == {"name": "test"}

    # Test null input with allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test null input with allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test invalid input
    try:
        reference.validate("invalid")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) > 0


# LLM-generated content at query #102
#--------------------------

```python
def test_Schema_validate():
    # Test case 1: Valid input with all required fields
    fields = {
        "name": Field(allow_null=False),
        "age": Field(allow_null=True),
    }
    schema = Schema(fields=fields)
    input_data = {"name": "John", "age": 30}
    result = schema.validate(input_data)
    assert result == {"name": "John", "age": 30}

    # Test case 2: Missing required field
    input_data = {"age": 30}
    try:
        schema.validate(input_data)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "required"
        assert e.messages[0].index == ["name"]

    # Test case 3: Non-string key in input
    input_data = {"name": "John", 123: "invalid"}
    try:
        schema.validate(input_data)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "invalid_key"
        assert e.messages[0].index == [123]

    # Test case 4: Null input with allow_null=False
    schema = Schema(fields=fields, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "null"

    # Test case 5: Non-dict input
    try:
        schema.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "type"

    # Test case 6: Field with default value
    fields_with_default = {
        "name": Field(allow_null=False),
        "age": Field(allow_null=True, default=25),
    }
    schema = Schema(fields=fields_with_default)
    input_data = {"name": "John"}
    result = schema.validate(input_data)
    assert result == {"name": "John", "age": 25}

    # Test case 7: Read-only field
    fields_with_readonly = {
        "name": Field(allow_null=False),
        "id": Field(allow_null=True, read_only=True),
    }
    schema = Schema(fields=fields_with_readonly)
    input_data = {"name": "John"}
    result = schema.validate(input_data)
    assert result == {"name": "John"}

    # Test case 8: Nested validation error
    nested_field = Field(allow_null=False)
    fields = {
        "name": Field(allow_null=False),
        "details": nested_field,
    }
    schema = Schema(fields=fields)
    input_data = {"name": "John", "details": None}
    try:
        schema.validate(input_data)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "null"
        assert e.messages[0].index == ["details"]

    # Test case 9: Null input with allow_null=True
    schema = Schema(fields=fields, allow_null=True)
    result = schema.validate(None)
    assert result is None


# LLM-generated content at query #103
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    schema = Schema(fields={"name": Field(), "age": Field()})
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test null input with allow_null=True
    schema = Schema(fields={"name": Field()}, allow_null=True)
    assert schema.validate(None) is None

    # Test null input with allow_null=False
    schema = Schema(fields={"name": Field()}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test non-dict input
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"

    # Test invalid key type
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate({1: "value"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

    # Test missing required field
    schema = Schema(fields={"name": Field(), "age": Field()})
    try:
        schema.validate({"name": "John"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"

    # Test field with default value
    schema = Schema(fields={"name": Field(), "age": Field(default=25)})
    assert schema.validate({"name": "John"}) == {"name": "John", "age": 25}

    # Test field validation error
    schema = Schema(fields={"name": Field(min_length=3)})
    try:
        schema.validate({"name": "Jo"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "min_length"


# LLM-generated content at query #104
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    fields = {
        "name": Field(allow_null=False),
        "age": Field(allow_null=True),
    }
    schema = Schema(fields=fields)

    # Valid input
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test with default value
    fields_with_default = {
        "name": Field(allow_null=False, default="Unknown"),
        "age": Field(allow_null=True, default=0),
    }
    schema_with_default = Schema(fields=fields_with_default)
    assert schema_with_default.validate({}) == {"name": "Unknown", "age": 0}

    # Test with null value
    schema_allow_null = Schema(fields=fields, allow_null=True)
    assert schema_allow_null.validate(None) is None

    # Test invalid input - null not allowed
    with pytest.raises(ValidationError) as exc_info:
        schema.validate(None)
    assert exc_info.value.messages[0].code == "null"

    # Test invalid input - not a dict
    with pytest.raises(ValidationError) as exc_info:
        schema.validate("not a dict")
    assert exc_info.value.messages[0].code == "type"

    # Test invalid input - non-string key
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({123: "value"})
    assert exc_info.value.messages[0].code == "invalid_key"

    # Test missing required field
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({"age": 30})
    assert exc_info.value.messages[0].code == "required"

    # Test nested validation error
    nested_fields = {
        "name": Field(allow_null=False),
        "age": Field(allow_null=False),
    }
    nested_schema = Schema(fields=nested_fields)
    with pytest.raises(ValidationError) as exc_info:
        nested_schema.validate({"name": "John", "age": None})
    assert exc_info.value.messages[0].code == "null"


# LLM-generated content at query #105
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field()
    definitions["test_field"] = target_field
    reference = Reference(to="test_field", definitions=definitions)

    # Test with valid value
    assert reference.validate("valid_value") == "valid_value"

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with target field raising ValidationError
    target_field_with_error = Field()
    target_field_with_error.validate = lambda x: (_ for _ in ()).throw(ValidationError(messages=[Message(text="error", code="error")]))
    definitions["error_field"] = target_field_with_error
    error_reference = Reference(to="error_field", definitions=definitions)
    try:
        error_reference.validate("any_value")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "error"


# LLM-generated content at query #106
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    schema = Schema(fields={"name": Field(), "age": Field()})
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test null input with allow_null=True
    schema = Schema(fields={"name": Field()}, allow_null=True)
    assert schema.validate(None) is None

    # Test null input with allow_null=False
    schema = Schema(fields={"name": Field()}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test non-dict input
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"

    # Test invalid key type
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate({1: "value"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

    # Test missing required field
    schema = Schema(fields={"name": Field(), "age": Field()})
    try:
        schema.validate({"name": "John"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"

    # Test field with default value
    schema = Schema(fields={"name": Field(), "age": Field(default=25)})
    assert schema.validate({"name": "John"}) == {"name": "John", "age": 25}

    # Test read-only field
    schema = Schema(fields={"name": Field(), "id": Field(read_only=True)})
    assert schema.validate({"name": "John"}) == {"name": "John"}


# LLM-generated content at query #107
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    field = Field()
    definitions['test_field'] = field
    reference = Reference(to='test_field', definitions=definitions)

    # Test with valid value
    assert reference.validate("valid_value") == "valid_value"

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    with pytest.raises(ValidationError):
        reference.validate(None)

    # Test with invalid value (assuming field.validate raises ValidationError)
    with pytest.raises(ValidationError):
        reference.validate("invalid_value")


# LLM-generated content at query #108
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    schema = Schema(fields={"name": Field(), "age": Field()})
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test null input with allow_null=True
    schema = Schema(fields={"name": Field()}, allow_null=True)
    assert schema.validate(None) is None

    # Test null input with allow_null=False
    schema = Schema(fields={"name": Field()}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test non-dict input
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"

    # Test non-string key
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate({123: "value"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

    # Test missing required field
    schema = Schema(fields={"name": Field(), "age": Field()})
    try:
        schema.validate({"name": "John"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"

    # Test field with default value
    schema = Schema(fields={"name": Field(), "age": Field(default=25)})
    assert schema.validate({"name": "John"}) == {"name": "John", "age": 25}

    # Test read-only field
    schema = Schema(fields={"name": Field(), "id": Field(read_only=True)})
    assert schema.validate({"name": "John"}) == {"name": "John"}

    # Test nested validation error
    inner_schema = Schema(fields={"street": Field(), "city": Field()})
    schema = Schema(fields={"name": Field(), "address": inner_schema})
    try:
        schema.validate({"name": "John", "address": {"street": "Main St"}})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"
        assert e.messages[0].index == ["address", "city"]


# LLM-generated content at query #109
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field()
    definitions["test_field"] = target_field
    reference = Reference(to="test_field", definitions=definitions)

    # Test valid value
    assert reference.validate("valid_value") == "valid_value"

    # Test null value with allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test null value with allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test that validate calls target's validate
    target_field.validate = lambda x: x.upper()
    assert reference.validate("test") == "TEST"


# LLM-generated content at query #110
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    fields = {
        "name": Field(allow_null=False),
        "age": Field(allow_null=True)
    }
    schema = Schema(fields=fields)
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test with default values
    fields_with_defaults = {
        "name": Field(allow_null=False, default="Unknown"),
        "age": Field(allow_null=True, default=0)
    }
    schema_with_defaults = Schema(fields=fields_with_defaults)
    assert schema_with_defaults.validate({}) == {"name": "Unknown", "age": 0}

    # Test with missing required field
    schema = Schema(fields=fields)
    try:
        schema.validate({"age": 30})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "required"
        assert e.messages[0].index == ["name"]

    # Test with null input when allow_null is False
    schema = Schema(fields=fields, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "null"

    # Test with non-dict input
    schema = Schema(fields=fields)
    try:
        schema.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "type"

    # Test with non-string keys
    schema = Schema(fields=fields)
    try:
        schema.validate({123: "value"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "invalid_key"
        assert e.messages[0].index == [123]

    # Test with nested validation errors
    nested_fields = {
        "user": Schema(fields={"name": Field(allow_null=False)})
    }
    nested_schema = Schema(fields=nested_fields)
    try:
        nested_schema.validate({"user": {"name": None}})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "null"
        assert e.messages[0].index == ["user", "name"]


# LLM-generated content at query #111
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    schema = Schema(fields={"name": Field(), "age": Field()})
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test null input with allow_null=True
    schema = Schema(fields={"name": Field()}, allow_null=True)
    assert schema.validate(None) is None

    # Test null input with allow_null=False
    schema = Schema(fields={"name": Field()}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test non-dict input
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"

    # Test invalid key type
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate({123: "invalid key"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

    # Test missing required field
    schema = Schema(fields={"name": Field(), "age": Field()})
    try:
        schema.validate({"name": "John"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"

    # Test field validation error
    schema = Schema(fields={"name": Field(min_length=5)})
    try:
        schema.validate({"name": "John"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "min_length"

    # Test default value
    schema = Schema(fields={"name": Field(default="default_name")})
    assert schema.validate({}) == {"name": "default_name"}

    # Test read-only field
    schema = Schema(fields={"name": Field(read_only=True)})
    assert schema.validate({}) == {}


# LLM-generated content at query #112
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    field = Field()
    definitions["test_field"] = field
    reference = Reference(to="test_field", definitions=definitions)

    # Test with valid value
    assert reference.validate("valid_value") == "valid_value"

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with invalid value (assuming field.validate raises ValidationError)
    try:
        reference.validate("invalid_value")
        assert False, "Expected ValidationError"
    except ValidationError:
        pass


# LLM-generated content at query #113
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    fields = {"name": Field(), "age": Field()}
    schema = Schema(fields=fields)
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test null input with allow_null=True
    schema = Schema(fields=fields, allow_null=True)
    assert schema.validate(None) is None

    # Test null input with allow_null=False
    schema = Schema(fields=fields, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test non-dict input
    try:
        schema.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"

    # Test invalid key type
    try:
        schema.validate({1: "value"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

    # Test missing required field
    try:
        schema.validate({"name": "John"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"

    # Test field with default value
    fields = {"name": Field(), "age": Field(default=25)}
    schema = Schema(fields=fields)
    assert schema.validate({"name": "John"}) == {"name": "John", "age": 25}

    # Test read-only field
    fields = {"name": Field(), "id": Field(read_only=True)}
    schema = Schema(fields=fields)
    assert schema.validate({"name": "John"}) == {"name": "John"}

    # Test child field validation error
    fields = {"name": Field(), "age": Field(min_value=0)}
    schema = Schema(fields=fields)
    try:
        schema.validate({"name": "John", "age": -5})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "min_value"


# LLM-generated content at query #114
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    field = Field()
    definitions['test_field'] = field
    reference = Reference(to='test_field', definitions=definitions)

    # Test with valid value
    assert reference.validate("valid_value") == "valid_value"

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    with pytest.raises(ValidationError) as exc_info:
        reference.validate(None)
    assert exc_info.value.messages[0].code == "null"

    # Test with invalid value
    with pytest.raises(ValidationError):
        field.validate = lambda x: (_ for _ in ()).throw(ValidationError("Invalid"))
        reference.validate("invalid_value")


# LLM-generated content at query #115
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field()
    definitions['test_field'] = target_field
    reference = Reference(to='test_field', definitions=definitions)

    # Test with valid value
    assert reference.validate("valid_value") == "valid_value"

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with target field validation error
    target_field_with_error = Field()
    target_field_with_error.validate = lambda x: 1/0  # Force an error
    definitions['error_field'] = target_field_with_error
    error_reference = Reference(to='error_field', definitions=definitions)
    try:
        error_reference.validate("any_value")
        assert False, "Expected ValidationError"
    except ValidationError:
        pass


# LLM-generated content at query #116
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    fields = {
        "name": Field(),
        "age": Field(),
    }
    schema = Schema(fields=fields)
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test None value with allow_null=True
    schema_with_null = Schema(fields=fields, allow_null=True)
    assert schema_with_null.validate(None) is None

    # Test None value with allow_null=False
    with pytest.raises(ValidationError) as exc_info:
        schema.validate(None)
    assert exc_info.value.messages[0].code == "null"

    # Test non-dict input
    with pytest.raises(ValidationError) as exc_info:
        schema.validate("not a dict")
    assert exc_info.value.messages[0].code == "type"

    # Test invalid key type
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({1: "value"})
    assert exc_info.value.messages[0].code == "invalid_key"

    # Test missing required field
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({"name": "John"})
    assert exc_info.value.messages[0].code == "required"

    # Test field with default value
    fields_with_default = {
        "name": Field(),
        "age": Field(default=25),
    }
    schema_with_default = Schema(fields=fields_with_default)
    assert schema_with_default.validate({"name": "John"}) == {"name": "John", "age": 25}

    # Test read-only field
    fields_with_readonly = {
        "name": Field(),
        "id": Field(read_only=True),
    }
    schema_with_readonly = Schema(fields=fields_with_readonly)
    assert schema_with_readonly.validate({"name": "John"}) == {"name": "John"}

    # Test nested validation error
    nested_fields = {
        "name": Field(),
        "details": Schema(fields={"age": Field()}),
    }
    nested_schema = Schema(fields=nested_fields)
    with pytest.raises(ValidationError) as exc_info:
        nested_schema.validate({"name": "John", "details": {"age": "invalid"}})
    assert exc_info.value.messages[0].index == ["details", "age"]


# LLM-generated content at query #117
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    fields = {
        "name": Field(allow_null=False),
        "age": Field(allow_null=True, default=0),
    }
    schema = Schema(fields=fields)
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test null input with allow_null=True
    schema_allow_null = Schema(fields=fields, allow_null=True)
    assert schema_allow_null.validate(None) is None

    # Test null input with allow_null=False
    with pytest.raises(ValidationError):
        schema.validate(None)

    # Test non-dict input
    with pytest.raises(ValidationError):
        schema.validate("not a dict")

    # Test invalid key type
    with pytest.raises(ValidationError):
        schema.validate({123: "invalid key"})

    # Test missing required field
    with pytest.raises(ValidationError):
        schema.validate({"age": 30})

    # Test default value for missing optional field
    assert schema.validate({"name": "John"}) == {"name": "John", "age": 0}

    # Test read-only field is ignored in validation
    fields_with_readonly = {
        "name": Field(allow_null=False),
        "id": Field(read_only=True),
    }
    schema_readonly = Schema(fields=fields_with_readonly)
    assert schema_readonly.validate({"name": "John"}) == {"name": "John"}


# LLM-generated content at query #118
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    schema = Schema(fields={"name": Field(), "age": Field()})
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test null input with allow_null=True
    schema = Schema(fields={"name": Field()}, allow_null=True)
    assert schema.validate(None) is None

    # Test null input with allow_null=False
    schema = Schema(fields={"name": Field()}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test non-dict input
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"

    # Test invalid key type
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate({123: "value"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

    # Test missing required field
    schema = Schema(fields={"name": Field(), "age": Field()})
    try:
        schema.validate({"name": "John"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"

    # Test field with default value
    schema = Schema(fields={"name": Field(default="default_name")})
    assert schema.validate({}) == {"name": "default_name"}

    # Test read-only field
    schema = Schema(fields={"name": Field(read_only=True), "age": Field()})
    assert schema.validate({"age": 30}) == {"age": 30}

    # Test nested validation error
    inner_schema = Schema(fields={"value": Field()})
    schema = Schema(fields={"inner": inner_schema})
    try:
        schema.validate({"inner": {"invalid": "data"}})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].index == ["inner", "required"]


# LLM-generated content at query #119
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_schema = Schema(fields={"name": Field()})
    definitions["test_schema"] = target_schema
    reference = Reference(to="test_schema", definitions=definitions)

    # Test valid input
    valid_input = {"name": "test"}
    assert reference.validate(valid_input) == {"name": "test"}

    # Test null input with allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test null input with allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError for null input"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test invalid input (passed to target schema)
    invalid_input = "not a dict"
    try:
        reference.validate(invalid_input)
        assert False, "Expected ValidationError for invalid input"
    except ValidationError as e:
        assert e.messages[0].code == "type"


# LLM-generated content at query #120
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    definitions["test_schema"] = Schema(fields={"name": Field()})
    reference = Reference(to="test_schema", definitions=definitions)

    # Test valid input
    valid_input = {"name": "test"}
    assert reference.validate(valid_input) == valid_input

    # Test null input with allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test null input with allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test invalid input (passed to target schema)
    invalid_input = {"name": 123}  # Assuming Field() expects a string
    try:
        reference.validate(invalid_input)
        assert False, "Expected ValidationError"
    except ValidationError:
        pass


# LLM-generated content at query #121
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    definitions["test_schema"] = Schema(fields={"name": Field()})

    reference = Reference(to="test_schema", definitions=definitions)

    # Test valid input
    valid_input = {"name": "test"}
    assert reference.validate(valid_input) == {"name": "test"}

    # Test null input with allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test null input with allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test invalid input
    try:
        reference.validate("invalid")
        assert False, "Expected ValidationError"
    except ValidationError:
        pass


# LLM-generated content at query #122
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field()
    definitions['test_field'] = target_field
    reference = Reference(to='test_field', definitions=definitions)

    # Test with valid value
    assert reference.validate("valid_value") == "valid_value"

    # Test with null value and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with null value and allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with target field validation error
    target_field.errors = {"type": "Invalid type"}
    target_field.validate = lambda value: (_ for _ in ()).throw(ValidationError(messages=[Message(text="Invalid type", code="type")]))
    try:
        reference.validate("invalid_value")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_Schema_serialize():
    # Test basic serialization
    schema = Schema({"name": Field(), "age": Field()})
    obj = {"name": "John", "age": 30}
    assert schema.serialize(obj) == {"name": "John", "age": 30}

    # Test with None
    assert schema.serialize(None) is None

    # Test with missing keys
    obj = {"name": "John"}
    assert schema.serialize(obj) == {"name": "John"}

    # Test with nested fields
    nested_schema = Schema({"name": Field(), "address": Schema({"city": Field()})})
    obj = {"name": "John", "address": {"city": "NYC"}}
    assert nested_schema.serialize(obj) == {"name": "John", "address": {"city": "NYC"}}

    # Test with object attributes
    class Person:
        def __init__(self):
            self.name = "John"
            self.age = 30

    obj = Person()
    assert schema.serialize(obj) == {"name": "John", "age": 30}


# LLM-generated content at query #2
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    schema = Schema(fields={"name": Field(), "age": Field()})
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test with None and allow_null=True
    schema = Schema(fields={"name": Field()}, allow_null=True)
    assert schema.validate(None) is None

    # Test with None and allow_null=False
    schema = Schema(fields={"name": Field()}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test invalid type
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"

    # Test invalid key type
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate({123: "invalid key"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

    # Test missing required field
    schema = Schema(fields={"name": Field(), "age": Field()})
    try:
        schema.validate({"name": "John"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"

    # Test with default value
    schema = Schema(fields={"name": Field(), "age": Field(default=25)})
    assert schema.validate({"name": "John"}) == {"name": "John", "age": 25}

    # Test with read-only field
    schema = Schema(fields={"name": Field(), "id": Field(read_only=True)})
    assert schema.validate({"name": "John"}) == {"name": "John"}

    # Test child field validation error
    child_field = Field()
    child_field.validate = lambda x: (None, ValidationError(messages=[Message(text="Invalid", code="invalid")]))
    schema = Schema(fields={"name": child_field})
    try:
        schema.validate({"name": "invalid"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid"


# LLM-generated content at query #3
#--------------------------

```python
def test_Schema_serialize():
    # Test with a simple schema and dict input
    schema = Schema(fields={
        'name': Field(),
        'age': Field(),
    })
    input_data = {'name': 'John', 'age': 30}
    assert schema.serialize(input_data) == {'name': 'John', 'age': 30}

    # Test with None input
    assert schema.serialize(None) is None

    # Test with an object that has attributes
    class Person:
        def __init__(self, name, age):
            self.name = name
            self.age = age

    person = Person('Jane', 25)
    assert schema.serialize(person) == {'name': 'Jane', 'age': 25}

    # Test with missing fields
    input_data = {'name': 'John'}
    assert schema.serialize(input_data) == {'name': 'John'}

    # Test with nested schema
    nested_schema = Schema(fields={
        'address': Schema(fields={
            'street': Field(),
            'city': Field(),
        }),
    })
    input_data = {'address': {'street': '123 Main St', 'city': 'New York'}}
    assert nested_schema.serialize(input_data) == {'address': {'street': '123 Main St', 'city': 'New York'}}


# LLM-generated content at query #4
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    fields = {
        "name": Field(allow_null=False),
        "age": Field(allow_null=True),
    }
    schema = Schema(fields=fields)
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test null input with allow_null=True
    schema_with_null = Schema(fields=fields, allow_null=True)
    assert schema_with_null.validate(None) is None

    # Test null input with allow_null=False
    schema_no_null = Schema(fields=fields, allow_null=False)
    try:
        schema_no_null.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test non-dict input
    try:
        schema.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"

    # Test invalid key type
    try:
        schema.validate({123: "invalid key"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

    # Test missing required field
    required_fields = {
        "name": Field(allow_null=False),
    }
    required_schema = Schema(fields=required_fields)
    try:
        required_schema.validate({"age": 30})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"

    # Test field with default value
    fields_with_default = {
        "name": Field(allow_null=False, default="default_name"),
    }
    schema_with_default = Schema(fields=fields_with_default)
    assert schema_with_default.validate({}) == {"name": "default_name"}

    # Test nested validation error
    nested_fields = {
        "name": Field(allow_null=False),
        "age": Field(allow_null=False),
    }
    nested_schema = Schema(fields=nested_fields)
    try:
        nested_schema.validate({"name": "John", "age": "invalid"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) > 0


# LLM-generated content at query #5
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    fields = {
        "name": Field(allow_null=False),
        "age": Field(allow_null=True),
    }
    schema = Schema(fields=fields)
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test null input with allow_null=True
    schema_with_null = Schema(fields=fields, allow_null=True)
    assert schema_with_null.validate(None) is None

    # Test null input with allow_null=False
    with pytest.raises(ValidationError) as exc_info:
        schema.validate(None)
    assert exc_info.value.messages[0].code == "null"

    # Test non-dict input
    with pytest.raises(ValidationError) as exc_info:
        schema.validate("not a dict")
    assert exc_info.value.messages[0].code == "type"

    # Test invalid key type
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({123: "invalid key"})
    assert exc_info.value.messages[0].code == "invalid_key"

    # Test missing required field
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({"age": 30})
    assert exc_info.value.messages[0].code == "required"

    # Test field validation error
    age_field = Field(allow_null=False)
    age_field.validate = lambda x: (None, ValidationError(messages=[Message(text="Invalid age", code="invalid")])) if x < 0 else (x, None)
    fields_with_error = {"age": age_field}
    schema_with_error = Schema(fields=fields_with_error)
    with pytest.raises(ValidationError) as exc_info:
        schema_with_error.validate({"age": -5})
    assert exc_info.value.messages[0].code == "invalid"


# LLM-generated content at query #6
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    definitions['test_schema'] = Schema(fields={'name': Field()})
    reference = Reference(to='test_schema', definitions=definitions)

    # Test valid input
    valid_input = {'name': 'test'}
    assert reference.validate(valid_input) == valid_input

    # Test null input with allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test null input with allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test invalid input (should raise error from target schema)
    try:
        reference.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"


# LLM-generated content at query #7
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    fields = {
        "name": Field(allow_null=False),
        "age": Field(allow_null=True),
    }
    schema = Schema(fields=fields)
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test null input with allow_null=True
    schema_with_null = Schema(fields=fields, allow_null=True)
    assert schema_with_null.validate(None) is None

    # Test null input with allow_null=False
    schema_no_null = Schema(fields=fields, allow_null=False)
    try:
        schema_no_null.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test non-dict input
    try:
        schema.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"

    # Test invalid key type
    try:
        schema.validate({123: "value"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

    # Test missing required field
    try:
        schema.validate({"age": 30})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"

    # Test field validation error
    fields_with_validation = {
        "name": Field(allow_null=False, validators=[lambda x: x == "John" or "Name must be John"]),
    }
    schema_with_validation = Schema(fields=fields_with_validation)
    try:
        schema_with_validation.validate({"name": "Jane"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid"

    # Test default value
    fields_with_default = {
        "name": Field(default="DefaultName"),
    }
    schema_with_default = Schema(fields=fields_with_default)
    assert schema_with_default.validate({}) == {"name": "DefaultName"}

    # Test read-only field
    fields_with_readonly = {
        "name": Field(read_only=True),
        "age": Field(),
    }
    schema_with_readonly = Schema(fields=fields_with_readonly)
    assert schema_with_readonly.validate({"age": 30}) == {"age": 30}


# LLM-generated content at query #8
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    fields = {
        "name": Field(allow_null=False),
        "age": Field(allow_null=True),
    }
    schema = Schema(fields=fields)
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test null input with allow_null=True
    schema_with_null = Schema(fields=fields, allow_null=True)
    assert schema_with_null.validate(None) is None

    # Test null input with allow_null=False
    with pytest.raises(ValidationError) as exc_info:
        schema.validate(None)
    assert exc_info.value.messages[0].code == "null"

    # Test non-dict input
    with pytest.raises(ValidationError) as exc_info:
        schema.validate("not a dict")
    assert exc_info.value.messages[0].code == "type"

    # Test invalid key type
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({1: "invalid key"})
    assert exc_info.value.messages[0].code == "invalid_key"

    # Test missing required field
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({"age": 30})
    assert exc_info.value.messages[0].code == "required"

    # Test field validation error
    fields_with_validation = {
        "name": Field(allow_null=False, validators=[lambda x: x != "invalid"]),
    }
    schema_with_validation = Schema(fields=fields_with_validation)
    with pytest.raises(ValidationError) as exc_info:
        schema_with_validation.validate({"name": "invalid"})
    assert exc_info.value.messages[0].code == "invalid"

    # Test default value
    fields_with_default = {
        "name": Field(allow_null=False, default="default"),
    }
    schema_with_default = Schema(fields=fields_with_default)
    assert schema_with_default.validate({}) == {"name": "default"}

    # Test read-only field
    fields_with_readonly = {
        "name": Field(allow_null=False, read_only=True),
    }
    schema_with_readonly = Schema(fields=fields_with_readonly)
    assert schema_with_readonly.validate({}) == {}


# LLM-generated content at query #9
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    definitions["test_schema"] = Schema(fields={"name": Field()})
    reference = Reference(to="test_schema", definitions=definitions)

    # Test valid input
    valid_input = {"name": "test"}
    assert reference.validate(valid_input) == {"name": "test"}

    # Test null input with allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test null input with allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test invalid input (passed to target schema validation)
    try:
        reference.validate({"invalid_key": 123})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert any(msg.code == "invalid_key" for msg in e.messages)


# LLM-generated content at query #10
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    schema = Schema(fields={"name": Field(), "age": Field()})
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test null input with allow_null=True
    schema = Schema(fields={"name": Field()}, allow_null=True)
    assert schema.validate(None) is None

    # Test null input with allow_null=False
    schema = Schema(fields={"name": Field()}, allow_null=False)
    with pytest.raises(ValidationError):
        schema.validate(None)

    # Test non-dict input
    schema = Schema(fields={"name": Field()})
    with pytest.raises(ValidationError):
        schema.validate("not a dict")

    # Test invalid key type
    schema = Schema(fields={"name": Field()})
    with pytest.raises(ValidationError):
        schema.validate({123: "invalid key"})

    # Test required field missing
    schema = Schema(fields={"name": Field(), "age": Field()})
    with pytest.raises(ValidationError):
        schema.validate({"name": "John"})

    # Test field with default value
    schema = Schema(fields={"name": Field(), "age": Field(default=25)})
    assert schema.validate({"name": "John"}) == {"name": "John", "age": 25}

    # Test read-only field
    schema = Schema(fields={"name": Field(), "id": Field(read_only=True)})
    assert schema.validate({"name": "John"}) == {"name": "John"}

    # Test child field validation error
    class PositiveField(Field):
        def validate(self, value):
            if value <= 0:
                raise self.validation_error("must be positive")
            return value

    schema = Schema(fields={"name": Field(), "age": PositiveField()})
    with pytest.raises(ValidationError):
        schema.validate({"name": "John", "age": -5})


# LLM-generated content at query #11
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    schema = Schema(fields={"name": Field(), "age": Field()})
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test null input with allow_null=True
    schema = Schema(fields={"name": Field()}, allow_null=True)
    assert schema.validate(None) is None

    # Test null input with allow_null=False
    schema = Schema(fields={"name": Field()}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test non-dict input
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"

    # Test non-string keys
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate({123: "invalid key"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

    # Test missing required field
    schema = Schema(fields={"name": Field(), "age": Field()})
    try:
        schema.validate({"name": "John"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"

    # Test field with default value
    schema = Schema(fields={"name": Field(), "age": Field(default=25)})
    assert schema.validate({"name": "John"}) == {"name": "John", "age": 25}

    # Test read-only field
    schema = Schema(fields={"name": Field(), "id": Field(read_only=True)})
    assert schema.validate({"name": "John"}) == {"name": "John"}

    # Test nested validation error
    inner_schema = Schema(fields={"street": Field(), "city": Field()})
    schema = Schema(fields={"name": Field(), "address": inner_schema})
    try:
        schema.validate({"name": "John", "address": {"street": "Main St"}})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"
        assert e.messages[0].index == ["address", "city"]


# LLM-generated content at query #12
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    definitions['test_schema'] = Schema(fields={'name': Field()})

    reference = Reference(to='test_schema', definitions=definitions)

    # Test valid input
    assert reference.validate({'name': 'value'}) == {'name': 'value'}

    # Test null input with allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test null input with allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test invalid input (should propagate from target schema)
    try:
        reference.validate({'invalid_key': 123})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"


# LLM-generated content at query #13
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field()
    definitions['test_field'] = target_field
    reference = Reference(to='test_field', definitions=definitions)

    # Test valid value
    assert reference.validate("valid_value") == "valid_value"

    # Test null value with allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test null value with allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test that validate delegates to target's validate
    target_field.validate = lambda x: x.upper()
    assert reference.validate("test") == "TEST"


# LLM-generated content at query #14
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    fields = {
        "name": Field(allow_null=False),
        "age": Field(allow_null=True),
    }
    schema = Schema(fields=fields)
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test null input with allow_null=True
    schema_allow_null = Schema(fields=fields, allow_null=True)
    assert schema_allow_null.validate(None) is None

    # Test null input with allow_null=False
    with pytest.raises(ValidationError) as exc_info:
        schema.validate(None)
    assert exc_info.value.messages[0].code == "null"

    # Test non-dict input
    with pytest.raises(ValidationError) as exc_info:
        schema.validate("not a dict")
    assert exc_info.value.messages[0].code == "type"

    # Test invalid key type
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({123: "invalid key"})
    assert exc_info.value.messages[0].code == "invalid_key"

    # Test missing required field
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({"age": 30})
    assert exc_info.value.messages[0].code == "required"

    # Test field validation error
    age_field = Field(allow_null=False)
    age_field.validate = lambda x: (None, ValidationError(messages=[Message(text="Invalid age", code="invalid")])) if x < 0 else (x, None)
    fields_with_error = {"age": age_field}
    schema_with_error = Schema(fields=fields_with_error)
    with pytest.raises(ValidationError) as exc_info:
        schema_with_error.validate({"age": -5})
    assert exc_info.value.messages[0].code == "invalid"

    # Test default value
    field_with_default = Field(default="default_value")
    fields_with_default = {"name": field_with_default}
    schema_with_default = Schema(fields=fields_with_default)
    assert schema_with_default.validate({}) == {"name": "default_value"}


# LLM-generated content at query #15
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    schema = Schema(fields={"name": Field(), "age": Field()})
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test null input with allow_null=True
    schema = Schema(fields={"name": Field()}, allow_null=True)
    assert schema.validate(None) is None

    # Test null input with allow_null=False
    schema = Schema(fields={"name": Field()}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test non-dict input
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"

    # Test non-string keys
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate({123: "value"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

    # Test missing required field
    schema = Schema(fields={"name": Field(), "age": Field()})
    try:
        schema.validate({"name": "John"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"

    # Test field with default value
    schema = Schema(fields={"name": Field(default="default_name")})
    assert schema.validate({}) == {"name": "default_name"}

    # Test read-only field
    schema = Schema(fields={"name": Field(), "id": Field(read_only=True)})
    assert schema.validate({"name": "John"}) == {"name": "John"}

    # Test nested validation errors
    child_schema = Field()
    child_schema.validate = lambda x: (None, ValidationError(messages=[Message(text="error", code="child_error")]))
    schema = Schema(fields={"child": child_schema})
    try:
        schema.validate({"child": "value"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "child_error"


# LLM-generated content at query #16
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    fields = {
        "name": Field(allow_null=False),
        "age": Field(allow_null=True),
    }
    schema = Schema(fields=fields)
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test null input with allow_null=True
    schema_allow_null = Schema(fields=fields, allow_null=True)
    assert schema_allow_null.validate(None) is None

    # Test null input with allow_null=False
    with pytest.raises(ValidationError):
        schema.validate(None)

    # Test non-dict input
    with pytest.raises(ValidationError):
        schema.validate("not a dict")

    # Test invalid key type
    with pytest.raises(ValidationError):
        schema.validate({1: "value"})

    # Test missing required field
    with pytest.raises(ValidationError):
        schema.validate({"age": 30})

    # Test field validation error
    fields_with_validation = {
        "name": Field(allow_null=False, validators=[lambda x: x == "John"]),
    }
    schema_with_validation = Schema(fields=fields_with_validation)
    with pytest.raises(ValidationError):
        schema_with_validation.validate({"name": "Jane"})

    # Test default value
    fields_with_default = {
        "name": Field(allow_null=False, default="DefaultName"),
    }
    schema_with_default = Schema(fields=fields_with_default)
    assert schema_with_default.validate({}) == {"name": "DefaultName"}

    # Test read-only field
    fields_with_readonly = {
        "name": Field(allow_null=False, read_only=True),
    }
    schema_with_readonly = Schema(fields=fields_with_readonly)
    assert schema_with_readonly.validate({}) == {}


# LLM-generated content at query #17
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    schema = Schema(fields={"name": Field(), "age": Field()})
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test null input with allow_null=True
    schema = Schema(fields={"name": Field()}, allow_null=True)
    assert schema.validate(None) is None

    # Test null input with allow_null=False
    schema = Schema(fields={"name": Field()}, allow_null=False)
    with pytest.raises(ValidationError):
        schema.validate(None)

    # Test non-dict input
    schema = Schema(fields={"name": Field()})
    with pytest.raises(ValidationError):
        schema.validate("not a dict")

    # Test invalid key type
    schema = Schema(fields={"name": Field()})
    with pytest.raises(ValidationError):
        schema.validate({1: "value"})

    # Test missing required field
    schema = Schema(fields={"name": Field(), "age": Field()})
    with pytest.raises(ValidationError):
        schema.validate({"name": "John"})

    # Test field with default value
    schema = Schema(fields={"name": Field(), "age": Field(default=25)})
    assert schema.validate({"name": "John"}) == {"name": "John", "age": 25}

    # Test read-only field
    schema = Schema(fields={"name": Field(), "id": Field(read_only=True)})
    assert schema.validate({"name": "John"}) == {"name": "John"}


# LLM-generated content at query #18
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field()
    definitions['test_ref'] = target_field
    reference = Reference(to='test_ref', definitions=definitions)

    # Test with valid value
    assert reference.validate("valid_value") == "valid_value"

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test that target's validate is called
    target_field.validate = lambda x: x.upper()
    assert reference.validate("test") == "TEST"


# LLM-generated content at query #19
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    schema = Schema(fields={"name": Field(), "age": Field()})
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test null input with allow_null=True
    schema = Schema(fields={"name": Field()}, allow_null=True)
    assert schema.validate(None) is None

    # Test null input with allow_null=False
    schema = Schema(fields={"name": Field()}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test non-dict input
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"

    # Test invalid key type
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate({123: "invalid key"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

    # Test missing required field
    schema = Schema(fields={"name": Field(), "age": Field()})
    try:
        schema.validate({"name": "John"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"

    # Test field with default value
    schema = Schema(fields={"name": Field(), "age": Field(default=25)})
    assert schema.validate({"name": "John"}) == {"name": "John", "age": 25}

    # Test read-only field
    schema = Schema(fields={"name": Field(), "id": Field(read_only=True)})
    assert schema.validate({"name": "John"}) == {"name": "John"}

    # Test nested validation error
    inner_schema = Schema(fields={"street": Field(), "city": Field()})
    schema = Schema(fields={"name": Field(), "address": inner_schema})
    try:
        schema.validate({"name": "John", "address": {"street": "Main St"}})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"
        assert e.messages[0].index == ["address", "city"]


# LLM-generated content at query #20
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    definitions['test_schema'] = Schema(fields={'name': Field()})

    reference = Reference(to='test_schema', definitions=definitions)

    # Test valid input
    valid_input = {'name': 'value'}
    assert reference.validate(valid_input) == valid_input

    # Test null input with allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test null input with allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test invalid input (passed to target schema)
    try:
        reference.validate({'invalid_key': 123})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"


# LLM-generated content at query #21
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    schema = Schema(fields={"name": Field(), "age": Field()})
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test null input with allow_null=True
    schema = Schema(fields={"name": Field()}, allow_null=True)
    assert schema.validate(None) is None

    # Test null input with allow_null=False
    schema = Schema(fields={"name": Field()}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test non-dict input
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"

    # Test non-string key
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate({123: "value"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

    # Test missing required field
    schema = Schema(fields={"name": Field(), "age": Field()})
    try:
        schema.validate({"name": "John"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"

    # Test field with default value
    schema = Schema(fields={"name": Field(), "age": Field(default=25)})
    assert schema.validate({"name": "John"}) == {"name": "John", "age": 25}

    # Test read-only field
    schema = Schema(fields={"name": Field(), "id": Field(read_only=True)})
    assert schema.validate({"name": "John"}) == {"name": "John"}

    # Test nested validation error
    inner_schema = Schema(fields={"value": Field()})
    schema = Schema(fields={"inner": inner_schema})
    try:
        schema.validate({"inner": {"value": None}})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].index == ["inner", "value"]


# LLM-generated content at query #22
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    schema = Schema(fields={"name": Field()})
    definitions["test_schema"] = schema
    reference = Reference(to="test_schema", definitions=definitions)

    # Test valid input
    assert reference.validate({"name": "test"}) == {"name": "test"}

    # Test null input with allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test null input with allow_null=False
    reference.allow_null = False
    with pytest.raises(ValidationError) as exc_info:
        reference.validate(None)
    assert exc_info.value.messages[0].code == "null"

    # Test invalid input (passed to target schema)
    with pytest.raises(ValidationError) as exc_info:
        reference.validate("not a dict")
    assert exc_info.value.messages[0].code == "type"


# LLM-generated content at query #23
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field()
    definitions['test_ref'] = target_field
    ref = Reference(to='test_ref', definitions=definitions)

    # Test with valid value
    assert ref.validate("valid_value") == "valid_value"

    # Test with None and allow_null=True
    ref.allow_null = True
    assert ref.validate(None) is None

    # Test with None and allow_null=False
    ref.allow_null = False
    try:
        ref.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with target field raising ValidationError
    target_field = Field()
    target_field.validate = lambda x: (_ for _ in ()).throw(ValidationError(messages=[Message(text="test", code="test")]))
    definitions['test_ref'] = target_field
    ref = Reference(to='test_ref', definitions=definitions)
    try:
        ref.validate("invalid_value")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "test"


# LLM-generated content at query #24
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field(allow_null=False)
    definitions["test_field"] = target_field
    reference = Reference(to="test_field", definitions=definitions, allow_null=False)

    # Test valid input
    assert reference.validate("valid_value") == "valid_value"

    # Test null input with allow_null=False
    with pytest.raises(ValidationError):
        reference.validate(None)

    # Test null input with allow_null=True
    reference_allow_null = Reference(to="test_field", definitions=definitions, allow_null=True)
    assert reference_allow_null.validate(None) is None

    # Test invalid input (should raise ValidationError from target field)
    with pytest.raises(ValidationError):
        reference.validate(None)


# LLM-generated content at query #25
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    fields = {
        "name": Field(allow_null=False),
        "age": Field(allow_null=True),
    }
    schema = Schema(fields=fields)
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test null input with allow_null=True
    schema = Schema(fields=fields, allow_null=True)
    assert schema.validate(None) is None

    # Test null input with allow_null=False
    schema = Schema(fields=fields, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test non-dict input
    try:
        schema.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"

    # Test invalid key type
    try:
        schema.validate({1: "value"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

    # Test missing required field
    try:
        schema.validate({"age": 30})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"

    # Test field validation error
    fields = {
        "name": Field(allow_null=False),
        "age": Field(allow_null=False),
    }
    schema = Schema(fields=fields)
    try:
        schema.validate({"name": "John", "age": None})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test default value
    fields = {
        "name": Field(allow_null=False),
        "age": Field(allow_null=True, default=25),
    }
    schema = Schema(fields=fields)
    assert schema.validate({"name": "John"}) == {"name": "John", "age": 25}

    # Test read-only field
    fields = {
        "name": Field(allow_null=False),
        "age": Field(allow_null=True, read_only=True),
    }
    schema = Schema(fields=fields)
    assert schema.validate({"name": "John"}) == {"name": "John"}


# LLM-generated content at query #26
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    schema = Schema(fields={"name": Field(), "age": Field()})
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test null input with allow_null=True
    schema = Schema(fields={"name": Field()}, allow_null=True)
    assert schema.validate(None) is None

    # Test null input with allow_null=False
    schema = Schema(fields={"name": Field()}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test non-dict input
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"

    # Test invalid key type
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate({123: "invalid key"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

    # Test missing required field
    schema = Schema(fields={"name": Field(), "age": Field()})
    try:
        schema.validate({"name": "John"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"

    # Test field with default value
    schema = Schema(fields={"name": Field(), "age": Field(default=25)})
    assert schema.validate({"name": "John"}) == {"name": "John", "age": 25}

    # Test read-only field
    schema = Schema(fields={"name": Field(), "id": Field(read_only=True)})
    assert schema.validate({"name": "John"}) == {"name": "John"}

    # Test nested validation error
    inner_schema = Schema(fields={"street": Field(), "city": Field()})
    schema = Schema(fields={"name": Field(), "address": inner_schema})
    try:
        schema.validate({"name": "John", "address": {"street": "123 Main St"}})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"
        assert e.messages[0].index == ["address", "city"]


# LLM-generated content at query #27
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    field = Field()
    definitions['test_field'] = field
    reference = Reference(to='test_field', definitions=definitions)

    # Test with valid value
    assert reference.validate("valid_value") == "valid_value"

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with invalid value
    field.validate = lambda x: 1 / 0  # Force an error
    try:
        reference.validate("invalid_value")
        assert False, "Expected ValidationError"
    except ValidationError:
        pass


# LLM-generated content at query #28
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    schema = Schema(fields={"name": Field(), "age": Field()})
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test null input with allow_null=True
    schema = Schema(fields={"name": Field()}, allow_null=True)
    assert schema.validate(None) is None

    # Test null input with allow_null=False
    schema = Schema(fields={"name": Field()}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test non-dict input
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"

    # Test invalid key type
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate({123: "invalid key"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

    # Test missing required field
    schema = Schema(fields={"name": Field(), "age": Field()})
    try:
        schema.validate({"name": "John"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"

    # Test field validation error
    class CustomField(Field):
        def validate(self, value):
            if value == "invalid":
                raise self.validation_error("invalid")
            return value

    schema = Schema(fields={"name": CustomField()})
    try:
        schema.validate({"name": "invalid"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid"

    # Test default value
    class DefaultField(Field):
        def __init__(self, default="default", **kwargs):
            super().__init__(**kwargs)
            self._default = default

        def has_default(self):
            return True

        def get_default_value(self):
            return self._default

    schema = Schema(fields={"name": DefaultField()})
    assert schema.validate({}) == {"name": "default"}

    # Test read-only field
    schema = Schema(fields={"name": Field(read_only=True), "age": Field()})
    assert schema.validate({"age": 30}) == {"age": 30}


# LLM-generated content at query #29
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    field = Field()
    definitions['test_field'] = field
    reference = Reference(to='test_field', definitions=definitions)

    # Test with valid value
    with patch.object(field, 'validate', return_value='validated_value'):
        assert reference.validate('test_value') == 'validated_value'
        field.validate.assert_called_once_with('test_value')

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    with pytest.raises(ValidationError) as excinfo:
        reference.validate(None)
    assert excinfo.value.messages[0].code == 'null'

    # Test that target.validate is called
    with patch.object(field, 'validate', return_value='another_value'):
        assert reference.validate('another_test') == 'another_value'


# LLM-generated content at query #30
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    schema = Schema(fields={"name": Field(), "age": Field()})
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test with None when allow_null is True
    schema_with_null = Schema(fields={"name": Field()}, allow_null=True)
    assert schema_with_null.validate(None) is None

    # Test with None when allow_null is False
    schema_without_null = Schema(fields={"name": Field()}, allow_null=False)
    try:
        schema_without_null.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with non-dict input
    try:
        schema.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"

    # Test with non-string keys
    try:
        schema.validate({1: "value"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

    # Test with missing required field
    schema_required = Schema(fields={"name": Field(), "age": Field()})
    try:
        schema_required.validate({"name": "John"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"

    # Test with default value
    schema_default = Schema(fields={"name": Field(default="default_name")})
    assert schema_default.validate({}) == {"name": "default_name"}

    # Test with read-only field
    schema_readonly = Schema(fields={"name": Field(), "id": Field(read_only=True)})
    assert schema_readonly.validate({"name": "John"}) == {"name": "John"}

    # Test with nested validation error
    nested_schema = Schema(fields={"user": Schema(fields={"name": Field()})})
    try:
        nested_schema.validate({"user": {"invalid_key": "value"}})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"


# LLM-generated content at query #31
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field(allow_null=True)
    definitions["test_field"] = target_field
    reference = Reference(to="test_field", definitions=definitions, allow_null=True)

    # Test with None and allow_null=True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Reset allow_null for further tests
    reference.allow_null = True

    # Test with valid value
    with patch.object(target_field, 'validate', return_value="validated_value") as mock_validate:
        result = reference.validate("test_value")
        mock_validate.assert_called_once_with("test_value")
        assert result == "validated_value"

    # Test with invalid value
    mock_error = ValidationError(messages=[Message(text="error", code="error")])
    with patch.object(target_field, 'validate', side_effect=mock_error) as mock_validate:
        try:
            reference.validate("invalid_value")
            assert False, "Expected ValidationError"
        except ValidationError as e:
            assert len(e.messages) == 1
            assert e.messages[0].code == "error"
            mock_validate.assert_called_once_with("invalid_value")


# LLM-generated content at query #32
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    field = Field()
    definitions['test_field'] = field
    reference = Reference(to='test_field', definitions=definitions)

    # Test with valid value
    assert reference.validate("valid_value") == "valid_value"

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with invalid value
    field.validate = lambda x: 1/0 if x == "invalid" else x
    try:
        reference.validate("invalid")
        assert False, "Expected ValidationError"
    except ValidationError:
        pass


# LLM-generated content at query #33
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    schema = Schema(fields={"name": Field(), "age": Field()})
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test null input with allow_null=True
    schema = Schema(fields={"name": Field()}, allow_null=True)
    assert schema.validate(None) is None

    # Test null input with allow_null=False
    schema = Schema(fields={"name": Field()}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test non-dict input
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"

    # Test invalid key type
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate({123: "invalid key"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

    # Test missing required field
    schema = Schema(fields={"name": Field(), "age": Field()})
    try:
        schema.validate({"name": "John"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"

    # Test field with default value
    schema = Schema(fields={"name": Field(), "age": Field(default=25)})
    assert schema.validate({"name": "John"}) == {"name": "John", "age": 25}

    # Test read-only field
    schema = Schema(fields={"name": Field(), "id": Field(read_only=True)})
    assert schema.validate({"name": "John"}) == {"name": "John"}

    # Test nested validation error
    inner_schema = Schema(fields={"street": Field(), "city": Field()})
    schema = Schema(fields={"name": Field(), "address": inner_schema})
    try:
        schema.validate({"name": "John", "address": {"street": "Main St"}})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"
        assert e.messages[0].index == ["address", "city"]


# LLM-generated content at query #34
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    schema = Schema(fields={"name": Field(), "age": Field()})
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test null input with allow_null=True
    schema = Schema(fields={"name": Field()}, allow_null=True)
    assert schema.validate(None) is None

    # Test null input with allow_null=False
    schema = Schema(fields={"name": Field()}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test non-dict input
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"

    # Test non-string key
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate({1: "value"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

    # Test missing required field
    schema = Schema(fields={"name": Field(), "age": Field()})
    try:
        schema.validate({"name": "John"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"

    # Test field with default value
    schema = Schema(fields={"name": Field(), "age": Field(default=25)})
    assert schema.validate({"name": "John"}) == {"name": "John", "age": 25}

    # Test read-only field
    schema = Schema(fields={"name": Field(), "id": Field(read_only=True)})
    assert schema.validate({"name": "John"}) == {"name": "John"}

    # Test nested validation error
    inner_schema = Schema(fields={"street": Field(), "city": Field()})
    schema = Schema(fields={"name": Field(), "address": inner_schema})
    try:
        schema.validate({"name": "John", "address": {"street": "123 Main St"}})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"
        assert e.messages[0].index == ["address", "city"]


# LLM-generated content at query #35
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    schema = Schema(fields={"name": Field(), "age": Field()})
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test null input with allow_null=True
    schema = Schema(fields={"name": Field()}, allow_null=True)
    assert schema.validate(None) is None

    # Test null input with allow_null=False
    schema = Schema(fields={"name": Field()}, allow_null=False)
    with pytest.raises(ValidationError):
        schema.validate(None)

    # Test non-dict input
    schema = Schema(fields={"name": Field()})
    with pytest.raises(ValidationError):
        schema.validate("not a dict")

    # Test invalid key type
    schema = Schema(fields={"name": Field()})
    with pytest.raises(ValidationError):
        schema.validate({123: "invalid key"})

    # Test missing required field
    schema = Schema(fields={"name": Field(), "age": Field()})
    with pytest.raises(ValidationError):
        schema.validate({"name": "John"})

    # Test field with default value
    schema = Schema(fields={"name": Field(), "age": Field(default=25)})
    assert schema.validate({"name": "John"}) == {"name": "John", "age": 25}

    # Test field validation error
    schema = Schema(fields={"name": Field(), "age": Field()})
    with pytest.raises(ValidationError):
        schema.validate({"name": "John", "age": "invalid age"})

    # Test read-only field
    schema = Schema(fields={"name": Field(), "id": Field(read_only=True)})
    assert schema.validate({"name": "John"}) == {"name": "John"}


# LLM-generated content at query #36
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    field = Field()
    definitions['test_field'] = field
    reference = Reference(to='test_field', definitions=definitions)

    # Test with valid value
    field.validate = lambda x: x  # Mock validate to return input
    assert reference.validate("valid_value") == "valid_value"

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with target's validate raising error
    field.validate = lambda x: 1 / 0  # Mock validate to raise error
    try:
        reference.validate("any_value")
        assert False, "Expected error from target.validate"
    except ZeroDivisionError:
        pass


# LLM-generated content at query #37
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    field = Field()
    definitions['test_field'] = field
    reference = Reference(to='test_field', definitions=definitions)

    # Test with valid value
    with pytest.raises(NotImplementedError):
        reference.validate("valid_value")

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    with pytest.raises(ValidationError):
        reference.validate(None)


# LLM-generated content at query #38
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field()
    definitions['test_field'] = target_field
    reference = Reference(to='test_field', definitions=definitions)

    # Test valid input
    assert reference.validate("test_value") is None

    # Test null input with allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test null input with allow_null=False
    reference.allow_null = False
    with pytest.raises(ValidationError) as exc_info:
        reference.validate(None)
    assert exc_info.value.messages[0].code == "null"

    # Test target field validation error
    target_field_with_error = Field()
    target_field_with_error.validate = lambda x: (_ for _ in ()).throw(ValidationError("Test error"))
    definitions['error_field'] = target_field_with_error
    error_reference = Reference(to='error_field', definitions=definitions)
    with pytest.raises(ValidationError):
        error_reference.validate("test_value")


# LLM-generated content at query #39
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    schema = Schema(fields={"name": Field(), "age": Field()})
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test with None and allow_null=True
    schema = Schema(fields={"name": Field()}, allow_null=True)
    assert schema.validate(None) is None

    # Test with None and allow_null=False
    schema = Schema(fields={"name": Field()}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with non-dict input
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"

    # Test with non-string keys
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate({1: "value"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

    # Test with missing required field
    schema = Schema(fields={"name": Field(), "age": Field()})
    try:
        schema.validate({"name": "John"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"

    # Test with field having default value
    schema = Schema(fields={"name": Field(), "age": Field(default=25)})
    assert schema.validate({"name": "John"}) == {"name": "John", "age": 25}

    # Test with read-only field
    schema = Schema(fields={"name": Field(), "id": Field(read_only=True)})
    assert schema.validate({"name": "John"}) == {"name": "John"}

    # Test with nested validation errors
    child_schema = Field()
    child_schema.validate = lambda x: (None, ValidationError(messages=[Message(text="Invalid", code="invalid")]))
    schema = Schema(fields={"name": child_schema})
    try:
        schema.validate({"name": "invalid"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid"


# LLM-generated content at query #40
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    definitions["test_schema"] = Schema(fields={"name": Field()})

    reference = Reference(to="test_schema", definitions=definitions)

    # Test valid input
    valid_data = {"name": "test"}
    assert reference.validate(valid_data) == {"name": "test"}

    # Test null input with allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test null input with allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError for null input"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test invalid input (should raise ValidationError from target schema)
    try:
        reference.validate({"invalid_key": 123})
        assert False, "Expected ValidationError for invalid key"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"


# LLM-generated content at query #41
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    schema = Schema(fields={"name": Field()})
    definitions["test_schema"] = schema
    reference = Reference(to="test_schema", definitions=definitions)

    # Test valid input
    valid_input = {"name": "test"}
    assert reference.validate(valid_input) == valid_input

    # Test null input with allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test null input with allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test invalid input (passed to target schema)
    try:
        reference.validate("invalid")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"


# LLM-generated content at query #42
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    field = Field()
    definitions['test_field'] = field
    reference = Reference(to='test_field', definitions=definitions)

    # Test with valid value
    with patch.object(field, 'validate', return_value='validated_value'):
        assert reference.validate('test_value') == 'validated_value'
        field.validate.assert_called_once_with('test_value')

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    with pytest.raises(ValidationError) as exc_info:
        reference.validate(None)
    assert exc_info.value.messages[0].code == 'null'

    # Test that validate calls target's validate
    with patch.object(reference.target, 'validate', return_value='another_value'):
        assert reference.validate('another_test') == 'another_value'
        reference.target.validate.assert_called_once_with('another_test')


# LLM-generated content at query #43
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field()
    definitions['test_ref'] = target_field
    ref = Reference(to='test_ref', definitions=definitions)

    # Test with valid value
    assert ref.validate("valid_value") == "valid_value"

    # Test with None and allow_null=True
    ref.allow_null = True
    assert ref.validate(None) is None

    # Test with None and allow_null=False
    ref.allow_null = False
    try:
        ref.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test that target's validate is called
    target_field.validate = lambda x: x.upper()
    assert ref.validate("test") == "TEST"


# LLM-generated content at query #44
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    schema = Schema(fields={"name": Field(), "age": Field()})
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test None input with allow_null=True
    schema = Schema(fields={"name": Field()}, allow_null=True)
    assert schema.validate(None) is None

    # Test None input with allow_null=False
    schema = Schema(fields={"name": Field()}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test non-dict input
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"

    # Test invalid key type
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate({1: "invalid key"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

    # Test missing required field
    schema = Schema(fields={"name": Field(), "age": Field()})
    try:
        schema.validate({"name": "John"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"

    # Test field with default value
    schema = Schema(fields={"name": Field(), "age": Field(default=25)})
    assert schema.validate({"name": "John"}) == {"name": "John", "age": 25}

    # Test read-only field
    schema = Schema(fields={"name": Field(), "id": Field(read_only=True)})
    assert schema.validate({"name": "John"}) == {"name": "John"}

    # Test nested validation error
    inner_schema = Schema(fields={"street": Field(), "city": Field()})
    schema = Schema(fields={"name": Field(), "address": inner_schema})
    try:
        schema.validate({"name": "John", "address": {"street": "123 Main St"}})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"
        assert e.messages[0].index == ["address", "city"]


# LLM-generated content at query #45
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field(allow_null=True)
    definitions['test_field'] = target_field
    reference = Reference(to='test_field', definitions=definitions, allow_null=True)

    # Test with valid value
    assert reference.validate("valid_value") == "valid_value"

    # Test with None and allow_null=True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with invalid value (should raise error from target field)
    target_field.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"


# LLM-generated content at query #46
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    field = Field()
    definitions['test_field'] = field
    reference = Reference(to='test_field', definitions=definitions)

    # Test with valid value
    assert reference.validate('valid_value') == 'valid_value'

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == 'null'

    # Test with invalid value
    with pytest.raises(ValidationError):
        reference.validate('invalid_value')


# LLM-generated content at query #47
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field()
    definitions["test_field"] = target_field
    reference = Reference(to="test_field", definitions=definitions)

    # Test with valid value
    assert reference.validate("valid_value") == "valid_value"

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with target field validation error
    target_field_with_error = Field()
    target_field_with_error.validate = lambda x: 1/0  # Force an error
    definitions["error_field"] = target_field_with_error
    error_reference = Reference(to="error_field", definitions=definitions)
    try:
        error_reference.validate("any_value")
        assert False, "Expected ValidationError"
    except ValidationError:
        pass


# LLM-generated content at query #48
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field()
    definitions["test_field"] = target_field
    reference = Reference(to="test_field", definitions=definitions)

    # Test valid value
    assert reference.validate("valid_value") == "valid_value"

    # Test null value with allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test null value with allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test target field validation
    target_field_with_validation = Field()
    target_field_with_validation.validate = lambda x: x if x == "expected" else None
    definitions["test_field_validation"] = target_field_with_validation
    reference_with_validation = Reference(to="test_field_validation", definitions=definitions)

    assert reference_with_validation.validate("expected") == "expected"
    try:
        reference_with_validation.validate("unexpected")
        assert False, "Expected ValidationError"
    except ValidationError:
        pass


# LLM-generated content at query #49
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field()
    definitions["test_field"] = target_field
    reference = Reference(to="test_field", definitions=definitions)

    # Test with valid value
    assert reference.validate("valid_value") == "valid_value"

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with target field validation error
    target_field_with_error = Field()
    target_field_with_error.validate = lambda x: 1/0  # Force error
    definitions["error_field"] = target_field_with_error
    error_reference = Reference(to="error_field", definitions=definitions)
    try:
        error_reference.validate("any_value")
        assert False, "Expected ValidationError"
    except ValidationError:
        pass


# LLM-generated content at query #50
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    definitions['test_schema'] = Schema(fields={'name': Field()})
    reference = Reference(to='test_schema', definitions=definitions)

    # Test valid input
    assert reference.validate({'name': 'test'}) == {'name': 'test'}

    # Test null input with allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test null input with allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == 'null'

    # Test invalid input
    try:
        reference.validate('invalid')
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == 'type'


# LLM-generated content at query #51
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field()
    definitions['test_field'] = target_field
    reference = Reference(to='test_field', definitions=definitions)

    # Test with valid value
    assert reference.validate("valid_value") is None

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with target field validation error
    target_field_with_error = Field()
    target_field_with_error.validate = lambda x: (_ for _ in ()).throw(ValidationError(messages=[Message(text="error", code="error")]))
    definitions['error_field'] = target_field_with_error
    error_reference = Reference(to='error_field', definitions=definitions)
    try:
        error_reference.validate("any_value")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "error"


# LLM-generated content at query #52
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field()
    definitions['test_field'] = target_field
    reference = Reference(to='test_field', definitions=definitions)

    # Test with valid value
    valid_value = "valid"
    assert reference.validate(valid_value) == valid_value

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with target field validation error
    target_field_with_error = Field()
    target_field_with_error.validate = lambda x: 1 / 0  # Force error
    definitions['error_field'] = target_field_with_error
    error_reference = Reference(to='error_field', definitions=definitions)
    try:
        error_reference.validate("any_value")
        assert False, "Expected ValidationError"
    except ValidationError:
        pass


# LLM-generated content at query #53
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field()
    definitions["test_field"] = target_field
    reference = Reference(to="test_field", definitions=definitions)

    # Test with valid value
    assert reference.validate("valid_value") == "valid_value"

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with target field validation error
    target_field_with_error = Field()
    target_field_with_error.validate = lambda value: 1 / 0  # Force error
    definitions["error_field"] = target_field_with_error
    error_reference = Reference(to="error_field", definitions=definitions)
    try:
        error_reference.validate("any_value")
        assert False, "Expected ValidationError"
    except ValidationError:
        pass


# LLM-generated content at query #54
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field(allow_null=True)
    definitions["test_field"] = target_field
    reference = Reference(to="test_field", definitions=definitions, allow_null=True)

    # Test with valid value
    assert reference.validate("valid_value") == "valid_value"

    # Test with None when allow_null is True
    assert reference.validate(None) is None

    # Test with None when allow_null is False
    reference_no_null = Reference(to="test_field", definitions=definitions, allow_null=False)
    try:
        reference_no_null.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with target field validation error
    target_field_with_validation = Field(allow_null=False)
    definitions["test_field_error"] = target_field_with_validation
    reference_error = Reference(to="test_field_error", definitions=definitions, allow_null=False)
    try:
        reference_error.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"


# LLM-generated content at query #55
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field()
    definitions['target'] = target_field
    reference = Reference(to='target', definitions=definitions)

    # Test with valid value
    assert reference.validate("valid_value") == "valid_value"

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with target field raising ValidationError
    target_field.validate = lambda x: (_ for _ in ()).throw(ValidationError(messages=[Message(text="test", code="test")]))
    try:
        reference.validate("invalid_value")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "test"


# LLM-generated content at query #56
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field()
    definitions["test_field"] = target_field
    reference = Reference(to="test_field", definitions=definitions)

    # Test with valid value
    assert reference.validate("valid_value") == "valid_value"

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with target field validation error
    target_field_with_error = Field()
    target_field_with_error.validate = lambda x: 1/0  # Force an error
    definitions["error_field"] = target_field_with_error
    error_reference = Reference(to="error_field", definitions=definitions)
    try:
        error_reference.validate("any_value")
        assert False, "Expected ValidationError"
    except ValidationError:
        pass


# LLM-generated content at query #57
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    schema = Schema(fields={"name": Field(), "age": Field()})
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test null input with allow_null=True
    schema = Schema(fields={"name": Field()}, allow_null=True)
    assert schema.validate(None) is None

    # Test null input with allow_null=False
    schema = Schema(fields={"name": Field()}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test non-dict input
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"

    # Test invalid key type
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate({123: "invalid key"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

    # Test missing required field
    schema = Schema(fields={"name": Field(), "age": Field()})
    try:
        schema.validate({"name": "John"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"

    # Test field with default value
    schema = Schema(fields={"name": Field(default="default_name")})
    assert schema.validate({}) == {"name": "default_name"}

    # Test read-only field
    schema = Schema(fields={"name": Field(read_only=True)})
    assert schema.validate({}) == {}

    # Test nested field validation
    inner_schema = Schema(fields={"city": Field()})
    schema = Schema(fields={"name": Field(), "address": inner_schema})
    try:
        schema.validate({"name": "John", "address": {"city": 123}})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"


# LLM-generated content at query #58
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field()
    definitions['test_ref'] = target_field
    reference = Reference(to='test_ref', definitions=definitions)

    # Test with valid value
    target_field.validate = lambda x: x
    assert reference.validate("valid_value") == "valid_value"

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with target validation error
    target_field.validate = lambda x: 1/0  # Force error
    try:
        reference.validate("invalid_value")
        assert False, "Expected ValidationError"
    except ValidationError:
        pass


# LLM-generated content at query #59
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field()
    definitions['test_ref'] = target_field
    ref = Reference(to='test_ref', definitions=definitions)

    # Test with valid value
    target_field.validate = lambda x: x
    assert ref.validate("valid_value") == "valid_value"

    # Test with None and allow_null=True
    ref.allow_null = True
    assert ref.validate(None) is None

    # Test with None and allow_null=False
    ref.allow_null = False
    try:
        ref.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with target validation error
    target_field.validate = lambda x: 1 / 0  # Force error
    try:
        ref.validate("any_value")
        assert False, "Expected ValidationError"
    except ValidationError:
        pass


# LLM-generated content at query #60
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    schema = Schema(fields={"name": Field(), "age": Field()})
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test null input with allow_null=True
    schema = Schema(fields={"name": Field()}, allow_null=True)
    assert schema.validate(None) is None

    # Test null input with allow_null=False
    schema = Schema(fields={"name": Field()}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test non-dict input
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"

    # Test invalid key type
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate({1: "invalid key"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

    # Test required field missing
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate({})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"

    # Test field with default value
    schema = Schema(fields={"name": Field(default="default_name")})
    assert schema.validate({}) == {"name": "default_name"}

    # Test read-only field
    schema = Schema(fields={"name": Field(read_only=True)})
    assert schema.validate({}) == {}

    # Test child field validation error
    child_field = Field()
    child_field.validate = lambda x: (None, ValidationError(messages=[Message(text="error", code="error")]))
    schema = Schema(fields={"name": child_field})
    try:
        schema.validate({"name": "test"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "error"


# LLM-generated content at query #61
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field()
    definitions["test_field"] = target_field
    reference = Reference(to="test_field", definitions=definitions)

    # Test with valid value
    assert reference.validate("valid_value") == "valid_value"

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with target field validation error
    target_field_with_error = Field()
    target_field_with_error.validate = lambda value: 1 / 0  # Force error
    definitions["error_field"] = target_field_with_error
    error_reference = Reference(to="error_field", definitions=definitions)
    try:
        error_reference.validate("any_value")
        assert False, "Expected ValidationError"
    except ValidationError:
        pass


# LLM-generated content at query #62
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    schema = Schema(fields={"name": Field(), "age": Field()})
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test null input with allow_null=True
    schema = Schema(fields={"name": Field()}, allow_null=True)
    assert schema.validate(None) is None

    # Test null input with allow_null=False
    schema = Schema(fields={"name": Field()}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test non-dict input
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"

    # Test invalid key type
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate({123: "invalid key"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

    # Test missing required field
    schema = Schema(fields={"name": Field(), "age": Field()})
    try:
        schema.validate({"name": "John"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"

    # Test field validation error
    class CustomField(Field):
        def validate(self, value):
            if value == "invalid":
                raise self.validation_error("invalid")
            return value

    schema = Schema(fields={"name": CustomField()})
    try:
        schema.validate({"name": "invalid"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid"

    # Test default value
    class FieldWithDefault(Field):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._default = "default"

        def has_default(self):
            return True

        def get_default_value(self):
            return self._default

    schema = Schema(fields={"name": FieldWithDefault()})
    assert schema.validate({}) == {"name": "default"}

    # Test read-only field
    schema = Schema(fields={"name": Field(read_only=True), "age": Field()})
    assert schema.validate({"age": 30}) == {"age": 30}


# LLM-generated content at query #63
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    schema = Schema(fields={"name": Field(), "age": Field()})
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test with allow_null=True
    schema_with_null = Schema(fields={"name": Field()}, allow_null=True)
    assert schema_with_null.validate(None) is None

    # Test with allow_null=False (default)
    schema_no_null = Schema(fields={"name": Field()})
    try:
        schema_no_null.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test non-dict input
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"

    # Test invalid key type
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate({123: "invalid key"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

    # Test required field missing
    schema = Schema(fields={"name": Field(), "age": Field()})
    try:
        schema.validate({"name": "John"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert any(msg.code == "required" for msg in e.messages)

    # Test with default value
    schema = Schema(fields={"name": Field(), "age": Field(default=25)})
    assert schema.validate({"name": "John"}) == {"name": "John", "age": 25}

    # Test with read_only field
    schema = Schema(fields={"name": Field(), "id": Field(read_only=True)})
    assert schema.validate({"name": "John"}) == {"name": "John"}
    assert schema.validate({"name": "John", "id": 123}) == {"name": "John"}

    # Test child field validation error
    class PositiveField(Field):
        def validate(self, value):
            if value <= 0:
                raise self.validation_error("positive", "Must be positive")
            return value

    schema = Schema(fields={"age": PositiveField()})
    try:
        schema.validate({"age": -5})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "positive"


# LLM-generated content at query #64
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field()
    definitions['test_field'] = target_field

    reference = Reference(to='test_field', definitions=definitions)

    # Test with valid value
    assert reference.validate("valid_value") == "valid_value"

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with target field validation error
    target_field_with_error = Field()
    target_field_with_error.validate = lambda value: 1/0  # Force error
    definitions['error_field'] = target_field_with_error

    error_reference = Reference(to='error_field', definitions=definitions)
    try:
        error_reference.validate("any_value")
        assert False, "Expected ValidationError"
    except ValidationError:
        pass


# LLM-generated content at query #65
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    field = Field()
    definitions['test_field'] = field
    reference = Reference(to='test_field', definitions=definitions)

    # Test with valid value
    assert reference.validate("valid_value") == "valid_value"

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with invalid value
    field.validate = lambda x: 1/0  # Force an error
    try:
        reference.validate("invalid_value")
        assert False, "Expected ValidationError"
    except ValidationError:
        pass


# LLM-generated content at query #66
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field()
    definitions['test_field'] = target_field
    reference = Reference(to='test_field', definitions=definitions)

    # Test with valid input
    assert reference.validate("valid_value") == "valid_value"

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with target field validation error
    target_field_with_error = Field()
    target_field_with_error.validate = lambda x: 1/0  # Force an error
    definitions['error_field'] = target_field_with_error
    error_reference = Reference(to='error_field', definitions=definitions)
    try:
        error_reference.validate("any_value")
        assert False, "Expected ValidationError"
    except ValidationError:
        pass


# LLM-generated content at query #67
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field()
    definitions['test_ref'] = target_field
    reference = Reference(to='test_ref', definitions=definitions)

    # Test with valid value
    assert reference.validate("valid_value") == target_field.validate("valid_value")

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test that validate calls target's validate
    with patch.object(target_field, 'validate') as mock_validate:
        mock_validate.return_value = "mocked_value"
        assert reference.validate("test") == "mocked_value"
        mock_validate.assert_called_once_with("test")


# LLM-generated content at query #68
#--------------------------

```python
def test_Schema_validate():
    # Test 1: Valid input with all required fields
    fields = {
        "name": Field(required=True),
        "age": Field(required=True),
    }
    schema = Schema(fields=fields)
    input_data = {"name": "John", "age": 30}
    result = schema.validate(input_data)
    assert result == input_data

    # Test 2: Valid input with optional fields
    fields = {
        "name": Field(required=True),
        "age": Field(required=False),
    }
    schema = Schema(fields=fields)
    input_data = {"name": "John"}
    result = schema.validate(input_data)
    assert result == input_data

    # Test 3: Invalid input with missing required field
    fields = {
        "name": Field(required=True),
        "age": Field(required=True),
    }
    schema = Schema(fields=fields)
    input_data = {"name": "John"}
    with pytest.raises(ValidationError) as exc_info:
        schema.validate(input_data)
    assert exc_info.value.messages[0].code == "required"
    assert exc_info.value.messages[0].index == ["age"]

    # Test 4: Invalid input with non-string key
    fields = {
        "name": Field(required=True),
    }
    schema = Schema(fields=fields)
    input_data = {123: "John"}
    with pytest.raises(ValidationError) as exc_info:
        schema.validate(input_data)
    assert exc_info.value.messages[0].code == "invalid_key"
    assert exc_info.value.messages[0].index == [123]

    # Test 5: Invalid input with non-dict type
    fields = {
        "name": Field(required=True),
    }
    schema = Schema(fields=fields)
    input_data = "not a dict"
    with pytest.raises(ValidationError) as exc_info:
        schema.validate(input_data)
    assert exc_info.value.messages[0].code == "type"

    # Test 6: Valid input with null allowed
    fields = {
        "name": Field(required=True),
    }
    schema = Schema(fields=fields, allow_null=True)
    input_data = None
    result = schema.validate(input_data)
    assert result is None

    # Test 7: Invalid input with null not allowed
    fields = {
        "name": Field(required=True),
    }
    schema = Schema(fields=fields, allow_null=False)
    input_data = None
    with pytest.raises(ValidationError) as exc_info:
        schema.validate(input_data)
    assert exc_info.value.messages[0].code == "null"

    # Test 8: Valid input with default values
    fields = {
        "name": Field(required=True),
        "age": Field(required=False, default=25),
    }
    schema = Schema(fields=fields)
    input_data = {"name": "John"}
    result = schema.validate(input_data)
    assert result == {"name": "John", "age": 25}

    # Test 9: Valid input with nested schema
    nested_fields = {
        "street": Field(required=True),
        "city": Field(required=True),
    }
    fields = {
        "name": Field(required=True),
        "address": Schema(fields=nested_fields, required=True),
    }
    schema = Schema(fields=fields)
    input_data = {
        "name": "John",
        "address": {"street": "123 Main St", "city": "New York"}
    }
    result = schema.validate(input_data)
    assert result == input_data

    # Test 10: Invalid input with nested schema missing required field
    nested_fields = {
        "street": Field(required=True),
        "city": Field(required=True),
    }
    fields = {
        "name": Field(required=True),
        "address": Schema(fields=nested_fields, required=True),
    }
    schema = Schema(fields=fields)
    input_data = {
        "name": "John",
        "address": {"street": "123 Main St"}
    }
    with pytest.raises(ValidationError) as exc_info:
        schema.validate(input_data)
    assert exc_info.value.messages[0].code == "required"
    assert exc_info.value.messages[0].index == ["address", "city"]


# LLM-generated content at query #69
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_schema = Schema(fields={"name": Field()})
    definitions["test_schema"] = target_schema
    reference = Reference(to="test_schema", definitions=definitions)

    # Test valid input
    valid_input = {"name": "test"}
    assert reference.validate(valid_input) == {"name": "test"}

    # Test null input with allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test null input with allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test invalid input (passed to target schema)
    try:
        reference.validate("invalid")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"


# LLM-generated content at query #70
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field(allow_null=True)
    definitions['test_field'] = target_field
    reference = Reference(to='test_field', definitions=definitions, allow_null=True)

    # Test with valid value
    assert reference.validate("valid_value") == "valid_value"

    # Test with None and allow_null=True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference_no_null = Reference(to='test_field', definitions=definitions, allow_null=False)
    try:
        reference_no_null.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with invalid value (should raise ValidationError from target field)
    target_field_with_validation = Field(allow_null=False)
    target_field_with_validation.validate = lambda x: "valid" if x == "valid" else (1/0)
    definitions['strict_field'] = target_field_with_validation
    strict_reference = Reference(to='strict_field', definitions=definitions)

    try:
        strict_reference.validate("invalid_value")
        assert False, "Expected ValidationError"
    except ValidationError:
        pass


# LLM-generated content at query #71
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field()
    definitions["test_field"] = target_field
    reference = Reference(to="test_field", definitions=definitions)

    # Test with valid value
    assert reference.validate("valid_value") == "valid_value"

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with target field raising ValidationError
    target_field.validate = lambda value: (_ for _ in ()).throw(ValidationError(messages=[Message(text="test_error", code="test_code")]))
    try:
        reference.validate("invalid_value")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "test_code"


# LLM-generated content at query #72
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    field = Field()
    definitions['test_field'] = field
    reference = Reference(to='test_field', definitions=definitions)

    # Test with valid value
    assert reference.validate('valid_value') == 'valid_value'

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with invalid value
    field.validate = lambda x: 1/0 if x == 'invalid' else x  # Force error
    try:
        reference.validate('invalid')
        assert False, "Expected ValidationError"
    except ValidationError:
        pass


# LLM-generated content at query #73
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field(allow_null=True)
    definitions['test_field'] = target_field
    reference = Reference(to='test_field', definitions=definitions)

    # Test with valid value
    assert reference.validate("valid_value") == "valid_value"

    # Test with None and allow_null=True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with target field validation error
    target_field.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"


# LLM-generated content at query #74
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    schema = Schema(fields={"name": Field(), "age": Field()})
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test None input with allow_null=True
    schema = Schema(fields={"name": Field()}, allow_null=True)
    assert schema.validate(None) is None

    # Test None input with allow_null=False
    schema = Schema(fields={"name": Field()}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test non-dict input
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"

    # Test non-string key
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate({123: "value"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

    # Test missing required field
    schema = Schema(fields={"name": Field(), "age": Field()})
    try:
        schema.validate({"name": "John"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"

    # Test field with default value
    schema = Schema(fields={"name": Field(), "age": Field(default=25)})
    assert schema.validate({"name": "John"}) == {"name": "John", "age": 25}

    # Test read-only field
    schema = Schema(fields={"name": Field(), "id": Field(read_only=True)})
    assert schema.validate({"name": "John"}) == {"name": "John"}

    # Test nested validation error
    child_field = Field()
    child_field.validate = lambda x: (None, ValidationError(messages=[Message(text="Invalid", code="invalid")]))
    schema = Schema(fields={"name": child_field})
    try:
        schema.validate({"name": "invalid_value"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid"


# LLM-generated content at query #75
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    schema = Schema(fields={"name": Field(), "age": Field()})
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test null input with allow_null=True
    schema = Schema(fields={"name": Field()}, allow_null=True)
    assert schema.validate(None) is None

    # Test null input with allow_null=False
    schema = Schema(fields={"name": Field()}, allow_null=False)
    with pytest.raises(ValidationError):
        schema.validate(None)

    # Test non-dict input
    schema = Schema(fields={"name": Field()})
    with pytest.raises(ValidationError):
        schema.validate("not a dict")

    # Test invalid key type
    schema = Schema(fields={"name": Field()})
    with pytest.raises(ValidationError):
        schema.validate({1: "invalid key"})

    # Test missing required field
    schema = Schema(fields={"name": Field(), "age": Field()})
    with pytest.raises(ValidationError):
        schema.validate({"name": "John"})

    # Test field with default value
    schema = Schema(fields={"name": Field(), "age": Field(default=25)})
    assert schema.validate({"name": "John"}) == {"name": "John", "age": 25}

    # Test read-only field
    schema = Schema(fields={"name": Field(), "id": Field(read_only=True)})
    assert schema.validate({"name": "John"}) == {"name": "John"}
    assert schema.validate({"name": "John", "id": 123}) == {"name": "John"}

    # Test nested validation error
    child_schema = Field()
    child_schema.validate = lambda x: (None, ValidationError(messages=[Message(text="Invalid", code="invalid")]))
    schema = Schema(fields={"name": child_schema})
    with pytest.raises(ValidationError):
        schema.validate({"name": "invalid"})


# LLM-generated content at query #76
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_schema = Schema(fields={"name": Field(allow_null=True)})
    definitions["test_schema"] = target_schema
    reference = Reference(to="test_schema", definitions=definitions)

    # Test valid input
    valid_input = {"name": "test"}
    assert reference.validate(valid_input) == {"name": "test"}

    # Test null input with allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test null input with allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test invalid input (passed to target schema)
    try:
        reference.validate("invalid")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"


# LLM-generated content at query #77
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field()
    definitions['test_ref'] = target_field
    ref = Reference(to='test_ref', definitions=definitions)

    # Test with valid value
    assert ref.validate("valid_value") == target_field.validate("valid_value")

    # Test with None and allow_null=True
    ref.allow_null = True
    assert ref.validate(None) is None

    # Test with None and allow_null=False
    ref.allow_null = False
    try:
        ref.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test that validate calls target's validate
    with pytest.raises(ValidationError):
        target_field.validate = lambda x: (_ for _ in ()).throw(ValidationError("test error"))
        ref.validate("invalid_value")


# LLM-generated content at query #78
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    schema = Schema(fields={"name": Field(), "age": Field()})
    definitions["person"] = schema
    reference = Reference(to="person", definitions=definitions)

    # Test valid input
    valid_input = {"name": "John", "age": 30}
    assert reference.validate(valid_input) == valid_input

    # Test null input with allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test null input with allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test invalid input (not a dict)
    try:
        reference.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"

    # Test invalid input (invalid key type)
    try:
        reference.validate({1: "invalid key"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

    # Test missing required field
    try:
        reference.validate({"name": "John"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"


# LLM-generated content at query #79
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    fields = {
        "name": Field(allow_null=False),
        "age": Field(allow_null=True),
    }
    schema = Schema(fields=fields)
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test with None and allow_null=True
    schema_with_null = Schema(fields=fields, allow_null=True)
    assert schema_with_null.validate(None) is None

    # Test with None and allow_null=False
    with pytest.raises(ValidationError) as exc_info:
        schema.validate(None)
    assert exc_info.value.messages[0].code == "null"

    # Test with non-dict input
    with pytest.raises(ValidationError) as exc_info:
        schema.validate("not a dict")
    assert exc_info.value.messages[0].code == "type"

    # Test with non-string keys
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({1: "value"})
    assert exc_info.value.messages[0].code == "invalid_key"

    # Test with missing required field
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({"age": 30})
    assert exc_info.value.messages[0].code == "required"

    # Test with child field validation error
    fields_with_child_validation = {
        "name": Field(allow_null=False),
        "age": Field(allow_null=False, min_value=0),
    }
    schema_with_child_validation = Schema(fields=fields_with_child_validation)
    with pytest.raises(ValidationError) as exc_info:
        schema_with_child_validation.validate({"name": "John", "age": -1})
    assert exc_info.value.messages[0].code == "min_value"

    # Test with default value
    fields_with_default = {
        "name": Field(allow_null=False),
        "age": Field(allow_null=True, default=0),
    }
    schema_with_default = Schema(fields=fields_with_default)
    assert schema_with_default.validate({"name": "John"}) == {"name": "John", "age": 0}

    # Test with read-only field
    fields_with_read_only = {
        "name": Field(allow_null=False),
        "id": Field(allow_null=False, read_only=True),
    }
    schema_with_read_only = Schema(fields=fields_with_read_only)
    assert schema_with_read_only.validate({"name": "John"}) == {"name": "John"}


# LLM-generated content at query #80
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field()
    definitions["test_field"] = target_field

    reference = Reference(to="test_field", definitions=definitions)

    # Test with valid value
    assert reference.validate("valid_value") == "valid_value"

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with target field validation error
    target_field_with_error = Field()
    target_field_with_error.validate = lambda value: 1 / 0  # Force error
    definitions["error_field"] = target_field_with_error

    error_reference = Reference(to="error_field", definitions=definitions)
    try:
        error_reference.validate("any_value")
        assert False, "Expected ValidationError"
    except ValidationError:
        pass


# LLM-generated content at query #81
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    schema = Schema(fields={"name": Field(), "age": Field()})
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test null input with allow_null=True
    schema = Schema(fields={"name": Field()}, allow_null=True)
    assert schema.validate(None) is None

    # Test null input with allow_null=False
    schema = Schema(fields={"name": Field()}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test non-dict input
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"

    # Test invalid key type
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate({123: "invalid key"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

    # Test required field missing
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate({})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"

    # Test field with default value
    schema = Schema(fields={"name": Field(default="default_name")})
    assert schema.validate({}) == {"name": "default_name"}

    # Test read-only field
    schema = Schema(fields={"name": Field(read_only=True)})
    assert schema.validate({}) == {}

    # Test nested validation error
    child_schema = Field()
    child_schema.validate = lambda x: (None, ValidationError(messages=[Message(text="child error", code="child_error")]))
    schema = Schema(fields={"child": child_schema})
    try:
        schema.validate({"child": "invalid"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "child_error"
        assert e.messages[0].index == ["child"]


# LLM-generated content at query #82
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    field = Field()
    definitions['test_field'] = field
    reference = Reference(to='test_field', definitions=definitions)

    # Test with valid value
    assert reference.validate('valid_value') == 'valid_value'

    # Test with None when allow_null is True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None when allow_null is False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with invalid value
    field.validate = lambda x: 1/0 if x == 'invalid' else x
    try:
        reference.validate('invalid')
        assert False, "Expected ValidationError"
    except ValidationError:
        pass


# LLM-generated content at query #83
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    field = Field(allow_null=False)
    definitions['test_field'] = field
    reference = Reference(to='test_field', definitions=definitions)

    # Test valid value
    assert reference.validate("valid_value") == "valid_value"

    # Test null value with allow_null=False
    with pytest.raises(ValidationError):
        reference.validate(None)

    # Test null value with allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test invalid value (assuming field.validate raises ValidationError for invalid values)
    with pytest.raises(ValidationError):
        reference.validate("invalid_value")


# LLM-generated content at query #84
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field()
    definitions['test_field'] = target_field
    reference = Reference(to='test_field', definitions=definitions)

    # Test valid input
    assert reference.validate("valid_input") == "valid_input"

    # Test null input with allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test null input with allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test that validate calls target's validate
    target_field.validate = lambda x: x.upper()
    assert reference.validate("test") == "TEST"


# LLM-generated content at query #85
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field()
    definitions['test_field'] = target_field

    ref = Reference(to='test_field', definitions=definitions)

    # Test with valid value
    target_field.validate = lambda x: x
    assert ref.validate("valid_value") == "valid_value"

    # Test with None and allow_null=True
    ref.allow_null = True
    assert ref.validate(None) is None

    # Test with None and allow_null=False
    ref.allow_null = False
    try:
        ref.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with target validation error
    target_field.validate = lambda x: 1/0  # Force an error
    try:
        ref.validate("invalid_value")
        assert False, "Expected ValidationError"
    except ValidationError:
        pass


# LLM-generated content at query #86
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field()
    definitions['test_ref'] = target_field
    ref = Reference(to='test_ref', definitions=definitions)

    # Test with valid value
    assert ref.validate("valid_value") == "valid_value"

    # Test with None when allow_null is True
    ref.allow_null = True
    assert ref.validate(None) is None

    # Test with None when allow_null is False
    ref.allow_null = False
    try:
        ref.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with target field that raises ValidationError
    target_field_with_error = Field()
    target_field_with_error.validate = lambda x: 1/0  # Force error
    definitions['error_ref'] = target_field_with_error
    ref_error = Reference(to='error_ref', definitions=definitions)

    try:
        ref_error.validate("any_value")
        assert False, "Expected ValidationError"
    except ValidationError:
        pass


# LLM-generated content at query #87
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field()
    definitions['test_field'] = target_field
    reference = Reference(to='test_field', definitions=definitions)

    # Test with valid value
    target_field.validate = lambda x: x  # Mock validate to return input
    assert reference.validate("valid_value") == "valid_value"

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with target field raising ValidationError
    target_field.validate = lambda x: (_ for _ in ()).throw(ValidationError(messages=[Message(text="error", code="test")]))
    try:
        reference.validate("invalid_value")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "test"


# LLM-generated content at query #88
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    fields = {
        "name": Field(allow_null=False),
        "age": Field(allow_null=True),
    }
    schema = Schema(fields=fields)

    # Test valid input
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test with default value
    fields_with_default = {
        "name": Field(allow_null=False, default="Unknown"),
        "age": Field(allow_null=True),
    }
    schema_with_default = Schema(fields=fields_with_default)
    assert schema_with_default.validate({"age": 30}) == {"name": "Unknown", "age": 30}

    # Test with read-only field
    fields_with_readonly = {
        "name": Field(allow_null=False),
        "id": Field(allow_null=False, read_only=True),
    }
    schema_with_readonly = Schema(fields=fields_with_readonly)
    assert schema_with_readonly.validate({"name": "John"}) == {"name": "John"}

    # Test null input with allow_null=True
    schema_allow_null = Schema(fields=fields, allow_null=True)
    assert schema_allow_null.validate(None) is None

    # Test null input with allow_null=False
    with pytest.raises(ValidationError) as exc_info:
        schema.validate(None)
    assert exc_info.value.messages[0].code == "null"

    # Test non-dict input
    with pytest.raises(ValidationError) as exc_info:
        schema.validate("not a dict")
    assert exc_info.value.messages[0].code == "type"

    # Test invalid key type
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({123: "invalid key"})
    assert exc_info.value.messages[0].code == "invalid_key"

    # Test missing required field
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({"age": 30})
    assert exc_info.value.messages[0].code == "required"

    # Test child field validation error
    fields_with_child_validation = {
        "name": Field(allow_null=False, min_length=3),
    }
    schema_with_child_validation = Schema(fields=fields_with_child_validation)
    with pytest.raises(ValidationError) as exc_info:
        schema_with_child_validation.validate({"name": "Jo"})
    assert exc_info.value.messages[0].code == "min_length"


# LLM-generated content at query #89
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    schema = Schema(fields={"name": Field(), "age": Field()})
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test None input with allow_null=True
    schema = Schema(fields={"name": Field()}, allow_null=True)
    assert schema.validate(None) is None

    # Test None input with allow_null=False
    schema = Schema(fields={"name": Field()}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test non-dict input
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"

    # Test invalid key type
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate({123: "invalid key"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

    # Test missing required field
    schema = Schema(fields={"name": Field(), "age": Field()})
    try:
        schema.validate({"name": "John"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"

    # Test field with default value
    schema = Schema(fields={"name": Field(), "age": Field(default=25)})
    assert schema.validate({"name": "John"}) == {"name": "John", "age": 25}

    # Test read-only field
    schema = Schema(fields={"name": Field(), "id": Field(read_only=True)})
    assert schema.validate({"name": "John"}) == {"name": "John"}


# LLM-generated content at query #90
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    definitions["test_schema"] = Schema(fields={"name": Field()})

    reference = Reference(to="test_schema", definitions=definitions)

    # Test valid input
    assert reference.validate({"name": "value"}) == {"name": "value"}

    # Test null input with allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test null input with allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test invalid input (should raise ValidationError from target schema)
    try:
        reference.validate("invalid")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"


# LLM-generated content at query #91
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    definitions["test_schema"] = Schema(fields={"name": Field()})

    reference = Reference(to="test_schema", definitions=definitions)

    # Test valid input
    valid_input = {"name": "test"}
    assert reference.validate(valid_input) == valid_input

    # Test null input with allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test null input with allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError for null input"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test invalid input (should propagate from target schema)
    invalid_input = {"name": 123}  # Assuming Field() expects string
    try:
        reference.validate(invalid_input)
        assert False, "Expected ValidationError for invalid input"
    except ValidationError:
        pass  # Expected


# LLM-generated content at query #92
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    field = Field()
    definitions['test_field'] = field
    reference = Reference(to='test_field', definitions=definitions)

    # Test with valid value
    assert reference.validate("valid_value") == "valid_value"

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test that validate calls target's validate
    with pytest.raises(ValidationError):
        reference.validate("invalid_value")


# LLM-generated content at query #93
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    field = Field()
    definitions['test_field'] = field
    reference = Reference(to='test_field', definitions=definitions)

    # Test valid input
    assert reference.validate('valid_value') == 'valid_value'

    # Test null input with allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test null input with allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test validation error from target field
    field.validate = lambda x: 1/0  # Force an error
    try:
        reference.validate('value')
        assert False, "Expected ValidationError"
    except ValidationError:
        pass


# LLM-generated content at query #94
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    field = Field()
    definitions['test_field'] = field
    reference = Reference(to='test_field', definitions=definitions)

    # Test with valid value
    assert reference.validate('valid_value') == field.validate('valid_value')

    # Test with None when allow_null is True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None when allow_null is False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"


# LLM-generated content at query #95
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    field = Field()
    definitions['test_field'] = field
    reference = Reference(to='test_field', definitions=definitions)

    # Test with valid value
    assert reference.validate("valid_value") == "valid_value"

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with invalid value
    with pytest.raises(ValidationError):
        reference.validate("invalid_value")


# LLM-generated content at query #96
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field()
    definitions["target"] = target_field
    reference = Reference(to="target", definitions=definitions)

    # Test with valid value
    target_field.validate = lambda x: x
    assert reference.validate("valid_value") == "valid_value"

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with target validation error
    target_field.validate = lambda x: 1 / 0  # Force error
    try:
        reference.validate("any_value")
        assert False, "Expected ValidationError"
    except ValidationError:
        pass


# LLM-generated content at query #97
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    fields = {"name": Field(), "age": Field()}
    schema = Schema(fields=fields)
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test None with allow_null=True
    schema = Schema(fields=fields, allow_null=True)
    assert schema.validate(None) is None

    # Test None with allow_null=False
    schema = Schema(fields=fields, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test non-dict input
    try:
        schema.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"

    # Test invalid key type
    try:
        schema.validate({123: "value"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

    # Test missing required field
    required_fields = {"name": Field(), "age": Field(required=True)}
    schema = Schema(fields=required_fields)
    try:
        schema.validate({"name": "John"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"

    # Test field with default value
    fields_with_default = {"name": Field(), "age": Field(default=25)}
    schema = Schema(fields=fields_with_default)
    assert schema.validate({"name": "John"}) == {"name": "John", "age": 25}

    # Test nested validation error
    nested_fields = {"name": Field(), "details": Schema(fields={"age": Field()})}
    schema = Schema(fields=nested_fields)
    try:
        schema.validate({"name": "John", "details": {"age": "invalid"}})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].index == ["details", "age"]


# LLM-generated content at query #98
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    field = Field()
    definitions['test_field'] = field
    reference = Reference(to='test_field', definitions=definitions)

    # Test with valid value
    assert reference.validate("valid_value") == "valid_value"

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test that validate delegates to target's validate
    target_validate_called = False
    def mock_validate(value):
        nonlocal target_validate_called
        target_validate_called = True
        return value
    reference.target.validate = mock_validate
    reference.validate("test_value")
    assert target_validate_called


# LLM-generated content at query #99
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field()
    definitions["test_field"] = target_field
    reference = Reference(to="test_field", definitions=definitions)

    # Test with valid value
    assert reference.validate("valid_value") == "valid_value"

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with target field validation error
    target_field_with_error = Field()
    target_field_with_error.validate = lambda value: 1 / 0  # Force error
    definitions["error_field"] = target_field_with_error
    error_reference = Reference(to="error_field", definitions=definitions)
    try:
        error_reference.validate("any_value")
        assert False, "Expected ValidationError"
    except ValidationError:
        pass


# LLM-generated content at query #100
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field()
    definitions['test_field'] = target_field
    reference = Reference(to='test_field', definitions=definitions)

    # Test with valid value
    assert reference.validate("valid_value") == "valid_value"

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with target field validation error
    target_field_with_error = Field()
    target_field_with_error.validate = lambda x: 1/0  # Force error
    definitions['error_field'] = target_field_with_error
    error_reference = Reference(to='error_field', definitions=definitions)
    try:
        error_reference.validate("any_value")
        assert False, "Expected ValidationError"
    except ValidationError:
        pass


# LLM-generated content at query #101
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field()
    definitions['test_field'] = target_field
    reference = Reference(to='test_field', definitions=definitions)

    # Test with valid value
    assert reference.validate("valid_value") == "valid_value"

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with target field validation error
    target_field_with_error = Field()
    target_field_with_error.validate = lambda x: 1/0  # Force an error
    definitions['error_field'] = target_field_with_error
    error_reference = Reference(to='error_field', definitions=definitions)
    try:
        error_reference.validate("any_value")
        assert False, "Expected ValidationError"
    except ValidationError:
        pass


# LLM-generated content at query #102
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field()
    definitions['test_field'] = target_field
    reference = Reference(to='test_field', definitions=definitions)

    # Test with valid value
    assert reference.validate('valid_value') == 'valid_value'

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test that target's validate is called
    target_field.validate = lambda x: x.upper()
    assert reference.validate('test') == 'TEST'


# LLM-generated content at query #103
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field()
    definitions["test_field"] = target_field
    reference = Reference(to="test_field", definitions=definitions)

    # Test with valid value
    assert reference.validate("valid_value") == "valid_value"

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with target field raising ValidationError
    target_field.validate = lambda x: (_ for _ in ()).throw(ValidationError(messages=[Message(text="test", code="test")]))
    try:
        reference.validate("invalid_value")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "test"


# LLM-generated content at query #104
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    schema = Schema(fields={"name": Field(), "age": Field()})
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test null input with allow_null=True
    schema = Schema(fields={"name": Field()}, allow_null=True)
    assert schema.validate(None) is None

    # Test null input with allow_null=False
    schema = Schema(fields={"name": Field()}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test non-dict input
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"

    # Test invalid key type
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate({123: "invalid key"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

    # Test missing required field
    schema = Schema(fields={"name": Field(), "age": Field()})
    try:
        schema.validate({"name": "John"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"

    # Test field with default value
    schema = Schema(fields={"name": Field(), "age": Field(default=25)})
    assert schema.validate({"name": "John"}) == {"name": "John", "age": 25}

    # Test read-only field
    schema = Schema(fields={"name": Field(), "id": Field(read_only=True)})
    assert schema.validate({"name": "John"}) == {"name": "John"}

    # Test nested validation error
    child_schema = Field()
    child_schema.validate = lambda x: (None, ValidationError(messages=[Message(text="Invalid", code="invalid")]))
    schema = Schema(fields={"name": child_schema})
    try:
        schema.validate({"name": "invalid"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid"


# LLM-generated content at query #105
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field()
    definitions["test_field"] = target_field
    reference = Reference(to="test_field", definitions=definitions)

    # Test with valid value
    assert reference.validate("valid_value") == "valid_value"

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with target field validation error
    target_field_with_error = Field()
    target_field_with_error.validate = lambda x: 1 / 0  # Force an error
    definitions["error_field"] = target_field_with_error
    error_reference = Reference(to="error_field", definitions=definitions)
    try:
        error_reference.validate("any_value")
        assert False, "Expected ValidationError"
    except ValidationError:
        pass


# LLM-generated content at query #106
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    schema = Schema(fields={"name": Field(), "age": Field()})
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test null input with allow_null=True
    schema = Schema(fields={"name": Field()}, allow_null=True)
    assert schema.validate(None) is None

    # Test null input with allow_null=False
    schema = Schema(fields={"name": Field()}, allow_null=False)
    with pytest.raises(ValidationError):
        schema.validate(None)

    # Test non-dict input
    schema = Schema(fields={"name": Field()})
    with pytest.raises(ValidationError):
        schema.validate("not a dict")

    # Test invalid key type
    schema = Schema(fields={"name": Field()})
    with pytest.raises(ValidationError):
        schema.validate({123: "invalid key"})

    # Test required field missing
    schema = Schema(fields={"name": Field(), "age": Field()})
    with pytest.raises(ValidationError):
        schema.validate({"name": "John"})

    # Test field with default value
    schema = Schema(fields={"name": Field(), "age": Field(default=25)})
    assert schema.validate({"name": "John"}) == {"name": "John", "age": 25}

    # Test read-only field
    schema = Schema(fields={"name": Field(), "id": Field(read_only=True)})
    assert schema.validate({"name": "John"}) == {"name": "John"}

    # Test child field validation error
    schema = Schema(fields={"name": Field(min_length=3)})
    with pytest.raises(ValidationError):
        schema.validate({"name": "Jo"})

    # Test multiple errors
    schema = Schema(fields={"name": Field(min_length=3), "age": Field(min_value=18)})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({"name": "Jo", "age": 17})
    assert len(exc_info.value.messages) == 2


# LLM-generated content at query #107
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    field = Field()
    definitions['test_field'] = field
    reference = Reference(to='test_field', definitions=definitions)

    # Test with valid value
    with patch.object(field, 'validate', return_value='validated_value'):
        assert reference.validate('test_value') == 'validated_value'
        field.validate.assert_called_once_with('test_value')

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    with pytest.raises(ValidationError) as exc_info:
        reference.validate(None)
    assert exc_info.value.messages[0].code == 'null'

    # Test with validation error
    with patch.object(field, 'validate', side_effect=ValidationError(messages=[Message(text='error', code='error')])):
        with pytest.raises(ValidationError) as exc_info:
            reference.validate('test_value')
        assert exc_info.value.messages[0].code == 'error'


# LLM-generated content at query #108
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    field = Field()
    definitions['test_field'] = field
    reference = Reference(to='test_field', definitions=definitions)

    # Test with valid value
    assert reference.validate("valid_value") == "valid_value"

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with invalid value
    field.validate = lambda x: 1/0  # Force an error
    try:
        reference.validate("invalid_value")
        assert False, "Expected ValidationError"
    except ValidationError:
        pass


# LLM-generated content at query #109
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    schema = Schema(fields={"name": Field(), "age": Field()})
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test null input with allow_null=True
    schema = Schema(fields={"name": Field()}, allow_null=True)
    assert schema.validate(None) is None

    # Test null input with allow_null=False
    schema = Schema(fields={"name": Field()}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test non-dict input
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"

    # Test invalid key type
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate({1: "invalid key"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

    # Test required field missing
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate({})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"

    # Test field validation error
    schema = Schema(fields={"name": Field(min_length=5)})
    try:
        schema.validate({"name": "John"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "min_length"

    # Test default value
    schema = Schema(fields={"name": Field(default="Default")})
    assert schema.validate({}) == {"name": "Default"}

    # Test read-only field
    schema = Schema(fields={"name": Field(read_only=True)})
    assert schema.validate({"name": "John"}) == {}


# LLM-generated content at query #110
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    field = Field()
    definitions['test_field'] = field
    reference = Reference(to='test_field', definitions=definitions)

    # Test with valid value
    assert reference.validate('valid_value') == 'valid_value'

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with invalid value
    field.validate = lambda x: 1 / 0  # Force an error
    try:
        reference.validate('invalid_value')
        assert False, "Expected ValidationError"
    except ValidationError:
        pass


# LLM-generated content at query #111
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    field = Field()
    definitions['test_field'] = field
    reference = Reference(to='test_field', definitions=definitions)

    # Test valid input
    assert reference.validate("valid_value") == "valid_value"

    # Test allow_null
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test null not allowed
    reference.allow_null = False
    with pytest.raises(ValidationError):
        reference.validate(None)

    # Test target field validation
    field_with_error = Field()
    field_with_error.validate = lambda x: (1/0)  # Force error
    definitions['error_field'] = field_with_error
    error_reference = Reference(to='error_field', definitions=definitions)
    with pytest.raises(ValidationError):
        error_reference.validate("any_value")


# LLM-generated content at query #112
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field(allow_null=True)
    definitions['test_field'] = target_field
    reference = Reference(to='test_field', definitions=definitions, allow_null=True)

    # Test with valid value
    assert reference.validate("valid_value") == "valid_value"

    # Test with None when allow_null is True
    assert reference.validate(None) is None

    # Test with None when allow_null is False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with invalid value (assuming target_field raises ValidationError)
    target_field.validate = lambda x: 1/0 if x == "invalid" else x  # Force error
    try:
        reference.validate("invalid")
        assert False, "Expected ValidationError"
    except ValidationError:
        pass


# LLM-generated content at query #113
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    schema = Schema(fields={"name": Field(), "age": Field()})
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test None input with allow_null=True
    schema = Schema(fields={"name": Field()}, allow_null=True)
    assert schema.validate(None) is None

    # Test None input with allow_null=False
    schema = Schema(fields={"name": Field()}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test non-dict input
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"

    # Test non-string key
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate({123: "value"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

    # Test missing required field
    schema = Schema(fields={"name": Field(), "age": Field()})
    try:
        schema.validate({"name": "John"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"

    # Test field with default value
    schema = Schema(fields={"name": Field(), "age": Field(default=25)})
    assert schema.validate({"name": "John"}) == {"name": "John", "age": 25}

    # Test read-only field
    schema = Schema(fields={"name": Field(), "id": Field(read_only=True)})
    assert schema.validate({"name": "John"}) == {"name": "John"}


# LLM-generated content at query #114
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field(allow_null=True)
    definitions["test_field"] = target_field
    reference = Reference(to="test_field", definitions=definitions, allow_null=True)

    # Test with valid value
    assert reference.validate("valid_value") == "valid_value"

    # Test with None and allow_null=True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with invalid value (assuming target_field raises ValidationError)
    target_field.validate = lambda x: (_ for _ in ()).throw(ValidationError(messages=[Message(text="Invalid", code="invalid")]))
    try:
        reference.validate("invalid_value")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid"


# LLM-generated content at query #115
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    definitions['test_schema'] = Schema(fields={'name': Field()})
    reference = Reference(to='test_schema', definitions=definitions)

    # Test valid input
    valid_input = {'name': 'test'}
    assert reference.validate(valid_input) == {'name': 'test'}

    # Test null input with allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test null input with allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test invalid input
    try:
        reference.validate("invalid")
        assert False, "Expected ValidationError"
    except ValidationError:
        pass


# LLM-generated content at query #116
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field()
    definitions['test_ref'] = target_field
    reference = Reference(to='test_ref', definitions=definitions)

    # Test with valid value
    target_field.validate = lambda x: x  # Mock validate to return input
    assert reference.validate("valid_value") == "valid_value"

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with target validation error
    target_field.validate = lambda x: 1 / 0  # Mock validate to raise error
    try:
        reference.validate("invalid_value")
        assert False, "Expected ValidationError"
    except ValidationError:
        pass


# LLM-generated content at query #117
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field(allow_null=True)
    definitions['test_field'] = target_field
    reference = Reference(to='test_field', definitions=definitions, allow_null=True)

    # Test with valid value
    assert reference.validate("valid_value") == "valid_value"

    # Test with None and allow_null=True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with target field validation error
    target_field.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"


# LLM-generated content at query #118
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    field = Field()
    definitions['test_field'] = field
    reference = Reference(to='test_field', definitions=definitions)

    # Test with valid value
    field.validate = lambda x: x
    assert reference.validate("valid_value") == "valid_value"

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with field validation error
    field.validate = lambda x: 1/0  # Force an error
    try:
        reference.validate("invalid_value")
        assert False, "Expected ValidationError"
    except ValidationError:
        pass


# LLM-generated content at query #119
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field()
    definitions['test_ref'] = target_field
    ref = Reference(to='test_ref', definitions=definitions)

    # Test with valid value
    assert ref.validate("valid_value") == "valid_value"

    # Test with None and allow_null=True
    ref.allow_null = True
    assert ref.validate(None) is None

    # Test with None and allow_null=False
    ref.allow_null = False
    try:
        ref.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test that target's validate is called
    target_field.validate = lambda x: x.upper()
    assert ref.validate("test") == "TEST"


# LLM-generated content at query #120
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field()
    definitions["test_field"] = target_field
    reference = Reference(to="test_field", definitions=definitions)

    # Test with valid value
    assert reference.validate("valid_value") == "valid_value"

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with target field validation error
    target_field_with_error = Field()
    target_field_with_error.validate = lambda x: 1/0  # Force an error
    definitions["error_field"] = target_field_with_error
    error_reference = Reference(to="error_field", definitions=definitions)
    try:
        error_reference.validate("any_value")
        assert False, "Expected ValidationError"
    except ValidationError:
        pass


# LLM-generated content at query #121
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    field = Field()
    definitions['test_field'] = field
    reference = Reference(to='test_field', definitions=definitions)

    # Test with valid value
    assert reference.validate("valid_value") == "valid_value"

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].text == "May not be null."


# LLM-generated content at query #122
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    schema = Schema(fields={"name": Field(), "age": Field()})
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test null input with allow_null=True
    schema = Schema(fields={"name": Field()}, allow_null=True)
    assert schema.validate(None) is None

    # Test null input with allow_null=False
    schema = Schema(fields={"name": Field()}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test non-dict input
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"

    # Test non-string key
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate({123: "value"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

    # Test missing required field
    schema = Schema(fields={"name": Field(), "age": Field()})
    try:
        schema.validate({"name": "John"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"

    # Test field with default value
    schema = Schema(fields={"name": Field(), "age": Field(default=25)})
    assert schema.validate({"name": "John"}) == {"name": "John", "age": 25}

    # Test read-only field
    schema = Schema(fields={"name": Field(), "id": Field(read_only=True)})
    assert schema.validate({"name": "John"}) == {"name": "John"}

    # Test nested validation error
    inner_schema = Schema(fields={"value": Field()})
    schema = Schema(fields={"inner": inner_schema})
    try:
        schema.validate({"inner": {"invalid": "data"}})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].index == ["inner", "required"]


# LLM-generated content at query #123
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    fields = {
        "name": Field(allow_null=False),
        "age": Field(allow_null=True),
    }
    schema = Schema(fields=fields)
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test null input with allow_null=True
    schema_with_null = Schema(fields=fields, allow_null=True)
    assert schema_with_null.validate(None) is None

    # Test null input with allow_null=False
    with pytest.raises(ValidationError):
        schema.validate(None)

    # Test non-dict input
    with pytest.raises(ValidationError):
        schema.validate("not a dict")

    # Test invalid key type
    with pytest.raises(ValidationError):
        schema.validate({123: "invalid key"})

    # Test missing required field
    with pytest.raises(ValidationError):
        schema.validate({"age": 30})

    # Test field with default value
    field_with_default = Field(allow_null=False, default="default_value")
    fields_with_default = {"name": field_with_default}
    schema_with_default = Schema(fields=fields_with_default)
    assert schema_with_default.validate({}) == {"name": "default_value"}

    # Test field validation error
    def failing_validator(value):
        raise ValidationError(messages=[Message(text="Failed", code="failed")])

    fields_with_failing_validator = {"name": Field(validate=failing_validator)}
    schema_with_failing_validator = Schema(fields=fields_with_failing_validator)
    with pytest.raises(ValidationError):
        schema_with_failing_validator.validate({"name": "test"})

    # Test read-only field
    fields_with_readonly = {"name": Field(read_only=True), "age": Field()}
    schema_with_readonly = Schema(fields=fields_with_readonly)
    assert schema_with_readonly.validate({"age": 30}) == {"age": 30}


# LLM-generated content at query #124
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field()
    definitions['test_field'] = target_field

    reference = Reference(to='test_field', definitions=definitions)

    # Test with valid value
    assert reference.validate("valid_value") == "valid_value"

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test with target field validation error
    target_field_with_error = Field()
    target_field_with_error.validate = lambda x: 1/0  # Force error

    definitions['error_field'] = target_field_with_error
    error_reference = Reference(to='error_field', definitions=definitions)

    try:
        error_reference.validate("any_value")
        assert False, "Expected ValidationError"
    except ValidationError:
        pass


# LLM-generated content at query #125
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    schema = Schema(fields={"name": Field(), "age": Field()})
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test allow_null with None input
    schema = Schema(fields={"name": Field()}, allow_null=True)
    assert schema.validate(None) is None

    # Test non-nullable with None input
    schema = Schema(fields={"name": Field()}, allow_null=False)
    with pytest.raises(ValidationError):
        schema.validate(None)

    # Test non-dict input
    schema = Schema(fields={"name": Field()})
    with pytest.raises(ValidationError):
        schema.validate("not a dict")

    # Test invalid key type
    schema = Schema(fields={"name": Field()})
    with pytest.raises(ValidationError):
        schema.validate({123: "invalid key"})

    # Test required field missing
    schema = Schema(fields={"name": Field(), "age": Field()})
    with pytest.raises(ValidationError):
        schema.validate({"name": "John"})

    # Test read_only field
    schema = Schema(fields={"name": Field(), "id": Field(read_only=True)})
    assert schema.validate({"name": "John"}) == {"name": "John"}

    # Test field with default value
    schema = Schema(fields={"name": Field(), "age": Field(default=25)})
    assert schema.validate({"name": "John"}) == {"name": "John", "age": 25}

    # Test nested validation error
    inner_schema = Schema(fields={"street": Field()})
    schema = Schema(fields={"name": Field(), "address": inner_schema})
    with pytest.raises(ValidationError):
        schema.validate({"name": "John", "address": {"city": "NYC"}})


# LLM-generated content at query #126
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    fields = {
        "name": Field(allow_null=False),
        "age": Field(allow_null=True),
    }
    schema = Schema(fields=fields)
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test null input with allow_null=True
    schema = Schema(fields=fields, allow_null=True)
    assert schema.validate(None) is None

    # Test null input with allow_null=False
    schema = Schema(fields=fields, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test non-dict input
    try:
        schema.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"

    # Test invalid key type
    try:
        schema.validate({123: "value"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

    # Test missing required field
    try:
        schema.validate({"age": 30})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"

    # Test field validation error
    fields = {"name": Field(allow_null=False)}
    schema = Schema(fields=fields)
    try:
        schema.validate({"name": None})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test default value
    fields = {"name": Field(default="default_name")}
    schema = Schema(fields=fields)
    assert schema.validate({}) == {"name": "default_name"}

    # Test read-only field
    fields = {"name": Field(read_only=True, default="default_name")}
    schema = Schema(fields=fields)
    assert schema.validate({}) == {"name": "default_name"}


# LLM-generated content at query #127
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    fields = {
        "name": Field(),
        "age": Field(),
    }
    schema = Schema(fields=fields)
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test with default values
    fields_with_default = {
        "name": Field(default="Unknown"),
        "age": Field(),
    }
    schema_with_default = Schema(fields=fields_with_default)
    assert schema_with_default.validate({"age": 30}) == {"name": "Unknown", "age": 30}

    # Test with missing required field
    schema = Schema(fields=fields)
    with pytest.raises(ValidationError) as excinfo:
        schema.validate({"name": "John"})
    assert "required" in str(excinfo.value)

    # Test with invalid key type
    with pytest.raises(ValidationError) as excinfo:
        schema.validate({123: "John", "age": 30})
    assert "invalid_key" in str(excinfo.value)

    # Test with null value and allow_null=True
    schema_allow_null = Schema(fields=fields, allow_null=True)
    assert schema_allow_null.validate(None) is None

    # Test with null value and allow_null=False
    with pytest.raises(ValidationError) as excinfo:
        schema.validate(None)
    assert "null" in str(excinfo.value)

    # Test with non-dict input
    with pytest.raises(ValidationError) as excinfo:
        schema.validate("not a dict")
    assert "type" in str(excinfo.value)


