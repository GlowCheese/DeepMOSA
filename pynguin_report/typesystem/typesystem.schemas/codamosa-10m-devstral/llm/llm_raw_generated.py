####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_Schema_serialize():
    # Test with a simple schema
    schema = Schema(fields={
        'name': Field(),
        'age': Field()
    })
    obj = {'name': 'John', 'age': 30}
    result = schema.serialize(obj)
    assert result == {'name': 'John', 'age': 30}

    # Test with None
    result = schema.serialize(None)
    assert result is None

    # Test with an object (not a dict)
    class Person:
        def __init__(self):
            self.name = 'Jane'
            self.age = 25
    obj = Person()
    result = schema.serialize(obj)
    assert result == {'name': 'Jane', 'age': 25}

    # Test with missing keys
    obj = {'name': 'Bob'}
    result = schema.serialize(obj)
    assert result == {'name': 'Bob'}

    # Test with nested schema
    nested_schema = Schema(fields={
        'address': Schema(fields={
            'street': Field(),
            'city': Field()
        })
    })
    obj = {'address': {'street': '123 Main St', 'city': 'New York'}}
    result = nested_schema.serialize(obj)
    assert result == {'address': {'street': '123 Main St', 'city': 'New York'}}


# LLM-generated content at query #2
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
    try:
        schema.validate({"name": "John"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"

    # Test field with default value
    fields_with_default = {
        "name": Field(),
        "age": Field(default=25),
    }
    schema = Schema(fields=fields_with_default)
    assert schema.validate({"name": "John"}) == {"name": "John", "age": 25}

    # Test read-only field
    fields_with_readonly = {
        "name": Field(),
        "id": Field(read_only=True),
    }
    schema = Schema(fields=fields_with_readonly)
    assert schema.validate({"name": "John"}) == {"name": "John"}

    # Test child field validation error
    fields_with_child_validation = {
        "name": Field(),
        "age": Field(min_value=0),
    }
    schema = Schema(fields=fields_with_child_validation)
    try:
        schema.validate({"name": "John", "age": -5})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "min_value"


# LLM-generated content at query #3
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

    # Test read_only field
    schema = Schema(fields={"name": Field(read_only=True), "age": Field()})
    assert schema.validate({"age": 30}) == {"age": 30}


# LLM-generated content at query #4
#--------------------------

```python
def test_Schema():
    # Test basic initialization
    fields = {"name": Field(), "age": Field()}
    schema = Schema(fields=fields)
    assert schema.fields == fields
    assert schema.required == ["name", "age"]

    # Test with read_only field
    fields_with_readonly = {"name": Field(), "age": Field(read_only=True)}
    schema_with_readonly = Schema(fields=fields_with_readonly)
    assert schema_with_readonly.fields == fields_with_readonly
    assert schema_with_readonly.required == ["name"]

    # Test with default value
    fields_with_default = {"name": Field(), "age": Field(default=10)}
    schema_with_default = Schema(fields=fields_with_default)
    assert schema_with_default.fields == fields_with_default
    assert schema_with_default.required == ["name"]

    # Test with allow_null
    schema_with_null = Schema(fields=fields, allow_null=True)
    assert schema_with_null.allow_null is True


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


# LLM-generated content at query #6
#--------------------------

```python
def test_Schema_validate():
    # Test with valid input
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
        schema.validate({123: "invalid key"})

    # Test with missing required field
    schema = Schema(fields={"name": Field(), "age": Field()})
    with pytest.raises(ValidationError):
        schema.validate({"name": "John"})

    # Test with field having default value
    schema = Schema(fields={"name": Field(), "age": Field(default=25)})
    assert schema.validate({"name": "John"}) == {"name": "John", "age": 25}

    # Test with read-only field
    schema = Schema(fields={"name": Field(), "id": Field(read_only=True)})
    assert schema.validate({"name": "John"}) == {"name": "John"}

    # Test with nested validation error
    child_schema = Field()
    child_schema.validate = lambda x: (None, ValidationError(messages=[Message(text="Invalid", code="invalid")]))
    schema = Schema(fields={"name": child_schema})
    with pytest.raises(ValidationError):
        schema.validate({"name": "invalid value"})


# LLM-generated content at query #7
#--------------------------

```python
def test_Schema_serialize():
    # Test basic serialization with a simple schema
    schema = Schema(fields={"name": Field(), "age": Field()})
    obj = {"name": "John", "age": 30}
    assert schema.serialize(obj) == {"name": "John", "age": 30}

    # Test serialization with None value
    assert schema.serialize(None) is None

    # Test serialization with nested schema
    nested_schema = Schema(fields={"address": Schema(fields={"city": Field(), "zip": Field()})})
    obj = {"address": {"city": "New York", "zip": "10001"}}
    assert nested_schema.serialize(obj) == {"address": {"city": "New York", "zip": "10001"}}

    # Test serialization with missing fields
    obj = {"name": "John"}
    assert schema.serialize(obj) == {"name": "John"}

    # Test serialization with an object (not a dict)
    class Person:
        def __init__(self):
            self.name = "John"
            self.age = 30

    person = Person()
    assert schema.serialize(person) == {"name": "John", "age": 30}

    # Test serialization with a field that has a custom serialize method
    class CustomField(Field):
        def serialize(self, value):
            return f"custom_{value}"

    schema_with_custom = Schema(fields={"name": CustomField()})
    obj = {"name": "John"}
    assert schema_with_custom.serialize(obj) == {"name": "custom_John"}


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
        assert any(msg.code == "required" for msg in e.messages)

    # Test field with default value
    schema = Schema(fields={"name": Field(), "age": Field(default=25)})
    assert schema.validate({"name": "John"}) == {"name": "John", "age": 25}

    # Test read-only field
    schema = Schema(fields={"name": Field(), "id": Field(read_only=True)})
    assert schema.validate({"name": "John"}) == {"name": "John"}


# LLM-generated content at query #9
#--------------------------

```python
def test_Schema_serialize():
    # Test with a simple schema and valid input
    schema = Schema(fields={"name": Field(), "age": Field()})
    assert schema.serialize({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test with None input
    assert schema.serialize(None) is None

    # Test with an object that has attributes
    class Person:
        def __init__(self):
            self.name = "Jane"
            self.age = 25

    person = Person()
    assert schema.serialize(person) == {"name": "Jane", "age": 25}

    # Test with missing keys in input
    assert schema.serialize({"name": "John"}) == {"name": "John"}

    # Test with nested schema
    nested_schema = Schema(fields={"address": Schema(fields={"city": Field(), "zip": Field()})})
    assert nested_schema.serialize({"address": {"city": "NYC", "zip": "10001"}}) == {"address": {"city": "NYC", "zip": "10001"}}


# LLM-generated content at query #10
#--------------------------

```python
def test_Schema_serialize():
    # Test with a simple schema
    schema = Schema(fields={"name": Field(), "age": Field()})
    assert schema.serialize({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test with None
    assert schema.serialize(None) is None

    # Test with an object
    class Person:
        def __init__(self, name, age):
            self.name = name
            self.age = age

    person = Person("Jane", 25)
    assert schema.serialize(person) == {"name": "Jane", "age": 25}

    # Test with missing fields
    assert schema.serialize({"name": "Bob"}) == {"name": "Bob"}
    assert schema.serialize({"age": 40}) == {"age": 40}

    # Test with nested schema
    nested_schema = Schema(fields={
        "address": Schema(fields={
            "street": Field(),
            "city": Field()
        })
    })
    assert nested_schema.serialize({
        "address": {"street": "123 Main St", "city": "New York"}
    }) == {
        "address": {"street": "123 Main St", "city": "New York"}
    }

    # Test with read-only fields
    schema_with_readonly = Schema(fields={
        "name": Field(),
        "id": Field(read_only=True)
    })
    assert schema_with_readonly.serialize({"name": "Alice", "id": 1}) == {"name": "Alice"}


# LLM-generated content at query #11
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


# LLM-generated content at query #12
#--------------------------

```python
def test_Schema_serialize():
    # Test with a simple schema
    schema = Schema(fields={"name": Field(), "age": Field()})
    assert schema.serialize({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test with nested schema
    nested_schema = Schema(fields={"address": Schema(fields={"city": Field(), "street": Field()})})
    assert nested_schema.serialize({"address": {"city": "New York", "street": "123 Main St"}}) == {"address": {"city": "New York", "street": "123 Main St"}}

    # Test with None value
    assert schema.serialize(None) is None

    # Test with object attributes
    class Person:
        def __init__(self, name, age):
            self.name = name
            self.age = age

    person = Person("Jane", 25)
    assert schema.serialize(person) == {"name": "Jane", "age": 25}

    # Test with missing keys
    assert schema.serialize({"name": "John"}) == {"name": "John"}

    # Test with extra keys
    assert schema.serialize({"name": "John", "age": 30, "extra": "value"}) == {"name": "John", "age": 30}


# LLM-generated content at query #13
#--------------------------

```python
def test_Schema_serialize():
    # Test basic serialization with dict input
    schema = Schema(fields={"name": Field(), "age": Field()})
    input_data = {"name": "John", "age": 30}
    assert schema.serialize(input_data) == {"name": "John", "age": 30}

    # Test serialization with object input
    class Person:
        def __init__(self):
            self.name = "Jane"
            self.age = 25

    person = Person()
    assert schema.serialize(person) == {"name": "Jane", "age": 25}

    # Test serialization with None input
    assert schema.serialize(None) is None

    # Test serialization with missing fields
    partial_data = {"name": "Bob"}
    assert schema.serialize(partial_data) == {"name": "Bob"}

    # Test serialization with nested fields
    nested_schema = Schema(fields={
        "user": Schema(fields={"name": Field(), "age": Field()})
    })
    nested_data = {"user": {"name": "Alice", "age": 28}}
    assert nested_schema.serialize(nested_data) == {"user": {"name": "Alice", "age": 28}}

    # Test serialization with custom field serialization
    class CustomField(Field):
        def serialize(self, value):
            return f"custom_{value}"

    custom_schema = Schema(fields={"custom": CustomField()})
    assert custom_schema.serialize({"custom": "value"}) == {"custom": "custom_value"}


# LLM-generated content at query #14
#--------------------------

```python
def test_Schema_serialize():
    # Test with a simple schema
    schema = Schema(fields={
        'name': Field(),
        'age': Field()
    })
    assert schema.serialize({'name': 'John', 'age': 30}) == {'name': 'John', 'age': 30}

    # Test with nested fields
    nested_schema = Schema(fields={
        'user': Schema(fields={
            'name': Field(),
            'age': Field()
        })
    })
    assert nested_schema.serialize({'user': {'name': 'John', 'age': 30}}) == {'user': {'name': 'John', 'age': 30}}

    # Test with None value
    assert schema.serialize(None) is None

    # Test with object attributes
    class User:
        def __init__(self):
            self.name = 'John'
            self.age = 30

    user = User()
    assert schema.serialize(user) == {'name': 'John', 'age': 30}

    # Test with missing fields
    assert schema.serialize({'name': 'John'}) == {'name': 'John'}

    # Test with extra fields
    assert schema.serialize({'name': 'John', 'age': 30, 'extra': 'field'}) == {'name': 'John', 'age': 30}


# LLM-generated content at query #15
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field()
    definitions["target"] = target_field
    reference = Reference(to="target", definitions=definitions)

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


# LLM-generated content at query #16
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field(allow_null=True)
    definitions["test_field"] = target_field
    reference = Reference(to="test_field", definitions=definitions)

    # Test with valid input
    assert reference.validate("valid_value") == "valid_value"

    # Test with None when allow_null is True
    assert reference.validate(None) is None

    # Test with None when allow_null is False
    reference.allow_null = False
    with pytest.raises(ValidationError) as exc_info:
        reference.validate(None)
    assert exc_info.value.messages[0].code == "null"

    # Test with target field that raises ValidationError
    target_field.allow_null = False
    with pytest.raises(ValidationError):
        reference.validate(None)


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


# LLM-generated content at query #18
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
    schema = Schema(fields={"name": Field(read_only=True)})
    assert schema.validate({}) == {}


# LLM-generated content at query #19
#--------------------------

```python
def test_Schema_serialize():
    # Test basic serialization with dict input
    schema = Schema(fields={"name": Field(), "age": Field()})
    assert schema.serialize({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test serialization with missing keys
    assert schema.serialize({"name": "John"}) == {"name": "John"}

    # Test serialization with None input
    assert schema.serialize(None) is None

    # Test serialization with object input
    class Person:
        def __init__(self):
            self.name = "Jane"
            self.age = 25

    assert schema.serialize(Person()) == {"name": "Jane", "age": 25}

    # Test serialization with nested fields
    nested_schema = Schema(fields={
        "address": Schema(fields={
            "street": Field(),
            "city": Field()
        })
    })
    assert nested_schema.serialize({"address": {"street": "123 Main", "city": "NYC"}}) == {
        "address": {"street": "123 Main", "city": "NYC"}
    }

    # Test serialization with read-only fields
    schema_with_readonly = Schema(fields={
        "id": Field(read_only=True),
        "name": Field()
    })
    assert schema_with_readonly.serialize({"id": 1, "name": "Test"}) == {"name": "Test"}


# LLM-generated content at query #20
#--------------------------

```python
def test_Schema_serialize():
    # Test with a simple schema
    schema = Schema(fields={
        'name': Field(),
        'age': Field()
    })
    obj = {'name': 'John', 'age': 30}
    assert schema.serialize(obj) == {'name': 'John', 'age': 30}

    # Test with a None object
    assert schema.serialize(None) is None

    # Test with an object that has extra fields
    obj_with_extra = {'name': 'John', 'age': 30, 'extra': 'field'}
    assert schema.serialize(obj_with_extra) == {'name': 'John', 'age': 30}

    # Test with an object that has missing fields
    obj_missing = {'name': 'John'}
    assert schema.serialize(obj_missing) == {'name': 'John'}

    # Test with a non-dict object
    class MockObject:
        def __init__(self):
            self.name = 'John'
            self.age = 30

    mock_obj = MockObject()
    assert schema.serialize(mock_obj) == {'name': 'John', 'age': 30}

    # Test with nested fields
    nested_schema = Schema(fields={
        'user': Schema(fields={
            'name': Field(),
            'age': Field()
        })
    })
    nested_obj = {'user': {'name': 'John', 'age': 30}}
    assert nested_schema.serialize(nested_obj) == {'user': {'name': 'John', 'age': 30}}


# LLM-generated content at query #21
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
    target_field.validate = lambda x: 1/0  # Force error
    try:
        ref.validate("any_value")
        assert False, "Expected ValidationError"
    except ValidationError:
        pass


# LLM-generated content at query #22
#--------------------------

```python
def test_Schema_serialize():
    # Test with a simple schema and dict input
    schema = Schema(fields={"name": Field(), "age": Field()})
    input_data = {"name": "John", "age": 30}
    assert schema.serialize(input_data) == {"name": "John", "age": 30}

    # Test with None input
    assert schema.serialize(None) is None

    # Test with an object input (simulating an object with attributes)
    class MockObject:
        def __init__(self, name, age):
            self.name = name
            self.age = age

    obj = MockObject("Jane", 25)
    assert schema.serialize(obj) == {"name": "Jane", "age": 25}

    # Test with missing keys in input
    partial_input = {"name": "Bob"}
    assert schema.serialize(partial_input) == {"name": "Bob"}

    # Test with nested schema
    nested_schema = Schema(fields={
        "user": Schema(fields={"name": Field(), "age": Field()})
    })
    nested_input = {"user": {"name": "Alice", "age": 20}}
    assert nested_schema.serialize(nested_input) == {"user": {"name": "Alice", "age": 20}}


# LLM-generated content at query #23
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
    target_field_with_error.validate = lambda x: 1 / 0  # Force error
    definitions["error_field"] = target_field_with_error
    error_reference = Reference(to="error_field", definitions=definitions)
    try:
        error_reference.validate("any_value")
        assert False, "Expected ValidationError"
    except ValidationError:
        pass


# LLM-generated content at query #24
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field()
    definitions['target'] = target_field
    reference = Reference(to='target', definitions=definitions)

    # Test with None and allow_null=True
    reference.allow_null = True
    assert reference.validate(None) is None

    # Test with None and allow_null=False
    reference.allow_null = False
    with pytest.raises(ValidationError) as exc_info:
        reference.validate(None)
    assert exc_info.value.messages[0].code == "null"

    # Test with valid value
    target_field.validate = lambda x: x  # Mock validate to return input
    assert reference.validate("test_value") == "test_value"

    # Test with invalid value (target validation fails)
    target_field.validate = lambda x: 1 / 0  # Mock validate to raise error
    with pytest.raises(Exception):  # Should raise whatever target.validate raises
        reference.validate("test_value")


# LLM-generated content at query #25
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field()
    definitions["test_ref"] = target_field
    reference = Reference(to="test_ref", definitions=definitions)

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
    target_field_with_error.validate = lambda x: 1/0  # Force error
    definitions["error_ref"] = target_field_with_error
    error_reference = Reference(to="error_ref", definitions=definitions)
    try:
        error_reference.validate("any_value")
        assert False, "Expected ValidationError"
    except ValidationError:
        pass


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


# LLM-generated content at query #27
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

    # Test with target field raising ValidationError
    target_field_with_error = Field()
    target_field_with_error.validate = lambda x: 1/0  # Force error
    definitions['error_ref'] = target_field_with_error
    error_reference = Reference(to='error_ref', definitions=definitions)

    try:
        error_reference.validate("any_value")
        assert False, "Expected ValidationError"
    except ValidationError:
        pass


# LLM-generated content at query #28
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

    # Test invalid key type
    schema = Schema(fields={"name": Field()})
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
    schema = Schema(fields={"name": Field(read_only=True), "age": Field()})
    assert schema.validate({"age": 30}) == {"age": 30}

    # Test child field validation error
    child_field = Field()
    child_field.validate = lambda x: (None, ValidationError(messages=[Message(text="error", code="error")]))
    schema = Schema(fields={"name": child_field})
    try:
        schema.validate({"name": "value"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "error"


# LLM-generated content at query #30
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


# LLM-generated content at query #31
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
        def __init__(self, default="default", **kwargs):
            super().__init__(**kwargs)
            self._default = default

        def has_default(self):
            return True

        def get_default_value(self):
            return self._default

    schema = Schema(fields={"name": FieldWithDefault()})
    assert schema.validate({}) == {"name": "default"}

    # Test read-only field
    class ReadOnlyField(Field):
        def __init__(self, read_only=True, **kwargs):
            super().__init__(**kwargs)
            self.read_only = read_only

    schema = Schema(fields={"name": ReadOnlyField(), "age": Field()})
    assert schema.validate({"age": 30}) == {"age": 30}


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
        schema.validate({1: "value"})
    assert exc_info.value.messages[0].code == "invalid_key"

    # Test missing required field
    schema = Schema(fields={"name": Field(), "age": Field()})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({"name": "John"})
    assert exc_info.value.messages[0].code == "required"

    # Test field validation error
    schema = Schema(fields={"name": Field(min_length=5)})
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({"name": "John"})
    assert exc_info.value.messages[0].code == "min_length"

    # Test default value
    schema = Schema(fields={"name": Field(default="Default")})
    assert schema.validate({}) == {"name": "Default"}

    # Test read-only field
    schema = Schema(fields={"name": Field(read_only=True), "age": Field()})
    assert schema.validate({"age": 30}) == {"age": 30}


# LLM-generated content at query #33
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
    with patch.object(field, 'validate', return_value='another_value'):
        assert reference.validate('another_test') == 'another_value'
        field.validate.assert_called_once_with('another_test')


# LLM-generated content at query #34
#--------------------------

```python
def test_Reference_validate():
    # Test case 1: Valid value
    definitions = Definitions()
    target_schema = Schema(fields={"name": Field()})
    definitions["test_schema"] = target_schema
    reference = Reference(to="test_schema", definitions=definitions)

    valid_value = {"name": "John"}
    result = reference.validate(valid_value)
    assert result == {"name": "John"}

    # Test case 2: Null value with allow_null=True
    reference_allow_null = Reference(to="test_schema", definitions=definitions, allow_null=True)
    null_value = None
    result = reference_allow_null.validate(null_value)
    assert result is None

    # Test case 3: Null value with allow_null=False
    reference_no_null = Reference(to="test_schema", definitions=definitions, allow_null=False)
    with pytest.raises(ValidationError) as exc_info:
        reference_no_null.validate(null_value)
    assert exc_info.value.messages[0].code == "null"

    # Test case 4: Invalid value
    invalid_value = "not a dict"
    with pytest.raises(ValidationError) as exc_info:
        reference.validate(invalid_value)
    assert exc_info.value.messages[0].code == "type"


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
        assert any(msg.code == "required" for msg in e.messages)

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


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_Schema_serialize():
    # Test with a simple schema
    schema = Schema(fields={"name": Field(), "age": Field()})

    # Test with a dict object
    obj = {"name": "John", "age": 30}
    assert schema.serialize(obj) == {"name": "John", "age": 30}

    # Test with an object with attributes
    class Person:
        def __init__(self, name, age):
            self.name = name
            self.age = age

    person = Person("Jane", 25)
    assert schema.serialize(person) == {"name": "Jane", "age": 25}

    # Test with None
    assert schema.serialize(None) is None

    # Test with missing keys
    obj_missing = {"name": "Bob"}
    assert schema.serialize(obj_missing) == {"name": "Bob"}

    # Test with extra keys
    obj_extra = {"name": "Alice", "age": 28, "city": "NYC"}
    assert schema.serialize(obj_extra) == {"name": "Alice", "age": 28}

    # Test with nested schema
    nested_schema = Schema(fields={
        "address": Schema(fields={
            "street": Field(),
            "city": Field()
        })
    })
    nested_obj = {"address": {"street": "123 Main St", "city": "Boston"}}
    assert nested_schema.serialize(nested_obj) == {"address": {"street": "123 Main St", "city": "Boston"}}


# LLM-generated content at query #2
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    fields = {
        "name": Field(allow_null=False),
        "age": Field(allow_null=True)
    }
    schema = Schema(fields=fields)

    # Test with valid data
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test with missing required field
    try:
        schema.validate({"age": 30})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "required"

    # Test with null value when not allowed
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

    # Test with non-string keys
    try:
        schema.validate({1: "value"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "invalid_key"

    # Test with default value
    default_field = Field(allow_null=True, default="default")
    fields_with_default = {"name": default_field}
    schema_with_default = Schema(fields=fields_with_default)
    assert schema_with_default.validate({}) == {"name": "default"}

    # Test with read-only field
    read_only_field = Field(read_only=True)
    fields_with_read_only = {"name": read_only_field}
    schema_with_read_only = Schema(fields=fields_with_read_only)
    assert schema_with_read_only.validate({}) == {}


# LLM-generated content at query #3
#--------------------------

```python
def test_Schema():
    # Test basic initialization
    schema = Schema(fields={})
    assert schema.fields == {}
    assert schema.required == []

    # Test with fields
    field1 = Field()
    field2 = Field(read_only=True)
    field3 = Field(default="default_value")
    schema = Schema(fields={"field1": field1, "field2": field2, "field3": field3})
    assert schema.fields == {"field1": field1, "field2": field2, "field3": field3}
    assert schema.required == ["field1"]

    # Test with additional kwargs
    schema = Schema(fields={"field1": field1}, allow_null=True, description="Test schema")
    assert schema.fields == {"field1": field1}
    assert schema.allow_null is True
    assert schema.description == "Test schema"


# LLM-generated content at query #4
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

    # Test that validate calls target's validate
    with patch.object(field, 'validate', return_value='another_value'):
        assert reference.validate('another_test') == 'another_value'
        field.validate.assert_called_once_with('another_test')


# LLM-generated content at query #5
#--------------------------

```python
def test_Schema():
    # Test basic initialization
    fields = {"name": Field(), "age": Field()}
    schema = Schema(fields=fields)
    assert schema.fields == fields
    assert schema.required == ["name", "age"]

    # Test with read_only field
    fields = {"name": Field(read_only=True), "age": Field()}
    schema = Schema(fields=fields)
    assert schema.fields == fields
    assert schema.required == ["age"]

    # Test with field having default
    fields = {"name": Field(default="default_name"), "age": Field()}
    schema = Schema(fields=fields)
    assert schema.fields == fields
    assert schema.required == ["age"]

    # Test with allow_null
    fields = {"name": Field(), "age": Field()}
    schema = Schema(fields=fields, allow_null=True)
    assert schema.allow_null is True


# LLM-generated content at query #6
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field()
    definitions['test'] = target_field

    reference = Reference(to='test', definitions=definitions)

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


# LLM-generated content at query #7
#--------------------------

```python
def test_Schema_validate():
    # Test valid input
    fields = {
        "name": Field(allow_null=False),
        "age": Field(allow_null=False),
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
        schema.validate({"name": "John"})
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
        "name": Field(allow_null=False, default="Default"),
    }
    schema_with_default = Schema(fields=fields_with_default)
    assert schema_with_default.validate({}) == {"name": "Default"}

    # Test read-only field
    fields_with_readonly = {
        "name": Field(allow_null=False, read_only=True),
    }
    schema_with_readonly = Schema(fields=fields_with_readonly)
    assert schema_with_readonly.validate({}) == {}


# LLM-generated content at query #8
#--------------------------

```python
def test_Schema_serialize():
    # Test with a simple schema and a dictionary input
    schema = Schema(fields={"name": Field(), "age": Field()})
    input_dict = {"name": "John", "age": 30}
    assert schema.serialize(input_dict) == {"name": "John", "age": 30}

    # Test with a schema containing nested fields
    nested_schema = Schema(fields={
        "user": Schema(fields={"name": Field(), "age": Field()}),
        "status": Field()
    })
    nested_input = {"user": {"name": "Alice", "age": 25}, "status": "active"}
    assert nested_schema.serialize(nested_input) == {
        "user": {"name": "Alice", "age": 25},
        "status": "active"
    }

    # Test with an object input (non-dictionary)
    class User:
        def __init__(self, name, age):
            self.name = name
            self.age = age

    user = User("Bob", 35)
    assert schema.serialize(user) == {"name": "Bob", "age": 35}

    # Test with None input
    assert schema.serialize(None) is None

    # Test with missing keys in input
    partial_input = {"name": "Charlie"}
    assert schema.serialize(partial_input) == {"name": "Charlie"}

    # Test with extra keys in input (should be ignored)
    extra_input = {"name": "Dave", "age": 40, "extra": "ignored"}
    assert schema.serialize(extra_input) == {"name": "Dave", "age": 40}


# LLM-generated content at query #9
#--------------------------

```python
def test_Schema_serialize():
    # Test with a simple schema
    schema = Schema(fields={"name": Field(), "age": Field()})
    assert schema.serialize({"name": "John", "age": 30}) == {"name": "John", "age": 30}

    # Test with None
    assert schema.serialize(None) is None

    # Test with an object (not a dict)
    class Person:
        def __init__(self):
            self.name = "Jane"
            self.age = 25

    person = Person()
    assert schema.serialize(person) == {"name": "Jane", "age": 25}

    # Test with missing fields
    assert schema.serialize({"name": "Bob"}) == {"name": "Bob"}

    # Test with nested schema
    nested_schema = Schema(fields={
        "address": Schema(fields={
            "street": Field(),
            "city": Field()
        })
    })
    assert nested_schema.serialize({
        "address": {"street": "123 Main St", "city": "New York"}
    }) == {
        "address": {"street": "123 Main St", "city": "New York"}
    }

    # Test with empty dict
    assert schema.serialize({}) == {}


# LLM-generated content at query #10
#--------------------------

```python
def test_Schema_serialize():
    # Test basic serialization
    schema = Schema(fields={
        'name': Field(),
        'age': Field(),
    })
    obj = {'name': 'John', 'age': 30}
    assert schema.serialize(obj) == {'name': 'John', 'age': 30}

    # Test with None
    assert schema.serialize(None) is None

    # Test with object attributes
    class Person:
        def __init__(self):
            self.name = 'Jane'
            self.age = 25
    person = Person()
    assert schema.serialize(person) == {'name': 'Jane', 'age': 25}

    # Test with missing keys
    obj_missing = {'name': 'Bob'}
    assert schema.serialize(obj_missing) == {'name': 'Bob'}

    # Test with nested schema
    nested_schema = Schema(fields={
        'address': Schema(fields={
            'street': Field(),
            'city': Field(),
        }),
    })
    nested_obj = {'address': {'street': '123 Main', 'city': 'Springfield'}}
    assert nested_schema.serialize(nested_obj) == {'address': {'street': '123 Main', 'city': 'Springfield'}}


# LLM-generated content at query #11
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

    # Test validation error from target field
    target_field.validate = lambda x: 1/0  # Force an error
    try:
        reference.validate("any_value")
        assert False, "Expected ValidationError"
    except ValidationError:
        pass


# LLM-generated content at query #12
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
    schema = Schema(fields={"name": Field()})
    try:
        schema.validate({})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"

    # Test field with default value
    schema = Schema(fields={"name": Field(default="default_name")})
    assert schema.validate({}) == {"name": "default_name"}

    # Test field validation error
    schema = Schema(fields={"age": Field(min_value=0)})
    try:
        schema.validate({"age": -1})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "min_value"


# LLM-generated content at query #13
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

    # Test read_only field
    schema = Schema(fields={"name": Field(read_only=True)})
    assert schema.validate({}) == {}

    # Test nested field validation
    nested_field = Field()
    nested_field.validate = lambda x: (x, None) if x == "valid" else (None, ValidationError(messages=[Message(text="invalid", code="invalid")]))
    schema = Schema(fields={"nested": nested_field})
    try:
        schema.validate({"nested": "invalid"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid"


# LLM-generated content at query #14
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    field = Field()
    definitions['test'] = field
    reference = Reference(to='test', definitions=definitions)

    # Test with valid value
    assert reference.validate("valid_value") == field.validate("valid_value")

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


# LLM-generated content at query #15
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
    reference_allow_null = Reference(to='test_schema', definitions=definitions, allow_null=True)
    assert reference_allow_null.validate(None) is None

    # Test null input with allow_null=False
    with pytest.raises(ValidationError):
        reference.validate(None)

    # Test invalid input
    with pytest.raises(ValidationError):
        reference.validate({'invalid_key': 'value'})

    # Test nested validation error
    with pytest.raises(ValidationError):
        reference.validate({'name': None})


# LLM-generated content at query #16
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


# LLM-generated content at query #17
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


# LLM-generated content at query #18
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
    with pytest.raises(ValidationError):
        schema.validate(None)

    # Test with non-dict input
    schema = Schema(fields={"name": Field()})
    with pytest.raises(ValidationError):
        schema.validate("not a dict")

    # Test with non-string keys
    schema = Schema(fields={"name": Field()})
    with pytest.raises(ValidationError):
        schema.validate({1: "John"})

    # Test required field missing
    schema = Schema(fields={"name": Field(), "age": Field()})
    with pytest.raises(ValidationError):
        schema.validate({"name": "John"})

    # Test with default value
    schema = Schema(fields={"name": Field(), "age": Field(default=25)})
    assert schema.validate({"name": "John"}) == {"name": "John", "age": 25}

    # Test with read-only field
    schema = Schema(fields={"name": Field(), "id": Field(read_only=True)})
    assert schema.validate({"name": "John"}) == {"name": "John"}

    # Test nested schema
    nested_schema = Schema(fields={"street": Field(), "city": Field()})
    schema = Schema(fields={"name": Field(), "address": nested_schema})
    assert schema.validate({"name": "John", "address": {"street": "123 Main", "city": "Springfield"}}) == {
        "name": "John",
        "address": {"street": "123 Main", "city": "Springfield"}
    }


# LLM-generated content at query #19
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    target_field = Field(allow_null=True)
    definitions['test_field'] = target_field
    reference = Reference(to='test_field', definitions=definitions)

    # Test valid value
    assert reference.validate("valid_value") == "valid_value"

    # Test allow_null with None
    assert reference.validate(None) is None

    # Test null not allowed
    reference.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

    # Test target field validation error
    target_field.allow_null = False
    try:
        reference.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"


# LLM-generated content at query #20
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
    child_schema = Field()
    child_schema.validate = lambda x: (None, ValidationError(messages=[Message(text="error", code="error")]))
    schema = Schema(fields={"name": child_schema})
    try:
        schema.validate({"name": "John"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "error"


# LLM-generated content at query #21
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

    # Test None input with allow_null=True
    schema_with_null = Schema(fields=fields, allow_null=True)
    assert schema_with_null.validate(None) is None

    # Test None input with allow_null=False
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
    child_field = Field(allow_null=False)
    child_field.validate = lambda x: (None, ValidationError(messages=[Message(text="Invalid", code="invalid")]))  # type: ignore
    fields_with_error = {"name": child_field}
    schema_with_error = Schema(fields=fields_with_error)
    with pytest.raises(ValidationError) as exc_info:
        schema_with_error.validate({"name": "invalid"})
    assert exc_info.value.messages[0].code == "invalid"

    # Test default value for missing field
    child_field_with_default = Field(allow_null=False, default="default")
    fields_with_default = {"name": child_field_with_default}
    schema_with_default = Schema(fields=fields_with_default)
    assert schema_with_default.validate({}) == {"name": "default"}


# LLM-generated content at query #22
#--------------------------

```python
def test_Schema_validate():
    # Test case 1: Valid input with all required fields
    fields = {
        "name": Field(required=True),
        "age": Field(required=True),
    }
    schema = Schema(fields=fields)
    input_data = {"name": "John", "age": 30}
    assert schema.validate(input_data) == input_data

    # Test case 2: Valid input with optional fields
    fields = {
        "name": Field(required=True),
        "age": Field(required=False),
    }
    schema = Schema(fields=fields)
    input_data = {"name": "John"}
    assert schema.validate(input_data) == input_data

    # Test case 3: Missing required field
    fields = {
        "name": Field(required=True),
        "age": Field(required=True),
    }
    schema = Schema(fields=fields)
    input_data = {"name": "John"}
    with pytest.raises(ValidationError) as excinfo:
        schema.validate(input_data)
    assert any(msg.code == "required" for msg in excinfo.value.messages)

    # Test case 4: Non-string keys
    fields = {
        "name": Field(required=True),
    }
    schema = Schema(fields=fields)
    input_data = {123: "John"}
    with pytest.raises(ValidationError) as excinfo:
        schema.validate(input_data)
    assert any(msg.code == "invalid_key" for msg in excinfo.value.messages)

    # Test case 5: Null input with allow_null=True
    fields = {
        "name": Field(required=True),
    }
    schema = Schema(fields=fields, allow_null=True)
    assert schema.validate(None) is None

    # Test case 6: Null input with allow_null=False
    fields = {
        "name": Field(required=True),
    }
    schema = Schema(fields=fields, allow_null=False)
    with pytest.raises(ValidationError) as excinfo:
        schema.validate(None)
    assert any(msg.code == "null" for msg in excinfo.value.messages)

    # Test case 7: Non-dict input
    fields = {
        "name": Field(required=True),
    }
    schema = Schema(fields=fields)
    with pytest.raises(ValidationError) as excinfo:
        schema.validate("not a dict")
    assert any(msg.code == "type" for msg in excinfo.value.messages)

    # Test case 8: Field with default value
    fields = {
        "name": Field(required=True),
        "age": Field(default=25),
    }
    schema = Schema(fields=fields)
    input_data = {"name": "John"}
    assert schema.validate(input_data) == {"name": "John", "age": 25}

    # Test case 9: Read-only field
    fields = {
        "name": Field(required=True),
        "id": Field(read_only=True),
    }
    schema = Schema(fields=fields)
    input_data = {"name": "John", "id": 1}
    assert schema.validate(input_data) == {"name": "John"}

    # Test case 10: Nested field validation error
    fields = {
        "name": Field(required=True),
        "age": Field(required=True, validators=[lambda x: x > 0]),
    }
    schema = Schema(fields=fields)
    input_data = {"name": "John", "age": -5}
    with pytest.raises(ValidationError) as excinfo:
        schema.validate(input_data)
    assert any(msg.code == "invalid" for msg in excinfo.value.messages)


# LLM-generated content at query #23
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
        reference.validate("invalid_value")
        assert False, "Expected ValidationError"
    except ValidationError:
        pass


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
    with pytest.raises(ValidationError) as exc_info:
        reference.validate(None)
    assert exc_info.value.messages[0].code == 'null'


# LLM-generated content at query #25
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


# LLM-generated content at query #26
#--------------------------

```python
def test_Reference_validate():
    # Setup
    definitions = Definitions()
    definitions["test_schema"] = Schema(fields={"name": Field()})

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

    # Test invalid input (passed to target schema)
    try:
        reference.validate("invalid")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"


# LLM-generated content at query #27
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
    def failing_validate(value):
        raise ValidationError(messages=[Message(text="error", code="error")])
    target_field.validate = failing_validate
    try:
        reference.validate("any_value")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "error"


# LLM-generated content at query #28
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


# LLM-generated content at query #30
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

    # Test None input with allow_null=True
    schema_with_null = Schema(fields=fields, allow_null=True)
    assert schema_with_null.validate(None) is None

    # Test None input with allow_null=False
    with pytest.raises(ValidationError) as exc_info:
        schema.validate(None)
    assert exc_info.value.messages[0].code == "null"

    # Test non-dict input
    with pytest.raises(ValidationError) as exc_info:
        schema.validate("not a dict")
    assert exc_info.value.messages[0].code == "type"

    # Test invalid key type
    with pytest.raises(ValidationError) as exc_info:
        schema.validate({123: "value"})
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
        "name": Field(allow_null=False, default="Default"),
        "age": Field(allow_null=True),
    }
    schema_with_default = Schema(fields=fields_with_default)
    assert schema_with_default.validate({"age": 30}) == {"name": "Default", "age": 30}

    # Test read-only field
    fields_with_readonly = {
        "name": Field(allow_null=False, read_only=True),
        "age": Field(allow_null=True),
    }
    schema_with_readonly = Schema(fields=fields_with_readonly)
    assert schema_with_readonly.validate({"age": 30}) == {"age": 30}


