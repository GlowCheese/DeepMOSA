####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_validate_with_null_value_and_allow_null_true():
    reference = Reference(to="test", definitions={"test": Field()}, allow_null=True)
    assert reference.validate(None) is None

def test_validate_with_null_value_and_allow_null_false():
    reference = Reference(to="test", definitions={"test": Field()}, allow_null=False)
    try:
        reference.validate(None)
    except ValidationError as e:
        assert str(e) == "May not be null."

def test_validate_with_non_null_value():
    reference = Reference(to="test", definitions={"test": Field()})
    assert reference.validate("valid_value") == "valid_value"


# LLM-generated content at query #2
#--------------------------

```python
def test_schema_constructor_with_fields():
    fields = {"name": String(), "age": Integer()}
    schema = Schema(fields=fields)
    assert schema.fields == fields
    assert schema.required == ["name", "age"]

def test_schema_constructor_with_optional_fields():
    fields = {"name": String(default=""), "age": Integer(default=0)}
    schema = Schema(fields=fields)
    assert schema.fields == fields
    assert schema.required == []

def test_schema_constructor_with_readonly_fields():
    fields = {"name": String(read_only=True), "age": Integer()}
    schema = Schema(fields=fields)
    assert schema.fields == fields
    assert schema.required == ["age"]

def test_schema_constructor_with_allow_null():
    fields = {"name": String(), "age": Integer()}
    schema = Schema(fields=fields, allow_null=True)
    assert schema.fields == fields
    assert schema.allow_null is True
    assert schema.required == ["name", "age"]

def test_schema_constructor_with_title_and_description():
    fields = {"name": String(), "age": Integer()}
    schema = Schema(fields=fields, title="Test Schema", description="A test schema")
    assert schema.fields == fields
    assert schema.title == "Test Schema"
    assert schema.description == "A test schema"


# LLM-generated content at query #3
#--------------------------

```python
def test_validate_with_none_value_and_allow_null():
    schema = Schema(fields={}, allow_null=True)
    assert schema.validate(None) is None

def test_validate_with_none_value_and_not_allow_null():
    schema = Schema(fields={}, allow_null=False)
    try:
        schema.validate(None)
    except ValidationError as e:
        assert e.messages[0].code == "null"
    else:
        assert False, "Expected ValidationError"

def test_validate_with_non_dict_value():
    schema = Schema(fields={})
    try:
        schema.validate("not a dict")
    except ValidationError as e:
        assert e.messages[0].code == "type"
    else:
        assert False, "Expected ValidationError"

def test_validate_with_invalid_key_type():
    schema = Schema(fields={})
    try:
        schema.validate({123: "value"})
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"
    else:
        assert False, "Expected ValidationError"

def test_validate_with_missing_required_field():
    required_field = Field(required=True)
    schema = Schema(fields={"required_field": required_field})
    try:
        schema.validate({})
    except ValidationError as e:
        assert e.messages[0].code == "required"
    else:
        assert False, "Expected ValidationError"

def test_validate_with_read_only_field():
    read_only_field = Field(read_only=True)
    schema = Schema(fields={"read_only_field": read_only_field})
    result = schema.validate({"read_only_field": "value"})
    assert "read_only_field" not in result

def test_validate_with_default_value():
    field_with_default = Field(default="default_value")
    schema = Schema(fields={"field_with_default": field_with_default})
    result = schema.validate({})
    assert result["field_with_default"] == "default_value"

def test_validate_with_valid_data():
    field = Field()
    schema = Schema(fields={"field": field})
    result = schema.validate({"field": "value"})
    assert result == {"field": "value"}

def test_validate_with_child_validation_error():
    child_field = Field()
    child_field.validate = lambda x: (None, ValidationError(messages=[Message(text="error", code="error")]))
    schema = Schema(fields={"child_field": child_field})
    try:
        schema.validate({"child_field": "value"})
    except ValidationError as e:
        assert e.messages[0].code == "error"
    else:
        assert False, "Expected ValidationError"


# LLM-generated content at query #4
#--------------------------

```python
def test_validate_missing_required_key():
    schema = Schema(fields={"required_field": Field(required=True)})
    value = {}
    try:
        schema.validate(value)
    except ValidationError as e:
        assert any(msg.code == "required" for msg in e.messages)
    else:
        assert False, "Expected ValidationError"


# LLM-generated content at query #5
#--------------------------

```python
def test_predicate_at_line_32_evaluates_to_false():
    schema = Schema(fields={})
    assert not (key not in value)


# LLM-generated content at query #6
#--------------------------

```python
def test_validate_with_none_and_allow_null():
    schema = Schema(fields={}, allow_null=True)
    assert schema.validate(None) is None

def test_validate_with_none_and_not_allow_null():
    schema = Schema(fields={}, allow_null=False)
    try:
        schema.validate(None)
    except ValidationError as e:
        assert e.messages[0].code == "null"
    else:
        assert False, "Expected ValidationError"

def test_validate_with_non_dict_type():
    schema = Schema(fields={})
    try:
        schema.validate("not a dict")
    except ValidationError as e:
        assert e.messages[0].code == "type"
    else:
        assert False, "Expected ValidationError"

def test_validate_with_non_string_key():
    schema = Schema(fields={})
    try:
        schema.validate({123: "value"})
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"
    else:
        assert False, "Expected ValidationError"

def test_validate_with_missing_required_field():
    required_field = Field(required=True)
    schema = Schema(fields={"required_field": required_field})
    try:
        schema.validate({})
    except ValidationError as e:
        assert e.messages[0].code == "required"
    else:
        assert False, "Expected ValidationError"

def test_validate_with_read_only_field():
    read_only_field = Field(read_only=True)
    schema = Schema(fields={"read_only_field": read_only_field})
    result = schema.validate({"read_only_field": "value"})
    assert "read_only_field" not in result

def test_validate_with_default_value():
    field_with_default = Field(default="default_value")
    schema = Schema(fields={"field_with_default": field_with_default})
    result = schema.validate({})
    assert result["field_with_default"] == "default_value"

def test_validate_with_valid_data():
    field = Field()
    schema = Schema(fields={"field": field})
    result = schema.validate({"field": "value"})
    assert result["field"] == "value"

def test_validate_with_child_validation_error():
    child_field = Field()
    schema = Schema(fields={"child_field": child_field})
    try:
        schema.validate({"child_field": "invalid_value"})
    except ValidationError as e:
        assert e.messages[0].code == "invalid"
    else:
        assert False, "Expected ValidationError"


# LLM-generated content at query #7
#--------------------------

```python
def test_validate_with_none_value_and_allow_null():
    schema = Schema(fields={}, allow_null=True)
    assert schema.validate(None) is None

def test_validate_with_none_value_and_not_allow_null():
    schema = Schema(fields={}, allow_null=False)
    try:
        schema.validate(None)
    except ValidationError as e:
        assert e.messages[0].code == "null"
    else:
        assert False, "Expected ValidationError"

def test_validate_with_non_dict_value():
    schema = Schema(fields={})
    try:
        schema.validate("not a dict")
    except ValidationError as e:
        assert e.messages[0].code == "type"
    else:
        assert False, "Expected ValidationError"

def test_validate_with_non_string_key():
    schema = Schema(fields={})
    try:
        schema.validate({123: "value"})
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"
    else:
        assert False, "Expected ValidationError"

def test_validate_with_missing_required_field():
    required_field = Field()
    schema = Schema(fields={"required_field": required_field})
    try:
        schema.validate({})
    except ValidationError as e:
        assert e.messages[0].code == "required"
    else:
        assert False, "Expected ValidationError"

def test_validate_with_valid_data():
    field = Field()
    schema = Schema(fields={"key": field})
    assert schema.validate({"key": "value"}) == {"key": "value"}

def test_validate_with_read_only_field():
    read_only_field = Field(read_only=True)
    schema = Schema(fields={"read_only": read_only_field})
    assert schema.validate({}) == {}

def test_validate_with_default_value():
    field_with_default = Field(default="default_value")
    schema = Schema(fields={"optional": field_with_default})
    assert schema.validate({}) == {"optional": "default_value"}

def test_validate_with_child_validation_error():
    child_field = Field()
    child_field.validate = lambda x: (None, ValidationError(messages=[Message(text="error", code="child_error")]))
    schema = Schema(fields={"child": child_field})
    try:
        schema.validate({"child": "invalid"})
    except ValidationError as e:
        assert e.messages[0].code == "child_error"
    else:
        assert False, "Expected ValidationError"


# LLM-generated content at query #8
#--------------------------

```python
def test_serialize_with_none_input():
    schema = Schema(fields={})
    assert schema.serialize(None) is None

def test_serialize_with_empty_fields():
    schema = Schema(fields={})
    assert schema.serialize({}) == {}

def test_serialize_with_dict_input():
    field1 = Field()
    field1.serialize = lambda x: x.upper() if isinstance(x, str) else x
    field2 = Field()
    field2.serialize = lambda x: x * 2 if isinstance(x, int) else x
    schema = Schema(fields={"name": field1, "age": field2})
    input_data = {"name": "alice", "age": 25}
    expected_output = {"name": "ALICE", "age": 50}
    assert schema.serialize(input_data) == expected_output

def test_serialize_with_object_input():
    class TestObject:
        def __init__(self):
            self.name = "bob"
            self.age = 30

    field1 = Field()
    field1.serialize = lambda x: x.upper() if isinstance(x, str) else x
    field2 = Field()
    field2.serialize = lambda x: x * 2 if isinstance(x, int) else x
    schema = Schema(fields={"name": field1, "age": field2})
    obj = TestObject()
    expected_output = {"name": "BOB", "age": 60}
    assert schema.serialize(obj) == expected_output

def test_serialize_with_missing_keys():
    field1 = Field()
    field1.serialize = lambda x: x.upper() if isinstance(x, str) else x
    field2 = Field()
    field2.serialize = lambda x: x * 2 if isinstance(x, int) else x
    schema = Schema(fields={"name": field1, "age": field2})
    input_data = {"name": "charlie"}
    expected_output = {"name": "CHARLIE"}
    assert schema.serialize(input_data) == expected_output


# LLM-generated content at query #9
#--------------------------

```python
def test_validate_with_none_value_and_allow_null():
    schema = Schema(fields={}, allow_null=True)
    assert schema.validate(None) is None

def test_validate_with_none_value_and_not_allow_null():
    schema = Schema(fields={}, allow_null=False)
    try:
        schema.validate(None)
    except ValidationError as e:
        assert e.messages[0].code == "null"

def test_validate_with_non_dict_value():
    schema = Schema(fields={})
    try:
        schema.validate("not a dict")
    except ValidationError as e:
        assert e.messages[0].code == "type"

def test_validate_with_non_string_key():
    schema = Schema(fields={})
    try:
        schema.validate({123: "value"})
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

def test_validate_with_missing_required_field():
    required_field = Field(required=True)
    schema = Schema(fields={"required_field": required_field})
    try:
        schema.validate({})
    except ValidationError as e:
        assert e.messages[0].code == "required"

def test_validate_with_valid_data():
    field = Field()
    schema = Schema(fields={"field": field})
    assert schema.validate({"field": "value"}) == {"field": "value"}

def test_validate_with_read_only_field():
    read_only_field = Field(read_only=True)
    schema = Schema(fields={"read_only_field": read_only_field})
    assert schema.validate({}) == {}

def test_validate_with_default_value():
    field_with_default = Field(default="default_value")
    schema = Schema(fields={"field": field_with_default})
    assert schema.validate({}) == {"field": "default_value"}

def test_validate_with_child_validation_error():
    child_field = Field()
    child_field.validate = lambda x: (None, ValidationError(messages=[Message(text="error", code="error")]))
    schema = Schema(fields={"child_field": child_field})
    try:
        schema.validate({"child_field": "invalid"})
    except ValidationError as e:
        assert e.messages[0].code == "error"


# LLM-generated content at query #10
#--------------------------

```python
def test_validate_error_case():
    schema = Schema(fields={"key": Field()})
    value = {"key": "invalid_value"}
    schema.fields["key"].validate_or_error = lambda x: (None, "error")
    assert schema.validate(value) is None


# LLM-generated content at query #11
#--------------------------

```python
def test_validate_with_none_value_and_allow_null():
    schema = Schema(fields={}, allow_null=True)
    assert schema.validate(None) is None

def test_validate_with_none_value_and_not_allow_null():
    schema = Schema(fields={}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

def test_validate_with_non_dict_value():
    schema = Schema(fields={})
    try:
        schema.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"

def test_validate_with_non_string_key():
    schema = Schema(fields={})
    try:
        schema.validate({123: "value"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

def test_validate_with_missing_required_field():
    required_field = Field(required=True)
    schema = Schema(fields={"required_field": required_field})
    try:
        schema.validate({})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"

def test_validate_with_read_only_field():
    read_only_field = Field(read_only=True)
    schema = Schema(fields={"read_only_field": read_only_field})
    result = schema.validate({"read_only_field": "value"})
    assert "read_only_field" not in result

def test_validate_with_default_value():
    field_with_default = Field(default="default_value")
    schema = Schema(fields={"field_with_default": field_with_default})
    result = schema.validate({})
    assert result == {"field_with_default": "default_value"}

def test_validate_with_valid_input():
    field = Field()
    schema = Schema(fields={"field": field})
    result = schema.validate({"field": "value"})
    assert result == {"field": "value"}

def test_validate_with_child_validation_error():
    child_field = Field()
    child_field.validate = lambda x: (None, ValidationError(messages=[Message(text="error", code="error")]))
    schema = Schema(fields={"child_field": child_field})
    try:
        schema.validate({"child_field": "invalid"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "error"


# LLM-generated content at query #12
#--------------------------

```python
def test_validate_error_condition():
    schema = Schema(fields={})
    error = ValidationError(messages=[])
    assert not error


# LLM-generated content at query #13
#--------------------------

```python
def test_validate_predicate_line_37_false():
    field = Field()
    schema = Schema(fields={"key": field})
    value = {"key": "invalid_value"}
    field.validate_or_error = lambda x: (None, ValidationError(messages=[Message(text="error", code="error")]))
    assert schema.validate(value) == ValidationError(messages=[Message(text="error", code="error", index=["key"])])


# LLM-generated content at query #14
#--------------------------

```python
def test_validate_error_is_none():
    schema = Schema(fields={})
    value = {}
    child_value, error = schema.validate_or_error(value)
    assert error is None


# LLM-generated content at query #15
#--------------------------

```python
def test_validate_error_handling():
    schema = Schema(fields={})
    value = {"key": "value"}
    error = ValidationError(messages=[])
    assert schema.validate(value) == value
    assert not error


# LLM-generated content at query #16
#--------------------------

```python
def test_validate_no_error_adds_validated_value():
    schema = Schema(fields={"key": Field()})
    value = {"key": "valid_value"}
    validated = schema.validate(value)
    assert "key" in validated


# LLM-generated content at query #17
#--------------------------

```python
def test_validate_error_case():
    field = Field()
    schema = Schema(fields={"key": field})
    value = {"key": "invalid_value"}
    field.validate_or_error = lambda x: (None, ValidationError(messages=[Message(text="error", code="error")]))
    assert schema.validate(value) == ValidationError(messages=[Message(text="error", code="error", index=["key"])])


# LLM-generated content at query #18
#--------------------------

```python
def test_validate_with_none_value_and_allow_null():
    schema = Schema(fields={}, allow_null=True)
    assert schema.validate(None) is None

def test_validate_with_none_value_and_not_allow_null():
    schema = Schema(fields={}, allow_null=False)
    try:
        schema.validate(None)
    except ValidationError as e:
        assert e.messages[0].code == "null"
    else:
        assert False, "Expected ValidationError"

def test_validate_with_non_dict_value():
    schema = Schema(fields={})
    try:
        schema.validate("not a dict")
    except ValidationError as e:
        assert e.messages[0].code == "type"
    else:
        assert False, "Expected ValidationError"

def test_validate_with_non_string_key():
    schema = Schema(fields={})
    try:
        schema.validate({123: "value"})
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"
    else:
        assert False, "Expected ValidationError"

def test_validate_with_missing_required_field():
    required_field = Field(required=True)
    schema = Schema(fields={"required_field": required_field})
    try:
        schema.validate({})
    except ValidationError as e:
        assert e.messages[0].code == "required"
    else:
        assert False, "Expected ValidationError"

def test_validate_with_valid_data():
    field = Field()
    schema = Schema(fields={"test_field": field})
    result = schema.validate({"test_field": "test_value"})
    assert result == {"test_field": "test_value"}

def test_validate_with_read_only_field():
    read_only_field = Field(read_only=True)
    schema = Schema(fields={"read_only_field": read_only_field})
    result = schema.validate({"other_field": "value"})
    assert result == {}

def test_validate_with_default_value():
    field_with_default = Field(default="default_value")
    schema = Schema(fields={"field_with_default": field_with_default})
    result = schema.validate({})
    assert result == {"field_with_default": "default_value"}

def test_validate_with_child_validation_error():
    child_field = Field()
    schema = Schema(fields={"child_field": child_field})
    try:
        schema.validate({"child_field": "invalid_value"})
    except ValidationError as e:
        assert len(e.messages) > 0
    else:
        assert False, "Expected ValidationError"


# LLM-generated content at query #19
#--------------------------

```python
def test_validate_with_none_value_and_allow_null():
    schema = Schema(fields={}, allow_null=True)
    assert schema.validate(None) is None

def test_validate_with_none_value_and_not_allow_null():
    schema = Schema(fields={}, allow_null=False)
    try:
        schema.validate(None)
    except ValidationError as e:
        assert e.messages[0].code == "null"
    else:
        assert False, "Expected ValidationError"

def test_validate_with_non_dict_value():
    schema = Schema(fields={})
    try:
        schema.validate("not a dict")
    except ValidationError as e:
        assert e.messages[0].code == "type"
    else:
        assert False, "Expected ValidationError"

def test_validate_with_invalid_key_type():
    schema = Schema(fields={})
    try:
        schema.validate({123: "value"})
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"
    else:
        assert False, "Expected ValidationError"

def test_validate_with_missing_required_field():
    required_field = Field(required=True)
    schema = Schema(fields={"required_field": required_field})
    try:
        schema.validate({})
    except ValidationError as e:
        assert e.messages[0].code == "required"
    else:
        assert False, "Expected ValidationError"

def test_validate_with_read_only_field():
    read_only_field = Field(read_only=True)
    schema = Schema(fields={"read_only_field": read_only_field})
    result = schema.validate({"read_only_field": "value"})
    assert "read_only_field" not in result

def test_validate_with_default_value():
    field_with_default = Field(default="default_value")
    schema = Schema(fields={"field_with_default": field_with_default})
    result = schema.validate({})
    assert result["field_with_default"] == "default_value"

def test_validate_with_valid_data():
    field = Field()
    schema = Schema(fields={"field": field})
    result = schema.validate({"field": "value"})
    assert result["field"] == "value"

def test_validate_with_child_validation_error():
    child_field = Field()
    schema = Schema(fields={"child_field": child_field})
    try:
        schema.validate({"child_field": "invalid_value"})
    except ValidationError as e:
        assert e.messages[0].index == ["child_field"]
    else:
        assert False, "Expected ValidationError"


# LLM-generated content at query #20
#--------------------------

```python
def test_validate_child_schema_error():
    schema = Schema(fields={"key": Field()})
    value = {"key": "invalid_value"}
    schema.fields["key"].validate_or_error = lambda x: (None, "error")
    assert schema.validate(value) == ValidationError(messages=[Message(text="error", code="error", index=["key"])])


# LLM-generated content at query #21
#--------------------------

```python
def test_validate_with_none_value_and_allow_null():
    schema = Schema(fields={}, allow_null=True)
    assert schema.validate(None) is None

def test_validate_with_none_value_and_not_allow_null():
    schema = Schema(fields={}, allow_null=False)
    try:
        schema.validate(None)
    except ValidationError as e:
        assert e.messages[0].code == "null"
    else:
        assert False, "Expected ValidationError"

def test_validate_with_non_dict_value():
    schema = Schema(fields={})
    try:
        schema.validate("not a dict")
    except ValidationError as e:
        assert e.messages[0].code == "type"
    else:
        assert False, "Expected ValidationError"

def test_validate_with_invalid_key_type():
    schema = Schema(fields={})
    try:
        schema.validate({123: "value"})
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"
    else:
        assert False, "Expected ValidationError"

def test_validate_with_missing_required_field():
    required_field = Field(required=True)
    schema = Schema(fields={"required_field": required_field})
    try:
        schema.validate({})
    except ValidationError as e:
        assert e.messages[0].code == "required"
    else:
        assert False, "Expected ValidationError"

def test_validate_with_valid_data():
    field = Field()
    schema = Schema(fields={"field": field})
    result = schema.validate({"field": "value"})
    assert result == {"field": "value"}

def test_validate_with_read_only_field():
    read_only_field = Field(read_only=True)
    schema = Schema(fields={"read_only_field": read_only_field})
    result = schema.validate({})
    assert result == {}

def test_validate_with_default_value():
    field_with_default = Field(default="default_value")
    schema = Schema(fields={"field_with_default": field_with_default})
    result = schema.validate({})
    assert result == {"field_with_default": "default_value"}

def test_validate_with_child_validation_error():
    child_field = Field()
    schema = Schema(fields={"child_field": child_field})
    try:
        schema.validate({"child_field": "invalid_value"})
    except ValidationError as e:
        assert len(e.messages) > 0
    else:
        assert False, "Expected ValidationError"


# LLM-generated content at query #22
#--------------------------

```python
def test_validate_with_error_in_child_schema():
    schema = Schema(fields={"key": Field(allow_null=False)})
    value = {"key": None}
    with pytest.raises(ValidationError):
        schema.validate(value)


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_validate_with_none_value_and_allow_null():
    schema = Schema(fields={}, allow_null=True)
    assert schema.validate(None) is None

def test_validate_with_none_value_and_not_allow_null():
    schema = Schema(fields={}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

def test_validate_with_non_dict_value():
    schema = Schema(fields={})
    try:
        schema.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"

def test_validate_with_non_string_keys():
    schema = Schema(fields={})
    try:
        schema.validate({1: "value"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

def test_validate_with_missing_required_field():
    schema = Schema(fields={"required_field": Field(required=True)})
    try:
        schema.validate({})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"

def test_validate_with_valid_data():
    schema = Schema(fields={"field1": Field(), "field2": Field()})
    result = schema.validate({"field1": "value1", "field2": "value2"})
    assert result == {"field1": "value1", "field2": "value2"}

def test_validate_with_read_only_field():
    schema = Schema(fields={"read_only_field": Field(read_only=True)})
    result = schema.validate({})
    assert result == {}

def test_validate_with_default_value():
    schema = Schema(fields={"field_with_default": Field(default="default_value")})
    result = schema.validate({})
    assert result == {"field_with_default": "default_value"}

def test_validate_with_child_validation_error():
    child_schema = Field()
    child_schema.validate = lambda x: (None, ValidationError(messages=[Message(text="error", code="error")]))
    schema = Schema(fields={"child_field": child_schema})
    try:
        schema.validate({"child_field": "invalid"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "error"


# LLM-generated content at query #2
#--------------------------

```python
def test_serialize_with_none_input():
    schema = Schema(fields={})
    assert schema.serialize(None) is None

def test_serialize_with_empty_fields():
    schema = Schema(fields={})
    assert schema.serialize({}) == {}

def test_serialize_with_dict_input():
    schema = Schema(fields={"name": Field()})
    assert schema.serialize({"name": "test"}) == {"name": "test"}

def test_serialize_with_object_input():
    class TestObj:
        def __init__(self):
            self.name = "test"
    schema = Schema(fields={"name": Field()})
    assert schema.serialize(TestObj()) == {"name": "test"}

def test_serialize_with_missing_key():
    schema = Schema(fields={"name": Field()})
    assert schema.serialize({"other": "value"}) == {}

def test_serialize_with_missing_attribute():
    class TestObj:
        pass
    schema = Schema(fields={"name": Field()})
    assert schema.serialize(TestObj()) == {}

def test_serialize_with_nested_schema():
    nested_schema = Schema(fields={"age": Field()})
    schema = Schema(fields={"user": nested_schema})
    input_data = {"user": {"age": 25}}
    expected_output = {"user": {"age": 25}}
    assert schema.serialize(input_data) == expected_output


# LLM-generated content at query #3
#--------------------------

```python
def test_validate_error_handling():
    schema = Schema(fields={"key": Field()})
    value = {"key": "invalid_value"}
    schema.fields["key"].validate_or_error = lambda x: (None, "error")
    assert schema.validate(value) == {"key": None}


# LLM-generated content at query #4
#--------------------------

```python
def test_validate_error_handling():
    schema = Schema(fields={"key": Field()})
    value = {"key": "invalid_value"}
    schema.fields["key"].validate_or_error = lambda x: (None, "error")
    result = schema.validate(value)
    assert result == {}


# LLM-generated content at query #5
#--------------------------

```python
def test_setitem_adds_new_key_value_pair():
    definitions = Definitions()
    definitions["key"] = "value"
    assert definitions["key"] == "value"

def test_setitem_raises_assertion_error_for_existing_key():
    definitions = Definitions({"key": "value"})
    try:
        definitions["key"] = "new_value"
        assert False, "Expected AssertionError"
    except AssertionError as e:
        assert str(e) == r"Definition for 'key' has already been set."


# LLM-generated content at query #6
#--------------------------

```python
def test_validate_with_null_value_and_allow_null_false():
    reference = Reference(to="test", definitions={"test": Field()}, allow_null=False)
    with pytest.raises(ValidationError) as excinfo:
        reference.validate(None)
    assert str(excinfo.value) == "May not be null."

def test_validate_with_null_value_and_allow_null_true():
    reference = Reference(to="test", definitions={"test": Field()}, allow_null=True)
    assert reference.validate(None) is None

def test_validate_with_non_null_value():
    reference = Reference(to="test", definitions={"test": Field()})
    assert reference.validate("valid_value") == "valid_value"


# LLM-generated content at query #7
#--------------------------

```python
def test_validate_with_none_value_and_allow_null():
    schema = Schema(fields={}, allow_null=True)
    assert schema.validate(None) is None

def test_validate_with_none_value_and_not_allow_null():
    schema = Schema(fields={}, allow_null=False)
    try:
        schema.validate(None)
    except ValidationError as e:
        assert e.messages[0].code == "null"

def test_validate_with_non_dict_value():
    schema = Schema(fields={})
    try:
        schema.validate("not a dict")
    except ValidationError as e:
        assert e.messages[0].code == "type"

def test_validate_with_invalid_key_type():
    schema = Schema(fields={})
    try:
        schema.validate({123: "value"})
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

def test_validate_with_missing_required_field():
    required_field = Field(required=True)
    schema = Schema(fields={"required_field": required_field})
    try:
        schema.validate({})
    except ValidationError as e:
        assert e.messages[0].code == "required"

def test_validate_with_read_only_field():
    read_only_field = Field(read_only=True)
    schema = Schema(fields={"read_only_field": read_only_field})
    result = schema.validate({"read_only_field": "value"})
    assert "read_only_field" not in result

def test_validate_with_default_value():
    field_with_default = Field(default="default_value")
    schema = Schema(fields={"field_with_default": field_with_default})
    result = schema.validate({})
    assert result["field_with_default"] == "default_value"

def test_validate_with_valid_input():
    field = Field()
    schema = Schema(fields={"field": field})
    result = schema.validate({"field": "value"})
    assert result["field"] == "value"

def test_validate_with_child_validation_error():
    child_field = Field()
    schema = Schema(fields={"child_field": child_field})
    try:
        schema.validate({"child_field": None})
    except ValidationError as e:
        assert e.messages[0].index == ["child_field"]


# LLM-generated content at query #8
#--------------------------

```python
def test_serialize_with_none_input():
    schema = Schema(fields={})
    assert schema.serialize(None) is None

def test_serialize_with_empty_fields():
    schema = Schema(fields={})
    assert schema.serialize({}) == {}

def test_serialize_with_dict_input():
    field = Field()
    schema = Schema(fields={"key": field})
    assert schema.serialize({"key": "value"}) == {"key": field.serialize("value")}

def test_serialize_with_object_input():
    class TestObject:
        def __init__(self):
            self.key = "value"
    field = Field()
    schema = Schema(fields={"key": field})
    obj = TestObject()
    assert schema.serialize(obj) == {"key": field.serialize("value")}

def test_serialize_with_missing_key_in_dict():
    field = Field()
    schema = Schema(fields={"key": field})
    assert schema.serialize({"other_key": "value"}) == {}

def test_serialize_with_missing_attribute_in_object():
    class TestObject:
        pass
    field = Field()
    schema = Schema(fields={"key": field})
    obj = TestObject()
    assert schema.serialize(obj) == {}

def test_serialize_with_multiple_fields():
    field1 = Field()
    field2 = Field()
    schema = Schema(fields={"key1": field1, "key2": field2})
    assert schema.serialize({"key1": "value1", "key2": "value2"}) == {
        "key1": field1.serialize("value1"),
        "key2": field2.serialize("value2")
    }


# LLM-generated content at query #9
#--------------------------

```python
def test_validate_with_none_and_allow_null():
    schema = Schema(fields={}, allow_null=True)
    assert schema.validate(None) is None

def test_validate_with_none_and_not_allow_null():
    schema = Schema(fields={}, allow_null=False)
    try:
        schema.validate(None)
    except ValidationError as e:
        assert e.messages[0].code == "null"

def test_validate_with_non_dict_input():
    schema = Schema(fields={})
    try:
        schema.validate("not a dict")
    except ValidationError as e:
        assert e.messages[0].code == "type"

def test_validate_with_non_string_key():
    schema = Schema(fields={})
    try:
        schema.validate({123: "value"})
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

def test_validate_with_missing_required_field():
    required_field = Field(required=True)
    schema = Schema(fields={"required_field": required_field})
    try:
        schema.validate({})
    except ValidationError as e:
        assert e.messages[0].code == "required"

def test_validate_with_read_only_field():
    read_only_field = Field(read_only=True)
    schema = Schema(fields={"read_only_field": read_only_field})
    result = schema.validate({"read_only_field": "value"})
    assert "read_only_field" not in result

def test_validate_with_default_value():
    field_with_default = Field(default="default_value")
    schema = Schema(fields={"field_with_default": field_with_default})
    result = schema.validate({})
    assert result["field_with_default"] == "default_value"

def test_validate_with_valid_input():
    field = Field()
    schema = Schema(fields={"field": field})
    result = schema.validate({"field": "value"})
    assert result["field"] == "value"

def test_validate_with_child_validation_error():
    child_field = Field()
    schema = Schema(fields={"child_field": child_field})
    try:
        schema.validate({"child_field": None})
    except ValidationError as e:
        assert e.messages[0].code == "null"


# LLM-generated content at query #10
#--------------------------

```python
def test_serialize_with_missing_key_raises_keyerror():
    schema = Schema(fields={})
    obj = {"key": "value"}
    assert schema.serialize(obj) == {}


# LLM-generated content at query #11
#--------------------------

```python
def test_setitem_adds_new_key_value_pair():
    definitions = Definitions()
    definitions["key"] = "value"
    assert definitions["key"] == "value"

def test_setitem_raises_assertion_error_when_key_exists():
    definitions = Definitions({"key": "value"})
    try:
        definitions["key"] = "new_value"
        assert False, "Expected AssertionError"
    except AssertionError as e:
        assert str(e) == r"Definition for 'key' has already been set."


# LLM-generated content at query #12
#--------------------------

```python
def test_setitem_adds_new_key_value_pair():
    definitions = Definitions()
    definitions["key"] = "value"
    assert definitions["key"] == "value"

def test_setitem_raises_assertion_error_for_existing_key():
    definitions = Definitions({"key": "value"})
    try:
        definitions["key"] = "new_value"
        assert False, "Expected AssertionError"
    except AssertionError as e:
        assert str(e) == r"Definition for 'key' has already been set."


# LLM-generated content at query #13
#--------------------------

```python
def test_validate_with_none_value_and_allow_null():
    schema = Schema(fields={}, allow_null=True)
    assert schema.validate(None) is None

def test_validate_with_none_value_and_not_allow_null():
    schema = Schema(fields={}, allow_null=False)
    try:
        schema.validate(None)
    except ValidationError as e:
        assert e.messages[0].code == "null"
    else:
        assert False, "Expected ValidationError"

def test_validate_with_non_dict_value():
    schema = Schema(fields={})
    try:
        schema.validate("not a dict")
    except ValidationError as e:
        assert e.messages[0].code == "type"
    else:
        assert False, "Expected ValidationError"

def test_validate_with_non_string_key():
    schema = Schema(fields={})
    try:
        schema.validate({123: "value"})
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"
    else:
        assert False, "Expected ValidationError"

def test_validate_with_missing_required_field():
    required_field = Field(required=True)
    schema = Schema(fields={"required_field": required_field})
    try:
        schema.validate({})
    except ValidationError as e:
        assert e.messages[0].code == "required"
    else:
        assert False, "Expected ValidationError"

def test_validate_with_read_only_field():
    read_only_field = Field(read_only=True)
    schema = Schema(fields={"read_only_field": read_only_field})
    result = schema.validate({"read_only_field": "value"})
    assert "read_only_field" not in result

def test_validate_with_default_value():
    field_with_default = Field(default="default_value")
    schema = Schema(fields={"field_with_default": field_with_default})
    result = schema.validate({})
    assert result["field_with_default"] == "default_value"

def test_validate_with_valid_data():
    field = Field()
    schema = Schema(fields={"field": field})
    result = schema.validate({"field": "value"})
    assert result == {"field": "value"}

def test_validate_with_child_validation_error():
    child_field = Field()
    schema = Schema(fields={"child_field": child_field})
    try:
        schema.validate({"child_field": "invalid_value"})
    except ValidationError as e:
        assert e.messages[0].index == ["child_field"]
    else:
        assert False, "Expected ValidationError"


# LLM-generated content at query #14
#--------------------------

```python
def test_setitem_raises_assertion_error_when_key_exists():
    definitions = Definitions()
    definitions["existing_key"] = "value"
    try:
        definitions["existing_key"] = "new_value"
        assert False, "Expected AssertionError"
    except AssertionError as e:
        assert str(e) == r"Definition for 'existing_key' has already been set."


# LLM-generated content at query #15
#--------------------------

```python
def test_validate_child_schema_error():
    schema = Schema(fields={"key": Field()})
    schema.fields["key"].validate_or_error = lambda x: (None, True)
    assert schema.validate({"key": "value"}) == {"key": "value"}


# LLM-generated content at query #16
#--------------------------

```python
def test_serialize_with_none_input():
    schema = Schema(fields={})
    assert schema.serialize(None) is None

def test_serialize_with_empty_dict_and_empty_fields():
    schema = Schema(fields={})
    assert schema.serialize({}) == {}

def test_serialize_with_dict_input():
    field = Field()
    field.serialize = lambda x: x * 2
    schema = Schema(fields={"a": field, "b": field})
    assert schema.serialize({"a": 1, "b": 2}) == {"a": 2, "b": 4}

def test_serialize_with_object_input():
    class MockObject:
        def __init__(self):
            self.a = 1
            self.b = 2

    field = Field()
    field.serialize = lambda x: x * 2
    schema = Schema(fields={"a": field, "b": field})
    obj = MockObject()
    assert schema.serialize(obj) == {"a": 2, "b": 4}

def test_serialize_with_missing_keys_in_dict():
    field = Field()
    field.serialize = lambda x: x * 2
    schema = Schema(fields={"a": field, "b": field})
    assert schema.serialize({"a": 1}) == {"a": 2}

def test_serialize_with_missing_attributes_in_object():
    class MockObject:
        def __init__(self):
            self.a = 1

    field = Field()
    field.serialize = lambda x: x * 2
    schema = Schema(fields={"a": field, "b": field})
    obj = MockObject()
    assert schema.serialize(obj) == {"a": 2}


# LLM-generated content at query #17
#--------------------------

```python
def test_validate_adds_validated_child_value_when_no_error():
    class MockField:
        def __init__(self, read_only=False, default=None):
            self.read_only = read_only
            self._default = default

        def has_default(self):
            return self._default is not None

        def get_default_value(self):
            return self._default

        def validate_or_error(self, value):
            return value, None

    schema = Schema(fields={"key": MockField()})
    result = schema.validate({"key": "value"})
    assert result == {"key": "value"}


# LLM-generated content at query #18
#--------------------------

```python
def test_serialize_with_none_input():
    schema = Schema(fields={})
    assert schema.serialize(None) is None

def test_serialize_with_empty_fields():
    schema = Schema(fields={})
    assert schema.serialize({}) == {}

def test_serialize_with_mapping_input():
    field = Field()
    schema = Schema(fields={"key1": field, "key2": field})
    obj = {"key1": "value1", "key2": "value2"}
    assert schema.serialize(obj) == {"key1": "value1", "key2": "value2"}

def test_serialize_with_non_mapping_input():
    field = Field()
    schema = Schema(fields={"key1": field, "key2": field})

    class Obj:
        def __init__(self):
            self.key1 = "value1"
            self.key2 = "value2"

    obj = Obj()
    assert schema.serialize(obj) == {"key1": "value1", "key2": "value2"}

def test_serialize_with_missing_keys():
    field = Field()
    schema = Schema(fields={"key1": field, "key2": field})
    obj = {"key1": "value1"}
    assert schema.serialize(obj) == {"key1": "value1"}

def test_serialize_with_nested_schema():
    inner_field = Field()
    inner_schema = Schema(fields={"inner_key": inner_field})
    outer_schema = Schema(fields={"outer_key": inner_schema})
    obj = {"outer_key": {"inner_key": "inner_value"}}
    assert outer_schema.serialize(obj) == {"outer_key": {"inner_key": "inner_value"}}


# LLM-generated content at query #19
#--------------------------

```python
def test_validate_with_none_value_and_allow_null():
    schema = Schema(fields={}, allow_null=True)
    assert schema.validate(None) is None

def test_validate_with_none_value_and_not_allow_null():
    schema = Schema(fields={}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

def test_validate_with_non_dict_value():
    schema = Schema(fields={})
    try:
        schema.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "type"

def test_validate_with_non_string_key():
    schema = Schema(fields={})
    try:
        schema.validate({1: "value"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

def test_validate_with_missing_required_field():
    schema = Schema(fields={"required_field": Field(required=True)})
    try:
        schema.validate({})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"

def test_validate_with_read_only_field():
    schema = Schema(fields={"read_only_field": Field(read_only=True)})
    result = schema.validate({"read_only_field": "value"})
    assert "read_only_field" not in result

def test_validate_with_default_value():
    schema = Schema(fields={"default_field": Field(default="default_value")})
    result = schema.validate({})
    assert result["default_field"] == "default_value"

def test_validate_with_valid_data():
    schema = Schema(fields={"field1": Field(), "field2": Field()})
    result = schema.validate({"field1": "value1", "field2": "value2"})
    assert result == {"field1": "value1", "field2": "value2"}

def test_validate_with_child_validation_error():
    child_schema = Field()
    child_schema.validate = lambda x: (None, ValidationError(messages=[Message(text="error", code="child_error")]))
    schema = Schema(fields={"child_field": child_schema})
    try:
        schema.validate({"child_field": "invalid_value"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "child_error"


# LLM-generated content at query #20
#--------------------------

```python
def test_setitem_adds_new_key_value_pair():
    definitions = Definitions()
    definitions["key"] = "value"
    assert definitions["key"] == "value"

def test_setitem_raises_assertion_error_for_existing_key():
    definitions = Definitions({"key": "value"})
    try:
        definitions["key"] = "new_value"
        assert False, "Expected AssertionError"
    except AssertionError as e:
        assert str(e) == r"Definition for 'key' has already been set."


# LLM-generated content at query #21
#--------------------------

```python
def test_validate_without_error_adds_value_to_validated():
    schema = Schema(fields={})
    schema.fields["test"] = Field()
    schema.fields["test"].validate_or_error = lambda x: (x, None)
    value = {"test": "value"}
    assert schema.validate(value) == {"test": "value"}


# LLM-generated content at query #22
#--------------------------

```python
def test_serialize_with_none_input():
    schema = Schema(fields={})
    assert schema.serialize(None) is None

def test_serialize_with_empty_fields():
    schema = Schema(fields={})
    assert schema.serialize({}) == {}

def test_serialize_with_mapping_input():
    schema = Schema(fields={"name": StringField(), "age": IntegerField()})
    input_data = {"name": "Alice", "age": 30}
    assert schema.serialize(input_data) == {"name": "Alice", "age": 30}

def test_serialize_with_non_mapping_input():
    class User:
        def __init__(self):
            self.name = "Bob"
            self.age = 25
    schema = Schema(fields={"name": StringField(), "age": IntegerField()})
    user = User()
    assert schema.serialize(user) == {"name": "Bob", "age": 25}

def test_serialize_with_missing_keys():
    schema = Schema(fields={"name": StringField(), "age": IntegerField()})
    input_data = {"name": "Charlie"}
    assert schema.serialize(input_data) == {"name": "Charlie"}

def test_serialize_with_extra_keys():
    schema = Schema(fields={"name": StringField()})
    input_data = {"name": "Dave", "extra": "ignored"}
    assert schema.serialize(input_data) == {"name": "Dave"}

def test_serialize_with_nested_schema():
    nested_schema = Schema(fields={"street": StringField(), "city": StringField()})
    schema = Schema(fields={"name": StringField(), "address": nested_schema})
    input_data = {"name": "Eve", "address": {"street": "123 Main St", "city": "Metropolis"}}
    assert schema.serialize(input_data) == {"name": "Eve", "address": {"street": "123 Main St", "city": "Metropolis"}}


# LLM-generated content at query #23
#--------------------------

```python
def test_validate_error_case():
    schema = Schema(fields={"key": Field()})
    value = {"key": "invalid_value"}
    schema.fields["key"].validate_or_error = lambda x: (None, "error")
    assert schema.validate(value) is None


# LLM-generated content at query #24
#--------------------------

```python
def test_setitem_raises_assertion_error_when_key_exists():
    definitions = Definitions()
    definitions['key'] = 'value'
    try:
        definitions['key'] = 'new_value'
        assert False, "Expected AssertionError"
    except AssertionError as e:
        assert str(e) == r"Definition for 'key' has already been set."


# LLM-generated content at query #25
#--------------------------

```python
def test_serialize_with_none_obj():
    schema = Schema(fields={})
    assert schema.serialize(None) is None


# LLM-generated content at query #26
#--------------------------

```python
def test_validate_error_path():
    schema = Schema(fields={"key": Field()})
    value = {"key": "invalid_value"}
    schema.fields["key"].validate_or_error = lambda x: (None, "error")
    assert schema.validate(value) == ValidationError(messages=[Message(text="error", code=None, index=["key"])])


# LLM-generated content at query #27
#--------------------------

```python
def test_validate_adds_validated_child_value_when_no_error():
    schema = Schema(fields={"name": Field()})
    value = {"name": "test"}
    schema.fields["name"].validate_or_error = lambda x: (x, None)
    assert schema.validate(value) == {"name": "test"}


# LLM-generated content at query #28
#--------------------------

```python
def test_validate_with_none_value_and_allow_null():
    schema = Schema(fields={}, allow_null=True)
    result = schema.validate(None)
    assert result is None

def test_validate_with_none_value_and_not_allow_null():
    schema = Schema(fields={}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Expected validation error"
    except ValidationError as e:
        assert e.messages[0].code == "null"

def test_validate_with_non_dict_value():
    schema = Schema(fields={})
    try:
        schema.validate("not a dict")
        assert False, "Expected validation error"
    except ValidationError as e:
        assert e.messages[0].code == "type"

def test_validate_with_invalid_key_type():
    schema = Schema(fields={})
    try:
        schema.validate({123: "value"})
        assert False, "Expected validation error"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

def test_validate_with_missing_required_field():
    required_field = Field(required=True)
    schema = Schema(fields={"required_field": required_field})
    try:
        schema.validate({})
        assert False, "Expected validation error"
    except ValidationError as e:
        assert e.messages[0].code == "required"

def test_validate_with_valid_data():
    field = Field()
    schema = Schema(fields={"field": field})
    result = schema.validate({"field": "value"})
    assert result == {"field": "value"}

def test_validate_with_default_value():
    field = Field(default="default_value")
    schema = Schema(fields={"field": field})
    result = schema.validate({})
    assert result == {"field": "default_value"}

def test_validate_with_read_only_field():
    read_only_field = Field(read_only=True)
    schema = Schema(fields={"read_only_field": read_only_field})
    result = schema.validate({"read_only_field": "value"})
    assert "read_only_field" not in result

def test_validate_with_child_validation_error():
    child_field = Field()
    schema = Schema(fields={"child_field": child_field})
    try:
        schema.validate({"child_field": None})
        assert False, "Expected validation error"
    except ValidationError as e:
        assert e.messages[0].code == "null"


# LLM-generated content at query #29
#--------------------------

```python
def test_serialize_with_none_obj():
    schema = Schema(fields={})
    assert schema.serialize(None) is None


# LLM-generated content at query #30
#--------------------------

```python
def test_validate_without_error():
    schema = Schema(fields={"key": Field()})
    value = {"key": "value"}
    assert schema.validate(value) == {"key": "value"}


# LLM-generated content at query #31
#--------------------------

```python
def test_validate_with_none_and_allow_null():
    schema = Schema(fields={}, allow_null=True)
    assert schema.validate(None) is None

def test_validate_with_none_and_not_allow_null():
    schema = Schema(fields={}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Expected validation error"
    except ValidationError as e:
        assert e.messages[0].code == "null"

def test_validate_with_non_dict_value():
    schema = Schema(fields={})
    try:
        schema.validate("not a dict")
        assert False, "Expected validation error"
    except ValidationError as e:
        assert e.messages[0].code == "type"

def test_validate_with_non_string_key():
    schema = Schema(fields={})
    try:
        schema.validate({123: "value"})
        assert False, "Expected validation error"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

def test_validate_with_missing_required_field():
    required_field = Field(required=True)
    schema = Schema(fields={"required_field": required_field})
    try:
        schema.validate({})
        assert False, "Expected validation error"
    except ValidationError as e:
        assert e.messages[0].code == "required"

def test_validate_with_valid_data():
    field = Field()
    schema = Schema(fields={"key": field})
    assert schema.validate({"key": "value"}) == {"key": "value"}

def test_validate_with_read_only_field():
    read_only_field = Field(read_only=True)
    schema = Schema(fields={"read_only": read_only_field})
    assert schema.validate({}) == {}

def test_validate_with_default_value():
    field_with_default = Field(default="default_value")
    schema = Schema(fields={"default_field": field_with_default})
    assert schema.validate({}) == {"default_field": "default_value"}

def test_validate_with_child_validation_error():
    child_field = Field()
    child_field.validate = lambda x: (None, ValidationError(messages=[Message(text="error", code="child_error")]))
    schema = Schema(fields={"child": child_field})
    try:
        schema.validate({"child": "invalid"})
        assert False, "Expected validation error"
    except ValidationError as e:
        assert e.messages[0].code == "child_error"


# LLM-generated content at query #32
#--------------------------

```python
def test_validate_error_path():
    schema = Schema(fields={"key": Field()})
    value = {"key": "invalid_value"}
    schema.fields["key"].validate_or_error = lambda x: (None, "error")
    assert schema.validate(value) == {"key": "invalid_value"}


# LLM-generated content at query #33
#--------------------------

```python
def test_serialize_with_none_obj():
    schema = Schema(fields={})
    assert schema.serialize(None) is None


