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
        schema.validate({1: "value"})
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
    result = schema.validate({"other_field": "value"})
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


# LLM-generated content at query #2
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
    assert result == {"field_with_default": "default_value"}

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
        schema.validate({"child_field": "invalid"})
    except ValidationError as e:
        assert e.messages[0].code == "error"
    else:
        assert False, "Expected ValidationError"


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

def test_validate_with_valid_input():
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
    child_field.validate = lambda x: (None, ValidationError(messages=[Message(text="error", code="error")]))
    schema = Schema(fields={"child_field": child_field})
    try:
        schema.validate({"child_field": "invalid"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "error"


# LLM-generated content at query #4
#--------------------------

```python
def test_validate_error_messages_when_child_validation_fails():
    schema = Schema(fields={"name": String()})
    value = {"name": 123}
    try:
        schema.validate(value)
    except ValidationError as e:
        assert e.messages() == [Message(text="Must be a string.", code="type", index=["name"])]
    else:
        assert False, "Expected ValidationError"


# LLM-generated content at query #5
#--------------------------

```python
def test_validate_with_null_value_and_allow_null():
    ref = Reference(to="test", definitions={"test": Field()}, allow_null=True)
    assert ref.validate(None) is None

def test_validate_with_null_value_and_not_allow_null():
    ref = Reference(to="test", definitions={"test": Field()}, allow_null=False)
    try:
        ref.validate(None)
        assert False, "Expected validation error"
    except ValidationError as e:
        assert str(e) == "May not be null."

def test_validate_with_non_null_value():
    ref = Reference(to="test", definitions={"test": Field()})
    assert ref.validate("valid_value") == "valid_value"


# LLM-generated content at query #6
#--------------------------

```python
def test_validate_returns_none_when_value_is_none_and_allow_null_is_true():
    schema = Schema(fields={}, allow_null=True)
    assert schema.validate(None) is None


# LLM-generated content at query #7
#--------------------------

```python
def test_validate_with_none_and_allow_null():
    schema = Schema(fields={}, allow_null=True)
    assert schema.validate(None) is None

def test_validate_with_none_and_not_allow_null():
    schema = Schema(fields={}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "null"

def test_validate_with_non_dict():
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
    required_field = Field(required=True)
    schema = Schema(fields={"required_field": required_field})
    try:
        schema.validate({})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "required"

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
    child_field.validate = lambda x: (None, ValidationError(messages=[Message(text="error", code="error")]))
    schema = Schema(fields={"child_field": child_field})
    try:
        schema.validate({"child_field": "value"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "error"


# LLM-generated content at query #8
#--------------------------

```python
def test_validate_error_messages_when_child_schema_validation_fails():
    child_field = Field(allow_null=False)
    schema = Schema(fields={"key": child_field})
    value = {"key": None}
    try:
        schema.validate(value)
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "null"


# LLM-generated content at query #9
#--------------------------

```python
def test_validate_with_null_value_and_allow_null_true():
    reference = Reference(to="test", definitions={"test": Field()}, allow_null=True)
    assert reference.validate(None) is None

def test_validate_with_null_value_and_allow_null_false():
    reference = Reference(to="test", definitions={"test": Field()}, allow_null=False)
    try:
        reference.validate(None)
        assert False, "Expected validation error"
    except ValidationError as e:
        assert str(e) == "May not be null."

def test_validate_with_non_null_value():
    reference = Reference(to="test", definitions={"test": Field()}, allow_null=False)
    assert reference.validate("valid_value") == "valid_value"


# LLM-generated content at query #10
#--------------------------

```python
def test_validate_returns_none_when_value_is_none_and_allow_null_is_true():
    ref = Reference(to="test", definitions={}, allow_null=True)
    assert ref.validate(None) is None


# LLM-generated content at query #11
#--------------------------

```python
def test_validate_error_case():
    schema = Schema(fields={})
    value = {"key": "value"}
    error = ValidationError(messages=[Message(text="Error", code="error", index=["key"])])
    with patch.object(schema, 'validate_or_error', return_value=("value", error)):
        assert schema.validate(value) is None


# LLM-generated content at query #12
#--------------------------

```python
def test_validate_returns_none_when_value_is_none_and_allow_null_is_true():
    reference = Reference(to="test", definitions={}, allow_null=True)
    assert reference.validate(None) is None


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
    assert result == {"field_with_default": "default_value"}

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
        assert e.messages[0].code == "invalid"
    else:
        assert False, "Expected ValidationError"


# LLM-generated content at query #14
#--------------------------

```python
def test_validate_error_raises_validation_error():
    schema = Schema(fields={})
    value = {"key": "value"}
    error = ValidationError(messages=[Message(text="error", code="error", index=["key"])])
    with patch.object(schema, "validate_or_error", return_value=("value", error)):
        assert_raises(ValidationError, schema.validate, value)


# LLM-generated content at query #15
#--------------------------

```python
def test_validate_with_null_value_and_allow_null():
    reference = Reference(to="test", definitions={"test": Field()}, allow_null=True)
    assert reference.validate(None) is None

def test_validate_with_null_value_and_not_allow_null():
    reference = Reference(to="test", definitions={"test": Field()}, allow_null=False)
    try:
        reference.validate(None)
    except ValidationError as e:
        assert str(e) == "May not be null."
    else:
        assert False, "Expected ValidationError"

def test_validate_with_non_null_value():
    reference = Reference(to="test", definitions={"test": Field()})
    assert reference.validate("test_value") == "test_value"


# LLM-generated content at query #16
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

def test_validate_with_valid_input():
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


# LLM-generated content at query #17
#--------------------------

```python
def test_validate_with_null_value_and_allow_null_true():
    reference = Reference(to="test", definitions={"test": Field()}, allow_null=True)
    assert reference.validate(None) is None

def test_validate_with_null_value_and_allow_null_false():
    reference = Reference(to="test", definitions={"test": Field()}, allow_null=False)
    try:
        reference.validate(None)
        assert False, "Expected validation error"
    except ValidationError as e:
        assert str(e) == "May not be null."

def test_validate_with_non_null_value():
    reference = Reference(to="test", definitions={"test": Field()})
    assert reference.validate("valid_value") == "valid_value"


# LLM-generated content at query #18
#--------------------------

```python
def test_validate_returns_none_when_value_is_none_and_allow_null_is_true():
    schema = Schema(fields={}, allow_null=True)
    assert schema.validate(None) is None


# LLM-generated content at query #19
#--------------------------

```python
def test_validate_returns_none_when_value_is_none_and_allow_null_is_true():
    reference = Reference(to="test", definitions={}, allow_null=True)
    assert reference.validate(None) is None


# LLM-generated content at query #20
#--------------------------

```python
def test_validate_error_path():
    schema = Schema(fields={"key": Field()})
    with pytest.raises(ValidationError):
        schema.validate({"key": "invalid_value"})


# LLM-generated content at query #21
#--------------------------

```python
def test_validate_returns_none_when_value_is_none_and_allow_null_is_true():
    schema = Schema(fields={}, allow_null=True)
    assert schema.validate(None) is None


# LLM-generated content at query #22
#--------------------------

```python
def test_validate_returns_none_when_value_is_none_and_allow_null_is_true():
    reference = Reference(to="test", definitions={}, allow_null=True)
    assert reference.validate(None) is None


# LLM-generated content at query #23
#--------------------------

```python
def test_validate_returns_none_when_value_is_none_and_allow_null_is_true():
    schema = Schema(fields={}, allow_null=True)
    assert schema.validate(None) is None


# LLM-generated content at query #24
#--------------------------

```python
def test_validate_error_handling():
    schema = Schema(fields={"key": Field()})
    value = {"key": "invalid_value"}
    schema.fields["key"].validate_or_error = lambda x: (None, "error_message")
    result = schema.validate(value)
    assert result == {}


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_validate_with_null_value_and_allow_null():
    schema = Schema(fields={}, allow_null=True)
    assert schema.validate(None) is None

def test_validate_with_null_value_and_not_allow_null():
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
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) > 0


# LLM-generated content at query #2
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

def test_validate_with_non_dict():
    schema = Schema(fields={})
    try:
        schema.validate("not a dict")
        assert False, "Expected validation error"
    except ValidationError as e:
        assert e.messages[0].code == "type"

def test_validate_with_non_string_key():
    schema = Schema(fields={})
    try:
        schema.validate({1: "value"})
        assert False, "Expected validation error"
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

def test_validate_with_missing_required_field():
    fields = {"required_field": Field(required=True)}
    schema = Schema(fields=fields)
    try:
        schema.validate({})
        assert False, "Expected validation error"
    except ValidationError as e:
        assert e.messages[0].code == "required"

def test_validate_with_read_only_field():
    fields = {"read_only_field": Field(read_only=True)}
    schema = Schema(fields=fields)
    result = schema.validate({"read_only_field": "value"})
    assert "read_only_field" not in result

def test_validate_with_default_value():
    fields = {"default_field": Field(default="default")}
    schema = Schema(fields=fields)
    result = schema.validate({})
    assert result["default_field"] == "default"

def test_validate_with_valid_data():
    fields = {"field1": Field(), "field2": Field()}
    schema = Schema(fields=fields)
    result = schema.validate({"field1": "value1", "field2": "value2"})
    assert result == {"field1": "value1", "field2": "value2"}


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

def test_validate_with_valid_input():
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
    child_field.validate = lambda x: (None, ValidationError(messages=[Message(text="error", code="child_error")]))
    schema = Schema(fields={"child_field": child_field})
    try:
        schema.validate({"child_field": "invalid"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "child_error"


# LLM-generated content at query #4
#--------------------------

```python
def test_validate_error_case():
    field = Field()
    child_schema = Schema(fields={"key": field})
    value = {"key": "invalid_value"}
    child_value, error = child_schema.fields["key"].validate_or_error(value["key"])
    assert error is not None


# LLM-generated content at query #5
#--------------------------

```python
def test_validate_with_none_and_allow_null():
    reference = Reference(to="test", definitions={"test": None}, allow_null=True)
    assert reference.validate(None) is None

def test_validate_with_none_and_not_allow_null():
    reference = Reference(to="test", definitions={"test": None}, allow_null=False)
    try:
        reference.validate(None)
        assert False, "Expected validation error"
    except ValidationError as e:
        assert str(e) == "May not be null."

def test_validate_with_valid_value():
    mock_target = Mock()
    mock_target.validate.return_value = "validated_value"
    reference = Reference(to="test", definitions={"test": mock_target})
    assert reference.validate("test_value") == "validated_value"


# LLM-generated content at query #6
#--------------------------

```python
def test_validate_with_null_value_and_allow_null_true():
    reference = Reference(to="test", definitions={"test": Field()}, allow_null=True)
    assert reference.validate(None) is None

def test_validate_with_null_value_and_allow_null_false():
    reference = Reference(to="test", definitions={"test": Field()}, allow_null=False)
    try:
        reference.validate(None)
        assert False, "Expected validation error"
    except ValidationError as e:
        assert str(e) == "null"

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
    schema = Schema(fields={"required_field": Field(required=True)})
    try:
        schema.validate({})
        assert False, "Expected validation error"
    except ValidationError as e:
        assert e.messages[0].code == "required"

def test_validate_with_valid_data():
    schema = Schema(fields={"field1": Field(), "field2": Field()})
    result = schema.validate({"field1": "value1", "field2": "value2"})
    assert result == {"field1": "value1", "field2": "value2"}

def test_validate_with_default_value():
    schema = Schema(fields={"field1": Field(default="default_value")})
    result = schema.validate({})
    assert result == {"field1": "default_value"}

def test_validate_with_read_only_field():
    schema = Schema(fields={"field1": Field(read_only=True)})
    result = schema.validate({"field1": "value1"})
    assert result == {}

def test_validate_with_child_validation_error():
    child_schema = Field()
    child_schema.validate = lambda x: (None, ValidationError(messages=[Message(text="error", code="error")]))
    schema = Schema(fields={"child_field": child_schema})
    try:
        schema.validate({"child_field": "invalid"})
        assert False, "Expected validation error"
    except ValidationError as e:
        assert e.messages[0].code == "error"


# LLM-generated content at query #8
#--------------------------

```python
def test_validate_error_case():
    schema = Schema(fields={})
    value = {"key": "value"}
    error = ValidationError(messages=[Message(text="error", code="error", index=[])])
    with patch.object(schema, 'validate_or_error', return_value=("value", error)):
        assert schema.validate(value) == ValidationError


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
        schema.validate({"child_field": "invalid_value"})
    except ValidationError as e:
        assert e.messages[0].index == ["child_field"]


# LLM-generated content at query #10
#--------------------------

```python
def test_validate_returns_none_when_value_is_none_and_allow_null_is_true():
    schema = Schema(fields={}, allow_null=True)
    result = schema.validate(None)
    assert result is None


# LLM-generated content at query #11
#--------------------------

```python
def test_validate_returns_none_when_value_is_none_and_allow_null_is_true():
    reference = Reference(to="test", definitions={}, allow_null=True)
    result = reference.validate(None)
    assert result is None


# LLM-generated content at query #12
#--------------------------

```python
def test_validate_when_child_schema_validation_succeeds():
    schema = Schema(fields={"name": Field()})
    value = {"name": "test"}
    result = schema.validate(value)
    assert "name" in result


# LLM-generated content at query #13
#--------------------------

```python
def test_validate_error_path_with_error():
    schema = Schema(fields={"key": Field()})
    value = {"key": "invalid_value"}
    schema.fields["key"].validate_or_error = lambda x: (None, ValidationError(messages=[Message(text="error", code="error")]))
    try:
        schema.validate(value)
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "error"


# LLM-generated content at query #14
#--------------------------

```python
def test_validate_with_none_value_and_allow_null_true():
    reference = Reference(to="some_type", definitions={"some_type": Field()}, allow_null=True)
    assert reference.validate(None) is None

def test_validate_with_none_value_and_allow_null_false():
    reference = Reference(to="some_type", definitions={"some_type": Field()}, allow_null=False)
    try:
        reference.validate(None)
    except ValidationError as e:
        assert str(e) == "May not be null."
    else:
        assert False, "Expected ValidationError"

def test_validate_with_non_none_value():
    mock_field = Field()
    mock_field.validate = lambda x: x.upper()
    reference = Reference(to="some_type", definitions={"some_type": mock_field})
    assert reference.validate("test") == "TEST"


# LLM-generated content at query #15
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
    except ValidationError as e:
        assert e.messages[0].code == "child_error"
    else:
        assert False, "Expected ValidationError"


# LLM-generated content at query #16
#--------------------------

```python
def test_validate_with_valid_child_value():
    schema = Schema(fields={"key": Field()})
    value = {"key": "valid_value"}
    result = schema.validate(value)
    assert result == {"key": "valid_value"}


# LLM-generated content at query #17
#--------------------------

```python
def test_validate_error_case():
    schema = Schema(fields={"key": Field()})
    value = {"key": "invalid_value"}
    with pytest.raises(ValidationError):
        schema.validate(value)


# LLM-generated content at query #18
#--------------------------

```python
def test_validate_with_none_value_and_allow_null_true():
    schema = Schema(fields={}, allow_null=True)
    assert schema.validate(None) is None


# LLM-generated content at query #19
#--------------------------

```python
def test_validate_with_valid_child_schema():
    class MockField:
        def __init__(self, read_only=False, default=None):
            self.read_only = read_only
            self.default = default

        def has_default(self):
            return self.default is not None

        def get_default_value(self):
            return self.default

        def validate_or_error(self, value):
            return value, None

    fields = {"key1": MockField()}
    schema = Schema(fields=fields, allow_null=False)
    value = {"key1": "valid_value"}

    result = schema.validate(value)

    assert result == {"key1": "valid_value"}


# LLM-generated content at query #20
#--------------------------

```python
def test_validate_returns_none_when_value_is_none_and_allow_null_is_true():
    reference = Reference(to="test", definitions={}, allow_null=True)
    assert reference.validate(None) is None


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

def test_validate_with_non_dict_value():
    schema = Schema(fields={})
    try:
        schema.validate("not a dict")
    except ValidationError as e:
        assert e.messages[0].code == "type"

def test_validate_with_non_string_key():
    schema = Schema(fields={})
    try:
        schema.validate({1: "value"})
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

def test_validate_with_missing_required_field():
    schema = Schema(fields={"required_field": Field(required=True)})
    try:
        schema.validate({})
    except ValidationError as e:
        assert e.messages[0].code == "required"

def test_validate_with_read_only_field():
    schema = Schema(fields={"read_only_field": Field(read_only=True)})
    assert schema.validate({}) == {}

def test_validate_with_default_value():
    schema = Schema(fields={"default_field": Field(default="default_value")})
    assert schema.validate({}) == {"default_field": "default_value"}

def test_validate_with_valid_data():
    schema = Schema(fields={"field1": Field(), "field2": Field()})
    assert schema.validate({"field1": "value1", "field2": "value2"}) == {"field1": "value1", "field2": "value2"}

def test_validate_with_child_validation_error():
    child_schema = Field()
    child_schema.validate = lambda x: (None, ValidationError(messages=[Message(text="error", code="error")]))
    schema = Schema(fields={"child_field": child_schema})
    try:
        schema.validate({"child_field": "invalid"})
    except ValidationError as e:
        assert e.messages[0].code == "error"


# LLM-generated content at query #22
#--------------------------

```python
def test_validate_with_none_value_and_allow_null_true():
    schema = Schema(fields={}, allow_null=True)
    assert schema.validate(None) is None

def test_validate_with_none_value_and_allow_null_false():
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

def test_validate_with_invalid_key_type():
    schema = Schema(fields={"valid_key": Field()})
    try:
        schema.validate({123: "value"})
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
    schema = Schema(fields={"key1": Field(), "key2": Field()})
    result = schema.validate({"key1": "value1", "key2": "value2"})
    assert result == {"key1": "value1", "key2": "value2"}

def test_validate_with_read_only_field():
    schema = Schema(fields={"read_only_field": Field(read_only=True)})
    result = schema.validate({"other_field": "value"})
    assert result == {}

def test_validate_with_default_value():
    schema = Schema(fields={"field_with_default": Field(default="default_value")})
    result = schema.validate({})
    assert result == {"field_with_default": "default_value"}

def test_validate_with_child_validation_error():
    child_field = Field()
    child_field.validate = lambda x: (None, ValidationError(messages=[Message(text="error", code="child_error")]))
    schema = Schema(fields={"child_field": child_field})
    try:
        schema.validate({"child_field": "invalid"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert e.messages[0].code == "child_field.child_error"


# LLM-generated content at query #23
#--------------------------

```python
def test_validate_returns_none_when_value_is_none_and_allow_null_is_true():
    schema = Schema(fields={}, allow_null=True)
    assert schema.validate(None) is None


# LLM-generated content at query #24
#--------------------------

```python
def test_validate_with_error_in_child_schema():
    child_schema = Field()
    child_schema.validate_or_error = lambda x: (None, ValidationError(messages=[Message(text="error", code="error")]))
    schema = Schema(fields={"key": child_schema})
    with pytest.raises(ValidationError):
        schema.validate({"key": "value"})


# LLM-generated content at query #25
#--------------------------

```python
def test_validate_returns_none_when_value_is_none_and_allow_null_is_true():
    reference = Reference(to="test", definitions={"test": Field()})
    reference.allow_null = True
    assert reference.validate(None) is None


# LLM-generated content at query #26
#--------------------------

```python
def test_validate_with_null_value_and_allow_null_true():
    reference = Reference(to="test", definitions={"test": None}, allow_null=True)
    assert reference.validate(None) is None

def test_validate_with_null_value_and_allow_null_false():
    reference = Reference(to="test", definitions={"test": None}, allow_null=False)
    try:
        reference.validate(None)
        assert False, "Expected validation error"
    except ValidationError as e:
        assert str(e) == "May not be null."

def test_validate_with_non_null_value():
    target = Mock()
    target.validate.return_value = "validated"
    reference = Reference(to="test", definitions={"test": target})
    assert reference.validate("value") == "validated"
    target.validate.assert_called_once_with("value")


# LLM-generated content at query #27
#--------------------------

```python
def test_validate_returns_none_when_value_is_none_and_allow_null():
    schema = Schema(fields={}, allow_null=True)
    assert schema.validate(None) is None


# LLM-generated content at query #28
#--------------------------

```python
def test_validate_no_error_adds_validated_value():
    schema = Schema(fields={"field1": Field()})
    mock_value = {"field1": "valid_value"}
    mock_field = schema.fields["field1"]
    mock_field.validate_or_error = lambda x: ("validated_value", None)
    result = schema.validate(mock_value)
    assert result == {"field1": "validated_value"}


# LLM-generated content at query #29
#--------------------------

```python
def test_validate_error_case():
    schema = Schema(fields={"key": Field()})
    value = {"key": "invalid_value"}
    schema.fields["key"].validate_or_error = lambda x: (None, ValidationError(messages=[Message(text="error", code="error")]))
    with pytest.raises(ValidationError):
        schema.validate(value)


# LLM-generated content at query #30
#--------------------------

```python
def test_validate_error_path():
    schema = Schema(fields={"key": Field()})
    value = {"key": "invalid_value"}
    schema.fields["key"].validate_or_error = lambda x: (None, "error")
    try:
        schema.validate(value)
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "error"
    else:
        assert False, "ValidationError not raised"


# LLM-generated content at query #31
#--------------------------

```python
def test_validate_returns_none_when_value_is_none_and_allow_null_is_true():
    schema = Schema(fields={}, allow_null=True)
    assert schema.validate(None) is None


# LLM-generated content at query #32
#--------------------------

```python
def test_validate_returns_none_when_value_is_none_and_allow_null_is_true():
    reference = Reference(to="test", definitions={}, allow_null=True)
    assert reference.validate(None) is None


# LLM-generated content at query #33
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
        schema.validate({1: "value"})
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

def test_validate_with_missing_required_field():
    schema = Schema(fields={"required_field": Field(required=True)})
    try:
        schema.validate({})
    except ValidationError as e:
        assert e.messages[0].code == "required"

def test_validate_with_read_only_field():
    schema = Schema(fields={"read_only_field": Field(read_only=True)})
    assert schema.validate({"read_only_field": "value"}) == {}

def test_validate_with_default_value():
    schema = Schema(fields={"default_field": Field(default="default_value")})
    assert schema.validate({}) == {"default_field": "default_value"}

def test_validate_with_valid_data():
    schema = Schema(fields={"field1": Field(), "field2": Field()})
    assert schema.validate({"field1": "value1", "field2": "value2"}) == {"field1": "value1", "field2": "value2"}

def test_validate_with_child_validation_error():
    child_schema = Field()
    child_schema.validate = lambda x: (None, ValidationError(messages=[Message(text="error", code="error")]))
    schema = Schema(fields={"child_field": child_schema})
    try:
        schema.validate({"child_field": "invalid"})
    except ValidationError as e:
        assert e.messages[0].code == "error"


