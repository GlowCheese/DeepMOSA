####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_validate_with_null_value_and_allow_null():
    definitions = {"target": Field(allow_null=True)}
    field = Reference(to="target", definitions=definitions, allow_null=True)
    assert field.validate(None) is None

def test_validate_with_null_value_and_not_allow_null():
    definitions = {"target": Field()}
    field = Reference(to="target", definitions=definitions)
    try:
        field.validate(None)
        assert False
    except Exception as e:
        assert str(e) == "May not be null."

def test_validate_with_non_null_value():
    definitions = {"target": Field()}
    field = Reference(to="target", definitions=definitions)
    assert field.validate("value") == "value"

def test_validate_with_target_field_validation():
    definitions = {"target": Field(allow_null=False)}
    field = Reference(to="target", definitions=definitions)
    try:
        field.validate(None)
        assert False
    except Exception as e:
        assert str(e) == "May not be null."


# LLM-generated content at query #2
#--------------------------

```python
def test_validate_null():
    schema = Schema(fields={})
    schema.allow_null = True
    result = schema.validate(None)
    assert result is None

def test_validate_null_not_allowed():
    schema = Schema(fields={})
    schema.allow_null = False
    try:
        schema.validate(None)
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "null"

def test_validate_not_dict():
    schema = Schema(fields={})
    try:
        schema.validate("not a dict")
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "type"

def test_validate_invalid_key():
    schema = Schema(fields={})
    try:
        schema.validate({1: "invalid key"})
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "invalid_key"

def test_validate_required_field_missing():
    field = Field()
    schema = Schema(fields={"required_field": field})
    try:
        schema.validate({})
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "required"

def test_validate_required_field_present():
    field = Field()
    schema = Schema(fields={"required_field": field})
    result = schema.validate({"required_field": "value"})
    assert result == {"required_field": "value"}

def test_validate_read_only_field():
    field = Field()
    field.read_only = True
    schema = Schema(fields={"read_only_field": field})
    result = schema.validate({"read_only_field": "value"})
    assert result == {}

def test_validate_field_with_default():
    field = Field()
    field.default = "default_value"
    schema = Schema(fields={"field_with_default": field})
    result = schema.validate({})
    assert result == {"field_with_default": "default_value"}

def test_validate_field_with_validation_error():
    field = Field()
    def validate(value):
        raise ValidationError(messages=[Message(text="error", code="error")])
    field.validate = validate
    schema = Schema(fields={"field_with_error": field})
    try:
        schema.validate({"field_with_error": "value"})
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "error"

def test_validate_multiple_errors():
    field1 = Field()
    field2 = Field()
    schema = Schema(fields={"field1": field1, "field2": field2})
    try:
        schema.validate({1: "invalid key", "field2": "value"})
        assert False
    except ValidationError as e:
        assert len(e.messages) == 2
        assert e.messages[0].code == "invalid_key"
        assert e.messages[1].code == "required"


# LLM-generated content at query #3
#--------------------------

```python
def test_validate_returns_none_when_value_is_none_and_allow_null_is_true():
    definitions = Definitions()
    reference = Reference(to="example", definitions=definitions, allow_null=True)
    result = reference.validate(None)
    assert result is None


# LLM-generated content at query #4
#--------------------------

```python
def test_validate_returns_none_when_value_is_none_and_allow_null_is_true():
    fields = {}
    schema = Schema(fields, allow_null=True)
    result = schema.validate(None)
    assert result is None


# LLM-generated content at query #5
#--------------------------

```python
def test_validate_returns_none_when_value_is_none_and_allow_null_is_true():
    definitions = {}
    reference = Reference(to="target", definitions=definitions, allow_null=True)
    result = reference.validate(None)
    assert result is None


# LLM-generated content at query #6
#--------------------------

```python
def test_predicate_at_line_28_evaluates_to_true():
    class MockField:
        def __init__(self, read_only):
            self.read_only = read_only

    fields = {"key1": MockField(read_only=True)}
    schema = Schema(fields=fields)
    value = {"key1": "value1"}
    schema.validate(value)


# LLM-generated content at query #7
#--------------------------

```python
def test_validate_allows_null_when_allow_null_is_true():
    definitions = {"target": Mock()}
    reference = Reference(to="target", definitions=definitions, allow_null=True)
    result = reference.validate(None)
    assert result is None

def test_validate_raises_error_when_value_is_null_and_allow_null_is_false():
    definitions = {"target": Mock()}
    reference = Reference(to="target", definitions=definitions, allow_null=False)
    try:
        reference.validate(None)
        assert False
    except ValidationError as e:
        assert str(e) == "May not be null."

def test_validate_delegates_to_target_when_value_is_not_null():
    target_mock = Mock()
    target_mock.validate.return_value = "validated_value"
    definitions = {"target": target_mock}
    reference = Reference(to="target", definitions=definitions)
    result = reference.validate("value")
    assert result == "validated_value"
    target_mock.validate.assert_called_once_with("value")


# LLM-generated content at query #8
#--------------------------

```
def test_validate_null_value_when_allow_null_is_true():
    field = Field(allow_null=True)
    schema = Schema(fields={}, allow_null=True)
    assert schema.validate(None) is None

def test_validate_null_value_when_allow_null_is_false():
    schema = Schema(fields={}, allow_null=False)
    try:
        schema.validate(None)
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "null"

def test_validate_non_dict_value():
    schema = Schema(fields={})
    try:
        schema.validate("invalid")
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "type"

def test_validate_dict_with_non_string_keys():
    schema = Schema(fields={})
    try:
        schema.validate({1: "value"})
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "invalid_key"

def test_validate_missing_required_field():
    field = Field()
    schema = Schema(fields={"field": field})
    try:
        schema.validate({})
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "required"

def test_validate_with_default_value():
    field = Field(default="default")
    schema = Schema(fields={"field": field})
    assert schema.validate({}) == {"field": "default"}

def test_validate_with_read_only_field():
    field = Field(read_only=True)
    schema = Schema(fields={"field": field})
    assert schema.validate({"field": "value"}) == {}

def test_validate_with_valid_fields():
    field = Field()
    schema = Schema(fields={"field": field})
    assert schema.validate({"field": "value"}) == {"field": "value"}

def test_validate_with_nested_errors():
    nested_field = Field()
    nested_field.validate = lambda x: (_ for _ in ()).throw(ValidationError(messages=[Message(text="error", code="error")]))
    schema = Schema(fields={"field": nested_field})
    try:
        schema.validate({"field": "value"})
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "error"


# LLM-generated content at query #9
#--------------------------

```
def test_validate_allows_null_when_allow_null_is_true():
    definitions = {"target": "some_definition"}
    field = Reference(to="target", definitions=definitions, allow_null=True)
    result = field.validate(None)
    assert result is None

def test_validate_raises_error_when_null_and_allow_null_is_false():
    definitions = {"target": "some_definition"}
    field = Reference(to="target", definitions=definitions, allow_null=False)
    try:
        field.validate(None)
        assert False, "Expected validation error"
    except Exception as e:
        assert str(e) == "May not be null."

def test_validate_delegates_to_target_validation():
    class MockField:
        def validate(self, value):
            return "validated_" + str(value)

    definitions = {"target": MockField()}
    field = Reference(to="target", definitions=definitions)
    result = field.validate("test_value")
    assert result == "validated_test_value"


# LLM-generated content at query #10
#--------------------------

```python
def test_validate_with_null_value_and_allow_null():
    field = Field(allow_null=True)
    schema = Schema({})
    assert schema.validate(None) is None

def test_validate_with_null_value_and_disallow_null():
    field = Field(allow_null=False)
    schema = Schema({})
    try:
        schema.validate(None)
    except ValidationError as e:
        assert e.messages[0].code == "null"

def test_validate_with_non_dict_value():
    field = Field()
    schema = Schema({})
    try:
        schema.validate(123)
    except ValidationError as e:
        assert e.messages[0].code == "type"

def test_validate_with_invalid_key_type():
    field = Field()
    schema = Schema({})
    try:
        schema.validate({123: "value"})
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

def test_validate_with_missing_required_field():
    field = Field()
    schema = Schema({"required_field": field})
    try:
        schema.validate({})
    except ValidationError as e:
        assert e.messages[0].code == "required"

def test_validate_with_valid_fields():
    field = Field()
    schema = Schema({"field": field})
    assert schema.validate({"field": "value"}) == {"field": "value"}

def test_validate_with_read_only_field():
    field = Field(read_only=True)
    schema = Schema({"read_only_field": field})
    assert schema.validate({"read_only_field": "value"}) == {}

def test_validate_with_field_having_default():
    field = Field(default="default_value")
    schema = Schema({"field_with_default": field})
    assert schema.validate({}) == {"field_with_default": "default_value"}

def test_validate_with_nested_field_error():
    nested_field = Field()
    nested_schema = Schema({"nested_field": nested_field})
    schema = Schema({"nested": nested_schema})
    try:
        schema.validate({"nested": {}})
    except ValidationError as e:
        assert e.messages[0].code == "required"


# LLM-generated content at query #11
#--------------------------

```python
def test_schema_validate_null_with_allow_null():
    field = Schema(fields={}, allow_null=True)
    assert field.validate(None) is None

def test_schema_validate_null_without_allow_null():
    field = Schema(fields={}, allow_null=False)
    try:
        field.validate(None)
        assert False
    except ValidationError as e:
        assert e.messages[0].code == "null"

def test_schema_validate_non_dict_value():
    field = Schema(fields={})
    try:
        field.validate("not a dict")
        assert False
    except ValidationError as e:
        assert e.messages[0].code == "type"

def test_schema_validate_invalid_key_type():
    field = Schema(fields={})
    try:
        field.validate({1: "value"})
        assert False
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"

def test_schema_validate_missing_required_field():
    child_field = Field(required=True)
    field = Schema(fields={"test": child_field})
    try:
        field.validate({})
        assert False
    except ValidationError as e:
        assert e.messages[0].code == "required"

def test_schema_validate_with_default_value():
    child_field = Field(default="default")
    field = Schema(fields={"test": child_field})
    result = field.validate({})
    assert result["test"] == "default"

def test_schema_validate_with_read_only_field():
    child_field = Field(read_only=True)
    field = Schema(fields={"test": child_field})
    result = field.validate({"test": "value"})
    assert "test" not in result

def test_schema_validate_with_valid_fields():
    child_field = Field()
    field = Schema(fields={"test": child_field})
    result = field.validate({"test": "value"})
    assert result["test"] == "value"

def test_schema_validate_with_nested_validation_error():
    child_field = Field(required=True)
    field = Schema(fields={"test": child_field})
    try:
        field.validate({"test": None})
        assert False
    except ValidationError as e:
        assert e.messages[0].code == "required


# LLM-generated content at query #12
#--------------------------

```python
def test_validate_with_error():
    field = Field()
    schema = Schema(fields={"test_field": field})
    value = {"test_field": "invalid_value"}
    field.validate_or_error = lambda x: (None, ValidationError(messages=[Message(text="Invalid value", code="invalid_value", index=["test_field"])]))
    try:
        schema.validate(value)
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Invalid value"


# LLM-generated content at query #13
#--------------------------

```python
def test_validate_allows_null_when_allow_null_is_true():
    fields = {}
    schema = Schema(fields, allow_null=True)
    result = schema.validate(None)
    assert result is None


# LLM-generated content at query #14
#--------------------------

```python
def test_validate_allow_null():
    fields = {"field1": Field()}
    schema = Schema(fields, allow_null=True)
    assert schema.validate(None) is None

def test_validate_raise_null_error():
    fields = {"field1": Field()}
    schema = Schema(fields, allow_null=False)
    try:
        schema.validate(None)
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "null"

def test_validate_raise_type_error():
    fields = {"field1": Field()}
    schema = Schema(fields)
    try:
        schema.validate("not_a_dict")
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "type"

def test_validate_raise_invalid_key_error():
    fields = {"field1": Field()}
    schema = Schema(fields)
    try:
        schema.validate({1: "invalid_key"})
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "invalid_key"

def test_validate_raise_required_error():
    fields = {"field1": Field()}
    schema = Schema(fields)
    try:
        schema.validate({})
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "required"

def test_validate_success():
    fields = {"field1": Field()}
    schema = Schema(fields)
    validated = schema.validate({"field1": "value1"})
    assert validated == {"field1": "value1"}

def test_validate_with_default_value():
    fields = {"field1": Field(default="default_value")}
    schema = Schema(fields)
    validated = schema.validate({})
    assert validated == {"field1": "default_value"}

def test_validate_with_read_only_field():
    fields = {"field1": Field(read_only=True)}
    schema = Schema(fields)
    validated = schema.validate({"field1": "value1"})
    assert validated == {}

def test_validate_with_nested_errors():
    nested_field = Field()
    nested_field.validate = lambda x: (None, ValidationError(messages=[Message(text="nested_error", code="nested_error")]))
    fields = {"field1": nested_field}
    schema = Schema(fields)
    try:
        schema.validate({"field1": "value1"})
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "nested_error"


# LLM-generated content at query #15
#--------------------------

```
def test_validate_with_error_in_child_schema():
    class MockField(Field):
        def validate_or_error(self, value):
            return None, ValidationError(messages=[Message(text="error", code="error", index=[])])

    fields = {"test_field": MockField()}
    schema = Schema(fields=fields)
    value = {"test_field": "invalid_value"}
    try:
        schema.validate(value)
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "error"


# LLM-generated content at query #16
#--------------------------

```
def test_validate_returns_none_when_value_is_none_and_allow_null_is_true():
    field_mock = MagicMock()
    field_mock.read_only = False
    field_mock.has_default.return_value = False
    schema = Schema(fields={"key": field_mock}, allow_null=True)
    result = schema.validate(None)
    assert result is None


# LLM-generated content at query #17
#--------------------------

```python
def test_validate_with_error_in_child_schema():
    class MockField:
        def __init__(self, has_error=False, read_only=False, has_default=False):
            self.has_error = has_error
            self.read_only = read_only
            self.has_default = has_default

        def validate_or_error(self, value):
            if self.has_error:
                return None, ValidationError(messages=[Message(text="error", code="error")])
            return value, None

        def has_default(self):
            return self.has_default

        def get_default_value(self):
            return "default"

    fields = {
        "test_field": MockField(has_error=True)
    }
    schema = Schema(fields=fields)
    value = {"test_field": "test_value"}
    validated = schema.validate(value)


# LLM-generated content at query #18
#--------------------------

```python
def test_validate_returns_none_when_value_is_null_and_allow_null_is_true():
    definitions = {"to": "definition"}
    reference = Reference(to="to", definitions=definitions, allow_null=True)
    result = reference.validate(None)
    assert result is None

def test_validate_raises_validation_error_when_value_is_null_and_allow_null_is_false():
    definitions = {"to": "definition"}
    reference = Reference(to="to", definitions=definitions, allow_null=False)
    try:
        reference.validate(None)
        assert False
    except Exception as e:
        assert str(e) == "May not be null."

def test_validate_returns_target_validate_result_when_value_is_not_null():
    definitions = {"to": "definition"}
    target_mock = lambda value: value
    definitions["to"] = target_mock
    reference = Reference(to="to", definitions=definitions)
    result = reference.validate("some_value")
    assert result == "some_value"


# LLM-generated content at query #19
#--------------------------

```
def test_validate_returns_none_when_value_is_none_and_allow_null_is_true():
    field = Field(allow_null=True)
    schema = Schema(fields={}, allow_null=True)
    result = schema.validate(None)
    assert result is None


# LLM-generated content at query #20
#--------------------------

```
def test_validate_null_when_allow_null():
    field = Schema(fields={}, allow_null=True)
    assert field.validate(None) is None

def test_validate_null_when_not_allow_null():
    field = Schema(fields={}, allow_null=False)
    try:
        field.validate(None)
        assert False
    except ValidationError as e:
        assert e.messages[0].text == "May not be null."

def test_validate_non_dict_value():
    field = Schema(fields={})
    try:
        field.validate("not a dict")
        assert False
    except ValidationError as e:
        assert e.messages[0].text == "Must be an object."

def test_validate_invalid_key_type():
    field = Schema(fields={})
    try:
        field.validate({123: "value"})
        assert False
    except ValidationError as e:
        assert e.messages[0].text == "All object keys must be strings."

def test_validate_missing_required_field():
    field = Schema(fields={"name": Field()})
    try:
        field.validate({})
        assert False
    except ValidationError as e:
        assert e.messages[0].text == "This field is required."

def test_validate_with_default_value():
    child_field = Field(default="default_value")
    field = Schema(fields={"name": child_field})
    validated = field.validate({})
    assert validated["name"] == "default_value"

def test_validate_with_read_only_field():
    child_field = Field(read_only=True)
    field = Schema(fields={"name": child_field})
    validated = field.validate({"name": "value"})
    assert "name" not in validated

def test_validate_with_valid_data():
    child_field = Field()
    field = Schema(fields={"name": child_field})
    validated = field.validate({"name": "value"})
    assert validated["name"] == "value"

def test_validate_with_nested_errors():
    nested_field = Schema(fields={"inner": Field()})
    field = Schema(fields={"nested": nested_field})
    try:
        field.validate({"nested": {}})
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "This field is required."
        assert e.messages[0].index == ["nested", "inner"]


# LLM-generated content at query #21
#--------------------------

```
def test_validate_returns_none_when_value_is_none_and_allow_null_is_true():
    field = Field(allow_null=True)
    schema = Schema(fields={}, allow_null=True)
    assert schema.validate(None) is None


# LLM-generated content at query #22
#--------------------------

```python
def test_validate_returns_none_when_value_is_none_and_allow_null_is_true():
    definitions = {}
    reference = Reference(to="target", definitions=definitions, allow_null=True)
    assert reference.validate(None) is None


# LLM-generated content at query #23
#--------------------------

```python
def test_validate_allows_null_when_allow_null_is_true():
    fields = {}
    schema = Schema(fields, allow_null=True)
    assert schema.validate(None) is None

def test_validate_raises_error_when_null_and_allow_null_is_false():
    fields = {}
    schema = Schema(fields, allow_null=False)
    try:
        schema.validate(None)
        assert False
    except ValidationError as e:
        assert e.messages[0].text == "May not be null."

def test_validate_raises_error_when_not_dict_or_mapping():
    fields = {}
    schema = Schema(fields)
    try:
        schema.validate([])
        assert False
    except ValidationError as e:
        assert e.messages[0].text == "Must be an object."

def test_validate_raises_error_when_keys_are_not_strings():
    fields = {"field": Field()}
    schema = Schema(fields)
    try:
        schema.validate({1: "value"})
        assert False
    except ValidationError as e:
        assert e.messages[0].text == "All object keys must be strings."

def test_validate_raises_error_when_required_field_is_missing():
    fields = {"field": Field()}
    schema = Schema(fields)
    try:
        schema.validate({})
        assert False
    except ValidationError as e:
        assert e.messages[0].text == "This field is required."

def test_validate_sets_default_value_when_field_is_missing_and_has_default():
    fields = {"field": Field(default="default_value")}
    schema = Schema(fields)
    validated = schema.validate({})
    assert validated["field"] == "default_value"

def test_validate_validates_child_fields():
    fields = {"field": Field()}
    schema = Schema(fields)
    validated = schema.validate({"field": "value"})
    assert validated["field"] == "value"

def test_validate_raises_error_when_child_field_validation_fails():
    class FailingField(Field):
        def validate(self, value):
            raise self.validation_error("custom_error")
    
    fields = {"field": FailingField()}
    schema = Schema(fields)
    try:
        schema.validate({"field": "value"})
        assert False
    except ValidationError as e:
        assert e.messages[0].text == "custom_error"

def test_validate_skips_read_only_fields():
    fields = {"field": Field(read_only=True)}
    schema = Schema(fields)
    validated = schema.validate({"field": "value"})
    assert "field" not in validated


# LLM-generated content at query #24
#--------------------------

```python
def test_validate_with_null_and_allow_null():
    schema = Schema(fields={}, allow_null=True)
    result = schema.validate(None)
    assert result is None


# LLM-generated content at query #25
#--------------------------

```python
def test_validate_allows_null_when_allow_null_is_true():
    definitions = Definitions({"example": {"type": "string"}})
    reference = Reference(to="example", definitions=definitions, allow_null=True)
    result = reference.validate(None)
    assert result is None

def test_validate_raises_error_when_null_and_allow_null_is_false():
    definitions = Definitions({"example": {"type": "string"}})
    reference = Reference(to="example", definitions=definitions, allow_null=False)
    try:
        reference.validate(None)
        assert False, "Expected validation error"
    except Exception as e:
        assert str(e) == "May not be null."

def test_validate_delegates_to_target_validate():
    definitions = Definitions({"example": {"type": "string"}})
    reference = Reference(to="example", definitions=definitions)
    result = reference.validate("test")
    assert result == "test"

def test_validate_raises_error_from_target_validate():
    definitions = Definitions({"example": {"type": "string"}})
    reference = Reference(to="example", definitions=definitions)
    try:
        reference.validate(123)
        assert False, "Expected validation error from target"
    except Exception as e:
        assert str(e) != "May not be null."


# LLM-generated content at query #26
#--------------------------

```python
def test_validate_predicate_at_line_37_evaluates_to_false():
    from typing import Dict
    from unittest.mock import Mock

    class Field:
        def __init__(self, read_only=False, has_default=False, default_value=None):
            self.read_only = read_only
            self._has_default = has_default
            self._default_value = default_value

        def has_default(self):
            return self._has_default

        def get_default_value(self):
            return self._default_value

        def validate_or_error(self, value):
            return value, Mock()

    schema = Schema(fields={"key": Field()})
    value = {"key": "invalid_value"}
    try:
        schema.validate(value)
    except ValidationError:
        pass


# LLM-generated content at query #27
#--------------------------

```python
def test_validate_with_null_value_and_allow_null_true():
    fields = {"name": Field()}
    schema = Schema(fields, allow_null=True)
    assert schema.validate(None) is None

def test_validate_with_null_value_and_allow_null_false():
    fields = {"name": Field()}
    schema = Schema(fields, allow_null=False)
    try:
        schema.validate(None)
        assert False
    except ValidationError as e:
        assert e.messages[0].text == "May not be null."

def test_validate_with_non_dict_value():
    fields = {"name": Field()}
    schema = Schema(fields)
    try:
        schema.validate("not a dict")
        assert False
    except ValidationError as e:
        assert e.messages[0].text == "Must be an object."

def test_validate_with_invalid_key_type():
    fields = {"name": Field()}
    schema = Schema(fields)
    try:
        schema.validate({1: "value"})
        assert False
    except ValidationError as e:
        assert e.messages[0].text == "All object keys must be strings."

def test_validate_with_missing_required_field():
    fields = {"name": Field()}
    schema = Schema(fields)
    try:
        schema.validate({})
        assert False
    except ValidationError as e:
        assert e.messages[0].text == "This field is required."

def test_validate_with_valid_dict():
    fields = {"name": Field()}
    schema = Schema(fields)
    validated = schema.validate({"name": "John"})
    assert validated == {"name": "John"}

def test_validate_with_default_value():
    fields = {"name": Field(default="Default Name")}
    schema = Schema(fields)
    validated = schema.validate({})
    assert validated == {"name": "Default Name"}

def test_validate_with_read_only_field():
    fields = {"name": Field(read_only=True)}
    schema = Schema(fields)
    validated = schema.validate({"name": "John"})
    assert validated == {}

def test_validate_with_nested_field_validation_error():
    nested_field = Field()
    nested_field.validate = lambda x: (None, ValidationError(messages=[Message(text="Invalid nested field", code="invalid_nested")]))
    fields = {"nested": nested_field}
    schema = Schema(fields)
    try:
        schema.validate({"nested": "value"})
        assert False
    except ValidationError as e:
        assert e.messages[0].text == "Invalid nested field"

def test_validate_with_multiple_errors():
    fields = {"name": Field(), "age": Field()}
    schema = Schema(fields)
    try:
        schema.validate({"age": 25})
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "This field is required."


# LLM-generated content at query #28
#--------------------------

```python
def test_validate_none_with_allow_null():
    definitions = {"target": type("MockField", (), {"validate": lambda self, value: value})()}
    field = Reference(to="target", definitions=definitions, allow_null=True)
    assert field.validate(None) is None

def test_validate_none_without_allow_null():
    definitions = {"target": type("MockField", (), {"validate": lambda self, value: value})()}
    field = Reference(to="target", definitions=definitions, allow_null=False)
    try:
        field.validate(None)
        assert False
    except Exception as e:
        assert str(e) == "May not be null."

def test_validate_non_none_value():
    definitions = {"target": type("MockField", (), {"validate": lambda self, value: value})()}
    field = Reference(to="target", definitions=definitions, allow_null=False)
    assert field.validate("test") == "test"


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_validate_allow_null():
    definitions = {"target": DummyField(allow_null=True)}
    ref = Reference(to="target", definitions=definitions, allow_null=True)
    assert ref.validate(None) is None

def test_validate_not_allow_null():
    definitions = {"target": DummyField(allow_null=False)}
    ref = Reference(to="target", definitions=definitions, allow_null=False)
    try:
        ref.validate(None)
        assert False, "Expected validation_error"
    except Exception as e:
        assert str(e) == "May not be null."

def test_validate_valid_value():
    definitions = {"target": DummyField()}
    ref = Reference(to="target", definitions=definitions)
    assert ref.validate({"key": "value"}) == {"key": "value"}


# LLM-generated content at query #2
#--------------------------

```python
def test_validate_null_value():
    schema = Schema({}, allow_null=True)
    assert schema.validate(None) is None

def test_validate_null_value_not_allowed():
    schema = Schema({}, allow_null=False)
    try:
        schema.validate(None)
        assert False
    except ValidationError as e:
        assert e.messages[0].text == "May not be null."

def test_validate_non_dict_value():
    schema = Schema({})
    try:
        schema.validate("not a dict")
        assert False
    except ValidationError as e:
        assert e.messages[0].text == "Must be an object."

def test_validate_invalid_key_type():
    schema = Schema({})
    try:
        schema.validate({1: "value"})
        assert False
    except ValidationError as e:
        assert e.messages[0].text == "All object keys must be strings."

def test_validate_missing_required_field():
    schema = Schema({"field": Field()})
    try:
        schema.validate({})
        assert False
    except ValidationError as e:
        assert e.messages[0].text == "This field is required."

def test_validate_with_default_value():
    schema = Schema({"field": Field(default="default_value")})
    assert schema.validate({}) == {"field": "default_value"}

def test_validate_with_read_only_field():
    schema = Schema({"field": Field(read_only=True)})
    assert schema.validate({"field": "value"}) == {}

def test_validate_successful():
    schema = Schema({"field": Field()})
    assert schema.validate({"field": "value"}) == {"field": "value"}


# LLM-generated content at query #3
#--------------------------

```
def test_serialize_returns_none_when_obj_is_none():
    schema = Schema(fields={})
    assert schema.serialize(None) is None

def test_serialize_handles_dict_object():
    field = Field()
    schema = Schema(fields={"name": field})
    obj = {"name": "test"}
    assert schema.serialize(obj) == {"name": None}

def test_serialize_handles_non_dict_object():
    field = Field()
    schema = Schema(fields={"name": field})
    class TestObj:
        name = "test"
    obj = TestObj()
    assert schema.serialize(obj) == {"name": None}

def test_serialize_ignores_missing_keys_in_dict():
    field = Field()
    schema = Schema(fields={"name": field})
    obj = {"other": "value"}
    assert schema.serialize(obj) == {}

def test_serialize_ignores_missing_attributes_in_object():
    field = Field()
    schema = Schema(fields={"name": field})
    class TestObj:
        other = "value"
    obj = TestObj()
    assert schema.serialize(obj) == {}

def test_serialize_processes_multiple_fields():
    field1 = Field()
    field2 = Field()
    schema = Schema(fields={"name": field1, "age": field2})
    obj = {"name": "test", "age": 20}
    assert schema.serialize(obj) == {"name": None, "age": None}

def test_serialize_handles_nested_fields():
    nested_schema = Schema(fields={"nested_name": Field()})
    schema = Schema(fields={"nested": nested_schema})
    obj = {"nested": {"nested_name": "test"}}
    assert schema.serialize(obj) == {"nested": {"nested_name": None}}


# LLM-generated content at query #4
#--------------------------

```python
def test_serialize_predicate_evaluates_to_false():
    class MockField:
        def serialize(self, value):
            return value

    class MockObj:
        def __init__(self, attr_value):
            self.attr_value = attr_value

    fields = {"attr": MockField()}
    schema = Schema(fields)
    obj = MockObj("test_value")
    result = schema.serialize(obj)
    assert result == {"attr": "test_value"}


# LLM-generated content at query #5
#--------------------------

```python
def test_validate_returns_none_when_value_is_none_and_allow_null_is_true():
    definitions = {}
    field = Reference(to="some_key", definitions=definitions, allow_null=True)
    assert field.validate(None) is None


# LLM-generated content at query #6
#--------------------------

```python
def test_validate_or_error_returns_error():
    class MockField(Field):
        def validate_or_error(self, value):
            return None, ValidationError(messages=[Message(text="error", code="error", index=[])])

    fields = {"key": MockField()}
    schema = Schema(fields)
    value = {"key": "invalid_value"}
    try:
        schema.validate(value)
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "error"


# LLM-generated content at query #7
#--------------------------

```python
def test_serialize_returns_none_for_none_input():
    schema = Schema(fields={})
    result = schema.serialize(None)
    assert result is None

def test_serialize_returns_dict_for_dict_input():
    schema = Schema(fields={"field1": Field()})
    result = schema.serialize({"field1": "value1"})
    assert result == {"field1": "value1"}

def test_serialize_returns_dict_for_object_input():
    class TestObject:
        def __init__(self):
            self.field1 = "value1"
    schema = Schema(fields={"field1": Field()})
    result = schema.serialize(TestObject())
    assert result == {"field1": "value1"}

def test_serialize_skips_missing_keys():
    schema = Schema(fields={"field1": Field(), "field2": Field()})
    result = schema.serialize({"field1": "value1"})
    assert result == {"field1": "value1"}

def test_serialize_handles_read_only_fields():
    field = Field(read_only=True)
    schema = Schema(fields={"field1": field})
    result = schema.serialize({"field1": "value1"})
    assert result == {}


# LLM-generated content at query #8
#--------------------------

```python
def test_serialize_predicate_evaluates_to_false():
    class TestObject:
        def __init__(self):
            self.test_field = "value"

    field = Field()
    schema = Schema({"test_field": field})
    obj = TestObject()
    result = schema.serialize(obj)
    assert result == {"test_field": None}


# LLM-generated content at query #9
#--------------------------

```python
def test_validate_allow_null():
    schema = Schema(fields={}, allow_null=True)
    result = schema.validate(None)
    assert result is None


# LLM-generated content at query #10
#--------------------------

```python
def test_validate_null_when_allow_null_is_true():
    field = Schema(fields={}, allow_null=True)
    assert field.validate(None) is None

def test_validate_null_when_allow_null_is_false():
    field = Schema(fields={}, allow_null=False)
    try:
        field.validate(None)
        assert False
    except ValidationError as e:
        assert e.messages[0].text == "May not be null."

def test_validate_non_dict_value():
    field = Schema(fields={})
    try:
        field.validate("not a dict")
        assert False
    except ValidationError as e:
        assert e.messages[0].text == "Must be an object."

def test_validate_invalid_key_type():
    field = Schema(fields={})
    try:
        field.validate({1: "value"})
        assert False
    except ValidationError as e:
        assert e.messages[0].text == "All object keys must be strings."

def test_validate_missing_required_field():
    child_field = Field()
    field = Schema(fields={"required_field": child_field})
    try:
        field.validate({})
        assert False
    except ValidationError as e:
        assert e.messages[0].text == "This field is required."

def test_validate_with_default_value():
    child_field = Field(default="default_value")
    field = Schema(fields={"optional_field": child_field})
    validated = field.validate({})
    assert validated["optional_field"] == "default_value"

def test_validate_with_read_only_field():
    child_field = Field(read_only=True)
    field = Schema(fields={"read_only_field": child_field})
    validated = field.validate({"read_only_field": "value"})
    assert "read_only_field" not in validated

def test_validate_with_child_validation_error():
    child_field = Field()
    def validate(value):
        raise ValidationError(messages=[Message(text="Invalid", code="invalid")])
    child_field.validate = validate
    field = Schema(fields={"child_field": child_field})
    try:
        field.validate({"child_field": "value"})
        assert False
    except ValidationError as e:
        assert e.messages[0].text == "Invalid"

def test_validate_successful():
    child_field = Field()
    field = Schema(fields={"valid_field": child_field})
    validated = field.validate({"valid_field": "value"})
    assert validated["valid_field"] == "value"


# LLM-generated content at query #11
#--------------------------

```python
def test_validate_null_value():
    schema = Schema({}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Expected validation error for null value"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "May not be null."
        assert e.messages[0].code == "null"

def test_validate_non_dict_value():
    schema = Schema({})
    try:
        schema.validate("not_a_dict")
        assert False, "Expected validation error for non-dict value"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Must be an object."
        assert e.messages[0].code == "type"

def test_validate_invalid_key_type():
    schema = Schema({"field": Field()})
    try:
        schema.validate({1: "value"})
        assert False, "Expected validation error for invalid key type"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "All object keys must be strings."
        assert e.messages[0].code == "invalid_key"
        assert e.messages[0].index == [1]

def test_validate_missing_required_field():
    schema = Schema({"field": Field()})
    try:
        schema.validate({})
        assert False, "Expected validation error for missing required field"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "This field is required."
        assert e.messages[0].code == "required"
        assert e.messages[0].index == ["field"]

def test_validate_valid_value():
    schema = Schema({"field": Field()})
    validated = schema.validate({"field": "value"})
    assert validated == {"field": "value"}

def test_validate_field_with_default():
    schema = Schema({"field": Field(default="default_value")})
    validated = schema.validate({})
    assert validated == {"field": "default_value"}

def test_validate_read_only_field():
    schema = Schema({"field": Field(read_only=True)})
    validated = schema.validate({"field": "value"})
    assert validated == {}

def test_validate_nested_validation_error():
    schema = Schema({"field": Field(allow_null=False)})
    try:
        schema.validate({"field": None})
        assert False, "Expected validation error for nested null value"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "May not be null."
        assert e.messages[0].code == "null"
        assert e.messages[0].index == ["field"]


# LLM-generated content at query #12
#--------------------------

```python
def test_validate_with_null_value_and_allow_null():
    schema = Schema({"field": Field()}, allow_null=True)
    validated_value = schema.validate(None)
    assert validated_value is None

def test_validate_with_null_value_and_disallow_null():
    schema = Schema({"field": Field()}, allow_null=False)
    try:
        schema.validate(None)
        assert False
    except ValidationError as e:
        assert str(e) == "May not be null."

def test_validate_with_non_dict_value():
    schema = Schema({"field": Field()})
    try:
        schema.validate("not a dict")
        assert False
    except ValidationError as e:
        assert str(e) == "Must be an object."

def test_validate_with_invalid_key_type():
    schema = Schema({"field": Field()})
    try:
        schema.validate({1: "value"})
        assert False
    except ValidationError as e:
        assert str(e) == "All object keys must be strings."

def test_validate_with_missing_required_field():
    schema = Schema({"field": Field()})
    try:
        schema.validate({"other_field": "value"})
        assert False
    except ValidationError as e:
        assert str(e) == "This field is required."

def test_validate_with_valid_data():
    schema = Schema({"field": Field()})
    validated_value = schema.validate({"field": "value"})
    assert validated_value == {"field": "value"}

def test_validate_with_default_value():
    schema = Schema({"field": Field(default="default_value")})
    validated_value = schema.validate({})
    assert validated_value == {"field": "default_value"}

def test_validate_with_read_only_field():
    schema = Schema({"field": Field(read_only=True)})
    validated_value = schema.validate({"field": "value"})
    assert validated_value == {}

def test_validate_with_nested_validation_error():
    nested_schema = Schema({"nested_field": Field()})
    schema = Schema({"field": nested_schema})
    try:
        schema.validate({"field": {"nested_field": None}})
        assert False
    except ValidationError as e:
        assert str(e) == "May not be null."


# LLM-generated content at query #13
#--------------------------

```python
def test_validate_allows_null_when_allow_null_is_true():
    field = Schema(fields={}, allow_null=True)
    assert field.validate(None) is None

def test_validate_raises_error_when_null_and_allow_null_is_false():
    field = Schema(fields={}, allow_null=False)
    try:
        field.validate(None)
        assert False
    except ValidationError as e:
        assert e.messages[0].text == "May not be null."

def test_validate_raises_error_when_value_is_not_dict():
    field = Schema(fields={})
    try:
        field.validate("not_a_dict")
        assert False
    except ValidationError as e:
        assert e.messages[0].text == "Must be an object."

def test_validate_raises_error_when_keys_are_not_strings():
    field = Schema(fields={"key": Field()})
    try:
        field.validate({1: "value"})
        assert False
    except ValidationError as e:
        assert e.messages[0].text == "All object keys must be strings."

def test_validate_raises_error_when_required_key_is_missing():
    field = Schema(fields={"key": Field()})
    try:
        field.validate({})
        assert False
    except ValidationError as e:
        assert e.messages[0].text == "This field is required."

def test_validate_sets_default_value_when_key_is_missing_and_field_has_default():
    field = Schema(fields={"key": Field(default="default_value")})
    validated = field.validate({})
    assert validated["key"] == "default_value"

def test_validate_returns_validated_values():
    field = Schema(fields={"key": Field()})
    validated = field.validate({"key": "value"})
    assert validated["key"] == "value"

def test_validate_raises_error_when_child_validation_fails():
    field = Schema(fields={"key": Field()})
    try:
        field.validate({"key": None})
        assert False
    except ValidationError as e:
        assert e.messages[0].text == "May not be null."


# LLM-generated content at query #14
#--------------------------

```python
def test_validate_returns_none_when_value_is_none_and_allow_null_is_true():
    definitions = {}
    field = Reference(to="some_target", definitions=definitions, allow_null=True)
    result = field.validate(None)
    assert result is None


# LLM-generated content at query #15
#--------------------------

```python
def test_schema_validate_null_with_allow_null():
    field = Schema(fields={}, allow_null=True)
    assert field.validate(None) is None

def test_schema_validate_null_without_allow_null():
    field = Schema(fields={}, allow_null=False)
    try:
        field.validate(None)
        assert False
    except ValidationError as e:
        assert e.messages[0].text == "May not be null."

def test_schema_validate_non_dict_value():
    field = Schema(fields={})
    try:
        field.validate("not a dict")
        assert False
    except ValidationError as e:
        assert e.messages[0].text == "Must be an object."

def test_schema_validate_invalid_key_type():
    field = Schema(fields={})
    try:
        field.validate({1: "value"})
        assert False
    except ValidationError as e:
        assert e.messages[0].text == "All object keys must be strings."

def test_schema_validate_missing_required_field():
    child_field = Field(required=True)
    field = Schema(fields={"test": child_field})
    try:
        field.validate({})
        assert False
    except ValidationError as e:
        assert e.messages[0].text == "This field is required."

def test_schema_validate_with_default_value():
    child_field = Field(default="default")
    field = Schema(fields={"test": child_field})
    result = field.validate({})
    assert result["test"] == "default"

def test_schema_validate_read_only_field():
    child_field = Field(read_only=True)
    field = Schema(fields={"test": child_field})
    result = field.validate({"test": "value"})
    assert "test" not in result

def test_schema_validate_valid_input():
    child_field = Field()
    field = Schema(fields={"test": child_field})
    result = field.validate({"test": "value"})
    assert result["test"] == "value"

def test_schema_validate_nested_validation_error():
    child_field = Field(required=True)
    field = Schema(fields={"test": child_field})
    try:
        field.validate({"test": None})
        assert False
    except ValidationError as e:
        assert len(e.messages) > 0


# LLM-generated content at query #16
#--------------------------

```
def test_validate_returns_none_when_value_is_null_and_allow_null_is_true():
    field = Field(allow_null=True)
    schema = Schema(fields={}, allow_null=True)
    result = schema.validate(None)
    assert result is None


# LLM-generated content at query #17
#--------------------------

```
def test_validate_allows_null_when_allow_null_is_true():
    definitions = {"target": "some_definition"}
    field = Reference(to="target", definitions=definitions, allow_null=True)
    assert field.validate(None) is None

def test_validate_raises_error_when_null_and_allow_null_is_false():
    definitions = {"target": "some_definition"}
    field = Reference(to="target", definitions=definitions, allow_null=False)
    try:
        field.validate(None)
        assert False, "Expected validation error"
    except Exception as e:
        assert str(e) == "May not be null."

def test_validate_delegates_to_target_validation():
    class MockField:
        def validate(self, value):
            return "validated_" + str(value)
    
    definitions = {"target": MockField()}
    field = Reference(to="target", definitions=definitions)
    assert field.validate("value") == "validated_value"


# LLM-generated content at query #18
#--------------------------

```
def test_validate_allows_null_when_allow_null_is_true():
    definitions = {"target": "some_definition"}
    field = Reference(to="target", definitions=definitions, allow_null=True)
    assert field.validate(None) is None

def test_validate_raises_error_when_null_and_allow_null_is_false():
    definitions = {"target": "some_definition"}
    field = Reference(to="target", definitions=definitions, allow_null=False)
    try:
        field.validate(None)
        assert False, "Expected validation error"
    except Exception as e:
        assert str(e) == "May not be null."

def test_validate_delegates_to_target_validation():
    class MockField:
        def validate(self, value):
            return "validated_" + str(value)
    
    definitions = {"target": MockField()}
    field = Reference(to="target", definitions=definitions)
    assert field.validate("value") == "validated_value"


# LLM-generated content at query #19
#--------------------------

```python
def test_predicate_at_line_37_evaluates_to_false():
    class MockField(Field):
        def validate_or_error(self, value):
            return None, ValidationError(messages=[Message(text="error", code="error")])

    schema = Schema(fields={"key": MockField()})
    value = {"key": "value"}
    try:
        schema.validate(value)
    except ValidationError:
        pass
    else:
        assert False, "Expected ValidationError to be raised"


# LLM-generated content at query #20
#--------------------------

```python
def test_predicate_evaluates_to_false():
    class MockField:
        def validate_or_error(self, value):
            return None, "error"

    fields = {"field1": MockField()}
    schema = Schema(fields=fields)
    value = {"field1": "invalid_value"}
    validated = schema.validate(value)
    assert validated == {}


# LLM-generated content at query #21
#--------------------------

```python
def test_predicate_evaluates_to_true():
    class MockField:
        def validate_or_error(self, value):
            return value, None

    fields = {"key": MockField()}
    schema = Schema(fields)
    value = {"key": "valid_value"}
    schema.validate(value)


# LLM-generated content at query #22
#--------------------------

```python
def test_validate_with_valid_child_value():
    class MockField:
        def validate_or_error(self, value):
            return "valid_value", None

    fields = {"test_field": MockField()}
    schema = Schema(fields=fields)
    value = {"test_field": "any_value"}
    validated = schema.validate(value)
    assert validated["test_field"] == "valid_value"


# LLM-generated content at query #23
#--------------------------

```python
def test_validate_with_error_in_child_schema():
    from dataclasses import dataclass
    from unittest.mock import Mock

    @dataclass
    class Message:
        text: str
        code: str
        index: list

    @dataclass
    class ValidationError:
        messages: list

    class Field:
        def __init__(self, **kwargs):
            self.allow_null = kwargs.get("allow_null", False)
            self.read_only = kwargs.get("read_only", False)
            self._default = kwargs.get("default", None)

        def has_default(self):
            return self._default is not None

        def get_default_value(self):
            return self._default

        def validate_or_error(self, value):
            return None, ValidationError(messages=[Message(text="error", code="error", index=[])])

        def validation_error(self, code):
            return ValidationError(messages=[Message(text=code, code=code, index=[])])

        def get_error_text(self, code):
            return code

    schema = Schema(fields={"field": Field()})
    value = {"field": "value"}
    validated = schema.validate(value)


# LLM-generated content at query #24
#--------------------------

```python
def test_validate_with_valid_child_schema():
    class MockField:
        def validate_or_error(self, value):
            return "valid_value", None

    fields = {"test_field": MockField()}
    schema = Schema(fields=fields)
    value = {"test_field": "any_value"}
    validated = schema.validate(value)
    assert validated["test_field"] == "valid_value"


# LLM-generated content at query #25
#--------------------------

```python
def test_validate_returns_none_when_value_is_null_and_allow_null_is_true():
    schema = Schema(fields={}, allow_null=True)
    result = schema.validate(None)
    assert result is None


# LLM-generated content at query #26
#--------------------------

```
def test_validate_returns_none_when_value_is_none_and_allow_null_is_true():
    definitions = {}
    field = Reference(to="test", definitions=definitions, allow_null=True)
    result = field.validate(None)
    assert result is None

def test_validate_raises_error_when_value_is_none_and_allow_null_is_false():
    definitions = {}
    field = Reference(to="test", definitions=definitions, allow_null=False)
    try:
        field.validate(None)
        assert False
    except Exception as e:
        assert str(e) == "May not be null."


# LLM-generated content at query #27
#--------------------------

```python
def test_validate_returns_none_when_value_is_none_and_allow_null_is_true():
    schema = Schema(fields={}, allow_null=True)
    result = schema.validate(None)
    assert result is None


# LLM-generated content at query #28
#--------------------------

```python
def test_predicate_at_line_37_evaluates_to_false():
    from schema import Field, Schema, ValidationError, Message
    from typing import Dict

    class MockField(Field):
        def validate_or_error(self, value):
            return None, ValidationError(messages=[Message(text="error", code="error", index=[])])

    fields = {"key": MockField()}
    schema = Schema(fields)
    value = {"key": "invalid_value"}
    try:
        schema.validate(value)
    except ValidationError:
        pass
    else:
        assert False, "Predicate at line 37 should evaluate to False"


# LLM-generated content at query #29
#--------------------------

```
def test_validate_returns_none_when_value_is_none_and_allow_null_is_true():
    definitions = {}
    field = Reference(to="test", definitions=definitions, allow_null=True)
    result = field.validate(None)
    assert result is None


# LLM-generated content at query #30
#--------------------------

```python
def test_validate_allows_null_when_allow_null_is_true():
    field = Schema(fields={}, allow_null=True)
    assert field.validate(None) is None

def test_validate_raises_error_when_null_and_allow_null_is_false():
    field = Schema(fields={}, allow_null=False)
    try:
        field.validate(None)
        assert False
    except ValidationError as e:
        assert e.messages[0].text == "May not be null."

def test_validate_raises_error_when_value_is_not_a_dict():
    field = Schema(fields={})
    try:
        field.validate("not a dict")
        assert False
    except ValidationError as e:
        assert e.messages[0].text == "Must be an object."

def test_validate_raises_error_when_keys_are_not_strings():
    field = Schema(fields={"valid_key": Field()})
    try:
        field.validate({123: "value"})
        assert False
    except ValidationError as e:
        assert e.messages[0].text == "All object keys must be strings."

def test_validate_raises_error_when_required_field_is_missing():
    field = Schema(fields={"required_key": Field()})
    try:
        field.validate({"other_key": "value"})
        assert False
    except ValidationError as e:
        assert e.messages[0].text == "This field is required."

def test_validate_sets_default_value_when_field_has_default():
    field = Schema(fields={"key_with_default": Field(default="default_value")})
    validated = field.validate({})
    assert validated["key_with_default"] == "default_value"

def test_validate_ignores_read_only_fields():
    field = Schema(fields={"read_only_key": Field(read_only=True)})
    validated = field.validate({"read_only_key": "value"})
    assert "read_only_key" not in validated

def test_validate_validates_child_fields():
    child_field = Field()
    field = Schema(fields={"child_key": child_field})
    validated = field.validate({"child_key": "valid_value"})
    assert validated["child_key"] == "valid_value"

def test_validate_collects_errors_from_child_fields():
    child_field = Field()
    field = Schema(fields={"child_key": child_field})
    try:
        field.validate({"child_key": "invalid_value"})
        assert False
    except ValidationError as e:
        assert len(e.messages) > 0

def test_validate_returns_validated_dict():
    field = Schema(fields={"key": Field()})
    validated = field.validate({"key": "value"})
    assert validated == {"key": "value"}


# LLM-generated content at query #31
#--------------------------

```
def test_validate_returns_none_when_value_is_none_and_allow_null_is_true():
    definitions = {}
    reference = Reference(to="test", definitions=definitions, allow_null=True)
    result = reference.validate(None)
    assert result is None

def test_validate_raises_error_when_value_is_none_and_allow_null_is_false():
    definitions = {}
    reference = Reference(to="test", definitions=definitions, allow_null=False)
    try:
        reference.validate(None)
        assert False
    except Exception as e:
        assert str(e) == "May not be null."

def test_validate_calls_target_validate_when_value_is_not_none():
    definitions = {"test": Mock()}
    reference = Reference(to="test", definitions=definitions)
    reference.validate("some_value")
    definitions["test"].validate.assert_called_once_with("some_value")


# LLM-generated content at query #32
#--------------------------

```
def test_validate_with_valid_child_value():
    field = Field()
    field.validate_or_error = lambda x: (x, None)
    schema = Schema(fields={"test": field})
    value = {"test": "valid"}
    validated = schema.validate(value)
    assert validated == {"test": "valid


# LLM-generated content at query #33
#--------------------------

```python
def test_validate_with_null_and_allow_null():
    fields = {}
    schema = Schema(fields=fields, allow_null=True)
    result = schema.validate(None)
    assert result is None


# LLM-generated content at query #34
#--------------------------

```
def test_validate_null_when_allow_null_is_true():
    field = Schema(fields={}, allow_null=True)
    assert field.validate(None) is None

def test_validate_null_when_allow_null_is_false():
    field = Schema(fields={}, allow_null=False)
    try:
        field.validate(None)
        assert False
    except ValidationError as e:
        assert e.messages[0].text == "May not be null."

def test_validate_non_dict_value():
    field = Schema(fields={})
    try:
        field.validate("not a dict")
        assert False
    except ValidationError as e:
        assert e.messages[0].text == "Must be an object."

def test_validate_invalid_key_type():
    field = Schema(fields={})
    try:
        field.validate({1: "value"})
        assert False
    except ValidationError as e:
        assert e.messages[0].text == "All object keys must be strings."

def test_validate_missing_required_field():
    child_field = Field()
    field = Schema(fields={"required_field": child_field})
    try:
        field.validate({})
        assert False
    except ValidationError as e:
        assert e.messages[0].text == "This field is required."

def test_validate_with_default_value():
    child_field = Field(default="default_value")
    field = Schema(fields={"optional_field": child_field})
    validated = field.validate({})
    assert validated["optional_field"] == "default_value"

def test_validate_with_read_only_field():
    child_field = Field(read_only=True)
    field = Schema(fields={"read_only_field": child_field})
    validated = field.validate({"read_only_field": "value"})
    assert "read_only_field" not in validated

def test_validate_nested_validation_error():
    child_field = Field()
    field = Schema(fields={"nested_field": child_field})
    try:
        field.validate({"nested_field": "invalid"})
        assert False
    except ValidationError as e:
        assert len(e.messages) > 0

def test_validate_successful_validation():
    child_field = Field()
    field = Schema(fields={"valid_field": child_field})
    validated = field.validate({"valid_field": "valid_value"})
    assert validated["valid_field"] == "valid_value


# LLM-generated content at query #35
#--------------------------

```
def test_validate_null_when_allow_null_is_true():
    field = Field(allow_null=True)
    schema = Schema(fields={}, allow_null=True)
    assert schema.validate(None) is None

def test_validate_null_when_allow_null_is_false():
    schema = Schema(fields={}, allow_null=False)
    try:
        schema.validate(None)
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "null"

def test_validate_non_dict_value():
    schema = Schema(fields={})
    try:
        schema.validate("not a dict")
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "type"

def test_validate_invalid_key_type():
    schema = Schema(fields={})
    try:
        schema.validate({1: "value"})
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "invalid_key"

def test_validate_missing_required_field():
    field = Field()
    schema = Schema(fields={"required_field": field})
    try:
        schema.validate({})
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "required"

def test_validate_with_default_value():
    field = Field(default="default_value")
    schema = Schema(fields={"field_with_default": field})
    validated = schema.validate({})
    assert validated["field_with_default"] == "default_value"

def test_validate_nested_field_error():
    nested_field = Field()
    nested_field.errors = {"type": "Nested error"}
    schema = Schema(fields={"nested": nested_field})
    try:
        schema.validate({"nested": "invalid"})
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "type"

def test_validate_successful():
    field = Field()
    schema = Schema(fields={"valid_field": field})
    validated = schema.validate({"valid_field": "valid_value"})
    assert validated["valid_field"] == "valid_value"

def test_validate_read_only_field_ignored():
    field = Field(read_only=True)
    schema = Schema(fields={"read_only_field": field})
    validated = schema.validate({"read_only_field": "ignored"})
    assert "read_only_field" not in validated


# LLM-generated content at query #36
#--------------------------

```python
def test_validate_with_null_value_and_allow_null():
    field = Field(allow_null=True)
    schema = Schema(fields={}, allow_null=True)
    result = schema.validate(None)
    assert result is None


# LLM-generated content at query #37
#--------------------------

```
def test_validate_returns_none_when_value_is_none_and_allow_null_is_true():
    definitions = {}
    field = Reference(to="test", definitions=definitions, allow_null=True)
    result = field.validate(None)
    assert result is None

def test_validate_raises_error_when_value_is_none_and_allow_null_is_false():
    definitions = {}
    field = Reference(to="test", definitions=definitions, allow_null=False)
    try:
        field.validate(None)
        assert False
    except Exception as e:
        assert str(e) == "May not be null."


# LLM-generated content at query #38
#--------------------------

```python
def test_validate_with_valid_child_value():
    class MockField(Field):
        def validate_or_error(self, value):
            return value, None

    fields = {"key": MockField()}
    schema = Schema(fields)
    value = {"key": "valid_value"}
    validated = schema.validate(value)
    assert validated == {"key": "valid_value"}


