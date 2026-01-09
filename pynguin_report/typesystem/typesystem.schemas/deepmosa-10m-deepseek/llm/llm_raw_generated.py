####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_validate_returns_none_when_value_is_none_and_allow_null_is_true():
    mock_definitions = {"target": MockField()}
    reference = Reference(to="target", definitions=mock_definitions, allow_null=True)
    result = reference.validate(None)
    assert result is None

def test_validate_raises_error_when_value_is_none_and_allow_null_is_false():
    mock_definitions = {"target": MockField()}
    reference = Reference(to="target", definitions=mock_definitions, allow_null=False)
    try:
        reference.validate(None)
        assert False
    except ValidationError as e:
        assert e.detail == "May not be null."

def test_validate_calls_target_validate_when_value_is_not_none():
    mock_target = MockField()
    mock_definitions = {"target": mock_target}
    reference = Reference(to="target", definitions=mock_definitions, allow_null=False)
    value = {"id": 1}
    mock_target.validate.return_value = "validated_value"
    result = reference.validate(value)
    mock_target.validate.assert_called_once_with(value)
    assert result == "validated_value"

def test_validate_works_with_allow_null_true_and_non_none_value():
    mock_target = MockField()
    mock_definitions = {"target": mock_target}
    reference = Reference(to="target", definitions=mock_definitions, allow_null=True)
    value = {"id": 1}
    mock_target.validate.return_value = "validated_value"
    result = reference.validate(value)
    mock_target.validate.assert_called_once_with(value)
    assert result == "validated_value"


# LLM-generated content at query #2
#--------------------------

def test_validate_returns_none_when_value_is_none_and_allow_null_is_true():
    field = Field(allow_null=True)
    schema = Schema(fields={}, allow_null=True)
    result = schema.validate(None)
    assert result is None

def test_validate_raises_null_error_when_value_is_none_and_allow_null_is_false():
    schema = Schema(fields={}, allow_null=False)
    try:
        schema.validate(None)
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "null"

def test_validate_raises_type_error_when_value_is_not_dict_or_mapping():
    schema = Schema(fields={})
    try:
        schema.validate("not a dict")
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "type"

def test_validate_raises_invalid_key_error_when_key_is_not_string():
    schema = Schema(fields={})
    try:
        schema.validate({123: "value"})
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "invalid_key"

def test_validate_raises_required_error_when_required_key_is_missing():
    field = Field()
    schema = Schema(fields={"required_field": field})
    try:
        schema.validate({})
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "required"

def test_validate_uses_default_value_when_key_is_missing_and_field_has_default():
    field = Field(default="default_value")
    schema = Schema(fields={"field_with_default": field})
    result = schema.validate({})
    assert result == {"field_with_default": "default_value"}

def test_validate_skips_read_only_fields():
    field = Field(read_only=True)
    schema = Schema(fields={"read_only_field": field})
    result = schema.validate({"read_only_field": "ignored"})
    assert result == {}

def test_validate_includes_validated_child_field_values():
    child_field = Field()
    schema = Schema(fields={"child": child_field})
    result = schema.validate({"child": "child_value"})
    assert result == {"child": "child_value"}

def test_validate_collects_child_field_validation_errors():
    child_field = Field(allow_null=False)
    schema = Schema(fields={"child": child_field})
    try:
        schema.validate({"child": None})
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "null"

def test_validate_returns_validated_dict_with_multiple_fields():
    field1 = Field()
    field2 = Field(default="default2")
    schema = Schema(fields={"key1": field1, "key2": field2})
    result = schema.validate({"key1": "value1"})
    assert result == {"key1": "value1", "key2": "default2"}

def test_validate_handles_mixed_valid_and_invalid_keys():
    valid_field = Field()
    invalid_field = Field(allow_null=False)
    schema = Schema(fields={"valid": valid_field, "invalid": invalid_field})
    try:
        schema.validate({"valid": "ok", "invalid": None})
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "null"

def test_validate_aggregates_multiple_errors():
    required_field = Field()
    schema = Schema(fields={"required": required_field})
    try:
        schema.validate({123: "value"})
        assert False
    except ValidationError as e:
        assert len(e.messages) == 2
        codes = {msg.code for msg in e.messages}
        assert codes == {"invalid_key", "required"}


# LLM-generated content at query #3
#--------------------------

def test_validate_with_error_in_child_schema():
    class MockField:
        read_only = False
        def has_default(self):
            return False
        def get_default_value(self):
            return None
        def validate_or_error(self, item):
            return None, ValidationError(messages=[Message(text="error", code="error", index=[])])
    fields = {"key": MockField()}
    schema = Schema(fields=fields)
    value = {"key": "value"}
    try:
        schema.validate(value)
    except ValidationError as e:
        assert any(msg.code == "error" for msg in e.messages)


# LLM-generated content at query #4
#--------------------------

def test_validate_allows_null_when_allow_null_is_true():
    mock_definitions = {"target_name": MockField()}
    reference = Reference(to="target_name", definitions=mock_definitions, allow_null=True)
    result = reference.validate(None)
    assert result is None

def test_validate_raises_error_for_null_when_allow_null_is_false():
    mock_definitions = {"target_name": MockField()}
    reference = Reference(to="target_name", definitions=mock_definitions, allow_null=False)
    try:
        reference.validate(None)
        assert False
    except ValidationError as e:
        assert e.message == "May not be null."

def test_validate_delegates_to_target_validate_for_non_null_value():
    mock_target = MockField()
    mock_definitions = {"target_name": mock_target}
    reference = Reference(to="target_name", definitions=mock_definitions, allow_null=False)
    input_value = {"key": "value"}
    mock_target.validate.return_value = "validated_result"
    result = reference.validate(input_value)
    mock_target.validate.assert_called_once_with(input_value)
    assert result == "validated_result"

def test_validate_delegates_to_target_validate_when_allow_null_is_true_but_value_not_null():
    mock_target = MockField()
    mock_definitions = {"target_name": mock_target}
    reference = Reference(to="target_name", definitions=mock_definitions, allow_null=True)
    input_value = {"key": "value"}
    mock_target.validate.return_value = "validated_result"
    result = reference.validate(input_value)
    mock_target.validate.assert_called_once_with(input_value)
    assert result == "validated_result"


# LLM-generated content at query #5
#--------------------------

def test_validate_with_error_in_child_schema():
    class MockField:
        read_only = False
        def has_default(self):
            return False
        def validate_or_error(self, value):
            return None, MockError()
    class MockError:
        def messages(self, add_prefix=None):
            return ["error"]
    fields = {"key": MockField()}
    schema = Schema(fields=fields)
    value = {"key": "invalid"}
    try:
        schema.validate(value)
    except ValidationError:
        pass


# LLM-generated content at query #6
#--------------------------

def test_validate_with_none_and_allow_null_true():
    mock_definitions = {}
    mock_target = Mock()
    mock_target.validate = Mock(return_value="validated_value")
    mock_definitions.__getitem__ = Mock(return_value=mock_target)
    field = Reference(to="some_ref", definitions=mock_definitions, allow_null=True)
    result = field.validate(None)
    assert result is None

def test_validate_with_none_and_allow_null_false():
    mock_definitions = {}
    mock_target = Mock()
    mock_target.validate = Mock(return_value="validated_value")
    mock_definitions.__getitem__ = Mock(return_value=mock_target)
    field = Reference(to="some_ref", definitions=mock_definitions, allow_null=False)
    field.validation_error = Mock(side_effect=Exception("null error"))
    try:
        field.validate(None)
        assert False
    except Exception as e:
        assert str(e) == "null error"

def test_validate_with_non_none_value():
    mock_definitions = {}
    mock_target = Mock()
    mock_target.validate = Mock(return_value="validated_value")
    mock_definitions.__getitem__ = Mock(return_value=mock_target)
    field = Reference(to="some_ref", definitions=mock_definitions, allow_null=False)
    result = field.validate("some_value")
    assert result == "validated_value"
    mock_target.validate.assert_called_once_with("some_value")


# LLM-generated content at query #7
#--------------------------

def test_validate_with_child_schema_error():
    from myapp import Field, Schema, ValidationError
    class MockField(Field):
        def validate_or_error(self, value):
            from myapp import ValidationError
            return None, ValidationError(messages=[])
    child_field = MockField()
    child_field.read_only = False
    schema = Schema(fields={"key": child_field})
    value = {"key": "invalid"}
    try:
        schema.validate(value)
    except ValidationError:
        pass


# LLM-generated content at query #8
#--------------------------

def test_validate_with_child_schema_error():
    class MockField:
        read_only = False
        def has_default(self):
            return False
        def get_default_value(self):
            return None
        def validate_or_error(self, item):
            return None, ValidationError(messages=[Message(text="error", code="error", index=[])])
    fields = {"key": MockField()}
    schema = Schema(fields=fields)
    value = {"key": "value"}
    try:
        schema.validate(value)
    except ValidationError:
        pass


# LLM-generated content at query #9
#--------------------------

def test_validate_with_child_schema_error():
    from myapp import Field, Schema, ValidationError
    class MockField(Field):
        def __init__(self, read_only=False, has_default_val=False, default_val=None, allow_null=False):
            super().__init__(allow_null=allow_null)
            self._read_only = read_only
            self._has_default_val = has_default_val
            self._default_val = default_val
        def read_only(self):
            return self._read_only
        def has_default(self):
            return self._has_default_val
        def get_default_value(self):
            return self._default_val
        def validate_or_error(self, value):
            return None, ValidationError(messages=[])
    error_field = MockField()
    error_field.validate_or_error = lambda v: (None, ValidationError(messages=[Message(text="error", code="error", index=[])]))
    fields = {"key": error_field}
    schema = Schema(fields=fields)
    value = {"key": "invalid"}
    try:
        schema.validate(value)
    except ValidationError:
        pass


# LLM-generated content at query #10
#--------------------------

def test_validate_with_null_and_allow_null():
    field = Field(allow_null=True)
    schema = Schema(fields={}, allow_null=True)
    result = schema.validate(None)
    assert result is None

def test_validate_with_null_and_not_allow_null():
    schema = Schema(fields={}, allow_null=False)
    try:
        schema.validate(None)
        assert False
    except ValidationError as e:
        assert any(m.code == "null" for m in e.messages)

def test_validate_with_non_dict_value():
    schema = Schema(fields={})
    try:
        schema.validate("not a dict")
        assert False
    except ValidationError as e:
        assert any(m.code == "type" for m in e.messages)

def test_validate_with_invalid_non_string_key():
    schema = Schema(fields={})
    try:
        schema.validate({1: "value"})
        assert False
    except ValidationError as e:
        assert any(m.code == "invalid_key" for m in e.messages)

def test_validate_with_missing_required_field():
    child_field = Field(read_only=False)
    schema = Schema(fields={"required_field": child_field})
    try:
        schema.validate({})
        assert False
    except ValidationError as e:
        assert any(m.code == "required" and m.index == ["required_field"] for m in e.messages)

def test_validate_with_read_only_field_present():
    child_field = Field(read_only=True)
    schema = Schema(fields={"read_only_field": child_field})
    result = schema.validate({"read_only_field": "value"})
    assert "read_only_field" not in result

def test_validate_with_field_having_default():
    child_field = Field(read_only=False, default="default_value")
    schema = Schema(fields={"field_with_default": child_field})
    result = schema.validate({})
    assert result["field_with_default"] == "default_value"

def test_validate_with_field_validation_error():
    child_field = Field(read_only=False)
    schema = Schema(fields={"problem_field": child_field})
    try:
        schema.validate({"problem_field": None})
        assert False
    except ValidationError as e:
        assert any("problem_field" in str(m.index) for m in e.messages)

def test_validate_success_with_all_fields_valid():
    child_field = Field(read_only=False)
    schema = Schema(fields={"valid_field": child_field})
    result = schema.validate({"valid_field": "valid_value"})
    assert result == {"valid_field": "valid_value"}

def test_validate_with_multiple_errors():
    child_field1 = Field(read_only=False)
    child_field2 = Field(read_only=False)
    schema = Schema(fields={"field1": child_field1, "field2": child_field2})
    try:
        schema.validate({"field1": None, "field2": None})
        assert False
    except ValidationError as e:
        messages = e.messages
        assert len(messages) >= 2
        assert any("field1" in str(m.index) for m in messages)
        assert any("field2" in str(m.index) for m in messages)


# LLM-generated content at query #11
#--------------------------

def test_validate_with_error_in_child_schema():
    class MockField:
        read_only = False
        def has_default(self):
            return False
        def get_default_value(self):
            return None
        def validate_or_error(self, item):
            return None, MockError()
    class MockError:
        def messages(self, add_prefix=None):
            return ["error"]
    fields = {"key": MockField()}
    schema = Schema(fields)
    value = {"key": "invalid"}
    try:
        schema.validate(value)
    except ValidationError:
        pass
    else:
        assert False, "Expected ValidationError"


# LLM-generated content at query #12
#--------------------------

def test_validate_with_error_in_child_schema():
    child_field = Field()
    child_field.read_only = False
    child_field.has_default = lambda: False
    child_field.validate_or_error = lambda x: (None, ValidationError(messages=[Message(text="error", code="error", index=[])]))
    schema = Schema(fields={"key": child_field})
    value = {"key": "invalid"}
    try:
        schema.validate(value)
    except ValidationError:
        pass


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_validate_allow_null():
    field = Field(allow_null=True)
    schema = Schema(fields={}, allow_null=True)
    result = schema.validate(None)
    assert result is None

def test_validate_null_without_allow_null():
    schema = Schema(fields={})
    try:
        schema.validate(None)
        assert False
    except ValidationError as e:
        assert any(m.code == "null" for m in e.messages)

def test_validate_non_dict_value():
    schema = Schema(fields={})
    try:
        schema.validate("not a dict")
        assert False
    except ValidationError as e:
        assert any(m.code == "type" for m in e.messages)

def test_validate_invalid_key_type():
    schema = Schema(fields={})
    try:
        schema.validate({1: "value"})
        assert False
    except ValidationError as e:
        assert any(m.code == "invalid_key" for m in e.messages)

def test_validate_missing_required_field():
    field = Field()
    schema = Schema(fields={"required_field": field})
    try:
        schema.validate({})
        assert False
    except ValidationError as e:
        assert any(m.code == "required" for m in e.messages)

def test_validate_read_only_field_ignored():
    field = Field(read_only=True)
    schema = Schema(fields={"read_only_field": field})
    result = schema.validate({"read_only_field": "value"})
    assert "read_only_field" not in result

def test_validate_field_with_default():
    field = Field(default="default_value")
    schema = Schema(fields={"field_with_default": field})
    result = schema.validate({})
    assert result["field_with_default"] == "default_value"

def test_validate_field_validation_error():
    field = Field()
    schema = Schema(fields={"test_field": field})
    try:
        schema.validate({"test_field": None})
        assert False
    except ValidationError as e:
        assert any(m.code == "null" for m in e.messages)

def test_validate_successful():
    field = Field()
    schema = Schema(fields={"test_field": field})
    result = schema.validate({"test_field": "valid"})
    assert result == {"test_field": "valid"}

def test_validate_multiple_errors():
    field = Field()
    schema = Schema(fields={"required": field})
    try:
        schema.validate({1: "invalid"})
        assert False
    except ValidationError as e:
        codes = [m.code for m in e.messages]
        assert "invalid_key" in codes
        assert "required" in codes


# LLM-generated content at query #2
#--------------------------

def test_validate_with_error_in_child_schema():
    class MockField:
        read_only = False
        def has_default(self):
            return False
        def validate_or_error(self, item):
            return None, MockError()
    class MockError:
        def messages(self, add_prefix):
            return [f"error with prefix {add_prefix}"]
    fields = {"key": MockField()}
    schema = Schema(fields)
    value = {"key": "invalid"}
    try:
        schema.validate(value)
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0] == "error with prefix key"


# LLM-generated content at query #3
#--------------------------

def test_validate_allows_null_when_allow_null_is_true():
    mock_definitions = {}
    mock_target = Mock()
    mock_target.validate = Mock(return_value="validated_value")
    mock_definitions.__getitem__ = Mock(return_value=mock_target)
    field = Reference(to="some_target", definitions=mock_definitions, allow_null=True)
    result = field.validate(None)
    assert result is None


def test_validate_raises_error_for_null_when_allow_null_is_false():
    mock_definitions = {}
    field = Reference(to="some_target", definitions=mock_definitions, allow_null=False)
    try:
        field.validate(None)
        assert False
    except Exception as e:
        assert str(e) == "May not be null."


def test_validate_calls_target_validate_for_non_null_value():
    mock_definitions = {}
    mock_target = Mock()
    mock_target.validate = Mock(return_value="validated_value")
    mock_definitions.__getitem__ = Mock(return_value=mock_target)
    field = Reference(to="some_target", definitions=mock_definitions)
    result = field.validate("some_value")
    mock_definitions.__getitem__.assert_called_once_with("some_target")
    mock_target.validate.assert_called_once_with("some_value")
    assert result == "validated_value"


# LLM-generated content at query #4
#--------------------------

def test_validate_with_null_and_allow_null():
    field = Field(allow_null=True)
    schema = Schema(fields={}, allow_null=True)
    result = schema.validate(None)
    assert result is None

def test_validate_with_null_and_not_allow_null():
    schema = Schema(fields={}, allow_null=False)
    try:
        schema.validate(None)
        assert False
    except ValidationError as e:
        assert any(m.code == "null" for m in e.messages)

def test_validate_with_non_dict_value():
    schema = Schema(fields={})
    try:
        schema.validate("not a dict")
        assert False
    except ValidationError as e:
        assert any(m.code == "type" for m in e.messages)

def test_validate_with_invalid_non_string_key():
    schema = Schema(fields={})
    try:
        schema.validate({1: "value"})
        assert False
    except ValidationError as e:
        assert any(m.code == "invalid_key" for m in e.messages)

def test_validate_missing_required_field():
    child_field = Field(read_only=False)
    schema = Schema(fields={"required_field": child_field})
    try:
        schema.validate({})
        assert False
    except ValidationError as e:
        assert any(m.code == "required" and m.index == ["required_field"] for m in e.messages)

def test_validate_read_only_field_ignored():
    child_field = Field(read_only=True)
    schema = Schema(fields={"read_only_field": child_field})
    result = schema.validate({"read_only_field": "some value"})
    assert "read_only_field" not in result

def test_validate_field_with_default_value():
    child_field = Field(read_only=False, default="default_value")
    schema = Schema(fields={"field_with_default": child_field})
    result = schema.validate({})
    assert result["field_with_default"] == "default_value"

def test_validate_field_with_validation_error():
    child_field = Field(read_only=False)
    schema = Schema(fields={"problem_field": child_field})
    try:
        schema.validate({"problem_field": "invalid"})
        assert False
    except ValidationError as e:
        assert any(m.index == ["problem_field"] for m in e.messages)

def test_validate_successful():
    child_field = Field(read_only=False)
    schema = Schema(fields={"valid_field": child_field})
    result = schema.validate({"valid_field": "valid_value"})
    assert result["valid_field"] == "valid_value"

def test_validate_with_multiple_errors():
    child_field1 = Field(read_only=False)
    child_field2 = Field(read_only=False)
    schema = Schema(fields={"field1": child_field1, "field2": child_field2})
    try:
        schema.validate({"field1": "invalid", "field2": "invalid"})
        assert False
    except ValidationError as e:
        messages = e.messages
        assert len(messages) == 2
        assert all(m.index in [["field1"], ["field2"]] for m in messages)


# LLM-generated content at query #5
#--------------------------

def test_validate_with_null_and_allow_null():
    field = Field(allow_null=True)
    schema = Schema(fields={}, allow_null=True)
    result = schema.validate(None)
    assert result is None

def test_validate_with_null_and_not_allow_null():
    schema = Schema(fields={}, allow_null=False)
    try:
        schema.validate(None)
        assert False
    except ValidationError as e:
        assert any(msg.code == "null" for msg in e.messages)

def test_validate_with_non_dict_value():
    schema = Schema(fields={})
    try:
        schema.validate("not a dict")
        assert False
    except ValidationError as e:
        assert any(msg.code == "type" for msg in e.messages)

def test_validate_with_invalid_non_string_key():
    schema = Schema(fields={})
    try:
        schema.validate({1: "value"})
        assert False
    except ValidationError as e:
        assert any(msg.code == "invalid_key" for msg in e.messages)

def test_validate_missing_required_field():
    child_field = Field()
    schema = Schema(fields={"required_field": child_field})
    try:
        schema.validate({})
        assert False
    except ValidationError as e:
        assert any(msg.code == "required" and msg.index == ["required_field"] for msg in e.messages)

def test_validate_with_read_only_field():
    child_field = Field(read_only=True)
    schema = Schema(fields={"read_only_field": child_field})
    result = schema.validate({"read_only_field": "value"})
    assert "read_only_field" not in result

def test_validate_with_field_having_default():
    child_field = Field(default="default_value")
    schema = Schema(fields={"field_with_default": child_field})
    result = schema.validate({})
    assert result["field_with_default"] == "default_value"

def test_validate_with_valid_child_field():
    child_field = Field()
    schema = Schema(fields={"child": child_field})
    result = schema.validate({"child": "valid"})
    assert result["child"] == "valid"

def test_validate_with_invalid_child_field():
    child_field = Field(allow_null=False)
    schema = Schema(fields={"child": child_field})
    try:
        schema.validate({"child": None})
        assert False
    except ValidationError as e:
        assert any(msg.code == "null" and msg.index == ["child"] for msg in e.messages)

def test_validate_with_multiple_errors():
    child_field1 = Field()
    child_field2 = Field()
    schema = Schema(fields={"req1": child_field1, "req2": child_field2})
    try:
        schema.validate({})
        assert False
    except ValidationError as e:
        messages = e.messages
        required_errors = [msg for msg in messages if msg.code == "required"]
        assert len(required_errors) == 2

def test_validate_with_nested_valid_data():
    nested_schema = Schema(fields={"nested_field": Field()})
    schema = Schema(fields={"nested": nested_schema})
    result = schema.validate({"nested": {"nested_field": "value"}})
    assert result["nested"]["nested_field"] == "value"

def test_validate_with_nested_invalid_data():
    nested_schema = Schema(fields={"nested_field": Field(allow_null=False)})
    schema = Schema(fields={"nested": nested_schema})
    try:
        schema.validate({"nested": {"nested_field": None}})
        assert False
    except ValidationError as e:
        assert any(msg.index == ["nested", "nested_field"] for msg in e.messages)


# LLM-generated content at query #6
#--------------------------

def test_validate_returns_none_when_value_is_none_and_allow_null_is_true():
    field = Field(allow_null=True)
    schema = Schema(fields={}, allow_null=True)
    result = schema.validate(None)
    assert result is None

def test_validate_raises_null_error_when_value_is_none_and_allow_null_is_false():
    schema = Schema(fields={}, allow_null=False)
    try:
        schema.validate(None)
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "null"

def test_validate_raises_type_error_when_value_is_not_dict_or_mapping():
    schema = Schema(fields={})
    try:
        schema.validate("not a dict")
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "type"

def test_validate_raises_invalid_key_error_when_key_is_not_string():
    schema = Schema(fields={})
    try:
        schema.validate({123: "value"})
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "invalid_key"

def test_validate_raises_required_error_when_required_key_is_missing():
    field = Field(read_only=False)
    field.has_default = lambda: False
    schema = Schema(fields={"required_key": field})
    try:
        schema.validate({})
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "required"

def test_validate_uses_default_value_when_key_is_missing_and_field_has_default():
    field = Field(read_only=False)
    field.has_default = lambda: True
    field.get_default_value = lambda: "default"
    field.validate_or_error = lambda x: (x, None)
    schema = Schema(fields={"key_with_default": field})
    result = schema.validate({})
    assert result == {"key_with_default": "default"}

def test_validate_skips_read_only_fields():
    field = Field(read_only=True)
    schema = Schema(fields={"read_only_key": field})
    result = schema.validate({"read_only_key": "value"})
    assert result == {}

def test_validate_includes_validated_child_values():
    field = Field(read_only=False)
    field.has_default = lambda: False
    field.validate_or_error = lambda x: (x.upper(), None)
    schema = Schema(fields={"child_key": field})
    result = schema.validate({"child_key": "value"})
    assert result == {"child_key": "VALUE"}

def test_validate_collects_child_validation_errors():
    field = Field(read_only=False)
    field.has_default = lambda: False
    error_message = Message(text="Child error", code="child_error", index=[])
    error = ValidationError(messages=[error_message])
    field.validate_or_error = lambda x: (None, error)
    schema = Schema(fields={"child_key": field})
    try:
        schema.validate({"child_key": "value"})
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "child_error"

def test_validate_returns_validated_dict_with_multiple_fields():
    field1 = Field(read_only=False)
    field1.has_default = lambda: False
    field1.validate_or_error = lambda x: (x + 1, None)
    field2 = Field(read_only=False)
    field2.has_default = lambda: True
    field2.get_default_value = lambda: 100
    field2.validate_or_error = lambda x: (x, None)
    schema = Schema(fields={"field1": field1, "field2": field2})
    result = schema.validate({"field1": 5})
    assert result == {"field1": 6, "field2": 100}


# LLM-generated content at query #7
#--------------------------

def test_validate_with_child_schema_error():
    from my_module import Field, Schema, ValidationError
    class MockField(Field):
        def __init__(self, read_only=False, has_default=False, default_value=None, validation_error=None):
            super().__init__()
            self._read_only = read_only
            self._has_default = has_default
            self._default_value = default_value
            self._validation_error = validation_error
        def read_only(self):
            return self._read_only
        def has_default(self):
            return self._has_default
        def get_default_value(self):
            return self._default_value
        def validate_or_error(self, value):
            if self._validation_error:
                return None, self._validation_error
            return value, None
    error_mock = ValidationError(messages=[])
    error_mock.messages = lambda add_prefix=None: [f"error for {add_prefix}"]
    child_field = MockField(validation_error=error_mock)
    schema = Schema(fields={"key": child_field})
    value = {"key": "some_value"}
    try:
        schema.validate(value)
    except ValidationError as e:
        assert any("error for key" in str(msg) for msg in e.messages)


# LLM-generated content at query #8
#--------------------------

def test_validate_returns_none_when_value_is_none_and_allow_null_is_true():
    field = Field(allow_null=True)
    schema = Schema(fields={}, allow_null=True)
    result = schema.validate(None)
    assert result is None

def test_validate_raises_null_error_when_value_is_none_and_allow_null_is_false():
    schema = Schema(fields={}, allow_null=False)
    try:
        schema.validate(None)
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "null"

def test_validate_raises_type_error_when_value_is_not_a_mapping():
    schema = Schema(fields={})
    try:
        schema.validate("not a dict")
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "type"

def test_validate_raises_invalid_key_error_when_key_is_not_string():
    schema = Schema(fields={})
    try:
        schema.validate({123: "value"})
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "invalid_key"

def test_validate_raises_required_error_when_required_key_is_missing():
    field = Field()
    schema = Schema(fields={"required_field": field})
    try:
        schema.validate({})
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "required"

def test_validate_uses_default_value_when_key_is_missing_and_field_has_default():
    field = Field(default="default_value")
    schema = Schema(fields={"field_with_default": field})
    result = schema.validate({})
    assert result == {"field_with_default": "default_value"}

def test_validate_skips_read_only_fields():
    field = Field(read_only=True)
    schema = Schema(fields={"read_only_field": field})
    result = schema.validate({"read_only_field": "ignored"})
    assert result == {}

def test_validate_includes_validated_child_field_values():
    child_field = Field()
    schema = Schema(fields={"child": child_field})
    result = schema.validate({"child": "value"})
    assert result == {"child": "value"}

def test_validate_collects_child_field_validation_errors():
    child_field = Field()
    child_field.validate = lambda x: (_ for _ in ()).throw(ValidationError(messages=[Message(text="error", code="error", index=[])]))
    schema = Schema(fields={"child": child_field})
    try:
        schema.validate({"child": "value"})
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "error"

def test_validate_returns_validated_dict_with_multiple_fields():
    field1 = Field()
    field2 = Field(default="default2")
    schema = Schema(fields={"key1": field1, "key2": field2})
    result = schema.validate({"key1": "value1"})
    assert result == {"key1": "value1", "key2": "default2"}

def test_validate_handles_nested_errors_with_prefix():
    child_field = Field()
    child_field.validate = lambda x: (_ for _ in ()).throw(ValidationError(messages=[Message(text="child error", code="child_error", index=[])]))
    schema = Schema(fields={"nested": child_field})
    try:
        schema.validate({"nested": "value"})
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].index == ["nested"]


# LLM-generated content at query #9
#--------------------------

def test_validate_with_valid_child_field():
    child_field = Field()
    child_field.validate_or_error = lambda x: (x, None)
    schema = Schema(fields={"key": child_field})
    value = {"key": "valid"}
    result = schema.validate(value)
    assert result == {"key": "valid"}


# LLM-generated content at query #10
#--------------------------

def test_validate_with_child_schema_no_error():
    class MockField:
        read_only = False
        def has_default(self):
            return False
        def validate_or_error(self, value):
            return value, None
    mock_field = MockField()
    schema = Schema(fields={"key": mock_field})
    result = schema.validate({"key": "value"})
    assert result == {"key": "value"}


# LLM-generated content at query #11
#--------------------------

def test_validate_with_child_schema_no_error():
    child_schema = Field()
    child_schema.validate_or_error = lambda x: (x, None)
    schema = Schema(fields={"key": child_schema})
    value = {"key": "value"}
    result = schema.validate(value)
    assert result == {"key": "value"}


# LLM-generated content at query #12
#--------------------------

def test_validate_returns_none_when_value_is_none_and_allow_null_is_true():
    field = Field(allow_null=True)
    schema = Schema(fields={}, allow_null=True)
    result = schema.validate(None)
    assert result is None

def test_validate_raises_null_error_when_value_is_none_and_allow_null_is_false():
    schema = Schema(fields={}, allow_null=False)
    try:
        schema.validate(None)
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "null"

def test_validate_raises_type_error_when_value_is_not_dict_or_mapping():
    schema = Schema(fields={})
    try:
        schema.validate("not a dict")
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "type"

def test_validate_raises_invalid_key_error_when_key_is_not_string():
    schema = Schema(fields={})
    try:
        schema.validate({123: "value"})
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "invalid_key"

def test_validate_raises_required_error_when_required_key_is_missing():
    child_field = Field()
    schema = Schema(fields={"required_key": child_field})
    try:
        schema.validate({})
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "required"

def test_validate_uses_default_value_when_key_is_missing_and_field_has_default():
    child_field = Field(default="default_value")
    schema = Schema(fields={"optional_key": child_field})
    result = schema.validate({})
    assert result == {"optional_key": "default_value"}

def test_validate_skips_read_only_fields():
    child_field = Field(read_only=True)
    schema = Schema(fields={"read_only_key": child_field})
    result = schema.validate({"read_only_key": "value"})
    assert result == {}

def test_validate_includes_validated_child_field_values():
    child_field = Field()
    schema = Schema(fields={"child_key": child_field})
    result = schema.validate({"child_key": "child_value"})
    assert result == {"child_key": "child_value"}

def test_validate_collects_child_field_validation_errors():
    child_field = Field()
    child_field.validate = lambda x: (_ for _ in ()).throw(ValidationError(messages=[Message(text="error", code="error", index=[])]))
    schema = Schema(fields={"child_key": child_field})
    try:
        schema.validate({"child_key": "invalid"})
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "error"

def test_validate_returns_validated_dict_with_multiple_fields():
    child_field1 = Field()
    child_field2 = Field(default="default")
    schema = Schema(fields={"key1": child_field1, "key2": child_field2})
    result = schema.validate({"key1": "value1"})
    assert result == {"key1": "value1", "key2": "default"}

def test_validate_combines_multiple_errors():
    child_field = Field()
    schema = Schema(fields={"required_key": child_field})
    try:
        schema.validate({123: "value"})
        assert False
    except ValidationError as e:
        assert len(e.messages) == 2
        codes = [msg.code for msg in e.messages]
        assert "invalid_key" in codes
        assert "required" in codes


# LLM-generated content at query #13
#--------------------------

def test_validate_with_error_in_child_schema():
    child_field = Field()
    child_field.validate_or_error = lambda x: (None, ValidationError(messages=[Message(text="Invalid", code="invalid", index=[])]))
    child_field.read_only = False
    child_field.has_default = lambda: False
    schema = Schema(fields={"key": child_field})
    value = {"key": "invalid_value"}
    try:
        schema.validate(value)
    except ValidationError as e:
        assert any(msg.code == "invalid" for msg in e.messages)


# LLM-generated content at query #14
#--------------------------

def test_validate_returns_none_when_value_is_none_and_allow_null_is_true():
    field = Field(allow_null=True)
    schema = Schema(fields={}, allow_null=True)
    result = schema.validate(None)
    assert result is None

def test_validate_raises_null_error_when_value_is_none_and_allow_null_is_false():
    schema = Schema(fields={}, allow_null=False)
    try:
        schema.validate(None)
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "null"

def test_validate_raises_type_error_when_value_is_not_mapping():
    schema = Schema(fields={})
    try:
        schema.validate("not a dict")
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "type"

def test_validate_raises_invalid_key_error_when_key_is_not_string():
    schema = Schema(fields={})
    try:
        schema.validate({123: "value"})
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "invalid_key"
        assert e.messages[0].index == [123]

def test_validate_raises_required_error_when_required_key_is_missing():
    field = Field()
    schema = Schema(fields={"required_key": field})
    try:
        schema.validate({})
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "required"
        assert e.messages[0].index == ["required_key"]

def test_validate_uses_default_value_when_key_is_missing_and_field_has_default():
    field = Field(default="default_value")
    schema = Schema(fields={"key_with_default": field})
    result = schema.validate({})
    assert result == {"key_with_default": "default_value"}

def test_validate_skips_read_only_fields():
    field = Field(read_only=True)
    schema = Schema(fields={"read_only_key": field})
    result = schema.validate({"read_only_key": "some_value"})
    assert result == {}

def test_validate_includes_validated_child_field_values():
    child_field = Field()
    schema = Schema(fields={"child": child_field})
    result = schema.validate({"child": "child_value"})
    assert result == {"child": "child_value"}

def test_validate_collects_child_field_validation_errors():
    child_field = Field()
    child_field.validate = lambda x: (_ for _ in ()).throw(ValidationError(messages=[Message(text="Child error", code="child_error", index=[])]))
    schema = Schema(fields={"child": child_field})
    try:
        schema.validate({"child": "bad_value"})
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "child_error"

def test_validate_returns_validated_dict_with_multiple_fields():
    field1 = Field()
    field2 = Field(default="default2")
    schema = Schema(fields={"key1": field1, "key2": field2})
    result = schema.validate({"key1": "value1"})
    assert result == {"key1": "value1", "key2": "default2"}

def test_validate_aggregates_multiple_errors():
    required_field = Field()
    schema = Schema(fields={"required": required_field})
    try:
        schema.validate({"invalid_key": 123})
        assert False
    except ValidationError as e:
        codes = [msg.code for msg in e.messages]
        assert "invalid_key" in codes
        assert "required" in codes


# LLM-generated content at query #15
#--------------------------

def test_validate_with_no_error_from_child_schema():
    class MockField:
        read_only = False
        def has_default(self):
            return False
        def get_default_value(self):
            return None
        def validate_or_error(self, item):
            return item, None
    fields = {"key": MockField()}
    schema = Schema(fields=fields)
    value = {"key": "value"}
    result = schema.validate(value)
    assert result == {"key": "value"}


# LLM-generated content at query #16
#--------------------------

def test_validate_with_child_schema_error():
    class MockField:
        read_only = False
        def has_default(self):
            return False
        def get_default_value(self):
            return None
        def validate_or_error(self, item):
            return None, MockError()
    class MockError:
        def messages(self, add_prefix):
            return [f"error for {add_prefix}"]
    fields = {"key": MockField()}
    schema = Schema(fields)
    value = {"key": "invalid"}
    try:
        schema.validate(value)
    except ValidationError:
        pass


# LLM-generated content at query #17
#--------------------------

def test_validate_with_null_and_allow_null():
    field = Field(allow_null=True)
    schema = Schema(fields={}, allow_null=True)
    result = schema.validate(None)
    assert result is None

def test_validate_with_null_and_not_allow_null():
    schema = Schema(fields={}, allow_null=False)
    try:
        schema.validate(None)
        assert False
    except ValidationError as e:
        assert any(msg.code == "null" for msg in e.messages)

def test_validate_with_non_dict_value():
    schema = Schema(fields={})
    try:
        schema.validate("not a dict")
        assert False
    except ValidationError as e:
        assert any(msg.code == "type" for msg in e.messages)

def test_validate_with_invalid_non_string_key():
    schema = Schema(fields={})
    try:
        schema.validate({123: "value"})
        assert False
    except ValidationError as e:
        assert any(msg.code == "invalid_key" for msg in e.messages)

def test_validate_missing_required_field():
    child_field = Field()
    schema = Schema(fields={"required_field": child_field})
    try:
        schema.validate({})
        assert False
    except ValidationError as e:
        assert any(msg.code == "required" and msg.index == ["required_field"] for msg in e.messages)

def test_validate_with_read_only_field_ignored():
    child_field = Field(read_only=True)
    schema = Schema(fields={"read_only_field": child_field})
    result = schema.validate({})
    assert result == {}

def test_validate_with_field_using_default():
    child_field = Field(default="default_value")
    schema = Schema(fields={"field_with_default": child_field})
    result = schema.validate({})
    assert result == {"field_with_default": "default_value"}

def test_validate_with_valid_child_field():
    child_field = Field()
    schema = Schema(fields={"child": child_field})
    result = schema.validate({"child": "valid"})
    assert result == {"child": "valid"}

def test_validate_with_invalid_child_field():
    child_field = Field(allow_null=False)
    schema = Schema(fields={"child": child_field})
    try:
        schema.validate({"child": None})
        assert False
    except ValidationError as e:
        assert any(msg.index == ["child"] for msg in e.messages)

def test_validate_with_multiple_errors():
    child_field1 = Field()
    child_field2 = Field()
    schema = Schema(fields={"req1": child_field1, "req2": child_field2})
    try:
        schema.validate({})
        assert False
    except ValidationError as e:
        codes = [msg.code for msg in e.messages]
        assert "required" in codes
        assert len(e.messages) >= 2


# LLM-generated content at query #18
#--------------------------

def test_validate_with_error_in_child_schema():
    class MockField:
        read_only = False
        def has_default(self):
            return False
        def validate_or_error(self, item):
            return None, ValidationError(messages=[Message(text="error", code="error", index=[])])
    fields = {"key": MockField()}
    schema = Schema(fields=fields)
    value = {"key": "value"}
    try:
        schema.validate(value)
    except ValidationError:
        pass


