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

def test_validate_null_not_allowed():
    schema = Schema(fields={}, allow_null=False)
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

def test_validate_invalid_key_type():
    field = Field()
    schema = Schema(fields={"valid": field})
    try:
        schema.validate({1: "value"})
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "invalid_key"
        assert e.messages[0].index == [1]

def test_validate_missing_required():
    field = Field()
    schema = Schema(fields={"required_field": field})
    try:
        schema.validate({})
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "required"
        assert e.messages[0].index == ["required_field"]

def test_validate_read_only_field_ignored():
    field = Field(read_only=True)
    schema = Schema(fields={"read_only": field})
    result = schema.validate({"read_only": "ignored"})
    assert result == {}

def test_validate_field_with_default():
    field = Field(default="default_value")
    schema = Schema(fields={"with_default": field})
    result = schema.validate({})
    assert result == {"with_default": "default_value"}

def test_validate_field_validation_error():
    field = Field()
    schema = Schema(fields={"test": field})
    try:
        schema.validate({"test": None})
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "null"

def test_validate_success():
    field = Field()
    schema = Schema(fields={"name": field})
    result = schema.validate({"name": "John"})
    assert result == {"name": "John"}

def test_validate_multiple_errors():
    field1 = Field()
    field2 = Field()
    schema = Schema(fields={"req1": field1, "req2": field2})
    try:
        schema.validate({"invalid_key": 123})
        assert False
    except ValidationError as e:
        codes = [msg.code for msg in e.messages]
        assert "invalid_key" in codes
        assert "required" in codes
        assert len(e.messages) == 3


# LLM-generated content at query #2
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
        assert e.detail == "May not be null."

def test_validate_calls_target_validate_with_value():
    mock_definitions = {"target_name": MockField()}
    mock_definitions["target_name"].validate = Mock(return_value="validated_value")
    reference = Reference(to="target_name", definitions=mock_definitions)
    result = reference.validate("input_value")
    assert result == "validated_value"
    mock_definitions["target_name"].validate.assert_called_once_with("input_value")

def test_validate_passes_through_target_validation_error():
    mock_definitions = {"target_name": MockField()}
    mock_definitions["target_name"].validate = Mock(side_effect=ValidationError("Target error"))
    reference = Reference(to="target_name", definitions=mock_definitions)
    try:
        reference.validate("input_value")
        assert False
    except ValidationError as e:
        assert e.detail == "Target error"


# LLM-generated content at query #3
#--------------------------

def test_validate_allows_null_when_allow_null_is_true():
    mock_definitions = {"target": MockField()}
    reference = Reference(to="target", definitions=mock_definitions, allow_null=True)
    result = reference.validate(None)
    assert result is None

def test_validate_raises_error_for_null_when_allow_null_is_false():
    mock_definitions = {"target": MockField()}
    reference = Reference(to="target", definitions=mock_definitions, allow_null=False)
    try:
        reference.validate(None)
        assert False
    except ValidationError as e:
        assert e.message == "May not be null."

def test_validate_calls_target_validate_with_value():
    mock_definitions = {"target": MockField()}
    reference = Reference(to="target", definitions=mock_definitions, allow_null=False)
    test_value = {"id": 1}
    reference.validate(test_value)
    assert mock_definitions["target"].validate_called_with == test_value

def test_validate_returns_target_validate_result():
    mock_definitions = {"target": MockField()}
    mock_definitions["target"].validate_return = "validated_result"
    reference = Reference(to="target", definitions=mock_definitions, allow_null=False)
    test_value = {"id": 1}
    result = reference.validate(test_value)
    assert result == "validated_result"


# LLM-generated content at query #4
#--------------------------

def test_setitem_adds_new_key():
    definitions = Definitions()
    definitions["key"] = "value"
    assert definitions["key"] == "value"

def test_setitem_raises_assertion_on_duplicate_key():
    definitions = Definitions()
    definitions["key"] = "value"
    try:
        definitions["key"] = "new_value"
        assert False
    except AssertionError as e:
        assert "Definition for 'key' has already been set." in str(e)

def test_setitem_works_with_empty_dict():
    definitions = Definitions()
    definitions["a"] = 1
    assert len(definitions) == 1
    assert definitions["a"] == 1

def test_setitem_multiple_unique_keys():
    definitions = Definitions()
    definitions["key1"] = "value1"
    definitions["key2"] = "value2"
    definitions["key3"] = "value3"
    assert definitions["key1"] == "value1"
    assert definitions["key2"] == "value2"
    assert definitions["key3"] == "value3"
    assert len(definitions) == 3

def test_setitem_after_deletion():
    definitions = Definitions({"a": 1})
    del definitions["a"]
    definitions["a"] = 2
    assert definitions["a"] == 2

def test_setitem_with_none_value():
    definitions = Definitions()
    definitions["key"] = None
    assert definitions["key"] is None


# LLM-generated content at query #5
#--------------------------

def test_validate_allow_null():
    definitions = {"target": MockField()}
    reference = Reference(to="target", definitions=definitions, allow_null=True)
    result = reference.validate(None)
    assert result is None

def test_validate_not_allow_null():
    definitions = {"target": MockField()}
    reference = Reference(to="target", definitions=definitions, allow_null=False)
    try:
        reference.validate(None)
        assert False
    except ValidationError as e:
        assert e.message == "May not be null."

def test_validate_with_value():
    definitions = {"target": MockField()}
    mock_field = MockField()
    mock_field.validate = lambda x: x
    definitions["target"] = mock_field
    reference = Reference(to="target", definitions=definitions, allow_null=False)
    value = {"id": 1}
    result = reference.validate(value)
    assert result == value

def test_validate_calls_target_validate():
    definitions = {"target": MockField()}
    mock_field = MockField()
    mock_field.validate = lambda x: x
    definitions["target"] = mock_field
    reference = Reference(to="target", definitions=definitions, allow_null=False)
    value = {"id": 1}
    result = reference.validate(value)
    assert result == value


# LLM-generated content at query #6
#--------------------------

def test_validate_with_no_error_from_child_schema():
    class MockField:
        read_only = False
        def has_default(self):
            return False
        def get_default_value(self):
            return None
        def validate_or_error(self, item):
            return "validated_value", None
    mock_field = MockField()
    fields = {"field1": mock_field}
    schema = Schema(fields=fields)
    value = {"field1": "some_value"}
    result = schema.validate(value)
    assert result == {"field1": "validated_value"}


# LLM-generated content at query #7
#--------------------------

def test_validate_with_valid_child_schema():
    class MockField:
        read_only = False
        def has_default(self):
            return False
        def validate_or_error(self, value):
            return value, None
    field = MockField()
    schema = Schema(fields={"key": field})
    value = {"key": "valid"}
    result = schema.validate(value)
    assert result == {"key": "valid"}


# LLM-generated content at query #8
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


# LLM-generated content at query #9
#--------------------------

def test_validate_allow_null():
    definitions = {"target": MockField()}
    reference = Reference(to="target", definitions=definitions, allow_null=True)
    result = reference.validate(None)
    assert result is None

def test_validate_not_allow_null():
    definitions = {"target": MockField()}
    reference = Reference(to="target", definitions=definitions, allow_null=False)
    try:
        reference.validate(None)
        assert False
    except ValidationError as e:
        assert e.message == "May not be null."

def test_validate_delegates_to_target():
    definitions = {"target": MockField()}
    reference = Reference(to="target", definitions=definitions, allow_null=False)
    value = {"key": "value"}
    result = reference.validate(value)
    assert result == value

def test_validate_target_validation_error():
    definitions = {"target": MockField(raises_error=True)}
    reference = Reference(to="target", definitions=definitions, allow_null=False)
    value = {"key": "value"}
    try:
        reference.validate(value)
        assert False
    except ValidationError as e:
        assert e.message == "Target validation error."


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

def test_validate_with_missing_required_field():
    child_field = Field(read_only=False)
    child_field.has_default = lambda: False
    schema = Schema(fields={"required_key": child_field})
    try:
        schema.validate({})
        assert False
    except ValidationError as e:
        assert any(msg.code == "required" and msg.index == ["required_key"] for msg in e.messages)

def test_validate_with_read_only_field_ignored():
    child_field = Field(read_only=True)
    schema = Schema(fields={"read_only_key": child_field})
    result = schema.validate({"read_only_key": "some_value"})
    assert "read_only_key" not in result

def test_validate_with_field_has_default_and_missing():
    child_field = Field(read_only=False)
    child_field.has_default = lambda: True
    child_field.get_default_value = lambda: "default_value"
    child_field.validate_or_error = lambda x: (x, None)
    schema = Schema(fields={"key_with_default": child_field})
    result = schema.validate({})
    assert result["key_with_default"] == "default_value"

def test_validate_with_field_validation_error():
    child_field = Field(read_only=False)
    child_field.has_default = lambda: False
    error_msg = Message(text="error", code="error", index=[])
    error = ValidationError(messages=[error_msg])
    child_field.validate_or_error = lambda x: (None, error)
    schema = Schema(fields={"problem_key": child_field})
    try:
        schema.validate({"problem_key": "value"})
        assert False
    except ValidationError as e:
        assert any(msg.code == "error" for msg in e.messages)

def test_validate_successful_with_all_fields():
    child_field = Field(read_only=False)
    child_field.has_default = lambda: False
    child_field.validate_or_error = lambda x: (x, None)
    schema = Schema(fields={"valid_key": child_field})
    result = schema.validate({"valid_key": "valid_value"})
    assert result == {"valid_key": "valid_value"}


# LLM-generated content at query #11
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
        schema.validate({1: "value"})
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "invalid_key"
        assert e.messages[0].index == [1]

def test_validate_raises_required_error_for_missing_required_field():
    child_field = Field()
    schema = Schema(fields={"required_field": child_field})
    try:
        schema.validate({})
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "required"
        assert e.messages[0].index == ["required_field"]

def test_validate_does_not_require_read_only_field():
    child_field = Field(read_only=True)
    schema = Schema(fields={"read_only_field": child_field})
    result = schema.validate({})
    assert result == {}

def test_validate_uses_default_value_for_missing_field_with_default():
    child_field = Field(default="default_value")
    schema = Schema(fields={"field_with_default": child_field})
    result = schema.validate({})
    assert result == {"field_with_default": "default_value"}

def test_validate_includes_validated_child_field_value():
    child_field = Field()
    schema = Schema(fields={"child": child_field})
    result = schema.validate({"child": "child_value"})
    assert result == {"child": "child_value"}

def test_validate_collects_child_field_validation_errors():
    child_field = Field()
    schema = Schema(fields={"child": child_field})
    try:
        schema.validate({"child": None})
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "null"

def test_validate_aggregates_multiple_errors():
    child_field1 = Field()
    child_field2 = Field()
    schema = Schema(fields={"child1": child_field1, "child2": child_field2})
    try:
        schema.validate({"child1": None, "child2": None})
        assert False
    except ValidationError as e:
        assert len(e.messages) == 2

def test_validate_ignores_extra_keys_not_in_schema():
    child_field = Field()
    schema = Schema(fields={"defined": child_field})
    result = schema.validate({"defined": "value", "extra": "ignored"})
    assert result == {"defined": "value"}


# LLM-generated content at query #12
#--------------------------

def test_validate_with_valid_child_schema():
    class MockField:
        read_only = False
        def has_default(self):
            return False
        def validate_or_error(self, value):
            return value, None
    field = MockField()
    schema = Schema(fields={"key": field})
    result = schema.validate({"key": "value"})
    assert result == {"key": "value"}


# LLM-generated content at query #13
#--------------------------

def test_validate_with_error_in_child_schema():
    from myapp import Schema, Field, ValidationError
    class MockField(Field):
        def __init__(self, read_only=False, has_default=False, default_value=None, will_error=False):
            super().__init__()
            self._read_only = read_only
            self._has_default = has_default
            self._default_value = default_value
            self._will_error = will_error
        def read_only(self):
            return self._read_only
        def has_default(self):
            return self._has_default
        def get_default_value(self):
            return self._default_value
        def validate_or_error(self, value):
            if self._will_error:
                from myapp import ValidationError
                error = ValidationError(messages=[])
                return None, error
            return value, None
    error_field = MockField(will_error=True)
    fields = {"test_key": error_field}
    schema = Schema(fields=fields)
    value = {"test_key": "some_value"}
    try:
        schema.validate(value)
    except ValidationError as e:
        pass
    else:
        assert False, "Expected ValidationError"


# LLM-generated content at query #14
#--------------------------

def test_validate_when_value_is_none_and_allow_null_is_true():
    mock_definitions = {}
    mock_target = Mock()
    mock_target.validate = Mock(return_value="validated")
    mock_definitions.__getitem__ = Mock(return_value=mock_target)
    field = Reference(to="some_ref", definitions=mock_definitions, allow_null=True)
    result = field.validate(None)
    assert result is None

def test_validate_when_value_is_none_and_allow_null_is_false():
    mock_definitions = {}
    mock_target = Mock()
    mock_target.validate = Mock(return_value="validated")
    mock_definitions.__getitem__ = Mock(return_value=mock_target)
    field = Reference(to="some_ref", definitions=mock_definitions, allow_null=False)
    field.validation_error = Mock(side_effect=Exception("null error"))
    try:
        field.validate(None)
        assert False
    except Exception as e:
        assert str(e) == "null error"

def test_validate_when_value_is_not_none():
    mock_definitions = {}
    mock_target = Mock()
    mock_target.validate = Mock(return_value="validated")
    mock_definitions.__getitem__ = Mock(return_value=mock_target)
    field = Reference(to="some_ref", definitions=mock_definitions, allow_null=False)
    result = field.validate("some_value")
    assert result == "validated"
    mock_target.validate.assert_called_once_with("some_value")


# LLM-generated content at query #15
#--------------------------

def test_validate_child_schema_error():
    from myapp.schemas import Schema, Field, ValidationError, Message
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
            return None, ValidationError(messages=[Message(text="error", code="error", index=[])])
    child_field = MockField(read_only=False, has_default_val=False)
    schema = Schema(fields={"key": child_field})
    value = {"key": "some_value"}
    try:
        schema.validate(value)
    except ValidationError as e:
        pass


# LLM-generated content at query #16
#--------------------------

def test_validate_child_schema_error():
    class MockField:
        read_only = False
        def has_default(self):
            return False
        def validate_or_error(self, item):
            return None, MockError()
    class MockError:
        def messages(self, add_prefix):
            return ["error"]
    schema = Schema(fields={"key": MockField()})
    value = {"key": "invalid"}
    try:
        schema.validate(value)
    except ValidationError:
        pass


# LLM-generated content at query #17
#--------------------------

def test_validate_with_child_schema_error():
    class MockField:
        read_only = False
        def has_default(self):
            return False
        def validate_or_error(self, item):
            from dataclasses import dataclass
            @dataclass
            class MockError:
                messages: list
                def messages(self, add_prefix):
                    return [f"error for {add_prefix}"]
            return None, MockError(messages=[])
    mock_field = MockField()
    schema = Schema(fields={"key": mock_field})
    mock_field.validate_or_error = lambda x: (None, MockField.MockError(messages=[]))
    value = {"key": "invalid"}
    try:
        schema.validate(value)
    except ValidationError as e:
        pass


# LLM-generated content at query #18
#--------------------------

def test_validate_with_error_in_child_schema():
    class MockField:
        read_only = False
        def has_default(self):
            return False
        def validate_or_error(self, value):
            return None, ValidationError(messages=[Message(text="error", code="error", index=[])])
    fields = {"field": MockField()}
    schema = Schema(fields=fields)
    value = {"field": "invalid"}
    try:
        schema.validate(value)
    except ValidationError as e:
        pass
    else:
        assert False, "Expected ValidationError"


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_validate_allows_null_when_allow_null_is_true():
    field = Field(allow_null=True)
    schema = Schema(fields={}, allow_null=True)
    result = schema.validate(None)
    assert result is None

def test_validate_raises_error_on_null_when_allow_null_is_false():
    schema = Schema(fields={}, allow_null=False)
    try:
        schema.validate(None)
        assert False
    except ValidationError as e:
        assert any(m.code == "null" for m in e.messages)

def test_validate_raises_error_on_non_dict_value():
    schema = Schema(fields={})
    try:
        schema.validate("not a dict")
        assert False
    except ValidationError as e:
        assert any(m.code == "type" for m in e.messages)

def test_validate_raises_error_on_non_string_key():
    schema = Schema(fields={})
    try:
        schema.validate({123: "value"})
        assert False
    except ValidationError as e:
        assert any(m.code == "invalid_key" for m in e.messages)

def test_validate_raises_error_on_missing_required_field():
    child_field = Field()
    schema = Schema(fields={"required_field": child_field})
    try:
        schema.validate({})
        assert False
    except ValidationError as e:
        assert any(m.code == "required" and m.index == ["required_field"] for m in e.messages)

def test_validate_uses_default_value_for_missing_field_with_default():
    child_field = Field(default="default_value")
    schema = Schema(fields={"field_with_default": child_field})
    result = schema.validate({})
    assert result == {"field_with_default": "default_value"}

def test_validate_skips_read_only_field():
    child_field = Field(read_only=True)
    schema = Schema(fields={"read_only_field": child_field})
    result = schema.validate({})
    assert result == {}

def test_validate_includes_valid_field():
    child_field = Field()
    schema = Schema(fields={"valid_field": child_field})
    result = schema.validate({"valid_field": "some_value"})
    assert result == {"valid_field": "some_value"}

def test_validate_collects_errors_from_child_fields():
    child_field = Field()
    child_field.validate = lambda x: (_ for _ in ()).throw(ValidationError(messages=[Message(text="Child error", code="child_error", index=[])]))
    schema = Schema(fields={"problem_field": child_field})
    try:
        schema.validate({"problem_field": "bad_value"})
        assert False
    except ValidationError as e:
        assert any(m.code == "child_error" for m in e.messages)

def test_validate_returns_validated_dict_with_multiple_fields():
    child_field1 = Field()
    child_field2 = Field(default="default2")
    schema = Schema(fields={"field1": child_field1, "field2": child_field2})
    result = schema.validate({"field1": "value1"})
    assert result == {"field1": "value1", "field2": "default2"}


# LLM-generated content at query #2
#--------------------------

def test_validate_allows_null_when_allow_null_is_true():
    mock_definitions = {}
    mock_target = Mock()
    mock_target.validate = Mock(return_value="validated_value")
    mock_definitions.__getitem__ = Mock(return_value=mock_target)
    reference = Reference(to="some_key", definitions=mock_definitions, allow_null=True)
    result = reference.validate(None)
    assert result is None

def test_validate_raises_error_for_null_when_allow_null_is_false():
    mock_definitions = {}
    reference = Reference(to="some_key", definitions=mock_definitions, allow_null=False)
    try:
        reference.validate(None)
        assert False
    except Exception as e:
        assert str(e) == "May not be null."

def test_validate_calls_target_validate_with_value():
    mock_definitions = {}
    mock_target = Mock()
    mock_target.validate = Mock(return_value="validated_value")
    mock_definitions.__getitem__ = Mock(return_value=mock_target)
    reference = Reference(to="some_key", definitions=mock_definitions)
    result = reference.validate("input_value")
    mock_definitions.__getitem__.assert_called_once_with("some_key")
    mock_target.validate.assert_called_once_with("input_value")
    assert result == "validated_value"


# LLM-generated content at query #3
#--------------------------

def test_setitem_adds_new_key():
    definitions = Definitions()
    definitions["key"] = "value"
    assert definitions["key"] == "value"

def test_setitem_raises_assertion_on_duplicate_key():
    definitions = Definitions()
    definitions["key"] = "value"
    try:
        definitions["key"] = "new_value"
        assert False, "Expected AssertionError"
    except AssertionError as e:
        assert "Definition for 'key' has already been set." in str(e)

def test_setitem_works_with_empty_dict():
    definitions = Definitions()
    definitions["test"] = 123
    assert definitions["test"] == 123

def test_setitem_works_with_prefilled_dict():
    definitions = Definitions({"a": 1})
    definitions["b"] = 2
    assert definitions["a"] == 1
    assert definitions["b"] == 2

def test_setitem_key_not_in_definitions():
    definitions = Definitions()
    definitions["new"] = "item"
    assert "new" in definitions._definitions

def test_setitem_value_stored_correctly():
    definitions = Definitions()
    definitions["number"] = 42
    assert definitions._definitions["number"] == 42


# LLM-generated content at query #4
#--------------------------

def test_validate_with_child_schema_error():
    from myapp.schemas import Schema, Field, ValidationError, Message
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
    mock_error = ValidationError(messages=[Message(text="Invalid", code="invalid", index=[])])
    child_field = MockField(validation_error=mock_error)
    schema = Schema(fields={"key": child_field})
    value = {"key": "some_value"}
    try:
        schema.validate(value)
    except ValidationError as e:
        pass


# LLM-generated content at query #5
#--------------------------

def test_serialize_with_none_object():
    from myapp import Schema, StringField
    fields = {"name": StringField()}
    schema = Schema(fields=fields)
    result = schema.serialize(None)
    assert result is None

def test_serialize_with_dict_object():
    from myapp import Schema, StringField, IntegerField
    fields = {"name": StringField(), "age": IntegerField()}
    schema = Schema(fields=fields)
    obj = {"name": "John", "age": 30}
    result = schema.serialize(obj)
    assert result == {"name": "John", "age": 30}

def test_serialize_with_object_attributes():
    from myapp import Schema, StringField, IntegerField
    fields = {"name": StringField(), "age": IntegerField()}
    schema = Schema(fields=fields)
    class Person:
        def __init__(self, name, age):
            self.name = name
            self.age = age
    obj = Person(name="John", age=30)
    result = schema.serialize(obj)
    assert result == {"name": "John", "age": 30}

def test_serialize_missing_key_in_dict():
    from myapp import Schema, StringField, IntegerField
    fields = {"name": StringField(), "age": IntegerField()}
    schema = Schema(fields=fields)
    obj = {"name": "John"}
    result = schema.serialize(obj)
    assert result == {"name": "John"}

def test_serialize_missing_attribute_in_object():
    from myapp import Schema, StringField, IntegerField
    fields = {"name": StringField(), "age": IntegerField()}
    schema = Schema(fields=fields)
    class Person:
        def __init__(self, name):
            self.name = name
    obj = Person(name="John")
    result = schema.serialize(obj)
    assert result == {"name": "John"}

def test_serialize_with_nested_field_serialization():
    from myapp import Schema, StringField, IntegerField, ListField
    fields = {"name": StringField(), "scores": ListField(child=IntegerField())}
    schema = Schema(fields=fields)
    obj = {"name": "John", "scores": [85, 90, 78]}
    result = schema.serialize(obj)
    assert result == {"name": "John", "scores": [85, 90, 78]}

def test_serialize_with_read_only_field_in_dict():
    from myapp import Schema, StringField, IntegerField
    from myapp import Field
    class ReadOnlyField(Field):
        read_only = True
        def serialize(self, value):
            return value
    fields = {"name": StringField(), "id": ReadOnlyField()}
    schema = Schema(fields=fields)
    obj = {"name": "John", "id": 123}
    result = schema.serialize(obj)
    assert result == {"name": "John"}

def test_serialize_with_read_only_field_in_object():
    from myapp import Schema, StringField, IntegerField
    from myapp import Field
    class ReadOnlyField(Field):
        read_only = True
        def serialize(self, value):
            return value
    fields = {"name": StringField(), "id": ReadOnlyField()}
    schema = Schema(fields=fields)
    class Person:
        def __init__(self, name, id):
            self.name = name
            self.id = id
    obj = Person(name="John", id=123)
    result = schema.serialize(obj)
    assert result == {"name": "John"}

def test_serialize_empty_dict():
    from myapp import Schema, StringField
    fields = {"name": StringField()}
    schema = Schema(fields=fields)
    obj = {}
    result = schema.serialize(obj)
    assert result == {}

def test_serialize_empty_object():
    from myapp import Schema, StringField
    fields = {"name": StringField()}
    schema = Schema(fields=fields)
    class Empty:
        pass
    obj = Empty()
    result = schema.serialize(obj)
    assert result == {}


# LLM-generated content at query #6
#--------------------------

def test_serialize_with_non_mapping_object_and_missing_attribute():
    class MockField:
        def serialize(self, value):
            return value
    fields = {"key": MockField()}
    schema = Schema(fields)
    obj = object()
    result = schema.serialize(obj)
    assert result == {}


# LLM-generated content at query #7
#--------------------------

def test_serialize_with_non_mapping_object_without_attribute():
    class MockField:
        def serialize(self, value):
            return value
    fields = {"test_key": MockField()}
    schema = Schema(fields)
    obj = object()
    result = schema.serialize(obj)
    assert result == {}


# LLM-generated content at query #8
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
            return [MockMessage()]
    class MockMessage:
        pass
    fields = {"key": MockField()}
    schema = Schema(fields)
    value = {"key": "invalid"}
    try:
        schema.validate(value)
    except ValidationError as e:
        assert isinstance(e, ValidationError)


# LLM-generated content at query #9
#--------------------------

def test_serialize_returns_none_for_none_object():
    field = Field()
    schema = Schema(fields={"test": field})
    result = schema.serialize(None)
    assert result is None

def test_serialize_handles_dict_object():
    field = Field()
    schema = Schema(fields={"test": field})
    obj = {"test": "value"}
    result = schema.serialize(obj)
    assert result == {"test": None}

def test_serialize_handles_object_with_attributes():
    field = Field()
    schema = Schema(fields={"test": field})
    class MockObj:
        test = "value"
    obj = MockObj()
    result = schema.serialize(obj)
    assert result == {"test": None}

def test_serialize_ignores_missing_keys_in_dict():
    field = Field()
    schema = Schema(fields={"test": field})
    obj = {"other": "value"}
    result = schema.serialize(obj)
    assert result == {}

def test_serialize_ignores_missing_attributes_in_object():
    field = Field()
    schema = Schema(fields={"test": field})
    class MockObj:
        other = "value"
    obj = MockObj()
    result = schema.serialize(obj)
    assert result == {}

def test_serialize_calls_field_serialize_for_each_key():
    mock_field = Field()
    mock_field.serialize = lambda x: f"serialized_{x}"
    schema = Schema(fields={"test": mock_field})
    obj = {"test": "value"}
    result = schema.serialize(obj)
    assert result == {"test": "serialized_value"}

def test_serialize_handles_multiple_fields():
    field1 = Field()
    field2 = Field()
    schema = Schema(fields={"field1": field1, "field2": field2})
    obj = {"field1": "value1", "field2": "value2"}
    result = schema.serialize(obj)
    assert result == {"field1": None, "field2": None}

def test_serialize_with_mixed_present_and_missing_keys():
    field1 = Field()
    field2 = Field()
    schema = Schema(fields={"field1": field1, "field2": field2})
    obj = {"field1": "value1"}
    result = schema.serialize(obj)
    assert result == {"field1": None}


# LLM-generated content at query #10
#--------------------------

def test_setitem_raises_assertion_error_when_key_exists():
    definitions = Definitions()
    definitions["key1"] = "value1"
    try:
        definitions["key1"] = "value2"
        assert False
    except AssertionError as e:
        assert str(e) == "Definition for 'key1' has already been set."


# LLM-generated content at query #11
#--------------------------

def test_validate_allows_null_when_allow_null_is_true():
    field = Field(allow_null=True)
    schema = Schema(fields={}, allow_null=True)
    result = schema.validate(None)
    assert result is None

def test_validate_raises_error_on_null_when_allow_null_is_false():
    schema = Schema(fields={}, allow_null=False)
    try:
        schema.validate(None)
        assert False
    except ValidationError as e:
        assert any(msg.code == "null" for msg in e.messages)

def test_validate_raises_error_on_non_dict_value():
    schema = Schema(fields={})
    try:
        schema.validate("not a dict")
        assert False
    except ValidationError as e:
        assert any(msg.code == "type" for msg in e.messages)

def test_validate_raises_error_on_non_string_key():
    schema = Schema(fields={})
    try:
        schema.validate({1: "value"})
        assert False
    except ValidationError as e:
        assert any(msg.code == "invalid_key" for msg in e.messages)

def test_validate_raises_error_on_missing_required_field():
    field = Field(read_only=False)
    schema = Schema(fields={"required_field": field})
    try:
        schema.validate({})
        assert False
    except ValidationError as e:
        assert any(msg.code == "required" for msg in e.messages)

def test_validate_uses_default_value_for_missing_field_with_default():
    field = Field(read_only=False, default="default_value")
    schema = Schema(fields={"field_with_default": field})
    result = schema.validate({})
    assert result == {"field_with_default": "default_value"}

def test_validate_skips_read_only_fields():
    field = Field(read_only=True)
    schema = Schema(fields={"read_only_field": field})
    result = schema.validate({"read_only_field": "some_value"})
    assert result == {}

def test_validate_includes_validated_child_field_values():
    child_field = Field()
    schema = Schema(fields={"child": child_field})
    result = schema.validate({"child": "child_value"})
    assert result == {"child": "child_value"}

def test_validate_collects_child_field_validation_errors():
    child_field = Field()
    child_field.validate = lambda v: (_ for _ in ()).throw(ValidationError(messages=[Message(text="Child error", code="child_error", index=[])]))
    schema = Schema(fields={"child": child_field})
    try:
        schema.validate({"child": "bad_value"})
        assert False
    except ValidationError as e:
        assert any(msg.code == "child_error" for msg in e.messages)

def test_validate_returns_empty_dict_for_empty_input_and_no_fields():
    schema = Schema(fields={})
    result = schema.validate({})
    assert result == {}

def test_validate_handles_mapping_subclass():
    class CustomDict(dict):
        pass
    field = Field()
    schema = Schema(fields={"test": field})
    value = CustomDict({"test": "data"})
    result = schema.validate(value)
    assert result == {"test": "data"}


# LLM-generated content at query #12
#--------------------------

def test_validate_with_valid_child_schema():
    class MockField:
        read_only = False
        def has_default(self):
            return False
        def validate_or_error(self, value):
            return value, None
    field = MockField()
    schema = Schema(fields={"key": field})
    value = {"key": "valid"}
    result = schema.validate(value)
    assert result == {"key": "valid"}


# LLM-generated content at query #13
#--------------------------

def test_validate_allow_null():
    field = Field(allow_null=True)
    schema = Schema(fields={}, allow_null=True)
    result = schema.validate(None)
    assert result is None

def test_validate_null_not_allowed():
    schema = Schema(fields={}, allow_null=False)
    try:
        schema.validate(None)
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "null"

def test_validate_non_dict_type():
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
    child_field = Field(allow_null=False)
    schema = Schema(fields={"required_field": child_field})
    try:
        schema.validate({})
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "required"

def test_validate_read_only_field_ignored():
    child_field = Field(read_only=True)
    schema = Schema(fields={"read_only_field": child_field})
    result = schema.validate({"read_only_field": "value"})
    assert "read_only_field" not in result

def test_validate_field_with_default():
    child_field = Field(default="default_value")
    schema = Schema(fields={"field_with_default": child_field})
    result = schema.validate({})
    assert result["field_with_default"] == "default_value"

def test_validate_field_validation_error():
    child_field = Field(allow_null=False)
    schema = Schema(fields={"child": child_field})
    try:
        schema.validate({"child": None})
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "null"

def test_validate_success():
    child_field = Field()
    schema = Schema(fields={"child": child_field})
    result = schema.validate({"child": "value"})
    assert result == {"child": "value"}

def test_validate_multiple_errors():
    child_field = Field(allow_null=False)
    schema = Schema(fields={"required": child_field})
    try:
        schema.validate({1: "invalid", "required": None})
        assert False
    except ValidationError as e:
        assert len(e.messages) == 2
        codes = {msg.code for msg in e.messages}
        assert codes == {"invalid_key", "null"}


# LLM-generated content at query #14
#--------------------------

def test_validate_with_error_in_child_schema():
    from myapp import Schema, Field, ValidationError
    class MockField(Field):
        def __init__(self, read_only=False, has_default=False, default_value=None, error_on_validate=False):
            super().__init__()
            self.read_only = read_only
            self._has_default = has_default
            self.default_value = default_value
            self.error_on_validate = error_on_validate
        def has_default(self):
            return self._has_default
        def get_default_value(self):
            return self.default_value
        def validate_or_error(self, value):
            if self.error_on_validate:
                from myapp import ValidationError as VE
                error = VE(messages=[])
                return None, error
            return value, None
    child_field = MockField(error_on_validate=True)
    schema = Schema(fields={"key": child_field})
    value = {"key": "some_value"}
    try:
        schema.validate(value)
    except ValidationError:
        pass
    else:
        assert False, "Expected ValidationError"


# LLM-generated content at query #15
#--------------------------

def test_setitem_raises_assertion_error_when_key_exists():
    definitions = Definitions()
    definitions["key1"] = "value1"
    try:
        definitions["key1"] = "value2"
        raised = False
    except AssertionError:
        raised = True
    assert raised


# LLM-generated content at query #16
#--------------------------

def test_validate_with_valid_child_value():
    class MockField:
        read_only = False
        def has_default(self):
            return False
        def validate_or_error(self, item):
            return item, None
    fields = {"key": MockField()}
    schema = Schema(fields)
    value = {"key": "valid"}
    result = schema.validate(value)
    assert result == {"key": "valid"}


# LLM-generated content at query #17
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
    result = schema.validate({"read_only_key": "value"})
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
    field3 = Field(read_only=True)
    schema = Schema(fields={"key1": field1, "key2": field2, "key3": field3})
    result = schema.validate({"key1": "value1", "key3": "value3"})
    assert result == {"key1": "value1", "key2": "default2"}

def test_validate_aggregates_multiple_errors():
    required_field = Field()
    schema = Schema(fields={"required": required_field})
    try:
        schema.validate({"invalid_key": 123})
        assert False
    except ValidationError as e:
        assert len(e.messages) == 2
        codes = {msg.code for msg in e.messages}
        assert codes == {"invalid_key", "required"}


# LLM-generated content at query #18
#--------------------------

def test_setitem_raises_assertion_error_on_duplicate_key():
    definitions = Definitions()
    definitions['key1'] = 'value1'
    try:
        definitions['key1'] = 'value2'
        assert False, "Expected AssertionError was not raised"
    except AssertionError as e:
        assert str(e) == "Definition for 'key1' has already been set."


# LLM-generated content at query #19
#--------------------------

def test_setitem_adds_new_key():
    definitions = Definitions()
    definitions["key1"] = "value1"
    assert definitions["key1"] == "value1"

def test_setitem_raises_assertion_on_duplicate_key():
    definitions = Definitions()
    definitions["key1"] = "value1"
    try:
        definitions["key1"] = "value2"
        assert False, "Expected AssertionError"
    except AssertionError as e:
        assert "Definition for 'key1' has already been set." in str(e)

def test_setitem_works_with_existing_dict():
    initial_dict = {"a": 1, "b": 2}
    definitions = Definitions(initial_dict)
    definitions["c"] = 3
    assert definitions["c"] == 3

def test_setitem_preserves_other_keys():
    definitions = Definitions({"x": 10})
    definitions["y"] = 20
    assert definitions["x"] == 10
    assert definitions["y"] == 20

def test_setitem_after_delitem():
    definitions = Definitions({"temp": 5})
    del definitions["temp"]
    definitions["temp"] = 10
    assert definitions["temp"] == 10


# LLM-generated content at query #20
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
    schema = Schema(fields)
    value = {"key": "invalid"}
    try:
        schema.validate(value)
    except ValidationError as e:
        pass


# LLM-generated content at query #21
#--------------------------

def test_validate_with_no_error_from_child_schema():
    from myapp import Schema, Field, ValidationError, Message
    class MockField(Field):
        def __init__(self, read_only=False, has_default_val=False, default_val=None):
            super().__init__()
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
            return value, None
    child_field = MockField()
    schema = Schema(fields={"key": child_field})
    value = {"key": "valid"}
    result = schema.validate(value)
    assert result == {"key": "valid"}


# LLM-generated content at query #22
#--------------------------

def test_validate_with_no_error_from_child_schema():
    from myapp import Schema, Field, Message, ValidationError
    class MockField(Field):
        def __init__(self, read_only=False, has_default_val=False, default_val=None):
            super().__init__()
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
            return value, None
    child_field = MockField(read_only=False, has_default_val=False)
    schema = Schema(fields={"key": child_field})
    value = {"key": "valid_value"}
    result = schema.validate(value)
    assert result == {"key": "valid_value"}


# LLM-generated content at query #23
#--------------------------

def test_validate_with_error_in_child_schema():
    class MockField:
        read_only = False
        def has_default(self):
            return False
        def validate_or_error(self, value):
            return None, ValidationError(messages=[Message(text="error", code="error", index=[])])
    fields = {"field": MockField()}
    schema = Schema(fields=fields)
    value = {"field": "invalid"}
    try:
        schema.validate(value)
    except ValidationError as e:
        assert any(msg.code == "error" for msg in e.messages)


# LLM-generated content at query #24
#--------------------------

def test_setitem_raises_assertion_error_when_key_exists():
    definitions = Definitions()
    definitions["key1"] = "value1"
    try:
        definitions["key1"] = "value2"
        raised = False
    except AssertionError as e:
        raised = True
        assert str(e) == "Definition for 'key1' has already been set."
    assert raised


# LLM-generated content at query #25
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

def test_validate_with_read_only_field_ignored():
    child_field = Field(read_only=True)
    schema = Schema(fields={"read_only_field": child_field})
    result = schema.validate({})
    assert result == {}

def test_validate_with_field_having_default():
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
    child_field = Field()
    schema = Schema(fields={"child": child_field})
    try:
        schema.validate({"child": None})
        assert False
    except ValidationError as e:
        assert any(msg.index == ["child"] for msg in e.messages)

def test_validate_combines_multiple_errors():
    child_field = Field()
    schema = Schema(fields={"required": child_field})
    try:
        schema.validate({1: "value"})
        assert False
    except ValidationError as e:
        codes = [msg.code for msg in e.messages]
        assert "invalid_key" in codes
        assert "required" in codes


# LLM-generated content at query #26
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
    field = Field()
    schema = Schema(fields={"valid": field})
    try:
        schema.validate({123: "value"})
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

def test_validate_read_only_field_ignored():
    field = Field(read_only=True)
    schema = Schema(fields={"read_only": field})
    result = schema.validate({"read_only": "ignored"})
    assert result == {}

def test_validate_field_with_default():
    field = Field(default="default_value")
    schema = Schema(fields={"with_default": field})
    result = schema.validate({})
    assert result == {"with_default": "default_value"}

def test_validate_field_validation_error():
    field = Field()
    schema = Schema(fields={"test": field})
    try:
        schema.validate({"test": None})
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "null"

def test_validate_success():
    field = Field()
    schema = Schema(fields={"name": field})
    result = schema.validate({"name": "John"})
    assert result == {"name": "John"}

def test_validate_multiple_errors():
    field = Field()
    schema = Schema(fields={"required": field})
    try:
        schema.validate({123: "invalid"})
        assert False
    except ValidationError as e:
        assert len(e.messages) == 2
        codes = [msg.code for msg in e.messages]
        assert "invalid_key" in codes
        assert "required" in codes


# LLM-generated content at query #27
#--------------------------

def test_validate_child_schema_no_error():
    from myapp import Schema, Field, ValidationError, Message
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
            return value, None
    field1 = MockField()
    field2 = MockField()
    schema = Schema(fields={"field1": field1, "field2": field2})
    value = {"field1": "value1", "field2": "value2"}
    result = schema.validate(value)
    assert result == {"field1": "value1", "field2": "value2"}


# LLM-generated content at query #28
#--------------------------

def test_validate_with_child_schema_error():
    child_field = Field()
    child_field.read_only = False
    child_field.has_default = lambda: False
    child_field.validate_or_error = lambda x: (None, ValidationError(messages=[Message(text="error", code="error", index=[])]))
    schema = Schema(fields={"key": child_field})
    value = {"key": "value"}
    try:
        schema.validate(value)
    except ValidationError as e:
        pass


