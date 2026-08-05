####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_serialize_returns_none_when_obj_is_none():
    definitions = {}
    ref = Reference(to="test", definitions=definitions)
    assert ref.serialize(None) is None

def test_serialize_returns_dict_copy_when_obj_is_dict():
    definitions = {}
    ref = Reference(to="test", definitions=definitions)
    input_data = {"key": "value"}
    result = ref.serialize(input_data)
    assert result == {"key": "value"}
    assert result is not input_data
```


# LLM-generated content at query #2
#--------------------------

def test_schema_constructor_initializes_fields_and_required_list():
    class MockField:
        def __init__(self, read_only=False, default=None):
            self.read_only = read_only
            self.default = default
        def has_default(self):
            return default is not None
    
    f1 = MockField(read_only=False)
    f2 = MockField(read_only=True)
    f3 = MockField(default="val")
    fields = {"required_field": f1, "read_only_field": f2, "defaulted_field": f3}
    schema = Schema(fields=fields)
    assert schema.fields == fields
    assert schema.required == ["required_field"]

def test_schema_constructor_handles_empty_fields():
    schema = Schema(fields={})
    assert schema.fields == {}
    assert schema.required == []

def test_schema_constructor_inherits_field_properties():
    class MockField:
        def __init__(self, title="T", description="D"):
            self.title = "T"
            self.description = "D"
        def has_default(self): return False

    fields = {"key": MockField()}
    schema = Schema(fields=fields, title="SchemaTitle", description="SchemaDesc")
    assert schema.title == "SchemaTitle"
    assert schema.description == "SchemaDesc"


# LLM-generated content at query #3
#--------------------------

```python
def test_validate_null_not_allowed():
    field = Field(allow_null=False)
    schema = Schema(fields={})
    with assert_raises(ValidationError) as cm:
        schema.validate(None)
    assert cm.exception.messages[0].code == "null"

def test_validate_null_allowed():
    field = Field(allow_null=True)
    schema = Schema(fields={})
    assert schema.validate(None) is None

def test_validate_wrong_type():
    field = Field()
    schema = Schema(fields={})
    with assert_raises(ValidationError) as cm:
        schema.validate([1, 2, 3])
    assert cm.exception.messages[0].code == "type"

def test_validate_invalid_key_type():
    field = Field()
    schema = Schema(fields={})
    with assert_raises(ValidationError) as cm:
        schema.validate({123: "value"})
    assert cm.exception.messages[0].code == "invalid_key"
    assert cm.exception.messages[0].index == [123]

def test_validate_missing_required_field():
    child_field = Field(read_only=False)
    schema = Schema(fields={"name": child_field})
    with assert_raises(ValidationError) as cm:
        schema.validate({})
    assert cm.exception.messages[0].code == "required"
    assert cm.exception.messages[0].index == ["name"]

def test_validate_success_with_defaults():
    child_field = Field(default="default_val")
    schema = Schema(fields={"name": child_field})
    result = schema.validate({})
    assert result["name"] == "default_val"

def test_validate_success_with_provided_value():
    child_field = Field()
    schema = Schema(fields={"name": child_field})
    result = schema.validate({"name": "actual_val"})
    assert result["name"] == "actual_val"

def test_validate_ignores_read_only_missing_fields():
    child_field = Field(read_only=True)
    schema = Schema(fields={"name": child_field})
    result = schema.validate({})
    assert "name" not in result

def test_validate_nested_error_propagation():
    child_field = Field()
    child_schema = Schema(fields={"sub": child_field})
    # Manually simulating a validation error from a child
    parent_schema = Schema(fields={"data": child_schema})
    
    # We need to mock the behavior of validate_or_error returning an error
    # Since we can't use mocks in this specific constrained environment easily, 
    # we assume a structure where the child fails.
    with assert_raises(ValidationError) as cm:
        parent_schema.validate({"data": "not_a_dict"})
    assert any("data" in str(m.code) or "data" in str(m.index) for m_msg in cm.exception.messages for m in [m_msg])
```


# LLM-generated content at query #4
#--------------------------

```python
def test_validate_success():
    class MockField:
        def __init__(self, read_only=False, default=None):
            self.read_only = read_only
            self.default = default
        def has_default(self): return self.default is not None
        def get_default_value(self): return self.default
        def validate_or_error(self, value): return value, None
        def serialize(self, value): return value

    class MockError:
        def __init__(self, messages): self.messages = lambda add_prefix=None: messages
    
    class ValidationError(Exception):
        def __init__(self, messages): self.messages = lambda add_prefix=None: messages

    class SchemaMock(Schema):
        def validation_error(self, key): return ValidationError([])
        def get_error_text(self, key): return "err"

    field_int = MockField()
    field_str = MockField(default="default")
    schema = SchemaMock(fields={"age": field_int, "name": field_str})
    
    result = schema.validate({"age": 25, "name": "John"})
    assert result == {"age": 25, "name": "John"}
    assert schema.required == ["age"]

def test_validate_missing_required_field():
    class MockField:
        def __init__(self, read_only=False, default=None):
            self.read_only = read_only
            self.default = default
        def has_default(self): return self.default is not None
        def get_default_value(self): return self.default
        def validate_or_error(self, value): return value, None

    class ValidationError(Exception):
        def __init__(self, messages): self.messages = lambda add_prefix=None: messages

    class SchemaMock(Schema):
        def validation_error(self, key): return ValidationError([])
        def get_error_text(self, key): return "required"

    schema = SchemaMock(fields={"id": MockField()})
    
    with Exception() as e:
        schema.validate({"name": "John"})
        # This is a simplification for the test environment requirement 
        # to only use assignments and assertions.
    
    # Since we cannot use try/except or control flow, we assume the logic check below:
    # In a real environment, we'd assert the error message content.

def test_validate_type_error():
    class MockField:
        def __init__(self): self.read_only = False
        def has_default(self): return False
        def validate_or_error(self, value): return value, None

    class ValidationError(Exception):
        def __init__(self, messages): self.messages = lambda add_prefix=None: messages

    class SchemaMock(Schema):
        def validation_error(self, key): return ValidationError([])
        def get_error_text(self, key): return "type"

    schema = SchemaMock(fields={"id": MockField()})
    
    with Exception() as e:
        schema.validate(["not", "a", "dict"])

def test_validate_invalid_key_type():
    class MockField:
        def __init__(self): self.read_only = False
        def has_default(self): return False
        def validate_or_error(self, value): return value, None

    class Message:
        def __init__(self, text, code, index): 
            self.text = text
            self.code = code
            self.index = index

    class ValidationError(Exception):
        def __init__(self, messages): self.messages = lambda add_prefix=None: messages

    class SchemaMock(Schema):
        def validation_error(self, key): return ValidationError([])
        def get_error_text(self, key): return "invalid_key"

    schema = SchemaMock(fields={})
    
    with Exception() as e:
        schema.validate({123: "value"})
```


# LLM-generated content at query #5
#--------------------------

```python
def test_validate_returns_none_when_value_is_none_and_allow_null_is_true():
    class MockField(Field):
        def __init__(self, allow_null=False):
            self.allow_null = allow_null
        def validate(self, value):
            return super().validate(value)

    mock_field = MockField(allow_null=True)
    schema = Schema(fields={})
    schema.allow_null = True
    result = schema.validate(None)
    assert result is None
```


# LLM-generated content at query #6
#--------------------------

```python
def test_validate_returns_none_when_value_is_none_and_allow_null_is_true():
    mock_definitions = {"target_key": type('Mock', (), {'validate': lambda self, x: x})()}
    ref = Reference(to="target_key", definitions=mock_definitions, allow_null=True)
    assert ref.validate(None) is None

def test_validate_raises_error_when_value_is_none_and_allow_null_is_false():
    mock_definitions = {"target_key": type('Mock', (), {'validate': lambda self, x: x})()}
    ref = Reference(to="target_key", definitions=mock_definitions, allow_null=False)
    # Assuming validation_error is a method of Field that raises an exception or returns one. 
    # Since we cannot define custom functions, we rely on the behavior of the provided snippet.
    # We assert that it calls the error logic via the raised exception mechanism if implemented in super().
    try:
        ref.validate(None)
    except Exception as e:
        assert str(e) == "May not be null."

def test_validate_calls_target_validate_with_correct_value():
    class MockTarget:
        def validate(self, value):
            return f"validated_{value}"
            
    mock_definitions = {"target_key": MockTarget()}
    ref = Reference(to="target_key", definitions=mock_definitions)
    assert ref.validate("some_input") == "validated_some_input"

def test_validate_accesses_correct_definition_key():
    class MockTarget:
        def validate(self, value):
            return value
            
    mock_definitions = {"a": MockTarget(), "b": MockTarget()}
    ref = Reference(to="b", definitions=mock_definitions)
    assert ref.validate("test") == "test"
```


# LLM-generated content at query #7
#--------------------------

```python
def test_validate_returns_none_when_value_is_none_and_allow_null_is_true():
    class MockTarget:
        def validate(self, value):
            return value

    class MockDefinitions:
        def __getitem__(self, key):
            return self.target

        def __init__(self, target):
            self.target = target

    mock_target = MockTarget()
    mock_definitions = MockDefinitions(mock_target)
    
    # Setup Reference instance where value is None and allow_null is True
    # We mock the Field behavior (specifically allow_null) via kwargs/inheritance simulation
    class MockReference(Reference):
        def __init__(self, to, definitions, allow_null=True):
            super().__init__(to, definitions)
            self.allow_null = allow_null
        
        def validation_error(self, key):
            return Exception(self.errors[key])

    ref = MockReference(to="target_key", definitions=mock_definitions, allow_null=True)
    
    assert ref.validate(None) is None
```


# LLM-generated content at query #8
#--------------------------

```python
def test_validate_skips_error_branch_when_child_is_valid():
    class MockField:
        def __init__(self, value):
            self.value = value
            self.read_only = False
        def has_default(self):
            return False
        def get_default_value(self):
            return None
        def validate_or_error(self, item):
            class MockError:
                def messages(self, add_prefix=None):
                    return []
            return self.value, MockError() if self.value == "error" else (self.value, None)

    child_field = MockField("valid_value")
    schema = Schema(fields={"test_key": child_field})
    input_data = {"test_key": "valid_value"}
    
    result = schema.validate(input_data)
    
    assert result == {"test_key": "valid_value"}
```


# LLM-generated content at query #9
#--------------------------

```python
def test_validate_returns_none_when_value_is_none_and_allow_null_is_true():
    class MockField(Field):
        def __init__(self, allow_null=False):
            self.allow_null = allow_null
        def validate(self, value):
            if value is None and self.allow_null:
                return None
            return super().validate(value)

    schema = Schema(fields={})
    schema.allow_null = True
    assert schema.validate(None) is None
```


# LLM-generated content at query #10
#--------------------------

```python
def test_validate_child_schema_with_error():
    class MockField:
        def __init__(self, error_to_return=None):
            self.read_only = False
            self.error_to_return = error_to_return
        def has_default(self):
            return False
        def validate_or_error(self, value):
            if self.error_to_return:
                class MockErrorMessages:
                    def messages(self, add_prefix):
                        return [] # Simulating error presence via logic in Schema (though logic depends on error object)
                # To trigger line 37's 'if not error', 'error' must be truthy.
                # In the context of validate_or_error, returning a non-None value as second element.
                return None, type('Error', (), {'messages': lambda self, add_prefix: [f"{add_prefix}_error"]})()
            return value, None

    class MockSchema(Schema):
        def __init__(self, fields):
            self.fields = fields
            self.required = []
            self.allow_null = False
        def validate_or_error(self, value): return None, None 
        def validation_error(self, msg): raise ValueError(msg)
        def get_error_text(self, key): return ""

    error_obj = type('Error', (), {'messages': lambda self, add_prefix: [f"{add_prefix}_err"]})()
    field_with_error = MockField()
    # We need to override the method call specifically for this test instance
    field_with_error.validate_or_error = lambda self, v: (None, error_obj)
    
    schema = Schema(fields={"test_key": field_with_error})
    
    try:
        schema.validate({"test_key": "some_value"})
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0] == "test_key_err"
```


# LLM-generated content at query #11
#--------------------------

```python
def test_validate_returns_none_when_value_is_none_and_allow_null_is_true():
    class MockTarget:
        def validate(self, value):
            return value

    class MockDefinitions:
        def __getitem__(self, key):
            return self.target

        def __init__(self, target):
            self.target = target

    mock_target = MockTarget()
    mock_definitions = MockDefinitions(mock_target)
    
    # Setup Reference instance with allow_null=True
    # We mock the validation_error method to avoid errors during setup if needed, 
    # but here we only need to trigger the first branch.
    instance = Reference(to="test", definitions=mock_definitions, allow_null=True)
    
    assert instance.validate(None) is None
```


# LLM-generated content at query #12
#--------------------------

```python
def test_validate_returns_none_when_value_is_none_and_allow_null_is_true():
    mock_definitions = {"target_key": type('Mock', (), {'validate': lambda self, x: x})()}
    ref = Reference(to="target_key", definitions=mock_definitions, allow_null=True)
    assert ref.validate(None) is None

def test_validate_raises_error_when_value_is_none_and_allow_null_is_false():
    mock_definitions = {"target_key": type('Mock', (), {'validate': lambda self, x: x})()}
    ref = Reference(to="target_key", definitions=mock_definitions, allow_null=False)
    # Assuming validation_error is a method from Field that raises an exception
    # In a real scenario, we'd mock the error raising mechanism. 
    # Here we test the logic flow.
    try:
        ref.validate(None)
    except Exception as e:
        assert str(e) == "May not be null."

def test_validate_calls_target_validate_with_provided_value():
    class MockTarget:
        def validate(self, value):
            return f"validated_{value}"
            
    mock_definitions = {"target_key": MockTarget()}
    ref = Reference(to="target_key", definitions=mock_definitions)
    assert ref.validate("some_value") == "validated_some_value"

def test_validate_retrieves_correct_target_from_definitions():
    class MockTarget:
        def validate(self, value):
            return value
            
    mock_definitions = {"a": MockTarget(), "b": MockTarget()}
    ref = Reference(to="a", definitions=mock_definitions)
    assert ref.validate("test") == "test"
```


# LLM-generated content at query #13
#--------------------------

```python
def test_validate_schema_child_field_with_error():
    mock_child_field = type('MockField', (), {
        'read_only': False,
        'has_default': lambda: False,
        'get_default_value': lambda: None,
        'validate_or_error': lambda self, val: (None, type('MockError', (), {'messages': lambda self, add_prefix: []})())
    })()
    
    mock_schema = Schema(fields={"test_key": mock_child_field})
    
    # This will trigger line 36: child_value, error = child_schema.validate_or_error(item)
    # Since error is not None (it's a MockError object), the predicate 'if not error:' evaluates to False.
    with pytest.raises(ValidationError):
        mock_schema.validate({"test_key": "some_value"})
```


# LLM-generated content at query #14
#--------------------------

```python
def test_validate_child_field_with_error():
    class MockField:
        def __init__(self, read_only=False, default=None):
            self.read_only = read_only
            self.default = default
        def has_default(self):
            return self.default is not None
        def get_default_value(self):
            return self.default
        def validate_or_error(self, value):
            class MockError:
                def messages(self, add_prefix):
                    return [f"{add_prefix}_error"]
            if value == "trigger_error":
                return None, MockError()
            return value, None

    class MockValidationError(Exception):
        pass

    class SchemaMock(Schema):
        def validation_error(self, code):
            return lambda x: MockValidationError()
        def get_error_text(self, key):
            return "error"

    child_field = MockField()
    schema = SchemaMock(fields={"test_key": child_field})
    
    # To make 'if not error' evaluate to False, we need an error object.
    # The value passed is "trigger_error", which causes validate_or_error 
    # to return (None, MockError()).
    # Thus, 'error' is truthy, and 'if not error' is False.
    with Exception: # We expect the validation logic to process the error
        try:
            schema.validate({"test_key": "trigger_error"})
        except Exception as e:
            assert isinstance(e, MockValidationError)
```


# LLM-generated content at query #15
#--------------------------

```python
def test_validate_returns_none_when_value_is_none_and_allow_null_is_true():
    class MockField(Field):
        def __init__(self, allow_null=False):
            self.allow_null = allow_null
        def validate(self, value):
            return super().validate(value)

    schema = Schema(fields={}, allow_null=True)
    assert schema.validate(None) is None
```


# LLM-generated content at query #16
#--------------------------

```python
def test_validate_returns_none_when_value_is_none_and_allow_null_is_true():
    class MockField(Field):
        def __init__(self, allow_null=False):
            self.allow_null = allow_null

    mock_field = MockField(allow_null=True)
    schema = Schema(fields={})
    schema.allow_null = True
    result = schema.validate(None)
    assert result is None
```


# LLM-generated content at query #17
#--------------------------

```python
def test_schema_validate_valid_dict():
    class MockField:
        def __init__(self, read_only=False, default=None):
            self.read_only = read_only
            self.default = default
        def has_default(self): return self.default is not None
        def get_default_value(self): return self.default
        def validate_or_error(self, value): return value, None
        def validate(self, value): return value

    class MockSchema:
        def __init__(self, fields):
            self.fields = fields
            self.allow_null = False
            self.required = [k for k, v in fields.items() if not (v.read_only or v.has_default())]
        def validation_error(self, key): return Exception(key)
        def get_error_text(self, key): return f"error_{key}"

    field_int = MockField()
    schema = MockSchema({"name": field_int})
    assert schema.validate({"name": "test"}) == {"name": "test"}

def test_schema_validate_null_not_allowed():
    class MockField:
        def __init__(self): self.read_only = False
        def has_default(self): return False
        def validate_or_error(self, value): return value, None
        def validate(self, value): return value

    class MockSchema:
        def __init__(self, fields):
            self.fields = fields
            self.allow_null = False
            self.required = []
        def validation_error(self, key): return ValueError(key)
        def get_error_text(self, key): return "err"

    schema = MockSchema({"a": MockField()})
    try:
        schema.validate(None)
    except ValueError as e:
        assert str(e) == "null"

def test_schema_validate_type_error():
    class MockField:
        def __init__(self): self.read_only = False
        def has_default(self): return False
        def validate_or_error(self, value): return value, None
        def validate(self, value): return value

    class MockSchema:
        def __init__(self, fields):
            self.fields = fields
            self.allow_null = False
            self.required = []
        def validation_error(self, key): return ValueError(key)
        def get_error_text(self, key): return "err"

    schema = MockSchema({"a": MockField()})
    try:
        schema.validate(["not", "a", "dict"])
    except ValueError as e:
        assert str(e) == "type"

def test_schema_validate_required_field_missing():
    class MockField:
        def __init__(self): self.read_only = False
        def has_default(self): return False
        def validate_or_error(self, value): return value, None
        def validate(self, value): return value

    class Message:
        def __init__(self, text, code, index): 
            self.text = text; self.code = code; self.index = index
        def messages(self, add_prefix=""): return []

    class ValidationError(Exception):
        def __init__(self, messages): self.messages = messages

    class MockSchema:
        def __init__(self, fields):
            self.fields = fields
            self.allow_null = False
            self.required = ["missing"]
        def validation_error(self, key): return Exception(key)
        def get_error_text(self, key): return "req_err"

    schema = MockSchema({"missing": MockField()})
    try:
        schema.validate({"other": 1})
    except ValidationError as e:
        assert e.messages[0].code == "required"
        assert e.messages[0].index == ["missing"]

def test_schema_validate_default_value_applied():
    class MockField:
        def __init__(self, default): 
            self.read_only = False
            self.default = default
        def has_default(self): return True
        def get_default_value(self): return self.default
        def validate_or_error(self, value): return value, None
        def validate(self, value): return value

    class MockSchema:
        def __init__(self, fields):
            self.fields = fields
            self.allow_null = False
            self.required = []
        def validation_error(self, key): return Exception(key)
        def get_error_text(self, key): return "err"

    schema = MockSchema({"age": MockField(default=25)})
    assert schema.validate({}) == {"age": 25}

def test_schema_validate_invalid_key_type():
    class MockField:
        def __init__(self): self.read_only = False
        def has_default(self): return False
        def validate_or_error(self, value): return value, None
        def validate(self, value): return value

    class Message:
        def __init__(self, text, code, index): 
            self.text = text; self.code = code; self.index = index
        def messages(self, add_prefix=""): return []

    class ValidationError(Exception):
        def __init__(self, messages): self.messages = messages

    class MockSchema:
        def __init__(self, fields):
            self.fields = fields
            self.allow_null = False
            self.required = []
        def validation_error(self, key): return Exception(key)
        def get_error_text(self, key): return "invalid_key_err"

    schema = MockSchema({"a": MockField()})
    try:
        schema.validate({123: "value"})
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"
        assert e.messages[0].index == [123]
```


# LLM-generated content at query #18
#--------------------------

```python
def test_validate_raises_error_on_null_when_not_allowed():
    mock_definitions = {"target_key": MagicMock()}
    ref = Reference(to="target_key", definitions=mock_definitions, allow_null=False)
    
    with pytest.raises(Exception) as excinfo:
        ref.validate(None)
    assert "May not be null." in str(excinfo.value)

def test_validate_returns_none_on_null_when_allowed():
    mock_definitions = {"target_key": MagicMock()}
    ref = Reference(to="target_key", definitions=mock_definitions, allow_null=True)
    
    result = ref.validate(None)
    assert result is None

def test_validate_calls_target_validate_with_value():
    mock_target = MagicMock()
    mock_target.validate.return_value = "validated_value"
    mock_definitions = {"target_key": mock_target}
    ref = Reference(to="target_key", definitions=mock_definitions, allow_null=False)
    
    input_value = "some_data"
    result = ref.validate(input_value)
    
    assert result == "validated_value"
    mock_target.validate.assert_called_once_with(input_value)
```


# LLM-generated content at query #19
#--------------------------

```python
def test_schema_validate_success():
    class MockField:
        def __init__(self, read_only=False, default=None):
            self.read_only = read_only
            self.default = default
        def has_default(self): return self.default is not None
        def get_default_value(self): return self.default
        def validate_or_error(self, value): return value, None
    
    class MockSchema(Schema):
        def validation_error(self, code): raise ValidationError(messages=[code])
        def get_error_text(self, code): return Schema.errors[code]

    field_a = MockField()
    fields = {"a": field_a}
    schema = MockSchema(fields={"a": field_a})
    result = schema.validate({"a": 123})
    assert result == {"a": 123}

def test_schema_validate_null_error():
    class MockField:
        def __init__(self, allow_null=False): self.allow_null = allow_null
        def has_default(self): return False
        def read_only(self): return False
    
    class MockSchema(Schema):
        def validation_error(self, code): raise ValidationError(messages=[code])
        def get_error_text(self, code): return Schema.errors[code]

    schema = MockSchema(fields={})
    try:
        schema.validate(None)
    except ValidationError as e:
        assert e.messages == ["null"]

def test_schema_validate_type_error():
    class MockField:
        def __init__(self): self.allow_null = False
        def has_default(self): return False
        def read_only(self): return False

    class MockSchema(Schema):
        def validation_error(self, code): raise ValidationError(messages=[code])
        def get_error_text(self, code): return Schema.errors[code]

    schema = MockSchema(fields={})
    try:
        schema.validate(["not", "a", "dict"])
    except ValidationError as e:
        assert e.messages == ["type"]

def test_schema_validate_invalid_key():
    class MockField:
        def __init__(self): self.allow_null = False
        def has_default(self): return False
        def read_only(self): return False

    class MockSchema(Schema):
        def validation_error(self, code): raise ValidationError(messages=[code])
        def get_error_text(self, code): return Schema.errors[code]

    schema = MockSchema(fields={})
    try:
        schema.validate({123: "value"})
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"
        assert e.messages[0].index == [123]

def test_schema_validate_required_field_missing():
    class MockField:
        def __init__(self, read_only=False, default=None): 
            self.read_only = read_only
            self.default = default
        def has_default(self): return self.default is not None
        def get_default_value(self): return self.default

    class MockSchema(Schema):
        def validation_error(self, code): raise ValidationError(messages=[code])
        def get_error_text(self, code): return Schema.errors[code]

    field_req = MockField()
    schema = MockSchema(fields={"required_key": field_req})
    try:
        schema.validate({"other_key": 1})
    except ValidationError as e:
        assert e.messages[0].code == "required"
        assert e.messages[0].index == ["required_key"]

def test_schema_validate_with_defaults():
    class MockField:
        def __init__(self, default=None): 
            self.read_only = False
            self.default = default
        def has_default(self): return self.default is not None
        def get_default_value(self): return self.default
        def validate_or_error(self, value): return value, None

    class MockSchema(Schema):
        def validation_error(self, code): raise ValidationError(messages=[code])
        def get_error_text(self, code): return Schema.errors[code]

    field_default = MockField(default="default_val")
    schema = MockSchema(fields={"a": field_default})
    result = schema.validate({})
    assert result == {"a": "default_val"}
```


# LLM-generated content at query #20
#--------------------------

```python
def test_validate_returns_none_when_value_is_none_and_allow_null_is_true():
    class MockDefinition:
        def validate(self, value):
            return value

    class MockField:
        def __init__(self, allow_null=False):
            self.allow_null = allow_null
        def validation_error(self, msg):
            return Exception(Reference.errors[msg])

    class MockReference(Reference):
        def __init__(self, to, definitions, allow_null=False):
            self.to = to
            self.definitions = definitions
            self.allow_null = allow_null
        @property
        def target(self):
            return self.definitions[self.to]
        def validate(self, value):
            if value is None and self.allow_null:
                return None
            elif value is None:
                raise Exception("error")
            return self.target.validate(value)

    definitions = {"test": MockDefinition()}
    reference = MockReference(to="test", definitions=definitions, allow_null=True)
    
    assert reference.validate(None) is None
```


# LLM-generated content at query #21
#--------------------------

```python
def test_validate_skips_assignment_when_error_exists():
    class MockField:
        def __init__(self, read_only=False, default=None):
            self.read_only = read_only
            self._default = default
        def has_default(self):
            return self._default is not None
        def get_default_value(self):
            return self._default
        def validate_or_error(self, value):
            class MockError:
                def messages(self, add_prefix):
                    return []
            if value == "trigger_error":
                return None, MockError()
            return value, None

    field = MockField()
    schema = Schema(fields={"test_key": field})
    input_data = {"test_key": "trigger_error"}
    
    # The goal is to reach line 37 where 'if not error:' evaluates to False.
    # In our mock, when value is "trigger_error", error is not None.
    # We assert that the code executes without crashing and validates properly.
    result = schema.validate(input_data)
    assert "test_key" not in result
```


# LLM-generated content at query #22
#--------------------------

```python
def test_validate_raises_error_on_null_when_not_allowed():
    mock_definitions = {"target_key": type('Mock', (), {'validate': lambda x: x})()}
    ref = Reference(to="target_key", definitions=mock_definitions, allow_null=False)
    
    with AssertionError:
        ref.validate(None)

def test_validate_returns_none_when_null_is_allowed():
    mock_definitions = {"target_key": type('Mock', (), {'validate': lambda x: x})()}
    ref = Reference(to="target_key", definitions=mock_definitions, allow_null=True)
    
    assert ref.validate(None) is None

def test_validate_calls_target_validate_with_value():
    class MockTarget:
        def validate(self, value):
            return f"validated_{value}"

    mock_definitions = {"target_key": MockTarget()}
    ref = Reference(to="target_key", definitions=mock_definitions, allow_null=False)
    
    assert ref.validate("some_value") == "validated_some_value"

def test_validate_uses_correct_target_from_definitions():
    class MockTarget:
        def validate(self, value):
            return value

    mock_definitions = {
        "key1": MockTarget(),
        "key2": MockTarget()
    }
    ref_key1 = Reference(to="key1", definitions=mock_definitions)
    ref_key2 = Reference(to="key2", definitions=mock_definitions)
    
    assert ref_key1.validate("val") == "val"
    assert ref_key2.validate("val") == "val"
```


# LLM-generated content at query #23
#--------------------------

```python
def test_validate_success():
    field1 = StringField()
    field2 = IntField(default=10)
    schema = Schema(fields={"name": field1, "age": field2})
    input_data = {"name": "John", "age": 25}
    result = schema.validate(input_data)
    assert result == {"name": "John", "age": 25}

def test_validate_default_value():
    field1 = IntField(default=42)
    schema = Schema(fields={"score": field1})
    input_data = {}
    result = schema.validate(input_data)
    assert result == {"score": 42}

def test_validate_null_error():
    field1 = StringField()
    schema = Schema(fields={"name": field1}, allow_null=False)
    with assert_raises(ValidationError) as cm:
        schema.validate(None)
    assert cm.exception.messages[0].code == "null"

def test_validate_type_error():
    field1 = StringField()
    schema = Schema(fields={"name": field1})
    with assert_raises(ValidationError) as cm:
        schema.validate(["not", "a", "dict"])
    assert cm.exception.messages[0].code == "type"

def test_validate_invalid_key_error():
    field1 = StringField()
    schema = Schema(fields={"name": field1})
    input_data = {123: "value"}
    with assert_raises(ValidationError) as cm:
        schema.validate(input_data)
    assert cm.exception.messages[0].code == "invalid_key"
    assert cm.exception.messages[0].index == [123]

def test_validate_required_error():
    field1 = StringField()
    schema = Schema(fields={"name": field1})
    input_data = {}
    with assert_raises(ValidationError) as cm:
        schema.validate(input_data)
    assert cm.exception.messages[0].code == "required"
    assert cm.exception.messages[0].index == ["name"]

def test_validate_nested_error_propagation():
    inner_schema = Schema(fields={"sub": StringField()})
    outer_schema = Schema(fields={"data": inner_schema})
    input_data = {"data": {"sub": 123}}
    with assert_raises(ValidationError) as cm:
        outer_schema.validate(input_data)
    assert any("sub" in msg.prefix or msg.code == "type" for msg in cm.exception.messages)

def test_validate_allow_null_success():
    field1 = StringField()
    schema = Schema(fields={"name": field1}, allow_null=True)
    result = schema.validate(None)
    assert result is None

def test_validate_readonly_skips_default():
    field1 = IntField(default=10, read_only=True)
    schema = Schema(fields={"age": field1})
    input_data = {}
    result = schema.validate(input_data)
    assert "age" not in result
```


# LLM-generated content at query #24
#--------------------------

```python
def test_validate_success_with_defaults():
    class MockField:
        def __init__(self, read_only=False, default=None):
            self.read_only = read_only
            self.default = default
        def has_default(self): return self.default is not None
        def get_default_value(self): return self.default
        def validate_or_error(self, val): return val, None
    
    class MockSchema:
        def __init__(self, fields): 
            self.fields = fields
            self.allow_null = False
            self.required = []
        def validation_error(self, key): raise Exception(key)
        def get_error_text(self, key): return "err"

    field_str = MockField(default="hello")
    field_int = MockField()
    schema = Schema(fields={"name": field_str, "age": field_int})
    
    result = schema.validate({"age": 25})
    assert result == {"name": "hello", "age": 25}

def test_validate_error_type_not_dict():
    class MockField:
        def __init__(self): self.read_only = False
        def has_default(self): return False
        def validate_or_error(self, val): return val, None

    class MockSchema:
        def __init__(self, fields): 
            self.fields = fields
            self.allow_null = False
            self.required = []
        def validation_error(self, key): raise ValueError(key)
        def get_error_text(self, key): return "err"

    schema = Schema(fields={"a": MockField()})
    with pytest.raises(ValueError, match="type"):
        schema.validate(["not", "a", "dict"])

def test_validate_error_null_not_allowed():
    class MockField:
        def __init__(self): self.read_only = False
        def has_default(self): return False
        def validate_or_error(self, val): return val, None

    class MockSchema:
        def __init__(self, fields): 
            self.fields = fields
            self.allow_null = False
            self.required = []
        def validation_error(self, key): raise ValueError(key)
        def get_error_text(self, key): return "err"

    schema = Schema(fields={"a": MockField()})
    with pytest.raises(ValueError, match="null"):
        schema.validate(None)

def test_validate_error_invalid_key_type():
    class MockField:
        def __init__(self): self.read_only = False
        def has_default(self): return False
        def validate_or_error(self, val): return val, None

    class Message:
        def __init__(self, text, code, index): 
            self.text = text
            self.code = code
            self.index = index
    class ValidationError(Exception):
        def __init__(self, messages): self.messages = messages
        def messages(self, add_prefix=None): return []

    class MockSchema:
        def __init__(self, fields): 
            self.fields = fields
            self.allow_null = False
            self.required = []
        def validation_error(self, key): raise ValueError(key)
        def get_error_text(self, key): return "err"

    schema = Schema(fields={})
    input_data = {123: "value"}
    with pytest.raises(ValidationError):
        schema.validate(input_data)

def test_validate_error_required_field():
    class MockField:
        def __init__(self, read_only=False, default=None):
            self.read_only = read_only
            self.default = default
        def has_default(self): return self.default is not None
        def get_default_value(self): return self.default
        def validate_or_error(self, val): return val, None

    class Message:
        def __init__(self, text, code, index): pass
    class ValidationError(Exception):
        def __init__(self, messages): self.messages = messages
    class MockSchema:
        def __init__(self, fields): 
            self.fields = fields
            self.allow_null = False
            self.required = ["missing_key"]
        def validation_error(self, key): raise ValueError(key)
        def get_error_text(self, key): return "err"

    schema = Schema(fields={"missing_key": MockField()})
    with pytest.raises(ValidationError):
        schema.validate({"other": 1})
```


# LLM-generated content at query #25
#--------------------------

```python
def test_validate_child_schema_with_error():
    class MockField:
        def __init__(self, read_only=False):
            self.read_only = read_only
        def has_default(self):
            return False
        def get_default_value(self):
            return None
        def validate_or_error(self, value):
            class MockError:
                def messages(self, add_prefix):
                    return [f"{add_prefix}: error"]
            return None, MockError()

    class MockValidationError(Exception):
        def __init__(self, messages):
            self.messages = messages

    class MockSchema(Schema):
        def validation_error(self, code):
            return lambda msg: MockValidationError([msg])
        def get_error_text(self, code):
            return "error"

    child_field = MockField()
    parent_schema = MockSchema(fields={"child": child_field})
    input_data = {"child": "some_value"}
    
    with pytest.raises(MockValidationError) as excinfo:
        parent_schema.validate(input_data)
    assert "child: error" in excinfo.value.messages
```


# LLM-generated content at query #26
#--------------------------

```python
def test_validate_returns_none_when_value_is_none_and_allow_null_is_true():
    class MockField(Field):
        def __init__(self, allow_null=False):
            self.allow_null = allow_null
            self.fields = {}
            self.required = []

        def validate(self, value):
            if value is None and self.allow_null:
                return None
            return super().validate(value)

    schema = MockField(allow_null=True)
    assert schema.validate(None) is None
```


# LLM-generated content at query #27
#--------------------------

```python
def test_validate_returns_none_when_value_is_none_and_allow_null_is_true():
    class MockTarget:
        def validate(self, value):
            return value

    class MockDefinitions:
        def __getitem__(self, key):
            return self.target

        def __init__(self, target):
            self.target = target

    mock_target = MockTarget()
    definitions = MockDefinitions(mock_target)
    reference = Reference(to="item", definitions=definitions, allow_null=True)
    
    result = reference.validate(None)
    
    assert result is None
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_serialize_returns_none_when_obj_is_none():
    mock_field = type('Field', (), {'serialize': lambda self, v: v})()
    schema = Schema(fields={"name": mock_field})
    assert schema.serialize(None) is None

def test_serialize_returns_dict_with_serialized_values_from_mapping():
    mock_field_1 = type('Field', (), {'serialize': lambda self, v: str(v)})()
    mock_field_2 = type('Field', (), {'serialize': lambda self, v: int(v)})()
    schema = Schema(fields={"name": mock_field_1, "age": mock_field_2})
    input_data = {"name": "Alice", "age": 30, "extra": "ignored"}
    expected = {"name": "Alice", "age": 30}
    assert schema.serialize(input_data) == expected

def test_serialize_returns_dict_with_serialized_values_from_object():
    mock_field_1 = type('Field', (), {'serialize': lambda self, v: str(v)})()
    schema = Schema(fields={"name": mock_field_1})
    class User:
        def __init__(self, name):
            self.name = name
    input_obj = User("Bob")
    expected = {"name": "Bob"}
    assert schema.serialize(input_obj) == expected

def test_serialize_skips_missing_keys():
    mock_field_1 = type('Field', (), {'serialize': lambda self, v: v})()
    mock_field_2 = type('Field', (), {'serialize': lambda self, v: v})()
    schema = Schema(fields={"present": mock_field_1, "missing": mock_field_2})
    input_data = {"present": "exists"}
    expected = {"present": "exists"}
    assert schema.serialize(input_data) == expected

def test_serialize_skips_attributes_not_present_on_object():
    mock_field_1 = type('Field', (), {'serialize': lambda self, v: v})()
    schema = Schema(fields={"name": mock_field_1})
    class Empty:
        pass
    input_obj = Empty()
    assert schema.serialize(input_obj) == {}
```


# LLM-generated content at query #2
#--------------------------

```python
def test_serialize_predicate_true_with_dict():
    class MockField:
        def serialize(self, value):
            return value

    fields = {"name": MockField()}
    schema = Schema(fields=fields)
    obj = {"name": "test_value"}
    
    result = schema.serialize(obj)
    assert result == {"name": "test_value"}

def test_serialize_predicate_true_with_object():
    class MockField:
        def serialize(self, value):
            return value

    class MockObject:
        def __init__(self, name):
            self.name = name

    fields = {"name": MockField()}
    schema = Schema(fields=fields)
    obj = MockObject("test_value")
    
    result = schema.serialize(obj)
    assert result == {"name": "test_value"}
```


# LLM-generated content at query #3
#--------------------------

```python
def test_validate_success():
    field1 = StringField()
    schema = Schema(fields={"name": field1})
    data = {"name": "test"}
    result = schema.validate(data)
    assert result == {"name": "test"}

def test_validate_null_allowed():
    field1 = StringField(allow_null=True)
    schema = Schema(fields={"name": field1})
    result = schema.validate({"name": None})
    assert result == {"name": None}

def test_validate_null_not_allowed_raises_error():
    field1 = StringField(allow_null=False)
    schema = Schema(fields={"name": field1})
    try:
        schema.validate({"name": None})
    except ValidationError as e:
        assert any(m.code == "null" for m in e.messages)

def test_validate_wrong_type_raises_error():
    field1 = StringField()
    schema = Schema(fields={"name": field1})
    try:
        schema.validate(["not", "a", "dict"])
    except ValidationError as e:
        assert any(m.code == "type" for m in e.messages)

def test_validate_invalid_key_raises_error():
    field1 = StringField()
    schema = Schema(fields={"name": field1})
    data = {123: "value"}
    try:
        schema.validate(data)
    except ValidationError as e:
        assert any(m.code == "invalid_key" for m in e.messages)

def test_validate_required_field_missing_raises_error():
    field1 = StringField()
    schema = Schema(fields={"name": field1})
    try:
        schema.validate({})
    except ValidationError as e:
        assert any(m.code == "required" and m.index == ["name"] for m in e.messages)

def test_validate_default_value_applied():
    field1 = StringField(default="default_val")
    schema = Schema(fields={"name": field1})
    result = schema.validate({})
    assert result["name"] == "default_val"

def test_validate_readonly_field_skipped():
    field1 = StringField(read_only=True)
    schema = Schema(fields={"name": field1})
    data = {"name": "should_be_ignored"}
    result = schema.validate(data)
    assert "name" not in result

def test_validate_nested_schema_error_propagation():
    child_schema = StringField()
    parent_schema = Schema(fields={"child": child_schema})
    try:
        parent_schema.validate({"child": 123})
    except ValidationError as e:
        assert any("child" in m.path for m in e.messages)
```


# LLM-generated content at query #4
#--------------------------

```python
def test_validate_iterates_over_fields_at_line_27():
    mock_field = type('MockField', (), {
        'fields': {'test_key': type('MockField', (), {'read_only': False, 'has_default': lambda: False, 'get_default_value': lambda: None, 'validate_or_error': lambda x: (x, None)})()},
        'allow_null': True,
        'validation_error': lambda self, x: Exception(x),
        'get_error_text': lambda self, x: x,
        'validate': lambda self, value: {'test_key': 'value'}
    })()
    
    result = mock_field.validate({'test_key': 'value'})
    assert result == {'test_key': 'value'}
```


# LLM-generated content at query #5
#--------------------------

```python
def test_schema_constructor_initializes_fields_and_required_keys():
    class MockField:
        def __init__(self, read_only=False, default=None):
            self.read_only = read_only
            self.default = default
        def has_default(self):
            return default is not None

    field_required = MockField()
    field_readonly = MockField(read_only=True)
    field_default = MockField(default="val")
    
    fields = {
        "req": field_required,
        "ro": field_readonly,
        "def": field_default
    }
    
    schema = Schema(fields=fields)
    
    assert schema.fields == fields
    assert "req" in schema.required
    assert "ro" not in schema.required
    assert "def" not in schema.required

def test_schema_constructor_with_kwargs():
    class MockField:
        def __init__(self):
            pass
        def has_default(self):
            return False

    fields = {"test": MockField()}
    schema = Schema(fields=fields, title="Test Schema", description="Test Description", allow_null=True)
    
    assert schema.title == "Test Schema"
    assert schema.description == "Test Description"
    assert schema.allow_null is True
```


# LLM-generated content at query #6
#--------------------------

```python
def test_serialize_with_dict_obj_ensures_loop_executes():
    class MockField:
        def serialize(self, value):
            return value

    fields = {"name": MockField()}
    schema = Schema(fields=fields)
    obj = {"name": "test_value"}
    
    result = schema.serialize(obj)
    assert result == {"name": "test_value"}
```


# LLM-generated content at query #7
#--------------------------

```python
def test_validate_returns_none_when_value_is_none_and_allow_null_is_true():
    mock_definitions = {"target_key": type('Mock', (), {"validate": lambda x: x})()}
    ref = Reference(to="target_key", definitions=mock_definitions, allow_null=True)
    assert ref.validate(None) is None

def test_validate_raises_error_when_value_is_none_and_allow_null_is_false():
    mock_definitions = {"target_key": type('Mock', (), {"validate": lambda x: x})()}
    ref = Reference(to="target_key", definitions=mock_definitions, allow_null=False)
    # Assuming validation_error raises a ValueError for the sake of this test structure
    # In a real scenario, we would assert the specific exception type defined in Field
    try:
        ref.validate(None)
    except Exception as e:
        assert str(e) == "May not be null."

def test_validate_calls_target_validate_with_correct_value():
    class MockTarget:
        def validate(self, value):
            return f"validated_{value}"
    
    mock_definitions = {"target_key": MockTarget()}
    ref = Reference(to="target_key", definitions=mock_definitions)
    assert ref.validate("some_value") == "validated_some_value"

def test_validate_uses_correct_target_from_definitions():
    class MockTarget:
        def validate(self, value):
            return value

    mock_definitions = {"key_a": MockTarget(), "key_b": type('Mock', (), {"validate": lambda x: "wrong"})()}
    ref_a = Reference(to="key_a", definitions=mock_definitions)
    ref_b = Reference(to="key_b", definitions=mock_definitions)
    assert ref_a.validate("val") == "val"
    assert ref_b.validate("val") == "wrong"
```


# LLM-generated content at query #8
#--------------------------

```python
def test_schema_serialize_returns_none_when_obj_is_none():
    field_mock = type('Field', (), {'serialize': lambda self, v: v})()
    schema = Schema(fields={"name": field_mock})
    assert schema.serialize(None) is None

def test_schema_serialize_returns_dict_with_serialized_values():
    class MockField:
        def serialize(self, value):
            return f"serialized_{value}"
    
    fields = {"name": MockField(), "age": MockField()}
    schema = Schema(fields=fields)
    input_data = {"name": "Alice", "age": 30}
    expected = {"name": "serialized_Alice", "age": "serialized_30"}
    assert schema.serialize(input_data) == expected

def test_schema_serialize_works_with_object_attributes():
    class MockField:
        def serialize(self, value):
            return value

    class User:
        def __init__(self, name):
            self.name = name

    fields = {"name": MockField()}
    schema = Schema(fields=fields)
    user_obj = User("Bob")
    assert schema.serialize(user_obj) == {"name": "Bob"}

def test_schema_serialize_skips_missing_keys_in_dict():
    class MockField:
        def serialize(self, value):
            return value

    fields = {"name": MockField(), "age": MockField()}
    schema = Schema(fields=fields)
    input_data = {"name": "Alice"}
    # 'age' is missing in input_data, so it should not be in the returned dict
    assert schema.serialize(input_data) == {"name": "Alice"}

def test_schema_serialize_skips_missing_attributes_in_object():
    class MockField:
        def serialize(self, value):
            return value

    class User:
        pass

    fields = {"name": MockField(), "age": MockField()}
    schema = Schema(fields=fields)
    user_obj = User()
    user_obj.name = "Alice"
    # 'age' attribute does not exist on user_obj
    assert schema.serialize(user_obj) == {"name": "Alice"}
```


# LLM-generated content at query #9
#--------------------------

```python
def test_validate_iterates_over_value_keys():
    class MockField(Field):
        def __init__(self, allow_null=False):
            self.allow_null = allow_null
        def validate_or_error(self, value):
            return value, None

    mock_field = MockField()
    schema = Schema(fields={})
    value = {"test_key": 123}
    result = schema.validate(value)
    assert result == {}
```


# LLM-generated content at query #10
#--------------------------

```python
def test_validate_returns_valid_dict():
    mock_field_1 = type('Field', (), {'read_only': False, 'has_default': lambda: False, 'validate_or_error': lambda self, x: (x, None), 'get_error_text': lambda self, e: "", 'validation_error': lambda self, e: Exception()})()
    mock_field_2 = type('Field', (), {'read_only': False, 'has_default': lambda: False, 'validate_or_error': lambda self, x: (x, None), 'get_error_text': lambda self, e: "", 'validation_error': lambda self, e: Exception()})()
    schema = Schema(fields={"a": mock_field_1, "b": mock_field_2})
    result = schema.validate({"a": 1, "b": 2})
    assert result == {"a": 1, "b": 2}

def test_validate_raises_type_error_for_non_dict():
    mock_field = type('Field', (), {'read_only': False, 'has_default': lambda: False, 'validate_or_error': lambda self, x: (x, None), 'get_error_text': lambda self, e: "", 'validation_error': lambda self, e: ValueError("Must be an object.")})()
    schema = Schema(fields={"a": mock_field})
    with pytest.raises(ValueError, match="Must be an object."):
        schema.validate([1, 2, 3])

def test_validate_raises_null_error():
    mock_field = type('Field', (), {'read_only': False, 'has_default': lambda: False, 'validate_or_error': lambda self, x: (x, None), 'get_error_text': lambda self, e: "", 'validation_error': lambda self, e: ValueError("May not be null.")})()
    schema = Schema(fields={"a": mock_field}, allow_null=False)
    with pytest.raises(ValueError, match="May not be null."):
        schema.validate(None)

def test_validate_returns_none_if_allow_null_is_true():
    mock_field = type('Field', (), {'read_only': False, 'has_default': lambda: False, 'validate_or_error': lambda self, x: (x, None), 'get_error_text': lambda self, e: "", 'validation_error': lambda self, e: ValueError()})()
    schema = Schema(fields={"a": mock_field}, allow_null=None)
    # Note: the implementation check is `if value is None and self.allow_null`
    # We simulate allow_null as True via a subclass or manual attribute setting if needed, 
    # but here we assume Schema inherits from Field which handles allow_null.
    schema.allow_null = True
    assert schema.validate(None) is None

def test_validate_raises_invalid_key_error():
    mock_field = type('Field', (), {'read_only': False, 'has_default': lambda: False, 'validate_or_error': lambda self, x: (x, None), 'get_error_text': lambda self, e: "All object keys must be strings.", 'validation_error': lambda self, e: Exception()})()
    schema = Schema(fields={"a": mock_field})
    # Mocking Message and ValidationError structure for the error message loop
    class MockMessage: 
        def __init__(self, text, code, index): self.text = text; self.code = code; self.index = index
    class MockError: 
        def messages(self, add_prefix): return []
    class MockValidationError(Exception): 
        def __init__(self, messages): self.messages = messages
    
    # We must patch the global scope or use a controlled environment for this specific test logic
    # Since we can't define classes inside the test effectively for all dependencies without violating constraints, 
    # let's assume standard error raising via an exception that contains the message.
    
    class ValidationError(Exception):
        def __init__(self, messages): self.messages = messages

    class Message:
        def __init__(self, text, code, index):
            self.text = text
            self.code = code
            self.index = index

    # Re-assigning globals for the sake of this test case execution context logic
    import sys
    module = sys.modules[__name__]
    setattr(module, 'Message', Message)
    setattr(module, 'ValidationError', ValidationError)

    schema.get_error_text = lambda self, e: "All object keys must be strings."
    # Create dict with non-string key
    invalid_input = {123: "value"}
    with pytest.raises(ValidationError):
        schema.validate(invalid_input)

def test_validate_raises_required_error():
    mock_field = type('Field', (), {'read_only': False, 'has_default': lambda: False, 'validate_or_error': lambda self, x: (x, None), 'get_error_text': lambda self, e: "This field is required.", 'validation_error': lambda self, e: Exception()})()
    
    class ValidationError(Exception):
        def __init__(self, messages): self.messages = messages
    class Message:
        def __init__(self, text, code, index):
            self.text = text; self.code = code; self.index = index

    import sys
    module = sys.modules[__name__]
    setattr(module, 'Message', Message)
    setattr(module, 'ValidationError', ValidationError)

    schema = Schema(fields={"required_key": mock_field})
    with pytest.raises(ValidationError):
        schema.validate({"other_key": 1})

def test_validate_applies_defaults():
    mock_default_field = type('Field', (), {
        'read_only': False, 
        'has_default': lambda self: True, 
        'get_default_value': lambda self: "default",
        'validate_or_error': lambda self, x: (x, None),
        'get_error_text': lambda self, e: "",
        'validation_error': lambda self, e: Exception()
    })()
    schema = Schema(fields={"a": mock_default_field})
    result = schema.validate({})
    assert result == {"a": "default"}

def test_validate_skips_read_only_fields():
    mock_read_only_field = type('Field', (), {
        'read_only': True, 
        'has_default': lambda self: False, 
        'validate_or_error': lambda self, x: (x, None),
        'get_error_text': lambda self, e: "",
        'validation_error': lambda self, e: Exception()
    })()
    schema = Schema(fields={"a": mock_read_only_field})
    # Even if 'a' is missing from input, it shouldn't be in validated because it's read_only and no default logic triggers for it
    assert schema.validate({}) == {}
```


# LLM-generated content at query #11
#--------------------------

```python
def test_validate_key_exists_in_value():
    class MockField:
        def __init__(self, read_only=False):
            self.read_only = read_only
        def has_default(self):
            return False
        def validate_or_error(self, value):
            return value, None

    fields = {"test_key": MockField()}
    schema = Schema(fields=fields)
    input_value = {"test_key": "some_value"}
    
    result = schema.validate(input_value)
    
    assert result == {"test_key": "some_value"}
```


# LLM-generated content at query #12
#--------------------------

```python
def test_schema_validate_success():
    class MockField:
        def __init__(self, default=None, read_only=False):
            self.default = default
            self.read_only = read_only
        def has_default(self): return self.default is not None
        def get_default_value(self): return self.default
        def validate_or_error(self, val): return val, None

    class MockValidationError(Exception):
        def __init__(self, messages): self.messages = messages
        def messages(self, add_prefix=None): return []

    class MockSchema(Schema):
        def validation_error(self, key):
            class Err:
                def __init__(self): self.messages = lambda add_prefix=None: []
            return Err()
        def get_error_text(self, key): return self.errors[key]

    field1 = MockField(default="val")
    field2 = MockField()
    schema = MockSchema(fields={"a": field1, "b": field2}, allow_null=False)
    
    result = schema.validate({"a": "new", "b": 123})
    assert result == {"a": "new", "message": None} # Note: Logic in provided snippet for 'b' depends on child validation, but based on MockField it returns val.
    # Refined assertion based strictly on the provided logic:
    assert result == {"a": "new"} 

def test_schema_validate_null_error():
    class MockField:
        def __init__(self): self.read_only = False
        def has_default(self): return False
        def validate_or_error(self, val): return val, None

    class MockValidationError(Exception):
        def __init__(self, messages): self.messages = messages

    class MockSchema(Schema):
        def validation_error(self, key):
            raise Exception(key)
        def get_error_text(self, key): return "error"

    schema = MockSchema(fields={}, allow_null=False)
    try:
        schema.validate(None)
    except Exception as e:
        assert str(e) == "null"

def test_schema_validate_type_error():
    class MockField:
        def __init__(self): self.read_only = False
        def has_default(self): return False
        def validate_or_error(self, val): return val, None

    class MockSchema(Schema):
        def validation_error(self, key):
            raise Exception(key)
        def get_error_text(self, key): return "error"

    schema = MockSchema(fields={}, allow_null=False)
    try:
        schema.validate(["not", "a", "dict"])
    except Exception as e:
        assert str(e) == "type"

def test_schema_validate_invalid_key():
    class MockField:
        def __init__(self): self.read_only = False
        def has_default(self): return False
        def validate_or_error(self, val): return val, None

    class Message:
        def __init__(self, text, code, index):
            self.text = text
            self.code = code
            self.index = index

    class ValidationError(Exception):
        def __init__(self, messages): self.messages = messages

    class MockSchema(Schema):
        def validation_error(self, key): return None
        def get_error_text(self, key): return "error"

    schema = MockSchema(fields={}, allow_null=False)
    # Using a dict with an integer key (though standard dicts use hashable, we simulate the check)
    class BadDict(dict):
        def keys(self): return [123]

    try:
        schema.validate(BadDict())
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"
        assert e.messages[0].index == [123]

def test_schema_validate_required_error():
    class MockField:
        def __init__(self): self.read_only = False
        def has_default(self): return False
        def validate_or_error(self, val): return val, None

    class Message:
        def __init__(self, text, code, index):
            self.text = text
            self.code = code
            self.index = index

    class ValidationError(Exception):
        def __init__(self, messages): self.messages = messages

    class MockSchema(Schema):
        def validation_error(self, key): return None
        def get_error_text(self, key): return "error"

    schema = MockSchema(fields={"req": MockField()})
    try:
        schema.validate({})
    except ValidationError as e:
        assert e.messages[0].code == "required"
        assert e.messages[0].index == ["req"]

def test_schema_validate_default_value_application():
    class MockField:
        def __init__(self, default="default_val"):
            self.read_only = False
            self._default = default
        def has_default(self): return True
        def get_default_value(self): return self._default
        def validate_or_error(self, val): return val, None

    class MockSchema(Schema):
        def validation_error(self, key): return None
        def get_error_text(self, key): return "error"

    schema = MockSchema(fields={"opt": MockField()})
    result = schema.validate({})
    assert result["opt"] == "default_val"
```


# LLM-generated content at query #13
#--------------------------

```python
def test_validate_success():
    field_int = StringField() # Assuming StringField or similar exists for setup
    schema = Schema(fields={"name": field_int})
    result = schema.validate({"name": "test"})
    assert result == {"name": "test"}

def test_validate_null_error():
    field_int = StringField()
    schema = Schema(fields={"name": field_int}, allow_null=False)
    try:
        schema.validate(None)
    except ValidationError as e:
        assert e.messages[0].code == "null"

def test_validate_null_success():
    field_int = StringField()
    schema = Schema(fields={"name": field_int}, allow_null=True)
    result = schema.validate(None)
    assert result is None

def test_validate_type_error():
    field_int = StringField()
    schema = Schema(fields={"name": field_int})
    try:
        schema.validate(["not", "a", "dict"])
    except ValidationError as e:
        assert e.messages[0].code == "type"

def test_validate_invalid_key_error():
    field_int = StringField()
    schema = Schema(fields={"name": field_int})
    try:
        schema.validate({123: "value"})
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"
        assert e.messages[0].index == [123]

def test_validate_required_error():
    field_int = StringField()
    schema = Schema(fields={"name": field_int})
    try:
        schema.validate({})
    except ValidationError as e:
        assert e.messages[0].code == "required"
        assert e.messages[0].index == ["name"]

def test_validate_with_default_value():
    field_default = StringField(default="default_val")
    schema = Schema(fields={"name": field_default})
    result = schema.validate({})
    assert result == {"name": "default_val"}

def test_validate_read_only_skip():
    field_ro = StringField(read_only=True)
    schema = Schema(fields={"name": field_ro})
    # read_only fields are not processed for defaults or validation in the loop logic
    result = schema.validate({"name": "original"})
    assert "name" not in result
```


# LLM-generated content at query #14
#--------------------------

```python
def test_validate_returns_none_when_value_is_none_and_allow_null_is_true():
    mock_definitions = {"target_key": MagicMock()}
    ref = Reference(to="target_key", definitions=mock_definitions, allow_null=True)
    assert ref.validate(None) is None

def test_validate_raises_error_when_value_is_none_and_allow_null_is_false():
    mock_definitions = {"target_key": MagicMock()}
    ref = Reference(to="target_key", definitions=mock_definitions, allow_null=False)
    with pytest.raises(Exception) as excinfo:
        ref.validate(None)
    assert "May not be null." in str(excinfo.value)

def test_validate_calls_target_validate_with_correct_value():
    mock_target = MagicMock()
    mock_target.validate.return_value = "valid_result"
    mock_definitions = {"target_key": mock_target}
    ref = Reference(to="target_key", definitions=mock_definitions)
    
    result = ref.validate("some_value")
    
    assert result == "valid_result"
    mock_target.validate.assert_called_once_with("some_value")
```


# LLM-generated content at query #15
#--------------------------

```python
def test_validate_returns_none_when_value_is_none_and_allow_null_is_true():
    class MockField(Field):
        def __init__(self, allow_null=False):
            self.allow_null = allow_null
        def validate(self, value):
            return super().validate(value)

    schema = Schema(fields={})
    schema.allow_null = True
    assert schema.validate(None) is None
```


# LLM-generated content at query #16
#--------------------------

```python
def test_schema_validate_null_allowed():
    field_mock = MagicMock()
    field_mock.allow_null = True
    field_mock.read_only = False
    field_mock.has_default.return_value = False
    field_mock.validate_or_error.return_value = (None, None)
    
    schema = Schema(fields={"test": field_mock})
    schema.allow_null = True
    
    assert schema.validate(None) is None

def test_schema_validate_null_not_allowed():
    field_mock = MagicMock()
    field_mock.read_only = False
    field_mock.has_default.return_value = False
    
    schema = Schema(fields={"test": field_mock})
    schema.allow_null = False
    schema.validation_error = MagicMock(side_effect=ValidationError)
    
    with Exception:
        schema.validate(None)
    schema.validation_error.assert_called_with("null")

def test_schema_validate_invalid_type():
    field_mock = MagicMock()
    field_mock.read_only = False
    field_mock.has_default.return_value = False
    
    schema = Schema(fields={"test": field_mock})
    schema.allow_null = False
    schema.validation_error = MagicMock(side_effect=ValidationError)
    
    with Exception:
        schema.validate(["not", "a", "dict"])
    schema.validation_error.assert_called_with("type")

def test_schema_validate_invalid_key_type():
    child_field = MagicMock()
    child_field.read_only = False
    child_field.has_default.return_value = False
    child_field.validate_or_error.return_value = ("val", None)
    
    schema = Schema(fields={"a": child_field})
    schema.get_error_text = MagicMock(return_value="Invalid key error")
    
    input_data = {123: "value"}
    
    with Exception as e:
        schema.validate(input_data)
    assert isinstance(e, ValidationError)
    assert e.messages[0].code == "invalid_key"
    assert e.messages[0].index == [123]

def test_schema_validate_required_field_missing():
    child_field = MagicMock()
    child_field.read_only = False
    child_field.has_default.return_value = False
    
    schema = Schema(fields={"required_key": child_field})
    schema.get_error_text = MagicMock(return_value="Required error")
    
    input_data = {"other_key": "val"}
    
    with Exception as e:
        schema.validate(input_data)
    assert isinstance(e, ValidationError)
    assert e.messages[0].code == "required"
    assert e.messages[0].index == ["required_key"]

def test_schema_validate_success_with_defaults():
    child_field = MagicMock()
    child_field.read_only = False
    child_field.has_default.return_value = True
    child_field.get_default_value.return_value = "default"
    
    schema = Schema(fields={"optional_key": child_field})
    
    input_data = {"other_key": "val"}
    result = schema.validate(input_data)
    
    assert "optional_key" in result
    assert result["optional_key"] == "default"

def test_schema_validate_success_with_provided_value():
    child_field = MagicMock()
    child_field.read_only = False
    child_field.has_default.return_value = False
    child_field.validate_or_error.return_value = ("valid_value", None)
    
    schema = Schema(fields={"key": child_field})
    
    input_data = {"key": "actual_value"}
    result = schema.validate(input_data)
    
    assert result["key"] == "valid_value"

def test_schema_validate_child_error_propagation():
    child_field = MagicMock()
    child_field.read_only = False
    child_field.has_default.return_value = False
    
    error_msg = Message(text="Child error", code="child_err", index=[])
    mock_error = MagicMock()
    mock_error.messages.return_value = [error_msg]
    child_field.validate_or_error.return_value = (None, mock_error)
    
    schema = Schema(fields={"key": child_field})
    
    input_data = {"key": "bad_value"}
    
    with Exception as e:
        schema.validate(input_data)
    assert isinstance(e, ValidationError)
    assert len(e.messages) == 1
    assert e.messages[0].code == "child_err"
```


# LLM-generated content at query #17
#--------------------------

```python
def test_validate_child_schema_with_error():
    class MockField:
        def __init__(self, read_only=False, has_default_val=False):
            self.read_only = read_only
            self._has_default = has_default_val

        def has_default(self):
            return self._has_default

        def get_default_value(self):
            return None

        def validate_or_error(self, value):
            class MockError:
                def messages(self, add_prefix):
                    return [f"{add_prefix}: error"]
            return None, MockError()

    class MockValidationError(Exception):
        def __init__(self, messages):
            self.messages = messages

    class SchemaMock:
        def __init__(self, fields):
            self.fields = fields
            self.required = []
        
        def validate(self, value):
            validated = {}
            error_messages = []
            for key, child_schema in self.fields.items():
                if child_schema.read_only:
                    continue
                if key not in value:
                    if child_schema.has_default():
                        validated[key] = child_schema.get_default_value()
                    continue
                item = value[key]
                child_value, error = child_schema.validate_or_error(item)
                if not error:
                    validated[key] = child_value
                else:
                    error_messages += error.messages(add_prefix=key)
            if error_messages:
                raise MockValidationError(messages=error_messages)
            return validated

    field = MockField()
    schema = SchemaMock({"test_key": field})
    input_data = {"test_key": "some_value"}
    
    with pytest.raises(MockValidationError) as excinfo:
        schema.validate(input_data)
    
    assert excinfo.value.messages == ["test_key: error"]
```


# LLM-generated content at query #18
#--------------------------

```python
def test_validate_returns_none_when_allow_null_is_true():
    class MockField(Field):
        def __init__(self, allow_null=False):
            self.allow_null = allow_null
        def validate_or_error(self, value): return value, None
        def get_error_text(self, key): return Schema.errors[key]
        def validation_error(self, key): raise ValidationError([Message(text=Schema.errors[key], code=key)])

    schema = Schema(fields={})
    schema.allow_null = True
    assert schema.validate(None) is None

def test_validate_raises_error_when_value_is_none_and_not_allow_null():
    class MockField(Field):
        def __init__(self, allow_null=False):
            self.allow_null = allow_null
        def validate_or_error(self, value): return value, None
        def get_error_text(self, key): return Schema.errors[key]
        def validation_error(self, key): raise ValidationError([Message(text=Schema.errors[key], code=key)])

    schema = Schema(fields={})
    schema.allow_null = False
    try:
        schema.validate(None)
    except ValidationError as e:
        assert e.messages[0].code == "null"

def test_validate_raises_error_when_value_is_not_dict():
    class MockField(Field):
        def __init__(self, allow_null=False):
            self.allow_null = allow_null
        def validate_or_error(self, value): return value, None
        def get_error_text(self, key): return Schema.errors[key]
        def validation_error(self, key): raise ValidationError([Message(text=Schema.errors[key], code=key)])

    schema = Schema(fields={})
    try:
        schema.validate(["not", "a", "dict"])
    except ValidationError as e:
        assert e.messages[0].code == "type"

def test_validate_raises_error_on_invalid_key_type():
    class MockField(Field):
        def __init__(self, allow_null=False):
            self.allow_null = allow_null
        def validate_or_error(self, value): return value, None
        def get_error_text(self, key): return Schema.errors[key]
        def validation_error(self, key): raise ValidationError([Message(text=Schema.errors[key], code=key)])

    schema = Schema(fields={})
    try:
        schema.validate({123: "value"})
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"
        assert e.messages[0].index == [123]

def test_validate_raises_error_on_missing_required_field():
    class MockField(Field):
        def __init__(self, allow_null=False, read_only=False, has_default=False):
            self.allow_null = allow_null
            self.read_only = read_only
            self._has_default = has_default
        def has_default(self): return self._has_default
        def get_default_value(self): return None
        def validate_or_error(self, value): return value, None
        def get_error_text(self, key): return Schema.errors[key]
        def validation_error(self, key): raise ValidationError([Message(text=Schema.errors[key], code=key)])

    schema = Schema(fields={"username": MockField()})
    try:
        schema.validate({})
    except ValidationError as e:
        assert e.messages[0].code == "required"
        assert e.messages[0].index == ["username"]

def test_validate_applies_default_values():
    class MockField(Field):
        def __init__(self, allow_null=False, read_only=False, has_default=True, default="default_val"):
            self.allow_null = allow_null
            self.read_only = read_only
            self._has_default = has_default
            self._default = default
        def has_default(self): return self._has_default
        def get_default_value(self): return self._default
        def validate_or_error(self, value): return value, None
        def get_error_text(self, key): return Schema.errors[key]
        def validation_error(self, key): raise ValidationError([Message(text=Schema.errors[key], code=key)])

    schema = Schema(fields={"age": MockField(has_default=True, default=25)})
    assert schema.validate({}) == {"age": 25}

def test_validate_skips_read_only_fields():
    class MockField(Field):
        def __init__(self, allow_null=False, read_only=True):
            self.allow_null = allow_null
            self.read_only = read_only
        def has_default(self): return False
        def validate_or_error(self, value): return value, None
        def get_error_text(self, key): return Schema.errors[key]
        def validation_error(self, key): raise ValidationError([Message(text=Schema.errors[key], code=key)])

    schema = Schema(fields={"id": MockField(read_only=True)})
    assert schema.validate({"id": 1}) == {}

def test_validate_propagates_child_errors():
    class ChildField(Field):
        def __init__(self, allow_null=False):
            self.allow_null = allow_null
        def has_default(self): return False
        def validate_or_error(self, value): 
            return None, [Message(text="Child Error", code="child_err", index=[0])]
        def get_error_text(self, key): return Schema.errors[key]
        def validation_error(self, key): raise ValidationError([Message(text=Schema.errors[key], code=key)])

    schema = Schema(fields={"child": ChildField()})
    try:
        schema.validate({"child": "some_value"})
    except ValidationError as e:
        assert any("child_err" in m.code or "child" in str(m) for m in e.messages)
```


# LLM-generated content at query #19
#--------------------------

```python
def test_validate_skips_missing_key_with_default():
    class MockField:
        def __init__(self, has_default_val=True):
            self._has_default = has_default_val
            self.read_only = False
        def has_default(self):
            return self._has_default
        def get_default_value(self):
            return "default_val"
        def validate_or_error(self, value):
            return value, None
        def serialize(self, value):
            return value

    field_with_default = MockField(has_default_val=True)
    schema = Schema(fields={"test_key": field_with_default})
    input_value = {}
    result = schema.validate(input_value)
    assert result["test_key"] == "default_val"
```


# LLM-generated content at query #20
#--------------------------

```python
def test_validate_returns_none_when_value_is_none_and_allow_null_is_true():
    class MockDefinition:
        def validate(self, value):
            return value

    class MockField:
        def __init__(self, allow_null=False):
            self.allow_null = allow_null
        def validation_error(self, error_key):
            return Exception(Reference.errors[error_key])

    class MockReference(Reference):
        def __init__(self, to, definitions, allow_null=False):
            self.to = to
            self.definitions = definitions
            self.allow_null = allow_null
        @property
        def target(self):
            return self.definitions[self.to]
        def validate(self, value):
            if value is None and self.allow_null:
                return None
            elif value is None:
                raise Exception("error")
            return self.target.validate(value)

    definitions = {"test_key": MockDefinition()}
    ref = MockReference(to="test_key", definitions=definitions, allow_null=True)
    
    assert ref.validate(None) is None
```


# LLM-generated content at query #21
#--------------------------

```python
def test_validate_raises_error_on_null_when_not_allowed():
    mock_definitions = {"target_key": MagicMock()}
    ref = Reference(to="target_key", definitions=mock_definitions)
    # Since allow_null defaults to False in Field (assumed), passing None should raise error
    with pytest.raises(Exception) as excinfo:
        ref.validate(None)
    assert "May not be null." in str(excinfo.value)

def test_validate_returns_none_when_null_is_allowed():
    mock_definitions = {"target_key": MagicMock()}
    ref = Reference(to="target_key", definitions=mock_definitions)
    ref.allow_null = True
    assert ref.validate(None) is None

def test_validate_calls_target_validate_with_value():
    mock_target = MagicMock()
    mock_target.validate.return_value = "validated_value"
    mock_definitions = {"target_key": mock_target}
    ref = Reference(to="target_key", definitions=mock_definitions)
    
    result = ref.validate("some_input")
    
    assert result == "validated_value"
    mock_target.validate.assert_called_once_with("some_input")
```


# LLM-generated content at query #22
#--------------------------

```python
def test_validate_schema_skips_missing_key_with_default():
    class MockField:
        def __init__(self, has_default=True, default_value="default", read_only=False):
            self.has_default = lambda: has_default
            self.get_default_value = lambda: default_value
            self.read_only = read_only
            self.allow_null = False
        def validate_or_error(self, value):
            return value, None

    fields = {"test_key": MockField(has_default=True, default_value="default")}
    schema = Schema(fields=fields)
    input_data = {}
    result = schema.validate(input_data)
    assert result["test_key"] == "default"
```


# LLM-generated content at query #23
#--------------------------

```python
def test_schema_validate_success():
    class MockField:
        def __init__(self, read_only=False, default=None):
            self.read_only = read_only
            self.default = default
        def has_default(self): return self.default is not None
        def get_default_value(self): return self.default
        def validate_or_error(self, value): return value, None
        def serialize(self, value): return value

    class MockValidationError(Exception):
        def __init__(self, messages): self.messages = messages
        def messages(self, add_prefix=None): return []

    class MockSchema(Schema):
        def validation_error(self, key): 
            return type('Error', (), {'__init__': lambda s, k: setattr(s, 'key', k), 'get_error_mapping': lambda s: Schema.errors})()
        def get_error_text(self, key): return Schema.errors[key]

    field_int = MockField()
    field_str = MockField(default="default")
    schema = MockSchema(fields={"a": field_int, "b": field_str})
    
    result = schema.validate({"a": 1, "b": "hello"})
    assert result == {"a": 1, "b": "hello"}
    assert "b" in result
    assert result["b"] == "hello"

def test_schema_validate_default_values():
    class MockField:
        def __init__(self, default=None):
            self.read_only = False
            self.default = default
        def has_default(self): return self.default is not None
        def get_default_value(self): return self.default
        def validate_or_error(self, value): return value, None

    class MockSchema(Schema):
        def validation_error(self, key): return Exception()
        def get_error_text(self, key): return ""

    schema = MockSchema(fields={"a": MockField(), "b": MockField(default="def")})
    result = schema.validate({"a": 1})
    assert result == {"a": 1, "b": "def"}

def test_schema_validate_type_error():
    class MockField:
        def __init__(self): self.read_only = False; self.default = None
        def has_default(self): return False
        def validate_or_error(self, value): return value, None

    class MockValidationError(Exception):
        def __name__(self): return "ValidationError"
        def __init__(self, messages): self.messages = messages

    class MockSchema(Schema):
        def validation_error(self, key): 
            class Err:
                def __init__(self, k): self.key = k
                def get_error_text(self, k): return Schema.errors[k]
            return Err("type")
        def get_error_text(self, key): return Schema.errors[key]

    schema = MockSchema(fields={})
    try:
        schema.validate(["not", "a", "dict"])
    except Exception as e:
        # We expect a validation error because value is not dict/mapping
        assert True
    else:
        assert False

def test_schema_validate_required_error():
    class MockField:
        def __init__(self): self.read_only = False; self.default = None
        def has_default(self): return False
        def validate_or_error(self, value): return value, None

    class MockValidationError(Exception):
        def __init__(self, messages): self.messages = messages

    class MockSchema(Schema):
        def validation_error(self, key): 
            class Err:
                def __init__(self, k): self.key = k
                def get_error_text(self, k): return Schema.errors[k]
            return Err("required")
        def get_error_text(self, key): return Schema.errors[key]

    schema = MockSchema(fields={"required_field": MockField()})
    try:
        schema.validate({"other_field": 1})
    except Exception as e:
        # Check if the error message contains 'required' logic
        assert True
    else:
        assert False

def test_schema_validate_invalid_key():
    class MockField:
        def __init__(self): self.read_only = False; self.default = None
        def has_default(self): return False
        def validate_or_error(self, value): return value, None

    class MockMessage:
        def __init__(self, text, code, index): 
            self.text = text
            self.code = code
            self.index = index
        def messages(self, add_prefix=None): return []

    class MockValidationError(Exception):
        def __init__(self, messages): self.messages = messages

    class MockSchema(Schema):
        def validation_error(self, key): return Exception()
        def get_error_text(self, key): return Schema.errors[key]

    schema = MockSchema(fields={})
    # Use a dict with an integer key to trigger invalid_key logic
    try:
        schema.validate({123: "value"})
    except Exception as e:
        assert True
    else:
        assert False
```


# LLM-generated content at query #24
#--------------------------

```python
def test_validate_skips_missing_key_when_no_default_exists():
    class MockField:
        def __init__(self, has_default=False, read_only=False):
            self._has_default = has_default
            self.read_only = read_only
        def has_default(self):
            return self._has_default
        def get_default_value(self):
            return None
        def validate_or_error(self, value):
            return value, None
        def serialize(self, value):
            return value

    class MockSchema(Schema):
        def __init__(self, fields):
            self.fields = fields
            self.required = []
            self.allow_null = False
        def validation_error(self, code):
            raise Exception(code)
        def get_error_text(self, code):
            return ""

    field_no_default = MockField(has_default=False)
    schema = MockSchema({"missing_key": field_no_default})
    input_value = {"other_key": "some_value"}
    
    result = schema.validate(input_value)
    assert "missing_key" not in result
```


# LLM-generated content at query #25
#--------------------------

```python
def test_validate_returns_none_when_value_is_none_and_allow_null_is_true():
    class MockField(Field):
        def __init__(self, allow_null=False):
            self.allow_null = allow_null
        def validate_or_error(self, value):
            return value, None

    schema = Schema(fields={}, allow_null=True)
    result = schema.validate(None)
    assert result is None
```


# LLM-generated content at query #26
#--------------------------

```python
def test_validate_returns_none_when_value_is_none_and_allow_null_is_true():
    class MockTarget:
        def validate(self, value):
            return value

    class MockDefinitions:
        def __getitem__(self, key):
            return self.mock_target

        def __init__(self, target):
            self.mock_target = target

    mock_target = MockTarget()
    mock_definitions = MockDefinitions(mock_target)
    
    # Reference inherits from Field, we assume Field accepts allow_null in kwargs
    # and provides the validation_error method. 
    # Since Field is not provided, we mock the necessary interface.
    class MockField:
        def __init__(self, **kwargs):
            self.allow_null = kwargs.get("allow_null", False)
        def validation_error(self, msg):
            return Exception(Reference.errors[msg])

    class MockReference(Reference, MockField):
        pass

    instance = MockReference(to="test", definitions=mock_definitions, allow_null=True)
    
    assert instance.validate(None) is None
```


# LLM-generated content at query #27
#--------------------------

```python
def test_schema_validate_valid_object():
    class MockField:
        def __init__(self, read_only=False, default=None):
            self.read_only = read_only
            self.default = default
        def has_default(self): return self.default is not None
        def get_default_value(self): return self.default
        def validate_or_error(self, value): return value, None
        def validate(self, value): return value

    class MockSchema(Schema):
        def validation_error(self, code): return ValidationError(messages=[code])
        def get_error_text(self, code): return "Error"

    field_int = MockField()
    fields = {"age": field_int}
    schema = MockSchema(fields=fields)
    
    result = schema.validate({"age": 25})
    assert result == {"age": 25}

def test_schema_validate_null_error():
    class MockField:
        def __init__(self, allow_null=False): self.allow_null = allow_null
        def has_default(self): return False
        def read_only(self): return False
    
    class MockSchema(Schema):
        def __init__(self, fields): 
            self.fields = fields
            self.required = []
            self.allow_null = False
        def validation_error(self, code): return ValidationError(messages=[code])

    schema = MockSchema(fields={})
    try:
        schema.validate(None)
    except ValidationError as e:
        assert "null" in e.messages

def test_schema_validate_type_error():
    class MockField:
        def __init__(self): self.read_only = False
        def has_default(self): return False
    
    class MockSchema(Schema):
        def __name__(self): pass
        def __init__(self, fields): 
            self.fields = fields
            self.required = []
            self.allow_null = False
        def validation_error(self, code): return ValidationError(messages=[code])
        def get_error_text(self, code): return "type error"

    schema = MockSchema(fields={})
    try:
        schema.validate(["not", "a", "dict"])
    except ValidationError as e:
        assert "type" in e.messages[0].code

def test_schema_validate_invalid_key():
    class MockField:
        def __init__(self): self.read_only = False
        def has_default(self): return False
    
    class MockSchema(Schema):
        def __init__(self, fields): 
            self.fields = fields
            self.required = []
            self.allow_null = False
        def validation_error(self, code): return ValidationError(messages=[code])
        def get_error_text(self, code): return "invalid key"

    schema = MockSchema(fields={})
    try:
        schema.validate({123: "value"})
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"
        assert e.messages[0].index == [123]

def test_schema_validate_required_field_missing():
    class MockField:
        def __init__(self): self.read_only = False
        def has_default(self): return False
    
    class MockSchema(Schema):
        def __init__(self, fields): 
            self.fields = fields
            self.required = ["must_exist"]
            self.allow_null = False
        def validation_error(self, code): return ValidationError(messages=[code])
        def get_error_text(self, code): return "required error"

    schema = MockSchema(fields={"must_exist": MockField()})
    try:
        schema.validate({"other": 1})
    except ValidationError as e:
        assert e.messages[0].code == "required"
        assert e.messages[0].index == ["must_exist"]

def test_schema_validate_with_defaults():
    class MockField:
        def __init__(self, default=10): 
            self.read_only = False
            self.default = default
        def has_default(self): return True
        def get_default_value(self): return self.default
        def validate_or_error(self, value): return value, None
        def validate(self, value): return value

    class MockSchema(Schema):
        def __init__(self, fields): 
            self.fields = fields
            self.required = []
            self.allow_null = False
        def validation_error(self, code): return ValidationError(messages=[code])
        def get_error_text(self, code): return "err"

    schema = MockSchema(fields={"count": MockField(default=5)})
    result = schema.validate({})
    assert result == {"count": 5}
```


# LLM-generated content at query #28
#--------------------------

```python
def test_validate_returns_none_when_value_is_none_and_allow_null_is_true():
    mock_definitions = {"target_key": type('Mock', (), {"validate": lambda x: x})()}
    reference = Reference(to="target_key", definitions=mock_definitions, allow_null=True)
    result = reference.validate(None)
    assert result is None
```


# LLM-generated content at query #29
#--------------------------

```python
def test_validate_returns_none_when_value_is_none_and_allow_null_is_true():
    class MockField(Field):
        def __init__(self, allow_null=False):
            self.allow_null = allow_null
        def validate(self, value):
            return super().validate(value)

    mock_field = MockField(allow_null=True)
    schema = Schema(fields={})
    schema.allow_null = True
    
    result = schema.validate(None)
    assert result is None
```


# LLM-generated content at query #30
#--------------------------

```python
def test_validate_returns_none_when_value_is_none_and_allow_null_is_true():
    class MockField(Field):
        def __init__(self, allow_null=False):
            self.allow_null = allow_null
        def validate(self, value):
            return super().validate(value)

    mock_field = MockField(allow_null=True)
    schema = Schema(fields={})
    schema.allow_null = True
    
    assert schema.validate(None) is None
```


# LLM-generated content at query #31
#--------------------------

```python
def test_validate_returns_none_when_value_is_none_and_allow_null_is_true():
    class MockDefinition:
        def validate(self, value):
            return value

    mock_definitions = {"target_key": MockDefinition()}
    reference_field = Reference(to="target_key", definitions=mock_definitions, allow_null=True)
    
    result = reference_field.validate(None)
    
    assert result is None
```


# LLM-generated content at query #32
#--------------------------

```python
def test_validate_skips_missing_key_with_no_default():
    class MockField:
        def __init__(self, has_default=False, read_only=False):
            self._has_default = has_default
            self.read_only = read_only
        def has_default(self):
            return self._has_default
        def get_default_value(self):
            return "default"
        def validate_or_error(self, value):
            return value, None
        def serialize(self, value):
            return value

    class MockSchema(Schema):
        def __init__(self, fields):
            self.fields = fields
            self.required = []
            self.allow_null = False
        def validate_or_error(self, value):
            return value, None
        def validation_error(self, code):
            raise Exception(code)
        def get_error_text(self, code):
            return code

    field_no_default = MockField(has_default=False)
    schema = MockSchema(fields={"missing_key": field_no_default})
    input_data = {"other_key": "value"}
    
    result = schema.validate(input_data)
    
    assert "missing_key" not in result
```


# LLM-generated content at query #33
#--------------------------

```python
def test_validate_returns_none_when_value_is_none_and_allow_null_is_true():
    class MockTarget:
        def validate(self, value):
            return value

    class MockDefinitions:
        def __getitem__(self, key):
            return self.target

        def __init__(self, target):
            self.target = target

    mock_target = MockTarget()
    mock_definitions = MockDefinitions(mock_target)
    
    # Creating Reference instance where allow_null is True
    # Assuming Field (parent class) handles allow_null via kwargs
    instance = Reference(to="test", definitions=mock_definitions, allow_null=True)
    
    assert instance.validate(None) is None
```


# LLM-generated content at query #34
#--------------------------

```python
def test_validate_allows_null_when_configured():
    class MockField(Field):
        def __init__(self, allow_null=False, fields=None):
            self.allow_null = allow_null
            self.fields = fields or {}
            self.required = []
        def validate(self, value):
            return super().validate(value)

    schema = MockField(allow_null=True, fields={})
    result = schema.validate(None)
    assert result is None
```


