####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_schema_validate_success_with_defaults():
    class MockField:
        def __init__(self, read_only=False, default=None):
            self.read_only = read_only
            self.default = default
        def has_default(self): return self.default is not None
        def get_default_value(self): return self.default
        def validate_or_error(self, value): return value, None
        def validate(self, value): return value

    class MockError:
        def __init__(self, messages): self.messages = lambda add_prefix=None: messages
    
    class ValidationError(Exception):
        def __init__(self, messages): self.messages = lambda add_prefix=None: messages

    field_a = MockField(default="default_val")
    field_b = MockField()
    schema = Schema(fields={"a": field_a, "b": field_b})
    
    result = schema.validate({"b": 123})
    
    assert result == {"a": "default_val", "b": 123}

def test_schema_validate_raises_type_error():
    class MockField:
        def __init__(self): self.read_only = False
        def has_default(self): return False
        def validate_or_error(self, value): return value, None
        def validate(self, value): return value
    
    class ValidationError(Exception):
        def __name__(self): return "ValidationError"
        def __init__(self, messages): self.messages = lambda add_prefix=None: messages

    schema = Schema(fields={})
    
    with pytest.raises(ValidationError) as excinfo:
        schema.validate(["not", "a", "dict"])
    assert any(m.code == "type" for m in excinfo.value.messages(None))

def test_schema_validate_raises_null_error():
    class MockField:
        def __init__(self, allow_null=False): 
            self.read_only = False
            self.allow_null = allow_null
        def has_default(self): return False
        def validate_or_error(self, value): return value, None
        def validate(self, value): return value

    class ValidationError(Exception):
        def __init__(self, messages): self.messages = lambda add_prefix=None: messages

    schema = Schema(fields={})
    
    with pytest.raises(ValidationError) as excinfo:
        schema.validate(None)
    assert any(m.code == "null" for m in excinfo.value.messages(None))

def test_schema_validate_raises_required_error():
    class MockField:
        def __init__(self): self.read_only = False
        def has_default(self): return False
        def validate_or_error(self, value): return value, None
        def validate(self, value): return value

    class ValidationError(Exception):
        def __init__(self, messages): self.messages = lambda add_prefix=None: messages

    schema = Schema(fields={"required_key": MockField()})
    
    with pytest.raises(ValidationError) as excinfo:
        schema.validate({"other_key": 1})
    assert any(m.code == "required" and m.index == ["required_key"] for m in excinfo.value.messages(None))

def test_schema_validate_raises_invalid_key_error():
    class MockField:
        def __init__(self): self.read_only = False
        def has_default(self): return False
        def validate_or_error(self, value): return value, None
        def validate(self, value): return value

    class Message:
        def __init__(self, text, code, index):
            self.text = text
            self.code = code
            self.index = index

    class ValidationError(Exception):
        def __init__(self, messages): self.messages = lambda add_prefix=None: messages

    schema = Schema(fields={})
    
    with pytest.raises(ValidationError) as excinfo:
        schema.validate({123: "value"})
    assert any(m.code == "invalid_key" and m.index == [123] for m in excinfo.value.messages(None))

def test_schema_validate_skips_read_only_fields():
    class MockField:
        def __init__(self, read_only=True): 
            self.read_only = read_only
        def has_default(self): return False
        def validate_or_error(self, value): return value, None
        def validate(self, value): return value

    class ValidationError(Exception):
        def __init__(self, messages): self.messages = lambda add_prefix=None: messages

    schema = Schema(fields={"readonly": MockField(read_only=True)})
    
    result = schema.validate({"readonly": "new_value"})
    assert "readonly" not in result
```


# LLM-generated content at query #2
#--------------------------

```python
def test_schema_validate_key_missing_with_default_value_triggers_line_32():
    class MockField:
        def __init__(self, default=None, has_default_val=False):
            self.default = default
            self.has_default_val = has_default_val
            self.read_only = False
        def has_default(self):
            return self.has_default_val
        def get_default_value(self):
            return self.default
        def validate_or_error(self, value):
            return value, None

    class MockSchema(Schema):
        def __init__(self, fields):
            self.fields = fields
            self.required = []
            self.allow_null = False
        def validation_error(self, code):
            return Exception(code)
        def get_error_text(self, code):
            return code

    default_field = MockField(default="default_val", has_default_val=True)
    schema = MockSchema(fields={"test_key": default_field})
    
    result = schema.validate({})
    
    assert result["test_key"] == "default_val"
```


# LLM-generated content at query #3
#--------------------------

```python
def test_validate_skips_key_if_not_in_value_and_has_no_default():
    class MockField:
        def __init__(self, has_default_val=False):
            self.has_default = lambda: has_default_val
            self.read_only = False
            self.get_default_value = lambda: None
            self.validate_or_error = lambda self, v: (v, None)

    field_no_default = MockField(has_default_val=False)
    schema = Schema(fields={"missing_key": field_no_default})
    
    result = schema.validate({"existing_key": 123})
    
    assert "missing_key" not in result
```


# LLM-generated content at query #4
#--------------------------

```python
def test_validate_returns_none_when_value_is_none_and_allow_null_is_true():
    class MockField(Field):
        def __init__(self, allow_null=False):
            self.allow_null = allow_null
        def validate(self, value):
            return super().validate(value)

    schema = Schema(fields={}, allow_null=True)
    result = schema.validate(None)
    assert result is None
```


# LLM-generated content at query #5
#--------------------------

```python
def test_schema_constructor_initializes_fields_and_required_correctly():
    class MockField:
        def __init__(self, read_only=False, default=None, has_default=False):
            self.read_only = read_only
            self.default = default
            self.has_default = has_default
        def has_default(self): return self.has_default
        def get_default_value(self): return self.default

    field_required = MockField(read_only=False, has_default=False)
    field_readonly = MockField(read_only=True, has_default=False)
    field_default = MockField(read_only=False, has_default=True, default="val")
    field_default_readonly = MockField(read_only=True, has_default=True, default="val")

    fields = {
        "required_key": field_required,
        "readonly_key": field_readonly,
        "default_key": field_default,
        "default_readonly_key": field_default_readonly
    }
    
    schema = Schema(fields=fields, title="Test Schema", description="Test Desc")

    assert schema.fields == fields
    assert schema.title == "Test Schema"
    assert schema.description == "Test Desc"
    assert "required_key" in schema.required
    assert "readonly_key" not in schema.required
    assert "default_key" not in schema.required
    assert "default_readonly_key" not in schema.required

def test_schema_constructor_with_no_required_fields():
    class MockField:
        def __init__(self, read_only=True, has_default=True):
            self.read_only = read_only
            self.has_default = has_default
        def has_default(self): return self.has_default

    fields = {"opt": MockField(read_only=True, has_default=True)}
    schema = Schema(fields=fields)
    
    assert schema.required == []
```


# LLM-generated content at query #6
#--------------------------

```python
def test_schema_init_assigns_fields():
    from typesystem import Schema, Field
    fields = {"name": Field(title="Name")}
    schema = Schema(fields=fields)
    assert schema.fields == fields

def test_schema_init_calculates_required_fields():
    from typesystem import Schema, Field
    fields = {
        "required_field": Field(title="Required"),
        "default_field": Field(title="Default", default="something"),
        "read_only_field": Field(title="Read Only", read_only=True),
    }
    schema = Schema(fields=fields)
    assert schema.required == ["required_field"]

def test_schema_init_handles_empty_fields():
    from typesystem import Schema, Field
    schema = Schema(fields={})
    assert schema.fields == {}
    assert schema.required == []
```


# LLM-generated content at query #7
#--------------------------

```python
def test_validate_skips_error_when_child_is_valid():
    class MockField:
        def __init__(self, value, error=None):
            self.value = value
            self.error = error
            self.read_only = False
        def validate_or_error(self, item):
            return self.value, self.error
        def has_default(self):
            return False
        def get_default_value(self):
            return None

    class MockError:
        def messages(self, add_prefix):
            return []

    valid_child = MockField(value="valid_data", error=None)
    schema = Schema(fields={"test_key": valid_child})
    
    result = schema.validate({"test_key": "input_data"})
    
    assert result["test_key"] == "valid_data"
```


# LLM-generated content at query #8
#--------------------------

```python
def test_validate_child_schema_with_error():
    class MockField:
        def __init__(self, error_msg=None):
            self.read_only = False
            self.error_msg = error_msg
        def has_default(self):
            return False
        def validate_or_error(self, value):
            if self.error_msg:
                class MockError:
                    def messages(self, add_prefix):
                        return [f"{add_prefix}: {self.error_msg}"]
                return None, MockError()
            return value, None

    class MockValidationError(Exception):
        def __init__(self, messages):
            self.messages = messages

    class MockSchema(Schema):
        def __init__(self, fields):
            self.fields = fields
            self.required = []
            self.allow_null = False
        def validation_error(self, code):
            return lambda msg: MockValidationError([msg])
        def get_error_text(self, key):
            return self.errors[key]

    error_field = MockField(error_msg="invalid value")
    schema = MockSchema(fields={"test_key": error_field})
    
    # To ensure line 37 (if not error) evaluates to False, 
    # we must trigger the 'else' block by providing an error.
    # The assertion checks that the validation error is raised containing the error message.
    with pytest.raises(MockValidationError) as excinfo:
        schema.validate({"test_key": "some_value"})
    
    assert "test_key: invalid value" in excinfo.value.messages
```


# LLM-generated content at query #9
#--------------------------

```python
def test_schema_validate_skips_read_only_fields():
    class MockField:
        def __init__(self, read_only=False, has_default=False, default_value=None):
            self.read_only = read_only
            self._has_default = has_default
            self._default_value = default_value
        def has_default(self):
            return self._has_default
        def get_default_value(self):
            return self._default_value
        def validate_or_error(self, value):
            return value, None
        def serialize(self, value):
            return value

    read_only_field = MockField(read_only=True)
    writable_field = MockField(read_only=False)
    
    schema = Schema(fields={
        "readonly_key": read_only_field,
        "writable_key": writable_field
    })
    
    input_data = {"readonly_key": "old_value", "writable_key": "new_value"}
    result = schema.validate(input_data)
    
    assert "writable_key" in result
    assert "readonly_key" not in result
    assert result["writable_key"] == "new_value"
```


# LLM-generated content at query #10
#--------------------------

```python
def test_validate_raises_error_when_value_is_none_and_allow_null_is_false():
    definitions = {"target_key": type('Mock', (), {'validate': lambda self, x: x})()}
    reference = Reference(to="target_key", definitions=definitions, allow_null=False)
    
    with pytest.raises(Exception) as excinfo:
        reference.validate(None)
    assert "May not be null." in str(excinfo.value)

def test_validate_returns_none_when_value_is_none_and_allow_null_is_true():
    definitions = {"target_key": type('Mock', (), {'validate': lambda self, x: x})()}
    reference = Reference(to="target_key", definitions=definitions, allow_null=True)
    
    assert reference.validate(None) is None

def test_validate_calls_target_validate_with_correct_value():
    class MockTarget:
        def validate(self, value):
            return f"validated_{value}"

    definitions = {"target_key": MockTarget()}
    reference = Reference(to="target_key", definitions=definitions)
    
    assert reference.validate("some_value") == "validated_some_value"

def test_validate_uses_correct_definition_lookup():
    class MockTarget:
        def validate(self, value):
            return value

    definitions = {"key_a": MockTarget(), "key_b": MockTarget()}
    reference_a = Reference(to="key_a", definitions=definitions)
    reference_b = Reference(to="key_b", definitions=definitions)
    
    assert reference_a.target == definitions["key_a"]
    assert reference_b.target == definitions["key_b"]
```


# LLM-generated content at query #11
#--------------------------

```python
def test_schema_validate_success():
    field_int = StringField() # Assuming StringField/IntField exist as subclasses of Field
    schema = Schema(fields={"name": field_int})
    result = schema.validate({"name": "test"})
    assert result == {"name": "test"}

def test_schema_validate_null_error():
    field_int = StringField()
    schema = Schema(fields={"name": field_int})
    try:
        schema.validate(None)
    except ValidationError as e:
        assert e.messages[0].code == "null"

def test_schema_validate_type_error():
    field_int = StringField()
    schema = Schema(fields={"name": field_int})
    try:
        schema.validate(["not", "a", "dict"])
    except ValidationError as e:
        assert e.messages[0].code == "type"

def test_schema_validate_invalid_key_error():
    field_int = StringField()
    schema = Schema(fields={"name": field_int})
    try:
        schema.validate({123: "value"})
    except ValidationError as e:
        assert e.messages[0].code == "invalid_key"
        assert e.messages[0].index == [123]

def test_schema_validate_required_error():
    field_int = StringField()
    schema = Schema(fields={"name": field_int})
    try:
        schema.validate({})
    except ValidationError as e:
        assert e.messages[0].code == "required"
        assert e.messages[0].index == ["name"]

def test_schema_validate_with_defaults():
    field_int = StringField(default="default_val")
    schema = Schema(fields={"name": field_int})
    result = schema.validate({})
    assert result == {"name": "default_val"}

def test_schema_validate_skips_read_only_on_missing():
    field_read_only = StringField(read_only=True)
    schema = Schema(fields={"name": field_read_only})
    result = schema.validate({})
    assert "name" not in result

def test_schema_validate_nested_error_propagation():
    child_field = StringField()
    child_schema = Schema(fields={"sub": child_field})
    parent_schema = Schema(fields={"parent": child_schema})
    try:
        parent_schema.validate({"parent": {"sub": 123}})
    except ValidationError as e:
        # Assuming error messages are prefixed by key as per code: error.messages(add_prefix=key)
        assert any("parent.sub" in msg.text or "parent" in str(msg) for msg in e.messages)
```


# LLM-generated content at query #12
#--------------------------

```python
def test_validate_child_schema_success():
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

    class MockSchema(Schema):
        def __init__(self, fields):
            self.fields = fields
            self.required = []
            self.allow_null = False
        def validation_error(self, code):
            raise Exception(code)
        def get_error_text(self, code):
            return code

    child_field = MockField()
    schema = MockSchema({"test_key": child_field})
    input_value = {"test_key": "valid_value"}
    
    result = schema.validate(input_value)
    
    assert result["test_key"] == "valid_value"
```


# LLM-generated content at query #13
#--------------------------

```python
def test_schema_validate_skips_read_only_fields():
    class MockField:
        def __init__(self, read_only=False, has_default=False, default_value=None):
            self.read_only = read_only
            self._has_default = has_default
            self._default_value = default_value
        def has_default(self):
            return self._has_default
        def get_default_value(self):
            return self._default_value
        def validate_or_error(self, value):
            return value, None
        def serialize(self, value):
            return value

    read_only_field = MockField(read_only=True)
    writable_field = MockField(read_only=False)
    
    schema = Schema(fields={"readonly": read_only_field, "writable": writable_field})
    
    input_value = {"writable": "correct_value"}
    # If line 28 works, 'readonly' is skipped and 'writable' is processed.
    # 'readonly' is not in input_value, so it won't be in the output.
    result = schema.validate(input_value)
    
    assert "writable" in result
    assert result["writable"] == "correct_value"
    assert "readonly" not in result
```


# LLM-generated content at query #14
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
        def validate(self, val): return val
    
    class MockSchema:
        def __init__(self, fields):
            self.fields = fields
            self.allow_null = False
            self.required = [k for k, v in fields.items() if not (v.read_only or v.has_default())]
        def validation_error(self, key): return Exception(key)
        def get_error_text(self, key): return f"error_{key}"
        def validate(self, value):
            # Simplified internal logic for test execution
            if value is None: raise Exception("null")
            if not isinstance(value, dict): raise Exception("type")
            validated = {}
            errors = []
            for k in value.keys():
                if not isinstance(k, str): errors.append(k)
            for k in self.required:
                if k not in value: errors.append(k)
            for k, field in self.fields.items():
                if field.read_only: continue
                if k not in value:
                    if field.has_default(): validated[k] = field.get_default_value()
                    continue
                val, err = field.validate_or_error(value[k])
                if err: errors.append(err)
                else: validated[k] = val
            if errors: raise Exception(errors)
            return validated

    f1 = MockField(default="default_val")
    f2 = MockField(read_only=True)
    schema = MockSchema({"a": f1, "b": f2})
    result = schema.validate({"a": "actual_val", "b": "ignored_val"})
    assert result == {"a": "actual_val"}

def test_validate_raises_type_error():
    class MockField:
        def __init__(self): self.read_only = False
        def has_default(self): return False
        def validate_or_error(self, val): return val, None
    
    class MockSchema:
        def __init__(self, fields):
            self.fields = fields
            self.allow_null = False
            self.required = []
        def validation_error(self, key): return ValueError(key)
        def validate(self, value):
            if not isinstance(value, dict): raise ValueError("type")
            return {}

    schema = MockSchema({"a": MockField()})
    with pytest.raises(ValueError) as excinfo:
        schema.validate(["not", "a", "dict"])
    assert "type" in str(excinfo.value)

def test_validate_raises_null_error():
    class MockField:
        def __init__(self): self.read_only = False
        def has_default(self): return False
        def validate_or_error(self, val): return val, None

    class MockSchema:
        def __init__(self, fields):
            self.fields = fields
            self.allow_null = False
            self.required = []
        def validation_error(self, key): return ValueError(key)
        def validate(self, value):
            if value is None: raise ValueError("null")
            return {}

    schema = MockSchema({"a": MockField()})
    with pytestly.raises(ValueError) as excinfo:
        schema.validate(None)
    assert "null" in str(excinfo.value)

def test_validate_raises_required_error():
    class MockField:
        def __init__(self): self.read_only = False
        def has_default(self): return False
        def validate_or_error(self, val): return val, None

    class MockSchema:
        def __init__(self, fields):
            self.fields = fields
            self.allow_null = False
            self.required = ["missing_key"]
        def validation_error(self, key): return ValueError(key)
        def get_error_text(self, key): return "required"
        def validate(self, value):
            if "missing_key" not in value:
                raise ValueError("required")
            return {}

    schema = MockSchema({"missing_key": MockField()})
    with pytest.raises(ValueError) as excinfo:
        schema.validate({"other_key": 123})
    assert "required" in str(excinfo.value)

def test_validate_invalid_key_type():
    class MockField:
        def __init__(self): self.read_only = False
        def has_default(self): return False
        def validate_or_error(self, val): return val, None

    class MockSchema:
        def __init__(self, fields):
            self.fields = fields
            self.allow_null = False
            self.required = []
        def validation_error(self, key): return ValueError(key)
        def get_error_text(self, key): return "invalid_key"
        def validate(self, value):
            for k in value.keys():
                if not isinstance(k, str): raise ValueError("invalid_key")
            return {}

    schema = MockSchema({"a": MockField()})
    with pytest.raises(ValueError) as excinfo:
        schema.validate({123: "value"})
    assert "invalid_key" in str(excinfo.value)
```


# LLM-generated content at query #15
#--------------------------

```python
def test_validate_key_not_in_value_with_default_triggering_line_32():
    class MockField:
        def __init__(self, has_default_val=True, default_val="default"):
            self.has_default_val = has_default_val
            self.default_val = default_val
            self.read_only = False
        def has_default(self):
            return self.has_default_val
        def get_default_value(self):
            return self.default_val
        def validate_or_error(self, value):
            return value, None

    fields = {"test_key": MockField(has_default_val=True, default_val="default_val")}
    schema = Schema(fields=fields)
    input_value = {}
    
    result = schema.validate(input_value)
    
    assert result["test_key"] == "default_val"
```


# LLM-generated content at query #16
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
            self.required = [k for k, v in fields.items() if not (v.read_only or v.has_default())]
        def validation_error(self, code): return Exception(code)
        def get_error_text(self, code): return code
        def validate(self, value):
            if value is None: raise Exception("null")
            if not isinstance(value, dict): raise Exception("type")
            validated = {}
            errors = []
            for k in value.keys():
                if not isinstance(k, str): errors.append("invalid_key")
            for k in self.required:
                if k not in value: errors.append("required")
            for k, field in self.fields.items():
                if field.read_only: continue
                if k not in value:
                    if field.has_default(): validated[k] = field.get_default_value()
                    continue
                val, err = field.validate_or_error(value[k])
                if err: errors.append(err)
                else: validated[k] = val
            if errors: raise Exception(errors)
            return validated

    field_str = MockField()
    field_int = MockField(default=10)
    schema = MockSchema({"name": field_str, "age": field_int})
    result = schema.validate({"name": "Alice"})
    assert result == {"name": "Alice", "age": 10}

def test_validate_error_null():
    class MockField:
        def __init__(self, allow_null=False): self.allow_null = allow_null
        def has_default(self): return False
        def read_only(self): return False
    
    class MockSchema:
        def __init__(self, fields):
            self.fields = fields
            self.allow_null = False
            self.required = []
        def validation_error(self, code): return ValueError(code)
        def validate(self, value):
            if value is None: raise ValueError("null")
            return {}

    schema = MockSchema({})
    with pytest.raises(ValueError, match="null"):
        schema.validate(None)

def test_validate_error_type():
    class MockField:
        def __init__(self): self.read_only = False
        def has_default(self): return False
    
    class MockSchema:
        def __init__(self, fields):
            self.fields = fields
            self.allow_null = False
            self.required = []
        def validation_error(self, code): return ValueError(code)
        def validate(self, value):
            if not isinstance(value, dict): raise ValueError("type")
            return {}

    schema = MockSchema({})
    with pytest.raises(ValueError, match="type"):
        schema.validate(["not", "a", "dict"])

def test_validate_error_invalid_key():
    class MockField:
        def __init__(self): self.read_only = False
        def has_default(self): return False
    
    class MockSchema:
        def __init__(self, fields):
            self.fields = fields
            self.allow_null = False
            self.required = []
        def validation_error(self, code): return ValueError(code)
        def get_error_text(self, code): return code
        def validate(self, value):
            for k in value.keys():
                if not isinstance(k, str): raise ValueError("invalid_key")
            return {}

    schema = MockSchema({})
    with pytest.raises(ValueError, match="invalid_key"):
        schema.validate({123: "value"})

def test_validate_error_required():
    class MockField:
        def __init__(self, read_only=False, default=None):
            self.read_only = read_only
            self.default = default
        def has_default(self): return self.default is not None
    
    class MockSchema:
        def __init__(self, fields):
            self.fields = fields
            self.allow_null = False
            self.required = ["id"]
        def validation_error(self, code): return ValueError(code)
        def get_error_text(self, code): return code
        def validate(self, value):
            for k in self.required:
                if k not in value: raise ValueError("required")
            return {}

    schema = MockSchema({"id": MockField()})
    with pytest.raises(ValueError, match="required"):
        schema.validate({"name": "Alice"})
```


# LLM-generated content at query #17
#--------------------------

```python
def test_validate_child_schema_success():
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

    class MockSchema(Schema):
        def __init__(self, fields):
            self.fields = fields
            self.required = []
            self.allow_null = False
        def validation_error(self, code):
            raise Exception(code)
        def get_error_text(self, code):
            return ""

    child_field = MockField()
    parent_schema = MockSchema({"key": child_field})
    input_value = {"key": "valid_value"}
    
    result = parent_schema.validate(input_value)
    
    assert result["key"] == "valid_value"
```


# LLM-generated content at query #18
#--------------------------

```python
def test_validate_success_on_child_field():
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

    child_field = MockField()
    schema = Schema(fields={"name": child_field})
    input_value = {"name": "test_value"}
    
    result = schema.validate(input_value)
    
    assert result["name"] == "test_value"
```


# LLM-generated content at query #19
#--------------------------

```python
def test_validate_skips_error_branch_when_child_validation_succeeds():
    class MockField:
        def __init__(self, value, error=None):
            self.value = value
            self.error = error
            self.read_only = False
        def validate_or_error(self, item):
            return self.value, self.error
        def has_default(self):
            return False

    child_field = MockField(value="success", error=None)
    schema = Schema(fields={"name": child_field})
    input_data = {"name": "test_value"}
    
    result = schema.validate(input_data)
    
    assert result == {"name": "success"}
```


# LLM-generated content at query #20
#--------------------------

```python
def test_validate_error_at_line_37():
    class MockField:
        def __init__(self, read_only=False, has_default_val=False):
            self.read_only = read_only
            self.has_default_val = has_default_val
        def has_default(self):
            return self.has_default_val
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
            return MockValidationError([code])
        def get_error_text(self, code):
            return self.errors[code]

    field = MockField()
    schema = MockSchema(fields={"test_key": field})
    input_value = {"test_key": "some_value"}
    
    with Exception as e:
        try:
            schema.validate(input_value)
        except Exception as caught_error:
            e = caught_error
    
    assert "test_key: error" in e.messages
```


# LLM-generated content at query #21
#--------------------------

```python
def test_validate_success_with_defaults():
    class MockField:
        def __init__(self, read_only=False, default=None):
            self.read_only = read_only
            self.default = default
        def has_default(self): return self.default is not None
        def get_default_value(self): return self.default
        def validate_or_error(self, value): return value, None
        def validate(self, value): return value

    field_str = MockField(default="hello")
    field_int = MockField()
    schema = Schema(fields={"name": field_str, "age": field_int})
    
    input_data = {"name": "world", "age": 25}
    result = schema.validate(input_data)
    
    assert result == {"name": "world", "age": 25}
    assert "name" not in schema.required
    assert "age" in schema.required

def test_validate_success_with_missing_optional_and_defaults():
    class MockField:
        def __init__(self, default=None):
            self.read_only = False
            self.default = default
        def has_default(self): return self.default is not None
        def get_default_value(self): return self.default
        def validate_or_error(self, value): return value, None
        def validate(self, value): return value

    schema = Schema(fields={"opt": MockField(default="default_val")})
    result = schema.validate({})
    assert result == {"opt": "default_val"}

def test_validate_error_null_not_allowed():
    class MockField:
        def __init__(self, allow_null=False):
            self.read_only = False
            self.allow_null = allow_null
        def has_default(self): return False
        def validate_or_error(self, value): return value, None
        def validate(self, value): return value

    schema = Schema(fields={"data": MockField(allow_null=False)})
    
    class MockValidationError(Exception):
        def __init__(self, code): self.code = code
    
    # We assume validation_error returns an exception that we can catch
    # For the sake of this unit test, we simulate the behavior via a mock-like setup
    # Since we can't define custom classes/logic, we assume the environment 
    # provides the necessary error structure.
    
    # Assuming the error message 'null' is raised
    with pytest.raises(ValidationError) as excinfo:
        schema.validate(None)
    assert excinfo.value.messages[0].code == "null"

def test_validate_error_type_mismatch():
    class MockField:
        def __init__(self):
            self.read_only = False
        def has_default(self): return False
        def validate_or_error(self, value): return value, None
        def validate(self, value): return value

    schema = Schema(fields={"data": MockField()})
    
    with pytest.raises(ValidationError) as excinfo:
        schema.validate(["not", "a", "dict"])
    assert excinfo.value.messages[0].code == "type"

def test_validate_error_invalid_key_type():
    class MockField:
        def __init__(self):
            self.read_only = False
        def has_default(self): return False
        def validate_or_error(self, value): return value, None
        def validate(self, value): return value

    schema = Schema(fields={})
    
    with pytest_raises(ValidationError) as excinfo:
        schema.validate({123: "value"})
    assert excinfo.value.messages[0].code == "invalid_key"
    assert excinfo.value.messages[0].index == [123]

def test_validate_error_required_field_missing():
    class MockField:
        def __init__(self):
            self.read_only = False
        def has_default(self): return False
        def validate_or_error(self, value): return value, None
        def validate(self, value): return value

    schema = Schema(fields={"required_key": MockField()})
    
    with pytest.raises(ValidationError) as excinfo:
        schema.validate({"other_key": "value"})
    assert excinfo.value.messages[0].code == "required"
    assert excinfo.value.messages[0].index == ["required_key"]

def test_validate_error_child_field_validation_failure():
    class MockField:
        def __init__(self):
            self.read_only = False
        def has_default(self): return False
        def validate_or_error(self, value): 
            # Simulate a child validation error
            class MockError:
                def messages(self, add_prefix):
                    return [Message(text="err", code="child_err", index=[add_prefix])]
            return None, MockError()
        def validate(self, value): return value

    schema = Schema(fields={"child": MockField()})
    
    with pytest.raises(ValidationError) as excinfo:
        schema.validate({"child": "bad_value"})
    assert excinfo.value.messages[0].code == "child_err"
    assert excinfo.value.messages[0].index == ["child"]
```


# LLM-generated content at query #22
#--------------------------

```python
def test_schema_validate_skips_read_only_fields():
    class MockField:
        def __init__(self, read_only=False, has_default=False, default_value=None):
            self.read_only = read_only
            self._has_default = has_default
            self._default_value = default_value
        def has_default(self):
            return self._has_default
        def get_default_value(self):
            return self._default_value
        def validate_or_error(self, value):
            return value, None
        def serialize(self, value):
            return value

    read_only_field = MockField(read_only=True)
    writable_field = MockField(read_only=False)
    
    schema = Schema(fields={"readonly": read_only_field, "writable": writable_field})
    
    input_value = {"writable": "data"}
    # If line 28 works, 'readonly' is skipped and doesn't look for it in input_value.
    # If it didn't skip, it would attempt to validate 'readonly' which is missing in input_value.
    # Since 'readonly' is read_only, it is not in the 'required' list of Schema __init__.
    # The result should contain only the 'writable' key.
    result = schema.validate(input_value)
    
    assert "writable" in result
    assert "readonly" not in result
    assert result["writable"] == "data"
```


# LLM-generated content at query #23
#--------------------------

```python
def test_schema_validate_key_missing_with_default_value_triggers_line_32():
    class MockField:
        def __init__(self, default_value=None, has_default=False, read_only=False):
            self.default_value = default_value
            self.has_default_flag = has_default
            self.read_only = read_only
        def has_default(self):
            return self.has_default_flag
        def get_default_value(self):
            return self.default_value
        def validate_or_error(self, value):
            return value, None

    mock_field = MockField(default_value="default", has_default=True)
    schema = Schema(fields={"test_key": mock_field})
    
    # value is {} so 'test_key' is not in value. 
    # child_schema.has_default() will be True.
    result = schema.validate({})
    
    assert result["test_key"] == "default"
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_schema_constructor_initializes_fields_and_required_correctly():
    class StringField(Field):
        def validate(self, value):
            return str(value)

    class IntField(Field):
        def validate(self, value):
            return int(value)

    class ReadOnlyField(Field):
        def validate(self, value):
            return value

    fields = {
        "required_string": StringField(),
        "optional_string": StringField(default="default"),
        "required_int": IntField(),
        "readonly_int": ReadallyIntField := ReadOnlyField(read_only=True),
        "nullable_string": StringField(allow_null=True, default=None),
    }
    
    schema = Schema(fields=fields)
    
    assert schema.fields == fields
    assert "required_string" in schema.required
    assert "optional_string" not in schema.required
    assert "required_int" in schema.required
    assert "readonly_int" not in schema.required
    assert "nullable_string" not in schema.required

def test_schema_constructor_with_kwargs():
    class StringField(Field):
        def validate(self, value):
            return str(value)

    fields = {"name": StringField()}
    schema = Schema(fields=fields, title="UserSchema", description="Schema for users", allow_null=True)
    
    assert schema.fields == fields
    assert schema.title == "UserSchema"
    assert schema.description == "Schema for users"
    assert schema.allow_null is True
```


# LLM-generated content at query #2
#--------------------------

```python
def test_validate_null_error():
    definitions = {"item": type('Mock', (), {'validate': lambda x: x})()}
    ref = Reference(to="item", definitions=definitions, allow_null=False)
    with pytest.raises(Exception) as excinfo:
        ref.validate(None)
    assert "May not be null." in str(excinfo.value)

def test_validate_null_allowed():
    definitions = {"item": type('Mock', (), {'validate': lambda x: x})()}
    ref = Reference(to="item", definitions=definitions, allow_null=True)
    assert ref.validate(None) is None

def test_validate_success_delegation():
    mock_target = type('Mock', (), {'validate': lambda self, x: x + 1})()
    definitions = {"item": mock_target}
    ref = Reference(to="item", definitions=definitions)
    assert ref.validate(10) == 11

def test_validate_target_lookup():
    mock_target = type('Mock', (), {'validate': lambda self, x: x})()
    definitions = {"item": mock_target}
    ref = Reference(to="item", definitions=definitions)
    assert ref.target == mock_target
```


# LLM-generated content at query #3
#--------------------------

```python
def test_schema_validate_success():
    field_int = Field(type="integer")
    field_str = Field(type="string")
    schema = Schema(fields={"age": field_int, "name": field_str})
    input_data = {"age": 25, "name": "John"}
    result = schema.validate(input_data)
    assert result == {"age": 25, "name": "John"}

def test_schema_validate_null_error():
    field_int = Field(type="integer", allow_null=False)
    schema = Schema(fields={"age": field_int})
    with assert_raises(ValidationError) as cm:
        schema.validate(None)
    assert cm.exception.messages[0].code == "null"

def test_schema_validate_null_success():
    field_int = Field(type="integer", allow_null=True)
    schema = Schema(fields={"age": field_int})
    result = schema.validate(None)
    assert result is None

def test_schema_validate_type_error():
    field_int = Field(type="integer")
    schema = Schema(fields={"age": field_int})
    with assert_raises(ValidationError) as cm:
        schema.validate(["not", "a", "dict"])
    assert cm.exception.messages[0].code == "type"

def test_schema_validate_invalid_key_error():
    field_int = Field(type="integer")
    schema = Schema(fields={"age": field_int})
    input_data = {123: "invalid key type"}
    with assert_raises(ValidationError) as cm:
        schema.validate(input_data)
    assert cm.exception.messages[0].code == "invalid_key"
    assert cm.exception.messages[0].index == [123]

def test_schema_validate_required_error():
    field_int = Field(type="integer")
    schema = Schema(fields={"age": field_int})
    input_data = {"name": "John"}
    with assert_raises(ValidationError) as cm:
        schema.validate(input_data)
    assert cm.exception.messages[0].code == "required"
    assert cm.exception.messages[0].index == ["age"]

def test_schema_validate_default_value_injection():
    field_int = Field(type="integer", default=10)
    schema = Schema(fields={"age": field_int})
    input_data = {}
    result = schema.validate(input_data)
    assert result["age"] == 10

def test_schema_validate_readonly_skips_logic():
    field_int = Field(type="integer", read_only=True)
    schema = Schema(fields={"age": field_int})
    input_data = {"age": 25}
    result = schema.validate(input_data)
    assert "age" not in result
```


# LLM-generated content at query #4
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
        def validate(self, val): return val

    class MockSchema(Schema):
        def validation_error(self, key): return ValidationError(messages=[])
        def get_error_text(self, key): return "error"

    field_int = MockField(default=10)
    field_str = MockField(read_only=True)
    schema = MockSchema(fields={"age": field_int, "name": field_str})
    
    input_data = {"age": 25, "name": "John"}
    result = schema.validate(input_data)
    
    assert result == {"age": 25}
    assert "age" in schema.required
    assert "name" not in schema.required

def test_validate_error_type_not_object():
    class MockField:
        def __init__(self): self.read_only = False
        def has_default(self): return False
        def validate_or_error(self, val): return val, None
        def validate(self, val): return val

    class MockSchema(Schema):
        def validation_error(self, key): return ValidationError(messages=[])
        def get_error_text(self, key): return "type error"

    schema = MockSchema(fields={"test": MockField()})
    
    with Exception as e:
        schema.validate([1, 2, 3])
        assert isinstance(e, ValidationError)

def test_validate_error_null_not_allowed():
    class MockField:
        def __init__(self): self.read_only = False
        def has_default(self): return False
        def validate_or_error(self, val): return val, None
        def validate(self, val): return val

    class MockSchema(Schema):
        allow_null = False
        def validation_error(self, key): return ValidationError(messages=[])
        def get_error_text(self, key): return "null error"

    schema = MockSchema(fields={"test": MockField()})
    
    with Exception as e:
        schema.validate(None)
        assert isinstance(e, ValidationError)

def test_validate_error_invalid_key_type():
    class MockField:
        def __init__(self): self.read_only = False
        def has_default(self): return False
        def validate_or_error(self, val): return val, None
        def validate(self, val): return val

    class MockMessage:
        def __init__(self, text, code, index):
            self.text = text
            self.code = code
            self.index = index
    
    class MockValidationError(Exception):
        def __init__(self, messages): self.messages = messages

    class MockSchema(Schema):
        def validation_error(self, key): return MockValidationError([])
        def get_error_text(self, key): return "invalid key"
        def messages(self, add_prefix): return []

    schema = MockSchema(fields={})
    input_data = {1: "value"}
    
    with Exception as e:
        schema.validate(input_data)
        assert isinstance(e, MockValidationError)
        assert e.messages[0].code == "invalid_key"

def test_validate_error_required_field_missing():
    class MockField:
        def __init__(self): self.read_only = False
        def has_default(self): return False
        def validate_or_error(self, val): return val, None
        def validate(self, val): return val

    class MockMessage:
        def __init__(self, text, code, index):
            self.text = text
            self.code = code
            self.index = index

    class MockValidationError(Exception):
        def __init__(self, messages): self.messages = messages

    class MockSchema(Schema):
        def validation_error(self, key): return MockValidationError([])
        def get_error_text(self, key): return "required"

    schema = MockSchema(fields={"must_exist": MockField()})
    
    with Exception as e:
        schema.validate({"other": 123})
        assert isinstance(e, MockValidationError)
        assert e.messages[0].code == "required"
        assert e.messages[0].index == ["must_exist"]

def test_validate_child_field_error_propagation():
    class MockField:
        def __init__(self): self.read_only = False
        def has_default(self): return False
        def validate_or_error(self, val): return None, MockValidationError(messages=["error"])
        def validate(self, val): return val
        def serialize(self, val): return val

    class MockMessage:
        def __init__(self, text, code, index): pass

    class MockValidationError(Exception):
        def __init__(self, messages): self.messages = messages
        def messages(self, add_prefix): return [MockMessage("err", "err", [])]

    class MockSchema(Schema):
        def validation_error(self, key): return MockValidationError([])
        def get_error_text(self, key): return "error"

    schema = MockSchema(fields={"child": MockField()})
    
    with Exception as e:
        schema.validate({"child": "some_val"})
        assert isinstance(e, MockValidationError)
```


# LLM-generated content at query #5
#--------------------------

```python
def test_schema_validate_type_error():
    field_int = StringField()
    schema = Schema(fields={"name": field_int})
    with pytest.raises(ValidationError) as excinfo:
        schema.validate(["not", "a", "dict"])
    assert excinfo.value.messages[0].code == "type"

def test_schema_validate_null_error():
    field_int = StringField()
    schema = Schema(fields={"name": field_int})
    with pytest.raises(ValidationError) as excinfo:
        schema.validate(None)
    assert excinfo.value.messages[0].code == "null"

def test_schema_validate_null_allowed():
    field_int = StringField(allow_null=True)
    schema = Schema(fields={"name": field_int})
    assert schema.validate(None) is None

def test_schema_validate_invalid_key_type():
    field_int = StringField()
    schema = Schema(fields={"name": field_int})
    with pytest.raises(ValidationError) as excinfo:
        schema.validate({123: "value"})
    assert excinfo.value.messages[0].code == "invalid_key"
    assert excinfo.value.messages[0].index == [123]

def test_schema_validate_required_field_missing():
    field_int = StringField()
    schema = Schema(fields={"name": field_int})
    with pytest.append(schema.required == ["name"])
    with pytest.raises(ValidationError) as excinfo:
        schema.validate({"age": 30})
    assert excinfo.value.messages[0].code == "required"
    assert excinfo.value.messages[0].index == ["name"]

def test_schema_validate_success_with_defaults():
    field_int = IntField(default=10)
    schema = Schema(fields={"age": field_int})
    result = schema.validate({"name": "test"})
    assert result == {"age": 10}

def test_schema_validate_success_with_provided_values():
    field_str = StringField()
    field_int = IntField()
    schema = Schema(fields={"name": field_str, "age": field_int})
    result = schema.validate({"name": "John", "age": 25})
    assert result == {"name": "John", "age": 25}

def test_schema_validate_nested_error_propagation():
    child_schema = Schema(fields={"sub": StringField()})
    parent_schema = Schema(fields={"child": child_schema})
    with pytest.raises(ValidationError) as excinfo:
        parent_schema.validate({"child": {"sub": 123}})
    assert any(m.code == "type" for m in excinfo.value.messages)
```


# LLM-generated content at query #6
#--------------------------

```python
def test_validate_success():
    field1 = StringField()
    field2 = IntField(default=10)
    schema = Schema(fields={"name": field1, "age": field2})
    input_data = {"name": "John", "age": 25}
    result = schema.validate(input_data)
    assert result == {"name": "John", "age": 25}

def test_validate_success_with_defaults():
    field1 = StringField()
    field2 = IntField(default=10)
    schema = Schema(fields={"name": field1, "age": field2})
    input_data = {"name": "John"}
    result = schema.validate(input_data)
    assert result == {"name": "John", "age": 10}

def test_validate_error_type_not_object():
    schema = Schema(fields={})
    with assert_raises(ValidationError) as context:
        schema.validate("not a dict")
    assert context.exception.messages[0].code == "type"

def test_validate_error_null_not_allowed():
    schema = Schema(fields={}, allow_null=False)
    with assert_raises(ValidationError) as context:
        schema.validate(None)
    assert context.exception.messages[0].code == "null"

def test_validate_error_null_allowed():
    schema = Schema(fields={}, allow_null=True)
    result = schema.validate(None)
    assert result is None

def test_validate_error_invalid_key_type():
    schema = Schema(fields={})
    input_data = {123: "value"}
    with assert_raises(ValidationError) as context:
        schema.validate(input_data)
    assert context.exception.messages[0].code == "invalid_key"
    assert context.exception.messages[0].index == [123]

def test_validate_error_required_field_missing():
    field1 = StringField()
    schema = Schema(fields={"name": field1})
    input_data = {}
    with assert_raises(ValidationError) as context:
        schema.validate(input_data)
    assert context.exception.messages[0].code == "required"
    assert context.exception.messages[0].index == ["name"]

def test_validate_error_nested_validation_failure():
    child_field = StringField()
    parent_schema = Schema(fields={"child": child_field})
    schema = Schema(fields={"parent": parent_schema})
    input_data = {"parent": {"child": 123}}
    with assert_raises(ValidationError) as context:
        schema.validate(input_data)
    assert any("child" in msg.text or msg.code == "child" for msg in context.exception.messages)

def test_validate_ignores_read_only_fields_in_required_check():
    field1 = StringField(read_only=True)
    schema = Schema(fields={"name": field1})
    input_data = {}
    result = schema.validate(input_data)
    assert "name" not in result
```


# LLM-generated content at query #7
#--------------------------

```python
def test_schema_validate_skips_read_only_fields():
    class MockField:
        def __init__(self, read_only=False, has_default=False, default_value=None):
            self.read_only = read_only
            self._has_default = has_default
            self._default_value = default_value
        def has_default(self):
            return self._has_default
        def get_default_value(self):
            return self._default_value
        def validate_or_error(self, value):
            return value, None
        def serialize(self, value):
            return value

    read_only_field = MockField(read_only=True)
    writable_field = MockField(read_only=False)
    
    schema = Schema(fields={"readonly": read_only_field, "writable": writable_field})
    
    input_value = {"readonly": "should_be_ignored", "writable": "should_be_kept"}
    result = schema.validate(input_value)
    
    assert "writable" in result
    assert "readonly" not in result
```


# LLM-generated content at query #8
#--------------------------

```python
def test_validate_skips_missing_key_with_no_default():
    class MockField:
        def __init__(self, has_default_val=False):
            self.has_default = lambda: has_default_val
            self.read_only = False
        def validate_or_error(self, value):
            return value, None
        def get_default_value(self):
            return "default"

    schema = Schema(fields={"test_key": MockField(has_default_val=False)})
    result = schema.validate({"other_key": 123})
    assert "test_key" not in result
```


