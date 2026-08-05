####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Schema_serialize():
    # Mock Fields
    field1 = MagicMock(spec=Field)
    field1.serialize.return_value = "value1"
    
    field2 = MagicMock(spec=Field)
    field2.serialize.return_value = 123
    
    field3 = MagicMock(spec=Field)
    field3.serialize.return_value = None

    fields = {
        "key1": field1,
        "key2": field2,
        "key3": field3
    }
    
    schema = Schema(fields=fields)

    # Test 1: Serialize None
    assert schema.serialize(None) is None

    # Test 2: Serialize Dictionary (Mapping)
    input_dict = {"key1": "original1", "key2": "original2", "key3": "original3"}
    expected_output = {"key1": "value1", "key2": 123, "key3": None}
    assert schema.serialize(input_dict) == expected_output
    
    field1.serialize.assert_called_with("original1")
    field2.serialize.assert_called_with("original2")
    field3.serialize.assert_called_with("original3")

    # Test 3: Serialize Object (Attribute access)
    class MockObj:
        def __init__(self, k1, k2, k3):
            self.key1 = k1
            self.key2 = k2
            self.key3 = k3

    input_obj = MockObj("attr1", "attr2", "attr3")
    expected_output_obj = {"key1": "value1", "key2": 123, "key3": None}
    assert schema.serialize(input_obj) == expected_output_obj

    # Test 4: Serialize dictionary with missing keys (should skip via KeyError)
    input_incomplete = {"key1": "only_one"}
    expected_output_incomplete = {"key1": "value1"}
    assert schema.serialize(input_incomplete) == expected_output_incomplete

    # Test 5: Serialize object with missing attributes (should skip via AttributeError)
    class IncompleteObj:
        def __init__(self, k1):
            self.key1 = k1
    
    input_incomplete_obj = IncompleteObj("attr_only")
    assert schema.serialize(input_incomplete_obj) == {"key1": "value1"}
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Reference_validate():
    # Mock definitions and a target field
    mock_target_field = MagicMock(spec=Field)
    definitions = Definitions({"my_ref": mock_target_field})
    
    # 1. Test valid value passing through to target
    ref = Reference(to="my_ref", definitions=definitions, allow_null=False)
    mock_target_field.validate.return_value = "valid_data"
    assert ref.validate({"key": "value"}) == "valid_data"
    mock_target_field.validate.assert_called_with({"key": "value"})

    # 2. Test null value when allow_null is True
    ref_allow_null = Reference(to="my_ref", definitions=definitions, allow_null=True)
    assert ref_allow_null.validate(None) is None

    # 3. Test null value when allow_null is False (should raise ValidationError)
    ref_no_null = Reference(to="my_ref", definitions=definitions, allow_null=False)
    with pytest.raises(ValidationError) as excinfo:
        ref_no_null.validate(None)
    # Check if the error message corresponds to the "null" key in Reference.errors
    assert any("May not be null" in str(msg) for msg in excinfo.value.messages)

    # 4. Test validation error propagated from target field
    mock_target_field.validate.side_effect = ValidationError(messages=[Message(text="Target Error", code="target_err")])
    with pytest.raises(ValidationError) as excinfo:
        ref.validate({"invalid": "data"})
    assert len(excinfo.value.messages) == 1
    assert excinfo.value.messages[0].code == "target_err"
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest

def test_Schema_serialize():
    # Mock Field class to simulate behavior of typesystem.fields.Field
    class MockField:
        def __init__(self, value=None, serialize_val=None):
            self.value = value
            self.serialize_val = serialize_val

        def serialize(self, value):
            return self.serialize_val

    # Setup fields for Schema
    fields = {
        "name": MockField(value="John", serialize_val="John"),
        "age": MockField(value=30, serialize_val=30),
        "extra": MockField(value="hidden", serialize_val="hidden"),
    }
    
    schema = Schema(fields=fields)

    # Test Case 1: Serialize a dictionary
    input_dict = {"name": "John", "age": 30, "extra": "hidden"}
    expected_output = {"name": "John", "age": 30, "extra": "hidden"}
    assert schema.serialize(input_dict) == expected_output

    # Test Case 2: Serialize an object with attributes (AttributeError fallback)
    class MockObject:
        def __init__(self):
            self.name = "John"
            self.age = 30
            self.extra = "hidden"

    input_obj = MockObject()
    assert schema.serialize(input_obj) == expected_output

    # Test Case 3: Serialize when keys are missing in the input (KeyError/AttributeError fallback)
    input_partial = {"name": "John"}
    # Should only contain keys that were present in the input and exist in fields
    assert schema.serialize(input_partial) == {"name": "John"}

    # Test Case 4: Serialize None
    assert schema.serialize(None) is None

    # Test Case 5: Ensure it handles different types of return values from field.serialize
    complex_fields = {
        "nested": MockField(value={"a": 1}, serialize_val={"a": 1}),
        "simple": MockField(value=1, serialize_val="one")
    }
    schema_complex = Schema(fields=complex_fields)
    input_complex = {"nested": {"a": 1}, "simple": 1}
    expected_complex = {"nested": {"a": 1}, "simple": "one"}
    assert schema_complex.serialize(input_complex) == expected_complex
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest

def test_Schema_serialize():
    # Mock Field classes to avoid complex dependencies
    class MockField(Field):
        def __init__(self, default=None, read_only=False):
            self.default = default
            self.read_only = read_only
            self._has_default = default is not None

        def has_default(self):
            return self._has_default

        def get_default_value(self):
            return self.default

        def serialize(self, value):
            if isinstance(value, MockField):
                # Handle nested schemas in serialization
                return value.serialize(value)
            return value

    # Setup fields for the schema
    fields = {
        "name": MockField(),
        "age": MockField(),
        "read_only_field": MockField(read_only=True),
        "nested": MockField()
    }
    
    schema = Schema(fields)

    # Case 1: Serialize None
    assert schema.serialize(None) is None

    # Case 2: Serialize dictionary input
    input_dict = {
        "name": "John",
        "age": 30,
        "read_only_field": "don't change me",
        "nested": {"inner": "value"}
    }
    expected_dict = {
        "name": "John",
        "age": 30,
        "read_only_field": "don't change me",
        "nested": {"inner": "value"}
    }
    assert schema.serialize(input_dict) == expected_dict

    # Case 3: Serialize object input (using getattr)
    class MockObject:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    input_obj = MockObject(name="Jane", age=25, read_only_field="static")
    expected_obj_output = {
        "name": "Jane",
        "age": 25,
        "read_only_field": "static"
    }
    # Note: 'nested' won't be in expected because it's missing from input_obj
    result = schema.serialize(input_obj)
    assert result["name"] == "Jane"
    assert result["age"] == 25
    assert result["read_only_field"] == "static"

    # Case 4: Missing keys in input (should be omitted from output based on serialize logic)
    input_incomplete = {"name": "Only Name"}
    result_incomplete = schema.serialize(input_incomplete)
    assert "name" in result_incomplete
    assert "age" not in result_incomplete

    # Case 5: Verify that nested serialization is called
    nested_field = MockField()
    inner_schema = Schema({"key": MockField()})
    # We manually inject a specialized field to test recursive-like behavior if needed,
    # but based on the provided code, serialize simply calls child.serialize(value)
    fields["nested"] = MockField() 
    # The logic in serialize: ret[key] = field.serialize(value)
    # If value is a dict, it returns the dict as is (via MockField implementation above)
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Schema():
    # Mocking Fields
    field_required = MagicMock(spec=Field)
    field_required.read_only = False
    field_required.has_default.return_value = False

    field_optional = MagicMock(spec=Field)
    field_optional.read_only = False
    field_optional.has_default.return_value = True

    field_readonly = MagicMock(spec=Field)
    field_readonly.read_only = True
    field_readonly.has_default.return_value = False

    fields = {
        "req": field_required,
        "opt": field_optional,
        "ro": field_readonly
    }

    # Test Initialization
    schema = Schema(fields=fields)

    # Assertions for required list logic
    # 'req' is required because it's not read_only and has no default
    # 'opt' is NOT required because it has a default
    # 'ro' is NOT required because it is read_only
    assert "req" in schema.required
    assert "opt" not in schema.required
    assert "ro" not in schema.required
    assert len(schema.required) == 1

    # Test with empty fields
    empty_schema = Schema(fields={})
    assert empty_schema.required == []

    # Test kwargs passing to super().__init__ (e.g., allow_null)
    schema_with_kwargs = Schema(fields=fields, allow_null=True)
    assert schema_with_kwargs.allow_null is True
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Schema_validate():
    # Setup mock fields
    field_str = MagicMock(spec=Field)
    field_str.read_only = False
    field_str.has_default.return_value = False
    field_str.validate_or_error.side_effect = lambda x: (x, None)
    field_str.serialize.side_effect = lambda x: x

    field_int = MagicMock(spec=Field)
    field_int.read_only = False
    field_int.has_default.return_value = True
    field_int.get_default_value.return_value = 10
    field_int.validate_or_error.side_effect = lambda x: (x, None)
    field_int.serialize.side_effect = lambda x: x

    field_readonly = MagicMock(spec=Field)
    field_readonly.read_only = True
    field_readonly.has_default.return_value = False
    field_readonly.validate_or_error.side_effect = lambda x: (x, None)
    field_readonly.serialize.side_effect = lambda x: x

    fields = {
        "name": field_str,
        "age": field_int,
        "meta": field_readonly,
    }

    schema = Schema(fields=fields)

    # 1. Test valid input with all features (required, default, and read_only)
    input_data = {"name": "John", "age": 25, "meta": "some_meta"}
    # Note: meta is read_only so it shouldn't be processed in the 'properties' loop logic for validation results
    result = schema.validate(input_data)
    assert result["name"] == "John"
    assert result["age"] == 25
    assert "meta" not in result

    # 2. Test default value injection
    input_no_age = {"name": "Jane"}
    result_default = schema.validate(input_no_age)
    assert result_default["age"] == 10

    # 3. Test validation error: Null not allowed
    schema_not_null = Schema(fields=fields, allow_null=False)
    with pytest.raises(ValidationError) as excinfo:
        schema_not_null.validate(None)
    assert any("May not be null" in str(m) for m in excinfo.value.messages)

    # 4. Test validation error: Type must be object (dict)
    with pytest.raises(ValidationError) as excinfo:
        schema.validate(["not", "a", "dict"])
    assert any("Must be an object" in str(m) for m in excinfo.value.messages)

    # 5. Test validation error: Required field missing
    with pytest.raises(ValidationError) as excinfo:
        schema.validate({"age": 30}) # 'name' is required
    assert any("This field is required" in str(m) for m in excinfo.value.messages)

    # 6. Test validation error: Invalid key type (non-string key)
    with pytest.raises(ValidationError) as excinfo:
        schema.validate({123: "value"})
    assert any("All object keys must be strings" in str(m) for m in excinfo.value.messages)

    # 7. Test child field validation error propagation
    field_err = MagicMock(spec=Field)
    field_err.read_only = False
    field_err.has_default.return_value = False
    # Mocking the error return from child field
    error_msg = Message(text="Child error", code="child_err", index=["name"])
    error_obj = MagicMock()
    error_obj.messages.return_value = [error_msg]
    field_err.validate_or_error.return_value = (None, error_obj)

    schema_with_error = Schema(fields={"name": field_err})
    with pytest.raises(ValidationError) as excinfo:
        schema_with_error.validate({"name": "bad_val"})
    assert any("Child error" in str(m) for m in excinfo.value.messages)
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Reference_validate():
    # Mock definitions and a target field
    mock_definitions = Definitions()
    mock_target_field = MagicMock(spec=Field)
    mock_definitions["my_ref"] = mock_target_field
    
    # 1. Test validation with valid value (delegates to target)
    ref_valid = Reference(to="my_ref", definitions=mock_definitions)
    input_value = {"data": 123}
    mock_target_field.validate.return_value = input_value
    
    assert ref_valid.validate(input_value) == input_value
    mock_target_field.validate.assert_called_with(input_value)

    # 2. Test validation with None and allow_null=True
    ref_allow_null = Reference(to="my_ref", definitions=mock_definitions, allow_null=True)
    assert ref_allow_null.validate(None) is None

    # 3. Test validation with None and allow_null=False (raises error)
    ref_no_null = Reference(to="my_ref", definitions=mock_definitions, allow_null=False)
    with pytest.raises(ValidationError) as excinfo:
        ref_no_null.validate(None)
    # Check if the error message relates to 'null' (based on Reference.errors)
    assert any("May not be null" in str(m) for m in excinfo.value.messages)

    # 4. Test validation failure delegated from target
    mock_target_field.validate.side_effect = ValidationError(messages=[Message(text="Invalid", code="error")])
    with pytest.raises(ValidationError) as excinfo:
        ref_valid.validate({"wrong": "data"})
    assert len(excinfo.value.messages) == 1
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Schema():
    # Setup mock fields
    field_required = MagicMock(spec=Field)
    field_required.read_only = False
    field_required.has_default.return_value = False

    field_optional = MagicMock(spec=Field)
    field_optional.read_only = False
    field_optional.has_default.return_value = True

    field_readonly = MagicMock(spec=Field)
    field_readonly.read_only = True
    field_readonly.has_default.return_value = False

    fields_dict = {
        "required_field": field_required,
        "optional_field": field_optional,
        "readonly_field": field_readonly
    }

    # Test Constructor
    schema = Schema(fields=fields_dict)

    # Assertions
    assert schema.fields == fields_dict
    # 'required_field' is required because it's not read_only and has no default
    # 'optional_field' should NOT be in required list (has_default=True)
    # 'readonly_field' should NOT be in required list (read_only=True)
    assert "required_field" in schema.required
    assert "optional_field" not in schema.required
    assert "readonly_field" not in schema.required

    # Test with empty fields
    empty_schema = Schema(fields={})
    assert empty_schema.required == []

    # Test kwargs propagation (allow_null)
    schema_nullable = Schema(fields=fields_dict, allow_null=True)
    assert schema_nullable.allow_null is True
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Schema_validate():
    # Mock Fields
    field_str = MagicMock(spec=Field)
    field_str.read_only = False
    field_str.has_default.return_value = False
    field_str.validate_or_error.side_effect = lambda x: (x, None)
    field_str.serialize.side_effect = lambda x: x

    field_int = MagicMock(spec=Field)
    field_int.read_only = False
    field_int.has_default.return_value = True
    field_int.get_default_value.return_value = 10
    field_int.validate_or_error.side_effect = lambda x: (x, None)
    field_int.serialize.side_effect = lambda x: x

    field_readonly = MagicMock(spec=Field)
    field_readonly.read_only = True
    field_readonly.has_default.return_value = False
    field_readonly.validate_or_error.side_effect = lambda x: (x, None)
    field_readonly.serialize.side_effect = lambda x: x

    fields = {
        "name": field_str,
        "age": field_int,
        "meta": field_readonly
    }

    schema = Schema(fields=fields)

    # 1. Test valid input with all fields present
    input_data = {"name": "John", "age": 30, "meta": "some_meta"}
    result = schema.validate(input_data)
    assert result == {"name": "John", "age": 30}
    # 'meta' is read_only so it shouldn't be in validated output if not processed by Schema logic loop
    # But according to code: if key not in value and has_default, it adds. 
    # If key in value but read_only, it skips the loop for that key.

    # 2. Test default value injection
    input_data_missing_age = {"name": "John"}
    result_default = schema.validate(input_data_missing_age)
    assert result_default["age"] == 10

    # 3. Test null validation error (not allowed by default)
    with pytest.raises(ValidationError) as excinfo:
        schema.validate(None)
    assert any("May not be null" in str(m.text) for m in excinfo.value.messages)

    # 4. Test type error (not a mapping)
    with pytest.raises(ValidationError) as excinfo:
        schema.validate(["not", "a", "dict"])
    assert any("Must be an object" in str(m.text) for m in excinfo.value.messages)

    # 5. Test invalid key type (integer key)
    with pytest.raises(ValidationError) as excinfo:
        schema.validate({123: "value"})
    assert any("All object keys must be strings" in str(m.text) for m in excinfo.value.messages)

    # 6. Test required field missing
    with pytest.raises(ValidationError) as excinfo:
        schema.validate({"age": 20})
    assert any("This field is required" in str(m.text) for m in excinfo.value.messages)

    # 7. Test nested validation error propagation
    field_str.validate_or_error.side_effect = lambda x: (None, MagicMock(messages=lambda add_prefix: [Message(text=f"Error in {add_prefix}name", code="error")]))
    with pytest.raises(ValidationError) as excinfo:
        schema.validate({"name": "John"})
    assert any("Error in name" in str(m.text) for m in excinfo.value.messages)

    # 8. Test allow_null = True
    schema_nullable = Schema(fields=fields, allow_null=True)
    assert schema_nullable.validate(None) is None
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Schema_validate():
    # Mock Fields
    field_str = MagicMock(spec=Field)
    field_str.read_only = False
    field_str.has_default.return_value = False
    field_str.validate_or_error.side_effect = lambda x: (x, None)
    field_str.serialize.return_value = "val"

    field_int = MagicMock(spec=Field)
    field_int.read_only = False
    field_int.has_default.return_value = True
    field_int.get_default_value.return_value = 10
    field_int.validate_or_error.side_effect = lambda x: (x, None)
    field_int.serialize.return_value = 10

    field_readonly = MagicMock(spec=Field)
    field_readonly.read_only = True
    field_readonly.has_default.return_value = False
    field_readonly.validate_or_error.side_effect = lambda x: (x, None)

    fields = {
        "name": field_str,
        "age": field_int,
        "active": field_readonly,
        "required_field": field_str
    }

    schema = Schema(fields=fields)

    # 1. Test valid input with all fields present
    valid_input = {"name": "John", "age": 30, "active": True, "required_field": "exists"}
    assert schema.validate(valid_input) == {"name": "John", "age": 30}

    # 2. Test valid input with defaults (age is missing but has default)
    input_with_defaults = {"name": "Jane", "required_field": "exists"}
    assert schema.validate(input_with_defaults) == {"name": "Jane", "age": 10}

    # 3. Test Null error (allow_null is False by default)
    with pytest.raises(ValidationError) as excinfo:
        schema.validate(None)
    assert any("May not be null" in str(m) for m in excinfo.value.messages)

    # 4. Test Type error (input is list instead of dict)
    with pytest.raises(ValidationError) as excinfo:
        schema.validate(["not", "a", "dict"])
    assert any("Must be an object" in str(m) for m in excinfo.value.messages)

    # 5. Test Invalid Key error (non-string key)
    with pytest.raises(ValidationError) as excinfo:
        schema.validate({123: "value", "name": "John", "required_field": "exists"})
    assert any("All object keys must be strings" in str(m) for m in excinfo.value.messages)

    # 6. Test Required error (missing required_field)
    with pytest.raises(ValidationError) as excinfo:
        schema.validate({"name": "John", "age": 30})
    assert any("This field is required" in str(m) for m in excinfo.value.messages)

    # 7. Test Nested Validation Error propagation
    field_str.validate_or_error.side_effect = lambda x: (None, MagicMock(messages=lambda add_prefix: [MagicMock(text=f"Error in {x}", code="nested")]))
    with pytest.raises(ValidationError) as excinfo:
        schema.validate({"name": "John", "required_field": "exists"})
    # Check if the error message contains the prefix from child field validation
    assert any("name" in str(m) for m in excinfo.value.messages)

    # 8. Test allow_null = True
    schema_nullable = Schema(fields=fields, allow_null=True)
    assert schema_nullable.validate(None) is None
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Schema_validate():
    # Mock Fields
    class MockField(Field):
        def __init__(self, default=None, read_only=False, has_default=False):
            self.default = default
            self.read_only = read_only
            self._has_default = has_default
        def has_default(self): return self._has_default
        def get_default_value(self): return self.default
        def validate_or_error(self, value):
            # Simple mock validation logic
            if value == "error":
                err = MagicMock()
                err.messages.return_value = [Message(text="child error", code="child_error")]
                return None, err
            return value, None

    string_field = MockField()
    int_field = MockField(default=10, has_default=True)
    read_only_field = MockField(read_only=True)

    schema_fields = {
        "name": string_field,
        "age": int_field,
        "status": read_only_field
    }
    schema = Schema(schema_fields)

    # 1. Test valid input with all fields present
    valid_input = {"name": "Alice", "age": 25, "status": "active"}
    assert schema.validate(valid_input) == {"name": "Alice", "age": 25}

    # 2. Test default values for missing optional fields
    input_missing_optional = {"name": "Bob"}
    result = schema.validate(input_missing_optional)
    assert result["name"] == "Bob"
    assert result["age"] == 10  # Default applied

    # 3. Test required field missing
    with pytest.raises(ValidationError) as excinfo:
        schema.validate({"age": 30})
    assert any(msg.code == "required" and msg.index == ["name"] for msgly in excinfo.value.messages)

    # 4. Test invalid type (not a mapping)
    with pytest.raises(ValidationError) as excinfo:
        schema.validate("not a dict")
    assert any(msg.code == "type" for msg in excinfo.value.messages)

    # 5. Test null value when not allowed
    with pytest.raises(ValidationError) as excinfo:
        schema.validate(None)
    assert any(msg.code == "null" for msg in excinfo.value.messages)

    # 6. Test null value when allowed
    schema_nullable = Schema({"name": MockField()}, allow_null=True)
    assert schema_nullable.validate(None) is None

    # 7. Test invalid key type (non-string key)
    with pytest.raises(ValidationError) as excinfo:
        schema.validate({123: "value"})
    assert any(msg.code == "invalid_key" and msg.index == [123] for msg in excinfo.value.messages)

    # 8. Test child field validation error propagation
    with pytest.raises(ValidationError) as excinfo:
        schema.validate({"name": "error", "age": 25})
    # Check if the error message is prefixed with the key name
    assert any("name" in str(msg.text) or msg.code == "child_error" for msg in excinfo.value.messages)

    # 9. Test read_only field is ignored during validation of input values
    # Even if 'status' is provided in input, it shouldn't be in the output validated dict
    input_with_readonly = {"name": "Charlie", "age": 40, "status": "should_be_ignored"}
    result_readonly = schema.validate(input_with_readonly)
    assert "status" not in result_readonly
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Reference_validate():
    # Mocking Definitions container
    definitions = Definitions()
    
    # Mock target field (the schema being referenced)
    target_field = MagicMock(spec=Field)
    definitions["my_ref"] = target_field
    
    # 1. Test successful validation
    reference = Reference(to="my_ref", definitions=definitions)
    valid_value = {"name": "test"}
    target_field.validate.return_value = valid_value
    target_field.validate_or_error.return_value = (valid_value, None)

    assert reference.validate(valid_value) == valid_value
    target_field.validate.assert_called_with(valid_value)

    # 2. Test null value with allow_null=True
    reference_allow_null = Reference(to="my_ref", definitions=definitions, allow_null=True)
    assert reference_allow_null.validate(None) is None

    # 3. Test null value with allow_null=False (default)
    reference_no_null = Reference(to="my_ref", definitions=definitions, allow_null=False)
    with pytest.raises(ValidationError) as excinfo:
        reference_no_null.validate(None)
    # Check if the error message corresponds to the "null" key in Reference.errors
    assert any("May not be null." in str(m) for m in excinfo.value.messages)

    # 4. Test validation failure propagation (when target field raises error)
    error_message = Message(text="Invalid value", code="error_code")
    target_field.validate.side_effect = ValidationError(messages=[error_message])
    
    with pytest.raises(ValidationError) as excinfo:
        reference.validate({"bad": "data"})
    assert len(excinfo.value.messages) == 1
    assert excinfo.value.messages[0].code == "error_code"

    # 5. Test target lookup via property
    assert reference.target == target_field
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Schema_validate():
    # Mocking Field behavior for sub-schemas
    def create_mock_field(required=True, default=None, value=None):
        field = MagicMock(spec=Field)
        field.read_only = False
        field.has_default = lambda: default is not None
        field.get_default_value = lambda: default
        # validate_or_error returns (validated_value, error_list)
        if value is not None:
            field.validate_or_error.return_value = (value, [])
        else:
            # Simulate a validation error in child field
            err_msg = Message(text="Error", code="error", index=["sub"])
            err = MagicMock()
            err.messages.return_value = [err_msg]
            field.validate_or_error.return_value = (None, err)
        return field

    # 1. Test Valid Input
    field_a = create_mock_field(value="val_a")
    field_b = create_mock_field(default="def_b")
    schema = Schema(fields={"a": field_a, "b": field_b})
    
    input_data = {"a": "val_a"}
    # b is missing but has default
    result = schema.validate(input_data)
    assert result["a"] == "val_a"
    assert result["b"] == "def_b"

    # 2. Test Null Error
    schema_no_null = Schema(fields={"a": field_a}, allow_null=False)
    with pytest.raises(ValidationError) as excinfo:
        schema_no_null.validate(None)
    assert any("May not be null." in str(m) for m in excinfo.value.messages)

    # 3. Test Null Success (allow_null=True)
    schema_allow_null = Schema(fields={"a": field_a}, allow_null=True)
    assert schema_allow_null.validate(None) is None

    # 4. Test Type Error (not a dict)
    with pytest.raises(ValidationError) as excinfo:
        schema.validate(["not", "a", "dict"])
    assert any("Must be an object." in str(m) for m in excinfo.value.messages)

    # 5. Test Invalid Key Type (non-string key)
    with pytest.raises(ValidationError) as excinfo:
        schema.validate({123: "value"})
    assert any("All object keys must be strings." in str(m) for m in excinfo.value.messages)

    # 6. Test Required Field Missing
    required_field = create_mock_field() # required=True, no default
    schema_req = Schema(fields={"a": required_field})
    with pytest.raises(ValidationError) as excinfo:
        schema_req.validate({"b": 1})
    assert any("This field is required." in str(m) for m in excinfo.value.messages)

    # 7. Test Nested Validation Error propagation
    field_error = create_mock_field(value=None) # This triggers error logic in our mock
    schema_nested = Schema(fields={"a": field_error})
    with pytest.raises(ValidationError) as excinfo:
        schema_nested.validate({"a": "some_value"})
    # Check if the error from child field propagated with prefix
    assert any("a" in str(m.index) for m in excinfo.value.messages)

    # 8. Test Read Only Field (should be ignored during validation/processing)
    ro_field = create_mock_field(value="should_not_appear")
    ro_field.read_only = True
    schema_ro = Schema(fields={"a": ro_field})
    # Even if 'a' is in input, it shouldn't be in validated output because it's read_only
    result_ro = schema_ro.validate({"a": "val"})
    assert "a" not in result_ro
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Schema_validate():
    # Mocking Field for schema definitions
    def create_mock_field(required=True, default=None, read_only=False):
        field = MagicMock(spec=Field)
        field.has_default.return_value = default is not None
        field.get_default_value.return_value = default
        field.read_only = read_only
        # Mock validate_or_error to return (value, error)
        # If error is None, validation passed
        field.validate_or_error.side_effect = lambda x: (x, None)
        return field

    # 1. Test valid input
    string_field = create_mock_field()
    int_field = create_mock_field(default=10)
    schema = Schema({"name": string_field, "age": int_field})
    
    input_data = {"name": "John", "age": 25}
    assert schema.validate(input_data) == {"name": "John", "age": 25}

    # 2. Test default value injection
    schema_with_default = Schema({"age": int_field})
    assert schema_with_default.validate({"name": "John"}) == {"age": 10}

    # 3. Test null error (not allowed)
    schema_not_nullable = Schema({"name": string_field}, allow_null=False)
    with pytest.raises(ValidationError) as excinfo:
        schema_not_nullable.validate(None)
    assert any("May not be null" in str(m.text) for m in excinfo.value.messages)

    # 4. Test null allowed
    schema_nullable = Schema({"name": string_field}, allow_null=True)
    assert schema_nullable.validate(None) is None

    # 5. Test invalid type (not a dict/mapping)
    with pytest.raises(ValidationError) as excinfo:
        schema.validate(["not", "a", "dict"])
    assert any("Must be an object" in str(m.text) for m in excinfo.value.messages)

    # 6. Test invalid key type (non-string keys)
    with pytest.raises(ValidationError) as excinfo:
        schema.validate({123: "value"})
    assert any("All object keys must be strings" in str(m.text) for m in excinfo.value.messages)

    # 7. Test required field missing
    with pytest.raises(ValidationError) as excinfo:
        schema.validate({"age": 25})  # 'name' is required
    assert any("This field is required" in str(m.text) for m in excinfo.value.messages)

    # 8. Test child field validation failure
    error_field = MagicMock(spec=Field)
    error_field.has_default.return_value = False
    error_field.read_only = False
    # Simulate a validation error from the child field
    child_msg = Message(text="Child Error", code="child_err")
    error_class = MagicMock()
    error_class.messages.return_value = [child_msg]
    error_field.validate_or_error.return_value = (None, error_class)

    schema_with_error = Schema({"child": error_field})
    with pytest.raises(ValidationError) as excinfo:
        schema_with_error.validate({"child": "bad_data"})
    
    # Check if the error message is prefixed with the key name
    assert any("child: Child Error" in str(m.text) for m in excinfo.value.messages)

    # 9. Test read_only field (should be skipped during validation loop)
    read_only_field = create_mock_field(read_only=True)
    schema_readonly = Schema({"const": read_only_field})
    # Even if 'const' is missing from input, it shouldn't trigger 'required' logic 
    # because the loop skips keys that are read_only.
    assert schema_readonly.validate({}) == {}
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Schema():
    # Setup mock fields
    field_required = MagicMock(spec=Field)
    field_required.read_only = False
    field_required.has_default.return_value = False

    field_optional = MagicMock(spec=Field)
    field_optional.read_only = False
    field_optional.has_default.return_value = True

    field_readonly = MagicMock(spec=Field)
    field_readonly.read_only = True
    field_readonly.has_default.return_value = False

    fields = {
        "req": field_required,
        "opt": field_optional,
        "ro": field_readonly
    }

    # Test Constructor
    schema = Schema(fields=fields)

    # Assertions on fields assignment
    assert schema.fields == fields

    # Assertions on required list logic
    # 'req' is required (not read_only and no default)
    # 'opt' is not required (has default)
    # 'ro' is not required (is read_only)
    assert "req" in schema.required
    assert "opt" not in schema.required
    assert "ro" not in schema.required
    assert len(schema.required) == 1

    # Test with empty fields
    empty_schema = Schema(fields={})
    assert empty_schema.fields == {}
    assert empty_schema.required == []

    # Test kwargs propagation (allow_null)
    schema_with_null = Schema(fields=fields, allow_null=True)
    assert schema_with_null.allow_null is True
```


# LLM-generated content at query #16
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Reference_validate():
    # Setup mocks for dependencies
    mock_target_field = MagicMock(spec=Field)
    definitions = Definitions({"user": mock_target_field})
    
    # 1. Test valid value (delegates to target field)
    ref_valid = Reference(to="user", definitions=definitions)
    mock_target_field.validate.return_value = {"id": 1, "name": "John"}
    assert ref_valid.validate({"id": 1, "name": "John"}) == {"id": 1, "name": "John"}
    mock_target_field.validate.assert_called_with({"id": 1, "name": "John"})

    # 2. Test null value when allow_null is True
    ref_allow_null = Reference(to="user", definitions=definitions, allow_null=True)
    assert ref_allow_null.validate(None) is None

    # 3. Test null value when allow_null is False (default)
    ref_not_allow_null = Reference(to="user", definitions=definitions, allow_null=False)
    with pytest.raises(ValidationError) as excinfo:
        ref_not_allow_null.validate(None)
    # Check if the error message matches the "null" key in Reference.errors
    assert any("May not be null." in str(msg) for msg in excinfo.value.messages)

    # 4. Test value that triggers validation error in target field
    mock_target_field.validate.side_effect = ValidationError(messages=[Message(text="Invalid", code="error")])
    with pytest.raises(ValidationError) as excinfo:
        ref_valid.validate({"id": "invalid"})
    assert len(excinfo.value.messages) == 1
    assert excinfo.value.messages[0].code == "error"

    # 5. Test missing key in definitions (should raise KeyError via target property access)
    ref_missing = Reference(to="nonexistent", definitions=definitions)
    with pytest.raises(KeyError):
        ref_missing.validate({"some": "data"})
```


# LLM-generated content at query #17
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Schema_validate():
    # Mock Fields
    field_string = MagicMock(spec=Field)
    field_string.read_only = False
    field_string.has_default.return_value = False
    field_string.validate_or_error.side_effect = lambda x: (x, None)
    field_string.serialize.return_value = "val"

    field_int = MagicMock(spec=Field)
    field_int.read_only = False
    field_int.has_default.return_value = True
    field_int.get_default_value.return_value = 10
    field_int.validate_or_error.side_effect = lambda x: (x, None)
    field_int.serialize.return_value = 10

    field_readonly = MagicMock(spec=Field)
    field_readonly.read_only = True
    field_readonly.has_default.return_value = False
    field_readonly.validate_or_error.side_effect = lambda x: (x, None)
    field_readonly.serialize.return_value = "readonly"

    fields = {
        "name": field_string,
        "age": field_int,
        "meta": field_readonly
    }

    schema = Schema(fields=fields)

    # 1. Test Valid Input (including defaults and skipping read-only)
    input_data = {"name": "John", "age": 25, "meta": "hidden"}
    result = schema.validate(input_data)
    assert result["name"] == "John"
    assert result["age"] == 25
    assert "meta" not in result  # read_only fields are skipped in validation loop

    # 2. Test Default Value Application
    input_data_no_age = {"name": "John"}
    result_defaults = schema.validate(input_data_no_age)
    assert result_defaults["age"] == 10

    # 3. Test Null Error (when allow_null is False)
    schema.allow_null = False
    with pytest.raises(ValidationError) as excinfo:
        schema.validate(None)
    assert any("May not be null" in str(m) for m in excinfo.value.messages)

    # 4. Test Null Success (when allow_null is True)
    schema.allow_null = True
    assert schema.validate(None) is None

    # 5. Test Type Error (not a mapping)
    with pytest.raises(ValidationError) as excinfo:
        schema.validate(["not", "a", "dict"])
    assert any("Must be an object" in str(m) for m in excinfo.value.messages)

    # 6. Test Invalid Key (non-string key)
    with pytest.raises(ValidationError) as excinfo:
        schema.validate({123: "value"})
    assert any("All object keys must be strings" in str(m) for m in excinfo.value.messages)

    # 7. Test Required Field Missing
    # 'name' is required because it has no default and is not read_only
    with pytest.raises(ValidationError) as excinfo:
        schema.validate({"age": 25})
    assert any("This field is required" in str(m) for m in excinfo.value.messages)

    # 8. Test Child Field Validation Error
    field_string.validate_or_error.side_effect = lambda x: (None, [Message(text="Bad string", code="err", index=[])])
    with pytest.raises(ValidationError) as excinfo:
        schema.validate({"name": 123})
    # Check if the error is prefixed with the key name 'name'
    assert any("name: Bad string" in str(m) for m in excinfo.value.messages)
```


# LLM-generated content at query #18
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Reference_validate():
    # Setup common mocks
    definitions = Definitions()
    target_field = MagicMock(spec=Field)
    definitions["my_ref"] = target_field

    # Test case 1: value is None and allow_null is True
    ref_allow_null = Reference(to="my_ref", definitions=definitions, allow_null=True)
    assert ref_allow_null.validate(None) is None

    # Test case 2: value is None and allow_null is False
    ref_no_null = Reference(to="my_ref", definitions=definitions, allow_null=False)
    with pytest.raises(ValidationError) as excinfo:
        ref_no_null.validate(None)
    # Check if the error message relates to 'null'
    assert any("May not be null" in str(m) for m in excinfo.value.messages)

    # Test case 3: value is not None, validate calls target field with correct value
    test_value = {"key": "value"}
    target_field.validate.return_value = test_value
    
    ref = Reference(to="my_ref", definitions=definitions)
    result = ref.validate(test_value)
    
    target_field.validate.assert_called_once_with(test_value)
    assert result == test_value

    # Test case 4: value is not None, target field raises ValidationError
    target_field.validate.side_effect = ValidationError(messages=[MagicMock(text="error")])
    ref_fail = Reference(to="my_ref", definitions=definitions)
    with pytest.raises(ValidationError):
        ref_fail.validate(test_value)

    # Test case 5: value is not None, target field returns error via validate_or_error (logic check)
    # Note: Reference.validate calls target.validate directly, so we test the delegation logic.
    target_field.validate.side_effect = None
    target_field.validate.return_value = "valid_result"
    assert ref.validate("some_input") == "valid_result"
```


# LLM-generated content at query #19
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Reference_validate():
    # Setup mock definitions and target field
    mock_definitions = Definitions()
    mock_target_field = MagicMock(spec=Field)
    mock_definitions["my_ref"] = mock_target_field
    
    # Test Case 1: Value is None and allow_null is True
    ref_allow_null = Reference(to="my_ref", definitions=mock_definitions, allow_null=True)
    assert ref_allow_null.validate(None) is None

    # Test Case 2: Value is None and allow_null is False
    ref_no_null = Reference(to="my_ref", definitions=mock_definitions, allow_null=False)
    with pytest.raises(ValidationError) as excinfo:
        ref_no_null.validate(None)
    assert "May not be null." in str(excinfo.value)

    # Test Case 3: Successful validation (delegates to target)
    expected_value = {"key": "value"}
    mock_target_field.validate.return_value = expected_value
    
    ref_success = Reference(to="my_ref", definitions=mock_definitions)
    input_data = {"some": "data"}
    result = ref_success.validate(input_data)
    
    mock_target_field.validate.assert_called_once_with(input_data)
    assert result == expected_value

    # Test Case 4: Validation error (delegates error from target)
    mock_error = ValidationError(messages=[Message(text="Error in target", code="error")])
    mock_target_field.validate.side_effect = mock_error
    
    ref_error = Reference(to="my_ref", definitions=mock_definitions)
    with pytest.raises(ValidationError) as excinfo:
        ref_error.validate({"bad": "data"})
    assert "Error in target" in str(excinfo.value)
```


# LLM-generated content at query #20
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Schema_validate():
    # Mocking Field behavior
    def create_mock_field(name, required=True, default=None, read_only=False):
        field = MagicMock(spec=Field)
        field.read_only = read_only
        field.has_default.return_value = default is not None
        field.get_default_value.return_value = default
        # Mocking validate_or_error to return (value, error)
        # If error is None, it's valid.
        field.validate_or_error.side_effect = lambda v: (v, None)
        return field

    # 1. Test successful validation of a valid dictionary
    fields = {
        "name": create_mock_field("name"),
        "age": create_mock_field("age", required=False)
    }
    schema = Schema(fields)
    input_data = {"name": "John", "age": 30}
    assert schema.validate(input_data) == {"name": "John", "age": 30}

    # 2. Test validation failure: input is not a mapping (type error)
    with pytest.raises(ValidationError) as excinfo:
        schema.validate(["not", "a", "dict"])
    assert any(msg.code == "type" for msg in excinfo.value.messages)

    # 3. Test validation failure: input is None and not allow_null
    with pytest.raises(ValidationError) as excinfo:
        schema.validate(None)
    assert any(msg.code == "null" for msg in excinfo.value.messages)

    # 4. Test validation failure: invalid key type (non-string key)
    input_data_invalid_key = {123: "value"}
    with pytest.raises(ValidationError) as excinfo:
        schema.validate(input_data_invalid_key)
    assert any(msg.code == "invalid_key" for msg in excinfo.value.messages)

    # 5. Test validation failure: missing required field
    input_data_missing_req = {"age": 30} # 'name' is required
    with pytest.raises(ValidationError) as excinfo:
        schema.validate(input_data_missing_req)
    assert any(msg.code == "required" and msg.index == ["name"] for msg in excinfo.value.messages)

    # 6. Test default value injection
    fields_with_default = {
        "role": create_mock_field("role", default="user")
    }
    schema_default = Schema(fields_with_default)
    input_data_empty = {}
    # 'role' is not in required because it has a default. 
    # It should be populated by the default value during validation.
    result = schema_default.validate(input_data_empty)
    assert result["role"] == "user"

    # 7. Test nested field error propagation
    child_field = create_mock_field("child")
    # Simulate a child validation error
    error_msg = Message(text="Child error", code="child_err", index=[])
    child_field.validate_or_error.side_effect = lambda v: (None, MagicMock(messages=lambda add_prefix: [Message(text=f"{add_prefix}error", code="child_err", index=[])]))
    
    schema_nested = Schema({"child": child_field})
    with pytest.raises(ValidationError) as excinfo:
        schema_nested.validate({"child": {"some": "data"}})
    assert any("childerror" in msg.text for msg in excinfo.value.messages)

    # 8. Test allow_null functionality
    schema_nullable = Schema({"name": create_mock_field("name")}, allow_null=True)
    assert schema_nullable.validate(None) is None
```


# LLM-generated content at query #21
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Reference_validate():
    # Setup mocks for definitions and target field
    mock_target_field = MagicMock(spec=Field)
    definitions = Definitions({"my_ref": mock_target_field})
    
    # Case 1: value is None and allow_null is True
    ref_allow_null = Reference(to="my_ref", definitions=definitions, allow_null=True)
    assert ref_allow_null.validate(None) is None

    # Case 2: value is None and allow_null is False (default)
    ref_no_null = Reference(to="my_ref", definitions=definitions, allow_null=False)
    with pytest.raises(ValidationError) as excinfo:
        ref_no_null.validate(None)
    # Check if the error message corresponds to "null" key in Reference.errors
    assert any("May not be null." in str(msg) for msg in excinfo.value.messages)

    # Case 3: value is valid, delegate validation to target field
    input_value = {"data": 123}
    mock_target_field.validate.return_value = {"data": 123}
    ref_delegate = Reference(to="my_ref", definitions=definitions)
    result = ref_delegate.validate(input_value)
    
    assert result == {"data": 123}
    mock_target_field.validate.assert_called_once_with(input_value)

    # Case 4: value is valid, but target field raises ValidationError
    mock_error = MagicMock(spec=ValidationError)
    mock_error.messages.return_value = [MagicMock(text="Error", code="err")]
    mock_target_field.validate_or_error.return_value = (None, mock_error)
    # Note: Reference calls .validate(), not .validate_or_error() directly, 
    # but if target.validate raises ValidationError, it bubbles up.
    mock_target_field.validate.side_effect = mock_error
    
    ref_delegate = Reference(to="my_ref", definitions=definitions)
    with pytest.raises(ValidationError):
        ref_delegate.validate({"bad": "data"})
```


# LLM-generated content at query #22
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Reference_validate():
    # Setup mocks
    definitions = Definitions()
    target_field = MagicMock(spec=Field)
    definitions["my_ref"] = target_field
    
    # 1. Test valid value (delegates to target.validate)
    reference = Reference(to="my_ref", definitions=definitions, allow_null=False)
    target_field.validate.return_value = "valid_data"
    assert reference.validate({"some": "data"}) == "valid_data"
    target_field.validate.assert_called_with({"some": "data"})

    # 2. Test null value when allow_null is False
    reference_no_null = Reference(to="my_ref", definitions=definitions, allow_null=False)
    with pytest.raises(ValidationError) as excinfo:
        reference_no_null.validate(None)
    assert "May not be null." in str(excinfo.value)

    # 3. Test null value when allow_null is True
    reference_allow_null = Reference(to="my_ref", definitions=definitions, allow_null=True)
    assert reference_allow_null.validate(None) is None

    # 4. Test error propagation from target field
    error_msg = Message(text="Target error", code="target_err", index=[])
    validation_error = ValidationError(messages=[error_msg])
    target_field.validate.side_effect = validation_error
    
    with pytest.raises(ValidationError) as excinfo:
        reference.validate({"bad": "data"})
    assert "Target error" in str(excinfo.value)
```


# LLM-generated content at query #23
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Schema_validate():
    # Mocking Field behavior
    def create_mock_field(required=True, default=None, read_only=False):
        field = MagicMock(spec=Field)
        field.has_default.return_value = default is not None
        field.get_default_value.return_value = default
        field.read_only = read_only
        # Mocking validate_or_error to return (value, error)
        field.validate_or_error.side_effect = lambda x: (x, None)
        field.serialize.side_effect = lambda x: x
        return field

    # Case 1: Successful validation of a simple object
    field_a = create_mock_field()
    field_b = create_mock_field(default="default_val")
    schema = Schema({"a": field_a, "b": field_b})
    
    input_data = {"a": 1, "b": 2}
    assert schema.validate(input_data) == {"a": 1, "b": 2}

    # Case 2: Validating with default value injection
    input_data_missing_b = {"a": 1}
    assert schema.validate(input_data_missing_b) == {"a": 1, "b": "default_val"}

    # Case 3: Null error when allow_null is False (default)
    with pytest.raises(ValidationError) as excinfo:
        schema.validate(None)
    assert any("May not be null" in str(m) for m in excinfo.value.messages)

    # Case 4: Error when value is not a dictionary/mapping
    with pytest.raises(ValidationError) as excinfo:
        schema.validate([1, 2, 3])
    assert any("Must be an object" in str(m) for m in excinfo.value.messages)

    # Case 5: Error when a required field is missing
    with pytest.raises(ValidationError) as excinfo:
        schema.validate({"b": 2})
    assert any("This field is required" in str(m) for m in excinfo.value.messages)

    # Case 6: Error when keys are not strings
    with pytest.raises(ValidationError) as excinfo:
        schema.validate({123: "value"})
    assert any("All object keys must be strings" in str(m) for m in excinfo.value.messages)

    # Case 7: Error propagation from child field validation
    child_field = MagicMock(spec=Field)
    child_field.has_default.return_value = False
    child_field.read_only = False
    # Simulate a validation error in the child field
    error_msg = Message(text="Child Error", code="child_err", index=[])
    child_field.validate_or_error.return_value = (None, MagicMock(messages=lambda add_prefix: [error_msg]))
    
    schema_with_child = Schema({"child": child_field})
    with pytest.raises(ValidationError) as excinfo:
        schema_with_child.validate({"child": "invalid"})
    # Check if error message contains the prefix (the key name)
    assert any("child" in str(m.index) or "Child Error" in str(m.text) for m in excinfo.value.messages)

    # Case 8: allow_null is True
    schema_nullable = Schema({"a": field_a}, allow_null=True)
    assert schema_nullable.validate(None) is None

    # Case 9: Read only fields are skipped during processing
    read_only_field = create_mock_field(read_only=True)
    schema_readonly = Schema({"readonly": read_only_field})
    # Even if 'readonly' is in input, it shouldn't be processed/validated by the loop logic
    assert schema_readonly.validate({"readonly": "some_val"}) == {}
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Schema():
    # Setup mock fields
    field_required = MagicMock(spec=Field)
    field_required.read_only = False
    field_required.has_default.return_value = False

    field_optional = MagicMock(spec=Field)
    field_optional.read_only = False
    field_optional.has_default.return_value = True

    field_readonly = MagicMock(spec=Field)
    field_readonly.read_only = True
    field_readonly.has_default.return_value = False

    fields = {
        "required_key": field_required,
        "optional_key": field_optional,
        "readonly_key": field_readonly,
    }

    # Test constructor and required attribute logic
    schema = Schema(fields=fields)

    assert schema.fields == fields
    # Only 'required_key' should be in the required list 
    # because 'optional_key' has a default and 'readonly_key' is read_only
    assert "required_key" in schema.required
    assert "optional_key" not in schema.required
    assert "readonly_key" not in schema.required
    assert len(schema.required) == 1

    # Test constructor with extra kwargs (passed to super().__init__)
    schema_with_kwargs = Schema(fields=fields, allow_null=True)
    assert schema_with_kwargs.allow_null is True
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Schema_validate():
    # Setup Mock Fields
    field_string = MagicMock(spec=Field)
    field_string.read_only = False
    field_string.has_default.return_value = False
    field_string.validate_or_error.side_effect = lambda x: (x, None)
    field_string.serialize.return_value = "val"

    field_int = MagicMock(spec=Field)
    field_int.read_only = False
    field_int.has_default.return_value = True
    field_int.get_default_value.return_value = 10
    field_int.validate_or_error.side_effect = lambda x: (x, None)
    field_int.serialize.return_value = 10

    field_readonly = MagicMock(spec=Field)
    field_readonly.read_only = True
    field_readonly.has_default.return_value = False
    field_readonly.validate_or_error.side_effect = lambda x: (x, None)
    field_readonly.serialize.return_value = "read-only"

    fields = {
        "name": field_string,
        "age": field_int,
        "status": field_readonly
    }

    schema = Schema(fields=fields)

    # 1. Test valid input (including default value application and skipping read_only)
    input_data = {"name": "Alice", "age": 25, "status": "active"}
    # Note: 'age' is provided, so it should use 25. 'status' is read_only, so it should be ignored in validation loop.
    result = schema.validate(input_data)
    assert result == {"name": "Alice", "age": 25}

    # 2. Test default value application when key is missing
    input_missing_default = {"name": "Bob"}
    result_default = schema.validate(input_missing_default)
    assert result_default["age"] == 10
    assert result_default["name"] == "Bob"

    # 3. Test Type Error (not a dict)
    with pytest.raises(ValidationError) as excinfo:
        schema.validate(["not", "a", "dict"])
    assert any("Must be an object" in str(m.text) for m in excinfo.value.messages)

    # 4. Test Null Error (when allow_null is False)
    schema.allow_null = False
    with pytest.raises(ValidationError) as excinfo:
        schema.validate(None)
    assert any("May not be null" in str(m.text) for m in excinfo.value.messages)

    # 5. Test Required Field Error
    with pytest.raises(ValidationError) as excinfo:
        schema.validate({"age": 30}) # 'name' is required
    assert any("This field is required" in str(m.text) for m in excinfo.value.messages)

    # 6. Test Invalid Key Error (non-string key in input)
    with pytest.raises(ValidationError) as excinfo:
        schema.validate({123: "invalid"})
    assert any("All object keys must be strings" in str(m.text) for m in excinfo.value.messages)

    # 7. Test Child Validation Error propagation
    field_string.validate_or_error.side_effect = lambda x: (None, [Message(text="Invalid string", code="err", index=[])])
    with pytest.raises(ValidationError) as excinfo:
        schema.validate({"name": 123})
    # The error should be prefixed with the key name 'name.'
    assert any("name.Invalid string" in str(m.text) for m in excinfo.value.messages)
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Schema():
    # Mocking Field objects to control read_only and has_default behavior
    field_required = MagicMock(spec=Field)
    field_required.read_only = False
    field_required.has_default.return_value = False

    field_optional = MagicMock(spec=Field)
    field_optional.read_only = False
    field_optional.has_default.return_value = True

    field_readonly = MagicMock(spec=Field)
    field_readonly.read_only = True
    field_readonly.has_default.return_value = False

    fields = {
        "req": field_required,
        "opt": field_optional,
        "ro": field_readonly
    }

    # Test initialization and required keys logic
    schema = Schema(fields=fields)
    
    assert schema.fields == fields
    # 'req' is required because read_only=False and has_default=False
    # 'opt' is NOT required because it has a default
    # 'ro' is NOT required because it is read_only
    assert "req" in schema.required
    assert "opt" not in schema.required
    assert "ro" not in schema.required

    # Test initialization with empty fields
    empty_schema = Schema(fields={})
    assert empty_schema.required == []

    # Test kwargs propagation to parent Field class
    schema_with_kwargs = Schema(fields=fields, allow_null=True)
    assert schema_with_kwargs.allow_null is True
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Schema():
    # Mock fields
    field_required = MagicMock(spec=Field)
    field_required.read_only = False
    field_required.has_default.return_value = False

    field_optional = MagicMock(spec=Field)
    field_optional.read_only = False
    field_optional.has_default.return_value = True

    field_readonly = MagicMock(spec=Field)
    field_readonly.read_only = True
    field_readonly.has_default.return_value = False

    fields = {
        "req": field_required,
        "opt": field_optional,
        "ro": field_readonly
    }

    # Test initialization of required fields list
    schema = Schema(fields=fields)
    
    assert "req" in schema.required
    assert "opt" not in schema.required
    assert "ro" not in schema.required
    assert len(schema.required) == 1

    # Test initialization with empty fields
    empty_schema = Schema(fields={})
    assert empty_schema.required == []

    # Test kwargs propagation to superclass (Field)
    allow_null_schema = Schema(fields=fields, allow_null=True)
    assert allow_null_schema.allow_null is True

    # Test that all fields are stored in the instance
    assert schema.fields == fields
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest

def test_Reference_serialize():
    # Mocking a Field to use as a target in Definitions
    class MockField(Field):
        def serialize(self, value):
            return f"serialized_{value}"
        def validate(self, value):
            return value

    definitions = Definitions()
    definitions["my_ref"] = MockField()
    
    # Initialize Reference
    reference = Reference(to="my_ref", definitions=definitions)

    # Case 1: Value is None (should return None according to implementation)
    assert reference.serialize(None) is None

    # Case 2: Value is a dictionary
    input_data = {"key": "value", "nested": 123}
    # Reference.serialize returns dict(obj) if obj is not None
    result = reference.serialize(input_data)
    assert result == {"key": "value", "nested": 123}
    assert isinstance(result, dict)
    assert result is not input_data  # Ensure it's a copy

    # Case 3: Value is another type (e.g., list), should still return dict(obj) if possible
    # Note: Reference.serialize calls dict(obj). If obj is not a mapping, it might raise TypeError.
    # We test the standard intended usage where obj is expected to be a mapping-like object.
    input_list = [("key", "value")]
    result_from_list = reference.serialize(input_list)
    assert result_from_list == {"key": "value"}

    # Case 4: Verify it handles non-dict objects that can be cast to dict
    class ObjectWithDictInterface:
        def __init__(self, d):
            self.d = d
        def __iter__(self):
            return iter(self.d)
        def __getitem__(self, key):
            return self.d[key]

    obj_interface = ObjectWithDictInterface({"a": 1})
    assert reference.serialize(obj_interface) == {"a": 1}
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Schema_validate():
    # Mock fields
    string_field = MagicMock(spec=Field)
    string_field.read_only = False
    string_field.has_default.return_value = False
    string_field.validate_or_error.side_effect = lambda x: (x, None)
    string_field.serialize.return_value = "val"

    int_field = MagicMock(spec=Field)
    int_field.read_only = False
    int_field.has_default.return_value = True
    int_field.get_default_value.return_value = 0
    int_field.validate_or_error.side_effect = lambda x: (x, None)
    int_field.serialize.return_value = 0

    read_only_field = MagicMock(spec=Field)
    read_only_field.read_only = True
    read_only_field.has_default.return_value = False
    read_only_field.validate_or_error.side_effect = lambda x: (x, None)

    fields = {
        "name": string_field,
        "age": int_field,
        "metadata": read_only_field
    }

    schema = Schema(fields=fields)

    # 1. Test valid input with all fields present
    input_data = {"name": "Alice", "age": 30, "metadata": "some_meta"}
    assert schema.validate(input_data) == {"name": "Alice", "age": 30}

    # 2. Test valid input with missing optional field (uses default)
    input_data_missing_age = {"name": "Bob"}
    assert schema.validate(input_data_missing_age) == {"name": "Bob", "age": 0}

    # 3. Test null value when allow_null is True
    schema_nullable = Schema(fields=fields, allow_null=True)
    assert schema_nullable.validate(None) is None

    # 4. Test null value when allow_null is False (raises ValidationError)
    with pytest.raises(ValidationError) as excinfo:
        schema.validate(None)
    assert any("May not be null" in str(m) for m in excinfo.value.messages)

    # 5. Test invalid type (not a dict)
    with pytest.raises(ValidationError) as excinfo:
        schema.validate(["not", "a", "dict"])
    assert any("Must be an object" in str(m) for m in excinfo.value.messages)

    # 6. Test invalid key type (non-string key)
    with pytest.raises(ValidationError) as excinfo:
        schema.validate({123: "value"})
    assert any("All object keys must be strings" in str(m) for m in excinfo.value.messages)

    # 7. Test missing required field
    with pytest.raises(ValidationError) as excinfo:
        schema.validate({"age": 25})
    assert any("This field is required" in str(m) for m in excinfo.value.messages)

    # 8. Test child field validation error propagation
    error_field = MagicMock(spec=Field)
    error_field.read_only = False
    error_field.has_default.return_value = False
    
    # Mock a ValidationError from the child field
    child_error_msg = Message(text="Child error", code="child_err")
    class MockError:
        def messages(self, add_prefix=None):
            return [Message(text=f"{add_prefix}Child error", code="child_err")]
    
    error_field.validate_or_error.return_value = (None, MockError())

    schema_with_error = Schema(fields={"error_prop": error_field})
    with pytest.raises(ValidationError) as excinfo:
        schema_with_error.validate({"error_prop": "bad_data"})
    assert any("error_propChild error" in str(m) for m in excinfo.value.messages)
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Reference_validate():
    # Mock definitions and target field
    mock_definitions = MagicMock(spec=Definitions)
    mock_target_field = MagicMock(spec=Field)
    mock_definitions.__getitem__.return_value = mock_target_field
    
    # 1. Test validation with valid non-null value
    ref = Reference(to="target_key", definitions=mock_definitions)
    test_value = {"data": 123}
    mock_target_field.validate.return_value = {"data": 123}
    
    result = ref.validate(test_value)
    
    assert result == {"data": 123}
    mock_target_field.validate.assert_called_once_with(test_value)

    # 2. Test validation with None and allow_null=True
    ref_allow_null = Reference(to="target_key", definitions=mock_definitions, allow_null=True)
    result_null = ref_allow_null.validate(None)
    assert result_null is None

    # 3. Test validation with None and allow_null=False (default)
    ref_no_null = Reference(to="target_key", definitions=mock_definitions, allow_null=False)
    with pytest.raises(ValidationError) as excinfo:
        ref_no_null.validate(None)
    # Checking if the error message matches the "null" key in Reference.errors
    assert any("May not be null" in str(msg) for msg in excinfo.value.messages)

    # 4. Test validation failure passed from target field
    mock_target_field.validate.side_effect = ValidationError(messages=[Message(text="Invalid", code="error")])
    with pytest.raises(ValidationError) as excinfo:
        ref.validate({"bad": "data"})
    assert len(excinfo.value.messages) == 1
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Schema_validate():
    # Mock Fields
    string_field = MagicMock(spec=Field)
    string_field.read_only = False
    string_field.has_default.return_value = False
    string_field.validate_or_error.side_effect = lambda x: (x, None)
    string_field.serialize.return_value = "val"

    int_field = MagicMock(spec=Field)
    int_field.read_only = False
    int_field.has_default.return_value = True
    int_field.get_default_value.return_value = 0
    int_field.validate_or_error.side_effect = lambda x: (x, None)
    int_field.serialize.return_value = 0

    read_only_field = MagicMock(spec=Field)
    read_only_field.read_only = True
    read_only_field.has_default.return_value = False
    read_only_field.validate_or_error.side_effect = lambda x: (x, None)

    fields = {
        "name": string_field,
        "age": int_field,
        "status": read_only_field
    }

    schema = Schema(fields=fields, allow_null=False)

    # 1. Test Valid Input (all provided)
    input_data = {"name": "John", "age": 30, "status": "active"}
    result = schema.validate(input_data)
    assert result == {"name": "John", "age": 30}
    # Note: 'status' is skipped in validation loop because it is read_only

    # 2. Test Default Value Injection
    input_with_missing_default = {"name": "John"}
    result_default = schema.validate(input_with_missing_default)
    assert result_default["age"] == 0
    assert "name" in result_default

    # 3. Test Required Field Missing
    with pytest.raises(ValidationError) as excinfo:
        schema.validate({"age": 30})
    assert any("This field is required." in str(m) for mcap in excinfo.value.messages for m in [mcap])

    # 4. Test Null Value (when not allowed)
    with pytest.raises(ValidationError) as excinfo:
        schema.validate(None)
    assert any("May not be null." in str(m) for m in excinfo.value.messages)

    # 5. Test Null Value (when allowed)
    schema_nullable = Schema(fields=fields, allow_null=True)
    assert schema_nullable.validate(None) is None

    # 6. Test Invalid Type (not a mapping)
    with pytest.raises(ValidationError) as excinfo:
        schema.validate(["not", "a", "dict"])
    assert any("Must be an object." in str(m) for m in excinfo.value.messages)

    # 7. Test Invalid Key Type (non-string key)
    with pytest.raises(ValidationError) as excinfo:
        schema.validate({123: "value"})
    assert any("All object keys must be strings." in str(m) for m in excinfo.value.messages)

    # 8. Test Child Field Validation Error
    error_field = MagicMock(spec=Field)
    error_field.read_only = False
    error_field.has_default.return_value = False
    
    # Mocking the error structure returned by validate_or_error
    class MockError:
        def messages(self, add_prefix):
            return [Message(text=f"Error in {add_prefix}", code="child_err")]

    error_field.validate_or_error.return_value = (None, MockError())
    
    error_schema = Schema(fields={"error_key": error_field})
    with pytest.raises(ValidationError) as excinfo:
        error_schema.validate({"error_key": "bad_data"})
    assert any("Error in error_key" in str(m) for m in excinfo.value.messages)
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Schema_validate():
    # Setup mock fields
    string_field = MagicMock(spec=Field)
    string_field.read_only = False
    string_field.has_default.return_value = False
    string_field.validate_or_error.side_effect = lambda x: (x, None)
    string_field.serialize.return_value = "val"

    int_field = MagicMock(spec=Field)
    int_field.read_only = False
    int_field.has_default.return_value = True
    int_field.get_default_value.return_value = 10
    int_field.validate_or_error.side_effect = lambda x: (x, None)
    int_field.serialize.return_value = 10

    read_only_field = MagicMock(spec=Field)
    read_only_field.read_only = True
    read_only_field.has_default.return_value = False
    read_only_field.validate_or_error.side_effect = lambda x: (x, None)

    fields = {
        "name": string_field,
        "age": int_field,
        "meta": read_only_field
    }

    schema = Schema(fields=fields)

    # Test 1: Valid input with all fields present
    input_data = {"name": "John", "age": 25, "meta": "some_meta"}
    result = schema.validate(input_data)
    assert result == {"name": "John", "age": 25}
    # Note: 'meta' is skipped in validated output because it is read_only

    # Test 2: Valid input with missing optional field (uses default)
    input_data_missing_age = {"name": "John"}
    result = schema.validate(input_data_missing_age)
    assert result == {"name": "json", "age": 10} # Note: 'name' in input was 'John', but logic depends on mock

    # Test 3: Null value error (allow_null=False by default)
    with pytest.raises(ValidationError) as excinfo:
        schema.validate(None)
    assert "May not be null." in str(excinfo.value)

    # Test 4: Type error (not a mapping)
    with pytest.raises(ValidationError) as excinfo:
        schema.validate(["not", "a", "dict"])
    assert "Must be an object." in str(excinfo.value)

    # Test 5: Invalid key type (key is not a string)
    input_data_bad_key = {123: "value"}
    with pytest.raises(ValidationError) as excinfo:
        schema.validate(input_data_bad_key)
    assert any("All object keys must be strings." in str(m.text) for m in excinfo.value.messages)

    # Test 6: Required field missing
    input_data_missing_req = {"age": 25}
    with pytest.raises(ValidationError) as excinfo:
        schema.validate(input_data_missing_req)
    assert any("This field is required." in str(m.text) for m in excinfo.value.messages)

    # Test 7: Child field validation error propagation
    error_field = MagicMock(spec=Field)
    error_field.read_only = False
    error_field.has_default.return_value = False
    # Simulate an error from the child field
    error_msg = Message(text="Child Error", code="child_err")
    error_field.validate_or_error.return_value = (None, MagicMock(messages=lambda add_prefix: [error_msg]))
    
    error_schema = Schema(fields={"child": error_field})
    with pytest.raises(ValidationError) as excinfo:
        error_schema.validate({"child": "bad_data"})
    assert any("Child Error" in str(m.text) for m in excinfo.value.messages)

    # Test 8: allow_null = True
    schema_nullable = Schema(fields={"name": string_field}, allow_null=True)
    assert schema_nullable.validate(None) is None
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Schema():
    # Mock fields
    field_required = MagicMock(spec=Field)
    field_required.read_only = False
    field_required.has_default.return_value = False

    field_optional = MagicMock(spec=Field)
    field_optional.read_only = False
    field_optional.has_default.return_value = True

    field_readonly = MagicMock(spec=Field)
    field_readonly.read_only = True
    field_readonly.has_default.return_value = False

    fields = {
        "required_key": field_required,
        "optional_key": field_optional,
        "readonly_key": field_readonly
    }

    # Test Initialization and required list logic
    schema = Schema(fields=fields)
    
    assert schema.fields == fields
    # Only 'required_key' should be in self.required 
    # because 'optional_key' has a default and 'readonly_key' is read_only
    assert "required_key" in schema.required
    assert "optional_key" not in schema.required
    assert "readonly_key" not in schema.required

    # Test initialization with empty fields
    empty_schema = Schema(fields={})
    assert empty_schema.required == []

    # Test instantiation parameters (kwargs) are passed to super
    schema_with_kwargs = Schema(fields=fields, allow_null=True)
    assert schema_with_kwargs.allow_null is True
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Schema():
    # Mocking Field classes to control behavior
    class MockField(Field):
        def __init__(self, read_only=False, default=None):
            super().__init__()
            self.read_only = read_only
            self._default = default

        def has_default(self):
            return self._default is not None

        def get_default_value(self):
            return self._default

    # Scenario 1: All fields are required (no defaults, not read_only)
    fields_required = {
        "f1": MockField(read_only=False, default=None),
        "f2": MockFragment() if False else MockField(read_only=False, default=None) 
    }
    # Note: The implementation uses 'not (field.read_only or field.has_default())'
    # So required = True if read_only is False AND has_default is False.
    schema_req = Schema(fields=fields_required)
    assert "f1" in schema_req.required
    assert "f2" in schema_req.required

    # Scenario 2: Fields with defaults should not be in required list
    fields_with_defaults = {
        "f1": MockField(read_only=False, default="default_val"),
        "f2": MockField(read_only=False, default=0)
    }
    schema_def = Schema(fields=fields_with_defaults)
    assert "f1" not in schema_def.required
    assert "f2" not in schema_def.required

    # Scenario 3: Read-only fields should not be in required list
    fields_readonly = {
        "f1": MockField(read_only=True, default=None)
    }
    schema_ro = Schema(fields=fields_readonly)
    assert "f1" not in schema_ro.required

    # Scenario 4: Mixed fields
    fields_mixed = {
        "req": MockField(read_only=False, default=None),
        "opt": MockField(read_only=False, default="something"),
        "ro": MockField(read_only=True, default=None)
    }
    schema_mixed = Schema(fields=fields_mixed)
    assert "req" in schema_mixed.required
    assert "opt" not in schema_mixed.required
    assert "ro" not in schema_mixed.required

    # Scenario 5: Verify kwargs are passed to super().__init__
    # We check this by verifying if allow_null is set via kwargs
    schema_kwargs = Schema(fields={}, allow_null=True)
    assert schema_kwargs.allow_null is True
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Reference_validate():
    # Mocking Definitions and target Field
    definitions = Definitions()
    mock_target_field = MagicMock(spec=Field)
    definitions["my_ref"] = mock_target_field

    # 1. Test valid value passes through to target validation
    ref_valid = Reference(to="my_ref", definitions=definitions)
    test_value = {"key": "value"}
    mock_target_field.validate.return_value = test_value
    
    assert ref_valid.validate(test_value) == test_value
    mock_target_field.validate.assert_called_with(test_value)

    # 2. Test null value when allow_null is True
    ref_allow_null = Reference(to="my_ref", definitions=definitions, allow_null=True)
    assert ref_allow_null.validate(None) is None

    # 3. Test null value when allow_null is False (raises ValidationError)
    ref_no_null = Reference(to="my_ref", definitions=definitions, allow_null=False)
    with pytest.raises(ValidationError) as excinfo:
        ref_no_null.validate(None)
    # Check if the error message corresponds to 'null' key in Reference.errors
    assert any("May not be null." in str(msg) for msg in excinfo.value.messages)

    # 4. Test value propagation of ValidationError from target field
    mock_error = MagicMock(spec=ValidationError)
    mock_target_field.validate.side_effect = mock_error
    
    with pytest.raises(ValidationError):
        ref_valid.validate({"some": "data"})
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Schema():
    # Setup mock fields
    field_required = MagicMock(spec=Field)
    field_required.read_only = False
    field_required.has_default.return_value = False

    field_optional = MagicMock(spec=Field)
    field_optional.read_only = False
    field_optional.has_default.return_value = True

    field_readonly = MagicMock(spec=Field)
    field_readonly.read_only = True
    field_readonly.has_default.return_value = False

    fields = {
        "required_key": field_required,
        "optional_key": field_optional,
        "readonly_key": field_readonly,
    }

    # Test constructor logic for 'required' list calculation
    schema = Schema(fields=fields)
    
    # The 'required' attribute should only contain keys that are NOT read_only 
    # AND do NOT have a default value.
    assert "required_key" in schema.required
    assert "optional_key" not in schema.required
    assert "readonly_key" not in schema.required
    assert len(schema.required) == 1

    # Test constructor with empty fields
    empty_schema = Schema(fields={})
    assert empty_schema.required == []

    # Test constructor with kwargs passed to super (Field)
    schema_with_kwargs = Schema(fields=fields, allow_null=True)
    assert schema_with_kwargs.allow_null is True
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Schema():
    # Mock fields
    field_required = MagicMock(spec=Field)
    field_required.read_only = False
    field_required.has_default.return_value = False

    field_optional = MagicMock(spec=Field)
    field_optional.read_only = False
    field_optional.has_default.return_value = True

    field_readonly = MagicMock(spec=Field)
    field_readonly.read_only = True
    field_readonly.has_default.return_value = False

    fields = {
        "required_key": field_required,
        "optional_key": field_optional,
        "readonly_key": field_readonly,
    }

    # Test constructor logic for 'required' list identification
    schema = Schema(fields=fields)
    
    # The 'required' attribute should only contain keys that are NOT read_only AND do NOT have a default
    assert "required_key" in schema.required
    assert "optional_key" not in schema.required
    assert "readonly_key" not in schema.required
    assert len(schema.required) == 1

    # Test constructor with empty fields
    empty_schema = Schema(fields={})
    assert empty_schema.required == []

    # Test kwargs propagation (e.g., allow_null)
    schema_with_null = Schema(fields=fields, allow_null=True)
    assert schema_with_null.allow_null is True

    # Verify Field superclass __init__ was called (implicitly via coverage/logic)
    # We check if the instance has the fields attribute assigned correctly
    assert schema.fields == fields
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Reference_validate():
    # Setup definitions and mock target field
    mock_target_field = MagicMock(spec=Field)
    definitions = Definitions({"user": mock_target_field})
    
    # Case 1: Value is None and allow_null is True
    ref_allow_null = Reference(to="user", definitions=definitions, allow_null=True)
    assert ref_allow_null.validate(None) is None

    # Case 2: Value is None and allow_null is False (default)
    ref_no_null = Reference(to="user", definitions=definitions, allow_null=False)
    with pytest.raises(ValidationError) as excinfo:
        ref_no_null.validate(None)
    assert "May not be null" in str(excinfo.value)

    # Case 3: Value is valid and target field validates it successfully
    ref_valid = Reference(to="user", definitions=definitions)
    input_data = {"id": 1, "name": "Test"}
    mock_target_field.validate.return_value = input_data
    
    result = ref_valid.validate(input_data)
    assert result == input_data
    mock_target_field.validate.assert_called_with(input_data)

    # Case 4: Value is valid but target field raises ValidationError
    ref_error = Reference(to="user", definitions=definitions)
    mock_target_field.validate.side_effect = ValidationError(messages=["Invalid data"])
    
    with pytest.raises(ValidationError) as excinfo:
        ref_error.validate({"invalid": "data"})
    assert "Invalid data" in str(excinfo.value)
```


# LLM-generated content at query #16
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Schema_validate():
    # Mock Fields
    field_str = MagicMock(spec=Field)
    field_str.read_only = False
    field_str.has_default.return_value = False
    field_str.validate_or_error.side_effect = lambda x: (x, None)
    field_str.serialize.return_value = "val"

    field_int = MagicMock(spec=Field)
    field_int.read_only = False
    field_int.has_default.return_value = True
    field_int.get_default_value.return_value = 0
    field_int.validate_or_error.side_effect = lambda x: (x, None)
    field_int.serialize.return_value = 0

    field_readonly = MagicMock(spec=Field)
    field_readonly.read_only = True
    field_readonly.has_default.return_value = False
    field_readonly.validate_or_error.side_effect = lambda x: (x, None)

    fields = {
        "name": field_str,
        "age": field_int,
        "status": field_readonly,
    }

    schema = Schema(fields=fields)

    # 1. Test Valid Input
    input_data = {"name": "John", "age": 30, "status": "active", "extra": "ignored"}
    result = schema.validate(input_data)
    assert result == {"name": "John", "age": 30}  # status is read_only, age has default but provided

    # 2. Test Null Error (not allow_null)
    schema.allow_null = False
    with pytest.raises(ValidationError) as excinfo:
        schema.validate(None)
    assert any("May not be null" in msg.text for msg in excinfo.value.messages)

    # 3. Test Null Success (allow_null)
    schema.allow_null = True
    assert schema.validate(None) is None

    # 4. Test Type Error (not a mapping)
    with pytest.raises(ValidationError) as excinfo:
        schema.validate(["not", "a", "dict"])
    assert any("Must be an object" in msg.text for msg in excinfo.value.messages)

    # 5. Test Invalid Key (non-string key)
    with pytest.raises(ValidationError) as excinfo:
        schema.validate({123: "value"})
    assert any("All object keys must be strings" in msg.text for msg in excinfo.value.messages)

    # 6. Test Required Field Missing
    # 'name' is required because it has no default and is not read_only
    with pytest.raises(ValidationError) as excinfo:
        schema.validate({"age": 30})
    assert any("This field is required" in msg.text for msg in excinfo.value.messages)

    # 7. Test Child Field Validation Error
    child_error_msg = Message(text="Invalid string", code="error", index=[])
    error_container = MagicMock()
    error_container.messages.return_value = [Message(text="name.error", code="error", index=[])]
    field_str.validate_or_error.side_effect = lambda x: (None, error_container)

    with pytest.raises(ValidationError) as excinfo:
        schema.validate({"name": "wrong", "age": 30})
    assert any("name.error" in msg.text for msg in excinfo.value.messages)

    # 8. Test Default Value Injection
    # 'age' has a default, so if missing from input, it should be populated
    input_no_age = {"name": "John"}
    result_default = schema.validate(input_no_age)
    assert result_default["age"] == 0
```


# LLM-generated content at query #17
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Reference_validate():
    # Setup common dependencies
    definitions = Definitions()
    
    # Mock a target field that will be referenced
    mock_target_field = MagicMock(spec=Field)
    definitions["my_ref"] = mock_target_field

    # 1. Test successful validation (value passes through to target)
    ref_success = Reference(to="my_ref", definitions=definitions)
    mock_target_field.validate.return_value = "validated_value"
    assert ref_success.validate({"data": 123}) == "validated_value"
    mock_target_field.validate.assert_called_with({"data": 123})

    # 2. Test null value when allow_null is True
    ref_allow_null = Reference(to="my_ref", definitions=definitions, allow_null=True)
    assert ref_allow_null.validate(None) is None

    # 3. Test null value when allow_null is False (should raise ValidationError)
    ref_no_null = Reference(to="my_ref", definitions=definitions, allow_null=False)
    with pytest.raises(ValidationError) as excinfo:
        ref_no_null.validate(None)
    assert "May not be null." in str(excinfo.value)

    # 4. Test validation error propagation from target field
    ref_error_prop = Reference(to="my_ref", definitions=definitions)
    mock_target_field.validate.side_effect = ValidationError(messages=["Error in target"])
    with pytest.raises(ValidationError) as excinfo:
        ref_error_prop.validate({"invalid": "data"})
    assert "Error in target" in str(excinfo.value)

    # 5. Test property 'target' access
    assert ref_success.target == mock_target_field
```


# LLM-generated content at query #18
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Reference_validate():
    # Setup Mock Definitions and Target Field
    mock_definitions = MagicMock(spec=Definitions)
    mock_target_field = MagicMock(spec=Field)
    mock_definitions.__getitem__.return_value = mock_target_field
    
    # Case 1: Value is None and allow_null is True
    ref_allow_null = Reference(to="user", definitions=mock_definitions, allow_null=True)
    assert ref_allow_null.validate(None) is None

    # Case 2: Value is None and allow_null is False (Default)
    ref_not_allow_null = Reference(to="user", definitions=mock_definitions, allow_null=False)
    with pytest.raises(ValidationError) as excinfo:
        ref_not_allow_null.validate(None)
    # Assuming validation_error("null") raises ValidationError with specific message structure
    assert any("May not be null" in str(m) for m in excinfo.value.messages)

    # Case 3: Value is valid and target field validates successfully
    expected_value = {"id": 1, "name": "Test"}
    mock_target_field.validate.return_value = expected_value
    ref = Reference(to="user", definitions=mock_definitions)
    input_data = {"id": 1}
    
    result = ref.validate(input_data)
    
    assert result == expected_value
    mock_target_field.validate.assert_called_with(input_data)

    # Case 4: Value is valid but target field raises ValidationError
    mock_error_message = Message(text="Invalid type", code="type")
    mock_error = MagicMock()
    mock_error.messages.return_value = [mock_error_message]
    
    mock_target_field.validate_or_error.return_value = (None, mock_error)
    # Note: Reference.validate calls target.validate, not validate_or_error directly.
    # If we mock the behavior of validate raising the error:
    mock_target_field.validate.side_effect = ValidationError(messages=[mock_error_message])
    
    with pytest.raises(ValidationError) as excinfo:
        ref.validate({"id": "not_an_int"})
    assert len(excinfo.value.messages) == 1
```


# LLM-generated content at query #19
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Schema_validate():
    # Setup Mock Fields
    string_field = MagicMock(spec=Field)
    string_field.read_only = False
    string_field.has_default.return_value = False
    string_field.validate_or_error.side_effect = lambda x: (x, None)
    string_field.serialize.return_value = "val"

    integer_field = MagicMock(spec=Field)
    integer_field.read_only = False
    integer_field.has_default.return_value = True
    integer_field.get_default_value.return_value = 0
    integer_field.validate_or_error.side_effect = lambda x: (x, None)
    integer_field.serialize.return_value = 0

    read_only_field = MagicMock(spec=Field)
    read_only_field.read_only = True
    read_only_field.has_default.return_value = False
    read_only_field.validate_or_error.side_effect = lambda x: (x, None)

    fields = {
        "name": string_field,
        "age": integer_field,
        "status": read_only_field
    }

    schema = Schema(fields=fields)

    # 1. Test Valid Input
    valid_input = {"name": "Alice", "age": 30, "status": "active"}
    assert schema.validate(valid_input) == {"name": "Alice", "age": 30}

    # 2. Test Null value when allow_null is False (default)
    with pytest.raises(ValidationError) as excinfo:
        schema.validate(None)
    assert any("May not be null" in str(m.text) for m in excinfo.value.messages)

    # 3. Test Null value when allow_null is True
    schema_nullable = Schema(fields=fields, allow_null=True)
    assert schema_nullable.validate(None) is None

    # 4. Test Invalid Type (not a dict/mapping)
    with pytest.raises(ValidationError) as excinfo:
        schema.validate(["not", "a", "dict"])
    assert any("Must be an object" in str(m.text) for m in excinfo.value.messages)

    # 5. Test Invalid Key (non-string key)
    with pytest.raises(ValidationError) as excinfo:
        schema.validate({123: "value"})
    assert any("All object keys must be strings" in str(m.text) for m in excinfo.value.messages)

    # 6. Test Required Field Missing
    with pytest.raises(ValidationError) as excinfo:
        schema.validate({"age": 30})  # 'name' is required
    assert any("This field is required" in str(m.text) for m in excinfo.value.messages)

    # 7. Test Child Field Validation Error
    error_msg = Message(text="Invalid name", code="error", index=["name"])
    string_field.validate_or_error.side_effect = lambda x: (None, MagicMock(messages=lambda add_prefix: [error_msg]))
    
    with pytest.raises(ValidationError) as excinfo:
        schema.validate({"name": "bad", "age": 30})
    assert any("Invalid name" in str(m.text) for m in excinfo.value.messages)

    # 8. Test Default Value Injection
    # Input missing 'age', but 'age' has a default
    input_missing_default = {"name": "Bob"}
    result = schema.validate(input_missing_default)
    assert result["age"] == 0
```


# LLM-generated content at query #20
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Schema():
    # Mocking Fields
    field_required = MagicMock(spec=Field)
    field_required.read_only = False
    field_required.has_default.return_value = False

    field_optional = MagicMock(spec=Field)
    field_optional.read_only = False
    field_optional.has_default.return_value = True

    field_readonly = MagicMock(spec=Field)
    field_readonly.read_only = True
    field_readonly.has_default.return_value = False

    fields = {
        "req": field_required,
        "opt": field_optional,
        "ro": field_readonly
    }

    # Test constructor and required logic
    schema = Schema(fields=fields)

    assert schema.fields == fields
    # 'req' is required because it is not read_only and has no default
    # 'opt' is NOT required because it has a default
    # 'ro' is NOT required because it is read_only
    assert "req" in schema.required
    assert "opt" not in schema.required
    assert "ro" not in schema.required

    # Test constructor with extra kwargs (passed to Field superclass)
    schema_with_kwargs = Schema(fields=fields, allow_null=True)
    assert schema_with_kwargs.allow_null is True

    # Test empty fields
    empty_schema = Schema(fields={})
    assert empty_schema.required == []
```


# LLM-generated content at query #21
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Reference_validate():
    # Setup Mock Definitions and Target Field
    mock_definitions = MagicMock(spec=Definitions)
    mock_target_field = MagicMock(spec=Field)
    mock_definitions.__getitem__.return_value = mock_target_field
    
    # Test Case 1: Value is None and allow_null is True
    ref_allow_null = Reference(to="target_key", definitions=mock_definitions, allow_null=True)
    assert ref_allow_null.validate(None) is None

    # Test Case 2: Value is None and allow_null is False
    ref_no_null = Reference(to="target_key", definitions=mock_definitions, allow_null=False)
    with pytest.raises(ValidationError) as excinfo:
        ref_no_null.validate(None)
    # Check if the error code corresponds to "null" in Reference.errors
    assert any("May not be null." in str(m) for m in excinfo.value.messages)

    # Test Case 3: Value is valid and delegated to target field
    valid_value = {"name": "test"}
    mock_target_field.validate.return_value = valid_value
    
    ref_delegate = Reference(to="target_key", definitions=mock_definitions)
    result = ref_delegate.validate({"some": "data"})
    
    # Verify target field was called with the correct value
    mock_target_field.validate.assert_called_once_with({"some": "data"})
    assert result == valid_value

    # Test Case 4: Value is invalid and target field raises ValidationError
    mock_target_field.validate.side_effect = ValidationError(messages=[Message(text="Error", code="err")])
    ref_error = Reference(to="target_key", definitions=mock_definitions)
    
    with pytest.raises(ValidationError) as excinfo:
        ref_error.validate({"bad": "data"})
    assert len(excinfo.value.messages) == 1
```


# LLM-generated content at query #22
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Reference_validate():
    # Setup mock definitions and target field
    mock_target_field = MagicMock(spec=Field)
    definitions = Definitions({"my_ref": mock_target_field})
    
    # 1. Test valid value (delegates to target)
    ref = Reference(to="my_ref", definitions=definitions, allow_null=False)
    mock_target_field.validate.return_value = "valid_data"
    assert ref.validate({"key": "val"}) == "valid_data"
    mock_target_field.validate.assert_called_with({"key": "val"})

    # 2. Test null value when allow_null is True
    ref_allow_null = Reference(to="my_ref", definitions=definitions, allow_null=True)
    assert ref_allow_null.validate(None) is None

    # 3. Test null value when allow_null is False (should raise ValidationError)
    ref_no_null = Reference(to="my_ref", definitions=definitions, allow_null=False)
    with pytest.raises(ValidationError) as excinfo:
        ref_no_null.validate(None)
    # Checking if the error message contains 'null' (as defined in Reference.errors)
    assert any("May not be null" in str(m) for m in excinfo.value.messages)

    # 4. Test error propagation from target field
    mock_error = MagicMock()
    mock_error.messages.return_value = [MagicMock(text="Error in target", code="error")]
    
    # Mock validate_or_error to simulate a validation failure in the target
    # Note: Reference uses .validate(), which is called by validate_or_error internally in typesystem
    mock_target_field.validate.side_effect = ValidationError(messages=[MagicMock(text="Error in target", code="error")])
    
    with pytest.raises(ValidationError) as excinfo:
        ref.validate({"invalid": "data"})
    assert len(excinfo.value.messages) > 0
```


# LLM-generated content at query #23
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Schema():
    # Setup Mock Fields
    # Field 1: Required (no default, not read_only)
    field_req = MagicMock(spec=Field)
    field_req.read_only = False
    field_req.has_default.return_value = False

    # Field 2: Optional with default (has default, not read_only)
    field_opt_default = MagicMock(spec=Field)
    field_opt_default.read_only = False
    field_opt_default.has_default.return_value = True

    # Field 3: Read-only (read_only is True)
    field_readonly = MagicMock(spec=Field)
    field_readonly.read_only = True
    field_readonly.has_default.return_value = False

    fields = {
        "required_key": field_req,
        "optional_key": field_opt_default,
        "readonly_key": field_readonly,
    }

    # Test Constructor
    schema = Schema(fields=fields)

    # Assertions for initialization
    assert schema.fields == fields
    # required should only contain keys where (not read_only AND not has_default)
    # "required_key" -> True
    # "optional_key" -> False (has default)
    # "readonly_key" -> False (is read_only)
    assert "required_key" in schema.required
    assert "optional_key" not in schema.required
    assert "readonly_key" not in schema.required
    assert len(schema.required) == 1

    # Test constructor with kwargs passed to super().__init__
    schema_with_kwargs = Schema(fields=fields, allow_null=True)
    assert schema_with_kwargs.allow_null is True

    # Test empty fields
    empty_schema = Schema(fields={})
    assert empty_schema.required == []
```


# LLM-generated content at query #24
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Schema_validate():
    # Mocking Field to simulate different behaviors
    class MockField(Field):
        def __init__(self, value=None, read_only=False, default=None, allow_null=False):
            super().__init__()
            self.value = value
            self._read_only = read_mapping_attr(read_only)
            self._default = default
            self.allow_null = allow_null

        @property
        def read_only(self): return self._read_only
        
        def has_default(self): return self._default is not None
        def get_default_value(self): return self._default
        
        def validate_or_error(self, value):
            # Simplified mock validation logic
            if self.value == "fail":
                err = ValidationError(messages=[Message(text="error", code="error", index=[])])
                return None, err
            return value, None

    def read_mapping_attr(val): return val

    # Setup fields
    string_field = MockField(value="ok")
    int_field = MockField(default=10)
    read_only_field = MockField(value="readonly", read_only=True)
    fail_field = MockField(value="fail")

    fields = {
        "name": string_field,
        "age": int_field,
        "status": read_only_field,
        "error_trigger": fail_field
    }
    schema = Schema(fields=fields)

    # 1. Test successful validation
    input_data = {"name": "Alice", "age": 30, "status": "active", "error_trigger": "ok"}
    result = schema.validate(input_data)
    assert result["name"] == "Alice"
    assert result["age"] == 30
    assert result["status"] == "active"  # Included because it's in input, even if read_only

    # 2. Test default values
    input_with_missing_default = {"name": "Bob"}
    result_defaults = schema.validate(input_with_missing_default)
    assert result_defaults["age"] == 10  # Applied from default

    # 3. Test required field missing
    with pytest.raises(ValidationError) as excinfo:
        schema.validate({"age": 25})
    assert any("required" in str(m.code) for m in excinfo.value.messages)

    # 4. Test invalid type (not a dict)
    with pytest.raises(ValidationError) as excinfo:
        schema.validate(["not", "a", "dict"])
    assert any("type" in str(m.code) for m in excinfo.value.messages)

    # 5. Test null value when not allowed
    with pytest.raises(ValidationError) as excinfo:
        schema.validate(None)
    assert any("null" in str(m.code) for m in excinfo.value.messages)

    # 6. Test invalid key type (key is an int)
    with pytest.raises(ValidationError) as excinfo:
        schema.validate({123: "value"})
    assert any("invalid_key" in str(m.code) for m in excinfo.value.messages)

    # 7. Test child field validation error
    with pytest.raises(ValidationError) as excinfo:
        schema.validate({"name": "Alice", "error_trigger": "fail"})
    assert any("error" in str(m.code) for m in excinfo.value.messages)

    # 8. Test allow_null=True
    nullable_schema = Schema(fields={"name": MockField(allow_null=True)})
    assert nullable_schema.validate(None) is None
```


# LLM-generated content at query #25
#--------------------------

```python
import pytest

def test_Reference_validate():
    # Mocking a Field/Schema to act as target
    class MockField(Field):
        def __init__(self, value_to_return, **kwargs):
            super().__init__(**kwargs)
            self.value_to_return = value_to_return
            self.valid_data = None

        def validate(self, value):
            if value == "trigger_error":
                raise ValidationError(messages=[Message(text="error", code="err")])
            return self.value_to_return

    # Setup definitions and Reference
    target_field = MockField(value_to_return={"id": 1})
    definitions = Definitions({"my_ref": target_field})
    
    # Case 1: Valid input
    ref_valid = Reference(to="my_ref", definitions=definitions)
    assert ref_valid.validate({"id": 1}) == {"id": 1}

    # Case 2: Null value allowed
    ref_allow_null = Reference(to="my_ref", definitions=definitions, allow_null=True)
    assert ref_allow_null.validate(None) is None

    # Case 3: Null value not allowed (should raise ValidationError with 'null' error)
    ref_no_null = Reference(to="my_ref", definitions=definitions, allow_null=False)
    with pytest.raises(ValidationError) as excinfo:
        ref_no_null.validate(None)
    assert any("May not be null" in str(m.text) for m in excinfo.value.messages)

    # Case 4: Value triggers error in target field
    ref_error = Reference(to="my_ref", definitions=definitions)
    with pytest.raises(ValidationError) as excinfo:
        ref_error.validate("trigger_error")
    assert len(excinfo.value.messages) > 0
```


# LLM-generated content at query #26
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Schema():
    # Mocking Field objects to control required/read_only/has_default behavior
    field_req = MagicMock(spec=Field)
    field_req.read_only = False
    field_req.has_default.return_value = False

    field_opt = MagicMock(spec=Field)
    field_opt.read_only = False
    field_opt.has_default.return_value = True

    field_ro = MagicMock(spec=Field)
    field_ro.read_only = True
    field_ro.has_default.return_value = False

    fields = {
        "required_field": field_req,
        "optional_field": field_opt,
        "readonly_field": field_ro,
    }

    # Test initialization and determination of required fields
    schema = Schema(fields=fields)

    assert schema.fields == fields
    # Only 'required_field' should be in the required list 
    # because 'optional_field' has a default and 'readonly_field' is read_only
    assert "required_field" in schema.required
    assert "optional_field" not in schema.required
    assert "readonly_field" not in schema.required
    assert len(schema.required) == 1

    # Test initialization with empty fields
    empty_schema = Schema(fields={})
    assert empty_schema.required == []

    # Test initialization with kwargs passed to super().__init__
    schema_with_kwargs = Schema(fields=fields, allow_null=True)
    assert schema_with_kwargs.allow_null is True
```


# LLM-generated content at query #27
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Schema_validate():
    # Setup mock fields
    field_str = MagicMock(spec=Field)
    field_str.read_only = False
    field_str.has_default.return_value = False
    field_str.validate_or_error.side_effect = lambda x: (x, None)
    field_str.serialize.return_value = "val"

    field_int = MagicMock(spec=Field)
    field_int.read_only = False
    field_int.has_default.return_value = True
    field_int.get_default_value.return_value = 0
    field_int.validate_or_error.side_effect = lambda x: (x, None)
    field_int.serialize.return_value = 0

    field_read_only = MagicMock(spec=Field)
    field_read_only.read_only = True
    field_read_only.has_default.return_value = False
    field_read_only.validate_or_error.side_effect = lambda x: (x, None)

    fields = {
        "name": field_str,
        "age": field_int,
        "metadata": field_read_only
    }

    schema = Schema(fields=fields)

    # Test 1: Success Case (All valid)
    input_data = {"name": "Alice", "age": 30, "metadata": "extra"}
    result = schema.validate(input_data)
    assert result == {"name": "Alice", "age": 30}
    # metadata is skipped because it's read_only

    # Test 2: Success Case (Defaults applied)
    input_data_minimal = {"name": "Bob"}
    result = schema.validate(input_data_minimal)
    assert result == {"name": "Bob", "age": 0}

    # Test 3: Failure - Null value when not allowed
    with pytest.raises(ValidationError) as excinfo:
        schema.validate(None)
    assert any("May not be null" in str(m.text) for m in excinfo.value.messages)

    # Test 4: Success - Null value when allowed
    schema_nullable = Schema(fields=fields, allow_null=True)
    assert schema_nullable.validate(None) is None

    # Test 5: Failure - Not an object (type error)
    with pytest.raises(ValidationError) as excinfo:
        schema.validate(["not", "a", "dict"])
    assert any("Must be an object" in str(m.text) for m in excinfo.value.messages)

    # Test 6: Failure - Invalid key type (key is not a string)
    with pytest.raises(ValidationError) as excinfo:
        schema.validate({123: "value"})
    assert any("All object keys must be strings" in str(m.text) for m in excinfo.value.messages)

    # Test 7: Failure - Required field missing
    with pytest.raises(ValidationError) as excinfo:
        schema.validate({"age": 25})
    assert any("This field is required" in str(m.text) for m in excinfo.value.messages)

    # Test 8: Failure - Child field validation error (bubbling up)
    field_err = MagicMock(spec=Field)
    field_err.read_only = False
    field_err.has_default.return_value = False
    
    # Mocking the error structure returned by validate_or_error
    error_msg = Message(text="Invalid age", code="error", index=[])
    class ErrorContainer:
        def messages(self, add_prefix=None):
            msg = error_msg
            if add_prefix:
                # Simulate prefixing behavior if necessary
                pass
            return [msg]
    
    field_err.validate_or_error.return_value = (None, ErrorContainer())

    schema_with_error = Schema(fields={"age": field_err})
    with pytest.raises(ValidationError) as excinfo:
        schema_with_error.validate({"age": "not_a_number"})
    assert any("Invalid age" in str(m.text) for m in excinfo.value.messages)
```


