####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
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

    fields_dict = {
        "req": field_required,
        "opt": field_optional,
        "ro": field_readonly
    }

    # Test 1: Constructor correctly identifies required fields
    # A field is required if it is NOT read_only AND does NOT have a default
    schema = Schema(fields=fields_dict)
    assert "req" in schema.required
    assert "opt" not in schema.required
    assert "ro" not in schema.required
    assert len(schema.required) == 1

    # Test 2: Constructor handles empty fields
    schema_empty = Schema(fields={})
    assert schema_empty.required == []

    # Test 3: Constructor handles kwargs (passed to super().__init__)
    schema_with_kwargs = Schema(fields=fields_dict, allow_null=True)
    assert schema_with_kwargs.allow_null is True

    # Test 4: Verifying field mapping integrity
    assert schema.fields == fields_dict
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Schema_validate():
    # Setup Mock Fields
    # Field 1: Required, no default
    field_req = MagicMock(spec=Field)
    field_req.read_only = False
    field_req.has_default.return_value = False
    field_req.validate_or_error.side_effect = lambda x: (x, None)
    field_req.serialize.side_effect = lambda x: x

    # Field 2: Optional, with default
    field_opt = MagicMock(spec=Field)
    field_opt.read_only = False
    field_opt.has_default.return_value = True
    field_opt.get_default_value.return_value = "default_val"
    field_opt.validate_or_error.side_effect = lambda x: (x, None)
    field_opt.serialize.side_effect = lambda x: x

    # Field 3: Read-only
    field_ro = MagicMock(spec=Field)
    field_ro.read_only = True
    field_ro.has_default.return_value = False
    field_ro.validate_or_error.side_effect = lambda x: (x, None)
    field_ro.serialize.side_effect = lambda x: x

    fields = {
        "req_field": field_req,
        "opt_field": field_opt,
        "ro_field": field_ro
    }

    schema = Schema(fields=fields, allow_null=False)

    # 1. Test Success Case: All valid inputs
    valid_input = {"req_field": "val1", "opt_field": "val2", "ro_field": "val3"}
    result = schema.validate(valid_input)
    assert result == {"req_field": "val1", "opt_field": "val2", "ro_field": "val3"}

    # 2. Test Success Case: Missing optional field uses default
    input_missing_opt = {"req_field": "val1", "ro_field": "val3"}
    result = schema.validate(input_missing_opt)
    assert result["opt_field"] == "default_val"
    assert result["req_field"] == "val1"

    # 3. Test Error: Value is None and allow_null is False
    with pytest.raises(ValidationError) as excinfo:
        schema.validate(None)
    assert any("May not be null." in str(m) for m in excinfo.value.messages)

    # 4. Test Error: Value is not a mapping (type error)
    with pytest.raises(ValidationError) as excinfo:
        schema.validate(["not", "a", "dict"])
    assert any("Must be an object." in str(m) for m in excinfo.value.messages)

    # 5. Test Error: Invalid key type (key is not a string)
    with pytest.raises(ValidationError) as excinfo:
        schema.validate({123: "value", "req_field": "val1"})
    assert any("All object keys must be strings." in str(m) for m in excinfo.value.messages)

    # 6. Test Error: Required field is missing
    with pytest.raises(ValidationError) as excinfo:
        schema.validate({"opt_field": "val2"})
    assert any("This field is required." in str(m) for m in excinfo.value.messages)

    # 7. Test Error: Child field validation fails
    # Mocking child field error
    error_msg = Message(text="Child error", code="child_err", index=[])
    error_mock_field = MagicMock(spec=Field)
    error_mock_field.read_only = False
    error_mock_field.has_default.return_value = False
    error_mock_field.validate_or_error.return_value = (None, [error_msg])
    
    schema_with_error = Schema(fields={"child": error_mock_field})
    with pytest.raises(ValidationError) as excinfo:
        schema_with_error.validate({"child": "bad_value"})
    # Check if the error message contains the prefix (the key name)
    assert any("child: Child error" in str(m) for m in excinfo.value.messages)

    # 8. Test Success Case: allow_null is True
    schema_nullable = Schema(fields=fields, allow_null=True)
    assert schema_nullable.validate(None) is None
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Schema():
    # Mocking Field for dependencies
    def create_mock_field(read_only=False, has_default=False):
        field = MagicMock(spec=Field)
        field.read_only = read_only
        field.has_default.return_value = has_default
        return field

    # Case 1: Required fields (not read_only and no default)
    fields_req = {
        "req_field": create_mock_field(read_only=False, has_default=False),
        "opt_field": create_mock_field(read_only=True, has_default=False),
    }
    schema_req = Schema(fields=fields_req)
    assert "req_field" in schema_req.required
    assert "opt_field" not in schema_req.required

    # Case 2: Optional fields (has default)
    fields_def = {
        "def_field": create_mock_field(read_only=False, has_default=True),
    }
    schema_def = Schema(fields=fields_def)
    assert "def_field" not in schema_def.required

    # Case 3: Optional fields (read_only)
    fields_ro = {
        "ro_field": create_mock_field(read_only=True, has_default=False),
    }
    schema_ro = Schema(fields=fields_ro)
    assert "ro_field" not in schema_ro.required

    # Case 4: Empty fields
    schema_empty = Schema(fields={})
    assert schema_empty.required == []

    # Case 5: Verify kwargs are passed to super().__init__
    # We check this by seeing if allow_null is set on the instance
    schema_kwargs = Schema(fields={}, allow_null=True)
    assert schema_kwargs.allow_null is True
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Schema_validate():
    # Mocking Field behavior
    def create_mock_field(name, required=False, default=None, value="valid"):
        field = MagicMock(spec=Field)
        field.read_only = False
        field.has_default.return_value = default is not None
        field.get_default_value.return_value = default
        # validate_or_error returns (value, error_list)
        field.validate_or_error.return_value = (value, [])
        field.serialize.return_value = value
        return field

    # Setup Schema with various fields
    fields = {
        "name": create_mock_field("name"),
        "age": create_mock_field("age", default=18),
        "read_only_field": create_mock_field("read_only_field")
    }
    fields["read_only_field"].read_only = True
    
    schema = Schema(fields=fields)

    # 1. Test valid input
    valid_input = {"name": "John", "age": 30}
    # Note: 'age' is in input, so default shouldn't be used. 
    # 'read_only_field' is ignored in validation loop because it's read_only.
    result = schema.validate(valid_input)
    assert result["name"] == "John"
    assert "age" in result

    # 2. Test default value injection
    input_no_age = {"name": "John"}
    result_with_default = schema.validate(input_no_age)
    assert result_with_default["age"] == 18

    # 3. Test null error (not allowed)
    with pytest.raises(ValidationError) as excinfo:
        schema.validate(None)
    assert any("May not be null" in str(m.text) for m_msg in excinfo.value.messages for m in [m_msg])

    # 4. Test type error (not a dict)
    with pytest.raises(ValidationError) as excinfo:
        schema.validate(["not", "a", "dict"])
    assert any("Must be an object" in str(m.text) for m_msg in excinfo.value.messages for m in [m_msg])

    # 5. Test invalid key type (keys must be strings)
    with pytest.raises(ValidationError) as excinfo:
        schema.validate({123: "integer key"})
    assert any("All object keys must be strings" in str(m.text) for m_msg in excinfo.value.messages for m in [m_msg])

    # 6. Test required field missing
    # 'name' is required because it has no default and is not read_only
    with pytest.raises(ValidationError) as excinfo:
        schema.validate({"age": 25})
    assert any("This field is required" in str(m.text) for m_msg in excinfo.value.messages for m in [m_msg])

    # 7. Test child field validation error propagation
    error_field = create_mock_field("error_field")
    error_field.validate_or_error.return_value = (None, [Message(text="Child error", code="child_err", index=[])])
    
    error_schema = Schema(fields={"error_field": error_field})
    with pytest.raises(ValidationError) as excinfo:
        error_schema.validate({"error_field": "bad_data"})
    # Check if the error message is prefixed with the key name
    assert any("error_field: Child error" in str(m.text) for m_msg in excinfo.value.messages for m in [m_msg])

    # 8. Test allow_null = True
    schema_nullable = Schema(fields={"name": create_mock_field("name")}, allow_null=True)
    assert schema_nullable.validate(None) is None
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Reference_validate():
    # Mocking the Definitions object
    definitions = MagicMock(spec=Definitions)
    
    # Mocking the target Field
    mock_target_field = MagicMock(spec=Field)
    definitions.__getitem__.return_value = mock_target_field
    
    # Case 1: Value is None and allow_null is True
    ref_allow_null = Reference(to="target_key", definitions=definitions, allow_null=True)
    assert ref_allow_null.validate(None) is None
    
    # Case 2: Value is None and allow_null is False
    ref_no_null = Reference(to="target_key", definitions=definitions, allow_null=False)
    with pytest.raises(ValidationError) as excinfo:
        ref_no_null.validate(None)
    # Checking if the error message corresponds to the "null" key in Reference.errors
    assert any("May not be null" in str(msg) for msg in excinfo.value.messages)

    # Case 3: Value is provided, validate calls target.validate
    valid_value = {"id": 1}
    mock_target_field.validate.return_value = valid_value
    
    ref_valid = Reference(to="target_key", definitions=definitions)
    result = ref_valid.validate(valid_value)
    
    # Verify target.validate was called with the correct value
    mock_target_field.validate.assert_called_with(valid_value)
    assert result == valid_value

    # Case 4: Value is provided, but target.validate raises ValidationError
    mock_target_field.validate.side_effect = ValidationError(messages=["Invalid value"])
    
    ref_invalid = Reference(to="target_key", definitions=definitions)
    with pytest.raises(ValidationError) as excinfo:
        ref_invalid.validate(valid_value)
    assert "Invalid value" in str(excinfo.value.messages)
```


# LLM-generated content at query #6
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

    # Test Constructor: Required fields identification
    schema = Schema(fields=fields)
    assert "req" in schema.required
    assert "opt" not in schema.required
    assert "ro" not in schema.required

    # Test Constructor: Empty fields
    schema_empty = Schema(fields={})
    assert schema_empty.required == []

    # Test Constructor: All required
    field_req_2 = MagicMock(spec=Field)
    field_req_2.read_only = False
    field_req_2.has_default.return_value = False
    schema_all_req = Schema(fields={"a": field_req_2, "b": field_req_2})
    assert len(schema_all_req.required) == 2
    assert "a" in schema_all_req.required
    assert "b" in schema_all_req.required

    # Test Constructor: Kwargs pass-through (e.g., allow_null)
    schema_kwargs = Schema(fields=fields, allow_null=True)
    assert schema_kwargs.allow_null is True
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Schema_validate():
    # Setup Mock Fields
    # Field 1: Required, no default
    field_req = MagicMock(spec=Field)
    field_req.read_only = False
    field_req.has_default.return_value = False
    field_req.validate_or_error.return_value = ("val1", None)
    field_req.get_default_value.return_value = None

    # Field 2: Not required, has default
    field_def = MagicMock(spec=Field)
    field_def.read_only = False
    field_def.has_default.return_value = True
    field_def.validate_or_error.return_value = ("val2", None)
    field_def.get_default_value.return_value = "default_val"

    # Field 3: Read only
    field_ro = MagicMock(spec=Field)
    field_ro.read_only = True
    field_ro.has_default.return_value = False

    # Field 4: Child field that will fail validation
    field_fail = MagicMock(spec=Field)
    field_fail.read_only = False
    field_fail.has_default.return_value = False
    # Mocking the error structure returned by validate_or_error
    error_msg = Message(text="Child error", code="child_err")
    class MockError:
        def messages(self, add_prefix):
            return [Message(text=f"{add_prefix}.Child error", code="child_err", index=[])]
    
    field_fail.validate_or_error.return_value = (None, MockError())

    fields = {
        "req_field": field_req,
        "def_field": field_def,
        "ro_field": field_ro,
        "fail_field": field_fail
    }

    schema = Schema(fields=fields)

    # 1. Test Null value when allow_null is False
    schema.allow_null = False
    with pytest.raises(ValidationError) as excinfo:
        schema.validate(None)
    assert any("May not be null" in str(m.text) for m in excinfo.value.messages)

    # 2. Test Null value when allow_null is True
    schema.allow_null = True
    assert schema.validate(None) is None

    # 3. Test invalid type (not a dict/mapping)
    with pytest.raises(ValidationError) as excinfo:
        schema.validate("not a dict")
    assert any("Must be an object" in str(m.text) for m in excinfo.value.messages)

    # 4. Test invalid key type (key is an integer)
    with pytest.raises(ValidationError) as excinfo:
        schema.validate({123: "value"})
    assert any("All object keys must be strings" in str(m.text) for m in excinfo.value.messages)

    # 5. Test missing required field
    with pytest.raises(ValidationError) as excinfo:
        schema.validate({"def_field": "some_val"})
    assert any("This field is required" in str(m.text) for m in excinfo.value.messages)

    # 6. Test successful validation with defaults and child errors
    # We include 'fail_field' in input to trigger the child error logic
    input_data = {
        "req_field": "val1",
        "fail_field": "bad_data"
    }
    with pytest.raises(ValidationError) as excinfo:
        schema.validate(input_data)
    # Check if the error prefixing worked
    assert any("fail_field.Child error" in str(m.text) for m in excinfo.value.messages)

    # 7. Test full success
    # We use a clean setup where no errors are triggered
    field_req.validate_or_error.return_value = ("val1", None)
    field_def.validate_or_error.return_value = ("val2", None)
    field_fail.validate_or_error.return_value = ("val4", None)
    
    success_input = {
        "req_field": "val1",
        "def_field": "val2",
        "fail_field": "val4"
    }
    result = schema.validate(success_input)
    
    # 'def_field' should be present (from input)
    assert result["req_field"] == "val1"
    assert result["def_field"] == "val2"
    assert result["fail_field"] == "val4"
    # 'ro_field' should not be in result because it was skipped in the loop
    assert "ro_field" not in result

    # 8. Test default value injection
    # Provide input without 'def_field'
    input_with_defaults = {
        "req_field": "val1",
        "fail_field": "val4"
    }
    result_with_defaults = schema.validate(input_with_defaults)
    assert result_with_defaults["def_field"] == "default_val"
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Schema_validate():
    # Setup Mock Fields
    # Field 1: Required, no default
    field_req = MagicMock(spec=Field)
    field_req.read_only = False
    field_req.has_default.return_value = False
    field_req.validate_or_error.return_value = ("val1", None)
    field_req.get_default_value.return_value = None

    # Field 2: Optional, with default
    field_opt = MagicMock(spec=Field)
    field_opt.read_only = False
    field_opt.has_default.return_value = True
    field_opt.validate_or_error.return_value = ("val2", None)
    field_opt.get_default_value.return_value = "default_val"

    # Field 3: Read-only
    field_ro = MagicMock(spec=Field)
    field_ro.read_only = True
    field_ro.has_default.return_value = False
    field_ro.validate_or_error.return_value = ("val3", None)

    # Field 4: Child field that fails validation
    field_fail = MagicMock(spec=Field)
    field_fail.read_only = False
    field_fail.has_default.return_value = False
    # Mocking the error structure returned by child field
    error_msg = Message(text="Child error", code="error_code")
    class MockError:
        def messages(self, add_prefix):
            return [Message(text=f"{add_prefix}.error", code="error_code", index=[])]
    
    field_fail.validate_or_error.return_value = (None, MockError())

    fields = {
        "required_field": field_req,
        "optional_field": field_opt,
        "readonly_field": field_ro,
        "failing_field": field_fail
    }

    schema = Schema(fields=fields)

    # --- Test Case 1: Valid input ---
    input_valid = {"required_field": "val1", "optional_field": "val2", "readonly_field": "val3"}
    result = schema.validate(input_valid)
    assert result["required_field"] == "val1"
    assert result["optional_field"] == "val2"
    assert result["readonly_field"] == "val3"

    # --- Test Case 2: Null value (not allowed) ---
    with pytest.raises(ValidationError) as excinfo:
        schema.validate(None)
    assert any("May not be null" in str(m.text) for m in excinfo.value.messages)

    # --- Test Case 3: Not an object (type error) ---
    with pytest.raises(ValidationError) as excinfo:
        schema.validate(["not", "a", "dict"])
    assert any("Must be an object" in str(m.text) for m in excinfo.value.messages)

    # --- Test Case 4: Invalid key type (non-string key) ---
    input_invalid_key = {123: "value"}
    with pytest.raises(ValidationError) as excinfo:
        schema.validate(input_invalid_key)
    assert any("All object keys must be strings" in str(m.text) for m in excinfo.value.messages)

    # --- Test Case 5: Missing required field ---
    input_missing_req = {"optional_field": "val2"}
    with pytest.raises(ValidationError) as excinfo:
        schema.validate(input_missing_req)
    assert any("This field is required" in str(m.text) for m in excinfo.value.messages)

    # --- Test Case 6: Default value injection ---
    input_with_defaults = {"required_field": "val1"}
    result_defaults = schema.validate(input_with_defaults)
    assert result_defaults["optional_field"] == "default_val"

    # --- Test Case 7: Child field validation error propagation ---
    input_child_error = {
        "required_field": "val1",
        "failing_field": "bad_data"
    }
    with pytest.raises(ValidationError) as excinfo:
        schema.validate(input_child_error)
    # Check if the prefix 'failing_field.' was added to the error message
    assert any("failing_field.error" in str(m.text) for m in excinfo.value.messages)
```


# LLM-generated content at query #9
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
    schema = Schema(fields=fields, allow_null=True)

    # Assertions
    assert schema.fields == fields
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

    # Test kwargs propagation
    schema_with_kwargs = Schema(fields=fields, allow_null=False)
    assert schema_with_kwargs.allow_null is False
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Reference_validate():
    # Setup mock definitions and a mock target field
    mock_target_field = MagicMock(spec=Field)
    definitions = Definitions({"user": mock_target_field})
    
    # 1. Test valid input (delegates to target.validate)
    ref = Reference(to="user", definitions=definitions)
    input_value = {"id": 1, "name": "John"}
    mock_target_field.validate.return_value = {"id": 1, "name": "John"}
    
    result = ref.validate(input_value)
    
    assert result == {"id": 1, "name": "John"}
    mock_target_field.validate.assert_called_with(input_value)

    # 2. Test null value when allow_null is True
    ref_allow_null = Reference(to="user", definitions=definitions, allow_null=True)
    assert ref_allow_null.validate(None) is None

    # 3. Test null value when allow_null is False (should raise ValidationError)
    ref_no_null = Reference(to="user", definitions=definitions, allow_null=False)
    with pytest.raises(ValidationError) as excinfo:
        ref_no_null.validate(None)
    
    # Check if the error message corresponds to the 'null' key in Reference.errors
    # Note: Reference.validation_error is a method of Field that wraps the error string
    assert any("May not be null" in str(msg.text) for msg in excinfo.value.messages)

    # 4. Test error delegation (when target.validate raises ValidationError)
    error_msg = Message(text="Invalid field", code="error_code")
    mock_target_field.validate.side_effect = ValidationError(messages=[error_msg])
    
    with pytest.raises(ValidationError) as excinfo:
        ref.validate({"invalid": "data"})
    
    assert len(excinfo.value.messages) == 1
    assert excinfo.value.messages[0].text == "Invalid field"
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Reference_validate():
    # Setup Mock Definitions
    mock_definitions = MagicMock(spec=Definitions)
    
    # Setup Mock Target Field
    mock_target_field = MagicMock(spec=Field)
    mock_definitions.__getitem__.return_value = mock_target_field
    
    # 1. Test Valid Value (delegates to target)
    mock_target_field.validate.return_value = "valid_data"
    ref_field = Reference(to="user_schema", definitions=mock_definitions)
    assert ref_field.validate({"id": 1}) == "valid_data"
    mock_target_field.validate.assert_called_with({"id": 1})

    # 2. Test Null Value with allow_null=True
    ref_field_allow_null = Reference(to="user_schema", definitions=mock_definitions, allow_null=True)
    assert ref_field_allow_null.validate(None) is None

    # 3. Test Null Value with allow_null=False (Default)
    ref_field_no_null = Reference(to="user_schema", definitions=mock_definitions, allow_null=False)
    with pytest.raises(ValidationError) as excinfo:
        ref_field_no_null.validate(None)
    # Check if the error message corresponds to the 'null' key in Reference.errors
    assert any("May not be null." in str(msg) for msg in excinfo.value.messages)

    # 4. Test Validation Error from Target
    mock_target_field.validate.side_effect = ValidationError(messages=["Target error"])
    with pytest.raises(ValidationError) as excinfo:
        ref_field.validate({"invalid": "data"})
    assert "Target error" in str(excinfo.value.messages)
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Schema():
    # Mocking Field for dependency injection
    def create_mock_field(read_only=False, has_default=False):
        field = MagicMock(spec=Field)
        field.read_only = read_only
        field.has_default.return_value = has_default
        return field

    # Case 1: All fields are required (no defaults, not read_only)
    fields_required = {
        "name": create_mock_field(read_only=False, has_default=False),
        "age": create_mock_field(read_only=False, has_default=False),
    }
    schema_req = Schema(fields=fields_required)
    assert schema_req.required == ["name", "age"]

    # Case 2: Some fields are read_only
    fields_mixed = {
        "id": create_mock_field(read_only=True, has_default=False),
        "name": create_mock_field(read_only=False, has_default=False),
    }
    schema_mixed = Schema(fields=fields_mixed)
    assert "id" not in schema_mixed.required
    assert "name" in schema_mixed.required

    # Case 3: Some fields have defaults
    fields_defaults = {
        "name": create_mock_field(read_only=False, has_default=False),
        "role": create_mock_field(read_only=False, has_default=True),
    }
    schema_def = Schema(fields=fields_defaults)
    assert "name" in schema_def.required
    assert "role" not in schema_def.required

    # Case 4: All fields are optional (read_only or has_default)
    fields_optional = {
        "id": create_mock_field(read_only=True, has_default=False),
        "role": create_mock_field(read_only=False, has_default=True),
    }
    schema_opt = Schema(fields=fields_optional)
    assert schema_opt.required == []

    # Case 5: Verify kwargs are passed to super (Field)
    schema_kwargs = Schema(fields={}, allow_null=True)
    assert schema_kwargs.allow_null is True
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Schema_validate():
    # Mock Field setup
    def create_mock_field(required=False, default=None, read_only=False):
        field = MagicMock(spec=Field)
        field.required = required
        field.read_only = read_only
        field.has_default.return_value = default is not None
        field.get_default_value.return_value = default
        # Mock validate_or_error to return (value, None) for success
        field.validate_or_error.side_effect = lambda x: (x, None)
        return field

    # 1. Test Success Case
    field_a = create_mock_field()
    field_b = create_mock_field(default="default_val")
    schema = Schema(fields={"a": field_a, "b": field_b})
    
    input_data = {"a": 1, "b": 2}
    assert schema.validate(input_data) == {"a": 1, "b": 2}
    
    # Test default value injection
    input_data_missing_b = {"a": 1}
    assert schema.validate(input_data_missing_b) == {"a": 1, "b": "default_val"}

    # 2. Test Null Violation
    schema_not_null = Schema(fields={"a": field_a}, allow_null=False)
    with pytest.raises(ValidationError) as excinfo:
        schema_not_null.validate(None)
    assert any("May not be null" in str(m) for m in excinfo.value.messages)

    # 3. Test Type Violation (not a mapping)
    schema_obj = Schema(fields={"a": field_a})
    with pytest.raises(ValidationError) as excinfo:
        schema_obj.validate([1, 2, 3])
    assert any("Must be an object" in str(m) for m in excinfo.value.messages)

    # 4. Test Invalid Key Type (non-string keys)
    schema_keys = Schema(fields={"a": field_a})
    with pytest.raises(ValidationError) as excinfo:
        schema_keys.validate({123: "value"})
    assert any("All object keys must be strings" in str(m) for m in excinfo.value.messages)
    assert excinfo.value.messages[0].index == [123]

    # 5. Test Required Field Missing
    field_req = create_mock_field(required=True)
    schema_req = Schema(fields={"required_key": field_req})
    with pytest.raises(ValidationError) as excinfo:
        schema_req.validate({"other_key": 1})
    assert any("This field is required" in str(m) for m in excinfo.value.messages)
    assert excinfo.value.messages[0].index == ["required_key"]

    # 6. Test Child Field Validation Error
    field_fail = create_mock_field()
    # Mocking child error: validate_or_error returns (None, [error_message])
    error_msg = Message(text="Child Error", code="child_err", index=["sub_key"])
    field_fail.validate_or_error.side_effect = lambda x: (None, [error_msg])
    
    schema_child_err = Schema(fields={"child": field_fail})
    with pytest.raises(ValidationError) as excinfo:
        schema_child_err.validate({"child": "bad_data"})
    # Check if the error message is prefixed with the key name
    assert any("child: Child Error" in str(m) for m in excinfo.value.messages)

    # 7. Test Read Only Field (should be skipped in validation logic)
    field_ro = create_mock_field(read_only=True)
    schema_ro = Schema(fields={"ro": field_ro})
    # Should not trigger child validation even if data is provided
    assert schema_ro.validate({"ro": "some_val"}) == {}
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Schema():
    # Mocking Field objects
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

    # Assertions
    assert schema.fields == fields
    # 'req' is required because read_only=False and has_default=False
    # 'opt' is not required because has_default=True
    # 'ro' is not required because read_only=True
    assert "req" in schema.required
    assert "opt" not in schema.required
    assert "ro" not in schema.required
    assert len(schema.required) == 1

    # Test Constructor with additional kwargs (passed to super().__init__)
    schema_with_kwargs = Schema(fields=fields, allow_null=True)
    assert schema_with_kwargs.allow_null is True
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Schema():
    # Mock Fields
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

    # Test 1: Constructor correctly identifies required fields
    schema = Schema(fields=fields)
    assert "req" in schema.required
    assert "opt" not in schema.required
    assert "ro" not in schema.required

    # Test 2: Constructor handles empty fields
    schema_empty = Schema(fields={})
    assert schema_empty.required == []

    # Test 3: Constructor preserves kwargs (passed to super().__init__)
    schema_kwargs = Schema(fields=fields, allow_null=True)
    assert schema_kwargs.allow_null is True

    # Test 4: Check field mapping integrity
    assert schema.fields == fields
```


####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Schema_validate():
    # Mock Fields
    def create_mock_field(required=False, default=None, read_only=False):
        field = MagicMock(spec=Field)
        field.read_only = read_only
        field.has_default.return_value = default is not None
        field.get_default_value.return_value = default
        # For child validation logic
        field.validate_or_error.return_value = (None, None)
        return field

    # Test Case 1: Null value not allowed
    schema_not_nullable = Schema(fields={"name": create_mock_field()})
    schema_not_nullable.allow_null = False
    with pytest.raises(ValidationError) as excinfo:
        schema_not_nullable.validate(None)
    assert any("May not be null" in str(m) for m in excinfo.value.messages)

    # Test Case 2: Null value allowed
    schema_nullable = Schema(fields={"name": create_mock_field()})
    schema_nullable.allow_null = True
    assert schema_nullable.validate(None) is None

    # Test Case 3: Invalid type (not a dict/mapping)
    schema = Schema(fields={"name": create_mock_field()})
    with pytest.raises(ValidationError) as excinfo:
        schema.validate(["not", "a", "dict"])
    assert any("Must be an object" in str(m) for m in excinfo.value.messages)

    # Test Case 4: Invalid key type (key is not a string)
    schema = Schema(fields={"name": create_mock_field()})
    with pytest.raises(ValidationError) as excinfo:
        schema.validate({123: "value"})
    assert any("All object keys must be string" in str(m) for m in excinfo.value.messages)

    # Test Case 5: Missing required field
    name_field = create_mock_field() # Not read_only, no default -> Required
    schema = Schema(fields={"name": name_field})
    with pytest.raises(ValidationError) as excinfo:
        schema.validate({"age": 30})
    assert any("This field is required" in str(m) for m in excinfo.value.messages)

    # Test Case 6: Successful validation with defaults and child validation
    child_field = create_mock_field()
    child_field.validate_or_error.return_value = ("valid_child", None)
    
    default_field = create_mock_field(default="default_val")
    
    schema = Schema(fields={
        "name": create_mock_field(), # required
        "age": create_mock_field(default=25), # optional with default
        "metadata": child_field, # child validation
        "read_only_field": create_mock_field(read_only=True)
    })

    input_data = {
        "name": "John",
        "metadata": {"key": "val"},
        "read_only_field": "should_be_ignored"
    }
    
    result = schema.validate(input_data)
    
    assert result["name"] == "John"
    assert result["age"] == 25  # Filled from default
    assert result["metadata"] == "valid_child"
    assert "read_only_field" not in result

    # Test Case 7: Child field validation error propagation
    error_msg = Message(text="Child error", code="child_err", index=[])
    error_obj = MagicMock()
    error_obj.messages.return_value = [Message(text="child_err_prefix:Child error", code="child_err", index=[])]
    
    child_field.validate_or_error.return_value = (None, error_obj)
    
    schema = Schema(fields={"metadata": child_field})
    with pytest.raises(ValidationError) as excinfo:
        schema.validate({"metadata": {"bad": "data"}})
    assert any("child_err_prefix:Child error" in str(m) for m in excinfo.value.messages)
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Schema_validate():
    # Mock Fields
    def create_mock_field(required=False, default=None, read_only=False):
        field = MagicMock(spec=Field)
        field.read_only = read_only
        field.has_default.return_value = default is not None
        field.get_default_value.return_value = default
        # Mock validate_or_error to return (value, None) for success
        field.validate_or_error.return_value = (default if default is not None else "", None)
        return field

    # Setup 1: Basic Valid Schema
    field_a = create_mock_field()
    field_b = create_mock_field(default="default_val")
    schema = Schema(fields={"a": field_a, "b": field_b})
    
    # Test successful validation
    assert schema.validate({"a": "val", "b": "val"}) == {"a": "val", "b": "val"}
    # Test default value injection
    assert schema.validate({"a": "val"}) == {"a": "val", "b": "default_val"}

    # Setup 2: Null handling
    schema_nullable = Schema(fields={"a": field_a}, allow_null=True)
    assert schema_nullable.validate(None) is None
    
    schema_non_nullable = Schema(fields={"a": field_a}, allow_null=False)
    with pytest.raises(ValidationError) as excinfo:
        schema_non_nullable.validate(None)
    assert any("May not be null" in str(m.text) for m in excinfo.value.messages)

    # Setup 3: Type Error (Not a mapping)
    with pytest.raises(ValidationError) as excinfo:
        schema.validate(["not", "a", "dict"])
    assert any("Must be an object" in str(m.text) for m in excinfo.value.messages)

    # Setup 4: Invalid Keys (Non-string keys)
    with pytest.raises(ValidationError) as excinfo:
        schema.validate({123: "value"})
    assert any("All object keys must be string" in str(m.text) for m in excinfo.value.messages)

    # Setup 5: Required Fields Missing
    schema_req = Schema(fields={"a": field_a}) # 'a' is required because it has no default
    with pytest.raises(ValidationError) as excinfo:
        schema_req.validate({})
    assert any("This field is required" in str(m.text) for m in excinfo.value.messages)

    # Setup 6: Child Field Validation Error
    field_error = create_mock_field()
    # Mocking child error: validate_or_error returns (None, [ErrorMessage])
    error_msg = Message(text="Child Error", code="child_err", index=["a"])
    error_obj = MagicMock()
    error_obj.messages.return_value = [error_msg]
    field_error.validate_or_error.return_value = (None, error_obj)
    
    schema_child_err = Schema(fields={"a": field_error})
    with pytest.raises(ValidationError) as excinfo:
        schema_child_err.validate({"a": "bad_data"})
    # Check if the error propagates with the key prefix
    assert any("a: Child Error" in str(m.text) or "Child Error" in str(m.text) for m in excinfo.value.messages)

    # Setup 7: Read Only Fields (Should be skipped during validation loop)
    field_ro = create_mock_field(read_only=True)
    schema_ro = Schema(fields={"ro": field_ro})
    # Even if 'ro' is missing in input, it shouldn't be in the output 'validated' dict
    assert "ro" not in schema_ro.validate({"other": 1})
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Reference_validate():
    # Setup mocks for definitions and target field
    mock_target_field = MagicMock(spec=Field)
    definitions = Definitions({"my_ref": mock_target_field})
    
    # 1. Test successful validation
    ref_success = Reference(to="my_ref", definitions=definitions)
    mock_target_field.validate.return_value = "valid_value"
    assert ref_success.validate({"key": "value"}) == "valid_value"
    mock_target_field.validate.assert_called_with({"key": "value"})

    # 2. Test null value when allow_null is True
    ref_allow_null = Reference(to="my_ref", definitions=definitions, allow_null=True)
    assert ref_allow_null.validate(None) is None

    # 3. Test null value when allow_null is False (default)
    ref_no_null = Reference(to="my_ref", definitions=definitions, allow_null=False)
    with pytest.raises(ValidationError) as excinfo:
        ref_no_null.validate(None)
    # Check if the error message corresponds to the 'null' key in Reference.errors
    assert any("May not be null" in str(msg) for msg in excinfo.value.messages)

    # 4. Test validation error propagation from target field
    mock_error = MagicMock(spec=ValidationError)
    mock_error.messages.return_value = [MagicMock(text="Error in target", code="err")]
    mock_target_field.validate_or_error.return_value = (None, mock_error)
    
    # Note: Reference.validate calls target.validate, not validate_or_error directly,
    # but if target.validate raises ValidationError, it should propagate.
    mock_target_field.validate.side_effect = mock_error
    
    with pytest.raises(ValidationError) as excinfo:
        ref_success.validate({"bad": "data"})
    assert len(excinfo.value.messages) == 1
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Schema_validate():
    # Mocking Field behavior
    def create_mock_field(required=True, default=None, read_only=False):
        field = MagicMock(spec=Field)
        field.read_only = read_only
        field.has_default.return_value = default is not None
        field.get_default_value.return_value = default
        # Mock validate_or_error to return (value, None) or (None, error_list)
        field.validate_or_error.side_effect = lambda x: (x, None)
        return field

    # 1. Test valid input
    field_a = create_mock_field()
    field_b = create_mock_field(default="default_val")
    schema = Schema({"a": field_a, "b": field_b})
    
    result = schema.validate({"a": 1, "b": 2})
    assert result == {"a": 1, "b": 2}

    # 2. Test default value injection
    schema_with_default = Schema({"b": field_b})
    result_default = schema_with_default.validate({"a": 1}) # 'b' is missing but has default
    assert result_default["b"] == "default_val"

    # 3. Test null error (when allow_null is False)
    schema_not_null = Schema({"a": field_a}, allow_null=False)
    with pytest.raises(ValidationError) as excinfo:
        schema_not_null.validate(None)
    assert any("May not be null" in str(m) for m in excinfo.value.messages)

    # 4. Test null success (when allow_null is True)
    schema_allow_null = Schema({"a": field_a}, allow_null=True)
    assert schema_allow_null.validate(None) is None

    # 5. Test type error (input is not a dict/mapping)
    schema_type_error = Schema({"a": field_a})
    with pytest.raises(ValidationError) as excinfo:
        schema_type_error.validate([1, 2, 3])
    assert any("Must be an object" in str(m) for m in excinfo.value.messages)

    # 6. Test invalid key type (key is not a string)
    schema_invalid_key = Schema({"a": field_a})
    with pytest.raises(ValidationError) as excinfo:
        schema_invalid_key.validate({123: "value"})
    assert any("All object keys must be strings" in str(m) for m in excinfo.value.messages)

    # 7. Test required field missing
    schema_req = Schema({"a": field_a})
    with pytest.raises(ValidationError) as excinfo:
        schema_req.validate({"b": 1})
    assert any("This field is required" in str(m) for m in excinfo.value.messages)

    # 8. Test child field validation failure
    child_error_msg = Message(text="Child Error", code="child_err", index=[])
    error_field = MagicMock(spec=Field)
    error_field.read_only = False
    error_field.has_default.return_value = False
    error_field.validate_or_error.return_value = (None, [child_error_msg])
    
    schema_child_err = Schema({"a": error_field})
    with pytest.raises(ValidationError) as excinfo:
        schema_child_err.validate({"a": "bad_value"})
    # Check if the error message is prefixed with the key 'a'
    assert any("a: Child Error" in str(m) for m in excinfo.value.messages)

    # 9. Test read_only field (should not be processed/validated)
    ro_field = create_mock_field(read_only=True)
    schema_ro = Schema({"a": ro_field})
    # Even if 'a' is missing, it shouldn't trigger 'required' error because it's read_only
    # and the loop for 'required' uses self.required which excludes read_only
    assert schema_ro.validate({}) == {}
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Schema_serialize():
    # Mock fields to be used in the Schema
    field1 = MagicMock(spec=Field)
    field1.serialize.return_value = "value1"
    
    field2 = MagicMock(spec=Field)
    field2.serialize.return_value = 123
    
    field3 = MagicMock(spec=Field)
    field3.serialize.return_value = True

    fields = {
        "string_field": field1,
        "int_field": field2,
        "bool_field": field3
    }

    schema = Schema(fields=fields)

    # Test Case 1: Serialize dictionary input
    input_dict = {
        "string_field": "input_string",
        "int_field": 10,
        "bool_field": False
    }
    expected_dict = {
        "string_field": "value1",
        "int_field": 123,
        "bool_field": True
    }
    assert schema.serialize(input_dict) == expected_dict
    
    # Verify field.serialize was called with correct values from dict
    field1.serialize.assert_any_call("input_string")
    field2.serialize.assert_any_call(10)
    field3.serialize.assert_any_call(False)

    # Test Case 2: Serialize object input (using getattr)
    class MockObject:
        def __init__(self, s, i, b):
            self.string_field = s
            self.int_field = i
            self.bool_field = b

    input_obj = MockObject("obj_s", 99, True)
    expected_obj_output = {
        "string_field": "value1",
        "int_field": 123,
        "bool_field": True
    }
    assert schema.serialize(input_obj) == expected_obj_output
    
    # Verify field.serialize was called with correct values from object
    field1.serialize.assert_any_call("obj_s")
    field2.serialize.assert_any_call(99)
    field3.serialize.assert_any_call(True)

    # Test Case 3: Serialize None
    assert schema.serialize(None) is None

    # Test Case 4: Missing keys in input (should be skipped in output)
    input_incomplete = {"string_field": "only_one"}
    expected_incomplete = {"string_field": "value1"}
    assert schema.serialize(input_incomplete) == expected_incomplete

    # Test Case 5: Input is not a mapping or object with attributes (e.g., an empty list)
    # Since the code uses is_mapping = isinstance(obj, dict), 
    # if it's not a dict, it tries getattr.
    # If it's a list, getattr will fail to find the keys, returning empty dict.
    assert schema.serialize([]) == {}
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest

def test_Definitions___setitem__():
    # Test successful setting of a new item
    defs = Definitions()
    defs["key1"] = "value1"
    assert defs["key1"] == "value1"
    assert len(defs) == 1

    # Test setting an item that already exists should raise AssertionError
    with pytest.raises(AssertionError) as excinfo:
        defs["key1"] = "value2"
    assert "Definition for 'key1' has already been set." in str(excinfo.value)

    # Test updating via standard dict behavior (if allowed by implementation)
    # Note: The current implementation's __setitem__ specifically forbids overwriting.
    # We verify that the existing logic holds.
    defs["key2"] = "value2"
    assert "key2" in defs
    assert defs["key2"] == "value2"
```


# LLM-generated content at query #7
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
    field_string.serialize.return_value = lambda x: x
    field_string.get_default_value.return_value = None

    field_int = MagicMock(spec=Field)
    field_int.read_only = False
    field_int.has_default.return_value = True
    field_int.validate_or_error.side_effect = lambda x: (x, None)
    field_int.serialize.return_value = lambda x: x
    field_int.get_default_value.return_value = 0

    field_readonly = MagicMock(spec=Field)
    field_readonly.read_only = True
    field_readonly.has_default.return_value = False
    field_readonly.validate_or_error.side_effect = lambda x: (x, None)
    field_readonly.serialize.return_value = lambda x: x

    fields = {
        "name": field_string,
        "age": field_int,
        "meta": field_readonly
    }

    schema = Schema(fields=fields)

    # 1. Test Valid Input
    valid_input = {"name": "John", "age": 30, "meta": "ignored"}
    # Note: 'meta' is read_only, so it shouldn't be processed in 'validated' dict
    # 'age' has a default, so if missing, it should be populated
    result = schema.validate(valid_input)
    assert result["name"] == "John"
    assert result["age"] == 30
    assert "meta" not in result

    # 2. Test Default Value Injection
    input_missing_age = {"name": "John"}
    result_default = schema.validate(input_missing_age)
    assert result_default["age"] == 0

    # 3. Test Null Error (when allow_null is False)
    schema.allow_null = False
    with pytest.raises(ValidationError) as excinfo:
        schema.validate(None)
    assert any("May not be null" in str(m) for m in excinfo.value.messages)

    # 4. Test Type Error (not a dict)
    with pytest.raises(ValidationError) as excinfo:
        schema.validate(["not", "a", "dict"])
    assert any("Must be an object" in str(m) for m in excinfo.value.messages)

    # 5. Test Invalid Key (non-string key)
    with pytest.raises(ValidationError) as excinfo:
        schema.validate({123: "value"})
    assert any("All object keys must be strings" in str(m) for m in excinfo.value.messages)

    # 6. Test Required Field Missing
    with pytest.raises(ValidationError) as excinfo:
        schema.validate({"age": 30}) # 'name' is required
    assert any("This field is required" in str(m) for m in excinfo.value.messages)

    # 7. Test Child Field Validation Error
    field_string.validate_or_error.side_effect = lambda x: (None, [Message(text="Invalid name", code="err", index=[])])
    with pytest.raises(ValidationError) as excinfo:
        schema.validate({"name": "Invalid", "age": 30})
    # Check if the error is prefixed with the key name
    assert any("name: Invalid name" in str(m) for m in excinfo.value.messages)
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Reference_validate():
    # Mocking the Definitions class
    definitions = Definitions()
    
    # Mocking the target field
    mock_target_field = MagicMock(spec=Field)
    definitions["user"] = mock_target_field
    
    # 1. Test Case: Value is None and allow_null is True
    ref_allow_null = Reference(to="user", definitions=definitions, allow_null=True)
    assert ref_allow_null.validate(None) is None
    
    # 2. Test Case: Value is None and allow_null is False
    ref_no_null = Reference(to="user", definitions=definitions, allow_null=False)
    with pytest.raises(ValidationError) as excinfo:
        ref_no_null.validate(None)
    # Assuming validation_error("null") raises ValidationError with "May not be null."
    assert any("May not be null" in str(msg.text) for msg in excinfo.value.messages)

    # 3. Test Case: Valid value passed through to target field
    valid_value = {"id": 1, "name": "John"}
    mock_target_field.validate.return_value = valid_value
    
    result = ref_allow_null.validate(valid_value)
    assert result == valid_value
    mock_target_field.validate.assert_called_with(valid_value)

    # 4. Test Case: Target field raises ValidationError
    mock_error = MagicMock(spec=ValidationError)
    mock_error.messages.return_value = [MagicMock(text="Invalid user", code="error")]
    mock_target_field.validate.side_effect = mock_error
    
    with pytest.raises(ValidationError):
        ref_allow_null.validate({"invalid": "data"})
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Schema_validate():
    # Mocking Field for child schemas
    def create_mock_field(required=False, default=None, value=None, error=None):
        field = MagicMock(spec=Field)
        field.read_only = False
        field.has_default.return_value = default is not None
        field.get_default_value.return_value = default
        # validate_or_error returns (validated_value, error_list)
        field.validate_or_error.return_value = (value, error)
        return field

    # Setup 1: Valid case
    field_a = create_mock_field(value="val_a")
    field_b = create_mock_field(default="default_b")
    schema_valid = Schema(fields={"a": field_a, "b": field_b})
    
    input_valid = {"a": "val_a"}
    # Should return validated dict including default for b
    result = schema_valid.validate(input_valid)
    assert result == {"a": "val_a", "b": "default_b"}

    # Setup 2: Null error (not allowed)
    schema_no_null = Schema(fields={"a": field_a}, allow_null=False)
    with pytest.raises(ValidationError) as excinfo:
        schema_no_null.validate(None)
    assert any("May not be null" in str(m) for m in excinfo.value.messages)

    # Setup 3: Null allowed
    schema_allow_null = Schema(fields={"a": field_a}, allow_null=True)
    assert schema_allow_null.validate(None) is None

    # Setup 4: Type error (not a mapping)
    schema_type_err = Schema(fields={"a": field_a})
    with pytest.raises(ValidationError) as excinfo:
        schema_type_err.validate(["not", "a", "dict"])
    assert any("Must be an object" in str(m) for m in excinfo.value.messages)

    # Setup 5: Invalid key (non-string key)
    schema_key_err = Schema(fields={"a": field_a})
    with pytest.raises(ValidationError) as excinfo:
        schema_key_err.validate({123: "value"})
    assert any("All object keys must be strings" in str(m) for m in excinfo.value.messages)

    # Setup 6: Required field missing
    field_req = create_mock_field() # required by default in Schema logic if no default
    schema_req = Schema(fields={"a": field_req})
    with pytest.raises(ValidationError) as excinfo:
        schema_req.validate({"b": 1})
    assert any("This field is required" in str(m) for m in excinfo.value.messages)

    # Setup 7: Child field validation error
    child_error = ValidationError(messages=[Message(text="child_err", code="child_err", index=[])])
    field_err = create_mock_field(value=None, error=child_error)
    schema_child_err = Schema(fields={"a": field_err})
    with pytest.raises(ValidationError) as excinfo:
        schema_child_err.validate({"a": "bad_value"})
    # Check if error message is prefixed with the key name 'a'
    assert any("a: child_err" in str(m) for m in excinfo.value.messages)

    # Setup 8: Read-only field should be skipped during validation processing
    field_ro = create_mock_field(value="ro_value")
    field_ro.read_only = True
    schema_ro = Schema(fields={"a": field_ro})
    # Even if 'a' is in input, it shouldn't be processed/added to validated dict if logic skips it
    # (Note: The implementation skips 'continue' for read_only, so it won't appear in 'validated' dict)
    result_ro = schema_ro.validate({"a": "ro_value"})
    assert "a" not in result_ro
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest

def test_Definitions___setitem__():
    # Test successful setting of a new item
    defs = Definitions({"existing": 1})
    defs["new_key"] = 2
    assert defs["new_key"] == 2
    assert len(defs) == 2

    # Test assertion error when setting an existing key
    with pytest.raises(AssertionError) as excinfo:
        defs["existing"] = 99
    assert "Definition for 'existing' has already been set." in str(excinfo.value)

    # Test updating via standard dict-like behavior on a fresh instance
    defs2 = Definitions()
    defs2["a"] = "apple"
    assert defs2["a"] == "apple"
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Schema():
    # Mock Fields
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

    # Verify fields were assigned
    assert schema.fields == fields

    # Verify required list calculation:
    # 'req' is not read_only and has no default -> should be in required
    # 'opt' has default -> should NOT be in required
    # 'ro' is read_only -> should NOT be in required
    assert "req" in schema.required
    assert "opt" in schema.required is False
    assert "ro" in schema.required is False
    assert len(schema.required) == 1

    # Test with empty fields
    empty_schema = Schema(fields={})
    assert empty_schema.required == []

    # Test with all required fields
    all_req_fields = {
        "a": MagicMock(spec=Field, read_only=False, has_default=lambda: False),
        "b": MagicMock(spec=Field, read_only=False, has_default=lambda: False),
    }
    # Note: In the actual implementation, has_default is a method call
    # We adjust the mock to match the code: if not (field.read_only or field.has_default())
    for f in all_req_fields.values():
        f.read_only = False
        f.has_default.return_value = False

    schema_all_req = Schema(fields=all_req_fields)
    assert set(schema_all_req.required) == {"a", "b"}
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Schema():
    # Mocking Field class for the sub-fields
    def create_mock_field(read_only=False, default_value=None):
        field = MagicMock(spec=Field)
        field.read_only = read_only
        field.has_default.return_value = default_value is not None
        field.get_default_value.return_value = default_value
        return field

    # Case 1: All fields are required (no default, not read_only)
    fields_required = {
        "name": create_mock_field(read_only=False, default_value=None),
        "age": create_mock_field(read_only=False, default_value=None),
    }
    schema_req = Schema(fields=fields_required)
    assert schema_req.required == ["name", "age"]

    # Case 2: Some fields are read_only
    fields_mixed = {
        "name": create_mock_field(read_only=False, default_value=None),
        "id": create_mock_field(read_only=True, default_value=1),
    }
    schema_mixed = Schema(fields=fields_mixed)
    assert "name" in schema_mixed.required
    assert "id" not in schema_mixed.required

    # Case 3: Some fields have defaults
    fields_defaults = {
        "name": create_mock_field(read_only=False, default_value=None),
        "role": create_mock_field(read_only=False, default_value="user"),
    }
    schema_defaults = Schema(fields=fields_defaults)
    assert "name" in schema_defaults.required
    assert "role" not in schema_defaults.required

    # Case 4: All fields have defaults
    fields_all_defaults = {
        "name": create_mock_field(read_only=False, default_value="guest"),
        "role": create_mock_field(read_only=False, default_value="user"),
    }
    schema_all_defaults = Schema(fields=fields_all_defaults)
    assert schema_all_defaults.required == []

    # Case 5: Empty fields
    schema_empty = Schema(fields={})
    assert schema_empty.required == []
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Reference_validate():
    # Mock Definitions
    definitions = Definitions()
    
    # Mock Target Field
    mock_target_field = MagicMock(spec=Field)
    definitions["my_ref"] = mock_target_field
    
    # 1. Test valid value (passes through to target)
    ref_valid = Reference(to="my_ref", definitions=definitions)
    mock_target_field.validate.return_value = "valid_data"
    assert ref_valid.validate({"key": "value"}) == "valid_data"
    mock_target_field.validate.assert_called_with({"key": "value"})

    # 2. Test null value with allow_null=True
    ref_allow_null = Reference(to="my_ref", definitions=definitions, allow_null=True)
    assert ref_allow_null.validate(None) is None

    # 3. Test null value with allow_null=False (default)
    ref_no_null = Reference(to="my_ref", definitions=definitions, allow_null=False)
    with pytest.raises(ValidationError) as excinfo:
        ref_no_null.validate(None)
    # Check if the error message/code corresponds to 'null'
    assert any("null" in str(m.code) or "May not be null" in str(m.text) for m in excinfo.value.messages)

    # 4. Test target validation error (error propagates from target)
    mock_error = ValidationError(messages=[Message(text="Target error", code="error_code")])
    mock_target_field.validate.side_effect = mock_error
    
    with pytest.raises(ValidationError) as excinfo:
        ref_valid.validate({"bad": "data"})
    assert len(excinfo.value.messages) == 1
    assert excinfo.value.messages[0].code == "error_code"
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Schema():
    # Mocking Field objects
    # Case 1: A required field (no default, not read_only)
    field_req = MagicMock(spec=Field)
    field_req.read_only = False
    field_req.has_default.return_value = False

    # Case 2: An optional field with a default
    field_opt = MagicMock(spec=Field)
    field_opt.read_only = False
    field_opt.has_default.return_value = True

    # Case 3: A read-only field
    field_ro = MagicMock(spec=Field)
    field_ro.read_only = True
    field_ro.has_default.return_value = False

    fields = {
        "required_field": field_req,
        "optional_field": field_opt,
        "readonly_field": field_ro,
    }

    # Test initialization
    schema = Schema(fields=fields)

    # Assertions for the constructor logic
    # The 'required' list should only contain keys that are not read_only AND do not have a default
    assert "required_field" in schema.required
    assert "optional_field" not in schema.required
    assert "readonly_field" not in schema.required
    assert len(schema.required) == 1

    # Test kwargs propagation to superclass
    schema_with_kwargs = Schema(fields=fields, allow_null=True)
    assert schema_with_kwargs.allow_null is True

    # Test empty fields
    schema_empty = Schema(fields={})
    assert schema_empty.required == []
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Schema_validate():
    # Mocking Field for child schemas
    def create_mock_field(required=False, default=None, allow_null=False):
        field = MagicMock(spec=Field)
        field.read_only = False
        field.has_default.return_value = default is not None
        field.get_default_value.return_value = default
        field.allow_null = allow_null
        # Mock validate_or_error to return (value, None) for success
        field.validate_or_error.side_effect = lambda x: (x, None)
        return field

    # 1. Test valid input
    fields = {
        "name": create_mock_field(),
        "age": create_mock_field(default=18)
    }
    schema = Schema(fields=fields)
    input_data = {"name": "Alice", "age": 25}
    assert schema.validate(input_data) == {"name": "Alice", "age": 25}

    # 2. Test default value injection
    input_with_missing_optional = {"name": "Bob"}
    # 'age' has default 18, so it should be added to validated dict
    result = schema.validate(input_with_missing_optional)
    assert result["age"] == 18
    assert "name" in result

    # 3. Test Null error (when not allowed)
    schema_no_null = Schema(fields={"name": create_mock_field()})
    with pytest.raises(ValidationError) as excinfo:
        schema_no_null.validate(None)
    assert any("May not be null" in str(m) for m_msg in excinfo.value.messages for m in [m_msg])

    # 4. Test Null allowed
    schema_allow_null = Schema(fields={"name": create_mock_field()}, allow_null=True)
    assert schema_allow_null.validate(None) is None

    # 5. Test Type error (not a dict/mapping)
    with pytest.raises(ValidationError) as excinfo:
        schema.validate(["not", "a", "dict"])
    assert any("Must be an object" in str(m) for m_msg in excinfo.value.messages for m in [m_msg])

    # 6. Test Invalid Key error (keys must be strings)
    with pytest.raises(ValidationError) as excinfo:
        schema.validate({123: "value"})
    assert any("All object keys must be strings" in str(m) for m_msg in excinfo.value.messages for m in [m_msg])

    # 7. Test Required field missing
    required_field = create_mock_field() # No default, so it's in self.required
    schema_required = Schema(fields={"username": required_field})
    with pytest.raises(ValidationError) as excinfo:
        schema_required.validate({"email": "test@test.com"})
    assert any("This field is required" in str(m) for m_msg in excinfo.value.messages for m in [m_msg])

    # 8. Test Child Validation error propagation
    error_field = create_mock_field()
    # Mocking a validation error in a child field
    error_field.validate_or_error.side_effect = lambda x: (None, [Message(text="Invalid age", code="error", index=[])])
    
    schema_nested = Schema(fields={"age": error_field})
    with pytest.raises(ValidationError) as excinfo:
        schema_nested.validate({"age": "not_a_number"})
    # Check if the error message contains the prefix (the key 'age')
    assert any("age: Invalid age" in str(m) for m_msg in excinfo.value.messages for m in [m_msg])

    # 9. Test Read-only field (should be skipped during validation loop)
    ro_field = create_mock_field()
    ro_field.read_only = True
    schema_ro = Schema(fields={"id": ro_field})
    # Even if 'id' is missing in input, it shouldn't trigger 'required' because it's read_only
    # and shouldn't be processed in the property loop.
    assert schema_ro.validate({}) == {}
```


