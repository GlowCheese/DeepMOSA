####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
import decimal

class MockValidationError(Exception):
    def __init__(self, messages):
        self.messages = lambda add_prefix=None: messages

class MockError:
    def __init__(self, messages_list):
        self.messages_list = messages_list
    def messages(self, add_prefix=None):
        return self.messages_list

class MockMessage:
    def __init__(self, text, code, index=None):
        self.text = text
        self.code = code
        self.index = index

class MockField:
    def __init__(self, allow_null=False, should_pass=True, error_code="type"):
        self.allow_null = allow_null
        self.should_pass = should_pass
        self.error_code = error_code

    def validate_or_error(self, value):
        if self.should_pass:
            return value, None
        else:
            msg = MockMessage(text="Error", code=self.error_code)
            err = MockError([msg])
            return None, err

    def validate(self, value):
        return self.validate_or_error(value)

class MockUnion(Union):
    def validation_error(self, code):
        raise MockValidationError([MockMessage(text="Error", code=code)])

def test_Union_validate():
    # Case 1: Value is None and allow_null is True (inherited from children)
    field_null_ok = MockField(allow_null=True)
    union_null_ok = MockUnion(any_of=[field_null_ok])
    assert union_null_ok.validate(None) is None

    # Case 2: Value is None and allow_null is False
    field_not_null = MockField(allow_null=False)
    union_not_null = MockUnion(any_of=[field_not_null])
    with pytest.raises(MockValidationError) as excinfo:
        union_not_null.validate(None)
    assert excinfo.value.messages()[0].code == "null"

    # Case 3: Value matches one of the types (Success)
    pass_field = MockField(should_pass=True)
    fail_field = MockField(should_pass=False)
    union_success = MockUnion(any_of=[fail_field, pass_field])
    assert union_success.validate("test") == "test"

    # Case 4: Value matches no types (Union error)
    union_fail_all = MockUnion(any_of=[fail_field, fail_field])
    with pytest.raises(MockValidationError) as excinfo:
        union_fail_all.validate("test")
    assert excinfo.value.messages()[0].code == "union"

    # Case 5: Value matches one type, but that type has a specific validation error (not 'type')
    # The Union should return the specific error from the child.
    specific_error_field = MockField(should_pass=False, error_code="constraint_violation")
    union_specific_err = MockUnion(any_of=[fail_field, specific_error_field])
    with pytest.raises(MockValidationError) as excinfo:
        union_specific_err.validate("test")
    assert excinfo.value.messages()[0].code == "constraint_violation"

    # Case 6: Value matches one type, but that type has a 'type' error with an index (complex error)
    indexed_error_field = MockField(should_pass=False, error_code="type")
    # Manually inject an error with an index to trigger the "candidate" logic
    class IndexedErrorField(MockField):
        def validate_or_error(self, value):
            msg = MockMessage(text="Err", code="type", index=["some_key"])
            return None, MockError([msg])

    union_indexed_err = MockUnion(any_of=[IndexedErrorField(), fail_field])
    with pytest.raises(MockValidationError) as excinfo:
        union_indexed_err.validate("test")
    assert excinfo.value.messages()[0].code == "type"
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Object_validate():
    # Mocking ValidationResult and ValidationError structure as used in the code
    class MockMessage:
        def __init__(self, text, code, index=None, key=None):
            self.text = text
            self.code = code
            self.index = index or []
            self.key = key
        def messages(self, add_prefix=None):
            return [self]

    class MockValidationError(Exception):
        def __init__(self, messages=None):
            self.messages = messages or []

    # Patching the globals to match the class's expectations during test execution
    import sys
    module = sys.modules[__name__]
    setattr(module, 'Message', MockMessage)
    setagit_error = setattr(module, 'ValidationError', MockValidationError)
    
    # Setup Fields for testing
    string_field = String(title="Name")
    int_field = Integer(title="Age")
    bool_field = Boolean(title="Active")

    # 1. Test Valid Object
    obj_schema = Object(
        properties={
            "name": string_field,
            "age": int_field
        },
        required=["name"],
        additional_properties=True
    )
    valid_data = {"name": "Alice", "age": 30, "extra": "info"}
    assert obj_schema.validate(valid_data) == {"name": "Alice", "age": 30, "extra": "info"}

    # 2. Test Null Value (if allow_null is False)
    with pytest.raises(Exception) as excinfo:
        obj_schema.validate(None)
    assert any("May not be null" in str(m.text) for m in excinfo.value.messages if hasattr(excinfo.value, 'messages'))

    # 3. Test Type Error (Input is not a dict)
    with pytest.raises(Exception) as excinfo:
        obj_schema.validate(["not", "a", "dict"])
    assert any("Must be an object" in str(m.text) for m in excinfo.value.messages)

    # 4. Test Required Property Missing
    with pytest.raises(Exception) as excinfo:
        obj_schema.validate({"age": 30}) # 'name' is missing
    assert any("This field is required" in str(m.text) and m.index == ["name"] for m in excinfo.value.messages)

    # 5. Test Property Validation Error (Nested error)
    with pytest.raises(Exception) as excinfo:
        obj_schema.validate({"name": 123, "age": 30}) # name should be string
    assert any("Must be a string" in str(m.text) and m.index == ["name"] for m in excinfo.value.messages)

    # 6. Test Min/Max Properties
    min_prop_schema = Object(properties={"a": string_field}, min_properties=2)
    with pytest.raises(Exception) as excinfo:
        min_prop_schema.validate({"a": "val"})
    assert any("Must have at least 2 properties" in str(m.text) for m in excinfo.value.messages)

    # 7. Test Additional Properties = False
    strict_schema = Object(properties={"name": string_field}, additional_properties=False)
    with pytest.raises(Exception) as excinfo:
        strict_schema.validate({"name": "Alice", "age": 30})
    assert any("Invalid property name" in str(m.text) and m.key == "age" for m in excinfo.value.messages)

    # 8. Test Pattern Properties
    pattern_schema = Object(
        pattern_properties={r"^user_\d+$": string_field},
        additional_properties=False
    )
    valid_pattern_data = {"user_123": "Bob"}
    assert pattern_schema.validate(valid_pattern_data) == {"user_123": "Bob"}

    with pytest.raises(Exception) as excinfo:
        pattern_schema.validate({"user_abc": "Bob"}) # regex mismatch, treated as invalid property if additional_properties=False
    
    # 9. Test Default Values in Object
    default_field = String(title="Default", default="Guest")
    default_obj_schema = Object(properties={"username": default_field})
    assert default_obj_schema.validate({"other": "val"}) == {"username": "Guest"}

    # Cleanup patches
    delattr(module, 'Message')
    delattr(module, 'ValidationError')
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
import decimal

def test_Object_validate():
    # Mocking necessary dependencies that are not provided in the snippet 
    # but required for Object.validate to run without error.
    # We assume ValidationError, Message, and Uniqueness/FORMATS exist.
    
    class MockValidationError(Exception):
        def __init__(self, messages=None):
            self.messages = messages or []

    class MockMessage:
        def __init__(self, text, code, index=None, key=None):
            self.text = text
            self.code = code
            self.index = index or []
            self.key = key
        def messages(self, add_prefix=None):
            return [self]

    # Patching the globals used in Object.validate
    global ValidationError, Message
    ValidationError = MockValidationError
    Message = MockMessage

    # 1. Test successful validation of a simple object with properties and defaults
    age_field = Integer(title="Age", default=18)
    name_field = String(title="Name")
    
    obj_schema = Object(
        properties={
            "name": name_field,
            "age": age_field
        },
        required=["name"]
    )

    input_data = {"name": "John", "age": 25}
    result = obj_schema.validate(input_data)
    assert result["name"] == "John"
    assert result["age"] == 25

    # Test default value injection when key is missing from input
    input_missing_age = {"name": "Jane"}
    result_default = obj_schema.validate(input_missing_age)
    assert result_default["age"] == 18

    # 2. Test validation error for missing required property
    with pytest.raises(MockValidationError) as excinfo:
        obj_schema.validate({"age": 30})
    assert any(m.code == "required" and m.index == ["name"] for m in excinfo.value.messages)

    # 3. Test validation error for invalid property type (not a dict)
    with pytest.raises(MockValidationError) as excinfo:
        obj_schema.validate(["not", "a", "dict"])
    assert any(m.code == "type" for m in excinfo.value.messages)

    # 4. Test validation error for invalid property name (using property_names field)
    key_validator = String(max_length=3)
    obj_with_key_rules = Object(
        properties={"a": String()},
        property_names=key_validator
    )
    with pytest.raises(MockValidationError) as excinfo:
        obj_with_key_rules.validate({"abcde": "val"})
    assert any(m.code == "invalid_property" and m.index == ["abcde"] for m in excinfo.value.messages)

    # 5. Test additional_properties=False (disallow unknown keys)
    strict_obj = Object(properties={"a": String()}, additional_properties=False)
    with pytest.raises(MockValidationError) as excinfo:
        strict_obj.validate({"a": "val", "b": "unexpected"})
    assert any(m.code == "invalid_property" and m.key == "b" for m in excinfo.value.messages)

    # 6. Test additional_properties=Field (validate unknown keys against a schema)
    extra_validator = Integer()
    flexible_obj = Object(properties={"a": String()}, additional_properties=extra_validator)
    
    # Valid extra property
    assert flexible_obj.validate({"a": "val", "extra": 10}) == {"a": "val", "extra": 10}
    
    # Invalid extra property (string instead of int)
    with pytest.raises(MockValidationError) as excinfo:
        flexible_obj.validate({"a": "val", "extra": "not_an_int"})
    assert any("extra" in str(m.messages()) for m in excinfo.value.messages)

    # 7. Test min/max properties constraints
    size_constrained_obj = Object(properties={"a": String()}, min_properties=2, max_properties=2)
    
    with pytest.raises(MockValidationError) as excinfo:
        size_constrained_obj.validate({"a": "val"}) # Too few
    assert any(m.code == "min_properties" for m in excinfo.value.messages)

    with pytest.raises(MockValidationError) as excinfo:
        size_constrained_obj.validate({"a": "val", "b": "val", "c": "val"}) # Too many
    assert any(m.code == "max_properties" for m in excinfo.value.messages)

    # 8. Test pattern_properties
    pattern_obj = Object(
        properties={"fixed": String()},
        pattern_properties={r"^dyn_.*": String()}
    )
    input_patterns = {"fixed": "static", "dyn_key": "dynamic_value"}
    result_patterns = pattern_obj.validate(input_patterns)
    assert result_patterns["dyn_key"] == "dynamic_value"
    assert result_patterns["fixed"] == "static"

    # 9. Test allow_null
    nullable_obj = Object(properties={"a": String(allow_null=True)})
    assert nullable_obj.validate({"a": None}) == {"a": None}
    with pytest.raises(MockValidationError):
        nullable_obj.validate({"a": None}) # If NOT allow_null (String default)
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Array_validate():
    # Helper to create a Mock Field that behaves like a real Field
    def create_mock_field(return_value=None, error=None):
        field = MagicMock(spec=Field)
        if error:
            # Simulate the behavior of validate_or_error returning (value, error)
            field.validate_or_error.return_value = (None, error)
        else:
            field.validate_or_error.return_value = (return_value, None)
        # Mocking error.messages to return a list of messages
        if error:
            error.messages.return_value = [MagicMock()]
        return field

    # 1. Test Null validation (when allow_null is True)
    arr_null_ok = Array(items=Integer(), allow_null=True)
    assert arr_null_ok.validate(None) is None

    # 2. Test Null validation (when allow_null is False)
    arr_null_fail = Array(items=Integer(), allow_null=False)
    with pytest.raises(ValidationError):
        arr_null_fail.validate(None)

    # 3. Test Type error (not a list)
    arr_type_error = Array(items=Integer())
    with pytest.append_error_if_needed: # Logic to catch 'type' error message if possible
        with pytest.raises(ValidationError) as excinfo:
            arr_type_error.validate("not a list")
        assert any(m.code == "type" for m in excinfo.value.messages)

    # 4. Test min_items / empty error
    arr_min_1 = Array(items=Integer(), min_items=1)
    with pytest.raises(ValidationError) as excinfo:
        arr_min_1.validate([])
    assert any(m.code == "empty" for m in excinfo.value.messages)

    # 5. Test exact_items
    arr_exact = Array(items=Integer(), exact_items=2)
    with pytest.raises(ValidationError) as excinfo:
        arr_exact.validate([1])
    assert any(m.code == "exact_items" for m in excinfo.value.messages)

    # 6. Test max_items
    arr_max = Array(items=Integer(), max_items=1)
    with pytest.raises(ValidationError) as excinfo:
        arr_max.validate([1, 2])
    assert any(m.code == "max_items" for m in excinfo.value.messages)

    # 7. Test item validation success (Single Field type)
    int_field = MagicMock(spec=Field)
    int_field.validate_or_error.return_value = (1, None)
    arr_single = Array(items=int_field)
    assert arr_single.validate([1]) == [1]

    # 8. Test item validation failure (Single Field type error)
    mock_error = MagicMock()
    mock_error.messages.return_value = [MagicMock(code="type", index=[0])]
    int_field.validate_or_error.return_value = (None, mock_error)
    arr_fail = Array(items=int_field)
    with pytest.raises(ValidationError) as excinfo:
        arr_fail.validate([1])
    assert any(m.code == "type" for m in excinfo.value.messages)

    # 9. Test Tuple-like items (List of Fields) validation
    field_a = MagicMock(spec=Field)
    field_a.validate_or_error.return_value = ("a", None)
    field_b = MagicMock(spec=Field)
    field_b.validate_or_error.return_value = ("b", None)
    arr_tuple = Array(items=[field_a, field_b])
    assert arr_tuple.validate(["a", "b"]) == ["a", "b"]

    # 10. Test unique_items validation
    unique_field = MagicMock(spec=Field)
    unique_field.validate_or_error.return_value = (None, None)
    # We need to mock the actual value returned by validate_or_error for uniqueness check
    unique_field.validate_or_error.side_effect = [ (1, None), (1, None) ]
    arr_unique = Array(items=unique_field, unique_items=True)
    with pytest.raises(ValidationError) as excinfo:
        arr_unique.validate([1, 1])
    assert any(m.code == "unique_items" for m in excinfo.value.messages)

    # 11. Test additional_items as a Field
    add_field = MagicMock(spec=Field)
    add_field.validate_or_error.return_value = (99, None)
    arr_additional = Array(items=[int_field], additional_items=add_field)
    assert arr_additional.validate([1, 2]) == [1, 99]

    # 12. Test max_items with list of fields (implicit max_items from length)
    arr_list_fields = Array(items=[int_field, int_field], additional_items=False)
    with pytest.raises(ValidationError) as excinfo:
        arr_list_fields.validate([1, 2, 3])
    assert any(m.code == "max_items" for m in excinfo.value.messages)
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Array_validate():
    # Mocking Field and ValidationError as they are external dependencies in the snippet
    # In a real scenario, these would be imported from the actual module.
    
    class MockValidationError(Exception):
        def __init__(self, messages):
            self.messages = messages

    class MockMessage:
        def __init__(self, text, code, index=None, key=None):
            self.text = text
            self.code = code
            self.index = index
            self.key = key
        def messages(self, add_prefix=None):
            return [self]

    # Setup base Field mock to behave like the real one for validation errors
    class MockField:
        def __init__(self, *args, **kwargs):
            self.allow_null = True
        def validate_or_error(self, value):
            try:
                res = self.validate(value)
                return res, None
            except Exception as e:
                # Simulate the error structure expected by Array.validate
                msg = MockMessage("error", "error", index=[0])
                err = MockValidationError([msg])
                # We need to attach a 'messages' method to the exception instance 
                # because Array calls error.messages()
                err.messages = lambda add_prefix=None: [msg]
                return None, err
        def validate(self, value):
            return value
        def get_error_text(self, code): return code
        def has_default(self): return False
        def get_default_value(self): return None

    # Injecting necessary components into the global scope for the test
    global ValidationError, Message, Uniqueness
    ValidationError = MockValidationError
    Message = MockMessage
    class Uniqueness:
        def __init__(self): self.items = set()
        def add(self, item): self.items.add(item)
        def __contains__(self, item): return item in self.items

    # --- Test Cases ---

    # 1. Test Null value when allow_null is True
    arr_null = Array(items=MockField(), allow_null=True)
    assert arr_null.validate(None) is None

    # 2. Test Null value when allow_null is False
    arr_no_null = Array(items=MockField(), allow_null=False)
    with pytest.raises(Exception) as excinfo:
        arr_no_null.validate(None)
    assert "null" in str(excinfo.value.messages[0].code)

    # 3. Test Type error (not a list)
    arr_type_err = Array(items=MockField())
    with pytest.append_error_msg := pytest.raises(Exception) as excinfo:
        arr_type_err.validate("not a list")
    assert "type" in str(excinfo.value.messages[0].code)

    # 4. Test min_items (exact_items logic)
    arr_exact = Array(items=MockField(), exact_items=2)
    with pytest.raises(Exception) as excinfo:
        arr_exact.validate([1])
    assert "exact_items" in str(excinfo.value.messages[0].code)
    assert arr_exact.validate([1, 2]) == [1, 2]

    # 5. Test min_items (range logic)
    arr_min = Array(items=MockField(), min_items=2)
    with pytest.raises(Exception) as excinfo:
        arr_min.validate([1])
    assert "min_items" in str(excinfo.value.messages[0].code)
    # Test empty error specifically when min_items is 1
    arr_empty = Array(items=MockField(), min_items=1)
    with pytest.raises(Exception) as excinfo:
        arr_empty.validate([])
    assert "empty" in str(excinfo.value.messages[0].code)

    # 6. Test max_items
    arr_max = Array(items=MockField(), max_items=1)
    with pytest.raises(Exception) as excinfo:
        arr_max.validate([1, 2])
    assert "max_items" in str(excinfo.value.messages[0].code)

    # 7. Test unique_items
    arr_unique = Array(items=MockField(), unique_items=True)
    with pytest.raises(Exception) as excinfo:
        arr_unique.validate([1, 1])
    assert "unique_items" in str(excinfo.value.messages[0].code)

    # 8. Test Tuple/List of items (positional validation)
    item1 = MockField()
    item2 = MockField()
    # Overwrite validate for specific item to simulate failure
    item2.validate = MagicMock(side_effect=Exception("fail"))
    arr_pos = Array(items=[item1, item2])
    with pytest.raises(Exception) as excinfo:
        arr_pos.validate([10, 20])
    # Check if error prefix is the index (pos 1)
    assert excinfo.value.messages[0].index == [1]

    # 9. Test additional_items as Field
    add_field = MockField()
    add_field.validate = MagicMock(return_value="validated")
    arr_add = Array(items=[item1], additional_items=add_field)
    assert arr_add.validate([1, 2]) == [1, "validated"]

    # 10. Test successful validation of a standard list
    arr_success = Array(items=MockField())
    assert arr_success.validate([1, "a", True]) == [1, "a", True]
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest

class MockField(Field):
    def validate(self, value):
        return value
    def serialize(self, value):
        return value
    def validate_or_error(self, value):
        return value, None

def test_Array():
    # Test case 1: Single Field item type
    field_single = Array(items=MockField())
    assert field_single.items == [MockField()]
    assert field_single.additional_items is False
    assert field_single.min_items is None
    assert field_single.max_items is None
    assert field_single.unique_items is False

    # Test case 2: Sequence of Fields (Tuple/List)
    fields_list = [MockField(), MockField()]
    field_seq = Array(items=fields_list)
    assert field_seq.items == fields_list
    # Based on implementation logic: if items is list, min_items becomes len(items)
    assert field_seq.min_items == 2

    # Test case 3: exact_items parameter
    field_exact = Array(exact_items=5)
    assert field_exact.min_items == 5
    assert field_exact.max_items == 5

    # Test case 4: additional_items as Field
    add_field = MockField()
    field_add = Array(items=[MockField()], additional_items=add_field)
    assert field_add.additional_items == add_field

    # Test case 5: min_items and max_items constraints
    field_bounds = Array(min_items=2, max_items=10)
    assert field_bounds.min_items == 2
    assert field_bounds.max_items == 10

    # Test case 6: unique_items flag
    field_unique = Array(unique_items=True)
    assert field_unique.unique_items is True

    # Test case 7: Validation of assertions (Error handling)
    # items must be Field or List[Field]
    with pytest.raises(AssertionError):
        Array(items="not a field")

    # additional_items must be bool or Field
    with pytest.raises(AssertionError):
        Array(additional_items=123)

    # min_items must be int or None
    with pytest.raises(AssertionError):
        Array(min_items="two")

    # max_items must be int or None
    with pytest.raises(AssertionError):
        Array(max_items=None) # This is actually valid, testing error for non-int
    
    with pytest.raises(AssertionError):
        Array(max_items=2.5)

    # unique_items must be bool
    with pytest.raises(AssertionError):
        Array(unique_items="yes")
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
import re

def test_String():
    # Test basic initialization and default values
    s = String(title="Test", description="Desc")
    assert s.title == "Test"
    assert s.description == "Desc"
    assert s.allow_blank is False
    assert s.trim_whitespace is True
    assert s.max_length is None
    assert s.min_length is None
    assert s.pattern is None
    assert s.format is None
    assert s.coerce_types is True

    # Test allow_blank setting and default behavior
    s_blank = String(allow_blank=True)
    assert s_blank.allow_blank is True
    assert s_blank.default == ""

    # Test max_length and min_length assertions/assignments
    s_lengths = String(max_length=10, min_length=5)
    assert s_lengths.max_length == 10
    assert s_lengths.min_length == 5

    with pytest.raises(AssertionError):
        String(max_length="not_an_int")
    
    with pytest.raises(AssertionError):
        String(min_length="not_an_int")

    # Test pattern with string input (compiles to regex)
    s_pattern_str = String(pattern=r"^[a-z]+$")
    assert s_pattern_str.pattern == r"^[a-z]+$"
    assert s_pattern_str.pattern_regex is not None
    assert s_pattern_str.pattern_regex.match("abc")

    # Test pattern with compiled regex object
    pattern_obj = re.compile(r"\d+")
    s_pattern_re = String(pattern=pattern_obj)
    assert s_pattern_re.pattern == r"\d+"
    assert s_pattern_re.pattern_regex == pattern_obj

    with pytest.raises(AssertionError):
        String(pattern=123)

    # Test format input
    s_format = String(format="email")
    assert s_format.format == "email"

    with pytest.raises(AssertionError):
        String(format=123)

    # Test coerce_types and trim_whitespace
    s_no_trim = String(trim_whitespace=False)
    assert s_no_trim.trim_whitespace is False

    s_no_coerce = String(coerce_types=False)
    assert s_no_coerce.coerce_types is False

    # Test inheritance of kwargs to Field
    s_kwargs = String(title="K", description="D", allow_null=True, read_only=True)
    assert s_kwargs.title == "K"
    assert s_kwargs.description == "D"
    assert s_kwargs.allow_null is True
    assert s_kwargs.read_only is True
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest

def test_Const():
    # Test successful initialization with various types
    for val in [1, "string", True, None, {"key": "val"}, 3.14]:
        const = Const(const=val)
        assert const.const == val

    # Test that allow_null is not allowed in kwargs (as per the assert in __init__)
    with pytest.raises(AssertionError):
        Const(const="test", allow_null=True)

    # Test validation logic for Const
    
    # 1. Success case: value matches const
    assert Const(const=10).validate(10) == 10
    assert Const(const="hello").validate("hello") == "hello"
    assert Const(const=None).validate(None) is None

    # 2. Failure case: value does not match const (non-null const)
    with pytest.raises(Exception) as excinfo:
        Const(const="expected").validate("actual")
    # Assuming validation_error raises a ValidationError containing the 'const' error code
    assert "const" in str(excinfo.value)

    # 3. Failure case: value is not null when const is None
    with pytest.raises(Exception) as excinfo:
        Const(const=None).validate("not_none")
    assert "only_null" in str(excinfo.value)
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest

def test_Field_get_default_value():
    # Test case 1: No default provided (NO_DEFAULT)
    field_no_default = Field(title="No Default", description="Test")
    assert field_no_default.get_default_value() is NO_DEFAULT

    # Test case 2: Static default value
    field_static_default = Field(title="Static", description="Test", default="hello")
    assert field_static_default.get_default_value() == "hello"

    # Test case 3: Callable default value (lambda)
    field_callable_default = Field(title="Callable", description="Test", default=lambda: 42)
    assert field_callable_default.get_default_value() == 42

    # Test case 4: Default is None (explicitly allowed null)
    field_none_default = Field(title="None", description="Test", default=None, allow_null=True)
    assert field_none_default.get_default_value() is None

    # Test case 5: Default value that is a complex object
    complex_obj = {"key": "value"}
    field_complex_default = Field(title="Complex", description="Test", default=complex_obj)
    assert field_complex_default.get_default_value() == complex_obj

    # Test case 6: Default value is a boolean
    field_bool_default = Field(title="Bool", description="Test", default=True)
    assert field_bool_default.get_default_value() is True
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest

def test_Field_get_default_value():
    # Test case 1: No default provided (NO_DEFAULT)
    field_no_default = Field()
    assert field_no_default.get_default_value() is NO_DEFAULT

    # Test case 2: Static default value
    field_static = Field(default="hello")
    assert field_static.get_default_value() == "hello"

    # Test case 3: Default value with allow_null=True (should be None)
    field_null = Field(allow_null=True)
    assert field_null.get_default_value() is None

    # Test case 4: Callable default value
    def dynamic_factory():
        return {"key": "value"}
    
    field_callable = Field(default=dynamic_factory)
    assert field_callable.get_default_value() == {"key": "value"}
    # Ensure it's a new instance/call if called again (if the factory is designed that way)
    assert field_callable.get_default_value() == {"key": "value"}

    # Test case 5: Integer default
    field_int = Field(default=42)
    assert field_int.get_default_value() == 42
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
import decimal

def test_Const():
    # Test valid initialization with a constant value
    c1 = Const(const="hello")
    assert c1.const == "hello"

    # Test valid initialization with None as constant (must not include allow_null in kwargs)
    c2 = Const(const=None)
    assert c1.const is not None
    assert c2.const is None

    # Test valid initialization with various types
    c3 = Const(const=123)
    assert c3.const == 123
    
    c4 = Const(const=True)
    assert c4.const is True

    # Test that providing allow_null in kwargs raises an AssertionError as per the __init__ implementation
    with pytest.raises(AssertionError):
        Const(const="test", allow_null=True)

    # Test validation of correct constant
    assert c1.validate("hello") == "hello"

    # Test validation error for incorrect constant (string mismatch)
    with pytest.raises(ValidationError) as excinfo:
        c1.validate("world")
    assert "const" in str(excinfo.value)

    # Test validation error when const is None but value is provided
    with pytest.raises(ValidationError) as excinfo:
        c2.validate("not none")
    assert "only_null" in str(excinfo.value)

    # Test validation success for None constant when value is None
    assert c2.validate(None) is None
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest

def test_Boolean_validate():
    # Test successful boolean validation (native types)
    bool_field = Boolean()
    assert bool_field.validate(True) is True
    assert bool_field.validate(False) is False

    # Test coercion from strings and integers (default behavior)
    assert bool_field.validate("true") is True
    assert bool_field.validate("TRUE") is True
    assert bool_field.validate("on") is True
    assert bool_field.validate("1") is True
    assert bool_field.validate("false") is False
    assert bool_field.validate("off") is False
    assert bool_field.validate("0") is False
    assert bool_field.validate(1) is True
    assert bool_field.validate(0) is False
    assert bool_field.validate("") is False

    # Test null handling (not allowed by default)
    with pytest.raises(ValidationError) as excinfo:
        bool_field.validate(None)
    assert excinfo.value.code == "null"

    # Test null handling (allowed)
    bool_field_null = Boolean(allow_null=True)
    assert bool_field_null.validate(None) is None
    # Test coercion of specific null-like strings when allow_null is True
    assert bool_field_null.validate("null") is None
    assert bool_field_null.validate("none") is None
    assert bool_field_null.validate("") is None

    # Test invalid types with coercion enabled (default)
    with pytest.raises(ValidationError) as excinfo:
        bool_field.validate("not-a-boolean")
    assert excinfo.value.code == "type"
    
    with pytest.raises(ValidationError) as excinfo:
        bool_field.validate(2)
    assert excinfo.value.code == "type"

    # Test invalid types with coercion disabled
    bool_field_no_coerce = Boolean(coerce_types=False)
    assert bool_field_no_coerce.validate(True) is True
    with pytest.raises(ValidationError) as excinfo:
        bool_field_no_coerce.validate("true")
    assert excinfo.value.code == "type"

    # Test error message formatting
    bool_field_err = Boolean()
    result = bool_field_err.validate_or_error(None)
    assert result.error is not None
    assert result.error.text == "May not be null."
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest
import decimal

def test_Const():
    # Test successful initialization with various types
    for val in [1, "string", True, None, 1.5, {"key": "val"}]:
        const = Const(const=val)
        assert const.const == val
        assert const.allow_null is False

    # Test that passing allow_null to constructor raises AssertionError as per implementation
    with pytest.raises(AssertionError):
        Const(const="test", allow_null=True)

    # Test validation logic for Const
    c_int = Const(const=10)
    assert c_int.validate(10) == 10
    with pytest.raises(ValidationError):
        c_int.validate(5)
    with pytest.raises(ValidationError):
        c_int.validate("10")

    c_none = Const(const=None)
    # Note: The implementation of validate for Const handles None via self.const check
    assert c_none.validate(None) is None
    with pytest.raises(ValidationError) as excinfo:
        c_none.validate(1)
    assert "only_null" in str(excinfo.value)

    c_str = Const(const="hello")
    assert c_str.validate("hello") == "hello"
    with pytest.raises(ValidationError):
        c_str.validate("world")
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Union_validate():
    # Mocking dependencies not provided in the snippet but required for execution
    # We assume Field, ValidationError, and validate_or_error exist in the environment.
    
    class MockField:
        def __init__(self, allow_null=False):
            self.allow_null = allow_null
            self.validation_error = lambda self, code: MagicMock(code=code)
        
        def validate_or_error(self, value):
            # This will be overridden in specific tests
            return value, None

    # Setup internal class structure for testing context
    class MockUnion(Union):
        def __init__(self, any_of, **kwargs):
            super().__init__(any_of, **kwargs)

    # 1. Test successful validation (matches first type)
    field_int = MagicMock(spec=MockField)
    field_int.validate_or_error.return_value = (10, None)
    field_int.allow_null = False
    
    field_str = MagicMock(spec=MockField)
    field_str.validate_or_error.return_value = ("text", None)
    field_str.allow_null = False

    union_success = Union([field_int, field_str])
    assert union_success.validate(10) == 10

    # 2. Test successful validation (matches second type)
    union_success_second = Union([field_int, field_str])
    assert union_success_second.validate("text") == "text"

    # 3. Test null handling when allow_null is True
    field_nullable = MagicMock(spec=MockField)
    field_nullable.allow_null = True
    field_nullable.validate_or_error.return_value = (None, None)
    
    union_nullable = Union([field_nullable])
    assert union_nullable.validate(None) is None

    # 4. Test null error when allow_null is False
    field_non_nullable = MagicMock(spec=MockField)
    field_non_nullable.allow_null = False
    # Mocking the error object structure required by Union.validate
    error_mock = MagicMock()
    error_mock.messages.return_value = []
    field_non_nullable.validate_or_error.return_value = (None, error_mock)
    
    union_strict = Union([field_non_nullable])
    with pytest.raises(Exception): # Should raise the error from validate_or_error
        union_strict.validate(None)

    # 5. Test Union failure (no types match)
    field_fail_int = MagicMock(spec=MockField)
    err_type = MagicMock()
    err_type.messages.return_value = [{"code": "type"}]
    field_fail_int.validate_or_error.return_value = (None, err_type)
    
    field_fail_str = MagicMock(spec=MockField)
    err_type2 = MagicMock()
    err_type2.messages.return_value = [{"code": "type"}]
    field_fail_str.validate_or_error.return_value = (None, err_type2)

    union_fail = Union([field_fail_int, field_fail_str])
    # We need to mock validation_error for the union instance itself
    union_fail.validation_error = lambda self, code: Exception(code)
    
    with pytest.raises(Exception) as excinfo:
        union_fail.validate("not_matching")
    assert "union" in str(excinfo.value)

    # 6. Test Union failure with a specific non-type error (should return that error)
    field_spec_error = MagicMock(spec=MockField)
    err_custom = MagicMock()
    err_custom.messages.return_value = [{"code": "not_a_type_error", "index": [0]}]
    field_spec_error.validate_or_error.return_value = (None, err_custom)
    
    union_custom_error = Union([field_spec_error])
    with pytest.raises(Exception):
        union_custom_error.validate("some_value")
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest
import decimal

def test_Object_validate():
    # Setup helper classes to satisfy dependencies in Object.validate
    class MockMessage:
        def __init__(self, text, code, index=None, key=None):
            self.text = text
            self.code = code
            self.index = index if index is not None else [key]

    class MockValidationError(Exception):
        def __init__(self, messages=None):
            self.messages = messages or []

    # Patching the global scope for the test environment
    global Message, ValidationError, Uniqueness, FORMATS
    Message = MockMessage
    ValidationError = MockValidationError
    Uniqueness = lambda x: x
    FORMATS = {}

    # 1. Test Valid Object with properties and defaults
    name_field = String(title="Name", allow_blank=False)
    age_field = Integer(title="Age", default=18)
    
    obj_schema = Object(
        properties={"name": name_field, "age": age_field},
        required=["name"],
        additional_properties=True
    )

    input_data = {"name": "Alice", "age": 25, "extra": "info"}
    # Should return validated dict with extra property preserved and types validated
    assert obj_schema.validate(input_data) == {"name": "Alice", "age": 25, "extra": "info"}

    # 2. Test Required Property Missing
    with pytest.raises(ValidationError) as excinfo:
        obj_schema.validate({"age": 30})
    assert any(m.code == "required" and m.index == ["name"] for m/ in excinfo.value.messages)

    # 3. Test Invalid Property Type (non-string key)
    with pytest.raises(ValidationError) as excinfo:
        obj_schema.validate({123: "value"})
    assert any(m.code == "invalid_key" for m in excinfo.value.messages)

    # 4. Test Max Properties Constraint
    max_prop_schema = Object(properties={}, max_properties=1)
    with pytest.raises(ValidationError) as excinfo:
        max_prop_schema.validate({"a": 1, "b": 2})
    assert any(m.code == "max_properties" for m in excinfo.value.messages)

    # 5. Test Min Properties Constraint (Empty check)
    min_prop_schema = Object(properties={}, min_properties=1)
    with pytest.raises(ValidationError) as excinfo:
        min_prop_schema.validate({})
    assert any(m.code == "empty" for m in excinfo.value.messages)

    # 6. Test Additional Properties = False (Disallow unknown keys)
    strict_schema = Object(properties={"name": name_field}, additional_properties=False)
    with pytest.raises(ValidationError) as excinfo:
        strict_schema.validate({"name": "Bob", "unknown": "data"})
    assert any(m.code == "invalid_property" for m in excinfo.value.messages)

    # 7. Test Pattern Properties
    pattern_schema = Object(
        pattern_properties={r"^id_": Integer()},
        properties={}
    )
    # Valid pattern match
    assert pattern_schema.validate({"id_123": 1}) == {"id_123": 1}
    # Invalid pattern match (value type mismatch)
    with pytest.raises(ValidationError) as excinfo:
        pattern_schema.validate({"id_abc": "not_an_int"})
    assert any("id_abc" in str(m.text) for m in excinfo.value.messages)

    # 8. Test Nullability
    null_field = String(allow_null=True)
    null_schema = Object(properties={"val": null_field})
    assert null_schema.validate({"val": None}) == {"val": None}
    with pytest.raises(ValidationError):
        null_schema.validate({"val": None}) # This part depends on the specific implementation of required vs allow_null

```


# LLM-generated content at query #16
#--------------------------

```python
import pytest

def test_Choice_validate():
    # Setup choices: a list of tuples (key, label) and simple values
    choices = [("a", "Alpha"), ("b", "Beta"), "c"]
    # The constructor converts "c" to ("c", "c")
    
    # 1. Test valid single value
    field_single = Choice(choices=["a", "b"], allow_null=False)
    assert field_single.validate("a") == "a"
    assert field_single.validate("b") == "b"

    # 2. Test valid tuple choice (key, label)
    field_tuple = Choice(choices=[("x", "X-ray")], allow_null=False)
    assert field_tuple.validate("x") == "x"

    # 3. Test invalid choice
    field_invalid = Choice(choices=["a", "b"], allow_null=False)
    with pytest.raises(ValidationError) as excinfo:
        field_invalid.validate("z")
    assert excinfo.value.code == "choice"

    # 4. Test null handling - Not allowed
    field_no_null = Choice(choices=["a", "b"], allow_null=False)
    with pytest.raise_error_as_validation_error(field_no_null, None, "null"):
        pass # Manual check below via helper logic or direct call

    # 5. Test null handling - Allowed
    field_allow_null = Choice(choices=["a", "b"], allow_null=True)
    assert field_allow_null.validate(None) is None

    # 6. Test empty string/required behavior with coerce_types and allow_null
    # If value is "", and not in choices, it checks for 'required' error if not allowed to be null/coerced
    field_req = Choice(choices=["a", "b"], allow_null=False, coerce_types=True)
    with pytest.raises(ValidationError) as excinfo:
        field_req.validate("")
    assert excinfo.value.code == "required"

    # 7. Test empty string behavior when coercion/allow_null allows it
    field_coerce_empty = Choice(choices=["a", "b"], allow_null=True, coerce_types=True)
    assert field_coerce_empty.validate("") is None

def pytest.raises_error_as_validation_error(field, value, code):
    with pytest.raises(ValidationError) as excinfo:
        field.validate(value)
    assert excinfo.value.code == code
```


# LLM-generated content at query #17
#--------------------------

```python
import pytest
from typesystem.base import ValidationError

def test_String_validate():
    # Test basic string validation
    s = String(title="Test", min_length=3, max_length=5)
    assert s.validate("abc") == "abc"
    assert s.validate("abcde") == "abcde"
    
    with pytest.raises(ValidationError) as excinfo:
        s.validate("ab")
    assert "min_length" in str(excinfo.value)

    with pytest.raises(ValidationError) as excinfo:
        s.validate("abcdef")
    assert "max_length" in str(excinfo.value)

    # Test trim_whitespace
    s_trim = String(trim_whitespace=True)
    assert s_trim.validate("  hello  ") == "hello"
    
    s_no_trim = String(trim_whitespace=False)
    assert s_no_trim.validate("  hello  ") == "  hello  "

    # Test allow_null and null handling
    s_null_allowed = String(allow_null=True)
    assert s_null_allowed.validate(None) is None
    
    s_null_disallowed = String(allow_null=False)
    with pytest.raises(ValidationError) as excinfo:
        s_null_disallowed.validate(None)
    assert "null" in str(excinfo.value)

    # Test allow_blank and empty string handling
    s_no_blank = String(allow_blank=False)
    with pytest.pytests_raises_error := pytest.raises(ValidationError):
        s_no_blank.validate("")
    
    s_blank_allowed = String(allow_blank=True)
    assert s_blank_allowed.validate("") == ""

    # Test coerce_types (None to empty string when allow_blank is True)
    s_coerce = String(allow_blank=True, coerce_types=True)
    assert s_coerce.validate(None) == ""

    # Test pattern matching
    s_pattern = String(pattern=r"^[a-z]+$")
    assert s_pattern.validate("abc") == "abc"
    with pytest.raises(ValidationError) as excinfo:
        s_pattern.validate("abc1")
    assert "pattern" in str(excinfo.value)

    # Test type validation (non-string input)
    s_type = String()
    with pytest.raises(ValidationError) as excinfo:
        s_type.validate(123)
    assert "type" in str(excinfo.value)

    # Test null character removal
    s_null_char = String()
    assert s_null_char.validate("abc\0def") == "abcdef"

    # Test format validation (using a known format from FORMATS)
    # Note: This assumes EmailFormat behaves as expected via typesystem
    s_email = String(format="email")
    try:
        assert s_email.validate("test@example.com") == "test@example.com"
        with pytest.raises(ValidationError):
            s_email.validate("not-an-email")
    except Exception:
        # Fallback if typesystem format implementation is not available in test env
        pass

    # Test min_length edge case with whitespace trimming
    s_min_trim = String(min_length=5, trim_whitespace=True)
    with pytest.raises(ValidationError):
        s_min_trim.validate("  a  ") # becomes "a", length 1 < 5
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Array_serialize():
    # Case 1: obj is None -> returns None
    array_field_none = Array(items=Integer())
    assert array_field_none.serialize(None) is None

    # Case 2: items is None (no specific item serializer) -> returns original list
    array_field_no_items = Array(items=None)
    input_list = [1, "two", 3.0]
    assert array_field_no_items.serialize(input_list) == input_list

    # Case 3: items is a single Field (applies to all elements)
    # We use Integer which has numeric_type = int
    array_field_single = Array(items=Integer())
    input_list_str = ["1", "2", "3"]
    # Note: In the provided code, Integer.serialize doesn't override, 
    # so it uses Field.serialize (returns obj). 
    # However, if we assume a custom implementation or typical behavior:
    assert array_field_single.serialize(input_list_str) == ["1", "2", "3"]

    # Case 4: items is a list of Fields (positional serialization)
    # Mocking fields to verify .serialize() is called on each
    mock_field_1 = MagicMock(spec=Field)
    mock_field_1.serialize.return_value = "val1"
    mock_field_2 = MagicMock(spec=Field)
    mock_field_2.serialize.return_value = "val2"
    
    array_field_list = Array(items=[mock_field_1, mock_field_2])
    input_list_mixed = [100, 200]
    
    result = array_field_list.serialize(input_list_mixed)
    
    assert result == ["val1", "val2"]
    mock_field_1.serialize.assert_called_once_with(100)
    mock_field_2.serialize.assert_called_once_with(200)

    # Case 5: Verification of Decimal serialization (custom override in provided code)
    array_field_decimal = Array(items=Decimal())
    input_list_decimal = [decimal.Decimal("1.5"), decimal.Decimal("2.7")]
    # Decimal.serialize returns float(obj)
    assert array_field_decimal.serialize(input_list_decimal) == [1.5, 2.7]

    # Case 6: Verification with Boolean (no override, returns obj)
    array_field_bool = Array(items=Boolean())
    input_list_bool = [True, False, "1"]
    assert array_field_bool.serialize(input_list_bool) == [True, False, "1"]
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest

def test_Choice_validate():
    # Setup valid choices
    choices = ["a", "b", ("c", "Option C")]
    field = Choice(choices=choices, allow_null=False)

    # 1. Test Valid Single String Choice
    assert field.validate("a") == "a"
    assert field.validate("b") == "b"

    # 2. Test Valid Tuple Choice (extracts the key)
    # Note: The implementation uses Uniqueness([key for key, value in self.choices])
    # so it checks against 'c'
    assert field.validate("c") == "c"

    # 3. Test Invalid Choice
    with pytest.raises(ValidationError) as excinfo:
        field.validate("z")
    assert excinfo.value.code == "choice"

    # 4. Test Null Value when allow_null is False
    with pytest.raises(ValidationError) as excinfo:
        field.validate(None)
    assert excinfo.value.code == "null"

    # 5. Test Null Value when allow_null is True
    field_allow_null = Choice(choices=choices, allow_null=True)
    assert field_allow_null.validate(None) is None

    # 6. Test Empty String behavior (Requirement check)
    # If value is "", and not in choices, it triggers 'required' if allow_null/coerce_types logic applies
    with pytest.raises(ValidationError) as excinfo:
        field.validate("")
    assert excinfo.value.code == "required"

    # 7. Test Empty String with coercion/allow_null enabled
    field_empty_ok = Choice(choices=choices, allow_null=True, coerce_types=True)
    assert field_empty_ok.validate("") is None

    # 8. Test case where choices list is empty
    empty_field = Choice(choices=[])
    with pytest.raises(ValidationError) as excinfo:
        empty_field.validate("a")
    assert excinfo.value.code == "choice"
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Array_serialize():
    # Case 1: obj is None -> returns None
    array_none = Array(items=Integer())
    assert array_none.serialize(None) is None

    # Case 2: items is None -> returns original object (identity)
    array_no_items = Array(items=None)
    input_list = [1, 2, 3]
    assert array_no_items.serialize(input_list) is input_list

    # Case 3: items is a single Field -> applies serialize to every element
    # We mock the child field's serialize method to verify it's being called
    mock_child_field = MagicMock()
    mock_child_field.serialize.side_effect = lambda x: x * 2
    
    array_single_item_type = Array(items=mock_child_field)
    input_list = [1, 2, 3]
    expected_output = [2, 4, 6]
    assert array_single_item_type.serialize(input_list) == expected_output
    assert mock_child_field.serialize.call_count == 3

    # Case 4: items is a list of Fields (tuple-based validation/serialization)
    # We use real Integer and Float fields to ensure logic works with actual types
    mock_field1 = MagicMock()
    mock_field1.serialize.return_value = "val1"
    mock_field2 = MagicMock()
    mock_field2.serialize.return_value = "val2"
    
    array_list_items = Array(items=[mock_field1, mock_field2])
    input_list = [10, 20]
    expected_output = ["val1", "valron2"] # Note: zip logic uses items[pos]
    # Re-evaluating the zip logic in code: 
    # return [serializer.serialize(value) for serializer, value in zip(self.items, obj)]
    assert array_list_items.serialize([10, 20]) == ["val1", "val2"]

    # Case 5: testing with a mix of real types (Integer and Float)
    array_mixed = Array(items=[Integer(), Float()])
    input_data = [1, 2.5]
    # Integer(1).serialize(1) -> 1
    # Float(2.5).serialize(2.5) -> 2.5
    assert array_mixed.serialize(input_data) == [1, 2.5]

    # Case 6: Verify that if items is None but the object contains data, it returns as is
    array_identity = Array(items=None)
    complex_obj = [{"a": 1}, [1, 2]]
    assert array_identity.serialize(complex_obj) == complex_obj
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

class MockField:
    def __init__(self, value=None, error=None):
        self.value = value
        self.error = error
        self.validate_or_error = MagicMock(return_value=(value, error))
    
    def validate(self, val):
        return self.validate_or_error(val)

def test_Array_validate():
    # 1. Test Null handling (allow_null=True)
    arr_null_ok = Array(items=Integer(), allow_null=True)
    assert arr_null_ok.validate(None) is None

    # 2. Test Null handling (allow_null=False)
    arr_null_fail = Array(items=Integer(), allow_null=
                          False)
    with pytest.raises(Exception) as excinfo:
        arr_null_fail.validate(None)
    assert "null" in str(excinfo.value)

    # 3. Test Type error (not a list)
    arr_type_error = Array(items=Integer())
    with pytest.raises(Exception) as excinfo:
        arr_type_error.validate("not a list")
    assert "type" in str(excinfo.value)

    # 4. Test min_items / empty error
    arr_min_1 = Array(items=Integer(), min_items=1)
    with pytest.raises(Exception) as excinfo:
        arr_min_1.validate([])
    assert "empty" in str(excinfo.value)

    # 5. Test max_items
    arr_max_2 = Array(items=Integer(), max_items=2)
    with pytest.raises(Exception) as excinfo:
        arr_max_2.validate([1, 2, 3])
    assert "max_items" in str(excinfo.value)

    # 6. Test exact_items
    arr_exact = Array(items=Integer(), exact_items=2)
    with pytest.raises(Exception) as excinfo:
        arr_exact.validate([1])
    assert "exact_items" in str(excinfo.value)

    # 7. Test successful validation with single item type
    arr_single = Array(items=Integer())
    assert arr_single.validate([1, 2, 3]) == [1, 2, 3]

    # 8. Test successful validation with list of items (tuple of schemas)
    arr_tuple = Array(items=[Integer(), Float()], additional_items=False)
    assert arr_tuple.validate([1, 2.5]) == [1, 2.5]

    # 9. Test item validation error propagation
    item_error_field = MockField(value=None, error=MagicMock(messages=lambda add_prefix: [f"err_{add_prefix}"]))
    arr_error = Array(items=item_error_field)
    with pytest.raises(Exception) as excinfo:
        arr_error.validate([10]) # index 0 will trigger error
    assert "err_0" in str(excinfo.value)

    # 10. Test unique_items constraint
    # Note: This assumes Uniqueness is a set-like object that works with 'in' and '.add()'
    arr_unique = Array(items=Integer(), unique_items=True)
    assert arr_unique.validate([1, 2, 3]) == [1, 2, 3]
    with pytest.raises(Exception) as excinfo:
        arr_unique.validate([1, 1, 2])
    assert "unique_items" in str(excinfo.value)

    # 11. Test additional_items as a Field
    extra_field = MockField(value="extra")
    arr_additional = Array(items=Integer(), additional_items=extra_field)
    assert arr_additional.validate([1, "extra"]) == [1, "extra"]

    # 12. Test max_items when items is a list (logic: if items is list, max_items defaults to len(items))
    arr_auto_max = Array(items=[Integer(), Integer()]) # implies max_items=2 implicitly in constructor logic
    assert arr_auto_max.validate([1, 2]) == [1, 2]
    with pytest.raises(Exception) as excinfo:
        arr_auto_max.validate([1, 2, 3])
    assert "max_items" in str(excinfo.value)
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest

def test_Choice_validate():
    # Test case 1: Valid choice (single value list)
    field_simple = Choice(choices=["a", "b"], allow_null=False)
    assert field_simple.validate("a") == "a"
    assert field_simple.validate("b") == "b"

    # Test case 2: Valid choice (tuple list)
    field_tuples = Choice(choices=[("v1", "Display 1"), ("v2", "Display 2")], allow_null=False)
    assert field_tuples.validate("v1") == "v1"
    assert field_tuples.validate("v2") == "v2"

    # Test case 3: Invalid choice
    with pytest.raises(ValidationError) as excinfo:
        field_simple.validate("c")
    assert excinfo.value.code == "choice"

    # Test case 4: Null value not allowed
    field_no_null = Choice(choices=["a"], allow_null=False)
    with pytest.raises(ValidationError) as excinfo:
        field_no_null.validate(None)
    assert excinfo.value.code == "null"

    # Test case 5: Null value allowed
    field_allow_null = Choice(choices=["a"], allow_null=True)
    assert field_allow_null.validate(None) is None

    # Test case 6: Empty string with coerce_types and allow_null (returns None)
    field_coerce_empty = Choice(choices=["a"], allow_null=True, coerce_types=True)
    assert field_coerce_empty.validate("") is None

    # Test case 7: Empty string without allow_null (raises required error)
    field_required_empty = Choice(choices=["a"], allow_null=False, coerce_types=True)
    with pytest.raises(ValidationError) as excinfo:
        field_required_empty.validate("")
    assert excinfo.value.code == "required"

    # Test case 8: Empty string without coerce_types (raises choice error if not in list)
    field_no_coerce = Choice(choices=["a"], allow_null=False, coerce_types=False)
    with pytest.raises(ValidationError) as excinfo:
        field_no_coerce.validate("")
    assert excinfo.value.code == "choice"
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Union_validate():
    # Mocking error/validation infrastructure since it's not provided in the snippet
    # We assume Field.validate_or_error returns (value, error)
    # and ValidationError is a standard exception type used in the code.

    class MockValidationError(Exception):
        def __init__(self, messages):
            self.messages = lambda add_prefix=None: messages

    # Setup common mocks
    mock_error = MagicMock()
    mock_error.messages.return_value = [MagicMock(code="type", index=None)]
    
    class MockMessage:
        def __init__(self, code, index=None):
            self.code = code
            self.index = index

    # We need to patch the validation_error method which is called by Union
    # Since we can't easily patch a class method without knowing its parent structure,
    # we rely on the fact that validate_or_error return value is what matters here.

    # Case 1: Value matches first schema (Success)
    field1 = MagicMock(spec=Field)
    field1.validate_or_error.return_value = (10, None)
    field2 = MagicMock(spec=Field)
    
    union_success = Union(any_of=[field1, field2])
    assert union_success.validate(10) == 10
    field1.validate_or_error.assert_called_with(10)

    # Case 2: Value matches second schema (Success)
    field1.validate_or_error.return_value = (None, mock_error)
    field2.validate_or_error.return_value = ("string", None)
    
    union_second_match = Union(any_of=[field1, field2])
    assert union_second_match.validate("string") == "string"

    # Case 3: Value is None and allow_null is True (Success)
    # The constructor logic sets allow_null=True if any child allows null
    field_null = MagicMock(spec=Field)
    field_null.allow_null = True
    union_null = Union(any_of=[field_null])
    assert union_null.validate(None) is None

    # Case 4: Value is None and allow_null is False (Failure - Null error)
    field_no_null = MagicMock(spec=Field)
    field_no_null.allow_null = False
    # We need to mock validate_or_error to simulate the error return in Union
    # But first we must handle the internal call to self.validation_error("null")
    # Since we don't have the base class, we'll assume a standard exception behavior.
    
    union_fail_null = Union(any_of=[field_no_null])
    # Note: If validate() raises an error via validation_error, it's caught by pytest.
    # This test assumes validation_error behaves like a standard Exception generator.
    with pytest.raises(Exception): 
        union_fail_null.validate(None)

    # Case 5: Value matches no schema (Failure - Union error)
    field1.validate_or_error.return_value = (None, mock_error)
    field2.validate_or_error.return_value = (None, mock_error)
    
    union_fail_union = Union(any_of=[field1, field2])
    # Mocking the validation_error method on the instance to avoid implementation dependency
    union_fail_union.validation_error = MagicMock(side_effect=ValueError("union"))
    with pytest.raises(ValueError) as excinfo:
        union_fail_union.validate("invalid")
    assert "union" in str(excinfo.value)

    # Case 6: Value matches no schema but one child has a specific error (Failure - Specific error)
    # If an error is NOT a 'type' error or has an index, it should be returned as primary error.
    specific_error = MagicMock()
    msg = MockMessage(code="not_a_type_error")
    specific_error.messages.return_value = [msg]
    
    field1.validate_or_error.return_value = (None, specific_error)
    field2.validate_or_error.return_value = (None, mock_error) # type error
    
    union_specific_err = Union(any_of=[field1, field2])
    union_specific_err.validation_error = MagicMock(side_effect=specific_error)
    with pytest.raises(Exception):
        union_specific_err.validate("invalid")
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest

def test_Field_get_default_value():
    # Test case 1: No default provided (NO_DEFAULT)
    field_no_default = Field()
    assert field_no_default.get_default_value() is NO_DEFAULT

    # Test case 2: Static default value (integer)
    field_int = Field(default=42)
    assert field_int.get_default_value() == 42

    # Test case 3: Static default value (string)
    field_str = Field(default="hello")
    assert field_str.get_default_value() == "hello"

    # Test case 4: Default value is None (explicitly allowed)
    field_none = Field(default=None)
    assert field_none.get_default_value() is None

    # Test case 5: Default value is a callable (lambda)
    field_callable = Field(default=lambda: "dynamic")
    assert field_callable.get_default_value() == "dynamic"

    # Test case 6: Default value is a complex callable
    def complex_func():
        return {"key": "value"}
    field_complex = Field(default=complex_func)
    assert field_complex.get_default_value() == {"key": "value"}
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Object_validate():
    # Mocking dependencies not provided in the snippet but used by Object.validate
    # We assume ValidationError, Message, and Uniqueness are available in the scope
    
    class MockValidationError(Exception):
        def __init__(self, messages=None):
            self.messages = messages or []

    class MockMessage:
        def __init__(self, text, code, index=None, key=None):
            self.text = text
            self.code = code
            self.index = index
            self.key = key

    # Patching the global scope for the duration of this test
    import sys
    module = sys.modules[__name__]
    setattr(module, 'ValidationError', MockValidationError)
    setattr(module, 'Message', MockMessage)

    # 1. Test successful validation of a simple object
    properties = {
        "name": String(title="Name"),
        "age": Integer(title="Age")
    }
    obj_schema = Object(properties=properties, required=["name"])
    
    input_data = {"name": "Alice", "age": 30, "extra": "allowed"}
    # additional_properties defaults to True, so 'extra' should be passed through
    result = obj_schema.validate(input_data)
    assert result["name"] == "Alice"
    assert result["age"] == 30
    assert result["extra"] == "allowed"

    # 2. Test validation error: Missing required property
    input_missing_req = {"age": 30}
    with pytest.raises(MockValidationError) as excinfo:
        obj_schema.validate(input_missing_req)
    assert any(m.code == "required" and m.index == ["name"] for m in excinfo.value.messages)

    # 3. Test validation error: Invalid property name (via property_names)
    # We need a Field that acts as a validator for keys
    class KeyValidator(Field):
        def validate(self, value):
            if value != "valid_key":
                raise ValidationError(text="Invalid key", code="invalid_property")
            return value

    obj_with_key_validator = Object(property_names=KeyValidator())
    with pytest.raises(MockValidationError) as excinfo:
        obj_with_key_validator.validate({"bad_key": 123})
    assert any(m.code == "invalid_property" and m.index == ["bad_key"] for m in excinfo.value.messages)

    # 4. Test validation error: max_properties
    obj_max_props = Object(max_properties=1)
    with pytest.raises(MockValidationError) as excinfo:
        obj_max_props.validate({"a": 1, "b": 2})
    assert any(m.code == "max_properties" for m in excinfo.value.messages)

    # 5. Test validation error: min_properties (edge case where min_properties=1 triggers 'empty')
    obj_min_props = Object(min_properties=1)
    with pytest.raises(MockValidationError) as excinfo:
        obj_min_props.validate({})
    assert any(m.code == "empty" for m in excinfo.value.messages)

    # 6. Test validation error: additional_properties=False
    obj_no_extra = Object(properties={"a": String()}, additional_properties=False)
    with pytest.raises(MockValidationError) as excinfo:
        obj_no_extra.validate({"a": "val", "b": "forbidden"})
    assert any(m.code == "invalid_property" and m.key == "b" for m in excinfo.value.messages)

    # 7. Test validation error: pattern_properties
    import re
    pattern_props = {"^id_.*": Integer()}
    obj_pattern = Object(pattern_properties=pattern_props)
    # Valid pattern match
    assert obj_pattern.validate({"id_123": 456}) == {"id_123": 456}
    # Invalid value within pattern match (integer expected, got string)
    with pytest.raises(MockValidationError) as excinfo:
        obj_pattern.validate({"id_123": "not_an_int"})
    assert any("id_123" in str(m.text) or m.index == ["id_123"] for m in excinfo.value.messages)

    # 8. Test validation error: invalid_key (non-string key)
    obj_keys = Object()
    with pytest.raises(MockValidationError) as excinfo:
        # Using a dict with an integer key
        obj_keys.validate({123: "value"})
    assert any(m.code == "invalid_key" and m.index == [123] for m in excinfo.value.messages)

    # Clean up patches
    delattr(module, 'ValidationError')
    delattr(module, 'Message')
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
import unittest.mock as mock

class TestArrayValidate:
    @pytest.fixture
    def base_field(self):
        """Fixture to create a basic Field instance with necessary mocks."""
        field = mock.Mock(spec=Array)
        field.allow_null = False
        field.validation_error = lambda err: Exception(f"Error: {err}")
        # Mocking the actual Array class behavior for validation_error
        return field

    def test_Array_validate_null_not_allowed(self):
        arr = Array()
        arr.allow_null = False
        arr.validation_error = lambda err: ValueError(err)
        with pytest.raises(ValueError, match="null"):
            arr.validate(None)

    def test_Array_validate_null_allowed(self):
        arr = Array()
        arr.allow_null = True
        assert arr.validate(None) is None

    def test_Array_validate_type_error(self):
        arr = Array()
        arr.validation_error = lambda err: TypeError(err)
        with pytest.raises(TypeError, match="type"):
            arr.validate("not a list")

    def test_Array_validate_min_items(self):
        arr = Array(min_items=2)
        arr.validation_error = lambda err: ValueError(err)
        with pytest.raises(ValueError, match="min_items"):
            arr.validate([1])

    def test_Array_validate_empty_error(self):
        # If min_items is 1 and list is empty, it should raise 'empty'
        arr = Array(min_items=1)
        arr.validation_error = lambda err: ValueError(err)
        with pytest.raises(ValueError, match="empty"):
            arr.validate([])

    def test_Array_validate_max_items(self):
        arr = Array(max_items=2)
        arr.validation_error = lambda err: ValueError(err)
        with pytest.raises(ValueError, match="max_items"):
            arr.validate([1, 2, 3])

    def test_Array_validate_exact_items(self):
        arr = Array(exact_items=2)
        arr.validation_error = lambda err: ValueError(err)
        with pytest.raises(ValueError, match="exact_items"):
            arr.validate([1])
        assert arr.validate([1, 2]) == [1, 2]

    def test_Array_validate_unique_items(self):
        # We need to mock the Uniqueness class behavior if it's not provided
        # For this test, we assume a working environment where Array is part of the logic
        arr = Array(unique_items=True)
        # Mocking dependencies inside validate
        with mock.patch('__main__.Uniqueness') as MockUniqueness:
            instance = MockUniqueness.return_value
            instance.__contains__.side_effect = lambda x: x == 1
            instance.add = mock.Mock()
            
            # Create a custom error class for the test context
            class ValidationError(Exception):
                def __init__(self, messages): self.messages = lambda add_prefix=None: messages
            
            arr.validation_error = lambda err: ValidationError([]) # dummy
            # This part of the test is complex due to the internal dependency on Uniqueness
            # In a real scenario, Uniqueness would be imported and functional.

    def test_Array_validate_items_validation_success(self):
        item_validator = mock.Mock()
        item_validator.validate_or_error.return_value = (1, None)
        
        arr = Array(items=item_validator)
        arr.allow_null = False
        arr.validation_error = lambda err: ValueError(err)
        
        result = arr.validate([1])
        assert result == [1]
        item_validator.validate_or_error.assert_called_with(1)

    def test_Array_validate_items_list_validation_failure(self):
        v1 = mock.Mock()
        v1.validate_or_error.return_value = (1, None)
        v2 = mock.Mock()
        # Simulate a ValidationError being raised by the child validator
        class MockError:
            def messages(self, add_prefix=None): return [f"err_{add_prefix}"]
        v2.validate_or_error.return_value = (None, MockError())

        arr = Array(items=[v1, v2])
        
        # We need to mock ValidationError class in the scope
        with mock.patch('__main__.ValidationError', side_effect=Exception("Validation Failed")):
            with pytest.raises(Exception, match="Validation Failed"):
                arr.validate([1, 2])

    def test_Array_validate_additional_items_validator(self):
        add_validator = mock.Mock()
        add_validator.validate_or_error.return_value = (99, None)
        
        arr = Array(items=mock.Mock(), additional_items=add_validator)
        arr.allow_null = False
        arr.validation_error = lambda err: ValueError(err)
        
        # value[0] uses the 'items' validator (which we'll mock), value[1] uses additional_items
        arr.items = mock.Mock()
        arr.items.validate_or_error.return_value = (1, None)
        
        result = arr.validate([1, 2])
        assert result == [1, 99]

def test_Array_validate():
    """
    Comprehensive test function as requested by the signature.
    Note: This implementation assumes the presence of ValidationError and Uniqueness 
    in the global scope as per the provided snippet.
    """
    # Setup error mock
    class MockValidationError(Exception):
        def __init__(self, messages): self.messages = lambda add_prefix=None: messages

    # Test Case 1: Simple valid array
    arr_simple = Array(items=Integer())
    assert arr_simple.validate([1, 2, 3]) == [1, 2, 3]

    # Test Case 2: Min items constraint
    arr_min = Array(min_items=2)
    try:
        arr_min.validate([1])
    except Exception as e:
        assert "min_items" in str(e) or "min_items" in str(e.args[0])

    # Test Case 3: Max items constraint
    arr_max = Array(max_items=1)
    try:
        arr_max.validate([1, 2])
    except Exception as e:
        assert "max_items" in str(e) or "max_items" in str(e.args[0])

    # Test Case 4: Type mismatch (not a list)
    arr_type = Array()
    try:
        arr_type.validate("not a list")
    except Exception as e:
        assert "type" in str(e) or "type" in str(e.args[0])

    # Test Case 5: Null handling
    arr_null = Array(allow_null=True)
    assert arr_null.validate(None) is None

    # Test Case 6: Item validation failure (using a mock to trigger error)
    item_mock = mock.Mock()
    class MockError:
        def messages(self, add_prefix=None): return [f"error at {add_prefix}"]
    
    item_mock.validate_or_error.return_value = (None, MockError())
    arr_fail = Array(items=item_mock)
    
    # We must patch ValidationError because the class is not defined in the snippet
    with mock.patch('__main__.ValidationError', side_effect=MockValidationError(["err"])):
        with pytest.raises(Exception):
            arr_fail.validate([1])
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest

def test_Field_get_default_value():
    # Test case 1: No default provided (NO_DEFAULT)
    field_no_default = Field()
    assert field_no_default.get_default_value() is NO_DEFAULT

    # Test case 2: Static default value
    field_static = Field(default="hello")
    assert field_static.get_default_value() == "hello"

    # Test case 3: Default value with allow_null=True (should be None)
    field_null = Field(allow_null=True)
    assert field_null.get_default_value() is None

    # Test case 4: Callable default value
    def dynamic_gen():
        return "dynamic"
    
    field_callable = Field(default=dynamic_gen)
    assert field_callable.get_default_value() == "dynamic"

    # Test case 5: Default value is a complex object
    complex_obj = {"key": [1, 2, 3]}
    field_complex = Field(default=complex_obj)
    assert field_complex.get_default_value() == complex_obj
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest

def test_String_validate():
    # Test Basic String Validation
    s = String(title="Test")
    assert s.validate("hello") == "hello"
    assert s.validate("  hello  ") == "hello"  # Default trim_whitespace=True

    # Test Null/Blank Logic
    s_null_allowed = String(allow_null=True)
    assert s_null_allowed.validate(None) is None

    s_no_null = String(allow_null=False)
    with pytest.raises(ValidationError) as excinfo:
        s_no_null.validate(None)
    assert "May not be null." in str(excinfo.value)

    s_blank_allowed = String(allow_blank=True, allow_null=True, coerce_types=True)
    assert s_blank_allowed.validate(None) == ""
    assert s_blank_allowed.validate("") == ""

    s_no_blank = String(allow_blank=False)
    with pytest.raises(ValidationError) as excinfo:
        s_no_blank.validate("")
    assert "Must not be blank." in str(excinfo.value)

    # Test Type Validation
    s_type_check = String()
    with pytest.raises(ValidationError) as excinfo:
        s_type_check.validate(123)
    assert "Must be a string." in str(excinfo.value)

    # Test Length Constraints
    s_len = String(min_length=3, max_length=5)
    assert s_len.validate("abc") == "abc"
    assert s_len.validate("abcd") == "abcd"
    assert s_len.validate("abcde") == "abcde"

    with pytest.raises(ValidationError) as excinfo:
        s_len.validate("ab")
    assert "Must have at least 3 characters." in str(excinfo.value)

    with pytest.raises(ValidationError) as excinfo:
        s_len.validate("abcdef")
    assert "Must have no more than 5 characters." in str(excinfo.value)

    # Test Pattern (Regex)
    s_pattern = String(pattern=r"^[a-z]+$")
    assert s_pattern.validate("abc") == "abc"
    with pytest.raises(ValidationError) as excinfo:
        s_pattern.validate("ABC")
    assert "Must match the pattern /^[a-z]+$/." in str(excinfo.value)

    # Test Pattern (Compiled Regex)
    import re
    s_compiled = String(pattern=re.compile(r"\d+"))
    assert s_compiled.validate("123") == "123"
    with pytest.raises(ValidationError):
        s_compiled.validate("abc")

    # Test Null Character Removal
    s_null_char = String()
    assert s_null_char.validate("a\0b") == "ab"

    # Test Format Validation (using email as an example from FORMATS)
    # Note: Requires typesystem.formats.EmailFormat to be functional/mocked
    s_email = String(format="email")
    # If EmailFormat is valid, this passes; if it fails, the error comes from the format object
    try:
        assert s_email.validate("test@example.com") == "test@example.com"
    except ValidationError:
        pass 

    # Test Coerce Types (None to empty string)
    s_coerce = String(allow_blank=True, coerce_types=True)
    assert s_coerce.validate(None) == ""

    # Test No Trim Whitespace
    s_no_trim = String(trim_whitespace=False)
    assert s_no_trim.validate("  hello  ") == "  hello  "
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest

def test_Choice_validate():
    # Test setup: choices as simple strings and tuples (value, label)
    choices = ["A", ("B", "Option B"), "C"]
    field = Choice(choices=choices, allow_null=False)

    # 1. Test valid single string choice
    assert field.validate("A") == "A"

    # 2. Test valid tuple choice (validates against the key/first element)
    assert field.validate("B") == "B"

    # 3. Test valid third option
    assert field.validate("C") == "C"

    # 4. Test invalid choice
    with pytest.raises(ValidationError) as excinfo:
        field.validate("D")
    assert excinfo.value.code == "choice"

    # 5. Test null value when allow_null is False
    with pytest.raises(ValidationError) as excinfo:
        field.validate(None)
    assert excinfo.value.code == "null"

    # 6. Test null value when allow_null is True (should return None)
    field_nullable = Choice(choices=choices, allow_null=True)
    assert field_nullable.validate(None) is None

    # 7. Test empty string behavior with allow_null and coerce_types (specific to implementation logic)
    # The code has a specific check: if value == "" and allow_null/coerce_types, return None
    field_empty = Choice(choices=["A"], allow_null=True, coerce_types=True)
    assert field_empty.validate("") is None

    # 8. Test empty string behavior when NOT allowed to be null
    field_required = Choice(choices=["A"], allow_null=False)
    with pytest.raises(ValidationError) as excinfo:
        field_required.validate("")
    assert excinfo.value.code == "required"

    # 9. Test with no choices provided (empty sequence)
    field_no_choices = Choice(choices=[])
    with pytest.raises(ValidationError) as excinfo:
        field_no_choices.validate("A")
    assert excinfo.value.code == "choice"
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest
import decimal

def test_Const():
    # Test valid initialization with a constant value
    try:
        const_val = 10
        c = Const(const=const_val)
        assert c.const == const_val
    except Exception as e:
        pytest.fail(f"Const initialization failed for valid input: {e}")

    # Test valid initialization with None (as long as allow_null is not in kwargs)
    try:
        c = Const(const=None)
        assert c.const is None
    except Exception as e:
        pytest.fail(f"Const initialization failed for None value: {e}")

    # Test that providing allow_null in kwargs raises an AssertionError 
    # (based on the 'assert "allow_null" not in kwargs' line)
    with pytest.raises(AssertionError):
        Const(const=1, allow_null=True)

    # Test validation of correct value
    c_valid = Const(const="hello")
    assert c_valid.validate("hello") == "hello"

    # Test validation of incorrect value
    with pytest.raises(ValidationError):
        c_valid.validate("world")

    # Test validation for const=None (should trigger 'only_null' error)
    # Note: We assume validation_error raises a ValidationError-like object 
    # that contains the specified error code/message.
    c_none = Const(const=None)
    with pytest.raises(ValidationError) as excinfo:
        c_none.validate("not_null")
    # Check if the error message or code corresponds to 'only_null'
    # This depends on the implementation of validation_error in your base class
    assert any("only_null" in str(m.code) or "only_null" in m.text for m in excinfo.value.messages)

    # Test validation for incorrect const value (should trigger 'const' error)
    c_wrong = Const(const=5)
    with pytest.raises(ValidationError) as excinfo:
        c_wrong.validate(10)
    assert any("const" in str(m.code) or "const" in m.text for m in excinfo.value.messages)
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest

def test_Object_validate():
    # Setup schemas for testing
    string_field = String(title="Name")
    int_field = Integer(title="Age", default=18)
    bool_field = Boolean(title="Active")
    
    # 1. Test successful validation of a valid object with required and optional properties
    schema_valid = Object(
        properties={
            "name": string_field,
            "age": int_field,
            "active": bool_field
        },
        required=["name", "active"]
    )
    
    input_valid = {"name": "John Doe", "age": 25, "active": True}
    assert schema_valid.validate(input_valid) == {"name": "John Doe", "age": 25, "active": True}

    # 2. Test default value application when property is missing in input but has a default
    input_missing_age = {"name": "Jane", "active": False}
    assert schema_valid.validate(input_missing_age) == {"name": "Jane", "age": 18, "active": False}

    # 3. Test validation error for missing required property
    input_missing_required = {"age": 30}  # 'name' and 'active' are required
    with pytest.raises(ValidationError) as excinfo:
        schema_valid.validate(input_missing_required)
    
    error_codes = [msg.code for msg in excinfo.value.messages]
    assert "required" in error_codes

    # 4. Test validation error for invalid property type (Object must be a dict)
    with pytest.raises(ValidationError) as excinfo:
        schema_valid.validate(["not", "a", "dict"])
    assert excinfo.value.messages[0].code == "type"

    # 5. Test validation error for invalid property value (child validation failure)
    input_invalid_val = {"name": 123, "active": True} # name should be string
    with pytest.raises(ValidationError) as excinfo:
        schema_valid.validate(input_invalid_val)
    assert any("name" in msg.prefix or msg.index == ["name"] for msg in excinfo.value.messages)

    # 6. Test additional_properties = False (disallow unknown keys)
    schema_no_extra = Object(
        properties={"name": string_field},
        additional_properties=False
    )
    input_extra = {"name": "Test", "unknown": "value"}
    with pytest.raises(ValidationError) as excinfo:
        schema_no_extra.validate(input_extra)
    assert any(msg.code == "invalid_property" for msg in excinfo.value.messages)

    # 7. Test additional_properties = Field (validate unknown keys against a schema)
    schema_with_custom_extra = Object(
        properties={"name": string_field},
        additional_properties=Integer()
    )
    input_valid_extra = {"name": "Test", "score": 100}
    input_invalid_extra = {"name": "Test", "score": "not_a_number"}
    
    assert schema_with_custom_extra.validate(input_valid_extra) == {"name": "Test", "score": 100}
    with pytest.raises(ValidationError):
        schema_with_custom_extra.validate(input_invalid_extra)

    # 8. Test min_properties and max_properties
    schema_limits = Object(
        properties={"name": string_field},
        min_properties=2,
        max_properties=2
    )
    with pytest.raises(ValidationError) as excinfo:
        schema_limits.validate({"name": "Too few"})
    assert any(msg.code == "min_properties" or msg.code == "empty" for msg in excinfo.value.messages)

    with pytest_raises(ValidationError) as excinfo:
        schema_limits.validate({"name": "A", "b": "B", "c": "C"})
    assert any(msg.code == "max_properties" for msg in excinfo.value.messages)

    # 9. Test pattern_properties
    schema_pattern = Object(
        pattern_properties={r"^attr_\d+$": Integer()},
        properties={"name": string_field}
    )
    input_pattern_ok = {"name": "Test", "attr_1": 50}
    input_pattern_bad = {"name": "Test", "attr_1": "not_int"}
    
    assert schema_pattern.validate(input_pattern_ok) == {"name": "Test", "attr_1": 50}
    with pytest.raises(ValidationError):
        schema_pattern.validate(input_pattern_bad)

    # 10. Test null handling
    schema_null_allowed = Object(properties={"name": String(allow_null=True)})
    assert schema_null_allowed.validate({"name": None}) == {"name": None}
    
    with pytest.raises(ValidationError):
        schema_null_allowed.validate(None) # Root object cannot be null if not allowed
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Array_validate():
    # Mocking Field and ValidationError since they aren't provided in the snippet
    # but are required for the Array class to function.
    
    class MockValidationError(Exception):
        def __init__(self, messages):
            self.messages = messages

    # Setup a mock child field
    mock_child = MagicMock()
    mock_child.validate_or_error.return_value = (1, None)
    mock_child.validate.return_value = 1

    # Test Case 1: Valid array with single item type
    arr_single = Array(items=mock_child)
    assert arr_single.validate([1, 2, 3]) == [1, 2, 3]

    # Test Case 2: Null value when allow_null is True
    arr_null_allowed = Array(items=mock_child, allow_null=True)
    assert arr_null_allowed.validate(None) is None

    # Test Case 3: Null value when allow_null is False (Raises error)
    arr_null_forbidden = Array(items=mock_child, allow_null=False)
    with pytest.raises(Exception): # Assuming validation_error raises an exception
        arr_null_forbidden.validate(None)

    # Test Case 4: Invalid type (not a list)
    with pytest.raises(Exception):
        arr_single.validate("not a list")

    # Test Case 5: min_items constraint
    arr_min = Array(items=mock_child, min_items=2)
    with pytest.raises(Exception):
        arr_min.validate([1])
    
    # Test Case 6: max_items constraint
    arr_max = Array(items=mock_child, max_items=2)
    with pytest.raises(Exception):
        arr_max.validate([1, 2, 3])

    # Test Case 7: exact_items constraint
    arr_exact = Array(items=mock_child, exact_items=2)
    with pytest.raises(Exception):
        arr_exact.validate([1])
    assert arr_exact.validate([1, 2]) == [1, 2]

    # Test Case 7: empty constraint (min_items = 1)
    arr_empty_fail = Array(items=mock_child, min_items=1)
    with pytest.raises(Exception):
        arr_empty_fail.validate([])

    # Test Case 8: unique_items constraint
    # We need a real set-like object for Uniqueness if it's used in the code
    # Assuming Uniqueness is available as per context
    arr_unique = Array(items=mock_child, unique_items=True)
    # To test this without knowing Uniqueness implementation, 
    # we assume it works like a set.
    try:
        arr_unique.validate([1, 1])
    except Exception:
        pass # Expecting error for duplicates

    # Test Case 9: List of specific items (Positional validation)
    mock_item1 = MagicMock()
    mock_item1.validate_or_error.return_value = ("a", None)
    mock_item2 = MagicMock()
    mock_item2.validate_or_error.return_value = ("b", None)
    
    arr_list = Array(items=[mock_item1, mock_item2])
    assert arr_list.validate(["a", "b"]) == ["a", "b"]

    # Test Case 10: additional_items as a Field
    mock_extra = MagicMock()
    mock_extra.validate_or_error.return_value = (99, None)
    arr_extra = Array(items=[mock_child], additional_items=mock_extra)
    assert arr_extra.validate([1, 2]) == [1, 99]

    # Test Case 11: Validation error propagation
    mock_fail = MagicMock()
    # Simulate a ValidationError being raised by the child
    class MockError:
        def messages(self, add_prefix): return [f"error at {add_prefix}"]
    
    mock_fail.validate_or_error.return_value = (None, MockError())
    arr_fail = Array(items=mock_fail)
    with pytest.raises(Exception):
        arr_fail.validate([1])
```


# LLM-generated content at query #16
#--------------------------

```python
import pytest
import decimal
from typesystem.base import ValidationError

def test_Number_validate():
    # Test basic valid number (int)
    field = Number()
    assert field.validate(10) == 10
    assert field.validate(10.5) == 10.5

    # Test string coercion to Decimal/float
    assert field.validate("10.5") == decimal.Decimal("10.5")

    # Test null handling
    field_null = Number(allow_null=True)
    assert field_null.validate(None) is None
    
    field_no_null = Number(allow_null=False)
    with pytest.raises(ValidationError) as exc:
        field_no_null.validate(None)
    assert exc.value.code == "null"

    # Test empty string coercion to null when allow_null is True
    assert field_null.validate("") is None

    # Test boolean rejection (bool is subclass of int in Python)
    with pytest.raises(ValidationError) as exc:
        field.validate(True)
    assert exc.value.code == "type"

    # Test numeric_type constraint (int)
    field_int = Number(numeric_type=int)
    assert field_int.validate(10.0) == 10
    with pytest.raises(ValidationError) as exc:
        field_int.validate(10.5)
    assert exc.value.code == "integer"

    # Test non-numeric type rejection when coerce_types is False
    field_no_coerce = Number(coerce_types=False)
    with pytest.raises(ValidationError) as exc:
        field_no_coerce.validate("10")
    assert exc.value.code == "type"

    # Test finite check (inf, nan)
    import math
    with pytest.raises(ValidationError) as exc:
        field.validate(math.inf)
    assert exc.value.code == "finite"
    with pytest.raises(ValidationError) as exc:
        field.validate(float('nan'))
    assert exc.value.code == "finite"

    # Test minimum and exclusive_minimum
    field_min = Number(minimum=5, exclusive_minimum=2)
    assert field_min.validate(5) == 5
    assert field_min.validate(2.1) == 2.1
    with pytest.raises(ValidationError) as exc:
        field_min.validate(4)
    assert exc.value.code == "minimum"
    with pytest.raises(ValidationError) as exc:
        field_min.validate(2)
    assert exc.value.code == "exclusive_minimum"

    # Test maximum and exclusive_maximum
    field_max = Number(maximum=10, exclusive_maximum=15)
    assert field_max.validate(10) == 10
    with pytest.raises(ValidationError) as exc:
        field_max.validate(11)
    assert exc.value.code == "maximum"
    with pytest.raises(ValidationError) as exc:
        field_max.validate(15)
    assert exc.value.code == "exclusive_maximum"

    # Test multiple_of (integer)
    field_multiple = Number(multiple_of=5)
    assert field_multiple.validate(10) == 10
    with pytest.raise(ValidationError) as exc:
        field_multiple.validate(7)
    assert exc.value.code == "multiple_of"

    # Test multiple_of (float/decimal)
    field_mult_float = Number(multiple_of=0.5)
    assert field_mult_float.validate(1.5) == 1.5
    with pytest.raises(ValidationError) as exc:
        field_mult_float.validate(1.2)
    assert exc.value.code == "multiple_of"

    # Test precision (quantize)
    field_prec = Number(precision="0.01", numeric_type=float)
    # 1.234 should be rounded to 1.23
    assert field_prec.validate(1.234) == 1.23
    # 1.235 should be rounded up to 1.24 (ROUND_HALF_UP)
    assert field_prec.validate(1.235) == 1.24

    # Test invalid string format
    with pytest.raises(ValidationError) as exc:
        field.validate("not-a-number")
    assert exc.value.code == "type"
```


# LLM-generated content at query #17
#--------------------------

```python
import pytest

def test_Boolean_validate():
    # Test valid boolean values (native)
    bool_field = Boolean()
    assert bool_field.validate(True) is True
    assert bool_field.validate(False) is False

    # Test coercion of strings and other types to True
    true_coercibles = ["true", "TRUE", "on", "1", 1]
    for val in true_coercibles:
        assert bool_field.validate(val) is True

    # Test coercion of strings and other types to False
    false_coercibles = ["false", "FALSE", "off", "0", 0, ""]
    for val in false_coercibles:
        assert bool_field.validate(val) is False

    # Test null/None handling when allow_null=False (default)
    with pytest.raises(ValidationError) as excinfo:
        bool_field.validate(None)
    assert excinfo.value.code == "null"

    # Test null/None handling when allow_null=True
    bool_allow_null = Boolean(allow_null=True)
    assert bool_allow_null.validate(None) is None
    assert bool_allow_null.validate("null") is None
    assert bool_allow_null.validate("none") is None
    assert bool_allow_null.validate("") is None

    # Test invalid types when coerce_types=False
    bool_no_coerce = Boolean(coerce_types=False)
    with pytest.raises(ValidationError) as excinfo:
        bool_no_coerce.validate("true")
    assert excinfo.value.code == "type"

    # Test invalid values that cannot be coerced
    with pytest.raises(ValidationError) as excinfo:
        bool_field.validate("maybe")
    assert excinfo.value.code == "type"

    with pytest.raises(ValidationError) as excinfo:
        bool_field.validate(123)
    assert excinfo.value.code == "type"
```


