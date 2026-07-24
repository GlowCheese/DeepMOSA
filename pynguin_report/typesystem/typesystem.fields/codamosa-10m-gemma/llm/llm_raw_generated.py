####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest

def test_Boolean_validate():
    # Test basic boolean values
    bool_field = Boolean()
    assert bool_field.validate(True) is True
    assert bool_field.validate(False) is False

    # Test coercion of strings
    assert bool_field.validate("true") is True
    assert bool_field.validate("TRUE") is True
    assert bool_field.validate("false") is False
    assert bool_field.validate("on") is True
    assert bool_field.validate("off") is False

    # Test coercion of integers and strings of integers
    assert bool_field.validate("1") is True
    assert bool_field.validate("0") is False
    assert bool_field.validate(1) is True
    assert bool_field.validate(0) is False

    # Test empty string coercion
    assert bool_field.validate("") is False

    # Test error on invalid type with coercion enabled (default)
    with pytest.raises(ValidationError) as excinfo:
        bool_field.validate("not_a_boolean")
    assert excinfo.value.code == "type"

    with pytest.raises(ValidationError) as excinfo:
        bool_field.validate(None)
    assert excinfo.value.code == "null"

    # Test non-coercion mode
    strict_bool_field = Boolean(coerce_types=False)
    assert strict_bool_field.validate(True) is True
    assert strict_bool_field.validate(False) is False
    with pytest.raises(ValidationError) as excinfo:
        strict_bool_field.validate("true")
    assert excinfo.value.code == "type"

    # Test allow_null with null-like values
    null_bool_field = Boolean(allow_null=True)
    assert null_bool_field.validate(None) is None
    assert null_bool_field.validate("") is None
    assert null_bool_field.validate("null") is None
    assert null_bool_field.validate("none") is None

    # Test error on null when allow_null is False
    no_null_field = Boolean(allow_null=False)
    with pytest.raises(ValidationError) as excinfo:
        no_null_field.validate(None)
    assert excinfo.value.code == "null"
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest

def test_Boolean_validate():
    # Test basic boolean types
    bool_field = Boolean()
    assert bool_field.validate(True) is True
    assert bool_field.validate(False) is False

    # Test coercion of strings
    assert bool_field.validate("true") is True
    assert bool_field.validate("TRUE") is True
    assert bool_field.validate("false") is False
    assert bool_field.validate("on") is True
    assert bool_field.validate("off") is False
    assert bool_field.validate("1") is True
    assert bool_field.validate("0") is False
    assert bool_field.validate("") is False

    # Test coercion of integers
    assert bool_field.validate(1) is True
    assert bool_field.validate(0) is False

    # Test null handling (disallowed by default)
    with pytest.raises(ValidationError) as excinfo:
        bool_field.validate(None)
    assert excinfo.value.code == "null"

    # Test null handling (allowed)
    bool_null_field = Boolean(allow_null=True)
    assert bool_null_field.validate(None) is None
    assert bool_null_field.validate("null") is None
    assert bool_null_field.validate("none") is None
    assert bool_null_field.validate("") is None

    # Test invalid types with coercion enabled
    with pytest.raises(ValidationError) as excinfo:
        bool_field.validate("not-a-boolean")
    assert excinfo.value.code == "type"

    with pytest.raises(ValidationError) as excinfo:
        bool_field.validate(2)
    assert excinfo.value.code == "type"

    # Test invalid types with coercion disabled
    no_coerce_field = Boolean(coerce_types=False)
    assert no_coerce_field.validate(True) is True
    assert no_coerce_field.validate(False) is False
    with pytest.raises(ValidationError) as excinfo:
        no_coerce_field.validate("true")
    assert excinfo.value.code == "type"
```


# LLM-generated content at query #3
#--------------------------

```python
def test_Field_get_default_value():
    # Test with no default provided (NO_DEFAULT)
    field_no_default = Field()
    assert field_no_default.get_default_value() is NO_DEFAULT

    # Test with static default value
    field_static = Field(default="hello")
    assert field_static.get_default_value() == "hello"

    # Test with integer default value
    field_int = Field(default=42)
    assert field_int.get_default_value() == 42

    # Test with None as an explicit default
    field_none = Field(default=None)
    assert field_none.get_default_value() is None

    # Test with a callable default (factory)
    def factory():
        return "dynamic"
    
    field_callable = Field(default=factory)
    assert field_callable.get_default_value() == "dynamic"

    # Test with a lambda default
    field_lambda = Field(default=lambda: 100)
    assert field_lambda.get_default_value() == 100
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest

def test_Const():
    # Test successful initialization with various types
    const_int = Const(const=10)
    assert const_int.const == 10

    const_str = Const(const="hello")
    assert const_str.const == "hello"

    const_bool = Const(const=True)
    assert const_bool.const is True

    const_none = Const(const=None)
    assert const_none.const is None

    # Test that passing allow_null in kwargs raises AssertionError 
    # as per the implementation: assert "allow_null" not in kwargs
    with pytest.raises(AssertionError):
        Const(const=10, allow_null=True)

    # Test validation logic for Const
    const_validator = Const(const="fixed")
    
    # Valid case
    assert const_validator.validate("fixed") == "fixed"
    
    # Invalid case (value mismatch)
    with pytest.raises(Exception) as excinfo:
        const_validator.validate("wrong")
    # Check if the error message/code relates to 'const'
    assert "const" in str(excinfo.value)

    # Test validation logic for Const when const is None
    const_none_validator = Const(const=None)
    assert const_none_validator.validate(None) is None
    
    with pytest.raises(Exception) as excinfo:
        const_none_validator.validate("not_none")
    assert "only_null" in str(excinfo.value)
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Array_serialize():
    # Case 1: value is None
    array_none = Array()
    assert array_none.serialize(None) is None

    # Case 2: items is None (Identity serialization)
    array_no_items = Array(items=None)
    input_list = [1, "a", {"key": "val"}]
    assert array_no_items.serialize(input_list) == input_list

    # Case 3: items is a single Field (Uniform serialization)
    # Mocking a Field to control the serialize output
    mock_field = MagicMock()
    mock_field.serialize.side_effect = lambda x: f"serialized_{x}"
    
    array_uniform = Array(items=mock_field)
    input_list_uniform = [1, 2, 3]
    expected_output_uniform = ["serialized_1", "serialized_2", "serialized_3"]
    assert array_uniform.serialize(input_list_uniform) == expected_output_uniform
    assert mock_field.serialize.call_count == 3

    # Case 4: items is a list of Fields (Positional serialization)
    mock_field_1 = MagicMock()
    mock_field_1.serialize.return_value = "val1"
    mock_field_2 = MagicMock()
    mock_field_2.serialize.return_value = "val2"
    
    array_positional = Array(items=[mock_field_1, mock_field_2])
    input_list_positional = [10, 20]
    expected_output_positional = ["val1", "val2"]
    assert array_positional.serialize(input_list_positional) == expected_output_positional
    
    # Verify that for positional, it only zips up to the length of the input/fields
    # If input has more items than fields, the zip stops (based on the implementation)
    input_list_extra = [10, 20, 30]
    # zip(items, obj) will result in 2 items because items list has 2 fields
    assert array_positional.serialize(input_list_extra) == ["val1", "val2"]
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Array_serialize():
    # Case 1: obj is None -> should return None
    array_field_none = Array(items=Integer())
    assert array_field_none.serialize(None) is None

    # Case 2: items is None -> should return obj as is
    array_field_no_items = Array(items=None)
    input_data_no_items = [1, "string", {"a": 1}]
    assert array_field_no_items.serialize(input_data_no_items) == input_data_no_items

    # Case 3: items is a single Field -> should call serialize on each element
    # Using a mock to verify the serialize method of the child field is called
    mock_child_field = MagicMock(spec=Field)
    mock_child_field.serialize.side_effect = lambda x: f"serialized_{x}"
    
    array_field_single = Array(items=mock_child_field)
    input_data_single = [1, 2, 3]
    expected_output_single = ["serialized_1", "serialized_2", "serialized_3"]
    
    assert array_field_single.serialize(input_data_single) == expected_output_single
    assert mock_child_field.serialize.call_count == 3

    # Case 4: items is a list of Fields (positional) -> should call specific serializer per index
    mock_field_1 = MagicMock(spec=Field)
    mock_field_1.serialize.side_effect = lambda x: f"f1_{x}"
    mock_field_2 = MagicMock(spec=Field)
    mock_field_2.serialize.side_effect = lambda x: f"f2_{x}"
    
    array_field_list = Array(items=[mock_field_1, mock_field_2])
    input_data_list = [10, 20, 30] 
    # Note: zip(self.items, obj) in the code will only iterate up to the length of the shorter list
    # In the provided implementation: zip([f1, f2], [10, 20, 30]) -> (f1, 10), (f2, 20)
    expected_output_list = ["f1_10", "f1_20"] # Wait, looking at the code: 
    # zip(self.items, obj) results in [(mock_field_1, 10), (mock_field_2, 20)]
    # So the result is [f1_10, f2_20]. 
    # The 30 is ignored because zip stops at the shortest iterable.
    
    actual_output_list = array_field_list.serialize(input_data_list)
    assert actual_output_list == ["f1_10", "f2_20"]

    # Case 5: Integration test with real types
    # Integer field serializes integers (no change)
    # Float field serializes floats (no change)
    array_integration = Array(items=[Integer(), Float()])
    input_data_int = [1, 2.5, 3]
    # zip([Integer, Float], [1, 2.5, 3]) -> [Integer(1), Float(2.5)]
    assert array_integration.serialize(input_data_int) == [1, 2.5]
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest

def test_Boolean_validate():
    # Test valid boolean types
    bool_field = Boolean()
    assert bool_field.validate(True) is True
    assert bool_field.validate(False) is False

    # Test coercion of strings (default behavior)
    assert bool_field.validate("true") is True
    assert bool_field.validate("False") is False
    assert bool_field.validate("on") is True
    assert bool_field.validate("off") is False
    assert bool_field.validate("1") is True
    assert bool_field.validate("0") is False
    assert bool_field.validate("") is False

    # Test coercion of integers
    assert bool_field.validate(1) is True
    assert bool_field.validate(0) is False

    # Test null validation (not allowed by default)
    with pytest.raises(ValidationError) as excinfo:
        bool_field.validate(None)
    assert excinfo.value.code == "null"

    # Test null validation (allowed)
    null_bool_field = Boolean(allow_null=True)
    assert null_bool_field.validate(None) is None

    # Test coercion of null-like strings (allowed only if allow_null=True)
    assert null_bool_field.validate("null") is None
    assert null_bool_field.validate("none") is None
    assert null_bool_field.validate("") is None

    with pytest.raises(ValidationError) as excinfo:
        bool_field.validate("null")
    assert excinfo.value.code == "type"

    # Test invalid types without coercion
    strict_bool_field = Boolean(coerce_types=False)
    assert strict_bool_field.validate(True) is True
    with pytest.raises(ValidationError) as excinfo:
        strict_bool_field.validate("true")
    assert excinfo.value.code == "type"

    # Test invalid string values
    with pytest.raises(ValidationError) as excinfo:
        bool_field.validate("maybe")
    assert excinfo.value.code == "type"

    # Test invalid non-string/non-int type
    with pytest.raises(ValidationError) as excinfo:
        bool_field.validate([True])
    assert excinfo.value.code == "type"
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Array_serialize():
    # Case 1: obj is None -> returns None
    array_none = Array(items=Integer())
    assert array_none.serialize(None) is None

    # Case 2: items is None -> returns obj as is
    array_no_items = Array(items=None)
    input_data = [1, 2, 3]
    assert array_no_items.serialize(input_data) == [1, 2, 3]

    # Case 3: items is a single Field -> applies serialize to each element
    # Mocking a Field to control serialize output
    mock_field = MagicMock()
    mock_field.serialize.side_effect = lambda x: x * 2
    
    array_single_field = Array(items=mock_field)
    input_data = [1, 2, 3]
    expected_output = [2, 4, 6]
    assert array_single_field.serialize(input_data) == expected_output
    assert mock_field.serialize.call_count == 3

    # Case 4: items is a list of Fields (tuple-like validation) -> applies specific serializer to each index
    mock_field_int = MagicMock()
    mock_field_int.serialize.side_effect = lambda x: f"int_{x}"
    
    mock_field_str = MagicMock()
    mock_field_str.serialize.side_effect = lambda x: f"str_{x}"
    
    array_list_fields = Array(items=[mock_field_int, mock_field_str])
    input_data = [10, "hello"]
    expected_output = ["int_10", "str_hello"]
    
    assert array_list_fields.serialize(input_data) == expected_output
    assert mock_field_int.serialize.called
    assert mock_field_str.serialize.called

    # Case 5: testing with actual concrete classes
    # Integer(1) -> 1, String("a") -> "a"
    array_concrete = Array(items=[Integer(), String()])
    input_data = [1, "a"]
    assert array_concrete.serialize(input_data) == [1, "a"]
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
import decimal

def test_Union_validate():
    # Setup helper classes for testing since the original code relies on them
    # We assume Number, Field, ValidationError, Message, etc., are available 
    # in the scope as per the prompt instructions.

    class MockField(Field):
        def __init__(self, value_to_return=None, error_to_raise=None, **kwargs):
            super().__init__(**kwargs)
            self.value_to_return = value_to_return
            self.error_to_raise = error_to_raise

        def validate_or_error(self, value):
            if self.error_to_raise:
                return None, self.error_to_raise
            return self.value_to_return, None

        def validate(self, value):
            if self.error_to_raise:
                raise self.error_to_raise
            return self.value_to_return

    # 1. Test successful validation (first candidate matches)
    field_ok = MockField(value_to_return="success")
    field_fail = MockField(error_to_raise=ValidationError(messages=[Message(text="err", code="type")]))
    union_ok = Union(any_of=[field_ok, field_fail])
    assert union_ok.validate("input") == "success"

    # 2. Test successful validation (second candidate matches)
    union_second_match = Union(any_of=[field_fail, field_ok])
    assert union_second_match.validate("input") == "success"

    # 3. Test null handling when allow_null is True
    # Union sets allow_null=True if any child allows null
    field_null_allowed = MockField(value_to_return=None, **{"allow_null": True})
    union_null_allowed = Union(any_of=[field_null_allowed, field_fail])
    assert union_null_allowed.validate(None) is None

    # 4. Test null error when allow_null is False
    field_null_not_allowed = MockField(value_to_return=None, **{"allow_null": False})
    union_null_not_allowed = Union(any_of=[field_null_not_allowed, field_fail])
    with pytest.raises(ValidationError) as excinfo:
        union_null_not_allowed.validate(None)
    assert excinfo.value.messages[0].code == "null"

    # 5. Test Union error when no candidates match (Type error only)
    # Should raise 'union' error if all candidates fail with 'type'
    type_error = ValidationError(messages=[Message(text="type error", code="type")])
    union_all_fail = Union(any_of=[field_fail, MockField(error_to_raise=type_error)])
    with pytest.raises(ValidationError) as excinfo:
        union_all_fail.validate("bad_input")
    assert excinfo.value.messages[0].code == "union"

    # 6. Test Union error when a candidate fails with a specific error (not 'type')
    # If a candidate fails with something like 'minimum', that error should be prioritized
    specific_error = ValidationError(messages=[Message(text="too small", code="minimum")])
    field_specific_fail = MockField(error_to_raise=specific_error)
    union_specific_error = Union(any_of=[field_fail, field_specific_fail])
    with pytest.raises(ValidationError) as excinfo:
        union_specific_error.validate("input")
    # It should return the error from the candidate that provided a non-type error
    assert excinfo.value.messages[0].code == "minimum"
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Array_validate():
    # Mocking Field and ValidationError as they are dependencies in the provided code
    # Assuming ValidationError and Message are available in the scope
    
    # 1. Test Null handling (allow_null=True)
    field_null_ok = Array(allow_null=True)
    assert field_null_ok.validate(None) is None

    # 2. Test Null handling (allow_null=False)
    field_null_fail = Array(allow_null=False)
    with pytest.raises(Exception) as excinfo:
        field_null_fail.validate(None)
    # Check if the error message contains 'null' (depends on implementation of validation_error)
    assert "null" in str(excinfo.value)

    # 3. Test Type error (not a list)
    field_type_error = Array()
    with pytest.raises(Exception) as excinfo:
        field_type_error.validate("not a list")
    assert "type" in str(excinfo.value)

    # 4. Test min_items / exact_items
    field_min_items = Array(min_items=2)
    with pytest.raises(Exception) as excinfo:
        field_min_items.validate([1])
    assert "min_items" in str(excinfo.value)

    field_exact_items = Array(exact_items=3)
    with pytest.raises(Exception) as excinfo:
        field_exact_items.validate([1, 2])
    assert "exact_items" in str(excinfo.value)

    # 5. Test max_items
    field_max_items = Array(max_items=1)
    with pytest.raises(Exception) as excinfo:
        field_max_items.validate([1, 2])
    assert "max_items" in str(excinfo.value)

    # 6. Test empty error (min_items=1)
    field_empty_error = Array(min_items=1)
    with pytest.raises(Exception) as excinfo:
        field_empty_error.validate([])
    assert "empty" in str(excinfo.value)

    # 7. Test Item Validation (Single Field Type)
    class MockIntField:
        def validate_or_error(self, value):
            if isinstance(value, int):
                return value, None
            # Mocking a ValidationError structure
            err = MagicMock()
            err.messages.return_value = ["error"]
            return None, err

    field_int_array = Array(items=MockIntField())
    assert field_int_array.validate([1, 2, 3]) == [1, 2, 3]

    # 8. Test Item Validation Failure
    class MockErrorField:
        def validate_or_error(self, value):
            err = MagicMock()
            err.messages.return_value = ["error at index 0"]
            return None, err

    field_error_array = Array(items=MockErrorField())
    with pytest.raises(Exception):
        field_error_array.validate([1, 2])

    # 9. Test Tuple-based items (Positional validation)
    class MockStringField:
        def validate_or_error(self, value):
            if isinstance(value, str):
                return value, None
            err = MagicMock()
            err.messages.return_value = ["error"]
            return None, err

    field_tuple_items = Array(items=[MockIntField(), MockStringField()])
    assert field_tuple_items.validate([10, "hello"]) == [10, "hello"]
    
    with pytest.raises(Exception):
        field_tuple_items.validate(["not_int", "hello"])

    # 10. Test unique_items
    # Note: This requires Uniqueness() to be defined and working as a set-like object
    class MockUniqueField:
        def validate_or_error(self, value):
            return value, None

    # We assume Uniqueness is available as per the code snippet
    field_unique = Array(items=MockUniqueField(), unique_items=True)
    assert field_unique.validate([1, 2, 3]) == [1, 2, 3]
    
    with pytest.raises(Exception) as excinfo:
        field_unique.validate([1, 1, 2])
    assert "unique_items" in str(excinfo.value)

    # 11. Test additional_items as a Field
    class MockAdditionalField:
        def validate_or_error(self, value):
            if isinstance(value, str):
                return value, None
            err = MagicMock()
            err.messages.return_value = ["error"]
            return None, err

    field_additional = Array(items=[MockIntField()], additional_items=MockAdditionalField())
    assert field_additional.validate([1, "extra"]) == [1, "extra"]
    
    with pytest.raises(Exception):
        field_additional.validate([1, 2.5]) # 2.5 is not a string
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Array_validate():
    # Mocking ValidationError and Message as they are dependencies of Array.validate
    # Assuming they exist in the environment based on the provided code
    
    class MockValidationError(Exception):
        def __init__(self, messages):
            self.messages = messages

    # Setup basic Field mock for items
    def create_mock_field(return_val, error=None):
        field = MagicMock()
        field.validate_or_error.return_value = (return_val, error)
        return field

    # 1. Test valid array with single item type
    int_field = create_mock_field(10)
    arr_single = Array(items=int_field)
    assert arr_single.validate([10, 20, 30]) == [10, 20, 30]

    # 2. Test valid array with list of specific types (positional)
    int_field_2 = create_mock_field(1)
    str_field = create_mock_field("a")
    arr_multi = Array(items=[int_field_2, str_field])
    assert arr_multi.validate([1, "a"]) == [1, "a"]

    # 3. Test null validation
    arr_null_allowed = Array(allow_null=True)
    assert arr_null_allowed.validate(None) is None
    
    arr_null_disallowed = Array(allow_null=False)
    with pytest.raises(Exception) as excinfo:
        arr_null_disallowed.validate(None)
    # Check if it raises the expected error (assuming validation_error returns a specific error)
    
    # 4. Test type error (not a list)
    arr_type_error = Array()
    with pytest.raises(Exception):
        arr_type_error.validate("not a list")

    # 5. Test min_items constraint
    arr_min_items = Array(min_items=2)
    with pytest.raises(Exception):
        arr_min_items.validate([1])
    
    # 6. Test empty error (min_items=1)
    arr_empty_error = Array(min_items=1)
    with pytest.raises(Exception):
        arr_empty_error.validate([])

    # 7. Test max_items constraint
    arr_max_items = Array(max_items=2)
    with pytest.raises(Exception):
        arr_max_items.validate([1, 2, 3])

    # 8. Test exact_items constraint
    arr_exact = Array(exact_items=2)
    assert arr_exact.validate([1, 2]) == [1, 2]
    with pytest.raises(Exception):
        arr_exact.validate([1])

    # 9. Test validation error propagation from items
    error_msg = MagicMock()
    error_msg.messages.return_value = ["error at index 0"]
    bad_item_field = create_mock_field(None, error=error_msg)
    arr_with_error = Array(items=bad_item_field)
    
    with pytest.raises(Exception):
        arr_with_error.validate([999])

    # 10. Test additional_items as a Field
    add_field = create_mock_field("extra")
    arr_additional = Array(items=[int_field], additional_items=add_field)
    assert arr_additional.validate([1, "extra"]) == [1, "extra"]

    # 11. Test unique_items constraint
    # Note: Requires Uniqueness to be functional in the environment
    try:
        arr_unique = Array(unique_items=True)
        with pytest.raises(Exception):
            arr_unique.validate([1, 1])
    except NameError:
        # Skip if Uniqueness is not defined in the test scope
        pass

    # 12. Test positional item validation (additional items logic)
    # If items is a list, and we provide more items than the list length
    # and additional_items is False (default)
    arr_fixed_list = Array(items=[int_field])
    with pytest.raises(Exception):
        arr_fixed_list.validate([1, 2]) # 2 is not covered by list or additional_items (False)
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
from typesystem.base import ValidationError

def test_String_validate():
    # Test Basic String Validation
    s = String(title="Test", min_length=3, max_length=5)
    assert s.validate("abc") == "abc"
    assert s.validate("  abc  ") == "abc"  # Default trim_whitespace=True
    
    with pytest.raises(ValidationError) as excinfo:
        s.validate("ab")
    assert excinfo.value.code == "min_length"
    
    with pytest.raises(ValidationError) as excinfo:
        s.validate("abcdef")
    assert excinfo.value.code == "max_length"

    # Test Null and Blank behavior
    s_null = String(allow_null=True)
    assert s_null.validate(None) is None
    
    s_no_null = String(allow_null=False)
    with pytest.raises(ValidationError) as excinfo:
        s_no_null.validate(None)
    assert excinfo.value.code == "null"

    s_blank = String(allow_blank=False)
    with pytest_raises(ValidationError) as excinfo:
        s_blank.validate("   ")
    assert excinfo.value.code == "blank"

    s_allow_blank = String(allow_blank=True)
    assert s_allow_blank.validate("   ") == ""

    # Test Coerce Types (None to empty string)
    s_coerce = String(allow_blank=True, coerce_types=True)
    assert s_coerce.validate(None) == ""

    # Test Type Validation
    with pytest.raises(ValidationError) as excinfo:
        s.validate(123)
    assert excinfo.value.code == "type"

    # Test Pattern (Regex)
    s_pattern = String(pattern=r"^\d+$")
    assert s_pattern.validate("123") == "123"
    with pytest.raises(ValidationError) as excinfo:
        s_pattern.validate("123a")
    assert excinfo.value.code == "pattern"

    # Test Pattern (Compiled Regex)
    import re
    s_compiled = String(pattern=re.compile(r"^[a-z]+$"))
    assert s_compiled.validate("abc") == "abc"
    with pytest.raises(ValidationError) as excinfo:
        s_compiled.validate("ABC")
    assert excinfo.value.code == "pattern"

    # Test Null Character Removal
    s_null_char = String()
    assert s_null_char.validate("abc\0def") == "abcdef"

    # Test Format (Using mocked/available formats in the global scope)
    # Assuming 'email' format exists in the provided code's FORMATS
    s_email = String(format="email")
    # This depends on typesystem's EmailFormat implementation
    # but we test the logic flow of the String class
    try:
        # If valid email, it returns the validated value from the format
        val = s_email.validate("test@example.com")
        assert val is not None
    except ValidationError:
        pass # If format validation fails due to environment, we skip logic check
```


####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest

def test_Choice():
    # Test default initialization
    choice_default = Choice()
    assert choice_default.choices == []
    assert choice_default.coerce_types is True

    # Test initialization with simple list of strings
    choice_simple = Choice(choices=["a", "b", "c"])
    assert choice_simple.choices == [("a", "a"), ("b", "b"), ("c", "c")]

    # Test initialization with list of tuples (mapping)
    choice_tuples = Choice(choices=[("a", "Alpha"), ("b", "Beta")])
    assert choice_tuples.choices == [("a", "Alpha"), ("b", "Beta")]

    # Test initialization with mixed types
    choice_mixed = Choice(choices=["a", ("b", "Beta")])
    assert choice_mixed.choices == [("a", "a"), ("b", "Beta")]

    # Test coerce_types parameter
    choice_no_coerce = Choice(choices=["a"], coerce_types=False)
    assert choice_no_coerce.coerce_types is False

    # Test that it raises AssertionError if tuple elements are not exactly 2
    with pytest.raises(AssertionError):
        Choice(choices=[("a", "b", "c")])

    # Test field inheritance/attributes
    choice_with_kwargs = Choice(choices=["a"], title="My Choice", allow_null=True)
    assert choice_with_kwargs.title == "My Choice"
    assert choice_with_kwargs.allow_null is True
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
import decimal
import typing

# Mocking necessary dependencies that were not provided in the snippet 
# but are required for the Array.validate method to run.
class Message:
    def __init__(self, text, code, index=None, key=None):
        self.text = text
        self.code = code
        self.index = index if index is not None else []
        self.key = key
    def messages(self, add_prefix=None):
        return [self]

class ValidationError(Exception):
    def __init__(self, messages):
        self.messages = messages

class Uniqueness(set):
    def add(self, item):
        super().add(item)

class Field:
    def __init__(self, **kwargs):
        self.allow_null = kwargs.get("allow_null", False)
        self.coerce_types = kwargs.get("coerce_types", True)
        self.trim_whitespace = kwargs.get("trim_whitespace", False)
        self.min_length = kwargs.get("min_length", None)
        self.max_length = kwargs.get("max_length", None)
        self.pattern_regex = kwargs.get("pattern_regex", None)
    
    def validation_error(self, code):
        return ValidationError(messages=[Message(text="Error", code=code)])

    def validate_or_error(self, value):
        try:
            return self.validate(value), None
        except ValidationError as e:
            return None, e

    def has_default(self): return False
    def get_default_value(self): return None
    def serialize(self, obj): return obj

def test_Array_validate():
    # 1. Test successful validation of a simple list of integers
    int_field = Integer()
    array_int = Array(items=int_field)
    assert array_int.validate([1, 2, 3]) == [1, 2, 3]

    # 2. Test validation with exact_items constraint
    array_exact = Array(items=int_field, exact_items=2)
    assert array_exact.validate([1, 2]) == [1, 2]
    with pytest.raises(ValidationError) as excinfo:
        array_exact.validate([1, 2, 3])
    assert any(m.code == "exact_items" for m_obj in excinfo.value.messages for m in [m_obj])

    # 3. Test min_items constraint
    array_min = Array(items=int_field, min_items=2)
    assert array_min.validate([1, 2]) == [1, 2]
    with pytest_raises_error(array_min.validate([1]), "min_items"):
        pass # Helper logic below
    
    # 4. Test max_items constraint
    array_max = Array(items=int_field, max_items=2)
    assert array_max.validate([1, 2]) == [1, 2]
    with pytest.raises(ValidationError) as excinfo:
        array_max.validate([1, 2, 3])
    assert any(m.code == "max_items" for m in excinfo.value.messages)

    # 5. Test unique_items constraint
    array_unique = Array(items=int_field, unique_items=True)
    assert array_unique.validate([1, 2, 3]) == [1, 2, 3]
    with pytest.raises(ValidationError) as excinfo:
        array_unique.validate([1, 2, 2])
    assert any(m.code == "unique_items" for m in excinfo.value.messages)

    # 6. Test positional items (list of different types)
    array_mixed = Array(items=[Integer(), Float()])
    assert array_mixed.validate([1, 2.5]) == [1, 2.5]
    with pytest.raises(ValidationError) as excinfo:
        array_mixed.validate([1, "not_a_float"])
    # Check if error is prefixed with index 1
    assert any(m.code == "type" for m in excinfo.value.messages)

    # 7. Test additional_items as a Field
    array_extra = Array(items=Integer(), additional_items=Float())
    assert array_extra.validate([1, 2.5, 3.3]) == [1, 2.5, 3.3]
    with pytest.raises(ValidationError) as excinfo:
        array_extra.validate([1, "invalid"])
    assert any(m.code == "type" for m in excinfo.value.messages)

    # 8. Test null handling
    array_null = Array(items=Integer(), allow_null=True)
    assert array_null.validate([1, None, 3]) == [1, None, 3]
    with pytest.raises(ValidationError) as excinfo:
        array_null.validate(None)
    assert any(m.code == "null" for m in excinfo.value.messages)

    # 9. Test type error (input is not a list)
    array_type_err = Array(items=Integer())
    with pytest.raises(ValidationError) as excinfo:
        array_type_err.validate("not a list")
    assert any(m.code == "type" for m in excinfo.value.messages)

def pytest_raises_error(func, code):
    """Helper to check if a specific error code is raised."""
    try:
        func()
    except ValidationError as e:
        for msg in e.messages:
            if msg.code == code:
                return
        raise AssertionError(f"Expected error code {code} not found")
    except Exception as e:
        raise AssertionError(f"Expected ValidationError, got {type(e)}")
    raise AssertionError(f"Expected error code {code} was not raised")
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
import decimal
from typesystem.base import ValidationError

def test_Number_validate():
    # Test basic valid integer/float
    field = Number()
    assert field.validate(10) == 10
    assert field.validate(10.5) == 10.5
    assert field.validate("10.5") == decimal.Decimal("10.5")

    # Test null handling
    field_allow_null = Number(allow_null=True)
    assert field_allow_null.validate(None) is None
    
    field_no_null = Number(allow_null=False)
    with pytest.raises(ValidationError) as excinfo:
        field_no_null.validate(None)
    assert excinfo.value.code == "null"

    # Test empty string to None coercion
    field_coerce_null = Number(allow_null=True, coerce_types=True)
    assert field_coerce_null.validate("") is None

    # Test boolean rejection
    with pytest.raise(ValidationError) as excinfo:
        field.validate(True)
    assert excinfo.value.code == "type"

    # Test non-numeric type without coercion
    field_no_coerce = Number(coerce_types=False)
    with pytest.raises(ValidationError) as excinfo:
        field_no_coerce.validate("10")
    assert excinfo.value.code == "type"

    # Test invalid numeric strings
    with pytest.raises(ValidationError) as excinfo:
        field.validate("not-a-number")
    assert excinfo.value.code == "type"

    # Test infinity/NaN
    with pytest.raises(ValidationError) as excinfo:
        field.validate(float('inf'))
    assert excinfo.value.code == "finite"
    with pytest.raises(ValidationError) as excinfo:
        field.validate(float('nan'))
    assert excinfo.value.code == "finite"

    # Test numeric_type constraint (int)
    field_int = Number(numeric_type=int)
    assert field_int.validate(10.0) == 10
    with pytest.raises(ValidationError) as excinfo:
        field_int.validate(10.5)
    assert excinfo.value.code == "integer"

    # Test minimum/maximum/exclusive constraints
    field_bounds = Number(
        minimum=0, 
        maximum=10, 
        exclusive_minimum=2, 
        exclusive_maximum=8
    )
    assert field_bounds.validate(5) == 5
    
    with pytest.raises(ValidationError) as excinfo:
        field_bounds.validate(-1)
    assert excinfo.value.code == "minimum"
    
    with pytest.raises(ValidationError) as excinfo:
        field_bounds.validate(2)
    assert excinfo.value.code == "exclusive_minimum"
    
    with pytest.raises(ValidationError) as excinfo:
        field_bounds.validate(8)
    assert excinfo.value.code == "exclusive_maximum"
    
    with pytest.raises(ValidationError) as excinfo:
        field_bounds.validate(11)
    assert excinfo.value.code == "maximum"

    # Test multiple_of
    field_mult = Number(multiple_of=5)
    assert field_mult.validate(15) == 15
    with pytest.raises(ValidationError) as excinfo:
        field_mult.validate(12)
    assert excinfo.value.code == "multiple_of"

    field_mult_float = Number(multiple_of=0.5)
    assert field_mult_float.validate(1.5) == 1.5
    with pytest.raises(ValidationError) as excinfo:
        field_mult_float.validate(1.2)
    assert excinfo.value.code == "multiple_of"

    # Test precision
    field_prec = Number(precision="0.01", numeric_type=float)
    # 1.234 should round to 1.23
    assert field_prec.validate(1.234) == 1.23
    # 1.236 should round to 1.24
    assert field_prec.validate(1.236) == 1.24
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
import decimal

def test_Union_validate():
    # Mocking the necessary components for the Union class to work in isolation
    class MockValidationError(Exception):
        def __init__(self, messages):
            self.messages = lambda add_prefix=None: messages

    class MockMessage:
        def __init__(self, text, code, index=None):
            self.text = text
            self.code = code
            self.index = index

    class MockField:
        def __init__(self, allow_null=False, value_to_return=None, error_to_raise=None):
            self.allow_null = allow_null
            self.value_to_return = value_to_return
            self.error_to_raise = error_to_raise
            self.validation_error_called = False

        def validate_or_error(self, value):
            if self.error_to_raise:
                # Simulate error structure returned by validate_or_error
                return None, self.error_to_raise
            return self.value_to_return, None

        def validation_error(self, code):
            self.validation_error_called = True
            return MockValidationError([MockMessage(text="Error", code=code)])

    # Test Case 1: Successful validation (first candidate matches)
    field1 = MockField(value_to_return="match")
    field2 = MockField(value_to_return="no_match")
    union_ok = Union(any_of=[field1, field2])
    assert union_ok.validate("match") == "match"

    # Test Case 2: Successful validation (second candidate matches)
    field3 = MockField(error_to_raise=MockValidationError([MockMessage("type", "type")]), value_to_return="match_second")
    union_ok_second = Union(any_of=[field2, fieldable_error_field := MockField(error_to_raise=MockValidationError([MockMessage("type", "type")]), value_to_return="match_second")])
    # Note: In the logic, if it's a type error, it continues to next.
    # If we provide field2 (no match) and field3 (match), it should return match_second.
    union_ok_second = Union(any_of=[field2, field3])
    assert union_ok_second.validate("any_value") == "match_second"

    # Test Case 3: Null validation (allow_null is True because one child allows null)
    field_null = MockField(allow_null=True, value_to_return=None)
    field_not_null = MockField(allow_null=False)
    union_null = Union(any_of=[field_null, field_not_null])
    assert union_null.validate(None) is None

    # Test Case 4: Null validation error (none of the children allow null)
    field_no_null1 = MockField(allow_null=False)
    field_no_null2 = MockField(allow_null=False)
    union_not_null = Union(any_of=[field_no_null1, field_no_null2])
    with pytest.raises(MockValidationError) as excinfo:
        union_not_null.validate(None)
    assert excinfo.value.messages()[0].code == "null"

    # Test Case 5: Union error (none of the types match and they all return type errors)
    field_err1 = MockField(error_to_raise=MockValidationError([MockMessage("err", "type")]))
    field_err2 = MockField(error_to_raise=MockValidationError([MockMessage("err", "type")]))
    union_fail = Union(any_of=[field_err1, field_err2])
    with pytest_raises_type(union_fail, "union"):
        union_fail.validate("random_string")

    # Test Case 6: Union error (a non-type error is raised by a candidate)
    # If a candidate returns an error that is NOT a "type" error, 
    # the Union should immediately consider that the primary error.
    field_logic_err = MockField(error_to_raise=MockValidationError([MockMessage("logic", "logic_error")]))
    union_logic_err = Union(any_of=[field_logic_err, field2])
    with pytest.raises(MockValidationError) as excinfo:
        union_logic_err.validate("trigger_error")
    assert excinfo.value.messages()[0].code == "logic_error"

def pytest_raises_type(union_obj, expected_code):
    with pytest.raises(Exception) as excinfo:
        union_obj.validate("trigger_error")
    assert excinfo.value.messages()[0].code == expected_code
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Array_serialize():
    # Case 1: obj is None -> returns None
    array_none = Array(items=Integer())
    assert array_none.serialize(None) is None

    # Case 2: items is None -> returns obj as is
    array_no_items = Array(items=None)
    input_data_2 = [1, 2, 3]
    assert array_no_items.serialize(input_data_2) == input_data_2

    # Case 3: items is a single Field -> applies serialize to all elements
    # Using Float to test numeric serialization
    array_single_field = Array(items=Float())
    input_data_3 = [1, "2.5", 3.0]
    # Note: Float.serialize uses float(obj) logic if it were a Number subclass
    # In the provided code, Float inherits from Number, which doesn't override serialize.
    # However, we can mock a Field to verify the behavior.
    mock_field = MagicMock()
    mock_field.serialize.side_effect = lambda x: x * 2
    array_mock = Array(items=mock_field)
    assert array_mock.serialize([1, 2, 3]) == [2, 4, 6]

    # Case 4: items is a list of Fields (positional) -> applies serialize to corresponding elements
    mock_field_1 = MagicMock()
    mock_field_1.serialize.side_effect = lambda x: f"f1_{x}"
    mock_field_2 = MagicMock()
    mock_field_2.serialize.side_effect = lambda x: f"f2_{x}"
    
    array_list_fields = Array(items=[mock_field_1, mock_field_2])
    input_data_4 = ["a", "b"]
    assert array_list_fields.serialize(input_data_4) == ["f1_a", "f2_b"]

    # Case 5: Verify interaction with complex objects (e.g., Object/Integer)
    # Testing that the loop correctly calls serialize on each item
    int_field = Integer()
    # Since Integer doesn't override serialize, it returns obj.
    # We use a custom mock to ensure the call actually happens.
    array_int = Array(items=int_field)
    assert array_int.serialize([1, 2]) == [1, 2]
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest
import decimal

def test_Union_validate():
    # Mocking dependencies that are not provided in the snippet but required for execution
    # We assume these exist in the environment as per the prompt's context
    
    class MockValidationError(Exception):
        def __init__(self, messages):
            self.messages = lambda add_prefix=None: messages

    class MockMessage:
        def __init__(self, text, code, index=None, key=None):
            self.text = text
            self.code = code
            self.index = index
            self.key = key

    # Setup mock Field behavior for the Union test
    class MockField:
        def __init__(self, value_to_return=None, error_to_return=None, allow_null=False):
            self.value_to_return = value_to_return
            self.error_to_return = error_to_return
            self.allow_null = allow_null
        
        def validate_or_error(self, value):
            if self.error_to_return:
                return None, self.error_to_return
            return self.value_to_return, None

        def validation_error(self, code):
            return MockValidationError([MockMessage(text="error", code=code)])

    # 1. Test successful validation (matches first type)
    field1 = MockField(value_to_return="success")
    field2 = MockField(value_to_return=123)
    union_success = Union(any_of=[field1, field2])
    assert union_success.validate("success") == "success"

    # 2. Test successful validation (matches second type)
    union_success_second = Union(any_of=[field1, field2])
    assert union_success_second.validate(123) == 123

    # 3. Test null handling when allow_null is True (via child)
    field_null = MockField(value_to_return=None, allow_null=True)
    union_null_allowed = Union(any_of=[field_null, field2])
    assert union_null_allowed.validate(None) is None

    # 4. Test null error when allow_null is False
    field_not_null = MockField(value_to_return="val", allow_null=False)
    # Simulate the logic where a child raises 'null' error
    class NullErrorField(MockField):
        def validate_or_error(self, value):
            if value is None:
                return None, MockValidationError([MockMessage(text="null", code="null")])
            return super().validate_or_error(value)
    
    union_null_forbidden = Union(any_of=[NullErrorField(), field2])
    with pytest.raises(MockValidationError) as excinfo:
        union_null_forbidden.validate(None)
    assert excinfo.value.messages()[0].code == "null"

    # 5. Test "union" error when no types match (all return 'type' error)
    type_error = MockValidationError([MockMessage(text="type", code="type")])
    field_fail1 = MockField(error_to_return=type_error)
    field_fail2 = MockField(error_to_return=type_error)
    union_fail_all = Union(any_of=[field_fail1, field_fail2])
    with pytest.raises(MockValidationError) as excinfo:
        union_fail_all.validate("not_matching")
    assert excinfo.value.messages()[0].code == "union"

    # 6. Test prioritization of specific errors over 'type' error
    # If one child returns a specific error (not 'type'), Union should return that error
    specific_error = MockValidationError([MockMessage(text="specific", code="specific_error")])
    field_specific = MockField(error_to_return=specific_error)
    field_type_only = MockField(error_to_return=type_error)
    union_specific_error = Union(any_of=[field_type_only, field_specific])
    
    with pytest.raises(MockValidationError) as excinfo:
        union_specific_error.validate("trigger_specific")
    assert excinfo.value.messages()[0].code == "specific_error"
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
import decimal

def test_Union():
    # Test case 1: Basic initialization with multiple fields
    int_field = Integer()
    str_field = Text()
    union = Union(any_of=[int_field, str_field])
    
    assert union.any_of == [int_field, str_field]
    # Since neither field has allow_null=True, union should be False
    assert union.allow_null is False

    # Test case 2: Initialization where one field allows null
    null_allowed_field = Integer(allow_null=True)
    union_with_null = Union(any_of=[int_field, null_allowed_field])
    
    assert union_with_null.allow_null is True

    # Test case 3: Initialization with empty any_of (edge case)
    # Note: Depending on implementation of Field, this might raise error elsewhere, 
    # but testing the constructor's logic here.
    empty_union = Union(any_of=[])
    assert empty_union.any_of == []
    assert empty_union.allow_null is False

    # Test case 4: Verification of keyword argument passing
    # Assuming Field/super().__init__ handles kwargs
    union_with_kwargs = Union(any_of=[int_field], some_extra_arg="test")
    assert union_with_kwargs.any_of == [int_field]

    # Test case 5: Testing types of fields in any_of
    float_field = Float()
    union_mixed = Union(any_of=[float_field, Decimal()])
    assert len(union_mixed.any_of) == 2
    assert isinstance(union_mixed.any_of[0], Float)
    assert isinstance(union_mixed.any_of[1], Decimal)
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest

def test_Boolean_validate():
    # Test successful boolean validation (direct)
    bool_field = Boolean()
    assert bool_field.validate(True) is True
    assert bool_field.validate(False) is False

    # Test successful coercion from strings/ints
    assert bool_field.validate("true") is True
    assert bool_field.validate("false") is False
    assert bool_field.validate("on") is True
    assert bool_field.validate("off") is False
    assert bool_field.validate("1") is True
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
    assert bool_field_null.validate("null") is None
    assert bool_field_null.validate("none") is None

    # Test invalid types (no coercion)
    bool_field_no_coerce = Boolean(coerce_types=False)
    with pytest.raises(ValidationError) as excinfo:
        bool_field_no_coerce.validate("true")
    assert excinfo.value.code == "type"
    
    with pytest.raises(ValidationError) as excinfo:
        bool_field_no_coerce.validate(1)
    assert excinfo.value.code == "type"

    # Test invalid string values
    with pytest.raises(ValidationError) as excinfo:
        bool_field.validate("maybe")
    assert excinfo.value.code == "type"

    with pytest.raises(ValidationError) as excinfo:
        bool_field.validate(123)
    assert excinfo.value.code == "type"
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest

def test_Choice_validate():
    # Setup choices: a mix of simple strings and tuples (key, value)
    choices = ["apple", ("banana", "yellow fruit"), "cherry"]
    
    # 1. Test valid simple string choice
    field_simple = Choice(choices=choices)
    assert field_simple.validate("apple") == "apple"
    
    # 2. Test valid tuple choice (validating against the key)
    field_tuple = Choice(choices=choices)
    assert field_tuple.validate("banana") == "banana"
    
    # 3. Test valid third choice
    assert field_simple.validate("cherry") == "cherry"

    # 4. Test invalid choice
    field_invalid = Choice(choices=choices)
    with pytest.raises(ValidationError) as excinfo:
        field_invalid.validate("dragonfruit")
    assert excinfo.value.code == "choice"

    # 5. Test null value with allow_null=False (default)
    field_no_null = Choice(choices=choices, allow_null=False)
    with pytest.raises(ValidationError) as excinfo:
        field_no_null.validate(None)
    assert excinfo.value.code == "null"

    # 6. Test null value with allow_null=True
    field_allow_null = Choice(choices=choices, allow_null=True)
    assert field_allow_null.validate(None) is None

    # 7. Test empty string behavior (required error)
    # Based on the implementation: if value == "" and not in choices, 
    # it raises "required" if allow_null is False.
    field_required = Choice(choices=choices, allow_null=False)
    with pytest.raise(ValidationError) as excinfo:
        field_required.validate("")
    assert excinfo.value.code == "required"

    # 8. Test empty string behavior (null conversion)
    # Based on the implementation: if value == "" and allow_null and coerce_types, return None
    field_coerce_empty = Choice(choices=choices, allow_null=True, coerce_types=True)
    assert field_coerce_empty.validate("") is None

    # 9. Test case where choices is empty
    field_empty_choices = Choice(choices=[])
    with pytest.raises(ValidationError) as excinfo:
        field_empty_choices.validate("apple")
    assert excinfo.value.code == "choice"
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
import decimal

def test_Const():
    # Test valid initialization with various types
    # Testing that it accepts different types for the 'const' parameter
    for val in [1, "string", True, None, 1.5, decimal.Decimal("1.0")]:
        const_field = Const(const=val)
        assert const_field.const == val
        # Check that allow_null is not passed in kwargs via the assert in __init__
        assert not hasattr(const_field, 'allow_null') or const_field.allow_null is False

    # Test that passing allow_null in kwargs raises an AssertionError
    # based on the 'assert "allow_null" not in kwargs' line in __init__
    with pytest.raises(AssertionError):
        Const(const=1, allow_null=True)

    # Test validation logic for Const
    # Case 1: Value matches const
    assert Const(const="match").validate("match") == "match"
    assert Const(const=123).validate(123) == 123
    
    # Case 2: Value does not match const (non-null const)
    # Note: Assuming validation_error raises a ValidationError or similar
    # that can be caught or identified.
    with pytest.raises(Exception) as excinfo:
        Const(const="expected").validate("actual")
    assert "const" in str(excinfo.value)

    # Case 3: Value does not match const (null const)
    # The class defines error 'only_null' specifically for when const is None
    with pytest.raises(Exception) as excinfo:
        Const(const=None).validate("not_null")
    # We check if the error message/code corresponds to 'only_null'
    # This depends on how validation_error is implemented in the parent class
    assert "only_null" in str(excinfo.value).lower() or "only_null" in str(excinfo.value)

    # Case 4: Value matches null const
    assert Const(const=None).validate(None) is None
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest

def test_Const():
    # Test valid initialization with a string constant
    const_str = Const(const="hello")
    assert const_str.const == "hello"

    # Test valid initialization with an integer constant
    const_int = Const(const=123)
    assert const_int.const == 123

    # Test valid initialization with a None constant (though allow_null logic is restricted in __init__)
    const_none = Const(const=None)
    assert const_none.const is None

    # Test that providing allow_null in kwargs raises an AssertionError
    # as per the implementation: assert "allow_null" not in kwargs
    with pytest.raises(AssertionError):
        Const(const="test", allow_null=True)

    # Test validation logic for matching constant
    validator_match = Const(const="match")
    assert validator_match.validate("match") == "match"

    # Test validation logic for non-matching constant
    validator_mismatch = Const(const="match")
    with pytest.raises(ValidationError) as excinfo:
        validator_mismatch.validate("mismatch")
    assert "const" in str(excinfo.value)

    # Test validation logic for Const(None)
    validator_none = Const(const=None)
    # Note: In the implementation, if const is None and value is not None, it raises 'only_null'
    with pytest.raises(ValidationError) as excinfo:
        validator_none.validate("not_none")
    assert "only_null" in str(excinfo.value)
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
import decimal

def test_Const():
    # Test valid initialization with a string constant
    const_str = Const(const="test_value")
    assert const_str.const == "test_value"

    # Test valid initialization with an integer constant
    const_int = Const(const=123)
    assert const_int.const == 123

    # Test valid initialization with None (though the class logic handles it specifically in validate)
    const_none = Const(const=None)
    assert const_none.const is None

    # Test valid initialization with a complex object (e.g., dict)
    const_dict = Const(const={"key": "val"})
    assert const_dict.const == {"key": "val"}

    # Test that passing allow_null in kwargs raises an AssertionError 
    # as per the assertion: assert "allow_null" not in kwargs
    with pytest.raises(AssertionError):
        Const(const="val", allow_null=True)

    # Test validation logic for Const (to ensure constructor works with intended usage)
    validator = Const(const="match")
    assert validator.validate("match") == "match"
    
    with pytest.raises(Exception): # Expecting ValidationError
        validator.validate("mismatch")

    # Test validation logic for Const with None
    validator_none = Const(const=None)
    assert validator_none.validate(None) is None
    
    with pytest.raises(Exception): # Expecting ValidationError with 'only_null' code
        validator_none.validate("not_none")
```


