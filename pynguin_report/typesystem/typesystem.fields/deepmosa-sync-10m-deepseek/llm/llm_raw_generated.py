####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_validate_null_with_allow_null():
    field = String(allow_null=True)
    result = field.validate(None)
    assert result is None

def test_validate_null_without_allow_null():
    field = String(allow_null=False)
    try:
        field.validate(None)
        assert False
    except Exception as e:
        assert str(e) == "May not be null."

def test_validate_null_with_allow_blank_and_coerce_types():
    field = String(allow_blank=True, coerce_types=True)
    result = field.validate(None)
    assert result == ""

def test_validate_non_string_type():
    field = String()
    try:
        field.validate(123)
        assert False
    except Exception as e:
        assert str(e) == "Must be a string."

def test_validate_string_with_null_character():
    field = String()
    result = field.validate("hello\0world")
    assert result == "helloworld"

def test_validate_trim_whitespace():
    field = String(trim_whitespace=True)
    result = field.validate("  hello  ")
    assert result == "hello"

def test_validate_no_trim_whitespace():
    field = String(trim_whitespace=False)
    result = field.validate("  hello  ")
    assert result == "  hello  "

def test_validate_blank_without_allow_blank():
    field = String(allow_blank=False)
    try:
        field.validate("")
        assert False
    except Exception as e:
        assert str(e) == "Must not be blank."

def test_validate_blank_with_allow_blank():
    field = String(allow_blank=True)
    result = field.validate("")
    assert result == ""

def test_validate_blank_with_allow_null_and_coerce_types():
    field = String(allow_null=True, coerce_types=True, trim_whitespace=True)
    result = field.validate("   ")
    assert result is None

def test_validate_min_length_valid():
    field = String(min_length=3)
    result = field.validate("abc")
    assert result == "abc"

def test_validate_min_length_invalid():
    field = String(min_length=3)
    try:
        field.validate("ab")
        assert False
    except Exception as e:
        assert str(e) == "Must have at least 3 characters."

def test_validate_max_length_valid():
    field = String(max_length=5)
    result = field.validate("abcde")
    assert result == "abcde"

def test_validate_max_length_invalid():
    field = String(max_length=5)
    try:
        field.validate("abcdef")
        assert False
    except Exception as e:
        assert str(e) == "Must have no more than 5 characters."

def test_validate_pattern_valid():
    field = String(pattern="^[a-z]+$")
    result = field.validate("hello")
    assert result == "hello"

def test_validate_pattern_invalid():
    field = String(pattern="^[a-z]+$")
    try:
        field.validate("hello123")
        assert False
    except Exception as e:
        assert str(e) == "Must match the pattern /^[a-z]+$/."

def test_validate_format_valid():
    field = String(format="email")
    result = field.validate("test@example.com")
    assert result == "test@example.com"

def test_validate_format_invalid():
    field = String(format="email")
    try:
        field.validate("invalid-email")
        assert False
    except Exception as e:
        assert "Must be a valid email." in str(e)

def test_validate_native_type_with_format():
    field = String(format="email")
    result = field.validate("test@example.com")
    assert result == "test@example.com"


# LLM-generated content at query #2
#--------------------------

def test_validate_null_with_allow_null():
    field = Object(allow_null=True)
    result = field.validate(None)
    assert result is None

def test_validate_null_without_allow_null():
    field = Object(allow_null=False)
    try:
        field.validate(None)
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "null"

def test_validate_non_dict():
    field = Object()
    try:
        field.validate("not a dict")
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "type"

def test_validate_invalid_key_type():
    field = Object()
    try:
        field.validate({1: "value"})
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "invalid_key"

def test_validate_property_names_invalid():
    property_names_field = Field(allow_null=False)
    field = Object(property_names=property_names_field)
    try:
        field.validate({"": "value"})
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "invalid_property"

def test_validate_min_properties_empty():
    field = Object(min_properties=1)
    try:
        field.validate({})
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "empty"

def test_validate_min_properties():
    field = Object(min_properties=2)
    try:
        field.validate({"a": 1})
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "min_properties"

def test_validate_max_properties():
    field = Object(max_properties=1)
    try:
        field.validate({"a": 1, "b": 2})
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "max_properties"

def test_validate_required_missing():
    field = Object(required=["required_key"])
    try:
        field.validate({})
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "required"

def test_validate_properties_with_default():
    child_field = Field(default="default_value")
    field = Object(properties={"key": child_field})
    result = field.validate({})
    assert result == {"key": "default_value"}

def test_validate_properties_valid():
    child_field = Field()
    field = Object(properties={"key": child_field})
    result = field.validate({"key": "value"})
    assert result == {"key": "value"}

def test_validate_properties_invalid():
    child_field = Field(allow_null=False)
    field = Object(properties={"key": child_field})
    try:
        field.validate({"key": None})
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "null"

def test_validate_pattern_properties():
    child_field = Field()
    field = Object(pattern_properties={"^a.*": child_field})
    result = field.validate({"abc": "value", "def": "ignored"})
    assert result == {"abc": "value"}

def test_validate_additional_properties_true():
    field = Object(additional_properties=True)
    result = field.validate({"extra": "value"})
    assert result == {"extra": "value"}

def test_validate_additional_properties_false():
    field = Object(additional_properties=False)
    try:
        field.validate({"extra": "value"})
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "invalid_property"

def test_validate_additional_properties_field():
    additional_field = Field()
    field = Object(additional_properties=additional_field)
    result = field.validate({"extra": "value"})
    assert result == {"extra": "value"}

def test_validate_additional_properties_field_invalid():
    additional_field = Field(allow_null=False)
    field = Object(additional_properties=additional_field)
    try:
        field.validate({"extra": None})
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "null"

def test_validate_combined_errors():
    child_field = Field(allow_null=False)
    field = Object(properties={"key": child_field}, required=["required_key"])
    try:
        field.validate({"key": None})
    except ValidationError as e:
        assert len(e.messages()) == 2
        codes = {msg.code for msg in e.messages()}
        assert codes == {"null", "required"}


# LLM-generated content at query #3
#--------------------------

def test_validate_null_allowed():
    field = Number(allow_null=True)
    result = field.validate(None)
    assert result is None

def test_validate_null_not_allowed():
    field = Number(allow_null=False)
    try:
        field.validate(None)
        assert False
    except Exception as e:
        assert str(e) == "May not be null."

def test_validate_empty_string_coerced_to_null():
    field = Number(allow_null=True, coerce_types=True)
    result = field.validate("")
    assert result is None

def test_validate_boolean_raises_type_error():
    field = Number()
    try:
        field.validate(True)
        assert False
    except Exception as e:
        assert str(e) == "Must be a number."

def test_validate_string_coerced_to_number():
    field = Number(coerce_types=True)
    result = field.validate("123")
    assert result == 123

def test_validate_string_not_coerced_raises_error():
    field = Number(coerce_types=False)
    try:
        field.validate("123")
        assert False
    except Exception as e:
        assert str(e) == "Must be a number."

def test_validate_float_for_integer_field_raises_error():
    field = Number(numeric_type=int)
    try:
        field.validate(123.5)
        assert False
    except Exception as e:
        assert str(e) == "Must be an integer."

def test_validate_infinity_raises_finite_error():
    field = Number()
    try:
        field.validate(float('inf'))
        assert False
    except Exception as e:
        assert str(e) == "Must be finite."

def test_validate_nan_raises_finite_error():
    field = Number()
    try:
        field.validate(float('nan'))
        assert False
    except Exception as e:
        assert str(e) == "Must be finite."

def test_validate_minimum_violation():
    field = Number(minimum=10)
    try:
        field.validate(5)
        assert False
    except Exception as e:
        assert str(e) == "Must be greater than or equal to 10."

def test_validate_exclusive_minimum_violation():
    field = Number(exclusive_minimum=10)
    try:
        field.validate(10)
        assert False
    except Exception as e:
        assert str(e) == "Must be greater than 10."

def test_validate_maximum_violation():
    field = Number(maximum=10)
    try:
        field.validate(15)
        assert False
    except Exception as e:
        assert str(e) == "Must be less than or equal to 10."

def test_validate_exclusive_maximum_violation():
    field = Number(exclusive_maximum=10)
    try:
        field.validate(10)
        assert False
    except Exception as e:
        assert str(e) == "Must be less than 10."

def test_validate_multiple_of_int_violation():
    field = Number(multiple_of=5)
    try:
        field.validate(12)
        assert False
    except Exception as e:
        assert str(e) == "Must be a multiple of 5."

def test_validate_multiple_of_float_violation():
    field = Number(multiple_of=0.5)
    try:
        field.validate(1.2)
        assert False
    except Exception as e:
        assert str(e) == "Must be a multiple of 0.5."

def test_validate_precision_rounding():
    field = Number(precision='0.00')
    result = field.validate(1.234)
    assert result == 1.23

def test_validate_valid_number_with_minimum():
    field = Number(minimum=5)
    result = field.validate(10)
    assert result == 10

def test_validate_valid_number_with_maximum():
    field = Number(maximum=20)
    result = field.validate(15)
    assert result == 15

def test_validate_valid_number_with_multiple_of():
    field = Number(multiple_of=3)
    result = field.validate(9)
    assert result == 9

def test_validate_valid_integer_field():
    field = Number(numeric_type=int)
    result = field.validate(42)
    assert result == 42

def test_validate_valid_float_field():
    field = Number(numeric_type=float)
    result = field.validate(42.5)
    assert result == 42.5


# LLM-generated content at query #4
#--------------------------

def test_validate_with_null_and_allow_null():
    from .fields import Integer, String, Union
    integer_field = Integer(allow_null=True)
    union_field = Union(any_of=[integer_field])
    result = union_field.validate(None)
    assert result is None

def test_validate_with_null_and_not_allow_null():
    from .fields import Integer, String, Union
    integer_field = Integer(allow_null=False)
    union_field = Union(any_of=[integer_field])
    try:
        union_field.validate(None)
        assert False
    except Exception as e:
        assert e.messages()[0].code == "null"

def test_validate_with_matching_first_child():
    from .fields import Integer, String, Union
    integer_field = Integer()
    string_field = String()
    union_field = Union(any_of=[integer_field, string_field])
    result = union_field.validate(123)
    assert result == 123

def test_validate_with_matching_second_child():
    from .fields import Integer, String, Union
    integer_field = Integer()
    string_field = String()
    union_field = Union(any_of=[integer_field, string_field])
    result = union_field.validate("hello")
    assert result == "hello"

def test_validate_with_no_matching_child():
    from .fields import Integer, String, Union
    integer_field = Integer()
    string_field = String()
    union_field = Union(any_of=[integer_field, string_field])
    try:
        union_field.validate(3.14)
        assert False
    except Exception as e:
        assert e.messages()[0].code == "union"

def test_validate_with_child_type_error_and_other_errors():
    from .fields import Integer, String, Union
    class CustomField:
        def __init__(self):
            self.allow_null = False
        def validate_or_error(self, value):
            from .exceptions import ValidationError
            if isinstance(value, int):
                raise ValidationError([{"code": "custom", "index": []}])
            return None, ValidationError([{"code": "type", "index": []}])
    custom_field = CustomField()
    union_field = Union(any_of=[custom_field])
    try:
        union_field.validate(123)
        assert False
    except Exception as e:
        assert e.messages()[0].code == "custom"

def test_validate_with_multiple_candidate_errors():
    from .fields import Integer, String, Union
    class CustomField1:
        def __init__(self):
            self.allow_null = False
        def validate_or_error(self, value):
            from .exceptions import ValidationError
            if isinstance(value, int):
                raise ValidationError([{"code": "custom1", "index": []}])
            return None, ValidationError([{"code": "type", "index": []}])
    class CustomField2:
        def __init__(self):
            self.allow_null = False
        def validate_or_error(self, value):
            from .exceptions import ValidationError
            if isinstance(value, int):
                raise ValidationError([{"code": "custom2", "index": []}])
            return None, ValidationError([{"code": "type", "index": []}])
    custom_field1 = CustomField1()
    custom_field2 = CustomField2()
    union_field = Union(any_of=[custom_field1, custom_field2])
    try:
        union_field.validate(123)
        assert False
    except Exception as e:
        assert e.messages()[0].code == "union"

def test_validate_with_single_candidate_error():
    from .fields import Integer, String, Union
    class CustomField1:
        def __init__(self):
            self.allow_null = False
        def validate_or_error(self, value):
            from .exceptions import ValidationError
            if isinstance(value, int):
                raise ValidationError([{"code": "custom1", "index": []}])
            return None, ValidationError([{"code": "type", "index": []}])
    class CustomField2:
        def __init__(self):
            self.allow_null = False
        def validate_or_error(self, value):
            from .exceptions import ValidationError
            return None, ValidationError([{"code": "type", "index": []}])
    custom_field1 = CustomField1()
    custom_field2 = CustomField2()
    union_field = Union(any_of=[custom_field1, custom_field2])
    try:
        union_field.validate(123)
        assert False
    except Exception as e:
        assert e.messages()[0].code == "custom1"

def test_validate_with_child_allow_null_enables_union_allow_null():
    from .fields import Integer, String, Union
    integer_field = Integer(allow_null=True)
    string_field = String(allow_null=False)
    union_field = Union(any_of=[integer_field, string_field])
    result = union_field.validate(None)
    assert result is None


# LLM-generated content at query #5
#--------------------------

```python
def test_additional_properties_field_validation_success():
    from typesystem.fields import Object, String
    from typesystem.base import ValidationError
    additional_field = String(max_length=5)
    schema = Object(additional_properties=additional_field)
    value = {"extra": "hello"}
    result = schema.validate(value)
    assert result == {"extra": "hello"}


# LLM-generated content at query #6
#--------------------------

def test_array_constructor_with_defaults():
    field = Array()
    assert field.items is None
    assert field.additional_items is False
    assert field.min_items is None
    assert field.max_items is None
    assert field.unique_items is False
    assert field.allow_null is False
    assert field.read_only is False
    assert field.title == ""
    assert field.description == ""

def test_array_constructor_with_single_item_field():
    item_field = Field()
    field = Array(items=item_field)
    assert field.items == item_field
    assert field.additional_items is False
    assert field.min_items is None
    assert field.max_items is None

def test_array_constructor_with_item_list():
    item_field1 = Field()
    item_field2 = Field()
    field = Array(items=[item_field1, item_field2])
    assert field.items == [item_field1, item_field2]
    assert field.additional_items is False
    assert field.min_items == 2
    assert field.max_items == 2

def test_array_constructor_with_additional_items_field():
    additional_field = Field()
    field = Array(additional_items=additional_field)
    assert field.additional_items == additional_field

def test_array_constructor_with_additional_items_false():
    field = Array(additional_items=False)
    assert field.additional_items is False

def test_array_constructor_with_min_items():
    field = Array(min_items=5)
    assert field.min_items == 5
    assert field.max_items is None

def test_array_constructor_with_max_items():
    field = Array(max_items=10)
    assert field.min_items is None
    assert field.max_items == 10

def test_array_constructor_with_exact_items():
    field = Array(exact_items=7)
    assert field.min_items == 7
    assert field.max_items == 7

def test_array_constructor_with_unique_items():
    field = Array(unique_items=True)
    assert field.unique_items is True

def test_array_constructor_with_allow_null():
    field = Array(allow_null=True)
    assert field.allow_null is True

def test_array_constructor_with_read_only():
    field = Array(read_only=True)
    assert field.read_only is True

def test_array_constructor_with_title_and_description():
    field = Array(title="Test Title", description="Test Description")
    assert field.title == "Test Title"
    assert field.description == "Test Description"

def test_array_constructor_with_item_list_and_additional_items_false():
    item_field1 = Field()
    item_field2 = Field()
    field = Array(items=[item_field1, item_field2], additional_items=False)
    assert field.items == [item_field1, item_field2]
    assert field.additional_items is False
    assert field.min_items == 2
    assert field.max_items == 2

def test_array_constructor_with_item_list_and_additional_items_field():
    item_field1 = Field()
    item_field2 = Field()
    additional_field = Field()
    field = Array(items=[item_field1, item_field2], additional_items=additional_field)
    assert field.items == [item_field1, item_field2]
    assert field.additional_items == additional_field
    assert field.min_items == 2
    assert field.max_items is None

def test_array_constructor_with_item_list_and_min_items_override():
    item_field1 = Field()
    item_field2 = Field()
    field = Array(items=[item_field1, item_field2], min_items=5)
    assert field.items == [item_field1, item_field2]
    assert field.min_items == 5
    assert field.max_items == 2

def test_array_constructor_with_item_list_and_max_items_override():
    item_field1 = Field()
    item_field2 = Field()
    field = Array(items=[item_field1, item_field2], max_items=10)
    assert field.items == [item_field1, item_field2]
    assert field.min_items == 2
    assert field.max_items == 10

def test_array_constructor_with_item_list_and_exact_items_override():
    item_field1 = Field()
    item_field2 = Field()
    field = Array(items=[item_field1, item_field2], exact_items=7)
    assert field.items == [item_field1, item_field2]
    assert field.min_items == 7
    assert field.max_items == 7

def test_array_constructor_with_default_value():
    field = Array(default=[1, 2, 3])
    assert field.get_default_value() == [1, 2, 3]


# LLM-generated content at query #7
#--------------------------

def test_predicate_at_line_17_evaluates_true_for_multiple_messages():
    from my_module import Union, Field
    class MockField(Field):
        def validate_or_error(self, value):
            class MockError:
                def messages(self):
                    return [type('obj', (object,), {'code': 'type', 'index': None})(), type('obj', (object,), {'code': 'type', 'index': None})()]
            return None, MockError()
    child = MockField()
    union = Union(any_of=[child])
    candidate_errors = []
    validated, error = child.validate_or_error("test")
    messages = error.messages()
    predicate_result = len(messages) != 1 or messages[0].code != "type" or messages[0].index
    assert predicate_result == True

def test_predicate_at_line_17_evaluates_true_for_single_message_with_non_type_code():
    from my_module import Union, Field
    class MockField(Field):
        def validate_or_error(self, value):
            class MockError:
                def messages(self):
                    return [type('obj', (object,), {'code': 'max_length', 'index': None})()]
            return None, MockError()
    child = MockField()
    union = Union(any_of=[child])
    candidate_errors = []
    validated, error = child.validate_or_error("test")
    messages = error.messages()
    predicate_result = len(messages) != 1 or messages[0].code != "type" or messages[0].index
    assert predicate_result == True

def test_predicate_at_line_17_evaluates_true_for_single_message_with_type_code_and_index():
    from my_module import Union, Field
    class MockField(Field):
        def validate_or_error(self, value):
            class MockError:
                def messages(self):
                    return [type('obj', (object,), {'code': 'type', 'index': [0]})()]
            return None, MockError()
    child = MockField()
    union = Union(any_of=[child])
    candidate_errors = []
    validated, error = child.validate_or_error("test")
    messages = error.messages()
    predicate_result = len(messages) != 1 or messages[0].code != "type" or messages[0].index
    assert predicate_result == True

def test_predicate_at_line_17_evaluates_false_for_single_message_with_type_code_and_no_index():
    from my_module import Union, Field
    class MockField(Field):
        def validate_or_error(self, value):
            class MockError:
                def messages(self):
                    return [type('obj', (object,), {'code': 'type', 'index': None})()]
            return None, MockError()
    child = MockField()
    union = Union(any_of=[child])
    candidate_errors = []
    validated, error = child.validate_or_error("test")
    messages = error.messages()
    predicate_result = len(messages) != 1 or messages[0].code != "type" or messages[0].index
    assert predicate_result == False


# LLM-generated content at query #8
#--------------------------

def test_multiple_of_int_condition_false():
    field = Number(multiple_of=3)
    result = field.validate(6)
    assert result == 6


# LLM-generated content at query #9
#--------------------------

def test_validate_boolean_true():
    field = Boolean()
    result = field.validate(True)
    assert result is True

def test_validate_boolean_false():
    field = Boolean()
    result = field.validate(False)
    assert result is False

def test_validate_allow_null_with_none():
    field = Boolean(allow_null=True)
    result = field.validate(None)
    assert result is None

def test_validate_disallow_null_with_none():
    field = Boolean(allow_null=False)
    try:
        field.validate(None)
        assert False
    except Exception as e:
        assert str(e) == "May not be null."

def test_validate_non_boolean_with_coerce_types_true():
    field = Boolean(coerce_types=True)
    result = field.validate("true")
    assert result is True

def test_validate_non_boolean_with_coerce_types_false():
    field = Boolean(coerce_types=False)
    try:
        field.validate("true")
        assert False
    except Exception as e:
        assert str(e) == "Must be a boolean."

def test_validate_coerce_values_string_true():
    field = Boolean()
    result = field.validate("true")
    assert result is True

def test_validate_coerce_values_string_false():
    field = Boolean()
    result = field.validate("false")
    assert result is False

def test_validate_coerce_values_string_on():
    field = Boolean()
    result = field.validate("on")
    assert result is True

def test_validate_coerce_values_string_off():
    field = Boolean()
    result = field.validate("off")
    assert result is False

def test_validate_coerce_values_string_1():
    field = Boolean()
    result = field.validate("1")
    assert result is True

def test_validate_coerce_values_string_0():
    field = Boolean()
    result = field.validate("0")
    assert result is False

def test_validate_coerce_values_string_empty():
    field = Boolean()
    result = field.validate("")
    assert result is False

def test_validate_coerce_values_int_1():
    field = Boolean()
    result = field.validate(1)
    assert result is True

def test_validate_coerce_values_int_0():
    field = Boolean()
    result = field.validate(0)
    assert result is False

def test_validate_coerce_null_values_with_allow_null_empty_string():
    field = Boolean(allow_null=True)
    result = field.validate("")
    assert result is None

def test_validate_coerce_null_values_with_allow_null_string_null():
    field = Boolean(allow_null=True)
    result = field.validate("null")
    assert result is None

def test_validate_coerce_null_values_with_allow_null_string_none():
    field = Boolean(allow_null=True)
    result = field.validate("none")
    assert result is None

def test_validate_coerce_null_values_without_allow_null_empty_string():
    field = Boolean(allow_null=False)
    result = field.validate("")
    assert result is False

def test_validate_invalid_string_with_coerce_types():
    field = Boolean(coerce_types=True)
    try:
        field.validate("invalid")
        assert False
    except Exception as e:
        assert str(e) == "Must be a boolean."

def test_validate_invalid_type_with_coerce_types():
    field = Boolean(coerce_types=True)
    try:
        field.validate([])
        assert False
    except Exception as e:
        assert str(e) == "Must be a boolean."


# LLM-generated content at query #10
#--------------------------

def test_or_with_two_simple_fields():
    field1 = Field()
    field2 = Field()
    union = field1 | field2
    assert isinstance(union, Union)
    assert len(union.any_of) == 2
    assert union.any_of[0] is field1
    assert union.any_of[1] is field2

def test_or_with_field_and_union():
    field1 = Field()
    field2 = Field()
    field3 = Field()
    union1 = field1 | field2
    union2 = union1 | field3
    assert isinstance(union2, Union)
    assert len(union2.any_of) == 3
    assert union2.any_of[0] is field1
    assert union2.any_of[1] is field2
    assert union2.any_of[2] is field3

def test_or_with_union_and_field():
    field1 = Field()
    field2 = Field()
    field3 = Field()
    union1 = field1 | field2
    union2 = field3 | union1
    assert isinstance(union2, Union)
    assert len(union2.any_of) == 3
    assert union2.any_of[0] is field3
    assert union2.any_of[1] is field1
    assert union2.any_of[2] is field2

def test_or_with_two_unions():
    field1 = Field()
    field2 = Field()
    field3 = Field()
    field4 = Field()
    union1 = field1 | field2
    union2 = field3 | field4
    union3 = union1 | union2
    assert isinstance(union3, Union)
    assert len(union3.any_of) == 4
    assert union3.any_of[0] is field1
    assert union3.any_of[1] is field2
    assert union3.any_of[2] is field3
    assert union3.any_of[3] is field4

def test_or_chaining():
    field1 = Field()
    field2 = Field()
    field3 = Field()
    union = field1 | field2 | field3
    assert isinstance(union, Union)
    assert len(union.any_of) == 3
    assert union.any_of[0] is field1
    assert union.any_of[1] is field2
    assert union.any_of[2] is field3


# LLM-generated content at query #11
#--------------------------

def test_validate_allow_null():
    field = Choice(choices=[("a", "A"), ("b", "B")], allow_null=True)
    result = field.validate(None)
    assert result is None

def test_validate_null_not_allowed():
    field = Choice(choices=[("a", "A"), ("b", "B")], allow_null=False)
    try:
        field.validate(None)
        assert False
    except Exception as e:
        assert str(e) == "May not be null."

def test_validate_valid_choice():
    field = Choice(choices=[("a", "A"), ("b", "B")], allow_null=False)
    result = field.validate("a")
    assert result == "a"

def test_validate_invalid_choice():
    field = Choice(choices=[("a", "A"), ("b", "B")], allow_null=False)
    try:
        field.validate("c")
        assert False
    except Exception as e:
        assert str(e) == "Not a valid choice."

def test_validate_empty_string_null_allowed_coerce():
    field = Choice(choices=[("a", "A"), ("b", "B")], allow_null=True, coerce_types=True)
    result = field.validate("")
    assert result is None

def test_validate_empty_string_null_not_allowed():
    field = Choice(choices=[("a", "A"), ("b", "B")], allow_null=False, coerce_types=True)
    try:
        field.validate("")
        assert False
    except Exception as e:
        assert str(e) == "This field is required."

def test_validate_empty_string_null_allowed_no_coerce():
    field = Choice(choices=[("a", "A"), ("b", "B")], allow_null=True, coerce_types=False)
    try:
        field.validate("")
        assert False
    except Exception as e:
        assert str(e) == "Not a valid choice."

def test_validate_choices_with_tuples():
    field = Choice(choices=[("key1", "Display 1"), ("key2", "Display 2")], allow_null=False)
    result = field.validate("key1")
    assert result == "key1"

def test_validate_choices_with_single_strings():
    field = Choice(choices=["x", "y"], allow_null=False)
    result = field.validate("x")
    assert result == "x"

def test_validate_boolean_true_choice():
    field = Choice(choices=[(True, "Yes"), (False, "No")], allow_null=False)
    result = field.validate(True)
    assert result is True

def test_validate_boolean_false_choice():
    field = Choice(choices=[(True, "Yes"), (False, "No")], allow_null=False)
    result = field.validate(False)
    assert result is False

def test_validate_integer_choice():
    field = Choice(choices=[(1, "One"), (2, "Two")], allow_null=False)
    result = field.validate(1)
    assert result == 1

def test_validate_float_choice():
    field = Choice(choices=[(1.5, "One and half"), (2.5, "Two and half")], allow_null=False)
    result = field.validate(1.5)
    assert result == 1.5

def test_validate_list_choice():
    field = Choice(choices=[([1, 2], "List 1"), ([3, 4], "List 2")], allow_null=False)
    result = field.validate([1, 2])
    assert result == [1, 2]

def test_validate_dict_choice():
    field = Choice(choices=[({"a": 1}, "Dict A"), ({"b": 2}, "Dict B")], allow_null=False)
    result = field.validate({"a": 1})
    assert result == {"a": 1}


# LLM-generated content at query #12
#--------------------------

def test_choice_constructor_with_default_parameters():
    field = Choice()
    assert field.title == ""
    assert field.description == ""
    assert field.allow_null == False
    assert field.read_only == False
    assert field.choices == []
    assert field.coerce_types == True

def test_choice_constructor_with_custom_parameters():
    field = Choice(title="Test Title", description="Test Description", allow_null=True, read_only=True, choices=[("a", "A"), ("b", "B")], coerce_types=False)
    assert field.title == "Test Title"
    assert field.description == "Test Description"
    assert field.allow_null == True
    assert field.read_only == True
    assert field.choices == [("a", "A"), ("b", "B")]
    assert field.coerce_types == False

def test_choice_constructor_with_choices_as_strings():
    field = Choice(choices=["option1", "option2"])
    assert field.choices == [("option1", "option1"), ("option2", "option2")]

def test_choice_constructor_with_choices_as_tuples():
    field = Choice(choices=[("key1", "value1"), ("key2", "value2")])
    assert field.choices == [("key1", "value1"), ("key2", "value2")]

def test_choice_constructor_with_empty_choices():
    field = Choice(choices=[])
    assert field.choices == []

def test_choice_constructor_with_coerce_types_true():
    field = Choice(coerce_types=True)
    assert field.coerce_types == True

def test_choice_constructor_with_coerce_types_false():
    field = Choice(coerce_types=False)
    assert field.coerce_types == False

def test_choice_constructor_inherits_field_defaults():
    field = Choice()
    assert not hasattr(field, "default")

def test_choice_constructor_with_default_parameter():
    field = Choice(default="default_value")
    assert field.default == "default_value"

def test_choice_constructor_with_allow_null_and_no_default():
    field = Choice(allow_null=True)
    assert field.default == None

def test_choice_constructor_with_allow_null_and_explicit_default():
    field = Choice(allow_null=True, default="explicit_default")
    assert field.default == "explicit_default"


# LLM-generated content at query #13
#--------------------------

def test_validate_basic_object():
    field = Object()
    value = {"key": "value"}
    result = field.validate(value)
    assert result == {"key": "value"}

def test_validate_null_with_allow_null():
    field = Object(allow_null=True)
    result = field.validate(None)
    assert result is None

def test_validate_null_without_allow_null():
    field = Object()
    try:
        field.validate(None)
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "null"

def test_validate_non_dict_type():
    field = Object()
    try:
        field.validate("not a dict")
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "type"

def test_validate_invalid_key_type():
    field = Object()
    value = {123: "value"}
    try:
        field.validate(value)
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "invalid_key"

def test_validate_property_names():
    property_names_field = String(max_length=3)
    field = Object(property_names=property_names_field)
    value = {"longkey": "value"}
    try:
        field.validate(value)
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "invalid_property"

def test_validate_min_properties_empty():
    field = Object(min_properties=1)
    value = {}
    try:
        field.validate(value)
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "empty"

def test_validate_min_properties():
    field = Object(min_properties=2)
    value = {"a": 1}
    try:
        field.validate(value)
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "min_properties"

def test_validate_max_properties():
    field = Object(max_properties=1)
    value = {"a": 1, "b": 2}
    try:
        field.validate(value)
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "max_properties"

def test_validate_required():
    field = Object(required=["key"])
    value = {}
    try:
        field.validate(value)
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "required"

def test_validate_properties():
    child_field = Integer()
    field = Object(properties={"key": child_field})
    value = {"key": 123}
    result = field.validate(value)
    assert result == {"key": 123}

def test_validate_properties_with_default():
    child_field = Integer(default=456)
    field = Object(properties={"key": child_field})
    value = {}
    result = field.validate(value)
    assert result == {"key": 456}

def test_validate_properties_with_error():
    child_field = Integer()
    field = Object(properties={"key": child_field})
    value = {"key": "not an integer"}
    try:
        field.validate(value)
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "type"

def test_validate_pattern_properties():
    child_field = Integer()
    field = Object(pattern_properties={"^a": child_field})
    value = {"abc": 123, "def": 456}
    result = field.validate(value)
    assert result == {"abc": 123}

def test_validate_additional_properties_true():
    field = Object(additional_properties=True)
    value = {"key": "value"}
    result = field.validate(value)
    assert result == {"key": "value"}

def test_validate_additional_properties_false():
    field = Object(additional_properties=False)
    value = {"key": "value"}
    try:
        field.validate(value)
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "invalid_property"

def test_validate_additional_properties_field():
    child_field = Integer()
    field = Object(additional_properties=child_field)
    value = {"key": 123}
    result = field.validate(value)
    assert result == {"key": 123}

def test_validate_additional_properties_field_with_error():
    child_field = Integer()
    field = Object(additional_properties=child_field)
    value = {"key": "not an integer"}
    try:
        field.validate(value)
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "type"

def test_validate_multiple_errors():
    field = Object(required=["key1", "key2"])
    value = {}
    try:
        field.validate(value)
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 2
        codes = {msg.code for msg in e.messages()}
        assert codes == {"required"}


# LLM-generated content at query #14
#--------------------------

def test_serialize_with_none_and_allow_null():
    field = Array(allow_null=True)
    result = field.serialize(None)
    assert result is None

def test_serialize_with_none_and_not_allow_null():
    field = Array(allow_null=False)
    result = field.serialize(None)
    assert result is None

def test_serialize_with_list_and_no_items_field():
    field = Array()
    obj = [1, 2, 3]
    result = field.serialize(obj)
    assert result == [1, 2, 3]

def test_serialize_with_single_item_field():
    from unittest.mock import Mock
    mock_field = Mock()
    mock_field.serialize.side_effect = lambda x: x * 2
    field = Array(items=mock_field)
    obj = [1, 2, 3]
    result = field.serialize(obj)
    assert result == [2, 4, 6]

def test_serialize_with_list_of_item_fields():
    from unittest.mock import Mock
    mock_field1 = Mock()
    mock_field1.serialize.return_value = 'a'
    mock_field2 = Mock()
    mock_field2.serialize.return_value = 'b'
    field = Array(items=[mock_field1, mock_field2])
    obj = [1, 2]
    result = field.serialize(obj)
    assert result == ['a', 'b']

def test_serialize_with_list_of_item_fields_and_longer_obj():
    from unittest.mock import Mock
    mock_field1 = Mock()
    mock_field1.serialize.return_value = 'a'
    mock_field2 = Mock()
    mock_field2.serialize.return_value = 'b'
    field = Array(items=[mock_field1, mock_field2])
    obj = [1, 2, 3]
    result = field.serialize(obj)
    assert result == ['a', 'b', 3]

def test_serialize_with_list_of_item_fields_and_shorter_obj():
    from unittest.mock import Mock
    mock_field1 = Mock()
    mock_field1.serialize.return_value = 'a'
    mock_field2 = Mock()
    mock_field2.serialize.return_value = 'b'
    field = Array(items=[mock_field1, mock_field2])
    obj = [1]
    result = field.serialize(obj)
    assert result == ['a', 1]


# LLM-generated content at query #15
#--------------------------

def test_validate_null_allowed():
    field = Array(allow_null=True)
    result = field.validate(None)
    assert result is None

def test_validate_null_not_allowed():
    field = Array()
    try:
        field.validate(None)
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "null"

def test_validate_not_list():
    field = Array()
    try:
        field.validate("not a list")
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "type"

def test_validate_empty_with_min_items_one():
    field = Array(min_items=1)
    try:
        field.validate([])
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "empty"

def test_validate_min_items():
    field = Array(min_items=3)
    try:
        field.validate([1, 2])
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "min_items"

def test_validate_max_items():
    field = Array(max_items=2)
    try:
        field.validate([1, 2, 3])
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "max_items"

def test_validate_exact_items():
    field = Array(exact_items=2)
    try:
        field.validate([1])
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "exact_items"

def test_validate_exact_items_success():
    field = Array(exact_items=2)
    result = field.validate([1, 2])
    assert result == [1, 2]

def test_validate_with_single_item_validator():
    item_field = Field()
    item_field.validate = lambda x: x * 2
    field = Array(items=item_field)
    result = field.validate([1, 2, 3])
    assert result == [2, 4, 6]

def test_validate_with_list_item_validators():
    item_field1 = Field()
    item_field1.validate = lambda x: x + 1
    item_field2 = Field()
    item_field2.validate = lambda x: x * 2
    field = Array(items=[item_field1, item_field2])
    result = field.validate([1, 2])
    assert result == [2, 4]

def test_validate_with_list_item_validators_and_additional_items_false():
    item_field1 = Field()
    item_field1.validate = lambda x: x + 1
    item_field2 = Field()
    item_field2.validate = lambda x: x * 2
    field = Array(items=[item_field1, item_field2], additional_items=False)
    try:
        field.validate([1, 2, 3])
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "additional_items"

def test_validate_with_list_item_validators_and_additional_items_field():
    item_field1 = Field()
    item_field1.validate = lambda x: x + 1
    item_field2 = Field()
    item_field2.validate = lambda x: x * 2
    additional_field = Field()
    additional_field.validate = lambda x: x - 1
    field = Array(items=[item_field1, item_field2], additional_items=additional_field)
    result = field.validate([1, 2, 3, 4])
    assert result == [2, 4, 2, 3]

def test_validate_unique_items():
    field = Array(unique_items=True)
    try:
        field.validate([1, 2, 1])
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "unique_items"

def test_validate_unique_items_success():
    field = Array(unique_items=True)
    result = field.validate([1, 2, 3])
    assert result == [1, 2, 3]

def test_validate_with_item_validation_error():
    item_field = Field()
    item_field.validate = lambda x: (_ for _ in ()).throw(ValidationError(messages=[Message(text="Invalid", code="invalid")]))
    field = Array(items=item_field)
    try:
        field.validate([1, 2])
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 2
        assert e.messages()[0].code == "invalid"
        assert e.messages()[0].index == [0]
        assert e.messages()[1].code == "invalid"
        assert e.messages()[1].index == [1]

def test_validate_with_list_item_validators_and_validation_errors():
    item_field1 = Field()
    item_field1.validate = lambda x: (_ for _ in ()).throw(ValidationError(messages=[Message(text="Error1", code="error1")]))
    item_field2 = Field()
    item_field2.validate = lambda x: x * 2
    field = Array(items=[item_field1, item_field2])
    try:
        field.validate([1, 2])
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "error1"
        assert e.messages()[0].index == [0]

def test_validate_with_additional_items_field_validation_error():
    item_field = Field()
    item_field.validate = lambda x: x + 1
    additional_field = Field()
    additional_field.validate = lambda x: (_ for _ in ()).throw(ValidationError(messages=[Message(text="Additional error", code="additional_error")]))
    field = Array(items=[item_field], additional_items=additional_field)
    try:
        field.validate([1, 2])
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "additional_error"
        assert e.messages()[0].index == [1]

def test_validate_unique_items_with_complex_types():
    field = Array(unique_items=True)
    result = field.validate([True, False, 1, 0])
    assert result == [True, False, 1, 0]

def test_validate_unique_items_with_list_and_dict():
    field = Array(unique_items=True)
    result = field.validate([{"a": 1}, [1, 2], {"a": 1}, [1, 2]])
    assert len(result) == 2
    assert result[0] == {"a": 1}
    assert result[1] == [1, 2]


# LLM-generated content at query #16
#--------------------------

def test_assert_all_choices_have_length_two():
    choices = [("key1", "value1"), ("key2", "value2")]
    field = Choice(choices=choices)
    assert len(field.choices) == 2
    assert all(len(choice) == 2 for choice in field.choices)


# LLM-generated content at query #17
#--------------------------

```python
def test_additional_items_field_condition_false():
    from typesystem.fields import Array, Integer
    from typesystem.base import Message
    from typesystem.exceptions import ValidationError
    field = Array(items=[Integer()], additional_items=False)
    result = field.validate([1, 2])
    assert result == [1, 2]


# LLM-generated content at query #18
#--------------------------

```python
def test_pattern_properties_with_non_string_key():
    from typesystem.fields import Object, String
    import re
    field = Object(pattern_properties={"^a.*": String()})
    value = {123: "test"}
    result = field.validate(value)
    assert result == {}


# LLM-generated content at query #19
#--------------------------

def test_serialize_with_items_as_field_and_obj_not_none():
    field = Field()
    array = Array(items=field)
    obj = [1, 2, 3]
    result = array.serialize(obj)
    assert result == [field.serialize(1), field.serialize(2), field.serialize(3)]


# LLM-generated content at query #20
#--------------------------

def test_validate_null_with_allow_null():
    field = String(allow_null=True)
    result = field.validate(None)
    assert result is None

def test_validate_null_without_allow_null():
    field = String(allow_null=False)
    try:
        field.validate(None)
        assert False
    except Exception as e:
        assert str(e) == "May not be null."

def test_validate_null_with_allow_blank_and_coerce_types():
    field = String(allow_blank=True, coerce_types=True)
    result = field.validate(None)
    assert result == ""

def test_validate_non_string_type():
    field = String()
    try:
        field.validate(123)
        assert False
    except Exception as e:
        assert str(e) == "Must be a string."

def test_validate_string_with_null_character():
    field = String()
    result = field.validate("a\0b")
    assert result == "ab"

def test_validate_trim_whitespace():
    field = String(trim_whitespace=True)
    result = field.validate("  test  ")
    assert result == "test"

def test_validate_no_trim_whitespace():
    field = String(trim_whitespace=False)
    result = field.validate("  test  ")
    assert result == "  test  "

def test_validate_blank_without_allow_blank():
    field = String(allow_blank=False)
    try:
        field.validate("")
        assert False
    except Exception as e:
        assert str(e) == "Must not be blank."

def test_validate_blank_with_allow_blank():
    field = String(allow_blank=True)
    result = field.validate("")
    assert result == ""

def test_validate_blank_with_allow_null_and_coerce_types():
    field = String(allow_null=True, coerce_types=True, trim_whitespace=True)
    result = field.validate("   ")
    assert result is None

def test_validate_min_length_valid():
    field = String(min_length=3)
    result = field.validate("abc")
    assert result == "abc"

def test_validate_min_length_invalid():
    field = String(min_length=3)
    try:
        field.validate("ab")
        assert False
    except Exception as e:
        assert str(e) == "Must have at least 3 characters."

def test_validate_max_length_valid():
    field = String(max_length=5)
    result = field.validate("abcde")
    assert result == "abcde"

def test_validate_max_length_invalid():
    field = String(max_length=5)
    try:
        field.validate("abcdef")
        assert False
    except Exception as e:
        assert str(e) == "Must have no more than 5 characters."

def test_validate_pattern_valid():
    field = String(pattern="^[a-z]+$")
    result = field.validate("abc")
    assert result == "abc"

def test_validate_pattern_invalid():
    field = String(pattern="^[a-z]+$")
    try:
        field.validate("abc123")
        assert False
    except Exception as e:
        assert str(e) == "Must match the pattern /^[a-z]+$/."

def test_validate_format_valid():
    field = String(format="email")
    result = field.validate("test@example.com")
    assert result == "test@example.com"

def test_validate_format_invalid():
    field = String(format="email")
    try:
        field.validate("invalid-email")
        assert False
    except Exception as e:
        assert "Must be a valid email." in str(e)

def test_validate_native_type_with_format():
    field = String(format="email")
    native_value = FORMATS["email"].native_type("test@example.com")
    result = field.validate(native_value)
    assert result == native_value


# LLM-generated content at query #21
#--------------------------

def test_assert_all_choices_have_length_two():
    choices = [("a", "b"), ("c", "d"), ("e", "f")]
    field = Choice(choices=choices)
    assert len(field.choices) == 3


# LLM-generated content at query #22
#--------------------------

```python
def test_validate_max_items_error():
    field = Array(max_items=2)
    value = [1, 2, 3]
    try:
        field.validate(value)
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "max_items"
        assert e.messages()[0].text == "Must have no more than 2 items."


# LLM-generated content at query #23
#--------------------------

def test_string_constructor_defaults():
    field = String()
    assert field.title == ""
    assert field.description == ""
    assert field.allow_null == False
    assert field.read_only == False
    assert field.allow_blank == False
    assert field.trim_whitespace == True
    assert field.max_length == None
    assert field.min_length == None
    assert field.pattern == None
    assert field.pattern_regex == None
    assert field.format == None
    assert field.coerce_types == True
    assert not hasattr(field, "default")

def test_string_constructor_with_allow_blank():
    field = String(allow_blank=True)
    assert field.allow_blank == True
    assert field.default == ""

def test_string_constructor_with_title_and_description():
    field = String(title="Name", description="Enter your name")
    assert field.title == "Name"
    assert field.description == "Enter your name"

def test_string_constructor_with_allow_null():
    field = String(allow_null=True)
    assert field.allow_null == True

def test_string_constructor_with_read_only():
    field = String(read_only=True)
    assert field.read_only == True

def test_string_constructor_with_max_length():
    field = String(max_length=10)
    assert field.max_length == 10

def test_string_constructor_with_min_length():
    field = String(min_length=2)
    assert field.min_length == 2

def test_string_constructor_with_pattern_string():
    field = String(pattern="^[a-z]+$")
    assert field.pattern == "^[a-z]+$"
    assert field.pattern_regex is not None

def test_string_constructor_with_pattern_regex():
    import re
    regex = re.compile("^[a-z]+$")
    field = String(pattern=regex)
    assert field.pattern == "^[a-z]+$"
    assert field.pattern_regex is regex

def test_string_constructor_with_format():
    field = String(format="email")
    assert field.format == "email"

def test_string_constructor_with_coerce_types_false():
    field = String(coerce_types=False)
    assert field.coerce_types == False

def test_string_constructor_with_allow_blank_and_default():
    field = String(allow_blank=True, default="test")
    assert field.allow_blank == True
    assert field.default == "test"

def test_string_constructor_with_allow_null_and_default():
    field = String(allow_null=True, default=None)
    assert field.allow_null == True
    assert field.default == None

def test_string_constructor_with_all_parameters():
    import re
    pattern = re.compile("^[A-Z]+$")
    field = String(
        title="Title",
        description="Description",
        default="DEFAULT",
        allow_null=True,
        read_only=True,
        allow_blank=True,
        trim_whitespace=False,
        max_length=100,
        min_length=1,
        pattern=pattern,
        format="uppercase",
        coerce_types=False
    )
    assert field.title == "Title"
    assert field.description == "Description"
    assert field.default == "DEFAULT"
    assert field.allow_null == True
    assert field.read_only == True
    assert field.allow_blank == True
    assert field.trim_whitespace == False
    assert field.max_length == 100
    assert field.min_length == 1
    assert field.pattern == "^[A-Z]+$"
    assert field.pattern_regex is pattern
    assert field.format == "uppercase"
    assert field.coerce_types == False


# LLM-generated content at query #24
#--------------------------

```python
def test_unique_items_duplicate_primitive():
    from typesystem.fields import Array
    from typesystem.base import Message
    from typesystem.exceptions import ValidationError
    field = Array(unique_items=True)
    value = [1, 2, 3, 1]
    try:
        field.validate(value)
    except ValidationError as e:
        assert len(e.messages()) == 1
        message = e.messages()[0]
        assert message.text == "Items must be unique."
        assert message.code == "unique_items"
        assert message.index == [3]
    else:
        assert False


# LLM-generated content at query #25
#--------------------------

def test_union_validate_candidate_errors_condition_true():
    from myapp.fields import Field, Union, Integer, String
    integer_field = Integer()
    string_field = String()
    union_field = Union(any_of=[integer_field, string_field])
    value = "not_an_integer"
    validated, error = integer_field.validate_or_error(value)
    messages = error.messages()
    condition_result = len(messages) != 1 or messages[0].code != "type" or messages[0].index
    assert condition_result == True


# LLM-generated content at query #26
#--------------------------

def test_validate_returns_none_for_null_when_allow_null():
    field = Boolean(allow_null=True)
    result = field.validate(None)
    assert result is None

def test_validate_raises_null_error_for_null_when_not_allow_null():
    field = Boolean(allow_null=False)
    try:
        field.validate(None)
        assert False
    except Exception as e:
        assert str(e) == "May not be null."

def test_validate_returns_true_for_true():
    field = Boolean()
    result = field.validate(True)
    assert result is True

def test_validate_returns_false_for_false():
    field = Boolean()
    result = field.validate(False)
    assert result is False

def test_validate_raises_type_error_for_non_bool_when_coerce_types_false():
    field = Boolean(coerce_types=False)
    try:
        field.validate("true")
        assert False
    except Exception as e:
        assert str(e) == "Must be a boolean."

def test_validate_coerces_string_true_to_true():
    field = Boolean()
    result = field.validate("true")
    assert result is True

def test_validate_coerces_string_false_to_false():
    field = Boolean()
    result = field.validate("false")
    assert result is False

def test_validate_coerces_string_on_to_true():
    field = Boolean()
    result = field.validate("on")
    assert result is True

def test_validate_coerces_string_off_to_false():
    field = Boolean()
    result = field.validate("off")
    assert result is False

def test_validate_coerces_string_1_to_true():
    field = Boolean()
    result = field.validate("1")
    assert result is True

def test_validate_coerces_string_0_to_false():
    field = Boolean()
    result = field.validate("0")
    assert result is False

def test_validate_coerces_empty_string_to_false():
    field = Boolean()
    result = field.validate("")
    assert result is False

def test_validate_coerces_int_1_to_true():
    field = Boolean()
    result = field.validate(1)
    assert result is True

def test_validate_coerces_int_0_to_false():
    field = Boolean()
    result = field.validate(0)
    assert result is False

def test_validate_coerces_empty_string_to_null_when_allow_null():
    field = Boolean(allow_null=True)
    result = field.validate("")
    assert result is None

def test_validate_coerces_string_null_to_null_when_allow_null():
    field = Boolean(allow_null=True)
    result = field.validate("null")
    assert result is None

def test_validate_coerces_string_none_to_null_when_allow_null():
    field = Boolean(allow_null=True)
    result = field.validate("none")
    assert result is None

def test_validate_raises_type_error_for_uncoercible_string():
    field = Boolean()
    try:
        field.validate("invalid")
        assert False
    except Exception as e:
        assert str(e) == "Must be a boolean."

def test_validate_raises_type_error_for_uncoercible_type():
    field = Boolean()
    try:
        field.validate([])
        assert False
    except Exception as e:
        assert str(e) == "Must be a boolean."


# LLM-generated content at query #27
#--------------------------

def test_allow_null_false_and_value_not_in_coerce_null_values():
    field = Boolean(allow_null=False)
    result = field.validate("")
    assert result is False


# LLM-generated content at query #28
#--------------------------

def test_validate_null_when_allowed():
    field = Object(allow_null=True)
    result = field.validate(None)
    assert result is None

def test_validate_null_when_not_allowed():
    field = Object(allow_null=False)
    try:
        field.validate(None)
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "null"

def test_validate_non_dict_type():
    field = Object()
    try:
        field.validate("not a dict")
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "type"

def test_validate_invalid_key_type():
    field = Object()
    try:
        field.validate({1: "value"})
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "invalid_key"

def test_validate_property_names_invalid():
    property_names_field = Field(allow_null=False)
    field = Object(property_names=property_names_field)
    try:
        field.validate({"": "value"})
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "invalid_property"

def test_validate_min_properties_empty():
    field = Object(min_properties=1)
    try:
        field.validate({})
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "empty"

def test_validate_min_properties_violation():
    field = Object(min_properties=2)
    try:
        field.validate({"a": 1})
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "min_properties"

def test_validate_max_properties_violation():
    field = Object(max_properties=1)
    try:
        field.validate({"a": 1, "b": 2})
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "max_properties"

def test_validate_required_missing():
    field = Object(required=["key"])
    try:
        field.validate({})
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "required"

def test_validate_properties_with_default():
    child_field = Field(default="default_value")
    field = Object(properties={"key": child_field})
    result = field.validate({})
    assert result == {"key": "default_value"}

def test_validate_properties_valid():
    child_field = Field()
    field = Object(properties={"key": child_field})
    result = field.validate({"key": "value"})
    assert result == {"key": "value"}

def test_validate_properties_invalid():
    child_field = Field(allow_null=False)
    field = Object(properties={"key": child_field})
    try:
        field.validate({"key": None})
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "null"

def test_validate_pattern_properties_match():
    child_field = Field()
    field = Object(pattern_properties={"^a.*": child_field})
    result = field.validate({"abc": "value"})
    assert result == {"abc": "value"}

def test_validate_pattern_properties_no_match():
    child_field = Field()
    field = Object(pattern_properties={"^a.*": child_field})
    result = field.validate({"xyz": "value"})
    assert result == {"xyz": "value"}

def test_validate_additional_properties_true():
    field = Object(additional_properties=True)
    result = field.validate({"extra": "value"})
    assert result == {"extra": "value"}

def test_validate_additional_properties_false():
    field = Object(additional_properties=False)
    try:
        field.validate({"extra": "value"})
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "invalid_property"

def test_validate_additional_properties_field_valid():
    additional_field = Field()
    field = Object(additional_properties=additional_field)
    result = field.validate({"extra": "value"})
    assert result == {"extra": "value"}

def test_validate_additional_properties_field_invalid():
    additional_field = Field(allow_null=False)
    field = Object(additional_properties=additional_field)
    try:
        field.validate({"extra": None})
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "null"

def test_validate_multiple_errors():
    field = Object(required=["req"], additional_properties=False)
    try:
        field.validate({"extra": "value"})
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 2
        codes = {msg.code for msg in e.messages()}
        assert codes == {"required", "invalid_property"}

def test_validate_complex_nested():
    child_field = Field()
    field = Object(properties={"key": child_field}, additional_properties=False)
    result = field.validate({"key": "value"})
    assert result == {"key": "value"}


# LLM-generated content at query #29
#--------------------------

def test_format_in_formats_and_is_native_type():
    field = String(format="email")
    value = "test@example.com"
    result = field.validate(value)
    assert result == value


# LLM-generated content at query #30
#--------------------------

```python
def test_validate_with_non_null_value_and_allow_null_false():
    from typesystem.fields import Object
    field = Object(allow_null=False)
    value = {"key": "value"}
    result = field.validate(value)
    assert result == {"key": "value"}


# LLM-generated content at query #31
#--------------------------

def test_validate_null_allowed():
    field = Number(allow_null=True)
    result = field.validate(None)
    assert result is None

def test_validate_null_not_allowed():
    field = Number(allow_null=False)
    try:
        field.validate(None)
        assert False
    except Exception as e:
        assert str(e) == "May not be null."

def test_validate_empty_string_coerce_to_null():
    field = Number(allow_null=True, coerce_types=True)
    result = field.validate("")
    assert result is None

def test_validate_boolean_raises_type_error():
    field = Number()
    try:
        field.validate(True)
        assert False
    except Exception as e:
        assert str(e) == "Must be a number."

def test_validate_string_integer():
    field = Number()
    result = field.validate("123")
    assert result == 123

def test_validate_string_float():
    field = Number()
    result = field.validate("123.45")
    assert result == 123.45

def test_validate_string_invalid_raises_type_error():
    field = Number()
    try:
        field.validate("abc")
        assert False
    except Exception as e:
        assert str(e) == "Must be a number."

def test_validate_integer_type_with_float_non_integer():
    field = Number(numeric_type=int)
    try:
        field.validate(123.45)
        assert False
    except Exception as e:
        assert str(e) == "Must be an integer."

def test_validate_coerce_types_false_with_string():
    field = Number(coerce_types=False)
    try:
        field.validate("123")
        assert False
    except Exception as e:
        assert str(e) == "Must be a number."

def test_validate_infinite_value():
    field = Number()
    try:
        field.validate(float('inf'))
        assert False
    except Exception as e:
        assert str(e) == "Must be finite."

def test_validate_nan_value():
    field = Number()
    try:
        field.validate(float('nan'))
        assert False
    except Exception as e:
        assert str(e) == "Must be finite."

def test_validate_minimum_violation():
    field = Number(minimum=10)
    try:
        field.validate(5)
        assert False
    except Exception as e:
        assert str(e) == "Must be greater than or equal to 10."

def test_validate_minimum_satisfied():
    field = Number(minimum=10)
    result = field.validate(10)
    assert result == 10

def test_validate_exclusive_minimum_violation():
    field = Number(exclusive_minimum=10)
    try:
        field.validate(10)
        assert False
    except Exception as e:
        assert str(e) == "Must be greater than 10."

def test_validate_exclusive_minimum_satisfied():
    field = Number(exclusive_minimum=10)
    result = field.validate(11)
    assert result == 11

def test_validate_maximum_violation():
    field = Number(maximum=10)
    try:
        field.validate(15)
        assert False
    except Exception as e:
        assert str(e) == "Must be less than or equal to 10."

def test_validate_maximum_satisfied():
    field = Number(maximum=10)
    result = field.validate(10)
    assert result == 10

def test_validate_exclusive_maximum_violation():
    field = Number(exclusive_maximum=10)
    try:
        field.validate(10)
        assert False
    except Exception as e:
        assert str(e) == "Must be less than 10."

def test_validate_exclusive_maximum_satisfied():
    field = Number(exclusive_maximum=10)
    result = field.validate(9)
    assert result == 9

def test_validate_multiple_of_int_violation():
    field = Number(multiple_of=5)
    try:
        field.validate(12)
        assert False
    except Exception as e:
        assert str(e) == "Must be a multiple of 5."

def test_validate_multiple_of_int_satisfied():
    field = Number(multiple_of=5)
    result = field.validate(15)
    assert result == 15

def test_validate_multiple_of_float_violation():
    field = Number(multiple_of=0.5)
    try:
        field.validate(1.2)
        assert False
    except Exception as e:
        assert str(e) == "Must be a multiple of 0.5."

def test_validate_multiple_of_float_satisfied():
    field = Number(multiple_of=0.5)
    result = field.validate(1.5)
    assert result == 1.5

def test_validate_precision_rounding():
    field = Number(precision="0.01")
    result = field.validate(1.234)
    assert result == 1.23

def test_validate_precision_rounding_up():
    field = Number(precision="0.01")
    result = field.validate(1.235)
    assert result == 1.24


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_array_constructor_with_defaults():
    field = Array()
    assert field.items is None
    assert field.additional_items is False
    assert field.min_items is None
    assert field.max_items is None
    assert field.unique_items is False
    assert field.allow_null is False
    assert field.read_only is False
    assert field.title == ""
    assert field.description == ""

def test_array_constructor_with_single_item_field():
    item_field = Field()
    field = Array(items=item_field)
    assert field.items == item_field
    assert field.additional_items is False
    assert field.min_items is None
    assert field.max_items is None
    assert field.unique_items is False

def test_array_constructor_with_list_of_items():
    item_field1 = Field()
    item_field2 = Field()
    field = Array(items=[item_field1, item_field2])
    assert field.items == [item_field1, item_field2]
    assert field.additional_items is False
    assert field.min_items == 2
    assert field.max_items == 2

def test_array_constructor_with_additional_items_as_field():
    item_field = Field()
    additional_field = Field()
    field = Array(items=item_field, additional_items=additional_field)
    assert field.items == item_field
    assert field.additional_items == additional_field
    assert field.min_items is None
    assert field.max_items is None

def test_array_constructor_with_additional_items_as_false():
    item_field = Field()
    field = Array(items=item_field, additional_items=False)
    assert field.items == item_field
    assert field.additional_items is False

def test_array_constructor_with_min_items():
    field = Array(min_items=3)
    assert field.min_items == 3
    assert field.max_items is None

def test_array_constructor_with_max_items():
    field = Array(max_items=5)
    assert field.min_items is None
    assert field.max_items == 5

def test_array_constructor_with_exact_items():
    field = Array(exact_items=4)
    assert field.min_items == 4
    assert field.max_items == 4

def test_array_constructor_with_unique_items():
    field = Array(unique_items=True)
    assert field.unique_items is True

def test_array_constructor_with_allow_null():
    field = Array(allow_null=True)
    assert field.allow_null is True

def test_array_constructor_with_read_only():
    field = Array(read_only=True)
    assert field.read_only is True

def test_array_constructor_with_title_and_description():
    field = Array(title="Test Array", description="An array for testing")
    assert field.title == "Test Array"
    assert field.description == "An array for testing"

def test_array_constructor_with_list_items_and_additional_items_false_sets_max_items():
    item_field1 = Field()
    item_field2 = Field()
    field = Array(items=[item_field1, item_field2], additional_items=False)
    assert field.min_items == 2
    assert field.max_items == 2

def test_array_constructor_with_list_items_and_additional_items_field_does_not_set_max_items():
    item_field1 = Field()
    item_field2 = Field()
    additional_field = Field()
    field = Array(items=[item_field1, item_field2], additional_items=additional_field)
    assert field.min_items == 2
    assert field.max_items is None

def test_array_constructor_exact_items_overrides_min_and_max():
    field = Array(exact_items=7, min_items=1, max_items=10)
    assert field.min_items == 7
    assert field.max_items == 7

def test_array_constructor_with_default_value():
    field = Array(default=[1, 2, 3])
    assert field.has_default()
    assert field.get_default_value() == [1, 2, 3]

def test_array_constructor_with_callable_default():
    def default_func():
        return [4, 5]
    field = Array(default=default_func)
    assert field.get_default_value() == [4, 5]


# LLM-generated content at query #2
#--------------------------

def test_validate_returns_none_when_value_is_null_and_allow_null_is_true():
    child_field = Field(allow_null=False)
    union_field = Union(any_of=[child_field], allow_null=True)
    result = union_field.validate(None)
    assert result is None

def test_validate_raises_null_error_when_value_is_null_and_allow_null_is_false():
    child_field = Field(allow_null=False)
    union_field = Union(any_of=[child_field], allow_null=False)
    try:
        union_field.validate(None)
        assert False
    except ValidationError as e:
        assert e.messages()[0].code == "null"

def test_validate_returns_validated_value_when_one_child_validates():
    child_field = Field()
    child_field.validate = lambda x: x
    child_field.validate_or_error = lambda x: (x, None)
    union_field = Union(any_of=[child_field])
    result = union_field.validate("test")
    assert result == "test"

def test_validate_raises_child_error_when_one_child_has_non_type_error():
    child_field = Field()
    child_error = ValidationError(messages=[Message(code="custom", index=None)])
    child_field.validate_or_error = lambda x: (None, child_error)
    union_field = Union(any_of=[child_field])
    try:
        union_field.validate("test")
        assert False
    except ValidationError as e:
        assert e is child_error

def test_validate_raises_union_error_when_no_child_validates_and_multiple_candidate_errors():
    child_field1 = Field()
    child_error1 = ValidationError(messages=[Message(code="custom1", index=None)])
    child_field1.validate_or_error = lambda x: (None, child_error1)
    child_field2 = Field()
    child_error2 = ValidationError(messages=[Message(code="custom2", index=None)])
    child_field2.validate_or_error = lambda x: (None, child_error2)
    union_field = Union(any_of=[child_field1, child_field2])
    try:
        union_field.validate("test")
        assert False
    except ValidationError as e:
        assert e.messages()[0].code == "union"

def test_validate_raises_child_error_when_only_one_child_has_non_type_error():
    child_field1 = Field()
    child_error1 = ValidationError(messages=[Message(code="type", index=None)])
    child_field1.validate_or_error = lambda x: (None, child_error1)
    child_field2 = Field()
    child_error2 = ValidationError(messages=[Message(code="custom", index=None)])
    child_field2.validate_or_error = lambda x: (None, child_error2)
    union_field = Union(any_of=[child_field1, child_field2])
    try:
        union_field.validate("test")
        assert False
    except ValidationError as e:
        assert e is child_error2

def test_validate_raises_union_error_when_all_children_have_type_errors_without_index():
    child_field1 = Field()
    child_error1 = ValidationError(messages=[Message(code="type", index=None)])
    child_field1.validate_or_error = lambda x: (None, child_error1)
    child_field2 = Field()
    child_error2 = ValidationError(messages=[Message(code="type", index=None)])
    child_field2.validate_or_error = lambda x: (None, child_error2)
    union_field = Union(any_of=[child_field1, child_field2])
    try:
        union_field.validate("test")
        assert False
    except ValidationError as e:
        assert e.messages()[0].code == "union"

def test_validate_raises_child_error_when_child_type_error_has_index():
    child_field = Field()
    child_error = ValidationError(messages=[Message(code="type", index=[0])])
    child_field.validate_or_error = lambda x: (None, child_error)
    union_field = Union(any_of=[child_field])
    try:
        union_field.validate("test")
        assert False
    except ValidationError as e:
        assert e is child_error

def test_validate_allow_null_set_true_if_any_child_allows_null():
    child_field1 = Field(allow_null=True)
    child_field2 = Field(allow_null=False)
    union_field = Union(any_of=[child_field1, child_field2])
    assert union_field.allow_null is True


# LLM-generated content at query #3
#--------------------------

def test_get_default_value_with_default_attribute():
    field = Field(default="test_default")
    result = field.get_default_value()
    assert result == "test_default"

def test_get_default_value_with_callable_default():
    field = Field(default=lambda: "callable_result")
    result = field.get_default_value()
    assert result == "callable_result"

def test_get_default_value_without_default_attribute():
    field = Field()
    result = field.get_default_value()
    assert result is None

def test_get_default_value_with_none_default():
    field = Field(default=None)
    result = field.get_default_value()
    assert result is None

def test_get_default_value_with_allow_null_and_no_default():
    field = Field(allow_null=True)
    result = field.get_default_value()
    assert result is None


# LLM-generated content at query #4
#--------------------------

def test_serialize_with_none_and_allow_null():
    field = Array(allow_null=True)
    result = field.serialize(None)
    assert result is None

def test_serialize_with_none_and_not_allow_null():
    field = Array()
    result = field.serialize(None)
    assert result is None

def test_serialize_with_no_items_field():
    field = Array()
    obj = [1, 2, 3]
    result = field.serialize(obj)
    assert result == [1, 2, 3]

def test_serialize_with_single_item_field():
    mock_field = Mock()
    mock_field.serialize.side_effect = lambda x: x * 2
    field = Array(items=mock_field)
    obj = [1, 2, 3]
    result = field.serialize(obj)
    assert result == [2, 4, 6]

def test_serialize_with_list_of_item_fields():
    mock_field1 = Mock()
    mock_field1.serialize.side_effect = lambda x: x + 1
    mock_field2 = Mock()
    mock_field2.serialize.side_effect = lambda x: x * 2
    field = Array(items=[mock_field1, mock_field2])
    obj = [5, 10]
    result = field.serialize(obj)
    assert result == [6, 20]

def test_serialize_with_list_of_item_fields_and_shorter_obj():
    mock_field1 = Mock()
    mock_field1.serialize.side_effect = lambda x: x + 1
    mock_field2 = Mock()
    mock_field2.serialize.side_effect = lambda x: x * 2
    field = Array(items=[mock_field1, mock_field2])
    obj = [5]
    result = field.serialize(obj)
    assert result == [6]

def test_serialize_with_list_of_item_fields_and_longer_obj():
    mock_field1 = Mock()
    mock_field1.serialize.side_effect = lambda x: x + 1
    mock_field2 = Mock()
    mock_field2.serialize.side_effect = lambda x: x * 2
    field = Array(items=[mock_field1, mock_field2])
    obj = [5, 10, 15]
    result = field.serialize(obj)
    assert result == [6, 20, 15]


# LLM-generated content at query #5
#--------------------------

def test_validate_null_when_allowed():
    field = Array(allow_null=True)
    result = field.validate(None)
    assert result is None

def test_validate_null_when_not_allowed():
    field = Array()
    try:
        field.validate(None)
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].text == "May not be null."
        assert e.messages()[0].code == "null"

def test_validate_non_list_type():
    field = Array()
    try:
        field.validate("not a list")
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].text == "Must be an array."
        assert e.messages()[0].code == "type"

def test_validate_empty_list_with_min_items_one():
    field = Array(min_items=1)
    try:
        field.validate([])
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].text == "Must not be empty."
        assert e.messages()[0].code == "empty"

def test_validate_list_with_min_items():
    field = Array(min_items=3)
    try:
        field.validate([1, 2])
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].text == "Must have at least 3 items."
        assert e.messages()[0].code == "min_items"

def test_validate_list_with_max_items():
    field = Array(max_items=2)
    try:
        field.validate([1, 2, 3])
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].text == "Must have no more than 2 items."
        assert e.messages()[0].code == "max_items"

def test_validate_list_with_exact_items():
    field = Array(exact_items=2)
    try:
        field.validate([1])
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].text == "Must have 2 items."
        assert e.messages()[0].code == "exact_items"

def test_validate_list_with_exact_items_mismatch():
    field = Array(exact_items=2)
    try:
        field.validate([1, 2, 3])
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].text == "Must have 2 items."
        assert e.messages()[0].code == "exact_items"

def test_validate_list_with_item_validator():
    item_field = Integer()
    field = Array(items=item_field)
    result = field.validate([1, 2, 3])
    assert result == [1, 2, 3]

def test_validate_list_with_item_validator_error():
    item_field = Integer()
    field = Array(items=item_field)
    try:
        field.validate([1, "invalid", 3])
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].text == "Must be a valid integer."
        assert e.messages()[0].index == [1]

def test_validate_list_with_multiple_item_validators():
    item_fields = [Integer(), String()]
    field = Array(items=item_fields)
    result = field.validate([1, "hello"])
    assert result == [1, "hello"]

def test_validate_list_with_multiple_item_validators_error():
    item_fields = [Integer(), String()]
    field = Array(items=item_fields)
    try:
        field.validate(["invalid", 123])
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].text == "Must be a valid integer."
        assert e.messages()[0].index == [0]

def test_validate_list_with_additional_items_false():
    item_fields = [Integer(), String()]
    field = Array(items=item_fields, additional_items=False)
    try:
        field.validate([1, "hello", "extra"])
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].text == "May not contain additional items."
        assert e.messages()[0].code == "additional_items"

def test_validate_list_with_additional_items_field():
    item_fields = [Integer(), String()]
    additional_field = Boolean()
    field = Array(items=item_fields, additional_items=additional_field)
    result = field.validate([1, "hello", True, False])
    assert result == [1, "hello", True, False]

def test_validate_list_with_additional_items_field_error():
    item_fields = [Integer(), String()]
    additional_field = Boolean()
    field = Array(items=item_fields, additional_items=additional_field)
    try:
        field.validate([1, "hello", "not_bool"])
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].text == "Must be a valid boolean."
        assert e.messages()[0].index == [2]

def test_validate_list_with_unique_items():
    field = Array(unique_items=True)
    try:
        field.validate([1, 2, 1])
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].text == "Items must be unique."
        assert e.messages()[0].code == "unique_items"
        assert e.messages()[0].key == 2

def test_validate_list_with_unique_items_and_booleans():
    field = Array(unique_items=True)
    result = field.validate([True, False, 1, 0])
    assert result == [True, False, 1, 0]

def test_validate_list_with_unique_items_and_nested():
    field = Array(unique_items=True)
    result = field.validate([[1, 2], [1, 2]])
    assert result == [[1, 2], [1, 2]]

def test_validate_list_with_unique_items_and_dicts():
    field = Array(unique_items=True)
    result = field.validate([{"a": 1}, {"a": 1}])
    assert result == [{"a": 1}, {"a": 1}]

def test_validate_list_with_multiple_errors():
    item_field = Integer()
    field = Array(items=item_field, min_items=2, unique_items=True)
    try:
        field.validate([1, "invalid", 1])
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 2
        codes = {msg.code for msg in e.messages()}
        assert codes == {"invalid", "unique_items"}

def test_validate_list_with_no_item_validators():
    field = Array()
    result = field.validate([1, "hello", True])
    assert result == [1, "hello", True]

def test_validate_list_with_single_item_validator_for_all():
    item_field = Integer()
    field = Array(items=item_field)
    result = field.validate([1, 2, 3])
    assert result == [1, 2, 3]

def test_validate_list_with_single_item_validator_error_for_all():
    item_field = Integer()
    field = Array(items=item_field)
    try:
        field.validate([1, "invalid", 3])
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].index == [1]


# LLM-generated content at query #6
#--------------------------

def test_condition_true_when_multiple_messages():
    from myapp.fields import Union, Integer, String
    from myapp.exceptions import ValidationError
    class CustomField:
        def validate_or_error(self, value):
            error = ValidationError(messages=[ValidationError.Message(code="type", index=None), ValidationError.Message(code="required", index=None)])
            return None, error
    union = Union(any_of=[CustomField()])
    try:
        union.validate(value=5)
    except ValidationError as e:
        pass

def test_condition_true_when_single_message_not_type():
    from myapp.fields import Union, Integer, String
    from myapp.exceptions import ValidationError
    class CustomField:
        def validate_or_error(self, value):
            error = ValidationError(messages=[ValidationError.Message(code="required", index=None)])
            return None, error
    union = Union(any_of=[CustomField()])
    try:
        union.validate(value=5)
    except ValidationError as e:
        pass

def test_condition_true_when_single_type_message_with_index():
    from myapp.fields import Union, Integer, String
    from myapp.exceptions import ValidationError
    class CustomField:
        def validate_or_error(self, value):
            error = ValidationError(messages=[ValidationError.Message(code="type", index=[0])])
            return None, error
    union = Union(any_of=[CustomField()])
    try:
        union.validate(value=5)
    except ValidationError as e:
        pass


# LLM-generated content at query #7
#--------------------------

def test_serialize_with_none():
    decimal_instance = Decimal()
    result = decimal_instance.serialize(None)
    assert result is None


def test_serialize_with_decimal():
    decimal_instance = Decimal()
    decimal_obj = decimal.Decimal('10.5')
    result = decimal_instance.serialize(decimal_obj)
    assert result == 10.5


def test_serialize_with_integer_decimal():
    decimal_instance = Decimal()
    decimal_obj = decimal.Decimal('7')
    result = decimal_instance.serialize(decimal_obj)
    assert result == 7.0


# LLM-generated content at query #8
#--------------------------

```python
def test_max_items_validation():
    field = Array(max_items=2)
    value = [1, 2, 3]
    try:
        field.validate(value)
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "max_items"
        assert e.messages()[0].text == "Must have no more than 2 items."


# LLM-generated content at query #9
#--------------------------

def test_validate_null_when_allowed():
    field = Object(allow_null=True)
    result = field.validate(None)
    assert result is None

def test_validate_null_when_not_allowed():
    field = Object()
    try:
        field.validate(None)
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "null"

def test_validate_non_dict_type():
    field = Object()
    try:
        field.validate("not a dict")
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "type"

def test_validate_invalid_key_type():
    field = Object()
    try:
        field.validate({1: "value"})
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "invalid_key"

def test_validate_property_names_invalid():
    property_names_field = Field()
    property_names_field.validate = lambda x: (_ for _ in ()).throw(ValidationError(messages=[Message(text="Invalid", code="custom")]))
    field = Object(property_names=property_names_field)
    try:
        field.validate({"invalid_key": "value"})
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "invalid_property"

def test_validate_min_properties_empty():
    field = Object(min_properties=1)
    try:
        field.validate({})
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "empty"

def test_validate_min_properties_violation():
    field = Object(min_properties=2)
    try:
        field.validate({"a": 1})
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "min_properties"

def test_validate_max_properties_violation():
    field = Object(max_properties=1)
    try:
        field.validate({"a": 1, "b": 2})
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "max_properties"

def test_validate_required_missing():
    field = Object(required=["required_key"])
    try:
        field.validate({"other_key": "value"})
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "required"

def test_validate_properties_with_default():
    child_field = Field(default="default_value")
    field = Object(properties={"key": child_field})
    result = field.validate({})
    assert result == {"key": "default_value"}

def test_validate_properties_valid():
    child_field = Field()
    child_field.validate = lambda x: x
    field = Object(properties={"key": child_field})
    result = field.validate({"key": "value"})
    assert result == {"key": "value"}

def test_validate_properties_invalid():
    child_field = Field()
    child_field.validate_or_error = lambda x: (None, ValidationError(messages=[Message(text="Error", code="custom")]))
    field = Object(properties={"key": child_field})
    try:
        field.validate({"key": "value"})
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "custom"

def test_validate_pattern_properties_valid():
    child_field = Field()
    child_field.validate = lambda x: x
    field = Object(pattern_properties={"^a.*": child_field})
    result = field.validate({"abc": "value"})
    assert result == {"abc": "value"}

def test_validate_pattern_properties_invalid():
    child_field = Field()
    child_field.validate_or_error = lambda x: (None, ValidationError(messages=[Message(text="Error", code="custom")]))
    field = Object(pattern_properties={"^a.*": child_field})
    try:
        field.validate({"abc": "value"})
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "custom"

def test_validate_additional_properties_true():
    field = Object(additional_properties=True)
    result = field.validate({"extra": "value"})
    assert result == {"extra": "value"}

def test_validate_additional_properties_false():
    field = Object(additional_properties=False)
    try:
        field.validate({"extra": "value"})
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "invalid_property"

def test_validate_additional_properties_field_valid():
    child_field = Field()
    child_field.validate = lambda x: x
    field = Object(additional_properties=child_field)
    result = field.validate({"extra": "value"})
    assert result == {"extra": "value"}

def test_validate_additional_properties_field_invalid():
    child_field = Field()
    child_field.validate_or_error = lambda x: (None, ValidationError(messages=[Message(text="Error", code="custom")]))
    field = Object(additional_properties=child_field)
    try:
        field.validate({"extra": "value"})
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "custom"

def test_validate_multiple_errors():
    field = Object(required=["req"], additional_properties=False)
    try:
        field.validate({"extra": "value"})
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 2
        codes = {msg.code for msg in e.messages()}
        assert codes == {"required", "invalid_property"}


# LLM-generated content at query #10
#--------------------------

def test_max_items_not_set_when_additional_items_is_field():
    field = Field()
    array = Array(items=[field], additional_items=field)
    assert array.max_items is None


# LLM-generated content at query #11
#--------------------------

def test_validate_null_without_allow_null():
    field = Number()
    try:
        field.validate(None)
        assert False
    except Exception as e:
        assert str(e) == "May not be null."

def test_validate_null_with_allow_null():
    field = Number(allow_null=True)
    result = field.validate(None)
    assert result is None

def test_validate_empty_string_with_allow_null_and_coerce_types():
    field = Number(allow_null=True, coerce_types=True)
    result = field.validate("")
    assert result is None

def test_validate_empty_string_without_allow_null():
    field = Number(allow_null=False)
    try:
        field.validate("")
        assert False
    except Exception as e:
        assert str(e) == "Must be a number."

def test_validate_boolean():
    field = Number()
    try:
        field.validate(True)
        assert False
    except Exception as e:
        assert str(e) == "Must be a number."

def test_validate_string_integer():
    field = Number()
    result = field.validate("123")
    assert result == 123

def test_validate_string_float():
    field = Number()
    result = field.validate("123.45")
    assert result == 123.45

def test_validate_invalid_string():
    field = Number()
    try:
        field.validate("abc")
        assert False
    except Exception as e:
        assert str(e) == "Must be a number."

def test_validate_float_for_int_numeric_type():
    field = Number(numeric_type=int)
    try:
        field.validate(123.5)
        assert False
    except Exception as e:
        assert str(e) == "Must be an integer."

def test_validate_infinite():
    field = Number()
    try:
        field.validate(float('inf'))
        assert False
    except Exception as e:
        assert str(e) == "Must be finite."

def test_validate_nan():
    field = Number()
    try:
        field.validate(float('nan'))
        assert False
    except Exception as e:
        assert str(e) == "Must be finite."

def test_validate_minimum_violation():
    field = Number(minimum=10)
    try:
        field.validate(5)
        assert False
    except Exception as e:
        assert str(e) == "Must be greater than or equal to 10."

def test_validate_minimum_satisfied():
    field = Number(minimum=10)
    result = field.validate(10)
    assert result == 10

def test_validate_exclusive_minimum_violation():
    field = Number(exclusive_minimum=10)
    try:
        field.validate(10)
        assert False
    except Exception as e:
        assert str(e) == "Must be greater than 10."

def test_validate_exclusive_minimum_satisfied():
    field = Number(exclusive_minimum=10)
    result = field.validate(11)
    assert result == 11

def test_validate_maximum_violation():
    field = Number(maximum=10)
    try:
        field.validate(15)
        assert False
    except Exception as e:
        assert str(e) == "Must be less than or equal to 10."

def test_validate_maximum_satisfied():
    field = Number(maximum=10)
    result = field.validate(10)
    assert result == 10

def test_validate_exclusive_maximum_violation():
    field = Number(exclusive_maximum=10)
    try:
        field.validate(10)
        assert False
    except Exception as e:
        assert str(e) == "Must be less than 10."

def test_validate_exclusive_maximum_satisfied():
    field = Number(exclusive_maximum=10)
    result = field.validate(9)
    assert result == 9

def test_validate_multiple_of_int_violation():
    field = Number(multiple_of=5)
    try:
        field.validate(12)
        assert False
    except Exception as e:
        assert str(e) == "Must be a multiple of 5."

def test_validate_multiple_of_int_satisfied():
    field = Number(multiple_of=5)
    result = field.validate(15)
    assert result == 15

def test_validate_multiple_of_float_violation():
    field = Number(multiple_of=0.5)
    try:
        field.validate(1.2)
        assert False
    except Exception as e:
        assert str(e) == "Must be a multiple of 0.5."

def test_validate_multiple_of_float_satisfied():
    field = Number(multiple_of=0.5)
    result = field.validate(1.5)
    assert result == 1.5

def test_validate_precision():
    field = Number(precision="0.01")
    result = field.validate(1.234)
    assert result == 1.23

def test_validate_coerce_types_disabled():
    field = Number(coerce_types=False)
    try:
        field.validate("123")
        assert False
    except Exception as e:
        assert str(e) == "Must be a number."


# LLM-generated content at query #12
#--------------------------

def test_validate_null_with_allow_null():
    field = String(allow_null=True)
    result = field.validate(None)
    assert result is None

def test_validate_null_without_allow_null():
    field = String(allow_null=False)
    try:
        field.validate(None)
        assert False
    except Exception as e:
        assert str(e) == "May not be null."

def test_validate_null_with_allow_blank_and_coerce_types():
    field = String(allow_blank=True, coerce_types=True)
    result = field.validate(None)
    assert result == ""

def test_validate_non_string_type():
    field = String()
    try:
        field.validate(123)
        assert False
    except Exception as e:
        assert str(e) == "Must be a string."

def test_validate_string_with_null_character():
    field = String()
    result = field.validate("hello\0world")
    assert result == "helloworld"

def test_validate_trim_whitespace():
    field = String(trim_whitespace=True)
    result = field.validate("  hello  ")
    assert result == "hello"

def test_validate_no_trim_whitespace():
    field = String(trim_whitespace=False)
    result = field.validate("  hello  ")
    assert result == "  hello  "

def test_validate_blank_without_allow_blank():
    field = String(allow_blank=False)
    try:
        field.validate("")
        assert False
    except Exception as e:
        assert str(e) == "Must not be blank."

def test_validate_blank_with_allow_blank():
    field = String(allow_blank=True)
    result = field.validate("")
    assert result == ""

def test_validate_blank_with_allow_null_and_coerce_types():
    field = String(allow_null=True, coerce_types=True, trim_whitespace=True)
    result = field.validate("  ")
    assert result is None

def test_validate_min_length_violation():
    field = String(min_length=5)
    try:
        field.validate("hi")
        assert False
    except Exception as e:
        assert str(e) == "Must have at least 5 characters."

def test_validate_min_length_satisfied():
    field = String(min_length=3)
    result = field.validate("hey")
    assert result == "hey"

def test_validate_max_length_violation():
    field = String(max_length=5)
    try:
        field.validate("hello world")
        assert False
    except Exception as e:
        assert str(e) == "Must have no more than 5 characters."

def test_validate_max_length_satisfied():
    field = String(max_length=10)
    result = field.validate("hello")
    assert result == "hello"

def test_validate_pattern_violation():
    field = String(pattern="^[a-z]+$")
    try:
        field.validate("hello123")
        assert False
    except Exception as e:
        assert str(e) == "Must match the pattern /^[a-z]+$/."

def test_validate_pattern_satisfied():
    field = String(pattern="^[a-z]+$")
    result = field.validate("hello")
    assert result == "hello"

def test_validate_format():
    field = String(format="email")
    result = field.validate("test@example.com")
    assert result == "test@example.com"

def test_validate_format_violation():
    field = String(format="email")
    try:
        field.validate("invalid-email")
        assert False
    except Exception as e:
        assert "email" in str(e)


# LLM-generated content at query #13
#--------------------------

def test_max_items_not_set_when_additional_items_is_field():
    field = Array(items=[], additional_items=Field())
    assert field.max_items is None


# LLM-generated content at query #14
#--------------------------

def test_union_validation_with_one_candidate_error():
    from myapp.fields import Union, Integer, String
    union_field = Union(any_of=[Integer(), String()])
    result = union_field.validate("not_an_integer")
    assert isinstance(result, str)


# LLM-generated content at query #15
#--------------------------

def test_validate_with_numeric_type_int_and_float_integer():
    field = Number(numeric_type=int)
    result = field.validate(5.0)
    assert result == 5

def test_validate_with_numeric_type_int_and_string_integer():
    field = Number(numeric_type=int)
    result = field.validate("5")
    assert result == 5

def test_validate_with_numeric_type_float_and_integer():
    field = Number(numeric_type=float)
    result = field.validate(5)
    assert result == 5.0

def test_validate_with_numeric_type_none_and_integer():
    field = Number()
    result = field.validate(5)
    assert result == 5

def test_validate_with_numeric_type_none_and_float():
    field = Number()
    result = field.validate(5.5)
    assert result == 5.5

def test_validate_with_numeric_type_none_and_string_float():
    field = Number()
    result = field.validate("5.5")
    assert result == 5.5

def test_validate_with_numeric_type_int_and_integer():
    field = Number(numeric_type=int)
    result = field.validate(5)
    assert result == 5


# LLM-generated content at query #16
#--------------------------

def test_validate_allow_blank_false_allow_null_false_coerce_types_true_empty_string():
    field = String(allow_blank=False, allow_null=False, coerce_types=True)
    try:
        field.validate("")
    except Exception as e:
        assert e.detail == "Must not be blank."

def test_validate_allow_blank_false_allow_null_false_coerce_types_false_empty_string():
    field = String(allow_blank=False, allow_null=False, coerce_types=False)
    try:
        field.validate("")
    except Exception as e:
        assert e.detail == "Must not be blank."

def test_validate_allow_blank_false_allow_null_true_coerce_types_false_empty_string():
    field = String(allow_blank=False, allow_null=True, coerce_types=False)
    try:
        field.validate("")
    except Exception as e:
        assert e.detail == "Must not be blank."

def test_validate_allow_blank_true_allow_null_false_coerce_types_true_empty_string():
    field = String(allow_blank=True, allow_null=False, coerce_types=True)
    result = field.validate("")
    assert result == ""

def test_validate_allow_blank_true_allow_null_true_coerce_types_true_empty_string():
    field = String(allow_blank=True, allow_null=True, coerce_types=True)
    result = field.validate("")
    assert result == ""


# LLM-generated content at query #17
#--------------------------

def test_serialize_with_items_as_list():
    from my_module import Array, Field
    class MockField(Field):
        def serialize(self, value):
            return f"serialized_{value}"
    field1 = MockField()
    field2 = MockField()
    array = Array(items=[field1, field2])
    obj = ["a", "b"]
    result = array.serialize(obj)
    assert result == ["serialized_a", "serialized_b"]


# LLM-generated content at query #18
#--------------------------

def test_predicate_at_line_17_evaluates_true_for_multiple_messages():
    from validators import Field, ValidationError
    class MockChild(Field):
        def validate_or_error(self, value):
            error = ValidationError(messages=[ValidationError.Message(code="type"), ValidationError.Message(code="type")])
            return None, error
    child = MockChild()
    union = Union(any_of=[child])
    validated, error = child.validate_or_error("test")
    messages = error.messages()
    result = len(messages) != 1 or messages[0].code != "type" or messages[0].index
    assert result == True

def test_predicate_at_line_17_evaluates_true_for_non_type_code():
    from validators import Field, ValidationError
    class MockChild(Field):
        def validate_or_error(self, value):
            error = ValidationError(messages=[ValidationError.Message(code="max_length")])
            return None, error
    child = MockChild()
    union = Union(any_of=[child])
    validated, error = child.validate_or_error("test")
    messages = error.messages()
    result = len(messages) != 1 or messages[0].code != "type" or messages[0].index
    assert result == True

def test_predicate_at_line_17_evaluates_true_for_index_present():
    from validators import Field, ValidationError
    class MockChild(Field):
        def validate_or_error(self, value):
            error = ValidationError(messages=[ValidationError.Message(code="type", index=[0])])
            return None, error
    child = MockChild()
    union = Union(any_of=[child])
    validated, error = child.validate_or_error("test")
    messages = error.messages()
    result = len(messages) != 1 or messages[0].code != "type" or messages[0].index
    assert result == True

def test_predicate_at_line_17_evaluates_false_for_single_type_no_index():
    from validators import Field, ValidationError
    class MockChild(Field):
        def validate_or_error(self, value):
            error = ValidationError(messages=[ValidationError.Message(code="type")])
            return None, error
    child = MockChild()
    union = Union(any_of=[child])
    validated, error = child.validate_or_error("test")
    messages = error.messages()
    result = len(messages) != 1 or messages[0].code != "type" or messages[0].index
    assert result == False


# LLM-generated content at query #19
#--------------------------

def test_validate_null_allowed():
    field = Array(allow_null=True)
    result = field.validate(None)
    assert result is None

def test_validate_null_not_allowed():
    field = Array()
    try:
        field.validate(None)
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "null"

def test_validate_not_list():
    field = Array()
    try:
        field.validate("not a list")
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "type"

def test_validate_empty_with_min_items():
    field = Array(min_items=1)
    try:
        field.validate([])
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "empty"

def test_validate_exact_items_failure():
    field = Array(exact_items=3)
    try:
        field.validate([1, 2])
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "exact_items"

def test_validate_min_items_failure():
    field = Array(min_items=3)
    try:
        field.validate([1, 2])
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "min_items"

def test_validate_max_items_failure():
    field = Array(max_items=2)
    try:
        field.validate([1, 2, 3])
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "max_items"

def test_validate_with_item_validator():
    field = Array(items=Integer())
    result = field.validate([1, 2, 3])
    assert result == [1, 2, 3]

def test_validate_with_item_validator_error():
    field = Array(items=Integer())
    try:
        field.validate([1, "invalid", 3])
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].index == [1]

def test_validate_with_list_item_validators():
    field = Array(items=[Integer(), String()])
    result = field.validate([1, "hello"])
    assert result == [1, "hello"]

def test_validate_with_list_item_validators_error():
    field = Array(items=[Integer(), String()])
    try:
        field.validate(["invalid", 123])
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].index == [0]

def test_validate_additional_items_false():
    field = Array(items=[Integer()], additional_items=False)
    try:
        field.validate([1, 2])
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "additional_items"

def test_validate_additional_items_field():
    field = Array(items=[Integer()], additional_items=String())
    result = field.validate([1, "extra"])
    assert result == [1, "extra"]

def test_validate_additional_items_field_error():
    field = Array(items=[Integer()], additional_items=String())
    try:
        field.validate([1, 123])
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].index == [1]

def test_validate_unique_items_failure():
    field = Array(unique_items=True)
    try:
        field.validate([1, 2, 1])
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "unique_items"

def test_validate_unique_items_with_booleans():
    field = Array(unique_items=True)
    result = field.validate([True, False, 1, 0])
    assert result == [True, False, 1, 0]

def test_validate_unique_items_with_lists():
    field = Array(unique_items=True)
    result = field.validate([[1, 2], [1, 2]])
    assert result == [[1, 2], [1, 2]]

def test_validate_unique_items_with_dicts():
    field = Array(unique_items=True)
    result = field.validate([{"a": 1}, {"a": 1}])
    assert result == [{"a": 1}, {"a": 1}]

def test_validate_multiple_errors():
    field = Array(items=[Integer(), String()], unique_items=True)
    try:
        field.validate(["invalid", 123, "duplicate", "duplicate"])
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 3
        codes = {msg.code for msg in e.messages()}
        assert codes == {"type", "type", "unique_items"}


# LLM-generated content at query #20
#--------------------------

def test_multiple_of_non_int_condition_false():
    field = Number(multiple_of=0.5)
    result = field.validate(2.0)
    assert result == 2.0


# LLM-generated content at query #21
#--------------------------

def test_serialize_when_items_is_list_and_obj_is_shorter():
    from myapp import Array, Field
    class MockField(Field):
        def serialize(self, value):
            return f"serialized_{value}"
    items = [MockField(), MockField()]
    array = Array(items=items)
    obj = [1]
    result = array.serialize(obj)
    assert result == ["serialized_1"]


# LLM-generated content at query #22
#--------------------------

def test_string_constructor_defaults():
    field = String()
    assert field.title == ""
    assert field.description == ""
    assert not field.allow_null
    assert not field.read_only
    assert not field.allow_blank
    assert field.trim_whitespace
    assert field.max_length is None
    assert field.min_length is None
    assert field.pattern is None
    assert field.pattern_regex is None
    assert field.format is None
    assert field.coerce_types

def test_string_constructor_with_arguments():
    field = String(title="Name", description="Full name", allow_null=True, read_only=True, allow_blank=True, trim_whitespace=False, max_length=10, min_length=1, pattern="^[a-z]+$", format="email", coerce_types=False)
    assert field.title == "Name"
    assert field.description == "Full name"
    assert field.allow_null
    assert field.read_only
    assert field.allow_blank
    assert not field.trim_whitespace
    assert field.max_length == 10
    assert field.min_length == 1
    assert field.pattern == "^[a-z]+$"
    assert field.pattern_regex is not None
    assert field.format == "email"
    assert not field.coerce_types

def test_string_constructor_with_pattern_as_regex():
    import re
    regex = re.compile("^[a-z]+$")
    field = String(pattern=regex)
    assert field.pattern == "^[a-z]+$"
    assert field.pattern_regex is regex

def test_string_constructor_allow_blank_sets_default():
    field = String(allow_blank=True)
    assert field.has_default()
    assert field.get_default_value() == ""

def test_string_constructor_allow_blank_with_explicit_default():
    field = String(allow_blank=True, default="custom")
    assert field.get_default_value() == "custom"

def test_string_constructor_allow_null_sets_default():
    field = String(allow_null=True)
    assert field.has_default()
    assert field.get_default_value() is None

def test_string_constructor_allow_null_with_explicit_default():
    field = String(allow_null=True, default="not null")
    assert field.get_default_value() == "not null"

def test_string_constructor_invalid_max_length_type():
    try:
        String(max_length="10")
    except AssertionError:
        pass

def test_string_constructor_invalid_min_length_type():
    try:
        String(min_length="1")
    except AssertionError:
        pass

def test_string_constructor_invalid_pattern_type():
    try:
        String(pattern=123)
    except AssertionError:
        pass

def test_string_constructor_invalid_format_type():
    try:
        String(format=123)
    except AssertionError:
        pass


# LLM-generated content at query #23
#--------------------------

```python
def test_pattern_properties_condition_false_for_non_string_key():
    from typesystem.fields import Object, String
    import re
    field = Object(pattern_properties={"^a.*": String()})
    value = {123: "test"}
    result = field.validate(value)
    assert result == {}


# LLM-generated content at query #24
#--------------------------

def test_array_constructor_with_defaults():
    field = Array()
    assert field.items is None
    assert field.additional_items is False
    assert field.min_items is None
    assert field.max_items is None
    assert field.unique_items is False
    assert field.allow_null is False
    assert field.read_only is False
    assert field.title == ""
    assert field.description == ""

def test_array_constructor_with_single_item_field():
    item_field = Field()
    field = Array(items=item_field)
    assert field.items == item_field
    assert field.additional_items is False
    assert field.min_items is None
    assert field.max_items is None

def test_array_constructor_with_list_of_item_fields():
    item_fields = [Field(), Field()]
    field = Array(items=item_fields)
    assert field.items == item_fields
    assert field.additional_items is False
    assert field.min_items == 2
    assert field.max_items == 2

def test_array_constructor_with_additional_items_as_field():
    item_field = Field()
    additional_field = Field()
    field = Array(items=item_field, additional_items=additional_field)
    assert field.items == item_field
    assert field.additional_items == additional_field
    assert field.min_items is None
    assert field.max_items is None

def test_array_constructor_with_additional_items_as_false():
    item_field = Field()
    field = Array(items=item_field, additional_items=False)
    assert field.items == item_field
    assert field.additional_items is False

def test_array_constructor_with_min_items():
    field = Array(min_items=3)
    assert field.min_items == 3
    assert field.max_items is None

def test_array_constructor_with_max_items():
    field = Array(max_items=5)
    assert field.min_items is None
    assert field.max_items == 5

def test_array_constructor_with_exact_items():
    field = Array(exact_items=4)
    assert field.min_items == 4
    assert field.max_items == 4

def test_array_constructor_with_unique_items():
    field = Array(unique_items=True)
    assert field.unique_items is True

def test_array_constructor_with_allow_null():
    field = Array(allow_null=True)
    assert field.allow_null is True

def test_array_constructor_with_read_only():
    field = Array(read_only=True)
    assert field.read_only is True

def test_array_constructor_with_title_and_description():
    field = Array(title="Test Title", description="Test Description")
    assert field.title == "Test Title"
    assert field.description == "Test Description"

def test_array_constructor_with_default_value():
    field = Array(default=[1, 2, 3])
    assert field.has_default() is True
    assert field.get_default_value() == [1, 2, 3]

def test_array_constructor_with_callable_default():
    default_func = lambda: [1, 2]
    field = Array(default=default_func)
    assert field.has_default() is True
    assert field.get_default_value() == [1, 2]

def test_array_constructor_with_allow_null_and_default():
    field = Array(allow_null=True, default=[1, 2])
    assert field.allow_null is True
    assert field.has_default() is True
    assert field.get_default_value() == [1, 2]

def test_array_constructor_with_items_list_and_min_max_inferred():
    item_fields = [Field(), Field(), Field()]
    field = Array(items=item_fields)
    assert field.min_items == 3
    assert field.max_items == 3

def test_array_constructor_with_items_list_and_additional_items_false_and_min_max_inferred():
    item_fields = [Field(), Field()]
    field = Array(items=item_fields, additional_items=False)
    assert field.min_items == 2
    assert field.max_items == 2

def test_array_constructor_with_items_list_and_additional_items_field_and_min_inferred():
    item_fields = [Field(), Field()]
    additional_field = Field()
    field = Array(items=item_fields, additional_items=additional_field)
    assert field.min_items == 2
    assert field.max_items is None

def test_array_constructor_with_exact_items_overrides_min_max():
    field = Array(exact_items=5, min_items=1, max_items=10)
    assert field.min_items == 5
    assert field.max_items == 5


# LLM-generated content at query #25
#--------------------------

def test_validate_allow_null_true():
    field = Boolean(allow_null=True)
    result = field.validate(None)
    assert result is None

def test_validate_allow_null_false():
    field = Boolean(allow_null=False)
    try:
        field.validate(None)
        assert False
    except Exception as e:
        assert str(e) == "May not be null."

def test_validate_bool_true():
    field = Boolean()
    result = field.validate(True)
    assert result is True

def test_validate_bool_false():
    field = Boolean()
    result = field.validate(False)
    assert result is False

def test_validate_coerce_types_false():
    field = Boolean(coerce_types=False)
    try:
        field.validate("true")
        assert False
    except Exception as e:
        assert str(e) == "Must be a boolean."

def test_validate_coerce_string_true():
    field = Boolean()
    result = field.validate("true")
    assert result is True

def test_validate_coerce_string_false():
    field = Boolean()
    result = field.validate("false")
    assert result is False

def test_validate_coerce_string_on():
    field = Boolean()
    result = field.validate("on")
    assert result is True

def test_validate_coerce_string_off():
    field = Boolean()
    result = field.validate("off")
    assert result is False

def test_validate_coerce_string_1():
    field = Boolean()
    result = field.validate("1")
    assert result is True

def test_validate_coerce_string_0():
    field = Boolean()
    result = field.validate("0")
    assert result is False

def test_validate_coerce_string_empty():
    field = Boolean()
    result = field.validate("")
    assert result is False

def test_validate_coerce_int_1():
    field = Boolean()
    result = field.validate(1)
    assert result is True

def test_validate_coerce_int_0():
    field = Boolean()
    result = field.validate(0)
    assert result is False

def test_validate_coerce_allow_null_empty_string():
    field = Boolean(allow_null=True)
    result = field.validate("")
    assert result is None

def test_validate_coerce_allow_null_string_null():
    field = Boolean(allow_null=True)
    result = field.validate("null")
    assert result is None

def test_validate_coerce_allow_null_string_none():
    field = Boolean(allow_null=True)
    result = field.validate("none")
    assert result is None

def test_validate_coerce_invalid_string():
    field = Boolean()
    try:
        field.validate("invalid")
        assert False
    except Exception as e:
        assert str(e) == "Must be a boolean."

def test_validate_coerce_invalid_type():
    field = Boolean()
    try:
        field.validate([])
        assert False
    except Exception as e:
        assert str(e) == "Must be a boolean."


# LLM-generated content at query #26
#--------------------------

def test_choice_constructor_with_default_parameters():
    field = Choice()
    assert field.title == ""
    assert field.description == ""
    assert field.allow_null == False
    assert field.read_only == False
    assert field.choices == []
    assert field.coerce_types == True

def test_choice_constructor_with_custom_parameters():
    field = Choice(title="Test Title", description="Test Description", allow_null=True, read_only=True, choices=[("a", "A"), ("b", "B")], coerce_types=False)
    assert field.title == "Test Title"
    assert field.description == "Test Description"
    assert field.allow_null == True
    assert field.read_only == True
    assert field.choices == [("a", "A"), ("b", "B")]
    assert field.coerce_types == False

def test_choice_constructor_with_choices_as_strings():
    field = Choice(choices=["option1", "option2"])
    assert field.choices == [("option1", "option1"), ("option2", "option2")]

def test_choice_constructor_with_choices_as_tuples():
    field = Choice(choices=[("key1", "value1"), ("key2", "value2")])
    assert field.choices == [("key1", "value1"), ("key2", "value2")]

def test_choice_constructor_with_empty_choices():
    field = Choice(choices=[])
    assert field.choices == []

def test_choice_constructor_with_none_choices():
    field = Choice(choices=None)
    assert field.choices == []

def test_choice_constructor_inherits_field_default():
    field = Choice(default="default_value")
    assert field.has_default() == True
    assert field.get_default_value() == "default_value"

def test_choice_constructor_with_allow_null_and_default():
    field = Choice(allow_null=True)
    assert field.allow_null == True
    assert not hasattr(field, 'default')


# LLM-generated content at query #27
#--------------------------

def test_format_in_formats_and_is_native_type():
    field = String(format="email")
    value = "test@example.com"
    result = field.validate(value)
    assert result == value


# LLM-generated content at query #28
#--------------------------

```python
def test_property_names_validation_success():
    from typesystem.fields import Object, String
    from typesystem.base import Message
    property_names = String(min_length=1, max_length=10)
    field = Object(property_names=property_names)
    value = {"valid_key": "value"}
    result = field.validate(value)
    assert result == {"valid_key": "value"}


# LLM-generated content at query #29
#--------------------------

```python
def test_validate_properties_with_error():
    from typesystem.fields import Object, String
    from typesystem.base import ValidationError
    field = Object(properties={"name": String(max_length=5)})
    value = {"name": "too_long_string"}
    try:
        field.validate(value)
    except ValidationError as exc:
        pass
    else:
        raise AssertionError("Expected ValidationError")


# LLM-generated content at query #30
#--------------------------

def test_validate_allows_null_when_allow_null_true():
    field = Choice(choices=[("a", "A")], allow_null=True)
    result = field.validate(None)
    assert result is None

def test_validate_raises_null_error_when_allow_null_false():
    field = Choice(choices=[("a", "A")], allow_null=False)
    try:
        field.validate(None)
        assert False
    except Exception as exc:
        assert str(exc) == "May not be null."

def test_validate_returns_value_for_valid_choice():
    field = Choice(choices=[("a", "A"), ("b", "B")])
    result = field.validate("a")
    assert result == "a"

def test_validate_raises_choice_error_for_invalid_choice():
    field = Choice(choices=[("a", "A"), ("b", "B")])
    try:
        field.validate("c")
        assert False
    except Exception as exc:
        assert str(exc) == "Not a valid choice."

def test_validate_coerces_empty_string_to_null_when_allow_null_and_coerce_types_true():
    field = Choice(choices=[("a", "A")], allow_null=True, coerce_types=True)
    result = field.validate("")
    assert result is None

def test_validate_raises_required_error_for_empty_string_when_allow_null_false():
    field = Choice(choices=[("a", "A")], allow_null=False, coerce_types=True)
    try:
        field.validate("")
        assert False
    except Exception as exc:
        assert str(exc) == "This field is required."

def test_validate_raises_required_error_for_empty_string_when_coerce_types_false():
    field = Choice(choices=[("a", "A")], allow_null=True, coerce_types=False)
    try:
        field.validate("")
        assert False
    except Exception as exc:
        assert str(exc) == "This field is required."

def test_validate_handles_choices_with_tuple_format():
    field = Choice(choices=[("key1", "Display 1"), ("key2", "Display 2")])
    result = field.validate("key2")
    assert result == "key2"

def test_validate_handles_choices_with_single_string():
    field = Choice(choices=["x", "y"])
    result = field.validate("y")
    assert result == "y"


# LLM-generated content at query #31
#--------------------------

def test_choice_constructor_with_default_parameters():
    field = Choice()
    assert field.title == ""
    assert field.description == ""
    assert field.allow_null == False
    assert field.read_only == False
    assert field.choices == []
    assert field.coerce_types == True

def test_choice_constructor_with_custom_parameters():
    field = Choice(title="Test Title", description="Test Description", allow_null=True, read_only=True, choices=[("a", "A"), ("b", "B")], coerce_types=False)
    assert field.title == "Test Title"
    assert field.description == "Test Description"
    assert field.allow_null == True
    assert field.read_only == True
    assert field.choices == [("a", "A"), ("b", "B")]
    assert field.coerce_types == False

def test_choice_constructor_with_choices_as_strings():
    field = Choice(choices=["option1", "option2"])
    assert field.choices == [("option1", "option1"), ("option2", "option2")]

def test_choice_constructor_with_choices_as_tuples():
    field = Choice(choices=[("key1", "value1"), ("key2", "value2")])
    assert field.choices == [("key1", "value1"), ("key2", "value2")]

def test_choice_constructor_with_empty_choices():
    field = Choice(choices=[])
    assert field.choices == []

def test_choice_constructor_with_none_choices():
    field = Choice(choices=None)
    assert field.choices == []

def test_choice_constructor_inherits_field_defaults():
    field = Choice()
    assert not hasattr(field, "default")

def test_choice_constructor_sets_allow_null_and_default():
    field = Choice(allow_null=True)
    assert field.allow_null == True
    assert field.default == None

def test_choice_constructor_with_explicit_default():
    field = Choice(default="default_value")
    assert field.default == "default_value"

def test_choice_constructor_coerce_types_default():
    field = Choice()
    assert field.coerce_types == True

def test_choice_constructor_coerce_types_false():
    field = Choice(coerce_types=False)
    assert field.coerce_types == False


# LLM-generated content at query #32
#--------------------------

```python
def test_additional_properties_is_none_does_not_trigger_assert():
    from typesystem.fields import Object, String
    field = Object(additional_properties=None)
    result = field.validate({"key": "value"})
    assert result == {"key": "value"}


# LLM-generated content at query #33
#--------------------------

def test_max_items_not_set_when_additional_items_is_field():
    field = Array(items=[], additional_items=Field())
    assert field.max_items is None


# LLM-generated content at query #34
#--------------------------

def test_validate_basic_object():
    field = Object()
    value = {"key": "value"}
    result = field.validate(value)
    assert result == {"key": "value"}

def test_validate_null_with_allow_null():
    field = Object(allow_null=True)
    result = field.validate(None)
    assert result is None

def test_validate_null_without_allow_null():
    field = Object()
    try:
        field.validate(None)
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "null"

def test_validate_non_dict_type():
    field = Object()
    try:
        field.validate("not a dict")
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "type"

def test_validate_invalid_key_type():
    field = Object()
    value = {123: "value"}
    try:
        field.validate(value)
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "invalid_key"

def test_validate_property_names_invalid():
    property_names_field = String(max_length=3)
    field = Object(property_names=property_names_field)
    value = {"longkey": "value"}
    try:
        field.validate(value)
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "invalid_property"

def test_validate_min_properties_empty():
    field = Object(min_properties=1)
    value = {}
    try:
        field.validate(value)
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "empty"

def test_validate_min_properties_violation():
    field = Object(min_properties=2)
    value = {"a": 1}
    try:
        field.validate(value)
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "min_properties"

def test_validate_max_properties_violation():
    field = Object(max_properties=1)
    value = {"a": 1, "b": 2}
    try:
        field.validate(value)
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "max_properties"

def test_validate_required_missing():
    field = Object(required=["key"])
    value = {}
    try:
        field.validate(value)
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "required"

def test_validate_properties_with_default():
    child_field = String(default="default_value")
    field = Object(properties={"key": child_field})
    value = {}
    result = field.validate(value)
    assert result == {"key": "default_value"}

def test_validate_properties_valid():
    child_field = String()
    field = Object(properties={"key": child_field})
    value = {"key": "value"}
    result = field.validate(value)
    assert result == {"key": "value"}

def test_validate_properties_invalid():
    child_field = Integer()
    field = Object(properties={"key": child_field})
    value = {"key": "not an integer"}
    try:
        field.validate(value)
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "type"

def test_validate_pattern_properties_matching():
    child_field = Integer()
    field = Object(pattern_properties={"^a.*": child_field})
    value = {"abc": 123}
    result = field.validate(value)
    assert result == {"abc": 123}

def test_validate_pattern_properties_non_matching():
    child_field = Integer()
    field = Object(pattern_properties={"^a.*": child_field})
    value = {"bcd": 123}
    result = field.validate(value)
    assert result == {"bcd": 123}

def test_validate_additional_properties_true():
    field = Object(additional_properties=True)
    value = {"extra": "value"}
    result = field.validate(value)
    assert result == {"extra": "value"}

def test_validate_additional_properties_false():
    field = Object(additional_properties=False)
    value = {"extra": "value"}
    try:
        field.validate(value)
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "invalid_property"

def test_validate_additional_properties_field_valid():
    additional_field = Integer()
    field = Object(additional_properties=additional_field)
    value = {"extra": 123}
    result = field.validate(value)
    assert result == {"extra": 123}

def test_validate_additional_properties_field_invalid():
    additional_field = Integer()
    field = Object(additional_properties=additional_field)
    value = {"extra": "not an integer"}
    try:
        field.validate(value)
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "type"

def test_validate_combined_errors():
    field = Object(required=["req"], properties={"prop": Integer()}, additional_properties=False)
    value = {"prop": "not int", "extra": "value"}
    try:
        field.validate(value)
        assert False
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 3
        codes = {msg.code for msg in messages}
        assert "required" in codes
        assert "type" in codes
        assert "invalid_property" in codes


# LLM-generated content at query #35
#--------------------------

```python
def test_unique_items_duplicate_primitive():
    from typesystem.fields import Array
    from typesystem.base import Message
    from typesystem.exceptions import ValidationError
    field = Array(unique_items=True)
    value = [1, 2, 3, 1]
    try:
        field.validate(value)
    except ValidationError as exc:
        assert len(exc.messages()) == 1
        message = exc.messages()[0]
        assert message.text == "Items must be unique."
        assert message.code == "unique_items"
        assert message.index == [3]
    else:
        assert False

def test_unique_items_duplicate_boolean_true():
    from typesystem.fields import Array
    from typesystem.base import Message
    from typesystem.exceptions import ValidationError
    field = Array(unique_items=True)
    value = [True, False, True]
    try:
        field.validate(value)
    except ValidationError as exc:
        assert len(exc.messages()) == 1
        message = exc.messages()[0]
        assert message.text == "Items must be unique."
        assert message.code == "unique_items"
        assert message.index == [2]
    else:
        assert False

def test_unique_items_duplicate_boolean_false():
    from typesystem.fields import Array
    from typesystem.base import Message
    from typesystem.exceptions import ValidationError
    field = Array(unique_items=True)
    value = [False, True, False]
    try:
        field.validate(value)
    except ValidationError as exc:
        assert len(exc.messages()) == 1
        message = exc.messages()[0]
        assert message.text == "Items must be unique."
        assert message.code == "unique_items"
        assert message.index == [2]
    else:
        assert False

def test_unique_items_duplicate_list():
    from typesystem.fields import Array
    from typesystem.base import Message
    from typesystem.exceptions import ValidationError
    field = Array(unique_items=True)
    value = [[1, 2], [3, 4], [1, 2]]
    try:
        field.validate(value)
    except ValidationError as exc:
        assert len(exc.messages()) == 1
        message = exc.messages()[0]
        assert message.text == "Items must be unique."
        assert message.code == "unique_items"
        assert message.index == [2]
    else:
        assert False

def test_unique_items_duplicate_dict():
    from typesystem.fields import Array
    from typesystem.base import Message
    from typesystem.exceptions import ValidationError
    field = Array(unique_items=True)
    value = [{"a": 1}, {"b": 2}, {"a": 1}]
    try:
        field.validate(value)
    except ValidationError as exc:
        assert len(exc.messages()) == 1
        message = exc.messages()[0]
        assert message.text == "Items must be unique."
        assert message.code == "unique_items"
        assert message.index == [2]
    else:
        assert False

def test_unique_items_duplicate_with_validator():
    from typesystem.fields import Array, Integer
    from typesystem.base import Message
    from typesystem.exceptions import ValidationError
    field = Array(items=Integer(), unique_items=True)
    value = [1, 2, 3, 1]
    try:
        field.validate(value)
    except ValidationError as exc:
        assert len(exc.messages()) == 1
        message = exc.messages()[0]
        assert message.text == "Items must be unique."
        assert message.code == "unique_items"
        assert message.index == [3]
    else:
        assert False

def test_unique_items_duplicate_with_list_validators():
    from typesystem.fields import Array, Integer, String
    from typesystem.base import Message
    from typesystem.exceptions import ValidationError
    field = Array(items=[Integer(), String()], unique_items=True)
    value = [1, "hello", 1]
    try:
        field.validate(value)
    except ValidationError as exc:
        assert len(exc.messages()) == 1
        message = exc.messages()[0]
        assert message.text == "Items must be unique."
        assert message.code == "unique_items"
        assert message.index == [2]
    else:
        assert False

def test_unique_items_duplicate_additional_items():
    from typesystem.fields import Array, Integer, String
    from typesystem.base import Message
    from typesystem.exceptions import ValidationError
    field = Array(items=[Integer()], additional_items=String(), unique_items=True)
    value = [1, "hello", "world", "hello"]
    try:
        field.validate(value)
    except ValidationError as exc:
        assert len(exc.messages()) == 1
        message = exc.messages()[0]
        assert message.text == "Items must be unique."
        assert message.code == "unique_items"
        assert message.index == [3]
    else:
        assert False


