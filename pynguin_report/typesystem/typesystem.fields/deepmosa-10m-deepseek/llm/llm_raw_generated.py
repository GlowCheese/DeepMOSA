####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_validate_union_with_null_allowed():
    field1 = Integer()
    field2 = String(allow_null=True)
    union = Union(any_of=[field1, field2])
    result = union.validate(None)
    assert result is None

def test_validate_union_with_null_not_allowed():
    field1 = Integer()
    field2 = String()
    union = Union(any_of=[field1, field2])
    try:
        union.validate(None)
        assert False
    except ValidationError as e:
        assert e.messages()[0].code == "null"

def test_validate_union_matches_first_field():
    field1 = Integer()
    field2 = String()
    union = Union(any_of=[field1, field2])
    result = union.validate(42)
    assert result == 42

def test_validate_union_matches_second_field():
    field1 = Integer()
    field2 = String()
    union = Union(any_of=[field1, field2])
    result = union.validate("hello")
    assert result == "hello"

def test_validate_union_no_match_raises_union_error():
    field1 = Integer()
    field2 = String()
    union = Union(any_of=[field1, field2])
    try:
        union.validate(3.14)
        assert False
    except ValidationError as e:
        assert e.messages()[0].code == "union"

def test_validate_union_single_candidate_error_raised():
    field1 = Integer(min_value=10)
    field2 = String()
    union = Union(any_of=[field1, field2])
    try:
        union.validate(5)
        assert False
    except ValidationError as e:
        assert e.messages()[0].code == "min_value"

def test_validate_union_multiple_candidate_errors_raises_union_error():
    field1 = Integer(min_value=10)
    field2 = String(min_length=5)
    union = Union(any_of=[field1, field2])
    try:
        union.validate(3)
        assert False
    except ValidationError as e:
        assert e.messages()[0].code == "union"


# LLM-generated content at query #2
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
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "null"

def test_validate_non_dict_type():
    field = Object()
    try:
        field.validate("not a dict")
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "type"

def test_validate_invalid_key_type():
    field = Object()
    value = {123: "value"}
    try:
        field.validate(value)
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "invalid_key"

def test_validate_property_names_invalid():
    property_names_field = String(max_length=5)
    field = Object(property_names=property_names_field)
    value = {"toolongkey": "value"}
    try:
        field.validate(value)
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "invalid_property"

def test_validate_min_properties_empty():
    field = Object(min_properties=1)
    value = {}
    try:
        field.validate(value)
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "empty"

def test_validate_min_properties():
    field = Object(min_properties=2)
    value = {"a": 1}
    try:
        field.validate(value)
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "min_properties"

def test_validate_max_properties():
    field = Object(max_properties=1)
    value = {"a": 1, "b": 2}
    try:
        field.validate(value)
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "max_properties"

def test_validate_required_missing():
    field = Object(required=["key"])
    value = {}
    try:
        field.validate(value)
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
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "type"

def test_validate_pattern_properties():
    child_field = Integer()
    field = Object(pattern_properties={r"^num_": child_field})
    value = {"num_1": 42, "num_2": 100}
    result = field.validate(value)
    assert result == {"num_1": 42, "num_2": 100}

def test_validate_pattern_properties_invalid():
    child_field = Integer()
    field = Object(pattern_properties={r"^num_": child_field})
    value = {"num_1": "not an integer"}
    try:
        field.validate(value)
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "type"

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
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "invalid_property"

def test_validate_additional_properties_field():
    additional_field = Integer()
    field = Object(additional_properties=additional_field)
    value = {"extra": 42}
    result = field.validate(value)
    assert result == {"extra": 42}

def test_validate_additional_properties_field_invalid():
    additional_field = Integer()
    field = Object(additional_properties=additional_field)
    value = {"extra": "not an integer"}
    try:
        field.validate(value)
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "type"

def test_validate_multiple_errors():
    field = Object(required=["req"], min_properties=2)
    value = {}
    try:
        field.validate(value)
    except ValidationError as e:
        assert len(e.messages()) == 2
        codes = {msg.code for msg in e.messages()}
        assert codes == {"empty", "required"}


# LLM-generated content at query #3
#--------------------------

def test_validate_null_when_allow_null():
    field = String(allow_null=True)
    result = field.validate(None)
    assert result is None

def test_validate_null_when_allow_blank_and_coerce_types():
    field = String(allow_blank=True, coerce_types=True)
    result = field.validate(None)
    assert result == ""

def test_validate_null_raises_error():
    field = String()
    try:
        field.validate(None)
        assert False
    except Exception as e:
        assert str(e) == "May not be null."

def test_validate_non_string_raises_type_error():
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

def test_validate_trims_whitespace():
    field = String(trim_whitespace=True)
    result = field.validate("  hello  ")
    assert result == "hello"

def test_validate_does_not_trim_whitespace():
    field = String(trim_whitespace=False)
    result = field.validate("  hello  ")
    assert result == "  hello  "

def test_validate_empty_string_raises_blank_error():
    field = String()
    try:
        field.validate("")
        assert False
    except Exception as e:
        assert str(e) == "Must not be blank."

def test_validate_empty_string_with_allow_blank():
    field = String(allow_blank=True)
    result = field.validate("")
    assert result == ""

def test_validate_blank_string_with_allow_null_and_coerce_types():
    field = String(allow_null=True, coerce_types=True)
    result = field.validate("")
    assert result is None

def test_validate_min_length_error():
    field = String(min_length=5)
    try:
        field.validate("hi")
        assert False
    except Exception as e:
        assert str(e) == "Must have at least 5 characters."

def test_validate_max_length_error():
    field = String(max_length=5)
    try:
        field.validate("hello world")
        assert False
    except Exception as e:
        assert str(e) == "Must have no more than 5 characters."

def test_validate_pattern_error():
    field = String(pattern="^[a-z]+$")
    try:
        field.validate("Hello123")
        assert False
    except Exception as e:
        assert str(e) == "Must match the pattern /^[a-z]+$/."

def test_validate_pattern_success():
    field = String(pattern="^[a-z]+$")
    result = field.validate("hello")
    assert result == "hello"

def test_validate_format():
    field = String(format="email")
    result = field.validate("test@example.com")
    assert result == "test@example.com"

def test_validate_format_error():
    field = String(format="email")
    try:
        field.validate("invalid-email")
        assert False
    except Exception as e:
        assert "Must be a valid email." in str(e)


# LLM-generated content at query #4
#--------------------------

def test_validate_null_with_allow_null():
    from typesystem.fields import Object
    field = Object(allow_null=True)
    result = field.validate(None)
    assert result is None

def test_validate_null_without_allow_null():
    from typesystem.fields import Object
    from typesystem.exceptions import ValidationError
    field = Object(allow_null=False)
    try:
        field.validate(None)
        assert False
    except ValidationError as exc:
        assert len(exc.messages()) == 1
        assert exc.messages()[0].code == "null"

def test_validate_non_dict_type():
    from typesystem.fields import Object
    from typesystem.exceptions import ValidationError
    field = Object()
    try:
        field.validate("not a dict")
        assert False
    except ValidationError as exc:
        assert len(exc.messages()) == 1
        assert exc.messages()[0].code == "type"

def test_validate_invalid_key_non_string():
    from typesystem.fields import Object
    from typesystem.exceptions import ValidationError
    field = Object()
    try:
        field.validate({1: "value"})
        assert False
    except ValidationError as exc:
        assert len(exc.messages()) == 1
        assert exc.messages()[0].code == "invalid_key"

def test_validate_property_names_invalid():
    from typesystem.fields import Object, String
    from typesystem.exceptions import ValidationError
    property_names = String(max_length=3)
    field = Object(property_names=property_names)
    try:
        field.validate({"longkey": "value"})
        assert False
    except ValidationError as exc:
        assert len(exc.messages()) == 1
        assert exc.messages()[0].code == "invalid_property"

def test_validate_min_properties_empty():
    from typesystem.fields import Object
    from typesystem.exceptions import ValidationError
    field = Object(min_properties=1)
    try:
        field.validate({})
        assert False
    except ValidationError as exc:
        assert len(exc.messages()) == 1
        assert exc.messages()[0].code == "empty"

def test_validate_min_properties():
    from typesystem.fields import Object
    from typesystem.exceptions import ValidationError
    field = Object(min_properties=2)
    try:
        field.validate({"a": 1})
        assert False
    except ValidationError as exc:
        assert len(exc.messages()) == 1
        assert exc.messages()[0].code == "min_properties"

def test_validate_max_properties():
    from typesystem.fields import Object
    from typesystem.exceptions import ValidationError
    field = Object(max_properties=2)
    try:
        field.validate({"a": 1, "b": 2, "c": 3})
        assert False
    except ValidationError as exc:
        assert len(exc.messages()) == 1
        assert exc.messages()[0].code == "max_properties"

def test_validate_required_missing():
    from typesystem.fields import Object
    from typesystem.exceptions import ValidationError
    field = Object(required=["key1"])
    try:
        field.validate({})
        assert False
    except ValidationError as exc:
        assert len(exc.messages()) == 1
        assert exc.messages()[0].code == "required"

def test_validate_properties_with_default():
    from typesystem.fields import Object, String
    field = Object(properties={"key": String(default="default_value")})
    result = field.validate({})
    assert result == {"key": "default_value"}

def test_validate_properties_valid():
    from typesystem.fields import Object, Integer
    field = Object(properties={"age": Integer()})
    result = field.validate({"age": 25})
    assert result == {"age": 25}

def test_validate_properties_invalid():
    from typesystem.fields import Object, Integer
    from typesystem.exceptions import ValidationError
    field = Object(properties={"age": Integer()})
    try:
        field.validate({"age": "not an integer"})
        assert False
    except ValidationError as exc:
        assert len(exc.messages()) == 1
        assert exc.messages()[0].code == "type"

def test_validate_pattern_properties():
    from typesystem.fields import Object, Integer
    import re
    field = Object(pattern_properties={r"^x_": Integer()})
    result = field.validate({"x_1": 10, "x_2": 20})
    assert result == {"x_1": 10, "x_2": 20}

def test_validate_additional_properties_true():
    from typesystem.fields import Object
    field = Object(additional_properties=True)
    result = field.validate({"extra": "value"})
    assert result == {"extra": "value"}

def test_validate_additional_properties_false():
    from typesystem.fields import Object
    from typesystem.exceptions import ValidationError
    field = Object(additional_properties=False)
    try:
        field.validate({"extra": "value"})
        assert False
    except ValidationError as exc:
        assert len(exc.messages()) == 1
        assert exc.messages()[0].code == "invalid_property"

def test_validate_additional_properties_field():
    from typesystem.fields import Object, String
    from typesystem.exceptions import ValidationError
    field = Object(additional_properties=String(max_length=5))
    try:
        field.validate({"extra": "too long"})
        assert False
    except ValidationError as exc:
        assert len(exc.messages()) == 1
        assert exc.messages()[0].code == "max_length"

def test_validate_combined_errors():
    from typesystem.fields import Object, Integer
    from typesystem.exceptions import ValidationError
    field = Object(properties={"age": Integer()}, required=["name"])
    try:
        field.validate({"age": "invalid"})
        assert False
    except ValidationError as exc:
        assert len(exc.messages()) == 2
        codes = {msg.code for msg in exc.messages()}
        assert "type" in codes
        assert "required" in codes


# LLM-generated content at query #5
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


# LLM-generated content at query #6
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
        assert e.detail == "May not be null."

def test_validate_returns_true_for_bool_true():
    field = Boolean()
    result = field.validate(True)
    assert result is True

def test_validate_returns_false_for_bool_false():
    field = Boolean()
    result = field.validate(False)
    assert result is False

def test_validate_coerces_string_true_to_bool():
    field = Boolean(coerce_types=True)
    result = field.validate("true")
    assert result is True

def test_validate_coerces_string_false_to_bool():
    field = Boolean(coerce_types=True)
    result = field.validate("false")
    assert result is False

def test_validate_coerces_string_on_to_bool():
    field = Boolean(coerce_types=True)
    result = field.validate("on")
    assert result is True

def test_validate_coerces_string_off_to_bool():
    field = Boolean(coerce_types=True)
    result = field.validate("off")
    assert result is False

def test_validate_coerces_string_1_to_bool():
    field = Boolean(coerce_types=True)
    result = field.validate("1")
    assert result is True

def test_validate_coerces_string_0_to_bool():
    field = Boolean(coerce_types=True)
    result = field.validate("0")
    assert result is False

def test_validate_coerces_empty_string_to_false():
    field = Boolean(coerce_types=True)
    result = field.validate("")
    assert result is False

def test_validate_coerces_int_1_to_bool():
    field = Boolean(coerce_types=True)
    result = field.validate(1)
    assert result is True

def test_validate_coerces_int_0_to_bool():
    field = Boolean(coerce_types=True)
    result = field.validate(0)
    assert result is False

def test_validate_coerces_null_string_to_none_when_allow_null():
    field = Boolean(coerce_types=True, allow_null=True)
    result = field.validate("null")
    assert result is None

def test_validate_coerces_none_string_to_none_when_allow_null():
    field = Boolean(coerce_types=True, allow_null=True)
    result = field.validate("none")
    assert result is None

def test_validate_coerces_empty_string_to_none_when_allow_null():
    field = Boolean(coerce_types=True, allow_null=True)
    result = field.validate("")
    assert result is None

def test_validate_raises_type_error_for_invalid_string_when_coerce_types():
    field = Boolean(coerce_types=True)
    try:
        field.validate("invalid")
        assert False
    except Exception as e:
        assert e.detail == "Must be a boolean."

def test_validate_raises_type_error_for_invalid_type_when_coerce_types():
    field = Boolean(coerce_types=True)
    try:
        field.validate([])
        assert False
    except Exception as e:
        assert e.detail == "Must be a boolean."

def test_validate_raises_type_error_for_non_bool_when_coerce_types_false():
    field = Boolean(coerce_types=False)
    try:
        field.validate("true")
        assert False
    except Exception as e:
        assert e.detail == "Must be a boolean."

def test_validate_raises_type_error_for_int_when_coerce_types_false():
    field = Boolean(coerce_types=False)
    try:
        field.validate(1)
        assert False
    except Exception as e:
        assert e.detail == "Must be a boolean."


# LLM-generated content at query #7
#--------------------------

def test_validate_null_with_allow_null():
    field = Array(allow_null=True)
    result = field.validate(None)
    assert result is None

def test_validate_null_without_allow_null():
    field = Array()
    try:
        field.validate(None)
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "null"

def test_validate_non_list_type():
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

def test_validate_with_items_field():
    item_field = Field(allow_null=True)
    field = Array(items=item_field)
    result = field.validate([None, 5])
    assert result == [None, 5]

def test_validate_with_items_list():
    item_field1 = Field()
    item_field2 = Field(allow_null=True)
    field = Array(items=[item_field1, item_field2])
    result = field.validate([10, None])
    assert result == [10, None]

def test_validate_with_items_list_and_additional_items_false():
    item_field = Field()
    field = Array(items=[item_field], additional_items=False)
    try:
        field.validate([1, 2])
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "additional_items"

def test_validate_with_items_list_and_additional_items_field():
    item_field = Field()
    additional_field = Field(allow_null=True)
    field = Array(items=[item_field], additional_items=additional_field)
    result = field.validate([1, None, 3])
    assert result == [1, None, 3]

def test_validate_unique_items():
    field = Array(unique_items=True)
    try:
        field.validate([1, 2, 1])
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "unique_items"

def test_validate_unique_items_with_false_and_true():
    field = Array(unique_items=True)
    result = field.validate([False, True])
    assert result == [False, True]

def test_validate_unique_items_with_0_and_false():
    field = Array(unique_items=True)
    result = field.validate([0, False])
    assert result == [0, False]

def test_validate_unique_items_with_1_and_true():
    field = Array(unique_items=True)
    result = field.validate([1, True])
    assert result == [1, True]

def test_validate_unique_items_with_nested_lists():
    field = Array(unique_items=True)
    try:
        field.validate([[1, 2], [1, 2]])
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "unique_items"

def test_validate_unique_items_with_nested_dicts():
    field = Array(unique_items=True)
    try:
        field.validate([{"a": 1}, {"a": 1}])
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "unique_items"

def test_validate_with_item_validation_error():
    item_field = Field(minimum=10)
    field = Array(items=item_field)
    try:
        field.validate([5, 15])
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "minimum"
        assert e.messages()[0].index == [0]

def test_validate_with_multiple_item_validation_errors():
    item_field = Field(minimum=10)
    field = Array(items=item_field)
    try:
        field.validate([5, 7])
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 2
        assert e.messages()[0].code == "minimum"
        assert e.messages()[0].index == [0]
        assert e.messages()[1].code == "minimum"
        assert e.messages()[1].index == [1]

def test_validate_with_items_list_and_item_validation_error():
    item_field1 = Field(minimum=10)
    item_field2 = Field(maximum=5)
    field = Array(items=[item_field1, item_field2])
    try:
        field.validate([5, 10])
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 2
        assert e.messages()[0].code == "minimum"
        assert e.messages()[0].index == [0]
        assert e.messages()[1].code == "maximum"
        assert e.messages()[1].index == [1]

def test_validate_with_additional_items_field_validation_error():
    item_field = Field()
    additional_field = Field(minimum=10)
    field = Array(items=[item_field], additional_items=additional_field)
    try:
        field.validate([1, 5])
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "minimum"
        assert e.messages()[0].index == [1]

def test_validate_combined_unique_and_item_validation_error():
    item_field = Field(minimum=10)
    field = Array(items=item_field, unique_items=True)
    try:
        field.validate([5, 5])
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 2
        codes = {msg.code for msg in e.messages()}
        assert codes == {"minimum", "unique_items"}

def test_validate_success_with_no_constraints():
    field = Array()
    result = field.validate([1, "two", True, None])
    assert result == [1, "two", True, None]

def test_validate_success_with_items_field():
    item_field = Field()
    field = Array(items=item_field)
    result = field.validate([1, 2, 3])
    assert result == [1, 2, 3]

def test_validate_success_with_items_list_exact():
    item_field1 = Field()
    item_field2 = Field(allow_null=True)
    field = Array(items=[item_field1, item_field2])
    result = field.validate([10, None])
    assert result == [10, None]

def test_validate_success_with_items_list_and_additional_items_field():
    item_field = Field()
    additional_field = Field()
    field = Array(items=[item_field], additional_items=additional_field)
    result = field.validate([1, 2, 3])
    assert result == [1, 2, 3]

def test_validate_success_with_unique_items():
    field = Array(unique_items=True)
    result = field.validate([1, 2, 3])
    assert result == [1, 2, 3]


# LLM-generated content at query #8
#--------------------------

def test_allow_null_false_and_value_not_in_coerce_null_values():
    field = Boolean(allow_null=False)
    result = field.validate("null")
    assert result is False


# LLM-generated content at query #9
#--------------------------

```python
def test_additional_properties_is_none_does_not_enter_assert_block():
    from typesystem.fields import Object, String
    field = Object(additional_properties=None)
    result = field.validate({"extra": "value"})
    assert result == {}


# LLM-generated content at query #10
#--------------------------

```python
def test_validate_with_non_null_value_and_allow_null_false():
    from typesystem.fields import Object
    field = Object(allow_null=False)
    value = {"key": "value"}
    result = field.validate(value)
    assert result == {"key": "value"}


# LLM-generated content at query #11
#--------------------------

def test_validate_null_when_not_allowed():
    field = Number()
    try:
        field.validate(None)
        assert False
    except Exception as e:
        assert str(e) == "May not be null."

def test_validate_null_when_allowed():
    field = Number(allow_null=True)
    result = field.validate(None)
    assert result is None

def test_validate_empty_string_coerces_to_null_when_allowed():
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

def test_validate_string_coerces_to_number():
    field = Number(coerce_types=True)
    result = field.validate("123")
    assert result == 123

def test_validate_string_fails_when_coerce_false():
    field = Number(coerce_types=False)
    try:
        field.validate("123")
        assert False
    except Exception as e:
        assert str(e) == "Must be a number."

def test_validate_float_for_integer_field_raises_integer_error():
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
    field = Number(precision="0.01")
    result = field.validate(1.234)
    assert result == 1.23

def test_validate_valid_integer_with_minimum_and_maximum():
    field = Number(minimum=5, maximum=15)
    result = field.validate(10)
    assert result == 10

def test_validate_valid_float_with_exclusive_bounds():
    field = Number(exclusive_minimum=5.0, exclusive_maximum=15.0)
    result = field.validate(10.0)
    assert result == 10.0

def test_validate_valid_number_with_multiple_of():
    field = Number(multiple_of=3)
    result = field.validate(9)
    assert result == 9

def test_validate_coerce_string_to_float():
    field = Number(coerce_types=True)
    result = field.validate("3.14")
    assert result == 3.14

def test_validate_coerce_decimal_string():
    field = Number(coerce_types=True)
    result = field.validate("2.5")
    assert result == 2.5

def test_validate_invalid_string_raises_type_error():
    field = Number(coerce_types=True)
    try:
        field.validate("abc")
        assert False
    except Exception as e:
        assert str(e) == "Must be a number."


# LLM-generated content at query #12
#--------------------------

```python
def test_unique_items_with_duplicate_primitive_values():
    from typesystem.fields import Array
    from typesystem.base import Message
    from typesystem.exceptions import ValidationError
    field = Array(unique_items=True)
    value = [1, 2, 3, 1]
    try:
        result = field.validate(value)
    except ValidationError as e:
        error_messages = e.messages()
        assert len(error_messages) == 1
        assert error_messages[0].code == "unique_items"
        assert error_messages[0].index == [3]
    else:
        assert False, "Expected ValidationError"

def test_unique_items_with_duplicate_lists():
    from typesystem.fields import Array
    from typesystem.base import Message
    from typesystem.exceptions import ValidationError
    field = Array(unique_items=True)
    value = [[1, 2], [3, 4], [1, 2]]
    try:
        result = field.validate(value)
    except ValidationError as e:
        error_messages = e.messages()
        assert len(error_messages) == 1
        assert error_messages[0].code == "unique_items"
        assert error_messages[0].index == [2]
    else:
        assert False, "Expected ValidationError"

def test_unique_items_with_duplicate_dicts():
    from typesystem.fields import Array
    from typesystem.base import Message
    from typesystem.exceptions import ValidationError
    field = Array(unique_items=True)
    value = [{"a": 1}, {"b": 2}, {"a": 1}]
    try:
        result = field.validate(value)
    except ValidationError as e:
        error_messages = e.messages()
        assert len(error_messages) == 1
        assert error_messages[0].code == "unique_items"
        assert error_messages[0].index == [2]
    else:
        assert False, "Expected ValidationError"

def test_unique_items_with_duplicate_boolean_and_integer():
    from typesystem.fields import Array
    from typesystem.base import Message
    from typesystem.exceptions import ValidationError
    field = Array(unique_items=True)
    value = [True, 1, False, 0]
    try:
        result = field.validate(value)
    except ValidationError as e:
        error_messages = e.messages()
        assert len(error_messages) == 0
    else:
        assert True

def test_unique_items_with_duplicate_strings():
    from typesystem.fields import Array
    from typesystem.base import Message
    from typesystem.exceptions import ValidationError
    field = Array(unique_items=True)
    value = ["a", "b", "a"]
    try:
        result = field.validate(value)
    except ValidationError as e:
        error_messages = e.messages()
        assert len(error_messages) == 1
        assert error_messages[0].code == "unique_items"
        assert error_messages[0].index == [2]
    else:
        assert False, "Expected ValidationError"

def test_unique_items_with_duplicate_floats():
    from typesystem.fields import Array
    from typesystem.base import Message
    from typesystem.exceptions import ValidationError
    field = Array(unique_items=True)
    value = [1.5, 2.5, 1.5]
    try:
        result = field.validate(value)
    except ValidationError as e:
        error_messages = e.messages()
        assert len(error_messages) == 1
        assert error_messages[0].code == "unique_items"
        assert error_messages[0].index == [2]
    else:
        assert False, "Expected ValidationError"

def test_unique_items_with_duplicate_none():
    from typesystem.fields import Array
    from typesystem.base import Message
    from typesystem.exceptions import ValidationError
    field = Array(unique_items=True)
    value = [None, "a", None]
    try:
        result = field.validate(value)
    except ValidationError as e:
        error_messages = e.messages()
        assert len(error_messages) == 1
        assert error_messages[0].code == "unique_items"
        assert error_messages[0].index == [2]
    else:
        assert False, "Expected ValidationError"


# LLM-generated content at query #13
#--------------------------

def test_condition_true_when_multiple_error_messages():
    from myapp.fields import Union, Field
    class MockField(Field):
        def validate_or_error(self, value):
            class MockError:
                def messages(self):
                    return [type('obj', (object,), {'code': 'type', 'index': None})(), type('obj', (object,), {'code': 'required', 'index': None})()]
            return None, MockError()
    field = Union(any_of=[MockField()])
    field.validate(value=5)

def test_condition_true_when_single_message_code_not_type():
    from myapp.fields import Union, Field
    class MockField(Field):
        def validate_or_error(self, value):
            class MockError:
                def messages(self):
                    return [type('obj', (object,), {'code': 'max_length', 'index': None})()]
            return None, MockError()
    field = Union(any_of=[MockField()])
    field.validate(value=5)

def test_condition_true_when_single_type_message_with_index():
    from myapp.fields import Union, Field
    class MockField(Field):
        def validate_or_error(self, value):
            class MockError:
                def messages(self):
                    return [type('obj', (object,), {'code': 'type', 'index': [0]})()]
            return None, MockError()
    field = Union(any_of=[MockField()])
    field.validate(value=5)


# LLM-generated content at query #14
#--------------------------

```python
def test_unique_items_with_duplicate_primitives():
    from typesystem.fields import Array
    from typesystem.base import Message
    from typesystem import ValidationError

    field = Array(unique_items=True)
    value = [1, 2, 1]
    try:
        field.validate(value)
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "unique_items"
        assert e.messages()[0].index == [2]
    else:
        assert False, "Expected ValidationError"

def test_unique_items_with_duplicate_lists():
    from typesystem.fields import Array
    from typesystem.base import Message
    from typesystem import ValidationError

    field = Array(unique_items=True)
    value = [[1, 2], [3, 4], [1, 2]]
    try:
        field.validate(value)
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "unique_items"
        assert e.messages()[0].index == [2]
    else:
        assert False, "Expected ValidationError"

def test_unique_items_with_duplicate_dicts():
    from typesystem.fields import Array
    from typesystem.base import Message
    from typesystem import ValidationError

    field = Array(unique_items=True)
    value = [{"a": 1}, {"b": 2}, {"a": 1}]
    try:
        field.validate(value)
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "unique_items"
        assert e.messages()[0].index == [2]
    else:
        assert False, "Expected ValidationError"

def test_unique_items_with_duplicate_booleans():
    from typesystem.fields import Array
    from typesystem.base import Message
    from typesystem import ValidationError

    field = Array(unique_items=True)
    value = [True, False, True]
    try:
        field.validate(value)
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "unique_items"
        assert e.messages()[0].index == [2]
    else:
        assert False, "Expected ValidationError"

def test_unique_items_with_mixed_types():
    from typesystem.fields import Array
    from typesystem.base import Message
    from typesystem import ValidationError

    field = Array(unique_items=True)
    value = [1, True, 1]
    try:
        field.validate(value)
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "unique_items"
        assert e.messages()[0].index == [2]
    else:
        assert False, "Expected ValidationError"

def test_unique_items_with_duplicate_nested_structures():
    from typesystem.fields import Array
    from typesystem.base import Message
    from typesystem import ValidationError

    field = Array(unique_items=True)
    value = [[{"a": [1, 2]}], [{"b": [3, 4]}], [{"a": [1, 2]}]]
    try:
        field.validate(value)
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "unique_items"
        assert e.messages()[0].index == [2]
    else:
        assert False, "Expected ValidationError"


# LLM-generated content at query #15
#--------------------------

def test_validate_null_allowed():
    field = Object(allow_null=True)
    result = field.validate(None)
    assert result is None

def test_validate_null_not_allowed():
    field = Object(allow_null=False)
    try:
        field.validate(None)
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "null"

def test_validate_non_dict():
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

def test_validate_pattern_properties_matching():
    child_field = Field()
    field = Object(pattern_properties={"^a.*": child_field})
    result = field.validate({"abc": "value"})
    assert result == {"abc": "value"}

def test_validate_pattern_properties_non_matching():
    child_field = Field()
    field = Object(pattern_properties={"^a.*": child_field})
    result = field.validate({"bcd": "value"})
    assert result == {"bcd": "value"}

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

def test_validate_combined_errors():
    child_field = Field(allow_null=False)
    field = Object(properties={"key": child_field}, required=["missing"])
    try:
        field.validate({"key": None})
        assert False
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 2
        codes = {msg.code for msg in messages}
        assert codes == {"null", "required"}


# LLM-generated content at query #16
#--------------------------

def test_validate_with_numeric_type_int_and_float_integer():
    field = Number(numeric_type=int)
    result = field.validate(5.0)
    assert result == 5

def test_validate_with_numeric_type_none_and_string_value():
    field = Number()
    result = field.validate("123")
    assert result == 123

def test_validate_with_numeric_type_int_and_integer_value():
    field = Number(numeric_type=int)
    result = field.validate(42)
    assert result == 42

def test_validate_with_numeric_type_none_and_integer_value():
    field = Number()
    result = field.validate(42)
    assert result == 42

def test_validate_with_numeric_type_none_and_float_value():
    field = Number()
    result = field.validate(3.14)
    assert result == 3.14

def test_validate_with_numeric_type_int_and_string_integer():
    field = Number(numeric_type=int)
    result = field.validate("42")
    assert result == 42


# LLM-generated content at query #17
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

def test_validate_unique_items_violation():
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

def test_validate_item_validation_error():
    item_field = Field()
    item_field.validate = lambda x: (_ for _ in ()).throw(ValidationError(messages=[Message(text="Invalid", code="invalid")]))
    field = Array(items=item_field)
    try:
        field.validate([1])
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "invalid"
        assert e.messages()[0].index == [0]

def test_validate_multiple_item_validation_errors():
    item_field = Field()
    item_field.validate = lambda x: (_ for _ in ()).throw(ValidationError(messages=[Message(text="Invalid", code="invalid")]))
    field = Array(items=[item_field, item_field])
    try:
        field.validate([1, 2])
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 2
        assert e.messages()[0].code == "invalid"
        assert e.messages()[0].index == [0]
        assert e.messages()[1].code == "invalid"
        assert e.messages()[1].index == [1]

def test_validate_combined_errors():
    item_field = Field()
    item_field.validate = lambda x: (_ for _ in ()).throw(ValidationError(messages=[Message(text="Invalid", code="invalid")]))
    field = Array(items=item_field, unique_items=True, min_items=2)
    try:
        field.validate([1])
        assert False
    except ValidationError as e:
        codes = [msg.code for msg in e.messages()]
        assert "min_items" in codes
        assert "invalid" in codes

def test_validate_no_item_validator():
    field = Array()
    result = field.validate([1, "two", {"three": 3}])
    assert result == [1, "two", {"three": 3}]

def test_validate_unique_items_with_complex_types():
    field = Array(unique_items=True)
    result = field.validate([[1, 2], [3, 4]])
    assert result == [[1, 2], [3, 4]]

def test_validate_unique_items_with_duplicate_complex_types():
    field = Array(unique_items=True)
    try:
        field.validate([[1, 2], [1, 2]])
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "unique_items"


# LLM-generated content at query #18
#--------------------------

```python
def test_validate_with_non_null_value_and_allow_null_false():
    from typesystem.fields import Object
    from typesystem import Integer
    field = Object(properties={"age": Integer()}, allow_null=False)
    value = {"age": 25}
    result = field.validate(value)
    assert result == {"age": 25}


# LLM-generated content at query #19
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
    assert not hasattr(field, 'default')

def test_choice_constructor_with_allow_null_and_default():
    field = Choice(allow_null=True)
    assert field.allow_null == True


# LLM-generated content at query #20
#--------------------------

```python
def test_validate_with_non_none_value_and_allow_null_false():
    from typesystem.fields import Object
    field = Object(allow_null=False)
    value = {"key": "value"}
    result = field.validate(value)
    assert result == {"key": "value"}


# LLM-generated content at query #21
#--------------------------

def test_validate_allow_null():
    field = Number(allow_null=True)
    result = field.validate(None)
    assert result is None

def test_validate_null_error():
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

def test_validate_bool_type_error():
    field = Number()
    try:
        field.validate(True)
        assert False
    except Exception as e:
        assert str(e) == "Must be a number."

def test_validate_integer_type_error_for_float():
    field = Number(numeric_type=int)
    try:
        field.validate(3.14)
        assert False
    except Exception as e:
        assert str(e) == "Must be an integer."

def test_validate_type_error_without_coercion():
    field = Number(coerce_types=False)
    try:
        field.validate("123")
        assert False
    except Exception as e:
        assert str(e) == "Must be a number."

def test_validate_string_coercion():
    field = Number()
    result = field.validate("123")
    assert result == 123

def test_validate_finite_error_for_inf():
    field = Number()
    try:
        field.validate(float('inf'))
        assert False
    except Exception as e:
        assert str(e) == "Must be finite."

def test_validate_precision_rounding():
    field = Number(precision="0.01")
    result = field.validate(3.14159)
    assert result == 3.14

def test_validate_minimum_error():
    field = Number(minimum=10)
    try:
        field.validate(5)
        assert False
    except Exception as e:
        assert str(e) == "Must be greater than or equal to 10."

def test_validate_exclusive_minimum_error():
    field = Number(exclusive_minimum=10)
    try:
        field.validate(10)
        assert False
    except Exception as e:
        assert str(e) == "Must be greater than 10."

def test_validate_maximum_error():
    field = Number(maximum=10)
    try:
        field.validate(15)
        assert False
    except Exception as e:
        assert str(e) == "Must be less than or equal to 10."

def test_validate_exclusive_maximum_error():
    field = Number(exclusive_maximum=10)
    try:
        field.validate(10)
        assert False
    except Exception as e:
        assert str(e) == "Must be less than 10."

def test_validate_multiple_of_int_error():
    field = Number(multiple_of=5)
    try:
        field.validate(12)
        assert False
    except Exception as e:
        assert str(e) == "Must be a multiple of 5."

def test_validate_multiple_of_float_error():
    field = Number(multiple_of=0.5)
    try:
        field.validate(1.2)
        assert False
    except Exception as e:
        assert str(e) == "Must be a multiple of 0.5."


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_serialize_with_none():
    decimal_instance = Decimal()
    result = decimal_instance.serialize(None)
    assert result is None

def test_serialize_with_decimal():
    decimal_instance = Decimal()
    decimal_obj = decimal.Decimal('123.456')
    result = decimal_instance.serialize(decimal_obj)
    assert result == 123.456

def test_serialize_with_integer_decimal():
    decimal_instance = Decimal()
    decimal_obj = decimal.Decimal('789')
    result = decimal_instance.serialize(decimal_obj)
    assert result == 789.0


# LLM-generated content at query #2
#--------------------------

def test_validate_null_allowed():
    field = Object(allow_null=True)
    result = field.validate(None)
    assert result is None

def test_validate_null_not_allowed():
    field = Object(allow_null=False)
    try:
        field.validate(None)
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "null"

def test_validate_not_dict():
    field = Object()
    try:
        field.validate("not a dict")
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "type"

def test_validate_invalid_key():
    field = Object()
    try:
        field.validate({1: "value"})
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "invalid_key"

def test_validate_property_names_invalid():
    property_names_field = String(max_length=3)
    field = Object(property_names=property_names_field)
    try:
        field.validate({"longkey": "value"})
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

def test_validate_min_properties():
    field = Object(min_properties=2)
    try:
        field.validate({"a": 1})
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "min_properties"

def test_validate_max_properties():
    field = Object(max_properties=2)
    try:
        field.validate({"a": 1, "b": 2, "c": 3})
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "max_properties"

def test_validate_required_missing():
    field = Object(required=["key1"])
    try:
        field.validate({})
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "required"

def test_validate_properties_with_default():
    child_field = String(default="default_value")
    field = Object(properties={"key": child_field})
    result = field.validate({})
    assert result == {"key": "default_value"}

def test_validate_properties_valid():
    child_field = String()
    field = Object(properties={"key": child_field})
    result = field.validate({"key": "value"})
    assert result == {"key": "value"}

def test_validate_properties_invalid():
    child_field = Integer()
    field = Object(properties={"key": child_field})
    try:
        field.validate({"key": "not an integer"})
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "type"

def test_validate_pattern_properties():
    child_field = Integer()
    field = Object(pattern_properties={"^a.*": child_field})
    result = field.validate({"abc": 123, "ade": 456})
    assert result == {"abc": 123, "ade": 456}

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

def test_validate_additional_properties_field():
    additional_field = Integer()
    field = Object(additional_properties=additional_field)
    result = field.validate({"extra": 123})
    assert result == {"extra": 123}

def test_validate_additional_properties_field_invalid():
    additional_field = Integer()
    field = Object(additional_properties=additional_field)
    try:
        field.validate({"extra": "not an integer"})
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "type"

def test_validate_multiple_errors():
    field = Object(required=["req"], property_names=String(max_length=2), additional_properties=False)
    try:
        field.validate({"longkey": "value"})
        assert False
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 2
        codes = {msg.code for msg in messages}
        assert "invalid_property" in codes
        assert "required" in codes


# LLM-generated content at query #3
#--------------------------

def test_validate_basic_object():
    field = Object()
    value = {"key": "value"}
    result = field.validate(value)
    assert result == value

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
    property_names_field = Field(allow_null=False)
    field = Object(property_names=property_names_field)
    value = {"valid": "ok", None: "invalid"}
    try:
        field.validate(value)
        assert False
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 2
        assert any(msg.code == "invalid_key" for msg in messages)
        assert any(msg.code == "invalid_property" for msg in messages)

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
    field = Object(max_properties=2)
    value = {"a": 1, "b": 2, "c": 3}
    try:
        field.validate(value)
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "max_properties"

def test_validate_required_missing():
    field = Object(required=["required_key"])
    value = {"other_key": "value"}
    try:
        field.validate(value)
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "required"

def test_validate_properties_with_default():
    child_field = Field(default="default_value")
    field = Object(properties={"key": child_field})
    value = {}
    result = field.validate(value)
    assert result == {"key": "default_value"}

def test_validate_properties_with_error():
    child_field = Field(allow_null=False)
    field = Object(properties={"key": child_field})
    value = {"key": None}
    try:
        field.validate(value)
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "null"

def test_validate_pattern_properties():
    child_field = Field(type="integer")
    field = Object(pattern_properties={"^a.*": child_field})
    value = {"apple": 5, "banana": "not int", "apricot": 3}
    try:
        field.validate(value)
        assert False
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 1
        assert messages[0].code == "type"

def test_validate_additional_properties_true():
    field = Object(additional_properties=True)
    value = {"extra": "allowed"}
    result = field.validate(value)
    assert result == value

def test_validate_additional_properties_false():
    field = Object(additional_properties=False)
    value = {"extra": "not allowed"}
    try:
        field.validate(value)
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "invalid_property"

def test_validate_additional_properties_field():
    additional_field = Field(type="integer")
    field = Object(additional_properties=additional_field)
    value = {"extra": "not an int"}
    try:
        field.validate(value)
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "type"

def test_validate_combined_errors():
    child_field = Field(allow_null=False)
    field = Object(properties={"key": child_field}, required=["required"], additional_properties=False)
    value = {"key": None, "extra": "invalid"}
    try:
        field.validate(value)
        assert False
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 3
        codes = {msg.code for msg in messages}
        assert "null" in codes
        assert "required" in codes
        assert "invalid_property" in codes


# LLM-generated content at query #4
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

def test_validate_non_dict():
    field = Object()
    try:
        field.validate("not a dict")
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "type"

def test_validate_invalid_key_non_string():
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

def test_validate_min_properties_violated():
    field = Object(min_properties=2)
    try:
        field.validate({"a": 1})
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "min_properties"

def test_validate_max_properties_violated():
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

def test_validate_pattern_properties_matching():
    child_field = Field()
    child_field.validate = lambda x: x
    field = Object(pattern_properties={"^a.*": child_field})
    result = field.validate({"abc": "value"})
    assert result == {"abc": "value"}

def test_validate_pattern_properties_non_matching():
    child_field = Field()
    child_field.validate = lambda x: x
    field = Object(pattern_properties={"^a.*": child_field})
    result = field.validate({"bcd": "value"})
    assert result == {"bcd": "value"}

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
    field = Object(required=["req"], min_properties=2)
    try:
        field.validate({"invalid": 123})
        assert False
    except ValidationError as e:
        codes = {msg.code for msg in e.messages()}
        assert "required" in codes
        assert "min_properties" in codes


# LLM-generated content at query #5
#--------------------------

def test_validate_null_allowed():
    field = String(allow_null=True)
    result = field.validate(None)
    assert result is None

def test_validate_null_not_allowed():
    field = String(allow_null=False)
    try:
        field.validate(None)
        assert False
    except Exception as e:
        assert str(e) == "May not be null."

def test_validate_null_coerced_to_blank():
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

def test_validate_string_with_null_char():
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

def test_validate_blank_not_allowed():
    field = String(allow_blank=False)
    try:
        field.validate("")
        assert False
    except Exception as e:
        assert str(e) == "Must not be blank."

def test_validate_blank_allowed():
    field = String(allow_blank=True)
    result = field.validate("")
    assert result == ""

def test_validate_blank_coerced_to_null():
    field = String(allow_null=True, coerce_types=True, trim_whitespace=True)
    result = field.validate("  ")
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

def test_validate_native_type_for_format():
    field = String(format="email")
    native_value = FORMATS["email"].native_type()
    result = field.validate(native_value)
    assert result == native_value


# LLM-generated content at query #6
#--------------------------

def test_get_default_value_returns_default_when_not_callable():
    field = Field(default="test_default")
    result = field.get_default_value()
    assert result == "test_default"

def test_get_default_value_returns_callable_result_when_default_is_callable():
    field = Field(default=lambda: "callable_result")
    result = field.get_default_value()
    assert result == "callable_result"

def test_get_default_value_returns_none_when_no_default_set():
    field = Field()
    result = field.get_default_value()
    assert result is None

def test_get_default_value_returns_none_when_default_is_none():
    field = Field(default=None)
    result = field.get_default_value()
    assert result is None

def test_get_default_value_handles_complex_callable():
    field = Field(default=lambda: {"key": "value"})
    result = field.get_default_value()
    assert result == {"key": "value"}

def test_get_default_value_with_allow_null_and_no_default():
    field = Field(allow_null=True)
    result = field.get_default_value()
    assert result is None


# LLM-generated content at query #7
#--------------------------

def test_validate_null_when_not_allowed():
    field = Number(allow_null=False)
    try:
        field.validate(None)
        assert False
    except Exception as e:
        assert str(e) == "May not be null."

def test_validate_null_when_allowed():
    field = Number(allow_null=True)
    result = field.validate(None)
    assert result is None

def test_validate_empty_string_when_null_allowed_and_coerce():
    field = Number(allow_null=True, coerce_types=True)
    result = field.validate("")
    assert result is None

def test_validate_empty_string_when_null_not_allowed():
    field = Number(allow_null=False, coerce_types=True)
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

def test_validate_float_for_integer_field():
    field = Number(numeric_type=int)
    try:
        field.validate(3.14)
        assert False
    except Exception as e:
        assert str(e) == "Must be an integer."

def test_validate_non_numeric_string_without_coercion():
    field = Number(coerce_types=False)
    try:
        field.validate("abc")
        assert False
    except Exception as e:
        assert str(e) == "Must be a number."

def test_validate_non_numeric_string_with_coercion():
    field = Number(coerce_types=True)
    try:
        field.validate("abc")
        assert False
    except Exception as e:
        assert str(e) == "Must be a number."

def test_validate_valid_numeric_string():
    field = Number()
    result = field.validate("123")
    assert result == 123

def test_validate_infinity():
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

def test_validate_with_precision():
    field = Number(precision="0.01")
    result = field.validate(3.14159)
    assert result == 3.14

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

def test_validate_coerce_to_int():
    field = Number(numeric_type=int)
    result = field.validate("42")
    assert result == 42

def test_validate_coerce_to_float():
    field = Number(numeric_type=float)
    result = field.validate("3.14")
    assert result == 3.14

def test_validate_valid_integer():
    field = Number()
    result = field.validate(42)
    assert result == 42

def test_validate_valid_float():
    field = Number()
    result = field.validate(3.14)
    assert result == 3.14


# LLM-generated content at query #8
#--------------------------

def test_validate_allows_null_when_allow_null_true():
    field = Choice(choices=[("a", "A"), ("b", "B")], allow_null=True)
    result = field.validate(None)
    assert result is None

def test_validate_raises_null_error_when_allow_null_false():
    field = Choice(choices=[("a", "A"), ("b", "B")], allow_null=False)
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

def test_validate_handles_empty_string_with_allow_null_and_coerce_types():
    field = Choice(choices=[("a", "A"), ("b", "B")], allow_null=True, coerce_types=True)
    result = field.validate("")
    assert result is None

def test_validate_raises_required_error_for_empty_string_without_allow_null():
    field = Choice(choices=[("a", "A"), ("b", "B")], allow_null=False)
    try:
        field.validate("")
        assert False
    except Exception as exc:
        assert str(exc) == "This field is required."

def test_validate_raises_choice_error_for_empty_string_with_allow_null_no_coerce():
    field = Choice(choices=[("a", "A"), ("b", "B")], allow_null=True, coerce_types=False)
    try:
        field.validate("")
        assert False
    except Exception as exc:
        assert str(exc) == "Not a valid choice."

def test_validate_works_with_choices_as_tuples():
    field = Choice(choices=[("key1", "Display 1"), ("key2", "Display 2")])
    result = field.validate("key2")
    assert result == "key2"

def test_validate_works_with_choices_as_strings():
    field = Choice(choices=["x", "y"])
    result = field.validate("y")
    assert result == "y"

def test_validate_distinguishes_true_from_one():
    field = Choice(choices=[(True, "Yes"), (1, "One")])
    result = field.validate(True)
    assert result is True

def test_validate_distinguishes_false_from_zero():
    field = Choice(choices=[(False, "No"), (0, "Zero")])
    result = field.validate(False)
    assert result is False

def test_validate_handles_list_as_choice_key():
    field = Choice(choices=[(["a", "b"], "List AB"), ("c", "C")])
    result = field.validate(["a", "b"])
    assert result == ["a", "b"]

def test_validate_handles_dict_as_choice_key():
    field = Choice(choices=[({"x": 1}, "Dict X"), ("y", "Y")])
    result = field.validate({"x": 1})
    assert result == {"x": 1}

def test_validate_raises_error_for_invalid_list_choice():
    field = Choice(choices=[(["a", "b"], "List AB")])
    try:
        field.validate(["a", "c"])
        assert False
    except Exception as exc:
        assert str(exc) == "Not a valid choice."

def test_validate_raises_error_for_invalid_dict_choice():
    field = Choice(choices=[({"x": 1}, "Dict X")])
    try:
        field.validate({"x": 2})
        assert False
    except Exception as exc:
        assert str(exc) == "Not a valid choice."


# LLM-generated content at query #9
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

def test_validate_with_item_field():
    field = Array(items=Integer())
    result = field.validate([1, 2, 3])
    assert result == [1, 2, 3]

def test_validate_with_item_field_invalid():
    field = Array(items=Integer())
    try:
        field.validate([1, "invalid", 3])
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "type"

def test_validate_with_item_list():
    field = Array(items=[Integer(), String()])
    result = field.validate([1, "hello"])
    assert result == [1, "hello"]

def test_validate_with_item_list_invalid():
    field = Array(items=[Integer(), String()])
    try:
        field.validate(["invalid", "hello"])
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "type"

def test_validate_additional_items_false():
    field = Array(items=[Integer(), String()], additional_items=False)
    try:
        field.validate([1, "hello", "extra"])
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "additional_items"

def test_validate_additional_items_field():
    field = Array(items=[Integer()], additional_items=String())
    result = field.validate([1, "extra1", "extra2"])
    assert result == [1, "extra1", "extra2"]

def test_validate_additional_items_field_invalid():
    field = Array(items=[Integer()], additional_items=String())
    try:
        field.validate([1, 2, "extra"])
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "type"

def test_validate_unique_items():
    field = Array(unique_items=True)
    try:
        field.validate([1, 2, 1])
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "unique_items"

def test_validate_unique_items_with_false_and_zero():
    field = Array(unique_items=True)
    result = field.validate([False, 0])
    assert result == [False, 0]

def test_validate_unique_items_with_true_and_one():
    field = Array(unique_items=True)
    result = field.validate([True, 1])
    assert result == [True, 1]

def test_validate_unique_items_with_list_items():
    field = Array(unique_items=True)
    try:
        field.validate([[1, 2], [1, 2]])
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "unique_items"

def test_validate_unique_items_with_dict_items():
    field = Array(unique_items=True)
    try:
        field.validate([{"a": 1}, {"a": 1}])
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "unique_items"

def test_validate_multiple_errors():
    field = Array(items=[Integer(), String()], unique_items=True)
    try:
        field.validate(["invalid", 2, "extra", "extra"])
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 3
        codes = [msg.code for msg in e.messages()]
        assert "type" in codes
        assert "additional_items" in codes
        assert "unique_items" in codes


# LLM-generated content at query #10
#--------------------------

def test_string_constructor_defaults():
    field = String()
    assert field.title == ""
    assert field.description == ""
    assert not field.has_default()
    assert field.allow_null is False
    assert field.read_only is False
    assert field.allow_blank is False
    assert field.trim_whitespace is True
    assert field.max_length is None
    assert field.min_length is None
    assert field.pattern is None
    assert field.pattern_regex is None
    assert field.format is None
    assert field.coerce_types is True

def test_string_constructor_with_allow_blank():
    field = String(allow_blank=True)
    assert field.allow_blank is True
    assert field.has_default() is True
    assert field.get_default_value() == ""

def test_string_constructor_with_allow_null():
    field = String(allow_null=True)
    assert field.allow_null is True
    assert not field.has_default()

def test_string_constructor_with_title_and_description():
    field = String(title="Name", description="Full name")
    assert field.title == "Name"
    assert field.description == "Full name"

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
    assert field.coerce_types is False

def test_string_constructor_with_trim_whitespace_false():
    field = String(trim_whitespace=False)
    assert field.trim_whitespace is False

def test_string_constructor_with_default():
    field = String(default="hello")
    assert field.has_default() is True
    assert field.get_default_value() == "hello"

def test_string_constructor_with_callable_default():
    field = String(default=lambda: "callable")
    assert field.has_default() is True
    assert field.get_default_value() == "callable"

def test_string_constructor_with_allow_blank_and_default():
    field = String(allow_blank=True, default="custom")
    assert field.allow_blank is True
    assert field.has_default() is True
    assert field.get_default_value() == "custom"

def test_string_constructor_with_read_only():
    field = String(read_only=True)
    assert field.read_only is True

def test_string_constructor_with_allow_null_and_default():
    field = String(allow_null=True, default="not null")
    assert field.allow_null is True
    assert field.has_default() is True
    assert field.get_default_value() == "not null"


# LLM-generated content at query #11
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

def test_array_constructor_with_items_field():
    item_field = Field()
    field = Array(items=item_field)
    assert field.items == item_field
    assert field.additional_items is False
    assert field.min_items is None
    assert field.max_items is None
    assert field.unique_items is False

def test_array_constructor_with_items_list():
    item_field1 = Field()
    item_field2 = Field()
    field = Array(items=[item_field1, item_field2])
    assert field.items == [item_field1, item_field2]
    assert field.additional_items is False
    assert field.min_items == 2
    assert field.max_items == 2

def test_array_constructor_with_additional_items_field():
    item_field = Field()
    additional_field = Field()
    field = Array(items=item_field, additional_items=additional_field)
    assert field.items == item_field
    assert field.additional_items == additional_field
    assert field.min_items is None
    assert field.max_items is None

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

def test_array_constructor_with_items_list_and_additional_items_false():
    item_field1 = Field()
    item_field2 = Field()
    field = Array(items=[item_field1, item_field2], additional_items=False)
    assert field.items == [item_field1, item_field2]
    assert field.additional_items is False
    assert field.min_items == 2
    assert field.max_items == 2

def test_array_constructor_with_items_list_and_additional_items_field():
    item_field1 = Field()
    item_field2 = Field()
    additional_field = Field()
    field = Array(items=[item_field1, item_field2], additional_items=additional_field)
    assert field.items == [item_field1, item_field2]
    assert field.additional_items == additional_field
    assert field.min_items == 2
    assert field.max_items is None

def test_array_constructor_with_items_list_and_min_items_override():
    item_field1 = Field()
    item_field2 = Field()
    field = Array(items=[item_field1, item_field2], min_items=5)
    assert field.items == [item_field1, item_field2]
    assert field.min_items == 5
    assert field.max_items == 2

def test_array_constructor_with_items_list_and_max_items_override():
    item_field1 = Field()
    item_field2 = Field()
    field = Array(items=[item_field1, item_field2], max_items=10)
    assert field.items == [item_field1, item_field2]
    assert field.min_items == 2
    assert field.max_items == 10

def test_array_constructor_with_items_list_and_additional_items_false_and_max_items_override():
    item_field1 = Field()
    item_field2 = Field()
    field = Array(items=[item_field1, item_field2], additional_items=False, max_items=5)
    assert field.items == [item_field1, item_field2]
    assert field.additional_items is False
    assert field.min_items == 2
    assert field.max_items == 2


# LLM-generated content at query #12
#--------------------------

def test_format_in_formats_and_is_native_type():
    field = String(format="email")
    value = "test@example.com"
    result = field.validate(value)
    assert result == value


# LLM-generated content at query #13
#--------------------------

def test_allow_null_false_coerce_types_false_blank_string():
    field = String(allow_null=False, coerce_types=False)
    try:
        field.validate("")
    except Exception as e:
        assert e.detail == "Must not be blank."


# LLM-generated content at query #14
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
    field = Boolean()
    try:
        field.validate(None)
        assert False
    except Exception as e:
        assert str(e) == "May not be null."

def test_validate_string_true():
    field = Boolean()
    result = field.validate("true")
    assert result is True

def test_validate_string_false():
    field = Boolean()
    result = field.validate("false")
    assert result is False

def test_validate_string_on():
    field = Boolean()
    result = field.validate("on")
    assert result is True

def test_validate_string_off():
    field = Boolean()
    result = field.validate("off")
    assert result is False

def test_validate_string_1():
    field = Boolean()
    result = field.validate("1")
    assert result is True

def test_validate_string_0():
    field = Boolean()
    result = field.validate("0")
    assert result is False

def test_validate_empty_string():
    field = Boolean()
    result = field.validate("")
    assert result is False

def test_validate_integer_1():
    field = Boolean()
    result = field.validate(1)
    assert result is True

def test_validate_integer_0():
    field = Boolean()
    result = field.validate(0)
    assert result is False

def test_validate_coerce_types_false_with_string():
    field = Boolean(coerce_types=False)
    try:
        field.validate("true")
        assert False
    except Exception as e:
        assert str(e) == "Must be a boolean."

def test_validate_coerce_types_false_with_integer():
    field = Boolean(coerce_types=False)
    try:
        field.validate(1)
        assert False
    except Exception as e:
        assert str(e) == "Must be a boolean."

def test_validate_allow_null_with_empty_string():
    field = Boolean(allow_null=True)
    result = field.validate("")
    assert result is None

def test_validate_allow_null_with_string_null():
    field = Boolean(allow_null=True)
    result = field.validate("null")
    assert result is None

def test_validate_allow_null_with_string_none():
    field = Boolean(allow_null=True)
    result = field.validate("none")
    assert result is None

def test_validate_invalid_string():
    field = Boolean()
    try:
        field.validate("invalid")
        assert False
    except Exception as e:
        assert str(e) == "Must be a boolean."

def test_validate_invalid_type():
    field = Boolean()
    try:
        field.validate([])
        assert False
    except Exception as e:
        assert str(e) == "Must be a boolean."


# LLM-generated content at query #15
#--------------------------

def test_allow_null_false_and_value_not_in_coerce_null_values():
    field = Boolean(allow_null=False)
    result = field.validate("")
    assert result is False


# LLM-generated content at query #16
#--------------------------

def test_validate_empty_string_with_allow_null_false():
    field = Number(allow_null=False)
    result = field.validate("")
    assert result is not None

def test_validate_empty_string_with_coerce_types_false():
    field = Number(allow_null=True, coerce_types=False)
    result = field.validate("")
    assert result is not None

def test_validate_empty_string_with_allow_null_false_and_coerce_types_false():
    field = Number(allow_null=False, coerce_types=False)
    result = field.validate("")
    assert result is not None

def test_validate_non_empty_string():
    field = Number(allow_null=True, coerce_types=True)
    result = field.validate("123")
    assert result == 123

def test_validate_zero():
    field = Number(allow_null=True, coerce_types=True)
    result = field.validate(0)
    assert result == 0

def test_validate_positive_integer():
    field = Number(allow_null=True, coerce_types=True)
    result = field.validate(42)
    assert result == 42

def test_validate_negative_float():
    field = Number(allow_null=True, coerce_types=True)
    result = field.validate(-3.14)
    assert result == -3.14

def test_validate_numeric_string():
    field = Number(allow_null=True, coerce_types=True)
    result = field.validate("99.9")
    assert result == 99.9


# LLM-generated content at query #17
#--------------------------

def test_validate_allow_blank_false_allow_null_false_coerce_types_true_empty_string_raises_blank():
    field = String(allow_blank=False, allow_null=False, coerce_types=True)
    try:
        field.validate("")
        assert False
    except Exception as e:
        assert str(e) == "Must not be blank."

def test_validate_allow_blank_false_allow_null_false_coerce_types_false_empty_string_raises_blank():
    field = String(allow_blank=False, allow_null=False, coerce_types=False)
    try:
        field.validate("")
        assert False
    except Exception as e:
        assert str(e) == "Must not be blank."

def test_validate_allow_blank_false_allow_null_true_coerce_types_false_empty_string_raises_blank():
    field = String(allow_blank=False, allow_null=True, coerce_types=False)
    try:
        field.validate("")
        assert False
    except Exception as e:
        assert str(e) == "Must not be blank."

def test_validate_allow_blank_true_allow_null_false_coerce_types_true_empty_string_returns_empty_string():
    field = String(allow_blank=True, allow_null=False, coerce_types=True)
    result = field.validate("")
    assert result == ""

def test_validate_allow_blank_true_allow_null_true_coerce_types_true_empty_string_returns_empty_string():
    field = String(allow_blank=True, allow_null=True, coerce_types=True)
    result = field.validate("")
    assert result == ""

def test_validate_allow_blank_true_allow_null_false_coerce_types_false_empty_string_returns_empty_string():
    field = String(allow_blank=True, allow_null=False, coerce_types=False)
    result = field.validate("")
    assert result == ""

def test_validate_allow_blank_true_allow_null_true_coerce_types_false_empty_string_returns_empty_string():
    field = String(allow_blank=True, allow_null=True, coerce_types=False)
    result = field.validate("")
    assert result == ""

def test_validate_allow_blank_false_allow_null_true_coerce_types_true_empty_string_returns_none():
    field = String(allow_blank=False, allow_null=True, coerce_types=True)
    result = field.validate("")
    assert result is None

def test_validate_allow_blank_false_allow_null_true_coerce_types_true_whitespace_string_returns_none():
    field = String(allow_blank=False, allow_null=True, coerce_types=True, trim_whitespace=True)
    result = field.validate("   ")
    assert result is None

def test_validate_allow_blank_false_allow_null_true_coerce_types_true_non_empty_string_returns_string():
    field = String(allow_blank=False, allow_null=True, coerce_types=True)
    result = field.validate("hello")
    assert result == "hello"


# LLM-generated content at query #18
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
    item_field = Field()
    item_field.serialize = lambda x: f"serialized_{x}"
    field = Array(items=item_field)
    obj = [1, 2, 3]
    result = field.serialize(obj)
    assert result == ["serialized_1", "serialized_2", "serialized_3"]

def test_serialize_with_list_of_item_fields():
    item_field1 = Field()
    item_field1.serialize = lambda x: f"a_{x}"
    item_field2 = Field()
    item_field2.serialize = lambda x: f"b_{x}"
    field = Array(items=[item_field1, item_field2])
    obj = [10, 20]
    result = field.serialize(obj)
    assert result == ["a_10", "b_20"]

def test_serialize_with_list_of_item_fields_and_shorter_obj():
    item_field1 = Field()
    item_field1.serialize = lambda x: f"a_{x}"
    item_field2 = Field()
    item_field2.serialize = lambda x: f"b_{x}"
    field = Array(items=[item_field1, item_field2])
    obj = [10]
    result = field.serialize(obj)
    assert result == ["a_10"]

def test_serialize_with_list_of_item_fields_and_longer_obj():
    item_field1 = Field()
    item_field1.serialize = lambda x: f"a_{x}"
    item_field2 = Field()
    item_field2.serialize = lambda x: f"b_{x}"
    field = Array(items=[item_field1, item_field2])
    obj = [10, 20, 30]
    result = field.serialize(obj)
    assert result == ["a_10", "b_20", 30]


# LLM-generated content at query #19
#--------------------------

```python
def test_validate_with_non_null_value_and_allow_null_false():
    from typesystem.fields import Object
    field = Object(allow_null=False)
    value = {"key": "value"}
    result = field.validate(value)
    assert result == {"key": "value"}


# LLM-generated content at query #20
#--------------------------

```python
def test_additional_properties_is_none_does_not_trigger_assertion():
    from typesystem.fields import Object, String
    field = Object(additional_properties=None)
    result = field.validate({"key": "value"})
    assert result == {"key": "value"}


# LLM-generated content at query #21
#--------------------------

def test_validate_returns_none_when_value_is_none_and_allow_null():
    from dataclasses import dataclass
    from typing import Any
    from unittest.mock import Mock
    mock_field = Mock()
    mock_field.allow_null = True
    mock_field.validate_or_error = Mock(return_value=(None, None))
    union = Union(any_of=[mock_field])
    result = union.validate(None)
    assert result is None

def test_validate_raises_null_error_when_value_is_none_and_not_allow_null():
    from dataclasses import dataclass
    from typing import Any
    from unittest.mock import Mock
    mock_field = Mock()
    mock_field.allow_null = False
    union = Union(any_of=[mock_field])
    try:
        union.validate(None)
        assert False
    except Exception as e:
        assert e.messages()[0].code == "null"

def test_validate_returns_validated_value_from_first_matching_child():
    from dataclasses import dataclass
    from typing import Any
    from unittest.mock import Mock
    mock_field1 = Mock()
    mock_field1.validate_or_error = Mock(return_value=("validated", None))
    mock_field2 = Mock()
    union = Union(any_of=[mock_field1, mock_field2])
    result = union.validate("test")
    assert result == "validated"

def test_validate_raises_single_candidate_error_when_one_child_has_non_type_error():
    from dataclasses import dataclass
    from typing import Any
    from unittest.mock import Mock
    from core.exceptions import ValidationError
    error = ValidationError(messages=[Mock(code="custom", index=None)])
    mock_field1 = Mock()
    mock_field1.validate_or_error = Mock(return_value=(None, error))
    mock_field2 = Mock()
    mock_field2.validate_or_error = Mock(return_value=(None, ValidationError(messages=[Mock(code="type", index=None)])))
    union = Union(any_of=[mock_field1, mock_field2])
    try:
        union.validate("test")
        assert False
    except Exception as e:
        assert e is error

def test_validate_raises_union_error_when_no_child_matches_and_no_single_candidate_error():
    from dataclasses import dataclass
    from typing import Any
    from unittest.mock import Mock
    from core.exceptions import ValidationError
    error1 = ValidationError(messages=[Mock(code="type", index=None)])
    error2 = ValidationError(messages=[Mock(code="type", index=None)])
    mock_field1 = Mock()
    mock_field1.validate_or_error = Mock(return_value=(None, error1))
    mock_field2 = Mock()
    mock_field2.validate_or_error = Mock(return_value=(None, error2))
    union = Union(any_of=[mock_field1, mock_field2])
    try:
        union.validate("test")
        assert False
    except Exception as e:
        assert e.messages()[0].code == "union"

def test_validate_raises_union_error_when_multiple_candidate_errors_exist():
    from dataclasses import dataclass
    from typing import Any
    from unittest.mock import Mock
    from core.exceptions import ValidationError
    error1 = ValidationError(messages=[Mock(code="custom1", index=None)])
    error2 = ValidationError(messages=[Mock(code="custom2", index=None)])
    mock_field1 = Mock()
    mock_field1.validate_or_error = Mock(return_value=(None, error1))
    mock_field2 = Mock()
    mock_field2.validate_or_error = Mock(return_value=(None, error2))
    union = Union(any_of=[mock_field1, mock_field2])
    try:
        union.validate("test")
        assert False
    except Exception as e:
        assert e.messages()[0].code == "union"

def test_validate_raises_error_with_index_from_child():
    from dataclasses import dataclass
    from typing import Any
    from unittest.mock import Mock
    from core.exceptions import ValidationError
    error = ValidationError(messages=[Mock(code="type", index=[0])])
    mock_field = Mock()
    mock_field.validate_or_error = Mock(return_value=(None, error))
    union = Union(any_of=[mock_field])
    try:
        union.validate("test")
        assert False
    except Exception as e:
        assert e.messages()[0].code == "union"

def test_validate_allow_null_set_true_if_any_child_allows_null():
    from dataclasses import dataclass
    from typing import Any
    from unittest.mock import Mock
    mock_field1 = Mock()
    mock_field1.allow_null = True
    mock_field2 = Mock()
    mock_field2.allow_null = False
    union = Union(any_of=[mock_field1, mock_field2])
    assert union.allow_null == True


# LLM-generated content at query #22
#--------------------------

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

def test_validate_non_dict():
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
    property_names = String(max_length=3)
    field = Object(property_names=property_names)
    try:
        field.validate({"longkey": "value"})
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

def test_validate_min_properties():
    field = Object(min_properties=2)
    try:
        field.validate({"a": 1})
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "min_properties"

def test_validate_max_properties():
    field = Object(max_properties=1)
    try:
        field.validate({"a": 1, "b": 2})
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "max_properties"

def test_validate_required_missing():
    field = Object(required=["key1"])
    try:
        field.validate({})
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "required"

def test_validate_properties_with_default():
    child = String(default="default_value")
    field = Object(properties={"key": child})
    result = field.validate({})
    assert result == {"key": "default_value"}

def test_validate_properties_valid():
    child = String()
    field = Object(properties={"key": child})
    result = field.validate({"key": "value"})
    assert result == {"key": "value"}

def test_validate_properties_invalid():
    child = String(max_length=3)
    field = Object(properties={"key": child})
    try:
        field.validate({"key": "toolong"})
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "max_length"

def test_validate_pattern_properties():
    child = String(max_length=3)
    field = Object(pattern_properties={"^a": child})
    result = field.validate({"a1": "val", "b1": "ignored"})
    assert result == {"a1": "val"}

def test_validate_pattern_properties_invalid():
    child = String(max_length=3)
    field = Object(pattern_properties={"^a": child})
    try:
        field.validate({"a1": "toolong"})
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "max_length"

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

def test_validate_additional_properties_field():
    child = String(max_length=3)
    field = Object(additional_properties=child)
    result = field.validate({"extra": "val"})
    assert result == {"extra": "val"}

def test_validate_additional_properties_field_invalid():
    child = String(max_length=3)
    field = Object(additional_properties=child)
    try:
        field.validate({"extra": "toolong"})
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "max_length"

def test_validate_multiple_errors():
    field = Object(required=["req"], additional_properties=False)
    try:
        field.validate({"extra": "value"})
        assert False
    except ValidationError as e:
        codes = {msg.code for msg in e.messages()}
        assert codes == {"required", "invalid_property"}


# LLM-generated content at query #23
#--------------------------

def test_validate_choice_not_in_uniqueness():
    field = Choice(choices=[("key1", "value1"), ("key2", "value2")], allow_null=False, coerce_types=True)
    result = field.validate("key3")
    assert result is None


# LLM-generated content at query #24
#--------------------------

```python
def test_additional_properties_is_field_assertion_passes():
    from typesystem.fields import Object, String
    from typesystem.schemas import Schema
    from typesystem import ValidationError
    field = Object(additional_properties=String(max_length=5))
    value = {"extra": "valid"}
    result = field.validate(value)
    assert result == {"extra": "valid"}


# LLM-generated content at query #25
#--------------------------

def test_validate_value_not_in_uniqueness():
    field = Choice(choices=[("key1", "value1"), ("key2", "value2")], allow_null=False, coerce_types=True)
    result = field.validate("invalid")
    assert result is None


