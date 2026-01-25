####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_validate_null_with_allow_null():
    field = String(allow_null=True)
    assert field.validate(None) is None

def test_validate_null_without_allow_null():
    field = String(allow_null=False)
    try:
        field.validate(None)
        assert False
    except Exception as e:
        assert str(e) == "May not be null."

def test_validate_non_string_type():
    field = String()
    try:
        field.validate(123)
        assert False
    except Exception as e:
        assert str(e) == "Must be a string."

def test_validate_empty_string_with_allow_blank():
    field = String(allow_blank=True)
    assert field.validate("") == ""

def test_validate_empty_string_without_allow_blank():
    field = String(allow_blank=False)
    try:
        field.validate("")
        assert False
    except Exception as e:
        assert str(e) == "Must not be blank."

def test_validate_whitespace_string_with_trim_whitespace():
    field = String(trim_whitespace=True)
    assert field.validate("  hello  ") == "hello"

def test_validate_whitespace_string_without_trim_whitespace():
    field = String(trim_whitespace=False)
    assert field.validate("  hello  ") == "  hello  "

def test_validate_min_length():
    field = String(min_length=5)
    assert field.validate("hello") == "hello"

def test_validate_min_length_error():
    field = String(min_length=5)
    try:
        field.validate("hi")
        assert False
    except Exception as e:
        assert str(e) == "Must have at least 5 characters."

def test_validate_max_length():
    field = String(max_length=5)
    assert field.validate("hello") == "hello"

def test_validate_max_length_error():
    field = String(max_length=5)
    try:
        field.validate("hello world")
        assert False
    except Exception as e:
        assert str(e) == "Must have no more than 5 characters."

def test_validate_pattern():
    field = String(pattern=r"^\d+$")
    assert field.validate("123") == "123"

def test_validate_pattern_error():
    field = String(pattern=r"^\d+$")
    try:
        field.validate("abc")
        assert False
    except Exception as e:
        assert str(e) == "Must match the pattern /^\\d+$/."

def test_validate_format():
    field = String(format="email")
    assert field.validate("test@example.com") == "test@example.com"

def test_validate_null_character():
    field = String()
    assert field.validate("hello\0world") == "helloworld


# LLM-generated content at query #2
#--------------------------

```
def test_get_default_value_with_default():
    field = Field(default="hello")
    assert field.get_default_value() == "hello"

def test_get_default_value_with_callable_default():
    field = Field(default=lambda: "world")
    assert field.get_default_value() == "world"

def test_get_default_value_without_default():
    field = Field()
    assert field.get_default_value() is None

def test_get_default_value_with_default_none():
    field = Field(default=None)
    assert field.get_default_value() is None


# LLM-generated content at query #3
#--------------------------

```
def test_validate_null_allowed():
    field = Union(any_of=[Field(allow_null=True)], allow_null=True)
    assert field.validate(None) is None

def test_validate_null_not_allowed():
    field = Union(any_of=[Field()])
    try:
        field.validate(None)
        assert False
    except Exception as e:
        assert str(e) == "May not be null."

def test_validate_valid_value():
    field = Union(any_of=[Field()])
    assert field.validate("test") == "test"

def test_validate_invalid_value():
    field = Union(any_of=[Field()])
    try:
        field.validate(123)
        assert False
    except Exception as e:
        assert str(e) == "Did not match any valid type."

def test_validate_multiple_fields_valid():
    field = Union(any_of=[Field(), Field()])
    assert field.validate("test") == "test"

def test_validate_multiple_fields_invalid():
    field = Union(any_of=[Field(), Field()])
    try:
        field.validate(123)
        assert False
    except Exception as e:
        assert str(e) == "Did not match any valid type."

def test_validate_child_error_propagation():
    child = Field()
    field = Union(any_of=[child])
    try:
        field.validate(123)
        assert False
    except Exception as e:
        assert str(e) == "Did not match any valid type."


# LLM-generated content at query #4
#--------------------------

```python
def test_number_validate_null_allowed():
    field = Number(allow_null=True)
    assert field.validate(None) is None

def test_number_validate_null_not_allowed():
    field = Number(allow_null=False)
    try:
        field.validate(None)
        assert False
    except Exception as e:
        assert str(e) == "May not be null."

def test_number_validate_bool():
    field = Number()
    try:
        field.validate(True)
        assert False
    except Exception as e:
        assert str(e) == "Must be a number."

def test_number_validate_string_coerce():
    field = Number(coerce_types=True)
    assert field.validate("123") == 123

def test_number_validate_string_no_coerce():
    field = Number(coerce_types=False)
    try:
        field.validate("123")
        assert False
    except Exception as e:
        assert str(e) == "Must be a number."

def test_number_validate_infinity():
    field = Number()
    try:
        field.validate(float('inf'))
        assert False
    except Exception as e:
        assert str(e) == "Must be finite."

def test_number_validate_minimum():
    field = Number(minimum=10)
    try:
        field.validate(5)
        assert False
    except Exception as e:
        assert str(e) == "Must be greater than or equal to 10."

def test_number_validate_exclusive_minimum():
    field = Number(exclusive_minimum=10)
    try:
        field.validate(10)
        assert False
    except Exception as e:
        assert str(e) == "Must be greater than 10."

def test_number_validate_maximum():
    field = Number(maximum=10)
    try:
        field.validate(15)
        assert False
    except Exception as e:
        assert str(e) == "Must be less than or equal to 10."

def test_number_validate_exclusive_maximum():
    field = Number(exclusive_maximum=10)
    try:
        field.validate(10)
        assert False
    except Exception as e:
        assert str(e) == "Must be less than 10."

def test_number_validate_multiple_of_int():
    field = Number(multiple_of=5)
    try:
        field.validate(7)
        assert False
    except Exception as e:
        assert str(e) == "Must be a multiple of 5."

def test_number_validate_multiple_of_float():
    field = Number(multiple_of=0.5)
    try:
        field.validate(0.7)
        assert False
    except Exception as e:
        assert str(e) == "Must be a multiple of 0.5."

def test_number_validate_precision():
    field = Number(precision="0.01")
    assert field.validate(1.234) == 1.23


# LLM-generated content at query #5
#--------------------------

```python
def test_array_constructor_with_default_values():
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

def test_array_constructor_with_items():
    field = Array(items=Field())
    assert isinstance(field.items, Field)
    assert field.additional_items is False
    assert field.min_items is None
    assert field.max_items is None
    assert field.unique_items is False

def test_array_constructor_with_list_of_items():
    field = Array(items=[Field(), Field()])
    assert isinstance(field.items, list)
    assert len(field.items) == 2
    assert all(isinstance(item, Field) for item in field.items)
    assert field.min_items == 2
    assert field.max_items == 2

def test_array_constructor_with_additional_items():
    field = Array(additional_items=True)
    assert field.additional_items is True

def test_array_constructor_with_min_max_items():
    field = Array(min_items=1, max_items=10)
    assert field.min_items == 1
    assert field.max_items == 10

def test_array_constructor_with_exact_items():
    field = Array(exact_items=5)
    assert field.min_items == 5
    assert field.max_items == 5

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


# LLM-generated content at query #6
#--------------------------

```python
def test_validate_null_value_when_null_allowed():
    array = Array(allow_null=True)
    assert array.validate(None) is None

def test_validate_null_value_when_null_not_allowed():
    array = Array(allow_null=False)
    try:
        array.validate(None)
    except ValidationError as e:
        assert e.messages[0].text == "May not be null."

def test_validate_non_list_value():
    array = Array()
    try:
        array.validate("not a list")
    except ValidationError as e:
        assert e.messages[0].text == "Must be an array."

def test_validate_empty_list_when_min_items_not_set():
    array = Array()
    assert array.validate([]) == []

def test_validate_empty_list_when_min_items_set_to_one():
    array = Array(min_items=1)
    try:
        array.validate([])
    except ValidationError as e:
        assert e.messages[0].text == "Must not be empty."

def test_validate_list_with_less_than_min_items():
    array = Array(min_items=3)
    try:
        array.validate([1, 2])
    except ValidationError as e:
        assert e.messages[0].text == "Must have at least 3 items."

def test_validate_list_with_more_than_max_items():
    array = Array(max_items=2)
    try:
        array.validate([1, 2, 3])
    except ValidationError as e:
        assert e.messages[0].text == "Must have no more than 2 items."

def test_validate_list_with_exact_items():
    array = Array(exact_items=2)
    try:
        array.validate([1])
    except ValidationError as e:
        assert e.messages[0].text == "Must have 2 items."

def test_validate_list_with_unique_items():
    array = Array(unique_items=True)
    try:
        array.validate([1, 1])
    except ValidationError as e:
        assert e.messages[0].text == "Items must be unique."

def test_validate_list_with_items_validator():
    array = Array(items=Field(type="integer"))
    try:
        array.validate(["not an integer"])
    except ValidationError as e:
        assert e.messages[0].text == "Must be an integer."

def test_validate_list_with_items_validator_and_additional_items():
    array = Array(items=[Field(type="integer")], additional_items=Field(type="string"))
    try:
        array.validate([1, "not a string"])
    except ValidationError as e:
        assert e.messages[0].text == "Must be a string."

def test_validate_list_with_items_validator_and_no_additional_items():
    array = Array(items=[Field(type="integer")], additional_items=False)
    try:
        array.validate([1, 2])
    except ValidationError as e:
        assert e.messages[0].text == "May not contain additional items."

def test_validate_list_with_items_validator_and_unique_items():
    array = Array(items=[Field(type="integer")], unique_items=True)
    try:
        array.validate([1, 1])
    except ValidationError as e:
        assert e.messages[0].text == "Items must be unique."


# LLM-generated content at query #7
#--------------------------

```python
def test_validate_allows_null_when_allow_null_is_true():
    field = Boolean(allow_null=True)
    assert field.validate(None) is None

def test_validate_raises_error_when_null_and_allow_null_is_false():
    field = Boolean(allow_null=False)
    try:
        field.validate(None)
    except Exception as e:
        assert e.args[0] == "May not be null."

def test_validate_returns_true_for_true():
    field = Boolean()
    assert field.validate(True) is True

def test_validate_returns_false_for_false():
    field = Boolean()
    assert field.validate(False) is False

def test_validate_coerces_string_true_to_true():
    field = Boolean()
    assert field.validate("true") is True

def test_validate_coerces_string_false_to_false():
    field = Boolean()
    assert field.validate("false") is False

def test_validate_coerces_string_on_to_true():
    field = Boolean()
    assert field.validate("on") is True

def test_validate_coerces_string_off_to_false():
    field = Boolean()
    assert field.validate("off") is False

def test_validate_coerces_string_1_to_true():
    field = Boolean()
    assert field.validate("1") is True

def test_validate_coerces_string_0_to_false():
    field = Boolean()
    assert field.validate("0") is False

def test_validate_coerces_empty_string_to_false():
    field = Boolean()
    assert field.validate("") is False

def test_validate_coerces_numeric_1_to_true():
    field = Boolean()
    assert field.validate(1) is True

def test_validate_coerces_numeric_0_to_false():
    field = Boolean()
    assert field.validate(0) is False

def test_validate_raises_error_when_invalid_string():
    field = Boolean()
    try:
        field.validate("invalid")
    except Exception as e:
        assert e.args[0] == "Must be a boolean."

def test_validate_raises_error_when_invalid_type():
    field = Boolean()
    try:
        field.validate(123)
    except Exception as e:
        assert e.args[0] == "Must be a boolean."

def test_validate_returns_null_for_null_string_when_allow_null_is_true():
    field = Boolean(allow_null=True)
    assert field.validate("null") is None

def test_validate_returns_null_for_none_string_when_allow_null_is_true():
    field = Boolean(allow_null=True)
    assert field.validate("none") is None

def test_validate_returns_null_for_empty_string_when_allow_null_is_true():
    field = Boolean(allow_null=True)
    assert field.validate("") is None

def test_validate_raises_error_when_null_string_and_allow_null_is_false():
    field = Boolean(allow_null=False)
    try:
        field.validate("null")
    except Exception as e:
        assert e.args[0] == "May not be null."

def test_validate_raises_error_when_none_string_and_allow_null_is_false():
    field = Boolean(allow_null=False)
    try:
        field.validate("none")
    except Exception as e:
        assert e.args[0] == "May not be null."

def test_validate_raises_error_when_empty_string_and_allow_null_is_false():
    field = Boolean(allow_null=False)
    try:
        field.validate("")
    except Exception as e:
        assert e.args[0] == "May not be null."

def test_validate_raises_error_when_coerce_types_is_false_and_value_is_not_bool():
    field = Boolean(coerce_types=False)
    try:
        field.validate("true")
    except Exception as e:
        assert e.args[0] == "Must be a boolean."


# LLM-generated content at query #8
#--------------------------

```python
def test_validate_null_value():
    obj = Object(allow_null=True)
    assert obj.validate(None) is None

def test_validate_non_null_value():
    obj = Object(allow_null=False)
    try:
        obj.validate(None)
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "May not be null."
        assert e.messages[0].code == "null"

def test_validate_non_dict_value():
    obj = Object()
    try:
        obj.validate("not_a_dict")
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Must be an object."
        assert e.messages[0].code == "type"

def test_validate_invalid_key():
    obj = Object()
    try:
        obj.validate({1: "value"})
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "All object keys must be strings."
        assert e.messages[0].code == "invalid_key"

def test_validate_min_properties():
    obj = Object(min_properties=2)
    try:
        obj.validate({"key": "value"})
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Must have at least 2 properties."
        assert e.messages[0].code == "min_properties"

def test_validate_max_properties():
    obj = Object(max_properties=1)
    try:
        obj.validate({"key1": "value1", "key2": "value2"})
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Must have no more than 1 properties."
        assert e.messages[0].code == "max_properties"

def test_validate_required_property():
    obj = Object(required=["key"])
    try:
        obj.validate({})
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "This field is required."
        assert e.messages[0].code == "required"

def test_validate_properties():
    obj = Object(properties={"key": Field()})
    assert obj.validate({"key": "value"}) == {"key": "value"}

def test_validate_pattern_properties():
    obj = Object(pattern_properties={r".*": Field()})
    assert obj.validate({"key": "value"}) == {"key": "value"}

def test_validate_additional_properties_true():
    obj = Object(additional_properties=True)
    assert obj.validate({"key": "value"}) == {"key": "value"}

def test_validate_additional_properties_false():
    obj = Object(additional_properties=False)
    try:
        obj.validate({"key": "value"})
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Invalid property name."
        assert e.messages[0].code == "invalid_property"

def test_validate_additional_properties_field():
    obj = Object(additional_properties=Field())
    assert obj.validate({"key": "value"}) == {"key": "value"}


# LLM-generated content at query #9
#--------------------------

```python
def test_pattern_properties_validation_success():
    class MockField:
        def validate_or_error(self, value):
            return value, None

    pattern_properties = {"a+": MockField()}
    obj = Object(pattern_properties=pattern_properties)
    value = {"a": "valid"}
    validated = obj.validate(value)
    assert validated == {"a": "valid"}


# LLM-generated content at query #10
#--------------------------

```python
def test_validate_empty_string_with_allow_null_and_coerce_types():
    field = String(allow_null=True, coerce_types=True)
    result = field.validate("")
    assert result is None


# LLM-generated content at query #11
#--------------------------

```python
def test_validate_returns_none_when_value_is_null_and_allow_null_is_true():
    obj = Object(allow_null=True)
    result = obj.validate(None)
    assert result is None


# LLM-generated content at query #12
#--------------------------

Here are the test cases for the `validate` method of the `Object` class:


# LLM-generated content at query #13
#--------------------------

```python
def test_predicate_at_line_17_evaluates_to_true():
    class MockField:
        def validate_or_error(self, value):
            return None, MockError(messages=[MockMessage(code="invalid", index=1)])

    class MockError:
        def __init__(self, messages):
            self._messages = messages

        def messages(self):
            return self._messages

    class MockMessage:
        def __init__(self, code, index):
            self.code = code
            self.index = index

    field = Union(any_of=[MockField()])
    value = "invalid_value"
    field.validate(value)


# LLM-generated content at query #14
#--------------------------

```python
def test_validate_returns_none_when_value_is_none_and_allow_null_is_true():
    field = Array(allow_null=True)
    result = field.validate(None)
    assert result is None


# LLM-generated content at query #15
#--------------------------

```
def test_validate_with_exact_items_mismatch():
    field = Array(items=None, exact_items=3)
    try:
        field.validate([1, 2])
    except ValidationError as e:
        assert e.messages()[0].code == "exact_items"


# LLM-generated content at query #16
#--------------------------

```
def test_string_constructor_default_values():
    field = String()
    assert field.title == ""
    assert field.description == ""
    assert field.allow_null is False
    assert field.read_only is False
    assert field.allow_blank is False
    assert field.trim_whitespace is True
    assert field.max_length is None
    assert field.min_length is None
    assert field.pattern is None
    assert field.format is None
    assert field.coerce_types is True

def test_string_constructor_custom_values():
    field = String(
        title="Test Title",
        description="Test Description",
        allow_null=True,
        read_only=True,
        allow_blank=True,
        trim_whitespace=False,
        max_length=10,
        min_length=5,
        pattern="test",
        format="email",
        coerce_types=False,
    )
    assert field.title == "Test Title"
    assert field.description == "Test Description"
    assert field.allow_null is True
    assert field.read_only is True
    assert field.allow_blank is True
    assert field.trim_whitespace is False
    assert field.max_length == 10
    assert field.min_length == 5
    assert field.pattern == "test"
    assert field.format == "email"
    assert field.coerce_types is False

def test_string_constructor_allow_blank_with_default():
    field = String(allow_blank=True)
    assert field.default == ""

def test_string_constructor_max_length_type_check():
    try:
        String(max_length="invalid")
    except AssertionError:
        pass

def test_string_constructor_min_length_type_check():
    try:
        String(min_length="invalid")
    except AssertionError:
        pass

def test_string_constructor_pattern_type_check():
    try:
        String(pattern=123)
    except AssertionError:
        pass

def test_string_constructor_format_type_check():
    try:
        String(format=123)
    except AssertionError:
        pass


# LLM-generated content at query #17
#--------------------------

```
def test_validate_null_value_when_allow_null_is_true():
    field = String(allow_null=True)
    assert field.validate(None) is None

def test_validate_null_value_when_allow_null_is_false():
    field = String(allow_null=False)
    try:
        field.validate(None)
        assert False
    except Exception as e:
        assert str(e) == "May not be null."

def test_validate_null_value_when_allow_blank_is_true():
    field = String(allow_blank=True, coerce_types=True)
    assert field.validate(None) == ""

def test_validate_non_string_value():
    field = String()
    try:
        field.validate(123)
        assert False
    except Exception as e:
        assert str(e) == "Must be a string."

def test_validate_empty_string_when_allow_blank_is_false():
    field = String(allow_blank=False)
    try:
        field.validate("")
        assert False
    except Exception as e:
        assert str(e) == "Must not be blank."

def test_validate_empty_string_when_allow_blank_is_true():
    field = String(allow_blank=True)
    assert field.validate("") == ""

def test_validate_whitespace_string_when_trim_whitespace_is_true():
    field = String(trim_whitespace=True)
    assert field.validate("  test  ") == "test"

def test_validate_whitespace_string_when_trim_whitespace_is_false():
    field = String(trim_whitespace=False)
    assert field.validate("  test  ") == "  test  "

def test_validate_string_with_null_character():
    field = String()
    assert field.validate("te\0st") == "test"

def test_validate_string_with_min_length_constraint():
    field = String(min_length=5)
    try:
        field.validate("test")
        assert False
    except Exception as e:
        assert str(e) == "Must have at least 5 characters."

def test_validate_string_with_max_length_constraint():
    field = String(max_length=3)
    try:
        field.validate("test")
        assert False
    except Exception as e:
        assert str(e) == "Must have no more than 3 characters."

def test_validate_string_with_pattern_constraint():
    field = String(pattern="^[a-z]+$")
    try:
        field.validate("test123")
        assert False
    except Exception as e:
        assert str(e) == "Must match the pattern /^[a-z]+$/."

def test_validate_valid_string_with_pattern_constraint():
    field = String(pattern="^[a-z]+$")
    assert field.validate("test") == "test"

def test_validate_string_with_format():
    field = String(format="email")
    try:
        field.validate("invalid-email")
        assert False
    except Exception as e:
        assert "Must be a valid email" in str(e)

def test_validate_valid_string_with_format():
    field = String(format="email")
    assert field.validate("test@example.com") == "test@example.com"


# LLM-generated content at query #18
#--------------------------

```python
def test_predicate_at_line_26_evaluates_to_false():
    field = Array(items=[Field()], min_items=5, additional_items=True)


# LLM-generated content at query #19
#--------------------------

```python
def test_validate_valid_object():
    field = Object()
    value = {"key": "value"}
    result = field.validate(value)
    assert result == {"key": "value"}

def test_validate_null_value():
    field = Object(allow_null=True)
    result = field.validate(None)
    assert result is None

def test_validate_null_value_not_allowed():
    field = Object()
    try:
        field.validate(None)
    except ValidationError as e:
        assert e.messages[0].text == "May not be null."
        assert e.messages[0].code == "null"

def test_validate_non_dict_value():
    field = Object()
    try:
        field.validate("not a dict")
    except ValidationError as e:
        assert e.messages[0].text == "Must be an object."
        assert e.messages[0].code == "type"

def test_validate_invalid_key():
    field = Object()
    value = {1: "value"}
    try:
        field.validate(value)
    except ValidationError as e:
        assert e.messages[0].text == "All object keys must be strings."
        assert e.messages[0].code == "invalid_key"
        assert e.messages[0].index == [1]

def test_validate_min_properties():
    field = Object(min_properties=2)
    value = {"key1": "value1"}
    try:
        field.validate(value)
    except ValidationError as e:
        assert e.messages[0].text == "Must have at least 2 properties."
        assert e.messages[0].code == "min_properties"

def test_validate_max_properties():
    field = Object(max_properties=1)
    value = {"key1": "value1", "key2": "value2"}
    try:
        field.validate(value)
    except ValidationError as e:
        assert e.messages[0].text == "Must have no more than 1 properties."
        assert e.messages[0].code == "max_properties"

def test_validate_required_properties():
    field = Object(required=["key1"])
    value = {"key2": "value2"}
    try:
        field.validate(value)
    except ValidationError as e:
        assert e.messages[0].text == "This field is required."
        assert e.messages[0].code == "required"
        assert e.messages[0].index == ["key1"]

def test_validate_additional_properties_false():
    field = Object(additional_properties=False)
    value = {"key1": "value1"}
    try:
        field.validate(value)
    except ValidationError as e:
        assert e.messages[0].text == "Invalid property name."
        assert e.messages[0].code == "invalid_property"
        assert e.messages[0].key == "key1"

def test_validate_additional_properties_field():
    field = Object(additional_properties=String())
    value = {"key1": "value1"}
    result = field.validate(value)
    assert result == {"key1": "value1"}

def test_validate_additional_properties_field_invalid():
    field = Object(additional_properties=Integer())
    value = {"key1": "not an integer"}
    try:
        field.validate(value)
    except ValidationError as e:
        assert e.messages[0].text == "Must be a valid integer."
        assert e.messages[0].code == "type"
        assert e.messages[0].index == ["key1"]


# LLM-generated content at query #20
#--------------------------

```
def test_allow_null_and_coerce_types_with_empty_string():
    field = String(allow_null=True, coerce_types=True)
    result = field.validate("")
    assert result is None

def test_allow_null_and_coerce_types_with_whitespace_string():
    field = String(allow_null=True, coerce_types=True, trim_whitespace=True)
    result = field.validate("   ")
    assert result is None

def test_allow_null_and_coerce_types_with_empty_string_and_trim_whitespace_false():
    field = String(allow_null=True, coerce_types=True, trim_whitespace=False)
    result = field.validate("")
    assert result is None


# LLM-generated content at query #21
#--------------------------

```python
def test_validate_false_predicate():
    field = Object(allow_null=True)
    result = field.validate(None)
    assert result is None


# LLM-generated content at query #22
#--------------------------

```python
def test_validate_raises_integer_error_for_float_with_non_integer_value_when_numeric_type_is_int():
    field = Number(numeric_type=int)
    try:
        field.validate(3.14)
        assert False, "Expected validation_error('integer') to be raised"
    except Exception as e:
        assert str(e) == "Must be an integer."


# LLM-generated content at query #23
#--------------------------

Here's the unit test to ensure the predicate at line 63 evaluates to False:


# LLM-generated content at query #24
#--------------------------

```python
def test_validate_allow_null():
    field = Number(allow_null=True)
    assert field.validate(None) is None

def test_validate_empty_string_with_allow_null():
    field = Number(allow_null=True, coerce_types=True)
    assert field.validate("") is None

def test_validate_null_without_allow_null():
    field = Number(allow_null=False)
    try:
        field.validate(None)
        assert False
    except Exception as e:
        assert str(e) == "May not be null."

def test_validate_boolean():
    field = Number()
    try:
        field.validate(True)
        assert False
    except Exception as e:
        assert str(e) == "Must be a number."

def test_validate_integer_type_with_float():
    field = Number(numeric_type=int)
    try:
        field.validate(1.5)
        assert False
    except Exception as e:
        assert str(e) == "Must be an integer."

def test_validate_non_numeric_without_coerce_types():
    field = Number(coerce_types=False)
    try:
        field.validate("abc")
        assert False
    except Exception as e:
        assert str(e) == "Must be a number."

def test_validate_string_with_coerce_types():
    field = Number(coerce_types=True)
    assert field.validate("123") == 123

def test_validate_infinite_value():
    field = Number()
    try:
        field.validate(float("inf"))
        assert False
    except Exception as e:
        assert str(e) == "Must be finite."

def test_validate_minimum():
    field = Number(minimum=10)
    try:
        field.validate(9)
        assert False
    except Exception as e:
        assert str(e) == "Must be greater than or equal to 10."

def test_validate_exclusive_minimum():
    field = Number(exclusive_minimum=10)
    try:
        field.validate(10)
        assert False
    except Exception as e:
        assert str(e) == "Must be greater than 10."

def test_validate_maximum():
    field = Number(maximum=10)
    try:
        field.validate(11)
        assert False
    except Exception as e:
        assert str(e) == "Must be less than or equal to 10."

def test_validate_exclusive_maximum():
    field = Number(exclusive_maximum=10)
    try:
        field.validate(10)
        assert False
    except Exception as e:
        assert str(e) == "Must be less than 10."

def test_validate_multiple_of_int():
    field = Number(multiple_of=2)
    try:
        field.validate(3)
        assert False
    except Exception as e:
        assert str(e) == "Must be a multiple of 2."

def test_validate_multiple_of_float():
    field = Number(multiple_of=0.5)
    try:
        field.validate(1.2)
        assert False
    except Exception as e:
        assert str(e) == "Must be a multiple of 0.5."

def test_validate_precision():
    field = Number(precision="0.01")
    assert field.validate(1.234) == 1.23


# LLM-generated content at query #25
#--------------------------

```python
def test_const_constructor_with_const_value():
    instance = Const(const=42)
    assert instance.const == 42
    assert instance.title == ""
    assert instance.description == ""
    assert instance.allow_null is False
    assert instance.read_only is False

def test_const_constructor_with_custom_kwargs():
    instance = Const(const=10, title="Example", description="Test Description", read_only=True)
    assert instance.const == 10
    assert instance.title == "Example"
    assert instance.description == "Test Description"
    assert instance.allow_null is False
    assert instance.read_only is True

def test_const_constructor_with_allow_null_not_allowed():
    try:
        Const(const=5, allow_null=True)
    except AssertionError:
        pass
    else:
        assert False, "Constructor should raise AssertionError when allow_null is provided"


# LLM-generated content at query #26
#--------------------------

```
class MockField:
    def validate_or_error(self, value):
        return None, MockValidationError()

class MockValidationError:
    def messages(self):
        return [MockMessage(code="invalid", index=1)]

class MockMessage:
    def __init__(self, code, index):
        self.code = code
        self.index = index

def test_predicate_evaluates_to_true():
    field = Union(any_of=[MockField()])
    field.validate("invalid_value")


# LLM-generated content at query #27
#--------------------------

```python
def test_validate_pattern_properties_with_error():
    class DummyField(Field):
        def validate(self, value):
            if value != "valid":
                raise self.validation_error("invalid")

    pattern_properties = {"pattern": DummyField()}
    obj = Object(pattern_properties=pattern_properties)
    value = {"pattern": "invalid"}
    try:
        obj.validate(value)
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "invalid"


# LLM-generated content at query #28
#--------------------------

```
def test_allow_blank_sets_default_to_empty_string_when_no_default():
    field = String(allow_blank=True)
    assert field.default == ""
    assert field.has_default()

def test_allow_blank_does_not_override_existing_default():
    field = String(allow_blank=True, default="existing")
    assert field.default == "existing"
    assert field.has_default()

def test_allow_blank_false_does_not_set_default():
    field = String(allow_blank=False)
    assert not hasattr(field, "default")
    assert not field.has_default()


# LLM-generated content at query #29
#--------------------------

```python
def test_validate_null_value_allowed():
    field = Number(allow_null=True)
    assert field.validate(None) is None

def test_validate_null_value_not_allowed():
    field = Number(allow_null=False)
    try:
        field.validate(None)
        assert False
    except Exception as e:
        assert str(e) == "May not be null."

def test_validate_empty_string_allowed():
    field = Number(allow_null=True, coerce_types=True)
    assert field.validate("") is None

def test_validate_empty_string_not_allowed():
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

def test_validate_integer():
    field = Number(numeric_type=int)
    assert field.validate(42) == 42

def test_validate_float_not_integer():
    field = Number(numeric_type=int)
    try:
        field.validate(42.5)
        assert False
    except Exception as e:
        assert str(e) == "Must be an integer."

def test_validate_non_numeric_string():
    field = Number(coerce_types=False)
    try:
        field.validate("abc")
        assert False
    except Exception as e:
        assert str(e) == "Must be a number."

def test_validate_numeric_string():
    field = Number()
    assert field.validate("42") == 42

def test_validate_infinite_value():
    field = Number()
    try:
        field.validate(float('inf'))
        assert False
    except Exception as e:
        assert str(e) == "Must be finite."

def test_validate_precision():
    field = Number(precision="0.01")
    assert field.validate(42.123) == 42.12

def test_validate_minimum():
    field = Number(minimum=10)
    assert field.validate(15) == 15

def test_validate_below_minimum():
    field = Number(minimum=10)
    try:
        field.validate(5)
        assert False
    except Exception as e:
        assert str(e) == "Must be greater than or equal to 10."

def test_validate_exclusive_minimum():
    field = Number(exclusive_minimum=10)
    assert field.validate(15) == 15

def test_validate_below_exclusive_minimum():
    field = Number(exclusive_minimum=10)
    try:
        field.validate(10)
        assert False
    except Exception as e:
        assert str(e) == "Must be greater than 10."

def test_validate_maximum():
    field = Number(maximum=20)
    assert field.validate(15) == 15

def test_validate_above_maximum():
    field = Number(maximum=20)
    try:
        field.validate(25)
        assert False
    except Exception as e:
        assert str(e) == "Must be less than or equal to 20."

def test_validate_exclusive_maximum():
    field = Number(exclusive_maximum=20)
    assert field.validate(15) == 15

def test_validate_above_exclusive_maximum():
    field = Number(exclusive_maximum=20)
    try:
        field.validate(20)
        assert False
    except Exception as e:
        assert str(e) == "Must be less than 20."

def test_validate_multiple_of():
    field = Number(multiple_of=5)
    assert field.validate(15) == 15

def test_validate_not_multiple_of():
    field = Number(multiple_of=5)
    try:
        field.validate(16)
        assert False
    except Exception as e:
        assert str(e) == "Must be a multiple of 5."


# LLM-generated content at query #30
#--------------------------

```python
def test_validate_with_null_value_and_allow_null():
    field1 = Field()
    field2 = Field(allow_null=True)
    union_field = Union([field1, field2])
    assert union_field.validate(None) is None

def test_validate_with_null_value_and_not_allow_null():
    field1 = Field()
    field2 = Field()
    union_field = Union([field1, field2])
    try:
        union_field.validate(None)
        assert False
    except ValidationError as e:
        assert e.message() == "May not be null."

def test_validate_with_valid_value():
    field1 = Field()
    field2 = Field()
    field3 = Field()
    union_field = Union([field1, field2, field3])
    assert union_field.validate("valid") == "valid"

def test_validate_with_invalid_value():
    field1 = Field()
    field2 = Field()
    field3 = Field()
    union_field = Union([field1, field2, field3])
    try:
        union_field.validate("invalid")
        assert False
    except ValidationError as e:
        assert e.message() == "Did not match any valid type."

def test_validate_with_candidate_errors():
    field1 = Field()
    field2 = Field()
    field3 = Field()
    union_field = Union([field1, field2, field3])
    try:
        union_field.validate("invalid")
        assert False
    except ValidationError as e:
        assert e.message() == "Did not match any valid type."


# LLM-generated content at query #31
#--------------------------

```python
def test_validate_with_null_value():
    field = Array(allow_null=False)
    try:
        field.validate(None)
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "May not be null."

def test_validate_with_non_list_value():
    field = Array()
    try:
        field.validate("not a list")
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Must be an array."

def test_validate_with_empty_list():
    field = Array(min_items=1)
    try:
        field.validate([])
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Must not be empty."

def test_validate_with_exact_items():
    field = Array(exact_items=3)
    try:
        field.validate([1, 2])
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Must have 3 items."

def test_validate_with_min_items():
    field = Array(min_items=2)
    try:
        field.validate([1])
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Must have at least 2 items."

def test_validate_with_max_items():
    field = Array(max_items=2)
    try:
        field.validate([1, 2, 3])
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Must have no more than 2 items."

def test_validate_with_unique_items():
    field = Array(unique_items=True)
    try:
        field.validate([1, 2, 2])
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Items must be unique."

def test_validate_with_valid_list():
    field = Array()
    result = field.validate([1, 2, 3])
    assert result == [1, 2, 3]

def test_validate_with_additional_items():
    field = Array(items=[Integer()], additional_items=True)
    result = field.validate([1, "2", 3])
    assert result == [1, "2", 3]

def test_validate_with_additional_items_field():
    field = Array(items=[Integer()], additional_items=String())
    result = field.validate([1, "2", 3])
    assert result == [1, "2", 3]

def test_validate_with_item_validation_error():
    field = Array(items=[Integer()])
    try:
        field.validate([1, "2"])
        assert False
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Must be a valid integer."

def test_validate_with_multiple_errors():
    field = Array(min_items=2, max_items=3, unique_items=True)
    try:
        field.validate([1, 2, 3, 4, 1])
        assert False
    except ValidationError as e:
        assert len(e.messages) == 3
        assert e.messages[0].text == "Must have no more than 3 items."
        assert e.messages[1].text == "Items must be unique."


# LLM-generated content at query #32
#--------------------------

```python
def test_unique_items_with_duplicate_values():
    field = Array(unique_items=True)
    value = [1, 2, 2]
    seen_items = Uniqueness([1, 2])
    assert field.validate(value) == None


# LLM-generated content at query #33
#--------------------------

Here are the test cases to ensure the predicate at line 46 evaluates to True:


# LLM-generated content at query #34
#--------------------------

```python
def test_validate_pattern_properties_with_error():
    field = Object(pattern_properties={"^a.*": Field()})
    value = {"abc": "invalid"}
    try:
        field.validate(value)
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "invalid"


# LLM-generated content at query #35
#--------------------------

```python
def test_max_items_not_set_when_additional_items_is_field():
    field = Array(items=[], additional_items=Field())
    assert not hasattr(field, "max_items")


# LLM-generated content at query #36
#--------------------------

```python
def test_validate_empty_string_with_null_and_coerce_types():
    field = Number(allow_null=True, coerce_types=True)
    result = field.validate("")
    assert result is None


# LLM-generated content at query #37
#--------------------------

```python
def test_min_properties_equal_to_one_and_empty_dict():
    field = Object(min_properties=1)
    try:
        field.validate({})
        assert False, "Expected validation error"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "empty"


# LLM-generated content at query #38
#--------------------------

```python
def test_validate_child_schema_with_error():
    from typesystem.fields import Object, String
    from typesystem.exceptions import ValidationError

    schema = Object(properties={"name": String(max_length=5)})
    value = {"name": "too_long_string"}
    _, error = schema.validate_or_error(value)
    assert isinstance(error, ValidationError)


# LLM-generated content at query #39
#--------------------------

Here's the unit test to ensure the predicate at line 20 evaluates to False:


# LLM-generated content at query #40
#--------------------------

```python
def test_pattern_properties_with_non_string_key():
    from typesystem.fields import Object, String
    field = Object(pattern_properties={r"^[a-z]+$": String()})
    value = {123: "value"}
    result = field.validate(value)
    assert result == {}


# LLM-generated content at query #41
#--------------------------

```python
def test_validate_items_not_list_and_not_none():
    field = Field()
    array = Array(items=field)
    value = [1, 2, 3]
    validator = array.validate(value)
    assert validator == value


# LLM-generated content at query #42
#--------------------------

```python
def test_allow_blank_without_default_sets_default_to_empty_string():
    field = String(allow_blank=True)
    assert field.get_default_value() == ""


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```
def test_object_constructor_with_default_values():
    obj = Object()
    assert obj.properties == {}
    assert obj.pattern_properties == {}
    assert obj.additional_properties is True
    assert obj.property_names is None
    assert obj.min_properties is None
    assert obj.max_properties is None
    assert obj.required == []
    assert obj.title == ""
    assert obj.description == ""
    assert obj.allow_null is False
    assert obj.read_only is False

def test_object_constructor_with_custom_properties():
    field = Field()
    obj = Object(properties={"name": field})
    assert obj.properties == {"name": field}
    assert obj.pattern_properties == {}
    assert obj.additional_properties is True
    assert obj.property_names is None
    assert obj.min_properties is None
    assert obj.max_properties is None
    assert obj.required == []

def test_object_constructor_with_pattern_properties():
    field = Field()
    obj = Object(pattern_properties={"^test_": field})
    assert obj.properties == {}
    assert obj.pattern_properties == {"^test_": field}
    assert obj.additional_properties is True
    assert obj.property_names is None
    assert obj.min_properties is None
    assert obj.max_properties is None
    assert obj.required == []

def test_object_constructor_with_additional_properties_field():
    field = Field()
    obj = Object(additional_properties=field)
    assert obj.properties == {}
    assert obj.pattern_properties == {}
    assert obj.additional_properties == field
    assert obj.property_names is None
    assert obj.min_properties is None
    assert obj.max_properties is None
    assert obj.required == []

def test_object_constructor_with_property_names():
    field = Field()
    obj = Object(property_names=field)
    assert obj.properties == {}
    assert obj.pattern_properties == {}
    assert obj.additional_properties is True
    assert obj.property_names == field
    assert obj.min_properties is None
    assert obj.max_properties is None
    assert obj.required == []

def test_object_constructor_with_min_max_properties():
    obj = Object(min_properties=1, max_properties=10)
    assert obj.properties == {}
    assert obj.pattern_properties == {}
    assert obj.additional_properties is True
    assert obj.property_names is None
    assert obj.min_properties == 1
    assert obj.max_properties == 10
    assert obj.required == []

def test_object_constructor_with_required_fields():
    obj = Object(required=["name", "age"])
    assert obj.properties == {}
    assert obj.pattern_properties == {}
    assert obj.additional_properties is True
    assert obj.property_names is None
    assert obj.min_properties is None
    assert obj.max_properties is None
    assert obj.required == ["name", "age"]

def test_object_constructor_with_inherited_fields():
    obj = Object(title="Test", description="Test description", allow_null=True, read_only=True)
    assert obj.properties == {}
    assert obj.pattern_properties == {}
    assert obj.additional_properties is True
    assert obj.property_names is None
    assert obj.min_properties is None
    assert obj.max_properties is None
    assert obj.required == []
    assert obj.title == "Test"
    assert obj.description == "Test description"
    assert obj.allow_null is True
    assert obj.read_only is True


# LLM-generated content at query #2
#--------------------------

```python
def test_validate_null_value():
    field = String(allow_null=True)
    assert field.validate(None) is None

def test_validate_null_value_without_allow_null():
    field = String(allow_null=False)
    try:
        field.validate(None)
        assert False
    except Exception as e:
        assert str(e) == "May not be null."

def test_validate_null_value_with_allow_blank():
    field = String(allow_blank=True, coerce_types=True)
    assert field.validate(None) == ""

def test_validate_non_string_value():
    field = String()
    try:
        field.validate(123)
        assert False
    except Exception as e:
        assert str(e) == "Must be a string."

def test_validate_empty_string_without_allow_blank():
    field = String(allow_blank=False)
    try:
        field.validate("")
        assert False
    except Exception as e:
        assert str(e) == "Must not be blank."

def test_validate_empty_string_with_allow_blank():
    field = String(allow_blank=True)
    assert field.validate("") == ""

def test_validate_string_with_min_length():
    field = String(min_length=5)
    try:
        field.validate("abc")
        assert False
    except Exception as e:
        assert str(e) == "Must have at least 5 characters."

def test_validate_string_with_max_length():
    field = String(max_length=5)
    try:
        field.validate("abcdef")
        assert False
    except Exception as e:
        assert str(e) == "Must have no more than 5 characters."

def test_validate_string_with_pattern():
    field = String(pattern="^[a-z]+$")
    try:
        field.validate("123")
        assert False
    except Exception as e:
        assert str(e) == "Must match the pattern /^[a-z]+$/."

def test_validate_string_with_format():
    field = String(format="email")
    try:
        field.validate("not-an-email")
        assert False
    except Exception as e:
        assert str(e) != "Must be a valid email."

def test_validate_string_with_trim_whitespace():
    field = String(trim_whitespace=True)
    assert field.validate("  abc  ") == "abc"

def test_validate_string_without_trim_whitespace():
    field = String(trim_whitespace=False)
    assert field.validate("  abc  ") == "  abc  "


# LLM-generated content at query #3
#--------------------------

```
def test_properties_values_are_not_all_fields():
    try:
        Object(properties={"key": "not a field"})
    except AssertionError:
        pass
    else:
        assert False, "Expected AssertionError"


# LLM-generated content at query #4
#--------------------------

Here are the test cases for the `validate` method of the `Object` class:


# LLM-generated content at query #5
#--------------------------

```python
def test_validate_null_value():
    field = Array()
    value = None
    try:
        field.validate(value)
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].text == "May not be null."

def test_validate_non_array_value():
    field = Array()
    value = "not an array"
    try:
        field.validate(value)
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].text == "Must be an array."

def test_validate_empty_array():
    field = Array(min_items=1)
    value = []
    try:
        field.validate(value)
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].text == "Must not be empty."

def test_validate_array_with_min_items():
    field = Array(min_items=2)
    value = [1]
    try:
        field.validate(value)
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].text == "Must have at least 2 items."

def test_validate_array_with_max_items():
    field = Array(max_items=2)
    value = [1, 2, 3]
    try:
        field.validate(value)
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].text == "Must have no more than 2 items."

def test_validate_array_with_exact_items():
    field = Array(exact_items=2)
    value = [1]
    try:
        field.validate(value)
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].text == "Must have 2 items."

def test_validate_array_with_unique_items():
    field = Array(unique_items=True)
    value = [1, 1]
    try:
        field.validate(value)
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].text == "Items must be unique."

def test_validate_array_with_valid_items():
    field = Array(items=Integer())
    value = [1, 2, 3]
    validated = field.validate(value)
    assert validated == [1, 2, 3]

def test_validate_array_with_invalid_items():
    field = Array(items=Integer())
    value = [1, "invalid", 3]
    try:
        field.validate(value)
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].text == "Must be a valid integer."

def test_validate_array_with_additional_items():
    field = Array(items=[Integer()], additional_items=True)
    value = [1, 2, 3]
    validated = field.validate(value)
    assert validated == [1, 2, 3]

def test_validate_array_with_additional_items_field():
    field = Array(items=[Integer()], additional_items=String())
    value = [1, "valid"]
    validated = field.validate(value)
    assert validated == [1, "valid"]

def test_validate_array_with_additional_items_field_invalid():
    field = Array(items=[Integer()], additional_items=String())
    value = [1, 2]
    try:
        field.validate(value)
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].text == "Must be a valid string."


# LLM-generated content at query #6
#--------------------------

```python
def test_validate_null_value_allowed():
    field = Number(allow_null=True)
    assert field.validate(None) is None

def test_validate_null_value_not_allowed():
    field = Number(allow_null=False)
    try:
        field.validate(None)
        assert False
    except Exception as e:
        assert str(e) == "May not be null."

def test_validate_boolean_value():
    field = Number()
    try:
        field.validate(True)
        assert False
    except Exception as e:
        assert str(e) == "Must be a number."

def test_validate_string_value_coerce_types():
    field = Number(coerce_types=True)
    assert field.validate("123") == 123

def test_validate_string_value_no_coerce_types():
    field = Number(coerce_types=False)
    try:
        field.validate("123")
        assert False
    except Exception as e:
        assert str(e) == "Must be a number."

def test_validate_integer_value():
    field = Number()
    assert field.validate(123) == 123

def test_validate_float_value():
    field = Number()
    assert field.validate(123.45) == 123.45

def test_validate_non_finite_value():
    field = Number()
    try:
        field.validate(float('inf'))
        assert False
    except Exception as e:
        assert str(e) == "Must be finite."

def test_validate_minimum_constraint():
    field = Number(minimum=10)
    assert field.validate(10) == 10
    try:
        field.validate(9)
        assert False
    except Exception as e:
        assert str(e) == "Must be greater than or equal to 10."

def test_validate_exclusive_minimum_constraint():
    field = Number(exclusive_minimum=10)
    assert field.validate(11) == 11
    try:
        field.validate(10)
        assert False
    except Exception as e:
        assert str(e) == "Must be greater than 10."

def test_validate_maximum_constraint():
    field = Number(maximum=20)
    assert field.validate(20) == 20
    try:
        field.validate(21)
        assert False
    except Exception as e:
        assert str(e) == "Must be less than or equal to 20."

def test_validate_exclusive_maximum_constraint():
    field = Number(exclusive_maximum=20)
    assert field.validate(19) == 19
    try:
        field.validate(20)
        assert False
    except Exception as e:
        assert str(e) == "Must be less than 20."

def test_validate_multiple_of_constraint_int():
    field = Number(multiple_of=5)
    assert field.validate(15) == 15
    try:
        field.validate(16)
        assert False
    except Exception as e:
        assert str(e) == "Must be a multiple of 5."

def test_validate_multiple_of_constraint_float():
    field = Number(multiple_of=0.5)
    assert field.validate(1.5) == 1.5
    try:
        field.validate(1.6)
        assert False
    except Exception as e:
        assert str(e) == "Must be a multiple of 0.5."

def test_validate_precision_constraint():
    field = Number(precision="0.01")
    assert field.validate(123.45) == 123.45
    assert field.validate(123.456) == 123.46

def test_validate_integer_type_constraint():
    field = Number(numeric_type=int)
    assert field.validate(123) == 123
    try:
        field.validate(123.45)
        assert False
    except Exception as e:
        assert str(e) == "Must be an integer."


# LLM-generated content at query #7
#--------------------------

```python
def test_properties_keys_are_not_all_strings():
    class DummyField(Field):
        pass

    properties = {1: DummyField(), "key2": DummyField()}
    Object(properties=properties)


# LLM-generated content at query #8
#--------------------------

```python
def test_additional_properties_is_not_field_instance():
    field = Object(additional_properties=True)
    assert not isinstance(field.additional_properties, Field)


# LLM-generated content at query #9
#--------------------------

```python
def test_string_constructor_default_values():
    field = String()
    assert field.title == ""
    assert field.description == ""
    assert not hasattr(field, "default")
    assert field.allow_null == False
    assert field.read_only == False
    assert field.allow_blank == False
    assert field.trim_whitespace == True
    assert field.max_length == None
    assert field.min_length == None
    assert field.pattern == None
    assert field.format == None
    assert field.coerce_types == True

def test_string_constructor_with_custom_values():
    field = String(
        title="Title",
        description="Description",
        default="default",
        allow_null=True,
        read_only=True,
        allow_blank=True,
        trim_whitespace=False,
        max_length=100,
        min_length=10,
        pattern="pattern",
        format="format",
        coerce_types=False
    )
    assert field.title == "Title"
    assert field.description == "Description"
    assert field.default == "default"
    assert field.allow_null == True
    assert field.read_only == True
    assert field.allow_blank == True
    assert field.trim_whitespace == False
    assert field.max_length == 100
    assert field.min_length == 10
    assert field.pattern == "pattern"
    assert field.format == "format"
    assert field.coerce_types == False

def test_string_constructor_with_allow_blank_sets_default():
    field = String(allow_blank=True)
    assert field.default == ""

def test_string_constructor_with_pattern_as_regex():
    import re
    pattern = re.compile("test")
    field = String(pattern=pattern)
    assert field.pattern == "test"

def test_string_constructor_assertions():
    try:
        String(max_length="invalid")
        assert False
    except AssertionError:
        pass

    try:
        String(min_length="invalid")
        assert False
    except AssertionError:
        pass

    try:
        String(pattern=123)
        assert False
    except AssertionError:
        pass

    try:
        String(format=123)
        assert False
    except AssertionError:
        pas


# LLM-generated content at query #10
#--------------------------

```
def test_choice_constructor_with_default_parameters():
    field = Choice(choices=[("a", "A"), ("b", "B")])
    assert field.title == ""
    assert field.description == ""
    assert not hasattr(field, "default")
    assert field.allow_null is False
    assert field.read_only is False
    assert field.choices == [("a", "A"), ("b", "B")]
    assert field.coerce_types is True

def test_choice_constructor_with_custom_parameters():
    field = Choice(
        title="Test Title",
        description="Test Description",
        default="a",
        allow_null=True,
        read_only=True,
        choices=[("a", "A"), ("b", "B")],
        coerce_types=False
    )
    assert field.title == "Test Title"
    assert field.description == "Test Description"
    assert field.default == "a"
    assert field.allow_null is True
    assert field.read_only is True
    assert field.choices == [("a", "A"), ("b", "B")]
    assert field.coerce_types is False

def test_choice_constructor_with_single_string_choices():
    field = Choice(choices=["a", "b"])
    assert field.choices == [("a", "a"), ("b", "b")]

def test_choice_constructor_with_empty_choices():
    field = Choice(choices=[])
    assert field.choices == []

def test_choice_constructor_with_allow_null_and_default_none():
    field = Choice(choices=[], allow_null=True)
    assert field.allow_null is True
    assert field.default is None


# LLM-generated content at query #11
#--------------------------

```python
def test_validate_returns_none_when_value_is_none_and_allow_null_is_true():
    field = String(allow_null=True)
    result = field.validate(None)
    assert result is None


# LLM-generated content at query #12
#--------------------------

```
def test_validate_returns_none_when_value_is_none_and_allow_null_is_true():
    field = Boolean(allow_null=True)
    assert field.validate(None) is None

def test_validate_raises_null_error_when_value_is_none_and_allow_null_is_false():
    field = Boolean(allow_null=False)
    try:
        field.validate(None)
        assert False
    except Exception as e:
        assert str(e) == "May not be null."

def test_validate_returns_boolean_when_value_is_boolean():
    field = Boolean()
    assert field.validate(True) is True
    assert field.validate(False) is False

def test_validate_raises_type_error_when_value_is_not_boolean_and_coerce_types_is_false():
    field = Boolean(coerce_types=False)
    try:
        field.validate("true")
        assert False
    except Exception as e:
        assert str(e) == "Must be a boolean."

def test_validate_coerces_string_values_to_boolean():
    field = Boolean()
    assert field.validate("true") is True
    assert field.validate("false") is False
    assert field.validate("on") is True
    assert field.validate("off") is False
    assert field.validate("1") is True
    assert field.validate("0") is False
    assert field.validate("") is False

def test_validate_coerces_numeric_values_to_boolean():
    field = Boolean()
    assert field.validate(1) is True
    assert field.validate(0) is False

def test_validate_returns_none_when_value_is_in_coerce_null_values_and_allow_null_is_true():
    field = Boolean(allow_null=True)
    assert field.validate("") is None
    assert field.validate("null") is None
    assert field.validate("none") is None

def test_validate_raises_type_error_when_value_cannot_be_coerced():
    field = Boolean()
    try:
        field.validate("invalid")
        assert False
    except Exception as e:
        assert str(e) == "Must be a boolean."


# LLM-generated content at query #13
#--------------------------

```python
def test_predicate_line_15_evaluates_to_true():
    field = Boolean(allow_null=True)
    result = field.validate("null")
    assert result is None


# LLM-generated content at query #14
#--------------------------

```
def test_array_constructor_with_default_values():
    field = Array()
    assert field.items is None
    assert field.additional_items is False
    assert field.min_items is None
    assert field.max_items is None
    assert field.unique_items is False
    assert field.title == ""
    assert field.description == ""
    assert field.allow_null is False
    assert field.read_only is False

def test_array_constructor_with_items_field():
    item_field = Field()
    field = Array(items=item_field)
    assert field.items == item_field
    assert field.additional_items is False
    assert field.min_items is None
    assert field.max_items is None
    assert field.unique_items is False

def test_array_constructor_with_items_list():
    item_fields = [Field(), Field()]
    field = Array(items=item_fields)
    assert field.items == item_fields
    assert field.min_items == 2
    assert field.max_items == 2
    assert field.additional_items is False

def test_array_constructor_with_additional_items_field():
    additional_field = Field()
    field = Array(additional_items=additional_field)
    assert field.additional_items == additional_field

def test_array_constructor_with_min_max_items():
    field = Array(min_items=1, max_items=10)
    assert field.min_items == 1
    assert field.max_items == 10

def test_array_constructor_with_exact_items():
    field = Array(exact_items=5)
    assert field.min_items == 5
    assert field.max_items == 5

def test_array_constructor_with_unique_items():
    field = Array(unique_items=True)
    assert field.unique_items is True

def test_array_constructor_with_title_and_description():
    field = Array(title="Test", description="Test description")
    assert field.title == "Test"
    assert field.description == "Test description"

def test_array_constructor_with_allow_null():
    field = Array(allow_null=True)
    assert field.allow_null is True

def test_array_constructor_with_read_only():
    field = Array(read_only=True)
    assert field.read_only is True


# LLM-generated content at query #15
#--------------------------

```python
def test_min_properties_equals_one():
    field = Object(min_properties=1)
    value = {}
    try:
        field.validate(value)
    except ValidationError as e:
        assert str(e) == "Must not be empty."


# LLM-generated content at query #16
#--------------------------

def test_validate_null_value_when_null_not_allowed():
    field = Object(allow_null=False)
    try:
        field.validate(None)
        assert False
    except ValidationError as error:
        assert error.messages()[0].text == "May not be null."
        assert error.messages()[0].code == "null"

def test_validate_null_value_when_null_allowed():
    field = Object(allow_null=True)
    assert field.validate(None) is None

def test_validate_non_dict_value():
    field = Object()
    try:
        field.validate("not a dict")
        assert False
    except ValidationError as error:
        assert error.messages()[0].text == "Must be an object."
        assert error.messages()[0].code == "type"

def test_validate_invalid_key_type():
    field = Object()
    try:
        field.validate({1: "value"})
        assert False
    except ValidationError as error:
        assert error.messages()[0].text == "All object keys must be strings."
        assert error.messages()[0].code == "invalid_key"
        assert error.messages()[0].index == [1]

def test_validate_min_properties_error():
    field = Object(min_properties=2)
    try:
        field.validate({"key": "value"})
        assert False
    except ValidationError as error:
        assert error.messages()[0].text == "Must have at least 2 properties."
        assert error.messages()[0].code == "min_properties"

def test_validate_max_properties_error():
    field = Object(max_properties=1)
    try:
        field.validate({"key1": "value1", "key2": "value2"})
        assert False
    except ValidationError as error:
        assert error.messages()[0].text == "Must have no more than 1 properties."
        assert error.messages()[0].code == "max_properties"

def test_validate_required_property_missing():
    field = Object(required=["required_key"])
    try:
        field.validate({"other_key": "value"})
        assert False
    except ValidationError as error:
        assert error.messages()[0].text == "This field is required."
        assert error.messages()[0].code == "required"
        assert error.messages()[0].index == ["required_key"]

def test_validate_property_with_default():
    child_field = Field(default="default_value")
    field = Object(properties={"key": child_field})
    assert field.validate({}) == {"key": "default_value"}

def test_validate_pattern_properties():
    child_field = Field()
    field = Object(pattern_properties={"^test_": child_field})
    assert field.validate({"test_key": "value"}) == {"test_key": "value"}

def test_validate_additional_properties_true():
    field = Object(additional_properties=True)
    assert field.validate({"extra_key": "value"}) == {"extra_key": "value"}

def test_validate_additional_properties_false():
    field = Object(additional_properties=False)
    try:
        field.validate({"extra_key": "value"})
        assert False
    except ValidationError as error:
        assert error.messages()[0].text == "Invalid property name."
        assert error.messages()[0].code == "invalid_property"
        assert error.messages()[0].index == ["extra_key"]

def test_validate_additional_properties_with_field():
    child_field = Field()
    field = Object(additional_properties=child_field)
    assert field.validate({"extra_key": "value"}) == {"extra_key": "value"}

def test_validate_property_names():
    name_field = Field(pattern="^valid_")
    field = Object(property_names=name_field)
    try:
        field.validate({"invalid_name": "value"})
        assert False
    except ValidationError as error:
        assert error.messages()[0].text == "Invalid property name."
        assert error.messages()[0].code == "invalid_property"
        assert error.messages()[0].index == ["invalid_name"]


# LLM-generated content at query #17
#--------------------------

```python
def test_validate_allows_null_when_allow_null_is_true():
    field = Choice(choices=[("a", "A")], allow_null=True)
    assert field.validate(None) is None

def test_validate_raises_null_error_when_allow_null_is_false():
    field = Choice(choices=[("a", "A")], allow_null=False)
    try:
        field.validate(None)
    except Exception as e:
        assert str(e) == "May not be null."

def test_validate_raises_choice_error_when_value_not_in_choices():
    field = Choice(choices=[("a", "A")])
    try:
        field.validate("b")
    except Exception as e:
        assert str(e) == "Not a valid choice."

def test_validate_returns_value_when_value_in_choices():
    field = Choice(choices=[("a", "A")])
    assert field.validate("a") == "a"

def test_validate_raises_required_error_when_value_is_empty_string():
    field = Choice(choices=[("a", "A")], allow_null=False, coerce_types=True)
    try:
        field.validate("")
    except Exception as e:
        assert str(e) == "This field is required."

def test_validate_returns_null_when_value_is_empty_string_and_coerce_types_is_true_and_allow_null_is_true():
    field = Choice(choices=[("a", "A")], allow_null=True, coerce_types=True)
    assert field.validate("") is None

def test_validate_handles_tuple_choices():
    field = Choice(choices=[(("a", "b"), "AB")])
    assert field.validate(("a", "b")) == ("a", "b")

def test_validate_handles_list_choices():
    field = Choice(choices=[(["a", "b"], "AB")])
    assert field.validate(["a", "b"]) == ["a", "b"]

def test_validate_handles_dict_choices():
    field = Choice(choices=[({"a": "b"}, "AB")])
    assert field.validate({"a": "b"}) == {"a": "b"}

def test_validate_handles_mixed_choices():
    field = Choice(choices=[("a", "A"), (1, "One"), (True, "True")])
    assert field.validate("a") == "a"
    assert field.validate(1) == 1
    assert field.validate(True) is True


# LLM-generated content at query #18
#--------------------------

```
def test_validate_null_value():
    field = Number(allow_null=True)
    assert field.validate(None) is None

def test_validate_null_value_with_coerce():
    field = Number(allow_null=True, coerce_types=True)
    assert field.validate("") is None

def test_validate_null_value_not_allowed():
    field = Number(allow_null=False)
    try:
        field.validate(None)
        assert False
    except Exception as e:
        assert str(e) == "May not be null."

def test_validate_boolean_value():
    field = Number()
    try:
        field.validate(True)
        assert False
    except Exception as e:
        assert str(e) == "Must be a number."

def test_validate_invalid_integer():
    field = Number(numeric_type=int)
    try:
        field.validate(1.23)
        assert False
    except Exception as e:
        assert str(e) == "Must be an integer."

def test_validate_non_numeric_string():
    field = Number(coerce_types=False)
    try:
        field.validate("abc")
        assert False
    except Exception as e:
        assert str(e) == "Must be a number."

def test_validate_numeric_string():
    field = Number()
    assert field.validate("123") == 123

def test_validate_infinite_value():
    field = Number()
    try:
        field.validate(float("inf"))
        assert False
    except Exception as e:
        assert str(e) == "Must be finite."

def test_validate_minimum_constraint():
    field = Number(minimum=10)
    assert field.validate(10) == 10
    try:
        field.validate(9)
        assert False
    except Exception as e:
        assert str(e) == "Must be greater than or equal to 10."

def test_validate_exclusive_minimum_constraint():
    field = Number(exclusive_minimum=10)
    assert field.validate(11) == 11
    try:
        field.validate(10)
        assert False
    except Exception as e:
        assert str(e) == "Must be greater than 10."

def test_validate_maximum_constraint():
    field = Number(maximum=100)
    assert field.validate(100) == 100
    try:
        field.validate(101)
        assert False
    except Exception as e:
        assert str(e) == "Must be less than or equal to 100."

def test_validate_exclusive_maximum_constraint():
    field = Number(exclusive_maximum=100)
    assert field.validate(99) == 99
    try:
        field.validate(100)
        assert False
    except Exception as e:
        assert str(e) == "Must be less than 100."

def test_validate_multiple_of_constraint_int():
    field = Number(multiple_of=5)
    assert field.validate(10) == 10
    try:
        field.validate(12)
        assert False
    except Exception as e:
        assert str(e) == "Must be a multiple of 5."

def test_validate_multiple_of_constraint_float():
    field = Number(multiple_of=0.5)
    assert field.validate(1.0) == 1.0
    try:
        field.validate(1.2)
        assert False
    except Exception as e:
        assert str(e) == "Must be a multiple of 0.5."

def test_validate_precision():
    field = Number(precision="0.00")
    assert field.validate(1.234) == 1.23
    assert field.validate(1.235) == 1.24


# LLM-generated content at query #19
#--------------------------

```python
def test_array_constructor_with_default_parameters():
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

def test_array_constructor_with_items():
    item_field = Field()
    field = Array(items=item_field)
    assert field.items == item_field

def test_array_constructor_with_items_list():
    item_fields = [Field(), Field()]
    field = Array(items=item_fields)
    assert field.items == item_fields

def test_array_constructor_with_additional_items():
    field = Array(additional_items=True)
    assert field.additional_items is True

def test_array_constructor_with_additional_items_field():
    item_field = Field()
    field = Array(additional_items=item_field)
    assert field.additional_items == item_field

def test_array_constructor_with_min_items():
    field = Array(min_items=5)
    assert field.min_items == 5

def test_array_constructor_with_max_items():
    field = Array(max_items=10)
    assert field.max_items == 10

def test_array_constructor_with_exact_items():
    field = Array(exact_items=3)
    assert field.min_items == 3
    assert field.max_items == 3

def test_array_constructor_with_unique_items():
    field = Array(unique_items=True)
    assert field.unique_items is True

def test_array_constructor_with_allow_null():
    field = Array(allow_null=True)
    assert field.allow_null is True

def test_array_constructor_with_read_only():
    field = Array(read_only=True)
    assert field.read_only is True

def test_array_constructor_with_title():
    field = Array(title="Test Title")
    assert field.title == "Test Title"

def test_array_constructor_with_description():
    field = Array(description="Test Description")
    assert field.description == "Test Description"


# LLM-generated content at query #20
#--------------------------

```
def test_const_constructor_with_const_value():
    field = Const(const=42, title="Answer", description="The answer to everything")
    assert field.const == 42
    assert field.title == "Answer"
    assert field.description == "The answer to everything"
    assert not field.allow_null
    assert not field.read_only

def test_const_constructor_with_default_values():
    field = Const(const="test")
    assert field.const == "test"
    assert field.title == ""
    assert field.description == ""
    assert not field.allow_null
    assert not field.read_only

def test_const_constructor_disallows_allow_null():
    try:
        Const(const=None, allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass

def test_const_constructor_with_null_const():
    field = Const(const=None)
    assert field.const is None
    assert not field.allow_null


# LLM-generated content at query #21
#--------------------------

```python
def test_validate_null_when_allow_null_is_true():
    field = Union(any_of=[Field(allow_null=True)])
    assert field.validate(None) is None

def test_validate_null_when_allow_null_is_false():
    field = Union(any_of=[Field(allow_null=False)])
    try:
        field.validate(None)
        assert False
    except Exception as e:
        assert str(e) == "May not be null."

def test_validate_with_valid_value():
    field = Union(any_of=[Field()])
    assert field.validate("valid") == "valid"

def test_validate_with_invalid_value():
    field = Union(any_of=[Field()])
    try:
        field.validate("invalid")
        assert False
    except Exception as e:
        assert str(e) == "Did not match any valid type."

def test_validate_with_multiple_fields_and_valid_value():
    field = Union(any_of=[Field(), Field()])
    assert field.validate("valid") == "valid"

def test_validate_with_multiple_fields_and_invalid_value():
    field = Union(any_of=[Field(), Field()])
    try:
        field.validate("invalid")
        assert False
    except Exception as e:
        assert str(e) == "Did not match any valid type."

def test_validate_with_candidate_error():
    field1 = Field()
    field2 = Field()
    field = Union(any_of=[field1, field2])
    try:
        field.validate("invalid")
        assert False
    except Exception as e:
        assert str(e) == "Did not match any valid type."


# LLM-generated content at query #22
#--------------------------

```python
def test_min_items_not_none_if_items_is_list():
    field = Array(items=[Field()], min_items=5)
    assert field.min_items == 5


# LLM-generated content at query #23
#--------------------------

```python
def test_predicate_at_line_20_evaluates_to_false():
    field = Boolean(coerce_types=True, allow_null=False)
    value = "invalid_value"
    try:
        field.validate(value)
    except Exception as e:
        assert str(e) == "Must be a boolean."


# LLM-generated content at query #24
#--------------------------

def test_validate_null_value_when_not_allowed():
    field = Object(allow_null=False)
    try:
        field.validate(None)
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].text == "May not be null."
        assert e.messages()[0].code == "null"

def test_validate_null_value_when_allowed():
    field = Object(allow_null=True)
    assert field.validate(None) is None

def test_validate_non_dict_value():
    field = Object()
    try:
        field.validate("not a dict")
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].text == "Must be an object."
        assert e.messages()[0].code == "type"

def test_validate_invalid_key_type():
    field = Object()
    try:
        field.validate({1: "value"})
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].text == "All object keys must be strings."
        assert e.messages()[0].code == "invalid_key"
        assert e.messages()[0].index == [1]

def test_validate_min_properties():
    field = Object(min_properties=2)
    try:
        field.validate({"key": "value"})
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert "Must have at least 2 properties." in e.messages()[0].text
        assert e.messages()[0].code == "min_properties"

def test_validate_max_properties():
    field = Object(max_properties=1)
    try:
        field.validate({"key1": "value1", "key2": "value2"})
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert "Must have no more than 1 properties." in e.messages()[0].text
        assert e.messages()[0].code == "max_properties"

def test_validate_required_property_missing():
    field = Object(required=["required_key"])
    try:
        field.validate({"other_key": "value"})
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].text == "This field is required."
        assert e.messages()[0].code == "required"
        assert e.messages()[0].index == ["required_key"]

def test_validate_property_with_default():
    child_field = Field(default="default_value")
    field = Object(properties={"key": child_field})
    assert field.validate({}) == {"key": "default_value"}

def test_validate_pattern_properties():
    child_field = Field()
    field = Object(pattern_properties={"^test_": child_field})
    assert field.validate({"test_key": "value"}) == {"test_key": "value"}

def test_validate_additional_properties_true():
    field = Object(additional_properties=True)
    assert field.validate({"key": "value"}) == {"key": "value"}

def test_validate_additional_properties_false():
    field = Object(additional_properties=False)
    try:
        field.validate({"key": "value"})
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].text == "Invalid property name."
        assert e.messages()[0].code == "invalid_property"
        assert e.messages()[0].index == ["key"]

def test_validate_additional_properties_with_field():
    child_field = Field()
    field = Object(additional_properties=child_field)
    assert field.validate({"key": "value"}) == {"key": "value"}

def test_validate_multiple_errors():
    field = Object(required=["req_key"], additional_properties=False)
    try:
        field.validate({"invalid_key": 123, 456: "value"})
        assert False
    except ValidationError as e:
        assert len(e.messages()) == 3
        codes = {msg.code for msg in e.messages()}
        assert codes == {"invalid_key", "invalid_property", "required"}


# LLM-generated content at query #25
#--------------------------

```python
def test_validate_allow_null_with_null_value():
    field = Boolean(allow_null=True)
    assert field.validate(None) is None


# LLM-generated content at query #26
#--------------------------

```python
def test_validate_numeric_type_int_with_float_non_integer():
    field = Number(numeric_type=int)
    value = 3.14
    try:
        field.validate(value)
    except Exception as e:
        assert str(e) == "Must be an integer


# LLM-generated content at query #27
#--------------------------

```
def test_validate_float_with_non_integer_value_when_numeric_type_is_int():
    field = Number(numeric_type=int)
    try:
        field.validate(3.14)
    except Exception as e:
        assert str(e) == "Must be an integer


# LLM-generated content at query #28
#--------------------------

```
def test_validate_returns_none_when_value_is_null_and_allow_null_is_true():
    field = Union(any_of=[], allow_null=True)
    assert field.validate(None) is None

def test_validate_raises_null_error_when_value_is_null_and_allow_null_is_false():
    field = Union(any_of=[])
    try:
        field.validate(None)
        assert False
    except Exception as error:
        assert str(error) == "May not be null."

def test_validate_returns_validated_value_when_one_child_validates_successfully():
    class MockField:
        def validate_or_error(self, value):
            return value, None
    field = Union(any_of=[MockField()])
    assert field.validate("test") == "test"

def test_validate_raises_child_error_when_one_child_has_non_type_error():
    class MockField:
        def validate_or_error(self, value):
            return None, Exception("child error")
    field = Union(any_of=[MockField()])
    try:
        field.validate("test")
        assert False
    except Exception as error:
        assert str(error) == "child error"

def test_validate_raises_union_error_when_no_child_validates_successfully():
    class MockField:
        def validate_or_error(self, value):
            return None, Exception("type error")
    field = Union(any_of=[MockField()])
    try:
        field.validate("test")
        assert False
    except Exception as error:
        assert str(error) == "Did not match any valid type."


# LLM-generated content at query #29
#--------------------------

```python
def test_get_default_value_returns_default_value():
    field = Field(default=42)
    assert field.get_default_value() == 42

def test_get_default_value_returns_callable_result():
    field = Field(default=lambda: 100)
    assert field.get_default_value() == 100

def test_get_default_value_returns_none_when_no_default():
    field = Field()
    assert field.get_default_value() is None


# LLM-generated content at query #30
#--------------------------

```python
def test_validate_null_allowed():
    field = Number(allow_null=True)
    assert field.validate(None) is None

def test_validate_null_not_allowed():
    field = Number(allow_null=False)
    try:
        field.validate(None)
        assert False
    except Exception as e:
        assert str(e) == "May not be null."

def test_validate_non_numeric_type():
    field = Number()
    try:
        field.validate("not a number")
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

def test_validate_integer_required():
    field = Number(numeric_type=int)
    try:
        field.validate(3.14)
        assert False
    except Exception as e:
        assert str(e) == "Must be an integer."

def test_validate_non_finite():
    field = Number()
    try:
        field.validate(float('inf'))
        assert False
    except Exception as e:
        assert str(e) == "Must be finite."

def test_validate_minimum():
    field = Number(minimum=10)
    try:
        field.validate(5)
        assert False
    except Exception as e:
        assert str(e) == "Must be greater than or equal to 10."

def test_validate_exclusive_minimum():
    field = Number(exclusive_minimum=10)
    try:
        field.validate(10)
        assert False
    except Exception as e:
        assert str(e) == "Must be greater than 10."

def test_validate_maximum():
    field = Number(maximum=10)
    try:
        field.validate(15)
        assert False
    except Exception as e:
        assert str(e) == "Must be less than or equal to 10."

def test_validate_exclusive_maximum():
    field = Number(exclusive_maximum=10)
    try:
        field.validate(10)
        assert False
    except Exception as e:
        assert str(e) == "Must be less than 10."

def test_validate_multiple_of_int():
    field = Number(multiple_of=5)
    try:
        field.validate(7)
        assert False
    except Exception as e:
        assert str(e) == "Must be a multiple of 5."

def test_validate_multiple_of_float():
    field = Number(multiple_of=0.5)
    try:
        field.validate(0.7)
        assert False
    except Exception as e:
        assert str(e) == "Must be a multiple of 0.5."

def test_validate_coerce_string():
    field = Number(coerce_types=True)
    assert field.validate("42") == 42

def test_validate_precision():
    field = Number(precision="0.01")
    assert field.validate(3.14159) == 3.14


# LLM-generated content at query #31
#--------------------------

```python
def test_predicate_at_line_42_evaluates_to_false():
    field = Array(items=Field())
    value = [1]
    validated_value = field.validate(value)
    assert validated_value == [1]


# LLM-generated content at query #32
#--------------------------

```python
def test_validate_with_null_value_and_allow_null():
    field = String(allow_null=True)
    assert field.validate(None) is None


# LLM-generated content at query #33
#--------------------------

```
def test_validate_float_with_non_integer_value_when_numeric_type_is_int():
    field = Number(numeric_type=int)
    try:
        field.validate(3.14)
    except Exception as e:
        assert str(e) == "Must be an integer."


# LLM-generated content at query #34
#--------------------------

```python
def test_validate_does_not_raise_type_error_for_valid_numeric_type_conversion():
    field = Number(numeric_type=int, coerce_types=True)
    field.validate("42")


# LLM-generated content at query #35
#--------------------------

```python
def test_validate_non_none_value():
    obj = Object(allow_null=True)
    value = {"key": "value"}
    result = obj.validate(value)
    assert result == value


# LLM-generated content at query #36
#--------------------------

```python
def test_choice_validate_null_allowed():
    field = Choice(choices=[("a", "A"), ("b", "B")], allow_null=True)
    assert field.validate(None) is None

def test_choice_validate_null_not_allowed():
    field = Choice(choices=[("a", "A"), ("b", "B")], allow_null=False)
    try:
        field.validate(None)
        assert False
    except Exception as exc:
        assert str(exc) == "May not be null."

def test_choice_validate_valid_choice():
    field = Choice(choices=[("a", "A"), ("b", "B")])
    assert field.validate("a") == "a"

def test_choice_validate_invalid_choice():
    field = Choice(choices=[("a", "A"), ("b", "B")])
    try:
        field.validate("c")
        assert False
    except Exception as exc:
        assert str(exc) == "Not a valid choice."

def test_choice_validate_empty_string_null_allowed():
    field = Choice(choices=[("a", "A"), ("b", "B")], allow_null=True, coerce_types=True)
    assert field.validate("") is None

def test_choice_validate_empty_string_null_not_allowed():
    field = Choice(choices=[("a", "A"), ("b", "B")], allow_null=False)
    try:
        field.validate("")
        assert False
    except Exception as exc:
        assert str(exc) == "This field is required."


# LLM-generated content at query #37
#--------------------------

```python
def test_validate_with_null_and_allow_null_true():
    field = Array(allow_null=True)
    result = field.validate(None)
    assert result is None


