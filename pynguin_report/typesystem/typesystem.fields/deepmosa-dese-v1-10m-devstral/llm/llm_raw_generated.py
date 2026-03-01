####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_validate_with_none_value_and_allow_null():
    field = Object(allow_null=True)
    assert field.validate(None) is None

def test_validate_with_none_value_and_not_allow_null():
    field = Object(allow_null=False)
    try:
        field.validate(None)
    except ValidationError as e:
        assert e.messages == [Message(text="May not be null.", code="null")]

def test_validate_with_non_dict_value():
    field = Object()
    try:
        field.validate("not a dict")
    except ValidationError as e:
        assert e.messages == [Message(text="Must be an object.", code="type")]

def test_validate_with_invalid_key_type():
    field = Object()
    try:
        field.validate({123: "value"})
    except ValidationError as e:
        assert e.messages == [Message(text="All object keys must be strings.", code="invalid_key", index=[123])]

def test_validate_with_invalid_property_name():
    field = Object(property_names=String())
    try:
        field.validate({"invalid_key!": "value"})
    except ValidationError as e:
        assert e.messages == [Message(text="Invalid property name.", code="invalid_property", index=["invalid_key!"])]

def test_validate_with_min_properties():
    field = Object(min_properties=2)
    try:
        field.validate({"key": "value"})
    except ValidationError as e:
        assert e.messages == [Message(text="Must have at least 2 properties.", code="min_properties")]

def test_validate_with_max_properties():
    field = Object(max_properties=1)
    try:
        field.validate({"key1": "value1", "key2": "value2"})
    except ValidationError as e:
        assert e.messages == [Message(text="Must have no more than 1 properties.", code="max_properties")]

def test_validate_with_required_property_missing():
    field = Object(required=["required_key"])
    try:
        field.validate({"other_key": "value"})
    except ValidationError as e:
        assert e.messages == [Message(text="This field is required.", code="required", index=["required_key"])]

def test_validate_with_property_default_value():
    field = Object(properties={"key": String(default="default_value")})
    assert field.validate({}) == {"key": "default_value"}

def test_validate_with_pattern_properties():
    field = Object(pattern_properties={r"^test_": String()})
    assert field.validate({"test_key": "value"}) == {"test_key": "value"}

def test_validate_with_invalid_additional_property():
    field = Object(additional_properties=False, properties={"allowed_key": String()})
    try:
        field.validate({"allowed_key": "value", "invalid_key": "value"})
    except ValidationError as e:
        assert e.messages == [Message(text="Invalid property name.", code="invalid_property", key="invalid_key")]

def test_validate_with_additional_properties_field():
    field = Object(additional_properties=String(), properties={"allowed_key": String()})
    assert field.validate({"allowed_key": "value", "additional_key": "value"}) == {"allowed_key": "value", "additional_key": "value"}

def test_validate_with_valid_input():
    field = Object(properties={"key": String()})
    assert field.validate({"key": "value"}) == {"key": "value"}


# LLM-generated content at query #2
#--------------------------

```python
def test_validate_with_none_and_allow_null():
    string_field = String(allow_null=True)
    assert string_field.validate(None) is None

def test_validate_with_none_and_allow_blank():
    string_field = String(allow_blank=True)
    assert string_field.validate(None) == ""

def test_validate_with_none_and_no_allow_null_or_blank():
    string_field = String()
    try:
        string_field.validate(None)
        assert False, "Expected validation error"
    except Exception as e:
        assert str(e) == "May not be null."

def test_validate_with_non_string_type():
    string_field = String()
    try:
        string_field.validate(123)
        assert False, "Expected validation error"
    except Exception as e:
        assert str(e) == "Must be a string."

def test_validate_with_null_character():
    string_field = String()
    assert string_field.validate("a\0b") == "ab"

def test_validate_with_trim_whitespace():
    string_field = String(trim_whitespace=True)
    assert string_field.validate("  hello  ") == "hello"

def test_validate_with_blank_and_no_allow_blank():
    string_field = String(allow_blank=False)
    try:
        string_field.validate("")
        assert False, "Expected validation error"
    except Exception as e:
        assert str(e) == "Must not be blank."

def test_validate_with_blank_and_allow_blank():
    string_field = String(allow_blank=True)
    assert string_field.validate("") == ""

def test_validate_with_min_length():
    string_field = String(min_length=3)
    try:
        string_field.validate("ab")
        assert False, "Expected validation error"
    except Exception as e:
        assert str(e) == "Must have at least 3 characters."

def test_validate_with_max_length():
    string_field = String(max_length=3)
    try:
        string_field.validate("abcd")
        assert False, "Expected validation error"
    except Exception as e:
        assert str(e) == "Must have no more than 3 characters."

def test_validate_with_pattern():
    string_field = String(pattern=r"^[a-z]+$")
    try:
        string_field.validate("123")
        assert False, "Expected validation error"
    except Exception as e:
        assert str(e) == "Must match the pattern /^[a-z]+$/."

def test_validate_with_format():
    string_field = String(format="email")
    try:
        string_field.validate("invalid-email")
        assert False, "Expected validation error"
    except Exception as e:
        assert str(e) == "Must be a valid email."

def test_validate_with_valid_string():
    string_field = String()
    assert string_field.validate("valid string") == "valid string"


# LLM-generated content at query #3
#--------------------------

```python
def test_validate_with_none_and_allow_null():
    choice = Choice(choices=[("a", "a")], allow_null=True)
    assert choice.validate(None) is None

def test_validate_with_none_and_not_allow_null():
    choice = Choice(choices=[("a", "a")], allow_null=False)
    try:
        choice.validate(None)
        assert False, "Expected validation error"
    except ValidationError as e:
        assert e.detail == "May not be null."

def test_validate_with_empty_string_and_allow_null_and_coerce_types():
    choice = Choice(choices=[("a", "a")], allow_null=True, coerce_types=True)
    assert choice.validate("") is None

def test_validate_with_empty_string_and_allow_null_and_not_coerce_types():
    choice = Choice(choices=[("a", "a")], allow_null=True, coerce_types=False)
    try:
        choice.validate("")
        assert False, "Expected validation error"
    except ValidationError as e:
        assert e.detail == "This field is required."

def test_validate_with_empty_string_and_not_allow_null():
    choice = Choice(choices=[("a", "a")], allow_null=False)
    try:
        choice.validate("")
        assert False, "Expected validation error"
    except ValidationError as e:
        assert e.detail == "This field is required."

def test_validate_with_valid_choice():
    choice = Choice(choices=[("a", "a"), ("b", "b")])
    assert choice.validate("a") == "a"

def test_validate_with_invalid_choice():
    choice = Choice(choices=[("a", "a"), ("b", "b")])
    try:
        choice.validate("c")
        assert False, "Expected validation error"
    except ValidationError as e:
        assert e.detail == "Not a valid choice."

def test_validate_with_boolean_true_and_coerce_types():
    choice = Choice(choices=[("a", "a"), (True, True)], coerce_types=True)
    assert choice.validate(True) == True

def test_validate_with_boolean_false_and_coerce_types():
    choice = Choice(choices=[("a", "a"), (False, False)], coerce_types=True)
    assert choice.validate(False) == False

def test_validate_with_integer_one_and_coerce_types():
    choice = Choice(choices=[("a", "a"), (1, 1)], coerce_types=True)
    assert choice.validate(1) == 1

def test_validate_with_integer_zero_and_coerce_types():
    choice = Choice(choices=[("a", "a"), (0, 0)], coerce_types=True)
    assert choice.validate(0) == 0

def test_validate_with_list_choice():
    choice = Choice(choices=[("a", "a"), (["b", "c"], ["b", "c"])])

    assert choice.validate(["b", "c"]) == ["b", "c"]

def test_validate_with_dict_choice():
    choice = Choice(choices=[("a", "a"), ({"b": "c"}, {"b": "c"})])
    assert choice.validate({"b": "c"}) == {"b": "c"}


# LLM-generated content at query #4
#--------------------------

```python
def test_validate_with_none_and_allow_null():
    field = Boolean(allow_null=True)
    assert field.validate(None) is None

def test_validate_with_none_and_not_allow_null():
    field = Boolean(allow_null=False)
    try:
        field.validate(None)
    except Exception as e:
        assert str(e) == "May not be null."

def test_validate_with_bool_value():
    field = Boolean()
    assert field.validate(True) is True
    assert field.validate(False) is False

def test_validate_with_non_bool_and_coerce_disabled():
    field = Boolean(coerce_types=False)
    try:
        field.validate("true")
    except Exception as e:
        assert str(e) == "Must be a boolean."

def test_validate_with_string_true():
    field = Boolean()
    assert field.validate("true") is True

def test_validate_with_string_false():
    field = Boolean()
    assert field.validate("false") is False

def test_validate_with_string_on():
    field = Boolean()
    assert field.validate("on") is True

def test_validate_with_string_off():
    field = Boolean()
    assert field.validate("off") is False

def test_validate_with_string_1():
    field = Boolean()
    assert field.validate("1") is True

def test_validate_with_string_0():
    field = Boolean()
    assert field.validate("0") is False

def test_validate_with_empty_string():
    field = Boolean()
    assert field.validate("") is False

def test_validate_with_integer_1():
    field = Boolean()
    assert field.validate(1) is True

def test_validate_with_integer_0():
    field = Boolean()
    assert field.validate(0) is False

def test_validate_with_null_string_and_allow_null():
    field = Boolean(allow_null=True)
    assert field.validate("null") is None

def test_validate_with_none_string_and_allow_null():
    field = Boolean(allow_null=True)
    assert field.validate("none") is None

def test_validate_with_empty_string_and_allow_null():
    field = Boolean(allow_null=True)
    assert field.validate("") is None

def test_validate_with_invalid_string():
    field = Boolean()
    try:
        field.validate("invalid")
    except Exception as e:
        assert str(e) == "Must be a boolean."

def test_validate_with_invalid_type():
    field = Boolean()
    try:
        field.validate([1, 2, 3])
    except Exception as e:
        assert str(e) == "Must be a boolean."


# LLM-generated content at query #5
#--------------------------

```python
def test_array_constructor_with_valid_parameters():
    items = [Field(), Field()]
    additional_items = Field()
    min_items = 1
    max_items = 10
    exact_items = 5
    unique_items = True
    title = "Test Array"
    description = "A test array field"
    default = [1, 2, 3]
    allow_null = True
    read_only = True

    array = Array(
        items=items,
        additional_items=additional_items,
        min_items=min_items,
        max_items=max_items,
        exact_items=exact_items,
        unique_items=unique_items,
        title=title,
        description=description,
        default=default,
        allow_null=allow_null,
        read_only=read_only,
    )

    assert array.items == items
    assert array.additional_items == additional_items
    assert array.min_items == exact_items
    assert array.max_items == exact_items
    assert array.unique_items == unique_items
    assert array.title == title
    assert array.description == description
    assert array.default == default
    assert array.allow_null == allow_null
    assert array.read_only == read_only

def test_array_constructor_with_single_item():
    item = Field()
    array = Array(items=item)
    assert array.items == item
    assert array.min_items is None
    assert array.max_items is None

def test_array_constructor_with_list_items():
    items = [Field(), Field()]
    array = Array(items=items)
    assert array.items == items
    assert array.min_items == len(items)
    assert array.max_items == len(items)

def test_array_constructor_with_additional_items_false():
    items = [Field(), Field()]
    array = Array(items=items, additional_items=False)
    assert array.additional_items is False
    assert array.min_items == len(items)
    assert array.max_items == len(items)

def test_array_constructor_with_exact_items():
    array = Array(exact_items=5)
    assert array.min_items == 5
    assert array.max_items == 5

def test_array_constructor_with_no_items():
    array = Array()
    assert array.items is None
    assert array.additional_items is False
    assert array.min_items is None
    assert array.max_items is None
    assert array.unique_items is False

def test_array_constructor_with_allow_null_and_no_default():
    array = Array(allow_null=True)
    assert array.allow_null is True
    assert array.default is None

def test_array_constructor_with_default_value():
    default = [1, 2, 3]
    array = Array(default=default)
    assert array.default == default

def test_array_constructor_with_callable_default():
    def get_default():
        return [1, 2, 3]

    array = Array(default=get_default)
    assert callable(array.default)


# LLM-generated content at query #6
#--------------------------

```python
def test_validate_with_none_and_allow_null():
    union = Union(any_of=[StringField()], allow_null=True)
    assert union.validate(None) is None

def test_validate_with_none_and_not_allow_null():
    union = Union(any_of=[StringField()], allow_null=False)
    try:
        union.validate(None)
    except ValidationError as e:
        assert e.messages() == [{"code": "null", "message": "May not be null."}]

def test_validate_with_matching_child():
    union = Union(any_of=[StringField(), IntegerField()])
    assert union.validate("test") == "test"

def test_validate_with_non_matching_child():
    union = Union(any_of=[StringField(), IntegerField()])
    try:
        union.validate(["test"])
    except ValidationError as e:
        assert e.messages() == [{"code": "union", "message": "Did not match any valid type."}]

def test_validate_with_candidate_error():
    union = Union(any_of=[StringField(min_length=5), IntegerField()])
    try:
        union.validate("test")
    except ValidationError as e:
        assert e.messages() == [{"code": "min_length", "message": "Shorter than minimum length 5."}]

def test_validate_with_multiple_candidate_errors():
    union = Union(any_of=[StringField(min_length=5), IntegerField(min_value=10)])
    try:
        union.validate("test")
    except ValidationError as e:
        assert e.messages() == [{"code": "union", "message": "Did not match any valid type."}]


# LLM-generated content at query #7
#--------------------------

```python
def test_object_constructor_with_defaults():
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

def test_object_constructor_with_properties():
    field = Field()
    properties = {"key": field}
    obj = Object(properties=properties)
    assert obj.properties == properties
    assert obj.pattern_properties == {}
    assert obj.additional_properties is True
    assert obj.property_names is None
    assert obj.min_properties is None
    assert obj.max_properties is None
    assert obj.required == []

def test_object_constructor_with_pattern_properties():
    field = Field()
    pattern_properties = {"pattern": field}
    obj = Object(pattern_properties=pattern_properties)
    assert obj.pattern_properties == pattern_properties
    assert obj.properties == {}
    assert obj.additional_properties is True
    assert obj.property_names is None
    assert obj.min_properties is None
    assert obj.max_properties is None
    assert obj.required == []

def test_object_constructor_with_additional_properties():
    field = Field()
    obj = Object(additional_properties=field)
    assert obj.additional_properties == field
    assert obj.properties == {}
    assert obj.pattern_properties == {}
    assert obj.property_names is None
    assert obj.min_properties is None
    assert obj.max_properties is None
    assert obj.required == []

def test_object_constructor_with_property_names():
    field = Field()
    obj = Object(property_names=field)
    assert obj.property_names == field
    assert obj.properties == {}
    assert obj.pattern_properties == {}
    assert obj.additional_properties is True
    assert obj.min_properties is None
    assert obj.max_properties is None
    assert obj.required == []

def test_object_constructor_with_min_properties():
    obj = Object(min_properties=1)
    assert obj.min_properties == 1
    assert obj.properties == {}
    assert obj.pattern_properties == {}
    assert obj.additional_properties is True
    assert obj.property_names is None
    assert obj.max_properties is None
    assert obj.required == []

def test_object_constructor_with_max_properties():
    obj = Object(max_properties=10)
    assert obj.max_properties == 10
    assert obj.properties == {}
    assert obj.pattern_properties == {}
    assert obj.additional_properties is True
    assert obj.property_names is None
    assert obj.min_properties is None
    assert obj.required == []

def test_object_constructor_with_required():
    obj = Object(required=["key"])
    assert obj.required == ["key"]
    assert obj.properties == {}
    assert obj.pattern_properties == {}
    assert obj.additional_properties is True
    assert obj.property_names is None
    assert obj.min_properties is None
    assert obj.max_properties is None

def test_object_constructor_with_inherited_attributes():
    obj = Object(title="Test", description="Description", allow_null=True, read_only=True)
    assert obj.title == "Test"
    assert obj.description == "Description"
    assert obj.allow_null is True
    assert obj.read_only is True
    assert obj.properties == {}
    assert obj.pattern_properties == {}
    assert obj.additional_properties is True
    assert obj.property_names is None
    assert obj.min_properties is None
    assert obj.max_properties is None
    assert obj.required == []

def test_object_constructor_with_properties_as_field():
    field = Field()
    obj = Object(properties=field)
    assert obj.properties == {}
    assert obj.additional_properties == field
    assert obj.pattern_properties == {}
    assert obj.property_names is None
    assert obj.min_properties is None
    assert obj.max_properties is None
    assert obj.required == []


# LLM-generated content at query #8
#--------------------------

```python
def test_array_init_max_items_not_set_when_additional_items_is_not_false():
    field = Array(items=[Field(), Field()], additional_items=True)
    assert field.max_items is None


# LLM-generated content at query #9
#--------------------------

```python
def test_array_validate_with_none_and_allow_null():
    field = Array(allow_null=True)
    assert field.validate(None) is None

def test_array_validate_with_none_and_not_allow_null():
    field = Array(allow_null=False)
    try:
        field.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "May not be null."

def test_array_validate_with_non_list():
    field = Array()
    try:
        field.validate("not a list")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Must be an array."

def test_array_validate_with_exact_items():
    field = Array(exact_items=3)
    assert field.validate([1, 2, 3]) == [1, 2, 3]

def test_array_validate_with_exact_items_failure():
    field = Array(exact_items=3)
    try:
        field.validate([1, 2])
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Must have 3 items."

def test_array_validate_with_min_items():
    field = Array(min_items=2)
    assert field.validate([1, 2, 3]) == [1, 2, 3]

def test_array_validate_with_min_items_failure():
    field = Array(min_items=2)
    try:
        field.validate([1])
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Must have at least 2 items."

def test_array_validate_with_min_items_one():
    field = Array(min_items=1)
    try:
        field.validate([])
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Must not be empty."

def test_array_validate_with_max_items():
    field = Array(max_items=2)
    assert field.validate([1, 2]) == [1, 2]

def test_array_validate_with_max_items_failure():
    field = Array(max_items=2)
    try:
        field.validate([1, 2, 3])
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Must have no more than 2 items."

def test_array_validate_with_unique_items():
    field = Array(unique_items=True)
    assert field.validate([1, 2, 3]) == [1, 2, 3]

def test_array_validate_with_unique_items_failure():
    field = Array(unique_items=True)
    try:
        field.validate([1, 2, 2])
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Items must be unique."

def test_array_validate_with_items_field():
    field = Array(items=Integer())
    assert field.validate([1, 2, 3]) == [1, 2, 3]

def test_array_validate_with_items_field_failure():
    field = Array(items=Integer())
    try:
        field.validate([1, "not an integer", 3])
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Must be an integer."

def test_array_validate_with_items_list():
    field = Array(items=[Integer(), String()])
    assert field.validate([1, "two"]) == [1, "two"]

def test_array_validate_with_items_list_failure():
    field = Array(items=[Integer(), String()])
    try:
        field.validate([1, 2])
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Must be a string."

def test_array_validate_with_additional_items_field():
    field = Array(items=[Integer()], additional_items=String())
    assert field.validate([1, "two", "three"]) == [1, "two", "three"]

def test_array_validate_with_additional_items_false():
    field = Array(items=[Integer()], additional_items=False)
    try:
        field.validate([1, "two"])
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "May not contain additional items."

def test_array_validate_with_no_items():
    field = Array()
    assert field.validate([1, 2, 3]) == [1, 2, 3]


# LLM-generated content at query #10
#--------------------------

```python
def test_validate_with_none_and_allow_null():
    number = Number(allow_null=True)
    assert number.validate(None) is None

def test_validate_with_empty_string_and_allow_null_and_coerce_types():
    number = Number(allow_null=True, coerce_types=True)
    assert number.validate("") is None

def test_validate_with_none_and_not_allow_null():
    number = Number(allow_null=False)
    try:
        number.validate(None)
        assert False, "Expected validation error"
    except Exception as e:
        assert str(e) == "May not be null."

def test_validate_with_boolean():
    number = Number()
    try:
        number.validate(True)
        assert False, "Expected validation error"
    except Exception as e:
        assert str(e) == "Must be a number."

def test_validate_with_float_and_numeric_type_int():
    number = Number(numeric_type=int)
    try:
        number.validate(3.14)
        assert False, "Expected validation error"
    except Exception as e:
        assert str(e) == "Must be an integer."

def test_validate_with_string_and_coerce_types():
    number = Number(coerce_types=True)
    assert number.validate("42") == 42

def test_validate_with_infinite_value():
    number = Number()
    try:
        number.validate(float('inf'))
        assert False, "Expected validation error"
    except Exception as e:
        assert str(e) == "Must be finite."

def test_validate_with_precision():
    number = Number(precision="0.01")
    assert number.validate(3.14159) == 3.14

def test_validate_with_minimum():
    number = Number(minimum=10)
    try:
        number.validate(5)
        assert False, "Expected validation error"
    except Exception as e:
        assert str(e) == "Must be greater than or equal to 10."

def test_validate_with_exclusive_minimum():
    number = Number(exclusive_minimum=10)
    try:
        number.validate(10)
        assert False, "Expected validation error"
    except Exception as e:
        assert str(e) == "Must be greater than 10."

def test_validate_with_maximum():
    number = Number(maximum=100)
    try:
        number.validate(105)
        assert False, "Expected validation error"
    except Exception as e:
        assert str(e) == "Must be less than or equal to 100."

def test_validate_with_exclusive_maximum():
    number = Number(exclusive_maximum=100)
    try:
        number.validate(100)
        assert False, "Expected validation error"
    except Exception as e:
        assert str(e) == "Must be less than 100."

def test_validate_with_multiple_of():
    number = Number(multiple_of=5)
    try:
        number.validate(7)
        assert False, "Expected validation error"
    except Exception as e:
        assert str(e) == "Must be a multiple of 5."

def test_validate_with_valid_value():
    number = Number()
    assert number.validate(42) == 42


# LLM-generated content at query #11
#--------------------------

```python
def test_validate_returns_none_when_value_is_none_and_allow_null_is_true():
    field = Array(allow_null=True)
    result = field.validate(None)
    assert result is None


# LLM-generated content at query #12
#--------------------------

```python
def test_string_constructor_with_defaults():
    field = String()
    assert field.title == ""
    assert field.description == ""
    assert field.allow_null == False
    assert field.read_only == False
    assert field.allow_blank == False
    assert field.trim_whitespace == True
    assert field.max_length is None
    assert field.min_length is None
    assert field.pattern is None
    assert field.pattern_regex is None
    assert field.format is None
    assert field.coerce_types == True
    assert not field.has_default()

def test_string_constructor_with_custom_values():
    field = String(
        title="Test Title",
        description="Test Description",
        default="default_value",
        allow_null=True,
        read_only=True,
        allow_blank=True,
        trim_whitespace=False,
        max_length=100,
        min_length=10,
        pattern=r"^[a-z]+$",
        format="email",
        coerce_types=False,
    )
    assert field.title == "Test Title"
    assert field.description == "Test Description"
    assert field.default == "default_value"
    assert field.allow_null == True
    assert field.read_only == True
    assert field.allow_blank == True
    assert field.trim_whitespace == False
    assert field.max_length == 100
    assert field.min_length == 10
    assert field.pattern == r"^[a-z]+$"
    assert field.pattern_regex.pattern == r"^[a-z]+$"
    assert field.format == "email"
    assert field.coerce_types == False
    assert field.has_default() == True

def test_string_constructor_with_pattern_regex():
    import re
    pattern = re.compile(r"^[0-9]+$")
    field = String(pattern=pattern)
    assert field.pattern == r"^[0-9]+$"
    assert field.pattern_regex == pattern

def test_string_constructor_with_allow_blank_and_no_default():
    field = String(allow_blank=True)
    assert field.default == ""
    assert field.has_default() == True

def test_string_constructor_with_allow_null_and_no_default():
    field = String(allow_null=True)
    assert field.default is None
    assert field.has_default() == True


# LLM-generated content at query #13
#--------------------------

```python
def test_allow_null_and_value_in_coerce_null_values():
    field = Boolean(allow_null=True)
    assert field.validate("null") is None


# LLM-generated content at query #14
#--------------------------

```python
def test_validate_with_none_value_and_allow_null():
    field = Object(allow_null=True)
    assert field.validate(None) is None

def test_validate_with_none_value_and_not_allow_null():
    field = Object(allow_null=False)
    try:
        field.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "May not be null."
        assert e.messages[0].code == "null"

def test_validate_with_non_dict_value():
    field = Object()
    try:
        field.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Must be an object."
        assert e.messages[0].code == "type"

def test_validate_with_invalid_key_type():
    field = Object()
    try:
        field.validate({123: "value"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "All object keys must be strings."
        assert e.messages[0].code == "invalid_key"
        assert e.messages[0].index == [123]

def test_validate_with_invalid_property_name():
    field = Object(property_names=String())
    try:
        field.validate({"invalid@key": "value"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Invalid property name."
        assert e.messages[0].code == "invalid_property"
        assert e.messages[0].index == ["invalid@key"]

def test_validate_with_min_properties_not_met():
    field = Object(min_properties=2)
    try:
        field.validate({"key": "value"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Must have at least 2 properties."
        assert e.messages[0].code == "min_properties"

def test_validate_with_min_properties_equal_to_one_and_empty():
    field = Object(min_properties=1)
    try:
        field.validate({})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Must not be empty."
        assert e.messages[0].code == "empty"

def test_validate_with_max_properties_exceeded():
    field = Object(max_properties=2)
    try:
        field.validate({"key1": "value1", "key2": "value2", "key3": "value3"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Must have no more than 2 properties."
        assert e.messages[0].code == "max_properties"

def test_validate_with_missing_required_property():
    field = Object(required=["required_key"])
    try:
        field.validate({"other_key": "value"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "This field is required."
        assert e.messages[0].code == "required"
        assert e.messages[0].index == ["required_key"]

def test_validate_with_property_validation_error():
    field = Object(properties={"key": String(max_length=5)})
    try:
        field.validate({"key": "too long value"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "max_length"
        assert e.messages[0].index == ["key"]

def test_validate_with_pattern_property_validation_error():
    field = Object(pattern_properties={r"^test_": String(max_length=5)})
    try:
        field.validate({"test_key": "too long value"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "max_length"
        assert e.messages[0].index == ["test_key"]

def test_validate_with_additional_properties_false():
    field = Object(properties={"allowed_key": String()}, additional_properties=False)
    try:
        field.validate({"allowed_key": "value", "extra_key": "extra_value"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Invalid property name."
        assert e.messages[0].code == "invalid_property"
        assert e.messages[0].key == "extra_key"

def test_validate_with_additional_properties_field_validation_error():
    field = Object(properties={"allowed_key": String()}, additional_properties=String(max_length=5))
    try:
        field.validate({"allowed_key": "value", "extra_key": "too long value"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "max_length"
        assert e.messages[0].index == ["extra_key"]

def test_validate_with_valid_input():
    field = Object(
        properties={
            "name": String(),
            "age": Integer(),
        },
        required=["name"],
        additional_properties=False,
    )
    result = field.validate({"name": "John", "age": 30})
    assert result == {"name": "John", "age": 30}

def test_validate_with_default_value():
    field = Object(properties={"optional": String(default="default_value")})
    result = field.validate({})
    assert result == {"optional": "default_value"}


# LLM-generated content at query #15
#--------------------------

```python
def test_validate_with_invalid_choice():
    choice_field = Choice(choices=[("a", "Option A"), ("b", "Option B")], allow_null=False, coerce_types=True)
    with pytest.raises(ValidationError) as exc_info:
        choice_field.validate("invalid")
    assert exc_info.value.detail == "Not a valid choice."


# LLM-generated content at query #16
#--------------------------

```python
def test_validate_with_none_and_allow_null():
    field = Array(allow_null=True)
    assert field.validate(None) is None

def test_validate_with_none_and_not_allow_null():
    field = Array(allow_null=False)
    try:
        field.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "May not be null."

def test_validate_with_non_list_value():
    field = Array()
    try:
        field.validate("not a list")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Must be an array."

def test_validate_with_exact_items():
    field = Array(exact_items=3)
    assert field.validate([1, 2, 3]) == [1, 2, 3]

def test_validate_with_exact_items_failure():
    field = Array(exact_items=3)
    try:
        field.validate([1, 2])
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Must have 3 items."

def test_validate_with_min_items():
    field = Array(min_items=2)
    assert field.validate([1, 2, 3]) == [1, 2, 3]

def test_validate_with_min_items_failure():
    field = Array(min_items=2)
    try:
        field.validate([1])
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Must have at least 2 items."

def test_validate_with_min_items_one():
    field = Array(min_items=1)
    try:
        field.validate([])
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Must not be empty."

def test_validate_with_max_items():
    field = Array(max_items=3)
    assert field.validate([1, 2, 3]) == [1, 2, 3]

def test_validate_with_max_items_failure():
    field = Array(max_items=2)
    try:
        field.validate([1, 2, 3])
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Must have no more than 2 items."

def test_validate_with_items_field():
    field = Array(items=Integer())
    assert field.validate([1, 2, 3]) == [1, 2, 3]

def test_validate_with_items_field_failure():
    field = Array(items=Integer())
    try:
        field.validate([1, "not an integer", 3])
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Must be an integer."

def test_validate_with_items_list():
    field = Array(items=[Integer(), String()])
    assert field.validate([1, "two"]) == [1, "two"]

def test_validate_with_items_list_failure():
    field = Array(items=[Integer(), String()])
    try:
        field.validate([1, 2])
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Must be a string."

def test_validate_with_additional_items_field():
    field = Array(items=[Integer()], additional_items=String())
    assert field.validate([1, "two", "three"]) == [1, "two", "three"]

def test_validate_with_additional_items_false():
    field = Array(items=[Integer()], additional_items=False)
    try:
        field.validate([1, "two"])
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "May not contain additional items."

def test_validate_with_unique_items():
    field = Array(unique_items=True)
    assert field.validate([1, 2, 3]) == [1, 2, 3]

def test_validate_with_unique_items_failure():
    field = Array(unique_items=True)
    try:
        field.validate([1, 2, 2])
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Items must be unique."


# LLM-generated content at query #17
#--------------------------

```python
def test_union_raises_single_candidate_error():
    child1 = Field()
    child1.validate_or_error = lambda value: (None, ValidationError("error1", code="type", index=None))
    child2 = Field()
    child2.validate_or_error = lambda value: (None, ValidationError("error2", code="other", index=None))
    union = Union([child1, child2])
    with pytest.raises(ValidationError) as excinfo:
        union.validate("invalid")
    assert str(excinfo.value) == "error2"


# LLM-generated content at query #18
#--------------------------

```python
def test_validate_with_none_and_allow_null():
    number = Number(allow_null=True)
    assert number.validate(None) is None

def test_validate_with_empty_string_and_allow_null_and_coerce_types():
    number = Number(allow_null=True, coerce_types=True)
    assert number.validate("") is None

def test_validate_with_none_and_not_allow_null():
    number = Number(allow_null=False)
    try:
        number.validate(None)
        assert False, "Expected validation error"
    except Exception as e:
        assert str(e) == "May not be null."

def test_validate_with_boolean():
    number = Number()
    try:
        number.validate(True)
        assert False, "Expected validation error"
    except Exception as e:
        assert str(e) == "Must be a number."

def test_validate_with_float_and_numeric_type_int():
    number = Number(numeric_type=int)
    try:
        number.validate(3.14)
        assert False, "Expected validation error"
    except Exception as e:
        assert str(e) == "Must be an integer."

def test_validate_with_string_and_no_coerce_types():
    number = Number(coerce_types=False)
    try:
        number.validate("123")
        assert False, "Expected validation error"
    except Exception as e:
        assert str(e) == "Must be a number."

def test_validate_with_invalid_string():
    number = Number()
    try:
        number.validate("abc")
        assert False, "Expected validation error"
    except Exception as e:
        assert str(e) == "Must be a number."

def test_validate_with_infinite_value():
    number = Number()
    try:
        number.validate(float('inf'))
        assert False, "Expected validation error"
    except Exception as e:
        assert str(e) == "Must be finite."

def test_validate_with_precision():
    number = Number(precision="0.01")
    assert number.validate("3.14159") == 3.14

def test_validate_with_minimum():
    number = Number(minimum=5)
    try:
        number.validate(3)
        assert False, "Expected validation error"
    except Exception as e:
        assert str(e) == "Must be greater than or equal to 5."

def test_validate_with_exclusive_minimum():
    number = Number(exclusive_minimum=5)
    try:
        number.validate(5)
        assert False, "Expected validation error"
    except Exception as e:
        assert str(e) == "Must be greater than 5."

def test_validate_with_maximum():
    number = Number(maximum=10)
    try:
        number.validate(15)
        assert False, "Expected validation error"
    except Exception as e:
        assert str(e) == "Must be less than or equal to 10."

def test_validate_with_exclusive_maximum():
    number = Number(exclusive_maximum=10)
    try:
        number.validate(10)
        assert False, "Expected validation error"
    except Exception as e:
        assert str(e) == "Must be less than 10."

def test_validate_with_multiple_of_int():
    number = Number(multiple_of=3)
    try:
        number.validate(5)
        assert False, "Expected validation error"
    except Exception as e:
        assert str(e) == "Must be a multiple of 3."

def test_validate_with_multiple_of_float():
    number = Number(multiple_of=0.5)
    try:
        number.validate(1.2)
        assert False, "Expected validation error"
    except Exception as e:
        assert str(e) == "Must be a multiple of 0.5."

def test_validate_with_valid_input():
    number = Number()
    assert number.validate(42) == 42
    assert number.validate("42") == 42
    assert number.validate(3.14) == 3.14


# LLM-generated content at query #19
#--------------------------

```python
def test_predicate_evaluates_to_true():
    class MockField:
        def __init__(self, allow_null=False):
            self.allow_null = allow_null

        def validate_or_error(self, value):
            class MockError:
                def messages(self):
                    return [MockMessage(), MockMessage()]

            class MockMessage:
                code = "type"
                index = None

            return None, MockError()

    union = Union([MockField()])
    union.any_of[0].validate_or_error = lambda value: (None, MockError())
    assert len(union.any_of[0].validate_or_error(None)[1].messages()) != 1


# LLM-generated content at query #20
#--------------------------

```python
def test_validate_with_none_and_allow_null():
    field = Object(allow_null=True)
    assert field.validate(None) is None

def test_validate_with_none_and_not_allow_null():
    field = Object(allow_null=False)
    try:
        field.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "null"

def test_validate_with_non_dict():
    field = Object()
    try:
        field.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "type"

def test_validate_with_non_string_keys():
    field = Object()
    try:
        field.validate({123: "value"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "invalid_key"

def test_validate_with_invalid_property_names():
    field = Object(property_names=String())
    try:
        field.validate({"123": "value"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "invalid_property"

def test_validate_with_min_properties():
    field = Object(min_properties=2)
    try:
        field.validate({"a": 1})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "min_properties"

def test_validate_with_max_properties():
    field = Object(max_properties=1)
    try:
        field.validate({"a": 1, "b": 2})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "max_properties"

def test_validate_with_required_properties():
    field = Object(required=["a"])
    try:
        field.validate({"b": 2})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "required"

def test_validate_with_properties():
    field = Object(properties={"a": Integer()})
    result = field.validate({"a": "123"})
    assert result == {"a": 123}

def test_validate_with_pattern_properties():
    field = Object(pattern_properties={r"^a.*": Integer()})
    result = field.validate({"a": "123", "abc": "456"})
    assert result == {"a": 123, "abc": 456}

def test_validate_with_additional_properties_false():
    field = Object(properties={"a": Integer()}, additional_properties=False)
    try:
        field.validate({"a": 123, "b": 456})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "invalid_property"

def test_validate_with_additional_properties_field():
    field = Object(properties={"a": Integer()}, additional_properties=String())
    result = field.validate({"a": 123, "b": 456})
    assert result == {"a": 123, "b": "456"}

def test_validate_with_default_values():
    field = Object(properties={"a": Integer(default=100)})
    result = field.validate({})
    assert result == {"a": 100}

def test_validate_with_empty_object_and_min_properties_1():
    field = Object(min_properties=1)
    try:
        field.validate({})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "empty"


# LLM-generated content at query #21
#--------------------------

```python
def test_union_predicate_evaluates_to_true():
    union = Union(any_of=[Field()])
    error = ValidationError({"messages": [{"code": "not_type", "index": None}]})
    messages = error.messages()
    assert (len(messages) != 1 or messages[0].code != "type" or messages[0].index)


# LLM-generated content at query #22
#--------------------------

```python
def test_array_init_with_list_items_and_min_items_not_none():
    field = Array(items=[Field(), Field()], min_items=1)
    assert field.min_items == 1


# LLM-generated content at query #23
#--------------------------

```python
def test_validate_raises_error_for_invalid_value():
    field = Boolean(coerce_types=True, allow_null=False)
    with pytest.raises(Exception) as excinfo:
        field.validate("invalid")
    assert str(excinfo.value) == "Must be a boolean."


# LLM-generated content at query #24
#--------------------------

```python
def test_serialize_with_none_value():
    array = Array()
    assert array.serialize(None) is None

def test_serialize_with_empty_list():
    array = Array()
    assert array.serialize([]) == []

def test_serialize_with_single_field():
    field = Field()
    array = Array(items=field)
    obj = [1, 2, 3]
    assert array.serialize(obj) == [field.serialize(1), field.serialize(2), field.serialize(3)]

def test_serialize_with_multiple_fields():
    field1 = Field()
    field2 = Field()
    array = Array(items=[field1, field2])
    obj = [1, 2]
    assert array.serialize(obj) == [field1.serialize(1), field2.serialize(2)]

def test_serialize_with_additional_items_field():
    field = Field()
    additional_field = Field()
    array = Array(items=[field], additional_items=additional_field)
    obj = [1, 2, 3]
    assert array.serialize(obj) == [field.serialize(1), additional_field.serialize(2), additional_field.serialize(3)]

def test_serialize_with_no_items_field():
    array = Array(items=None)
    obj = [1, 2, 3]
    assert array.serialize(obj) == obj


# LLM-generated content at query #25
#--------------------------

```python
def test_validate_with_none_and_allow_null():
    field = Object(allow_null=True)
    result = field.validate(None)
    assert result is None

def test_validate_with_none_and_not_allow_null():
    field = Object(allow_null=False)
    try:
        field.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "May not be null."
        assert e.messages[0].code == "null"

def test_validate_with_non_dict_value():
    field = Object()
    try:
        field.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Must be an object."
        assert e.messages[0].code == "type"

def test_validate_with_non_string_keys():
    field = Object()
    try:
        field.validate({123: "value"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "All object keys must be strings."
        assert e.messages[0].code == "invalid_key"
        assert e.messages[0].index == [123]

def test_validate_with_invalid_property_name():
    field = Object(property_names=String())
    try:
        field.validate({"123": "value"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Invalid property name."
        assert e.messages[0].code == "invalid_property"
        assert e.messages[0].index == ["123"]

def test_validate_with_min_properties():
    field = Object(min_properties=2)
    try:
        field.validate({"a": 1})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Must have at least 2 properties."
        assert e.messages[0].code == "min_properties"

def test_validate_with_min_properties_equal_to_one():
    field = Object(min_properties=1)
    try:
        field.validate({})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Must not be empty."
        assert e.messages[0].code == "empty"

def test_validate_with_max_properties():
    field = Object(max_properties=2)
    try:
        field.validate({"a": 1, "b": 2, "c": 3})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Must have no more than 2 properties."
        assert e.messages[0].code == "max_properties"

def test_validate_with_required_properties():
    field = Object(required=["a", "b"])
    try:
        field.validate({"a": 1})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "This field is required."
        assert e.messages[0].code == "required"
        assert e.messages[0].index == ["b"]

def test_validate_with_properties():
    field = Object(properties={"a": Integer(), "b": String()})
    result = field.validate({"a": 1, "b": "hello"})
    assert result == {"a": 1, "b": "hello"}

def test_validate_with_properties_and_invalid_value():
    field = Object(properties={"a": Integer()})
    try:
        field.validate({"a": "not an integer"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Must be an integer."
        assert e.messages[0].code == "type"
        assert e.messages[0].index == ["a"]

def test_validate_with_pattern_properties():
    field = Object(pattern_properties={r"^a.*$": Integer()})
    result = field.validate({"a": 1, "ab": 2})
    assert result == {"a": 1, "ab": 2}

def test_validate_with_pattern_properties_and_invalid_value():
    field = Object(pattern_properties={r"^a.*$": Integer()})
    try:
        field.validate({"a": "not an integer"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Must be an integer."
        assert e.messages[0].code == "type"
        assert e.messages[0].index == ["a"]

def test_validate_with_additional_properties_false():
    field = Object(properties={"a": Integer()}, additional_properties=False)
    try:
        field.validate({"a": 1, "b": 2})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Invalid property name."
        assert e.messages[0].code == "invalid_property"
        assert e.messages[0].key == "b"

def test_validate_with_additional_properties_as_field():
    field = Object(properties={"a": Integer()}, additional_properties=String())
    result = field.validate({"a": 1, "b": "hello"})
    assert result == {"a": 1, "b": "hello"}

def test_validate_with_additional_properties_as_field_and_invalid_value():
    field = Object(properties={"a": Integer()}, additional_properties=String())
    try:
        field.validate({"a": 1, "b": 123})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Must be a string."
        assert e.messages[0].code == "type"
        assert e.messages[0].index == ["b"]

def test_validate_with_default_values():
    field = Object(properties={"a": Integer(default=0)})
    result = field.validate({})
    assert result == {"a": 0}

def test_validate_with_multiple_errors():
    field = Object(
        properties={"a": Integer()},
        required=["a", "b"],
        additional_properties=False
    )
    try:
        field.validate({"a": "not an integer", "c": 1})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 3
        assert any(msg.text == "Must be an integer." and msg.index == ["a"] for msg in e.messages)
        assert any(msg.text == "This field is required." and msg.index == ["b"] for msg in e.messages)
        assert any(msg.text == "Invalid property name." and msg.key == "c" for msg in e.messages)


# LLM-generated content at query #26
#--------------------------

```python
def test_numeric_type_is_int_and_value_is_non_integer_float():
    number = Number(numeric_type=int)
    assert number.numeric_type is int
    assert isinstance(1.5, float)
    assert not (1.5).is_integer()


# LLM-generated content at query #27
#--------------------------

```python
def test_pattern_properties_predicate_false():
    field = Object(pattern_properties={"^a": String()})
    value = {"b": "test"}
    assert field.validate(value) == {"b": "test"}


# LLM-generated content at query #28
#--------------------------

```python
def test_validate_with_null_value_and_allow_null():
    union = Union(any_of=[String()], allow_null=True)
    assert union.validate(None) is None

def test_validate_with_null_value_and_no_allow_null():
    union = Union(any_of=[String()])
    try:
        union.validate(None)
        assert False, "Expected validation error"
    except ValidationError as e:
        assert e.messages()[0].code == "null"

def test_validate_with_matching_child():
    union = Union(any_of=[String(), Integer()])
    assert union.validate("test") == "test"

def test_validate_with_non_matching_children():
    union = Union(any_of=[String(), Integer()])
    try:
        union.validate(3.14)
        assert False, "Expected validation error"
    except ValidationError as e:
        assert e.messages()[0].code == "union"

def test_validate_with_one_candidate_error():
    union = Union(any_of=[String(min_length=5), Integer()])
    try:
        union.validate("abc")
        assert False, "Expected validation error"
    except ValidationError as e:
        assert e.messages()[0].code == "min_length"

def test_validate_with_multiple_candidate_errors():
    union = Union(any_of=[String(min_length=5), Integer(min_value=10)])
    try:
        union.validate("abc")
        assert False, "Expected validation error"
    except ValidationError as e:
        assert e.messages()[0].code == "union"


# LLM-generated content at query #29
#--------------------------

```python
def test_validate_returns_none_for_null_coerce_value():
    boolean_field = Boolean(allow_null=True)
    assert boolean_field.validate("null") is None


# LLM-generated content at query #30
#--------------------------

```python
def test_validate_with_invalid_string_raises_type_error():
    number_field = Number(numeric_type=int, coerce_types=True)
    with pytest.raises(Exception) as excinfo:
        number_field.validate("invalid")
    assert "type" in str(excinfo.value)


# LLM-generated content at query #31
#--------------------------

```python
def test_validate_returns_none_when_value_is_none_and_allow_null_is_true():
    number = Number(allow_null=True)
    assert number.validate(None) is None


# LLM-generated content at query #32
#--------------------------

```python
def test_validate_empty_string_with_allow_null_and_coerce_types():
    field = String(allow_null=True, coerce_types=True)
    result = field.validate("")
    assert result is None


# LLM-generated content at query #33
#--------------------------

```python
def test_validate_returns_none_when_value_is_none_and_allow_null_is_true():
    number = Number(allow_null=True)
    assert number.validate(None) is None


# LLM-generated content at query #34
#--------------------------

```python
def test_allow_blank_sets_default_when_no_default_exists():
    field = String(allow_blank=True)
    assert field.default == ""


# LLM-generated content at query #35
#--------------------------

```python
def test_validate_returns_none_when_value_is_none_and_allow_null_is_true():
    field = Object(allow_null=True)
    result = field.validate(None)
    assert result is None


# LLM-generated content at query #36
#--------------------------

```python
def test_validate_with_none_value_and_allow_null():
    field = Object(allow_null=True)
    assert field.validate(None) is None

def test_validate_with_none_value_and_not_allow_null():
    field = Object(allow_null=False)
    try:
        field.validate(None)
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "May not be null."
        assert e.messages[0].code == "null"

def test_validate_with_non_dict_value():
    field = Object()
    try:
        field.validate("not a dict")
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Must be an object."
        assert e.messages[0].code == "type"

def test_validate_with_empty_dict_and_min_properties():
    field = Object(min_properties=1)
    try:
        field.validate({})
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Must not be empty."
        assert e.messages[0].code == "empty"

def test_validate_with_dict_exceeding_max_properties():
    field = Object(max_properties=2)
    try:
        field.validate({"a": 1, "b": 2, "c": 3})
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Must have no more than 2 properties."
        assert e.messages[0].code == "max_properties"

def test_validate_with_missing_required_property():
    field = Object(required=["username"])
    try:
        field.validate({"email": "test@example.com"})
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "This field is required."
        assert e.messages[0].code == "required"
        assert e.messages[0].index == ["username"]

def test_validate_with_non_string_key():
    field = Object()
    try:
        field.validate({123: "value"})
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "All object keys must be strings."
        assert e.messages[0].code == "invalid_key"
        assert e.messages[0].index == [123]

def test_validate_with_invalid_property_name():
    field = Object(property_names=String(min_length=5))
    try:
        field.validate({"abc": "value"})
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Invalid property name."
        assert e.messages[0].code == "invalid_property"
        assert e.messages[0].index == ["abc"]

def test_validate_with_additional_properties_false():
    field = Object(properties={"username": String()}, additional_properties=False)
    try:
        field.validate({"username": "test", "extra": "value"})
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Invalid property name."
        assert e.messages[0].code == "invalid_property"
        assert e.messages[0].key == "extra"

def test_validate_with_additional_properties_as_field():
    field = Object(properties={"username": String()}, additional_properties=String())
    result = field.validate({"username": "test", "extra": "value"})
    assert result == {"username": "test", "extra": "value"}

def test_validate_with_pattern_properties():
    field = Object(pattern_properties={r"^user_": String()})
    result = field.validate({"user_name": "test", "other": 123})
    assert result == {"user_name": "test"}

def test_validate_with_valid_input():
    field = Object(properties={"username": String(), "age": Integer()})
    result = field.validate({"username": "test", "age": 25})
    assert result == {"username": "test", "age": 25}

def test_validate_with_default_values():
    field = Object(properties={"username": String(default="guest"), "age": Integer()})
    result = field.validate({"age": 25})
    assert result == {"username": "guest", "age": 25}


# LLM-generated content at query #37
#--------------------------

```python
def test_get_default_value_with_non_callable_default():
    field = Field(default=42)
    assert field.get_default_value() == 42

def test_get_default_value_with_callable_default():
    field = Field(default=lambda: 42)
    assert field.get_default_value() == 42

def test_get_default_value_without_default():
    field = Field()
    assert field.get_default_value() is None

def test_get_default_value_with_none_default():
    field = Field(default=None)
    assert field.get_default_value() is None


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_validate_none_with_allow_null():
    number = Number(allow_null=True)
    assert number.validate(None) is None

def test_validate_empty_string_with_allow_null_and_coerce_types():
    number = Number(allow_null=True, coerce_types=True)
    assert number.validate("") is None

def test_validate_none_without_allow_null():
    number = Number(allow_null=False)
    try:
        number.validate(None)
        assert False, "Expected validation error"
    except Exception as e:
        assert str(e) == "May not be null."

def test_validate_boolean():
    number = Number()
    try:
        number.validate(True)
        assert False, "Expected validation error"
    except Exception as e:
        assert str(e) == "Must be a number."

def test_validate_float_non_integer_with_int_type():
    number = Number(numeric_type=int)
    try:
        number.validate(3.14)
        assert False, "Expected validation error"
    except Exception as e:
        assert str(e) == "Must be an integer."

def test_validate_non_numeric_without_coerce_types():
    number = Number(coerce_types=False)
    try:
        number.validate("123")
        assert False, "Expected validation error"
    except Exception as e:
        assert str(e) == "Must be a number."

def test_validate_invalid_string():
    number = Number()
    try:
        number.validate("abc")
        assert False, "Expected validation error"
    except Exception as e:
        assert str(e) == "Must be a number."

def test_validate_infinite_value():
    number = Number()
    try:
        number.validate(float('inf'))
        assert False, "Expected validation error"
    except Exception as e:
        assert str(e) == "Must be finite."

def test_validate_precision():
    number = Number(precision="0.00")
    assert number.validate(3.14159) == 3.14

def test_validate_minimum():
    number = Number(minimum=5)
    try:
        number.validate(3)
        assert False, "Expected validation error"
    except Exception as e:
        assert str(e) == "Must be greater than or equal to 5."

def test_validate_exclusive_minimum():
    number = Number(exclusive_minimum=5)
    try:
        number.validate(5)
        assert False, "Expected validation error"
    except Exception as e:
        assert str(e) == "Must be greater than 5."

def test_validate_maximum():
    number = Number(maximum=10)
    try:
        number.validate(15)
        assert False, "Expected validation error"
    except Exception as e:
        assert str(e) == "Must be less than or equal to 10."

def test_validate_exclusive_maximum():
    number = Number(exclusive_maximum=10)
    try:
        number.validate(10)
        assert False, "Expected validation error"
    except Exception as e:
        assert str(e) == "Must be less than 10."

def test_validate_multiple_of_int():
    number = Number(multiple_of=3)
    try:
        number.validate(5)
        assert False, "Expected validation error"
    except Exception as e:
        assert str(e) == "Must be a multiple of 3."

def test_validate_multiple_of_float():
    number = Number(multiple_of=0.5)
    try:
        number.validate(1.2)
        assert False, "Expected validation error"
    except Exception as e:
        assert str(e) == "Must be a multiple of 0.5."

def test_validate_valid_integer():
    number = Number()
    assert number.validate(42) == 42

def test_validate_valid_float():
    number = Number()
    assert number.validate(3.14) == 3.14

def test_validate_valid_string():
    number = Number()
    assert number.validate("123") == 123

def test_validate_valid_multiple_of():
    number = Number(multiple_of=5)
    assert number.validate(10) == 10


# LLM-generated content at query #2
#--------------------------

```python
def test_validate_with_none_value_and_allow_null():
    field = String(allow_null=True)
    assert field.validate(None) is None

def test_validate_with_none_value_and_allow_blank():
    field = String(allow_blank=True)
    assert field.validate(None) == ""

def test_validate_with_none_value_and_no_allow_null_or_blank():
    field = String()
    try:
        field.validate(None)
        assert False, "Expected validation error"
    except Exception as e:
        assert str(e) == "May not be null."

def test_validate_with_non_string_value():
    field = String()
    try:
        field.validate(123)
        assert False, "Expected validation error"
    except Exception as e:
        assert str(e) == "Must be a string."

def test_validate_with_null_character():
    field = String()
    assert field.validate("a\0b") == "ab"

def test_validate_with_trim_whitespace():
    field = String(trim_whitespace=True)
    assert field.validate("  hello  ") == "hello"

def test_validate_with_blank_value_and_no_allow_blank():
    field = String()
    try:
        field.validate("")
        assert False, "Expected validation error"
    except Exception as e:
        assert str(e) == "Must not be blank."

def test_validate_with_blank_value_and_allow_blank():
    field = String(allow_blank=True)
    assert field.validate("") == ""

def test_validate_with_min_length():
    field = String(min_length=3)
    try:
        field.validate("ab")
        assert False, "Expected validation error"
    except Exception as e:
        assert str(e) == "Must have at least 3 characters."

def test_validate_with_max_length():
    field = String(max_length=3)
    try:
        field.validate("abcd")
        assert False, "Expected validation error"
    except Exception as e:
        assert str(e) == "Must have no more than 3 characters."

def test_validate_with_pattern():
    field = String(pattern=r"^\d+$")
    try:
        field.validate("abc")
        assert False, "Expected validation error"
    except Exception as e:
        assert str(e) == "Must match the pattern /^\\d+$/."

def test_validate_with_format():
    field = String(format="email")
    try:
        field.validate("invalid-email")
        assert False, "Expected validation error"
    except Exception as e:
        assert str(e) == "Must be a valid email."


# LLM-generated content at query #3
#--------------------------

```python
def test_string_constructor_defaults():
    field = String()
    assert field.title == ""
    assert field.description == ""
    assert not field.has_default()
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

def test_string_constructor_with_all_params():
    field = String(
        title="Test Title",
        description="Test Description",
        default="default_value",
        allow_null=True,
        read_only=True,
        allow_blank=True,
        trim_whitespace=False,
        max_length=10,
        min_length=2,
        pattern=r"^[a-z]+$",
        format="email",
        coerce_types=False,
    )
    assert field.title == "Test Title"
    assert field.description == "Test Description"
    assert field.has_default()
    assert field.get_default_value() == "default_value"
    assert field.allow_null
    assert field.read_only
    assert field.allow_blank
    assert not field.trim_whitespace
    assert field.max_length == 10
    assert field.min_length == 2
    assert field.pattern == r"^[a-z]+$"
    assert field.pattern_regex.pattern == r"^[a-z]+$"
    assert field.format == "email"
    assert not field.coerce_types

def test_string_constructor_with_pattern_regex():
    pattern = re.compile(r"^[a-z]+$")
    field = String(pattern=pattern)
    assert field.pattern == r"^[a-z]+$"
    assert field.pattern_regex == pattern

def test_string_constructor_with_allow_blank_no_default():
    field = String(allow_blank=True)
    assert field.has_default()
    assert field.get_default_value() == ""

def test_string_constructor_with_allow_null_no_default():
    field = String(allow_null=True)
    assert field.has_default()
    assert field.get_default_value() is None

def test_string_constructor_invalid_max_length():
    with pytest.raises(AssertionError):
        String(max_length="invalid")

def test_string_constructor_invalid_min_length():
    with pytest.raises(AssertionError):
        String(min_length="invalid")

def test_string_constructor_invalid_pattern():
    with pytest.raises(AssertionError):
        String(pattern=123)

def test_string_constructor_invalid_format():
    with pytest.raises(AssertionError):
        String(format=123)


# LLM-generated content at query #4
#--------------------------

```python
def test_validate_with_none_value_and_allow_null():
    field = Object(allow_null=True)
    assert field.validate(None) is None

def test_validate_with_none_value_and_not_allow_null():
    field = Object(allow_null=False)
    try:
        field.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "May not be null."
        assert e.messages[0].code == "null"

def test_validate_with_non_dict_value():
    field = Object()
    try:
        field.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Must be an object."
        assert e.messages[0].code == "type"

def test_validate_with_non_string_keys():
    field = Object()
    try:
        field.validate({123: "value"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "All object keys must be strings."
        assert e.messages[0].code == "invalid_key"
        assert e.messages[0].index == [123]

def test_validate_with_invalid_property_names():
    field = Object(property_names=String(min_length=5))
    try:
        field.validate({"abc": "value"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Invalid property name."
        assert e.messages[0].code == "invalid_property"
        assert e.messages[0].index == ["abc"]

def test_validate_with_min_properties():
    field = Object(min_properties=2)
    try:
        field.validate({"a": 1})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Must have at least 2 properties."
        assert e.messages[0].code == "min_properties"

def test_validate_with_min_properties_equal_to_one():
    field = Object(min_properties=1)
    try:
        field.validate({})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Must not be empty."
        assert e.messages[0].code == "empty"

def test_validate_with_max_properties():
    field = Object(max_properties=2)
    try:
        field.validate({"a": 1, "b": 2, "c": 3})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Must have no more than 2 properties."
        assert e.messages[0].code == "max_properties"

def test_validate_with_required_properties():
    field = Object(required=["username"])
    try:
        field.validate({"email": "test@example.com"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "This field is required."
        assert e.messages[0].code == "required"
        assert e.messages[0].index == ["username"]

def test_validate_with_properties_validation_error():
    field = Object(properties={"age": Integer(min_value=18)})
    try:
        field.validate({"age": 17})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Must be at least 18."
        assert e.messages[0].code == "min_value"
        assert e.messages[0].index == ["age"]

def test_validate_with_pattern_properties():
    field = Object(pattern_properties={r"^test_": String()})
    result = field.validate({"test_key": "value", "other": 123})
    assert result == {"test_key": "value", "other": 123}

def test_validate_with_pattern_properties_validation_error():
    field = Object(pattern_properties={r"^test_": Integer()})
    try:
        field.validate({"test_key": "not an integer"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Must be an integer."
        assert e.messages[0].code == "type"
        assert e.messages[0].index == ["test_key"]

def test_validate_with_additional_properties_false():
    field = Object(properties={"allowed": String()}, additional_properties=False)
    try:
        field.validate({"allowed": "value", "not_allowed": "value"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Invalid property name."
        assert e.messages[0].code == "invalid_property"
        assert e.messages[0].key == "not_allowed"

def test_validate_with_additional_properties_as_field():
    field = Object(properties={"allowed": String()}, additional_properties=Integer())
    result = field.validate({"allowed": "value", "extra": 123})
    assert result == {"allowed": "value", "extra": 123}

def test_validate_with_additional_properties_as_field_validation_error():
    field = Object(properties={"allowed": String()}, additional_properties=Integer())
    try:
        field.validate({"allowed": "value", "extra": "not an integer"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Must be an integer."
        assert e.messages[0].code == "type"
        assert e.messages[0].index == ["extra"]

def test_validate_with_properties_default_values():
    field = Object(properties={"optional": String(default="default_value")})
    result = field.validate({})
    assert result == {"optional": "default_value"}

def test_validate_with_valid_input():
    field = Object(properties={"name": String(), "age": Integer()})
    result = field.validate({"name": "John", "age": 30})
    assert result == {"name": "John", "age": 30}

def test_validate_with_additional_properties_true():
    field = Object(properties={"name": String()}, additional_properties=True)
    result = field.validate({"name": "John", "extra": "value"})
    assert result == {"name": "John", "extra": "value"}


# LLM-generated content at query #5
#--------------------------

```python
def test_validate_with_none_value_and_allow_null():
    choice = Choice(choices=[("a", "a")], allow_null=True)
    assert choice.validate(None) is None

def test_validate_with_none_value_and_no_allow_null():
    choice = Choice(choices=[("a", "a")], allow_null=False)
    try:
        choice.validate(None)
        assert False, "Expected validation error"
    except ValidationError as e:
        assert e.detail == "May not be null."

def test_validate_with_empty_string_and_allow_null_and_coerce_types():
    choice = Choice(choices=[("a", "a")], allow_null=True, coerce_types=True)
    assert choice.validate("") is None

def test_validate_with_empty_string_and_allow_null_and_no_coerce_types():
    choice = Choice(choices=[("a", "a")], allow_null=True, coerce_types=False)
    try:
        choice.validate("")
        assert False, "Expected validation error"
    except ValidationError as e:
        assert e.detail == "This field is required."

def test_validate_with_empty_string_and_no_allow_null():
    choice = Choice(choices=[("a", "a")], allow_null=False)
    try:
        choice.validate("")
        assert False, "Expected validation error"
    except ValidationError as e:
        assert e.detail == "This field is required."

def test_validate_with_valid_choice():
    choice = Choice(choices=[("a", "a"), ("b", "b")])
    assert choice.validate("a") == "a"

def test_validate_with_invalid_choice():
    choice = Choice(choices=[("a", "a"), ("b", "b")])
    try:
        choice.validate("c")
        assert False, "Expected validation error"
    except ValidationError as e:
        assert e.detail == "Not a valid choice."

def test_validate_with_true_and_false_as_choices():
    choice = Choice(choices=[(True, "True"), (False, "False")])
    assert choice.validate(True) == True
    assert choice.validate(False) == False

def test_validate_with_1_and_0_as_choices():
    choice = Choice(choices=[(1, "1"), (0, "0")])
    assert choice.validate(1) == 1
    assert choice.validate(0) == 0

def test_validate_with_list_as_choice():
    choice = Choice(choices=[(["a", "b"], "list")])
    assert choice.validate(["a", "b"]) == ["a", "b"]

def test_validate_with_dict_as_choice():
    choice = Choice(choices=[({"a": "b"}, "dict")])
    assert choice.validate({"a": "b"}) == {"a": "b"}


# LLM-generated content at query #6
#--------------------------

```python
def test_validate_with_none_and_allow_null():
    union = Union(any_of=[String()], allow_null=True)
    assert union.validate(None) is None

def test_validate_with_none_and_not_allow_null():
    union = Union(any_of=[String()], allow_null=False)
    try:
        union.validate(None)
    except ValidationError as e:
        assert e.messages()[0].code == "null"

def test_validate_with_matching_child():
    union = Union(any_of=[String(), Integer()])
    assert union.validate("test") == "test"

def test_validate_with_non_matching_child():
    union = Union(any_of=[String(), Integer()])
    try:
        union.validate(3.14)
    except ValidationError as e:
        assert e.messages()[0].code == "union"

def test_validate_with_candidate_error():
    union = Union(any_of=[String(min_length=5), Integer()])
    try:
        union.validate("abc")
    except ValidationError as e:
        assert e.messages()[0].code == "min_length"

def test_validate_with_multiple_candidate_errors():
    union = Union(any_of=[String(min_length=5), Integer(min_value=10)])
    try:
        union.validate("abc")
    except ValidationError as e:
        assert e.messages()[0].code == "union"


# LLM-generated content at query #7
#--------------------------

```python
def test_union_validate_returns_none_when_value_is_none_and_allow_null_is_true():
    union = Union(any_of=[String()], allow_null=True)
    assert union.validate(None) is None


# LLM-generated content at query #8
#--------------------------

```python
def test_validate_returns_none_when_value_is_none_and_allow_null_is_true():
    boolean_field = Boolean(allow_null=True)
    assert boolean_field.validate(None) is None

def test_validate_raises_validation_error_when_value_is_none_and_allow_null_is_false():
    boolean_field = Boolean(allow_null=False)
    try:
        boolean_field.validate(None)
        assert False, "Expected validation error"
    except ValidationError as e:
        assert e.message == "May not be null."

def test_validate_returns_true_when_value_is_true():
    boolean_field = Boolean()
    assert boolean_field.validate(True) is True

def test_validate_returns_false_when_value_is_false():
    boolean_field = Boolean()
    assert boolean_field.validate(False) is False

def test_validate_raises_validation_error_when_value_is_not_boolean_and_coerce_types_is_false():
    boolean_field = Boolean(coerce_types=False)
    try:
        boolean_field.validate("true")
        assert False, "Expected validation error"
    except ValidationError as e:
        assert e.message == "Must be a boolean."

def test_validate_returns_true_when_value_is_string_true_and_coerce_types_is_true():
    boolean_field = Boolean(coerce_types=True)
    assert boolean_field.validate("true") is True

def test_validate_returns_false_when_value_is_string_false_and_coerce_types_is_true():
    boolean_field = Boolean(coerce_types=True)
    assert boolean_field.validate("false") is False

def test_validate_returns_true_when_value_is_string_on_and_coerce_types_is_true():
    boolean_field = Boolean(coerce_types=True)
    assert boolean_field.validate("on") is True

def test_validate_returns_false_when_value_is_string_off_and_coerce_types_is_true():
    boolean_field = Boolean(coerce_types=True)
    assert boolean_field.validate("off") is False

def test_validate_returns_true_when_value_is_string_1_and_coerce_types_is_true():
    boolean_field = Boolean(coerce_types=True)
    assert boolean_field.validate("1") is True

def test_validate_returns_false_when_value_is_string_0_and_coerce_types_is_true():
    boolean_field = Boolean(coerce_types=True)
    assert boolean_field.validate("0") is False

def test_validate_returns_false_when_value_is_empty_string_and_coerce_types_is_true():
    boolean_field = Boolean(coerce_types=True)
    assert boolean_field.validate("") is False

def test_validate_returns_true_when_value_is_integer_1_and_coerce_types_is_true():
    boolean_field = Boolean(coerce_types=True)
    assert boolean_field.validate(1) is True

def test_validate_returns_false_when_value_is_integer_0_and_coerce_types_is_true():
    boolean_field = Boolean(coerce_types=True)
    assert boolean_field.validate(0) is False

def test_validate_returns_none_when_value_is_empty_string_and_allow_null_is_true():
    boolean_field = Boolean(allow_null=True, coerce_types=True)
    assert boolean_field.validate("") is None

def test_validate_returns_none_when_value_is_string_null_and_allow_null_is_true():
    boolean_field = Boolean(allow_null=True, coerce_types=True)
    assert boolean_field.validate("null") is None

def test_validate_returns_none_when_value_is_string_none_and_allow_null_is_true():
    boolean_field = Boolean(allow_null=True, coerce_types=True)
    assert boolean_field.validate("none") is None

def test_validate_raises_validation_error_when_value_is_invalid_string_and_coerce_types_is_true():
    boolean_field = Boolean(coerce_types=True)
    try:
        boolean_field.validate("invalid")
        assert False, "Expected validation error"
    except ValidationError as e:
        assert e.message == "Must be a boolean."


# LLM-generated content at query #9
#--------------------------

```python
def test_allow_blank_sets_default_to_empty_string():
    field = String(allow_blank=True)
    assert field.default == ""


# LLM-generated content at query #10
#--------------------------

```python
def test_serialize_with_none_value():
    array = Array()
    assert array.serialize(None) is None

def test_serialize_with_empty_list():
    array = Array()
    assert array.serialize([]) == []

def test_serialize_with_items_as_none():
    array = Array(items=None)
    assert array.serialize([1, 2, 3]) == [1, 2, 3]

def test_serialize_with_single_field():
    field = Field()
    array = Array(items=field)
    assert array.serialize([1, 2, 3]) == [field.serialize(1), field.serialize(2), field.serialize(3)]

def test_serialize_with_list_of_fields():
    field1 = Field()
    field2 = Field()
    array = Array(items=[field1, field2])
    assert array.serialize([1, 2]) == [field1.serialize(1), field2.serialize(2)]

def test_serialize_with_more_items_than_fields():
    field1 = Field()
    field2 = Field()
    array = Array(items=[field1, field2])
    assert array.serialize([1, 2, 3]) == [field1.serialize(1), field2.serialize(2), 3]

def test_serialize_with_additional_items_field():
    field1 = Field()
    field2 = Field()
    additional_field = Field()
    array = Array(items=[field1, field2], additional_items=additional_field)
    assert array.serialize([1, 2, 3]) == [field1.serialize(1), field2.serialize(2), additional_field.serialize(3)]


# LLM-generated content at query #11
#--------------------------

```python
def test_validate_with_none_and_allow_null():
    field = Array(allow_null=True)
    assert field.validate(None) is None

def test_validate_with_none_and_not_allow_null():
    field = Array(allow_null=False)
    try:
        field.validate(None)
    except ValidationError as e:
        assert e.messages[0].text == "May not be null."
        assert e.messages[0].code == "null"

def test_validate_with_non_list():
    field = Array()
    try:
        field.validate("not a list")
    except ValidationError as e:
        assert e.messages[0].text == "Must be an array."
        assert e.messages[0].code == "type"

def test_validate_with_exact_items():
    field = Array(exact_items=3)
    assert field.validate([1, 2, 3]) == [1, 2, 3]
    try:
        field.validate([1, 2])
    except ValidationError as e:
        assert e.messages[0].text == "Must have 3 items."
        assert e.messages[0].code == "exact_items"

def test_validate_with_min_items():
    field = Array(min_items=2)
    assert field.validate([1, 2, 3]) == [1, 2, 3]
    try:
        field.validate([1])
    except ValidationError as e:
        assert e.messages[0].text == "Must have at least 2 items."
        assert e.messages[0].code == "min_items"

def test_validate_with_min_items_one():
    field = Array(min_items=1)
    try:
        field.validate([])
    except ValidationError as e:
        assert e.messages[0].text == "Must not be empty."
        assert e.messages[0].code == "empty"

def test_validate_with_max_items():
    field = Array(max_items=2)
    assert field.validate([1, 2]) == [1, 2]
    try:
        field.validate([1, 2, 3])
    except ValidationError as e:
        assert e.messages[0].text == "Must have no more than 2 items."
        assert e.messages[0].code == "max_items"

def test_validate_with_items_field():
    field = Array(items=Integer())
    assert field.validate([1, 2, 3]) == [1, 2, 3]
    try:
        field.validate([1, "not an integer", 3])
    except ValidationError as e:
        assert e.messages[0].text == "Must be an integer."
        assert e.messages[0].code == "type"
        assert e.messages[0].index == [1]

def test_validate_with_items_list():
    field = Array(items=[Integer(), String()])
    assert field.validate([1, "two"]) == [1, "two"]
    try:
        field.validate([1, 2])
    except ValidationError as e:
        assert e.messages[0].text == "Must be a string."
        assert e.messages[0].code == "type"
        assert e.messages[0].index == [1]

def test_validate_with_additional_items_field():
    field = Array(items=[Integer()], additional_items=String())
    assert field.validate([1, "two", "three"]) == [1, "two", "three"]
    try:
        field.validate([1, 2])
    except ValidationError as e:
        assert e.messages[0].text == "Must be a string."
        assert e.messages[0].code == "type"
        assert e.messages[0].index == [1]

def test_validate_with_additional_items_false():
    field = Array(items=[Integer()], additional_items=False)
    assert field.validate([1]) == [1]
    try:
        field.validate([1, 2])
    except ValidationError as e:
        assert e.messages[0].text == "May not contain additional items."
        assert e.messages[0].code == "additional_items"
        assert e.messages[0].index == [1]

def test_validate_with_unique_items():
    field = Array(unique_items=True)
    assert field.validate([1, 2, 3]) == [1, 2, 3]
    try:
        field.validate([1, 2, 1])
    except ValidationError as e:
        assert e.messages[0].text == "Items must be unique."
        assert e.messages[0].code == "unique_items"
        assert e.messages[0].index == [2]

def test_validate_with_unique_items_and_complex_types():
    field = Array(unique_items=True)
    assert field.validate([[1, 2], [3, 4]]) == [[1, 2], [3, 4]]
    try:
        field.validate([[1, 2], [1, 2]])
    except ValidationError as e:
        assert e.messages[0].text == "Items must be unique."
        assert e.messages[0].code == "unique_items"
        assert e.messages[0].index == [1]

def test_validate_with_allow_null_items():
    field = Array(items=Integer(allow_null=True))
    assert field.validate([1, None, 3]) == [1, None, 3]

def test_validate_with_serialize():
    field = Array(items=Integer())
    assert field.serialize([1, 2, 3]) == [1, 2, 3]
    field = Array(items=[Integer(), String()])
    assert field.serialize([1, "two"]) == [1, "two"]
    field = Array(items=Integer(), additional_items=String())
    assert field.serialize([1, "two", "three"]) == [1, "two", "three"]


# LLM-generated content at query #12
#--------------------------

```python
def test_array_constructor_with_valid_single_item():
    field = Array(items=Field())
    assert field.items == Field()
    assert field.additional_items is False
    assert field.min_items is None
    assert field.max_items is None
    assert field.unique_items is False

def test_array_constructor_with_valid_list_items():
    field = Array(items=[Field(), Field()])
    assert field.items == [Field(), Field()]
    assert field.additional_items is False
    assert field.min_items == 2
    assert field.max_items == 2
    assert field.unique_items is False

def test_array_constructor_with_additional_items_field():
    field = Array(items=Field(), additional_items=Field())
    assert field.items == Field()
    assert field.additional_items == Field()
    assert field.min_items is None
    assert field.max_items is None
    assert field.unique_items is False

def test_array_constructor_with_min_items():
    field = Array(items=Field(), min_items=5)
    assert field.items == Field()
    assert field.additional_items is False
    assert field.min_items == 5
    assert field.max_items is None
    assert field.unique_items is False

def test_array_constructor_with_max_items():
    field = Array(items=Field(), max_items=10)
    assert field.items == Field()
    assert field.additional_items is False
    assert field.min_items is None
    assert field.max_items == 10
    assert field.unique_items is False

def test_array_constructor_with_exact_items():
    field = Array(items=Field(), exact_items=3)
    assert field.items == Field()
    assert field.additional_items is False
    assert field.min_items == 3
    assert field.max_items == 3
    assert field.unique_items is False

def test_array_constructor_with_unique_items():
    field = Array(items=Field(), unique_items=True)
    assert field.items == Field()
    assert field.additional_items is False
    assert field.min_items is None
    assert field.max_items is None
    assert field.unique_items is True

def test_array_constructor_with_kwargs():
    field = Array(items=Field(), title="Test", description="Description", default=[], allow_null=True, read_only=True)
    assert field.items == Field()
    assert field.title == "Test"
    assert field.description == "Description"
    assert field.default == []
    assert field.allow_null is True
    assert field.read_only is True

def test_array_constructor_with_none_items():
    field = Array(items=None)
    assert field.items is None
    assert field.additional_items is False
    assert field.min_items is None
    assert field.max_items is None
    assert field.unique_items is False


# LLM-generated content at query #13
#--------------------------

```python
def test_validate_with_none_value_and_allow_null_false():
    field = Object(allow_null=False)
    try:
        field.validate(None)
    except ValidationError as e:
        assert e.messages[0].text == "May not be null."
        assert e.messages[0].code == "null"
    else:
        assert False, "Expected ValidationError"


# LLM-generated content at query #14
#--------------------------

```python
def test_empty_string_with_allow_null_and_coerce_types():
    number = Number(allow_null=True, coerce_types=True)
    assert number.validate("") is None


# LLM-generated content at query #15
#--------------------------

```python
def test_validate_with_none_value_and_allow_null_false():
    field = Object(allow_null=False)
    try:
        field.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as exc:
        assert len(exc.messages) == 1
        assert exc.messages[0].text == "May not be null."
        assert exc.messages[0].code == "null"


# LLM-generated content at query #16
#--------------------------

```python
def test_validate_with_invalid_choice():
    choice_field = Choice(choices=[("a", "A"), ("b", "B")])
    try:
        choice_field.validate("c")
    except ValidationError as e:
        assert e.message == "Not a valid choice."
    else:
        assert False, "Expected ValidationError"


# LLM-generated content at query #17
#--------------------------

```python
def test_validate_with_none_value_and_allow_null():
    boolean_field = Boolean(allow_null=True)
    assert boolean_field.validate(None) is None

def test_validate_with_none_value_and_not_allow_null():
    boolean_field = Boolean(allow_null=False)
    try:
        boolean_field.validate(None)
        assert False, "Expected validation error"
    except ValidationError as e:
        assert e.message == "May not be null."

def test_validate_with_boolean_value():
    boolean_field = Boolean()
    assert boolean_field.validate(True) is True
    assert boolean_field.validate(False) is False

def test_validate_with_non_boolean_value_and_coerce_types_disabled():
    boolean_field = Boolean(coerce_types=False)
    try:
        boolean_field.validate("true")
        assert False, "Expected validation error"
    except ValidationError as e:
        assert e.message == "Must be a boolean."

def test_validate_with_string_value_and_coerce_types_enabled():
    boolean_field = Boolean()
    assert boolean_field.validate("true") is True
    assert boolean_field.validate("false") is False
    assert boolean_field.validate("on") is True
    assert boolean_field.validate("off") is False
    assert boolean_field.validate("1") is True
    assert boolean_field.validate("0") is False
    assert boolean_field.validate("") is False

def test_validate_with_integer_value_and_coerce_types_enabled():
    boolean_field = Boolean()
    assert boolean_field.validate(1) is True
    assert boolean_field.validate(0) is False

def test_validate_with_null_string_value_and_allow_null():
    boolean_field = Boolean(allow_null=True)
    assert boolean_field.validate("null") is None
    assert boolean_field.validate("none") is None
    assert boolean_field.validate("") is None

def test_validate_with_invalid_value_and_coerce_types_enabled():
    boolean_field = Boolean()
    try:
        boolean_field.validate("invalid")
        assert False, "Expected validation error"
    except ValidationError as e:
        assert e.message == "Must be a boolean."


# LLM-generated content at query #18
#--------------------------

```python
def test_validate_with_none_and_allow_null():
    union = Union(any_of=[StringField()], allow_null=True)
    assert union.validate(None) is None

def test_validate_with_none_and_not_allow_null():
    union = Union(any_of=[StringField()], allow_null=False)
    with pytest.raises(ValidationError) as excinfo:
        union.validate(None)
    assert excinfo.value.code == "null"

def test_validate_with_matching_child():
    union = Union(any_of=[StringField(), IntegerField()])
    assert union.validate("test") == "test"

def test_validate_with_non_matching_children():
    union = Union(any_of=[StringField(), IntegerField()])
    with pytest.raises(ValidationError) as excinfo:
        union.validate(3.14)
    assert excinfo.value.code == "union"

def test_validate_with_single_candidate_error():
    union = Union(any_of=[StringField(min_length=5), IntegerField()])
    with pytest.raises(ValidationError) as excinfo:
        union.validate("abc")
    assert excinfo.value.code == "min_length"

def test_validate_with_multiple_candidate_errors():
    union = Union(any_of=[StringField(min_length=5), IntegerField(min_value=10)])
    with pytest.raises(ValidationError) as excinfo:
        union.validate("abc")
    assert excinfo.value.code == "union"


# LLM-generated content at query #19
#--------------------------

```python
def test_const_constructor_with_valid_const():
    const_value = "test"
    field = Const(const=const_value)
    assert field.const == const_value
    assert field.title == ""
    assert field.description == ""
    assert not field.allow_null
    assert not field.read_only

def test_const_constructor_with_none_const():
    field = Const(const=None)
    assert field.const is None
    assert field.title == ""
    assert field.description == ""
    assert not field.allow_null
    assert not field.read_only

def test_const_constructor_with_allow_null_raises_assertion():
    try:
        Const(const="test", allow_null=True)
        assert False, "Expected AssertionError"
    except AssertionError:
        pass

def test_const_constructor_with_custom_title_and_description():
    const_value = 42
    title = "Custom Title"
    description = "Custom Description"
    field = Const(const=const_value, title=title, description=description)
    assert field.const == const_value
    assert field.title == title
    assert field.description == description
    assert not field.allow_null
    assert not field.read_only

def test_const_constructor_with_read_only():
    const_value = True
    field = Const(const=const_value, read_only=True)
    assert field.const == const_value
    assert field.title == ""
    assert field.description == ""
    assert not field.allow_null
    assert field.read_only


# LLM-generated content at query #20
#--------------------------

```python
def test_get_default_value_with_callable_default():
    field = Field(default=lambda: "default_value")
    assert field.get_default_value() == "default_value"

def test_get_default_value_with_non_callable_default():
    field = Field(default="default_value")
    assert field.get_default_value() == "default_value"

def test_get_default_value_without_default():
    field = Field()
    assert field.get_default_value() is None

def test_get_default_value_with_none_default():
    field = Field(default=None)
    assert field.get_default_value() is None


# LLM-generated content at query #21
#--------------------------

```python
def test_validate_returns_none_when_value_is_none_and_allow_null_is_true():
    union_field = Union(any_of=[], allow_null=True)
    assert union_field.validate(None) is None


# LLM-generated content at query #22
#--------------------------

```python
def test_predicate_evaluates_to_true():
    union = Union([Field()])
    error = Error()
    error.add_message("type", index=None)
    messages = error.messages()
    assert (len(messages) != 1 or messages[0].code != "type" or messages[0].index)


# LLM-generated content at query #23
#--------------------------

```python
def test_empty_string_with_allow_null_and_coerce_types():
    number = Number(allow_null=True, coerce_types=True)
    assert number.validate("") is None


# LLM-generated content at query #24
#--------------------------

```python
def test_validate_raises_type_error_for_invalid_value():
    field = Boolean(coerce_types=True, allow_null=False)
    with pytest.raises(Exception) as exc_info:
        field.validate("invalid")
    assert exc_info.value.args[0] == "Must be a boolean."


# LLM-generated content at query #25
#--------------------------

```python
def test_validate_with_invalid_choice():
    choice_field = Choice(choices=[("a", "A"), ("b", "B")])
    assert "Not a valid choice." in str(choice_field.validate("c"))


# LLM-generated content at query #26
#--------------------------

```python
def test_validate_none_with_allow_null():
    field = String(allow_null=True)
    assert field.validate(None) is None

def test_validate_none_with_allow_blank():
    field = String(allow_blank=True)
    assert field.validate(None) == ""

def test_validate_none_without_allow_null_or_blank():
    field = String()
    try:
        field.validate(None)
    except Exception as e:
        assert str(e) == "May not be null."

def test_validate_non_string_type():
    field = String()
    try:
        field.validate(123)
    except Exception as e:
        assert str(e) == "Must be a string."

def test_validate_empty_string_with_allow_blank():
    field = String(allow_blank=True)
    assert field.validate("") == ""

def test_validate_empty_string_without_allow_blank():
    field = String()
    try:
        field.validate("")
    except Exception as e:
        assert str(e) == "Must not be blank."

def test_validate_string_with_null_character():
    field = String()
    assert field.validate("a\0b") == "ab"

def test_validate_string_with_trim_whitespace():
    field = String()
    assert field.validate("  hello  ") == "hello"

def test_validate_string_without_trim_whitespace():
    field = String(trim_whitespace=False)
    assert field.validate("  hello  ") == "  hello  "

def test_validate_string_with_min_length():
    field = String(min_length=3)
    try:
        field.validate("ab")
    except Exception as e:
        assert str(e) == "Must have at least 3 characters."

def test_validate_string_with_max_length():
    field = String(max_length=3)
    try:
        field.validate("abcd")
    except Exception as e:
        assert str(e) == "Must have no more than 3 characters."

def test_validate_string_with_pattern():
    field = String(pattern=r"^\d+$")
    try:
        field.validate("abc")
    except Exception as e:
        assert str(e) == "Must match the pattern /^\\d+$/."

def test_validate_string_with_format():
    field = String(format="email")
    try:
        field.validate("invalid-email")
    except Exception as e:
        assert str(e) == "Must be a valid email."


# LLM-generated content at query #27
#--------------------------

```python
def test_validate_with_none_and_allow_null():
    field = Object(allow_null=True)
    assert field.validate(None) is None

def test_validate_with_none_and_not_allow_null():
    field = Object(allow_null=False)
    try:
        field.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "May not be null."
        assert e.messages[0].code == "null"

def test_validate_with_non_dict():
    field = Object()
    try:
        field.validate("not a dict")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Must be an object."
        assert e.messages[0].code == "type"

def test_validate_with_invalid_key_type():
    field = Object()
    try:
        field.validate({123: "value"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "All object keys must be strings."
        assert e.messages[0].code == "invalid_key"
        assert e.messages[0].index == [123]

def test_validate_with_invalid_property_name():
    field = Object(property_names=String())
    try:
        field.validate({"invalid@key": "value"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Invalid property name."
        assert e.messages[0].code == "invalid_property"
        assert e.messages[0].index == ["invalid@key"]

def test_validate_with_min_properties():
    field = Object(min_properties=2)
    try:
        field.validate({"key": "value"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Must have at least 2 properties."
        assert e.messages[0].code == "min_properties"

def test_validate_with_min_properties_equal_to_one():
    field = Object(min_properties=1)
    try:
        field.validate({})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Must not be empty."
        assert e.messages[0].code == "empty"

def test_validate_with_max_properties():
    field = Object(max_properties=2)
    try:
        field.validate({"key1": "value1", "key2": "value2", "key3": "value3"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Must have no more than 2 properties."
        assert e.messages[0].code == "max_properties"

def test_validate_with_required_property_missing():
    field = Object(required=["required_key"])
    try:
        field.validate({"other_key": "value"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "This field is required."
        assert e.messages[0].code == "required"
        assert e.messages[0].index == ["required_key"]

def test_validate_with_property_validation_error():
    field = Object(properties={"key": String(max_length=5)})
    try:
        field.validate({"key": "too long value"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "max_length"
        assert e.messages[0].index == ["key"]

def test_validate_with_pattern_property():
    field = Object(pattern_properties={r"^test_": String()})
    result = field.validate({"test_key": "value", "other_key": 123})
    assert result == {"test_key": "value"}

def test_validate_with_additional_properties_false():
    field = Object(properties={"allowed": String()}, additional_properties=False)
    try:
        field.validate({"allowed": "value", "not_allowed": "value"})
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Invalid property name."
        assert e.messages[0].code == "invalid_property"
        assert e.messages[0].key == "not_allowed"

def test_validate_with_additional_properties_field():
    field = Object(properties={"allowed": String()}, additional_properties=Integer())
    result = field.validate({"allowed": "value", "additional": 123})
    assert result == {"allowed": "value", "additional": 123}

def test_validate_with_additional_properties_true():
    field = Object(properties={"allowed": String()}, additional_properties=True)
    result = field.validate({"allowed": "value", "additional": 123})
    assert result == {"allowed": "value", "additional": 123}

def test_validate_with_default_value():
    field = Object(properties={"key": String(default="default_value")})
    result = field.validate({})
    assert result == {"key": "default_value"}

def test_validate_successful():
    field = Object(properties={"key": String()})
    result = field.validate({"key": "value"})
    assert result == {"key": "value"}


# LLM-generated content at query #28
#--------------------------

```python
def test_validate_with_null_value_and_allow_null():
    field = Array(allow_null=True)
    assert field.validate(None) is None

def test_validate_with_null_value_and_not_allow_null():
    field = Array(allow_null=False)
    try:
        field.validate(None)
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "May not be null."

def test_validate_with_non_list_value():
    field = Array()
    try:
        field.validate("not a list")
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Must be an array."

def test_validate_with_exact_items():
    field = Array(exact_items=3)
    assert field.validate([1, 2, 3]) == [1, 2, 3]

def test_validate_with_exact_items_failure():
    field = Array(exact_items=3)
    try:
        field.validate([1, 2])
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Must have 3 items."

def test_validate_with_min_items():
    field = Array(min_items=2)
    assert field.validate([1, 2, 3]) == [1, 2, 3]

def test_validate_with_min_items_failure():
    field = Array(min_items=2)
    try:
        field.validate([1])
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Must have at least 2 items."

def test_validate_with_min_items_empty():
    field = Array(min_items=1)
    try:
        field.validate([])
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Must not be empty."

def test_validate_with_max_items():
    field = Array(max_items=3)
    assert field.validate([1, 2, 3]) == [1, 2, 3]

def test_validate_with_max_items_failure():
    field = Array(max_items=2)
    try:
        field.validate([1, 2, 3])
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Must have no more than 2 items."

def test_validate_with_items_field():
    field = Array(items=Integer())
    assert field.validate([1, 2, 3]) == [1, 2, 3]

def test_validate_with_items_field_failure():
    field = Array(items=Integer())
    try:
        field.validate([1, "not an integer", 3])
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Must be an integer."

def test_validate_with_items_list():
    field = Array(items=[Integer(), String()])
    assert field.validate([1, "two"]) == [1, "two"]

def test_validate_with_items_list_failure():
    field = Array(items=[Integer(), String()])
    try:
        field.validate([1, 2])
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Must be a string."

def test_validate_with_additional_items_field():
    field = Array(items=[Integer()], additional_items=String())
    assert field.validate([1, "two", "three"]) == [1, "two", "three"]

def test_validate_with_additional_items_false():
    field = Array(items=[Integer()], additional_items=False)
    try:
        field.validate([1, 2])
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "May not contain additional items."

def test_validate_with_unique_items():
    field = Array(unique_items=True)
    assert field.validate([1, 2, 3]) == [1, 2, 3]

def test_validate_with_unique_items_failure():
    field = Array(unique_items=True)
    try:
        field.validate([1, 2, 2])
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Items must be unique."

def test_validate_with_unique_items_complex_types():
    field = Array(unique_items=True)
    assert field.validate([[1, 2], [3, 4], [1, 2]]) == [[1, 2], [3, 4], [1, 2]]

def test_validate_with_unique_items_complex_types_failure():
    field = Array(unique_items=True)
    try:
        field.validate([[1, 2], [3, 4], [1, 2]])
        assert False, "Expected ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Items must be unique."


# LLM-generated content at query #29
#--------------------------

```python
def test_validate_returns_none_when_value_is_none_and_allow_null_is_true():
    boolean_field = Boolean(allow_null=True)
    assert boolean_field.validate(None) is None


# LLM-generated content at query #30
#--------------------------

```python
def test_unique_items_duplicate_detection():
    array_field = Array(unique_items=True)
    with pytest.raises(ValidationError) as exc_info:
        array_field.validate([1, 2, 1])
    assert exc_info.value.messages[0].code == "unique_items"
    assert exc_info.value.messages[0].key == 2


# LLM-generated content at query #31
#--------------------------

```python
def test_validate_returns_none_when_value_is_none_and_allow_null_is_true():
    field = Array(allow_null=True)
    result = field.validate(None)
    assert result is None


# LLM-generated content at query #32
#--------------------------

```python
def test_validate_with_none_and_allow_null():
    field = Object(allow_null=True)
    assert field.validate(None) is None

def test_validate_with_none_and_not_allow_null():
    field = Object(allow_null=False)
    try:
        field.validate(None)
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "May not be null."
        assert e.messages[0].code == "null"

def test_validate_with_non_dict():
    field = Object()
    try:
        field.validate("not a dict")
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Must be an object."
        assert e.messages[0].code == "type"

def test_validate_with_invalid_key_type():
    field = Object()
    try:
        field.validate({123: "value"})
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "All object keys must be strings."
        assert e.messages[0].code == "invalid_key"
        assert e.messages[0].index == [123]

def test_validate_with_invalid_property_name():
    field = Object(property_names=String())
    try:
        field.validate({"123": "value"})
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Invalid property name."
        assert e.messages[0].code == "invalid_property"
        assert e.messages[0].index == ["123"]

def test_validate_with_min_properties():
    field = Object(min_properties=2)
    try:
        field.validate({"a": 1})
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Must have at least 2 properties."
        assert e.messages[0].code == "min_properties"

def test_validate_with_min_properties_equal_to_one():
    field = Object(min_properties=1)
    try:
        field.validate({})
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Must not be empty."
        assert e.messages[0].code == "empty"

def test_validate_with_max_properties():
    field = Object(max_properties=2)
    try:
        field.validate({"a": 1, "b": 2, "c": 3})
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Must have no more than 2 properties."
        assert e.messages[0].code == "max_properties"

def test_validate_with_required_property_missing():
    field = Object(required=["a"])
    try:
        field.validate({"b": 2})
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "This field is required."
        assert e.messages[0].code == "required"
        assert e.messages[0].index == ["a"]

def test_validate_with_property_validation_error():
    field = Object(properties={"a": Integer()})
    try:
        field.validate({"a": "not an integer"})
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Must be an integer."
        assert e.messages[0].code == "type"
        assert e.messages[0].index == ["a"]

def test_validate_with_pattern_property_validation_error():
    field = Object(pattern_properties={r"^a.*": Integer()})
    try:
        field.validate({"abc": "not an integer"})
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Must be an integer."
        assert e.messages[0].code == "type"
        assert e.messages[0].index == ["abc"]

def test_validate_with_additional_properties_false():
    field = Object(properties={"a": Integer()}, additional_properties=False)
    try:
        field.validate({"a": 1, "b": 2})
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Invalid property name."
        assert e.messages[0].code == "invalid_property"
        assert e.messages[0].key == "b"

def test_validate_with_additional_properties_field_validation_error():
    field = Object(properties={"a": Integer()}, additional_properties=Integer())
    try:
        field.validate({"a": 1, "b": "not an integer"})
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].text == "Must be an integer."
        assert e.messages[0].code == "type"
        assert e.messages[0].index == ["b"]

def test_validate_with_valid_input():
    field = Object(properties={"a": Integer()}, additional_properties=String())
    result = field.validate({"a": 1, "b": "valid"})
    assert result == {"a": 1, "b": "valid"}

def test_validate_with_default_value():
    field = Object(properties={"a": Integer(default=0)})
    result = field.validate({"b": "valid"})
    assert result == {"a": 0, "b": "valid"}


# LLM-generated content at query #33
#--------------------------

```python
def test_multiple_of_predicate_false():
    number_field = Number(multiple_of=3)
    try:
        number_field.validate(5)
    except Exception as e:
        assert str(e) == "Must be a multiple of 3."


# LLM-generated content at query #34
#--------------------------

```python
def test_union_predicate_evaluates_to_true():
    child_field = Field()
    child_field.validate_or_error = lambda value: (None, Error({"code": "type", "index": None}))
    union = Union([child_field])
    union.validate_or_error = lambda value: (None, Error({"code": "type", "index": None}))
    assert union.validate("test_value") is None


# LLM-generated content at query #35
#--------------------------

```python
def test_validate_returns_none_when_value_is_none_and_allow_null_is_true():
    field = Array(allow_null=True)
    assert field.validate(None) is None


# LLM-generated content at query #36
#--------------------------

```python
def test_validate_allow_null_with_none():
    number = Number(allow_null=True)
    assert number.validate(None) is None

def test_validate_allow_null_with_empty_string():
    number = Number(allow_null=True, coerce_types=True)
    assert number.validate("") is None

def test_validate_none_raises_error():
    number = Number(allow_null=False)
    try:
        number.validate(None)
        assert False, "Expected validation error"
    except Exception as e:
        assert "null" in str(e)

def test_validate_bool_raises_error():
    number = Number()
    try:
        number.validate(True)
        assert False, "Expected validation error"
    except Exception as e:
        assert "type" in str(e)

def test_validate_non_integer_float_with_int_type_raises_error():
    number = Number(numeric_type=int)
    try:
        number.validate(3.14)
        assert False, "Expected validation error"
    except Exception as e:
        assert "integer" in str(e)

def test_validate_non_numeric_string_without_coerce_raises_error():
    number = Number(coerce_types=False)
    try:
        number.validate("abc")
        assert False, "Expected validation error"
    except Exception as e:
        assert "type" in str(e)

def test_validate_infinity_raises_error():
    number = Number()
    try:
        number.validate(float('inf'))
        assert False, "Expected validation error"
    except Exception as e:
        assert "finite" in str(e)

def test_validate_negative_infinity_raises_error():
    number = Number()
    try:
        number.validate(float('-inf'))
        assert False, "Expected validation error"
    except Exception as e:
        assert "finite" in str(e)

def test_validate_nan_raises_error():
    number = Number()
    try:
        number.validate(float('nan'))
        assert False, "Expected validation error"
    except Exception as e:
        assert "finite" in str(e)

def test_validate_below_minimum_raises_error():
    number = Number(minimum=5)
    try:
        number.validate(4)
        assert False, "Expected validation error"
    except Exception as e:
        assert "minimum" in str(e)

def test_validate_equal_to_exclusive_minimum_raises_error():
    number = Number(exclusive_minimum=5)
    try:
        number.validate(5)
        assert False, "Expected validation error"
    except Exception as e:
        assert "exclusive_minimum" in str(e)

def test_validate_above_maximum_raises_error():
    number = Number(maximum=10)
    try:
        number.validate(11)
        assert False, "Expected validation error"
    except Exception as e:
        assert "maximum" in str(e)

def test_validate_equal_to_exclusive_maximum_raises_error():
    number = Number(exclusive_maximum=10)
    try:
        number.validate(10)
        assert False, "Expected validation error"
    except Exception as e:
        assert "exclusive_maximum" in str(e)

def test_validate_not_multiple_of_raises_error():
    number = Number(multiple_of=3)
    try:
        number.validate(5)
        assert False, "Expected validation error"
    except Exception as e:
        assert "multiple_of" in str(e)

def test_validate_valid_integer():
    number = Number()
    assert number.validate(42) == 42

def test_validate_valid_float():
    number = Number()
    assert number.validate(3.14) == 3.14

def test_validate_valid_string_number():
    number = Number()
    assert number.validate("42") == 42

def test_validate_valid_string_float():
    number = Number()
    assert number.validate("3.14") == 3.14

def test_validate_with_precision():
    number = Number(precision="0.01")
    assert number.validate(3.14159) == 3.14

def test_validate_with_minimum():
    number = Number(minimum=5)
    assert number.validate(5) == 5

def test_validate_with_exclusive_minimum():
    number = Number(exclusive_minimum=5)
    assert number.validate(6) == 6

def test_validate_with_maximum():
    number = Number(maximum=10)
    assert number.validate(10) == 10

def test_validate_with_exclusive_maximum():
    number = Number(exclusive_maximum=10)
    assert number.validate(9) == 9

def test_validate_with_multiple_of():
    number = Number(multiple_of=3)
    assert number.validate(6) == 6


# LLM-generated content at query #37
#--------------------------

```python
def test_validate_none_with_allow_null():
    field = Number(allow_null=True)
    assert field.validate(None) is None

def test_validate_empty_string_with_allow_null_and_coerce_types():
    field = Number(allow_null=True, coerce_types=True)
    assert field.validate("") is None

def test_validate_none_without_allow_null():
    field = Number(allow_null=False)
    try:
        field.validate(None)
        assert False, "Expected validation error"
    except Exception as e:
        assert str(e) == "May not be null."

def test_validate_boolean():
    field = Number()
    try:
        field.validate(True)
        assert False, "Expected validation error"
    except Exception as e:
        assert str(e) == "Must be a number."

def test_validate_float_with_integer_type():
    field = Number(numeric_type=int)
    try:
        field.validate(3.14)
        assert False, "Expected validation error"
    except Exception as e:
        assert str(e) == "Must be an integer."

def test_validate_non_numeric_without_coerce_types():
    field = Number(coerce_types=False)
    try:
        field.validate("123")
        assert False, "Expected validation error"
    except Exception as e:
        assert str(e) == "Must be a number."

def test_validate_infinite():
    field = Number()
    try:
        field.validate(float('inf'))
        assert False, "Expected validation error"
    except Exception as e:
        assert str(e) == "Must be finite."

def test_validate_precision():
    field = Number(precision="0.01")
    assert field.validate(3.14159) == 3.14

def test_validate_minimum():
    field = Number(minimum=5)
    try:
        field.validate(3)
        assert False, "Expected validation error"
    except Exception as e:
        assert str(e) == "Must be greater than or equal to 5."

def test_validate_exclusive_minimum():
    field = Number(exclusive_minimum=5)
    try:
        field.validate(5)
        assert False, "Expected validation error"
    except Exception as e:
        assert str(e) == "Must be greater than 5."

def test_validate_maximum():
    field = Number(maximum=10)
    try:
        field.validate(15)
        assert False, "Expected validation error"
    except Exception as e:
        assert str(e) == "Must be less than or equal to 10."

def test_validate_exclusive_maximum():
    field = Number(exclusive_maximum=10)
    try:
        field.validate(10)
        assert False, "Expected validation error"
    except Exception as e:
        assert str(e) == "Must be less than 10."

def test_validate_multiple_of_integer():
    field = Number(multiple_of=3)
    try:
        field.validate(5)
        assert False, "Expected validation error"
    except Exception as e:
        assert str(e) == "Must be a multiple of 3."

def test_validate_multiple_of_float():
    field = Number(multiple_of=0.5)
    try:
        field.validate(1.2)
        assert False, "Expected validation error"
    except Exception as e:
        assert str(e) == "Must be a multiple of 0.5."

def test_validate_valid_integer():
    field = Number()
    assert field.validate(42) == 42

def test_validate_valid_float():
    field = Number()
    assert field.validate(3.14) == 3.14

def test_validate_valid_string_with_coerce_types():
    field = Number(coerce_types=True)
    assert field.validate("42") == 42

def test_validate_valid_string_with_integer_type():
    field = Number(numeric_type=int, coerce_types=True)
    assert field.validate("42") == 42

def test_validate_valid_string_with_float_type():
    field = Number(numeric_type=float, coerce_types=True)
    assert field.validate("3.14") == 3.14

def test_validate_multiple_of_valid_integer():
    field = Number(multiple_of=3)
    assert field.validate(6) == 6

def test_validate_multiple_of_valid_float():
    field = Number(multiple_of=0.5)
    assert field.validate(1.5) == 1.5


