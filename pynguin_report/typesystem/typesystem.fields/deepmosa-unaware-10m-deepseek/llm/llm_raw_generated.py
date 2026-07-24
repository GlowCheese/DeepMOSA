####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_Array_serialize():
    # Test 1: Serialize None when allow_null is True
    field = Array(items=Integer(), allow_null=True)
    assert field.serialize(None) is None

    # Test 2: Serialize None when allow_null is False (should still return None)
    field = Array(items=Integer(), allow_null=False)
    assert field.serialize(None) is None

    # Test 3: Serialize with items as a single Field
    field = Array(items=Integer())
    obj = [1, 2, 3]
    result = field.serialize(obj)
    assert result == [1, 2, 3]

    # Test 4: Serialize with items as a list of Fields
    field = Array(items=[Integer(), String(), Boolean()])
    obj = [42, "hello", True]
    result = field.serialize(obj)
    assert result == [42, "hello", True]

    # Test 5: Serialize with nested serialization
    field = Array(items=Decimal())
    obj = [decimal.Decimal("1.5"), decimal.Decimal("2.75")]
    result = field.serialize(obj)
    assert result == [1.5, 2.75]

    # Test 6: Serialize when items is None (should return original object)
    field = Array(items=None)
    obj = [1, "two", {"three": 3}]
    result = field.serialize(obj)
    assert result == obj

    # Test 7: Serialize with empty list
    field = Array(items=String())
    obj = []
    result = field.serialize(obj)
    assert result == []

    # Test 8: Serialize with mixed types when items is a single Field
    field = Array(items=String())
    obj = ["a", "b", "c"]
    result = field.serialize(obj)
    assert result == ["a", "b", "c"]

    # Test 9: Serialize with items list longer than object
    field = Array(items=[Integer(), String()])
    obj = [1]
    result = field.serialize(obj)
    assert result == [1]

    # Test 10: Serialize with items list shorter than object (additional_items=True)
    field = Array(items=[Integer()], additional_items=True)
    obj = [1, "extra", 3.5]
    result = field.serialize(obj)
    assert result == [1, "extra", 3.5]


# LLM-generated content at query #2
#--------------------------

```python
def test_Array_validate():
    # Test basic validation with allow_null
    field = Array(allow_null=True)
    assert field.validate(None) is None
    
    field = Array(allow_null=False)
    try:
        field.validate(None)
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages()[0].code == "null"
    
    # Test type validation
    field = Array()
    try:
        field.validate("not a list")
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages()[0].code == "type"
    
    # Test empty array validation
    field = Array(min_items=1)
    try:
        field.validate([])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages()[0].code == "empty"
    
    # Test min_items validation
    field = Array(min_items=3)
    try:
        field.validate([1, 2])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages()[0].code == "min_items"
    
    # Test max_items validation
    field = Array(max_items=2)
    try:
        field.validate([1, 2, 3])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages()[0].code == "max_items"
    
    # Test exact_items validation
    field = Array(exact_items=2)
    assert field.validate([1, 2]) == [1, 2]
    
    try:
        field.validate([1])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages()[0].code == "exact_items"
    
    try:
        field.validate([1, 2, 3])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages()[0].code == "exact_items"
    
    # Test items validation with single validator
    field = Array(items=Integer())
    assert field.validate([1, 2, 3]) == [1, 2, 3]
    
    try:
        field.validate([1, "invalid", 3])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages()[0].code == "type"
        assert e.messages()[0].index == [1]
    
    # Test items validation with list of validators
    field = Array(items=[Integer(), String()])
    assert field.validate([1, "hello"]) == [1, "hello"]
    
    try:
        field.validate(["invalid", "hello"])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages()[0].code == "type"
        assert e.messages()[0].index == [0]
    
    # Test additional_items=False
    field = Array(items=[Integer(), String()], additional_items=False)
    assert field.validate([1, "hello"]) == [1, "hello"]
    
    try:
        field.validate([1, "hello", "extra"])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages()[0].code == "additional_items"
    
    # Test additional_items with Field validator
    field = Array(items=[Integer()], additional_items=String())
    assert field.validate([1, "extra1", "extra2"]) == [1, "extra1", "extra2"]
    
    try:
        field.validate([1, 2, 3])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages()[0].code == "type"
        assert e.messages()[0].index == [1]
    
    # Test unique_items validation
    field = Array(unique_items=True)
    assert field.validate([1, 2, 3]) == [1, 2, 3]
    
    try:
        field.validate([1, 2, 1])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages()[0].code == "unique_items"
        assert e.messages()[0].key == 2
    
    # Test complex nested validation
    field = Array(items=Object(properties={"id": Integer(), "name": String()}))
    data = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
    assert field.validate(data) == data
    
    # Test with no items validator (should pass through)
    field = Array()
    data = [1, "hello", {"key": "value"}]
    assert field.validate(data) == data
    
    # Test error message aggregation
    field = Array(items=Integer())
    try:
        field.validate(["a", "b", "c"])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert len(e.messages()) == 3
        assert all(msg.code == "type" for msg in e.messages())
        assert [msg.index for msg in e.messages()] == [[0], [1], [2]]


# LLM-generated content at query #3
#--------------------------

```python
def test_Boolean_validate():
    # Test basic boolean validation
    field = Boolean()
    assert field.validate(True) == True
    assert field.validate(False) == False
    
    # Test null handling
    field_null = Boolean(allow_null=True)
    assert field_null.validate(None) == None
    
    field_not_null = Boolean()
    try:
        field_not_null.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "null"
    
    # Test type coercion
    field_coerce = Boolean(coerce_types=True)
    assert field_coerce.validate("true") == True
    assert field_coerce.validate("false") == False
    assert field_coerce.validate("on") == True
    assert field_coerce.validate("off") == False
    assert field_coerce.validate("1") == True
    assert field_coerce.validate("0") == False
    assert field_coerce.validate("") == False
    assert field_coerce.validate(1) == True
    assert field_coerce.validate(0) == False
    
    # Test null coercion with allow_null
    field_coerce_null = Boolean(allow_null=True, coerce_types=True)
    assert field_coerce_null.validate("") == None
    assert field_coerce_null.validate("null") == None
    assert field_coerce_null.validate("none") == None
    
    # Test without coercion
    field_no_coerce = Boolean(coerce_types=False)
    assert field_no_coerce.validate(True) == True
    assert field_no_coerce.validate(False) == False
    try:
        field_no_coerce.validate("true")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "type"
    
    # Test invalid values with coercion
    field_coerce = Boolean(coerce_types=True)
    try:
        field_coerce.validate("invalid")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "type"
    
    try:
        field_coerce.validate(2)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "type"
    
    # Test case sensitivity in string coercion
    assert field_coerce.validate("TRUE") == True
    assert field_coerce.validate("FALSE") == False
    assert field_coerce.validate("ON") == True
    assert field_coerce.validate("OFF") == False


# LLM-generated content at query #4
#--------------------------

```python
def test_Boolean_validate():
    # Test basic boolean validation
    field = Boolean()
    assert field.validate(True) is True
    assert field.validate(False) is False
    
    # Test null handling
    field = Boolean(allow_null=True)
    assert field.validate(None) is None
    
    field = Boolean(allow_null=False)
    try:
        field.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "null"
    
    # Test type validation without coercion
    field = Boolean(coerce_types=False)
    try:
        field.validate("true")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "type"
    
    # Test string coercion
    field = Boolean(coerce_types=True)
    assert field.validate("true") is True
    assert field.validate("false") is False
    assert field.validate("on") is True
    assert field.validate("off") is False
    assert field.validate("1") is True
    assert field.validate("0") is False
    assert field.validate("") is False
    
    # Test case-insensitive string coercion
    assert field.validate("TRUE") is True
    assert field.validate("FALSE") is False
    assert field.validate("On") is True
    assert field.validate("Off") is False
    
    # Test numeric coercion
    assert field.validate(1) is True
    assert field.validate(0) is False
    
    # Test null coercion with strings
    field = Boolean(coerce_types=True, allow_null=True)
    assert field.validate("null") is None
    assert field.validate("none") is None
    assert field.validate("") is None
    
    # Test invalid string values
    field = Boolean(coerce_types=True)
    try:
        field.validate("invalid")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "type"
    
    # Test invalid type coercion
    try:
        field.validate([])
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "type"
    
    # Test validation_or_error method
    field = Boolean()
    result = field.validate_or_error(True)
    assert result.value is True
    assert result.error is None
    
    result = field.validate_or_error("invalid")
    assert result.value is None
    assert result.error is not None
    assert result.error.code == "type"


# LLM-generated content at query #5
#--------------------------

```python
def test_Array_serialize():
    # Test 1: Serialize None when allow_null is True
    field = Array(items=String(), allow_null=True)
    assert field.serialize(None) is None

    # Test 2: Serialize None when allow_null is False (should still return None)
    field = Array(items=String(), allow_null=False)
    assert field.serialize(None) is None

    # Test 3: Serialize with items as a single Field
    field = Array(items=Integer())
    obj = [1, 2, 3]
    result = field.serialize(obj)
    assert result == [1, 2, 3]

    # Test 4: Serialize with items as a list of Fields
    field = Array(items=[Integer(), String(), Boolean()])
    obj = [42, "hello", True]
    result = field.serialize(obj)
    assert result == [42, "hello", True]

    # Test 5: Serialize with nested serialization
    field = Array(items=Array(items=Integer()))
    obj = [[1, 2], [3, 4]]
    result = field.serialize(obj)
    assert result == [[1, 2], [3, 4]]

    # Test 6: Serialize with items=None (should return obj as-is)
    field = Array(items=None)
    obj = [1, "two", {"three": 3}]
    result = field.serialize(obj)
    assert result == obj

    # Test 7: Serialize with custom field that has serialize method
    class CustomField(Field):
        def serialize(self, obj):
            return f"custom_{obj}"

    field = Array(items=CustomField())
    obj = ["a", "b", "c"]
    result = field.serialize(obj)
    assert result == ["custom_a", "custom_b", "custom_c"]

    # Test 8: Serialize empty list
    field = Array(items=String())
    obj = []
    result = field.serialize(obj)
    assert result == []

    # Test 9: Serialize with mixed types when items is a single Field
    field = Array(items=String())
    obj = ["a", "b", "c"]
    result = field.serialize(obj)
    assert result == ["a", "b", "c"]

    # Test 10: Serialize with items list shorter than obj (should use items for matching positions only)
    field = Array(items=[Integer(), String()])
    obj = [1, "two", "extra", "items"]
    result = field.serialize(obj)
    assert result == [1, "two", "extra", "items"]


# LLM-generated content at query #6
#--------------------------

```python
def test_Boolean_validate():
    # Test basic boolean validation
    field = Boolean()
    assert field.validate(True) is True
    assert field.validate(False) is False
    
    # Test null handling
    field = Boolean(allow_null=True)
    assert field.validate(None) is None
    
    field = Boolean(allow_null=False)
    try:
        field.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "null"
    
    # Test type coercion
    field = Boolean(coerce_types=True)
    assert field.validate("true") is True
    assert field.validate("false") is False
    assert field.validate("on") is True
    assert field.validate("off") is False
    assert field.validate("1") is True
    assert field.validate("0") is False
    assert field.validate("") is False
    assert field.validate(1) is True
    assert field.validate(0) is False
    
    # Test null coercion with allow_null
    field = Boolean(coerce_types=True, allow_null=True)
    assert field.validate("") is None
    assert field.validate("null") is None
    assert field.validate("none") is None
    
    # Test without type coercion
    field = Boolean(coerce_types=False)
    assert field.validate(True) is True
    assert field.validate(False) is False
    
    try:
        field.validate("true")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "type"
    
    try:
        field.validate(1)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "type"
    
    # Test invalid values with coercion
    field = Boolean(coerce_types=True)
    try:
        field.validate("invalid")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "type"
    
    try:
        field.validate(2)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "type"
    
    # Test case sensitivity in string coercion
    field = Boolean(coerce_types=True)
    assert field.validate("TRUE") is True
    assert field.validate("FALSE") is False
    assert field.validate("On") is True
    assert field.validate("Off") is False
    
    # Test validation_error method
    field = Boolean()
    error = field.validation_error("type")
    assert isinstance(error, ValidationError)
    assert error.code == "type"
    assert error.text == "Must be a boolean."


# LLM-generated content at query #7
#--------------------------

```python
def test_Boolean_validate():
    # Test basic boolean validation
    field = Boolean()
    assert field.validate(True) is True
    assert field.validate(False) is False
    
    # Test null handling
    field = Boolean(allow_null=True)
    assert field.validate(None) is None
    
    field = Boolean(allow_null=False)
    try:
        field.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "null"
    
    # Test type validation without coercion
    field = Boolean(coerce_types=False)
    try:
        field.validate("true")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "type"
    
    # Test string coercion
    field = Boolean(coerce_types=True)
    assert field.validate("true") is True
    assert field.validate("false") is False
    assert field.validate("on") is True
    assert field.validate("off") is False
    assert field.validate("1") is True
    assert field.validate("0") is False
    assert field.validate("") is False
    
    # Test numeric coercion
    assert field.validate(1) is True
    assert field.validate(0) is False
    
    # Test case-insensitive string coercion
    assert field.validate("TRUE") is True
    assert field.validate("FALSE") is False
    assert field.validate("On") is True
    assert field.validate("Off") is False
    
    # Test null coercion with strings
    field = Boolean(allow_null=True, coerce_types=True)
    assert field.validate("null") is None
    assert field.validate("none") is None
    assert field.validate("") is None
    
    # Test invalid string values
    field = Boolean(coerce_types=True)
    try:
        field.validate("invalid")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "type"
    
    # Test invalid type coercion
    try:
        field.validate([])
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "type"
    
    # Test validation with default settings
    field = Boolean()
    assert field.validate(True) is True
    assert field.validate(False) is False
    try:
        field.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError:
        pass


# LLM-generated content at query #8
#--------------------------

```python
def test_Array_validate():
    # Test basic validation with no constraints
    field = Array()
    assert field.validate([1, 2, 3]) == [1, 2, 3]
    assert field.validate([]) == []
    
    # Test null handling
    field = Array(allow_null=True)
    assert field.validate(None) is None
    
    field = Array(allow_null=False)
    try:
        field.validate(None)
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages[0].code == "null"
    
    # Test type validation
    field = Array()
    try:
        field.validate("not a list")
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages[0].code == "type"
    
    # Test min_items constraint
    field = Array(min_items=2)
    assert field.validate([1, 2]) == [1, 2]
    assert field.validate([1, 2, 3]) == [1, 2, 3]
    
    try:
        field.validate([1])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages[0].code == "min_items"
    
    # Test min_items=1 gives "empty" error
    field = Array(min_items=1)
    try:
        field.validate([])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages[0].code == "empty"
    
    # Test max_items constraint
    field = Array(max_items=2)
    assert field.validate([1]) == [1]
    assert field.validate([1, 2]) == [1, 2]
    
    try:
        field.validate([1, 2, 3])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages[0].code == "max_items"
    
    # Test exact_items constraint
    field = Array(exact_items=2)
    assert field.validate([1, 2]) == [1, 2]
    
    try:
        field.validate([1])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages[0].code == "exact_items"
    
    try:
        field.validate([1, 2, 3])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages[0].code == "exact_items"
    
    # Test items validation with single validator
    item_field = Integer()
    field = Array(items=item_field)
    assert field.validate([1, 2, 3]) == [1, 2, 3]
    
    try:
        field.validate([1, "invalid", 3])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages[0].code == "type"
        assert e.messages[0].index == [1]
    
    # Test items validation with list of validators
    item_fields = [Integer(), String()]
    field = Array(items=item_fields)
    assert field.validate([1, "text"]) == [1, "text"]
    
    try:
        field.validate(["invalid", "text"])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages[0].code == "type"
        assert e.messages[0].index == [0]
    
    # Test additional_items=False with list validators
    item_fields = [Integer(), String()]
    field = Array(items=item_fields, additional_items=False)
    assert field.validate([1, "text"]) == [1, "text"]
    
    # Should accept extra items when additional_items=False but no max_items
    assert field.validate([1, "text", "extra"]) == [1, "text", "extra"]
    
    # Test additional_items=False with max_items constraint
    field = Array(items=item_fields, additional_items=False, max_items=2)
    try:
        field.validate([1, "text", "extra"])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages[0].code == "max_items"
    
    # Test additional_items with Field validator
    additional_field = String()
    item_fields = [Integer()]
    field = Array(items=item_fields, additional_items=additional_field)
    assert field.validate([1, "valid", "text"]) == [1, "valid", "text"]
    
    try:
        field.validate([1, 2, 3])  # 2 and 3 should fail string validation
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert len(e.messages) == 2
        assert e.messages[0].code == "type"
        assert e.messages[0].index == [1]
    
    # Test unique_items constraint
    field = Array(unique_items=True)
    assert field.validate([1, 2, 3]) == [1, 2, 3]
    
    try:
        field.validate([1, 2, 1])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages[0].code == "unique_items"
        assert e.messages[0].key == 2
    
    # Test combined constraints
    item_field = Integer(minimum=0, maximum=10)
    field = Array(items=item_field, min_items=1, max_items=3, unique_items=True)
    assert field.validate([1, 2, 3]) == [1, 2, 3]
    
    try:
        field.validate([1, 2, 1])  # Duplicate
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages[0].code == "unique_items"
    
    try:
        field.validate([1, 2, 11])  # Out of range
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages[0].code == "maximum"
    
    # Test error message accumulation
    item_field = Integer(minimum=0)
    field = Array(items=item_field)
    try:
        field.validate([-1, "invalid", -2])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert len(e.messages) == 3
        assert e.messages[0].code == "minimum"
        assert e.messages[0].index == [0]
        assert e.messages[1].code == "type"
        assert e.messages[1].index == [1]
        assert e.messages[2].code == "minimum"
        assert e.messages[2].index == [2]


# LLM-generated content at query #9
#--------------------------

```python
def test_Field_get_default_value():
    # Test with no default value
    field1 = Field()
    assert not field1.has_default()
    
    # Test with non-callable default value
    field2 = Field(default="test_default")
    assert field2.has_default()
    assert field2.get_default_value() == "test_default"
    
    # Test with callable default value
    def custom_default():
        return "callable_result"
    
    field3 = Field(default=custom_default)
    assert field3.has_default()
    assert field3.get_default_value() == "callable_result"
    
    # Test with None as default value
    field4 = Field(default=None)
    assert field4.has_default()
    assert field4.get_default_value() is None
    
    # Test with integer default value
    field5 = Field(default=42)
    assert field5.has_default()
    assert field5.get_default_value() == 42
    
    # Test with list default value
    field6 = Field(default=[1, 2, 3])
    assert field6.has_default()
    assert field6.get_default_value() == [1, 2, 3]
    
    # Test with allow_null=True (should set default to None)
    field7 = Field(allow_null=True)
    assert field7.has_default()
    assert field7.get_default_value() is None
    
    # Test with allow_null=True and explicit default
    field8 = Field(allow_null=True, default="explicit")
    assert field8.has_default()
    assert field8.get_default_value() == "explicit"


# LLM-generated content at query #10
#--------------------------

```python
def test_Array_validate():
    # Test basic validation
    field = Array()
    assert field.validate([1, 2, 3]) == [1, 2, 3]
    
    # Test null handling
    field = Array(allow_null=True)
    assert field.validate(None) is None
    
    field = Array(allow_null=False)
    try:
        field.validate(None)
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages()[0].code == "null"
    
    # Test type validation
    field = Array()
    try:
        field.validate("not an array")
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages()[0].code == "type"
    
    # Test min_items
    field = Array(min_items=2)
    assert field.validate([1, 2]) == [1, 2]
    assert field.validate([1, 2, 3]) == [1, 2, 3]
    
    try:
        field.validate([1])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages()[0].code == "min_items"
    
    # Test min_items=1 error message
    field = Array(min_items=1)
    try:
        field.validate([])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages()[0].code == "empty"
    
    # Test max_items
    field = Array(max_items=3)
    assert field.validate([1, 2, 3]) == [1, 2, 3]
    assert field.validate([1, 2]) == [1, 2]
    
    try:
        field.validate([1, 2, 3, 4])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages()[0].code == "max_items"
    
    # Test exact_items
    field = Array(exact_items=2)
    assert field.validate([1, 2]) == [1, 2]
    
    try:
        field.validate([1])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages()[0].code == "exact_items"
    
    try:
        field.validate([1, 2, 3])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages()[0].code == "exact_items"
    
    # Test items validation with single validator
    item_field = Integer()
    field = Array(items=item_field)
    assert field.validate([1, 2, 3]) == [1, 2, 3]
    
    try:
        field.validate([1, "invalid", 3])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].index == [1]
    
    # Test items validation with list of validators
    item_fields = [Integer(), String(), Boolean()]
    field = Array(items=item_fields)
    assert field.validate([1, "test", True]) == [1, "test", True]
    
    try:
        field.validate([1, "test", "not boolean"])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].index == [2]
    
    # Test additional_items=False
    item_fields = [Integer(), String()]
    field = Array(items=item_fields, additional_items=False)
    assert field.validate([1, "test"]) == [1, "test"]
    
    try:
        field.validate([1, "test", "extra"])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages()[0].code == "additional_items"
    
    # Test additional_items with Field validator
    item_fields = [Integer(), String()]
    additional_field = Boolean()
    field = Array(items=item_fields, additional_items=additional_field)
    assert field.validate([1, "test", True, False]) == [1, "test", True, False]
    
    try:
        field.validate([1, "test", "not boolean"])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].index == [2]
    
    # Test unique_items
    field = Array(unique_items=True)
    assert field.validate([1, 2, 3]) == [1, 2, 3]
    
    try:
        field.validate([1, 2, 1])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages()[0].code == "unique_items"
        assert e.messages()[0].key == 2
    
    # Test complex nested validation
    nested_field = Array(items=Integer())
    field = Array(items=nested_field)
    assert field.validate([[1, 2], [3, 4]]) == [[1, 2], [3, 4]]
    
    try:
        field.validate([[1, 2], ["invalid", 4]])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].index == [1, 0]
    
    # Test with no items validator (all items pass through)
    field = Array()
    assert field.validate([1, "test", True, None]) == [1, "test", True, None]
    
    # Test error accumulation
    item_field = Integer(minimum=0)
    field = Array(items=item_field)
    try:
        field.validate([-1, -2, -3])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert len(e.messages()) == 3
        assert all(msg.code == "minimum" for msg in e.messages())
        assert [msg.index for msg in e.messages()] == [[0], [1], [2]]


# LLM-generated content at query #11
#--------------------------

```python
def test_String_validate():
    # Test basic string validation
    field = String()
    assert field.validate("hello") == "hello"
    
    # Test null handling with allow_null
    field = String(allow_null=True)
    assert field.validate(None) is None
    
    # Test null handling without allow_null
    field = String()
    try:
        field.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "null"
    
    # Test blank handling with allow_blank
    field = String(allow_blank=True)
    assert field.validate("") == ""
    
    # Test blank handling without allow_blank
    field = String()
    try:
        field.validate("")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "blank"
    
    # Test trim_whitespace
    field = String(trim_whitespace=True)
    assert field.validate("  hello  ") == "hello"
    
    field = String(trim_whitespace=False)
    assert field.validate("  hello  ") == "  hello  "
    
    # Test min_length validation
    field = String(min_length=3)
    assert field.validate("abc") == "abc"
    
    try:
        field.validate("ab")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "min_length"
    
    # Test max_length validation
    field = String(max_length=5)
    assert field.validate("abcde") == "abcde"
    
    try:
        field.validate("abcdef")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "max_length"
    
    # Test pattern validation
    field = String(pattern=r"^\d+$")
    assert field.validate("123") == "123"
    
    try:
        field.validate("abc")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "pattern"
    
    # Test format validation
    field = String(format="email")
    assert field.validate("test@example.com") == "test@example.com"
    
    try:
        field.validate("not-an-email")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "format"
    
    # Test null character removal
    field = String()
    assert field.validate("hello\0world") == "helloworld"
    
    # Test type validation for non-string input
    field = String()
    try:
        field.validate(123)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "type"
    
    # Test coerce_types with allow_blank
    field = String(allow_blank=True, coerce_types=True)
    assert field.validate(None) == ""
    
    # Test coerce_types with allow_null
    field = String(allow_null=True, coerce_types=True)
    assert field.validate("") is None
    
    # Test format with native type
    field = String(format="uuid")
    uuid_obj = formats.UUIDFormat().validate("12345678-1234-5678-1234-567812345678")
    assert field.validate(uuid_obj) == uuid_obj
    
    # Test combination of constraints
    field = String(min_length=2, max_length=4, pattern=r"^[a-z]+$")
    assert field.validate("abc") == "abc"
    
    try:
        field.validate("a")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "min_length"
    
    try:
        field.validate("abcde")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "max_length"
    
    try:
        field.validate("123")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "pattern"


# LLM-generated content at query #12
#--------------------------

```python
def test_Union():
    from core.fields import Field, Union, String, Integer, Boolean
    from core.exceptions import ValidationError

    # Test basic union with different field types
    union_field = Union(any_of=[String(), Integer()])
    assert union_field.allow_null is False
    assert len(union_field.any_of) == 2
    assert isinstance(union_field.any_of[0], String)
    assert isinstance(union_field.any_of[1], Integer)

    # Test union with null-allowing child
    union_with_null = Union(any_of=[String(allow_null=True), Integer()])
    assert union_with_null.allow_null is True

    # Test validation with string value
    result = union_field.validate("hello")
    assert result == "hello"

    # Test validation with integer value
    result = union_field.validate(123)
    assert result == 123

    # Test validation with null when allowed
    result = union_with_null.validate(None)
    assert result is None

    # Test validation with null when not allowed
    try:
        union_field.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.messages()[0].code == "null"

    # Test validation with invalid type
    try:
        union_field.validate(True)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.messages()[0].code == "union"

    # Test validation with nested errors
    string_field = String(min_length=5)
    integer_field = Integer(minimum=10)
    complex_union = Union(any_of=[string_field, integer_field])

    # Test with string that fails min_length
    try:
        complex_union.validate("hi")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.messages()[0].code == "min_length"

    # Test with integer that fails minimum
    try:
        complex_union.validate(5)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.messages()[0].code == "minimum"

    # Test with completely invalid type
    try:
        complex_union.validate(True)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.messages()[0].code == "union"

    # Test with multiple valid types
    multi_union = Union(any_of=[String(), Boolean(), Integer()])
    
    result = multi_union.validate("test")
    assert result == "test"
    
    result = multi_union.validate(True)
    assert result is True
    
    result = multi_union.validate(42)
    assert result == 42

    # Test with empty any_of list
    empty_union = Union(any_of=[])
    try:
        empty_union.validate("anything")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.messages()[0].code == "union"


# LLM-generated content at query #13
#--------------------------

```python
def test_Object_validate():
    # Test basic validation with simple properties
    schema = Object(properties={"name": String(), "age": Integer()})
    result = schema.validate({"name": "John", "age": 30})
    assert result == {"name": "John", "age": 30}

    # Test null handling with allow_null
    schema = Object(properties={"name": String()}, allow_null=True)
    assert schema.validate(None) is None

    # Test null handling without allow_null
    schema = Object(properties={"name": String()}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "null"

    # Test type validation
    schema = Object(properties={"name": String()})
    try:
        schema.validate("not an object")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "type"

    # Test required properties
    schema = Object(properties={"name": String()}, required=["name"])
    try:
        schema.validate({"age": 30})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert any(m.code == "required" for m in e.messages)

    # Test default values
    schema = Object(properties={"name": String(default="Unknown")})
    result = schema.validate({})
    assert result == {"name": "Unknown"}

    # Test min_properties
    schema = Object(properties={"a": String(), "b": String()}, min_properties=1)
    try:
        schema.validate({})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "empty"

    # Test max_properties
    schema = Object(properties={"a": String()}, max_properties=1)
    try:
        schema.validate({"a": "test", "b": "test2"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "max_properties"

    # Test additional_properties=True
    schema = Object(properties={"name": String()}, additional_properties=True)
    result = schema.validate({"name": "John", "extra": "value"})
    assert result == {"name": "John", "extra": "value"}

    # Test additional_properties=False
    schema = Object(properties={"name": String()}, additional_properties=False)
    try:
        schema.validate({"name": "John", "extra": "value"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert any(m.code == "invalid_property" for m in e.messages)

    # Test additional_properties as Field
    schema = Object(
        properties={"name": String()},
        additional_properties=Integer()
    )
    result = schema.validate({"name": "John", "extra": 42})
    assert result == {"name": "John", "extra": 42}

    # Test pattern_properties
    schema = Object(
        pattern_properties={"^test_": String()}
    )
    result = schema.validate({"test_field": "value"})
    assert result == {"test_field": "value"}

    # Test property_names validation
    schema = Object(
        property_names=String(pattern="^[a-z]+$"),
        additional_properties=True
    )
    try:
        schema.validate({"UPPERCASE": "value"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert any(m.code == "invalid_property" for m in e.messages)

    # Test invalid key type
    schema = Object(additional_properties=True)
    try:
        schema.validate({123: "value"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert any(m.code == "invalid_key" for m in e.messages)

    # Test nested validation errors
    schema = Object(properties={
        "nested": Object(properties={
            "inner": Integer()
        })
    })
    try:
        schema.validate({"nested": {"inner": "not an int"}})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages) > 0

    # Test complex scenario with multiple validations
    schema = Object(
        properties={
            "name": String(min_length=2),
            "age": Integer(minimum=0)
        },
        required=["name"],
        min_properties=1,
        max_properties=3
    )
    result = schema.validate({"name": "John", "age": 30})
    assert result == {"name": "John", "age": 30}


# LLM-generated content at query #14
#--------------------------

```python
def test_Object_validate():
    # Test basic validation with simple properties
    schema = Object(properties={"name": String(), "age": Integer()})
    assert schema.validate({"name": "John", "age": 30}) == {"name": "John", "age": 30}
    
    # Test null handling
    schema = Object(properties={"name": String()}, allow_null=True)
    assert schema.validate(None) is None
    
    # Test null not allowed
    schema = Object(properties={"name": String()}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "null"
    
    # Test type validation
    schema = Object(properties={"name": String()})
    try:
        schema.validate("not an object")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "type"
    
    # Test required properties
    schema = Object(properties={"name": String()}, required=["name"])
    try:
        schema.validate({})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "required"
    
    # Test default values
    schema = Object(properties={"name": String(default="Unknown")})
    assert schema.validate({}) == {"name": "Unknown"}
    
    # Test min_properties
    schema = Object(properties={"a": String(), "b": String()}, min_properties=1)
    try:
        schema.validate({})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "empty"
    
    # Test max_properties
    schema = Object(properties={"a": String()}, max_properties=1)
    try:
        schema.validate({"a": "test", "b": "extra"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "max_properties"
    
    # Test pattern_properties
    schema = Object(pattern_properties={"^test_": String()})
    result = schema.validate({"test_key": "value", "other": "ignored"})
    assert result == {"test_key": "value", "other": "ignored"}
    
    # Test additional_properties=False
    schema = Object(
        properties={"name": String()},
        additional_properties=False
    )
    try:
        schema.validate({"name": "John", "extra": "field"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert any(msg.code == "invalid_property" for msg in e.messages)
    
    # Test additional_properties as Field
    schema = Object(
        properties={"name": String()},
        additional_properties=Integer()
    )
    result = schema.validate({"name": "John", "extra": 42})
    assert result == {"name": "John", "extra": 42}
    
    # Test property_names validation
    schema = Object(
        properties={},
        property_names=String(pattern="^[a-z]+$")
    )
    try:
        schema.validate({"INVALID": "value"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert any(msg.code == "invalid_property" for msg in e.messages)
    
    # Test invalid key type
    schema = Object(properties={"name": String()})
    try:
        schema.validate({123: "value"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert any(msg.code == "invalid_key" for msg in e.messages)
    
    # Test nested validation errors
    schema = Object(properties={
        "person": Object(properties={
            "age": Integer(minimum=0)
        })
    })
    try:
        schema.validate({"person": {"age": -5}})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages) > 0
    
    # Test complex scenario with multiple validations
    schema = Object(
        properties={"id": Integer(), "email": String()},
        required=["id"],
        min_properties=1,
        max_properties=3
    )
    result = schema.validate({"id": 1, "email": "test@example.com"})
    assert result == {"id": 1, "email": "test@example.com"}


# LLM-generated content at query #15
#--------------------------

```python
def test_Array_validate():
    # Test basic validation
    field = Array()
    assert field.validate([1, 2, 3]) == [1, 2, 3]
    
    # Test null handling
    field = Array(allow_null=True)
    assert field.validate(None) is None
    
    field = Array(allow_null=False)
    try:
        field.validate(None)
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages[0].code == "null"
    
    # Test type validation
    field = Array()
    try:
        field.validate("not a list")
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages[0].code == "type"
    
    # Test min_items
    field = Array(min_items=2)
    assert field.validate([1, 2]) == [1, 2]
    assert field.validate([1, 2, 3]) == [1, 2, 3]
    
    try:
        field.validate([1])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages[0].code == "min_items"
    
    # Test min_items = 1 gives "empty" error
    field = Array(min_items=1)
    try:
        field.validate([])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages[0].code == "empty"
    
    # Test max_items
    field = Array(max_items=3)
    assert field.validate([1, 2, 3]) == [1, 2, 3]
    assert field.validate([1, 2]) == [1, 2]
    
    try:
        field.validate([1, 2, 3, 4])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages[0].code == "max_items"
    
    # Test exact_items
    field = Array(exact_items=3)
    assert field.validate([1, 2, 3]) == [1, 2, 3]
    
    try:
        field.validate([1, 2])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages[0].code == "exact_items"
    
    try:
        field.validate([1, 2, 3, 4])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages[0].code == "exact_items"
    
    # Test items validation with single Field
    field = Array(items=Integer())
    assert field.validate([1, 2, 3]) == [1, 2, 3]
    
    try:
        field.validate([1, "invalid", 3])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages[0].code == "type"
        assert e.messages[0].index == [1]
    
    # Test items validation with list of Fields
    field = Array(items=[Integer(), String()])
    assert field.validate([1, "hello"]) == [1, "hello"]
    
    try:
        field.validate(["invalid", "hello"])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages[0].code == "type"
        assert e.messages[0].index == [0]
    
    # Test additional_items = False
    field = Array(items=[Integer(), String()], additional_items=False)
    assert field.validate([1, "hello"]) == [1, "hello"]
    
    try:
        field.validate([1, "hello", "extra"])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages[0].code == "additional_items"
    
    # Test additional_items = True
    field = Array(items=[Integer(), String()], additional_items=True)
    assert field.validate([1, "hello", "extra", 4]) == [1, "hello", "extra", 4]
    
    # Test additional_items as Field
    field = Array(items=[Integer()], additional_items=String())
    assert field.validate([1, "hello", "world"]) == [1, "hello", "world"]
    
    try:
        field.validate([1, 2, 3])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages[0].code == "type"
        assert e.messages[0].index == [1]
    
    # Test unique_items
    field = Array(unique_items=True)
    assert field.validate([1, 2, 3]) == [1, 2, 3]
    
    try:
        field.validate([1, 2, 1])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages[0].code == "unique_items"
        assert e.messages[0].key == 2
    
    # Test complex nested validation
    field = Array(items=Object(properties={"id": Integer(), "name": String()}))
    data = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
    assert field.validate(data) == data
    
    try:
        field.validate([{"id": "invalid", "name": "Alice"}])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages[0].code == "type"
        assert e.messages[0].index == [0, "id"]
    
    # Test error accumulation
    field = Array(items=Integer())
    try:
        field.validate(["a", "b", "c"])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert len(e.messages) == 3
        assert all(msg.code == "type" for msg in e.messages)
        assert [msg.index[0] for msg in e.messages] == [0, 1, 2]
    
    # Test with no items validator
    field = Array()
    complex_data = [1, "string", {"key": "value"}, [1, 2, 3]]
    assert field.validate(complex_data) == complex_data


# LLM-generated content at query #16
#--------------------------

```python
def test_Union_validate():
    from decimal import Decimal
    from typing import List
    
    # Test 1: Union with null allowed
    field1 = Integer(allow_null=True)
    field2 = String(allow_null=True)
    union_field = Union(any_of=[field1, field2])
    
    assert union_field.validate(None) is None
    
    # Test 2: Union with null not allowed
    field1 = Integer()
    field2 = String()
    union_field = Union(any_of=[field1, field2])
    
    try:
        union_field.validate(None)
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "null"
    
    # Test 3: Union matches first field
    field1 = Integer()
    field2 = String()
    union_field = Union(any_of=[field1, field2])
    
    assert union_field.validate(42) == 42
    
    # Test 4: Union matches second field
    field1 = Integer()
    field2 = String()
    union_field = Union(any_of=[field1, field2])
    
    assert union_field.validate("test") == "test"
    
    # Test 5: Union with nested validation errors
    field1 = Integer(minimum=10)
    field2 = String(min_length=5)
    union_field = Union(any_of=[field1, field2])
    
    # Should match first field but fail minimum constraint
    try:
        union_field.validate(5)
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "minimum"
    
    # Test 6: Union with type mismatch for all fields
    field1 = Integer()
    field2 = String()
    union_field = Union(any_of=[field1, field2])
    
    try:
        union_field.validate(True)
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "union"
    
    # Test 7: Union with multiple candidate errors
    field1 = Integer(minimum=10)
    field2 = Integer(maximum=5)
    union_field = Union(any_of=[field1, field2])
    
    # Value 7 doesn't match either constraint
    try:
        union_field.validate(7)
        assert False, "Should have raised validation error"
    except ValidationError as e:
        # Should raise the single candidate error
        assert len(e.messages()) == 1
    
    # Test 8: Union with complex nested types
    field1 = Array(items=Integer())
    field2 = Object(properties={"value": Integer()})
    union_field = Union(any_of=[field1, field2])
    
    assert union_field.validate([1, 2, 3]) == [1, 2, 3]
    assert union_field.validate({"value": 42}) == {"value": 42}
    
    # Test 9: Union with mixed null policies
    field1 = Integer(allow_null=True)
    field2 = String()
    union_field = Union(any_of=[field1, field2])
    
    # Should allow null because at least one child allows it
    assert union_field.validate(None) is None
    
    # Test 10: Union with Decimal type
    field1 = Decimal()
    field2 = String()
    union_field = Union(any_of=[field1, field2])
    
    result = union_field.validate("123.45")
    assert isinstance(result, Decimal)
    assert result == Decimal("123.45")


# LLM-generated content at query #17
#--------------------------

```python
def test_Choice_validate():
    # Test basic validation with valid choices
    field = Choice(choices=[("a", "Option A"), ("b", "Option B")])
    assert field.validate("a") == "a"
    assert field.validate("b") == "b"
    
    # Test validation with allow_null=True
    field = Choice(choices=[("a", "Option A"), ("b", "Option B")], allow_null=True)
    assert field.validate(None) is None
    
    # Test validation with allow_null=False (default)
    field = Choice(choices=[("a", "Option A"), ("b", "Option B")])
    try:
        field.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "null"
    
    # Test validation with invalid choice
    field = Choice(choices=[("a", "Option A"), ("b", "Option B")])
    try:
        field.validate("c")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "choice"
    
    # Test validation with empty string and coerce_types=True (default)
    field = Choice(choices=[("a", "Option A"), ("b", "Option B")])
    try:
        field.validate("")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "required"
    
    # Test validation with empty string, allow_null=True, and coerce_types=True
    field = Choice(choices=[("a", "Option A"), ("b", "Option B")], allow_null=True)
    assert field.validate("") is None
    
    # Test validation with empty string, allow_null=True, and coerce_types=False
    field = Choice(choices=[("a", "Option A"), ("b", "Option B")], allow_null=True, coerce_types=False)
    try:
        field.validate("")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "choice"
    
    # Test validation with numeric choices
    field = Choice(choices=[(1, "One"), (2, "Two")])
    assert field.validate(1) == 1
    assert field.validate(2) == 2
    
    # Test validation with mixed type choices
    field = Choice(choices=[("1", "String One"), (2, "Number Two")])
    assert field.validate("1") == "1"
    assert field.validate(2) == 2
    
    # Test validation with empty choices list
    field = Choice(choices=[])
    try:
        field.validate("anything")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "choice"
    
    # Test validation with None choices (defaults to empty list)
    field = Choice()
    try:
        field.validate("anything")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "choice"


# LLM-generated content at query #18
#--------------------------

```python
def test_Choice_validate():
    # Test basic validation with valid choices
    field = Choice(choices=[("a", "Option A"), ("b", "Option B")])
    assert field.validate("a") == "a"
    assert field.validate("b") == "b"
    
    # Test with allow_null=True
    field = Choice(choices=[("a", "A"), ("b", "B")], allow_null=True)
    assert field.validate(None) is None
    
    # Test with allow_null=False (default)
    field = Choice(choices=[("a", "A"), ("b", "B")])
    try:
        field.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "null"
    
    # Test invalid choice
    field = Choice(choices=[("a", "A"), ("b", "B")])
    try:
        field.validate("c")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "choice"
    
    # Test empty string with allow_null=False and coerce_types=True (default)
    field = Choice(choices=[("a", "A"), ("b", "B")])
    try:
        field.validate("")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "required"
    
    # Test empty string with allow_null=True and coerce_types=True (default)
    field = Choice(choices=[("a", "A"), ("b", "B")], allow_null=True)
    assert field.validate("") is None
    
    # Test with coerce_types=False
    field = Choice(choices=[("a", "A"), ("b", "B")], coerce_types=False)
    try:
        field.validate("")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "choice"
    
    # Test with single string choices (should be converted to tuples)
    field = Choice(choices=["a", "b"])
    assert field.validate("a") == "a"
    assert field.validate("b") == "b"
    
    # Test with mixed choices format
    field = Choice(choices=["a", ("b", "Option B")])
    assert field.validate("a") == "a"
    assert field.validate("b") == "b"
    
    # Test validation with Uniqueness wrapper
    field = Choice(choices=[("a", "A"), ("b", "B"), ("c", "C")])
    assert field.validate("c") == "c"
    
    # Test that choices are properly formatted as tuples
    assert all(len(choice) == 2 for choice in field.choices)


# LLM-generated content at query #19
#--------------------------

```python
def test_Field_get_default_value():
    # Test with no default value set
    field_without_default = Field()
    assert not hasattr(field_without_default, 'default')
    assert field_without_default.get_default_value() is None

    # Test with non-callable default value
    field_with_static_default = Field(default="test_value")
    assert field_with_static_default.get_default_value() == "test_value"

    # Test with integer default value
    field_with_int_default = Field(default=42)
    assert field_with_int_default.get_default_value() == 42

    # Test with list default value
    field_with_list_default = Field(default=[1, 2, 3])
    assert field_with_list_default.get_default_value() == [1, 2, 3]

    # Test with callable default value (function)
    def sample_function():
        return "dynamic_value"

    field_with_callable_func = Field(default=sample_function)
    assert field_with_callable_func.get_default_value() == "dynamic_value"

    # Test with callable default value (lambda)
    field_with_lambda = Field(default=lambda: 100)
    assert field_with_lambda.get_default_value() == 100

    # Test with callable default value that returns None
    field_with_none_returning_callable = Field(default=lambda: None)
    assert field_with_none_returning_callable.get_default_value() is None

    # Test with callable default value that returns complex object
    field_with_complex_callable = Field(default=lambda: {"key": "value"})
    assert field_with_complex_callable.get_default_value() == {"key": "value"}

    # Test that callable is actually called, not returned
    callable_obj = lambda: "called"
    field = Field(default=callable_obj)
    result = field.get_default_value()
    assert result == "called"
    assert not callable(result)  # Should return the result, not the callable

    # Test with allow_null=True (should set default to None)
    field_with_allow_null = Field(allow_null=True)
    assert field_with_allow_null.default is None
    assert field_with_allow_null.get_default_value() is None

    # Test with allow_null=True and explicit default
    field_with_allow_null_and_default = Field(allow_null=True, default="explicit")
    assert field_with_allow_null_and_default.get_default_value() == "explicit"


# LLM-generated content at query #20
#--------------------------

```python
def test_Field_get_default_value():
    # Test with no default value
    field_without_default = Field()
    assert not field_without_default.has_default()
    
    # Test with non-callable default value
    field_with_static_default = Field(default="test_value")
    assert field_with_static_default.has_default()
    assert field_with_static_default.get_default_value() == "test_value"
    
    # Test with callable default value
    def custom_default():
        return "dynamic_value"
    
    field_with_callable_default = Field(default=custom_default)
    assert field_with_callable_default.has_default()
    assert field_with_callable_default.get_default_value() == "dynamic_value"
    
    # Test with callable that returns different values
    counter = 0
    def counter_default():
        nonlocal counter
        counter += 1
        return counter
    
    field_with_counter_default = Field(default=counter_default)
    assert field_with_counter_default.get_default_value() == 1
    assert field_with_counter_default.get_default_value() == 2
    
    # Test with None as default value
    field_with_none_default = Field(default=None)
    assert field_with_none_default.has_default()
    assert field_with_none_default.get_default_value() is None
    
    # Test with allow_null=True (should set default to None)
    field_with_allow_null = Field(allow_null=True)
    assert field_with_allow_null.has_default()
    assert field_with_allow_null.get_default_value() is None
    
    # Test with allow_null=True and explicit default
    field_with_allow_null_and_default = Field(allow_null=True, default="explicit")
    assert field_with_allow_null_and_default.has_default()
    assert field_with_allow_null_and_default.get_default_value() == "explicit"


# LLM-generated content at query #21
#--------------------------

```python
def test_String_validate():
    # Test basic string validation
    field = String()
    assert field.validate("hello") == "hello"
    
    # Test null handling
    field = String(allow_null=True)
    assert field.validate(None) is None
    
    field = String(allow_null=False)
    try:
        field.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "null"
    
    # Test type validation
    try:
        field.validate(123)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "type"
    
    # Test blank handling
    field = String(allow_blank=True)
    assert field.validate("") == ""
    
    field = String(allow_blank=False)
    try:
        field.validate("")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "blank"
    
    # Test trim_whitespace
    field = String(trim_whitespace=True)
    assert field.validate("  hello  ") == "hello"
    
    field = String(trim_whitespace=False)
    assert field.validate("  hello  ") == "  hello  "
    
    # Test min_length
    field = String(min_length=3)
    assert field.validate("abc") == "abc"
    
    try:
        field.validate("ab")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "min_length"
    
    # Test max_length
    field = String(max_length=5)
    assert field.validate("abcde") == "abcde"
    
    try:
        field.validate("abcdef")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "max_length"
    
    # Test pattern matching
    field = String(pattern=r"^\d+$")
    assert field.validate("123") == "123"
    
    try:
        field.validate("abc")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "pattern"
    
    # Test format validation
    field = String(format="email")
    assert field.validate("test@example.com") == "test@example.com"
    
    try:
        field.validate("not-an-email")
        assert False, "Should have raised ValidationError"
    except ValidationError:
        pass
    
    # Test null character removal
    field = String()
    assert field.validate("hello\0world") == "helloworld"
    
    # Test coerce_types with allow_blank
    field = String(allow_blank=True, coerce_types=True)
    assert field.validate(None) == ""
    
    # Test coerce_types with allow_null
    field = String(allow_null=True, coerce_types=True, trim_whitespace=True)
    assert field.validate("") is None
    
    # Test format with native type
    field = String(format="uuid")
    uuid_obj = formats.UUIDFormat().validate("12345678-1234-5678-1234-567812345678")
    assert field.validate(uuid_obj) == uuid_obj
    
    # Test combined constraints
    field = String(min_length=2, max_length=4, pattern=r"^[a-z]+$")
    assert field.validate("abc") == "abc"
    
    try:
        field.validate("a")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "min_length"
    
    try:
        field.validate("abcde")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "max_length"
    
    try:
        field.validate("123")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "pattern"


# LLM-generated content at query #22
#--------------------------

```python
def test_Choice_validate():
    # Test basic validation with valid choices
    field = Choice(choices=[("a", "Option A"), ("b", "Option B")])
    assert field.validate("a") == "a"
    assert field.validate("b") == "b"
    
    # Test validation with null allowed
    field = Choice(choices=[("a", "Option A")], allow_null=True)
    assert field.validate(None) is None
    
    # Test validation with null not allowed
    field = Choice(choices=[("a", "Option A")], allow_null=False)
    try:
        field.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "null"
    
    # Test validation with empty string when allow_null and coerce_types
    field = Choice(choices=[("a", "Option A")], allow_null=True, coerce_types=True)
    assert field.validate("") is None
    
    # Test validation with empty string when not allow_null
    field = Choice(choices=[("a", "Option A")], allow_null=False)
    try:
        field.validate("")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "required"
    
    # Test validation with invalid choice
    field = Choice(choices=[("a", "Option A"), ("b", "Option B")])
    try:
        field.validate("c")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "choice"
    
    # Test validation with tuple choices
    field = Choice(choices=[("key1", "Value 1"), ("key2", "Value 2")])
    assert field.validate("key1") == "key1"
    assert field.validate("key2") == "key2"
    
    # Test validation with list choices
    field = Choice(choices=[["key1", "Value 1"], ["key2", "Value 2"]])
    assert field.validate("key1") == "key1"
    
    # Test validation with string choices (auto-converted to tuples)
    field = Choice(choices=["option1", "option2"])
    assert field.validate("option1") == "option1"
    assert field.validate("option2") == "option2"
    
    # Test validation with numeric choices
    field = Choice(choices=[(1, "One"), (2, "Two")])
    assert field.validate(1) == 1
    assert field.validate(2) == 2
    
    # Test validation with boolean choices
    field = Choice(choices=[(True, "Yes"), (False, "No")])
    assert field.validate(True) is True
    assert field.validate(False) is False


# LLM-generated content at query #23
#--------------------------

```python
def test_Const():
    # Test basic const validation
    const_field = Const("test_value")
    assert const_field.validate("test_value") == "test_value"
    
    # Test const with None value
    const_none = Const(None)
    assert const_none.validate(None) is None
    
    # Test const with integer value
    const_int = Const(42)
    assert const_int.validate(42) == 42
    
    # Test const with float value
    const_float = Const(3.14)
    assert const_float.validate(3.14) == 3.14
    
    # Test const with boolean value
    const_bool = Const(True)
    assert const_bool.validate(True) is True
    
    # Test const with list value
    const_list = Const([1, 2, 3])
    assert const_field.validate([1, 2, 3]) == [1, 2, 3]
    
    # Test const with dict value
    const_dict = Const({"key": "value"})
    assert const_field.validate({"key": "value"}) == {"key": "value"}
    
    # Test error when value doesn't match const
    const_field = Const("expected")
    try:
        const_field.validate("wrong")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "const"
    
    # Test error when expecting None but got value
    const_none = Const(None)
    try:
        const_none.validate("not_none")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "only_null"
    
    # Test error when expecting value but got None
    const_field = Const("expected")
    try:
        const_field.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert len(e.messages) == 1
        assert e.messages[0].code == "const"


# LLM-generated content at query #24
#--------------------------

```python
def test_Array_validate():
    # Test basic validation
    field = Array()
    assert field.validate([1, 2, 3]) == [1, 2, 3]
    assert field.validate([]) == []
    
    # Test null handling
    field = Array(allow_null=True)
    assert field.validate(None) is None
    
    field = Array(allow_null=False)
    try:
        field.validate(None)
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages[0].code == "null"
    
    # Test type validation
    field = Array()
    try:
        field.validate("not a list")
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages[0].code == "type"
    
    # Test min_items
    field = Array(min_items=2)
    assert field.validate([1, 2]) == [1, 2]
    assert field.validate([1, 2, 3]) == [1, 2, 3]
    
    try:
        field.validate([1])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages[0].code == "min_items"
    
    # Test min_items with empty array
    field = Array(min_items=1)
    try:
        field.validate([])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages[0].code == "empty"
    
    # Test max_items
    field = Array(max_items=3)
    assert field.validate([1, 2, 3]) == [1, 2, 3]
    assert field.validate([1, 2]) == [1, 2]
    
    try:
        field.validate([1, 2, 3, 4])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages[0].code == "max_items"
    
    # Test exact_items
    field = Array(exact_items=3)
    assert field.validate([1, 2, 3]) == [1, 2, 3]
    
    try:
        field.validate([1, 2])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages[0].code == "exact_items"
    
    try:
        field.validate([1, 2, 3, 4])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages[0].code == "exact_items"
    
    # Test items validation with single Field
    item_field = Integer()
    field = Array(items=item_field)
    assert field.validate([1, 2, 3]) == [1, 2, 3]
    
    try:
        field.validate([1, "invalid", 3])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages[0].code == "type"
    
    # Test items validation with list of Fields
    item_fields = [Integer(), String(), Boolean()]
    field = Array(items=item_fields)
    assert field.validate([1, "test", True]) == [1, "test", True]
    
    try:
        field.validate([1, "test", "not boolean"])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages[0].code == "type"
    
    # Test additional_items=False
    item_fields = [Integer(), String()]
    field = Array(items=item_fields, additional_items=False)
    assert field.validate([1, "test"]) == [1, "test"]
    
    try:
        field.validate([1, "test", "extra"])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages[0].code == "additional_items"
    
    # Test additional_items as Field
    item_fields = [Integer(), String()]
    additional_field = Boolean()
    field = Array(items=item_fields, additional_items=additional_field)
    assert field.validate([1, "test", True, False]) == [1, "test", True, False]
    
    try:
        field.validate([1, "test", "not boolean"])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages[0].code == "type"
    
    # Test unique_items
    field = Array(unique_items=True)
    assert field.validate([1, 2, 3]) == [1, 2, 3]
    
    try:
        field.validate([1, 2, 1])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages[0].code == "unique_items"
    
    # Test complex nested validation
    nested_field = Array(items=Integer())
    field = Array(items=nested_field)
    assert field.validate([[1, 2], [3, 4]]) == [[1, 2], [3, 4]]
    
    try:
        field.validate([[1, 2], ["invalid", 4]])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages[0].code == "type"
    
    # Test error message accumulation
    item_fields = [Integer(), Integer()]
    field = Array(items=item_fields)
    
    try:
        field.validate(["invalid1", "invalid2"])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert len(e.messages) == 2
        assert all(msg.code == "type" for msg in e.messages)


# LLM-generated content at query #25
#--------------------------

```python
def test_Boolean_validate():
    # Test basic boolean validation
    field = Boolean()
    assert field.validate(True) is True
    assert field.validate(False) is False
    
    # Test null handling
    field = Boolean(allow_null=True)
    assert field.validate(None) is None
    
    field = Boolean(allow_null=False)
    try:
        field.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "null"
    
    # Test type coercion with coerce_types=True (default)
    field = Boolean(coerce_types=True)
    assert field.validate("true") is True
    assert field.validate("false") is False
    assert field.validate("on") is True
    assert field.validate("off") is False
    assert field.validate("1") is True
    assert field.validate("0") is False
    assert field.validate("") is False
    assert field.validate(1) is True
    assert field.validate(0) is False
    
    # Test type coercion with allow_null
    field = Boolean(coerce_types=True, allow_null=True)
    assert field.validate("null") is None
    assert field.validate("none") is None
    assert field.validate("") is None
    
    # Test without type coercion
    field = Boolean(coerce_types=False)
    assert field.validate(True) is True
    assert field.validate(False) is False
    
    try:
        field.validate("true")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "type"
    
    try:
        field.validate(1)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "type"
    
    # Test invalid values with coercion
    field = Boolean(coerce_types=True)
    try:
        field.validate("invalid")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "type"
    
    try:
        field.validate(2)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "type"
    
    # Test case sensitivity in string coercion
    field = Boolean(coerce_types=True)
    assert field.validate("TRUE") is True
    assert field.validate("FALSE") is False
    assert field.validate("On") is True
    assert field.validate("Off") is False
    
    # Test validation_or_error method
    field = Boolean()
    result = field.validate_or_error(True)
    assert result.value is True
    assert result.error is None
    
    result = field.validate_or_error("invalid")
    assert result.value is None
    assert result.error is not None
    assert result.error.code == "type"


# LLM-generated content at query #26
#--------------------------

```python
def test_Union_validate():
    from decimal import Decimal
    from typing import List

    # Test 1: Union with null allowed
    field1 = Integer(allow_null=True)
    field2 = String(allow_null=True)
    union = Union(any_of=[field1, field2])
    assert union.validate(None) is None

    # Test 2: Union with null not allowed
    field1 = Integer()
    field2 = String()
    union = Union(any_of=[field1, field2])
    try:
        union.validate(None)
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "null"

    # Test 3: Union with one matching type
    field1 = Integer()
    field2 = String()
    union = Union(any_of=[field1, field2])
    assert union.validate(42) == 42
    assert union.validate("hello") == "hello"

    # Test 4: Union with no matching type
    field1 = Integer()
    field2 = String()
    union = Union(any_of=[field1, field2])
    try:
        union.validate(3.14)
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "union"

    # Test 5: Union with nested validation errors
    field1 = Integer(minimum=10)
    field2 = String(min_length=5)
    union = Union(any_of=[field1, field2])
    
    # Should match Integer type but fail minimum validation
    try:
        union.validate(5)
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "minimum"

    # Test 6: Union with multiple candidate errors
    field1 = Integer(minimum=10)
    field2 = String(min_length=5)
    union = Union(any_of=[field1, field2])
    
    # Should match both types but fail both validations
    try:
        union.validate(7)
        assert False, "Should have raised validation error"
    except ValidationError as e:
        # Should get union error since multiple candidates failed
        assert len(e.messages()) == 1
        assert e.messages()[0].code == "union"

    # Test 7: Union with mixed null policies
    field1 = Integer(allow_null=True)
    field2 = String()
    union = Union(any_of=[field1, field2])
    assert union.allow_null is True
    assert union.validate(None) is None

    # Test 8: Union with complex nested types
    field1 = Array(items=Integer())
    field2 = Object(properties={"value": Integer()})
    union = Union(any_of=[field1, field2])
    
    assert union.validate([1, 2, 3]) == [1, 2, 3]
    assert union.validate({"value": 42}) == {"value": 42}

    # Test 9: Union with type coercion
    field1 = Integer(coerce_types=True)
    field2 = Boolean(coerce_types=True)
    union = Union(any_of=[field1, field2])
    
    assert union.validate("42") == 42
    assert union.validate("true") is True

    # Test 10: Union with Decimal type
    field1 = Decimal()
    field2 = Float()
    union = Union(any_of=[field1, field2])
    
    assert union.validate(Decimal("3.14")) == Decimal("3.14")
    assert union.validate(3.14) == 3.14


# LLM-generated content at query #27
#--------------------------

```python
def test_Const():
    # Test with integer constant
    const_int = Const(42)
    assert const_int.const == 42
    assert const_int.allow_null is False

    # Test with string constant
    const_str = Const("test")
    assert const_str.const == "test"
    assert const_str.allow_null is False

    # Test with None constant
    const_none = Const(None)
    assert const_none.const is None
    assert const_none.allow_null is False

    # Test with boolean constant
    const_bool = Const(True)
    assert const_bool.const is True
    assert const_bool.allow_null is False

    # Test with list constant
    const_list = Const([1, 2, 3])
    assert const_list.const == [1, 2, 3]
    assert const_list.allow_null is False

    # Test with dict constant
    const_dict = Const({"key": "value"})
    assert const_dict.const == {"key": "value"}
    assert const_str.allow_null is False

    # Test that allow_null cannot be explicitly set
    const_default = Const("default")
    assert const_default.allow_null is False


# LLM-generated content at query #28
#--------------------------

```python
def test_Union_validate():
    from decimal import Decimal
    from unittest.mock import Mock

    # Test 1: Union with null allowed
    field1 = Mock(spec=Field, allow_null=False)
    field2 = Mock(spec=Field, allow_null=True)
    union = Union(any_of=[field1, field2])
    assert union.allow_null == True
    
    # Test 2: Null value when allow_null=True
    field1 = Mock(spec=Field, allow_null=False)
    field2 = Mock(spec=Field, allow_null=True)
    union = Union(any_of=[field1, field2])
    result = union.validate(None)
    assert result is None
    
    # Test 3: Null value when allow_null=False
    field1 = Mock(spec=Field, allow_null=False)
    field2 = Mock(spec=Field, allow_null=False)
    union = Union(any_of=[field1, field2])
    try:
        union.validate(None)
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert any(msg.code == "null" for msg in e.messages)
    
    # Test 4: Valid value matching first child
    field1 = Mock(spec=Field)
    field1.validate_or_error.return_value = ("valid", None)
    field2 = Mock(spec=Field)
    field2.validate_or_error.return_value = (None, Mock())
    union = Union(any_of=[field1, field2])
    result = union.validate("test")
    assert result == "valid"
    
    # Test 5: Valid value matching second child
    field1 = Mock(spec=Field)
    field1.validate_or_error.return_value = (None, Mock())
    field2 = Mock(spec=Field)
    field2.validate_or_error.return_value = ("valid", None)
    union = Union(any_of=[field1, field2])
    result = union.validate("test")
    assert result == "valid"
    
    # Test 6: No matching child - union error
    field1 = Mock(spec=Field)
    error1 = Mock()
    msg1 = Mock(code="type", index=None)
    error1.messages.return_value = [msg1]
    field1.validate_or_error.return_value = (None, error1)
    
    field2 = Mock(spec=Field)
    error2 = Mock()
    msg2 = Mock(code="type", index=None)
    error2.messages.return_value = [msg2]
    field2.validate_or_error.return_value = (None, error2)
    
    union = Union(any_of=[field1, field2])
    try:
        union.validate("test")
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert any(msg.code == "union" for msg in e.messages)
    
    # Test 7: Child with non-type error (should be candidate)
    field1 = Mock(spec=Field)
    error1 = Mock()
    msg1 = Mock(code="minimum", index=None)
    error1.messages.return_value = [msg1]
    field1.validate_or_error.return_value = (None, error1)
    
    field2 = Mock(spec=Field)
    error2 = Mock()
    msg2 = Mock(code="type", index=None)
    error2.messages.return_value = [msg2]
    field2.validate_or_error.return_value = (None, error2)
    
    union = Union(any_of=[field1, field2])
    try:
        union.validate("test")
        assert False, "Should have raised validation error"
    except ValidationError as e:
        # Should raise error1 since it's not just a type error
        assert error1.messages.called
    
    # Test 8: Child with type error but with index (should be candidate)
    field1 = Mock(spec=Field)
    error1 = Mock()
    msg1 = Mock(code="type", index=["nested"])
    error1.messages.return_value = [msg1]
    field1.validate_or_error.return_value = (None, error1)
    
    field2 = Mock(spec=Field)
    error2 = Mock()
    msg2 = Mock(code="type", index=None)
    error2.messages.return_value = [msg2]
    field2.validate_or_error.return_value = (None, error2)
    
    union = Union(any_of=[field1, field2])
    try:
        union.validate("test")
        assert False, "Should have raised validation error"
    except ValidationError as e:
        # Should raise error1 since it has an index
        assert error1.messages.called
    
    # Test 9: Multiple candidate errors - should raise union error
    field1 = Mock(spec=Field)
    error1 = Mock()
    msg1 = Mock(code="minimum", index=None)
    error1.messages.return_value = [msg1]
    field1.validate_or_error.return_value = (None, error1)
    
    field2 = Mock(spec=Field)
    error2 = Mock()
    msg2 = Mock(code="maximum", index=None)
    error2.messages.return_value = [msg2]
    field2.validate_or_error.return_value = (None, error2)
    
    union = Union(any_of=[field1, field2])
    try:
        union.validate("test")
        assert False, "Should have raised validation error"
    except ValidationError as e:
        # Should raise union error since multiple candidates
        assert any(msg.code == "union" for msg in e.messages)


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_Boolean_validate():
    # Test basic boolean validation
    field = Boolean()
    assert field.validate(True) is True
    assert field.validate(False) is False
    
    # Test null handling
    field_null = Boolean(allow_null=True)
    assert field_null.validate(None) is None
    
    # Test null without allow_null
    field_not_null = Boolean()
    try:
        field_not_null.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "null"
    
    # Test type error without coercion
    field_no_coerce = Boolean(coerce_types=False)
    try:
        field_no_coerce.validate("true")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "type"
    
    # Test string coercion
    field_coerce = Boolean(coerce_types=True)
    assert field_coerce.validate("true") is True
    assert field_coerce.validate("false") is False
    assert field_coerce.validate("on") is True
    assert field_coerce.validate("off") is False
    assert field_coerce.validate("1") is True
    assert field_coerce.validate("0") is False
    assert field_coerce.validate("") is False
    
    # Test numeric coercion
    assert field_coerce.validate(1) is True
    assert field_coerce.validate(0) is False
    
    # Test case insensitive string coercion
    assert field_coerce.validate("TRUE") is True
    assert field_coerce.validate("FALSE") is False
    assert field_coerce.validate("ON") is True
    assert field_coerce.validate("OFF") is False
    
    # Test null coercion with allow_null
    field_coerce_null = Boolean(allow_null=True, coerce_types=True)
    assert field_coerce_null.validate("null") is None
    assert field_coerce_null.validate("none") is None
    assert field_coerce_null.validate("") is None
    
    # Test invalid string values
    try:
        field_coerce.validate("invalid")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "type"
    
    # Test invalid type
    try:
        field_coerce.validate([])
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "type"


# LLM-generated content at query #2
#--------------------------

```python
def test_Choice_validate():
    # Test basic validation with valid choices
    field = Choice(choices=[("a", "Option A"), ("b", "Option B")])
    assert field.validate("a") == "a"
    assert field.validate("b") == "b"
    
    # Test validation with allow_null=True
    field = Choice(choices=[("a", "Option A"), ("b", "Option B")], allow_null=True)
    assert field.validate(None) is None
    
    # Test validation with allow_null=False (default)
    field = Choice(choices=[("a", "Option A"), ("b", "Option B")])
    try:
        field.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "null"
    
    # Test validation with invalid choice
    field = Choice(choices=[("a", "Option A"), ("b", "Option B")])
    try:
        field.validate("c")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "choice"
    
    # Test validation with empty string and coerce_types=True
    field = Choice(choices=[("a", "Option A"), ("b", "Option B")], coerce_types=True)
    try:
        field.validate("")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "required"
    
    # Test validation with empty string, allow_null=True, and coerce_types=True
    field = Choice(choices=[("a", "Option A"), ("b", "Option B")], 
                   allow_null=True, coerce_types=True)
    assert field.validate("") is None
    
    # Test validation with empty string, allow_null=False, and coerce_types=True
    field = Choice(choices=[("a", "Option A"), ("b", "Option B")], 
                   allow_null=False, coerce_types=True)
    try:
        field.validate("")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "required"
    
    # Test validation with numeric choices
    field = Choice(choices=[(1, "One"), (2, "Two")])
    assert field.validate(1) == 1
    assert field.validate(2) == 2
    
    # Test validation with mixed type choices
    field = Choice(choices=[("1", "String One"), (2, "Number Two")])
    assert field.validate("1") == "1"
    assert field.validate(2) == 2
    
    # Test validation with boolean choices
    field = Choice(choices=[(True, "Yes"), (False, "No")])
    assert field.validate(True) is True
    assert field.validate(False) is False
    
    # Test validation with tuple choices format
    field = Choice(choices=[("key1", "Display 1"), ("key2", "Display 2")])
    assert field.validate("key1") == "key1"
    assert field.validate("key2") == "key2"
    
    # Test validation with list choices format
    field = Choice(choices=[["key1", "Display 1"], ["key2", "Display 2"]])
    assert field.validate("key1") == "key1"
    assert field.validate("key2") == "key2"


# LLM-generated content at query #3
#--------------------------

```python
def test_Object_validate():
    # Test basic validation with simple properties
    schema = Object(properties={"name": String(), "age": Integer()})
    result = schema.validate({"name": "John", "age": 30})
    assert result == {"name": "John", "age": 30}

    # Test null handling
    schema = Object(properties={"name": String()}, allow_null=True)
    assert schema.validate(None) is None

    schema = Object(properties={"name": String()}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "null"

    # Test type validation
    schema = Object(properties={"name": String()})
    try:
        schema.validate("not an object")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "type"

    # Test required properties
    schema = Object(properties={"name": String()}, required=["name"])
    result = schema.validate({"name": "John"})
    assert result == {"name": "John"}

    try:
        schema.validate({"age": 30})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert any(msg.code == "required" for msg in e.messages)

    # Test default values
    schema = Object(properties={"name": String(default="Unknown")})
    result = schema.validate({})
    assert result == {"name": "Unknown"}

    # Test min_properties
    schema = Object(properties={"a": String(), "b": String()}, min_properties=1)
    result = schema.validate({"a": "test"})
    assert result == {"a": "test"}

    try:
        schema.validate({})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "empty"

    schema = Object(properties={"a": String(), "b": String(), "c": String()}, min_properties=2)
    try:
        schema.validate({"a": "test"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "min_properties"

    # Test max_properties
    schema = Object(properties={"a": String(), "b": String()}, max_properties=2)
    result = schema.validate({"a": "test", "b": "test2"})
    assert result == {"a": "test", "b": "test2"}

    try:
        schema.validate({"a": "test", "b": "test2", "c": "test3"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "max_properties"

    # Test pattern_properties
    schema = Object(pattern_properties={"^test_": String()})
    result = schema.validate({"test_key": "value"})
    assert result == {"test_key": "value"}

    # Test additional_properties=True
    schema = Object(properties={"name": String()}, additional_properties=True)
    result = schema.validate({"name": "John", "extra": "value"})
    assert result == {"name": "John", "extra": "value"}

    # Test additional_properties=False
    schema = Object(properties={"name": String()}, additional_properties=False)
    result = schema.validate({"name": "John"})
    assert result == {"name": "John"}

    try:
        schema.validate({"name": "John", "extra": "value"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert any(msg.code == "invalid_property" for msg in e.messages)

    # Test additional_properties as Field
    schema = Object(
        properties={"name": String()},
        additional_properties=Integer()
    )
    result = schema.validate({"name": "John", "extra": 42})
    assert result == {"name": "John", "extra": 42}

    try:
        schema.validate({"name": "John", "extra": "not a number"})
        assert False, "Should have raised ValidationError"
    except ValidationError:
        pass

    # Test property_names validation
    schema = Object(
        properties={"name": String()},
        property_names=String(pattern="^[a-z]+$")
    )
    result = schema.validate({"name": "John", "age": "value"})
    assert "age" in result

    try:
        schema.validate({"Name": "John"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert any(msg.code == "invalid_property" for msg in e.messages)

    # Test invalid key type
    schema = Object(properties={"name": String()})
    try:
        schema.validate({123: "value"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert any(msg.code == "invalid_key" for msg in e.messages)

    # Test nested validation errors
    schema = Object(properties={
        "person": Object(properties={
            "name": String(required=True),
            "age": Integer(minimum=0)
        })
    })
    try:
        schema.validate({"person": {"age": -5}})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        messages = e.messages()
        codes = [msg.code for msg in messages]
        assert "required" in codes or "minimum" in codes

    # Test complex scenario with multiple validations
    schema = Object(
        properties={
            "username": String(min_length=3, max_length=20),
            "email": String()
        },
        pattern_properties={"^meta_": String()},
        additional_properties=False,
        required=["username"],
        min_properties=1,
        max_properties=3
    )
    
    # Valid case
    result = schema.validate({
        "username": "johndoe",
        "meta_info": "some info"
    })
    assert "username" in result
    assert "meta_info" in result
    
    # Invalid - too many properties
    try:
        schema.validate({
            "username": "jd",
            "email": "test@test.com",
            "meta_1": "a",
            "meta_2": "b"
        })
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "max_properties"


# LLM-generated content at query #4
#--------------------------

```python
def test_Array_serialize():
    # Test 1: Serialize None when allow_null is True
    field = Array(items=String(), allow_null=True)
    assert field.serialize(None) is None

    # Test 2: Serialize None when allow_null is False (should still return None)
    field = Array(items=String(), allow_null=False)
    assert field.serialize(None) is None

    # Test 3: Serialize with items as a single Field
    field = Array(items=Integer())
    obj = [1, 2, 3]
    result = field.serialize(obj)
    assert result == [1, 2, 3]

    # Test 4: Serialize with items as a list of Fields
    field = Array(items=[Integer(), String(), Boolean()])
    obj = [42, "hello", True]
    result = field.serialize(obj)
    assert result == [42, "hello", True]

    # Test 5: Serialize with items=None (should return original object)
    field = Array(items=None)
    obj = [1, "test", {"key": "value"}]
    result = field.serialize(obj)
    assert result == obj

    # Test 6: Serialize with nested serialization
    field = Array(items=Object(properties={"id": Integer(), "name": String()}))
    obj = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
    result = field.serialize(obj)
    assert result == obj

    # Test 7: Serialize empty list
    field = Array(items=String())
    obj = []
    result = field.serialize(obj)
    assert result == []

    # Test 8: Serialize with items having custom serialize method
    decimal_field = Array(items=Decimal())
    obj = [decimal.Decimal("1.5"), decimal.Decimal("2.75")]
    result = decimal_field.serialize(obj)
    assert result == [1.5, 2.75]

    # Test 9: Serialize with mixed types when items=None
    field = Array(items=None)
    obj = [1, "string", True, None, {"key": "value"}]
    result = field.serialize(obj)
    assert result == obj

    # Test 10: Serialize with items list shorter than object
    field = Array(items=[Integer(), String()])
    obj = [1, "hello", "extra", "items"]
    result = field.serialize(obj)
    assert result == [1, "hello", "extra", "items"]


# LLM-generated content at query #5
#--------------------------

```python
def test_String():
    # Test basic initialization with default values
    field = String()
    assert field.allow_blank == False
    assert field.trim_whitespace == True
    assert field.max_length == None
    assert field.min_length == None
    assert field.pattern == None
    assert field.pattern_regex == None
    assert field.format == None
    assert field.coerce_types == True
    assert field.allow_null == False
    assert field.read_only == False
    assert field.title == ""
    assert field.description == ""

    # Test initialization with all parameters
    field = String(
        allow_blank=True,
        trim_whitespace=False,
        max_length=100,
        min_length=5,
        pattern=r"^\d+$",
        format="email",
        coerce_types=False,
        allow_null=True,
        read_only=True,
        title="Test Title",
        description="Test Description"
    )
    assert field.allow_blank == True
    assert field.trim_whitespace == False
    assert field.max_length == 100
    assert field.min_length == 5
    assert field.pattern == r"^\d+$"
    assert field.pattern_regex is not None
    assert field.format == "email"
    assert field.coerce_types == False
    assert field.allow_null == True
    assert field.read_only == True
    assert field.title == "Test Title"
    assert field.description == "Test Description"
    assert field.default == ""

    # Test pattern as compiled regex
    import re
    regex = re.compile(r"^[A-Z]+$")
    field = String(pattern=regex)
    assert field.pattern == r"^[A-Z]+$"
    assert field.pattern_regex == regex

    # Test with allow_blank=True should set default to empty string
    field = String(allow_blank=True)
    assert field.default == ""

    # Test with allow_blank=False should not set default
    field = String(allow_blank=False)
    assert not hasattr(field, 'default')

    # Test with allow_null=True should set default to None
    field = String(allow_null=True)
    assert field.default == None

    # Test with explicit default and allow_null
    field = String(default="test", allow_null=True)
    assert field.default == "test"

    # Test with explicit default and allow_blank
    field = String(default="custom", allow_blank=True)
    assert field.default == "custom"

    # Test type assertions
    try:
        String(max_length="invalid")
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass

    try:
        String(min_length="invalid")
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass

    try:
        String(pattern=123)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass

    try:
        String(format=123)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass


# LLM-generated content at query #6
#--------------------------

```python
def test_Union_validate():
    from decimal import Decimal
    from unittest.mock import Mock

    # Test 1: Null value with allow_null=True
    field1 = Mock(spec=Field, allow_null=True)
    field2 = Mock(spec=Field, allow_null=False)
    union = Union(any_of=[field1, field2])
    assert union.validate(None) is None

    # Test 2: Null value with allow_null=False
    field1 = Mock(spec=Field, allow_null=False)
    field2 = Mock(spec=Field, allow_null=False)
    union = Union(any_of=[field1, field2])
    try:
        union.validate(None)
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages()[0].code == "null"

    # Test 3: Valid value matched by first child
    field1 = Mock(spec=Field)
    field1.validate_or_error.return_value = ("valid", None)
    field2 = Mock(spec=Field)
    union = Union(any_of=[field1, field2])
    assert union.validate("test") == "valid"

    # Test 4: Valid value matched by second child
    field1 = Mock(spec=Field)
    field1.validate_or_error.return_value = (None, Mock())
    field2 = Mock(spec=Field)
    field2.validate_or_error.return_value = ("valid", None)
    union = Union(any_of=[field1, field2])
    assert union.validate("test") == "valid"

    # Test 5: No match - all children return type errors
    field1 = Mock(spec=Field)
    error1 = Mock()
    error1.messages.return_value = [Message(text="type error", code="type", index=None)]
    field1.validate_or_error.return_value = (None, error1)
    
    field2 = Mock(spec=Field)
    error2 = Mock()
    error2.messages.return_value = [Message(text="type error", code="type", index=None)]
    field2.validate_or_error.return_value = (None, error2)
    
    union = Union(any_of=[field1, field2])
    try:
        union.validate("test")
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages()[0].code == "union"

    # Test 6: No match - one child has non-type error with index
    field1 = Mock(spec=Field)
    error1 = Mock()
    error1.messages.return_value = [Message(text="min error", code="minimum", index=["field"])]
    field1.validate_or_error.return_value = (None, error1)
    
    field2 = Mock(spec=Field)
    error2 = Mock()
    error2.messages.return_value = [Message(text="type error", code="type", index=None)]
    field2.validate_or_error.return_value = (None, error2)
    
    union = Union(any_of=[field1, field2])
    try:
        union.validate("test")
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages()[0].code == "minimum"

    # Test 7: No match - multiple children have non-type errors
    field1 = Mock(spec=Field)
    error1 = Mock()
    error1.messages.return_value = [Message(text="min error", code="minimum", index=None)]
    field1.validate_or_error.return_value = (None, error1)
    
    field2 = Mock(spec=Field)
    error2 = Mock()
    error2.messages.return_value = [Message(text="max error", code="maximum", index=None)]
    field2.validate_or_error.return_value = (None, error2)
    
    union = Union(any_of=[field1, field2])
    try:
        union.validate("test")
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages()[0].code == "union"

    # Test 8: Child returns multiple error messages
    field1 = Mock(spec=Field)
    error1 = Mock()
    error1.messages.return_value = [
        Message(text="type error", code="type", index=None),
        Message(text="min error", code="minimum", index=None)
    ]
    field1.validate_or_error.return_value = (None, error1)
    
    field2 = Mock(spec=Field)
    error2 = Mock()
    error2.messages.return_value = [Message(text="type error", code="type", index=None)]
    field2.validate_or_error.return_value = (None, error2)
    
    union = Union(any_of=[field1, field2])
    try:
        union.validate("test")
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages()[0].code == "minimum"

    # Test 9: Child returns type error with index
    field1 = Mock(spec=Field)
    error1 = Mock()
    error1.messages.return_value = [Message(text="type error", code="type", index=["field"])]
    field1.validate_or_error.return_value = (None, error1)
    
    field2 = Mock(spec=Field)
    error2 = Mock()
    error2.messages.return_value = [Message(text="type error", code="type", index=None)]
    field2.validate_or_error.return_value = (None, error2)
    
    union = Union(any_of=[field1, field2])
    try:
        union.validate("test")
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages()[0].code == "type"


# LLM-generated content at query #7
#--------------------------

```python
def test_String_validate():
    # Test basic string validation
    field = String()
    assert field.validate("hello") == "hello"
    
    # Test null handling
    field = String(allow_null=True)
    assert field.validate(None) is None
    
    field = String(allow_null=False)
    try:
        field.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "null"
    
    # Test type validation
    try:
        field.validate(123)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "type"
    
    # Test blank handling
    field = String(allow_blank=False)
    try:
        field.validate("")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "blank"
    
    field = String(allow_blank=True)
    assert field.validate("") == ""
    
    # Test trim_whitespace
    field = String(trim_whitespace=True)
    assert field.validate("  hello  ") == "hello"
    
    field = String(trim_whitespace=False)
    assert field.validate("  hello  ") == "  hello  "
    
    # Test min_length
    field = String(min_length=3)
    assert field.validate("abc") == "abc"
    
    try:
        field.validate("ab")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "min_length"
    
    # Test max_length
    field = String(max_length=5)
    assert field.validate("abcde") == "abcde"
    
    try:
        field.validate("abcdef")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "max_length"
    
    # Test pattern matching
    field = String(pattern=r"^\d+$")
    assert field.validate("123") == "123"
    
    try:
        field.validate("abc")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "pattern"
    
    # Test format validation
    field = String(format="email")
    assert field.validate("test@example.com") == "test@example.com"
    
    try:
        field.validate("not-an-email")
        assert False, "Should have raised ValidationError"
    except ValidationError:
        pass
    
    # Test null character removal
    field = String()
    assert field.validate("hello\0world") == "helloworld"
    
    # Test coerce_types with allow_blank
    field = String(allow_blank=True, coerce_types=True)
    assert field.validate(None) == ""
    
    # Test coerce_types with allow_null
    field = String(allow_null=True, coerce_types=True)
    assert field.validate("") is None
    
    # Test native type handling for formats
    field = String(format="uuid")
    uuid_obj = formats.UUIDFormat().validate("12345678-1234-5678-1234-567812345678")
    assert field.validate(uuid_obj) == uuid_obj


# LLM-generated content at query #8
#--------------------------

```python
def test_Choice_validate():
    # Test basic validation with valid choices
    field = Choice(choices=[("a", "Option A"), ("b", "Option B")])
    assert field.validate("a") == "a"
    assert field.validate("b") == "b"
    
    # Test with allow_null=True
    field = Choice(choices=[("a", "A"), ("b", "B")], allow_null=True)
    assert field.validate(None) is None
    
    # Test with allow_null=False (default)
    field = Choice(choices=[("a", "A"), ("b", "B")])
    try:
        field.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "null"
    
    # Test invalid choice
    field = Choice(choices=[("a", "A"), ("b", "B")])
    try:
        field.validate("c")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "choice"
    
    # Test empty string with allow_null=False and coerce_types=True (default)
    field = Choice(choices=[("a", "A"), ("b", "B")])
    try:
        field.validate("")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "required"
    
    # Test empty string with allow_null=True and coerce_types=True
    field = Choice(choices=[("a", "A"), ("b", "B")], allow_null=True)
    assert field.validate("") is None
    
    # Test empty string with allow_null=False and coerce_types=False
    field = Choice(choices=[("a", "A"), ("b", "B")], coerce_types=False)
    try:
        field.validate("")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "choice"
    
    # Test with numeric choices
    field = Choice(choices=[(1, "One"), (2, "Two")])
    assert field.validate(1) == 1
    assert field.validate(2) == 2
    
    # Test with mixed type choices
    field = Choice(choices=[("1", "String One"), (2, "Number Two")])
    assert field.validate("1") == "1"
    assert field.validate(2) == 2
    
    # Test validation with Uniqueness wrapper
    field = Choice(choices=[("a", "A"), ("b", "B")])
    assert field.validate("a") == "a"
    
    # Test validate_or_error method
    field = Choice(choices=[("a", "A"), ("b", "B")])
    result = field.validate_or_error("a")
    assert result.value == "a"
    assert result.error is None
    
    result = field.validate_or_error("c")
    assert result.value is None
    assert result.error is not None
    assert result.error.code == "choice"


# LLM-generated content at query #9
#--------------------------

```python
def test_Boolean_validate():
    # Test basic boolean validation
    field = Boolean()
    assert field.validate(True) is True
    assert field.validate(False) is False
    
    # Test null handling
    field = Boolean(allow_null=True)
    assert field.validate(None) is None
    
    field = Boolean(allow_null=False)
    try:
        field.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "null"
    
    # Test type validation without coercion
    field = Boolean(coerce_types=False)
    try:
        field.validate("true")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "type"
    
    # Test string coercion
    field = Boolean(coerce_types=True)
    assert field.validate("true") is True
    assert field.validate("false") is False
    assert field.validate("on") is True
    assert field.validate("off") is False
    assert field.validate("1") is True
    assert field.validate("0") is False
    assert field.validate("") is False
    
    # Test numeric coercion
    assert field.validate(1) is True
    assert field.validate(0) is False
    
    # Test case-insensitive string coercion
    assert field.validate("TRUE") is True
    assert field.validate("FALSE") is False
    assert field.validate("On") is True
    assert field.validate("Off") is False
    
    # Test null coercion with strings
    field = Boolean(allow_null=True, coerce_types=True)
    assert field.validate("null") is None
    assert field.validate("none") is None
    assert field.validate("") is None
    
    # Test invalid string values
    field = Boolean(coerce_types=True)
    try:
        field.validate("invalid")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "type"
    
    # Test invalid type coercion
    try:
        field.validate([])
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "type"


# LLM-generated content at query #10
#--------------------------

```python
def test_String_validate():
    # Test basic string validation
    field = String()
    assert field.validate("hello") == "hello"
    
    # Test null handling
    field = String(allow_null=True)
    assert field.validate(None) is None
    
    field = String(allow_null=False)
    try:
        field.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "null"
    
    # Test blank handling
    field = String(allow_blank=True)
    assert field.validate("") == ""
    
    field = String(allow_blank=False)
    try:
        field.validate("")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "blank"
    
    # Test trim_whitespace
    field = String(trim_whitespace=True)
    assert field.validate("  hello  ") == "hello"
    
    field = String(trim_whitespace=False)
    assert field.validate("  hello  ") == "  hello  "
    
    # Test min_length
    field = String(min_length=3)
    assert field.validate("abc") == "abc"
    
    try:
        field.validate("ab")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "min_length"
    
    # Test max_length
    field = String(max_length=5)
    assert field.validate("abcde") == "abcde"
    
    try:
        field.validate("abcdef")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "max_length"
    
    # Test pattern
    field = String(pattern=r"^\d+$")
    assert field.validate("123") == "123"
    
    try:
        field.validate("abc")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "pattern"
    
    # Test format
    field = String(format="email")
    assert field.validate("test@example.com") == "test@example.com"
    
    try:
        field.validate("not-an-email")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "format"
    
    # Test null character removal
    field = String()
    assert field.validate("hello\0world") == "helloworld"
    
    # Test type validation
    field = String()
    try:
        field.validate(123)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "type"
    
    # Test coerce_types with allow_blank
    field = String(allow_blank=True, coerce_types=True)
    assert field.validate(None) == ""
    
    # Test coerce_types with allow_null
    field = String(allow_null=True, coerce_types=True)
    field_with_blank = String(allow_blank=False, allow_null=True, coerce_types=True)
    assert field_with_blank.validate("") is None
    
    # Test native type handling for formats
    field = String(format="uuid")
    uuid_obj = formats.UUIDFormat().validate("12345678-1234-5678-1234-567812345678")
    assert field.validate(uuid_obj) == uuid_obj


# LLM-generated content at query #11
#--------------------------

```python
def test_Const():
    # Test with non-None const value
    const_field = Const(const=42)
    assert const_field.const == 42
    assert const_field.allow_null is False
    
    # Test with None const value
    const_field_none = Const(const=None)
    assert const_field_none.const is None
    assert const_field_none.allow_null is False
    
    # Test with string const value
    const_field_str = Const(const="test")
    assert const_field_str.const == "test"
    
    # Test with boolean const value
    const_field_bool = Const(const=True)
    assert const_field_bool.const is True
    
    # Test with list const value
    const_field_list = Const(const=[1, 2, 3])
    assert const_field_list.const == [1, 2, 3]
    
    # Test with dict const value
    const_field_dict = Const(const={"key": "value"})
    assert const_field_dict.const == {"key": "value"}
    
    # Test that allow_null cannot be overridden
    const_field = Const(const=100)
    assert const_field.allow_null is False


# LLM-generated content at query #12
#--------------------------

```python
def test_Object_validate():
    # Test basic validation with simple properties
    schema = Object(properties={"name": String(), "age": Integer()})
    result = schema.validate({"name": "John", "age": 30})
    assert result == {"name": "John", "age": 30}

    # Test null handling
    schema = Object(properties={"name": String()}, allow_null=True)
    assert schema.validate(None) is None

    # Test null not allowed
    schema = Object(properties={"name": String()}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "null"

    # Test type validation
    schema = Object(properties={"name": String()})
    try:
        schema.validate("not an object")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "type"

    # Test required fields
    schema = Object(properties={"name": String()}, required=["name"])
    try:
        schema.validate({"age": 30})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert any(m.code == "required" for m in e.messages)

    # Test default values
    schema = Object(properties={"name": String(default="Unknown")})
    result = schema.validate({})
    assert result == {"name": "Unknown"}

    # Test min_properties
    schema = Object(properties={"a": String(), "b": String()}, min_properties=1)
    try:
        schema.validate({})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "empty"

    # Test max_properties
    schema = Object(properties={"a": String()}, max_properties=1)
    try:
        schema.validate({"a": "test", "b": "test2"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "max_properties"

    # Test pattern_properties
    schema = Object(pattern_properties={"^test_": String()})
    result = schema.validate({"test_1": "value1", "test_2": "value2"})
    assert result == {"test_1": "value1", "test_2": "value2"}

    # Test additional_properties = True
    schema = Object(properties={"name": String()}, additional_properties=True)
    result = schema.validate({"name": "John", "extra": "field"})
    assert result == {"name": "John", "extra": "field"}

    # Test additional_properties = False
    schema = Object(properties={"name": String()}, additional_properties=False)
    try:
        schema.validate({"name": "John", "extra": "field"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert any(m.code == "invalid_property" for m in e.messages)

    # Test additional_properties as Field
    schema = Object(
        properties={"name": String()},
        additional_properties=Integer()
    )
    result = schema.validate({"name": "John", "count": 5})
    assert result == {"name": "John", "count": 5}

    # Test property_names validation
    schema = Object(
        property_names=String(pattern="^[a-z]+$"),
        additional_properties=True
    )
    try:
        schema.validate({"valid": "test", "INVALID": "test"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert any(m.code == "invalid_property" for m in e.messages)

    # Test invalid key type
    schema = Object(additional_properties=True)
    try:
        schema.validate({123: "value"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert any(m.code == "invalid_key" for m in e.messages)

    # Test nested validation errors
    schema = Object(properties={
        "person": Object(properties={
            "name": String(required=True),
            "age": Integer(minimum=0)
        })
    })
    try:
        schema.validate({"person": {"age": -5}})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        messages = e.messages()
        assert any(m.code == "required" for m in messages)
        assert any(m.code == "minimum" for m in messages)

    # Test complex scenario with multiple validations
    schema = Object(
        properties={"id": Integer()},
        pattern_properties={"^attr_": String()},
        additional_properties=Boolean(),
        min_properties=2,
        max_properties=4
    )
    result = schema.validate({
        "id": 1,
        "attr_name": "test",
        "enabled": True
    })
    assert result == {"id": 1, "attr_name": "test", "enabled": True}


# LLM-generated content at query #13
#--------------------------

```python
def test_Array_validate():
    # Test basic validation
    field = Array()
    assert field.validate([1, 2, 3]) == [1, 2, 3]
    assert field.validate([]) == []
    
    # Test null handling
    field = Array(allow_null=True)
    assert field.validate(None) is None
    
    field = Array(allow_null=False)
    try:
        field.validate(None)
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages()[0].code == "null"
    
    # Test type validation
    field = Array()
    try:
        field.validate("not a list")
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages()[0].code == "type"
    
    # Test min_items
    field = Array(min_items=2)
    assert field.validate([1, 2]) == [1, 2]
    assert field.validate([1, 2, 3]) == [1, 2, 3]
    
    try:
        field.validate([1])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages()[0].code == "min_items"
    
    # Test min_items with empty array
    field = Array(min_items=1)
    try:
        field.validate([])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages()[0].code == "empty"
    
    # Test max_items
    field = Array(max_items=2)
    assert field.validate([1]) == [1]
    assert field.validate([1, 2]) == [1, 2]
    
    try:
        field.validate([1, 2, 3])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages()[0].code == "max_items"
    
    # Test exact_items
    field = Array(exact_items=3)
    assert field.validate([1, 2, 3]) == [1, 2, 3]
    
    try:
        field.validate([1, 2])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages()[0].code == "exact_items"
    
    try:
        field.validate([1, 2, 3, 4])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages()[0].code == "exact_items"
    
    # Test items validation with single validator
    item_field = Integer()
    field = Array(items=item_field)
    assert field.validate([1, 2, 3]) == [1, 2, 3]
    
    try:
        field.validate([1, "invalid", 3])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].index == [1]
    
    # Test items validation with list of validators
    item_fields = [Integer(), String()]
    field = Array(items=item_fields)
    assert field.validate([1, "test"]) == [1, "test"]
    
    try:
        field.validate(["invalid", "test"])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].index == [0]
    
    # Test additional_items=False
    item_fields = [Integer(), String()]
    field = Array(items=item_fields, additional_items=False)
    assert field.validate([1, "test"]) == [1, "test"]
    
    try:
        field.validate([1, "test", "extra"])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages()[0].code == "additional_items"
    
    # Test additional_items with Field validator
    item_fields = [Integer(), String()]
    additional_field = Boolean()
    field = Array(items=item_fields, additional_items=additional_field)
    assert field.validate([1, "test", True, False]) == [1, "test", True, False]
    
    try:
        field.validate([1, "test", "not boolean"])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].index == [2]
    
    # Test unique_items
    field = Array(unique_items=True)
    assert field.validate([1, 2, 3]) == [1, 2, 3]
    
    try:
        field.validate([1, 2, 1])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages()[0].code == "unique_items"
        assert e.messages()[0].key == 2
    
    # Test complex nested validation
    nested_field = Array(items=Integer())
    field = Array(items=nested_field)
    assert field.validate([[1, 2], [3, 4]]) == [[1, 2], [3, 4]]
    
    try:
        field.validate([[1, 2], [3, "invalid"]])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert len(e.messages()) == 1
        assert e.messages()[0].index == [1, 1]
    
    # Test multiple errors
    field = Array(items=Integer(), min_items=2, max_items=3)
    try:
        field.validate([1, "invalid", 3, 4, 5])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert len(e.messages()) == 3
    
    # Test with allow_null on items
    item_field = String(allow_null=True)
    field = Array(items=item_field)
    assert field.validate(["a", None, "c"]) == ["a", None, "c"]


# LLM-generated content at query #14
#--------------------------

```python
def test_Object_validate():
    # Test basic validation with simple properties
    schema = Object(properties={"name": String(), "age": Integer()})
    result = schema.validate({"name": "John", "age": 30})
    assert result == {"name": "John", "age": 30}

    # Test null handling
    schema = Object(properties={"name": String()}, allow_null=True)
    assert schema.validate(None) is None

    schema = Object(properties={"name": String()}, allow_null=False)
    try:
        schema.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "null"

    # Test type validation
    schema = Object(properties={"name": String()})
    try:
        schema.validate("not an object")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "type"

    # Test required properties
    schema = Object(properties={"name": String()}, required=["name"])
    try:
        schema.validate({"age": 30})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert any(msg.code == "required" for msg in e.messages)

    # Test default values
    schema = Object(properties={"name": String(default="Unknown")})
    result = schema.validate({})
    assert result == {"name": "Unknown"}

    # Test invalid key types
    schema = Object(properties={"name": String()})
    try:
        schema.validate({123: "value"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert any(msg.code == "invalid_key" for msg in e.messages)

    # Test property names validation
    property_names = String(pattern="^[a-z_]+$")
    schema = Object(properties={}, property_names=property_names)
    try:
        schema.validate({"InvalidKey": "value"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert any(msg.code == "invalid_property" for msg in e.messages)

    # Test min_properties
    schema = Object(properties={}, min_properties=2)
    try:
        schema.validate({"key1": "value1"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "min_properties"

    schema = Object(properties={}, min_properties=1)
    try:
        schema.validate({})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "empty"

    # Test max_properties
    schema = Object(properties={}, max_properties=2)
    try:
        schema.validate({"k1": "v1", "k2": "v2", "k3": "v3"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "max_properties"

    # Test pattern_properties
    schema = Object(pattern_properties={"^test_": String()})
    result = schema.validate({"test_key": "value", "other": "ignored"})
    assert result == {"test_key": "value"}

    # Test additional_properties=True
    schema = Object(properties={"name": String()}, additional_properties=True)
    result = schema.validate({"name": "John", "extra": "value"})
    assert result == {"name": "John", "extra": "value"}

    # Test additional_properties=False
    schema = Object(properties={"name": String()}, additional_properties=False)
    try:
        schema.validate({"name": "John", "extra": "value"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert any(msg.code == "invalid_property" for msg in e.messages)

    # Test additional_properties as Field
    schema = Object(
        properties={"name": String()},
        additional_properties=Integer()
    )
    result = schema.validate({"name": "John", "extra": 42})
    assert result == {"name": "John", "extra": 42}

    try:
        schema.validate({"name": "John", "extra": "not a number"})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert any("extra" in str(msg.index) for msg in e.messages)

    # Test nested validation errors
    schema = Object(properties={
        "name": String(min_length=2),
        "age": Integer(minimum=0)
    })
    try:
        schema.validate({"name": "J", "age": -5})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 2
        assert any("min_length" in msg.code for msg in messages)
        assert any("minimum" in msg.code for msg in messages)

    # Test complex nested structure
    address_schema = Object(properties={
        "street": String(),
        "city": String()
    })
    person_schema = Object(properties={
        "name": String(),
        "address": address_schema
    })
    result = person_schema.validate({
        "name": "John",
        "address": {"street": "123 Main", "city": "Anytown"}
    })
    assert result == {
        "name": "John",
        "address": {"street": "123 Main", "city": "Anytown"}
    }

    # Test with additional_properties=None
    schema = Object(properties={"name": String()}, additional_properties=None)
    result = schema.validate({"name": "John", "extra": "value"})
    assert result == {"name": "John"}


# LLM-generated content at query #15
#--------------------------

```python
def test_Number_validate():
    # Test basic validation with integer
    field = Number()
    assert field.validate(42) == 42
    assert field.validate(0) == 0
    assert field.validate(-10) == -10
    
    # Test basic validation with float
    assert field.validate(3.14) == 3.14
    assert field.validate(-2.5) == -2.5
    
    # Test null handling
    field_with_null = Number(allow_null=True)
    assert field_with_null.validate(None) is None
    
    field_without_null = Number(allow_null=False)
    try:
        field_without_null.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "null"
    
    # Test type coercion
    field_coerce = Number(coerce_types=True)
    assert field_coerce.validate("42") == 42
    assert field_coerce.validate("3.14") == 3.14
    
    field_no_coerce = Number(coerce_types=False)
    try:
        field_no_coerce.validate("42")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "type"
    
    # Test boolean rejection
    try:
        field.validate(True)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "type"
    
    try:
        field.validate(False)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "type"
    
    # Test minimum constraint
    field_min = Number(minimum=10)
    assert field_min.validate(10) == 10
    assert field_min.validate(15) == 15
    
    try:
        field_min.validate(5)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "minimum"
    
    # Test maximum constraint
    field_max = Number(maximum=100)
    assert field_max.validate(100) == 100
    assert field_max.validate(50) == 50
    
    try:
        field_max.validate(150)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "maximum"
    
    # Test exclusive minimum
    field_excl_min = Number(exclusive_minimum=10)
    assert field_excl_min.validate(11) == 11
    
    try:
        field_excl_min.validate(10)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "exclusive_minimum"
    
    # Test exclusive maximum
    field_excl_max = Number(exclusive_maximum=100)
    assert field_excl_max.validate(99) == 99
    
    try:
        field_excl_max.validate(100)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "exclusive_maximum"
    
    # Test multiple_of with integer
    field_multiple = Number(multiple_of=5)
    assert field_multiple.validate(10) == 10
    assert field_multiple.validate(15) == 15
    assert field_multiple.validate(-20) == -20
    
    try:
        field_multiple.validate(12)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "multiple_of"
    
    # Test multiple_of with float
    field_multiple_float = Number(multiple_of=0.5)
    assert field_multiple_float.validate(1.0) == 1.0
    assert field_multiple_float.validate(1.5) == 1.5
    
    try:
        field_multiple_float.validate(1.2)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "multiple_of"
    
    # Test precision
    field_precision = Number(precision="0.01")
    assert field_precision.validate(1.234) == 1.23
    assert field_precision.validate(1.235) == 1.24  # ROUND_HALF_UP
    
    # Test finite validation
    field_finite = Number()
    try:
        field_finite.validate(float('inf'))
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "finite"
    
    try:
        field_finite.validate(float('-inf'))
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "finite"
    
    try:
        field_finite.validate(float('nan'))
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "finite"
    
    # Test integer-specific validation
    integer_field = Number()
    integer_field.numeric_type = int
    assert integer_field.validate(42) == 42
    assert integer_field.validate(0) == 0
    
    try:
        integer_field.validate(3.14)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "integer"
    
    # Test decimal.Decimal handling
    field_decimal = Number()
    assert field_decimal.validate(decimal.Decimal("10.5")) == decimal.Decimal("10.5")
    
    # Test empty string with allow_null and coerce_types
    field_coerce_null = Number(allow_null=True, coerce_types=True)
    assert field_coerce_null.validate("") is None
    
    # Test combined constraints
    field_combo = Number(minimum=0, maximum=100, multiple_of=10)
    assert field_combo.validate(0) == 0
    assert field_combo.validate(50) == 50
    assert field_combo.validate(100) == 100
    
    try:
        field_combo.validate(55)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "multiple_of"


# LLM-generated content at query #16
#--------------------------

```python
def test_String():
    # Test basic initialization with default values
    field = String()
    assert field.allow_blank == False
    assert field.trim_whitespace == True
    assert field.max_length == None
    assert field.min_length == None
    assert field.pattern == None
    assert field.pattern_regex == None
    assert field.format == None
    assert field.coerce_types == True
    assert field.allow_null == False
    assert field.read_only == False
    assert field.title == ""
    assert field.description == ""

    # Test initialization with all parameters
    field = String(
        allow_blank=True,
        trim_whitespace=False,
        max_length=100,
        min_length=5,
        pattern=r"^\d+$",
        format="email",
        coerce_types=False,
        allow_null=True,
        read_only=True,
        title="Test Title",
        description="Test Description"
    )
    assert field.allow_blank == True
    assert field.trim_whitespace == False
    assert field.max_length == 100
    assert field.min_length == 5
    assert field.pattern == r"^\d+$"
    assert field.pattern_regex is not None
    assert field.format == "email"
    assert field.coerce_types == False
    assert field.allow_null == True
    assert field.read_only == True
    assert field.title == "Test Title"
    assert field.description == "Test Description"
    assert field.default == ""

    # Test pattern as compiled regex
    import re
    regex = re.compile(r"^[A-Z]+$")
    field = String(pattern=regex)
    assert field.pattern == r"^[A-Z]+$"
    assert field.pattern_regex == regex

    # Test with allow_blank=True should set default to empty string
    field = String(allow_blank=True)
    assert field.has_default() == True
    assert field.get_default_value() == ""

    # Test with allow_blank=False should not set default
    field = String(allow_blank=False)
    assert not hasattr(field, "default")

    # Test with allow_null=True and no default should set default to None
    field = String(allow_null=True)
    assert field.has_default() == True
    assert field.get_default_value() == None

    # Test with allow_null=True and explicit default
    field = String(allow_null=True, default="custom")
    assert field.get_default_value() == "custom"

    # Test with allow_blank=True and explicit default
    field = String(allow_blank=True, default="explicit")
    assert field.get_default_value() == "explicit"

    # Test type assertions
    try:
        String(max_length="invalid")
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass

    try:
        String(min_length="invalid")
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass

    try:
        String(pattern=123)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass

    try:
        String(format=123)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass


# LLM-generated content at query #17
#--------------------------

```python
def test_Number_validate():
    # Test basic validation with integer
    field = Number()
    assert field.validate(42) == 42
    assert field.validate(0) == 0
    assert field.validate(-10) == -10
    
    # Test basic validation with float
    assert field.validate(3.14) == 3.14
    assert field.validate(-2.5) == -2.5
    
    # Test null handling
    field_with_null = Number(allow_null=True)
    assert field_with_null.validate(None) is None
    
    # Test null without allow_null
    field_no_null = Number()
    try:
        field_no_null.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "null"
    
    # Test string coercion with coerce_types=True (default)
    assert field.validate("42") == 42
    assert field.validate("3.14") == 3.14
    
    # Test string coercion with coerce_types=False
    field_no_coerce = Number(coerce_types=False)
    try:
        field_no_coerce.validate("42")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "type"
    
    # Test boolean rejection
    try:
        field.validate(True)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "type"
    
    try:
        field.validate(False)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "type"
    
    # Test minimum constraint
    field_min = Number(minimum=10)
    assert field_min.validate(10) == 10
    assert field_min.validate(15) == 15
    
    try:
        field_min.validate(5)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "minimum"
    
    # Test maximum constraint
    field_max = Number(maximum=100)
    assert field_max.validate(100) == 100
    assert field_max.validate(50) == 50
    
    try:
        field_max.validate(150)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "maximum"
    
    # Test exclusive minimum
    field_excl_min = Number(exclusive_minimum=10)
    assert field_excl_min.validate(11) == 11
    assert field_excl_min.validate(10.1) == 10.1
    
    try:
        field_excl_min.validate(10)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "exclusive_minimum"
    
    # Test exclusive maximum
    field_excl_max = Number(exclusive_maximum=100)
    assert field_excl_max.validate(99) == 99
    assert field_excl_max.validate(99.9) == 99.9
    
    try:
        field_excl_max.validate(100)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "exclusive_maximum"
    
    # Test multiple_of with integer
    field_multiple_int = Number(multiple_of=5)
    assert field_multiple_int.validate(10) == 10
    assert field_multiple_int.validate(0) == 0
    assert field_multiple_int.validate(-15) == -15
    
    try:
        field_multiple_int.validate(7)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "multiple_of"
    
    # Test multiple_of with float
    field_multiple_float = Number(multiple_of=0.5)
    assert field_multiple_float.validate(1.0) == 1.0
    assert field_multiple_float.validate(2.5) == 2.5
    
    try:
        field_multiple_float.validate(1.2)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "multiple_of"
    
    # Test precision
    field_precision = Number(precision="0.01")
    assert field_precision.validate(1.23) == 1.23
    assert field_precision.validate(1.234) == 1.23  # Should round
    
    # Test non-finite values
    try:
        field.validate(float('inf'))
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "finite"
    
    try:
        field.validate(float('-inf'))
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "finite"
    
    try:
        field.validate(float('nan'))
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "finite"
    
    # Test integer-specific validation
    integer_field = Number()
    integer_field.numeric_type = int
    
    assert integer_field.validate(42) == 42
    assert integer_field.validate(0) == 0
    
    try:
        integer_field.validate(3.14)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "integer"
    
    # Test empty string with allow_null and coerce_types
    field_null_coerce = Number(allow_null=True, coerce_types=True)
    assert field_null_coerce.validate("") is None
    
    # Test invalid string input
    try:
        field.validate("not a number")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "type"
    
    # Test decimal input
    field.validate(decimal.Decimal("10.5"))
    
    # Test combined constraints
    field_combined = Number(minimum=0, maximum=100, multiple_of=10)
    assert field_combined.validate(0) == 0
    assert field_combined.validate(50) == 50
    assert field_combined.validate(100) == 100
    
    try:
        field_combined.validate(45)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "multiple_of"


# LLM-generated content at query #18
#--------------------------

```python
def test_Boolean_validate():
    # Test basic boolean validation
    field = Boolean()
    assert field.validate(True) is True
    assert field.validate(False) is False
    
    # Test null handling
    field = Boolean(allow_null=True)
    assert field.validate(None) is None
    
    field = Boolean(allow_null=False)
    try:
        field.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "null"
    
    # Test type coercion with coerce_types=True (default)
    field = Boolean()
    assert field.validate("true") is True
    assert field.validate("false") is False
    assert field.validate("on") is True
    assert field.validate("off") is False
    assert field.validate("1") is True
    assert field.validate("0") is False
    assert field.validate("") is False
    assert field.validate(1) is True
    assert field.validate(0) is False
    
    # Test type coercion with coerce_types=False
    field = Boolean(coerce_types=False)
    assert field.validate(True) is True
    assert field.validate(False) is False
    try:
        field.validate("true")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "type"
    
    # Test null coercion with strings
    field = Boolean(allow_null=True)
    assert field.validate("null") is None
    assert field.validate("none") is None
    assert field.validate("") is None
    
    # Test invalid values with coercion
    field = Boolean()
    try:
        field.validate("invalid")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "type"
    
    try:
        field.validate(2)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "type"
    
    # Test case sensitivity in string coercion
    field = Boolean()
    assert field.validate("TRUE") is True
    assert field.validate("FALSE") is False
    assert field.validate("On") is True
    assert field.validate("Off") is False
    
    # Test that non-coercible types raise ValidationError
    field = Boolean()
    try:
        field.validate([])
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "type"
    
    try:
        field.validate({})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "type"


# LLM-generated content at query #19
#--------------------------

```python
def test_Union_validate():
    from decimal import Decimal
    from typing import List
    import pytest
    
    # Test 1: Union with null value when allow_null is True
    field1 = Integer(allow_null=True)
    field2 = String(allow_null=True)
    union = Union(any_of=[field1, field2])
    assert union.validate(None) is None
    
    # Test 2: Union with null value when allow_null is False
    field1 = Integer(allow_null=False)
    field2 = String(allow_null=False)
    union = Union(any_of=[field1, field2])
    with pytest.raises(ValidationError) as exc_info:
        union.validate(None)
    assert exc_info.value.messages()[0].code == "null"
    
    # Test 3: Union with valid value matching first field
    field1 = Integer()
    field2 = String()
    union = Union(any_of=[field1, field2])
    assert union.validate(42) == 42
    
    # Test 4: Union with valid value matching second field
    field1 = Integer()
    field2 = String()
    union = Union(any_of=[field1, field2])
    assert union.validate("hello") == "hello"
    
    # Test 5: Union with no matching field
    field1 = Integer()
    field2 = String()
    union = Union(any_of=[field1, field2])
    with pytest.raises(ValidationError) as exc_info:
        union.validate(True)
    assert exc_info.value.messages()[0].code == "union"
    
    # Test 6: Union with child field validation error (not type error)
    field1 = Integer(minimum=10)
    field2 = String()
    union = Union(any_of=[field1, field2])
    with pytest.raises(ValidationError) as exc_info:
        union.validate(5)
    assert exc_info.value.messages()[0].code == "minimum"
    
    # Test 7: Union with multiple candidate errors (should raise union error)
    field1 = Integer(minimum=10)
    field2 = String(min_length=5)
    union = Union(any_of=[field1, field2])
    with pytest.raises(ValidationError) as exc_info:
        union.validate(3)
    assert exc_info.value.messages()[0].code == "union"
    
    # Test 8: Union with nested fields
    field1 = Array(items=Integer())
    field2 = Object(properties={"value": Integer()})
    union = Union(any_of=[field1, field2])
    assert union.validate([1, 2, 3]) == [1, 2, 3]
    assert union.validate({"value": 42}) == {"value": 42}
    
    # Test 9: Union with Decimal field
    field1 = Decimal()
    field2 = Float()
    union = Union(any_of=[field1, field2])
    assert union.validate(Decimal("3.14")) == Decimal("3.14")
    assert union.validate(3.14) == 3.14
    
    # Test 10: Union with Boolean field
    field1 = Boolean()
    field2 = Integer()
    union = Union(any_of=[field1, field2])
    assert union.validate(True) is True
    assert union.validate(False) is False
    assert union.validate(42) == 42
    
    # Test 11: Union with one field allowing null
    field1 = Integer(allow_null=True)
    field2 = String(allow_null=False)
    union = Union(any_of=[field1, field2])
    assert union.allow_null is True
    assert union.validate(None) is None
    
    # Test 12: Union with type coercion
    field1 = Integer(coerce_types=True)
    field2 = String(coerce_types=True)
    union = Union(any_of=[field1, field2])
    assert union.validate("42") == 42
    assert union.validate(42) == "42"
    
    # Test 13: Union with Choice field
    field1 = Choice(choices=[("a", "A"), ("b", "B")])
    field2 = Integer()
    union = Union(any_of=[field1, field2])
    assert union.validate("a") == "a"
    assert union.validate(100) == 100
    
    # Test 14: Union with Date field
    field1 = Date()
    field2 = DateTime()
    union = Union(any_of=[field1, field2])
    # Assuming String validation for dates
    assert isinstance(union.validate("2023-01-01"), str)
    
    # Test 15: Complex union with multiple validation paths
    field1 = Object(properties={"type": String(), "value": Integer()})
    field2 = Array(items=String())
    field3 = Boolean()
    union = Union(any_of=[field1, field2, field3])
    
    assert union.validate({"type": "test", "value": 42}) == {"type": "test", "value": 42}
    assert union.validate(["a", "b", "c"]) == ["a", "b", "c"]
    assert union.validate(True) is True
    
    with pytest.raises(ValidationError) as exc_info:
        union.validate({"type": "test"})  # Missing required field
    assert exc_info.value.messages()[0].code == "required"


# LLM-generated content at query #20
#--------------------------

```python
def test_Const():
    # Test with integer constant
    const_int = Const(42)
    assert const_int.const == 42
    assert const_int.allow_null is False
    
    # Test with string constant
    const_str = Const("hello")
    assert const_str.const == "hello"
    assert const_str.allow_null is False
    
    # Test with None constant
    const_none = Const(None)
    assert const_none.const is None
    assert const_none.allow_null is False
    
    # Test with boolean constant
    const_bool = Const(True)
    assert const_bool.const is True
    assert const_none.allow_null is False
    
    # Test with list constant
    const_list = Const([1, 2, 3])
    assert const_list.const == [1, 2, 3]
    assert const_none.allow_null is False
    
    # Test with dict constant
    const_dict = Const({"key": "value"})
    assert const_dict.const == {"key": "value"}
    assert const_none.allow_null is False
    
    # Test that allow_null cannot be overridden
    try:
        Const(42, allow_null=True)
        assert False, "Should have raised an assertion error"
    except AssertionError:
        pass


# LLM-generated content at query #21
#--------------------------

```python
def test_Array_validate():
    # Test basic validation with no constraints
    field = Array()
    assert field.validate([1, 2, 3]) == [1, 2, 3]
    assert field.validate([]) == []
    
    # Test null handling
    field = Array(allow_null=True)
    assert field.validate(None) is None
    
    field = Array(allow_null=False)
    try:
        field.validate(None)
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages[0].code == "null"
    
    # Test type validation
    field = Array()
    try:
        field.validate("not an array")
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages[0].code == "type"
    
    # Test min_items constraint
    field = Array(min_items=2)
    assert field.validate([1, 2]) == [1, 2]
    assert field.validate([1, 2, 3]) == [1, 2, 3]
    
    try:
        field.validate([1])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages[0].code == "min_items"
    
    # Test min_items=1 gives "empty" error
    field = Array(min_items=1)
    try:
        field.validate([])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages[0].code == "empty"
    
    # Test max_items constraint
    field = Array(max_items=3)
    assert field.validate([1, 2, 3]) == [1, 2, 3]
    assert field.validate([1, 2]) == [1, 2]
    
    try:
        field.validate([1, 2, 3, 4])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages[0].code == "max_items"
    
    # Test exact_items constraint
    field = Array(exact_items=3)
    assert field.validate([1, 2, 3]) == [1, 2, 3]
    
    try:
        field.validate([1, 2])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages[0].code == "exact_items"
    
    try:
        field.validate([1, 2, 3, 4])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages[0].code == "exact_items"
    
    # Test unique_items constraint
    field = Array(unique_items=True)
    assert field.validate([1, 2, 3]) == [1, 2, 3]
    
    try:
        field.validate([1, 2, 1])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages[0].code == "unique_items"
    
    # Test with item validator
    field = Array(items=Integer())
    assert field.validate([1, 2, 3]) == [1, 2, 3]
    
    try:
        field.validate([1, "invalid", 3])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages[0].code == "type"
    
    # Test with list of item validators
    field = Array(items=[Integer(), String()])
    assert field.validate([1, "hello"]) == [1, "hello"]
    
    try:
        field.validate(["invalid", "hello"])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages[0].code == "type"
    
    # Test additional_items=False with list of validators
    field = Array(items=[Integer(), String()], additional_items=False)
    assert field.validate([1, "hello"]) == [1, "hello"]
    
    try:
        field.validate([1, "hello", "extra"])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages[0].code == "additional_items"
    
    # Test additional_items=True with list of validators
    field = Array(items=[Integer(), String()], additional_items=True)
    assert field.validate([1, "hello", "extra", 4]) == [1, "hello", "extra", 4]
    
    # Test additional_items as Field validator
    field = Array(items=[Integer(), String()], additional_items=Boolean())
    assert field.validate([1, "hello", True, False]) == [1, "hello", True, False]
    
    try:
        field.validate([1, "hello", "not boolean"])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages[0].code == "type"
    
    # Test error message aggregation
    field = Array(items=Integer(), unique_items=True)
    try:
        field.validate([1, "invalid", 1])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        codes = [msg.code for msg in e.messages]
        assert "type" in codes
        assert "unique_items" in codes
    
    # Test with nested validators
    field = Array(items=Array(items=Integer()))
    assert field.validate([[1, 2], [3, 4]]) == [[1, 2], [3, 4]]
    
    try:
        field.validate([[1, 2], ["invalid", 4]])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages[0].code == "type"


# LLM-generated content at query #22
#--------------------------

```python
def test_Array():
    # Test basic initialization with no arguments
    field = Array()
    assert field.items is None
    assert field.additional_items is False
    assert field.min_items is None
    assert field.max_items is None
    assert field.unique_items is False
    assert field.allow_null is False

    # Test initialization with items as a single Field
    string_field = String()
    field = Array(items=string_field)
    assert field.items == string_field
    assert field.additional_items is False
    assert field.min_items is None
    assert field.max_items is None

    # Test initialization with items as a list of Fields
    fields = [String(), Integer()]
    field = Array(items=fields)
    assert field.items == fields
    assert field.additional_items is False
    assert field.min_items == 2
    assert field.max_items == 2

    # Test initialization with additional_items as a Field
    additional_field = Boolean()
    field = Array(items=fields, additional_items=additional_field)
    assert field.items == fields
    assert field.additional_items == additional_field
    assert field.min_items == 2
    assert field.max_items is None

    # Test initialization with additional_items as True
    field = Array(items=fields, additional_items=True)
    assert field.items == fields
    assert field.additional_items is True
    assert field.min_items == 2
    assert field.max_items is None

    # Test initialization with min_items and max_items
    field = Array(min_items=1, max_items=10)
    assert field.min_items == 1
    assert field.max_items == 10
    assert field.items is None

    # Test initialization with exact_items
    field = Array(exact_items=5)
    assert field.min_items == 5
    assert field.max_items == 5

    # Test initialization with unique_items
    field = Array(unique_items=True)
    assert field.unique_items is True

    # Test initialization with allow_null
    field = Array(allow_null=True)
    assert field.allow_null is True

    # Test that exact_items overrides min_items and max_items
    field = Array(exact_items=3, min_items=1, max_items=10)
    assert field.min_items == 3
    assert field.max_items == 3

    # Test that when items is a list and additional_items is False, max_items is set to len(items)
    fields = [String(), Integer(), Boolean()]
    field = Array(items=fields)
    assert field.max_items == 3

    # Test that when items is a list and additional_items is a Field, max_items is not set
    additional_field = String()
    field = Array(items=fields, additional_items=additional_field)
    assert field.max_items is None

    # Test that when items is a list and min_items is provided, it's not overridden
    field = Array(items=fields, min_items=1)
    assert field.min_items == 1

    # Test that when items is a list and max_items is provided, it's not overridden
    field = Array(items=fields, max_items=5)
    assert field.max_items == 5

    # Test that when items is a single Field, min_items and max_items are not automatically set
    single_field = String()
    field = Array(items=single_field)
    assert field.min_items is None
    assert field.max_items is None


# LLM-generated content at query #23
#--------------------------

```python
def test_Array_validate():
    # Test basic validation with allow_null
    field = Array(allow_null=True)
    assert field.validate(None) is None
    
    field = Array(allow_null=False)
    try:
        field.validate(None)
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages()[0].code == "null"
    
    # Test type validation
    field = Array()
    try:
        field.validate("not a list")
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages()[0].code == "type"
    
    # Test min_items validation
    field = Array(min_items=2)
    try:
        field.validate([1])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages()[0].code == "min_items"
    
    assert field.validate([1, 2]) == [1, 2]
    assert field.validate([1, 2, 3]) == [1, 2, 3]
    
    # Test max_items validation
    field = Array(max_items=2)
    try:
        field.validate([1, 2, 3])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages()[0].code == "max_items"
    
    assert field.validate([1]) == [1]
    assert field.validate([1, 2]) == [1, 2]
    
    # Test exact_items validation
    field = Array(exact_items=2)
    try:
        field.validate([1])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages()[0].code == "exact_items"
    
    try:
        field.validate([1, 2, 3])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages()[0].code == "exact_items"
    
    assert field.validate([1, 2]) == [1, 2]
    
    # Test empty array with min_items=1
    field = Array(min_items=1)
    try:
        field.validate([])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages()[0].code == "empty"
    
    # Test items validation with single validator
    field = Array(items=Integer())
    assert field.validate([1, 2, 3]) == [1, 2, 3]
    
    try:
        field.validate([1, "invalid", 3])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages()[0].code == "type"
        assert e.messages()[0].index == [1]
    
    # Test items validation with list of validators
    field = Array(items=[Integer(), String()])
    assert field.validate([1, "test"]) == [1, "test"]
    
    try:
        field.validate(["invalid", "test"])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages()[0].code == "type"
        assert e.messages()[0].index == [0]
    
    # Test additional_items=False
    field = Array(items=[Integer(), String()], additional_items=False)
    try:
        field.validate([1, "test", "extra"])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages()[0].code == "additional_items"
    
    # Test additional_items=True
    field = Array(items=[Integer(), String()], additional_items=True)
    assert field.validate([1, "test", "extra", 4]) == [1, "test", "extra", 4]
    
    # Test additional_items with Field validator
    field = Array(items=[Integer()], additional_items=String())
    assert field.validate([1, "test", "another"]) == [1, "test", "another"]
    
    try:
        field.validate([1, 2, 3])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages()[0].code == "type"
        assert e.messages()[0].index == [1]
    
    # Test unique_items validation
    field = Array(unique_items=True)
    assert field.validate([1, 2, 3]) == [1, 2, 3]
    
    try:
        field.validate([1, 2, 1])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages()[0].code == "unique_items"
        assert e.messages()[0].key == 2
    
    # Test complex nested validation
    field = Array(items=Array(items=Integer()))
    assert field.validate([[1, 2], [3, 4]]) == [[1, 2], [3, 4]]
    
    try:
        field.validate([[1, 2], ["invalid", 4]])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages()[0].code == "type"
        assert e.messages()[0].index == [1, 0]
    
    # Test with no items validator
    field = Array()
    assert field.validate([1, "test", True, None]) == [1, "test", True, None]
    
    # Test error accumulation
    field = Array(items=Integer(), min_items=2)
    try:
        field.validate(["invalid"])
        assert False, "Should have raised validation error"
    except ValidationError as e:
        messages = e.messages()
        assert len(messages) == 2
        assert messages[0].code == "type"
        assert messages[0].index == [0]
        assert messages[1].code == "min_items"


# LLM-generated content at query #24
#--------------------------

```python
def test_Choice_validate():
    # Test basic validation with string choices
    field = Choice(choices=["a", "b", "c"])
    assert field.validate("a") == "a"
    assert field.validate("b") == "b"
    assert field.validate("c") == "c"
    
    # Test validation with tuple choices
    field = Choice(choices=[("a", "Option A"), ("b", "Option B")])
    assert field.validate("a") == "a"
    assert field.validate("b") == "b"
    
    # Test null handling with allow_null=False (default)
    field = Choice(choices=["a", "b"])
    try:
        field.validate(None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "null"
    
    # Test null handling with allow_null=True
    field = Choice(choices=["a", "b"], allow_null=True)
    assert field.validate(None) is None
    
    # Test invalid choice
    field = Choice(choices=["a", "b"])
    try:
        field.validate("c")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "choice"
    
    # Test empty string with allow_null=False
    field = Choice(choices=["a", "b"])
    try:
        field.validate("")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "required"
    
    # Test empty string with allow_null=True and coerce_types=True (default)
    field = Choice(choices=["a", "b"], allow_null=True)
    assert field.validate("") is None
    
    # Test empty string with allow_null=True and coerce_types=False
    field = Choice(choices=["a", "b"], allow_null=True, coerce_types=False)
    try:
        field.validate("")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "choice"
    
    # Test Uniqueness ensures no duplicate keys
    field = Choice(choices=[("a", "A1"), ("a", "A2"), ("b", "B")])
    assert field.validate("a") == "a"
    assert field.validate("b") == "b"
    try:
        field.validate("c")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "choice"


# LLM-generated content at query #25
#--------------------------

```python
def test_Union_validate():
    from decimal import Decimal
    from unittest.mock import Mock

    # Test 1: Null value when allow_null is True
    field1 = Mock(spec=Field)
    field1.allow_null = True
    field1.validate_or_error = Mock(return_value=(None, None))
    union = Union(any_of=[field1])
    assert union.validate(None) is None

    # Test 2: Null value when allow_null is False
    field2 = Mock(spec=Field)
    field2.allow_null = False
    union = Union(any_of=[field2])
    try:
        union.validate(None)
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages()[0].code == "null"

    # Test 3: Valid value matching first child
    field3 = Mock(spec=Field)
    field3.validate_or_error = Mock(return_value=("valid", None))
    union = Union(any_of=[field3])
    assert union.validate("test") == "valid"

    # Test 4: Valid value matching second child after first fails
    field4 = Mock(spec=Field)
    field4.validate_or_error = Mock(return_value=(None, ValidationError(messages=[Message(text="type", code="type")])))
    field5 = Mock(spec=Field)
    field5.validate_or_error = Mock(return_value=("valid", None))
    union = Union(any_of=[field4, field5])
    assert union.validate("test") == "valid"

    # Test 5: No matching child - all return type errors
    field6 = Mock(spec=Field)
    field6.validate_or_error = Mock(return_value=(None, ValidationError(messages=[Message(text="type", code="type")])))
    field7 = Mock(spec=Field)
    field7.validate_or_error = Mock(return_value=(None, ValidationError(messages=[Message(text="type", code="type")])))
    union = Union(any_of=[field6, field7])
    try:
        union.validate("test")
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages()[0].code == "union"

    # Test 6: Single candidate error with non-type error
    field8 = Mock(spec=Field)
    field8.validate_or_error = Mock(return_value=(None, ValidationError(messages=[Message(text="minimum", code="minimum")])))
    field9 = Mock(spec=Field)
    field9.validate_or_error = Mock(return_value=(None, ValidationError(messages=[Message(text="type", code="type")])))
    union = Union(any_of=[field8, field9])
    try:
        union.validate(5)
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages()[0].code == "minimum"

    # Test 7: Multiple candidate errors
    field10 = Mock(spec=Field)
    field10.validate_or_error = Mock(return_value=(None, ValidationError(messages=[Message(text="minimum", code="minimum")])))
    field11 = Mock(spec=Field)
    field11.validate_or_error = Mock(return_value=(None, ValidationError(messages=[Message(text="maximum", code="maximum")])))
    union = Union(any_of=[field10, field11])
    try:
        union.validate(5)
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages()[0].code == "union"

    # Test 8: Child with index in error message
    field12 = Mock(spec=Field)
    field12.validate_or_error = Mock(return_value=(None, ValidationError(messages=[Message(text="type", code="type", index=["key"])])))
    union = Union(any_of=[field12])
    try:
        union.validate("test")
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages()[0].code == "type"

    # Test 9: Child with multiple error messages
    field13 = Mock(spec=Field)
    field13.validate_or_error = Mock(return_value=(None, ValidationError(messages=[Message(text="type", code="type"), Message(text="minimum", code="minimum")])))
    union = Union(any_of=[field13])
    try:
        union.validate(5)
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert e.messages()[0].code == "type"

    # Test 10: Union inherits allow_null from children
    field14 = Mock(spec=Field)
    field14.allow_null = True
    field14.validate_or_error = Mock(return_value=(None, None))
    field15 = Mock(spec=Field)
    field15.allow_null = False
    union = Union(any_of=[field14, field15])
    assert union.allow_null is True
    assert union.validate(None) is None


# LLM-generated content at query #26
#--------------------------

```python
def test_Union():
    # Test basic initialization
    field1 = Integer()
    field2 = String()
    union = Union(any_of=[field1, field2])
    assert union.any_of == [field1, field2]
    assert union.allow_null is False

    # Test with null-allowing child
    field3 = Integer(allow_null=True)
    union2 = Union(any_of=[field3, field2])
    assert union2.allow_null is True

    # Test validation with matching type
    result = union.validate(123)
    assert result == 123

    # Test validation with other matching type
    result = union.validate("test")
    assert result == "test"

    # Test validation with null when not allowed
    try:
        union.validate(None)
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert any(msg.code == "null" for msg in e.messages())

    # Test validation with null when allowed
    result = union2.validate(None)
    assert result is None

    # Test validation with non-matching type
    try:
        union.validate(True)
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert any(msg.code == "union" for msg in e.messages())

    # Test validation with nested errors
    field4 = Integer(minimum=10)
    field5 = String(max_length=3)
    union3 = Union(any_of=[field4, field5])
    
    # Should trigger minimum error from field4
    try:
        union3.validate(5)
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert any(msg.code == "minimum" for msg in e.messages())

    # Should trigger max_length error from field5
    try:
        union3.validate("toolong")
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert any(msg.code == "max_length" for msg in e.messages())

    # Test with empty any_of list
    union4 = Union(any_of=[])
    try:
        union4.validate("anything")
        assert False, "Should have raised validation error"
    except ValidationError as e:
        assert any(msg.code == "union" for msg in e.messages())


