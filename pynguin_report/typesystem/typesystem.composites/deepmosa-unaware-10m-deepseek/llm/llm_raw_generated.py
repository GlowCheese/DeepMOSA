####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_Not():
    from typesystem.fields import String, Integer
    
    # Test basic initialization
    string_field = String()
    not_field = Not(string_field)
    assert not_field.negated == string_field
    assert not_field.errors == {"negated": "Must not match."}
    
    # Test that allow_null cannot be passed
    try:
        Not(string_field, allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test validation with non-matching value
    integer_field = Integer()
    not_integer = Not(integer_field)
    result = not_integer.validate("hello")
    assert result == "hello"
    
    # Test validation with matching value (should raise error)
    try:
        not_integer.validate(42)
        assert False, "Should have raised ValidationError"
    except Exception as e:
        assert "negated" in str(e)
    
    # Test with nested field
    from typesystem.fields import Boolean
    bool_field = Boolean()
    not_bool = Not(bool_field)
    result = not_bool.validate("not a boolean")
    assert result == "not a boolean"
    
    try:
        not_bool.validate(True)
        assert False, "Should have raised ValidationError"
    except Exception as e:
        assert "negated" in str(e)


# LLM-generated content at query #2
#--------------------------

```python
def test_IfThenElse_validate():
    from typesystem.fields import Integer, String, Boolean
    
    # Test basic if-then-else behavior
    int_field = Integer()
    string_field = String()
    bool_field = Boolean()
    
    # Test 1: If condition passes, then clause executes
    field1 = IfThenElse(
        if_clause=Integer(),
        then_clause=String(),
        else_clause=Boolean()
    )
    # Integer value should trigger then clause (String validation)
    with pytest.raises(Exception) as exc_info:
        field1.validate(123)
    assert "Must be a string." in str(exc_info.value)
    
    # Test 2: If condition fails, else clause executes
    field2 = IfThenElse(
        if_clause=String(),
        then_clause=Integer(),
        else_clause=Boolean()
    )
    # Non-string value should trigger else clause (Boolean validation)
    with pytest.raises(Exception) as exc_info:
        field2.validate(123)
    assert "Must be a boolean." in str(exc_info.value)
    
    # Test 3: Valid then clause execution
    field3 = IfThenElse(
        if_clause=Integer(minimum=0),
        then_clause=Integer(maximum=100),
        else_clause=String()
    )
    # Positive integer should pass then clause validation
    assert field3.validate(50) == 50
    
    # Test 4: Valid else clause execution
    field4 = IfThenElse(
        if_clause=Integer(minimum=0),
        then_clause=String(),
        else_clause=Integer(maximum=100)
    )
    # Negative integer should pass else clause validation
    assert field4.validate(-10) == -10
    
    # Test 5: Default then clause (Any)
    field5 = IfThenElse(
        if_clause=Integer(),
        else_clause=String()
    )
    # Integer should pass with default then clause (Any)
    assert field5.validate(123) == 123
    
    # Test 6: Default else clause (Any)
    field6 = IfThenElse(
        if_clause=String(),
        then_clause=Integer()
    )
    # Non-string should pass with default else clause (Any)
    assert field6.validate(True) == True
    
    # Test 7: Both clauses default to Any
    field7 = IfThenElse(if_clause=String())
    # Any value should pass
    assert field7.validate(123) == 123
    assert field7.validate("test") == "test"
    assert field7.validate(True) == True
    
    # Test 8: Nested conditions
    inner_field = IfThenElse(
        if_clause=Integer(minimum=10),
        then_clause=String(),
        else_clause=Boolean()
    )
    outer_field = IfThenElse(
        if_clause=inner_field,
        then_clause=Integer(),
        else_clause=String()
    )
    # Test nested validation
    with pytest.raises(Exception) as exc_info:
        outer_field.validate(5)
    assert "Must be a boolean." in str(exc_info.value)
    
    # Test 9: Complex condition with validation in clauses
    field9 = IfThenElse(
        if_clause=Integer(minimum=0, maximum=100),
        then_clause=Integer(minimum=50),
        else_clause=Integer(maximum=0)
    )
    # Value 75 passes if clause and then clause
    assert field9.validate(75) == 75
    # Value 25 passes if clause but fails then clause
    with pytest.raises(Exception) as exc_info:
        field9.validate(25)
    assert "Must be greater than or equal to 50" in str(exc_info.value)
    # Value -10 fails if clause but passes else clause
    assert field9.validate(-10) == -10
    # Value 150 fails if clause and fails else clause
    with pytest.raises(Exception) as exc_info:
        field9.validate(150)
    assert "Must be less than or equal to 0" in str(exc_info.value)


# LLM-generated content at query #3
#--------------------------

```python
def test_AllOf():
    from typesystem.fields import String, Integer, Boolean

    # Test basic initialization
    string_field = String()
    integer_field = Integer()
    all_of = AllOf([string_field, integer_field])
    assert all_of.all_of == [string_field, integer_field]
    assert len(all_of.all_of) == 2

    # Test that allow_null cannot be passed
    try:
        AllOf([string_field], allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass

    # Test with empty list
    all_of_empty = AllOf([])
    assert all_of_empty.all_of == []

    # Test with single field
    single_field = AllOf([string_field])
    assert single_field.all_of == [string_field]

    # Test with multiple different field types
    bool_field = Boolean()
    all_of_mixed = AllOf([string_field, integer_field, bool_field])
    assert len(all_of_mixed.all_of) == 3
    assert isinstance(all_of_mixed.all_of[0], String)
    assert isinstance(all_of_mixed.all_of[1], Integer)
    assert isinstance(all_of_mixed.all_of[2], Boolean)

    # Test that parent Field kwargs are passed through
    all_of_with_kwargs = AllOf([string_field], title="Test AllOf", description="Test description")
    assert all_of_with_kwargs.title == "Test AllOf"
    assert all_of_with_kwargs.description == "Test description"


# LLM-generated content at query #4
#--------------------------

```python
def test_IfThenElse_validate():
    from typesystem.fields import Integer, String, Boolean
    
    # Test 1: Basic if-then-else functionality
    int_field = Integer()
    string_field = String()
    bool_field = Boolean()
    
    # If integer, then string, else boolean
    field = IfThenElse(
        if_clause=int_field,
        then_clause=string_field,
        else_clause=bool_field
    )
    
    # Test with integer (should go to then_clause and fail with string validation)
    with pytest.raises(Exception) as exc_info:
        field.validate(123)
    assert "Must be a string." in str(exc_info.value)
    
    # Test with boolean value (should go to else_clause and succeed)
    result = field.validate(True)
    assert result is True
    
    # Test 2: If-then without else (uses Any() as default)
    field2 = IfThenElse(
        if_clause=String(min_length=3),
        then_clause=Integer()
    )
    
    # Test with long string (should go to then_clause and fail with integer validation)
    with pytest.raises(Exception) as exc_info:
        field2.validate("hello")
    assert "Must be a number." in str(exc_info.value)
    
    # Test with short string (should go to else_clause which is Any() and succeed)
    result = field2.validate("hi")
    assert result == "hi"
    
    # Test 3: If without then or else (both default to Any())
    field3 = IfThenElse(if_clause=Boolean())
    
    # Test with boolean (should go to then_clause which is Any() and succeed)
    result = field3.validate(False)
    assert result is False
    
    # Test with non-boolean (should go to else_clause which is Any() and succeed)
    result = field3.validate("not a boolean")
    assert result == "not a boolean"
    
    # Test 4: Nested conditional logic
    inner_field = IfThenElse(
        if_clause=Integer(minimum=10),
        then_clause=String(),
        else_clause=Boolean()
    )
    
    # Test with integer >= 10 (should go to then_clause and fail with string validation)
    with pytest.raises(Exception) as exc_info:
        inner_field.validate(15)
    assert "Must be a string." in str(exc_info.value)
    
    # Test with integer < 10 (should go to else_clause and succeed with boolean validation)
    with pytest.raises(Exception) as exc_info:
        inner_field.validate(5)
    assert "Must be a boolean." in str(exc_info.value)
    
    # Test 5: Complex condition with successful validation
    field5 = IfThenElse(
        if_clause=Integer(maximum=100),
        then_clause=Integer(minimum=50),  # Requires 50-100
        else_clause=Integer(minimum=0)    # Requires >= 0
    )
    
    # Test with value 75 (matches if_clause, then validates with then_clause)
    result = field5.validate(75)
    assert result == 75
    
    # Test with value 25 (matches if_clause, but fails then_clause validation)
    with pytest.raises(Exception) as exc_info:
        field5.validate(25)
    assert "Must be greater than or equal to 50." in str(exc_info.value)
    
    # Test with value -10 (doesn't match if_clause, goes to else_clause, fails validation)
    with pytest.raises(Exception) as exc_info:
        field5.validate(-10)
    assert "Must be greater than or equal to 0." in str(exc_info.value)
    
    # Test 6: Ensure allow_null is not allowed in constructor
    with pytest.raises(AssertionError):
        IfThenElse(if_clause=Integer(), allow_null=True)


# LLM-generated content at query #5
#--------------------------

```python
def test_Not_validate():
    from typesystem.fields import Integer, String
    
    # Test that Not rejects values that match the negated field
    not_int = Not(negated=Integer())
    with pytest.raises(Exception) as exc_info:
        not_int.validate(42)
    assert "Must not match" in str(exc_info.value)
    
    # Test that Not accepts values that don't match the negated field
    not_int = Not(negated=Integer())
    result = not_int.validate("hello")
    assert result == "hello"
    
    # Test with String field
    not_string = Not(negated=String(max_length=5))
    with pytest.raises(Exception) as exc_info:
        not_string.validate("short")
    assert "Must not match" in str(exc_info.value)
    
    # Test that Not accepts values that don't match the negated String field
    result = not_string.validate(123)
    assert result == 123
    
    # Test with nested field validation
    not_positive = Not(negated=Integer(minimum=0))
    with pytest.raises(Exception) as exc_info:
        not_positive.validate(10)
    assert "Must not match" in str(exc_info.value)
    
    # Test that Not accepts negative values when negating positive integers
    result = not_positive.validate(-5)
    assert result == -5
    
    # Test with null value when negated field doesn't allow null
    not_nullable = Not(negated=Integer())
    result = not_nullable.validate(None)
    assert result is None
    
    # Test with complex field
    from typesystem.fields import Boolean
    not_bool = Not(negated=Boolean())
    with pytest.raises(Exception) as exc_info:
        not_bool.validate(True)
    assert "Must not match" in str(exc_info.value)
    
    # Test that Not accepts non-boolean values
    result = not_bool.validate("not a boolean")
    assert result == "not a boolean"


# LLM-generated content at query #6
#--------------------------

```python
def test_Not():
    # Test basic Not functionality with a simple field
    from typesystem.fields import Integer
    
    integer_field = Integer()
    not_field = Not(negated=integer_field)
    
    # Should accept non-integer values
    assert not_field.validate("string") == "string"
    assert not_field.validate(True) == True
    assert not_field.validate(None) == None
    
    # Should reject integer values
    try:
        not_field.validate(42)
        assert False, "Should have raised validation error for integer"
    except Exception as e:
        assert "Must not match" in str(e)
    
    # Test with more complex field
    from typesystem.fields import String
    
    string_field = String(max_length=5)
    not_string_field = Not(negated=string_field)
    
    # Should reject short strings
    try:
        not_string_field.validate("hello")
        assert False, "Should have raised validation error for valid string"
    except Exception as e:
        assert "Must not match" in str(e)
    
    # Should accept long strings
    assert not_string_field.validate("very long string") == "very long string"
    
    # Test that allow_null is not allowed in constructor
    try:
        Not(negated=integer_field, allow_null=True)
        assert False, "Should have raised assertion error for allow_null"
    except AssertionError:
        pass
    
    # Test with custom error message
    not_field_with_error = Not(negated=integer_field, errors={"negated": "Custom error"})
    try:
        not_field_with_error.validate(42)
        assert False, "Should have raised validation error"
    except Exception as e:
        assert "Custom error" in str(e)


# LLM-generated content at query #7
#--------------------------

```python
def test_NeverMatch():
    field = NeverMatch()
    assert isinstance(field, Field)
    assert field.errors == {"never": "This never validates."}
    
    with pytest.raises(AssertionError):
        NeverMatch(allow_null=True)
    
    with pytest.raises(field.validation_error("never")):
        field.validate(None)
    
    with pytest.raises(field.validation_error("never")):
        field.validate("test")
    
    with pytest.raises(field.validation_error("never")):
        field.validate(123)
    
    with pytest.raises(field.validation_error("never")):
        field.validate([])


# LLM-generated content at query #8
#--------------------------

```python
def test_OneOf():
    # Test basic initialization
    field1 = Any()
    field2 = NeverMatch()
    one_of = OneOf([field1, field2])
    assert one_of.one_of == [field1, field2]
    
    # Test that allow_null cannot be passed
    try:
        OneOf([field1], allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test with empty list
    one_of_empty = OneOf([])
    assert one_of_empty.one_of == []
    
    # Test validation with single match
    result = one_of.validate("test")
    assert result == "test"
    
    # Test validation with no matches
    try:
        one_of.validate("test")
    except Exception as e:
        assert str(e) == "Did not match any valid type."
    
    # Test validation with multiple matches
    field3 = Any()
    field4 = Any()
    one_of_multiple = OneOf([field3, field4])
    try:
        one_of_multiple.validate("test")
    except Exception as e:
        assert str(e) == "Matched more than one type."


# LLM-generated content at query #9
#--------------------------

```python
def test_OneOf():
    # Test basic initialization with list of fields
    field1 = Any()
    field2 = NeverMatch()
    one_of = OneOf([field1, field2])
    assert one_of.one_of == [field1, field2]
    
    # Test that allow_null cannot be passed
    try:
        OneOf([field1], allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test with empty list
    one_of_empty = OneOf([])
    assert one_of_empty.one_of == []
    
    # Test validation with single match
    result = one_of.validate("test")
    assert result == "test"
    
    # Test validation with no matches
    try:
        one_of.validate("test")
    except Exception as e:
        assert str(e) == "Did not match any valid type."
    
    # Test validation with multiple matches
    field3 = Any()
    field4 = Any()
    one_of_multiple = OneOf([field3, field4])
    try:
        one_of_multiple.validate("test")
    except Exception as e:
        assert str(e) == "Matched more than one type."


# LLM-generated content at query #10
#--------------------------

```python
def test_OneOf():
    from typesystem.fields import Integer, String, Boolean
    
    # Test basic initialization
    field = OneOf(one_of=[Integer(), String()])
    assert field.one_of[0].__class__.__name__ == "Integer"
    assert field.one_of[1].__class__.__name__ == "String"
    
    # Test that allow_null cannot be passed
    try:
        OneOf(one_of=[Integer()], allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test validation with exactly one match
    field = OneOf(one_of=[Integer(), String()])
    result = field.validate(42)
    assert result == 42
    
    # Test validation with no matches
    field = OneOf(one_of=[Integer(), Boolean()])
    try:
        field.validate("not a match")
        assert False, "Should have raised validation error"
    except Exception as e:
        assert "no_match" in str(e)
    
    # Test validation with multiple matches (ambiguous)
    field = OneOf(one_of=[Any(), Any()])
    try:
        field.validate("any value")
        assert False, "Should have raised validation error"
    except Exception as e:
        assert "multiple_matches" in str(e)
    
    # Test with empty one_of list
    field = OneOf(one_of=[])
    try:
        field.validate("anything")
        assert False, "Should have raised validation error"
    except Exception as e:
        assert "no_match" in str(e)


# LLM-generated content at query #11
#--------------------------

```python
def test_Not():
    # Test basic Not functionality with a simple field
    from typesystem.fields import Integer
    
    integer_field = Integer()
    not_field = Not(negated=integer_field)
    
    # Should reject integer values
    with pytest.raises(not_field.validation_error("negated")):
        not_field.validate(42)
    
    # Should accept non-integer values
    assert not_field.validate("string") == "string"
    assert not_field.validate(3.14) == 3.14
    assert not_field.validate(True) == True
    
    # Test with allow_null=False (default)
    from typesystem.fields import String
    
    string_field = String()
    not_string = Not(negated=string_field)
    
    # Should reject string values
    with pytest.raises(not_string.validation_error("negated")):
        not_string.validate("hello")
    
    # Should accept non-string values
    assert not_string.validate(123) == 123
    
    # Test that allow_null cannot be passed to Not constructor
    with pytest.raises(AssertionError):
        Not(negated=integer_field, allow_null=True)
    
    # Test with complex field
    from typesystem.fields import Array
    
    array_field = Array(items=Integer())
    not_array = Not(negated=array_field)
    
    # Should reject arrays
    with pytest.raises(not_array.validation_error("negated")):
        not_array.validate([1, 2, 3])
    
    # Should accept non-arrays
    assert not_array.validate("not an array") == "not an array"
    
    # Test error message
    try:
        not_field.validate(42)
    except Exception as exc:
        assert "Must not match." in str(exc)


# LLM-generated content at query #12
#--------------------------

```python
def test_OneOf():
    from typesystem.fields import Integer, String, Boolean
    
    # Test basic initialization
    field = OneOf(one_of=[Integer(), String()])
    assert field.one_of[0].__class__.__name__ == "Integer"
    assert field.one_of[1].__class__.__name__ == "String"
    
    # Test that allow_null cannot be passed
    try:
        OneOf(one_of=[Integer()], allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test validation with exactly one match
    field = OneOf(one_of=[Integer(), String()])
    result = field.validate(42)
    assert result == 42
    
    # Test validation with no matches
    field = OneOf(one_of=[Integer(), String()])
    try:
        field.validate(True)
        assert False, "Should have raised validation error"
    except Exception as e:
        assert "no_match" in str(e)
    
    # Test validation with multiple matches (ambiguous)
    field = OneOf(one_of=[Any(), Integer()])
    try:
        field.validate(42)
        assert False, "Should have raised validation error"
    except Exception as e:
        assert "multiple_matches" in str(e)
    
    # Test with empty one_of list
    field = OneOf(one_of=[])
    try:
        field.validate("anything")
        assert False, "Should have raised validation error"
    except Exception as e:
        assert "no_match" in str(e)
    
    # Test with nested fields
    field = OneOf(one_of=[Boolean(), OneOf(one_of=[Integer(), String()])])
    result = field.validate(True)
    assert result is True
    
    # Test validation_error method exists and works
    field = OneOf(one_of=[Integer()])
    try:
        field.validate("not an integer")
        assert False, "Should have raised validation error"
    except Exception as e:
        assert "no_match" in str(e)


# LLM-generated content at query #13
#--------------------------

```python
def test_NeverMatch():
    # Test that NeverMatch cannot be instantiated with allow_null
    with pytest.raises(AssertionError):
        NeverMatch(allow_null=True)
    
    # Test that NeverMatch can be instantiated without allow_null
    field = NeverMatch()
    assert isinstance(field, NeverMatch)
    assert isinstance(field, Field)
    
    # Test that validate always raises validation error
    with pytest.raises(field.validation_error("never")):
        field.validate(None)
    
    with pytest.raises(field.validation_error("never")):
        field.validate("any value")
    
    with pytest.raises(field.validation_error("never")):
        field.validate(123)
    
    with pytest.raises(field.validation_error("never")):
        field.validate({"key": "value"})
    
    # Test that the error message is correct
    try:
        field.validate("test")
    except Exception as e:
        assert str(e) == "This never validates."


# LLM-generated content at query #14
#--------------------------

```python
def test_AllOf():
    from typesystem.fields import Integer, String
    
    # Test basic initialization
    int_field = Integer()
    str_field = String()
    all_of = AllOf([int_field, str_field])
    
    assert all_of.all_of == [int_field, str_field]
    assert len(all_of.all_of) == 2
    
    # Test that allow_null cannot be passed
    try:
        AllOf([int_field], allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test with empty list
    all_of_empty = AllOf([])
    assert all_of_empty.all_of == []
    
    # Test validation passes when all fields validate
    result = all_of.validate(123)
    assert result == 123
    
    # Test validation fails when any field fails
    try:
        all_of.validate("not_a_number")
        assert False, "Should have raised ValidationError"
    except Exception as e:
        assert "Must be a number." in str(e)


# LLM-generated content at query #15
#--------------------------

```python
def test_AllOf():
    # Test basic initialization
    field1 = Any()
    field2 = Any()
    all_of = AllOf([field1, field2])
    assert all_of.all_of == [field1, field2]
    
    # Test that allow_null cannot be passed
    try:
        AllOf([field1], allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test with empty list
    all_of_empty = AllOf([])
    assert all_of_empty.all_of == []
    
    # Test validation with multiple fields
    from typesystem.fields import Integer, String
    
    int_field = Integer()
    str_field = String()
    all_of_mixed = AllOf([int_field, str_field])
    
    # This should fail because value can't be both integer and string
    try:
        all_of_mixed.validate(123)
        assert False, "Should have raised ValidationError"
    except Exception:
        pass
    
    # Test validation with compatible fields
    from typesystem.fields import Number
    
    number1 = Number(minimum=0)
    number2 = Number(maximum=10)
    all_of_compatible = AllOf([number1, number2])
    
    # Should validate successfully
    result = all_of_compatible.validate(5)
    assert result == 5
    
    # Should fail validation
    try:
        all_of_compatible.validate(15)
        assert False, "Should have raised ValidationError"
    except Exception:
        pass
    
    try:
        all_of_compatible.validate(-5)
        assert False, "Should have raised ValidationError"
    except Exception:
        pass


# LLM-generated content at query #16
#--------------------------

```python
def test_NeverMatch():
    # Test that NeverMatch cannot be instantiated with allow_null
    with pytest.raises(AssertionError):
        NeverMatch(allow_null=True)
    
    # Test that NeverMatch can be instantiated without allow_null
    field = NeverMatch()
    assert isinstance(field, NeverMatch)
    assert isinstance(field, Field)
    
    # Test that validate always raises validation error
    with pytest.raises(field.validation_error) as exc_info:
        field.validate("any value")
    assert str(exc_info.value) == "This never validates."
    
    # Test with different types of values
    with pytest.raises(field.validation_error):
        field.validate(None)
    
    with pytest.raises(field.validation_error):
        field.validate(123)
    
    with pytest.raises(field.validation_error):
        field.validate({"key": "value"})
    
    # Test that errors dictionary is correctly set
    assert field.errors == {"never": "This never validates."}


# LLM-generated content at query #17
#--------------------------

```python
def test_Not():
    # Test basic initialization
    negated_field = Any()
    not_field = Not(negated=negated_field)
    assert not_field.negated == negated_field
    assert not_field.errors == {"negated": "Must not match."}
    
    # Test that allow_null cannot be passed
    try:
        Not(negated=negated_field, allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test with different field types
    from typesystem import Integer, String
    int_field = Integer()
    not_int = Not(negated=int_field)
    assert not_int.negated == int_field
    
    string_field = String()
    not_string = Not(negated=string_field)
    assert not_string.negated == string_field
    
    # Test validation when negated field fails
    not_field = Not(negated=Integer())
    result = not_field.validate("not an integer")
    assert result == "not an integer"
    
    # Test validation when negated field succeeds
    not_field = Not(negated=Any())
    try:
        not_field.validate("any value")
        assert False, "Should have raised validation error"
    except Exception as e:
        assert "negated" in str(e)
    
    # Test with custom error messages
    not_field = Not(negated=Integer(), error_messages={"negated": "Custom error"})
    try:
        not_field.validate(123)
        assert False, "Should have raised validation error"
    except Exception as e:
        assert "Custom error" in str(e)


# LLM-generated content at query #18
#--------------------------

```python
def test_AllOf():
    from typesystem.fields import Integer, String
    
    # Test basic initialization
    integer_field = Integer()
    string_field = String()
    all_of = AllOf([integer_field, string_field])
    
    assert all_of.all_of == [integer_field, string_field]
    assert all_of.allow_null is False
    
    # Test that allow_null cannot be passed
    try:
        AllOf([integer_field], allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test with empty list
    all_of_empty = AllOf([])
    assert all_of_empty.all_of == []
    
    # Test validation with multiple fields
    all_of = AllOf([Integer(minimum=0), Integer(maximum=10)])
    
    # Should validate successfully
    result = all_of.validate(5)
    assert result == 5
    
    # Should fail validation (less than minimum)
    try:
        all_of.validate(-1)
        assert False, "Should have raised ValidationError"
    except Exception as e:
        assert "minimum" in str(e).lower()
    
    # Should fail validation (greater than maximum)
    try:
        all_of.validate(11)
        assert False, "Should have raised ValidationError"
    except Exception as e:
        assert "maximum" in str(e).lower()
    
    # Test with different field types
    all_of_mixed = AllOf([Integer(), String(max_length=5)])
    
    # Integer validation should pass
    result = all_of_mixed.validate(42)
    assert result == 42
    
    # String validation should pass
    result = all_of_mixed.validate("test")
    assert result == "test"
    
    # String too long should fail
    try:
        all_of_mixed.validate("toolong")
        assert False, "Should have raised ValidationError"
    except Exception as e:
        assert "maximum" in str(e).lower() or "length" in str(e).lower()


# LLM-generated content at query #19
#--------------------------

```python
def test_NeverMatch():
    field = NeverMatch()
    assert isinstance(field, Field)
    assert field.errors == {"never": "This never validates."}
    
    with pytest.raises(AssertionError):
        NeverMatch(allow_null=True)
    
    with pytest.raises(field.validation_error("never")):
        field.validate("any value")
    
    with pytest.raises(field.validation_error("never")):
        field.validate(None)
    
    with pytest.raises(field.validation_error("never")):
        field.validate(123)
    
    with pytest.raises(field.validation_error("never")):
        field.validate([])


# LLM-generated content at query #20
#--------------------------

```python
def test_NeverMatch():
    # Test that NeverMatch cannot be instantiated with allow_null
    try:
        NeverMatch(allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError as e:
        assert "allow_null" in str(e)
    
    # Test that NeverMatch can be instantiated without allow_null
    field = NeverMatch()
    assert isinstance(field, NeverMatch)
    assert isinstance(field, Field)
    
    # Test that validate always raises validation error
    try:
        field.validate(None)
        assert False, "Should have raised validation error"
    except Exception as e:
        assert hasattr(e, 'code')
        assert e.code == 'never'
    
    try:
        field.validate(123)
        assert False, "Should have raised validation error"
    except Exception as e:
        assert hasattr(e, 'code')
        assert e.code == 'never'
    
    try:
        field.validate("test")
        assert False, "Should have raised validation error"
    except Exception as e:
        assert hasattr(e, 'code')
        assert e.code == 'never'


# LLM-generated content at query #21
#--------------------------

```python
def test_OneOf():
    # Test basic initialization with list of fields
    field1 = Any()
    field2 = NeverMatch()
    one_of = OneOf([field1, field2])
    assert one_of.one_of == [field1, field2]
    
    # Test that allow_null cannot be passed
    try:
        OneOf([field1], allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test with empty list
    one_of_empty = OneOf([])
    assert one_of_empty.one_of == []
    
    # Test validation with single match
    result = one_of.validate("test")
    assert result == "test"
    
    # Test validation with no matches
    try:
        one_of.validate("test")
    except Exception as e:
        assert str(e) == "Did not match any valid type."
    
    # Test validation with multiple matches
    field3 = Any()
    field4 = Any()
    one_of_multiple = OneOf([field3, field4])
    try:
        one_of_multiple.validate("test")
    except Exception as e:
        assert str(e) == "Matched more than one type."


# LLM-generated content at query #22
#--------------------------

```python
def test_AllOf():
    # Test basic initialization
    field1 = Any()
    field2 = Any()
    all_of = AllOf([field1, field2])
    assert all_of.all_of == [field1, field2]
    
    # Test that allow_null cannot be passed
    try:
        AllOf([field1], allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test with empty list
    all_of_empty = AllOf([])
    assert all_of_empty.all_of == []
    
    # Test validation passes when all fields validate
    class AlwaysPass(Field):
        def validate(self, value):
            return value
    
    pass1 = AlwaysPass()
    pass2 = AlwaysPass()
    all_of_pass = AllOf([pass1, pass2])
    result = all_of_pass.validate("test_value")
    assert result == "test_value"
    
    # Test validation fails when any field fails
    class AlwaysFail(Field):
        def validate(self, value):
            raise self.validation_error("test_error")
    
    fail_field = AlwaysFail()
    all_of_mixed = AllOf([pass1, fail_field])
    try:
        all_of_mixed.validate("test_value")
        assert False, "Should have raised ValidationError"
    except ValidationError:
        pass
    
    # Test validation order - all fields should be validated
    validation_order = []
    
    class TrackingField(Field):
        def __init__(self, name):
            super().__init__()
            self.name = name
        
        def validate(self, value):
            validation_order.append(self.name)
            return value
    
    field_a = TrackingField("A")
    field_b = TrackingField("B")
    field_c = TrackingField("C")
    
    all_of_tracking = AllOf([field_a, field_b, field_c])
    all_of_tracking.validate("test")
    assert validation_order == ["A", "B", "C"]


# LLM-generated content at query #23
#--------------------------

```python
def test_NeverMatch():
    field = NeverMatch()
    assert isinstance(field, NeverMatch)
    assert field.errors == {"never": "This never validates."}
    
    with pytest.raises(AssertionError):
        NeverMatch(allow_null=True)
    
    with pytest.raises(field.validation_error("never")):
        field.validate("any value")
    
    with pytest.raises(field.validation_error("never")):
        field.validate(None)
    
    with pytest.raises(field.validation_error("never")):
        field.validate(123)
    
    with pytest.raises(field.validation_error("never")):
        field.validate([])


# LLM-generated content at query #24
#--------------------------

```python
def test_Not():
    # Test basic Not functionality with a simple field
    from typesystem.fields import Integer
    
    integer_field = Integer()
    not_field = Not(negated=integer_field)
    
    # Should accept non-integer values
    assert not_field.validate("string") == "string"
    assert not_field.validate(True) == True
    assert not_field.validate(None) == None
    
    # Should reject integer values
    try:
        not_field.validate(42)
        assert False, "Should have raised validation error for integer"
    except Exception as e:
        assert "Must not match" in str(e)
    
    # Test with more complex field
    from typesystem.fields import String
    
    string_field = String(max_length=5)
    not_string_field = Not(negated=string_field)
    
    # Should reject short strings
    try:
        not_string_field.validate("short")
        assert False, "Should have raised validation error for short string"
    except Exception as e:
        assert "Must not match" in str(e)
    
    # Should accept long strings
    assert not_string_field.validate("very long string") == "very long string"
    
    # Test that allow_null is not allowed in constructor
    try:
        Not(negated=integer_field, allow_null=True)
        assert False, "Should have raised assertion error for allow_null"
    except AssertionError:
        pass
    
    # Test with nested field
    from typesystem.fields import Boolean
    from typesystem.schemas import Schema
    
    class TestSchema(Schema):
        field = Integer()
    
    schema_field = TestSchema()
    not_schema_field = Not(negated=schema_field)
    
    # Should reject valid schema data
    try:
        not_schema_field.validate({"field": 42})
        assert False, "Should have raised validation error for schema match"
    except Exception as e:
        assert "Must not match" in str(e)
    
    # Should accept invalid schema data
    assert not_schema_field.validate({"field": "not integer"}) == {"field": "not integer"}


# LLM-generated content at query #25
#--------------------------

```python
def test_NeverMatch():
    # Test that NeverMatch cannot be instantiated with allow_null
    with pytest.raises(AssertionError):
        NeverMatch(allow_null=True)
    
    # Test that NeverMatch can be instantiated without allow_null
    field = NeverMatch()
    assert isinstance(field, NeverMatch)
    assert isinstance(field, Field)
    
    # Test that validate always raises validation error
    with pytest.raises(field.validation_error) as exc_info:
        field.validate("any value")
    assert str(exc_info.value) == "This never validates."
    
    # Test with different types of values
    with pytest.raises(field.validation_error):
        field.validate(None)
    
    with pytest.raises(field.validation_error):
        field.validate(123)
    
    with pytest.raises(field.validation_error):
        field.validate({"key": "value"})
    
    with pytest.raises(field.validation_error):
        field.validate([])
    
    # Test that errors dictionary is correctly set
    assert field.errors == {"never": "This never validates."}


# LLM-generated content at query #26
#--------------------------

```python
def test_OneOf():
    from typesystem.fields import String, Integer, Boolean
    
    # Test basic initialization
    field = OneOf(one_of=[String(), Integer()])
    assert field.one_of[0].__class__.__name__ == "String"
    assert field.one_of[1].__class__.__name__ == "Integer"
    
    # Test that allow_null cannot be passed
    try:
        OneOf(one_of=[String()], allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test validation with exactly one match
    field = OneOf(one_of=[String(), Integer()])
    result = field.validate("hello")
    assert result == "hello"
    
    result = field.validate(42)
    assert result == 42
    
    # Test validation with no matches
    try:
        field.validate(True)
        assert False, "Should have raised validation error"
    except Exception as e:
        assert "no_match" in str(e)
    
    # Test validation with multiple matches
    # Create a field that matches both string and boolean
    field = OneOf(one_of=[String(), Boolean()])
    try:
        field.validate(True)
        assert False, "Should have raised validation error"
    except Exception as e:
        assert "multiple_matches" in str(e)
    
    # Test with nested fields
    nested_field = OneOf(one_of=[String(max_length=5), Integer(minimum=0)])
    result = nested_field.validate("test")
    assert result == "test"
    
    result = nested_field.validate(10)
    assert result == 10
    
    try:
        nested_field.validate("toolongstring")
        assert False, "Should have raised validation error"
    except Exception:
        pass
    
    try:
        nested_field.validate(-5)
        assert False, "Should have raised validation error"
    except Exception:
        pass


# LLM-generated content at query #27
#--------------------------

```python
def test_AllOf():
    # Test basic initialization
    field1 = Any()
    field2 = Any()
    all_of_field = AllOf([field1, field2])
    assert all_of_field.all_of == [field1, field2]
    
    # Test that allow_null cannot be passed
    try:
        AllOf([field1], allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test with empty list
    empty_all_of = AllOf([])
    assert empty_all_of.all_of == []
    
    # Test validation with multiple fields
    from typesystem.fields import Integer, String
    
    int_field = Integer()
    positive_field = Integer(minimum=0)
    all_of_field = AllOf([int_field, positive_field])
    
    # Should validate successfully
    result = all_of_field.validate(5)
    assert result == 5
    
    # Should raise validation error for non-integer
    try:
        all_of_field.validate("not an integer")
        assert False, "Should have raised ValidationError"
    except Exception as e:
        assert "Must be a number." in str(e)
    
    # Should raise validation error for negative integer
    try:
        all_of_field.validate(-1)
        assert False, "Should have raised ValidationError"
    except Exception as e:
        assert "Must be greater than or equal to 0." in str(e)
    
    # Test validation with empty all_of list
    empty_all_of = AllOf([])
    result = empty_all_of.validate("any value")
    assert result == "any value"


# LLM-generated content at query #28
#--------------------------

```python
def test_Not():
    # Test basic initialization
    negated_field = Any()
    not_field = Not(negated=negated_field)
    assert not_field.negated == negated_field
    assert not_field.errors == {"negated": "Must not match."}

    # Test that allow_null cannot be passed
    try:
        Not(negated=negated_field, allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass

    # Test validation when negated field fails (should succeed)
    negated_field = NeverMatch()
    not_field = Not(negated=negated_field)
    result = not_field.validate("any_value")
    assert result == "any_value"

    # Test validation when negated field succeeds (should fail)
    negated_field = Any()
    not_field = Not(negated=negated_field)
    try:
        not_field.validate("any_value")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "negated"
        assert e.text == "Must not match."

    # Test with more complex negated field
    from typesystem import Integer
    negated_field = Integer()
    not_field = Not(negated=negated_field)
    
    # String should pass (not an integer)
    result = not_field.validate("not_an_integer")
    assert result == "not_an_integer"
    
    # Integer should fail
    try:
        not_field.validate(42)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert e.code == "negated"


# LLM-generated content at query #29
#--------------------------

```python
def test_OneOf():
    from typesystem.fields import Integer, String, Boolean
    
    # Test basic initialization
    integer_field = Integer()
    string_field = String()
    one_of = OneOf(one_of=[integer_field, string_field])
    assert one_of.one_of == [integer_field, string_field]
    
    # Test that allow_null cannot be passed
    try:
        OneOf(one_of=[integer_field], allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test with empty list
    one_of_empty = OneOf(one_of=[])
    assert one_of_empty.one_of == []
    
    # Test with single field
    single_field = OneOf(one_of=[integer_field])
    assert single_field.one_of == [integer_field]
    
    # Test with three fields
    boolean_field = Boolean()
    one_of_three = OneOf(one_of=[integer_field, string_field, boolean_field])
    assert len(one_of_three.one_of) == 3
    assert isinstance(one_of_three.one_of[0], Integer)
    assert isinstance(one_of_three.one_of[1], String)
    assert isinstance(one_of_three.one_of[2], Boolean)


# LLM-generated content at query #30
#--------------------------

```python
def test_OneOf():
    # Test basic initialization
    field1 = Any()
    field2 = Any()
    one_of = OneOf(one_of=[field1, field2])
    assert one_of.one_of == [field1, field2]
    
    # Test that allow_null cannot be passed
    try:
        OneOf(one_of=[field1], allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test with empty list
    one_of_empty = OneOf(one_of=[])
    assert one_of_empty.one_of == []
    
    # Test with single field
    single_field = Any()
    one_of_single = OneOf(one_of=[single_field])
    assert one_of_single.one_of == [single_field]
    
    # Test that it inherits from Field
    assert isinstance(one_of, Field)


# LLM-generated content at query #31
#--------------------------

```python
def test_OneOf():
    # Test basic initialization with list of fields
    from typesystem.fields import String, Integer
    string_field = String()
    integer_field = Integer()
    one_of = OneOf(one_of=[string_field, integer_field])
    assert one_of.one_of == [string_field, integer_field]
    assert one_of.errors == {
        "no_match": "Did not match any valid type.",
        "multiple_matches": "Matched more than one type."
    }

    # Test that allow_null cannot be passed in kwargs
    try:
        OneOf(one_of=[string_field], allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass

    # Test with empty list
    one_of_empty = OneOf(one_of=[])
    assert one_of_empty.one_of == []

    # Test with single field
    single_field = OneOf(one_of=[string_field])
    assert single_field.one_of == [string_field]

    # Test that parent Field __init__ is called with kwargs
    custom_one_of = OneOf(one_of=[string_field], title="Test Field")
    assert custom_one_of.title == "Test Field"


# LLM-generated content at query #32
#--------------------------

```python
def test_OneOf():
    from typesystem.fields import Integer, String, Boolean
    
    # Test basic initialization
    field = OneOf(one_of=[Integer(), String()])
    assert field.one_of[0].__class__.__name__ == "Integer"
    assert field.one_of[1].__class__.__name__ == "String"
    
    # Test that allow_null cannot be passed
    try:
        OneOf(one_of=[Integer()], allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test validation with exactly one match
    field = OneOf(one_of=[Integer(), String()])
    result = field.validate(42)
    assert result == 42
    
    # Test validation with no matches
    field = OneOf(one_of=[Integer(), String()])
    try:
        field.validate(True)
        assert False, "Should have raised validation error"
    except Exception as e:
        assert "no_match" in str(e)
    
    # Test validation with multiple matches (should be rare but possible)
    field = OneOf(one_of=[Any(), Any()])
    try:
        field.validate("anything")
        assert False, "Should have raised validation error"
    except Exception as e:
        assert "multiple_matches" in str(e)
    
    # Test with empty list
    field = OneOf(one_of=[])
    try:
        field.validate("anything")
        assert False, "Should have raised validation error"
    except Exception as e:
        assert "no_match" in str(e)
    
    # Test that kwargs are passed to parent
    field = OneOf(one_of=[Integer()], title="Test Field")
    assert field.title == "Test Field"


# LLM-generated content at query #33
#--------------------------

```python
def test_NeverMatch():
    field = NeverMatch()
    assert isinstance(field, NeverMatch)
    assert field.errors == {"never": "This never validates."}
    
    with pytest.raises(AssertionError):
        NeverMatch(allow_null=True)
    
    with pytest.raises(AssertionError):
        NeverMatch(allow_null=False)
    
    with pytest.raises(field.validation_error):
        field.validate("any value")
    
    with pytest.raises(field.validation_error):
        field.validate(None)
    
    with pytest.raises(field.validation_error):
        field.validate(123)
    
    with pytest.raises(field.validation_error):
        field.validate([])


# LLM-generated content at query #34
#--------------------------

```python
def test_AllOf():
    from typesystem.fields import Integer, String
    
    # Test basic initialization
    integer_field = Integer()
    string_field = String()
    all_of = AllOf([integer_field, string_field])
    
    assert all_of.all_of == [integer_field, string_field]
    assert all_of.allow_null is False
    
    # Test that allow_null cannot be passed
    try:
        AllOf([integer_field], allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test with empty list
    all_of_empty = AllOf([])
    assert all_of_empty.all_of == []
    
    # Test validation with multiple fields
    all_of = AllOf([Integer(minimum=0), Integer(maximum=10)])
    
    # Should validate successfully
    result = all_of.validate(5)
    assert result == 5
    
    # Should fail validation (fails first condition)
    try:
        all_of.validate(-1)
        assert False, "Should have raised ValidationError"
    except Exception as e:
        assert "Must be greater than or equal to 0" in str(e)
    
    # Should fail validation (fails second condition)
    try:
        all_of.validate(15)
        assert False, "Should have raised ValidationError"
    except Exception as e:
        assert "Must be less than or equal to 10" in str(e)
    
    # Test with different field types
    all_of_mixed = AllOf([Integer(), String(max_length=5)])
    
    # Integer validation should pass
    result = all_of_mixed.validate(42)
    assert result == 42
    
    # String validation should pass
    result = all_of_mixed.validate("test")
    assert result == "test"
    
    # String validation should fail (too long)
    try:
        all_of_mixed.validate("toolong")
        assert False, "Should have raised ValidationError"
    except Exception as e:
        assert "Must have no more than 5 characters" in str(e)


# LLM-generated content at query #35
#--------------------------

```python
def test_OneOf():
    # Test basic initialization with list of fields
    field1 = Any()
    field2 = NeverMatch()
    one_of = OneOf([field1, field2])
    assert one_of.one_of == [field1, field2]
    
    # Test that allow_null cannot be passed
    try:
        OneOf([field1], allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test with empty list
    one_of_empty = OneOf([])
    assert one_of_empty.one_of == []
    
    # Test validation with exactly one match
    result = one_of.validate("test")
    assert result == "test"
    
    # Test validation with no matches
    try:
        one_of.validate("test")
    except Exception as e:
        assert str(e) == "Did not match any valid type."
    
    # Test validation with multiple matches
    field3 = Any()
    one_of_multiple = OneOf([field1, field3])
    result = one_of_multiple.validate("test")
    assert result == "test"
    
    # Test validation with multiple matches that should fail
    field4 = Any()
    field5 = Any()
    one_of_conflict = OneOf([field4, field5])
    try:
        one_of_conflict.validate("test")
        assert False, "Should have raised validation error"
    except Exception as e:
        assert str(e) == "Matched more than one type."


# LLM-generated content at query #36
#--------------------------

```python
def test_NeverMatch():
    # Test that NeverMatch cannot be instantiated with allow_null
    try:
        NeverMatch(allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError as e:
        assert "allow_null" in str(e)
    
    # Test that NeverMatch can be instantiated without allow_null
    field = NeverMatch()
    assert isinstance(field, NeverMatch)
    assert isinstance(field, Field)
    
    # Test that validate always raises validation error
    try:
        field.validate(None)
        assert False, "Should have raised validation error"
    except Exception as e:
        assert "never" in str(e)
    
    try:
        field.validate("test")
        assert False, "Should have raised validation error"
    except Exception as e:
        assert "never" in str(e)
    
    try:
        field.validate(123)
        assert False, "Should have raised validation error"
    except Exception as e:
        assert "never" in str(e)
    
    # Test with additional kwargs
    field_with_kwargs = NeverMatch(description="Test field")
    assert field_with_kwargs.description == "Test field"


# LLM-generated content at query #37
#--------------------------

```python
def test_OneOf():
    # Test basic initialization with list of fields
    field1 = Any()
    field2 = NeverMatch()
    one_of = OneOf(one_of=[field1, field2])
    assert one_of.one_of == [field1, field2]
    assert one_of.errors == {
        "no_match": "Did not match any valid type.",
        "multiple_matches": "Matched more than one type.",
    }

    # Test that allow_null cannot be passed
    try:
        OneOf(one_of=[], allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass

    # Test with empty list
    one_of_empty = OneOf(one_of=[])
    assert one_of_empty.one_of == []

    # Test with single field
    single_field = Any()
    one_of_single = OneOf(one_of=[single_field])
    assert one_of_single.one_of == [single_field]

    # Test that validate method works correctly
    # Test case: exactly one match
    string_field = Field(type="string")
    number_field = Field(type="number")
    one_of_fields = OneOf(one_of=[string_field, number_field])
    
    # Should match string field only
    result = one_of_fields.validate("test")
    assert result == "test"
    
    # Should match number field only
    result = one_of_fields.validate(123)
    assert result == 123

    # Test case: no match
    try:
        one_of_fields.validate(True)
        assert False, "Should have raised validation error"
    except Exception as e:
        assert "no_match" in str(e)

    # Test case: multiple matches (both fields accept the value)
    any_field1 = Any()
    any_field2 = Any()
    one_of_any = OneOf(one_of=[any_field1, any_field2])
    
    try:
        one_of_any.validate("anything")
        assert False, "Should have raised validation error for multiple matches"
    except Exception as e:
        assert "multiple_matches" in str(e)

    # Test with NeverMatch field
    never_field = NeverMatch()
    one_of_with_never = OneOf(one_of=[field1, never_field])
    result = one_of_with_never.validate("test")
    assert result == "test"


# LLM-generated content at query #38
#--------------------------

```python
def test_Not():
    # Test that Not raises assertion error if allow_null is passed
    try:
        Not(negated=Any(), allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass

    # Test that Not validates successfully when negated field fails
    negated_field = NeverMatch()
    not_field = Not(negated=negated_field)
    result = not_field.validate("any_value")
    assert result == "any_value"

    # Test that Not raises validation error when negated field succeeds
    negated_field = Any()
    not_field = Not(negated=negated_field)
    try:
        not_field.validate("any_value")
        assert False, "Should have raised validation error"
    except Exception as e:
        assert e.code == "negated"
        assert str(e) == "Must not match."

    # Test with a more complex negated field
    from typesystem import Integer
    negated_field = Integer()
    not_field = Not(negated=negated_field)
    
    # Should succeed with non-integer value
    result = not_field.validate("string")
    assert result == "string"
    
    # Should fail with integer value
    try:
        not_field.validate(42)
        assert False, "Should have raised validation error"
    except Exception as e:
        assert e.code == "negated"

    # Test that negated attribute is accessible
    negated_field = Any()
    not_field = Not(negated=negated_field)
    assert not_field.negated is negated_field


# LLM-generated content at query #39
#--------------------------

```python
def test_Not():
    # Test basic initialization with a negated field
    from typesystem.fields import String
    negated_field = String(max_length=5)
    not_field = Not(negated=negated_field)
    assert not_field.negated == negated_field
    assert not_field.errors == {"negated": "Must not match."}
    
    # Test that allow_null cannot be passed in kwargs
    try:
        Not(negated=negated_field, allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test validation with value that doesn't match negated field
    result = not_field.validate("toolongstring")
    assert result == "toolongstring"
    
    # Test validation with value that matches negated field (should raise error)
    try:
        not_field.validate("short")
        assert False, "Should have raised validation error"
    except Exception as e:
        assert "Must not match." in str(e)
    
    # Test with different field types
    from typesystem.fields import Integer
    int_field = Integer(minimum=0)
    not_int = Not(negated=int_field)
    
    # Should accept negative numbers (don't match Integer with min=0)
    result = not_int.validate(-5)
    assert result == -5
    
    # Should reject positive numbers (match Integer with min=0)
    try:
        not_int.validate(10)
        assert False, "Should have raised validation error"
    except Exception as e:
        assert "Must not match." in str(e)


# LLM-generated content at query #40
#--------------------------

```python
def test_OneOf():
    # Test basic initialization
    field1 = Any()
    field2 = NeverMatch()
    one_of = OneOf(one_of=[field1, field2])
    assert one_of.one_of == [field1, field2]
    
    # Test that allow_null cannot be passed
    try:
        OneOf(one_of=[field1], allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test with empty list
    one_of_empty = OneOf(one_of=[])
    assert one_of_empty.one_of == []
    
    # Test validation with exactly one match
    result = one_of.validate("test")
    assert result == "test"
    
    # Test validation with no matches
    try:
        one_of.validate(None)
        assert False, "Should have raised validation error"
    except Exception as e:
        assert "no_match" in str(e)
    
    # Test validation with multiple matches (both Any fields)
    field3 = Any()
    one_of_multiple = OneOf(one_of=[field1, field3])
    try:
        one_of_multiple.validate("test")
        assert False, "Should have raised validation error"
    except Exception as e:
        assert "multiple_matches" in str(e)


# LLM-generated content at query #41
#--------------------------

```python
def test_IfThenElse():
    # Test basic initialization with all clauses provided
    if_field = Any()
    then_field = Any()
    else_field = Any()
    field = IfThenElse(if_field, then_field, else_field)
    
    assert field.if_clause == if_field
    assert field.then_clause == then_field
    assert field.else_clause == else_field
    
    # Test initialization with only if_clause (then and else should default to Any)
    field = IfThenElse(if_field)
    assert field.if_clause == if_field
    assert isinstance(field.then_clause, Any)
    assert isinstance(field.else_clause, Any)
    
    # Test initialization with if_clause and then_clause only
    field = IfThenElse(if_field, then_field)
    assert field.if_clause == if_field
    assert field.then_clause == then_field
    assert isinstance(field.else_clause, Any)
    
    # Test that allow_null cannot be passed in kwargs
    try:
        IfThenElse(if_field, allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test validation when if_clause validates successfully
    from typesystem.fields import String, Integer
    
    string_field = String()
    integer_field = Integer()
    if_then_else = IfThenElse(
        if_clause=String(),
        then_clause=Integer(),
        else_clause=String()
    )
    
    # If clause matches (string), should validate with then_clause (integer)
    # This should fail since "test" is not an integer
    try:
        if_then_else.validate("test")
        assert False, "Should have raised validation error"
    except Exception:
        pass
    
    # Test with proper values
    if_then_else = IfThenElse(
        if_clause=String(min_length=3),
        then_clause=String(max_length=5),
        else_clause=Integer()
    )
    
    # If clause matches (string with length >= 3), then_clause should validate
    result = if_then_else.validate("test")
    assert result == "test"
    
    # If clause doesn't match (string with length < 3), else_clause should validate
    try:
        if_then_else.validate("ab")
        assert False, "Should have raised validation error from else_clause"
    except Exception:
        pass
    
    # Test with integer in else_clause
    result = if_then_else.validate(123)
    assert result == 123


# LLM-generated content at query #42
#--------------------------

```python
def test_OneOf():
    # Test basic initialization
    field1 = Any()
    field2 = Any()
    one_of = OneOf(one_of=[field1, field2])
    assert one_of.one_of == [field1, field2]
    
    # Test that allow_null cannot be passed
    try:
        OneOf(one_of=[field1], allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test with empty list
    one_of_empty = OneOf(one_of=[])
    assert one_of_empty.one_of == []
    
    # Test with single field
    single_field = Any()
    one_of_single = OneOf(one_of=[single_field])
    assert one_of_single.one_of == [single_field]
    
    # Test with multiple fields
    fields = [Any() for _ in range(5)]
    one_of_multi = OneOf(one_of=fields)
    assert len(one_of_multi.one_of) == 5
    assert one_of_multi.one_of == fields


# LLM-generated content at query #43
#--------------------------

```python
def test_NeverMatch():
    field = NeverMatch()
    assert isinstance(field, NeverMatch)
    assert field.errors == {"never": "This never validates."}
    
    with pytest.raises(AssertionError):
        NeverMatch(allow_null=True)
    
    with pytest.raises(field.validation_error("never")):
        field.validate("any value")
    
    with pytest.raises(field.validation_error("never")):
        field.validate(None)
    
    with pytest.raises(field.validation_error("never")):
        field.validate(123)
    
    with pytest.raises(field.validation_error("never")):
        field.validate([])


# LLM-generated content at query #44
#--------------------------

```python
def test_AllOf():
    from typesystem.fields import Integer, String, Boolean

    # Test basic initialization
    integer_field = Integer()
    string_field = String()
    all_of = AllOf([integer_field, string_field])
    assert all_of.all_of == [integer_field, string_field]
    assert len(all_of.all_of) == 2

    # Test with single field
    single_field = AllOf([Integer()])
    assert len(single_field.all_of) == 1
    assert isinstance(single_field.all_of[0], Integer)

    # Test with multiple fields of different types
    bool_field = Boolean()
    multi_all_of = AllOf([integer_field, string_field, bool_field])
    assert len(multi_all_of.all_of) == 3
    assert isinstance(multi_all_of.all_of[0], Integer)
    assert isinstance(multi_all_of.all_of[1], String)
    assert isinstance(multi_all_of.all_of[2], Boolean)

    # Test that allow_null is not allowed in kwargs
    try:
        AllOf([integer_field], allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass

    # Test with empty list (edge case)
    empty_all_of = AllOf([])
    assert empty_all_of.all_of == []


# LLM-generated content at query #45
#--------------------------

```python
def test_OneOf():
    from typesystem.fields import String, Integer, Boolean
    
    # Test basic initialization
    string_field = String()
    integer_field = Integer()
    one_of = OneOf(one_of=[string_field, integer_field])
    
    assert one_of.one_of == [string_field, integer_field]
    assert one_of.errors == {
        "no_match": "Did not match any valid type.",
        "multiple_matches": "Matched more than one type."
    }
    
    # Test validation with single match
    result = one_of.validate("hello")
    assert result == "hello"
    
    result = one_of.validate(123)
    assert result == 123
    
    # Test validation with no match
    try:
        one_of.validate(True)
        assert False, "Should have raised validation error"
    except Exception as e:
        assert str(e) == "Did not match any valid type."
    
    # Test validation with multiple matches (when fields overlap)
    bool_field = Boolean()
    string_or_bool = OneOf(one_of=[string_field, bool_field])
    
    # Boolean True can be interpreted as string "True" by String field
    # and as boolean True by Boolean field, causing multiple matches
    try:
        string_or_bool.validate(True)
        assert False, "Should have raised multiple_matches error"
    except Exception as e:
        assert str(e) == "Matched more than one type."
    
    # Test with empty one_of list
    empty_one_of = OneOf(one_of=[])
    try:
        empty_one_of.validate("anything")
        assert False, "Should have raised validation error"
    except Exception as e:
        assert str(e) == "Did not match any valid type."
    
    # Test that allow_null is not allowed in constructor
    try:
        OneOf(one_of=[string_field], allow_null=True)
        assert False, "Should have raised assertion error"
    except AssertionError:
        pass


# LLM-generated content at query #46
#--------------------------

```python
def test_NeverMatch():
    field = NeverMatch()
    assert field.errors == {"never": "This never validates."}
    with pytest.raises(AssertionError):
        NeverMatch(allow_null=True)
    with pytest.raises(field.validation_error):
        field.validate("any value")
    with pytest.raises(field.validation_error):
        field.validate(None)
    with pytest.raises(field.validation_error):
        field.validate(123)
    with pytest.raises(field.validation_error):
        field.validate([])
    with pytest.raises(field.validation_error):
        field.validate({})


# LLM-generated content at query #47
#--------------------------

```python
def test_AllOf():
    from typesystem.fields import String, Integer, Boolean
    
    # Test basic initialization
    string_field = String()
    integer_field = Integer()
    all_of = AllOf([string_field, integer_field])
    assert all_of.all_of == [string_field, integer_field]
    
    # Test that allow_null is not allowed in kwargs
    try:
        AllOf([string_field], allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test validation with multiple fields
    all_of = AllOf([String(max_length=5), String(pattern="^[a-z]+$")])
    assert all_of.validate("abc") == "abc"
    
    # Test validation failure with first field
    try:
        all_of.validate("abcdef")
        assert False, "Should have raised ValidationError"
    except Exception as e:
        assert "max_length" in str(e)
    
    # Test validation failure with second field
    try:
        all_of.validate("123")
        assert False, "Should have raised ValidationError"
    except Exception as e:
        assert "pattern" in str(e)
    
    # Test validation with three fields
    all_of = AllOf([String(), Integer(), Boolean()])
    # Note: This would fail in practice since a value can't be all three types
    # This test shows the structure, but actual usage would need type conversion
    
    # Test empty all_of list
    all_of = AllOf([])
    assert all_of.validate("any_value") == "any_value"
    
    # Test inheritance from Field
    assert hasattr(all_of, 'errors')
    assert hasattr(all_of, 'validate_or_error')


# LLM-generated content at query #48
#--------------------------

```python
def test_OneOf():
    from typesystem.fields import String, Integer, Boolean
    
    # Test basic initialization
    field = OneOf(one_of=[String(), Integer()])
    assert field.one_of[0].__class__.__name__ == "String"
    assert field.one_of[1].__class__.__name__ == "Integer"
    
    # Test that allow_null is not allowed in kwargs
    try:
        OneOf(one_of=[String()], allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test validation with exactly one match
    field = OneOf(one_of=[String(), Integer()])
    result = field.validate("hello")
    assert result == "hello"
    
    result = field.validate(42)
    assert result == 42
    
    # Test validation with no matches
    try:
        field.validate(True)
        assert False, "Should have raised validation error"
    except Exception as e:
        assert "no_match" in str(e)
    
    # Test validation with multiple matches (using Any field)
    field = OneOf(one_of=[Any(), Any()])
    try:
        field.validate("anything")
        assert False, "Should have raised validation error"
    except Exception as e:
        assert "multiple_matches" in str(e)
    
    # Test with empty one_of list
    field = OneOf(one_of=[])
    try:
        field.validate("anything")
        assert False, "Should have raised validation error"
    except Exception as e:
        assert "no_match" in str(e)
    
    # Test with three fields where only one matches
    field = OneOf(one_of=[String(), Integer(), Boolean()])
    result = field.validate(True)
    assert result is True
    
    # Test error messages
    field = OneOf(one_of=[String(), Integer()])
    try:
        field.validate(3.14)
    except Exception as e:
        assert "no_match" in str(e)
    
    field = OneOf(one_of=[Any(), Any()])
    try:
        field.validate("test")
    except Exception as e:
        assert "multiple_matches" in str(e)


# LLM-generated content at query #49
#--------------------------

```python
def test_OneOf():
    from typesystem.fields import String, Integer, Boolean
    
    # Test basic initialization
    string_field = String()
    integer_field = Integer()
    one_of = OneOf(one_of=[string_field, integer_field])
    
    assert one_of.one_of == [string_field, integer_field]
    assert one_of.errors == {
        "no_match": "Did not match any valid type.",
        "multiple_matches": "Matched more than one type."
    }
    
    # Test that allow_null cannot be passed
    try:
        OneOf(one_of=[string_field], allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test with empty list
    one_of_empty = OneOf(one_of=[])
    assert one_of_empty.one_of == []
    
    # Test validation with single match
    result = one_of.validate("hello")
    assert result == "hello"
    
    result = one_of.validate(123)
    assert result == 123
    
    # Test validation with no match
    try:
        one_of.validate(True)
        assert False, "Should have raised validation error"
    except Exception as e:
        assert "no_match" in str(e)
    
    # Test validation with multiple matches (when fields overlap)
    bool_field = Boolean()
    string_or_bool = OneOf(one_of=[string_field, bool_field])
    
    # Boolean True can be interpreted as string "True" by String field
    # and as boolean True by Boolean field
    try:
        string_or_bool.validate(True)
        assert False, "Should have raised multiple_matches error"
    except Exception as e:
        assert "multiple_matches" in str(e)
    
    # Test with three fields
    one_of_three = OneOf(one_of=[string_field, integer_field, bool_field])
    assert len(one_of_three.one_of) == 3


# LLM-generated content at query #50
#--------------------------

```python
def test_NeverMatch():
    # Test that NeverMatch cannot be instantiated with allow_null
    with pytest.raises(AssertionError):
        NeverMatch(allow_null=True)
    
    # Test that NeverMatch can be instantiated without allow_null
    field = NeverMatch()
    assert isinstance(field, NeverMatch)
    assert isinstance(field, Field)
    
    # Test that validate always raises validation error
    with pytest.raises(field.validation_error("never")):
        field.validate(None)
    
    with pytest.raises(field.validation_error("never")):
        field.validate("any value")
    
    with pytest.raises(field.validation_error("never")):
        field.validate(123)
    
    with pytest.raises(field.validation_error("never")):
        field.validate({"key": "value"})
    
    # Test that errors attribute is set correctly
    assert field.errors == {"never": "This never validates."}


# LLM-generated content at query #51
#--------------------------

```python
def test_Not():
    # Test basic initialization
    negated_field = Any()
    not_field = Not(negated=negated_field)
    assert not_field.negated == negated_field
    assert not_field.errors == {"negated": "Must not match."}

    # Test that allow_null cannot be passed
    try:
        Not(negated=negated_field, allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass

    # Test validation when negated field fails
    negated_field = NeverMatch()
    not_field = Not(negated=negated_field)
    result = not_field.validate("any_value")
    assert result == "any_value"

    # Test validation when negated field succeeds
    negated_field = Any()
    not_field = Not(negated=negated_field)
    try:
        not_field.validate("any_value")
        assert False, "Should have raised validation error"
    except Exception as e:
        assert str(e) == "Must not match."

    # Test with custom error messages
    not_field = Not(negated=Any(), errors={"negated": "Custom error"})
    try:
        not_field.validate("test")
        assert False, "Should have raised validation error"
    except Exception as e:
        assert str(e) == "Custom error"


# LLM-generated content at query #52
#--------------------------

```python
def test_NeverMatch():
    field = NeverMatch()
    assert isinstance(field, NeverMatch)
    assert field.errors == {"never": "This never validates."}
    
    with pytest.raises(AssertionError):
        NeverMatch(allow_null=True)
    
    with pytest.raises(field.validation_error("never")):
        field.validate("any value")
    
    with pytest.raises(field.validation_error("never")):
        field.validate(None)
    
    with pytest.raises(field.validation_error("never")):
        field.validate(123)
    
    with pytest.raises(field.validation_error("never")):
        field.validate([])


# LLM-generated content at query #53
#--------------------------

```python
def test_Not():
    # Test basic Not functionality with a simple field
    from typesystem.fields import Integer
    
    integer_field = Integer()
    not_field = Not(negated=integer_field)
    
    # Should reject integer values
    with pytest.raises(not_field.validation_error("negated")):
        not_field.validate(42)
    
    # Should accept non-integer values
    assert not_field.validate("string") == "string"
    assert not_field.validate(True) == True
    assert not_field.validate(None) == None
    
    # Test with a more complex field
    from typesystem.fields import String
    
    string_field = String(max_length=5)
    not_string_field = Not(negated=string_field)
    
    # Should reject valid strings
    with pytest.raises(not_string_field.validation_error("negated")):
        not_string_field.validate("hello")
    
    # Should accept invalid strings (too long)
    assert not_string_field.validate("too long string") == "too long string"
    
    # Should accept non-string values
    assert not_string_field.validate(123) == 123
    
    # Test that allow_null is not allowed in constructor
    with pytest.raises(AssertionError):
        Not(negated=integer_field, allow_null=True)
    
    # Test with nested Not fields
    not_not_field = Not(negated=not_field)
    
    # Double negation should accept integers
    assert not_not_field.validate(42) == 42
    
    # Double negation should reject non-integers
    with pytest.raises(not_not_field.validation_error("negated")):
        not_not_field.validate("string")


# LLM-generated content at query #54
#--------------------------

```python
def test_NeverMatch():
    field = NeverMatch()
    assert isinstance(field, NeverMatch)
    assert field.errors == {"never": "This never validates."}
    
    with pytest.raises(AssertionError):
        NeverMatch(allow_null=True)
    
    with pytest.raises(field.validation_error):
        field.validate("any value")
    
    with pytest.raises(field.validation_error):
        field.validate(None)
    
    with pytest.raises(field.validation_error):
        field.validate(123)
    
    with pytest.raises(field.validation_error):
        field.validate([])


# LLM-generated content at query #55
#--------------------------

```python
def test_Not():
    from typesystem.fields import String, Integer
    
    # Test basic Not functionality
    string_field = String()
    not_string = Not(string_field)
    
    # Should accept non-string values
    assert not_string.validate(123) == 123
    assert not_string.validate(True) == True
    assert not_string.validate(None) == None
    
    # Should reject string values
    with pytest.raises(not_string.validation_error("negated")):
        not_string.validate("hello")
    with pytest.raises(not_string.validation_error("negated")):
        not_string.validate("")
    
    # Test with integer field
    integer_field = Integer()
    not_integer = Not(integer_field)
    
    # Should accept non-integer values
    assert not_integer.validate("string") == "string"
    assert not_integer.validate(3.14) == 3.14
    
    # Should reject integer values
    with pytest.raises(not_integer.validation_error("negated")):
        not_integer.validate(42)
    with pytest.raises(not_integer.validation_error("negated")):
        not_integer.validate(0)
    
    # Test that allow_null cannot be passed
    with pytest.raises(AssertionError):
        Not(String(), allow_null=True)
    
    # Test with complex field
    from typesystem.fields import Array
    array_field = Array(items=String())
    not_array = Not(array_field)
    
    # Should accept non-array values
    assert not_array.validate("single") == "single"
    assert not_array.validate(123) == 123
    
    # Should reject array values
    with pytest.raises(not_array.validation_error("negated")):
        not_array.validate(["item1", "item2"])


# LLM-generated content at query #56
#--------------------------

```python
def test_NeverMatch():
    field = NeverMatch()
    assert isinstance(field, Field)
    assert field.errors == {"never": "This never validates."}
    
    with pytest.raises(AssertionError):
        NeverMatch(allow_null=True)
    
    with pytest.raises(field.validation_error("never")):
        field.validate(None)
    
    with pytest.raises(field.validation_error("never")):
        field.validate("any value")
    
    with pytest.raises(field.validation_error("never")):
        field.validate(123)
    
    with pytest.raises(field.validation_error("never")):
        field.validate([])


# LLM-generated content at query #57
#--------------------------

```python
def test_NeverMatch():
    field = NeverMatch()
    assert isinstance(field, Field)
    assert field.errors == {"never": "This never validates."}
    
    with pytest.raises(AssertionError):
        NeverMatch(allow_null=True)
    
    with pytest.raises(field.validation_error("never")):
        field.validate(None)
    
    with pytest.raises(field.validation_error("never")):
        field.validate("any value")
    
    with pytest.raises(field.validation_error("never")):
        field.validate(123)
    
    with pytest.raises(field.validation_error("never")):
        field.validate([])


# LLM-generated content at query #58
#--------------------------

```python
def test_NeverMatch():
    # Test that NeverMatch cannot be initialized with allow_null
    with pytest.raises(AssertionError):
        NeverMatch(allow_null=True)
    
    # Test that NeverMatch can be initialized without allow_null
    field = NeverMatch()
    assert isinstance(field, NeverMatch)
    assert isinstance(field, Field)
    
    # Test that validate always raises validation error
    with pytest.raises(field.validation_error("never")):
        field.validate(None)
    
    with pytest.raises(field.validation_error("never")):
        field.validate("any value")
    
    with pytest.raises(field.validation_error("never")):
        field.validate(123)
    
    with pytest.raises(field.validation_error("never")):
        field.validate({"key": "value"})
    
    # Test error message
    try:
        field.validate("test")
    except Exception as e:
        assert str(e) == "This never validates."


# LLM-generated content at query #59
#--------------------------

```python
def test_NeverMatch():
    # Test that NeverMatch cannot be instantiated with allow_null
    try:
        NeverMatch(allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError as e:
        assert "allow_null" in str(e)

    # Test that NeverMatch can be instantiated without allow_null
    field = NeverMatch()
    assert isinstance(field, NeverMatch)
    assert isinstance(field, Field)

    # Test that validate always raises validation error
    try:
        field.validate(None)
        assert False, "Should have raised validation error"
    except Exception as e:
        assert hasattr(e, "code")
        assert e.code == "never"

    # Test with different values
    try:
        field.validate("test")
        assert False, "Should have raised validation error"
    except Exception as e:
        assert e.code == "never"

    try:
        field.validate(123)
        assert False, "Should have raised validation error"
    except Exception as e:
        assert e.code == "never"

    try:
        field.validate({"key": "value"})
        assert False, "Should have raised validation error"
    except Exception as e:
        assert e.code == "never"


# LLM-generated content at query #60
#--------------------------

```python
def test_AllOf():
    from typesystem.fields import String, Integer, Boolean
    
    # Test basic initialization
    string_field = String()
    integer_field = Integer()
    all_of = AllOf([string_field, integer_field])
    assert all_of.all_of == [string_field, integer_field]
    
    # Test that allow_null cannot be passed
    try:
        AllOf([string_field], allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test validation with multiple fields
    # Note: This will fail since a value can't be both string and integer
    all_of = AllOf([String(), Integer()])
    try:
        all_of.validate("test")
        assert False, "Should have raised validation error"
    except Exception:
        pass
    
    # Test validation with compatible fields
    # Using fields that all accept the same type
    from typesystem.fields import MinLength, MaxLength
    min_len = MinLength(3)
    max_len = MaxLength(10)
    all_of = AllOf([min_len, max_len])
    
    # Valid value that satisfies both constraints
    result = all_of.validate("test")
    assert result == "test"
    
    # Invalid value - too short
    try:
        all_of.validate("ab")
        assert False, "Should have raised validation error"
    except Exception:
        pass
    
    # Invalid value - too long
    try:
        all_of.validate("thisistoolong")
        assert False, "Should have raised validation error"
    except Exception:
        pass
    
    # Test with empty list
    all_of = AllOf([])
    result = all_of.validate("any value")
    assert result == "any value"
    
    # Test that parent class initialization works
    all_of = AllOf([string_field], description="Test description")
    assert all_of.description == "Test description"


# LLM-generated content at query #61
#--------------------------

```python
def test_IfThenElse():
    # Test with only if_clause
    field1 = IfThenElse(if_clause=Any())
    assert field1.if_clause is not None
    assert isinstance(field1.then_clause, Any)
    assert isinstance(field1.else_clause, Any)
    
    # Test with if_clause and then_clause
    from typesystem.fields import String
    field2 = IfThenElse(if_clause=String(), then_clause=String())
    assert isinstance(field2.if_clause, String)
    assert isinstance(field2.then_clause, String)
    assert isinstance(field2.else_clause, Any)
    
    # Test with if_clause, then_clause, and else_clause
    from typesystem.fields import Integer
    field3 = IfThenElse(if_clause=String(), then_clause=Integer(), else_clause=String())
    assert isinstance(field3.if_clause, String)
    assert isinstance(field3.then_clause, Integer)
    assert isinstance(field3.else_clause, String)
    
    # Test that allow_null cannot be passed
    try:
        IfThenElse(if_clause=Any(), allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test validation when if_clause matches
    value1 = "test"
    result1 = field2.validate(value1)
    assert result1 == value1
    
    # Test validation when if_clause doesn't match
    value2 = 123
    result2 = field2.validate(value2)
    assert result2 == value2
    
    # Test validation with full if-then-else
    field4 = IfThenElse(
        if_clause=String(min_length=5),
        then_clause=String(max_length=10),
        else_clause=Integer()
    )
    
    # if_clause matches (string with length >= 5)
    result3 = field4.validate("hello")
    assert result3 == "hello"
    
    # if_clause doesn't match (string with length < 5)
    result4 = field4.validate("hi")
    assert result4 == "hi"
    
    # if_clause doesn't match (not a string)
    result5 = field4.validate(42)
    assert result5 == 42


# LLM-generated content at query #62
#--------------------------

```python
def test_Not():
    # Test basic initialization with a negated field
    negated_field = Any()
    not_field = Not(negated=negated_field)
    assert not_field.negated == negated_field
    assert not_field.errors == {"negated": "Must not match."}

    # Test that allow_null cannot be passed in kwargs
    try:
        Not(negated=Any(), allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass

    # Test validation when negated field fails (should succeed)
    negated_string = Field(type="string")
    not_string = Not(negated=negated_string)
    
    # Integer should pass since it fails the string validation
    result = not_string.validate(123)
    assert result == 123
    
    # Test validation when negated field succeeds (should fail)
    try:
        not_string.validate("hello")
        assert False, "Should have raised ValidationError"
    except Exception as e:
        assert "negated" in str(e)

    # Test with custom negated field
    class AlwaysFail(Field):
        def validate(self, value):
            raise self.validation_error("always_fails")
    
    not_always_fail = Not(negated=AlwaysFail())
    # Should always pass since negated field always fails
    result = not_always_fail.validate("anything")
    assert result == "anything"

    # Test with custom negated field that always passes
    class AlwaysPass(Field):
        def validate(self, value):
            return value
    
    not_always_pass = Not(negated=AlwaysPass())
    # Should always fail since negated field always passes
    try:
        not_always_pass.validate("anything")
        assert False, "Should have raised ValidationError"
    except Exception as e:
        assert "negated" in str(e)


# LLM-generated content at query #63
#--------------------------

```python
def test_Not():
    # Test basic initialization
    negated_field = Any()
    not_field = Not(negated=negated_field)
    assert not_field.negated == negated_field
    assert not_field.errors == {"negated": "Must not match."}
    
    # Test that allow_null cannot be passed
    try:
        Not(negated=negated_field, allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test validation when negated field fails (should succeed)
    negated_field = NeverMatch()
    not_field = Not(negated=negated_field)
    result = not_field.validate("any value")
    assert result == "any value"
    
    # Test validation when negated field succeeds (should fail)
    negated_field = Any()
    not_field = Not(negated=negated_field)
    try:
        not_field.validate("any value")
        assert False, "Should have raised ValidationError"
    except Exception as e:
        assert str(e) == "Must not match."
    
    # Test with more complex negated field
    from typesystem import Integer
    negated_field = Integer(minimum=10)
    not_field = Not(negated=negated_field)
    
    # Value that fails negated field (less than 10) should succeed
    result = not_field.validate(5)
    assert result == 5
    
    # Value that passes negated field (>= 10) should fail
    try:
        not_field.validate(15)
        assert False, "Should have raised ValidationError"
    except Exception as e:
        assert str(e) == "Must not match."


# LLM-generated content at query #64
#--------------------------

```python
def test_AllOf():
    from typesystem.fields import Integer, String, Boolean
    
    # Test basic initialization
    int_field = Integer()
    str_field = String()
    all_of = AllOf([int_field, str_field])
    assert all_of.all_of == [int_field, str_field]
    
    # Test that allow_null cannot be passed
    try:
        AllOf([int_field], allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test with empty list
    all_of_empty = AllOf([])
    assert all_of_empty.all_of == []
    
    # Test with multiple fields
    bool_field = Boolean()
    all_of_multi = AllOf([int_field, str_field, bool_field])
    assert len(all_of_multi.all_of) == 3
    assert all_of_multi.all_of[0] is int_field
    assert all_of_multi.all_of[1] is str_field
    assert all_of_multi.all_of[2] is bool_field
    
    # Test that fields are stored as references, not copies
    assert all_of_multi.all_of[0] is int_field
    
    # Test validation passes through to constructor
    all_of_with_title = AllOf([int_field], title="Test AllOf")
    assert all_of_with_title.title == "Test AllOf"


# LLM-generated content at query #65
#--------------------------

```python
def test_NeverMatch():
    field = NeverMatch()
    assert field.errors == {"never": "This never validates."}
    with pytest.raises(AssertionError):
        NeverMatch(allow_null=True)
    with pytest.raises(field.validation_error):
        field.validate("any value")
    with pytest.raises(field.validation_error):
        field.validate(None)
    with pytest.raises(field.validation_error):
        field.validate(123)
    with pytest.raises(field.validation_error):
        field.validate([])


# LLM-generated content at query #66
#--------------------------

```python
def test_Not():
    from typesystem.fields import String, Integer
    
    # Test basic initialization
    string_field = String()
    not_field = Not(string_field)
    assert not_field.negated == string_field
    assert not_field.errors == {"negated": "Must not match."}
    
    # Test that allow_null cannot be passed
    try:
        Not(string_field, allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test validation with non-matching value
    integer_field = Integer()
    not_integer = Not(integer_field)
    result = not_integer.validate("hello")
    assert result == "hello"
    
    # Test validation with matching value (should raise error)
    try:
        not_integer.validate(123)
        assert False, "Should have raised validation error"
    except Exception as exc:
        assert "negated" in str(exc)
    
    # Test with nested field
    from typesystem.fields import Boolean
    bool_field = Boolean()
    not_bool = Not(bool_field)
    
    # Boolean field would accept True/False, so Not should reject them
    try:
        not_bool.validate(True)
        assert False, "Should have raised validation error for True"
    except Exception:
        pass
    
    try:
        not_bool.validate(False)
        assert False, "Should have raised validation error for False"
    except Exception:
        pass
    
    # Non-boolean values should pass
    assert not_bool.validate("string") == "string"
    assert not_bool.validate(123) == 123
    assert not_bool.validate(None) == None
    
    # Test with complex field
    from typesystem.fields import Array
    array_field = Array(items=String())
    not_array = Not(array_field)
    
    # Array should pass non-array values
    assert not_array.validate("not an array") == "not an array"
    
    # Array should reject actual arrays
    try:
        not_array.validate(["item1", "item2"])
        assert False, "Should have raised validation error for array"
    except Exception:
        pass


# LLM-generated content at query #67
#--------------------------

```python
def test_NeverMatch():
    # Test that NeverMatch cannot be instantiated with allow_null
    with pytest.raises(AssertionError):
        NeverMatch(allow_null=True)
    
    # Test that NeverMatch can be instantiated without allow_null
    field = NeverMatch()
    assert isinstance(field, NeverMatch)
    assert isinstance(field, Field)
    
    # Test that validate always raises validation error
    with pytest.raises(field.validation_error) as exc_info:
        field.validate("any value")
    assert str(exc_info.value) == "This never validates."
    
    # Test with different types of values
    with pytest.raises(field.validation_error):
        field.validate(None)
    
    with pytest.raises(field.validation_error):
        field.validate(123)
    
    with pytest.raises(field.validation_error):
        field.validate({"key": "value"})
    
    with pytest.raises(field.validation_error):
        field.validate([1, 2, 3])


# LLM-generated content at query #68
#--------------------------

```python
def test_OneOf():
    from typesystem.fields import Integer, String, Boolean

    # Test basic initialization
    field = OneOf(one_of=[Integer(), String()])
    assert field.one_of[0].__class__.__name__ == "Integer"
    assert field.one_of[1].__class__.__name__ == "String"

    # Test validation with single match
    result = field.validate(42)
    assert result == 42

    result = field.validate("hello")
    assert result == "hello"

    # Test validation with no match
    try:
        field.validate(True)
        assert False, "Should have raised validation error"
    except Exception as e:
        assert "no_match" in str(e)

    # Test validation with multiple matches
    field2 = OneOf(one_of=[Integer(minimum=0), Integer(maximum=100)])
    try:
        field2.validate(50)
        assert False, "Should have raised validation error"
    except Exception as e:
        assert "multiple_matches" in str(e)

    # Test with empty list
    field3 = OneOf(one_of=[])
    try:
        field3.validate("anything")
        assert False, "Should have raised validation error"
    except Exception as e:
        assert "no_match" in str(e)

    # Test that allow_null is not allowed in kwargs
    try:
        OneOf(one_of=[Integer()], allow_null=True)
        assert False, "Should have raised assertion error"
    except AssertionError:
        pass


# LLM-generated content at query #69
#--------------------------

```python
def test_AllOf():
    # Test basic initialization
    field1 = Any()
    field2 = Any()
    all_of = AllOf([field1, field2])
    assert all_of.all_of == [field1, field2]
    
    # Test that allow_null cannot be passed
    try:
        AllOf([field1], allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test with empty list
    all_of_empty = AllOf([])
    assert all_of_empty.all_of == []
    
    # Test validation passes when all fields validate
    class AlwaysPass(Field):
        def validate(self, value):
            return value
    
    pass1 = AlwaysPass()
    pass2 = AlwaysPass()
    all_of_pass = AllOf([pass1, pass2])
    result = all_of_pass.validate("test_value")
    assert result == "test_value"
    
    # Test validation fails when any field fails
    class AlwaysFail(Field):
        def validate(self, value):
            raise self.validation_error("test_error")
    
    pass_field = AlwaysPass()
    fail_field = AlwaysFail()
    all_of_mixed = AllOf([pass_field, fail_field])
    
    try:
        all_of_mixed.validate("test_value")
        assert False, "Should have raised ValidationError"
    except Exception as e:
        assert hasattr(e, 'code')
        assert e.code == "test_error"
    
    # Test validation order - all fields should be validated
    validation_order = []
    
    class OrderTracker(Field):
        def __init__(self, name):
            super().__init__()
            self.name = name
        
        def validate(self, value):
            validation_order.append(self.name)
            return value
    
    tracker1 = OrderTracker("first")
    tracker2 = OrderTracker("second")
    tracker3 = OrderTracker("third")
    
    all_of_tracked = AllOf([tracker1, tracker2, tracker3])
    all_of_tracked.validate("test")
    
    assert validation_order == ["first", "second", "third"]


# LLM-generated content at query #70
#--------------------------

```python
def test_AllOf():
    # Test basic initialization
    field1 = Any()
    field2 = Any()
    all_of = AllOf([field1, field2])
    assert all_of.all_of == [field1, field2]
    
    # Test that allow_null cannot be passed
    try:
        AllOf([field1], allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test with empty list
    all_of_empty = AllOf([])
    assert all_of_empty.all_of == []
    
    # Test validation passes when all fields validate
    class AlwaysValid(Field):
        def validate(self, value):
            return value
    
    valid_field1 = AlwaysValid()
    valid_field2 = AlwaysValid()
    all_of_valid = AllOf([valid_field1, valid_field2])
    result = all_of_valid.validate("test_value")
    assert result == "test_value"
    
    # Test validation fails when any field fails
    class FailingField(Field):
        def validate(self, value):
            raise self.validation_error("test_error")
    
    failing_field = FailingField()
    all_of_mixed = AllOf([valid_field1, failing_field])
    try:
        all_of_mixed.validate("test_value")
        assert False, "Should have raised ValidationError"
    except Exception as e:
        assert hasattr(e, 'code')


# LLM-generated content at query #71
#--------------------------

```python
def test_AllOf():
    from typesystem.fields import Integer, String
    
    # Test basic initialization
    integer_field = Integer()
    string_field = String()
    all_of = AllOf([integer_field, string_field])
    
    assert all_of.all_of == [integer_field, string_field]
    assert all_of.allow_null is False
    
    # Test that allow_null cannot be set
    try:
        AllOf([integer_field], allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test with empty list
    all_of_empty = AllOf([])
    assert all_of_empty.all_of == []
    
    # Test validation with multiple fields
    all_of = AllOf([Integer(minimum=0), Integer(maximum=10)])
    
    # Should validate successfully
    result = all_of.validate(5)
    assert result == 5
    
    # Should fail validation (fails first constraint)
    try:
        all_of.validate(-5)
        assert False, "Should have raised ValidationError"
    except Exception as e:
        assert "minimum" in str(e).lower()
    
    # Should fail validation (fails second constraint)
    try:
        all_of.validate(15)
        assert False, "Should have raised ValidationError"
    except Exception as e:
        assert "maximum" in str(e).lower()
    
    # Test with different field types
    all_of_mixed = AllOf([Integer(), String(max_length=5)])
    
    # Integer validation should work
    result = all_of_mixed.validate(42)
    assert result == 42
    
    # String validation should work
    result = all_of_mixed.validate("test")
    assert result == "test"
    
    # String too long should fail
    try:
        all_of_mixed.validate("toolongstring")
        assert False, "Should have raised ValidationError"
    except Exception as e:
        assert "maximum length" in str(e).lower()


# LLM-generated content at query #72
#--------------------------

```python
def test_Not():
    # Test that Not raises assertion error when allow_null is passed
    try:
        Not(negated=Any(), allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass

    # Test that Not correctly validates when negated field fails
    negated_field = Any()
    not_field = Not(negated=negated_field)
    result = not_field.validate("test_value")
    assert result == "test_value"

    # Test that Not raises validation error when negated field succeeds
    negated_field = Any()
    not_field = Not(negated=negated_field)
    try:
        not_field.validate("test_value")
    except Exception as e:
        assert str(e) == "Must not match."

    # Test with a more specific negated field
    from typesystem import Integer
    not_integer = Not(negated=Integer())
    
    # String should pass (not an integer)
    result = not_integer.validate("not_a_number")
    assert result == "not_a_number"
    
    # Integer should fail
    try:
        not_integer.validate(42)
        assert False, "Should have raised validation error"
    except Exception as e:
        assert str(e) == "Must not match."

    # Test that negated field is stored correctly
    original_field = Any()
    not_field = Not(negated=original_field)
    assert not_field.negated is original_field


# LLM-generated content at query #73
#--------------------------

```python
def test_AllOf():
    # Test basic initialization
    field1 = Any()
    field2 = Any()
    all_of = AllOf([field1, field2])
    assert all_of.all_of == [field1, field2]
    
    # Test that allow_null cannot be passed
    try:
        AllOf([field1], allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test validation with multiple fields
    string_field = Field(str)
    int_field = Field(int)
    all_of_mixed = AllOf([string_field, int_field])
    
    # Test validation passes when all fields pass
    # Note: This will fail because string_field and int_field have conflicting types
    # This test demonstrates the expected behavior
    try:
        all_of_mixed.validate("test")
        assert False, "Should have raised validation error"
    except Exception:
        pass
    
    # Test with empty list
    all_of_empty = AllOf([])
    assert all_of_empty.all_of == []
    assert all_of_empty.validate("anything") == "anything"
    
    # Test that validate returns the original value when all pass
    any_field1 = Any()
    any_field2 = Any()
    all_of_any = AllOf([any_field1, any_field2])
    test_value = {"key": "value"}
    result = all_of_any.validate(test_value)
    assert result == test_value


# LLM-generated content at query #74
#--------------------------

```python
def test_AllOf():
    from typesystem.fields import Integer, String
    
    # Test basic initialization
    int_field = Integer()
    str_field = String()
    all_of = AllOf([int_field, str_field])
    assert all_of.all_of == [int_field, str_field]
    
    # Test that allow_null cannot be passed
    try:
        AllOf([int_field], allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test with empty list
    all_of_empty = AllOf([])
    assert all_of_empty.all_of == []
    
    # Test validation passes when all fields validate
    all_of = AllOf([Integer(minimum=0), Integer(maximum=10)])
    result = all_of.validate(5)
    assert result == 5
    
    # Test validation fails when any field fails
    all_of = AllOf([Integer(minimum=0), Integer(maximum=5)])
    try:
        all_of.validate(10)
        assert False, "Should have raised ValidationError"
    except Exception as e:
        assert hasattr(e, 'code')
    
    # Test validation with multiple field types
    class CustomField(Field):
        def validate(self, value):
            if value != "custom":
                raise self.validation_error("invalid")
            return value
    
    custom_field = CustomField()
    all_of = AllOf([String(min_length=1), custom_field])
    result = all_of.validate("custom")
    assert result == "custom"


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_OneOf():
    from typesystem.fields import Integer, String, Boolean
    
    # Test basic initialization
    integer_field = Integer()
    string_field = String()
    one_of = OneOf(one_of=[integer_field, string_field])
    
    assert one_of.one_of == [integer_field, string_field]
    assert one_of.errors == {
        "no_match": "Did not match any valid type.",
        "multiple_matches": "Matched more than one type."
    }
    
    # Test with single field
    single_one_of = OneOf(one_of=[integer_field])
    assert single_one_of.one_of == [integer_field]
    
    # Test with empty list (edge case)
    empty_one_of = OneOf(one_of=[])
    assert empty_one_of.one_of == []
    
    # Test that allow_null cannot be passed
    try:
        OneOf(one_of=[integer_field], allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test with three fields
    boolean_field = Boolean()
    three_one_of = OneOf(one_of=[integer_field, string_field, boolean_field])
    assert len(three_one_of.one_of) == 3
    assert three_one_of.one_of[0] == integer_field
    assert three_one_of.one_of[1] == string_field
    assert three_one_of.one_of[2] == boolean_field
    
    # Test that parent constructor is called with kwargs
    custom_one_of = OneOf(one_of=[integer_field], title="Test Field", description="Test description")
    assert custom_one_of.title == "Test Field"
    assert custom_one_of.description == "Test description"


# LLM-generated content at query #2
#--------------------------

```python
def test_AllOf():
    from typesystem.fields import Integer, String
    
    # Test basic initialization
    integer_field = Integer()
    string_field = String()
    all_of = AllOf([integer_field, string_field])
    
    assert all_of.all_of == [integer_field, string_field]
    assert all_of.allow_null is False
    
    # Test that allow_null cannot be passed
    try:
        AllOf([integer_field], allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test validation with multiple fields
    all_of = AllOf([Integer(minimum=5), Integer(maximum=10)])
    
    # Should validate successfully
    result = all_of.validate(7)
    assert result == 7
    
    # Should fail validation (fails first constraint)
    try:
        all_of.validate(3)
        assert False, "Should have raised ValidationError"
    except Exception as e:
        assert hasattr(e, 'code')
    
    # Should fail validation (fails second constraint)
    try:
        all_of.validate(12)
        assert False, "Should have raised ValidationError"
    except Exception as e:
        assert hasattr(e, 'code')
    
    # Test with different field types
    all_of = AllOf([Integer(), String(max_length=5)])
    
    # Should fail validation (value is integer but not string)
    try:
        all_of.validate(42)
        assert False, "Should have raised ValidationError"
    except Exception as e:
        assert hasattr(e, 'code')
    
    # Should fail validation (value is string but not integer)
    try:
        all_of.validate("hello")
        assert False, "Should have raised ValidationError"
    except Exception as e:
        assert hasattr(e, 'code')
    
    # Test with single field
    all_of = AllOf([Integer()])
    result = all_of.validate(42)
    assert result == 42
    
    # Test with empty list (should validate any value)
    all_of = AllOf([])
    result = all_of.validate("anything")
    assert result == "anything"
    
    # Test that original value is returned
    all_of = AllOf([Integer(), Integer()])
    value = 100
    result = all_of.validate(value)
    assert result is value


# LLM-generated content at query #3
#--------------------------

```python
def test_NeverMatch():
    field = NeverMatch()
    assert isinstance(field, Field)
    assert field.errors == {"never": "This never validates."}
    
    with pytest.raises(AssertionError):
        NeverMatch(allow_null=True)
    
    with pytest.raises(AssertionError):
        NeverMatch(allow_null=False)
    
    with pytest.raises(field.validation_error):
        field.validate(None)
    
    with pytest.raises(field.validation_error):
        field.validate("test")
    
    with pytest.raises(field.validation_error):
        field.validate(123)
    
    with pytest.raises(field.validation_error):
        field.validate([])


# LLM-generated content at query #4
#--------------------------

```python
def test_Not():
    # Test basic initialization
    negated_field = Any()
    not_field = Not(negated=negated_field)
    assert not_field.negated == negated_field
    assert not_field.errors == {"negated": "Must not match."}

    # Test that allow_null cannot be passed
    try:
        Not(negated=negated_field, allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass

    # Test validation with non-matching value
    string_field = Field(type="string")
    not_string = Not(negated=string_field)
    result = not_string.validate(123)
    assert result == 123

    # Test validation with matching value (should raise error)
    try:
        not_string.validate("hello")
        assert False, "Should have raised validation error"
    except Exception as e:
        assert "negated" in str(e)

    # Test with nested field
    number_field = Field(type="number")
    not_number = Not(negated=number_field)
    result = not_number.validate("text")
    assert result == "text"

    # Test with value that matches negated field
    try:
        not_number.validate(42)
        assert False, "Should have raised validation error"
    except Exception as e:
        assert "negated" in str(e)


# LLM-generated content at query #5
#--------------------------

```python
def test_NeverMatch():
    field = NeverMatch()
    assert isinstance(field, Field)
    assert field.errors == {"never": "This never validates."}
    
    with pytest.raises(AssertionError):
        NeverMatch(allow_null=True)
    
    with pytest.raises(field.validation_error("never")):
        field.validate("any value")
    
    with pytest.raises(field.validation_error("never")):
        field.validate(None)
    
    with pytest.raises(field.validation_error("never")):
        field.validate(123)
    
    with pytest.raises(field.validation_error("never")):
        field.validate([])


# LLM-generated content at query #6
#--------------------------

```python
def test_NeverMatch():
    field = NeverMatch()
    assert isinstance(field, NeverMatch)
    assert field.errors == {"never": "This never validates."}
    
    with pytest.raises(AssertionError):
        NeverMatch(allow_null=True)
    
    with pytest.raises(AssertionError):
        NeverMatch(allow_null=False)
    
    with pytest.raises(field.validation_error):
        field.validate(None)
    
    with pytest.raises(field.validation_error):
        field.validate("test")
    
    with pytest.raises(field.validation_error):
        field.validate(123)
    
    with pytest.raises(field.validation_error):
        field.validate({"key": "value"})


# LLM-generated content at query #7
#--------------------------

```python
def test_IfThenElse():
    from typesystem.fields import String, Integer, Boolean, Any
    
    # Test basic initialization with all clauses provided
    if_field = Boolean()
    then_field = String()
    else_field = Integer()
    field = IfThenElse(if_field, then_field, else_field)
    
    assert field.if_clause == if_field
    assert field.then_clause == then_field
    assert field.else_clause == else_field
    
    # Test initialization with only if_clause (then and else should default to Any)
    field = IfThenElse(if_field)
    assert field.if_clause == if_field
    assert isinstance(field.then_clause, Any)
    assert isinstance(field.else_clause, Any)
    
    # Test initialization with if_clause and then_clause only
    field = IfThenElse(if_field, then_field)
    assert field.if_clause == if_field
    assert field.then_clause == then_field
    assert isinstance(field.else_clause, Any)
    
    # Test that allow_null cannot be passed in kwargs
    try:
        IfThenElse(if_field, allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError as e:
        assert "allow_null" in str(e)
    
    # Test validation when if_clause matches
    if_field = Integer(minimum=0)
    then_field = String(min_length=3)
    else_field = String(max_length=2)
    field = IfThenElse(if_field, then_field, else_field)
    
    # Value matches if_clause (positive integer), should validate with then_clause
    result = field.validate(5)
    assert result == "5"
    
    # Value doesn't match if_clause (negative integer), should validate with else_clause
    result = field.validate(-5)
    assert result == "-5"
    
    # Test with nested validation errors
    if_field = Boolean()
    then_field = Integer(minimum=10)
    else_field = Integer(maximum=0)
    field = IfThenElse(if_field, then_field, else_field)
    
    # True matches if_clause, but 5 fails then_clause validation
    try:
        field.validate(True)
        assert False, "Should have raised ValidationError"
    except Exception as e:
        assert "minimum" in str(e)
    
    # False doesn't match if_clause, but 5 fails else_clause validation
    try:
        field.validate(False)
        assert False, "Should have raised ValidationError"
    except Exception as e:
        assert "maximum" in str(e)


# LLM-generated content at query #8
#--------------------------

```python
def test_OneOf():
    # Test basic initialization
    field1 = Any()
    field2 = NeverMatch()
    one_of = OneOf(one_of=[field1, field2])
    assert one_of.one_of == [field1, field2]
    
    # Test that allow_null cannot be passed
    try:
        OneOf(one_of=[], allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test with empty list
    one_of_empty = OneOf(one_of=[])
    assert one_of_empty.one_of == []
    
    # Test with single field
    single_field = Any()
    one_of_single = OneOf(one_of=[single_field])
    assert one_of_single.one_of == [single_field]
    
    # Test that other kwargs are passed to parent
    one_of_kwargs = OneOf(one_of=[], title="Test", description="Test field")
    assert one_of_kwargs.title == "Test"
    assert one_of_kwargs.description == "Test field"


# LLM-generated content at query #9
#--------------------------

```python
def test_AllOf():
    # Test basic initialization
    field1 = Any()
    field2 = Any()
    all_of = AllOf([field1, field2])
    assert all_of.all_of == [field1, field2]
    
    # Test that allow_null cannot be passed
    try:
        AllOf([field1], allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test with empty list
    all_of_empty = AllOf([])
    assert all_of_empty.all_of == []
    
    # Test validation passes when all fields validate
    class AlwaysValid(Field):
        def validate(self, value):
            return value
    
    valid1 = AlwaysValid()
    valid2 = AlwaysValid()
    all_of_valid = AllOf([valid1, valid2])
    result = all_of_valid.validate("test_value")
    assert result == "test_value"
    
    # Test validation fails when any field fails
    class FailingField(Field):
        def validate(self, value):
            raise self.validation_error("test_error")
    
    failing = FailingField()
    all_of_mixed = AllOf([valid1, failing])
    try:
        all_of_mixed.validate("test_value")
        assert False, "Should have raised ValidationError"
    except Exception as e:
        assert hasattr(e, 'code')
    
    # Test validation order - all fields should be validated
    validation_order = []
    
    class TrackingField(Field):
        def __init__(self, name):
            super().__init__()
            self.name = name
        
        def validate(self, value):
            validation_order.append(self.name)
            return value
    
    tracker1 = TrackingField("first")
    tracker2 = TrackingField("second")
    tracker3 = TrackingField("third")
    
    all_of_tracked = AllOf([tracker1, tracker2, tracker3])
    all_of_tracked.validate("test")
    assert validation_order == ["first", "second", "third"]


# LLM-generated content at query #10
#--------------------------

```python
def test_AllOf():
    # Test basic initialization
    field1 = Any()
    field2 = Any()
    all_of = AllOf([field1, field2])
    assert all_of.all_of == [field1, field2]
    
    # Test that allow_null cannot be passed
    try:
        AllOf([field1], allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test validation with multiple fields
    from typesystem.fields import Integer, String
    
    int_field = Integer()
    positive_field = Integer(minimum=0)
    all_of = AllOf([int_field, positive_field])
    
    # Valid case: matches all fields
    result = all_of.validate(5)
    assert result == 5
    
    # Invalid case: fails one field
    try:
        all_of.validate(-1)
        assert False, "Should have raised ValidationError"
    except Exception as e:
        assert "Must be greater than or equal to 0" in str(e)
    
    # Invalid case: fails another field
    try:
        all_of.validate("not a number")
        assert False, "Should have raised ValidationError"
    except Exception as e:
        assert "Must be a number" in str(e)
    
    # Test with three fields
    string_field = String(max_length=5)
    all_of = AllOf([int_field, positive_field, string_field])
    
    # This should fail because it's not a string
    try:
        all_of.validate(3)
        assert False, "Should have raised ValidationError"
    except Exception as e:
        assert "Must be a string" in str(e)
    
    # Test that original value is returned when all validations pass
    value = {"complex": "object"}
    all_of = AllOf([Any(), Any()])
    result = all_of.validate(value)
    assert result is value


# LLM-generated content at query #11
#--------------------------

```python
def test_NeverMatch():
    # Test that NeverMatch cannot be initialized with allow_null
    try:
        NeverMatch(allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass

    # Test that NeverMatch can be initialized without allow_null
    field = NeverMatch()
    assert isinstance(field, NeverMatch)
    assert field.errors == {"never": "This never validates."}

    # Test that validate always raises validation error
    try:
        field.validate(None)
        assert False, "Should have raised validation error"
    except Exception as e:
        assert str(e) == "This never validates."

    # Test with different values
    try:
        field.validate("test")
        assert False, "Should have raised validation error"
    except Exception as e:
        assert str(e) == "This never validates."

    try:
        field.validate(123)
        assert False, "Should have raised validation error"
    except Exception as e:
        assert str(e) == "This never validates."

    try:
        field.validate({"key": "value"})
        assert False, "Should have raised validation error"
    except Exception as e:
        assert str(e) == "This never validates."


# LLM-generated content at query #12
#--------------------------

```python
def test_NeverMatch():
    # Test that NeverMatch cannot be initialized with allow_null
    try:
        NeverMatch(allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass

    # Test that NeverMatch can be initialized without allow_null
    field = NeverMatch()
    assert isinstance(field, NeverMatch)
    assert field.errors == {"never": "This never validates."}

    # Test that validate always raises validation error
    try:
        field.validate(None)
        assert False, "Should have raised validation error"
    except Exception as e:
        assert str(e) == "This never validates."

    # Test with different values
    try:
        field.validate("test")
        assert False, "Should have raised validation error"
    except Exception as e:
        assert str(e) == "This never validates."

    try:
        field.validate(123)
        assert False, "Should have raised validation error"
    except Exception as e:
        assert str(e) == "This never validates."

    try:
        field.validate({"key": "value"})
        assert False, "Should have raised validation error"
    except Exception as e:
        assert str(e) == "This never validates."


# LLM-generated content at query #13
#--------------------------

```python
def test_Not():
    # Test basic initialization with a field
    from typesystem.fields import Integer
    integer_field = Integer()
    not_field = Not(negated=integer_field)
    assert not_field.negated == integer_field
    assert not_field.errors == {"negated": "Must not match."}
    
    # Test that allow_null cannot be passed in kwargs
    import pytest
    with pytest.raises(AssertionError):
        Not(negated=integer_field, allow_null=True)
    
    # Test validation with non-matching value
    result = not_field.validate("not an integer")
    assert result == "not an integer"
    
    # Test validation with matching value (should raise error)
    with pytest.raises(not_field.validation_error("negated")):
        not_field.validate(42)
    
    # Test with different field types
    from typesystem.fields import String
    string_field = String(min_length=5)
    not_string_field = Not(negated=string_field)
    
    # Should pass for strings shorter than 5 characters
    result = not_string_field.validate("abc")
    assert result == "abc"
    
    # Should fail for strings with 5 or more characters
    with pytest.raises(not_string_field.validation_error("negated")):
        not_string_field.validate("abcde")
    
    # Test with nested Not fields
    not_not_field = Not(negated=not_field)
    # This should now pass for integers (since we're negating the negation)
    result = not_not_field.validate(42)
    assert result == 42


# LLM-generated content at query #14
#--------------------------

```python
def test_NeverMatch():
    field = NeverMatch()
    assert isinstance(field, NeverMatch)
    assert isinstance(field, Field)
    assert field.errors == {"never": "This never validates."}

    with pytest.raises(AssertionError):
        NeverMatch(allow_null=True)

    with pytest.raises(AssertionError):
        NeverMatch(allow_null=False)

    try:
        field.validate("any value")
        assert False, "Should have raised validation error"
    except Exception as e:
        assert isinstance(e, field.validation_error("never").__class__)


# LLM-generated content at query #15
#--------------------------

```python
def test_AllOf():
    from typesystem.fields import Integer, String, Boolean
    
    # Test basic initialization
    int_field = Integer()
    str_field = String()
    all_of = AllOf([int_field, str_field])
    assert all_of.all_of == [int_field, str_field]
    
    # Test that allow_null cannot be passed
    try:
        AllOf([int_field], allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test with empty list
    all_of_empty = AllOf([])
    assert all_of_empty.all_of == []
    
    # Test with single field
    single_field = AllOf([int_field])
    assert single_field.all_of == [int_field]
    
    # Test with multiple fields
    bool_field = Boolean()
    all_of_multi = AllOf([int_field, str_field, bool_field])
    assert len(all_of_multi.all_of) == 3
    assert all_of_multi.all_of[0] == int_field
    assert all_of_multi.all_of[1] == str_field
    assert all_of_multi.all_of[2] == bool_field


# LLM-generated content at query #16
#--------------------------

```python
def test_AllOf():
    from typesystem.fields import String, Integer, Boolean
    
    # Test basic initialization
    string_field = String()
    integer_field = Integer()
    all_of = AllOf([string_field, integer_field])
    
    assert all_of.all_of == [string_field, integer_field]
    assert all_of.allow_null is False
    
    # Test that allow_null cannot be set in kwargs
    try:
        AllOf([string_field], allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test with empty list
    all_of_empty = AllOf([])
    assert all_of_empty.all_of == []
    
    # Test with single field
    single_field = AllOf([Boolean()])
    assert len(single_field.all_of) == 1
    assert isinstance(single_field.all_of[0], Boolean)
    
    # Test validation passes when all fields validate
    all_of_simple = AllOf([String(max_length=5), String(min_length=2)])
    result = all_of_simple.validate("test")
    assert result == "test"
    
    # Test validation fails when any field fails
    try:
        all_of_simple.validate("toolongstring")
        assert False, "Should have raised ValidationError"
    except Exception as e:
        assert "Must have no more than 5 characters" in str(e)
    
    try:
        all_of_simple.validate("a")
        assert False, "Should have raised ValidationError"
    except Exception as e:
        assert "Must have at least 2 characters" in str(e)
    
    # Test with mixed field types
    mixed_all_of = AllOf([String(pattern="^[A-Z]+$"), String(max_length=3)])
    result = mixed_all_of.validate("ABC")
    assert result == "ABC"
    
    try:
        mixed_all_of.validate("abcd")
        assert False, "Should have raised ValidationError"
    except Exception:
        pass
    
    try:
        mixed_all_of.validate("abc")
        assert False, "Should have raised ValidationError"
    except Exception:
        pass
    
    # Test that original value is returned when all validations pass
    all_of_return = AllOf([Integer(minimum=0), Integer(maximum=100)])
    result = all_of_return.validate(50)
    assert result == 50


# LLM-generated content at query #17
#--------------------------

```python
def test_Not():
    # Test basic Not functionality with a simple field
    from typesystem.fields import Integer
    
    integer_field = Integer(minimum=0, maximum=10)
    not_field = Not(negated=integer_field)
    
    # Value that should pass (doesn't match the negated field)
    result = not_field.validate(-5)
    assert result == -5
    
    # Value that should fail (matches the negated field)
    try:
        not_field.validate(5)
        assert False, "Should have raised validation error"
    except Exception as exc:
        assert "Must not match" in str(exc)
    
    # Test with allow_null=False by default
    assert not_field.allow_null is False
    
    # Test that allow_null cannot be passed in kwargs
    try:
        Not(negated=integer_field, allow_null=True)
        assert False, "Should have raised assertion error"
    except AssertionError:
        pass
    
    # Test with different field types
    from typesystem.fields import String
    
    string_field = String(min_length=3)
    not_string_field = Not(negated=string_field)
    
    # Short string should pass (doesn't match String with min_length=3)
    result = not_string_field.validate("ab")
    assert result == "ab"
    
    # Long string should fail (matches String with min_length=3)
    try:
        not_string_field.validate("abc")
        assert False, "Should have raised validation error"
    except Exception as exc:
        assert "Must not match" in str(exc)


# LLM-generated content at query #18
#--------------------------

```python
def test_Not():
    from typesystem.fields import Integer, String
    
    # Test that Not raises validation error when negated field validates
    integer_field = Integer()
    not_integer = Not(negated=integer_field)
    
    # Should raise error for integer value
    with pytest.raises(not_integer.validation_error("negated")):
        not_integer.validate(42)
    
    # Should pass for non-integer value
    assert not_integer.validate("hello") == "hello"
    assert not_integer.validate(True) == True
    
    # Test with String field
    string_field = String()
    not_string = Not(negated=string_field)
    
    # Should raise error for string value
    with pytest.raises(not_string.validation_error("negated")):
        not_string.validate("test")
    
    # Should pass for non-string value
    assert not_string.validate(123) == 123
    assert not_string.validate(None) == None
    
    # Test that allow_null cannot be passed to constructor
    with pytest.raises(AssertionError):
        Not(negated=integer_field, allow_null=True)
    
    # Test with complex field
    from typesystem.fields import Boolean
    boolean_field = Boolean()
    not_boolean = Not(negated=boolean_field)
    
    # Should raise error for boolean value
    with pytest.raises(not_boolean.validation_error("negated")):
        not_boolean.validate(True)
    
    with pytest.raises(not_boolean.validation_error("negated")):
        not_boolean.validate(False)
    
    # Should pass for non-boolean value
    assert not_boolean.validate("not boolean") == "not boolean"
    assert not_boolean.validate(123) == 123


# LLM-generated content at query #19
#--------------------------

```python
def test_NeverMatch():
    # Test that NeverMatch cannot be instantiated with allow_null
    try:
        NeverMatch(allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError as e:
        assert "allow_null" in str(e)

    # Test that NeverMatch can be instantiated without allow_null
    field = NeverMatch()
    assert isinstance(field, NeverMatch)
    assert isinstance(field, Field)
    assert field.errors == {"never": "This never validates."}

    # Test that validate always raises validation error
    try:
        field.validate(None)
        assert False, "Should have raised validation error"
    except Exception as e:
        assert hasattr(e, "code")
        assert e.code == "never"

    # Test with different values
    for value in [None, "", 0, [], {}, "test", 123]:
        try:
            field.validate(value)
            assert False, f"Should have raised validation error for value: {value}"
        except Exception as e:
            assert hasattr(e, "code")
            assert e.code == "never"


# LLM-generated content at query #20
#--------------------------

```python
def test_OneOf():
    from typesystem.fields import String, Integer, Boolean
    
    # Test basic initialization
    string_field = String()
    integer_field = Integer()
    one_of = OneOf(one_of=[string_field, integer_field])
    assert one_of.one_of == [string_field, integer_field]
    
    # Test that allow_null cannot be passed
    try:
        OneOf(one_of=[string_field], allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test with empty list
    one_of_empty = OneOf(one_of=[])
    assert one_of_empty.one_of == []
    
    # Test with single field
    one_of_single = OneOf(one_of=[string_field])
    assert one_of_single.one_of == [string_field]
    
    # Test with three fields
    boolean_field = Boolean()
    one_of_three = OneOf(one_of=[string_field, integer_field, boolean_field])
    assert len(one_of_three.one_of) == 3
    assert one_of_three.one_of[0] == string_field
    assert one_of_three.one_of[1] == integer_field
    assert one_of_three.one_of[2] == boolean_field


# LLM-generated content at query #21
#--------------------------

```python
def test_Not():
    # Test basic initialization with a field
    from typesystem.fields import String
    string_field = String(max_length=10)
    not_field = Not(string_field)
    
    assert not_field.negated is string_field
    assert not_field.errors == {"negated": "Must not match."}
    
    # Test that allow_null cannot be passed
    import pytest
    with pytest.raises(AssertionError):
        Not(string_field, allow_null=True)
    
    # Test validation with non-matching value
    result = not_field.validate("too_long_string_for_field")
    assert result == "too_long_string_for_field"
    
    # Test validation with matching value (should raise error)
    with pytest.raises(not_field.validation_error("negated")):
        not_field.validate("short")
    
    # Test with different field types
    from typesystem.fields import Integer
    int_field = Integer(minimum=0)
    not_int = Not(int_field)
    
    # Negative number doesn't match Integer(minimum=0), so should pass
    result = not_int.validate(-5)
    assert result == -5
    
    # Positive number matches, so should fail
    with pytest.raises(not_int.validation_error("negated")):
        not_int.validate(10)


# LLM-generated content at query #22
#--------------------------

```python
def test_Not():
    # Test basic initialization
    negated_field = Any()
    not_field = Not(negated=negated_field)
    assert not_field.negated == negated_field
    assert not_field.errors == {"negated": "Must not match."}

    # Test that allow_null cannot be passed
    try:
        Not(negated=negated_field, allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass

    # Test validation when negated field fails (should succeed)
    failing_field = NeverMatch()
    not_field = Not(negated=failing_field)
    result = not_field.validate("test_value")
    assert result == "test_value"

    # Test validation when negated field succeeds (should fail)
    passing_field = Any()
    not_field = Not(negated=passing_field)
    try:
        not_field.validate("test_value")
        assert False, "Should have raised validation error"
    except Exception as e:
        assert str(e) == "Must not match."

    # Test with more complex negated field
    from typesystem import Integer
    integer_field = Integer()
    not_integer = Not(negated=integer_field)
    
    # String should pass (not an integer)
    result = not_integer.validate("not a number")
    assert result == "not a number"
    
    # Integer should fail
    try:
        not_integer.validate(42)
        assert False, "Should have raised validation error"
    except Exception as e:
        assert str(e) == "Must not match."


# LLM-generated content at query #23
#--------------------------

```python
def test_Not():
    from typesystem.fields import String, Integer
    
    # Test basic initialization
    string_field = String()
    not_field = Not(string_field)
    assert not_field.negated == string_field
    assert not_field.errors == {"negated": "Must not match."}
    
    # Test that allow_null cannot be passed
    try:
        Not(string_field, allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test validation with non-matching value
    integer_field = Integer()
    not_integer = Not(integer_field)
    result = not_integer.validate("hello")
    assert result == "hello"
    
    # Test validation with matching value (should raise error)
    try:
        not_integer.validate(42)
        assert False, "Should have raised ValidationError"
    except Exception as e:
        assert "negated" in str(e)
    
    # Test with nested field
    min_length_string = String(min_length=5)
    not_min_length = Not(min_length_string)
    
    # Should pass for strings shorter than 5 characters
    result = not_min_length.validate("hi")
    assert result == "hi"
    
    # Should fail for strings with 5 or more characters
    try:
        not_min_length.validate("hello")
        assert False, "Should have raised ValidationError"
    except Exception as e:
        assert "negated" in str(e)


# LLM-generated content at query #24
#--------------------------

```python
def test_NeverMatch():
    # Test that NeverMatch cannot be instantiated with allow_null
    try:
        NeverMatch(allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError as e:
        assert "allow_null" in str(e)
    
    # Test that NeverMatch can be instantiated without allow_null
    field = NeverMatch()
    assert isinstance(field, NeverMatch)
    assert isinstance(field, Field)
    
    # Test that validate always raises validation error
    try:
        field.validate(None)
        assert False, "Should have raised validation error"
    except Exception as e:
        assert "never" in str(e)
    
    try:
        field.validate("test")
        assert False, "Should have raised validation error"
    except Exception as e:
        assert "never" in str(e)
    
    try:
        field.validate(123)
        assert False, "Should have raised validation error"
    except Exception as e:
        assert "never" in str(e)
    
    # Test with additional kwargs
    field2 = NeverMatch(title="Test Field", description="A field that never validates")
    assert field2.title == "Test Field"
    assert field2.description == "A field that never validates"


# LLM-generated content at query #25
#--------------------------

```python
def test_NeverMatch():
    # Test that NeverMatch cannot be instantiated with allow_null
    with pytest.raises(AssertionError):
        NeverMatch(allow_null=True)
    
    # Test that NeverMatch can be instantiated without allow_null
    field = NeverMatch()
    assert isinstance(field, NeverMatch)
    assert isinstance(field, Field)
    
    # Test that validate always raises validation error
    with pytest.raises(field.validation_error("never")):
        field.validate(None)
    
    with pytest.raises(field.validation_error("never")):
        field.validate(123)
    
    with pytest.raises(field.validation_error("never")):
        field.validate("test")
    
    with pytest.raises(field.validation_error("never")):
        field.validate({"key": "value"})
    
    # Test that errors dict contains the correct error message
    assert field.errors == {"never": "This never validates."}


# LLM-generated content at query #26
#--------------------------

```python
def test_AllOf():
    from typesystem.fields import Integer, String, Boolean
    
    # Test basic initialization
    integer_field = Integer()
    string_field = String()
    all_of = AllOf([integer_field, string_field])
    
    assert all_of.all_of == [integer_field, string_field]
    assert all_of.allow_null is False
    
    # Test that allow_null cannot be passed in kwargs
    try:
        AllOf([integer_field], allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test with empty list
    all_of_empty = AllOf([])
    assert all_of_empty.all_of == []
    
    # Test with single field
    single_field = AllOf([integer_field])
    assert single_field.all_of == [integer_field]
    
    # Test with three fields
    boolean_field = Boolean()
    all_of_three = AllOf([integer_field, string_field, boolean_field])
    assert len(all_of_three.all_of) == 3
    assert all_of_three.all_of[0] == integer_field
    assert all_of_three.all_of[1] == string_field
    assert all_of_three.all_of[2] == boolean_field


# LLM-generated content at query #27
#--------------------------

```python
def test_OneOf():
    # Test basic initialization
    field1 = Any()
    field2 = NeverMatch()
    one_of = OneOf(one_of=[field1, field2])
    assert one_of.one_of == [field1, field2]
    
    # Test that allow_null cannot be passed
    try:
        OneOf(one_of=[field1], allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test with empty list
    one_of_empty = OneOf(one_of=[])
    assert one_of_empty.one_of == []
    
    # Test validation with single match
    result = one_of.validate("test")
    assert result == "test"
    
    # Test validation with no matches
    try:
        one_of.validate(None)
        assert False, "Should have raised validation error"
    except Exception as e:
        assert "no_match" in str(e)
    
    # Test validation with multiple matches
    field3 = Any()
    one_of_multiple = OneOf(one_of=[field1, field3])
    try:
        one_of_multiple.validate("test")
        assert False, "Should have raised validation error"
    except Exception as e:
        assert "multiple_matches" in str(e)


# LLM-generated content at query #28
#--------------------------

```python
def test_AllOf():
    # Test basic initialization
    field1 = Any()
    field2 = Any()
    all_of = AllOf([field1, field2])
    assert all_of.all_of == [field1, field2]
    
    # Test that allow_null cannot be passed
    try:
        AllOf([field1], allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test validation with multiple fields
    from typesystem.fields import Integer, String
    
    int_field = Integer()
    pos_field = Integer(minimum=0)
    all_of = AllOf([int_field, pos_field])
    
    # Valid case - matches all fields
    assert all_of.validate(5) == 5
    
    # Invalid case - fails one field
    try:
        all_of.validate(-1)
        assert False, "Should have raised ValidationError"
    except Exception as e:
        assert "minimum" in str(e).lower()
    
    # Invalid case - fails another field
    try:
        all_of.validate("not a number")
        assert False, "Should have raised ValidationError"
    except Exception as e:
        assert "number" in str(e).lower() or "integer" in str(e).lower()
    
    # Test with three fields
    int_field = Integer()
    min_field = Integer(minimum=0)
    max_field = Integer(maximum=10)
    all_of = AllOf([int_field, min_field, max_field])
    
    assert all_of.validate(5) == 5
    
    try:
        all_of.validate(15)
        assert False, "Should have raised ValidationError"
    except Exception as e:
        assert "maximum" in str(e).lower()
    
    try:
        all_of.validate(-5)
        assert False, "Should have raised ValidationError"
    except Exception as e:
        assert "minimum" in str(e).lower()


# LLM-generated content at query #29
#--------------------------

```python
def test_IfThenElse():
    # Test basic initialization with all parameters
    if_field = Any()
    then_field = Any()
    else_field = Any()
    field = IfThenElse(if_field, then_field, else_field)
    assert field.if_clause == if_field
    assert field.then_clause == then_field
    assert field.else_clause == else_field

    # Test initialization with only if_clause (then/else should default to Any)
    field = IfThenElse(if_field)
    assert field.if_clause == if_field
    assert isinstance(field.then_clause, Any)
    assert isinstance(field.else_clause, Any)

    # Test initialization with if_clause and then_clause only
    field = IfThenElse(if_field, then_field)
    assert field.if_clause == if_field
    assert field.then_clause == then_field
    assert isinstance(field.else_clause, Any)

    # Test that allow_null cannot be passed in kwargs
    try:
        IfThenElse(if_field, allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError as e:
        assert "allow_null" in str(e)

    # Test validation when if_clause matches
    if_field = Field()
    then_field = Field()
    else_field = Field()
    field = IfThenElse(if_field, then_field, else_field)
    
    # Mock validate_or_error to simulate if_clause match
    original_validate_or_error = if_field.validate_or_error
    if_field.validate_or_error = lambda x: (x, None)
    then_field.validate = lambda x: x
    
    result = field.validate("test_value")
    assert result == "test_value"
    
    if_field.validate_or_error = original_validate_or_error

    # Test validation when if_clause doesn't match
    if_field.validate_or_error = lambda x: (x, "error")
    else_field.validate = lambda x: x
    
    result = field.validate("test_value")
    assert result == "test_value"
    
    # Clean up
    if_field.validate_or_error = original_validate_or_error


# LLM-generated content at query #30
#--------------------------

```python
def test_AllOf():
    # Test basic initialization
    field1 = Any()
    field2 = Any()
    all_of = AllOf([field1, field2])
    assert all_of.all_of == [field1, field2]
    
    # Test that allow_null cannot be passed
    try:
        AllOf([field1], allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test with empty list
    all_of_empty = AllOf([])
    assert all_of_empty.all_of == []
    
    # Test validation passes when all fields validate
    class AlwaysValid(Field):
        def validate(self, value):
            return value
    
    valid_field1 = AlwaysValid()
    valid_field2 = AlwaysValid()
    all_of_valid = AllOf([valid_field1, valid_field2])
    result = all_of_valid.validate("test_value")
    assert result == "test_value"
    
    # Test validation fails when any field fails
    class FailingField(Field):
        def validate(self, value):
            raise self.validation_error("test_error")
    
    failing_field = FailingField()
    all_of_mixed = AllOf([valid_field1, failing_field])
    try:
        all_of_mixed.validate("test_value")
        assert False, "Should have raised ValidationError"
    except Exception as e:
        assert hasattr(e, 'code')
        assert e.code == "test_error"


# LLM-generated content at query #31
#--------------------------

```python
def test_IfThenElse():
    from typesystem.fields import Integer, String, Boolean
    
    # Test basic initialization with all clauses
    int_field = Integer()
    string_field = String()
    bool_field = Boolean()
    
    field = IfThenElse(
        if_clause=int_field,
        then_clause=string_field,
        else_clause=bool_field
    )
    
    assert field.if_clause == int_field
    assert field.then_clause == string_field
    assert field.else_clause == bool_field
    
    # Test initialization with only if_clause (then/else should default to Any)
    field2 = IfThenElse(if_clause=int_field)
    assert field2.if_clause == int_field
    assert isinstance(field2.then_clause, Any)
    assert isinstance(field2.else_clause, Any)
    
    # Test initialization with if_clause and then_clause only
    field3 = IfThenElse(if_clause=int_field, then_clause=string_field)
    assert field3.if_clause == int_field
    assert field3.then_clause == string_field
    assert isinstance(field3.else_clause, Any)
    
    # Test initialization with if_clause and else_clause only
    field4 = IfThenElse(if_clause=int_field, else_clause=bool_field)
    assert field4.if_clause == int_field
    assert isinstance(field4.then_clause, Any)
    assert field4.else_clause == bool_field
    
    # Test that allow_null cannot be passed
    try:
        IfThenElse(if_clause=int_field, allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test validation when if_clause matches
    field5 = IfThenElse(
        if_clause=Integer(minimum=0),
        then_clause=String(max_length=5),
        else_clause=Boolean()
    )
    
    # Positive number should trigger then_clause (String validation)
    try:
        field5.validate(10)
        assert False, "Should have raised validation error for string"
    except Exception as e:
        assert "Must be a string." in str(e)
    
    # Test validation when if_clause doesn't match
    # Negative number should trigger else_clause (Boolean validation)
    try:
        field5.validate(-5)
        assert False, "Should have raised validation error for boolean"
    except Exception as e:
        assert "Must be a boolean." in str(e)
    
    # Test successful validation path
    field6 = IfThenElse(
        if_clause=Integer(minimum=0),
        then_clause=String(),
        else_clause=Boolean()
    )
    
    # Valid string should pass when if_clause matches
    result = field6.validate(5)
    assert result == "5"
    
    # Valid boolean should pass when if_clause doesn't match
    result = field6.validate(False)
    assert result is False


# LLM-generated content at query #32
#--------------------------

```python
def test_OneOf():
    # Test basic initialization with list of fields
    field1 = Any()
    field2 = NeverMatch()
    one_of = OneOf([field1, field2])
    assert one_of.one_of == [field1, field2]
    
    # Test that allow_null cannot be passed
    try:
        OneOf([field1], allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test with empty list
    one_of_empty = OneOf([])
    assert one_of_empty.one_of == []
    
    # Test that parent Field initialization works
    one_of_with_title = OneOf([field1], title="Test Field")
    assert one_of_with_title.title == "Test Field"


# LLM-generated content at query #33
#--------------------------

```python
def test_AllOf():
    from typesystem.fields import Integer, String
    
    # Test basic initialization
    integer_field = Integer()
    string_field = String()
    all_of = AllOf([integer_field, string_field])
    
    assert all_of.all_of == [integer_field, string_field]
    assert all_of.allow_null is False
    
    # Test that allow_null cannot be passed
    try:
        AllOf([integer_field], allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test with empty list
    all_of_empty = AllOf([])
    assert all_of_empty.all_of == []
    
    # Test validation with multiple fields
    all_of = AllOf([Integer(minimum=0), Integer(maximum=10)])
    
    # Should validate successfully
    result = all_of.validate(5)
    assert result == 5
    
    # Should fail validation (fails first constraint)
    try:
        all_of.validate(-5)
        assert False, "Should have raised ValidationError"
    except Exception as e:
        assert hasattr(e, 'code')
    
    # Should fail validation (fails second constraint)
    try:
        all_of.validate(15)
        assert False, "Should have raised ValidationError"
    except Exception as e:
        assert hasattr(e, 'code')
    
    # Test with different field types
    all_of_mixed = AllOf([Integer(), String(max_length=5)])
    
    # Integer validation should work
    result = all_of_mixed.validate(42)
    assert result == 42
    
    # String validation should work
    result = all_of_mixed.validate("test")
    assert result == "test"
    
    # String too long should fail
    try:
        all_of_mixed.validate("toolong")
        assert False, "Should have raised ValidationError"
    except Exception as e:
        assert hasattr(e, 'code')


# LLM-generated content at query #34
#--------------------------

```python
def test_OneOf():
    from typesystem.fields import String, Integer, Boolean
    
    # Test basic initialization
    one_of_field = OneOf(one_of=[String(), Integer()])
    assert len(one_of_field.one_of) == 2
    assert isinstance(one_of_field.one_of[0], String)
    assert isinstance(one_of_field.one_of[1], Integer)
    
    # Test that allow_null cannot be passed
    try:
        OneOf(one_of=[String()], allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test with empty list
    one_of_empty = OneOf(one_of=[])
    assert one_of_empty.one_of == []
    
    # Test with multiple field types
    fields = [String(max_length=10), Integer(minimum=0), Boolean()]
    one_of_mixed = OneOf(one_of=fields)
    assert len(one_of_mixed.one_of) == 3
    assert one_of_mixed.one_of[0].max_length == 10
    assert one_of_mixed.one_of[1].minimum == 0
    assert isinstance(one_of_mixed.one_of[2], Boolean)
    
    # Test that parent constructor is called
    one_of_with_title = OneOf(one_of=[String()], title="Test Field")
    assert one_of_with_title.title == "Test Field"


# LLM-generated content at query #35
#--------------------------

```python
def test_AllOf():
    from typesystem.fields import Integer, String, Boolean

    # Test basic initialization
    integer_field = Integer()
    string_field = String()
    all_of = AllOf([integer_field, string_field])
    assert all_of.all_of == [integer_field, string_field]

    # Test that allow_null cannot be passed
    try:
        AllOf([integer_field], allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass

    # Test with empty list
    all_of_empty = AllOf([])
    assert all_of_empty.all_of == []

    # Test with multiple fields
    boolean_field = Boolean()
    all_of_multi = AllOf([integer_field, string_field, boolean_field])
    assert len(all_of_multi.all_of) == 3
    assert all_of_multi.all_of[0] == integer_field
    assert all_of_multi.all_of[1] == string_field
    assert all_of_multi.all_of[2] == boolean_field

    # Test that parent class initialization works
    assert hasattr(all_of, 'errors')
    assert hasattr(all_of, 'validate')


# LLM-generated content at query #36
#--------------------------

```python
def test_AllOf():
    from typesystem.fields import String, Integer, Boolean
    
    # Test basic initialization
    string_field = String()
    integer_field = Integer()
    all_of = AllOf([string_field, integer_field])
    assert all_of.all_of == [string_field, integer_field]
    
    # Test that allow_null cannot be passed
    try:
        AllOf([string_field], allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test with empty list
    all_of_empty = AllOf([])
    assert all_of_empty.all_of == []
    
    # Test validation with multiple fields
    all_of = AllOf([String(max_length=5), String(min_length=2)])
    # Valid case
    result = all_of.validate("test")
    assert result == "test"
    
    # Invalid case - fails first constraint
    try:
        all_of.validate("toolongstring")
        assert False, "Should have raised ValidationError"
    except Exception as e:
        assert "Must have no more than 5 characters" in str(e)
    
    # Invalid case - fails second constraint
    try:
        all_of.validate("a")
        assert False, "Should have raised ValidationError"
    except Exception as e:
        assert "Must have at least 2 characters" in str(e)
    
    # Test with mixed field types
    all_of_mixed = AllOf([String(), Integer()])
    # This should fail since a value can't be both string and integer
    try:
        all_of_mixed.validate("test")
        assert False, "Should have raised ValidationError"
    except Exception:
        pass
    
    # Test that validate returns the original value when all pass
    all_of_simple = AllOf([String(), String(max_length=10)])
    value = "hello"
    result = all_of_simple.validate(value)
    assert result is value


# LLM-generated content at query #37
#--------------------------

```python
def test_NeverMatch():
    # Test that NeverMatch cannot be initialized with allow_null
    with pytest.raises(AssertionError):
        NeverMatch(allow_null=True)
    
    # Test that NeverMatch can be initialized without allow_null
    field = NeverMatch()
    assert isinstance(field, NeverMatch)
    assert isinstance(field, Field)
    
    # Test that validate always raises validation error
    with pytest.raises(field.validation_error("never")):
        field.validate(None)
    
    with pytest.raises(field.validation_error("never")):
        field.validate("any value")
    
    with pytest.raises(field.validation_error("never")):
        field.validate(123)
    
    with pytest.raises(field.validation_error("never")):
        field.validate({"key": "value"})
    
    # Test error message
    try:
        field.validate("test")
    except Exception as e:
        assert str(e) == "This never validates."


# LLM-generated content at query #38
#--------------------------

```python
def test_Not():
    # Test basic Not field initialization
    from typesystem.fields import String
    
    string_field = String(max_length=10)
    not_field = Not(string_field)
    
    assert not_field.negated == string_field
    assert not_field.errors == {"negated": "Must not match."}
    
    # Test that Not field rejects values that match the negated field
    with pytest.raises(not_field.validation_error("negated")):
        not_field.validate("hello")
    
    # Test that Not field accepts values that don't match the negated field
    result = not_field.validate(123)
    assert result == 123
    
    # Test with different field types
    integer_field = Integer(minimum=0)
    not_integer = Not(integer_field)
    
    # Should reject positive integers
    with pytest.raises(not_integer.validation_error("negated")):
        not_integer.validate(5)
    
    # Should accept negative integers
    result = not_integer.validate(-5)
    assert result == -5
    
    # Test that allow_null cannot be passed to constructor
    with pytest.raises(AssertionError):
        Not(string_field, allow_null=True)


# LLM-generated content at query #39
#--------------------------

```python
def test_OneOf():
    # Test basic initialization
    field1 = Any()
    field2 = NeverMatch()
    one_of = OneOf(one_of=[field1, field2])
    assert one_of.one_of == [field1, field2]
    
    # Test that allow_null cannot be passed
    try:
        OneOf(one_of=[], allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test with empty list
    one_of_empty = OneOf(one_of=[])
    assert one_of_empty.one_of == []
    
    # Test with single field
    single_field = Any()
    one_of_single = OneOf(one_of=[single_field])
    assert one_of_single.one_of == [single_field]
    
    # Test with multiple fields
    fields = [Any(), NeverMatch(), Any()]
    one_of_multi = OneOf(one_of=fields)
    assert one_of_multi.one_of == fields


# LLM-generated content at query #40
#--------------------------

```python
def test_NeverMatch():
    field = NeverMatch()
    assert isinstance(field, NeverMatch)
    assert isinstance(field, Field)
    assert field.errors == {"never": "This never validates."}
    
    with pytest.raises(AssertionError):
        NeverMatch(allow_null=True)
    
    with pytest.raises(field.validation_error("never")):
        field.validate(None)
    
    with pytest.raises(field.validation_error("never")):
        field.validate("test")
    
    with pytest.raises(field.validation_error("never")):
        field.validate(123)
    
    with pytest.raises(field.validation_error("never")):
        field.validate([])


# LLM-generated content at query #41
#--------------------------

```python
def test_IfThenElse():
    # Test basic initialization with all parameters
    if_field = Any()
    then_field = Any()
    else_field = Any()
    field = IfThenElse(if_field, then_field, else_field)
    assert field.if_clause == if_field
    assert field.then_clause == then_field
    assert field.else_clause == else_field
    
    # Test initialization with only if_clause (then/else should default to Any)
    field = IfThenElse(if_field)
    assert field.if_clause == if_field
    assert isinstance(field.then_clause, Any)
    assert isinstance(field.else_clause, Any)
    
    # Test initialization with if_clause and then_clause only
    field = IfThenElse(if_field, then_field)
    assert field.if_clause == if_field
    assert field.then_clause == then_field
    assert isinstance(field.else_clause, Any)
    
    # Test that allow_null cannot be passed in kwargs
    try:
        IfThenElse(if_field, allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test validation when if_clause passes
    if_field = Any()
    then_field = Field()
    else_field = Field()
    field = IfThenElse(if_field, then_field, else_field)
    
    # Mock validate method on then_field to track calls
    then_validated = []
    original_validate = then_field.validate
    then_field.validate = lambda x: then_validated.append(x) or x
    
    result = field.validate("test_value")
    assert result == "test_value"
    assert then_validated == ["test_value"]
    
    # Restore original validate
    then_field.validate = original_validate
    
    # Test validation when if_clause fails
    if_field = NeverMatch()
    then_field = Field()
    else_field = Field()
    field = IfThenElse(if_field, then_field, else_field)
    
    # Mock validate method on else_field to track calls
    else_validated = []
    original_validate = else_field.validate
    else_field.validate = lambda x: else_validated.append(x) or x
    
    result = field.validate("test_value")
    assert result == "test_value"
    assert else_validated == ["test_value"]
    
    # Restore original validate
    else_field.validate = original_validate


# LLM-generated content at query #42
#--------------------------

```python
def test_OneOf():
    # Test basic initialization
    field1 = Any()
    field2 = NeverMatch()
    one_of = OneOf(one_of=[field1, field2])
    assert one_of.one_of == [field1, field2]
    assert one_of.errors == {
        "no_match": "Did not match any valid type.",
        "multiple_matches": "Matched more than one type.",
    }

    # Test that allow_null cannot be passed
    try:
        OneOf(one_of=[field1], allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass

    # Test with empty list
    one_of_empty = OneOf(one_of=[])
    assert one_of_empty.one_of == []

    # Test validation with exactly one match
    result = one_of.validate("test")
    assert result == "test"

    # Test validation with no matches
    try:
        one_of.validate("test")
    except Exception as e:
        assert str(e) == "Did not match any valid type."

    # Test validation with multiple matches
    field3 = Any()
    one_of_multiple = OneOf(one_of=[field1, field3])
    try:
        one_of_multiple.validate("test")
        assert False, "Should have raised validation error"
    except Exception as e:
        assert str(e) == "Matched more than one type."

    # Test validation with nested fields
    nested_field = OneOf(one_of=[Any()])
    result = nested_field.validate("nested")
    assert result == "nested"


# LLM-generated content at query #43
#--------------------------

```python
def test_OneOf():
    # Test basic initialization with list of fields
    field1 = Any()
    field2 = NeverMatch()
    one_of = OneOf([field1, field2])
    assert one_of.one_of == [field1, field2]
    
    # Test that allow_null cannot be passed
    try:
        OneOf([field1], allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test with empty list
    one_of_empty = OneOf([])
    assert one_of_empty.one_of == []
    
    # Test validation with single match
    result = one_of.validate("test")
    assert result == "test"
    
    # Test validation with no matches
    try:
        one_of.validate("test")
    except Exception as e:
        assert str(e) == "Did not match any valid type."
    
    # Test validation with multiple matches
    field3 = Any()
    field4 = Any()
    one_of_multiple = OneOf([field3, field4])
    try:
        one_of_multiple.validate("test")
        assert False, "Should have raised validation error"
    except Exception as e:
        assert str(e) == "Matched more than one type."


# LLM-generated content at query #44
#--------------------------

```python
def test_NeverMatch():
    field = NeverMatch()
    assert isinstance(field, Field)
    assert field.errors == {"never": "This never validates."}
    
    with pytest.raises(AssertionError):
        NeverMatch(allow_null=True)
    
    with pytest.raises(field.validation_error("never")):
        field.validate("any value")
    
    with pytest.raises(field.validation_error("never")):
        field.validate(None)
    
    with pytest.raises(field.validation_error("never")):
        field.validate(123)
    
    with pytest.raises(field.validation_error("never")):
        field.validate([])


# LLM-generated content at query #45
#--------------------------

```python
def test_NeverMatch():
    # Test that NeverMatch cannot be initialized with allow_null
    with pytest.raises(AssertionError):
        NeverMatch(allow_null=True)
    
    # Test that NeverMatch can be initialized without allow_null
    field = NeverMatch()
    assert isinstance(field, NeverMatch)
    assert isinstance(field, Field)
    
    # Test that validate always raises validation error
    with pytest.raises(field.validation_error("never")):
        field.validate(None)
    
    with pytest.raises(field.validation_error("never")):
        field.validate("any value")
    
    with pytest.raises(field.validation_error("never")):
        field.validate(123)
    
    with pytest.raises(field.validation_error("never")):
        field.validate({"key": "value"})
    
    # Test that errors dict contains the correct error message
    assert field.errors == {"never": "This never validates."}


# LLM-generated content at query #46
#--------------------------

```python
def test_NeverMatch():
    field = NeverMatch()
    assert field.errors == {"never": "This never validates."}
    assert field.allow_null is False

    with pytest.raises(AssertionError):
        NeverMatch(allow_null=True)

    with pytest.raises(AssertionError):
        NeverMatch(allow_null=False)

    with pytest.raises(field.validation_error("never")):
        field.validate(None)

    with pytest.raises(field.validation_error("never")):
        field.validate("test")

    with pytest.raises(field.validation_error("never")):
        field.validate(123)

    with pytest.raises(field.validation_error("never")):
        field.validate([])

    with pytest.raises(field.validation_error("never")):
        field.validate({})


# LLM-generated content at query #47
#--------------------------

```python
def test_IfThenElse():
    # Test basic initialization with all clauses provided
    if_field = Any()
    then_field = Any()
    else_field = Any()
    field = IfThenElse(if_field, then_field, else_field)
    assert field.if_clause == if_field
    assert field.then_clause == then_field
    assert field.else_clause == else_field
    
    # Test initialization with only if_clause (then/else should default to Any)
    field = IfThenElse(if_field)
    assert field.if_clause == if_field
    assert isinstance(field.then_clause, Any)
    assert isinstance(field.else_clause, Any)
    
    # Test initialization with if_clause and then_clause only
    field = IfThenElse(if_field, then_field)
    assert field.if_clause == if_field
    assert field.then_clause == then_field
    assert isinstance(field.else_clause, Any)
    
    # Test that allow_null cannot be passed in kwargs
    try:
        IfThenElse(if_field, allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test validation when if_clause matches
    if_field = Any()
    then_field = Any()
    else_field = NeverMatch()
    field = IfThenElse(if_field, then_field, else_field)
    result = field.validate("test_value")
    assert result == "test_value"
    
    # Test validation when if_clause doesn't match
    if_field = NeverMatch()
    then_field = NeverMatch()
    else_field = Any()
    field = IfThenElse(if_field, then_field, else_field)
    result = field.validate("test_value")
    assert result == "test_value"
    
    # Test validation with specific field types
    from typesystem.fields import String, Integer
    
    if_field = String()
    then_field = Integer()
    else_field = String()
    
    field = IfThenElse(if_field, then_field, else_field)
    
    # If clause matches (string), then clause should be applied
    try:
        field.validate("not_an_integer")
        assert False, "Should have raised validation error"
    except Exception as e:
        assert "Must be a number." in str(e)
    
    # If clause doesn't match (not a string), else clause should be applied
    result = field.validate(123)
    assert result == "123"


# LLM-generated content at query #48
#--------------------------

```python
def test_AllOf():
    from typesystem.fields import Integer, String

    # Test basic initialization
    integer_field = Integer()
    string_field = String()
    all_of = AllOf([integer_field, string_field])
    assert all_of.all_of == [integer_field, string_field]
    assert all_of.allow_null is False

    # Test that allow_null cannot be passed in kwargs
    try:
        AllOf([integer_field], allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass

    # Test validation with valid value
    result = all_of.validate(123)
    assert result == 123

    # Test validation with invalid value (fails first field)
    try:
        all_of.validate("not_an_integer")
        assert False, "Should have raised ValidationError"
    except Exception as e:
        assert "Must be a number." in str(e)

    # Test validation with invalid value (fails second field)
    try:
        all_of.validate(123.45)
        assert False, "Should have raised ValidationError"
    except Exception as e:
        assert "Must be a string." in str(e)

    # Test with empty list of fields
    empty_all_of = AllOf([])
    result = empty_all_of.validate("any_value")
    assert result == "any_value"

    # Test with nested AllOf
    nested_all_of = AllOf([all_of, String(max_length=5)])
    result = nested_all_of.validate(123)
    assert result == 123

    try:
        nested_all_of.validate(123456)
        assert False, "Should have raised ValidationError"
    except Exception as e:
        assert "Must be a string." in str(e) or "Must have no more than 5 characters." in str(e)


# LLM-generated content at query #49
#--------------------------

```python
def test_OneOf():
    from typesystem.fields import String, Integer, Boolean
    
    # Test basic initialization
    string_field = String()
    integer_field = Integer()
    one_of = OneOf(one_of=[string_field, integer_field])
    assert one_of.one_of == [string_field, integer_field]
    
    # Test that allow_null cannot be passed
    try:
        OneOf(one_of=[string_field], allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test with empty list
    one_of_empty = OneOf(one_of=[])
    assert one_of_empty.one_of == []
    
    # Test validation with exactly one match
    string_only = OneOf(one_of=[string_field])
    result = string_only.validate("test")
    assert result == "test"
    
    # Test validation with no matches
    try:
        one_of.validate(True)
        assert False, "Should have raised validation error"
    except Exception as e:
        assert "no_match" in str(e)
    
    # Test validation with multiple matches (when fields overlap)
    any_field = Any()
    overlapping = OneOf(one_of=[any_field, string_field])
    try:
        overlapping.validate("test")
        assert False, "Should have raised validation error"
    except Exception as e:
        assert "multiple_matches" in str(e)
    
    # Test validation with exactly one match from multiple options
    result = one_of.validate("test")
    assert result == "test"
    
    result = one_of.validate(123)
    assert result == 123


# LLM-generated content at query #50
#--------------------------

```python
def test_NeverMatch():
    field = NeverMatch()
    assert field.errors == {"never": "This never validates."}
    with pytest.raises(AssertionError):
        NeverMatch(allow_null=True)
    with pytest.raises(field.validation_error):
        field.validate("any value")
    with pytest.raises(field.validation_error):
        field.validate(None)
    with pytest.raises(field.validation_error):
        field.validate(123)
    with pytest.raises(field.validation_error):
        field.validate([])
    with pytest.raises(field.validation_error):
        field.validate({})


# LLM-generated content at query #51
#--------------------------

```python
def test_OneOf():
    from typesystem.fields import String, Integer, Boolean
    
    # Test basic initialization with list of fields
    string_field = String()
    integer_field = Integer()
    one_of = OneOf(one_of=[string_field, integer_field])
    assert one_of.one_of == [string_field, integer_field]
    
    # Test that allow_null cannot be passed
    try:
        OneOf(one_of=[string_field], allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test with empty list
    one_of_empty = OneOf(one_of=[])
    assert one_of_empty.one_of == []
    
    # Test with single field
    one_of_single = OneOf(one_of=[string_field])
    assert one_of_single.one_of == [string_field]
    
    # Test with three fields
    boolean_field = Boolean()
    one_of_three = OneOf(one_of=[string_field, integer_field, boolean_field])
    assert len(one_of_three.one_of) == 3
    assert isinstance(one_of_three.one_of[0], String)
    assert isinstance(one_of_three.one_of[1], Integer)
    assert isinstance(one_of_three.one_of[2], Boolean)


# LLM-generated content at query #52
#--------------------------

```python
def test_OneOf():
    # Test basic initialization
    field1 = Any()
    field2 = NeverMatch()
    one_of = OneOf([field1, field2])
    assert one_of.one_of == [field1, field2]
    
    # Test that allow_null cannot be passed
    try:
        OneOf([field1], allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test with empty list
    one_of_empty = OneOf([])
    assert one_of_empty.one_of == []
    
    # Test validation with exactly one match
    result = one_of.validate("test")
    assert result == "test"
    
    # Test validation with no matches
    try:
        one_of.validate("test")
    except Exception as e:
        assert str(e) == "Did not match any valid type."
    
    # Test validation with multiple matches (when multiple fields accept the same value)
    field3 = Any()
    field4 = Any()
    one_of_multiple = OneOf([field3, field4])
    try:
        one_of_multiple.validate("test")
        assert False, "Should have raised validation error for multiple matches"
    except Exception as e:
        assert str(e) == "Matched more than one type."
    
    # Test validation with nested OneOf
    nested_one_of = OneOf([one_of])
    result = nested_one_of.validate("test")
    assert result == "test"


# LLM-generated content at query #53
#--------------------------

```python
def test_AllOf():
    # Test basic initialization
    field1 = Any()
    field2 = Any()
    all_of = AllOf([field1, field2])
    assert all_of.all_of == [field1, field2]
    
    # Test that allow_null cannot be passed
    try:
        AllOf([field1], allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test with empty list
    all_of_empty = AllOf([])
    assert all_of_empty.all_of == []
    
    # Test validation passes when all fields validate
    class AlwaysValid(Field):
        def validate(self, value):
            return value
    
    valid_field1 = AlwaysValid()
    valid_field2 = AlwaysValid()
    all_of_valid = AllOf([valid_field1, valid_field2])
    result = all_of_valid.validate("test_value")
    assert result == "test_value"
    
    # Test validation fails when any field fails
    class FailingField(Field):
        def validate(self, value):
            raise self.validation_error("test_error")
    
    failing_field = FailingField()
    all_of_mixed = AllOf([valid_field1, failing_field])
    try:
        all_of_mixed.validate("test_value")
        assert False, "Should have raised ValidationError"
    except ValidationError:
        pass
    
    # Test that original value is returned when all validations pass
    test_value = {"key": "value"}
    result = all_of_valid.validate(test_value)
    assert result is test_value


# LLM-generated content at query #54
#--------------------------

```python
def test_NeverMatch():
    # Test that NeverMatch cannot be instantiated with allow_null
    with pytest.raises(AssertionError):
        NeverMatch(allow_null=True)
    
    # Test that NeverMatch can be instantiated without allow_null
    field = NeverMatch()
    assert isinstance(field, NeverMatch)
    assert isinstance(field, Field)
    
    # Test that validate always raises validation error
    with pytest.raises(field.validation_error("never")):
        field.validate(None)
    
    with pytest.raises(field.validation_error("never")):
        field.validate("any value")
    
    with pytest.raises(field.validation_error("never")):
        field.validate(123)
    
    with pytest.raises(field.validation_error("never")):
        field.validate({"key": "value"})
    
    # Test error message
    try:
        field.validate("test")
    except Exception as e:
        assert str(e) == "This never validates."


# LLM-generated content at query #55
#--------------------------

```python
def test_AllOf():
    from typesystem.fields import Integer, String, Boolean
    
    # Test basic initialization
    integer_field = Integer()
    string_field = String()
    all_of = AllOf([integer_field, string_field])
    
    assert all_of.all_of == [integer_field, string_field]
    assert all_of.allow_null is False
    
    # Test that allow_null cannot be passed in kwargs
    try:
        AllOf([integer_field], allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test with empty list
    all_of_empty = AllOf([])
    assert all_of_empty.all_of == []
    
    # Test validation with multiple fields
    all_of = AllOf([Integer(minimum=5), Integer(maximum=10)])
    
    # Should validate successfully
    result = all_of.validate(7)
    assert result == 7
    
    # Should fail validation (less than minimum)
    try:
        all_of.validate(3)
        assert False, "Should have raised ValidationError"
    except Exception as e:
        assert hasattr(e, 'code')
    
    # Should fail validation (greater than maximum)
    try:
        all_of.validate(12)
        assert False, "Should have raised ValidationError"
    except Exception as e:
        assert hasattr(e, 'code')
    
    # Test with different field types
    all_of_mixed = AllOf([String(min_length=2), String(max_length=5)])
    
    # Valid string
    result = all_of_mixed.validate("test")
    assert result == "test"
    
    # Too short
    try:
        all_of_mixed.validate("a")
        assert False, "Should have raised ValidationError"
    except Exception as e:
        assert hasattr(e, 'code')
    
    # Too long
    try:
        all_of_mixed.validate("toolong")
        assert False, "Should have raised ValidationError"
    except Exception as e:
        assert hasattr(e, 'code')


# LLM-generated content at query #56
#--------------------------

```python
def test_OneOf():
    # Test basic initialization with list of fields
    field1 = Any()
    field2 = NeverMatch()
    one_of = OneOf([field1, field2])
    assert one_of.one_of == [field1, field2]
    
    # Test that allow_null cannot be passed
    try:
        OneOf([field1], allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test with empty list
    one_of_empty = OneOf([])
    assert one_of_empty.one_of == []
    
    # Test validation with single match
    result = one_of.validate("test")
    assert result == "test"
    
    # Test validation with no matches
    try:
        one_of.validate(None)
        assert False, "Should have raised validation error"
    except Exception as e:
        assert "no_match" in str(e)
    
    # Test validation with multiple matches
    field3 = Any()
    field4 = Any()
    one_of_multiple = OneOf([field3, field4])
    try:
        one_of_multiple.validate("test")
        assert False, "Should have raised validation error"
    except Exception as e:
        assert "multiple_matches" in str(e)


# LLM-generated content at query #57
#--------------------------

```python
def test_Not():
    # Test basic initialization
    negated_field = Any()
    not_field = Not(negated=negated_field)
    assert not_field.negated == negated_field
    assert not_field.errors == {"negated": "Must not match."}

    # Test that allow_null cannot be passed
    try:
        Not(negated=negated_field, allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass

    # Test validation when negated field fails (should succeed)
    failing_field = NeverMatch()
    not_field = Not(negated=failing_field)
    result = not_field.validate("any value")
    assert result == "any value"

    # Test validation when negated field succeeds (should fail)
    passing_field = Any()
    not_field = Not(negated=passing_field)
    try:
        not_field.validate("any value")
        assert False, "Should have raised ValidationError"
    except Exception as e:
        assert str(e) == "Must not match."

    # Test with more complex negated field
    from typesystem import Integer
    not_integer = Not(negated=Integer())
    result = not_integer.validate("string")
    assert result == "string"
    try:
        not_integer.validate(123)
        assert False, "Should have raised ValidationError"
    except Exception as e:
        assert str(e) == "Must not match."


# LLM-generated content at query #58
#--------------------------

```python
def test_NeverMatch():
    field = NeverMatch()
    assert isinstance(field, NeverMatch)
    assert field.errors == {"never": "This never validates."}
    
    with pytest.raises(AssertionError):
        NeverMatch(allow_null=True)
    
    with pytest.raises(AssertionError):
        NeverMatch(allow_null=False)
    
    with pytest.raises(field.validation_error("never")):
        field.validate(None)
    
    with pytest.raises(field.validation_error("never")):
        field.validate("test")
    
    with pytest.raises(field.validation_error("never")):
        field.validate(123)
    
    with pytest.raises(field.validation_error("never")):
        field.validate([])
    
    with pytest.raises(field.validation_error("never")):
        field.validate({})


# LLM-generated content at query #59
#--------------------------

```python
def test_NeverMatch():
    # Test that NeverMatch cannot be instantiated with allow_null
    try:
        NeverMatch(allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass

    # Test that NeverMatch can be instantiated without allow_null
    field = NeverMatch()
    assert isinstance(field, NeverMatch)
    assert field.errors == {"never": "This never validates."}

    # Test that validate always raises validation error
    try:
        field.validate(None)
        assert False, "Should have raised validation error"
    except Exception as e:
        assert str(e) == "This never validates."

    # Test with different values
    try:
        field.validate(123)
        assert False, "Should have raised validation error"
    except Exception as e:
        assert str(e) == "This never validates."

    try:
        field.validate("test")
        assert False, "Should have raised validation error"
    except Exception as e:
        assert str(e) == "This never validates."

    try:
        field.validate({"key": "value"})
        assert False, "Should have raised validation error"
    except Exception as e:
        assert str(e) == "This never validates."


# LLM-generated content at query #60
#--------------------------

```python
def test_Not():
    # Test basic Not field initialization
    from typesystem.fields import String
    string_field = String(max_length=5)
    not_field = Not(negated=string_field)
    
    assert not_field.negated == string_field
    assert not_field.errors == {"negated": "Must not match."}
    
    # Test that Not rejects values that match the negated field
    # String with length <= 5 should match string_field, so Not should reject it
    with pytest.raises(not_field.validation_error("negated")):
        not_field.validate("hello")
    
    # Test that Not accepts values that don't match the negated field
    # String with length > 5 should not match string_field, so Not should accept it
    result = not_field.validate("hello world")
    assert result == "hello world"
    
    # Test with different field type
    from typesystem.fields import Integer
    int_field = Integer(minimum=0, maximum=10)
    not_int = Not(negated=int_field)
    
    # Value within range should be rejected
    with pytest.raises(not_int.validation_error("negated")):
        not_int.validate(5)
    
    # Value outside range should be accepted
    assert not_int.validate(-5) == -5
    assert not_int.validate(15) == 15
    
    # Test that allow_null cannot be passed to Not
    with pytest.raises(AssertionError):
        Not(negated=string_field, allow_null=True)


# LLM-generated content at query #61
#--------------------------

```python
def test_OneOf():
    from typesystem.fields import Integer, String, Boolean
    
    # Test basic initialization
    field = OneOf(one_of=[Integer(), String()])
    assert field.one_of[0].__class__.__name__ == "Integer"
    assert field.one_of[1].__class__.__name__ == "String"
    
    # Test validation with exactly one match
    field = OneOf(one_of=[Integer(), String()])
    assert field.validate(42) == 42
    assert field.validate("hello") == "hello"
    
    # Test validation with no matches
    field = OneOf(one_of=[Integer(), Boolean()])
    try:
        field.validate("hello")
        assert False, "Should have raised validation error"
    except Exception as e:
        assert "no_match" in str(e)
    
    # Test validation with multiple matches
    field = OneOf(one_of=[Integer(), Any()])
    try:
        field.validate(42)
        assert False, "Should have raised validation error"
    except Exception as e:
        assert "multiple_matches" in str(e)
    
    # Test with empty list
    field = OneOf(one_of=[])
    try:
        field.validate("anything")
        assert False, "Should have raised validation error"
    except Exception as e:
        assert "no_match" in str(e)
    
    # Test that allow_null is not allowed
    try:
        OneOf(one_of=[Integer()], allow_null=True)
        assert False, "Should have raised assertion error"
    except AssertionError:
        pass


# LLM-generated content at query #62
#--------------------------

```python
def test_NeverMatch():
    # Test that NeverMatch cannot be instantiated with allow_null
    with pytest.raises(AssertionError):
        NeverMatch(allow_null=True)
    
    # Test that NeverMatch can be instantiated without allow_null
    field = NeverMatch()
    assert isinstance(field, NeverMatch)
    assert isinstance(field, Field)
    
    # Test that validate always raises validation error
    with pytest.raises(field.validation_error("never")):
        field.validate(None)
    
    with pytest.raises(field.validation_error("never")):
        field.validate("any value")
    
    with pytest.raises(field.validation_error("never")):
        field.validate(123)
    
    with pytest.raises(field.validation_error("never")):
        field.validate({"key": "value"})


# LLM-generated content at query #63
#--------------------------

```python
def test_AllOf():
    from typesystem.fields import Integer, String, Boolean
    
    # Test basic initialization
    integer_field = Integer()
    string_field = String()
    all_of = AllOf(all_of=[integer_field, string_field])
    
    assert all_of.all_of == [integer_field, string_field]
    assert all_of.allow_null is False
    
    # Test that allow_null cannot be passed in kwargs
    try:
        AllOf(all_of=[integer_field], allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test with empty list
    all_of_empty = AllOf(all_of=[])
    assert all_of_empty.all_of == []
    
    # Test with single field
    single_field = AllOf(all_of=[Boolean()])
    assert len(single_field.all_of) == 1
    assert isinstance(single_field.all_of[0], Boolean)
    
    # Test that parent constructor is called properly
    all_of_with_title = AllOf(all_of=[integer_field], title="Test AllOf")
    assert all_of_with_title.title == "Test AllOf"


# LLM-generated content at query #64
#--------------------------

```python
def test_OneOf():
    from typesystem.fields import String, Integer, Boolean
    
    # Test basic initialization
    string_field = String()
    integer_field = Integer()
    one_of = OneOf(one_of=[string_field, integer_field])
    
    assert one_of.one_of == [string_field, integer_field]
    assert one_of.errors == {
        "no_match": "Did not match any valid type.",
        "multiple_matches": "Matched more than one type."
    }
    
    # Test validation with single match
    result = one_of.validate("hello")
    assert result == "hello"
    
    result = one_of.validate(123)
    assert result == 123
    
    # Test validation with no match
    try:
        one_of.validate(True)
        assert False, "Should have raised validation error"
    except Exception as e:
        assert "no_match" in str(e)
    
    # Test validation with multiple matches (using fields that both accept the same value)
    bool_field = Boolean()
    any_field = Any()
    one_of_multi = OneOf(one_of=[bool_field, any_field])
    
    try:
        one_of_multi.validate(True)
        assert False, "Should have raised validation error"
    except Exception as e:
        assert "multiple_matches" in str(e)
    
    # Test with empty list
    one_of_empty = OneOf(one_of=[])
    try:
        one_of_empty.validate("anything")
        assert False, "Should have raised validation error"
    except Exception as e:
        assert "no_match" in str(e)
    
    # Test that allow_null is not allowed in kwargs
    try:
        OneOf(one_of=[], allow_null=True)
        assert False, "Should have raised assertion error"
    except AssertionError:
        pass


# LLM-generated content at query #65
#--------------------------

```python
def test_AllOf():
    from typesystem.fields import Integer, String
    
    # Test basic initialization
    int_field = Integer()
    str_field = String()
    all_of = AllOf([int_field, str_field])
    
    assert all_of.all_of == [int_field, str_field]
    assert all_of.allow_null is False
    
    # Test that allow_null cannot be passed
    try:
        AllOf([int_field], allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test with empty list
    all_of_empty = AllOf([])
    assert all_of_empty.all_of == []
    
    # Test validation with multiple fields
    all_of = AllOf([Integer(minimum=5), Integer(maximum=10)])
    
    # Should validate successfully
    result = all_of.validate(7)
    assert result == 7
    
    # Should fail validation (less than minimum)
    try:
        all_of.validate(3)
        assert False, "Should have raised ValidationError"
    except Exception as e:
        assert "minimum" in str(e)
    
    # Should fail validation (greater than maximum)
    try:
        all_of.validate(12)
        assert False, "Should have raised ValidationError"
    except Exception as e:
        assert "maximum" in str(e)
    
    # Test with different field types
    all_of_mixed = AllOf([Integer(), String(max_length=5)])
    
    # Integer validation should work
    result = all_of_mixed.validate(42)
    assert result == 42
    
    # String validation should work
    result = all_of_mixed.validate("test")
    assert result == "test"
    
    # String too long should fail
    try:
        all_of_mixed.validate("toolong")
        assert False, "Should have raised ValidationError"
    except Exception as e:
        assert "max_length" in str(e)


# LLM-generated content at query #66
#--------------------------

```python
def test_OneOf():
    # Test basic initialization
    field1 = Any()
    field2 = Any()
    one_of = OneOf(one_of=[field1, field2])
    assert one_of.one_of == [field1, field2]
    assert one_of.errors == {
        "no_match": "Did not match any valid type.",
        "multiple_matches": "Matched more than one type.",
    }

    # Test that allow_null cannot be passed
    try:
        OneOf(one_of=[field1], allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass

    # Test validation with exactly one match
    class AlwaysMatch(Field):
        def validate(self, value):
            return value

    class NeverMatchField(Field):
        def validate(self, value):
            raise self.validation_error("test")

    always_match = AlwaysMatch()
    never_match = NeverMatchField()
    one_of = OneOf(one_of=[always_match, never_match])
    
    # Should match the always_match field
    result = one_of.validate("test_value")
    assert result == "test_value"

    # Test validation with no matches
    one_of = OneOf(one_of=[never_match, never_match])
    try:
        one_of.validate("test_value")
        assert False, "Should have raised validation error"
    except Exception as e:
        assert str(e) == "Did not match any valid type."

    # Test validation with multiple matches
    class AlwaysMatch2(AlwaysMatch):
        pass

    always_match1 = AlwaysMatch()
    always_match2 = AlwaysMatch2()
    one_of = OneOf(one_of=[always_match1, always_match2])
    
    try:
        one_of.validate("test_value")
        assert False, "Should have raised validation error"
    except Exception as e:
        assert str(e) == "Matched more than one type."

    # Test with empty list
    one_of = OneOf(one_of=[])
    try:
        one_of.validate("anything")
        assert False, "Should have raised validation error"
    except Exception as e:
        assert str(e) == "Did not match any valid type."


# LLM-generated content at query #67
#--------------------------

```python
def test_Not():
    # Test basic initialization
    negated_field = Any()
    not_field = Not(negated=negated_field)
    assert not_field.negated == negated_field
    assert not_field.errors == {"negated": "Must not match."}

    # Test that allow_null cannot be passed
    try:
        Not(negated=negated_field, allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass

    # Test validation when negated field fails (should succeed)
    negated_field = NeverMatch()
    not_field = Not(negated=negated_field)
    result = not_field.validate("any value")
    assert result == "any value"

    # Test validation when negated field succeeds (should fail)
    negated_field = Any()
    not_field = Not(negated=negated_field)
    try:
        not_field.validate("any value")
        assert False, "Should have raised ValidationError"
    except Exception as e:
        assert str(e) == "Must not match."

    # Test with more complex negated field
    from typesystem.fields import Integer
    negated_field = Integer()
    not_field = Not(negated=negated_field)
    
    # String should pass (not an integer)
    result = not_field.validate("not a number")
    assert result == "not a number"
    
    # Integer should fail (is an integer)
    try:
        not_field.validate(42)
        assert False, "Should have raised ValidationError"
    except Exception as e:
        assert str(e) == "Must not match."


# LLM-generated content at query #68
#--------------------------

```python
def test_OneOf():
    from typesystem.fields import Integer, String, Boolean
    
    # Test basic initialization
    field = OneOf(one_of=[Integer(), String()])
    assert field.one_of[0].__class__.__name__ == "Integer"
    assert field.one_of[1].__class__.__name__ == "String"
    
    # Test validation with single match
    field = OneOf(one_of=[Integer(), String()])
    result = field.validate(42)
    assert result == 42
    
    # Test validation with no matches
    field = OneOf(one_of=[Integer(), String()])
    try:
        field.validate(True)
        assert False, "Should have raised validation error"
    except Exception as e:
        assert "no_match" in str(e)
    
    # Test validation with multiple matches
    field = OneOf(one_of=[Integer(), Any()])
    try:
        field.validate(42)
        assert False, "Should have raised validation error"
    except Exception as e:
        assert "multiple_matches" in str(e)
    
    # Test with empty list
    field = OneOf(one_of=[])
    try:
        field.validate("anything")
        assert False, "Should have raised validation error"
    except Exception as e:
        assert "no_match" in str(e)
    
    # Test that allow_null is not allowed
    try:
        field = OneOf(one_of=[Integer()], allow_null=True)
        assert False, "Should have raised assertion error"
    except AssertionError:
        pass


# LLM-generated content at query #69
#--------------------------

```python
def test_AllOf():
    from typesystem.fields import String, Integer, Boolean
    
    # Test basic initialization
    string_field = String()
    integer_field = Integer()
    all_of = AllOf([string_field, integer_field])
    assert all_of.all_of == [string_field, integer_field]
    
    # Test that allow_null cannot be passed
    try:
        AllOf([string_field], allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test with empty list
    all_of_empty = AllOf([])
    assert all_of_empty.all_of == []
    
    # Test validation with multiple fields
    all_of_mixed = AllOf([String(max_length=5), String(pattern="^[a-z]+$")])
    result = all_of_mixed.validate("test")
    assert result == "test"
    
    # Test validation failure
    try:
        all_of_mixed.validate("TEST")
        assert False, "Should have raised ValidationError"
    except Exception as e:
        assert hasattr(e, 'error')
    
    # Test that all fields must pass
    complex_all_of = AllOf([Integer(minimum=0), Integer(maximum=10)])
    result = complex_all_of.validate(5)
    assert result == 5
    
    try:
        complex_all_of.validate(-1)
        assert False, "Should have raised ValidationError"
    except Exception:
        pass
    
    try:
        complex_all_of.validate(11)
        assert False, "Should have raised ValidationError"
    except Exception:
        pass


# LLM-generated content at query #70
#--------------------------

```python
def test_AllOf():
    # Test basic initialization
    field1 = Any()
    field2 = Any()
    all_of = AllOf([field1, field2])
    assert all_of.all_of == [field1, field2]
    
    # Test that allow_null cannot be passed
    try:
        AllOf([field1], allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test with empty list
    all_of_empty = AllOf([])
    assert all_of_empty.all_of == []
    
    # Test validation passes when all fields validate
    class AlwaysValid(Field):
        def validate(self, value):
            return value
    
    valid_field1 = AlwaysValid()
    valid_field2 = AlwaysValid()
    all_of_valid = AllOf([valid_field1, valid_field2])
    result = all_of_valid.validate("test_value")
    assert result == "test_value"
    
    # Test validation fails when any field fails
    class FailingField(Field):
        def validate(self, value):
            raise self.validation_error("test_error")
    
    failing_field = FailingField()
    all_of_mixed = AllOf([valid_field1, failing_field])
    try:
        all_of_mixed.validate("test_value")
        assert False, "Should have raised ValidationError"
    except Exception as e:
        assert hasattr(e, 'code')
    
    # Test that original value is returned when all validations pass
    test_value = {"key": "value"}
    result = all_of_valid.validate(test_value)
    assert result == test_value


# LLM-generated content at query #71
#--------------------------

```python
def test_NeverMatch():
    field = NeverMatch()
    assert isinstance(field, Field)
    assert field.errors == {"never": "This never validates."}
    
    with pytest.raises(AssertionError):
        NeverMatch(allow_null=True)
    
    with pytest.raises(field.validation_error("never")):
        field.validate("any value")
    
    with pytest.raises(field.validation_error("never")):
        field.validate(None)
    
    with pytest.raises(field.validation_error("never")):
        field.validate(123)
    
    with pytest.raises(field.validation_error("never")):
        field.validate([])


# LLM-generated content at query #72
#--------------------------

```python
def test_OneOf():
    from typesystem.fields import Integer, String, Boolean
    
    # Test basic initialization
    field = OneOf(one_of=[Integer(), String()])
    assert field.one_of[0].__class__.__name__ == "Integer"
    assert field.one_of[1].__class__.__name__ == "String"
    
    # Test that allow_null cannot be passed
    try:
        OneOf(one_of=[Integer()], allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test validation with exactly one match
    field = OneOf(one_of=[Integer(), String()])
    result = field.validate(42)
    assert result == 42
    
    # Test validation with no matches
    field = OneOf(one_of=[Integer(), String()])
    try:
        field.validate(True)
        assert False, "Should have raised validation error"
    except Exception as e:
        assert "no_match" in str(e)
    
    # Test validation with multiple matches (ambiguous case)
    field = OneOf(one_of=[Any(), Any()])
    try:
        field.validate("test")
        assert False, "Should have raised validation error"
    except Exception as e:
        assert "multiple_matches" in str(e)
    
    # Test with empty one_of list
    field = OneOf(one_of=[])
    try:
        field.validate("anything")
        assert False, "Should have raised validation error"
    except Exception as e:
        assert "no_match" in str(e)


# LLM-generated content at query #73
#--------------------------

```python
def test_Not():
    # Test basic initialization
    negated_field = Any()
    not_field = Not(negated=negated_field)
    assert not_field.negated == negated_field
    assert not_field.errors == {"negated": "Must not match."}

    # Test that allow_null cannot be passed
    try:
        Not(negated=negated_field, allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass

    # Test validation when negated field fails (should succeed)
    result = not_field.validate("test_value")
    assert result == "test_value"

    # Test validation when negated field succeeds (should fail)
    always_valid_field = Any()
    not_always_valid = Not(negated=always_valid_field)
    try:
        not_always_valid.validate("any_value")
        assert False, "Should have raised validation error"
    except Exception as e:
        assert "negated" in str(e)

    # Test with a more complex negated field
    from typesystem import Integer
    not_integer = Not(negated=Integer())
    
    # String should pass (not an integer)
    result = not_integer.validate("not a number")
    assert result == "not a number"
    
    # Integer should fail (is an integer)
    try:
        not_integer.validate(42)
        assert False, "Should have raised validation error"
    except Exception as e:
        assert "negated" in str(e)


# LLM-generated content at query #74
#--------------------------

```python
def test_NeverMatch():
    field = NeverMatch()
    assert isinstance(field, NeverMatch)
    assert field.errors == {"never": "This never validates."}
    
    with pytest.raises(AssertionError):
        NeverMatch(allow_null=True)
    
    with pytest.raises(field.validation_error):
        field.validate("any value")
    
    with pytest.raises(field.validation_error):
        field.validate(None)
    
    with pytest.raises(field.validation_error):
        field.validate(123)
    
    with pytest.raises(field.validation_error):
        field.validate([])


# LLM-generated content at query #75
#--------------------------

```python
def test_AllOf():
    # Test basic initialization
    field1 = Any()
    field2 = Any()
    all_of_field = AllOf([field1, field2])
    assert all_of_field.all_of == [field1, field2]
    
    # Test that allow_null cannot be passed
    try:
        AllOf([field1], allow_null=True)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    
    # Test with empty list
    all_of_empty = AllOf([])
    assert all_of_empty.all_of == []
    
    # Test validation passes when all fields validate
    class AlwaysPass(Field):
        def validate(self, value):
            return value
    
    pass1 = AlwaysPass()
    pass2 = AlwaysPass()
    all_of_pass = AllOf([pass1, pass2])
    result = all_of_pass.validate("test_value")
    assert result == "test_value"
    
    # Test validation fails when any field fails
    class AlwaysFail(Field):
        def validate(self, value):
            raise self.validation_error("test_error")
    
    pass_field = AlwaysPass()
    fail_field = AlwaysFail()
    all_of_mixed = AllOf([pass_field, fail_field])
    
    try:
        all_of_mixed.validate("test_value")
        assert False, "Should have raised ValidationError"
    except Exception as e:
        assert hasattr(e, 'code')
        assert e.code == "test_error"


