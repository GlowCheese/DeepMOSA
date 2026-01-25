####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_if_then_else_constructor_with_only_if_clause():
    from typesystem.fields import Any
    from typesystem.composites import IfThenElse
    if_clause = Any()
    field = IfThenElse(if_clause=if_clause)
    assert field.if_clause is if_clause
    assert isinstance(field.then_clause, Any)
    assert isinstance(field.else_clause, Any)

def test_if_then_else_constructor_with_then_clause():
    from typesystem.fields import Any
    from typesystem.composites import IfThenElse
    if_clause = Any()
    then_clause = Any()
    field = IfThenElse(if_clause=if_clause, then_clause=then_clause)
    assert field.if_clause is if_clause
    assert field.then_clause is then_clause
    assert isinstance(field.else_clause, Any)

def test_if_then_else_constructor_with_else_clause():
    from typesystem.fields import Any
    from typesystem.composites import IfThenElse
    if_clause = Any()
    else_clause = Any()
    field = IfThenElse(if_clause=if_clause, else_clause=else_clause)
    assert field.if_clause is if_clause
    assert isinstance(field.then_clause, Any)
    assert field.else_clause is else_clause

def test_if_then_else_constructor_with_both_clauses():
    from typesystem.fields import Any
    from typesystem.composites import IfThenElse
    if_clause = Any()
    then_clause = Any()
    else_clause = Any()
    field = IfThenElse(if_clause=if_clause, then_clause=then_clause, else_clause=else_clause)
    assert field.if_clause is if_clause
    assert field.then_clause is then_clause
    assert field.else_clause is else_clause

def test_if_then_else_constructor_with_allow_null_not_allowed():
    from typesystem.fields import Any
    from typesystem.composites import IfThenElse
    if_clause = Any()
    try:
        field = IfThenElse(if_clause=if_clause, allow_null=True)
        assert False
    except TypeError:
        pass


# LLM-generated content at query #2
#--------------------------

def test_allof_constructor_with_valid_fields():
    field1 = Field()
    field2 = Field()
    all_of = AllOf(all_of=[field1, field2])
    assert all_of.all_of == [field1, field2]

def test_allof_constructor_with_single_field():
    field = Field()
    all_of = AllOf(all_of=[field])
    assert all_of.all_of == [field]

def test_allof_constructor_with_empty_list():
    all_of = AllOf(all_of=[])
    assert all_of.all_of == []

def test_allof_constructor_with_allow_null_not_in_kwargs():
    field = Field()
    all_of = AllOf(all_of=[field])
    assert all_of.allow_null == False

def test_allof_constructor_with_kwargs_passed_to_super():
    field = Field()
    all_of = AllOf(all_of=[field], title="Test", description="Test description")
    assert all_of.title == "Test"
    assert all_of.description == "Test description"

def test_allof_constructor_with_read_only_kwarg():
    field = Field()
    all_of = AllOf(all_of=[field], read_only=True)
    assert all_of.read_only == True

def test_allof_constructor_with_default_kwarg():
    field = Field()
    all_of = AllOf(all_of=[field], default="default")
    assert all_of.default == "default"

def test_allof_constructor_with_callable_default():
    field = Field()
    all_of = AllOf(all_of=[field], default=lambda: "callable")
    assert callable(all_of.default)


# LLM-generated content at query #3
#--------------------------

def test_allof_constructor_with_valid_fields():
    field1 = Field()
    field2 = Field()
    all_of = AllOf(all_of=[field1, field2])
    assert all_of.all_of == [field1, field2]

def test_allof_constructor_with_single_field():
    field = Field()
    all_of = AllOf(all_of=[field])
    assert all_of.all_of == [field]

def test_allof_constructor_with_empty_list():
    all_of = AllOf(all_of=[])
    assert all_of.all_of == []

def test_allof_constructor_inherits_allow_null_from_field():
    field1 = Field(allow_null=True)
    field2 = Field(allow_null=False)
    all_of = AllOf(all_of=[field1, field2])
    assert all_of.allow_null == False

def test_allof_constructor_with_kwargs():
    field = Field()
    all_of = AllOf(all_of=[field], title="Test", description="Description")
    assert all_of.title == "Test"
    assert all_of.description == "Description"

def test_allof_constructor_allow_null_not_in_kwargs():
    field = Field()
    all_of = AllOf(all_of=[field])
    assert all_of.allow_null == False

def test_allof_constructor_with_union_field():
    field1 = Field()
    field2 = Field()
    union_field = field1 | field2
    all_of = AllOf(all_of=[union_field])
    assert all_of.all_of == [union_field]

def test_allof_constructor_with_read_only():
    field = Field(read_only=True)
    all_of = AllOf(all_of=[field], read_only=True)
    assert all_of.read_only == True

def test_allof_constructor_with_default():
    field = Field()
    all_of = AllOf(all_of=[field], default="default")
    assert all_of.default == "default"

def test_allof_constructor_with_callable_default():
    field = Field()
    all_of = AllOf(all_of=[field], default=lambda: "callable")
    assert callable(all_of.default)


# LLM-generated content at query #4
#--------------------------

def test_validate_returns_value_when_negated_field_has_error():
    mock_negated = Mock()
    mock_negated.validate_or_error.return_value = (None, "error")
    not_field = Not(negated=mock_negated)
    result = not_field.validate("test_value")
    assert result == "test_value"

def test_validate_raises_validation_error_when_negated_field_has_no_error():
    mock_negated = Mock()
    mock_negated.validate_or_error.return_value = ("valid", None)
    not_field = Not(negated=mock_negated)
    try:
        not_field.validate("test_value")
        assert False
    except ValidationError as e:
        assert e.code == "negated"


# LLM-generated content at query #5
#--------------------------

def test_not_constructor_with_valid_field():
    field = Integer()
    not_field = Not(negated=field)
    assert not_field.negated is field
    assert not_field.allow_null is False
    assert not_field.read_only is False
    assert not_field.title == ""
    assert not_field.description == ""

def test_not_constructor_with_title_and_description():
    field = Integer()
    not_field = Not(negated=field, title="Test Title", description="Test Description")
    assert not_field.negated is field
    assert not_field.title == "Test Title"
    assert not_field.description == "Test Description"

def test_not_constructor_with_read_only():
    field = Integer()
    not_field = Not(negated=field, read_only=True)
    assert not_field.negated is field
    assert not_field.read_only is True

def test_not_constructor_rejects_allow_null():
    field = Integer()
    try:
        Not(negated=field, allow_null=True)
        assert False
    except AssertionError:
        pass

def test_not_constructor_with_union_field():
    field = Integer() | String()
    not_field = Not(negated=field)
    assert not_field.negated is field

def test_not_constructor_with_not_field():
    inner_field = Integer()
    nested_not = Not(negated=inner_field)
    not_field = Not(negated=nested_not)
    assert not_field.negated is nested_not


# LLM-generated content at query #6
#--------------------------

def test_nevermatch_constructor_without_allow_null():
    field = NeverMatch()
    assert field.allow_null == False
    assert field.read_only == False
    assert field.title == ""
    assert field.description == ""

def test_nevermatch_constructor_with_title_and_description():
    field = NeverMatch(title="Test Title", description="Test Description")
    assert field.title == "Test Title"
    assert field.description == "Test Description"

def test_nevermatch_constructor_with_read_only():
    field = NeverMatch(read_only=True)
    assert field.read_only == True

def test_nevermatch_constructor_raises_assertion_error_on_allow_null():
    try:
        NeverMatch(allow_null=True)
        assert False
    except AssertionError:
        pass

def test_nevermatch_constructor_with_default():
    field = NeverMatch(default="default_value")
    assert field.has_default() == True
    assert field.get_default_value() == "default_value"

def test_nevermatch_constructor_with_callable_default():
    field = NeverMatch(default=lambda: "callable_default")
    assert field.has_default() == True
    assert field.get_default_value() == "callable_default"


# LLM-generated content at query #7
#--------------------------

def test_oneof_constructor_with_valid_fields():
    field1 = Field()
    field2 = Field()
    one_of = OneOf(one_of=[field1, field2])
    assert one_of.one_of == [field1, field2]
    assert one_of.allow_null == False

def test_oneof_constructor_with_title_and_description():
    field1 = Field()
    one_of = OneOf(one_of=[field1], title="Test Title", description="Test Description")
    assert one_of.title == "Test Title"
    assert one_of.description == "Test Description"

def test_oneof_constructor_with_read_only():
    field1 = Field()
    one_of = OneOf(one_of=[field1], read_only=True)
    assert one_of.read_only == True

def test_oneof_constructor_with_default():
    field1 = Field()
    one_of = OneOf(one_of=[field1], default="default_value")
    assert one_of.has_default() == True
    assert one_of.get_default_value() == "default_value"

def test_oneof_constructor_with_callable_default():
    field1 = Field()
    one_of = OneOf(one_of=[field1], default=lambda: "callable_default")
    assert one_of.has_default() == True
    assert one_of.get_default_value() == "callable_default"

def test_oneof_constructor_allow_null_not_allowed():
    field1 = Field()
    try:
        one_of = OneOf(one_of=[field1], allow_null=True)
        assert False
    except AssertionError:
        pass


# LLM-generated content at query #8
#--------------------------

def test_validate_matches_exactly_one():
    field1 = Field()
    field1.validate = lambda x: (x, None) if isinstance(x, int) else (None, "error")
    field2 = Field()
    field2.validate = lambda x: (x, None) if isinstance(x, str) else (None, "error")
    one_of = OneOf([field1, field2])
    result = one_of.validate(5)
    assert result == 5

def test_validate_no_match():
    field1 = Field()
    field1.validate = lambda x: (None, "error") if not isinstance(x, int) else (x, None)
    field2 = Field()
    field2.validate = lambda x: (None, "error") if not isinstance(x, str) else (x, None)
    one_of = OneOf([field1, field2])
    try:
        one_of.validate(5.0)
        assert False
    except ValidationError as e:
        assert e.code == "no_match"

def test_validate_multiple_matches():
    field1 = Field()
    field1.validate = lambda x: (x, None) if isinstance(x, (int, float)) else (None, "error")
    field2 = Field()
    field2.validate = lambda x: (x, None) if isinstance(x, (float, int)) else (None, "error")
    one_of = OneOf([field1, field2])
    try:
        one_of.validate(5.0)
        assert False
    except ValidationError as e:
        assert e.code == "multiple_matches"

def test_validate_with_null_not_allowed():
    field1 = Field()
    field1.validate = lambda x: (x, None) if x is None else (None, "error")
    field2 = Field()
    field2.validate = lambda x: (x, None) if isinstance(x, str) else (None, "error")
    one_of = OneOf([field1, field2])
    result = one_of.validate(None)
    assert result is None

def test_validate_empty_one_of_list():
    one_of = OneOf([])
    try:
        one_of.validate("anything")
        assert False
    except ValidationError as e:
        assert e.code == "no_match"

def test_validate_single_field_list():
    field = Field()
    field.validate = lambda x: (x.upper(), None) if isinstance(x, str) else (None, "error")
    one_of = OneOf([field])
    result = one_of.validate("test")
    assert result == "TEST"

def test_validate_returns_validated_value():
    field1 = Field()
    field1.validate = lambda x: (x * 2, None) if isinstance(x, int) else (None, "error")
    field2 = Field()
    field2.validate = lambda x: (x + " processed", None) if isinstance(x, str) else (None, "error")
    one_of = OneOf([field1, field2])
    result = one_of.validate(3)
    assert result == 6

def test_validate_nested_one_of():
    inner_field1 = Field()
    inner_field1.validate = lambda x: (x, None) if x == "inner1" else (None, "error")
    inner_field2 = Field()
    inner_field2.validate = lambda x: (x, None) if x == "inner2" else (None, "error")
    inner_one_of = OneOf([inner_field1, inner_field2])
    outer_field = Field()
    outer_field.validate = lambda x: (x, None) if isinstance(x, int) else (None, "error")
    one_of = OneOf([inner_one_of, outer_field])
    result = one_of.validate("inner1")
    assert result == "inner1"


# LLM-generated content at query #9
#--------------------------

def test_match_count_equals_one_returns_candidate():
    from my_module import OneOf, Field
    class AlwaysValidField(Field):
        def validate(self, value):
            return value
        def validate_or_error(self, value):
            return value, None
    field = OneOf([AlwaysValidField()])
    result = field.validate("test")
    assert result == "test"


# LLM-generated content at query #10
#--------------------------

def test_validate_returns_value_when_negated_field_has_error():
    from my_module import Not, Field
    class MockField(Field):
        def validate_or_error(self, value):
            return None, "error"
    negated_field = MockField()
    not_field = Not(negated_field)
    result = not_field.validate("test_value")
    assert result == "test_value"

def test_validate_raises_error_when_negated_field_has_no_error():
    from my_module import Not, Field
    class MockField(Field):
        def validate_or_error(self, value):
            return value, None
    negated_field = MockField()
    not_field = Not(negated_field)
    try:
        not_field.validate("test_value")
        assert False
    except Exception as e:
        assert str(e) == "Must not match."


# LLM-generated content at query #11
#--------------------------

def test_match_count_equals_one_returns_candidate():
    from my_module import OneOf, Field
    class AlwaysValidField(Field):
        def validate(self, value):
            return value
        def validate_or_error(self, value):
            return value, None
    field1 = AlwaysValidField()
    field2 = AlwaysValidField()
    one_of = OneOf([field1, field2])
    result = one_of.validate("test_value")
    assert result == "test_value"


# LLM-generated content at query #12
#--------------------------

def test_validate_returns_value_when_negated_field_has_error():
    mock_negated = Mock()
    mock_negated.validate_or_error.return_value = (None, "some error")
    not_field = Not(negated=mock_negated)
    result = not_field.validate("test_value")
    assert result == "test_value"

def test_validate_raises_validation_error_when_negated_field_has_no_error():
    mock_negated = Mock()
    mock_negated.validate_or_error.return_value = ("valid", None)
    not_field = Not(negated=mock_negated)
    try:
        not_field.validate("test_value")
        assert False
    except ValidationError as e:
        assert e.code == "negated"


# LLM-generated content at query #13
#--------------------------

def test_allof_constructor_with_valid_fields():
    field1 = Field()
    field2 = Field()
    all_of = AllOf(all_of=[field1, field2])
    assert all_of.all_of == [field1, field2]
    assert all_of.allow_null == False

def test_allof_constructor_with_single_field():
    field = Field()
    all_of = AllOf(all_of=[field])
    assert all_of.all_of == [field]

def test_allof_constructor_with_multiple_fields():
    field1 = Field()
    field2 = Field()
    field3 = Field()
    all_of = AllOf(all_of=[field1, field2, field3])
    assert all_of.all_of == [field1, field2, field3]

def test_allof_constructor_with_empty_list():
    all_of = AllOf(all_of=[])
    assert all_of.all_of == []

def test_allof_constructor_allow_null_not_allowed():
    try:
        AllOf(all_of=[], allow_null=True)
        assert False
    except AssertionError:
        pass

def test_allof_constructor_inherits_field_properties():
    field = Field(title="test", description="desc", read_only=True)
    all_of = AllOf(all_of=[field])
    assert all_of.title == ""
    assert all_of.description == ""
    assert all_of.read_only == False

def test_allof_constructor_with_union_field():
    field1 = Field()
    field2 = Field()
    union = field1 | field2
    all_of = AllOf(all_of=[union])
    assert all_of.all_of == [union]


# LLM-generated content at query #14
#--------------------------

def test_validate_returns_value_when_negated_field_has_error():
    mock_negated = Mock()
    mock_negated.validate_or_error.return_value = (None, "some error")
    not_field = Not(negated=mock_negated)
    result = not_field.validate("test_value")
    assert result == "test_value"

def test_validate_raises_validation_error_when_negated_field_has_no_error():
    mock_negated = Mock()
    mock_negated.validate_or_error.return_value = ("valid", None)
    not_field = Not(negated=mock_negated)
    not_field.validation_error = Mock()
    not_field.validation_error.return_value = Exception("Must not match.")
    try:
        not_field.validate("test_value")
        assert False
    except Exception as e:
        assert str(e) == "Must not match."


# LLM-generated content at query #15
#--------------------------

def test_match_count_equals_one_returns_candidate():
    class MockField:
        def validate_or_error(self, value):
            return value, None
    mock_field = MockField()
    one_of_field = OneOf([mock_field])
    result = one_of_field.validate("test")
    assert result == "test"

def test_match_count_equals_one_with_multiple_fields_only_one_matches():
    class MockFieldMatch:
        def validate_or_error(self, value):
            return value, None
    class MockFieldNoMatch:
        def validate_or_error(self, value):
            return None, "error"
    mock_field_match = MockFieldMatch()
    mock_field_no_match = MockFieldNoMatch()
    one_of_field = OneOf([mock_field_no_match, mock_field_match])
    result = one_of_field.validate("test")
    assert result == "test"

def test_match_count_equals_one_with_three_fields_only_one_matches():
    class MockFieldMatch:
        def validate_or_error(self, value):
            return value, None
    class MockFieldNoMatch:
        def validate_or_error(self, value):
            return None, "error"
    mock_field_match = MockFieldMatch()
    mock_field_no_match = MockFieldNoMatch()
    one_of_field = OneOf([mock_field_no_match, mock_field_match, mock_field_no_match])
    result = one_of_field.validate("test")
    assert result == "test"


# LLM-generated content at query #16
#--------------------------

def test_validate_one_match():
    from myapp.fields import OneOf, String, Integer
    field = OneOf([String(), Integer()])
    result = field.validate("hello")
    assert result == "hello"

def test_validate_one_match_integer():
    from myapp.fields import OneOf, String, Integer
    field = OneOf([String(), Integer()])
    result = field.validate(42)
    assert result == 42

def test_validate_no_match():
    from myapp.fields import OneOf, String, Integer
    field = OneOf([String(), Integer()])
    try:
        field.validate(3.14)
        assert False
    except Exception as e:
        assert str(e) == "Did not match any valid type."

def test_validate_multiple_matches():
    from myapp.fields import OneOf, String, Integer, Any
    field = OneOf([String(), Any()])
    try:
        field.validate("hello")
        assert False
    except Exception as e:
        assert str(e) == "Matched more than one type."

def test_validate_with_nested_fields():
    from myapp.fields import OneOf, String, Integer, Boolean
    field = OneOf([String(), Integer(), Boolean()])
    result = field.validate(True)
    assert result == True

def test_validate_empty_list():
    from myapp.fields import OneOf, String
    field = OneOf([])
    try:
        field.validate("anything")
        assert False
    except Exception as e:
        assert str(e) == "Did not match any valid type."


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_validate_matches_exactly_one():
    from myapp.fields import String, Integer, OneOf
    field = OneOf([String(), Integer()])
    result = field.validate("hello")
    assert result == "hello"

def test_validate_matches_exactly_one_with_integer():
    from myapp.fields import String, Integer, OneOf
    field = OneOf([String(), Integer()])
    result = field.validate(123)
    assert result == 123

def test_validate_no_match():
    from myapp.fields import String, Integer, OneOf
    field = OneOf([String(), Integer()])
    try:
        field.validate(3.14)
        assert False
    except Exception as e:
        assert str(e) == "Did not match any valid type."

def test_validate_multiple_matches():
    from myapp.fields import String, Integer, OneOf
    class AlwaysMatchField:
        def validate_or_error(self, value):
            return value, None
    field = OneOf([AlwaysMatchField(), AlwaysMatchField()])
    try:
        field.validate("anything")
        assert False
    except Exception as e:
        assert str(e) == "Matched more than one type."

def test_validate_with_nested_fields():
    from myapp.fields import String, Integer, List, OneOf
    field = OneOf([String(), List(Integer())])
    result = field.validate([1, 2, 3])
    assert result == [1, 2, 3]

def test_validate_with_null_not_allowed():
    from myapp.fields import String, Integer, OneOf
    field = OneOf([String(), Integer()])
    try:
        field.validate(None)
        assert False
    except Exception as e:
        assert str(e) == "Did not match any valid type."


# LLM-generated content at query #2
#--------------------------

def test_nevermatch_constructor_without_allow_null():
    field = NeverMatch()
    assert field.allow_null is False

def test_nevermatch_constructor_with_title_and_description():
    field = NeverMatch(title="Title", description="Description")
    assert field.title == "Title"
    assert field.description == "Description"

def test_nevermatch_constructor_raises_assertion_error_on_allow_null():
    try:
        NeverMatch(allow_null=True)
        assert False
    except AssertionError:
        pass

def test_nevermatch_constructor_with_default():
    field = NeverMatch(default="default")
    assert field.has_default() is True
    assert field.get_default_value() == "default"

def test_nevermatch_constructor_with_callable_default():
    field = NeverMatch(default=lambda: "callable")
    assert field.has_default() is True
    assert field.get_default_value() == "callable"

def test_nevermatch_constructor_read_only():
    field = NeverMatch(read_only=True)
    assert field.read_only is True


# LLM-generated content at query #3
#--------------------------

def test_if_then_else_constructor_with_only_if_clause():
    from typesystem.fields import Field
    from typesystem.composites import IfThenElse
    if_clause = Field()
    field = IfThenElse(if_clause=if_clause)
    assert field.if_clause is if_clause
    assert isinstance(field.then_clause, Field)
    assert isinstance(field.else_clause, Field)

def test_if_then_else_constructor_with_then_clause():
    from typesystem.fields import Field
    from typesystem.composites import IfThenElse
    if_clause = Field()
    then_clause = Field()
    field = IfThenElse(if_clause=if_clause, then_clause=then_clause)
    assert field.if_clause is if_clause
    assert field.then_clause is then_clause
    assert isinstance(field.else_clause, Field)

def test_if_then_else_constructor_with_else_clause():
    from typesystem.fields import Field
    from typesystem.composites import IfThenElse
    if_clause = Field()
    else_clause = Field()
    field = IfThenElse(if_clause=if_clause, else_clause=else_clause)
    assert field.if_clause is if_clause
    assert isinstance(field.then_clause, Field)
    assert field.else_clause is else_clause

def test_if_then_else_constructor_with_all_clauses():
    from typesystem.fields import Field
    from typesystem.composites import IfThenElse
    if_clause = Field()
    then_clause = Field()
    else_clause = Field()
    field = IfThenElse(if_clause=if_clause, then_clause=then_clause, else_clause=else_clause)
    assert field.if_clause is if_clause
    assert field.then_clause is then_clause
    assert field.else_clause is else_clause

def test_if_then_else_constructor_allow_null_not_allowed():
    from typesystem.fields import Field
    from typesystem.composites import IfThenElse
    if_clause = Field()
    try:
        field = IfThenElse(if_clause=if_clause, allow_null=True)
        assert False, "Should have raised an exception"
    except AssertionError as e:
        assert "allow_null" in str(e)


# LLM-generated content at query #4
#--------------------------

def test_oneof_constructor_with_valid_arguments():
    field1 = Field()
    field2 = Field()
    one_of = OneOf(one_of=[field1, field2])
    assert one_of.one_of == [field1, field2]
    assert one_of.allow_null == False

def test_oneof_constructor_with_title_and_description():
    field1 = Field()
    one_of = OneOf(one_of=[field1], title="Test Title", description="Test Description")
    assert one_of.title == "Test Title"
    assert one_of.description == "Test Description"

def test_oneof_constructor_allow_null_not_allowed():
    field1 = Field()
    try:
        OneOf(one_of=[field1], allow_null=True)
        assert False
    except AssertionError:
        pass

def test_oneof_constructor_with_empty_list():
    one_of = OneOf(one_of=[])
    assert one_of.one_of == []

def test_oneof_constructor_with_single_field():
    field1 = Field()
    one_of = OneOf(one_of=[field1])
    assert one_of.one_of == [field1]

def test_oneof_constructor_with_multiple_fields():
    field1 = Field()
    field2 = Field()
    field3 = Field()
    one_of = OneOf(one_of=[field1, field2, field3])
    assert one_of.one_of == [field1, field2, field3]


# LLM-generated content at query #5
#--------------------------

def test_predicate_at_line_12_evaluates_to_true():
    from my_module import OneOf, Field
    class AlwaysValid(Field):
        def validate(self, value):
            return value
        def validate_or_error(self, value):
            return value, None
    class NeverValid(Field):
        def validate(self, value):
            raise self.validation_error("error")
        def validate_or_error(self, value):
            return None, self.validation_error("error")
    field1 = AlwaysValid()
    field2 = AlwaysValid()
    one_of = OneOf([field1, field2])
    try:
        one_of.validate("test")
    except Exception as e:
        assert str(e) == "Matched more than one type."
    else:
        assert False, "Expected exception not raised"


# LLM-generated content at query #6
#--------------------------

def test_not_constructor_with_valid_negated_field():
    from typesystem.fields import Field
    from typesystem.composites import Not
    field = Field()
    not_field = Not(negated=field)
    assert not_field.negated is field
    assert not_field.allow_null is False
    assert not_field.read_only is False
    assert not_field.title == ""
    assert not_field.description == ""

def test_not_constructor_with_custom_kwargs():
    from typesystem.fields import Field
    from typesystem.composites import Not
    field = Field()
    not_field = Not(negated=field, title="Title", description="Description", read_only=True)
    assert not_field.negated is field
    assert not_field.allow_null is False
    assert not_field.read_only is True
    assert not_field.title == "Title"
    assert not_field.description == "Description"

def test_not_constructor_rejects_allow_null_kwarg():
    from typesystem.fields import Field
    from typesystem.composites import Not
    field = Field()
    try:
        Not(negated=field, allow_null=True)
        assert False
    except AssertionError:
        pass

def test_not_constructor_with_negated_field_having_allow_null():
    from typesystem.fields import Field
    from typesystem.composites import Not
    field = Field(allow_null=True)
    not_field = Not(negated=field)
    assert not_field.negated is field
    assert not_field.allow_null is False


# LLM-generated content at query #7
#--------------------------

def test_validate_matches_exactly_one():
    from myapp.fields import OneOf, Integer, String
    field = OneOf([Integer(), String()])
    result = field.validate(123)
    assert result == 123

def test_validate_matches_exactly_one_string():
    from myapp.fields import OneOf, Integer, String
    field = OneOf([Integer(), String()])
    result = field.validate("hello")
    assert result == "hello"

def test_validate_no_match():
    from myapp.fields import OneOf, Integer, String
    field = OneOf([Integer(), String()])
    try:
        field.validate(True)
        assert False
    except Exception as e:
        assert str(e) == "Did not match any valid type."

def test_validate_multiple_matches():
    from myapp.fields import OneOf, Integer, Any
    field = OneOf([Integer(), Any()])
    try:
        field.validate(456)
        assert False
    except Exception as e:
        assert str(e) == "Matched more than one type."

def test_validate_with_null_not_allowed():
    from myapp.fields import OneOf, Integer, String
    field = OneOf([Integer(), String()])
    try:
        field.validate(None)
        assert False
    except Exception as e:
        assert str(e) == "Did not match any valid type."

def test_validate_nested_one_of():
    from myapp.fields import OneOf, Integer, String, Boolean
    inner = OneOf([Integer(), String()])
    outer = OneOf([inner, Boolean()])
    result = outer.validate(True)
    assert result == True

def test_validate_nested_one_of_inner_match():
    from myapp.fields import OneOf, Integer, String, Boolean
    inner = OneOf([Integer(), String()])
    outer = OneOf([inner, Boolean()])
    result = outer.validate(789)
    assert result == 789

def test_validate_empty_one_of_list():
    from myapp.fields import OneOf
    field = OneOf([])
    try:
        field.validate("anything")
        assert False
    except Exception as e:
        assert str(e) == "Did not match any valid type."


# LLM-generated content at query #8
#--------------------------

def test_allof_constructor_with_list_of_fields():
    field1 = Field()
    field2 = Field()
    all_of = AllOf(all_of=[field1, field2])
    assert all_of.all_of == [field1, field2]

def test_allof_constructor_with_empty_list():
    all_of = AllOf(all_of=[])
    assert all_of.all_of == []

def test_allof_constructor_with_single_field():
    field = Field()
    all_of = AllOf(all_of=[field])
    assert all_of.all_of == [field]

def test_allof_constructor_with_allow_null_not_in_kwargs():
    field = Field()
    all_of = AllOf(all_of=[field], title="test", description="test description")
    assert all_of.title == "test"
    assert all_of.description == "test description"
    assert all_of.allow_null == False

def test_allof_constructor_inherits_field_defaults():
    field = Field()
    all_of = AllOf(all_of=[field])
    assert all_of.read_only == False
    assert not hasattr(all_of, 'default')

def test_allof_constructor_with_multiple_fields():
    field1 = Field(title="field1")
    field2 = Field(description="field2")
    field3 = Field(allow_null=True)
    all_of = AllOf(all_of=[field1, field2, field3])
    assert len(all_of.all_of) == 3
    assert all_of.all_of[0].title == "field1"
    assert all_of.all_of[1].description == "field2"
    assert all_of.all_of[2].allow_null == True


# LLM-generated content at query #9
#--------------------------

def test_allow_null_not_in_kwargs():
    field = Field()
    AllOf(all_of=[field])


# LLM-generated content at query #10
#--------------------------

def test_allow_null_not_in_kwargs():
    field = Field()
    AllOf(all_of=[field])


# LLM-generated content at query #11
#--------------------------

def test_allow_null_not_in_kwargs():
    from typesystem.fields import Field
    from typesystem.composites import AllOf
    child1 = Field()
    child2 = Field()
    AllOf(all_of=[child1, child2])


# LLM-generated content at query #12
#--------------------------

def test_allof_constructor_with_valid_fields():
    field1 = Field()
    field2 = Field()
    all_of = AllOf(all_of=[field1, field2])
    assert all_of.all_of == [field1, field2]
    assert all_of.allow_null == False

def test_allof_constructor_with_single_field():
    field = Field()
    all_of = AllOf(all_of=[field])
    assert all_of.all_of == [field]

def test_allof_constructor_with_empty_list():
    all_of = AllOf(all_of=[])
    assert all_of.all_of == []

def test_allof_constructor_with_allow_null_keyword():
    field = Field()
    all_of = AllOf(all_of=[field], allow_null=True)
    assert all_of.allow_null == True

def test_allof_constructor_with_title_and_description():
    field = Field()
    all_of = AllOf(all_of=[field], title="Test", description="Test description")
    assert all_of.title == "Test"
    assert all_of.description == "Test description"

def test_allof_constructor_with_default():
    field = Field()
    all_of = AllOf(all_of=[field], default="default")
    assert all_of.has_default() == True
    assert all_of.get_default_value() == "default"

def test_allof_constructor_with_callable_default():
    field = Field()
    all_of = AllOf(all_of=[field], default=lambda: "callable")
    assert all_of.has_default() == True
    assert all_of.get_default_value() == "callable"

def test_allof_constructor_with_read_only():
    field = Field()
    all_of = AllOf(all_of=[field], read_only=True)
    assert all_of.read_only == True


