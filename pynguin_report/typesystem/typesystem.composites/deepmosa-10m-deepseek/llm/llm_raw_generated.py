####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_validate_returns_value_when_negated_field_has_error():
    mock_negated = Mock()
    mock_negated.validate_or_error.return_value = (None, "error")
    not_field = Not(negated=mock_negated)
    result = not_field.validate("test_value")
    assert result == "test_value"

def test_validate_raises_validation_error_when_negated_field_validates():
    mock_negated = Mock()
    mock_negated.validate_or_error.return_value = ("valid", None)
    not_field = Not(negated=mock_negated)
    try:
        not_field.validate("test_value")
        assert False
    except ValidationError as e:
        assert e.code == "negated"


# LLM-generated content at query #2
#--------------------------

def test_oneof_constructor_with_valid_arguments():
    field1 = Field()
    field2 = Field()
    one_of = OneOf(one_of=[field1, field2])
    assert one_of.one_of == [field1, field2]
    assert one_of.allow_null is False
    assert one_of.read_only is False
    assert one_of.title == ""
    assert one_of.description == ""

def test_oneof_constructor_with_title_and_description():
    field1 = Field()
    one_of = OneOf(one_of=[field1], title="Test Title", description="Test Description")
    assert one_of.title == "Test Title"
    assert one_of.description == "Test Description"

def test_oneof_constructor_with_read_only():
    field1 = Field()
    one_of = OneOf(one_of=[field1], read_only=True)
    assert one_of.read_only is True

def test_oneof_constructor_allow_null_not_allowed():
    field1 = Field()
    try:
        OneOf(one_of=[field1], allow_null=True)
    except AssertionError:
        pass
    else:
        assert False, "Expected AssertionError"

def test_oneof_constructor_with_single_field_list():
    field1 = Field()
    one_of = OneOf(one_of=[field1])
    assert len(one_of.one_of) == 1
    assert one_of.one_of[0] is field1

def test_oneof_constructor_with_multiple_fields():
    field1 = Field()
    field2 = Field()
    field3 = Field()
    one_of = OneOf(one_of=[field1, field2, field3])
    assert len(one_of.one_of) == 3
    assert one_of.one_of == [field1, field2, field3]


# LLM-generated content at query #3
#--------------------------

def test_not_constructor_with_valid_negated_field():
    field = Integer()
    not_field = Not(negated=field)
    assert not_field.negated is field
    assert not_field.allow_null is False
    assert not_field.read_only is False
    assert not_field.title == ""
    assert not_field.description == ""

def test_not_constructor_with_custom_kwargs():
    field = String()
    not_field = Not(negated=field, title="Title", description="Description", read_only=True)
    assert not_field.negated is field
    assert not_field.allow_null is False
    assert not_field.read_only is True
    assert not_field.title == "Title"
    assert not_field.description == "Description"

def test_not_constructor_rejects_allow_null_kwarg():
    field = Integer()
    try:
        Not(negated=field, allow_null=True)
        assert False
    except AssertionError:
        pass

def test_not_constructor_with_union_field():
    union_field = Integer() | String()
    not_field = Not(negated=union_field)
    assert not_field.negated is union_field

def test_not_constructor_with_not_field():
    inner_not = Not(negated=Integer())
    outer_not = Not(negated=inner_not)
    assert outer_not.negated is inner_not


# LLM-generated content at query #4
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

def test_allof_constructor_allow_null_not_allowed():
    try:
        AllOf(all_of=[], allow_null=True)
        assert False
    except AssertionError:
        pass

def test_allof_constructor_inherits_field_attributes():
    field1 = Field(title="Title1", description="Desc1", read_only=True)
    field2 = Field(title="Title2", description="Desc2", read_only=False)
    all_of = AllOf(all_of=[field1, field2], title="AllTitle", description="AllDesc", read_only=True)
    assert all_of.title == "AllTitle"
    assert all_of.description == "AllDesc"
    assert all_of.read_only == True

def test_allof_constructor_default_values():
    field = Field()
    all_of = AllOf(all_of=[field])
    assert not hasattr(all_of, 'default')
    assert all_of.allow_null == False
    assert all_of.read_only == False
    assert all_of.title == ""
    assert all_of.description == ""


# LLM-generated content at query #5
#--------------------------

def test_if_then_else_constructor_with_only_if_clause():
    from typesystem.composites import IfThenElse
    from typesystem.fields import Any
    if_clause = Any()
    field = IfThenElse(if_clause=if_clause)
    assert field.if_clause is if_clause
    assert isinstance(field.then_clause, Any)
    assert isinstance(field.else_clause, Any)

def test_if_then_else_constructor_with_all_clauses():
    from typesystem.composites import IfThenElse
    from typesystem.fields import Any
    if_clause = Any()
    then_clause = Any()
    else_clause = Any()
    field = IfThenElse(if_clause=if_clause, then_clause=then_clause, else_clause=else_clause)
    assert field.if_clause is if_clause
    assert field.then_clause is then_clause
    assert field.else_clause is else_clause

def test_if_then_else_constructor_with_then_clause_only():
    from typesystem.composites import IfThenElse
    from typesystem.fields import Any
    if_clause = Any()
    then_clause = Any()
    field = IfThenElse(if_clause=if_clause, then_clause=then_clause)
    assert field.if_clause is if_clause
    assert field.then_clause is then_clause
    assert isinstance(field.else_clause, Any)

def test_if_then_else_constructor_with_else_clause_only():
    from typesystem.composites import IfThenElse
    from typesystem.fields import Any
    if_clause = Any()
    else_clause = Any()
    field = IfThenElse(if_clause=if_clause, else_clause=else_clause)
    assert field.if_clause is if_clause
    assert isinstance(field.then_clause, Any)
    assert field.else_clause is else_clause

def test_if_then_else_constructor_rejects_allow_null():
    from typesystem.composites import IfThenElse
    from typesystem.fields import Any
    if_clause = Any()
    try:
        field = IfThenElse(if_clause=if_clause, allow_null=True)
        assert False
    except TypeError:
        pass


# LLM-generated content at query #6
#--------------------------

def test_nevermatch_constructor_without_allow_null():
    field = NeverMatch()
    assert field.allow_null is False

def test_nevermatch_constructor_with_title_and_description():
    field = NeverMatch(title="Title", description="Description")
    assert field.title == "Title"
    assert field.description == "Description"

def test_nevermatch_constructor_raises_assertion_error_if_allow_null_passed():
    try:
        NeverMatch(allow_null=True)
        assert False
    except AssertionError:
        pass

def test_nevermatch_constructor_sets_default_read_only():
    field = NeverMatch()
    assert field.read_only is False

def test_nevermatch_constructor_sets_read_only():
    field = NeverMatch(read_only=True)
    assert field.read_only is True

def test_nevermatch_constructor_has_no_default_attribute():
    field = NeverMatch()
    assert not hasattr(field, 'default')

def test_nevermatch_constructor_errors_dict():
    field = NeverMatch()
    assert field.errors == {"never": "This never validates."}


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_validate_one_match():
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
        one_of.validate(5.5)
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
    field1.validate = lambda x: (x, None) if x is not None else (None, "error")
    field2 = Field()
    field2.validate = lambda x: (x, None) if isinstance(x, str) else (None, "error")
    one_of = OneOf([field1, field2])
    try:
        one_of.validate(None)
        assert False
    except ValidationError as e:
        assert e.code == "no_match"

def test_validate_single_field_list():
    field = Field()
    field.validate = lambda x: (x, None) if isinstance(x, int) else (None, "error")
    one_of = OneOf([field])
    result = one_of.validate(42)
    assert result == 42

def test_validate_empty_list():
    one_of = OneOf([])
    try:
        one_of.validate("anything")
        assert False
    except ValidationError as e:
        assert e.code == "no_match"

def test_validate_candidate_preserved():
    field1 = Field()
    field1.validate = lambda x: (x * 2, None) if isinstance(x, int) else (None, "error")
    field2 = Field()
    field2.validate = lambda x: (x.upper(), None) if isinstance(x, str) else (None, "error")
    one_of = OneOf([field1, field2])
    result = one_of.validate(21)
    assert result == 42

def test_validate_error_from_child():
    field1 = Field()
    field1.validate = lambda x: (None, "child_error")
    field2 = Field()
    field2.validate = lambda x: (x, None) if isinstance(x, str) else (None, "error")
    one_of = OneOf([field1, field2])
    result = one_of.validate("test")
    assert result == "test"


# LLM-generated content at query #2
#--------------------------

def test_if_then_else_constructor_with_only_if_clause():
    from typesystem.composites import IfThenElse
    from typesystem.fields import Any
    if_clause = Any()
    field = IfThenElse(if_clause=if_clause)
    assert field.if_clause is if_clause
    assert isinstance(field.then_clause, Any)
    assert isinstance(field.else_clause, Any)

def test_if_then_else_constructor_with_then_clause():
    from typesystem.composites import IfThenElse
    from typesystem.fields import Any
    if_clause = Any()
    then_clause = Any()
    field = IfThenElse(if_clause=if_clause, then_clause=then_clause)
    assert field.if_clause is if_clause
    assert field.then_clause is then_clause
    assert isinstance(field.else_clause, Any)

def test_if_then_else_constructor_with_else_clause():
    from typesystem.composites import IfThenElse
    from typesystem.fields import Any
    if_clause = Any()
    else_clause = Any()
    field = IfThenElse(if_clause=if_clause, else_clause=else_clause)
    assert field.if_clause is if_clause
    assert isinstance(field.then_clause, Any)
    assert field.else_clause is else_clause

def test_if_then_else_constructor_with_both_then_and_else_clauses():
    from typesystem.composites import IfThenElse
    from typesystem.fields import Any
    if_clause = Any()
    then_clause = Any()
    else_clause = Any()
    field = IfThenElse(if_clause=if_clause, then_clause=then_clause, else_clause=else_clause)
    assert field.if_clause is if_clause
    assert field.then_clause is then_clause
    assert field.else_clause is else_clause

def test_if_then_else_constructor_with_allow_null_not_allowed():
    from typesystem.composites import IfThenElse
    from typesystem.fields import Any
    if_clause = Any()
    try:
        IfThenElse(if_clause=if_clause, allow_null=True)
        assert False
    except AssertionError:
        pass

def test_if_then_else_constructor_inherits_other_kwargs():
    from typesystem.composites import IfThenElse
    from typesystem.fields import Any
    if_clause = Any()
    field = IfThenElse(if_clause=if_clause, title="Title", description="Description", read_only=True)
    assert field.title == "Title"
    assert field.description == "Description"
    assert field.read_only is True


# LLM-generated content at query #3
#--------------------------

def test_nevermatch_constructor_without_allow_null():
    field = NeverMatch()
    assert field.allow_null == False

def test_nevermatch_constructor_with_title_and_description():
    field = NeverMatch(title="Title", description="Description")
    assert field.title == "Title"
    assert field.description == "Description"

def test_nevermatch_constructor_raises_assertion_error_if_allow_null_passed():
    try:
        NeverMatch(allow_null=True)
        assert False
    except AssertionError:
        pass

def test_nevermatch_constructor_default_values():
    field = NeverMatch()
    assert field.title == ""
    assert field.description == ""
    assert field.allow_null == False
    assert field.read_only == False

def test_nevermatch_constructor_with_read_only():
    field = NeverMatch(read_only=True)
    assert field.read_only == True


# LLM-generated content at query #4
#--------------------------

def test_oneof_constructor_with_valid_fields():
    field1 = Field()
    field2 = Field()
    one_of = OneOf(one_of=[field1, field2])
    assert one_of.one_of == [field1, field2]
    assert one_of.allow_null == False

def test_oneof_constructor_with_allow_null_keyword():
    field1 = Field()
    field2 = Field()
    one_of = OneOf(one_of=[field1, field2], allow_null=True)
    assert one_of.allow_null == True

def test_oneof_constructor_with_title_and_description():
    field1 = Field()
    one_of = OneOf(one_of=[field1], title="Test Title", description="Test Description")
    assert one_of.title == "Test Title"
    assert one_of.description == "Test Description"

def test_oneof_constructor_with_default():
    field1 = Field()
    one_of = OneOf(one_of=[field1], default="default_value")
    assert one_of.default == "default_value"

def test_oneof_constructor_with_read_only():
    field1 = Field()
    one_of = OneOf(one_of=[field1], read_only=True)
    assert one_of.read_only == True

def test_oneof_constructor_with_single_field_list():
    field1 = Field()
    one_of = OneOf(one_of=[field1])
    assert one_of.one_of == [field1]

def test_oneof_constructor_with_empty_list():
    one_of = OneOf(one_of=[])
    assert one_of.one_of == []

def test_oneof_constructor_with_multiple_fields():
    field1 = Field()
    field2 = Field()
    field3 = Field()
    one_of = OneOf(one_of=[field1, field2, field3])
    assert one_of.one_of == [field1, field2, field3]


# LLM-generated content at query #5
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

def test_allof_constructor_passes_kwargs_to_parent():
    all_of = AllOf(all_of=[], title="Test", description="Description")
    assert all_of.title == "Test"
    assert all_of.description == "Description"

def test_allof_constructor_disallows_allow_null_kwarg():
    field = Field()
    try:
        AllOf(all_of=[field], allow_null=True)
        assert False
    except AssertionError:
        pass

def test_allof_constructor_with_multiple_fields():
    field1 = Field(title="Field1")
    field2 = Field(description="Field2")
    field3 = Field()
    all_of = AllOf(all_of=[field1, field2, field3])
    assert len(all_of.all_of) == 3
    assert all_of.all_of[0] == field1
    assert all_of.all_of[1] == field2
    assert all_of.all_of[2] == field3


# LLM-generated content at query #6
#--------------------------

def test_not_constructor_with_valid_arguments():
    from typesystem.composites import Not
    from typesystem.fields import Field
    inner_field = Field()
    not_field = Not(negated=inner_field)
    assert not_field.negated is inner_field
    assert not_field.allow_null is False

def test_not_constructor_with_title_and_description():
    from typesystem.composites import Not
    from typesystem.fields import Field
    inner_field = Field()
    not_field = Not(negated=inner_field, title="Title", description="Description")
    assert not_field.negated is inner_field
    assert not_field.title == "Title"
    assert not_field.description == "Description"

def test_not_constructor_rejects_allow_null():
    from typesystem.composites import Not
    from typesystem.fields import Field
    inner_field = Field()
    try:
        Not(negated=inner_field, allow_null=True)
        assert False
    except AssertionError:
        pass

def test_not_constructor_default_read_only():
    from typesystem.composites import Not
    from typesystem.fields import Field
    inner_field = Field()
    not_field = Not(negated=inner_field)
    assert not_field.read_only is False

def test_not_constructor_with_read_only():
    from typesystem.composites import Not
    from typesystem.fields import Field
    inner_field = Field()
    not_field = Not(negated=inner_field, read_only=True)
    assert not_field.read_only is True


