# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.composites as module_0
import typesystem.fields as module_1

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    var_1 = module_0.NeverMatch()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert module_0.NeverMatch.errors == {'never': 'This never validates.'}
    var_2 = '\x0b]Io'
    var_3 = {var_2: var_0, var_2: var_0}
    module_0.AllOf(var_0, **var_3)

def test_case_1():
    var_0 = []
    var_1 = module_0.OneOf(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.one_of == []
    assert module_0.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    var_1 = '\x0b]Io'
    var_2 = {var_1: var_0, var_1: var_0}
    module_0.AllOf(var_0, **var_2)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = []
    var_1 = module_0.AllOf(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.all_of == []
    var_2 = module_0.NeverMatch()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert module_0.NeverMatch.errors == {'never': 'This never validates.'}
    var_3 = None
    var_4 = module_0.IfThenElse(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.if_clause is None
    assert f'{type(var_4.then_clause).__module__}.{type(var_4.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_4.else_clause).__module__}.{type(var_4.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_5 = var_1.validate(var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    var_6 = var_2.get_default_value()
    var_7 = 'lo(_)D:'
    var_8 = {var_7: var_4, var_7: var_6, var_7: var_4, var_7: var_4, var_7: var_4}
    var_9 = var_2.validate_or_error(var_3)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_9.value is None
    assert f'{type(var_9.error).__module__}.{type(var_9.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_9.error) == 1
    var_10 = module_0.Not(var_3)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.composites.Not'
    assert var_10.title == ''
    assert var_10.description == ''
    assert var_10.allow_null is False
    assert var_10.read_only is False
    assert var_10.negated is None
    assert module_0.Not.errors == {'negated': 'Must not match.'}
    module_0.OneOf(var_3, **var_8)

def test_case_4():
    var_0 = False
    var_1 = module_0.Not(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.composites.Not'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.negated is False
    assert module_0.Not.errors == {'negated': 'Must not match.'}

def test_case_5():
    var_0 = None
    var_1 = module_0.IfThenElse(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.if_clause is None
    assert f'{type(var_1.then_clause).__module__}.{type(var_1.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_1.else_clause).__module__}.{type(var_1.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_2 = var_1.get_default_value()

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = None
    var_1 = module_0.NeverMatch()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert module_0.NeverMatch.errors == {'never': 'This never validates.'}
    var_2 = [var_1, var_1, var_1]
    var_3 = module_0.OneOf(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.one_of).__module__}.{type(var_3.one_of).__qualname__}' == 'builtins.list'
    assert len(var_3.one_of) == 3
    assert module_0.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_3.validate(var_0)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = module_0.NeverMatch()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert module_0.NeverMatch.errors == {'never': 'This never validates.'}
    var_1 = None
    var_2 = module_0.IfThenElse(var_1, var_1, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.if_clause is None
    assert f'{type(var_2.then_clause).__module__}.{type(var_2.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_2.else_clause).__module__}.{type(var_2.else_clause).__qualname__}' == 'typesystem.composites.NeverMatch'
    var_3 = None
    var_4 = module_0.AllOf(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.all_of is None
    var_5 = 'a'
    var_6 = '19I\na7$'
    var_7 = {var_5: var_3, var_6: var_3, var_6: var_3, var_5: var_3}
    module_0.Not(var_3, **var_7)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = module_1.String()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.String'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.allow_blank is False
    assert var_0.trim_whitespace is True
    assert var_0.max_length is None
    assert var_0.min_length is None
    assert var_0.format is None
    assert var_0.coerce_types is True
    assert var_0.pattern is None
    assert var_0.pattern_regex is None
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_1 = module_1.Integer()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Integer'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.minimum is None
    assert var_1.maximum is None
    assert var_1.exclusive_minimum is None
    assert var_1.exclusive_maximum is None
    assert var_1.multiple_of is None
    assert var_1.precision is None
    assert var_1.coerce_types is True
    var_2 = [var_0, var_1]
    var_3 = module_0.OneOf(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.one_of).__module__}.{type(var_3.one_of).__qualname__}' == 'builtins.list'
    assert len(var_3.one_of) == 2
    assert module_0.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_4 = '123'
    var_3.validate(var_4)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = 'Va.-R72Pa'
    var_1 = None
    var_2 = module_1.Any(description=var_0, read_only=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Any'
    assert var_2.title == ''
    assert var_2.description == 'Va.-R72Pa'
    assert var_2.allow_null is False
    assert var_2.read_only is None
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    var_3 = [var_2, var_2]
    var_4 = module_0.AllOf(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.all_of).__module__}.{type(var_4.all_of).__qualname__}' == 'builtins.list'
    assert len(var_4.all_of) == 2
    var_5 = var_4.validate_or_error(var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationResult'
    assert f'{type(var_5.value).__module__}.{type(var_5.value).__qualname__}' == 'typesystem.fields.Any'
    assert var_5.error is None
    var_6 = None
    var_7 = module_0.OneOf(var_3)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert f'{type(var_7.one_of).__module__}.{type(var_7.one_of).__qualname__}' == 'builtins.list'
    assert len(var_7.one_of) == 2
    assert module_0.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_8 = module_0.AllOf(var_6)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert var_8.all_of is None
    var_9 = module_0.Not(var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.composites.Not'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert f'{type(var_9.negated).__module__}.{type(var_9.negated).__qualname__}' == 'typesystem.composites.AllOf'
    assert module_0.Not.errors == {'negated': 'Must not match.'}
    var_10 = var_4.validate(var_0)
    assert var_10 == 'Va.-R72Pa'
    var_11 = None
    var_12 = None
    var_13 = "@0'o:"
    var_14 = {var_0: var_1, var_13: var_1, var_13: var_11, var_13: var_12}
    module_0.IfThenElse(var_8, **var_14)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = None
    var_1 = []
    var_2 = module_0.OneOf(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.one_of == []
    assert module_0.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_2.validate(var_0)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = module_0.NeverMatch()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert module_0.NeverMatch.errors == {'never': 'This never validates.'}
    var_1 = module_0.IfThenElse(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(var_1.if_clause).__module__}.{type(var_1.if_clause).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert f'{type(var_1.then_clause).__module__}.{type(var_1.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_1.else_clause).__module__}.{type(var_1.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_2 = module_0.AllOf(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(var_2.all_of).__module__}.{type(var_2.all_of).__qualname__}' == 'typesystem.composites.NeverMatch'
    var_3 = 'lo(_)ul'
    var_4 = {var_3: var_1, var_3: var_3, var_3: var_2, var_3: var_1, var_3: var_1}
    var_5 = var_1.validate(var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    module_0.Not(var_5, **var_4)

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = module_0.NeverMatch()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert module_0.NeverMatch.errors == {'never': 'This never validates.'}
    var_1 = 'lo(_ ul'
    var_2 = var_0.get_default_value()
    var_3 = module_0.Not(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.composites.Not'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.negated is None
    assert module_0.Not.errors == {'negated': 'Must not match.'}
    var_4 = {var_1: var_2, var_1: var_1, var_1: var_3, var_1: var_2, var_1: var_2}
    var_5 = module_0.IfThenElse(var_2, var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert var_5.if_clause is None
    assert var_5.then_clause == 'lo(_ ul'
    assert f'{type(var_5.else_clause).__module__}.{type(var_5.else_clause).__qualname__}' == 'typesystem.fields.Any'
    module_0.IfThenElse(var_2, var_2, **var_4)

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = 'Va.-R72Pa'
    var_1 = None
    var_2 = module_1.Any(description=var_0, read_only=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Any'
    assert var_2.title == ''
    assert var_2.description == 'Va.-R72Pa'
    assert var_2.allow_null is False
    assert var_2.read_only is None
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    var_3 = [var_2, var_2]
    var_4 = module_0.AllOf(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.all_of).__module__}.{type(var_4.all_of).__qualname__}' == 'builtins.list'
    assert len(var_4.all_of) == 2
    var_5 = var_4.validate_or_error(var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationResult'
    assert f'{type(var_5.value).__module__}.{type(var_5.value).__qualname__}' == 'typesystem.fields.Any'
    assert var_5.error is None
    var_6 = None
    var_7 = module_0.OneOf(var_3)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert f'{type(var_7.one_of).__module__}.{type(var_7.one_of).__qualname__}' == 'builtins.list'
    assert len(var_7.one_of) == 2
    assert module_0.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_8 = module_0.IfThenElse(var_4, var_4, var_1)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert f'{type(var_8.if_clause).__module__}.{type(var_8.if_clause).__qualname__}' == 'typesystem.composites.AllOf'
    assert f'{type(var_8.then_clause).__module__}.{type(var_8.then_clause).__qualname__}' == 'typesystem.composites.AllOf'
    assert f'{type(var_8.else_clause).__module__}.{type(var_8.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_9 = module_0.AllOf(var_6)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert var_9.all_of is None
    var_10 = module_0.Not(var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.composites.Not'
    assert var_10.title == ''
    assert var_10.description == ''
    assert var_10.allow_null is False
    assert var_10.read_only is False
    assert f'{type(var_10.negated).__module__}.{type(var_10.negated).__qualname__}' == 'typesystem.composites.AllOf'
    assert module_0.Not.errors == {'negated': 'Must not match.'}
    var_11 = None
    var_12 = var_8.validate(var_5)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.ValidationResult'
    assert f'{type(var_12.value).__module__}.{type(var_12.value).__qualname__}' == 'typesystem.fields.Any'
    assert var_12.error is None
    var_13 = None
    var_14 = "@0'o:"
    var_15 = {var_0: var_1, var_14: var_1, var_14: var_11, var_14: var_13}
    module_0.IfThenElse(var_9, **var_15)

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = None
    var_1 = '.ZcW'
    var_2 = module_1.Any(title=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Any'
    assert var_2.title == '.ZcW'
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    var_3 = [var_2]
    var_4 = module_0.OneOf(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.one_of).__module__}.{type(var_4.one_of).__qualname__}' == 'builtins.list'
    assert len(var_4.one_of) == 1
    assert module_0.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_5 = var_4.validate(var_0)
    var_5.validate(var_0)

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = None
    var_1 = '.ZcW'
    var_2 = module_1.Any(title=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Any'
    assert var_2.title == '.ZcW'
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    var_3 = [var_2]
    var_4 = module_0.OneOf(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.one_of).__module__}.{type(var_4.one_of).__qualname__}' == 'builtins.list'
    assert len(var_4.one_of) == 1
    assert module_0.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_5 = module_0.Not(var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.composites.Not'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(var_5.negated).__module__}.{type(var_5.negated).__qualname__}' == 'typesystem.fields.Any'
    assert module_0.Not.errors == {'negated': 'Must not match.'}
    var_5.validate(var_0)

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = module_0.NeverMatch()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert module_0.NeverMatch.errors == {'never': 'This never validates.'}
    var_1 = module_0.IfThenElse(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(var_1.if_clause).__module__}.{type(var_1.if_clause).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert f'{type(var_1.then_clause).__module__}.{type(var_1.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_1.else_clause).__module__}.{type(var_1.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_2 = None
    var_3 = module_0.AllOf(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.all_of).__module__}.{type(var_3.all_of).__qualname__}' == 'typesystem.composites.NeverMatch'
    var_4 = 'lo(_)hl'
    var_5 = var_0.validate_or_error(var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_5.value is None
    assert f'{type(var_5.error).__module__}.{type(var_5.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_5.error) == 1
    var_6 = module_0.Not(var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.composites.Not'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert var_6.negated is None
    assert module_0.Not.errors == {'negated': 'Must not match.'}
    var_7 = '8.vC'
    var_8 = {var_4: var_2, var_4: var_4, var_4: var_2, var_7: var_4, var_7: var_4, var_4: var_3, var_4: var_2, var_4: var_2, var_4: var_1}
    var_9 = var_0.get_default_value()
    var_10 = module_0.OneOf(var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_10.title == ''
    assert var_10.description == ''
    assert var_10.allow_null is False
    assert var_10.read_only is False
    assert var_10.one_of is None
    assert module_0.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_11 = module_0.OneOf(var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_11.title == ''
    assert var_11.description == ''
    assert var_11.allow_null is False
    assert var_11.read_only is False
    assert f'{type(var_11.one_of).__module__}.{type(var_11.one_of).__qualname__}' == 'typesystem.composites.OneOf'
    var_12 = module_0.Not(var_0)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.composites.Not'
    assert var_12.title == ''
    assert var_12.description == ''
    assert var_12.allow_null is False
    assert var_12.read_only is False
    assert f'{type(var_12.negated).__module__}.{type(var_12.negated).__qualname__}' == 'typesystem.composites.NeverMatch'
    var_13 = var_12.validate(var_10)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_13.title == ''
    assert var_13.description == ''
    assert var_13.allow_null is False
    assert var_13.read_only is False
    assert var_13.one_of is None
    module_0.IfThenElse(var_9, var_2, **var_8)