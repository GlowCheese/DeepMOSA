# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.composites as module_0
import typesystem.fields as module_1

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = '4QNS'
    var_1 = None
    var_2 = {var_0: var_1}
    module_0.NeverMatch(**var_2)

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
    var_2 = None
    var_3 = module_0.IfThenElse(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.if_clause is None
    assert f'{type(var_3.then_clause).__module__}.{type(var_3.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_3.else_clause).__module__}.{type(var_3.else_clause).__qualname__}' == 'typesystem.fields.Any'

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    var_1 = False
    var_2 = module_1.Any(default=var_0, read_only=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Any'
    assert var_2.default is None
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    var_3 = module_0.Not(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.composites.Not'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.negated).__module__}.{type(var_3.negated).__qualname__}' == 'typesystem.fields.Any'
    assert module_0.Not.errors == {'negated': 'Must not match.'}
    var_4 = [var_2, var_2, var_2]
    var_5 = {}
    var_6 = module_0.OneOf(var_4, **var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(var_6.one_of).__module__}.{type(var_6.one_of).__qualname__}' == 'builtins.list'
    assert len(var_6.one_of) == 3
    assert module_0.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_6.validate(var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = []
    var_1 = module_0.OneOf(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.one_of == []
    assert module_0.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_2 = None
    var_1.validate(var_2)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    var_1 = None
    var_2 = module_0.AllOf(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.all_of is None
    var_2.validate(var_0)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = None
    var_1 = module_0.OneOf(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.one_of is None
    assert module_0.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_2 = [var_1]
    var_3 = var_1.get_default_value()
    var_4 = module_0.AllOf(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.all_of).__module__}.{type(var_4.all_of).__qualname__}' == 'builtins.list'
    assert len(var_4.all_of) == 1
    var_4.validate(var_2)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = None
    var_1 = module_0.OneOf(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.one_of is None
    assert module_0.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_2 = []
    var_3 = module_0.NeverMatch()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert module_0.NeverMatch.errors == {'never': 'This never validates.'}
    var_4 = module_0.AllOf(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.all_of == []
    var_5 = None
    var_6 = var_4.validate(var_2)
    var_6.validate(var_5)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = None
    var_1 = 'h\x0ba'
    var_2 = '$4^)<(m7M\tA'
    var_3 = {var_1: var_0, var_2: var_0}
    module_0.Not(var_0, **var_3)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = None
    var_1 = None
    var_2 = module_0.IfThenElse(var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.if_clause is None
    assert f'{type(var_2.then_clause).__module__}.{type(var_2.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_2.else_clause).__module__}.{type(var_2.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_3 = var_2.get_default_value()
    var_3.validate(var_0)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = None
    var_1 = 'I2lYH'
    var_2 = {var_1: var_0}
    module_0.IfThenElse(var_0, **var_2)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = module_0.NeverMatch()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert module_0.NeverMatch.errors == {'never': 'This never validates.'}
    var_1 = None
    var_0.validate(var_1)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = None
    var_1 = {}
    var_2 = module_0.NeverMatch()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert module_0.NeverMatch.errors == {'never': 'This never validates.'}
    var_3 = module_0.IfThenElse(var_0, else_clause=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.if_clause is None
    assert f'{type(var_3.then_clause).__module__}.{type(var_3.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert var_3.else_clause == {}
    var_4 = 'Z}Z85a{WE<`j '
    var_5 = 'l( &4nEe'
    var_6 = '__IvaEUVATU'
    var_7 = {var_4: var_2, var_5: var_2, var_6: var_2, var_6: var_0}
    module_0.AllOf(var_5, **var_7)

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = None
    var_1 = False
    var_2 = module_1.Any(default=var_0, read_only=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Any'
    assert var_2.default is None
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    var_3 = module_0.Not(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.composites.Not'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.negated).__module__}.{type(var_3.negated).__qualname__}' == 'typesystem.fields.Any'
    assert module_0.Not.errors == {'negated': 'Must not match.'}
    var_4 = '_e~Z5h])eD'
    var_5 = module_1.Field(description=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Field'
    assert var_5.title == ''
    assert var_5.description == '_e~Z5h])eD'
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert module_1.Field.errors == {}
    var_3.validate(var_0)

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = None
    var_1 = module_0.IfThenElse(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.if_clause is None
    assert f'{type(var_1.then_clause).__module__}.{type(var_1.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_1.else_clause).__module__}.{type(var_1.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_2 = var_1.has_default()
    assert var_2 is False
    var_3 = module_0.AllOf(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.all_of is None
    var_4 = 'lloinl\r'
    var_5 = {var_4: var_0, var_4: var_0, var_4: var_0}
    var_6 = module_0.OneOf(var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert var_6.one_of is False
    assert module_0.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_7 = module_0.NeverMatch()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert module_0.NeverMatch.errors == {'never': 'This never validates.'}
    var_8 = module_0.Not(var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.composites.Not'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert var_8.negated is None
    assert module_0.Not.errors == {'negated': 'Must not match.'}
    var_9 = module_0.IfThenElse(var_8, var_1)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert f'{type(var_9.if_clause).__module__}.{type(var_9.if_clause).__qualname__}' == 'typesystem.composites.Not'
    assert f'{type(var_9.then_clause).__module__}.{type(var_9.then_clause).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert f'{type(var_9.else_clause).__module__}.{type(var_9.else_clause).__qualname__}' == 'typesystem.fields.Any'
    module_0.Not(var_9, **var_5)

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = module_0.NeverMatch()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert module_0.NeverMatch.errors == {'never': 'This never validates.'}
    var_1 = None
    var_2 = module_0.IfThenElse(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(var_2.if_clause).__module__}.{type(var_2.if_clause).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert f'{type(var_2.then_clause).__module__}.{type(var_2.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_2.else_clause).__module__}.{type(var_2.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_3 = var_2.validate(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    module_1.Any(description=var_1)

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = None
    var_1 = False
    var_2 = module_1.Any(default=var_0, read_only=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Any'
    assert var_2.default is None
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    var_3 = module_0.IfThenElse(var_0, var_0, var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.if_clause is None
    assert f'{type(var_3.then_clause).__module__}.{type(var_3.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_3.else_clause).__module__}.{type(var_3.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_4 = module_0.Not(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.composites.Not'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.negated).__module__}.{type(var_4.negated).__qualname__}' == 'typesystem.fields.Any'
    assert module_0.Not.errors == {'negated': 'Must not match.'}
    var_5 = var_2.validate(var_0)
    var_6 = [var_2, var_4, var_2]
    var_7 = module_0.NeverMatch()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert module_0.NeverMatch.errors == {'never': 'This never validates.'}
    var_8 = var_4.get_default_value()
    var_9 = module_0.OneOf(var_6)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert f'{type(var_9.one_of).__module__}.{type(var_9.one_of).__qualname__}' == 'builtins.list'
    assert len(var_9.one_of) == 3
    assert module_0.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_10 = '_e~Z5h])eD'
    var_11 = module_1.Field(description=var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.fields.Field'
    assert var_11.title == ''
    assert var_11.description == '_e~Z5h])eD'
    assert var_11.allow_null is False
    assert var_11.read_only is False
    assert module_1.Field.errors == {}
    var_12 = module_0.IfThenElse(var_10, else_clause=var_9)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_12.title == ''
    assert var_12.description == ''
    assert var_12.allow_null is False
    assert var_12.read_only is False
    assert var_12.if_clause == '_e~Z5h])eD'
    assert f'{type(var_12.then_clause).__module__}.{type(var_12.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_12.else_clause).__module__}.{type(var_12.else_clause).__qualname__}' == 'typesystem.composites.OneOf'
    var_13 = var_12.serialize(var_9)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_13.title == ''
    assert var_13.description == ''
    assert var_13.allow_null is False
    assert var_13.read_only is False
    assert f'{type(var_13.one_of).__module__}.{type(var_13.one_of).__qualname__}' == 'builtins.list'
    assert len(var_13.one_of) == 3
    var_14 = module_0.AllOf(var_8)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_14.title == ''
    assert var_14.description == ''
    assert var_14.allow_null is False
    assert var_14.read_only is False
    assert var_14.all_of is None
    var_9.validate(var_13)

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = None
    var_1 = module_0.IfThenElse(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.if_clause is None
    assert f'{type(var_1.then_clause).__module__}.{type(var_1.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_1.else_clause).__module__}.{type(var_1.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_2 = var_1.get_default_value()
    var_3 = var_1.get_default_value()
    var_4 = 'lloinl\r'
    var_5 = {var_4: var_0, var_4: var_0, var_4: var_0}
    var_6 = module_0.NeverMatch()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert module_0.NeverMatch.errors == {'never': 'This never validates.'}
    var_7 = module_0.Not(var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.composites.Not'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert f'{type(var_7.negated).__module__}.{type(var_7.negated).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert module_0.Not.errors == {'negated': 'Must not match.'}
    var_8 = var_7.validate(var_0)
    var_9 = module_0.OneOf(var_2)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert var_9.one_of is None
    assert module_0.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_10 = module_0.AllOf(var_3)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_10.title == ''
    assert var_10.description == ''
    assert var_10.allow_null is False
    assert var_10.read_only is False
    assert var_10.all_of is None
    module_0.NeverMatch(**var_5)

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = None
    var_1 = module_0.IfThenElse(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.if_clause is None
    assert f'{type(var_1.then_clause).__module__}.{type(var_1.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_1.else_clause).__module__}.{type(var_1.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_2 = module_0.Not(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.composites.Not'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.negated is None
    assert module_0.Not.errors == {'negated': 'Must not match.'}
    var_3 = module_0.NeverMatch()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert module_0.NeverMatch.errors == {'never': 'This never validates.'}
    var_4 = module_0.AllOf(var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.all_of is None
    var_5 = 'linl\r'
    var_6 = {var_5: var_0, var_5: var_0, var_5: var_0}
    var_7 = [var_3, var_1, var_2]
    var_8 = module_0.OneOf(var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert f'{type(var_8.one_of).__module__}.{type(var_8.one_of).__qualname__}' == 'builtins.list'
    assert len(var_8.one_of) == 3
    assert module_0.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_9 = 'EjizAD\nVryc'
    var_10 = None
    var_11 = module_0.IfThenElse(var_0, else_clause=var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_11.title == ''
    assert var_11.description == ''
    assert var_11.allow_null is False
    assert var_11.read_only is False
    assert var_11.if_clause is None
    assert f'{type(var_11.then_clause).__module__}.{type(var_11.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_11.else_clause).__module__}.{type(var_11.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_12 = module_0.IfThenElse(var_3, var_3)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_12.title == ''
    assert var_12.description == ''
    assert var_12.allow_null is False
    assert var_12.read_only is False
    assert f'{type(var_12.if_clause).__module__}.{type(var_12.if_clause).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert f'{type(var_12.then_clause).__module__}.{type(var_12.then_clause).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert f'{type(var_12.else_clause).__module__}.{type(var_12.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_13 = module_0.IfThenElse(var_4)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_13.title == ''
    assert var_13.description == ''
    assert var_13.allow_null is False
    assert var_13.read_only is False
    assert f'{type(var_13.if_clause).__module__}.{type(var_13.if_clause).__qualname__}' == 'typesystem.composites.AllOf'
    assert f'{type(var_13.then_clause).__module__}.{type(var_13.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_13.else_clause).__module__}.{type(var_13.else_clause).__qualname__}' == 'typesystem.fields.Any'
    var_14 = var_8.serialize(var_0)
    var_15 = module_0.IfThenElse(var_12, else_clause=var_5)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_15.title == ''
    assert var_15.description == ''
    assert var_15.allow_null is False
    assert var_15.read_only is False
    assert f'{type(var_15.if_clause).__module__}.{type(var_15.if_clause).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert f'{type(var_15.then_clause).__module__}.{type(var_15.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert var_15.else_clause == 'linl\r'
    var_16 = {var_8, var_13, var_9}
    var_17 = var_15.validate(var_14)
    module_0.AllOf(var_16, **var_6)

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = None
    var_1 = False
    var_2 = module_1.Any(default=var_0, read_only=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Any'
    assert var_2.default is None
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    var_3 = var_2.validate(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Any'
    assert var_3.default is None
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    var_4 = module_0.Not(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.composites.Not'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.negated).__module__}.{type(var_4.negated).__qualname__}' == 'typesystem.fields.Any'
    assert module_0.Not.errors == {'negated': 'Must not match.'}
    var_5 = [var_2, var_4]
    var_6 = module_0.NeverMatch()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.composites.NeverMatch'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert module_0.NeverMatch.errors == {'never': 'This never validates.'}
    var_7 = var_4.get_default_value()
    var_8 = module_0.OneOf(var_5)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert f'{type(var_8.one_of).__module__}.{type(var_8.one_of).__qualname__}' == 'builtins.list'
    assert len(var_8.one_of) == 2
    assert module_0.OneOf.errors == {'no_match': 'Did not match any valid type.', 'multiple_matches': 'Matched more than one type.'}
    var_9 = '_e~Z5h])eD'
    var_10 = module_1.Field(description=var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.fields.Field'
    assert var_10.title == ''
    assert var_10.description == '_e~Z5h])eD'
    assert var_10.allow_null is False
    assert var_10.read_only is False
    assert module_1.Field.errors == {}
    var_11 = module_0.IfThenElse(var_9, else_clause=var_8)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.composites.IfThenElse'
    assert var_11.title == ''
    assert var_11.description == ''
    assert var_11.allow_null is False
    assert var_11.read_only is False
    assert var_11.if_clause == '_e~Z5h])eD'
    assert f'{type(var_11.then_clause).__module__}.{type(var_11.then_clause).__qualname__}' == 'typesystem.fields.Any'
    assert f'{type(var_11.else_clause).__module__}.{type(var_11.else_clause).__qualname__}' == 'typesystem.composites.OneOf'
    var_12 = var_11.serialize(var_8)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_12.title == ''
    assert var_12.description == ''
    assert var_12.allow_null is False
    assert var_12.read_only is False
    assert f'{type(var_12.one_of).__module__}.{type(var_12.one_of).__qualname__}' == 'builtins.list'
    assert len(var_12.one_of) == 2
    var_13 = module_0.AllOf(var_7)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.composites.AllOf'
    assert var_13.title == ''
    assert var_13.description == ''
    assert var_13.allow_null is False
    assert var_13.read_only is False
    assert var_13.all_of is None
    var_14 = var_8.validate(var_12)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.composites.OneOf'
    assert var_14.title == ''
    assert var_14.description == ''
    assert var_14.allow_null is False
    assert var_14.read_only is False
    assert f'{type(var_14.one_of).__module__}.{type(var_14.one_of).__qualname__}' == 'builtins.list'
    assert len(var_14.one_of) == 2
    var_15 = var_12.validate(var_0)
    var_16 = var_12.validate(var_15)
    var_17 = var_14.validate(var_7)
    var_18 = '~b"='
    var_17.validate(var_18)