# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.forms as module_0
import jinja2.filters as module_1

def test_case_0():
    with pytest.raises(AssertionError):
        module_0.Jinja2Forms()

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = '@|7&fl1a6= -IPc'
    module_0.Jinja2Forms(directory=var_0, package=var_0)

def test_case_2():
    var_0 = "*j\t_'mh5B"
    var_1 = None
    var_2 = module_0.Jinja2Forms(directory=var_0, package=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.forms.Jinja2Forms'
    assert f'{type(var_2.env).__module__}.{type(var_2.env).__qualname__}' == 'jinja2.environment.Environment'

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = '@|7&fl1Zoa6= -IPc'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.forms.Jinja2Forms'
    assert f'{type(var_1.env).__module__}.{type(var_1.env).__qualname__}' == 'jinja2.environment.Environment'
    var_2 = module_1.do_striptags(var_0)
    assert var_2 == '@|7&fl1Zoa6= -IPc'
    assert f'{type(module_1.F).__module__}.{type(module_1.F).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.K).__module__}.{type(module_1.K).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.V).__module__}.{type(module_1.V).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.FILTERS).__module__}.{type(module_1.FILTERS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FILTERS) == 54
    var_3 = var_2.__str__()
    assert var_3 == '@|7&fl1Zoa6= -IPc'
    var_4 = None
    var_5 = var_2.__str__()
    assert var_5 == '@|7&fl1Zoa6= -IPc'
    var_1.create_form(var_4)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = '@|7&fl1a6= -IPc'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.forms.Jinja2Forms'
    assert f'{type(var_1.env).__module__}.{type(var_1.env).__qualname__}' == 'jinja2.environment.Environment'
    var_1.load_template_env(package=var_0)

def test_case_5():
    var_0 = '@|7&fl1a6= -IPc'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.forms.Jinja2Forms'
    assert f'{type(var_1.env).__module__}.{type(var_1.env).__qualname__}' == 'jinja2.environment.Environment'
    with pytest.raises(AssertionError):
        var_1.load_template_env()

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = '9'
    module_0.Jinja2Forms(package=var_0)