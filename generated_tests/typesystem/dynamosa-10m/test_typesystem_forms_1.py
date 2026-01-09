# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.forms as module_0


def test_case_0():
    with pytest.raises(AssertionError):
        module_0.Jinja2Forms()

def test_case_1():
    var_0 = '@|7&fl1a6= -IPc'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.forms.Jinja2Forms'
    assert f'{type(var_1.env).__module__}.{type(var_1.env).__qualname__}' == 'jinja2.environment.Environment'

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    module_0.Form(env=var_0, schema=var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = "J{rN/Nh3@\\'P\t|e>Tj"
    var_1 = module_0.Jinja2Forms(directory=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.forms.Jinja2Forms'
    assert f'{type(var_1.env).__module__}.{type(var_1.env).__qualname__}' == 'jinja2.environment.Environment'
    var_1.create_form(var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = 't<'
    module_0.Jinja2Forms(package=var_0)

def test_case_5():
    var_0 = '@|7&fl1a6= -IPc'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.forms.Jinja2Forms'
    assert f'{type(var_1.env).__module__}.{type(var_1.env).__qualname__}' == 'jinja2.environment.Environment'
    with pytest.raises(AssertionError):
        var_1.load_template_env()

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = '@|7&fl1a6= -IPc'
    module_0.Jinja2Forms(directory=var_0, package=var_0)