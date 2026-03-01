# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import cookiecutter.prompt as module_0
import rich.prompt as module_1
import enum as module_2
import typing as module_3

def test_case_0():
    var_0 = None
    var_1 = module_0.render_variable(var_0, var_0, var_0)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'

def test_case_1():
    var_0 = None
    var_1 = 'IZzjdT6{Ca9/Mv),b7i'
    with pytest.raises(ValueError):
        module_0.read_user_choice(var_0, var_0, prefix=var_1)

def test_case_2():
    var_0 = None
    with pytest.raises(module_1.InvalidResponse):
        module_0.process_json(var_0)

def test_case_3():
    var_0 = None
    var_1 = '3+o,rO!243Qblx8V_'
    with pytest.raises(TypeError):
        module_0.read_user_dict(var_1, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    var_1 = '2\r>q#,*-u\r(=\x0c\n'
    var_2 = 'tSlhVsR'
    var_3 = {var_0: var_1, var_1: var_2, var_2: var_1, var_2: var_0}
    var_4 = True
    module_0.prompt_choice_for_template(var_1, var_3, var_4)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = None
    module_0.read_repo_password(var_0)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = module_2._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = module_3.Generic(**var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typing.Generic'
    assert module_3.EXCLUDED_ATTRIBUTES == ['__parameters__', '__orig_bases__', '__orig_class__', '_is_protocol', '_is_runtime_protocol', '__abstractmethods__', '__annotations__', '__dict__', '__doc__', '__init__', '__module__', '__new__', '__slots__', '__subclasshook__', '__weakref__', '__class_getitem__', '_MutableMapping__marker']
    assert f'{type(module_3.T).__module__}.{type(module_3.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_3.KT).__module__}.{type(module_3.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_3.VT).__module__}.{type(module_3.VT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_3.T_co).__module__}.{type(module_3.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_3.V_co).__module__}.{type(module_3.V_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_3.VT_co).__module__}.{type(module_3.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_3.T_contra).__module__}.{type(module_3.T_contra).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_3.CT_co).__module__}.{type(module_3.CT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_3.AnyStr).__module__}.{type(module_3.AnyStr).__qualname__}' == 'typing.TypeVar'
    assert module_3.TYPE_CHECKING is False
    module_0.render_variable(var_1, var_1, var_0)

def test_case_7():
    var_0 = None
    var_1 = module_0.render_variable(var_0, var_0, var_0)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    var_2 = module_2._EnumDict()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'enum._EnumDict'
    assert len(var_2) == 0
    with pytest.raises(ValueError):
        module_0.prompt_choice_for_config(var_2, var_2, var_0, var_2, var_2)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = None
    var_1 = {}
    module_0.prompt_choice_for_template(var_0, var_1, var_0)

def test_case_9():
    var_0 = None
    var_1 = module_2._EnumDict()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'enum._EnumDict'
    assert len(var_1) == 0
    var_2 = module_0.render_variable(var_1, var_1, var_0)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    var_3 = '`=5i\x0cRZ'
    with pytest.raises(module_1.InvalidResponse):
        module_0.process_json(var_3)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = None
    var_1 = module_0.render_variable(var_0, var_0, var_0)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert module_0.TYPE_CHECKING is False
    assert module_0.DEFAULT_DISPLAY == 'default'
    var_2 = var_1.__repr__()
    assert var_2 == 'None'
    module_0.prompt_choice_for_config(var_2, var_2, var_0, var_2, var_2)