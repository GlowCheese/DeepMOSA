# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import cookiecutter.generate as module_0
import cookiecutter.exceptions as module_1
import codecs as module_2
import jinja2.environment as module_3

def test_case_0():
    var_0 = '7+m0m%`{8tgr@]be\x0bE'
    var_1 = 'L|~8ynQO\ruipIvM\x0b[O]'
    var_2 = "CF>e3SOJU\\^g;2o Y'"
    var_3 = '@hi%}q{Fa'
    var_4 = {var_1: var_1, var_2: var_1, var_3: var_2, var_3: var_0}
    var_5 = module_0.is_copy_only_path(var_0, var_4)
    assert var_5 is False
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'cookiecutter.generate'
    assert module_0.logger.level == 0
    assert f'{type(module_0.logger.parent).__module__}.{type(module_0.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.propagate is True
    assert module_0.logger.handlers == []
    assert module_0.logger.disabled is False
    assert f'{type(module_0.logger.manager).__module__}.{type(module_0.logger.manager).__qualname__}' == 'logging.Manager'

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    module_0.is_copy_only_path(var_0, var_0)

def test_case_2():
    var_0 = 'oNbBk K}NFXF'
    var_1 = {var_0: var_0, var_0: var_0}
    var_2 = module_0.apply_overwrites_to_context(var_1, var_1)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'cookiecutter.generate'
    assert module_0.logger.level == 0
    assert f'{type(module_0.logger.parent).__module__}.{type(module_0.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.propagate is True
    assert module_0.logger.handlers == []
    assert module_0.logger.disabled is False
    assert f'{type(module_0.logger.manager).__module__}.{type(module_0.logger.manager).__qualname__}' == 'logging.Manager'

@pytest.mark.xfail(strict=True)
def test_case_3():
    module_0.generate_context()

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = 'P_(E\rY*ePh'
    var_1 = None
    module_0.render_and_create_dir(var_0, var_1, var_0, var_1)

def test_case_5():
    var_0 = None
    with pytest.raises(module_1.EmptyDirNameException):
        module_0.render_and_create_dir(var_0, var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = '2L'
    module_0.generate_files(var_0, accept_hooks=var_0)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = 'oNbBk K}NFXF'
    var_1 = {var_0: var_0, var_0: var_0}
    var_2 = module_0.apply_overwrites_to_context(var_1, var_1)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'cookiecutter.generate'
    assert module_0.logger.level == 0
    assert f'{type(module_0.logger.parent).__module__}.{type(module_0.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.propagate is True
    assert module_0.logger.handlers == []
    assert module_0.logger.disabled is False
    assert f'{type(module_0.logger.manager).__module__}.{type(module_0.logger.manager).__qualname__}' == 'logging.Manager'
    module_0.generate_files(var_0, var_1, var_2, var_2)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = 'oNbBk K}NFXF'
    var_1 = {var_0: var_0, var_0: var_0}
    var_2 = {}
    var_3 = module_0.apply_overwrites_to_context(var_2, var_1)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'cookiecutter.generate'
    assert module_0.logger.level == 0
    assert f'{type(module_0.logger.parent).__module__}.{type(module_0.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.propagate is True
    assert module_0.logger.handlers == []
    assert module_0.logger.disabled is False
    assert f'{type(module_0.logger.manager).__module__}.{type(module_0.logger.manager).__qualname__}' == 'logging.Manager'
    module_0.apply_overwrites_to_context(var_3, var_3, in_dictionary_variable=var_3)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = 'oNbBk K}NFXF'
    var_1 = None
    var_2 = 'cr;\nVY3dA('
    var_3 = {var_2: var_1}
    var_4 = module_0.is_copy_only_path(var_1, var_3)
    assert var_4 is False
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'cookiecutter.generate'
    assert module_0.logger.level == 0
    assert f'{type(module_0.logger.parent).__module__}.{type(module_0.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.propagate is True
    assert module_0.logger.handlers == []
    assert module_0.logger.disabled is False
    assert f'{type(module_0.logger.manager).__module__}.{type(module_0.logger.manager).__qualname__}' == 'logging.Manager'
    var_5 = {var_0: var_0, var_0: var_0}
    var_6 = '`'
    var_7 = module_0.is_copy_only_path(var_6, var_5)
    assert var_7 is False
    var_8 = True
    var_9 = module_0.apply_overwrites_to_context(var_5, var_3, in_dictionary_variable=var_8)
    module_0.apply_overwrites_to_context(var_9, var_5)

def test_case_10():
    var_0 = '=H6l8!qB'
    var_1 = True
    var_2 = {var_0: var_1, var_0: var_1, var_0: var_1}
    var_3 = module_0.apply_overwrites_to_context(var_2, var_2)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'cookiecutter.generate'
    assert module_0.logger.level == 0
    assert f'{type(module_0.logger.parent).__module__}.{type(module_0.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.propagate is True
    assert module_0.logger.handlers == []
    assert module_0.logger.disabled is False
    assert f'{type(module_0.logger.manager).__module__}.{type(module_0.logger.manager).__qualname__}' == 'logging.Manager'

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = None
    var_1 = 'cr;\nVY3dA('
    var_2 = {var_1: var_1, var_1: var_1}
    var_3 = module_0.is_copy_only_path(var_1, var_2)
    assert var_3 is False
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'cookiecutter.generate'
    assert module_0.logger.level == 0
    assert f'{type(module_0.logger.parent).__module__}.{type(module_0.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.propagate is True
    assert module_0.logger.handlers == []
    assert module_0.logger.disabled is False
    assert f'{type(module_0.logger.manager).__module__}.{type(module_0.logger.manager).__qualname__}' == 'logging.Manager'
    module_0.generate_context(var_3, var_0)

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = True
    var_1 = None
    module_0.generate_context(var_0, var_1)

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = 'oNbBk K}NFXF'
    var_1 = None
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_0, var_0: var_2}
    var_4 = '`'
    var_5 = module_0.is_copy_only_path(var_4, var_3)
    assert var_5 is False
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'cookiecutter.generate'
    assert module_0.logger.level == 0
    assert f'{type(module_0.logger.parent).__module__}.{type(module_0.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.propagate is True
    assert module_0.logger.handlers == []
    assert module_0.logger.disabled is False
    assert f'{type(module_0.logger.manager).__module__}.{type(module_0.logger.manager).__qualname__}' == 'logging.Manager'
    var_6 = module_0.apply_overwrites_to_context(var_3, var_2, in_dictionary_variable=var_5)
    module_0.generate_files(var_6)

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = 'oNbBk K}NFXF'
    var_1 = None
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_0, var_0: var_2}
    var_4 = '`'
    var_5 = module_0.is_copy_only_path(var_4, var_3)
    assert var_5 is False
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'cookiecutter.generate'
    assert module_0.logger.level == 0
    assert f'{type(module_0.logger.parent).__module__}.{type(module_0.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.propagate is True
    assert module_0.logger.handlers == []
    assert module_0.logger.disabled is False
    assert f'{type(module_0.logger.manager).__module__}.{type(module_0.logger.manager).__qualname__}' == 'logging.Manager'
    var_6 = module_0.apply_overwrites_to_context(var_3, var_3, in_dictionary_variable=var_1)
    var_7 = False
    module_0.generate_files(var_0, overwrite_if_exists=var_6, skip_if_file_exists=var_5, keep_project_on_failure=var_7)

def test_case_15():
    var_0 = 'name'
    var_1 = 'version'
    var_2 = 'new'
    var_3 = 0
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_4, var_4)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'cookiecutter.generate'
    assert module_0.logger.level == 0
    assert f'{type(module_0.logger.parent).__module__}.{type(module_0.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.propagate is True
    assert module_0.logger.handlers == []
    assert module_0.logger.disabled is False
    assert f'{type(module_0.logger.manager).__module__}.{type(module_0.logger.manager).__qualname__}' == 'logging.Manager'
    var_6 = 'choices'
    var_7 = 'Q[5'
    var_8 = 'b'
    var_9 = [var_7, var_8, var_1]
    var_10 = {var_6: var_9}
    var_11 = [var_7, var_8]
    var_12 = {var_6: var_11}
    var_13 = module_0.apply_overwrites_to_context(var_10, var_12)
    with pytest.raises(LookupError):
        module_2.getincrementaldecoder(var_2)

def test_case_16():
    var_0 = 'z'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_0}
    with pytest.raises(ValueError):
        module_0.apply_overwrites_to_context(var_2, var_3)

def test_case_17():
    var_0 = 'choices'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_5, var_5)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'cookiecutter.generate'
    assert module_0.logger.level == 0
    assert f'{type(module_0.logger.parent).__module__}.{type(module_0.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.propagate is True
    assert module_0.logger.handlers == []
    assert module_0.logger.disabled is False
    assert f'{type(module_0.logger.manager).__module__}.{type(module_0.logger.manager).__qualname__}' == 'logging.Manager'
    var_7 = [var_1, var_2, var_3]
    var_8 = {var_0: var_7}
    var_9 = module_0.apply_overwrites_to_context(var_8, var_5)
    var_10 = {var_0: var_2}
    var_11 = module_0.apply_overwrites_to_context(var_5, var_10)
    var_12 = 'key3'
    var_13 = 'enabled'
    var_14 = False
    var_15 = {var_13: var_14}
    var_16 = {var_13: var_12}
    with pytest.raises(ValueError):
        module_0.apply_overwrites_to_context(var_15, var_16)

def test_case_18():
    var_0 = 'name'
    var_1 = 'version'
    var_2 = 'old'
    var_3 = 1
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'new'
    var_6 = 2
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_4, var_7)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'cookiecutter.generate'
    assert module_0.logger.level == 0
    assert f'{type(module_0.logger.parent).__module__}.{type(module_0.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.propagate is True
    assert module_0.logger.handlers == []
    assert module_0.logger.disabled is False
    assert f'{type(module_0.logger.manager).__module__}.{type(module_0.logger.manager).__qualname__}' == 'logging.Manager'
    var_9 = 'choices'
    var_10 = 'a'
    var_11 = 'b'
    var_12 = 'c'
    var_13 = [var_10, var_11, var_12]
    var_14 = {var_9: var_13}
    var_15 = [var_10, var_11]
    var_16 = {var_9: var_15}
    var_17 = module_0.apply_overwrites_to_context(var_14, var_16)
    var_18 = [var_10, var_11, var_12]
    var_19 = {var_9: var_18}
    var_20 = 'z'
    var_21 = [var_10, var_20]
    var_22 = {var_9: var_21}
    with pytest.raises(ValueError):
        module_0.apply_overwrites_to_context(var_19, var_22)

def test_case_19():
    var_0 = 'name'
    var_1 = 'version'
    var_2 = 'old'
    var_3 = 1
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'new'
    var_6 = module_0.apply_overwrites_to_context(var_4, var_4)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'cookiecutter.generate'
    assert module_0.logger.level == 0
    assert f'{type(module_0.logger.parent).__module__}.{type(module_0.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.propagate is True
    assert module_0.logger.handlers == []
    assert module_0.logger.disabled is False
    assert f'{type(module_0.logger.manager).__module__}.{type(module_0.logger.manager).__qualname__}' == 'logging.Manager'
    var_7 = 'choices'
    var_8 = 'a'
    var_9 = 'b'
    var_10 = 'c'
    var_11 = {}
    var_12 = [var_8, var_9]
    var_13 = {var_7: var_12}
    var_14 = module_0.apply_overwrites_to_context(var_11, var_13)
    var_15 = [var_8, var_9, var_10]
    var_16 = {var_7: var_15}
    var_17 = {var_7: var_15}
    var_18 = module_0.apply_overwrites_to_context(var_16, var_17)
    var_19 = [var_8, var_9, var_10]
    var_20 = {var_7: var_19}
    var_21 = {var_7: var_5}
    with pytest.raises(ValueError):
        module_0.apply_overwrites_to_context(var_20, var_21)

def test_case_20():
    var_0 = 'name'
    var_1 = 'version'
    var_2 = 'choices'
    var_3 = 'a'
    var_4 = [var_3, var_3, var_3]
    var_5 = [var_3, var_0]
    var_6 = {var_2: var_5}
    var_7 = module_0.apply_overwrites_to_context(var_6, var_6)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'cookiecutter.generate'
    assert module_0.logger.level == 0
    assert f'{type(module_0.logger.parent).__module__}.{type(module_0.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.propagate is True
    assert module_0.logger.handlers == []
    assert module_0.logger.disabled is False
    assert f'{type(module_0.logger.manager).__module__}.{type(module_0.logger.manager).__qualname__}' == 'logging.Manager'
    var_8 = True
    var_9 = module_0.apply_overwrites_to_context(var_6, var_6, in_dictionary_variable=var_8)
    var_10 = [var_3, var_8, var_1]
    var_11 = {var_2: var_10}
    var_12 = {var_2: var_10}
    var_13 = module_0.apply_overwrites_to_context(var_11, var_12)
    var_14 = {var_2: var_4}
    var_15 = module_0.apply_overwrites_to_context(var_6, var_14)
    var_16 = 'key3'
    var_17 = 'enabled'
    var_18 = {var_17: var_8}
    var_19 = {var_17: var_16}
    with pytest.raises(ValueError):
        module_0.apply_overwrites_to_context(var_18, var_19)

def test_case_21():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = '*.bin'
    var_3 = 'docs/manual.pdf'
    var_4 = 'assets/*'
    var_5 = [var_2, var_3, var_4]
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = 'test.bin'
    var_9 = module_0.is_copy_only_path(var_8, var_7)
    assert var_9 is True
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'cookiecutter.generate'
    assert module_0.logger.level == 0
    assert f'{type(module_0.logger.parent).__module__}.{type(module_0.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.propagate is True
    assert module_0.logger.handlers == []
    assert module_0.logger.disabled is False
    assert f'{type(module_0.logger.manager).__module__}.{type(module_0.logger.manager).__qualname__}' == 'logging.Manager'
    var_10 = module_0.is_copy_only_path(var_3, var_7)
    assert var_10 is True
    var_11 = 'assets/image.png'
    var_12 = module_0.is_copy_only_path(var_11, var_7)
    assert var_12 is True
    var_13 = 'src/main.py'
    var_14 = module_0.is_copy_only_path(var_13, var_7)
    assert var_14 is False
    var_15 = {var_1: var_5}
    var_16 = {var_0: var_15}
    var_17 = 'script.py'
    var_18 = module_0.is_copy_only_path(var_17, var_16)
    assert var_18 is False
    var_19 = {}
    var_20 = {var_0: var_19}
    var_21 = module_0.is_copy_only_path(var_8, var_20)
    assert var_21 is False

@pytest.mark.xfail(strict=True)
def test_case_22():
    var_0 = module_3.Environment()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'jinja2.environment.Environment'
    assert var_0.block_start_string == '{%'
    assert var_0.block_end_string == '%}'
    assert var_0.variable_start_string == '{{'
    assert var_0.variable_end_string == '}}'
    assert var_0.comment_start_string == '{#'
    assert var_0.comment_end_string == '#}'
    assert var_0.line_statement_prefix is None
    assert var_0.line_comment_prefix is None
    assert var_0.trim_blocks is False
    assert var_0.lstrip_blocks is False
    assert var_0.newline_sequence == '\n'
    assert var_0.keep_trailing_newline is False
    assert var_0.optimized is True
    assert var_0.finalize is None
    assert var_0.autoescape is False
    assert f'{type(var_0.filters).__module__}.{type(var_0.filters).__qualname__}' == 'builtins.dict'
    assert len(var_0.filters) == 54
    assert f'{type(var_0.tests).__module__}.{type(var_0.tests).__qualname__}' == 'builtins.dict'
    assert len(var_0.tests) == 39
    assert f'{type(var_0.globals).__module__}.{type(var_0.globals).__qualname__}' == 'builtins.dict'
    assert len(var_0.globals) == 6
    assert var_0.loader is None
    assert f'{type(var_0.cache).__module__}.{type(var_0.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_0.cache) == 0
    assert var_0.bytecode_cache is None
    assert var_0.auto_reload is True
    assert var_0.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_0.extensions == {}
    assert var_0.is_async is False
    assert module_3.BLOCK_END_STRING == '%}'
    assert module_3.BLOCK_START_STRING == '{%'
    assert module_3.COMMENT_END_STRING == '#}'
    assert module_3.COMMENT_START_STRING == '{#'
    assert f'{type(module_3.DEFAULT_FILTERS).__module__}.{type(module_3.DEFAULT_FILTERS).__qualname__}' == 'builtins.dict'
    assert len(module_3.DEFAULT_FILTERS) == 54
    assert f'{type(module_3.DEFAULT_NAMESPACE).__module__}.{type(module_3.DEFAULT_NAMESPACE).__qualname__}' == 'builtins.dict'
    assert len(module_3.DEFAULT_NAMESPACE) == 6
    assert module_3.DEFAULT_POLICIES == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert f'{type(module_3.DEFAULT_TESTS).__module__}.{type(module_3.DEFAULT_TESTS).__qualname__}' == 'builtins.dict'
    assert len(module_3.DEFAULT_TESTS) == 39
    assert module_3.KEEP_TRAILING_NEWLINE is False
    assert module_3.LINE_COMMENT_PREFIX is None
    assert module_3.LINE_STATEMENT_PREFIX is None
    assert module_3.LSTRIP_BLOCKS is False
    assert module_3.NEWLINE_SEQUENCE == '\n'
    assert module_3.TRIM_BLOCKS is False
    assert module_3.VARIABLE_END_STRING == '}}'
    assert module_3.VARIABLE_START_STRING == '{{'
    assert f'{type(module_3.missing).__module__}.{type(module_3.missing).__qualname__}' == 'jinja2.utils._MissingType'
    assert module_3.Environment.sandboxed is False
    assert module_3.Environment.overlayed is False
    assert module_3.Environment.linked_to is None
    assert module_3.Environment.shared is False
    assert f'{type(module_3.Environment.lexer).__module__}.{type(module_3.Environment.lexer).__qualname__}' == 'builtins.property'
    var_1 = ''
    var_2 = {}
    module_0.generate_file(var_1, var_1, var_2, var_0)

@pytest.mark.xfail(strict=True)
def test_case_23():
    var_0 = module_3.Environment()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'jinja2.environment.Environment'
    assert var_0.block_start_string == '{%'
    assert var_0.block_end_string == '%}'
    assert var_0.variable_start_string == '{{'
    assert var_0.variable_end_string == '}}'
    assert var_0.comment_start_string == '{#'
    assert var_0.comment_end_string == '#}'
    assert var_0.line_statement_prefix is None
    assert var_0.line_comment_prefix is None
    assert var_0.trim_blocks is False
    assert var_0.lstrip_blocks is False
    assert var_0.newline_sequence == '\n'
    assert var_0.keep_trailing_newline is False
    assert var_0.optimized is True
    assert var_0.finalize is None
    assert var_0.autoescape is False
    assert f'{type(var_0.filters).__module__}.{type(var_0.filters).__qualname__}' == 'builtins.dict'
    assert len(var_0.filters) == 54
    assert f'{type(var_0.tests).__module__}.{type(var_0.tests).__qualname__}' == 'builtins.dict'
    assert len(var_0.tests) == 39
    assert f'{type(var_0.globals).__module__}.{type(var_0.globals).__qualname__}' == 'builtins.dict'
    assert len(var_0.globals) == 6
    assert var_0.loader is None
    assert f'{type(var_0.cache).__module__}.{type(var_0.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_0.cache) == 0
    assert var_0.bytecode_cache is None
    assert var_0.auto_reload is True
    assert var_0.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_0.extensions == {}
    assert var_0.is_async is False
    assert module_3.BLOCK_END_STRING == '%}'
    assert module_3.BLOCK_START_STRING == '{%'
    assert module_3.COMMENT_END_STRING == '#}'
    assert module_3.COMMENT_START_STRING == '{#'
    assert f'{type(module_3.DEFAULT_FILTERS).__module__}.{type(module_3.DEFAULT_FILTERS).__qualname__}' == 'builtins.dict'
    assert len(module_3.DEFAULT_FILTERS) == 54
    assert f'{type(module_3.DEFAULT_NAMESPACE).__module__}.{type(module_3.DEFAULT_NAMESPACE).__qualname__}' == 'builtins.dict'
    assert len(module_3.DEFAULT_NAMESPACE) == 6
    assert module_3.DEFAULT_POLICIES == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert f'{type(module_3.DEFAULT_TESTS).__module__}.{type(module_3.DEFAULT_TESTS).__qualname__}' == 'builtins.dict'
    assert len(module_3.DEFAULT_TESTS) == 39
    assert module_3.KEEP_TRAILING_NEWLINE is False
    assert module_3.LINE_COMMENT_PREFIX is None
    assert module_3.LINE_STATEMENT_PREFIX is None
    assert module_3.LSTRIP_BLOCKS is False
    assert module_3.NEWLINE_SEQUENCE == '\n'
    assert module_3.TRIM_BLOCKS is False
    assert module_3.VARIABLE_END_STRING == '}}'
    assert module_3.VARIABLE_START_STRING == '{{'
    assert f'{type(module_3.missing).__module__}.{type(module_3.missing).__qualname__}' == 'jinja2.utils._MissingType'
    assert module_3.Environment.sandboxed is False
    assert module_3.Environment.overlayed is False
    assert module_3.Environment.linked_to is None
    assert module_3.Environment.shared is False
    assert f'{type(module_3.Environment.lexer).__module__}.{type(module_3.Environment.lexer).__qualname__}' == 'builtins.property'
    var_1 = 'aD#mk-BlVBDT'
    var_2 = {}
    module_0.generate_file(var_1, var_1, var_2, var_0, var_0)

def test_case_24():
    var_0 = '6\nOI;y\rG\n^QPvsFH5t'
    var_1 = module_3.Environment()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'jinja2.environment.Environment'
    assert var_1.block_start_string == '{%'
    assert var_1.block_end_string == '%}'
    assert var_1.variable_start_string == '{{'
    assert var_1.variable_end_string == '}}'
    assert var_1.comment_start_string == '{#'
    assert var_1.comment_end_string == '#}'
    assert var_1.line_statement_prefix is None
    assert var_1.line_comment_prefix is None
    assert var_1.trim_blocks is False
    assert var_1.lstrip_blocks is False
    assert var_1.newline_sequence == '\n'
    assert var_1.keep_trailing_newline is False
    assert var_1.optimized is True
    assert var_1.finalize is None
    assert var_1.autoescape is False
    assert f'{type(var_1.filters).__module__}.{type(var_1.filters).__qualname__}' == 'builtins.dict'
    assert len(var_1.filters) == 54
    assert f'{type(var_1.tests).__module__}.{type(var_1.tests).__qualname__}' == 'builtins.dict'
    assert len(var_1.tests) == 39
    assert f'{type(var_1.globals).__module__}.{type(var_1.globals).__qualname__}' == 'builtins.dict'
    assert len(var_1.globals) == 6
    assert var_1.loader is None
    assert f'{type(var_1.cache).__module__}.{type(var_1.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_1.cache) == 0
    assert var_1.bytecode_cache is None
    assert var_1.auto_reload is True
    assert var_1.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_1.extensions == {}
    assert var_1.is_async is False
    assert module_3.BLOCK_END_STRING == '%}'
    assert module_3.BLOCK_START_STRING == '{%'
    assert module_3.COMMENT_END_STRING == '#}'
    assert module_3.COMMENT_START_STRING == '{#'
    assert f'{type(module_3.DEFAULT_FILTERS).__module__}.{type(module_3.DEFAULT_FILTERS).__qualname__}' == 'builtins.dict'
    assert len(module_3.DEFAULT_FILTERS) == 54
    assert f'{type(module_3.DEFAULT_NAMESPACE).__module__}.{type(module_3.DEFAULT_NAMESPACE).__qualname__}' == 'builtins.dict'
    assert len(module_3.DEFAULT_NAMESPACE) == 6
    assert module_3.DEFAULT_POLICIES == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert f'{type(module_3.DEFAULT_TESTS).__module__}.{type(module_3.DEFAULT_TESTS).__qualname__}' == 'builtins.dict'
    assert len(module_3.DEFAULT_TESTS) == 39
    assert module_3.KEEP_TRAILING_NEWLINE is False
    assert module_3.LINE_COMMENT_PREFIX is None
    assert module_3.LINE_STATEMENT_PREFIX is None
    assert module_3.LSTRIP_BLOCKS is False
    assert module_3.NEWLINE_SEQUENCE == '\n'
    assert module_3.TRIM_BLOCKS is False
    assert module_3.VARIABLE_END_STRING == '}}'
    assert module_3.VARIABLE_START_STRING == '{{'
    assert f'{type(module_3.missing).__module__}.{type(module_3.missing).__qualname__}' == 'jinja2.utils._MissingType'
    assert module_3.Environment.sandboxed is False
    assert module_3.Environment.overlayed is False
    assert module_3.Environment.linked_to is None
    assert module_3.Environment.shared is False
    assert f'{type(module_3.Environment.lexer).__module__}.{type(module_3.Environment.lexer).__qualname__}' == 'builtins.property'
    var_2 = 'repo_dir'
    var_3 = module_2.make_identity_dict(var_0)
    assert module_2.BOM_UTF8 == b'\xef\xbb\xbf'
    assert module_2.BOM_LE == b'\xff\xfe'
    assert module_2.BOM_UTF16_LE == b'\xff\xfe'
    assert module_2.BOM_BE == b'\xfe\xff'
    assert module_2.BOM_UTF16_BE == b'\xfe\xff'
    assert module_2.BOM_UTF32_LE == b'\xff\xfe\x00\x00'
    assert module_2.BOM_UTF32_BE == b'\x00\x00\xfe\xff'
    assert module_2.BOM == b'\xff\xfe'
    assert module_2.BOM_UTF16 == b'\xff\xfe'
    assert module_2.BOM_UTF32 == b'\xff\xfe\x00\x00'
    assert module_2.BOM32_LE == b'\xff\xfe'
    assert module_2.BOM32_BE == b'\xfe\xff'
    assert module_2.BOM64_LE == b'\xff\xfe\x00\x00'
    assert module_2.BOM64_BE == b'\x00\x00\xfe\xff'
    var_4 = module_0.apply_overwrites_to_context(var_3, var_3)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'cookiecutter.generate'
    assert module_0.logger.level == 0
    assert f'{type(module_0.logger.parent).__module__}.{type(module_0.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.propagate is True
    assert module_0.logger.handlers == []
    assert module_0.logger.disabled is False
    assert f'{type(module_0.logger.manager).__module__}.{type(module_0.logger.manager).__qualname__}' == 'logging.Manager'
    var_5 = ''
    var_6 = True
    var_7 = module_0.generate_file(var_2, var_5, var_3, var_1, var_6)

@pytest.mark.xfail(strict=True)
def test_case_25():
    var_0 = ':F}'
    var_1 = module_3.Environment()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'jinja2.environment.Environment'
    assert var_1.block_start_string == '{%'
    assert var_1.block_end_string == '%}'
    assert var_1.variable_start_string == '{{'
    assert var_1.variable_end_string == '}}'
    assert var_1.comment_start_string == '{#'
    assert var_1.comment_end_string == '#}'
    assert var_1.line_statement_prefix is None
    assert var_1.line_comment_prefix is None
    assert var_1.trim_blocks is False
    assert var_1.lstrip_blocks is False
    assert var_1.newline_sequence == '\n'
    assert var_1.keep_trailing_newline is False
    assert var_1.optimized is True
    assert var_1.finalize is None
    assert var_1.autoescape is False
    assert f'{type(var_1.filters).__module__}.{type(var_1.filters).__qualname__}' == 'builtins.dict'
    assert len(var_1.filters) == 54
    assert f'{type(var_1.tests).__module__}.{type(var_1.tests).__qualname__}' == 'builtins.dict'
    assert len(var_1.tests) == 39
    assert f'{type(var_1.globals).__module__}.{type(var_1.globals).__qualname__}' == 'builtins.dict'
    assert len(var_1.globals) == 6
    assert var_1.loader is None
    assert f'{type(var_1.cache).__module__}.{type(var_1.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_1.cache) == 0
    assert var_1.bytecode_cache is None
    assert var_1.auto_reload is True
    assert var_1.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_1.extensions == {}
    assert var_1.is_async is False
    assert module_3.BLOCK_END_STRING == '%}'
    assert module_3.BLOCK_START_STRING == '{%'
    assert module_3.COMMENT_END_STRING == '#}'
    assert module_3.COMMENT_START_STRING == '{#'
    assert f'{type(module_3.DEFAULT_FILTERS).__module__}.{type(module_3.DEFAULT_FILTERS).__qualname__}' == 'builtins.dict'
    assert len(module_3.DEFAULT_FILTERS) == 54
    assert f'{type(module_3.DEFAULT_NAMESPACE).__module__}.{type(module_3.DEFAULT_NAMESPACE).__qualname__}' == 'builtins.dict'
    assert len(module_3.DEFAULT_NAMESPACE) == 6
    assert module_3.DEFAULT_POLICIES == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert f'{type(module_3.DEFAULT_TESTS).__module__}.{type(module_3.DEFAULT_TESTS).__qualname__}' == 'builtins.dict'
    assert len(module_3.DEFAULT_TESTS) == 39
    assert module_3.KEEP_TRAILING_NEWLINE is False
    assert module_3.LINE_COMMENT_PREFIX is None
    assert module_3.LINE_STATEMENT_PREFIX is None
    assert module_3.LSTRIP_BLOCKS is False
    assert module_3.NEWLINE_SEQUENCE == '\n'
    assert module_3.TRIM_BLOCKS is False
    assert module_3.VARIABLE_END_STRING == '}}'
    assert module_3.VARIABLE_START_STRING == '{{'
    assert f'{type(module_3.missing).__module__}.{type(module_3.missing).__qualname__}' == 'jinja2.utils._MissingType'
    assert module_3.Environment.sandboxed is False
    assert module_3.Environment.overlayed is False
    assert module_3.Environment.linked_to is None
    assert module_3.Environment.shared is False
    assert f'{type(module_3.Environment.lexer).__module__}.{type(module_3.Environment.lexer).__qualname__}' == 'builtins.property'
    var_2 = 'W'
    var_3 = {}
    var_4 = module_0.render_and_create_dir(var_2, var_3, var_2, var_1, var_1)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'cookiecutter.generate'
    assert module_0.logger.level == 0
    assert f'{type(module_0.logger.parent).__module__}.{type(module_0.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.propagate is True
    assert module_0.logger.handlers == []
    assert module_0.logger.disabled is False
    assert f'{type(module_0.logger.manager).__module__}.{type(module_0.logger.manager).__qualname__}' == 'logging.Manager'
    var_5 = module_2.make_identity_dict(var_0)
    assert module_2.BOM_UTF8 == b'\xef\xbb\xbf'
    assert module_2.BOM_LE == b'\xff\xfe'
    assert module_2.BOM_UTF16_LE == b'\xff\xfe'
    assert module_2.BOM_BE == b'\xfe\xff'
    assert module_2.BOM_UTF16_BE == b'\xfe\xff'
    assert module_2.BOM_UTF32_LE == b'\xff\xfe\x00\x00'
    assert module_2.BOM_UTF32_BE == b'\x00\x00\xfe\xff'
    assert module_2.BOM == b'\xff\xfe'
    assert module_2.BOM_UTF16 == b'\xff\xfe'
    assert module_2.BOM_UTF32 == b'\xff\xfe\x00\x00'
    assert module_2.BOM32_LE == b'\xff\xfe'
    assert module_2.BOM32_BE == b'\xfe\xff'
    assert module_2.BOM64_LE == b'\xff\xfe\x00\x00'
    assert module_2.BOM64_BE == b'\x00\x00\xfe\xff'
    var_6 = module_0.apply_overwrites_to_context(var_5, var_5)
    var_7 = 'rSA:*Gid;c@+e'
    module_0.generate_file(var_7, var_0, var_5, var_1, var_0)

def test_case_26():
    var_0 = module_3.Environment()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'jinja2.environment.Environment'
    assert var_0.block_start_string == '{%'
    assert var_0.block_end_string == '%}'
    assert var_0.variable_start_string == '{{'
    assert var_0.variable_end_string == '}}'
    assert var_0.comment_start_string == '{#'
    assert var_0.comment_end_string == '#}'
    assert var_0.line_statement_prefix is None
    assert var_0.line_comment_prefix is None
    assert var_0.trim_blocks is False
    assert var_0.lstrip_blocks is False
    assert var_0.newline_sequence == '\n'
    assert var_0.keep_trailing_newline is False
    assert var_0.optimized is True
    assert var_0.finalize is None
    assert var_0.autoescape is False
    assert f'{type(var_0.filters).__module__}.{type(var_0.filters).__qualname__}' == 'builtins.dict'
    assert len(var_0.filters) == 54
    assert f'{type(var_0.tests).__module__}.{type(var_0.tests).__qualname__}' == 'builtins.dict'
    assert len(var_0.tests) == 39
    assert f'{type(var_0.globals).__module__}.{type(var_0.globals).__qualname__}' == 'builtins.dict'
    assert len(var_0.globals) == 6
    assert var_0.loader is None
    assert f'{type(var_0.cache).__module__}.{type(var_0.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_0.cache) == 0
    assert var_0.bytecode_cache is None
    assert var_0.auto_reload is True
    assert var_0.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_0.extensions == {}
    assert var_0.is_async is False
    assert module_3.BLOCK_END_STRING == '%}'
    assert module_3.BLOCK_START_STRING == '{%'
    assert module_3.COMMENT_END_STRING == '#}'
    assert module_3.COMMENT_START_STRING == '{#'
    assert f'{type(module_3.DEFAULT_FILTERS).__module__}.{type(module_3.DEFAULT_FILTERS).__qualname__}' == 'builtins.dict'
    assert len(module_3.DEFAULT_FILTERS) == 54
    assert f'{type(module_3.DEFAULT_NAMESPACE).__module__}.{type(module_3.DEFAULT_NAMESPACE).__qualname__}' == 'builtins.dict'
    assert len(module_3.DEFAULT_NAMESPACE) == 6
    assert module_3.DEFAULT_POLICIES == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert f'{type(module_3.DEFAULT_TESTS).__module__}.{type(module_3.DEFAULT_TESTS).__qualname__}' == 'builtins.dict'
    assert len(module_3.DEFAULT_TESTS) == 39
    assert module_3.KEEP_TRAILING_NEWLINE is False
    assert module_3.LINE_COMMENT_PREFIX is None
    assert module_3.LINE_STATEMENT_PREFIX is None
    assert module_3.LSTRIP_BLOCKS is False
    assert module_3.NEWLINE_SEQUENCE == '\n'
    assert module_3.TRIM_BLOCKS is False
    assert module_3.VARIABLE_END_STRING == '}}'
    assert module_3.VARIABLE_START_STRING == '{{'
    assert f'{type(module_3.missing).__module__}.{type(module_3.missing).__qualname__}' == 'jinja2.utils._MissingType'
    assert module_3.Environment.sandboxed is False
    assert module_3.Environment.overlayed is False
    assert module_3.Environment.linked_to is None
    assert module_3.Environment.shared is False
    assert f'{type(module_3.Environment.lexer).__module__}.{type(module_3.Environment.lexer).__qualname__}' == 'builtins.property'
    var_1 = '_(1~'
    var_2 = {}
    var_3 = None
    with pytest.raises(module_1.OutputDirExistsException):
        module_0.render_and_create_dir(var_1, var_2, var_1, var_0, var_3)

@pytest.mark.xfail(strict=True)
def test_case_27():
    var_0 = '6\nOI;[y\rG\n^QPvsFH5t'
    var_1 = module_3.Environment()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'jinja2.environment.Environment'
    assert var_1.block_start_string == '{%'
    assert var_1.block_end_string == '%}'
    assert var_1.variable_start_string == '{{'
    assert var_1.variable_end_string == '}}'
    assert var_1.comment_start_string == '{#'
    assert var_1.comment_end_string == '#}'
    assert var_1.line_statement_prefix is None
    assert var_1.line_comment_prefix is None
    assert var_1.trim_blocks is False
    assert var_1.lstrip_blocks is False
    assert var_1.newline_sequence == '\n'
    assert var_1.keep_trailing_newline is False
    assert var_1.optimized is True
    assert var_1.finalize is None
    assert var_1.autoescape is False
    assert f'{type(var_1.filters).__module__}.{type(var_1.filters).__qualname__}' == 'builtins.dict'
    assert len(var_1.filters) == 54
    assert f'{type(var_1.tests).__module__}.{type(var_1.tests).__qualname__}' == 'builtins.dict'
    assert len(var_1.tests) == 39
    assert f'{type(var_1.globals).__module__}.{type(var_1.globals).__qualname__}' == 'builtins.dict'
    assert len(var_1.globals) == 6
    assert var_1.loader is None
    assert f'{type(var_1.cache).__module__}.{type(var_1.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_1.cache) == 0
    assert var_1.bytecode_cache is None
    assert var_1.auto_reload is True
    assert var_1.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_1.extensions == {}
    assert var_1.is_async is False
    assert module_3.BLOCK_END_STRING == '%}'
    assert module_3.BLOCK_START_STRING == '{%'
    assert module_3.COMMENT_END_STRING == '#}'
    assert module_3.COMMENT_START_STRING == '{#'
    assert f'{type(module_3.DEFAULT_FILTERS).__module__}.{type(module_3.DEFAULT_FILTERS).__qualname__}' == 'builtins.dict'
    assert len(module_3.DEFAULT_FILTERS) == 54
    assert f'{type(module_3.DEFAULT_NAMESPACE).__module__}.{type(module_3.DEFAULT_NAMESPACE).__qualname__}' == 'builtins.dict'
    assert len(module_3.DEFAULT_NAMESPACE) == 6
    assert module_3.DEFAULT_POLICIES == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert f'{type(module_3.DEFAULT_TESTS).__module__}.{type(module_3.DEFAULT_TESTS).__qualname__}' == 'builtins.dict'
    assert len(module_3.DEFAULT_TESTS) == 39
    assert module_3.KEEP_TRAILING_NEWLINE is False
    assert module_3.LINE_COMMENT_PREFIX is None
    assert module_3.LINE_STATEMENT_PREFIX is None
    assert module_3.LSTRIP_BLOCKS is False
    assert module_3.NEWLINE_SEQUENCE == '\n'
    assert module_3.TRIM_BLOCKS is False
    assert module_3.VARIABLE_END_STRING == '}}'
    assert module_3.VARIABLE_START_STRING == '{{'
    assert f'{type(module_3.missing).__module__}.{type(module_3.missing).__qualname__}' == 'jinja2.utils._MissingType'
    assert module_3.Environment.sandboxed is False
    assert module_3.Environment.overlayed is False
    assert module_3.Environment.linked_to is None
    assert module_3.Environment.shared is False
    assert f'{type(module_3.Environment.lexer).__module__}.{type(module_3.Environment.lexer).__qualname__}' == 'builtins.property'
    var_2 = '_~'
    var_3 = {}
    var_4 = module_0.render_and_create_dir(var_2, var_3, var_2, var_1, var_1)
    assert f'{type(module_0.annotations).__module__}.{type(module_0.annotations).__qualname__}' == '__future__._Feature'
    assert module_0.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_0.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_0.annotations.compiler_flag == 16777216
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'cookiecutter.generate'
    assert module_0.logger.level == 0
    assert f'{type(module_0.logger.parent).__module__}.{type(module_0.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.propagate is True
    assert module_0.logger.handlers == []
    assert module_0.logger.disabled is False
    assert f'{type(module_0.logger.manager).__module__}.{type(module_0.logger.manager).__qualname__}' == 'logging.Manager'
    var_5 = module_2.make_identity_dict(var_0)
    assert module_2.BOM_UTF8 == b'\xef\xbb\xbf'
    assert module_2.BOM_LE == b'\xff\xfe'
    assert module_2.BOM_UTF16_LE == b'\xff\xfe'
    assert module_2.BOM_BE == b'\xfe\xff'
    assert module_2.BOM_UTF16_BE == b'\xfe\xff'
    assert module_2.BOM_UTF32_LE == b'\xff\xfe\x00\x00'
    assert module_2.BOM_UTF32_BE == b'\x00\x00\xfe\xff'
    assert module_2.BOM == b'\xff\xfe'
    assert module_2.BOM_UTF16 == b'\xff\xfe'
    assert module_2.BOM_UTF32 == b'\xff\xfe\x00\x00'
    assert module_2.BOM32_LE == b'\xff\xfe'
    assert module_2.BOM32_BE == b'\xfe\xff'
    assert module_2.BOM64_LE == b'\xff\xfe\x00\x00'
    assert module_2.BOM64_BE == b'\x00\x00\xfe\xff'
    var_6 = module_0.apply_overwrites_to_context(var_5, var_5)
    var_7 = var_5.__dir__()
    module_0.apply_overwrites_to_context(var_7, var_5, in_dictionary_variable=var_7)