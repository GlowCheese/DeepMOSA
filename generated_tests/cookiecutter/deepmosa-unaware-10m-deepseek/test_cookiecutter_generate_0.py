# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import cookiecutter.generate as module_0
import cookiecutter.exceptions as module_1
import cookiecutter.utils as module_2

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    var_1 = 'm4_:!3np8NAqtB[&8'
    var_2 = {var_1: var_1}
    var_3 = module_0.is_copy_only_path(var_0, var_2)
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
    module_0.generate_context(default_context=var_0, extra_context=var_3)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = True
    var_1 = 'd+@F[y{+a\\3g+'
    var_2 = '[F!Ao.Q!Z4Ex'
    var_3 = {var_1: var_2, var_2: var_1, var_2: var_1, var_1: var_0, var_2: var_1}
    var_4 = {var_1: var_2, var_1: var_0}
    var_5 = module_0.apply_overwrites_to_context(var_4, var_3)
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
    var_6 = ''
    module_0.is_copy_only_path(var_6, var_5)

@pytest.mark.xfail(strict=True)
def test_case_2():
    module_0.generate_context()

def test_case_3():
    var_0 = None
    with pytest.raises(module_1.EmptyDirNameException):
        module_0.render_and_create_dir(var_0, var_0, var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    module_0.generate_files(var_0, var_0)

def test_case_5():
    var_0 = '\nw)'
    var_1 = {var_0: var_0, var_0: var_0, var_0: var_0}
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

def test_case_6():
    var_0 = True
    var_1 = 'o\\\tw EWZQ'
    var_2 = {var_1: var_1, var_1: var_1, var_1: var_0}
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
def test_case_7():
    var_0 = 'Tj{cXV"=1:c'
    var_1 = None
    module_0.render_and_create_dir(var_0, var_1, var_0, var_1)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = None
    var_1 = True
    var_2 = 'o\\\tw EWZQ'
    var_3 = {var_0: var_2, var_0: var_0, var_0: var_1}
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
    var_5 = {var_2: var_0, var_2: var_2}
    var_6 = 'T{'
    var_7 = 'HquQ'
    var_8 = {var_6: var_4, var_2: var_5, var_7: var_1}
    var_9 = True
    module_0.generate_files(var_0, var_8, skip_if_file_exists=var_9)

def test_case_9():
    var_0 = '{d'
    var_1 = True
    var_2 = '[F!Ao.Q!Z4Ex'
    var_3 = {var_0: var_2, var_2: var_0, var_0: var_1}
    var_4 = {var_0: var_2, var_0: var_1}
    var_5 = module_0.apply_overwrites_to_context(var_4, var_3)
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

def test_case_10():
    var_0 = False
    var_1 = 'd+@F[y{-+a\\3g+'
    var_2 = 'rYtee5Y}vx!q::4k`*M'
    var_3 = {var_1: var_2, var_2: var_1, var_1: var_0}
    var_4 = {var_1: var_2, var_2: var_0}
    with pytest.raises(ValueError):
        module_0.apply_overwrites_to_context(var_4, var_3)

def test_case_11():
    var_0 = True
    var_1 = 'd+@F[y{-+a\\3g+'
    var_2 = 'rYtee5Y}vx!q::4k`*M'
    var_3 = {var_1: var_2, var_2: var_1, var_1: var_0}
    var_4 = {var_1: var_2, var_2: var_0, var_2: var_0, var_1: var_2, var_1: var_3}
    with pytest.raises(ValueError):
        module_0.apply_overwrites_to_context(var_4, var_3)

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = None
    var_1 = '=z;dBh{r0'
    var_2 = [var_1, var_1, var_1]
    var_3 = '\nf\x0c'
    var_4 = {var_3: var_0, var_3: var_2, var_1: var_0, var_3: var_3, var_3: var_1}
    var_5 = '?>'
    var_6 = {var_1: var_4, var_5: var_1, var_5: var_5}
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
    var_8 = module_2.create_env_with_context(var_4)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'cookiecutter.environment.StrictEnvironment'
    assert var_8.block_start_string == '{%'
    assert var_8.block_end_string == '%}'
    assert var_8.variable_start_string == '{{'
    assert var_8.variable_end_string == '}}'
    assert var_8.comment_start_string == '{#'
    assert var_8.comment_end_string == '#}'
    assert var_8.line_statement_prefix is None
    assert var_8.line_comment_prefix is None
    assert var_8.trim_blocks is False
    assert var_8.lstrip_blocks is False
    assert var_8.newline_sequence == '\n'
    assert var_8.keep_trailing_newline is True
    assert var_8.optimized is True
    assert var_8.finalize is None
    assert var_8.autoescape is False
    assert f'{type(var_8.filters).__module__}.{type(var_8.filters).__qualname__}' == 'builtins.dict'
    assert len(var_8.filters) == 56
    assert f'{type(var_8.tests).__module__}.{type(var_8.tests).__qualname__}' == 'builtins.dict'
    assert len(var_8.tests) == 39
    assert f'{type(var_8.globals).__module__}.{type(var_8.globals).__qualname__}' == 'builtins.dict'
    assert len(var_8.globals) == 8
    assert var_8.loader is None
    assert f'{type(var_8.cache).__module__}.{type(var_8.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_8.cache) == 0
    assert var_8.bytecode_cache is None
    assert var_8.auto_reload is True
    assert var_8.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_8.datetime_format == '%Y-%m-%d'
    assert f'{type(var_8.extensions).__module__}.{type(var_8.extensions).__qualname__}' == 'builtins.dict'
    assert len(var_8.extensions) == 5
    assert var_8.is_async is False
    assert f'{type(module_2.annotations).__module__}.{type(module_2.annotations).__qualname__}' == '__future__._Feature'
    assert module_2.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_2.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_2.annotations.compiler_flag == 16777216
    assert module_2.TYPE_CHECKING is False
    assert f'{type(module_2.logger).__module__}.{type(module_2.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_2.logger.filters == []
    assert module_2.logger.name == 'cookiecutter.utils'
    assert module_2.logger.level == 0
    assert f'{type(module_2.logger.parent).__module__}.{type(module_2.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_2.logger.propagate is True
    assert module_2.logger.handlers == []
    assert module_2.logger.disabled is False
    assert f'{type(module_2.logger.manager).__module__}.{type(module_2.logger.manager).__qualname__}' == 'logging.Manager'
    module_0.generate_file(var_3, var_3, var_6, var_8)

def test_case_13():
    var_0 = None
    var_1 = True
    var_2 = '+@F[y+\\3g+'
    var_3 = 'Va\na'
    var_4 = [var_3, var_3, var_3]
    var_5 = {var_3: var_0, var_3: var_4, var_2: var_1, var_2: var_2, var_2: var_2}
    var_6 = {var_2: var_2, var_2: var_2, var_2: var_2, var_2: var_1}
    var_7 = True
    with pytest.raises(ValueError):
        module_0.apply_overwrites_to_context(var_6, var_5, in_dictionary_variable=var_7)

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = None
    var_1 = 'Va\na'
    var_2 = [var_1, var_1, var_1, var_1]
    var_3 = '\nf\x0c'
    var_4 = {var_1: var_1, var_1: var_0, var_1: var_2, var_3: var_1}
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
    module_0.generate_context(extra_context=var_5)

def test_case_15():
    var_0 = None
    var_1 = '+@F[y+\\3g+'
    var_2 = [var_1, var_1, var_1, var_1]
    var_3 = '\nf]\x0c'
    var_4 = {var_1: var_1, var_1: var_0, var_1: var_2, var_3: var_1}
    var_5 = 'Y|*=MlEd9gW8>v'
    var_6 = module_0.is_copy_only_path(var_5, var_4)
    assert var_6 is False
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
    var_7 = module_0.apply_overwrites_to_context(var_4, var_4)
    var_8 = '[F!Ao.Q!Z4Ex'
    var_9 = {var_1: var_6}
    var_10 = {var_1: var_8, var_1: var_6}
    var_11 = module_0.apply_overwrites_to_context(var_10, var_9)
    with pytest.raises(ValueError):
        module_0.apply_overwrites_to_context(var_4, var_9, in_dictionary_variable=var_11)

def test_case_16():
    var_0 = 'name'
    var_1 = 'version'
    var_2 = 'old'
    var_3 = '1.0'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'new'
    var_6 = {var_0: var_5}
    var_7 = module_0.apply_overwrites_to_context(var_4, var_6)
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
    var_8 = 'value'
    var_9 = {var_2: var_8}
    var_10 = 'new_var'
    var_11 = 'new_value'
    var_12 = {var_10: var_11}
    var_13 = module_0.apply_overwrites_to_context(var_9, var_12)
    var_14 = 'choices'
    var_15 = 'a'
    var_16 = 'b'
    var_17 = 'c'
    var_18 = 'd'
    var_19 = [var_15, var_16, var_17, var_18]
    var_20 = {var_14: var_19}
    var_21 = [var_16, var_17]
    var_22 = {var_14: var_21}
    var_23 = module_0.apply_overwrites_to_context(var_20, var_22)
    var_24 = [var_15, var_16, var_17]
    var_25 = {var_14: var_24}
    var_26 = [var_16, var_18]
    var_27 = {var_14: var_26}
    with pytest.raises(ValueError):
        module_0.apply_overwrites_to_context(var_25, var_27)

def test_case_17():
    var_0 = 'name'
    var_1 = 'version'
    var_2 = 'old_value'
    var_3 = '1.0'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'new_value'
    var_6 = {var_0: var_5}
    var_7 = module_0.apply_overwrites_to_context(var_4, var_6)
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
    var_8 = 'existing'
    var_9 = 'value'
    var_10 = {var_8: var_9}
    var_11 = 'new_var'
    var_12 = {var_11: var_5}
    var_13 = module_0.apply_overwrites_to_context(var_10, var_12)
    var_14 = 'choice_var'
    var_15 = 'option1'
    var_16 = 'option2'
    var_17 = 'option3'
    var_18 = [var_15, var_16, var_17]
    var_19 = {var_14: var_18}
    var_20 = {var_14: var_16}
    var_21 = module_0.apply_overwrites_to_context(var_19, var_20)
    var_22 = [var_15, var_16]
    var_23 = {var_14: var_22}
    var_24 = 'invalid_option'
    var_25 = {var_14: var_24}
    with pytest.raises(ValueError):
        module_0.apply_overwrites_to_context(var_23, var_25)

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = '\nf\x0c'
    var_1 = '$)9*T3'
    var_2 = {}
    var_3 = module_0.is_copy_only_path(var_0, var_2)
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
    var_4 = module_2.create_env_with_context(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'cookiecutter.environment.StrictEnvironment'
    assert var_4.block_start_string == '{%'
    assert var_4.block_end_string == '%}'
    assert var_4.variable_start_string == '{{'
    assert var_4.variable_end_string == '}}'
    assert var_4.comment_start_string == '{#'
    assert var_4.comment_end_string == '#}'
    assert var_4.line_statement_prefix is None
    assert var_4.line_comment_prefix is None
    assert var_4.trim_blocks is False
    assert var_4.lstrip_blocks is False
    assert var_4.newline_sequence == '\n'
    assert var_4.keep_trailing_newline is True
    assert var_4.optimized is True
    assert var_4.finalize is None
    assert var_4.autoescape is False
    assert f'{type(var_4.filters).__module__}.{type(var_4.filters).__qualname__}' == 'builtins.dict'
    assert len(var_4.filters) == 56
    assert f'{type(var_4.tests).__module__}.{type(var_4.tests).__qualname__}' == 'builtins.dict'
    assert len(var_4.tests) == 39
    assert f'{type(var_4.globals).__module__}.{type(var_4.globals).__qualname__}' == 'builtins.dict'
    assert len(var_4.globals) == 8
    assert var_4.loader is None
    assert f'{type(var_4.cache).__module__}.{type(var_4.cache).__qualname__}' == 'jinja2.utils.LRUCache'
    assert len(var_4.cache) == 0
    assert var_4.bytecode_cache is None
    assert var_4.auto_reload is True
    assert var_4.policies == {'compiler.ascii_str': True, 'urlize.rel': 'noopener', 'urlize.target': None, 'urlize.extra_schemes': None, 'truncate.leeway': 5, 'json.dumps_function': None, 'json.dumps_kwargs': {'sort_keys': True}, 'ext.i18n.trimmed': False}
    assert var_4.datetime_format == '%Y-%m-%d'
    assert f'{type(var_4.extensions).__module__}.{type(var_4.extensions).__qualname__}' == 'builtins.dict'
    assert len(var_4.extensions) == 5
    assert var_4.is_async is False
    assert f'{type(module_2.annotations).__module__}.{type(module_2.annotations).__qualname__}' == '__future__._Feature'
    assert module_2.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_2.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_2.annotations.compiler_flag == 16777216
    assert module_2.TYPE_CHECKING is False
    assert f'{type(module_2.logger).__module__}.{type(module_2.logger).__qualname__}' == 'libs.custom_logger.logger.CustomLogger'
    assert module_2.logger.filters == []
    assert module_2.logger.name == 'cookiecutter.utils'
    assert module_2.logger.level == 0
    assert f'{type(module_2.logger.parent).__module__}.{type(module_2.logger.parent).__qualname__}' == 'logging.RootLogger'
    assert module_2.logger.propagate is True
    assert module_2.logger.handlers == []
    assert module_2.logger.disabled is False
    assert f'{type(module_2.logger.manager).__module__}.{type(module_2.logger.manager).__qualname__}' == 'logging.Manager'
    module_0.render_and_create_dir(var_1, var_2, var_0, var_4)