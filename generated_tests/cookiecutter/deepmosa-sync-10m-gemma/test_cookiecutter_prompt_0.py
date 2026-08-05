# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import re as module_0
import cookiecutter.prompt as module_1
import rich.prompt as module_2
import jinja2.environment as module_3
import cookiecutter.exceptions as module_4
import pathlib as module_5

def test_case_0():
    pass

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = module_0.purge()
    assert module_0.ASCII == module_0.RegexFlag.ASCII
    assert module_0.A == module_0.RegexFlag.ASCII
    assert module_0.IGNORECASE == module_0.RegexFlag.IGNORECASE
    assert module_0.I == module_0.RegexFlag.IGNORECASE
    assert module_0.LOCALE == module_0.RegexFlag.LOCALE
    assert module_0.L == module_0.RegexFlag.LOCALE
    assert module_0.UNICODE == module_0.RegexFlag.UNICODE
    assert module_0.U == module_0.RegexFlag.UNICODE
    assert module_0.MULTILINE == module_0.RegexFlag.MULTILINE
    assert module_0.M == module_0.RegexFlag.MULTILINE
    assert module_0.DOTALL == module_0.RegexFlag.DOTALL
    assert module_0.S == module_0.RegexFlag.DOTALL
    assert module_0.VERBOSE == module_0.RegexFlag.VERBOSE
    assert module_0.X == module_0.RegexFlag.VERBOSE
    assert module_0.TEMPLATE == module_0.RegexFlag.TEMPLATE
    assert module_0.T == module_0.RegexFlag.TEMPLATE
    assert module_0.DEBUG == module_0.RegexFlag.DEBUG
    module_1.read_user_variable(var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    module_1.read_user_yes_no(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = 'en$:2'
    module_1.read_user_choice(var_0, var_0, var_0)

def test_case_4():
    var_0 = 'o`A(Jt\x0cQ5E'
    with pytest.raises(module_2.InvalidResponse):
        module_1.process_json(var_0)

def test_case_5():
    var_0 = None
    with pytest.raises(TypeError):
        module_1.read_user_dict(var_0, var_0, prefix=var_0)

def test_case_6():
    var_0 = None
    var_1 = module_1.render_variable(var_0, var_0, var_0)
    assert f'{type(module_1.annotations).__module__}.{type(module_1.annotations).__qualname__}' == '__future__._Feature'
    assert module_1.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_1.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_1.annotations.compiler_flag == 16777216
    assert module_1.TYPE_CHECKING is False
    assert module_1.DEFAULT_DISPLAY == 'default'

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = 'o`u,Nt\x0cQ5E'
    module_1.prompt_and_delete(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = None
    module_1.prompt_and_delete(var_0)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = None
    module_1.read_repo_password(var_0)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = None
    var_1 = module_1.JsonPrompt(choices=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'cookiecutter.prompt.JsonPrompt'
    assert f'{type(var_1.console).__module__}.{type(var_1.console).__qualname__}' == 'rich.console.Console'
    assert f'{type(var_1.prompt).__module__}.{type(var_1.prompt).__qualname__}' == 'rich.text.Text'
    assert len(var_1.prompt) == 0
    assert var_1.password is False
    assert var_1.case_sensitive is True
    assert var_1.show_default is True
    assert var_1.show_choices is True
    assert f'{type(module_1.annotations).__module__}.{type(module_1.annotations).__qualname__}' == '__future__._Feature'
    assert module_1.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_1.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_1.annotations.compiler_flag == 16777216
    assert module_1.TYPE_CHECKING is False
    assert module_1.DEFAULT_DISPLAY == 'default'
    assert module_1.JsonPrompt.default is None
    assert module_1.JsonPrompt.validate_error_message == '[prompt.invalid]  Please enter a valid JSON string'
    var_1.process_response(var_0)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = None
    var_1 = [var_0, var_0, var_0, var_0]
    module_1.read_user_choice(var_0, var_1, var_0, var_0)

def test_case_12():
    var_0 = None
    with pytest.raises(ValueError):
        module_1.read_user_choice(var_0, var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = '>5>~2\nialR'
    var_1 = {var_0: var_0, var_0: var_0}
    var_2 = None
    module_1.prompt_choice_for_template(var_0, var_1, var_2)

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = 'delete'
    var_1 = 'Are you sure you want to delete this?'
    var_2 = {var_1: var_1, var_0: var_1, var_0: var_1}
    var_3 = False
    module_1.read_user_yes_no(var_0, var_3, var_2)
    assert var_4 is False

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = 123
    module_1.render_variable(var_0, var_0, var_0)
    assert var_1 == '123'

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = '\x0b_@l%P:af]9\t'
    var_1 = None
    module_1.prompt_choice_for_config(var_1, var_1, var_1, var_0, var_1, prefix=var_1)

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = 'o`u,Nt\x0cQ5E'
    module_1.render_variable(var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = 'cookiecutter'
    var_1 = {var_0: var_0, var_0: var_0}
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_2}
    var_4 = False
    module_1.prompt_for_config(var_3, var_4)

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = 'uer^da'
    var_1 = 'Enter details'
    var_2 = {var_0: var_1}
    module_1.read_user_dict(var_0, var_2, var_2, var_0)

def test_case_20():
    var_0 = {}
    var_1 = module_1._prompts_from_options(var_0)
    assert f'{type(module_1.annotations).__module__}.{type(module_1.annotations).__qualname__}' == '__future__._Feature'
    assert module_1.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_1.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_1.annotations.compiler_flag == 16777216
    assert module_1.TYPE_CHECKING is False
    assert module_1.DEFAULT_DISPLAY == 'default'
    var_2 = bool(var_1 == {'__prompt__': 'Select a template'})
    assert var_2 is True

def test_case_21():
    var_0 = 'choice1'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = '__prompt__'
    var_4 = 'Select a template'
    var_5 = {var_3: var_4, var_0: var_0}
    var_6 = module_1._prompts_from_options(var_2)
    assert f'{type(module_1.annotations).__module__}.{type(module_1.annotations).__qualname__}' == '__future__._Feature'
    assert module_1.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_1.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_1.annotations.compiler_flag == 16777216
    assert module_1.TYPE_CHECKING is False
    assert module_1.DEFAULT_DISPLAY == 'default'
    var_7 = bool(var_6 == var_5)
    assert var_7 is True

def test_case_22():
    var_0 = 'choice1'
    var_1 = 'description'
    var_2 = 'Only Description'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = '__prompt__'
    var_6 = 'Select a template'
    var_7 = 'choice1 (Only Description)'
    var_8 = {var_5: var_6, var_0: var_7}
    var_9 = module_1._prompts_from_options(var_4)
    assert f'{type(module_1.annotations).__module__}.{type(module_1.annotations).__qualname__}' == '__future__._Feature'
    assert module_1.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_1.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_1.annotations.compiler_flag == 16777216
    assert module_1.TYPE_CHECKING is False
    assert module_1.DEFAULT_DISPLAY == 'default'
    var_10 = bool(var_9 == var_8)
    assert var_10 is True

def test_case_23():
    var_0 = 'cookiecutter'
    var_1 = 'is_enabled'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = module_1.prompt_for_config(var_4, var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'collections.OrderedDict'
    assert len(var_5) == 1
    assert f'{type(module_1.annotations).__module__}.{type(module_1.annotations).__qualname__}' == '__future__._Feature'
    assert module_1.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_1.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_1.annotations.compiler_flag == 16777216
    assert module_1.TYPE_CHECKING is False
    assert module_1.DEFAULT_DISPLAY == 'default'

def test_case_24():
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
    var_1 = 'S"^'
    var_2 = [var_1, var_1]
    var_3 = module_1.JsonPrompt(choices=var_2, case_sensitive=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'cookiecutter.prompt.JsonPrompt'
    assert f'{type(var_3.console).__module__}.{type(var_3.console).__qualname__}' == 'rich.console.Console'
    assert f'{type(var_3.prompt).__module__}.{type(var_3.prompt).__qualname__}' == 'rich.text.Text'
    assert len(var_3.prompt) == 0
    assert var_3.password is False
    assert var_3.choices == ['S"^', 'S"^']
    assert var_3.case_sensitive == ['S"^', 'S"^']
    assert var_3.show_default is True
    assert var_3.show_choices is True
    assert f'{type(module_1.annotations).__module__}.{type(module_1.annotations).__qualname__}' == '__future__._Feature'
    assert module_1.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_1.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_1.annotations.compiler_flag == 16777216
    assert module_1.TYPE_CHECKING is False
    assert module_1.DEFAULT_DISPLAY == 'default'
    assert module_1.JsonPrompt.default is None
    assert module_1.JsonPrompt.validate_error_message == '[prompt.invalid]  Please enter a valid JSON string'
    var_4 = {}
    var_5 = None
    var_6 = module_1.render_variable(var_5, var_4, var_5)

def test_case_25():
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
    var_1 = 'cookiecutter'
    var_2 = 'name'
    var_3 = 'test'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = '{{ cookiecutter.name }}'
    var_7 = 'static'
    var_8 = [var_6, var_7]
    var_9 = module_1.render_variable(var_0, var_8, var_5)
    assert f'{type(module_1.annotations).__module__}.{type(module_1.annotations).__qualname__}' == '__future__._Feature'
    assert module_1.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_1.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_1.annotations.compiler_flag == 16777216
    assert module_1.TYPE_CHECKING is False
    assert module_1.DEFAULT_DISPLAY == 'default'
    var_10 = bool(var_9 == ['test', 'static'])

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
    var_1 = {}
    var_2 = []
    var_3 = 'key'
    var_4 = True
    with pytest.raises(ValueError):
        module_1.prompt_choice_for_config(var_1, var_0, var_3, var_2, var_4)

@pytest.mark.xfail(strict=True)
def test_case_27():
    var_0 = 'cookiecutter'
    var_1 = []
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_2}
    var_4 = False
    module_1.prompt_for_config(var_3, var_4)

def test_case_28():
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
    var_1 = 'project_name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = '{{ cookiecutter.project_name }}_repo'
    var_5 = '{{ cookiecutter.project_name }}_app'
    var_6 = [var_4, var_5]
    var_7 = 'key'
    var_8 = True
    var_9 = module_1.prompt_choice_for_config(var_3, var_0, var_7, var_6, var_8)
    assert var_9 == 'test_repo'
    assert f'{type(module_1.annotations).__module__}.{type(module_1.annotations).__qualname__}' == '__future__._Feature'
    assert module_1.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_1.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_1.annotations.compiler_flag == 16777216
    assert module_1.TYPE_CHECKING is False
    assert module_1.DEFAULT_DISPLAY == 'default'

@pytest.mark.xfail(strict=True)
def test_case_29():
    var_0 = module_0.RegexFlag.VERBOSE
    module_1.read_user_variable(var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_30():
    var_0 = None
    var_1 = {}
    module_1.prompt_choice_for_template(var_0, var_1, var_0)

def test_case_31():
    var_0 = '"just a string"'
    with pytest.raises(module_2.InvalidResponse):
        module_1.process_json(var_0)

@pytest.mark.xfail(strict=True)
def test_case_32():
    var_0 = {}
    var_1 = '\r4bmb,%d[s9bMEv^'
    var_2 = True
    module_1.prompt_choice_for_template(var_1, var_0, var_2)

def test_case_33():
    var_0 = 'd@b]DS&`V'
    var_1 = None
    var_2 = module_1.YesNoPrompt(var_0, console=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'cookiecutter.prompt.YesNoPrompt'
    assert f'{type(var_2.console).__module__}.{type(var_2.console).__qualname__}' == 'rich.console.Console'
    assert f'{type(var_2.prompt).__module__}.{type(var_2.prompt).__qualname__}' == 'rich.text.Text'
    assert len(var_2.prompt) == 9
    assert var_2.password is False
    assert var_2.case_sensitive is True
    assert var_2.show_default is True
    assert var_2.show_choices is True
    assert f'{type(module_1.annotations).__module__}.{type(module_1.annotations).__qualname__}' == '__future__._Feature'
    assert module_1.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_1.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_1.annotations.compiler_flag == 16777216
    assert module_1.TYPE_CHECKING is False
    assert module_1.DEFAULT_DISPLAY == 'default'
    assert module_1.YesNoPrompt.yes_choices == ['1', 'true', 't', 'yes', 'y', 'on']
    assert module_1.YesNoPrompt.no_choices == ['0', 'false', 'f', 'no', 'n', 'off']
    with pytest.raises(module_2.InvalidResponse):
        var_2.process_response(var_0)

@pytest.mark.xfail(strict=True)
def test_case_34():
    var_0 = 'en$:2('
    var_1 = '&:\x0c@'
    var_2 = [var_0, var_1, var_0]
    module_1.read_user_choice(var_1, var_2, var_0)

@pytest.mark.xfail(strict=True)
def test_case_35():
    var_0 = 'en$2'
    var_1 = '1?vd<S0'
    var_2 = None
    module_1.read_user_variable(var_1, var_2, var_0)

@pytest.mark.xfail(strict=True)
def test_case_36():
    var_0 = ']keyS'
    var_1 = {var_0: var_0}
    var_2 = 'nukr '
    module_1.read_user_dict(var_2, var_1, var_1, var_2)

def test_case_37():
    var_0 = '{"a": 1, "b": 2, "c": 3}'
    var_1 = module_1.process_json(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'collections.OrderedDict'
    assert len(var_1) == 3
    assert f'{type(module_1.annotations).__module__}.{type(module_1.annotations).__qualname__}' == '__future__._Feature'
    assert module_1.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_1.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_1.annotations.compiler_flag == 16777216
    assert module_1.TYPE_CHECKING is False
    assert module_1.DEFAULT_DISPLAY == 'default'

def test_case_38():
    var_0 = 'cookiecutter'
    var_1 = '_internal_var'
    var_2 = 'public_var'
    var_3 = {var_1: var_0, var_2: var_1}
    var_4 = {var_0: var_3}
    var_5 = True
    var_6 = module_1.prompt_for_config(var_4, var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'collections.OrderedDict'
    assert len(var_6) == 2
    assert f'{type(module_1.annotations).__module__}.{type(module_1.annotations).__qualname__}' == '__future__._Feature'
    assert module_1.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_1.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_1.annotations.compiler_flag == 16777216
    assert module_1.TYPE_CHECKING is False
    assert module_1.DEFAULT_DISPLAY == 'default'

def test_case_39():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'broken'
    var_3 = 'test'
    var_4 = '{{ cookiecutter.non_existent }}'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = True
    with pytest.raises(module_4.UndefinedVariableInTemplate):
        module_1.prompt_for_config(var_6, var_7)

def test_case_40():
    var_0 = 'cookiecutter'
    var_1 = 'metadata'
    var_2 = '4fF<o;M 6z6xXl4=r'
    var_3 = 'a&mi(;n'
    var_4 = 'MIT'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = True
    var_9 = module_1.prompt_for_config(var_7, var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'collections.OrderedDict'
    assert len(var_9) == 1
    assert f'{type(module_1.annotations).__module__}.{type(module_1.annotations).__qualname__}' == '__future__._Feature'
    assert module_1.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_1.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_1.annotations.compiler_flag == 16777216
    assert module_1.TYPE_CHECKING is False
    assert module_1.DEFAULT_DISPLAY == 'default'
    var_10 = var_9['metadata']
    var_11 = bool(var_9['metadata'] == {'author': 'admin', 'license': 'MIT'})

@pytest.mark.xfail(strict=True)
def test_case_41():
    var_0 = 'cookiecutter'
    var_1 = module_0.purge()
    assert module_0.ASCII == module_0.RegexFlag.ASCII
    assert module_0.A == module_0.RegexFlag.ASCII
    assert module_0.IGNORECASE == module_0.RegexFlag.IGNORECASE
    assert module_0.I == module_0.RegexFlag.IGNORECASE
    assert module_0.LOCALE == module_0.RegexFlag.LOCALE
    assert module_0.L == module_0.RegexFlag.LOCALE
    assert module_0.UNICODE == module_0.RegexFlag.UNICODE
    assert module_0.U == module_0.RegexFlag.UNICODE
    assert module_0.MULTILINE == module_0.RegexFlag.MULTILINE
    assert module_0.M == module_0.RegexFlag.MULTILINE
    assert module_0.DOTALL == module_0.RegexFlag.DOTALL
    assert module_0.S == module_0.RegexFlag.DOTALL
    assert module_0.VERBOSE == module_0.RegexFlag.VERBOSE
    assert module_0.X == module_0.RegexFlag.VERBOSE
    assert module_0.TEMPLATE == module_0.RegexFlag.TEMPLATE
    assert module_0.T == module_0.RegexFlag.TEMPLATE
    assert module_0.DEBUG == module_0.RegexFlag.DEBUG
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_2}
    var_4 = False
    module_1.prompt_for_config(var_3, var_4)

@pytest.mark.xfail(strict=True)
def test_case_42():
    var_0 = 'test_var'
    var_1 = 'Custom Question'
    var_2 = {var_0: var_1}
    var_3 = 'test_var'
    var_4 = 'default'
    module_1.read_user_variable(var_3, var_4, var_2)
    assert var_5 == 'user_input'

@pytest.mark.xfail(strict=True)
def test_case_43():
    var_0 = 'cookiecutter'
    var_1 = {var_0: var_0, var_0: var_0, var_0: var_0}
    module_1.read_user_choice(var_0, var_0, var_1, var_0)

def test_case_44():
    var_0 = 'cookiecutter'
    var_1 = 'config_dict'
    var_2 = 'Test'
    var_3 = 'key'
    var_4 = '{{ cookiecutter.project_name }}'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_2, var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = True
    with pytest.raises(module_4.UndefinedVariableInTemplate):
        module_1.prompt_for_config(var_7, var_8)

@pytest.mark.xfail(strict=True)
def test_case_45():
    var_0 = 'cookiecutter'
    var_1 = 'templates'
    var_2 = 'option1'
    var_3 = '`DWu)J#y'
    var_4 = {var_3: var_3}
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = '/tmp'
    module_1.choose_nested_template(var_7, var_8)

@pytest.mark.xfail(strict=True)
def test_case_46():
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
    var_1 = 'project_name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = '{{ cookiecutter.project_name }}'
    var_5 = [var_4]
    var_6 = 'key'
    var_7 = '__prompt__'
    var_8 = 'Custom Prompt'
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = False
    module_1.prompt_choice_for_config(var_3, var_0, var_6, var_5, var_11, var_10)
    assert var_12 == 'test'

def test_case_47():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'default_name'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    with pytest.raises(ValueError):
        module_1.choose_nested_template(var_4, var_1)

def test_case_48():
    var_0 = 'cookiecutter'
    var_1 = 'template'
    var_2 = 'choice1 (templates/old_choice)'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}
    var_6 = '.'
    var_7 = [var_6]
    var_8 = module_5.Path(*var_7, **var_4)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pathlib.PosixPath'
    assert module_5.EINVAL == 22
    assert module_5.ENOENT == 2
    assert module_5.ENOTDIR == 20
    assert module_5.EBADF == 9
    assert module_5.ELOOP == 40
    var_9 = var_8.resolve()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pathlib.PosixPath'
    var_10 = True
    var_11 = module_1.choose_nested_template(var_5, var_9, var_10)
    assert var_11 == '/workspace/run/templates/old_choice'
    assert f'{type(module_1.annotations).__module__}.{type(module_1.annotations).__qualname__}' == '__future__._Feature'
    assert module_1.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_1.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_1.annotations.compiler_flag == 16777216
    assert module_1.TYPE_CHECKING is False
    assert module_1.DEFAULT_DISPLAY == 'default'

def test_case_49():
    var_0 = 'cookiecutter'
    var_1 = 'templates'
    var_2 = 'option1'
    var_3 = 'path'
    var_4 = '/absolute/path/to/template'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = '.'
    var_10 = [var_9]
    var_11 = {}
    var_12 = module_5.Path(*var_10, **var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pathlib.PosixPath'
    assert module_5.EINVAL == 22
    assert module_5.ENOENT == 2
    assert module_5.ENOTDIR == 20
    assert module_5.EBADF == 9
    assert module_5.ELOOP == 40
    var_13 = var_12.resolve()
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pathlib.PosixPath'
    var_14 = True
    with pytest.raises(ValueError):
        module_1.choose_nested_template(var_8, var_13, var_14)

def test_case_50():
    var_0 = 'test_directory_to_delete'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_5.Path(*var_1, **var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pathlib.PosixPath'
    assert module_5.EINVAL == 22
    assert module_5.ENOENT == 2
    assert module_5.ENOTDIR == 20
    assert module_5.EBADF == 9
    assert module_5.ELOOP == 40
    var_4 = True
    var_5 = var_3.mkdir(parents=var_4, exist_ok=var_4)
    var_6 = True
    var_7 = module_1.prompt_and_delete(var_3, var_6)
    assert var_7 is True
    assert f'{type(module_1.annotations).__module__}.{type(module_1.annotations).__qualname__}' == '__future__._Feature'
    assert module_1.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_1.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_1.annotations.compiler_flag == 16777216
    assert module_1.TYPE_CHECKING is False
    assert module_1.DEFAULT_DISPLAY == 'default'
    var_8 = var_3.exists()
    assert var_8 is False
    var_9 = bool(not var_8)
    assert var_9 is True
    var_10 = '.'
    var_11 = [var_10]
    var_12 = {}
    var_13 = module_5.Path(*var_11, **var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pathlib.PosixPath'

@pytest.mark.xfail(strict=True)
def test_case_51():
    var_0 = 'cookiecutter'
    var_1 = 'is_enaied'
    var_2 = False
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    module_1.prompt_for_config(var_4, var_2)

def test_case_52():
    var_0 = module_1.YesNoPrompt()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'cookiecutter.prompt.YesNoPrompt'
    assert f'{type(var_0.console).__module__}.{type(var_0.console).__qualname__}' == 'rich.console.Console'
    assert f'{type(var_0.prompt).__module__}.{type(var_0.prompt).__qualname__}' == 'rich.text.Text'
    assert len(var_0.prompt) == 0
    assert var_0.password is False
    assert var_0.case_sensitive is True
    assert var_0.show_default is True
    assert var_0.show_choices is True
    assert f'{type(module_1.annotations).__module__}.{type(module_1.annotations).__qualname__}' == '__future__._Feature'
    assert module_1.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_1.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_1.annotations.compiler_flag == 16777216
    assert module_1.TYPE_CHECKING is False
    assert module_1.DEFAULT_DISPLAY == 'default'
    assert module_1.YesNoPrompt.yes_choices == ['1', 'true', 't', 'yes', 'y', 'on']
    assert module_1.YesNoPrompt.no_choices == ['0', 'false', 'f', 'no', 'n', 'off']
    var_1 = '1'
    var_2 = var_0.process_response(var_1)
    assert var_2 is True
    var_3 = 'T'
    var_4 = var_0.process_response(var_3)
    assert var_4 is True
    var_5 = 'YES'
    var_6 = var_0.process_response(var_5)
    assert var_6 is True
    var_7 = 'y'
    var_8 = var_0.process_response(var_7)
    assert var_8 is True
    var_9 = '  on  '
    var_10 = var_0.process_response(var_9)
    assert var_10 is True

@pytest.mark.xfail(strict=True)
def test_case_53():
    var_0 = module_1.YesNoPrompt()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'cookiecutter.prompt.YesNoPrompt'
    assert f'{type(var_0.console).__module__}.{type(var_0.console).__qualname__}' == 'rich.console.Console'
    assert f'{type(var_0.prompt).__module__}.{type(var_0.prompt).__qualname__}' == 'rich.text.Text'
    assert len(var_0.prompt) == 0
    assert var_0.password is False
    assert var_0.case_sensitive is True
    assert var_0.show_default is True
    assert var_0.show_choices is True
    assert f'{type(module_1.annotations).__module__}.{type(module_1.annotations).__qualname__}' == '__future__._Feature'
    assert module_1.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_1.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_1.annotations.compiler_flag == 16777216
    assert module_1.TYPE_CHECKING is False
    assert module_1.DEFAULT_DISPLAY == 'default'
    assert module_1.YesNoPrompt.yes_choices == ['1', 'true', 't', 'yes', 'y', 'on']
    assert module_1.YesNoPrompt.no_choices == ['0', 'false', 'f', 'no', 'n', 'off']
    var_1 = '0'
    var_2 = var_0.process_response(var_1)
    assert var_2 is False
    var_3 = 'false'
    var_4 = var_0.process_response(var_3)
    assert var_4 is False
    var_5 = 'f'
    var_6 = var_0.process_response(var_5)
    assert var_6 is False
    var_7 = '?'
    var_8 = None
    module_1.read_user_yes_no(var_8, var_7)

def test_case_54():
    var_0 = 'cookiecutter'
    var_1 = '_private_var'
    var_2 = '__internal_var__'
    var_3 = 'some_value'
    var_4 = 'template_string'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = True
    var_8 = module_1.prompt_for_config(var_6, var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'collections.OrderedDict'
    assert len(var_8) == 2
    assert f'{type(module_1.annotations).__module__}.{type(module_1.annotations).__qualname__}' == '__future__._Feature'
    assert module_1.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_1.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_1.annotations.compiler_flag == 16777216
    assert module_1.TYPE_CHECKING is False
    assert module_1.DEFAULT_DISPLAY == 'default'
    var_9 = var_8['_private_var']
    assert var_9 == 'some_value'