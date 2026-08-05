# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import vulture.config as module_0
import tomli._parser as module_1

def test_case_0():
    pass

def test_case_1():
    var_0 = None
    var_1 = module_0.InputError(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'vulture.config.InputError'
    assert var_1.message is None
    assert module_0.DEFAULTS == {'config': 'pyproject.toml', 'min_confidence': 0, 'paths': [], 'exclude': [], 'ignore_decorators': [], 'ignore_names': [], 'make_whitelist': False, 'sort_by_size': False, 'verbose': False}

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = ()
    module_0.make_config(var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = 'MDGz'
    var_1 = module_0.make_config(var_0)
    assert module_0.DEFAULTS == {'config': 'pyproject.toml', 'min_confidence': 0, 'paths': [], 'exclude': [], 'ignore_decorators': [], 'ignore_names': [], 'make_whitelist': False, 'sort_by_size': False, 'verbose': False}
    module_1.parse_basic_str(var_1, var_1, multiline=var_1)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    var_1 = module_1.make_safe_parse_float(var_0)
    assert f'{type(module_1.annotations).__module__}.{type(module_1.annotations).__qualname__}' == '__future__._Feature'
    assert module_1.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_1.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_1.annotations.compiler_flag == 16777216
    assert f'{type(module_1.RE_DATETIME).__module__}.{type(module_1.RE_DATETIME).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.RE_LOCALTIME).__module__}.{type(module_1.RE_LOCALTIME).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.RE_NUMBER).__module__}.{type(module_1.RE_NUMBER).__qualname__}' == 're.Pattern'
    assert module_1.TYPE_CHECKING is False
    assert module_1.MAX_INLINE_NESTING == 1000
    assert module_1.MAX_KEY_PARTS == 1000
    assert f'{type(module_1.ASCII_CTRL).__module__}.{type(module_1.ASCII_CTRL).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.ASCII_CTRL) == 33
    assert f'{type(module_1.ILLEGAL_BASIC_STR_CHARS).__module__}.{type(module_1.ILLEGAL_BASIC_STR_CHARS).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.ILLEGAL_BASIC_STR_CHARS) == 32
    assert f'{type(module_1.ILLEGAL_MULTILINE_BASIC_STR_CHARS).__module__}.{type(module_1.ILLEGAL_MULTILINE_BASIC_STR_CHARS).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.ILLEGAL_MULTILINE_BASIC_STR_CHARS) == 31
    assert f'{type(module_1.ILLEGAL_LITERAL_STR_CHARS).__module__}.{type(module_1.ILLEGAL_LITERAL_STR_CHARS).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.ILLEGAL_LITERAL_STR_CHARS) == 32
    assert f'{type(module_1.ILLEGAL_MULTILINE_LITERAL_STR_CHARS).__module__}.{type(module_1.ILLEGAL_MULTILINE_LITERAL_STR_CHARS).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.ILLEGAL_MULTILINE_LITERAL_STR_CHARS) == 31
    assert f'{type(module_1.ILLEGAL_COMMENT_CHARS).__module__}.{type(module_1.ILLEGAL_COMMENT_CHARS).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.ILLEGAL_COMMENT_CHARS) == 32
    assert f'{type(module_1.TOML_WS).__module__}.{type(module_1.TOML_WS).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.TOML_WS) == 2
    assert f'{type(module_1.TOML_WS_AND_NEWLINE).__module__}.{type(module_1.TOML_WS_AND_NEWLINE).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.TOML_WS_AND_NEWLINE) == 3
    assert f'{type(module_1.BARE_KEY_CHARS).__module__}.{type(module_1.BARE_KEY_CHARS).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.BARE_KEY_CHARS) == 64
    assert f'{type(module_1.KEY_INITIAL_CHARS).__module__}.{type(module_1.KEY_INITIAL_CHARS).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.KEY_INITIAL_CHARS) == 66
    assert f'{type(module_1.HEXDIGIT_CHARS).__module__}.{type(module_1.HEXDIGIT_CHARS).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.HEXDIGIT_CHARS) == 22
    assert f'{type(module_1.BASIC_STR_ESCAPE_REPLACEMENTS).__module__}.{type(module_1.BASIC_STR_ESCAPE_REPLACEMENTS).__qualname__}' == 'builtins.mappingproxy'
    assert len(module_1.BASIC_STR_ESCAPE_REPLACEMENTS) == 8
    var_2 = var_1.__dir__()
    module_0.make_config(var_2, var_2)

def test_case_5():
    var_0 = '--exclude'
    var_1 = 'file1.py,file2.py'
    var_2 = '--ignore-names'
    var_3 = 'path'
    var_4 = [var_0, var_1, var_2, var_3, var_3]
    var_5 = module_0.make_config(var_4)
    assert module_0.DEFAULTS == {'config': 'pyproject.toml', 'min_confidence': 0, 'paths': [], 'exclude': [], 'ignore_decorators': [], 'ignore_names': [], 'make_whitelist': False, 'sort_by_size': False, 'verbose': False}

def test_case_6():
    var_0 = '--min-confidence'
    var_1 = '80'
    var_2 = 'src/'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.make_config(var_3)
    assert module_0.DEFAULTS == {'config': 'pyproject.toml', 'min_confidence': 0, 'paths': [], 'exclude': [], 'ignore_decorators': [], 'ignore_names': [], 'make_whitelist': False, 'sort_by_size': False, 'verbose': False}
    var_5 = '--exclude'
    var_6 = 'ignore1.py,ignore2.py'
    var_7 = [var_5, var_6, var_2]
    var_8 = module_0.make_config(var_7)
    var_9 = 'min_confidence'
    var_10 = 'not_an_int'
    var_11 = {var_9: var_10}
    with pytest.raises(module_0.InputError):
        module_0._check_input_config(var_11)

def test_case_7():
    var_0 = '--min-confidence'
    var_1 = '80'
    var_2 = 'src/'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.make_config(var_3)
    assert module_0.DEFAULTS == {'config': 'pyproject.toml', 'min_confidence': 0, 'paths': [], 'exclude': [], 'ignore_decorators': [], 'ignore_names': [], 'make_whitelist': False, 'sort_by_size': False, 'verbose': False}
    var_5 = '--exclude'
    var_6 = 'ignore1.py,ignore2.py'
    var_7 = [var_5, var_6, var_2]
    var_8 = module_0.make_config(var_7)
    var_9 = 'not_an_int'
    var_10 = {var_0: var_9}
    with pytest.raises(module_0.InputError):
        module_0._check_input_config(var_10)