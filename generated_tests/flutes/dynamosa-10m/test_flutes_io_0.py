# Check out: https://github.com/GlowCheese/deepmosa
import flutes.io as module_0
import pytest


def test_case_0():
    var_0 = module_0.shut_up()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'contextlib._GeneratorContextManager'
    assert f'{type(var_0.gen).__module__}.{type(var_0.gen).__qualname__}' == 'builtins.generator'
    assert var_0.args == ()
    assert var_0.kwds == {}
    assert f'{type(module_0.AnyStr).__module__}.{type(module_0.AnyStr).__qualname__}' == 'typing.TypeVar'
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

def test_case_1():
    var_0 = module_0.shut_up()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'contextlib._GeneratorContextManager'
    assert f'{type(var_0.gen).__module__}.{type(var_0.gen).__qualname__}' == 'builtins.generator'
    assert var_0.args == ()
    assert var_0.kwds == {}
    assert f'{type(module_0.AnyStr).__module__}.{type(module_0.AnyStr).__qualname__}' == 'typing.TypeVar'
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.__enter__()

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    module_0.progress_open(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = None
    module_0.progress_open(var_0, verbose=var_0)

def test_case_4():
    var_0 = True
    with pytest.raises(ValueError):
        module_0.reverse_open(var_0, encoding=var_0, buffer_size=var_0)

def test_case_5():
    var_0 = False
    var_1 = module_0.reverse_open(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'flutes.io._ReverseReadlineFile'
    assert f'{type(var_1.fp).__module__}.{type(var_1.fp).__qualname__}' == '_io.BufferedReader'
    assert f'{type(var_1.gen).__module__}.{type(var_1.gen).__qualname__}' == 'builtins.generator'
    assert f'{type(module_0.AnyStr).__module__}.{type(module_0.AnyStr).__qualname__}' == 'typing.TypeVar'
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    assert module_0._ReverseReadlineFile.MAX_CHAR_BYTES == 4

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = None
    module_0._ProgressBufferedReader(var_0, bar_fn=var_0)

def test_case_7():
    var_0 = module_0.shut_up()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'contextlib._GeneratorContextManager'
    assert f'{type(var_0.gen).__module__}.{type(var_0.gen).__qualname__}' == 'builtins.generator'
    assert var_0.args == ()
    assert var_0.kwds == {}
    assert f'{type(module_0.AnyStr).__module__}.{type(module_0.AnyStr).__qualname__}' == 'typing.TypeVar'
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_1 = module_0._ReverseReadlineFile(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'flutes.io._ReverseReadlineFile'
    assert module_0._ReverseReadlineFile.MAX_CHAR_BYTES == 4

def test_case_8():
    var_0 = None
    var_1 = module_0._ReverseReadlineFile(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'flutes.io._ReverseReadlineFile'
    assert var_1.fp is None
    assert var_1.gen is None
    assert f'{type(module_0.AnyStr).__module__}.{type(module_0.AnyStr).__qualname__}' == 'typing.TypeVar'
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    assert module_0._ReverseReadlineFile.MAX_CHAR_BYTES == 4
    var_2 = var_1.__iter__()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.io._ReverseReadlineFile'
    assert var_2.fp is None
    assert var_2.gen is None

def test_case_9():
    var_0 = None
    var_1 = module_0._ReverseReadlineFile(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'flutes.io._ReverseReadlineFile'
    assert var_1.fp is None
    assert var_1.gen is None
    assert f'{type(module_0.AnyStr).__module__}.{type(module_0.AnyStr).__qualname__}' == 'typing.TypeVar'
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    assert module_0._ReverseReadlineFile.MAX_CHAR_BYTES == 4
    var_2 = var_1.__enter__()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'flutes.io._ReverseReadlineFile'
    assert var_2.fp is None
    assert var_2.gen is None

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = None
    var_1 = module_0._ReverseReadlineFile(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'flutes.io._ReverseReadlineFile'
    assert var_1.fp is None
    assert var_1.gen is None
    assert f'{type(module_0.AnyStr).__module__}.{type(module_0.AnyStr).__qualname__}' == 'typing.TypeVar'
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    assert module_0._ReverseReadlineFile.MAX_CHAR_BYTES == 4
    var_1.__exit__(var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = None
    module_0.progress_open(var_0, encoding=var_0, buffer_size=var_0, bar_fn=var_0)

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = None
    var_1 = module_0._ReverseReadlineFile(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'flutes.io._ReverseReadlineFile'
    assert var_1.fp is None
    assert var_1.gen is None
    assert f'{type(module_0.AnyStr).__module__}.{type(module_0.AnyStr).__qualname__}' == 'typing.TypeVar'
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    assert module_0._ReverseReadlineFile.MAX_CHAR_BYTES == 4
    var_1.readline()

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = None
    var_1 = module_0._ReverseReadlineFile(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'flutes.io._ReverseReadlineFile'
    assert var_1.fp is None
    assert var_1.gen is None
    assert f'{type(module_0.AnyStr).__module__}.{type(module_0.AnyStr).__qualname__}' == 'typing.TypeVar'
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    assert module_0._ReverseReadlineFile.MAX_CHAR_BYTES == 4
    var_1.__next__()

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = module_0.shut_up()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'contextlib._GeneratorContextManager'
    assert f'{type(var_0.gen).__module__}.{type(var_0.gen).__qualname__}' == 'builtins.generator'
    assert var_0.args == ()
    assert var_0.kwds == {}
    assert f'{type(module_0.AnyStr).__module__}.{type(module_0.AnyStr).__qualname__}' == 'typing.TypeVar'
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    module_0.progress_open(var_0, buffer_size=var_0, bar_fn=var_0)

def test_case_15():
    var_0 = False
    var_1 = module_0.shut_up(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'contextlib._GeneratorContextManager'
    assert f'{type(var_1.gen).__module__}.{type(var_1.gen).__qualname__}' == 'builtins.generator'
    assert var_1.args == (False,)
    assert var_1.kwds == {}
    assert f'{type(module_0.AnyStr).__module__}.{type(module_0.AnyStr).__qualname__}' == 'typing.TypeVar'
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_2 = var_1.__enter__()

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = b'\x90\xcd\xf7/\xe5\x1c\xb2/@Kw\xa9\xef\x13\xa3t\xec\xfeF\xc4'
    var_1 = module_0.shut_up(stdout=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'contextlib._GeneratorContextManager'
    assert f'{type(var_1.gen).__module__}.{type(var_1.gen).__qualname__}' == 'builtins.generator'
    assert var_1.args == ()
    assert var_1.kwds == {'stdout': b'\x90\xcd\xf7/\xe5\x1c\xb2/@Kw\xa9\xef\x13\xa3t\xec\xfeF\xc4'}
    assert f'{type(module_0.AnyStr).__module__}.{type(module_0.AnyStr).__qualname__}' == 'typing.TypeVar'
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_1.__enter__()

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = True
    var_1 = module_0.reverse_open(var_0, encoding=var_0, allow_empty_lines=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'flutes.io._ReverseReadlineFile'
    assert f'{type(var_1.fp).__module__}.{type(var_1.fp).__qualname__}' == '_io.BufferedReader'
    assert f'{type(var_1.gen).__module__}.{type(var_1.gen).__qualname__}' == 'builtins.generator'
    assert f'{type(module_0.AnyStr).__module__}.{type(module_0.AnyStr).__qualname__}' == 'typing.TypeVar'
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    assert module_0._ReverseReadlineFile.MAX_CHAR_BYTES == 4
    var_1.__next__()

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = None
    var_1 = module_0.shut_up()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'contextlib._GeneratorContextManager'
    assert f'{type(var_1.gen).__module__}.{type(var_1.gen).__qualname__}' == 'builtins.generator'
    assert var_1.args == ()
    assert var_1.kwds == {}
    assert f'{type(module_0.AnyStr).__module__}.{type(module_0.AnyStr).__qualname__}' == 'typing.TypeVar'
    assert module_0.TYPE_CHECKING is False
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'
    var_1.__enter__()
    var_2 = var_1.__exit__(var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = None
    module_0.progress_open(var_0, encoding=var_0, verbose=var_0, bar_fn=var_0)