# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import flutes.network as module_0

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = '\nede;d*5#dl=fq'
    module_0.download(var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    module_0.download(var_0)

def test_case_2():
    pass

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = '\n;;d*5#dl=^q'
    module_0.download(var_0, var_0, progress=var_0, bar_fn=var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = 'Q}J[v/tU9R}Ky'
    module_0.download(var_0, progress=var_0)

def test_case_5():
    var_0 = None
    var_1 = "9c7r?Wp'XK_5SL~tI/"
    var_2 = 'GhfpUR\x0b\x0b0vY'
    var_3 = '0/Ev`WpF\\vek'
    var_4 = {var_2: var_0, var_3: var_0}
    var_5 = module_0.download(var_1, filename=var_0, **var_4)
    assert var_5 == '/tmp/'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'

def test_case_6():
    var_0 = 'https://drive.google.com/file/d/my_secret_id/view?usp=sharing'
    var_1 = module_0._extract_google_drive_file_id(var_0)
    assert var_1 == 'my_secret_id'
    assert f'{type(module_0.PathType).__module__}.{type(module_0.PathType).__qualname__}' == 'typing.TypeVar'