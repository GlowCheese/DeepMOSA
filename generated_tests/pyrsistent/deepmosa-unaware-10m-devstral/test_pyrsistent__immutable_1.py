# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = "7T6':;\tt,~~;xE&F\\"
    var_1 = module_0.immutable(verbose=var_0)
    with pytest.raises(SyntaxError):
        module_0.immutable(var_0)

def test_case_1():
    var_0 = module_0.immutable()

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = module_0.immutable()
    var_1 = module_0.immutable(verbose=var_0)
    module_0.immutable(var_0)

def test_case_3():
    var_0 = None
    with pytest.raises(SyntaxError):
        module_0.immutable(name=var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    module_0.immutable(var_0)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = '/)\t]+?Nvw<MT\x0c'
    var_1 = module_0.immutable(verbose=var_0)
    module_0.immutable(var_0)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = module_0.immutable()
    var_1 = 'FF5BZa\n\\LRc;Y_'
    module_0.immutable(var_1, verbose=var_1)