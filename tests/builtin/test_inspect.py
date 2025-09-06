import inspect
import pytest
import asyncio
import sys
import importlib


def sample_function(x, *, y: int = 10) -> int:
    return x + y

async def sample_coroutine():
    await asyncio.sleep(1)
    return "done"

async def sample_asyncgen():
    yield 1
    yield 2

class SampleClass:
    class_attr = 42
    def sample_method(self):
        return "hello"

@pytest.fixture
def sample_module():
    return importlib.import_module('tests.fixtures.dummy')


def test_signature():
    sig = inspect.signature(sample_function)
    assert str(sig) == "(x, *, y: int = 10) -> int"
    
    params = sig.parameters
    assert "x" in params
    assert isinstance(params["x"], inspect.Parameter)

    assert params["x"].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert params["y"].kind == inspect.Parameter.KEYWORD_ONLY
    


def test_getsource(sample_module):
    with open("tests/fixtures/dummy.py", "r") as file:
        source1 = file.read()
    
    source2 = inspect.getsource(sample_module)

    # usually getsource add an endline to
    # the end of source file for some reason
    assert source1.rstrip('\n') == source2.rstrip('\n')

    source3 = inspect.getsource(sample_function)
    assert source3.rstrip('\n') == (
        "def sample_function(x, *, y: int = 10) -> int:\n"
        "    return x + y"
    )


def test_issomething(sample_module):
    assert inspect.ismodule(sample_module)
    assert inspect.isfunction(sample_function)
    assert inspect.isclass(SampleClass)
    assert inspect.ismethod(SampleClass().sample_method) 
    assert inspect.isfunction(SampleClass.sample_method)  # unbound method = function
    assert inspect.iscoroutinefunction(sample_coroutine)
    assert inspect.isasyncgenfunction(sample_asyncgen)


def test_getmembers():
    members = inspect.getmembers(SampleClass)

    all_method_names = set(SampleClass().__dir__())
    assert set(name for name, _ in members) == all_method_names

    for name, med in members:
        if name == "sample_method":
            assert inspect.isfunction(med)
            break
    else:
        assert False, f"cannot find sample_method of {SampleClass.__name__}"
