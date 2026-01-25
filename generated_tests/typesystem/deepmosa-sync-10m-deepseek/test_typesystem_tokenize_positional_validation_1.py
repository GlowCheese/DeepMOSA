# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.tokenize.positional_validation as module_0

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.validate_with_positions(token=var_0, validator=var_0)

def test_case_1():
    pass