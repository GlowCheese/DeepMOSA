# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import mimesis.shortcuts as module_0

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = 'cShwZ\x0btaA/(Qwo0{\t0'
    module_0.luhn_checksum(var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = ()
    var_1 = module_0.luhn_checksum(var_0)
    assert var_1 == '0'
    var_2 = 't94 i!@2Y<5v`@g1m='
    module_0.luhn_checksum(var_2)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = "'fMa)B=<K,o{85"
    module_0.luhn_checksum(var_0)