# Check out: https://github.com/GlowCheese/deepmosa
import mimesis.shortcuts as module_0


def test_case_0():
    var_0 = b'\x1c\x08/\xd0>\x0f\x84\x9d\x85\xd8y\xf3[\xf5\xf8%f\xbf\xf2'
    var_1 = module_0.luhn_checksum(var_0)
    assert var_1 == '0'