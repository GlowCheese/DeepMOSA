# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import mimesis.providers.payment as module_0
import enum as module_1
import re as module_2
import mimesis.exceptions as module_3

def test_case_0():
    var_0 = module_0.Payment()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.payment.Payment'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert module_0.CREDIT_CARD_NETWORKS == ['Visa', 'MasterCard', 'Chase', 'American Express', 'Discover']
    var_1 = var_0.credit_card_owner()

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = module_0.Payment()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.payment.Payment'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert module_0.CREDIT_CARD_NETWORKS == ['Visa', 'MasterCard', 'Chase', 'American Express', 'Discover']
    var_1 = var_0.paypal()
    var_2 = None
    var_0.validate_enum(var_2, var_2)

def test_case_2():
    var_0 = module_0.Payment()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.payment.Payment'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert module_0.CREDIT_CARD_NETWORKS == ['Visa', 'MasterCard', 'Chase', 'American Express', 'Discover']
    var_1 = var_0.credit_card_owner()
    var_2 = var_0.bitcoin_address()

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = module_0.Payment()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.payment.Payment'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert module_0.CREDIT_CARD_NETWORKS == ['Visa', 'MasterCard', 'Chase', 'American Express', 'Discover']
    var_1 = module_1._EnumDict()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'enum._EnumDict'
    assert len(var_1) == 0
    var_2 = None
    var_3 = var_0.credit_card_network()
    assert var_3 == 'American Express'
    var_4 = var_0.credit_card_number(var_2)
    var_2.ethereum_address()

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = module_0.Payment()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.payment.Payment'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert module_0.CREDIT_CARD_NETWORKS == ['Visa', 'MasterCard', 'Chase', 'American Express', 'Discover']
    var_1 = module_0.Payment()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.providers.payment.Payment'
    assert f'{type(var_1.random).__module__}.{type(var_1.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_1.seed).__module__}.{type(var_1.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_2 = var_1.cid()
    var_3 = var_0.ethereum_address()
    var_0.credit_card_owner(var_2)

def test_case_5():
    var_0 = module_0.Payment()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.payment.Payment'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert module_0.CREDIT_CARD_NETWORKS == ['Visa', 'MasterCard', 'Chase', 'American Express', 'Discover']
    var_1 = var_0.credit_card_owner()
    var_2 = var_0.cvv()

def test_case_6():
    var_0 = module_0.Payment()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.payment.Payment'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert module_0.CREDIT_CARD_NETWORKS == ['Visa', 'MasterCard', 'Chase', 'American Express', 'Discover']
    var_1 = var_0.credit_card_owner()

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = module_0.Payment()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.payment.Payment'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert module_0.CREDIT_CARD_NETWORKS == ['Visa', 'MasterCard', 'Chase', 'American Express', 'Discover']
    var_1 = var_0.credit_card_owner()
    module_2.escape(var_0)

def test_case_8():
    var_0 = module_0.Payment()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.payment.Payment'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert module_0.CREDIT_CARD_NETWORKS == ['Visa', 'MasterCard', 'Chase', 'American Express', 'Discover']
    var_1 = var_0.credit_card_owner()
    with pytest.raises(module_3.NonEnumerableError):
        var_0.credit_card_number(var_0)