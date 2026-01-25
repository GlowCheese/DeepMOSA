# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import mimesis.providers.payment as module_0
import re as module_1
import mimesis.exceptions as module_2

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    var_0.paypal()

def test_case_1():
    var_0 = module_0.Payment()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.payment.Payment'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert module_0.CREDIT_CARD_NETWORKS == ['Visa', 'MasterCard', 'Chase', 'American Express', 'Discover']
    var_1 = var_0.credit_card_number()

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = module_0.Payment()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.payment.Payment'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert module_0.CREDIT_CARD_NETWORKS == ['Visa', 'MasterCard', 'Chase', 'American Express', 'Discover']
    var_1 = var_0.paypal()
    var_2 = None
    var_0.validate_enum(var_2, var_2)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = module_0.Payment()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.payment.Payment'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert module_0.CREDIT_CARD_NETWORKS == ['Visa', 'MasterCard', 'Chase', 'American Express', 'Discover']
    var_1 = None
    var_2 = var_0.credit_card_owner()
    var_3 = var_0.paypal()
    var_4 = var_0.credit_card_network()
    var_5 = var_0.paypal()
    module_1.finditer(var_1, var_1)

def test_case_4():
    var_0 = module_0.Payment()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.payment.Payment'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert module_0.CREDIT_CARD_NETWORKS == ['Visa', 'MasterCard', 'Chase', 'American Express', 'Discover']
    var_1 = var_0.credit_card_owner()

def test_case_5():
    var_0 = module_0.Payment()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.payment.Payment'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert module_0.CREDIT_CARD_NETWORKS == ['Visa', 'MasterCard', 'Chase', 'American Express', 'Discover']
    var_1 = None
    var_2 = var_0.credit_card_owner(var_1)
    var_3 = var_0.cid()

def test_case_6():
    var_0 = module_0.Payment()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.payment.Payment'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert module_0.CREDIT_CARD_NETWORKS == ['Visa', 'MasterCard', 'Chase', 'American Express', 'Discover']
    var_1 = var_0.credit_card_number()
    var_2 = var_0.bitcoin_address()

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = module_0.Payment()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.payment.Payment'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert module_0.CREDIT_CARD_NETWORKS == ['Visa', 'MasterCard', 'Chase', 'American Express', 'Discover']
    var_0.credit_card_owner(var_0)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = module_0.Payment()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.payment.Payment'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert module_0.CREDIT_CARD_NETWORKS == ['Visa', 'MasterCard', 'Chase', 'American Express', 'Discover']
    var_1 = None
    var_2 = var_0.ethereum_address()
    var_3 = None
    var_4 = var_0.credit_card_owner()
    var_5 = var_0.reseed(var_3)
    assert var_0.seed is None
    var_6 = var_0.ethereum_address()
    var_0.validate_enum(var_1, var_1)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = module_0.Payment()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.payment.Payment'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert module_0.CREDIT_CARD_NETWORKS == ['Visa', 'MasterCard', 'Chase', 'American Express', 'Discover']
    var_1 = None
    var_2 = var_0.credit_card_owner()
    var_3 = var_0.cvv()
    var_4 = var_0.bitcoin_address()
    var_0.validate_enum(var_1, var_2)

def test_case_10():
    var_0 = module_0.Payment()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.payment.Payment'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert module_0.CREDIT_CARD_NETWORKS == ['Visa', 'MasterCard', 'Chase', 'American Express', 'Discover']
    var_1 = var_0.credit_card_number()
    var_2 = var_0.credit_card_owner()
    var_3 = var_0.ethereum_address()
    with pytest.raises(module_2.NonEnumerableError):
        var_0.credit_card_number(var_3)