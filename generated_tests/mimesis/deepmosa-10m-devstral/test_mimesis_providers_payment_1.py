# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import mimesis.providers.payment as module_0
import re as module_1
import mimesis.exceptions as module_2

def test_case_0():
    var_0 = module_0.Payment()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.payment.Payment'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert module_0.CREDIT_CARD_NETWORKS == ['Visa', 'MasterCard', 'Chase', 'American Express', 'Discover']
    var_1 = var_0.credit_card_owner()

def test_case_1():
    var_0 = module_0.Payment()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.payment.Payment'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert module_0.CREDIT_CARD_NETWORKS == ['Visa', 'MasterCard', 'Chase', 'American Express', 'Discover']
    var_1 = var_0.paypal()
    var_2 = var_0.paypal()

def test_case_2():
    var_0 = module_0.Payment()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.payment.Payment'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert module_0.CREDIT_CARD_NETWORKS == ['Visa', 'MasterCard', 'Chase', 'American Express', 'Discover']
    var_1 = var_0.credit_card_network()
    var_2 = var_0.paypal()

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = module_0.Payment()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.payment.Payment'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert module_0.CREDIT_CARD_NETWORKS == ['Visa', 'MasterCard', 'Chase', 'American Express', 'Discover']
    var_1 = var_0.credit_card_number()
    var_2 = None
    var_3 = var_0.cid()
    module_1.sub(var_2, var_2, var_2)

def test_case_4():
    var_0 = module_0.Payment()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.payment.Payment'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert module_0.CREDIT_CARD_NETWORKS == ['Visa', 'MasterCard', 'Chase', 'American Express', 'Discover']
    var_1 = var_0.credit_card_number()

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = module_0.Payment()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.payment.Payment'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert module_0.CREDIT_CARD_NETWORKS == ['Visa', 'MasterCard', 'Chase', 'American Express', 'Discover']
    var_1 = var_0.credit_card_network()
    var_2 = var_0.credit_card_number()
    var_3 = var_0.bitcoin_address()
    var_4 = None
    var_5 = var_0.credit_card_number()
    var_6 = var_0.credit_card_expiration_date()
    var_7 = var_0.cid()
    module_1.match(var_4, var_4)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = module_0.Payment()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.payment.Payment'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert module_0.CREDIT_CARD_NETWORKS == ['Visa', 'MasterCard', 'Chase', 'American Express', 'Discover']
    var_1 = var_0.credit_card_network()
    var_2 = var_0.ethereum_address()
    var_3 = None
    var_4 = var_0.credit_card_number(var_3)
    var_5 = var_0.credit_card_network()
    var_6 = module_1.RegexFlag.UNICODE
    var_7 = module_0.Payment()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'mimesis.providers.payment.Payment'
    assert f'{type(var_7.random).__module__}.{type(var_7.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_7.seed).__module__}.{type(var_7.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert module_1.ASCII == module_1.RegexFlag.ASCII
    assert module_1.A == module_1.RegexFlag.ASCII
    assert module_1.IGNORECASE == module_1.RegexFlag.IGNORECASE
    assert module_1.I == module_1.RegexFlag.IGNORECASE
    assert module_1.LOCALE == module_1.RegexFlag.LOCALE
    assert module_1.L == module_1.RegexFlag.LOCALE
    assert module_1.UNICODE == module_1.RegexFlag.UNICODE
    assert module_1.U == module_1.RegexFlag.UNICODE
    assert module_1.MULTILINE == module_1.RegexFlag.MULTILINE
    assert module_1.M == module_1.RegexFlag.MULTILINE
    assert module_1.DOTALL == module_1.RegexFlag.DOTALL
    assert module_1.S == module_1.RegexFlag.DOTALL
    assert module_1.VERBOSE == module_1.RegexFlag.VERBOSE
    assert module_1.X == module_1.RegexFlag.VERBOSE
    assert module_1.TEMPLATE == module_1.RegexFlag.TEMPLATE
    assert module_1.T == module_1.RegexFlag.TEMPLATE
    assert module_1.DEBUG == module_1.RegexFlag.DEBUG
    var_6.paypal()

def test_case_7():
    var_0 = module_0.Payment()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.payment.Payment'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert module_0.CREDIT_CARD_NETWORKS == ['Visa', 'MasterCard', 'Chase', 'American Express', 'Discover']
    var_1 = var_0.cvv()
    var_2 = var_0.credit_card_owner()

def test_case_8():
    var_0 = module_0.Payment()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.payment.Payment'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert module_0.CREDIT_CARD_NETWORKS == ['Visa', 'MasterCard', 'Chase', 'American Express', 'Discover']
    var_1 = var_0.credit_card_number()
    var_2 = var_0.credit_card_number()

def test_case_9():
    var_0 = module_0.Payment()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.payment.Payment'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert module_0.CREDIT_CARD_NETWORKS == ['Visa', 'MasterCard', 'Chase', 'American Express', 'Discover']
    var_1 = var_0.credit_card_network()
    assert var_1 == 'MasterCard'
    with pytest.raises(module_2.NonEnumerableError):
        var_0.credit_card_number(var_0)