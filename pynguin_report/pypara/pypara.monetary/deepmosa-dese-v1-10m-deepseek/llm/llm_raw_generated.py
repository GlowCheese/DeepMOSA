####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_qty_or_none_returns_qty_for_defined_price():
    from pypara.currencies import Currencies
    from pypara.monetary import Price, Date
    from decimal import Decimal
    price = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    result = price.qty_or_none()
    assert result == Decimal('1')

def test_qty_or_none_returns_none_for_undefined_price():
    from pypara.monetary import Price
    from decimal import Decimal
    price = Price.of(None, Decimal('1'), None)
    result = price.qty_or_none()
    assert result is None


# LLM-generated content at query #2
#--------------------------

def test___ge___with_same_currency_and_greater_qty():
    ccy = Currency("USD")
    m1 = SomeMoney(ccy, Decimal("100.0"), Date(2023, 1, 1))
    m2 = SomeMoney(ccy, Decimal("50.0"), Date(2023, 1, 1))
    result = m1 >= m2
    assert result is True

def test___ge___with_same_currency_and_equal_qty():
    ccy = Currency("USD")
    m1 = SomeMoney(ccy, Decimal("100.0"), Date(2023, 1, 1))
    m2 = SomeMoney(ccy, Decimal("100.0"), Date(2023, 1, 1))
    result = m1 >= m2
    assert result is True

def test___ge___with_same_currency_and_lesser_qty():
    ccy = Currency("USD")
    m1 = SomeMoney(ccy, Decimal("50.0"), Date(2023, 1, 1))
    m2 = SomeMoney(ccy, Decimal("100.0"), Date(2023, 1, 1))
    result = m1 >= m2
    assert result is False

def test___ge___with_different_currency_raises_error():
    ccy1 = Currency("USD")
    ccy2 = Currency("EUR")
    m1 = SomeMoney(ccy1, Decimal("100.0"), Date(2023, 1, 1))
    m2 = SomeMoney(ccy2, Decimal("100.0"), Date(2023, 1, 1))
    try:
        _ = m1 >= m2
        assert False
    except IncompatibleCurrencyError as e:
        assert e.ccy1 == ccy1
        assert e.ccy2 == ccy2
        assert e.operation == ">= comparison"

def test___ge___with_non_somemoney_returns_true():
    ccy = Currency("USD")
    m1 = SomeMoney(ccy, Decimal("100.0"), Date(2023, 1, 1))
    other = object()
    result = m1 >= other
    assert result is True


# LLM-generated content at query #3
#--------------------------

def test_gte_returns_true_when_other_is_not_somemoney():
    ccy = Currency("USD")
    qty = Decimal("100.00")
    dov = Date(2023, 1, 1)
    money = SomeMoney(ccy, qty, dov)
    other = NoMoney
    result = money.gte(other)
    assert result == True

def test_gte_raises_incompatiblecurrencyerror_when_currencies_differ():
    ccy1 = Currency("USD")
    ccy2 = Currency("EUR")
    qty1 = Decimal("100.00")
    qty2 = Decimal("90.00")
    dov = Date(2023, 1, 1)
    money1 = SomeMoney(ccy1, qty1, dov)
    money2 = SomeMoney(ccy2, qty2, dov)
    try:
        money1.gte(money2)
        assert False
    except IncompatibleCurrencyError as e:
        assert e.ccy1 == ccy1
        assert e.ccy2 == ccy2
        assert e.operation == ">= comparison"

def test_gte_returns_true_when_qty_greater():
    ccy = Currency("USD")
    qty1 = Decimal("100.00")
    qty2 = Decimal("90.00")
    dov = Date(2023, 1, 1)
    money1 = SomeMoney(ccy, qty1, dov)
    money2 = SomeMoney(ccy, qty2, dov)
    result = money1.gte(money2)
    assert result == True

def test_gte_returns_true_when_qty_equal():
    ccy = Currency("USD")
    qty = Decimal("100.00")
    dov = Date(2023, 1, 1)
    money1 = SomeMoney(ccy, qty, dov)
    money2 = SomeMoney(ccy, qty, dov)
    result = money1.gte(money2)
    assert result == True

def test_gte_returns_false_when_qty_less():
    ccy = Currency("USD")
    qty1 = Decimal("90.00")
    qty2 = Decimal("100.00")
    dov = Date(2023, 1, 1)
    money1 = SomeMoney(ccy, qty1, dov)
    money2 = SomeMoney(ccy, qty2, dov)
    result = money1.gte(money2)
    assert result == False


# LLM-generated content at query #4
#--------------------------

def test_is_equal_with_same_defined_price():
    from pypara.currencies import Currencies
    from datetime import date
    price1 = Price.of(Currencies["USD"], Decimal('100'), date(2023, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('100'), date(2023, 1, 1))
    result = price1.is_equal(price2)
    assert result == True

def test_is_equal_with_different_quantity():
    from pypara.currencies import Currencies
    from datetime import date
    price1 = Price.of(Currencies["USD"], Decimal('100'), date(2023, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('200'), date(2023, 1, 1))
    result = price1.is_equal(price2)
    assert result == False

def test_is_equal_with_different_currency():
    from pypara.currencies import Currencies
    from datetime import date
    price1 = Price.of(Currencies["USD"], Decimal('100'), date(2023, 1, 1))
    price2 = Price.of(Currencies["EUR"], Decimal('100'), date(2023, 1, 1))
    result = price1.is_equal(price2)
    assert result == False

def test_is_equal_with_different_date():
    from pypara.currencies import Currencies
    from datetime import date
    price1 = Price.of(Currencies["USD"], Decimal('100'), date(2023, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('100'), date(2023, 1, 2))
    result = price1.is_equal(price2)
    assert result == False

def test_is_equal_with_undefined_price():
    from pypara.currencies import Currencies
    from datetime import date
    price1 = Price.na()
    price2 = Price.na()
    result = price1.is_equal(price2)
    assert result == True

def test_is_equal_defined_vs_undefined():
    from pypara.currencies import Currencies
    from datetime import date
    price1 = Price.of(Currencies["USD"], Decimal('100'), date(2023, 1, 1))
    price2 = Price.na()
    result = price1.is_equal(price2)
    assert result == False

def test_is_equal_with_non_price_object():
    from pypara.currencies import Currencies
    from datetime import date
    price = Price.of(Currencies["USD"], Decimal('100'), date(2023, 1, 1))
    other = "not a price"
    result = price.is_equal(other)
    assert result == False

def test_is_equal_with_none():
    from pypara.currencies import Currencies
    from datetime import date
    price = Price.of(Currencies["USD"], Decimal('100'), date(2023, 1, 1))
    result = price.is_equal(None)
    assert result == False


# LLM-generated content at query #5
#--------------------------

def test_or_else_defined_money_returns_itself():
    from pypara.currencies import Currencies
    from pypara.money import Money, Date
    from decimal import Decimal
    somemoney = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    fallback = Money.of(Currencies["EUR"], Decimal('2'), Date(2019, 1, 2))
    result = somemoney.or_else(lambda: fallback)
    assert result is somemoney

def test_or_else_undefined_money_returns_fallback():
    from pypara.currencies import Currencies
    from pypara.money import Money, Date
    from decimal import Decimal
    nonemoney = Money.na()
    fallback = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    result = nonemoney.or_else(lambda: fallback)
    assert result is fallback

def test_or_else_undefined_money_with_none_components_returns_fallback():
    from pypara.currencies import Currencies
    from pypara.money import Money, Date
    from decimal import Decimal
    nonemoney = Money.of(None, Decimal('1'), None)
    fallback = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    result = nonemoney.or_else(lambda: fallback)
    assert result is fallback

def test_or_else_fallback_lambda_called_only_for_undefined():
    from pypara.currencies import Currencies
    from pypara.money import Money, Date
    from decimal import Decimal
    call_count = 0
    def counter():
        nonlocal call_count
        call_count += 1
        return Money.na()
    somemoney = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    somemoney.or_else(counter)
    assert call_count == 0
    nonemoney = Money.na()
    nonemoney.or_else(counter)
    assert call_count == 1


# LLM-generated content at query #6
#--------------------------

def test_floordiv_with_defined_money_and_non_zero_scalar():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from decimal import Decimal
    from datetime import date
    money = Money.of(Currencies["USD"], Decimal('10'), date(2023, 1, 1))
    result = money.floor_divide(Decimal('3'))
    assert result.qty == Decimal('3')
    assert result.ccy == Currencies["USD"]
    assert result.dov == date(2023, 1, 1)

def test_floordiv_with_defined_money_and_zero_scalar():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from decimal import Decimal
    from datetime import date
    money = Money.of(Currencies["USD"], Decimal('10'), date(2023, 1, 1))
    result = money.floor_divide(Decimal('0'))
    assert result.undefined

def test_floordiv_with_undefined_money():
    from pypara.money import Money
    from decimal import Decimal
    money = Money.na()
    result = money.floor_divide(Decimal('5'))
    assert result is money

def test_floordiv_with_defined_money_and_integer_scalar():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from decimal import Decimal
    from datetime import date
    money = Money.of(Currencies["EUR"], Decimal('7'), date(2023, 1, 1))
    result = money.floor_divide(2)
    assert result.qty == Decimal('3')
    assert result.ccy == Currencies["EUR"]
    assert result.dov == date(2023, 1, 1)

def test_floordiv_with_defined_money_and_float_scalar():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from decimal import Decimal
    from datetime import date
    money = Money.of(Currencies["JPY"], Decimal('100'), date(2023, 1, 1))
    result = money.floor_divide(30.0)
    assert result.qty == Decimal('3')
    assert result.ccy == Currencies["JPY"]
    assert result.dov == date(2023, 1, 1)

def test_floordiv_with_defined_money_negative_scalar():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from decimal import Decimal
    from datetime import date
    money = Money.of(Currencies["USD"], Decimal('10'), date(2023, 1, 1))
    result = money.floor_divide(Decimal('-3'))
    assert result.qty == Decimal('-4')
    assert result.ccy == Currencies["USD"]
    assert result.dov == date(2023, 1, 1)

def test_floordiv_with_defined_money_negative_quantity():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from decimal import Decimal
    from datetime import date
    money = Money.of(Currencies["USD"], Decimal('-10'), date(2023, 1, 1))
    result = money.floor_divide(Decimal('3'))
    assert result.qty == Decimal('-4')
    assert result.ccy == Currencies["USD"]
    assert result.dov == date(2023, 1, 1)


# LLM-generated content at query #7
#--------------------------

def test_convert_same_currency_no_conversion_needed():
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    money = SomeMoney(USD, Decimal("100.00"), Date(2023, 1, 1))
    result = money.convert(USD, Date(2023, 1, 1), strict=False)
    assert result == SomeMoney(USD, Decimal("100.00"), Date(2023, 1, 1))

def test_convert_with_valid_fx_rate():
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    EUR = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    money = SomeMoney(USD, Decimal("100.00"), Date(2023, 1, 1))
    rate = FXRate(USD, EUR, Decimal("0.85"), Date(2023, 1, 1))
    FXRateService.default = MockFXRateService(rate)
    result = money.convert(EUR, Date(2023, 1, 1), strict=False)
    assert result == SomeMoney(EUR, Decimal("85.00"), Date(2023, 1, 1))

def test_convert_with_strict_true_and_rate_found():
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    EUR = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    money = SomeMoney(USD, Decimal("100.00"), Date(2023, 1, 1))
    rate = FXRate(USD, EUR, Decimal("0.85"), Date(2023, 1, 1))
    FXRateService.default = MockFXRateService(rate)
    result = money.convert(EUR, Date(2023, 1, 1), strict=True)
    assert result == SomeMoney(EUR, Decimal("85.00"), Date(2023, 1, 1))

def test_convert_with_strict_true_and_rate_not_found():
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    EUR = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    money = SomeMoney(USD, Decimal("100.00"), Date(2023, 1, 1))
    FXRateService.default = MockFXRateService(None)
    try:
        money.convert(EUR, Date(2023, 1, 1), strict=True)
        assert False
    except FXRateLookupError:
        assert True

def test_convert_with_strict_false_and_rate_not_found():
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    EUR = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    money = SomeMoney(USD, Decimal("100.00"), Date(2023, 1, 1))
    FXRateService.default = MockFXRateService(None)
    result = money.convert(EUR, Date(2023, 1, 1), strict=False)
    assert result == NoMoney

def test_convert_with_asof_date_different_from_dov():
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    EUR = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    money = SomeMoney(USD, Decimal("100.00"), Date(2023, 1, 1))
    rate = FXRate(USD, EUR, Decimal("0.90"), Date(2023, 6, 1))
    FXRateService.default = MockFXRateService(rate)
    result = money.convert(EUR, Date(2023, 6, 1), strict=False)
    assert result == SomeMoney(EUR, Decimal("90.00"), Date(2023, 6, 1))

def test_convert_with_asof_date_none_uses_dov():
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    EUR = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    money = SomeMoney(USD, Decimal("100.00"), Date(2023, 1, 1))
    rate = FXRate(USD, EUR, Decimal("0.85"), Date(2023, 1, 1))
    FXRateService.default = MockFXRateService(rate)
    result = money.convert(EUR, None, strict=False)
    assert result == SomeMoney(EUR, Decimal("85.00"), Date(2023, 1, 1))

def test_convert_with_fx_rate_service_not_set():
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    EUR = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    money = SomeMoney(USD, Decimal("100.00"), Date(2023, 1, 1))
    FXRateService.default = None
    try:
        money.convert(EUR, Date(2023, 1, 1), strict=False)
        assert False
    except ProgrammingError:
        assert True

def test_convert_quantizes_result_to_target_currency():
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    JPY = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    money = SomeMoney(USD, Decimal("100.555"), Date(2023, 1, 1))
    rate = FXRate(USD, JPY, Decimal("110.123"), Date(2023, 1, 1))
    FXRateService.default = MockFXRateService(rate)
    result = money.convert(JPY, Date(2023, 1, 1), strict=False)
    assert result == SomeMoney(JPY, Decimal("11079"), Date(2023, 1, 1))

def test_convert_with_crypto_currency_and_negative_decimals():
    BTC = Currency.of("BTC", "Bitcoin", -1, CurrencyType.CRYPTO)
    ETH = Currency.of("ETH", "Ethereum", -1, CurrencyType.CRYPTO)
    money = SomeMoney(BTC, Decimal("1.123456789012345"), Date(2023, 1, 1))
    rate = FXRate(BTC, ETH, Decimal("15.123456789012345"), Date(2023, 1, 1))
    FXRateService.default = MockFXRateService(rate)
    result = money.convert(ETH, Date(2023, 1, 1), strict=False)
    expected_qty = (Decimal("1.123456789012345") * Decimal("15.123456789012345")).quantize(ETH.quantizer)
    assert result == SomeMoney(ETH, expected_qty, Date(2023, 1, 1))


# LLM-generated content at query #8
#--------------------------

def test_mul_with_defined_money_and_positive_scalar():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from decimal import Decimal
    from datetime import date
    m = Money.of(Currencies["USD"], Decimal('10'), date(2023, 1, 1))
    result = m * Decimal('2')
    assert result.qty == Decimal('20.00')
    assert result.ccy.code == 'USD'
    assert result.dov == date(2023, 1, 1)

def test_mul_with_defined_money_and_negative_scalar():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from decimal import Decimal
    from datetime import date
    m = Money.of(Currencies["EUR"], Decimal('5'), date(2023, 1, 1))
    result = m * Decimal('-3')
    assert result.qty == Decimal('-15.00')
    assert result.ccy.code == 'EUR'
    assert result.dov == date(2023, 1, 1)

def test_mul_with_defined_money_and_zero_scalar():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from decimal import Decimal
    from datetime import date
    m = Money.of(Currencies["JPY"], Decimal('7'), date(2023, 1, 1))
    result = m * Decimal('0')
    assert result.qty == Decimal('0')
    assert result.ccy.code == 'JPY'
    assert result.dov == date(2023, 1, 1)

def test_mul_with_undefined_money():
    from pypara.money import Money
    from decimal import Decimal
    m = Money.na()
    result = m * Decimal('5')
    assert result.undefined

def test_mul_with_defined_money_and_integer_scalar():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from decimal import Decimal
    from datetime import date
    m = Money.of(Currencies["USD"], Decimal('3'), date(2023, 1, 1))
    result = m * 4
    assert result.qty == Decimal('12.00')
    assert result.ccy.code == 'USD'
    assert result.dov == date(2023, 1, 1)

def test_mul_with_defined_money_and_float_scalar():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from decimal import Decimal
    from datetime import date
    m = Money.of(Currencies["EUR"], Decimal('2'), date(2023, 1, 1))
    result = m * 2.5
    assert result.qty == Decimal('5.00')
    assert result.ccy.code == 'EUR'
    assert result.dov == date(2023, 1, 1)

def test_mul_commutative_property():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from decimal import Decimal
    from datetime import date
    m = Money.of(Currencies["USD"], Decimal('3'), date(2023, 1, 1))
    scalar = Decimal('4')
    result1 = m * scalar
    result2 = scalar * m
    assert result1.qty == result2.qty
    assert result1.ccy.code == result2.ccy.code
    assert result1.dov == result2.dov

def test_mul_with_defined_money_and_decimal_scalar():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from decimal import Decimal
    from datetime import date
    m = Money.of(Currencies["GBP"], Decimal('8'), date(2023, 1, 1))
    result = m * Decimal('0.5')
    assert result.qty == Decimal('4.00')
    assert result.ccy.code == 'GBP'
    assert result.dov == date(2023, 1, 1)

def test_mul_preserves_currency_and_date():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from decimal import Decimal
    from datetime import date
    m = Money.of(Currencies["CAD"], Decimal('100'), date(2023, 5, 15))
    result = m * Decimal('2')
    assert result.ccy.code == 'CAD'
    assert result.dov == date(2023, 5, 15)


# LLM-generated content at query #9
#--------------------------

def test_with_qty_returns_new_money_with_given_quantity_when_defined():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from decimal import Decimal
    from datetime import date
    original = Money.of(Currencies["USD"], Decimal('10'), date(2023, 1, 1))
    new_qty = Decimal('20')
    result = original.with_qty(new_qty)
    assert result.qty == new_qty
    assert result.ccy == original.ccy
    assert result.dov == original.dov
    assert result is not original

def test_with_qty_returns_itself_when_undefined():
    from pypara.money import Money
    from decimal import Decimal
    undefined_money = Money.na()
    result = undefined_money.with_qty(Decimal('20'))
    assert result is undefined_money

def test_with_qty_quantity_is_quantized_to_currency():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from decimal import Decimal
    from datetime import date
    ccy = Currencies["JPY"]
    original = Money.of(ccy, Decimal('100'), date(2023, 1, 1))
    new_qty = Decimal('123.456')
    result = original.with_qty(new_qty)
    expected_qty = ccy.quantize(new_qty)
    assert result.qty == expected_qty

def test_with_qty_handles_zero_quantity():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from decimal import Decimal
    from datetime import date
    original = Money.of(Currencies["EUR"], Decimal('5'), date(2023, 1, 1))
    result = original.with_qty(Decimal('0'))
    assert result.qty == Decimal('0')

def test_with_qty_handles_negative_quantity():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from decimal import Decimal
    from datetime import date
    original = Money.of(Currencies["GBP"], Decimal('10'), date(2023, 1, 1))
    new_qty = Decimal('-5')
    result = original.with_qty(new_qty)
    assert result.qty == new_qty


# LLM-generated content at query #10
#--------------------------

def test_round_positive_quantity_round_down():
    ccy = Currency(code="USD", decimals=2)
    qty = Decimal("123.456")
    dov = Date(2023, 1, 1)
    money = SomeMoney(ccy, qty, dov)
    result = money.round(1)
    expected = SomeMoney(ccy, Decimal("123.4"), dov)
    assert result == expected

def test_round_positive_quantity_round_up():
    ccy = Currency(code="USD", decimals=2)
    qty = Decimal("123.456")
    dov = Date(2023, 1, 1)
    money = SomeMoney(ccy, qty, dov)
    result = money.round(2)
    expected = SomeMoney(ccy, Decimal("123.46"), dov)
    assert result == expected

def test_round_negative_quantity_round_down():
    ccy = Currency(code="USD", decimals=2)
    qty = Decimal("-123.456")
    dov = Date(2023, 1, 1)
    money = SomeMoney(ccy, qty, dov)
    result = money.round(1)
    expected = SomeMoney(ccy, Decimal("-123.4"), dov)
    assert result == expected

def test_round_negative_quantity_round_up():
    ccy = Currency(code="USD", decimals=2)
    qty = Decimal("-123.456")
    dov = Date(2023, 1, 1)
    money = SomeMoney(ccy, qty, dov)
    result = money.round(2)
    expected = SomeMoney(ccy, Decimal("-123.46"), dov)
    assert result == expected

def test_round_zero_ndigits():
    ccy = Currency(code="USD", decimals=2)
    qty = Decimal("123.456")
    dov = Date(2023, 1, 1)
    money = SomeMoney(ccy, qty, dov)
    result = money.round(0)
    expected = SomeMoney(ccy, Decimal("123"), dov)
    assert result == expected

def test_round_ndigits_greater_than_decimals():
    ccy = Currency(code="USD", decimals=2)
    qty = Decimal("123.456")
    dov = Date(2023, 1, 1)
    money = SomeMoney(ccy, qty, dov)
    result = money.round(5)
    expected = SomeMoney(ccy, Decimal("123.46"), dov)
    assert result == expected

def test_round_exact_half_up():
    ccy = Currency(code="USD", decimals=2)
    qty = Decimal("123.455")
    dov = Date(2023, 1, 1)
    money = SomeMoney(ccy, qty, dov)
    result = money.round(2)
    expected = SomeMoney(ccy, Decimal("123.46"), dov)
    assert result == expected

def test_round_exact_half_down():
    ccy = Currency(code="JPY", decimals=0)
    qty = Decimal("123.5")
    dov = Date(2023, 1, 1)
    money = SomeMoney(ccy, qty, dov)
    result = money.round(0)
    expected = SomeMoney(ccy, Decimal("124"), dov)
    assert result == expected

def test_round_negative_ndigits():
    ccy = Currency(code="USD", decimals=2)
    qty = Decimal("123.456")
    dov = Date(2023, 1, 1)
    money = SomeMoney(ccy, qty, dov)
    result = money.round(-1)
    expected = SomeMoney(ccy, Decimal("120"), dov)
    assert result == expected

def test_round_negative_ndigits_greater_than_magnitude():
    ccy = Currency(code="USD", decimals=2)
    qty = Decimal("123.456")
    dov = Date(2023, 1, 1)
    money = SomeMoney(ccy, qty, dov)
    result = money.round(-3)
    expected = SomeMoney(ccy, Decimal("0"), dov)
    assert result == expected


# LLM-generated content at query #11
#--------------------------

def test_as_boolean_returns_false_for_undefined_money():
    money = Money.na()
    result = money.as_boolean()
    assert result == False

def test_as_boolean_returns_false_for_zero_quantity():
    from pypara.currencies import Currencies
    from datetime import date
    money = Money.of(Currencies["USD"], Decimal('0'), date(2019, 1, 1))
    result = money.as_boolean()
    assert result == False

def test_as_boolean_returns_true_for_positive_quantity():
    from pypara.currencies import Currencies
    from datetime import date
    money = Money.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    result = money.as_boolean()
    assert result == True

def test_as_boolean_returns_true_for_negative_quantity():
    from pypara.currencies import Currencies
    from datetime import date
    money = Money.of(Currencies["USD"], Decimal('-1'), date(2019, 1, 1))
    result = money.as_boolean()
    assert result == True


# LLM-generated content at query #12
#--------------------------

def test___neg___with_defined_money():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from decimal import Decimal
    from datetime import date
    m = Money.of(Currencies["USD"], Decimal("10.5"), date(2023, 1, 1))
    result = -m
    assert result.qty == Decimal("-10.5")
    assert result.ccy == Currencies["USD"]
    assert result.dov == date(2023, 1, 1)

def test___neg___with_undefined_money():
    from pypara.money import Money
    m = Money.na()
    result = -m
    assert result.undefined

def test___neg___with_zero_quantity():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from decimal import Decimal
    from datetime import date
    m = Money.of(Currencies["EUR"], Decimal("0"), date(2023, 1, 1))
    result = -m
    assert result.qty == Decimal("0")
    assert result.ccy == Currencies["EUR"]
    assert result.dov == date(2023, 1, 1)


# LLM-generated content at query #13
#--------------------------

def test_subtract_defined_money_same_currency():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from datetime import date
    from decimal import Decimal
    m1 = Money.of(Currencies["USD"], Decimal('10'), date(2023, 1, 1))
    m2 = Money.of(Currencies["USD"], Decimal('3'), date(2023, 1, 1))
    result = m1.subtract(m2)
    assert result.defined
    assert result.qty == Decimal('7')
    assert result.ccy.code == 'USD'

def test_subtract_defined_money_different_currency_raises():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from datetime import date
    from decimal import Decimal
    from pypara.money import IncompatibleCurrencyError
    m1 = Money.of(Currencies["USD"], Decimal('10'), date(2023, 1, 1))
    m2 = Money.of(Currencies["EUR"], Decimal('3'), date(2023, 1, 1))
    try:
        m1.subtract(m2)
        assert False
    except IncompatibleCurrencyError:
        assert True

def test_subtract_first_operand_undefined():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from datetime import date
    from decimal import Decimal
    m1 = Money.na()
    m2 = Money.of(Currencies["USD"], Decimal('3'), date(2023, 1, 1))
    result = m1.subtract(m2)
    assert result is m2

def test_subtract_second_operand_undefined():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from datetime import date
    from decimal import Decimal
    m1 = Money.of(Currencies["USD"], Decimal('10'), date(2023, 1, 1))
    m2 = Money.na()
    result = m1.subtract(m2)
    assert result is m1

def test_subtract_both_operands_undefined():
    from pypara.money import Money
    m1 = Money.na()
    m2 = Money.na()
    result = m1.subtract(m2)
    assert result is m1

def test_subtract_date_carried_forward():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from datetime import date
    from decimal import Decimal
    m1 = Money.of(Currencies["USD"], Decimal('10'), date(2023, 1, 1))
    m2 = Money.of(Currencies["USD"], Decimal('3'), date(2023, 1, 2))
    result = m1.subtract(m2)
    assert result.dov == date(2023, 1, 2)


# LLM-generated content at query #14
#--------------------------

def test_as_boolean_returns_false_for_undefined_price():
    price = Price.na()
    result = price.as_boolean()
    assert result == False

def test_as_boolean_returns_false_for_defined_price_with_zero_quantity():
    from pypara.currencies import Currencies
    from datetime import date
    price = Price.of(Currencies["USD"], Decimal('0'), date(2019, 1, 1))
    result = price.as_boolean()
    assert result == False

def test_as_boolean_returns_true_for_defined_price_with_positive_quantity():
    from pypara.currencies import Currencies
    from datetime import date
    price = Price.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    result = price.as_boolean()
    assert result == True

def test_as_boolean_returns_true_for_defined_price_with_negative_quantity():
    from pypara.currencies import Currencies
    from datetime import date
    price = Price.of(Currencies["USD"], Decimal('-1'), date(2019, 1, 1))
    result = price.as_boolean()
    assert result == True


# LLM-generated content at query #15
#--------------------------

def test_divide_defined_price_by_positive_number():
    price = Price.of(Currencies["USD"], Decimal('10'), Date(2023, 1, 1))
    result = price.divide(Decimal('2'))
    assert result.qty == Decimal('5')
    assert result.ccy == Currencies["USD"]
    assert result.dov == Date(2023, 1, 1)

def test_divide_defined_price_by_one():
    price = Price.of(Currencies["EUR"], Decimal('7.5'), Date(2023, 2, 1))
    result = price.divide(Decimal('1'))
    assert result.qty == Decimal('7.5')
    assert result.ccy == Currencies["EUR"]
    assert result.dov == Date(2023, 2, 1)

def test_divide_defined_price_by_negative_number():
    price = Price.of(Currencies["JPY"], Decimal('100'), Date(2023, 3, 1))
    result = price.divide(Decimal('-2'))
    assert result.qty == Decimal('-50')
    assert result.ccy == Currencies["JPY"]
    assert result.dov == Date(2023, 3, 1)

def test_divide_defined_price_by_zero_yields_undefined():
    price = Price.of(Currencies["GBP"], Decimal('15'), Date(2023, 4, 1))
    result = price.divide(Decimal('0'))
    assert result.undefined

def test_divide_undefined_price_returns_itself():
    price = Price.na()
    result = price.divide(Decimal('5'))
    assert result is price
    assert result.undefined

def test_divide_defined_price_by_decimal_fraction():
    price = Price.of(Currencies["CAD"], Decimal('3'), Date(2023, 5, 1))
    result = price.divide(Decimal('0.5'))
    assert result.qty == Decimal('6')
    assert result.ccy == Currencies["CAD"]
    assert result.dov == Date(2023, 5, 1)

def test_divide_defined_price_by_integer():
    price = Price.of(Currencies["AUD"], Decimal('9'), Date(2023, 6, 1))
    result = price.divide(3)
    assert result.qty == Decimal('3')
    assert result.ccy == Currencies["AUD"]
    assert result.dov == Date(2023, 6, 1)

def test_divide_defined_price_by_float():
    price = Price.of(Currencies["CHF"], Decimal('8'), Date(2023, 7, 1))
    result = price.divide(2.0)
    assert result.qty == Decimal('4')
    assert result.ccy == Currencies["CHF"]
    assert result.dov == Date(2023, 7, 1)


# LLM-generated content at query #16
#--------------------------

def test___eq__():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from datetime import date
    from decimal import Decimal
    usd = Currencies["USD"]
    eur = Currencies["EUR"]
    date1 = date(2019, 1, 1)
    date2 = date(2019, 1, 2)
    m1 = Money.of(usd, Decimal('100'), date1)
    m2 = Money.of(usd, Decimal('100'), date1)
    m3 = Money.of(usd, Decimal('200'), date1)
    m4 = Money.of(usd, Decimal('100'), date2)
    m5 = Money.of(eur, Decimal('100'), date1)
    m6 = Money.na()
    m7 = Money.na()
    assert m1.is_equal(m2)
    assert not m1.is_equal(m3)
    assert not m1.is_equal(m4)
    assert not m1.is_equal(m5)
    assert not m1.is_equal(m6)
    assert m6.is_equal(m7)
    assert not m1.is_equal(None)
    assert not m1.is_equal("string")
    assert not m1.is_equal(123)


# LLM-generated content at query #17
#--------------------------

def test_qty_or_none_returns_qty_for_defined_money():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from datetime import date
    from decimal import Decimal
    somemoney = Money.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    result = somemoney.qty_or_none()
    assert result == Decimal('1.00')

def test_qty_or_none_returns_none_for_undefined_money():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from decimal import Decimal
    nonemoney = Money.of(None, Decimal('1'), None)
    result = nonemoney.qty_or_none()
    assert result is None

def test_qty_or_none_returns_none_for_na_money():
    from pypara.money import Money
    nonemoney = Money.na()
    result = nonemoney.qty_or_none()
    assert result is None


# LLM-generated content at query #18
#--------------------------

def test_divide_defined_money_by_positive_number():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from decimal import Decimal
    from datetime import date
    money = Money.of(Currencies["USD"], Decimal('10'), date(2023, 1, 1))
    result = money.divide(Decimal('2'))
    assert result.defined
    assert result.qty == Decimal('5.00')
    assert result.ccy.code == 'USD'
    assert result.dov == date(2023, 1, 1)

def test_divide_defined_money_by_one():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from decimal import Decimal
    from datetime import date
    money = Money.of(Currencies["EUR"], Decimal('7.50'), date(2023, 1, 1))
    result = money.divide(Decimal('1'))
    assert result.defined
    assert result.qty == Decimal('7.50')
    assert result.ccy.code == 'EUR'
    assert result.dov == date(2023, 1, 1)

def test_divide_defined_money_by_negative_number():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from decimal import Decimal
    from datetime import date
    money = Money.of(Currencies["USD"], Decimal('10'), date(2023, 1, 1))
    result = money.divide(Decimal('-2'))
    assert result.defined
    assert result.qty == Decimal('-5.00')
    assert result.ccy.code == 'USD'
    assert result.dov == date(2023, 1, 1)

def test_divide_defined_money_by_zero_yields_undefined():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from decimal import Decimal
    from datetime import date
    money = Money.of(Currencies["USD"], Decimal('10'), date(2023, 1, 1))
    result = money.divide(Decimal('0'))
    assert result.undefined

def test_divide_undefined_money_returns_itself():
    from pypara.money import Money
    from decimal import Decimal
    undefined_money = Money.na()
    result = undefined_money.divide(Decimal('5'))
    assert result is undefined_money

def test_divide_defined_money_by_decimal_fraction():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from decimal import Decimal
    from datetime import date
    money = Money.of(Currencies["USD"], Decimal('1'), date(2023, 1, 1))
    result = money.divide(Decimal('0.5'))
    assert result.defined
    assert result.qty == Decimal('2.00')
    assert result.ccy.code == 'USD'
    assert result.dov == date(2023, 1, 1)

def test_divide_defined_money_by_integer():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from decimal import Decimal
    from datetime import date
    money = Money.of(Currencies["USD"], Decimal('9'), date(2023, 1, 1))
    result = money.divide(3)
    assert result.defined
    assert result.qty == Decimal('3.00')
    assert result.ccy.code == 'USD'
    assert result.dov == date(2023, 1, 1)

def test_divide_defined_money_by_float():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from decimal import Decimal
    from datetime import date
    money = Money.of(Currencies["USD"], Decimal('10'), date(2023, 1, 1))
    result = money.divide(2.5)
    assert result.defined
    assert result.qty == Decimal('4.00')
    assert result.ccy.code == 'USD'
    assert result.dov == date(2023, 1, 1)


# LLM-generated content at query #19
#--------------------------

def test_abs_defined_positive():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from decimal import Decimal
    from datetime import date
    money = Money.of(Currencies["USD"], Decimal('100.50'), date(2023, 1, 1))
    result = money.abs()
    assert result.qty == Decimal('100.50')
    assert result.ccy == Currencies["USD"]
    assert result.dov == date(2023, 1, 1)

def test_abs_defined_negative():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from decimal import Decimal
    from datetime import date
    money = Money.of(Currencies["EUR"], Decimal('-200.75'), date(2023, 2, 1))
    result = money.abs()
    assert result.qty == Decimal('200.75')
    assert result.ccy == Currencies["EUR"]
    assert result.dov == date(2023, 2, 1)

def test_abs_defined_zero():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from decimal import Decimal
    from datetime import date
    money = Money.of(Currencies["JPY"], Decimal('0'), date(2023, 3, 1))
    result = money.abs()
    assert result.qty == Decimal('0')
    assert result.ccy == Currencies["JPY"]
    assert result.dov == date(2023, 3, 1)

def test_abs_undefined():
    from pypara.money import Money
    money = Money.na()
    result = money.abs()
    assert result is money
    assert result.undefined


# LLM-generated content at query #20
#--------------------------

def test_convert_same_currency():
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    price = SomePrice(USD, Decimal("100.00"), Date(2023, 1, 1))
    result = price.convert(USD, Date(2023, 1, 1))
    assert result.ccy == USD
    assert result.qty == Decimal("100.00")
    assert result.dov == Date(2023, 1, 1)

def test_convert_different_currency_with_rate():
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    EUR = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    FXRateService.default = lambda: None
    FXRateService.default.query = lambda ccy_from, ccy_to, asof, strict: SomeFXRate(USD, EUR, Decimal("0.85"), Date(2023, 1, 1))
    price = SomePrice(USD, Decimal("100.00"), Date(2023, 1, 1))
    result = price.convert(EUR, Date(2023, 1, 1))
    assert result.ccy == EUR
    assert result.qty == Decimal("85.00")
    assert result.dov == Date(2023, 1, 1)

def test_convert_different_currency_no_rate_strict_false():
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    EUR = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    FXRateService.default = lambda: None
    FXRateService.default.query = lambda ccy_from, ccy_to, asof, strict: None
    price = SomePrice(USD, Decimal("100.00"), Date(2023, 1, 1))
    result = price.convert(EUR, Date(2023, 1, 1), strict=False)
    assert result == NoPrice

def test_convert_different_currency_no_rate_strict_true():
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    EUR = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    FXRateService.default = lambda: None
    FXRateService.default.query = lambda ccy_from, ccy_to, asof, strict: None
    price = SomePrice(USD, Decimal("100.00"), Date(2023, 1, 1))
    try:
        price.convert(EUR, Date(2023, 1, 1), strict=True)
        assert False
    except FXRateLookupError:
        assert True

def test_convert_with_asof_date():
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    EUR = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    FXRateService.default = lambda: None
    FXRateService.default.query = lambda ccy_from, ccy_to, asof, strict: SomeFXRate(USD, EUR, Decimal("0.90"), asof)
    price = SomePrice(USD, Decimal("100.00"), Date(2023, 1, 1))
    result = price.convert(EUR, Date(2023, 6, 1))
    assert result.ccy == EUR
    assert result.qty == Decimal("90.00")
    assert result.dov == Date(2023, 6, 1)

def test_convert_without_asof_date_uses_dov():
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    EUR = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    FXRateService.default = lambda: None
    FXRateService.default.query = lambda ccy_from, ccy_to, asof, strict: SomeFXRate(USD, EUR, Decimal("0.80"), asof)
    price = SomePrice(USD, Decimal("100.00"), Date(2023, 5, 1))
    result = price.convert(EUR)
    assert result.ccy == EUR
    assert result.qty == Decimal("80.00")
    assert result.dov == Date(2023, 5, 1)

def test_convert_fx_rate_service_not_set():
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    EUR = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    FXRateService.default = None
    price = SomePrice(USD, Decimal("100.00"), Date(2023, 1, 1))
    try:
        price.convert(EUR)
        assert False
    except ProgrammingError:
        assert True

def test_convert_fx_rate_service_raises_attribute_error():
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    EUR = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    FXRateService.default = lambda: None
    FXRateService.default.query = lambda ccy_from, ccy_to, asof, strict: (_ for _ in ()).throw(AttributeError("test"))
    price = SomePrice(USD, Decimal("100.00"), Date(2023, 1, 1))
    try:
        price.convert(EUR)
        assert False
    except AttributeError as exc:
        assert str(exc) == "test"


# LLM-generated content at query #21
#--------------------------

def test___neg___with_defined_price():
    from pypara.currencies import Currencies
    from pypara.monetary import Price
    from decimal import Decimal
    from datetime import date
    price = Price.of(Currencies["USD"], Decimal("10.5"), date(2023, 1, 1))
    neg_price = -price
    assert neg_price.qty == Decimal("-10.5")
    assert neg_price.ccy == Currencies["USD"]
    assert neg_price.dov == date(2023, 1, 1)

def test___neg___with_undefined_price():
    from pypara.monetary import Price
    undefined_price = Price.na()
    neg_price = -undefined_price
    assert neg_price is undefined_price

def test___neg___with_zero_quantity():
    from pypara.currencies import Currencies
    from pypara.monetary import Price
    from decimal import Decimal
    from datetime import date
    price = Price.of(Currencies["EUR"], Decimal("0"), date(2023, 1, 1))
    neg_price = -price
    assert neg_price.qty == Decimal("0")
    assert neg_price.ccy == Currencies["EUR"]
    assert neg_price.dov == date(2023, 1, 1)

def test___neg___with_negative_quantity():
    from pypara.currencies import Currencies
    from pypara.monetary import Price
    from decimal import Decimal
    from datetime import date
    price = Price.of(Currencies["JPY"], Decimal("-15.75"), date(2023, 1, 1))
    neg_price = -price
    assert neg_price.qty == Decimal("15.75")
    assert neg_price.ccy == Currencies["JPY"]
    assert neg_price.dov == date(2023, 1, 1)


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_neg_positive_quantity():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currencies
    money = SomeMoney(Currencies["USD"], Decimal("100.50"), date(2023, 1, 1))
    result = money.__neg__()
    assert result.ccy == Currencies["USD"]
    assert result.qty == Decimal("-100.50")
    assert result.dov == date(2023, 1, 1)

def test_neg_negative_quantity():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currencies
    money = SomeMoney(Currencies["EUR"], Decimal("-200.75"), date(2023, 1, 1))
    result = money.__neg__()
    assert result.ccy == Currencies["EUR"]
    assert result.qty == Decimal("200.75")
    assert result.dov == date(2023, 1, 1)

def test_neg_zero_quantity():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currencies
    money = SomeMoney(Currencies["GBP"], Decimal("0.00"), date(2023, 1, 1))
    result = money.__neg__()
    assert result.ccy == Currencies["GBP"]
    assert result.qty == Decimal("0.00")
    assert result.dov == date(2023, 1, 1)

def test_neg_currency_preserved():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currencies
    money = SomeMoney(Currencies["JPY"], Decimal("500"), date(2023, 1, 1))
    result = money.__neg__()
    assert result.ccy == Currencies["JPY"]
    assert result.qty == Decimal("-500")
    assert result.dov == date(2023, 1, 1)

def test_neg_date_preserved():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currencies
    money = SomeMoney(Currencies["CAD"], Decimal("123.45"), date(2022, 12, 31))
    result = money.__neg__()
    assert result.ccy == Currencies["CAD"]
    assert result.qty == Decimal("-123.45")
    assert result.dov == date(2022, 12, 31)


# LLM-generated content at query #2
#--------------------------

def test_lte_undefined_price_less_than_or_equal_to_defined_price():
    from pypara.currencies import Currencies
    from pypara.monetary import Price
    undefined_price = Price.na()
    defined_price = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    result = undefined_price.lte(defined_price)
    assert result == True

def test_lte_undefined_price_less_than_or_equal_to_undefined_price():
    from pypara.currencies import Currencies
    from pypara.monetary import Price
    undefined_price1 = Price.na()
    undefined_price2 = Price.na()
    result = undefined_price1.lte(undefined_price2)
    assert result == True

def test_lte_defined_price_less_than_or_equal_to_undefined_price():
    from pypara.currencies import Currencies
    from pypara.monetary import Price
    defined_price = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    undefined_price = Price.na()
    result = defined_price.lte(undefined_price)
    assert result == False

def test_lte_defined_price_less_than_or_equal_to_same_currency_lower_qty():
    from pypara.currencies import Currencies
    from pypara.monetary import Price
    price1 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    result = price1.lte(price2)
    assert result == True

def test_lte_defined_price_less_than_or_equal_to_same_currency_equal_qty():
    from pypara.currencies import Currencies
    from pypara.monetary import Price
    price1 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    result = price1.lte(price2)
    assert result == True

def test_lte_defined_price_less_than_or_equal_to_same_currency_higher_qty():
    from pypara.currencies import Currencies
    from pypara.monetary import Price
    price1 = Price.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    result = price1.lte(price2)
    assert result == False

def test_lte_raises_incompatible_currency_error():
    from pypara.currencies import Currencies
    from pypara.monetary import Price
    from pypara.monetary import IncompatibleCurrencyError
    price1 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["EUR"], Decimal('1'), Date(2019, 1, 1))
    try:
        price1.lte(price2)
        assert False
    except IncompatibleCurrencyError:
        assert True


# LLM-generated content at query #3
#--------------------------

def test_dov_or_none_with_defined_money():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from datetime import date
    somemoney = Money.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    result = somemoney.dov_or_none()
    assert result == date(2019, 1, 1)

def test_dov_or_none_with_undefined_money():
    from pypara.currencies import Currencies
    from pypara.money import Money
    nonemoney = Money.of(None, None, date(2019, 1, 1))
    result = nonemoney.dov_or_none()
    assert result is None

def test_dov_or_none_with_undefined_money_all_none():
    from pypara.money import Money
    nonemoney = Money.na()
    result = nonemoney.dov_or_none()
    assert result is None

def test_dov_or_none_with_defined_money_other_date():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from datetime import date
    somemoney = Money.of(Currencies["EUR"], Decimal('2.5'), date(2020, 12, 31))
    result = somemoney.dov_or_none()
    assert result == date(2020, 12, 31)


# LLM-generated content at query #4
#--------------------------

def test_ccy_or_returns_ccy_when_money_is_defined():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from datetime import date
    somemoney = Money.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    result = somemoney.ccy_or(Currencies["EUR"])
    assert result.code == "USD"

def test_ccy_or_returns_default_when_money_is_undefined():
    from pypara.currencies import Currencies
    from pypara.money import Money
    nonemoney = Money.of(None, Decimal('1'), None)
    result = nonemoney.ccy_or(Currencies["EUR"])
    assert result.code == "EUR"

def test_ccy_or_returns_default_when_ccy_is_none():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from datetime import date
    nonemoney = Money.of(None, Decimal('1'), date(2019, 1, 1))
    result = nonemoney.ccy_or(Currencies["EUR"])
    assert result.code == "EUR"

def test_ccy_or_returns_default_when_qty_is_none():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from datetime import date
    nonemoney = Money.of(Currencies["USD"], None, date(2019, 1, 1))
    result = nonemoney.ccy_or(Currencies["EUR"])
    assert result.code == "EUR"

def test_ccy_or_returns_default_when_dov_is_none():
    from pypara.currencies import Currencies
    from pypara.money import Money
    nonemoney = Money.of(Currencies["USD"], Decimal('1'), None)
    result = nonemoney.ccy_or(Currencies["EUR"])
    assert result.code == "EUR"

def test_ccy_or_returns_ccy_for_defined_money_with_different_default():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from datetime import date
    somemoney = Money.of(Currencies["GBP"], Decimal('100'), date(2020, 5, 15))
    result = somemoney.ccy_or(Currencies["JPY"])
    assert result.code == "GBP"

def test_ccy_or_returns_default_for_na_money_instance():
    from pypara.currencies import Currencies
    from pypara.money import Money
    na_money = Money.na()
    result = na_money.ccy_or(Currencies["CAD"])
    assert result.code == "CAD"


# LLM-generated content at query #5
#--------------------------

def test_as_integer_defined_price():
    from pypara.currencies import Currencies
    from pypara.monetary import Price
    from decimal import Decimal
    from datetime import date
    price = Price.of(Currencies["USD"], Decimal('10'), date(2023, 1, 1))
    result = price.as_integer()
    assert result == 10

def test_as_integer_undefined_price():
    from pypara.monetary import Price
    price = Price.na()
    try:
        price.as_integer()
        assert False
    except Exception as e:
        assert "MonetaryOperationException" in str(type(e).__name__)

def test_as_integer_zero_quantity():
    from pypara.currencies import Currencies
    from pypara.monetary import Price
    from decimal import Decimal
    from datetime import date
    price = Price.of(Currencies["USD"], Decimal('0'), date(2023, 1, 1))
    result = price.as_integer()
    assert result == 0

def test_as_integer_negative_quantity():
    from pypara.currencies import Currencies
    from pypara.monetary import Price
    from decimal import Decimal
    from datetime import date
    price = Price.of(Currencies["USD"], Decimal('-5'), date(2023, 1, 1))
    result = price.as_integer()
    assert result == -5

def test_as_integer_large_quantity():
    from pypara.currencies import Currencies
    from pypara.monetary import Price
    from decimal import Decimal
    from datetime import date
    price = Price.of(Currencies["USD"], Decimal('123456789'), date(2023, 1, 1))
    result = price.as_integer()
    assert result == 123456789

def test_as_integer_fractional_quantity_rounds_down():
    from pypara.currencies import Currencies
    from pypara.monetary import Price
    from decimal import Decimal
    from datetime import date
    price = Price.of(Currencies["USD"], Decimal('3.14'), date(2023, 1, 1))
    result = price.as_integer()
    assert result == 3

def test_as_integer_fractional_quantity_rounds_up():
    from pypara.currencies import Currencies
    from pypara.monetary import Price
    from decimal import Decimal
    from datetime import date
    price = Price.of(Currencies["USD"], Decimal('2.99'), date(2023, 1, 1))
    result = price.as_integer()
    assert result == 2


# LLM-generated content at query #6
#--------------------------

def test_constructor_with_valid_arguments():
    ccy = Currency("USD")
    qty = Decimal("100.50")
    dov = Date(2023, 1, 1)
    price = SomePrice(ccy, qty, dov)
    assert price.ccy == ccy
    assert price.qty == qty
    assert price.dov == dov

def test_constructor_with_zero_quantity():
    ccy = Currency("EUR")
    qty = Decimal("0")
    dov = Date(2023, 1, 1)
    price = SomePrice(ccy, qty, dov)
    assert price.qty == qty

def test_constructor_with_negative_quantity():
    ccy = Currency("GBP")
    qty = Decimal("-50.75")
    dov = Date(2023, 1, 1)
    price = SomePrice(ccy, qty, dov)
    assert price.qty == qty

def test_constructor_with_different_currency():
    ccy = Currency("JPY")
    qty = Decimal("1000")
    dov = Date(2023, 1, 1)
    price = SomePrice(ccy, qty, dov)
    assert price.ccy == ccy

def test_constructor_with_different_date():
    ccy = Currency("USD")
    qty = Decimal("200.00")
    dov = Date(2024, 12, 31)
    price = SomePrice(ccy, qty, dov)
    assert price.dov == dov

def test_constructor_creates_namedtuple_subclass():
    ccy = Currency("USD")
    qty = Decimal("100.50")
    dov = Date(2023, 1, 1)
    price = SomePrice(ccy, qty, dov)
    assert isinstance(price, tuple)
    assert hasattr(price, '_fields')
    assert price._fields == ('ccy', 'qty', 'dov')

def test_constructor_slots_are_empty():
    ccy = Currency("USD")
    qty = Decimal("100.50")
    dov = Date(2023, 1, 1)
    price = SomePrice(ccy, qty, dov)
    assert price.__slots__ == ()


# LLM-generated content at query #7
#--------------------------

def test_floordiv_with_valid_numeric():
    price = SomePrice(Currency("USD"), Decimal("10.5"), Date(2023, 1, 1))
    result = price.__floordiv__(2)
    expected = SomePrice(Currency("USD"), Decimal("5"), Date(2023, 1, 1))
    assert result == expected

def test_floordiv_with_zero_division():
    price = SomePrice(Currency("USD"), Decimal("10.5"), Date(2023, 1, 1))
    result = price.__floordiv__(0)
    assert result == NoPrice

def test_floordiv_with_negative_numeric():
    price = SomePrice(Currency("USD"), Decimal("10.5"), Date(2023, 1, 1))
    result = price.__floordiv__(-2)
    expected = SomePrice(Currency("USD"), Decimal("-6"), Date(2023, 1, 1))
    assert result == expected

def test_floordiv_with_decimal_numeric():
    price = SomePrice(Currency("USD"), Decimal("10.5"), Date(2023, 1, 1))
    result = price.__floordiv__(Decimal("2.5"))
    expected = SomePrice(Currency("USD"), Decimal("4"), Date(2023, 1, 1))
    assert result == expected

def test_floordiv_with_invalid_operation():
    price = SomePrice(Currency("USD"), Decimal("10.5"), Date(2023, 1, 1))
    result = price.__floordiv__(Decimal("NaN"))
    assert result == NoPrice


# LLM-generated content at query #8
#--------------------------

def test_floordiv_defined_price_with_positive_divisor():
    from pypara.currencies import Currencies
    from pypara.monetary import Price
    from decimal import Decimal
    from datetime import date
    price = Price.of(Currencies["USD"], Decimal('10'), date(2023, 1, 1))
    result = price.floor_divide(Decimal('3'))
    assert result.qty == Decimal('3')
    assert result.ccy.code == 'USD'
    assert result.dov == date(2023, 1, 1)

def test_floordiv_defined_price_with_negative_divisor():
    from pypara.currencies import Currencies
    from pypara.monetary import Price
    from decimal import Decimal
    from datetime import date
    price = Price.of(Currencies["EUR"], Decimal('10'), date(2023, 1, 1))
    result = price.floor_divide(Decimal('-3'))
    assert result.qty == Decimal('-4')
    assert result.ccy.code == 'EUR'
    assert result.dov == date(2023, 1, 1)

def test_floordiv_defined_price_with_zero_divisor():
    from pypara.currencies import Currencies
    from pypara.monetary import Price
    from decimal import Decimal
    from datetime import date
    price = Price.of(Currencies["USD"], Decimal('10'), date(2023, 1, 1))
    result = price.floor_divide(Decimal('0'))
    assert result.undefined

def test_floordiv_undefined_price():
    from pypara.monetary import Price
    from decimal import Decimal
    undefined_price = Price.na()
    result = undefined_price.floor_divide(Decimal('5'))
    assert result is undefined_price

def test_floordiv_defined_price_with_integer_divisor():
    from pypara.currencies import Currencies
    from pypara.monetary import Price
    from decimal import Decimal
    from datetime import date
    price = Price.of(Currencies["JPY"], Decimal('7'), date(2023, 1, 1))
    result = price.floor_divide(2)
    assert result.qty == Decimal('3')
    assert result.ccy.code == 'JPY'
    assert result.dov == date(2023, 1, 1)

def test_floordiv_defined_price_with_float_divisor():
    from pypara.currencies import Currencies
    from pypara.monetary import Price
    from decimal import Decimal
    from datetime import date
    price = Price.of(Currencies["GBP"], Decimal('10'), date(2023, 1, 1))
    result = price.floor_divide(3.0)
    assert result.qty == Decimal('3')
    assert result.ccy.code == 'GBP'
    assert result.dov == date(2023, 1, 1)


# LLM-generated content at query #9
#--------------------------

def test_with_dov_returns_new_money_with_given_dov_when_defined():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from datetime import date
    original = Money.of(Currencies["USD"], Decimal("100.50"), date(2023, 1, 1))
    new_dov = date(2023, 12, 31)
    result = original.with_dov(new_dov)
    assert result.defined
    assert result.dov_or_none() == new_dov
    assert result.ccy_or_none() == Currencies["USD"]
    assert result.qty_or_none() == Decimal("100.50")

def test_with_dov_returns_itself_when_undefined():
    from pypara.money import Money
    undefined_money = Money.na()
    new_dov = date(2023, 12, 31)
    result = undefined_money.with_dov(new_dov)
    assert result is undefined_money
    assert result.undefined

def test_with_dov_preserves_other_attributes():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from datetime import date
    original = Money.of(Currencies["EUR"], Decimal("200.75"), date(2022, 5, 15))
    new_dov = date(2022, 6, 20)
    result = original.with_dov(new_dov)
    assert result.ccy_or_none() == original.ccy_or_none()
    assert result.qty_or_none() == original.qty_or_none()
    assert result.dov_or_none() == new_dov

def test_with_dov_with_same_dov_returns_equal_money():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from datetime import date
    original_dov = date(2023, 7, 4)
    original = Money.of(Currencies["GBP"], Decimal("150.25"), original_dov)
    result = original.with_dov(original_dov)
    assert result.is_equal(original)

def test_with_dov_on_undefined_money_does_not_change_undefined_state():
    from pypara.money import Money
    from datetime import date
    undefined_money = Money.na()
    new_dov = date(2024, 1, 1)
    result = undefined_money.with_dov(new_dov)
    assert result.undefined
    assert not result.defined


# LLM-generated content at query #10
#--------------------------

def test_convert_same_currency():
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    money = SomeMoney(USD, Decimal("100.00"), Date(2023, 1, 1))
    result = money.convert(USD, Date(2023, 1, 1))
    assert result.ccy == USD
    assert result.qty == Decimal("100.00")
    assert result.dov == Date(2023, 1, 1)

def test_convert_different_currency_with_rate():
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    EUR = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    money = SomeMoney(USD, Decimal("100.00"), Date(2023, 1, 1))
    FXRateService.default = lambda: None
    FXRateService.default.query = lambda ccy1, ccy2, asof, strict: FXRate(USD, EUR, Decimal("0.85"), Date(2023, 1, 1))
    result = money.convert(EUR, Date(2023, 1, 1))
    assert result.ccy == EUR
    assert result.qty == Decimal("85.00")
    assert result.dov == Date(2023, 1, 1)

def test_convert_strict_mode_no_rate():
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    EUR = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    money = SomeMoney(USD, Decimal("100.00"), Date(2023, 1, 1))
    FXRateService.default = lambda: None
    FXRateService.default.query = lambda ccy1, ccy2, asof, strict: None
    try:
        money.convert(EUR, Date(2023, 1, 1), strict=True)
        assert False
    except FXRateLookupError:
        assert True

def test_convert_non_strict_mode_no_rate():
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    EUR = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    money = SomeMoney(USD, Decimal("100.00"), Date(2023, 1, 1))
    FXRateService.default = lambda: None
    FXRateService.default.query = lambda ccy1, ccy2, asof, strict: None
    result = money.convert(EUR, Date(2023, 1, 1), strict=False)
    assert result == NoMoney

def test_convert_uses_asof_date():
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    EUR = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    money = SomeMoney(USD, Decimal("100.00"), Date(2023, 1, 1))
    FXRateService.default = lambda: None
    FXRateService.default.query = lambda ccy1, ccy2, asof, strict: FXRate(USD, EUR, Decimal("0.85"), asof)
    result = money.convert(EUR, Date(2023, 6, 1))
    assert result.dov == Date(2023, 6, 1)

def test_convert_default_asof_is_dov():
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    EUR = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    money = SomeMoney(USD, Decimal("100.00"), Date(2023, 1, 1))
    FXRateService.default = lambda: None
    FXRateService.default.query = lambda ccy1, ccy2, asof, strict: FXRate(USD, EUR, Decimal("0.85"), asof)
    result = money.convert(EUR)
    assert result.dov == Date(2023, 1, 1)

def test_convert_no_fx_service_raises_error():
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    EUR = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    money = SomeMoney(USD, Decimal("100.00"), Date(2023, 1, 1))
    FXRateService.default = None
    try:
        money.convert(EUR)
        assert False
    except ProgrammingError:
        assert True

def test_convert_quantizes_result():
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    JPY = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    money = SomeMoney(USD, Decimal("100.555"), Date(2023, 1, 1))
    FXRateService.default = lambda: None
    FXRateService.default.query = lambda ccy1, ccy2, asof, strict: FXRate(USD, JPY, Decimal("110.123"), Date(2023, 1, 1))
    result = money.convert(JPY, Date(2023, 1, 1))
    assert result.ccy == JPY
    assert result.qty == Decimal("11076")
    assert result.dov == Date(2023, 1, 1)


# LLM-generated content at query #11
#--------------------------

def test_truediv_with_defined_money_and_non_zero_divisor():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from decimal import Decimal
    from datetime import date
    money = Money.of(Currencies["USD"], Decimal('10'), date(2023, 1, 1))
    result = money / Decimal('2')
    assert result.defined
    assert result.qty == Decimal('5.00')
    assert result.ccy == Currencies["USD"]
    assert result.dov == date(2023, 1, 1)

def test_truediv_with_defined_money_and_zero_divisor():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from decimal import Decimal
    from datetime import date
    money = Money.of(Currencies["USD"], Decimal('10'), date(2023, 1, 1))
    result = money / Decimal('0')
    assert result.undefined

def test_truediv_with_undefined_money():
    from pypara.money import Money
    from decimal import Decimal
    money = Money.na()
    result = money / Decimal('5')
    assert result is money

def test_truediv_with_defined_money_and_integer_divisor():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from decimal import Decimal
    from datetime import date
    money = Money.of(Currencies["USD"], Decimal('9'), date(2023, 1, 1))
    result = money / 3
    assert result.defined
    assert result.qty == Decimal('3.00')
    assert result.ccy == Currencies["USD"]
    assert result.dov == date(2023, 1, 1)

def test_truediv_with_defined_money_and_float_divisor():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from decimal import Decimal
    from datetime import date
    money = Money.of(Currencies["USD"], Decimal('5'), date(2023, 1, 1))
    result = money / 2.5
    assert result.defined
    assert result.qty == Decimal('2.00')
    assert result.ccy == Currencies["USD"]
    assert result.dov == date(2023, 1, 1)

def test_truediv_ensures_quantization():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from decimal import Decimal
    from datetime import date
    money = Money.of(Currencies["USD"], Decimal('1'), date(2023, 1, 1))
    result = money / Decimal('3')
    assert result.defined
    assert result.qty == Decimal('0.33')
    assert result.ccy == Currencies["USD"]
    assert result.dov == date(2023, 1, 1)


# LLM-generated content at query #12
#--------------------------

def test___ge___returns_true_when_self_qty_greater_than_other_qty():
    ccy = Currency("USD")
    qty1 = Decimal("10.5")
    qty2 = Decimal("5.5")
    dov = Date(2023, 1, 1)
    price1 = SomePrice(ccy, qty1, dov)
    price2 = SomePrice(ccy, qty2, dov)
    result = price1.__ge__(price2)
    assert result is True

def test___ge___returns_true_when_self_qty_equal_to_other_qty():
    ccy = Currency("USD")
    qty = Decimal("10.5")
    dov = Date(2023, 1, 1)
    price1 = SomePrice(ccy, qty, dov)
    price2 = SomePrice(ccy, qty, dov)
    result = price1.__ge__(price2)
    assert result is True

def test___ge___returns_false_when_self_qty_less_than_other_qty():
    ccy = Currency("USD")
    qty1 = Decimal("5.5")
    qty2 = Decimal("10.5")
    dov = Date(2023, 1, 1)
    price1 = SomePrice(ccy, qty1, dov)
    price2 = SomePrice(ccy, qty2, dov)
    result = price1.__ge__(price2)
    assert result is False

def test___ge___raises_IncompatibleCurrencyError_when_currencies_differ():
    ccy1 = Currency("USD")
    ccy2 = Currency("EUR")
    qty = Decimal("10.5")
    dov = Date(2023, 1, 1)
    price1 = SomePrice(ccy1, qty, dov)
    price2 = SomePrice(ccy2, qty, dov)
    try:
        price1.__ge__(price2)
        assert False
    except IncompatibleCurrencyError as e:
        assert e.ccy1 == ccy1
        assert e.ccy2 == ccy2
        assert e.operation == ">= comparison"

def test___ge___returns_True_when_other_is_not_SomePrice():
    ccy = Currency("USD")
    qty = Decimal("10.5")
    dov = Date(2023, 1, 1)
    price = SomePrice(ccy, qty, dov)
    other = NoPrice
    result = price.__ge__(other)
    assert result is True


# LLM-generated content at query #13
#--------------------------

def test___ge___with_defined_money_same_currency_greater():
    m1 = Money.of(Currencies["USD"], Decimal('10'), Date(2020, 1, 1))
    m2 = Money.of(Currencies["USD"], Decimal('5'), Date(2020, 1, 1))
    result = m1 >= m2
    assert result is True

def test___ge___with_defined_money_same_currency_equal():
    m1 = Money.of(Currencies["USD"], Decimal('5'), Date(2020, 1, 1))
    m2 = Money.of(Currencies["USD"], Decimal('5'), Date(2020, 1, 1))
    result = m1 >= m2
    assert result is True

def test___ge___with_defined_money_same_currency_less():
    m1 = Money.of(Currencies["USD"], Decimal('3'), Date(2020, 1, 1))
    m2 = Money.of(Currencies["USD"], Decimal('5'), Date(2020, 1, 1))
    result = m1 >= m2
    assert result is False

def test___ge___with_defined_money_different_currency():
    m1 = Money.of(Currencies["USD"], Decimal('10'), Date(2020, 1, 1))
    m2 = Money.of(Currencies["EUR"], Decimal('5'), Date(2020, 1, 1))
    try:
        m1 >= m2
        assert False
    except IncompatibleCurrencyError:
        assert True

def test___ge___with_undefined_money_and_defined_money():
    m1 = Money.na()
    m2 = Money.of(Currencies["USD"], Decimal('5'), Date(2020, 1, 1))
    result = m1 >= m2
    assert result is False

def test___ge___with_defined_money_and_undefined_money():
    m1 = Money.of(Currencies["USD"], Decimal('5'), Date(2020, 1, 1))
    m2 = Money.na()
    result = m1 >= m2
    assert result is True

def test___ge___with_both_undefined_money():
    m1 = Money.na()
    m2 = Money.na()
    result = m1 >= m2
    assert result is True


# LLM-generated content at query #14
#--------------------------

def test___add___same_currency_and_dates():
    ccy = Currency("USD")
    qty1 = Decimal("10.5")
    qty2 = Decimal("5.2")
    dov = Date(2023, 1, 1)
    price1 = SomePrice(ccy, qty1, dov)
    price2 = SomePrice(ccy, qty2, dov)
    result = price1.__add__(price2)
    expected = SomePrice(ccy, Decimal("15.7"), dov)
    assert result == expected

def test___add___same_currency_different_dates_later_dov1():
    ccy = Currency("USD")
    qty1 = Decimal("10.5")
    qty2 = Decimal("5.2")
    dov1 = Date(2023, 1, 2)
    dov2 = Date(2023, 1, 1)
    price1 = SomePrice(ccy, qty1, dov1)
    price2 = SomePrice(ccy, qty2, dov2)
    result = price1.__add__(price2)
    expected = SomePrice(ccy, Decimal("15.7"), dov1)
    assert result == expected

def test___add___same_currency_different_dates_later_dov2():
    ccy = Currency("USD")
    qty1 = Decimal("10.5")
    qty2 = Decimal("5.2")
    dov1 = Date(2023, 1, 1)
    dov2 = Date(2023, 1, 2)
    price1 = SomePrice(ccy, qty1, dov1)
    price2 = SomePrice(ccy, qty2, dov2)
    result = price1.__add__(price2)
    expected = SomePrice(ccy, Decimal("15.7"), dov2)
    assert result == expected

def test___add___different_currency_raises_incompatible_currency_error():
    ccy1 = Currency("USD")
    ccy2 = Currency("EUR")
    qty1 = Decimal("10.5")
    qty2 = Decimal("5.2")
    dov = Date(2023, 1, 1)
    price1 = SomePrice(ccy1, qty1, dov)
    price2 = SomePrice(ccy2, qty2, dov)
    try:
        price1.__add__(price2)
        assert False
    except IncompatibleCurrencyError as e:
        assert e.ccy1 == ccy1
        assert e.ccy2 == ccy2
        assert e.operation == "addition"

def test___add___with_undefined_price_returns_self():
    ccy = Currency("USD")
    qty = Decimal("10.5")
    dov = Date(2023, 1, 1)
    price = SomePrice(ccy, qty, dov)
    undefined_price = NoPrice
    result = price.__add__(undefined_price)
    assert result == price

def test___add___commutative_property():
    ccy = Currency("USD")
    qty1 = Decimal("10.5")
    qty2 = Decimal("5.2")
    dov = Date(2023, 1, 1)
    price1 = SomePrice(ccy, qty1, dov)
    price2 = SomePrice(ccy, qty2, dov)
    result1 = price1.__add__(price2)
    result2 = price2.__add__(price1)
    assert result1 == result2

def test___add___zero_quantity():
    ccy = Currency("USD")
    qty1 = Decimal("10.5")
    qty2 = Decimal("0")
    dov = Date(2023, 1, 1)
    price1 = SomePrice(ccy, qty1, dov)
    price2 = SomePrice(ccy, qty2, dov)
    result = price1.__add__(price2)
    expected = SomePrice(ccy, Decimal("10.5"), dov)
    assert result == expected

def test___add___negative_quantity():
    ccy = Currency("USD")
    qty1 = Decimal("10.5")
    qty2 = Decimal("-5.2")
    dov = Date(2023, 1, 1)
    price1 = SomePrice(ccy, qty1, dov)
    price2 = SomePrice(ccy, qty2, dov)
    result = price1.__add__(price2)
    expected = SomePrice(ccy, Decimal("5.3"), dov)
    assert result == expected

def test___add___large_quantities():
    ccy = Currency("USD")
    qty1 = Decimal("999999.99")
    qty2 = Decimal("0.01")
    dov = Date(2023, 1, 1)
    price1 = SomePrice(ccy, qty1, dov)
    price2 = SomePrice(ccy, qty2, dov)
    result = price1.__add__(price2)
    expected = SomePrice(ccy, Decimal("1000000.00"), dov)
    assert result == expected

def test___add___using_operator_overload():
    ccy = Currency("USD")
    qty1 = Decimal("10.5")
    qty2 = Decimal("5.2")
    dov = Date(2023, 1, 1)
    price1 = SomePrice(ccy, qty1, dov)
    price2 = SomePrice(ccy, qty2, dov)
    result = price1 + price2
    expected = SomePrice(ccy, Decimal("15.7"), dov)
    assert result == expected


# LLM-generated content at query #15
#--------------------------

def test_le_with_defined_prices_same_currency():
    from pypara.currencies import Currencies
    from pypara.monetary import Price
    from datetime import date
    from decimal import Decimal
    price1 = Price.of(Currencies["USD"], Decimal('10'), date(2023, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('20'), date(2023, 1, 1))
    result = price1 <= price2
    assert result is True

def test_le_with_defined_prices_same_currency_equal():
    from pypara.currencies import Currencies
    from pypara.monetary import Price
    from datetime import date
    from decimal import Decimal
    price1 = Price.of(Currencies["USD"], Decimal('10'), date(2023, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('10'), date(2023, 1, 1))
    result = price1 <= price2
    assert result is True

def test_le_with_defined_prices_same_currency_greater():
    from pypara.currencies import Currencies
    from pypara.monetary import Price
    from datetime import date
    from decimal import Decimal
    price1 = Price.of(Currencies["USD"], Decimal('20'), date(2023, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('10'), date(2023, 1, 1))
    result = price1 <= price2
    assert result is False

def test_le_with_undefined_price_left():
    from pypara.currencies import Currencies
    from pypara.monetary import Price
    from datetime import date
    from decimal import Decimal
    price1 = Price.na()
    price2 = Price.of(Currencies["USD"], Decimal('10'), date(2023, 1, 1))
    result = price1 <= price2
    assert result is True

def test_le_with_undefined_price_right():
    from pypara.currencies import Currencies
    from pypara.monetary import Price
    from datetime import date
    from decimal import Decimal
    price1 = Price.of(Currencies["USD"], Decimal('10'), date(2023, 1, 1))
    price2 = Price.na()
    result = price1 <= price2
    assert result is False

def test_le_with_both_undefined():
    from pypara.monetary import Price
    price1 = Price.na()
    price2 = Price.na()
    result = price1 <= price2
    assert result is True

def test_le_raises_incompatible_currency_error():
    from pypara.currencies import Currencies
    from pypara.monetary import Price, IncompatibleCurrencyError
    from datetime import date
    from decimal import Decimal
    price1 = Price.of(Currencies["USD"], Decimal('10'), date(2023, 1, 1))
    price2 = Price.of(Currencies["EUR"], Decimal('10'), date(2023, 1, 1))
    try:
        _ = price1 <= price2
        assert False, "Expected IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        assert True


# LLM-generated content at query #16
#--------------------------

def test_divide_defined_price_by_positive_number():
    from pypara.currencies import Currencies
    from pypara.monetary import Price
    from decimal import Decimal
    from datetime import date
    price = Price.of(Currencies["USD"], Decimal('10'), date(2023, 1, 1))
    result = price.divide(Decimal('2'))
    assert result.defined
    assert result.qty_or_none() == Decimal('5')
    assert result.ccy_or_none() == Currencies["USD"]
    assert result.dov_or_none() == date(2023, 1, 1)

def test_divide_defined_price_by_one():
    from pypara.currencies import Currencies
    from pypara.monetary import Price
    from decimal import Decimal
    from datetime import date
    price = Price.of(Currencies["EUR"], Decimal('7.5'), date(2023, 1, 1))
    result = price.divide(Decimal('1'))
    assert result.defined
    assert result.qty_or_none() == Decimal('7.5')
    assert result.ccy_or_none() == Currencies["EUR"]
    assert result.dov_or_none() == date(2023, 1, 1)

def test_divide_defined_price_by_negative_number():
    from pypara.currencies import Currencies
    from pypara.monetary import Price
    from decimal import Decimal
    from datetime import date
    price = Price.of(Currencies["JPY"], Decimal('100'), date(2023, 1, 1))
    result = price.divide(Decimal('-2'))
    assert result.defined
    assert result.qty_or_none() == Decimal('-50')
    assert result.ccy_or_none() == Currencies["JPY"]
    assert result.dov_or_none() == date(2023, 1, 1)

def test_divide_defined_price_by_zero_yields_undefined():
    from pypara.currencies import Currencies
    from pypara.monetary import Price
    from decimal import Decimal
    from datetime import date
    price = Price.of(Currencies["USD"], Decimal('10'), date(2023, 1, 1))
    result = price.divide(Decimal('0'))
    assert result.undefined

def test_divide_undefined_price_returns_itself():
    from pypara.monetary import Price
    from decimal import Decimal
    undefined_price = Price.na()
    result = undefined_price.divide(Decimal('5'))
    assert result is undefined_price
    assert result.undefined

def test_divide_defined_price_by_decimal_fraction():
    from pypara.currencies import Currencies
    from pypara.monetary import Price
    from decimal import Decimal
    from datetime import date
    price = Price.of(Currencies["GBP"], Decimal('1'), date(2023, 1, 1))
    result = price.divide(Decimal('0.5'))
    assert result.defined
    assert result.qty_or_none() == Decimal('2')
    assert result.ccy_or_none() == Currencies["GBP"]
    assert result.dov_or_none() == date(2023, 1, 1)

def test_divide_defined_price_by_large_number():
    from pypara.currencies import Currencies
    from pypara.monetary import Price
    from decimal import Decimal
    from datetime import date
    price = Price.of(Currencies["USD"], Decimal('1000'), date(2023, 1, 1))
    result = price.divide(Decimal('1000'))
    assert result.defined
    assert result.qty_or_none() == Decimal('1')
    assert result.ccy_or_none() == Currencies["USD"]
    assert result.dov_or_none() == date(2023, 1, 1)

def test_divide_defined_price_with_negative_quantity():
    from pypara.currencies import Currencies
    from pypara.monetary import Price
    from decimal import Decimal
    from datetime import date
    price = Price.of(Currencies["EUR"], Decimal('-8'), date(2023, 1, 1))
    result = price.divide(Decimal('4'))
    assert result.defined
    assert result.qty_or_none() == Decimal('-2')
    assert result.ccy_or_none() == Currencies["EUR"]
    assert result.dov_or_none() == date(2023, 1, 1)

def test_divide_defined_price_by_float_like_decimal():
    from pypara.currencies import Currencies
    from pypara.monetary import Price
    from decimal import Decimal
    from datetime import date
    price = Price.of(Currencies["USD"], Decimal('9'), date(2023, 1, 1))
    result = price.divide(Decimal('2.5'))
    assert result.defined
    assert result.qty_or_none() == Decimal('3.6')
    assert result.ccy_or_none() == Currencies["USD"]
    assert result.dov_or_none() == date(2023, 1, 1)


# LLM-generated content at query #17
#--------------------------

def test_subtract_defined_prices_same_currency():
    from pypara.currencies import Currencies
    from pypara.monetary import Price
    from datetime import date
    from decimal import Decimal
    p1 = Price.of(Currencies["USD"], Decimal('10'), date(2023, 1, 1))
    p2 = Price.of(Currencies["USD"], Decimal('3'), date(2023, 1, 1))
    result = p1.subtract(p2)
    assert result.qty == Decimal('7')
    assert result.ccy == Currencies["USD"]
    assert result.dov == date(2023, 1, 1)

def test_subtract_defined_prices_different_currency_raises():
    from pypara.currencies import Currencies
    from pypara.monetary import Price, IncompatibleCurrencyError
    from datetime import date
    from decimal import Decimal
    p1 = Price.of(Currencies["USD"], Decimal('10'), date(2023, 1, 1))
    p2 = Price.of(Currencies["EUR"], Decimal('3'), date(2023, 1, 1))
    try:
        p1.subtract(p2)
        assert False
    except IncompatibleCurrencyError:
        assert True

def test_subtract_first_undefined_returns_second():
    from pypara.currencies import Currencies
    from pypara.monetary import Price
    from datetime import date
    from decimal import Decimal
    p1 = Price.na()
    p2 = Price.of(Currencies["USD"], Decimal('3'), date(2023, 1, 1))
    result = p1.subtract(p2)
    assert result is p2

def test_subtract_second_undefined_returns_first():
    from pypara.currencies import Currencies
    from pypara.monetary import Price
    from datetime import date
    from decimal import Decimal
    p1 = Price.of(Currencies["USD"], Decimal('10'), date(2023, 1, 1))
    p2 = Price.na()
    result = p1.subtract(p2)
    assert result is p1

def test_subtract_both_undefined_returns_undefined():
    from pypara.monetary import Price
    p1 = Price.na()
    p2 = Price.na()
    result = p1.subtract(p2)
    assert result.undefined

def test_subtract_negative_result():
    from pypara.currencies import Currencies
    from pypara.monetary import Price
    from datetime import date
    from decimal import Decimal
    p1 = Price.of(Currencies["USD"], Decimal('3'), date(2023, 1, 1))
    p2 = Price.of(Currencies["USD"], Decimal('10'), date(2023, 1, 1))
    result = p1.subtract(p2)
    assert result.qty == Decimal('-7')
    assert result.ccy == Currencies["USD"]
    assert result.dov == date(2023, 1, 1)

def test_subtract_zero_result():
    from pypara.currencies import Currencies
    from pypara.monetary import Price
    from datetime import date
    from decimal import Decimal
    p1 = Price.of(Currencies["USD"], Decimal('5'), date(2023, 1, 1))
    p2 = Price.of(Currencies["USD"], Decimal('5'), date(2023, 1, 1))
    result = p1.subtract(p2)
    assert result.qty == Decimal('0')
    assert result.ccy == Currencies["USD"]
    assert result.dov == date(2023, 1, 1)


# LLM-generated content at query #18
#--------------------------

def test_qty_map_defined_money():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from decimal import Decimal
    from datetime import date
    somemoney = Money.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    result = somemoney.qty_map(lambda x: x + Decimal('1'), lambda: Decimal('42'))
    assert result == Decimal('2.00')

def test_qty_map_undefined_money():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from decimal import Decimal
    from datetime import date
    nonemoney = Money.of(None, Decimal('1'), None)
    result = nonemoney.qty_map(lambda x: x + Decimal('1'), lambda: Decimal('42'))
    assert result == Decimal('42')

def test_qty_map_defined_money_with_different_return_type():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from decimal import Decimal
    from datetime import date
    somemoney = Money.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    result = somemoney.qty_map(lambda x: str(x), lambda: "fallback")
    assert result == "1.00"

def test_qty_map_undefined_money_with_different_return_type():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from decimal import Decimal
    from datetime import date
    nonemoney = Money.of(None, Decimal('1'), None)
    result = nonemoney.qty_map(lambda x: str(x), lambda: "fallback")
    assert result == "fallback"

def test_qty_map_defined_money_zero_quantity():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from decimal import Decimal
    from datetime import date
    somemoney = Money.of(Currencies["USD"], Decimal('0'), date(2019, 1, 1))
    result = somemoney.qty_map(lambda x: x * Decimal('2'), lambda: Decimal('99'))
    assert result == Decimal('0.00')

def test_qty_map_undefined_money_with_none_quantity():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from decimal import Decimal
    from datetime import date
    nonemoney = Money.of(Currencies["USD"], None, date(2019, 1, 1))
    result = nonemoney.qty_map(lambda x: x + Decimal('1'), lambda: Decimal('100'))
    assert result == Decimal('100')


# LLM-generated content at query #19
#--------------------------

def test_as_float_defined_money():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from decimal import Decimal
    from datetime import date
    money = Money.of(Currencies["USD"], Decimal('123.456'), date(2023, 1, 1))
    result = money.as_float()
    expected = 123.456
    assert result == expected

def test_as_float_undefined_money():
    from pypara.money import Money
    money = Money.na()
    try:
        money.as_float()
        assert False
    except MonetaryOperationException:
        assert True

def test_as_float_zero_quantity():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from decimal import Decimal
    from datetime import date
    money = Money.of(Currencies["USD"], Decimal('0'), date(2023, 1, 1))
    result = money.as_float()
    expected = 0.0
    assert result == expected

def test_as_float_negative_quantity():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from decimal import Decimal
    from datetime import date
    money = Money.of(Currencies["USD"], Decimal('-123.456'), date(2023, 1, 1))
    result = money.as_float()
    expected = -123.456
    assert result == expected

def test_as_float_large_quantity():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from decimal import Decimal
    from datetime import date
    money = Money.of(Currencies["USD"], Decimal('999999.999'), date(2023, 1, 1))
    result = money.as_float()
    expected = 999999.999
    assert result == expected


# LLM-generated content at query #20
#--------------------------

def test_convert_defined_price_with_valid_fx_rate():
    from pypara.currencies import Currencies
    from pypara.currencies import Currency
    from datetime import date
    from decimal import Decimal
    from pypara.prices import Price
    from pypara.fx import FXRateService
    from unittest.mock import Mock
    fx_service = Mock(spec=FXRateService)
    fx_service.query.return_value = Decimal('0.85')
    price = Price.of(Currencies["USD"], Decimal('100'), date(2023, 1, 1))
    converted = price.convert(Currencies["EUR"], asof=date(2023, 1, 1), strict=True, fx=fx_service)
    assert converted.ccy == Currencies["EUR"]
    assert converted.qty == Decimal('85')
    assert converted.dov == date(2023, 1, 1)

def test_convert_undefined_price_returns_itself():
    from pypara.currencies import Currencies
    from pypara.prices import Price
    undefined_price = Price.na()
    converted = undefined_price.convert(Currencies["EUR"])
    assert converted is undefined_price

def test_convert_same_currency_returns_same_price():
    from pypara.currencies import Currencies
    from datetime import date
    from decimal import Decimal
    from pypara.prices import Price
    price = Price.of(Currencies["USD"], Decimal('100'), date(2023, 1, 1))
    converted = price.convert(Currencies["USD"])
    assert converted is price

def test_convert_without_asof_uses_price_dov():
    from pypara.currencies import Currencies
    from datetime import date
    from decimal import Decimal
    from pypara.prices import Price
    from pypara.fx import FXRateService
    from unittest.mock import Mock
    fx_service = Mock(spec=FXRateService)
    fx_service.query.return_value = Decimal('0.85')
    price = Price.of(Currencies["USD"], Decimal('100'), date(2023, 1, 1))
    converted = price.convert(Currencies["EUR"], fx=fx_service)
    fx_service.query.assert_called_with(Currencies["USD"], Currencies["EUR"], date(2023, 1, 1))
    assert converted.ccy == Currencies["EUR"]
    assert converted.qty == Decimal('85')
    assert converted.dov == date(2023, 1, 1)

def test_convert_with_asof_overrides_dov():
    from pypara.currencies import Currencies
    from datetime import date
    from decimal import Decimal
    from pypara.prices import Price
    from pypara.fx import FXRateService
    from unittest.mock import Mock
    fx_service = Mock(spec=FXRateService)
    fx_service.query.return_value = Decimal('0.90')
    price = Price.of(Currencies["USD"], Decimal('100'), date(2023, 1, 1))
    converted = price.convert(Currencies["EUR"], asof=date(2023, 6, 1), fx=fx_service)
    fx_service.query.assert_called_with(Currencies["USD"], Currencies["EUR"], date(2023, 6, 1))
    assert converted.ccy == Currencies["EUR"]
    assert converted.qty == Decimal('90')
    assert converted.dov == date(2023, 6, 1)

def test_convert_strict_true_raises_on_missing_fx_rate():
    from pypara.currencies import Currencies
    from datetime import date
    from decimal import Decimal
    from pypara.prices import Price
    from pypara.fx import FXRateService, FXRateLookupError
    from unittest.mock import Mock
    fx_service = Mock(spec=FXRateService)
    fx_service.query.side_effect = FXRateLookupError("Rate not found")
    price = Price.of(Currencies["USD"], Decimal('100'), date(2023, 1, 1))
    try:
        price.convert(Currencies["EUR"], asof=date(2023, 1, 1), strict=True, fx=fx_service)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError:
        pass

def test_convert_strict_false_returns_undefined_on_missing_fx_rate():
    from pypara.currencies import Currencies
    from datetime import date
    from decimal import Decimal
    from pypara.prices import Price
    from pypara.fx import FXRateService, FXRateLookupError
    from unittest.mock import Mock
    fx_service = Mock(spec=FXRateService)
    fx_service.query.side_effect = FXRateLookupError("Rate not found")
    price = Price.of(Currencies["USD"], Decimal('100'), date(2023, 1, 1))
    converted = price.convert(Currencies["EUR"], asof=date(2023, 1, 1), strict=False, fx=fx_service)
    assert converted.undefined

def test_convert_uses_default_fx_service_when_not_provided():
    from pypara.currencies import Currencies
    from datetime import date
    from decimal import Decimal
    from pypara.prices import Price
    price = Price.of(Currencies["USD"], Decimal('100'), date(2023, 1, 1))
    try:
        price.convert(Currencies["EUR"])
        assert False, "Expected FXRateLookupError due to missing default service"
    except FXRateLookupError:
        pass

def test_convert_zero_quantity_returns_zero_in_target_currency():
    from pypara.currencies import Currencies
    from datetime import date
    from decimal import Decimal
    from pypara.prices import Price
    from pypara.fx import FXRateService
    from unittest.mock import Mock
    fx_service = Mock(spec=FXRateService)
    fx_service.query.return_value = Decimal('0.85')
    price = Price.of(Currencies["USD"], Decimal('0'), date(2023, 1, 1))
    converted = price.convert(Currencies["EUR"], fx=fx_service)
    assert converted.ccy == Currencies["EUR"]
    assert converted.qty == Decimal('0')
    assert converted.dov == date(2023, 1, 1)

def test_convert_negative_quantity_converts_correctly():
    from pypara.currencies import Currencies
    from datetime import date
    from decimal import Decimal
    from pypara.prices import Price
    from pypara.fx import FXRateService
    from unittest.mock import Mock
    fx_service = Mock(spec=FXRateService)
    fx_service.query.return_value = Decimal('0.85')
    price = Price.of(Currencies["USD"], Decimal('-100'), date(2023, 1, 1))
    converted = price.convert(Currencies["EUR"], fx=fx_service)
    assert converted.ccy == Currencies["EUR"]
    assert converted.qty == Decimal('-85')
    assert converted.dov == date(2023, 1, 1)


