####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_is_equal_with_same_money_objects():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    money1 = Money.of(Currencies["USD"], Decimal('100.00'), date(2023, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('100.00'), date(2023, 1, 1))
    assert money1.is_equal(money2) is True

def test_is_equal_with_different_money_objects():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    money1 = Money.of(Currencies["USD"], Decimal('100.00'), date(2023, 1, 1))
    money2 = Money.of(Currencies["EUR"], Decimal('100.00'), date(2023, 1, 1))
    assert money1.is_equal(money2) is False

def test_is_equal_with_undefined_money():
    money1 = Money.na()
    money2 = Money.na()
    assert money1.is_equal(money2) is True

def test_is_equal_with_undefined_and_defined_money():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    money1 = Money.na()
    money2 = Money.of(Currencies["USD"], Decimal('100.00'), date(2023, 1, 1))
    assert money1.is_equal(money2) is False

def test_is_equal_with_non_money_object():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    money = Money.of(Currencies["USD"], Decimal('100.00'), date(2023, 1, 1))
    assert money.is_equal("not a money object") is False


# LLM-generated content at query #2
#--------------------------

```python
def test_ccy_or_defined_price():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    price = Price.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    assert price.ccy_or(Currencies["EUR"]).code == "USD"

def test_ccy_or_undefined_price():
    from pypara.currencies import Currencies
    from decimal import Decimal
    price = Price.of(None, Decimal('1'), None)
    assert price.ccy_or(Currencies["EUR"]).code == "EUR"


# LLM-generated content at query #3
#--------------------------

```python
def test_lte_undefined_price_is_less_than_or_equal_to_defined_price():
    undefined_price = Price.na()
    defined_price = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert undefined_price.lte(defined_price) is True

def test_lte_defined_price_is_not_less_than_or_equal_to_undefined_price():
    defined_price = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    undefined_price = Price.na()
    assert defined_price.lte(undefined_price) is False

def test_lte_same_defined_prices_are_equal():
    price1 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert price1.lte(price2) is True

def test_lte_lesser_defined_price_is_less_than_greater_defined_price():
    price1 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    assert price1.lte(price2) is True

def test_lte_greater_defined_price_is_not_less_than_or_equal_to_lesser_defined_price():
    price1 = Price.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert price1.lte(price2) is False

def test_lte_incompatible_currency_error_raised_for_different_currencies():
    price1 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["EUR"], Decimal('1'), Date(2019, 1, 1))
    try:
        price1.lte(price2)
        assert False, "Expected IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass


# LLM-generated content at query #4
#--------------------------

```python
def test_some_price_equality():
    price1 = SomePrice(Currency("USD"), Decimal("100.00"), Date(2023, 1, 1))
    price2 = SomePrice(Currency("USD"), Decimal("100.00"), Date(2023, 1, 1))
    price3 = SomePrice(Currency("USD"), Decimal("200.00"), Date(2023, 1, 1))
    price4 = SomePrice(Currency("EUR"), Decimal("100.00"), Date(2023, 1, 1))

    assert price1.__eq__(price2) is True
    assert price1.__eq__(price3) is False
    assert price1.__eq__(price4) is False
    assert price1.__eq__(None) is False
    assert price1.__eq__("not a price") is False


# LLM-generated content at query #5
#--------------------------

```python
def test_divide_defined_price():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    price = Price.of(Currencies["USD"], Decimal('10'), date(2019, 1, 1))
    result = price.divide(Decimal('2'))
    assert result.qty == Decimal('5')
    assert result.ccy == Currencies["USD"]
    assert result.dov == date(2019, 1, 1)

def test_divide_undefined_price():
    price = Price.na()
    result = price.divide(Decimal('2'))
    assert result.undefined

def test_divide_by_zero():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    price = Price.of(Currencies["USD"], Decimal('10'), date(2019, 1, 1))
    result = price.divide(Decimal('0'))
    assert result.undefined


# LLM-generated content at query #6
#--------------------------

```python
def test_fmap_defined_price():
    from pypara.currencies import Currencies
    from datetime import date, timedelta
    someprice = Price.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    new = someprice.fmap(lambda x: Price.of(x.ccy, x.qty + Decimal('1'), x.dov + timedelta(days=10)))
    assert new.ccy.code == 'USD'
    assert new.qty == Decimal('2')
    assert new.dov == date(2019, 1, 11)

def test_fmap_undefined_price():
    noneprice = Price.of(None, Decimal('1'), None)
    assert noneprice.fmap(lambda sp: Price.of(sp.ccy, sp.qty + Decimal('1'), sp.dov)) is Price.na()


# LLM-generated content at query #7
#--------------------------

```python
def test_multiply_defined_price():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    price = Price.of(Currencies["USD"], Decimal('10'), date(2019, 1, 1))
    result = price.multiply(2)
    assert result.qty == Decimal('20')
    assert result.ccy == Currencies["USD"]
    assert result.dov == date(2019, 1, 1)

def test_multiply_undefined_price():
    price = Price.na()
    result = price.multiply(2)
    assert result.undefined


# LLM-generated content at query #8
#--------------------------

```python
def test_divide_defined_money():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    usd = Currencies["USD"]
    money = Money.of(usd, Decimal('10'), date(2019, 1, 1))
    result = money.divide(Decimal('2'))
    assert result.qty == Decimal('5.00')
    assert result.ccy == usd
    assert result.dov == date(2019, 1, 1)

def test_divide_undefined_money():
    money = Money.na()
    result = money.divide(Decimal('2'))
    assert result.undefined

def test_divide_by_zero():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    usd = Currencies["USD"]
    money = Money.of(usd, Decimal('10'), date(2019, 1, 1))
    result = money.divide(Decimal('0'))
    assert result.undefined


# LLM-generated content at query #9
#--------------------------

```python
def test_price_sub_defined_undefined():
    from pypara.currencies import Currencies
    from datetime import date
    from decimal import Decimal
    price1 = Price.of(Currencies["USD"], Decimal('100.00'), date(2023, 1, 1))
    price2 = Price.na()
    result = price1 - price2
    assert result.is_equal(price1)

def test_price_sub_undefined_defined():
    from pypara.currencies import Currencies
    from datetime import date
    from decimal import Decimal
    price1 = Price.na()
    price2 = Price.of(Currencies["USD"], Decimal('100.00'), date(2023, 1, 1))
    result = price1 - price2
    assert result.is_equal(price2)

def test_price_sub_undefined_undefined():
    price1 = Price.na()
    price2 = Price.na()
    result = price1 - price2
    assert result.is_equal(price1)

def test_price_sub_defined_defined_same_currency():
    from pypara.currencies import Currencies
    from datetime import date
    from decimal import Decimal
    price1 = Price.of(Currencies["USD"], Decimal('200.00'), date(2023, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('100.00'), date(2023, 1, 2))
    result = price1 - price2
    assert result.ccy.code == "USD"
    assert result.qty == Decimal('100.00')
    assert result.dov == date(2023, 1, 2)


# LLM-generated content at query #10
#--------------------------

```python
def test_qty_or_else_defined_price():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    price = Price.of(Currencies["USD"], Decimal('10.5'), date(2023, 1, 1))
    assert price.qty_or_else(lambda: Decimal('20.0')) == Decimal('10.5')
    assert price.qty_or_else(lambda: "fallback") == Decimal('10.5')

def test_qty_or_else_undefined_price():
    price = Price.na()
    assert price.qty_or_else(lambda: Decimal('20.0')) == Decimal('20.0')
    assert price.qty_or_else(lambda: "fallback") == "fallback"


# LLM-generated content at query #11
#--------------------------

```python
def test_add_defined_money_with_same_currency():
    from pypara.currencies import Currencies
    from datetime import date
    from decimal import Decimal
    money1 = Money.of(Currencies["USD"], Decimal('10.50'), date(2023, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('5.25'), date(2023, 1, 2))
    result = money1 + money2
    assert result.ccy == Currencies["USD"]
    assert result.qty == Decimal('15.75')
    assert result.dov == date(2023, 1, 2)

def test_add_defined_money_with_different_currency():
    from pypara.currencies import Currencies
    from datetime import date
    from decimal import Decimal
    money1 = Money.of(Currencies["USD"], Decimal('10.50'), date(2023, 1, 1))
    money2 = Money.of(Currencies["EUR"], Decimal('5.25'), date(2023, 1, 2))
    try:
        result = money1 + money2
        assert False, "Expected IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass

def test_add_undefined_money_with_defined():
    from pypara.currencies import Currencies
    from datetime import date
    from decimal import Decimal
    money1 = Money.na()
    money2 = Money.of(Currencies["USD"], Decimal('5.25'), date(2023, 1, 2))
    result = money1 + money2
    assert result is money2

def test_add_defined_money_with_undefined():
    from pypara.currencies import Currencies
    from datetime import date
    from decimal import Decimal
    money1 = Money.of(Currencies["USD"], Decimal('10.50'), date(2023, 1, 1))
    money2 = Money.na()
    result = money1 + money2
    assert result is money1

def test_add_two_undefined_money():
    money1 = Money.na()
    money2 = Money.na()
    result = money1 + money2
    assert result is money1


# LLM-generated content at query #12
#--------------------------

```python
def test_le_undefined_vs_defined():
    assert Price.na() <= Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))

def test_le_defined_vs_undefined():
    assert not Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1)) <= Price.na()

def test_le_same_currency():
    assert Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1)) <= Price.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))

def test_le_different_currency():
    with pytest.raises(IncompatibleCurrencyError):
        Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1)) <= Price.of(Currencies["EUR"], Decimal('1'), Date(2019, 1, 1))


# LLM-generated content at query #13
#--------------------------

```python
def test_or_else_defined_money_returns_itself():
    from pypara.currencies import Currencies
    from datetime import date
    from decimal import Decimal
    somemoney = Money.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    fallback = Money.of(Currencies["EUR"], Decimal('2'), date(2019, 1, 2))
    assert somemoney.or_else(lambda: fallback) is somemoney

def test_or_else_undefined_money_returns_fallback():
    from pypara.currencies import Currencies
    from datetime import date
    from decimal import Decimal
    fallback = Money.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    nonemoney = Money.of(None, Decimal('1'), None)
    assert nonemoney.or_else(lambda: fallback) is fallback


# LLM-generated content at query #14
#--------------------------

```python
def test_is_equal_returns_true_for_same_price_objects():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    price1 = Price.of(Currencies["USD"], Decimal('100.50'), date(2023, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('100.50'), date(2023, 1, 1))
    assert price1.is_equal(price2) is True

def test_is_equal_returns_false_for_different_price_objects():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    price1 = Price.of(Currencies["USD"], Decimal('100.50'), date(2023, 1, 1))
    price2 = Price.of(Currencies["EUR"], Decimal('100.50'), date(2023, 1, 1))
    assert price1.is_equal(price2) is False

def test_is_equal_returns_false_for_undefined_price():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    price1 = Price.of(Currencies["USD"], Decimal('100.50'), date(2023, 1, 1))
    price2 = Price.na()
    assert price1.is_equal(price2) is False

def test_is_equal_returns_true_for_two_undefined_prices():
    price1 = Price.na()
    price2 = Price.na()
    assert price1.is_equal(price2) is True

def test_is_equal_returns_false_for_non_price_object():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    price = Price.of(Currencies["USD"], Decimal('100.50'), date(2023, 1, 1))
    assert price.is_equal("not a price") is False


# LLM-generated content at query #15
#--------------------------

```python
def test_price_truediv_defined():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    price = Price.of(Currencies["USD"], Decimal('10'), date(2019, 1, 1))
    result = price / Decimal('2')
    assert result.qty == Decimal('5')
    assert result.ccy == Currencies["USD"]
    assert result.dov == date(2019, 1, 1)

def test_price_truediv_undefined():
    price = Price.na()
    result = price / Decimal('2')
    assert result.undefined

def test_price_truediv_by_zero():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    price = Price.of(Currencies["USD"], Decimal('10'), date(2019, 1, 1))
    result = price / Decimal('0')
    assert result.undefined


# LLM-generated content at query #16
#--------------------------

```python
def test_convert_with_valid_fx_rate():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    price = SomePrice(usd, Decimal("100.00"), Date(2023, 1, 1))
    FXRateService.default = MockFXRateService({FXRate(usd, eur, Decimal("0.85"), Date(2023, 1, 1))})
    converted_price = price.convert(eur, Date(2023, 1, 1))
    assert converted_price == SomePrice(eur, Decimal("85.00"), Date(2023, 1, 1))

def test_convert_with_no_fx_rate_non_strict():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    price = SomePrice(usd, Decimal("100.00"), Date(2023, 1, 1))
    FXRateService.default = MockFXRateService({})
    converted_price = price.convert(eur, Date(2023, 1, 1), strict=False)
    assert converted_price == NoPrice

def test_convert_with_no_fx_rate_strict():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    price = SomePrice(usd, Decimal("100.00"), Date(2023, 1, 1))
    FXRateService.default = MockFXRateService({})
    try:
        price.convert(eur, Date(2023, 1, 1), strict=True)
    except FXRateLookupError:
        pass
    else:
        assert False, "Expected FXRateLookupError"

def test_convert_with_no_fx_service():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    price = SomePrice(usd, Decimal("100.00"), Date(2023, 1, 1))
    FXRateService.default = None
    try:
        price.convert(eur, Date(2023, 1, 1))
    except ProgrammingError:
        pass
    else:
        assert False, "Expected ProgrammingError"

def test_convert_with_default_asof():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    price = SomePrice(usd, Decimal("100.00"), Date(2023, 1, 1))
    FXRateService.default = MockFXRateService({FXRate(usd, eur, Decimal("0.85"), Date(2023, 1, 1))})
    converted_price = price.convert(eur)
    assert converted_price == SomePrice(eur, Decimal("85.00"), Date(2023, 1, 1))


# LLM-generated content at query #17
#--------------------------

```python
def test_none_money_constructor():
    none_money = NoneMoney()
    assert isinstance(none_money, NoneMoney)
    assert not none_money.defined
    assert none_money.undefined


# LLM-generated content at query #18
#--------------------------

```python
def test_none_money_constructor():
    none_money = NoneMoney()
    assert isinstance(none_money, NoneMoney)
    assert not none_money.defined
    assert none_money.undefined


# LLM-generated content at query #19
#--------------------------

```python
def test_subtract_defined_prices_same_currency():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date

    price1 = Price.of(Currencies["USD"], Decimal('100'), date(2023, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('50'), date(2023, 1, 2))
    result = price1.subtract(price2)

    assert result.defined
    assert result.ccy == Currencies["USD"]
    assert result.qty == Decimal('50')
    assert result.dov == date(2023, 1, 2)

def test_subtract_defined_prices_different_currency():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    from pypara.errors import IncompatibleCurrencyError

    price1 = Price.of(Currencies["USD"], Decimal('100'), date(2023, 1, 1))
    price2 = Price.of(Currencies["EUR"], Decimal('50'), date(2023, 1, 2))

    try:
        price1.subtract(price2)
        assert False, "Expected IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass

def test_subtract_undefined_price_with_defined():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date

    price1 = Price.na()
    price2 = Price.of(Currencies["USD"], Decimal('50'), date(2023, 1, 2))
    result = price1.subtract(price2)

    assert result is price2
    assert result.defined

def test_subtract_defined_price_with_undefined():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date

    price1 = Price.of(Currencies["USD"], Decimal('100'), date(2023, 1, 1))
    price2 = Price.na()
    result = price1.subtract(price2)

    assert result is price1
    assert result.defined

def test_subtract_two_undefined_prices():
    price1 = Price.na()
    price2 = Price.na()
    result = price1.subtract(price2)

    assert result is price1
    assert result.undefined


# LLM-generated content at query #20
#--------------------------

```python
def test_abs_defined_price():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    price = Price.of(Currencies["USD"], Decimal('-1.5'), date(2019, 1, 1))
    abs_price = price.abs()
    assert abs_price.qty == Decimal('1.5')
    assert abs_price.ccy == Currencies["USD"]
    assert abs_price.dov == date(2019, 1, 1)

def test_abs_undefined_price():
    undefined_price = Price.na()
    abs_undefined_price = undefined_price.abs()
    assert abs_undefined_price is undefined_price


# LLM-generated content at query #21
#--------------------------

```python
def test_convert_with_valid_fx_rate():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    price = SomePrice(usd, Decimal("100.00"), Date(2023, 1, 1))
    FXRateService.default = MockFXRateService({FXRate(usd, eur, Date(2023, 1, 1), Decimal("0.90"))})
    result = price.convert(eur, Date(2023, 1, 1))
    assert result == SomePrice(eur, Decimal("90.00"), Date(2023, 1, 1))

def test_convert_with_no_fx_rate_non_strict():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    price = SomePrice(usd, Decimal("100.00"), Date(2023, 1, 1))
    FXRateService.default = MockFXRateService({})
    result = price.convert(eur, Date(2023, 1, 1), strict=False)
    assert result == NoPrice

def test_convert_with_no_fx_rate_strict():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    price = SomePrice(usd, Decimal("100.00"), Date(2023, 1, 1))
    FXRateService.default = MockFXRateService({})
    try:
        price.convert(eur, Date(2023, 1, 1), strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError:
        pass

def test_convert_with_no_fx_service():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    price = SomePrice(usd, Decimal("100.00"), Date(2023, 1, 1))
    FXRateService.default = None
    try:
        price.convert(eur, Date(2023, 1, 1))
        assert False, "Expected ProgrammingError"
    except ProgrammingError:
        pass

def test_convert_with_default_date():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    price = SomePrice(usd, Decimal("100.00"), Date(2023, 1, 1))
    FXRateService.default = MockFXRateService({FXRate(usd, eur, Date(2023, 1, 1), Decimal("0.90"))})
    result = price.convert(eur)
    assert result == SomePrice(eur, Decimal("90.00"), Date(2023, 1, 1))


# LLM-generated content at query #22
#--------------------------

```python
def test_subtract_defined_minus_defined():
    from pypara.currencies import Currencies
    from datetime import date
    from decimal import Decimal
    m1 = Money.of(Currencies["USD"], Decimal('10.00'), date(2023, 1, 1))
    m2 = Money.of(Currencies["USD"], Decimal('3.00'), date(2023, 1, 2))
    result = m1.subtract(m2)
    assert result.defined
    assert result.ccy == Currencies["USD"]
    assert result.qty == Decimal('7.00')
    assert result.dov == date(2023, 1, 2)

def test_subtract_defined_minus_undefined():
    from pypara.currencies import Currencies
    from datetime import date
    from decimal import Decimal
    m1 = Money.of(Currencies["USD"], Decimal('10.00'), date(2023, 1, 1))
    m2 = Money.na()
    result = m1.subtract(m2)
    assert result.is_equal(m1)

def test_subtract_undefined_minus_defined():
    from pypara.currencies import Currencies
    from datetime import date
    from decimal import Decimal
    m1 = Money.na()
    m2 = Money.of(Currencies["USD"], Decimal('3.00'), date(2023, 1, 2))
    result = m1.subtract(m2)
    assert result.is_equal(m2)

def test_subtract_undefined_minus_undefined():
    m1 = Money.na()
    m2 = Money.na()
    result = m1.subtract(m2)
    assert result.undefined

def test_subtract_incompatible_currency():
    from pypara.currencies import Currencies
    from datetime import date
    from decimal import Decimal
    from pypara.errors import IncompatibleCurrencyError
    m1 = Money.of(Currencies["USD"], Decimal('10.00'), date(2023, 1, 1))
    m2 = Money.of(Currencies["EUR"], Decimal('3.00'), date(2023, 1, 2))
    try:
        m1.subtract(m2)
        assert False, "Expected IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_somemoney_constructor():
    ccy = Currency("USD")
    qty = Decimal("100.50")
    dov = Date(2023, 1, 1)
    money = SomeMoney(ccy, qty, dov)
    assert money.ccy == ccy
    assert money.qty == qty
    assert money.dov == dov
    assert money.defined is True
    assert money.undefined is False


# LLM-generated content at query #2
#--------------------------

```python
def test_dov_or_defined_money():
    from pypara.currencies import Currencies
    from datetime import date
    somemoney = Money.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    assert somemoney.dov_or(date(2001, 1, 1)) == date(2019, 1, 1)

def test_dov_or_undefined_money():
    nonemoney = Money.of(None, None, date(2019, 1, 1))
    assert nonemoney.dov_or(date(2001, 1, 1)) == date(2001, 1, 1)


# LLM-generated content at query #3
#--------------------------

```python
def test_qty_or():
    usd = Currency("USD")
    money = SomeMoney(usd, Decimal("100.50"), Date(2023, 1, 1))
    assert money.qty_or(Decimal("0")) == Decimal("100.50")
    assert money.qty_or(Decimal("999.99")) == Decimal("100.50")


# LLM-generated content at query #4
#--------------------------

```python
def test_scalar_add_with_defined_price():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date

    price = Price.of(Currencies["USD"], Decimal('10.5'), date(2023, 1, 1))
    result = price.scalar_add(Decimal('5.5'))

    assert result.defined is True
    assert result.ccy == Currencies["USD"]
    assert result.qty == Decimal('16.0')
    assert result.dov == date(2023, 1, 1)

def test_scalar_add_with_undefined_price():
    undefined_price = Price.na()
    result = undefined_price.scalar_add(Decimal('5.5'))

    assert result.defined is False
    assert result is undefined_price

def test_scalar_add_with_integer():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date

    price = Price.of(Currencies["USD"], Decimal('10.5'), date(2023, 1, 1))
    result = price.scalar_add(5)

    assert result.defined is True
    assert result.ccy == Currencies["USD"]
    assert result.qty == Decimal('15.5')
    assert result.dov == date(2023, 1, 1)

def test_scalar_add_with_float():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date

    price = Price.of(Currencies["USD"], Decimal('10.5'), date(2023, 1, 1))
    result = price.scalar_add(5.5)

    assert result.defined is True
    assert result.ccy == Currencies["USD"]
    assert result.qty == Decimal('16.0')
    assert result.dov == date(2023, 1, 1)


# LLM-generated content at query #5
#--------------------------

```python
def test_mul_defined_money_with_scalar():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    money = Money.of(Currencies["USD"], Decimal('10.50'), date(2023, 1, 1))
    result = money * 2
    assert result.qty == Decimal('21.00')
    assert result.ccy == Currencies["USD"]
    assert result.dov == date(2023, 1, 1)

def test_mul_defined_money_with_zero():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    money = Money.of(Currencies["USD"], Decimal('10.50'), date(2023, 1, 1))
    result = money * 0
    assert result.qty == Decimal('0.00')
    assert result.ccy == Currencies["USD"]
    assert result.dov == date(2023, 1, 1)

def test_mul_undefined_money():
    result = Money.na() * 5
    assert result.undefined

def test_mul_defined_money_with_negative_scalar():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    money = Money.of(Currencies["USD"], Decimal('10.50'), date(2023, 1, 1))
    result = money * -1
    assert result.qty == Decimal('-10.50')
    assert result.ccy == Currencies["USD"]
    assert result.dov == date(2023, 1, 1)

def test_mul_defined_money_with_float():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    money = Money.of(Currencies["USD"], Decimal('10.50'), date(2023, 1, 1))
    result = money * 1.5
    assert result.qty == Decimal('15.75')
    assert result.ccy == Currencies["USD"]
    assert result.dov == date(2023, 1, 1)


# LLM-generated content at query #6
#--------------------------

```python
def test_abs_defined_positive_price():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    price = Price.of(Currencies["USD"], Decimal('10.5'), date(2019, 1, 1))
    assert price.abs().qty == Decimal('10.5')

def test_abs_defined_negative_price():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    price = Price.of(Currencies["USD"], Decimal('-10.5'), date(2019, 1, 1))
    assert price.abs().qty == Decimal('10.5')

def test_abs_undefined_price():
    price = Price.na()
    assert price.abs() is price


# LLM-generated content at query #7
#--------------------------

```python
def test_add_defined_prices_with_same_currency():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date

    price1 = Price.of(Currencies["USD"], Decimal('10.50'), date(2023, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('5.25'), date(2023, 1, 2))
    result = price1.add(price2)

    assert result.defined
    assert result.ccy == Currencies["USD"]
    assert result.qty == Decimal('15.75')
    assert result.dov == date(2023, 1, 2)

def test_add_undefined_price_with_defined_price():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date

    price1 = Price.na()
    price2 = Price.of(Currencies["USD"], Decimal('5.25'), date(2023, 1, 2))
    result = price1.add(price2)

    assert result.defined
    assert result.ccy == Currencies["USD"]
    assert result.qty == Decimal('5.25')
    assert result.dov == date(2023, 1, 2)

def test_add_defined_price_with_undefined_price():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date

    price1 = Price.of(Currencies["USD"], Decimal('10.50'), date(2023, 1, 1))
    price2 = Price.na()
    result = price1.add(price2)

    assert result.defined
    assert result.ccy == Currencies["USD"]
    assert result.qty == Decimal('10.50')
    assert result.dov == date(2023, 1, 1)

def test_add_two_undefined_prices():
    price1 = Price.na()
    price2 = Price.na()
    result = price1.add(price2)

    assert result.undefined

def test_add_defined_prices_with_different_currencies_raises_error():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    from pypara.errors import IncompatibleCurrencyError

    price1 = Price.of(Currencies["USD"], Decimal('10.50'), date(2023, 1, 1))
    price2 = Price.of(Currencies["EUR"], Decimal('5.25'), date(2023, 1, 2))

    try:
        result = price1.add(price2)
        assert False, "Expected IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass


# LLM-generated content at query #8
#--------------------------

```python
def test_subtract_defined_money():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    money1 = Money.of(Currencies["USD"], Decimal('10.50'), date(2023, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('5.25'), date(2023, 1, 2))
    result = money1.subtract(money2)
    assert result.ccy == Currencies["USD"]
    assert result.qty == Decimal('5.25')
    assert result.dov == date(2023, 1, 1)

def test_subtract_undefined_money():
    money1 = Money.na()
    money2 = Money.of(Currencies["USD"], Decimal('5.25'), date(2023, 1, 2))
    result = money1.subtract(money2)
    assert result is money2

def test_subtract_with_incompatible_currency():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    from pypara.errors import IncompatibleCurrencyError
    money1 = Money.of(Currencies["USD"], Decimal('10.50'), date(2023, 1, 1))
    money2 = Money.of(Currencies["EUR"], Decimal('5.25'), date(2023, 1, 2))
    try:
        money1.subtract(money2)
        assert False, "Expected IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass


# LLM-generated content at query #9
#--------------------------

```python
def test_truediv_valid_division():
    price = SomePrice(Currency("USD"), Decimal("10.0"), Date(2023, 1, 1))
    result = price / 2
    assert result == SomePrice(Currency("USD"), Decimal("5.0"), Date(2023, 1, 1))

def test_truediv_division_by_zero():
    price = SomePrice(Currency("USD"), Decimal("10.0"), Date(2023, 1, 1))
    result = price / 0
    assert result == NoPrice

def test_truediv_division_by_invalid_operand():
    price = SomePrice(Currency("USD"), Decimal("10.0"), Date(2023, 1, 1))
    result = price / Decimal("NaN")
    assert result == NoPrice


# LLM-generated content at query #10
#--------------------------

```python
def test_price_add_defined_prices_same_currency():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date

    price1 = Price.of(Currencies["USD"], Decimal('10.50'), date(2023, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('5.25'), date(2023, 1, 2))
    result = price1 + price2

    assert result.defined
    assert result.ccy == Currencies["USD"]
    assert result.qty == Decimal('15.75')
    assert result.dov == date(2023, 1, 2)

def test_price_add_undefined_and_defined():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date

    price1 = Price.na()
    price2 = Price.of(Currencies["USD"], Decimal('5.25'), date(2023, 1, 2))
    result = price1 + price2

    assert result is price2

def test_price_add_defined_and_undefined():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date

    price1 = Price.of(Currencies["USD"], Decimal('10.50'), date(2023, 1, 1))
    price2 = Price.na()
    result = price1 + price2

    assert result is price1

def test_price_add_both_undefined():
    price1 = Price.na()
    price2 = Price.na()
    result = price1 + price2

    assert result.undefined


# LLM-generated content at query #11
#--------------------------

```python
def test_somemoney_constructor():
    ccy = Currency("USD")
    qty = Decimal("100.50")
    dov = Date(2023, 1, 1)
    money = SomeMoney(ccy, qty, dov)
    assert money.ccy == ccy
    assert money.qty == qty
    assert money.dov == dov
    assert money.defined is True
    assert money.undefined is False


# LLM-generated content at query #12
#--------------------------

```python
def test_or_else_returns_itself_when_defined():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    price = Price.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    fallback = Price.of(Currencies["EUR"], Decimal('2'), date(2019, 1, 2))
    assert price.or_else(lambda: fallback) is price

def test_or_else_returns_fallback_when_undefined():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    fallback = Price.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    undefined_price = Price.na()
    assert undefined_price.or_else(lambda: fallback) is fallback


# LLM-generated content at query #13
#--------------------------

```python
def test_somemoney_constructor():
    ccy = Currency("USD")
    qty = Decimal("100.50")
    dov = Date(2023, 1, 1)
    money = SomeMoney(ccy, qty, dov)
    assert money.ccy == ccy
    assert money.qty == qty
    assert money.dov == dov
    assert money.defined is True
    assert money.undefined is False


# LLM-generated content at query #14
#--------------------------

```python
def test_qty_map_defined_money():
    from pypara.currencies import Currencies
    somemoney = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    result = somemoney.qty_map(lambda x: x + Decimal('1'), lambda: Decimal('42'))
    assert result == Decimal('2.00')

def test_qty_map_undefined_money():
    nonemoney = Money.of(None, Decimal('1'), None)
    result = nonemoney.qty_map(lambda x: x + Decimal('1'), lambda: Decimal('42'))
    assert result == Decimal('42')


# LLM-generated content at query #15
#--------------------------

```python
def test_fmap_defined_money():
    from pypara.currencies import Currencies
    from datetime import timedelta
    somemoney = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    new = somemoney.fmap(lambda x: Money.of(x.ccy, x.qty + Decimal('1'), x.dov + timedelta(days=10)))
    assert new.ccy.code == 'USD'
    assert new.qty == Decimal('2.00')
    assert new.dov == Date(2019, 1, 11)

def test_fmap_undefined_money():
    nonemoney = Money.of(None, Decimal('1'), None)
    assert nonemoney.fmap(lambda sm: Money.of(sm.ccy, sm.qty + Decimal('1'), sm.dov)) is Money.na()


# LLM-generated content at query #16
#--------------------------

```python
def test_qty_or_defined_price():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    price = Price.of(Currencies["USD"], Decimal('10.5'), date(2023, 1, 1))
    assert price.qty_or(Decimal('0')) == Decimal('10.5')

def test_qty_or_undefined_price():
    price = Price.na()
    assert price.qty_or(Decimal('0')) == Decimal('0')


# LLM-generated content at query #17
#--------------------------

```python
def test_float_defined_price():
    from pypara.currencies import Currencies
    price = Price.of(Currencies["USD"], Decimal('1.5'), Date(2019, 1, 1))
    assert float(price) == 1.5


# LLM-generated content at query #18
#--------------------------

```python
def test_add_defined_money_objects():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date

    usd_money1 = Money.of(Currencies["USD"], Decimal('10.50'), date(2023, 1, 1))
    usd_money2 = Money.of(Currencies["USD"], Decimal('5.25'), date(2023, 1, 2))
    result = usd_money1 + usd_money2

    assert result.ccy == Currencies["USD"]
    assert result.qty == Decimal('15.75')
    assert result.dov == date(2023, 1, 2)
    assert result.defined

def test_add_undefined_and_defined_money_objects():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date

    undefined_money = Money.na()
    usd_money = Money.of(Currencies["USD"], Decimal('10.50'), date(2023, 1, 1))

    result1 = undefined_money + usd_money
    result2 = usd_money + undefined_money

    assert result1 is usd_money
    assert result2 is usd_money

def test_add_undefined_money_objects():
    undefined_money1 = Money.na()
    undefined_money2 = Money.na()

    result = undefined_money1 + undefined_money2

    assert result is undefined_money1
    assert result.undefined

def test_add_incompatible_currencies():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    from pypara.errors import IncompatibleCurrencyError

    usd_money = Money.of(Currencies["USD"], Decimal('10.50'), date(2023, 1, 1))
    eur_money = Money.of(Currencies["EUR"], Decimal('5.25'), date(2023, 1, 2))

    try:
        result = usd_money + eur_money
        assert False, "Expected IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass


# LLM-generated content at query #19
#--------------------------

```python
def test_convert_with_valid_fx_rate():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    money = SomeMoney(usd, Decimal("100.00"), Date(2023, 1, 1))
    FXRateService.default = MockFXRateService({(usd, eur, Date(2023, 1, 1)): Decimal("0.85")})
    result = money.convert(eur, Date(2023, 1, 1))
    assert result == SomeMoney(eur, Decimal("85.00"), Date(2023, 1, 1))

def test_convert_with_no_fx_rate_non_strict():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    money = SomeMoney(usd, Decimal("100.00"), Date(2023, 1, 1))
    FXRateService.default = MockFXRateService({})
    result = money.convert(eur, Date(2023, 1, 1), strict=False)
    assert result == NoMoney

def test_convert_with_no_fx_rate_strict():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    money = SomeMoney(usd, Decimal("100.00"), Date(2023, 1, 1))
    FXRateService.default = MockFXRateService({})
    try:
        money.convert(eur, Date(2023, 1, 1), strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError:
        pass

def test_convert_with_no_fx_service():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    money = SomeMoney(usd, Decimal("100.00"), Date(2023, 1, 1))
    FXRateService.default = None
    try:
        money.convert(eur, Date(2023, 1, 1))
        assert False, "Expected ProgrammingError"
    except ProgrammingError as e:
        assert str(e) == "Did you implement and set the default FX rate service?"

def test_convert_with_different_decimals():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    money = SomeMoney(usd, Decimal("100.50"), Date(2023, 1, 1))
    FXRateService.default = MockFXRateService({(usd, jpy, Date(2023, 1, 1)): Decimal("110.25")})
    result = money.convert(jpy, Date(2023, 1, 1))
    assert result == SomeMoney(jpy, Decimal("11073"), Date(2023, 1, 1))

def test_convert_with_default_date():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    money = SomeMoney(usd, Decimal("100.00"), Date(2023, 1, 1))
    FXRateService.default = MockFXRateService({(usd, eur, Date(2023, 1, 1)): Decimal("0.85")})
    result = money.convert(eur)
    assert result == SomeMoney(eur, Decimal("85.00"), Date(2023, 1, 1))


# LLM-generated content at query #20
#--------------------------

```python
def test_someprice_add_with_same_currency():
    price1 = SomePrice(Currency("USD"), Decimal("10.50"), Date(2023, 1, 1))
    price2 = SomePrice(Currency("USD"), Decimal("5.25"), Date(2023, 1, 2))
    result = price1 + price2
    assert result == SomePrice(Currency("USD"), Decimal("15.75"), Date(2023, 1, 1))

def test_someprice_add_with_different_currency():
    price1 = SomePrice(Currency("USD"), Decimal("10.50"), Date(2023, 1, 1))
    price2 = SomePrice(Currency("EUR"), Decimal("5.25"), Date(2023, 1, 2))
    try:
        result = price1 + price2
        assert False, "Expected IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass

def test_someprice_add_with_undefined_price():
    price1 = SomePrice(Currency("USD"), Decimal("10.50"), Date(2023, 1, 1))
    price2 = NoPrice()
    result = price1 + price2
    assert result == price1


# LLM-generated content at query #21
#--------------------------

```python
def test_defined_price_with_nonzero_quantity_is_truthy():
    from pypara.currencies import Currencies
    price = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert bool(price) is True

def test_defined_price_with_zero_quantity_is_falsy():
    from pypara.currencies import Currencies
    price = Price.of(Currencies["USD"], Decimal('0'), Date(2019, 1, 1))
    assert bool(price) is False

def test_undefined_price_is_falsy():
    price = Price.na()
    assert bool(price) is False


# LLM-generated content at query #22
#--------------------------

```python
def test_with_dov_defined_money():
    from pypara.currencies import Currencies
    from datetime import date
    from decimal import Decimal

    original_money = Money.of(Currencies["USD"], Decimal('100.50'), date(2023, 1, 1))
    new_dov = date(2023, 12, 31)
    result = original_money.with_dov(new_dov)

    assert result.defined
    assert result.ccy == Currencies["USD"]
    assert result.qty == Decimal('100.50')
    assert result.dov == new_dov

def test_with_dov_undefined_money():
    undefined_money = Money.na()
    new_dov = date(2023, 12, 31)
    result = undefined_money.with_dov(new_dov)

    assert result.undefined
    assert result is undefined_money


# LLM-generated content at query #23
#--------------------------

```python
def test_floor_divide_defined_money():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    money = Money.of(Currencies["USD"], Decimal('10'), date(2019, 1, 1))
    result = money // Decimal('3')
    assert result.qty == Decimal('3')
    assert result.ccy == Currencies["USD"]
    assert result.dov == date(2019, 1, 1)

def test_floor_divide_undefined_money():
    money = Money.na()
    result = money // Decimal('3')
    assert result.undefined

def test_floor_divide_by_zero():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    money = Money.of(Currencies["USD"], Decimal('10'), date(2019, 1, 1))
    result = money // Decimal('0')
    assert result.undefined


# LLM-generated content at query #24
#--------------------------

```python
def test_with_qty_defined_money():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    money = Money.of(Currencies["USD"], Decimal('10.50'), date(2023, 1, 1))
    new_money = money.with_qty(Decimal('20.75'))
    assert new_money.qty == Decimal('20.75')
    assert new_money.ccy == Currencies["USD"]
    assert new_money.dov == date(2023, 1, 1)

def test_with_qty_undefined_money():
    undefined_money = Money.na()
    new_money = undefined_money.with_qty(Decimal('20.75'))
    assert new_money.undefined


# LLM-generated content at query #25
#--------------------------

```python
def test_none_money_constructor():
    none_money = NoneMoney()
    assert none_money.defined == False
    assert none_money.undefined == True
    assert none_money.as_boolean() == False
    assert none_money.is_equal(NoneMoney()) == True
    assert none_money.abs() is none_money
    assert none_money.negative() is none_money
    assert none_money.positive() is none_money
    assert none_money.price is NoPrice


# LLM-generated content at query #26
#--------------------------

```python
def test_multiply_defined_money_by_numeric():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date

    money = Money.of(Currencies["USD"], Decimal('10.50'), date(2023, 1, 1))
    result = money.multiply(2)
    assert result.qty == Decimal('21.00')
    assert result.ccy == Currencies["USD"]
    assert result.dov == date(2023, 1, 1)

def test_multiply_undefined_money_by_numeric():
    undefined_money = Money.na()
    result = undefined_money.multiply(5)
    assert result.undefined

def test_multiply_defined_money_by_zero():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date

    money = Money.of(Currencies["USD"], Decimal('100.00'), date(2023, 1, 1))
    result = money.multiply(0)
    assert result.qty == Decimal('0.00')
    assert result.ccy == Currencies["USD"]
    assert result.dov == date(2023, 1, 1)

def test_multiply_defined_money_by_negative_numeric():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date

    money = Money.of(Currencies["USD"], Decimal('10.50'), date(2023, 1, 1))
    result = money.multiply(-3)
    assert result.qty == Decimal('-31.50')
    assert result.ccy == Currencies["USD"]
    assert result.dov == date(2023, 1, 1)

def test_multiply_defined_money_by_fractional_numeric():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date

    money = Money.of(Currencies["USD"], Decimal('10.00'), date(2023, 1, 1))
    result = money.multiply(Decimal('0.5'))
    assert result.qty == Decimal('5.00')
    assert result.ccy == Currencies["USD"]
    assert result.dov == date(2023, 1, 1)


# LLM-generated content at query #27
#--------------------------

```python
def test_sub_same_currency_and_different_dov():
    price1 = SomePrice(Currency("USD"), Decimal("10.5"), Date(2023, 1, 1))
    price2 = SomePrice(Currency("USD"), Decimal("5.25"), Date(2023, 1, 2))
    result = price1 - price2
    assert result == SomePrice(Currency("USD"), Decimal("5.25"), Date(2023, 1, 1))

def test_sub_same_currency_and_same_dov():
    price1 = SomePrice(Currency("USD"), Decimal("10.5"), Date(2023, 1, 1))
    price2 = SomePrice(Currency("USD"), Decimal("5.25"), Date(2023, 1, 1))
    result = price1 - price2
    assert result == SomePrice(Currency("USD"), Decimal("5.25"), Date(2023, 1, 1))

def test_sub_different_currency_raises_error():
    price1 = SomePrice(Currency("USD"), Decimal("10.5"), Date(2023, 1, 1))
    price2 = SomePrice(Currency("EUR"), Decimal("5.25"), Date(2023, 1, 2))
    try:
        result = price1 - price2
        assert False, "Expected IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass

def test_sub_undefined_price_returns_self():
    price1 = SomePrice(Currency("USD"), Decimal("10.5"), Date(2023, 1, 1))
    price2 = NoPrice()
    result = price1 - price2
    assert result == price1


# LLM-generated content at query #28
#--------------------------

```python
def test_add_same_currency():
    usd = Currency("USD")
    money1 = SomeMoney(usd, Decimal("10.50"), Date(2023, 1, 1))
    money2 = SomeMoney(usd, Decimal("5.25"), Date(2023, 1, 2))
    result = money1 + money2
    assert result == SomeMoney(usd, Decimal("15.75"), Date(2023, 1, 1))

def test_add_different_currency_raises_error():
    usd = Currency("USD")
    eur = Currency("EUR")
    money1 = SomeMoney(usd, Decimal("10.50"), Date(2023, 1, 1))
    money2 = SomeMoney(eur, Decimal("5.25"), Date(2023, 1, 2))
    try:
        result = money1 + money2
        assert False, "Expected IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass

def test_add_with_undefined_money():
    usd = Currency("USD")
    money1 = SomeMoney(usd, Decimal("10.50"), Date(2023, 1, 1))
    money2 = NoMoney
    result = money1 + money2
    assert result == money1


# LLM-generated content at query #29
#--------------------------

```python
def test_some_money_le_with_same_currency_and_smaller_quantity():
    usd = Currency("USD", 2)
    money1 = SomeMoney(usd, Decimal("10.00"), Date(2023, 1, 1))
    money2 = SomeMoney(usd, Decimal("20.00"), Date(2023, 1, 1))
    assert money1 <= money2

def test_some_money_le_with_same_currency_and_equal_quantity():
    usd = Currency("USD", 2)
    money1 = SomeMoney(usd, Decimal("10.00"), Date(2023, 1, 1))
    money2 = SomeMoney(usd, Decimal("10.00"), Date(2023, 1, 1))
    assert money1 <= money2

def test_some_money_le_with_same_currency_and_larger_quantity():
    usd = Currency("USD", 2)
    money1 = SomeMoney(usd, Decimal("20.00"), Date(2023, 1, 1))
    money2 = SomeMoney(usd, Decimal("10.00"), Date(2023, 1, 1))
    assert not (money1 <= money2)

def test_some_money_le_with_different_currency():
    usd = Currency("USD", 2)
    eur = Currency("EUR", 2)
    money1 = SomeMoney(usd, Decimal("10.00"), Date(2023, 1, 1))
    money2 = SomeMoney(eur, Decimal("10.00"), Date(2023, 1, 1))
    try:
        money1 <= money2
        assert False, "Expected IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass

def test_some_money_le_with_non_money_object():
    usd = Currency("USD", 2)
    money = SomeMoney(usd, Decimal("10.00"), Date(2023, 1, 1))
    assert not (money <= "not a money object")


# LLM-generated content at query #30
#--------------------------

```python
def test_subtract_defined_prices_same_currency():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    price1 = Price.of(Currencies["USD"], Decimal('10.50'), date(2023, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('5.25'), date(2023, 1, 2))
    result = price1.subtract(price2)
    assert result.qty == Decimal('5.25')
    assert result.ccy == Currencies["USD"]
    assert result.dov == date(2023, 1, 1)

def test_subtract_defined_prices_different_currency():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    from pypara.errors import IncompatibleCurrencyError
    price1 = Price.of(Currencies["USD"], Decimal('10.50'), date(2023, 1, 1))
    price2 = Price.of(Currencies["EUR"], Decimal('5.25'), date(2023, 1, 2))
    try:
        price1.subtract(price2)
        assert False, "Expected IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass

def test_subtract_undefined_price_with_defined():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    price1 = Price.na()
    price2 = Price.of(Currencies["USD"], Decimal('5.25'), date(2023, 1, 2))
    result = price1.subtract(price2)
    assert result.undefined

def test_subtract_defined_price_with_undefined():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    price1 = Price.of(Currencies["USD"], Decimal('10.50'), date(2023, 1, 1))
    price2 = Price.na()
    result = price1.subtract(price2)
    assert result.undefined


# LLM-generated content at query #31
#--------------------------

```python
def test_convert_with_valid_fx_rate():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    price = SomePrice(usd, Decimal("100.00"), Date(2023, 1, 1))
    FXRateService.default = MockFXRateService({FXRate(usd, eur, Decimal("0.90"), Date(2023, 1, 1))})
    converted_price = price.convert(eur, Date(2023, 1, 1))
    assert converted_price == SomePrice(eur, Decimal("90.00"), Date(2023, 1, 1))

def test_convert_with_no_fx_rate_non_strict():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    price = SomePrice(usd, Decimal("100.00"), Date(2023, 1, 1))
    FXRateService.default = MockFXRateService({})
    converted_price = price.convert(eur, Date(2023, 1, 1), strict=False)
    assert converted_price == NoPrice

def test_convert_with_no_fx_rate_strict():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    price = SomePrice(usd, Decimal("100.00"), Date(2023, 1, 1))
    FXRateService.default = MockFXRateService({})
    try:
        price.convert(eur, Date(2023, 1, 1), strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError:
        pass

def test_convert_with_no_fx_service():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    price = SomePrice(usd, Decimal("100.00"), Date(2023, 1, 1))
    FXRateService.default = None
    try:
        price.convert(eur, Date(2023, 1, 1))
        assert False, "Expected ProgrammingError"
    except ProgrammingError:
        pass

def test_convert_with_default_date():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    price = SomePrice(usd, Decimal("100.00"), Date(2023, 1, 1))
    FXRateService.default = MockFXRateService({FXRate(usd, eur, Decimal("0.90"), Date(2023, 1, 1))})
    converted_price = price.convert(eur)
    assert converted_price == SomePrice(eur, Decimal("90.00"), Date(2023, 1, 1))


# LLM-generated content at query #32
#--------------------------

```python
def test_subtract_with_undefined_other_returns_self():
    price = SomePrice(Currency("USD"), Decimal("10.0"), Date(2023, 1, 1))
    result = price.subtract(NoPrice)
    assert result is price


# LLM-generated content at query #33
#--------------------------

```python
def test_add_with_undefined_price():
    usd = Currency("USD")
    price = SomePrice(usd, Decimal("10.50"), Date(2023, 1, 1))
    undefined_price = NoPrice()
    assert price.add(undefined_price) == price


# LLM-generated content at query #34
#--------------------------

```python
def test_someprice_ge_with_same_currency():
    price1 = SomePrice(Currency("USD"), Decimal("10.50"), Date(2023, 1, 1))
    price2 = SomePrice(Currency("USD"), Decimal("10.00"), Date(2023, 1, 1))
    assert price1 >= price2

def test_someprice_ge_with_different_currency():
    price1 = SomePrice(Currency("USD"), Decimal("10.00"), Date(2023, 1, 1))
    price2 = SomePrice(Currency("EUR"), Decimal("10.00"), Date(2023, 1, 1))
    try:
        price1 >= price2
        assert False, "Expected IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass

def test_someprice_ge_with_none_price():
    price1 = SomePrice(Currency("USD"), Decimal("10.00"), Date(2023, 1, 1))
    assert price1 >= None

def test_someprice_ge_with_equal_price():
    price1 = SomePrice(Currency("USD"), Decimal("10.00"), Date(2023, 1, 1))
    price2 = SomePrice(Currency("USD"), Decimal("10.00"), Date(2023, 1, 1))
    assert price1 >= price2

def test_someprice_ge_with_less_price():
    price1 = SomePrice(Currency("USD"), Decimal("9.00"), Date(2023, 1, 1))
    price2 = SomePrice(Currency("USD"), Decimal("10.00"), Date(2023, 1, 1))
    assert not (price1 >= price2)


# LLM-generated content at query #35
#--------------------------

```python
def test_round_defined_money():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    money = Money.of(Currencies["USD"], Decimal('3.14159'), date(2019, 1, 1))
    assert money.round(2).qty == Decimal('3.14')

def test_round_undefined_money():
    money = Money.na()
    assert money.round(2) is money


