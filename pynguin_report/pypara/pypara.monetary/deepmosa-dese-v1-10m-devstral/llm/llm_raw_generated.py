####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_qty_or_none_defined_price():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    price = Price.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    assert price.qty_or_none() == Decimal('1')

def test_qty_or_none_undefined_price():
    price = Price.na()
    assert price.qty_or_none() is None


# LLM-generated content at query #2
#--------------------------

```python
def test_round_positive_ndigits():
    price = SomePrice(Currency("USD"), Decimal("123.456"), Date(2023, 1, 1))
    assert price.round(2) == SomePrice(Currency("USD"), Decimal("123.46"), Date(2023, 1, 1))

def test_round_negative_ndigits():
    price = SomePrice(Currency("USD"), Decimal("123.456"), Date(2023, 1, 1))
    assert price.round(-1) == SomePrice(Currency("USD"), Decimal("120"), Date(2023, 1, 1))

def test_round_zero_ndigits():
    price = SomePrice(Currency("USD"), Decimal("123.456"), Date(2023, 1, 1))
    assert price.round() == SomePrice(Currency("USD"), Decimal("123"), Date(2023, 1, 1))

def test_round_negative_price():
    price = SomePrice(Currency("USD"), Decimal("-123.456"), Date(2023, 1, 1))
    assert price.round(2) == SomePrice(Currency("USD"), Decimal("-123.46"), Date(2023, 1, 1))


# LLM-generated content at query #3
#--------------------------

```python
def test_as_boolean_defined_nonzero():
    from pypara.currencies import Currencies
    price = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert price.as_boolean() is True

def test_as_boolean_defined_zero():
    from pypara.currencies import Currencies
    price = Price.of(Currencies["USD"], Decimal('0'), Date(2019, 1, 1))
    assert price.as_boolean() is False

def test_as_boolean_undefined():
    price = Price.na()
    assert price.as_boolean() is False


# LLM-generated content at query #4
#--------------------------

```python
def test_qty_or_none_returns_qty():
    price = SomePrice(Currency("USD"), Decimal("100.50"), Date(2023, 1, 1))
    assert price.qty_or_none() == Decimal("100.50")


# LLM-generated content at query #5
#--------------------------

```python
def test_gte_defined_greater_than_defined():
    from pypara.currencies import Currencies
    price1 = Price.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 2))
    assert price1.gte(price2) is True

def test_gte_defined_equal_to_defined():
    from pypara.currencies import Currencies
    price1 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 2))
    assert price1.gte(price2) is True

def test_gte_defined_less_than_defined():
    from pypara.currencies import Currencies
    price1 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 2))
    assert price1.gte(price2) is False

def test_gte_undefined_greater_than_undefined():
    price1 = Price.na()
    price2 = Price.na()
    assert price1.gte(price2) is True

def test_gte_undefined_less_than_defined():
    from pypara.currencies import Currencies
    price1 = Price.na()
    price2 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert price1.gte(price2) is False

def test_gte_defined_greater_than_undefined():
    from pypara.currencies import Currencies
    price1 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    price2 = Price.na()
    assert price1.gte(price2) is True

def test_gte_incompatible_currency_error():
    from pypara.currencies import Currencies
    price1 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["EUR"], Decimal('1'), Date(2019, 1, 2))
    try:
        price1.gte(price2)
        assert False, "Expected IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass


# LLM-generated content at query #6
#--------------------------

```python
def test_round_defined_money():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    money = Money.of(Currencies["USD"], Decimal('1.234'), date(2019, 1, 1))
    assert money.round(2).qty == Decimal('1.23')
    assert money.round(1).qty == Decimal('1.2')
    assert money.round(0).qty == Decimal('1')
    assert money.round().qty == Decimal('1')

def test_round_undefined_money():
    undefined_money = Money.na()
    assert undefined_money.round(2) is undefined_money
    assert undefined_money.round() is undefined_money


# LLM-generated content at query #7
#--------------------------

```python
def test_ge_with_same_currency_and_greater_quantity():
    usd = Currency("USD", 2)
    money1 = SomeMoney(usd, Decimal("100.50"), Date(2023, 1, 1))
    money2 = SomeMoney(usd, Decimal("50.25"), Date(2023, 1, 1))
    assert money1 >= money2

def test_ge_with_same_currency_and_equal_quantity():
    usd = Currency("USD", 2)
    money1 = SomeMoney(usd, Decimal("75.00"), Date(2023, 1, 1))
    money2 = SomeMoney(usd, Decimal("75.00"), Date(2023, 1, 1))
    assert money1 >= money2

def test_ge_with_same_currency_and_lesser_quantity():
    usd = Currency("USD", 2)
    money1 = SomeMoney(usd, Decimal("25.00"), Date(2023, 1, 1))
    money2 = SomeMoney(usd, Decimal("50.25"), Date(2023, 1, 1))
    assert not (money1 >= money2)

def test_ge_with_different_currency():
    usd = Currency("USD", 2)
    eur = Currency("EUR", 2)
    money1 = SomeMoney(usd, Decimal("100.00"), Date(2023, 1, 1))
    money2 = SomeMoney(eur, Decimal("50.00"), Date(2023, 1, 1))
    try:
        money1 >= money2
        assert False, "Expected IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass

def test_ge_with_non_money_object():
    usd = Currency("USD", 2)
    money = SomeMoney(usd, Decimal("100.00"), Date(2023, 1, 1))
    assert money >= "not a money object"


# LLM-generated content at query #8
#--------------------------

```python
def test_round_defined_price():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    price = Price.of(Currencies["USD"], Decimal('3.14159'), date(2019, 1, 1))
    assert price.round(2).qty == Decimal('3.14')

def test_round_undefined_price():
    price = Price.na()
    assert price.round(2) is price


# LLM-generated content at query #9
#--------------------------

```python
def test_qty_or_zero_defined_money():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    money = Money.of(Currencies["USD"], Decimal('1.50'), date(2019, 1, 1))
    assert money.qty_or_zero() == Decimal('1.50')

def test_qty_or_zero_undefined_money():
    money = Money.na()
    assert money.qty_or_zero() == Decimal('0')


# LLM-generated content at query #10
#--------------------------

```python
def test_none_money_constructor():
    none_money = NoneMoney()
    assert isinstance(none_money, NoneMoney)
    assert not none_money.defined
    assert none_money.undefined


# LLM-generated content at query #11
#--------------------------

```python
def test_price_is_equal_returns_true_for_same_price_objects():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    price1 = Price.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    assert price1.is_equal(price2)

def test_price_is_equal_returns_false_for_different_price_objects():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    price1 = Price.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    price2 = Price.of(Currencies["EUR"], Decimal('1'), date(2019, 1, 1))
    assert not price1.is_equal(price2)

def test_price_is_equal_returns_false_for_undefined_price():
    price1 = Price.na()
    price2 = Price.of(None, Decimal('1'), None)
    assert not price1.is_equal(price2)

def test_price_is_equal_returns_true_for_same_undefined_price():
    price1 = Price.na()
    price2 = Price.na()
    assert price1.is_equal(price2)


# LLM-generated content at query #12
#--------------------------

```python
def test_subtract_defined_minus_defined():
    from pypara.currencies import Currencies
    from datetime import date
    from decimal import Decimal
    money1 = Money.of(Currencies["USD"], Decimal('10.00'), date(2023, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('5.00'), date(2023, 1, 2))
    result = money1.subtract(money2)
    assert result.defined
    assert result.ccy == Currencies["USD"]
    assert result.qty == Decimal('5.00')
    assert result.dov == date(2023, 1, 2)

def test_subtract_defined_minus_undefined():
    from pypara.currencies import Currencies
    from datetime import date
    from decimal import Decimal
    money1 = Money.of(Currencies["USD"], Decimal('10.00'), date(2023, 1, 1))
    money2 = Money.na()
    result = money1.subtract(money2)
    assert result.defined
    assert result.ccy == Currencies["USD"]
    assert result.qty == Decimal('10.00')
    assert result.dov == date(2023, 1, 1)

def test_subtract_undefined_minus_defined():
    from pypara.currencies import Currencies
    from datetime import date
    from decimal import Decimal
    money1 = Money.na()
    money2 = Money.of(Currencies["USD"], Decimal('5.00'), date(2023, 1, 2))
    result = money1.subtract(money2)
    assert result.undefined

def test_subtract_undefined_minus_undefined():
    money1 = Money.na()
    money2 = Money.na()
    result = money1.subtract(money2)
    assert result.undefined

def test_subtract_incompatible_currency():
    from pypara.currencies import Currencies
    from datetime import date
    from decimal import Decimal
    from pypara.errors import IncompatibleCurrencyError
    money1 = Money.of(Currencies["USD"], Decimal('10.00'), date(2023, 1, 1))
    money2 = Money.of(Currencies["EUR"], Decimal('5.00'), date(2023, 1, 2))
    try:
        money1.subtract(money2)
        assert False, "Expected IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass


# LLM-generated content at query #13
#--------------------------

```python
def test_gt_undefined_vs_defined():
    undefined_price = Price.na()
    defined_price = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert not undefined_price.gt(defined_price)

def test_gt_defined_vs_undefined():
    defined_price = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    undefined_price = Price.na()
    assert defined_price.gt(undefined_price)

def test_gt_same_currency_defined():
    price1 = Price.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert price1.gt(price2)

def test_gt_same_currency_not_greater():
    price1 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    assert not price1.gt(price2)

def test_gt_different_currency_raises_error():
    price1 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["EUR"], Decimal('1'), Date(2019, 1, 1))
    try:
        price1.gt(price2)
        assert False, "Expected IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass


# LLM-generated content at query #14
#--------------------------

```python
def test_is_equal_defined_money_same_values():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    money1 = Money.of(Currencies["USD"], Decimal('100.00'), date(2023, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('100.00'), date(2023, 1, 1))
    assert money1.is_equal(money2) is True

def test_is_equal_defined_money_different_values():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    money1 = Money.of(Currencies["USD"], Decimal('100.00'), date(2023, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('200.00'), date(2023, 1, 1))
    assert money1.is_equal(money2) is False

def test_is_equal_undefined_money():
    money1 = Money.na()
    money2 = Money.na()
    assert money1.is_equal(money2) is True

def test_is_equal_defined_vs_undefined():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    money1 = Money.of(Currencies["USD"], Decimal('100.00'), date(2023, 1, 1))
    money2 = Money.na()
    assert money1.is_equal(money2) is False

def test_is_equal_non_money_object():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    money = Money.of(Currencies["USD"], Decimal('100.00'), date(2023, 1, 1))
    assert money.is_equal("not a money object") is False


# LLM-generated content at query #15
#--------------------------

```python
def test_dov_or_none_returns_dov():
    price = SomePrice(Currency("USD"), Decimal("10.5"), Date(2023, 1, 1))
    assert price.dov_or_none() == Date(2023, 1, 1)


# LLM-generated content at query #16
#--------------------------

```python
def test_convert_same_currency():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    money = SomeMoney(usd, Decimal("100.00"), Date(2023, 1, 1))
    result = money.convert(usd)
    assert result == SomeMoney(usd, Decimal("100.00"), Date(2023, 1, 1))

def test_convert_different_currency_with_rate():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    money = SomeMoney(usd, Decimal("100.00"), Date(2023, 1, 1))
    FXRateService.default = MockFXRateService({(usd, eur, Date(2023, 1, 1)): Decimal("0.85")})
    result = money.convert(eur)
    assert result == SomeMoney(eur, Decimal("85.00"), Date(2023, 1, 1))

def test_convert_different_currency_no_rate_non_strict():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    money = SomeMoney(usd, Decimal("100.00"), Date(2023, 1, 1))
    FXRateService.default = MockFXRateService({})
    result = money.convert(eur, strict=False)
    assert result == NoMoney

def test_convert_different_currency_no_rate_strict():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    money = SomeMoney(usd, Decimal("100.00"), Date(2023, 1, 1))
    FXRateService.default = MockFXRateService({})
    try:
        money.convert(eur, strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError:
        pass

def test_convert_with_custom_asof_date():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    money = SomeMoney(usd, Decimal("100.00"), Date(2023, 1, 1))
    FXRateService.default = MockFXRateService({(usd, eur, Date(2023, 1, 2)): Decimal("0.90")})
    result = money.convert(eur, asof=Date(2023, 1, 2))
    assert result == SomeMoney(eur, Decimal("90.00"), Date(2023, 1, 2))

def test_convert_no_fx_service_set():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    money = SomeMoney(usd, Decimal("100.00"), Date(2023, 1, 1))
    FXRateService.default = None
    try:
        money.convert(eur)
        assert False, "Expected ProgrammingError"
    except ProgrammingError as e:
        assert str(e) == "Did you implement and set the default FX rate service?"


# LLM-generated content at query #17
#--------------------------

```python
def test_neg_defined_price():
    from pypara.currencies import Currencies
    price = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    neg_price = -price
    assert neg_price.qty == Decimal('-1')
    assert neg_price.ccy == Currencies["USD"]
    assert neg_price.dov == Date(2019, 1, 1)

def test_neg_undefined_price():
    undefined_price = Price.na()
    neg_undefined_price = -undefined_price
    assert neg_undefined_price.undefined
    assert neg_undefined_price is undefined_price


# LLM-generated content at query #18
#--------------------------

```python
def test_or_else_returns_self():
    price = SomePrice(Currency("USD"), Decimal("10.5"), Date(2023, 1, 1))
    result = price.or_else(lambda: SomePrice(Currency("EUR"), Decimal("20.0"), Date(2023, 1, 2)))
    assert result == price


# LLM-generated content at query #19
#--------------------------

```python
def test_qty_or_zero():
    price = SomePrice(Currency("USD"), Decimal("10.5"), Date(2023, 1, 1))
    assert price.qty_or_zero() == Decimal("10.5")


# LLM-generated content at query #20
#--------------------------

```python
def test_qty_map_defined_money():
    from pypara.currencies import Currencies
    somemoney = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert somemoney.qty_map(lambda x: x + Decimal('1'), lambda: Decimal('42')) == Decimal('2.00')

def test_qty_map_undefined_money():
    nonemoney = Money.of(None, Decimal('1'), None)
    assert nonemoney.qty_map(lambda x: x + Decimal('1'), lambda: Decimal('42')) == Decimal('42')


# LLM-generated content at query #21
#--------------------------

```python
def test_add_same_currency():
    price1 = SomePrice(Currency("USD"), Decimal("10.50"), Date(2023, 1, 1))
    price2 = SomePrice(Currency("USD"), Decimal("5.25"), Date(2023, 1, 2))
    result = price1 + price2
    assert result == SomePrice(Currency("USD"), Decimal("15.75"), Date(2023, 1, 1))

def test_add_different_currency_raises_error():
    price1 = SomePrice(Currency("USD"), Decimal("10.50"), Date(2023, 1, 1))
    price2 = SomePrice(Currency("EUR"), Decimal("5.25"), Date(2023, 1, 2))
    try:
        result = price1 + price2
        assert False, "Expected IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass

def test_add_with_undefined_price():
    price1 = SomePrice(Currency("USD"), Decimal("10.50"), Date(2023, 1, 1))
    price2 = NoPrice()
    result = price1 + price2
    assert result == price1


# LLM-generated content at query #22
#--------------------------

```python
def test_subtract_defined_minus_defined():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    usd1 = Money.of(Currencies["USD"], Decimal('10.50'), date(2023, 1, 1))
    usd2 = Money.of(Currencies["USD"], Decimal('3.25'), date(2023, 1, 2))
    result = usd1.subtract(usd2)
    assert result.defined
    assert result.ccy == Currencies["USD"]
    assert result.qty == Decimal('7.25')
    assert result.dov == date(2023, 1, 1)

def test_subtract_defined_minus_undefined():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    usd = Money.of(Currencies["USD"], Decimal('10.50'), date(2023, 1, 1))
    none = Money.na()
    result = usd.subtract(none)
    assert result is usd

def test_subtract_undefined_minus_defined():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    none = Money.na()
    usd = Money.of(Currencies["USD"], Decimal('10.50'), date(2023, 1, 1))
    result = none.subtract(usd)
    assert result is usd

def test_subtract_undefined_minus_undefined():
    none1 = Money.na()
    none2 = Money.na()
    result = none1.subtract(none2)
    assert result.undefined

def test_subtract_incompatible_currency():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    from pypara.errors import IncompatibleCurrencyError
    usd = Money.of(Currencies["USD"], Decimal('10.50'), date(2023, 1, 1))
    eur = Money.of(Currencies["EUR"], Decimal('3.25'), date(2023, 1, 2))
    try:
        usd.subtract(eur)
        assert False, "Expected IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass


# LLM-generated content at query #23
#--------------------------

```python
def test_neg_defined_money():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    money = Money.of(Currencies["USD"], Decimal('10.50'), date(2023, 1, 1))
    neg_money = -money
    assert neg_money.qty == Decimal('-10.50')
    assert neg_money.ccy == Currencies["USD"]
    assert neg_money.dov == date(2023, 1, 1)

def test_neg_undefined_money():
    undefined_money = Money.na()
    neg_undefined_money = -undefined_money
    assert neg_undefined_money.undefined


# LLM-generated content at query #24
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

def test_some_money_le_with_non_some_money_object():
    usd = Currency("USD", 2)
    money = SomeMoney(usd, Decimal("10.00"), Date(2023, 1, 1))
    assert not (money <= "not a money object")


# LLM-generated content at query #25
#--------------------------

```python
def test_none_money_constructor():
    none_money = NoneMoney()
    assert isinstance(none_money, NoneMoney)
    assert not none_money.defined
    assert none_money.undefined


# LLM-generated content at query #26
#--------------------------

```python
def test_subtract_same_currency():
    ccy = Currency("USD", 2)
    qty1 = Decimal("100.50")
    qty2 = Decimal("50.25")
    dov1 = Date(2023, 1, 1)
    dov2 = Date(2023, 1, 2)
    money1 = SomeMoney(ccy, qty1, dov1)
    money2 = SomeMoney(ccy, qty2, dov2)
    result = money1 - money2
    assert result == SomeMoney(ccy, Decimal("50.25"), dov1)

def test_subtract_different_currency_raises_error():
    ccy1 = Currency("USD", 2)
    ccy2 = Currency("EUR", 2)
    qty1 = Decimal("100.50")
    qty2 = Decimal("50.25")
    dov = Date(2023, 1, 1)
    money1 = SomeMoney(ccy1, qty1, dov)
    money2 = SomeMoney(ccy2, qty2, dov)
    try:
        result = money1 - money2
        assert False, "Expected IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass

def test_subtract_undefined_money():
    ccy = Currency("USD", 2)
    qty = Decimal("100.50")
    dov = Date(2023, 1, 1)
    money1 = SomeMoney(ccy, qty, dov)
    money2 = NoMoney
    result = money1 - money2
    assert result == money1


# LLM-generated content at query #27
#--------------------------

```python
def test_scalar_subtract_defined_money():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    money = Money.of(Currencies["USD"], Decimal('10.50'), date(2019, 1, 1))
    result = money.scalar_subtract(Decimal('2.50'))
    assert result.qty == Decimal('8.00')
    assert result.ccy == Currencies["USD"]
    assert result.dov == date(2019, 1, 1)

def test_scalar_subtract_undefined_money():
    money = Money.na()
    result = money.scalar_subtract(Decimal('2.50'))
    assert result is money


# LLM-generated content at query #28
#--------------------------

```python
def test_lt_undefined_less_than_defined():
    assert Price.na().lt(Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))) == True

def test_lt_defined_not_less_than_undefined():
    assert Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1)).lt(Price.na()) == False

def test_lt_defined_less_than_defined():
    assert Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1)).lt(Price.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))) == True

def test_lt_defined_not_less_than_defined():
    assert Price.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1)).lt(Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))) == False

def test_lt_defined_equal_not_less_than():
    assert Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1)).lt(Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))) == False

def test_lt_incompatible_currency_raises_error():
    try:
        Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1)).lt(Price.of(Currencies["EUR"], Decimal('1'), Date(2019, 1, 1)))
        assert False, "Expected IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass


# LLM-generated content at query #29
#--------------------------

```python
def test_le_undefined_money_is_less_than_or_equal_to_defined_money():
    from pypara.currencies import Currencies
    undefined_money = Money.na()
    defined_money = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert undefined_money <= defined_money

def test_le_defined_money_is_not_less_than_or_equal_to_undefined_money():
    from pypara.currencies import Currencies
    defined_money = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    undefined_money = Money.na()
    assert not (defined_money <= undefined_money)

def test_le_defined_money_with_same_currency_and_quantity():
    from pypara.currencies import Currencies
    money1 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 2))
    assert money1 <= money2

def test_le_defined_money_with_same_currency_and_less_quantity():
    from pypara.currencies import Currencies
    money1 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 2))
    assert money1 <= money2

def test_le_defined_money_with_same_currency_and_greater_quantity():
    from pypara.currencies import Currencies
    money1 = Money.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 2))
    assert not (money1 <= money2)

def test_le_defined_money_with_different_currency_raises_error():
    from pypara.currencies import Currencies
    money1 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["EUR"], Decimal('1'), Date(2019, 1, 2))
    try:
        money1 <= money2
        assert False, "Expected IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass


# LLM-generated content at query #30
#--------------------------

```python
def test_add_same_currency():
    usd = Currency("USD", 2)
    money1 = SomeMoney(usd, Decimal("100.50"), Date(2023, 1, 1))
    money2 = SomeMoney(usd, Decimal("50.25"), Date(2023, 1, 2))
    result = money1 + money2
    assert result == SomeMoney(usd, Decimal("150.75"), Date(2023, 1, 1))

def test_add_different_currency_raises_error():
    usd = Currency("USD", 2)
    eur = Currency("EUR", 2)
    money1 = SomeMoney(usd, Decimal("100.50"), Date(2023, 1, 1))
    money2 = SomeMoney(eur, Decimal("50.25"), Date(2023, 1, 2))
    try:
        result = money1 + money2
        assert False, "Expected IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass

def test_add_with_undefined_money():
    usd = Currency("USD", 2)
    money1 = SomeMoney(usd, Decimal("100.50"), Date(2023, 1, 1))
    money2 = NoMoney
    result = money1 + money2
    assert result == money1


# LLM-generated content at query #31
#--------------------------

```python
def test_as_integer_defined_price():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    price = Price.of(Currencies["USD"], Decimal('10.5'), date(2019, 1, 1))
    assert price.as_integer() == 10

def test_as_integer_undefined_price_raises_exception():
    undefined_price = Price.na()
    try:
        undefined_price.as_integer()
        assert False, "Expected MonetaryOperationException"
    except MonetaryOperationException:
        pass


# LLM-generated content at query #32
#--------------------------

```python
def test_add_defined_money_objects():
    from pypara.currencies import Currencies
    from datetime import date
    from decimal import Decimal

    usd1 = Money.of(Currencies["USD"], Decimal('1.00'), date(2019, 1, 1))
    usd2 = Money.of(Currencies["USD"], Decimal('2.00'), date(2019, 1, 2))
    result = usd1 + usd2
    assert result.qty == Decimal('3.00')
    assert result.ccy == Currencies["USD"]
    assert result.dov == date(2019, 1, 2)

def test_add_undefined_money_objects():
    na_money = Money.na()
    usd_money = Money.of(Currencies["USD"], Decimal('1.00'), date(2019, 1, 1))
    result1 = na_money + usd_money
    result2 = usd_money + na_money
    assert result1 is usd_money
    assert result2 is usd_money

def test_add_incompatible_currency():
    from pypara.currencies import Currencies
    from datetime import date
    from decimal import Decimal

    usd_money = Money.of(Currencies["USD"], Decimal('1.00'), date(2019, 1, 1))
    eur_money = Money.of(Currencies["EUR"], Decimal('1.00'), date(2019, 1, 1))
    try:
        usd_money + eur_money
        assert False, "Expected IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass


# LLM-generated content at query #33
#--------------------------

```python
def test_int_defined_money():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    money = Money.of(Currencies["USD"], Decimal('10.50'), date(2019, 1, 1))
    assert int(money) == 10

def test_int_undefined_money():
    money = Money.na()
    assert int(money) == 0


# LLM-generated content at query #34
#--------------------------

```python
def test_scalar_subtract_defined_money():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    money = Money.of(Currencies["USD"], Decimal('10.50'), date(2023, 1, 1))
    result = money.scalar_subtract(Decimal('2.30'))
    assert result.qty == Decimal('8.20')
    assert result.ccy == Currencies["USD"]
    assert result.dov == date(2023, 1, 1)

def test_scalar_subtract_undefined_money():
    money = Money.na()
    result = money.scalar_subtract(Decimal('2.30'))
    assert result.undefined


# LLM-generated content at query #35
#--------------------------

```python
def test_times_defined_price():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    price = Price.of(Currencies["USD"], Decimal('10'), date(2019, 1, 1))
    result = price.times(Decimal('2'))
    assert result.qty == Decimal('20')
    assert result.ccy == Currencies["USD"]
    assert result.dov == date(2019, 1, 1)

def test_times_undefined_price():
    price = Price.na()
    result = price.times(Decimal('2'))
    assert result is price


# LLM-generated content at query #36
#--------------------------

```python
def test_abs_defined_price():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    price = Price.of(Currencies["USD"], Decimal('-1.5'), date(2019, 1, 1))
    assert price.abs().qty == Decimal('1.5')
    assert price.abs().ccy == Currencies["USD"]
    assert price.abs().dov == date(2019, 1, 1)

def test_abs_undefined_price():
    undefined_price = Price.na()
    assert undefined_price.abs() is undefined_price


# LLM-generated content at query #37
#--------------------------

```python
def test_add_with_different_currencies_raises_error():
    price1 = SomePrice(Currency("USD"), Decimal("100.00"), Date(2023, 1, 1))
    price2 = SomePrice(Currency("EUR"), Decimal("50.00"), Date(2023, 1, 1))

    with pytest.raises(IncompatibleCurrencyError):
        price1.add(price2)


# LLM-generated content at query #38
#--------------------------

```python
def test_qty_map_defined_price():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    someprice = Price.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    assert someprice.qty_map(lambda x: x + Decimal('1'), lambda: Decimal('42')) == Decimal('2')

def test_qty_map_undefined_price():
    noneprice = Price.of(None, Decimal('1'), None)
    assert noneprice.qty_map(lambda x: x + Decimal('1'), lambda: Decimal('42')) == Decimal('42')


# LLM-generated content at query #39
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


# LLM-generated content at query #40
#--------------------------

```python
def test_add_raises_incompatible_currency_error():
    usd = Currency("USD")
    eur = Currency("EUR")
    money1 = SomeMoney(usd, Decimal("100.00"), Date(2023, 1, 1))
    money2 = SomeMoney(eur, Decimal("50.00"), Date(2023, 1, 1))

    with pytest.raises(IncompatibleCurrencyError):
        money1.add(money2)


# LLM-generated content at query #41
#--------------------------

```python
def test_convert_same_currency():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    price = SomePrice(usd, Decimal("100.00"), Date(2023, 1, 1))
    result = price.convert(usd)
    assert result == SomePrice(usd, Decimal("100.00"), Date(2023, 1, 1))

def test_convert_different_currency_with_rate():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    price = SomePrice(usd, Decimal("100.00"), Date(2023, 1, 1))
    FXRateService.default = MockFXRateService({(usd, eur, Date(2023, 1, 1)): Decimal("0.85")})
    result = price.convert(eur)
    assert result == SomePrice(eur, Decimal("85.00"), Date(2023, 1, 1))

def test_convert_different_currency_no_rate_non_strict():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    price = SomePrice(usd, Decimal("100.00"), Date(2023, 1, 1))
    FXRateService.default = MockFXRateService({})
    result = price.convert(eur, strict=False)
    assert result == NoPrice

def test_convert_different_currency_no_rate_strict():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    price = SomePrice(usd, Decimal("100.00"), Date(2023, 1, 1))
    FXRateService.default = MockFXRateService({})
    try:
        price.convert(eur, strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError:
        pass

def test_convert_with_asof_date():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    price = SomePrice(usd, Decimal("100.00"), Date(2023, 1, 1))
    FXRateService.default = MockFXRateService({(usd, eur, Date(2023, 1, 2)): Decimal("0.85")})
    result = price.convert(eur, asof=Date(2023, 1, 2))
    assert result == SomePrice(eur, Decimal("85.00"), Date(2023, 1, 2))

def test_convert_no_fx_service():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    price = SomePrice(usd, Decimal("100.00"), Date(2023, 1, 1))
    FXRateService.default = None
    try:
        price.convert(eur)
        assert False, "Expected ProgrammingError"
    except ProgrammingError as e:
        assert str(e) == "Did you implement and set the default FX rate service?"


# LLM-generated content at query #42
#--------------------------

```python
def test_lte_undefined_always_less_than_or_equal_to_defined():
    undefined_money = Money.na()
    defined_money = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert undefined_money.lte(defined_money) is True

def test_lte_defined_always_less_than_or_equal_to_undefined():
    defined_money = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    undefined_money = Money.na()
    assert defined_money.lte(undefined_money) is False

def test_lte_both_undefined():
    undefined_money1 = Money.na()
    undefined_money2 = Money.na()
    assert undefined_money1.lte(undefined_money2) is True

def test_lte_same_currency_and_quantity():
    money1 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert money1.lte(money2) is True

def test_lte_same_currency_less_quantity():
    money1 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    assert money1.lte(money2) is True

def test_lte_same_currency_greater_quantity():
    money1 = Money.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert money1.lte(money2) is False

def test_lte_different_currency_raises_error():
    money1 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["EUR"], Decimal('1'), Date(2019, 1, 1))
    try:
        money1.lte(money2)
        assert False, "Expected IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass


# LLM-generated content at query #43
#--------------------------

```python
def test_positive_defined_price_returns_same():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date

    price = Price.of(Currencies["USD"], Decimal('1.5'), date(2019, 1, 1))
    result = price.positive()
    assert result.is_equal(price)

def test_positive_undefined_price_returns_itself():
    undefined_price = Price.na()
    result = undefined_price.positive()
    assert result.is_equal(undefined_price)


# LLM-generated content at query #44
#--------------------------

```python
def test_negative_defined_price():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date

    price = Price.of(Currencies["USD"], Decimal('100.50'), date(2023, 1, 1))
    neg_price = price.negative()

    assert neg_price.defined is True
    assert neg_price.qty == Decimal('-100.50')
    assert neg_price.ccy == Currencies["USD"]
    assert neg_price.dov == date(2023, 1, 1)

def test_negative_undefined_price():
    undefined_price = Price.na()
    neg_undefined_price = undefined_price.negative()

    assert neg_undefined_price.undefined is True
    assert neg_undefined_price is undefined_price


# LLM-generated content at query #45
#--------------------------

```python
def test_qty_or_defined_price():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    price = Price.of(Currencies["USD"], Decimal('1.0'), date(2019, 1, 1))
    assert price.qty_or(Decimal('0.0')) == Decimal('1.0')

def test_qty_or_undefined_price():
    price = Price.na()
    assert price.qty_or(Decimal('0.0')) == Decimal('0.0')


# LLM-generated content at query #46
#--------------------------

```python
def test_with_ccy_defined_price():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    price = Price.of(Currencies["USD"], Decimal('100.50'), date(2023, 1, 1))
    new_price = price.with_ccy(Currencies["EUR"])
    assert new_price.ccy.code == "EUR"
    assert new_price.qty == Decimal('100.50')
    assert new_price.dov == date(2023, 1, 1)

def test_with_ccy_undefined_price():
    undefined_price = Price.na()
    new_price = undefined_price.with_ccy(Currencies["EUR"])
    assert new_price is undefined_price


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_neg_returns_negative_quantity():
    ccy = Currency("USD")
    qty = Decimal("100.50")
    dov = Date(2023, 1, 1)
    money = SomeMoney(ccy, qty, dov)
    neg_money = -money
    assert neg_money.qty == Decimal("-100.50")
    assert neg_money.ccy == ccy
    assert neg_money.dov == dov


# LLM-generated content at query #2
#--------------------------

```python
def test_sub_defined_money():
    from pypara.currencies import Currencies
    from datetime import date
    from decimal import Decimal

    usd = Currencies["USD"]
    money1 = Money.of(usd, Decimal('10.50'), date(2023, 1, 1))
    money2 = Money.of(usd, Decimal('3.25'), date(2023, 1, 2))

    result = money1 - money2

    assert result.defined
    assert result.ccy == usd
    assert result.qty == Decimal('7.25')
    assert result.dov == date(2023, 1, 2)

def test_sub_undefined_money():
    from pypara.currencies import Currencies
    from datetime import date
    from decimal import Decimal

    usd = Currencies["USD"]
    money1 = Money.of(usd, Decimal('10.50'), date(2023, 1, 1))
    money2 = Money.na()

    result = money1 - money2

    assert result is money1

def test_sub_with_undefined_result():
    from pypara.currencies import Currencies
    from datetime import date
    from decimal import Decimal

    money1 = Money.na()
    money2 = Money.of(Currencies["USD"], Decimal('3.25'), date(2023, 1, 2))

    result = money1 - money2

    assert result is money2

def test_sub_incompatible_currency():
    from pypara.currencies import Currencies
    from datetime import date
    from decimal import Decimal

    usd = Currencies["USD"]
    eur = Currencies["EUR"]
    money1 = Money.of(usd, Decimal('10.50'), date(2023, 1, 1))
    money2 = Money.of(eur, Decimal('3.25'), date(2023, 1, 2))

    try:
        _ = money1 - money2
        assert False, "Expected IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass


# LLM-generated content at query #3
#--------------------------

```python
def test_dov_or_returns_correct_date():
    ccy = Currency("USD")
    qty = Decimal("100.00")
    dov = Date(2023, 1, 1)
    money = SomeMoney(ccy, qty, dov)
    assert money.dov_or(Date(2023, 1, 2)) == dov


# LLM-generated content at query #4
#--------------------------

```python
def test_le_same_currency_true():
    price1 = SomePrice(Currency("USD"), Decimal("10.0"), Date(2023, 1, 1))
    price2 = SomePrice(Currency("USD"), Decimal("15.0"), Date(2023, 1, 1))
    assert price1 <= price2

def test_le_same_currency_false():
    price1 = SomePrice(Currency("USD"), Decimal("20.0"), Date(2023, 1, 1))
    price2 = SomePrice(Currency("USD"), Decimal("15.0"), Date(2023, 1, 1))
    assert not (price1 <= price2)

def test_le_same_currency_equal():
    price1 = SomePrice(Currency("USD"), Decimal("15.0"), Date(2023, 1, 1))
    price2 = SomePrice(Currency("USD"), Decimal("15.0"), Date(2023, 1, 1))
    assert price1 <= price2

def test_le_different_currency_raises_error():
    price1 = SomePrice(Currency("USD"), Decimal("10.0"), Date(2023, 1, 1))
    price2 = SomePrice(Currency("EUR"), Decimal("15.0"), Date(2023, 1, 1))
    try:
        price1 <= price2
        assert False, "Expected IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass

def test_le_non_someprice_returns_false():
    price1 = SomePrice(Currency("USD"), Decimal("10.0"), Date(2023, 1, 1))
    assert not (price1 <= "not a price")


# LLM-generated content at query #5
#--------------------------

```python
def test_somemoney_constructor():
    ccy = Currency("USD", 2)
    qty = Decimal("100.50")
    dov = Date(2023, 1, 1)
    money = SomeMoney(ccy, qty, dov)
    assert money.ccy == ccy
    assert money.qty == qty
    assert money.dov == dov
    assert money.defined is True
    assert money.undefined is False


# LLM-generated content at query #6
#--------------------------

```python
def test_or_else_returns_itself_when_defined():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    somemoney = Money.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    fallback = Money.of(Currencies["EUR"], Decimal('2'), date(2019, 1, 2))
    assert somemoney.or_else(lambda: fallback) is somemoney

def test_or_else_returns_fallback_when_undefined():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    nonemoney = Money.na()
    fallback = Money.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    assert nonemoney.or_else(lambda: fallback) is fallback


# LLM-generated content at query #7
#--------------------------

```python
def test_sub_same_currency():
    usd = Currency("USD")
    price1 = SomePrice(usd, Decimal("10.5"), Date(2023, 1, 1))
    price2 = SomePrice(usd, Decimal("5.25"), Date(2023, 1, 2))
    result = price1 - price2
    assert result == SomePrice(usd, Decimal("5.25"), Date(2023, 1, 1))

def test_sub_different_currency_raises_error():
    usd = Currency("USD")
    eur = Currency("EUR")
    price1 = SomePrice(usd, Decimal("10.5"), Date(2023, 1, 1))
    price2 = SomePrice(eur, Decimal("5.25"), Date(2023, 1, 2))
    try:
        result = price1 - price2
        assert False, "Expected IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass

def test_sub_with_undefined_price():
    usd = Currency("USD")
    price1 = SomePrice(usd, Decimal("10.5"), Date(2023, 1, 1))
    price2 = NoPrice()
    result = price1 - price2
    assert result == price1


# LLM-generated content at query #8
#--------------------------

```python
def test_qty_or_defined_money():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    somemoney = Money.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    assert somemoney.qty_or(Decimal(0)) == Decimal('1.00')

def test_qty_or_undefined_money():
    from decimal import Decimal
    nonemoney = Money.of(None, Decimal('1'), None)
    assert nonemoney.qty_or(Decimal(0)) == Decimal('0')


# LLM-generated content at query #9
#--------------------------

```python
def test_subtract_incompatible_currency_error():
    usd = Currency("USD")
    eur = Currency("EUR")
    price_usd = SomePrice(usd, Decimal("100"), Date(2023, 1, 1))
    price_eur = SomePrice(eur, Decimal("50"), Date(2023, 1, 1))
    try:
        price_usd.subtract(price_eur)
        assert False, "Expected IncompatibleCurrencyError"
    except IncompatibleCurrencyError as e:
        assert e.operation == "subtraction"
        assert e.ccy1 == usd
        assert e.ccy2 == eur


# LLM-generated content at query #10
#--------------------------

```python
def test_subtract_defined_prices_same_currency():
    from pypara.currencies import Currencies
    from datetime import date
    from decimal import Decimal
    price1 = Price.of(Currencies["USD"], Decimal('10.50'), date(2023, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('5.25'), date(2023, 1, 2))
    result = price1.subtract(price2)
    assert result.qty == Decimal('5.25')
    assert result.ccy == Currencies["USD"]
    assert result.dov == date(2023, 1, 1)

def test_subtract_undefined_price_with_defined():
    from pypara.currencies import Currencies
    from datetime import date
    from decimal import Decimal
    price1 = Price.na()
    price2 = Price.of(Currencies["USD"], Decimal('5.25'), date(2023, 1, 2))
    result = price1.subtract(price2)
    assert result is price2

def test_subtract_defined_price_with_undefined():
    from pypara.currencies import Currencies
    from datetime import date
    from decimal import Decimal
    price1 = Price.of(Currencies["USD"], Decimal('10.50'), date(2023, 1, 1))
    price2 = Price.na()
    result = price1.subtract(price2)
    assert result is price1

def test_subtract_undefined_prices():
    price1 = Price.na()
    price2 = Price.na()
    result = price1.subtract(price2)
    assert result is Price.na()

def test_subtract_defined_prices_different_currency():
    from pypara.currencies import Currencies
    from pypara.errors import IncompatibleCurrencyError
    from datetime import date
    from decimal import Decimal
    price1 = Price.of(Currencies["USD"], Decimal('10.50'), date(2023, 1, 1))
    price2 = Price.of(Currencies["EUR"], Decimal('5.25'), date(2023, 1, 2))
    try:
        price1.subtract(price2)
        assert False, "Expected IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass


# LLM-generated content at query #11
#--------------------------

```python
def test_scalar_add_defined_money():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    money = Money.of(Currencies["USD"], Decimal('10.50'), date(2023, 1, 1))
    result = money.scalar_add(Decimal('5.25'))
    assert result.qty == Decimal('15.75')
    assert result.ccy == Currencies["USD"]
    assert result.dov == date(2023, 1, 1)

def test_scalar_add_undefined_money():
    undefined_money = Money.na()
    result = undefined_money.scalar_add(Decimal('5.25'))
    assert result.undefined
    assert result is undefined_money


# LLM-generated content at query #12
#--------------------------

```python
def test_qty_or_zero_defined_money():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date

    money = Money.of(Currencies["USD"], Decimal('10.50'), date(2023, 1, 1))
    assert money.qty_or_zero() == Decimal('10.50')

def test_qty_or_zero_undefined_money():
    money = Money.na()
    assert money.qty_or_zero() == Decimal('0')


# LLM-generated content at query #13
#--------------------------

```python
def test_some_money_ge_with_same_currency():
    usd = Currency("USD", 2)
    money1 = SomeMoney(usd, Decimal("100.00"), Date(2023, 1, 1))
    money2 = SomeMoney(usd, Decimal("50.00"), Date(2023, 1, 1))
    assert money1 >= money2

def test_some_money_ge_with_different_currency():
    usd = Currency("USD", 2)
    eur = Currency("EUR", 2)
    money1 = SomeMoney(usd, Decimal("100.00"), Date(2023, 1, 1))
    money2 = SomeMoney(eur, Decimal("50.00"), Date(2023, 1, 1))
    try:
        money1 >= money2
        assert False, "Expected IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass

def test_some_money_ge_with_non_some_money():
    usd = Currency("USD", 2)
    money = SomeMoney(usd, Decimal("100.00"), Date(2023, 1, 1))
    assert money >= "not a money object"


# LLM-generated content at query #14
#--------------------------

```python
def test_gt_undefined_vs_defined():
    assert not Price.na().gt(Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1)))

def test_gt_defined_vs_undefined():
    assert Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1)).gt(Price.na())

def test_gt_defined_same_currency():
    assert Price.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1)).gt(Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1)))

def test_gt_defined_different_currency():
    with pytest.raises(IncompatibleCurrencyError):
        Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1)).gt(Price.of(Currencies["EUR"], Decimal('1'), Date(2019, 1, 1)))


# LLM-generated content at query #15
#--------------------------

```python
def test_with_dov_defined_money():
    from pypara.currencies import Currencies
    from datetime import date
    from decimal import Decimal
    money = Money.of(Currencies["USD"], Decimal('100.50'), date(2023, 1, 1))
    new_money = money.with_dov(date(2023, 2, 1))
    assert new_money.dov == date(2023, 2, 1)
    assert new_money.ccy == Currencies["USD"]
    assert new_money.qty == Decimal('100.50')

def test_with_dov_undefined_money():
    undefined_money = Money.na()
    new_money = undefined_money.with_dov(date(2023, 2, 1))
    assert new_money is undefined_money


# LLM-generated content at query #16
#--------------------------

```python
def test_price_equality():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date

    price1 = Price.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    price3 = Price.of(Currencies["EUR"], Decimal('1'), date(2019, 1, 1))
    price4 = Price.of(Currencies["USD"], Decimal('2'), date(2019, 1, 1))
    price5 = Price.of(Currencies["USD"], Decimal('1'), date(2019, 1, 2))
    undefined_price = Price.na()

    assert price1 == price2
    assert not (price1 == price3)
    assert not (price1 == price4)
    assert not (price1 == price5)
    assert not (price1 == undefined_price)
    assert undefined_price == Price.na()


# LLM-generated content at query #17
#--------------------------

```python
def test_round_defined_price():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    price = Price.of(Currencies["USD"], Decimal('3.14159'), date(2019, 1, 1))
    assert price.round(2).qty == Decimal('3.14')
    assert price.round(0).qty == Decimal('3')
    assert price.round(4).qty == Decimal('3.1416')

def test_round_undefined_price():
    undefined_price = Price.na()
    assert undefined_price.round(2) is undefined_price


# LLM-generated content at query #18
#--------------------------

```python
def test_is_equal_with_same_money():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    money1 = Money.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    assert money1.is_equal(money2) is True

def test_is_equal_with_different_currency():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    money1 = Money.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    money2 = Money.of(Currencies["EUR"], Decimal('1'), date(2019, 1, 1))
    assert money1.is_equal(money2) is False

def test_is_equal_with_different_quantity():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    money1 = Money.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('2'), date(2019, 1, 1))
    assert money1.is_equal(money2) is False

def test_is_equal_with_different_date():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    money1 = Money.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('1'), date(2019, 1, 2))
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
    money2 = Money.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    assert money1.is_equal(money2) is False


# LLM-generated content at query #19
#--------------------------

```python
def test_floor_divide_defined_price():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    price = Price.of(Currencies["USD"], Decimal('10'), date(2019, 1, 1))
    result = price.floor_divide(Decimal('3'))
    assert result.qty == Decimal('3')
    assert result.ccy == Currencies["USD"]
    assert result.dov == date(2019, 1, 1)

def test_floor_divide_undefined_price():
    undefined_price = Price.na()
    result = undefined_price.floor_divide(Decimal('3'))
    assert result.undefined

def test_floor_divide_by_zero():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    price = Price.of(Currencies["USD"], Decimal('10'), date(2019, 1, 1))
    result = price.floor_divide(Decimal('0'))
    assert result.undefined


# LLM-generated content at query #20
#--------------------------

```python
def test_ccy_or_none_defined_money():
    from pypara.currencies import Currencies
    somemoney = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert somemoney.ccy_or_none().code == 'USD'

def test_ccy_or_none_undefined_money():
    nonemoney = Money.of(None, Decimal('1'), None)
    assert nonemoney.ccy_or_none() is None


# LLM-generated content at query #21
#--------------------------

```python
def test_positive_defined_money():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    money = Money.of(Currencies["USD"], Decimal('1.5'), date(2019, 1, 1))
    result = money.positive()
    assert result.defined
    assert result.qty == Decimal('1.5')
    assert result.ccy.code == 'USD'
    assert result.dov == date(2019, 1, 1)

def test_positive_undefined_money():
    money = Money.na()
    result = money.positive()
    assert result.undefined


# LLM-generated content at query #22
#--------------------------

```python
def test_some_money_truediv():
    usd = Currency("USD", 2)
    money = SomeMoney(usd, Decimal("10.50"), Date(2023, 1, 1))
    result = money / 2
    assert result == SomeMoney(usd, Decimal("5.25"), Date(2023, 1, 1))

def test_some_money_truediv_by_zero():
    usd = Currency("USD", 2)
    money = SomeMoney(usd, Decimal("10.50"), Date(2023, 1, 1))
    result = money / 0
    assert result == NoMoney


# LLM-generated content at query #23
#--------------------------

```python
def test_price_ge_defined_same_currency():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    price1 = Price.of(Currencies["USD"], Decimal('10'), date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('5'), date(2019, 1, 1))
    assert price1 >= price2

def test_price_ge_defined_different_currency():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    price1 = Price.of(Currencies["USD"], Decimal('10'), date(2019, 1, 1))
    price2 = Price.of(Currencies["EUR"], Decimal('5'), date(2019, 1, 1))
    try:
        price1 >= price2
        assert False, "Expected IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass

def test_price_ge_undefined_left():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    price1 = Price.na()
    price2 = Price.of(Currencies["USD"], Decimal('5'), date(2019, 1, 1))
    assert not (price1 >= price2)

def test_price_ge_undefined_right():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    price1 = Price.of(Currencies["USD"], Decimal('5'), date(2019, 1, 1))
    price2 = Price.na()
    assert price1 >= price2

def test_price_ge_both_undefined():
    price1 = Price.na()
    price2 = Price.na()
    assert price1 >= price2

def test_price_ge_equal_quantities():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    price1 = Price.of(Currencies["USD"], Decimal('5'), date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('5'), date(2019, 1, 1))
    assert price1 >= price2


# LLM-generated content at query #24
#--------------------------

```python
def test_none_price_constructor():
    none_price = NonePrice()
    assert isinstance(none_price, NonePrice)
    assert not none_price.defined
    assert none_price.undefined


# LLM-generated content at query #25
#--------------------------

```python
def test_as_integer_defined_price():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    price = Price.of(Currencies["USD"], Decimal('10.5'), date(2019, 1, 1))
    assert price.as_integer() == 10

def test_as_integer_undefined_price_raises_exception():
    price = Price.na()
    try:
        price.as_integer()
        assert False, "Expected MonetaryOperationException"
    except Exception as e:
        assert str(e) == "MonetaryOperationException"


# LLM-generated content at query #26
#--------------------------

```python
def test_fmap_defined_price():
    from pypara.currencies import Currencies
    from pypara.prices import Price
    from datetime import date, timedelta
    someprice = Price.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    new = someprice.fmap(lambda x: Price.of(x.ccy, x.qty + Decimal('1'), x.dov + timedelta(days=10)))
    assert new.ccy.code == 'USD'
    assert new.qty == Decimal('2')
    assert new.dov == date(2019, 1, 11)

def test_fmap_undefined_price():
    from pypara.prices import Price
    noneprice = Price.of(None, Decimal('1'), None)
    assert noneprice.fmap(lambda sp: Price.of(sp.ccy, sp.qty + Decimal('1'), sp.dov)) is Price.na()


# LLM-generated content at query #27
#--------------------------

```python
def test_scalar_add_defined_price():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date

    price = Price.of(Currencies["USD"], Decimal('10.5'), date(2023, 1, 1))
    result = price.scalar_add(Decimal('5.5'))

    assert result.defined
    assert result.ccy == Currencies["USD"]
    assert result.qty == Decimal('16.0')
    assert result.dov == date(2023, 1, 1)

def test_scalar_add_undefined_price():
    undefined_price = Price.na()
    result = undefined_price.scalar_add(Decimal('5.5'))

    assert result.undefined
    assert result is undefined_price


# LLM-generated content at query #28
#--------------------------

```python
def test_mul_defined_money():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    money = Money.of(Currencies["USD"], Decimal('10.50'), date(2023, 1, 1))
    result = money * 2
    assert result.qty == Decimal('21.00')
    assert result.ccy == Currencies["USD"]
    assert result.dov == date(2023, 1, 1)

def test_mul_undefined_money():
    money = Money.na()
    result = money * 5
    assert result.undefined

def test_mul_by_zero():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    money = Money.of(Currencies["USD"], Decimal('10.50'), date(2023, 1, 1))
    result = money * 0
    assert result.qty == Decimal('0.00')
    assert result.ccy == Currencies["USD"]
    assert result.dov == date(2023, 1, 1)

def test_mul_by_negative():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    money = Money.of(Currencies["USD"], Decimal('10.50'), date(2023, 1, 1))
    result = money * -1
    assert result.qty == Decimal('-10.50')
    assert result.ccy == Currencies["USD"]
    assert result.dov == date(2023, 1, 1)

def test_mul_by_float():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    money = Money.of(Currencies["USD"], Decimal('10.50'), date(2023, 1, 1))
    result = money * 1.5
    assert result.qty == Decimal('15.75')
    assert result.ccy == Currencies["USD"]
    assert result.dov == date(2023, 1, 1)


# LLM-generated content at query #29
#--------------------------

```python
def test_is_equal_returns_true_for_equal_prices():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    price1 = Price.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    assert price1.is_equal(price2) is True

def test_is_equal_returns_false_for_different_prices():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    price1 = Price.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    price2 = Price.of(Currencies["EUR"], Decimal('1'), date(2019, 1, 1))
    assert price1.is_equal(price2) is False

def test_is_equal_returns_false_for_undefined_price():
    price1 = Price.na()
    price2 = Price.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    assert price1.is_equal(price2) is False


# LLM-generated content at query #30
#--------------------------

```python
def test_lt_undefined_vs_defined():
    assert Money.na().lt(Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))) == True

def test_lt_defined_vs_undefined():
    assert Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1)).lt(Money.na()) == False

def test_lt_undefined_vs_undefined():
    assert Money.na().lt(Money.na()) == False

def test_lt_same_currency():
    assert Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1)).lt(Money.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))) == True
    assert Money.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1)).lt(Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))) == False

def test_lt_different_currency():
    with pytest.raises(IncompatibleCurrencyError):
        Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1)).lt(Money.of(Currencies["EUR"], Decimal('1'), Date(2019, 1, 1)))


# LLM-generated content at query #31
#--------------------------

```python
def test_convert_defined_price():
    from pypara.currencies import Currencies
    from datetime import date
    from decimal import Decimal
    usd_price = Price.of(Currencies["USD"], Decimal('100'), date(2023, 1, 1))
    eur_price = usd_price.convert(Currencies["EUR"], asof=date(2023, 1, 1))
    assert eur_price.ccy == Currencies["EUR"]
    assert eur_price.dov == date(2023, 1, 1)
    assert eur_price.qty != Decimal('100')

def test_convert_undefined_price():
    na_price = Price.na()
    converted_price = na_price.convert(Currencies["EUR"], asof=date(2023, 1, 1))
    assert converted_price.undefined
    assert converted_price is na_price

def test_convert_same_currency():
    from pypara.currencies import Currencies
    from datetime import date
    from decimal import Decimal
    usd_price = Price.of(Currencies["USD"], Decimal('100'), date(2023, 1, 1))
    same_currency_price = usd_price.convert(Currencies["USD"], asof=date(2023, 1, 1))
    assert same_currency_price.ccy == Currencies["USD"]
    assert same_currency_price.qty == Decimal('100')
    assert same_currency_price.dov == date(2023, 1, 1)


# LLM-generated content at query #32
#--------------------------

```python
def test_some_money_le_with_same_currency():
    usd = Currency("USD", "USD", "US Dollar", 2)
    money1 = SomeMoney(usd, Decimal("10.00"), Date(2023, 1, 1))
    money2 = SomeMoney(usd, Decimal("15.00"), Date(2023, 1, 1))
    assert money1 <= money2

def test_some_money_le_with_different_currency():
    usd = Currency("USD", "USD", "US Dollar", 2)
    eur = Currency("EUR", "EUR", "Euro", 2)
    money1 = SomeMoney(usd, Decimal("10.00"), Date(2023, 1, 1))
    money2 = SomeMoney(eur, Decimal("10.00"), Date(2023, 1, 1))
    try:
        money1 <= money2
        assert False, "Expected IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass

def test_some_money_le_with_non_money_object():
    usd = Currency("USD", "USD", "US Dollar", 2)
    money = SomeMoney(usd, Decimal("10.00"), Date(2023, 1, 1))
    assert not (money <= "not a money object")


# LLM-generated content at query #33
#--------------------------

```python
def test_gt_undefined_vs_defined():
    undefined_price = Price.na()
    defined_price = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert not undefined_price.gt(defined_price)

def test_gt_defined_vs_undefined():
    defined_price = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    undefined_price = Price.na()
    assert defined_price.gt(undefined_price)

def test_gt_defined_vs_defined_same_currency():
    price1 = Price.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert price1.gt(price2)

def test_gt_defined_vs_defined_different_currency():
    price1 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["EUR"], Decimal('1'), Date(2019, 1, 1))
    try:
        price1.gt(price2)
        assert False, "Expected IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass

def test_gt_undefined_vs_undefined():
    undefined_price1 = Price.na()
    undefined_price2 = Price.na()
    assert not undefined_price1.gt(undefined_price2)


# LLM-generated content at query #34
#--------------------------

```python
def test_mul_defined_price_with_scalar():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    price = Price.of(Currencies["USD"], Decimal('10.5'), date(2023, 1, 1))
    result = price * 2
    assert result.qty == Decimal('21.0')
    assert result.ccy == Currencies["USD"]
    assert result.dov == date(2023, 1, 1)

def test_mul_undefined_price_with_scalar():
    price = Price.na()
    result = price * 5
    assert result.undefined
    assert result is price

def test_mul_defined_price_with_zero():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    price = Price.of(Currencies["USD"], Decimal('10.5'), date(2023, 1, 1))
    result = price * 0
    assert result.qty == Decimal('0')
    assert result.ccy == Currencies["USD"]
    assert result.dov == date(2023, 1, 1)

def test_mul_defined_price_with_negative_scalar():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    price = Price.of(Currencies["USD"], Decimal('10.5'), date(2023, 1, 1))
    result = price * -1
    assert result.qty == Decimal('-10.5')
    assert result.ccy == Currencies["USD"]
    assert result.dov == date(2023, 1, 1)

def test_mul_defined_price_with_float():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    price = Price.of(Currencies["USD"], Decimal('10.5'), date(2023, 1, 1))
    result = price * 1.5
    assert result.qty == Decimal('15.75')
    assert result.ccy == Currencies["USD"]
    assert result.dov == date(2023, 1, 1)


# LLM-generated content at query #35
#--------------------------

```python
def test_scalar_subtract_defined_money():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    money = Money.of(Currencies["USD"], Decimal('10.50'), date(2019, 1, 1))
    result = money.scalar_subtract(Decimal('5.25'))
    assert result.qty == Decimal('5.25')
    assert result.ccy == Currencies["USD"]
    assert result.dov == date(2019, 1, 1)

def test_scalar_subtract_undefined_money():
    money = Money.na()
    result = money.scalar_subtract(Decimal('5.25'))
    assert result.undefined


# LLM-generated content at query #36
#--------------------------

```python
def test_gt_undefined_vs_defined():
    assert Price.na().gt(Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))) == False

def test_gt_defined_vs_undefined():
    assert Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1)).gt(Price.na()) == True

def test_gt_defined_vs_defined_same_currency():
    assert Price.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1)).gt(Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))) == True
    assert Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1)).gt(Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))) == False
    assert Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1)).gt(Price.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))) == False

def test_gt_defined_vs_defined_different_currency():
    with pytest.raises(IncompatibleCurrencyError):
        Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1)).gt(Price.of(Currencies["EUR"], Decimal('1'), Date(2019, 1, 1)))


# LLM-generated content at query #37
#--------------------------

```python
def test_SomePrice___ge__():
    price1 = SomePrice(Currency("USD"), Decimal("10.00"), Date(2023, 1, 1))
    price2 = SomePrice(Currency("USD"), Decimal("5.00"), Date(2023, 1, 1))
    price3 = SomePrice(Currency("USD"), Decimal("10.00"), Date(2023, 1, 1))
    price4 = SomePrice(Currency("EUR"), Decimal("10.00"), Date(2023, 1, 1))

    assert price1 >= price2
    assert price1 >= price3
    assert not price2 >= price1
    assert price1.__ge__(price2)
    assert price1.__ge__(price3)
    assert not price2.__ge__(price1)


# LLM-generated content at query #38
#--------------------------

```python
def test_gt_undefined_vs_defined():
    assert not Money.na().gt(Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1)))

def test_gt_defined_vs_undefined():
    assert Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1)).gt(Money.na())

def test_gt_defined_vs_defined_same_currency():
    assert Money.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1)).gt(Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1)))
    assert not Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1)).gt(Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1)))
    assert not Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1)).gt(Money.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1)))

def test_gt_defined_vs_defined_different_currency():
    with pytest.raises(IncompatibleCurrencyError):
        Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1)).gt(Money.of(Currencies["EUR"], Decimal('1'), Date(2019, 1, 1)))


# LLM-generated content at query #39
#--------------------------

```python
def test_dov_or_none_defined_price():
    from pypara.currencies import Currencies
    from datetime import date
    someprice = Price.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    assert someprice.dov_or_none() == date(2019, 1, 1)

def test_dov_or_none_undefined_price():
    noneprice = Price.of(None, None, date(2019, 1, 1))
    assert noneprice.dov_or_none() is None


# LLM-generated content at query #40
#--------------------------

```python
def test_scalar_subtract_defined_money():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    money = Money.of(Currencies["USD"], Decimal('10.50'), date(2019, 1, 1))
    result = money.scalar_subtract(Decimal('2.50'))
    assert result.qty == Decimal('8.00')
    assert result.ccy == Currencies["USD"]
    assert result.dov == date(2019, 1, 1)

def test_scalar_subtract_undefined_money():
    money = Money.na()
    result = money.scalar_subtract(Decimal('2.50'))
    assert result.undefined


# LLM-generated content at query #41
#--------------------------

```python
def test_add_defined_money_objects():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    money1 = Money.of(Currencies["USD"], Decimal('10.50'), date(2023, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('5.25'), date(2023, 1, 2))
    result = money1 + money2
    assert result.ccy == Currencies["USD"]
    assert result.qty == Decimal('15.75')
    assert result.dov == date(2023, 1, 2)

def test_add_undefined_money_objects():
    money1 = Money.na()
    money2 = Money.na()
    result = money1 + money2
    assert result.undefined

def test_add_defined_and_undefined_money_objects():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    money1 = Money.of(Currencies["USD"], Decimal('10.50'), date(2023, 1, 1))
    money2 = Money.na()
    result = money1 + money2
    assert result == money1

def test_add_incompatible_currencies():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    from pypara.errors import IncompatibleCurrencyError
    money1 = Money.of(Currencies["USD"], Decimal('10.50'), date(2023, 1, 1))
    money2 = Money.of(Currencies["EUR"], Decimal('5.25'), date(2023, 1, 2))
    try:
        result = money1 + money2
        assert False, "Expected IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass


# LLM-generated content at query #42
#--------------------------

```python
def test_or_else_defined_price_returns_itself():
    from pypara.currencies import Currencies
    fallback = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    someprice = Price.of(Currencies["EUR"], Decimal('2'), Date(2019, 1, 2))
    assert someprice.or_else(lambda: fallback) is someprice

def test_or_else_undefined_price_returns_fallback():
    from pypara.currencies import Currencies
    fallback = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    noneprice = Price.of(None, Decimal('1'), None)
    assert noneprice.or_else(lambda: fallback) is fallback


# LLM-generated content at query #43
#--------------------------

```python
def test_round_defined_price():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date

    price = Price.of(Currencies["USD"], Decimal('3.14159'), date(2019, 1, 1))
    rounded = price.round(2)
    assert rounded.qty == Decimal('3.14')

def test_round_undefined_price():
    undefined_price = Price.na()
    assert undefined_price.round(2) is undefined_price


# LLM-generated content at query #44
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
    from decimal import Decimal
    from datetime import date
    noneprice = Price.of(None, Decimal('1'), None)
    fallback = Price.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    assert noneprice.or_else(lambda: fallback) is fallback


# LLM-generated content at query #45
#--------------------------

```python
def test_equality_of_two_money_objects():
    from pypara.currencies import Currencies
    money1 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert money1 == money2

def test_equality_of_money_with_different_currency():
    from pypara.currencies import Currencies
    money1 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["EUR"], Decimal('1'), Date(2019, 1, 1))
    assert money1 != money2

def test_equality_of_money_with_different_quantity():
    from pypara.currencies import Currencies
    money1 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    assert money1 != money2

def test_equality_of_money_with_different_date():
    from pypara.currencies import Currencies
    money1 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 2))
    assert money1 != money2

def test_equality_of_undefined_money():
    assert Money.na() == Money.na()

def test_equality_of_defined_and_undefined_money():
    from pypara.currencies import Currencies
    money = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert money != Money.na()


