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
    price = Price.of(Currencies["USD"], Decimal('1.0'), date(2019, 1, 1))
    assert price.qty_or_none() == Decimal('1.0')

def test_qty_or_none_undefined_price():
    price = Price.na()
    assert price.qty_or_none() is None


# LLM-generated content at query #2
#--------------------------

```python
def test_someprice_abs():
    price = SomePrice(Currency("USD"), Decimal("-100.50"), Date(2023, 1, 1))
    abs_price = price.__abs__()
    assert abs_price == SomePrice(Currency("USD"), Decimal("100.50"), Date(2023, 1, 1))


# LLM-generated content at query #3
#--------------------------

```python
def test_with_ccy_defined_price():
    from pypara.currencies import Currencies
    from datetime import date
    from decimal import Decimal

    price = Price.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    new_price = price.with_ccy(Currencies["EUR"])

    assert new_price.ccy.code == "EUR"
    assert new_price.qty == Decimal('1')
    assert new_price.dov == date(2019, 1, 1)

def test_with_ccy_undefined_price():
    price = Price.na()
    new_price = price.with_ccy(Currencies["EUR"])

    assert new_price is price


# LLM-generated content at query #4
#--------------------------

```python
def test_sub_defined_money():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    m1 = Money.of(Currencies["USD"], Decimal('10'), date(2020, 1, 1))
    m2 = Money.of(Currencies["USD"], Decimal('5'), date(2020, 1, 1))
    result = m1 - m2
    assert result.qty == Decimal('5')
    assert result.ccy == Currencies["USD"]
    assert result.dov == date(2020, 1, 1)

def test_sub_undefined_money():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    m1 = Money.of(Currencies["USD"], Decimal('10'), date(2020, 1, 1))
    m2 = Money.na()
    result = m1 - m2
    assert result is m1

def test_sub_incompatible_currency():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    from pypara.errors import IncompatibleCurrencyError
    m1 = Money.of(Currencies["USD"], Decimal('10'), date(2020, 1, 1))
    m2 = Money.of(Currencies["EUR"], Decimal('5'), date(2020, 1, 1))
    try:
        result = m1 - m2
        assert False, "Expected IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass


# LLM-generated content at query #5
#--------------------------

```python
def test_ccy_or_none_defined_price():
    from pypara.currencies import Currencies
    someprice = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert someprice.ccy_or_none().code == 'USD'

def test_ccy_or_none_undefined_price():
    someprice = Price.of(None, Decimal('1'), None)
    assert someprice.ccy_or_none() is None


# LLM-generated content at query #6
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


# LLM-generated content at query #7
#--------------------------

```python
def test_someprice_neg():
    ccy = Currency("USD")
    qty = Decimal("10.5")
    dov = Date(2023, 1, 1)
    price = SomePrice(ccy, qty, dov)
    neg_price = -price
    assert neg_price.ccy == ccy
    assert neg_price.qty == Decimal("-10.5")
    assert neg_price.dov == dov


# LLM-generated content at query #8
#--------------------------

```python
def test_dov_or_none():
    ccy = Currency("USD")
    qty = Decimal("100.00")
    dov = Date(2023, 1, 1)
    price = SomePrice(ccy, qty, dov)
    assert price.dov_or_none() == dov


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

def test_truediv_invalid_operation():
    price = SomePrice(Currency("USD"), Decimal("10.0"), Date(2023, 1, 1))
    result = price / Decimal("NaN")
    assert result == NoPrice


# LLM-generated content at query #10
#--------------------------

```python
def test_ccy_or_returns_currency():
    ccy = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    price = SomePrice(ccy, Decimal("10.5"), Date(2023, 1, 1))
    assert price.ccy_or(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)) == ccy


# LLM-generated content at query #11
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


# LLM-generated content at query #12
#--------------------------

```python
def test_floor_divide_defined_money():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    money = Money.of(Currencies["USD"], Decimal('10'), date(2019, 1, 1))
    result = money // Decimal('3')
    assert result.defined
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


# LLM-generated content at query #13
#--------------------------

```python
def test_gt_undefined_vs_defined():
    undefined_money = Money.na()
    defined_money = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert not undefined_money.gt(defined_money)

def test_gt_defined_vs_undefined():
    defined_money = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    undefined_money = Money.na()
    assert defined_money.gt(undefined_money)

def test_gt_same_currency_and_quantity():
    money1 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 2))
    assert not money1.gt(money2)
    assert not money2.gt(money1)

def test_gt_same_currency_different_quantity():
    money1 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 2))
    assert money2.gt(money1)
    assert not money1.gt(money2)

def test_gt_different_currency():
    money1 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["EUR"], Decimal('1'), Date(2019, 1, 2))
    try:
        money1.gt(money2)
        assert False, "Expected IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass


# LLM-generated content at query #14
#--------------------------

```python
def test_lte_undefined_less_than_or_equal_to_defined():
    assert Money.na().lte(Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1)))

def test_lte_undefined_less_than_or_equal_to_undefined():
    assert Money.na().lte(Money.na())

def test_lte_defined_less_than_or_equal_to_undefined():
    assert not Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1)).lte(Money.na())

def test_lte_same_currency_less_than_or_equal():
    assert Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1)).lte(Money.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 2)))
    assert Money.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1)).lte(Money.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 2)))
    assert not Money.of(Currencies["USD"], Decimal('3'), Date(2019, 1, 1)).lte(Money.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 2)))

def test_lte_different_currency_raises_error():
    with pytest.raises(IncompatibleCurrencyError):
        Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1)).lte(Money.of(Currencies["EUR"], Decimal('1'), Date(2019, 1, 1)))


# LLM-generated content at query #15
#--------------------------

```python
def test_pos_defined_price():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    price = Price.of(Currencies["USD"], Decimal('1.5'), date(2019, 1, 1))
    result = +price
    assert result.is_equal(price)
    assert result.qty == Decimal('1.5')
    assert result.ccy == Currencies["USD"]
    assert result.dov == date(2019, 1, 1)

def test_pos_undefined_price():
    price = Price.na()
    result = +price
    assert result.is_equal(price)
    assert result.undefined


# LLM-generated content at query #16
#--------------------------

```python
def test_multiply_defined_money():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    money = Money.of(Currencies["USD"], Decimal('10'), date(2019, 1, 1))
    result = money.multiply(2)
    assert result.qty == Decimal('20')
    assert result.ccy == Currencies["USD"]
    assert result.dov == date(2019, 1, 1)

def test_multiply_undefined_money():
    money = Money.na()
    result = money.multiply(2)
    assert result.undefined
    assert result is money

def test_multiply_by_zero():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    money = Money.of(Currencies["USD"], Decimal('10'), date(2019, 1, 1))
    result = money.multiply(0)
    assert result.qty == Decimal('0')
    assert result.ccy == Currencies["USD"]
    assert result.dov == date(2019, 1, 1)

def test_multiply_by_negative():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    money = Money.of(Currencies["USD"], Decimal('10'), date(2019, 1, 1))
    result = money.multiply(-1)
    assert result.qty == Decimal('-10')
    assert result.ccy == Currencies["USD"]
    assert result.dov == date(2019, 1, 1)

def test_multiply_by_float():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    money = Money.of(Currencies["USD"], Decimal('10'), date(2019, 1, 1))
    result = money.multiply(1.5)
    assert result.qty == Decimal('15.00')
    assert result.ccy == Currencies["USD"]
    assert result.dov == date(2019, 1, 1)


# LLM-generated content at query #17
#--------------------------

```python
def test_lt_undefined_vs_defined():
    assert Price.na().lt(Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))) is True

def test_lt_defined_vs_undefined():
    assert Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1)).lt(Price.na()) is False

def test_lt_defined_vs_defined_same_currency():
    assert Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1)).lt(Price.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))) is True
    assert Price.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1)).lt(Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))) is False

def test_lt_defined_vs_defined_different_currency():
    with pytest.raises(IncompatibleCurrencyError):
        Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1)).lt(Price.of(Currencies["EUR"], Decimal('2'), Date(2019, 1, 1)))


# LLM-generated content at query #18
#--------------------------

```python
def test_scalar_add_defined_price():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    price = Price.of(Currencies["USD"], Decimal('10.5'), date(2023, 1, 1))
    result = price.scalar_add(Decimal('5.5'))
    assert result.qty == Decimal('16.0')
    assert result.ccy == Currencies["USD"]
    assert result.dov == date(2023, 1, 1)

def test_scalar_add_undefined_price():
    undefined_price = Price.na()
    result = undefined_price.scalar_add(Decimal('5.5'))
    assert result.undefined


# LLM-generated content at query #19
#--------------------------

```python
def test_neg_defined_price():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    price = Price.of(Currencies["USD"], Decimal('10.50'), date(2023, 1, 1))
    neg_price = -price
    assert neg_price.qty == Decimal('-10.50')
    assert neg_price.ccy == Currencies["USD"]
    assert neg_price.dov == date(2023, 1, 1)

def test_neg_undefined_price():
    undefined_price = Price.na()
    neg_undefined_price = -undefined_price
    assert neg_undefined_price.undefined
    assert neg_undefined_price is undefined_price


# LLM-generated content at query #20
#--------------------------

```python
def test_dov_or_defined_price():
    from pypara.currencies import Currencies
    from datetime import date
    from decimal import Decimal
    price = Price.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    assert price.dov_or(date(2001, 1, 1)) == date(2019, 1, 1)

def test_dov_or_undefined_price():
    from pypara.currencies import Currencies
    from datetime import date
    from decimal import Decimal
    price = Price.of(None, None, date(2019, 1, 1))
    assert price.dov_or(date(2001, 1, 1)) == date(2001, 1, 1)


# LLM-generated content at query #21
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


# LLM-generated content at query #22
#--------------------------

```python
def test_someprice_gt_true():
    usd = Currency("USD")
    price1 = SomePrice(usd, Decimal("10.00"), Date(2023, 1, 1))
    price2 = SomePrice(usd, Decimal("5.00"), Date(2023, 1, 1))
    assert price1 > price2

def test_someprice_gt_false():
    usd = Currency("USD")
    price1 = SomePrice(usd, Decimal("5.00"), Date(2023, 1, 1))
    price2 = SomePrice(usd, Decimal("10.00"), Date(2023, 1, 1))
    assert not (price1 > price2)

def test_someprice_gt_equal():
    usd = Currency("USD")
    price1 = SomePrice(usd, Decimal("10.00"), Date(2023, 1, 1))
    price2 = SomePrice(usd, Decimal("10.00"), Date(2023, 1, 1))
    assert not (price1 > price2)

def test_someprice_gt_different_currency():
    usd = Currency("USD")
    eur = Currency("EUR")
    price1 = SomePrice(usd, Decimal("10.00"), Date(2023, 1, 1))
    price2 = SomePrice(eur, Decimal("5.00"), Date(2023, 1, 1))
    try:
        assert price1 > price2
    except IncompatibleCurrencyError:
        pass
    else:
        assert False, "Expected IncompatibleCurrencyError"

def test_someprice_gt_non_someprice():
    usd = Currency("USD")
    price = SomePrice(usd, Decimal("10.00"), Date(2023, 1, 1))
    assert price > "not a price"


# LLM-generated content at query #23
#--------------------------

```python
def test_price_defined_positive_quantity():
    from pypara.currencies import Currencies
    price = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert bool(price) is True

def test_price_defined_zero_quantity():
    from pypara.currencies import Currencies
    price = Price.of(Currencies["USD"], Decimal('0'), Date(2019, 1, 1))
    assert bool(price) is False

def test_price_undefined():
    price = Price.na()
    assert bool(price) is False


# LLM-generated content at query #24
#--------------------------

```python
def test_dov_or_defined_price():
    from pypara.currencies import Currencies
    from datetime import date
    from decimal import Decimal
    price = Price.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    assert price.dov_or(date(2001, 1, 1)) == date(2019, 1, 1)

def test_dov_or_undefined_price():
    price = Price.na()
    assert price.dov_or(date(2001, 1, 1)) == date(2001, 1, 1)


# LLM-generated content at query #25
#--------------------------

```python
def test_convert_same_currency():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    money = SomeMoney(usd, Decimal("100.00"), Date(2023, 1, 1))
    converted = money.convert(usd)
    assert converted == SomeMoney(usd, Decimal("100.00"), Date(2023, 1, 1))

def test_convert_different_currency_with_rate():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    money = SomeMoney(usd, Decimal("100.00"), Date(2023, 1, 1))
    FXRateService.default = MockFXRateService({(usd, eur, Date(2023, 1, 1)): Decimal("0.85")})
    converted = money.convert(eur)
    assert converted == SomeMoney(eur, Decimal("85.00"), Date(2023, 1, 1))

def test_convert_different_currency_no_rate_non_strict():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    money = SomeMoney(usd, Decimal("100.00"), Date(2023, 1, 1))
    FXRateService.default = MockFXRateService({})
    converted = money.convert(eur, strict=False)
    assert converted == NoMoney

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
    FXRateService.default = MockFXRateService({(usd, eur, Date(2023, 2, 1)): Decimal("0.90")})
    converted = money.convert(eur, asof=Date(2023, 2, 1))
    assert converted == SomeMoney(eur, Decimal("90.00"), Date(2023, 2, 1))

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


# LLM-generated content at query #26
#--------------------------

```python
def test_equality_of_two_defined_money_objects_with_same_attributes():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    money1 = Money.of(Currencies["USD"], Decimal('100.00'), date(2023, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('100.00'), date(2023, 1, 1))
    assert money1 == money2

def test_equality_of_two_defined_money_objects_with_different_currencies():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    money1 = Money.of(Currencies["USD"], Decimal('100.00'), date(2023, 1, 1))
    money2 = Money.of(Currencies["EUR"], Decimal('100.00'), date(2023, 1, 1))
    assert not (money1 == money2)

def test_equality_of_two_defined_money_objects_with_different_quantities():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    money1 = Money.of(Currencies["USD"], Decimal('100.00'), date(2023, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('200.00'), date(2023, 1, 1))
    assert not (money1 == money2)

def test_equality_of_two_defined_money_objects_with_different_dates():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    money1 = Money.of(Currencies["USD"], Decimal('100.00'), date(2023, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('100.00'), date(2023, 1, 2))
    assert not (money1 == money2)

def test_equality_of_defined_and_undefined_money_objects():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    money1 = Money.of(Currencies["USD"], Decimal('100.00'), date(2023, 1, 1))
    money2 = Money.na()
    assert not (money1 == money2)

def test_equality_of_two_undefined_money_objects():
    money1 = Money.na()
    money2 = Money.na()
    assert money1 == money2

def test_equality_of_money_object_with_non_money_object():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    money = Money.of(Currencies["USD"], Decimal('100.00'), date(2023, 1, 1))
    assert not (money == "not a money object")


# LLM-generated content at query #27
#--------------------------

```python
def test_none_price_constructor():
    none_price = NonePrice()
    assert isinstance(none_price, NonePrice)
    assert not none_price.defined
    assert none_price.undefined


# LLM-generated content at query #28
#--------------------------

```python
def test_some_money_ge_with_same_currency_and_greater_quantity():
    usd = Currency("USD", 2)
    money1 = SomeMoney(usd, Decimal("100.00"), Date(2023, 1, 1))
    money2 = SomeMoney(usd, Decimal("50.00"), Date(2023, 1, 1))
    assert money1 >= money2

def test_some_money_ge_with_same_currency_and_equal_quantity():
    usd = Currency("USD", 2)
    money1 = SomeMoney(usd, Decimal("100.00"), Date(2023, 1, 1))
    money2 = SomeMoney(usd, Decimal("100.00"), Date(2023, 1, 1))
    assert money1 >= money2

def test_some_money_ge_with_same_currency_and_less_quantity():
    usd = Currency("USD", 2)
    money1 = SomeMoney(usd, Decimal("50.00"), Date(2023, 1, 1))
    money2 = SomeMoney(usd, Decimal("100.00"), Date(2023, 1, 1))
    assert not (money1 >= money2)

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

def test_some_money_ge_with_non_money_object():
    usd = Currency("USD", 2)
    money = SomeMoney(usd, Decimal("100.00"), Date(2023, 1, 1))
    assert money >= "not a money object"


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_neg_returns_negative_quantity():
    usd = Currency("USD", 2)
    money = SomeMoney(usd, Decimal("100.00"), Date(2023, 1, 1))
    neg_money = -money
    assert neg_money.qty == Decimal("-100.00")
    assert neg_money.ccy == usd
    assert neg_money.dov == Date(2023, 1, 1)


# LLM-generated content at query #2
#--------------------------

```python
def test_someprice_truediv_valid_division():
    price1 = SomePrice(Currency("USD"), Decimal("10.0"), Date(2023, 1, 1))
    price2 = price1 / Decimal("2.0")
    assert price2 == SomePrice(Currency("USD"), Decimal("5.0"), Date(2023, 1, 1))

def test_someprice_truediv_by_zero():
    price = SomePrice(Currency("USD"), Decimal("10.0"), Date(2023, 1, 1))
    result = price / Decimal("0.0")
    assert result == NoPrice

def test_someprice_truediv_invalid_operation():
    price = SomePrice(Currency("USD"), Decimal("10.0"), Date(2023, 1, 1))
    result = price / "invalid"
    assert result == NoPrice


# LLM-generated content at query #3
#--------------------------

```python
def test_with_ccy_defined_price():
    from pypara.currencies import Currencies
    price = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    new_price = price.with_ccy(Currencies["EUR"])
    assert new_price.ccy.code == "EUR"
    assert new_price.qty == Decimal('1')
    assert new_price.dov == Date(2019, 1, 1)

def test_with_ccy_undefined_price():
    price = Price.na()
    new_price = price.with_ccy(Currencies["EUR"])
    assert new_price is price


# LLM-generated content at query #4
#--------------------------

```python
def test_some_money_ge_with_same_currency_and_greater_quantity():
    usd = Currency("USD", 2)
    money1 = SomeMoney(usd, Decimal("10.50"), Date(2023, 1, 1))
    money2 = SomeMoney(usd, Decimal("9.50"), Date(2023, 1, 1))
    assert money1 >= money2

def test_some_money_ge_with_same_currency_and_equal_quantity():
    usd = Currency("USD", 2)
    money1 = SomeMoney(usd, Decimal("10.00"), Date(2023, 1, 1))
    money2 = SomeMoney(usd, Decimal("10.00"), Date(2023, 1, 1))
    assert money1 >= money2

def test_some_money_ge_with_same_currency_and_lesser_quantity():
    usd = Currency("USD", 2)
    money1 = SomeMoney(usd, Decimal("9.50"), Date(2023, 1, 1))
    money2 = SomeMoney(usd, Decimal("10.50"), Date(2023, 1, 1))
    assert not (money1 >= money2)

def test_some_money_ge_with_different_currency():
    usd = Currency("USD", 2)
    eur = Currency("EUR", 2)
    money1 = SomeMoney(usd, Decimal("10.00"), Date(2023, 1, 1))
    money2 = SomeMoney(eur, Decimal("10.00"), Date(2023, 1, 1))
    try:
        assert money1 >= money2
    except IncompatibleCurrencyError:
        pass

def test_some_money_ge_with_non_money_object():
    usd = Currency("USD", 2)
    money = SomeMoney(usd, Decimal("10.00"), Date(2023, 1, 1))
    assert money >= "not a money object"


# LLM-generated content at query #5
#--------------------------

```
def test_some_money_equality():
    usd = Currency("USD", 2)
    eur = Currency("EUR", 2)
    qty1 = Decimal("100.00")
    qty2 = Decimal("100.00")
    qty3 = Decimal("200.00")
    dov1 = Date(2023, 1, 1)
    dov2 = Date(2023, 1, 1)
    dov3 = Date(2023, 1, 2)

    money1 = SomeMoney(usd, qty1, dov1)
    money2 = SomeMoney(usd, qty2, dov2)
    money3 = SomeMoney(usd, qty3, dov1)
    money4 = SomeMoney(eur, qty1, dov1)

    assert money1 == money2
    assert not (money1 == money3)
    assert not (money1 == money4)
    assert not (money1 == "not a money object")


# LLM-generated content at query #6
#--------------------------

```python
def test_none_price_constructor():
    none_price = NonePrice()
    assert isinstance(none_price, NonePrice)
    assert none_price.defined is False
    assert none_price.undefined is True


# LLM-generated content at query #7
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
    money = Money.na()
    result = money.scalar_add(Decimal('5.25'))
    assert result is money
    assert result.undefined


# LLM-generated content at query #8
#--------------------------

```python
def test_some_money_int_conversion():
    usd = Currency("USD", 2)
    money = SomeMoney(usd, Decimal("123.45"), Date(2023, 1, 1))
    assert money.__int__() == 123


# LLM-generated content at query #9
#--------------------------

```python
def test_le_with_undefined_money():
    undefined_money = Money.na()
    defined_money = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert undefined_money <= defined_money
    assert not (defined_money <= undefined_money)

def test_le_with_same_currency():
    money1 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 2))
    assert money1 <= money2
    assert not (money2 <= money1)

def test_le_with_different_currency():
    money1 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["EUR"], Decimal('1'), Date(2019, 1, 1))
    try:
        money1 <= money2
        assert False, "Expected IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass

def test_le_with_equal_money():
    money1 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 2))
    assert money1 <= money2
    assert money2 <= money1


# LLM-generated content at query #10
#--------------------------

```python
def test_with_ccy_defined_money():
    from pypara.currencies import Currencies
    usd_money = Money.of(Currencies["USD"], Decimal('100'), Date(2023, 1, 1))
    eur_money = usd_money.with_ccy(Currencies["EUR"])
    assert eur_money.ccy.code == "EUR"
    assert eur_money.qty == Decimal('100')
    assert eur_money.dov == Date(2023, 1, 1)

def test_with_ccy_undefined_money():
    undefined_money = Money.na()
    result = undefined_money.with_ccy(Currencies["EUR"])
    assert result is undefined_money


# LLM-generated content at query #11
#--------------------------

```python
def test_someprice_constructor():
    ccy = Currency("USD")
    qty = Decimal("100.50")
    dov = Date(2023, 1, 1)
    price = SomePrice(ccy, qty, dov)
    assert price.ccy == ccy
    assert price.qty == qty
    assert price.dov == dov
    assert price.defined is True
    assert price.undefined is False


# LLM-generated content at query #12
#--------------------------

```python
def test_price_ge_defined_vs_defined():
    from pypara.currencies import Currencies
    price1 = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('5'), Date(2019, 1, 2))
    assert price1 >= price2

def test_price_ge_defined_vs_undefined():
    from pypara.currencies import Currencies
    price1 = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    price2 = Price.na()
    assert price1 >= price2

def test_price_ge_undefined_vs_defined():
    price1 = Price.na()
    price2 = Price.of(Currencies["USD"], Decimal('5'), Date(2019, 1, 2))
    assert not (price1 >= price2)

def test_price_ge_undefined_vs_undefined():
    price1 = Price.na()
    price2 = Price.na()
    assert price1 >= price2

def test_price_ge_incompatible_currency():
    from pypara.currencies import Currencies
    from pypara.errors import IncompatibleCurrencyError
    price1 = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["EUR"], Decimal('5'), Date(2019, 1, 2))
    try:
        assert price1 >= price2
    except IncompatibleCurrencyError:
        pass


# LLM-generated content at query #13
#--------------------------

```python
def test_multiply_defined_price():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    price = Price.of(Currencies["USD"], Decimal('2.5'), date(2019, 1, 1))
    result = price.multiply(2)
    assert result.qty == Decimal('5.0')
    assert result.ccy == Currencies["USD"]
    assert result.dov == date(2019, 1, 1)

def test_multiply_undefined_price():
    undefined_price = Price.na()
    result = undefined_price.multiply(5)
    assert result.undefined


# LLM-generated content at query #14
#--------------------------

```python
def test_floor_divide_defined_money():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date

    usd = Currencies["USD"]
    money = Money.of(usd, Decimal('10'), date(2019, 1, 1))
    result = money.floor_divide(Decimal('3'))
    assert result.qty == Decimal('3')
    assert result.ccy == usd
    assert result.dov == date(2019, 1, 1)

def test_floor_divide_undefined_money():
    undefined_money = Money.na()
    result = undefined_money.floor_divide(Decimal('3'))
    assert result is undefined_money

def test_floor_divide_by_zero():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date

    usd = Currencies["USD"]
    money = Money.of(usd, Decimal('10'), date(2019, 1, 1))
    result = money.floor_divide(0)
    assert result is Money.na()


# LLM-generated content at query #15
#--------------------------

```python
def test_lt_with_same_currency():
    usd = Currency("USD", 2)
    money1 = SomeMoney(usd, Decimal("100.00"), Date(2023, 1, 1))
    money2 = SomeMoney(usd, Decimal("200.00"), Date(2023, 1, 2))
    assert money1 < money2
    assert not (money2 < money1)

def test_lt_with_different_currency():
    usd = Currency("USD", 2)
    eur = Currency("EUR", 2)
    money1 = SomeMoney(usd, Decimal("100.00"), Date(2023, 1, 1))
    money2 = SomeMoney(eur, Decimal("200.00"), Date(2023, 1, 2))
    try:
        money1 < money2
        assert False, "Expected IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass

def test_lt_with_non_somemoney_object():
    usd = Currency("USD", 2)
    money = SomeMoney(usd, Decimal("100.00"), Date(2023, 1, 1))
    assert not (money < "not a money object")


# LLM-generated content at query #16
#--------------------------

```python
def test_subtract_same_currency():
    usd = Currency("USD", 2)
    money1 = SomeMoney(usd, Decimal("100.50"), Date(2023, 1, 1))
    money2 = SomeMoney(usd, Decimal("50.25"), Date(2023, 1, 2))
    result = money1 - money2
    assert result == SomeMoney(usd, Decimal("50.25"), Date(2023, 1, 1))

def test_subtract_different_currency_raises_error():
    usd = Currency("USD", 2)
    eur = Currency("EUR", 2)
    money1 = SomeMoney(usd, Decimal("100.50"), Date(2023, 1, 1))
    money2 = SomeMoney(eur, Decimal("50.25"), Date(2023, 1, 2))
    try:
        result = money1 - money2
        assert False, "Expected IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass

def test_subtract_undefined_money():
    usd = Currency("USD", 2)
    money1 = SomeMoney(usd, Decimal("100.50"), Date(2023, 1, 1))
    money2 = NoMoney
    result = money1 - money2
    assert result == money1


# LLM-generated content at query #17
#--------------------------

```python
def test_qty_or_zero_defined_price():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    price = Price.of(Currencies["USD"], Decimal('1.5'), date(2019, 1, 1))
    assert price.qty_or_zero() == Decimal('1.5')

def test_qty_or_zero_undefined_price():
    price = Price.na()
    assert price.qty_or_zero() == Decimal('0')


# LLM-generated content at query #18
#--------------------------

```python
def test_mul_defined_price_with_scalar():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    price = Price.of(Currencies["USD"], Decimal('10.5'), date(2019, 1, 1))
    result = price * 2
    assert result.qty == Decimal('21.0')
    assert result.ccy == Currencies["USD"]
    assert result.dov == date(2019, 1, 1)

def test_mul_undefined_price_with_scalar():
    price = Price.na()
    result = price * 5
    assert result.undefined


# LLM-generated content at query #19
#--------------------------

```python
def test_abs_defined_price():
    from pypara.currencies import Currencies
    price = Price.of(Currencies["USD"], Decimal('-1.5'), Date(2019, 1, 1))
    assert price.abs().qty == Decimal('1.5')
    assert price.abs().ccy == Currencies["USD"]
    assert price.abs().dov == Date(2019, 1, 1)

def test_abs_undefined_price():
    price = Price.na()
    assert price.abs() is price


# LLM-generated content at query #20
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


# LLM-generated content at query #21
#--------------------------

```python
def test_qty_or_zero_defined_money():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    money = Money.of(Currencies["USD"], Decimal('1.00'), date(2019, 1, 1))
    assert money.qty_or_zero() == Decimal('1.00')

def test_qty_or_zero_undefined_money():
    money = Money.na()
    assert money.qty_or_zero() == Decimal('0')


# LLM-generated content at query #22
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


# LLM-generated content at query #23
#--------------------------

```python
def test_as_integer_defined_money():
    from pypara.currencies import Currencies
    money = Money.of(Currencies["USD"], Decimal('10.50'), Date(2019, 1, 1))
    assert money.as_integer() == 10

def test_as_integer_undefined_money():
    money = Money.na()
    try:
        money.as_integer()
        assert False, "Expected MonetaryOperationException"
    except MonetaryOperationException:
        pass


# LLM-generated content at query #24
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

def test_gt_defined_same_currency():
    price1 = Price.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert price1.gt(price2)

def test_gt_defined_different_currency():
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


# LLM-generated content at query #25
#--------------------------

```python
def test_bool_defined_nonzero_money():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    money = Money.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    assert bool(money) is True

def test_bool_defined_zero_money():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    money = Money.of(Currencies["USD"], Decimal('0'), date(2019, 1, 1))
    assert bool(money) is False

def test_bool_undefined_money():
    money = Money.na()
    assert bool(money) is False


# LLM-generated content at query #26
#--------------------------

```python
def test_floor_divide_defined_price():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    price = Price.of(Currencies["USD"], Decimal('10.5'), date(2019, 1, 1))
    result = price // Decimal('3')
    assert result.qty == Decimal('3')

def test_floor_divide_undefined_price():
    price = Price.na()
    result = price // Decimal('3')
    assert result.undefined

def test_floor_divide_by_zero():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    price = Price.of(Currencies["USD"], Decimal('10.5'), date(2019, 1, 1))
    result = price // Decimal('0')
    assert result.undefined


# LLM-generated content at query #27
#--------------------------

```python
def test_price_le_defined_vs_undefined():
    from pypara.currencies import Currencies
    from pypara.price import Price
    from datetime import date

    usd_price = Price.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    undefined_price = Price.na()

    assert usd_price <= undefined_price is False
    assert undefined_price <= usd_price is True

def test_price_le_same_currency():
    from pypara.currencies import Currencies
    from pypara.price import Price
    from datetime import date

    price1 = Price.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('2'), date(2019, 1, 1))
    price3 = Price.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))

    assert price1 <= price2 is True
    assert price2 <= price1 is False
    assert price1 <= price3 is True

def test_price_le_different_currency():
    from pypara.currencies import Currencies
    from pypara.price import Price
    from datetime import date
    from pypara.errors import IncompatibleCurrencyError

    usd_price = Price.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    eur_price = Price.of(Currencies["EUR"], Decimal('1'), date(2019, 1, 1))

    try:
        usd_price <= eur_price
        assert False, "Expected IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass


# LLM-generated content at query #28
#--------------------------

```python
def test_int_defined_money():
    from pypara.currencies import Currencies
    money = Money.of(Currencies["USD"], Decimal('123.45'), Date(2019, 1, 1))
    assert int(money) == 123

def test_int_undefined_money():
    money = Money.na()
    assert int(money) == 0


# LLM-generated content at query #29
#--------------------------

```python
def test_lt_undefined_money_is_less_than_defined_money():
    from pypara.currencies import Currencies
    undefined_money = Money.na()
    defined_money = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert undefined_money.lt(defined_money) is True

def test_lt_defined_money_is_not_less_than_undefined_money():
    from pypara.currencies import Currencies
    defined_money = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    undefined_money = Money.na()
    assert defined_money.lt(undefined_money) is False

def test_lt_defined_money_with_same_currency_and_smaller_quantity():
    from pypara.currencies import Currencies
    money1 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    assert money1.lt(money2) is True

def test_lt_defined_money_with_same_currency_and_larger_quantity():
    from pypara.currencies import Currencies
    money1 = Money.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert money1.lt(money2) is False

def test_lt_defined_money_with_same_currency_and_equal_quantity():
    from pypara.currencies import Currencies
    money1 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert money1.lt(money2) is False

def test_lt_defined_money_with_different_currency_raises_error():
    from pypara.currencies import Currencies
    from pypara.errors import IncompatibleCurrencyError
    money1 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["EUR"], Decimal('1'), Date(2019, 1, 1))
    try:
        money1.lt(money2)
        assert False, "Expected IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass


# LLM-generated content at query #30
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

def test_mul_undefined_money_with_scalar():
    result = Money.na() * 5
    assert result.undefined

def test_mul_defined_money_with_zero():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    money = Money.of(Currencies["USD"], Decimal('10.50'), date(2023, 1, 1))
    result = money * 0
    assert result.qty == Decimal('0.00')
    assert result.ccy == Currencies["USD"]
    assert result.dov == date(2023, 1, 1)

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


# LLM-generated content at query #31
#--------------------------

```python
def test_dov_or_defined_price():
    from pypara.currencies import Currencies
    from datetime import date
    from decimal import Decimal
    price = Price.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    assert price.dov_or(date(2001, 1, 1)) == date(2019, 1, 1)

def test_dov_or_undefined_price():
    from datetime import date
    price = Price.na()
    assert price.dov_or(date(2001, 1, 1)) == date(2001, 1, 1)


# LLM-generated content at query #32
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
    nonemoney = Money.na()
    assert nonemoney.or_else(lambda: fallback) is fallback


# LLM-generated content at query #33
#--------------------------

```python
def test_gte_with_defined_prices_same_currency():
    from pypara.currencies import Currencies
    price1 = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('5'), Date(2019, 1, 2))
    assert price1.gte(price2) is True

def test_gte_with_defined_prices_different_currency():
    from pypara.currencies import Currencies
    price1 = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["EUR"], Decimal('5'), Date(2019, 1, 2))
    try:
        price1.gte(price2)
        assert False, "Expected IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass

def test_gte_with_undefined_price_and_defined_price():
    price1 = Price.na()
    price2 = Price.of(Currencies["USD"], Decimal('5'), Date(2019, 1, 2))
    assert price1.gte(price2) is False

def test_gte_with_defined_price_and_undefined_price():
    from pypara.currencies import Currencies
    price1 = Price.of(Currencies["USD"], Decimal('5'), Date(2019, 1, 1))
    price2 = Price.na()
    assert price1.gte(price2) is True

def test_gte_with_both_undefined_prices():
    price1 = Price.na()
    price2 = Price.na()
    assert price1.gte(price2) is True

def test_gte_with_equal_defined_prices():
    from pypara.currencies import Currencies
    price1 = Price.of(Currencies["USD"], Decimal('5'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('5'), Date(2019, 1, 2))
    assert price1.gte(price2) is True


# LLM-generated content at query #34
#--------------------------

```python
def test_dov_or_defined_money():
    from pypara.currencies import Currencies
    from datetime import date
    somemoney = Money.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    assert somemoney.dov_or(date(2001, 1, 1)) == date(2019, 1, 1)

def test_dov_or_undefined_money():
    from pypara.currencies import Currencies
    from datetime import date
    nonemoney = Money.of(None, None, date(2019, 1, 1))
    assert nonemoney.dov_or(date(2001, 1, 1)) == date(2001, 1, 1)


# LLM-generated content at query #35
#--------------------------

```python
def test_pos_defined_price():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    price = Price.of(Currencies["USD"], Decimal('1.5'), date(2019, 1, 1))
    result = +price
    assert result.is_equal(price)

def test_pos_undefined_price():
    undefined_price = Price.na()
    result = +undefined_price
    assert result.is_equal(undefined_price)


# LLM-generated content at query #36
#--------------------------

```python
def test_or_else_returns_itself_when_defined():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    someprice = Price.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    fallback = Price.of(Currencies["EUR"], Decimal('2'), date(2019, 1, 2))
    assert someprice.or_else(lambda: fallback) is someprice

def test_or_else_returns_fallback_when_undefined():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    fallback = Price.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    noneprice = Price.of(None, Decimal('1'), None)
    assert noneprice.or_else(lambda: fallback) is fallback


# LLM-generated content at query #37
#--------------------------

```python
def test_qty_or_none_defined_money():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    money = Money.of(Currencies["USD"], Decimal('10.50'), date(2023, 1, 1))
    assert money.qty_or_none() == Decimal('10.50')

def test_qty_or_none_undefined_money():
    money = Money.na()
    assert money.qty_or_none() is None


# LLM-generated content at query #38
#--------------------------

```python
def test_price_le_undefined_vs_defined():
    undefined_price = Price.na()
    defined_price = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert undefined_price <= defined_price

def test_price_le_defined_vs_undefined():
    defined_price = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    undefined_price = Price.na()
    assert not defined_price <= undefined_price

def test_price_le_same_currency():
    price1 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    assert price1 <= price2
    assert not price2 <= price1

def test_price_le_equal_prices():
    price1 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert price1 <= price2
    assert price2 <= price1

def test_price_le_different_currency():
    price1 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["EUR"], Decimal('1'), Date(2019, 1, 1))
    try:
        price1 <= price2
        assert False, "Expected IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass


