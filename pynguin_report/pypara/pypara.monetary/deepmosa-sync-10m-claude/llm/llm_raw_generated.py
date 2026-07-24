####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_is_equal():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    
    # Test equality of two defined money objects with same values
    money1 = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    assert money1.is_equal(money2) is True
    
    # Test inequality of two defined money objects with different quantities
    money3 = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    money4 = Money.of(Currencies["USD"], Decimal('200'), Date(2019, 1, 1))
    assert money3.is_equal(money4) is False
    
    # Test inequality of two defined money objects with different currencies
    money5 = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    money6 = Money.of(Currencies["EUR"], Decimal('100'), Date(2019, 1, 1))
    assert money5.is_equal(money6) is False
    
    # Test inequality of two defined money objects with different dates
    money7 = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    money8 = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 2))
    assert money7.is_equal(money8) is False
    
    # Test equality of two undefined money objects
    money9 = Money.na()
    money10 = Money.na()
    assert money9.is_equal(money10) is True
    
    # Test inequality of defined and undefined money objects
    money11 = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    money12 = Money.na()
    assert money11.is_equal(money12) is False
    
    # Test inequality when comparing with non-Money object
    money13 = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    assert money13.is_equal("not a money object") is False
    assert money13.is_equal(100) is False
    assert money13.is_equal(None) is False


# LLM-generated content at query #2
#--------------------------

```python
def test_dov_or():
    from datetime import date
    from decimal import Decimal
    
    ccy = Currency(code="USD", quantizer=Decimal("0.01"))
    qty = Decimal("100.00")
    dov = date(2023, 1, 15)
    default_date = date(2024, 12, 31)
    
    price = SomePrice(ccy=ccy, qty=qty, dov=dov)
    
    result = price.dov_or(default_date)
    
    assert result == dov
    assert result == date(2023, 1, 15)


# LLM-generated content at query #3
#--------------------------

```python
def test_someprice_int():
    from decimal import Decimal
    from datetime import date
    
    # Create a Currency mock
    class MockCurrency:
        def __init__(self, code):
            self.code = code
            self.quantizer = Decimal('0.01')
    
    ccy = MockCurrency('USD')
    qty = Decimal('123.456')
    dov = date(2024, 1, 1)
    
    price = SomePrice(ccy, qty, dov)
    result = int(price)
    
    assert result == 123
    assert isinstance(result, int)


# LLM-generated content at query #4
#--------------------------

```python
def test_money_float_conversion():
    from decimal import Decimal
    from datetime import date as Date
    from pypara.currencies import Currencies
    from pypara.money import Money
    
    # Test __float__ on defined money
    defined_money = Money.of(Currencies["USD"], Decimal('123.45'), Date(2019, 1, 1))
    result = float(defined_money)
    assert isinstance(result, float)
    assert result == 123.45
    
    # Test __float__ on undefined money should raise exception
    undefined_money = Money.na()
    try:
        float(undefined_money)
        assert False, "Expected MonetaryOperationException"
    except Exception as e:
        assert "MonetaryOperationException" in str(type(e).__name__)
    
    # Test __float__ with negative quantity
    negative_money = Money.of(Currencies["EUR"], Decimal('-99.99'), Date(2020, 6, 15))
    result = float(negative_money)
    assert result == -99.99
    
    # Test __float__ with zero quantity
    zero_money = Money.of(Currencies["GBP"], Decimal('0'), Date(2021, 12, 31))
    result = float(zero_money)
    assert result == 0.0


# LLM-generated content at query #5
#--------------------------

```python
def test_price_is_equal():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    # Create two identical defined prices
    price1 = Price.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    assert price1.is_equal(price2) is True
    
    # Create two undefined prices
    price3 = Price.na()
    price4 = Price.na()
    assert price3.is_equal(price4) is True
    
    # Compare same instance with itself
    assert price1.is_equal(price1) is True
    
    # Compare prices with different quantities
    price5 = Price.of(Currencies["USD"], Decimal('200'), Date(2019, 1, 1))
    assert price1.is_equal(price5) is False
    
    # Compare prices with different currencies
    price6 = Price.of(Currencies["EUR"], Decimal('100'), Date(2019, 1, 1))
    assert price1.is_equal(price6) is False
    
    # Compare prices with different dates
    price7 = Price.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 2))
    assert price1.is_equal(price7) is False
    
    # Compare defined price with undefined price
    assert price1.is_equal(price3) is False
    
    # Compare with non-price object
    assert price1.is_equal("not a price") is False
    assert price1.is_equal(100) is False
    assert price1.is_equal(None) is False


# LLM-generated content at query #6
#--------------------------

```python
def test_ccy_or_none():
    from datetime import date as Date
    from decimal import Decimal
    from pypara.currencies import Currencies
    from pypara.money import Money
    
    # Test with defined money - should return the currency
    somemoney = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    result = somemoney.ccy_or_none()
    assert result is not None
    assert result.code == "USD"
    
    # Test with undefined money - should return None
    nonemoney = Money.of(Currencies["USD"], None, None)
    result = nonemoney.ccy_or_none()
    assert result is None
    
    # Test with Money.na() - should return None
    result = Money.na().ccy_or_none()
    assert result is None


# LLM-generated content at query #7
#--------------------------

```python
def test_price_floordiv():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.price import Price
    
    # Test floor division with defined price
    price1 = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result1 = price1 // Decimal('3')
    assert result1.qty_or_zero() == Decimal('3')
    assert result1.ccy_or_none().code == "USD"
    
    # Test floor division with zero divisor returns undefined price
    price2 = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result2 = price2 // Decimal('0')
    assert result2.undefined
    
    # Test floor division with undefined price returns undefined
    price3 = Price.na()
    result3 = price3 // Decimal('5')
    assert result3.undefined
    
    # Test floor division with negative divisor
    price4 = Price.of(Currencies["EUR"], Decimal('20'), Date(2019, 1, 1))
    result4 = price4 // Decimal('-3')
    assert result4.qty_or_zero() == Decimal('-7')
    assert result4.ccy_or_none().code == "EUR"
    
    # Test floor division with decimal divisor
    price5 = Price.of(Currencies["GBP"], Decimal('7'), Date(2019, 1, 1))
    result5 = price5 // Decimal('2.5')
    assert result5.qty_or_zero() == Decimal('2')
    assert result5.ccy_or_none().code == "GBP"


# LLM-generated content at query #8
#--------------------------

```python
def test_price_negative():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    # Test negative on defined price with positive quantity
    price_positive = Price.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    price_negated = price_positive.negative()
    assert price_negated.qty_or_zero() == Decimal('-100')
    assert price_negated.ccy_or_none().code == "USD"
    assert price_negated.dov_or_none() == Date(2019, 1, 1)
    
    # Test negative on defined price with negative quantity
    price_negative = Price.of(Currencies["USD"], Decimal('-50'), Date(2019, 1, 1))
    price_negated_again = price_negative.negative()
    assert price_negated_again.qty_or_zero() == Decimal('50')
    
    # Test negative on defined price with zero quantity
    price_zero = Price.of(Currencies["USD"], Decimal('0'), Date(2019, 1, 1))
    price_negated_zero = price_zero.negative()
    assert price_negated_zero.qty_or_zero() == Decimal('0')
    
    # Test negative on undefined price returns itself
    price_undefined = Price.na()
    price_negated_undefined = price_undefined.negative()
    assert price_negated_undefined.undefined
    assert price_negated_undefined is price_undefined


# LLM-generated content at query #9
#--------------------------

```python
def test_with_dov():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    
    # Test with_dov on defined money
    defined_money = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    new_date = Date(2020, 6, 15)
    result = defined_money.with_dov(new_date)
    assert result.dov_or_none() == new_date
    assert result.ccy_or_none().code == "USD"
    assert result.qty_or_none() == Decimal('1.00')
    
    # Test with_dov on undefined money
    undefined_money = Money.na()
    result_undefined = undefined_money.with_dov(Date(2020, 6, 15))
    assert result_undefined.undefined
    assert result_undefined is undefined_money


# LLM-generated content at query #10
#--------------------------

```python
def test_price_add_defined_prices_same_currency():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    price1 = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('5'), Date(2019, 1, 2))
    
    result = price1.add(price2)
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('15')
    assert result.ccy_or_none().code == "USD"


def test_price_add_defined_with_undefined():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    price1 = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    price2 = Price.na()
    
    result = price1.add(price2)
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('10')


def test_price_add_undefined_with_defined():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    price1 = Price.na()
    price2 = Price.of(Currencies["USD"], Decimal('5'), Date(2019, 1, 1))
    
    result = price1.add(price2)
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('5')


def test_price_add_two_undefined():
    price1 = Price.na()
    price2 = Price.na()
    
    result = price1.add(price2)
    
    assert result.undefined


def test_price_add_negative_quantities():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    price1 = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('-3'), Date(2019, 1, 2))
    
    result = price1.add(price2)
    
    assert result.qty_or_zero() == Decimal('7')


def test_price_add_decimal_quantities():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    price1 = Price.of(Currencies["USD"], Decimal('10.5'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('5.25'), Date(2019, 1, 2))
    
    result = price1.add(price2)
    
    assert result.qty_or_zero() == Decimal('15.75')


# LLM-generated content at query #11
#--------------------------

```python
def test_price_bool():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.price import Price
    
    # Test defined price with non-zero quantity returns True
    defined_price = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert bool(defined_price) is True
    
    # Test defined price with zero quantity returns False
    zero_price = Price.of(Currencies["USD"], Decimal('0'), Date(2019, 1, 1))
    assert bool(zero_price) is False
    
    # Test undefined price returns False
    undefined_price = Price.na()
    assert bool(undefined_price) is False
    
    # Test defined price with negative quantity returns True
    negative_price = Price.of(Currencies["USD"], Decimal('-5'), Date(2019, 1, 1))
    assert bool(negative_price) is True
    
    # Test defined price with positive quantity returns True
    positive_price = Price.of(Currencies["EUR"], Decimal('100'), Date(2020, 6, 15))
    assert bool(positive_price) is True


# LLM-generated content at query #12
#--------------------------

```python
def test_price_float():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.price import Price
    
    # Test that __float__ returns float representation of a defined price
    defined_price = Price.of(Currencies["USD"], Decimal('123.45'), Date(2019, 1, 1))
    assert float(defined_price) == 123.45
    assert isinstance(float(defined_price), float)
    
    # Test that __float__ works with negative quantities
    negative_price = Price.of(Currencies["USD"], Decimal('-50.25'), Date(2019, 1, 1))
    assert float(negative_price) == -50.25
    
    # Test that __float__ works with zero quantity
    zero_price = Price.of(Currencies["USD"], Decimal('0'), Date(2019, 1, 1))
    assert float(zero_price) == 0.0
    
    # Test that __float__ works with large numbers
    large_price = Price.of(Currencies["USD"], Decimal('999999.99'), Date(2019, 1, 1))
    assert float(large_price) == 999999.99
    
    # Test that __float__ works with small decimal places
    small_price = Price.of(Currencies["USD"], Decimal('0.001'), Date(2019, 1, 1))
    assert float(small_price) == 0.001


# LLM-generated content at query #13
#--------------------------

```python
def test_dov_or_none():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.money import Money

    # Test with defined money object
    somemoney = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    result = somemoney.dov_or_none()
    assert result == Date(2019, 1, 1)

    # Test with undefined money object
    nonemoney = Money.of(None, None, Date(2019, 1, 1))
    result = nonemoney.dov_or_none()
    assert result is None

    # Test with Money.na()
    undefined_money = Money.na()
    result = undefined_money.dov_or_none()
    assert result is None


# LLM-generated content at query #14
#--------------------------

```python
def test_noneprice_constructor():
    none_price = NonePrice()
    assert none_price is not None
    assert isinstance(none_price, NonePrice)


# LLM-generated content at query #15
#--------------------------

```python
def test_price_add():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.price import Price
    
    # Test adding two defined prices with same currency
    price1 = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('5'), Date(2019, 1, 2))
    result = price1.add(price2)
    assert result.defined
    assert result.qty_or_zero() == Decimal('15')
    assert result.ccy_or_none().code == "USD"
    
    # Test adding undefined price to defined price
    price3 = Price.of(Currencies["USD"], Decimal('20'), Date(2019, 1, 3))
    undefined_price = Price.na()
    result2 = price3.add(undefined_price)
    assert result2 is price3
    
    # Test adding defined price to undefined price
    result3 = undefined_price.add(price3)
    assert result3 is price3
    
    # Test adding two undefined prices
    result4 = undefined_price.add(undefined_price)
    assert result4.undefined
    
    # Test adding prices with different currencies raises error
    price_eur = Price.of(Currencies["EUR"], Decimal('10'), Date(2019, 1, 1))
    try:
        price1.add(price_eur)
        assert False, "Should raise IncompatibleCurrencyError"
    except Exception:
        pass


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_somemoney_constructor():
    from decimal import Decimal
    from datetime import date
    
    # Create a Currency mock object
    class MockCurrency:
        def __init__(self, code, decimals=2):
            self.code = code
            self.decimals = decimals
            self.quantizer = Decimal(10) ** -decimals
    
    # Create instances with valid parameters
    ccy = MockCurrency("USD", 2)
    qty = Decimal("100.50")
    dov = date(2024, 1, 15)
    
    # Test constructor
    money = SomeMoney(ccy, qty, dov)
    
    # Verify the attributes are set correctly
    assert money.ccy == ccy
    assert money.qty == qty
    assert money.dov == dov
    
    # Verify it's a tuple-like object
    assert money[0] == ccy
    assert money[1] == qty
    assert money[2] == dov
    
    # Verify defined property
    assert money.defined is True
    
    # Verify undefined property
    assert money.undefined is False
    
    # Test with different currency and amounts
    ccy2 = MockCurrency("EUR", 2)
    qty2 = Decimal("50.25")
    dov2 = date(2024, 6, 30)
    
    money2 = SomeMoney(ccy2, qty2, dov2)
    assert money2.ccy == ccy2
    assert money2.qty == qty2
    assert money2.dov == dov2


# LLM-generated content at query #2
#--------------------------

```python
def test_dimap_with_defined_money():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from decimal import Decimal
    from datetime import date as Date
    
    somemoney = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    result = somemoney.dimap(lambda x: x.ccy.code, lambda: "EUR")
    assert result == "USD"


def test_dimap_with_undefined_money():
    from pypara.money import Money
    from decimal import Decimal
    
    nonemoney = Money.of(None, Decimal('1'), None)
    result = nonemoney.dimap(lambda x: x.ccy.code, lambda: "EUR")
    assert result == "EUR"


def test_dimap_applies_function_to_defined_money():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from decimal import Decimal
    from datetime import date as Date
    
    somemoney = Money.of(Currencies["USD"], Decimal('42'), Date(2020, 5, 15))
    result = somemoney.dimap(lambda x: x.qty, lambda: Decimal('0'))
    assert result == Decimal('42.00')


def test_dimap_uses_combinator_for_undefined_money():
    from pypara.money import Money
    from decimal import Decimal
    
    nonemoney = Money.of(None, Decimal('100'), None)
    result = nonemoney.dimap(lambda x: x.qty + Decimal('50'), lambda: Decimal('999'))
    assert result == Decimal('999')


def test_dimap_with_date_extraction():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from decimal import Decimal
    from datetime import date as Date
    
    somemoney = Money.of(Currencies["EUR"], Decimal('5'), Date(2021, 12, 25))
    result = somemoney.dimap(lambda x: x.dov, lambda: Date(2000, 1, 1))
    assert result == Date(2021, 12, 25)


def test_dimap_undefined_money_with_date_combinator():
    from pypara.money import Money
    from decimal import Decimal
    from datetime import date as Date
    
    nonemoney = Money.of(None, Decimal('1'), None)
    result = nonemoney.dimap(lambda x: x.dov, lambda: Date(2000, 1, 1))
    assert result == Date(2000, 1, 1)


# LLM-generated content at query #3
#--------------------------

```python
def test_someprice_add():
    from decimal import Decimal
    from datetime import date
    
    # Create test currencies and prices
    usd = Currency(code="USD", quantizer=Decimal("0.01"))
    eur = Currency(code="EUR", quantizer=Decimal("0.01"))
    
    price1 = SomePrice(ccy=usd, qty=Decimal("100.00"), dov=date(2024, 1, 1))
    price2 = SomePrice(ccy=usd, qty=Decimal("50.00"), dov=date(2024, 1, 2))
    price_different_ccy = SomePrice(ccy=eur, qty=Decimal("50.00"), dov=date(2024, 1, 1))
    
    # Test adding two prices with same currency
    result = price1 + price2
    assert result.ccy == usd
    assert result.qty == Decimal("150.00")
    assert result.dov == date(2024, 1, 2)
    
    # Test adding prices with different currencies raises exception
    try:
        price1 + price_different_ccy
        assert False, "Should raise IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass
    
    # Test adding undefined price returns self
    no_price = NoPrice
    result = price1 + no_price
    assert result == price1
    
    # Test adding with date selection (newer date)
    price3 = SomePrice(ccy=usd, qty=Decimal("25.00"), dov=date(2024, 1, 3))
    result = price1 + price3
    assert result.dov == date(2024, 1, 3)
    
    # Test adding with date selection (older date)
    price4 = SomePrice(ccy=usd, qty=Decimal("75.00"), dov=date(2023, 12, 31))
    result = price1 + price4
    assert result.dov == date(2024, 1, 1)


# LLM-generated content at query #4
#--------------------------

```python
def test_money_floordiv():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    
    # Test floor division with defined money
    money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = money.__floordiv__(Decimal('3'))
    assert result.qty == Decimal('3')
    assert result.ccy == Currencies["USD"]
    
    # Test floor division with zero divisor returns undefined money
    money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = money.__floordiv__(Decimal('0'))
    assert result.undefined
    
    # Test floor division with undefined money returns itself
    money = Money.na()
    result = money.__floordiv__(Decimal('5'))
    assert result.undefined
    
    # Test floor division with negative divisor
    money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = money.__floordiv__(Decimal('-3'))
    assert result.qty == Decimal('-4')
    
    # Test floor division with decimal divisor
    money = Money.of(Currencies["USD"], Decimal('10.5'), Date(2019, 1, 1))
    result = money.__floordiv__(Decimal('2.5'))
    assert result.qty == Decimal('4')


# LLM-generated content at query #5
#--------------------------

```python
def test_convert_with_valid_rate():
    from datetime import date
    from decimal import Decimal
    from pypara.currencies import Currency, CurrencyType
    from pypara.monetary import SomePrice
    from pypara.fxrate import FXRate, FXRateService
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    
    price = SomePrice(usd, Decimal("100.00"), date(2023, 1, 1))
    
    fx_rate = FXRate(usd, eur, Decimal("0.85"), date(2023, 1, 1))
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return fx_rate
    
    original_default = FXRateService.default
    FXRateService.default = MockFXRateService()
    
    try:
        result = price.convert(eur)
        assert result.ccy == eur
        assert result.qty == Decimal("85.00")
        assert result.dov == date(2023, 1, 1)
    finally:
        FXRateService.default = original_default


def test_convert_with_custom_asof_date():
    from datetime import date
    from decimal import Decimal
    from pypara.currencies import Currency, CurrencyType
    from pypara.monetary import SomePrice
    from pypara.fxrate import FXRate, FXRateService
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    gbp = Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY)
    
    price = SomePrice(usd, Decimal("100.00"), date(2023, 1, 1))
    
    fx_rate = FXRate(usd, gbp, Decimal("0.73"), date(2023, 6, 1))
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return fx_rate
    
    original_default = FXRateService.default
    FXRateService.default = MockFXRateService()
    
    try:
        result = price.convert(gbp, asof=date(2023, 6, 1))
        assert result.ccy == gbp
        assert result.qty == Decimal("73.00")
        assert result.dov == date(2023, 6, 1)
    finally:
        FXRateService.default = original_default


def test_convert_with_no_rate_non_strict():
    from datetime import date
    from decimal import Decimal
    from pypara.currencies import Currency, CurrencyType
    from pypara.monetary import SomePrice, NoPrice
    from pypara.fxrate import FXRateService
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    
    price = SomePrice(usd, Decimal("100.00"), date(2023, 1, 1))
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
    
    original_default = FXRateService.default
    FXRateService.default = MockFXRateService()
    
    try:
        result = price.convert(jpy, strict=False)
        assert result == NoPrice
    finally:
        FXRateService.default = original_default


def test_convert_with_no_rate_strict():
    from datetime import date
    from decimal import Decimal
    from pypara.currencies import Currency, CurrencyType
    from pypara.monetary import SomePrice
    from pypara.fxrate import FXRateService, FXRateLookupError
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    chf = Currency.of("CHF", "Swiss Franc", 2, CurrencyType.MONEY)
    
    price = SomePrice(usd, Decimal("100.00"), date(2023, 1, 1))
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
    
    original_default = FXRateService.default
    FXRateService.default = MockFXRateService()
    
    try:
        error_raised = False
        try:
            price.convert(chf, strict=True)
        except FXRateLookupError:
            error_raised = True
        assert error_raised
    finally:
        FXRateService.default = original_default


def test_convert_with_no_default_service():
    from datetime import date
    from decimal import Decimal
    from pypara.currencies import Currency, CurrencyType
    from pypara.monetary import SomePrice
    from pypara.fxrate import FXRateService
    from pypara.errors import ProgrammingError
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    
    price = SomePrice(usd, Decimal("100.00"), date(2023, 1, 1))
    
    original_default = FXRateService.default
    FXRateService.default = None
    
    try:
        error_raised = False
        try:
            price.convert(eur)
        except ProgrammingError:
            error_raised = True
        assert error_raised
    finally:
        FXRateService.default = original_default


# LLM-generated content at query #6
#--------------------------

```python
def test_qty_or_none():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    
    # Test with defined money
    somemoney = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    result = somemoney.qty_or_none()
    assert result == Decimal('1.00')
    
    # Test with undefined money
    nonemoney = Money.of(None, Decimal('1'), None)
    result = nonemoney.qty_or_none()
    assert result is None
    
    # Test with Money.na()
    na_money = Money.na()
    result = na_money.qty_or_none()
    assert result is None


# LLM-generated content at query #7
#--------------------------

```python
def test_price_or_else_returns_self_when_defined():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    fallback = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    someprice = Price.of(Currencies["EUR"], Decimal('2'), Date(2019, 1, 2))
    result = someprice.or_else(lambda: fallback)
    
    assert result is someprice


def test_price_or_else_returns_fallback_when_undefined():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    fallback = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    noneprice = Price.na()
    result = noneprice.or_else(lambda: fallback)
    
    assert result is fallback


def test_price_or_else_fallback_not_called_when_defined():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    someprice = Price.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 2))
    fallback_called = []
    
    def fallback():
        fallback_called.append(True)
        return Price.na()
    
    result = someprice.or_else(fallback)
    
    assert len(fallback_called) == 0
    assert result is someprice


def test_price_or_else_fallback_called_when_undefined():
    from datetime import date as Date
    from decimal import Decimal
    
    fallback_price = Price.na()
    noneprice = Price.na()
    fallback_called = []
    
    def fallback():
        fallback_called.append(True)
        return fallback_price
    
    result = noneprice.or_else(fallback)
    
    assert len(fallback_called) == 1
    assert result is fallback_price


# LLM-generated content at query #8
#--------------------------

```python
def test_price_ge_defined_prices_same_currency():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    price1 = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('5'), Date(2019, 1, 1))
    price3 = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    
    assert price1.gte(price2) == True
    assert price2.gte(price1) == False
    assert price1.gte(price3) == True


def test_price_ge_undefined_with_undefined():
    undefined_price1 = Price.na()
    undefined_price2 = Price.na()
    
    assert undefined_price1.gte(undefined_price2) == True


def test_price_ge_undefined_with_defined():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    undefined_price = Price.na()
    defined_price = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    
    assert undefined_price.gte(defined_price) == False


def test_price_ge_defined_with_undefined():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    defined_price = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    undefined_price = Price.na()
    
    assert defined_price.gte(undefined_price) == True


def test_price_ge_equal_prices():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    price1 = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    
    assert price1.gte(price2) == True
    assert price2.gte(price1) == True


# LLM-generated content at query #9
#--------------------------

```python
def test_price_floordiv():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    # Test floor division with defined prices
    price1 = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result1 = price1 // Decimal('3')
    assert result1.qty_or_zero() == Decimal('3')
    assert result1.defined
    
    # Test floor division with zero divisor returns undefined
    price2 = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result2 = price2 // Decimal('0')
    assert result2.undefined
    
    # Test floor division with undefined price returns itself
    price3 = Price.na()
    result3 = price3 // Decimal('5')
    assert result3.undefined
    
    # Test floor division with negative numbers
    price4 = Price.of(Currencies["USD"], Decimal('-10'), Date(2019, 1, 1))
    result4 = price4 // Decimal('3')
    assert result4.qty_or_zero() == Decimal('-4')
    
    # Test floor division with decimal divisor
    price5 = Price.of(Currencies["USD"], Decimal('7.5'), Date(2019, 1, 1))
    result5 = price5 // Decimal('2.5')
    assert result5.qty_or_zero() == Decimal('3')


# LLM-generated content at query #10
#--------------------------

```python
def test_qty_or():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    # Test with defined price - should return quantity
    someprice = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    result = someprice.qty_or(Decimal(0))
    assert result == Decimal('1')
    
    # Test with undefined price - should return default
    noneprice = Price.of(None, Decimal('1'), None)
    result = noneprice.qty_or(Decimal(0))
    assert result == Decimal('0')
    
    # Test with defined price and different default - should return quantity
    someprice = Price.of(Currencies["USD"], Decimal('5'), Date(2019, 1, 1))
    result = someprice.qty_or(Decimal(100))
    assert result == Decimal('5')
    
    # Test with undefined price and different default - should return default
    noneprice = Price.of(None, Decimal('1'), None)
    result = noneprice.qty_or(Decimal(42))
    assert result == Decimal('42')


# LLM-generated content at query #11
#--------------------------

```python
def test_round_defined_money():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money = Money.of(Currencies["USD"], Decimal('1.567'), Date(2019, 1, 1))
    rounded = money.round(2)
    assert rounded.qty_or_zero() == Decimal('1.57')


def test_round_undefined_money():
    money = Money.na()
    rounded = money.round(2)
    assert rounded.undefined


def test_round_zero_digits():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money = Money.of(Currencies["USD"], Decimal('1.567'), Date(2019, 1, 1))
    rounded = money.round(0)
    assert rounded.qty_or_zero() == Decimal('2')


def test_round_negative_quantity():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money = Money.of(Currencies["USD"], Decimal('-1.567'), Date(2019, 1, 1))
    rounded = money.round(2)
    assert rounded.qty_or_zero() == Decimal('-1.57')


def test_round_preserves_currency():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money = Money.of(Currencies["EUR"], Decimal('1.567'), Date(2019, 1, 1))
    rounded = money.round(2)
    assert rounded.ccy_or_none().code == "EUR"


def test_round_preserves_date():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    original_date = Date(2019, 1, 1)
    money = Money.of(Currencies["USD"], Decimal('1.567'), original_date)
    rounded = money.round(2)
    assert rounded.dov_or_none() == original_date


def test_round_already_rounded():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money = Money.of(Currencies["USD"], Decimal('1.50'), Date(2019, 1, 1))
    rounded = money.round(2)
    assert rounded.qty_or_zero() == Decimal('1.50')


# LLM-generated content at query #12
#--------------------------

```python
def test_abs_defined_positive_price():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.price import Price
    
    price = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = price.abs()
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('10')
    assert result.ccy_or_none().code == 'USD'


def test_abs_defined_negative_price():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.price import Price
    
    price = Price.of(Currencies["USD"], Decimal('-10'), Date(2019, 1, 1))
    result = price.abs()
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('10')
    assert result.ccy_or_none().code == 'USD'


def test_abs_defined_zero_price():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.price import Price
    
    price = Price.of(Currencies["USD"], Decimal('0'), Date(2019, 1, 1))
    result = price.abs()
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('0')
    assert result.ccy_or_none().code == 'USD'


def test_abs_undefined_price():
    from pypara.price import Price
    
    price = Price.na()
    result = price.abs()
    
    assert result.undefined
    assert result is price


# LLM-generated content at query #13
#--------------------------

```python
def test_somemoney_lt():
    from datetime import date
    from decimal import Decimal
    
    # Create test currencies and money objects
    usd = Currency(code="USD", decimals=2)
    eur = Currency(code="EUR", decimals=2)
    
    test_date = date(2023, 1, 1)
    
    money1 = SomeMoney(usd, Decimal("100.00"), test_date)
    money2 = SomeMoney(usd, Decimal("200.00"), test_date)
    money3 = SomeMoney(usd, Decimal("100.00"), test_date)
    money_eur = SomeMoney(eur, Decimal("100.00"), test_date)
    
    # Test: less than with smaller quantity
    assert money1 < money2
    
    # Test: not less than with equal quantity
    assert not (money1 < money3)
    
    # Test: not less than with greater quantity
    assert not (money2 < money1)
    
    # Test: less than with non-SomeMoney object returns False
    assert not (money1 < "not a money object")
    assert not (money1 < 100)
    
    # Test: incompatible currency raises exception
    try:
        money1 < money_eur
        assert False, "Should have raised IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass


# LLM-generated content at query #14
#--------------------------

```python
def test_dov_or_none():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.money import Money

    # Test with defined money - should return the dov
    somemoney = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    result = somemoney.dov_or_none()
    assert result == Date(2019, 1, 1)

    # Test with undefined money - should return None
    nonemoney = Money.of(None, None, Date(2019, 1, 1))
    result = nonemoney.dov_or_none()
    assert result is None

    # Test with undefined money created via Money.na() - should return None
    undefined_money = Money.na()
    result = undefined_money.dov_or_none()
    assert result is None


# LLM-generated content at query #15
#--------------------------

```python
def test_money_add():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    
    # Test adding two defined money objects with same currency
    money1 = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('5'), Date(2019, 1, 1))
    result = money1.add(money2)
    assert result.qty == Decimal('15.00')
    assert result.ccy.code == "USD"
    
    # Test adding defined money with undefined money
    money_defined = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    money_undefined = Money.na()
    result = money_defined.add(money_undefined)
    assert result.is_equal(money_defined)
    
    # Test adding undefined money with defined money
    money_undefined = Money.na()
    money_defined = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = money_undefined.add(money_defined)
    assert result.is_equal(money_defined)
    
    # Test adding two undefined money objects
    money_undefined1 = Money.na()
    money_undefined2 = Money.na()
    result = money_undefined1.add(money_undefined2)
    assert result.undefined
    
    # Test adding negative quantities
    money1 = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('-3'), Date(2019, 1, 1))
    result = money1.add(money2)
    assert result.qty == Decimal('7.00')
    
    # Test date is carried forward
    money1 = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('5'), Date(2019, 1, 15))
    result = money1.add(money2)
    assert result.dov == Date(2019, 1, 15)


# LLM-generated content at query #16
#--------------------------

```python
def test_scalar_add_with_defined_price():
    from decimal import Decimal
    from datetime import date as Date
    from pypara.currencies import Currencies
    from pypara.price import Price
    
    price = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = price.scalar_add(Decimal('5'))
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('15')
    assert result.ccy_or_none().code == 'USD'


def test_scalar_add_with_undefined_price():
    from decimal import Decimal
    from pypara.price import Price
    
    price = Price.na()
    result = price.scalar_add(Decimal('5'))
    
    assert result.undefined
    assert result is price


def test_scalar_add_with_negative_scalar():
    from decimal import Decimal
    from datetime import date as Date
    from pypara.currencies import Currencies
    from pypara.price import Price
    
    price = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = price.scalar_add(Decimal('-3'))
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('7')


def test_scalar_add_with_zero_scalar():
    from decimal import Decimal
    from datetime import date as Date
    from pypara.currencies import Currencies
    from pypara.price import Price
    
    price = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = price.scalar_add(Decimal('0'))
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('10')


def test_scalar_add_with_decimal_scalar():
    from decimal import Decimal
    from datetime import date as Date
    from pypara.currencies import Currencies
    from pypara.price import Price
    
    price = Price.of(Currencies["EUR"], Decimal('100'), Date(2020, 6, 15))
    result = price.scalar_add(Decimal('25.50'))
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('125.50')
    assert result.ccy_or_none().code == 'EUR'


# LLM-generated content at query #17
#--------------------------

```python
def test_floordiv_with_valid_divisor():
    from decimal import Decimal
    from datetime import date
    
    ccy = type('Currency', (), {'quantizer': Decimal('0.01')})()
    price = SomePrice(ccy, Decimal('10'), date(2024, 1, 1))
    result = price // 3
    
    assert result.qty == Decimal('3')
    assert result.ccy == ccy
    assert result.dov == date(2024, 1, 1)


def test_floordiv_with_zero_divisor():
    from decimal import Decimal
    from datetime import date
    
    ccy = type('Currency', (), {'quantizer': Decimal('0.01')})()
    price = SomePrice(ccy, Decimal('10'), date(2024, 1, 1))
    result = price // 0
    
    assert result is NoPrice


def test_floordiv_with_decimal_divisor():
    from decimal import Decimal
    from datetime import date
    
    ccy = type('Currency', (), {'quantizer': Decimal('0.01')})()
    price = SomePrice(ccy, Decimal('10.5'), date(2024, 1, 1))
    result = price // Decimal('2.5')
    
    assert result.qty == Decimal('4')
    assert result.ccy == ccy
    assert result.dov == date(2024, 1, 1)


def test_floordiv_preserves_date_of_valuation():
    from decimal import Decimal
    from datetime import date
    
    ccy = type('Currency', (), {'quantizer': Decimal('0.01')})()
    dov = date(2024, 6, 15)
    price = SomePrice(ccy, Decimal('100'), dov)
    result = price // 7
    
    assert result.dov == dov


def test_floordiv_with_string_numeric():
    from decimal import Decimal
    from datetime import date
    
    ccy = type('Currency', (), {'quantizer': Decimal('0.01')})()
    price = SomePrice(ccy, Decimal('20'), date(2024, 1, 1))
    result = price // '3'
    
    assert result.qty == Decimal('6')


# LLM-generated content at query #18
#--------------------------

```python
def test_with_qty():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    
    # Test with_qty on defined money
    money = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    new_money = money.with_qty(Decimal('200'))
    assert new_money.qty_or_none() == Decimal('200')
    assert new_money.ccy_or_none().code == 'USD'
    assert new_money.dov_or_none() == Date(2019, 1, 1)
    
    # Test with_qty on undefined money returns itself
    undefined_money = Money.na()
    result = undefined_money.with_qty(Decimal('300'))
    assert result.undefined
    
    # Test with_qty with zero quantity
    money2 = Money.of(Currencies["EUR"], Decimal('50'), Date(2020, 6, 15))
    new_money2 = money2.with_qty(Decimal('0'))
    assert new_money2.qty_or_none() == Decimal('0')
    
    # Test with_qty with negative quantity
    money3 = Money.of(Currencies["GBP"], Decimal('75'), Date(2021, 3, 20))
    new_money3 = money3.with_qty(Decimal('-100'))
    assert new_money3.qty_or_none() == Decimal('-100')


# LLM-generated content at query #19
#--------------------------

```python
def test_floor_divide_defined_price():
    from pypara.currencies import Currencies
    from pypara.price import Price
    from decimal import Decimal
    from datetime import date as Date
    
    price = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = price.floor_divide(Decimal('3'))
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('3')
    assert result.ccy_or_none().code == "USD"


def test_floor_divide_undefined_price():
    from pypara.price import Price
    from decimal import Decimal
    
    price = Price.na()
    result = price.floor_divide(Decimal('3'))
    
    assert result.undefined


def test_floor_divide_by_zero():
    from pypara.currencies import Currencies
    from pypara.price import Price
    from decimal import Decimal
    from datetime import date as Date
    
    price = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = price.floor_divide(Decimal('0'))
    
    assert result.undefined


def test_floor_divide_negative_quantity():
    from pypara.currencies import Currencies
    from pypara.price import Price
    from decimal import Decimal
    from datetime import date as Date
    
    price = Price.of(Currencies["USD"], Decimal('-10'), Date(2019, 1, 1))
    result = price.floor_divide(Decimal('3'))
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('-4')


def test_floor_divide_preserves_currency():
    from pypara.currencies import Currencies
    from pypara.price import Price
    from decimal import Decimal
    from datetime import date as Date
    
    price = Price.of(Currencies["EUR"], Decimal('20'), Date(2019, 1, 1))
    result = price.floor_divide(Decimal('4'))
    
    assert result.ccy_or_none().code == "EUR"


# LLM-generated content at query #20
#--------------------------

```python
def test_money_lte():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.money import Money
    
    # Test: defined money lte defined money with same currency and smaller quantity
    usd = Currencies["USD"]
    money1 = Money.of(usd, Decimal('1'), Date(2019, 1, 1))
    money2 = Money.of(usd, Decimal('2'), Date(2019, 1, 1))
    assert money1.lte(money2) is True
    
    # Test: defined money lte defined money with same currency and equal quantity
    money3 = Money.of(usd, Decimal('2'), Date(2019, 1, 1))
    assert money2.lte(money3) is True
    
    # Test: defined money lte defined money with same currency and greater quantity
    money4 = Money.of(usd, Decimal('3'), Date(2019, 1, 1))
    assert money4.lte(money2) is False
    
    # Test: undefined money lte defined money
    undefined_money = Money.na()
    assert undefined_money.lte(money1) is True
    
    # Test: undefined money lte undefined money
    assert undefined_money.lte(Money.na()) is True
    
    # Test: defined money lte undefined money
    assert money1.lte(undefined_money) is False


# LLM-generated content at query #21
#--------------------------

```python
def test_dov_or_none():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    from pypara.money import Money
    
    # Test with defined money - should return the dov
    somemoney = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    result = somemoney.dov_or_none()
    assert result == Date(2019, 1, 1)
    
    # Test with undefined money - should return None
    nonemoney = Money.of(None, None, Date(2019, 1, 1))
    result = nonemoney.dov_or_none()
    assert result is None
    
    # Test with Money.na() - should return None
    na_money = Money.na()
    result = na_money.dov_or_none()
    assert result is None


# LLM-generated content at query #22
#--------------------------

```python
def test_positive():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.price import Price
    
    # Test positive on defined price with positive quantity
    defined_price_positive = Price.of(Currencies["USD"], Decimal('5'), Date(2019, 1, 1))
    result_positive = defined_price_positive.positive()
    assert result_positive.defined
    assert result_positive.qty_or_zero() == Decimal('5')
    assert result_positive.ccy_or_none().code == "USD"
    
    # Test positive on defined price with negative quantity
    defined_price_negative = Price.of(Currencies["USD"], Decimal('-5'), Date(2019, 1, 1))
    result_negative = defined_price_negative.positive()
    assert result_negative.defined
    assert result_negative.qty_or_zero() == Decimal('-5')
    
    # Test positive on defined price with zero quantity
    defined_price_zero = Price.of(Currencies["USD"], Decimal('0'), Date(2019, 1, 1))
    result_zero = defined_price_zero.positive()
    assert result_zero.defined
    assert result_zero.qty_or_zero() == Decimal('0')
    
    # Test positive on undefined price returns itself
    undefined_price = Price.na()
    result_undefined = undefined_price.positive()
    assert result_undefined.undefined
    assert result_undefined is undefined_price


# LLM-generated content at query #23
#--------------------------

```python
def test_truediv_basic_division():
    from decimal import Decimal
    from datetime import date
    
    ccy = Currency(code="USD", decimals=2)
    money = SomeMoney(ccy, Decimal("100.00"), date(2024, 1, 1))
    result = money / 2
    
    assert result.ccy == ccy
    assert result.qty == Decimal("50.00")
    assert result.dov == date(2024, 1, 1)


def test_truediv_with_decimal():
    from decimal import Decimal
    from datetime import date
    
    ccy = Currency(code="EUR", decimals=2)
    money = SomeMoney(ccy, Decimal("75.50"), date(2024, 1, 1))
    result = money / Decimal("2.5")
    
    assert result.qty == Decimal("30.20")


def test_truediv_with_float():
    from decimal import Decimal
    from datetime import date
    
    ccy = Currency(code="GBP", decimals=2)
    money = SomeMoney(ccy, Decimal("100.00"), date(2024, 1, 1))
    result = money / 4.0
    
    assert result.qty == Decimal("25.00")


def test_truediv_by_zero_returns_nomoney():
    from decimal import Decimal
    from datetime import date
    
    ccy = Currency(code="USD", decimals=2)
    money = SomeMoney(ccy, Decimal("100.00"), date(2024, 1, 1))
    result = money / 0
    
    assert result == NoMoney


def test_truediv_quantizes_result():
    from decimal import Decimal
    from datetime import date
    
    ccy = Currency(code="USD", decimals=2)
    money = SomeMoney(ccy, Decimal("100.00"), date(2024, 1, 1))
    result = money / 3
    
    assert result.ccy == ccy
    assert result.qty.as_tuple().exponent == Decimal("0.01").as_tuple().exponent


def test_truediv_preserves_date():
    from decimal import Decimal
    from datetime import date
    
    ccy = Currency(code="JPY", decimals=0)
    test_date = date(2024, 6, 15)
    money = SomeMoney(ccy, Decimal("1000"), test_date)
    result = money / 5
    
    assert result.dov == test_date


def test_truediv_with_integer():
    from decimal import Decimal
    from datetime import date
    
    ccy = Currency(code="USD", decimals=2)
    money = SomeMoney(ccy, Decimal("50.00"), date(2024, 1, 1))
    result = money / 5
    
    assert result.qty == Decimal("10.00")


# LLM-generated content at query #24
#--------------------------

```python
def test_money_int_conversion():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.money import Money
    
    # Test __int__ on defined money with positive quantity
    money_positive = Money.of(Currencies["USD"], Decimal('42.75'), Date(2019, 1, 1))
    result_positive = int(money_positive)
    assert result_positive == 42
    
    # Test __int__ on defined money with negative quantity
    money_negative = Money.of(Currencies["USD"], Decimal('-42.75'), Date(2019, 1, 1))
    result_negative = int(money_negative)
    assert result_negative == -42
    
    # Test __int__ on defined money with zero quantity
    money_zero = Money.of(Currencies["USD"], Decimal('0'), Date(2019, 1, 1))
    result_zero = int(money_zero)
    assert result_zero == 0
    
    # Test __int__ on defined money with integer quantity
    money_integer = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    result_integer = int(money_integer)
    assert result_integer == 100


# LLM-generated content at query #25
#--------------------------

```python
def test_convert_with_valid_rate():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currency, CurrencyType
    from pypara.monetary import SomeMoney
    from pypara.fx import FXRate, FXRateService
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    
    money = SomeMoney(usd, Decimal("100.00"), date(2023, 1, 1))
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if ccy1 == usd and ccy2 == eur:
                return FXRate(usd, eur, Decimal("0.85"), date(2023, 1, 1))
            return None
    
    original_default = FXRateService.default
    FXRateService.default = MockFXRateService()
    
    try:
        result = money.convert(eur)
        assert result.ccy == eur
        assert result.qty == Decimal("85.00")
        assert result.dov == date(2023, 1, 1)
    finally:
        FXRateService.default = original_default


def test_convert_with_custom_asof_date():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currency, CurrencyType
    from pypara.monetary import SomeMoney
    from pypara.fx import FXRate, FXRateService
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    gbp = Currency.of("GBP", "British Pounds", 2, CurrencyType.MONEY)
    
    money = SomeMoney(usd, Decimal("100.00"), date(2023, 1, 1))
    custom_date = date(2023, 6, 15)
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if ccy1 == usd and ccy2 == gbp and asof == custom_date:
                return FXRate(usd, gbp, Decimal("0.80"), custom_date)
            return None
    
    original_default = FXRateService.default
    FXRateService.default = MockFXRateService()
    
    try:
        result = money.convert(gbp, asof=custom_date)
        assert result.ccy == gbp
        assert result.qty == Decimal("80.00")
        assert result.dov == custom_date
    finally:
        FXRateService.default = original_default


def test_convert_no_rate_non_strict():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currency, CurrencyType
    from pypara.monetary import SomeMoney, NoMoney
    from pypara.fx import FXRateService
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    
    money = SomeMoney(usd, Decimal("100.00"), date(2023, 1, 1))
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
    
    original_default = FXRateService.default
    FXRateService.default = MockFXRateService()
    
    try:
        result = money.convert(jpy, strict=False)
        assert result is NoMoney
    finally:
        FXRateService.default = original_default


def test_convert_no_rate_strict_raises_error():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currency, CurrencyType
    from pypara.monetary import SomeMoney
    from pypara.fx import FXRateService
    from pypara.errors import FXRateLookupError
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    cad = Currency.of("CAD", "Canadian Dollars", 2, CurrencyType.MONEY)
    
    money = SomeMoney(usd, Decimal("100.00"), date(2023, 1, 1))
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
    
    original_default = FXRateService.default
    FXRateService.default = MockFXRateService()
    
    try:
        error_raised = False
        try:
            money.convert(cad, strict=True)
        except FXRateLookupError:
            error_raised = True
        assert error_raised
    finally:
        FXRateService.default = original_default


def test_convert_no_fx_service_raises_error():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currency, CurrencyType
    from pypara.monetary import SomeMoney
    from pypara.fx import FXRateService
    from pypara.errors import ProgrammingError
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    aud = Currency.of("AUD", "Australian Dollars", 2, CurrencyType.MONEY)
    
    money = SomeMoney(usd, Decimal("100.00"), date(2023, 1, 1))
    
    original_default = FXRateService.default
    FXRateService.default = None
    
    try:
        error_raised = False
        try:
            money.convert(aud)
        except ProgrammingError:
            error_raised = True
        assert error_raised
    finally:
        FXRateService.default = original_default


def test_convert_uses_money_dov_when_asof_not_provided():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currency, CurrencyType
    from pypara.monetary import SomeMoney
    from pypara.fx import FXRate, FXRateService
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    chf = Currency.of("CHF", "Swiss Francs", 2, CurrencyType.MONEY)
    
    money_dov = date(2023, 3, 15)
    money = SomeMoney(usd, Decimal("100.00"), money_dov)
    
    queried_dates = []
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            queried_dates.append(asof)
            if asof == money_dov:
                return FXRate(usd, chf, Decimal("0.92"), asof)
            return None
    
    original_default = FXRateService.default
    FXRateService.default = MockFXRateService()
    
    try:
        result = money.convert(chf)
        assert len(queried_dates) > 0
        assert queried_dates[0] == money_dov
        assert result.qty == Decimal("92.00")
    finally:
        FXRateService.default = original_default


def test_convert_quantizes_result():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currency, CurrencyType
    from pypara.monetary import SomeMoney
    from pypara.


# LLM-generated content at query #26
#--------------------------

```python
def test_as_float():
    from decimal import Decimal
    from datetime import date as Date
    from pypara.currencies import Currencies
    from pypara.price import Price
    
    # Test as_float with defined price
    defined_price = Price.of(Currencies["USD"], Decimal('123.456'), Date(2019, 1, 1))
    result = defined_price.as_float()
    assert isinstance(result, float)
    assert result == 123.456
    
    # Test as_float with undefined price raises exception
    undefined_price = Price.na()
    try:
        undefined_price.as_float()
        assert False, "Expected MonetaryOperationException"
    except Exception as e:
        assert "MonetaryOperationException" in str(type(e))
    
    # Test as_float with integer quantity
    int_price = Price.of(Currencies["EUR"], Decimal('100'), Date(2020, 6, 15))
    result = int_price.as_float()
    assert result == 100.0
    
    # Test as_float with negative quantity
    neg_price = Price.of(Currencies["GBP"], Decimal('-50.75'), Date(2021, 3, 10))
    result = neg_price.as_float()
    assert result == -50.75
    
    # Test as_float with zero quantity
    zero_price = Price.of(Currencies["JPY"], Decimal('0'), Date(2022, 12, 31))
    result = zero_price.as_float()
    assert result == 0.0


# LLM-generated content at query #27
#--------------------------

```python
def test_ccy_or_none():
    from datetime import date as Date
    from decimal import Decimal
    from pypara.currencies import Currencies
    from pypara.price import Price

    # Test with defined price
    defined_price = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert defined_price.ccy_or_none() is not None
    assert defined_price.ccy_or_none().code == "USD"

    # Test with undefined price (all None)
    undefined_price = Price.of(None, None, None)
    assert undefined_price.ccy_or_none() is None

    # Test with undefined price (ccy is None)
    undefined_price_no_ccy = Price.of(None, Decimal('1'), Date(2019, 1, 1))
    assert undefined_price_no_ccy.ccy_or_none() is None

    # Test with undefined price (qty is None)
    undefined_price_no_qty = Price.of(Currencies["USD"], None, Date(2019, 1, 1))
    assert undefined_price_no_qty.ccy_or_none() is None

    # Test with undefined price (dov is None)
    undefined_price_no_dov = Price.of(Currencies["USD"], Decimal('1'), None)
    assert undefined_price_no_dov.ccy_or_none() is None

    # Test with Price.na()
    na_price = Price.na()
    assert na_price.ccy_or_none() is None


# LLM-generated content at query #28
#--------------------------

```python
def test_money_truediv_defined_money_positive_divisor():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.money import Money
    
    money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = money / Decimal('2')
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('5.00')
    assert result.ccy_or_none().code == 'USD'


def test_money_truediv_defined_money_negative_divisor():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.money import Money
    
    money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = money / Decimal('-2')
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('-5.00')


def test_money_truediv_defined_money_zero_divisor():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.money import Money
    
    money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = money / Decimal('0')
    
    assert result.undefined


def test_money_truediv_undefined_money():
    from decimal import Decimal
    from pypara.money import Money
    
    money = Money.na()
    result = money / Decimal('2')
    
    assert result.undefined


def test_money_truediv_decimal_result():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.money import Money
    
    money = Money.of(Currencies["USD"], Decimal('7'), Date(2019, 1, 1))
    result = money / Decimal('2')
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('3.50')


def test_money_truediv_with_one():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.money import Money
    
    money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = money / Decimal('1')
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('10.00')


# LLM-generated content at query #29
#--------------------------

```python
def test_multiply_defined_money_positive_scalar():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = money.multiply(Decimal('2'))
    
    assert result.qty_or_zero() == Decimal('20.00')
    assert result.ccy_or_none().code == 'USD'
    assert result.dov_or_none() == Date(2019, 1, 1)


def test_multiply_defined_money_negative_scalar():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = money.multiply(Decimal('-3'))
    
    assert result.qty_or_zero() == Decimal('-30.00')
    assert result.defined is True


def test_multiply_defined_money_zero_scalar():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = money.multiply(Decimal('0'))
    
    assert result.qty_or_zero() == Decimal('0.00')


def test_multiply_defined_money_decimal_scalar():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = money.multiply(Decimal('1.5'))
    
    assert result.qty_or_zero() == Decimal('15.00')


def test_multiply_undefined_money_returns_itself():
    from decimal import Decimal
    
    money = Money.na()
    result = money.multiply(Decimal('5'))
    
    assert result.undefined is True
    assert result is money


def test_multiply_defined_money_integer_scalar():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money = Money.of(Currencies["USD"], Decimal('5'), Date(2019, 1, 1))
    result = money.multiply(4)
    
    assert result.qty_or_zero() == Decimal('20.00')


def test_multiply_defined_money_float_scalar():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = money.multiply(2.5)
    
    assert result.qty_or_zero() == Decimal('25.00')


# LLM-generated content at query #30
#--------------------------

```python
def test_price_gt():
    from datetime import date as Date
    from decimal import Decimal
    from pypara.currencies import Currencies
    from pypara.price import Price
    
    # Test: defined price greater than undefined price
    defined_price = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    undefined_price = Price.na()
    assert defined_price.gt(undefined_price) is True
    
    # Test: undefined price not greater than defined price
    assert undefined_price.gt(defined_price) is False
    
    # Test: undefined price not greater than undefined price
    assert undefined_price.gt(undefined_price) is False
    
    # Test: defined price greater than smaller defined price with same currency
    price1 = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('5'), Date(2019, 1, 1))
    assert price1.gt(price2) is True
    
    # Test: defined price not greater than larger defined price with same currency
    price3 = Price.of(Currencies["USD"], Decimal('3'), Date(2019, 1, 1))
    price4 = Price.of(Currencies["USD"], Decimal('8'), Date(2019, 1, 1))
    assert price3.gt(price4) is False
    
    # Test: equal prices - not greater than
    price5 = Price.of(Currencies["USD"], Decimal('5'), Date(2019, 1, 1))
    price6 = Price.of(Currencies["USD"], Decimal('5'), Date(2019, 1, 1))
    assert price5.gt(price6) is False


