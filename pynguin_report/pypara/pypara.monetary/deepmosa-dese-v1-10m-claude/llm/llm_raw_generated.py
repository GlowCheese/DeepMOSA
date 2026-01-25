####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_qty_or_none():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.price import Price

    # Test case 1: defined price returns qty
    someprice = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    result = someprice.qty_or_none()
    assert result == Decimal('1')

    # Test case 2: undefined price returns None
    noneprice = Price.of(None, Decimal('1'), None)
    result = noneprice.qty_or_none()
    assert result is None

    # Test case 3: another defined price with different qty
    price_with_qty_5 = Price.of(Currencies["EUR"], Decimal('5'), Date(2020, 6, 15))
    result = price_with_qty_5.qty_or_none()
    assert result == Decimal('5')

    # Test case 4: na() returns None
    na_price = Price.na()
    result = na_price.qty_or_none()
    assert result is None


# LLM-generated content at query #2
#--------------------------

```python
def test_gte_defined_money_greater_than_or_equal():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    
    money1 = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('50'), Date(2019, 1, 1))
    
    assert money1.gte(money2) is True


def test_gte_defined_money_equal():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    
    money1 = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    
    assert money1.gte(money2) is True


def test_gte_defined_money_less_than():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    
    money1 = Money.of(Currencies["USD"], Decimal('50'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    
    assert money1.gte(money2) is False


def test_gte_undefined_money_with_defined():
    undefined_money = Money.na()
    defined_money = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    
    assert undefined_money.gte(defined_money) is False


def test_gte_undefined_money_with_undefined():
    undefined_money1 = Money.na()
    undefined_money2 = Money.na()
    
    assert undefined_money1.gte(undefined_money2) is True


def test_gte_defined_money_with_undefined():
    defined_money = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    undefined_money = Money.na()
    
    assert defined_money.gte(undefined_money) is True


def test_gte_different_currencies_raises_error():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    
    money_usd = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    money_eur = Money.of(Currencies["EUR"], Decimal('100'), Date(2019, 1, 1))
    
    try:
        money_usd.gte(money_eur)
        assert False, "Expected IncompatibleCurrencyError"
    except Exception as e:
        assert "IncompatibleCurrencyError" in str(type(e))


# LLM-generated content at query #3
#--------------------------

```python
def test_money_pos():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.money import Money
    
    # Test __pos__ on defined money
    money = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    result = +money
    assert result.ccy == money.ccy
    assert result.qty == money.qty
    assert result.dov == money.dov
    
    # Test __pos__ on undefined money
    undefined_money = Money.na()
    result_undefined = +undefined_money
    assert result_undefined.undefined
    
    # Test __pos__ on negative money
    negative_money = Money.of(Currencies["EUR"], Decimal('-50'), Date(2019, 6, 15))
    result_negative = +negative_money
    assert result_negative.qty == Decimal('-50')
    
    # Test __pos__ on zero money
    zero_money = Money.of(Currencies["GBP"], Decimal('0'), Date(2020, 1, 1))
    result_zero = +zero_money
    assert result_zero.qty == Decimal('0')


# LLM-generated content at query #4
#--------------------------

```python
def test_dov_or_none():
    from datetime import date as Date
    from decimal import Decimal
    from pypara.currencies import Currencies
    from pypara.money import Money
    
    # Test with defined money - should return the dov
    somemoney = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    result = somemoney.dov_or_none()
    assert result == Date(2019, 1, 1)
    
    # Test with undefined money (None dov) - should return None
    nonemoney = Money.of(None, None, Date(2019, 1, 1))
    result = nonemoney.dov_or_none()
    assert result is None
    
    # Test with completely undefined money - should return None
    completely_undefined = Money.na()
    result = completely_undefined.dov_or_none()
    assert result is None


# LLM-generated content at query #5
#--------------------------

```python
def test_someprice_ge():
    from datetime import date
    from decimal import Decimal
    
    # Create mock Currency objects
    class MockCurrency:
        def __init__(self, code):
            self.code = code
            self.quantizer = Decimal('0.01')
        
        def __eq__(self, other):
            return isinstance(other, MockCurrency) and self.code == other.code
        
        def __ne__(self, other):
            return not self.__eq__(other)
    
    usd = MockCurrency('USD')
    eur = MockCurrency('EUR')
    
    # Create SomePrice instances
    price1 = SomePrice(usd, Decimal('100.00'), date(2024, 1, 1))
    price2 = SomePrice(usd, Decimal('50.00'), date(2024, 1, 1))
    price3 = SomePrice(usd, Decimal('100.00'), date(2024, 1, 1))
    price_eur = SomePrice(eur, Decimal('100.00'), date(2024, 1, 1))
    
    # Test: price1 >= price2 (100 >= 50) should return True
    assert price1.__ge__(price2) is True
    
    # Test: price2 >= price1 (50 >= 100) should return False
    assert price2.__ge__(price1) is False
    
    # Test: price1 >= price3 (100 >= 100) should return True
    assert price1.__ge__(price3) is True
    
    # Test: price1 >= non-SomePrice should return True
    assert price1.__ge__("not a price") is True
    
    # Test: incompatible currencies should raise IncompatibleCurrencyError
    try:
        price1.__ge__(price_eur)
        assert False, "Should have raised IncompatibleCurrencyError"
    except Exception as e:
        assert "IncompatibleCurrencyError" in str(type(e).__name__)


# LLM-generated content at query #6
#--------------------------

```python
def test_money_int_conversion():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    # Test with defined money - should return integer conversion of quantity
    defined_money = Money.of(Currencies["USD"], Decimal('42.75'), Date(2019, 1, 1))
    result = int(defined_money)
    assert result == 42
    
    # Test with defined money - negative quantity
    negative_money = Money.of(Currencies["EUR"], Decimal('-15.99'), Date(2019, 1, 1))
    result = int(negative_money)
    assert result == -15
    
    # Test with defined money - zero quantity
    zero_money = Money.of(Currencies["GBP"], Decimal('0.50'), Date(2019, 1, 1))
    result = int(zero_money)
    assert result == 0
    
    # Test with undefined money - should raise MonetaryOperationException
    undefined_money = Money.na()
    try:
        int(undefined_money)
        assert False, "Should have raised MonetaryOperationException"
    except Exception as e:
        assert "MonetaryOperationException" in str(type(e))


# LLM-generated content at query #7
#--------------------------

```python
def test_dov_or_none():
    from datetime import date as Date
    from decimal import Decimal
    from pypara.currencies import Currencies
    from pypara.price import Price
    
    # Test with defined price - should return the dov
    someprice = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    result = someprice.dov_or_none()
    assert result == Date(2019, 1, 1)
    
    # Test with undefined price (None currency and quantity) - should return None
    noneprice = Price.of(None, None, Date(2019, 1, 1))
    result = noneprice.dov_or_none()
    assert result is None
    
    # Test with undefined price created via na() - should return None
    na_price = Price.na()
    result = na_price.dov_or_none()
    assert result is None


# LLM-generated content at query #8
#--------------------------

```python
def test_money_bool():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    # Test that defined money with non-zero quantity returns True
    defined_nonzero = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert bool(defined_nonzero) is True
    
    # Test that defined money with zero quantity returns False
    defined_zero = Money.of(Currencies["USD"], Decimal('0'), Date(2019, 1, 1))
    assert bool(defined_zero) is False
    
    # Test that undefined money returns False
    undefined_money = Money.na()
    assert bool(undefined_money) is False
    
    # Test that defined money with negative quantity returns True
    defined_negative = Money.of(Currencies["USD"], Decimal('-5'), Date(2019, 1, 1))
    assert bool(defined_negative) is True
    
    # Test that defined money with positive decimal quantity returns True
    defined_decimal = Money.of(Currencies["USD"], Decimal('0.01'), Date(2019, 1, 1))
    assert bool(defined_decimal) is True


# LLM-generated content at query #9
#--------------------------

```python
def test_money_floordiv():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.money import Money
    
    # Test floor division with defined money
    money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = money.floor_divide(Decimal('3'))
    assert result.defined
    assert result.qty == Decimal('3')
    
    # Test floor division with decimal result
    money = Money.of(Currencies["USD"], Decimal('7'), Date(2019, 1, 1))
    result = money.floor_divide(Decimal('2'))
    assert result.defined
    assert result.qty == Decimal('3')
    
    # Test floor division with negative divisor
    money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = money.floor_divide(Decimal('-3'))
    assert result.defined
    assert result.qty == Decimal('-4')
    
    # Test floor division by zero returns undefined money
    money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = money.floor_divide(Decimal('0'))
    assert result.undefined
    
    # Test floor division on undefined money returns itself
    money = Money.na()
    result = money.floor_divide(Decimal('3'))
    assert result.undefined
    assert result is money
    
    # Test floor division with integer
    money = Money.of(Currencies["USD"], Decimal('15'), Date(2019, 1, 1))
    result = money.floor_divide(4)
    assert result.defined
    assert result.qty == Decimal('3')


# LLM-generated content at query #10
#--------------------------

```python
def test_price_round():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    # Test rounding defined price with default ndigits=0
    price_defined = Price.of(Currencies["USD"], Decimal('1.567'), Date(2019, 1, 1))
    rounded_price = price_defined.round()
    assert rounded_price.qty == Decimal('2')
    
    # Test rounding defined price with specific ndigits
    price_defined_2 = Price.of(Currencies["USD"], Decimal('1.567'), Date(2019, 1, 1))
    rounded_price_2 = price_defined_2.round(ndigits=2)
    assert rounded_price_2.qty == Decimal('1.57')
    
    # Test rounding defined price with ndigits=1
    price_defined_3 = Price.of(Currencies["USD"], Decimal('1.567'), Date(2019, 1, 1))
    rounded_price_3 = price_defined_3.round(ndigits=1)
    assert rounded_price_3.qty == Decimal('1.6')
    
    # Test rounding undefined price returns itself
    price_undefined = Price.na()
    rounded_undefined = price_undefined.round()
    assert rounded_undefined.undefined
    
    # Test rounding undefined price with ndigits parameter
    price_undefined_2 = Price.na()
    rounded_undefined_2 = price_undefined_2.round(ndigits=2)
    assert rounded_undefined_2.undefined
    
    # Test rounding with negative quantity
    price_negative = Price.of(Currencies["USD"], Decimal('-1.567'), Date(2019, 1, 1))
    rounded_negative = price_negative.round(ndigits=2)
    assert rounded_negative.qty == Decimal('-1.57')
    
    # Test rounding preserves currency
    price_eur = Price.of(Currencies["EUR"], Decimal('2.345'), Date(2019, 1, 1))
    rounded_eur = price_eur.round(ndigits=1)
    assert rounded_eur.ccy.code == "EUR"
    
    # Test rounding preserves date
    test_date = Date(2019, 6, 15)
    price_with_date = Price.of(Currencies["USD"], Decimal('3.456'), test_date)
    rounded_with_date = price_with_date.round(ndigits=1)
    assert rounded_with_date.dov == test_date


# LLM-generated content at query #11
#--------------------------

```python
def test_gte_predicate_not_isinstance_someprice():
    from decimal import Decimal
    from datetime import date
    
    # Create a SomePrice instance
    ccy = type('Currency', (), {'quantizer': Decimal('0.01')})()
    price = SomePrice(ccy, Decimal('100'), date(2023, 1, 1))
    
    # Test with an object that is not a SomePrice instance
    not_a_price = "not a price"
    result = price.gte(not_a_price)
    
    assert result is True


# LLM-generated content at query #12
#--------------------------

```python
def test_money_eq():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    
    # Test equality of two defined money objects with same values
    money1 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert money1 == money2
    
    # Test inequality of two defined money objects with different quantities
    money3 = Money.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    assert not (money1 == money3)
    
    # Test inequality of two defined money objects with different currencies
    money4 = Money.of(Currencies["EUR"], Decimal('1'), Date(2019, 1, 1))
    assert not (money1 == money4)
    
    # Test inequality of two defined money objects with different dates
    money5 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 2))
    assert not (money1 == money5)
    
    # Test equality of two undefined money objects
    money6 = Money.na()
    money7 = Money.na()
    assert money6 == money7
    
    # Test inequality of defined and undefined money objects
    assert not (money1 == money6)
    
    # Test inequality with non-money object
    assert not (money1 == "not a money")
    assert not (money1 == 1)
    assert not (money1 == None)
    
    # Test equality with itself
    assert money1 == money1


# LLM-generated content at query #13
#--------------------------

```python
def test_qty_or_else():
    from decimal import Decimal
    from datetime import date
    
    # Create a mock Currency object
    class MockCurrency:
        def __init__(self, code):
            self.code = code
    
    ccy = MockCurrency("USD")
    qty = Decimal("100.50")
    dov = date(2023, 1, 15)
    
    some_price = SomePrice(ccy, qty, dov)
    
    # Test that qty_or_else returns the qty when called
    result = some_price.qty_or_else(lambda: Decimal("999"))
    assert result == Decimal("100.50")
    
    # Test with different callable
    result2 = some_price.qty_or_else(lambda: "default_value")
    assert result2 == Decimal("100.50")
    
    # Test with zero quantity
    some_price_zero = SomePrice(ccy, Decimal("0"), dov)
    result3 = some_price_zero.qty_or_else(lambda: Decimal("50"))
    assert result3 == Decimal("0")
    
    # Test with negative quantity
    some_price_neg = SomePrice(ccy, Decimal("-25.75"), dov)
    result4 = some_price_neg.qty_or_else(lambda: Decimal("100"))
    assert result4 == Decimal("-25.75")


# LLM-generated content at query #14
#--------------------------

```python
def test_somemoney_constructor():
    from decimal import Decimal
    from datetime import date
    
    # Assuming Currency is a class that can be instantiated
    # and has necessary attributes like decimals and quantizer
    usd = Currency(code="USD", decimals=2, quantizer=Decimal("0.01"))
    eur = Currency(code="EUR", decimals=2, quantizer=Decimal("0.01"))
    
    qty = Decimal("100.50")
    dov = date(2024, 1, 15)
    
    # Test basic constructor
    money1 = SomeMoney(usd, qty, dov)
    assert money1.ccy == usd
    assert money1.qty == qty
    assert money1.dov == dov
    
    # Test with different currency
    money2 = SomeMoney(eur, Decimal("50.25"), date(2024, 2, 20))
    assert money2.ccy == eur
    assert money2.qty == Decimal("50.25")
    assert money2.dov == date(2024, 2, 20)
    
    # Test with zero quantity
    money3 = SomeMoney(usd, Decimal("0"), dov)
    assert money3.qty == Decimal("0")
    
    # Test with negative quantity
    money4 = SomeMoney(usd, Decimal("-25.75"), dov)
    assert money4.qty == Decimal("-25.75")
    
    # Test that tuple unpacking works
    c, q, d = money1
    assert c == usd
    assert q == qty
    assert d == dov
    
    # Test indexing
    assert money1[0] == usd
    assert money1[1] == qty
    assert money1[2] == dov


# LLM-generated content at query #15
#--------------------------

```python
def test_someprice_convert_with_valid_rate():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currency, CurrencyType
    from pypara.monetary import SomePrice
    from pypara.fx import FXRate, FXRateService
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    
    price = SomePrice(usd, Decimal("100.00"), date(2023, 1, 1))
    
    fx_rate = FXRate(usd, eur, date(2023, 1, 1), Decimal("0.92"))
    
    class MockFXRateService:
        def query(self, ccy1, ccy2, asof, strict):
            return fx_rate
    
    original_default = FXRateService.default
    FXRateService.default = MockFXRateService()
    
    result = price.convert(eur)
    
    FXRateService.default = original_default
    
    assert result.ccy == eur
    assert result.qty == Decimal("92.00")
    assert result.dov == date(2023, 1, 1)


def test_someprice_convert_with_custom_asof_date():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currency, CurrencyType
    from pypara.monetary import SomePrice
    from pypara.fx import FXRate, FXRateService
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    gbp = Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY)
    
    price = SomePrice(usd, Decimal("100.00"), date(2023, 1, 1))
    
    fx_rate = FXRate(usd, gbp, date(2023, 6, 15), Decimal("0.80"))
    
    class MockFXRateService:
        def query(self, ccy1, ccy2, asof, strict):
            return fx_rate
    
    original_default = FXRateService.default
    FXRateService.default = MockFXRateService()
    
    result = price.convert(gbp, asof=date(2023, 6, 15))
    
    FXRateService.default = original_default
    
    assert result.ccy == gbp
    assert result.qty == Decimal("80.00")
    assert result.dov == date(2023, 6, 15)


def test_someprice_convert_no_rate_non_strict():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currency, CurrencyType
    from pypara.monetary import SomePrice, NoPrice
    from pypara.fx import FXRateService
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    
    price = SomePrice(usd, Decimal("100.00"), date(2023, 1, 1))
    
    class MockFXRateService:
        def query(self, ccy1, ccy2, asof, strict):
            return None
    
    original_default = FXRateService.default
    FXRateService.default = MockFXRateService()
    
    result = price.convert(jpy, strict=False)
    
    FXRateService.default = original_default
    
    assert result == NoPrice


def test_someprice_convert_uses_price_dov_when_asof_not_provided():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currency, CurrencyType
    from pypara.monetary import SomePrice
    from pypara.fx import FXRate, FXRateService
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    chf = Currency.of("CHF", "Swiss Franc", 2, CurrencyType.MONEY)
    
    price_date = date(2023, 3, 15)
    price = SomePrice(usd, Decimal("100.00"), price_date)
    
    fx_rate = FXRate(usd, chf, price_date, Decimal("0.88"))
    
    queried_dates = []
    
    class MockFXRateService:
        def query(self, ccy1, ccy2, asof, strict):
            queried_dates.append(asof)
            return fx_rate
    
    original_default = FXRateService.default
    FXRateService.default = MockFXRateService()
    
    result = price.convert(chf)
    
    FXRateService.default = original_default
    
    assert queried_dates[0] == price_date
    assert result.dov == price_date


# LLM-generated content at query #16
#--------------------------

```python
def test_convert_with_valid_fx_rate():
    from datetime import date
    from decimal import Decimal
    from pypara.currencies import Currency, CurrencyType
    from pypara.monetary import SomePrice
    from pypara.fx import FXRate, FXRateService
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    
    price = SomePrice(usd, Decimal("100"), date(2023, 1, 1))
    
    fx_rate = FXRate(usd, eur, Decimal("0.85"), date(2023, 1, 1))
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return fx_rate
    
    original_service = FXRateService.default
    FXRateService.default = MockFXRateService()
    
    try:
        result = price.convert(eur)
        assert result.ccy == eur
        assert result.qty == Decimal("85")
        assert result.dov == date(2023, 1, 1)
    finally:
        FXRateService.default = original_service


def test_convert_with_custom_asof_date():
    from datetime import date
    from decimal import Decimal
    from pypara.currencies import Currency, CurrencyType
    from pypara.monetary import SomePrice
    from pypara.fx import FXRate, FXRateService
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    
    price = SomePrice(usd, Decimal("100"), date(2023, 1, 1))
    custom_date = date(2023, 6, 15)
    
    fx_rate = FXRate(usd, eur, Decimal("0.92"), custom_date)
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return fx_rate if asof == custom_date else None
    
    original_service = FXRateService.default
    FXRateService.default = MockFXRateService()
    
    try:
        result = price.convert(eur, asof=custom_date)
        assert result.ccy == eur
        assert result.qty == Decimal("92")
        assert result.dov == custom_date
    finally:
        FXRateService.default = original_service


def test_convert_without_fx_rate_non_strict():
    from datetime import date
    from decimal import Decimal
    from pypara.currencies import Currency, CurrencyType
    from pypara.monetary import SomePrice, NoPrice
    from pypara.fx import FXRateService
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    
    price = SomePrice(usd, Decimal("100"), date(2023, 1, 1))
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
    
    original_service = FXRateService.default
    FXRateService.default = MockFXRateService()
    
    try:
        result = price.convert(eur, strict=False)
        assert result is NoPrice
    finally:
        FXRateService.default = original_service


def test_convert_without_fx_rate_strict():
    from datetime import date
    from decimal import Decimal
    from pypara.currencies import Currency, CurrencyType
    from pypara.monetary import SomePrice
    from pypara.fx import FXRateService, FXRateLookupError
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    
    price = SomePrice(usd, Decimal("100"), date(2023, 1, 1))
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
    
    original_service = FXRateService.default
    FXRateService.default = MockFXRateService()
    
    try:
        error_raised = False
        try:
            price.convert(eur, strict=True)
        except FXRateLookupError:
            error_raised = True
        assert error_raised
    finally:
        FXRateService.default = original_service


def test_convert_no_default_fx_service():
    from datetime import date
    from decimal import Decimal
    from pypara.currencies import Currency, CurrencyType
    from pypara.monetary import SomePrice
    from pypara.fx import FXRateService
    from pypara.errors import ProgrammingError
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    
    price = SomePrice(usd, Decimal("100"), date(2023, 1, 1))
    
    original_service = FXRateService.default
    FXRateService.default = None
    
    try:
        error_raised = False
        try:
            price.convert(eur)
        except ProgrammingError:
            error_raised = True
        assert error_raised
    finally:
        FXRateService.default = original_service


# LLM-generated content at query #17
#--------------------------

```python
def test_dov_or_none():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    from pypara.money import Money
    
    # Test with defined money - should return the date of value
    somemoney = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    result = somemoney.dov_or_none()
    assert result == Date(2019, 1, 1)
    
    # Test with undefined money (None dov) - should return None
    nonemoney = Money.of(None, None, Date(2019, 1, 1))
    result = nonemoney.dov_or_none()
    assert result is None
    
    # Test with undefined money (Money.na()) - should return None
    undefined_money = Money.na()
    result = undefined_money.dov_or_none()
    assert result is None


# LLM-generated content at query #18
#--------------------------

```python
def test_someprice_constructor():
    from decimal import Decimal
    from datetime import date
    
    ccy = Currency(code="USD", quantizer=Decimal("0.01"))
    qty = Decimal("100.50")
    dov = date(2024, 1, 15)
    
    price = SomePrice(ccy, qty, dov)
    
    assert price.ccy == ccy
    assert price.qty == qty
    assert price.dov == dov
    assert price[0] == ccy
    assert price[1] == qty
    assert price[2] == dov


def test_someprice_constructor_with_different_values():
    from decimal import Decimal
    from datetime import date
    
    ccy = Currency(code="EUR", quantizer=Decimal("0.01"))
    qty = Decimal("250.75")
    dov = date(2023, 6, 30)
    
    price = SomePrice(ccy, qty, dov)
    
    assert price.ccy == ccy
    assert price.qty == qty
    assert price.dov == dov


def test_someprice_constructor_creates_tuple():
    from decimal import Decimal
    from datetime import date
    
    ccy = Currency(code="GBP", quantizer=Decimal("0.01"))
    qty = Decimal("50.00")
    dov = date(2024, 12, 25)
    
    price = SomePrice(ccy, qty, dov)
    
    assert isinstance(price, tuple)
    assert len(price) == 3
    assert tuple(price) == (ccy, qty, dov)


# LLM-generated content at query #19
#--------------------------

```python
def test_convert_predicate_line_12_evaluates_to_false():
    from datetime import date
    from decimal import Decimal
    from pypara.currencies import Currency, CurrencyType
    from pypara.monetary import SomePrice
    from unittest.mock import Mock, patch
    
    # Create test currencies
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    
    # Create a SomePrice instance
    price = SomePrice(usd, Decimal("100.00"), date(2023, 1, 1))
    
    # Create a mock FXRateService with a non-None default
    mock_service = Mock()
    mock_rate = Mock()
    mock_rate.value = Decimal("0.85")
    mock_service.query.return_value = mock_rate
    
    # Patch FXRateService.default to be the mock service (not None)
    with patch('pypara.monetary.FXRateService') as mock_fx_service:
        mock_fx_service.default = mock_service
        
        # Call convert - this should execute without raising ProgrammingError
        # because FXRateService.default is not None (line 12 predicate is False)
        result = price.convert(eur, date(2023, 1, 1), strict=False)
        
        # Verify the result is a SomePrice with the converted currency
        assert result.ccy == eur
        assert result.qty == Decimal("85.00")
        assert result.dov == date(2023, 1, 1)


# LLM-generated content at query #20
#--------------------------

```python
def test_gte():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    
    # Test: defined money >= defined money with same currency and greater quantity
    money1 = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('5'), Date(2019, 1, 1))
    assert money1.gte(money2) is True
    
    # Test: defined money >= defined money with same currency and equal quantity
    money3 = Money.of(Currencies["USD"], Decimal('5'), Date(2019, 1, 1))
    money4 = Money.of(Currencies["USD"], Decimal('5'), Date(2019, 1, 1))
    assert money3.gte(money4) is True
    
    # Test: defined money >= defined money with same currency and lesser quantity
    money5 = Money.of(Currencies["USD"], Decimal('3'), Date(2019, 1, 1))
    money6 = Money.of(Currencies["USD"], Decimal('5'), Date(2019, 1, 1))
    assert money5.gte(money6) is False
    
    # Test: defined money >= undefined money
    money7 = Money.of(Currencies["USD"], Decimal('5'), Date(2019, 1, 1))
    money8 = Money.na()
    assert money7.gte(money8) is True
    
    # Test: undefined money >= undefined money
    money9 = Money.na()
    money10 = Money.na()
    assert money9.gte(money10) is True
    
    # Test: undefined money >= defined money
    money11 = Money.na()
    money12 = Money.of(Currencies["USD"], Decimal('5'), Date(2019, 1, 1))
    assert money11.gte(money12) is False


# LLM-generated content at query #21
#--------------------------

```python
def test_as_float_defined_price():
    from pypara.currencies import Currencies
    from datetime import date
    from decimal import Decimal
    
    price = Price.of(Currencies["USD"], Decimal('123.45'), date(2019, 1, 1))
    result = price.as_float()
    assert result == 123.45
    assert isinstance(result, float)


def test_as_float_undefined_price():
    from decimal import Decimal
    
    price = Price.na()
    try:
        price.as_float()
        assert False, "Expected MonetaryOperationException"
    except Exception as e:
        assert "MonetaryOperationException" in str(type(e).__name__)


def test_as_float_with_zero():
    from pypara.currencies import Currencies
    from datetime import date
    from decimal import Decimal
    
    price = Price.of(Currencies["USD"], Decimal('0'), date(2019, 1, 1))
    result = price.as_float()
    assert result == 0.0


def test_as_float_with_negative():
    from pypara.currencies import Currencies
    from datetime import date
    from decimal import Decimal
    
    price = Price.of(Currencies["USD"], Decimal('-50.75'), date(2019, 1, 1))
    result = price.as_float()
    assert result == -50.75


def test_as_float_with_large_number():
    from pypara.currencies import Currencies
    from datetime import date
    from decimal import Decimal
    
    price = Price.of(Currencies["USD"], Decimal('999999.999999'), date(2019, 1, 1))
    result = price.as_float()
    assert abs(result - 999999.999999) < 0.000001


# LLM-generated content at query #22
#--------------------------

```python
def test_multiply_defined_money_positive_scalar():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from decimal import Decimal
    from datetime import date as Date
    
    money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = money.multiply(Decimal('2'))
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('20.00')
    assert result.ccy_or_none().code == "USD"


def test_multiply_defined_money_negative_scalar():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from decimal import Decimal
    from datetime import date as Date
    
    money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = money.multiply(Decimal('-2'))
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('-20.00')


def test_multiply_defined_money_zero_scalar():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from decimal import Decimal
    from datetime import date as Date
    
    money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = money.multiply(Decimal('0'))
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('0.00')


def test_multiply_defined_money_decimal_scalar():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from decimal import Decimal
    from datetime import date as Date
    
    money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = money.multiply(Decimal('0.5'))
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('5.00')


def test_multiply_undefined_money():
    from pypara.money import Money
    
    money = Money.na()
    result = money.multiply(Decimal('2'))
    
    assert result.undefined
    assert result is money


def test_multiply_defined_money_integer_scalar():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from decimal import Decimal
    from datetime import date as Date
    
    money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = money.multiply(5)
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('50.00')


def test_multiply_preserves_currency():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from decimal import Decimal
    from datetime import date as Date
    
    money = Money.of(Currencies["EUR"], Decimal('100'), Date(2020, 6, 15))
    result = money.multiply(Decimal('3'))
    
    assert result.ccy_or_none().code == "EUR"


def test_multiply_preserves_date():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from decimal import Decimal
    from datetime import date as Date
    
    original_date = Date(2020, 6, 15)
    money = Money.of(Currencies["GBP"], Decimal('50'), original_date)
    result = money.multiply(Decimal('2'))
    
    assert result.dov_or_none() == original_date


# LLM-generated content at query #23
#--------------------------

```python
def test_money_add_defined_same_currency():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.money import Money
    
    money1 = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('20'), Date(2019, 1, 2))
    result = money1.add(money2)
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('30.00')
    assert result.ccy_or_none().code == 'USD'


def test_money_add_defined_different_currency_raises():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.money import Money
    
    money1 = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["EUR"], Decimal('20'), Date(2019, 1, 2))
    
    try:
        money1.add(money2)
        assert False, "Should raise IncompatibleCurrencyError"
    except Exception as e:
        assert "IncompatibleCurrencyError" in str(type(e))


def test_money_add_undefined_with_defined():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.money import Money
    
    money1 = Money.na()
    money2 = Money.of(Currencies["USD"], Decimal('20'), Date(2019, 1, 2))
    result = money1.add(money2)
    
    assert result is money2


def test_money_add_defined_with_undefined():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.money import Money
    
    money1 = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    money2 = Money.na()
    result = money1.add(money2)
    
    assert result is money1


def test_money_add_both_undefined():
    from pypara.money import Money
    
    money1 = Money.na()
    money2 = Money.na()
    result = money1.add(money2)
    
    assert result.undefined


def test_money_add_carries_forward_date():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.money import Money
    
    money1 = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('20'), Date(2019, 1, 15))
    result = money1.add(money2)
    
    assert result.dov_or_none() == Date(2019, 1, 15)


# LLM-generated content at query #24
#--------------------------

```python
def test_times_with_defined_price():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    price = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = price.times(Decimal('2'))
    
    assert result.defined
    assert result.qty == Decimal('20')
    assert result.ccy.code == 'USD'


def test_times_with_undefined_price():
    price = Price.na()
    result = price.times(Decimal('5'))
    
    assert result.undefined


def test_times_with_zero_multiplier():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    price = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = price.times(Decimal('0'))
    
    assert result.defined
    assert result.qty == Decimal('0')


def test_times_with_negative_multiplier():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    price = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = price.times(Decimal('-3'))
    
    assert result.defined
    assert result.qty == Decimal('-30')


def test_times_with_decimal_multiplier():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    price = Price.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    result = price.times(Decimal('0.5'))
    
    assert result.defined
    assert result.qty == Decimal('50')


def test_times_preserves_currency():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    price = Price.of(Currencies["EUR"], Decimal('50'), Date(2019, 6, 15))
    result = price.times(Decimal('4'))
    
    assert result.ccy.code == 'EUR'


def test_times_preserves_date():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    original_date = Date(2019, 3, 20)
    price = Price.of(Currencies["GBP"], Decimal('25'), original_date)
    result = price.times(Decimal('2'))
    
    assert result.dov == original_date


# LLM-generated content at query #25
#--------------------------

```python
def test_money_lte():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    # Test 1: Defined money less than other defined money with same currency
    money1 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    assert money1.lte(money2) is True
    
    # Test 2: Defined money equal to other defined money with same currency
    money3 = Money.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    money4 = Money.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    assert money3.lte(money4) is True
    
    # Test 3: Defined money greater than other defined money with same currency
    money5 = Money.of(Currencies["USD"], Decimal('3'), Date(2019, 1, 1))
    money6 = Money.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    assert money5.lte(money6) is False
    
    # Test 4: Undefined money is always less than or equal to defined money
    undefined_money = Money.na()
    defined_money = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert undefined_money.lte(defined_money) is True
    
    # Test 5: Undefined money is less than or equal to undefined money
    undefined_money1 = Money.na()
    undefined_money2 = Money.na()
    assert undefined_money1.lte(undefined_money2) is True
    
    # Test 6: Defined money is not less than or equal to undefined money
    defined_money2 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    undefined_money3 = Money.na()
    assert defined_money2.lte(undefined_money3) is False


# LLM-generated content at query #26
#--------------------------

```python
def test_fmap_with_defined_price():
    from pypara.currencies import Currencies
    from pypara.price import Price
    from decimal import Decimal
    from datetime import date as Date
    import datetime
    
    someprice = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    new = someprice.fmap(lambda x: Price.of(x.ccy, x.qty + Decimal('1'), x.dov + datetime.timedelta(days=10)))
    
    assert new.ccy.code == 'USD'
    assert new.qty == Decimal('2')
    assert new.dov == Date(2019, 1, 11)


def test_fmap_with_undefined_price():
    from pypara.price import Price
    from decimal import Decimal
    
    noneprice = Price.of(None, Decimal('1'), None)
    result = noneprice.fmap(lambda sp: Price.of(sp.ccy, sp.qty + Decimal('1'), sp.dov))
    
    assert result.undefined is True


def test_fmap_preserves_defined_price():
    from pypara.currencies import Currencies
    from pypara.price import Price
    from decimal import Decimal
    from datetime import date as Date
    
    someprice = Price.of(Currencies["USD"], Decimal('5'), Date(2020, 6, 15))
    result = someprice.fmap(lambda x: x)
    
    assert result.defined is True
    assert result.ccy.code == 'USD'
    assert result.qty == Decimal('5')


def test_fmap_with_transformation_function():
    from pypara.currencies import Currencies
    from pypara.price import Price
    from decimal import Decimal
    from datetime import date as Date
    import datetime
    
    someprice = Price.of(Currencies["EUR"], Decimal('10'), Date(2021, 3, 20))
    new = someprice.fmap(lambda x: Price.of(x.ccy, x.qty * Decimal('2'), x.dov + datetime.timedelta(days=5)))
    
    assert new.ccy.code == 'EUR'
    assert new.qty == Decimal('20')
    assert new.dov == Date(2021, 3, 25)


# LLM-generated content at query #27
#--------------------------

```python
def test_money_abs():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    # Test abs on defined money with positive quantity
    positive_money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result_positive = positive_money.abs()
    assert result_positive.qty == Decimal('10.00')
    assert result_positive.ccy.code == 'USD'
    
    # Test abs on defined money with negative quantity
    negative_money = Money.of(Currencies["USD"], Decimal('-10'), Date(2019, 1, 1))
    result_negative = negative_money.abs()
    assert result_negative.qty == Decimal('10.00')
    assert result_negative.ccy.code == 'USD'
    
    # Test abs on defined money with zero quantity
    zero_money = Money.of(Currencies["USD"], Decimal('0'), Date(2019, 1, 1))
    result_zero = zero_money.abs()
    assert result_zero.qty == Decimal('0.00')
    
    # Test abs on undefined money
    undefined_money = Money.na()
    result_undefined = undefined_money.abs()
    assert result_undefined.undefined
    assert result_undefined is undefined_money


# LLM-generated content at query #28
#--------------------------

```python
def test_abs():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.money import Money
    
    # Test abs on defined money with positive quantity
    positive_money = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    result_positive = positive_money.abs()
    assert result_positive.qty_or_zero() == Decimal('100.00')
    assert result_positive.ccy_or_none().code == "USD"
    
    # Test abs on defined money with negative quantity
    negative_money = Money.of(Currencies["USD"], Decimal('-100'), Date(2019, 1, 1))
    result_negative = negative_money.abs()
    assert result_negative.qty_or_zero() == Decimal('100.00')
    assert result_negative.ccy_or_none().code == "USD"
    
    # Test abs on defined money with zero quantity
    zero_money = Money.of(Currencies["USD"], Decimal('0'), Date(2019, 1, 1))
    result_zero = zero_money.abs()
    assert result_zero.qty_or_zero() == Decimal('0.00')
    
    # Test abs on undefined money
    undefined_money = Money.na()
    result_undefined = undefined_money.abs()
    assert result_undefined.undefined
    assert result_undefined is undefined_money


# LLM-generated content at query #29
#--------------------------

```python
def test_money_sub():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    
    money1 = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('3'), Date(2019, 1, 1))
    result = money1 - money2
    
    assert result.qty_or_zero() == Decimal('7.00')
    assert result.ccy_or_none().code == 'USD'


def test_money_sub_with_undefined():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    
    money1 = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    undefined_money = Money.na()
    result = money1 - undefined_money
    
    assert result.qty_or_zero() == Decimal('10.00')
    assert result.ccy_or_none().code == 'USD'


def test_money_sub_both_undefined():
    undefined_money1 = Money.na()
    undefined_money2 = Money.na()
    result = undefined_money1 - undefined_money2
    
    assert result.undefined


def test_money_sub_different_currencies():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    
    money1 = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["EUR"], Decimal('3'), Date(2019, 1, 1))
    
    try:
        result = money1 - money2
        assert False, "Should raise IncompatibleCurrencyError"
    except Exception:
        pass


def test_money_sub_negative_result():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    
    money1 = Money.of(Currencies["USD"], Decimal('3'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = money1 - money2
    
    assert result.qty_or_zero() == Decimal('-7.00')


# LLM-generated content at query #30
#--------------------------

```python
def test_qty_map_defined_money():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.money import Money
    
    somemoney = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    result = somemoney.qty_map(lambda x: x + Decimal('1'), lambda: Decimal('42'))
    assert result == Decimal('2.00')


def test_qty_map_undefined_money():
    from decimal import Decimal
    from pypara.money import Money
    
    nonemoney = Money.of(None, Decimal('1'), None)
    result = nonemoney.qty_map(lambda x: x + Decimal('1'), lambda: Decimal('42'))
    assert result == Decimal('42')


def test_qty_map_with_different_function():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.money import Money
    
    somemoney = Money.of(Currencies["USD"], Decimal('5'), Date(2019, 1, 1))
    result = somemoney.qty_map(lambda x: x * Decimal('2'), lambda: Decimal('0'))
    assert result == Decimal('10.00')


def test_qty_map_undefined_with_different_combinator():
    from decimal import Decimal
    from pypara.money import Money
    
    nonemoney = Money.of(None, Decimal('5'), None)
    result = nonemoney.qty_map(lambda x: x * Decimal('2'), lambda: Decimal('999'))
    assert result == Decimal('999')


def test_qty_map_defined_money_with_string_function():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.money import Money
    
    somemoney = Money.of(Currencies["EUR"], Decimal('10'), Date(2020, 6, 15))
    result = somemoney.qty_map(lambda x: str(x), lambda: "fallback")
    assert result == "10.00"


def test_qty_map_undefined_money_with_string_combinator():
    from decimal import Decimal
    from pypara.money import Money
    
    nonemoney = Money.of(None, Decimal('10'), None)
    result = nonemoney.qty_map(lambda x: str(x), lambda: "fallback")
    assert result == "fallback"


# LLM-generated content at query #31
#--------------------------

```python
def test_price_abs():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.price import Price
    
    # Test abs on defined price with positive quantity
    positive_price = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    abs_positive = positive_price.abs()
    assert abs_positive.qty_or_zero() == Decimal('10')
    assert abs_positive.ccy_or_none().code == 'USD'
    
    # Test abs on defined price with negative quantity
    negative_price = Price.of(Currencies["USD"], Decimal('-10'), Date(2019, 1, 1))
    abs_negative = negative_price.abs()
    assert abs_negative.qty_or_zero() == Decimal('10')
    assert abs_negative.ccy_or_none().code == 'USD'
    
    # Test abs on undefined price returns itself
    undefined_price = Price.na()
    abs_undefined = undefined_price.abs()
    assert abs_undefined.undefined
    assert abs_undefined is undefined_price
    
    # Test abs on zero quantity
    zero_price = Price.of(Currencies["EUR"], Decimal('0'), Date(2020, 6, 15))
    abs_zero = zero_price.abs()
    assert abs_zero.qty_or_zero() == Decimal('0')
    assert abs_zero.ccy_or_none().code == 'EUR'


# LLM-generated content at query #32
#--------------------------

```python
def test_money_lt():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    # Test: defined money less than defined money with same currency
    money1 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    assert money1.lt(money2) is True
    
    # Test: defined money not less than defined money with same currency (equal)
    money3 = Money.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    money4 = Money.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    assert money3.lt(money4) is False
    
    # Test: defined money not less than smaller defined money with same currency
    money5 = Money.of(Currencies["USD"], Decimal('3'), Date(2019, 1, 1))
    money6 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert money5.lt(money6) is False
    
    # Test: undefined money less than defined money
    undefined_money = Money.na()
    defined_money = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert undefined_money.lt(defined_money) is True
    
    # Test: undefined money not less than undefined money
    undefined_money2 = Money.na()
    assert undefined_money.lt(undefined_money2) is False
    
    # Test: defined money not less than undefined money
    assert defined_money.lt(undefined_money) is False


# LLM-generated content at query #33
#--------------------------

```python
def test_qty_or_zero():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    # Test with defined price - should return the quantity
    defined_price = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert defined_price.qty_or_zero() == Decimal('1')
    
    # Test with undefined price - should return Decimal('0')
    undefined_price = Price.of(None, Decimal('1'), None)
    assert undefined_price.qty_or_zero() == Decimal('0')
    
    # Test with defined price and different quantity
    defined_price_2 = Price.of(Currencies["EUR"], Decimal('42.5'), Date(2020, 6, 15))
    assert defined_price_2.qty_or_zero() == Decimal('42.5')
    
    # Test with undefined price created via Price.na()
    na_price = Price.na()
    assert na_price.qty_or_zero() == Decimal('0')


# LLM-generated content at query #34
#--------------------------

```python
def test_nonmoney_constructor():
    none_money = NoneMoney()
    assert none_money is not None
    assert isinstance(none_money, NoneMoney)


# LLM-generated content at query #35
#--------------------------

```python
def test_money_eq():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    
    money1 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    money3 = Money.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    money4 = Money.of(Currencies["EUR"], Decimal('1'), Date(2019, 1, 1))
    money5 = Money.na()
    money6 = Money.na()
    
    assert money1 == money2
    assert not (money1 == money3)
    assert not (money1 == money4)
    assert money5 == money6
    assert not (money1 == money5)
    assert not (money1 == "not a money")
    assert not (money1 == None)
    assert not (money1 == 1)


# LLM-generated content at query #36
#--------------------------

```python
def test_money_gt():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    
    # Test: defined money > undefined money returns True
    defined_money = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    undefined_money = Money.na()
    assert defined_money.gt(undefined_money) == True
    
    # Test: undefined money > defined money returns False
    assert undefined_money.gt(defined_money) == False
    
    # Test: undefined money > undefined money returns False
    assert undefined_money.gt(undefined_money) == False
    
    # Test: defined money > defined money with same currency (greater quantity)
    money1 = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('50'), Date(2019, 1, 1))
    assert money1.gt(money2) == True
    
    # Test: defined money > defined money with same currency (lesser quantity)
    assert money2.gt(money1) == False
    
    # Test: defined money > defined money with same currency (equal quantity)
    money3 = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    money4 = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    assert money3.gt(money4) == False


# LLM-generated content at query #37
#--------------------------

```python
def test_price_as_float():
    from decimal import Decimal
    from datetime import date as Date
    from pypara.currencies import Currencies
    from pypara.price import Price
    
    # Test as_float with a defined price
    defined_price = Price.of(Currencies["USD"], Decimal('123.45'), Date(2019, 1, 1))
    result = defined_price.as_float()
    assert result == 123.45
    assert isinstance(result, float)
    
    # Test as_float with a defined price with integer quantity
    defined_price_int = Price.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    result_int = defined_price_int.as_float()
    assert result_int == 100.0
    assert isinstance(result_int, float)
    
    # Test as_float with a defined price with negative quantity
    defined_price_neg = Price.of(Currencies["USD"], Decimal('-50.5'), Date(2019, 1, 1))
    result_neg = defined_price_neg.as_float()
    assert result_neg == -50.5
    assert isinstance(result_neg, float)
    
    # Test as_float with undefined price raises MonetaryOperationException
    undefined_price = Price.na()
    try:
        undefined_price.as_float()
        assert False, "Should have raised MonetaryOperationException"
    except Exception as e:
        assert "MonetaryOperationException" in str(type(e).__name__)


# LLM-generated content at query #38
#--------------------------

```python
def test_someprice_lt():
    from decimal import Decimal
    from datetime import date
    
    # Create test currencies and dates
    usd = Currency(code="USD", quantizer=Decimal("0.01"))
    eur = Currency(code="EUR", quantizer=Decimal("0.01"))
    test_date = date(2024, 1, 1)
    
    # Create test prices
    price1 = SomePrice(usd, Decimal("100.00"), test_date)
    price2 = SomePrice(usd, Decimal("200.00"), test_date)
    price3 = SomePrice(usd, Decimal("100.00"), test_date)
    price_eur = SomePrice(eur, Decimal("100.00"), test_date)
    
    # Test: less than with smaller quantity
    assert price1.lt(price2) is True
    
    # Test: less than with equal quantity
    assert price1.lt(price3) is False
    
    # Test: less than with larger quantity
    assert price2.lt(price1) is False
    
    # Test: less than with non-SomePrice returns False
    assert price1.lt("not a price") is False
    
    # Test: less than with different currencies raises exception
    try:
        price1.lt(price_eur)
        assert False, "Should raise IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass


# LLM-generated content at query #39
#--------------------------

```python
def test_convert_with_valid_rate():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currency, CurrencyType
    from pypara.monetary import SomeMoney
    from pypara.fx import FXRateService, FXRate
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    
    money = SomeMoney(usd, Decimal("100.00"), date(2023, 1, 1))
    
    rate = FXRate(usd, eur, Decimal("0.92"), date(2023, 1, 1))
    original_service = FXRateService.default
    
    class MockFXRateService:
        def query(self, from_ccy, to_ccy, asof, strict):
            return rate
    
    FXRateService.default = MockFXRateService()
    
    result = money.convert(eur)
    
    FXRateService.default = original_service
    
    assert result.ccy == eur
    assert result.qty == Decimal("92.00")
    assert result.dov == date(2023, 1, 1)


def test_convert_with_explicit_asof_date():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currency, CurrencyType
    from pypara.monetary import SomeMoney
    from pypara.fx import FXRateService, FXRate
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    gbp = Currency.of("GBP", "British Pounds", 2, CurrencyType.MONEY)
    
    money = SomeMoney(usd, Decimal("100.00"), date(2023, 1, 1))
    asof_date = date(2023, 6, 15)
    
    rate = FXRate(usd, gbp, Decimal("0.79"), asof_date)
    original_service = FXRateService.default
    
    class MockFXRateService:
        def query(self, from_ccy, to_ccy, asof, strict):
            return rate
    
    FXRateService.default = MockFXRateService()
    
    result = money.convert(gbp, asof=asof_date)
    
    FXRateService.default = original_service
    
    assert result.ccy == gbp
    assert result.qty == Decimal("79.00")
    assert result.dov == asof_date


def test_convert_no_rate_non_strict():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currency, CurrencyType
    from pypara.monetary import SomeMoney, NoMoney
    from pypara.fx import FXRateService
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    
    money = SomeMoney(usd, Decimal("100.00"), date(2023, 1, 1))
    
    original_service = FXRateService.default
    
    class MockFXRateService:
        def query(self, from_ccy, to_ccy, asof, strict):
            return None
    
    FXRateService.default = MockFXRateService()
    
    result = money.convert(jpy, strict=False)
    
    FXRateService.default = original_service
    
    assert result == NoMoney


def test_convert_no_rate_strict_raises_error():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currency, CurrencyType
    from pypara.monetary import SomeMoney
    from pypara.fx import FXRateService, FXRateLookupError
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    chf = Currency.of("CHF", "Swiss Franc", 2, CurrencyType.MONEY)
    
    money = SomeMoney(usd, Decimal("100.00"), date(2023, 1, 1))
    
    original_service = FXRateService.default
    
    class MockFXRateService:
        def query(self, from_ccy, to_ccy, asof, strict):
            return None
    
    FXRateService.default = MockFXRateService()
    
    try:
        money.convert(chf, strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError:
        pass
    finally:
        FXRateService.default = original_service


def test_convert_with_quantization():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currency, CurrencyType
    from pypara.monetary import SomeMoney
    from pypara.fx import FXRateService, FXRate
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    
    money = SomeMoney(usd, Decimal("100.50"), date(2023, 1, 1))
    
    rate = FXRate(usd, jpy, Decimal("130.5"), date(2023, 1, 1))
    original_service = FXRateService.default
    
    class MockFXRateService:
        def query(self, from_ccy, to_ccy, asof, strict):
            return rate
    
    FXRateService.default = MockFXRateService()
    
    result = money.convert(jpy)
    
    FXRateService.default = original_service
    
    assert result.ccy == jpy
    assert result.qty == Decimal("13110")
    assert result.dov == date(2023, 1, 1)


def test_convert_no_default_service_raises_error():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currency, CurrencyType
    from pypara.monetary import SomeMoney
    from pypara.fx import FXRateService
    from pypara.errors import ProgrammingError
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    
    money = SomeMoney(usd, Decimal("100.00"), date(2023, 1, 1))
    
    original_service = FXRateService.default
    FXRateService.default = None
    
    try:
        money.convert(eur)
        assert False, "Expected ProgrammingError"
    except ProgrammingError:
        pass
    finally:
        FXRateService.default = original_service


# LLM-generated content at query #40
#--------------------------

```python
def test_scalar_add():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    
    # Test scalar_add on defined money
    money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = money.scalar_add(Decimal('5'))
    assert result.qty_or_none() == Decimal('15.00')
    assert result.ccy_or_none().code == "USD"
    
    # Test scalar_add on undefined money returns itself
    undefined_money = Money.na()
    result_undefined = undefined_money.scalar_add(Decimal('5'))
    assert result_undefined.undefined
    
    # Test scalar_add with negative number
    money2 = Money.of(Currencies["EUR"], Decimal('20'), Date(2019, 1, 1))
    result2 = money2.scalar_add(Decimal('-8'))
    assert result2.qty_or_none() == Decimal('12.00')
    
    # Test scalar_add with zero
    money3 = Money.of(Currencies["GBP"], Decimal('100'), Date(2019, 1, 1))
    result3 = money3.scalar_add(Decimal('0'))
    assert result3.qty_or_none() == Decimal('100.00')
    
    # Test scalar_add with decimal places
    money4 = Money.of(Currencies["USD"], Decimal('10.50'), Date(2019, 1, 1))
    result4 = money4.scalar_add(Decimal('0.25'))
    assert result4.qty_or_none() == Decimal('10.75')


# LLM-generated content at query #41
#--------------------------

```python
def test_ccy_or_returns_currency_when_money_is_defined():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    from pypara.money import Money
    
    somemoney = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    result = somemoney.ccy_or(Currencies["EUR"])
    
    assert result.code == "USD"


def test_ccy_or_returns_default_when_money_is_undefined():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from pypara.money import Money
    
    nonemoney = Money.of(Currencies["USD"], None, None)
    result = nonemoney.ccy_or(Currencies["EUR"])
    
    assert result.code == "EUR"


def test_ccy_or_with_na_money():
    from pypara.currencies import Currencies
    from pypara.money import Money
    
    nonemoney = Money.na()
    result = nonemoney.ccy_or(Currencies["USD"])
    
    assert result.code == "USD"


# LLM-generated content at query #42
#--------------------------

```python
def test_convert_with_defined_price():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    usd_price = Price.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    eur_price = usd_price.convert(Currencies["EUR"], asof=Date(2019, 1, 1))
    
    assert eur_price.defined
    assert eur_price.ccy_or_none().code == "EUR"


def test_convert_with_undefined_price():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    undefined_price = Price.na()
    result = undefined_price.convert(Currencies["EUR"])
    
    assert result.undefined


def test_convert_same_currency():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    usd_price = Price.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    converted = usd_price.convert(Currencies["USD"])
    
    assert converted.defined
    assert converted.ccy_or_none().code == "USD"
    assert converted.qty_or_zero() == Decimal('100')


def test_convert_with_asof_date():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    usd_price = Price.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    eur_price = usd_price.convert(Currencies["EUR"], asof=Date(2019, 6, 15))
    
    assert eur_price.defined
    assert eur_price.dov_or_none() == Date(2019, 6, 15)


def test_convert_strict_mode():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    usd_price = Price.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    converted = usd_price.convert(Currencies["EUR"], strict=True)
    
    assert converted is not None


def test_convert_preserves_quantity_on_same_currency():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    gbp_price = Price.of(Currencies["GBP"], Decimal('50'), Date(2019, 1, 1))
    result = gbp_price.convert(Currencies["GBP"])
    
    assert result.qty_or_zero() == Decimal('50')


# LLM-generated content at query #43
#--------------------------

```python
def test_price_le():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.price import Price

    # Test case 1: defined price <= defined price with same currency (less than)
    price1 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    assert price1 <= price2

    # Test case 2: defined price <= defined price with same currency (equal)
    price3 = Price.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    price4 = Price.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    assert price3 <= price4

    # Test case 3: defined price <= defined price with same currency (greater than)
    price5 = Price.of(Currencies["USD"], Decimal('3'), Date(2019, 1, 1))
    price6 = Price.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    assert not (price5 <= price6)

    # Test case 4: undefined price <= defined price
    undefined_price = Price.na()
    defined_price = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert undefined_price <= defined_price

    # Test case 5: undefined price <= undefined price
    undefined_price1 = Price.na()
    undefined_price2 = Price.na()
    assert undefined_price1 <= undefined_price2

    # Test case 6: defined price <= undefined price
    defined_price2 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    undefined_price3 = Price.na()
    assert not (defined_price2 <= undefined_price3)


# LLM-generated content at query #44
#--------------------------

```python
def test_convert_with_valid_currency_and_date():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    usd = Currencies["USD"]
    eur = Currencies["EUR"]
    money_usd = Money.of(usd, Decimal('100'), Date(2019, 1, 1))
    
    converted_money = money_usd.convert(eur, asof=Date(2019, 1, 1))
    
    assert converted_money is not None


def test_convert_undefined_money_returns_itself():
    from pypara.currencies import Currencies
    from datetime import date as Date
    
    eur = Currencies["EUR"]
    undefined_money = Money.na()
    
    result = undefined_money.convert(eur, asof=Date(2019, 1, 1))
    
    assert result.undefined


def test_convert_without_asof_date():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    usd = Currencies["USD"]
    eur = Currencies["EUR"]
    money_usd = Money.of(usd, Decimal('100'), Date(2019, 1, 1))
    
    converted_money = money_usd.convert(eur)
    
    assert converted_money is not None


def test_convert_same_currency():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    usd = Currencies["USD"]
    money_usd = Money.of(usd, Decimal('100'), Date(2019, 1, 1))
    
    converted_money = money_usd.convert(usd, asof=Date(2019, 1, 1))
    
    assert converted_money.defined


def test_convert_with_strict_mode():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    usd = Currencies["USD"]
    eur = Currencies["EUR"]
    money_usd = Money.of(usd, Decimal('100'), Date(2019, 1, 1))
    
    converted_money = money_usd.convert(eur, asof=Date(2019, 1, 1), strict=True)
    
    assert converted_money is not None


# LLM-generated content at query #45
#--------------------------

```python
def test_someprice_sub():
    from decimal import Decimal
    from datetime import date
    
    # Create test currencies and prices
    usd = Currency(code="USD", quantizer=Decimal("0.01"))
    eur = Currency(code="EUR", quantizer=Decimal("0.01"))
    
    price1 = SomePrice(usd, Decimal("100.00"), date(2024, 1, 1))
    price2 = SomePrice(usd, Decimal("30.00"), date(2024, 1, 1))
    price3 = SomePrice(eur, Decimal("50.00"), date(2024, 1, 1))
    
    # Test basic subtraction with same currency
    result = price1 - price2
    assert result.ccy == usd
    assert result.qty == Decimal("70.00")
    assert result.dov == date(2024, 1, 1)
    
    # Test subtraction with different dates (should use later date)
    price4 = SomePrice(usd, Decimal("20.00"), date(2024, 1, 15))
    result = price1 - price4
    assert result.dov == date(2024, 1, 15)
    assert result.qty == Decimal("80.00")
    
    # Test subtraction with NoPrice (undefined) should return self
    result = price1 - NoPrice
    assert result == price1
    
    # Test subtraction with incompatible currencies should raise error
    try:
        result = price1 - price3
        assert False, "Should have raised IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass
    
    # Test subtraction resulting in negative quantity
    result = price2 - price1
    assert result.qty == Decimal("-70.00")
    assert result.ccy == usd
    
    # Test subtraction with zero
    price_zero = SomePrice(usd, Decimal("0.00"), date(2024, 1, 1))
    result = price1 - price_zero
    assert result.qty == Decimal("100.00")


# LLM-generated content at query #46
#--------------------------

```python
def test_qty_or_zero():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.money import Money
    
    somemoney = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    result_defined = somemoney.qty_or_zero()
    assert result_defined == Decimal('1.00')
    
    nonemoney = Money.of(None, Decimal('1'), None)
    result_undefined = nonemoney.qty_or_zero()
    assert result_undefined == Decimal('0')


# LLM-generated content at query #47
#--------------------------

```python
def test_money_ge_operator():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.money import Money
    
    # Test: defined money >= defined money with same currency and greater quantity
    money1 = Money.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert (money1 >= money2) == True
    
    # Test: defined money >= defined money with same currency and equal quantity
    money3 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    money4 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert (money3 >= money4) == True
    
    # Test: defined money >= defined money with same currency and lesser quantity
    money5 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    money6 = Money.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    assert (money5 >= money6) == False
    
    # Test: defined money >= undefined money
    money7 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    money8 = Money.na()
    assert (money7 >= money8) == True
    
    # Test: undefined money >= undefined money
    money9 = Money.na()
    money10 = Money.na()
    assert (money9 >= money10) == True
    
    # Test: undefined money >= defined money
    money11 = Money.na()
    money12 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert (money11 >= money12) == False


# LLM-generated content at query #48
#--------------------------

```python
def test_price_int():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    # Test defined price with integer quantity
    defined_price = Price.of(Currencies["USD"], Decimal('42'), Date(2019, 1, 1))
    result = int(defined_price)
    assert result == 42
    
    # Test defined price with decimal quantity
    defined_price_decimal = Price.of(Currencies["USD"], Decimal('42.7'), Date(2019, 1, 1))
    result_decimal = int(defined_price_decimal)
    assert result_decimal == 42
    
    # Test undefined price raises exception
    undefined_price = Price.na()
    try:
        int(undefined_price)
        assert False, "Expected MonetaryOperationException"
    except Exception as e:
        assert "MonetaryOperationException" in str(type(e))


# LLM-generated content at query #49
#--------------------------

```python
def test_price_eq():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    price1 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    price3 = Price.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    price4 = Price.of(Currencies["EUR"], Decimal('1'), Date(2019, 1, 1))
    price5 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 2))
    undefined1 = Price.na()
    undefined2 = Price.na()
    
    assert price1 == price2
    assert price1 != price3
    assert price1 != price4
    assert price1 != price5
    assert undefined1 == undefined2
    assert price1 != undefined1
    assert price1 != "not a price"
    assert price1 != None


# LLM-generated content at query #50
#--------------------------

```python
def test_subtract_two_defined_money_same_currency():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money1 = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('30'), Date(2019, 1, 1))
    result = money1.subtract(money2)
    
    assert result.defined
    assert result.qty == Decimal('70.00')
    assert result.ccy.code == 'USD'


def test_subtract_two_defined_money_different_currency_raises_error():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money1 = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["EUR"], Decimal('30'), Date(2019, 1, 1))
    
    try:
        money1.subtract(money2)
        assert False, "Expected IncompatibleCurrencyError"
    except Exception as e:
        assert "IncompatibleCurrencyError" in str(type(e))


def test_subtract_defined_money_with_undefined_money():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money1 = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    money2 = Money.na()
    result = money1.subtract(money2)
    
    assert result is money1


def test_subtract_undefined_money_with_defined_money():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money1 = Money.na()
    money2 = Money.of(Currencies["USD"], Decimal('30'), Date(2019, 1, 1))
    result = money1.subtract(money2)
    
    assert result is money2


def test_subtract_undefined_money_with_undefined_money():
    money1 = Money.na()
    money2 = Money.na()
    result = money1.subtract(money2)
    
    assert result.undefined


def test_subtract_negative_quantity():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money1 = Money.of(Currencies["USD"], Decimal('30'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    result = money1.subtract(money2)
    
    assert result.defined
    assert result.qty == Decimal('-70.00')


def test_subtract_carries_forward_date():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money1 = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 5))
    money2 = Money.of(Currencies["USD"], Decimal('30'), Date(2019, 1, 1))
    result = money1.subtract(money2)
    
    assert result.dov == Date(2019, 1, 5)


# LLM-generated content at query #51
#--------------------------

```python
def test_money_add_defined_same_currency():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money1 = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('20'), Date(2019, 1, 1))
    result = money1.add(money2)
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('30.00')
    assert result.ccy_or_none().code == "USD"


def test_money_add_defined_different_currency_raises_error():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money1 = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["EUR"], Decimal('20'), Date(2019, 1, 1))
    
    try:
        result = money1.add(money2)
        assert False, "Should raise IncompatibleCurrencyError"
    except Exception as e:
        assert "IncompatibleCurrencyError" in str(type(e))


def test_money_add_with_undefined_first_operand():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money1 = Money.na()
    money2 = Money.of(Currencies["USD"], Decimal('20'), Date(2019, 1, 1))
    result = money1.add(money2)
    
    assert result is money2


def test_money_add_with_undefined_second_operand():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money1 = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    money2 = Money.na()
    result = money1.add(money2)
    
    assert result is money1


def test_money_add_both_undefined():
    money1 = Money.na()
    money2 = Money.na()
    result = money1.add(money2)
    
    assert result.undefined


def test_money_add_carries_forward_date():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money1 = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('20'), Date(2019, 1, 5))
    result = money1.add(money2)
    
    assert result.dov_or_none() == Date(2019, 1, 5)


# LLM-generated content at query #52
#--------------------------

```python
def test_money_truediv():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.money import Money
    
    # Test division of defined money by a positive number
    money = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    result = money / Decimal('2')
    assert result.defined
    assert result.qty_or_zero() == Decimal('50.00')
    assert result.ccy_or_none().code == "USD"
    
    # Test division of defined money by a negative number
    money = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    result = money / Decimal('-2')
    assert result.defined
    assert result.qty_or_zero() == Decimal('-50.00')
    
    # Test division of defined money by 1
    money = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    result = money / Decimal('1')
    assert result.defined
    assert result.qty_or_zero() == Decimal('100.00')
    
    # Test division of defined money by a fractional number
    money = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    result = money / Decimal('0.5')
    assert result.defined
    assert result.qty_or_zero() == Decimal('200.00')
    
    # Test division by zero yields undefined money
    money = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    result = money / Decimal('0')
    assert result.undefined
    
    # Test division of undefined money returns undefined
    money = Money.na()
    result = money / Decimal('2')
    assert result.undefined
    
    # Test division with integer
    money = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    result = money / 2
    assert result.defined
    assert result.qty_or_zero() == Decimal('50.00')
    
    # Test division with float
    money = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    result = money / 2.0
    assert result.defined
    assert result.qty_or_zero() == Decimal('50.00')


# LLM-generated content at query #53
#--------------------------

```python
def test_price_abs():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    from pypara.price import Price

    # Test abs on a defined price with positive quantity
    positive_price = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    abs_positive = positive_price.abs()
    assert abs_positive.qty_or_zero() == Decimal('10')
    assert abs_positive.ccy_or_none().code == 'USD'

    # Test abs on a defined price with negative quantity
    negative_price = Price.of(Currencies["USD"], Decimal('-10'), Date(2019, 1, 1))
    abs_negative = negative_price.abs()
    assert abs_negative.qty_or_zero() == Decimal('10')
    assert abs_negative.ccy_or_none().code == 'USD'

    # Test abs on an undefined price
    undefined_price = Price.na()
    abs_undefined = undefined_price.abs()
    assert abs_undefined.undefined

    # Test abs on a price with zero quantity
    zero_price = Price.of(Currencies["USD"], Decimal('0'), Date(2019, 1, 1))
    abs_zero = zero_price.abs()
    assert abs_zero.qty_or_zero() == Decimal('0')


# LLM-generated content at query #54
#--------------------------

```python
def test_price_float():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    # Test defined price with positive quantity
    price_defined = Price.of(Currencies["USD"], Decimal('123.45'), Date(2019, 1, 1))
    assert float(price_defined) == 123.45
    
    # Test defined price with negative quantity
    price_negative = Price.of(Currencies["EUR"], Decimal('-50.75'), Date(2020, 6, 15))
    assert float(price_negative) == -50.75
    
    # Test defined price with zero quantity
    price_zero = Price.of(Currencies["GBP"], Decimal('0'), Date(2021, 3, 10))
    assert float(price_zero) == 0.0
    
    # Test undefined price raises exception
    price_undefined = Price.na()
    try:
        float(price_undefined)
        assert False, "Should raise MonetaryOperationException"
    except Exception:
        pass


# LLM-generated content at query #55
#--------------------------

```python
def test_divide():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    
    # Test divide with defined money
    money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = money.divide(Decimal('2'))
    assert result.qty_or_zero() == Decimal('5.00')
    assert result.ccy_or_none().code == "USD"
    
    # Test divide by zero yields undefined money
    money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = money.divide(Decimal('0'))
    assert result.undefined
    
    # Test divide undefined money returns itself
    undefined_money = Money.na()
    result = undefined_money.divide(Decimal('2'))
    assert result.undefined
    
    # Test divide with decimal divisor
    money = Money.of(Currencies["EUR"], Decimal('100'), Date(2020, 6, 15))
    result = money.divide(Decimal('4'))
    assert result.qty_or_zero() == Decimal('25.00')
    
    # Test divide resulting in fractional quantity
    money = Money.of(Currencies["GBP"], Decimal('10'), Date(2021, 3, 10))
    result = money.divide(Decimal('3'))
    assert result.defined
    assert result.ccy_or_none().code == "GBP"


# LLM-generated content at query #56
#--------------------------

```python
def test_with_dov():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.price import Price

    # Test with_dov on a defined price object
    original_price = Price.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    new_date = Date(2020, 6, 15)
    new_price = original_price.with_dov(new_date)
    
    assert new_price.dov_or_none() == new_date
    assert new_price.ccy_or_none().code == "USD"
    assert new_price.qty_or_none() == Decimal('100')
    
    # Test with_dov on an undefined price object
    undefined_price = Price.na()
    result = undefined_price.with_dov(Date(2020, 1, 1))
    
    assert result is undefined_price
    assert result.undefined is True


# LLM-generated content at query #57
#--------------------------

```python
def test_ccy_or_defined_price():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    someprice = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    result = someprice.ccy_or(Currencies["EUR"])
    assert result.code == 'USD'


def test_ccy_or_undefined_price():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    noneprice = Price.of(None, Decimal('1'), None)
    result = noneprice.ccy_or(Currencies["EUR"])
    assert result.code == 'EUR'


def test_ccy_or_returns_default_when_ccy_is_none():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    price_with_none_ccy = Price.of(None, Decimal('5'), Date(2019, 1, 1))
    result = price_with_none_ccy.ccy_or(Currencies["GBP"])
    assert result.code == 'GBP'


def test_ccy_or_returns_own_currency_when_defined():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    price = Price.of(Currencies["JPY"], Decimal('100'), Date(2020, 6, 15))
    result = price.ccy_or(Currencies["USD"])
    assert result.code == 'JPY'


# LLM-generated content at query #58
#--------------------------

```python
def test_with_qty():
    from datetime import date as Date
    from decimal import Decimal
    from pypara.currencies import Currencies
    from pypara.price import Price

    # Test with_qty on a defined price object
    original_price = Price.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    new_qty = Decimal('200')
    updated_price = original_price.with_qty(new_qty)
    
    assert updated_price.qty_or_none() == Decimal('200')
    assert updated_price.ccy_or_none().code == 'USD'
    assert updated_price.dov_or_none() == Date(2019, 1, 1)
    assert updated_price.defined is True

    # Test with_qty on an undefined price object
    undefined_price = Price.na()
    result = undefined_price.with_qty(Decimal('300'))
    
    assert result.undefined is True
    assert result is undefined_price

    # Test with_qty with different quantity values
    price = Price.of(Currencies["EUR"], Decimal('50'), Date(2020, 6, 15))
    price_with_zero = price.with_qty(Decimal('0'))
    
    assert price_with_zero.qty_or_none() == Decimal('0')
    assert price_with_zero.ccy_or_none().code == 'EUR'
    
    # Test with_qty with negative quantity
    price_with_negative = price.with_qty(Decimal('-50'))
    
    assert price_with_negative.qty_or_none() == Decimal('-50')
    assert price_with_negative.defined is True


# LLM-generated content at query #59
#--------------------------

```python
def test_as_integer():
    from decimal import Decimal
    from datetime import date as Date
    from pypara.currencies import Currencies
    from pypara.price import Price
    
    # Test with defined price - integer quantity
    price_int = Price.of(Currencies["USD"], Decimal('42'), Date(2019, 1, 1))
    assert price_int.as_integer() == 42
    
    # Test with defined price - decimal quantity
    price_decimal = Price.of(Currencies["USD"], Decimal('42.7'), Date(2019, 1, 1))
    assert price_decimal.as_integer() == 42
    
    # Test with defined price - negative quantity
    price_negative = Price.of(Currencies["USD"], Decimal('-42'), Date(2019, 1, 1))
    assert price_negative.as_integer() == -42
    
    # Test with defined price - zero quantity
    price_zero = Price.of(Currencies["USD"], Decimal('0'), Date(2019, 1, 1))
    assert price_zero.as_integer() == 0
    
    # Test with undefined price - should raise MonetaryOperationException
    undefined_price = Price.na()
    try:
        undefined_price.as_integer()
        assert False, "Expected MonetaryOperationException"
    except Exception as e:
        assert "MonetaryOperationException" in type(e).__name__


# LLM-generated content at query #60
#--------------------------

```python
def test_somemoney_sub():
    from decimal import Decimal
    from datetime import date
    
    # Create mock Currency objects
    class MockCurrency:
        def __init__(self, code, decimals=2):
            self.code = code
            self.decimals = decimals
            self.quantizer = Decimal(10) ** -decimals
        
        def __eq__(self, other):
            return self.code == other.code
        
        def __ne__(self, other):
            return not self.__eq__(other)
    
    usd = MockCurrency("USD", 2)
    eur = MockCurrency("EUR", 2)
    
    # Test basic subtraction with same currency
    money1 = SomeMoney(usd, Decimal("100.00"), date(2023, 1, 1))
    money2 = SomeMoney(usd, Decimal("30.00"), date(2023, 1, 1))
    result = money1 - money2
    
    assert result.ccy == usd
    assert result.qty == Decimal("70.00")
    assert result.dov == date(2023, 1, 1)
    
    # Test subtraction with different dates (should use later date)
    money3 = SomeMoney(usd, Decimal("100.00"), date(2023, 1, 1))
    money4 = SomeMoney(usd, Decimal("30.00"), date(2023, 1, 15))
    result2 = money3 - money4
    
    assert result2.qty == Decimal("70.00")
    assert result2.dov == date(2023, 1, 15)
    
    # Test subtraction with negative result
    money5 = SomeMoney(usd, Decimal("30.00"), date(2023, 1, 1))
    money6 = SomeMoney(usd, Decimal("100.00"), date(2023, 1, 1))
    result3 = money5 - money6
    
    assert result3.qty == Decimal("-70.00")
    
    # Test subtraction with NoMoney (undefined) should return self
    money7 = SomeMoney(usd, Decimal("100.00"), date(2023, 1, 1))
    no_money = type('NoMoney', (), {'undefined': True})()
    result4 = money7.subtract(no_money)
    
    assert result4 == money7
    
    # Test subtraction with incompatible currency raises error
    money8 = SomeMoney(usd, Decimal("100.00"), date(2023, 1, 1))
    money9 = SomeMoney(eur, Decimal("50.00"), date(2023, 1, 1))
    
    try:
        money8 - money9
        assert False, "Should have raised IncompatibleCurrencyError"
    except Exception as e:
        assert "IncompatibleCurrencyError" in str(type(e))


# LLM-generated content at query #61
#--------------------------

```python
def test_is_equal_with_same_defined_prices():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    price1 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    
    assert price1.is_equal(price2) is True


def test_is_equal_with_different_quantities():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    price1 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    
    assert price1.is_equal(price2) is False


def test_is_equal_with_different_currencies():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    price1 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["EUR"], Decimal('1'), Date(2019, 1, 1))
    
    assert price1.is_equal(price2) is False


def test_is_equal_with_different_dates():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    price1 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 2))
    
    assert price1.is_equal(price2) is False


def test_is_equal_with_undefined_prices():
    price1 = Price.na()
    price2 = Price.na()
    
    assert price1.is_equal(price2) is True


def test_is_equal_with_defined_and_undefined():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    price1 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    price2 = Price.na()
    
    assert price1.is_equal(price2) is False


def test_is_equal_with_non_price_object():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    price = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    
    assert price.is_equal("not a price") is False


def test_is_equal_with_none_object():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    price = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    
    assert price.is_equal(None) is False


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_somemoney_neg():
    from decimal import Decimal
    from datetime import date
    
    ccy = Currency(code="USD", decimals=2)
    qty = Decimal("100.50")
    dov = date(2023, 1, 15)
    
    money = SomeMoney(ccy, qty, dov)
    negated_money = -money
    
    assert negated_money.ccy == ccy
    assert negated_money.qty == Decimal("-100.50")
    assert negated_money.dov == dov
    assert isinstance(negated_money, SomeMoney)


# LLM-generated content at query #2
#--------------------------

```python
def test_qty_or_zero():
    from decimal import Decimal
    from datetime import date
    
    # Create a mock Currency object
    class MockCurrency:
        def __init__(self, code, decimals=2):
            self.code = code
            self.decimals = decimals
            self.quantizer = Decimal(10) ** -decimals
        
        def __eq__(self, other):
            return self.code == other.code
        
        def __ne__(self, other):
            return not self.__eq__(other)
    
    ccy = MockCurrency("USD")
    qty = Decimal("100.50")
    dov = date(2023, 1, 1)
    
    money = SomeMoney(ccy, qty, dov)
    result = money.qty_or_zero()
    
    assert result == Decimal("100.50")
    assert isinstance(result, Decimal)


def test_qty_or_zero_with_zero_quantity():
    from decimal import Decimal
    from datetime import date
    
    class MockCurrency:
        def __init__(self, code, decimals=2):
            self.code = code
            self.decimals = decimals
            self.quantizer = Decimal(10) ** -decimals
        
        def __eq__(self, other):
            return self.code == other.code
        
        def __ne__(self, other):
            return not self.__eq__(other)
    
    ccy = MockCurrency("EUR")
    qty = Decimal("0")
    dov = date(2023, 1, 1)
    
    money = SomeMoney(ccy, qty, dov)
    result = money.qty_or_zero()
    
    assert result == Decimal("0")


def test_qty_or_zero_with_negative_quantity():
    from decimal import Decimal
    from datetime import date
    
    class MockCurrency:
        def __init__(self, code, decimals=2):
            self.code = code
            self.decimals = decimals
            self.quantizer = Decimal(10) ** -decimals
        
        def __eq__(self, other):
            return self.code == other.code
        
        def __ne__(self, other):
            return not self.__eq__(other)
    
    ccy = MockCurrency("GBP")
    qty = Decimal("-50.25")
    dov = date(2023, 1, 1)
    
    money = SomeMoney(ccy, qty, dov)
    result = money.qty_or_zero()
    
    assert result == Decimal("-50.25")


# LLM-generated content at query #3
#--------------------------

```python
def test_subtract_defined_money_objects():
    from decimal import Decimal
    from datetime import date as Date
    from pypara.currencies import Currencies
    from pypara.money import Money
    
    money1 = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('3'), Date(2019, 1, 1))
    result = money1.subtract(money2)
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('7.00')
    assert result.ccy_or_none().code == 'USD'


def test_subtract_with_undefined_operand_left():
    from decimal import Decimal
    from datetime import date as Date
    from pypara.currencies import Currencies
    from pypara.money import Money
    
    undefined_money = Money.na()
    money2 = Money.of(Currencies["USD"], Decimal('3'), Date(2019, 1, 1))
    result = undefined_money.subtract(money2)
    
    assert result is money2


def test_subtract_with_undefined_operand_right():
    from decimal import Decimal
    from datetime import date as Date
    from pypara.currencies import Currencies
    from pypara.money import Money
    
    money1 = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    undefined_money = Money.na()
    result = money1.subtract(undefined_money)
    
    assert result is money1


def test_subtract_both_undefined():
    from pypara.money import Money
    
    undefined_money1 = Money.na()
    undefined_money2 = Money.na()
    result = undefined_money1.subtract(undefined_money2)
    
    assert result.undefined


def test_subtract_incompatible_currencies():
    from decimal import Decimal
    from datetime import date as Date
    from pypara.currencies import Currencies
    from pypara.money import Money
    
    money_usd = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    money_eur = Money.of(Currencies["EUR"], Decimal('3'), Date(2019, 1, 1))
    
    try:
        result = money_usd.subtract(money_eur)
        assert False, "Should raise IncompatibleCurrencyError"
    except Exception as e:
        assert "IncompatibleCurrencyError" in str(type(e).__name__)


def test_subtract_negative_result():
    from decimal import Decimal
    from datetime import date as Date
    from pypara.currencies import Currencies
    from pypara.money import Money
    
    money1 = Money.of(Currencies["USD"], Decimal('3'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = money1.subtract(money2)
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('-7.00')


def test_subtract_zero_result():
    from decimal import Decimal
    from datetime import date as Date
    from pypara.currencies import Currencies
    from pypara.money import Money
    
    money1 = Money.of(Currencies["USD"], Decimal('5'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('5'), Date(2019, 1, 1))
    result = money1.subtract(money2)
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('0.00')


def test_subtract_carries_forward_date():
    from decimal import Decimal
    from datetime import date as Date
    from pypara.currencies import Currencies
    from pypara.money import Money
    
    money1 = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 5))
    money2 = Money.of(Currencies["USD"], Decimal('3'), Date(2019, 1, 1))
    result = money1.subtract(money2)
    
    assert result.dov_or_none() == Date(2019, 1, 5)


# LLM-generated content at query #4
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


def test_price_ge_undefined_prices():
    undefined1 = Price.na()
    undefined2 = Price.na()
    
    assert undefined1.gte(undefined2) == True


def test_price_ge_undefined_and_defined():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    undefined = Price.na()
    defined = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    
    assert undefined.gte(defined) == False
    assert defined.gte(undefined) == True


def test_price_ge_defined_prices_different_currency():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    price_usd = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    price_eur = Price.of(Currencies["EUR"], Decimal('10'), Date(2019, 1, 1))
    
    try:
        price_usd.gte(price_eur)
        assert False, "Should raise IncompatibleCurrencyError"
    except Exception as e:
        assert "IncompatibleCurrencyError" in str(type(e))


def test_price_ge_equal_quantities():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    price1 = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    
    assert price1.gte(price2) == True
    assert price2.gte(price1) == True


# LLM-generated content at query #5
#--------------------------

```python
def test_price_bool():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    # Test defined price with non-zero quantity should be True
    defined_price = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert bool(defined_price) is True
    
    # Test defined price with zero quantity should be False
    zero_price = Price.of(Currencies["USD"], Decimal('0'), Date(2019, 1, 1))
    assert bool(zero_price) is False
    
    # Test undefined price should be False
    undefined_price = Price.na()
    assert bool(undefined_price) is False
    
    # Test defined price with negative quantity should be True
    negative_price = Price.of(Currencies["USD"], Decimal('-5'), Date(2019, 1, 1))
    assert bool(negative_price) is True
    
    # Test defined price with positive quantity should be True
    positive_price = Price.of(Currencies["EUR"], Decimal('100'), Date(2019, 1, 2))
    assert bool(positive_price) is True


# LLM-generated content at query #6
#--------------------------

```python
def test_somemoney_constructor():
    from decimal import Decimal
    from datetime import date
    
    # Create test data
    usd = Currency(code="USD", decimals=2)
    quantity = Decimal("100.50")
    dov = date(2024, 1, 15)
    
    # Test constructor with all parameters
    money = SomeMoney(ccy=usd, qty=quantity, dov=dov)
    
    # Verify all fields are set correctly
    assert money.ccy == usd
    assert money.qty == quantity
    assert money.dov == dov
    
    # Verify tuple unpacking works
    c, q, d = money
    assert c == usd
    assert q == quantity
    assert d == dov
    
    # Verify indexing works
    assert money[0] == usd
    assert money[1] == quantity
    assert money[2] == dov
    
    # Verify defined property
    assert money.defined is True
    assert money.undefined is False


# LLM-generated content at query #7
#--------------------------

```python
def test_someprice_add():
    from decimal import Decimal
    from datetime import date
    
    # Create test currencies and prices
    usd = Currency(code="USD", quantizer=Decimal("0.01"))
    eur = Currency(code="EUR", quantizer=Decimal("0.01"))
    
    price1 = SomePrice(ccy=usd, qty=Decimal("100.50"), dov=date(2023, 1, 1))
    price2 = SomePrice(ccy=usd, qty=Decimal("50.25"), dov=date(2023, 1, 15))
    price3 = SomePrice(ccy=eur, qty=Decimal("75.00"), dov=date(2023, 1, 10))
    
    # Test adding two prices with same currency
    result = price1 + price2
    assert result.ccy == usd
    assert result.qty == Decimal("150.75")
    assert result.dov == date(2023, 1, 15)  # Should be the later date
    
    # Test adding with undefined price
    result = price1 + NoPrice
    assert result == price1
    
    # Test adding prices with different currencies raises error
    try:
        price1 + price3
        assert False, "Should raise IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass
    
    # Test adding with zero quantity
    price_zero = SomePrice(ccy=usd, qty=Decimal("0"), dov=date(2023, 1, 1))
    result = price1 + price_zero
    assert result.qty == Decimal("100.50")
    
    # Test adding negative quantities
    price_negative = SomePrice(ccy=usd, qty=Decimal("-30.00"), dov=date(2023, 1, 5))
    result = price1 + price_negative
    assert result.qty == Decimal("70.50")


# LLM-generated content at query #8
#--------------------------

```python
def test_qty_or_zero():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    
    # Test with defined money - should return the quantity
    somemoney = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    result = somemoney.qty_or_zero()
    assert result == Decimal('1.00')
    
    # Test with undefined money - should return 0
    nonemoney = Money.of(None, Decimal('1'), None)
    result = nonemoney.qty_or_zero()
    assert result == Decimal('0')
    
    # Test with Money.na() - should return 0
    result = Money.na().qty_or_zero()
    assert result == Decimal('0')
    
    # Test with different quantity values
    somemoney2 = Money.of(Currencies["EUR"], Decimal('42.50'), Date(2020, 6, 15))
    result = somemoney2.qty_or_zero()
    assert result == Decimal('42.50')


# LLM-generated content at query #9
#--------------------------

```python
def test_qty_or():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    # Test with defined price - should return the quantity
    someprice = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    result = someprice.qty_or(Decimal(0))
    assert result == Decimal('1')
    
    # Test with undefined price - should return the default value
    noneprice = Price.of(None, Decimal('1'), None)
    result = noneprice.qty_or(Decimal(0))
    assert result == Decimal('0')
    
    # Test with different default value
    someprice = Price.of(Currencies["EUR"], Decimal('5'), Date(2020, 6, 15))
    result = someprice.qty_or(Decimal(100))
    assert result == Decimal('5')
    
    # Test undefined price with different default
    noneprice = Price.of(None, Decimal('10'), None)
    result = noneprice.qty_or(Decimal(42))
    assert result == Decimal('42')


# LLM-generated content at query #10
#--------------------------

```python
def test_lt_defined_prices_with_same_currency():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    price1 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    
    assert price1.lt(price2) is True
    assert price2.lt(price1) is False


def test_lt_defined_prices_with_different_currency():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    price1 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["EUR"], Decimal('2'), Date(2019, 1, 1))
    
    try:
        price1.lt(price2)
        assert False, "Should raise IncompatibleCurrencyError"
    except Exception:
        pass


def test_lt_undefined_price_less_than_defined():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    undefined_price = Price.na()
    defined_price = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    
    assert undefined_price.lt(defined_price) is True


def test_lt_undefined_price_not_less_than_undefined():
    undefined_price1 = Price.na()
    undefined_price2 = Price.na()
    
    assert undefined_price1.lt(undefined_price2) is False


def test_lt_defined_not_less_than_undefined():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    defined_price = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    undefined_price = Price.na()
    
    assert defined_price.lt(undefined_price) is False


def test_lt_equal_defined_prices():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    price1 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    
    assert price1.lt(price2) is False


# LLM-generated content at query #11
#--------------------------

```python
def test_price_convert_with_valid_currency():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    price = Price.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    converted_price = price.convert(Currencies["EUR"], asof=Date(2019, 1, 1))
    
    assert converted_price.defined
    assert converted_price.ccy_or_none().code == "EUR"


def test_price_convert_undefined_price():
    from pypara.currencies import Currencies
    from datetime import date as Date
    
    undefined_price = Price.na()
    converted_price = undefined_price.convert(Currencies["EUR"], asof=Date(2019, 1, 1))
    
    assert converted_price.undefined


def test_price_convert_with_asof_date():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    price = Price.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    converted_price = price.convert(Currencies["EUR"], asof=Date(2019, 6, 15))
    
    assert converted_price.defined
    assert converted_price.dov_or_none() == Date(2019, 6, 15)


def test_price_convert_without_asof_date():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    price = Price.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    converted_price = price.convert(Currencies["EUR"])
    
    assert converted_price.defined
    assert converted_price.ccy_or_none().code == "EUR"


def test_price_convert_same_currency():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    price = Price.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    converted_price = price.convert(Currencies["USD"], asof=Date(2019, 1, 1))
    
    assert converted_price.defined
    assert converted_price.ccy_or_none().code == "USD"
    assert converted_price.qty_or_zero() == Decimal('100')


def test_price_convert_strict_mode():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    price = Price.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    converted_price = price.convert(Currencies["EUR"], asof=Date(2019, 1, 1), strict=True)
    
    assert converted_price.defined or converted_price.undefined


# LLM-generated content at query #12
#--------------------------

```python
def test_somemoney_bool():
    from decimal import Decimal
    from datetime import date
    
    # Mock Currency class
    class MockCurrency:
        def __init__(self, code, decimals=2):
            self.code = code
            self.decimals = decimals
            self.quantizer = Decimal(10) ** -decimals
    
    # Create test instances
    usd = MockCurrency("USD", 2)
    
    # Test with non-zero quantity (should be True)
    money_positive = SomeMoney(usd, Decimal("100.00"), date(2023, 1, 1))
    assert bool(money_positive) is True
    
    # Test with zero quantity (should be False)
    money_zero = SomeMoney(usd, Decimal("0.00"), date(2023, 1, 1))
    assert bool(money_zero) is False
    
    # Test with negative quantity (should be True)
    money_negative = SomeMoney(usd, Decimal("-50.00"), date(2023, 1, 1))
    assert bool(money_negative) is True
    
    # Test with small positive quantity (should be True)
    money_small = SomeMoney(usd, Decimal("0.01"), date(2023, 1, 1))
    assert bool(money_small) is True


# LLM-generated content at query #13
#--------------------------

```python
def test_truediv_divides_quantity_by_numeric():
    from decimal import Decimal
    from datetime import date
    
    currency = type('Currency', (), {'quantizer': Decimal('0.01'), 'decimals': 2})()
    money = type('SomeMoney', (), {
        'ccy': currency,
        'qty': Decimal('100.00'),
        'dov': date(2024, 1, 1),
        '__getitem__': lambda self, i: [currency, Decimal('100.00'), date(2024, 1, 1)][i],
        '__truediv__': SomeMoney.__truediv__
    })()
    
    result = money / 2
    assert result.qty == Decimal('50.00')
    assert result.ccy == currency
    assert result.dov == date(2024, 1, 1)


def test_truediv_quantizes_result():
    from decimal import Decimal
    from datetime import date
    
    currency = type('Currency', (), {'quantizer': Decimal('0.01'), 'decimals': 2})()
    money = type('SomeMoney', (), {
        'ccy': currency,
        'qty': Decimal('100.00'),
        'dov': date(2024, 1, 1),
        '__getitem__': lambda self, i: [currency, Decimal('100.00'), date(2024, 1, 1)][i],
        '__truediv__': SomeMoney.__truediv__
    })()
    
    result = money / 3
    assert result.qty == Decimal('33.33')


def test_truediv_by_zero_returns_no_money():
    from decimal import Decimal
    from datetime import date
    
    currency = type('Currency', (), {'quantizer': Decimal('0.01'), 'decimals': 2})()
    money = type('SomeMoney', (), {
        'ccy': currency,
        'qty': Decimal('100.00'),
        'dov': date(2024, 1, 1),
        '__getitem__': lambda self, i: [currency, Decimal('100.00'), date(2024, 1, 1)][i],
        '__truediv__': SomeMoney.__truediv__
    })()
    
    result = money / 0
    assert result is NoMoney


def test_truediv_with_decimal_divisor():
    from decimal import Decimal
    from datetime import date
    
    currency = type('Currency', (), {'quantizer': Decimal('0.01'), 'decimals': 2})()
    money = type('SomeMoney', (), {
        'ccy': currency,
        'qty': Decimal('100.00'),
        'dov': date(2024, 1, 1),
        '__getitem__': lambda self, i: [currency, Decimal('100.00'), date(2024, 1, 1)][i],
        '__truediv__': SomeMoney.__truediv__
    })()
    
    result = money / Decimal('4')
    assert result.qty == Decimal('25.00')


def test_truediv_with_float_divisor():
    from decimal import Decimal
    from datetime import date
    
    currency = type('Currency', (), {'quantizer': Decimal('0.01'), 'decimals': 2})()
    money = type('SomeMoney', (), {
        'ccy': currency,
        'qty': Decimal('100.00'),
        'dov': date(2024, 1, 1),
        '__getitem__': lambda self, i: [currency, Decimal('100.00'), date(2024, 1, 1)][i],
        '__truediv__': SomeMoney.__truediv__
    })()
    
    result = money / 5.0
    assert result.qty == Decimal('20.00')


# LLM-generated content at query #14
#--------------------------

```python
def test_times_with_defined_price():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    price = Price.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    result = price.times(Decimal('3'))
    
    assert result.defined
    assert result.qty == Decimal('6')
    assert result.ccy.code == "USD"


def test_times_with_undefined_price():
    price = Price.na()
    result = price.times(Decimal('5'))
    
    assert result.undefined


def test_times_with_zero_multiplier():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    price = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = price.times(Decimal('0'))
    
    assert result.defined
    assert result.qty == Decimal('0')


def test_times_with_negative_multiplier():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    price = Price.of(Currencies["USD"], Decimal('5'), Date(2019, 1, 1))
    result = price.times(Decimal('-2'))
    
    assert result.defined
    assert result.qty == Decimal('-10')


def test_times_with_decimal_multiplier():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    price = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = price.times(Decimal('0.5'))
    
    assert result.defined
    assert result.qty == Decimal('5')


# LLM-generated content at query #15
#--------------------------

```python
def test_price_mul():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    # Test multiplication with defined price
    price = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = price * Decimal('2')
    assert result.qty_or_zero() == Decimal('20')
    assert result.ccy_or_none().code == "USD"
    
    # Test multiplication with undefined price
    undefined_price = Price.na()
    result_undefined = undefined_price * Decimal('5')
    assert result_undefined.undefined
    
    # Test multiplication with zero
    price_zero = Price.of(Currencies["EUR"], Decimal('15'), Date(2019, 1, 1))
    result_zero = price_zero * Decimal('0')
    assert result_zero.qty_or_zero() == Decimal('0')
    
    # Test multiplication with negative number
    price_neg = Price.of(Currencies["GBP"], Decimal('100'), Date(2019, 1, 1))
    result_neg = price_neg * Decimal('-3')
    assert result_neg.qty_or_zero() == Decimal('-300')
    
    # Test multiplication with decimal
    price_decimal = Price.of(Currencies["JPY"], Decimal('50'), Date(2019, 1, 1))
    result_decimal = price_decimal * Decimal('1.5')
    assert result_decimal.qty_or_zero() == Decimal('75')


# LLM-generated content at query #16
#--------------------------

```python
def test_someprice_ge():
    from decimal import Decimal
    from datetime import date
    
    # Create mock Currency objects
    class MockCurrency:
        def __init__(self, code):
            self.code = code
            self.quantizer = Decimal('0.01')
        
        def __eq__(self, other):
            return isinstance(other, MockCurrency) and self.code == other.code
        
        def __ne__(self, other):
            return not self.__eq__(other)
    
    usd = MockCurrency('USD')
    eur = MockCurrency('EUR')
    
    # Test case 1: self >= other with same currency (true)
    price1 = SomePrice(usd, Decimal('100'), date(2024, 1, 1))
    price2 = SomePrice(usd, Decimal('50'), date(2024, 1, 1))
    assert price1 >= price2 is True
    
    # Test case 2: self >= other with same currency (false)
    price3 = SomePrice(usd, Decimal('30'), date(2024, 1, 1))
    price4 = SomePrice(usd, Decimal('50'), date(2024, 1, 1))
    assert price3 >= price4 is False
    
    # Test case 3: self >= other with same values (equal)
    price5 = SomePrice(usd, Decimal('75'), date(2024, 1, 1))
    price6 = SomePrice(usd, Decimal('75'), date(2024, 1, 1))
    assert price5 >= price6 is True
    
    # Test case 4: self >= other when other is not SomePrice (returns True)
    price7 = SomePrice(usd, Decimal('100'), date(2024, 1, 1))
    assert price7 >= "not a price" is True
    
    # Test case 5: self >= other with different currencies (raises exception)
    price8 = SomePrice(usd, Decimal('100'), date(2024, 1, 1))
    price9 = SomePrice(eur, Decimal('100'), date(2024, 1, 1))
    try:
        price8 >= price9
        assert False, "Should have raised IncompatibleCurrencyError"
    except Exception as e:
        assert "IncompatibleCurrencyError" in str(type(e).__name__)


# LLM-generated content at query #17
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
    
    # Test inequality with non-money object
    money13 = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    assert money13.is_equal("not a money object") is False
    
    # Test inequality with None
    money14 = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    assert money14.is_equal(None) is False


# LLM-generated content at query #18
#--------------------------

```python
def test_somemoney_neg():
    from decimal import Decimal
    from datetime import date
    
    # Create a mock Currency object
    class MockCurrency:
        def __init__(self, code, decimals=2):
            self.code = code
            self.decimals = decimals
            self.quantizer = Decimal(10) ** -decimals
        
        def __eq__(self, other):
            return self.code == other.code
    
    ccy = MockCurrency("USD")
    qty = Decimal("100.50")
    dov = date(2024, 1, 1)
    
    money = SomeMoney(ccy, qty, dov)
    result = -money
    
    assert result.ccy == ccy
    assert result.qty == Decimal("-100.50")
    assert result.dov == dov
    assert isinstance(result, SomeMoney)


def test_somemoney_neg_negative_quantity():
    from decimal import Decimal
    from datetime import date
    
    class MockCurrency:
        def __init__(self, code, decimals=2):
            self.code = code
            self.decimals = decimals
            self.quantizer = Decimal(10) ** -decimals
        
        def __eq__(self, other):
            return self.code == other.code
    
    ccy = MockCurrency("EUR")
    qty = Decimal("-50.25")
    dov = date(2024, 6, 15)
    
    money = SomeMoney(ccy, qty, dov)
    result = -money
    
    assert result.qty == Decimal("50.25")
    assert result.ccy == ccy
    assert result.dov == dov


def test_somemoney_neg_zero():
    from decimal import Decimal
    from datetime import date
    
    class MockCurrency:
        def __init__(self, code, decimals=2):
            self.code = code
            self.decimals = decimals
            self.quantizer = Decimal(10) ** -decimals
        
        def __eq__(self, other):
            return self.code == other.code
    
    ccy = MockCurrency("GBP")
    qty = Decimal("0")
    dov = date(2024, 12, 31)
    
    money = SomeMoney(ccy, qty, dov)
    result = -money
    
    assert result.qty == Decimal("0")
    assert result.ccy == ccy
    assert result.dov == dov


# LLM-generated content at query #19
#--------------------------

```python
def test_money_lte():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.money import Money
    
    # Test: defined money with same currency, less than
    money_usd_1 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    money_usd_2 = Money.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    assert money_usd_1.lte(money_usd_2) is True
    
    # Test: defined money with same currency, equal
    money_usd_2_copy = Money.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    assert money_usd_2.lte(money_usd_2_copy) is True
    
    # Test: defined money with same currency, greater than
    assert money_usd_2.lte(money_usd_1) is False
    
    # Test: undefined money is always less than or equal to defined money
    undefined_money = Money.na()
    assert undefined_money.lte(money_usd_1) is True
    
    # Test: undefined money is always less than or equal to undefined money
    undefined_money_2 = Money.na()
    assert undefined_money.lte(undefined_money_2) is True
    
    # Test: defined money is greater than undefined money, so lte is False
    assert money_usd_1.lte(undefined_money) is False


# LLM-generated content at query #20
#--------------------------

```python
def test_price_int_conversion():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    # Test __int__ on a defined price with positive quantity
    price_positive = Price.of(Currencies["USD"], Decimal('42.7'), Date(2019, 1, 1))
    assert int(price_positive) == 42
    
    # Test __int__ on a defined price with negative quantity
    price_negative = Price.of(Currencies["USD"], Decimal('-42.7'), Date(2019, 1, 1))
    assert int(price_negative) == -42
    
    # Test __int__ on a defined price with zero quantity
    price_zero = Price.of(Currencies["USD"], Decimal('0'), Date(2019, 1, 1))
    assert int(price_zero) == 0
    
    # Test __int__ on an undefined price raises MonetaryOperationException
    undefined_price = Price.na()
    try:
        int(undefined_price)
        assert False, "Expected MonetaryOperationException"
    except Exception as e:
        assert "MonetaryOperationException" in str(type(e))


# LLM-generated content at query #21
#--------------------------

```python
def test_money_add_defined_same_currency():
    from decimal import Decimal
    from datetime import date as Date
    from pypara.currencies import Currencies
    from pypara.money import Money
    
    money1 = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('5'), Date(2019, 1, 1))
    result = money1.add(money2)
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('15.00')
    assert result.ccy_or_none().code == 'USD'


def test_money_add_undefined_with_defined():
    from decimal import Decimal
    from datetime import date as Date
    from pypara.currencies import Currencies
    from pypara.money import Money
    
    undefined_money = Money.na()
    defined_money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = undefined_money.add(defined_money)
    
    assert result.defined
    assert result is defined_money


def test_money_add_defined_with_undefined():
    from decimal import Decimal
    from datetime import date as Date
    from pypara.currencies import Currencies
    from pypara.money import Money
    
    defined_money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    undefined_money = Money.na()
    result = defined_money.add(undefined_money)
    
    assert result.defined
    assert result is defined_money


def test_money_add_both_undefined():
    from pypara.money import Money
    
    undefined_money1 = Money.na()
    undefined_money2 = Money.na()
    result = undefined_money1.add(undefined_money2)
    
    assert result.undefined


def test_money_add_negative_quantities():
    from decimal import Decimal
    from datetime import date as Date
    from pypara.currencies import Currencies
    from pypara.money import Money
    
    money1 = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('-3'), Date(2019, 1, 1))
    result = money1.add(money2)
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('7.00')


def test_money_add_zero_quantities():
    from decimal import Decimal
    from datetime import date as Date
    from pypara.currencies import Currencies
    from pypara.money import Money
    
    money1 = Money.of(Currencies["USD"], Decimal('0'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('5'), Date(2019, 1, 1))
    result = money1.add(money2)
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('5.00')


def test_money_add_carries_forward_date():
    from decimal import Decimal
    from datetime import date as Date
    from pypara.currencies import Currencies
    from pypara.money import Money
    
    money1 = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('5'), Date(2019, 1, 15))
    result = money1.add(money2)
    
    assert result.defined
    assert result.dov_or_none() is not None


def test_money_add_incompatible_currency_raises_error():
    from decimal import Decimal
    from datetime import date as Date
    from pypara.currencies import Currencies
    from pypara.money import Money
    
    money1 = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["EUR"], Decimal('5'), Date(2019, 1, 1))
    
    try:
        result = money1.add(money2)
        assert False, "Expected IncompatibleCurrencyError"
    except Exception as e:
        assert "IncompatibleCurrencyError" in str(type(e).__name__) or "currency" in str(e).lower()


# LLM-generated content at query #22
#--------------------------

```python
def test_price_add():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    # Test adding two defined prices with same currency
    price1 = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('5'), Date(2019, 1, 1))
    result = price1.add(price2)
    assert result.defined
    assert result.qty_or_zero() == Decimal('15')
    assert result.ccy_or_none().code == 'USD'
    
    # Test adding defined price with undefined price
    price3 = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    price4 = Price.na()
    result2 = price3.add(price4)
    assert result2 is price3
    
    # Test adding undefined price with defined price
    price5 = Price.na()
    price6 = Price.of(Currencies["EUR"], Decimal('20'), Date(2019, 1, 2))
    result3 = price5.add(price6)
    assert result3 is price6
    
    # Test adding two undefined prices
    price7 = Price.na()
    price8 = Price.na()
    result4 = price7.add(price8)
    assert result4.undefined


# LLM-generated content at query #23
#--------------------------

```python
def test_price_abs():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    from pypara.price import Price
    
    # Test abs on defined price with positive quantity
    positive_price = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    abs_positive = positive_price.abs()
    assert abs_positive.qty_or_zero() == Decimal('10')
    assert abs_positive.ccy_or_none().code == "USD"
    
    # Test abs on defined price with negative quantity
    negative_price = Price.of(Currencies["USD"], Decimal('-10'), Date(2019, 1, 1))
    abs_negative = negative_price.abs()
    assert abs_negative.qty_or_zero() == Decimal('10')
    assert abs_negative.ccy_or_none().code == "USD"
    
    # Test abs on undefined price
    undefined_price = Price.na()
    abs_undefined = undefined_price.abs()
    assert abs_undefined.undefined is True
    assert abs_undefined is undefined_price
    
    # Test abs on zero quantity
    zero_price = Price.of(Currencies["EUR"], Decimal('0'), Date(2019, 1, 1))
    abs_zero = zero_price.abs()
    assert abs_zero.qty_or_zero() == Decimal('0')
    assert abs_zero.ccy_or_none().code == "EUR"


# LLM-generated content at query #24
#--------------------------

```python
def test_someprice_gt():
    from decimal import Decimal
    from datetime import date
    
    # Create mock Currency objects
    class MockCurrency:
        def __init__(self, code):
            self.code = code
        
        def __eq__(self, other):
            return self.code == other.code
        
        def __ne__(self, other):
            return self.code != other.code
    
    usd = MockCurrency("USD")
    eur = MockCurrency("EUR")
    
    # Test case 1: gt returns True when self.qty > other.qty with same currency
    price1 = SomePrice(usd, Decimal("100"), date(2024, 1, 1))
    price2 = SomePrice(usd, Decimal("50"), date(2024, 1, 1))
    assert price1.gt(price2) is True
    
    # Test case 2: gt returns False when self.qty < other.qty with same currency
    price3 = SomePrice(usd, Decimal("30"), date(2024, 1, 1))
    price4 = SomePrice(usd, Decimal("50"), date(2024, 1, 1))
    assert price3.gt(price4) is False
    
    # Test case 3: gt returns False when self.qty == other.qty with same currency
    price5 = SomePrice(usd, Decimal("50"), date(2024, 1, 1))
    price6 = SomePrice(usd, Decimal("50"), date(2024, 1, 1))
    assert price5.gt(price6) is False
    
    # Test case 4: gt returns True when comparing with non-SomePrice object
    price7 = SomePrice(usd, Decimal("100"), date(2024, 1, 1))
    assert price7.gt("not a price") is True
    
    # Test case 5: gt raises IncompatibleCurrencyError when currencies differ
    price8 = SomePrice(usd, Decimal("100"), date(2024, 1, 1))
    price9 = SomePrice(eur, Decimal("50"), date(2024, 1, 1))
    try:
        price8.gt(price9)
        assert False, "Should have raised IncompatibleCurrencyError"
    except Exception as e:
        assert "IncompatibleCurrencyError" in str(type(e))


# LLM-generated content at query #25
#--------------------------

```python
def test_money_abs_defined():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from decimal import Decimal
    from datetime import date as Date
    
    negative_money = Money.of(Currencies["USD"], Decimal('-10'), Date(2019, 1, 1))
    result = abs(negative_money)
    
    assert result.qty_or_zero() == Decimal('10.00')
    assert result.ccy_or_none().code == "USD"


def test_money_abs_positive():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from decimal import Decimal
    from datetime import date as Date
    
    positive_money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = abs(positive_money)
    
    assert result.qty_or_zero() == Decimal('10.00')
    assert result.ccy_or_none().code == "USD"


def test_money_abs_zero():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from decimal import Decimal
    from datetime import date as Date
    
    zero_money = Money.of(Currencies["USD"], Decimal('0'), Date(2019, 1, 1))
    result = abs(zero_money)
    
    assert result.qty_or_zero() == Decimal('0.00')


def test_money_abs_undefined():
    from pypara.money import Money
    
    undefined_money = Money.na()
    result = abs(undefined_money)
    
    assert result.undefined
    assert result is undefined_money


# LLM-generated content at query #26
#--------------------------

```python
def test_as_boolean_defined_price_with_nonzero_quantity():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.price import Price
    
    price = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert price.as_boolean() is True


def test_as_boolean_defined_price_with_zero_quantity():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.price import Price
    
    price = Price.of(Currencies["USD"], Decimal('0'), Date(2019, 1, 1))
    assert price.as_boolean() is False


def test_as_boolean_defined_price_with_negative_quantity():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.price import Price
    
    price = Price.of(Currencies["USD"], Decimal('-5'), Date(2019, 1, 1))
    assert price.as_boolean() is True


def test_as_boolean_undefined_price():
    from pypara.price import Price
    
    price = Price.na()
    assert price.as_boolean() is False


# LLM-generated content at query #27
#--------------------------

```python
def test_someprice_sub():
    from decimal import Decimal
    from datetime import date
    
    ccy = Currency(code="USD", quantizer=Decimal("0.01"))
    price1 = SomePrice(ccy=ccy, qty=Decimal("100.50"), dov=date(2024, 1, 15))
    price2 = SomePrice(ccy=ccy, qty=Decimal("30.25"), dov=date(2024, 1, 10))
    
    result = price1 - price2
    
    assert isinstance(result, SomePrice)
    assert result.ccy == ccy
    assert result.qty == Decimal("70.25")
    assert result.dov == date(2024, 1, 15)


def test_someprice_sub_with_undefined_price():
    from decimal import Decimal
    from datetime import date
    
    ccy = Currency(code="USD", quantizer=Decimal("0.01"))
    price1 = SomePrice(ccy=ccy, qty=Decimal("100.50"), dov=date(2024, 1, 15))
    
    result = price1 - NoPrice
    
    assert result is price1


def test_someprice_sub_incompatible_currency():
    from decimal import Decimal
    from datetime import date
    
    ccy_usd = Currency(code="USD", quantizer=Decimal("0.01"))
    ccy_eur = Currency(code="EUR", quantizer=Decimal("0.01"))
    price1 = SomePrice(ccy=ccy_usd, qty=Decimal("100.50"), dov=date(2024, 1, 15))
    price2 = SomePrice(ccy=ccy_eur, qty=Decimal("30.25"), dov=date(2024, 1, 10))
    
    try:
        result = price1 - price2
        assert False, "Expected IncompatibleCurrencyError"
    except IncompatibleCurrencyError as e:
        assert e.ccy1 == ccy_usd
        assert e.ccy2 == ccy_eur
        assert e.operation == "subtraction"


def test_someprice_sub_negative_result():
    from decimal import Decimal
    from datetime import date
    
    ccy = Currency(code="USD", quantizer=Decimal("0.01"))
    price1 = SomePrice(ccy=ccy, qty=Decimal("30.25"), dov=date(2024, 1, 15))
    price2 = SomePrice(ccy=ccy, qty=Decimal("100.50"), dov=date(2024, 1, 10))
    
    result = price1 - price2
    
    assert isinstance(result, SomePrice)
    assert result.qty == Decimal("-70.25")


def test_someprice_sub_uses_later_date():
    from decimal import Decimal
    from datetime import date
    
    ccy = Currency(code="USD", quantizer=Decimal("0.01"))
    price1 = SomePrice(ccy=ccy, qty=Decimal("100.50"), dov=date(2024, 1, 20))
    price2 = SomePrice(ccy=ccy, qty=Decimal("30.25"), dov=date(2024, 1, 10))
    
    result = price1 - price2
    
    assert result.dov == date(2024, 1, 20)


def test_someprice_sub_same_date():
    from decimal import Decimal
    from datetime import date
    
    ccy = Currency(code="USD", quantizer=Decimal("0.01"))
    price1 = SomePrice(ccy=ccy, qty=Decimal("100.50"), dov=date(2024, 1, 15))
    price2 = SomePrice(ccy=ccy, qty=Decimal("30.25"), dov=date(2024, 1, 15))
    
    result = price1 - price2
    
    assert result.dov == date(2024, 1, 15)


# LLM-generated content at query #28
#--------------------------

```python
def test_fmap_with_defined_price():
    from pypara.currencies import Currencies
    from pypara.price import Price
    from decimal import Decimal
    from datetime import date as Date
    import datetime
    
    someprice = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    new = someprice.fmap(lambda x: Price.of(x.ccy, x.qty + Decimal('1'), x.dov + datetime.timedelta(days=10)))
    
    assert new.ccy.code == 'USD'
    assert new.qty == Decimal('2')
    assert new.dov == Date(2019, 1, 11)


def test_fmap_with_undefined_price():
    from pypara.price import Price
    from decimal import Decimal
    
    noneprice = Price.of(None, Decimal('1'), None)
    result = noneprice.fmap(lambda sp: Price.of(sp.ccy, sp.qty + Decimal('1'), sp.dov))
    
    assert result.undefined


def test_fmap_applies_function_to_defined_price():
    from pypara.currencies import Currencies
    from pypara.price import Price
    from decimal import Decimal
    from datetime import date as Date
    
    someprice = Price.of(Currencies["USD"], Decimal('5'), Date(2020, 6, 15))
    new = someprice.fmap(lambda x: Price.of(x.ccy, x.qty * Decimal('2'), x.dov))
    
    assert new.qty == Decimal('10')
    assert new.ccy.code == 'USD'
    assert new.dov == Date(2020, 6, 15)


def test_fmap_returns_undefined_for_undefined_price():
    from pypara.price import Price
    from decimal import Decimal
    
    noneprice = Price.of(None, None, None)
    result = noneprice.fmap(lambda sp: Price.of(sp.ccy, sp.qty, sp.dov))
    
    assert result.undefined


# LLM-generated content at query #29
#--------------------------

```python
def test_as_integer():
    from decimal import Decimal
    from datetime import date as Date
    from pypara.currencies import Currencies
    from pypara.price import Price
    
    # Test with a defined price with integer quantity
    price_int = Price.of(Currencies["USD"], Decimal('42'), Date(2019, 1, 1))
    assert price_int.as_integer() == 42
    
    # Test with a defined price with decimal quantity
    price_decimal = Price.of(Currencies["USD"], Decimal('42.7'), Date(2019, 1, 1))
    assert price_decimal.as_integer() == 42
    
    # Test with a defined price with negative quantity
    price_negative = Price.of(Currencies["USD"], Decimal('-42'), Date(2019, 1, 1))
    assert price_negative.as_integer() == -42
    
    # Test with undefined price (should raise MonetaryOperationException)
    undefined_price = Price.na()
    try:
        undefined_price.as_integer()
        assert False, "Expected MonetaryOperationException"
    except Exception as e:
        assert "MonetaryOperationException" in type(e).__name__


# LLM-generated content at query #30
#--------------------------

```python
def test_price_truediv_with_defined_price():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.price import Price
    
    price = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = price / Decimal('2')
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('5')
    assert result.ccy_or_none().code == 'USD'


def test_price_truediv_with_undefined_price():
    from pypara.price import Price
    from decimal import Decimal
    
    price = Price.na()
    result = price / Decimal('2')
    
    assert result.undefined


def test_price_truediv_by_zero():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.price import Price
    
    price = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = price / Decimal('0')
    
    assert result.undefined


def test_price_truediv_preserves_currency():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.price import Price
    
    price = Price.of(Currencies["EUR"], Decimal('20'), Date(2019, 6, 15))
    result = price / Decimal('4')
    
    assert result.ccy_or_none().code == 'EUR'
    assert result.qty_or_zero() == Decimal('5')


def test_price_truediv_with_decimal_divisor():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.price import Price
    
    price = Price.of(Currencies["USD"], Decimal('7'), Date(2019, 1, 1))
    result = price / Decimal('2')
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('3.5')


# LLM-generated content at query #31
#--------------------------

```python
def test_somemoney_eq_same_values():
    from decimal import Decimal
    from datetime import date
    
    ccy = Currency("USD", 2)
    qty = Decimal("100.00")
    dov = date(2024, 1, 1)
    
    money1 = SomeMoney(ccy, qty, dov)
    money2 = SomeMoney(ccy, qty, dov)
    
    assert money1 == money2


def test_somemoney_eq_different_quantities():
    from decimal import Decimal
    from datetime import date
    
    ccy = Currency("USD", 2)
    dov = date(2024, 1, 1)
    
    money1 = SomeMoney(ccy, Decimal("100.00"), dov)
    money2 = SomeMoney(ccy, Decimal("200.00"), dov)
    
    assert not (money1 == money2)


def test_somemoney_eq_different_currencies():
    from decimal import Decimal
    from datetime import date
    
    ccy_usd = Currency("USD", 2)
    ccy_eur = Currency("EUR", 2)
    qty = Decimal("100.00")
    dov = date(2024, 1, 1)
    
    money1 = SomeMoney(ccy_usd, qty, dov)
    money2 = SomeMoney(ccy_eur, qty, dov)
    
    assert not (money1 == money2)


def test_somemoney_eq_different_dates():
    from decimal import Decimal
    from datetime import date
    
    ccy = Currency("USD", 2)
    qty = Decimal("100.00")
    
    money1 = SomeMoney(ccy, qty, date(2024, 1, 1))
    money2 = SomeMoney(ccy, qty, date(2024, 1, 2))
    
    assert not (money1 == money2)


def test_somemoney_eq_different_type():
    from decimal import Decimal
    from datetime import date
    
    ccy = Currency("USD", 2)
    qty = Decimal("100.00")
    dov = date(2024, 1, 1)
    
    money = SomeMoney(ccy, qty, dov)
    other = "not a money object"
    
    assert not (money == other)


def test_somemoney_eq_with_nomoney():
    from decimal import Decimal
    from datetime import date
    
    ccy = Currency("USD", 2)
    qty = Decimal("100.00")
    dov = date(2024, 1, 1)
    
    money = SomeMoney(ccy, qty, dov)
    
    assert not (money == NoMoney)


# LLM-generated content at query #32
#--------------------------

```python
def test_scalar_add():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    # Test scalar_add on defined money
    money_usd = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = money_usd.scalar_add(Decimal('5'))
    assert result.qty == Decimal('15.00')
    assert result.ccy.code == 'USD'
    
    # Test scalar_add with negative scalar
    result_negative = money_usd.scalar_add(Decimal('-3'))
    assert result_negative.qty == Decimal('7.00')
    
    # Test scalar_add on undefined money
    undefined_money = Money.na()
    result_undefined = undefined_money.scalar_add(Decimal('10'))
    assert result_undefined.undefined
    
    # Test scalar_add with zero
    result_zero = money_usd.scalar_add(Decimal('0'))
    assert result_zero.qty == Decimal('10.00')
    
    # Test scalar_add with integer converted to Decimal
    result_int = money_usd.scalar_add(Decimal('2'))
    assert result_int.qty == Decimal('12.00')


# LLM-generated content at query #33
#--------------------------

```python
def test_noneprice_constructor():
    none_price = NonePrice()
    assert none_price is not None
    assert isinstance(none_price, NonePrice)


# LLM-generated content at query #34
#--------------------------

```python
def test_qty_or_none():
    from decimal import Decimal
    from datetime import date
    
    # Create a mock Currency object
    class MockCurrency:
        def __init__(self, code, decimals):
            self.code = code
            self.decimals = decimals
            self.quantizer = Decimal(10) ** -decimals
        
        def __eq__(self, other):
            return self.code == other.code
        
        def __hash__(self):
            return hash(self.code)
    
    ccy = MockCurrency("USD", 2)
    qty = Decimal("100.50")
    dov = date(2023, 1, 1)
    
    some_money = SomeMoney(ccy, qty, dov)
    result = some_money.qty_or_none()
    
    assert result == qty
    assert isinstance(result, Decimal)


# LLM-generated content at query #35
#--------------------------

```python
def test_lt():
    from decimal import Decimal
    from datetime import date
    
    # Create mock Currency objects
    class MockCurrency:
        def __init__(self, code, decimals=2):
            self.code = code
            self.decimals = decimals
            self.quantizer = Decimal(10) ** -decimals
        
        def __eq__(self, other):
            return self.code == other.code
        
        def __ne__(self, other):
            return self.code != other.code
    
    usd = MockCurrency("USD")
    eur = MockCurrency("EUR")
    
    # Create SomeMoney instances
    money1 = SomeMoney(usd, Decimal("100.00"), date(2023, 1, 1))
    money2 = SomeMoney(usd, Decimal("200.00"), date(2023, 1, 1))
    money3 = SomeMoney(usd, Decimal("100.00"), date(2023, 1, 1))
    money4 = SomeMoney(eur, Decimal("100.00"), date(2023, 1, 1))
    
    # Test: money1 < money2 should be True
    assert money1.lt(money2) is True
    
    # Test: money2 < money1 should be False
    assert money2.lt(money1) is False
    
    # Test: money1 < money3 (equal quantities) should be False
    assert money1.lt(money3) is False
    
    # Test: comparing with non-SomeMoney should return False
    assert money1.lt("not a money") is False
    
    # Test: comparing different currencies should raise IncompatibleCurrencyError
    try:
        money1.lt(money4)
        assert False, "Should have raised IncompatibleCurrencyError"
    except Exception as e:
        assert "IncompatibleCurrencyError" in str(type(e).__name__)


# LLM-generated content at query #36
#--------------------------

```python
def test_qty_or_else_with_defined_money():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    somemoney = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    result = somemoney.qty_or_else(lambda: Decimal('42'))
    assert result == Decimal('1.00')


def test_qty_or_else_with_defined_money_returns_qty_not_combinator():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    somemoney = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    result = somemoney.qty_or_else(lambda: True)
    assert result == Decimal('1.00')


def test_qty_or_else_with_undefined_money_returns_combinator_result():
    from datetime import date as Date
    from decimal import Decimal
    
    nonemoney = Money.of(None, Decimal('1'), None)
    result = nonemoney.qty_or_else(lambda: Decimal('42'))
    assert result == Decimal('42')


def test_qty_or_else_with_undefined_money_returns_combinator_result_non_decimal():
    from datetime import date as Date
    from decimal import Decimal
    
    nonemoney = Money.of(None, Decimal('1'), None)
    result = nonemoney.qty_or_else(lambda: False)
    assert result is False


def test_qty_or_else_with_undefined_money_calls_combinator():
    from datetime import date as Date
    from decimal import Decimal
    
    nonemoney = Money.of(None, Decimal('1'), None)
    result = nonemoney.qty_or_else(lambda: Decimal('99'))
    assert result == Decimal('99')


# LLM-generated content at query #37
#--------------------------

```python
def test_price_dimap():
    from datetime import date as Date
    from decimal import Decimal
    from pypara.currencies import Currencies
    from pypara.price import Price
    
    # Test dimap with defined price - applies function f
    defined_price = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    result_defined = defined_price.dimap(lambda x: x.ccy.code, lambda: "EUR")
    assert result_defined == "USD"
    
    # Test dimap with undefined price - applies function e
    undefined_price = Price.of(None, Decimal('1'), None)
    result_undefined = undefined_price.dimap(lambda x: x.ccy.code, lambda: "EUR")
    assert result_undefined == "EUR"
    
    # Test dimap with defined price and different mapping function
    defined_price_2 = Price.of(Currencies["EUR"], Decimal('42'), Date(2020, 6, 15))
    result_qty = defined_price_2.dimap(lambda x: x.qty, lambda: Decimal('0'))
    assert result_qty == Decimal('42')
    
    # Test dimap with undefined price and different mapping function
    undefined_price_2 = Price.of(None, None, None)
    result_fallback = undefined_price_2.dimap(lambda x: x.qty * 2, lambda: Decimal('99'))
    assert result_fallback == Decimal('99')
    
    # Test dimap with defined price mapping to boolean
    defined_price_3 = Price.of(Currencies["GBP"], Decimal('100'), Date(2021, 3, 10))
    result_bool = defined_price_3.dimap(lambda x: x.qty > Decimal('50'), lambda: False)
    assert result_bool is True
    
    # Test dimap with undefined price mapping to boolean
    undefined_price_3 = Price.of(None, Decimal('50'), None)
    result_bool_fallback = undefined_price_3.dimap(lambda x: x.qty > Decimal('50'), lambda: False)
    assert result_bool_fallback is False


# LLM-generated content at query #38
#--------------------------

```python
def test_money_mul_with_positive_scalar():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.money import Money
    
    money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = money * Decimal('2')
    
    assert result.qty == Decimal('20.00')
    assert result.ccy.code == 'USD'
    assert result.dov == Date(2019, 1, 1)


def test_money_mul_with_negative_scalar():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.money import Money
    
    money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = money * Decimal('-2')
    
    assert result.qty == Decimal('-20.00')
    assert result.ccy.code == 'USD'


def test_money_mul_with_zero():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.money import Money
    
    money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = money * Decimal('0')
    
    assert result.qty == Decimal('0.00')


def test_money_mul_with_fraction():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.money import Money
    
    money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = money * Decimal('0.5')
    
    assert result.qty == Decimal('5.00')


def test_money_mul_undefined_money():
    from pypara.money import Money
    from decimal import Decimal
    
    money = Money.na()
    result = money * Decimal('2')
    
    assert result.undefined


def test_money_rmul_with_positive_scalar():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.money import Money
    
    money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = Decimal('3') * money
    
    assert result.qty == Decimal('30.00')
    assert result.ccy.code == 'USD'


# LLM-generated content at query #39
#--------------------------

```python
def test_somemoney_sub():
    from decimal import Decimal
    from datetime import date
    
    # Create test currencies and money objects
    usd = Currency(code="USD", decimals=2)
    eur = Currency(code="EUR", decimals=2)
    
    test_date = date(2024, 1, 1)
    
    money1 = SomeMoney(usd, Decimal("100.00"), test_date)
    money2 = SomeMoney(usd, Decimal("30.00"), test_date)
    money_eur = SomeMoney(eur, Decimal("50.00"), test_date)
    
    # Test normal subtraction
    result = money1 - money2
    assert result.ccy == usd
    assert result.qty == Decimal("70.00")
    assert result.dov == test_date
    
    # Test subtraction with different dates (should use later date)
    later_date = date(2024, 1, 15)
    money3 = SomeMoney(usd, Decimal("20.00"), later_date)
    result2 = money1 - money3
    assert result2.dov == later_date
    
    # Test subtraction with incompatible currency
    try:
        money1 - money_eur
        assert False, "Should raise IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass
    
    # Test subtraction with undefined money (NoMoney)
    result3 = money1 - NoMoney
    assert result3 == money1
    
    # Test subtraction resulting in negative quantity
    money4 = SomeMoney(usd, Decimal("10.00"), test_date)
    money5 = SomeMoney(usd, Decimal("50.00"), test_date)
    result4 = money4 - money5
    assert result4.qty == Decimal("-40.00")


# LLM-generated content at query #40
#--------------------------

```python
def test_gte_raises_incompatible_currency_error():
    from decimal import Decimal
    from datetime import date
    
    # Create two SomePrice instances with different currencies
    ccy1 = Currency(code="USD", quantizer=Decimal("0.01"))
    ccy2 = Currency(code="EUR", quantizer=Decimal("0.01"))
    
    price1 = SomePrice(ccy1, Decimal("100.00"), date(2024, 1, 1))
    price2 = SomePrice(ccy2, Decimal("50.00"), date(2024, 1, 1))
    
    # Call gte which should raise IncompatibleCurrencyError at line 4
    try:
        price1.gte(price2)
        assert False, "Expected IncompatibleCurrencyError to be raised"
    except IncompatibleCurrencyError as e:
        assert e.args[0] == ccy1
        assert e.args[1] == ccy2
        assert ">= comparison" in str(e)


# LLM-generated content at query #41
#--------------------------

```python
def test_subtract_raises_error_when_currencies_differ():
    from datetime import date
    from decimal import Decimal
    
    # Create mock Currency objects
    class MockCurrency:
        def __init__(self, code):
            self.code = code
            self.decimals = 2
            self.quantizer = Decimal('0.01')
        
        def __eq__(self, other):
            return self.code == other.code
        
        def __ne__(self, other):
            return not self.__eq__(other)
        
        def __hash__(self):
            return hash(self.code)
    
    usd = MockCurrency('USD')
    eur = MockCurrency('EUR')
    
    # Create SomeMoney instances with different currencies
    money1 = SomeMoney(usd, Decimal('100.00'), date(2024, 1, 1))
    money2 = SomeMoney(eur, Decimal('50.00'), date(2024, 1, 1))
    
    # Verify that the predicate c1 != c2 evaluates to True
    c1, q1, d1 = money1
    c2, q2, d2 = money2
    
    assert c1 != c2


# LLM-generated content at query #42
#--------------------------

```python
def test_with_qty():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    # Test with_qty on defined money
    defined_money = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    new_money = defined_money.with_qty(Decimal('200'))
    
    assert new_money.qty_or_none() == Decimal('200')
    assert new_money.ccy_or_none().code == 'USD'
    assert new_money.dov_or_none() == Date(2019, 1, 1)
    assert new_money.defined is True
    
    # Test with_qty on undefined money
    undefined_money = Money.na()
    result = undefined_money.with_qty(Decimal('300'))
    
    assert result.undefined is True
    assert result is undefined_money


# LLM-generated content at query #43
#--------------------------

```python
def test_somemoney_le():
    from decimal import Decimal
    from datetime import date
    
    # Create mock Currency objects
    usd = type('Currency', (), {'__eq__': lambda self, other: self is other or (hasattr(other, '__class__') and other.__class__.__name__ == 'Currency' and True)})()
    usd.__class__.__name__ = 'Currency'
    eur = type('Currency', (), {'__eq__': lambda self, other: False})()
    eur.__class__.__name__ = 'Currency'
    
    # Create test money objects
    money1 = SomeMoney(usd, Decimal('100.00'), date(2024, 1, 1))
    money2 = SomeMoney(usd, Decimal('200.00'), date(2024, 1, 1))
    money3 = SomeMoney(usd, Decimal('100.00'), date(2024, 1, 1))
    
    # Test: money1 <= money2 (True, 100 <= 200)
    assert money1.__le__(money2) == True
    
    # Test: money2 <= money1 (False, 200 > 100)
    assert money2.__le__(money1) == False
    
    # Test: money1 <= money3 (True, 100 <= 100)
    assert money1.__le__(money3) == True
    
    # Test: comparison with non-SomeMoney object returns False
    assert money1.__le__("not money") == False
    assert money1.__le__(100) == False
    assert money1.__le__(None) == False


# LLM-generated content at query #44
#--------------------------

```python
def test_with_dov():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.price import Price
    
    # Test with_dov on a defined price object
    original_price = Price.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    new_date = Date(2020, 6, 15)
    updated_price = original_price.with_dov(new_date)
    
    assert updated_price.dov_or_none() == new_date
    assert updated_price.ccy_or_none().code == "USD"
    assert updated_price.qty_or_none() == Decimal('100')
    assert updated_price.defined is True
    
    # Test with_dov on an undefined price object
    undefined_price = Price.na()
    result = undefined_price.with_dov(Date(2020, 1, 1))
    
    assert result is undefined_price
    assert result.undefined is True


# LLM-generated content at query #45
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
        
        def __eq__(self, other):
            return isinstance(other, MockCurrency) and self.code == other.code
        
        def __ne__(self, other):
            return not self.__eq__(other)
    
    ccy = MockCurrency("USD", 2)
    qty = Decimal("100.50")
    dov = date(2023, 1, 15)
    
    # Test basic constructor
    money = SomeMoney(ccy, qty, dov)
    
    assert money.ccy == ccy
    assert money.qty == qty
    assert money.dov == dov
    assert money[0] == ccy
    assert money[1] == qty
    assert money[2] == dov
    
    # Test with different values
    ccy2 = MockCurrency("EUR", 2)
    qty2 = Decimal("50.25")
    dov2 = date(2023, 6, 30)
    
    money2 = SomeMoney(ccy2, qty2, dov2)
    
    assert money2.ccy == ccy2
    assert money2.qty == qty2
    assert money2.dov == dov2
    
    # Test tuple unpacking
    c, q, d = money
    assert c == ccy
    assert q == qty
    assert d == dov


# LLM-generated content at query #46
#--------------------------

```python
def test_somemoney_gt():
    from datetime import date
    from decimal import Decimal
    
    # Create test currencies and money objects
    usd = Currency(code="USD", decimals=2)
    eur = Currency(code="EUR", decimals=2)
    
    money1 = SomeMoney(usd, Decimal("100.00"), date(2024, 1, 1))
    money2 = SomeMoney(usd, Decimal("50.00"), date(2024, 1, 1))
    money3 = SomeMoney(usd, Decimal("100.00"), date(2024, 1, 1))
    money_eur = SomeMoney(eur, Decimal("100.00"), date(2024, 1, 1))
    
    # Test greater than with same currency
    assert money1 > money2 is True
    assert money2 > money1 is False
    assert money1 > money3 is False
    
    # Test greater than with non-SomeMoney object
    assert money1 > "not a money" is True
    assert money1 > 100 is True
    assert money1 > None is True
    
    # Test greater than with different currencies raises exception
    try:
        money1 > money_eur
        assert False, "Should have raised IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass


# LLM-generated content at query #47
#--------------------------

```python
def test_money_int_conversion():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    # Test __int__ on defined money with positive quantity
    money_positive = Money.of(Currencies["USD"], Decimal('42.75'), Date(2019, 1, 1))
    assert int(money_positive) == 42
    
    # Test __int__ on defined money with negative quantity
    money_negative = Money.of(Currencies["USD"], Decimal('-42.75'), Date(2019, 1, 1))
    assert int(money_negative) == -42
    
    # Test __int__ on defined money with zero quantity
    money_zero = Money.of(Currencies["USD"], Decimal('0'), Date(2019, 1, 1))
    assert int(money_zero) == 0
    
    # Test __int__ on defined money with large quantity
    money_large = Money.of(Currencies["USD"], Decimal('999999.99'), Date(2019, 1, 1))
    assert int(money_large) == 999999
    
    # Test __int__ on undefined money raises exception
    undefined_money = Money.na()
    try:
        int(undefined_money)
        assert False, "Expected MonetaryOperationException"
    except Exception as e:
        assert "MonetaryOperationException" in str(type(e))


# LLM-generated content at query #48
#--------------------------

```python
def test_with_dov():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    
    # Test with_dov on defined money
    somemoney = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    new_dov = Date(2020, 6, 15)
    result = somemoney.with_dov(new_dov)
    assert result.dov_or_none() == new_dov
    assert result.ccy_or_none().code == "USD"
    assert result.qty_or_none() == Decimal('1.00')
    
    # Test with_dov on undefined money
    nonemoney = Money.na()
    result_undefined = nonemoney.with_dov(new_dov)
    assert result_undefined.undefined
    assert result_undefined is nonemoney


# LLM-generated content at query #49
#--------------------------

```python
def test_subtract():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    
    # Test subtracting two defined prices with same currency
    price1 = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('3'), Date(2019, 1, 1))
    result = price1.subtract(price2)
    assert result.defined
    assert result.qty_or_zero() == Decimal('7')
    
    # Test subtracting undefined price from defined price returns the defined price
    defined_price = Price.of(Currencies["USD"], Decimal('5'), Date(2019, 1, 1))
    undefined_price = Price.na()
    result = defined_price.subtract(undefined_price)
    assert result.defined
    assert result.qty_or_zero() == Decimal('5')
    
    # Test subtracting defined price from undefined price returns the defined price
    result = undefined_price.subtract(defined_price)
    assert result.defined
    assert result.qty_or_zero() == Decimal('5')
    
    # Test subtracting two undefined prices returns undefined
    result = undefined_price.subtract(undefined_price)
    assert result.undefined
    
    # Test subtracting negative quantity
    price3 = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    price4 = Price.of(Currencies["USD"], Decimal('-5'), Date(2019, 1, 1))
    result = price3.subtract(price4)
    assert result.defined
    assert result.qty_or_zero() == Decimal('15')


# LLM-generated content at query #50
#--------------------------

```python
def test_with_dov():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    
    # Test with_dov on defined money
    somemoney = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    new_dov = Date(2020, 6, 15)
    result = somemoney.with_dov(new_dov)
    assert result.dov_or_none() == new_dov
    assert result.ccy_or_none().code == "USD"
    assert result.qty_or_none() == Decimal('1.00')
    
    # Test with_dov on undefined money
    nonemoney = Money.of(None, Decimal('1'), None)
    result_undefined = nonemoney.with_dov(new_dov)
    assert result_undefined is nonemoney
    assert result_undefined.undefined


# LLM-generated content at query #51
#--------------------------

```python
def test_convert_with_valid_rate():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currency, CurrencyType
    from pypara.monetary import SomeMoney
    from pypara.fx import FXRate, FXRateService
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    
    money = SomeMoney(usd, Decimal("100.00"), date(2023, 1, 1))
    
    class MockFXRateService:
        default = None
        def query(self, from_ccy, to_ccy, asof, strict):
            if from_ccy == usd and to_ccy == eur:
                return FXRate(usd, eur, Decimal("0.85"), date(2023, 1, 1))
            return None
    
    original_service = FXRateService.default
    try:
        FXRateService.default = MockFXRateService()
        result = money.convert(eur, asof=date(2023, 1, 1), strict=False)
        assert result.ccy == eur
        assert result.qty == Decimal("85.00")
        assert result.dov == date(2023, 1, 1)
    finally:
        FXRateService.default = original_service


def test_convert_uses_dov_when_asof_not_provided():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currency, CurrencyType
    from pypara.monetary import SomeMoney
    from pypara.fx import FXRate, FXRateService
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    gbp = Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY)
    
    dov = date(2023, 6, 15)
    money = SomeMoney(usd, Decimal("100.00"), dov)
    
    class MockFXRateService:
        default = None
        def query(self, from_ccy, to_ccy, asof, strict):
            if asof == dov and from_ccy == usd and to_ccy == gbp:
                return FXRate(usd, gbp, Decimal("0.79"), dov)
            return None
    
    original_service = FXRateService.default
    try:
        FXRateService.default = MockFXRateService()
        result = money.convert(gbp, strict=False)
        assert result.ccy == gbp
        assert result.qty == Decimal("79.00")
        assert result.dov == dov
    finally:
        FXRateService.default = original_service


def test_convert_returns_nomoney_when_rate_not_found_non_strict():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currency, CurrencyType
    from pypara.monetary import SomeMoney, NoMoney
    from pypara.fx import FXRateService
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    
    money = SomeMoney(usd, Decimal("100.00"), date(2023, 1, 1))
    
    class MockFXRateService:
        default = None
        def query(self, from_ccy, to_ccy, asof, strict):
            return None
    
    original_service = FXRateService.default
    try:
        FXRateService.default = MockFXRateService()
        result = money.convert(jpy, asof=date(2023, 1, 1), strict=False)
        assert result == NoMoney
    finally:
        FXRateService.default = original_service


def test_convert_raises_error_when_rate_not_found_strict():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currency, CurrencyType
    from pypara.monetary import SomeMoney
    from pypara.fx import FXRateService, FXRateLookupError
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    chf = Currency.of("CHF", "Swiss Franc", 2, CurrencyType.MONEY)
    
    money = SomeMoney(usd, Decimal("100.00"), date(2023, 1, 1))
    
    class MockFXRateService:
        default = None
        def query(self, from_ccy, to_ccy, asof, strict):
            return None
    
    original_service = FXRateService.default
    try:
        FXRateService.default = MockFXRateService()
        error_raised = False
        try:
            money.convert(chf, asof=date(2023, 1, 1), strict=True)
        except FXRateLookupError:
            error_raised = True
        assert error_raised
    finally:
        FXRateService.default = original_service


def test_convert_quantizes_result_to_target_currency():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currency, CurrencyType
    from pypara.monetary import SomeMoney
    from pypara.fx import FXRate, FXRateService
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    
    money = SomeMoney(usd, Decimal("100.00"), date(2023, 1, 1))
    
    class MockFXRateService:
        default = None
        def query(self, from_ccy, to_ccy, asof, strict):
            if from_ccy == usd and to_ccy == jpy:
                return FXRate(usd, jpy, Decimal("110.567"), date(2023, 1, 1))
            return None
    
    original_service = FXRateService.default
    try:
        FXRateService.default = MockFXRateService()
        result = money.convert(jpy, asof=date(2023, 1, 1), strict=False)
        assert result.ccy == jpy
        assert result.qty == Decimal("11057")
    finally:
        FXRateService.default = original_service


def test_convert_raises_error_when_fx_rate_service_not_set():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currency, CurrencyType
    from pypara.monetary import SomeMoney
    from pypara.fx import FXRateService
    from pypara.errors import ProgrammingError
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    
    money = SomeMoney(usd, Decimal("100.00"), date(2023, 1, 1))
    
    original_service = FXRateService.default
    try:
        FXRateService.default = None
        error_raised = False
        try:
            money.convert(eur, asof=date(2023, 1, 1), strict=False)
        except ProgrammingError:
            error_raised = True
        assert error_raised
    finally:
        FXRateService.default = original_service


# LLM-generated content at query #52
#--------------------------

```python
def test_price_lte():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    
    # Create test prices
    price_usd_1 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    price_usd_2 = Price.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    price_usd_1_copy = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    undefined_price = Price.na()
    
    # Test: defined price <= smaller defined price should be False
    assert not price_usd_2.lte(price_usd_1)
    
    # Test: defined price <= equal defined price should be True
    assert price_usd_1.lte(price_usd_1_copy)
    
    # Test: defined price <= greater defined price should be True
    assert price_usd_1.lte(price_usd_2)
    
    # Test: undefined price <= undefined price should be True
    assert undefined_price.lte(undefined_price)
    
    # Test: undefined price <= defined price should be True
    assert undefined_price.lte(price_usd_1)
    
    # Test: defined price <= undefined price should be False
    assert not price_usd_1.lte(undefined_price)


# LLM-generated content at query #53
#--------------------------

```python
def test_price_neg():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    # Test negation of a defined price
    price = Price.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    negated_price = -price
    
    assert negated_price.qty == Decimal('-100')
    assert negated_price.ccy.code == 'USD'
    assert negated_price.dov == Date(2019, 1, 1)
    
    # Test double negation returns to original
    double_negated = -negated_price
    assert double_negated.qty == Decimal('100')
    
    # Test negation of undefined price returns itself
    undefined_price = Price.na()
    negated_undefined = -undefined_price
    assert negated_undefined.undefined
    
    # Test negation of zero quantity
    zero_price = Price.of(Currencies["EUR"], Decimal('0'), Date(2020, 6, 15))
    negated_zero = -zero_price
    assert negated_zero.qty == Decimal('0')
    
    # Test negation of negative quantity
    negative_price = Price.of(Currencies["GBP"], Decimal('-50'), Date(2021, 3, 10))
    negated_negative = -negative_price
    assert negated_negative.qty == Decimal('50')


# LLM-generated content at query #54
#--------------------------

```python
def test_multiply_defined_money_with_positive_scalar():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = money.multiply(Decimal('2'))
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('20.00')
    assert result.ccy_or_none().code == "USD"


def test_multiply_defined_money_with_negative_scalar():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = money.multiply(Decimal('-3'))
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('-30.00')


def test_multiply_defined_money_with_zero():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = money.multiply(Decimal('0'))
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('0.00')


def test_multiply_defined_money_with_decimal_scalar():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = money.multiply(Decimal('1.5'))
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('15.00')


def test_multiply_undefined_money():
    from decimal import Decimal
    
    money = Money.na()
    result = money.multiply(Decimal('5'))
    
    assert result.undefined
    assert result is money


def test_multiply_preserves_currency():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money = Money.of(Currencies["EUR"], Decimal('100'), Date(2020, 6, 15))
    result = money.multiply(Decimal('0.5'))
    
    assert result.ccy_or_none().code == "EUR"


def test_multiply_preserves_date():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    original_date = Date(2021, 3, 20)
    money = Money.of(Currencies["GBP"], Decimal('50'), original_date)
    result = money.multiply(Decimal('2'))
    
    assert result.dov_or_none() == original_date


# LLM-generated content at query #55
#--------------------------

```python
def test_round_defined_money():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.money import Money
    
    money = Money.of(Currencies["USD"], Decimal('1.456'), Date(2019, 1, 1))
    rounded = money.round(2)
    
    assert rounded.qty == Decimal('1.46')
    assert rounded.ccy == Currencies["USD"]
    assert rounded.dov == Date(2019, 1, 1)


def test_round_undefined_money():
    from pypara.money import Money
    
    money = Money.na()
    rounded = money.round(2)
    
    assert rounded.undefined


def test_round_with_default_ndigits():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.money import Money
    
    money = Money.of(Currencies["USD"], Decimal('1.456'), Date(2019, 1, 1))
    rounded = money.round()
    
    assert rounded.qty == Decimal('1')


def test_round_zero_ndigits():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.money import Money
    
    money = Money.of(Currencies["USD"], Decimal('1.567'), Date(2019, 1, 1))
    rounded = money.round(0)
    
    assert rounded.qty == Decimal('2')


def test_round_negative_ndigits():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.money import Money
    
    money = Money.of(Currencies["USD"], Decimal('1234.567'), Date(2019, 1, 1))
    rounded = money.round(-1)
    
    assert rounded.qty == Decimal('1230')


def test_round_preserves_currency():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.money import Money
    
    money = Money.of(Currencies["EUR"], Decimal('99.999'), Date(2020, 6, 15))
    rounded = money.round(1)
    
    assert rounded.ccy == Currencies["EUR"]


def test_round_preserves_date():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.money import Money
    
    original_date = Date(2021, 12, 25)
    money = Money.of(Currencies["GBP"], Decimal('55.555'), original_date)
    rounded = money.round(2)
    
    assert rounded.dov == original_date


# LLM-generated content at query #56
#--------------------------

```python
def test_money_le_with_defined_money_objects():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.money import Money
    
    money1 = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('20'), Date(2019, 1, 1))
    money3 = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    
    assert money1 <= money2
    assert money1 <= money3
    assert not (money2 <= money1)


def test_money_le_with_undefined_money_objects():
    from pypara.money import Money
    
    undefined_money = Money.na()
    
    assert undefined_money <= undefined_money


def test_money_le_with_undefined_and_defined():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.money import Money
    
    undefined_money = Money.na()
    defined_money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    
    assert undefined_money <= defined_money
    assert not (defined_money <= undefined_money)


def test_money_le_with_different_currencies_raises_error():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.money import Money
    from pypara.exceptions import IncompatibleCurrencyError
    
    money_usd = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    money_eur = Money.of(Currencies["EUR"], Decimal('10'), Date(2019, 1, 1))
    
    try:
        money_usd <= money_eur
        assert False, "Expected IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass


# LLM-generated content at query #57
#--------------------------

```python
def test_money_eq():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money1 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    money3 = Money.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    money4 = Money.of(Currencies["EUR"], Decimal('1'), Date(2019, 1, 1))
    money5 = Money.na()
    money6 = Money.na()
    
    assert money1 == money2
    assert not (money1 == money3)
    assert not (money1 == money4)
    assert money5 == money6
    assert not (money1 == money5)
    assert not (money1 == None)
    assert not (money1 == "not a money")
    assert not (money1 == 42)


# LLM-generated content at query #58
#--------------------------

```python
def test_subtract():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    # Test subtract with two defined money objects with same currency
    money1 = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('3'), Date(2019, 1, 1))
    result = money1.subtract(money2)
    assert result.defined
    assert result.qty == Decimal('7.00')
    assert result.ccy.code == "USD"
    
    # Test subtract with negative result
    money3 = Money.of(Currencies["USD"], Decimal('5'), Date(2019, 1, 1))
    money4 = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result2 = money3.subtract(money4)
    assert result2.defined
    assert result2.qty == Decimal('-5.00')
    
    # Test subtract with undefined money on left side
    undefined_money = Money.na()
    money5 = Money.of(Currencies["USD"], Decimal('5'), Date(2019, 1, 1))
    result3 = undefined_money.subtract(money5)
    assert result3 is money5
    
    # Test subtract with undefined money on right side
    money6 = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result4 = money6.subtract(undefined_money)
    assert result4 is money6
    
    # Test subtract with both undefined
    result5 = undefined_money.subtract(undefined_money)
    assert result5.undefined
    
    # Test subtract with zero
    money7 = Money.of(Currencies["USD"], Decimal('5'), Date(2019, 1, 1))
    money8 = Money.of(Currencies["USD"], Decimal('0'), Date(2019, 1, 1))
    result6 = money7.subtract(money8)
    assert result6.qty == Decimal('5.00')


# LLM-generated content at query #59
#--------------------------

```python
def test_gt_raises_incompatible_currency_error():
    from decimal import Decimal
    from datetime import date
    
    # Create mock Currency objects with different values
    class MockCurrency:
        def __init__(self, code):
            self.code = code
            self.decimals = 2
            self.quantizer = Decimal('0.01')
        
        def __eq__(self, other):
            return self.code == other.code
        
        def __ne__(self, other):
            return self.code != other.code
    
    class IncompatibleCurrencyError(Exception):
        def __init__(self, ccy1, ccy2, operation):
            self.ccy1 = ccy1
            self.ccy2 = ccy2
            self.operation = operation
    
    # Monkey patch the exception into the module
    import sys
    if 'IncompatibleCurrencyError' not in dir():
        globals()['IncompatibleCurrencyError'] = IncompatibleCurrencyError
    
    ccy1 = MockCurrency('USD')
    ccy2 = MockCurrency('EUR')
    
    money1 = SomeMoney(ccy1, Decimal('100.00'), date(2023, 1, 1))
    money2 = SomeMoney(ccy2, Decimal('50.00'), date(2023, 1, 1))
    
    try:
        money1.gt(money2)
        assert False, "Expected IncompatibleCurrencyError to be raised"
    except IncompatibleCurrencyError as e:
        assert e.ccy1 == ccy1
        assert e.ccy2 == ccy2
        assert e.operation == "> comparison"


# LLM-generated content at query #60
#--------------------------

```python
def test_price_gte():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    # Create test prices
    price_usd_100 = Price.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    price_usd_50 = Price.of(Currencies["USD"], Decimal('50'), Date(2019, 1, 1))
    price_usd_100_other = Price.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 2))
    
    # Test: defined price >= smaller defined price
    assert price_usd_100.gte(price_usd_50) is True
    
    # Test: defined price >= equal defined price
    assert price_usd_100.gte(price_usd_100_other) is True
    
    # Test: smaller defined price >= larger defined price
    assert price_usd_50.gte(price_usd_100) is False
    
    # Test: defined price >= undefined price
    undefined_price = Price.na()
    assert price_usd_100.gte(undefined_price) is True
    
    # Test: undefined price >= undefined price
    assert undefined_price.gte(undefined_price) is True
    
    # Test: undefined price >= defined price
    assert undefined_price.gte(price_usd_100) is False


# LLM-generated content at query #61
#--------------------------

```python
def test_convert_with_valid_rate():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currency, CurrencyType
    from pypara.monetary import SomePrice
    from pypara.services import FXRateService
    
    usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    
    price = SomePrice(usd, Decimal("100.00"), date(2023, 1, 1))
    
    class MockFXRateService:
        default = None
    
    original_service = FXRateService.default
    try:
        mock_rate = type('MockRate', (), {'value': Decimal("0.85")})()
        
        class TestFXRateService:
            @staticmethod
            def query(ccy1, ccy2, asof, strict):
                return mock_rate
        
        FXRateService.default = TestFXRateService()
        result = price.convert(eur, date(2023, 1, 1), False)
        
        assert result.ccy == eur
        assert result.qty == Decimal("85.00")
        assert result.dov == date(2023, 1, 1)
    finally:
        FXRateService.default = original_service


def test_convert_uses_dov_when_asof_not_provided():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currency, CurrencyType
    from pypara.monetary import SomePrice
    from pypara.services import FXRateService
    
    usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    
    dov = date(2023, 6, 15)
    price = SomePrice(usd, Decimal("100.00"), dov)
    
    original_service = FXRateService.default
    try:
        mock_rate = type('MockRate', (), {'value': Decimal("0.85")})()
        queried_dates = []
        
        def mock_query(ccy1, ccy2, asof, strict):
            queried_dates.append(asof)
            return mock_rate
        
        class TestFXRateService:
            query = staticmethod(mock_query)
        
        FXRateService.default = TestFXRateService()
        result = price.convert(eur, None, False)
        
        assert queried_dates[0] == dov
    finally:
        FXRateService.default = original_service


def test_convert_no_rate_non_strict():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currency, CurrencyType
    from pypara.monetary import SomePrice, NoPrice
    from pypara.services import FXRateService
    
    usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    
    price = SomePrice(usd, Decimal("100.00"), date(2023, 1, 1))
    
    original_service = FXRateService.default
    try:
        class TestFXRateService:
            @staticmethod
            def query(ccy1, ccy2, asof, strict):
                return None
        
        FXRateService.default = TestFXRateService()
        result = price.convert(eur, date(2023, 1, 1), False)
        
        assert result is NoPrice
    finally:
        FXRateService.default = original_service


def test_convert_no_rate_strict_raises_error():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currency, CurrencyType
    from pypara.monetary import SomePrice
    from pypara.services import FXRateService
    from pypara.errors import FXRateLookupError
    
    usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    
    price = SomePrice(usd, Decimal("100.00"), date(2023, 1, 1))
    
    original_service = FXRateService.default
    try:
        class TestFXRateService:
            @staticmethod
            def query(ccy1, ccy2, asof, strict):
                return None
        
        FXRateService.default = TestFXRateService()
        error_raised = False
        try:
            result = price.convert(eur, date(2023, 1, 1), True)
        except FXRateLookupError:
            error_raised = True
        
        assert error_raised
    finally:
        FXRateService.default = original_service


def test_convert_no_default_service_raises_programming_error():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currency, CurrencyType
    from pypara.monetary import SomePrice
    from pypara.services import FXRateService
    from pypara.errors import ProgrammingError
    
    usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    
    price = SomePrice(usd, Decimal("100.00"), date(2023, 1, 1))
    
    original_service = FXRateService.default
    try:
        FXRateService.default = None
        error_raised = False
        try:
            result = price.convert(eur, date(2023, 1, 1), False)
        except ProgrammingError:
            error_raised = True
        
        assert error_raised
    finally:
        FXRateService.default = original_service


# LLM-generated content at query #62
#--------------------------

```python
def test_money_gt():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    # Test: defined money > undefined money returns True
    defined_money = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    undefined_money = Money.na()
    assert defined_money.gt(undefined_money) is True
    
    # Test: undefined money > defined money returns False
    assert undefined_money.gt(defined_money) is False
    
    # Test: undefined money > undefined money returns False
    assert undefined_money.gt(undefined_money) is False
    
    # Test: defined money with larger quantity > defined money with smaller quantity returns True
    money_larger = Money.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    money_smaller = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert money_larger.gt(money_smaller) is True
    
    # Test: defined money with smaller quantity > defined money with larger quantity returns False
    assert money_smaller.gt(money_larger) is False
    
    # Test: defined money with equal quantity > defined money with equal quantity returns False
    money_equal1 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    money_equal2 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 2))
    assert money_equal1.gt(money_equal2) is False
    
    # Test: IncompatibleCurrencyError is raised when comparing two defined money objects with different currencies
    money_usd = Money.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    money_eur = Money.of(Currencies["EUR"], Decimal('1'), Date(2019, 1, 1))
    try:
        money_usd.gt(money_eur)
        assert False, "Expected IncompatibleCurrencyError"
    except Exception as e:
        assert "IncompatibleCurrencyError" in str(type(e))


# LLM-generated content at query #63
#--------------------------

```python
def test_money_truediv():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    # Test division of defined money by a positive number
    money = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    result = money / Decimal('2')
    assert result.qty == Decimal('50.00')
    assert result.ccy == Currencies["USD"]
    
    # Test division of defined money by a negative number
    money = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    result = money / Decimal('-2')
    assert result.qty == Decimal('-50.00')
    
    # Test division by zero returns undefined money
    money = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    result = money / Decimal('0')
    assert result.undefined
    
    # Test division of undefined money returns itself
    money = Money.na()
    result = money / Decimal('2')
    assert result.undefined
    
    # Test division with float
    money = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    result = money / 2.5
    assert result.qty == Decimal('40.00')
    
    # Test division with integer
    money = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    result = money / 4
    assert result.qty == Decimal('25.00')


# LLM-generated content at query #64
#--------------------------

```python
def test_as_integer():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    # Test with a defined price
    defined_price = Price.of(Currencies["USD"], Decimal('42'), Date(2019, 1, 1))
    result = defined_price.as_integer()
    assert result == 42
    assert isinstance(result, int)
    
    # Test with a decimal quantity that needs conversion
    defined_price_decimal = Price.of(Currencies["EUR"], Decimal('99.99'), Date(2019, 1, 1))
    result_decimal = defined_price_decimal.as_integer()
    assert result_decimal == 99
    
    # Test with an undefined price - should raise MonetaryOperationException
    undefined_price = Price.na()
    try:
        undefined_price.as_integer()
        assert False, "Expected MonetaryOperationException to be raised"
    except Exception as e:
        assert "MonetaryOperationException" in str(type(e))


# LLM-generated content at query #65
#--------------------------

```python
def test_qty_or():
    from decimal import Decimal
    from datetime import date as Date
    from pypara.currencies import Currencies
    from pypara.money import Money

    somemoney = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    result_defined = somemoney.qty_or(Decimal(0))
    assert result_defined == Decimal('1.00')

    nonemoney = Money.of(None, Decimal('1'), None)
    result_undefined = nonemoney.qty_or(Decimal(0))
    assert result_undefined == Decimal('0')

    nonemoney2 = Money.of(None, None, None)
    result_undefined2 = nonemoney2.qty_or(Decimal(42))
    assert result_undefined2 == Decimal('42')

    somemoney2 = Money.of(Currencies["EUR"], Decimal('5.5'), Date(2020, 6, 15))
    result_defined2 = somemoney2.qty_or(Decimal(100))
    assert result_defined2 == Decimal('5.50')


# LLM-generated content at query #66
#--------------------------

```python
def test_ccy_or_none():
    from decimal import Decimal
    from datetime import date as Date
    from pypara.currencies import Currencies
    from pypara.price import Price
    
    # Test with defined price (all parameters provided)
    defined_price = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    result_defined = defined_price.ccy_or_none()
    assert result_defined is not None
    assert result_defined.code == 'USD'
    
    # Test with undefined price (qty is None)
    undefined_price_qty_none = Price.of(Currencies["USD"], None, Date(2019, 1, 1))
    result_qty_none = undefined_price_qty_none.ccy_or_none()
    assert result_qty_none is None
    
    # Test with undefined price (ccy is None)
    undefined_price_ccy_none = Price.of(None, Decimal('1'), Date(2019, 1, 1))
    result_ccy_none = undefined_price_ccy_none.ccy_or_none()
    assert result_ccy_none is None
    
    # Test with undefined price (dov is None)
    undefined_price_dov_none = Price.of(Currencies["USD"], Decimal('1'), None)
    result_dov_none = undefined_price_dov_none.ccy_or_none()
    assert result_dov_none is None
    
    # Test with Price.na()
    none_price = Price.na()
    result_na = none_price.ccy_or_none()
    assert result_na is None


# LLM-generated content at query #67
#--------------------------

```python
def test_truediv_with_valid_divisor():
    from decimal import Decimal
    from datetime import date
    
    ccy = Currency(code="USD", quantizer=Decimal("0.01"))
    price = SomePrice(ccy=ccy, qty=Decimal("100"), dov=date(2024, 1, 1))
    result = price / 2
    
    assert isinstance(result, SomePrice)
    assert result.ccy == ccy
    assert result.qty == Decimal("50")
    assert result.dov == date(2024, 1, 1)


def test_truediv_with_zero_divisor():
    from decimal import Decimal
    from datetime import date
    
    ccy = Currency(code="USD", quantizer=Decimal("0.01"))
    price = SomePrice(ccy=ccy, qty=Decimal("100"), dov=date(2024, 1, 1))
    result = price / 0
    
    assert result is NoPrice


def test_truediv_with_decimal_divisor():
    from decimal import Decimal
    from datetime import date
    
    ccy = Currency(code="USD", quantizer=Decimal("0.01"))
    price = SomePrice(ccy=ccy, qty=Decimal("100"), dov=date(2024, 1, 1))
    result = price / Decimal("4")
    
    assert isinstance(result, SomePrice)
    assert result.qty == Decimal("25")


def test_truediv_with_float_divisor():
    from decimal import Decimal
    from datetime import date
    
    ccy = Currency(code="USD", quantizer=Decimal("0.01"))
    price = SomePrice(ccy=ccy, qty=Decimal("100"), dov=date(2024, 1, 1))
    result = price / 2.5
    
    assert isinstance(result, SomePrice)
    assert result.qty == Decimal("40")


def test_truediv_preserves_currency_and_date():
    from decimal import Decimal
    from datetime import date
    
    ccy = Currency(code="EUR", quantizer=Decimal("0.01"))
    dov = date(2024, 6, 15)
    price = SomePrice(ccy=ccy, qty=Decimal("200"), dov=dov)
    result = price / 5
    
    assert result.ccy == ccy
    assert result.dov == dov
    assert result.qty == Decimal("40")


# LLM-generated content at query #68
#--------------------------

```python
def test_add_two_defined_money_objects_with_same_currency():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money1 = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('20'), Date(2019, 1, 2))
    result = money1.add(money2)
    
    assert result.defined is True
    assert result.qty == Decimal('30.00')
    assert result.ccy.code == 'USD'


def test_add_defined_money_with_undefined_money():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money1 = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    money2 = Money.na()
    result = money1.add(money2)
    
    assert result.defined is True
    assert result.qty == Decimal('10.00')
    assert result.ccy.code == 'USD'


def test_add_undefined_money_with_defined_money():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money1 = Money.na()
    money2 = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = money1.add(money2)
    
    assert result.defined is True
    assert result.qty == Decimal('10.00')
    assert result.ccy.code == 'USD'


def test_add_two_undefined_money_objects():
    money1 = Money.na()
    money2 = Money.na()
    result = money1.add(money2)
    
    assert result.undefined is True


def test_add_incompatible_currencies_raises_error():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money1 = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["EUR"], Decimal('20'), Date(2019, 1, 1))
    
    try:
        money1.add(money2)
        assert False, "Should raise IncompatibleCurrencyError"
    except Exception as e:
        assert "IncompatibleCurrencyError" in str(type(e))


def test_add_carries_forward_date():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money1 = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('20'), Date(2019, 1, 15))
    result = money1.add(money2)
    
    assert result.dov == Date(2019, 1, 15)


def test_add_negative_quantities():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money1 = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('-5'), Date(2019, 1, 1))
    result = money1.add(money2)
    
    assert result.qty == Decimal('5.00')
    assert result.defined is True


def test_add_zero_quantities():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money1 = Money.of(Currencies["USD"], Decimal('0'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = money1.add(money2)
    
    assert result.qty == Decimal('10.00')


# LLM-generated content at query #69
#--------------------------

```python
def test_qty_or():
    from datetime import date as Date
    from decimal import Decimal
    from pypara.currencies import Currencies
    from pypara.price import Price
    
    # Test with defined price - should return qty
    someprice = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    result = someprice.qty_or(Decimal(0))
    assert result == Decimal('1')
    
    # Test with undefined price - should return default
    noneprice = Price.of(None, Decimal('1'), None)
    result = noneprice.qty_or(Decimal(0))
    assert result == Decimal('0')
    
    # Test with defined price and different default value
    someprice2 = Price.of(Currencies["EUR"], Decimal('42'), Date(2020, 5, 15))
    result = someprice2.qty_or(Decimal(99))
    assert result == Decimal('42')
    
    # Test with undefined price and different default value
    noneprice2 = Price.of(None, Decimal('10'), None)
    result = noneprice2.qty_or(Decimal(100))
    assert result == Decimal('100')


# LLM-generated content at query #70
#--------------------------

```python
def test_price_eq():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.prices import Price
    
    price1 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    price3 = Price.of(Currencies["EUR"], Decimal('1'), Date(2019, 1, 1))
    price4 = Price.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    price5 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 2))
    undefined1 = Price.na()
    undefined2 = Price.na()
    
    assert price1 == price2
    assert price1 != price3
    assert price1 != price4
    assert price1 != price5
    assert undefined1 == undefined2
    assert price1 != undefined1
    assert price1 != "not a price"
    assert price1 != None
    assert price1 != 1


# LLM-generated content at query #71
#--------------------------

```python
def test_scalar_subtract():
    from decimal import Decimal
    from datetime import date as Date
    from pypara.currencies import Currencies
    from pypara.money import Money
    
    # Test scalar_subtract with defined money
    defined_money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = defined_money.scalar_subtract(Decimal('3'))
    assert result.qty == Decimal('7.00')
    assert result.ccy.code == 'USD'
    assert result.dov == Date(2019, 1, 1)
    
    # Test scalar_subtract with zero
    result_zero = defined_money.scalar_subtract(Decimal('0'))
    assert result_zero.qty == Decimal('10.00')
    
    # Test scalar_subtract resulting in negative
    result_negative = defined_money.scalar_subtract(Decimal('15'))
    assert result_negative.qty == Decimal('-5.00')
    
    # Test scalar_subtract with undefined money
    undefined_money = Money.na()
    result_undefined = undefined_money.scalar_subtract(Decimal('5'))
    assert result_undefined.undefined
    
    # Test scalar_subtract with decimal precision
    precise_money = Money.of(Currencies["EUR"], Decimal('100.50'), Date(2020, 6, 15))
    result_precise = precise_money.scalar_subtract(Decimal('0.25'))
    assert result_precise.qty == Decimal('100.25')
    assert result_precise.ccy.code == 'EUR'
    assert result_precise.dov == Date(2020, 6, 15)


# LLM-generated content at query #72
#--------------------------

```python
def test_with_ccy_defined_money():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.money import Money
    
    original_money = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    new_money = original_money.with_ccy(Currencies["EUR"])
    
    assert new_money.ccy_or_none() == Currencies["EUR"]
    assert new_money.qty_or_none() == original_money.qty_or_none()
    assert new_money.dov_or_none() == original_money.dov_or_none()
    assert new_money.defined is True


def test_with_ccy_undefined_money():
    from pypara.currencies import Currencies
    from pypara.money import Money
    
    undefined_money = Money.na()
    result = undefined_money.with_ccy(Currencies["USD"])
    
    assert result is undefined_money
    assert result.undefined is True


def test_with_ccy_changes_currency_only():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.money import Money
    
    original_money = Money.of(Currencies["GBP"], Decimal('50.25'), Date(2020, 6, 15))
    new_money = original_money.with_ccy(Currencies["JPY"])
    
    assert new_money.ccy_or_none() == Currencies["JPY"]
    assert new_money.qty_or_none() == Decimal('50.25')
    assert new_money.dov_or_none() == Date(2020, 6, 15)


# LLM-generated content at query #73
#--------------------------

```python
def test_subtract():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    
    # Test subtracting two defined money objects with same currency
    money1 = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('3'), Date(2019, 1, 1))
    result = money1.subtract(money2)
    assert result.defined
    assert result.qty == Decimal('7.00')
    assert result.ccy.code == "USD"
    
    # Test subtracting undefined money from defined money
    defined_money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    undefined_money = Money.na()
    result = defined_money.subtract(undefined_money)
    assert result.defined
    assert result.qty == defined_money.qty
    
    # Test subtracting defined money from undefined money
    result = undefined_money.subtract(defined_money)
    assert result.defined
    assert result.qty == defined_money.qty
    
    # Test subtracting two undefined money objects
    result = undefined_money.subtract(undefined_money)
    assert result.undefined
    
    # Test subtracting money with different currencies raises IncompatibleCurrencyError
    money_usd = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    money_eur = Money.of(Currencies["EUR"], Decimal('5'), Date(2019, 1, 1))
    try:
        money_usd.subtract(money_eur)
        assert False, "Expected IncompatibleCurrencyError"
    except Exception as e:
        assert "IncompatibleCurrencyError" in str(type(e))
    
    # Test subtracting negative amounts
    money_pos = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    money_neg = Money.of(Currencies["USD"], Decimal('-5'), Date(2019, 1, 1))
    result = money_pos.subtract(money_neg)
    assert result.qty == Decimal('15.00')
    
    # Test date is carried forward
    money_a = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 5))
    money_b = Money.of(Currencies["USD"], Decimal('50'), Date(2019, 1, 3))
    result = money_a.subtract(money_b)
    assert result.dov == Date(2019, 1, 5)


# LLM-generated content at query #74
#--------------------------

```python
def test_price_fmap_with_defined_price():
    from datetime import date as Date
    from decimal import Decimal
    from pypara.currencies import Currencies
    from pypara.price import Price
    
    someprice = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    new = someprice.fmap(lambda x: Price.of(x.ccy, x.qty + Decimal('1'), x.dov))
    
    assert new.ccy.code == 'USD'
    assert new.qty == Decimal('2')
    assert new.dov == Date(2019, 1, 1)


def test_price_fmap_with_undefined_price():
    from decimal import Decimal
    from pypara.price import Price
    
    noneprice = Price.of(None, Decimal('1'), None)
    result = noneprice.fmap(lambda sp: Price.of(sp.ccy, sp.qty + Decimal('1'), sp.dov))
    
    assert result.undefined is True


def test_price_fmap_transforms_quantity():
    from datetime import date as Date
    from decimal import Decimal
    from pypara.currencies import Currencies
    from pypara.price import Price
    
    someprice = Price.of(Currencies["USD"], Decimal('5'), Date(2019, 1, 1))
    new = someprice.fmap(lambda x: Price.of(x.ccy, x.qty * Decimal('2'), x.dov))
    
    assert new.qty == Decimal('10')


def test_price_fmap_transforms_date():
    from datetime import date as Date, timedelta
    from decimal import Decimal
    from pypara.currencies import Currencies
    from pypara.price import Price
    
    someprice = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    new = someprice.fmap(lambda x: Price.of(x.ccy, x.qty, x.dov + timedelta(days=10)))
    
    assert new.dov == Date(2019, 1, 11)


def test_price_fmap_transforms_currency():
    from datetime import date as Date
    from decimal import Decimal
    from pypara.currencies import Currencies
    from pypara.price import Price
    
    someprice = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    new = someprice.fmap(lambda x: Price.of(Currencies["EUR"], x.qty, x.dov))
    
    assert new.ccy.code == 'EUR'


def test_price_fmap_preserves_defined_state():
    from datetime import date as Date
    from decimal import Decimal
    from pypara.currencies import Currencies
    from pypara.price import Price
    
    someprice = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    new = someprice.fmap(lambda x: Price.of(x.ccy, x.qty, x.dov))
    
    assert new.defined is True


# LLM-generated content at query #75
#--------------------------

```python
def test_price_floordiv():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    # Test floor division with defined price and numeric value
    price1 = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result1 = price1 // Decimal('3')
    assert result1.qty_or_zero() == Decimal('3')
    assert result1.ccy_or_none().code == "USD"
    
    # Test floor division with defined price and integer
    price2 = Price.of(Currencies["EUR"], Decimal('20'), Date(2019, 1, 1))
    result2 = price2 // 6
    assert result2.qty_or_zero() == Decimal('3')
    assert result2.ccy_or_none().code == "EUR"
    
    # Test floor division by zero returns undefined price
    price3 = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result3 = price3 // Decimal('0')
    assert result3.undefined
    
    # Test floor division on undefined price returns itself
    price4 = Price.na()
    result4 = price4 // Decimal('5')
    assert result4.undefined
    
    # Test floor division with negative quantity
    price5 = Price.of(Currencies["USD"], Decimal('-10'), Date(2019, 1, 1))
    result5 = price5 // Decimal('3')
    assert result5.qty_or_zero() == Decimal('-4')
    
    # Test floor division with decimal result
    price6 = Price.of(Currencies["GBP"], Decimal('7'), Date(2019, 1, 1))
    result6 = price6 // Decimal('2')
    assert result6.qty_or_zero() == Decimal('3')


# LLM-generated content at query #76
#--------------------------

```python
def test_price_fmap_with_defined_price():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    import datetime
    
    someprice = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    new = someprice.fmap(lambda x: Price.of(x.ccy, x.qty + Decimal('1'), x.dov + datetime.timedelta(days=10)))
    
    assert new.ccy.code == 'USD'
    assert new.qty == Decimal('2')
    assert new.dov == Date(2019, 1, 11)


def test_price_fmap_with_undefined_price():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    
    noneprice = Price.of(None, Decimal('1'), None)
    result = noneprice.fmap(lambda sp: Price.of(sp.ccy, sp.qty + Decimal('1'), sp.dov))
    
    assert result.undefined


def test_price_fmap_identity_function():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    
    someprice = Price.of(Currencies["USD"], Decimal('5'), Date(2020, 6, 15))
    result = someprice.fmap(lambda x: x)
    
    assert result is someprice


def test_price_fmap_chain_operations():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    import datetime
    
    someprice = Price.of(Currencies["EUR"], Decimal('10'), Date(2021, 3, 1))
    result = someprice.fmap(lambda x: Price.of(x.ccy, x.qty * Decimal('2'), x.dov + datetime.timedelta(days=5)))
    
    assert result.qty == Decimal('20')
    assert result.dov == Date(2021, 3, 6)
    assert result.ccy.code == 'EUR'


# LLM-generated content at query #77
#--------------------------

```python
def test_money_int_conversion():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.money import Money
    
    # Test __int__ with defined money
    money_defined = Money.of(Currencies["USD"], Decimal('42.75'), Date(2019, 1, 1))
    result = int(money_defined)
    assert result == 42
    
    # Test __int__ with negative quantity
    money_negative = Money.of(Currencies["EUR"], Decimal('-15.99'), Date(2019, 1, 1))
    result_negative = int(money_negative)
    assert result_negative == -15
    
    # Test __int__ with zero quantity
    money_zero = Money.of(Currencies["GBP"], Decimal('0'), Date(2019, 1, 1))
    result_zero = int(money_zero)
    assert result_zero == 0
    
    # Test __int__ with undefined money raises exception
    money_undefined = Money.na()
    try:
        int(money_undefined)
        assert False, "Should have raised MonetaryOperationException"
    except Exception as e:
        assert "MonetaryOperationException" in str(type(e).__name__)


# LLM-generated content at query #78
#--------------------------

```python
def test_ccy_or_none():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    from pypara.price import Price
    
    # Test defined price returns currency
    someprice = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    result = someprice.ccy_or_none()
    assert result is not None
    assert result.code == 'USD'
    
    # Test undefined price returns None
    noneprice = Price.of(None, Decimal('1'), None)
    result = noneprice.ccy_or_none()
    assert result is None
    
    # Test another currency
    eur_price = Price.of(Currencies["EUR"], Decimal('2'), Date(2019, 1, 1))
    result = eur_price.ccy_or_none()
    assert result is not None
    assert result.code == 'EUR'


# LLM-generated content at query #79
#--------------------------

```python
def test_qty_or_zero():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    
    # Test with defined money - should return the quantity
    somemoney = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    result = somemoney.qty_or_zero()
    assert result == Decimal('1.00')
    
    # Test with undefined money - should return 0
    nonemoney = Money.of(None, Decimal('1'), None)
    result = nonemoney.qty_or_zero()
    assert result == Decimal('0')
    
    # Test with Money.na() - should return 0
    result = Money.na().qty_or_zero()
    assert result == Decimal('0')
    
    # Test with defined money with different quantity
    somemoney2 = Money.of(Currencies["EUR"], Decimal('42.50'), Date(2020, 6, 15))
    result = somemoney2.qty_or_zero()
    assert result == Decimal('42.50')


# LLM-generated content at query #80
#--------------------------

```python
def test_price_gt():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    # Create test prices
    price_usd_10 = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    price_usd_5 = Price.of(Currencies["USD"], Decimal('5'), Date(2019, 1, 1))
    price_undefined = Price.na()
    
    # Test: defined price is greater than smaller defined price
    assert price_usd_10.gt(price_usd_5) is True
    
    # Test: defined price is not greater than larger defined price
    assert price_usd_5.gt(price_usd_10) is False
    
    # Test: defined price is not greater than equal defined price
    assert price_usd_10.gt(price_usd_10) is False
    
    # Test: defined price is greater than undefined price
    assert price_usd_10.gt(price_undefined) is True
    
    # Test: undefined price is never greater than defined price
    assert price_undefined.gt(price_usd_10) is False
    
    # Test: undefined price is never greater than undefined price
    assert price_undefined.gt(price_undefined) is False


# LLM-generated content at query #81
#--------------------------

```python
def test_ccy_or_none():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    from pypara.money import Money

    # Test with defined money - should return currency
    somemoney = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    result = somemoney.ccy_or_none()
    assert result is not None
    assert result.code == 'USD'

    # Test with undefined money - should return None
    nonemoney = Money.of(None, Decimal('1'), None)
    result = nonemoney.ccy_or_none()
    assert result is None

    # Test with Money.na() - should return None
    na_money = Money.na()
    result = na_money.ccy_or_none()
    assert result is None


# LLM-generated content at query #82
#--------------------------

```python
def test_divide_defined_price():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    price = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = price.divide(Decimal('2'))
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('5')
    assert result.ccy_or_none().code == "USD"


def test_divide_undefined_price():
    price = Price.na()
    result = price.divide(Decimal('2'))
    
    assert result.undefined
    assert result is price


def test_divide_by_zero():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    price = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = price.divide(Decimal('0'))
    
    assert result.undefined


def test_divide_preserves_currency():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    price = Price.of(Currencies["EUR"], Decimal('100'), Date(2019, 6, 15))
    result = price.divide(Decimal('4'))
    
    assert result.defined
    assert result.ccy_or_none().code == "EUR"
    assert result.qty_or_zero() == Decimal('25')


def test_divide_fractional_result():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    price = Price.of(Currencies["USD"], Decimal('7'), Date(2019, 1, 1))
    result = price.divide(Decimal('2'))
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('3.5')


# LLM-generated content at query #83
#--------------------------

```python
def test_is_equal():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal

    # Create two identical defined prices
    price1 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    
    # Create a different defined price
    price3 = Price.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    
    # Create two undefined prices
    price_na1 = Price.na()
    price_na2 = Price.na()
    
    # Test equal defined prices
    assert price1.is_equal(price2) == True
    
    # Test different quantities
    assert price1.is_equal(price3) == False
    
    # Test different currencies
    price4 = Price.of(Currencies["EUR"], Decimal('1'), Date(2019, 1, 1))
    assert price1.is_equal(price4) == False
    
    # Test different dates
    price5 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 2))
    assert price1.is_equal(price5) == False
    
    # Test undefined prices
    assert price_na1.is_equal(price_na2) == True
    
    # Test defined vs undefined
    assert price1.is_equal(price_na1) == False
    assert price_na1.is_equal(price1) == False
    
    # Test comparison with non-Price object
    assert price1.is_equal("not a price") == False
    assert price1.is_equal(None) == False
    assert price1.is_equal(Decimal('1')) == False


# LLM-generated content at query #84
#--------------------------

```python
def test_floordiv_with_valid_divisor():
    from decimal import Decimal
    from datetime import date
    
    ccy = Currency(code="USD", quantizer=Decimal("0.01"))
    price = SomePrice(ccy=ccy, qty=Decimal("10.5"), dov=date(2023, 1, 1))
    
    result = price.__floordiv__(3)
    
    assert isinstance(result, SomePrice)
    assert result.ccy == ccy
    assert result.qty == Decimal("3")
    assert result.dov == date(2023, 1, 1)


def test_floordiv_with_decimal_divisor():
    from decimal import Decimal
    from datetime import date
    
    ccy = Currency(code="USD", quantizer=Decimal("0.01"))
    price = SomePrice(ccy=ccy, qty=Decimal("20"), dov=date(2023, 1, 1))
    
    result = price.__floordiv__(Decimal("3"))
    
    assert isinstance(result, SomePrice)
    assert result.qty == Decimal("6")


def test_floordiv_with_zero_divisor():
    from decimal import Decimal
    from datetime import date
    
    ccy = Currency(code="USD", quantizer=Decimal("0.01"))
    price = SomePrice(ccy=ccy, qty=Decimal("10"), dov=date(2023, 1, 1))
    
    result = price.__floordiv__(0)
    
    assert result is NoPrice


def test_floordiv_preserves_currency_and_date():
    from decimal import Decimal
    from datetime import date
    
    ccy = Currency(code="EUR", quantizer=Decimal("0.01"))
    dov = date(2023, 6, 15)
    price = SomePrice(ccy=ccy, qty=Decimal("100"), dov=dov)
    
    result = price.__floordiv__(7)
    
    assert result.ccy == ccy
    assert result.dov == dov


def test_floordiv_with_negative_divisor():
    from decimal import Decimal
    from datetime import date
    
    ccy = Currency(code="USD", quantizer=Decimal("0.01"))
    price = SomePrice(ccy=ccy, qty=Decimal("10"), dov=date(2023, 1, 1))
    
    result = price.__floordiv__(-3)
    
    assert isinstance(result, SomePrice)
    assert result.qty == Decimal("-4")


def test_floordiv_with_fractional_result():
    from decimal import Decimal
    from datetime import date
    
    ccy = Currency(code="USD", quantizer=Decimal("0.01"))
    price = SomePrice(ccy=ccy, qty=Decimal("7.5"), dov=date(2023, 1, 1))
    
    result = price.__floordiv__(2)
    
    assert result.qty == Decimal("3")


# LLM-generated content at query #85
#--------------------------

```python
def test_price_pos():
    from decimal import Decimal
    from datetime import date as Date
    from pypara.currencies import Currencies
    from pypara.price import Price

    # Test with defined price
    defined_price = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    pos_result = +defined_price
    assert pos_result.qty_or_zero() == Decimal('10')
    assert pos_result.ccy_or_none().code == 'USD'
    assert pos_result.dov_or_none() == Date(2019, 1, 1)

    # Test with undefined price
    undefined_price = Price.na()
    pos_result_undefined = +undefined_price
    assert pos_result_undefined.undefined


# LLM-generated content at query #86
#--------------------------

```python
def test_somemoney_ge():
    from decimal import Decimal
    from datetime import date
    
    # Create mock Currency object
    class MockCurrency:
        def __init__(self, code):
            self.code = code
            self.decimals = 2
            self.quantizer = Decimal('0.01')
        
        def __eq__(self, other):
            return self.code == other.code
        
        def __ne__(self, other):
            return not self.__eq__(other)
    
    usd = MockCurrency('USD')
    eur = MockCurrency('EUR')
    
    # Create SomeMoney instances
    money1 = SomeMoney(usd, Decimal('100.00'), date(2024, 1, 1))
    money2 = SomeMoney(usd, Decimal('50.00'), date(2024, 1, 1))
    money3 = SomeMoney(usd, Decimal('100.00'), date(2024, 1, 1))
    money4 = SomeMoney(eur, Decimal('100.00'), date(2024, 1, 1))
    
    # Test greater than or equal with same currency, money1 >= money2
    assert money1.__ge__(money2) is True
    
    # Test greater than or equal with same currency, money2 >= money1
    assert money2.__ge__(money1) is False
    
    # Test greater than or equal with same currency and equal quantities
    assert money1.__ge__(money3) is True
    
    # Test greater than or equal with NoMoney (undefined)
    assert money1.__ge__(SomeMoney(usd, Decimal('200.00'), date(2024, 1, 1))) is False
    
    # Test greater than or equal with different currency raises exception
    try:
        money1.__ge__(money4)
        assert False, "Should raise IncompatibleCurrencyError"
    except Exception as e:
        assert "IncompatibleCurrencyError" in str(type(e).__name__)


# LLM-generated content at query #87
#--------------------------

```python
def test_price_is_equal():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    # Create two identical defined prices
    price1 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert price1.is_equal(price2) is True
    
    # Create two different defined prices with different quantities
    price3 = Price.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    assert price1.is_equal(price3) is False
    
    # Create two different defined prices with different currencies
    price4 = Price.of(Currencies["EUR"], Decimal('1'), Date(2019, 1, 1))
    assert price1.is_equal(price4) is False
    
    # Create two different defined prices with different dates
    price5 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 2))
    assert price1.is_equal(price5) is False
    
    # Compare two undefined prices
    undefined1 = Price.na()
    undefined2 = Price.na()
    assert undefined1.is_equal(undefined2) is True
    
    # Compare defined price with undefined price
    assert price1.is_equal(undefined1) is False
    
    # Compare undefined price with defined price
    assert undefined1.is_equal(price1) is False
    
    # Compare price with non-price object
    assert price1.is_equal("not a price") is False
    assert price1.is_equal(123) is False
    assert price1.is_equal(None) is False


# LLM-generated content at query #88
#--------------------------

```python
def test_convert_predicate_at_line_12_evaluates_to_false():
    from datetime import date
    from decimal import Decimal
    from pypara.currencies import Currency, CurrencyType
    from pypara.monetary import SomePrice
    from unittest.mock import Mock, patch
    
    # Create currencies
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    
    # Create a price
    price = SomePrice(usd, Decimal("100.00"), date(2023, 1, 1))
    
    # Create a mock FX rate
    mock_rate = Mock()
    mock_rate.value = Decimal("0.85")
    
    # Patch FXRateService.default to be a non-None object
    with patch('pypara.monetary.FXRateService') as mock_service:
        mock_service.default = Mock()
        mock_service.default.query = Mock(return_value=mock_rate)
        
        # Call convert - this should succeed without raising ProgrammingError
        result = price.convert(eur, date(2023, 1, 1), strict=False)
        
        # Verify the result
        assert result.ccy == eur
        assert result.qty == Decimal("85.00")
        assert result.dov == date(2023, 1, 1)


# LLM-generated content at query #89
#--------------------------

```python
def test_with_ccy():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    # Test with_ccy on a defined price object
    usd = Currencies["USD"]
    eur = Currencies["EUR"]
    original_price = Price.of(usd, Decimal('100'), Date(2019, 1, 1))
    new_price = original_price.with_ccy(eur)
    
    assert new_price.ccy_or_none() == eur
    assert new_price.qty_or_zero() == Decimal('100')
    assert new_price.dov_or_none() == Date(2019, 1, 1)
    
    # Test with_ccy on an undefined price object
    undefined_price = Price.na()
    result_price = undefined_price.with_ccy(eur)
    
    assert result_price is undefined_price
    assert result_price.undefined


# LLM-generated content at query #90
#--------------------------

```python
def test_someprice_le():
    from decimal import Decimal
    from datetime import date
    
    # Create mock Currency objects
    class MockCurrency:
        def __init__(self, code):
            self.code = code
        
        def __eq__(self, other):
            return isinstance(other, MockCurrency) and self.code == other.code
        
        def __ne__(self, other):
            return not self.__eq__(other)
    
    usd = MockCurrency("USD")
    eur = MockCurrency("EUR")
    
    # Create SomePrice instances
    price1 = SomePrice(usd, Decimal("100"), date(2023, 1, 1))
    price2 = SomePrice(usd, Decimal("150"), date(2023, 1, 1))
    price3 = SomePrice(usd, Decimal("100"), date(2023, 1, 1))
    
    # Test: price1 <= price2 (less than)
    assert price1 <= price2
    
    # Test: price1 <= price3 (equal)
    assert price1 <= price3
    
    # Test: price2 <= price1 (greater than)
    assert not (price2 <= price1)
    
    # Test: with non-SomePrice object
    assert not (price1 <= "not a price")
    assert not (price1 <= None)
    assert not (price1 <= 100)
    
    # Test: incompatible currencies raises error
    price_eur = SomePrice(eur, Decimal("100"), date(2023, 1, 1))
    try:
        price1 <= price_eur
        assert False, "Should have raised IncompatibleCurrencyError"
    except Exception as e:
        assert "IncompatibleCurrencyError" in str(type(e).__name__)


# LLM-generated content at query #91
#--------------------------

```python
def test_convert_predicate_line_12_evaluates_to_false():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currency, CurrencyType
    from pypara.monetary import SomeMoney
    from unittest.mock import Mock, patch
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    
    money = SomeMoney(usd, Decimal("100.00"), date(2023, 1, 1))
    
    mock_rate = Mock()
    mock_rate.value = Decimal("0.85")
    
    with patch('pypara.monetary.FXRateService') as mock_service:
        mock_service.default = Mock()
        mock_service.default.query = Mock(return_value=mock_rate)
        
        result = money.convert(eur, date(2023, 1, 1), False)
        
        assert result.ccy == eur
        assert result.qty == Decimal("85.00")


# LLM-generated content at query #92
#--------------------------

```python
def test_price_gt():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    # Create test prices
    price_usd_100 = Price.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    price_usd_50 = Price.of(Currencies["USD"], Decimal('50'), Date(2019, 1, 1))
    price_usd_100_other_date = Price.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 2))
    price_undefined = Price.na()
    
    # Test: defined price greater than smaller defined price with same currency
    assert price_usd_100.gt(price_usd_50) is True
    
    # Test: defined price not greater than larger defined price with same currency
    assert price_usd_50.gt(price_usd_100) is False
    
    # Test: defined price not greater than equal defined price with same currency
    assert price_usd_100.gt(price_usd_100_other_date) is False
    
    # Test: defined price is always greater than undefined price
    assert price_usd_100.gt(price_undefined) is True
    
    # Test: undefined price is never greater than defined price
    assert price_undefined.gt(price_usd_100) is False
    
    # Test: undefined price is never greater than undefined price
    assert price_undefined.gt(price_undefined) is False


# LLM-generated content at query #93
#--------------------------

```python
def test_lte_raises_error_when_currencies_differ():
    from decimal import Decimal
    from datetime import date
    
    # Create mock Currency objects
    class MockCurrency:
        def __init__(self, code):
            self.code = code
            self.decimals = 2
            self.quantizer = Decimal('0.01')
        
        def __eq__(self, other):
            return self.code == other.code
        
        def __ne__(self, other):
            return self.code != other.code
    
    # Create mock IncompatibleCurrencyError
    class IncompatibleCurrencyError(Exception):
        def __init__(self, ccy1, ccy2, operation):
            self.ccy1 = ccy1
            self.ccy2 = ccy2
            self.operation = operation
    
    # Patch the exception in the module
    import sys
    from unittest.mock import patch
    
    ccy_usd = MockCurrency("USD")
    ccy_eur = MockCurrency("EUR")
    test_date = date(2024, 1, 1)
    
    money1 = SomeMoney(ccy_usd, Decimal("100.00"), test_date)
    money2 = SomeMoney(ccy_eur, Decimal("100.00"), test_date)
    
    try:
        money1.lte(money2)
        assert False, "Expected IncompatibleCurrencyError to be raised"
    except Exception as e:
        assert "comparison" in str(e.__class__.__name__).lower() or "<=" in str(e)


# LLM-generated content at query #94
#--------------------------

```python
def test_money_eq():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    # Test equality of two defined money objects with same values
    money1 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert money1 == money2
    
    # Test inequality of two defined money objects with different quantities
    money3 = Money.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    assert money1 != money3
    
    # Test inequality of two defined money objects with different currencies
    money4 = Money.of(Currencies["EUR"], Decimal('1'), Date(2019, 1, 1))
    assert money1 != money4
    
    # Test inequality of two defined money objects with different dates
    money5 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 2))
    assert money1 != money5
    
    # Test equality of two undefined money objects
    money_na1 = Money.na()
    money_na2 = Money.na()
    assert money_na1 == money_na2
    
    # Test inequality of defined and undefined money objects
    assert money1 != money_na1
    
    # Test inequality of money object and non-money object
    assert money1 != Decimal('1')
    assert money1 != "1"
    assert money1 != None
    
    # Test equality of same money object instance
    assert money1 == money1


# LLM-generated content at query #95
#--------------------------

```python
def test_money_gt():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    # Test: defined money > undefined money should be True
    defined_money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    undefined_money = Money.na()
    assert defined_money.gt(undefined_money) is True
    
    # Test: undefined money > defined money should be False
    assert undefined_money.gt(defined_money) is False
    
    # Test: undefined money > undefined money should be False
    assert undefined_money.gt(undefined_money) is False
    
    # Test: defined money with larger qty > defined money with smaller qty should be True
    larger_money = Money.of(Currencies["USD"], Decimal('20'), Date(2019, 1, 1))
    smaller_money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    assert larger_money.gt(smaller_money) is True
    
    # Test: defined money with smaller qty > defined money with larger qty should be False
    assert smaller_money.gt(larger_money) is False
    
    # Test: defined money with equal qty > defined money with equal qty should be False
    equal_money1 = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    equal_money2 = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 2))
    assert equal_money1.gt(equal_money2) is False


# LLM-generated content at query #96
#--------------------------

```python
def test_gte_raises_incompatible_currency_error_when_currencies_differ():
    from decimal import Decimal
    from datetime import date
    
    ccy1 = Currency(code="USD", quantizer=Decimal("0.01"))
    ccy2 = Currency(code="EUR", quantizer=Decimal("0.01"))
    dov = date(2023, 1, 1)
    
    price1 = SomePrice(ccy1, Decimal("100.00"), dov)
    price2 = SomePrice(ccy2, Decimal("50.00"), dov)
    
    try:
        price1.gte(price2)
        assert False, "Expected IncompatibleCurrencyError to be raised"
    except IncompatibleCurrencyError as e:
        assert e.ccy1 == ccy1
        assert e.ccy2 == ccy2
        assert ">= comparison" in str(e.operation)


# LLM-generated content at query #97
#--------------------------

```python
def test_price_float_conversion():
    from decimal import Decimal
    from datetime import date as Date
    from pypara.currencies import Currencies
    from pypara.price import Price
    
    # Test with defined price
    defined_price = Price.of(Currencies["USD"], Decimal('123.456'), Date(2019, 1, 1))
    result = float(defined_price)
    assert result == 123.456
    assert isinstance(result, float)
    
    # Test with zero quantity
    zero_price = Price.of(Currencies["USD"], Decimal('0'), Date(2019, 1, 1))
    result = float(zero_price)
    assert result == 0.0
    
    # Test with negative quantity
    negative_price = Price.of(Currencies["USD"], Decimal('-50.25'), Date(2019, 1, 1))
    result = float(negative_price)
    assert result == -50.25
    
    # Test with large quantity
    large_price = Price.of(Currencies["USD"], Decimal('999999.999999'), Date(2019, 1, 1))
    result = float(large_price)
    assert result == 999999.999999


# LLM-generated content at query #98
#--------------------------

```python
def test_divide_defined_price_by_positive_number():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.price import Price
    
    price = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = price.divide(Decimal('2'))
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('5')


def test_divide_defined_price_by_zero():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.price import Price
    
    price = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = price.divide(Decimal('0'))
    
    assert result.undefined


def test_divide_undefined_price():
    from pypara.price import Price
    
    price = Price.na()
    result = price.divide(Decimal('2'))
    
    assert result.undefined


def test_divide_defined_price_by_negative_number():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.price import Price
    
    price = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = price.divide(Decimal('-2'))
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('-5')


def test_divide_defined_price_by_decimal_number():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.price import Price
    
    price = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = price.divide(Decimal('2.5'))
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('4')


def test_divide_preserves_currency():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.price import Price
    
    price = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = price.divide(Decimal('2'))
    
    assert result.ccy_or_none().code == 'USD'


def test_divide_preserves_date():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.price import Price
    
    original_date = Date(2019, 1, 1)
    price = Price.of(Currencies["USD"], Decimal('10'), original_date)
    result = price.divide(Decimal('2'))
    
    assert result.dov_or_none() == original_date


# LLM-generated content at query #99
#--------------------------

```python
def test_divide_by_zero_returns_no_price():
    from decimal import Decimal, InvalidOperation, DivisionByZero
    from datetime import date
    
    # Create a mock Currency object
    class MockCurrency:
        def __init__(self):
            self.quantizer = Decimal('0.01')
    
    ccy = MockCurrency()
    qty = Decimal('100')
    dov = date(2023, 1, 1)
    
    price = SomePrice(ccy, qty, dov)
    result = price.divide(0)
    
    assert result is NoPrice


# LLM-generated content at query #100
#--------------------------

```python
def test_dimap_with_defined_money():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    somemoney = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    result = somemoney.dimap(lambda x: x.ccy.code, lambda: "EUR")
    assert result == "USD"


def test_dimap_with_undefined_money():
    from decimal import Decimal
    
    nonemoney = Money.of(None, Decimal('1'), None)
    result = nonemoney.dimap(lambda x: x.ccy.code, lambda: "EUR")
    assert result == "EUR"


def test_dimap_applies_function_to_defined_money():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    somemoney = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    result = somemoney.dimap(lambda x: x.qty * Decimal('2'), lambda: Decimal('0'))
    assert result == Decimal('200.00')


def test_dimap_calls_else_for_undefined_money():
    from decimal import Decimal
    
    nonemoney = Money.of(None, Decimal('1'), None)
    result = nonemoney.dimap(lambda x: x.qty + Decimal('10'), lambda: Decimal('999'))
    assert result == Decimal('999')


def test_dimap_with_date_extraction():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    somemoney = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    result = somemoney.dimap(lambda x: x.dov, lambda: Date(2000, 1, 1))
    assert result == Date(2019, 1, 1)


def test_dimap_with_undefined_money_date_fallback():
    from decimal import Decimal
    from datetime import date as Date
    
    nonemoney = Money.of(None, Decimal('1'), None)
    result = nonemoney.dimap(lambda x: x.dov, lambda: Date(2000, 1, 1))
    assert result == Date(2000, 1, 1)


