####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_Price_qty_or():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currencies
    
    # Test with defined price
    defined_price = Price.of(Currencies["USD"], Decimal("10.5"), date(2023, 1, 1))
    assert defined_price.qty_or(Decimal("0")) == Decimal("10.5")
    assert defined_price.qty_or(Decimal("100")) == Decimal("10.5")
    
    # Test with undefined price (None currency)
    undefined_price1 = Price.of(None, Decimal("10.5"), date(2023, 1, 1))
    assert undefined_price1.qty_or(Decimal("0")) == Decimal("0")
    assert undefined_price1.qty_or(Decimal("100")) == Decimal("100")
    
    # Test with undefined price (None quantity)
    undefined_price2 = Price.of(Currencies["USD"], None, date(2023, 1, 1))
    assert undefined_price2.qty_or(Decimal("0")) == Decimal("0")
    assert undefined_price2.qty_or(Decimal("50.25")) == Decimal("50.25")
    
    # Test with undefined price (None date)
    undefined_price3 = Price.of(Currencies["EUR"], Decimal("20.75"), None)
    assert undefined_price3.qty_or(Decimal("0")) == Decimal("0")
    assert undefined_price3.qty_or(Decimal("1.5")) == Decimal("1.5")
    
    # Test with completely undefined price
    undefined_price4 = Price.na()
    assert undefined_price4.qty_or(Decimal("0")) == Decimal("0")
    assert undefined_price4.qty_or(Decimal("999.99")) == Decimal("999.99")
    
    # Test with different default types (should accept any Decimal)
    assert defined_price.qty_or(Decimal("-10")) == Decimal("10.5")
    assert undefined_price1.qty_or(Decimal("-10")) == Decimal("-10")
    
    # Test that default is returned exactly as provided
    default_value = Decimal("123.456789")
    result = undefined_price1.qty_or(default_value)
    assert result == default_value
    assert result is default_value


# LLM-generated content at query #2
#--------------------------

```python
def test_NoneMoney_dov_or():
    from datetime import date
    from decimal import Decimal
    from pypara.currencies import Currencies
    from pypara.money import Money

    # Create a NoneMoney instance
    none_money = Money.na()
    
    # Test with a default date
    default_date = date(2001, 1, 1)
    result = none_money.dov_or(default_date)
    
    # Should return the default date
    assert result == default_date
    
    # Test with a different default date
    another_default = date(2023, 12, 31)
    result2 = none_money.dov_or(another_default)
    assert result2 == another_default
    
    # Test that it's not returning None
    assert none_money.dov_or(default_date) is not None
    
    # Verify it's actually a NoneMoney instance
    assert Money.is_none(none_money)
    assert none_money.undefined
    assert not none_money.defined


# LLM-generated content at query #3
#--------------------------

```python
def test_SomePrice_dov_or():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currencies
    
    # Test with defined price
    price = Price.of(Currencies["USD"], Decimal("1"), date(2019, 1, 1))
    assert price.dov_or(date(2001, 1, 1)) == date(2019, 1, 1)
    
    # Test with undefined price (NonePrice)
    undefined_price = Price.of(None, None, date(2019, 1, 1))
    default_date = date(2001, 1, 1)
    assert undefined_price.dov_or(default_date) == default_date
    
    # Test with different default dates
    price = Price.of(Currencies["EUR"], Decimal("100"), date(2020, 12, 31))
    assert price.dov_or(date(1900, 1, 1)) == date(2020, 12, 31)
    
    # Test with price created via SomePrice directly
    some_price = SomePrice(Currencies["JPY"], Decimal("500"), date(2021, 6, 15))
    assert some_price.dov_or(date(2000, 1, 1)) == date(2021, 6, 15)
    
    # Test with Price.na()
    na_price = Price.na()
    assert na_price.dov_or(date(1999, 12, 31)) == date(1999, 12, 31)


# LLM-generated content at query #4
#--------------------------

```python
def test_SomePrice___eq__():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currencies
    
    # Create test data
    usd = Currencies["USD"]
    eur = Currencies["EUR"]
    date1 = date(2023, 1, 1)
    date2 = date(2023, 1, 2)
    
    # Test equality with identical SomePrice objects
    price1 = SomePrice(usd, Decimal("100"), date1)
    price2 = SomePrice(usd, Decimal("100"), date1)
    assert price1 == price2
    
    # Test inequality with different currency
    price3 = SomePrice(eur, Decimal("100"), date1)
    assert price1 != price3
    
    # Test inequality with different quantity
    price4 = SomePrice(usd, Decimal("200"), date1)
    assert price1 != price4
    
    # Test inequality with different date
    price5 = SomePrice(usd, Decimal("100"), date2)
    assert price1 != price5
    
    # Test equality with itself
    assert price1 == price1
    
    # Test inequality with different type
    assert price1 != "not a price"
    assert price1 != 123
    assert price1 != None
    
    # Test inequality with NoPrice
    no_price = Price.na()
    assert price1 != no_price
    
    # Test equality with tuple unpacking (since SomePrice is a NamedTuple)
    # This tests the tuple comparison in is_equal method
    price6 = SomePrice(usd, Decimal("100"), date1)
    assert tuple(price1) == tuple(price6)
    assert price1 == price6
    
    # Test with zero quantity
    price7 = SomePrice(usd, Decimal("0"), date1)
    price8 = SomePrice(usd, Decimal("0"), date1)
    assert price7 == price8
    
    # Test with negative quantity
    price9 = SomePrice(usd, Decimal("-50"), date1)
    price10 = SomePrice(usd, Decimal("-50"), date1)
    assert price9 == price10
    
    # Test with very precise decimal
    price11 = SomePrice(usd, Decimal("123.456789"), date1)
    price12 = SomePrice(usd, Decimal("123.456789"), date1)
    assert price11 == price12


# LLM-generated content at query #5
#--------------------------

```python
def test_Money_round():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currencies
    
    # Test with defined money
    eur = Currencies["EUR"]
    money = Money.of(eur, Decimal("123.456"), date(2023, 1, 1))
    
    # Test rounding to 0 digits (default)
    rounded = money.round()
    assert rounded.qty == Decimal("123")
    assert rounded.ccy == eur
    assert rounded.dov == date(2023, 1, 1)
    
    # Test rounding to 2 digits
    rounded = money.round(2)
    assert rounded.qty == Decimal("123.46")
    
    # Test rounding to -1 digits
    rounded = money.round(-1)
    assert rounded.qty == Decimal("120")
    
    # Test HALF_EVEN rounding
    money_half_even = Money.of(eur, Decimal("123.5"), date(2023, 1, 1))
    rounded = money_half_even.round()
    assert rounded.qty == Decimal("124")  # 123.5 rounds to 124 (HALF_EVEN)
    
    money_half_even2 = Money.of(eur, Decimal("122.5"), date(2023, 1, 1))
    rounded = money_half_even2.round()
    assert rounded.qty == Decimal("122")  # 122.5 rounds to 122 (HALF_EVEN)
    
    # Test with undefined money
    undefined_money = Money.na()
    rounded = undefined_money.round(2)
    assert rounded is undefined_money
    assert rounded.undefined
    
    # Test with __round__ method
    rounded = round(money, 1)
    assert rounded.qty == Decimal("123.5")
    
    # Test __round__ with None (should use default 0)
    rounded = round(money, None)
    assert isinstance(rounded, int)
    assert rounded == 123
    
    # Test __round__ without ndigits
    rounded = round(money)
    assert isinstance(rounded, int)
    assert rounded == 123
    
    # Test rounding with different currencies
    usd = Currencies["USD"]
    usd_money = Money.of(usd, Decimal("99.999"), date(2023, 1, 1))
    rounded = usd_money.round(2)
    assert rounded.qty == Decimal("100.00")
    assert rounded.ccy == usd
    
    # Test rounding with exact value
    exact_money = Money.of(eur, Decimal("100.000"), date(2023, 1, 1))
    rounded = exact_money.round(2)
    assert rounded.qty == Decimal("100.00")
    
    # Test rounding with negative quantity
    negative_money = Money.of(eur, Decimal("-123.456"), date(2023, 1, 1))
    rounded = negative_money.round(2)
    assert rounded.qty == Decimal("-123.46")


# LLM-generated content at query #6
#--------------------------

```python
def test_Price_with_qty():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currencies
    
    # Test with defined price
    usd = Currencies["USD"]
    original_price = Price.of(usd, Decimal("100.50"), date(2023, 1, 1))
    new_qty = Decimal("200.75")
    result = original_price.with_qty(new_qty)
    
    assert result.defined
    assert result.ccy == usd
    assert result.qty == new_qty
    assert result.dov == date(2023, 1, 1)
    assert result is not original_price
    
    # Test with undefined price
    undefined_price = Price.na()
    result = undefined_price.with_qty(Decimal("100"))
    
    assert result.undefined
    assert result is undefined_price
    
    # Test with zero quantity
    result = original_price.with_qty(Decimal("0"))
    assert result.qty == Decimal("0")
    
    # Test with negative quantity
    result = original_price.with_qty(Decimal("-50.25"))
    assert result.qty == Decimal("-50.25")
    
    # Test with large quantity
    large_qty = Decimal("999999999.99")
    result = original_price.with_qty(large_qty)
    assert result.qty == large_qty
    
    # Test that other attributes remain unchanged
    eur = Currencies["EUR"]
    price_with_eur = Price.of(eur, Decimal("50"), date(2023, 12, 31))
    result = price_with_eur.with_qty(Decimal("75"))
    
    assert result.ccy == eur
    assert result.dov == date(2023, 12, 31)
    assert result.qty == Decimal("75")


# LLM-generated content at query #7
#--------------------------

```python
def test_SomePrice___floordiv__():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currencies
    
    # Test normal floor division
    price = Price.of(Currencies["USD"], Decimal("10"), date(2023, 1, 1))
    result = price.__floordiv__(Decimal("3"))
    assert result.ccy_or_none().code == "USD"
    assert result.qty_or_none() == Decimal("3")
    assert result.dov_or_none() == date(2023, 1, 1)
    
    # Test floor division by 1
    price = Price.of(Currencies["EUR"], Decimal("7"), date(2023, 1, 1))
    result = price.__floordiv__(Decimal("1"))
    assert result.ccy_or_none().code == "EUR"
    assert result.qty_or_none() == Decimal("7")
    
    # Test floor division resulting in zero
    price = Price.of(Currencies["JPY"], Decimal("2"), date(2023, 1, 1))
    result = price.__floordiv__(Decimal("5"))
    assert result.qty_or_none() == Decimal("0")
    
    # Test floor division by zero returns NoPrice
    price = Price.of(Currencies["USD"], Decimal("10"), date(2023, 1, 1))
    result = price.__floordiv__(Decimal("0"))
    assert result.undefined
    assert Price.is_none(result)
    
    # Test floor division with negative divisor
    price = Price.of(Currencies["USD"], Decimal("10"), date(2023, 1, 1))
    result = price.__floordiv__(Decimal("-3"))
    assert result.qty_or_none() == Decimal("-4")
    
    # Test floor division with negative dividend
    price = Price.of(Currencies["USD"], Decimal("-10"), date(2023, 1, 1))
    result = price.__floordiv__(Decimal("3"))
    assert result.qty_or_none() == Decimal("-4")
    
    # Test floor division with both negative
    price = Price.of(Currencies["USD"], Decimal("-10"), date(2023, 1, 1))
    result = price.__floordiv__(Decimal("-3"))
    assert result.qty_or_none() == Decimal("3")
    
    # Test floor division with integer
    price = Price.of(Currencies["USD"], Decimal("10"), date(2023, 1, 1))
    result = price.__floordiv__(3)
    assert result.qty_or_none() == Decimal("3")
    
    # Test floor division with float
    price = Price.of(Currencies["USD"], Decimal("10"), date(2023, 1, 1))
    result = price.__floordiv__(3.0)
    assert result.qty_or_none() == Decimal("3")
    
    # Test floor division preserves date
    price = Price.of(Currencies["USD"], Decimal("10"), date(2023, 12, 31))
    result = price.__floordiv__(Decimal("3"))
    assert result.dov_or_none() == date(2023, 12, 31)
    
    # Test floor division preserves currency
    price = Price.of(Currencies["GBP"], Decimal("10"), date(2023, 1, 1))
    result = price.__floordiv__(Decimal("3"))
    assert result.ccy_or_none().code == "GBP"


# LLM-generated content at query #8
#--------------------------

```python
def test_Money_with_qty():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currencies
    
    # Test with defined money
    usd = Currencies["USD"]
    original_money = Money.of(usd, Decimal("100.50"), date(2023, 1, 1))
    new_qty = Decimal("200.75")
    result = original_money.with_qty(new_qty)
    
    assert result.defined
    assert result.ccy == usd
    assert result.qty == usd.quantize(new_qty)
    assert result.dov == date(2023, 1, 1)
    assert result is not original_money
    
    # Test with undefined money
    undefined_money = Money.na()
    result = undefined_money.with_qty(Decimal("100"))
    assert result is undefined_money
    assert result.undefined
    
    # Test with zero quantity
    result = original_money.with_qty(Decimal("0"))
    assert result.defined
    assert result.qty == usd.quantize(Decimal("0"))
    
    # Test with negative quantity
    result = original_money.with_qty(Decimal("-50.25"))
    assert result.defined
    assert result.qty == usd.quantize(Decimal("-50.25"))
    
    # Test with very large quantity
    large_qty = Decimal("999999999.99")
    result = original_money.with_qty(large_qty)
    assert result.defined
    assert result.qty == usd.quantize(large_qty)
    
    # Test that quantity is properly quantized to currency precision
    result = original_money.with_qty(Decimal("123.456789"))
    assert result.defined
    assert result.qty == usd.quantize(Decimal("123.456789"))


# LLM-generated content at query #9
#--------------------------

```python
def test_SomeMoney_convert():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currencies
    from pypara.money import SomeMoney, NoMoney
    from pypara.fx import FXRateService, FXRateLookupError
    from pypara.exceptions import ProgrammingError

    # Mock FXRateService for testing
    class MockFXRateService:
        def __init__(self):
            self.rates = {}
        
        def query(self, ccy_from, ccy_to, asof, strict):
            return self.rates.get((ccy_from, ccy_to, asof))
    
    # Save original default service
    original_default = FXRateService.default
    
    # Test 1: Successful conversion with explicit asof date
    mock_service = MockFXRateService()
    FXRateService.default = mock_service
    
    usd = Currencies["USD"]
    eur = Currencies["EUR"]
    test_date = date(2023, 1, 1)
    conversion_date = date(2023, 1, 2)
    
    # Setup conversion rate
    from pypara.fx import FXRate
    mock_service.rates[(usd, eur, conversion_date)] = FXRate(usd, eur, conversion_date, Decimal('0.85'))
    
    money = SomeMoney(usd, Decimal('100'), test_date)
    result = money.convert(eur, asof=conversion_date)
    
    assert isinstance(result, SomeMoney)
    assert result.ccy == eur
    assert result.qty == Decimal('85.00')  # 100 * 0.85 = 85.00
    assert result.dov == conversion_date
    
    # Test 2: Successful conversion using dov as asof date
    mock_service.rates[(usd, eur, test_date)] = FXRate(usd, eur, test_date, Decimal('0.90'))
    
    result = money.convert(eur)  # No asof specified, should use dov
    assert result.ccy == eur
    assert result.qty == Decimal('90.00')  # 100 * 0.90 = 90.00
    assert result.dov == test_date
    
    # Test 3: Conversion with strict=False when rate not found
    result = money.convert(Currencies["JPY"], strict=False)
    assert result is NoMoney
    
    # Test 4: Conversion with strict=True when rate not found (should raise)
    try:
        money.convert(Currencies["JPY"], strict=True)
        assert False, "Should have raised FXRateLookupError"
    except FXRateLookupError:
        pass  # Expected
    
    # Test 5: Test with no default FX rate service
    FXRateService.default = None
    try:
        money.convert(eur)
        assert False, "Should have raised ProgrammingError"
    except ProgrammingError as e:
        assert "Did you implement and set the default FX rate service" in str(e)
    
    # Test 6: Test proper quantization
    mock_service = MockFXRateService()
    FXRateService.default = mock_service
    mock_service.rates[(usd, eur, test_date)] = FXRate(usd, eur, test_date, Decimal('0.833333'))
    
    money = SomeMoney(usd, Decimal('100'), test_date)
    result = money.convert(eur)
    assert result.qty == Decimal('83.33')  # Should be quantized to 2 decimals for EUR
    
    # Test 7: Test conversion with different date comparison
    money1 = SomeMoney(usd, Decimal('100'), date(2023, 1, 1))
    money2 = SomeMoney(usd, Decimal('50'), date(2023, 1, 15))
    
    mock_service.rates[(usd, eur, date(2023, 1, 15))] = FXRate(usd, eur, date(2023, 1, 15), Decimal('0.88'))
    
    result = money1.convert(eur, asof=date(2023, 1, 15))
    assert result.dov == date(2023, 1, 15)
    
    # Restore original default service
    FXRateService.default = original_default


# LLM-generated content at query #10
#--------------------------

```python
def test_SomePrice___ge__():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currencies
    from pypara.money import SomePrice, NoPrice
    from pypara.currencies import IncompatibleCurrencyError

    # Test with defined prices in same currency
    usd = Currencies["USD"]
    price1 = SomePrice(usd, Decimal("10"), date(2023, 1, 1))
    price2 = SomePrice(usd, Decimal("5"), date(2023, 1, 1))
    price3 = SomePrice(usd, Decimal("10"), date(2023, 1, 1))
    price4 = SomePrice(usd, Decimal("15"), date(2023, 1, 1))

    assert price1 >= price2
    assert price1 >= price3
    assert not (price1 >= price4)
    assert price4 >= price1

    # Test with different currencies should raise IncompatibleCurrencyError
    eur = Currencies["EUR"]
    price_eur = SomePrice(eur, Decimal("10"), date(2023, 1, 1))
    
    try:
        price1 >= price_eur
        assert False, "Should have raised IncompatibleCurrencyError"
    except IncompatibleCurrencyError as e:
        assert e.ccy1 == usd
        assert e.ccy2 == eur
        assert e.operation == ">= comparison"

    # Test with undefined price (NoPrice)
    # SomePrice should be greater than NoPrice
    assert price1 >= NoPrice
    assert not (NoPrice >= price1)

    # Test with equal quantities
    price_same1 = SomePrice(usd, Decimal("7.5"), date(2023, 1, 1))
    price_same2 = SomePrice(usd, Decimal("7.5"), date(2023, 1, 1))
    assert price_same1 >= price_same2
    assert price_same2 >= price_same1

    # Test with fractional quantities
    price_frac1 = SomePrice(usd, Decimal("3.14159"), date(2023, 1, 1))
    price_frac2 = SomePrice(usd, Decimal("3.14"), date(2023, 1, 1))
    price_frac3 = SomePrice(usd, Decimal("3.1416"), date(2023, 1, 1))
    
    assert price_frac1 >= price_frac2
    assert not (price_frac1 >= price_frac3)
    assert price_frac3 >= price_frac1

    # Test with zero and negative quantities
    price_zero = SomePrice(usd, Decimal("0"), date(2023, 1, 1))
    price_neg = SomePrice(usd, Decimal("-5"), date(2023, 1, 1))
    price_pos = SomePrice(usd, Decimal("5"), date(2023, 1, 1))
    
    assert price_zero >= price_neg
    assert not (price_neg >= price_zero)
    assert price_pos >= price_zero
    assert price_pos >= price_neg
    assert not (price_neg >= price_pos)

    # Test with different dates (date shouldn't affect comparison)
    price_early = SomePrice(usd, Decimal("10"), date(2022, 1, 1))
    price_late = SomePrice(usd, Decimal("10"), date(2023, 1, 1))
    assert price_early >= price_late
    assert price_late >= price_early


# LLM-generated content at query #11
#--------------------------

```python
def test_Money_is_equal():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currencies
    
    # Test equal defined money objects
    usd1 = Money.of(Currencies["USD"], Decimal("100.50"), date(2023, 1, 1))
    usd2 = Money.of(Currencies["USD"], Decimal("100.50"), date(2023, 1, 1))
    assert usd1 == usd2
    assert not (usd1 != usd2)
    
    # Test different currencies
    eur = Money.of(Currencies["EUR"], Decimal("100.50"), date(2023, 1, 1))
    assert usd1 != eur
    assert not (usd1 == eur)
    
    # Test different quantities
    usd3 = Money.of(Currencies["USD"], Decimal("200.50"), date(2023, 1, 1))
    assert usd1 != usd3
    assert not (usd1 == usd3)
    
    # Test different dates
    usd4 = Money.of(Currencies["USD"], Decimal("100.50"), date(2023, 1, 2))
    assert usd1 != usd4
    assert not (usd1 == usd4)
    
    # Test undefined money objects
    none1 = Money.na()
    none2 = Money.na()
    assert none1 == none2
    assert not (none1 != none2)
    
    # Test defined vs undefined
    assert usd1 != none1
    assert not (usd1 == none1)
    
    # Test equality with non-Money objects
    assert usd1 != "not money"
    assert usd1 != 123
    assert usd1 != None
    
    # Test with factory method creating undefined money
    none3 = Money.of(None, Decimal("100.50"), date(2023, 1, 1))
    none4 = Money.of(Currencies["USD"], None, date(2023, 1, 1))
    none5 = Money.of(Currencies["USD"], Decimal("100.50"), None)
    assert none3 == none1
    assert none4 == none1
    assert none5 == none1
    
    # Test that undefined money objects are equal regardless of parameters
    assert none3 == none4
    assert none4 == none5
    assert none5 == none3


# LLM-generated content at query #12
#--------------------------

```python
def test_Money___abs__():
    from decimal import Decimal
    from pypara.currencies import Currencies
    from pypara.money import Money, SomeMoney, NoMoney
    from datetime import date

    # Test with defined positive money
    usd = Currencies["USD"]
    positive_money = Money.of(usd, Decimal("100.50"), date(2023, 1, 1))
    abs_result = abs(positive_money)
    assert isinstance(abs_result, SomeMoney)
    assert abs_result.ccy == usd
    assert abs_result.qty == Decimal("100.50")
    assert abs_result.dov == date(2023, 1, 1)

    # Test with defined negative money
    negative_money = Money.of(usd, Decimal("-100.50"), date(2023, 1, 1))
    abs_result = abs(negative_money)
    assert isinstance(abs_result, SomeMoney)
    assert abs_result.ccy == usd
    assert abs_result.qty == Decimal("100.50")
    assert abs_result.dov == date(2023, 1, 1)

    # Test with defined zero money
    zero_money = Money.of(usd, Decimal("0"), date(2023, 1, 1))
    abs_result = abs(zero_money)
    assert isinstance(abs_result, SomeMoney)
    assert abs_result.ccy == usd
    assert abs_result.qty == Decimal("0.00")
    assert abs_result.dov == date(2023, 1, 1)

    # Test with undefined money
    undefined_money = Money.na()
    abs_result = abs(undefined_money)
    assert abs_result is undefined_money
    assert isinstance(abs_result, NoMoney)

    # Test that __abs__ returns same as abs() method
    money = Money.of(usd, Decimal("-50.25"), date(2023, 1, 1))
    assert abs(money) == money.abs()
    assert abs(money).qty == Decimal("50.25")

    # Test with different currency
    eur = Currencies["EUR"]
    eur_money = Money.of(eur, Decimal("-75.30"), date(2023, 1, 1))
    abs_result = abs(eur_money)
    assert abs_result.ccy == eur
    assert abs_result.qty == Decimal("75.30")


# LLM-generated content at query #13
#--------------------------

```python
def test_Money_ccy_or():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currencies
    
    # Test with defined money
    usd_money = Money.of(Currencies["USD"], Decimal("100.50"), date(2023, 1, 1))
    eur_currency = Currencies["EUR"]
    
    # Should return the money's own currency (USD)
    assert usd_money.ccy_or(eur_currency).code == "USD"
    assert usd_money.ccy_or(eur_currency) is Currencies["USD"]
    
    # Test with undefined money (NoMoney)
    undefined_money = Money.na()
    
    # Should return the default currency (EUR)
    assert undefined_money.ccy_or(eur_currency).code == "EUR"
    assert undefined_money.ccy_or(eur_currency) is eur_currency
    
    # Test with another undefined case using Money.of with None values
    undefined_money2 = Money.of(Currencies["USD"], None, date(2023, 1, 1))
    assert undefined_money2.ccy_or(eur_currency).code == "EUR"
    
    undefined_money3 = Money.of(None, Decimal("100.50"), date(2023, 1, 1))
    assert undefined_money3.ccy_or(eur_currency).code == "EUR"
    
    undefined_money4 = Money.of(Currencies["USD"], Decimal("100.50"), None)
    assert undefined_money4.ccy_or(eur_currency).code == "EUR"
    
    # Test with different default currencies
    jpy_currency = Currencies["JPY"]
    assert undefined_money.ccy_or(jpy_currency).code == "JPY"
    
    # Test that default currency is returned exactly as provided
    gbp_currency = Currencies["GBP"]
    result = undefined_money.ccy_or(gbp_currency)
    assert result.code == "GBP"
    assert result is gbp_currency


# LLM-generated content at query #14
#--------------------------

```python
def test_SomePrice___lt__():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currencies
    from pypara.money import SomePrice, NoPrice
    from pypara.currencies.exceptions import IncompatibleCurrencyError

    # Test with same currency
    price1 = SomePrice(Currencies["USD"], Decimal("10"), date(2023, 1, 1))
    price2 = SomePrice(Currencies["USD"], Decimal("20"), date(2023, 1, 1))
    assert price1 < price2
    assert not (price2 < price1)

    # Test with equal quantities
    price3 = SomePrice(Currencies["USD"], Decimal("10"), date(2023, 1, 1))
    assert not (price1 < price3)
    assert not (price3 < price1)

    # Test with different currencies (should raise IncompatibleCurrencyError)
    price4 = SomePrice(Currencies["EUR"], Decimal("5"), date(2023, 1, 1))
    try:
        _ = price1 < price4
        assert False, "Should have raised IncompatibleCurrencyError"
    except IncompatibleCurrencyError as e:
        assert e.ccy1 == Currencies["USD"]
        assert e.ccy2 == Currencies["EUR"]
        assert e.operation == "< comparison"

    # Test with NoPrice (should return False)
    assert not (price1 < NoPrice)
    assert not (NoPrice < price1)

    # Test with non-SomePrice object (should return False)
    assert not (price1 < "not a price")
    assert not (price1 < 123)

    # Test with negative quantities
    price5 = SomePrice(Currencies["USD"], Decimal("-5"), date(2023, 1, 1))
    price6 = SomePrice(Currencies["USD"], Decimal("5"), date(2023, 1, 1))
    assert price5 < price6
    assert not (price6 < price5)

    # Test with zero quantity
    price7 = SomePrice(Currencies["USD"], Decimal("0"), date(2023, 1, 1))
    assert price5 < price7
    assert not (price6 < price7)
    assert not (price7 < price7)

    # Test with different dates (date should not affect comparison)
    price8 = SomePrice(Currencies["USD"], Decimal("10"), date(2022, 12, 31))
    price9 = SomePrice(Currencies["USD"], Decimal("15"), date(2023, 1, 1))
    assert price8 < price9
    assert not (price9 < price8)


# LLM-generated content at query #15
#--------------------------

```python
def test_Money_scalar_subtract():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currencies
    
    # Test with defined money object
    usd = Currencies["USD"]
    money = Money.of(usd, Decimal("100.50"), date(2023, 1, 1))
    
    # Test subtraction with integer
    result = money.scalar_subtract(50)
    assert result.defined
    assert result.ccy == usd
    assert result.qty == Decimal("50.50")
    assert result.dov == date(2023, 1, 1)
    
    # Test subtraction with float
    result = money.scalar_subtract(25.25)
    assert result.defined
    assert result.ccy == usd
    assert result.qty == Decimal("75.25")
    assert result.dov == date(2023, 1, 1)
    
    # Test subtraction with Decimal
    result = money.scalar_subtract(Decimal("10.10"))
    assert result.defined
    assert result.ccy == usd
    assert result.qty == Decimal("90.40")
    assert result.dov == date(2023, 1, 1)
    
    # Test subtraction with negative number
    result = money.scalar_subtract(-20)
    assert result.defined
    assert result.ccy == usd
    assert result.qty == Decimal("120.50")
    assert result.dov == date(2023, 1, 1)
    
    # Test subtraction resulting in zero
    result = money.scalar_subtract(Decimal("100.50"))
    assert result.defined
    assert result.ccy == usd
    assert result.qty == Decimal("0.00")
    assert result.dov == date(2023, 1, 1)
    
    # Test subtraction resulting in negative
    result = money.scalar_subtract(200)
    assert result.defined
    assert result.ccy == usd
    assert result.qty == Decimal("-99.50")
    assert result.dov == date(2023, 1, 1)
    
    # Test with undefined money object
    undefined_money = Money.na()
    result = undefined_money.scalar_subtract(100)
    assert result is undefined_money
    assert result.undefined
    
    # Test with None quantity (should also be undefined)
    none_qty_money = Money.of(usd, None, date(2023, 1, 1))
    result = none_qty_money.scalar_subtract(50)
    assert result.undefined
    
    # Test with None currency (should also be undefined)
    none_ccy_money = Money.of(None, Decimal("100.50"), date(2023, 1, 1))
    result = none_ccy_money.scalar_subtract(50)
    assert result.undefined
    
    # Test with None date (should also be undefined)
    none_dov_money = Money.of(usd, Decimal("100.50"), None)
    result = none_dov_money.scalar_subtract(50)
    assert result.undefined
    
    # Test that original money object is not modified
    original_qty = money.qty
    _ = money.scalar_subtract(10)
    assert money.qty == original_qty


# LLM-generated content at query #16
#--------------------------

```python
def test_Price_round():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currencies
    from pypara.monetary import Price
    
    # Test with defined price
    price = Price.of(Currencies["USD"], Decimal("123.456"), date(2023, 1, 1))
    rounded = price.round(2)
    assert rounded.qty == Decimal("123.46")
    assert rounded.ccy == Currencies["USD"]
    assert rounded.dov == date(2023, 1, 1)
    
    # Test with zero ndigits
    rounded_zero = price.round(0)
    assert rounded_zero.qty == Decimal("123")
    
    # Test with negative ndigits
    rounded_neg = price.round(-2)
    assert rounded_neg.qty == Decimal("100")
    
    # Test HALF_EVEN rounding
    price_half_even = Price.of(Currencies["EUR"], Decimal("123.455"), date(2023, 1, 1))
    rounded_half_even = price_half_even.round(2)
    assert rounded_half_even.qty == Decimal("123.46")
    
    # Test with undefined price
    undefined_price = Price.na()
    rounded_undefined = undefined_price.round(2)
    assert rounded_undefined is undefined_price
    assert not rounded_undefined.defined
    
    # Test with __round__ method (default ndigits)
    rounded_default = round(price)
    assert rounded_default == 123
    
    # Test with __round__ method with explicit ndigits
    rounded_explicit = round(price, 1)
    assert rounded_explicit.qty == Decimal("123.5")
    
    # Test with __round__ method with None ndigits
    rounded_none = round(price, None)
    assert rounded_none == 123
    
    # Test edge case: exact half with even digit
    price_exact_half = Price.of(Currencies["JPY"], Decimal("123.5"), date(2023, 1, 1))
    rounded_exact = price_exact_half.round(0)
    assert rounded_exact.qty == Decimal("124")
    
    # Test with very small number
    price_small = Price.of(Currencies["GBP"], Decimal("0.0001"), date(2023, 1, 1))
    rounded_small = price_small.round(3)
    assert rounded_small.qty == Decimal("0.000")
    
    # Test with large number
    price_large = Price.of(Currencies["CAD"], Decimal("999999.999999"), date(2023, 1, 1))
    rounded_large = price_large.round(3)
    assert rounded_large.qty == Decimal("1000000.000")


# LLM-generated content at query #17
#--------------------------

```python
def test_Money___float__():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currencies
    
    # Test with defined money
    usd = Currencies["USD"]
    money = Money.of(usd, Decimal("123.45"), date(2023, 1, 1))
    assert float(money) == 123.45
    
    # Test with negative amount
    money_neg = Money.of(usd, Decimal("-67.89"), date(2023, 1, 1))
    assert float(money_neg) == -67.89
    
    # Test with zero
    money_zero = Money.of(usd, Decimal("0"), date(2023, 1, 1))
    assert float(money_zero) == 0.0
    
    # Test with large number
    money_large = Money.of(usd, Decimal("999999.99"), date(2023, 1, 1))
    assert float(money_large) == 999999.99
    
    # Test with undefined money - should raise MonetaryOperationException
    undefined_money = Money.na()
    try:
        float(undefined_money)
        assert False, "Should have raised MonetaryOperationException"
    except MonetaryOperationException:
        pass
    
    # Test with different currency
    eur = Currencies["EUR"]
    money_eur = Money.of(eur, Decimal("50.50"), date(2023, 1, 1))
    assert float(money_eur) == 50.50
    
    # Test with exact decimal representation
    money_exact = Money.of(usd, Decimal("100.0"), date(2023, 1, 1))
    assert float(money_exact) == 100.0


# LLM-generated content at query #18
#--------------------------

```python
def test_Money___int__():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currencies
    
    # Test with defined money object
    usd = Currencies["USD"]
    defined_money = Money.of(usd, Decimal("123.45"), date(2023, 1, 1))
    assert int(defined_money) == 123
    
    # Test with defined money object having exact integer quantity
    defined_money_exact = Money.of(usd, Decimal("100.00"), date(2023, 1, 1))
    assert int(defined_money_exact) == 100
    
    # Test with defined money object having negative quantity
    defined_money_negative = Money.of(usd, Decimal("-456.78"), date(2023, 1, 1))
    assert int(defined_money_negative) == -456
    
    # Test with defined money object having zero quantity
    defined_money_zero = Money.of(usd, Decimal("0.00"), date(2023, 1, 1))
    assert int(defined_money_zero) == 0
    
    # Test with defined money object having very large quantity
    defined_money_large = Money.of(usd, Decimal("999999.99"), date(2023, 1, 1))
    assert int(defined_money_large) == 999999
    
    # Test with undefined money object - should raise MonetaryOperationException
    undefined_money = Money.na()
    try:
        int(undefined_money)
        assert False, "Should have raised MonetaryOperationException"
    except MonetaryOperationException:
        pass
    
    # Test with money object created with None quantity
    none_qty_money = Money.of(usd, None, date(2023, 1, 1))
    try:
        int(none_qty_money)
        assert False, "Should have raised MonetaryOperationException"
    except MonetaryOperationException:
        pass
    
    # Test with money object created with None currency
    none_ccy_money = Money.of(None, Decimal("123.45"), date(2023, 1, 1))
    try:
        int(none_ccy_money)
        assert False, "Should have raised MonetaryOperationException"
    except MonetaryOperationException:
        pass
    
    # Test with money object created with None date
    none_dov_money = Money.of(usd, Decimal("123.45"), None)
    try:
        int(none_dov_money)
        assert False, "Should have raised MonetaryOperationException"
    except MonetaryOperationException:
        pass


# LLM-generated content at query #19
#--------------------------

```python
def test_Money_ccy_or_none():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currencies
    
    # Test with defined money
    usd_money = Money.of(Currencies["USD"], Decimal("100.50"), date(2023, 1, 1))
    assert usd_money.ccy_or_none() == Currencies["USD"]
    assert usd_money.ccy_or_none().code == "USD"
    
    # Test with undefined money (ccy is None)
    undefined_money1 = Money.of(None, Decimal("100.50"), date(2023, 1, 1))
    assert undefined_money1.ccy_or_none() is None
    
    # Test with undefined money (qty is None)
    undefined_money2 = Money.of(Currencies["EUR"], None, date(2023, 1, 1))
    assert undefined_money2.ccy_or_none() is None
    
    # Test with undefined money (dov is None)
    undefined_money3 = Money.of(Currencies["GBP"], Decimal("100.50"), None)
    assert undefined_money3.ccy_or_none() is None
    
    # Test with completely undefined money
    undefined_money4 = Money.of(None, None, None)
    assert undefined_money4.ccy_or_none() is None
    
    # Test with Money.na()
    na_money = Money.na()
    assert na_money.ccy_or_none() is None
    
    # Test with different currencies
    eur_money = Money.of(Currencies["EUR"], Decimal("200.75"), date(2023, 2, 1))
    assert eur_money.ccy_or_none() == Currencies["EUR"]
    assert eur_money.ccy_or_none().code == "EUR"
    
    jpy_money = Money.of(Currencies["JPY"], Decimal("1000"), date(2023, 3, 1))
    assert jpy_money.ccy_or_none() == Currencies["JPY"]
    assert jpy_money.ccy_or_none().code == "JPY"


# LLM-generated content at query #20
#--------------------------

```python
def test_Price_scalar_subtract():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currencies
    
    # Test with defined price
    price = Price.of(Currencies["USD"], Decimal("10.5"), date(2023, 1, 1))
    result = price.scalar_subtract(Decimal("2.5"))
    assert result.defined
    assert result.ccy == Currencies["USD"]
    assert result.qty == Decimal("8.0")
    assert result.dov == date(2023, 1, 1)
    
    # Test with integer scalar
    result = price.scalar_subtract(2)
    assert result.qty == Decimal("8.5")
    
    # Test with float scalar
    result = price.scalar_subtract(2.5)
    assert result.qty == Decimal("8.0")
    
    # Test with negative scalar
    result = price.scalar_subtract(Decimal("-3.5"))
    assert result.qty == Decimal("14.0")
    
    # Test with zero scalar
    result = price.scalar_subtract(0)
    assert result.qty == Decimal("10.5")
    
    # Test with undefined price
    undefined_price = Price.na()
    result = undefined_price.scalar_subtract(Decimal("5.0"))
    assert result is undefined_price
    assert result.undefined
    
    # Test with None price (created via factory)
    none_price = Price.of(None, Decimal("10.5"), date(2023, 1, 1))
    result = none_price.scalar_subtract(Decimal("2.5"))
    assert result.undefined
    
    # Test with None quantity
    none_qty_price = Price.of(Currencies["USD"], None, date(2023, 1, 1))
    result = none_qty_price.scalar_subtract(Decimal("2.5"))
    assert result.undefined
    
    # Test with None date
    none_date_price = Price.of(Currencies["USD"], Decimal("10.5"), None)
    result = none_date_price.scalar_subtract(Decimal("2.5"))
    assert result.undefined
    
    # Test that original price is not modified
    original_qty = price.qty
    price.scalar_subtract(Decimal("1.0"))
    assert price.qty == original_qty


# LLM-generated content at query #21
#--------------------------

```python
def test_Price_negative():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currencies
    
    # Test with defined price
    price = Price.of(Currencies["USD"], Decimal("100.50"), date(2023, 1, 1))
    negated = price.negative()
    
    assert negated.defined
    assert negated.ccy == Currencies["USD"]
    assert negated.qty == Decimal("-100.50")
    assert negated.dov == date(2023, 1, 1)
    
    # Test with zero price
    zero_price = Price.of(Currencies["EUR"], Decimal("0"), date(2023, 1, 1))
    negated_zero = zero_price.negative()
    
    assert negated_zero.defined
    assert negated_zero.ccy == Currencies["EUR"]
    assert negated_zero.qty == Decimal("0")
    assert negated_zero.dov == date(2023, 1, 1)
    
    # Test with negative price
    negative_price = Price.of(Currencies["GBP"], Decimal("-50.25"), date(2023, 1, 1))
    negated_negative = negative_price.negative()
    
    assert negated_negative.defined
    assert negated_negative.ccy == Currencies["GBP"]
    assert negated_negative.qty == Decimal("50.25")
    assert negated_negative.dov == date(2023, 1, 1)
    
    # Test with undefined price (should return itself)
    undefined_price = Price.na()
    negated_undefined = undefined_price.negative()
    
    assert negated_undefined.undefined
    assert negated_undefined is undefined_price
    
    # Test with partially undefined price (created via of method)
    partial_price = Price.of(None, Decimal("100"), date(2023, 1, 1))
    negated_partial = partial_price.negative()
    
    assert negated_partial.undefined
    assert negated_partial is partial_price
    
    # Test using __neg__ operator
    price2 = Price.of(Currencies["JPY"], Decimal("1000"), date(2023, 1, 1))
    negated2 = -price2
    
    assert negated2.defined
    assert negated2.ccy == Currencies["JPY"]
    assert negated2.qty == Decimal("-1000")
    assert negated2.dov == date(2023, 1, 1)


# LLM-generated content at query #22
#--------------------------

```python
def test_Money_subtract():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currencies
    from pypara.money import Money, SomeMoney, NoMoney
    from pypara.exceptions import IncompatibleCurrencyError

    # Test 1: Subtract two defined money objects with same currency
    usd = Currencies["USD"]
    m1 = Money.of(usd, Decimal("100.50"), date(2023, 1, 1))
    m2 = Money.of(usd, Decimal("50.25"), date(2023, 1, 1))
    result = m1.subtract(m2)
    assert isinstance(result, SomeMoney)
    assert result.ccy == usd
    assert result.qty == Decimal("50.25")
    assert result.dov == date(2023, 1, 1)

    # Test 2: Subtract with different currencies raises error
    eur = Currencies["EUR"]
    m3 = Money.of(eur, Decimal("100"), date(2023, 1, 1))
    try:
        m1.subtract(m3)
        assert False, "Should have raised IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass

    # Test 3: Subtract undefined from defined returns defined
    m4 = Money.na()
    result = m1.subtract(m4)
    assert result is m1

    # Test 4: Subtract defined from undefined returns defined (negated)
    result = m4.subtract(m1)
    assert isinstance(result, SomeMoney)
    assert result.ccy == usd
    assert result.qty == Decimal("-100.50")
    assert result.dov == date(2023, 1, 1)

    # Test 5: Subtract undefined from undefined returns undefined
    result = m4.subtract(Money.na())
    assert result is m4

    # Test 6: Subtract with different dates carries forward date
    m5 = Money.of(usd, Decimal("30"), date(2023, 1, 2))
    result = m1.subtract(m5)
    assert result.dov == date(2023, 1, 2)

    # Test 7: Subtract zero
    m6 = Money.of(usd, Decimal("0"), date(2023, 1, 1))
    result = m1.subtract(m6)
    assert result.qty == Decimal("100.50")

    # Test 8: Subtract negative amount
    m7 = Money.of(usd, Decimal("-20.75"), date(2023, 1, 1))
    result = m1.subtract(m7)
    assert result.qty == Decimal("121.25")

    # Test 9: Ensure quantity is properly quantized
    m8 = Money.of(usd, Decimal("33.333"), date(2023, 1, 1))
    m9 = Money.of(usd, Decimal("11.111"), date(2023, 1, 1))
    result = m8.subtract(m9)
    assert result.qty == Decimal("22.22")  # USD has 2 decimal places

    # Test 10: Using __sub__ operator
    result = m1 - m2
    assert result.qty == Decimal("50.25")


# LLM-generated content at query #23
#--------------------------

```python
def test_Price_with_dov():
    from datetime import date
    from decimal import Decimal
    from pypara.currencies import Currencies
    
    # Test with defined price
    usd = Currencies["USD"]
    original_price = Price.of(usd, Decimal("100.50"), date(2023, 1, 1))
    new_dov = date(2023, 12, 31)
    result = original_price.with_dov(new_dov)
    
    assert result.defined
    assert result.ccy == usd
    assert result.qty == Decimal("100.50")
    assert result.dov == new_dov
    
    # Test with undefined price
    undefined_price = Price.na()
    result = undefined_price.with_dov(date(2023, 12, 31))
    assert result is undefined_price
    assert result.undefined
    
    # Test with partially defined price (should be undefined)
    partial_price = Price.of(None, Decimal("100.50"), None)
    result = partial_price.with_dov(date(2023, 12, 31))
    assert result.undefined
    
    # Test that original price is not modified
    assert original_price.dov == date(2023, 1, 1)
    
    # Test with same dov returns same object or equal object
    same_dov_result = original_price.with_dov(date(2023, 1, 1))
    assert same_dov_result.ccy == usd
    assert same_dov_result.qty == Decimal("100.50")
    assert same_dov_result.dov == date(2023, 1, 1)


# LLM-generated content at query #24
#--------------------------

```python
def test_Price_dov_or():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currencies
    
    # Test with defined price
    some_price = Price.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    default_date = date(2001, 1, 1)
    result = some_price.dov_or(default_date)
    assert result == date(2019, 1, 1)
    
    # Test with undefined price (None ccy)
    none_price_ccy = Price.of(None, Decimal('1'), date(2019, 1, 1))
    result = none_price_ccy.dov_or(default_date)
    assert result == default_date
    
    # Test with undefined price (None qty)
    none_price_qty = Price.of(Currencies["USD"], None, date(2019, 1, 1))
    result = none_price_qty.dov_or(default_date)
    assert result == default_date
    
    # Test with undefined price (None dov)
    none_price_dov = Price.of(Currencies["USD"], Decimal('1'), None)
    result = none_price_dov.dov_or(default_date)
    assert result == default_date
    
    # Test with completely undefined price
    none_price_all = Price.of(None, None, None)
    result = none_price_all.dov_or(default_date)
    assert result == default_date
    
    # Test with Price.na()
    na_price = Price.na()
    result = na_price.dov_or(default_date)
    assert result == default_date
    
    # Test with different default date
    alt_default = date(2020, 12, 31)
    result = none_price_ccy.dov_or(alt_default)
    assert result == alt_default
    
    # Test that defined price returns its own dov even with different default
    result = some_price.dov_or(alt_default)
    assert result == date(2019, 1, 1)
    assert result != alt_default


# LLM-generated content at query #25
#--------------------------

```python
def test_Price_divide():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currencies
    from pypara.monetary import Price
    
    # Test 1: Division of defined price by positive number
    price1 = Price.of(Currencies["USD"], Decimal("10.0"), date(2023, 1, 1))
    result = price1.divide(Decimal("2"))
    assert Price.is_some(result)
    assert result.ccy == Currencies["USD"]
    assert result.qty == Decimal("5.0")
    assert result.dov == date(2023, 1, 1)
    
    # Test 2: Division of defined price by negative number
    price2 = Price.of(Currencies["EUR"], Decimal("15.0"), date(2023, 1, 2))
    result = price2.divide(Decimal("-3"))
    assert Price.is_some(result)
    assert result.ccy == Currencies["EUR"]
    assert result.qty == Decimal("-5.0")
    assert result.dov == date(2023, 1, 2)
    
    # Test 3: Division by zero yields undefined price
    price3 = Price.of(Currencies["GBP"], Decimal("100.0"), date(2023, 1, 3))
    result = price3.divide(Decimal("0"))
    assert Price.is_none(result)
    assert result.undefined
    
    # Test 4: Division by zero yields undefined price (float zero)
    result = price3.divide(0.0)
    assert Price.is_none(result)
    assert result.undefined
    
    # Test 5: Division by zero yields undefined price (int zero)
    result = price3.divide(0)
    assert Price.is_none(result)
    assert result.undefined
    
    # Test 6: Undefined price divided by any number returns undefined
    undefined_price = Price.na()
    result = undefined_price.divide(Decimal("5"))
    assert Price.is_none(result)
    assert result.undefined
    
    # Test 7: Division with float divisor
    price4 = Price.of(Currencies["JPY"], Decimal("1000"), date(2023, 1, 4))
    result = price4.divide(2.5)
    assert Price.is_some(result)
    assert result.ccy == Currencies["JPY"]
    assert result.qty == Decimal("400.0")
    assert result.dov == date(2023, 1, 4)
    
    # Test 8: Division with integer divisor
    price5 = Price.of(Currencies["CAD"], Decimal("21"), date(2023, 1, 5))
    result = price5.divide(3)
    assert Price.is_some(result)
    assert result.ccy == Currencies["CAD"]
    assert result.qty == Decimal("7")
    assert result.dov == date(2023, 1, 5)
    
    # Test 9: Division resulting in decimal
    price6 = Price.of(Currencies["AUD"], Decimal("10"), date(2023, 1, 6))
    result = price6.divide(Decimal("3"))
    assert Price.is_some(result)
    assert result.ccy == Currencies["AUD"]
    assert result.qty == Decimal("3.333333333333333333333333333")
    assert result.dov == date(2023, 1, 6)
    
    # Test 10: Division by one returns same quantity
    price7 = Price.of(Currencies["CHF"], Decimal("7.5"), date(2023, 1, 7))
    result = price7.divide(Decimal("1"))
    assert Price.is_some(result)
    assert result.ccy == Currencies["CHF"]
    assert result.qty == Decimal("7.5")
    assert result.dov == date(2023, 1, 7)


# LLM-generated content at query #26
#--------------------------

```python
def test_Money_positive():
    from decimal import Decimal
    from pypara.currencies import Currencies
    from datetime import date

    # Test with defined money
    usd_money = Money.of(Currencies["USD"], Decimal("100.50"), date(2023, 1, 1))
    positive_result = usd_money.positive()
    assert positive_result is usd_money  # Should return itself for defined money
    assert positive_result.ccy.code == "USD"
    assert positive_result.qty == Decimal("100.50")
    assert positive_result.dov == date(2023, 1, 1)

    # Test with negative defined money
    negative_money = Money.of(Currencies["EUR"], Decimal("-50.75"), date(2023, 1, 2))
    positive_negative = negative_money.positive()
    assert positive_negative is negative_money  # Should return itself for defined money
    assert positive_negative.ccy.code == "EUR"
    assert positive_negative.qty == Decimal("-50.75")

    # Test with undefined money
    undefined_money = Money.na()
    positive_undefined = undefined_money.positive()
    assert positive_undefined is undefined_money  # Should return itself for undefined money
    assert positive_undefined.undefined is True

    # Test with zero money
    zero_money = Money.of(Currencies["GBP"], Decimal("0"), date(2023, 1, 3))
    positive_zero = zero_money.positive()
    assert positive_zero is zero_money
    assert positive_zero.qty == Decimal("0.00")


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_SomeMoney_qty_or_zero():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currencies
    
    # Test with defined money
    somemoney = Money.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    assert somemoney.qty_or_zero() == Decimal('1.00')
    
    # Test with defined money with zero quantity
    zeromoney = Money.of(Currencies["USD"], Decimal('0'), date(2019, 1, 1))
    assert zeromoney.qty_or_zero() == Decimal('0.00')
    
    # Test with defined money with negative quantity
    negmoney = Money.of(Currencies["USD"], Decimal('-5'), date(2019, 1, 1))
    assert negmoney.qty_or_zero() == Decimal('-5.00')
    
    # Test with defined money with large quantity
    bigmoney = Money.of(Currencies["USD"], Decimal('123456.789'), date(2019, 1, 1))
    assert bigmoney.qty_or_zero() == Decimal('123456.79')
    
    # Test with JPY currency (0 decimal places)
    jpymoney = Money.of(Currencies["JPY"], Decimal('1500'), date(2019, 1, 1))
    assert jpymoney.qty_or_zero() == Decimal('1500')
    
    # Test with BHD currency (3 decimal places)
    bhdinoney = Money.of(Currencies["BHD"], Decimal('1.2345'), date(2019, 1, 1))
    assert bhdinoney.qty_or_zero() == Decimal('1.235')


# LLM-generated content at query #2
#--------------------------

```python
def test_Money_convert():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currencies
    from pypara.money import Money, SomeMoney, NoMoney
    from pypara.fx import FXRateLookupError

    # Test 1: Convert defined money to same currency (no conversion needed)
    usd_money = Money.of(Currencies["USD"], Decimal("100"), date(2023, 1, 1))
    converted = usd_money.convert(Currencies["USD"], date(2023, 1, 1))
    assert converted == usd_money
    assert isinstance(converted, SomeMoney)
    assert converted.ccy == Currencies["USD"]
    assert converted.qty == Decimal("100.00")
    assert converted.dov == date(2023, 1, 1)

    # Test 2: Convert defined money to different currency with valid FX rate
    # Mock FX rate service would be needed here
    # This test assumes FX rate service is properly mocked
    eur_money = Money.of(Currencies["EUR"], Decimal("100"), date(2023, 1, 1))
    # Note: Actual conversion would depend on FX rate service implementation

    # Test 3: Convert with specific asof date
    usd_money = Money.of(Currencies["USD"], Decimal("100"), date(2023, 1, 1))
    # Should use the provided asof date for FX rate lookup
    converted = usd_money.convert(Currencies["EUR"], date(2023, 1, 15))
    assert converted.dov == date(2023, 1, 15)

    # Test 4: Convert without asof date (should use money's dov)
    usd_money = Money.of(Currencies["USD"], Decimal("100"), date(2023, 1, 1))
    converted = usd_money.convert(Currencies["EUR"])
    assert converted.dov == date(2023, 1, 1)

    # Test 5: Convert undefined money (should return undefined money)
    undefined_money = Money.na()
    converted = undefined_money.convert(Currencies["USD"])
    assert isinstance(converted, NoMoney)
    assert converted.undefined

    # Test 6: Convert with strict=True when FX rate not found
    usd_money = Money.of(Currencies["USD"], Decimal("100"), date(2023, 1, 1))
    # This should raise FXRateLookupError when no FX rate is available
    # Note: This test would require mocking FX rate service to raise error

    # Test 7: Convert with strict=False when FX rate not found
    # Should return undefined money when FX rate not found and strict=False
    usd_money = Money.of(Currencies["USD"], Decimal("100"), date(2023, 1, 1))
    # Note: This test would require mocking FX rate service to return None

    # Test 8: Convert and verify quantity calculation
    # Assuming 1 USD = 0.85 EUR on date(2023, 1, 1)
    usd_money = Money.of(Currencies["USD"], Decimal("100"), date(2023, 1, 1))
    # Mock FX rate service to return 0.85
    # converted = usd_money.convert(Currencies["EUR"], date(2023, 1, 1))
    # assert converted.qty == Decimal("85.00")

    # Test 9: Convert with different date than money's dov
    usd_money = Money.of(Currencies["USD"], Decimal("100"), date(2023, 1, 1))
    converted = usd_money.convert(Currencies["EUR"], date(2023, 6, 1))
    assert converted.dov == date(2023, 6, 1)
    # Different FX rate should be used for different date

    # Test 10: Convert to currency with different decimal places
    jpy_money = Money.of(Currencies["JPY"], Decimal("1000"), date(2023, 1, 1))
    # JPY has 0 decimal places
    converted = jpy_money.convert(Currencies["USD"], date(2023, 1, 1))
    assert converted.ccy == Currencies["USD"]
    # USD has 2 decimal places, so quantity should be appropriately formatted

    # Test 11: Convert zero amount
    usd_money = Money.of(Currencies["USD"], Decimal("0"), date(2023, 1, 1))
    converted = usd_money.convert(Currencies["EUR"], date(2023, 1, 1))
    assert converted.qty == Decimal("0.00")
    assert converted.ccy == Currencies["EUR"]

    # Test 12: Convert negative amount
    usd_money = Money.of(Currencies["USD"], Decimal("-100"), date(2023, 1, 1))
    converted = usd_money.convert(Currencies["EUR"], date(2023, 1, 1))
    # Should maintain negative sign after conversion
    # assert converted.qty < Decimal("0")

    # Test 13: Convert with None asof date
    usd_money = Money.of(Currencies["USD"], Decimal("100"), date(2023, 1, 1))
    converted = usd_money.convert(Currencies["EUR"], None)
    # Should use money's dov when asof is None
    assert converted.dov == date(2023, 1, 1)

    # Test 14: Verify that conversion creates new SomeMoney instance
    usd_money = Money.of(Currencies["USD"], Decimal("100"), date(2023, 1, 1))
    converted = usd_money.convert(Currencies["EUR"], date(2023, 1, 1))
    assert converted is not usd_money
    assert isinstance(converted, SomeMoney)

    # Test 15: Convert with future date (should handle appropriately)
    usd_money = Money.of(Currencies["USD"], Decimal("100"), date(2023, 1, 1))
    converted = usd_money.convert(Currencies["EUR"], date(2024, 1, 1))
    assert converted.dov == date(2024, 1, 1)


# LLM-generated content at query #3
#--------------------------

```python
def test_Money___pos__():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currencies
    
    # Test with defined money
    usd = Currencies["USD"]
    defined_money = Money.of(usd, Decimal("100.50"), date(2023, 1, 1))
    result = +defined_money
    
    assert result.defined
    assert result.ccy == usd
    assert result.qty == Decimal("100.50")
    assert result.dov == date(2023, 1, 1)
    
    # Test with undefined money
    undefined_money = Money.na()
    result = +undefined_money
    
    assert result.undefined
    assert result is undefined_money
    
    # Test that positive returns same object for defined money
    assert (+defined_money) is defined_money
    
    # Test with negative quantity
    negative_money = Money.of(usd, Decimal("-50.25"), date(2023, 1, 1))
    result = +negative_money
    
    assert result.defined
    assert result.ccy == usd
    assert result.qty == Decimal("-50.25")
    assert result.dov == date(2023, 1, 1)
    
    # Test with zero quantity
    zero_money = Money.of(usd, Decimal("0"), date(2023, 1, 1))
    result = +zero_money
    
    assert result.defined
    assert result.ccy == usd
    assert result.qty == Decimal("0.00")
    assert result.dov == date(2023, 1, 1)


# LLM-generated content at query #4
#--------------------------

```python
def test_SomeMoney_qty_map():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currencies
    
    # Test with defined money (SomeMoney)
    somemoney = Money.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    result = somemoney.qty_map(lambda x: x + Decimal('1'), lambda: Decimal('42'))
    assert result == Decimal('2.00')
    
    # Test with different mapping function
    result = somemoney.qty_map(lambda x: x * Decimal('2'), lambda: Decimal('42'))
    assert result == Decimal('2.00')
    
    # Test with mapping that returns different type
    result = somemoney.qty_map(lambda x: str(x), lambda: "default")
    assert result == "1.00"
    
    # Test with undefined money (NoMoney)
    nonemoney = Money.of(None, Decimal('1'), None)
    result = nonemoney.qty_map(lambda x: x + Decimal('1'), lambda: Decimal('42'))
    assert result == Decimal('42')
    
    # Test with undefined money and different default combinator
    result = nonemoney.qty_map(lambda x: x + Decimal('1'), lambda: "error")
    assert result == "error"
    
    # Test with undefined money and None default
    result = nonemoney.qty_map(lambda x: x + Decimal('1'), lambda: None)
    assert result is None


# LLM-generated content at query #5
#--------------------------

```python
def test_Money_with_dov():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currencies
    
    # Test with defined money
    ccy = Currencies["USD"]
    qty = Decimal("100.50")
    original_dov = date(2023, 1, 1)
    new_dov = date(2023, 12, 31)
    
    money = Money.of(ccy, qty, original_dov)
    result = money.with_dov(new_dov)
    
    assert result.defined
    assert result.ccy == ccy
    assert result.qty == qty
    assert result.dov == new_dov
    
    # Test with undefined money
    undefined_money = Money.na()
    result = undefined_money.with_dov(new_dov)
    
    assert result is undefined_money
    assert result.undefined
    
    # Test with None dov
    result = money.with_dov(None)
    assert result.defined
    assert result.dov is None
    
    # Test with same dov returns same object
    result = money.with_dov(original_dov)
    assert result is money


# LLM-generated content at query #6
#--------------------------

```python
def test_Price_round():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currencies
    
    # Test with defined price
    price = Price.of(Currencies["USD"], Decimal("123.456"), date(2023, 1, 1))
    rounded = price.round(2)
    assert rounded.qty == Decimal("123.46")
    assert rounded.ccy == Currencies["USD"]
    assert rounded.dov == date(2023, 1, 1)
    
    # Test with zero digits
    rounded_zero = price.round(0)
    assert rounded_zero.qty == Decimal("123")
    
    # Test with negative digits
    rounded_neg = price.round(-2)
    assert rounded_neg.qty == Decimal("100")
    
    # Test with HALF_EVEN rounding
    price_half = Price.of(Currencies["EUR"], Decimal("123.455"), date(2023, 1, 1))
    rounded_half = price_half.round(2)
    assert rounded_half.qty == Decimal("123.46")
    
    # Test with undefined price (should return itself)
    undefined_price = Price.na()
    result = undefined_price.round(2)
    assert result is undefined_price
    assert result.undefined
    
    # Test with exact decimal
    price_exact = Price.of(Currencies["JPY"], Decimal("1000"), date(2023, 1, 1))
    rounded_exact = price_exact.round(2)
    assert rounded_exact.qty == Decimal("1000.00")
    
    # Test with __round__ method
    rounded_dunder = round(price, 1)
    assert rounded_dunder.qty == Decimal("123.5")
    
    # Test __round__ with None (should use default 0)
    rounded_none = round(price, None)
    assert isinstance(rounded_none, int)
    assert rounded_none == 123


# LLM-generated content at query #7
#--------------------------

```python
def test_Money_ccy_or_none():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currencies
    
    # Test with defined money
    usd_money = Money.of(Currencies["USD"], Decimal("100.50"), date(2023, 1, 1))
    assert usd_money.ccy_or_none() is not None
    assert usd_money.ccy_or_none().code == "USD"
    
    # Test with undefined money (None currency)
    undefined_money1 = Money.of(None, Decimal("100.50"), date(2023, 1, 1))
    assert undefined_money1.ccy_or_none() is None
    
    # Test with undefined money (None quantity)
    undefined_money2 = Money.of(Currencies["EUR"], None, date(2023, 1, 1))
    assert undefined_money2.ccy_or_none() is None
    
    # Test with undefined money (None date)
    undefined_money3 = Money.of(Currencies["GBP"], Decimal("100.50"), None)
    assert undefined_money3.ccy_or_none() is None
    
    # Test with completely undefined money
    undefined_money4 = Money.of(None, None, None)
    assert undefined_money4.ccy_or_none() is None
    
    # Test with na() factory method
    na_money = Money.na()
    assert na_money.ccy_or_none() is None
    
    # Test with different currency
    eur_money = Money.of(Currencies["EUR"], Decimal("200.75"), date(2023, 1, 2))
    assert eur_money.ccy_or_none() is not None
    assert eur_money.ccy_or_none().code == "EUR"


# LLM-generated content at query #8
#--------------------------

```python
def test_Money_lte():
    # Test with defined money objects with same currency
    m1 = Money.of(Currencies["USD"], Decimal("100.50"), Date(2023, 1, 1))
    m2 = Money.of(Currencies["USD"], Decimal("200.75"), Date(2023, 1, 1))
    m3 = Money.of(Currencies["USD"], Decimal("100.50"), Date(2023, 1, 1))
    
    assert m1.lte(m2) is True
    assert m2.lte(m1) is False
    assert m1.lte(m3) is True
    
    # Test with defined money objects with different currencies (should raise error)
    m4 = Money.of(Currencies["EUR"], Decimal("100.50"), Date(2023, 1, 1))
    
    try:
        m1.lte(m4)
        assert False, "Should have raised IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass
    
    # Test with undefined money objects
    undefined1 = Money.na()
    undefined2 = Money.na()
    
    # Undefined money objects are always less than or equal to other
    assert undefined1.lte(m1) is True
    assert undefined1.lte(undefined2) is True
    
    # Defined money objects are NOT less than or equal to undefined
    assert m1.lte(undefined1) is False
    
    # Test edge cases with zero and negative values
    m5 = Money.of(Currencies["USD"], Decimal("0"), Date(2023, 1, 1))
    m6 = Money.of(Currencies["USD"], Decimal("-50.25"), Date(2023, 1, 1))
    
    assert m6.lte(m5) is True
    assert m5.lte(m6) is False
    assert m5.lte(m5) is True
    assert m6.lte(m6) is True
    
    # Test with same quantity but different dates (should compare only quantity)
    m7 = Money.of(Currencies["USD"], Decimal("100.50"), Date(2023, 1, 1))
    m8 = Money.of(Currencies["USD"], Decimal("100.50"), Date(2023, 12, 31))
    
    assert m7.lte(m8) is True
    assert m8.lte(m7) is True


# LLM-generated content at query #9
#--------------------------

```python
def test_Price_gt():
    from decimal import Decimal
    from datetime import date
    
    # Test 1: Defined price > undefined price
    price1 = Price.of(Currencies["USD"], Decimal("100"), date(2023, 1, 1))
    price2 = Price.na()
    assert price1.gt(price2) is True
    
    # Test 2: Undefined price > defined price
    price1 = Price.na()
    price2 = Price.of(Currencies["USD"], Decimal("100"), date(2023, 1, 1))
    assert price1.gt(price2) is False
    
    # Test 3: Undefined price > undefined price
    price1 = Price.na()
    price2 = Price.na()
    assert price1.gt(price2) is False
    
    # Test 4: Defined price > defined price with same currency (greater)
    price1 = Price.of(Currencies["USD"], Decimal("200"), date(2023, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal("100"), date(2023, 1, 1))
    assert price1.gt(price2) is True
    
    # Test 5: Defined price > defined price with same currency (equal)
    price1 = Price.of(Currencies["USD"], Decimal("100"), date(2023, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal("100"), date(2023, 1, 1))
    assert price1.gt(price2) is False
    
    # Test 6: Defined price > defined price with same currency (less)
    price1 = Price.of(Currencies["USD"], Decimal("50"), date(2023, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal("100"), date(2023, 1, 1))
    assert price1.gt(price2) is False
    
    # Test 7: Defined price > defined price with different currencies (should raise error)
    price1 = Price.of(Currencies["USD"], Decimal("100"), date(2023, 1, 1))
    price2 = Price.of(Currencies["EUR"], Decimal("100"), date(2023, 1, 1))
    try:
        price1.gt(price2)
        assert False, "Should have raised IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass
    
    # Test 8: Using __gt__ operator
    price1 = Price.of(Currencies["USD"], Decimal("200"), date(2023, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal("100"), date(2023, 1, 1))
    assert (price1 > price2) is True
    
    # Test 9: Using __gt__ operator with undefined price
    price1 = Price.of(Currencies["USD"], Decimal("100"), date(2023, 1, 1))
    price2 = Price.na()
    assert (price1 > price2) is True
    
    # Test 10: Different dates should not affect comparison
    price1 = Price.of(Currencies["USD"], Decimal("200"), date(2023, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal("100"), date(2023, 12, 31))
    assert price1.gt(price2) is True


# LLM-generated content at query #10
#--------------------------

```python
def test_Money_qty_or():
    from decimal import Decimal
    from pypara.currencies import Currencies
    from pypara.money import Money
    from datetime import date

    # Test with defined money
    usd = Currencies["USD"]
    defined_money = Money.of(usd, Decimal("100.50"), date(2023, 1, 1))
    assert defined_money.qty_or(Decimal("0")) == Decimal("100.50")
    assert defined_money.qty_or(Decimal("999")) == Decimal("100.50")

    # Test with undefined money (NoMoney)
    undefined_money = Money.na()
    assert undefined_money.qty_or(Decimal("0")) == Decimal("0")
    assert undefined_money.qty_or(Decimal("999")) == Decimal("999")

    # Test with partially defined money (should be undefined)
    partial_money = Money.of(None, Decimal("100.50"), date(2023, 1, 1))
    assert partial_money.qty_or(Decimal("0")) == Decimal("0")

    # Test with different default types
    assert undefined_money.qty_or(Decimal("42.75")) == Decimal("42.75")
    assert undefined_money.qty_or(Decimal("-100")) == Decimal("-100")


# LLM-generated content at query #11
#--------------------------

```python
def test_Price___float__():
    from decimal import Decimal
    from pypara.currencies import Currencies
    from pypara.monetary import Price, Date
    import pytest

    # Test with defined price
    price = Price.of(Currencies["USD"], Decimal("123.45"), Date(2023, 1, 1))
    result = float(price)
    assert result == 123.45
    assert isinstance(result, float)

    # Test with defined price with more decimal places
    price = Price.of(Currencies["EUR"], Decimal("99.999"), Date(2023, 1, 1))
    result = float(price)
    assert result == 99.999
    assert isinstance(result, float)

    # Test with defined price with integer quantity
    price = Price.of(Currencies["JPY"], Decimal("1000"), Date(2023, 1, 1))
    result = float(price)
    assert result == 1000.0
    assert isinstance(result, float)

    # Test with undefined price - should raise MonetaryOperationException
    undefined_price = Price.na()
    with pytest.raises(MonetaryOperationException):
        float(undefined_price)

    # Test with price created with None values
    undefined_price2 = Price.of(None, Decimal("100"), Date(2023, 1, 1))
    with pytest.raises(MonetaryOperationException):
        float(undefined_price2)

    undefined_price3 = Price.of(Currencies["USD"], None, Date(2023, 1, 1))
    with pytest.raises(MonetaryOperationException):
        float(undefined_price3)

    undefined_price4 = Price.of(Currencies["USD"], Decimal("100"), None)
    with pytest.raises(MonetaryOperationException):
        float(undefined_price4)

    # Test with very large decimal
    price = Price.of(Currencies["USD"], Decimal("999999999.99"), Date(2023, 1, 1))
    result = float(price)
    assert result == 999999999.99
    assert isinstance(result, float)

    # Test with negative quantity
    price = Price.of(Currencies["USD"], Decimal("-50.75"), Date(2023, 1, 1))
    result = float(price)
    assert result == -50.75
    assert isinstance(result, float)

    # Test with zero quantity
    price = Price.of(Currencies["USD"], Decimal("0"), Date(2023, 1, 1))
    result = float(price)
    assert result == 0.0
    assert isinstance(result, float)

    # Test that __float__ method is called when using float() constructor
    price = Price.of(Currencies["GBP"], Decimal("42.5"), Date(2023, 1, 1))
    result = price.__float__()
    assert result == 42.5
    assert isinstance(result, float)


# LLM-generated content at query #12
#--------------------------

```python
def test_Price_dimap():
    # Test with defined price
    some_price = Price.of(Currencies["USD"], Decimal("1"), Date(2019, 1, 1))
    
    def extract_ccy_code(x):
        return x.ccy.code
    
    def fallback():
        return "EUR"
    
    result = some_price.dimap(extract_ccy_code, fallback)
    assert result == "USD"
    
    # Test with undefined price
    none_price = Price.of(None, Decimal("1"), None)
    
    result = none_price.dimap(extract_ccy_code, fallback)
    assert result == "EUR"
    
    # Test with different mapping functions
    def extract_qty(x):
        return x.qty
    
    def fallback_qty():
        return Decimal("0")
    
    result = some_price.dimap(extract_qty, fallback_qty)
    assert result == Decimal("1")
    
    result = none_price.dimap(extract_qty, fallback_qty)
    assert result == Decimal("0")
    
    # Test with complex mapping
    def complex_map(x):
        return f"{x.ccy.code}:{x.qty}:{x.dov}"
    
    def complex_fallback():
        return "UNDEFINED"
    
    result = some_price.dimap(complex_map, complex_fallback)
    assert result == "USD:1:2019-01-01"
    
    result = none_price.dimap(complex_map, complex_fallback)
    assert result == "UNDEFINED"


# LLM-generated content at query #13
#--------------------------

```python
def test_Price_qty_map():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currencies
    
    # Test with defined price
    some_price = Price.of(Currencies["USD"], Decimal('10'), date(2023, 1, 1))
    
    # Test with function that modifies quantity
    result = some_price.qty_map(lambda x: x * Decimal('2'), lambda: Decimal('0'))
    assert result == Decimal('20')
    
    # Test with function that returns different type
    result = some_price.qty_map(lambda x: str(x), lambda: "default")
    assert result == "10"
    
    # Test with undefined price (NoPrice)
    none_price = Price.na()
    
    # Should return default value from combinator
    result = none_price.qty_map(lambda x: x + Decimal('1'), lambda: Decimal('99'))
    assert result == Decimal('99')
    
    # Test with combinator returning different type
    result = none_price.qty_map(lambda x: x, lambda: "undefined")
    assert result == "undefined"
    
    # Test with partially defined price (should be NoPrice)
    partial_price = Price.of(None, Decimal('5'), date(2023, 1, 1))
    result = partial_price.qty_map(lambda x: x * Decimal('3'), lambda: Decimal('100'))
    assert result == Decimal('100')
    
    # Test with another currency
    eur_price = Price.of(Currencies["EUR"], Decimal('15.5'), date(2023, 1, 1))
    result = eur_price.qty_map(lambda x: x.quantize(Decimal('0.01')), lambda: Decimal('0'))
    assert result == Decimal('15.50')
    
    # Test with complex transformation
    complex_price = Price.of(Currencies["GBP"], Decimal('7.3'), date(2023, 1, 1))
    result = complex_price.qty_map(
        lambda x: (x * Decimal('100')).to_integral_value(),
        lambda: Decimal('-1')
    )
    assert result == Decimal('730')


# LLM-generated content at query #14
#--------------------------

```python
def test_Price_dov_or():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currencies
    
    # Test with defined price
    usd = Currencies["USD"]
    test_date = date(2023, 1, 1)
    price = Price.of(usd, Decimal("100.50"), test_date)
    
    default_date = date(2000, 1, 1)
    result = price.dov_or(default_date)
    assert result == test_date
    
    # Test with undefined price (None currency)
    undefined_price = Price.of(None, Decimal("100.50"), test_date)
    result = undefined_price.dov_or(default_date)
    assert result == default_date
    
    # Test with undefined price (None quantity)
    undefined_price2 = Price.of(usd, None, test_date)
    result = undefined_price2.dov_or(default_date)
    assert result == default_date
    
    # Test with undefined price (None date)
    undefined_price3 = Price.of(usd, Decimal("100.50"), None)
    result = undefined_price3.dov_or(default_date)
    assert result == default_date
    
    # Test with completely undefined price
    none_price = Price.na()
    result = none_price.dov_or(default_date)
    assert result == default_date
    
    # Test with different default date
    alt_default = date(1999, 12, 31)
    result = price.dov_or(alt_default)
    assert result == test_date  # Should still return original date
    
    # Test that method returns Date type
    assert isinstance(result, date)


# LLM-generated content at query #15
#--------------------------

```python
def test_Money_abs():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currencies
    
    # Test with defined positive money
    usd = Currencies["USD"]
    positive_money = Money.of(usd, Decimal("100.50"), date(2023, 1, 1))
    abs_result = positive_money.abs()
    assert abs_result.defined
    assert abs_result.ccy == usd
    assert abs_result.qty == Decimal("100.50")
    assert abs_result.dov == date(2023, 1, 1)
    
    # Test with defined negative money
    negative_money = Money.of(usd, Decimal("-100.50"), date(2023, 1, 1))
    abs_result = negative_money.abs()
    assert abs_result.defined
    assert abs_result.ccy == usd
    assert abs_result.qty == Decimal("100.50")
    assert abs_result.dov == date(2023, 1, 1)
    
    # Test with defined zero money
    zero_money = Money.of(usd, Decimal("0"), date(2023, 1, 1))
    abs_result = zero_money.abs()
    assert abs_result.defined
    assert abs_result.ccy == usd
    assert abs_result.qty == Decimal("0.00")
    assert abs_result.dov == date(2023, 1, 1)
    
    # Test with undefined money
    undefined_money = Money.na()
    abs_result = undefined_money.abs()
    assert abs_result is undefined_money
    assert not abs_result.defined
    assert abs_result.undefined
    
    # Test with partially undefined money (should be undefined according to Money.of)
    partial_money = Money.of(None, Decimal("100.50"), date(2023, 1, 1))
    abs_result = partial_money.abs()
    assert abs_result is partial_money
    assert not abs_result.defined
    assert abs_result.undefined
    
    # Test __abs__ dunder method
    money = Money.of(usd, Decimal("-50.25"), date(2023, 1, 1))
    dunder_result = abs(money)
    assert dunder_result.defined
    assert dunder_result.ccy == usd
    assert dunder_result.qty == Decimal("50.25")
    assert dunder_result.dov == date(2023, 1, 1)
    
    # Test __abs__ with undefined money
    undefined_dunder = abs(Money.na())
    assert undefined_dunder.undefined


# LLM-generated content at query #16
#--------------------------

```python
def test_Money_gte():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currencies

    # Test 1: Both defined, same currency, quantity greater
    m1 = Money.of(Currencies["USD"], Decimal("10.00"), date(2023, 1, 1))
    m2 = Money.of(Currencies["USD"], Decimal("5.00"), date(2023, 1, 1))
    assert m1.gte(m2) is True
    assert m2.gte(m1) is False

    # Test 2: Both defined, same currency, equal quantities
    m3 = Money.of(Currencies["EUR"], Decimal("7.50"), date(2023, 1, 1))
    m4 = Money.of(Currencies["EUR"], Decimal("7.50"), date(2023, 1, 1))
    assert m3.gte(m4) is True
    assert m4.gte(m3) is True

    # Test 3: Both defined, different currencies - should raise IncompatibleCurrencyError
    m5 = Money.of(Currencies["USD"], Decimal("10.00"), date(2023, 1, 1))
    m6 = Money.of(Currencies["EUR"], Decimal("10.00"), date(2023, 1, 1))
    try:
        m5.gte(m6)
        assert False, "Should have raised IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass

    # Test 4: Self undefined, other defined - should return False
    m7 = Money.na()
    m8 = Money.of(Currencies["USD"], Decimal("10.00"), date(2023, 1, 1))
    assert m7.gte(m8) is False
    assert m8.gte(m7) is True

    # Test 5: Both undefined - should return True
    m9 = Money.na()
    m10 = Money.na()
    assert m9.gte(m10) is True
    assert m10.gte(m9) is True

    # Test 6: Self defined, other undefined - should return True
    m11 = Money.of(Currencies["GBP"], Decimal("15.00"), date(2023, 1, 1))
    m12 = Money.na()
    assert m11.gte(m12) is True
    assert m12.gte(m11) is False

    # Test 7: Edge case with zero quantity
    m13 = Money.of(Currencies["JPY"], Decimal("0.00"), date(2023, 1, 1))
    m14 = Money.of(Currencies["JPY"], Decimal("0.00"), date(2023, 1, 1))
    assert m13.gte(m14) is True
    assert m14.gte(m13) is True

    # Test 8: Negative quantity comparison
    m15 = Money.of(Currencies["USD"], Decimal("-5.00"), date(2023, 1, 1))
    m16 = Money.of(Currencies["USD"], Decimal("5.00"), date(2023, 1, 1))
    assert m15.gte(m16) is False
    assert m16.gte(m15) is True

    # Test 9: Negative quantity comparison (both negative)
    m17 = Money.of(Currencies["USD"], Decimal("-3.00"), date(2023, 1, 1))
    m18 = Money.of(Currencies["USD"], Decimal("-5.00"), date(2023, 1, 1))
    assert m17.gte(m18) is True
    assert m18.gte(m17) is False


# LLM-generated content at query #17
#--------------------------

```python
def test_SomeMoney_with_dov():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currencies
    
    # Create a SomeMoney instance
    ccy = Currencies["USD"]
    qty = Decimal("100.50")
    original_dov = date(2023, 1, 1)
    money = SomeMoney(ccy, qty, original_dov)
    
    # Test with a new date
    new_dov = date(2023, 12, 31)
    result = money.with_dov(new_dov)
    
    # Verify the result is a new SomeMoney instance with updated dov
    assert isinstance(result, SomeMoney)
    assert result.ccy == ccy
    assert result.qty == qty
    assert result.dov == new_dov
    
    # Verify the original instance is unchanged
    assert money.dov == original_dov
    
    # Test with the same dov
    same_result = money.with_dov(original_dov)
    assert same_result.ccy == ccy
    assert same_result.qty == qty
    assert same_result.dov == original_dov
    
    # Test with an earlier date
    earlier_dov = date(2022, 6, 15)
    earlier_result = money.with_dov(earlier_dov)
    assert earlier_result.dov == earlier_dov


# LLM-generated content at query #18
#--------------------------

```python
def test_Money___pos__():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currencies
    
    # Test with defined money
    usd = Currencies["USD"]
    defined_money = Money.of(usd, Decimal("100.50"), date(2023, 1, 1))
    result = +defined_money
    
    assert result.defined
    assert result.ccy == usd
    assert result.qty == Decimal("100.50")
    assert result.dov == date(2023, 1, 1)
    
    # Test with negative defined money
    negative_money = Money.of(usd, Decimal("-50.25"), date(2023, 1, 1))
    result = +negative_money
    
    assert result.defined
    assert result.ccy == usd
    assert result.qty == Decimal("-50.25")
    assert result.dov == date(2023, 1, 1)
    
    # Test with zero defined money
    zero_money = Money.of(usd, Decimal("0"), date(2023, 1, 1))
    result = +zero_money
    
    assert result.defined
    assert result.ccy == usd
    assert result.qty == Decimal("0.00")
    assert result.dov == date(2023, 1, 1)
    
    # Test with undefined money
    undefined_money = Money.na()
    result = +undefined_money
    
    assert result.undefined
    assert result is undefined_money


# LLM-generated content at query #19
#--------------------------

```python
def test_Price_or_else():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currencies
    
    # Create test fixtures
    usd_price = Price.of(Currencies["USD"], Decimal("100"), date(2023, 1, 1))
    eur_price = Price.of(Currencies["EUR"], Decimal("200"), date(2023, 1, 2))
    undefined_price = Price.na()
    
    # Test 1: Defined price returns itself
    result = usd_price.or_else(lambda: eur_price)
    assert result is usd_price
    assert result.ccy.code == "USD"
    assert result.qty == Decimal("100")
    assert result.dov == date(2023, 1, 1)
    
    # Test 2: Undefined price returns fallback
    result = undefined_price.or_else(lambda: eur_price)
    assert result is eur_price
    assert result.ccy.code == "EUR"
    assert result.qty == Decimal("200")
    assert result.dov == date(2023, 1, 2)
    
    # Test 3: Fallback is only evaluated when needed
    call_count = 0
    def counting_fallback():
        nonlocal call_count
        call_count += 1
        return eur_price
    
    # Defined price shouldn't call fallback
    result = usd_price.or_else(counting_fallback)
    assert call_count == 0
    assert result is usd_price
    
    # Undefined price should call fallback
    result = undefined_price.or_else(counting_fallback)
    assert call_count == 1
    assert result is eur_price
    
    # Test 4: Fallback can return undefined price
    result = undefined_price.or_else(lambda: Price.na())
    assert result.undefined is True
    assert Price.is_none(result)
    
    # Test 5: Fallback can create new price dynamically
    dynamic_price = undefined_price.or_else(
        lambda: Price.of(Currencies["JPY"], Decimal("5000"), date(2023, 1, 3))
    )
    assert dynamic_price.ccy.code == "JPY"
    assert dynamic_price.qty == Decimal("5000")
    assert dynamic_price.dov == date(2023, 1, 3)
    
    # Test 6: Method works with partially defined price (should be treated as undefined)
    partial_price = Price.of(Currencies["USD"], None, date(2023, 1, 1))
    result = partial_price.or_else(lambda: eur_price)
    assert result is eur_price
    assert result.ccy.code == "EUR"


# LLM-generated content at query #20
#--------------------------

```python
def test_Money_dov_or_none():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currencies
    
    # Test with defined money
    usd = Currencies["USD"]
    test_date = date(2023, 1, 1)
    defined_money = Money.of(usd, Decimal("100.50"), test_date)
    
    assert defined_money.dov_or_none() == test_date
    
    # Test with undefined money (None currency)
    undefined_money1 = Money.of(None, Decimal("100.50"), test_date)
    assert undefined_money1.dov_or_none() is None
    
    # Test with undefined money (None quantity)
    undefined_money2 = Money.of(usd, None, test_date)
    assert undefined_money2.dov_or_none() is None
    
    # Test with undefined money (None date)
    undefined_money3 = Money.of(usd, Decimal("100.50"), None)
    assert undefined_money3.dov_or_none() is None
    
    # Test with completely undefined money
    undefined_money4 = Money.of(None, None, None)
    assert undefined_money4.dov_or_none() is None
    
    # Test with Money.na()
    na_money = Money.na()
    assert na_money.dov_or_none() is None


# LLM-generated content at query #21
#--------------------------

```python
def test_Money___add__():
    # Test addition of two defined money objects with same currency
    m1 = Money.of(Currencies["USD"], Decimal("100.50"), Date(2023, 1, 1))
    m2 = Money.of(Currencies["USD"], Decimal("50.25"), Date(2023, 1, 1))
    result = m1 + m2
    assert result.defined
    assert result.ccy.code == "USD"
    assert result.qty == Decimal("150.75")
    assert result.dov == Date(2023, 1, 1)

    # Test addition with undefined money (left operand)
    m_undefined = Money.na()
    m_defined = Money.of(Currencies["EUR"], Decimal("200"), Date(2023, 1, 2))
    result = m_undefined + m_defined
    assert result is m_defined

    # Test addition with undefined money (right operand)
    result = m_defined + m_undefined
    assert result is m_defined

    # Test addition of two undefined money objects
    result = m_undefined + Money.na()
    assert result.undefined

    # Test incompatible currency error
    m_eur = Money.of(Currencies["EUR"], Decimal("100"), Date(2023, 1, 1))
    m_usd = Money.of(Currencies["USD"], Decimal("100"), Date(2023, 1, 1))
    with pytest.raises(IncompatibleCurrencyError):
        m_eur + m_usd

    # Test date carry forward (should use date from first operand)
    m1 = Money.of(Currencies["GBP"], Decimal("50"), Date(2023, 1, 1))
    m2 = Money.of(Currencies["GBP"], Decimal("30"), Date(2023, 1, 15))
    result = m1 + m2
    assert result.dov == Date(2023, 1, 1)

    # Test with different date on second operand
    m1 = Money.of(Currencies["JPY"], Decimal("1000"), Date(2023, 2, 1))
    m2 = Money.of(Currencies["JPY"], Decimal("500"), Date(2023, 2, 28))
    result = m1 + m2
    assert result.dov == Date(2023, 2, 1)

    # Test addition with zero quantity
    m1 = Money.of(Currencies["CAD"], Decimal("0"), Date(2023, 1, 1))
    m2 = Money.of(Currencies["CAD"], Decimal("100"), Date(2023, 1, 1))
    result = m1 + m2
    assert result.qty == Decimal("100")

    # Test addition with negative quantities
    m1 = Money.of(Currencies["USD"], Decimal("-50"), Date(2023, 1, 1))
    m2 = Money.of(Currencies["USD"], Decimal("100"), Date(2023, 1, 1))
    result = m1 + m2
    assert result.qty == Decimal("50")

    # Test commutative property
    m1 = Money.of(Currencies["EUR"], Decimal("25.75"), Date(2023, 1, 1))
    m2 = Money.of(Currencies["EUR"], Decimal("14.25"), Date(2023, 1, 1))
    assert (m1 + m2).qty == (m2 + m1).qty

    # Test that result is quantized according to currency
    m1 = Money.of(Currencies["USD"], Decimal("100.123"), Date(2023, 1, 1))
    m2 = Money.of(Currencies["USD"], Decimal("50.456"), Date(2023, 1, 1))
    result = m1 + m2
    assert result.qty == Decimal("150.58")  # USD has 2 decimal places

    # Test with very large quantities
    m1 = Money.of(Currencies["USD"], Decimal("999999999.99"), Date(2023, 1, 1))
    m2 = Money.of(Currencies["USD"], Decimal("0.01"), Date(2023, 1, 1))
    result = m1 + m2
    assert result.qty == Decimal("1000000000.00")


# LLM-generated content at query #22
#--------------------------

```python
def test_Money_positive():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currencies
    
    # Test with defined money
    usd = Currencies["USD"]
    defined_money = Money.of(usd, Decimal("100.50"), date(2023, 1, 1))
    result = defined_money.positive()
    
    # Positive should return the same monetary value for defined money
    assert result is defined_money
    assert result.ccy == usd
    assert result.qty == Decimal("100.50")
    assert result.dov == date(2023, 1, 1)
    
    # Test with negative defined money
    negative_money = Money.of(usd, Decimal("-50.25"), date(2023, 1, 1))
    negative_result = negative_money.positive()
    
    # Positive should still return itself for negative defined money
    assert negative_result is negative_money
    assert negative_result.qty == Decimal("-50.25")
    
    # Test with undefined money
    undefined_money = Money.na()
    undefined_result = undefined_money.positive()
    
    # Positive should return itself for undefined money
    assert undefined_result is undefined_money
    assert undefined_result.undefined
    
    # Test with zero money
    zero_money = Money.of(usd, Decimal("0"), date(2023, 1, 1))
    zero_result = zero_money.positive()
    
    # Positive should return itself for zero money
    assert zero_result is zero_money
    assert zero_result.qty == Decimal("0")


# LLM-generated content at query #23
#--------------------------

```python
def test_Money___add__():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currencies
    from pypara.money import Money, SomeMoney, NoMoney
    from pypara.exceptions import IncompatibleCurrencyError

    # Test addition of two defined money objects with same currency
    usd1 = Money.of(Currencies["USD"], Decimal("100.50"), date(2023, 1, 1))
    usd2 = Money.of(Currencies["USD"], Decimal("200.75"), date(2023, 1, 2))
    result = usd1 + usd2
    assert isinstance(result, SomeMoney)
    assert result.ccy == Currencies["USD"]
    assert result.qty == Decimal("301.25")
    assert result.dov == date(2023, 1, 2)

    # Test addition with undefined money (left operand)
    undefined = Money.na()
    result = undefined + usd1
    assert result is usd1

    # Test addition with undefined money (right operand)
    result = usd1 + undefined
    assert result is usd1

    # Test addition of two undefined money objects
    result = undefined + undefined
    assert result is undefined

    # Test addition with incompatible currencies
    eur = Money.of(Currencies["EUR"], Decimal("100"), date(2023, 1, 1))
    try:
        usd1 + eur
        assert False, "Should have raised IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass

    # Test addition with same currency but different dates
    usd3 = Money.of(Currencies["USD"], Decimal("50"), date(2023, 1, 3))
    result = usd1 + usd3
    assert result.dov == date(2023, 1, 3)

    # Test addition with zero quantity
    zero_usd = Money.of(Currencies["USD"], Decimal("0"), date(2023, 1, 1))
    result = usd1 + zero_usd
    assert result.qty == Decimal("100.50")
    assert result.dov == date(2023, 1, 1)

    # Test addition with negative quantity
    negative_usd = Money.of(Currencies["USD"], Decimal("-50.25"), date(2023, 1, 4))
    result = usd1 + negative_usd
    assert result.qty == Decimal("50.25")
    assert result.dov == date(2023, 1, 4)

    # Test that addition returns new instance
    assert usd1 + usd2 is not usd1
    assert usd1 + usd2 is not usd2

    # Test addition with large numbers
    large1 = Money.of(Currencies["USD"], Decimal("999999.99"), date(2023, 1, 1))
    large2 = Money.of(Currencies["USD"], Decimal("0.01"), date(2023, 1, 1))
    result = large1 + large2
    assert result.qty == Decimal("1000000.00")


# LLM-generated content at query #24
#--------------------------

```python
def test_Price_as_integer():
    # Test with defined price
    defined_price = Price.of(Currencies["USD"], Decimal("123.45"), Date(2023, 1, 1))
    assert defined_price.as_integer() == 123
    
    # Test with defined price that has decimal part
    defined_price2 = Price.of(Currencies["EUR"], Decimal("99.99"), Date(2023, 1, 1))
    assert defined_price2.as_integer() == 99
    
    # Test with defined price that is exactly integer
    defined_price3 = Price.of(Currencies["JPY"], Decimal("100"), Date(2023, 1, 1))
    assert defined_price3.as_integer() == 100
    
    # Test with defined price that rounds down
    defined_price4 = Price.of(Currencies["GBP"], Decimal("0.49"), Date(2023, 1, 1))
    assert defined_price4.as_integer() == 0
    
    # Test with defined price that rounds up
    defined_price5 = Price.of(Currencies["CAD"], Decimal("0.50"), Date(2023, 1, 1))
    assert defined_price5.as_integer() == 0
    
    # Test with undefined price (should raise MonetaryOperationException)
    undefined_price = Price.na()
    with pytest.raises(MonetaryOperationException):
        undefined_price.as_integer()
    
    # Test with partially undefined price (no currency)
    partial_price1 = Price.of(None, Decimal("123.45"), Date(2023, 1, 1))
    with pytest.raises(MonetaryOperationException):
        partial_price1.as_integer()
    
    # Test with partially undefined price (no quantity)
    partial_price2 = Price.of(Currencies["USD"], None, Date(2023, 1, 1))
    with pytest.raises(MonetaryOperationException):
        partial_price2.as_integer()
    
    # Test with partially undefined price (no date)
    partial_price3 = Price.of(Currencies["USD"], Decimal("123.45"), None)
    with pytest.raises(MonetaryOperationException):
        partial_price3.as_integer()
    
    # Test with very large quantity
    large_price = Price.of(Currencies["USD"], Decimal("999999999.99"), Date(2023, 1, 1))
    assert large_price.as_integer() == 999999999
    
    # Test with negative quantity
    negative_price = Price.of(Currencies["USD"], Decimal("-123.45"), Date(2023, 1, 1))
    assert negative_price.as_integer() == -123
    
    # Test with zero quantity
    zero_price = Price.of(Currencies["USD"], Decimal("0.00"), Date(2023, 1, 1))
    assert zero_price.as_integer() == 0


# LLM-generated content at query #25
#--------------------------

```python
def test_Price_or_else():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currencies
    
    # Create test data
    usd = Currencies["USD"]
    eur = Currencies["EUR"]
    date1 = date(2019, 1, 1)
    date2 = date(2019, 1, 2)
    
    # Create price instances
    some_price = Price.of(usd, Decimal('1'), date1)
    fallback_price = Price.of(eur, Decimal('2'), date2)
    none_price = Price.of(None, Decimal('1'), None)
    
    # Test 1: Defined price returns itself
    result1 = some_price.or_else(lambda: fallback_price)
    assert result1 is some_price
    assert result1.ccy.code == 'USD'
    assert result1.qty == Decimal('1')
    assert result1.dov == date1
    
    # Test 2: Undefined price returns fallback
    result2 = none_price.or_else(lambda: fallback_price)
    assert result2 is fallback_price
    assert result2.ccy.code == 'EUR'
    assert result2.qty == Decimal('2')
    assert result2.dov == date2
    
    # Test 3: Fallback can be dynamically generated
    dynamic_fallback = Price.of(usd, Decimal('100'), date1)
    result3 = none_price.or_else(lambda: dynamic_fallback)
    assert result3 is dynamic_fallback
    assert result3.qty == Decimal('100')
    
    # Test 4: Multiple calls to or_else on undefined price
    result4 = none_price.or_else(lambda: Price.of(usd, Decimal('50'), date1))
    assert result4.qty == Decimal('50')
    
    # Test 5: Verify fallback is only called when needed
    call_count = 0
    def counting_fallback():
        nonlocal call_count
        call_count += 1
        return fallback_price
    
    # Should not call fallback
    result5 = some_price.or_else(counting_fallback)
    assert call_count == 0
    assert result5 is some_price
    
    # Should call fallback
    result6 = none_price.or_else(counting_fallback)
    assert call_count == 1
    assert result6 is fallback_price


# LLM-generated content at query #26
#--------------------------

```python
def test_Price___eq__():
    # Test equal defined prices
    price1 = Price.of(Currencies["USD"], Decimal("100.50"), Date(2023, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal("100.50"), Date(2023, 1, 1))
    assert price1 == price2
    assert not (price1 != price2)

    # Test different currencies
    price3 = Price.of(Currencies["EUR"], Decimal("100.50"), Date(2023, 1, 1))
    assert price1 != price3
    assert not (price1 == price3)

    # Test different quantities
    price4 = Price.of(Currencies["USD"], Decimal("200.50"), Date(2023, 1, 1))
    assert price1 != price4
    assert not (price1 == price4)

    # Test different dates
    price5 = Price.of(Currencies["USD"], Decimal("100.50"), Date(2023, 1, 2))
    assert price1 != price5
    assert not (price1 == price5)

    # Test undefined prices
    undefined1 = Price.na()
    undefined2 = Price.of(None, Decimal("100.50"), Date(2023, 1, 1))
    undefined3 = Price.of(Currencies["USD"], None, Date(2023, 1, 1))
    undefined4 = Price.of(Currencies["USD"], Decimal("100.50"), None)
    assert undefined1 == undefined2
    assert undefined1 == undefined3
    assert undefined1 == undefined4
    assert undefined2 == undefined3
    assert undefined2 == undefined4
    assert undefined3 == undefined4

    # Test defined vs undefined
    assert price1 != undefined1
    assert not (price1 == undefined1)

    # Test equality with non-Price object
    assert price1 != "not a price"
    assert price1 != 123
    assert price1 != None
    assert undefined1 != "not a price"
    assert undefined1 != 123

    # Test same object
    assert price1 == price1
    assert undefined1 == undefined1

    # Test with different decimal precision but same value
    price6 = Price.of(Currencies["USD"], Decimal("100.5"), Date(2023, 1, 1))
    assert price1 == price6


# LLM-generated content at query #27
#--------------------------

```python
def test_Price_fmap():
    from decimal import Decimal
    from datetime import date, timedelta
    from pypara.currencies import Currencies
    from pypara.monetary import Price

    # Test with defined price
    someprice = Price.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    
    def increment_price(x):
        return Price.of(x.ccy, x.qty + Decimal('1'), x.dov + timedelta(days=10))
    
    result = someprice.fmap(increment_price)
    
    assert Price.is_some(result)
    assert result.ccy.code == 'USD'
    assert result.qty == Decimal('2')
    assert result.dov == date(2019, 1, 11)
    
    # Test with undefined price
    noneprice = Price.of(None, Decimal('1'), None)
    
    def identity_price(sp):
        return Price.of(sp.ccy, sp.qty + Decimal('1'), sp.dov)
    
    result2 = noneprice.fmap(identity_price)
    
    assert Price.is_none(result2)
    assert result2 is Price.na()
    
    # Test that fmap returns undefined when function returns undefined
    def return_none(_):
        return Price.na()
    
    result3 = someprice.fmap(return_none)
    assert Price.is_none(result3)
    
    # Test with function that changes currency
    def change_currency(x):
        return Price.of(Currencies["EUR"], x.qty, x.dov)
    
    result4 = someprice.fmap(change_currency)
    assert Price.is_some(result4)
    assert result4.ccy.code == 'EUR'
    assert result4.qty == Decimal('1')
    assert result4.dov == date(2019, 1, 1)


# LLM-generated content at query #28
#--------------------------

```python
def test_SomePrice_convert():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currencies
    from pypara.currencies import Currency
    from pypara.fx import FXRateService, FXRateLookupError
    from pypara.money import SomeMoney
    from pypara.prices import SomePrice, NoPrice
    
    # Mock FXRateService for testing
    class MockFXRateService:
        def __init__(self):
            self.rates = {}
        
        def query(self, ccy_from, ccy_to, asof, strict):
            key = (ccy_from.code, ccy_to.code, asof)
            return self.rates.get(key)
    
    # Save original service and set up mock
    original_service = FXRateService.default
    mock_service = MockFXRateService()
    FXRateService.default = mock_service
    
    try:
        # Test 1: Successful conversion with explicit asof date
        usd = Currencies["USD"]
        eur = Currencies["EUR"]
        price = SomePrice(usd, Decimal("100"), date(2023, 1, 1))
        
        # Set up mock rate
        mock_service.rates[("USD", "EUR", date(2023, 1, 15))] = Decimal("0.92")
        
        result = price.convert(eur, date(2023, 1, 15))
        assert isinstance(result, SomePrice)
        assert result.ccy == eur
        assert result.qty == Decimal("92.00")  # 100 * 0.92
        assert result.dov == date(2023, 1, 15)
        
        # Test 2: Successful conversion using price's dov as asof
        mock_service.rates[("USD", "EUR", date(2023, 1, 1))] = Decimal("0.90")
        
        result = price.convert(eur)  # No asof specified, should use price's dov
        assert isinstance(result, SomePrice)
        assert result.ccy == eur
        assert result.qty == Decimal("90.00")  # 100 * 0.90
        assert result.dov == date(2023, 1, 1)
        
        # Test 3: Conversion with strict=True and rate not found (should raise)
        mock_service.rates.clear()  # Clear all rates
        
        try:
            price.convert(eur, strict=True)
            assert False, "Should have raised FXRateLookupError"
        except FXRateLookupError:
            pass  # Expected
        
        # Test 4: Conversion with strict=False and rate not found (should return NoPrice)
        result = price.convert(eur, strict=False)
        assert result is NoPrice
        
        # Test 5: Conversion with different asof date
        mock_service.rates[("USD", "EUR", date(2023, 2, 1))] = Decimal("0.95")
        
        result = price.convert(eur, date(2023, 2, 1))
        assert isinstance(result, SomePrice)
        assert result.ccy == eur
        assert result.qty == Decimal("95.00")  # 100 * 0.95
        assert result.dov == date(2023, 2, 1)
        
        # Test 6: Conversion to same currency (rate should be 1)
        result = price.convert(usd, date(2023, 1, 1))
        assert isinstance(result, SomePrice)
        assert result.ccy == usd
        assert result.qty == Decimal("100.00")
        assert result.dov == date(2023, 1, 1)
        
        # Test 7: FXRateService.default is None (should raise ProgrammingError)
        FXRateService.default = None
        try:
            price.convert(eur)
            assert False, "Should have raised ProgrammingError"
        except ProgrammingError:
            pass  # Expected
        
        # Restore mock for remaining tests
        FXRateService.default = mock_service
        
        # Test 8: Conversion with rate that results in decimal
        mock_service.rates[("USD", "EUR", date(2023, 1, 1))] = Decimal("0.9125")
        
        result = price.convert(eur)
        assert isinstance(result, SomePrice)
        assert result.ccy == eur
        assert result.qty == Decimal("91.25")  # 100 * 0.9125
        assert result.dov == date(2023, 1, 1)
        
        # Test 9: Verify money property after conversion
        result = price.convert(eur, date(2023, 1, 15))
        money = result.money
        assert isinstance(money, SomeMoney)
        assert money.ccy == eur
        assert money.qty == Decimal("92.00")  # Quantized to EUR's quantizer
        
    finally:
        # Restore original service
        FXRateService.default = original_service


# LLM-generated content at query #29
#--------------------------

```python
def test_Money_gt():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currencies

    # Create defined money objects
    usd_money1 = Money.of(Currencies["USD"], Decimal("100.50"), date(2023, 1, 1))
    usd_money2 = Money.of(Currencies["USD"], Decimal("200.75"), date(2023, 1, 1))
    eur_money = Money.of(Currencies["EUR"], Decimal("100.50"), date(2023, 1, 1))
    
    # Create undefined money object
    undefined_money = Money.na()

    # Test 1: Defined > Defined (same currency, greater quantity)
    assert usd_money2 > usd_money1
    assert usd_money2.gt(usd_money1)

    # Test 2: Defined > Defined (same currency, smaller quantity) - should be False
    assert not (usd_money1 > usd_money2)
    assert not usd_money1.gt(usd_money2)

    # Test 3: Defined > Defined (same currency, equal quantity)
    usd_money3 = Money.of(Currencies["USD"], Decimal("100.50"), date(2023, 1, 1))
    assert not (usd_money1 > usd_money3)
    assert not usd_money1.gt(usd_money3)

    # Test 4: Defined > Undefined - should be True
    assert usd_money1 > undefined_money
    assert usd_money1.gt(undefined_money)

    # Test 5: Undefined > Defined - should be False
    assert not (undefined_money > usd_money1)
    assert not undefined_money.gt(usd_money1)

    # Test 6: Undefined > Undefined - should be False
    undefined_money2 = Money.na()
    assert not (undefined_money > undefined_money2)
    assert not undefined_money.gt(undefined_money2)

    # Test 7: Different currencies should raise IncompatibleCurrencyError
    try:
        usd_money1 > eur_money
        assert False, "Should have raised IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass

    try:
        usd_money1.gt(eur_money)
        assert False, "Should have raised IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass

    # Test 8: Different dates but same currency and quantity
    usd_money4 = Money.of(Currencies["USD"], Decimal("100.50"), date(2023, 1, 2))
    assert not (usd_money1 > usd_money4)  # Same quantity, different dates
    assert not usd_money1.gt(usd_money4)

    # Test 9: Edge case with zero quantity
    usd_zero = Money.of(Currencies["USD"], Decimal("0"), date(2023, 1, 1))
    usd_negative = Money.of(Currencies["USD"], Decimal("-10"), date(2023, 1, 1))
    
    assert usd_zero > usd_negative
    assert usd_zero.gt(usd_negative)
    assert not (usd_negative > usd_zero)
    assert not usd_negative.gt(usd_zero)

    # Test 10: Using __gt__ method directly
    assert usd_money2.__gt__(usd_money1)
    assert not usd_money1.__gt__(usd_money2)
    assert usd_money1.__gt__(undefined_money)
    assert not undefined_money.__gt__(usd_money1)


