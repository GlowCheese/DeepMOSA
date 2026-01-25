####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_SomePrice_qty_or_none():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    
    # Test with defined price - should return the qty
    some_price = SomePrice(Currencies["USD"], Decimal('100.50'), Date(2019, 1, 1))
    assert some_price.qty_or_none() == Decimal('100.50')
    
    # Test with zero qty
    some_price_zero = SomePrice(Currencies["EUR"], Decimal('0'), Date(2020, 6, 15))
    assert some_price_zero.qty_or_none() == Decimal('0')
    
    # Test with negative qty
    some_price_negative = SomePrice(Currencies["GBP"], Decimal('-50.25'), Date(2021, 12, 31))
    assert some_price_negative.qty_or_none() == Decimal('-50.25')
    
    # Test with very large qty
    some_price_large = SomePrice(Currencies["JPY"], Decimal('999999999.99'), Date(2018, 3, 20))
    assert some_price_large.qty_or_none() == Decimal('999999999.99')
    
    # Test with very small qty
    some_price_small = SomePrice(Currencies["USD"], Decimal('0.01'), Date(2022, 5, 10))
    assert some_price_small.qty_or_none() == Decimal('0.01')


# LLM-generated content at query #2
#--------------------------

```python
def test_SomePrice___mul__():
    """Unit tests for SomePrice.__mul__ method"""
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    
    # Test multiplication with Decimal
    price = SomePrice(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = price * Decimal('2')
    assert isinstance(result, SomePrice)
    assert result.ccy == Currencies["USD"]
    assert result.qty == Decimal('20')
    assert result.dov == Date(2019, 1, 1)
    
    # Test multiplication with int
    result = price * 3
    assert result.qty == Decimal('30')
    
    # Test multiplication with float
    result = price * 2.5
    assert result.qty == Decimal('25')
    
    # Test multiplication with zero
    result = price * 0
    assert result.qty == Decimal('0')
    
    # Test multiplication with negative number
    result = price * Decimal('-1')
    assert result.qty == Decimal('-10')
    
    # Test multiplication preserves currency and date
    result = price * Decimal('5')
    assert result.ccy == Currencies["USD"]
    assert result.dov == Date(2019, 1, 1)
    
    # Test multiplication with different currency
    eur_price = SomePrice(Currencies["EUR"], Decimal('100'), Date(2019, 6, 15))
    result = eur_price * Decimal('0.5')
    assert result.ccy == Currencies["EUR"]
    assert result.qty == Decimal('50')
    assert result.dov == Date(2019, 6, 15)


# LLM-generated content at query #3
#--------------------------

```python
def test_SomeMoney_convert():
    from pypara.currencies import Currencies
    from pypara.fx import FXRate, FXRateService
    from datetime import date as Date
    from decimal import Decimal
    
    # Setup test currencies
    usd = Currencies["USD"]
    eur = Currencies["EUR"]
    gbp = Currencies["GBP"]
    
    # Create a SomeMoney instance
    money_usd = SomeMoney(usd, Decimal('100'), Date(2019, 1, 1))
    
    # Mock FXRateService
    class MockFXRateService:
        def query(self, from_ccy, to_ccy, asof, strict=False):
            if from_ccy == usd and to_ccy == eur:
                return FXRate(usd, eur, Decimal('0.85'), Date(2019, 1, 1))
            elif from_ccy == usd and to_ccy == gbp:
                return FXRate(usd, gbp, Decimal('0.73'), Date(2019, 1, 1))
            return None
    
    # Save original default service
    original_service = FXRateService.default
    FXRateService.default = MockFXRateService()
    
    try:
        # Test successful conversion with asof parameter
        result = money_usd.convert(eur, asof=Date(2019, 1, 1))
        assert isinstance(result, SomeMoney)
        assert result.ccy == eur
        assert result.qty == Decimal('85.00')
        assert result.dov == Date(2019, 1, 1)
        
        # Test successful conversion without asof parameter (uses dov)
        result = money_usd.convert(gbp)
        assert isinstance(result, SomeMoney)
        assert result.ccy == gbp
        assert result.qty == Decimal('73.00')
        assert result.dov == Date(2019, 1, 1)
        
        # Test conversion with non-existent rate in non-strict mode
        result = money_usd.convert(Currencies["JPY"], strict=False)
        assert result is NoMoney
        assert result.undefined
        
        # Test conversion with non-existent rate in strict mode
        with pytest.raises(FXRateLookupError):
            money_usd.convert(Currencies["JPY"], asof=Date(2019, 1, 1), strict=True)
        
        # Test conversion with None FXRateService raises ProgrammingError
        FXRateService.default = None
        with pytest.raises(ProgrammingError):
            money_usd.convert(eur)
            
    finally:
        # Restore original service
        FXRateService.default = original_service


# LLM-generated content at query #4
#--------------------------

```python
def test_Money_scalar_subtract():
    """Unit tests for Money.scalar_subtract method"""
    from decimal import Decimal
    from datetime import date as Date
    
    # Test 1: Scalar subtraction on defined money
    money = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    result = money.scalar_subtract(Decimal('30'))
    assert result.qty == Decimal('70.00')
    assert result.ccy.code == 'USD'
    assert result.dov == Date(2019, 1, 1)
    
    # Test 2: Scalar subtraction with negative number
    money = Money.of(Currencies["EUR"], Decimal('50'), Date(2019, 1, 15))
    result = money.scalar_subtract(Decimal('-20'))
    assert result.qty == Decimal('70.00')
    assert result.ccy.code == 'EUR'
    
    # Test 3: Scalar subtraction resulting in negative quantity
    money = Money.of(Currencies["GBP"], Decimal('10'), Date(2019, 2, 1))
    result = money.scalar_subtract(Decimal('25'))
    assert result.qty == Decimal('-15.00')
    
    # Test 4: Scalar subtraction with zero
    money = Money.of(Currencies["JPY"], Decimal('1000'), Date(2019, 3, 1))
    result = money.scalar_subtract(Decimal('0'))
    assert result.qty == Decimal('1000.00')
    
    # Test 5: Scalar subtraction with decimal places
    money = Money.of(Currencies["USD"], Decimal('100.50'), Date(2019, 1, 1))
    result = money.scalar_subtract(Decimal('0.25'))
    assert result.qty == Decimal('100.25')
    
    # Test 6: Undefined money returns itself
    undefined_money = Money.na()
    result = undefined_money.scalar_subtract(Decimal('50'))
    assert result.undefined
    assert result is undefined_money or result == Money.na()
    
    # Test 7: Scalar subtraction with integer
    money = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    result = money.scalar_subtract(10)
    assert result.qty == Decimal('90.00')
    
    # Test 8: Scalar subtraction with float
    money = Money.of(Currencies["EUR"], Decimal('100'), Date(2019, 1, 1))
    result = money.scalar_subtract(15.5)
    assert result.qty == Decimal('84.50')
    
    # Test 9: Date is preserved after scalar subtraction
    test_date = Date(2020, 6, 15)
    money = Money.of(Currencies["CHF"], Decimal('200'), test_date)
    result = money.scalar_subtract(Decimal('50'))
    assert result.dov == test_date
    
    # Test 10: Large scalar subtraction
    money = Money.of(Currencies["USD"], Decimal('1000000'), Date(2019, 1, 1))
    result = money.scalar_subtract(Decimal('999999'))
    assert result.qty == Decimal('1.00')


# LLM-generated content at query #5
#--------------------------

```python
def test_SomePrice_times():
    """Unit tests for SomePrice.times method"""
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    
    # Test 1: times with integer multiplier
    price = SomePrice(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = price.times(2)
    assert result.ccy == Currencies["USD"]
    assert result.qty == Decimal('20')
    assert result.dov == Date(2019, 1, 1)
    
    # Test 2: times with Decimal multiplier
    price = SomePrice(Currencies["EUR"], Decimal('15.5'), Date(2020, 6, 15))
    result = price.times(Decimal('3'))
    assert result.ccy == Currencies["EUR"]
    assert result.qty == Decimal('46.50')
    assert result.dov == Date(2020, 6, 15)
    
    # Test 3: times with float multiplier
    price = SomePrice(Currencies["GBP"], Decimal('100'), Date(2021, 12, 31))
    result = price.times(1.5)
    assert result.ccy == Currencies["GBP"]
    assert result.qty == Decimal('150.00')
    
    # Test 4: times with zero multiplier
    price = SomePrice(Currencies["JPY"], Decimal('1000'), Date(2019, 5, 10))
    result = price.times(0)
    assert result.ccy == Currencies["JPY"]
    assert result.qty == Decimal('0')
    
    # Test 5: times with negative multiplier
    price = SomePrice(Currencies["USD"], Decimal('50'), Date(2019, 3, 20))
    result = price.times(-2)
    assert result.ccy == Currencies["USD"]
    assert result.qty == Decimal('-100')
    
    # Test 6: times with fractional multiplier
    price = SomePrice(Currencies["CHF"], Decimal('100'), Date(2019, 7, 1))
    result = price.times(Decimal('0.25'))
    assert result.ccy == Currencies["CHF"]
    assert result.qty == Decimal('25.00')
    
    # Test 7: times returns Money type
    price = SomePrice(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = price.times(5)
    assert isinstance(result, type(price.money))
    assert result.ccy == Currencies["USD"]
    assert result.dov == Date(2019, 1, 1)


# LLM-generated content at query #6
#--------------------------

```python
def test_Money_ccy_or():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    # Test with defined money - should return the currency of the money object
    somemoney = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    result = somemoney.ccy_or(Currencies["EUR"])
    assert result.code == 'USD'
    assert result is Currencies["USD"]
    
    # Test with undefined money - should return the default currency
    nonemoney = Money.of(Currencies["USD"], None, None)
    result = nonemoney.ccy_or(Currencies["EUR"])
    assert result.code == 'EUR'
    assert result is Currencies["EUR"]
    
    # Test with undefined money (all None) - should return the default currency
    nonemoney2 = Money.of(None, None, None)
    result = nonemoney2.ccy_or(Currencies["GBP"])
    assert result.code == 'GBP'
    assert result is Currencies["GBP"]
    
    # Test with defined money with different currency - should return its own currency
    somemoney2 = Money.of(Currencies["EUR"], Decimal('100'), Date(2020, 6, 15))
    result = somemoney2.ccy_or(Currencies["USD"])
    assert result.code == 'EUR'
    assert result is Currencies["EUR"]


# LLM-generated content at query #7
#--------------------------

```python
def test_Price_lte():
    """Test the lte (less than or equal to) method of Price class."""
    from pypara.currencies import Currencies
    from datetime import date as Date
    
    # Test 1: Undefined price is always less than or equal to another undefined price
    undefined_price1 = Price.of(None, None, None)
    undefined_price2 = Price.of(None, None, None)
    assert undefined_price1.lte(undefined_price2) is True
    
    # Test 2: Undefined price is always less than or equal to a defined price
    undefined_price = Price.of(None, None, None)
    defined_price = Price.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    assert undefined_price.lte(defined_price) is True
    
    # Test 3: Defined price is not less than or equal to undefined price (when defined > undefined)
    defined_price = Price.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    undefined_price = Price.of(None, None, None)
    assert defined_price.lte(undefined_price) is False
    
    # Test 4: Equal defined prices should be less than or equal
    price1 = Price.of(Currencies["USD"], Decimal('50'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('50'), Date(2019, 1, 1))
    assert price1.lte(price2) is True
    
    # Test 5: Smaller defined price should be less than or equal to larger price
    price1 = Price.of(Currencies["USD"], Decimal('30'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('50'), Date(2019, 1, 1))
    assert price1.lte(price2) is True
    
    # Test 6: Larger defined price should not be less than or equal to smaller price
    price1 = Price.of(Currencies["USD"], Decimal('70'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('50'), Date(2019, 1, 1))
    assert price1.lte(price2) is False
    
    # Test 7: IncompatibleCurrencyError should be raised when comparing defined prices with different currencies
    price_usd = Price.of(Currencies["USD"], Decimal('50'), Date(2019, 1, 1))
    price_eur = Price.of(Currencies["EUR"], Decimal('50'), Date(2019, 1, 1))
    with pytest.raises(IncompatibleCurrencyError):
        price_usd.lte(price_eur)
    
    # Test 8: Negative prices comparison
    price_negative = Price.of(Currencies["USD"], Decimal('-10'), Date(2019, 1, 1))
    price_positive = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    assert price_negative.lte(price_positive) is True
    
    # Test 9: Two negative prices comparison
    price_neg1 = Price.of(Currencies["USD"], Decimal('-50'), Date(2019, 1, 1))
    price_neg2 = Price.of(Currencies["USD"], Decimal('-30'), Date(2019, 1, 1))
    assert price_neg1.lte(price_neg2) is True


# LLM-generated content at query #8
#--------------------------

```python
def test_SomeMoney___floordiv__():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    
    # Test floor division with positive divisor
    money = SomeMoney(Currencies["USD"], Decimal('10.50'), Date(2019, 1, 1))
    result = money // Decimal('3')
    assert isinstance(result, SomeMoney)
    assert result.ccy == Currencies["USD"]
    assert result.qty == Decimal('3')
    assert result.dov == Date(2019, 1, 1)
    
    # Test floor division with integer
    money = SomeMoney(Currencies["USD"], Decimal('20'), Date(2019, 1, 1))
    result = money // 3
    assert isinstance(result, SomeMoney)
    assert result.qty == Decimal('6')
    
    # Test floor division with decimal divisor
    money = SomeMoney(Currencies["EUR"], Decimal('15.75'), Date(2020, 6, 15))
    result = money // Decimal('2.5')
    assert isinstance(result, SomeMoney)
    assert result.ccy == Currencies["EUR"]
    assert result.qty == Decimal('6')
    assert result.dov == Date(2020, 6, 15)
    
    # Test floor division by 1
    money = SomeMoney(Currencies["GBP"], Decimal('42.99'), Date(2021, 3, 10))
    result = money // 1
    assert isinstance(result, SomeMoney)
    assert result.qty == Decimal('42.99')
    
    # Test floor division with float
    money = SomeMoney(Currencies["JPY"], Decimal('100'), Date(2019, 1, 1))
    result = money // 3.0
    assert isinstance(result, SomeMoney)
    assert result.qty == Decimal('33')
    
    # Test floor division by zero returns NoMoney
    money = SomeMoney(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = money // 0
    assert result.undefined
    assert result is NoMoney
    
    # Test floor division with Decimal zero returns NoMoney
    money = SomeMoney(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = money // Decimal('0')
    assert result.undefined
    
    # Test floor division maintains currency precision
    money = SomeMoney(Currencies["USD"], Decimal('100.99'), Date(2019, 1, 1))
    result = money // Decimal('7')
    assert result.ccy == Currencies["USD"]
    assert result.qty == Decimal('14.42')


# LLM-generated content at query #9
#--------------------------

```python
def test_Money_qty_map():
    """Test qty_map method of Money class."""
    from pypara.currencies import Currencies
    from datetime import date as Date
    
    # Test with defined money - applies function to quantity
    somemoney = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    result = somemoney.qty_map(lambda x: x + Decimal('1'), lambda: Decimal('42'))
    assert result == Decimal('2.00')
    
    # Test with defined money - function returns different type
    result = somemoney.qty_map(lambda x: str(x), lambda: "fallback")
    assert result == "1.00"
    
    # Test with undefined money - calls combinator
    nonemoney = Money.of(None, Decimal('1'), None)
    result = nonemoney.qty_map(lambda x: x + Decimal('1'), lambda: Decimal('42'))
    assert result == Decimal('42')
    
    # Test with undefined money - combinator returns different type
    result = nonemoney.qty_map(lambda x: x + Decimal('1'), lambda: "fallback")
    assert result == "fallback"
    
    # Test with defined money - function with multiplication
    somemoney = Money.of(Currencies["EUR"], Decimal('5'), Date(2020, 6, 15))
    result = somemoney.qty_map(lambda x: x * Decimal('2'), lambda: Decimal('0'))
    assert result == Decimal('10.00')
    
    # Test with defined money - function returns boolean
    result = somemoney.qty_map(lambda x: x > Decimal('0'), lambda: False)
    assert result is True
    
    # Test with undefined money - function never called
    nonemoney = Money.of(Currencies["USD"], None, None)
    call_count = []
    result = nonemoney.qty_map(
        lambda x: (call_count.append(1), x)[1],
        lambda: Decimal('100')
    )
    assert result == Decimal('100')
    assert len(call_count) == 0
    
    # Test with defined money - combinator never called
    somemoney = Money.of(Currencies["GBP"], Decimal('7'), Date(2021, 3, 20))
    call_count = []
    result = somemoney.qty_map(
        lambda x: x / Decimal('2'),
        lambda: (call_count.append(1), Decimal('0'))[1]
    )
    assert result == Decimal('3.50')
    assert len(call_count) == 0
    
    # Test with zero quantity
    somemoney = Money.of(Currencies["JPY"], Decimal('0'), Date(2022, 12, 25))
    result = somemoney.qty_map(lambda x: x + Decimal('10'), lambda: Decimal('-1'))
    assert result == Decimal('10.00')
    
    # Test with negative quantity
    somemoney = Money.of(Currencies["CAD"], Decimal('-5'), Date(2023, 1, 1))
    result = somemoney.qty_map(lambda x: abs(x), lambda: Decimal('0'))
    assert result == Decimal('5.00')


# LLM-generated content at query #10
#--------------------------

def test_Money___bool__():
    """Test the __bool__ method of Money class."""
    from decimal import Decimal
    from datetime import date as Date
    from pypara.currencies import Currencies
    
    # Test that defined money with non-zero quantity is truthy
    defined_money = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert bool(defined_money) is True
    
    # Test that defined money with zero quantity is falsy
    zero_money = Money.of(Currencies["USD"], Decimal('0'), Date(2019, 1, 1))
    assert bool(zero_money) is False
    
    # Test that undefined money is falsy
    undefined_money = Money.na()
    assert bool(undefined_money) is False
    
    # Test that defined money with negative quantity is truthy
    negative_money = Money.of(Currencies["USD"], Decimal('-1'), Date(2019, 1, 1))
    assert bool(negative_money) is True
    
    # Test that defined money with positive quantity is truthy
    positive_money = Money.of(Currencies["USD"], Decimal('100.50'), Date(2019, 1, 1))
    assert bool(positive_money) is True
    
    # Test in conditional statements
    if defined_money:
        assert True
    else:
        assert False, "Defined money with non-zero quantity should be truthy"
    
    if undefined_money:
        assert False, "Undefined money should be falsy"
    else:
        assert True


# LLM-generated content at query #11
#--------------------------

```python
def test_SomePrice_round():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    
    # Test rounding with default ndigits=0
    price = SomePrice(Currencies["USD"], Decimal('1.567'), Date(2019, 1, 1))
    rounded_price = price.round()
    assert isinstance(rounded_price, SomePrice)
    assert rounded_price.ccy == Currencies["USD"]
    assert rounded_price.qty == Decimal('2')
    assert rounded_price.dov == Date(2019, 1, 1)
    
    # Test rounding with positive ndigits
    price = SomePrice(Currencies["USD"], Decimal('1.567'), Date(2019, 1, 1))
    rounded_price = price.round(2)
    assert isinstance(rounded_price, SomePrice)
    assert rounded_price.qty == Decimal('1.57')
    assert rounded_price.ccy == Currencies["USD"]
    assert rounded_price.dov == Date(2019, 1, 1)
    
    # Test rounding with ndigits=0 explicitly
    price = SomePrice(Currencies["USD"], Decimal('2.4'), Date(2019, 1, 1))
    rounded_price = price.round(0)
    assert rounded_price.qty == Decimal('2')
    
    # Test rounding with ndigits=1
    price = SomePrice(Currencies["USD"], Decimal('1.567'), Date(2019, 1, 1))
    rounded_price = price.round(1)
    assert rounded_price.qty == Decimal('1.6')
    
    # Test rounding preserves currency and date
    price = SomePrice(Currencies["EUR"], Decimal('99.999'), Date(2020, 6, 15))
    rounded_price = price.round(2)
    assert rounded_price.ccy == Currencies["EUR"]
    assert rounded_price.dov == Date(2020, 6, 15)
    assert rounded_price.qty == Decimal('100.00')
    
    # Test rounding with negative quantity
    price = SomePrice(Currencies["GBP"], Decimal('-3.567'), Date(2021, 3, 10))
    rounded_price = price.round(1)
    assert rounded_price.qty == Decimal('-3.6')
    assert rounded_price.ccy == Currencies["GBP"]
    
    # Test rounding with zero
    price = SomePrice(Currencies["USD"], Decimal('0.567'), Date(2019, 1, 1))
    rounded_price = price.round(0)
    assert rounded_price.qty == Decimal('1')
    
    # Test rounding with large ndigits
    price = SomePrice(Currencies["USD"], Decimal('1.123456789'), Date(2019, 1, 1))
    rounded_price = price.round(5)
    assert rounded_price.qty == Decimal('1.12346')


# LLM-generated content at query #12
#--------------------------

```python
def test_Price_ccy_or():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    # Test with defined price - should return the price's currency
    defined_price = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert defined_price.ccy_or(Currencies["EUR"]) == Currencies["USD"]
    assert defined_price.ccy_or(Currencies["EUR"]).code == "USD"
    
    # Test with undefined price (no quantity) - should return default currency
    undefined_price_no_qty = Price.of(Currencies["USD"], None, None)
    assert undefined_price_no_qty.ccy_or(Currencies["EUR"]) == Currencies["EUR"]
    assert undefined_price_no_qty.ccy_or(Currencies["EUR"]).code == "EUR"
    
    # Test with undefined price (Price.na()) - should return default currency
    undefined_price_na = Price.na()
    assert undefined_price_na.ccy_or(Currencies["GBP"]) == Currencies["GBP"]
    assert undefined_price_na.ccy_or(Currencies["GBP"]).code == "GBP"
    
    # Test with defined price using different currencies
    eur_price = Price.of(Currencies["EUR"], Decimal('100'), Date(2020, 6, 15))
    assert eur_price.ccy_or(Currencies["USD"]) == Currencies["EUR"]
    assert eur_price.ccy_or(Currencies["USD"]).code == "EUR"
    
    # Test with None currency in undefined price - should return default
    none_ccy_price = Price.of(None, Decimal('1'), None)
    assert none_ccy_price.ccy_or(Currencies["JPY"]) == Currencies["JPY"]
    assert none_ccy_price.ccy_or(Currencies["JPY"]).code == "JPY"


# LLM-generated content at query #13
#--------------------------

```python
def test_SomePrice___gt__():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    
    # Test: SomePrice > SomePrice with same currency
    price1 = SomePrice(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    price2 = SomePrice(Currencies["USD"], Decimal('50'), Date(2019, 1, 1))
    assert price1 > price2 is True
    
    # Test: SomePrice > SomePrice with same currency (equal quantities)
    price3 = SomePrice(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    price4 = SomePrice(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    assert price3 > price4 is False
    
    # Test: SomePrice > SomePrice with same currency (less than)
    price5 = SomePrice(Currencies["USD"], Decimal('50'), Date(2019, 1, 1))
    price6 = SomePrice(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    assert price5 > price6 is False
    
    # Test: SomePrice > NonePrice (should return True)
    price7 = SomePrice(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    assert price7 > Price.na() is True
    
    # Test: SomePrice > SomePrice with different currencies (should raise)
    price8 = SomePrice(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    price9 = SomePrice(Currencies["EUR"], Decimal('100'), Date(2019, 1, 1))
    try:
        price8 > price9
        assert False, "Should have raised IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass


# LLM-generated content at query #14
#--------------------------

```python
def test_Price_as_boolean():
    from pypara.currencies import Currencies
    from datetime import date as Date
    
    # Test with defined price (non-zero quantity)
    price_nonzero = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert price_nonzero.as_boolean() is True
    
    # Test with defined price (zero quantity)
    price_zero = Price.of(Currencies["USD"], Decimal('0'), Date(2019, 1, 1))
    assert price_zero.as_boolean() is False
    
    # Test with defined price (negative quantity)
    price_negative = Price.of(Currencies["USD"], Decimal('-5'), Date(2019, 1, 1))
    assert price_negative.as_boolean() is True
    
    # Test with undefined price
    price_undefined = Price.na()
    assert price_undefined.as_boolean() is False
    
    # Test with defined price (positive decimal quantity)
    price_decimal = Price.of(Currencies["EUR"], Decimal('0.5'), Date(2020, 6, 15))
    assert price_decimal.as_boolean() is True
    
    # Test with defined price (very small positive quantity)
    price_small = Price.of(Currencies["GBP"], Decimal('0.001'), Date(2021, 3, 10))
    assert price_small.as_boolean() is True


# LLM-generated content at query #15
#--------------------------

```python
def test_Money_as_boolean():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    # Test with defined money (non-zero quantity)
    money_nonzero = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert bool(money_nonzero) is True
    
    # Test with defined money (zero quantity)
    money_zero = Money.of(Currencies["USD"], Decimal('0'), Date(2019, 1, 1))
    assert bool(money_zero) is False
    
    # Test with undefined money
    money_undefined = Money.na()
    assert bool(money_undefined) is False
    
    # Test with defined money (negative quantity)
    money_negative = Money.of(Currencies["USD"], Decimal('-1'), Date(2019, 1, 1))
    assert bool(money_negative) is True
    
    # Test with defined money (positive decimal quantity)
    money_decimal = Money.of(Currencies["EUR"], Decimal('0.01'), Date(2020, 6, 15))
    assert bool(money_decimal) is True


# LLM-generated content at query #16
#--------------------------

```python
def test_Price_dov_or():
    from pypara.currencies import Currencies
    from datetime import date as Date
    
    # Test case 1: defined price returns its dov
    defined_price = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    result = defined_price.dov_or(Date(2001, 1, 1))
    assert result == Date(2019, 1, 1)
    
    # Test case 2: undefined price with qty and dov returns default
    undefined_price = Price.of(None, Decimal('1'), Date(2019, 1, 1))
    result = undefined_price.dov_or(Date(2001, 1, 1))
    assert result == Date(2001, 1, 1)
    
    # Test case 3: undefined price with no dov returns default
    undefined_price_no_dov = Price.of(None, None, Date(2019, 1, 1))
    result = undefined_price_no_dov.dov_or(Date(2001, 1, 1))
    assert result == Date(2001, 1, 1)
    
    # Test case 4: price with None dov returns default
    price_none_dov = Price.of(Currencies["EUR"], Decimal('5'), None)
    result = price_none_dov.dov_or(Date(2005, 5, 5))
    assert result == Date(2005, 5, 5)
    
    # Test case 5: undefined price (Price.na()) returns default
    na_price = Price.na()
    result = na_price.dov_or(Date(2010, 10, 10))
    assert result == Date(2010, 10, 10)
    
    # Test case 6: defined price with different default date returns its own dov
    defined_price_2 = Price.of(Currencies["GBP"], Decimal('100'), Date(2020, 12, 25))
    result = defined_price_2.dov_or(Date(1999, 1, 1))
    assert result == Date(2020, 12, 25)


# LLM-generated content at query #17
#--------------------------

```python
def test_Price_convert():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    # Test converting a defined price to another currency
    usd_price = Price.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    eur_price = usd_price.convert(Currencies["EUR"], asof=Date(2019, 1, 1))
    
    assert eur_price.defined
    assert eur_price.ccy == Currencies["EUR"]
    assert eur_price.dov == Date(2019, 1, 1)
    
    # Test converting undefined price returns undefined
    undefined_price = Price.na()
    converted = undefined_price.convert(Currencies["EUR"])
    
    assert converted.undefined
    
    # Test converting without asof date
    gbp_price = usd_price.convert(Currencies["GBP"])
    
    assert gbp_price.defined
    assert gbp_price.ccy == Currencies["GBP"]
    
    # Test converting with strict mode
    strict_converted = usd_price.convert(Currencies["JPY"], asof=Date(2019, 1, 1), strict=True)
    
    assert strict_converted.defined or strict_converted.undefined
    
    # Test converting to same currency
    same_ccy_price = usd_price.convert(Currencies["USD"], asof=Date(2019, 1, 1))
    
    assert same_ccy_price.defined
    assert same_ccy_price.ccy == Currencies["USD"]
    assert same_ccy_price.qty == Decimal('100')
    
    # Test that conversion carries forward the date
    new_date = Date(2019, 6, 15)
    dated_conversion = usd_price.convert(Currencies["CHF"], asof=new_date)
    
    if dated_conversion.defined:
        assert dated_conversion.dov == new_date


# LLM-generated content at query #18
#--------------------------

```python
def test_Price_floor_divide():
    from pypara.currencies import Currencies
    from datetime import date as Date
    
    # Test floor_divide with defined price
    price = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = price.floor_divide(Decimal('3'))
    assert result.defined
    assert result.qty == Decimal('3')
    assert result.ccy == Currencies["USD"]
    
    # Test floor_divide with zero divisor (should return undefined)
    result_zero = price.floor_divide(Decimal('0'))
    assert result_zero.undefined
    
    # Test floor_divide with negative divisor
    result_neg = price.floor_divide(Decimal('-3'))
    assert result_neg.defined
    assert result_neg.qty == Decimal('-4')  # floor division of 10 / -3
    
    # Test floor_divide with decimal divisor
    result_decimal = price.floor_divide(Decimal('2.5'))
    assert result_decimal.defined
    assert result_decimal.qty == Decimal('4')  # floor of 10 / 2.5 = 4
    
    # Test floor_divide with undefined price (should return itself)
    undefined_price = Price.na()
    result_undefined = undefined_price.floor_divide(Decimal('5'))
    assert result_undefined.undefined
    
    # Test floor_divide with integer divisor
    result_int = price.floor_divide(2)
    assert result_int.defined
    assert result_int.qty == Decimal('5')
    
    # Test floor_divide with float divisor
    result_float = price.floor_divide(3.0)
    assert result_float.defined
    assert result_float.qty == Decimal('3')
    
    # Test floor_divide preserves currency
    result_ccy = price.floor_divide(Decimal('2'))
    assert result_ccy.ccy == Currencies["USD"]
    
    # Test floor_divide with very small divisor
    result_small = price.floor_divide(Decimal('0.1'))
    assert result_small.defined
    assert result_small.qty == Decimal('100')
    
    # Test floor_divide with one (identity)
    result_one = price.floor_divide(Decimal('1'))
    assert result_one.defined
    assert result_one.qty == Decimal('10')


# LLM-generated content at query #19
#--------------------------

```python
def test_Price___gt__():
    from datetime import date as Date
    from decimal import Decimal
    from pypara.currencies import Currencies
    
    # Test 1: Defined price greater than undefined price
    defined_price = Price.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    undefined_price = Price.na()
    assert defined_price > undefined_price
    
    # Test 2: Undefined price is never greater than defined price
    assert not (undefined_price > defined_price)
    
    # Test 3: Undefined price is never greater than undefined price
    assert not (undefined_price > Price.na())
    
    # Test 4: Defined price greater than another defined price with same currency
    price1 = Price.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('50'), Date(2019, 1, 1))
    assert price1 > price2
    assert not (price2 > price1)
    
    # Test 5: Defined price not greater than equal defined price
    price3 = Price.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    price4 = Price.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    assert not (price3 > price4)
    
    # Test 6: IncompatibleCurrencyError when comparing different currencies
    price_usd = Price.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    price_eur = Price.of(Currencies["EUR"], Decimal('100'), Date(2019, 1, 1))
    try:
        price_usd > price_eur
        assert False, "Should raise IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass
    
    # Test 7: Negative values comparison
    price_neg = Price.of(Currencies["USD"], Decimal('-50'), Date(2019, 1, 1))
    price_pos = Price.of(Currencies["USD"], Decimal('50'), Date(2019, 1, 1))
    assert price_pos > price_neg
    assert not (price_neg > price_pos)
    
    # Test 8: Decimal precision comparison
    price_high = Price.of(Currencies["USD"], Decimal('100.001'), Date(2019, 1, 1))
    price_low = Price.of(Currencies["USD"], Decimal('100.000'), Date(2019, 1, 1))
    assert price_high > price_low


# LLM-generated content at query #20
#--------------------------

```python
def test_Price_ccy_or():
    from pypara.currencies import Currencies
    from datetime import date as Date
    
    # Test case 1: Defined price returns its currency
    usd = Currencies["USD"]
    eur = Currencies["EUR"]
    defined_price = Price.of(usd, Decimal('1'), Date(2019, 1, 1))
    result = defined_price.ccy_or(eur)
    assert result == usd
    assert result.code == 'USD'
    
    # Test case 2: Undefined price (None currency) returns default currency
    undefined_price = Price.of(None, Decimal('1'), None)
    result = undefined_price.ccy_or(eur)
    assert result == eur
    assert result.code == 'EUR'
    
    # Test case 3: Undefined price (None quantity) returns default currency
    undefined_price_no_qty = Price.of(usd, None, None)
    result = undefined_price_no_qty.ccy_or(eur)
    assert result == eur
    assert result.code == 'EUR'
    
    # Test case 4: Undefined price (None date) returns default currency
    undefined_price_no_dov = Price.of(usd, Decimal('1'), None)
    result = undefined_price_no_dov.ccy_or(eur)
    assert result == eur
    assert result.code == 'EUR'
    
    # Test case 5: Price.na() returns default currency
    na_price = Price.na()
    result = na_price.ccy_or(usd)
    assert result == usd
    assert result.code == 'USD'


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_Money___truediv__():
    from decimal import Decimal
    from datetime import date as Date
    
    # Test with defined money objects
    usd = Currencies["USD"]
    money = Money.of(usd, Decimal('100'), Date(2019, 1, 1))
    
    # Test division by positive number
    result = money / Decimal('2')
    assert result.defined
    assert result.qty == Decimal('50.00')
    assert result.ccy == usd
    assert result.dov == Date(2019, 1, 1)
    
    # Test division by negative number
    result = money / Decimal('-2')
    assert result.defined
    assert result.qty == Decimal('-50.00')
    
    # Test division by 1
    result = money / Decimal('1')
    assert result.defined
    assert result.qty == Decimal('100.00')
    
    # Test division by decimal number
    result = money / Decimal('2.5')
    assert result.defined
    assert result.qty == Decimal('40.00')
    
    # Test division by zero yields undefined money
    result = money / Decimal('0')
    assert result.undefined
    
    # Test with undefined money object
    undefined_money = Money.na()
    result = undefined_money / Decimal('2')
    assert result.undefined
    
    # Test division with integer
    result = money / 2
    assert result.defined
    assert result.qty == Decimal('50.00')
    
    # Test division with float
    result = money / 2.0
    assert result.defined
    assert result.qty == Decimal('50.00')


# LLM-generated content at query #2
#--------------------------

```python
def test_Money_lte():
    """Test the lte (less than or equal to) method of Money class."""
    from decimal import Decimal
    from datetime import date as Date
    
    # Setup test data
    usd = Currencies["USD"]
    eur = Currencies["EUR"]
    
    # Test 1: Defined money less than other defined money
    money1 = Money.of(usd, Decimal('10'), Date(2019, 1, 1))
    money2 = Money.of(usd, Decimal('20'), Date(2019, 1, 1))
    assert money1.lte(money2) is True
    
    # Test 2: Defined money equal to other defined money
    money3 = Money.of(usd, Decimal('20'), Date(2019, 1, 1))
    money4 = Money.of(usd, Decimal('20'), Date(2019, 1, 1))
    assert money3.lte(money4) is True
    
    # Test 3: Defined money greater than other defined money
    money5 = Money.of(usd, Decimal('30'), Date(2019, 1, 1))
    money6 = Money.of(usd, Decimal('20'), Date(2019, 1, 1))
    assert money5.lte(money6) is False
    
    # Test 4: Undefined money is always less than or equal to defined money
    undefined_money = Money.na()
    defined_money = Money.of(usd, Decimal('10'), Date(2019, 1, 1))
    assert undefined_money.lte(defined_money) is True
    
    # Test 5: Undefined money is less than or equal to undefined money
    undefined_money1 = Money.na()
    undefined_money2 = Money.na()
    assert undefined_money1.lte(undefined_money2) is True
    
    # Test 6: Defined money is not less than or equal to undefined money
    defined_money2 = Money.of(usd, Decimal('10'), Date(2019, 1, 1))
    undefined_money3 = Money.na()
    assert defined_money2.lte(undefined_money3) is False
    
    # Test 7: IncompatibleCurrencyError when comparing defined money with different currencies
    money_usd = Money.of(usd, Decimal('10'), Date(2019, 1, 1))
    money_eur = Money.of(eur, Decimal('10'), Date(2019, 1, 1))
    with pytest.raises(IncompatibleCurrencyError):
        money_usd.lte(money_eur)
    
    # Test 8: Negative amounts comparison
    money_negative = Money.of(usd, Decimal('-10'), Date(2019, 1, 1))
    money_positive = Money.of(usd, Decimal('10'), Date(2019, 1, 1))
    assert money_negative.lte(money_positive) is True
    
    # Test 9: Zero comparison
    money_zero = Money.of(usd, Decimal('0'), Date(2019, 1, 1))
    money_ten = Money.of(usd, Decimal('10'), Date(2019, 1, 1))
    assert money_zero.lte(money_ten) is True
    assert money_ten.lte(money_zero) is False


# LLM-generated content at query #3
#--------------------------

```python
def test_Price_scalar_add():
    """Unit tests for Price.scalar_add method"""
    from decimal import Decimal
    from datetime import date as Date
    from pypara.currencies import Currencies
    
    # Test 1: scalar_add on defined price with positive number
    price = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = price.scalar_add(Decimal('5'))
    assert result.defined
    assert result.qty == Decimal('15')
    assert result.ccy == Currencies["USD"]
    assert result.dov == Date(2019, 1, 1)
    
    # Test 2: scalar_add on defined price with negative number
    price = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = price.scalar_add(Decimal('-3'))
    assert result.defined
    assert result.qty == Decimal('7')
    
    # Test 3: scalar_add on defined price with zero
    price = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = price.scalar_add(Decimal('0'))
    assert result.defined
    assert result.qty == Decimal('10')
    
    # Test 4: scalar_add on undefined price returns itself
    undefined_price = Price.na()
    result = undefined_price.scalar_add(Decimal('5'))
    assert result.undefined
    assert result is undefined_price
    
    # Test 5: scalar_add with float conversion
    price = Price.of(Currencies["EUR"], Decimal('20.5'), Date(2019, 6, 15))
    result = price.scalar_add(Decimal('10.25'))
    assert result.defined
    assert result.qty == Decimal('30.75')
    assert result.ccy == Currencies["EUR"]
    
    # Test 6: scalar_add with large numbers
    price = Price.of(Currencies["GBP"], Decimal('1000000'), Date(2020, 1, 1))
    result = price.scalar_add(Decimal('999999'))
    assert result.defined
    assert result.qty == Decimal('1999999')
    
    # Test 7: scalar_add with very small decimal
    price = Price.of(Currencies["JPY"], Decimal('100'), Date(2019, 1, 1))
    result = price.scalar_add(Decimal('0.001'))
    assert result.defined
    assert result.qty == Decimal('100.001')
    
    # Test 8: scalar_add preserves currency and date
    price = Price.of(Currencies["CHF"], Decimal('50'), Date(2018, 12, 25))
    result = price.scalar_add(Decimal('25'))
    assert result.ccy == Currencies["CHF"]
    assert result.dov == Date(2018, 12, 25)
    
    # Test 9: scalar_add with integer
    price = Price.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    result = price.scalar_add(5)
    assert result.defined
    assert result.qty == Decimal('105')
    
    # Test 10: scalar_add resulting in negative quantity
    price = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = price.scalar_add(Decimal('-20'))
    assert result.defined
    assert result.qty == Decimal('-10')


# LLM-generated content at query #4
#--------------------------

```python
def test_Money_add():
    """Unit tests for Money.add() method"""
    from pypara.currencies import Currencies
    from datetime import date as Date
    
    # Test 1: Adding two defined money objects with same currency
    money1 = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('50'), Date(2019, 1, 2))
    result = money1.add(money2)
    assert result.qty == Decimal('150.00')
    assert result.ccy.code == 'USD'
    
    # Test 2: Adding defined money with undefined money
    money_defined = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    money_undefined = Money.na()
    result = money_defined.add(money_undefined)
    assert result is money_defined
    
    # Test 3: Adding undefined money with defined money
    money_undefined = Money.na()
    money_defined = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    result = money_undefined.add(money_defined)
    assert result is money_defined
    
    # Test 4: Adding two undefined money objects
    money_undefined1 = Money.na()
    money_undefined2 = Money.na()
    result = money_undefined1.add(money_undefined2)
    assert result.undefined
    
    # Test 5: Adding money with different currencies raises error
    money_usd = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    money_eur = Money.of(Currencies["EUR"], Decimal('50'), Date(2019, 1, 1))
    with pytest.raises(IncompatibleCurrencyError):
        money_usd.add(money_eur)
    
    # Test 6: Date is carried forward from addition
    money1 = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('50'), Date(2019, 1, 10))
    result = money1.add(money2)
    assert result.dov == Date(2019, 1, 10)
    
    # Test 7: Adding negative quantities
    money1 = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('-30'), Date(2019, 1, 1))
    result = money1.add(money2)
    assert result.qty == Decimal('70.00')
    
    # Test 8: Adding decimal quantities with precision
    money1 = Money.of(Currencies["USD"], Decimal('10.25'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('20.75'), Date(2019, 1, 1))
    result = money1.add(money2)
    assert result.qty == Decimal('31.00')


# LLM-generated content at query #5
#--------------------------

```python
def test_Money_floor_divide():
    from decimal import Decimal
    from datetime import date as Date
    from pypara.currencies import Currencies
    
    # Test floor division with defined money
    money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = money.floor_divide(Decimal('3'))
    assert result.qty == Decimal('3')
    assert result.ccy == Currencies["USD"]
    assert result.dov == Date(2019, 1, 1)
    assert result.defined
    
    # Test floor division resulting in zero
    money = Money.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    result = money.floor_divide(Decimal('5'))
    assert result.qty == Decimal('0')
    
    # Test floor division with negative divisor
    money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = money.floor_divide(Decimal('-3'))
    assert result.qty == Decimal('-4')
    
    # Test floor division with negative dividend
    money = Money.of(Currencies["USD"], Decimal('-10'), Date(2019, 1, 1))
    result = money.floor_divide(Decimal('3'))
    assert result.qty == Decimal('-4')
    
    # Test floor division by zero yields undefined money
    money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = money.floor_divide(Decimal('0'))
    assert result.undefined
    
    # Test floor division with undefined money returns itself
    undefined_money = Money.na()
    result = undefined_money.floor_divide(Decimal('5'))
    assert result is undefined_money
    assert result.undefined
    
    # Test floor division with decimal divisor
    money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = money.floor_divide(Decimal('2.5'))
    assert result.qty == Decimal('4')
    
    # Test floor division with one as divisor
    money = Money.of(Currencies["USD"], Decimal('7.50'), Date(2019, 1, 1))
    result = money.floor_divide(Decimal('1'))
    assert result.qty == Decimal('7.50')


# LLM-generated content at query #6
#--------------------------

```python
def test_SomePrice_qty_or_none():
    from pypara.currencies import Currencies
    from datetime import date as Date
    
    # Test with defined price - should return the qty
    someprice = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert someprice.qty_or_none() == Decimal('1')
    
    # Test with undefined price (None ccy) - should return None
    noneprice = Price.of(None, Decimal('1'), None)
    assert noneprice.qty_or_none() is None
    
    # Test with undefined price (None qty) - should return None
    noneprice2 = Price.of(Currencies["USD"], None, Date(2019, 1, 1))
    assert noneprice2.qty_or_none() is None
    
    # Test with undefined price (None dov) - should return None
    noneprice3 = Price.of(Currencies["USD"], Decimal('1'), None)
    assert noneprice3.qty_or_none() is None
    
    # Test with different decimal values
    someprice2 = Price.of(Currencies["EUR"], Decimal('42.5'), Date(2020, 6, 15))
    assert someprice2.qty_or_none() == Decimal('42.5')
    
    # Test with zero quantity
    someprice3 = Price.of(Currencies["GBP"], Decimal('0'), Date(2021, 3, 30))
    assert someprice3.qty_or_none() == Decimal('0')
    
    # Test with negative quantity
    someprice4 = Price.of(Currencies["JPY"], Decimal('-100'), Date(2022, 12, 25))
    assert someprice4.qty_or_none() == Decimal('-100')


# LLM-generated content at query #7
#--------------------------

```python
def test_Price_divide():
    """Test the divide method of Price class."""
    from decimal import Decimal
    from datetime import date as Date
    from pypara.currencies import Currencies
    
    # Test divide on defined price
    price = Price.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    result = price.divide(Decimal('2'))
    assert result.defined
    assert result.qty == Decimal('50')
    assert result.ccy == Currencies["USD"]
    assert result.dov == Date(2019, 1, 1)
    
    # Test divide by zero yields undefined price
    result_zero = price.divide(Decimal('0'))
    assert result_zero.undefined
    
    # Test divide with decimal result
    price2 = Price.of(Currencies["EUR"], Decimal('10'), Date(2019, 1, 1))
    result2 = price2.divide(Decimal('3'))
    assert result2.defined
    assert result2.qty == Decimal('10') / Decimal('3')
    
    # Test divide on undefined price returns itself
    undefined_price = Price.na()
    result_undefined = undefined_price.divide(Decimal('5'))
    assert result_undefined.undefined
    
    # Test divide with integer
    price3 = Price.of(Currencies["GBP"], Decimal('50'), Date(2019, 1, 1))
    result3 = price3.divide(5)
    assert result3.defined
    assert result3.qty == Decimal('10')
    
    # Test divide with float
    price4 = Price.of(Currencies["JPY"], Decimal('100'), Date(2019, 1, 1))
    result4 = price4.divide(2.5)
    assert result4.defined
    assert result4.qty == Decimal('100') / Decimal('2.5')
    
    # Test division operator
    price5 = Price.of(Currencies["USD"], Decimal('200'), Date(2019, 1, 1))
    result5 = price5 / Decimal('4')
    assert result5.defined
    assert result5.qty == Decimal('50')


# LLM-generated content at query #8
#--------------------------

```python
def test_SomePrice___pos__():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    
    # Test positive of a positive price
    price = SomePrice(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    result = +price
    assert isinstance(result, SomePrice)
    assert result.ccy == Currencies["USD"]
    assert result.qty == Decimal('100')
    assert result.dov == Date(2019, 1, 1)
    
    # Test positive of a negative price
    price_neg = SomePrice(Currencies["USD"], Decimal('-50'), Date(2019, 1, 1))
    result_neg = +price_neg
    assert isinstance(result_neg, SomePrice)
    assert result_neg.ccy == Currencies["USD"]
    assert result_neg.qty == Decimal('-50')
    assert result_neg.dov == Date(2019, 1, 1)
    
    # Test positive of zero price
    price_zero = SomePrice(Currencies["USD"], Decimal('0'), Date(2019, 1, 1))
    result_zero = +price_zero
    assert isinstance(result_zero, SomePrice)
    assert result_zero.ccy == Currencies["USD"]
    assert result_zero.qty == Decimal('0')
    assert result_zero.dov == Date(2019, 1, 1)
    
    # Test that positive returns a new instance
    price_orig = SomePrice(Currencies["EUR"], Decimal('75.5'), Date(2020, 6, 15))
    result_orig = +price_orig
    assert result_orig is not price_orig
    assert result_orig == price_orig


# LLM-generated content at query #9
#--------------------------

```python
def test_Price_lt():
    """Unit tests for Price.lt() method"""
    from pypara.currencies import Currencies
    from datetime import date as Date
    
    # Test 1: Defined price less than another defined price with same currency
    price1 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    assert price1.lt(price2) is True
    assert price2.lt(price1) is False
    
    # Test 2: Equal defined prices
    price3 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    price4 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert price3.lt(price4) is False
    
    # Test 3: Undefined price is always less than defined price
    undefined = Price.na()
    defined = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert undefined.lt(defined) is True
    assert defined.lt(undefined) is False
    
    # Test 4: Two undefined prices
    undefined1 = Price.na()
    undefined2 = Price.na()
    assert undefined1.lt(undefined2) is False
    
    # Test 5: Incompatible currencies should raise IncompatibleCurrencyError
    price_usd = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    price_eur = Price.of(Currencies["EUR"], Decimal('2'), Date(2019, 1, 1))
    with pytest.raises(IncompatibleCurrencyError):
        price_usd.lt(price_eur)
    
    # Test 6: Negative quantities
    price_neg = Price.of(Currencies["USD"], Decimal('-5'), Date(2019, 1, 1))
    price_pos = Price.of(Currencies["USD"], Decimal('5'), Date(2019, 1, 1))
    assert price_neg.lt(price_pos) is True
    assert price_pos.lt(price_neg) is False
    
    # Test 7: Different dates with same currency and quantity
    price_date1 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    price_date2 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 2))
    assert price_date1.lt(price_date2) is False
    
    # Test 8: Zero quantity
    price_zero = Price.of(Currencies["USD"], Decimal('0'), Date(2019, 1, 1))
    price_one = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert price_zero.lt(price_one) is True
    assert price_one.lt(price_zero) is False


# LLM-generated content at query #10
#--------------------------

```python
def test_SomePrice_dov_or_none():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    # Test case 1: defined price returns the dov
    someprice = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert someprice.dov_or_none() == Date(2019, 1, 1)
    
    # Test case 2: undefined price with None dov returns None
    noneprice = Price.of(None, None, Date(2019, 1, 1))
    assert noneprice.dov_or_none() is None
    
    # Test case 3: undefined price with None ccy returns None
    noneprice2 = Price.of(None, Decimal('1'), Date(2019, 1, 1))
    assert noneprice2.dov_or_none() is None
    
    # Test case 4: undefined price with None qty returns None
    noneprice3 = Price.of(Currencies["USD"], None, Date(2019, 1, 1))
    assert noneprice3.dov_or_none() is None
    
    # Test case 5: completely undefined price returns None
    undefined_price = Price.na()
    assert undefined_price.dov_or_none() is None
    
    # Test case 6: defined price with different date
    someprice2 = Price.of(Currencies["EUR"], Decimal('100'), Date(2020, 12, 31))
    assert someprice2.dov_or_none() == Date(2020, 12, 31)


# LLM-generated content at query #11
#--------------------------

```python
def test_Price_qty_or():
    """Unit tests for Price.qty_or method."""
    from pypara.currencies import Currencies
    from datetime import date as Date
    
    # Test with defined price - should return the quantity
    defined_price = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert defined_price.qty_or(Decimal(0)) == Decimal('1')
    assert defined_price.qty_or(Decimal(999)) == Decimal('1')
    
    # Test with undefined price (None qty) - should return default
    undefined_price_qty = Price.of(None, Decimal('1'), None)
    assert undefined_price_qty.qty_or(Decimal(0)) == Decimal('0')
    assert undefined_price_qty.qty_or(Decimal(42)) == Decimal('42')
    
    # Test with undefined price (None ccy) - should return default
    undefined_price_ccy = Price.of(None, Decimal('5'), Date(2019, 1, 1))
    assert undefined_price_ccy.qty_or(Decimal(0)) == Decimal('0')
    assert undefined_price_ccy.qty_or(Decimal(100)) == Decimal('100')
    
    # Test with undefined price (None dov) - should return default
    undefined_price_dov = Price.of(Currencies["EUR"], Decimal('3'), None)
    assert undefined_price_dov.qty_or(Decimal(0)) == Decimal('0')
    assert undefined_price_dov.qty_or(Decimal(7)) == Decimal('7')
    
    # Test with all None - should return default
    all_none_price = Price.of(None, None, None)
    assert all_none_price.qty_or(Decimal(0)) == Decimal('0')
    assert all_none_price.qty_or(Decimal(55)) == Decimal('55')
    
    # Test with various decimal values
    price_with_decimal = Price.of(Currencies["GBP"], Decimal('123.456'), Date(2020, 6, 15))
    assert price_with_decimal.qty_or(Decimal(0)) == Decimal('123.456')
    
    # Test with negative quantity
    negative_price = Price.of(Currencies["JPY"], Decimal('-50'), Date(2021, 3, 10))
    assert negative_price.qty_or(Decimal(0)) == Decimal('-50')
    
    # Test with zero quantity
    zero_price = Price.of(Currencies["CHF"], Decimal('0'), Date(2022, 12, 25))
    assert zero_price.qty_or(Decimal(100)) == Decimal('0')
    
    # Test with NA price
    na_price = Price.na()
    assert na_price.qty_or(Decimal(42)) == Decimal('42')


# LLM-generated content at query #12
#--------------------------

```python
def test_Price___truediv__():
    from decimal import Decimal
    from datetime import date as Date
    from pypara.currencies import Currencies
    
    # Test division of defined price by a numeric value
    price = Price.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    result = price / Decimal('2')
    assert result.defined
    assert result.qty == Decimal('50')
    assert result.ccy == Currencies["USD"]
    assert result.dov == Date(2019, 1, 1)
    
    # Test division by integer
    price = Price.of(Currencies["EUR"], Decimal('10'), Date(2019, 1, 1))
    result = price / 5
    assert result.defined
    assert result.qty == Decimal('2')
    
    # Test division by float
    price = Price.of(Currencies["GBP"], Decimal('20'), Date(2019, 1, 1))
    result = price / 2.0
    assert result.defined
    assert result.qty == Decimal('10')
    
    # Test division of undefined price returns undefined
    undefined_price = Price.na()
    result = undefined_price / Decimal('2')
    assert result.undefined
    
    # Test division by zero yields undefined price
    price = Price.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    result = price / Decimal('0')
    assert result.undefined
    
    # Test division by zero (int) yields undefined price
    price = Price.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    result = price / 0
    assert result.undefined
    
    # Test division with decimal result
    price = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = price / Decimal('3')
    assert result.defined
    assert result.qty == Decimal('10') / Decimal('3')
    
    # Test division preserves currency and date
    price = Price.of(Currencies["JPY"], Decimal('1000'), Date(2020, 6, 15))
    result = price / Decimal('10')
    assert result.ccy == Currencies["JPY"]
    assert result.dov == Date(2020, 6, 15)
    assert result.qty == Decimal('100')


# LLM-generated content at query #13
#--------------------------

```python
def test_Price___abs__():
    """Test __abs__ method of Price class."""
    from pypara.currencies import Currencies
    from datetime import date as Date
    
    # Test abs() on defined price with positive quantity
    positive_price = Price.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    abs_positive = abs(positive_price)
    assert abs_positive.qty == Decimal('100')
    assert abs_positive.ccy == Currencies["USD"]
    assert abs_positive.dov == Date(2019, 1, 1)
    
    # Test abs() on defined price with negative quantity
    negative_price = Price.of(Currencies["USD"], Decimal('-100'), Date(2019, 1, 1))
    abs_negative = abs(negative_price)
    assert abs_negative.qty == Decimal('100')
    assert abs_negative.ccy == Currencies["USD"]
    assert abs_negative.dov == Date(2019, 1, 1)
    
    # Test abs() on defined price with zero quantity
    zero_price = Price.of(Currencies["USD"], Decimal('0'), Date(2019, 1, 1))
    abs_zero = abs(zero_price)
    assert abs_zero.qty == Decimal('0')
    assert abs_zero.ccy == Currencies["USD"]
    
    # Test abs() on undefined price
    undefined_price = Price.na()
    abs_undefined = abs(undefined_price)
    assert abs_undefined.undefined
    assert abs_undefined is undefined_price or abs_undefined == undefined_price
    
    # Test abs() with different currencies
    eur_price = Price.of(Currencies["EUR"], Decimal('-50.5'), Date(2020, 6, 15))
    abs_eur = abs(eur_price)
    assert abs_eur.qty == Decimal('50.5')
    assert abs_eur.ccy == Currencies["EUR"]
    assert abs_eur.dov == Date(2020, 6, 15)


# LLM-generated content at query #14
#--------------------------

```python
def test_Price_times():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    # Test times with defined price
    price = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = price.times(Decimal('2'))
    assert result.defined
    assert result.ccy == Currencies["USD"]
    assert result.qty == Decimal('20')
    assert result.dov == Date(2019, 1, 1)
    
    # Test times with integer
    result = price.times(3)
    assert result.defined
    assert result.qty == Decimal('30')
    
    # Test times with float
    result = price.times(1.5)
    assert result.defined
    assert result.qty == Decimal('15')
    
    # Test times with zero
    result = price.times(0)
    assert result.defined
    assert result.qty == Decimal('0')
    
    # Test times with negative number
    result = price.times(Decimal('-2'))
    assert result.defined
    assert result.qty == Decimal('-20')
    
    # Test times with undefined price (should return undefined)
    undefined_price = Price.na()
    result = undefined_price.times(Decimal('5'))
    assert result.undefined
    
    # Test times returns Money type
    result = price.times(Decimal('2'))
    assert isinstance(result, Money)


# LLM-generated content at query #15
#--------------------------

```python
def test_Money___sub__():
    from pypara.currencies import Currencies
    from datetime import date as Date
    
    # Test subtraction of two defined money objects with same currency
    money1 = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('30'), Date(2019, 1, 1))
    result = money1 - money2
    assert result.defined
    assert result.qty == Decimal('70.00')
    assert result.ccy.code == "USD"
    
    # Test subtraction with incompatible currencies raises error
    money_usd = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    money_eur = Money.of(Currencies["EUR"], Decimal('50'), Date(2019, 1, 1))
    with pytest.raises(IncompatibleCurrencyError):
        money_usd - money_eur
    
    # Test subtraction with undefined money (left operand undefined)
    undefined_money = Money.na()
    money_defined = Money.of(Currencies["USD"], Decimal('50'), Date(2019, 1, 1))
    result = undefined_money - money_defined
    assert result is money_defined
    
    # Test subtraction with undefined money (right operand undefined)
    result = money_defined - undefined_money
    assert result is money_defined
    
    # Test subtraction of two undefined money objects
    result = undefined_money - undefined_money
    assert result is undefined_money
    
    # Test subtraction resulting in negative quantity
    money_small = Money.of(Currencies["USD"], Decimal('20'), Date(2019, 1, 1))
    money_large = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    result = money_small - money_large
    assert result.defined
    assert result.qty == Decimal('-80.00')
    
    # Test subtraction with zero
    money_zero = Money.of(Currencies["USD"], Decimal('0'), Date(2019, 1, 1))
    result = money_defined - money_zero
    assert result.qty == Decimal('50.00')
    
    # Test dates are carried forward
    money_a = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    money_b = Money.of(Currencies["USD"], Decimal('30'), Date(2019, 1, 2))
    result = money_a - money_b
    assert result.dov == Date(2019, 1, 1)


# LLM-generated content at query #16
#--------------------------

```python
def test_Money___pos__():
    """Test the __pos__ method of Money class."""
    from pypara.currencies import Currencies
    from datetime import date as Date
    
    # Test with defined money (positive quantity)
    money_positive = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    result_positive = +money_positive
    assert result_positive.qty == Decimal('100.00')
    assert result_positive.ccy == Currencies["USD"]
    assert result_positive.dov == Date(2019, 1, 1)
    
    # Test with defined money (negative quantity)
    money_negative = Money.of(Currencies["USD"], Decimal('-50'), Date(2019, 1, 1))
    result_negative = +money_negative
    assert result_negative.qty == Decimal('-50.00')
    assert result_negative.ccy == Currencies["USD"]
    
    # Test with undefined money
    money_undefined = Money.na()
    result_undefined = +money_undefined
    assert result_undefined.undefined
    assert result_undefined is money_undefined or result_undefined.undefined
    
    # Test with zero quantity
    money_zero = Money.of(Currencies["EUR"], Decimal('0'), Date(2020, 6, 15))
    result_zero = +money_zero
    assert result_zero.qty == Decimal('0.00')
    assert result_zero.ccy == Currencies["EUR"]
    
    # Test that positive returns same monetary value if defined
    money_test = Money.of(Currencies["GBP"], Decimal('25.50'), Date(2021, 3, 10))
    result_test = +money_test
    assert result_test.defined
    assert result_test.qty == money_test.qty
    assert result_test.ccy == money_test.ccy
    assert result_test.dov == money_test.dov


# LLM-generated content at query #17
#--------------------------

```python
def test_Price_negative():
    """Test the negative method of Price class"""
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    # Test with defined price - positive quantity
    positive_price = Price.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    negative_result = positive_price.negative()
    assert negative_result.defined
    assert negative_result.qty == Decimal('-100')
    assert negative_result.ccy == Currencies["USD"]
    assert negative_result.dov == Date(2019, 1, 1)
    
    # Test with defined price - negative quantity
    negative_price = Price.of(Currencies["EUR"], Decimal('-50'), Date(2019, 1, 15))
    positive_result = negative_price.negative()
    assert positive_result.defined
    assert positive_result.qty == Decimal('50')
    assert positive_result.ccy == Currencies["EUR"]
    assert positive_result.dov == Date(2019, 1, 15)
    
    # Test with defined price - zero quantity
    zero_price = Price.of(Currencies["GBP"], Decimal('0'), Date(2019, 2, 1))
    zero_result = zero_price.negative()
    assert zero_result.defined
    assert zero_result.qty == Decimal('0')
    assert zero_result.ccy == Currencies["GBP"]
    
    # Test with undefined price - should return itself
    undefined_price = Price.na()
    undefined_result = undefined_price.negative()
    assert undefined_result.undefined
    assert undefined_result is Price.na() or undefined_result.undefined
    
    # Test with undefined price created via of() with None values
    undefined_price2 = Price.of(None, Decimal('100'), None)
    undefined_result2 = undefined_price2.negative()
    assert undefined_result2.undefined


# LLM-generated content at query #18
#--------------------------

```python
def test_Money_as_float():
    """Test as_float method of Money class"""
    from decimal import Decimal
    from datetime import date as Date
    from pypara.currencies import Currencies
    
    # Test with defined money - should return float value
    money_defined = Money.of(Currencies["USD"], Decimal('123.45'), Date(2019, 1, 1))
    result = money_defined.as_float()
    assert isinstance(result, float)
    assert result == 123.45
    
    # Test with defined money - negative value
    money_negative = Money.of(Currencies["EUR"], Decimal('-456.78'), Date(2019, 1, 2))
    result = money_negative.as_float()
    assert isinstance(result, float)
    assert result == -456.78
    
    # Test with defined money - zero
    money_zero = Money.of(Currencies["GBP"], Decimal('0'), Date(2019, 1, 3))
    result = money_zero.as_float()
    assert isinstance(result, float)
    assert result == 0.0
    
    # Test with defined money - large number
    money_large = Money.of(Currencies["JPY"], Decimal('999999.99'), Date(2019, 1, 4))
    result = money_large.as_float()
    assert isinstance(result, float)
    assert result == 999999.99
    
    # Test with undefined money - should raise MonetaryOperationException
    money_undefined = Money.na()
    with pytest.raises(MonetaryOperationException):
        money_undefined.as_float()
    
    # Test with undefined money created via Money.of with None qty
    money_none_qty = Money.of(None, Decimal('100'), None)
    with pytest.raises(MonetaryOperationException):
        money_none_qty.as_float()
    
    # Test with defined money - decimal precision
    money_precise = Money.of(Currencies["USD"], Decimal('0.01'), Date(2019, 1, 5))
    result = money_precise.as_float()
    assert isinstance(result, float)
    assert abs(result - 0.01) < 1e-10


# LLM-generated content at query #19
#--------------------------

```python
def test_Price_ccy_or():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    # Test with defined price - should return the price's currency
    defined_price = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    result = defined_price.ccy_or(Currencies["EUR"])
    assert result == Currencies["USD"]
    assert result.code == "USD"
    
    # Test with undefined price - should return the default currency
    undefined_price = Price.of(None, Decimal('1'), None)
    result = undefined_price.ccy_or(Currencies["EUR"])
    assert result == Currencies["EUR"]
    assert result.code == "EUR"
    
    # Test with different default currency
    defined_price = Price.of(Currencies["GBP"], Decimal('100'), Date(2020, 5, 15))
    result = defined_price.ccy_or(Currencies["JPY"])
    assert result == Currencies["GBP"]
    assert result.code == "GBP"
    
    # Test with na() instance - should return default
    na_price = Price.na()
    result = na_price.ccy_or(Currencies["USD"])
    assert result == Currencies["USD"]


# LLM-generated content at query #20
#--------------------------

```python
def test_Money___sub__():
    """Unit tests for Money.__sub__ method"""
    from pypara.currencies import Currencies
    from datetime import date as Date
    
    # Test 1: Subtract two defined money objects with same currency
    money1 = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('30'), Date(2019, 1, 1))
    result = money1 - money2
    assert result.qty == Decimal('70.00')
    assert result.ccy.code == 'USD'
    
    # Test 2: Subtract with different currencies should raise IncompatibleCurrencyError
    money_usd = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    money_eur = Money.of(Currencies["EUR"], Decimal('50'), Date(2019, 1, 1))
    with pytest.raises(IncompatibleCurrencyError):
        money_usd - money_eur
    
    # Test 3: Subtract undefined money from defined money
    money_defined = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    money_undefined = Money.na()
    result = money_defined - money_undefined
    assert result is money_defined
    
    # Test 4: Subtract defined money from undefined money
    money_undefined = Money.na()
    money_defined = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    result = money_undefined - money_defined
    assert result is money_defined
    
    # Test 5: Subtract two undefined money objects
    money_undefined1 = Money.na()
    money_undefined2 = Money.na()
    result = money_undefined1 - money_undefined2
    assert result.undefined
    
    # Test 6: Subtract resulting in negative value
    money_small = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    money_large = Money.of(Currencies["USD"], Decimal('50'), Date(2019, 1, 1))
    result = money_small - money_large
    assert result.qty == Decimal('-40.00')
    
    # Test 7: Subtract resulting in zero
    money1 = Money.of(Currencies["USD"], Decimal('50'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('50'), Date(2019, 1, 1))
    result = money1 - money2
    assert result.qty == Decimal('0.00')
    
    # Test 8: Date is carried forward from subtraction
    money1 = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('30'), Date(2019, 1, 2))
    result = money1 - money2
    assert result.dov == Date(2019, 1, 1)


