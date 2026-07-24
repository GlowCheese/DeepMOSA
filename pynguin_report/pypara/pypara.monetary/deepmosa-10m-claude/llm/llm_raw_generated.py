####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_qty_or_none():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    from pypara.price import Price
    
    someprice = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    result_defined = someprice.qty_or_none()
    assert result_defined == Decimal('1')
    
    noneprice = Price.of(None, Decimal('1'), None)
    result_undefined = noneprice.qty_or_none()
    assert result_undefined is None


# LLM-generated content at query #2
#--------------------------

```python
def test_scalar_subtract():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    # Test scalar subtraction on defined price
    price = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = price.scalar_subtract(Decimal('3'))
    assert result.qty_or_zero() == Decimal('7')
    assert result.ccy_or_none().code == "USD"
    
    # Test scalar subtraction with negative number
    price = Price.of(Currencies["EUR"], Decimal('5'), Date(2019, 1, 1))
    result = price.scalar_subtract(Decimal('-2'))
    assert result.qty_or_zero() == Decimal('7')
    
    # Test scalar subtraction on undefined price returns itself
    undefined_price = Price.na()
    result = undefined_price.scalar_subtract(Decimal('5'))
    assert result.undefined
    
    # Test scalar subtraction resulting in zero
    price = Price.of(Currencies["GBP"], Decimal('5'), Date(2019, 1, 1))
    result = price.scalar_subtract(Decimal('5'))
    assert result.qty_or_zero() == Decimal('0')
    
    # Test scalar subtraction resulting in negative quantity
    price = Price.of(Currencies["JPY"], Decimal('2'), Date(2019, 1, 1))
    result = price.scalar_subtract(Decimal('7'))
    assert result.qty_or_zero() == Decimal('-5')


# LLM-generated content at query #3
#--------------------------

```python
def test_dov_or():
    from datetime import date
    from decimal import Decimal
    
    # Create a mock Currency object
    class MockCurrency:
        def __init__(self, code):
            self.code = code
    
    ccy = MockCurrency("USD")
    qty = Decimal("100.00")
    dov = date(2023, 1, 15)
    
    some_price = SomePrice(ccy, qty, dov)
    
    # Test dov_or returns the price's dov when called with a default
    default_date = date(2020, 1, 1)
    result = some_price.dov_or(default_date)
    
    assert result == dov
    assert result == date(2023, 1, 15)
    
    # Test dov_or returns the price's dov even with different default
    another_default = date(2025, 12, 31)
    result2 = some_price.dov_or(another_default)
    
    assert result2 == dov
    assert result2 != another_default


# LLM-generated content at query #4
#--------------------------

```python
def test_fmap_with_defined_price():
    from pypara.currencies import Currencies
    from datetime import date, timedelta
    from decimal import Decimal
    from pypara.price import Price
    
    someprice = Price.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    new = someprice.fmap(lambda x: Price.of(x.ccy, x.qty + Decimal('1'), x.dov + timedelta(days=10)))
    
    assert new.ccy.code == 'USD'
    assert new.qty == Decimal('2')
    assert new.dov == date(2019, 1, 11)


def test_fmap_with_undefined_price():
    from pypara.price import Price
    from decimal import Decimal
    
    noneprice = Price.of(None, Decimal('1'), None)
    result = noneprice.fmap(lambda sp: Price.of(sp.ccy, sp.qty + Decimal('1'), sp.dov))
    
    assert result.undefined is True


def test_fmap_function_receives_correct_price():
    from pypara.currencies import Currencies
    from datetime import date
    from decimal import Decimal
    from pypara.price import Price
    
    someprice = Price.of(Currencies["USD"], Decimal('5'), date(2020, 6, 15))
    received_price = None
    
    def capture_price(x):
        nonlocal received_price
        received_price = x
        return Price.of(x.ccy, x.qty, x.dov)
    
    someprice.fmap(capture_price)
    
    assert received_price.ccy.code == 'USD'
    assert received_price.qty == Decimal('5')
    assert received_price.dov == date(2020, 6, 15)


def test_fmap_returns_result_of_function():
    from pypara.currencies import Currencies
    from datetime import date
    from decimal import Decimal
    from pypara.price import Price
    
    someprice = Price.of(Currencies["USD"], Decimal('10'), date(2019, 1, 1))
    expected_result = Price.of(Currencies["EUR"], Decimal('20'), date(2019, 1, 1))
    
    result = someprice.fmap(lambda x: expected_result)
    
    assert result is expected_result


# LLM-generated content at query #5
#--------------------------

```python
def test_round_defined_price():
    from decimal import Decimal
    from datetime import date as Date
    from pypara.currencies import Currencies
    from pypara.price import Price
    
    price = Price.of(Currencies["USD"], Decimal('1.456'), Date(2019, 1, 1))
    rounded_price = price.round(2)
    
    assert rounded_price.qty_or_zero() == Decimal('1.46')
    assert rounded_price.ccy_or_none().code == 'USD'
    assert rounded_price.dov_or_none() == Date(2019, 1, 1)


def test_round_undefined_price():
    from decimal import Decimal
    from pypara.price import Price
    
    undefined_price = Price.na()
    rounded_price = undefined_price.round(2)
    
    assert rounded_price.undefined
    assert rounded_price is undefined_price


def test_round_zero_digits():
    from decimal import Decimal
    from datetime import date as Date
    from pypara.currencies import Currencies
    from pypara.price import Price
    
    price = Price.of(Currencies["USD"], Decimal('1.567'), Date(2019, 1, 1))
    rounded_price = price.round(0)
    
    assert rounded_price.qty_or_zero() == Decimal('2')


def test_round_negative_quantity():
    from decimal import Decimal
    from datetime import date as Date
    from pypara.currencies import Currencies
    from pypara.price import Price
    
    price = Price.of(Currencies["USD"], Decimal('-1.456'), Date(2019, 1, 1))
    rounded_price = price.round(2)
    
    assert rounded_price.qty_or_zero() == Decimal('-1.46')


def test_round_half_even():
    from decimal import Decimal
    from datetime import date as Date
    from pypara.currencies import Currencies
    from pypara.price import Price
    
    price = Price.of(Currencies["USD"], Decimal('1.225'), Date(2019, 1, 1))
    rounded_price = price.round(2)
    
    assert rounded_price.qty_or_zero() == Decimal('1.22')


def test_round_large_ndigits():
    from decimal import Decimal
    from datetime import date as Date
    from pypara.currencies import Currencies
    from pypara.price import Price
    
    price = Price.of(Currencies["USD"], Decimal('1.123456789'), Date(2019, 1, 1))
    rounded_price = price.round(5)
    
    assert rounded_price.qty_or_zero() == Decimal('1.12346')


# LLM-generated content at query #6
#--------------------------

```python
def test_scalar_add_with_defined_money():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money = Money.of(Currencies["USD"], Decimal('10.00'), Date(2019, 1, 1))
    result = money.scalar_add(Decimal('5.00'))
    
    assert result.qty == Decimal('15.00')
    assert result.ccy.code == 'USD'
    assert result.dov == Date(2019, 1, 1)


def test_scalar_add_with_undefined_money():
    from decimal import Decimal
    
    money = Money.na()
    result = money.scalar_add(Decimal('5.00'))
    
    assert result.undefined


def test_scalar_add_with_negative_scalar():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money = Money.of(Currencies["USD"], Decimal('10.00'), Date(2019, 1, 1))
    result = money.scalar_add(Decimal('-3.00'))
    
    assert result.qty == Decimal('7.00')


def test_scalar_add_with_zero_scalar():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money = Money.of(Currencies["USD"], Decimal('10.00'), Date(2019, 1, 1))
    result = money.scalar_add(Decimal('0.00'))
    
    assert result.qty == Decimal('10.00')


def test_scalar_add_with_large_scalar():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money = Money.of(Currencies["USD"], Decimal('100.50'), Date(2019, 1, 1))
    result = money.scalar_add(Decimal('999999.99'))
    
    assert result.qty == Decimal('1000100.49')


# LLM-generated content at query #7
#--------------------------

```python
def test_is_equal_with_same_defined_money():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    
    money1 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert money1.is_equal(money2) is True


def test_is_equal_with_different_quantity():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    
    money1 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    assert money1.is_equal(money2) is False


def test_is_equal_with_different_currency():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    
    money1 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["EUR"], Decimal('1'), Date(2019, 1, 1))
    assert money1.is_equal(money2) is False


def test_is_equal_with_different_date():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    
    money1 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 2))
    assert money1.is_equal(money2) is False


def test_is_equal_with_undefined_money():
    money1 = Money.na()
    money2 = Money.na()
    assert money1.is_equal(money2) is True


def test_is_equal_with_defined_and_undefined():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    
    money1 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    money2 = Money.na()
    assert money1.is_equal(money2) is False


def test_is_equal_with_non_money_object():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    
    money = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert money.is_equal("not a money object") is False


def test_is_equal_with_none():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    
    money = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert money.is_equal(None) is False


# LLM-generated content at query #8
#--------------------------

```python
def test_someprice_sub():
    from decimal import Decimal
    from datetime import date
    
    # Create test currencies and prices
    usd = Currency(code="USD", quantizer=Decimal("0.01"))
    eur = Currency(code="EUR", quantizer=Decimal("0.01"))
    
    price1 = SomePrice(ccy=usd, qty=Decimal("100.00"), dov=date(2024, 1, 1))
    price2 = SomePrice(ccy=usd, qty=Decimal("30.00"), dov=date(2024, 1, 1))
    price3 = SomePrice(ccy=eur, qty=Decimal("50.00"), dov=date(2024, 1, 1))
    
    # Test subtraction of two prices with same currency
    result = price1 - price2
    assert result.ccy == usd
    assert result.qty == Decimal("70.00")
    assert result.dov == date(2024, 1, 1)
    
    # Test subtraction with different currencies raises error
    try:
        price1 - price3
        assert False, "Should raise IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass
    
    # Test subtraction with undefined price returns self
    no_price = NoPrice
    result = price1 - no_price
    assert result == price1
    
    # Test subtraction with different dates (should use the later date)
    price4 = SomePrice(ccy=usd, qty=Decimal("20.00"), dov=date(2024, 1, 15))
    result = price1 - price4
    assert result.ccy == usd
    assert result.qty == Decimal("80.00")
    assert result.dov == date(2024, 1, 15)


# LLM-generated content at query #9
#--------------------------

```python
def test_money_floordiv():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    
    # Test floor division with defined money
    money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = money.__floordiv__(Decimal('3'))
    assert result.qty == Decimal('3.00')
    assert result.ccy == Currencies["USD"]
    
    # Test floor division resulting in zero
    money = Money.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    result = money.__floordiv__(Decimal('5'))
    assert result.qty == Decimal('0.00')
    
    # Test floor division with negative divisor
    money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = money.__floordiv__(Decimal('-3'))
    assert result.qty == Decimal('-4.00')
    
    # Test floor division with undefined money
    undefined_money = Money.na()
    result = undefined_money.__floordiv__(Decimal('5'))
    assert result.undefined
    
    # Test floor division by zero yields undefined
    money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = money.__floordiv__(Decimal('0'))
    assert result.undefined
    
    # Test floor division with negative money
    money = Money.of(Currencies["USD"], Decimal('-10'), Date(2019, 1, 1))
    result = money.__floordiv__(Decimal('3'))
    assert result.qty == Decimal('-4.00')


# LLM-generated content at query #10
#--------------------------

```python
def test_abs_defined_positive_money():
    from decimal import Decimal
    from datetime import date as Date
    from pypara.currencies import Currencies
    from pypara.money import Money
    
    money = Money.of(Currencies["USD"], Decimal('10.50'), Date(2019, 1, 1))
    result = money.abs()
    
    assert result.qty == Decimal('10.50')
    assert result.ccy.code == 'USD'
    assert result.dov == Date(2019, 1, 1)


def test_abs_defined_negative_money():
    from decimal import Decimal
    from datetime import date as Date
    from pypara.currencies import Currencies
    from pypara.money import Money
    
    money = Money.of(Currencies["USD"], Decimal('-10.50'), Date(2019, 1, 1))
    result = money.abs()
    
    assert result.qty == Decimal('10.50')
    assert result.ccy.code == 'USD'
    assert result.dov == Date(2019, 1, 1)


def test_abs_defined_zero_money():
    from decimal import Decimal
    from datetime import date as Date
    from pypara.currencies import Currencies
    from pypara.money import Money
    
    money = Money.of(Currencies["EUR"], Decimal('0'), Date(2020, 6, 15))
    result = money.abs()
    
    assert result.qty == Decimal('0.00')
    assert result.ccy.code == 'EUR'


def test_abs_undefined_money():
    from pypara.money import Money
    
    money = Money.na()
    result = money.abs()
    
    assert result is money
    assert result.undefined is True


# LLM-generated content at query #11
#--------------------------

```python
def test_money_truediv_with_defined_money():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.money import Money
    
    money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = money.__truediv__(Decimal('2'))
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('5.00')
    assert result.ccy_or_none().code == "USD"


def test_money_truediv_with_undefined_money():
    from pypara.money import Money
    from decimal import Decimal
    
    money = Money.na()
    result = money.__truediv__(Decimal('2'))
    
    assert result.undefined


def test_money_truediv_by_zero():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.money import Money
    
    money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = money.__truediv__(Decimal('0'))
    
    assert result.undefined


def test_money_truediv_preserves_currency():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.money import Money
    
    money = Money.of(Currencies["EUR"], Decimal('20'), Date(2019, 1, 1))
    result = money.__truediv__(Decimal('4'))
    
    assert result.ccy_or_none().code == "EUR"
    assert result.qty_or_zero() == Decimal('5.00')


def test_money_truediv_with_decimal_result():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.money import Money
    
    money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = money.__truediv__(Decimal('3'))
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('3.33')


def test_money_truediv_with_one():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.money import Money
    
    money = Money.of(Currencies["USD"], Decimal('15'), Date(2019, 1, 1))
    result = money.__truediv__(Decimal('1'))
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('15.00')


# LLM-generated content at query #12
#--------------------------

```python
def test_divide_defined_price_by_positive_number():
    from decimal import Decimal
    from datetime import date as Date
    from pypara.currencies import Currencies
    from pypara.price import Price
    
    price = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = price.divide(Decimal('2'))
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('5')
    assert result.ccy_or_none().code == "USD"


def test_divide_defined_price_by_negative_number():
    from decimal import Decimal
    from datetime import date as Date
    from pypara.currencies import Currencies
    from pypara.price import Price
    
    price = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = price.divide(Decimal('-2'))
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('-5')


def test_divide_defined_price_by_zero():
    from decimal import Decimal
    from datetime import date as Date
    from pypara.currencies import Currencies
    from pypara.price import Price
    
    price = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = price.divide(Decimal('0'))
    
    assert result.undefined


def test_divide_undefined_price():
    from decimal import Decimal
    from pypara.price import Price
    
    price = Price.na()
    result = price.divide(Decimal('2'))
    
    assert result.undefined
    assert result is price


def test_divide_decimal_result():
    from decimal import Decimal
    from datetime import date as Date
    from pypara.currencies import Currencies
    from pypara.price import Price
    
    price = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = price.divide(Decimal('3'))
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('10') / Decimal('3')


def test_divide_preserves_currency():
    from decimal import Decimal
    from datetime import date as Date
    from pypara.currencies import Currencies
    from pypara.price import Price
    
    price = Price.of(Currencies["EUR"], Decimal('100'), Date(2019, 1, 1))
    result = price.divide(Decimal('5'))
    
    assert result.ccy_or_none().code == "EUR"
    assert result.qty_or_zero() == Decimal('20')


def test_divide_by_one():
    from decimal import Decimal
    from datetime import date as Date
    from pypara.currencies import Currencies
    from pypara.price import Price
    
    price = Price.of(Currencies["USD"], Decimal('42'), Date(2019, 1, 1))
    result = price.divide(Decimal('1'))
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('42')


# LLM-generated content at query #13
#--------------------------

```python
def test_qty_or_else():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.price import Price
    
    # Test with defined price - should return qty
    someprice = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    result = someprice.qty_or_else(lambda: Decimal('42'))
    assert result == Decimal('1')
    
    # Test with defined price and different return type - should return qty
    result = someprice.qty_or_else(lambda: True)
    assert result == Decimal('1')
    
    # Test with undefined price - should return combinator result (Decimal)
    noneprice = Price.of(None, Decimal('1'), None)
    result = noneprice.qty_or_else(lambda: Decimal('42'))
    assert result == Decimal('42')
    
    # Test with undefined price - should return combinator result (bool)
    result = noneprice.qty_or_else(lambda: False)
    assert result is False
    
    # Test with defined price and zero quantity
    zeroprice = Price.of(Currencies["EUR"], Decimal('0'), Date(2020, 6, 15))
    result = zeroprice.qty_or_else(lambda: Decimal('100'))
    assert result == Decimal('0')
    
    # Test with undefined price and zero default
    result = noneprice.qty_or_else(lambda: Decimal('0'))
    assert result == Decimal('0')
    
    # Test with defined price and negative quantity
    negprice = Price.of(Currencies["GBP"], Decimal('-5'), Date(2021, 3, 10))
    result = negprice.qty_or_else(lambda: Decimal('10'))
    assert result == Decimal('-5')


# LLM-generated content at query #14
#--------------------------

```python
def test_add_with_same_currency():
    from decimal import Decimal
    from datetime import date
    
    ccy = Currency(code="USD", quantizer=Decimal("0.01"))
    price1 = SomePrice(ccy=ccy, qty=Decimal("100.50"), dov=date(2023, 1, 1))
    price2 = SomePrice(ccy=ccy, qty=Decimal("50.25"), dov=date(2023, 1, 2))
    
    result = price1 + price2
    
    assert result.ccy == ccy
    assert result.qty == Decimal("150.75")
    assert result.dov == date(2023, 1, 2)


def test_add_with_different_currency_raises_error():
    from decimal import Decimal
    from datetime import date
    
    ccy_usd = Currency(code="USD", quantizer=Decimal("0.01"))
    ccy_eur = Currency(code="EUR", quantizer=Decimal("0.01"))
    price1 = SomePrice(ccy=ccy_usd, qty=Decimal("100.50"), dov=date(2023, 1, 1))
    price2 = SomePrice(ccy=ccy_eur, qty=Decimal("50.25"), dov=date(2023, 1, 2))
    
    try:
        result = price1 + price2
        assert False, "Expected IncompatibleCurrencyError"
    except IncompatibleCurrencyError as e:
        assert e.ccy1 == ccy_usd
        assert e.ccy2 == ccy_eur
        assert e.operation == "addition"


def test_add_with_undefined_price():
    from decimal import Decimal
    from datetime import date
    
    ccy = Currency(code="USD", quantizer=Decimal("0.01"))
    price1 = SomePrice(ccy=ccy, qty=Decimal("100.50"), dov=date(2023, 1, 1))
    
    result = price1 + NoPrice
    
    assert result == price1


def test_add_takes_later_date():
    from decimal import Decimal
    from datetime import date
    
    ccy = Currency(code="USD", quantizer=Decimal("0.01"))
    price1 = SomePrice(ccy=ccy, qty=Decimal("100.00"), dov=date(2023, 1, 5))
    price2 = SomePrice(ccy=ccy, qty=Decimal("50.00"), dov=date(2023, 1, 3))
    
    result = price1 + price2
    
    assert result.dov == date(2023, 1, 5)


def test_add_with_negative_quantities():
    from decimal import Decimal
    from datetime import date
    
    ccy = Currency(code="USD", quantizer=Decimal("0.01"))
    price1 = SomePrice(ccy=ccy, qty=Decimal("100.00"), dov=date(2023, 1, 1))
    price2 = SomePrice(ccy=ccy, qty=Decimal("-30.00"), dov=date(2023, 1, 1))
    
    result = price1 + price2
    
    assert result.qty == Decimal("70.00")


# LLM-generated content at query #15
#--------------------------

```python
def test_somemoney_sub():
    from decimal import Decimal
    from datetime import date
    
    # Create mock Currency object
    class MockCurrency:
        def __init__(self, code, decimals=2):
            self.code = code
            self.decimals = decimals
            self.quantizer = Decimal(10) ** -decimals
        
        def __eq__(self, other):
            return self.code == other.code
        
        def __ne__(self, other):
            return not self.__eq__(other)
    
    # Create test currencies
    usd = MockCurrency("USD", 2)
    eur = MockCurrency("EUR", 2)
    
    # Test subtraction of two SomeMoney objects with same currency
    money1 = SomeMoney(usd, Decimal("100.00"), date(2024, 1, 1))
    money2 = SomeMoney(usd, Decimal("30.00"), date(2024, 1, 1))
    result = money1.__sub__(money2)
    
    assert result.ccy == usd
    assert result.qty == Decimal("70.00")
    assert result.dov == date(2024, 1, 1)
    
    # Test subtraction uses later date
    money3 = SomeMoney(usd, Decimal("100.00"), date(2024, 1, 1))
    money4 = SomeMoney(usd, Decimal("30.00"), date(2024, 1, 15))
    result2 = money3.__sub__(money4)
    
    assert result2.dov == date(2024, 1, 15)
    assert result2.qty == Decimal("70.00")
    
    # Test subtraction with undefined money (NoMoney)
    from unittest.mock import Mock
    undefined_money = Mock()
    undefined_money.undefined = True
    
    money5 = SomeMoney(usd, Decimal("100.00"), date(2024, 1, 1))
    result3 = money5.__sub__(undefined_money)
    
    assert result3 == money5
    
    # Test subtraction with different currencies raises exception
    money6 = SomeMoney(usd, Decimal("100.00"), date(2024, 1, 1))
    money7 = SomeMoney(eur, Decimal("50.00"), date(2024, 1, 1))
    
    try:
        money6.__sub__(money7)
        assert False, "Should have raised IncompatibleCurrencyError"
    except Exception as e:
        assert "IncompatibleCurrencyError" in str(type(e))


# LLM-generated content at query #16
#--------------------------

```python
def test_add_returns_someprice_with_sum_of_quantities_and_max_date():
    from decimal import Decimal
    from datetime import date
    
    # Create mock Currency and Date objects
    class MockCurrency:
        def __eq__(self, other):
            return isinstance(other, MockCurrency)
        def __ne__(self, other):
            return not isinstance(other, MockCurrency)
    
    ccy = MockCurrency()
    date1 = date(2023, 1, 15)
    date2 = date(2023, 1, 20)
    
    # Create two SomePrice instances
    price1 = SomePrice(ccy, Decimal("100.50"), date1)
    price2 = SomePrice(ccy, Decimal("50.25"), date2)
    
    # Call add method
    result = price1.add(price2)
    
    # Verify the result is a SomePrice instance
    assert isinstance(result, SomePrice)
    
    # Verify the quantity is the sum of both quantities
    assert result.qty == Decimal("150.75")
    
    # Verify the currency is preserved
    assert result.ccy == ccy
    
    # Verify the date is the maximum of the two dates
    assert result.dov == date2


# LLM-generated content at query #17
#--------------------------

```python
def test_nonmoney_constructor():
    none_money = NoneMoney()
    assert none_money is not None
    assert isinstance(none_money, NoneMoney)


# LLM-generated content at query #18
#--------------------------

```python
def test_money_le_operator():
    from decimal import Decimal
    from datetime import date as Date
    from pypara.currencies import Currencies
    from pypara.money import Money
    
    # Test 1: defined money <= defined money with same currency (less than)
    money1 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    assert money1 <= money2
    
    # Test 2: defined money <= defined money with same currency (equal)
    money3 = Money.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    money4 = Money.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    assert money3 <= money4
    
    # Test 3: defined money <= defined money with same currency (greater than)
    money5 = Money.of(Currencies["USD"], Decimal('3'), Date(2019, 1, 1))
    money6 = Money.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    assert not (money5 <= money6)
    
    # Test 4: undefined money <= defined money
    undefined_money = Money.na()
    defined_money = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert undefined_money <= defined_money
    
    # Test 5: undefined money <= undefined money
    undefined_money1 = Money.na()
    undefined_money2 = Money.na()
    assert undefined_money1 <= undefined_money2
    
    # Test 6: defined money <= undefined money (should be False)
    defined_money2 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    undefined_money3 = Money.na()
    assert not (defined_money2 <= undefined_money3)


# LLM-generated content at query #19
#--------------------------

```python
def test_with_dov():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    
    # Test with defined money - should create new money with updated dov
    original_money = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    new_dov = Date(2020, 6, 15)
    updated_money = original_money.with_dov(new_dov)
    
    assert updated_money.dov_or_none() == new_dov
    assert updated_money.qty_or_none() == Decimal('100.00')
    assert updated_money.ccy_or_none().code == "USD"
    assert updated_money.defined is True
    
    # Test with undefined money - should return itself
    undefined_money = Money.na()
    result = undefined_money.with_dov(Date(2020, 1, 1))
    
    assert result.undefined is True
    assert result is undefined_money


# LLM-generated content at query #20
#--------------------------

```python
def test_someprice_convert_with_valid_rate():
    from datetime import date
    from decimal import Decimal
    from pypara.currencies import Currency, CurrencyType
    from pypara.monetary import SomePrice
    from pypara.fx import FXRate, FXRateService
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    
    price = SomePrice(usd, Decimal("100"), date(2023, 1, 1))
    
    fx_rate = FXRate(usd, eur, date(2023, 1, 1), Decimal("0.92"))
    
    class MockFXRateService:
        def query(self, ccy_from, ccy_to, asof, strict):
            return fx_rate
    
    original_service = FXRateService.default
    FXRateService.default = MockFXRateService()
    
    try:
        result = price.convert(eur, date(2023, 1, 1))
        assert result.ccy == eur
        assert result.qty == Decimal("92")
        assert result.dov == date(2023, 1, 1)
    finally:
        FXRateService.default = original_service


def test_someprice_convert_uses_dov_as_default_asof():
    from datetime import date
    from decimal import Decimal
    from pypara.currencies import Currency, CurrencyType
    from pypara.monetary import SomePrice
    from pypara.fx import FXRate, FXRateService
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    
    dov = date(2023, 6, 15)
    price = SomePrice(usd, Decimal("100"), dov)
    
    fx_rate = FXRate(usd, eur, dov, Decimal("0.92"))
    
    class MockFXRateService:
        def __init__(self):
            self.queried_date = None
        
        def query(self, ccy_from, ccy_to, asof, strict):
            self.queried_date = asof
            return fx_rate
    
    mock_service = MockFXRateService()
    original_service = FXRateService.default
    FXRateService.default = mock_service
    
    try:
        result = price.convert(eur)
        assert mock_service.queried_date == dov
    finally:
        FXRateService.default = original_service


def test_someprice_convert_with_no_rate_strict_true():
    from datetime import date
    from decimal import Decimal
    from pypara.currencies import Currency, CurrencyType
    from pypara.monetary import SomePrice
    from pypara.fx import FXRateService, FXRateLookupError
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    
    price = SomePrice(usd, Decimal("100"), date(2023, 1, 1))
    
    class MockFXRateService:
        def query(self, ccy_from, ccy_to, asof, strict):
            return None
    
    original_service = FXRateService.default
    FXRateService.default = MockFXRateService()
    
    try:
        try:
            price.convert(eur, date(2023, 1, 1), strict=True)
            assert False, "Should have raised FXRateLookupError"
        except FXRateLookupError:
            pass
    finally:
        FXRateService.default = original_service


def test_someprice_convert_with_no_rate_strict_false():
    from datetime import date
    from decimal import Decimal
    from pypara.currencies import Currency, CurrencyType
    from pypara.monetary import SomePrice, NoPrice
    from pypara.fx import FXRateService
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    
    price = SomePrice(usd, Decimal("100"), date(2023, 1, 1))
    
    class MockFXRateService:
        def query(self, ccy_from, ccy_to, asof, strict):
            return None
    
    original_service = FXRateService.default
    FXRateService.default = MockFXRateService()
    
    try:
        result = price.convert(eur, date(2023, 1, 1), strict=False)
        assert result is NoPrice
    finally:
        FXRateService.default = original_service


def test_someprice_convert_with_no_fx_rate_service():
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
        try:
            price.convert(eur, date(2023, 1, 1))
            assert False, "Should have raised ProgrammingError"
        except ProgrammingError as e:
            assert "Did you implement and set the default FX rate service?" in str(e)
    finally:
        FXRateService.default = original_service


# LLM-generated content at query #21
#--------------------------

```python
def test_as_boolean():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.money import Money

    # Test undefined money returns False
    undefined_money = Money.na()
    assert undefined_money.as_boolean() is False

    # Test defined money with zero quantity returns False
    zero_money = Money.of(Currencies["USD"], Decimal('0'), Date(2019, 1, 1))
    assert zero_money.as_boolean() is False

    # Test defined money with positive quantity returns True
    positive_money = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert positive_money.as_boolean() is True

    # Test defined money with negative quantity returns True
    negative_money = Money.of(Currencies["USD"], Decimal('-1'), Date(2019, 1, 1))
    assert negative_money.as_boolean() is True

    # Test defined money with small positive quantity returns True
    small_positive_money = Money.of(Currencies["USD"], Decimal('0.01'), Date(2019, 1, 1))
    assert small_positive_money.as_boolean() is True


# LLM-generated content at query #22
#--------------------------

```python
def test_divide_with_defined_money():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    result = money.divide(Decimal('2'))
    
    assert result.defined is True
    assert result.qty_or_zero() == Decimal('50')


def test_divide_with_undefined_money():
    money = Money.na()
    result = money.divide(Decimal('2'))
    
    assert result.undefined is True


def test_divide_by_zero_returns_undefined():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    result = money.divide(Decimal('0'))
    
    assert result.undefined is True


def test_divide_with_decimal_result():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    result = money.divide(Decimal('3'))
    
    assert result.defined is True
    assert result.qty_or_zero() == Decimal('33.33')


def test_divide_with_negative_divisor():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    result = money.divide(Decimal('-2'))
    
    assert result.defined is True
    assert result.qty_or_zero() == Decimal('-50')


def test_divide_preserves_currency():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money = Money.of(Currencies["EUR"], Decimal('100'), Date(2019, 1, 1))
    result = money.divide(Decimal('4'))
    
    assert result.defined is True
    assert result.ccy_or_none().code == 'EUR'


def test_divide_preserves_date():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    original_date = Date(2019, 1, 1)
    money = Money.of(Currencies["USD"], Decimal('100'), original_date)
    result = money.divide(Decimal('5'))
    
    assert result.defined is True
    assert result.dov_or_none() == original_date


# LLM-generated content at query #23
#--------------------------

```python
def test_price_gt():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    # Test: defined price > undefined price should be True
    defined_price = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    undefined_price = Price.na()
    assert defined_price.gt(undefined_price) == True
    
    # Test: undefined price > defined price should be False
    assert undefined_price.gt(defined_price) == False
    
    # Test: undefined price > undefined price should be False
    assert undefined_price.gt(undefined_price) == False
    
    # Test: defined price with greater quantity > defined price with lesser quantity should be True
    price_2 = Price.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    price_1 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert price_2.gt(price_1) == True
    
    # Test: defined price with lesser quantity > defined price with greater quantity should be False
    assert price_1.gt(price_2) == False
    
    # Test: defined price with equal quantity > defined price with equal quantity should be False
    price_1_copy = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert price_1.gt(price_1_copy) == False


# LLM-generated content at query #24
#--------------------------

```python
def test_qty_or_none():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    
    somemoney = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    result_defined = somemoney.qty_or_none()
    assert result_defined == Decimal('1.00')
    
    nonemoney = Money.of(None, Decimal('1'), None)
    result_undefined = nonemoney.qty_or_none()
    assert result_undefined is None


# LLM-generated content at query #25
#--------------------------

```python
def test_dov_or():
    from datetime import date as Date
    from decimal import Decimal
    from pypara.currencies import Currencies
    from pypara.money import Money

    # Test with defined money - should return the dov of the money object
    somemoney = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    result = somemoney.dov_or(Date(2001, 1, 1))
    assert result == Date(2019, 1, 1)

    # Test with undefined money - should return the default date
    nonemoney = Money.of(None, None, Date(2019, 1, 1))
    result = nonemoney.dov_or(Date(2001, 1, 1))
    assert result == Date(2001, 1, 1)

    # Test with undefined money and different default date
    nonemoney2 = Money.na()
    result = nonemoney2.dov_or(Date(2005, 6, 15))
    assert result == Date(2005, 6, 15)

    # Test with defined money and different dates
    somemoney2 = Money.of(Currencies["EUR"], Decimal('100'), Date(2020, 12, 25))
    result = somemoney2.dov_or(Date(2000, 1, 1))
    assert result == Date(2020, 12, 25)


# LLM-generated content at query #26
#--------------------------

```python
def test_price_abs():
    from datetime import date as Date
    from decimal import Decimal
    from pypara.currencies import Currencies
    from pypara.price import Price

    # Test abs with positive defined price
    positive_price = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result_positive = positive_price.abs()
    assert result_positive.qty_or_zero() == Decimal('10')
    assert result_positive.ccy_or_none().code == 'USD'

    # Test abs with negative defined price
    negative_price = Price.of(Currencies["USD"], Decimal('-10'), Date(2019, 1, 1))
    result_negative = negative_price.abs()
    assert result_negative.qty_or_zero() == Decimal('10')
    assert result_negative.ccy_or_none().code == 'USD'

    # Test abs with zero defined price
    zero_price = Price.of(Currencies["USD"], Decimal('0'), Date(2019, 1, 1))
    result_zero = zero_price.abs()
    assert result_zero.qty_or_zero() == Decimal('0')

    # Test abs with undefined price returns itself
    undefined_price = Price.na()
    result_undefined = undefined_price.abs()
    assert result_undefined.undefined
    assert result_undefined is undefined_price


# LLM-generated content at query #27
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
    
    undefined_money = Money.na()
    rounded = undefined_money.round(2)
    
    assert rounded is undefined_money


def test_round_zero_digits():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.money import Money
    
    money = Money.of(Currencies["USD"], Decimal('1.6'), Date(2019, 1, 1))
    rounded = money.round(0)
    
    assert rounded.qty == Decimal('2')


def test_round_negative_quantity():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.money import Money
    
    money = Money.of(Currencies["USD"], Decimal('-1.456'), Date(2019, 1, 1))
    rounded = money.round(2)
    
    assert rounded.qty == Decimal('-1.46')


def test_round_half_even_method():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.money import Money
    
    money = Money.of(Currencies["USD"], Decimal('2.5'), Date(2019, 1, 1))
    rounded = money.round(0)
    
    assert rounded.qty == Decimal('2')


# LLM-generated content at query #28
#--------------------------

```python
def test_someprice_gt():
    from decimal import Decimal
    from datetime import date
    
    # Create mock Currency objects
    class MockCurrency:
        def __init__(self, code):
            self.code = code
            self.quantizer = Decimal('0.01')
        
        def __eq__(self, other):
            return self.code == other.code
        
        def __ne__(self, other):
            return not self.__eq__(other)
    
    usd = MockCurrency('USD')
    eur = MockCurrency('EUR')
    
    # Create test instances
    price1 = SomePrice(usd, Decimal('100'), date(2024, 1, 1))
    price2 = SomePrice(usd, Decimal('50'), date(2024, 1, 1))
    price3 = SomePrice(usd, Decimal('100'), date(2024, 1, 1))
    price_eur = SomePrice(eur, Decimal('100'), date(2024, 1, 1))
    
    # Test: price1 > price2 (True case)
    assert price1.gt(price2) is True
    
    # Test: price2 > price1 (False case)
    assert price2.gt(price1) is False
    
    # Test: price1 > price3 (equal quantities, False case)
    assert price1.gt(price3) is False
    
    # Test: gt with non-SomePrice object returns True
    assert price1.gt("not a price") is True
    
    # Test: gt with different currencies raises IncompatibleCurrencyError
    try:
        price1.gt(price_eur)
        assert False, "Should have raised IncompatibleCurrencyError"
    except Exception as e:
        assert "IncompatibleCurrencyError" in str(type(e).__name__)


# LLM-generated content at query #29
#--------------------------

```python
def test_convert_predicate_line_11_evaluates_to_false():
    from datetime import date
    from decimal import Decimal
    from pypara.currencies import Currency, CurrencyType
    from pypara.monetary import SomePrice
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    
    price = SomePrice(usd, Decimal("100.00"), date(2023, 1, 1))
    
    try:
        price.convert(eur, date(2023, 1, 1), strict=False)
    except AttributeError:
        pass
    
    assert True


# LLM-generated content at query #30
#--------------------------

```python
def test_price_floordiv():
    from decimal import Decimal
    from datetime import date as Date
    from pypara.currencies import Currencies
    from pypara.price import Price
    
    # Test floor division with defined price
    price = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = price.__floordiv__(Decimal('3'))
    assert result.qty_or_zero() == Decimal('3')
    assert result.ccy_or_none().code == 'USD'
    
    # Test floor division with defined price and decimal divisor
    price = Price.of(Currencies["USD"], Decimal('7'), Date(2019, 1, 1))
    result = price.__floordiv__(Decimal('2'))
    assert result.qty_or_zero() == Decimal('3')
    
    # Test floor division with undefined price
    undefined_price = Price.na()
    result = undefined_price.__floordiv__(Decimal('2'))
    assert result.undefined
    
    # Test floor division by zero yields undefined price
    price = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = price.__floordiv__(Decimal('0'))
    assert result.undefined
    
    # Test floor division with negative quantity
    price = Price.of(Currencies["USD"], Decimal('-10'), Date(2019, 1, 1))
    result = price.__floordiv__(Decimal('3'))
    assert result.qty_or_zero() == Decimal('-4')


# LLM-generated content at query #31
#--------------------------

```python
def test_price_sub():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.price import Price
    
    # Test subtraction of two defined prices with same currency
    price1 = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('3'), Date(2019, 1, 1))
    result = price1 - price2
    assert result.defined
    assert result.qty_or_zero() == Decimal('7')
    assert result.ccy_or_none().code == "USD"
    
    # Test subtraction with undefined price (first operand undefined)
    undefined_price = Price.na()
    price_defined = Price.of(Currencies["USD"], Decimal('5'), Date(2019, 1, 1))
    result = undefined_price - price_defined
    assert result.defined == price_defined.defined
    
    # Test subtraction with undefined price (second operand undefined)
    result = price_defined - undefined_price
    assert result.defined == price_defined.defined
    
    # Test subtraction resulting in negative value
    price1 = Price.of(Currencies["USD"], Decimal('3'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = price1 - price2
    assert result.defined
    assert result.qty_or_zero() == Decimal('-7')
    
    # Test subtraction with zero
    price1 = Price.of(Currencies["USD"], Decimal('5'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('0'), Date(2019, 1, 1))
    result = price1 - price2
    assert result.qty_or_zero() == Decimal('5')


# LLM-generated content at query #32
#--------------------------

```python
def test_ccy_or_with_defined_money():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from decimal import Decimal
    from datetime import date as Date
    
    somemoney = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    result = somemoney.ccy_or(Currencies["EUR"])
    assert result.code == 'USD'


def test_ccy_or_with_undefined_money():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from decimal import Decimal
    from datetime import date as Date
    
    nonemoney = Money.of(None, Decimal('1'), None)
    result = nonemoney.ccy_or(Currencies["EUR"])
    assert result.code == 'EUR'


def test_ccy_or_with_none_currency_returns_default():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from decimal import Decimal
    from datetime import date as Date
    
    nonemoney = Money.of(Currencies["USD"], None, None)
    result = nonemoney.ccy_or(Currencies["EUR"])
    assert result.code == 'EUR'


def test_ccy_or_with_none_date_returns_default():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from decimal import Decimal
    from datetime import date as Date
    
    nonemoney = Money.of(Currencies["USD"], Decimal('1'), None)
    result = nonemoney.ccy_or(Currencies["EUR"])
    assert result.code == 'EUR'


# LLM-generated content at query #33
#--------------------------

```python
def test_qty_or_else():
    from datetime import date as Date
    from decimal import Decimal
    from pypara.currencies import Currencies
    from pypara.price import Price

    # Test with defined price - should return quantity
    defined_price = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    result = defined_price.qty_or_else(lambda: Decimal('42'))
    assert result == Decimal('1')

    # Test with defined price and different combinator type - should still return quantity
    defined_price2 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    result2 = defined_price2.qty_or_else(lambda: True)
    assert result2 == Decimal('1')

    # Test with undefined price - should return combinator result (Decimal)
    undefined_price = Price.of(None, Decimal('1'), None)
    result3 = undefined_price.qty_or_else(lambda: Decimal('42'))
    assert result3 == Decimal('42')

    # Test with undefined price - should return combinator result (boolean)
    undefined_price2 = Price.of(None, Decimal('1'), None)
    result4 = undefined_price2.qty_or_else(lambda: False)
    assert result4 is False

    # Test with undefined price and zero quantity - should return combinator result
    undefined_price3 = Price.of(None, Decimal('0'), None)
    result5 = undefined_price3.qty_or_else(lambda: Decimal('100'))
    assert result5 == Decimal('100')

    # Test with defined price and zero quantity - should return zero quantity
    defined_price3 = Price.of(Currencies["EUR"], Decimal('0'), Date(2020, 5, 15))
    result6 = defined_price3.qty_or_else(lambda: Decimal('50'))
    assert result6 == Decimal('0')


# LLM-generated content at query #34
#--------------------------

```python
def test_subtract_two_defined_prices_same_currency():
    from decimal import Decimal
    from datetime import date as Date
    from pypara.currencies import Currencies
    from pypara.price import Price
    
    price1 = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('3'), Date(2019, 1, 2))
    result = price1.subtract(price2)
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('7')
    assert result.ccy_or_none().code == 'USD'


def test_subtract_two_defined_prices_different_currency_raises_error():
    from decimal import Decimal
    from datetime import date as Date
    from pypara.currencies import Currencies
    from pypara.price import Price
    
    price1 = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["EUR"], Decimal('3'), Date(2019, 1, 2))
    
    try:
        price1.subtract(price2)
        assert False, "Expected IncompatibleCurrencyError"
    except Exception as e:
        assert "IncompatibleCurrencyError" in str(type(e).__name__)


def test_subtract_defined_from_undefined_returns_defined():
    from decimal import Decimal
    from datetime import date as Date
    from pypara.currencies import Currencies
    from pypara.price import Price
    
    undefined_price = Price.na()
    defined_price = Price.of(Currencies["USD"], Decimal('5'), Date(2019, 1, 1))
    result = undefined_price.subtract(defined_price)
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('5')


def test_subtract_undefined_from_defined_returns_defined():
    from decimal import Decimal
    from datetime import date as Date
    from pypara.currencies import Currencies
    from pypara.price import Price
    
    defined_price = Price.of(Currencies["USD"], Decimal('5'), Date(2019, 1, 1))
    undefined_price = Price.na()
    result = defined_price.subtract(undefined_price)
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('5')


def test_subtract_two_undefined_prices_returns_undefined():
    from pypara.price import Price
    
    undefined_price1 = Price.na()
    undefined_price2 = Price.na()
    result = undefined_price1.subtract(undefined_price2)
    
    assert result.undefined


def test_subtract_negative_quantity():
    from decimal import Decimal
    from datetime import date as Date
    from pypara.currencies import Currencies
    from pypara.price import Price
    
    price1 = Price.of(Currencies["USD"], Decimal('3'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 2))
    result = price1.subtract(price2)
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('-7')


def test_subtract_zero_quantity():
    from decimal import Decimal
    from datetime import date as Date
    from pypara.currencies import Currencies
    from pypara.price import Price
    
    price1 = Price.of(Currencies["USD"], Decimal('5'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('5'), Date(2019, 1, 2))
    result = price1.subtract(price2)
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('0')


def test_subtract_carries_forward_date():
    from decimal import Decimal
    from datetime import date as Date
    from pypara.currencies import Currencies
    from pypara.price import Price
    
    price1 = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 5))
    price2 = Price.of(Currencies["USD"], Decimal('3'), Date(2019, 1, 1))
    result = price1.subtract(price2)
    
    assert result.dov_or_none() == Date(2019, 1, 5)


# LLM-generated content at query #35
#--------------------------

```python
def test_positive_returns_same_money_when_defined():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.money import Money
    
    money = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    result = money.positive()
    
    assert result.ccy.code == "USD"
    assert result.qty == Decimal('100.00')
    assert result.dov == Date(2019, 1, 1)


def test_positive_returns_itself_when_undefined():
    from pypara.money import Money
    
    undefined_money = Money.na()
    result = undefined_money.positive()
    
    assert result.undefined is True
    assert result is undefined_money


def test_positive_with_negative_quantity():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.money import Money
    
    money = Money.of(Currencies["USD"], Decimal('-50'), Date(2019, 1, 1))
    result = money.positive()
    
    assert result.ccy.code == "USD"
    assert result.qty == Decimal('-50.00')
    assert result.dov == Date(2019, 1, 1)


def test_positive_with_zero_quantity():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.money import Money
    
    money = Money.of(Currencies["USD"], Decimal('0'), Date(2019, 1, 1))
    result = money.positive()
    
    assert result.ccy.code == "USD"
    assert result.qty == Decimal('0.00')
    assert result.dov == Date(2019, 1, 1)


# LLM-generated content at query #36
#--------------------------

```python
def test_dov_or_none():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from decimal import Decimal
    from datetime import date as Date
    
    # Test with defined money - should return the dov
    somemoney = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    result = somemoney.dov_or_none()
    assert result == Date(2019, 1, 1)
    
    # Test with undefined money - should return None
    nonemoney = Money.of(None, None, Date(2019, 1, 1))
    result = nonemoney.dov_or_none()
    assert result is None
    
    # Test with completely undefined money
    na_money = Money.na()
    result = na_money.dov_or_none()
    assert result is None


# LLM-generated content at query #37
#--------------------------

```python
def test_scalar_subtract():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.money import Money
    
    # Test scalar_subtract on defined money
    money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = money.scalar_subtract(Decimal('3'))
    assert result.qty_or_zero() == Decimal('7.00')
    assert result.ccy_or_none().code == "USD"
    
    # Test scalar_subtract with negative number
    money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = money.scalar_subtract(Decimal('-5'))
    assert result.qty_or_zero() == Decimal('15.00')
    
    # Test scalar_subtract on undefined money returns itself
    undefined_money = Money.na()
    result = undefined_money.scalar_subtract(Decimal('5'))
    assert result.undefined
    assert result is undefined_money or result.qty_or_none() is None
    
    # Test scalar_subtract with zero
    money = Money.of(Currencies["EUR"], Decimal('100'), Date(2020, 6, 15))
    result = money.scalar_subtract(Decimal('0'))
    assert result.qty_or_zero() == Decimal('100.00')
    
    # Test scalar_subtract resulting in negative quantity
    money = Money.of(Currencies["GBP"], Decimal('5'), Date(2021, 3, 10))
    result = money.scalar_subtract(Decimal('10'))
    assert result.qty_or_zero() == Decimal('-5.00')


# LLM-generated content at query #38
#--------------------------

```python
def test_qty_map_defined_price():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    someprice = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    result = someprice.qty_map(lambda x: x + Decimal('1'), lambda: Decimal('42'))
    assert result == Decimal('2')


def test_qty_map_undefined_price():
    from decimal import Decimal
    
    noneprice = Price.na()
    result = noneprice.qty_map(lambda x: x + Decimal('1'), lambda: Decimal('42'))
    assert result == Decimal('42')


def test_qty_map_defined_price_with_different_function():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    someprice = Price.of(Currencies["USD"], Decimal('5'), Date(2019, 1, 1))
    result = someprice.qty_map(lambda x: x * Decimal('2'), lambda: Decimal('0'))
    assert result == Decimal('10')


def test_qty_map_undefined_price_calls_else_function():
    from decimal import Decimal
    
    noneprice = Price.na()
    result = noneprice.qty_map(lambda x: x * Decimal('100'), lambda: Decimal('99'))
    assert result == Decimal('99')


def test_qty_map_defined_price_with_string_function():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    someprice = Price.of(Currencies["EUR"], Decimal('3'), Date(2020, 5, 15))
    result = someprice.qty_map(lambda x: str(x), lambda: "error")
    assert result == "3"


def test_qty_map_undefined_price_with_string_else():
    noneprice = Price.na()
    result = noneprice.qty_map(lambda x: str(x), lambda: "fallback")
    assert result == "fallback"


# LLM-generated content at query #39
#--------------------------

```python
def test_price_lte():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal

    # Test: defined price less than another defined price
    price1 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    assert price1.lte(price2) is True

    # Test: defined price equal to another defined price
    price3 = Price.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    price4 = Price.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    assert price3.lte(price4) is True

    # Test: defined price greater than another defined price
    price5 = Price.of(Currencies["USD"], Decimal('3'), Date(2019, 1, 1))
    price6 = Price.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    assert price5.lte(price6) is False

    # Test: undefined price is always less than or equal to defined price
    undefined_price = Price.na()
    defined_price = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert undefined_price.lte(defined_price) is True

    # Test: undefined price is less than or equal to undefined price
    undefined_price1 = Price.na()
    undefined_price2 = Price.na()
    assert undefined_price1.lte(undefined_price2) is True

    # Test: defined price is not less than or equal to undefined price
    defined_price2 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    undefined_price3 = Price.na()
    assert defined_price2.lte(undefined_price3) is False


# LLM-generated content at query #40
#--------------------------

```python
def test_lt_with_same_currency_less_than():
    from decimal import Decimal
    from datetime import date
    
    ccy = Currency(code="USD", quantizer=Decimal("0.01"))
    price1 = SomePrice(ccy, Decimal("10.00"), date(2024, 1, 1))
    price2 = SomePrice(ccy, Decimal("20.00"), date(2024, 1, 1))
    
    result = price1 < price2
    assert result is True


def test_lt_with_same_currency_not_less_than():
    from decimal import Decimal
    from datetime import date
    
    ccy = Currency(code="USD", quantizer=Decimal("0.01"))
    price1 = SomePrice(ccy, Decimal("20.00"), date(2024, 1, 1))
    price2 = SomePrice(ccy, Decimal("10.00"), date(2024, 1, 1))
    
    result = price1 < price2
    assert result is False


def test_lt_with_same_currency_equal():
    from decimal import Decimal
    from datetime import date
    
    ccy = Currency(code="USD", quantizer=Decimal("0.01"))
    price1 = SomePrice(ccy, Decimal("10.00"), date(2024, 1, 1))
    price2 = SomePrice(ccy, Decimal("10.00"), date(2024, 1, 1))
    
    result = price1 < price2
    assert result is False


def test_lt_with_different_currency_raises_error():
    from decimal import Decimal
    from datetime import date
    
    ccy_usd = Currency(code="USD", quantizer=Decimal("0.01"))
    ccy_eur = Currency(code="EUR", quantizer=Decimal("0.01"))
    price1 = SomePrice(ccy_usd, Decimal("10.00"), date(2024, 1, 1))
    price2 = SomePrice(ccy_eur, Decimal("10.00"), date(2024, 1, 1))
    
    try:
        result = price1 < price2
        assert False, "Expected IncompatibleCurrencyError"
    except IncompatibleCurrencyError as e:
        assert e.ccy1 == ccy_usd
        assert e.ccy2 == ccy_eur
        assert e.operation == "< comparison"


def test_lt_with_non_some_price_returns_false():
    from decimal import Decimal
    from datetime import date
    
    ccy = Currency(code="USD", quantizer=Decimal("0.01"))
    price = SomePrice(ccy, Decimal("10.00"), date(2024, 1, 1))
    other = "not a price"
    
    result = price < other
    assert result is False


def test_lt_with_no_price_returns_false():
    from decimal import Decimal
    from datetime import date
    
    ccy = Currency(code="USD", quantizer=Decimal("0.01"))
    price = SomePrice(ccy, Decimal("10.00"), date(2024, 1, 1))
    
    result = price < NoPrice
    assert result is False


# LLM-generated content at query #41
#--------------------------

```python
def test_fmap_with_defined_money():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    
    somemoney = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    new = somemoney.fmap(lambda x: Money.of(x.ccy, x.qty + Decimal('1'), x.dov))
    
    assert new.ccy.code == 'USD'
    assert new.qty == Decimal('2.00')
    assert new.dov == Date(2019, 1, 1)


def test_fmap_with_undefined_money():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    
    nonemoney = Money.of(None, Decimal('1'), None)
    result = nonemoney.fmap(lambda sm: Money.of(sm.ccy, sm.qty + Decimal('1'), sm.dov))
    
    assert result.undefined is True


def test_fmap_transforms_quantity():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    
    somemoney = Money.of(Currencies["EUR"], Decimal('5'), Date(2020, 6, 15))
    new = somemoney.fmap(lambda x: Money.of(x.ccy, x.qty * Decimal('2'), x.dov))
    
    assert new.qty == Decimal('10.00')
    assert new.ccy.code == 'EUR'


def test_fmap_transforms_date():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date, timedelta
    
    somemoney = Money.of(Currencies["GBP"], Decimal('100'), Date(2019, 1, 1))
    new = somemoney.fmap(lambda x: Money.of(x.ccy, x.qty, x.dov + timedelta(days=30)))
    
    assert new.dov == Date(2019, 1, 31)
    assert new.qty == Decimal('100.00')


def test_fmap_with_currency_change():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    
    somemoney = Money.of(Currencies["USD"], Decimal('50'), Date(2019, 6, 1))
    new = somemoney.fmap(lambda x: Money.of(Currencies["JPY"], x.qty, x.dov))
    
    assert new.ccy.code == 'JPY'
    assert new.qty == Decimal('50.00')


# LLM-generated content at query #42
#--------------------------

```python
def test_price_add_two_defined_prices_same_currency():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    
    price1 = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('5'), Date(2019, 1, 2))
    result = price1.add(price2)
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('15')
    assert result.ccy_or_none().code == "USD"


def test_price_add_defined_with_undefined():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    
    price1 = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    price2 = Price.na()
    result = price1.add(price2)
    
    assert result is price1


def test_price_add_undefined_with_defined():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    
    price1 = Price.na()
    price2 = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = price1.add(price2)
    
    assert result is price2


def test_price_add_two_undefined_prices():
    price1 = Price.na()
    price2 = Price.na()
    result = price1.add(price2)
    
    assert result.undefined


def test_price_add_different_currencies_raises_error():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    
    price1 = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["EUR"], Decimal('5'), Date(2019, 1, 1))
    
    try:
        price1.add(price2)
        assert False, "Expected IncompatibleCurrencyError"
    except Exception as e:
        assert "IncompatibleCurrencyError" in str(type(e))


def test_price_add_negative_quantities():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    
    price1 = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('-3'), Date(2019, 1, 2))
    result = price1.add(price2)
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('7')


def test_price_add_zero_quantities():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    
    price1 = Price.of(Currencies["USD"], Decimal('0'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('0'), Date(2019, 1, 2))
    result = price1.add(price2)
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('0')


def test_price_add_carries_forward_date():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    
    price1 = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('5'), Date(2019, 1, 15))
    result = price1.add(price2)
    
    assert result.defined
    assert result.dov_or_none() == Date(2019, 1, 15)


# LLM-generated content at query #43
#--------------------------

```python
def test_money_abs():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    # Test abs on defined money with positive quantity
    positive_money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    abs_positive = positive_money.abs()
    assert abs_positive.qty == Decimal('10.00')
    
    # Test abs on defined money with negative quantity
    negative_money = Money.of(Currencies["USD"], Decimal('-10'), Date(2019, 1, 1))
    abs_negative = negative_money.abs()
    assert abs_negative.qty == Decimal('10.00')
    
    # Test abs on undefined money
    undefined_money = Money.na()
    abs_undefined = undefined_money.abs()
    assert abs_undefined.undefined
    
    # Test abs on zero quantity
    zero_money = Money.of(Currencies["USD"], Decimal('0'), Date(2019, 1, 1))
    abs_zero = zero_money.abs()
    assert abs_zero.qty == Decimal('0.00')


# LLM-generated content at query #44
#--------------------------

```python
def test_add_returns_someprice_with_correct_values():
    from decimal import Decimal
    from datetime import date
    
    # Create mock Currency and Date objects
    class MockCurrency:
        def __eq__(self, other):
            return isinstance(other, MockCurrency)
        def __ne__(self, other):
            return not isinstance(other, MockCurrency)
    
    ccy = MockCurrency()
    date1 = date(2023, 1, 1)
    date2 = date(2023, 1, 2)
    qty1 = Decimal("10.00")
    qty2 = Decimal("5.00")
    
    # Create SomePrice instances
    price1 = SomePrice(ccy, qty1, date1)
    price2 = SomePrice(ccy, qty2, date2)
    
    # Call add method
    result = price1.add(price2)
    
    # Verify the result is a SomePrice instance
    assert isinstance(result, SomePrice)
    
    # Verify the currency is preserved
    assert result.ccy == ccy
    
    # Verify the quantities are added
    assert result.qty == Decimal("15.00")
    
    # Verify the date is the maximum of the two dates
    assert result.dov == date2


# LLM-generated content at query #45
#--------------------------

```python
def test_price_negative():
    from decimal import Decimal
    from datetime import date as Date
    from pypara.currencies import Currencies
    from pypara.price import Price
    
    # Test negative on defined price with positive quantity
    defined_price = Price.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    negated_price = defined_price.negative()
    assert negated_price.qty == Decimal('-100')
    assert negated_price.ccy == Currencies["USD"]
    assert negated_price.dov == Date(2019, 1, 1)
    assert negated_price.defined
    
    # Test negative on defined price with negative quantity
    negative_price = Price.of(Currencies["EUR"], Decimal('-50'), Date(2020, 6, 15))
    negated_negative = negative_price.negative()
    assert negated_negative.qty == Decimal('50')
    assert negated_negative.ccy == Currencies["EUR"]
    assert negated_negative.dov == Date(2020, 6, 15)
    assert negated_negative.defined
    
    # Test negative on undefined price returns itself
    undefined_price = Price.na()
    negated_undefined = undefined_price.negative()
    assert negated_undefined.undefined
    assert negated_undefined is undefined_price
    
    # Test negative on zero quantity
    zero_price = Price.of(Currencies["GBP"], Decimal('0'), Date(2021, 3, 10))
    negated_zero = zero_price.negative()
    assert negated_zero.qty == Decimal('0')
    assert negated_zero.ccy == Currencies["GBP"]
    assert negated_zero.defined


# LLM-generated content at query #46
#--------------------------

```python
def test_money_lte():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    
    # Test: undefined money is always less than or equal to defined money
    undefined_money = Money.na()
    defined_money = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    assert undefined_money.lte(defined_money) is True
    
    # Test: undefined money is less than or equal to undefined money
    undefined_money2 = Money.na()
    assert undefined_money.lte(undefined_money2) is True
    
    # Test: defined money is not less than or equal to undefined money when value is positive
    assert defined_money.lte(undefined_money) is False
    
    # Test: equal defined money objects with same currency
    money1 = Money.of(Currencies["USD"], Decimal('50'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('50'), Date(2019, 1, 2))
    assert money1.lte(money2) is True
    
    # Test: less than comparison with same currency
    money_less = Money.of(Currencies["USD"], Decimal('30'), Date(2019, 1, 1))
    money_more = Money.of(Currencies["USD"], Decimal('70'), Date(2019, 1, 1))
    assert money_less.lte(money_more) is True
    
    # Test: greater than comparison with same currency
    assert money_more.lte(money_less) is False
    
    # Test: incompatible currencies should raise error
    money_usd = Money.of(Currencies["USD"], Decimal('50'), Date(2019, 1, 1))
    money_eur = Money.of(Currencies["EUR"], Decimal('50'), Date(2019, 1, 1))
    try:
        money_usd.lte(money_eur)
        assert False, "Should raise IncompatibleCurrencyError"
    except Exception as e:
        assert "IncompatibleCurrencyError" in str(type(e).__name__)


# LLM-generated content at query #47
#--------------------------

```python
def test_somemoney_sub():
    from decimal import Decimal
    from datetime import date
    
    # Create test currencies and dates
    usd = Currency(code="USD", decimals=2)
    eur = Currency(code="EUR", decimals=2)
    test_date1 = date(2023, 1, 1)
    test_date2 = date(2023, 1, 2)
    
    # Test basic subtraction with same currency
    money1 = SomeMoney(usd, Decimal("100.00"), test_date1)
    money2 = SomeMoney(usd, Decimal("30.00"), test_date2)
    result = money1 - money2
    
    assert result.ccy == usd
    assert result.qty == Decimal("70.00")
    assert result.dov == test_date2  # Later date should be selected
    
    # Test subtraction resulting in negative quantity
    money3 = SomeMoney(usd, Decimal("20.00"), test_date1)
    money4 = SomeMoney(usd, Decimal("50.00"), test_date1)
    result2 = money3 - money4
    
    assert result2.qty == Decimal("-30.00")
    
    # Test subtraction with NoMoney (undefined money)
    money5 = SomeMoney(usd, Decimal("100.00"), test_date1)
    result3 = money5 - NoMoney
    
    assert result3 == money5
    
    # Test subtraction with different currencies raises error
    money6 = SomeMoney(usd, Decimal("100.00"), test_date1)
    money7 = SomeMoney(eur, Decimal("50.00"), test_date1)
    
    try:
        money6 - money7
        assert False, "Should have raised IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass
    
    # Test subtraction with zero
    money8 = SomeMoney(usd, Decimal("100.00"), test_date1)
    money9 = SomeMoney(usd, Decimal("0.00"), test_date1)
    result4 = money8 - money9
    
    assert result4.qty == Decimal("100.00")


# LLM-generated content at query #48
#--------------------------

```python
def test_as_boolean():
    from decimal import Decimal
    from datetime import date as Date
    from pypara.currencies import Currencies
    
    # Test undefined money returns False
    undefined_money = Money.na()
    assert undefined_money.as_boolean() is False
    
    # Test defined money with zero quantity returns False
    zero_money = Money.of(Currencies["USD"], Decimal('0'), Date(2019, 1, 1))
    assert zero_money.as_boolean() is False
    
    # Test defined money with positive quantity returns True
    positive_money = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert positive_money.as_boolean() is True
    
    # Test defined money with negative quantity returns True
    negative_money = Money.of(Currencies["USD"], Decimal('-1'), Date(2019, 1, 1))
    assert negative_money.as_boolean() is True
    
    # Test defined money with small positive quantity returns True
    small_money = Money.of(Currencies["EUR"], Decimal('0.01'), Date(2019, 1, 1))
    assert small_money.as_boolean() is True


# LLM-generated content at query #49
#--------------------------

```python
def test_money_sub_method():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    
    # Test subtraction of two defined money objects with same currency
    money1 = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('3'), Date(2019, 1, 1))
    result = money1 - money2
    assert result.qty_or_zero() == Decimal('7.00')
    assert result.ccy_or_none().code == "USD"
    
    # Test subtraction with negative result
    money3 = Money.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    money4 = Money.of(Currencies["USD"], Decimal('5'), Date(2019, 1, 1))
    result2 = money3 - money4
    assert result2.qty_or_zero() == Decimal('-3.00')
    
    # Test subtraction when first operand is undefined
    undefined_money = Money.na()
    money5 = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result3 = undefined_money - money5
    assert result3.undefined
    
    # Test subtraction when second operand is undefined
    money6 = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result4 = money6 - undefined_money
    assert result4.defined
    assert result4.qty_or_zero() == Decimal('10.00')
    
    # Test subtraction of two undefined money objects
    result5 = undefined_money - undefined_money
    assert result5.undefined
    
    # Test subtraction with different currencies raises error
    money_usd = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    money_eur = Money.of(Currencies["EUR"], Decimal('10'), Date(2019, 1, 1))
    try:
        result6 = money_usd - money_eur
        assert False, "Should raise IncompatibleCurrencyError"
    except Exception as e:
        assert "IncompatibleCurrencyError" in str(type(e))


# LLM-generated content at query #50
#--------------------------

```python
def test_money_neg():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.money import Money
    
    # Test __neg__ on defined money with positive quantity
    money_positive = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    money_negated = -money_positive
    assert money_negated.qty == Decimal('-100.00')
    assert money_negated.ccy == Currencies["USD"]
    assert money_negated.dov == Date(2019, 1, 1)
    
    # Test __neg__ on defined money with negative quantity
    money_negative = Money.of(Currencies["USD"], Decimal('-50'), Date(2019, 1, 1))
    money_negated_again = -money_negative
    assert money_negated_again.qty == Decimal('50.00')
    
    # Test __neg__ on defined money with zero quantity
    money_zero = Money.of(Currencies["USD"], Decimal('0'), Date(2019, 1, 1))
    money_negated_zero = -money_zero
    assert money_negated_zero.qty == Decimal('0.00')
    
    # Test __neg__ on undefined money
    money_undefined = Money.na()
    money_negated_undefined = -money_undefined
    assert money_negated_undefined.undefined


# LLM-generated content at query #51
#--------------------------

```python
def test_qty_or_none():
    from datetime import date as Date
    from decimal import Decimal
    from pypara.currencies import Currencies
    from pypara.price import Price

    # Test with defined price - should return the quantity
    defined_price = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    result = defined_price.qty_or_none()
    assert result == Decimal('1')

    # Test with undefined price - should return None
    undefined_price = Price.of(None, Decimal('1'), None)
    result = undefined_price.qty_or_none()
    assert result is None

    # Test with another defined price with different quantity
    defined_price_2 = Price.of(Currencies["EUR"], Decimal('42.50'), Date(2020, 6, 15))
    result = defined_price_2.qty_or_none()
    assert result == Decimal('42.50')

    # Test with Price.na() - should return None
    na_price = Price.na()
    result = na_price.qty_or_none()
    assert result is None


# LLM-generated content at query #52
#--------------------------

```python
def test_as_float():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    # Test with defined price
    defined_price = Price.of(Currencies["USD"], Decimal('123.45'), Date(2019, 1, 1))
    result = defined_price.as_float()
    assert result == 123.45
    assert isinstance(result, float)
    
    # Test with undefined price - should raise MonetaryOperationException
    undefined_price = Price.na()
    try:
        undefined_price.as_float()
        assert False, "Expected MonetaryOperationException"
    except Exception as e:
        assert "MonetaryOperationException" in str(type(e))


# LLM-generated content at query #53
#--------------------------

```python
def test_someprice_constructor():
    from decimal import Decimal
    from datetime import date
    
    ccy = Currency("USD")
    qty = Decimal("100.50")
    dov = date(2024, 1, 15)
    
    price = SomePrice(ccy, qty, dov)
    
    assert price.ccy == ccy
    assert price.qty == qty
    assert price.dov == dov
    assert price[0] == ccy
    assert price[1] == qty
    assert price[2] == dov


def test_someprice_constructor_with_different_currencies():
    from decimal import Decimal
    from datetime import date
    
    ccy_eur = Currency("EUR")
    qty = Decimal("250.75")
    dov = date(2024, 6, 30)
    
    price = SomePrice(ccy_eur, qty, dov)
    
    assert price.ccy == ccy_eur
    assert price.qty == qty
    assert price.dov == dov


def test_someprice_constructor_with_zero_quantity():
    from decimal import Decimal
    from datetime import date
    
    ccy = Currency("GBP")
    qty = Decimal("0")
    dov = date(2024, 12, 31)
    
    price = SomePrice(ccy, qty, dov)
    
    assert price.ccy == ccy
    assert price.qty == qty
    assert price.dov == dov


def test_someprice_constructor_with_negative_quantity():
    from decimal import Decimal
    from datetime import date
    
    ccy = Currency("JPY")
    qty = Decimal("-500.25")
    dov = date(2024, 3, 15)
    
    price = SomePrice(ccy, qty, dov)
    
    assert price.ccy == ccy
    assert price.qty == qty
    assert price.dov == dov


# LLM-generated content at query #54
#--------------------------

```python
def test_multiply_defined_money_positive_scalar():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = money.multiply(Decimal('2'))
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('20.00')
    assert result.ccy_or_none().code == "USD"


def test_multiply_defined_money_negative_scalar():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = money.multiply(Decimal('-3'))
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('-30.00')


def test_multiply_defined_money_zero_scalar():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = money.multiply(Decimal('0'))
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('0.00')


def test_multiply_defined_money_fractional_scalar():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    result = money.multiply(Decimal('0.5'))
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('50.00')


def test_multiply_undefined_money_returns_as_is():
    from decimal import Decimal
    
    money = Money.na()
    result = money.multiply(Decimal('5'))
    
    assert result.undefined
    assert result is Money.na() or result.undefined


def test_multiply_defined_money_with_integer_scalar():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money = Money.of(Currencies["EUR"], Decimal('25'), Date(2019, 6, 15))
    result = money.multiply(4)
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('100.00')
    assert result.ccy_or_none().code == "EUR"


def test_multiply_preserves_currency_and_date():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    original_date = Date(2019, 3, 10)
    money = Money.of(Currencies["GBP"], Decimal('50'), original_date)
    result = money.multiply(Decimal('2'))
    
    assert result.ccy_or_none().code == "GBP"
    assert result.dov_or_none() == original_date


# LLM-generated content at query #55
#--------------------------

```python
def test_subtract_defined_money_objects():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money1 = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('3'), Date(2019, 1, 1))
    result = money1.subtract(money2)
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('7.00')
    assert result.ccy_or_none().code == "USD"


def test_subtract_with_undefined_left_operand():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    undefined_money = Money.na()
    money2 = Money.of(Currencies["USD"], Decimal('3'), Date(2019, 1, 1))
    result = undefined_money.subtract(money2)
    
    assert result is money2


def test_subtract_with_undefined_right_operand():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money1 = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    undefined_money = Money.na()
    result = money1.subtract(undefined_money)
    
    assert result is money1


def test_subtract_both_undefined():
    undefined_money1 = Money.na()
    undefined_money2 = Money.na()
    result = undefined_money1.subtract(undefined_money2)
    
    assert result.undefined


def test_subtract_negative_result():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money1 = Money.of(Currencies["USD"], Decimal('3'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = money1.subtract(money2)
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('-7.00')


def test_subtract_same_currency_different_dates():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money1 = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('3'), Date(2019, 1, 15))
    result = money1.subtract(money2)
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('7.00')
    assert result.dov_or_none() == Date(2019, 1, 1)


def test_subtract_zero_result():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money1 = Money.of(Currencies["USD"], Decimal('5'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('5'), Date(2019, 1, 1))
    result = money1.subtract(money2)
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('0.00')


def test_subtract_incompatible_currencies_raises_error():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money1 = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["EUR"], Decimal('3'), Date(2019, 1, 1))
    
    try:
        result = money1.subtract(money2)
        assert False, "Expected IncompatibleCurrencyError"
    except Exception as e:
        assert "IncompatibleCurrencyError" in str(type(e).__name__)


# LLM-generated content at query #56
#--------------------------

```python
def test_dov_or_none():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    from pypara.price import Price
    
    # Test case 1: defined price returns dov
    someprice = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    result = someprice.dov_or_none()
    assert result == Date(2019, 1, 1)
    
    # Test case 2: undefined price (no currency) returns None
    noneprice = Price.of(None, None, Date(2019, 1, 1))
    result = noneprice.dov_or_none()
    assert result is None
    
    # Test case 3: undefined price (no quantity) returns None
    noneprice2 = Price.of(Currencies["USD"], None, Date(2019, 1, 1))
    result = noneprice2.dov_or_none()
    assert result is None
    
    # Test case 4: undefined price (no dov) returns None
    noneprice3 = Price.of(Currencies["USD"], Decimal('1'), None)
    result = noneprice3.dov_or_none()
    assert result is None
    
    # Test case 5: completely undefined price returns None
    noneprice4 = Price.na()
    result = noneprice4.dov_or_none()
    assert result is None


# LLM-generated content at query #57
#--------------------------

```python
def test_multiply_defined_money_by_positive_scalar():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = money.multiply(Decimal('2'))
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('20.00')
    assert result.ccy_or_none().code == "USD"


def test_multiply_defined_money_by_negative_scalar():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = money.multiply(Decimal('-3'))
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('-30.00')


def test_multiply_defined_money_by_zero():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = money.multiply(Decimal('0'))
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('0.00')


def test_multiply_defined_money_by_fractional_scalar():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = money.multiply(Decimal('0.5'))
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('5.00')


def test_multiply_undefined_money_returns_undefined():
    from decimal import Decimal
    
    money = Money.na()
    result = money.multiply(Decimal('5'))
    
    assert result.undefined
    assert result is money


# LLM-generated content at query #58
#--------------------------

```python
def test_convert_defined_price():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    usd = Currencies["USD"]
    eur = Currencies["EUR"]
    price = Price.of(usd, Decimal('100'), Date(2019, 1, 1))
    converted = price.convert(eur, Date(2019, 1, 1))
    
    assert converted.ccy_or_none() == eur


def test_convert_undefined_price():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    eur = Currencies["EUR"]
    undefined_price = Price.na()
    converted = undefined_price.convert(eur)
    
    assert converted.undefined


def test_convert_with_asof_date():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    usd = Currencies["USD"]
    gbp = Currencies["GBP"]
    price = Price.of(usd, Decimal('100'), Date(2019, 1, 1))
    converted = price.convert(gbp, asof=Date(2019, 6, 1))
    
    assert converted.ccy_or_none() == gbp


def test_convert_same_currency():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    usd = Currencies["USD"]
    price = Price.of(usd, Decimal('100'), Date(2019, 1, 1))
    converted = price.convert(usd)
    
    assert converted.ccy_or_none() == usd
    assert converted.qty_or_zero() == Decimal('100')


def test_convert_strict_mode():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    usd = Currencies["USD"]
    jpy = Currencies["JPY"]
    price = Price.of(usd, Decimal('100'), Date(2019, 1, 1))
    converted = price.convert(jpy, strict=True)
    
    assert converted.defined or converted.undefined


# LLM-generated content at query #59
#--------------------------

```python
def test_ccy_or():
    from datetime import date as Date
    from decimal import Decimal
    from pypara.currencies import Currencies
    from pypara.price import Price

    # Test with defined price - should return the price's currency
    someprice = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    result = someprice.ccy_or(Currencies["EUR"])
    assert result.code == 'USD'

    # Test with undefined price - should return the default currency
    noneprice = Price.of(None, Decimal('1'), None)
    result = noneprice.ccy_or(Currencies["EUR"])
    assert result.code == 'EUR'

    # Test with another defined price - should return its own currency
    gbpprice = Price.of(Currencies["GBP"], Decimal('100'), Date(2020, 5, 15))
    result = gbpprice.ccy_or(Currencies["JPY"])
    assert result.code == 'GBP'

    # Test with na() - should return the default currency
    naprice = Price.na()
    result = naprice.ccy_or(Currencies["CHF"])
    assert result.code == 'CHF'


# LLM-generated content at query #60
#--------------------------

```python
def test_dov_or():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    # Test with defined money - should return the dov of the money object
    somemoney = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    result = somemoney.dov_or(Date(2001, 1, 1))
    assert result == Date(2019, 1, 1)
    
    # Test with undefined money - should return the default date
    nonemoney = Money.of(None, None, Date(2019, 1, 1))
    result = nonemoney.dov_or(Date(2001, 1, 1))
    assert result == Date(2001, 1, 1)
    
    # Test with undefined money (all None) - should return the default date
    nonemoney2 = Money.na()
    result = nonemoney2.dov_or(Date(2005, 5, 5))
    assert result == Date(2005, 5, 5)


# LLM-generated content at query #61
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
    
    # Test adding defined price with undefined price
    price_defined = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    price_undefined = Price.na()
    result = price_defined.add(price_undefined)
    assert result is price_defined
    
    # Test adding undefined price with defined price
    price_undefined = Price.na()
    price_defined = Price.of(Currencies["EUR"], Decimal('20'), Date(2019, 1, 1))
    result = price_undefined.add(price_defined)
    assert result is price_defined
    
    # Test adding two undefined prices
    price_undefined1 = Price.na()
    price_undefined2 = Price.na()
    result = price_undefined1.add(price_undefined2)
    assert result.undefined
    
    # Test adding with negative quantities
    price1 = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('-3'), Date(2019, 1, 2))
    result = price1.add(price2)
    assert result.qty_or_zero() == Decimal('7')
    
    # Test that addition with different currencies raises error
    price1 = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["EUR"], Decimal('5'), Date(2019, 1, 2))
    try:
        result = price1.add(price2)
        assert False, "Should raise IncompatibleCurrencyError"
    except Exception as e:
        assert "IncompatibleCurrencyError" in str(type(e).__name__)


# LLM-generated content at query #62
#--------------------------

```python
def test_floor_divide_with_defined_price():
    from pypara.currencies import Currencies
    from pypara.price import Price
    from decimal import Decimal
    from datetime import date as Date
    
    price = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = price.floor_divide(Decimal('3'))
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('3')
    assert result.ccy_or_none().code == "USD"


def test_floor_divide_with_undefined_price():
    from pypara.price import Price
    from decimal import Decimal
    
    price = Price.na()
    result = price.floor_divide(Decimal('3'))
    
    assert result.undefined
    assert result is price


def test_floor_divide_by_zero():
    from pypara.currencies import Currencies
    from pypara.price import Price
    from decimal import Decimal
    from datetime import date as Date
    
    price = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = price.floor_divide(Decimal('0'))
    
    assert result.undefined


def test_floor_divide_preserves_currency():
    from pypara.currencies import Currencies
    from pypara.price import Price
    from decimal import Decimal
    from datetime import date as Date
    
    price = Price.of(Currencies["EUR"], Decimal('20'), Date(2019, 6, 15))
    result = price.floor_divide(Decimal('4'))
    
    assert result.ccy_or_none().code == "EUR"
    assert result.qty_or_zero() == Decimal('5')


def test_floor_divide_with_negative_divisor():
    from pypara.currencies import Currencies
    from pypara.price import Price
    from decimal import Decimal
    from datetime import date as Date
    
    price = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = price.floor_divide(Decimal('-3'))
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('-4')


def test_floor_divide_with_decimal_divisor():
    from pypara.currencies import Currencies
    from pypara.price import Price
    from decimal import Decimal
    from datetime import date as Date
    
    price = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = price.floor_divide(Decimal('2.5'))
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('4')


# LLM-generated content at query #63
#--------------------------

```python
def test_with_qty():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    # Test with_qty on defined money
    defined_money = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    new_money = defined_money.with_qty(Decimal('5'))
    assert new_money.qty_or_none() == Decimal('5')
    assert new_money.ccy_or_none().code == "USD"
    assert new_money.dov_or_none() == Date(2019, 1, 1)
    
    # Test with_qty on undefined money
    undefined_money = Money.na()
    result = undefined_money.with_qty(Decimal('10'))
    assert result.undefined
    assert result is undefined_money


# LLM-generated content at query #64
#--------------------------

```python
def test_lt_defined_prices_same_currency():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.price import Price
    
    price1 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    
    assert price1.lt(price2) is True
    assert price2.lt(price1) is False


def test_lt_defined_prices_equal():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.price import Price
    
    price1 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    
    assert price1.lt(price2) is False


def test_lt_undefined_vs_defined():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.price import Price
    
    undefined_price = Price.na()
    defined_price = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    
    assert undefined_price.lt(defined_price) is True


def test_lt_defined_vs_undefined():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.price import Price
    
    defined_price = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    undefined_price = Price.na()
    
    assert defined_price.lt(undefined_price) is False


def test_lt_undefined_vs_undefined():
    from pypara.price import Price
    
    undefined_price1 = Price.na()
    undefined_price2 = Price.na()
    
    assert undefined_price1.lt(undefined_price2) is False


def test_lt_incompatible_currencies():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.price import Price
    
    price_usd = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    price_eur = Price.of(Currencies["EUR"], Decimal('2'), Date(2019, 1, 1))
    
    try:
        price_usd.lt(price_eur)
        assert False, "Should raise IncompatibleCurrencyError"
    except Exception as e:
        assert "IncompatibleCurrencyError" in str(type(e))


# LLM-generated content at query #65
#--------------------------

```python
def test_dov_or_none():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    from pypara.price import Price
    
    # Test with defined price - should return the dov
    someprice = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    result = someprice.dov_or_none()
    assert result == Date(2019, 1, 1)
    
    # Test with undefined price - should return None
    noneprice = Price.of(None, None, Date(2019, 1, 1))
    result = noneprice.dov_or_none()
    assert result is None
    
    # Test with undefined price created via Price.na()
    na_price = Price.na()
    result = na_price.dov_or_none()
    assert result is None


# LLM-generated content at query #66
#--------------------------

```python
def test_price_int_conversion():
    from decimal import Decimal
    from datetime import date as Date
    from pypara.currencies import Currencies
    from pypara.price import Price
    
    # Test __int__ on defined price with positive quantity
    price_positive = Price.of(Currencies["USD"], Decimal('42.7'), Date(2019, 1, 1))
    assert int(price_positive) == 42
    
    # Test __int__ on defined price with negative quantity
    price_negative = Price.of(Currencies["USD"], Decimal('-42.7'), Date(2019, 1, 1))
    assert int(price_negative) == -42
    
    # Test __int__ on defined price with zero quantity
    price_zero = Price.of(Currencies["USD"], Decimal('0'), Date(2019, 1, 1))
    assert int(price_zero) == 0
    
    # Test __int__ on defined price with integer quantity
    price_integer = Price.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    assert int(price_integer) == 100


# LLM-generated content at query #67
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
    nomoney1 = Money.na()
    nomoney2 = Money.na()
    assert nomoney1 == nomoney2
    
    # Test inequality of defined and undefined money objects
    assert not (money1 == nomoney1)
    
    # Test equality with non-money object
    assert not (money1 == "not a money")
    assert not (money1 == 1)
    assert not (money1 == None)


# LLM-generated content at query #68
#--------------------------

```python
def test_divide_defined_price_by_positive_number():
    from pypara.currencies import Currencies
    from pypara.price import Price
    from decimal import Decimal
    from datetime import date as Date
    
    price = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = price.divide(Decimal('2'))
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('5')
    assert result.ccy_or_none().code == 'USD'


def test_divide_defined_price_by_negative_number():
    from pypara.currencies import Currencies
    from pypara.price import Price
    from decimal import Decimal
    from datetime import date as Date
    
    price = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = price.divide(Decimal('-2'))
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('-5')


def test_divide_defined_price_by_zero():
    from pypara.currencies import Currencies
    from pypara.price import Price
    from decimal import Decimal
    from datetime import date as Date
    
    price = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = price.divide(Decimal('0'))
    
    assert result.undefined


def test_divide_undefined_price():
    from pypara.price import Price
    
    price = Price.na()
    result = price.divide(Decimal('2'))
    
    assert result.undefined
    assert result is price


def test_divide_defined_price_by_fraction():
    from pypara.currencies import Currencies
    from pypara.price import Price
    from decimal import Decimal
    from datetime import date as Date
    
    price = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = price.divide(Decimal('0.5'))
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('20')


def test_divide_preserves_currency():
    from pypara.currencies import Currencies
    from pypara.price import Price
    from decimal import Decimal
    from datetime import date as Date
    
    price = Price.of(Currencies["EUR"], Decimal('100'), Date(2019, 1, 1))
    result = price.divide(Decimal('4'))
    
    assert result.ccy_or_none().code == 'EUR'
    assert result.qty_or_zero() == Decimal('25')


# LLM-generated content at query #69
#--------------------------

```python
def test_convert_with_valid_rate():
    from datetime import date
    from decimal import Decimal
    from pypara.currencies import Currency, CurrencyType
    from pypara.monetary import SomeMoney
    from pypara.fx import FXRate, FXRateService
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    
    money = SomeMoney(usd, Decimal("100.00"), date(2023, 1, 1))
    
    class MockFXRateService:
        default = None
        
        @staticmethod
        def query(ccy1, ccy2, asof, strict):
            return FXRate(ccy1, ccy2, Decimal("0.85"), asof)
    
    original_service = FXRateService.default
    FXRateService.default = MockFXRateService()
    
    try:
        result = money.convert(eur, date(2023, 1, 1))
        assert result.ccy == eur
        assert result.qty == Decimal("85.00")
        assert result.dov == date(2023, 1, 1)
    finally:
        FXRateService.default = original_service


def test_convert_uses_dov_when_asof_not_provided():
    from datetime import date
    from decimal import Decimal
    from pypara.currencies import Currency, CurrencyType
    from pypara.monetary import SomeMoney
    from pypara.fx import FXRate, FXRateService
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    gbp = Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY)
    
    dov = date(2023, 6, 15)
    money = SomeMoney(usd, Decimal("100.00"), dov)
    
    class MockFXRateService:
        default = None
        query_date = None
        
        @staticmethod
        def query(ccy1, ccy2, asof, strict):
            MockFXRateService.query_date = asof
            return FXRate(ccy1, ccy2, Decimal("0.79"), asof)
    
    original_service = FXRateService.default
    FXRateService.default = MockFXRateService()
    
    try:
        result = money.convert(gbp)
        assert MockFXRateService.query_date == dov
        assert result.dov == dov
    finally:
        FXRateService.default = original_service


def test_convert_with_no_rate_strict_mode():
    from datetime import date
    from decimal import Decimal
    from pypara.currencies import Currency, CurrencyType
    from pypara.monetary import SomeMoney
    from pypara.fx import FXRateService
    from pypara.errors import FXRateLookupError
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    
    money = SomeMoney(usd, Decimal("100.00"), date(2023, 1, 1))
    
    class MockFXRateService:
        default = None
        
        @staticmethod
        def query(ccy1, ccy2, asof, strict):
            return None
    
    original_service = FXRateService.default
    FXRateService.default = MockFXRateService()
    
    try:
        exception_raised = False
        try:
            money.convert(jpy, date(2023, 1, 1), strict=True)
        except FXRateLookupError:
            exception_raised = True
        assert exception_raised
    finally:
        FXRateService.default = original_service


def test_convert_with_no_rate_non_strict_mode():
    from datetime import date
    from decimal import Decimal
    from pypara.currencies import Currency, CurrencyType
    from pypara.monetary import SomeMoney, NoMoney
    from pypara.fx import FXRateService
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    chf = Currency.of("CHF", "Swiss Franc", 2, CurrencyType.MONEY)
    
    money = SomeMoney(usd, Decimal("100.00"), date(2023, 1, 1))
    
    class MockFXRateService:
        default = None
        
        @staticmethod
        def query(ccy1, ccy2, asof, strict):
            return None
    
    original_service = FXRateService.default
    FXRateService.default = MockFXRateService()
    
    try:
        result = money.convert(chf, date(2023, 1, 1), strict=False)
        assert result is NoMoney
    finally:
        FXRateService.default = original_service


def test_convert_no_service_raises_error():
    from datetime import date
    from decimal import Decimal
    from pypara.currencies import Currency, CurrencyType
    from pypara.monetary import SomeMoney
    from pypara.fx import FXRateService
    from pypara.errors import ProgrammingError
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    cad = Currency.of("CAD", "Canadian Dollar", 2, CurrencyType.MONEY)
    
    money = SomeMoney(usd, Decimal("100.00"), date(2023, 1, 1))
    
    original_service = FXRateService.default
    FXRateService.default = None
    
    try:
        exception_raised = False
        try:
            money.convert(cad, date(2023, 1, 1))
        except ProgrammingError:
            exception_raised = True
        assert exception_raised
    finally:
        FXRateService.default = original_service


def test_convert_quantizes_result():
    from datetime import date
    from decimal import Decimal
    from pypara.currencies import Currency, CurrencyType
    from pypara.monetary import SomeMoney
    from pypara.fx import FXRate, FXRateService
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    
    money = SomeMoney(usd, Decimal("100.00"), date(2023, 1, 1))
    
    class MockFXRateService:
        default = None
        
        @staticmethod
        def query(ccy1, ccy2, asof, strict):
            return FXRate(ccy1, ccy2, Decimal("0.8567"), asof)
    
    original_service = FXRateService.default
    FXRateService.default = MockFXRateService()
    
    try:
        result = money.convert(eur, date(2023, 1, 1))
        assert result.qty == Decimal("85.67")
    finally:
        FXRateService.default = original_service


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_somemoney_neg():
    from decimal import Decimal
    from datetime import date
    
    # Create a Currency mock with necessary attributes
    class MockCurrency:
        def __init__(self, code, decimals=2):
            self.code = code
            self.decimals = decimals
            self.quantizer = Decimal(10) ** -decimals
        
        def __eq__(self, other):
            return self.code == other.code
        
        def __hash__(self):
            return hash(self.code)
    
    ccy = MockCurrency("USD")
    qty = Decimal("100.50")
    dov = date(2023, 1, 1)
    
    money = SomeMoney(ccy, qty, dov)
    result = -money
    
    assert isinstance(result, SomeMoney)
    assert result.ccy == ccy
    assert result.qty == Decimal("-100.50")
    assert result.dov == dov


def test_somemoney_neg_negative_value():
    from decimal import Decimal
    from datetime import date
    
    class MockCurrency:
        def __init__(self, code, decimals=2):
            self.code = code
            self.decimals = decimals
            self.quantizer = Decimal(10) ** -decimals
        
        def __eq__(self, other):
            return self.code == other.code
        
        def __hash__(self):
            return hash(self.code)
    
    ccy = MockCurrency("EUR")
    qty = Decimal("-50.25")
    dov = date(2023, 6, 15)
    
    money = SomeMoney(ccy, qty, dov)
    result = -money
    
    assert isinstance(result, SomeMoney)
    assert result.ccy == ccy
    assert result.qty == Decimal("50.25")
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
        
        def __hash__(self):
            return hash(self.code)
    
    ccy = MockCurrency("GBP")
    qty = Decimal("0.00")
    dov = date(2023, 12, 31)
    
    money = SomeMoney(ccy, qty, dov)
    result = -money
    
    assert isinstance(result, SomeMoney)
    assert result.ccy == ccy
    assert result.qty == Decimal("0.00")
    assert result.dov == dov


# LLM-generated content at query #2
#--------------------------

```python
def test_scalar_add_with_defined_money():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = money.scalar_add(Decimal('5'))
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('15.00')
    assert result.ccy_or_none().code == "USD"


def test_scalar_add_with_undefined_money():
    from decimal import Decimal
    
    money = Money.na()
    result = money.scalar_add(Decimal('5'))
    
    assert result.undefined
    assert result is money


def test_scalar_add_with_negative_scalar():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = money.scalar_add(Decimal('-3'))
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('7.00')


def test_scalar_add_with_zero():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = money.scalar_add(Decimal('0'))
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('10.00')


def test_scalar_add_with_large_number():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money = Money.of(Currencies["USD"], Decimal('100.50'), Date(2019, 1, 1))
    result = money.scalar_add(Decimal('999.50'))
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('1100.00')


# LLM-generated content at query #3
#--------------------------

```python
def test_convert_same_currency():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    converted = money.convert(Currencies["USD"])
    
    assert converted.ccy.code == "USD"
    assert converted.qty == Decimal('100.00')


def test_convert_with_asof_date():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    converted = money.convert(Currencies["EUR"], asof=Date(2019, 6, 1))
    
    assert converted.ccy.code == "EUR"
    assert converted.dov == Date(2019, 6, 1)


def test_convert_undefined_money():
    from pypara.currencies import Currencies
    from datetime import date as Date
    
    undefined_money = Money.na()
    converted = undefined_money.convert(Currencies["USD"])
    
    assert converted.undefined


def test_convert_strict_mode():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    converted = money.convert(Currencies["EUR"], strict=True)
    
    assert converted.ccy.code == "EUR"


def test_convert_different_currency():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    converted = money.convert(Currencies["GBP"], asof=Date(2019, 1, 1))
    
    assert converted.ccy.code == "GBP"
    assert converted.defined


# LLM-generated content at query #4
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
    undefined_price = Price.na()
    
    # Test: defined price >= smaller defined price (should be True)
    assert price_usd_100.gte(price_usd_50) is True
    
    # Test: defined price >= equal defined price (should be True)
    assert price_usd_100.gte(price_usd_100_other) is True
    
    # Test: smaller defined price >= larger defined price (should be False)
    assert price_usd_50.gte(price_usd_100) is False
    
    # Test: defined price >= undefined price (should be True)
    assert price_usd_100.gte(undefined_price) is True
    
    # Test: undefined price >= defined price (should be False)
    assert undefined_price.gte(price_usd_100) is False
    
    # Test: undefined price >= undefined price (should be True)
    assert undefined_price.gte(undefined_price) is True


# LLM-generated content at query #5
#--------------------------

```python
def test_qty_or_zero():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    
    somemoney = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    result_defined = somemoney.qty_or_zero()
    assert result_defined == Decimal('1.00')
    
    nonemoney = Money.of(None, Decimal('1'), None)
    result_undefined = nonemoney.qty_or_zero()
    assert result_undefined == Decimal('0')


# LLM-generated content at query #6
#--------------------------

```python
def test_money_is_equal():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    # Create defined money objects with same values
    money1 = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    
    # Create defined money objects with different currency
    money3 = Money.of(Currencies["EUR"], Decimal('100'), Date(2019, 1, 1))
    
    # Create defined money objects with different quantity
    money4 = Money.of(Currencies["USD"], Decimal('200'), Date(2019, 1, 1))
    
    # Create defined money objects with different date
    money5 = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 2))
    
    # Create undefined money objects
    money_na1 = Money.na()
    money_na2 = Money.na()
    
    # Test equality of same defined money
    assert money1.is_equal(money2) is True
    
    # Test inequality with different currency
    assert money1.is_equal(money3) is False
    
    # Test inequality with different quantity
    assert money1.is_equal(money4) is False
    
    # Test inequality with different date
    assert money1.is_equal(money5) is False
    
    # Test inequality between defined and undefined
    assert money1.is_equal(money_na1) is False
    
    # Test equality of undefined money objects
    assert money_na1.is_equal(money_na2) is True
    
    # Test inequality with non-money object
    assert money1.is_equal("not a money") is False
    assert money1.is_equal(100) is False
    assert money1.is_equal(None) is False


# LLM-generated content at query #7
#--------------------------

```python
def test_with_dov():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    
    # Test with_dov on defined money
    defined_money = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    new_date = Date(2020, 6, 15)
    result = defined_money.with_dov(new_date)
    assert result.dov_or_none() == new_date
    assert result.ccy_or_none().code == "USD"
    assert result.qty_or_zero() == Decimal('100.00')
    
    # Test with_dov on undefined money
    undefined_money = Money.na()
    result_undefined = undefined_money.with_dov(new_date)
    assert result_undefined.undefined
    assert result_undefined is undefined_money


# LLM-generated content at query #8
#--------------------------

```python
def test_money_add_two_defined_money_objects_with_same_currency():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    from pypara.money import Money
    
    money1 = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('20'), Date(2019, 1, 1))
    result = money1.add(money2)
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('30.00')
    assert result.ccy_or_none().code == "USD"


def test_money_add_defined_with_undefined_money():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    from pypara.money import Money
    
    money1 = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    money2 = Money.na()
    result = money1.add(money2)
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('10.00')


def test_money_add_undefined_with_defined_money():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    from pypara.money import Money
    
    money1 = Money.na()
    money2 = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = money1.add(money2)
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('10.00')


def test_money_add_two_undefined_money_objects():
    from pypara.money import Money
    
    money1 = Money.na()
    money2 = Money.na()
    result = money1.add(money2)
    
    assert result.undefined


def test_money_add_with_different_currencies_raises_error():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    from pypara.money import Money
    from pypara.exceptions import IncompatibleCurrencyError
    
    money1 = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["EUR"], Decimal('20'), Date(2019, 1, 1))
    
    try:
        result = money1.add(money2)
        assert False, "Should have raised IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass


def test_money_add_carries_forward_date():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    from pypara.money import Money
    
    money1 = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('20'), Date(2019, 1, 5))
    result = money1.add(money2)
    
    assert result.dov_or_none() == Date(2019, 1, 5)


def test_money_add_negative_quantities():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    from pypara.money import Money
    
    money1 = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('-5'), Date(2019, 1, 1))
    result = money1.add(money2)
    
    assert result.qty_or_zero() == Decimal('5.00')


def test_money_add_zero_quantities():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    from pypara.money import Money
    
    money1 = Money.of(Currencies["USD"], Decimal('0'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = money1.add(money2)
    
    assert result.qty_or_zero() == Decimal('10.00')


# LLM-generated content at query #9
#--------------------------

```python
def test_price_truediv():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    # Test division of defined price by numeric value
    price = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = price / Decimal('2')
    assert result.defined
    assert result.qty_or_zero() == Decimal('5')
    assert result.ccy_or_none().code == "USD"
    
    # Test division by zero yields undefined price
    price = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = price / Decimal('0')
    assert result.undefined
    
    # Test division of undefined price returns itself
    price = Price.na()
    result = price / Decimal('2')
    assert result.undefined
    
    # Test division with float
    price = Price.of(Currencies["EUR"], Decimal('20'), Date(2019, 6, 15))
    result = price / 4.0
    assert result.defined
    assert result.qty_or_zero() == Decimal('5')
    
    # Test division with integer
    price = Price.of(Currencies["GBP"], Decimal('15'), Date(2019, 3, 20))
    result = price / 3
    assert result.defined
    assert result.qty_or_zero() == Decimal('5')


# LLM-generated content at query #10
#--------------------------

```python
def test_dov_or():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    somemoney = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    result1 = somemoney.dov_or(Date(2001, 1, 1))
    assert result1 == Date(2019, 1, 1)
    
    nonemoney = Money.of(None, None, Date(2019, 1, 1))
    result2 = nonemoney.dov_or(Date(2001, 1, 1))
    assert result2 == Date(2001, 1, 1)
    
    undefined_money = Money.na()
    result3 = undefined_money.dov_or(Date(2020, 6, 15))
    assert result3 == Date(2020, 6, 15)


# LLM-generated content at query #11
#--------------------------

```python
def test_someprice_le():
    from decimal import Decimal
    from datetime import date
    
    # Create test currencies and prices
    usd = Currency(code="USD", quantizer=Decimal("0.01"))
    eur = Currency(code="EUR", quantizer=Decimal("0.01"))
    
    price1 = SomePrice(usd, Decimal("100.00"), date(2024, 1, 1))
    price2 = SomePrice(usd, Decimal("100.00"), date(2024, 1, 1))
    price3 = SomePrice(usd, Decimal("150.00"), date(2024, 1, 1))
    price4 = SomePrice(usd, Decimal("50.00"), date(2024, 1, 1))
    price_eur = SomePrice(eur, Decimal("100.00"), date(2024, 1, 1))
    
    # Test equal quantities
    assert price1 <= price2 is True
    
    # Test less than
    assert price4 <= price1 is True
    
    # Test greater than
    assert price3 <= price1 is False
    
    # Test with non-SomePrice object
    assert price1 <= "not a price" is False
    
    # Test incompatible currency raises error
    try:
        price1 <= price_eur
        assert False, "Should raise IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass


# LLM-generated content at query #12
#--------------------------

```python
def test_qty_map_defined_price():
    from decimal import Decimal
    from datetime import date as Date
    from pypara.currencies import Currencies
    from pypara.price import Price
    
    someprice = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    result = someprice.qty_map(lambda x: x + Decimal('1'), lambda: Decimal('42'))
    assert result == Decimal('2')


def test_qty_map_undefined_price():
    from decimal import Decimal
    from pypara.price import Price
    
    noneprice = Price.of(None, Decimal('1'), None)
    result = noneprice.qty_map(lambda x: x + Decimal('1'), lambda: Decimal('42'))
    assert result == Decimal('42')


def test_qty_map_defined_price_with_different_function():
    from decimal import Decimal
    from datetime import date as Date
    from pypara.currencies import Currencies
    from pypara.price import Price
    
    someprice = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = someprice.qty_map(lambda x: x * Decimal('2'), lambda: Decimal('0'))
    assert result == Decimal('20')


def test_qty_map_undefined_price_calls_else_function():
    from decimal import Decimal
    from pypara.price import Price
    
    noneprice = Price.of(None, Decimal('5'), None)
    result = noneprice.qty_map(lambda x: x * Decimal('10'), lambda: Decimal('99'))
    assert result == Decimal('99')


def test_qty_map_defined_price_with_string_return():
    from decimal import Decimal
    from datetime import date as Date
    from pypara.currencies import Currencies
    from pypara.price import Price
    
    someprice = Price.of(Currencies["USD"], Decimal('5'), Date(2019, 1, 1))
    result = someprice.qty_map(lambda x: str(x), lambda: "fallback")
    assert result == "5"


def test_qty_map_undefined_price_with_string_return():
    from decimal import Decimal
    from pypara.price import Price
    
    noneprice = Price.of(None, Decimal('5'), None)
    result = noneprice.qty_map(lambda x: str(x), lambda: "fallback")
    assert result == "fallback"


# LLM-generated content at query #13
#--------------------------

```python
def test_money_truediv():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    # Test division of defined money by a numeric value
    money = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    result = money / Decimal('2')
    assert result.qty == Decimal('50.00')
    assert result.ccy.code == "USD"
    
    # Test division by zero returns undefined money
    money = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    result = money / Decimal('0')
    assert result.undefined
    
    # Test division of undefined money returns itself
    undefined_money = Money.na()
    result = undefined_money / Decimal('2')
    assert result.undefined
    
    # Test division with integer
    money = Money.of(Currencies["EUR"], Decimal('50'), Date(2019, 1, 1))
    result = money / 5
    assert result.qty == Decimal('10.00')
    
    # Test division with float
    money = Money.of(Currencies["GBP"], Decimal('100'), Date(2019, 1, 1))
    result = money / 2.5
    assert result.qty == Decimal('40.00')


# LLM-generated content at query #14
#--------------------------

```python
def test_lte_raises_incompatible_currency_error_when_currencies_differ():
    from datetime import date
    from decimal import Decimal
    
    # Create mock Currency objects
    class MockCurrency:
        def __init__(self, code):
            self.code = code
        
        def __eq__(self, other):
            return self.code == other.code
        
        def __ne__(self, other):
            return self.code != other.code
    
    ccy1 = MockCurrency("USD")
    ccy2 = MockCurrency("EUR")
    
    price1 = SomePrice(ccy1, Decimal("100"), date(2023, 1, 1))
    price2 = SomePrice(ccy2, Decimal("100"), date(2023, 1, 1))
    
    try:
        price1.lte(price2)
        assert False, "Expected IncompatibleCurrencyError to be raised"
    except IncompatibleCurrencyError as e:
        assert e.args[0]["ccy1"] == ccy1
        assert e.args[0]["ccy2"] == ccy2
        assert e.args[0]["operation"] == "<= comparison"


# LLM-generated content at query #15
#--------------------------

```python
def test_somemoney_add():
    from decimal import Decimal
    from datetime import date
    
    # Create test currencies and money objects
    usd = Currency(code="USD", decimals=2)
    eur = Currency(code="EUR", decimals=2)
    
    money1 = SomeMoney(usd, Decimal("100.00"), date(2024, 1, 1))
    money2 = SomeMoney(usd, Decimal("50.00"), date(2024, 1, 15))
    
    # Test adding two SomeMoney objects with same currency
    result = money1 + money2
    assert result.ccy == usd
    assert result.qty == Decimal("150.00")
    assert result.dov == date(2024, 1, 15)
    
    # Test adding with later date
    money3 = SomeMoney(usd, Decimal("25.00"), date(2024, 2, 1))
    result2 = money1 + money3
    assert result2.qty == Decimal("125.00")
    assert result2.dov == date(2024, 2, 1)
    
    # Test adding with same date
    money4 = SomeMoney(usd, Decimal("30.00"), date(2024, 1, 1))
    result3 = money1 + money4
    assert result3.qty == Decimal("130.00")
    assert result3.dov == date(2024, 1, 1)
    
    # Test adding with different currencies raises exception
    money_eur = SomeMoney(eur, Decimal("100.00"), date(2024, 1, 1))
    try:
        money1 + money_eur
        assert False, "Should raise IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass
    
    # Test adding with undefined money (NoMoney)
    result4 = money1 + NoMoney
    assert result4 == money1


# LLM-generated content at query #16
#--------------------------

```python
def test_qty_or():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.money import Money
    
    somemoney = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    result1 = somemoney.qty_or(Decimal(0))
    assert result1 == Decimal('1.00')
    
    nonemoney = Money.of(None, Decimal('1'), None)
    result2 = nonemoney.qty_or(Decimal(0))
    assert result2 == Decimal('0')
    
    somemoney2 = Money.of(Currencies["EUR"], Decimal('42.50'), Date(2020, 6, 15))
    result3 = somemoney2.qty_or(Decimal(100))
    assert result3 == Decimal('42.50')
    
    nonemoney2 = Money.of(None, Decimal('99'), None)
    result4 = nonemoney2.qty_or(Decimal(55))
    assert result4 == Decimal('55')


# LLM-generated content at query #17
#--------------------------

```python
def test_price_pos():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    # Test __pos__ on a defined price with positive quantity
    positive_price = Price.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    result_positive = +positive_price
    assert result_positive.qty == Decimal('100')
    assert result_positive.ccy.code == 'USD'
    assert result_positive.dov == Date(2019, 1, 1)
    
    # Test __pos__ on a defined price with negative quantity
    negative_price = Price.of(Currencies["EUR"], Decimal('-50'), Date(2020, 6, 15))
    result_negative = +negative_price
    assert result_negative.qty == Decimal('-50')
    assert result_negative.ccy.code == 'EUR'
    assert result_negative.dov == Date(2020, 6, 15)
    
    # Test __pos__ on an undefined price
    undefined_price = Price.na()
    result_undefined = +undefined_price
    assert result_undefined.undefined is True
    assert result_undefined is undefined_price


# LLM-generated content at query #18
#--------------------------

```python
def test_as_integer():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    # Test with defined money - positive value
    defined_money = Money.of(Currencies["USD"], Decimal('42.75'), Date(2019, 1, 1))
    result = defined_money.as_integer()
    assert result == 42
    
    # Test with defined money - negative value
    defined_money_negative = Money.of(Currencies["USD"], Decimal('-42.75'), Date(2019, 1, 1))
    result_negative = defined_money_negative.as_integer()
    assert result_negative == -42
    
    # Test with defined money - zero value
    defined_money_zero = Money.of(Currencies["USD"], Decimal('0'), Date(2019, 1, 1))
    result_zero = defined_money_zero.as_integer()
    assert result_zero == 0
    
    # Test with undefined money - should raise MonetaryOperationException
    undefined_money = Money.na()
    try:
        undefined_money.as_integer()
        assert False, "Expected MonetaryOperationException"
    except Exception as e:
        assert "MonetaryOperationException" in str(type(e).__name__)


# LLM-generated content at query #19
#--------------------------

```python
def test_as_boolean():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.money import Money

    # Test undefined money returns False
    undefined_money = Money.na()
    assert undefined_money.as_boolean() is False

    # Test defined money with zero quantity returns False
    zero_money = Money.of(Currencies["USD"], Decimal('0'), Date(2019, 1, 1))
    assert zero_money.as_boolean() is False

    # Test defined money with positive quantity returns True
    positive_money = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert positive_money.as_boolean() is True

    # Test defined money with negative quantity returns True
    negative_money = Money.of(Currencies["USD"], Decimal('-1'), Date(2019, 1, 1))
    assert negative_money.as_boolean() is True

    # Test defined money with small positive quantity returns True
    small_money = Money.of(Currencies["EUR"], Decimal('0.01'), Date(2020, 6, 15))
    assert small_money.as_boolean() is True


# LLM-generated content at query #20
#--------------------------

```python
def test_floordiv_with_valid_divisor():
    from decimal import Decimal
    from datetime import date
    
    # Mock Currency class
    class MockCurrency:
        def __init__(self, code, decimals=2):
            self.code = code
            self.decimals = decimals
            self.quantizer = Decimal(10) ** -decimals
    
    # Create test data
    ccy = MockCurrency("USD", 2)
    qty = Decimal("100.00")
    dov = date(2023, 1, 1)
    
    # Create SomeMoney instance
    money = SomeMoney(ccy, qty, dov)
    
    # Test floor division with valid divisor
    result = money.__floordiv__(3)
    
    assert isinstance(result, SomeMoney)
    assert result.ccy == ccy
    assert result.qty == Decimal("33.00")
    assert result.dov == dov


def test_floordiv_with_zero_divisor():
    from decimal import Decimal
    from datetime import date
    
    # Mock Currency class
    class MockCurrency:
        def __init__(self, code, decimals=2):
            self.code = code
            self.decimals = decimals
            self.quantizer = Decimal(10) ** -decimals
    
    # Create test data
    ccy = MockCurrency("USD", 2)
    qty = Decimal("100.00")
    dov = date(2023, 1, 1)
    
    # Create SomeMoney instance
    money = SomeMoney(ccy, qty, dov)
    
    # Test floor division by zero
    result = money.__floordiv__(0)
    
    # Should return NoMoney
    assert result is NoMoney


def test_floordiv_with_decimal_divisor():
    from decimal import Decimal
    from datetime import date
    
    # Mock Currency class
    class MockCurrency:
        def __init__(self, code, decimals=2):
            self.code = code
            self.decimals = decimals
            self.quantizer = Decimal(10) ** -decimals
    
    # Create test data
    ccy = MockCurrency("USD", 2)
    qty = Decimal("50.00")
    dov = date(2023, 1, 1)
    
    # Create SomeMoney instance
    money = SomeMoney(ccy, qty, dov)
    
    # Test floor division with decimal divisor
    result = money.__floordiv__(Decimal("1.5"))
    
    assert isinstance(result, SomeMoney)
    assert result.ccy == ccy
    assert result.qty == Decimal("33.00")
    assert result.dov == dov


# LLM-generated content at query #21
#--------------------------

```python
def test_price_float():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    # Test that __float__ returns float representation of quantity for defined price
    defined_price = Price.of(Currencies["USD"], Decimal('123.45'), Date(2019, 1, 1))
    result = float(defined_price)
    assert isinstance(result, float)
    assert result == 123.45
    
    # Test that __float__ raises MonetaryOperationException for undefined price
    undefined_price = Price.na()
    try:
        float(undefined_price)
        assert False, "Expected MonetaryOperationException"
    except Exception as e:
        assert "MonetaryOperationException" in str(type(e).__name__)
    
    # Test with negative quantity
    negative_price = Price.of(Currencies["EUR"], Decimal('-50.75'), Date(2020, 6, 15))
    result = float(negative_price)
    assert result == -50.75
    
    # Test with zero quantity
    zero_price = Price.of(Currencies["GBP"], Decimal('0'), Date(2021, 3, 10))
    result = float(zero_price)
    assert result == 0.0


# LLM-generated content at query #22
#--------------------------

```python
def test_qty_or_zero():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    
    # Test with defined money
    somemoney = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    result = somemoney.qty_or_zero()
    assert result == Decimal('1.00')
    
    # Test with undefined money
    nonemoney = Money.of(None, Decimal('1'), None)
    result = nonemoney.qty_or_zero()
    assert result == Decimal('0')
    
    # Test with defined money with different quantity
    somemoney2 = Money.of(Currencies["EUR"], Decimal('42.50'), Date(2020, 6, 15))
    result2 = somemoney2.qty_or_zero()
    assert result2 == Decimal('42.50')
    
    # Test with undefined money created via Money.na()
    nomoney = Money.na()
    result3 = nomoney.qty_or_zero()
    assert result3 == Decimal('0')


# LLM-generated content at query #23
#--------------------------

```python
def test_lt_with_same_currency_less_than():
    from decimal import Decimal
    from datetime import date
    
    ccy = Currency(code="USD", quantizer=Decimal("0.01"))
    price1 = SomePrice(ccy, Decimal("100.00"), date(2024, 1, 1))
    price2 = SomePrice(ccy, Decimal("200.00"), date(2024, 1, 1))
    
    result = price1.lt(price2)
    assert result is True


def test_lt_with_same_currency_not_less_than():
    from decimal import Decimal
    from datetime import date
    
    ccy = Currency(code="USD", quantizer=Decimal("0.01"))
    price1 = SomePrice(ccy, Decimal("200.00"), date(2024, 1, 1))
    price2 = SomePrice(ccy, Decimal("100.00"), date(2024, 1, 1))
    
    result = price1.lt(price2)
    assert result is False


def test_lt_with_same_currency_equal_quantities():
    from decimal import Decimal
    from datetime import date
    
    ccy = Currency(code="USD", quantizer=Decimal("0.01"))
    price1 = SomePrice(ccy, Decimal("100.00"), date(2024, 1, 1))
    price2 = SomePrice(ccy, Decimal("100.00"), date(2024, 1, 1))
    
    result = price1.lt(price2)
    assert result is False


def test_lt_with_different_currency_raises_error():
    from decimal import Decimal
    from datetime import date
    
    ccy_usd = Currency(code="USD", quantizer=Decimal("0.01"))
    ccy_eur = Currency(code="EUR", quantizer=Decimal("0.01"))
    price1 = SomePrice(ccy_usd, Decimal("100.00"), date(2024, 1, 1))
    price2 = SomePrice(ccy_eur, Decimal("100.00"), date(2024, 1, 1))
    
    try:
        price1.lt(price2)
        assert False, "Expected IncompatibleCurrencyError"
    except IncompatibleCurrencyError as e:
        assert e.ccy1 == ccy_usd
        assert e.ccy2 == ccy_eur
        assert "< comparison" in str(e.operation)


def test_lt_with_non_some_price_returns_false():
    from decimal import Decimal
    from datetime import date
    
    ccy = Currency(code="USD", quantizer=Decimal("0.01"))
    price = SomePrice(ccy, Decimal("100.00"), date(2024, 1, 1))
    
    result = price.lt("not a price")
    assert result is False


def test_lt_with_no_price_returns_false():
    from decimal import Decimal
    from datetime import date
    
    ccy = Currency(code="USD", quantizer=Decimal("0.01"))
    price = SomePrice(ccy, Decimal("100.00"), date(2024, 1, 1))
    
    result = price.lt(NoPrice)
    assert result is False


# LLM-generated content at query #24
#--------------------------

```python
def test_is_equal():
    from decimal import Decimal
    from datetime import date as Date
    from pypara.currencies import Currencies
    from pypara.price import Price
    
    # Test equality of two defined prices with same values
    price1 = Price.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    assert price1.is_equal(price2) is True
    
    # Test inequality of two defined prices with different quantities
    price3 = Price.of(Currencies["USD"], Decimal('200'), Date(2019, 1, 1))
    assert price1.is_equal(price3) is False
    
    # Test inequality of two defined prices with different currencies
    price4 = Price.of(Currencies["EUR"], Decimal('100'), Date(2019, 1, 1))
    assert price1.is_equal(price4) is False
    
    # Test inequality of two defined prices with different dates
    price5 = Price.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 2))
    assert price1.is_equal(price5) is False
    
    # Test equality of two undefined prices
    undefined1 = Price.na()
    undefined2 = Price.na()
    assert undefined1.is_equal(undefined2) is True
    
    # Test inequality of defined and undefined prices
    assert price1.is_equal(undefined1) is False
    assert undefined1.is_equal(price1) is False
    
    # Test inequality with non-price objects
    assert price1.is_equal("not a price") is False
    assert price1.is_equal(None) is False
    assert price1.is_equal(100) is False


# LLM-generated content at query #25
#--------------------------

```python
def test_floor_divide_defined_price_positive_divisor():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    price = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = price.floor_divide(Decimal('3'))
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('3')
    assert result.ccy_or_none().code == 'USD'


def test_floor_divide_defined_price_negative_divisor():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    price = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = price.floor_divide(Decimal('-3'))
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('-4')


def test_floor_divide_defined_price_by_one():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    price = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = price.floor_divide(Decimal('1'))
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('10')


def test_floor_divide_defined_price_by_zero():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    price = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = price.floor_divide(Decimal('0'))
    
    assert result.undefined


def test_floor_divide_undefined_price():
    price = Price.na()
    result = price.floor_divide(Decimal('5'))
    
    assert result is price
    assert result.undefined


def test_floor_divide_negative_quantity():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    price = Price.of(Currencies["USD"], Decimal('-10'), Date(2019, 1, 1))
    result = price.floor_divide(Decimal('3'))
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('-4')


def test_floor_divide_decimal_quantity():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    price = Price.of(Currencies["USD"], Decimal('7.5'), Date(2019, 1, 1))
    result = price.floor_divide(Decimal('2'))
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('3')


def test_floor_divide_preserves_currency():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    price = Price.of(Currencies["EUR"], Decimal('15'), Date(2019, 1, 1))
    result = price.floor_divide(Decimal('4'))
    
    assert result.defined
    assert result.ccy_or_none().code == 'EUR'
    assert result.qty_or_zero() == Decimal('3')


# LLM-generated content at query #26
#--------------------------

```python
def test_price_gte():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    # Create test prices
    usd_100 = Price.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    usd_50 = Price.of(Currencies["USD"], Decimal('50'), Date(2019, 1, 1))
    usd_100_other = Price.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 2))
    undefined_price = Price.na()
    
    # Test: defined price >= smaller defined price (same currency)
    assert usd_100.gte(usd_50) is True
    
    # Test: defined price >= equal defined price (same currency)
    assert usd_100.gte(usd_100_other) is True
    
    # Test: smaller defined price >= larger defined price (same currency)
    assert usd_50.gte(usd_100) is False
    
    # Test: undefined price >= undefined price
    assert undefined_price.gte(undefined_price) is True
    
    # Test: defined price >= undefined price
    assert usd_100.gte(undefined_price) is True
    
    # Test: undefined price >= defined price
    assert undefined_price.gte(usd_100) is False


# LLM-generated content at query #27
#--------------------------

```python
def test_money_eq():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    # Create two identical money objects
    money1 = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    
    # Create a money object with different quantity
    money3 = Money.of(Currencies["USD"], Decimal('200'), Date(2019, 1, 1))
    
    # Create a money object with different currency
    money4 = Money.of(Currencies["EUR"], Decimal('100'), Date(2019, 1, 1))
    
    # Create a money object with different date
    money5 = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 2))
    
    # Create two undefined money objects
    none_money1 = Money.na()
    none_money2 = Money.na()
    
    # Test equality of identical money objects
    assert money1 == money2
    
    # Test inequality with different quantity
    assert not (money1 == money3)
    
    # Test inequality with different currency
    assert not (money1 == money4)
    
    # Test inequality with different date
    assert not (money1 == money5)
    
    # Test equality of undefined money objects
    assert none_money1 == none_money2
    
    # Test inequality between defined and undefined money
    assert not (money1 == none_money1)
    
    # Test inequality with non-Money objects
    assert not (money1 == "100")
    assert not (money1 == 100)
    assert not (money1 == None)


# LLM-generated content at query #28
#--------------------------

```python
def test_ccy_or_defined_money():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from datetime import date as Date
    from decimal import Decimal
    
    somemoney = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    result = somemoney.ccy_or(Currencies["EUR"])
    assert result.code == 'USD'


def test_ccy_or_undefined_money():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from datetime import date as Date
    from decimal import Decimal
    
    nonemoney = Money.of(Currencies["USD"], None, None)
    result = nonemoney.ccy_or(Currencies["EUR"])
    assert result.code == 'EUR'


def test_ccy_or_with_none_currency():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from datetime import date as Date
    from decimal import Decimal
    
    nonemoney = Money.of(None, Decimal('1'), None)
    result = nonemoney.ccy_or(Currencies["USD"])
    assert result.code == 'USD'


def test_ccy_or_with_none_quantity():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from datetime import date as Date
    from decimal import Decimal
    
    nonemoney = Money.of(Currencies["GBP"], None, Date(2019, 1, 1))
    result = nonemoney.ccy_or(Currencies["EUR"])
    assert result.code == 'EUR'


def test_ccy_or_with_none_date():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from datetime import date as Date
    from decimal import Decimal
    
    nonemoney = Money.of(Currencies["JPY"], Decimal('100'), None)
    result = nonemoney.ccy_or(Currencies["CHF"])
    assert result.code == 'CHF'


# LLM-generated content at query #29
#--------------------------

```python
def test_as_boolean():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.money import Money

    # Test as_boolean returns False for undefined money
    undefined_money = Money.na()
    assert undefined_money.as_boolean() is False

    # Test as_boolean returns False for defined money with zero quantity
    zero_money = Money.of(Currencies["USD"], Decimal('0'), Date(2019, 1, 1))
    assert zero_money.as_boolean() is False

    # Test as_boolean returns True for defined money with positive quantity
    positive_money = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert positive_money.as_boolean() is True

    # Test as_boolean returns True for defined money with negative quantity
    negative_money = Money.of(Currencies["USD"], Decimal('-5'), Date(2019, 1, 1))
    assert negative_money.as_boolean() is True

    # Test as_boolean returns True for defined money with small positive quantity
    small_positive_money = Money.of(Currencies["EUR"], Decimal('0.01'), Date(2020, 6, 15))
    assert small_positive_money.as_boolean() is True


# LLM-generated content at query #30
#--------------------------

```python
def test_multiply_defined_price_by_positive_scalar():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    price = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = price.multiply(Decimal('2'))
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('20')
    assert result.ccy_or_none().code == 'USD'
    assert result.dov_or_none() == Date(2019, 1, 1)


def test_multiply_defined_price_by_negative_scalar():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    price = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = price.multiply(Decimal('-3'))
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('-30')
    assert result.ccy_or_none().code == 'USD'


def test_multiply_defined_price_by_zero():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    price = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = price.multiply(Decimal('0'))
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('0')


def test_multiply_defined_price_by_fractional_scalar():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    price = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = price.multiply(Decimal('0.5'))
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('5')


def test_multiply_undefined_price_returns_itself():
    from decimal import Decimal
    
    price = Price.na()
    result = price.multiply(Decimal('5'))
    
    assert result.undefined
    assert result is price or result.undefined


def test_multiply_defined_price_by_integer():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    price = Price.of(Currencies["EUR"], Decimal('25'), Date(2020, 6, 15))
    result = price.multiply(4)
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('100')
    assert result.ccy_or_none().code == 'EUR'


# LLM-generated content at query #31
#--------------------------

```python
def test_is_equal_defined_prices_same_values():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.price import Price
    
    price1 = Price.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    
    assert price1.is_equal(price2) is True


def test_is_equal_defined_prices_different_quantities():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.price import Price
    
    price1 = Price.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('200'), Date(2019, 1, 1))
    
    assert price1.is_equal(price2) is False


def test_is_equal_defined_prices_different_currencies():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.price import Price
    
    price1 = Price.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["EUR"], Decimal('100'), Date(2019, 1, 1))
    
    assert price1.is_equal(price2) is False


def test_is_equal_defined_prices_different_dates():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.price import Price
    
    price1 = Price.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 2))
    
    assert price1.is_equal(price2) is False


def test_is_equal_undefined_prices():
    from pypara.price import Price
    
    price1 = Price.na()
    price2 = Price.na()
    
    assert price1.is_equal(price2) is True


def test_is_equal_defined_and_undefined():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.price import Price
    
    price1 = Price.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    price2 = Price.na()
    
    assert price1.is_equal(price2) is False


def test_is_equal_with_non_price_object():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.price import Price
    
    price = Price.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    
    assert price.is_equal("not a price") is False
    assert price.is_equal(100) is False
    assert price.is_equal(None) is False


# LLM-generated content at query #32
#--------------------------

```python
def test_someprice_sub():
    from decimal import Decimal
    from datetime import date
    
    # Create mock Currency objects
    usd = type('Currency', (), {'__eq__': lambda self, other: True, '__hash__': lambda self: 1})()
    eur = type('Currency', (), {'__eq__': lambda self, other: False, '__hash__': lambda self: 2})()
    
    # Create SomePrice instances
    price1 = SomePrice(usd, Decimal('100.00'), date(2023, 1, 1))
    price2 = SomePrice(usd, Decimal('30.00'), date(2023, 1, 2))
    
    # Test subtraction with defined price
    result = price1.__sub__(price2)
    assert result.ccy == usd
    assert result.qty == Decimal('70.00')
    assert result.dov == date(2023, 1, 2)
    
    # Test subtraction returns self when other is undefined
    no_price = type('NoPrice', (), {'undefined': True})()
    result = price1.__sub__(no_price)
    assert result == price1
    
    # Test subtraction with same date
    price3 = SomePrice(usd, Decimal('40.00'), date(2023, 1, 1))
    result = price1.__sub__(price3)
    assert result.dov == date(2023, 1, 1)
    
    # Test subtraction with incompatible currencies raises error
    price_eur = SomePrice(eur, Decimal('50.00'), date(2023, 1, 1))
    try:
        price1.__sub__(price_eur)
        assert False, "Should have raised IncompatibleCurrencyError"
    except Exception as e:
        assert "subtraction" in str(e).lower() or "IncompatibleCurrencyError" in str(type(e).__name__)


# LLM-generated content at query #33
#--------------------------

```python
def test_price_or_else_with_defined_price():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    fallback = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    someprice = Price.of(Currencies["EUR"], Decimal('2'), Date(2019, 1, 2))
    result = someprice.or_else(lambda: fallback)
    
    assert result is someprice


def test_price_or_else_with_undefined_price():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    fallback = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    noneprice = Price.of(None, Decimal('1'), None)
    result = noneprice.or_else(lambda: fallback)
    
    assert result is fallback


def test_price_or_else_returns_self_when_defined():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    price = Price.of(Currencies["GBP"], Decimal('10'), Date(2020, 5, 15))
    fallback = Price.of(Currencies["USD"], Decimal('5'), Date(2020, 5, 15))
    result = price.or_else(lambda: fallback)
    
    assert result is price


def test_price_or_else_calls_combinator_when_undefined():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    fallback = Price.of(Currencies["JPY"], Decimal('100'), Date(2021, 3, 10))
    undefined_price = Price.na()
    result = undefined_price.or_else(lambda: fallback)
    
    assert result is fallback


# LLM-generated content at query #34
#--------------------------

```python
def test_price_gt():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    # Create defined price objects
    price_usd_100 = Price.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    price_usd_50 = Price.of(Currencies["USD"], Decimal('50'), Date(2019, 1, 1))
    price_usd_100_other_date = Price.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 2))
    
    # Create undefined price object
    undefined_price = Price.na()
    
    # Test 1: Defined price greater than smaller defined price
    assert price_usd_100.gt(price_usd_50) is True
    
    # Test 2: Defined price not greater than equal defined price
    assert price_usd_100.gt(price_usd_100_other_date) is False
    
    # Test 3: Defined price not greater than larger defined price
    assert price_usd_50.gt(price_usd_100) is False
    
    # Test 4: Defined price is always greater than undefined price
    assert price_usd_100.gt(undefined_price) is True
    assert price_usd_50.gt(undefined_price) is True
    
    # Test 5: Undefined price is never greater than defined price
    assert undefined_price.gt(price_usd_100) is False
    
    # Test 6: Undefined price is never greater than undefined price
    assert undefined_price.gt(undefined_price) is False


# LLM-generated content at query #35
#--------------------------

```python
def test_someprice_sub():
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
    
    # Create test dates
    date1 = date(2023, 1, 1)
    date2 = date(2023, 1, 2)
    
    # Create SomePrice instances
    price1 = SomePrice(usd, Decimal("100"), date1)
    price2 = SomePrice(usd, Decimal("30"), date1)
    price_different_ccy = SomePrice(eur, Decimal("50"), date1)
    price_later_date = SomePrice(usd, Decimal("20"), date2)
    
    # Test normal subtraction with same currency
    result = price1.__sub__(price2)
    assert result.ccy == usd
    assert result.qty == Decimal("70")
    assert result.dov == date1
    
    # Test subtraction with different dates (should use later date)
    result = price1.__sub__(price_later_date)
    assert result.ccy == usd
    assert result.qty == Decimal("80")
    assert result.dov == date2
    
    # Test subtraction with undefined price should return self
    class MockUndefinedPrice:
        undefined = True
    
    undefined_price = MockUndefinedPrice()
    result = price1.__sub__(undefined_price)
    assert result == price1
    
    # Test subtraction with incompatible currency raises error
    try:
        price1.__sub__(price_different_ccy)
        assert False, "Should raise IncompatibleCurrencyError"
    except Exception as e:
        assert "IncompatibleCurrencyError" in str(type(e).__name__)


# LLM-generated content at query #36
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
    
    # Test: undefined price never greater than defined price
    assert undefined_price.gt(defined_price) is False
    
    # Test: undefined price not greater than undefined price
    undefined_price2 = Price.na()
    assert undefined_price.gt(undefined_price2) is False
    
    # Test: defined price greater than another defined price with same currency
    price1 = Price.of(Currencies["USD"], Decimal('20'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 2))
    assert price1.gt(price2) is True
    assert price2.gt(price1) is False
    
    # Test: defined price not greater than equal defined price
    price3 = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    price4 = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 2))
    assert price3.gt(price4) is False
    assert price4.gt(price3) is False


# LLM-generated content at query #37
#--------------------------

```python
def test_price_ge():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    # Create test prices
    price_usd_100 = Price.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    price_usd_50 = Price.of(Currencies["USD"], Decimal('50'), Date(2019, 1, 1))
    price_usd_100_other = Price.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 2))
    undefined_price = Price.na()
    
    # Test: defined price >= smaller defined price
    assert price_usd_100 >= price_usd_50
    
    # Test: defined price >= equal defined price
    assert price_usd_100 >= price_usd_100_other
    
    # Test: smaller defined price is not >= larger defined price
    assert not (price_usd_50 >= price_usd_100)
    
    # Test: defined price >= undefined price
    assert price_usd_100 >= undefined_price
    
    # Test: undefined price >= undefined price
    assert undefined_price >= undefined_price
    
    # Test: undefined price is not >= defined price
    assert not (undefined_price >= price_usd_100)


# LLM-generated content at query #38
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
    assert result1.ccy_or_none().code == 'USD'
    
    # Test floor division with undefined price
    undefined_price = Price.na()
    result2 = undefined_price // Decimal('2')
    assert result2.undefined
    
    # Test floor division by zero yields undefined price
    price2 = Price.of(Currencies["EUR"], Decimal('5'), Date(2019, 1, 2))
    result3 = price2 // Decimal('0')
    assert result3.undefined
    
    # Test floor division with negative divisor
    price3 = Price.of(Currencies["GBP"], Decimal('7'), Date(2019, 1, 3))
    result4 = price3 // Decimal('-2')
    assert result4.qty_or_zero() == Decimal('-4')
    
    # Test floor division with decimal divisor
    price4 = Price.of(Currencies["JPY"], Decimal('100'), Date(2019, 1, 4))
    result5 = price4 // Decimal('2.5')
    assert result5.qty_or_zero() == Decimal('40')


# LLM-generated content at query #39
#--------------------------

```python
def test_money_truediv_defined_money_by_positive_numeric():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    result = money / Decimal('2')
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('50.00')
    assert result.ccy_or_none().code == 'USD'


def test_money_truediv_defined_money_by_negative_numeric():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    result = money / Decimal('-2')
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('-50.00')


def test_money_truediv_defined_money_by_zero():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    result = money / Decimal('0')
    
    assert result.undefined


def test_money_truediv_undefined_money():
    from decimal import Decimal
    
    money = Money.na()
    result = money / Decimal('2')
    
    assert result.undefined


def test_money_truediv_defined_money_by_integer():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    result = money / 4
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('25.00')


def test_money_truediv_defined_money_by_float():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    result = money / 2.5
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('40.00')


# LLM-generated content at query #40
#--------------------------

```python
def test_negative():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.money import Money
    
    # Test negative of a defined money with positive quantity
    positive_money = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    negative_result = positive_money.negative()
    assert negative_result.qty == Decimal('-100.00')
    assert negative_result.ccy == Currencies["USD"]
    assert negative_result.dov == Date(2019, 1, 1)
    
    # Test negative of a defined money with negative quantity
    negative_money = Money.of(Currencies["EUR"], Decimal('-50'), Date(2019, 1, 15))
    positive_result = negative_money.negative()
    assert positive_result.qty == Decimal('50.00')
    assert positive_result.ccy == Currencies["EUR"]
    assert positive_result.dov == Date(2019, 1, 15)
    
    # Test negative of zero quantity
    zero_money = Money.of(Currencies["GBP"], Decimal('0'), Date(2019, 2, 1))
    zero_negative = zero_money.negative()
    assert zero_negative.qty == Decimal('0.00')
    
    # Test negative of undefined money returns itself
    undefined_money = Money.na()
    undefined_negative = undefined_money.negative()
    assert undefined_negative is undefined_money
    assert undefined_negative.undefined is True


# LLM-generated content at query #41
#--------------------------

```python
def test_money_float_conversion():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    # Test defined money converts to float
    defined_money = Money.of(Currencies["USD"], Decimal('123.45'), Date(2019, 1, 1))
    result = float(defined_money)
    assert isinstance(result, float)
    assert result == 123.45
    
    # Test undefined money raises exception
    undefined_money = Money.na()
    try:
        float(undefined_money)
        assert False, "Should raise MonetaryOperationException"
    except Exception as e:
        assert "MonetaryOperationException" in str(type(e))
    
    # Test negative money converts to negative float
    negative_money = Money.of(Currencies["EUR"], Decimal('-50.25'), Date(2020, 6, 15))
    result = float(negative_money)
    assert result == -50.25
    
    # Test zero money converts to 0.0
    zero_money = Money.of(Currencies["GBP"], Decimal('0'), Date(2021, 12, 31))
    result = float(zero_money)
    assert result == 0.0


# LLM-generated content at query #42
#--------------------------

```python
def test_as_integer():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    # Test with a defined price
    defined_price = Price.of(Currencies["USD"], Decimal('42.7'), Date(2019, 1, 1))
    result = defined_price.as_integer()
    assert result == 42
    
    # Test with undefined price should raise MonetaryOperationException
    undefined_price = Price.na()
    try:
        undefined_price.as_integer()
        assert False, "Expected MonetaryOperationException"
    except Exception as e:
        assert "MonetaryOperationException" in str(type(e).__name__)


# LLM-generated content at query #43
#--------------------------

```python
def test_somemoney_ge():
    from decimal import Decimal
    from datetime import date
    
    # Create test currencies and money objects
    usd = Currency(code="USD", decimals=2)
    eur = Currency(code="EUR", decimals=2)
    
    test_date = date(2024, 1, 1)
    
    money1 = SomeMoney(usd, Decimal("100.00"), test_date)
    money2 = SomeMoney(usd, Decimal("50.00"), test_date)
    money3 = SomeMoney(usd, Decimal("100.00"), test_date)
    money_eur = SomeMoney(eur, Decimal("100.00"), test_date)
    
    # Test greater than or equal with same currency, greater quantity
    assert money1 >= money2
    
    # Test greater than or equal with same currency, equal quantity
    assert money1 >= money3
    
    # Test greater than or equal with same currency, lesser quantity
    assert not (money2 >= money1)
    
    # Test greater than or equal with different currency raises exception
    try:
        money1 >= money_eur
        assert False, "Should raise IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass
    
    # Test greater than or equal with non-SomeMoney object returns True
    assert money1 >= "not money"
    assert money1 >= 100
    assert money1 >= None


# LLM-generated content at query #44
#--------------------------

```python
def test_qty_map():
    from datetime import date as Date
    from decimal import Decimal
    from pypara.currencies import Currencies
    from pypara.price import Price

    # Test qty_map with defined price
    someprice = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    result = someprice.qty_map(lambda x: x + Decimal('1'), lambda: Decimal('42'))
    assert result == Decimal('2')

    # Test qty_map with undefined price
    noneprice = Price.of(None, Decimal('1'), None)
    result = noneprice.qty_map(lambda x: x + Decimal('1'), lambda: Decimal('42'))
    assert result == Decimal('42')

    # Test qty_map with defined price and different function
    someprice = Price.of(Currencies["EUR"], Decimal('5'), Date(2020, 6, 15))
    result = someprice.qty_map(lambda x: x * Decimal('2'), lambda: Decimal('0'))
    assert result == Decimal('10')

    # Test qty_map with undefined price and alternative function
    noneprice = Price.of(None, Decimal('100'), None)
    result = noneprice.qty_map(lambda x: x / Decimal('2'), lambda: Decimal('999'))
    assert result == Decimal('999')

    # Test qty_map with defined price returning different type
    someprice = Price.of(Currencies["GBP"], Decimal('7'), Date(2021, 3, 10))
    result = someprice.qty_map(lambda x: str(x), lambda: "fallback")
    assert result == "7"

    # Test qty_map with undefined price returning different type
    noneprice = Price.of(None, Decimal('50'), None)
    result = noneprice.qty_map(lambda x: str(x), lambda: "fallback")
    assert result == "fallback"


# LLM-generated content at query #45
#--------------------------

```python
def test_someprice_ge_with_same_currency_and_greater_quantity():
    from decimal import Decimal
    from datetime import date
    
    ccy = Currency(code="USD")
    price1 = SomePrice(ccy, Decimal("100"), date(2024, 1, 1))
    price2 = SomePrice(ccy, Decimal("50"), date(2024, 1, 1))
    
    result = price1 >= price2
    assert result is True


def test_someprice_ge_with_same_currency_and_equal_quantity():
    from decimal import Decimal
    from datetime import date
    
    ccy = Currency(code="USD")
    price1 = SomePrice(ccy, Decimal("100"), date(2024, 1, 1))
    price2 = SomePrice(ccy, Decimal("100"), date(2024, 1, 1))
    
    result = price1 >= price2
    assert result is True


def test_someprice_ge_with_same_currency_and_lesser_quantity():
    from decimal import Decimal
    from datetime import date
    
    ccy = Currency(code="USD")
    price1 = SomePrice(ccy, Decimal("50"), date(2024, 1, 1))
    price2 = SomePrice(ccy, Decimal("100"), date(2024, 1, 1))
    
    result = price1 >= price2
    assert result is False


def test_someprice_ge_with_different_currency_raises_error():
    from decimal import Decimal
    from datetime import date
    
    ccy_usd = Currency(code="USD")
    ccy_eur = Currency(code="EUR")
    price1 = SomePrice(ccy_usd, Decimal("100"), date(2024, 1, 1))
    price2 = SomePrice(ccy_eur, Decimal("50"), date(2024, 1, 1))
    
    try:
        price1 >= price2
        assert False, "Expected IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        assert True


def test_someprice_ge_with_non_someprice_returns_true():
    from decimal import Decimal
    from datetime import date
    
    ccy = Currency(code="USD")
    price1 = SomePrice(ccy, Decimal("100"), date(2024, 1, 1))
    other = "not a price"
    
    result = price1 >= other
    assert result is True


# LLM-generated content at query #46
#--------------------------

```python
def test_dimap_with_defined_price():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    someprice = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    result = someprice.dimap(lambda x: x.ccy.code, lambda: "EUR")
    assert result == "USD"


def test_dimap_with_undefined_price():
    from datetime import date as Date
    from decimal import Decimal
    
    noneprice = Price.of(None, Decimal('1'), None)
    result = noneprice.dimap(lambda x: x.ccy.code, lambda: "EUR")
    assert result == "EUR"


def test_dimap_applies_function_to_defined_price():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    someprice = Price.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    result = someprice.dimap(lambda x: x.qty * Decimal('2'), lambda: Decimal('0'))
    assert result == Decimal('200')


def test_dimap_calls_else_combinator_for_undefined_price():
    from datetime import date as Date
    from decimal import Decimal
    
    noneprice = Price.of(None, Decimal('50'), None)
    result = noneprice.dimap(lambda x: x.qty + Decimal('10'), lambda: Decimal('999'))
    assert result == Decimal('999')


def test_dimap_with_date_extraction_from_defined_price():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    someprice = Price.of(Currencies["EUR"], Decimal('5'), Date(2020, 6, 15))
    result = someprice.dimap(lambda x: x.dov, lambda: Date(2000, 1, 1))
    assert result == Date(2020, 6, 15)


def test_dimap_with_date_extraction_from_undefined_price():
    from datetime import date as Date
    from decimal import Decimal
    
    noneprice = Price.of(None, Decimal('5'), None)
    result = noneprice.dimap(lambda x: x.dov, lambda: Date(2000, 1, 1))
    assert result == Date(2000, 1, 1)


# LLM-generated content at query #47
#--------------------------

```python
def test_somemoney_constructor():
    from decimal import Decimal
    from datetime import date
    
    # Create a Currency object (assuming it exists)
    usd = Currency(code="USD", decimals=2)
    eur = Currency(code="EUR", decimals=2)
    
    # Test basic constructor with valid arguments
    test_date = date(2024, 1, 15)
    money1 = SomeMoney(usd, Decimal("100.50"), test_date)
    
    assert money1.ccy == usd
    assert money1.qty == Decimal("100.50")
    assert money1.dov == test_date
    
    # Test constructor with different currency
    money2 = SomeMoney(eur, Decimal("50.00"), test_date)
    
    assert money2.ccy == eur
    assert money2.qty == Decimal("50.00")
    assert money2.dov == test_date
    
    # Test constructor with zero quantity
    money3 = SomeMoney(usd, Decimal("0"), test_date)
    
    assert money3.qty == Decimal("0")
    
    # Test constructor with negative quantity
    money4 = SomeMoney(usd, Decimal("-25.75"), test_date)
    
    assert money4.qty == Decimal("-25.75")
    
    # Test that defined property returns True
    assert money1.defined is True
    
    # Test that undefined property returns False
    assert money1.undefined is False
    
    # Test that tuple unpacking works (NamedTuple behavior)
    c, q, d = money1
    assert c == usd
    assert q == Decimal("100.50")
    assert d == test_date


# LLM-generated content at query #48
#--------------------------

```python
def test_price_negative():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    # Test negative on a defined price with positive quantity
    defined_price = Price.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    negated_price = defined_price.negative()
    assert negated_price.qty_or_zero() == Decimal('-100')
    assert negated_price.ccy_or_none().code == 'USD'
    assert negated_price.dov_or_none() == Date(2019, 1, 1)
    
    # Test negative on a defined price with negative quantity
    negative_price = Price.of(Currencies["EUR"], Decimal('-50'), Date(2020, 6, 15))
    double_negated = negative_price.negative()
    assert double_negated.qty_or_zero() == Decimal('50')
    assert double_negated.ccy_or_none().code == 'EUR'
    
    # Test negative on a defined price with zero quantity
    zero_price = Price.of(Currencies["GBP"], Decimal('0'), Date(2021, 3, 10))
    negated_zero = zero_price.negative()
    assert negated_zero.qty_or_zero() == Decimal('0')
    
    # Test negative on undefined price returns itself
    undefined_price = Price.na()
    result = undefined_price.negative()
    assert result.undefined
    assert result is undefined_price


# LLM-generated content at query #49
#--------------------------

```python
def test_money_floordiv():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.money import Money
    
    # Test floor division with positive divisor
    money1 = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result1 = money1.__floordiv__(Decimal('3'))
    assert result1.qty == Decimal('3.00')
    
    # Test floor division with negative divisor
    money2 = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result2 = money2.__floordiv__(Decimal('-3'))
    assert result2.qty == Decimal('-4.00')
    
    # Test floor division with decimal divisor
    money3 = Money.of(Currencies["USD"], Decimal('7.5'), Date(2019, 1, 1))
    result3 = money3.__floordiv__(Decimal('2.5'))
    assert result3.qty == Decimal('3.00')
    
    # Test floor division of undefined money returns undefined
    money_na = Money.na()
    result_na = money_na.__floordiv__(Decimal('5'))
    assert result_na.undefined
    
    # Test floor division by zero returns undefined money
    money4 = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result4 = money4.__floordiv__(Decimal('0'))
    assert result4.undefined
    
    # Test floor division preserves currency
    money5 = Money.of(Currencies["EUR"], Decimal('20'), Date(2019, 1, 1))
    result5 = money5.__floordiv__(Decimal('6'))
    assert result5.ccy == Currencies["EUR"]
    assert result5.qty == Decimal('3.00')


# LLM-generated content at query #50
#--------------------------

```python
def test_lte_raises_incompatible_currency_error_when_currencies_differ():
    from decimal import Decimal
    from datetime import date
    
    ccy1 = Currency(code="USD", quantizer=Decimal("0.01"))
    ccy2 = Currency(code="EUR", quantizer=Decimal("0.01"))
    dov = date(2023, 1, 1)
    
    price1 = SomePrice(ccy1, Decimal("100.00"), dov)
    price2 = SomePrice(ccy2, Decimal("100.00"), dov)
    
    try:
        price1.lte(price2)
        assert False, "Expected IncompatibleCurrencyError to be raised"
    except IncompatibleCurrencyError as e:
        assert e.ccy1 == ccy1
        assert e.ccy2 == ccy2
        assert "<= comparison" in str(e.operation)


# LLM-generated content at query #51
#--------------------------

```python
def test_someprice_lt():
    from datetime import date
    from decimal import Decimal
    
    # Create mock Currency objects
    class MockCurrency:
        def __init__(self, code):
            self.code = code
        
        def __eq__(self, other):
            return self.code == other.code
        
        def __ne__(self, other):
            return not self.__eq__(other)
    
    usd = MockCurrency("USD")
    eur = MockCurrency("EUR")
    
    # Test case 1: less than with same currency - true case
    price1 = SomePrice(usd, Decimal("100"), date(2024, 1, 1))
    price2 = SomePrice(usd, Decimal("200"), date(2024, 1, 1))
    assert price1.lt(price2) is True
    
    # Test case 2: less than with same currency - false case
    price3 = SomePrice(usd, Decimal("300"), date(2024, 1, 1))
    price4 = SomePrice(usd, Decimal("200"), date(2024, 1, 1))
    assert price3.lt(price4) is False
    
    # Test case 3: less than with same currency - equal case
    price5 = SomePrice(usd, Decimal("200"), date(2024, 1, 1))
    price6 = SomePrice(usd, Decimal("200"), date(2024, 1, 1))
    assert price5.lt(price6) is False
    
    # Test case 4: less than with non-SomePrice object
    price7 = SomePrice(usd, Decimal("100"), date(2024, 1, 1))
    assert price7.lt("not a price") is False
    
    # Test case 5: less than with different currencies raises exception
    price8 = SomePrice(usd, Decimal("100"), date(2024, 1, 1))
    price9 = SomePrice(eur, Decimal("100"), date(2024, 1, 1))
    try:
        price8.lt(price9)
        assert False, "Expected IncompatibleCurrencyError"
    except Exception as e:
        assert "IncompatibleCurrencyError" in str(type(e).__name__)


# LLM-generated content at query #52
#--------------------------

```python
def test_price_bool():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.price import Price

    # Test defined price with non-zero quantity returns True
    defined_price_nonzero = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert bool(defined_price_nonzero) is True

    # Test defined price with zero quantity returns False
    defined_price_zero = Price.of(Currencies["USD"], Decimal('0'), Date(2019, 1, 1))
    assert bool(defined_price_zero) is False

    # Test undefined price returns False
    undefined_price = Price.na()
    assert bool(undefined_price) is False

    # Test defined price with positive quantity returns True
    positive_price = Price.of(Currencies["EUR"], Decimal('42.5'), Date(2020, 6, 15))
    assert bool(positive_price) is True

    # Test defined price with negative quantity returns True
    negative_price = Price.of(Currencies["GBP"], Decimal('-10.25'), Date(2021, 3, 20))
    assert bool(negative_price) is True


# LLM-generated content at query #53
#--------------------------

```python
def test_truediv_valid_division():
    from decimal import Decimal
    from datetime import date
    
    ccy = type('Currency', (), {'quantizer': Decimal('0.01')})()
    price = SomePrice(ccy, Decimal('100'), date(2024, 1, 1))
    result = price / Decimal('2')
    
    assert result.qty == Decimal('50')
    assert result.ccy == ccy
    assert result.dov == date(2024, 1, 1)


def test_truediv_division_by_zero():
    from decimal import Decimal
    from datetime import date
    
    ccy = type('Currency', (), {'quantizer': Decimal('0.01')})()
    price = SomePrice(ccy, Decimal('100'), date(2024, 1, 1))
    result = price / Decimal('0')
    
    assert result is NoPrice


def test_truediv_with_integer():
    from decimal import Decimal
    from datetime import date
    
    ccy = type('Currency', (), {'quantizer': Decimal('0.01')})()
    price = SomePrice(ccy, Decimal('100'), date(2024, 1, 1))
    result = price / 4
    
    assert result.qty == Decimal('25')
    assert result.ccy == ccy
    assert result.dov == date(2024, 1, 1)


def test_truediv_with_float():
    from decimal import Decimal
    from datetime import date
    
    ccy = type('Currency', (), {'quantizer': Decimal('0.01')})()
    price = SomePrice(ccy, Decimal('100'), date(2024, 1, 1))
    result = price / 2.5
    
    assert result.qty == Decimal('100') / Decimal('2.5')
    assert result.ccy == ccy
    assert result.dov == date(2024, 1, 1)


def test_truediv_preserves_date():
    from decimal import Decimal
    from datetime import date
    
    ccy = type('Currency', (), {'quantizer': Decimal('0.01')})()
    dov = date(2023, 6, 15)
    price = SomePrice(ccy, Decimal('150'), dov)
    result = price / Decimal('3')
    
    assert result.dov == dov
    assert result.qty == Decimal('50')


# LLM-generated content at query #54
#--------------------------

```python
def test_somemoney_constructor():
    from decimal import Decimal
    from datetime import date
    
    # Create a Currency object (assuming it exists)
    class Currency:
        def __init__(self, code, decimals=2):
            self.code = code
            self.decimals = decimals
            self.quantizer = Decimal(10) ** -decimals
        
        def __eq__(self, other):
            return self.code == other.code
        
        def __hash__(self):
            return hash(self.code)
    
    ccy = Currency("USD", 2)
    qty = Decimal("100.50")
    dov = date(2024, 1, 15)
    
    money = SomeMoney(ccy, qty, dov)
    
    assert money.ccy == ccy
    assert money.qty == qty
    assert money.dov == dov
    assert money[0] == ccy
    assert money[1] == qty
    assert money[2] == dov
    assert money.defined is True
    assert money.undefined is False


def test_somemoney_constructor_with_different_currencies():
    from decimal import Decimal
    from datetime import date
    
    class Currency:
        def __init__(self, code, decimals=2):
            self.code = code
            self.decimals = decimals
            self.quantizer = Decimal(10) ** -decimals
        
        def __eq__(self, other):
            return self.code == other.code
        
        def __hash__(self):
            return hash(self.code)
    
    ccy_eur = Currency("EUR", 2)
    qty = Decimal("50.25")
    dov = date(2024, 6, 30)
    
    money = SomeMoney(ccy_eur, qty, dov)
    
    assert money.ccy == ccy_eur
    assert money.qty == qty
    assert money.dov == dov


def test_somemoney_constructor_with_zero_quantity():
    from decimal import Decimal
    from datetime import date
    
    class Currency:
        def __init__(self, code, decimals=2):
            self.code = code
            self.decimals = decimals
            self.quantizer = Decimal(10) ** -decimals
        
        def __eq__(self, other):
            return self.code == other.code
        
        def __hash__(self):
            return hash(self.code)
    
    ccy = Currency("GBP", 2)
    qty = Decimal("0.00")
    dov = date(2024, 1, 1)
    
    money = SomeMoney(ccy, qty, dov)
    
    assert money.qty == Decimal("0.00")
    assert money.ccy == ccy
    assert money.dov == dov


def test_somemoney_constructor_with_negative_quantity():
    from decimal import Decimal
    from datetime import date
    
    class Currency:
        def __init__(self, code, decimals=2):
            self.code = code
            self.decimals = decimals
            self.quantizer = Decimal(10) ** -decimals
        
        def __eq__(self, other):
            return self.code == other.code
        
        def __hash__(self):
            return hash(self.code)
    
    ccy = Currency("JPY", 0)
    qty = Decimal("-1000")
    dov = date(2024, 3, 15)
    
    money = SomeMoney(ccy, qty, dov)
    
    assert money.qty == Decimal("-1000")
    assert money.ccy == ccy
    assert money.dov == dov


# LLM-generated content at query #55
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
    assert not (money1 == "not a money object")
    assert not (money1 == None)


# LLM-generated content at query #56
#--------------------------

```python
def test_convert_with_valid_rate():
    from datetime import date
    from decimal import Decimal
    from pypara.currencies import Currency, CurrencyType
    from pypara.monetary import SomeMoney
    from pypara.fx import FXRate, FXRateService
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    
    money = SomeMoney(usd, Decimal("100.00"), date(2023, 1, 1))
    
    class MockFXRateService:
        default = None
        
        @staticmethod
        def query(from_ccy, to_ccy, asof, strict):
            if from_ccy == usd and to_ccy == eur:
                return FXRate(usd, eur, Decimal("0.92"), date(2023, 1, 1))
            return None
    
    original_service = FXRateService.default
    FXRateService.default = MockFXRateService()
    
    try:
        result = money.convert(eur, asof=date(2023, 1, 1), strict=False)
        assert result.ccy == eur
        assert result.qty == Decimal("92.00")
        assert result.dov == date(2023, 1, 1)
    finally:
        FXRateService.default = original_service


def test_convert_with_undefined_rate_non_strict():
    from datetime import date
    from decimal import Decimal
    from pypara.currencies import Currency, CurrencyType
    from pypara.monetary import SomeMoney, NoMoney
    from pypara.fx import FXRateService
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    gbp = Currency.of("GBP", "British Pounds", 2, CurrencyType.MONEY)
    
    money = SomeMoney(usd, Decimal("100.00"), date(2023, 1, 1))
    
    class MockFXRateService:
        default = None
        
        @staticmethod
        def query(from_ccy, to_ccy, asof, strict):
            return None
    
    original_service = FXRateService.default
    FXRateService.default = MockFXRateService()
    
    try:
        result = money.convert(gbp, asof=date(2023, 1, 1), strict=False)
        assert result == NoMoney
    finally:
        FXRateService.default = original_service


def test_convert_with_undefined_rate_strict():
    from datetime import date
    from decimal import Decimal
    from pypara.currencies import Currency, CurrencyType
    from pypara.monetary import SomeMoney
    from pypara.fx import FXRateService, FXRateLookupError
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    gbp = Currency.of("GBP", "British Pounds", 2, CurrencyType.MONEY)
    
    money = SomeMoney(usd, Decimal("100.00"), date(2023, 1, 1))
    
    class MockFXRateService:
        default = None
        
        @staticmethod
        def query(from_ccy, to_ccy, asof, strict):
            return None
    
    original_service = FXRateService.default
    FXRateService.default = MockFXRateService()
    
    try:
        error_raised = False
        try:
            money.convert(gbp, asof=date(2023, 1, 1), strict=True)
        except FXRateLookupError:
            error_raised = True
        assert error_raised
    finally:
        FXRateService.default = original_service


def test_convert_without_explicit_asof_uses_dov():
    from datetime import date
    from decimal import Decimal
    from pypara.currencies import Currency, CurrencyType
    from pypara.monetary import SomeMoney
    from pypara.fx import FXRate, FXRateService
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    
    dov = date(2023, 6, 15)
    money = SomeMoney(usd, Decimal("100.00"), dov)
    
    class MockFXRateService:
        default = None
        
        @staticmethod
        def query(from_ccy, to_ccy, asof, strict):
            if from_ccy == usd and to_ccy == jpy and asof == dov:
                return FXRate(usd, jpy, Decimal("130.50"), dov)
            return None
    
    original_service = FXRateService.default
    FXRateService.default = MockFXRateService()
    
    try:
        result = money.convert(jpy, strict=False)
        assert result.ccy == jpy
        assert result.qty == Decimal("13050")
        assert result.dov == dov
    finally:
        FXRateService.default = original_service


def test_convert_no_default_fx_rate_service():
    from datetime import date
    from decimal import Decimal
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
        error_raised = False
        try:
            money.convert(eur)
        except ProgrammingError as e:
            error_raised = True
            assert "Did you implement and set the default FX rate service?" in str(e)
        assert error_raised
    finally:
        FXRateService.default = original_service


# LLM-generated content at query #57
#--------------------------

```python
def test_lt_defined_money_with_smaller_quantity():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    from pypara.money import Money
    
    money1 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    
    result = money1.lt(money2)
    assert result is True


def test_lt_defined_money_with_larger_quantity():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    from pypara.money import Money
    
    money1 = Money.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    
    result = money1.lt(money2)
    assert result is False


def test_lt_defined_money_with_equal_quantity():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    from pypara.money import Money
    
    money1 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    
    result = money1.lt(money2)
    assert result is False


def test_lt_undefined_money_with_defined_money():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    from pypara.money import Money
    
    money1 = Money.na()
    money2 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    
    result = money1.lt(money2)
    assert result is True


def test_lt_defined_money_with_undefined_money():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    from pypara.money import Money
    
    money1 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    money2 = Money.na()
    
    result = money1.lt(money2)
    assert result is False


def test_lt_undefined_money_with_undefined_money():
    from pypara.money import Money
    
    money1 = Money.na()
    money2 = Money.na()
    
    result = money1.lt(money2)
    assert result is False


def test_lt_different_currencies_raises_error():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    from pypara.money import Money
    
    money1 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["EUR"], Decimal('1'), Date(2019, 1, 1))
    
    try:
        money1.lt(money2)
        assert False, "Expected IncompatibleCurrencyError"
    except Exception as e:
        assert "IncompatibleCurrencyError" in str(type(e).__name__)


# LLM-generated content at query #58
#--------------------------

```python
def test_someprice_add():
    from decimal import Decimal
    from datetime import date
    
    # Create test currency and price objects
    usd = Currency(code="USD", quantizer=Decimal("0.01"))
    eur = Currency(code="EUR", quantizer=Decimal("0.01"))
    
    price1 = SomePrice(usd, Decimal("100.00"), date(2024, 1, 1))
    price2 = SomePrice(usd, Decimal("50.00"), date(2024, 1, 15))
    price3 = SomePrice(eur, Decimal("75.00"), date(2024, 1, 10))
    
    # Test adding two prices with same currency
    result = price1.__add__(price2)
    assert result.ccy == usd
    assert result.qty == Decimal("150.00")
    assert result.dov == date(2024, 1, 15)
    
    # Test adding with later date takes precedence
    result2 = price2.__add__(price1)
    assert result2.ccy == usd
    assert result2.qty == Decimal("150.00")
    assert result2.dov == date(2024, 1, 15)
    
    # Test adding with undefined price returns self
    no_price = NoPrice
    result3 = price1.__add__(no_price)
    assert result3 == price1
    
    # Test adding prices with different currencies raises error
    try:
        price1.__add__(price3)
        assert False, "Should have raised IncompatibleCurrencyError"
    except IncompatibleCurrencyError as e:
        assert e.ccy1 == usd
        assert e.ccy2 == eur
        assert e.operation == "addition"


# LLM-generated content at query #59
#--------------------------

```python
def test_price_abs():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    from pypara.price import Price
    
    # Test abs on defined price with positive quantity
    positive_price = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result_positive = positive_price.abs()
    assert result_positive.qty_or_zero() == Decimal('10')
    assert result_positive.ccy_or_none().code == 'USD'
    
    # Test abs on defined price with negative quantity
    negative_price = Price.of(Currencies["USD"], Decimal('-10'), Date(2019, 1, 1))
    result_negative = negative_price.abs()
    assert result_negative.qty_or_zero() == Decimal('10')
    assert result_negative.ccy_or_none().code == 'USD'
    
    # Test abs on undefined price returns itself
    undefined_price = Price.na()
    result_undefined = undefined_price.abs()
    assert result_undefined.undefined
    
    # Test abs on zero quantity
    zero_price = Price.of(Currencies["EUR"], Decimal('0'), Date(2019, 1, 1))
    result_zero = zero_price.abs()
    assert result_zero.qty_or_zero() == Decimal('0')
    assert result_zero.ccy_or_none().code == 'EUR'


# LLM-generated content at query #60
#--------------------------

```python
def test_price_ge():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    from pypara.price import Price
    
    # Create defined prices
    price_usd_100 = Price.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    price_usd_50 = Price.of(Currencies["USD"], Decimal('50'), Date(2019, 1, 1))
    price_usd_100_same = Price.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 2))
    
    # Create undefined price
    undefined_price = Price.na()
    
    # Test: defined price >= smaller defined price should be True
    assert price_usd_100.gte(price_usd_50)
    
    # Test: defined price >= equal defined price should be True
    assert price_usd_100.gte(price_usd_100_same)
    
    # Test: smaller defined price >= larger defined price should be False
    assert not price_usd_50.gte(price_usd_100)
    
    # Test: defined price >= undefined price should be True
    assert price_usd_100.gte(undefined_price)
    
    # Test: undefined price >= undefined price should be True
    assert undefined_price.gte(undefined_price)
    
    # Test: undefined price >= defined price should be False
    assert not undefined_price.gte(price_usd_100)


# LLM-generated content at query #61
#--------------------------

```python
def test_qty_or_else_with_defined_price():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    someprice = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    result = someprice.qty_or_else(lambda: Decimal('42'))
    assert result == Decimal('1')


def test_qty_or_else_with_defined_price_returns_qty_not_combinator():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    someprice = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    result = someprice.qty_or_else(lambda: True)
    assert result == Decimal('1')


def test_qty_or_else_with_undefined_price_returns_combinator_result():
    from datetime import date as Date
    from decimal import Decimal
    
    noneprice = Price.of(None, Decimal('1'), None)
    result = noneprice.qty_or_else(lambda: Decimal('42'))
    assert result == Decimal('42')


def test_qty_or_else_with_undefined_price_returns_non_decimal_combinator():
    from datetime import date as Date
    from decimal import Decimal
    
    noneprice = Price.of(None, Decimal('1'), None)
    result = noneprice.qty_or_else(lambda: False)
    assert result is False


def test_qty_or_else_with_undefined_price_calls_combinator():
    from datetime import date as Date
    from decimal import Decimal
    
    noneprice = Price.of(None, Decimal('1'), None)
    combinator_called = []
    
    def combinator():
        combinator_called.append(True)
        return Decimal('100')
    
    result = noneprice.qty_or_else(combinator)
    assert len(combinator_called) == 1
    assert result == Decimal('100')


# LLM-generated content at query #62
#--------------------------

```python
def test_ccy_or_with_defined_price():
    from pypara.currencies import Currencies
    from pypara.price import Price
    from decimal import Decimal
    from datetime import date as Date
    
    someprice = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    result = someprice.ccy_or(Currencies["EUR"])
    assert result.code == 'USD'


def test_ccy_or_with_undefined_price():
    from pypara.currencies import Currencies
    from pypara.price import Price
    from decimal import Decimal
    from datetime import date as Date
    
    noneprice = Price.of(None, Decimal('1'), None)
    result = noneprice.ccy_or(Currencies["EUR"])
    assert result.code == 'EUR'


def test_ccy_or_returns_default_when_price_undefined():
    from pypara.currencies import Currencies
    from pypara.price import Price
    from decimal import Decimal
    
    noneprice = Price.na()
    default_currency = Currencies["GBP"]
    result = noneprice.ccy_or(default_currency)
    assert result is default_currency


# LLM-generated content at query #63
#--------------------------

```python
def test_somemoney_gt():
    from decimal import Decimal
    from datetime import date
    
    # Create test currencies and money objects
    usd = Currency(code="USD", decimals=2)
    eur = Currency(code="EUR", decimals=2)
    
    money1 = SomeMoney(usd, Decimal("100.00"), date(2024, 1, 1))
    money2 = SomeMoney(usd, Decimal("50.00"), date(2024, 1, 1))
    money3 = SomeMoney(usd, Decimal("100.00"), date(2024, 1, 1))
    money_eur = SomeMoney(eur, Decimal("100.00"), date(2024, 1, 1))
    
    # Test: greater than with same currency
    assert money1 > money2 is True
    assert money2 > money1 is False
    assert money1 > money3 is False
    
    # Test: greater than with non-SomeMoney object returns True
    assert money1 > "not_money" is True
    assert money1 > 100 is True
    assert money1 > None is True
    
    # Test: greater than with different currency raises exception
    try:
        money1 > money_eur
        assert False, "Should raise IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass


# LLM-generated content at query #64
#--------------------------

```python
def test_as_boolean():
    from decimal import Decimal
    from datetime import date as Date
    from pypara.currencies import Currencies
    from pypara.price import Price
    
    # Test defined price with non-zero quantity returns True
    defined_nonzero_price = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert defined_nonzero_price.as_boolean() is True
    
    # Test defined price with positive quantity returns True
    defined_positive_price = Price.of(Currencies["EUR"], Decimal('100.50'), Date(2020, 6, 15))
    assert defined_positive_price.as_boolean() is True
    
    # Test defined price with negative quantity returns True
    defined_negative_price = Price.of(Currencies["GBP"], Decimal('-50.25'), Date(2021, 3, 10))
    assert defined_negative_price.as_boolean() is True
    
    # Test defined price with zero quantity returns False
    defined_zero_price = Price.of(Currencies["JPY"], Decimal('0'), Date(2022, 12, 25))
    assert defined_zero_price.as_boolean() is False
    
    # Test undefined price returns False
    undefined_price = Price.na()
    assert undefined_price.as_boolean() is False


# LLM-generated content at query #65
#--------------------------

```python
def test_floordiv_with_valid_numeric():
    from decimal import Decimal
    from datetime import date
    
    ccy = Currency(code="USD", quantizer=Decimal("0.01"))
    price = SomePrice(ccy=ccy, qty=Decimal("10.5"), dov=date(2023, 1, 1))
    result = price // 3
    
    assert isinstance(result, SomePrice)
    assert result.ccy == ccy
    assert result.qty == Decimal("3")
    assert result.dov == date(2023, 1, 1)


def test_floordiv_with_decimal():
    from decimal import Decimal
    from datetime import date
    
    ccy = Currency(code="EUR", quantizer=Decimal("0.01"))
    price = SomePrice(ccy=ccy, qty=Decimal("20"), dov=date(2023, 6, 15))
    result = price // Decimal("7")
    
    assert isinstance(result, SomePrice)
    assert result.qty == Decimal("2")
    assert result.ccy == ccy


def test_floordiv_with_zero_returns_no_price():
    from decimal import Decimal
    from datetime import date
    
    ccy = Currency(code="GBP", quantizer=Decimal("0.01"))
    price = SomePrice(ccy=ccy, qty=Decimal("100"), dov=date(2023, 3, 20))
    result = price // 0
    
    assert result is NoPrice


def test_floordiv_with_negative_divisor():
    from decimal import Decimal
    from datetime import date
    
    ccy = Currency(code="JPY", quantizer=Decimal("1"))
    price = SomePrice(ccy=ccy, qty=Decimal("50"), dov=date(2023, 12, 1))
    result = price // -2
    
    assert isinstance(result, SomePrice)
    assert result.qty == Decimal("-25")
    assert result.dov == date(2023, 12, 1)


def test_floordiv_with_float():
    from decimal import Decimal
    from datetime import date
    
    ccy = Currency(code="USD", quantizer=Decimal("0.01"))
    price = SomePrice(ccy=ccy, qty=Decimal("15.7"), dov=date(2023, 5, 10))
    result = price // 2.5
    
    assert isinstance(result, SomePrice)
    assert result.qty == Decimal("6")
    assert result.ccy == ccy


# LLM-generated content at query #66
#--------------------------

```python
def test_positive_returns_same_price_when_defined():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    price = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    result = price.positive()
    
    assert result.ccy.code == 'USD'
    assert result.qty == Decimal('1')
    assert result.dov == Date(2019, 1, 1)


def test_positive_returns_itself_when_undefined():
    price = Price.na()
    result = price.positive()
    
    assert result.undefined
    assert result is price


def test_positive_returns_same_price_with_negative_quantity():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    price = Price.of(Currencies["EUR"], Decimal('-5'), Date(2020, 6, 15))
    result = price.positive()
    
    assert result.ccy.code == 'EUR'
    assert result.qty == Decimal('-5')
    assert result.dov == Date(2020, 6, 15)


def test_positive_returns_same_price_with_zero_quantity():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    price = Price.of(Currencies["GBP"], Decimal('0'), Date(2021, 3, 20))
    result = price.positive()
    
    assert result.ccy.code == 'GBP'
    assert result.qty == Decimal('0')
    assert result.dov == Date(2021, 3, 20)


# LLM-generated content at query #67
#--------------------------

```python
def test_with_qty():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.money import Money
    
    # Test with_qty on defined money
    somemoney = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    new_money = somemoney.with_qty(Decimal('5'))
    assert new_money.qty == Decimal('5.00')
    assert new_money.ccy == Currencies["USD"]
    assert new_money.dov == Date(2019, 1, 1)
    assert new_money.defined
    
    # Test with_qty on undefined money
    nonemoney = Money.na()
    result = nonemoney.with_qty(Decimal('10'))
    assert result is nonemoney
    assert result.undefined
    
    # Test with_qty with zero quantity
    zero_money = somemoney.with_qty(Decimal('0'))
    assert zero_money.qty == Decimal('0.00')
    assert zero_money.defined
    
    # Test with_qty with negative quantity
    negative_money = somemoney.with_qty(Decimal('-3.50'))
    assert negative_money.qty == Decimal('-3.50')
    assert negative_money.defined


# LLM-generated content at query #68
#--------------------------

```python
def test_dimap_with_defined_money():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.money import Money
    
    somemoney = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    result = somemoney.dimap(lambda x: x.ccy.code, lambda: "EUR")
    assert result == "USD"


def test_dimap_with_undefined_money():
    from pypara.money import Money
    
    nonemoney = Money.na()
    result = nonemoney.dimap(lambda x: x.ccy.code, lambda: "EUR")
    assert result == "EUR"


def test_dimap_with_defined_money_numeric_result():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.money import Money
    
    somemoney = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    result = somemoney.dimap(lambda x: x.qty * Decimal('2'), lambda: Decimal('0'))
    assert result == Decimal('200')


def test_dimap_with_undefined_money_numeric_fallback():
    from decimal import Decimal
    from pypara.money import Money
    
    nonemoney = Money.na()
    result = nonemoney.dimap(lambda x: x.qty, lambda: Decimal('42'))
    assert result == Decimal('42')


def test_dimap_applies_function_to_defined_money():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.money import Money
    
    somemoney = Money.of(Currencies["EUR"], Decimal('50'), Date(2020, 6, 15))
    result = somemoney.dimap(lambda x: (x.ccy.code, x.qty, x.dov), lambda: None)
    assert result == ("EUR", Decimal('50'), Date(2020, 6, 15))


def test_dimap_calls_combinator_on_undefined_money():
    from pypara.money import Money
    
    nonemoney = Money.na()
    combinator_called = []
    
    def combinator():
        combinator_called.append(True)
        return "fallback_value"
    
    result = nonemoney.dimap(lambda x: "should_not_be_called", combinator)
    assert result == "fallback_value"
    assert len(combinator_called) == 1


# LLM-generated content at query #69
#--------------------------

```python
def test_money_eq():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money1 = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    money3 = Money.of(Currencies["USD"], Decimal('200'), Date(2019, 1, 1))
    money4 = Money.of(Currencies["EUR"], Decimal('100'), Date(2019, 1, 1))
    money5 = Money.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 2))
    nomoney1 = Money.na()
    nomoney2 = Money.na()
    
    assert money1 == money2
    assert not (money1 == money3)
    assert not (money1 == money4)
    assert not (money1 == money5)
    assert money1 != money3
    assert money1 != money4
    assert money1 != money5
    assert nomoney1 == nomoney2
    assert not (money1 == nomoney1)
    assert not (nomoney1 == money1)
    assert not (money1 == "not a money")
    assert not (money1 == 100)
    assert not (money1 == None)


# LLM-generated content at query #70
#--------------------------

```python
def test_subtract():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    # Test subtracting two defined money objects with same currency
    money1 = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('3'), Date(2019, 1, 1))
    result = money1.subtract(money2)
    assert result.defined
    assert result.qty == Decimal('7.00')
    assert result.ccy.code == "USD"
    
    # Test subtracting with negative result
    money3 = Money.of(Currencies["USD"], Decimal('3'), Date(2019, 1, 1))
    money4 = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result2 = money3.subtract(money4)
    assert result2.defined
    assert result2.qty == Decimal('-7.00')
    
    # Test subtracting undefined money from defined money returns defined money
    money5 = Money.of(Currencies["USD"], Decimal('5'), Date(2019, 1, 1))
    money_undefined = Money.na()
    result3 = money5.subtract(money_undefined)
    assert result3 is money5
    
    # Test subtracting defined money from undefined money returns defined money
    result4 = money_undefined.subtract(money5)
    assert result4 is money5
    
    # Test subtracting two undefined money objects
    result5 = money_undefined.subtract(money_undefined)
    assert result5.undefined
    
    # Test subtracting money with different currencies raises error
    money_usd = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    money_eur = Money.of(Currencies["EUR"], Decimal('5'), Date(2019, 1, 1))
    try:
        money_usd.subtract(money_eur)
        assert False, "Expected IncompatibleCurrencyError"
    except Exception as e:
        assert "IncompatibleCurrencyError" in str(type(e).__name__)


# LLM-generated content at query #71
#--------------------------

```python
def test_subtract_defined_money_same_currency():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    
    money1 = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('3'), Date(2019, 1, 2))
    result = money1.subtract(money2)
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('7.00')
    assert result.ccy_or_none().code == "USD"


def test_subtract_defined_money_different_currency():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    
    money1 = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["EUR"], Decimal('3'), Date(2019, 1, 2))
    
    try:
        money1.subtract(money2)
        assert False, "Should raise IncompatibleCurrencyError"
    except Exception as e:
        assert "IncompatibleCurrencyError" in str(type(e))


def test_subtract_undefined_left_operand():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    
    money1 = Money.na()
    money2 = Money.of(Currencies["USD"], Decimal('3'), Date(2019, 1, 2))
    result = money1.subtract(money2)
    
    assert result is money2


def test_subtract_undefined_right_operand():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    
    money1 = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    money2 = Money.na()
    result = money1.subtract(money2)
    
    assert result is money1


def test_subtract_both_undefined():
    money1 = Money.na()
    money2 = Money.na()
    result = money1.subtract(money2)
    
    assert result.undefined


def test_subtract_negative_result():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    
    money1 = Money.of(Currencies["USD"], Decimal('3'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 2))
    result = money1.subtract(money2)
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('-7.00')


def test_subtract_zero_result():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    
    money1 = Money.of(Currencies["USD"], Decimal('5'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('5'), Date(2019, 1, 2))
    result = money1.subtract(money2)
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('0.00')


def test_subtract_carries_forward_date():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    
    money1 = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('3'), Date(2019, 1, 15))
    result = money1.subtract(money2)
    
    assert result.dov_or_none() == Date(2019, 1, 1)


# LLM-generated content at query #72
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
    
    # Test with undefined price - should return None
    noneprice = Price.of(None, None, Date(2019, 1, 1))
    result = noneprice.dov_or_none()
    assert result is None
    
    # Test with completely undefined price - should return None
    completely_undefined = Price.na()
    result = completely_undefined.dov_or_none()
    assert result is None


# LLM-generated content at query #73
#--------------------------

```python
def test_price_neg():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    from pypara.price import Price
    
    # Test negation of defined price
    price = Price.of(Currencies["USD"], Decimal('100'), Date(2019, 1, 1))
    negated_price = -price
    assert negated_price.qty_or_zero() == Decimal('-100')
    assert negated_price.ccy_or_none().code == "USD"
    assert negated_price.dov_or_none() == Date(2019, 1, 1)
    
    # Test negation of undefined price
    undefined_price = Price.na()
    negated_undefined = -undefined_price
    assert negated_undefined.undefined
    
    # Test double negation
    double_negated = -(-price)
    assert double_negated.qty_or_zero() == Decimal('100')
    
    # Test negation of zero price
    zero_price = Price.of(Currencies["EUR"], Decimal('0'), Date(2020, 6, 15))
    negated_zero = -zero_price
    assert negated_zero.qty_or_zero() == Decimal('0')


# LLM-generated content at query #74
#--------------------------

```python
def test_with_dov():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    
    # Test with_dov on defined money object
    somemoney = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    new_date = Date(2020, 6, 15)
    result = somemoney.with_dov(new_date)
    assert result.dov_or_none() == new_date
    assert result.ccy_or_none().code == "USD"
    assert result.qty_or_none() == Decimal('1.00')
    
    # Test with_dov on undefined money object
    nonemoney = Money.of(None, Decimal('1'), None)
    result_undefined = nonemoney.with_dov(new_date)
    assert result_undefined is nonemoney
    assert result_undefined.undefined is True


# LLM-generated content at query #75
#--------------------------

```python
def test_truediv_divides_quantity_by_numeric():
    from decimal import Decimal
    from datetime import date
    
    usd = Currency(code="USD", decimals=2)
    money = SomeMoney(usd, Decimal("100.00"), date(2024, 1, 1))
    
    result = money / 2
    
    assert result.ccy == usd
    assert result.qty == Decimal("50.00")
    assert result.dov == date(2024, 1, 1)


def test_truediv_quantizes_result():
    from decimal import Decimal
    from datetime import date
    
    usd = Currency(code="USD", decimals=2)
    money = SomeMoney(usd, Decimal("100.00"), date(2024, 1, 1))
    
    result = money / Decimal("3")
    
    assert result.qty == Decimal("33.33")


def test_truediv_by_zero_returns_no_money():
    from decimal import Decimal
    from datetime import date
    
    usd = Currency(code="USD", decimals=2)
    money = SomeMoney(usd, Decimal("100.00"), date(2024, 1, 1))
    
    result = money / 0
    
    assert result is NoMoney


def test_truediv_with_float():
    from decimal import Decimal
    from datetime import date
    
    usd = Currency(code="USD", decimals=2)
    money = SomeMoney(usd, Decimal("100.00"), date(2024, 1, 1))
    
    result = money / 2.5
    
    assert result.qty == Decimal("40.00")
    assert result.ccy == usd


def test_truediv_with_string_numeric():
    from decimal import Decimal
    from datetime import date
    
    usd = Currency(code="USD", decimals=2)
    money = SomeMoney(usd, Decimal("100.00"), date(2024, 1, 1))
    
    result = money / "4"
    
    assert result.qty == Decimal("25.00")
    assert result.ccy == usd


# LLM-generated content at query #76
#--------------------------

```python
def test_price_int():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    # Test __int__ on defined price with integer quantity
    price_int = Price.of(Currencies["USD"], Decimal('42'), Date(2019, 1, 1))
    assert int(price_int) == 42
    
    # Test __int__ on defined price with decimal quantity (should truncate)
    price_decimal = Price.of(Currencies["USD"], Decimal('42.75'), Date(2019, 1, 1))
    assert int(price_decimal) == 42
    
    # Test __int__ on defined price with negative quantity
    price_negative = Price.of(Currencies["USD"], Decimal('-42.5'), Date(2019, 1, 1))
    assert int(price_negative) == -42
    
    # Test __int__ on undefined price (should raise MonetaryOperationException)
    price_undefined = Price.na()
    try:
        int(price_undefined)
        assert False, "Expected MonetaryOperationException"
    except Exception as e:
        assert "MonetaryOperationException" in str(type(e).__name__)


# LLM-generated content at query #77
#--------------------------

```python
def test_floor_divide_defined_price():
    from decimal import Decimal
    from datetime import date as Date
    from pypara.currencies import Currencies
    from pypara.price import Price
    
    price = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = price.floor_divide(Decimal('3'))
    
    assert result.defined is True
    assert result.qty_or_zero() == Decimal('3')
    assert result.ccy_or_none().code == 'USD'


def test_floor_divide_undefined_price():
    from decimal import Decimal
    from pypara.price import Price
    
    price = Price.na()
    result = price.floor_divide(Decimal('3'))
    
    assert result.undefined is True


def test_floor_divide_by_zero():
    from decimal import Decimal
    from datetime import date as Date
    from pypara.currencies import Currencies
    from pypara.price import Price
    
    price = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = price.floor_divide(Decimal('0'))
    
    assert result.undefined is True


def test_floor_divide_with_negative_divisor():
    from decimal import Decimal
    from datetime import date as Date
    from pypara.currencies import Currencies
    from pypara.price import Price
    
    price = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = price.floor_divide(Decimal('-3'))
    
    assert result.defined is True
    assert result.qty_or_zero() == Decimal('-4')


def test_floor_divide_with_fractional_divisor():
    from decimal import Decimal
    from datetime import date as Date
    from pypara.currencies import Currencies
    from pypara.price import Price
    
    price = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = price.floor_divide(Decimal('2.5'))
    
    assert result.defined is True
    assert result.qty_or_zero() == Decimal('4')


def test_floor_divide_preserves_currency():
    from decimal import Decimal
    from datetime import date as Date
    from pypara.currencies import Currencies
    from pypara.price import Price
    
    price = Price.of(Currencies["EUR"], Decimal('20'), Date(2019, 6, 15))
    result = price.floor_divide(Decimal('4'))
    
    assert result.ccy_or_none().code == 'EUR'


# LLM-generated content at query #78
#--------------------------

```python
def test_gt_defined_money_greater_than_undefined():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    defined_money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    undefined_money = Money.na()
    
    result = defined_money.gt(undefined_money)
    assert result is True


def test_gt_undefined_money_not_greater_than_defined():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    undefined_money = Money.na()
    defined_money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    
    result = undefined_money.gt(defined_money)
    assert result is False


def test_gt_undefined_money_not_greater_than_undefined():
    undefined_money1 = Money.na()
    undefined_money2 = Money.na()
    
    result = undefined_money1.gt(undefined_money2)
    assert result is False


def test_gt_defined_money_greater_than_defined_same_currency():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money1 = Money.of(Currencies["USD"], Decimal('20'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    
    result = money1.gt(money2)
    assert result is True


def test_gt_defined_money_not_greater_than_defined_same_currency():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money1 = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('20'), Date(2019, 1, 1))
    
    result = money1.gt(money2)
    assert result is False


def test_gt_defined_money_equal_quantities_not_greater():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    money1 = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    
    result = money1.gt(money2)
    assert result is False


def test_gt_defined_money_different_currency_raises_error():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    from pypara.exceptions import IncompatibleCurrencyError
    
    money1 = Money.of(Currencies["USD"], Decimal('20'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["EUR"], Decimal('10'), Date(2019, 1, 1))
    
    try:
        money1.gt(money2)
        assert False, "Expected IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        assert True


# LLM-generated content at query #79
#--------------------------

```python
def test_price_add_with_same_currency():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.price import Price
    
    price1 = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('5'), Date(2019, 1, 1))
    result = price1.add(price2)
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('15')
    assert result.ccy_or_none().code == 'USD'


def test_price_add_with_different_currency_raises_error():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.price import Price
    
    price1 = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["EUR"], Decimal('5'), Date(2019, 1, 1))
    
    try:
        result = price1.add(price2)
        assert False, "Should raise IncompatibleCurrencyError"
    except Exception as e:
        assert "IncompatibleCurrencyError" in str(type(e))


def test_price_add_with_undefined_operand():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.price import Price
    
    price1 = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    undefined_price = Price.na()
    result = price1.add(undefined_price)
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('10')
    assert result is price1


def test_price_add_both_undefined():
    from pypara.price import Price
    
    undefined_price1 = Price.na()
    undefined_price2 = Price.na()
    result = undefined_price1.add(undefined_price2)
    
    assert result.undefined


def test_price_add_carries_forward_date():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.price import Price
    
    price1 = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('5'), Date(2019, 1, 2))
    result = price1.add(price2)
    
    assert result.defined
    assert result.qty_or_zero() == Decimal('15')
    assert result.dov_or_none() == Date(2019, 1, 1)


# LLM-generated content at query #80
#--------------------------

```python
def test_scalar_subtract():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    
    # Test scalar_subtract on defined price
    price = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = price.scalar_subtract(Decimal('3'))
    assert result.qty_or_zero() == Decimal('7')
    assert result.ccy_or_none().code == "USD"
    
    # Test scalar_subtract with zero
    price = Price.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = price.scalar_subtract(Decimal('0'))
    assert result.qty_or_zero() == Decimal('10')
    
    # Test scalar_subtract resulting in negative
    price = Price.of(Currencies["USD"], Decimal('5'), Date(2019, 1, 1))
    result = price.scalar_subtract(Decimal('10'))
    assert result.qty_or_zero() == Decimal('-5')
    
    # Test scalar_subtract on undefined price returns undefined
    undefined_price = Price.na()
    result = undefined_price.scalar_subtract(Decimal('5'))
    assert result.undefined
    
    # Test scalar_subtract with decimal places
    price = Price.of(Currencies["EUR"], Decimal('10.75'), Date(2019, 1, 1))
    result = price.scalar_subtract(Decimal('2.25'))
    assert result.qty_or_zero() == Decimal('8.50')
    assert result.ccy_or_none().code == "EUR"


# LLM-generated content at query #81
#--------------------------

```python
def test_dov_or():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    from pypara.price import Price
    
    # Test with defined price - should return the dov of the price
    someprice = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    result = someprice.dov_or(Date(2001, 1, 1))
    assert result == Date(2019, 1, 1)
    
    # Test with undefined price - should return the default date
    noneprice = Price.of(None, None, Date(2019, 1, 1))
    result = noneprice.dov_or(Date(2001, 1, 1))
    assert result == Date(2001, 1, 1)
    
    # Test with another defined price with different date
    another_price = Price.of(Currencies["EUR"], Decimal('100'), Date(2020, 6, 15))
    result = another_price.dov_or(Date(2000, 1, 1))
    assert result == Date(2020, 6, 15)
    
    # Test with undefined price and different default date
    undefined_price = Price.of(None, Decimal('50'), None)
    result = undefined_price.dov_or(Date(2015, 12, 31))
    assert result == Date(2015, 12, 31)


# LLM-generated content at query #82
#--------------------------

```python
def test_lt_with_same_currency_less_than():
    from decimal import Decimal
    from datetime import date
    
    currency = Currency(code="USD", decimals=2)
    money1 = SomeMoney(currency, Decimal("10.00"), date(2024, 1, 1))
    money2 = SomeMoney(currency, Decimal("20.00"), date(2024, 1, 1))
    
    assert money1 < money2 is True


def test_lt_with_same_currency_not_less_than():
    from decimal import Decimal
    from datetime import date
    
    currency = Currency(code="USD", decimals=2)
    money1 = SomeMoney(currency, Decimal("20.00"), date(2024, 1, 1))
    money2 = SomeMoney(currency, Decimal("10.00"), date(2024, 1, 1))
    
    assert money1 < money2 is False


def test_lt_with_same_currency_equal():
    from decimal import Decimal
    from datetime import date
    
    currency = Currency(code="USD", decimals=2)
    money1 = SomeMoney(currency, Decimal("10.00"), date(2024, 1, 1))
    money2 = SomeMoney(currency, Decimal("10.00"), date(2024, 1, 1))
    
    assert money1 < money2 is False


def test_lt_with_non_somemoney_object():
    from decimal import Decimal
    from datetime import date
    
    currency = Currency(code="USD", decimals=2)
    money = SomeMoney(currency, Decimal("10.00"), date(2024, 1, 1))
    
    assert money < "not money" is False


def test_lt_with_different_currencies_raises_error():
    from decimal import Decimal
    from datetime import date
    
    currency_usd = Currency(code="USD", decimals=2)
    currency_eur = Currency(code="EUR", decimals=2)
    money1 = SomeMoney(currency_usd, Decimal("10.00"), date(2024, 1, 1))
    money2 = SomeMoney(currency_eur, Decimal("20.00"), date(2024, 1, 1))
    
    try:
        result = money1 < money2
        assert False, "Expected IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass


# LLM-generated content at query #83
#--------------------------

```python
def test_money_lt():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    
    # Test: defined money less than defined money with same currency
    money1 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    assert money1.lt(money2) is True
    
    # Test: defined money not less than defined money with same currency
    money3 = Money.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    money4 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert money3.lt(money4) is False
    
    # Test: defined money not less than equal defined money with same currency
    money5 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    money6 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert money5.lt(money6) is False
    
    # Test: undefined money is less than defined money
    undefined_money = Money.na()
    defined_money = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert undefined_money.lt(defined_money) is True
    
    # Test: undefined money is not less than undefined money
    undefined_money1 = Money.na()
    undefined_money2 = Money.na()
    assert undefined_money1.lt(undefined_money2) is False
    
    # Test: defined money is not less than undefined money
    defined_money1 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    undefined_money3 = Money.na()
    assert defined_money1.lt(undefined_money3) is False


# LLM-generated content at query #84
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
    
    undefined_money = Money.na()
    rounded = undefined_money.round(2)
    assert rounded is undefined_money


def test_round_default_ndigits():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.money import Money
    
    money = Money.of(Currencies["USD"], Decimal('1.567'), Date(2019, 1, 1))
    rounded = money.round()
    assert rounded.qty == Decimal('2.00')


def test_round_negative_quantity():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.money import Money
    
    money = Money.of(Currencies["USD"], Decimal('-1.456'), Date(2019, 1, 1))
    rounded = money.round(2)
    assert rounded.qty == Decimal('-1.46')


def test_round_zero_quantity():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.money import Money
    
    money = Money.of(Currencies["USD"], Decimal('0.0'), Date(2019, 1, 1))
    rounded = money.round(2)
    assert rounded.qty == Decimal('0.00')


# LLM-generated content at query #85
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
    
    # Test with undefined price - should return default value
    noneprice = Price.of(None, Decimal('1'), None)
    result = noneprice.qty_or(Decimal(0))
    assert result == Decimal('0')
    
    # Test with defined price and different default
    someprice = Price.of(Currencies["USD"], Decimal('42'), Date(2019, 1, 1))
    result = someprice.qty_or(Decimal('100'))
    assert result == Decimal('42')
    
    # Test with undefined price and different default
    noneprice = Price.of(None, Decimal('1'), None)
    result = noneprice.qty_or(Decimal('99'))
    assert result == Decimal('99')
    
    # Test with defined price and negative quantity
    someprice = Price.of(Currencies["USD"], Decimal('-5'), Date(2019, 1, 1))
    result = someprice.qty_or(Decimal('0'))
    assert result == Decimal('-5')


# LLM-generated content at query #86
#--------------------------

```python
def test_somemoney_ge():
    from decimal import Decimal
    from datetime import date
    
    # Assuming Currency and other dependencies are available
    # Create test currencies and dates
    usd = Currency(code="USD", decimals=2)
    eur = Currency(code="EUR", decimals=2)
    test_date = date(2024, 1, 1)
    
    # Create SomeMoney instances
    money1 = SomeMoney(usd, Decimal("100.00"), test_date)
    money2 = SomeMoney(usd, Decimal("50.00"), test_date)
    money3 = SomeMoney(usd, Decimal("100.00"), test_date)
    money_different_ccy = SomeMoney(eur, Decimal("100.00"), test_date)
    
    # Test: money1 >= money2 should be True (100 >= 50)
    assert money1 >= money2 is True
    
    # Test: money2 >= money1 should be False (50 >= 100)
    assert money2 >= money1 is False
    
    # Test: money1 >= money3 should be True (100 >= 100)
    assert money1 >= money3 is True
    
    # Test: comparing with different currency should raise IncompatibleCurrencyError
    try:
        money1 >= money_different_ccy
        assert False, "Expected IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass
    
    # Test: comparing with non-SomeMoney should return True
    assert money1 >= "not a money object" is True


# LLM-generated content at query #87
#--------------------------

```python
def test_convert_with_valid_currencies():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    usd = Currencies["USD"]
    eur = Currencies["EUR"]
    money_usd = Money.of(usd, Decimal('100'), Date(2019, 1, 1))
    
    converted_money = money_usd.convert(eur, asof=Date(2019, 1, 1))
    
    assert converted_money is not None
    assert converted_money.defined


def test_convert_undefined_money():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    eur = Currencies["EUR"]
    undefined_money = Money.na()
    
    result = undefined_money.convert(eur, asof=Date(2019, 1, 1))
    
    assert result.undefined


def test_convert_same_currency():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    usd = Currencies["USD"]
    money_usd = Money.of(usd, Decimal('100'), Date(2019, 1, 1))
    
    converted_money = money_usd.convert(usd, asof=Date(2019, 1, 1))
    
    assert converted_money.defined
    assert converted_money.ccy_or_none() == usd


def test_convert_with_asof_date():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    usd = Currencies["USD"]
    eur = Currencies["EUR"]
    money_usd = Money.of(usd, Decimal('100'), Date(2019, 1, 1))
    
    converted_money = money_usd.convert(eur, asof=Date(2019, 6, 15))
    
    assert converted_money is not None


def test_convert_without_asof_date():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    usd = Currencies["USD"]
    eur = Currencies["EUR"]
    money_usd = Money.of(usd, Decimal('100'), Date(2019, 1, 1))
    
    converted_money = money_usd.convert(eur)
    
    assert converted_money is not None


def test_convert_with_strict_mode():
    from pypara.currencies import Currencies
    from datetime import date as Date
    from decimal import Decimal
    
    usd = Currencies["USD"]
    eur = Currencies["EUR"]
    money_usd = Money.of(usd, Decimal('100'), Date(2019, 1, 1))
    
    converted_money = money_usd.convert(eur, asof=Date(2019, 1, 1), strict=True)
    
    assert converted_money is not None


# LLM-generated content at query #88
#--------------------------

```python
def test_qty_map_with_defined_money():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.money import Money
    
    somemoney = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    result = somemoney.qty_map(lambda x: x + Decimal('1'), lambda: Decimal('42'))
    assert result == Decimal('2.00')


def test_qty_map_with_undefined_money():
    from decimal import Decimal
    from pypara.money import Money
    
    nonemoney = Money.of(None, Decimal('1'), None)
    result = nonemoney.qty_map(lambda x: x + Decimal('1'), lambda: Decimal('42'))
    assert result == Decimal('42')


def test_qty_map_with_defined_money_different_function():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.money import Money
    
    somemoney = Money.of(Currencies["USD"], Decimal('5'), Date(2019, 1, 1))
    result = somemoney.qty_map(lambda x: x * Decimal('2'), lambda: Decimal('0'))
    assert result == Decimal('10.00')


def test_qty_map_with_undefined_money_string_fallback():
    from decimal import Decimal
    from pypara.money import Money
    
    nonemoney = Money.of(None, Decimal('1'), None)
    result = nonemoney.qty_map(lambda x: x + Decimal('1'), lambda: "fallback")
    assert result == "fallback"


def test_qty_map_with_defined_money_string_function():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date as Date
    from pypara.money import Money
    
    somemoney = Money.of(Currencies["USD"], Decimal('3'), Date(2019, 1, 1))
    result = somemoney.qty_map(lambda x: str(x), lambda: "error")
    assert result == "3.00"


# LLM-generated content at query #89
#--------------------------

```python
def test_as_float():
    from decimal import Decimal
    from datetime import date as Date
    from pypara.currencies import Currencies
    
    # Test as_float on defined money
    defined_money = Money.of(Currencies["USD"], Decimal('123.45'), Date(2019, 1, 1))
    result = defined_money.as_float()
    assert isinstance(result, float)
    assert result == 123.45
    
    # Test as_float on undefined money raises exception
    undefined_money = Money.na()
    try:
        undefined_money.as_float()
        assert False, "Expected MonetaryOperationException"
    except Exception as e:
        assert "MonetaryOperationException" in str(type(e).__name__)


# LLM-generated content at query #90
#--------------------------

```python
def test_divide_defined_money_by_positive_number():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from decimal import Decimal
    from datetime import date as Date
    
    money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = money.divide(Decimal('2'))
    
    assert result.defined
    assert result.qty == Decimal('5')
    assert result.ccy.code == 'USD'


def test_divide_defined_money_by_negative_number():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from decimal import Decimal
    from datetime import date as Date
    
    money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = money.divide(Decimal('-2'))
    
    assert result.defined
    assert result.qty == Decimal('-5')


def test_divide_defined_money_by_zero():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from decimal import Decimal
    from datetime import date as Date
    
    money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = money.divide(Decimal('0'))
    
    assert result.undefined


def test_divide_undefined_money():
    from pypara.money import Money
    from decimal import Decimal
    
    money = Money.na()
    result = money.divide(Decimal('2'))
    
    assert result.undefined
    assert result is money


def test_divide_defined_money_by_fractional_number():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from decimal import Decimal
    from datetime import date as Date
    
    money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = money.divide(Decimal('2.5'))
    
    assert result.defined
    assert result.qty == Decimal('4')


def test_divide_preserves_currency():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from decimal import Decimal
    from datetime import date as Date
    
    money = Money.of(Currencies["EUR"], Decimal('100'), Date(2019, 1, 1))
    result = money.divide(Decimal('4'))
    
    assert result.ccy.code == 'EUR'


def test_divide_by_one():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from decimal import Decimal
    from datetime import date as Date
    
    money = Money.of(Currencies["USD"], Decimal('42'), Date(2019, 1, 1))
    result = money.divide(Decimal('1'))
    
    assert result.defined
    assert result.qty == Decimal('42')


