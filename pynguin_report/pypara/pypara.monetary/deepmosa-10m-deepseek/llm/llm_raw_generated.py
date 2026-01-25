####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```
def test_qty_or_none_returns_qty_when_defined():
    from pypara.currencies import Currencies
    from datetime import date
    price = Price.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    assert price.qty_or_none() == Decimal('1')

def test_qty_or_none_returns_none_when_undefined():
    from pypara.currencies import Currencies
    price = Price.of(None, Decimal('1'), None)
    assert price.qty_or_none() is None


# LLM-generated content at query #2
#--------------------------

```python
def test_lt_with_defined_prices_same_currency():
    from pypara.currencies import Currencies
    price1 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    assert price1.lt(price2) is True
    assert price2.lt(price1) is False

def test_lt_with_defined_prices_different_currencies():
    from pypara.currencies import Currencies
    price1 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["EUR"], Decimal('1'), Date(2019, 1, 1))
    try:
        price1.lt(price2)
        assert False, "Expected IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass

def test_lt_with_undefined_price_and_defined_price():
    from pypara.currencies import Currencies
    undefined_price = Price.na()
    defined_price = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert undefined_price.lt(defined_price) is True
    assert defined_price.lt(undefined_price) is False

def test_lt_with_two_undefined_prices():
    undefined_price1 = Price.na()
    undefined_price2 = Price.na()
    assert undefined_price1.lt(undefined_price2) is False


# LLM-generated content at query #3
#--------------------------

```
def test_positive_defined_money():
    from pypara.currencies import Currencies
    from datetime import date
    money = Money.of(Currencies["USD"], Decimal('10.50'), date(2023, 1, 1))
    result = money.__pos__()
    assert result.defined
    assert result.qty == Decimal('10.50')
    assert result.ccy == Currencies["USD"]
    assert result.dov == date(2023, 1, 1)

def test_positive_undefined_money():
    money = Money.na()
    result = money.__pos__()
    assert result.undefined


# LLM-generated content at query #4
#--------------------------

```
def test___ge___with_same_currency():
    ccy = Currency("USD", 2)
    money1 = SomeMoney(ccy, Decimal("100.00"), Date(2023, 1, 1))
    money2 = SomeMoney(ccy, Decimal("50.00"), Date(2023, 1, 1))
    assert money1 >= money2

def test___ge___with_different_currency():
    ccy1 = Currency("USD", 2)
    ccy2 = Currency("EUR", 2)
    money1 = SomeMoney(ccy1, Decimal("100.00"), Date(2023, 1, 1))
    money2 = SomeMoney(ccy2, Decimal("50.00"), Date(2023, 1, 1))
    try:
        money1 >= money2
        assert False, "Expected IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass

def test___ge___with_equal_quantities():
    ccy = Currency("USD", 2)
    money1 = SomeMoney(ccy, Decimal("100.00"), Date(2023, 1, 1))
    money2 = SomeMoney(ccy, Decimal("100.00"), Date(2023, 1, 1))
    assert money1 >= money2

def test___ge___with_non_money_object():
    ccy = Currency("USD", 2)
    money = SomeMoney(ccy, Decimal("100.00"), Date(2023, 1, 1))
    assert money >= "not money"


# LLM-generated content at query #5
#--------------------------

```python
def test_times_with_defined_price():
    from pypara.currencies import Currencies
    from datetime import date
    price = Price.of(Currencies["USD"], Decimal('10'), date(2023, 1, 1))
    result = price.times(Decimal('2'))
    assert result.qty == Decimal('20')
    assert result.ccy == Currencies["USD"]
    assert result.dov == date(2023, 1, 1)

def test_times_with_undefined_price():
    undefined_price = Price.na()
    result = undefined_price.times(Decimal('2'))
    assert result == Price.na()


# LLM-generated content at query #6
#--------------------------

```
def test_gt_with_defined_money_and_greater_quantity():
    ccy = Currencies["USD"]
    money1 = Money.of(ccy, Decimal('2'), Date(2019, 1, 1))
    money2 = Money.of(ccy, Decimal('1'), Date(2019, 1, 1))
    assert money1.gt(money2) is True

def test_gt_with_defined_money_and_lesser_quantity():
    ccy = Currencies["USD"]
    money1 = Money.of(ccy, Decimal('1'), Date(2019, 1, 1))
    money2 = Money.of(ccy, Decimal('2'), Date(2019, 1, 1))
    assert money1.gt(money2) is False

def test_gt_with_defined_money_and_equal_quantity():
    ccy = Currencies["USD"]
    money1 = Money.of(ccy, Decimal('1'), Date(2019, 1, 1))
    money2 = Money.of(ccy, Decimal('1'), Date(2019, 1, 1))
    assert money1.gt(money2) is False

def test_gt_with_undefined_money_and_defined_money():
    money1 = Money.na()
    money2 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert money1.gt(money2) is False

def test_gt_with_defined_money_and_undefined_money():
    money1 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    money2 = Money.na()
    assert money1.gt(money2) is True

def test_gt_with_undefined_money_and_undefined_money():
    money1 = Money.na()
    money2 = Money.na()
    assert money1.gt(money2) is False

def test_gt_with_incompatible_currencies():
    money1 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["EUR"], Decimal('1'), Date(2019, 1, 1))
    try:
        money1.gt(money2)
        assert False
    except IncompatibleCurrencyError:
        assert True


# LLM-generated content at query #7
#--------------------------

```
def test_ccy_or_returns_ccy_when_price_is_defined():
    from pypara.currencies import Currencies
    someprice = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert someprice.ccy_or(Currencies["EUR"]).code == "USD"

def test_ccy_or_returns_default_when_price_is_undefined():
    from pypara.currencies import Currencies
    someprice = Price.of(Currencies["USD"], None, None)
    assert someprice.ccy_or(Currencies["EUR"]).code == "EUR"


# LLM-generated content at query #8
#--------------------------

```
def test_gte_defined_price_greater_than_defined_price_same_currency():
    from pypara.currencies import Currencies
    from pypara.monetary import Price
    price1 = Price.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert price1.gte(price2) is True

def test_gte_defined_price_equal_to_defined_price_same_currency():
    from pypara.currencies import Currencies
    from pypara.monetary import Price
    price1 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert price1.gte(price2) is True

def test_gte_defined_price_less_than_defined_price_same_currency():
    from pypara.currencies import Currencies
    from pypara.monetary import Price
    price1 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    assert price1.gte(price2) is False

def test_gte_undefined_price_greater_than_defined_price():
    from pypara.currencies import Currencies
    from pypara.monetary import Price
    price1 = Price.na()
    price2 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert price1.gte(price2) is False

def test_gte_undefined_price_greater_than_undefined_price():
    from pypara.monetary import Price
    price1 = Price.na()
    price2 = Price.na()
    assert price1.gte(price2) is True

def test_gte_defined_price_greater_than_undefined_price():
    from pypara.currencies import Currencies
    from pypara.monetary import Price
    price1 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    price2 = Price.na()
    assert price1.gte(price2) is True


# LLM-generated content at query #9
#--------------------------

```python
def test___bool__with_defined_price():
    from pypara.currencies import Currencies
    from datetime import date
    price = Price.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    assert bool(price) == True

def test___bool__with_undefined_price():
    price = Price.na()
    assert bool(price) == False

def test___bool__with_zero_quantity():
    from pypara.currencies import Currencies
    from datetime import date
    price = Price.of(Currencies["USD"], Decimal('0'), date(2019, 1, 1))
    assert bool(price) == False


# LLM-generated content at query #10
#--------------------------

```
def test_floor_divide_defined_money():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    money = Money.of(Currencies["USD"], Decimal('10'), date(2020, 1, 1))
    result = money.floor_divide(Decimal('3'))
    assert result.qty == Decimal('3')
    assert result.ccy.code == 'USD'
    assert result.dov == date(2020, 1, 1)

def test_floor_divide_undefined_money():
    undefined_money = Money.na()
    result = undefined_money.floor_divide(Decimal('5'))
    assert result is undefined_money

def test_floor_divide_by_zero():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    money = Money.of(Currencies["USD"], Decimal('10'), date(2020, 1, 1))
    result = money.floor_divide(Decimal('0'))
    assert result is Money.na()

def test_floor_divide_negative_divisor():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    money = Money.of(Currencies["USD"], Decimal('10'), date(2020, 1, 1))
    result = money.floor_divide(Decimal('-3'))
    assert result.qty == Decimal('-4')
    assert result.ccy.code == 'USD'
    assert result.dov == date(2020, 1, 1)


# LLM-generated content at query #11
#--------------------------

```python
def test_convert_with_valid_currency_and_date():
    from pypara.currencies import Currencies
    from pypara.prices import Price
    from datetime import date
    original_price = Price.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    converted_price = original_price.convert(Currencies["EUR"], date(2019, 1, 1))
    assert converted_price.ccy.code == "EUR"
    assert converted_price.qty > Decimal('0')
    assert converted_price.dov == date(2019, 1, 1)

def test_convert_with_undefined_price():
    from pypara.currencies import Currencies
    from pypara.prices import Price
    original_price = Price.na()
    converted_price = original_price.convert(Currencies["EUR"])
    assert converted_price.undefined

def test_convert_with_invalid_currency():
    from pypara.currencies import Currencies
    from pypara.prices import Price
    from datetime import date
    original_price = Price.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    try:
        original_price.convert(Currencies["XYZ"])
        assert False
    except FXRateLookupError:
        assert True

def test_convert_without_date():
    from pypara.currencies import Currencies
    from pypara.prices import Price
    from datetime import date
    original_price = Price.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    converted_price = original_price.convert(Currencies["EUR"])
    assert converted_price.ccy.code == "EUR"
    assert converted_price.qty > Decimal('0')
    assert converted_price.dov == date(2019, 1, 1)


# LLM-generated content at query #12
#--------------------------

```python
def test_lt_undefined_price():
    undefined_price = Price.na()
    defined_price = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert undefined_price.lt(defined_price) == True

def test_lt_defined_prices_with_different_currencies():
    price1 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["EUR"], Decimal('1'), Date(2019, 1, 1))
    try:
        price1.lt(price2)
        assert False, "Expected IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        assert True

def test_lt_defined_prices_with_same_currency():
    price1 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    assert price1.lt(price2) == True
    assert price2.lt(price1) == False

def test_lt_undefined_prices():
    undefined_price1 = Price.na()
    undefined_price2 = Price.na()
    assert undefined_price1.lt(undefined_price2) == False


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_qty_or_none_returns_none_for_undefined_price():
    undefined_price = Price.na()
    result = undefined_price.qty_or_none()
    assert result is None

def test_qty_or_none_returns_qty_for_defined_price():
    from pypara.currencies import Currencies
    from datetime import date
    defined_price = Price.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    result = defined_price.qty_or_none()
    assert result == Decimal('1')


# LLM-generated content at query #2
#--------------------------

```python
def test_eq():
    from pypara.currencies import Currencies
    from datetime import date

    money1 = Money.of(Currencies["USD"], Decimal('1'), date(2023, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('1'), date(2023, 1, 1))
    money3 = Money.of(Currencies["EUR"], Decimal('1'), date(2023, 1, 1))
    money4 = Money.of(Currencies["USD"], Decimal('2'), date(2023, 1, 1))
    money5 = Money.of(Currencies["USD"], Decimal('1'), date(2023, 1, 2))
    money6 = Money.na()

    assert money1 == money2
    assert money1 != money3
    assert money1 != money4
    assert money1 != money5
    assert money1 != money6
    assert money6 == money6


# LLM-generated content at query #3
#--------------------------

```
def test___truediv___with_defined_money():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    money = Money.of(Currencies["USD"], Decimal('10'), date(2020, 1, 1))
    result = money / Decimal('2')
    assert result.qty == Decimal('5')
    assert result.ccy == Currencies["USD"]
    assert result.dov == date(2020, 1, 1)

def test___truediv___with_undefined_money():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    money = Money.na()
    result = money / Decimal('2')
    assert result.undefined

def test___truediv___with_division_by_zero():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    money = Money.of(Currencies["USD"], Decimal('10'), date(2020, 1, 1))
    result = money / Decimal('0')
    assert result.undefined


# LLM-generated content at query #4
#--------------------------

```python
def test_none_money_constructor():
    money = NoneMoney()
    assert money.defined == False
    assert money.undefined == True
    assert money.as_boolean() == False
    assert money.is_equal(NoneMoney()) == True
    assert money.abs() == money
    assert money.round() == money
    assert money.negative() == money
    assert money.positive() == money
    assert money.add(NoneMoney()) == NoneMoney()
    assert money.scalar_add(1) == money
    assert money.subtract(NoneMoney()) == NoneMoney()
    assert money.scalar_subtract(1) == money
    assert money.multiply(2) == money
    assert money.divide(2) == money
    assert money.floor_divide(2) == money
    assert money.lt(NoneMoney()) == True
    assert money.lte(NoneMoney()) == True
    assert money.gt(NoneMoney()) == False
    assert money.gte(NoneMoney()) == True
    assert money.or_else(lambda: NoneMoney()) == NoneMoney()
    assert money.fmap(lambda x: x) == money
    assert money.dimap(lambda x: x, lambda: NoneMoney()) == NoneMoney()
    assert money.with_ccy("USD") == money
    assert money.with_qty(Decimal("1")) == money
    assert money.with_dov(date.today()) == money
    assert money.ccy_or("USD") == "USD"
    assert money.ccy_or_none() == None
    assert money.qty_or(Decimal("1")) == Decimal("1")
    assert money.qty_or_zero() == Decimal("0")
    assert money.qty_or_none() == None
    assert money.qty_or_else(lambda: Decimal("1")) == Decimal("1")
    assert money.qty_map(lambda x: x, lambda: Decimal("1")) == Decimal("1")
    assert money.dov_or(date.today()) == date.today()
    assert money.dov_or_none() == None
    assert money.convert("EUR") == money
    assert money.price == NoPrice


# LLM-generated content at query #5
#--------------------------

```python
def test_add_defined_money_with_same_currency():
    ccy = Currencies["USD"]
    m1 = Money.of(ccy, Decimal('10'), Date(2023, 1, 1))
    m2 = Money.of(ccy, Decimal('20'), Date(2023, 1, 2))
    result = m1.add(m2)
    assert result.defined
    assert result.ccy == ccy
    assert result.qty == Decimal('30')
    assert result.dov == Date(2023, 1, 2)

def test_add_defined_money_with_different_currency():
    ccy1 = Currencies["USD"]
    ccy2 = Currencies["EUR"]
    m1 = Money.of(ccy1, Decimal('10'), Date(2023, 1, 1))
    m2 = Money.of(ccy2, Decimal('20'), Date(2023, 1, 2))
    try:
        m1.add(m2)
        assert False
    except IncompatibleCurrencyError:
        assert True

def test_add_undefined_money_with_defined_money():
    ccy = Currencies["USD"]
    m1 = Money.na()
    m2 = Money.of(ccy, Decimal('20'), Date(2023, 1, 2))
    result = m1.add(m2)
    assert result.defined
    assert result.ccy == ccy
    assert result.qty == Decimal('20')
    assert result.dov == Date(2023, 1, 2)

def test_add_defined_money_with_undefined_money():
    ccy = Currencies["USD"]
    m1 = Money.of(ccy, Decimal('10'), Date(2023, 1, 1))
    m2 = Money.na()
    result = m1.add(m2)
    assert result.defined
    assert result.ccy == ccy
    assert result.qty == Decimal('10')
    assert result.dov == Date(2023, 1, 1)

def test_add_undefined_money_with_undefined_money():
    m1 = Money.na()
    m2 = Money.na()
    result = m1.add(m2)
    assert result.undefined


# LLM-generated content at query #6
#--------------------------

```python
def test_positive_defined_price():
    price = Price.of(Currencies["USD"], Decimal('100'), Date(2023, 1, 1))
    result = price.positive()
    assert result == price

def test_positive_undefined_price():
    price = Price.na()
    result = price.positive()
    assert result == price


# LLM-generated content at query #7
#--------------------------

```
def test_ccy_or_returns_ccy_when_defined():
    from pypara.currencies import Currencies
    somemoney = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert somemoney.ccy_or(Currencies["EUR"]).code == "USD"

def test_ccy_or_returns_default_when_undefined():
    from pypara.currencies import Currencies
    nonemoney = Money.of(Currencies["USD"], None, None)
    assert nonemoney.ccy_or(Currencies["EUR"]).code == "EUR"


# LLM-generated content at query #8
#--------------------------

```python
def test_as_boolean_returns_false_for_undefined_money():
    undefined_money = Money.na()
    assert not undefined_money.as_boolean()

def test_as_boolean_returns_false_for_zero_quantity():
    zero_money = Money.of(Currencies["USD"], Decimal('0'), Date(2019, 1, 1))
    assert not zero_money.as_boolean()

def test_as_boolean_returns_true_for_defined_non_zero_money():
    non_zero_money = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert non_zero_money.as_boolean()


# LLM-generated content at query #9
#--------------------------

```python
def test_with_qty_updates_quantity_for_defined_money():
    from pypara.currencies import Currencies
    from pypara.money import Money
    money = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    updated_money = money.with_qty(Decimal('2'))
    assert updated_money.qty == Decimal('2')

def test_with_qty_returns_self_for_undefined_money():
    from pypara.money import Money
    undefined_money = Money.na()
    updated_money = undefined_money.with_qty(Decimal('2'))
    assert updated_money is undefined_money


# LLM-generated content at query #10
#--------------------------

```python
def test_lt_with_defined_prices_same_currency():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    price1 = Price.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('2'), date(2019, 1, 1))
    assert price1.lt(price2) == True

def test_lt_with_undefined_price():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    price1 = Price.na()
    price2 = Price.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    assert price1.lt(price2) == True

def test_lt_with_undefined_prices():
    price1 = Price.na()
    price2 = Price.na()
    assert price1.lt(price2) == False

def test_lt_with_different_currencies():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    price1 = Price.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    price2 = Price.of(Currencies["EUR"], Decimal('1'), date(2019, 1, 1))
    try:
        price1.lt(price2)
        assert False, "Expected IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        assert True

def test_lt_with_equal_prices():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    price1 = Price.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    assert price1.lt(price2) == False


# LLM-generated content at query #11
#--------------------------

```python
def test_qty_or_none_defined_price():
    from decimal import Decimal
    from pypara.currencies import Currencies
    from pypara.prices import Price, Date
    someprice = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert someprice.qty_or_none() == Decimal('1')

def test_qty_or_none_undefined_price():
    from decimal import Decimal
    from pypara.prices import Price
    noneprice = Price.of(None, Decimal('1'), None)
    assert noneprice.qty_or_none() is None


# LLM-generated content at query #12
#--------------------------

```python
def test_gt_with_defined_prices():
    ccy = Currencies["USD"]
    dov = Date(2019, 1, 1)
    price1 = Price.of(ccy, Decimal('2'), dov)
    price2 = Price.of(ccy, Decimal('1'), dov)
    assert price1.gt(price2) == True

def test_gt_with_undefined_price():
    ccy = Currencies["USD"]
    dov = Date(2019, 1, 1)
    price1 = Price.na()
    price2 = Price.of(ccy, Decimal('1'), dov)
    assert price1.gt(price2) == False

def test_gt_with_both_undefined_prices():
    price1 = Price.na()
    price2 = Price.na()
    assert price1.gt(price2) == False

def test_gt_with_different_currencies():
    ccy1 = Currencies["USD"]
    ccy2 = Currencies["EUR"]
    dov = Date(2019, 1, 1)
    price1 = Price.of(ccy1, Decimal('2'), dov)
    price2 = Price.of(ccy2, Decimal('1'), dov)
    try:
        price1.gt(price2)
        assert False
    except IncompatibleCurrencyError:
        assert True


# LLM-generated content at query #13
#--------------------------

```python
def test_floor_divide_defined_money():
    from pypara.currencies import Currencies
    money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = money.floor_divide(Decimal('3'))
    assert result.qty == Decimal('3')

def test_floor_divide_undefined_money():
    undefined_money = Money.na()
    result = undefined_money.floor_divide(Decimal('3'))
    assert result.undefined

def test_floor_divide_by_zero():
    from pypara.currencies import Currencies
    money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = money.floor_divide(Decimal('0'))
    assert result.undefined

def test_floor_divide_negative_divisor():
    from pypara.currencies import Currencies
    money = Money.of(Currencies["USD"], Decimal('10'), Date(2019, 1, 1))
    result = money.floor_divide(Decimal('-3'))
    assert result.qty == Decimal('-4')


# LLM-generated content at query #14
#--------------------------

```python
def test___add___with_same_currency():
    ccy = Currency("USD")
    qty1 = Decimal("10.00")
    qty2 = Decimal("20.00")
    dov = Date.today()
    money1 = SomeMoney(ccy, qty1, dov)
    money2 = SomeMoney(ccy, qty2, dov)
    result = money1 + money2
    assert result == SomeMoney(ccy, Decimal("30.00"), dov)

def test___add___with_different_currencies():
    ccy1 = Currency("USD")
    ccy2 = Currency("EUR")
    qty1 = Decimal("10.00")
    qty2 = Decimal("20.00")
    dov = Date.today()
    money1 = SomeMoney(ccy1, qty1, dov)
    money2 = SomeMoney(ccy2, qty2, dov)
    try:
        money1 + money2
        assert False, "Expected IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass

def test___add___with_undefined_money():
    ccy = Currency("USD")
    qty1 = Decimal("10.00")
    dov = Date.today()
    money1 = SomeMoney(ccy, qty1, dov)
    money2 = NoMoney
    result = money1 + money2
    assert result == money1

def test___add___with_different_dates():
    ccy = Currency("USD")
    qty1 = Decimal("10.00")
    qty2 = Decimal("20.00")
    dov1 = Date.today()
    dov2 = Date.today() + timedelta(days=1)
    money1 = SomeMoney(ccy, qty1, dov1)
    money2 = SomeMoney(ccy, qty2, dov2)
    result = money1 + money2
    assert result == SomeMoney(ccy, Decimal("30.00"), dov2)


# LLM-generated content at query #15
#--------------------------

```python
def test_lt_defined_less_than_defined():
    money1 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    assert money1.lt(money2) == True

def test_lt_defined_not_less_than_defined():
    money1 = Money.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert money1.lt(money2) == False

def test_lt_undefined_less_than_defined():
    money1 = Money.na()
    money2 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert money1.lt(money2) == True

def test_lt_defined_not_less_than_undefined():
    money1 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    money2 = Money.na()
    assert money1.lt(money2) == False

def test_lt_undefined_not_less_than_undefined():
    money1 = Money.na()
    money2 = Money.na()
    assert money1.lt(money2) == False

def test_lt_incompatible_currencies():
    money1 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["EUR"], Decimal('1'), Date(2019, 1, 1))
    try:
        money1.lt(money2)
        assert False, "Expected IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        assert True


# LLM-generated content at query #16
#--------------------------

```python
def test_qty_or_zero_with_defined_money():
    from pypara.currencies import Currencies
    money = Money.of(Currencies["USD"], Decimal('100.50'), Date(2023, 1, 1))
    assert money.qty_or_zero() == Decimal('100.50')

def test_qty_or_zero_with_undefined_money():
    money = Money.na()
    assert money.qty_or_zero() == Decimal('0')

def test_qty_or_zero_with_zero_quantity():
    from pypara.currencies import Currencies
    money = Money.of(Currencies["USD"], Decimal('0'), Date(2023, 1, 1))
    assert money.qty_or_zero() == Decimal('0')


# LLM-generated content at query #17
#--------------------------

```python
def test___eq__():
    from pypara.currencies import Currencies
    from datetime import date

    price1 = Price.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    price3 = Price.of(Currencies["EUR"], Decimal('1'), date(2019, 1, 1))
    price4 = Price.of(Currencies["USD"], Decimal('2'), date(2019, 1, 1))
    price5 = Price.of(Currencies["USD"], Decimal('1'), date(2019, 1, 2))
    price6 = Price.na()

    assert price1 == price2
    assert not (price1 == price3)
    assert not (price1 == price4)
    assert not (price1 == price5)
    assert not (price1 == price6)
    assert price6 == Price.na()


# LLM-generated content at query #18
#--------------------------

```
def test_as_integer_defined():
    from pypara.currencies import Currencies
    from datetime import date
    money = Money.of(Currencies["USD"], Decimal('10.5'), date(2023, 1, 1))
    assert money.as_integer() == 10

def test_as_integer_undefined():
    money = Money.na()
    try:
        money.as_integer()
        assert False
    except MonetaryOperationException:
        assert True


# LLM-generated content at query #19
#--------------------------

```python
def test_with_qty_updates_quantity_correctly():
    price = SomePrice(Currency("USD"), Decimal("100.00"), Date("2023-10-01"))
    new_price = price.with_qty(Decimal("200.00"))
    assert new_price.qty == Decimal("200.00")
    assert new_price.ccy == Currency("USD")
    assert new_price.dov == Date("2023-10-01")

def test_with_qty_returns_new_instance():
    price = SomePrice(Currency("USD"), Decimal("100.00"), Date("2023-10-01"))
    new_price = price.with_qty(Decimal("200.00"))
    assert new_price is not price

def test_with_qty_handles_zero_quantity():
    price = SomePrice(Currency("USD"), Decimal("100.00"), Date("2023-10-01"))
    new_price = price.with_qty(Decimal("0.00"))
    assert new_price.qty == Decimal("0.00")

def test_with_qty_handles_negative_quantity():
    price = SomePrice(Currency("USD"), Decimal("100.00"), Date("2023-10-01"))
    new_price = price.with_qty(Decimal("-50.00"))
    assert new_price.qty == Decimal("-50.00")


# LLM-generated content at query #20
#--------------------------

```python
def test_subtract_defined_money_objects():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from datetime import date

    money1 = Money.of(Currencies["USD"], Decimal('10'), date(2023, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('3'), date(2023, 1, 1))
    result = money1 - money2
    assert result.qty == Decimal('7')
    assert result.ccy.code == 'USD'
    assert result.dov == date(2023, 1, 1)

def test_subtract_undefined_money_object():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from datetime import date

    money1 = Money.of(Currencies["USD"], Decimal('10'), date(2023, 1, 1))
    undefined_money = Money.na()
    result = money1 - undefined_money
    assert result == money1

def test_subtract_incompatible_currencies():
    from pypara.currencies import Currencies
    from pypara.money import Money
    from datetime import date

    money1 = Money.of(Currencies["USD"], Decimal('10'), date(2023, 1, 1))
    money2 = Money.of(Currencies["EUR"], Decimal('3'), date(2023, 1, 1))
    try:
        _ = money1 - money2
        assert False, "Expected IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        assert True


# LLM-generated content at query #21
#--------------------------

```python
def test_gt_defined_price_greater_than_undefined():
    defined_price = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    undefined_price = Price.na()
    result = defined_price.gt(undefined_price)
    assert result is True

def test_gt_undefined_price_not_greater_than_defined():
    defined_price = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    undefined_price = Price.na()
    result = undefined_price.gt(defined_price)
    assert result is False

def test_gt_defined_price_greater_than_defined_with_different_currency():
    price1 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["EUR"], Decimal('1'), Date(2019, 1, 1))
    try:
        price1.gt(price2)
        assert False, "Expected IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass

def test_gt_defined_price_greater_than_defined_with_same_currency():
    price1 = Price.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    result = price1.gt(price2)
    assert result is True

def test_gt_undefined_price_not_greater_than_undefined():
    undefined_price1 = Price.na()
    undefined_price2 = Price.na()
    result = undefined_price1.gt(undefined_price2)
    assert result is False


# LLM-generated content at query #22
#--------------------------

```python
def test___truediv___with_valid_divisor():
    ccy = Currency("USD")
    qty = Decimal("10.0")
    dov = Date(2023, 1, 1)
    price = SomePrice(ccy, qty, dov)
    result = price / Decimal("2.0")
    assert result == SomePrice(ccy, Decimal("5.0"), dov)

def test___truediv___with_zero_divisor():
    ccy = Currency("USD")
    qty = Decimal("10.0")
    dov = Date(2023, 1, 1)
    price = SomePrice(ccy, qty, dov)
    result = price / Decimal("0.0")
    assert result == NoPrice

def test___truediv___with_negative_divisor():
    ccy = Currency("USD")
    qty = Decimal("10.0")
    dov = Date(2023, 1, 1)
    price = SomePrice(ccy, qty, dov)
    result = price / Decimal("-2.0")
    assert result == SomePrice(ccy, Decimal("-5.0"), dov)

def test___truediv___with_integer_divisor():
    ccy = Currency("USD")
    qty = Decimal("10.0")
    dov = Date(2023, 1, 1)
    price = SomePrice(ccy, qty, dov)
    result = price / 2
    assert result == SomePrice(ccy, Decimal("5.0"), dov)

def test___truediv___with_float_divisor():
    ccy = Currency("USD")
    qty = Decimal("10.0")
    dov = Date(2023, 1, 1)
    price = SomePrice(ccy, qty, dov)
    result = price / 2.0
    assert result == SomePrice(ccy, Decimal("5.0"), dov)


# LLM-generated content at query #23
#--------------------------

```python
def test_neg_returns_negative_of_defined_price():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    price = Price.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    neg_price = price.__neg__()
    assert neg_price.qty == Decimal('-1')

def test_neg_returns_itself_for_undefined_price():
    undefined_price = Price.na()
    neg_price = undefined_price.__neg__()
    assert neg_price is undefined_price


# LLM-generated content at query #24
#--------------------------

```python
def test___lt___with_same_currency_and_lesser_qty():
    ccy = Currency("USD")
    price1 = SomePrice(ccy, Decimal("100.00"), Date(2023, 1, 1))
    price2 = SomePrice(ccy, Decimal("200.00"), Date(2023, 1, 1))
    assert price1 < price2

def test___lt___with_same_currency_and_greater_qty():
    ccy = Currency("USD")
    price1 = SomePrice(ccy, Decimal("200.00"), Date(2023, 1, 1))
    price2 = SomePrice(ccy, Decimal("100.00"), Date(2023, 1, 1))
    assert not (price1 < price2)

def test___lt___with_same_currency_and_equal_qty():
    ccy = Currency("USD")
    price1 = SomePrice(ccy, Decimal("100.00"), Date(2023, 1, 1))
    price2 = SomePrice(ccy, Decimal("100.00"), Date(2023, 1, 1))
    assert not (price1 < price2)

def test___lt___with_different_currency():
    ccy1 = Currency("USD")
    ccy2 = Currency("EUR")
    price1 = SomePrice(ccy1, Decimal("100.00"), Date(2023, 1, 1))
    price2 = SomePrice(ccy2, Decimal("100.00"), Date(2023, 1, 1))
    try:
        price1 < price2
        assert False, "Should raise IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass

def test___lt___with_non_SomePrice_object():
    ccy = Currency("USD")
    price = SomePrice(ccy, Decimal("100.00"), Date(2023, 1, 1))
    assert not (price < "not a SomePrice object")


# LLM-generated content at query #25
#--------------------------

```python
def test___bool__():
    from pypara.currencies import Currencies
    from datetime import date

    price_defined = Price.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    price_undefined = Price.na()

    assert bool(price_defined) == True
    assert bool(price_undefined) == False


# LLM-generated content at query #26
#--------------------------

```python
def test_round_defined_price():
    price = Price.of(Currencies["USD"], Decimal('1.2345'), Date(2019, 1, 1))
    rounded_price = price.round(2)
    assert rounded_price.qty == Decimal('1.23')

def test_round_undefined_price():
    undefined_price = Price.na()
    rounded_price = undefined_price.round(2)
    assert rounded_price is undefined_price

def test_round_zero_ndigits():
    price = Price.of(Currencies["USD"], Decimal('1.2345'), Date(2019, 1, 1))
    rounded_price = price.round(0)
    assert rounded_price.qty == Decimal('1')

def test_round_negative_ndigits():
    price = Price.of(Currencies["USD"], Decimal('123.45'), Date(2019, 1, 1))
    rounded_price = price.round(-1)
    assert rounded_price.qty == Decimal('120')


# LLM-generated content at query #27
#--------------------------

```python
def test_some_price_add_with_same_currency():
    ccy = Currency("USD")
    price1 = SomePrice(ccy, Decimal("10.50"), Date(2023, 1, 1))
    price2 = SomePrice(ccy, Decimal("5.25"), Date(2023, 1, 2))
    result = price1 + price2
    assert result.ccy == ccy
    assert result.qty == Decimal("15.75")
    assert result.dov == Date(2023, 1, 2)

def test_some_price_add_with_different_currency():
    ccy1 = Currency("USD")
    ccy2 = Currency("EUR")
    price1 = SomePrice(ccy1, Decimal("10.50"), Date(2023, 1, 1))
    price2 = SomePrice(ccy2, Decimal("5.25"), Date(2023, 1, 2))
    try:
        price1 + price2
        assert False, "Should have raised IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass

def test_some_price_add_with_undefined_price():
    ccy = Currency("USD")
    price1 = SomePrice(ccy, Decimal("10.50"), Date(2023, 1, 1))
    price2 = NoPrice
    result = price1 + price2
    assert result == price1

def test_some_price_add_with_later_date():
    ccy = Currency("USD")
    price1 = SomePrice(ccy, Decimal("10.50"), Date(2023, 1, 2))
    price2 = SomePrice(ccy, Decimal("5.25"), Date(2023, 1, 1))
    result = price1 + price2
    assert result.dov == Date(2023, 1, 2)

def test_some_price_add_with_earlier_date():
    ccy = Currency("USD")
    price1 = SomePrice(ccy, Decimal("10.50"), Date(2023, 1, 1))
    price2 = SomePrice(ccy, Decimal("5.25"), Date(2023, 1, 2))
    result = price1 + price2
    assert result.dov == Date(2023, 1, 2)


# LLM-generated content at query #28
#--------------------------

```python
def test_as_float_defined_money():
    from pypara.currencies import Currencies
    from pypara.money import Money
    money = Money.of(Currencies["USD"], Decimal('1.23'), Date(2019, 1, 1))
    assert money.as_float() == 1.23

def test_as_float_undefined_money():
    from pypara.money import Money
    money = Money.na()
    try:
        money.as_float()
        assert False, "Expected MonetaryOperationException"
    except MonetaryOperationException:
        assert True


# LLM-generated content at query #29
#--------------------------

```python
def test_convert_same_currency():
    ccy = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    money = SomeMoney(ccy, Decimal("100.00"), Date(2023, 1, 1))
    converted = money.convert(ccy, Date(2023, 1, 1))
    assert converted == money

def test_convert_different_currency():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    rate = FXRate(ccy1, ccy2, Date(2023, 1, 1), Decimal("0.85"))
    FXRateService.default = FXRateService(lambda c1, c2, d, s: rate)
    money = SomeMoney(ccy1, Decimal("100.00"), Date(2023, 1, 1))
    converted = money.convert(ccy2, Date(2023, 1, 1))
    assert converted == SomeMoney(ccy2, Decimal("85.00"), Date(2023, 1, 1))

def test_convert_strict_mode_no_rate():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    FXRateService.default = FXRateService(lambda c1, c2, d, s: None)
    money = SomeMoney(ccy1, Decimal("100.00"), Date(2023, 1, 1))
    try:
        money.convert(ccy2, Date(2023, 1, 1), strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError:
        pass

def test_convert_non_strict_mode_no_rate():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    FXRateService.default = FXRateService(lambda c1, c2, d, s: None)
    money = SomeMoney(ccy1, Decimal("100.00"), Date(2023, 1, 1))
    converted = money.convert(ccy2, Date(2023, 1, 1), strict=False)
    assert converted == NoMoney

def test_convert_with_asof_date():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    rate = FXRate(ccy1, ccy2, Date(2023, 1, 1), Decimal("0.85"))
    FXRateService.default = FXRateService(lambda c1, c2, d, s: rate)
    money = SomeMoney(ccy1, Decimal("100.00"), Date(2022, 12, 31))
    converted = money.convert(ccy2, Date(2023, 1, 1))
    assert converted == SomeMoney(ccy2, Decimal("85.00"), Date(2023, 1, 1))


# LLM-generated content at query #30
#--------------------------

```python
def test_lte_comparison_with_defined_prices():
    from pypara.currencies import Currencies
    from pypara.monetary import Price
    price1 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    price3 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert price1.lte(price2) == True
    assert price2.lte(price1) == False
    assert price1.lte(price3) == True

def test_lte_comparison_with_undefined_prices():
    from pypara.monetary import Price
    price1 = Price.na()
    price2 = Price.na()
    defined_price = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert price1.lte(defined_price) == True
    assert defined_price.lte(price1) == False
    assert price1.lte(price2) == True

def test_lte_comparison_with_incompatible_currencies():
    from pypara.currencies import Currencies
    from pypara.monetary import Price
    price1 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["EUR"], Decimal('1'), Date(2019, 1, 1))
    try:
        price1.lte(price2)
        assert False, "Expected IncompatibleCurrencyError to be raised"
    except IncompatibleCurrencyError:
        assert True


# LLM-generated content at query #31
#--------------------------

```python
def test_convert_handles_attribute_error_when_fx_rate_service_default_is_not_none():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    money = SomeMoney(usd, Decimal("100"), Date.today())
    FXRateService.default = object()  # Set to non-None but invalid service
    try:
        money.convert(usd)
        assert False, "Expected AttributeError to be raised"
    except AttributeError:
        pass
    finally:
        FXRateService.default = None


# LLM-generated content at query #32
#--------------------------

```python
def test_truediv_with_defined_price():
    ccy = Currency("USD", 2)
    price = Price.of(ccy, Decimal('10'), Date(2023, 1, 1))
    result = price / Decimal('2')
    assert result.qty == Decimal('5')
    assert result.ccy == ccy
    assert result.dov == Date(2023, 1, 1)

def test_truediv_with_undefined_price():
    price = Price.na()
    result = price / Decimal('2')
    assert result.undefined

def test_truediv_with_zero_divisor():
    ccy = Currency("USD", 2)
    price = Price.of(ccy, Decimal('10'), Date(2023, 1, 1))
    result = price / Decimal('0')
    assert result.undefined

def test_truediv_with_negative_divisor():
    ccy = Currency("USD", 2)
    price = Price.of(ccy, Decimal('10'), Date(2023, 1, 1))
    result = price / Decimal('-2')
    assert result.qty == Decimal('-5')
    assert result.ccy == ccy
    assert result.dov == Date(2023, 1, 1)

def test_truediv_with_non_decimal_divisor():
    ccy = Currency("USD", 2)
    price = Price.of(ccy, Decimal('10'), Date(2023, 1, 1))
    result = price / 2
    assert result.qty == Decimal('5')
    assert result.ccy == ccy
    assert result.dov == Date(2023, 1, 1)


# LLM-generated content at query #33
#--------------------------

```python
def test_with_qty_updates_quantity_for_defined_price():
    price = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    new_price = price.with_qty(Decimal('2'))
    assert new_price.qty == Decimal('2')

def test_with_qty_returns_same_for_undefined_price():
    price = Price.na()
    new_price = price.with_qty(Decimal('2'))
    assert new_price is price


# LLM-generated content at query #34
#--------------------------

```python
def test_is_equal_with_same_money_objects():
    from pypara.currencies import Currencies
    money1 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert money1.is_equal(money2)

def test_is_equal_with_different_money_objects():
    from pypara.currencies import Currencies
    money1 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    money2 = Money.of(Currencies["EUR"], Decimal('1'), Date(2019, 1, 1))
    assert not money1.is_equal(money2)

def test_is_equal_with_undefined_money():
    from pypara.currencies import Currencies
    money1 = Money.na()
    money2 = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert not money1.is_equal(money2)

def test_is_equal_with_two_undefined_money_objects():
    money1 = Money.na()
    money2 = Money.na()
    assert money1.is_equal(money2)


# LLM-generated content at query #35
#--------------------------

```python
def test_or_else_returns_itself_if_defined():
    from pypara.currencies import Currencies
    someprice = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    fallback = Price.of(Currencies["EUR"], Decimal('2'), Date(2019, 1, 2))
    assert someprice.or_else(lambda: fallback) is someprice

def test_or_else_returns_fallback_if_undefined():
    from pypara.currencies import Currencies
    noneprice = Price.of(None, Decimal('1'), None)
    fallback = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert noneprice.or_else(lambda: fallback) is fallback


# LLM-generated content at query #36
#--------------------------

```python
def test_as_integer_returns_integer_for_defined_money():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    money = Money.of(Currencies["USD"], Decimal("42.99"), date(2020, 1, 1))
    assert money.as_integer() == 42

def test_as_integer_raises_exception_for_undefined_money():
    from pypara.currencies import Currencies
    from decimal import Decimal
    money = Money.of(None, None, None)
    try:
        money.as_integer()
        assert False, "Expected MonetaryOperationException"
    except MonetaryOperationException:
        pass


# LLM-generated content at query #37
#--------------------------

```python
def test_qty_or_returns_quantity_when_money_is_defined():
    from pypara.currencies import Currencies
    somemoney = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert somemoney.qty_or(Decimal('0')) == Decimal('1.00')

def test_qty_or_returns_default_when_money_is_undefined():
    nonemoney = Money.of(None, Decimal('1'), None)
    assert nonemoney.qty_or(Decimal('0')) == Decimal('0')


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```
def test_neg_positive_quantity():
    ccy = Currency("USD", 2)
    qty = Decimal("100.00")
    dov = Date(2023, 1, 1)
    money = SomeMoney(ccy, qty, dov)
    result = money.__neg__()
    assert result.ccy == ccy
    assert result.qty == Decimal("-100.00")
    assert result.dov == dov

def test_neg_negative_quantity():
    ccy = Currency("USD", 2)
    qty = Decimal("-50.25")
    dov = Date(2023, 1, 1)
    money = SomeMoney(ccy, qty, dov)
    result = money.__neg__()
    assert result.ccy == ccy
    assert result.qty == Decimal("50.25")
    assert result.dov == dov

def test_neg_zero_quantity():
    ccy = Currency("USD", 2)
    qty = Decimal("0.00")
    dov = Date(2023, 1, 1)
    money = SomeMoney(ccy, qty, dov)
    result = money.__neg__()
    assert result.ccy == ccy
    assert result.qty == Decimal("0.00")
    assert result.dov == dov


# LLM-generated content at query #2
#--------------------------

```
def test_qty_map_defined_price():
    from pypara.currencies import Currencies
    from datetime import date
    someprice = Price.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    result = someprice.qty_map(lambda x: x + Decimal('1'), lambda: Decimal('42'))
    assert result == Decimal('2')

def test_qty_map_undefined_price():
    from pypara.currencies import Currencies
    from datetime import date
    noneprice = Price.of(None, Decimal('1'), None)
    result = noneprice.qty_map(lambda x: x + Decimal('1'), lambda: Decimal('42'))
    assert result == Decimal('42')


# LLM-generated content at query #3
#--------------------------

```python
def test_qty_or_zero_returns_zero_for_undefined_money():
    undefined_money = Money.na()
    assert undefined_money.qty_or_zero() == Decimal('0')

def test_qty_or_zero_returns_qty_for_defined_money():
    from pypara.currencies import Currencies
    defined_money = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert defined_money.qty_or_zero() == Decimal('1.00')


# LLM-generated content at query #4
#--------------------------

```python
def test_with_qty_updates_quantity_correctly():
    ccy = Currency("USD", 2)
    qty = Decimal("100.00")
    dov = Date(2023, 10, 1)
    money = SomeMoney(ccy, qty, dov)
    new_qty = Decimal("200.00")
    updated_money = money.with_qty(new_qty)
    assert updated_money.qty == new_qty

def test_with_qty_preserves_currency_and_date():
    ccy = Currency("USD", 2)
    qty = Decimal("100.00")
    dov = Date(2023, 10, 1)
    money = SomeMoney(ccy, qty, dov)
    new_qty = Decimal("200.00")
    updated_money = money.with_qty(new_qty)
    assert updated_money.ccy == ccy
    assert updated_money.dov == dov

def test_with_qty_quantizes_quantity_correctly():
    ccy = Currency("USD", 2)
    qty = Decimal("100.00")
    dov = Date(2023, 10, 1)
    money = SomeMoney(ccy, qty, dov)
    new_qty = Decimal("200.555")
    updated_money = money.with_qty(new_qty)
    assert updated_money.qty == Decimal("200.56")


# LLM-generated content at query #5
#--------------------------

```python
def test_fmap():
    ccy = Currency("USD", 2, Decimal("0.01"))
    money = SomeMoney(ccy, Decimal("100.00"), Date(2023, 1, 1))
    result = money.fmap(lambda x: SomeMoney(x.ccy, x.qty * 2, x.dov))
    assert result.ccy == ccy
    assert result.qty == Decimal("200.00")
    assert result.dov == Date(2023, 1, 1)

def test_fmap_returns_self_when_function_returns_none():
    ccy = Currency("USD", 2, Decimal("0.01"))
    money = SomeMoney(ccy, Decimal("100.00"), Date(2023, 1, 1))
    result = money.fmap(lambda x: None)
    assert result == money

def test_fmap_returns_self_when_function_raises_exception():
    ccy = Currency("USD", 2, Decimal("0.01"))
    money = SomeMoney(ccy, Decimal("100.00"), Date(2023, 1, 1))
    result = money.fmap(lambda x: 1/0)
    assert result == money


# LLM-generated content at query #6
#--------------------------

```
def test_dov_or_returns_dov_when_default_provided():
    ccy = Currency("USD", 2)
    qty = Decimal("100.00")
    dov = Date(2023, 1, 1)
    money = SomeMoney(ccy, qty, dov)
    default_dov = Date(2022, 12, 31)
    assert money.dov_or(default_dov) == dov

def test_dov_or_returns_default_when_dov_is_none():
    ccy = Currency("USD", 2)
    qty = Decimal("100.00")
    dov = None
    money = SomeMoney(ccy, qty, dov)
    default_dov = Date(2022, 12, 31)
    assert money.dov_or(default_dov) == default_dov


# LLM-generated content at query #7
#--------------------------

```python
def test___bool__():
    from pypara.currencies import Currencies
    from pypara.money import Money, NoMoney
    from pypara.temporal import Date

    money_defined = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    money_undefined = Money.na()

    assert bool(money_defined) is True
    assert bool(money_undefined) is False


# LLM-generated content at query #8
#--------------------------

```python
def test_abs_defined_price():
    price = Price.of(Currencies["USD"], Decimal('10.5'), Date(2023, 1, 1))
    result = price.abs()
    assert result.qty == Decimal('10.5')

def test_abs_negative_defined_price():
    price = Price.of(Currencies["USD"], Decimal('-10.5'), Date(2023, 1, 1))
    result = price.abs()
    assert result.qty == Decimal('10.5')

def test_abs_undefined_price():
    price = Price.na()
    result = price.abs()
    assert result.undefined


# LLM-generated content at query #9
#--------------------------

```python
def test_lte_with_defined_prices():
    from pypara.currencies import Currencies
    from pypara.monetary import Price
    price1 = Price.of(Currencies["USD"], Decimal('10'), Date(2023, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('20'), Date(2023, 1, 1))
    assert price1.lte(price2) == True

def test_lte_with_undefined_price():
    from pypara.monetary import Price
    price1 = Price.na()
    price2 = Price.of(Currencies["USD"], Decimal('20'), Date(2023, 1, 1))
    assert price1.lte(price2) == True

def test_lte_with_incompatible_currencies():
    from pypara.currencies import Currencies
    from pypara.monetary import Price
    price1 = Price.of(Currencies["USD"], Decimal('10'), Date(2023, 1, 1))
    price2 = Price.of(Currencies["EUR"], Decimal('10'), Date(2023, 1, 1))
    try:
        price1.lte(price2)
    except IncompatibleCurrencyError:
        pass
    else:
        assert False, "IncompatibleCurrencyError not raised"

def test_lte_with_equal_prices():
    from pypara.currencies import Currencies
    from pypara.monetary import Price
    price1 = Price.of(Currencies["USD"], Decimal('10'), Date(2023, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('10'), Date(2023, 1, 1))
    assert price1.lte(price2) == True

def test_lte_with_greater_price():
    from pypara.currencies import Currencies
    from pypara.monetary import Price
    price1 = Price.of(Currencies["USD"], Decimal('30'), Date(2023, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('20'), Date(2023, 1, 1))
    assert price1.lte(price2) == False


# LLM-generated content at query #10
#--------------------------

```python
def test_is_equal_with_defined_prices():
    ccy = Currencies["USD"]
    qty = Decimal('1')
    dov = Date(2019, 1, 1)
    price1 = Price.of(ccy, qty, dov)
    price2 = Price.of(ccy, qty, dov)
    assert price1.is_equal(price2)

def test_is_equal_with_different_defined_prices():
    ccy1 = Currencies["USD"]
    ccy2 = Currencies["EUR"]
    qty1 = Decimal('1')
    qty2 = Decimal('2')
    dov1 = Date(2019, 1, 1)
    dov2 = Date(2019, 1, 2)
    price1 = Price.of(ccy1, qty1, dov1)
    price2 = Price.of(ccy2, qty2, dov2)
    assert not price1.is_equal(price2)

def test_is_equal_with_undefined_prices():
    price1 = Price.na()
    price2 = Price.na()
    assert price1.is_equal(price2)

def test_is_equal_with_defined_and_undefined_prices():
    ccy = Currencies["USD"]
    qty = Decimal('1')
    dov = Date(2019, 1, 1)
    price1 = Price.of(ccy, qty, dov)
    price2 = Price.na()
    assert not price1.is_equal(price2)


# LLM-generated content at query #11
#--------------------------

```python
def test_divide_defined_price():
    price = Price.of(Currencies["USD"], Decimal('10'), Date(2023, 1, 1))
    result = price.divide(Decimal('2'))
    assert result.qty == Decimal('5')
    assert result.ccy.code == 'USD'
    assert result.dov == Date(2023, 1, 1)

def test_divide_by_zero():
    price = Price.of(Currencies["USD"], Decimal('10'), Date(2023, 1, 1))
    result = price.divide(Decimal('0'))
    assert result.undefined

def test_divide_undefined_price():
    price = Price.na()
    result = price.divide(Decimal('2'))
    assert result.undefined


# LLM-generated content at query #12
#--------------------------

```python
def test___add__():
    ccy1 = Currency("USD")
    ccy2 = Currency("EUR")
    qty1 = Decimal("100.00")
    qty2 = Decimal("200.00")
    dov1 = Date(2023, 10, 1)
    dov2 = Date(2023, 10, 2)
    price1 = SomePrice(ccy1, qty1, dov1)
    price2 = SomePrice(ccy1, qty2, dov2)
    result = price1 + price2
    assert result.ccy == ccy1
    assert result.qty == qty1 + qty2
    assert result.dov == dov2

def test___add__incompatible_currency():
    ccy1 = Currency("USD")
    ccy2 = Currency("EUR")
    qty1 = Decimal("100.00")
    qty2 = Decimal("200.00")
    dov1 = Date(2023, 10, 1)
    dov2 = Date(2023, 10, 2)
    price1 = SomePrice(ccy1, qty1, dov1)
    price2 = SomePrice(ccy2, qty2, dov2)
    try:
        price1 + price2
        assert False, "Expected IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass


# LLM-generated content at query #13
#--------------------------

```python
def test___add___with_undefined_price():
    price1 = SomePrice(Currency("USD"), Decimal("100.00"), Date("2023-01-01"))
    price2 = NoPrice
    result = price1 + price2
    assert result == price1

def test___add___with_same_currency():
    price1 = SomePrice(Currency("USD"), Decimal("100.00"), Date("2023-01-01"))
    price2 = SomePrice(Currency("USD"), Decimal("50.00"), Date("2023-01-02"))
    result = price1 + price2
    assert result == SomePrice(Currency("USD"), Decimal("150.00"), Date("2023-01-02"))

def test___add___with_different_currency():
    price1 = SomePrice(Currency("USD"), Decimal("100.00"), Date("2023-01-01"))
    price2 = SomePrice(Currency("EUR"), Decimal("50.00"), Date("2023-01-02"))
    try:
        price1 + price2
        assert False, "Should have raised IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        assert True

def test___add___with_later_date():
    price1 = SomePrice(Currency("USD"), Decimal("100.00"), Date("2023-01-02"))
    price2 = SomePrice(Currency("USD"), Decimal("50.00"), Date("2023-01-01"))
    result = price1 + price2
    assert result == SomePrice(Currency("USD"), Decimal("150.00"), Date("2023-01-02"))


# LLM-generated content at query #14
#--------------------------

```python
def test_qty_map_with_defined_price():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    someprice = Price.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    result = someprice.qty_map(lambda x: x + Decimal('1'), lambda: Decimal('42'))
    assert result == Decimal('2')

def test_qty_map_with_undefined_price():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    noneprice = Price.of(None, Decimal('1'), None)
    result = noneprice.qty_map(lambda x: x + Decimal('1'), lambda: Decimal('42'))
    assert result == Decimal('42')


# LLM-generated content at query #15
#--------------------------

```python
def test_positive_defined_price():
    from pypara.currencies import Currencies
    from datetime import date
    price = Price.of(Currencies["USD"], Decimal('10'), date(2023, 1, 1))
    result = price.positive()
    assert result.defined
    assert result.qty == Decimal('10')
    assert result.ccy == Currencies["USD"]
    assert result.dov == date(2023, 1, 1)

def test_positive_undefined_price():
    undefined_price = Price.na()
    result = undefined_price.positive()
    assert result.undefined
    assert result is undefined_price


# LLM-generated content at query #16
#--------------------------

```
def test_scalar_add_with_defined_money():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    money = Money.of(Currencies["USD"], Decimal('10'), date(2020, 1, 1))
    result = money.scalar_add(Decimal('5'))
    assert result.qty == Decimal('15')
    assert result.ccy == Currencies["USD"]
    assert result.dov == date(2020, 1, 1)

def test_scalar_add_with_undefined_money():
    from pypara.currencies import Currencies
    from decimal import Decimal
    money = Money.na()
    result = money.scalar_add(Decimal('5'))
    assert result.undefined

def test_scalar_add_with_zero():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    money = Money.of(Currencies["USD"], Decimal('10'), date(2020, 1, 1))
    result = money.scalar_add(Decimal('0'))
    assert result.qty == Decimal('10')
    assert result.ccy == Currencies["USD"]
    assert result.dov == date(2020, 1, 1)

def test_scalar_add_with_negative_value():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    money = Money.of(Currencies["USD"], Decimal('10'), date(2020, 1, 1))
    result = money.scalar_add(Decimal('-5'))
    assert result.qty == Decimal('5')
    assert result.ccy == Currencies["USD"]
    assert result.dov == date(2020, 1, 1)


# LLM-generated content at query #17
#--------------------------

```
def test_dov_or_none_with_defined_money():
    from pypara.currencies import Currencies
    from datetime import date
    money = Money.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    assert money.dov_or_none() == date(2019, 1, 1)

def test_dov_or_none_with_undefined_money():
    from pypara.currencies import Currencies
    money = Money.of(None, None, None)
    assert money.dov_or_none() is None

def test_dov_or_none_with_undefined_dov():
    from pypara.currencies import Currencies
    money = Money.of(Currencies["USD"], Decimal('1'), None)
    assert money.dov_or_none() is None


# LLM-generated content at query #18
#--------------------------

```python
def test_with_qty_returns_same_instance_if_price_is_undefined():
    undefined_price = Price.na()
    new_price = undefined_price.with_qty(Decimal('10'))
    assert new_price is undefined_price

def test_with_qty_returns_new_instance_with_updated_qty_if_price_is_defined():
    ccy = Currencies["USD"]
    dov = Date(2019, 1, 1)
    original_price = Price.of(ccy, Decimal('1'), dov)
    new_price = original_price.with_qty(Decimal('10'))
    assert new_price.qty == Decimal('10')
    assert new_price.ccy == ccy
    assert new_price.dov == dov


# LLM-generated content at query #19
#--------------------------

```python
def test_subtract_same_currency():
    ccy = Currency("USD", 2)
    money1 = SomeMoney(ccy, Decimal("100.00"), Date(2023, 10, 1))
    money2 = SomeMoney(ccy, Decimal("50.00"), Date(2023, 10, 1))
    result = money1 - money2
    assert result == SomeMoney(ccy, Decimal("50.00"), Date(2023, 10, 1))

def test_subtract_different_currency():
    ccy1 = Currency("USD", 2)
    ccy2 = Currency("EUR", 2)
    money1 = SomeMoney(ccy1, Decimal("100.00"), Date(2023, 10, 1))
    money2 = SomeMoney(ccy2, Decimal("50.00"), Date(2023, 10, 1))
    with pytest.raises(IncompatibleCurrencyError):
        money1 - money2

def test_subtract_with_undefined_money():
    ccy = Currency("USD", 2)
    money1 = SomeMoney(ccy, Decimal("100.00"), Date(2023, 10, 1))
    money2 = NoMoney
    result = money1 - money2
    assert result == money1

def test_subtract_with_different_dates():
    ccy = Currency("USD", 2)
    money1 = SomeMoney(ccy, Decimal("100.00"), Date(2023, 10, 2))
    money2 = SomeMoney(ccy, Decimal("50.00"), Date(2023, 10, 1))
    result = money1 - money2
    assert result == SomeMoney(ccy, Decimal("50.00"), Date(2023, 10, 2))


# LLM-generated content at query #20
#--------------------------

```
def test_subtract_defined_money_with_same_currency():
    from pypara.currencies import Currencies
    from datetime import date
    m1 = Money.of(Currencies["USD"], Decimal('10'), date(2020, 1, 1))
    m2 = Money.of(Currencies["USD"], Decimal('5'), date(2020, 1, 1))
    result = m1.subtract(m2)
    assert result.qty == Decimal('5')
    assert result.ccy == Currencies["USD"]
    assert result.dov == date(2020, 1, 1)

def test_subtract_defined_money_with_different_currency_raises_error():
    from pypara.currencies import Currencies
    from datetime import date
    m1 = Money.of(Currencies["USD"], Decimal('10'), date(2020, 1, 1))
    m2 = Money.of(Currencies["EUR"], Decimal('5'), date(2020, 1, 1))
    try:
        m1.subtract(m2)
        assert False
    except IncompatibleCurrencyError:
        assert True

def test_subtract_undefined_money_with_defined_money_returns_defined():
    from pypara.currencies import Currencies
    from datetime import date
    m1 = Money.na()
    m2 = Money.of(Currencies["USD"], Decimal('5'), date(2020, 1, 1))
    result = m1.subtract(m2)
    assert result is m2

def test_subtract_defined_money_with_undefined_money_returns_defined():
    from pypara.currencies import Currencies
    from datetime import date
    m1 = Money.of(Currencies["USD"], Decimal('10'), date(2020, 1, 1))
    m2 = Money.na()
    result = m1.subtract(m2)
    assert result is m1

def test_subtract_undefined_money_with_undefined_money_returns_undefined():
    m1 = Money.na()
    m2 = Money.na()
    result = m1.subtract(m2)
    assert result is Money.na()


# LLM-generated content at query #21
#--------------------------

```python
def test_with_dov_updates_date_for_defined_price():
    from pypara.currencies import Currencies
    from datetime import date
    original_price = Price.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    updated_price = original_price.with_dov(date(2020, 1, 1))
    assert updated_price.dov_or_none() == date(2020, 1, 1)

def test_with_dov_returns_self_for_undefined_price():
    from pypara.currencies import Currencies
    from datetime import date
    undefined_price = Price.of(None, Decimal('1'), None)
    updated_price = undefined_price.with_dov(date(2020, 1, 1))
    assert updated_price is undefined_price


# LLM-generated content at query #22
#--------------------------

```python
def test_convert_same_currency():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    price = SomePrice(usd, Decimal("100.00"), Date(2023, 10, 1))
    converted = price.convert(usd, Date(2023, 10, 1))
    assert converted.ccy == usd
    assert converted.qty == Decimal("100.00")
    assert converted.dov == Date(2023, 10, 1)

def test_convert_different_currency():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    price = SomePrice(usd, Decimal("100.00"), Date(2023, 10, 1))
    FXRateService.default = FXRateService()
    FXRateService.default.set_rate(FXRate(usd, eur, Decimal("0.85"), Date(2023, 10, 1)))
    converted = price.convert(eur, Date(2023, 10, 1))
    assert converted.ccy == eur
    assert converted.qty == Decimal("85.00")
    assert converted.dov == Date(2023, 10, 1)

def test_convert_with_strict_set_to_true_and_rate_not_found():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    price = SomePrice(usd, Decimal("100.00"), Date(2023, 10, 1))
    FXRateService.default = FXRateService()
    try:
        price.convert(eur, Date(2023, 10, 1), strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError:
        assert True

def test_convert_with_strict_set_to_false_and_rate_not_found():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    price = SomePrice(usd, Decimal("100.00"), Date(2023, 10, 1))
    FXRateService.default = FXRateService()
    converted = price.convert(eur, Date(2023, 10, 1), strict=False)
    assert converted == NoPrice

def test_convert_with_no_fx_rate_service_set():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    price = SomePrice(usd, Decimal("100.00"), Date(2023, 10, 1))
    FXRateService.default = None
    try:
        price.convert(eur, Date(2023, 10, 1))
        assert False, "Expected ProgrammingError"
    except ProgrammingError:
        assert True


# LLM-generated content at query #23
#--------------------------

```python
def test_positive_method_returns_same_money_for_defined_money():
    money = Money.of(Currencies["USD"], Decimal('10.50'), Date(2023, 1, 1))
    result = money.positive()
    assert result.is_equal(money)

def test_positive_method_returns_itself_for_undefined_money():
    undefined_money = Money.na()
    result = undefined_money.positive()
    assert result.is_equal(undefined_money)


# LLM-generated content at query #24
#--------------------------

```
def test_qty_or_none_returns_qty_when_defined():
    from pypara.currencies import Currencies
    from datetime import date
    price = Price.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    assert price.qty_or_none() == Decimal('1')

def test_qty_or_none_returns_none_when_undefined():
    from pypara.currencies import Currencies
    price = Price.of(None, Decimal('1'), None)
    assert price.qty_or_none() is None


# LLM-generated content at query #25
#--------------------------

```
def test_positive_defined_price():
    from pypara.currencies import Currencies
    from decimal import Decimal
    from datetime import date
    price = Price.of(Currencies["USD"], Decimal('10'), date(2023, 1, 1))
    result = price.__pos__()
    assert result.qty == Decimal('10')
    assert result.ccy == Currencies["USD"]
    assert result.dov == date(2023, 1, 1)

def test_positive_undefined_price():
    price = Price.na()
    result = price.__pos__()
    assert result.undefined


# LLM-generated content at query #26
#--------------------------

```python
def test_le_method_with_same_currency():
    ccy = Currency("USD")
    qty1 = Decimal("100.00")
    qty2 = Decimal("200.00")
    dov = Date(2023, 10, 1)
    price1 = SomePrice(ccy, qty1, dov)
    price2 = SomePrice(ccy, qty2, dov)
    assert price1 <= price2

def test_le_method_with_different_currency():
    ccy1 = Currency("USD")
    ccy2 = Currency("EUR")
    qty1 = Decimal("100.00")
    qty2 = Decimal("200.00")
    dov = Date(2023, 10, 1)
    price1 = SomePrice(ccy1, qty1, dov)
    price2 = SomePrice(ccy2, qty2, dov)
    try:
        price1 <= price2
        assert False
    except IncompatibleCurrencyError:
        assert True

def test_le_method_with_non_price_object():
    ccy = Currency("USD")
    qty = Decimal("100.00")
    dov = Date(2023, 10, 1)
    price = SomePrice(ccy, qty, dov)
    non_price_object = "Not a Price"
    assert not (price <= non_price_object)


# LLM-generated content at query #27
#--------------------------

```python
def test_qty_or_else_with_defined_price():
    from pypara.currencies import Currencies
    from datetime import date
    price = Price.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    assert price.qty_or_else(lambda: Decimal('42')) == Decimal('1')

def test_qty_or_else_with_defined_price_returns_qty():
    from pypara.currencies import Currencies
    from datetime import date
    price = Price.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    assert price.qty_or_else(lambda: True) == Decimal('1')

def test_qty_or_else_with_undefined_price_returns_default():
    price = Price.of(None, Decimal('1'), None)
    assert price.qty_or_else(lambda: Decimal('42')) == Decimal('42')

def test_qty_or_else_with_undefined_price_returns_default_value():
    price = Price.of(None, Decimal('1'), None)
    assert price.qty_or_else(lambda: False) == False


# LLM-generated content at query #28
#--------------------------

```python
def test_subtract_defined_money_objects():
    ccy = Currencies["USD"]
    m1 = Money.of(ccy, Decimal('10'), Date(2023, 1, 1))
    m2 = Money.of(ccy, Decimal('4'), Date(2023, 1, 1))
    result = m1 - m2
    assert result.qty == Decimal('6')
    assert result.ccy == ccy
    assert result.dov == Date(2023, 1, 1)

def test_subtract_undefined_money_object():
    ccy = Currencies["USD"]
    m1 = Money.of(ccy, Decimal('10'), Date(2023, 1, 1))
    m2 = Money.na()
    result = m1 - m2
    assert result == m1

def test_subtract_incompatible_currencies():
    m1 = Money.of(Currencies["USD"], Decimal('10'), Date(2023, 1, 1))
    m2 = Money.of(Currencies["EUR"], Decimal('4'), Date(2023, 1, 1))
    with pytest.raises(IncompatibleCurrencyError):
        m1 - m2

def test_subtract_scalar_from_defined_money():
    ccy = Currencies["USD"]
    m1 = Money.of(ccy, Decimal('10'), Date(2023, 1, 1))
    result = m1 - Decimal('4')
    assert result.qty == Decimal('6')
    assert result.ccy == ccy
    assert result.dov == Date(2023, 1, 1)

def test_subtract_scalar_from_undefined_money():
    m1 = Money.na()
    result = m1 - Decimal('4')
    assert result == m1


# LLM-generated content at query #29
#--------------------------

```python
def test_gte_defined_price_greater_than_undefined():
    defined_price = Price.of(Currencies["USD"], Decimal('10'), Date(2023, 1, 1))
    undefined_price = Price.na()
    assert defined_price.gte(undefined_price)

def test_gte_undefined_price_not_greater_than_defined():
    defined_price = Price.of(Currencies["USD"], Decimal('10'), Date(2023, 1, 1))
    undefined_price = Price.na()
    assert not undefined_price.gte(defined_price)

def test_gte_undefined_price_greater_than_undefined():
    undefined_price1 = Price.na()
    undefined_price2 = Price.na()
    assert undefined_price1.gte(undefined_price2)

def test_gte_defined_price_greater_than_defined_same_currency():
    price1 = Price.of(Currencies["USD"], Decimal('20'), Date(2023, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('10'), Date(2023, 1, 1))
    assert price1.gte(price2)

def test_gte_defined_price_not_greater_than_defined_same_currency():
    price1 = Price.of(Currencies["USD"], Decimal('5'), Date(2023, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('10'), Date(2023, 1, 1))
    assert not price1.gte(price2)

def test_gte_defined_price_equal_to_defined_same_currency():
    price1 = Price.of(Currencies["USD"], Decimal('10'), Date(2023, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('10'), Date(2023, 1, 1))
    assert price1.gte(price2)

def test_gte_raises_incompatible_currency_error():
    price1 = Price.of(Currencies["USD"], Decimal('10'), Date(2023, 1, 1))
    price2 = Price.of(Currencies["EUR"], Decimal('10'), Date(2023, 1, 1))
    try:
        price1.gte(price2)
    except IncompatibleCurrencyError:
        assert True
    else:
        assert False


# LLM-generated content at query #30
#--------------------------

```python
def test_gt_defined_price_greater_than_undefined():
    from pypara.currencies import Currencies
    from pypara.temporal import Date
    defined_price = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    undefined_price = Price.na()
    assert defined_price.gt(undefined_price) is True

def test_gt_undefined_price_not_greater_than_defined():
    from pypara.currencies import Currencies
    from pypara.temporal import Date
    defined_price = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    undefined_price = Price.na()
    assert undefined_price.gt(defined_price) is False

def test_gt_undefined_price_not_greater_than_undefined():
    undefined_price1 = Price.na()
    undefined_price2 = Price.na()
    assert undefined_price1.gt(undefined_price2) is False

def test_gt_defined_price_greater_than_defined_with_same_currency():
    from pypara.currencies import Currencies
    from pypara.temporal import Date
    price1 = Price.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert price1.gt(price2) is True

def test_gt_defined_price_not_greater_than_defined_with_same_currency():
    from pypara.currencies import Currencies
    from pypara.temporal import Date
    price1 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('2'), Date(2019, 1, 1))
    assert price1.gt(price2) is False

def test_gt_defined_price_equal_to_defined_with_same_currency():
    from pypara.currencies import Currencies
    from pypara.temporal import Date
    price1 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert price1.gt(price2) is False


# LLM-generated content at query #31
#--------------------------

```python
def test_qty_map_with_defined_money():
    from pypara.currencies import Currencies
    from datetime import date
    somemoney = Money.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    result = somemoney.qty_map(lambda x: x + Decimal('1'), lambda: Decimal('42'))
    assert result == Decimal('2')

def test_qty_map_with_undefined_money():
    from pypara.currencies import Currencies
    from datetime import date
    nonemoney = Money.of(None, Decimal('1'), None)
    result = nonemoney.qty_map(lambda x: x + Decimal('1'), lambda: Decimal('42'))
    assert result == Decimal('42')


# LLM-generated content at query #32
#--------------------------

```python
def test_constructor_with_valid_arguments():
    ccy = Currency("USD", 2)
    qty = Decimal("100.00")
    dov = Date(2023, 1, 1)
    money = SomeMoney(ccy, qty, dov)
    assert money.ccy == ccy
    assert money.qty == qty
    assert money.dov == dov
    assert money.defined is True
    assert money.undefined is False

def test_constructor_with_zero_quantity():
    ccy = Currency("EUR", 2)
    qty = Decimal("0.00")
    dov = Date(2023, 1, 1)
    money = SomeMoney(ccy, qty, dov)
    assert money.qty == qty
    assert bool(money) is False

def test_constructor_with_negative_quantity():
    ccy = Currency("JPY", 0)
    qty = Decimal("-500")
    dov = Date(2023, 1, 1)
    money = SomeMoney(ccy, qty, dov)
    assert money.qty == qty
    assert bool(money) is True

def test_constructor_with_future_date():
    ccy = Currency("GBP", 2)
    qty = Decimal("50.00")
    dov = Date(2050, 12, 31)
    money = SomeMoney(ccy, qty, dov)
    assert money.dov == dov

def test_constructor_with_minimal_currency_decimals():
    ccy = Currency("BTC", 8)
    qty = Decimal("0.00000001")
    dov = Date(2023, 1, 1)
    money = SomeMoney(ccy, qty, dov)
    assert money.qty == qty


# LLM-generated content at query #33
#--------------------------

```
def test_dov_or_none_with_defined_price():
    from pypara.currencies import Currencies
    from datetime import date
    price = Price.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    assert price.dov_or_none() == date(2019, 1, 1)

def test_dov_or_none_with_undefined_price():
    price = Price.of(None, None, None)
    assert price.dov_or_none() is None

def test_dov_or_none_with_undefined_dov():
    from pypara.currencies import Currencies
    price = Price.of(Currencies["USD"], Decimal('1'), None)
    assert price.dov_or_none() is None


# LLM-generated content at query #34
#--------------------------

```python
def test_multiply_with_defined_price():
    ccy = Currencies["USD"]
    qty = Decimal('10')
    dov = Date(2023, 1, 1)
    price = Price.of(ccy, qty, dov)
    result = price.multiply(Decimal('2'))
    assert result.qty == Decimal('20')
    assert result.ccy == ccy
    assert result.dov == dov

def test_multiply_with_undefined_price():
    price = Price.na()
    result = price.multiply(Decimal('2'))
    assert result.undefined

def test_multiply_with_zero():
    ccy = Currencies["USD"]
    qty = Decimal('10')
    dov = Date(2023, 1, 1)
    price = Price.of(ccy, qty, dov)
    result = price.multiply(Decimal('0'))
    assert result.qty == Decimal('0')
    assert result.ccy == ccy
    assert result.dov == dov

def test_multiply_with_negative_scalar():
    ccy = Currencies["USD"]
    qty = Decimal('10')
    dov = Date(2023, 1, 1)
    price = Price.of(ccy, qty, dov)
    result = price.multiply(Decimal('-3'))
    assert result.qty == Decimal('-30')
    assert result.ccy == ccy
    assert result.dov == dov


# LLM-generated content at query #35
#--------------------------

```python
def test_add_with_same_currency():
    ccy = Currency("USD")
    qty1 = Decimal("100")
    qty2 = Decimal("200")
    dov1 = Date(2023, 10, 1)
    dov2 = Date(2023, 10, 2)
    price1 = SomePrice(ccy, qty1, dov1)
    price2 = SomePrice(ccy, qty2, dov2)
    result = price1 + price2
    assert result.ccy == ccy
    assert result.qty == qty1 + qty2
    assert result.dov == dov2

def test_add_with_different_currency():
    ccy1 = Currency("USD")
    ccy2 = Currency("EUR")
    qty1 = Decimal("100")
    qty2 = Decimal("200")
    dov1 = Date(2023, 10, 1)
    dov2 = Date(2023, 10, 2)
    price1 = SomePrice(ccy1, qty1, dov1)
    price2 = SomePrice(ccy2, qty2, dov2)
    try:
        price1 + price2
        assert False, "Expected IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass

def test_add_with_one_undefined_price():
    ccy = Currency("USD")
    qty = Decimal("100")
    dov = Date(2023, 10, 1)
    price = SomePrice(ccy, qty, dov)
    undefined_price = NoPrice
    result = price + undefined_price
    assert result == price

def test_add_with_both_undefined_prices():
    undefined_price1 = NoPrice
    undefined_price2 = NoPrice
    result = undefined_price1 + undefined_price2
    assert result == NoPrice


# LLM-generated content at query #36
#--------------------------

```python
def test_convert_with_default_fx_rate_service_not_none():
    ccy = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    price = SomePrice(ccy, Decimal("100.00"), Date(2023, 1, 1))
    FXRateService.default = MockFXRateService()
    try:
        price.convert(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))
    except:
        assert False, "No exception should be raised when FXRateService.default is not None"


# LLM-generated content at query #37
#--------------------------

```python
def test_abs_defined_price():
    price = Price.of(Currencies["USD"], Decimal('10.5'), Date(2019, 1, 1))
    result = price.abs()
    assert result.qty == Decimal('10.5')
    assert result.ccy == Currencies["USD"]
    assert result.dov == Date(2019, 1, 1)

def test_abs_negative_price():
    price = Price.of(Currencies["USD"], Decimal('-10.5'), Date(2019, 1, 1))
    result = price.abs()
    assert result.qty == Decimal('10.5')
    assert result.ccy == Currencies["USD"]
    assert result.dov == Date(2019, 1, 1)

def test_abs_zero_price():
    price = Price.of(Currencies["USD"], Decimal('0'), Date(2019, 1, 1))
    result = price.abs()
    assert result.qty == Decimal('0')
    assert result.ccy == Currencies["USD"]
    assert result.dov == Date(2019, 1, 1)

def test_abs_undefined_price():
    price = Price.na()
    result = price.abs()
    assert result == Price.na()


# LLM-generated content at query #38
#--------------------------

```
def test_with_dov_defined_money():
    from pypara.currencies import Currencies
    from datetime import date
    original = Money.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    new_date = date(2020, 1, 1)
    result = original.with_dov(new_date)
    assert result.dov == new_date
    assert result.ccy == original.ccy
    assert result.qty == original.qty

def test_with_dov_undefined_money():
    from datetime import date
    original = Money.na()
    new_date = date(2020, 1, 1)
    result = original.with_dov(new_date)
    assert result is original


# LLM-generated content at query #39
#--------------------------

```python
def test_with_ccy_updates_ccy_for_defined_price():
    price = Price.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    updated_price = price.with_ccy(Currencies["EUR"])
    assert updated_price.ccy_or_none() == Currencies["EUR"]
    assert updated_price.qty_or_none() == Decimal('1')
    assert updated_price.dov_or_none() == Date(2019, 1, 1)

def test_with_ccy_returns_itself_for_undefined_price():
    price = Price.na()
    updated_price = price.with_ccy(Currencies["EUR"])
    assert updated_price == price


# LLM-generated content at query #40
#--------------------------

```python
def test_qty_or_else_returns_qty_when_defined():
    from pypara.currencies import Currencies
    somemoney = Money.of(Currencies["USD"], Decimal('1'), Date(2019, 1, 1))
    assert somemoney.qty_or_else(lambda: Decimal('42')) == Decimal('1.00')

def test_qty_or_else_returns_default_when_undefined():
    from pypara.currencies import Currencies
    nonemoney = Money.of(None, Decimal('1'), None)
    assert nonemoney.qty_or_else(lambda: Decimal('42')) == Decimal('42')

def test_qty_or_else_returns_non_decimal_default_when_undefined():
    from pypara.currencies import Currencies
    nonemoney = Money.of(None, Decimal('1'), None)
    assert nonemoney.qty_or_else(lambda: False) is False


# LLM-generated content at query #41
#--------------------------

```python
def test___add__():
    ccy1 = Currency("USD")
    ccy2 = Currency("EUR")
    qty1 = Decimal("100.00")
    qty2 = Decimal("200.00")
    dov1 = Date(2023, 10, 1)
    dov2 = Date(2023, 10, 2)

    money1 = SomeMoney(ccy1, qty1, dov1)
    money2 = SomeMoney(ccy1, qty2, dov2)
    money3 = SomeMoney(ccy2, qty1, dov1)

    result1 = money1.__add__(money2)
    assert result1.ccy == ccy1
    assert result1.qty == qty1 + qty2
    assert result1.dov == dov2

    try:
        money1.__add__(money3)
        assert False, "Expected IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        pass

    undefined_money = NoMoney
    result2 = money1.__add__(undefined_money)
    assert result2 == money1


# LLM-generated content at query #42
#--------------------------

```python
def test_qty_map_defined_money():
    from pypara.currencies import Currencies
    from datetime import date
    money = Money.of(Currencies["USD"], Decimal('1'), date(2019, 1, 1))
    result = money.qty_map(lambda x: x + Decimal('1'), lambda: Decimal('42'))
    assert result == Decimal('2.00')

def test_qty_map_undefined_money():
    from pypara.currencies import Currencies
    from datetime import date
    money = Money.of(None, Decimal('1'), None)
    result = money.qty_map(lambda x: x + Decimal('1'), lambda: Decimal('42'))
    assert result == Decimal('42')


# LLM-generated content at query #43
#--------------------------

```python
def test_times_returns_undefined_price_when_price_is_undefined():
    undefined_price = Price.na()
    result = undefined_price.times(Decimal("10"))
    assert result.undefined

def test_times_returns_money_with_multiplied_quantity():
    from pypara.currencies import Currencies
    price = Price.of(Currencies["USD"], Decimal("10"), Date(2020, 1, 1))
    result = price.times(Decimal("2"))
    assert result.qty == Decimal("20")
    assert result.ccy == Currencies["USD"]
    assert result.dov == Date(2020, 1, 1)

def test_times_returns_money_with_negative_quantity_when_multiplier_is_negative():
    from pypara.currencies import Currencies
    price = Price.of(Currencies["USD"], Decimal("10"), Date(2020, 1, 1))
    result = price.times(Decimal("-1"))
    assert result.qty == Decimal("-10")
    assert result.ccy == Currencies["USD"]
    assert result.dov == Date(2020, 1, 1)

def test_times_returns_zero_money_when_multiplied_by_zero():
    from pypara.currencies import Currencies
    price = Price.of(Currencies["USD"], Decimal("10"), Date(2020, 1, 1))
    result = price.times(Decimal("0"))
    assert result.qty == Decimal("0")
    assert result.ccy == Currencies["USD"]
    assert result.dov == Date(2020, 1, 1)


# LLM-generated content at query #44
#--------------------------

```python
def test_subtraction_with_defined_prices():
    price1 = Price.of(Currencies["USD"], Decimal("10"), Date(2023, 1, 1))
    price2 = Price.of(Currencies["USD"], Decimal("4"), Date(2023, 1, 1))
    result = price1 - price2
    assert result.qty == Decimal("6")
    assert result.ccy == Currencies["USD"]
    assert result.dov == Date(2023, 1, 1)

def test_subtraction_with_undefined_price():
    price1 = Price.of(Currencies["USD"], Decimal("10"), Date(2023, 1, 1))
    price2 = Price.na()
    result = price1 - price2
    assert result.qty == Decimal("10")
    assert result.ccy == Currencies["USD"]
    assert result.dov == Date(2023, 1, 1)

def test_subtraction_with_incompatible_currencies():
    price1 = Price.of(Currencies["USD"], Decimal("10"), Date(2023, 1, 1))
    price2 = Price.of(Currencies["EUR"], Decimal("4"), Date(2023, 1, 1))
    try:
        price1 - price2
        assert False, "Expected IncompatibleCurrencyError"
    except IncompatibleCurrencyError:
        assert True

def test_subtraction_with_undefined_price_as_first_operand():
    price1 = Price.na()
    price2 = Price.of(Currencies["USD"], Decimal("4"), Date(2023, 1, 1))
    result = price1 - price2
    assert result.qty == Decimal("-4")
    assert result.ccy == Currencies["USD"]
    assert result.dov == Date(2023, 1, 1)

def test_subtraction_with_both_undefined_prices():
    price1 = Price.na()
    price2 = Price.na()
    result = price1 - price2
    assert result.undefined


# LLM-generated content at query #45
#--------------------------

```python
def test_as_integer_with_defined_money():
    money = Money.of(Currencies["USD"], Decimal('10.5'), Date(2019, 1, 1))
    assert money.as_integer() == 10

def test_as_integer_with_undefined_money():
    money = Money.na()
    try:
        money.as_integer()
        assert False, "Expected MonetaryOperationException"
    except MonetaryOperationException:
        assert True


