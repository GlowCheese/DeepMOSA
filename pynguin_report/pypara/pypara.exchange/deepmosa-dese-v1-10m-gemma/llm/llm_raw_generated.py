####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import datetime
from decimal import Decimal
from pypara.currencies import Currencies

def test_fxrate_invert_basic():
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("2")
    nrate = FXRate(ccy1, ccy2, date, value)
    rrate = ~nrate
    assert rrate.ccy1 == ccy2
    assert rrate.ccy2 == ccy1
    assert rrate.date == date
    assert rrate.value == Decimal("0.5")

def test_fxrate_invert_identity():
    ccy = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("1")
    rate = FXRate(ccy, ccy, date, value)
    inverted_rate = ~rate
    assert inverted_rate == rate

def test_fxrate_invert_double_inversion():
    ccy1 = Currencies["GBP"]
    ccy2 = Currencies["JPY"]
    date = datetime.date(2023, 1, 1)
    value = Decimal("150.5")
    nrate = FXRate(ccy1, ccy2, date, value)
    assert ~~nrate == nrate
```


# LLM-generated content at query #2
#--------------------------

```python
import datetime
from decimal import Decimal
from typing import NamedTuple

# Mocking the dependencies required for the test to run in isolation
class Currency(str):
    pass

EUR = Currency("EUR")
USD = Currency("USD")
ZERO = Decimal("0")
ONE = Decimal("1")
Date = datetime.date

def test_invert_returns_swapped_currencies_and_reciprocal_value():
    rate = FXRate(EUR, USD, datetime.date(2023, 1, 1), Decimal("2"))
    inverted_rate = ~rate
    assert inverted_rate.ccy1 == USD
    assert inverted_rate.ccy2 == EUR
    assert inverted_rate.date == datetime.date(2023, 1, 1)
    assert inverted_rate.value == Decimal("0.5")

def test_invert_is_idempotent_double_inversion():
    rate = FXRate(EUR, USD, datetime.date(2023, 1, 1), Decimal("4"))
    inverted_rate = ~rate
    double_inverted_rate = ~inverted_rate
    assert double_inverted_rate == rate

def test_invert_with_one_remains_same_structure():
    rate = FXRate(EUR, USD, datetime.date(2023, 1, 1), Decimal("1"))
    inverted_rate = ~rate
    assert inverted_rate.ccy1 == USD
    assert inverted_rate.ccy2 == EUR
    assert inverted_rate.value == Decimal("1")
```


# LLM-generated content at query #3
#--------------------------

```python
from unittest.mock import MagicMock
from datetime import date
from decimal import Decimal

def test_queries_returns_correct_rates():
    service = MagicMock(spec=FXRateService)
    query_list = [
        ("USD", "EUR", date(2023, 1, 1)),
        ("GBP", "USD", date(2023, 1, 2))
    ]
    expected_rates = [Decimal("0.95"), Decimal("1.25")]
    service.queries.return_value = iter(expected_rates)

    results = list(service.queries(query_list))

    assert results == expected_rates
    service.queries.assert_called_once_with(query_list, strict=False)

def test_queries_with_strict_true():
    service = MagicMock(spec=FXRateService)
    query_list = [("USD", "JPY", date(2023, 1, 1))]
    service.queries.return_value = iter([Decimal("140.0")])

    results = list(service.queries(query_list, strict=True))

    assert results == [Decimal("140.0")]
    service.queries.assert_called_once_with(query_list, strict=True)

def test_queries_returns_none_for_missing_rates():
    service = MagicMock(spec=FXRateService)
    query_list = [("USD", "XYZ", date(2023, 1, 1))]
    service.queries.return_value = iter([None])

    results = list(service.queries(query_list))

    assert results == [None]
    service.queries.assert_called_once()
```


# LLM-generated content at query #4
#--------------------------

```python
from unittest.mock import MagicMock
from decimal import Decimal
from datetime import date

def test_queries_returns_correct_rates():
    service = MagicMock(spec=FXRateService)
    query_list = [("USD", "EUR", date(2023, 1, 1)), ("GBP", "JPY", date(2023, 1, 1))]
    expected_rates = [Decimal("0.92"), Decimal("160.50")]
    service.queries.return_value = iter(expected_rates)

    result = list(service.queries(query_list))

    assert result == expected_rates
    service.queries.assert_called_once_with(query_list, strict=False)

def test_queries_with_strict_mode_true():
    service = MagicMock(spec=FXRateService)
    query_list = [("USD", "CAD", date(2023, 5, 5))]
    service.queries.return_value = iter([Decimal("1.35")])

    result = list(service.queries(query_list, strict=True))

    assert result == [Decimal("1.35")]
    service.queries.assert_called_once_with(query_list, strict=True)

def test_queries_returns_none_for_missing_rate():
    service = MagicMock(spec=FXRateService)
    query_list = [("XYZ", "ABC", date(2000, 1, 1))]
    service.queries.return_value = iter([None])

    result = list(service.queries(query_list))

    assert result == [None]
    service.queries.assert_called_once()
```


# LLM-generated content at query #5
#--------------------------

```python
from decimal import Decimal
from datetime import date
from typing import Optional, Tuple
from unittest.mock import MagicMock

def test_query_returns_rate_when_found():
    service = MagicMock(spec=FXRateService)
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)
    expected_rate = MagicMock(spec=FXRate)
    service.query.return_value = expected_rate
    
    result = service.query(ccy1, ccy2, asof_date)
    
    assert result == expected_rate
    service.query.assert_called_once_with(ccy1, ccy2, asof_date)

def test_query_returns_none_when_not_found():
    service = MagicMock(spec=FXRateService)
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = MagicCV="JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)
    service.query.return_value = None
    
    result = service.query(ccy1, ccy2, asof_date)
    
    assert result is None

def test_query_respects_strict_parameter():
    service = MagicMock(spec=FXRateService)
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)
    
    service.query(ccy1, ccy2, asof_date, strict=True)
    service.query.assert_called_with(ccy1, ccy2, asof_date, strict=True)
    
    service.query(ccy1, ccy2, asof_date, strict=False)
    service.query.assert_called_with(ccy1, ccy2, asof_date, strict=False)
```


# LLM-generated content at query #6
#--------------------------

```python
from unittest.mock import MagicMock
from decimal import Decimal
from datetime import date

def test_queries_returns_correct_rates():
    service = MagicMock(spec=FXRateService)
    query_list = [
        ('USD', 'EUR', date(2023, 1, 1)),
        ('GBP', 'JPY', date(2023, 1, 2))
    ]
    expected_rates = [Decimal('0.92'), Decimal('160.50')]
    service.queries.return_value = iter(expected_rates)

    results = list(service.queries(query_list))

    assert results == expected_rates
    service.queries.assert_called_once_with(query_list, strict=False)

def test_queries_handles_none_values():
    service = MagicMock(spec=FXRateService)
    query_list = [('USD', 'CAD', date(2023, 1, 1))]
    expected_rates = [None]
    service.queries.return_value = iter(expected_rates)

    results = list(service.queries(query_list))

    assert results == [None]
    service.queries.assert_called_once_with(query_list, strict=False)

def test_queries_with_strict_mode():
    service = MagicMock(spec=FXRateService)
    query_list = [('USD', 'AUD', date(2023, 1, 1))]
    service.queries.return_value = iter([Decimal('1.50')])

    list(service.queries(query_list, strict=True))

    service.queries.assert_called_with(query_list, strict=True)
```


# LLM-generated content at query #7
#--------------------------

```python
from decimal import Decimal
from datetime import date
from typing import Optional, Tuple
from unittest.mock import MagicMock
from pypara.currencies import Currency, CurrencyType
from pypara.exchange import FXRateService

def test_query_returns_expected_fxrate_when_found():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)
    mock_rate = MagicMock()
    service = MagicMock(spec=FXRateService)
    service.query.return_value = mock_rate

    result = service.query(ccy1, ccy2, asof_date)

    assert result == mock_rate
    service.query.assert_called_once_with(ccy1, ccy2, asof_date, strict=False)

def test_query_returns_none_when_not_found():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)
    service = MagicMock(spec=FXRateService)
    service.query.return_value = None

    result = service.query(ccy1, ccy2, asof_date)

    assert result is None
    service.query.assert_called_once_with(ccy1, ccy2, asof_date, strict=False)

def test_query_respects_strict_parameter():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)
    service = MagicMock(spec=FXRateService)

    service.query(ccy1, ccy2, asof_date, strict=True)
    service.query.assert_called_with(ccy1, ccy2, asof_date, strict=True)
```


# LLM-generated content at query #8
#--------------------------

```python
from decimal import Decimal
from datetime import date
from typing import Optional, Tuple
from unittest.mock import MagicMock

def test_query_returns_rate_when_found():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = date(2023, 1, 1)
    expected_rate = FXRate(ccy1, ccy2, asof, Decimal("0.95"))
    
    service = MagicMock(spec=FXRateService)
    service.query.return_value = expected_rate
    
    result = service.query(ccy1, ccy2, asof)
    
    assert result == expected_rate
    service.query.assert_called_once_with(ccy1, ccy2, asof)

def test_query_returns_none_when_not_found():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    asof = date(2023, 1, 1)
    
    service = MagicMock(spec=FXRateService)
    service.query.return_value = None
    
    result = service.query(ccy1, ccy2, asof)
    
    assert result is None
    service.query.assert_called_once_with(ccy1, ccy2, asof)

def test_query_respects_strict_parameter():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY)
    asof = date(2023, 5, 20)
    
    service = MagicMock(spec=FXRateService)
    
    service.query(ccy1, ccy2, asof, strict=True)
    service.query.assert_called_with(ccy1, ccy2, asof, strict=True)
    
    service.query(ccy1, ccy2, asof, strict=False)
    service.query.assert_called_with(ccy1, ccy2, asof, strict=False)
```


# LLM-generated content at query #9
#--------------------------

```python
from unittest.mock import MagicMock
from decimal import Decimal
from datetime import date

def test_queries_returns_correct_rates():
    service = MagicMock(spec=FXRateService)
    query_list = [
        ("USD", "EUR", date(2023, 1, 1)),
        ("GBP", "JPY", date(2023, 1, 2))
    ]
    expected_rates = [Decimal("0.92"), Decimal("165.50")]
    service.queries.return_value = iter(expected_rates)

    results = list(service.queries(query_list))

    assert results == expected_rates
    service.queries.assert_called_once_with(query_list, strict=False)

def test_queries_with_strict_mode_raises_error():
    service = MagicMock(spec=FXRateService)
    query_list = [("USD", "CAD", date(202ss, 1, 1))]
    service.queries.side_effect = ValueError("Rate not found")

    with Exception as e:
        list(service.queries(query_list, strict=True))
        raise e

    assert str(e) == "Rate not found"
    service.queries.assert_called_once_with(query_list, strict=True)

def test_queries_handles_none_values():
    service = MagicMock(spec=FXRateService)
    query_list = [("USD", "XYZ", date(2023, 1, 1))]
    expected_rates = [None]
    service.queries.return_value = iter(expected_rates)

    results = list(service.queries(query_list))

    assert results == [None]
    service.queries.assert_called_once_with(query_list, strict=False)
```


# LLM-generated content at query #10
#--------------------------

```python
def test_fxrate_constructor_valid_data():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    today = datetime.date.today()
    val = Decimal("1.2")
    rate = FXRate(ccy1, ccy2, today, val)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == today
    assert rate.value == val

def test_fxrate_constructor_tuple_access():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    today = datetime.date.today()
    val = Decimal("1.2")
    rate = FXRate(ccy1, ccy2, today, val)
    assert rate[0] == ccy1
    assert rate[1] == ccy2
    assert rate[2] == today
    assert rate[3] == val

def test_fxrate_constructor_unvalidated_same_currency():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy = Currencies["EUR"]
    today = datetime.date.today()
    val = Decimal("1.5")
    rate = FXRate(ccy, ccy, today, val)
    assert rate.ccy1 == rate.ccy2
    assert rate.value == val
```


# LLM-generated content at query #11
#--------------------------

```python
from decimal import Decimal
from datetime import date
from typing import Optional, Tuple
from unittest.mock import MagicMock

def test_fxrateservice_query_returns_none_when_no_rate_found():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)
    service = MagicMock(spec=FXRateService)
    service.query.return_value = None
    
    result = service.query(cccy1, ccy2, asof_date)
    
    assert result is None
    service.query.assert_called_once_with(cccy1, ccy2, asof_date, False)

def test_fxrateservice_query_returns_rate_when_exists():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)
    expected_rate = MagicMock(spec=FXRate)
    service = MagicMock(spec=FXRateService)
    service.query.return_call = expected_rate
    service.query.return_value = expected_rate
    
    result = service.query(cccy1, ccy2, asof_date)
    
    assert result == expected_rate
    service.query.assert_called_once_with(cccy1, ccy2, asof_date, False)

def test_fxrateservice_query_respects_strict_parameter():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)
    service = MagicMock(spec=FXRateService)
    
    service.query(cccy1, ccy2, asof_date, strict=True)
    
    service.query.assert_called_with(cccy1, ccy2, asof_date, True)
```


# LLM-generated content at query #12
#--------------------------

```python
def test_fxrate_constructor_valid_input():
    from datetime import date
    from decimal import Decimal
    # Assuming Currency is a type/class available in context or using dummy objects if needed, 
    # but since we are testing the constructor directly:
    ccy1 = "EUR" 
    ccy2 = "USD"
    dt = date(2023, 1, 1)
    val = Decimal("1.2")
    rate = FXRate(ccy1, ccy2, dt, val)
    assert rate.ccy1 == "EUR"
    assert rate.ccy2 == "USD"
    assert rate.date == date(2023, 1, 1)
    assert rate.value == Decimal("1.2")

def test_fxrate_constructor_tuple_access():
    from datetime import date
    from decimal import Decimal
    ccy1 = "EUR"
    ccy2 = "USD"
    dt = date(2023, 1, 1)
    val = Decimal("1.5")
    rate = FXRate(ccy1, ccy2, dt, val)
    assert rate[0] == "EUR"
    assert rate[1] == "USD"
    assert rate[2] == date(2023, 1, 1)
    assert rate[3] == Decimal("1.5")

def test_fxrate_constructor_untyped_assignment():
    from datetime import date
    from decimal import Decimal
    # The constructor is explicitly noted as not safe and does not check input
    rate = FXRate(None, 123, "not-a-date", [0])
    assert rate.ccy1 is None
    assert rate.ccy2 == 123
    assert rate.date == "not-a-date"
    assert rate.value == [0]
```


# LLM-generated content at query #13
#--------------------------

```python
def test_fxrate_constructor_valid_input():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    date_val = datetime.date.today()
    value_val = Decimal("1.25")
    ccy1 = Currencies["USD"]
    ccy2 = Currencies["EUR"]
    rate = FXRate(ccy1, ccy2, date_val, value_val)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date_val
    assert rate.value == value_val
    assert rate[0] == ccy1
    assert rate[1] == ccy2
    assert rate[2] == date_val
    assert rate[3] == value_val

def test_fxrate_constructor_tuple_unpacking():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    date_val = datetime.date.today()
    value_val = Decimal("0.8")
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["GBP"]
    rate = FXRate(ccy1, ccy2, date_val, value_val)
    unpacked = list(rate)
    assert len(unpacked) == 4
    assert unpacked[0] == ccy1
    assert unpacked[3] == value_val
```


# LLM-generated content at query #14
#--------------------------

```python
def test_FXRate_constructor_valid():
    from decimal import Decimal
    import datetime
    from pypara.currencies import Currencies
    date_val = datetime.date.today()
    value_val = Decimal("1.5")
    rate = FXRate(Currencies["EUR"], Currencies["USD"], date_val, value_val)
    assert rate.ccy1 == Currencies["EUR"]
    assert rate.ccy2 == Currencies["USD"]
    assert rate.date == date_val
    assert rate.value == value_val

def test_FXRate_constructor_tuple_access():
    from decimal import Decimal
    import datetime
    from pypara.currencies import Currencies
    date_val = datetime.date.today()
    value_val = Decimal("2.0")
    rate = FXRate(Currencies["EUR"], Currencies["USD"], date_val, value_val)
    assert rate[0] == Currencies["EUR"]
    assert rate[1] == Currencies["USD"]
    assert rate[2] == date_val
    assert rate[3] == value_val

def test_FXRate_invert_logic():
    from decimal import Decimal
    import datetime
    from pypara.currencies import Currencies
    date_val = datetime.date.today()
    rate = FXRate(Currencies["EUR"], Currencies["USD"], date_val, Decimal("2"))
    inverted = ~rate
    assert inverted.ccy1 == Currencies["USD"]
    assert inverted.ccy2 == Currencies["EUR"]
    assert inverted.date == date_val
    assert inverted.value == Decimal("0.5")

def test_FXRate_of_valid():
    from decimal import Decimal
    import datetime
    from pypara.currencies import Currencies
    date_val = datetime.date.today()
    value_val = Decimal("1.25")
    rate = FXRate.of(Currencies["GBP"], Currencies["JPY"], date_val, value_val)
    assert rate.ccy1 == Currencies["GBP"]
    assert rate.ccy2 == Currencies["JPY"]
    assert rate.value == value_val

def test_FXRate_of_invalid_value_zero():
    from decimal import Decimal
    import datetime
    from pypara.currencies import Currencies
    date_val = datetime.date.today()
    with pytest.raises(ValueError, match="FX rate value can not be equal to or less than `zero`."):
        FXRate.of(Currencies["EUR"], Currencies["USD"], date_val, Decimal("0"))

def test_FXRate_of_invalid_same_currency():
    from decimal import Decimal
    import datetime
    from pypara.currencies import Currencies
    date_val = datetime.date.today()
    with pytest.raises(ValueError, match="FX rate to the same currency must be `one`."):
        FXRate.of(Currencies["EUR"], Currencies["USD"], date_val, Decimal("1.1"))

def test_FXRate_of_valid_identity():
    from decimal import Decimal
    import datetime
    from pypara.currencies import Currencies
    date_val = datetime.date.today()
    rate = FXRate.of(Currencies["EUR"], Currencies["EUR"], date_val, Decimal("1"))
    assert rate.ccy1 == rate.ccy2
    assert rate.value == Decimal("1")
```


# LLM-generated content at query #15
#--------------------------

```python
def test_fxrate_constructor_valid_input():
    import datetime
    from decimal import Decimal
    # Assuming Currency is a type that can be instantiated or mocked
    # Using a mock/stub approach as per the provided docstring examples
    class MockCurrency:
        def __init__(self, code): self.code = code
        def __eq__(self, other): return self.code == other.code

    ccy1 = MockCurrency("EUR")
    ccy2 = MockCurrency("USD")
    date_val = datetime.date.today()
    value_val = Decimal("2")
    
    rate = FXRate(ccy1, ccy2, date_val, value_val)
    
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date_val
    assert rate.value == value_val
    assert rate[0] == ccy1
    assert rate[1] == ccy2
    assert rate[2] == date_val
    assert rate[3] == value_val

def test_fxrate_constructor_tuple_unpacking():
    import datetime
    from decimal import Decimal
    class MockCurrency:
        def __init__(self, code): self.code = code
        def __eq__(self, other): return self.code == other.code

    ccy1 = MockCurrency("EUR")
    ccy2 = MockCurrency("USD")
    date_val = datetime.date.today()
    value_val = Decimal("1.5")
    
    rate = FXRate(ccy1, ccy2, date_val, value_val)
    c1, c2, d, v = rate
    
    assert c1 == ccy1
    assert c2 == ccy2
    assert d == date_val
    assert v == value_val
```


# LLM-generated content at query #16
#--------------------------

```python
def test_fxrate_constructor_valid_data():
    import datetime
    from decimal import Decimal
    from typing import NamedTuple
    # Mocking Currency and Date types as they are not provided in context but required for the constructor call
    class Currency(str): pass
    Date = datetime.date
    
    ccy1 = Currency("EUR")
    ccy2 = Currency("USD")
    today = datetime.date.today()
    value = Decimal("1.25")
    
    rate = FXRate(ccy1, ccy2, today, value)
    
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == today
    assert rate.value == value
    assert rate[0] == ccy1
    assert rate[1] == ccy2
    assert rate[2] == today
    assert rate[3] == value

def test_fxrate_constructor_tuple_unpacking():
    import datetime
    from decimal import Decimal
    class Currency(str): pass
    Date = datetime.date
    
    ccy1 = Currency("GBP")
    ccy2 = Currency("JPY")
    today = datetime.date.today()
    value = Decimal("150.0")
    
    rate = FXRate(ccy1, ccy2, today, value)
    unpacked_ccy1, unpacked_ccy2, unpacked_date, unpacked_value = rate
    
    assert unpacked_ccy1 == ccy1
    assert unpacked_ccy2 == ccy2
    assert unpacked_date == today
    assert unpacked_value == value
```


# LLM-generated content at query #17
#--------------------------

```python
def test_fxrate_constructor_valid_assignment():
    import datetime
    from decimal import Decimal
    from typing import NamedTuple
    # Mocking Currency and Date types as they are not provided in snippet but required for context
    class Currency(str): pass
    Date = datetime.date
    
    ccy1 = Currency("EUR")
    ccy2 = Currency("USD")
    today = datetime.date.today()
    val = Decimal("2.0")
    
    rate = FXRate(ccy1, ccy2, today, val)
    
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == today
    assert rate.value == val

def test_fxrate_constructor_indexed_access():
    import datetime
    from decimal import Decimal
    class Currency(str): pass
    
    ccy1 = Currency("EUR")
    ccy2 = Currency("USD")
    today = datetime.date.today()
    val = Decimal("1.5")
    
    rate = FXRate(ccy1, ccy2, today, val)
    
    assert rate[0] == ccy1
    assert rate[1] == ccy2
    assert rate[2] == today
    assert rate[3] == val

def test_fxrate_constructor_tuple_unpacking():
    import datetime
    from decimal import Decimal
    class Currency(str): pass
    
    ccy1 = Currency("GBP")
    ccy2 = Currency("JPY")
    today = datetime.date.today()
    val = Decimal("150.0")
    
    rate = FXRate(ccy1, ccy2, today, val)
    c1, c2, d, v = rate
    
    assert c1 == ccy1
    assert c2 == ccy2
    assert d == today
    assert v == val
```


# LLM-generated content at query #18
#--------------------------

```python
def test_fxrate_constructor_assignment():
    import datetime
    from decimal import Decimal
    from typing import NamedTuple
    # Mocking Currency and Date as they are part of the context but not defined in snippet
    class Currency(str): pass
    Date = datetime.date
    
    ccy1 = Currency("EUR")
    ccy2 = Currency("USD")
    today = datetime.date.today()
    val = Decimal("2.0")
    rate = FXRate(ccy1, ccy2, today, val)
    
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == today
    assert rate.value == val

def test_fxrate_constructor_tuple_access():
    import datetime
    from decimal import Decimal
    class Currency(str): pass
    
    ccy1 = Currency("EUR")
    ccy2 = Currency("USD")
    today = datetime.date.today()
    val = Decimal("2.0")
    rate = FXRate(ccy1, ccy2, today, val)
    
    assert rate[0] == ccy1
    assert rate[1] == ccy2
    assert rate[2] == today
    assert rate[3] == val

def test_fxrate_constructor_unpacking():
    import datetime
    from decimal import Decimal
    class Currency(str): pass
    
    ccy1 = Currency("EUR")
    ccy2 = Currency("USD")
    today = datetime.date.today()
    val = Decimal("2.0")
    rate = FXRate(ccy1, ccy2, today, val)
    
    c1, c2, d, v = rate
    assert c1 == ccy1
    assert c2 == ccy2
    assert d == today
    assert v == val
```


# LLM-generated content at query #19
#--------------------------

```python
from decimal import Decimal
from datetime import date
from typing import Optional, Tuple
from unittest.mock import MagicMock

def test_fxrate_service_query_returns_value_when_exists():
    mock_service = MagicMock()
    mock_rate = MagicMock()
    usd = MagicMock()
    eur = MagicMock()
    asof_date = date(2023, 1, 1)
    
    mock_service.query.return_value = mock_rate
    
    result = mock_service.query(usd, eur, asof_date, strict=True)
    
    mock_service.query.assert_called_once_with(usd, eur, asof_date, strict=True)
    assert result == mock_rate

def test_fxrate_service_query_returns_none_when_not_found():
    mock_service = MagicMock()
    usd = MagicMock()
    eur = MagicMock()
    asof_date = date(2023, 1, 1)
    
    mock_service.query.return_value = None
    
    result = mock_service.query(usd, eur, asof_date, strict=False)
    
    mock_service.query.assert_called_once_with(usd, eur, asof_date, strict=False)
    assert result is None

def test_fxrate_service_query_parameters_mapping():
    mock_service = MagicMock()
    usd = MagicMock()
    eur = MagicMock()
    asof_date = date(2023, 1, 1)
    
    mock_service.query(usd, eur, asof_date, strict=False)
    
    args, kwargs = mock_service.query.call_args
    assert args[0] == usd
    assert args[1] == eur
    assert args[2] == asof_date
    assert kwargs['strict'] is False
```


# LLM-generated content at query #20
#--------------------------

```python
from unittest.mock import MagicMock
from decimal import Decimal
from datetime import date

def test_queries_returns_expected_rates():
    mock_service = MagicMock(spec=FXRateService)
    query_list = [
        ("USD", "EUR", date(2023, 1, 1)),
        ("GBP", "USD", date(2023, 1, 2))
    ]
    expected_rates = [Decimal("0.92"), Decimal("1.25")]
    mock_service.queries.return_value = iter(expected_rates)

    results = list(mock_service.queries(query_list))

    assert len(results) == 2
    assert results[0] == Decimal("0.92")
    assert results[1] == Decimal("1.25")
    mock_service.queries.assert_called_once_with(query_list, strict=False)

def test_queries_with_strict_mode_true():
    mock_service = MagicMock(spec=FXRateService)
    query_list = [("USD", "JPY", date(2023, 1, 1))]
    mock_service.queries.return_value = iter([Decimal("130.0")])

    results = list(mock_service.queries(query_list, strict=True))

    assert results == [Decimal("130.0")]
    mock_service.queries.assert_called_once_with(query_list, strict=True)

def test_queries_returns_none_for_missing_rate():
    mock_service = MagicMock(spec=FXRateService)
    query_list = [("USD", "XYZ", date(2023, 1, 1))]
    mock_service.queries.return_value = iter([None])

    results = list(mock_service.queries(query_list))

    assert results == [None]
    mock_service.queries.assert_called_once()
```


# LLM-generated content at query #21
#--------------------------

```python
def test_fxrate_constructor_valid_input():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    date_val = datetime.date.today()
    value_val = Decimal("1.5")
    rate = FXRate(Currencies["EUR"], Currencies["USD"], date_val, value_val)
    assert rate.ccy1 == Currencies["EUR"]
    assert rate.ccy2 == Currencies["USD"]
    assert rate.date == date_val
    assert rate.value == value_val
    assert rate[0] == Currencies["EUR"]
    assert rate[1] == Currencies["USD"]
    assert rate[2] == date_val
    assert rate[3] == value_val

def test_fxrate_constructor_tuple_unpacking():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    date_val = datetime.date.today()
    value_val = Decimal("2.0")
    rate = FXRate(Currencies["USD"], Currencies["GBP"], date_val, value_val)
    ccy1, ccy2, date, value = rate
    assert ccy1 == Currencies["USD"]
    assert ccy2 == Currencies["GBP"]
    assert date == date_val
    assert value == value_val
```


# LLM-generated content at query #22
#--------------------------

```python
def test_fxrate_constructor_valid_assignment():
    from decimal import Decimal
    import datetime
    # Assuming Currency and Date are types available in the context
    ccy1 = "EUR" 
    ccy2 = "USD"
    date = datetime.date.today()
    value = Decimal("2")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value

def test_fxrate_constructor_tuple_unpacking():
    from decimal import Decimal
    import datetime
    ccy1 = "EUR"
    ccy2 = "USD"
    date = datetime.date.today()
    value = Decimal("1.5")
    rate = FXRate(ccy1, ccy2, date, value)
    c1, c2, d, v = rate
    assert c1 == "EUR"
    assert c2 == "USD"
    assert d == date
    assert v == Decimal("1.5")

def test_fxrate_constructor_indexed_access():
    from decimal import Decimal
    import datetime
    ccy1 = "GBP"
    ccy2 = "JPY"
    date = datetime.date(2023, 1, 1)
    value = Decimal("150")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate[0] == "GBP"
    assert rate[1] == "JPY"
    assert rate[2] == datetime.date(2023, 1, 1)
    assert rate[3] == Decimal("150")
```


# LLM-generated content at query #23
#--------------------------

```python
from unittest.mock import MagicMock
from decimal import Decimal
from datetime import date

def test_queries_returns_correct_values():
    service = MagicMock(spec=FXRateService)
    query_list = [('USD', 'EUR', date(2023, 1, 1)), ('GBP', 'JPY', date(2023, 1, 2))]
    expected_rates = [Decimal('0.92'), Decimal('160.50')]
    service.queries.return_value = iter(expected_rates)

    results = list(service.queries(query_list))

    assert results == expected_rates
    service.queries.assert_called_once_with(query_list, strict=False)

def test_queries_with_strict_true():
    service = MagicMock(spec=FXRateService)
    query_list = [('USD', 'CAD', date(2023, 1, 1))]
    service.queries.return_value = iter([Decimal('1.35')])

    results = list(service.queries(query_list, strict=True))

    assert results == [Decimal('1.35')]
    service.queries.assert_called_once_with(query_list, strict=True)

def test_queries_returns_none_for_missing_rate():
    service = MagicMock(spec=FXRateService)
    query_list = [('XYZ', 'ABC', date(2023, 1, 1))]
    service.queries.return_value = iter([None])

    results = list(service.queries(query_list))

    assert results == [None]
    service.queries.assert_called_once_with(query_list, strict=False)

def test_queries_handles_empty_iterable():
    service = MagicMock(spec=FXRateService)
    query_list = []
    service.queries.return_value = iter([])

    results = list(service.queries(query_list))

    assert results == []
    service.queries.assert_called_once_with([], strict=False)
```


# LLM-generated content at query #24
#--------------------------

```python
def test_fxrate_service_query_interface():
    from datetime import date
    from decimal import Decimal
    from typing import Optional
    from pypara.currencies import Currency, CurrencyType
    from pypara.exchange import FXRateService

    class MockFXRate:
        def __init__(self, rate: Decimal):
            self.rate = rate

    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: date, strict: bool = False) -> Optional[MockFXRate]:
            return MockFXRate(Decimal("1.2"))

    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    as_of_date = date(2023, 1, 1)
    service = MockFXRateService()
    result = service.query(usd, eur, as_of_date)

    assert isinstance(result, MockFXRate)
    assert result.rate == Decimal("1.2")

def test_fxrate_service_query_returns_none():
    from datetime import date
    from typing import Optional
    from pypara.currencies import Currency, CurrencyType
    from pypara.exchange import FXRateService

    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: date, strict: bool = False) -> Optional[object]:
            return None

    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    as_of_date = date(2023, 1, 1)
    service = MockFXRateService()
    result = service.query(usd, eur, as_of_date)

    assert result is None
```


# LLM-generated content at query #25
#--------------------------

```python
import datetime
from decimal import Decimal
from typing import NamedTuple

# Mocking dependencies based on the provided code context
class Currency(str):
    pass

class Currencies:
    def __getitem__(self, key):
        return Currency(key)

Currencies = Currencies()
ZERO = Decimal("0")
ONE = Decimal("1")

# The class to test (as provided in the prompt)
class FXRate(NamedTuple):
    ccy1: Currency
    ccy2: Currency
    date: datetime.date
    value: Decimal

    def __invert__(self) -> "FXRate":
        return FXRate(self[1], self[0], self[2], self[3] ** -1)

    @classmethod
    def of(cls, ccy1: Currency, ccy2: Currency, date: datetime.date, value: Decimal) -> "FXRate":
        if not isinstance(ccy1, Currency):
            raise ValueError("CCY/1 must be of type `Currency`.")
        if not isinstance(ccy2, Currency):
            raise ValueError("CCY/2 must be of type `Currency`.")
        if value <= ZERO:
            raise ValueError("FX rate value can not be equal to or less than `zero`.")
        if ccy1 == ccy2 and value != ONE:
            raise ValueError("FX rate to the same currency must be `one`.")
        return cls(ccy1, ccy2, date, value)

def test_fxrate_constructor_valid_assignment():
    date_val = datetime.date.today()
    value_val = Decimal("2.5")
    ccy1_val = Currencies["EUR"]
    ccy2_val = Currencies["USD"]
    rate = FXRate(ccy1_val, ccy2_val, date_val, value_val)
    assert rate.ccy1 == ccy1_val
    assert rate.ccy2 == ccy2_val
    assert rate.date == date_val
    assert rate.value == value_val

def test_fxrate_constructor_tuple_access():
    date_val = datetime.date.today()
    value_val = Decimal("1.5")
    ccy1_val = Currencies["GBP"]
    ccy2_val = Currencies["JPY"]
    rate = FXRate(ccy1_val, ccy2_val, date_val, value_val)
    assert rate[0] == ccy1_val
    assert rate[1] == ccy2_val
    assert rate[2] == date_val
    assert rate[3] == value_val

def test_fxrate_of_valid_creation():
    date_val = datetime.date.today()
    value_val = Decimal("1.2")
    ccy1_val = Currencies["EUR"]
    ccy2_val = Currencies["USD"]
    rate = FXRate.of(ccy1_val, ccy2_val, date_val, value_val)
    assert rate.ccy1 == ccy1_val
    assert rate.value == value_val

def test_fxrate_of_same_currency_with_one_is_valid():
    date_val = datetime.date.today()
    ccy_val = Currencies["EUR"]
    rate = FXRate.of(ccy_val, ccy_val, date_val, Decimal("1"))
    assert rate.ccy1 == rate.ccy2
    assert rate.value == Decimal("1")

def test_fxrate_of_same_currency_not_one_raises_error():
    date_val = datetime.date.today()
    ccy_val = Currencies["EUR"]
    try:
        FXRate.of(ccy_val, ccy_val, date_val, Decimal("1.5"))
        assert False
    except ValueError as e:
        assert str(e) == "FX rate to the same currency must be `one`."

def test_fxrate_of_zero_value_raises_error():
    date_val = datetime.date.today()
    ccy1_val = Currencies["EUR"]
    ccy2_val = Currencies["USD"]
    try:
        FXRate.of(ccy1_val, ccy2_val, date_val, Decimal("0"))
        assert False
    except ValueError as e:
        assert str(e) == "FX rate value can not be equal to or less than `zero`."

def test_fxrate_of_negative_value_raises_error():
    date_val = datetime.date.today()
    ccy1_val = Currencies["EUR"]
    ccy2_val = Currencies["USD"]
    try:
        FXRate.of(ccy1_val, ccy2_val, date_val, Decimal("-1.0"))
        assert False
    except ValueError as e:
        assert str(e) == "FX rate value can not be equal to or less than `zero`."

def test_fxrate_of_invalid_ccy1_type_raises_error():
    date_val = datetime.date.today()
    try:
        FXRate.of("NOT_A_CURRENCY", Currencies["USD"], date_val, Decimal("1"))
        assert False
    except ValueError as e:
        assert str(e) == "CCY/1 must be of type `Currency`."

def test_fxrate_of_invalid_ccy2_type_raises_error():
    date_val = datetime.date.today()
    try:
        FXRate.of(Currencies["EUR"], "NOT_A_CURRENCY", date_val, Decimal("1"))
        assert False
    except ValueError as e:
        assert str(e) == "CCY/2 must be of type `Currency`."
```


# LLM-generated content at query #26
#--------------------------

```python
import datetime
from decimal import Decimal
from typing import NamedTuple

# Mocking dependencies since they are not provided in the snippet but required for execution
class Currency(str):
    pass

class Currencies:
    def __getitem__(self, key):
        return Currency(key)

Currencies = Currencies()
ZERO = Decimal("0")
ONE = Decimal("1")

# The class to be tested (as provided in the prompt)
class FXRate(NamedTuple):
    ccy1: Currency
    ccy2: Currency
    date: datetime.date
    value: Decimal

    def __invert__(self) -> "FXRate":
        return FXRate(self[1], self[0], self[2], self[3] ** -1)

    @classmethod
    def of(cls, ccy1: Currency, ccy2: Currency, date: datetime.date, value: Decimal) -> "FXRate":
        if not isinstance(ccy1, Currency):
            raise ValueError("CCY/1 must be of type `Currency`.")
        if not isinstance(ccy2, Currency):
            raise ValueError("CCY/2 must be of type `Currency`.")
        if value <= ZERO:
            raise ValueError("FX rate value can not be equal to or less than `zero`.")
        if ccy1 == ccy2 and value != ONE:
            raise ValueError("FX rate to the same currency must be `one`.")
        return cls(ccy1, ccy2, date, value)

def test_fxrate_constructor_valid_input():
    date_val = datetime.date.today()
    value_val = Decimal("1.5")
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    rate = FXRate(ccy1, ccy2, date_val, value_val)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date_val
    assert rate.value == value_val

def test_fxrate_constructor_tuple_access():
    date_val = datetime.date.today()
    value_val = Decimal("2.0")
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["GBP"]
    rate = FXRate(ccy1, ccy2, date_val, value_val)
    assert rate[0] == ccy1
    assert rate[1] == ccy2
    assert rate[2] == date_val
    assert rate[3] == value_val

def test_fxrate_of_valid_input():
    date_val = datetime.date.today()
    value_val = Decimal("1.2")
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    rate = FXRate.of(ccy1, ccy2, date_val, value_val)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.value == value_val

def test_fxrate_of_invalid_ccy1_type():
    import pytest
    with pytest.raises(ValueError, match="CCY/1 must be of type `Currency`."):
        FXRate.of("NOT_A_CURRENCY", Currencies["USD"], datetime.date.today(), Decimal("1.0"))

def test_fxrate_of_invalid_ccy2_type():
    import pytest
    with pytest.raises(ValueError, match="CCY/2 must be of type `Currency`."):
        FXRate.of(Currencies["EUR"], "NOT_A_CURRENCY", datetime.date.today(), Decimal("1.0"))

def test_fxrate_of_zero_value():
    import pytest
    with pytest.raises(ValueError, match="FX rate value can not be equal to or less than `zero`."):
        FXRate.of(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("0"))

def test_fxrate_of_negative_value():
    import pytest
    with pytest.raises(ValueError, match="FX rate value can not be equal to or less than `zero`."):
        FXRate.of(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("-1.0"))

def test_fxrate_of_same_currency_invalid_value():
    import pytest
    with pytest.raises(ValueError, match="FX rate to the same currency must be `one`."):
        FXRate.of(Currencies["EUR"], Currencies["EUR"], datetime.date.today(), Decimal("1.5"))

def test_fxrate_of_same_currency_valid_value():
    rate = FXRate.of(Currencies["EUR"], Currencies["EUR"], datetime.date.today(), Decimal("1.0"))
    assert rate.ccy1 == rate.ccy2
    assert rate.value == Decimal("1.0")

def test_fxrate_inversion():
    date_val = datetime.date.today()
    nrate = FXRate(Currencies["EUR"], Currencies["USD"], date_val, Decimal("2.0"))
    rrate = ~nrate
    assert rrate.ccy1 == Currencies["USD"]
    assert rrate.ccy2 == Currencies["EUR"]
    assert rrate.value == Decimal("0.5")
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
from decimal import Decimal
from datetime import date
from unittest.mock import MagicMock
from pypara.currencies import Currency, CurrencyType
from pypara.exchange import FXRateService

def test_fxrate_service_query_returns_rate_when_exists():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)
    mock_rate = MagicMock()
    service = MagicMock(spec=FXRateService)
    service.query.return_value = mock_rate

    result = service.query(ccy1, ccy2, asof_date)

    assert result == mock_rate
    service.query.assert_called_once_with(ccy1, ccy2, asof_date, strict=False)

def test_fxrate_service_query_returns_none_when_not_found():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)
    service = MagicMock(spec=FXRateService)
    service.query.return_value = None

    result = service.query(ccy1, ccy2, asof_date)

    assert result is None
    service.query.assert_called_once_with(ccy1, ccy2, asof_date, strict=False)

def test_fxrate_service_query_respects_strict_parameter():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY)
    asof_date = date(2023, 5, 20)
    service = MagicMock(spec=FXRateService)

    service.query(ccy1, ccy2, asof_date, strict=True)

    service.query.assert_called_with(ccy1, ccy2, asof_date, strict=True)
```


# LLM-generated content at query #2
#--------------------------

```python
def test_fxrateservice_query_abstract_method_raises_error():
    from pypara.exchange import FXRateService
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currency, CurrencyType

    class MockFXService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return super().query(ccy1, ccy2, asof, strict)
        
        def queries(self, queries, strict=False):
            return super().queries(queries, strict)

    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)
    service = MockFXService()

    import pytest
    with pytest.raises(TypeError):
        service.query(usd, eur, asof_date)

def test_fxrateservice_query_interface_definition():
    from pypara.exchange import FXRateService
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currency, CurrencyType

    class MockFXService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: date, strict: bool = False):
            return Decimal("1.10")
        
        def queries(self, queries, strict=False):
            return []

    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)
    service = MockFXService()
    
    result = service.query(usd, eur, asof_date)
    assert result == Decimal("1.10")
```


# LLM-generated content at query #3
#--------------------------

def test_fxrate_service_query_interface():
    from abc import ABCMeta
    from datetime import date
    from decimal import Decimal
    from typing import Optional, Tuple
    from pypara.currencies import Currency, CurrencyType
    from pypara.exchange import FXRateService

    class MockFXRate:
        pass

    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: date, strict: bool = False) -> Optional[MockFXRate]:
            return MockFXRate()

        def queries(self, queries: any, strict: bool = False) -> any:
            return []

    service = MockFXRateService()
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    today = date.today()
    
    result = service.query(usd, eur, today)
    
    assert isinstance(result, MockFXRate)


# LLM-generated content at query #4
#--------------------------

```python
from decimal import Decimal
from datetime import date
from typing import Any, Optional
from unittest.mock import MagicMock

def test_query_returns_rate_when_found():
    mock_service = MagicMock()
    ccy1 = MagicMock()
    ccy2 = MagicMock()
    asof_date = date(2023, 1, 1)
    expected_rate = MagicMock()
    mock_service.query.return_value = expected_rate
    
    result = mock_service.query(ccy1, ccy2, asof_date)
    
    mock_service.query.assert_called_once_with(ccy1, ccy2, asof_date)
    assert result == expected_rate

def test_query_returns_none_when_not_found():
    mock_service = MagicMock()
    ccy1 = MagicMock()
    ccy2 = MagicMock()
    asof_date = date(2023, 1, 1)
    mock_service.query.return_value = None
    
    result = mock_service.query(ccy1, ccy2, asof_date)
    
    assert result is None

def test_query_respects_strict_parameter():
    mock_service = MagicMock()
    ccy1 = MagicMock()
    ccy2 = MagicMock()
    asof_date = date(2023, 1, 1)
    
    mock_service.query(ccy1, ccy2, asof_date, strict=True)
    mock_service.query.assert_called_with(ccy1, ccy2, asof_date, strict=True)
    
    mock_service.query(ccy1, ccy2, asof_date, strict=False)
    mock_service.query.assert_called_with(ccy1, ccy2, asof_date, strict=False)
```


# LLM-generated content at query #5
#--------------------------

```python
from unittest.mock import MagicMock
from decimal import Decimal
from datetime import date

def test_queries_returns_correct_rates():
    service = MagicMock(spec=FXRateService)
    query_list = [
        ("USD", "EUR", date(2023, 1, 1)),
        ("GBP", "USD", date(2023, 1, 2))
    ]
    expected_rates = [Decimal("0.94"), Decimal("1.22")]
    service.queries.return_value = iter(expected_rates)

    result = list(service.queries(query_list, strict=False))

    assert result == expected_rates
    service.queries.assert_called_once_with(query_list, strict=False)

def test_queries_with_strict_mode_raises_error():
    service = MagicMock(spec=FXRateService)
    query_list = [("USD", "JPY", date(202ss, 1, 1))]
    service.queries.side_effect = ValueError("Rate not found")

    with Exception as e:
        list(service.queries(query_list, strict=True))
    
    assert str(e) == "Rate not found"
    service.queries.assert_called_once_with(query_list, strict=True)

def test_queries_returns_none_for_missing_rate():
    service = MagicMock(spec=FXRateService)
    query_list = [("USD", "XYZ", date(2023, 1, 1))]
    service.queries.return_value = iter([None])

    result = list(service.queries(query_list))

    assert result == [None]
    service.queries.assert_called_once_with(query_list, strict=False)
```


# LLM-generated content at query #6
#--------------------------

```python
from unittest.mock import MagicMock
from decimal import Decimal
from datetime import date

def test_queries_returns_correct_rates_for_valid_input():
    service = MagicMock(spec=FXRateService)
    query_list = [("USD", "EUR", date(2023, 1, 1)), ("GBP", "JPY", date(2023, 1, 2))]
    expected_rates = [Decimal("0.92"), Decimal("160.5")]
    service.queries.return_value = iter(expected_rates)

    result = list(service.queries(query_list))

    assert result == expected_rates
    service.queries.assert_called_once_with(query_list, strict=False)

def test_queries_returns_none_when_rate_not_found():
    service = MagicMock(spec=FXRateService)
    query_list = [("USD", "XYZ", date(2023, 1, 1))]
    expected_rates = [None]
    service.queries.return_value = iter(expected_rates)

    result = list(service.queries(query_list))

    assert result == [None]
    service.queries.assert_called_once_with(query_list, strict=False)

def test_queries_with_strict_mode_true():
    service = MagicMock(spec=FXRateService)
    query_list = [("USD", "EUR", date(2023, 1, 1))]
    service.queries.return_value = iter([Decimal("0.92")])

    list(service.queries(query_list, strict=True))

    service.queries.assert_called_once_with(query_list, strict=True)

def test_queries_handles_empty_iterable():
    service = MagicMock(spec=FXRateService)
    query_list = []
    service.queries.return_value = iter([])

    result = list(service.queries(query_list))

    assert result == []
    service.queries.assert_called_once_with([], strict=False)
```


# LLM-generated content at query #7
#--------------------------

```python
from unittest.mock import MagicMock
from datetime import date
from decimal import Decimal

def test_queries_returns_correct_values():
    mock_service = MagicMock(spec=FXRateService)
    query_data = [
        ("USD", "EUR", date(2023, 1, 1)),
        ("GBP", "JPY", date(2023, 1, 2))
    ]
    expected_rates = [Decimal("0.92"), Decimal("160.50")]
    mock_service.queries.return_value = iter(expected_rates)

    result = list(mock_service.queries(query_data))

    assert result == expected_rates
    mock_service.queries.assert_called_once_with(query_data, strict=False)

def test_queries_handles_none_values():
    mock_service = MagicMock(spec=FXRateService)
    query_data = [("USD", "CAD", date(2023, 1, 1))]
    expected_rates = [None]
    mock_service.queries.return_value = iter(expected_rates)

    result = list(mock_service.queries(query_data))

    assert result == [None]
    mock_service.queries.assert_called_once_with(query_data, strict=False)

def test_queries_with_strict_mode():
    mock_service = MagicMock(spec=FXRateService)
    query_data = [("USD", "AUD", date(2023, 1, 1))]
    expected_rates = [Decimal("1.50")]
    mock_service.queries.return_value = iter(expected_rates)

    result = list(mock_service.queries(query_data, strict=True))

    assert result == [Decimal("1.50")]
    mock_service.queries.assert_called_once_with(query_data, strict=True)
```


# LLM-generated content at query #8
#--------------------------

```python
from unittest.mock import MagicMock
from decimal import Decimal
from datetime import date

def test_queries_returns_correct_values():
    service = MagicMock(spec=FXRateService)
    query_input = [("USD", "EUR", date(2023, 1, 1)), ("GBP", "USD", date(2023, 1, 1))]
    expected_output = [Decimal("0.92"), Decimal("1.22")]
    service.queries.return_value = iter(expected_output)

    result = list(service.queries(query_input))

    assert result == expected_output
    service.queries.assert_called_once_with(query_input, strict=False)

def test_queries_with_strict_true():
    service = MagicMock(spec=FXRateService)
    query_input = [("USD", "JPY", date(2023, 1, 1))]
    service.queries.return_value = iter([Decimal("130.0")])

    result = list(service.queries(query_input, strict=True))

    assert result == [Decimal("130.0")]
    service.queries.assert_called_once_with(query_input, strict=True)

def test_queries_returns_none_for_missing_rates():
    service = MagicMock(spec=FXRateService)
    query_input = [("USD", "XYZ", date(2023, 1, 1))]
    service.queries.return_value = iter([None])

    result = list(service.queries(query_input))

    assert result == [None]
    service.queries.assert_called_once()
```


# LLM-generated content at query #9
#--------------------------

```python
def test_fxrate_constructor_assignment():
    from decimal import Decimal
    import datetime
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("2")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value

def test_fxrate_constructor_tuple_access():
    from decimal import Decimal
    import datetime
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("2")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate[0] == ccy1
    assert rate[1] == ccy2
    assert rate[2] == date
    assert rate[3] == value
```


# LLM-generated content at query #10
#--------------------------

```python
def test_fxrate_constructor_valid_data():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("1.25")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value
    assert rate[0] == ccy1
    assert rate[1] == ccy2
    assert rate[2] == date
    assert rate[3] == value

def test_fxrate_constructor_tuple_unpacking():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["GBP"]
    ccy2 = Currencies["JPY"]
    date = datetime.date(2023, 1, 1)
    value = Decimal("150.0")
    rate = FXRate(ccy1, ccy2, date, value)
    unpacked_ccy1, unpacked_ccy2, unpacked_date, unpacked_value = rate
    assert unpacked_ccy1 == ccy1
    assert unpacked_ccy2 == ccy2
    assert unpacked_date == date
    assert unpacked_value == value
```


# LLM-generated content at query #11
#--------------------------

```python
def test_fxrate_constructor_valid_assignment():
    from datetime import date
    from decimal import Decimal
    from typing import NamedTuple
    # Mocking Currency and Date types as they are part of the context
    class Currency(str): pass
    Date = date
    
    ccy1 = Currency("EUR")
    ccy2 = Currency("USD")
    today = date.today()
    val = Decimal("2.0")
    
    rate = FXRate(ccy1, ccy2, today, val)
    
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == today
    assert rate.value == val
    assert rate[0] == ccy1
    assert rate[3] == val

def test_fxrate_constructor_tuple_unpacking():
    from datetime import date
    from decimal import Decimal
    class Currency(str): pass
    
    ccy1 = Currency("EUR")
    ccy2 = Currency("USD")
    today = date.today()
    val = Decimal("1.5")
    
    rate = FXRate(ccy1, ccy1, today, val) # Note: constructor doesn't validate same currency like .of does
    unpacked_ccy1, unpacked_ccy2, unpacked_date, unpacked_val = rate
    
    assert unpacked_ccy1 == ccy1
    assert unpacked_ccy2 == ccy2
    assert unpacked_date == today
    assert unpacked_val == val
```


# LLM-generated content at query #12
#--------------------------

```python
def test_fxrate_constructor_valid_input():
    import datetime
    from decimal import Decimal
    # Assuming Currency is a type that can be instantiated or mocked
    # Since we don't have the definition of Currency, we assume it works with strings for this test context
    class MockCurrency:
        def __init__(self, code): self.code = code
        def __eq__(self, other): return isinstance(other, MockCurrency) and self.code == other.code
    
    ccy1 = MockCurrency("EUR")
    ccy2 = MockCurrency("USD")
    dt = datetime.date.today()
    val = Decimal("2.0")
    
    rate = FXRate(ccy1, ccy2, dt, val)
    
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == dt
    assert rate.value == val
    assert rate[0] == ccy1
    assert rate[3] == val

def test_fxrate_constructor_tuple_access():
    import datetime
    from decimal import Decimal
    class MockCurrency:
        def __init__(self, code): self.code = code
        def __eq__(self, other): return isinstance(other, MockCurrency) and self.code == other.code

    ccy1 = MockCurrency("EUR")
    ccy2 = MockCurrency("USD")
    dt = datetime.date.today()
    val = Decimal("1.5")
    
    rate = FXRate(ccy1, ccy2, dt, val)
    
    assert rate[0] == ccy1
    assert rate[1] == ccy2
    assert rate[2] == dt
    assert rate[3] == val
```


# LLM-generated content at query #13
#--------------------------

```python
def test_fxrate_constructor_valid_input():
    import datetime
    from decimal import Decimal
    # Assuming Currency is a type that can be instantiated or mocked
    # For the purpose of this test, we use a mock/dummy object representing Currency
    class MockCurrency:
        def __init__(self, code):
            self.code = code
        def __eq__(self, other):
            return isinstance(other, MockCurrency) and self.code == other.code

    ccy1 = MockCurrency("EUR")
    ccy2 = MockCurrency("USD")
    date_val = datetime.date.today()
    value_val = Decimal("2.0")
    
    rate = FXRate(ccy1, ccy2, date_val, value_val)
    
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date_val
    assert rate.value == value_val
    assert rate[0] == ccy1
    assert rate[1] == ccy2
    assert rate[2] == date_val
    assert rate[3] == value_val

def test_fxrate_constructor_tuple_unpacking():
    import datetime
    from decimal import Decimal
    class MockCurrency:
        def __init__(self, code):
            self.code = code

    ccy1 = MockCurrency("EUR")
    ccy2 = MockCurrency("USD")
    date_val = datetime.date.append(datetime.date(2023, 1, 1)) # dummy manipulation
    date_val = datetime.date(2023, 1, 1)
    value_val = Decimal("1.5")
    
    rate = FXRate(ccy1, ccy2, date_val, value_val)
    
    unpacked_ccy1, unpacked_ccy2, unpacked_date, unpacked_value = rate
    
    assert unpacked_ccy1 == ccy1
    assert unpacked_ccy2 == ccy2
    assert unpacked_date == date_val
    assert unpacked_value == value_val
```


# LLM-generated content at query #14
#--------------------------

```python
def test_fxrate_constructor_valid():
    import datetime
    from decimal import Decimal
    # Mocking Currency as a simple type since it's not provided in the snippet but used by NamedTuple
    class Currency(str): pass
    EUR = Currency("EUR")
    USD = Currency("USD")
    today = datetime.date.today()
    value = Decimal("2.0")
    
    rate = FXRate(EUR, USD, today, value)
    
    assert rate.ccy1 == EUR
    assert rate.ccy2 == USD
    assert rate.date == today
    assert rate.value == value
    assert rate[0] == EUR
    assert rate[3] == value

def test_fxrate_constructor_tuple_access():
    import datetime
    from decimal import Decimal
    class Currency(str): pass
    EUR = Currency("EUR")
    USD = Currency("USD")
    today = datetime.date.today()
    value = Decimal("1.5")
    
    rate = FXRate(EUR, USD, today, value)
    
    assert rate[0] == EUR
    assert rate[1] == USD
    assert rate[2] == today
    assert rate[3] == value
```


# LLM-generated content at query #15
#--------------------------

```python
def test_fxrate_constructor_valid_assignment():
    from decimal import Decimal
    import datetime
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("2")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value

def test_fxrate_constructor_tuple_unpacking():
    from decimal import Decimal
    import datetime
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("2")
    rate = FXRate(ccy1, ccy2, date, value)
    r_ccy1, r_ccy2, r_date, r_value = rate
    assert r_ccy1 == ccy1
    assert r_ccy2 == ccy2
    assert r_date == date
    assert r_value == value

def test_fxrate_constructor_indexed_access():
    from decimal import Decimal
    import datetime
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("2")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate[0] == ccy1
    assert rate[1] == ccy2
    assert rate[2] == date
    assert rate[3] == value
```


# LLM-generated content at query #16
#--------------------------

```python
def test_fxrate_constructor_valid_assignment():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("2")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value

def test_fxrate_constructor_tuple_indexing():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("2")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate[0] == ccy1
    assert rate[1] == ccy2
    assert rate[2] == date
    assert rate[3] == value

def test_fxrate_constructor_untyped_assignment():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = "EUR"
    ccy2 = "USD"
    date = datetime.date.deafult() # This is just to show constructor doesn't check type
    value = Decimal("2")
    # The docstring explicitly states the constructor does not check input/is not safe.
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == "EUR"
```


# LLM-generated content at query #17
#--------------------------

```python
from unittest.mock import MagicMock
from datetime import date
from decimal import Decimal

def test_queries_returns_iterable_of_rates():
    service = MagicMock(spec=FXRateService)
    query_list = [("USD", "EUR", date(2023, 1, 1)), ("GBP", "USD", date(2023, 1, 2))]
    expected_rates = [Decimal("0.95"), Decimal("1.25")]
    service.queries.return_value = iter(expected_rates)

    result = list(service.queries(query_list))

    assert result == expected_rates
    service.queries.assert_called_once_with(query_list, strict=False)

def test_queries_with_strict_true():
    service = MagicMock(spec=FXRateService)
    query_list = [("USD", "JPY", date(2023, 1, 1))]
    service.queries.return_value = iter([Decimal("130.0")])

    result = list(service.queries(query_list, strict=True))

    assert result == [Decimal("130.0")]
    service.queries.assert_called_once_with(query_list, strict=True)

def test_queries_returns_none_for_missing_rates():
    service = MagicMock(spec=FXRateService)
    query_list = [("USD", "XYZ", date(2023, 1, 1))]
    service.queries.return_value = iter([None])

    result = list(service.queries(query_list))

    assert result == [None]
```


# LLM-generated content at query #18
#--------------------------

```python
def test_fxrateservice_query_interface_signature():
    from datetime import date
    from decimal import Decimal
    from typing import Optional
    # Since FXRateService is an abstract base class, we verify the method signature 
    # and behavior of a mock implementation.
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: date, strict: bool = False) -> Optional[Decimal]:
            return Decimal("1.23")
        
        def queries(self, queries: Iterable[TQuery], strict: bool = False) -> Iterable[Optional[Decimal]]:
            return [Decimal("1.23")]

    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    EUR = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    as_of_date = date(2023, 1, 1)
    service = MockFXRateService()
    
    rate = service.query(USD, EUR, as_of_date)
    
    assert rate == Decimal("1.23")
```


# LLM-generated content at query #19
#--------------------------

```python
def test_fxrate_constructor_valid_input():
    from datetime import date
    from decimal import Decimal
    from typing import NamedTuple
    # Mocking Currency as it's not provided, assuming a simple type for the sake of constructor test
    class Currency(str): pass
    
    ccy1 = Currency("EUR")
    ccy2 = Currency("USD")
    today = date.today()
    val = Decimal("2.0")
    
    rate = FXRate(ccy1, ccy2, today, val)
    
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == today
    assert rate.value == val
    assert rate[0] == ccy1
    assert rate[1] == ccy2
    assert rate[2] == today
    assert rate[3] == val

def test_fxrate_constructor_tuple_unpacking():
    from datetime import date
    from decimal import Decimal
    class Currency(str): pass
    
    ccy1 = Currency("EUR")
    ccy2 = Currency("USD")
    today = date.today()
    val = Decimal("1.5")
    
    rate = FXRate(ccy1, ccy1, today, val)
    unpacked_ccy1, unpacked_ccy2, unpacked_date, unpacked_value = rate
    
    assert unpacked_ccy1 == ccy1
    assert unpacked_ccy2 == ccy1
    assert unpacked_date == today
    assert unpacked_value == val
```


# LLM-generated content at query #20
#--------------------------

```python
def test_fxrate_constructor_valid_input():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    date_val = datetime.date.today()
    value_val = Decimal("1.5")
    rate = FXRate(Currencies["EUR"], Currencies["USD"], date_val, value_val)
    assert rate.ccy1 == Currencies["EUR"]
    assert rate.ccy2 == Currencies["USD"]
    assert rate.date == date_val
    assert rate.value == value_val
    assert rate[0] == Currencies["EUR"]
    assert rate[1] == Currencies["USD"]
    assert rate[2] == date_val
    assert rate[3] == value_val

def test_fxrate_constructor_tuple_unpacking():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    date_val = datetime.date.today()
    value_val = Decimal("2.0")
    rate = FXRate(Currencies["GBP"], Currencies["JPY"], date_val, value_val)
    ccy1, ccy2, d, v = rate
    assert ccy1 == Currencies["GBP"]
    assert ccy2 == Currencies["JPY"]
    assert d == date_val
    assert v == value_val
```


# LLM-generated content at query #21
#--------------------------

```python
from decimal import Decimal
from datetime import date
from unittest.mock import MagicMock
from pypara.currencies import Currency, CurrencyType
from pypara.exchange import FXRateService

def test_fxrate_service_query_returns_rate_when_found():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)
    mock_rate = MagicMock()
    service = MagicMock(spec=FXRateService)
    service.query.return_value = mock_rate

    result = service.query(ccy1, ccy2, asof_date)

    assert result == mock_rate
    service.query.assert_called_once_with(ccy1, ccy2, asof_date, False)

def test_fxrate_service_query_returns_none_when_not_found():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)
    service = MagicMock(spec=FXRateService)
    service.query.return_value = None

    result = service.query(ccy1, ccy2, asof_date)

    assert result is None
    service.query.assert_called_once_with(ccy1, ccy2, asof_date, False)

def test_fxrate_service_query_respects_strict_parameter():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)
    service = MagicMock(spec=FXRateService)

    service.query(ccy1, ccy2, asof_date, strict=True)
    service.query.assert_called_with(ccy1, ccy2, asof_date, True)

    service.query(ccy1, ccy2, asof_date, strict=False)
    service.query.assert_called_with(ccy1, ccy2, asof_date, False)
```


# LLM-generated content at query #22
#--------------------------

```python
from unittest.mock import MagicMock
from decimal import Decimal
from datetime import date

def test_queries_returns_correct_rates():
    mock_service = MagicMock(spec=FXRateService)
    query_list = [("USD", "EUR", date(2023, 1, 1)), ("GBP", "USD", date(2023, 1, 2))]
    expected_rates = [Decimal("0.92"), Decimal("1.25")]
    mock_service.queries.return_value = iter(expected_rates)

    results = list(mock_service.queries(query_list))

    assert results == expected_rates
    mock_service.queries.assert_called_once_with(query_list, strict=False)

def test_queries_with_strict_mode_raises_error():
    mock_service = MagicMock(spec=FXRateService)
    query_list = [("USD", "JPY", date(202im, 1, 1))]
    mock_service.queries.side_effect = ValueError("Rate not found")

    with Exception as e:
        list(mock_service.queries(query_list, strict=True))
    
    assert str(e) == "Rate not found"
    mock_service.queries.assert_called_once_with(query_list, strict=True)

def test_queries_returns_none_for_missing_rate():
    mock_service = MagicMock(spec=FXRateService)
    query_list = [("USD", "XYZ", date(2023, 1, 1))]
    mock_service.queries.return_value = iter([None])

    results = list(mock_service.queries(query_list))

    assert results == [None]
    mock_service.queries.assert_called_once_with(query_list, strict=False)
```


# LLM-generated content at query #23
#--------------------------

```python
def test_fxrate_constructor_assignment():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    date_val = datetime.date.today()
    value_val = Decimal("1.5")
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    rate = FXRate(ccy1, ccy2, date_val, value_val)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date_val
    assert rate.value == value_val

def test_fxrate_constructor_tuple_access():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    date_val = datetime.date.today()
    value_val = Decimal("2.0")
    rate = FXRate(Currencies["EUR"], Currencies["USD"], date_val, value_val)
    assert rate[0] == Currencies["EUR"]
    assert rate[1] == Currencies["USD"]
    assert rate[2] == date_val
    assert rate[3] == value_val
```


# LLM-generated content at query #24
#--------------------------

```python
def test_fxrate_constructor_valid_input():
    from decimal import Decimal
    import datetime
    # Assuming Currency is a type that can be instantiated or mocked. 
    # Using strings/mock objects if actual Currencies class isn't provided, 
    # but based on the docstring logic:
    ccy1 = "EUR" # In real context this would be a Currency instance
    ccy2 = "USD"
    date_val = datetime.date.today()
    value_val = Decimal("2")
    rate = FXRate(ccy1, ccy2, date_val, value_val)
    assert rate[0] == ccy1
    assert rate[1] == ccy2
    assert rate[2] == date_val
    assert rate[3] == value_val

def test_fxrate_constructor_tuple_unpacking():
    from decimal import Decimal
    import datetime
    ccy1 = "EUR"
    ccy2 = "USD"
    date_val = datetime.date.today()
    value_val = Decimal("1.5")
    rate = FXRate(ccy1, ccy2, date_val, value_val)
    unpacked_ccy1, unpacked_ccy2, unpacked_date, unpacked_value = rate
    assert unpacked_ccy1 == "EUR"
    assert unpacked_ccy2 == "USD"
    assert unpacked_date == datetime.date.today()
    assert unpacked_value == Decimal("1.5")

def test_fxrate_constructor_indexed_access():
    from decimal import Decimal
    import datetime
    rate = FXRate("EUR", "USD", datetime.date.today(), Decimal("2"))
    assert rate[0] == "EUR"
    assert rate[1] == "USD"
    assert rate[2] == datetime.date.today()
    assert rate[3] == Decimal("2")
```


# LLM-generated content at query #25
#--------------------------

```python
import datetime
from decimal import Decimal
from typing import NamedTuple

# Mocking Currency and Date types as they are dependencies in the provided snippet
class Currency(str):
    pass

Date = datetime.date

def test_fxrate_constructor_valid_assignment():
    ccy1 = Currency("EUR")
    ccy2 = Currency("USD")
    today = datetime.date.today()
    val = Decimal("2.0")
    rate = FXRate(ccy1, ccy2, today, val)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == today
    assert rate.value == val

def test_fxrate_constructor_tuple_unpacking():
    ccy1 = Currency("EUR")
    ccy2 = Currency("USD")
    today = datetime.date.today()
    val = Decimal("1.5")
    rate = FXRate(ccy1, ccu2 := Currency("GBP"), today, val)
    unpacked_ccy1, unpacked_ccy2, unpacked_date, unpacked_value = rate
    assert unpacked_ccy1 == ccy1
    assert unpacked_ccy2 == ccu2
    assert unpacked_date == today
    assert unpacked_value == val

def test_fxrate_constructor_indexed_access():
    ccy1 = Currency("JPY")
    ccy2 = Currency("USD")
    today = datetime.date.today()
    val = Decimal("0.007")
    rate = FXRate(ccy1, ccy2, today, val)
    assert rate[0] == ccy1
    assert rate[1] == ccy2
    assert rate[2] == today
    assert rate[3] == val

def test_fxrate_constructor_inversion():
    ccy1 = Currency("EUR")
    ccy2 = Currency("USD")
    today = datetime.date.today()
    val = Decimal("2")
    rate = FXRate(ccy1, ccy2, today, val)
    inverted = ~rate
    assert inverted.ccy1 == ccy2
    assert inverted.ccy2 == ccy1
    assert inverted.date == today
    assert inverted.value == Decimal("0.5")
```


# LLM-generated content at query #26
#--------------------------

```python
def test_fxrate_service_query_interface_definition():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currency, CurrencyType
    from pypara.exchange import FXRateService

    # Since FXRateService is an ABC (Abstract Base Class), we cannot instantiate it directly.
    # We must test the signature/interface behavior via a Mock or a concrete implementation.
    # However, since the requirement is to test the 'query' method of the class provided:
    
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    EUR = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)

    # We define a minimal concrete implementation to verify the method exists and can be called
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        
        def queries(self, queries, strict=False):
            return []

    service = MockFXRateService()
    result = service.query(USD, EUR, asof_date)
    
    assert result is None
```


# LLM-generated content at query #27
#--------------------------

```python
from unittest.mock import MagicMock
from decimal import Decimal
from datetime import date

def test_queries_returns_correct_rates():
    service = MagicMock(spec=FXRateService)
    query_list = [
        ("USD", "EUR", date(2023, 1, 1)),
        ("GBP", "JPY", date(2023, 1, 2))
    ]
    expected_rates = [Decimal("0.94"), Decimal("160.50")]
    service.queries.return_value = iter(expected_rates)

    result = list(service.queries(query_list))

    assert result == expected_rates
    service.queries.assert_called_once_with(query_list, strict=False)

def test_queries_with_strict_mode():
    service = MagicMock(spec=FXRateService)
    query_list = [("USD", "CAD", date(2023, 1, 1))]
    service.queries.return_value = iter([Decimal("1.35")])

    result = list(service.queries(query_list, strict=True))

    assert result == [Decimal("1.35")]
    service.queries.assert_called_once_with(query_list, strict=True)

def test_queries_handles_none_values():
    service = MagicMock(spec=FXRateService)
    query_list = [("USD", "XYZ", date(2023, 1, 1))]
    service.queries.return_value = iter([None])

    result = list(service.queries(query_list))

    assert result == [None]
    service.queries.assert_called_once()
```


# LLM-generated content at query #28
#--------------------------

```python
def test_fxrate_constructor_valid_input():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    date_val = datetime.date.today()
    value_val = Decimal("1.5")
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    rate = FXRate(ccy1, ccy2, date_val, value_val)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date_val
    assert rate.value == value_val

def test_fxrate_constructor_tuple_unpacking():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    date_val = datetime.date.today()
    value_val = Decimal("2.0")
    rate = FXRate(Currencies["GBP"], Currencies["JPY"], date_val, value_val)
    ccy1, ccy2, d, v = rate
    assert ccy1 == Currencies["GBP"]
    assert ccy2 == Currencies["JPY"]
    assert d == date_val
    assert v == value_val

def test_fxrate_constructor_indexed_access():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    date_val = datetime.date.today()
    value_val = Decimal("0.85")
    rate = FXRate(Currencies["USD"], Currencies["CAD"], date_val, value_val)
    assert rate[0] == Currencies["USD"]
    assert rate[1] == Currencies["CAD"]
    assert rate[2] == date_val
    assert rate[3] == value_val
```


# LLM-generated content at query #29
#--------------------------

```python
from decimal import Decimal
from datetime import date
from typing import Optional, Tuple
from unittest.mock import MagicMock

def test_query_returns_rate_when_found():
    service = MagicMock(spec=FXRateService)
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)
    expected_rate = MagicMock(spec=FXRate)
    service.query.return_value = expected_rate

    result = service.query(ccy1, ccy2, asof_date)

    assert result == expected_rate
    service.query.assert_called_once_with(ccy1, ccy2, asof_date)

def test_query_returns_none_when_not_found():
    service = MagicMock(spec=FXRateService)
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)
    service.query.return_value = None

    result = service.query(ccy1, ccy2, asof_date)

    assert result is None
    service.query.assert_called_once_with(ccy1, ccy2, asof_date)

def test_query_with_strict_parameter():
    service = MagicMock(spec=FXRateService)
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY)
    asof_date = date(2023, 5, 20)
    
    service.query(ccy1, ccy2, asof_date, strict=True)

    service.query.assert_called_with(ccy1, ccy2, asof_date, strict=True)
```


# LLM-generated content at query #30
#--------------------------

```python
def test_fxrate_constructor_assignment():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("1.5")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value

def test_fxrate_constructor_tuple_access():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("1.5")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate[0] == ccy1
    assert rate[1] == ccy2
    assert rate[2] == date
    assert rate[3] == value

def test_fxrate_constructor_inversion():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("2")
    rate = FXRate(ccy1, ccy2, date, value)
    inverted_rate = ~rate
    assert inverted_rate.ccy1 == ccy2
    assert inverted_rate.ccy2 == ccy1
    assert inverted_rate.date == date
    assert inverted_rate.value == Decimal("0.5")
```


