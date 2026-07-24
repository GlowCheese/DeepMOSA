####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import datetime
from decimal import Decimal
from typing import NamedTuple

# Mocking dependencies for the test environment
class Currency(str):
    pass

USD = Currency("USD")
EUR = Currency("EUR")
ZERO = Decimal("0")
ONE = Decimal("1")

def test_fxrate_invert_success():
    date_val = datetime.date.today()
    nrate = FXRate(EUR, USD, date_val, Decimal("2"))
    rrate = ~nrate
    assert rrate.ccy1 == USD
    assert rrate.ccy2 == EUR
    assert rrate.date == date_val
    assert rrate.value == Decimal("0.5")
    assert ~rrate == nrate

def test_fxrate_invert_identity():
    date_val = datetime.date.today()
    rate = FXRate(EUR, USD, date_val, Decimal("1"))
    inverted_rate = ~rate
    assert inverted_rate.ccy1 == USD
    assert inverted_rate.ccy2 == EUR
    assert inverted_rate.value == Decimal("1")
```


# LLM-generated content at query #2
#--------------------------

```python
from unittest.mock import MagicMock
from decimal import Decimal
from datetime import date

def test_queries_returns_correct_rates_for_valid_inputs():
    service = MagicMock(spec=FXRateService)
    query_list = [
        ('USD', 'EUR', date(2023, 1, 1)),
        ('GBP', 'USD', date(2023, 1, 2))
    ]
    expected_rates = [Decimal('0.92'), Decimal('1.25')]
    service.queries.return_value = iter(expected_rates)

    results = list(service.queries(query_list))

    assert results == expected_rates
    service.queries.assert_called_once_with(query_list, strict=False)

def test_queries_returns_none_when_rate_not_found():
    service = MagicMock(spec=FXRateService)
    query_list = [('USD', 'JPY', date(2023, 1, 1))]
    expected_rates = [None]
    service.queries.return_value = iter(expected_rates)

    results = list(service.queries(query_list))

    assert results == [None]
    service.queries.assert_called_once_with(query_list, strict=False)

def test_queries_with_strict_mode_true():
    service = MagicMock(spec=FXRateService)
    query_list = [('USD', 'EUR', date(2023, 1, 1))]
    service.queries.return_value = iter([Decimal('0.92')])

    list(service.queries(query_list, strict=True))

    service.queries.assert_called_once_with(query_list, strict=True)
```


# LLM-generated content at query #3
#--------------------------

```python
from decimal import Decimal
from datetime import date
from typing import Optional
from unittest.mock import MagicMock
from pypara.currencies import Currency, CurrencyType
from pypara.exchange import FXRateService

def test_fx_rate_service_query_returns_rate_when_exists():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)
    mock_rate_value = Decimal("0.92")
    
    mock_service = MagicMock(spec=FXRateService)
    mock_rate = MagicMock()
    mock_rate.rate = mock_rate_value
    mock_service.query.return_value = mock_rate

    result = mock_service.query(usd, eur, asof_date)

    assert result is not None
    assert result.rate == mock_rate_value
    mock_service.query.assert_called_once_with(usd, eur, asof_date, strict=False)

def test_fx_rate_service_query_returns_none_when_not_found():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)
    
    mock_service = MagicMock(spec=FXRateSrvice)
    mock_service.query.return_value = None

    result = mock_service.query(usd, eur, asof_date)

    assert result is None
    mock_service.query.assert_called_once_with(usd, eur, asof_date, strict=False)

def test_fx_rate_service_query_respects_strict_parameter():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)
    
    mock_service = MagicMock(spec=FXRateService)
    
    mock_service.query(usd, eur, asof_date, strict=True)
    mock_service.query.assert_called_with(usd, eur, asof_date, strict=True)
```


# LLM-generated content at query #4
#--------------------------

```python
from unittest.mock import MagicMock
from decimal import Decimal
from datetime import date

def test_queries_returns_correct_rates_for_valid_inputs():
    mock_service = MagicMock(spec=FXRateService)
    ccy_usd = "USD"
    ccy_eur = "EUR"
    asof_date = date(2023, 1, 1)
    expected_rate = Decimal("0.92")
    
    queries_input = [(ccy_usd, ccy_eur, asof_date)]
    mock_service.queries.return_value = [expected_rate]
    
    result = list(mock_service.queries(queries_input))
    
    assert result == [expected_rate]
    mock_service.queries.assert_called_once_with(queries_input, strict=False)

def test_queries_returns_none_when_rate_not_found():
    mock_service = MagicMock(spec=FXRateService)
    ccy_usd = "USD"
    ccy_jpy = "JPY"
    asof_date = date(2023, 1, 1)
    
    queries_input = [(ccy_usd, ccy_jpy, asof_date)]
    mock_service.queries.return_value = [None]
    
    result = list(mock_service.queries(queries_input))
    
    assert result == [None]
    mock_service.queries.assert_called_once_with(queries_input, strict=False)

def test_queries_handles_multiple_inputs_and_strict_mode():
    mock_service = MagicMock(spec=FXRateService)
    ccy_usd = "USD"
    ccy_eur = "EUR"
    ccy_gbp = "GBP"
    asof_date = date(2023, 1, 1)
    
    queries_input = [
        (ccy_usd, ccy_eur, asof_date),
        (ccy_usd, ccy_gbp, asof_date)
    ]
    expected_output = [Decimal("0.92"), Decimal("0.82")]
    mock_service.queries.return_value = expected_output
    
    result = list(mock_service.queries(queries_input, strict=True))
    
    assert result == expected_output
    mock_service.queries.assert_called_once_with(queries_input, strict=True)
```


# LLM-generated content at query #5
#--------------------------

```python
def test_fxrate_constructor_valid_input():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("2"))
    assert rate.ccy1 == Currencies["EUR"]
    assert rate.ccy2 == Currencies["USD"]
    assert rate.date == datetime.date.today()
    assert rate.value == Decimal("2")

def test_fxrate_constructor_tuple_access():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("2"))
    assert rate[0] == Currencies["EUR"]
    assert rate[1] == Currencies["USD"]
    assert rate[2] == datetime.date.today()
    assert rate[3] == Decimal("2")
```


# LLM-generated content at query #6
#--------------------------

```python
def test_queries_returns_correct_values_from_iterable():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        def queries(self, queries, strict=False):
            return [None, None]

    service = MockFXRateService()
    queries_input = [("USD", "EUR", "2023-01-01"), ("GBP", "JPY", "2023-01-02")]
    results = list(service.queries(queries_input))
    
    assert results == [None, None]
    assert len(results) == 2

def test_queries_with_strict_true_raises_error():
    class MockStrictFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        def queries(self, queries, strict=False):
            if strict:
                raise ValueError("Strict mode error")
            return [None]

    service = MockStrictFXRateService()
    queries_input = [("USD", "EUR", "2023-01-01")]
    
    try:
        service.queries(queries_input, strict=True)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert str(e) == "Strict mode error"

def test_queries_handles_empty_iterable():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        def queries(self, queries, strict=False):
            return []

    service = MockFXRateService()
    results = list(service.queries([]))
    
    assert results == []
```


# LLM-generated content at query #7
#--------------------------

```python
from unittest.mock import MagicMock
from decimal import Decimal
from datetime import date

def test_queries_returns_correct_rates_for_valid_input():
    mock_service = MagicMock(spec=FXRateService)
    ccy_usd = "USD"
    ccy_eur = "EUR"
    asof_date = date(2023, 1, 1)
    expected_rate = Decimal("0.92")
    
    query_list = [(ccy_usd, ccy_eur, asof_date)]
    mock_service.queries.return_value = [expected_rate]
    
    results = list(mock_service.queries(query_list))
    
    assert len(results) == 1
    assert results[0] == expected_rate
    mock_service.queries.assert_called_once_with(query_list, strict=False)

def test_queries_returns_none_when_rate_not_found():
    mock_service = MagicMock(spec=FXRateService)
    ccy_usd = "USD"
    ccy_gbp = "GBP"
    asof_date = date(2023, 1, 1)
    
    query_list = [(ccy_usd, ccy_gbp, asof_date)]
    mock_service.queries.return_value = [None]
    
    results = list(mock_service.queries(query_list))
    
    assert len(results) == 1
    assert results[0] is None
    mock_service.queries.assert_called_once_with(query_list, strict=False)

def test_queries_handles_multiple_inputs_and_strict_mode():
    mock_service = MagicMock(spec=FXRateService)
    ccy_usd = "USD"
    ccy_eur = "EUR"
    ccy_jpy = "JPY"
    asof_date = date(2023, 1, 1)
    
    query_list = [
        (ccy_usd, ccy_eur, asof_date),
        (ccy_usd, ccy_jpy, asof_date)
    ]
    expected_results = [Decimal("0.92"), Decimal("130.0")]
    mock_service.queries.return_value = expected_results
    
    results = list(mock_service.queries(query_list, strict=True))
    
    assert len(results) == 2
    assert results[0] == Decimal("0.92")
    assert results[1] == Decimal("130.0")
    mock_service.queries.assert_called_once_with(query_list, strict=True)
```


# LLM-generated content at query #8
#--------------------------

```python
from decimal import Decimal
from datetime import date
from typing import Optional, Tuple
from unittest.mock import MagicMock
from pypara.currencies import Currency, CurrencyType
from pypara.exchange import FXRateService

def test_fx_rate_service_query_returns_rate_when_exists():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)
    mock_rate = MagicMock()
    
    service = MagicMock(spec=FXRateService)
    service.query.return_value = mock_rate
    
    result = service.query(cccy1, ccy2, asof_date)
    
    assert result == mock_rate
    service.query.assert_called_once_with(cccy1, ccy2, asof_date, strict=False)

def test_fx_rate_service_query_returns_none_when_not_found():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)
    
    service = MagicMock(spec=FXRateService)
    service.query.return_value = None
    
    result = service.query(cccy1, ccy2, asof_date)
    
    assert result is None
    service.query.assert_called_once_with(cccy1, ccy2, asof_date, strict=False)

def test_fx_rate_service_query_with_strict_mode_true():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    asof_date = date(2023, 5, 5)
    
    service = MagicMock(spec=FXRateService)
    
    service.query(ccy1, ccy2, asof_date, strict=True)
    
    service.query.assert_called_with(ccy1, ccy2, asof_date, strict=True)
```


# LLM-generated content at query #9
#--------------------------

```python
from decimal import Decimal
from datetime import date
from typing import Optional, Tuple
from unittest.mock import MagicMock
from pypara.currencies import Currency, CurrencyType
from pypara.exchange import FXRateService

def test_fxrate_service_query_returns_rate_when_found():
    mock_service = MagicMock(spec=FXRateService)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)
    expected_rate = Decimal("0.95")
    mock_service.query.return_value = expected_rate
    
    result = mock_service.query(usd, eur, asof_date)
    
    assert result == expected_rate
    mock_service.query.assert_called_once_with(usd, eur, asof_date, strict=False)

def test_fxrate_service_query_returns_none_when_not_found():
    mock_service = MagicMock(spec=FXRateService)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.from_code_logic_if_available_or_mock("EUR", "Euros", 2, CurrencyType.MONEY) # Note: using Currency.of logic
    # Since I cannot define functions, I will assume Currency.of works as per the provided source
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)
    mock_service.query.return_value = None
    
    result = mock_service.query(usd, eur, asof_date)
    
    assert result is None
    mock_service.query.assert_called_once_with(usd, eur, asof_date, strict=False)

def test_fxrate_service_query_with_strict_mode_true():
    mock_service = MagicMock(spec=FXRateService)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)
    mock_service.query.return_value = Decimal("0.95")
    
    result = mock_service.query(usd, eur, asof_date, strict=True)
    
    assert result == Decimal("0.95")
    mock_service.query.assert_called_once_with(usd, eur, asof_date, strict=True)
```


# LLM-generated content at query #10
#--------------------------

```python
def test_fxrate_constructor_valid_inputs():
    from decimal import Decimal
    import datetime
    from pypara.currencies import Currencies
    
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    today = datetime.date.today()
    value = Decimal("2.0")
    
    rate = FXRate(ccy1, ccy2, today, value)
    
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == today
    assert rate.value == value
    assert rate[0] == ccy1
    assert rate[1] == ccy2
    assert rate[2] == today
    assert rate[3] == value

def test_fxrate_constructor_inversion():
    from decimal import Decimal
    import datetime
    from pypara.currencies import Currencies
    
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    today = datetime.date.today()
    value = Decimal("2.0")
    
    rate = FXRate(ccy1, ccy2, today, value)
    inverted_rate = ~rate
    
    assert inverted_rate.ccy1 == ccy2
    assert inverted_rate.ccy2 == ccy1
    assert inverted_rate.date == today
    assert inverted_rate.value == Decimal("0.5")
```


# LLM-generated content at query #11
#--------------------------

```python
from decimal import Decimal
from datetime import date
from typing import Optional
from unittest.mock import MagicMock

def test_query_returns_rate_when_found():
    mock_service = MagicMock(spec=FXRateService)
    mock_rate = MagicMock(spec=FXRate)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)
    mock_service.query.return_value = mock_rate
    
    result = mock_service.query(usd, eur, asof_date)
    
    assert result == mock_rate
    mock_service.query.assert_called_once_with(usd, eur, asof_date, False)

def test_query_returns_none_when_not_found():
    mock_service = MagicMock(spec=FXRateService)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)
    mock_service.query.return_value = None
    
    result = mock_service.query(usd, eur, asof_date)
    
    assert result is None
    mock_service.query.assert_called_once_with(usd, eur, asof_date, False)

def test_query_with_strict_parameter():
    mock_service = MagicMock(spec=FXRateService)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)
    
    mock_service.query(usd, eur, asof_date, strict=True)
    
    mock_service.query.assert_called_with(usd, eur, asof_date, True)
```


# LLM-generated content at query #12
#--------------------------

```python
def test_fxrate_constructor_valid_assignment():
    from datetime import date
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    dt = date.today()
    val = Decimal("2.0")
    rate = FXRate(ccy1, ccy2, dt, val)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == dt
    assert rate.value == val

def test_fxrate_constructor_tuple_access():
    from datetime import date
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    dt = date.today()
    val = Decimal("2.0")
    rate = FXRate(ccy1, ccy2, dt, val)
    assert rate[0] == ccy1
    assert rate[1] == ccy2
    assert rate[2] == dt
    assert rate[3] == val

def test_fxrate_constructor_untyped_assignment():
    from datetime import date
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = "EUR"
    ccy2 = "USD"
    dt = date.today()
    val = Decimal("2.0")
    rate = FXRate(ccy1, ccy2, dt, val)
    assert rate.ccy1 == "EUR"
    assert rate.ccy2 == "USD"
```


# LLM-generated content at query #13
#--------------------------

```python
def test_fxrate_constructor_assignment():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("2.5")
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
```


# LLM-generated content at query #14
#--------------------------

```python
def test_fxrate_constructor_valid_data():
    from datetime import date
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    dt = date(2023, 1, 1)
    val = Decimal("1.1")
    rate = FXRate(ccy1, ccy2, dt, val)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == dt
    assert rate.value == val

def test_fxrate_constructor_tuple_access():
    from datetime import date
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    dt = date(2023, 1, 1)
    val = Decimal("1.1")
    rate = FXRate(ccy1, ccy2, dt, val)
    assert rate[0] == ccy1
    assert rate[1] == ccy2
    assert rate[2] == dt
    assert rate[3] == val
```


# LLM-generated content at query #15
#--------------------------

```python
def test_fxrate_constructor_valid_assignment():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    dt = date.today()
    val = Decimal("2.0")
    rate = FXRate(ccy1, ccy2, dt, val)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == dt
    assert rate.value == val

def test_fxrate_constructor_tuple_access():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    dt = date.today()
    val = Decimal("2.0")
    rate = FXRate(ccy1, ccy2, dt, val)
    assert rate[0] == ccy1
    assert rate[1] == ccy2
    assert rate[2] == dt
    assert rate[3] == val
```


# LLM-generated content at query #16
#--------------------------

```python
def test_fxrate_constructor_valid_assignment():
    from datetime import date
    from decimal import Decimal
    from typing import NamedTuple
    # Mocking Currency for the scope of the test since the original code relies on it
    class Currency(str): pass
    
    EUR = Currency("EUR")
    USD = Currency("USD")
    today = date.today()
    val = Decimal("2.0")
    
    rate = FXRate(EUR, USD, today, val)
    
    assert rate.ccy1 == EUR
    assert rate.ccy2 == USD
    assert rate.date == today
    assert rate.value == val
    assert rate[0] == EUR
    assert rate[1] == USD
    assert rate[2] == today
    assert rate[3] == val

def test_fxrate_constructor_tuple_unpacking():
    from datetime import date
    from decimal import Decimal
    class Currency(str): pass
    
    EUR = Currency("EUR")
    USD = Currency("USD")
    today = date.today()
    val = Decimal("1.5")
    
    rate = FXRate(EUR, USD, today, val)
    ccy1, ccy2, d, v = rate
    
    assert ccy1 == EUR
    assert ccy2 == USD
    assert d == today
    assert v == val
```


# LLM-generated content at query #17
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
    value = Decimal("1.5")
    rate = FXRate(ccy1, ccy2, date, value)
    unpacked_ccy1, unpacked_ccy2, unpacked_date, unpacked_value = rate
    assert unpacked_ccy1 == ccy1
    assert unpacked_ccy2 == ccy2
    assert unpacked_date == date
    assert unpacked_value == value

def test_fxrate_constructor_indexed_access():
    from decimal import Decimal
    import datetime
    from pypara.currencies import Currencies
    rate = FXRate(Currencies["EUR"], Currencies["GBP"], datetime.date(2023, 1, 1), Decimal("0.85"))
    assert rate[0] == Currencies["EUR"]
    assert rate[1] == Currencies["GBP"]
    assert rate[2] == datetime.date(2023, 1, 1)
    assert rate[3] == Decimal("0.85")
```


# LLM-generated content at query #18
#--------------------------

```python
from decimal import Decimal
from datetime import date
from typing import Optional
from unittest.mock import MagicMock
from pypara.currencies import Currency, CurrencyType
from pypara.exchange import FXRateService

def test_fx_rate_service_query_returns_rate_when_exists():
    ccy_usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy_eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)
    mock_rate = MagicMock()
    service = MagicMock(spec=FXRateService)
    service.query.return_value = mock_rate
    
    result = service.query(ccy_usd, ccy_eur, asof_date)
    
    service.query.assert_called_once_with(ccy_usd, ccy_eur, asof_date, strict=False)
    assert result == mock_rate

def test_fx_rate_service_query_returns_none_when_not_found():
    ccy_usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy_eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)
    service = MagicMock(spec=FXRateService)
    service.query.return_value = None
    
    result = service.query(ccy_usd, ccy_eur, asof_date)
    
    assert result is None

def test_fx_rate_service_query_with_strict_true():
    ccy_usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy_eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)
    service = MagicMock(spec=FXRateService)
    
    service.query(ccy_usd, ccy_eur, asof_date, strict=True)
    
    service.query.assert_called_with(ccy_usd, ccy_eur, asof_date, strict=True)
```


# LLM-generated content at query #19
#--------------------------

```python
from unittest.mock import MagicMock
from decimal import Decimal
from datetime import date

def test_queries_returns_correct_rates_for_valid_inputs():
    mock_service = MagicMock(spec=FXRateService)
    query_data = [
        ("USD", "EUR", date(2023, 1, 1)),
        ("GBP", "USD", date(2023, 1, 2))
    ]
    expected_rates = [Decimal("0.92"), Decimal("1.22")]
    mock_service.queries.return_value = iter(expected_rates)

    results = list(mock_service.queries(query_data))

    assert results == expected_rates
    mock_service.queries.assert_called_once_with(query_data, strict=False)

def test_queries_returns_none_when_rate_not_found():
    mock_service = MagicMock(spec=FXRateService)
    query_data = [("USD", "XYZ", date(2023, 1, 1))]
    mock_service.queries.return_value = iter([None])

    results = list(mock_service.queries(query_data))

    assert results == [None]
    mock_service.queries.assert_called_once_with(query_data, strict=False)

def test_queries_with_strict_mode_enabled():
    mock_service = MagicMock(spec=FXRateService)
    query_data = [("USD", "EUR", date(2023, 1, 1))]
    mock_service.queries.return_value = iter([Decimal("0.92")])

    results = list(mock_service.queries(query_data, strict=True))

    assert results == [Decimal("0.92")]
    mock_service.queries.assert_called_once_with(query_data, strict=True)
```


# LLM-generated content at query #20
#--------------------------

```python
def test_fxrate_constructor_valid_assignment():
    from datetime import date
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    dt = date.today()
    val = Decimal("1.25")
    rate = FXRate(ccy1, ccy2, dt, val)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == dt
    assert rate.value == val

def test_fxrate_constructor_tuple_unpacking():
    from datetime import date
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["GBP"]
    ccy2 = Currencies["JPY"]
    dt = date(2023, 1, 1)
    val = Decimal("150.0")
    rate = FXRate(ccy1, ccy2, dt, val)
    unpacked_ccy1, unpacked_ccy2, unpacked_date, unpacked_value = rate
    assert unpacked_ccy1 == ccy1
    assert unpacked_ccy2 == ccy2
    assert unpacked_date == dt
    assert unpacked_value == val

def test_fxrate_constructor_indexed_access():
    from datetime import date
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["CHF"]
    ccy2 = Currencies["CAD"]
    dt = date(2023, 5, 5)
    val = Decimal("1.5")
    rate = FXRate(ccy1, ccy2, dt, val)
    assert rate[0] == ccy1
    assert rate[1] == ccy2
    assert rate[2] == dt
    assert rate[3] == val
```


# LLM-generated content at query #21
#--------------------------

```python
def test_fxrate_constructor_valid_args():
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
    ccy1, ccy2, date, value = rate
    assert ccy1 == Currencies["GBP"]
    assert ccy2 == Currencies["JPY"]
    assert date == date_val
    assert value == value_val
```


# LLM-generated content at query #22
#--------------------------

```python
def test_fxrate_constructor_valid_assignment():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    date_val = datetime.date.today()
    value_val = Decimal("1.25")
    rate = FXRate(Currencies["EUR"], Currencies["USD"], date_val, value_val)
    assert rate.ccy1 == Currencies["EUR"]
    assert rate.ccy2 == Currencies["USD"]
    assert rate.date == date_val
    assert rate.value == value_val

def test_fxrate_constructor_tuple_unpacking():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    date_val = datetime.date.today()
    value_val = Decimal("0.85")
    rate = FXRate(Currencies["USD"], Currencies["GBP"], date_val, value_val)
    ccy1, ccy2, date, value = rate
    assert ccy1 == Currencies["USD"]
    assert ccy2 == Currencies["GBP"]
    assert date == date_val
    assert value == value_val

def test_fxrate_constructor_indexed_access():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    date_val = datetime.date.today()
    value_val = Decimal("1.5")
    rate = FXRate(Currencies["JPY"], Currencies["USD"], date_val, value_val)
    assert rate[0] == Currencies["JPY"]
    assert rate[1] == Currencies["USD"]
    assert rate[2] == date_val
    assert rate[3] == value_val
```


# LLM-generated content at query #23
#--------------------------

```python
def test_queries_returns_iterable_of_rates():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        def queries(self, queries, strict=False):
            return [None for _ in queries]

    service = MockFXRateService()
    queries_input = [("USD", "EUR", "2023-01-01"), ("GBP", "JPY", "2023-01-02")]
    results = list(service.queries(queries_input))
    
    assert len(results) == 2
    assert results == [None, None]

def test_queries_with_strict_mode_raises_error():
    class MockStrictFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        def queries(self, queries, strict=False):
            if strict:
                raise ValueError("Lookup error")
            return [None for _ in queries]

    service = MockStrictFXRateService()
    queries_input = [("USD", "EUR", "2023-01-01")]
    
    import pytest
    with pytest.raises(ValueError, match="Lookup error"):
        list(service.queries(queries_input, strict=True))

def test_queries_with_mock_values():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        def queries(self, queries, strict=False):
            return [0.85, 150.0]

    service = MockFXRateService()
    queries_input = [("USD", "EUR", "2023-01-01"), ("GBP", "JPY", "2023-01-02")]
    results = list(service.queries(queries_input))
    
    assert results == [0.85, 150.0]
```


# LLM-generated content at query #24
#--------------------------

```python
def test_fxrate_constructor_assignment():
    import datetime
    from decimal import Decimal
    from typing import NamedTuple
    
    class Currency(str):
        pass

    class FXRate(NamedTuple):
        ccy1: Currency
        ccy2: Currency
        date: datetime.date
        value: Decimal

    ccy1 = Currency("EUR")
    ccy2 = Currency("USD")
    date = datetime.date.today()
    value = Decimal("2.0")
    
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
    from typing import NamedTuple

    class Currency(str):
        pass

    class FXRate(NamedTuple):
        ccy1: Currency
        ccy2: Currency
        date: datetime.date
        value: Decimal

    rate = FXRate(Currency("EUR"), Currency("USD"), datetime.date.today(), Decimal("1.5"))
    ccy1, ccy2, date, value = rate
    
    assert ccy1 == Currency("EUR")
    assert ccy2 == Currency("USD")
    assert value == Decimal("1.5")
```


# LLM-generated content at query #25
#--------------------------

```python
def test_fxrateservice_query_signature_and_return_type_logic():
    from decimal import Decimal
    from datetime import date
    from typing import Optional
    from pypara.currencies import Currency, CurrencyType
    from pypara.exchange import FXRateService

    class MockFXRate:
        def __init__(self, rate: Decimal):
            self.rate = rate

    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: date, strict: bool = False) -> Optional[MockFXRate]:
            if ccy1.code == "USD" and ccy2.code == "EUR":
                return MockFXRate(Decimal("0.85"))
            return None

    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    today = date.today()
    service = MockFXRateService()

    rate_found = service.query(usd, eur, today)
    rate_not_found = service.query(usd, jpy, today)

    assert isinstance(rate_found, MockFXRate)
    assert rate_found.rate == Decimal("0.85")
    assert rate_not_found is None
```


# LLM-generated content at query #26
#--------------------------

```python
from decimal import Decimal
from datetime import date
from typing import Optional, Any
from unittest.mock import MagicMock

def test_fxrate_service_query_returns_rate_when_exists():
    mock_service = MagicMock(spec=FXRateService)
    mock_rate = MagicMock(spec=Any)
    mock_service.query.return_value = mock_rate
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)
    
    result = mock_service.query(usd, eur, asof_date)
    
    assert result == mock_rate
    mock_service.query.assert_called_once_with(usd, eur, asof_date)

def test_fxrate_service_query_returns_none_when_not_found():
    mock_service = MagicMock(spec=FXRateService)
    mock_service.query.return_value = None
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)
    
    result = mock_service.query(usd, eur, asof_date)
    
    assert result is None
    mock_service.query.assert_called_once_with(usd, eur, asof_date)

def test_fxrate_service_query_respects_strict_flag():
    mock_service = MagicMock(spec=FXRateService)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)
    
    mock_service.query(usd, eur, asof_date, strict=True)
    mock_service.query.assert_called_with(usd, eur, asof_date, strict=True)
    
    mock_service.query(usd, eur, asof_date, strict=False)
    mock_service.query.assert_called_with(usd, eur, asof_date, strict=False)
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
from decimal import Decimal
from datetime import date
from typing import Optional
from unittest.mock import MagicMock

def test_fxrate_service_query_returns_rate_when_found():
    mock_service = MagicMock(spec=FXRateService)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)
    expected_rate = MagicMock(spec=FXRate)
    mock_service.query.return_value = expected_rate
    
    result = mock_service.query(usd, eur, asof_date, strict=False)
    
    assert result == expected_rate
    mock_service.query.assert_called_once_with(usd, eur, asof_date, strict=False)

def test_fxrate_service_query_returns_none_when_not_found():
    mock_service = MagicMock(spec=FXRateService)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)
    mock_service.query.return_value = None
    
    result = mock_service.query(usd, eur, asof_date, strict=False)
    
    assert result is None
    mock_service.query.assert_called_once_with(usd, eur, asof_date, strict=False)

def test_fxrate_service_query_with_strict_mode_raises_error():
    mock_service = MagicMock(spec=FXRateService)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)
    mock_service.query.side_effect = LookupError("Rate not found")
    
    try:
        mock_service.query(usd, eur, asof_date, strict=True)
    except LookupError as e:
        assert str(e) == "Rate not found"
    
    mock_service.query.assert_called_once_with(usd, eur, asof_date, strict=True)
```


# LLM-generated content at query #2
#--------------------------

```python
def test_queries_returns_correct_rates_from_mock_implementation():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        def queries(self, queries, strict=False):
            return [Decimal("1.2") if q[0] == "USD" else Decimal("1.0") for q in queries]

    service = MockFXRateService()
    from datetime import date
    queries = [("USD", "EUR", date(2023, 1, 1)), ("GBP", "JPY", date(2023, 1, 1))]
    results = list(service.queries(queries))
    
    assert len(results) == 2
    assert results[0] == Decimal("1.2")
    assert results[1] == Decimal("1.0")

def test_queries_with_strict_true_raises_error_in_mock_implementation():
    class StrictFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        def queries(self, queries, strict=False):
            if strict:
                raise ValueError("Lookup error")
            return [None for _ in queries]

    service = StrictFXRateService()
    from datetime import date
    queries = [("USD", "EUR", date(202im, 1, 1))]
    
    try:
        service.queries(queries, strict=True)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert str(e) == "Lookup error"

def test_queries_handles_empty_input():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        def queries(self, queries, strict=False):
            return []

    service = MockFXRateService()
    results = list(service.queries([]))
    assert results == []
```


# LLM-generated content at query #3
#--------------------------

```python
from decimal import Decimal
from datetime import date
from unittest.mock import MagicMock
from pypara.currencies import Currency, CurrencyType
from pypara.exchange import FXRateService

def test_query_returns_rate_when_found():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)
    mock_rate = MagicMock()
    
    service = MagicMock(spec=FXRateService)
    service.query.return_value = mock_rate
    
    result = service.query(ccy1, ccy2, asof_date, strict=False)
    
    service.query.assert_called_once_with(ccy1, ccy2, asof_date, strict=False)
    assert result == mock_rate

def test_query_returns_none_when_not_found():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)
    
    service = MagicMock(spec=FXRateService)
    service.query.return_value = None
    
    result = service.query(ccy1, ccy2, asof_date, strict=False)
    
    service.query.assert_called_once_with(ccy1, ccy2, asof_date, strict=False)
    assert result is None

def test_query_called_with_strict_true():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY)
    asof_date = date(2023, 5, 20)
    
    service = MagicMock(spec=FXRateService)
    
    service.query(ccy1, ccy2, asof_date, strict=True)
    
    service.query.assert_called_once_with(ccy1, ccy2, asof_date, strict=True)
```


# LLM-generated content at query #4
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
    
    result = service.query(cccy1=ccy1, ccy2=ccy2, asof=asof_date)
    
    assert result == mock_rate
    service.query.assert_called_once_with(ccy1=ccy1, ccy2=ccy2, asof=asof_date)

def test_fxrate_service_query_returns_none_when_not_found():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)
    
    service = MagicMock(spec=FXRateService)
    service.query.return_value = None
    
    result = service.query(ccy1=ccy1, ccy2=ccy2, asof=asof_date)
    
    assert result is None
    service.query.assert_called_once_with(ccy1=ccy1, ccy2=ccy2, asof=asof_date)

def test_fxrate_service_query_respects_strict_parameter():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)
    
    service = MagicMock(spec=FXRateService)
    
    service.query(ccy1=ccy1, ccy2=ccy2, asof=asof_date, strict=True)
    service.query.assert_called_with(ccy1=ccy1, ccy2=ccy2, asof=asof_date, strict=True)
    
    service.query(ccy1=ccy1, ccy2=ccy2, asof=asof_date, strict=False)
    service.query.assert_called_with(ccy1=ccy1, ccy2=ccy2, asof=asof_date, strict=False)
```


# LLM-generated content at query #5
#--------------------------

```python
def test_fxrate_service_query_interface_definition():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from decimal import Decimal
    from datetime import date

    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)
    
    # Since FXRateService is an ABC with abstract methods, 
    # we cannot instantiate it directly. 
    # We test the signature/contract via a mock or a concrete implementation.
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return Decimal("1.10")
        
        def queries(self, queries, strict=False):
            return [Decimal("1.10")]

    service = MockFXRateService()
    result = service.query(usd, eur, asof_date)
    
    assert isinstance(result, Decimal)
    assert result == Decimal("1.10")

def test_fxrate_service_query_parameters_usage():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date

    usd = Currency.of("USD", "US Dollars", 2, CurrencyRateType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)

    class SpyFXRateService(FXRateService):
        def __init__(self):
            self.captured_args = None
            
        def query(self, ccy1, ccy2, asof, strict=False):
            self.captured_args = (ccy1, ccy2, asof, strict)
            return None

        def queries(self, queries, strict=False):
            return []

    service = SpyFXRateService()
    service.query(usd, eur, asof_date, strict=True)
    
    assert service.captured_args[0] == usd
    assert service.captured_args[1] == eur
    assert service.captured_args[2] == asof_date
    assert service.captured_args[3] is True
```


# LLM-generated content at query #6
#--------------------------

```python
from unittest.mock import MagicMock
from decimal import Decimal
from datetime import date

def test_queries_returns_correct_values():
    mock_service = MagicMock(spec=FXRateService)
    query_input = [("USD", "EUR", date(2023, 1, 1)), ("GBP", "JPY", date(2023, 1, 2))]
    expected_output = [Decimal("0.92"), Decimal("160.50")]
    mock_service.queries.return_value = iter(expected_output)

    result = list(mock_service.queries(query_input))

    assert result == expected_output
    mock_service.queries.assert_called_once_with(query_input, strict=False)

def test_queries_with_strict_mode():
    mock_service = MagicMock(spec=FXRateService)
    query_input = [("USD", "CAD", date(202im, 5, 5))]
    mock_service.queries.return_value = iter([Decimal("1.35")])

    result = list(mock_service.queries(query_input, strict=True))

    assert result == [Decimal("1.35")]
    mock_service.queries.assert_called_once_with(query_input, strict=True)

def test_queries_returns_none_for_missing_rate():
    mock_service = MagicMock(spec=FXRateService)
    query_input = [("USD", "XYZ", date(2023, 1, 1))]
    mock_service.queries.return_value = iter([None])

    result = list(mock_service.queries(query_input))

    assert result == [None]
    mock_service.queries.assert_called_once_with(query_input, strict=False)
```


# LLM-generated content at query #7
#--------------------------

```python
from unittest.mock import MagicMock
from decimal import Decimal
from datetime import date

def test_queries_returns_iterable_of_rates():
    mock_service = MagicMock(spec=FXRateService)
    expected_rates = [Decimal("1.2"), Decimal("0.8"), None]
    mock_service.queries.return_value = iter(expected_rates)
    
    queries_input = [("USD", "EUR", date(2023, 1, 1)), ("GBP", "USD", date(2023, 1, 1)), ("JPY", "USD", date(2023, 1, 1))]
    result = list(mock_service.queries(queries_input, strict=False))
    
    assert result == expected_rates
    mock_service.queries.assert_called_once_with(queries_input, strict=False)

def test_queries_with_strict_mode_raises_error():
    mock_service = MagicMock(spec=FXRateService)
    mock_service.queries.side_effect = ValueError("Rate not found")
    
    queries_input = [("USD", "XYZ", date(202im, 1, 1))]
    
    try:
        list(mock_service.queries(queries_input, strict=True))
    except ValueError as e:
        assert str(e) == "Rate not found"
    
    mock_service.queries.assert_called_once_with(queries_input, strict=True)
```


# LLM-generated content at query #8
#--------------------------

```python
def test_FXRate_constructor_valid_assignment():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("2"))
    assert rate.ccy1 == Currencies["EUR"]
    assert rate.ccy2 == Currencies["USD"]
    assert rate.date == datetime.date.today()
    assert rate.value == Decimal("2")

def test_FXRate_constructor_tuple_unpacking():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("1.5"))
    ccy1, ccy2, date, value = rate
    assert ccy1 == Currencies["EUR"]
    assert ccy2 == Currencies["USD"]
    assert date == datetime.date.today()
    assert value == Decimal("1.5")

def test_FXRate_constructor_indexed_access():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    rate = FXRate(Currencies["GBP"], Currencies["JPY"], datetime.date.today(), Decimal("150"))
    assert rate[0] == Currencies["GBP"]
    assert rate[1] == Currencies["JPY"]
    assert rate[2] == datetime.date.today()
    assert rate[3] == Decimal("150")
```


# LLM-generated content at query #9
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
    expected_rates = [Decimal("0.92"), Decimal("1.22")]
    mock_service.queries.return_value = iter(expected_rates)

    results = list(mock_service.queries(query_list))

    assert results == expected_rates
    mock_service.queries.assert_called_once_with(query_list, strict=False)

def test_queries_with_strict_mode_raises_error():
    mock_service = MagicMock(spec=FXRateService)
    query_list = [("USD", "JPY", date(2023, 1, 1))]
    mock_service.queries.side_effect = ValueError("Rate not found")

    try:
        list(mock_service.queries(query_list, strict=True))
    except ValueError as e:
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


# LLM-generated content at query #10
#--------------------------

```python
def test_fxrate_constructor_valid_assignment():
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


# LLM-generated content at query #11
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
    value = Decimal("1.5")
    rate = FXRate(ccy1, ccy2, date, value)
    unpacked = list(rate)
    assert unpacked[0] == ccy1
    assert unpacked[1] == ccy2
    assert unpacked[2] == date
    assert unpacked[3] == value

def test_fxrate_constructor_indexed_access():
    from decimal import Decimal
    import datetime
    from pypara.currencies import Currencies
    ccy1 = Currencies["GBP"]
    ccy2 = Currencies["JPY"]
    date = datetime.date(2023, 1, 1)
    value = Decimal("150.0")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate[0] == ccy1
    assert rate[1] == ccy2
    assert rate[2] == date
    assert rate[3] == value
```


# LLM-generated content at query #12
#--------------------------

```python
def test_fxrate_constructor_valid_data():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currencies
    rate = FXRate(Currencies["EUR"], Currencies["USD"], date(2023, 1, 1), Decimal("1.2"))
    assert rate.ccy1 == Currencies["EUR"]
    assert rate.ccy2 == Currencies["USD"]
    assert rate.date == date(2023, 1, 1)
    assert rate.value == Decimal("1.2")

def test_fxrate_constructor_tuple_access():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currencies
    rate = FXRate(Currencies["EUR"], Currencies["USD"], date(2023, 1, 1), Decimal("1.2"))
    assert rate[0] == Currencies["EUR"]
    assert rate[1] == Currencies["USD"]
    assert rate[2] == date(2023, 1, 1)
    assert rate[3] == Decimal("1.2")
```


# LLM-generated content at query #13
#--------------------------

```python
def test_fxrate_constructor_assignment_and_indexing():
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
    assert rate[0] == ccy1
    assert rate[1] == ccy2
    assert rate[2] == date
    assert rate[3] == value

def test_fxrate_constructor_tuple_unpacking():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("2")
    rate = FXRate(ccy1, ccy2, date, value)
    unpacked_ccy1, unpacked_ccy2, unpacked_date, unpacked_value = rate
    assert unpacked_ccy1 == ccy1
    assert unpacked_ccy2 == ccy2
    assert unpacked_date == date
    assert unpacked_value == value
```


# LLM-generated content at query #14
#--------------------------

```python
def test_fxrate_constructor_valid_input():
    from decimal import Decimal
    from datetime import date
    # Mocking Currency as it is not provided, but assuming it behaves like a string or compatible object
    # In a real scenario, Currencies["EUR"] would be used.
    class MockCurrency:
        def __init__(self, code): self.code = code
        def __eq__(self, other): return self.code == other.code
    
    ccy1 = MockCurrency("EUR")
    ccy2 = MockCurrency("USD")
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
    from decimal import Decimal
    from datetime import date
    class MockCurrency:
        def __init__(self, code): self.code = code
        def __eq__(self, other): return self.code == other.code
    
    ccy1 = MockCurrency("EUR")
    ccy2 = MockCurrency("USD")
    today = date.today()
    val = Decimal("1.5")
    rate = FXRate(ccy1, ccy2, today, val)
    unpacked = tuple(rate)
    assert len(unpacked) == 4
    assert unpacked[0] == ccy1
    assert unpacked[3] == val
```


# LLM-generated content at query #15
#--------------------------

```python
def test_fxrate_constructor_valid_assignment():
    from decimal import Decimal
    from datetime import date
    # Assuming Currency is a type that can be instantiated or mocked
    # For the purpose of this test, we use a dummy class to represent Currency
    class Currency:
        def __init__(self, code):
            self.code = code
        def __eq__(self, other):
            return isinstance(other, Currency) and self.code == other.code

    ccy1 = Currency("EUR")
    ccy2 = Currency("USD")
    dt = date.today()
    val = Decimal("2.0")
    rate = FXRate(ccy1, ccy2, dt, val)
    
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == dt
    assert rate.value == val
    assert rate[0] == ccy1
    assert rate[1] == ccy2
    assert rate[2] == dt
    assert rate[3] == val

def test_fxrate_constructor_untyped_access():
    from decimal import Decimal
    from datetime import date
    class Currency:
        def __init__(self, code):
            self.code = code
    
    ccy1 = Currency("EUR")
    ccy2 = Currency("USD")
    dt = date.today()
    val = Decimal("1.5")
    rate = FXRate(ccy1, ccy2, dt, val)
    
    assert rate.ccy1 == Currency("EUR")
    assert rate.value == Decimal("1.5")
```


# LLM-generated content at query #16
#--------------------------

```python
def test_fxrate_constructor_valid_data():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("2"))
    assert rate.ccy1 == Currencies["EUR"]
    assert rate.ccy2 == Currencies["USD"]
    assert rate.date == datetime.date.today()
    assert rate.value == Decimal("2")

def test_fxrate_constructor_tuple_access():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("2"))
    assert rate[0] == Currencies["EUR"]
    assert rate[1] == Currencies["USD"]
    assert rate[2] == datetime.date.today()
    assert rate[3] == Decimal("2")

def test_fxrate_constructor_equality():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    rate1 = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("2"))
    rate2 = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("2"))
    assert rate1 == rate2
```


# LLM-generated content at query #17
#--------------------------

```python
from unittest.mock import MagicMock
from decimal import Decimal
from datetime import date

def test_queries_returns_correct_rates_for_valid_input():
    mock_service = MagicMock(spec=FXRateService)
    query_list = [
        ("USD", "EUR", date(2023, 1, 1)),
        ("GBP", "JPY", date(2023, 1, 2))
    ]
    expected_rates = [Decimal("0.92"), Decimal("160.50")]
    mock_service.queries.return_value = iter(expected_rates)

    results = list(mock_service.queries(query_list))

    assert results == expected_rates
    mock_service.queries.assert_called_once_with(query_list, strict=False)

def test_queries_returns_none_for_missing_rates():
    mock_service = MagicMock(spec=FXRateService)
    query_list = [("USD", "XYZ", date(2023, 1, 1))]
    expected_rates = [None]
    mock_service.queries.return_value = iter(expected_rates)

    results = list(mock_service.queries(query_list))

    assert results == [None]
    mock_service.queries.assert_called_once_with(query_list, strict=False)

def test_queries_raises_error_when_strict_is_true_and_rate_missing():
    mock_service = MagicMock(spec=FXRateService)
    query_list = [("USD", "XYZ", date(2023, 1, 1))]
    mock_service.queries.side_effect = ValueError("Rate not found")

    try:
        list(mock_service.queries(query_list, strict=True))
    except ValueError as e:
        assert str(e) == "Rate not found"
    
    mock_service.queries.assert_called_once_with(query_list, strict=True)
```


# LLM-generated content at query #18
#--------------------------

```python
from decimal import Decimal
from datetime import date
from typing import Optional
from unittest.mock import MagicMock
from pypara.currencies import Currency, CurrencyType
from pypara.exchange import FXRateService

def test_fxrate_service_query_returns_expected_rate():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)
    expected_rate = Decimal("0.92")
    
    service = MagicMock(spec=FXRateService)
    service.query.return_value = expected_rate
    
    result = service.query(usd, eur, asof_date)
    
    assert result == expected_rate
    service.query.assert_called_once_with(usd, eur, asof_date)

def test_fxrate_service_query_returns_none_when_not_found():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)
    
    service = MagicMock(spec=FXRateService)
    service.query.return_value = None
    
    result = service.query(usd, eur, asof_date)
    
    assert result is None
    service.query.assert_called_once_with(usd, eur, asof_date)

def test_fxrate_service_query_with_strict_flag():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)
    
    service = MagicMock(spec=FXRateService)
    service.query.return_value = Decimal("0.92")
    
    result = service.query(usd, eur, asof_date, strict=True)
    
    assert result == Decimal("0.92")
    service.query.assert_called_once_with(usd, eur, asof_date, strict=True)
```


# LLM-generated content at query #19
#--------------------------

```python
def test_fxrate_constructor_valid_assignment():
    from decimal import Decimal
    import datetime
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("2.0")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value

def test_fxrate_constructor_tuple_indexing():
    from decimal import Decimal
    import datetime
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
```


# LLM-generated content at query #20
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


# LLM-generated content at query #21
#--------------------------

```python
def test_query_interface_definition():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    from decimal import Decimal
    from typing import Optional

    # Since FXRateService is an abstract base class, we test the interface signature via a mock-like implementation
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: date, strict: bool = False) -> Optional[Decimal]:
            return Decimal("1.23")
        
        def queries(self, queries, strict=False):
            return []

    service = MockFXRateService()
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    EUR = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    today = date.today()
    
    result = service.query(USD, EUR, today)
    
    assert isinstance(result, Decimal)
    assert result == Decimal("1.23")

def test_query_returns_none_when_not_found():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date

    class MockNoneFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: date, strict: bool = False) -> None:
            return None
        
        def queries(self, queries, strict=False):
            return []

    service = MockNoneFXRateService()
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    EUR = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    
    result = service.query(USD, EUR, date.today())
    
    assert result is None
```


# LLM-generated content at query #22
#--------------------------

```python
def test_queries_returns_iterable_of_rates_from_concrete_implementation():
    from decimal import Decimal
    from datetime import date
    from typing import Iterable, Optional, Tuple

    class MockFXRate:
        def __init__(self, rate: Decimal):
            self.rate = rate
        def __eq__(self, other):
            return isinstance(other, MockFXRate) and self.rate == other.rate

    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        def queries(self, queries: Iterable[Tuple[str, str, date]], strict: bool = False):
            return [MockFXRate(Decimal("1.2")) if q[0] == "USD" else None for q in queries]

    service = MockFXRateService()
    test_queries = [("USD", "EUR", date(2023, 1, 1)), ("GBP", "JPY", date(2023, 1, 1))]
    expected_results = [MockFXRate(Decimal("1.2")), None]
    
    results = list(service.queries(test_queries))
    
    assert len(results) == 2
    assert results[0] == MockFXRate(Decimal("1.2"))
    assert results[1] is None
```


# LLM-generated content at query #23
#--------------------------

```python
def test_fxrate_constructor_valid_data():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currencies
    rate = FXRate(Currencies["EUR"], Currencies["USD"], date.today(), Decimal("1.2"))
    assert rate.ccy1 == Currencies["EUR"]
    assert rate.ccy2 == Currencies["USD"]
    assert rate.date == date.today()
    assert rate.value == Decimal("1.2")

def test_fxrate_constructor_tuple_access():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currencies
    rate = FXRate(Currencies["EUR"], Currencies["USD"], date.today(), Decimal("1.2"))
    assert rate[0] == Currencies["EUR"]
    assert rate[1] == Currencies["USD"]
    assert rate[2] == date.today()
    assert rate[3] == Decimal("1.2")

def test_fxrate_constructor_identity_property():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currencies
    rate = FXRate(Currencies["EUR"], Currencies["EUR"], date.today(), Decimal("1.0"))
    assert rate.ccy1 == rate.ccy2
    assert rate.value == Decimal("1.0")
```


# LLM-generated content at query #24
#--------------------------

```python
def test_FXRate_constructor_valid_assignment():
    from datetime import date
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    dt = date.today()
    val = Decimal("1.25")
    rate = FXRate(ccy1, ccy2, dt, val)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == dt
    assert rate.value == val

def test_FXRate_constructor_tuple_unpacking():
    from datetime import date
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["GBP"]
    ccy2 = Currencies["JPY"]
    dt = date(2023, 1, 1)
    val = Decimal("150.5")
    rate = FXRate(ccy1, ccy2, dt, val)
    r_ccy1, r_ccy2, r_date, r_value = rate
    assert r_ccy1 == ccy1
    assert r_ccy2 == ccy2
    assert r_date == dt
    assert r_value == val

def test_FXRate_constructor_indexed_access():
    from datetime import date
    from decimal import Decimal
    from pypara.currencies import Currencies
    rate = FXRate(Currencies["USD"], Currencies["EUR"], date.today(), Decimal("0.9"))
    assert rate[0] == Currencies["USD"]
    assert rate[1] == Currencies["EUR"]
    assert rate[2] == date.today()
    assert rate[3] == Decimal("0.9")
```


# LLM-generated content at query #25
#--------------------------

```python
from unittest.mock import MagicMock
from decimal import Decimal
from datetime import date

def test_queries_returns_correct_rates_for_valid_input():
    mock_service = MagicMock(spec=FXRateService)
    query_input = [
        ("USD", "EUR", date(2023, 1, 1)),
        ("GBP", "JPY", date(2023, 1, 2))
    ]
    expected_outputs = [Decimal("0.92"), Decimal("160.50")]
    mock_service.queries.return_value = iter(expected_outputs)

    results = list(mock_service.queries(query_input))

    assert results == expected_outputs
    mock_service.queries.assert_called_once_with(query_input, strict=False)

def test_queries_returns_none_when_rate_not_found():
    mock_service = MagicMock(spec=FXRateService)
    query_input = [("USD", "XYZ", date(2023, 1, 1))]
    expected_outputs = [None]
    mock_service.queries.return_value = iter(expected_outputs)

    results = list(mock_service.queries(query_input))

    assert results == [None]
    mock_service.queries.assert_called_once_with(query_input, strict=False)

def test_queries_with_strict_mode_enabled():
    mock_service = MagicMock(spec=FXRateService)
    query_input = [("USD", "EUR", date(2023, 1, 1))]
    mock_service.queries.return_value = iter([Decimal("0.92")])

    list(mock_service.queries(query_input, strict=True))

    mock_service.queries.assert_called_once_with(query_input, strict=True)
```


# LLM-generated content at query #26
#--------------------------

```python
from decimal import Decimal
from datetime import date
from typing import Optional
from unittest.mock import MagicMock
from pypara.currencies import Currency, CurrencyType
from pypara.exchange import FXRateService

def test_fxrateservice_query_returns_none_when_no_rate_found():
    service = MagicMock(spec=FXRateService)
    service.query.return_value = None
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)
    
    result = service.query(usd, jpy, asof_date)
    
    assert result is None
    service.query.assert_called_once_with(usd, jpy, asof_date)

def test_fxrateservice_query_returns_rate_when_rate_exists():
    service = MagicMock(spec=FXRateService)
    mock_rate = MagicMock()
    service.query.return_value = mock_rate
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)
    
    result = service.query(usd, jpy, asof_date)
    
    assert result == mock_rate
    service.query.assert_called_once_with(usd, jpy, asof_date)

def test_fxrateservice_query_with_strict_parameter_true():
    service = MagicMock(spec=FXRateService)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)
    
    service.query(usd, jpy, asof_date, strict=True)
    
    service.query.assert_called_once_with(usd, jpy, asof_date, strict=True)

def test_fxrateservice_query_with_strict_parameter_false():
    service = MagicMock(spec=FXRateService)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)
    
    service.query(usd, jpy, asof_date, strict=False)
    
    service.query.assert_called_once_with(usd, jpy, asof_date, strict=False)
```


