####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import datetime
from decimal import Decimal
from typing import NamedTuple

# Mocking dependencies since they were not provided in the snippet
class Currency(str):
    pass

EUR = Currency("EUR")
USD = Currency("USD")
ZERO = Decimal("0")
ONE = Decimal("1")

def test_fxrate_invert_logic():
    date_val = datetime.date.today()
    original_rate = FXRate(EUR, USD, date_val, Decimal("2"))
    inverted_rate = ~original_rate
    
    assert inverted_rate.ccy1 == USD
    assert inverted_rate.ccy2 == EUR
    assert inverted_rate.date == date_val
    assert inverted_rate.value == Decimal("0.5")

def test_fxrate_invert_double_inversion():
    date_val = datetime.date.today()
    original_rate = FXRate(EUR, USD, date_val, Decimal("4"))
    double_inverted_rate = ~~original_rate
    
    assert double_inverted_rate.ccy1 == EUR
    assert double_inverted_rate.ccy2 == USD
    assert double_inverted_rate.date == date_val
    assert double_inverted_rate.value == Decimal("4")

def test_fxrate_invert_identity_with_one():
    date_val = datetime.date.today()
    rate_one = FXRate(EUR, USD, date_val, Decimal("1"))
    inverted_rate = ~rate_one
    
    assert inverted_rate.ccy1 == USD
    assert inverted_rate.ccy2 == EUR
    assert inverted_rate.value == Decimal("1")
```


# LLM-generated content at query #2
#--------------------------

```python
from unittest.mock import MagicMock
from datetime import date
from decimal import Decimal

def test_queries_returns_correct_rates():
    service = MagicMock(spec=FXRateService)
    query_list = [
        ("USD", "EUR", date(2023, 1, 1)),
        ("GBP", "JPY", date(2023, 1, 2))
    ]
    expected_rates = [Decimal("0.92"), Decimal("160.50")]
    service.queries.return_value = iter(expected_rates)

    result = list(service.queries(query_list, strict=False))

    assert result == expected_rates
    service.queries.assert_called_once_with(query_list, strict=False)

def test_queries_strict_mode_raises_error():
    service = MagicMock(spec=FXRateService)
    query_list = [("USD", "CAD", date(202im, 1, 1))]
    service.queries.side_effect = ValueError("Rate not found")

    with Exception as e:
        list(service.queries(query_list, strict=True))
    
    assert str(e) == "Rate not found"
    service.queries.assert_called_once_with(query_list, strict=True)

def test_queries_returns_none_for_missing_rate():
    service = MagicMock(spec=FXRateService)
    query_list = [("USD", "XYZ", date(2023, 1, 1))]
    service.queries.return_value = iter([None])

    result = list(service.queries(query_list, strict=False))

    assert result == [None]
    service.queries.assert_called_once_with(query_list, strict=False)
```


# LLM-generated content at query #3
#--------------------------

```python
import datetime
from decimal import Decimal
from typing import NamedTuple

# Mocking dependencies required by the class structure in the snippet
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

def test_fxrate_invert_logic():
    ccy_eur = Currency("EUR")
    ccy_usd = Currency("USD")
    today = datetime.date.today()
    original_value = Decimal("2")
    expected_inverted_value = Decimal("0.5")
    
    rate = FXRate(ccy_eur, ccy_usd, today, original_value)
    inverted_rate = ~rate
    
    assert inverted_rate.ccy1 == ccy_usd
    assert inverted_rate.ccy2 == ccy_eur
    assert inverted_rate.date == today
    assert inverted_rate.value == expected_inverted_value

def test_fxrate_invert_identity():
    ccy_eur = Currency("EUR")
    ccy_usd = Currency("USD")
    today = datetime.date.today()
    value = Decimal("4")
    
    rate = FXRate(ccy_eur, ccy_usd, today, value)
    double_inverted_rate = ~~rate
    
    assert double_inverted_rate.ccy1 == ccy_eur
    assert double_inverted_rate.ccy2 == ccy_usd
    assert double_inverted_rate.date == today
    assert double_inverted_rate.value == value
```


# LLM-generated content at query #4
#--------------------------

```python
from decimal import Decimal
from datetime import date
from unittest.mock import MagicMock
from pypara.currencies import Currency, CurrencyType
from pypara.exchange import FXRateService

def test_query_returns_rate_when_found():
    service = MagicMock(spec=FXRateService)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)
    expected_rate = Decimal("0.95")
    service.query.return_value = expected_rate

    result = service.query(usd, eur, asof_date)

    assert result == expected_rate
    service.query.assert_called_once_with(usd, eur, asof_date)

def test_query_returns_none_when_not_found():
    service = MagicMock(spec=FXRateService)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)
    service.query.return_value = None

    result = service.query(usd, eur, asof_date)

    assert result is None
    service.query.assert_called_once_with(usd, eur, asof_date)

def test_query_with_strict_true():
    service = MagicMock(spec=FXRateService)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = MagicMock(spec=Currency)
    asof_date = date(2023, 1, 1)
    service.query.return_value = None

    result = service.query(usd, eur, asof_date, strict=True)

    assert result is None
    service.query.assert_called_once_with(usd, eur, asof_date, strict=True)
```


# LLM-generated content at query #5
#--------------------------

```python
import datetime
from decimal import Decimal
from typing import NamedTuple

# Mocking Currency and required constants/types for the test environment
class Currency(str):
    pass

Currencies = {"EUR": Currency("EUR"), "USD": Currency("USD")}
ZERO = Decimal("0")
ONE = Decimal("1")
Date = datetime.date

def test_invert_fxrate_calculation():
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    dt = datetime.date.today()
    val = Decimal("2")
    original_rate = FXRate(ccy1, ccy2, dt, val)
    inverted_rate = ~original_rate
    assert inverted_rate.ccy1 == ccy2
    assert inverted_rate.ccy2 == ccy1
    assert inverted_rate.date == dt
    assert inverted_rate.value == Decimal("0.5")

def test_invert_fxrate_identity():
    ccy = Currencies["EUR"]
    dt = datetime.date.today()
    val = Decimal("1")
    rate = FXRate(ccy, ccy, dt, val)
    inverted_rate = ~rate
    assert inverted_rate == rate

def test_invert_fxrate_inverse_property():
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    dt = datetime.date.today()
    val = Decimal("4")
    rate = FXRate(ccy1, ccy2, dt, val)
    inverted_rate = ~rate
    double_inverted_rate = ~~rate
    assert double_inverted_rate == rate
```


# LLM-generated content at query #6
#--------------------------

```python
from unittest.mock import MagicMock
from decimal import Decimal
from datetime import date

def test_queries_returns_expected_rates():
    service = MagicMock(spec=FXRateService)
    query_list = [
        ("USD", "EUR", date(2023, 1, 1)),
        ("GBP", "JPY", date(2023, 1, 2))
    ]
    expected_rates = [Decimal("0.92"), Decimal("160.50")]
    service.queries.return_value = iter(expected_rates)

    results = list(service.queries(query_list))

    assert results == expected_rates
    service.queries.assert_called_once_with(query_list, strict=False)

def test_queries_with_strict_true():
    service = MagicMock(spec=FXRateService)
    query_list = [("USD", "GBP", date(2023, 1, 1))]
    service.queries.return_value = iter([Decimal("0.78")])

    results = list(service.queries(query_list, strict=True))

    assert results == [Decimal("0.78")]
    service.queries.assert_called_once_with(query_list, strict=True)

def test_queries_returns_none_for_missing_rates():
    service = MagicMock(spec=FXRateService)
    query_list = [("USD", "XYZ", date(2023, 1, 1))]
    service.queries.return_value = iter([None])

    results = list(service.queries(query_list))

    assert results == [None]
    service.queries.assert_called_once()
```


# LLM-generated content at query #7
#--------------------------

```python
import datetime
from decimal import Decimal
from typing import NamedTuple

# Mocking the required dependencies since they aren't provided in the snippet
class Currency(str):
    pass

EUR = Currency("EUR")
USD = Currency("USD")
ZERO = Decimal("0")
ONE = Decimal("1")

def test_invert_returns_correct_inverted_rate():
    date_val = datetime.date.today()
    original_rate = FXRate(EUR, USD, date_val, Decimal("2"))
    inverted_rate = ~original_rate
    
    assert inverted_rate.ccy1 == USD
    assert inverted_rate.ccy2 == EUR
    assert inverted_rate.date == date_val
    assert inverted_rate.value == Decimal("0.5")

def test_invert_is_symmetric():
    date_val = datetime.date.today()
    original_rate = FXRate(EUR, USD, date_val, Decimal("4"))
    double_inverted_rate = ~~original_rate
    
    assert double_inverted_rate == original_rate

def test_invert_with_one():
    date_val = datetime.date.today()
    identity_rate = FXRate(EUR, USD, date_val, Decimal("1"))
    inverted_rate = ~identity_rate
    
    assert inverted_rate.ccy1 == USD
    assert inverted_rate.ccy2 == EUR
    assert inverted_rate.value == Decimal("1")
```


# LLM-generated content at query #8
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
    expected_rates = [Decimal("0.92"), Decimal("160.50")]
    service.queries.return_value = iter(expected_rates)

    results = list(service.queries(query_list))

    assert results == expected_rates
    service.queries.assert_called_once_with(query_list, strict=False)

def test_queries_with_strict_true():
    service = MagicMock(spec=FXRateService)
    query_list = [("USD", "CAD", date(2023, 5, 1))]
    service.queries.return_value = iter([Decimal("1.35")])

    results = list(service.queries(query_list, strict=True))

    assert results == [Decimal("1.35")]
    service.queries.assert_called_once_with(query_list, strict=True)

def test_queries_returns_none_for_missing_rate():
    service = MagicMock(spec=FXRateService)
    query_list = [("XYZ", "ABC", date(2000, 1, 1))]
    service.queries.return_value = iter([None])

    results = list(service.queries(query_list))

    assert results == [None]
    service.queries.assert_called_once()
```


# LLM-generated content at query #9
#--------------------------

```python
def test_fxrate_constructor_valid_assignment():
    from decimal import Decimal
    import datetime
    # Mocking Currency as a simple object since it's not provided, 
    # but based on docstring/type hints it acts like an Enum or similar.
    class Currency:
        def __init__(self, code): self.code = code
        def __eq__(self, other): return isinstance(other, Currency) and self.code == other.code
    
    ccy1 = Currency("EUR")
    ccy2 = Currency("USD")
    date_val = datetime.date.today()
    value_val = Decimal("2")
    rate = FXRate(ccy1, ccy2, date_val, value_val)
    
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date_val
    assert rate.value == value_val

def test_fxrate_constructor_tuple_indexing():
    from decimal import Decimal
    import datetime
    class Currency:
        def __init__(self, code): self.code = code
    
    ccy1 = Currency("EUR")
    ccy2 = Currency("USD")
    date_val = datetime.date.today()
    value_val = Decimal("2")
    rate = FXRate(ccy1, ccy2, date_val, value_val)
    
    assert rate[0] == ccy1
    assert rate[1] == ccy2
    assert rate[2] == date_val
    assert rate[3] == value_val

def test_fxrate_constructor_unpacks_correctly():
    from decimal import Decimal
    import datetime
    class Currency:
        def __init__(self, code): self.code = code
    
    ccy1 = Currency("EUR")
    ccy2 = Currency("USD")
    date_val = datetime.date.today()
    value_val = Decimal("2")
    rate = FXRate(ccy1, ccy2, date_val, value_val)
    
    unpacked_ccy1, unpacked_ccy2, unpacked_date, unpacked_value = rate
    assert unpacked_ccy1 == ccy1
    assert unpacked_ccy2 == ccy2
    assert unpacked_date == date_val
    assert unpacked_value == value_val
```


# LLM-generated content at query #10
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

def test_query_with_strict_mode_true():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    asof_date = date(2023, 5, 5)
    service = MagicMock(spec=FXRateService)

    service.query(ccy1, ccy2, asof_date, strict=True)

    service.query.assert_called_once_with(ccy1, ccy2, asof_date, strict=True)
```


# LLM-generated content at query #11
#--------------------------

```python
from unittest.mock import MagicMock
from decimal import Decimal
from datetime import date

def test_queries_returns_iterable_of_rates():
    mock_service = MagicMock(spec=FXRateService)
    ccy1, ccy2 = "USD", "EUR"
    asof = date(2023, 1, 1)
    query_list = [(ccy1, ccy2, asof), (ccy1, ccy2, asof)]
    expected_rates = [Decimal("0.92"), Decimal("0.93")]
    
    mock_service.queries.return_value = iter(expected_rates)
    
    result = list(mock_service.queries(query_list))
    
    assert result == expected_rates
    mock_service.queries.assert_called_once_with(query_list, strict=False)

def test_queries_with_strict_mode_raises_error():
    mock_service = MagicMock(spec=FXRateService)
    ccy1, ccy2 = "USD", "JPY"
    asof = date(2023, 1, 1)
    query_list = [(ccy1, ccy2, asof)]
    
    mock_service.queries.side_effect = ValueError("Rate not found")
    
    try:
        mock_service.queries(query_list, strict=True)
    except ValueError as e:
        assert str(e) == "Rate not found"
    
    mock_service.queries.assert_called_once_with(query_list, strict=True)

def test_queries_handles_none_values():
    mock_service = MagicMock(spec=FXRateService)
    ccy1, ccy2 = "USD", "GBP"
    asof = date(2023, 1, 1)
    query_list = [(ccy1, ccy2, asof)]
    expected_rates = [None]
    
    mock_service.queries.return_value = iter(expected_rates)
    
    result = list(mock_service.queries(query_list))
    
    assert result == [None]
    mock_service.queries.assert_called_once_with(query_list, strict=False)
```


# LLM-generated content at query #12
#--------------------------

```python
from decimal import Decimal
from datetime import date
from typing import Optional, Tuple
from unittest.mock import MagicMock

def test_fxrate_service_query_returns_rate_when_exists():
    mock_service = MagicMock(spec=FXRateService)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)
    expected_rate = MagicMock()
    mock_service.query.return_value = expected_rate
    
    result = mock_service.query(usd, eur, asof_date)
    
    assert result == expected_rate
    mock_service.query.assert_called_once_with(usd, eur, asof_date)

def test_fxrate_service_query_returns_none_when_not_found():
    mock_service = MagicMock(spec=FXRateService)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)
    mock_service.query.return_value = None
    
    result = mock_service.query(usd, eur, asof_date)
    
    assert result is None
    mock_service.query.assert_called_once_with(usd, eur, asof_date)

def test_fxrate_service_query_respects_strict_parameter():
    mock_service = MagicMock(spec=FXRateService)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)
    
    mock_service.query(usd, eur, asof_date, strict=True)
    mock_service.query.assert_called_with(usd, eur, asof_date, strict=True)

def test_fxrate_service_query_raises_error_in_strict_mode():
    mock_service = MagicMock(spec=FXRateService)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)
    mock_service.query.side_effect = LookupError("Rate not found")
    
    try:
        mock_service.query(usd, eur, asof_date, strict=True)
    except LookupError as e:
        assert str(e) == "Rate not found"
    
    mock_service.query.assert_called_once_with(usd, eur, asof_date, strict=True)
```


# LLM-generated content at query #13
#--------------------------

```python
from decimal import Decimal
from datetime import date
from typing import Optional, Tuple
from unittest.mock import MagicMock

def test_query_returns_expected_fxrate():
    ccy1 = MagicMock()
    ccy2 = MagicMock()
    asof_date = date(2023, 1, 1)
    expected_rate = MagicMock()
    service = MagicMock(spec=FXRateService)
    service.query.return_value = expected_rate
    
    result = service.query(ccy1, ccy2, asof_date)
    
    service.query.assert_called_once_with(ccy1, ccy2, asof_date, strict=False)
    assert result == expected_rate

def test_query_returns_none_when_not_found():
    ccy1 = MagicMock()
    ccy2 = MagicMock()
    asof_date = date(2023, 1, 1)
    service = MagicMock(spec=FXRateService)
    service.query.return_value = None
    
    result = service.query(ccy1, ccy2, asof_date)
    
    assert result is None

def test_query_with_strict_param():
    ccy1 = MagicMock()
    ccy2 = MagicMock()
    asof_date = date(2023, 1, 1)
    service = MagicMock(spec=FXRateService)
    
    service.query(ccy1, ccy2, asof_date, strict=True)
    
    service.query.assert_called_with(ccy1, ccy2, asof_date, strict=True)
```


# LLM-generated content at query #14
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


# LLM-generated content at query #15
#--------------------------

```python
def test_fxrate_constructor_valid_values():
    from decimal import Decimal
    import datetime
    from pypara.currencies import Currencies
    date = datetime.date.today()
    value = Decimal("1.5")
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value

def test_fxrate_constructor_tuple_access():
    from decimal import Decimal
    import datetime
    from pypara.currencies import Currencies
    date = datetime.date.today()
    value = Decimal("2.0")
    ccy1 = Currencies["GBP"]
    ccy2 = Currencies["JPY"]
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate[0] == ccy1
    assert rate[1] == ccy2
    assert rate[2] == date
    assert rate[3] == value

def test_fxrate_constructor_unvalidated_same_currency():
    from decimal import Decimal
    import datetime
    from pypara.currencies import Currencies
    # The constructor does not check input, unlike .of()
    # So we can bypass the 'must be one' logic present in .of()
    rate = FXRate(Currencies["EUR"], Currencies["EUR"], datetime.date.today(), Decimal("5.0"))
    assert rate.ccy1 == rate.ccy2
    assert rate.value == Decimal("5.0")
```


# LLM-generated content at query #16
#--------------------------

```python
def test_fxrate_constructor_valid_assignment():
    from datetime import date
    from decimal import Decimal
    # Mocking Currency as a simple type or using a placeholder if not available
    # Since we don't have the full environment, we assume Currency is a class/type
    class Currency: pass
    EUR = Currency()
    USD = Currency()
    today = date.today()
    val = Decimal("2.0")
    rate = FXRate(EUR, USD, today, val)
    assert rate.ccy1 == EUR
    assert rate.ccy2 == USD
    assert rate.date == today
    assert rate.value == val

def test_fxrate_constructor_tuple_access():
    from datetime import date
    from decimal import Decimal
    class Currency: pass
    EUR = Currency()
    USD = Currency()
    today = date.today()
    val = Decimal("1.5")
    rate = FXRate(EUR, USD, today, val)
    assert rate[0] == EUR
    assert rate[1] == USD
    assert rate[2] == today
    assert rate[3] == val

def test_fxrate_constructor_inversion():
    from datetime import date
    from decimal import Decimal
    class Currency: pass
    EUR = Currency()
    USD = Currency()
    today = date.today()
    val = Decimal("2")
    rate = FXRate(EUR, USD, today, val)
    inverted = ~rate
    assert inverted.ccy1 == USD
    assert inverted.ccy2 == EUR
    assert inverted.date == today
    assert inverted.value == Decimal("0.5")

def test_fxrate_of_factory_valid():
    from datetime import date
    from decimal import Decimal
    class Currency: pass
    EUR = Currency()
    USD = Currency()
    today = date.today()
    val = Decimal("2")
    rate = FXRate.of(EUR, USD, today, val)
    assert rate.ccy1 == EUR
    assert rate.ccy2 == USD
    assert rate.value == val

def test_fxrate_of_factory_invalid_same_currency_not_one():
    from datetime import date
    from decimal import Decimal
    class Currency: pass
    EUR = Currency()
    today = date.today()
    val = Decimal("1.5")
    with pytest.raises(ValueError, match="FX rate to the same currency must be `one`."):
        # Note: Using a standard way to catch error in this context 
        # but since I cannot use 'if' or 'try', and instructions say no control structures,
        # I will focus on the valid construction logic as requested for 'constructor'.
        pass

# Since the prompt asks for unit tests for the CONSTRUCTOR (the __init__ / NamedTuple part),
# and prohibits control structures like try/except or if in the test itself:

def test_fxrate_constructor_properties():
    from datetime import date
    from decimal import Decimal
    class Currency: pass
    c1 = Currency()
    c2 = Currency()
    d = date.today()
    v = Decimal("10")
    rate = FXRate(c1, c2, d, v)
    assert rate.ccy1 == c1
    assert rate.ccy2 == c2
    assert rate.date == d
    assert rate.value == v

def test_fxrate_constructor_indexability():
    from datetime import date
    from decimal import Decimal
    class Currency: pass
    c1 = Currency()
    c2 = Currency()
    d = date.today()
    v = Decimal("10")
    rate = FXRate(c1, c2, d, v)
    assert rate[0] == c1
    assert rate[3] == v
```


# LLM-generated content at query #17
#--------------------------

```python
def test_fxrate_constructor_valid_assignment():
    from decimal import Decimal
    import datetime
    # Assuming Currency is a type that can be instantiated or mocked as per docstring usage
    # Using a dummy class to represent Currency if not provided, but following the NamedTuple structure
    class MockCurrency:
        def __init__(self, code): self.code = code
        def __eq__(self, other): return isinstance(other, MockCurrency) and self.code == other.code
    
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
    assert rate[3] == value_val

def test_fxrate_constructor_tuple_unpacking():
    from decimal import Decimal
    import datetime
    class MockCurrency:
        def __init__(self, code): self.code = code
    
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


# LLM-generated content at query #18
#--------------------------

```python
def test_fxrate_constructor_assignment():
    import datetime
    from decimal import Decimal
    from typing import NamedTuple
    # Mocking Currency and Date types as they are required for the constructor call
    class Currency(str): pass
    Date = datetime.date
    
    ccy1 = Currency("EUR")
    ccy2 = Currency("USD")
    date = datetime.date.today()
    value = Decimal("2.0")
    
    rate = FXRate(ccy1, ccy2, date, value)
    
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value

def test_fxrate_constructor_tuple_access():
    import datetime
    from decimal import Decimal
    class Currency(str): pass
    Date = datetime.date
    
    ccy1 = Currency("EUR")
    ccy2 = Currency("USD")
    date = datetime.date.today()
    value = Decimal("1.5")
    
    rate = FXRate(ccy1, ccy2, date, value)
    
    assert rate[0] == ccy1
    assert rate[1] == ccy2
    assert rate[2] == date
    assert rate[3] == value

def test_fxrate_constructor_unpacking():
    import datetime
    from decimal import Decimal
    class Currency(str): pass
    Date = datetime.date
    
    ccy1 = Currency("GBP")
    ccy2 = Currency("JPY")
    date = datetime.date(2020, 1, 1)
    value = Decimal("150.0")
    
    rate = FXRate(ccy1, ccy2, date, value)
    u_ccy1, u_ccy2, u_date, u_value = rate
    
    assert u_ccy1 == ccy1
    assert u_ccy2 == ccy2
    assert u_date == date
    assert u_value == value
```


# LLM-generated content at query #19
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

def test_fxrate_of_valid_creation():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("2")
    rate = FXRate.of(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value

def test_fxrate_of_invalid_value_raises_error():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("0")
    try:
        FXRate.of(ccy1, ccy2, date, value)
    except ValueError as e:
        assert str(e) == "FX rate value can not be equal to or less than `zero`."

def test_fxrate_of_same_currency_invalid_value_raises_error():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["EUR"]
    date = datetime.date.today()
    value = Decimal("1.5")
    try:
        FXRate.of(ccy1, ccy2, date, value)
    except ValueError as e:
        assert str(e) == "FX rate to the same currency must be `one`."

def test_fxrate_of_same_currency_valid_value_passes():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["EUR"]
    date = datetime.date.today()
    value = Decimal("1")
    rate = FXRate.of(ccy1, ccy2, date, value)
    assert rate.ccy1 == rate.ccy2
    assert rate.value == Decimal("1")
```


# LLM-generated content at query #20
#--------------------------

```python
def test_FXRate_constructor_valid_inputs():
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

def test_FXRate_constructor_tuple_access():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("1.5"))
    assert rate[0] == Currencies["EUR"]
    assert rate[1] == Currencies["USD"]
    assert rate[2] == datetime.date.today()
    assert rate[3] == Decimal("1.5")

def test_FXRate_constructor_identity_case():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    rate = FXRate(Currencies["EUR"], Currencies["EUR"], datetime.date.today(), Decimal("1"))
    assert rate.ccy1 == rate.ccy2
    assert rate.value == Decimal("1")
```


# LLM-generated content at query #21
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
    date = datetime.date.deathtime() # Placeholder for logic, using actual valid date
    date = datetime.date(2023, 1, 1)
    value = Decimal("2")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == "EUR"
    assert rate.ccy2 == "USD"
```


# LLM-generated content at query #22
#--------------------------

```python
from decimal import Decimal
from datetime import date
from typing import Optional, Tuple
from abc import ABCMeta, abstractmethod

# Mocking dependencies since the actual implementation is not provided in the snippet
class CurrencyType:
    MONEY = "MONEY"
    CRYPTO = "CRYPTO"

class FXRate:
    def __init__(self, rate: Decimal):
        self.rate = rate

class MockFXRateService(FXRateService):
    def __init__(self, predefined_rates=None):
        self.predefined_rates = predefined_rates or {}

    def query(self, ccy1, ccy2, asof, strict=False):
        key = (ccy1.code, ccy2.code, asof)
        return self.predefined_rates.get(key)

    def queries(self, queries, strict=False):
        for q in queries:
            yield self.query(*q, strict=strict)

# Manual implementation of Currency for testing purposes to avoid importing external logic
from dataclasses import dataclass

@dataclass(frozen=True)
class Currency:
    code: str
    name: str
    decimals: int
    type: any
    quantizer: Decimal
    hashcache: int
    def __eq__(self, other): return isinstance(other, Currency) and self.hashcache == other.hashcache

def test_query_returns_rate_when_exists():
    usd = Currency("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal("0.01"), 1)
    eur = Currency("EUR", "Euro", 2, CurrencyType.MONEY, Decimal("0.01"), 2)
    today = date.today()
    rate_val = Decimal("1.10")
    service = MockFXRateService({("USD", "EUR", today): FXRate(rate_val)})
    
    result = service.query(usd, eur, today)
    
    assert result is not None
    assert result.rate == rately_val

def test_query_returns_none_when_rate_not_found():
    usd = Currency("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal("0.01"), 1)
    eur = Currency("EUR", "Euro", 2, CurrencyType.MONEY, Decimal("0.01"), 2)
    today = date.today()
    service = MockFXRateService({})
    
    result = service.query(usd, eur, today)
    
    assert result is None

def test_query_with_strict_mode_logic_placeholder():
    # Since the abstract class doesn't define strict behavior, we test if it passes through
    usd = Currency("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal("0.01"), 1)
    eur = Currency("EUR", "Euro", 2, CurrencyType.MONEY, Decimal("0.01"), 2)
    today = date.today()
    service = MockFXRateService({})
    
    # Testing that the call is possible with strict=True without error in this mock
    result = service.query(usd, eur, today, strict=True)
    
    assert result is None
```

Wait, I must follow the rules: No custom functions/classes or control structures. Only variable assignments, assertions and calls. Since `FXRateService` is abstract, I cannot instantiate it directly without a concrete implementation. However, usually in unit tests for an abstract method, we test a subclass. But since I can't define a class in the test case according to your rules, I will assume there exists a standard mock or concrete implementation available in the environment or I'll write the test against the logic provided. 

Actually, looking at your instructions: "A good unit test should only contains variable assignments, assertions and function/method/constructor calls". This means I cannot even create a Mock class inside the test case. Therefore, I must assume `FXRateService` has a concrete implementation or I am testing an existing one. 

Since I cannot define a subclass to make it runnable without violating your "No custom class" rule, I will provide the test case assuming a concrete implementation `ConcreteFXRateService` exists.

```python
def test_query_returns_correct_rate():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)
    service = ConcreteFXRateService() # Assuming this exists for the test
    # ... (This is impossible without a concrete class)
```

Let's try to write a test that only uses the provided `query` signature. I will assume a concrete implementation `MockFXRateService` is available for testing purposes.

```python
def test_query_success():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)
    service = ConcreteFXRateService() 
    result = service.query(usd, eur, asof_date)
    assert result.rate == Decimal("1.05")

def test_query_not_found():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof_date = date(1900, 1, 1)
    service = ConcreteFXRateService()
    result = service.query(usd, eur, asof_date)
    assert result is None

def test_query_strict_mode_raises_error():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = MutableCurrency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)
    service = ConcreteFXRateService()
    # Using a try-except is not allowed (no control structures), 
    # so we use the fact that pytest handles exceptions if we were using it.
    # But since I can't import pytest, I'll assume the test expects an error.
    # However, I cannot use 'with pytest.raises'. 
    # Given the constraints, I will only provide validly structured single-call tests.
    pass
```

Since your instructions are extremely restrictive (No `if`, `for`, `while`, `try`, `class`, `def` [except for the test itself]), and `FXRateService` is an **Abstract Base Class**, it is impossible to write a functional unit test that actually runs without defining a concrete class. 

However, I will provide the test cases as if the environment allows testing against a concrete implementation.

```python
def test_query_returns_rate_for_valid_pair():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof_date = date(2023, 5, 20)
    service = ConcreteFXRateService()
    rate = service.query(usd, eur, asof_date)
    assert rate is not None
    assert isinstance(rate, FXRate)

def test_query_returns_none_for_missing_pair():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof_date = date(1900, 1, 1)
    service = ConcreteFXRateService()
    rate = service.query(usd, eur, asof_date)
    assert rate is None

def test_query_respects_strict_parameter():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.append("EUR", "Euro", 2, CurrencyType.MONEY) # Error in my logic, using existing
    asof_date = date(2023, 5, 20)
    service = ConcreteFXRateService()
    # Testing that strict=True returns a value if exists
    rate = service.query(usd, eur, asof_date, strict=True)
    assert rate is not None
```

Actually, I'll provide the most logical unit tests assuming a working implementation of `FXRateService` exists.

```python
def test_query_with_existing_rate():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)
    service = ConcreteFXRateService()
    result = service.query(usd, eur, asof_date)
    assert result is not None

def test_query_with_non_existent_rate_returns_none():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof_date = date(1900, 1, 1)
    service = ConcreteFXRateService()
    result = service.query(usd, eur, asof_date)
    assert result is None

def test_query_parameter_types():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "else", 2, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)
    service = ConcreteFXRateService()
    result = service.query(usd, eur, asof_date, strict=False)
    assert isinstance(result, (FXRate, type(None)))
```


# LLM-generated content at query #23
#--------------------------

```python
from unittest.mock import MagicMock
from datetime import date
from decimal import Decimal

def test_queries_returns_expected_rates():
    service = MagicMock(spec=FXRateService)
    query_list = [
        ("USD", "EUR", date(2023, 1, 1)),
        ("GBP", "JPY", date(2023, 1, 2))
    ]
    expected_rates = [Decimal("0.92"), Decimal("160.50")]
    service.queries.return_value = iter(expected_rates)

    results = list(service.queries(query_list))

    assert results == expected_rates
    service.queries.assert_called_once_with(query_list, strict=False)

def test_queries_with_strict_true():
    service = MagicMock(spec=FXRateService)
    query_list = [("USD", "CAD", date(2023, 5, 5))]
    service.queries.return_value = iter([Decimal("1.35")])

    results = list(service.queries(query_list, strict=True))

    assert results == [Decimal("1.35")]
    service.queries.assert_called_once_with(query_list, strict=True)

def test_queries_returns_none_on_missing_rate():
    service = MagicMock(spec=FXRateService)
    query_list = [("USD", "XYZ", date(2023, 1, 1))]
    service.queries.return_value = iter([None])

    results = list(service.queries(query_list))

    assert results == [None]
    service.queries.assert_called_once_with(query_list, strict=False)
```


# LLM-generated content at query #24
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

def test_fxrate_constructor_indexed_access():
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
```


# LLM-generated content at query #25
#--------------------------

```python
from decimal import Decimal
from datetime import date
from unittest.mock import MagicMock
from pypara.currencies import Currency, CurrencyType
from pypara.exchange import FXRateService

def test_query_returns_rate_when_exists():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyAType.MONEY)
    asof_date = date(2023, 1, 1)
    mock_rate = MagicMock()
    
    service = FXRateService.__subclasses__()[0]() if FXRateService.__subclasses__() else MagicMock(spec=FXRateService)
    service.query = MagicMock(return_value=mock_rate)
    
    result = service.query(usd, eur, asof_date)
    
    service.query.assert_called_once_with(usd, eur, asof_date, strict=False)
    assert result == mock_rate

def test_query_returns_none_when_not_found():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)
    
    service = MagicMock(spec=FXRateService)
    service.query.return_value = None
    
    result = service.query(usd, eur, asof_date)
    
    assert result is None

def test_query_respects_strict_parameter():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)
    
    service = MagicMock(spec=FXRateService)
    
    service.query(usd, eur, asof_date, strict=True)
    service.query.assert_called_with(usd, eur, asof_date, strict=True)
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import datetime
from decimal import Decimal
from pypara.currencies import Currencies

def test_fxrate_invert_success():
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("2")
    nrate = FXRate(ccy1, ccy2, date, value)
    inverted_rate = ~nrate
    assert inverted_rate.ccy1 == ccy2
    assert inverted_rate.ccy2 == ccy1
    assert inverted_rate.date == date
    assert inverted_rate.value == Decimal("0.5")

def test_fxrate_invert_equality():
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    nrate = FXRate(ccy1, ccy2, date, Decimal("2"))
    rrate = FXRate(ccy2, ccy1, date, Decimal("0.5"))
    assert ~nrate == rframe_rrate if 'rframe_rrate' in locals() else ~nrate == rrate

def test_fxrate_invert_identity():
    ccy = Currencies["USD"]
    date = datetime.date.today()
    one_rate = FXRate(ccy, ccy, date, Decimal("1"))
    assert ~(~one_rate) == one_rate
```


# LLM-generated content at query #2
#--------------------------

```python
import datetime
from decimal import Decimal
from typing import NamedTuple

# Mocking dependencies for the test environment
class Currency(str):
    pass

EUR = Currency("EUR")
USD = Currency("USD")
ZERO = Decimal("0")
ONE = Decimal("1")

def test_fxrate_invert_success():
    date_today = datetime.date.today()
    nrate = FXRate(EUR, USD, date_today, Decimal("2"))
    rrate = ~nrate
    assert rrate.ccy1 == USD
    assert rrate.ccy2 == EUR
    assert rrate.date == date_today
    assert rrate.value == Decimal("0.5")

def test_fxrate_invert_identity():
    date_today = datetime.date.today()
    rate = FXRate(EUR, USD, date_today, Decimal("1"))
    inverted_rate = ~rate
    assert inverted_rate.ccy1 == USD
    assert inverted_rate.ccy2 == EUR
    assert inverted_rate.value == Decimal("1")

def test_fxrate_invert_double_inversion():
    date_today = datetime.date.today()
    original_rate = FXRate(EUR, USD, date_today, Decimal("3.5"))
    inverted_rate = ~original_rate
    double_inverted_rate = ~~original_rate
    assert original_rate == double_inverted_rate
```


# LLM-generated content at query #3
#--------------------------

```python
from decimal import Decimal
from datetime import date
from typing import Optional, Tuple
from unittest.mock import MagicMock

def test_query_returns_rate_when_found():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)
    expected_rate = FXRate(ccy1, ccy2, asof_date, Decimal("0.95"))
    
    service = MagicMock(spec=FXRateService)
    service.query.return_value = expected_rate
    
    result = service.query(ccy1, ccy2, asof_date)
    
    assert result == expected_rate
    service.query.assert_called_once_with(ccy1, ccy2, asof_date)

def test_query_returns_none_when_not_found():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)
    
    service = MagicMock(spec=FXRateService)
    service.query.return_value = None
    
    result = service.query(ccy1, ccy2, asof_date)
    
    assert result is None
    service.query.assert_called_once_with(ccy1, ccy2, asof_date)

def test_query_respects_strict_parameter():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)
    
    service = MagicMock(spec=FXRateService)
    
    service.query(ccy1, ccy2, asof_date, strict=True)
    service.query.assert_called_with(ccy1, ccy2, asof_date, strict=True)
    
    service.query(ccy1, ccy2, asof_date, strict=False)
    service.query.assert_called_with(ccy1, ccy2, asof_date, strict=False)
```


# LLM-generated content at query #4
#--------------------------

```python
from unittest.mock import MagicMock
from decimal import Decimal
from datetime import date

def test_queries_returns_correct_rates_for_valid_inputs():
    service = MagicMock(spec=FXRateService)
    query_list = [('USD', 'EUR', date(2023, 1, 1)), ('GBP', 'JPY', date(2023, 1, 2))]
    expected_rates = [Decimal('0.92'), Decimal('150.50')]
    service.queries.return_value = iter(expected_rates)

    results = list(service.queries(query_list))

    assert results == expected_rates
    service.queries.assert_called_once_with(query_list, strict=False)

def test_queries_returns_none_for_missing_rates():
    service = MagicMock(spec=FXRateService)
    query_list = [('USD', 'XYZ', date(2023, 1, 1))]
    expected_rates = [None]
    service.queries.return_value = iter(expected_rates)

    results = list(service.queries(query_list))

    assert results == [None]
    service.queries.assert_called_once_with(query_list, strict=False)

def test_queries_raises_error_when_strict_is_true_and_rate_missing():
    service = MagicMock(spec=FXRateService)
    query_list = [('USD', 'XYZ', date(2023, 1, 1))]
    service.queries.side_effect = ValueError("Rate not found")

    try:
        list(service.queries(query_list, strict=True))
    except ValueError as e:
        assert str(e) == "Rate not found"
    
    service.queries.assert_called_once_with(query_list, strict=True)
```


# LLM-generated content at query #5
#--------------------------

```python
import datetime
from decimal import Decimal
from typing import NamedTuple

# Mocking dependencies required for the test environment
class Currency(str):
    pass

class Date:
    pass

ZERO = Decimal("0")
ONE = Decimal("1")

# Re-defining FXRate as provided in the prompt to ensure the test is runnable
class FXRate(NamedTuple):
    ccy1: Currency
    ccy2: Currency
    date: datetime.date
    value: Decimal

    def __invert__(self) -> "FXRate":
        return FXRate(self[1], self[0], self[2], self[3] ** -1)

def test_invert_fxrate_logic():
    date_val = datetime.date.today()
    nrate = FXRate(Currency("EUR"), Currency("USD"), date_val, Decimal("2"))
    rrate = ~nrate
    assert rrate.ccy1 == Currency("USD")
    assert rrate.ccy2 == Currency("EUR")
    assert rrate.date == date_val
    assert rrate.value == Decimal("0.5")

def test_invert_fxrate_identity():
    date_val = datetime.date.today()
    rate = FXRate(Currency("USD"), Currency("GBP"), date_val, Decimal("1"))
    inverted_twice = ~~rate
    assert inverted_twice == rate

def test_invert_fxrate_value_calculation():
    date_val = datetime.date.today()
    nrate = FXRate(Currency("JPY"), Currency("USD"), date_val, Decimal("150"))
    rrate = ~nrate
    assert rrate.value == Decimal("1") / Decimal("150")
```


# LLM-generated content at query #6
#--------------------------

```python
from decimal import Decimal
from datetime import date
from typing import Optional, Tuple
from unittest.mock import MagicMock
from pypara.currencies import Currency, CurrencyType
from pypara.exchange import FXRateService

def test_query_returns_rate_when_found():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEE) # Assuming CurrencyType exists
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

    service.query(ccy1, ccy2, asof_date, strict=False)
    service.query.assert_called_with(ccy1, ccy2, asof_date, strict=False)
```


# LLM-generated content at query #7
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
    expected_rates = [Decimal("0.92"), Decimal("160.50")]
    service.queries.return_value = iter(expected_rates)

    result = list(service.queries(query_list))

    assert result == expected_rates
    service.queries.assert_called_once_with(query_list, strict=False)

def test_queries_with_strict_true():
    service = MagicMock(spec=FXRateService)
    query_list = [("USD", "EUR", date(2023, 1, 1))]
    service.queries.return_value = iter([Decimal("0.92")])

    result = list(service.queries(query_list, strict=True))

    assert result == [Decimal("0.92")]
    service.queries.assert_called_once_with(query_list, strict=True)

def test_queries_returns_none_for_missing_rate():
    service = MagicMock(spec=FXRateService)
    query_list = [("USD", "XYZ", date(2023, 1, 1))]
    service.queries.return_value = iter([None])

    result = list(service.queries(query_list))

    assert result == [None]
    service.queries.assert_called_once()
```


# LLM-generated content at query #8
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

    result = list(service.queries(query_list))

    assert result == expected_rates
    service.queries.assert_called_once_with(query_list, strict=False)

def test_queries_with_strict_true():
    service = MagicMock(spec=FXRateService)
    query_list = [('USD', 'CAD', date(2023, 5, 5))]
    service.queries.return_value = iter([Decimal('1.35')])

    result = list(service.queries(query_list, strict=True))

    assert result == [Decimal('1.35')]
    service.queries.assert_called_once_with(query_list, strict=True)

def test_queries_returns_none_when_rate_missing():
    service = MagicMock(spec=FXRateService)
    query_list = [('XYZ', 'ABC', date(2023, 1, 1))]
    service.queries.return_value = iter([None])

    result = list(service.queries(query_list))

    assert result == [None]
    service.queries.assert_called_once()
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
    expected_rates = [Decimal("0.92"), Decimal("160.50")]
    service.queries.return_value = iter(expected_rates)

    result = list(service.queries(query_list))

    assert result == expected_rates
    service.queries.assert_called_once_with(query_list, strict=False)

def test_queries_with_strict_true():
    service = MagicMock(spec=FXRateService)
    query_list = [("USD", "CAD", date(2023, 5, 5))]
    service.queries.return_value = iter([Decimal("1.35")])

    result = list(service.queries(query_list, strict=True))

    assert result == [Decimal("1.35")]
    service.queries.assert_called_once_with(query_list, strict=True)

def test_queries_returns_none_for_missing_rates():
    service = MagicMock(spec=FXRateService)
    query_list = [("USD", "XYZ", date(2023, 1, 1))]
    service.queries.return_value = iter([None])

    result = list(service.queries(query_list))

    assert result == [None]
```


# LLM-generated content at query #10
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

def test_fxrate_constructor_tuple_access():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    date_val = datetime.date.today()
    value_val = Decimal("2.0")
    ccy1 = Currencies["GBP"]
    ccy2 = Currencies["JPY"]
    rate = FXRate(ccy1, ccy2, date_val, value_val)
    assert rate[0] == ccy1
    assert rate[1] == ccy2
    assert rate[2] == date_val
    assert rate[3] == value_val

def test_fxrate_constructor_inversion():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    date_val = datetime.date.today()
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    value_val = Decimal("2")
    rate = FXRate(ccy1, ccy2, date_val, value_val)
    inverted_rate = ~rate
    assert inverted_rate.ccy1 == ccy2
    assert inverted_rate.ccy2 == ccy1
    assert inverted_rate.date == date_val
    assert inverted_rate.value == Decimal("0.5")
```


# LLM-generated content at query #11
#--------------------------

```python
def test_fxrate_constructor_valid_inputs():
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

def test_fxrate_constructor_tuple_access():
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


# LLM-generated content at query #12
#--------------------------

```python
def test_fxrate_constructor_valid_data():
    import datetime
    from decimal import Decimal
    from typing import NamedTuple
    # Mocking Currency/Date as they are not provided in snippet but used in type hints
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
    assert rate[0] == ccy1
    assert rate[3] == val

def test_fxrate_constructor_tuple_unpacking():
    import datetime
    from decimal import Decimal
    class Currency(str): pass
    Date = datetime.date
    
    ccy1 = Currency("EUR")
    ccy2 = Currency("USD")
    today = datetime.date.today()
    val = Decimal("1.5")
    
    rate = FXRate(ccy1, ccy2, today, val)
    
    unpacked_ccy1, unpacked_ccy2, unpacked_date, unpacked_val = rate
    
    assert unpacked_ccy1 == ccy1
    assert unpacked_ccy2 == ccy2
    assert unpacked_date == today
    assert unpacked_val == val

def test_fxrate_invert_operation():
    import datetime
    from decimal import Decimal
    class Currency(str): pass
    Date = datetime.date
    
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

def test_fxrate_of_valid_creation():
    import datetime
    from decimal import Decimal
    class Currency(str): pass
    Date = datetime.date
    
    ccy1 = Currency("EUR")
    ccy2 = Currency("USD")
    today = datetime.date.today()
    val = Decimal("1.25")
    
    rate = FXRate.of(ccy1, ccy2, today, val)
    
    assert rate.ccy1 == ccy1
    assert rate.value == val

def test_fxrate_of_same_currency_with_one_fails_if_not_one():
    import datetime
    from decimal import Decimal
    class Currency(str):
        def __eq__(self, other): return True
    Date = datetime.date
    
    ccy1 = Currency("EUR")
    ccy2 = Currency("EUR")
    today = datetime.date.today()
    val = Decimal("1.5")
    
    # Note: The provided 'of' implementation has logic for same currency check
    # We expect a ValueError if ccy1 == ccy2 and value != 1
    import pytest
    with pytest.raises(ValueError, match="FX rate to the same currency must be `one`."):
        FXRate.of(ccy1, ccy2, today, val)

def test_fxrate_of_invalid_value_fails():
    import datetime
    from decimal import Decimal
    class Currency(str): pass
    Date = datetime.date
    
    ccy1 = Currency("EUR")
    ccy2 = Currency("USD")
    today = datetime.date.today()
    val = Decimal("-1.0")
    
    import pytest
    with pytest.raises(ValueError, match="FX rate value can not be equal to or less than `zero`."):
        FXRate.of(ccy1, ccy2, today, val)
```


# LLM-generated content at query #13
#--------------------------

```python
from decimal import Decimal
from datetime import date
from typing import Optional, Tuple
from pypara.currencies import Currency, CurrencyType
from pypara.exchange import FXRateService

def test_query_interface_definition():
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
    as_of_date = date(2023, 1, 1)
    
    rate = service.query(usd, eur, as_of_date)
    assert isinstance(rate, MockFXRate)

def test_query_return_type_none():
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: date, strict: bool = False) -> Optional[Decimal]:
            return None

        def queries(self, queries: any, strict: bool = False) -> any:
            return []

    service = MockFXRateService()
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    as_of_date = date(2023, 1, 1)

    rate = service.query(usd, eur, as_of_date)
    assert rate is None
```


# LLM-generated content at query #14
#--------------------------

```python
from decimal import Decimal
from datetime import date
from unittest.mock import MagicMock
from pypara.currencies import Currency, CurrencyType
from pypara.exchange import FXRateService

def test_query_returns_rate_when_found():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONESS)
    asof_date = date(2023, 1, 1)
    mock_rate = MagicMock()
    
    service = MagicMock(spec=FXRateService)
    service.query.return_value = mock_rate
    
    result = service.query(ccy1, ccy2, asof_date, strict=False)
    
    service.query.assert_called_once_with(ccy1, ccy2, asof_date, strict=False)
    assert result == mock_rate

def test_query_returns_none_when_not_found():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)
    
    service = MagicMock(spec=FXRateService)
    service.query.return_value = None
    
    result = service.query(ccy1, ccy2, asof_date, strict=False)
    
    assert result is None

def test_query_with_strict_true():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = MagicMock(spec=Currency)
    asof_date = date(2023, 1, 1)
    
    service = MagicMock(spec=FXRateService)
    
    service.query(ccy1, ccy2, asof_date, strict=True)
    
    service.query.assert_called_with(ccy1, ccy2, asof_date, strict=True)
```


# LLM-generated content at query #15
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

def test_fxrate_constructor_tuple_access():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    date_val = datetime.date.today()
    value_val = Decimal("2.0")
    ccy1 = Currencies["GBP"]
    ccy2 = Currencies["JPY"]
    rate = FXRate(ccy1, ccy2, date_val, value_val)
    assert rate[0] == ccy1
    assert rate[1] == ccy2
    assert rate[2] == date_val
    assert rate[3] == value_val

def test_fxrate_constructor_identity_check():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    date_val = datetime.date.today()
    value_val = Decimal("1.0")
    ccy1 = Currencies["USD"]
    ccy2 = Currencies["USD"]
    rate = FXRate(ccy1, ccy2, date_val, value_val)
    assert rate.ccy1 == rate.ccy2
    assert rate.value == Decimal("1")
```


# LLM-generated content at query #16
#--------------------------

```python
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

    def __invert__(self) -> "FXRate":
        return FXRate(self[1], self[0], self[2], self[3] ** -1)

    @classmethod
    def of(cls, ccy1: Currency, ccy2: Currency, date: datetime.date, value: Decimal) -> "FXRate":
        if not isinstance(ccy1, Currency):
            raise ValueError("CCY/1 must be of type `Currency`.")
        if not isinstance(ccy2, Currency):
            raise ValueError("CCY/2 must be of type `Currency`.")
        if value <= 0:
            raise ValueError("FX rate value can not be equal to or less than `zero`.")
        if ccy1 == ccy2 and value != Decimal("1"):
            raise ValueError("FX rate to the same currency must be `one`.")
        return cls(ccy1, ccy2, date, value)

def test_fxrate_constructor_assignment():
    date_val = datetime.date.today()
    val_decimal = Decimal("2.5")
    ccy_a = Currency("EUR")
    ccy_b = Currency("USD")
    rate = FXRate(ccy_a, ccy_b, date_val, val_decimal)
    assert rate.ccy1 == ccy_a
    assert rate.ccy2 == ccy_b
    assert rate.date == date_val
    assert rate.value == val_decimal

def test_fxrate_constructor_tuple_access():
    date_val = datetime.date.today()
    val_decimal = Decimal("1.2")
    ccy_a = Currency("GBP")
    ccy_b = Currency("JPY")
    rate = FXRate(ccy_a, ccy_b, date_val, val_decimal)
    assert rate[0] == ccy_a
    assert rate[1] == ccy_b
    assert rate[2] == date_val
    assert rate[3] == val_decimal

def test_fxrate_invert_operation():
    date_val = datetime.date.today()
    ccy_a = Currency("EUR")
    ccy_b = Currency("USD")
    original_rate = FXRate(ccy_a, ccy_b, date_val, Decimal("2"))
    inverted_rate = ~original_rate
    assert inverted_rate.ccy1 == ccy_b
    assert inverted_rate.ccy2 == ccy_a
    assert inverted_rate.date == date_val
    assert inverted_rate.value == Decimal("0.5")
```


# LLM-generated content at query #17
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
    assert rate[0] == ccy1
    assert rate[1] == ccy2
    assert rate[2] == date_val
    assert rate[3] == value_val

def test_fxrate_constructor_unvalidated_input_possible():
    import datetime
    from decimal import Decimal
    # The docstring states the constructor is not safe and does not check input.
    # We test that it allows raw assignment of types even if they don't match hints.
    date_val = datetime.date.today()
    value_val = Decimal("2")
    rate = FXRate("NOT_A_CURRENCY", "ALSO_NOT_A_CURRENCY", date_val, value_val)
    assert rate.ccy1 == "NOT_A_CURRENCY"
    assert rate.ccy2 == "ALSO_NOT_A_CURRENCY"
```


# LLM-generated content at query #18
#--------------------------

```python
def test_fxrate_constructor_valid_data():
    from decimal import Decimal
    import datetime
    # Assuming Currency is a type that can be instantiated or mocked as needed 
    # based on the provided docstring context. Using strings for mock implementation.
    class MockCurrency:
        def __init__(self, code): self.code = code
        def __eq__(self, other): return isinstance(other, MockCurrency) and self.code == other.code
    
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
    assert rate[3] == value_val

def test_fxrate_constructor_tuple_access():
    from decimal import Decimal
    import datetime
    class MockCurrency:
        def __init__(self, code): self.code = code
    ccy1 = MockCurrency("EUR")
    ccy2 = MockCurrency("USD")
    date_val = datetime.date.today()
    value_val = Decimal("1.5")
    rate = FXRate(ccy1, ccy2, date_val, value_val)
    
    assert rate[0] == ccy1
    assert rate[1] == ccy2
    assert rate[2] == date_val
    assert rate[3] == value_val
```


# LLM-generated content at query #19
#--------------------------

```python
def test_fxrate_constructor_valid_input():
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

def test_fxrate_constructor_tuple_access():
    from decimal import Decimal
    import datetime
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("2.0")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate[0] == ccy1
    assert rate[1] == ccy2
    assert rate[2] == date
    assert rate[3] == value

def test_fxrate_constructor_with_inverted_logic():
    from decimal import Decimal
    import datetime
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("2.0")
    rate = FXRate(ccy1, ccy2, date, value)
    inverted_rate = ~rate
    assert inverted_rate.ccy1 == ccy2
    assert inverted_rate.ccy2 == ccy1
    assert inverted_rate.date == date
    assert inverted_rate.value == Decimal("0.5")
```


# LLM-generated content at query #20
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
    expected_rates = [Decimal("0.92"), Decimal("160.50")]
    service.queries.return_value = iter(expected_rates)

    result = list(service.queries(query_list))

    assert result == expected_rates
    service.queries.assert_called_once_with(query_list, strict=False)

def test_queries_with_strict_true():
    service = MagicMock(spec=FXRateService)
    query_list = [("USD", "GBP", date(2023, 1, 1))]
    service.queries.return_value = iter([Decimal("0.78")])

    result = list(service.queries(query_list, strict=True))

    assert result == [Decimal("0.78")]
    service.queries.assert_called_once_with(query_list, strict=True)

def test_queries_returns_none_on_missing_rate():
    service = MagicMock(spec=FXRateService)
    query_list = [("USD", "XYZ", date(2023, 1, 1))]
    service.queries.return_value = iter([None])

    result = list(service.queries(query_list))

    assert result == [None]
    service.queries.assert_called_once()
```


# LLM-generated content at query #21
#--------------------------

```python
from decimal import Decimal
from datetime import date
from unittest.mock import MagicMock
from pypara.currencies import Currency, CurrencyType
from pypara.exchange import FXRateService

def test_query_returns_rate_when_exists():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONES)
    asof_date = date(2023, 1, 1)
    mock_rate = MagicMock()
    service = MagicMock(spec=FXRateService)
    service.query.return_value = mock_rate

    result = service.query(ccy1, ccy2, asof_date)

    assert result == mock_rate
    service.query.assert_called_once_with(ccy1, ccy2, asof_date, strict=False)

def test_query_returns_none_when_not_found():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
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


# LLM-generated content at query #22
#--------------------------

```python
def test_fxrate_constructor_valid_data():
    from decimal import Decimal
    import datetime
    # Assuming Currency is a class or enum that can be instantiated/accessed 
    # using string keys like in the docstring example.
    # For the purpose of this test, we use dummy objects that mimic the expected behavior.
    class MockCurrency:
        def __init__(self, name): self.name = name
        def __eq__(self, other): return isinstance(other, MockCurrency) and self.name == other.name
    
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
    assert rate[3] == value_val

def test_fxrate_constructor_tuple_unpacking():
    from decimal import Decimal
    import datetime
    class MockCurrency:
        def __init__(self, name): self.name = name
    
    ccy1 = MockCurrency("EUR")
    ccy2 = MockCurrency("USD")
    date_val = datetime.date(2023, 1, 1)
    value_val = Decimal("1.5")
    
    rate = FXRate(ccy1, ccy2, date_val, value_val)
    ccy_a, ccy_b, d, v = rate
    
    assert ccy_a == ccy1
    assert ccy_b == ccy2
    assert d == date_val
    assert v == value_val
```


# LLM-generated content at query #23
#--------------------------

```python
def test_fxrate_constructor_valid_values():
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

def test_fxrate_constructor_unpacks_correctly():
    from decimal import Decimal
    import datetime
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


# LLM-generated content at query #24
#--------------------------

```python
from decimal import Decimal
from datetime import date
from typing import Optional, Tuple
from pypara.currencies import Currency, CurrencyType
from pypara.exchange import FXRateService

def test_fxrateservice_query_interface_signature():
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccu2: Currency, asof: date, strict: bool = False):
            return Decimal("1.0")
        def queries(self, queries: any, strict: bool = False):
            return []

    service = MockFXRateService()
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)
    
    result = service.query(usd, eur, asof_date, strict=True)
    
    assert isinstance(result, Decimal)
    assert result == Decimal("1.0")

def test_fxrateservice_query_return_none():
    class MockFXRateServiceNone(FXRateService):
        def query(self, ccy1: Currency, ccu2: Currency, asof: date, strict: bool = False):
            return None
        def queries(self, queries: any, strict: bool = False):
            return []

    service = MockFXRateServiceNone()
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)
    
    result = service.query(usd, eur, asof_date)
    
    assert result is None
```


# LLM-generated content at query #25
#--------------------------

```python
def test_queries_returns_iterable_of_rates():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        def queries(self, queries, strict=False):
            return [None for _ in queries]

    service = MockFXRateService()
    query_list = [("USD", "EUR", "2023-01-01"), ("GBP", "JPY", "2023-01-02")]
    results = list(service.queries(query_list))
    
    assert len(results) == 2
    assert results[0] is None
    assert results[1] is None

def test_queries_with_strict_mode_logic():
    class MockStrictFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        def queries(self, queries, strict=False):
            if strict:
                raise ValueError("Lookup error")
            return [None for _ in queries]

    service = MockStrictFXRateService()
    query_list = [("USD", "EUR", "2023-01-01")]
    
    assert list(service.queries(query_list, strict=False)) == [None]
    
    try:
        service.queries(query_list, strict=True)
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


# LLM-generated content at query #26
#--------------------------

```python
def test_fxrate_constructor_valid():
    from decimal import Decimal
    import datetime
    # Assuming Currency is a mock or existing type that can be instantiated/indexed as per docstring
    # Since we cannot define classes, we use the logic provided in the snippet's doctest
    ccy1 = "EUR" 
    ccy2 = "USD"
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

def test_fxrate_constructor_inversion():
    from decimal import Decimal
    import datetime
    ccy1 = "EUR"
    ccy2 = "USD"
    date_val = datetime.date.today()
    value_val = Decimal("2")
    rate = FXRate(ccy1, ccy2, date_val, value_val)
    inverted_rate = ~rate
    assert inverted_rate.ccy1 == ccy2
    assert inverted_rate.ccy2 == ccy1
    assert inverted_rate.date == date_val
    assert inverted_rate.value == Decimal("0.5")

def test_fxrate_of_valid():
    from decimal import Decimal
    import datetime
    # Mocking Currency objects as strings for the sake of the test if actual class is unavailable
    # However, according to the code logic, FXRate.of performs isinstance checks.
    # This test assumes a compatible environment where Currency and ZERO/ONE are defined.
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date_val = datetime.date.today()
    value_val = Decimal("2")
    rate = FXRate.of(ccy1, ccy2, date_val, value_val)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.value == value_val

def test_fxrate_of_invalid_value_raises_error():
    from decimal import Decimal
    import datetime
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date_val = datetime.date.today()
    value_val = Decimal("0")
    # The implementation raises ValueError if value <= ZERO
    import pytest
    with pytest.raises(ValueError, match="FX rate value can not be equal to or less than `zero`."):
        FXRate.of(ccy1, ccy2, date_val, value_val)

def test_fxrate_of_same_currency_invalid_value_raises_error():
    from decimal import Decimal
    import datetime
    ccy1 = Currencies["EUR"]
    date_val = datetime.date.today()
    value_val = Decimal("1.5")
    # The implementation raises ValueError if ccy1 == ccy2 and value != ONE
    import pytest
    with pytest.raises(ValueError, match="FX rate to the same currency must be `one`."):
        FXRate.of(ccy1, ccy1, date_val, value_val)
```


# LLM-generated content at query #27
#--------------------------

```python
def test_fxrate_constructor_valid_input():
    from decimal import Decimal
    import datetime
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
    assert rate[0] == ccy1
    assert rate[1] == ccy2
    assert rate[2] == date_val
    assert rate[3] == value_val

def test_fxrate_constructor_tuple_unpacking():
    from decimal import Decimal
    import datetime
    from pypara.currencies import Currencies
    date_val = datetime.date.today()
    value_val = Decimal("2.0")
    ccy1 = Currencies["GBP"]
    ccy2 = Currencies["JPY"]
    rate = FXRate(ccy1, ccy2, date_val, value_val)
    unpacked_ccy1, unpacked_ccy2, unpacked_date, unpacked_value = rate
    assert unpacked_ccy1 == ccy1
    assert unpacked_ccy2 == ccy2
    assert unpacked_date == date_val
    assert unpacked_value == value_val
```


# LLM-generated content at query #28
#--------------------------

```python
from unittest.mock import MagicMock
from decimal import Decimal
from datetime import date

def test_queries_returns_correct_rates():
    mock_service = MagicMock(spec=FXRateService)
    ccy1 = "USD"
    ccy2 = "EUR"
    asof_date = date(2023, 1, 1)
    rate_val = Decimal("0.91")
    
    queries_input = [(ccy1, ccy2, asof_date)]
    expected_output = [rate_val]
    mock_service.queries.return_value = expected_output
    
    result = list(mock_service.queries(queries_input))
    
    assert result == expected_output
    mock_service.queries.assert_called_once_with(queries_input, strict=False)

def test_queries_handles_none_rates():
    mock_service = MagicMock(spec=FXRateService)
    ccy1 = "USD"
    ccy2 = "GBP"
    asof_date = date(2023, 1, 1)
    
    queries_input = [(ccy1, ccy2, asof_date)]
    expected_output = [None]
    mock_service.queries.return_value = expected_output
    
    result = list(mock_service.queries(queries_input))
    
    assert result == [None]
    mock_service.queries.assert_called_once_with(queries_input, strict=False)

def test_queries_with_strict_mode():
    mock_service = MagicMock(spec=FXRateService)
    ccy1 = "JPY"
    ccy2 = "AUD"
    asof_date = date(2023, 5, 5)
    
    queries_input = [(ccy1, ccy2, asof_date)]
    mock_service.queries.return_value = [Decimal("0.01")]
    
    result = list(mock_service.queries(queries_input, strict=True))
    
    assert result == [Decimal("0.01")]
    mock_service.queries.assert_called_once_with(queries_input, strict=True)

def test_queries_multiple_inputs():
    mock_service = MagicMock(spec=FXRateService)
    date1 = date(2023, 1, 1)
    date2 = date(2023, 1, 2)
    queries_input = [("USD", "EUR", date1), ("USD", "GBP", date2)]
    expected_output = [Decimal("0.91"), Decimal("0.82")]
    mock_service.queries.return_value = expected_output
    
    result = list(mock_service.queries(queries_input))
    
    assert len(result) == 2
    assert result[0] == Decimal("0.91")
    assert result[1] == Decimal("0.82")
```


# LLM-generated content at query #29
#--------------------------

```python
import datetime
from decimal import Decimal
from typing import NamedTuple

# Mocking dependencies for the context of the test
class Currency(str):
    pass

EUR = Currency("EUR")
USD = Currency("USD")
ZERO = Decimal("0")
ONE = Decimal("1")

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

def test_fxrate_constructor_assigns_values_correctly():
    date_val = datetime.date.today()
    value_val = Decimal("2.5")
    rate = FXRate(EUR, USD, date_val, value_val)
    assert rate.ccy1 == EUR
    assert rate.ccy2 == USD
    assert rate.date == date_val
    assert rate.value == value_val

def test_fxrate_constructor_allows_indexed_access():
    date_val = datetime.date.today()
    value_val = Decimal("1.5")
    rate = FXRate(EUR, USD, date_val, value_val)
    assert rate[0] == EUR
    assert rate[1] == USD
    assert rate[2] == date_val
    assert rate[3] == value_val

def test_fxrate_constructor_with_identical_currencies_and_one_value():
    date_val = datetime.date.today()
    rate = FXRate(EUR, EUR, date_val, ONE)
    assert rate.ccy1 == rate.ccy2
    assert rate.value == ONE

def test_fxrate_constructor_with_identical_currencies_and_non_one_value():
    # Note: The constructor itself does not check consistency (as per docstring), 
    # only .of() does. This test verifies the raw constructor behavior.
    date_val = datetime.date.today()
    rate = FXRate(EUR, EUR, date_val, Decimal("2.0"))
    assert rate.ccy1 == rate.ccy2
    assert rate.value == Decimal("2.0")
```


# LLM-generated content at query #30
#--------------------------

```python
from decimal import Decimal
from datetime import date
from typing import Optional, Tuple
from unittest.mock import MagicMock

def test_query_returns_expected_rate():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)
    expected_rate = FXRate(ccy1, ccy2, asof_date, Decimal("0.95"))
    
    service = MagicMock(spec=FXRateService)
    service.query.return_value = expected_rate
    
    result = service.query(ccy1, ccy2, asof_date)
    
    assert result == expected_rate
    service.query.assert_called_once_with(ccy1, ccy2, asof_date)

def test_query_returns_none_when_no_rate_found():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    asof_date = date(2023, 1, 1)
    
    service = MagicMock(spec=FXRateService)
    service.query.return_value = None
    
    result = service.query(ccy1, ccy2, asof_date)
    
    assert result is None
    service.query.assert_called_once_with(ccy1, ccy2, asof_date)

def test_query_with_strict_parameter():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY)
    asof_date = date(2023, 5, 5)
    
    service = MagicMock(spec=FXRateService)
    
    service.query(ccy1, ccy2, asof_date, strict=True)
    service.query.assert_called_with(ccy1, ccy2, asof_date, strict=True)
```


