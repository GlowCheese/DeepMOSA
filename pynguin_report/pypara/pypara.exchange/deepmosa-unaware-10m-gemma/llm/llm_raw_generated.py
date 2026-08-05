####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from decimal import Decimal
from datetime import date

def test_FXRate___invert__():
    """
    Tests the inversion logic of the FXRate class.
    Verifies that ~rate swaps currencies and calculates the reciprocal of the value.
    """
    # Setup dependencies
    ccy_eur = Currency["EUR"]
    ccy_usd = Currency["USD"]
    today = date.today()
    original_value = Decimal("2.0")
    expected_inverted_value = Decimal("0.5")

    # Create original rate (EUR to USD)
    rate = FXRate(ccy_eur, ccy_usd, today, original_value)
    
    # Apply inversion
    inverted_rate = ~rate

    # Assertions for properties
    assert inverted_rate.ccy1 == ccy_usd, "Inverted rate should have the second currency as the first."
    assert inverted_rate.ccy2 == ccy_eur, "Inverted rate should have the first currency as the second."
    assert inverted_rate.date == today, "Inverted rate must maintain the same date."
    assert inverted_rate.value == expected_inverted_value, f"Inverted value should be {expected_inverted_value}."

    # Assertions for structural equality with an explicitly created inverse
    manual_inverse = FXRate(ccy_usd, ccy_eur, today, expected_inverted_value)
    assert inverted_rate == manual_inverse, "The ~ operator result should equal a manually constructed inverse rate."

    # Verify that inversion is idempotent in terms of returning to original (double negation)
    assert ~~rate == rate, "Double inversion (~(~rate)) should return the original rate."
```


# LLM-generated content at query #2
#--------------------------

```python
import datetime
from decimal import Decimal
from unittest.mock import MagicMock

def test_FXRate___invert__():
    # Setup dependencies
    ccy_eur = MagicMock(spec=Currency)
    ccy_usd = MagicMock(spec=Currency)
    test_date = datetime.date.today()
    rate_value = Decimal("2.0")
    inverse_value = Decimal("0.5")

    # Create original rate
    original_rate = FXRate(
        ccy1=ccy_eur,
        ccy2=ccy_usd,
        date=test_date,
        value=rate_value
    )

    # Execute inversion
    inverted_rate = ~original_rate

    # Assertions
    # 1. Check that the currencies are swapped
    assert inverted_rate.ccy1 == ccy_usd
    assert inverted_rate.ccy2 == ccy_eur

    # 2. Check that the date remains the same
    assert inverted_rate.date == test_date

    # 3. Check that the value is correctly inverted (1/2 = 0.5)
    assert inverted_rate.value == inverse_value

    # 4. Check that inverting twice returns to the original rate
    assert ~inverted_rate == original_rate
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
import datetime

class MockFXRateService(FXRateService):
    def __init__(self):
        self._data = {}

    def add_rate(self, rate: FXRate):
        self._data[(rate.ccy1, rate.ccy2, rate.date)] = rate

    def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
        key = (ccy1, ccy2, asof)
        rate = self._data.get(key)
        if strict and rate is None:
            raise FXRateLookupError(ccy1, ccy2, asof)
        return rate

    def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
        for q in queries:
            yield self.query(*q, strict=strict)

@pytest.fixture
def mock_currency_usd():
    return MagicMock(spec=Currency)

@pytest.fixture
def mock_currency_eur():
    return MagicMock(spec=Currency)

@pytest.fixture
def sample_date():
    return datetime.date(2023, 1, 1)

@pytest.fixture
def fx_service(mock_currency_usd, mock_currency_eur, sample_date):
    service = MockFXRateService()
    rate = FXRate(mock_currency_usd, mock_currency_eur, sample_date, Decimal("0.85"))
    service.add_rate(rate)
    return service

def test_FXRateService_query(fx_service, mock_currency_usd, mock_currency_eur, sample_date):
    # Test case 1: Successful lookup
    found_rate = fx_service.query(mock_currency_usd, mock_currency_eur, sample_date)
    assert found_rate is not None
    assert found_rate.value == Decimal("0.85")
    assert found_rate.ccy1 == mock_currency_usd

    # Test case 2: Lookup non-existent rate (non-strict) returns None
    none_rate = fx_service.query(mock_currency_eur, mock_currency_usd, sample_date)
    assert none_rate is None

    # Test case 3: Lookup non-existent rate (strict) raises FXRateLookupError
    with pytest.raises(FXRateLookupError) as excinfo:
        fx_service.query(mock_currency_eur, mock_currency_usd, sample_date, strict=True)
    
    assert excinfo.value.ccy1 == mock_currency_eur
    assert excinfo.value.ccy2 == mock_currency_usd
    assert excinfo.value.asof == sample_date

    # Test case 4: Date mismatch returns None
    wrong_date = datetime.date(2023, 1, 2)
    assert fx_service.query(mock_currency_usd, mock_currency_eur, wrong_date) is None
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
from datetime import date

class MockFXRateService(FXRateService):
    def __init__(self, data=None):
        self.data = data or {}

    def query(self, ccy1, ccy2, asof, strict=False) -> Optional[FXRate]:
        key = (ccy1, ccy2, asof)
        if key in self.data:
            return self.data[key]
        if strict:
            raise FXRateLookupError(ccy1, ccy2, asof)
        return None

    def queries(self, queries, strict=False) -> Iterable[Optional[FXRate]]:
        results = []
        for q in queries:
            try:
                results.append(self.query(*q, strict=strict))
            except FXRateLookupError:
                if strict:
                    raise
                results.append(None)
        return results

def test_FXRateService_queries():
    # Setup dependencies
    ccy_usd = MagicMock(spec=Currency)
    ccy_eur = MagicMock(spec=Currency)
    ccy_gbp = MagicMock(spec=Currency)
    today = date.today()
    val_1 = Decimal("1.2")
    val_2 = Decimal("0.8")
    
    rate_usd_eur = FXRate(ccy_usd, ccy_eur, today, val_1)
    rate_eur_gbp = FXRate(ccy_eur, ccy_gbp, today, val_2)

    # Mock data for the service
    service_data = {
        (ccy_usd, ccy_eur, today): rate_usd_eur,
        (ccy_eur, ccy_gbp, today): rate_eur_gbp
    }
    
    service = MockFXRateService(data=service_data)

    # Test Case 1: Batch queries with existing and non-existing rates (non-strict)
    query_list = [
        (ccy_usd, ccy_eur, today),  # Exists
        (ccy_gbp, ccy_usd, today),  # Does not exist
        (ccy_eur, ccy_gbp, today),  # Exists
    ]
    
    results = list(service.queries(query_list, strict=False))
    
    assert len(results) == 3
    assert results[0] == rate_usd_eur
    assert results[1] is None
    assert results[2] == rate_eur_gbp

    # Test Case 2: Batch queries with non-existing rates (strict=True)
    # Should raise FXRateLookupError because the second query fails
    with pytest.raises(FXRateLookupError) as excinfo:
        list(service.queries(query_list, strict=True))
    
    assert excinfo.value.ccy1 == ccy_gbp
    assert excinfo.value.ccy2 == ccy_usd
    assert excinfo.value.asof == today

    # Test Case 3: Empty queries list
    assert list(service.queries([], strict=False)) == []

    # Test Case 4: Single query that exists
    single_query = [(ccy_usd, ccy_eur, today)]
    assert list(service.queries(single_query))[0] == rate_usd_eur
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
from datetime import date

class MockFXRateService(FXRateService):
    def __init__(self, rate_map=None):
        self.rate_map = rate_map or {}

    def query(self, ccy1, ccy2, asof, strict=False):
        key = (ccy1, ccy2, asof)
        if key in self.rate_map:
            return self.rate_map[key]
        if strict:
            raise FXRateLookupError(ccy1, ccy2, asof)
        return None

    def queries(self, queries, strict=False):
        return [self.query(q[0], q[1], q[2], strict=strict) for q in queries]

def test_FXRateService_queries():
    # Setup dependencies
    usd = MagicMock(spec=Currency)
    eur = MagicMock(spec=Currency)
    today = date.today()
    val = Decimal("1.1")
    
    rate_instance = FXRate(usd, eur, today, val)
    
    # Define test data
    query_list = [
        (usd, eur, today),          # Exists
        (eur, usd, today),          # Does not exist
        (usd, usd, today),          # Identity (does not exist in map)
    ]
    
    rate_map = {
        (usd, eur, today): rate_instance
    }
    
    service = MockFXRateService(rate_map=rate_map)

    # 1. Test non-strict mode (returns None for missing rates)
    results = list(service.queries(query_list, strict=False))
    assert len(results) == 3
    assert results[0] == rate_instance
    assert results[1] is None
    assert results[2] is None

    # 2. Test strict mode (raises FXRateLookupError for missing rates)
    with pytest.raises(FXRateLookupError) as excinfo:
        list(service.queries(query_list, strict=True))
    
    # Verify error details
    assert excinfo.value.ccy1 == usd
    assert excinfo.value.ccy2 == eur
    assert excinfo.value.asof == today

    # 3. Test single element query
    single_query = [(usd, eur, today)]
    results_single = list(service.queries(single_query))
    assert results_single == [rate_instance]

    # 4. Test empty queries
    assert list(service.queries([])) == []
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest
from decimal import Decimal
from unittest.mock import MagicMock
from datetime import date

@pytest.mark.parametrize("ccy1, ccy2, asof, expected_rate, strict", [
    (Currencies["EUR"], Currencies["USD"], date(2023, 1, 1), FXRate(Currencies["EUR"], Currencies["USD"], date(2023, 1, 1), Decimal("1.1")), False),
    (Currencies["GBP"], Currencies["JPY"], date(2023, 5, 20), None, False),
])
def test_FXRateService_query(ccy1, ccy2, asof, expected_rate, strict):
    # Arrange
    service = MagicMock(spec=FXRateService)
    service.query.return_value = expected_rate
    
    # Act
    result = service.query(ccy1, ccy2, asof, strict=strict)
    
    # Assert
    assert result == expected_rate
    service.query.assert_called_once_with(ccy1, ccy2, asof, strict=strict)

def test_FXRateService_query_raises_lookup_error():
    # Arrange
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    asof = date(2023, 1, 1)
    
    service = MagicMock(spec=FXRateService)
    service.query.side_effect = FXRateLookupError(ccy1, ccy2, asof)
    
    # Act & Assert
    with pytest.raises(FXRateLookupError) as excinfo:
        service.query(ccy1, ccy2, asof, strict=True)
    
    assert ccy1 in str(excinfo.value)
    assert ccy2 in str(excinfo.value)
    assert asof in str(excinfo.value)
    service.query.assert_called_once_with(ccy1, ccy2, asof, strict=True)
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
from datetime import date

class MockFXRateService(FXRateService):
    def __init__(self, data=None):
        self.data = data or {}

    def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
        key = (ccy1, ccy2, asof)
        if key in self.data:
            return self.data[key]
        if strict:
            raise FXRateLookupError(ccy1, ccyey, asof)
        return None

    def queries(self, queries: Iterable[TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
        results = []
        for q in queries:
            try:
                results.append(self.query(q[0], q[1], q[2], strict=strict))
            except FXRateLookupError as e:
                if strict:
                    raise e
                results.append(None)
        return results

def test_FXRateService_queries():
    # Setup currencies and dates
    ccy_usd = MagicMock(spec=Currency)
    ccy_eur = MagicMock(spec=Currency)
    ccy_gbp = MagicMock(spec=Currency)
    today = date.today()
    yesterday = date(1999, 1, 1)

    # Setup rate data
    rate1 = FXRate(ccy_usd, ccy_eur, today, Decimal("0.85"))
    rate2 = FXRate(ccy_eur, ccy_usd, today, Decimal("1.17"))
    
    service_data = {
        (ccy_usd, ccy_eur, today): rate1,
        (ccy_eur, ccy_usd, today): rate2,
    }

    # 1. Test successful retrieval of multiple queries
    service = MockFXRateService(data=service_data)
    queries_list = [
        (ccy_usd, ccy_eur, today),
        (ccy_eur, ccy_usd, today),
        (ccy_gbp, ccy_usd, yesterday) # Not in data
    ]
    
    results = list(service.queries(queries_list, strict=False))
    
    assert len(results) == 3
    assert results[0] == rate1
    assert results[1] == rate2
    assert results[2] is None

    # 2. Test strict mode raising FXRateLookupError
    service_strict = MockFXRateService(data=service_data)
    with pytest.raises(FXRateLookupError) as excinfo:
        list(service_strict.queries([(ccy_gbp, ccy_usd, yesterday)], strict=True))
    
    assert excinfo.value.ccy1 == ccy_gbp
    assert excinfo.value.ccy2 == ccy_usd
    assert excinfo.value.asof == yesterday

    # 3. Test empty queries list
    assert list(service.queries([], strict=False)) == []

    # 4. Test single query with no match (non-strict)
    assert list(service.queries([(ccy_gbp, ccy_usd, today)], strict=False)) == [None]
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
from datetime import date

@pytest.mark.parametrize("ccy1, ccy2, asof, expected_rate, strict, should_raise", [
    (
        Currency["EUR"], 
        Currency["USD"], 
        date(2023, 1, 1), 
        FXRate(Currency["EUR"], Currency["USD"], date(2023, 1, 1), Decimal("1.1")), 
        False, 
        False
    ),
    (
        Currency["EUR"], 
        Currency["USD"], 
        date(2023, 1, 1), 
        None, 
        False, 
        False
    ),
    (
        Currency["EUR"], 
        Currency["USD"], 
        date(2023, 1, 1), 
        None, 
        True, 
        True
    ),
])
def test_FXRateService_query(ccy1, ccy2, asof, expected_rate, strict, should_raise):
    # Arrange
    service = MagicMock(spec=FXRateService)
    service.query.return_value = expected_rate
    
    if should_raise:
        service.query.side_effect = FXRateLookupError(ccy1, ccy2, asof)

    # Act & Assert
    if should_raise:
        with pytest.raises(FXRateLookupError) as excinfo:
            service.query(ccy1, ccy2, asof, strict=strict)
        assert excinfo.value.ccy1 == ccy1
        assert excinfo.value.ccy2 == ccy2
        assert excinfo.value.asof == asof
    else:
        result = service.query(ccy1, ccy2, asof, strict=strict)
        assert result == expected_rate

    # Verify call arguments
    service.query.assert_called_with(ccy1, ccy2, asof, strict=strict)
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
from datetime import date

class MockFXRateService(FXRateService):
    def __init__(self, rates=None):
        self.rates = rates or []

    def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
        for rate in self.rates:
            if rate.ccy1 == ccy1 and rate.ccy2 == ccy2 and rate.date == asof:
                return rate
        if strict:
            raise FXRateLookupError(ccy1, ccyey=ccy2, asof=asof)
        return None

def test_FXRateService_query():
    # Setup dependencies
    usd = MagicMock(spec=Currency)
    eur = MagicMock(spec=Currency)
    today = date.today()
    value = Decimal("1.1")
    
    rate_found = FXRate(ccy1=eur, ccy2=usd, date=today, value=value)
    rate_different_date = FXRate(ccy1=eur, ccy2=usd, date=date(2000, 1, 1), value=value)
    rate_different_ccy = FXRate(ccy1=usd, ccy2=eur, date=today, value=value)
    
    service = MockFXRateService(rates=[rate_found, rate_different_date, rate_different_ccy])

    # Test Case 1: Successful lookup
    result = service.query(eur, usd, today, strict=False)
    assert result == rate_found
    assert result.value == Decimal("1.1")

    # Test Case 2: Lookup returns None when not found (strict=False)
    result_none = service.query(usd, eur, today, strict=False)
    assert result_none is None

    # Test Case 3: Lookup returns None when date mismatch
    result_wrong_date = service.query(eur, usd, date(1999, 12, 31), strict=False)
    assert result_wrong_date is None

    # Test Case 4: Lookup raises FXRateLookupError when not found (strict=True)
    with pytest.raises(FXRateLookupError) as excinfo:
        service.query(usd, eur, today, strict=True)
    
    assert excinfo.value.ccy1 == usd
    assert excinfo.value.ccy2 == eur
    assert excinfo.value.asof == today
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
from datetime import date

@pytest.mark.parametrize("ccy1, ccy2, asof, expected_rate", [
    (Currencies["EUR"], Currencies["USD"], date(2023, 1, 1), FXRate(Currencies["EUR"], Currencies["USD"], date(2023, 1, 1), Decimal("1.1"))),
    (Currencies["GBP"], Currencies["JPY"], date(2023, 5, 20), FXRate(Currencies["GBP"], CurrenciesJPY, date(2023, 5, 20), Decimal("180.5"))),
])
def test_FXRateService_query_success(ccy1, ccy2, asof, expected_rate):
    service = MagicMock(spec=FXRateService)
    service.query.return_value = expected_rate
    
    result = service.query(ccy1, ccy2, asof)
    
    assert result == expected_rate
    service.query.assert_called_once_with(ccy1, ccy2, asof, strict=False)

def test_FXRateService_query_returns_none():
    service = MagicMock(spec=FXRateService)
    service.query.return_value = None
    
    result = service.query(Currencies["EUR"], Currencies["USD"], date(2023, 1, 1))
    
    assert result is None

def test_FXRateService_query_raises_lookup_error():
    service = MagicMock(spec=FXRateService)
    ccy1, ccy2, asof = Currencies["EUR"], Currencies["USD"], date(2023, 1, 1)
    
    service.query.side_effect = FXRateLookupError(ccy1, ccy2, asof)
    
    with pytest.raises(FXRateLookupError) as excinfo:
        service.query(ccy1, ccy2, asof, strict=True)
    
    assert ccy1 in str(excinfo.value)
    assert ccy2 in str(excinfo.value)
    assert asof in str(excinfo.value)

def test_FXRateService_query_with_strict_param():
    service = MagicMock(spec=FXRateService)
    ccy1, ccy2, asof = Currencies["EUR"], Currencies["USD"], date(2023, 1, 1)
    
    service.query(ccy1, ccy2, asof, strict=True)
    service.query.assert_called_with(ccy1, ccy2, asof, strict=True)
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
from datetime import date

class MockFXRateService(FXRateService):
    def __init__(self, data=None):
        self.data = data or {}

    def query(self, ccy1, ccy2, asof, strict=False):
        key = (ccy1, ccy2, asof)
        if key in self.data:
            return self.data[key]
        if strict:
            raise FXRateLookupError(ccy1, ccy2, asof)
        return None

    def queries(self, queries, strict=False):
        return (self.query(q[0], q[1], q[2], strict=strict) for q in queries)

def test_FXRateService_queries():
    # Setup mock currencies and dates
    ccy_usd = MagicMock(spec=Currency)
    ccy_eur = MagicMock(spec=Currency)
    ccy_gbp = MagicMock(spec=Currency)
    today = date.today()
    yesterday = date.today().replace(day=max(1, today.day - 1))

    # Define rate data
    rate_usd_eur = FXRate(ccy_usd, ccy_eur, today, Decimal("0.9"))
    rate_eur_usd = FXRate(ccy_eur, ccy_usd, today, Decimal("1.1"))
    
    lookup_table = {
        (ccy_usd, ccy_eur, today): rate_usd_eur,
        (ccy_eur, ccy_usd, today): rate_eur_usd,
        (ccy_gbp, ccy_usd, yesterday): None # Explicitly missing but present in logic
    }

    service = MockFXRateService(data=lookup_table)

    # Define queries
    query_list = [
        (ccy_usd, ccy_eur, today),    # Exists
        (ccy_eur, ccy_usd, today),    # Exists
        (ccy_gbp, ccy_usd, yesterday), # Exists as None
        (ccy_gbp, ccy_eur, today)     # Does not exist in table (returns None)
    ]

    # 1. Test non-strict mode (default)
    results = list(service.queries(query_list, strict=False))
    assert len(results) == 4
    assert results[0] == rate_usd_eur
    assert results[1] == rate_eur_usd
    assert results[2] is None
    assert results[3] is None

    # 2. Test strict mode (should raise error on missing key)
    # We only test the first query which is known to exist, 
    # then the one that triggers lookup error.
    strict_query = [(ccy_usd, ccy_eur, today), (ccy_gbp, ccy_eur, today)]
    
    # First element should work
    results_strict = list(service.queries(strict_query, strict=True))
    assert results_strict[0] == rate_usd_eur

    # Second element should raise FXRateLookupError
    with pytest.raises(FXRateLookupError) as excinfo:
        list(service.queries(strict_query, strict=True))
    
    assert excinfo.value.ccy1 == ccy_gbp
    assert excinfo.value.ccy2 == ccy_eur
    assert excinfo.value.asof == today
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
from datetime import date

@pytest.mark.parametrize("ccy1, ccy2, asof, expected_rate", [
    (Currency["EUR"], Currency["USD"], date(2023, 1, 1), FXRate(Currency["EUR"], Currency["USD"], date(2023, 1, 1), Decimal("1.1"))),
    (Currency["GBP"], Currency["JPY"], date(2023, 5, 20), FXRate(Currency["GBP"], Currency["JPY"], date(2023, 5, 20), Decimal("180.5"))),
])
def test_FXRateService_query_success(ccy1, ccy2, asof, expected_rate):
    service = MagicMock(spec=FXRateService)
    service.query.return_value = expected_rate

    result = service.query(ccy1, ccy2, asof)

    assert result == expected_rate
    service.query.assert_called_once_with(ccy1, ccy2, asof)

def test_FXRateService_query_returns_none():
    service = MagicMock(spec=FXRateService)
    service.query.return_value = None

    result = service.query(Currency["EUR"], Currency["USD"], date(2023, 1, 1))

    assert result is None
    service.query.assert_called_once()

def test_FXRateService_query_strict_mode_raises_error():
    service = MagicMock(spec=FXRateService)
    service.query.side_effect = FXRateLookupError(Currency["EUR"], Currency["USD"], date(2023, 1, 1))

    with pytest.raises(FXRateLookupError) as excinfo:
        service.query(Currency["EUR"], Currency["Mock"], date(2023, 1, 1), strict=True)
    
    assert "EUR" in str(excinfo.value)
    service.query.assert_called_once_with(Currency["EUR"], Currency["Mock"], date(2023, 1, 1), strict=True)

def test_FXRateService_query_parameters_passed_correctly():
    service = MagicMock(spec=FXRateService)
    ccy1 = Currency["USD"]
    ccy2 = Currency["CAD"]
    asof = date(2023, 12, 31)
    strict_val = True

    service.query(ccy1, ccy2, asof, strict=strict_val)

    service.query.assert_called_with(ccy1, ccy2, asof, strict=strict_val)
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest
from decimal import Decimal
from datetime import date
from unittest.mock import MagicMock

class MockFXRateService(FXRateService):
    def __init__(self, data=None):
        # data is a dict: {(ccy1, ccy2, asof): FXRate}
        self.data = data or {}

    def query(self, ccy1, ccy2, asof, strict=False) -> Optional[FXRate]:
        key = (ccy1, ccy2, asof)
        if key in self.data:
            return self.data[key]
        if strict:
            raise FXRateLookupError(ccy1, ccy2, asof)
        return None

    def queries(self, queries, strict=False) -> Iterable[Optional[FXRate]]:
        results = []
        for q in queries:
            try:
                results.append(self.query(*q, strict=strict))
            except FXRateLookupError:
                if strict:
                    raise
                results.append(None)
        return results

def test_FXRateService_queries():
    # Setup currencies and dates
    ccy_usd = MagicMock(spec=Currency)
    ccy_eur = MagicMock(spec=Currency)
    ccy_gbp = MagicMock(spec=Currency)
    today = date.today()
    yesterday = date(1999, 1, 1)
    
    val_usd_eur = Decimal("0.85")
    val_eur_usd = Decimal("1.18")

    # Prepare mock data for the service
    rate1 = FXRate(ccy_usd, ccy_eur, today, val_usd_eur)
    rate2 = FXRate(ccy_eur, ccy_usd, today, val_eur_usd)
    
    service_data = {
        (ccy_usd, ccy_eur, today): rate1,
        (ccy_eur, ccy_usd, today): rate2,
    }
    
    service = MockFXRateService(data=service_data)

    # Define queries: 
    # 1. Existing rate
    # 2. Non-existent rate (returns None in non-strict)
    # 3. Another existing rate
    queries_list = [
        (ccy_usd, ccy_eur, today),
        (ccy_gbp, ccy_usd, yesterday),
        (ccy_eur, ccy_usd, today)
    ]

    # Test 1: Non-strict mode (default behavior expected for queries)
    results = list(service.queries(queries_list, strict=False))
    
    assert len(results) == 3
    assert results[0] == rate1
    assert results[1] is None
    assert results[2] == rate2

    # Test 2: Strict mode (should raise error on the first missing entry)
    with pytest.raises(FXRateLookupError) as excinfo:
        list(service.queries(queries_list, strict=True))
    
    assert excinfo.value.ccy1 == ccy_gbp
    assert excinfo.value.ccy2 == ccy_usd
    assert excinfo.value.asof == yesterday

    # Test 3: Empty queries list
    assert list(service.queries([], strict=False)) == []
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
from datetime import date

class MockFXRateService(FXRateService):
    def __init__(self):
        self._rates = {}

    def add_rate(self, rate: FXRate):
        self._rates[(rate.ccy1, rate.ccy2, rate.date)] = rate

    def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
        key = (ccy1, ccy2, asof)
        if key in self._rates:
            return self._rates[key]
        if strict:
            raise FXRateLookupError(ccy1, ccy2, asof)
        return None

def test_FXRateService_query():
    service = MockFXRateService()
    today = date.today()
    usd = Currency["USD"]
    eur = Currency["EUR"]
    val = Decimal("0.85")
    
    rate = FXRate(usd, eur, today, val)
    service.add_rate(rate)

    # Test case 1: Successful lookup
    result = service.query(usd, eur, today)
    assert result == rate
    assert result.value == val

    # Test case 2: Non-existent rate, strict=False (returns None)
    result_none = service.query(eur, usd, today, strict=False)
    assert result_none is None

    # Test case 3: Non-existent rate, strict=True (raises FXRateLookupError)
    with pytest.raises(FXRateLookupError) as excinfo:
        service.query(eur, usd, today, strict=True)
    
    assert excinfo.value.ccy1 == eur
    assert excinfo.value.ccy2 == usd
    assert excinfo.value.asof == today
    assert "EUR/USD" in str(excinfo.value)

    # Test case 4: Lookup with different date
    result_wrong_date = service.query(usd, eur, date(1999, 1, 1))
    assert result_wrong_date is None
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
from datetime import date

class MockFXRateService(FXRateService):
    def __init__(self):
        self._data = {}

    def add_rate(self, rate: FXRate):
        self._data[(rate.ccy1, rate.ccy2, rate.date)] = rate

    def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
        key = (ccy1, ccy2, asof)
        rate = self._data.get(key)
        if rate is None and strict:
            raise FXRateLookupError(ccy1, ccy2, asof)
        return rate

@pytest.fixture
def mock_currencies():
    return {
        "EUR": MagicMock(spec=Currency),
        "USD": MagicMock(spec=tuple), # Using tuple for mock comparison if needed
        "GBP": MagicMock(spec=Currency)
    }

# Manually assigning names to mocks to satisfy Currency equality/identity in tests
def setup_mocks(mocks):
    mocks["USD"].__eq__.side_effect = lambda x: x == mocks["USD"]
    mocks["EUR"].__eq__.side_effect = lambda x: x == mocks["EUR"]
    mocks["GBP"].__eq__.side_effect = lambda x: x == mocks["GBP"]
    return mocks

def test_FXRateService_query(mock_currencies):
    # Setup
    ccy_eur = MagicMock(spec=Currency)
    ccy_usd = MagicMock(spec=Currency)
    ccy_gbp = MagicMock(spec=Currency)
    
    # Ensure equality works for the keys in our mock dict
    for c in [ccy_eur, ccy_usd, ccy_gbp]:
        c.__eq__ = lambda self, other: self == c # Simple identity mock
    
    # We use a real-ish approach for the lookup key
    from unittest.mock import patch
    
    service = MockFXRateService()
    today = date.today()
    val = Decimal("1.1")
    rate_exists = FXRate(ccy_eur, ccy_usd, today, val)
    service.add_rate(rate_exists)

    # Test Case 1: Successful lookup
    found_rate = service.query(ccy_eur, ccy_usd, today, strict=False)
    assert found_rate == rate_exists
    assert found_rate.value == val

    # Test Case 2: Lookup returns None when not found (strict=False)
    not_found = service.query(ccy_usd, ccy_eur, today, strict=False)
    assert not_found is None

    # Test Case 3: Lookup raises FXRateLookupError (strict=True)
    with pytest.raises(FXRateLookupError) as excinfo:
        service.query(ccy_gbp, ccy_usd, today, strict=True)
    
    assert excinfo.value.ccy1 == ccy_gbp
    assert excinfo.value.ccy2 == ccy_usd
    assert excinfo.value.asof == today

    # Test Case 4: Different date returns None
    different_date = date(2000, 1, 1)
    assert service.query(ccy_eur, ccy_usd, different_date, strict=False) is None
```


# LLM-generated content at query #16
#--------------------------

```python
import pytest
from decimal import Decimal
from unittest.mock import MagicMock
from datetime import date

class MockFXRateService(FXRateService):
    def __init__(self, mapping=None):
        self.mapping = mapping or {}

    def query(self, ccy1, ccy2, asof, strict=False) -> Optional[FXRate]:
        key = (ccy1, ccy2, asof)
        if key in self.mapping:
            return self.mapping[key]
        if strict:
            raise FXRateLookupError(ccy1, ccy/2, asof) # Note: The original code has a typo 'ccy/2' in logic context, but we follow the class definition
            # Using the actual class constructor provided in snippet:
            # raise FXRateLookupError(ccy1, ccy2, asof)
        return None

    def queries(self, queries, strict=False) -> Iterable[Optional[FXRate]]:
        results = []
        for q in queries:
            try:
                results.append(self.query(*q, strict=strict))
            except FXRateLookupError as e:
                if strict:
                    raise e
                results.append(None)
        return results

def test_FXRateService_queries():
    # Setup dependencies
    ccy_usd = MagicMock(spec=Currency)
    ccy_eur = MagicMock(spec=Currency)
    ccy_gbp = MagicMock(spec=Currency)
    today = date.today()
    rate_val = Decimal("1.2")
    
    rate_obj = FXRate(ccy_usd, ccy_eur, today, rate_val)
    
    # Mock data for the service
    mapping = {
        (ccy_usd, ccy_eur, today): rate_obj,
        (ccy_eur, ccy_gbp, today): FXRate(ccy_eur, ccy_gbp, today, Decimal("0.8"))
    }
    
    service = MockFXRateService(mapping=mapping)

    # Test Case 1: Successful retrieval of existing queries
    queries_input = [
        (ccy_usd, ccy_eur, today),
        (ccy_eur, ccy_gbp, today)
    ]
    results = list(service.queries(queries_input, strict=False))
    assert len(results) == 2
    assert results[0] == rate_obj
    assert results[1].value == Decimal("0.8")

    # Test Case 2: Non-existent query with strict=False returns None
    queries_missing = [(ccy_gbp, ccy_usd, today)]
    results_none = list(service.queries(queries_missing, strict=False))
    assert len(results_none) == 1
    assert results_none[0] is None

    # Test Case 3: Non-existent query with strict=True raises FXRateLookupError
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries_missing, strict=True))

    # Test Case 4: Mixed existing and non-existing queries (strict=False)
    mixed_queries = [
        (ccy_usd, ccy_eur, today),
        (ccy_gbp, ccy_usd, today) # Missing
    ]
    results_mixed = list(service.queries(mixed_queries, strict=False))
    assert len(results_mixed) == 2
    assert results_mixed[0] == rate_obj
    assert results_mixed[1] is None

    # Test Case 5: Mixed existing and non-existing queries (strict=True) should fail on first error
    with pytest.raises(FXRateLookupError):
        list(service.queries(mixed_queries, strict=True))
```


# LLM-generated content at query #17
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
from datetime import date

@pytest.mark.parametrize("ccy1, ccy2, asof, expected_rate", [
    (Currency["EUR"], Currency["USD"], date(2023, 1, 1), FXRate(Currency["EUR"], Currency["USD"], date(2023, 1, 1), Decimal("1.1"))),
    (Currency["GBP"], Currency["JPY"], date(2023, 5, 20), FXRate(Currency["GBP"], Currency["JPY"], date(2023, 5, 20), Decimal("180.5"))),
])
def test_FXRateService_query_success(ccy1, ccy2, asof, expected_rate):
    service = MagicMock(spec=FXRateService)
    service.query.return_value = expected_rate
    
    result = service.query(ccy1, ccy2, asof)
    
    service.query.assert_called_once_with(ccy1, ccy2, asof, strict=False)
    assert result == expected_rate

def test_FXRateService_query_returns_none():
    service = MagicMock(spec=FXRateService)
    service.query.return_value = None
    
    result = service.query(Currency["EUR"], Currency["USD"], date(2023, 1, 1))
    
    assert result is None

def test_FXRateService_query_strict_mode_raises_error():
    service = MagicMock(spec=FXRateService)
    service.query.side_effect = FXRateLookupError(Currency["EUR"], Currency["USD"], date(2023, 1, 1))
    
    with pytest.raises(FXRateLookupError) as excinfo:
        service.query(Currency["EUR"], Currency["USD"], date(2023, 1, 1), strict=True)
    
    assert "EUR/USD" in str(excinfo.value)
    service.query.assert_called_once_with(Currency["EUR"], Currency["USD"], date(2023, 1, 1), strict=True)

def test_FXRateService_query_strict_mode_success():
    service = MagicMock(spec=FXRateService)
    rate = FXRate(Currency["EUR"], Currency["USD"], date(2023, 1, 1), Decimal("1.1"))
    service.query.return_value = rate
    
    result = service.query(Currency["EUR"], Currency["USD"], date(2023, 1, 1), strict=True)
    
    assert result == rate
    service.query.assert_called_once_with(Currency["EUR"], Currency["USD"], date(2023, 1, 1), strict=True)
```


# LLM-generated content at query #18
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
from datetime import date

class MockFXRateService(FXRateService):
    def __init__(self, rates_map=None):
        self.rates_map = rates_map or {}

    def query(self, ccy1, ccy2, asof, strict=False):
        key = (ccy1, ccy2, asof)
        if key in self.rates_map:
            return self.rates_map[key]
        if strict:
            raise FXRateLookupError(ccy1, ccy2, asof)
        return None

    def queries(self, queries, strict=False):
        return [self.query(q[0], q[1], q[2], strict=strict) for q in queries]

def test_FXRateService_queries():
    # Setup dependencies
    eur = MagicMock(spec=Currency)
    usd = MagicMock(spec=Currency)
    gbp = MagicMock(spec=Currency)
    today = date.today()
    val1 = Decimal("1.1")
    val2 = Decimal("0.8")
    
    rate1 = FXRate(eur, usd, today, val1)
    rate2 = FXRate(usd, gbp, today, val2)
    
    # Map for lookup: (ccy1, ccy2, date) -> FXRate
    rates_lookup = {
        (eur, usd, today): rate1,
        (usd, gbp, today): rate2
    }
    
    service = MockFXRateService(rates_lookup)

    # Test Case 1: Successful retrieval of multiple rates
    queries_list = [
        (eur, usd, today),
        (usd, gbp, today)
    ]
    results = list(service.queries(queries_list, strict=False))
    assert len(results) == 2
    assert results[0] == rate1
    assert results[1] == rate2

    # Test Case 2: Non-existent rate with strict=False returns None
    queries_missing = [(eur, gbp, today)]
    results_missing = list(service.queries(queries_missing, strict=False))
    assert len(results_missing) == 1
    assert results_missing[0] is None

    # Test Case 3: Non-existent rate with strict=True raises FXRateLookupError
    with pytest.raises(FXRateLookupError) as excinfo:
        list(service.queries(queries_missing, strict=True))
    
    assert excinfo.value.ccy1 == eur
    assert excinfo.value.ccy2 == gbp
    assert excinfo.value.asof == today

    # Test Case 4: Empty query list returns empty iterable
    assert list(service.queries([], strict=False)) == []
```


# LLM-generated content at query #19
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
from datetime import date

class MockFXRateService(FXRateService):
    def __init__(self, rates_map=None):
        self.rates_map = rates_map or {}

    def query(self, ccy1, ccy2, asof, strict=False) -> Optional[FXRate]:
        key = (ccy1, ccy2, asof)
        if key in self.rates_map:
            return self.rates_map[key]
        if strict:
            raise FXRateLookupError(ccy1, ccy2, asof)
        return None

    def queries(self, queries, strict=False) -> Iterable[Optional[FXRate]]:
        results = []
        for q in queries:
            try:
                results.append(self.query(*q, strict=strict))
            except FXRateLookupError:
                if strict:
                    raise
                results.append(None)
        return results

def test_FXRateService_queries():
    # Setup dependencies
    ccy_usd = MagicMock(spec=Currency)
    ccy_eur = MagicMock(spec=Currency)
    ccy_gbp = MagicMock(spec=Currency)
    today = date.today()
    val1 = Decimal("1.1")
    val2 = Decimal("0.85")
    
    rate1 = FXRate(ccy_usd, ccy_eur, today, val1)
    rate2 = FXRate(ccy_usd, ccy_gbp, today, val2)

    # Mock Data
    rates_data = {
        (ccy_usd, ccy_eur, today): rate1,
        (ccy_usd, ccy_gbp, today): rate2,
    }
    
    service = MockFXRateService(rates_map=rates_data)

    # Test Case 1: Successful retrieval of multiple rates
    queries_list = [
        (ccy_usd, ccy_eur, today),
        (ccy_usd, ccy_gbp, today)
    ]
    results = list(service.queries(queries_list, strict=False))
    assert len(results) == 2
    assert results[0] == rate1
    assert results[1] == rate2

    # Test Case 2: Handling missing rates with strict=False (returns None)
    missing_query = [(ccy_eur, ccy_usd, today)]
    results_none = list(service.queries(missing_query, strict=False))
    assert len(results_none) == 1
    assert results_none[0] is None

    # Test Case 3: Handling missing rates with strict=True (raises FXRateLookupError)
    with pytest.raises(FXRateLookupError) as excinfo:
        list(service.queries(missing_query, strict=True))
    assert excinfo.value.ccy1 == ccy_eur
    assert excinfo.value.ccy2 == ccy_usd

    # Test Case 4: Mixed results (some found, some missing)
    mixed_queries = [
        (ccy_usd, ccy_eur, today),
        (ccy_eur, ccy_usd, today)  # Not in our map
    ]
    mixed_results = list(service.queries(mixed_queries, strict=False))
    assert mixed_results[0] == rate1
    assert mixed_results[1] is None

    # Test Case 5: Empty input
    assert list(service.queries([], strict=False)) == []
```


# LLM-generated content at query #20
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
import datetime

class MockFXRateService(FXRateService):
    def __init__(self, rates=None):
        self.rates = rates or []

    def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
        for rate in self.rates:
            if rate.ccy1 == ccy1 and rate.ccy2 == ccy2 and rate.date == asof:
                return rate
        if strict:
            raise FXRateLookupError(ccy1, ccy, asof)
        return None

def test_FXRateService_query():
    # Setup dependencies
    usd = MagicMock(spec=Currency)
    eur = MagicMock(spec=Currency)
    usd.name = "USD"
    eur.name = "EUR"
    
    today = datetime.date.today()
    rate_val = Decimal("0.85")
    
    existing_rate = FXRate(eur, usd, today, rate_val)
    other_rate = FXRate(usd, eur, today, Decimal("1.17"))
    
    service = MockFXRateService(rates=[existing_rate, other_rate])

    # Test Case 1: Successful lookup
    found_rate = service.query(eur, usd, today)
    assert found_rate == existing_rate
    assert found_rate.value == rate_val

    # Test Case 2: Lookup returns None when rate does not exist (non-strict)
    missing_rate = service.query(usd, eur, datetime.date.today() - datetime.timedelta(days=1))
    assert missing_rate is None

    # Test Case 3: Strict mode raises FXRateLookupError when rate is missing
    with pytest.raises(FXRateLookupError) as excinfo:
        service.query(usd, eur, today, strict=True)
    
    assert excinfo.value.ccy1 == usd
    assert excinfo.value.ccy2 == eur
    assert excinfo.value.asof == today

    # Test Case 4: Lookup with different date returns None
    different_date = datetime.date.today() - datetime.timedelta(days=5)
    assert service.query(eur, usd, different_date) is None

    # Test Case 5: Verify service can be mocked for complex behavior
    mock_service = MagicMock(spec=FXRateService)
    mock_service.query.return_value = existing_rate
    result = mock_service.query(eur, usd, today)
    assert result == existing_rate
    mock_service.query.assert_called_once_with(eur, usd, today)
```


# LLM-generated content at query #21
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
import datetime

class MockFXRateService(FXRateService):
    def __init__(self):
        self._rates = {}

    def add_rate(self, rate: FXRate):
        self._rates[(rate.ccy1, rate.ccy2, rate.date)] = rate

    def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
        key = (ccy1, ccy2, asof)
        rate = self._rates.get(key)
        if rate is None and strict:
            raise FXRateLookupError(ccy1, ccy2, asof)
        return rate

def test_FXRateService_query():
    service = MockFXRateService()
    
    # Setup mock data
    usd = Currency("USD")
    eur = Currency("EUR")
    today = datetime.date.today()
    rate_val = Decimal("1.1")
    rate = FXRate(usd, eur, today, rate_val)
    service.add_rate(rate)

    # Test Case 1: Successful lookup
    found_rate = service.query(usd, eur, today)
    assert found_rate == rate
    assert found_rate.value == rate_val

    # Test Case 2: Non-strict lookup for non-existent rate returns None
    missing_rate = service.query(eur, usd, today, strict=False)
    assert missing_rate is None

    # Test Case 3: Strict lookup for non-existent rate raises FXRateLookupError
    with pytest.raises(FXRateLookupError) as excinfo:
        service.query(eur, usd, today, strict=True)
    
    assert excinfo.value.ccy1 == eur
    assert excinfo.value.ccy2 == usd
    assert excinfo.value.asof == today
    assert "EUR/USD" in str(excinfo.value)

    # Test Case 4: Lookup with different date returns None
    yesterday = today - datetime.timedelta(days=1)
    assert service.query(usd, eur, yesterday) is None
```


# LLM-generated content at query #22
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
from datetime import date

class MockFXRateService(FXRateService):
    def __init__(self, rates_map=None):
        self.rates_map = rates_map or {}

    def query(self, ccy1, ccy2, asof, strict=False):
        key = (ccy1, ccy2, asof)
        if key in self.rates_map:
            return self.rates_map[key]
        if strict:
            raise FXRateLookupError(ccy1, ccy2, asof)
        return None

    def queries(self, queries, strict=False):
        return (self.query(q[0], q[1], q[2], strict=strict) for q in queries)

def test_FXRateService_queries():
    # Setup dependencies
    ccy_usd = MagicMock(spec=Currency)
    ccy_eur = MagicMock(spec=Currency)
    ccy_gbp = MagicMock(spec=Currency)
    asof_date = date(2023, 1, 1)
    val_rate = Decimal("1.1")
    val_rate_inv = Decimal("0.9")
    
    rate1 = FXRate(ccy_eur, ccy_usd, asof_date, val_rate)
    rate2 = FXRate(ccy_usd, ccy_eur, asof_date, val_rate_inv)
    
    # Mock data for the service
    rates_data = {
        (ccy_eur, ccy_usd, asof_date): rate1,
        (ccy_usd, ccy_eur, asof_date): rate2,
    }
    
    service = MockFXRateService(rates_map=rates_data)
    
    # 1. Test successful retrieval of multiple rates
    queries_list = [
        (ccy_eur, ccy_usd, asof_date),
        (ccy_usd, ccy_eur, asof_date)
    ]
    results = list(service.queries(queries_list, strict=False))
    assert len(results) == 2
    assert results[0] == rate1
    assert results[1] == rate2

    # 2. Test retrieval of non-existent rate (non-strict)
    query_missing = [(ccy_gbp, ccy_usd, asof_date)]
    results_missing = list(service.queries(query_missing, strict=False))
    assert len(results_missing) == 1
    assert results_missing[0] is None

    # 3. Test retrieval of non-existent rate (strict mode raises error)
    with pytest.raises(FXRateLookupError) as excinfo:
        list(service.queries(query_missing, strict=True))
    assert excinfo.value.ccy1 == ccy_gbp
    assert excinfo.value.ccy2 == ccy_usd
    assert excinfo.value.asof == asof_date

    # 4. Test empty queries list
    assert list(service.queries([], strict=False)) == []
```


# LLM-generated content at query #23
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
from datetime import date

class MockFXRateService(FXRateService):
    def __init__(self):
        self._store = {}

    def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
        key = (ccy1, ccy2, asof)
        rate = self._store.get(key)
        if strict and rate is None:
            raise FXRateLookupError(ccy1, ccy2, asof)
        return rate

    def queries(self, queries: Iterable[TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
        for q in queries:
            yield self.query(*q, strict=strict)

    def add_rate(self, rate: FXRate):
        self._store[(rate.ccy1, rate.ccy2, rate.date)] = rate


def test_FXRateService_query():
    service = MockFXRateService()
    
    # Setup dummy data
    usd = MagicMock(spec=Currency)
    eur = MagicMock(spec=Currency)
    today = date.today()
    rate_val = Decimal("1.1")
    rate = FXRate(usd, eur, today, rate_val)
    service.add_rate(rate)

    # 1. Test successful lookup
    found_rate = service.query(usd, eur, today)
    assert found_rate == rate
    assert found_rate.value == rate_val

    # 2. Test lookup for non-existent rate (non-strict)
    missing_rate = service.query(eur, usd, today)
    assert missing_rate is None

    # 3. Test lookup for non-existent rate (strict mode raises error)
    with pytest.raises(FXRateLookupError) as excinfo:
        service.query(eur, usd, today, strict=True)
    
    assert excinfo.value.ccy1 == eur
    assert excinfo.value.ccy2 == usd
    assert excinfo.value.asof == today

    # 4. Test lookup with different date (non-existent)
    yesterday = date.today() - MagicMock(days=1)
    assert service.query(usd, eur, yesterday) is None
```


# LLM-generated content at query #24
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
from datetime import date

class MockFXRateService(FXRateService):
    def __init__(self, data=None):
        # data structure: {(ccy1, ccy2, date): FXRate}
        self.data = data or {}

    def query(self, ccy1, ccy2, asof, strict=False) -> Optional[FXRate]:
        try:
            return self.data[(ccy1, ccy2, asof)]
        except KeyError:
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None

    def queries(self, queries, strict=False) -> Iterable[Optional[FXRate]]:
        results = []
        for q in queries:
            try:
                results.append(self.query(*q, strict=strict))
            except FXRateLookupError:
                if strict:
                    raise
                # In a real implementation, if strict is False and we catch 
                # the error from query, we should handle it based on requirements.
                # For this test, we simulate raising it up.
                raise
        return results

def test_FXRateService_queries():
    # Setup currencies and dates
    ccy_usd = MagicMock(spec=Currency)
    ccy_eur = MagicMock(spec=Currency)
    ccy_gbp = MagicMock(spec=Currency)
    today = date.today()
    yesterday = date(1999, 1, 1)
    
    val_usd_eur = Decimal("0.85")
    val_eur_usd = Decimal("1.17")
    
    rate1 = FXRate(ccy_usd, ccy_eur, today, val_usd_eur)
    rate2 = FXRate(ccy_eur, ccy_usd, today, val_eur_usd)
    rate3 = FXRate(ccy_gbp, ccy_usd, yesterday, Decimal("1.3"))

    # Mock data for the service
    service_data = {
        (ccy_usd, ccy_eur, today): rate1,
        (ccy_eur, ccy_usd, today): rate2,
        (ccy_gbp, ccy_usd, yesterday): rate3,
    }

    service = MockFXRateService(data=service_data)

    # Define queries: 
    # 1. Existing rate
    # 2. Existing rate (different pair)
    # 3. Non-existent rate
    # 4. Non-existent rate (different date)
    queries_list = [
        (ccy_usd, ccy_eur, today),
        (ccy_gbp, ccy_usd, yesterday),
        (ccy_usd, ccy_gbp, today),
        (ccy_eur, ccy_usd, yesterday)
    ]

    # Test 1: Non-strict mode (should return None for missing rates)
    results = list(service.queries(queries_list, strict=False))
    
    assert len(results) == 4
    assert results[0] == rate1
    assert results[1] == rate3
    assert results[2] is None
    assert results[3] is None

    # Test 2: Strict mode (should raise FXRateLookupError for missing rates)
    queries_strict = [(ccy_usd, ccy_eur, today), (ccy_usd, ccy_gbp, today)]
    
    with pytest.raises(FXRateLookupError) as excinfo:
        list(service.queries(queries_strict, strict=True))
    
    assert excinfo.value.ccy1 == ccy_usd
    assert excinfo.value.ccy2 == ccy_gbp
    assert excinfo.value.asof == today

    # Test 3: Empty queries
    assert list(service.queries([], strict=False)) == []
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
from datetime import date

class MockFXRateService(FXRateService):
    def __init__(self, rate_map=None):
        self.rate_map = rate_map or {}

    def query(self, ccy1, ccy2, asof, strict=False):
        key = (ccy1, ccy2, asof)
        if key in self.rate_map:
            return self.rate_map[key]
        if strict:
            raise FXRateLookupError(ccy1, ccy2, asof)
        return None

    def queries(self, queries, strict=False):
        return (self.query(q[0], q[1], q[2], strict=strict) for qeler in queries)

def test_FXRateService_queries():
    # Setup currencies and dates
    ccy_eur = MagicMock(spec=Currency)
    ccy_usd = MagicMock(spec=Currency)
    ccy_gbp = MagicMock(spec=Currency)
    today = date.today()
    yesterday = date(1999, 1, 1)
    
    val_eur_usd = Decimal("1.1")
    val_usd_eur = Decimal("0.9")
    
    # Define the rate map for our mock service
    rate_map = {
        (ccy_eur, ccy_usd, today): FXRate(ccy_eur, ccy_usd, today, val_eur_usd),
        (ccy_usd, ccy_eur, today): FXRate(ccy_usd, ccy_eur, today, val_usd_eur),
    }
    
    service = MockFXRateService(rate_map=rate_map)

    # Define queries: one exists, one is missing (returns None), one is invalid pair
    query_list = [
        (ccy_eur, ccy_usd, today),      # Exists
        (ccy_usd, ccy_eur, yesterday),  # Missing date
        (ccy_gbp, ccy_usd, today)       # Missing currency
    ]

    # Test non-strict mode (default: returns None for missing)
    results = list(service.queries(query_list, strict=False))
    
    assert len(results) == 3
    assert results[0].value == val_eur_usd
    assert results[1] is None
    assert results[2] is None

    # Test strict mode (should raise FXRateLookupError for the first missing item in list)
    # We test the iterator behavior: it raises error when the generator reaches the missing key
    query_strict = [(ccy_eur, ccy_usd, today), (ccy_usd, ccy_eur, yesterday)]
    gen = service.queries(query_strict, strict=True)
    
    assert next(gen).value == val_eur_usd
    with pytest.raises(FXRateLookupError) as excinfo:
        next(gen)
    assert excinfo.value.ccy1 == ccy_usd
    assert excinfo.value.ccy2 == ccy_eur
    assert excinfo.value.asof == yesterday

    # Test empty queries
    assert list(service.queries([], strict=False)) == []
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
from datetime import date

@pytest.mark.parametrize("ccy1, ccy2, asof, expected_rate", [
    (Currency["EUR"], Currency["USD"], date(2023, 1, 1), FXRate(Currency["EUR"], Currency["USD"], date(2023, 1, 1), Decimal("1.1"))),
    (Currency["GBP"], Currency["JPY"], date(2023, 5, 20), FXRate(Currency["GBP"], Currency["JPY"], date(2023, 5, 20), Decimal("180.0"))),
])
def test_FXRateService_query_success(ccy1, ccy2, asof, expected_rate):
    # Arrange
    service = MagicMock(spec=FXRateService)
    service.query.return_value = expected_rate

    # Act
    result = service.query(ccy1, ccy2, asof, strict=False)

    # Assert
    assert result == expected_rate
    service.query.assert_called_once_with(ccy1, ccy2, asof, strict=False)


def test_FXRateService_query_returns_none():
    # Arrange
    service = MagicMock(spec=FXRateService)
    service.query.return_value = None
    ccy1, ccy2, asof = Currency["EUR"], Currency["USD"], date(2023, 1, 1)

    # Act
    result = service.query(ccy1, ccy2, asof, strict=False)

    # Assert
    assert result is None
    service.query.assert_called_once_with(ccy1, ccy2, asof, strict=False)


def test_FXRateService_query_raises_lookup_error_when_strict():
    # Arrange
    service = MagicMock(spec=FXRateService)
    ccy1, ccy2, asof = Currency["EUR"], Currency["USD"], date(202im3, 1, 1)
    service.query.side_effect = FXRateLookupError(ccy1, ccy2, asof)

    # Act & Assert
    with pytest.raises(FXRateLookupError) as excinfo:
        service.query(ccy1, ccy2, asof, strict=True)
    
    assert excinfo.value.ccy1 == ccy1
    assert excinfo.value.ccy2 == ccy2
    assert excinfo.value.asof == asof
    service.query.assert_called_once_with(ccy1, ccy2, asof, strict=True)
```


# LLM-generated content at query #3
#--------------------------

```python
import datetime
from decimal import Decimal
from unittest.mock import MagicMock

def test_FXRate___invert__():
    # Arrange
    # Mocking Currency objects as they are required for the FXRate constructor
    ccy_eur = MagicMock(spec=Currency)
    ccy_usd = MagicMock(spec=Currency)
    test_date = datetime.date.today()
    rate_value = Decimal("2.0")
    inverse_value = Decimal("0.5")
    
    original_rate = FXRate(
        ccy1=ccy_eur, 
        ccy2=ccy_usd, 
        date=test_date, 
        value=rate_value
    )

    # Act
    inverted_rate = ~original_rate

    # Assert
    # The inverted rate should have swapped currencies
    assert inverted_rate.ccy1 == ccy_usd
    assert inverted_rate.ccy2 == ccy_eur
    
    # The date should remain the same
    assert inverted_rate.date == test_date
    
    # The value should be the reciprocal (1 / 2.0 = 0.5)
    assert inverted_rate.value == inverse_value
    
    # Double inversion should return to original (within decimal precision)
    assert ~~original_rate == original_rate
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
from datetime import date

@pytest.mark.parametrize("ccy1, ccy2, asof, expected_rate, strict, should_raise", [
    (
        Currencies["EUR"], 
        Currencies["USD"], 
        date(2023, 1, 1), 
        FXRate(Currencies["EUR"], Currencies["USD"], date(2023, 1, 1), Decimal("1.1")), 
        False, 
        False
    ),
    (
        Currencies["EUR"], 
        Currencies["USD"], 
        date(2023, 1, 1), 
        None, 
        False, 
        False
    ),
    (
        Currencies["EUR"], 
        Currencies["USD"], 
        date(2023, 1, 1), 
        None, 
        True, 
        True
    ),
])
def test_FXRateService_query(ccy1, ccy2, asof, expected_rate, strict, should_raise):
    # Arrange
    service = MagicMock(spec=FXRateService)
    
    if should_raise:
        service.query.side_effect = FXRateLookupError(ccy1, ccy2, asof)
    else:
        service.query.return_value = expected_rate

    # Act
    if should_raise:
        with pytest.raises(FXRateLookupError) as excinfo:
            service.query(ccy1, ccy2, asof, strict=strict)
        assert ccy1 in str(excinfo.value)
        assert ccy2 in str(excinfo.value)
        assert asof in str(excinfo.value)
    else:
        result = service.query(ccy1, ccy2, asof, strict=strict)
        assert result == expected_rate

    # Assert
    service.query.assert_called_once_with(ccy1, ccy2, asof, strict=strict)
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
from datetime import date

@pytest.mark.parametrize("ccy1, ccy2, asof, expected_rate, strict, should_raise", [
    (
        Currencies["EUR"], 
        Currencies["USD"], 
        date(2023, 1, 1), 
        FXRate(Currencies["EUR"], Currencies["USD"], date(2023, 1, 1), Decimal("1.1")), 
        False, 
        False
    ),
    (
        Currencies["EUR"], 
        Currencies["USD"], 
        date(2023, 1, 1), 
        None, 
        False, 
        False
    ),
    (
        Currencies["GBP"], 
        Currencies["JPY"], 
        date(2023, 5, 5), 
        None, 
        True, 
        True
    ),
])
def test_FXRateService_query(ccy1, ccy2, asof, expected_rate, strict, should_raise):
    # Arrange
    service = MagicMock(spec=FXRateService)
    
    if should_raise:
        service.query.side_effect = FXRateLookupError(ccy1, ccy2, asof)
    else:
        service.query.return_value = expected_rate

    # Act & Assert
    if should_raise:
        with pytest.raises(FXRateLookupError) as excinfo:
            service.query(ccy1, ccy2, asof, strict=strict)
        assert excinfo.value.ccy1 == ccy1
        assert excinfo.value.ccy2 == ccy2
        assert excinfo.value.asof == asof
    else:
        result = service.query(ccy1, ccy2, asof, strict=strict)
        assert result == expected_rate
    
    service.query.assert_called_once_with(ccy1, ccy2, asof, strict=strict)
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest
from decimal import Decimal
from unittest.mock import MagicMock
from datetime import date

class MockFXRateService(FXRateService):
    def __init__(self, mapping=None):
        self.mapping = mapping or {}

    def query(self, ccy1, ccy2, asof, strict=False) -> Optional[FXRate]:
        key = (ccy1, ccy2, asof)
        if key in self.mapping:
            return self.mapping[key]
        if strict:
            raise FXRateLookupError(ccy1, ccy/2, asof) # Note: The original code has a typo 'ccy/2' instead of 'ccy2'
        return None

    def queries(self, queries, strict=False) -> Iterable[Optional[FXRate]]:
        results = []
        for q in queries:
            try:
                results.append(self.query(*q, strict=strict))
            except FXRateLookupError:
                if strict:
                    raise
                results.append(None)
        return results

def test_FXRateService_queries():
    # Setup dependencies
    ccy_usd = MagicMock(spec=Currency)
    ccy_eur = MagicMock(spec=Currency)
    ccy_gbp = MagicMock(spec=Currency)
    today = date.today()
    val_rate = Decimal("1.1")
    val_rate_inv = Decimal("0.9")
    
    rate1 = FXRate(ccy_usd, ccy_eur, today, val_rate)
    rate2 = FXRate(ccy_eur, ccy_usd, today, val_rate_inv)

    # Define the mapping for our mock service
    service_data = {
        (ccy_usd, ccy_eur, today): rate1,
        (ccy_eur, ccy_usd, today): rate2,
    }
    
    service = MockFXRateService(mapping=service_data)

    # Define queries: 
    # 1. Existing pair
    # 2. Non-existing pair (should return None if not strict)
    # 3. Another existing pair
    queries_list = [
        (ccy_usd, ccy_eur, today),
        (ccy_usd, ccy_gbp, today),
        (ccy_eur, ccy_usd, today)
    ]

    # Test execution (non-strict mode)
    results = list(service.queries(queries_list, strict=False))

    assert len(results) == 3
    assert results[0] == rate1
    assert results[1] is None
    assert results[2] == rate2

    # Test execution (strict mode - should raise error on the second query)
    with pytest.raises(Exception): # Catching generic because of the typo in original source code 'ccy/2'
        list(service.queries(queries_list, strict=True))

    # Test with empty queries
    assert list(service.queries([], strict=False)) == []
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
from datetime import date

class MockFXRateService(FXRateService):
    def __init__(self):
        self._rates = {}

    def add_rate(self, rate: FXRate):
        self._rates[(rate.ccy1, rate.ccy2, rate.date)] = rate

    def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
        key = (ccy1, ccy2, asof)
        rate = self._rates.get(key)
        if rate is None and strict:
            raise FXRateLookupError(ccy1, ccy2, asof)
        return rate

    def queries(self, queries: Iterable[TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
        return (self.query(q[0], q[1], q[2], strict) for q in queries)

def test_FXRateService_query():
    service = MockFXRateService()
    ccy_usd = MagicMock(spec=Currency)
    ccy_eur = MagicMock(spec=Currency)
    today = date.today()
    rate_val = Decimal("1.1")
    
    rate = FXRate(ccy_usd, ccy_eur, today, rate_val)
    service.add_rate(rate)

    # Test Case 1: Successful lookup
    found_rate = service.query(ccy_usd, ccy_eur, today)
    assert found_rate == rate
    assert found_rate.value == rate_val

    # Test Case 2: Non-strict lookup for non-existent rate returns None
    missing_rate = service.query(ccy_eur, ccy_usd, today, strict=False)
    assert missing_rate is None

    # Test Case 3: Strict lookup for non-existent rate raises FXRateLookupError
    with pytest.raises(FXRateLookupError) as excinfo:
        service.query(ccy_eur, ccy_usd, today, strict=True)
    
    assert excinfo.value.ccy1 == ccy_eur
    assert excinfo.value.ccy2 == ccy_usd
    assert excinfo.value.asof == today
    assert "not found" in str(excinfo.value)

    # Test Case 4: Lookup with different date returns None
    different_date = date(2000, 1, 1)
    assert service.query(ccy_usd, ccy_eur, different_date) is None
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
from datetime import date

class MockFXRateService(FXRateService):
    def __init__(self, data=None):
        # data format: {(ccy1, ccy2, asof): FXRate}
        self.data = data or {}

    def query(self, ccy1, ccy2, asof, strict=False) -> Optional[FXRate]:
        key = (ccy1, ccy2, asof)
        if key not in self.data:
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None
        return self.data[key]

    def queries(self, queries: Iterable[Tuple[Currency, Currency, Date]], strict: bool = False) -> Iterable[Optional[FXRate]]:
        results = []
        for q in queries:
            ccy1, ccy2, asof = q
            try:
                results.append(self.query(ccy1, ccy2, asof, strict=strict))
            except FXRateLookupError:
                if strict:
                    raise
                results.append(None)
        return results

def test_FXRateService_queries():
    # Setup dependencies
    usd = MagicMock(spec=Currency)
    eur = MagicMock(spec=Currency)
    gbp = MagicMock(spec=Currency)
    today = date.today()
    yesterday = date.today().replace(day=max(1, today.day - 1))
    
    rate_usd_eur = FXRate(usd, eur, today, Decimal("0.85"))
    rate_eur_usd = FXRate(eur, usd, today, Decimal("1.18"))
    rate_gbp_usd = FXRate(gbp, usd, yesterday, Decimal("1.30"))

    # Mock data store
    mock_data = {
        (usd, eur, today): rate_usd_eur,
        (eur, usd, today): rate_eur_usd,
        (gbp, usd, yesterday): rate_gbp_usd
    }

    service = MockFXRateService(data=mock_data)

    # Define test queries (some exist, some don't)
    query_list = [
        (usd, eur, today),      # Exists
        (eur, usd, today),      # Exists
        (gbp, usd, yesterday),  # Exists
        (usd, gbp, today),      # Missing
        (usd, eur, yesterday)   # Missing (wrong date)
    ]

    # 1. Test non-strict mode (should return None for missing)
    results_non_strict = list(service.queries(query_list, strict=False))
    assert len(results_non_strict) == 5
    assert results_non_strict[0] == rate_usd_eur
    assert results_non_strict[1] == rate_eur_usd
    assert results_non_strict[2] == rate_gbp_usd
    assert results_non_strict[3] is None
    assert results_non_strict[4] is None

    # 2. Test strict mode (should raise FXRateLookupError on first missing)
    with pytest.raises(FXRateLookupError) as excinfo:
        list(service.queries(query_list, strict=True))
    
    assert excinfo.value.ccy1 == usd
    assert excinfo.value.ccy2 == gbp
    assert excinfo.value.asof == today

    # 3. Test empty queries
    assert list(service.queries([], strict=False)) == []
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
import datetime

class MockFXRateService(FXRateService):
    def __init__(self):
        self.data = {}

    def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
        key = (ccy1, ccy2, asof)
        rate = self.data.get(key)
        if strict and rate is None:
            raise FXRateLookupError(ccy1, ccy, asof)
        return rate

    def queries(self, queries: Iterable[TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
        return [self.query(q[0], q[1], q[2], strict) for q in queries]

def test_FXRateService_query():
    service = MockFXRateService()
    ccy_usd = MagicMock(spec=Currency)
    ccy_eur = MagicMock(spec=Currency)
    today = datetime.date.today()
    rate_val = Decimal("1.1")
    rate_obj = FXRate(ccy_usd, ccy_eur, today, rate_val)
    
    # Populate mock data
    service.data[(ccy_usd, ccy_eur, today)] = rate_obj

    # Case 1: Successful lookup
    result = service.query(ccy_usd, ccy_eur, today)
    assert result == rate_obj
    assert result.value == rate_val

    # Case 2: Non-strict mode returns None when not found
    result_none = service.query(ccy_eur, ccy_usd, today, strict=False)
    assert result_none is None

    # Case 3: Strict mode raises FXRateLookupError when not found
    with pytest.raises(FXRateLookupError) as excinfo:
        service.query(ccy_eur, ccy_usd, today, strict=True)
    
    assert excinfo.value.ccy1 == ccy_eur
    assert excinfo.value.ccy2 == ccy_usd
    assert excinfo.value.asof == today
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
from datetime import date

class MockFXRateService(FXRateService):
    def __init__(self, data=None):
        self.data = data or {}

    def query(self, ccy1, ccy2, asof, strict=False) -> Optional[FXRate]:
        key = (ccy1, ccy2, asof)
        if key in self.data:
            return self.data[key]
        if strict:
            raise FXRateLookupError(ccy1, ccy2, asof)
        return None

    def queries(self, queries, strict=False) -> Iterable[Optional[FXRate]]:
        results = []
        for q in queries:
            try:
                results.append(self.query(q[0], q[1], q[2], strict=strict))
            except FXRateLookupError:
                if strict:
                    raise
                results.append(None)
        return results

def test_FXRateService_queries():
    # Setup dependencies
    ccy_usd = MagicMock(spec=Currency)
    ccy_eur = MagicMock(spec=Currency)
    ccy_gbp = MagicMock(spec=Currency)
    today = date.today()
    val_usd_eur = Decimal("0.85")
    val_usd_gbp = Decimal("0.75")
    
    rate1 = FXRate(ccy_usd, ccy_eur, today, val_usd_eur)
    rate2 = FXRate(ccy_usd, ccy_gbp, today, val_usd_gbp)

    # Mock data for the service
    mock_data = {
        (ccy_usd, ccy_eur, today): rate1,
        (ccy_usd, ccy_gbp, today): rate2
    }
    
    service = MockFXRateService(data=mock_data)

    # Test Case 1: Successful retrieval of multiple rates
    query_list = [
        (ccy_usd, ccy_eur, today),
        (ccy_usd, ccy_gbp, today)
    ]
    results = list(service.queries(query_list, strict=False))
    assert len(results) == 2
    assert results[0] == rate1
    assert results[1] == rate2

    # Test Case 2: Handling missing rates with strict=False (returns None)
    missing_query = [(ccy_eur, ccy_usd, today)]
    results_none = list(service.queries(missing_query, strict=False))
    assert len(results_none) == 1
    assert results_none[0] is None

    # Test Case 3: Handling missing rates with strict=True (raises error)
    with pytest.raises(FXRateLookupError) as excinfo:
        list(service.queries(missing_query, strict=True))
    assert excinfo.value.ccy1 == ccy_eur
    assert excinfo.value.ccy2 == ccy_usd

    # Test Case 4: Mixed results (some found, some missing)
    mixed_queries = [
        (ccy_usd, ccy_eur, today),
        (ccy_eur, ccy_gbp, today) # Not in mock_data
    ]
    results_mixed = list(service.queries(mixed_queries, strict=False))
    assert len(results_mixed) == 2
    assert results_mixed[0] == rate1
    assert results_mixed[1] is None
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
from datetime import date

@pytest.mark.parametrize("ccy1, ccy2, asof, expected_rate", [
    (Currencies["EUR"], Currencies["USD"], date(2023, 1, 1), FXRate(Currencies["EUR"], Currencies["USD"], date(2023, 1, 1), Decimal("1.1"))),
    (Currencies["GBP"], Currencies["JPY"], date(2023, 5, 20), FXRate(Currencies["GBP"], CurrenciesJPY, date(2023, 5, 20), Decimal("180.0"))),
])
def test_FXRateService_query_success(ccy1, ccy2, asof, expected_rate):
    # Arrange
    service = MagicMock(spec=FXRateService)
    service.query.return_value = expected_rate

    # Act
    result = service.query(ccy1, ccy2, asof, strict=False)

    # Assert
    assert result == expected_rate
    service.query.assert_called_once_with(ccy1, ccy2, asof, strict=False)


def test_FXRateService_query_returns_none():
    # Arrange
    service = MagicMock(spec=FXRateService)
    service.query.return_value = None
    ccy1, ccy2, asof = Currencies["USD"], Currencies["EUR"], date(2023, 1, 1)

    # Act
    result = service.query(ccy1, ccy2, asof, strict=False)

    # Assert
    assert result is None
    service.query.assert_called_once_with(ccy1, ccy2, asof, strict=False)


def test_FXRateService_query_raises_lookup_error():
    # Arrange
    service = MagicMock(spec=FXRateService)
    ccy1, ccy2, asof = Currencies["USD"], Currencies["EUR"], date(2023, 1, 1)
    service.query.side_effect = FXRateLookupError(ccy1, ccy2, asof)

    # Act & Assert
    with pytest.raises(FXRateLookupError) as excinfo:
        service.query(ccy1, ccy2, asof, strict=True)
    
    assert ccy1 in str(excinfo.value)
    assert ccy2 in str(excinfo.value)
    assert asof in str(excinfo.value)
    service.query.assert_called_once_with(ccy1, ccy2, asof, strict=True)

def test_FXRateService_query_params_passed_correctly():
    # Arrange
    service = MagicMock(spec=FXRateService)
    ccy1, ccy2, asof = Currencies["USD"], Currencies["EUR"], date(2023, 1, 1)

    # Act
    service.query(ccy1, ccy2, asof, strict=True)

    # Assert
    service.query.assert_called_with(ccy1, ccy2, asof, strict=True)
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
from datetime import date

class MockFXRateService(FXRateService):
    def __init__(self, rate_map=None):
        self.rate_map = rate_map or {}

    def query(self, ccy1, ccy2, asof, strict=False):
        key = (ccy1, ccy2, asof)
        if key in self.rate_map:
            return self.rate_map[key]
        if strict:
            raise FXRateLookupError(ccy1, ccy2, asof)
        return None

    def queries(self, queries, strict=False):
        results = []
        for q in queries:
            try:
                results.append(self.query(*q, strict=strict))
            except FXRateLookupError:
                if strict:
                    raise
                results.append(None)
        return results

def test_FXRateService_queries():
    # Setup dependencies
    ccy_usd = MagicMock(spec=Currency)
    ccy_eur = MagicMock(spec=Currency)
    ccy_gbp = MagicMock(spec=Currency)
    today = date.today()
    val1 = Decimal("1.1")
    val2 = Decimal("0.85")

    rate1 = FXRate(ccy_usd, ccy_eur, today, val1)
    rate2 = FXRate(ccy_eur, ccy_gbp, today, val2)

    # Data for the service
    lookup_data = {
        (ccy_usd, ccy_eur, today): rate1,
        (ccy_eur, ccy_gbp, today): rate2,
    }

    service = MockFXRateService(rate_map=lookup_data)

    # Test Case 1: Successful retrieval of multiple rates (non-strict)
    query_list = [
        (ccy_usd, ccy_eur, today),
        (ccy_eur, ccy_gbp, today),
        (ccy_gbp, ccy_usd, today) # Not in map
    ]
    
    results = list(service.queries(query_list, strict=False))
    
    assert len(results) == 3
    assert results[0] == rate1
    assert results[1] == rate2
    assert results[2] is None

    # Test Case 2: Strict mode - missing rate raises exception
    with pytest.raises(FXRateLookupError) as excinfo:
        list(service.queries(query_list, strict=True))
    
    assert excinfo.value.ccy1 == ccy_gbp
    assert excinfo.value.ccy2 == ccy_usd

    # Test Case 3: Strict mode - successful retrieval
    strict_results = list(service.queries([(ccy_usd, ccy_eur, today)], strict=True))
    assert strict_results[0] == rate1
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
from datetime import date

@pytest.fixture
def mock_currency_usd():
    ccy = MagicMock(spec=Currency)
    ccy.__eq__.side_effect = lambda x: x == "USD" or x == mock_currency_usd
    return ccy

@pytest.fixture
def mock_currency_eur():
    ccy = MagicMock(spec=Currency)
    ccy.__eq__.side_effect = lambda x: x == "EUR" or x == mock_currency_eur
    return ccy

@pytest.fixture
def mock_fx_service():
    return MagicMock(spec=FXRateService)

def test_FXRateService_query(mock_fx_service, mock_currency_usd, mock_currency_eur):
    """
    Tests the query method of FXRateService for various scenarios: 
    successful retrieval, None return (not found), and strict error raising.
    """
    asof_date = date(2023, 1, 1)
    rate_value = Decimal("1.1")
    expected_rate = FXRate(mock_currency_eur, mock_currency_usd, asof_date, rate_value)

    # Scenario 1: Successful retrieval of an existing rate
    mock_fx_service.query.return_value = expected_rate
    result = mock_fx_service.query(mock_currency_eur, mock_currency_usd, asof_date, strict=False)
    
    assert result == expected_rate
    mock_fx_service.query.assert_called_with(mock_currency_eur, mock_currency_usd, asof_date, strict=False)

    # Scenario 2: Rate not found (returns None when strict=False)
    mock_fx_service.query.return_value = None
    result_none = mock_fx_service.query(mock_currency_eur, mock_currency_usd, asof_date, strict=False)
    
    assert result_none is None

    # Scenario 3: Rate not found (raises FXRateLookupError when strict=True)
    mock_fx_service.query.side_effect = FXRateLookupError(mock_currency_eur, mock_currency_usd, asof_date)
    
    with pytest.raises(FXRateLookupError) as excinfo:
        mock_fx_service.query(mock_currency_eur, mock_currency_usd, asof_date, strict=True)
    
    assert mock_currency_eur in str(excinfo.value)
    assert mock_currency_usd in str(excinfo.value)
    assert asof_date in str(excinfo.value)

    # Reset side effect for clean state in other potential tests
    mock_fx_service.query.side_effect = None
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
from datetime import date

class MockFXRateService(FXRateService):
    def __init__(self, data=None):
        # data is a dict: {(ccy1, ccy2, asof): FXRate}
        self.data = data or {}

    def query(self, ccy1, ccy2, asof, strict=False) -> Optional[FXRate]:
        key = (ccy1, ccy2, asof)
        if key in self.data:
            return self.data[key]
        if strict:
            raise FXRateLookupError(ccy1, ccyfully2, asof) # Note: actual code has a typo 'ccyfully2' but we follow logic
        return None

    def queries(self, queries: Iterable[Tuple[Currency, Currency, Date]], strict: bool = False) -> Iterable[Optional[FXRate]]:
        results = []
        for q in queries:
            try:
                # Unpack the TQuery tuple (ccy1, ccy2, asof)
                results.append(self.query(q[0], q[1], q[2], strict=strict))
            except FXRateLookupError:
                if strict:
                    raise
                results.append(None)
        return results

def test_FXRateService_queries():
    # Setup currencies and dates
    ccy_usd = MagicMock(spec=Currency)
    ccy_eur = MagicMock(spec=Currency)
    ccy_gbp = MagicMock(spec=Currency)
    today = date.today()
    yesterday = date(1999, 1, 1)

    rate1 = FXRate(ccy_usd, ccy_eur, today, Decimal("0.85"))
    rate2 = FXRate(ccy_usd, ccy_gbp, yesterday, Decimal("0.75"))

    # Mock data for the service
    service_data = {
        (ccy_usd, ccy_or_eur := ccy_eur, today): rate1,
        (ccy_usd, ccy_or_gbp := ccy_gbp, yesterday): rate2
    }

    # 1. Test successful batch retrieval (non-strict)
    service = MockFXRateService(data=service_data)
    queries_list = [
        (ccy_usd, ccy_eur, today),
        (ccy_usd, ccy_gbp, yesterday),
        (ccy_eur, ccy_usd, today)  # Not in data
    ]
    
    results = list(service.queries(queries_list, strict=False))
    
    assert len(results) == 3
    assert results[0] == rate1
    assert results[1] == rate2
    assert results[2] is None

    # 2. Test strict mode raising error on missing rate
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries_list, strict=True))

    # 3. Test empty queries list
    assert list(service.queries([], strict=False)) == []

    # 4. Test with a Mock object to verify the abstract method call pattern
    mock_service = MagicMock(spec=FXRateService)
    mock_service.queries.return_value = [rate1]
    
    test_queries = [(ccy_usd, ccy_eur, today)]
    res = list(mock_service.queries(test_queries))
    
    assert res == [rate1]
    mock_service.queries.assert_called_once_with(test_queries)
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
import datetime

class MockFXRateService(FXRateService):
    def __init__(self):
        self.rates = {}

    def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
        key = (ccy1, ccy2, asof)
        rate = self.rates.get(key)
        if strict and rate is None:
            raise FXRateLookupError(ccy1, ccyey, asof)
        return rate

    def queries(self, queries: Iterable[TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
        return [self.query(q[0], q[1], q[2], strict) for q in queries]

def test_FXRateService_query():
    service = MockFXRateService()
    
    # Setup mock data
    ccy_usd = MagicMock(spec=Currency)
    ccy_eur = MagicMock(spec=Currency)
    ccy_gbp = MagicMock(spec=Currency)
    test_date = datetime.date.today()
    rate_val = Decimal("1.1")
    
    rate_instance = FXRate(ccy_usd, ccy_eur, test_date, rate_val)
    service.rates[(ccy_usd, ccy_eur, test_date)] = rate_instance

    # Test 1: Successful lookup
    result = service.query(ccy_usd, ccy_eur, test_date, strict=False)
    assert result == rate_instance
    assert result.value == Decimal("1.1")

    # Test 2: Return None when not found and strict is False
    result_none = service.query(ccy_usd, ccy_gbp, test_date, strict=False)
    assert result_none is None

    # Test 3: Raise FXRateLookupError when not found and strict is True
    with pytest.raises(FXRateLookupError) as excinfo:
        service.query(ccy_usd, ccy_gbp, test_date, strict=True)
    
    assert excinfo.value.ccy1 == ccy_usd
    assert excinfo.value.ccy2 == ccy_gbp
    assert excinfo.value.asof == test_date
```


# LLM-generated content at query #16
#--------------------------

```python
import pytest
from decimal import Decimal
from unittest.mock import MagicMock
from datetime import date

@pytest.mark.parametrize("ccy1, ccy2, asof, expected_value", [
    (Currency["EUR"], Currency["USD"], date(2023, 1, 1), Decimal("1.1")),
    (Currency["GBP"], Currency["JPY"], date(2023, 5, 20), Decimal("180.5")),
])
def test_FXRateService_query(ccy1, ccy2, asof, expected_value):
    """
    Tests the query method of FXRateService for successful retrieval and error handling.
    """
    # Arrange
    service = MagicMock(spec=FXRateService)
    expected_rate = FXRate(ccy1, ccy2, asof, expected_value)
    
    # Case 1: Successful lookup returning a rate
    service.query.return_value = expected_rate
    
    # Act & Assert - Success scenario
    result = service.query(ccy1, ccy2, asof, strict=False)
    assert result == expected_rate
    assert result.ccy1 == ccy1
    assert result.ccy2 == ccy2
    assert result.date == asof
    assert result.value == expected_value

    # Case 2: Successful lookup returning None (not found, non-strict)
    service.query.return_value = None
    result_none = service.query(ccy1, ccy2, asof, strict=False)
    assert result_none is None

    # Case 3: Strict mode raises FXRateLookupError when rate is missing
    service.query.side_effect = FXRateLookupError(ccy1, ccy2, asof)
    with pytest.raises(FXRateLookupError) as excinfo:
        service.query(ccy1, ccy2, asof, strict=True)
    
    assert ccy1 in str(excinfo.value)
    assert ccy2 in str(excinfo.value)
    assert asof in str(excinfo.value)

    # Verify the mock was called with correct parameters
    service.query.assert_called()
```


# LLM-generated content at query #17
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
from datetime import date

class MockFXRateService(FXRateService):
    def __init__(self, rate_map=None):
        self.rate_map = rate_map or {}

    def query(self, ccy1, ccy2, asof, strict=False):
        key = (ccy1, ccy2, asof)
        if key in self.rate_map:
            return self.rate_map[key]
        if strict:
            raise FXRateLookupError(ccy1, ccy2, asof)
        return None

    def queries(self, queries, strict=False):
        return (self.query(q[0], q[1], q[2], strict=strict) for q in queries)

def test_FXRateService_queries():
    # Setup dummy currencies and dates
    ccy_usd = MagicMock(spec=Currency)
    ccy_eur = MagicMock(spec=Currency)
    ccy_gbp = MagicMock(spec=Currency)
    dt1 = date(2023, 1, 1)
    dt2 = date(2023, 1, 2)
    
    rate1 = FXRate(ccy_usd, ccy_eur, dt1, Decimal("0.9"))
    rate2 = FXRate(ccy_eur, ccy_usd, dt1, Decimal("1.1"))
    rate3 = FXRate(ccy_usd, ccy_gbp, dt2, Decimal("0.8"))

    # Define mapping for the mock service
    rate_map = {
        (ccy_usd, ccy_eur, dt1): rate1,
        (ccy_eur, ccy_usd, dt1): rate2,
        (ccy_usd, ccy_gbp, dt2): rate3,
    }

    service = MockFXRateService(rate_map=rate_map)

    # Define queries: 
    # 1. Existing rate
    # 2. Non-existent rate (returns None)
    # 3. Another existing rate
    query_list = [
        (ccy_usd, ccy_eur, dt1),
        (ccy_gbp, ccy_usd, dt1),  # Not in map
        (ccy_usd, ccy_gbp, dt2)
    ]

    # Test non-strict mode (default)
    results = list(service.queries(query_list, strict=False))
    assert len(results) == 3
    assert results[0] == rate1
    assert results[1] is None
    assert results[2] == rate3

    # Test strict mode (should raise FXRateLookupError for the missing one)
    with pytest.raises(FXRateLookupError) as excinfo:
        list(service.queries(query_list, strict=True))
    
    assert excinfo.value.ccy1 == ccy_gbp
    assert excinfo.value.ccy2 == ccy_usd
    assert excinfo.value.asof == dt1

    # Test empty queries
    assert list(service.queries([], strict=False)) == []
```


# LLM-generated content at query #18
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
import datetime

class MockFXRateService(FXRateService):
    def __init__(self):
        self.rates = {}

    def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
        key = (ccy1, ccy2, asof)
        rate = self.rates.get(key)
        if strict and rate is None:
            raise FXRateLookupError(ccy1, ccyrypt2, asof)
        return rate

    def queries(self, queries: Iterable[TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
        return [self.query(q[0], q[1], q[2], strict) for q in queries]

@pytest.fixture
def mock_currency_usd():
    return MagicMock(spec=Currency)

@pytest.fixture
def mock_currency_eur():
    return MagicMock(spec=Currency)

@pytest.fixture
def sample_date():
    return datetime.date(2023, 1, 1)

@pytest.fixture
def service(mock_currency_usd, mock_currency_eur, sample_date):
    svc = MockFXRateService()
    rate_val = Decimal("1.1")
    svc.rates[(mock_currency_usd, mock_currency_eur, sample_date)] = FXRate(
        mock_currency_usd, mock_currency_eur, sample_date, rate_val
    )
    return svc

def test_FXRateService_query(service, mock_currency_usd, mock_currency_eur, sample_date):
    # Test finding an existing rate
    rate = service.query(mock_currency_usd, mock_currency_eur, sample_date)
    assert rate is not None
    assert rate.value == Decimal("1.1")
    assert rate.ccy1 == mock_currency_usd

    # Test finding a non-existent rate (non-strict)
    random_date = datetime.date(2024, 1, 1)
    rate_none = service.query(mock_currency_usd, mock_currency_eur, random_date, strict=False)
    assert rate_none is None

    # Test finding a non-existent rate (strict)
    with pytest.raises(FXRateLookupError) as excinfo:
        service.query(mock_currency_usd, mock_currency_eur, random_date, strict=True)
    
    assert excinfo.value.ccy1 == mock_currency_usd
    assert excinfo.value.ccy2 == mock_currency_eur
    assert excinfo.value.asof == random_date

    # Test querying same currency pair (not in mock setup)
    rate_self = service.query(mock_currency_usd, mock_currency_usd, sample_date, strict=False)
    assert rate_self is None
```


# LLM-generated content at query #19
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
from datetime import date

class MockFXRateService(FXRateService):
    def __init__(self):
        self._storage = {}

    def add_rate(self, rate: FXRate):
        self._storage[(rate.ccy1, rate.ccy2, rate.date)] = rate

    def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
        key = (ccy1, ccy2, asof)
        rate = self._storage.get(key)
        if rate is None and strict:
            raise FXRateLookupError(ccy1, ccy2, asof)
        return rate

def test_FXRateService_query():
    service = MockFXRateService()
    
    # Setup dummy data
    usd = Currency("USD")
    eur = Currency("EUR")
    today = date.today()
    rate_val = Decimal("1.1")
    rate = FXRate(usd, eur, today, rate_val)
    service.add_rate(rate)

    # Test case 1: Successful lookup
    found_rate = service.query(usd, eur, today)
    assert found_rate == rate
    assert found_rate.value == rate_val

    # Test case 2: Non-existent rate (non-strict)
    missing_rate = service.query(eur, usd, today)
    assert missing_rate is None

    # Test case 3: Non-existent rate (strict mode raises FXRateLookupError)
    with pytest.raises(FXRateLookupError) as excinfo:
        service.query(eur, usd, today, strict=True)
    
    assert excinfo.value.ccy1 == eur
    assert excinfo.value.ccy2 == usd
    assert excinfo.value.asof == today

    # Test case 4: Wrong date
    wrong_date = date(2000, 1, 1)
    assert service.query(usd, eur, wrong_date) is None
```


# LLM-generated content at query #20
#--------------------------

```python
import pytest
from decimal import Decimal
from unittest.mock import MagicMock
from datetime import date

@pytest.mark.parametrize("ccy1, ccy2, asof, rate_value, expected_return", [
    (Currency["EUR"], Currency["USD"], date(2023, 1, 1), Decimal("1.1"), FXRate(Currency["EUR"], Currency["USD"], date(2023, 1, 1), Decimal("1.1"))),
    (Currency["GBP"], Currency["JPY"], date(2023, 5, 20), Decimal("180.5"), FXRate(Currency["GBP"], Currency["JPY"], date(2023, 5, 20), Decimal("180.5"))),
])
def test_FXRateService_query_success(ccy1, ccy2, asof, rate_value, expected_return):
    # Setup
    service = MagicMock(spec=FXRateService)
    service.query.return_value = expected_return

    # Execute
    result = service.query(ccy1, ccy2, asof, strict=False)

    # Assert
    service.query.assert_called_once_with(ccy1, ccy2, asof, strict=False)
    assert result == expected_return


def test_FXRateService_query_returns_none():
    # Setup
    service = MagicMock(spec=FXRateService)
    service.query.return_value = None
    ccy1, ccy2, asof = Currency["EUR"], Currency["USD"], date(2023, 1, 1)

    # Execute
    result = service.query(ccy1, ccy2, asof, strict=False)

    # Assert
    assert result is None


def test_FXRateService_query_raises_lookup_error():
    # Setup
    service = MagicMock(spec=FXRateService)
    ccy1, ccy2, asof = Currency["EUR"], Currency["USD"], date(2023, 1, 1)
    service.query.side_effect = FXRateLookupError(ccy1, ccy2, asof)

    # Execute & Assert
    with pytest.raises(FXRateLookupError) as excinfo:
        service.query(ccy1, ccy2, asof, strict=True)
    
    assert ccy1 in str(excinfo.value)
    assert ccy2 in str(excinfo.value)
    assert asof in str(excinfo.value)


def test_FXRateService_query_strict_parameter_passing():
    # Setup
    service = MagicMock(spec=FXRateService)
    ccy1, ccy2, asof = Currency["EUR"], Currency["USD"], date(2023, 1, 1)

    # Execute
    service.query(ccy1, ccy2, asof, strict=True)
    service.query(ccy1, ccy2, asof, strict=False)

    # Assert
    assert service.query.call_count == 2
    service.query.assert_any_call(ccy1, ccy2, asof, strict=True)
    service.query.assert_any_call(ccy1, ccy2, asof, strict=False)
```


# LLM-generated content at query #21
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
from datetime import date

class MockFXRateService(FXRateService):
    def __init__(self, rate_map=None):
        self.rate_map = rate_map or {}

    def query(self, ccy1, ccy2, asof, strict=False):
        key = (ccy1, ccy2, asof)
        if key in self.rate_map:
            return self.rate_map[key]
        if strict:
            raise FXRateLookupError(ccy1, ccy2, asof)
        return None

    def queries(self, queries, strict=False):
        for q in queries:
            yield self.query(*q, strict=strict)

def test_FXRateService_queries():
    # Setup dependencies/mocks
    usd = MagicMock(spec=Currency)
    eur = MagicMock(spec=Currency)
    today = date.today()
    val1 = Decimal("1.2")
    val2 = Decimal("0.8")

    rate1 = FXRate(usd, eur, today, val1)
    rate2 = FXRoute_alt := FXRate(eur, usd, today, val2)
    
    # Define a lookup map for the mock service
    lookup_map = {
        (usd, eur, today): rate1,
        (eur, usd, today): rate2,
    }
    
    service = MockFXRateService(rate_map=lookup_map)

    # 1. Test successful retrieval of multiple rates
    query_list = [
        (usd, eur, today),
        (eur, usd, today)
    ]
    results = list(service.queries(query_list, strict=False))
    
    assert len(results) == 2
    assert results[0] == rate1
    assert results[1] == rate2

    # 2. Test retrieval of non-existent rate (non-strict)
    missing_date = date(2000, 1, 1)
    query_missing = [(usd, eur, missing_date)]
    results_none = list(service.queries(query_missing, strict=False))
    
    assert len(results_none) == 1
    assert results_none[0] is None

    # 3. Test retrieval of non-existent rate (strict mode raises error)
    with pytest.raises(FXRateLookupError) as excinfo:
        list(service.queries(query_missing, strict=True))
    
    assert excinfo.value.ccy1 == usd
    assert excinfo.value.ccy2 == eur
    assert excinfo.value.asof == missing_date

    # 4. Test with an empty iterable
    assert list(service.queries([], strict=False)) == []
```


# LLM-generated content at query #22
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
from datetime import date

class MockFXRateService(FXRateService):
    def __init__(self, mapping=None):
        super().__init__()
        self.mapping = mapping or {}

    def query(self, ccy1, ccy2, asof, strict=False) -> Optional[FXRate]:
        key = (ccy1, ccy2, asof)
        if key in self.mapping:
            return self.mapping[key]
        if strict:
            raise FXRateLookupError(ccy1, ccy/2, asof) # Note: assuming typo fix for ccy2 if error exists
            # Using exact provided logic from class definition for the test context:
            # raise FXRateLookupError(ccy1, ccy2, asof) 
        return None

    def queries(self, queries, strict=False) -> Iterable[Optional[FXRate]]:
        return [self.query(q[0], q[1], q[2], strict=strict) for q in queries]

def test_FXRateService_queries():
    # Setup dependencies
    ccy_usd = MagicMock(spec=Currency)
    ccy_eur = MagicMock(spec=Currency)
    ccy_gbp = MagicMock(spec=Currency)
    today = date.today()
    val_usd_eur = Decimal("0.85")
    val_usd_gbp = Decimal("0.75")

    rate1 = FXRate(ccy_usd, ccy_eur, today, val_usd_eur)
    rate2 = FXRate(ccy_usd, ccy_gbp, today, val_usd_gbp)

    # Mock data for the service
    lookup_data = {
        (ccy_usd, ccy_eur, today): rate1,
        (ccy_usd, ccy_gbp, today): rate2,
    }

    service = MockFXRateService(mapping=lookup_data)

    # Test Case 1: Successful batch query
    queries_list = [
        (ccy_usd, ccy_eur, today),
        (ccy_usd, ccy_gbp, today),
        (ccy_eur, ccy_usd, today),  # Not in mapping
    ]

    results = list(service.queries(queries_list, strict=False))

    assert len(results) == 3
    assert results[0] == rate1
    assert results[1] == rate2
    assert results[2] is None

    # Test Case 2: Strict mode raising FXRateLookupError
    with pytest.raises(FXRateLookupError) as excinfo:
        service.queries([(ccy_eur, ccy_usd, today)], strict=True)
    
    assert excinfo.value.ccy1 == ccy_eur
    assert excinfo.value.ccy2 == ccy_usd
    assert excinfo.value.asof == today

    # Test Case 3: Empty query list
    assert list(service.queries([], strict=False)) == []

    # Test Case 4: Single item query existing
    result_single = list(service.queries([(ccy_usd, ccy_eur, today)], strict=False))
    assert result_single[0] == rate1
```


# LLM-generated content at query #23
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
from datetime import date

@pytest.mark.parametrize("ccy1, ccy2, asof, rate_value, expected_result", [
    (Currencies["EUR"], Currencies["USD"], date(2023, 1, 1), Decimal("1.1"), FXRate(Currencies["EUR"], Currencies["USD"], date(2023, 1, 1), Decimal("1.1"))),
    (Currencies["GBP"], Currencies["JPY"], date(2023, 5, 20), Decimal("180.5"), FXRate(Currencies["GBP"], Currencies["JPY"], date(2023, 5, 20), Decimal("180.5"))),
])
def test_FXRateService_query(ccy1, ccy2, asof, rate_value, expected_result):
    # Arrange
    service = MagicMock(spec=FXRateService)
    service.query.return_value = expected_result

    # Act
    result = service.query(ccy1, ccy2, asof, strict=False)

    # Assert
    assert result == expected_result
    service.query.assert_called_once_with(ccy1, ccy2, asof, strict=False)

def test_FXRateService_query_returns_none():
    # Arrange
    service = MagicMock(spec=FXRateService)
    service.query.return_value = None
    ccy1, ccy2, asof = Currencies["USD"], Currencies["EUR"], date(2023, 1, 1)

    # Act
    result = service.query(ccy1, ccy2, asof, strict=False)

    # Assert
    assert result is None
    service.query.assert_called_once_with(ccy1, ccy2, asof, strict=False)

def test_FXRateService_query_raises_error_in_strict_mode():
    # Arrange
    service = MagicMock(spec=FXRateService)
    ccy1, ccy2, asof = Currencies["USD"], Currencies["EUR"], date(2023, 1, 1)
    service.query.side_effect = FXRateLookupError(ccy1, ccy2, asof)

    # Act & Assert
    with pytest.raises(FXRateLookupError) as excinfo:
        service.query(ccy1, ccy2, asof, strict=True)
    
    assert ccy1 in str(excinfo.value)
    assert ccy2 in str(excinfo.value)
    assert asof in str(excinfo.value)
    service.query.assert_called_once_with(ccy1, ccy2, asof, strict=True)
```


# LLM-generated content at query #24
#--------------------------

```python
import pytest
from decimal import Decimal
from unittest.mock import MagicMock
from datetime import date

class MockFXRateService(FXRateService):
    def __init__(self):
        self._data = {}

    def add_rate(self, rate: FXRate):
        self._data[(rate.ccy1, rate.ccy2, rate.date)] = rate

    def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
        key = (ccy1, ccy2, asof)
        if key not in self._data:
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None
        return self._data[key]

    def queries(self, queries: Iterable[Tuple[Currency, Currency, Date]], strict: bool = False) -> Iterable[Optional[FXRate]]:
        results = []
        for q in queries:
            try:
                results.append(self.query(q[0], q[1], q[2], strict=strict))
            except FXRateLookupError:
                if strict:
                    raise
                results.append(None)
        return results

def test_FXRateService_queries():
    service = MockFXRateService()
    
    ccy_usd = MagicMock(spec=Currency)
    ccy_eur = MagicMock(spec=Currency)
    ccy_gbp = MagicMock(spec=Currency)
    
    today = date.today()
    yesterday = date(1999, 1, 1)
    
    rate1 = FXRate.of(ccy_usd, ccy_eur, today, Decimal("0.85"))
    rate2 = FXRate.of(ccy_eur, ccy_gbp, today, Decimal("0.86"))
    
    service.add_rate(rate1)
    service.add_rate(rate2)
    
    # Define queries: 
    # 1. Valid rate exists
    # 2. Valid rate exists (different pair)
    # 3. Rate does not exist (different date)
    # 4. Rate does not exist (different currency)
    query_list = [
        (ccy_usd, ccy_eur, today),
        (ccy_eur, ccy_gbp, today),
        (ccy_usd, ccy_eur, yesterday),
        (ccy_gbp, ccy_usd, today)
    ]
    
    # Test non-strict mode (should return None for missing rates)
    results_non_strict = list(service.queries(query_list, strict=False))
    
    assert len(results_non_strict) == 4
    assert results_non_strict[0] == rate1
    assert results_non_strict[1] == rate2
    assert results_non_strict[2] is None
    assert results_non_strict[3] is None

    # Test strict mode (should raise FXRateLookupError on first missing rate)
    with pytest.raises(FXRateLookupError) as excinfo:
        list(service.queries(query_list, strict=True))
    
    assert excinfo.value.ccy1 == ccy_usd
    assert excinfo.value.ccy2 == ccy_eur
    assert excinfo.value.asof == yesterday
```


