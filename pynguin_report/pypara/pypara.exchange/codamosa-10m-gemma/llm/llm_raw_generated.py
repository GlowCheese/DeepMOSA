####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from decimal import Decimal
from unittest.mock import MagicMock
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
        return [self.query(q[0], q[1], q[2], strict=strict) for q in queries]

def test_FXRateService_queries():
    # Setup dependencies
    ccy_usd = MagicMock(spec=Currency)
    ccy_eur = MagicMock(spec=Currency)
    ccy_gbp = MagicMock(spec=Currency)
    
    today = date.today()
    val_usd_eur = Decimal("0.85")
    val_usd_gbp = Decimal("0.75")
    
    # Mock data for the service
    rate_1 = FXRate(ccy_usd, ccy_eur, today, val_usd_eur)
    rate_2 = FXRate(ccy_usd, ccy_gbp, today, val_usd_gbp)
    
    lookup_data = {
        (ccy_usd, ccy_eur, today): rate_1,
        (ccy_usd, ccy_gbp, today): rate_2,
    }
    
    service = MockFXRateService(data=lookup_data)
    
    # Define queries
    query_list = [
        (ccy_usd, ccy_eur, today),
        (ccy_usd, ccy_gbp, today),
        (ccy_eur, ccy_usd, today),  # Not in data
        (ccy_usd, ccy_gbp, date(2000, 1, 1))  # Not in data
    ]
    
    # Test 1: Non-strict mode (should return None for missing rates)
    results_non_strict = list(service.queries(query_list, strict=False))
    assert len(results_non_strict) == 4
    assert results_non_strict[0] == rate_1
    assert results_non_strict[1] == rate_2
    assert results_non_strict[2] is None
    assert results_non_strict[3] is None

    # Test 2: Strict mode (should raise FXRateLookupError for missing rates)
    with pytest.raises(FXRateLookupError) as excinfo:
        list(service.queries(query_list, strict=True))
    
    # Verify the error contains correct context (checking the 3rd element of query_list which fails)
    # Since it's an iterable, it will fail on the first mismatching element in the list
    # The first mismatch is (ccy_eur, ccy_usd, today)
    assert excinfo.value.ccy1 == ccy_eur
    assert excinfo.value.ccy2 == ccy_usd
    assert excinfo.value.asof == today

    # Test 3: Empty queries
    assert list(service.queries([], strict=False)) == []
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from decimal import Decimal
from datetime import date
from .currencies import Currencies

def test_FXRate___invert__():
    # Setup base rate: EUR/USD = 2
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    asof = date.today()
    val = Decimal("2")
    
    nrate = FXRate(ccy1, ccy2, asof, val)
    
    # Execute inversion
    rrate = ~nrate
    
    # Assertions
    # The inverted rate should be USD/EUR = 1/2 = 0.5
    assert rrate.ccy1 == ccy2
    assert rrate.ccy2 == ccy1
    assert rrate.date == asof
    assert rrate.value == Decimal("0.5")
    
    # Assert that inverting twice returns the original rate
    assert ~rrate == nrate
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
from datetime import date

class MockFXRateService(FXRateService):
    def __init__(self):
        self.store = {}

    def query(self, ccy1, ccy2, asof, strict=False):
        key = (ccy1, ccy2, asof)
        rate = self.store.get(key)
        if strict and rate is None:
            raise FXRateLookupError(ccy1, ccy2, asof)
        return rate

    def queries(self, queries, strict=False):
        return [self.query(q[0], q[1], q[2], strict) for q in queries]

def test_FXRateService_query():
    service = MockFXRateService()
    
    # Setup mock data
    ccy_eur = MagicMock(spec=Currency)
    ccy_usd = MagicMock(spec=Currency)
    ccy_gbp = MagicMock(spec=Currency)
    today = date.today()
    rate_value = Decimal("1.1")
    
    rate_instance = FXRate(ccy_eur, ccy_usd, today, rate_value)
    service.store[(ccy_eur, ccy_usd, today)] = rate_instance

    # Test Case 1: Successful lookup
    result = service.query(ccy_eur, ccy_usd, today)
    assert result == rate_instance
    assert result.value == rate_value

    # Test Case 2: Non-strict lookup for non-existent rate (returns None)
    result_none = service.query(ccy_usd, ccy_eur, today, strict=False)
    assert result_none is None

    # Test Case 3: Strict lookup for non-existent rate (raises FXRateLookupError)
    with pytest.raises(FXRateLookupError) as excinfo:
        service.query(ccy_gbp, ccy_usd, today, strict=True)
    
    assert excinfo.value.ccy1 == ccy_gbp
    assert excinfo.value.ccy2 == ccy_usd
    assert excinfo.value.asof == today
    assert "Foreign exchange rate for" in str(excinfo.value)

    # Test Case 4: Inverted rate lookup (if we added it to store)
    inverted_rate = ~rate_instance
    service.store[(ccy_usd, ccy_eur, today)] = inverted_rate
    result_inverted = service.query(ccy_usd, ccy_eur, today)
    assert result_inverted == inverted_rate
    assert result_inverted.value == Decimal("1") / rate_value
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from decimal import Decimal
from unittest.mock import MagicMock
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
    # Setup currencies and dates
    ccy_usd = MagicMock(spec=Currency)
    ccy_eur = MagicMock(spec=Currency)
    ccy_gbp = MagicMock(spec=Currency)
    today = date.today()
    yesterday = date(1999, 1, 1)
    
    val_usd_eur = Decimal("0.85")
    val_eur_usd = Decimal("1.18")
    
    # Define the rate map for the mock service
    rate_map = {
        (ccy_usd, ccy_eur, today): FXRate(ccy_usd, ccy_eur, today, val_usd_eur),
        (ccy_eur, ccy_usd, today): FXRate(ccy_eur, ccy_usd, today, val_eur_usd),
    }
    
    service = MockFXRateService(rate_map=rate_map)
    
    # Define queries: 
    # 1. Existing rate
    # 2. Existing rate (inverted)
    # 3. Non-existent rate
    # 4. Rate with different date
    query_list = [
        (ccy_usd, ccy_eur, today),
        (ccy_eur, ccy_usd, today),
        (ccy_usd, ccy_gbp, today),
        (ccy_usd, ccy_eur, yesterday)
    ]
    
    # Test non-strict mode (default)
    results = list(service.queries(query_list, strict=False))
    
    assert len(results) == 4
    assert results[0] == rate_map[(ccy_usd, ccy_eur, today)]
    assert results[1] == rate_map[(ccy_eur, ccy_usd, today)]
    assert results[2] is None
    assert results[3] is None

    # Test strict mode (should raise FXRateLookupError for the missing rate)
    with pytest.raises(FXRateLookupError) as excinfo:
        list(service.queries([(ccy_usd, ccy_gbp, today)], strict=True))
    
    assert excinfo.value.ccy1 == ccy_usd
    assert excinfo.value.ccy2 == ccy_gbp
    assert excinfo.value.asof == today

    # Test strict mode with valid queries only
    valid_queries = [(ccy_usd, ccy_eur, today)]
    results_strict = list(service.queries(valid_queries, strict=True))
    assert len(results_strict) == 1
    assert results_strict[0].value == val_usd_eur
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from decimal import Decimal
from datetime import date
from unittest.mock import MagicMock

def test_FXRate___invert__():
    # Setup dependencies
    mock_ccy_eur = MagicMock(spec=Currency)
    mock_ccy_usd = MagicMock(spec=Hyperlink) # Assuming Currency is a type/class
    # Since we can't rely on the actual Currency class values, we use mocks
    # But the code uses equality checks, so we ensure they are distinct
    ccy1 = MagicMock(spec=Currency)
    ccy2 = MagicMock(spec=Currency)
    ccy1.__eq__.side_effect = lambda x: x == ccy1
    ccy2.__eq__.side_effect = lambda x: x == ccy2
    
    test_date = date(2023, 1, 1)
    original_value = Decimal("2.0")
    inverted_value = Decimal("0.5")
    
    # Create the original rate
    # Note: Using the constructor directly because .of() has strict type validation 
    # that might fail in a test environment without the full Currency/Date setup.
    original_rate = FXRate(
        ccy1=ccy1,
        ccy2=ccy2,
        date=test_date,
        value=original_value
    )
    
    # Execute inversion
    inverted_rate = ~original_rate
    
    # Assertions
    # 1. Check that the currencies are swapped
    assert inverted_rate.ccy1 == ccy2
    assert inverted_rate.ccy2 == ccy1
    
    # 2. Check that the date remains the same
    assert inverted_rate.date == test_date
    
    # 3. Check that the value is the mathematical inverse (1/x)
    assert inverted_rate.value == inverted_value
    
    # 4. Check that the inverted rate, when inverted again, returns to original
    assert ~inverted_rate == original_rate
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest
from decimal import Decimal
from unittest.mock import MagicMock
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
    ccy_usd = MagicMock(spec=Currency)
    ccy_eur = MagicMock(spec=Currency)
    ccy_gbp = MagicMock(spec=Currency)
    today = date.today()
    
    val1 = Decimal("1.1")
    val2 = Decimal("0.8")
    rate1 = FXRate(ccy_usd, ccy_eur, today, val1)
    rate2 = FXRate(ccy_usd, ccy_gbp, today, val2)

    # Define the mapping for the mock service
    rates_data = {
        (ccy_usd, ccy_eur, today): rate1,
        (ccy_usd, ccy_gbp, today): rate2
    }
    
    service = MockFXRateService(rates_map=rates_data)

    # 1. Test successful batch queries
    queries_input = [
        (ccy_usd, ccy_eur, today),
        (ccy_usd, ccy_gbp, today),
        (ccy_eur, ccy_usd, today)  # Not in map
    ]
    
    results = list(service.queries(queries_input, strict=False))
    
    assert len(results) == 3
    assert results[0] == rate1
    assert results[1] == rate2
    assert results[2] is None

    # 2. Test strict mode raising FXRateLookupError
    with pytest.raises(FXRateLookupError) as exc_info:
        list(service.queries(queries_input, strict=True))
    
    assert exc_info.value.ccy1 == ccy_eur
    assert exc_info.value.ccy2 == ccy_usd
    assert exc_info.value.asof == today

    # 3. Test empty input
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
        self.storage = {}

    def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
        key = (ccy1, ccy2, asof)
        rate = self.storage.get(key)
        if strict and rate is None:
            raise FXRateLookupError(ccy1, ccy2, asof)
        return rate

    def queries(self, queries: Iterable[TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
        return (self.query(q[0], q[1], q[2], strict) for q in queries)

def test_FXRateService_query():
    service = MockFXRateService()
    
    # Setup mock data
    ccy_usd = MagicMock(spec=Currency)
    ccy_eur = MagicMock(spec=Currency)
    ccy_gbp = MagicMock(spec=Currency)
    today = date.today()
    rate_val = Decimal("1.1")
    
    rate_obj = FXRate(ccy_usd, ccy_eur, today, rate_val)
    service.storage[(ccy_usd, ccy_eur, today)] = rate_obj
    
    # Test 1: Successful lookup
    result = service.query(ccy_usd, ccy_eur, today)
    assert result == rate_obj
    assert result.value == rate_val
    
    # Test 2: Non-strict lookup for non-existent rate returns None
    result_none = service.query(ccy_usd, ccy_gbp, today, strict=False)
    assert result_none is None
    
    # Test 3: Strict lookup for non-existent rate raises FXRateLookupError
    with pytest.raises(FXRateLookupError) as excinfo:
        service.query(ccy_usd, ccy_gbp, today, strict=True)
    
    assert excinfo.value.ccy1 == ccy_usd
    assert excinfo.value.ccy2 == ccy_gbp
    assert excinfo.value.asof == today
    assert "not found" in str(excinfo.value)

    # Test 4: Verify indexing access (as per FXRate docstring)
    assert rate_obj[0] == ccy_usd
    assert rate_obj[1] == ccy_eur
    assert rate_obj[2] == today
    assert rate_obj[3] == rate_val

    # Test 5: Verify inversion
    inverted = ~rate_obj
    assert inverted[0] == ccy_eur
    assert inverted[1] == ccy_usd
    assert inverted[3] == Decimal("1") / rate_val
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
from decimal import Decimal
from unittest.mock import MagicMock
from datetime import date

@pytest.mark.parametrize("ccy1, ccy2, asof, rate_value, expected_rate", [
    (Currency["USD"], Currency["EUR"], date(2023, 1, 1), Decimal("0.9"), FXRate(Currency["USD"], Currency["EUR"], date(2023, 1, 1), Decimal("0.9"))),
    (Currency["EUR"], Currency["USD"], date(2023, 1, 1), Decimal("1.1"), FXRate(Currency["EUR"], Currency["USD"], date(2023, 1, 1), Decimal("1.1"))),
])
def test_FXRateService_query_success(ccy1, ccy2, asof, rate_value, expected_rate):
    # Setup
    service = MagicMock(spec=FXRateService)
    service.query.return_value = expected_rate
    
    # Execute
    result = service.query(ccy1, ccy2, asof, strict=False)
    
    # Assert
    assert result == expected_rate
    service.query.assert_called_once_with(ccy1, ccy2, asof, strict=False)

def test_FXRateService_query_returns_none():
    # Setup
    service = MagicMock(spec=FXRateService)
    service.query.return_value = None
    ccy1, ccy2, asof = Currency["USD"], Currency["GBP"], date(2023, 1, 1)
    
    # Execute
    result = service.query(ccy1, ccy2, asof, strict=False)
    
    # Assert
    assert result is None
    service.query.assert_called_once_with(ccy1, ccy2, asof, strict=False)

def test_FXRateService_query_raises_lookup_error_when_strict():
    # Setup
    service = MagicMock(spec=FXRateService)
    ccy1, ccy2, asof = Currency["USD"], Currency["JPY"], date(2023, 1, 1)
    service.query.side_effect = FXRateLookupError(ccy1, ccy2, asof)
    
    # Execute & Assert
    with pytest.raises(FXRateLookupError) as excinfo:
        service.query(ccy1, ccy2, asof, strict=True)
    
    assert excinfo.value.ccy1 == ccy1
    assert excinfo.value.ccy2 == ccy2
    assert excinfo.value.asof == asof
    service.query.assert_called_once_with(ccy1, ccy2, asof, strict=True)
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
from datetime import date

class MockFXRateService(FXRateService):
    def __init__(self):
        self.rates = {}

    def add_rate(self, rate: FXRate):
        self.rates[(rate.ccy1, rate.ccy2, rate.date)] = rate

    def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
        key = (ccy1, ccy2, asof)
        rate = self.rates.get(key)
        if strict and rate is None:
            raise FXRateLookupError(ccy1, ccy2, asof)
        return rate

    def queries(self, queries: Iterable[Tuple[Currency, Currency, Date]], strict: bool = False) -> Iterable[Optional[FXRate]]:
        return (self.query(q[0], q[1], q[2], strict) for q in queries)

def test_FXRateService_query():
    service = MockFXRateService()
    
    # Setup mock data
    ccy_eur = MagicMock(spec=Currency)
    ccy_usd = MagicMock(spec=Currency)
    ccy_gbp = MagicMock(spec=Currency)
    today = date.today()
    rate_val = Decimal("1.1")
    
    rate_exists = FXRate(ccy_eur, ccy_usd, today, rate_val)
    service.add_rate(rate_exists)

    # Test Case 1: Successful lookup
    found_rate = service.query(ccy_eur, ccy_usd, today)
    assert found_rate == rate_exists
    assert found_rate.value == rate_val

    # Test Case 2: Non-existent rate (non-strict)
    not_found_rate = service.query(ccy_usd, ccy_eur, today)
    assert not_found_rate is None

    # Test Case 3: Non-existent rate (strict mode raises FXRateLookupError)
    with pytest.raises(FXRateLookupError) as excinfo:
        service.query(ccy_gbp, ccy_usd, today, strict=True)
    
    assert excinfo.value.ccy1 == ccy_gbp
    assert excinfo.value.ccy2 == ccy_usd
    assert excinfo.value.asof == today

    # Test Case 4: Different date
    different_date = date(2000, 1, 1)
    assert service.query(ccy_eur, ccy_usd, different_date) is None
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
from datetime import date

class MockFXRateService(FXRateService):
    def __init__(self):
        self.data = {}

    def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
        key = (ccy1, ccy2, asof)
        rate = self.data.get(key)
        if strict and rate is None:
            raise FXRateLookupError(ccy1, ccy2, asof)
        return rate

    def queries(self, queries: Iterable[TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
        return [self.query(q[0], q[1], q[2], strict) for q in queries]

def test_FXRateService_query():
    service = MockFXRateService()
    
    # Setup mock data
    ccy_eur = MagicMock(spec=Currency)
    ccy_usd = MagicMock(spec=Currency)
    ccy_gbp = MagicMock(spec=Currency)
    today = date.today()
    rate_val = Decimal("1.1")
    
    rate_obj = FXRate(ccy_eur, ccy_usd, today, rate_val)
    service.data[(ccy_eur, ccy_usd, today)] = rate_obj

    # Test Case 1: Successful lookup
    found_rate = service.query(ccy_eur, ccy_usd, today)
    assert found_rate == rate_obj
    assert found_rate.value == rate_val

    # Test Case 2: Non-existent rate (non-strict)
    missing_rate = service.query(ccy_usd, ccy_eur, today)
    assert missing_rate is None

    # Test Case 3: Non-existent rate (strict mode raises error)
    with pytest.raises(FXRateLookupError) as excinfo:
        service.query(ccy_usd, ccy_eur, today, strict=True)
    
    assert excinfo.value.ccy1 == ccy_usd
    assert excinfo.value.ccy2 == ccy_eur
    assert excinfo.value.asof == today

    # Test Case 4: Different date lookup
    yesterday = date.today() # In a real scenario, this would be different
    # Ensure it returns None if date doesn't match even if currencies do
    assert service.query(ccy_eur, ccy_usd, date(1999, 1, 1)) is None
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
from datetime import date

class MockFXRateService(FXRateService):
    def __init__(self):
        self.data = {}

    def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
        key = (ccy1, ccy2, asof)
        rate = self.data.get(key)
        if strict and rate is None:
            raise FXRateLookupError(ccy1, ccy2, asof)
        return rate

    def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
        return (self.query(q[0], q[1], q[2], strict) for q in queries)

def test_FXRateService_query():
    service = MockFXRateService()
    
    # Setup Mock Data
    ccy_usd = MagicMock(spec=Currency)
    ccy_eur = MagicMock(spec=Currency)
    ccy_gbp = MagicMock(spec=Currency)
    today = date.today()
    rate_value = Decimal("1.1")
    
    rate_instance = FXRate(ccy_usd, ccy_eur, today, rate_value)
    service.data[(ccy_usd, ccy_eur, today)] = rate_instance

    # Test Case 1: Successful lookup
    found_rate = service.query(ccy_usd, ccy_eur, today)
    assert found_rate == rate_instance
    assert found_rate.value == rate_value

    # Test Case 2: Non-existent rate returns None (strict=False)
    not_found_rate = service.query(ccy_usd, ccy_gbp, today, strict=False)
    assert not_found_rate is None

    # Test Case 3: Non-existent rate raises FXRateLookupError (strict=True)
    with pytest.raises(FXRateLookupError) as excinfo:
        service.query(ccy_usd, ccy_gbp, today, strict=True)
    
    assert excinfo.value.ccy1 == ccy_usd
    assert excinfo.value.ccy2 == ccy_gbp
    assert excinfo.value.asof == today
    assert "not found" in str(excinfo.value)

    # Test Case 4: Inverted rate lookup via service logic (if implemented)
    # Testing that the service respects the specific parameters provided
    rate_inverted = ~rate_instance
    service.data[(ccy_eur, ccy_usd, today)] = rate_inverted
    found_inverted = service.query(ccy_eur, ccy_usd, today)
    assert found_inverted.value == Decimal("1") / rate_value
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
from datetime import date

class MockFXRateService(FXRateService):
    def __init__(self):
        self.rates = {}

    def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
        key = (ccy1, ccy2, asof)
        rate = self.rates.get(key)
        if strict and rate is None:
            raise FXRateLookupError(ccy1, ccy2, asof)
        return rate

    def queries(self, queries: Iterable[TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
        return [self.query(q[0], q[1], q[2], strict) for q in queries]

def test_FXRateService_query():
    service = MockFXRateService()
    
    # Setup test data
    ccy_usd = MagicMock(spec=Currency)
    ccy_eur = MagicMock(spec=Currency)
    ccy_gbp = MagicMock(spec=Currency)
    test_date = date(2023, 1, 1)
    test_value = Decimal("1.1")
    
    rate_exists = FXRate(ccy_usd, ccy_eur, test_date, test_value)
    
    # Populate service
    service.rates[(ccy_usd, ccy_eur, test_date)] = rate_exists

    # Case 1: Successful lookup
    result = service.query(ccy_usd, ccy_eur, test_date)
    assert result == rate_exists
    assert result.value == Decimal("1.1")

    # Case 2: Non-strict lookup for non-existent rate returns None
    result_none = service.query(ccy_usd, ccy_gbp, test_date, strict=False)
    assert result_none is None

    # Case 3: Strict lookup for non-existent rate raises FXRateLookupError
    with pytest.raises(FXRateLookupError) as excinfo:
        service.query(ccy_usd, ccy_gbp, test_date, strict=True)
    
    assert excinfo.value.ccy1 == ccy_usd
    assert excinfo.value.ccy2 == ccy_gbp
    assert excinfo.value.asof == test_date
    assert "Foreign exchange rate for" in str(excinfo.value)

    # Case 4: Verification of property access via index (as per FXRate implementation)
    assert result[0] == ccy_usd
    assert result[1] == ccy_eur
    assert result[2] == test_date
    assert result[3] == test_value
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
from datetime import date

class MockFXRateService(FXRateService):
    def __init__(self):
        self.storage = {}

    def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
        key = (ccy1, ccy2, asof)
        rate = self.storage.get(key)
        if strict and rate is None:
            raise FXRateLookupError(ccy1, ccy2, asof)
        return rate

    def queries(self, queries: Iterable[TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
        return [self.query(q[0], q[1], q[2], strict) for q in queries]

def test_FXRateService_query():
    service = MockFXRateService()
    
    # Setup mock data
    ccy_usd = MagicMock(spec=Currency)
    ccy_eur = MagicMock(spec=Currency)
    ccy_gbp = MagicMock(spec=Currency)
    today = date.today()
    rate_val = Decimal("1.1")
    
    rate_instance = FXRate(ccy_usd, ccy_eur, today, rate_val)
    service.storage[(ccy_usd, ccy_eur, today)] = rate_instance

    # Test case 1: Successful lookup
    result = service.query(ccy_usd, ccy_eur, today)
    assert result == rate_instance
    assert result.value == rate_val

    # Test case 2: Non-strict lookup for missing rate returns None
    result_none = service.query(ccy_usd, ccy_gbp, today, strict=False)
    assert result_none is None

    # Test case 3: Strict lookup for missing rate raises FXRateLookupError
    with pytest.raises(FXRateLookupError) as excinfo:
        service.query(ccy_usd, ccy_gbp, today, strict=True)
    
    assert excinfo.value.ccy1 == ccy_usd
    assert excinfo.value.ccy2 == ccy_gbp
    assert excinfo.value.asof == today
    assert "Foreign exchange rate for" in str(excinfo.value)

    # Test case 4: Verification of the return type
    assert isinstance(result, FXRate)
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest
from decimal import Decimal
from unittest.mock import MagicMock
from datetime import date

class MockFXRateService(FXRateService):
    def __init__(self):
        self.data = {}

    def query(self, ccy1, ccy2, asof, strict=False):
        key = (ccy1, ccy2, asof)
        try:
            return self.data[key]
        except KeyError:
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None

    def queries(self, queries, strict=False):
        return (self.query(q[0], q[1], q[2], strict=strict) for q in queries)

def test_FXRateService_queries():
    service = MockFXRateService()
    
    # Setup mock data
    ccy_usd = MagicMock(spec=Currency)
    ccy_eur = MagicMock(spec=Currency)
    ccy_gbp = MagicMock(spec=Currency)
    today = date.today()
    
    rate1 = FXRate(ccy_usd, ccy_eur, today, Decimal("0.85"))
    rate2 = FXRate(ccy_eur, ccy_usd, today, Decimal("1.18"))
    
    service.data[(ccy_usd, ccy_eur, today)] = rate1
    service.data[(ccy_eur, ccy_usd, today)] = rate2

    # Define queries
    query_list = [
        (ccy_usd, ccy_eur, today),  # Exists
        (ccy_eur, ccy_usd, today),  # Exists
        (ccy_usd, ccy_gbp, today),  # Does not exist
    ]

    # Test non-strict mode (should return None for missing)
    results = list(service.queries(query_list, strict=False))
    assert len(results) == 3
    assert results[0] == rate1
    assert results[1] == rate2
    assert results[2] is None

    # Test strict mode (should raise FXRateLookupError for missing)
    with pytest.raises(FXRateLookupError) as excinfo:
        list(service.queries(query_list, strict=True))
    
    assert excinfo.value.ccy1 == ccy_usd
    assert excinfo.value.ccy2 == ccy_gbp
    assert excinfo.value.asof == today
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
        self.data = {}

    def query(self, ccy1, ccy2, asof, strict=False):
        key = (ccy1, ccy2, asof)
        try:
            return self.data[key]
        except KeyError:
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None

    def queries(self, queries, strict=False):
        return (self.query(q[0], q[1], q[2], strict=strict) for q in queries)

def test_FXRateService_queries():
    service = MockFXRateService()
    
    # Setup mock currencies and dates
    ccy_usd = MagicMock(spec=Currency)
    ccy_eur = MagicMock(spec=Currency)
    ccy_gbp = MagicMock(spec=Currency)
    today = date.today()
    yesterday = date.today() - MagicMock(spec=timedelta) # Using simple date logic
    
    # Setup rate data
    rate1 = FXRate(ccy_eur, ccy_usd, today, Decimal("1.1"))
    rate2 = FXRate(ccy_usd, ccy_gbp, today, Decimal("0.8"))
    
    service.data[(ccy_eur, ccy_usd, today)] = rate1
    service.data[(ccy_usd, ccy_gbp, today)] = rate2

    # Define queries
    query_list = [
        (ccy_eur, ccy_usd, today),      # Exists
        (ccy_usd, ccy_gbp, today),      # Exists
        (ccy_gbp, ccy_eur, today),      # Does not exist
    ]

    # Test 1: Non-strict mode (should return None for missing)
    results_non_strict = list(service.queries(query_list, strict=False))
    assert len(results_non_strict) == 3
    assert results_non_strict[0] == rate1
    assert results_non_strict[1] == rate2
    assert results_non_strict[2] is None

    # Test 2: Strict mode (should raise FXRateLookupError for missing)
    with pytest.raises(FXRateLookupError) as excinfo:
        list(service.queries(query_list, strict=True))
    
    assert excinfo.value.ccy1 == ccy_gbp
    assert excinfo.value.ccy2 == ccy_eur
    assert excinfo.value.asof == today

    # Test 3: Empty queries
    assert list(service.queries([], strict=False)) == []
```


# LLM-generated content at query #16
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
from datetime import date

class MockFXRateService(FXRateService):
    def __init__(self):
        self.rates = {}

    def query(self, ccy1, ccy2, asof, strict=False):
        key = (ccy1, ccy2, asof)
        rate = self.rates.get(key)
        if rate is None and strict:
            raise FXRateLookupError(ccy1, ccy2, asof)
        return rate

    def queries(self, queries, strict=False):
        return [self.query(q[0], q[1], q[2], strict) for q in queries]

def test_FXRateService_query():
    service = MockFXRateService()
    
    # Setup test data
    ccy_usd = MagicMock(spec=Currency)
    ccy_eur = MagicMock(spec=Currency)
    ccy_gbp = MagicMock(spec=Currency)
    today = date.today()
    rate_val = Decimal("1.1")
    
    rate_obj = FXRate(ccy_usd, ccy_eur, today, rate_val)
    
    # 1. Test successful lookup
    service.rates[(ccy_usd, ccy_eur, today)] = rate_obj
    result = service.query(ccy_usd, ccy_eur, today)
    assert result == rate_obj
    assert result.value == rate_val

    # 2. Test lookup failure (returns None)
    result_none = service.query(ccy_usd, ccy_gbp, today)
    assert result_none is None

    # 3. Test lookup failure with strict=True (raises FXRateLookupError)
    with pytest.raises(FXRateLookupError) as excinfo:
        service.query(ccy_usd, ccy_gbp, today, strict=True)
    
    assert excinfo.value.ccy1 == ccy_usd
    assert excinfo.value.ccy2 == ccy_gbp
    assert excinfo.value.asof == today

    # 4. Test lookup with different date
    result_different_date = service.query(ccy_usd, ccy_eur, date(2000, 1, 1))
    assert result_different_date is None
```


# LLM-generated content at query #17
#--------------------------

```python
import pytest
from decimal import Decimal
from unittest.mock import MagicMock
from datetime import date

class MockFXRateService(FXRateService):
    def __init__(self):
        self.storage = {}

    def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
        key = (ccy1, ccy2, asof)
        rate = self.storage.get(key)
        if strict and rate is None:
            raise FXRateLookupError(ccy1, ccy2, asof)
        return rate

    def queries(self, queries: Iterable[TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
        return (self.query(q[0], q[1], q[2], strict) for q in queries)

def test_FXRateService_query():
    service = MockFXRateService()
    
    # Setup mock data
    ccy_eur = MagicMock(spec=Currency)
    ccy_usd = MagicMock(spec=Currency)
    ccy_gbp = MagicMock(spec=Currency)
    today = date.today()
    rate_value = Decimal("1.1")
    
    rate_instance = FXRate(ccy_eur, ccy_usd, today, rate_value)
    
    # Populate service storage
    service.storage[(ccy_eur, ccy_usd, today)] = rate_instance

    # Test 1: Successful lookup
    found_rate = service.query(ccy_eur, ccy_usd, today)
    assert found_rate == rate_instance
    assert found_rate.value == rate_value

    # Test 2: Non-strict lookup for missing rate (returns None)
    missing_rate = service.query(ccy_usd, ccy_eur, today, strict=False)
    assert missing_rate is None

    # Test 3: Strict lookup for missing rate (raises FXRateLookupError)
    with pytest.raises(FXRateLookupError) as excinfo:
        service.query(ccy_gbp, ccy_usd, today, strict=True)
    
    assert excinfo.value.ccy1 == ccy_gbp
    assert excinfo.value.ccy2 == ccy_usd
    assert excinfo.value.asof == today
    assert "Foreign exchange rate for" in str(excinfo.value)

    # Test 4: Verify property access via index (as per FXRate implementation note)
    assert rate_instance[0] == ccy_eur
    assert rate_instance[1] == ccy_usd
    assert rate_instance[2] == today
    assert rate_instance[3] == rate_value

    # Test 5: Inversion logic
    inverted = ~rate_instance
    assert inverted.ccy1 == ccy_usd
    assert inverted.ccy2 == ccy_eur
    assert inverted.value == Decimal("1") / rate_value
```


# LLM-generated content at query #18
#--------------------------

```python
import pytest
from decimal import Decimal
from datetime import date
from unittest.mock import MagicMock

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
    # Setup mock currencies and dates
    ccy_usd = MagicMock(spec=Currency)
    ccy_eur = MagicMock(spec=Currency)
    ccy_gbp = MagicMock(spec=Currency)
    today = date.today()
    yesterday = date(1999, 1, 1)
    
    val_usd_eur = Decimal("0.85")
    val_eur_usd = Decimal("1.18")
    
    # Prepare rate data
    rate1 = FXRate(ccy_usd, ccy_eur, today, val_usd_eur)
    rate2 = FXRate(ccy_eur, ccy_usd, today, val_eur_usd)
    
    # Define the lookup map for the mock service
    rates_map = {
        (ccy_usd, ccy_eur, today): rate1,
        (ccy_eur, ccy_usd, today): rate2,
    }
    
    service = MockFXRateService(rates_map=rates_map)
    
    # Define queries: 
    # 1. Exists
    # 2. Exists (different pair)
    # 3. Does not exist
    queries_list = [
        (ccy_usd, ccy_eur, today),
        (ccy_eur, ccy_usd, today),
        (ccy_usd, ccy_gbp, today),
        (ccy_usd, ccy_eur, yesterday)
    ]
    
    # Test non-strict mode (should return None for missing rates)
    results = list(service.queries(queries_list, strict=False))
    
    assert len(results) == 4
    assert results[0] == rate1
    assert results[1] == rate2
    assert results[2] is None
    assert results[3] is None
    
    # Test strict mode (should raise FXRateLookupError for missing rates)
    with pytest.raises(FXRateLookupError) as excinfo:
        list(service.queries(queries_list, strict=True))
    
    # Verify error details
    assert excinfo.value.ccy1 == ccy_usd
    assert excinfo.value.ccy2 == ccy_gbp
    assert excinfo.value.asof == today
```


# LLM-generated content at query #19
#--------------------------

```python
import pytest
from decimal import Decimal
from unittest.mock import MagicMock
from datetime import date

@pytest.mark.parametrize("ccy1, ccy2, asof, rate_value, expected_return", [
    (Currency["EUR"], Currency["USD"], date(2023, 1, 1), Decimal("1.1"), FXRate(Currency["EUR"], Currency["USD"], date(2023, 1, 1), Decimal("1.1"))),
    (Currency["GBP"], Currency["JPY"], date(2023, 5, 20), Decimal("180.5"), FXRate(Currency["MSS"], Currency["JPY"], date(2023, 5, 20), Decimal("180.5"))),
])
def test_FXRateService_query_success(ccy1, ccy2, asof, rate_value, expected_return):
    service = MagicMock(spec=FXRateService)
    service.query.return_value = expected_return
    
    result = service.query(ccy1, ccy2, asof, strict=False)
    
    service.query.assert_called_once_with(ccy1, ccy2, asof, strict=False)
    assert result == expected_return

def test_FXRateService_query_returns_none():
    service = MagicMock(spec=FXRateService)
    service.query.return_value = None
    
    result = service.query(Currency["EUR"], Currency["USD"], date(2023, 1, 1), strict=False)
    
    assert result is None

def test_FXRateService_query_raises_lookup_error_when_strict():
    service = MagicMock(spec=FXRateService)
    ccy1, ccy2, asof = Currency["EUR"], Currency["USD"], date(2023, 1, 1)
    
    service.query.side_effect = FXRateLookupError(ccy1, ccy2, asof)
    
    with pytest.raises(FXRateLookupError) as excinfo:
        service.query(ccy1, ccy2, asof, strict=True)
    
    assert excinfo.value.ccy1 == ccy1
    assert excinfo.value.ccy2 == ccy2
    assert excinfo.value.asof == asof
    assert "not found" in str(excinfo.value)

def test_FXRateService_query_args_passed_correctly():
    service = MagicMock(spec=FXRateService)
    ccy1, ccy2, asof = Currency["USD"], Currency["CAD"], date(2023, 12, 31)
    
    service.query(ccy1, ccy2, asof, strict=True)
    
    service.query.assert_called_with(ccy1, ccy2, asof, strict=True)
```


####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
from datetime import date

class MockFXRateService(FXRateService):
    def __init__(self):
        self.rate_map = {}

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
    service = MockFXRateService()
    
    # Setup test data
    ccy_usd = MagicMock(spec=Currency)
    ccy_eur = MagicMock(spec=Currency)
    ccy_gbp = MagicMock(spec=Currency)
    today = date.today()
    yesterday = date.today() - MagicMock(spec=date) # Mocking date subtraction if needed, or use real date
    
    rate_val = Decimal("1.1")
    rate_obj = FXRate(ccy_usd, ccy_eur, today, rate_val)
    
    # Map existing rates
    service.rate_map[(ccy_usd, ccy_eur, today)] = rate_obj
    
    # Define queries
    query_exists = (ccy_usd, ccy_eur, today)
    query_missing = (ccy_eur, ccy_usd, today)
    query_different_date = (ccy_usd, ccy_eur, date.today()) # same as today
    query_third_pair = (ccy_gbp, ccy_usd, today)
    
    queries_list = [query_exists, query_missing, query_third_pair]
    
    # Test 1: Non-strict mode (returns None for missing)
    results = list(service.queries(queries_list, strict=False))
    assert len(results) == 3
    assert results[0] == rate_obj
    assert results[1] is None
    assert results[2] is None
    
    # Test 2: Strict mode (raises error for missing)
    with pytest.raises(FXRateLookupError) as excinfo:
        list(service.queries(queries_list, strict=True))
    
    # Verify error details
    assert excinfo.value.ccy1 == ccy_usd
    assert excinfo.value.ccy2 == ccy_eur
    assert excinfo.value.asof == today

    # Test 3: Empty queries
    assert list(service.queries([], strict=False)) == []
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
from datetime import date

class MockFXRateService(FXRateService):
    def __init__(self):
        self.storage = {}

    def query(self, ccy1, ccy2, asof, strict=False):
        key = (ccy1, ccy2, asof)
        try:
            return self.storage[key]
        except KeyError:
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None

    def queries(self, queries, strict=False):
        return (self.query(q[0], q[1], q[2], strict=strict) for q in queries)

def test_FXRateService_queries():
    service = MockFXRateService()
    
    # Setup mock currencies and dates
    ccy_usd = MagicMock(spec=Currency)
    ccy_eur = MagicMock(spec=Currency)
    ccy_gbp = MagicMock(spec=Currency)
    today = date.today()
    yesterday = date(2000, 1, 1)
    
    rate1 = FXRate(ccy_usd, ccy_eur, today, Decimal("0.85"))
    rate2 = FXRate(ccy_eur, ccy_usd, today, Decimal("1.18"))
    
    # Populate service
    service.storage[(ccy_usd, ccy_eur, today)] = rate1
    service.storage[(ccy_eur, ccy_usd, today)] = rate2
    service.storage[(ccy_gbp, ccy_usd, yesterday)] = None # Explicitly None
    
    # Define queries
    queries_list = [
        (ccy_usd, ccy_eur, today),    # Exists
        (ccy_eur, ccy_usd, today),    # Exists
        (ccy_gbp, ccy_usd, yesterday), # Exists as None
        (ccy_usd, ccy_gbp, yesterday)  # Missing in storage
    ]
    
    # Test 1: Non-strict mode (should return None for missing)
    results = list(service.queries(queries_list, strict=False))
    assert len(results) == 4
    assert results[0] == rate1
    assert results[1] == rate2
    assert results[2] is None
    assert results[3] is None

    # Test 2: Strict mode (should raise FXRateLookupError for missing)
    queries_strict = [(ccy_usd, ccy_gbp, yesterday)]
    with pytest.raises(FXRateLookupError) as excinfo:
        list(service.queries(queries_strict, strict=True))
    
    assert excinfo.value.ccy1 == ccy_usd
    assert excinfo.value.ccy2 == ccy_gbp
    assert excinfo.value.asof == yesterday

    # Test 3: Empty queries
    assert list(service.queries([], strict=False)) == []
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from decimal import Decimal
from unittest.mock import MagicMock
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
    
    today = date.today()
    val1 = Decimal("1.1")
    val2 = Decimal("0.8")
    
    rate1 = FXRate(ccy_usd, ccy_eur, today, val1)
    rate2 = FXRate(ccy_eur, ccy_gbp, today, val2)
    
    # Mock data for the service
    rates_data = {
        (ccy_usd, ccy_eur, today): rate1,
        (ccy_eur, ccy_gbp, today): rate2
    }
    
    service = MockFXRateService(rates_map=rates_data)
    
    # Test Case 1: Successful batch queries
    query_list = [
        (ccy_usd, ccy_eur, today),
        (ccy_eur, ccy_gbp, today)
    ]
    results = list(service.queries(query_list, strict=False))
    assert len(results) == 2
    assert results[0] == rate1
    assert results[1] == rate2

    # Test Case 2: Query with missing rate (non-strict)
    query_missing = [(ccy_gbp, ccy_usd, today)]
    results_missing = list(service.queries(query_missing, strict=False))
    assert len(results_missing) == 1
    assert results_missing[0] is None

    # Test Case 3: Query with missing rate (strict mode)
    with pytest.raises(FXRateLookupError) as excinfo:
        list(service.queries(query_missing, strict=True))
    assert excinfo.value.ccy1 == ccy_gbp
    assert excinfo.value.ccy2 == ccy_usd
    assert excinfo.value.asof == today

    # Test Case 4: Empty queries
    assert list(service.queries([], strict=False)) == []
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
from datetime import date

class MockFXRateService(FXRateService):
    def __init__(self):
        self.store = {}

    def query(self, ccy1, ccy2, asof, strict=False):
        key = (ccy1, ccy2, asof)
        rate = self.store.get(key)
        if rate is None and strict:
            raise FXRateLookupError(ccy1, ccy2, asof)
        return rate

    def queries(self, queries, strict=False):
        return (self.query(q[0], q[1], q[2], strict=strict) for q in queries)

def test_FXRateService_query():
    service = MockFXRateService()
    
    # Setup mock data
    ccy_usd = MagicMock(spec=Currency)
    ccy_eur = MagicMock(spec=Currency)
    asof_date = date(2023, 1, 1)
    rate_value = Decimal("1.1")
    
    rate_obj = FXRate(ccy_usd, ccy_eur, asof_date, rate_value)
    service.store[(ccy_usd, ccy_eur, asof_date)] = rate_obj

    # Test 1: Successful lookup
    result = service.query(ccy_usd, ccy_eur, asof_date)
    assert result == rate_obj
    assert result.value == Decimal("1.1")

    # Test 2: Non-strict lookup for non-existent rate returns None
    result_none = service.query(ccy_eur, ccy_usd, asof_date, strict=False)
    assert result_none is None

    # Test 3: Strict lookup for non-existent rate raises FXRateLookupError
    with pytest.raises(FXRateLookupError) as excinfo:
        service.query(ccy_eur, ccy_usd, asof_date, strict=True)
    
    assert excinfo.value.ccy1 == ccy_eur
    assert excinfo.value.ccy2 == ccy_usd
    assert excinfo.value.asof == asof_date
    assert "Foreign exchange rate for" in str(excinfo.value)

    # Test 4: Verifying the implementation of the abstract method via the mock
    # (Ensuring the logic behaves as an implementation of the defined interface)
    assert isinstance(service, FXRateService)
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
from datetime import date

class MockFXRateService(FXRateService):
    def __init__(self):
        self.storage = {}

    def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
        key = (ccy1, ccy2, asof)
        rate = self.storage.get(key)
        if strict and rate is None:
            raise FXRateLookupError(ccy1, ccy2, asof)
        return rate

    def queries(self, queries: Iterable[TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
        return [self.query(q[0], q[1], q[2], strict) for q in queries]

def test_FXRateService_query():
    service = MockFXRateService()
    
    # Setup mock data
    ccy_usd = MagicMock(spec=Currency)
    ccy_eur = MagicMock(spec=Currency)
    ccy_gbp = MagicMock(spec=Currency)
    today = date.today()
    rate_val = Decimal("1.1")
    
    rate_obj = FXRate(ccy_usd, ccy_eur, today, rate_val)
    service.storage[(ccy_usd, ccy_eur, today)] = rate_obj

    # Case 1: Successful lookup
    result = service.query(ccy_usd, ccy_eur, today)
    assert result == rate_obj
    assert result.value == rate_val

    # Case 2: Non-strict lookup returns None for missing rate
    result_none = service.query(ccy_usd, ccy_gbp, today, strict=False)
    assert result_none is None

    # Case 3: Strict lookup raises FXRateLookupError for missing rate
    with pytest.raises(FXRateLookupError) as excinfo:
        service.query(ccy_usd, ccy_gbp, today, strict=True)
    
    assert excinfo.value.ccy1 == ccy_usd
    assert excinfo.value.ccy2 == ccy_gbp
    assert excinfo.value.asof == today
    assert "not found" in str(excinfo.value)

    # Case 4: Verify internal consistency of the mock logic via queries
    queries_list = [(ccy_usd, ccy_eur, today), (ccy_eur, ccy_usd, today)]
    results = list(service.queries(queries_list))
    assert len(results) == 2
    assert results[0] == rate_obj
    assert results[1] is None
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
from datetime import date

class MockFXRateService(FXRateService):
    def __init__(self):
        self.data = {}

    def query(self, ccy1, ccy2, asof, strict=False):
        key = (ccy1, ccy2, asof)
        if key in self.data:
            return self.data[key]
        if strict:
            raise FXRateLookupError(ccy1, ccy2, asof)
        return None

    def queries(self, queries, strict=False):
        return (self.query(q[0], q[1], q[2], strict) for q in queries)

@pytest.fixture
def mock_currency_usd():
    return MagicMock(spec=Currency)

@pytest.fixture
def mock_currency_eur():
    return MagicMock(spec=Currency)

@pytest.fixture
def service():
    return MockFXRateService()

def test_FXRateService_query(service, mock_currency_usd, mock_currency_eur):
    test_date = date(2023, 1, 1)
    test_rate_value = Decimal("1.1")
    rate = FXRate(mock_currency_eur, mock_currency_usd, test_date, test_rate_value)
    
    # Setup mock data in service
    service.data[(mock_currency_eur, mock_currency_usd, test_date)] = rate

    # Test case 1: Successful lookup
    result = service.query(mock_currency_usd, mock_currency_eur, test_date)
    # Note: The mock implementation above uses a specific key, 
    # we test the logic of the query method implementation.
    
    # Test case 2: Return None when not found and strict=False
    result_none = service.query(mock_currency_usd, mock_currency_eur, date(2000, 1, 1), strict=False)
    assert result_none is None

    # Test case 3: Raise FXRateLookupError when not found and strict=True
    with pytest.raises(FXRateLookupError) as excinfo:
        service.query(mock_currency_usd, mock_currency_eur, date(2000, 1, 1), strict=True)
    
    assert excinfo.value.ccy1 == mock_currency_usd
    assert excinfo.value.ccy2 == mock_currency_eur
    assert excinfo.value.asof == date(2000, 1, 1)

    # Test case 4: Verify the specific rate we injected
    found_rate = service.query(mock_currency_eur, mock_currency_usd, test_date)
    assert found_rate == rate
    assert found_rate.value == test_rate_value
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
        self.data = {}

    def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
        key = (ccy1, ccy2, asof)
        try:
            rate = self.data[key]
            return rate
        except KeyError:
            if strict:
                raise FXRateLookupError(ccy1, ccy1, asof) # Using ccy1 for both as a placeholder for error logic
            return None

    def queries(self, queries: Iterable[TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
        return (self.query(q[0], q[1], q[2], strict) for q in queries)

def test_FXRateService_query():
    service = MockFXRateService()
    
    # Mocking dependencies
    ccy_usd = MagicMock(spec=Currency)
    ccy_eur = MagicMock(spec=Currency)
    ccy_gbp = MagicMock(spec=Currency)
    today = date.today()
    val_usd_eur = Decimal("0.85")
    val_usd_gbp = Decimal("0.75")
    
    rate1 = FXRate(ccy_usd, ccy_eur, today, val_usd_eur)
    rate2 = FXRate(ccy_usd, ccy_gbp, today, val_usd_gbp)
    
    # Setup internal state for the mock service
    service.data = {
        (ccy_usd, ccy_eur, today): rate1,
        (ccy_usd, ccy_gbp, today): rate2
    }

    # Case 1: Successful lookup
    result = service.query(ccy_usd, ccy_eur, today)
    assert result == rate1
    assert result.value == val_usd_eur

    # Case 2: Return None when rate not found and strict=False
    result_none = service.query(ccy_eur, ccy_usd, today, strict=False)
    assert result_none is None

    # Case 3: Raise FXRateLookupError when rate not found and strict=True
    with pytest.raises(FXRateLookupError) as excinfo:
        service.query(ccy_eur, ccy_usd, today, strict=True)
    
    # Verify exception contains expected info (based on implementation in Mock)
    assert "not found" in str(excinfo.value)

    # Case 4: Verify different date returns None
    tomorrow = date.today()
    import datetime
    tomorrow = datetime.date.today() + datetime.timedelta(days=1)
    assert service.query(ccy_usd, ccy_eur, tomorrow) is None
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
from datetime import date

class MockFXRateService(FXRateService):
    def __init__(self):
        self.store = {}

    def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
        key = (ccy1, ccy2, asof)
        rate = self.store.get(key)
        if strict and rate is None:
            raise FXRateLookupError(ccy1, ccy2, asof)
        return rate

    def queries(self, queries: Iterable[TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
        return (self.query(q[0], q[1], q[2], strict) for q in queries)

def test_FXRateService_query():
    service = MockFXRateService()
    
    # Setup Mock Data
    ccy_usd = MagicMock(spec=Currency)
    ccy_eur = MagicMock(spec=Currency)
    ccy_gbp = MagicMock(spec=Currency)
    test_date = date(2023, 1, 1)
    test_value = Decimal("1.1")
    
    rate_exists = FXRate(ccy_usd, ccy_eur, test_date, test_value)
    
    # Populate service store
    service.store[(ccy_usd, ccy_eur, test_date)] = rate_exists

    # Case 1: Successful lookup
    result = service.query(ccy_usd, ccy_eur, test_date)
    assert result == rate_exists
    assert result.value == Decimal("1.1")

    # Case 2: Non-strict lookup for missing rate returns None
    result_none = service.query(ccy_usd, ccy_gbp, test_date, strict=False)
    assert result_none is None

    # Case 3: Strict lookup for missing rate raises FXRateLookupError
    with pytest.raises(FXRateLookupError) as excinfo:
        service.query(ccy_usd, ccy_gbp, test_date, strict=True)
    
    assert excinfo.value.ccy1 == ccy_usd
    assert excinfo.value.ccy2 == ccy_gbp
    assert excinfo.value.asof == test_date
    assert "not found" in str(excinfo.value)

    # Case 4: Verify indexing/tuple behavior as defined in FXRate
    assert rate_exists[0] == ccy_usd
    assert rate_exists[1] == ccy_eur
    assert rate_exists[2] == test_date
    assert rate_exists[3] == test_value
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from decimal import Decimal
from unittest.mock import MagicMock
from datetime import date

class MockFXRateService(FXRateService):
    def __init__(self):
        self.rates = {}

    def query(self, ccy1, ccy2, asof, strict=False):
        key = (ccy1, ccy2, asof)
        if key in self.rates:
            return self.rates[key]
        if strict:
            raise FXRateLookupError(ccy1, ccy2, asof)
        return None

    def queries(self, queries, strict=False):
        return [self.query(q[0], q[1], q[2], strict=strict) for q in queries]

def test_FXRateService_queries():
    service = MockFXRateService()
    
    # Setup Mock Data
    ccy_usd = MagicMock(spec=Currency)
    ccy_eur = MagicMock(spec=Currency)
    ccy_gbp = MagicMock(spec=Currency)
    today = date.today()
    yesterday = date(1999, 1, 1)
    
    rate_val = Decimal("1.1")
    rate_val_2 = Decimal("0.85")
    
    rate1 = FXRate(ccy_usd, ccy_eur, today, rate_val)
    rate2 = FXRate(ccy_usd, ccy_gbp, yesterday, rate_val_2)
    
    service.rates[(ccy_usd, ccy_eur, today)] = rate1
    service.rates[(ccy_usd, ccy_gbp, yesterday)] = rate2

    # Define Queries
    # 1. Valid rate 1
    # 2. Valid rate 2
    # 3. Non-existent rate
    # 4. Same currency (identity)
    query_list = [
        (ccy_usd, ccy_eur, today),
        (ccy_usd, ccy_gbp, yesterday),
        (ccy_eur, ccy_usd, today),
        (ccy_gbp, ccy_usd, today)
    ]

    # Test Case 1: Non-strict mode (should return None for missing)
    results = list(service.queries(query_list, strict=False))
    
    assert len(results) == 4
    assert results[0] == rate1
    assert results[1] == rate2
    assert results[2] is None
    assert results[3] is None

    # Test Case 2: Strict mode (should raise FXRateLookupError for missing)
    with pytest.raises(FXRateLookupError) as excinfo:
        list(service.queries(query_list, strict=True))
    
    # Verify error details
    assert excinfo.value.ccy1 == ccy_eur
    assert excinfo.value.ccy2 == ccy_usd
    assert excinfo.value.asof == today

    # Test Case 3: Empty queries
    assert list(service.queries([], strict=False)) == []
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
from unittest.mock import MagicMock
from decimal import Decimal
from datetime import date

class MockFXRateService(FXRateService):
    def __init__(self):
        self.rates = {}

    def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
        key = (ccy1, ccy2, asof)
        rate = self.rates.get(key)
        if strict and rate is None:
            raise FXRateLookupError(ccy1, ccy2, asof)
        return rate

    def queries(self, queries: Iterable[TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
        return [self.query(q[0], q[1], q[2], strict) for q in queries]

def test_FXRateService_query():
    service = MockFXRateService()
    
    # Mock dependencies
    ccy_usd = MagicMock(spec=Currency)
    ccy_eur = MagicMock(spec=Currency)
    ccy_gbp = MagicMock(spec=Currency)
    today = date.today()
    val = Decimal("1.1")
    
    rate_obj = FXRate(ccy_usd, ccy_eur, today, val)
    
    # Setup state
    service.rates[(ccy_usd, ccy_eur, today)] = rate_obj
    
    # Test 1: Successful lookup
    result = service.query(ccy_usd, ccy_eur, today)
    assert result == rate_obj
    assert result.value == val

    # Test 2: Non-existent rate (non-strict)
    result_none = service.query(ccy_usd, ccy_gbp, today, strict=False)
    assert result_none is None

    # Test 3: Non-existent rate (strict mode)
    with pytest.raises(FXRateLookupError) as excinfo:
        service.query(ccy_usd, ccy_gbp, today, strict=True)
    
    assert excinfo.value.ccy1 == ccy_usd
    assert excinfo.value.ccy2 == ccy_gbp
    assert excinfo.value.asof == today

    # Test 4: Identity check (same currency/date but not in dict)
    # Even if ccy1 == ccy2, if not explicitly added to our mock dict, it should return None/Error
    result_identity = service.query(ccy_usd, ccy_usd, today, strict=False)
    assert result_identity is None
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
from decimal import Decimal
from unittest.mock import MagicMock
from datetime import date

class MockFXRateService(FXRateService):
    def __init__(self, rates_map=None):
        self.rates_map = rates_map or {}

    def query(self, ccy1, ccy2, asof, strict=False):
        key = (ccy1, ccy2, asof)
        rate = self.rates_map.get(key)
        if rate is None and strict:
            raise FXRateLookupError(ccy1, ccy, asof)
        return rate

def test_FXRateService_query():
    # Setup dependencies
    ccy_usd = MagicMock(spec=Currency)
    ccy_eur = MagicMock(spec=Currency)
    ccy_gbp = MagicMock(spec=Currency)
    
    today = date.today()
    val = Decimal("1.1")
    
    rate_obj = FXRate(ccy_usd, ccy_eur, today, val)
    
    # Prepare lookup data
    rates_data = {
        (ccy_usd, ccy_eur, today): rate_obj
    }
    
    service = MockFXRateService(rates_map=rates_data)

    # Test Case 1: Successful lookup
    result = service.query(ccy_usd, ccy_eur, today, strict=False)
    assert result == rate_obj
    assert result.value == val

    # Test Case 2: Lookup returns None when not found (strict=False)
    result_none = service.query(ccy_eur, ccy_usd, today, strict=False)
    assert result_none is None

    # Test Case 3: Lookup raises FXRateLookupError when not found (strict=True)
    with pytest.raises(FXRateLookupError) as excinfo:
        service.query(ccy_gbp, ccy_usd, today, strict=True)
    
    assert excinfo.value.ccy1 == ccy_gbp
    assert excinfo.value.ccy2 == ccy_usd
    assert excinfo.value.asof == today

    # Test Case 4: Different date returns None
    tomorrow = date.today()
    # Assuming date logic for tomorrow is handled via datetime/date
    import datetime
    tomorrow = datetime.date.today() + datetime.timedelta(days=1)
    
    result_wrong_date = service.query(ccy_usd, ccy_eur, tomorrow, strict=False)
    assert result_wrong_date is None
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
from decimal import Decimal
from datetime import date
from unittest.mock import MagicMock

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
    # Setup dependencies
    ccy_usd = MagicMock(spec=Currency)
    ccy_eur = MagicMock(spec=Currency)
    ccy_gbp = MagicMock(spec=Currency)
    
    today = date.today()
    val1 = Decimal("1.1")
    val2 = Decimal("0.85")
    
    rate1 = FXRate(ccy_usd, ccy_eur, today, val1)
    rate2 = FXRate(ccy_eur, ccy_gbp, today, val2)
    
    # Mock data mapping
    rate_map = {
        (ccy_usd, ccy_eur, today): rate1,
        (ccy_eur, ccy_gbp, today): rate2,
    }
    
    service = MockFXRateService(rate_map=rate_map)
    
    # Test Case 1: Successful batch retrieval
    queries_list = [
        (ccy_usd, ccy_eur, today),
        (ccy_eur, ccy_gbp, today)
    ]
    results = list(service.queries(queries_list, strict=False))
    assert len(results) == 2
    assert results[0] == rate1
    assert results[1] == rate2

    # Test Case 2: Handling missing rates (non-strict)
    ccy_missing = MagicMock(spec=Currency)
    queries_with_missing = [
        (ccy_usd, ccy_eur, today),
        (ccy_usd, ccy_missing, today)
    ]
    results_non_strict = list(service.queries(queries_with_missing, strict=False))
    assert results_non_strict[0] == rate1
    assert results_non_strict[1] is None

    # Test Case 3: Handling missing rates (strict)
    queries_with_missing_strict = [
        (ccy_usd, ccy_missing, today)
    ]
    with pytest.raises(FXRateLookupError) as excinfo:
        list(service.queries(queries_with_missing_strict, strict=True))
    
    assert excinfo.value.ccy1 == ccy_usd
    assert excinfo.value.ccy2 == ccy_missing
    assert excinfo.value.asof == today

    # Test Case 4: Empty input
    assert list(service.queries([], strict=False)) == []
```


# LLM-generated content at query #13
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
    ccy_eur = MagicMock(spec=Currency)
    ccy_usd = MagicMock(spec=Currency)
    ccy_gbp = MagicMock(spec=Currency)
    today = date.today()
    
    rate_val = Decimal("1.1")
    rate_val_2 = Decimal("0.8")
    
    rate1 = FXRate(ccy_eur, ccy_usd, today, rate_val)
    rate2 = FXRate(ccy_usd, ccy_gbp, today, rate_val_2)

    # Data for testing
    lookup_data = {
        (ccy_eur, ccy_usd, today): rate1,
        (ccy_usd, ccy_gbp, today): rate2
    }

    service = MockFXRateService(rate_map=lookup_data)

    # 1. Test successful retrieval of multiple rates
    query_list = [
        (ccy_eur, ccy_usd, today),
        (ccy_usd, ccy_gbp, today)
    ]
    results = list(service.queries(query_list, strict=False))
    assert len(results) == 2
    assert results[0] == rate1
    assert results[1] == rate2

    # 2. Test retrieval with missing rates (non-strict)
    query_with_missing = [
        (ccy_eur, ccy_usd, today),
        (ccy_gbp, ccy_eur, today)  # Not in map
    ]
    results_missing = list(service.queries(query_with_missing, strict=False))
    assert len(results_missing) == 2
    assert results_missing[0] == rate1
    assert results_missing[1] is None

    # 3. Test retrieval with missing rates (strict mode)
    query_strict = [
        (ccy_eur, ccy_usd, today),
        (ccy_gbp, ccy_eur, today)
    ]
    with pytest.raises(FXRateLookupError) as excinfo:
        list(service.queries(query_strict, strict=True))
    
    assert excinfo.value.ccy1 == ccy_gbp
    assert excinfo.value.ccy2 == ccy_eur
    assert excinfo.value.asof == today

    # 4. Test empty query list
    assert list(service.queries([], strict=False)) == []
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest
from decimal import Decimal
from unittest.mock import MagicMock
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
    ccy_usd = MagicMock(spec=Currency)
    ccy_eur = MagicMock(spec=Currency)
    ccy_gbp = MagicMock(spec=Currency)
    today = date.today()
    
    rate_val = Decimal("1.1")
    rate_val_inv = Decimal("0.9")
    
    rate1 = FXRate(ccy_usd, ccy_eur, today, rate_val)
    rate2 = FXRate(ccy_eur, ccy_usd, today, rate_val_inv)
    
    # Define the lookup map for the mock service
    lookup_data = {
        (ccy_usd, ccy_eur, today): rate1,
        (ccy_eur, ccy_usd, today): rate2,
    }
    
    service = MockFXRateService(rate_map=lookup_data)

    # 1. Test successful batch retrieval
    queries_list = [
        (ccy_usd, ccy_eur, today),
        (ccy_eur, ccy_usd, today)
    ]
    results = list(service.queries(queries_list, strict=False))
    
    assert len(results) == 2
    assert results[0] == rate1
    assert results[1] == rate2

    # 2. Test retrieval with missing rates (strict=False)
    missing_query = [(ccy_usd, ccy_gbp, today)]
    results_none = list(service.queries(missing_query, strict=False))
    
    assert len(results_none) == 1
    assert results_none[0] is None

    # 3. Test retrieval with missing rates (strict=True)
    # Should raise FXRateLookupError
    with pytest.raises(FXRateLookupError) as excinfo:
        list(service.queries(missing_query, strict=True))
    
    assert excinfo.value.ccy1 == ccy_usd
    assert excinfo.value.ccy2 == ccy_gbp
    assert excinfo.value.asof == today

    # 4. Test mixed results (some found, some missing)
    mixed_queries = [
        (ccy_usd, ccy_eur, today), # Found
        (ccy_usd, ccy_gbp, today)  # Not found
    ]
    results_mixed = list(service.queries(mixed_queries, strict=False))
    
    assert len(results_mixed) == 2
    assert results_mixed[0] == rate1
    assert results_mixed[1] is None
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
        self.data = {}

    def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
        key = (ccy1, ccy2, asof)
        rate = self.data.get(key)
        if strict and rate is None:
            raise FXRateLookupError(ccy1, ccy2, asof)
        return rate

    def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
        return (self.query(q[0], q[1], q[2], strict) for q in queries)

def test_FXRateService_query():
    service = MockFXRateService()
    
    # Setup test data
    ccy_eur = MagicMock(spec=Currency)
    ccy_usd = MagicMock(spec=Currency)
    ccy_gbp = MagicMock(spec=Currency)
    today = datetime.date.today()
    val_1 = Decimal("1.1")
    val_2 = Decimal("0.8")
    
    rate1 = FXRate(ccy_eur, ccy_usd, today, val_1)
    rate2 = FXRate(ccy_usd, ccy_gbp, today, val_2)
    
    service.data[(ccy_eur, ccy_usd, today)] = rate1
    service.data[(ccy_usd, ccy_gbp, today)] = rate2

    # Test Case 1: Successful lookup
    result = service.query(ccy_eur, ccy_usd, today)
    assert result == rate1
    assert result.value == val_1

    # Test Case 2: Lookup non-existent rate (non-strict)
    result_none = service.query(ccy_gbp, ccy_eur, today, strict=False)
    assert result_none is None

    # Test Case 3: Lookup non-existent rate (strict) -> Expect Exception
    with pytest.raises(FXRateLookupError) as excinfo:
        service.query(ccy_gbp, ccy_eur, today, strict=True)
    
    assert excinfo.value.ccy1 == ccy_gbp
    assert excinfo.value.ccy2 == ccy_eur
    assert excinfo.value.asof == today

    # Test Case 4: Verify argument passing to query
    # Using a spy/mock approach for the specific method call
    service.query = MagicMock(side_effect=service.query)
    service.query(ccy_eur, ccy_usd, today)
    service.query.assert_called_with(ccy_eur, ccy_usd, today)
```


# LLM-generated content at query #16
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
            raise FXRateLookupError(ccy1, ccyrypt2, asof)
        return rate

    def queries(self, queries: Iterable[TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
        return [self.query(q[0], q[1], q[2], strict) for q in queries]

def test_FXRateService_query():
    service = MockFXRateService()
    
    # Setup mock data
    ccy_eur = MagicMock(spec=Currency)
    ccy_usd = MagicMock(spec=Currency)
    ccy_gbp = MagicMock(spec=Currency)
    today = datetime.date.today()
    rate_value = Decimal("1.1")
    
    rate_obj = FXRate(ccy_eur, ccy_usd, today, rate_value)
    service.data[(ccy_eur, ccy_usd, today)] = rate_obj

    # Test Case 1: Successful lookup
    result = service.query(ccy_eur, ccy_usd, today)
    assert result == rate_obj
    assert result.value == rate_value

    # Test Case 2: Non-existent rate (non-strict)
    result_none = service.query(ccy_usd, ccy_eur, today)
    assert result_none is None

    # Test Case 3: Non-existent rate (strict)
    with pytest.raises(FXRateLookupError) as excinfo:
        service.query(ccy_usd, ccy_eur, today, strict=True)
    
    assert excinfo.value.ccy1 == ccy_usd
    assert excinfo.value.ccy2 == ccy_eur
    assert excinfo.value.asof == today

    # Test Case 4: Different date
    different_date = datetime.date(2000, 1, 1)
    assert service.query(ccy_eur, ccy_usd, different_date) is None

    # Test Case 5: Different currency pair
    assert service.query(ccy_eur, ccy_gbp, today) is None
```


# LLM-generated content at query #17
#--------------------------

```python
import pytest
from decimal import Decimal
from datetime import date
from unittest.mock import MagicMock

class MockFXRateService(FXRateService):
    def __init__(self):
        self.store = {}

    def query(self, ccy1, ccy2, asof, strict=False):
        key = (ccy1, ccy2, asof)
        rate = self.store.get(key)
        if strict and rate is None:
            raise FXRateLookupError(ccy1, ccy2, asof)
        return rate

    def queries(self, queries, strict=False):
        return [self.query(q[0], q[1], q[2], strict=strict) for q in queries]

def test_FXRateService_queries():
    service = MockFXRateService()
    
    # Setup currencies and dates
    ccy_eur = MagicMock(spec=Currency)
    ccy_usd = MagicMock(spec=Currency)
    ccy_gbp = MagicMock(spec=Currency)
    today = date.today()
    yesterday = date(1999, 1, 1)
    
    # Setup rates
    rate1 = FXRate(ccy_eur, ccy_usd, today, Decimal("1.1"))
    rate2 = FXRate(ccy_usd, ccy_gbp, yesterday, Decimal("0.8"))
    
    # Populate store
    service.store[(ccy_eur, ccy_usd, today)] = rate1
    service.store[(ccy_usd, ccy_gbp, yesterday)] = rate2
    
    # Define queries
    query_list = [
        (ccy_eur, ccy_usd, today),
        (ccy_usd, ccy_gbp, yesterday),
        (ccy_gbp, ccy_eur, today),  # Non-existent
    ]
    
    # Test 1: Non-strict mode (should return None for missing rates)
    results_non_strict = list(service.queries(query_list, strict=False))
    assert len(results_non_strict) == 3
    assert results_non_strict[0] == rate1
    assert results_non_strict[1] == rate2
    assert results_non_strict[2] is None
    
    # Test 2: Strict mode (should raise FXRateLookupError for missing rates)
    with pytest.raises(FXRateLookupError) as excinfo:
        list(service.queries(query_list, strict=True))
    
    assert excinfo.value.ccy1 == ccy_gbp
    assert excinfo.value.ccy2 == ccy_eur
    assert excinfo.value.asof == today
```


# LLM-generated content at query #18
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
    ccy_usd = MagicMock(spec=Currency)
    ccy_eur = MagicMock(spec=Currency)
    ccy_gbp = MagicMock(spec=Currency)
    
    as_of_today = date.today()
    as_of_yesterday = date(2023, 1, 1)
    
    val_usd_eur = Decimal("0.92")
    val_eur_usd = Decimal("1.08")
    
    rate1 = FXRate(ccy_usd, ccy_eur, as_of_today, val_usd_eur)
    rate2 = FXRate(ccy_eur, ccy_usd, as_of_today, val_eur_usd)
    
    # Define the rate map for the mock service
    rate_map = {
        (ccy_usd, ccy_eur, as_of_today): rate1,
        (ccy_eur, ccy_usd, as_of_today): rate2,
    }
    
    service = MockFXRateService(rate_map=rate_map)
    
    # Define queries: 
    # 1. Exists
    # 2. Does not exist
    # 3. Exists (different date)
    queries_list = [
        (ccy_usd, ccy_eur, as_of_today),
        (ccy_usd, ccy_gbp, as_of_today),
        (ccy_usd, ccy_eur, as_of_yesterday)
    ]
    
    # Test case 1: Non-strict mode (should return None for missing rates)
    results = list(service.queries(queries_list, strict=False))
    
    assert len(results) == 3
    assert results[0] == rate1
    assert results[1] is None
    assert results[2] is None
    
    # Test case 2: Strict mode (should raise FXRateLookupError for missing rates)
    with pytest.raises(FXRateLookupError) as excinfo:
        list(service.queries(queries_list, strict=True))
    
    assert excinfo.value.ccy1 == ccy_usd
    assert excinfo.value.ccy2 == ccy_gbp
    assert excinfo.value.asof == as_of_today

    # Test case 3: Single query item
    single_query = [(ccy_eur, ccy_usd, as_of_today)]
    results_single = list(service.queries(single_query))
    assert results_single == [rate2]
```


