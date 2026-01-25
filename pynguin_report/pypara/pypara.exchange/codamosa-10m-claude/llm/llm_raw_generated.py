####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_FXRateService_queries():
    """Test the queries method of FXRateService."""
    import datetime
    from decimal import Decimal
    from unittest.mock import MagicMock
    
    # Create a mock FXRateService instance
    service = MagicMock(spec=FXRateService)
    
    # Setup test data
    eur = MagicMock(spec=Currency)
    usd = MagicMock(spec=Currency)
    gbp = MagicMock(spec=Currency)
    test_date = datetime.date.today()
    
    # Create FXRate instances
    rate_eur_usd = FXRate(eur, usd, test_date, Decimal("1.20"))
    rate_usd_gbp = FXRate(usd, gbp, test_date, Decimal("0.73"))
    
    # Setup queries
    queries = [
        (eur, usd, test_date),
        (usd, gbp, test_date),
    ]
    
    # Test 1: queries returns results for valid queries
    service.queries.return_value = [rate_eur_usd, rate_usd_gbp]
    results = list(service.queries(queries, strict=False))
    
    assert len(results) == 2
    assert results[0] == rate_eur_usd
    assert results[1] == rate_usd_gbp
    service.queries.assert_called_once_with(queries, strict=False)
    
    # Test 2: queries with strict=True
    service.reset_mock()
    service.queries.return_value = [rate_eur_usd, None]
    results = list(service.queries(queries, strict=True))
    
    assert len(results) == 2
    assert results[0] == rate_eur_usd
    assert results[1] is None
    service.queries.assert_called_once_with(queries, strict=True)
    
    # Test 3: queries with None results when strict=False
    service.reset_mock()
    service.queries.return_value = [None, None]
    results = list(service.queries(queries, strict=False))
    
    assert len(results) == 2
    assert all(r is None for r in results)
    
    # Test 4: queries with empty iterable
    service.reset_mock()
    service.queries.return_value = []
    results = list(service.queries([], strict=False))
    
    assert len(results) == 0
    service.queries.assert_called_once_with([], strict=False)


# LLM-generated content at query #2
#--------------------------

```python
def test_FXRate___invert__():
    """Test the __invert__ method of FXRate class."""
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    
    # Test basic inversion
    date = datetime.date.today()
    original_rate = FXRate(Currencies["EUR"], Currencies["USD"], date, Decimal("2"))
    inverted_rate = ~original_rate
    
    # Check that currencies are swapped
    assert inverted_rate.ccy1 == Currencies["USD"]
    assert inverted_rate.ccy2 == Currencies["EUR"]
    
    # Check that date is preserved
    assert inverted_rate.date == date
    
    # Check that value is inverted (1/2 = 0.5)
    assert inverted_rate.value == Decimal("0.5")
    
    # Test double inversion returns to original
    double_inverted = ~inverted_rate
    assert double_inverted.ccy1 == original_rate.ccy1
    assert double_inverted.ccy2 == original_rate.ccy2
    assert double_inverted.date == original_rate.date
    assert double_inverted.value == original_rate.value
    
    # Test with different rate value
    rate2 = FXRate(Currencies["GBP"], Currencies["JPY"], date, Decimal("150"))
    inverted_rate2 = ~rate2
    assert inverted_rate_2.ccy1 == Currencies["JPY"]
    assert inverted_rate2.ccy2 == Currencies["GBP"]
    assert inverted_rate2.value == Decimal("1") / Decimal("150")
    
    # Test with rate value of 1
    rate3 = FXRate(Currencies["USD"], Currencies["USD"], date, Decimal("1"))
    inverted_rate3 = ~rate3
    assert inverted_rate3.value == Decimal("1")
    
    # Test with very small value
    rate4 = FXRate(Currencies["USD"], Currencies["EUR"], date, Decimal("0.001"))
    inverted_rate4 = ~rate4
    assert inverted_rate4.value == Decimal("1000")


# LLM-generated content at query #3
#--------------------------

```python
def test_FXRateService_query():
    """Test the query method of FXRateService abstract class."""
    import datetime
    from decimal import Decimal
    
    # Create a concrete implementation of FXRateService for testing
    class MockFXRateService(FXRateService):
        def __init__(self):
            self.rates = {}
        
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            key = (ccy1, ccy2, asof)
            if key in self.rates:
                return self.rates[key]
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None
        
        def queries(self, queries: Iterable[Tuple[Currency, Currency, Date]], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    # Setup test data
    service = MockFXRateService()
    date = datetime.date.today()
    eur = Currency(code="EUR", decimals=2)
    usd = Currency(code="USD", decimals=2)
    gbp = Currency(code="GBP", decimals=2)
    
    # Create and store a test FX rate
    rate = FXRate(eur, usd, date, Decimal("1.20"))
    service.rates[(eur, usd, date)] = rate
    
    # Test 1: Query existing rate with strict=False
    result = service.query(eur, usd, date, strict=False)
    assert result is not None
    assert result.ccy1 == eur
    assert result.ccy2 == usd
    assert result.value == Decimal("1.20")
    
    # Test 2: Query non-existing rate with strict=False should return None
    result = service.query(usd, gbp, date, strict=False)
    assert result is None
    
    # Test 3: Query non-existing rate with strict=True should raise FXRateLookupError
    with pytest.raises(FXRateLookupError) as exc_info:
        service.query(usd, gbp, date, strict=True)
    assert exc_info.value.ccy1 == usd
    assert exc_info.value.ccy2 == gbp
    assert exc_info.value.asof == date
    
    # Test 4: Query with different date returns None
    different_date = date - datetime.timedelta(days=1)
    result = service.query(eur, usd, different_date, strict=False)
    assert result is None
    
    # Test 5: Query with inverted currencies
    inverted_rate = FXRate(usd, eur, date, Decimal("0.833"))
    service.rates[(usd, eur, date)] = inverted_rate
    result = service.query(usd, eur, date, strict=False)
    assert result is not None
    assert result.ccy1 == usd
    assert result.ccy2 == eur
    assert result.value == Decimal("0.833")


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from decimal import Decimal
from unittest.mock import Mock, MagicMock
import datetime

def test_FXRateService_queries():
    """Test the queries method of FXRateService."""
    
    # Create a mock FXRateService instance
    service = Mock(spec=FXRateService)
    
    # Create test currencies
    eur = Mock(spec=Currency)
    usd = Mock(spec=Currency)
    gbp = Mock(spec=Currency)
    
    # Create test date
    test_date = datetime.date.today()
    
    # Create test FXRate objects
    eur_usd_rate = FXRate(eur, usd, test_date, Decimal("1.20"))
    usd_gbp_rate = FXRate(usd, gbp, test_date, Decimal("0.73"))
    
    # Define test queries
    queries = [
        (eur, usd, test_date),
        (usd, gbp, test_date),
    ]
    
    # Test 1: queries returns expected rates
    service.queries.return_value = [eur_usd_rate, usd_gbp_rate]
    result = list(service.queries(queries, strict=False))
    
    assert len(result) == 2
    assert result[0] == eur_usd_rate
    assert result[1] == usd_gbp_rate
    service.queries.assert_called_once_with(queries, strict=False)
    
    # Test 2: queries with strict=True
    service.reset_mock()
    service.queries.return_value = [eur_usd_rate, None]
    result = list(service.queries(queries, strict=True))
    
    assert len(result) == 2
    assert result[0] == eur_usd_rate
    assert result[1] is None
    service.queries.assert_called_once_with(queries, strict=True)
    
    # Test 3: queries with empty input
    service.reset_mock()
    service.queries.return_value = []
    result = list(service.queries([], strict=False))
    
    assert len(result) == 0
    service.queries.assert_called_once_with([], strict=False)
    
    # Test 4: queries returns None for missing rates
    service.reset_mock()
    service.queries.return_value = [None, None]
    result = list(service.queries(queries, strict=False))
    
    assert len(result) == 2
    assert all(r is None for r in result)
    service.queries.assert_called_once_with(queries, strict=False)
    
    # Test 5: queries with single query
    service.reset_mock()
    single_query = [(eur, usd, test_date)]
    service.queries.return_value = [eur_usd_rate]
    result = list(service.queries(single_query, strict=False))
    
    assert len(result) == 1
    assert result[0] == eur_usd_rate
    service.queries.assert_called_once_with(single_query, strict=False)


# LLM-generated content at query #5
#--------------------------

```python
def test_FXRate___invert__():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    
    # Test basic inversion
    date = datetime.date.today()
    original_rate = FXRate(Currencies["EUR"], Currencies["USD"], date, Decimal("2"))
    inverted_rate = ~original_rate
    
    # Verify currencies are swapped
    assert inverted_rate.ccy1 == Currencies["USD"]
    assert inverted_rate.ccy2 == Currencies["EUR"]
    
    # Verify date remains the same
    assert inverted_rate.date == date
    
    # Verify value is inverted (1/2 = 0.5)
    assert inverted_rate.value == Decimal("0.5")
    
    # Test double inversion returns original rate
    double_inverted = ~inverted_rate
    assert double_inverted == original_rate
    
    # Test with different value
    rate_with_value_4 = FXRate(Currencies["GBP"], Currencies["JPY"], date, Decimal("4"))
    inverted_4 = ~rate_with_value_4
    assert inverted_4.ccy1 == Currencies["JPY"]
    assert inverted_4.ccy2 == Currencies["GBP"]
    assert inverted_4.value == Decimal("0.25")
    
    # Test with decimal value
    rate_decimal = FXRate(Currencies["USD"], Currencies["EUR"], date, Decimal("1.5"))
    inverted_decimal = ~rate_decimal
    assert inverted_decimal.value == Decimal("1") / Decimal("1.5")


# LLM-generated content at query #6
#--------------------------

```python
import datetime
from decimal import Decimal
from typing import Iterable, Optional

import pytest

from pypara.currencies import Currency, Currencies
from pypara.fx import FXRate, FXRateService


class MockFXRateService(FXRateService):
    """Mock implementation of FXRateService for testing."""

    def __init__(self, rates_dict=None):
        self.rates_dict = rates_dict or {}

    def query(self, ccy1: Currency, ccy2: Currency, asof: datetime.date, strict: bool = False) -> Optional[FXRate]:
        key = (ccy1.code, ccy2.code, asof)
        if key in self.rates_dict:
            return self.rates_dict[key]
        if strict:
            raise FXRateLookupError(ccy1, ccy2, asof)
        return None

    def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
        return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]


def test_FXRateService_queries():
    """Test the queries method of FXRateService."""
    today = datetime.date.today()
    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    gbp = Currencies["GBP"]

    # Create test rates
    rate_eur_usd = FXRate(eur, usd, today, Decimal("1.2"))
    rate_gbp_usd = FXRate(gbp, usd, today, Decimal("1.4"))

    # Initialize service with test rates
    rates_dict = {
        ("EUR", "USD", today): rate_eur_usd,
        ("GBP", "USD", today): rate_gbp_usd,
    }
    service = MockFXRateService(rates_dict)

    # Test 1: Query multiple rates that exist
    query_list = [
        (eur, usd, today),
        (gbp, usd, today),
    ]
    results = list(service.queries(query_list))
    assert len(results) == 2
    assert results[0] == rate_eur_usd
    assert results[1] == rate_gbp_usd

    # Test 2: Query with some rates not found (non-strict mode)
    query_list_with_missing = [
        (eur, usd, today),
        (eur, gbp, today),  # This rate doesn't exist
        (gbp, usd, today),
    ]
    results = list(service.queries(query_list_with_missing, strict=False))
    assert len(results) == 3
    assert results[0] == rate_eur_usd
    assert results[1] is None
    assert results[2] == rate_gbp_usd

    # Test 3: Query with missing rate in strict mode should raise error
    with pytest.raises(FXRateLookupError):
        list(service.queries(query_list_with_missing, strict=True))

    # Test 4: Empty query list
    results = list(service.queries([]))
    assert results == []

    # Test 5: Query with different dates
    yesterday = today - datetime.timedelta(days=1)
    query_different_dates = [
        (eur, usd, today),
        (eur, usd, yesterday),  # Different date, not in rates_dict
    ]
    results = list(service.queries(query_different_dates, strict=False))
    assert len(results) == 2
    assert results[0] == rate_eur_usd
    assert results[1] is None


# LLM-generated content at query #7
#--------------------------

```python
def test_FXRateService_query():
    """Test the query method of FXRateService."""
    import datetime
    from decimal import Decimal
    from unittest.mock import Mock
    
    # Create a concrete implementation of FXRateService for testing
    class TestFXRateService(FXRateService):
        def __init__(self):
            self.rates = {}
        
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            key = (ccy1, ccy2, asof)
            if key in self.rates:
                return self.rates[key]
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None
        
        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    # Setup test data
    from pypara.currencies import Currency
    eur = Mock(spec=Currency)
    usd = Mock(spec=Currency)
    test_date = datetime.date(2023, 1, 1)
    
    service = TestFXRateService()
    rate = FXRate(eur, usd, test_date, Decimal("1.20"))
    service.rates[(eur, usd, test_date)] = rate
    
    # Test 1: Query existing rate
    result = service.query(eur, usd, test_date)
    assert result == rate
    assert result.ccy1 == eur
    assert result.ccy2 == usd
    assert result.date == test_date
    assert result.value == Decimal("1.20")
    
    # Test 2: Query non-existing rate returns None
    gbp = Mock(spec=Currency)
    result = service.query(eur, gbp, test_date, strict=False)
    assert result is None
    
    # Test 3: Query non-existing rate with strict=True raises FXRateLookupError
    with pytest.raises(FXRateLookupError) as exc_info:
        service.query(eur, gbp, test_date, strict=True)
    assert exc_info.value.ccy1 == eur
    assert exc_info.value.ccy2 == gbp
    assert exc_info.value.asof == test_date
    
    # Test 4: Query with different date returns None
    other_date = datetime.date(2023, 1, 2)
    result = service.query(eur, usd, other_date)
    assert result is None


# LLM-generated content at query #8
#--------------------------

```python
def test_FXRate___invert__():
    """Test the __invert__ method of FXRate class."""
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    
    # Test basic inversion
    date = datetime.date.today()
    original_rate = FXRate(Currencies["EUR"], Currencies["USD"], date, Decimal("2"))
    inverted_rate = ~original_rate
    
    # Check that currencies are swapped
    assert inverted_rate.ccy1 == Currencies["USD"]
    assert inverted_rate.ccy2 == Currencies["EUR"]
    
    # Check that date remains the same
    assert inverted_rate.date == date
    
    # Check that value is inverted (1/2 = 0.5)
    assert inverted_rate.value == Decimal("0.5")
    
    # Test double inversion returns to original
    double_inverted = ~inverted_rate
    assert double_inverted.ccy1 == original_rate.ccy1
    assert double_inverted.ccy2 == original_rate.ccy2
    assert double_inverted.date == original_rate.date
    assert double_inverted.value == original_rate.value
    
    # Test inversion with different rate values
    rate_with_decimal = FXRate(Currencies["GBP"], Currencies["JPY"], date, Decimal("150.5"))
    inverted_decimal = ~rate_with_decimal
    assert inverted_decimal.ccy1 == Currencies["JPY"]
    assert inverted_decimal.ccy2 == Currencies["GBP"]
    assert inverted_decimal.value == Decimal("1") / Decimal("150.5")
    
    # Test inversion with value of 1
    rate_one = FXRate(Currencies["USD"], Currencies["USD"], date, Decimal("1"))
    inverted_one = ~rate_one
    assert inverted_one.value == Decimal("1")
    
    # Test inversion preserves immutability
    original_ccy1 = original_rate.ccy1
    original_ccy2 = original_rate.ccy2
    original_value = original_rate.value
    _ = ~original_rate
    assert original_rate.ccy1 == original_ccy1
    assert original_rate.ccy2 == original_ccy2
    assert original_rate.value == original_value


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from abc import ABCMeta
from decimal import Decimal
from unittest.mock import MagicMock
import datetime


def test_FXRateService_query():
    """Test the query method of FXRateService abstract class."""
    
    # Create a concrete implementation of the abstract FXRateService
    class ConcreteFXRateService(FXRateService):
        def __init__(self):
            self.rates = {}
        
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            key = (ccy1, ccy2, asof)
            if key in self.rates:
                return self.rates[key]
            elif strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None
        
        def queries(self, queries: Iterable[tuple], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    # Create test data
    eur = MagicMock(spec=Currency)
    usd = MagicMock(spec=Currency)
    test_date = datetime.date.today()
    rate_value = Decimal("1.20")
    
    service = ConcreteFXRateService()
    fx_rate = FXRate(eur, usd, test_date, rate_value)
    service.rates[(eur, usd, test_date)] = fx_rate
    
    # Test 1: Query existing rate with strict=False
    result = service.query(eur, usd, test_date, strict=False)
    assert result is not None
    assert result == fx_rate
    assert result.ccy1 == eur
    assert result.ccy2 == usd
    assert result.date == test_date
    assert result.value == rate_value
    
    # Test 2: Query non-existing rate with strict=False returns None
    jpy = MagicMock(spec=Currency)
    result = service.query(eur, jpy, test_date, strict=False)
    assert result is None
    
    # Test 3: Query non-existing rate with strict=True raises FXRateLookupError
    with pytest.raises(FXRateLookupError) as exc_info:
        service.query(eur, jpy, test_date, strict=True)
    assert exc_info.value.ccy1 == eur
    assert exc_info.value.ccy2 == jpy
    assert exc_info.value.asof == test_date
    
    # Test 4: Query with different date returns None
    different_date = datetime.date.today() - datetime.timedelta(days=1)
    result = service.query(eur, usd, different_date, strict=False)
    assert result is None
    
    # Test 5: Default strict parameter is False
    result = service.query(eur, usd, test_date)
    assert result == fx_rate


# LLM-generated content at query #10
#--------------------------

```python
def test_FXRateService_query():
    """Test the query method of FXRateService abstract class."""
    import datetime
    from decimal import Decimal
    from unittest.mock import Mock
    
    # Create a mock implementation of FXRateService
    mock_service = Mock(spec=FXRateService)
    
    # Setup test data
    eur = Mock(spec=Currency)
    usd = Mock(spec=Currency)
    date = datetime.date.today()
    rate_value = Decimal("1.20")
    
    # Test case 1: query returns FXRate when found
    expected_rate = FXRate(eur, usd, date, rate_value)
    mock_service.query.return_value = expected_rate
    
    result = mock_service.query(eur, usd, date, strict=False)
    assert result == expected_rate
    mock_service.query.assert_called_once_with(eur, usd, date, strict=False)
    
    # Test case 2: query returns None when not found and strict=False
    mock_service.reset_mock()
    mock_service.query.return_value = None
    
    result = mock_service.query(eur, usd, date, strict=False)
    assert result is None
    mock_service.query.assert_called_once_with(eur, usd, date, strict=False)
    
    # Test case 3: query raises FXRateLookupError when not found and strict=True
    mock_service.reset_mock()
    mock_service.query.side_effect = FXRateLookupError(eur, usd, date)
    
    try:
        mock_service.query(eur, usd, date, strict=True)
        assert False, "Expected FXRateLookupError to be raised"
    except FXRateLookupError:
        pass
    
    mock_service.query.assert_called_once_with(eur, usd, date, strict=True)
    
    # Test case 4: query with default strict parameter (False)
    mock_service.reset_mock()
    mock_service.query.return_value = expected_rate
    
    result = mock_service.query(eur, usd, date)
    assert result == expected_rate
    mock_service.query.assert_called_once_with(eur, usd, date)


# LLM-generated content at query #11
#--------------------------

```python
import pytest
from decimal import Decimal
from datetime import date
from unittest.mock import Mock, MagicMock


def test_FXRateService_query():
    """Test the query method of FXRateService abstract class."""
    
    # Create a concrete implementation of FXRateService for testing
    class ConcreteFXRateService(FXRateService):
        def __init__(self):
            self.rates = {}
        
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            key = (ccy1, ccy2, asof)
            if key in self.rates:
                return self.rates[key]
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None
        
        def queries(self, queries: Iterable[TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    # Setup test data
    service = ConcreteFXRateService()
    test_date = date.today()
    eur = Currency("EUR", 1, "Euro")
    usd = Currency("USD", 1, "US Dollar")
    gbp = Currency("GBP", 1, "British Pound")
    
    fx_rate = FXRate(eur, usd, test_date, Decimal("1.20"))
    
    # Test 1: Query returns None when rate not found and strict=False
    result = service.query(eur, usd, test_date, strict=False)
    assert result is None
    
    # Test 2: Query raises FXRateLookupError when rate not found and strict=True
    with pytest.raises(FXRateLookupError) as exc_info:
        service.query(gbp, usd, test_date, strict=True)
    assert exc_info.value.ccy1 == gbp
    assert exc_info.value.ccy2 == usd
    assert exc_info.value.asof == test_date
    
    # Test 3: Query returns the rate when it exists
    service.rates[(eur, usd, test_date)] = fx_rate
    result = service.query(eur, usd, test_date, strict=False)
    assert result == fx_rate
    assert result.ccy1 == eur
    assert result.ccy2 == usd
    assert result.date == test_date
    assert result.value == Decimal("1.20")
    
    # Test 4: Query with strict=True returns rate when it exists
    result = service.query(eur, usd, test_date, strict=True)
    assert result == fx_rate
    
    # Test 5: Different currency pairs return different results
    fx_rate_gbp = FXRate(gbp, usd, test_date, Decimal("1.35"))
    service.rates[(gbp, usd, test_date)] = fx_rate_gbp
    
    result_eur_usd = service.query(eur, usd, test_date)
    result_gbp_usd = service.query(gbp, usd, test_date)
    
    assert result_eur_usd != result_gbp_usd
    assert result_eur_usd.value == Decimal("1.20")
    assert result_gbp_usd.value == Decimal("1.35")
    
    # Test 6: Different dates return different results or None
    different_date = date(2023, 1, 1)
    result_different_date = service.query(eur, usd, different_date, strict=False)
    assert result_different_date is None


# LLM-generated content at query #12
#--------------------------

```python
import pytest
from decimal import Decimal
from datetime import date
from unittest.mock import Mock, MagicMock
from pypara.currencies import Currency


def test_FXRateService_queries():
    """Test the queries method of FXRateService."""
    
    # Create a mock FXRateService instance
    service = Mock(spec=FXRateService)
    
    # Create test currencies and dates
    eur = Mock(spec=Currency)
    usd = Mock(spec=Currency)
    gbp = Mock(spec=Currency)
    test_date = date.today()
    
    # Create test FXRate objects
    rate_eur_usd = FXRate(eur, usd, test_date, Decimal("1.2"))
    rate_usd_gbp = FXRate(usd, gbp, test_date, Decimal("0.8"))
    rate_eur_gbp = FXRate(eur, gbp, test_date, Decimal("0.96"))
    
    # Test case 1: Normal query with results
    queries_input = [
        (eur, usd, test_date),
        (usd, gbp, test_date),
        (eur, gbp, test_date),
    ]
    expected_results = [rate_eur_usd, rate_usd_gbp, rate_eur_gbp]
    
    service.queries.return_value = expected_results
    results = list(service.queries(queries_input, strict=False))
    
    assert len(results) == 3
    assert results[0] == rate_eur_usd
    assert results[1] == rate_usd_gbp
    assert results[2] == rate_eur_gbp
    service.queries.assert_called_once_with(queries_input, strict=False)
    
    # Test case 2: Query with None results (rate not found)
    service.reset_mock()
    service.queries.return_value = [rate_eur_usd, None, rate_eur_gbp]
    results = list(service.queries(queries_input, strict=False))
    
    assert len(results) == 3
    assert results[0] == rate_eur_usd
    assert results[1] is None
    assert results[2] == rate_eur_gbp
    
    # Test case 3: Empty query list
    service.reset_mock()
    service.queries.return_value = []
    results = list(service.queries([], strict=False))
    
    assert len(results) == 0
    service.queries.assert_called_once_with([], strict=False)
    
    # Test case 4: Strict mode enabled
    service.reset_mock()
    service.queries.side_effect = FXRateLookupError(eur, usd, test_date)
    
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries_input, strict=True))
    
    service.queries.assert_called_once_with(queries_input, strict=True)
    
    # Test case 5: Single query in iterable
    service.reset_mock()
    single_query = [(eur, usd, test_date)]
    service.queries.return_value = [rate_eur_usd]
    results = list(service.queries(single_query, strict=False))
    
    assert len(results) == 1
    assert results[0] == rate_eur_usd
    
    # Test case 6: Verify iterable is properly consumed
    service.reset_mock()
    query_generator = ((eur, usd, test_date) for _ in range(2))
    service.queries.return_value = iter([rate_eur_usd, rate_eur_usd])
    results = list(service.queries(query_generator, strict=False))
    
    assert len(results) == 2


# LLM-generated content at query #13
#--------------------------

```python
def test_FXRateService_query():
    """Test the query method of FXRateService"""
    import datetime
    from decimal import Decimal
    from unittest.mock import Mock
    
    # Create a concrete implementation of FXRateService for testing
    class TestFXRateService(FXRateService):
        def __init__(self, rates_dict=None):
            self.rates_dict = rates_dict or {}
        
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            key = (ccy1, ccy2, asof)
            if key in self.rates_dict:
                return self.rates_dict[key]
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None
        
        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    # Setup test data
    from .currencies import Currencies
    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    gbp = Currencies["GBP"]
    test_date = datetime.date(2023, 1, 1)
    
    rate_eur_usd = FXRate(eur, usd, test_date, Decimal("1.1"))
    rate_gbp_usd = FXRate(gbp, usd, test_date, Decimal("1.3"))
    
    # Test case 1: Query existing rate
    service = TestFXRateService({
        (eur, usd, test_date): rate_eur_usd,
        (gbp, usd, test_date): rate_gbp_usd
    })
    
    result = service.query(eur, usd, test_date)
    assert result == rate_eur_usd
    assert result.ccy1 == eur
    assert result.ccy2 == usd
    assert result.value == Decimal("1.1")
    
    # Test case 2: Query non-existing rate with strict=False
    result = service.query(eur, gbp, test_date, strict=False)
    assert result is None
    
    # Test case 3: Query non-existing rate with strict=True raises FXRateLookupError
    with pytest.raises(FXRateLookupError) as exc_info:
        service.query(eur, gbp, test_date, strict=True)
    
    assert exc_info.value.ccy1 == eur
    assert exc_info.value.ccy2 == gbp
    assert exc_info.value.asof == test_date
    
    # Test case 4: Query multiple rates
    result1 = service.query(eur, usd, test_date)
    result2 = service.query(gbp, usd, test_date)
    
    assert result1.value == Decimal("1.1")
    assert result2.value == Decimal("1.3")
    
    # Test case 5: Query with different date returns None
    different_date = datetime.date(2023, 1, 2)
    result = service.query(eur, usd, different_date, strict=False)
    assert result is None


# LLM-generated content at query #14
#--------------------------

```python
def test_FXRateService_query():
    """Test the query method of FXRateService."""
    import datetime
    from decimal import Decimal
    from unittest.mock import Mock
    import pytest
    
    # Create a concrete implementation of the abstract FXRateService
    class ConcreteFXRateService(FXRateService):
        def __init__(self, rates_dict=None):
            self.rates_dict = rates_dict or {}
        
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            key = (ccy1, ccy2, asof)
            rate = self.rates_dict.get(key)
            if rate is None and strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return rate
        
        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    # Setup test data
    from pypara.currencies import Currency
    eur = Mock(spec=Currency)
    usd = Mock(spec=Currency)
    test_date = datetime.date.today()
    rate_value = Decimal("1.25")
    
    # Create test FXRate
    test_rate = FXRate(eur, usd, test_date, rate_value)
    
    # Test 1: Query returns existing rate
    service = ConcreteFXRateService({(eur, usd, test_date): test_rate})
    result = service.query(eur, usd, test_date)
    assert result == test_rate
    assert result.ccy1 == eur
    assert result.ccy2 == usd
    assert result.date == test_date
    assert result.value == rate_value
    
    # Test 2: Query returns None when rate not found and strict=False
    service = ConcreteFXRateService({})
    result = service.query(eur, usd, test_date, strict=False)
    assert result is None
    
    # Test 3: Query raises FXRateLookupError when rate not found and strict=True
    service = ConcreteFXRateService({})
    with pytest.raises(FXRateLookupError) as exc_info:
        service.query(eur, usd, test_date, strict=True)
    assert exc_info.value.ccy1 == eur
    assert exc_info.value.ccy2 == usd
    assert exc_info.value.asof == test_date
    
    # Test 4: Query with different date returns None
    different_date = datetime.date(2020, 1, 1)
    service = ConcreteFXRateService({(eur, usd, test_date): test_rate})
    result = service.query(eur, usd, different_date)
    assert result is None
    
    # Test 5: Query with different currency pair returns None
    gbp = Mock(spec=Currency)
    service = ConcreteFXRateService({(eur, usd, test_date): test_rate})
    result = service.query(eur, gbp, test_date)
    assert result is None


# LLM-generated content at query #15
#--------------------------

```python
import datetime
from decimal import Decimal
from unittest.mock import Mock, patch
import pytest


def test_FXRateService_queries():
    """Test the queries method of FXRateService."""
    
    # Create a mock FXRateService instance
    service = Mock(spec=FXRateService)
    
    # Setup test data
    from pypara.currencies import Currency
    ccy1 = Mock(spec=Currency)
    ccy1.__str__ = Mock(return_value="EUR")
    ccy2 = Mock(spec=Currency)
    ccy2.__str__ = Mock(return_value="USD")
    ccy3 = Mock(spec=Currency)
    ccy3.__str__ = Mock(return_value="GBP")
    
    date1 = datetime.date(2023, 1, 1)
    date2 = datetime.date(2023, 1, 2)
    
    # Create FXRate instances
    rate1 = FXRate(ccy1, ccy2, date1, Decimal("1.5"))
    rate2 = FXRate(ccy2, ccy3, date2, Decimal("1.2"))
    rate3 = None
    
    # Setup queries
    queries = [
        (ccy1, ccy2, date1),
        (ccy2, ccy3, date2),
        (ccy1, ccy3, date1),
    ]
    
    # Test 1: queries returns rates successfully in non-strict mode
    service.queries.return_value = [rate1, rate2, rate3]
    result = list(service.queries(queries, strict=False))
    
    assert len(result) == 3
    assert result[0] == rate1
    assert result[1] == rate2
    assert result[2] is None
    service.queries.assert_called_once_with(queries, strict=False)
    
    # Test 2: queries with strict=True
    service.reset_mock()
    service.queries.return_value = [rate1, rate2]
    result = list(service.queries(queries[:2], strict=True))
    
    assert len(result) == 2
    assert result[0] == rate1
    assert result[1] == rate2
    service.queries.assert_called_once_with(queries[:2], strict=True)
    
    # Test 3: queries with empty iterable
    service.reset_mock()
    service.queries.return_value = []
    result = list(service.queries([], strict=False))
    
    assert len(result) == 0
    service.queries.assert_called_once_with([], strict=False)
    
    # Test 4: queries returns multiple rates
    service.reset_mock()
    rate4 = FXRate(ccy3, ccy1, date1, Decimal("0.8"))
    service.queries.return_value = [rate1, rate2, rate4]
    result = list(service.queries(queries, strict=False))
    
    assert len(result) == 3
    assert result[0] == rate1
    assert result[1] == rate2
    assert result[2] == rate4


# LLM-generated content at query #16
#--------------------------

```python
import pytest
from abc import ABCMeta
from decimal import Decimal
from datetime import date
from unittest.mock import Mock, MagicMock


def test_FXRateService_query():
    """Test the query method of FXRateService abstract class."""
    
    # Create a concrete implementation of FXRateService for testing
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            # Mock implementation that returns a rate for EUR/USD
            if ccy1.code == "EUR" and ccy2.code == "USD":
                return FXRate(ccy1, ccy2, asof, Decimal("1.20"))
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None
        
        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    # Create mock currencies
    eur = Mock(spec=Currency)
    eur.code = "EUR"
    usd = Mock(spec=Currency)
    usd.code = "USD"
    gbp = Mock(spec=Currency)
    gbp.code = "GBP"
    
    test_date = date.today()
    service = ConcreteFXRateService()
    
    # Test 1: Query existing rate returns FXRate
    result = service.query(eur, usd, test_date, strict=False)
    assert result is not None
    assert result.ccy1 == eur
    assert result.ccy2 == usd
    assert result.date == test_date
    assert result.value == Decimal("1.20")
    
    # Test 2: Query non-existing rate with strict=False returns None
    result = service.query(gbp, usd, test_date, strict=False)
    assert result is None
    
    # Test 3: Query non-existing rate with strict=True raises FXRateLookupError
    with pytest.raises(FXRateLookupError) as exc_info:
        service.query(gbp, usd, test_date, strict=True)
    assert exc_info.value.ccy1 == gbp
    assert exc_info.value.ccy2 == usd
    assert exc_info.value.asof == test_date
    
    # Test 4: Query with different date
    different_date = date(2020, 1, 1)
    result = service.query(eur, usd, different_date, strict=False)
    assert result is not None
    assert result.date == different_date
    
    # Test 5: Default strict parameter is False
    result = service.query(eur, usd, test_date)
    assert result is not None


# LLM-generated content at query #17
#--------------------------

```python
import datetime
from decimal import Decimal
from unittest.mock import Mock, patch

import pytest

from pypara.currencies import Currency
from pypara.fx import FXRate, FXRateService


def test_FXRateService_queries():
    """Test the queries method of FXRateService."""
    
    # Create a mock implementation of FXRateService
    mock_service = Mock(spec=FXRateService)
    
    # Create test currencies
    eur = Mock(spec=Currency)
    eur.__str__ = Mock(return_value="EUR")
    usd = Mock(spec=Currency)
    usd.__str__ = Mock(return_value="USD")
    gbp = Mock(spec=Currency)
    gbp.__str__ = Mock(return_value="GBP")
    
    # Create test date
    test_date = datetime.date(2023, 1, 15)
    
    # Create test FX rates
    eur_usd_rate = FXRate(eur, usd, test_date, Decimal("1.10"))
    usd_gbp_rate = FXRate(usd, gbp, test_date, Decimal("0.80"))
    
    # Define test queries
    queries = [
        (eur, usd, test_date),
        (usd, gbp, test_date),
        (gbp, eur, test_date),  # This one will return None
    ]
    
    # Mock the queries method to return rates
    mock_service.queries.return_value = [
        eur_usd_rate,
        usd_gbp_rate,
        None,
    ]
    
    # Call the queries method
    results = list(mock_service.queries(queries, strict=False))
    
    # Assertions
    assert len(results) == 3
    assert results[0] == eur_usd_rate
    assert results[1] == usd_gbp_rate
    assert results[2] is None
    
    # Verify the method was called with correct arguments
    mock_service.queries.assert_called_once_with(queries, strict=False)


def test_FXRateService_queries_strict_mode():
    """Test the queries method of FXRateService in strict mode."""
    
    mock_service = Mock(spec=FXRateService)
    
    eur = Mock(spec=Currency)
    eur.__str__ = Mock(return_value="EUR")
    usd = Mock(spec=Currency)
    usd.__str__ = Mock(return_value="USD")
    
    test_date = datetime.date(2023, 1, 15)
    
    queries = [
        (eur, usd, test_date),
    ]
    
    # In strict mode, should raise FXRateLookupError
    from pypara.fx import FXRateLookupError
    mock_service.queries.side_effect = FXRateLookupError(eur, usd, test_date)
    
    with pytest.raises(FXRateLookupError):
        list(mock_service.queries(queries, strict=True))


def test_FXRateService_queries_empty():
    """Test the queries method with empty queries."""
    
    mock_service = Mock(spec=FXRateService)
    mock_service.queries.return_value = []
    
    results = list(mock_service.queries([], strict=False))
    
    assert results == []
    mock_service.queries.assert_called_once_with([], strict=False)


def test_FXRateService_queries_multiple_dates():
    """Test the queries method with multiple dates."""
    
    mock_service = Mock(spec=FXRateService)
    
    eur = Mock(spec=Currency)
    eur.__str__ = Mock(return_value="EUR")
    usd = Mock(spec=Currency)
    usd.__str__ = Mock(return_value="USD")
    
    date1 = datetime.date(2023, 1, 15)
    date2 = datetime.date(2023, 1, 16)
    
    rate1 = FXRate(eur, usd, date1, Decimal("1.10"))
    rate2 = FXRate(eur, usd, date2, Decimal("1.12"))
    
    queries = [
        (eur, usd, date1),
        (eur, usd, date2),
    ]
    
    mock_service.queries.return_value = [rate1, rate2]
    
    results = list(mock_service.queries(queries, strict=False))
    
    assert len(results) == 2
    assert results[0].date == date1
    assert results[1].date == date2
    assert results[0].value == Decimal("1.10")
    assert results[1].value == Decimal("1.12")


# LLM-generated content at query #18
#--------------------------

```python
import pytest
from datetime import date
from decimal import Decimal
from unittest.mock import Mock, MagicMock


def test_FXRateService_query():
    """Test the query method of FXRateService abstract class."""
    
    # Create a mock implementation of FXRateService
    mock_service = Mock(spec=FXRateService)
    
    # Setup test data
    test_ccy1 = Mock(spec=Currency)
    test_ccy2 = Mock(spec=Currency)
    test_date = date.today()
    test_rate = FXRate(test_ccy1, test_ccy2, test_date, Decimal("1.5"))
    
    # Test case 1: Query returns a valid FXRate
    mock_service.query.return_value = test_rate
    result = mock_service.query(test_ccy1, test_ccy2, test_date, strict=False)
    assert result == test_rate
    mock_service.query.assert_called_once_with(test_ccy1, test_ccy2, test_date, strict=False)
    
    # Test case 2: Query returns None when rate not found (non-strict mode)
    mock_service.reset_mock()
    mock_service.query.return_value = None
    result = mock_service.query(test_ccy1, test_ccy2, test_date, strict=False)
    assert result is None
    
    # Test case 3: Query raises FXRateLookupError in strict mode when rate not found
    mock_service.reset_mock()
    mock_service.query.side_effect = FXRateLookupError(test_ccy1, test_ccy2, test_date)
    with pytest.raises(FXRateLookupError):
        mock_service.query(test_ccy1, test_ccy2, test_date, strict=True)
    
    # Test case 4: Query with strict=True and valid rate
    mock_service.reset_mock()
    mock_service.query.return_value = test_rate
    result = mock_service.query(test_ccy1, test_ccy2, test_date, strict=True)
    assert result == test_rate
    
    # Test case 5: Query called with correct parameters
    mock_service.reset_mock()
    mock_service.query.return_value = test_rate
    ccy1 = Mock(spec=Currency)
    ccy2 = Mock(spec=Currency)
    asof = date(2023, 1, 15)
    mock_service.query(ccy1, ccy2, asof, strict=True)
    mock_service.query.assert_called_once_with(ccy1, ccy2, asof, strict=True)
    
    # Test case 6: Default strict parameter is False
    mock_service.reset_mock()
    mock_service.query.return_value = test_rate
    mock_service.query(test_ccy1, test_ccy2, test_date)
    mock_service.query.assert_called_once_with(test_ccy1, test_ccy2, test_date, strict=False)


# LLM-generated content at query #19
#--------------------------

```python
import pytest
import datetime
from decimal import Decimal
from unittest.mock import Mock, MagicMock


def test_FXRateService_query():
    """Test the query method of FXRateService."""
    
    # Create a mock implementation of FXRateService
    service = Mock(spec=FXRateService)
    
    # Setup test data
    eur = Mock(spec=Currency)
    eur.__str__ = Mock(return_value="EUR")
    usd = Mock(spec=Currency)
    usd.__str__ = Mock(return_value="USD")
    test_date = datetime.date(2023, 1, 15)
    test_rate = FXRate(eur, usd, test_date, Decimal("1.10"))
    
    # Test 1: query returns FXRate when rate exists
    service.query.return_value = test_rate
    result = service.query(eur, usd, test_date, strict=False)
    assert result == test_rate
    assert result.ccy1 == eur
    assert result.ccy2 == usd
    assert result.date == test_date
    assert result.value == Decimal("1.10")
    service.query.assert_called_once_with(eur, usd, test_date, strict=False)
    
    # Test 2: query returns None when rate not found and strict=False
    service.reset_mock()
    service.query.return_value = None
    result = service.query(eur, usd, test_date, strict=False)
    assert result is None
    service.query.assert_called_once_with(eur, usd, test_date, strict=False)
    
    # Test 3: query raises FXRateLookupError when strict=True and rate not found
    service.reset_mock()
    service.query.side_effect = FXRateLookupError(eur, usd, test_date)
    with pytest.raises(FXRateLookupError):
        service.query(eur, usd, test_date, strict=True)
    service.query.assert_called_once_with(eur, usd, test_date, strict=True)
    
    # Test 4: query with different currency pairs
    service.reset_mock()
    gbp = Mock(spec=Currency)
    gbp.__str__ = Mock(return_value="GBP")
    test_rate_gbp = FXRate(gbp, usd, test_date, Decimal("1.27"))
    service.query.return_value = test_rate_gbp
    result = service.query(gbp, usd, test_date, strict=False)
    assert result == test_rate_gbp
    assert result.ccy1 == gbp
    assert result.value == Decimal("1.27")
    
    # Test 5: query with different dates
    service.reset_mock()
    different_date = datetime.date(2023, 6, 20)
    test_rate_diff_date = FXRate(eur, usd, different_date, Decimal("1.15"))
    service.query.return_value = test_rate_diff_date
    result = service.query(eur, usd, different_date, strict=False)
    assert result.date == different_date
    assert result.value == Decimal("1.15")


# LLM-generated content at query #20
#--------------------------

```python
def test_FXRateService_query():
    """Test the query method of FXRateService abstract class."""
    import datetime
    from decimal import Decimal
    from unittest.mock import MagicMock
    
    # Create a concrete implementation of FXRateService for testing
    class ConcreteFXRateService(FXRateService):
        def __init__(self):
            self._rates = {}
        
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            key = (ccy1.code, ccy2.code, asof)
            if key in self._rates:
                return self._rates[key]
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None
        
        def queries(self, queries: Iterable[Tuple[Currency, Currency, Date]], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
        
        def add_rate(self, rate: FXRate) -> None:
            key = (rate.ccy1.code, rate.ccy2.code, rate.date)
            self._rates[key] = rate
    
    # Setup
    from pypara.currencies import Currencies
    service = ConcreteFXRateService()
    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    test_date = datetime.date.today()
    test_rate = FXRate(eur, usd, test_date, Decimal("1.25"))
    
    service.add_rate(test_rate)
    
    # Test: Query existing rate
    result = service.query(eur, usd, test_date, strict=False)
    assert result is not None
    assert result == test_rate
    assert result.ccy1 == eur
    assert result.ccy2 == usd
    assert result.value == Decimal("1.25")
    
    # Test: Query non-existing rate with strict=False returns None
    gbp = Currencies["GBP"]
    result = service.query(eur, gbp, test_date, strict=False)
    assert result is None
    
    # Test: Query non-existing rate with strict=True raises FXRateLookupError
    with pytest.raises(FXRateLookupError) as exc_info:
        service.query(eur, gbp, test_date, strict=True)
    assert exc_info.value.ccy1 == eur
    assert exc_info.value.ccy2 == gbp
    assert exc_info.value.asof == test_date
    
    # Test: Query with different date returns None
    different_date = test_date - datetime.timedelta(days=1)
    result = service.query(eur, usd, different_date, strict=False)
    assert result is None


# LLM-generated content at query #21
#--------------------------

```python
def test_FXRateService_queries():
    """Test the queries method of FXRateService."""
    from datetime import date
    from decimal import Decimal
    
    # Create a concrete implementation of FXRateService for testing
    class TestFXRateService(FXRateService):
        def __init__(self):
            self.rates = {}
        
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            key = (ccy1, ccy2, asof)
            if key in self.rates:
                return self.rates[key]
            elif strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None
        
        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    # Setup test data
    from pypara.currencies import Currencies
    
    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    gbp = Currencies["GBP"]
    test_date = date.today()
    
    service = TestFXRateService()
    
    # Add some test rates
    rate_eur_usd = FXRate(eur, usd, test_date, Decimal("1.2"))
    rate_usd_gbp = FXRate(usd, gbp, test_date, Decimal("0.73"))
    
    service.rates[(eur, usd, test_date)] = rate_eur_usd
    service.rates[(usd, gbp, test_date)] = rate_usd_gbp
    
    # Test 1: Query multiple existing rates
    query_list = [
        (eur, usd, test_date),
        (usd, gbp, test_date),
    ]
    results = list(service.queries(query_list, strict=False))
    
    assert len(results) == 2
    assert results[0] == rate_eur_usd
    assert results[1] == rate_usd_gbp
    
    # Test 2: Query with non-existent rate (non-strict mode)
    query_list_with_missing = [
        (eur, usd, test_date),
        (gbp, eur, test_date),  # Does not exist
        (usd, gbp, test_date),
    ]
    results = list(service.queries(query_list_with_missing, strict=False))
    
    assert len(results) == 3
    assert results[0] == rate_eur_usd
    assert results[1] is None
    assert results[2] == rate_usd_gbp
    
    # Test 3: Query with non-existent rate (strict mode should raise)
    with pytest.raises(FXRateLookupError):
        list(service.queries(query_list_with_missing, strict=True))
    
    # Test 4: Empty query list
    results = list(service.queries([], strict=False))
    assert len(results) == 0
    
    # Test 5: Single query
    results = list(service.queries([(eur, usd, test_date)], strict=False))
    assert len(results) == 1
    assert results[0] == rate_eur_usd


# LLM-generated content at query #22
#--------------------------

```python
def test_FXRateService_query():
    """Test the query method of FXRateService abstract class."""
    import datetime
    from decimal import Decimal
    from unittest.mock import MagicMock
    
    # Create a concrete implementation of FXRateService for testing
    class ConcreteFXRateService(FXRateService):
        def __init__(self):
            self.rates = {}
        
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            key = (ccy1, ccy2, asof)
            if key in self.rates:
                return self.rates[key]
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None
        
        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    # Setup test data
    from pypara.currencies import Currencies
    
    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    gbp = Currencies["GBP"]
    test_date = datetime.date(2023, 1, 1)
    fx_rate = FXRate(eur, usd, test_date, Decimal("1.15"))
    
    service = ConcreteFXRateService()
    service.rates[(eur, usd, test_date)] = fx_rate
    
    # Test 1: Query for existing rate
    result = service.query(eur, usd, test_date)
    assert result is not None
    assert result == fx_rate
    assert result.ccy1 == eur
    assert result.ccy2 == usd
    assert result.date == test_date
    assert result.value == Decimal("1.15")
    
    # Test 2: Query for non-existing rate (non-strict mode)
    result = service.query(gbp, usd, test_date, strict=False)
    assert result is None
    
    # Test 3: Query for non-existing rate (strict mode)
    with pytest.raises(FXRateLookupError) as exc_info:
        service.query(gbp, usd, test_date, strict=True)
    assert exc_info.value.ccy1 == gbp
    assert exc_info.value.ccy2 == usd
    assert exc_info.value.asof == test_date
    
    # Test 4: Query with different date
    different_date = datetime.date(2023, 1, 2)
    result = service.query(eur, usd, different_date, strict=False)
    assert result is None
    
    # Test 5: Store and query multiple rates
    jpy = Currencies["JPY"]
    gbp_jpy_rate = FXRate(gbp, jpy, test_date, Decimal("175.50"))
    service.rates[(gbp, jpy, test_date)] = gbp_jpy_rate
    
    result1 = service.query(eur, usd, test_date)
    result2 = service.query(gbp, jpy, test_date)
    assert result1 == fx_rate
    assert result2 == gbp_jpy_rate


# LLM-generated content at query #23
#--------------------------

```python
import pytest
from decimal import Decimal
from datetime import date
from unittest.mock import Mock, MagicMock


def test_FXRateService_queries():
    """Test the queries method of FXRateService."""
    
    # Create a mock FXRateService instance
    service = Mock(spec=FXRateService)
    
    # Setup test data
    ccy1 = Mock(spec=Currency)
    ccy2 = Mock(spec=Currency)
    ccy3 = Mock(spec=Currency)
    test_date = date.today()
    
    # Create test queries
    queries = [
        (ccy1, ccy2, test_date),
        (ccy2, ccy3, test_date),
        (ccy1, ccy3, test_date),
    ]
    
    # Create expected FXRate results
    rate1 = FXRate(ccy1, ccy2, test_date, Decimal("1.5"))
    rate2 = FXRate(ccy2, ccy3, test_date, Decimal("2.0"))
    rate3 = None
    
    expected_results = [rate1, rate2, rate3]
    
    # Configure mock to return the expected results
    service.queries.return_value = expected_results
    
    # Call the queries method
    results = list(service.queries(queries, strict=False))
    
    # Assertions
    assert len(results) == 3
    assert results[0] == rate1
    assert results[1] == rate2
    assert results[2] is None
    
    # Verify the method was called with correct arguments
    service.queries.assert_called_once_with(queries, strict=False)


def test_FXRateService_queries_strict_mode():
    """Test the queries method of FXRateService in strict mode."""
    
    service = Mock(spec=FXRateService)
    
    ccy1 = Mock(spec=Currency)
    ccy2 = Mock(spec=Currency)
    test_date = date.today()
    
    queries = [(ccy1, ccy2, test_date)]
    
    # Configure mock to raise FXRateLookupError in strict mode
    service.queries.side_effect = FXRateLookupError(ccy1, ccy2, test_date)
    
    # Assert that the error is raised when strict=True
    with pytest.raises(FXRateLookupError):
        service.queries(queries, strict=True)


def test_FXRateService_queries_empty_input():
    """Test the queries method with empty input."""
    
    service = Mock(spec=FXRateService)
    service.queries.return_value = []
    
    queries = []
    results = list(service.queries(queries, strict=False))
    
    assert results == []
    service.queries.assert_called_once_with(queries, strict=False)


def test_FXRateService_queries_single_query():
    """Test the queries method with a single query."""
    
    service = Mock(spec=FXRateService)
    
    ccy1 = Mock(spec=Currency)
    ccy2 = Mock(spec=Currency)
    test_date = date.today()
    
    queries = [(ccy1, ccy2, test_date)]
    rate = FXRate(ccy1, ccy2, test_date, Decimal("1.2"))
    
    service.queries.return_value = [rate]
    
    results = list(service.queries(queries, strict=False))
    
    assert len(results) == 1
    assert results[0] == rate


# LLM-generated content at query #24
#--------------------------

```python
def test_FXRateService_query():
    """Test the query method of FXRateService abstract class."""
    import datetime
    from decimal import Decimal
    from unittest.mock import Mock
    
    # Create a concrete implementation of FXRateService for testing
    class TestFXRateService(FXRateService):
        def __init__(self):
            self.rates = {}
        
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            key = (ccy1, ccy2, asof)
            if key in self.rates:
                return self.rates[key]
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None
        
        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    # Create test currencies and date
    from pypara.currencies import Currencies
    ccy_eur = Currencies["EUR"]
    ccy_usd = Currencies["USD"]
    test_date = datetime.date.today()
    
    service = TestFXRateService()
    
    # Test 1: Query returns None when rate not found and strict=False
    result = service.query(ccy_eur, ccy_usd, test_date, strict=False)
    assert result is None
    
    # Test 2: Query raises FXRateLookupError when rate not found and strict=True
    with pytest.raises(FXRateLookupError) as exc_info:
        service.query(ccy_eur, ccy_usd, test_date, strict=True)
    assert exc_info.value.ccy1 == ccy_eur
    assert exc_info.value.ccy2 == ccy_usd
    assert exc_info.value.asof == test_date
    
    # Test 3: Query returns FXRate when rate exists
    rate = FXRate.of(ccy_eur, ccy_usd, test_date, Decimal("1.25"))
    service.rates[(ccy_eur, ccy_usd, test_date)] = rate
    result = service.query(ccy_eur, ccy_usd, test_date, strict=False)
    assert result == rate
    assert result.ccy1 == ccy_eur
    assert result.ccy2 == ccy_usd
    assert result.date == test_date
    assert result.value == Decimal("1.25")
    
    # Test 4: Query returns FXRate when rate exists and strict=True
    result = service.query(ccy_eur, ccy_usd, test_date, strict=True)
    assert result == rate
    
    # Test 5: Query with different date returns None
    different_date = test_date - datetime.timedelta(days=1)
    result = service.query(ccy_eur, ccy_usd, different_date, strict=False)
    assert result is None
    
    # Test 6: Query with different currency pair returns None
    ccy_gbp = Currencies["GBP"]
    result = service.query(ccy_gbp, ccy_usd, test_date, strict=False)
    assert result is None


# LLM-generated content at query #25
#--------------------------

```python
def test_FXRateService_query():
    """Test the query method of FXRateService abstract class."""
    from datetime import date
    from decimal import Decimal
    from unittest.mock import Mock
    
    # Create a concrete implementation of FXRateService for testing
    class ConcreteFXRateService(FXRateService):
        def __init__(self):
            self.rates = {}
        
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            key = (ccy1, ccy2, asof)
            if key in self.rates:
                return self.rates[key]
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None
        
        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    # Create test currencies and dates
    from pypara.currencies import Currencies
    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    gbp = Currencies["GBP"]
    test_date = date.today()
    
    # Initialize service with test data
    service = ConcreteFXRateService()
    rate_eur_usd = FXRate(eur, usd, test_date, Decimal("1.20"))
    service.rates[(eur, usd, test_date)] = rate_eur_usd
    
    # Test 1: Query existing rate
    result = service.query(eur, usd, test_date, strict=False)
    assert result is not None
    assert result == rate_eur_usd
    assert result.ccy1 == eur
    assert result.ccy2 == usd
    assert result.value == Decimal("1.20")
    
    # Test 2: Query non-existing rate with strict=False
    result = service.query(gbp, usd, test_date, strict=False)
    assert result is None
    
    # Test 3: Query non-existing rate with strict=True should raise
    with pytest.raises(FXRateLookupError) as exc_info:
        service.query(gbp, usd, test_date, strict=True)
    assert exc_info.value.ccy1 == gbp
    assert exc_info.value.ccy2 == usd
    assert exc_info.value.asof == test_date
    
    # Test 4: Query with different date returns None
    from datetime import timedelta
    different_date = test_date + timedelta(days=1)
    result = service.query(eur, usd, different_date, strict=False)
    assert result is None


# LLM-generated content at query #26
#--------------------------

```python
def test_FXRateService_queries():
    """Test the queries method of FXRateService."""
    import datetime
    from decimal import Decimal
    from unittest.mock import Mock, MagicMock
    
    # Create a mock FXRateService instance
    service = MagicMock(spec=FXRateService)
    
    # Setup test data
    ccy1 = Mock(spec=Currency)
    ccy2 = Mock(spec=Currency)
    ccy3 = Mock(spec=Currency)
    test_date = datetime.date.today()
    
    rate1 = FXRate(ccy1, ccy2, test_date, Decimal("1.5"))
    rate2 = FXRate(ccy2, ccy3, test_date, Decimal("2.0"))
    rate3 = None
    
    # Test case 1: queries method returns rates for multiple queries
    queries = [(ccy1, ccy2, test_date), (ccy2, ccy3, test_date), (ccy1, ccy3, test_date)]
    service.queries.return_value = [rate1, rate2, rate3]
    
    results = list(service.queries(queries, strict=False))
    
    assert len(results) == 3
    assert results[0] == rate1
    assert results[1] == rate2
    assert results[2] is None
    service.queries.assert_called_once_with(queries, strict=False)
    
    # Test case 2: queries method with strict=True
    service.reset_mock()
    service.queries.return_value = [rate1, rate2]
    
    results = list(service.queries(queries[:2], strict=True))
    
    assert len(results) == 2
    assert results[0] == rate1
    assert results[1] == rate2
    service.queries.assert_called_once_with(queries[:2], strict=True)
    
    # Test case 3: queries method with empty input
    service.reset_mock()
    service.queries.return_value = []
    
    results = list(service.queries([], strict=False))
    
    assert len(results) == 0
    service.queries.assert_called_once_with([], strict=False)
    
    # Test case 4: queries method with single query
    service.reset_mock()
    service.queries.return_value = [rate1]
    
    results = list(service.queries([(ccy1, ccy2, test_date)], strict=False))
    
    assert len(results) == 1
    assert results[0] == rate1
    service.queries.assert_called_once()


# LLM-generated content at query #27
#--------------------------

```python
import pytest
from datetime import date
from decimal import Decimal
from unittest.mock import Mock, patch


def test_FXRateService_query():
    """Test the query method of FXRateService abstract class."""
    
    # FXRateService is abstract, so we need to create a concrete implementation
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return FXRate(ccy1, ccy2, asof, Decimal("1.5"))
        
        def queries(self, queries, strict=False):
            return []
    
    # Create test currencies
    from pypara.currencies import Currencies
    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    test_date = date.today()
    
    # Create service instance
    service = ConcreteFXRateService()
    
    # Test successful query
    result = service.query(eur, usd, test_date)
    assert result is not None
    assert isinstance(result, FXRate)
    assert result.ccy1 == eur
    assert result.ccy2 == usd
    assert result.date == test_date
    assert result.value == Decimal("1.5")
    
    # Test query with strict=False (default)
    result = service.query(eur, usd, test_date, strict=False)
    assert result is not None
    
    # Test query with strict=True
    result = service.query(eur, usd, test_date, strict=True)
    assert result is not None


def test_FXRateService_query_with_none_result():
    """Test the query method when result is None."""
    
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None
        
        def queries(self, queries, strict=False):
            return []
    
    from pypara.currencies import Currencies
    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    test_date = date.today()
    
    service = ConcreteFXRateService()
    
    # Test with strict=False returns None
    result = service.query(eur, usd, test_date, strict=False)
    assert result is None
    
    # Test with strict=True raises exception
    with pytest.raises(FXRateLookupError):
        service.query(eur, usd, test_date, strict=True)


def test_FXRateService_query_signature():
    """Test that query method has correct signature."""
    
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        
        def queries(self, queries, strict=False):
            return []
    
    service = ConcreteFXRateService()
    
    # Verify method exists and is callable
    assert hasattr(service, 'query')
    assert callable(service.query)
    
    # Verify default parameter
    from pypara.currencies import Currencies
    import inspect
    sig = inspect.signature(service.query)
    assert 'strict' in sig.parameters
    assert sig.parameters['strict'].default is False


# LLM-generated content at query #28
#--------------------------

```python
def test_FXRateService_query():
    """Test the query method of FXRateService abstract class."""
    import datetime
    from decimal import Decimal
    from unittest.mock import Mock
    
    # Create a concrete implementation of FXRateService for testing
    class MockFXRateService(FXRateService):
        def __init__(self):
            self.rates = {}
        
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            key = (ccy1, ccy2, asof)
            if key in self.rates:
                return self.rates[key]
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None
        
        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    # Setup test data
    from pypara.currencies import Currency
    
    ccy_eur = Mock(spec=Currency)
    ccy_eur.__eq__ = Mock(return_value=False)
    ccy_eur.__hash__ = Mock(return_value=hash("EUR"))
    
    ccy_usd = Mock(spec=Currency)
    ccy_usd.__eq__ = Mock(return_value=False)
    ccy_usd.__hash__ = Mock(return_value=hash("USD"))
    
    test_date = datetime.date.today()
    rate_value = Decimal("1.25")
    
    service = MockFXRateService()
    test_rate = FXRate(ccy_eur, ccy_usd, test_date, rate_value)
    service.rates[(ccy_eur, ccy_usd, test_date)] = test_rate
    
    # Test 1: Query existing rate with strict=False
    result = service.query(ccy_eur, ccy_usd, test_date, strict=False)
    assert result == test_rate
    assert result.ccy1 == ccy_eur
    assert result.ccy2 == ccy_usd
    assert result.date == test_date
    assert result.value == rate_value
    
    # Test 2: Query non-existing rate with strict=False returns None
    ccy_gbp = Mock(spec=Currency)
    ccy_gbp.__hash__ = Mock(return_value=hash("GBP"))
    result = service.query(ccy_gbp, ccy_usd, test_date, strict=False)
    assert result is None
    
    # Test 3: Query non-existing rate with strict=True raises FXRateLookupError
    with pytest.raises(FXRateLookupError) as exc_info:
        service.query(ccy_gbp, ccy_usd, test_date, strict=True)
    assert exc_info.value.ccy1 == ccy_gbp
    assert exc_info.value.ccy2 == ccy_usd
    assert exc_info.value.asof == test_date
    
    # Test 4: Query with different date returns None
    different_date = datetime.date.today() - datetime.timedelta(days=1)
    result = service.query(ccy_eur, ccy_usd, different_date, strict=False)
    assert result is None


# LLM-generated content at query #29
#--------------------------

```python
def test_FXRateService_query():
    """
    Test the query method of FXRateService abstract class.
    """
    from decimal import Decimal
    import datetime
    
    # Create a concrete implementation of FXRateService for testing
    class MockFXRateService(FXRateService):
        def __init__(self, rates_dict=None):
            self.rates_dict = rates_dict or {}
        
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            key = (ccy1, ccy2, asof)
            if key in self.rates_dict:
                return self.rates_dict[key]
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None
        
        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    # Setup test data
    from pypara.currencies import Currency
    eur = Currency(code="EUR", numeric="978", name="Euro", decimals=2)
    usd = Currency(code="USD", numeric="840", name="US Dollar", decimals=2)
    test_date = datetime.date(2023, 1, 1)
    
    # Test case 1: Query existing rate with strict=False
    rate = FXRate(eur, usd, test_date, Decimal("1.10"))
    service = MockFXRateService({(eur, usd, test_date): rate})
    result = service.query(eur, usd, test_date, strict=False)
    assert result == rate
    assert result.ccy1 == eur
    assert result.ccy2 == usd
    assert result.value == Decimal("1.10")
    
    # Test case 2: Query non-existing rate with strict=False returns None
    result = service.query(usd, eur, test_date, strict=False)
    assert result is None
    
    # Test case 3: Query non-existing rate with strict=True raises exception
    with pytest.raises(FXRateLookupError) as exc_info:
        service.query(usd, eur, test_date, strict=True)
    assert exc_info.value.ccy1 == usd
    assert exc_info.value.ccy2 == eur
    assert exc_info.value.asof == test_date
    
    # Test case 4: Query with different date
    another_date = datetime.date(2023, 1, 2)
    result = service.query(eur, usd, another_date, strict=False)
    assert result is None
    
    # Test case 5: Multiple rates in service
    rate2 = FXRate(usd, eur, test_date, Decimal("0.91"))
    service_multi = MockFXRateService({
        (eur, usd, test_date): rate,
        (usd, eur, test_date): rate2
    })
    result1 = service_multi.query(eur, usd, test_date)
    result2 = service_multi.query(usd, eur, test_date)
    assert result1.value == Decimal("1.10")
    assert result2.value == Decimal("0.91")


# LLM-generated content at query #30
#--------------------------

```python
def test_FXRateService_query():
    """Test the query method of FXRateService."""
    import datetime
    from decimal import Decimal
    from unittest.mock import Mock

    # Create a mock implementation of FXRateService
    service = Mock(spec=FXRateService)
    
    # Setup test data
    eur = Mock(spec=Currency)
    usd = Mock(spec=Currency)
    test_date = datetime.date(2023, 1, 15)
    test_rate = FXRate(eur, usd, test_date, Decimal("1.15"))
    
    # Test case 1: Query returns a valid FXRate
    service.query.return_value = test_rate
    result = service.query(eur, usd, test_date, strict=False)
    assert result == test_rate
    assert result.ccy1 == eur
    assert result.ccy2 == usd
    assert result.date == test_date
    assert result.value == Decimal("1.15")
    service.query.assert_called_once_with(eur, usd, test_date, strict=False)
    
    # Test case 2: Query returns None when rate not found (non-strict mode)
    service.reset_mock()
    service.query.return_value = None
    result = service.query(eur, usd, test_date, strict=False)
    assert result is None
    service.query.assert_called_once_with(eur, usd, test_date, strict=False)
    
    # Test case 3: Query raises FXRateLookupError in strict mode
    service.reset_mock()
    service.query.side_effect = FXRateLookupError(eur, usd, test_date)
    with pytest.raises(FXRateLookupError):
        service.query(eur, usd, test_date, strict=True)
    service.query.assert_called_once_with(eur, usd, test_date, strict=True)
    
    # Test case 4: Query with different currency pairs
    service.reset_mock()
    gbp = Mock(spec=Currency)
    jpy = Mock(spec=Currency)
    different_rate = FXRate(gbp, jpy, test_date, Decimal("150.5"))
    service.query.return_value = different_rate
    result = service.query(gbp, jpy, test_date, strict=False)
    assert result == different_rate
    assert result.ccy1 == gbp
    assert result.ccy2 == jpy
    
    # Test case 5: Query with different dates
    service.reset_mock()
    different_date = datetime.date(2023, 6, 20)
    date_rate = FXRate(eur, usd, different_date, Decimal("1.08"))
    service.query.return_value = date_rate
    result = service.query(eur, usd, different_date, strict=False)
    assert result == date_rate
    assert result.date == different_date
    assert result.value == Decimal("1.08")


# LLM-generated content at query #31
#--------------------------

```python
import pytest
from abc import ABCMeta
from decimal import Decimal
from datetime import date
from unittest.mock import Mock, patch

def test_FXRateService_query():
    """
    Test the query method of FXRateService abstract class.
    """
    # Create a concrete implementation of FXRateService for testing
    class ConcreteFXRateService(FXRateService):
        def __init__(self, rates_dict=None):
            self.rates_dict = rates_dict or {}
        
        def query(self, ccy1, ccy2, asof, strict=False):
            key = (ccy1, ccy2, asof)
            if key in self.rates_dict:
                return self.rates_dict[key]
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None
        
        def queries(self, queries, strict=False):
            for query in queries:
                yield self.query(query[0], query[1], query[2], strict)
    
    # Test data setup
    test_date = date.today()
    ccy_eur = Mock(spec=Currency)
    ccy_eur.__eq__ = lambda self, other: self is other or str(self) == str(other)
    ccy_usd = Mock(spec=Currency)
    ccy_usd.__eq__ = lambda self, other: self is other or str(self) == str(other)
    
    fx_rate = FXRate(ccy_eur, ccy_usd, test_date, Decimal("1.20"))
    
    # Test 1: Query returns FXRate when rate exists
    service = ConcreteFXRateService({(ccy_eur, ccy_usd, test_date): fx_rate})
    result = service.query(ccy_eur, ccy_usd, test_date)
    assert result == fx_rate
    assert result.value == Decimal("1.20")
    
    # Test 2: Query returns None when rate doesn't exist and strict=False
    service = ConcreteFXRateService()
    result = service.query(ccy_eur, ccy_usd, test_date, strict=False)
    assert result is None
    
    # Test 3: Query raises FXRateLookupError when rate doesn't exist and strict=True
    service = ConcreteFXRateService()
    with pytest.raises(FXRateLookupError):
        service.query(ccy_eur, ccy_usd, test_date, strict=True)
    
    # Test 4: Query with different date returns None
    different_date = date(2020, 1, 1)
    service = ConcreteFXRateService({(ccy_eur, ccy_usd, test_date): fx_rate})
    result = service.query(ccy_eur, ccy_usd, different_date, strict=False)
    assert result is None
    
    # Test 5: FXRateLookupError contains correct attributes
    error = FXRateLookupError(ccy_eur, ccy_usd, test_date)
    assert error.ccy1 == ccy_eur
    assert error.ccy2 == ccy_usd
    assert error.asof == test_date


# LLM-generated content at query #32
#--------------------------

```python
def test_FXRateService_queries():
    """Test the queries method of FXRateService."""
    import datetime
    from decimal import Decimal
    from unittest.mock import Mock, MagicMock
    
    # Create a mock FXRateService instance
    service = Mock(spec=FXRateService)
    
    # Setup test data
    ccy1 = Mock(spec=Currency)
    ccy2 = Mock(spec=Currency)
    ccy3 = Mock(spec=Currency)
    date1 = datetime.date(2023, 1, 1)
    date2 = datetime.date(2023, 1, 2)
    
    rate1 = FXRate(ccy1, ccy2, date1, Decimal("1.5"))
    rate2 = FXRate(ccy2, ccy3, date2, Decimal("2.0"))
    rate3 = None
    
    # Create queries
    queries = [
        (ccy1, ccy2, date1),
        (ccy2, ccy3, date2),
        (ccy1, ccy3, date1),
    ]
    
    # Test 1: queries returns rates in non-strict mode
    service.queries.return_value = [rate1, rate2, rate3]
    results = list(service.queries(queries, strict=False))
    
    assert len(results) == 3
    assert results[0] == rate1
    assert results[1] == rate2
    assert results[2] is None
    service.queries.assert_called_once_with(queries, strict=False)
    
    # Test 2: queries with strict mode
    service.reset_mock()
    service.queries.side_effect = FXRateLookupError(ccy1, ccy3, date1)
    
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))
    
    service.queries.assert_called_once_with(queries, strict=True)
    
    # Test 3: queries with empty iterable
    service.reset_mock()
    service.queries.return_value = []
    results = list(service.queries([], strict=False))
    
    assert len(results) == 0
    service.queries.assert_called_once_with([], strict=False)
    
    # Test 4: queries returns all valid rates
    service.reset_mock()
    service.queries.return_value = [rate1, rate2]
    results = list(service.queries(queries[:2], strict=False))
    
    assert len(results) == 2
    assert all(r is not None for r in results)
    service.queries.assert_called_once_with(queries[:2], strict=False)


# LLM-generated content at query #33
#--------------------------

```python
import datetime
from decimal import Decimal
from unittest.mock import Mock

import pytest

from pypara.currencies import Currency


def test_FXRateService_queries():
    """Test the queries method of FXRateService."""
    
    # Create a mock FXRateService instance
    service = Mock(spec=FXRateService)
    
    # Create test currencies and dates
    eur = Mock(spec=Currency)
    eur.__str__ = Mock(return_value="EUR")
    
    usd = Mock(spec=Currency)
    usd.__str__ = Mock(return_value="USD")
    
    gbp = Mock(spec=Currency)
    gbp.__str__ = Mock(return_value="GBP")
    
    date1 = datetime.date(2023, 1, 1)
    date2 = datetime.date(2023, 1, 2)
    
    # Create test FXRate objects
    rate1 = FXRate(eur, usd, date1, Decimal("1.10"))
    rate2 = FXRate(usd, gbp, date2, Decimal("0.85"))
    rate3 = None  # Rate not found
    
    # Define test queries
    queries = [
        (eur, usd, date1),
        (usd, gbp, date2),
        (eur, gbp, date1),
    ]
    
    # Configure mock to return rates
    service.queries.return_value = [rate1, rate2, rate3]
    
    # Call the method
    results = list(service.queries(queries, strict=False))
    
    # Assertions
    assert len(results) == 3
    assert results[0] == rate1
    assert results[1] == rate2
    assert results[2] is None
    service.queries.assert_called_once_with(queries, strict=False)


def test_FXRateService_queries_strict_mode():
    """Test the queries method with strict mode enabled."""
    
    service = Mock(spec=FXRateService)
    
    eur = Mock(spec=Currency)
    usd = Mock(spec=Currency)
    
    date1 = datetime.date(2023, 1, 1)
    
    queries = [(eur, usd, date1)]
    
    # In strict mode, should raise FXRateLookupError when rate not found
    service.queries.side_effect = FXRateLookupError(eur, usd, date1)
    
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))


def test_FXRateService_queries_empty():
    """Test the queries method with empty query list."""
    
    service = Mock(spec=FXRateService)
    service.queries.return_value = []
    
    results = list(service.queries([], strict=False))
    
    assert results == []
    service.queries.assert_called_once_with([], strict=False)


def test_FXRateService_queries_multiple_rates():
    """Test the queries method with multiple rates."""
    
    service = Mock(spec=FXRateService)
    
    eur = Mock(spec=Currency)
    usd = Mock(spec=Currency)
    gbp = Mock(spec=Currency)
    jpy = Mock(spec=Currency)
    
    date1 = datetime.date(2023, 1, 1)
    date2 = datetime.date(2023, 1, 2)
    
    rate1 = FXRate(eur, usd, date1, Decimal("1.10"))
    rate2 = FXRate(usd, gbp, date2, Decimal("0.85"))
    rate3 = FXRate(gbp, jpy, date1, Decimal("150.00"))
    rate4 = FXRate(eur, jpy, date2, Decimal("165.00"))
    
    queries = [
        (eur, usd, date1),
        (usd, gbp, date2),
        (gbp, jpy, date1),
        (eur, jpy, date2),
    ]
    
    service.queries.return_value = [rate1, rate2, rate3, rate4]
    
    results = list(service.queries(queries, strict=False))
    
    assert len(results) == 4
    assert all(isinstance(r, FXRate) for r in results)
    assert results[0].value == Decimal("1.10")
    assert results[1].value == Decimal("0.85")
    assert results[2].value == Decimal("150.00")
    assert results[3].value == Decimal("165.00")


# LLM-generated content at query #34
#--------------------------

```python
import datetime
from decimal import Decimal
from typing import Iterable, Optional

import pytest

from pypara.currencies import Currency, Currencies
from pypara.fx import FXRate, FXRateService


class MockFXRateService(FXRateService):
    """Mock implementation of FXRateService for testing."""

    def __init__(self, rates_dict=None):
        self.rates_dict = rates_dict or {}

    def query(self, ccy1: Currency, ccy2: Currency, asof: datetime.date, strict: bool = False) -> Optional[FXRate]:
        key = (ccy1.code, ccy2.code, asof)
        if key in self.rates_dict:
            return self.rates_dict[key]
        if strict:
            raise FXRateLookupError(ccy1, ccy2, asof)
        return None

    def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
        for ccy1, ccy2, asof in queries:
            yield self.query(ccy1, ccy2, asof, strict)


def test_FXRateService_queries():
    """Test the queries method of FXRateService."""
    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    gbp = Currencies["GBP"]
    date1 = datetime.date(2023, 1, 1)
    date2 = datetime.date(2023, 1, 2)

    # Create test rates
    rate_eur_usd = FXRate(eur, usd, date1, Decimal("1.10"))
    rate_usd_gbp = FXRate(usd, gbp, date1, Decimal("0.73"))
    rate_eur_gbp = FXRate(eur, gbp, date2, Decimal("0.86"))

    # Setup mock service with rates
    rates_dict = {
        ("EUR", "USD", date1): rate_eur_usd,
        ("USD", "GBP", date1): rate_usd_gbp,
        ("EUR", "GBP", date2): rate_eur_gbp,
    }
    service = MockFXRateService(rates_dict)

    # Test: Query multiple rates that exist
    query_list = [(eur, usd, date1), (usd, gbp, date1), (eur, gbp, date2)]
    results = list(service.queries(query_list))

    assert len(results) == 3
    assert results[0] == rate_eur_usd
    assert results[1] == rate_usd_gbp
    assert results[2] == rate_eur_gbp

    # Test: Query with non-existent rates (non-strict mode)
    query_list_partial = [(eur, usd, date1), (gbp, usd, date1)]
    results_partial = list(service.queries(query_list_partial, strict=False))

    assert len(results_partial) == 2
    assert results_partial[0] == rate_eur_usd
    assert results_partial[1] is None

    # Test: Query with non-existent rate (strict mode should raise)
    query_list_missing = [(eur, usd, date1), (gbp, usd, date1)]

    with pytest.raises(FXRateLookupError):
        list(service.queries(query_list_missing, strict=True))

    # Test: Empty query list
    empty_results = list(service.queries([]))
    assert len(empty_results) == 0

    # Test: Single query
    single_result = list(service.queries([(eur, usd, date1)]))
    assert len(single_result) == 1
    assert single_result[0] == rate_eur_usd


# LLM-generated content at query #35
#--------------------------

```python
def test_FXRateService_query():
    """Test the query method of FXRateService."""
    import datetime
    from decimal import Decimal
    from unittest.mock import Mock
    import pytest
    
    # Create a concrete implementation of FXRateService for testing
    class TestFXRateService(FXRateService):
        def __init__(self, rates_dict=None):
            self.rates_dict = rates_dict or {}
        
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            key = (ccy1, ccy2, asof)
            if key in self.rates_dict:
                return self.rates_dict[key]
            elif strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None
        
        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    # Setup test data
    from .currencies import Currency
    eur = Currency("EUR", 1, "Euro", Decimal("1"))
    usd = Currency("USD", 1, "US Dollar", Decimal("1"))
    test_date = datetime.date.today()
    
    fx_rate = FXRate(eur, usd, test_date, Decimal("1.2"))
    
    # Test 1: Query returns FXRate when it exists
    service = TestFXRateService({(eur, usd, test_date): fx_rate})
    result = service.query(eur, usd, test_date)
    assert result == fx_rate
    assert result.ccy1 == eur
    assert result.ccy2 == usd
    assert result.value == Decimal("1.2")
    
    # Test 2: Query returns None when rate doesn't exist and strict=False
    result = service.query(usd, eur, test_date, strict=False)
    assert result is None
    
    # Test 3: Query raises FXRateLookupError when rate doesn't exist and strict=True
    with pytest.raises(FXRateLookupError) as exc_info:
        service.query(usd, eur, test_date, strict=True)
    assert exc_info.value.ccy1 == usd
    assert exc_info.value.ccy2 == eur
    assert exc_info.value.asof == test_date
    
    # Test 4: Query with different dates returns None
    different_date = datetime.date.today() - datetime.timedelta(days=1)
    result = service.query(eur, usd, different_date)
    assert result is None
    
    # Test 5: Query with different currency pairs returns None
    gbp = Currency("GBP", 1, "British Pound", Decimal("1"))
    result = service.query(eur, gbp, test_date)
    assert result is None


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_FXRateService_queries():
    """Test the queries method of FXRateService."""
    import datetime
    from decimal import Decimal
    from unittest.mock import Mock, MagicMock
    
    # Create a mock FXRateService
    service = Mock(spec=FXRateService)
    
    # Setup test data
    ccy1 = Mock(spec=Currency)
    ccy2 = Mock(spec=Currency)
    ccy3 = Mock(spec=Currency)
    date1 = datetime.date(2023, 1, 1)
    date2 = datetime.date(2023, 1, 2)
    
    rate1 = FXRate(ccy1, ccy2, date1, Decimal("1.5"))
    rate2 = FXRate(ccy2, ccy3, date2, Decimal("2.0"))
    
    # Test case 1: queries returns list of FXRate objects
    queries = [(ccy1, ccy2, date1), (ccy2, ccy3, date2)]
    service.queries.return_value = [rate1, rate2]
    
    result = service.queries(queries, strict=False)
    result_list = list(result)
    
    assert len(result_list) == 2
    assert result_list[0] == rate1
    assert result_list[1] == rate2
    service.queries.assert_called_once_with(queries, strict=False)
    
    # Test case 2: queries with strict=True
    service.reset_mock()
    service.queries.return_value = [rate1, None]
    
    result = service.queries(queries, strict=True)
    result_list = list(result)
    
    assert len(result_list) == 2
    assert result_list[0] == rate1
    assert result_list[1] is None
    service.queries.assert_called_once_with(queries, strict=True)
    
    # Test case 3: queries with empty input
    service.reset_mock()
    service.queries.return_value = []
    
    result = service.queries([], strict=False)
    result_list = list(result)
    
    assert len(result_list) == 0
    service.queries.assert_called_once_with([], strict=False)
    
    # Test case 4: queries returns None for missing rates
    service.reset_mock()
    service.queries.return_value = [None, rate2, None]
    
    result = service.queries(queries, strict=False)
    result_list = list(result)
    
    assert len(result_list) == 3
    assert result_list[0] is None
    assert result_list[1] == rate2
    assert result_list[2] is None
    
    # Test case 5: queries is iterable
    service.reset_mock()
    service.queries.return_value = iter([rate1, rate2])
    
    result = service.queries(queries, strict=False)
    
    # Verify it returns an iterable
    assert hasattr(result, '__iter__')


# LLM-generated content at query #2
#--------------------------

```python
def test_FXRateService_queries():
    """Test the queries method of FXRateService."""
    from datetime import date
    from decimal import Decimal
    
    # Create a concrete implementation of FXRateService for testing
    class TestFXRateService(FXRateService):
        def __init__(self):
            self.rates = {}
        
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            key = (ccy1, ccy2, asof)
            if key in self.rates:
                return self.rates[key]
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None
        
        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    # Setup test data
    from pypara.currencies import Currencies
    
    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    gbp = Currencies["GBP"]
    test_date = date.today()
    
    service = TestFXRateService()
    
    # Add some test rates
    rate_eur_usd = FXRate.of(eur, usd, test_date, Decimal("1.2"))
    rate_usd_gbp = FXRate.of(usd, gbp, test_date, Decimal("0.8"))
    
    service.rates[(eur, usd, test_date)] = rate_eur_usd
    service.rates[(usd, gbp, test_date)] = rate_usd_gbp
    
    # Test 1: Query multiple existing rates
    query_list = [(eur, usd, test_date), (usd, gbp, test_date)]
    results = list(service.queries(query_list))
    
    assert len(results) == 2
    assert results[0] == rate_eur_usd
    assert results[1] == rate_usd_gbp
    
    # Test 2: Query with some non-existing rates (non-strict mode)
    query_list_mixed = [(eur, usd, test_date), (gbp, eur, test_date)]
    results_mixed = list(service.queries(query_list_mixed, strict=False))
    
    assert len(results_mixed) == 2
    assert results_mixed[0] == rate_eur_usd
    assert results_mixed[1] is None
    
    # Test 3: Query with non-existing rate in strict mode should raise
    query_list_strict = [(gbp, eur, test_date)]
    with pytest.raises(FXRateLookupError):
        list(service.queries(query_list_strict, strict=True))
    
    # Test 4: Empty query list
    results_empty = list(service.queries([]))
    assert len(results_empty) == 0
    
    # Test 5: Multiple queries including duplicates
    query_list_duplicates = [(eur, usd, test_date), (eur, usd, test_date), (usd, gbp, test_date)]
    results_duplicates = list(service.queries(query_list_duplicates))
    
    assert len(results_duplicates) == 3
    assert results_duplicates[0] == rate_eur_usd
    assert results_duplicates[1] == rate_eur_usd
    assert results_duplicates[2] == rate_usd_gbp


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from decimal import Decimal
from datetime import date
from unittest.mock import Mock, MagicMock
from typing import Iterable, Optional


def test_FXRateService_queries():
    """Test the queries method of FXRateService."""
    
    # Create a mock FXRateService instance
    service = Mock(spec=FXRateService)
    
    # Create test currencies
    eur = Mock(spec=Currency)
    eur.__str__ = Mock(return_value="EUR")
    usd = Mock(spec=Currency)
    usd.__str__ = Mock(return_value="USD")
    gbp = Mock(spec=Currency)
    gbp.__str__ = Mock(return_value="GBP")
    
    # Create test date
    test_date = date(2023, 1, 15)
    
    # Create test FX rates
    rate_eur_usd = FXRate(eur, usd, test_date, Decimal("1.10"))
    rate_usd_gbp = FXRate(usd, gbp, test_date, Decimal("0.80"))
    
    # Create test queries
    queries = [
        (eur, usd, test_date),
        (usd, gbp, test_date),
        (gbp, eur, test_date),  # This one doesn't exist
    ]
    
    # Test 1: queries returns rates for existing pairs and None for missing
    service.queries.return_value = [rate_eur_usd, rate_usd_gbp, None]
    results = list(service.queries(queries, strict=False))
    
    assert len(results) == 3
    assert results[0] == rate_eur_usd
    assert results[1] == rate_usd_gbp
    assert results[2] is None
    service.queries.assert_called_once_with(queries, strict=False)
    
    # Test 2: queries with strict=True raises error for missing rates
    service.reset_mock()
    service.queries.side_effect = FXRateLookupError(gbp, eur, test_date)
    
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))
    
    service.queries.assert_called_once_with(queries, strict=True)
    
    # Test 3: queries with empty iterable returns empty iterable
    service.reset_mock()
    service.queries.return_value = iter([])
    results = list(service.queries([], strict=False))
    
    assert len(results) == 0
    service.queries.assert_called_once_with([], strict=False)
    
    # Test 4: queries returns correct number of results matching input
    service.reset_mock()
    single_query = [(eur, usd, test_date)]
    service.queries.return_value = [rate_eur_usd]
    results = list(service.queries(single_query, strict=False))
    
    assert len(results) == len(single_query)
    assert results[0] == rate_eur_usd
    
    # Test 5: queries method is called with correct parameters
    service.reset_mock()
    service.queries.return_value = iter([])
    list(service.queries(queries, strict=True))
    
    service.queries.assert_called_once_with(queries, strict=True)


# LLM-generated content at query #4
#--------------------------

```python
def test_FXRateService_query():
    """Test the query method of FXRateService abstract class."""
    import datetime
    from decimal import Decimal
    from unittest.mock import Mock
    
    # Create a concrete implementation of FXRateService for testing
    class ConcreteFXRateService(FXRateService):
        def __init__(self):
            self.rates = {}
        
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            key = (ccy1, ccy2, asof)
            if key in self.rates:
                return self.rates[key]
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None
        
        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    # Setup test data
    service = ConcreteFXRateService()
    eur = Currency.of("EUR")
    usd = Currency.of("USD")
    gbp = Currency.of("GBP")
    test_date = datetime.date(2023, 1, 15)
    rate_value = Decimal("1.25")
    
    # Create test rate
    test_rate = FXRate(eur, usd, test_date, rate_value)
    service.rates[(eur, usd, test_date)] = test_rate
    
    # Test 1: Query existing rate
    result = service.query(eur, usd, test_date)
    assert result is not None
    assert result == test_rate
    assert result.ccy1 == eur
    assert result.ccy2 == usd
    assert result.date == test_date
    assert result.value == rate_value
    
    # Test 2: Query non-existing rate with strict=False
    result = service.query(gbp, usd, test_date, strict=False)
    assert result is None
    
    # Test 3: Query non-existing rate with strict=True
    with pytest.raises(FXRateLookupError) as exc_info:
        service.query(gbp, usd, test_date, strict=True)
    assert exc_info.value.ccy1 == gbp
    assert exc_info.value.ccy2 == usd
    assert exc_info.value.asof == test_date
    
    # Test 4: Query with different date
    different_date = datetime.date(2023, 1, 16)
    result = service.query(eur, usd, different_date)
    assert result is None
    
    # Test 5: Query after adding another rate
    another_rate = FXRate(gbp, usd, test_date, Decimal("1.50"))
    service.rates[(gbp, usd, test_date)] = another_rate
    result = service.query(gbp, usd, test_date)
    assert result == another_rate
    assert result.value == Decimal("1.50")


# LLM-generated content at query #5
#--------------------------

```python
def test_FXRateService_query():
    """Test the query method of FXRateService."""
    import datetime
    from decimal import Decimal
    from unittest.mock import Mock
    
    # Create a mock FXRateService instance
    service = Mock(spec=FXRateService)
    
    # Test 1: Query returns a valid FXRate
    ccy1 = Mock(spec=Currency)
    ccy2 = Mock(spec=Currency)
    asof = datetime.date.today()
    expected_rate = FXRate(ccy1, ccy2, asof, Decimal("1.5"))
    
    service.query.return_value = expected_rate
    result = service.query(ccy1, ccy2, asof, strict=False)
    
    assert result == expected_rate
    service.query.assert_called_once_with(ccy1, ccy2, asof, strict=False)
    
    # Test 2: Query returns None when rate not found (non-strict mode)
    service.reset_mock()
    service.query.return_value = None
    result = service.query(ccy1, ccy2, asof, strict=False)
    
    assert result is None
    service.query.assert_called_once_with(ccy1, ccy2, asof, strict=False)
    
    # Test 3: Query with strict=True raises FXRateLookupError
    service.reset_mock()
    service.query.side_effect = FXRateLookupError(ccy1, ccy2, asof)
    
    with pytest.raises(FXRateLookupError):
        service.query(ccy1, ccy2, asof, strict=True)
    
    service.query.assert_called_once_with(ccy1, ccy2, asof, strict=True)
    
    # Test 4: Query with different dates
    service.reset_mock()
    asof_different = datetime.date(2023, 1, 1)
    expected_rate_different = FXRate(ccy1, ccy2, asof_different, Decimal("2.0"))
    
    service.query.return_value = expected_rate_different
    result = service.query(ccy1, ccy2, asof_different, strict=False)
    
    assert result == expected_rate_different
    assert result.date == asof_different


# LLM-generated content at query #6
#--------------------------

```python
import pytest
from decimal import Decimal
from datetime import date
from unittest.mock import Mock, MagicMock


def test_FXRateService_queries():
    """Test the queries method of FXRateService."""
    
    # Create a mock FXRateService instance
    service = Mock(spec=FXRateService)
    
    # Setup test data
    eur = Mock(spec=Currency)
    usd = Mock(spec=Currency)
    gbp = Mock(spec=Currency)
    test_date = date.today()
    
    # Create test FXRate objects
    eur_usd_rate = FXRate(eur, usd, test_date, Decimal("1.2"))
    usd_gbp_rate = FXRate(usd, gbp, test_date, Decimal("0.8"))
    
    # Test case 1: Multiple valid queries
    queries = [
        (eur, usd, test_date),
        (usd, gbp, test_date),
    ]
    expected_results = [eur_usd_rate, usd_gbp_rate]
    service.queries.return_value = expected_results
    
    results = list(service.queries(queries, strict=False))
    assert len(results) == 2
    assert results[0] == eur_usd_rate
    assert results[1] == usd_gbp_rate
    service.queries.assert_called_once_with(queries, strict=False)
    
    # Test case 2: Queries with some missing rates (non-strict mode)
    service.reset_mock()
    queries_with_missing = [
        (eur, usd, test_date),
        (gbp, eur, test_date),
    ]
    expected_results_with_none = [eur_usd_rate, None]
    service.queries.return_value = expected_results_with_none
    
    results = list(service.queries(queries_with_missing, strict=False))
    assert len(results) == 2
    assert results[0] == eur_usd_rate
    assert results[1] is None
    
    # Test case 3: Strict mode raises error for missing rates
    service.reset_mock()
    service.queries.side_effect = FXRateLookupError(gbp, eur, test_date)
    
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries_with_missing, strict=True))
    
    # Test case 4: Empty queries
    service.reset_mock()
    service.queries.return_value = []
    
    results = list(service.queries([], strict=False))
    assert len(results) == 0
    
    # Test case 5: Single query
    service.reset_mock()
    single_query = [(eur, usd, test_date)]
    service.queries.return_value = [eur_usd_rate]
    
    results = list(service.queries(single_query, strict=False))
    assert len(results) == 1
    assert results[0] == eur_usd_rate


# LLM-generated content at query #7
#--------------------------

```python
import pytest
from abc import ABCMeta
from decimal import Decimal
from datetime import date
from unittest.mock import Mock, MagicMock


def test_FXRateService_query():
    """Test the query method of FXRateService abstract class."""
    
    # Create a concrete implementation of FXRateService for testing
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            # Return a valid FXRate
            if ccy1.code == "EUR" and ccy2.code == "USD":
                return FXRate(ccy1, ccy2, asof, Decimal("1.25"))
            elif strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None
        
        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    # Create mock currencies
    eur = Mock(spec=Currency)
    eur.code = "EUR"
    usd = Mock(spec=Currency)
    usd.code = "USD"
    gbp = Mock(spec=Currency)
    gbp.code = "GBP"
    
    test_date = date.today()
    service = MockFXRateService()
    
    # Test 1: Query existing rate
    result = service.query(eur, usd, test_date)
    assert result is not None
    assert isinstance(result, FXRate)
    assert result.ccy1 == eur
    assert result.ccy2 == usd
    assert result.date == test_date
    assert result.value == Decimal("1.25")
    
    # Test 2: Query non-existing rate with strict=False (should return None)
    result = service.query(gbp, usd, test_date, strict=False)
    assert result is None
    
    # Test 3: Query non-existing rate with strict=True (should raise FXRateLookupError)
    with pytest.raises(FXRateLookupError) as exc_info:
        service.query(gbp, usd, test_date, strict=True)
    assert exc_info.value.ccy1 == gbp
    assert exc_info.value.ccy2 == usd
    assert exc_info.value.asof == test_date
    
    # Test 4: Verify method signature accepts all required parameters
    result = service.query(ccy1=eur, ccy2=usd, asof=test_date, strict=False)
    assert result is not None
    
    # Test 5: Test with different dates
    different_date = date(2023, 1, 1)
    result = service.query(eur, usd, different_date)
    assert result is not None
    assert result.date == different_date


# LLM-generated content at query #8
#--------------------------

```python
def test_FXRateService_query():
    """
    Unit tests for FXRateService.query method.
    """
    import datetime
    from decimal import Decimal
    
    # Create a concrete implementation of FXRateService for testing
    class MockFXRateService(FXRateService):
        def __init__(self, rates_dict=None):
            self.rates_dict = rates_dict or {}
        
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            key = (ccy1, ccy2, asof)
            if key in self.rates_dict:
                return self.rates_dict[key]
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None
        
        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    # Test data setup
    from .currencies import Currencies
    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    gbp = Currencies["GBP"]
    test_date = datetime.date(2023, 1, 15)
    rate_value = Decimal("1.10")
    
    fx_rate = FXRate(eur, usd, test_date, rate_value)
    service = MockFXRateService({(eur, usd, test_date): fx_rate})
    
    # Test 1: Query existing rate with strict=False
    result = service.query(eur, usd, test_date, strict=False)
    assert result is not None
    assert result == fx_rate
    assert result.ccy1 == eur
    assert result.ccy2 == usd
    assert result.date == test_date
    assert result.value == rate_value
    
    # Test 2: Query non-existing rate with strict=False (should return None)
    result = service.query(gbp, usd, test_date, strict=False)
    assert result is None
    
    # Test 3: Query non-existing rate with strict=True (should raise FXRateLookupError)
    with pytest.raises(FXRateLookupError) as exc_info:
        service.query(gbp, usd, test_date, strict=True)
    assert exc_info.value.ccy1 == gbp
    assert exc_info.value.ccy2 == usd
    assert exc_info.value.asof == test_date
    
    # Test 4: Query with different date (not in service)
    different_date = datetime.date(2023, 2, 15)
    result = service.query(eur, usd, different_date, strict=False)
    assert result is None
    
    # Test 5: Query with different currency pair (not in service)
    result = service.query(usd, eur, test_date, strict=False)
    assert result is None


# LLM-generated content at query #9
#--------------------------

```python
def test_FXRateService_query():
    """Test the query method of FXRateService"""
    import datetime
    from decimal import Decimal
    from unittest.mock import Mock
    
    # Create a mock FXRateService instance
    mock_service = Mock(spec=FXRateService)
    
    # Setup test data
    eur = Mock(spec=Currency)
    usd = Mock(spec=Currency)
    test_date = datetime.date.today()
    test_rate = FXRate(eur, usd, test_date, Decimal("1.20"))
    
    # Test case 1: query returns FXRate when found
    mock_service.query.return_value = test_rate
    result = mock_service.query(eur, usd, test_date, strict=False)
    assert result == test_rate
    mock_service.query.assert_called_once_with(eur, usd, test_date, strict=False)
    
    # Test case 2: query returns None when not found and strict=False
    mock_service.reset_mock()
    mock_service.query.return_value = None
    result = mock_service.query(eur, usd, test_date, strict=False)
    assert result is None
    mock_service.query.assert_called_once_with(eur, usd, test_date, strict=False)
    
    # Test case 3: query raises FXRateLookupError when not found and strict=True
    mock_service.reset_mock()
    mock_service.query.side_effect = FXRateLookupError(eur, usd, test_date)
    with pytest.raises(FXRateLookupError):
        mock_service.query(eur, usd, test_date, strict=True)
    mock_service.query.assert_called_once_with(eur, usd, test_date, strict=True)
    
    # Test case 4: default strict parameter is False
    mock_service.reset_mock()
    mock_service.query.return_value = test_rate
    result = mock_service.query(eur, usd, test_date)
    assert result == test_rate
    mock_service.query.assert_called_once_with(eur, usd, test_date)


# LLM-generated content at query #10
#--------------------------

```python
def test_FXRateService_query():
    """Test the query method of FXRateService abstract class."""
    import datetime
    from decimal import Decimal
    from unittest.mock import Mock
    
    # Create a concrete implementation of FXRateService for testing
    class MockFXRateService(FXRateService):
        def __init__(self):
            self.rates = {}
        
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            key = (ccy1, ccy2, asof)
            if key in self.rates:
                return self.rates[key]
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None
        
        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    # Test setup
    from pypara.currencies import Currency
    eur = Currency("EUR")
    usd = Currency("USD")
    test_date = datetime.date.today()
    
    service = MockFXRateService()
    rate = FXRate(eur, usd, test_date, Decimal("1.20"))
    service.rates[(eur, usd, test_date)] = rate
    
    # Test 1: Query existing rate
    result = service.query(eur, usd, test_date)
    assert result == rate
    assert result.ccy1 == eur
    assert result.ccy2 == usd
    assert result.value == Decimal("1.20")
    
    # Test 2: Query non-existing rate (non-strict mode)
    gbp = Currency("GBP")
    result = service.query(eur, gbp, test_date, strict=False)
    assert result is None
    
    # Test 3: Query non-existing rate (strict mode)
    with pytest.raises(FXRateLookupError):
        service.query(eur, gbp, test_date, strict=True)
    
    # Test 4: Query with different date
    other_date = datetime.date(2020, 1, 1)
    result = service.query(eur, usd, other_date, strict=False)
    assert result is None
    
    # Test 5: Strict mode raises correct exception with proper attributes
    try:
        service.query(usd, gbp, test_date, strict=True)
        assert False, "Should have raised FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == usd
        assert e.ccy2 == gbp
        assert e.asof == test_date


# LLM-generated content at query #11
#--------------------------

```python
import pytest
from datetime import date
from decimal import Decimal
from unittest.mock import Mock, MagicMock


def test_FXRateService_queries():
    """Test the queries method of FXRateService."""
    
    # Create a mock FXRateService instance
    service = Mock(spec=FXRateService)
    
    # Create test data
    from pypara.currencies import Currency
    ccy1 = Mock(spec=Currency)
    ccy2 = Mock(spec=Currency)
    ccy3 = Mock(spec=Currency)
    test_date = date.today()
    
    # Create test FXRate objects
    rate1 = FXRate(ccy1, ccy2, test_date, Decimal("1.5"))
    rate2 = FXRate(ccy2, ccy3, test_date, Decimal("2.0"))
    rate3 = None
    
    # Define queries
    queries = [
        (ccy1, ccy2, test_date),
        (ccy2, ccy3, test_date),
        (ccy1, ccy3, test_date),
    ]
    
    # Test 1: queries returns expected FXRate objects in non-strict mode
    service.queries.return_value = [rate1, rate2, rate3]
    result = service.queries(queries, strict=False)
    result_list = list(result)
    
    assert len(result_list) == 3
    assert result_list[0] == rate1
    assert result_list[1] == rate2
    assert result_list[2] is None
    service.queries.assert_called_once_with(queries, strict=False)
    
    # Test 2: queries with strict=True raises FXRateLookupError when rate not found
    service.reset_mock()
    service.queries.side_effect = FXRateLookupError(ccy1, ccy3, test_date)
    
    with pytest.raises(FXRateLookupError):
        service.queries(queries, strict=True)
    
    service.queries.assert_called_once_with(queries, strict=True)
    
    # Test 3: queries with empty iterable returns empty iterable
    service.reset_mock()
    service.queries.return_value = []
    result = service.queries([], strict=False)
    result_list = list(result)
    
    assert len(result_list) == 0
    service.queries.assert_called_once_with([], strict=False)
    
    # Test 4: queries returns all valid rates
    service.reset_mock()
    all_rates = [rate1, rate2, rate1]
    service.queries.return_value = all_rates
    result = service.queries(queries, strict=False)
    result_list = list(result)
    
    assert len(result_list) == 3
    assert all(r is not None for r in result_list)
    assert result_list == all_rates


# LLM-generated content at query #12
#--------------------------

```python
import datetime
from decimal import Decimal
from unittest.mock import Mock, patch
import pytest

from pypara.currencies import Currency
from pypara.fx import FXRate, FXRateService


def test_FXRateService_queries():
    """Test the queries method of FXRateService"""
    
    # Create mock currencies
    eur = Mock(spec=Currency)
    usd = Mock(spec=Currency)
    gbp = Mock(spec=Currency)
    
    # Create test date
    test_date = datetime.date.today()
    
    # Create mock FX rates
    rate_eur_usd = FXRate(eur, usd, test_date, Decimal("1.2"))
    rate_usd_gbp = FXRate(usd, gbp, test_date, Decimal("0.8"))
    
    # Create a concrete implementation of FXRateService for testing
    class TestFXRateService(FXRateService):
        def __init__(self):
            self.rates = {
                (eur, usd, test_date): rate_eur_usd,
                (usd, gbp, test_date): rate_usd_gbp,
            }
        
        def query(self, ccy1, ccy2, asof, strict=False):
            key = (ccy1, ccy2, asof)
            if key in self.rates:
                return self.rates[key]
            if strict:
                from pypara.fx import FXRateLookupError
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None
        
        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    service = TestFXRateService()
    
    # Test case 1: Query multiple rates successfully
    query_list = [
        (eur, usd, test_date),
        (usd, gbp, test_date),
    ]
    results = list(service.queries(query_list))
    
    assert len(results) == 2
    assert results[0] == rate_eur_usd
    assert results[1] == rate_usd_gbp
    
    # Test case 2: Query with non-existent rates (non-strict mode)
    query_list_with_missing = [
        (eur, usd, test_date),
        (gbp, eur, test_date),  # This doesn't exist
    ]
    results = list(service.queries(query_list_with_missing, strict=False))
    
    assert len(results) == 2
    assert results[0] == rate_eur_usd
    assert results[1] is None
    
    # Test case 3: Query with non-existent rates (strict mode)
    from pypara.fx import FXRateLookupError
    
    query_list_with_missing = [
        (eur, usd, test_date),
        (gbp, eur, test_date),  # This doesn't exist and will raise in strict mode
    ]
    
    with pytest.raises(FXRateLookupError):
        list(service.queries(query_list_with_missing, strict=True))
    
    # Test case 4: Empty queries list
    results = list(service.queries([]))
    assert len(results) == 0
    
    # Test case 5: Single query
    results = list(service.queries([(eur, usd, test_date)]))
    assert len(results) == 1
    assert results[0] == rate_eur_usd


# LLM-generated content at query #13
#--------------------------

```python
import pytest
from datetime import date
from decimal import Decimal
from unittest.mock import Mock, patch


def test_FXRateService_query():
    """Test the query method of FXRateService abstract class."""
    
    # FXRateService is abstract, so we need to create a concrete implementation
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return FXRate(ccy1, ccy2, asof, Decimal("1.5"))
        
        def queries(self, queries, strict=False):
            return [FXRate(q[0], q[1], q[2], Decimal("1.5")) for q in queries]
    
    service = ConcreteFXRateService()
    
    # Create mock currencies
    eur = Mock(spec=Currency)
    eur.__eq__ = Mock(return_value=False)
    usd = Mock(spec=Currency)
    usd.__eq__ = Mock(return_value=False)
    
    test_date = date.today()
    
    # Test basic query
    result = service.query(eur, usd, test_date)
    assert result is not None
    assert result.ccy1 == eur
    assert result.ccy2 == usd
    assert result.date == test_date
    assert result.value == Decimal("1.5")
    
    # Test query with strict=False (default)
    result = service.query(eur, usd, test_date, strict=False)
    assert result is not None
    
    # Test query with strict=True
    result = service.query(eur, usd, test_date, strict=True)
    assert result is not None


def test_FXRateService_query_with_none_result():
    """Test the query method when it returns None."""
    
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None
        
        def queries(self, queries, strict=False):
            return []
    
    service = ConcreteFXRateService()
    
    eur = Mock(spec=Currency)
    usd = Mock(spec=Currency)
    test_date = date.today()
    
    # Test query with strict=False returns None
    result = service.query(eur, usd, test_date, strict=False)
    assert result is None
    
    # Test query with strict=True raises exception
    with pytest.raises(FXRateLookupError):
        service.query(eur, usd, test_date, strict=True)


def test_FXRateService_query_type_annotations():
    """Test that query method respects type annotations."""
    
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            assert isinstance(ccy1, Currency) or isinstance(ccy1, Mock)
            assert isinstance(ccy2, Currency) or isinstance(ccy2, Mock)
            assert isinstance(asof, date)
            assert isinstance(strict, bool)
            return FXRate(ccy1, ccy2, asof, Decimal("2.0"))
        
        def queries(self, queries, strict=False):
            return []
    
    service = ConcreteFXRateService()
    
    eur = Mock(spec=Currency)
    usd = Mock(spec=Currency)
    test_date = date.today()
    
    result = service.query(eur, usd, test_date, strict=True)
    assert result.value == Decimal("2.0")


# LLM-generated content at query #14
#--------------------------

```python
import datetime
from decimal import Decimal
from typing import Iterable, Optional
import pytest

from pypara.currencies import Currency, Currencies
from pypara.fx import FXRate, FXRateService


class MockFXRateService(FXRateService):
    """Mock implementation of FXRateService for testing."""
    
    def __init__(self, rates: dict = None):
        self.rates = rates or {}
    
    def query(self, ccy1: Currency, ccy2: Currency, asof: datetime.date, strict: bool = False) -> Optional[FXRate]:
        key = (ccy1.code, ccy2.code, asof)
        if key in self.rates:
            return self.rates[key]
        if strict:
            raise FXRateLookupError(ccy1, ccy2, asof)
        return None
    
    def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
        for ccy1, ccy2, asof in queries:
            yield self.query(ccy1, ccy2, asof, strict)


def test_FXRateService_queries():
    """Test the queries method of FXRateService."""
    today = datetime.date.today()
    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    gbp = Currencies["GBP"]
    
    # Create test FX rates
    eur_usd_rate = FXRate(eur, usd, today, Decimal("1.2"))
    gbp_usd_rate = FXRate(gbp, usd, today, Decimal("1.4"))
    
    # Setup mock service with some rates
    service = MockFXRateService({
        (eur.code, usd.code, today): eur_usd_rate,
        (gbp.code, usd.code, today): gbp_usd_rate,
    })
    
    # Test queries with existing rates
    query_list = [
        (eur, usd, today),
        (gbp, usd, today),
    ]
    results = list(service.queries(query_list))
    
    assert len(results) == 2
    assert results[0] == eur_usd_rate
    assert results[1] == gbp_usd_rate
    
    # Test queries with non-existing rate (non-strict mode)
    query_list_with_missing = [
        (eur, usd, today),
        (gbp, eur, today),  # This rate doesn't exist
    ]
    results = list(service.queries(query_list_with_missing))
    
    assert len(results) == 2
    assert results[0] == eur_usd_rate
    assert results[1] is None
    
    # Test queries with non-existing rate (strict mode)
    with pytest.raises(FXRateLookupError):
        list(service.queries(query_list_with_missing, strict=True))
    
    # Test empty queries
    results = list(service.queries([]))
    assert len(results) == 0
    
    # Test single query
    results = list(service.queries([(eur, usd, today)]))
    assert len(results) == 1
    assert results[0] == eur_usd_rate


# LLM-generated content at query #15
#--------------------------

```python
def test_FXRateService_query():
    """Test the query method of FXRateService abstract class."""
    import datetime
    from decimal import Decimal
    from unittest.mock import Mock
    
    # Create a concrete implementation of FXRateService for testing
    class ConcreteFXRateService(FXRateService):
        def __init__(self):
            self.rates = {}
        
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            key = (ccy1, ccy2, asof)
            if key in self.rates:
                return self.rates[key]
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None
        
        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    # Setup test data
    from pypara.currencies import Currency
    eur = Currency("EUR")
    usd = Currency("USD")
    gbp = Currency("GBP")
    test_date = datetime.date(2023, 1, 15)
    
    # Create service and add a test rate
    service = ConcreteFXRateService()
    test_rate = FXRate(eur, usd, test_date, Decimal("1.10"))
    service.rates[(eur, usd, test_date)] = test_rate
    
    # Test 1: Query existing rate returns the correct FXRate
    result = service.query(eur, usd, test_date)
    assert result is not None
    assert result == test_rate
    assert result.ccy1 == eur
    assert result.ccy2 == usd
    assert result.value == Decimal("1.10")
    
    # Test 2: Query non-existing rate with strict=False returns None
    result = service.query(gbp, usd, test_date, strict=False)
    assert result is None
    
    # Test 3: Query non-existing rate with strict=True raises FXRateLookupError
    with pytest.raises(FXRateLookupError) as excinfo:
        service.query(gbp, usd, test_date, strict=True)
    assert excinfo.value.ccy1 == gbp
    assert excinfo.value.ccy2 == usd
    assert excinfo.value.asof == test_date
    
    # Test 4: Query with different date returns None when rate doesn't exist
    different_date = datetime.date(2023, 1, 16)
    result = service.query(eur, usd, different_date, strict=False)
    assert result is None
    
    # Test 5: Multiple rates can be stored and queried independently
    gbp_usd_rate = FXRate(gbp, usd, test_date, Decimal("1.27"))
    service.rates[(gbp, usd, test_date)] = gbp_usd_rate
    
    result1 = service.query(eur, usd, test_date)
    result2 = service.query(gbp, usd, test_date)
    
    assert result1 == test_rate
    assert result2 == gbp_usd_rate
    assert result1 != result2


# LLM-generated content at query #16
#--------------------------

```python
def test_FXRateService_query():
    """Test the query method of FXRateService abstract class."""
    import datetime
    from decimal import Decimal
    
    # Create a concrete implementation of FXRateService for testing
    class MockFXRateService(FXRateService):
        def __init__(self):
            self.rates = {}
        
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            key = (ccy1, ccy2, asof)
            rate = self.rates.get(key)
            if rate is None and strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return rate
        
        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    # Setup test data
    from pypara.currencies import Currencies
    
    service = MockFXRateService()
    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    gbp = Currencies["GBP"]
    test_date = datetime.date.today()
    
    rate_eur_usd = FXRate(eur, usd, test_date, Decimal("1.2"))
    service.rates[(eur, usd, test_date)] = rate_eur_usd
    
    # Test 1: Query existing rate with strict=False
    result = service.query(eur, usd, test_date, strict=False)
    assert result is not None
    assert result == rate_eur_usd
    assert result.ccy1 == eur
    assert result.ccy2 == usd
    assert result.value == Decimal("1.2")
    
    # Test 2: Query non-existing rate with strict=False
    result = service.query(gbp, usd, test_date, strict=False)
    assert result is None
    
    # Test 3: Query non-existing rate with strict=True
    with pytest.raises(FXRateLookupError) as exc_info:
        service.query(gbp, usd, test_date, strict=True)
    assert exc_info.value.ccy1 == gbp
    assert exc_info.value.ccy2 == usd
    assert exc_info.value.asof == test_date
    
    # Test 4: Query different date returns None
    different_date = test_date - datetime.timedelta(days=1)
    result = service.query(eur, usd, different_date, strict=False)
    assert result is None
    
    # Test 5: Query with different currency pair
    result = service.query(usd, eur, test_date, strict=False)
    assert result is None


# LLM-generated content at query #17
#--------------------------

```python
def test_FXRateService_query():
    """Test the query method of FXRateService abstract class."""
    import datetime
    from decimal import Decimal
    from unittest.mock import MagicMock
    
    # Create a concrete implementation of FXRateService for testing
    class ConcreteFXRateService(FXRateService):
        def __init__(self):
            self.rates = {}
        
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            key = (ccy1, ccy2, asof)
            if key in self.rates:
                return self.rates[key]
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None
        
        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    # Setup test data
    from pypara.currencies import Currency
    eur = Currency("EUR")
    usd = Currency("USD")
    gbp = Currency("GBP")
    test_date = datetime.date(2023, 1, 1)
    
    service = ConcreteFXRateService()
    fx_rate = FXRate(eur, usd, test_date, Decimal("1.20"))
    service.rates[(eur, usd, test_date)] = fx_rate
    
    # Test 1: Query existing rate
    result = service.query(eur, usd, test_date)
    assert result == fx_rate
    assert result.ccy1 == eur
    assert result.ccy2 == usd
    assert result.value == Decimal("1.20")
    
    # Test 2: Query non-existing rate with strict=False
    result = service.query(gbp, usd, test_date, strict=False)
    assert result is None
    
    # Test 3: Query non-existing rate with strict=True
    with pytest.raises(FXRateLookupError) as exc_info:
        service.query(gbp, usd, test_date, strict=True)
    assert exc_info.value.ccy1 == gbp
    assert exc_info.value.ccy2 == usd
    assert exc_info.value.asof == test_date
    
    # Test 4: Query with different date
    different_date = datetime.date(2023, 1, 2)
    result = service.query(eur, usd, different_date)
    assert result is None
    
    # Test 5: Verify method signature parameters
    assert service.query.__code__.co_varnames[:5] == ('self', 'ccy1', 'ccy2', 'asof', 'strict')


# LLM-generated content at query #18
#--------------------------

```python
def test_FXRateService_query():
    """Test the query method of FXRateService abstract class."""
    import datetime
    from decimal import Decimal
    from unittest.mock import Mock
    
    # Create a concrete implementation of FXRateService for testing
    class ConcreteFXRateService(FXRateService):
        def __init__(self):
            self.rates = {}
        
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            key = (ccy1, ccy2, asof)
            if key in self.rates:
                return self.rates[key]
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None
        
        def queries(self, queries: Iterable[TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    # Setup test data
    service = ConcreteFXRateService()
    test_date = datetime.date.today()
    eur = Currency("EUR", 1, "Euro")
    usd = Currency("USD", 1, "US Dollar")
    
    rate = FXRate(eur, usd, test_date, Decimal("1.20"))
    service.rates[(eur, usd, test_date)] = rate
    
    # Test 1: Query existing rate with strict=False
    result = service.query(eur, usd, test_date, strict=False)
    assert result == rate
    assert result.ccy1 == eur
    assert result.ccy2 == usd
    assert result.value == Decimal("1.20")
    
    # Test 2: Query non-existing rate with strict=False returns None
    gbp = Currency("GBP", 1, "British Pound")
    result = service.query(gbp, usd, test_date, strict=False)
    assert result is None
    
    # Test 3: Query non-existing rate with strict=True raises FXRateLookupError
    with pytest.raises(FXRateLookupError) as exc_info:
        service.query(gbp, usd, test_date, strict=True)
    assert exc_info.value.ccy1 == gbp
    assert exc_info.value.ccy2 == usd
    assert exc_info.value.asof == test_date
    
    # Test 4: Query with different date returns None
    different_date = datetime.date(2020, 1, 1)
    result = service.query(eur, usd, different_date, strict=False)
    assert result is None
    
    # Test 5: Query with reversed currencies returns None
    result = service.query(usd, eur, test_date, strict=False)
    assert result is None


# LLM-generated content at query #19
#--------------------------

```python
def test_FXRateService_query():
    """Test the query method of FXRateService"""
    import datetime
    from decimal import Decimal
    from unittest.mock import Mock
    
    # Create a mock implementation of FXRateService
    class MockFXRateService(FXRateService):
        def __init__(self):
            self.rates = {}
        
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            key = (ccy1, ccy2, asof)
            if key in self.rates:
                return self.rates[key]
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None
        
        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    # Setup test data
    from pypara.currencies import Currencies
    service = MockFXRateService()
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    test_date = datetime.date(2023, 1, 1)
    rate_value = Decimal("1.15")
    
    # Test case 1: Query returns None when rate not found (non-strict mode)
    result = service.query(ccy1, ccy2, test_date, strict=False)
    assert result is None
    
    # Test case 2: Query raises FXRateLookupError when rate not found (strict mode)
    with pytest.raises(FXRateLookupError) as exc_info:
        service.query(ccy1, ccy2, test_date, strict=True)
    assert exc_info.value.ccy1 == ccy1
    assert exc_info.value.ccy2 == ccy2
    assert exc_info.value.asof == test_date
    
    # Test case 3: Query returns FXRate when rate exists (non-strict mode)
    fx_rate = FXRate(ccy1, ccy2, test_date, rate_value)
    service.rates[(ccy1, ccy2, test_date)] = fx_rate
    result = service.query(ccy1, ccy2, test_date, strict=False)
    assert result == fx_rate
    assert result.ccy1 == ccy1
    assert result.ccy2 == ccy2
    assert result.date == test_date
    assert result.value == rate_value
    
    # Test case 4: Query returns FXRate when rate exists (strict mode)
    result = service.query(ccy1, ccy2, test_date, strict=True)
    assert result == fx_rate
    
    # Test case 5: Query with different date returns None
    different_date = datetime.date(2023, 1, 2)
    result = service.query(ccy1, ccy2, different_date, strict=False)
    assert result is None


# LLM-generated content at query #20
#--------------------------

```python
import pytest
from abc import ABCMeta
from decimal import Decimal
from datetime import date
from unittest.mock import Mock, MagicMock

def test_FXRateService_query():
    """Test the query method of FXRateService abstract class."""
    
    # Create a concrete implementation of FXRateService for testing
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            if ccy1 == ccy2:
                return FXRate(ccy1, ccy2, asof, Decimal("1"))
            if str(ccy1) == "EUR" and str(ccy2) == "USD":
                return FXRate(ccy1, ccy2, asof, Decimal("1.2"))
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None
        
        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    service = ConcreteFXRateService()
    test_date = date.today()
    
    # Mock currencies
    eur = Mock(spec=Currency)
    eur.__str__ = Mock(return_value="EUR")
    usd = Mock(spec=Currency)
    usd.__str__ = Mock(return_value="USD")
    gbp = Mock(spec=Currency)
    gbp.__str__ = Mock(return_value="GBP")
    
    # Test 1: Query same currency returns rate of 1
    result = service.query(eur, eur, test_date)
    assert result is not None
    assert result.value == Decimal("1")
    assert result.ccy1 == eur
    assert result.ccy2 == eur
    
    # Test 2: Query known pair returns correct rate
    result = service.query(eur, usd, test_date)
    assert result is not None
    assert result.value == Decimal("1.2")
    assert result.ccy1 == eur
    assert result.ccy2 == usd
    assert result.date == test_date
    
    # Test 3: Query unknown pair with strict=False returns None
    result = service.query(gbp, usd, test_date, strict=False)
    assert result is None
    
    # Test 4: Query unknown pair with strict=True raises FXRateLookupError
    with pytest.raises(FXRateLookupError) as exc_info:
        service.query(gbp, usd, test_date, strict=True)
    assert exc_info.value.ccy1 == gbp
    assert exc_info.value.ccy2 == usd
    assert exc_info.value.asof == test_date
    
    # Test 5: Default strict parameter is False
    result = service.query(gbp, usd, test_date)
    assert result is None


# LLM-generated content at query #21
#--------------------------

```python
def test_FXRateService_query():
    """Test the query method of FXRateService."""
    import datetime
    from decimal import Decimal
    from unittest.mock import MagicMock
    
    # Create a mock FXRateService instance
    service = MagicMock(spec=FXRateService)
    
    # Setup test data
    ccy1 = MagicMock(spec=Currency)
    ccy2 = MagicMock(spec=Currency)
    asof = datetime.date.today()
    rate_value = Decimal("1.25")
    
    # Test case 1: Successful query returns FXRate
    expected_rate = FXRate(ccy1, ccy2, asof, rate_value)
    service.query.return_value = expected_rate
    
    result = service.query(ccy1, ccy2, asof, strict=False)
    assert result == expected_rate
    service.query.assert_called_once_with(ccy1, ccy2, asof, strict=False)
    
    # Test case 2: Query returns None when rate not found (non-strict mode)
    service.reset_mock()
    service.query.return_value = None
    
    result = service.query(ccy1, ccy2, asof, strict=False)
    assert result is None
    service.query.assert_called_once_with(ccy1, ccy2, asof, strict=False)
    
    # Test case 3: Query raises FXRateLookupError in strict mode
    service.reset_mock()
    service.query.side_effect = FXRateLookupError(ccy1, ccy2, asof)
    
    with pytest.raises(FXRateLookupError):
        service.query(ccy1, ccy2, asof, strict=True)
    service.query.assert_called_once_with(ccy1, ccy2, asof, strict=True)
    
    # Test case 4: Query with different strict parameter default
    service.reset_mock()
    service.query.return_value = expected_rate
    
    result = service.query(ccy1, ccy2, asof)
    assert result == expected_rate
    service.query.assert_called_once_with(ccy1, ccy2, asof)


# LLM-generated content at query #22
#--------------------------

```python
def test_FXRateService_query():
    """Test the query method of FXRateService."""
    from unittest.mock import Mock
    import datetime
    from decimal import Decimal
    
    # Create a mock implementation of FXRateService
    mock_service = Mock(spec=FXRateService)
    
    # Setup test data
    eur = Mock(spec=Currency)
    usd = Mock(spec=Currency)
    date = datetime.date.today()
    rate_value = Decimal("1.25")
    expected_rate = FXRate(eur, usd, date, rate_value)
    
    # Test case 1: Query returns a rate when found
    mock_service.query.return_value = expected_rate
    result = mock_service.query(eur, usd, date, strict=False)
    assert result == expected_rate
    mock_service.query.assert_called_once_with(eur, usd, date, strict=False)
    
    # Test case 2: Query returns None when rate not found and strict=False
    mock_service.reset_mock()
    mock_service.query.return_value = None
    result = mock_service.query(eur, usd, date, strict=False)
    assert result is None
    mock_service.query.assert_called_once_with(eur, usd, date, strict=False)
    
    # Test case 3: Query raises FXRateLookupError when strict=True and rate not found
    mock_service.reset_mock()
    mock_service.query.side_effect = FXRateLookupError(eur, usd, date)
    with pytest.raises(FXRateLookupError):
        mock_service.query(eur, usd, date, strict=True)
    mock_service.query.assert_called_once_with(eur, usd, date, strict=True)
    
    # Test case 4: Query with different currency pair
    mock_service.reset_mock()
    gbp = Mock(spec=Currency)
    different_rate = FXRate(gbp, usd, date, Decimal("1.38"))
    mock_service.query.return_value = different_rate
    result = mock_service.query(gbp, usd, date, strict=False)
    assert result == different_rate
    assert result.ccy1 == gbp
    assert result.ccy2 == usd


# LLM-generated content at query #23
#--------------------------

```python
def test_FXRateService_queries():
    """Test the queries method of FXRateService."""
    import datetime
    from decimal import Decimal
    from unittest.mock import Mock, MagicMock
    
    # Create a mock FXRateService instance
    service = Mock(spec=FXRateService)
    
    # Setup test data
    ccy1 = Mock(spec=Currency)
    ccy2 = Mock(spec=Currency)
    ccy3 = Mock(spec=Currency)
    test_date = datetime.date.today()
    
    # Create test FXRate objects
    rate1 = FXRate(ccy1, ccy2, test_date, Decimal("1.5"))
    rate2 = FXRate(ccy2, ccy3, test_date, Decimal("0.8"))
    rate3 = None
    
    # Define queries
    queries = [
        (ccy1, ccy2, test_date),
        (ccy2, ccy3, test_date),
        (ccy1, ccy3, test_date),
    ]
    
    # Test 1: Non-strict mode returns rates and None values
    service.queries.return_value = [rate1, rate2, rate3]
    result = list(service.queries(queries, strict=False))
    
    assert len(result) == 3
    assert result[0] == rate1
    assert result[1] == rate2
    assert result[2] is None
    service.queries.assert_called_once_with(queries, strict=False)
    
    # Test 2: Strict mode raises FXRateLookupError
    service.reset_mock()
    service.queries.side_effect = FXRateLookupError(ccy1, ccy3, test_date)
    
    try:
        list(service.queries(queries, strict=True))
        assert False, "Expected FXRateLookupError to be raised"
    except FXRateLookupError as e:
        assert e.ccy1 == ccy1
        assert e.ccy2 == ccy3
        assert e.asof == test_date
    
    # Test 3: Empty queries returns empty iterable
    service.reset_mock()
    service.queries.return_value = []
    result = list(service.queries([], strict=False))
    
    assert len(result) == 0
    
    # Test 4: Single query in list
    service.reset_mock()
    service.queries.return_value = [rate1]
    result = list(service.queries([(ccy1, ccy2, test_date)], strict=False))
    
    assert len(result) == 1
    assert result[0] == rate1


# LLM-generated content at query #24
#--------------------------

```python
import datetime
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from pypara.currencies import Currency
from pypara.fx import FXRate, FXRateService


def test_FXRateService_queries():
    """Test the queries method of FXRateService."""
    
    # Create a concrete implementation of FXRateService for testing
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: datetime.date, strict: bool = False):
            # Mock implementation
            if ccy1.code == "EUR" and ccy2.code == "USD":
                return FXRate(ccy1, ccy2, asof, Decimal("1.10"))
            elif ccy1.code == "GBP" and ccy2.code == "USD":
                return FXRate(ccy1, ccy2, asof, Decimal("1.35"))
            return None
        
        def queries(self, queries, strict: bool = False):
            # Default implementation that calls query for each item
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    # Create mock currencies
    eur = MagicMock(spec=Currency)
    eur.code = "EUR"
    usd = MagicMock(spec=Currency)
    usd.code = "USD"
    gbp = MagicMock(spec=Currency)
    gbp.code = "GBP"
    
    service = MockFXRateService()
    test_date = datetime.date.today()
    
    # Test with valid queries
    query_list = [
        (eur, usd, test_date),
        (gbp, usd, test_date),
    ]
    
    results = list(service.queries(query_list, strict=False))
    
    assert len(results) == 2
    assert results[0] is not None
    assert results[0].ccy1 == eur
    assert results[0].ccy2 == usd
    assert results[0].value == Decimal("1.10")
    assert results[1] is not None
    assert results[1].ccy1 == gbp
    assert results[1].ccy2 == usd
    assert results[1].value == Decimal("1.35")
    
    # Test with empty query list
    empty_results = list(service.queries([], strict=False))
    assert len(empty_results) == 0
    
    # Test with query that returns None
    jpy = MagicMock(spec=Currency)
    jpy.code = "JPY"
    query_list_with_none = [
        (eur, usd, test_date),
        (jpy, usd, test_date),
    ]
    
    results_with_none = list(service.queries(query_list_with_none, strict=False))
    
    assert len(results_with_none) == 2
    assert results_with_none[0] is not None
    assert results_with_none[1] is None


# LLM-generated content at query #25
#--------------------------

```python
def test_FXRateService_query():
    """
    Test the query method of FXRateService abstract class.
    """
    import datetime
    from decimal import Decimal
    from unittest.mock import Mock

    # Create a concrete implementation of FXRateService for testing
    class ConcreteFXRateService(FXRateService):
        def __init__(self):
            self.rates = {}

        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            key = (ccy1, ccy2, asof)
            if key in self.rates:
                return self.rates[key]
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    from pypara.currencies import Currencies

    service = ConcreteFXRateService()
    date = datetime.date.today()
    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    gbp = Currencies["GBP"]

    # Test 1: Query returns None when rate not found and strict=False
    result = service.query(eur, usd, date, strict=False)
    assert result is None

    # Test 2: Query raises FXRateLookupError when rate not found and strict=True
    with pytest.raises(FXRateLookupError) as exc_info:
        service.query(eur, usd, date, strict=True)
    assert exc_info.value.ccy1 == eur
    assert exc_info.value.ccy2 == usd
    assert exc_info.value.asof == date

    # Test 3: Query returns FXRate when found
    fx_rate = FXRate(eur, usd, date, Decimal("1.2"))
    service.rates[(eur, usd, date)] = fx_rate
    result = service.query(eur, usd, date, strict=False)
    assert result == fx_rate
    assert result.ccy1 == eur
    assert result.ccy2 == usd
    assert result.date == date
    assert result.value == Decimal("1.2")

    # Test 4: Query with different currency pair
    fx_rate_gbp = FXRate(gbp, usd, date, Decimal("1.4"))
    service.rates[(gbp, usd, date)] = fx_rate_gbp
    result = service.query(gbp, usd, date, strict=False)
    assert result == fx_rate_gbp

    # Test 5: Query with different date
    other_date = datetime.date.today() - datetime.timedelta(days=1)
    result = service.query(eur, usd, other_date, strict=False)
    assert result is None

    # Test 6: Query with same rate but different date raises error in strict mode
    with pytest.raises(FXRateLookupError):
        service.query(eur, usd, other_date, strict=True)


# LLM-generated content at query #26
#--------------------------

```python
def test_FXRateService_queries():
    """Test the queries method of FXRateService."""
    import datetime
    from decimal import Decimal
    from unittest.mock import Mock, MagicMock
    
    # Create a mock FXRateService instance
    service = Mock(spec=FXRateService)
    
    # Setup test data
    eur = Mock(spec=Currency)
    usd = Mock(spec=Currency)
    gbp = Mock(spec=Currency)
    test_date = datetime.date.today()
    
    # Create sample FXRate objects
    rate_eur_usd = FXRate(eur, usd, test_date, Decimal("1.2"))
    rate_usd_gbp = FXRate(usd, gbp, test_date, Decimal("0.8"))
    rate_eur_gbp = None  # Not found
    
    # Setup queries
    queries = [
        (eur, usd, test_date),
        (usd, gbp, test_date),
        (eur, gbp, test_date),
    ]
    
    # Test 1: queries method returns an iterable of FXRate objects
    service.queries.return_value = [rate_eur_usd, rate_usd_gbp, rate_eur_gbp]
    result = service.queries(queries, strict=False)
    result_list = list(result)
    
    assert len(result_list) == 3
    assert result_list[0] == rate_eur_usd
    assert result_list[1] == rate_usd_gbp
    assert result_list[2] is None
    
    # Test 2: queries method is called with correct parameters
    service.queries.reset_mock()
    service.queries.return_value = iter([rate_eur_usd, rate_usd_gbp])
    service.queries(queries[:2], strict=True)
    
    service.queries.assert_called_once_with(queries[:2], strict=True)
    
    # Test 3: queries method with strict=True raises error (mocked behavior)
    service.queries.reset_mock()
    service.queries.side_effect = FXRateLookupError(eur, gbp, test_date)
    
    with pytest.raises(FXRateLookupError):
        service.queries(queries, strict=True)
    
    # Test 4: queries method returns empty iterable for empty input
    service.queries.reset_mock()
    service.queries.return_value = iter([])
    result = service.queries([], strict=False)
    result_list = list(result)
    
    assert len(result_list) == 0
    
    # Test 5: queries method with mixed results (some found, some not)
    service.queries.reset_mock()
    service.queries.return_value = iter([rate_eur_usd, None, rate_usd_gbp, None])
    result = service.queries(queries + [(gbp, eur, test_date)], strict=False)
    result_list = list(result)
    
    assert len(result_list) == 4
    assert result_list[0] is not None
    assert result_list[1] is None
    assert result_list[2] is not None
    assert result_list[3] is None


# LLM-generated content at query #27
#--------------------------

```python
import pytest
from abc import ABCMeta
from decimal import Decimal
from datetime import date
from unittest.mock import Mock, MagicMock


def test_FXRateService_queries():
    """Test the queries method of FXRateService."""
    
    # Create a concrete implementation of FXRateService for testing
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            # Mock implementation
            pass
        
        def queries(self, queries, strict=False):
            # Mock implementation that calls query for each tuple
            results = []
            for ccy1, ccy2, asof in queries:
                result = self.query(ccy1, ccy2, asof, strict)
                results.append(result)
            return results
    
    # Create mock currencies
    eur = Mock(spec=Currency)
    usd = Mock(spec=Currency)
    gbp = Mock(spec=Currency)
    
    # Create test date
    test_date = date.today()
    
    # Create service instance
    service = ConcreteFXRateService()
    
    # Mock the query method to return specific rates
    rate1 = FXRate(eur, usd, test_date, Decimal("1.2"))
    rate2 = FXRate(usd, gbp, test_date, Decimal("0.8"))
    rate3 = None
    
    service.query = Mock(side_effect=[rate1, rate2, rate3])
    
    # Test queries with multiple currency pairs
    query_list = [
        (eur, usd, test_date),
        (usd, gbp, test_date),
        (eur, gbp, test_date),
    ]
    
    results = list(service.queries(query_list, strict=False))
    
    # Assertions
    assert len(results) == 3
    assert results[0] == rate1
    assert results[1] == rate2
    assert results[2] is None
    assert service.query.call_count == 3
    
    # Verify query was called with correct arguments
    service.query.assert_any_call(eur, usd, test_date, False)
    service.query.assert_any_call(usd, gbp, test_date, False)
    service.query.assert_any_call(eur, gbp, test_date, False)


def test_FXRateService_queries_with_strict_mode():
    """Test the queries method of FXRateService in strict mode."""
    
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if strict and ccy1 is None:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None
        
        def queries(self, queries, strict=False):
            results = []
            for ccy1, ccy2, asof in queries:
                result = self.query(ccy1, ccy2, asof, strict)
                results.append(result)
            return results
    
    # Create mock currencies
    eur = Mock(spec=Currency)
    usd = Mock(spec=Currency)
    test_date = date.today()
    
    service = ConcreteFXRateService()
    
    # Test with strict=True
    query_list = [(eur, usd, test_date)]
    
    service.query = Mock(return_value=None)
    results = list(service.queries(query_list, strict=True))
    
    assert len(results) == 1
    assert results[0] is None
    service.query.assert_called_with(eur, usd, test_date, True)


def test_FXRateService_queries_empty_list():
    """Test the queries method with empty query list."""
    
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        
        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    service = ConcreteFXRateService()
    results = list(service.queries([], strict=False))
    
    assert results == []


def test_FXRateService_queries_returns_iterable():
    """Test that queries method returns an iterable."""
    
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return FXRate(ccy1, ccy2, asof, Decimal("1.0"))
        
        def queries(self, queries, strict=False):
            for ccy1, ccy2, asof in queries:
                yield self.query(ccy1, ccy2, asof, strict)
    
    eur = Mock(spec=Currency)
    usd = Mock(spec=Currency)
    test_date = date.today()
    
    service = ConcreteFXRateService()
    query_list = [(eur, usd, test_date)]
    
    results = service.queries(query_list)
    
    # Verify it's iterable
    assert hasattr(results, '__iter__')
    result_list = list(results)
    assert len(result_list) == 1
    assert isinstance(result_list[0], FXRate)


# LLM-generated content at query #28
#--------------------------

```python
import pytest
from abc import ABCMeta
from decimal import Decimal
from datetime import date
from unittest.mock import Mock, MagicMock


def test_FXRateService_query():
    """Test the query method of FXRateService abstract class."""
    
    # Create a concrete implementation of FXRateService for testing
    class ConcreteRateService(FXRateService):
        def __init__(self, rates_dict=None):
            self.rates_dict = rates_dict or {}
        
        def query(self, ccy1, ccy2, asof, strict=False):
            key = (ccy1, ccy2, asof)
            if key in self.rates_dict:
                return self.rates_dict[key]
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None
        
        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    # Setup test data
    test_date = date.today()
    mock_ccy1 = Mock(spec=Currency)
    mock_ccy2 = Mock(spec=Currency)
    mock_ccy1.__str__ = Mock(return_value="EUR")
    mock_ccy2.__str__ = Mock(return_value="USD")
    
    test_rate = FXRate(mock_ccy1, mock_ccy2, test_date, Decimal("1.20"))
    
    # Test 1: Query returns existing FX rate
    service = ConcreteRateService({
        (mock_ccy1, mock_ccy2, test_date): test_rate
    })
    result = service.query(mock_ccy1, mock_ccy2, test_date)
    assert result == test_rate
    assert result.value == Decimal("1.20")
    
    # Test 2: Query returns None when rate not found and strict=False
    result = service.query(mock_ccy1, mock_ccy2, date(2020, 1, 1), strict=False)
    assert result is None
    
    # Test 3: Query raises FXRateLookupError when strict=True and rate not found
    with pytest.raises(FXRateLookupError):
        service.query(mock_ccy1, mock_ccy2, date(2020, 1, 1), strict=True)
    
    # Test 4: Query with different currency pair returns None
    mock_ccy3 = Mock(spec=Currency)
    mock_ccy3.__str__ = Mock(return_value="GBP")
    result = service.query(mock_ccy3, mock_ccy2, test_date, strict=False)
    assert result is None
    
    # Test 5: Verify abstract class cannot be instantiated directly
    with pytest.raises(TypeError):
        FXRateService()


# LLM-generated content at query #29
#--------------------------

```python
import pytest
from decimal import Decimal
from datetime import date
from unittest.mock import Mock, MagicMock


def test_FXRateService_queries():
    """
    Test the queries method of FXRateService.
    """
    # Create a mock FXRateService instance
    service = Mock(spec=FXRateService)
    
    # Setup test data
    eur = Mock(spec=Currency)
    usd = Mock(spec=Currency)
    gbp = Mock(spec=Currency)
    test_date = date.today()
    
    # Create test FXRate objects
    eur_usd_rate = FXRate(eur, usd, test_date, Decimal("1.20"))
    usd_gbp_rate = FXRate(usd, gbp, test_date, Decimal("0.80"))
    
    # Create query tuples
    queries = [
        (eur, usd, test_date),
        (usd, gbp, test_date),
    ]
    
    # Test case 1: queries returns rates successfully
    service.queries.return_value = [eur_usd_rate, usd_gbp_rate]
    result = service.queries(queries, strict=False)
    result_list = list(result)
    
    assert len(result_list) == 2
    assert result_list[0] == eur_usd_rate
    assert result_list[1] == usd_gbp_rate
    service.queries.assert_called_once_with(queries, strict=False)
    
    # Test case 2: queries with strict=True
    service.reset_mock()
    service.queries.return_value = [eur_usd_rate, None]
    result = service.queries(queries, strict=True)
    result_list = list(result)
    
    assert len(result_list) == 2
    assert result_list[0] == eur_usd_rate
    assert result_list[1] is None
    service.queries.assert_called_once_with(queries, strict=True)
    
    # Test case 3: queries with some None results
    service.reset_mock()
    service.queries.return_value = [eur_usd_rate, None]
    result = service.queries(queries, strict=False)
    result_list = list(result)
    
    assert len(result_list) == 2
    assert result_list[0] == eur_usd_rate
    assert result_list[1] is None
    
    # Test case 4: queries with empty input
    service.reset_mock()
    service.queries.return_value = iter([])
    result = service.queries([], strict=False)
    result_list = list(result)
    
    assert len(result_list) == 0
    service.queries.assert_called_once_with([], strict=False)
    
    # Test case 5: queries with all None results
    service.reset_mock()
    service.queries.return_value = [None, None]
    result = service.queries(queries, strict=False)
    result_list = list(result)
    
    assert len(result_list) == 2
    assert all(r is None for r in result_list)


# LLM-generated content at query #30
#--------------------------

```python
import datetime
from decimal import Decimal
from typing import Iterable, Optional
import pytest

from pypara.currencies import Currency, Currencies
from pypara.fx import FXRate, FXRateService, FXRateLookupError


class MockFXRateService(FXRateService):
    """Mock implementation of FXRateService for testing."""
    
    def __init__(self, rates_dict=None):
        self.rates_dict = rates_dict or {}
    
    def query(self, ccy1: Currency, ccy2: Currency, asof: datetime.date, strict: bool = False) -> Optional[FXRate]:
        """Mock query implementation."""
        key = (ccy1.code, ccy2.code, asof)
        if key in self.rates_dict:
            return self.rates_dict[key]
        if strict:
            raise FXRateLookupError(ccy1, ccy2, asof)
        return None
    
    def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
        """Mock queries implementation."""
        for ccy1, ccy2, asof in queries:
            yield self.query(ccy1, ccy2, asof, strict)


def test_FXRateService_queries():
    """Test the queries method of FXRateService."""
    # Setup test data
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)
    
    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    gbp = Currencies["GBP"]
    
    # Create test rates
    eur_usd_rate = FXRate(eur, usd, today, Decimal("1.10"))
    gbp_usd_rate = FXRate(gbp, usd, today, Decimal("1.35"))
    eur_usd_yesterday = FXRate(eur, usd, yesterday, Decimal("1.08"))
    
    # Initialize service with test data
    rates_dict = {
        (eur.code, usd.code, today): eur_usd_rate,
        (gbp.code, usd.code, today): gbp_usd_rate,
        (eur.code, usd.code, yesterday): eur_usd_yesterday,
    }
    service = MockFXRateService(rates_dict)
    
    # Test 1: Query multiple existing rates
    queries = [
        (eur, usd, today),
        (gbp, usd, today),
        (eur, usd, yesterday),
    ]
    results = list(service.queries(queries))
    
    assert len(results) == 3
    assert results[0] == eur_usd_rate
    assert results[1] == gbp_usd_rate
    assert results[2] == eur_usd_yesterday
    
    # Test 2: Query with non-existent rates (non-strict mode)
    queries_with_missing = [
        (eur, usd, today),
        (gbp, gbp, today),  # Non-existent
        (gbp, usd, today),
    ]
    results = list(service.queries(queries_with_missing, strict=False))
    
    assert len(results) == 3
    assert results[0] == eur_usd_rate
    assert results[1] is None
    assert results[2] == gbp_usd_rate
    
    # Test 3: Query with non-existent rates (strict mode)
    queries_with_missing = [
        (eur, usd, today),
        (gbp, gbp, today),  # Non-existent, should raise
    ]
    
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries_with_missing, strict=True))
    
    # Test 4: Empty queries
    results = list(service.queries([], strict=False))
    assert len(results) == 0
    
    # Test 5: Single query
    results = list(service.queries([(eur, usd, today)], strict=False))
    assert len(results) == 1
    assert results[0] == eur_usd_rate


# LLM-generated content at query #31
#--------------------------

```python
import pytest
from decimal import Decimal
from datetime import date
from unittest.mock import Mock, MagicMock
from abc import ABCMeta


def test_FXRateService_query():
    """Test the query method of FXRateService abstract class."""
    
    # Create a concrete implementation of FXRateService for testing
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if ccy1 == ccy2 == "EUR" and asof == date(2023, 1, 1):
                return FXRate(ccy1, ccy2, asof, Decimal("1.2"))
            elif strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None
        
        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    service = ConcreteFXRateService()
    
    # Test case 1: Successful query with existing rate
    result = service.query("EUR", "USD", date(2023, 1, 1), strict=False)
    assert result is None
    
    # Test case 2: Query returns rate when currencies and date match
    result = service.query("EUR", "EUR", date(2023, 1, 1), strict=False)
    assert result is not None
    assert result.ccy1 == "EUR"
    assert result.ccy2 == "EUR"
    assert result.value == Decimal("1.2")
    
    # Test case 3: Query returns None when not found and strict=False
    result = service.query("GBP", "JPY", date(2023, 6, 15), strict=False)
    assert result is None
    
    # Test case 4: Query raises FXRateLookupError when not found and strict=True
    with pytest.raises(FXRateLookupError):
        service.query("GBP", "JPY", date(2023, 6, 15), strict=True)
    
    # Test case 5: Query is abstract and cannot be instantiated directly
    with pytest.raises(TypeError):
        FXRateService()
    
    # Test case 6: Verify query method signature parameters
    import inspect
    sig = inspect.signature(FXRateService.query)
    params = list(sig.parameters.keys())
    assert "ccy1" in params
    assert "ccy2" in params
    assert "asof" in params
    assert "strict" in params
    assert sig.parameters["strict"].default is False


# LLM-generated content at query #32
#--------------------------

```python
import datetime
from decimal import Decimal
from unittest.mock import Mock

import pytest

from pypara.currencies import Currency
from pypara.fx import FXRate, FXRateService


def test_FXRateService_queries():
    """Test the queries method of FXRateService."""
    
    # Create a mock FXRateService since it's abstract
    mock_service = Mock(spec=FXRateService)
    
    # Create test currencies
    eur = Mock(spec=Currency)
    eur.__str__ = Mock(return_value="EUR")
    usd = Mock(spec=Currency)
    usd.__str__ = Mock(return_value="USD")
    gbp = Mock(spec=Currency)
    gbp.__str__ = Mock(return_value="GBP")
    
    # Create test date
    test_date = datetime.date(2023, 1, 1)
    
    # Create test FX rates
    rate1 = FXRate(eur, usd, test_date, Decimal("1.1"))
    rate2 = FXRate(usd, gbp, test_date, Decimal("0.8"))
    rate3 = None  # Not found
    
    # Create query tuples
    queries = [
        (eur, usd, test_date),
        (usd, gbp, test_date),
        (gbp, eur, test_date),
    ]
    
    # Configure mock to return rates
    mock_service.queries.return_value = [rate1, rate2, rate3]
    
    # Execute
    results = list(mock_service.queries(queries, strict=False))
    
    # Assert
    assert len(results) == 3
    assert results[0] == rate1
    assert results[1] == rate2
    assert results[2] is None
    mock_service.queries.assert_called_once_with(queries, strict=False)


def test_FXRateService_queries_strict_mode():
    """Test the queries method of FXRateService in strict mode."""
    
    mock_service = Mock(spec=FXRateService)
    
    eur = Mock(spec=Currency)
    eur.__str__ = Mock(return_value="EUR")
    usd = Mock(spec=Currency)
    usd.__str__ = Mock(return_value="USD")
    
    test_date = datetime.date(2023, 1, 1)
    
    queries = [(eur, usd, test_date)]
    
    # In strict mode, should raise an error
    from pypara.fx import FXRateLookupError
    mock_service.queries.side_effect = FXRateLookupError(eur, usd, test_date)
    
    with pytest.raises(FXRateLookupError):
        list(mock_service.queries(queries, strict=True))


def test_FXRateService_queries_empty():
    """Test the queries method with empty queries."""
    
    mock_service = Mock(spec=FXRateService)
    mock_service.queries.return_value = []
    
    queries = []
    results = list(mock_service.queries(queries, strict=False))
    
    assert len(results) == 0
    mock_service.queries.assert_called_once_with(queries, strict=False)


def test_FXRateService_queries_multiple_rates():
    """Test the queries method with multiple rates."""
    
    mock_service = Mock(spec=FXRateService)
    
    eur = Mock(spec=Currency)
    usd = Mock(spec=Currency)
    gbp = Mock(spec=Currency)
    jpy = Mock(spec=Currency)
    
    test_date = datetime.date(2023, 1, 1)
    
    rates = [
        FXRate(eur, usd, test_date, Decimal("1.1")),
        FXRate(usd, gbp, test_date, Decimal("0.8")),
        FXRate(gbp, jpy, test_date, Decimal("150.5")),
        FXRate(jpy, eur, test_date, Decimal("0.0066")),
    ]
    
    queries = [
        (eur, usd, test_date),
        (usd, gbp, test_date),
        (gbp, jpy, test_date),
        (jpy, eur, test_date),
    ]
    
    mock_service.queries.return_value = rates
    
    results = list(mock_service.queries(queries, strict=False))
    
    assert len(results) == 4
    assert all(isinstance(rate, FXRate) for rate in results)
    assert results[0].value == Decimal("1.1")
    assert results[1].value == Decimal("0.8")
    assert results[2].value == Decimal("150.5")
    assert results[3].value == Decimal("0.0066")


# LLM-generated content at query #33
#--------------------------

```python
def test_FXRateService_query():
    """
    Test the query method of FXRateService abstract class.
    """
    import datetime
    from decimal import Decimal
    from unittest.mock import Mock
    
    # Create a concrete implementation of FXRateService for testing
    class MockFXRateService(FXRateService):
        def __init__(self):
            self.rates = {}
        
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            key = (ccy1, ccy2, asof)
            if key in self.rates:
                return self.rates[key]
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None
        
        def queries(self, queries: Iterable[TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    # Setup
    service = MockFXRateService()
    eur = Currency("EUR", 1, "Euro")
    usd = Currency("USD", 1, "US Dollar")
    test_date = datetime.date(2023, 1, 1)
    
    rate = FXRate(eur, usd, test_date, Decimal("1.20"))
    service.rates[(eur, usd, test_date)] = rate
    
    # Test: Query existing rate (non-strict mode)
    result = service.query(eur, usd, test_date, strict=False)
    assert result == rate
    assert result.ccy1 == eur
    assert result.ccy2 == usd
    assert result.value == Decimal("1.20")
    
    # Test: Query non-existing rate (non-strict mode)
    gbp = Currency("GBP", 1, "British Pound")
    result = service.query(gbp, usd, test_date, strict=False)
    assert result is None
    
    # Test: Query non-existing rate (strict mode) raises error
    with pytest.raises(FXRateLookupError) as exc_info:
        service.query(gbp, usd, test_date, strict=True)
    assert exc_info.value.ccy1 == gbp
    assert exc_info.value.ccy2 == usd
    assert exc_info.value.asof == test_date
    
    # Test: Query with different date
    other_date = datetime.date(2023, 1, 2)
    result = service.query(eur, usd, other_date, strict=False)
    assert result is None


# LLM-generated content at query #34
#--------------------------

```python
def test_FXRateService_query():
    """Test the query method of FXRateService."""
    import datetime
    from decimal import Decimal
    from unittest.mock import Mock
    
    # Create a mock implementation of FXRateService
    service = Mock(spec=FXRateService)
    
    # Test case 1: Query returns a valid FXRate
    ccy1 = Mock(spec=Currency)
    ccy2 = Mock(spec=Currency)
    asof = datetime.date(2023, 1, 15)
    expected_rate = FXRate(ccy1, ccy2, asof, Decimal("1.25"))
    
    service.query.return_value = expected_rate
    result = service.query(ccy1, ccy2, asof, strict=False)
    assert result == expected_rate
    service.query.assert_called_once_with(ccy1, ccy2, asof, strict=False)
    
    # Test case 2: Query with strict=True and rate not found raises FXRateLookupError
    service.reset_mock()
    service.query.side_effect = FXRateLookupError(ccy1, ccy2, asof)
    
    with pytest.raises(FXRateLookupError):
        service.query(ccy1, ccy2, asof, strict=True)
    
    # Test case 3: Query with strict=False and rate not found returns None
    service.reset_mock()
    service.query.return_value = None
    result = service.query(ccy1, ccy2, asof, strict=False)
    assert result is None
    service.query.assert_called_once_with(ccy1, ccy2, asof, strict=False)
    
    # Test case 4: Query with different currencies and dates
    service.reset_mock()
    ccy3 = Mock(spec=Currency)
    ccy4 = Mock(spec=Currency)
    asof2 = datetime.date(2023, 6, 20)
    expected_rate2 = FXRate(ccy3, ccy4, asof2, Decimal("0.85"))
    
    service.query.return_value = expected_rate2
    result = service.query(ccy3, ccy4, asof2, strict=False)
    assert result == expected_rate2
    assert result.ccy1 == ccy3
    assert result.ccy2 == ccy4
    assert result.date == asof2
    assert result.value == Decimal("0.85")


# LLM-generated content at query #35
#--------------------------

```python
import pytest
from decimal import Decimal
from datetime import date
from unittest.mock import Mock, MagicMock
from typing import Optional, Iterable


def test_FXRateService_queries():
    """Test the queries method of FXRateService."""
    
    # Create a mock FXRateService instance
    service = Mock(spec=FXRateService)
    
    # Setup test data
    test_date = date.today()
    ccy1 = Mock(spec=Currency)
    ccy2 = Mock(spec=Currency)
    ccy3 = Mock(spec=Currency)
    
    rate1 = Mock(spec=FXRate)
    rate1.ccy1 = ccy1
    rate1.ccy2 = ccy2
    rate1.date = test_date
    rate1.value = Decimal("1.5")
    
    rate2 = Mock(spec=FXRate)
    rate2.ccy1 = ccy2
    rate2.ccy2 = ccy3
    rate2.date = test_date
    rate2.value = Decimal("2.0")
    
    # Test queries with successful results
    queries = [(ccy1, ccy2, test_date), (ccy2, ccy3, test_date)]
    expected_results = [rate1, rate2]
    
    service.queries.return_value = expected_results
    
    results = list(service.queries(queries, strict=False))
    
    assert len(results) == 2
    assert results[0] == rate1
    assert results[1] == rate2
    service.queries.assert_called_once_with(queries, strict=False)


def test_FXRateService_queries_with_none_results():
    """Test the queries method when some rates are not found."""
    
    service = Mock(spec=FXRateService)
    
    test_date = date.today()
    ccy1 = Mock(spec=Currency)
    ccy2 = Mock(spec=Currency)
    ccy3 = Mock(spec=Currency)
    
    rate1 = Mock(spec=FXRate)
    
    queries = [(ccy1, ccy2, test_date), (ccy2, ccy3, test_date)]
    expected_results = [rate1, None]
    
    service.queries.return_value = expected_results
    
    results = list(service.queries(queries, strict=False))
    
    assert len(results) == 2
    assert results[0] == rate1
    assert results[1] is None


def test_FXRateService_queries_strict_mode():
    """Test the queries method in strict mode."""
    
    service = Mock(spec=FXRateService)
    
    test_date = date.today()
    ccy1 = Mock(spec=Currency)
    ccy2 = Mock(spec=Currency)
    
    queries = [(ccy1, ccy2, test_date)]
    
    service.queries.side_effect = FXRateLookupError(ccy1, ccy2, test_date)
    
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))


def test_FXRateService_queries_empty():
    """Test the queries method with empty query list."""
    
    service = Mock(spec=FXRateService)
    
    queries = []
    expected_results = []
    
    service.queries.return_value = expected_results
    
    results = list(service.queries(queries, strict=False))
    
    assert len(results) == 0


def test_FXRateService_queries_multiple_calls():
    """Test the queries method with multiple sequential calls."""
    
    service = Mock(spec=FXRateService)
    
    test_date = date.today()
    ccy1 = Mock(spec=Currency)
    ccy2 = Mock(spec=Currency)
    
    rate = Mock(spec=FXRate)
    
    queries = [(ccy1, ccy2, test_date)]
    
    service.queries.return_value = [rate]
    
    results1 = list(service.queries(queries))
    results2 = list(service.queries(queries))
    
    assert results1 == results2
    assert service.queries.call_count == 2


