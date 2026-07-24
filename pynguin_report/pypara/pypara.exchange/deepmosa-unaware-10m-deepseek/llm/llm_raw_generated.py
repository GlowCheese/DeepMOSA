####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_FXRate___invert__():
    from datetime import date
    from decimal import Decimal
    from pypara.currencies import Currencies

    # Test basic inversion
    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    test_date = date(2023, 1, 1)
    rate_value = Decimal("1.1")
    
    original = FXRate(eur, usd, test_date, rate_value)
    inverted = ~original
    
    assert inverted.ccy1 == usd
    assert inverted.ccy2 == eur
    assert inverted.date == test_date
    assert inverted.value == Decimal("1") / rate_value
    
    # Test inversion of inverted rate returns original
    assert ~~original == original
    
    # Test inversion with different values
    rate_value2 = Decimal("2.5")
    original2 = FXRate(eur, usd, test_date, rate_value2)
    inverted2 = ~original2
    
    assert inverted2.value == Decimal("1") / rate_value2
    
    # Test inversion with same currency (rate should be 1)
    same_ccy = FXRate(eur, eur, test_date, Decimal("1"))
    inverted_same = ~same_ccy
    
    assert inverted_same.ccy1 == eur
    assert inverted_same.ccy2 == eur
    assert inverted_same.value == Decimal("1")
    
    # Test inversion preserves date
    different_date = date(2023, 12, 31)
    original3 = FXRate(usd, eur, different_date, Decimal("0.9"))
    inverted3 = ~original3
    
    assert inverted3.date == different_date


# LLM-generated content at query #2
#--------------------------

```python
def test_FXRate___invert__():
    from datetime import date
    from decimal import Decimal
    from pypara.currencies import Currencies

    # Test basic inversion
    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    test_date = date(2023, 1, 1)
    original_rate = FXRate(eur, usd, test_date, Decimal("1.1"))
    inverted_rate = ~original_rate
    
    assert inverted_rate.ccy1 == usd
    assert inverted_rate.ccy2 == eur
    assert inverted_rate.date == test_date
    assert inverted_rate.value == Decimal("1") / Decimal("1.1")
    
    # Test inversion of inverted rate returns original
    double_inverted = ~inverted_rate
    assert double_inverted == original_rate
    
    # Test with different values
    rate2 = FXRate(Currencies["GBP"], Currencies["JPY"], date(2023, 1, 2), Decimal("150.5"))
    inverted2 = ~rate2
    assert inverted2.ccy1 == Currencies["JPY"]
    assert inverted2.ccy2 == Currencies["GBP"]
    assert inverted2.date == date(2023, 1, 2)
    assert inverted2.value == Decimal("1") / Decimal("150.5")
    
    # Test that inversion works correctly with indexed access
    rate3 = FXRate(Currencies["CAD"], Currencies["AUD"], date(2023, 1, 3), Decimal("0.9"))
    inverted3 = ~rate3
    assert inverted3[0] == Currencies["AUD"]
    assert inverted3[1] == Currencies["CAD"]
    assert inverted3[2] == date(2023, 1, 3)
    assert inverted3[3] == Decimal("1") / Decimal("0.9")


# LLM-generated content at query #3
#--------------------------

```python
def test_FXRateService_query():
    # Create a concrete implementation for testing
    class TestFXRateService(FXRateService):
        def __init__(self, rates):
            self.rates = rates
        
        def query(self, ccy1, ccy2, asof, strict=False):
            key = (ccy1, ccy2, asof)
            if key in self.rates:
                return self.rates[key]
            elif strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None
        
        def queries(self, queries, strict=False):
            results = []
            for ccy1, ccy2, asof in queries:
                try:
                    results.append(self.query(ccy1, ccy2, asof, strict))
                except FXRateLookupError:
                    results.append(None)
            return results
    
    # Create test currencies
    class TestCurrency:
        def __init__(self, code):
            self.code = code
        
        def __eq__(self, other):
            return isinstance(other, TestCurrency) and self.code == other.code
        
        def __hash__(self):
            return hash(self.code)
    
    EUR = TestCurrency("EUR")
    USD = TestCurrency("USD")
    GBP = TestCurrency("GBP")
    
    # Create test dates
    date1 = Date(2023, 1, 1)
    date2 = Date(2023, 1, 2)
    
    # Create test FXRate
    rate1 = FXRate(EUR, USD, date1, Decimal("1.1"))
    rate2 = FXRate(EUR, USD, date2, Decimal("1.2"))
    rate3 = FXRate(USD, GBP, date1, Decimal("0.8"))
    
    # Setup service with rates
    service = TestFXRateService({
        (EUR, USD, date1): rate1,
        (EUR, USD, date2): rate2,
        (USD, GBP, date1): rate3
    })
    
    # Test 1: Query existing rate
    result = service.query(EUR, USD, date1)
    assert result == rate1
    assert result.ccy1 == EUR
    assert result.ccy2 == USD
    assert result.date == date1
    assert result.value == Decimal("1.1")
    
    # Test 2: Query existing rate with different date
    result = service.query(EUR, USD, date2)
    assert result == rate2
    assert result.value == Decimal("1.2")
    
    # Test 3: Query non-existing rate without strict mode
    result = service.query(GBP, EUR, date1)
    assert result is None
    
    # Test 4: Query non-existing rate with strict mode
    try:
        service.query(GBP, EUR, date1, strict=True)
        assert False, "Should have raised FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == GBP
        assert e.ccy2 == EUR
        assert e.asof == date1
    
    # Test 5: Query inverted pair (not in service, should return None)
    result = service.query(USD, EUR, date1)
    assert result is None
    
    # Test 6: Query with same currency (not in service, should return None)
    result = service.query(EUR, EUR, date1)
    assert result is None
    
    # Test 7: Test queries method integration
    query_list = [
        (EUR, USD, date1),
        (EUR, USD, date2),
        (GBP, EUR, date1),
        (USD, GBP, date1)
    ]
    
    results = list(service.queries(query_list))
    assert results == [rate1, rate2, None, rate3]
    
    # Test 8: Test queries with strict mode
    results = list(service.queries(query_list, strict=True))
    assert results == [rate1, rate2, None, rate3]


# LLM-generated content at query #4
#--------------------------

```python
def test_FXRateService_queries():
    from datetime import date
    from decimal import Decimal
    from typing import List, Optional
    from unittest.mock import Mock, call

    class TestFXRateService(FXRateService):
        def __init__(self, rates: List[Optional[FXRate]]):
            self.rates = rates
            self.query_calls = []

        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            self.query_calls.append((ccy1, ccy2, asof, strict))
            return None

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return self.rates

    eur = Mock(spec=Currency)
    usd = Mock(spec=Currency)
    gbp = Mock(spec=Currency)
    date1 = date(2023, 1, 1)
    date2 = date(2023, 1, 2)

    rate1 = FXRate(eur, usd, date1, Decimal("1.1"))
    rate2 = FXRate(usd, gbp, date2, Decimal("0.8"))
    rate3 = None

    service = TestFXRateService([rate1, rate2, rate3])
    queries = [(eur, usd, date1), (usd, gbp, date2), (eur, gbp, date1)]
    
    results = list(service.queries(queries))
    
    assert results == [rate1, rate2, rate3]
    assert len(results) == 3
    assert results[0] is rate1
    assert results[1] is rate2
    assert results[2] is rate3

    service_with_strict = TestFXRateService([rate1, rate2])
    results_strict = list(service_with_strict.queries(queries[:2], strict=True))
    
    assert results_strict == [rate1, rate2]

    empty_service = TestFXRateService([])
    empty_results = list(empty_service.queries([]))
    
    assert empty_results == []


# LLM-generated content at query #5
#--------------------------

```python
def test_FXRateService_queries():
    from datetime import date
    from decimal import Decimal
    from typing import List, Optional
    from unittest.mock import Mock, call

    class MockFXRateService(FXRateService):
        def __init__(self, rates: List[Optional[FXRate]]):
            self.rates = rates
            self.query_calls = []

        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            self.query_calls.append((ccy1, ccy2, asof, strict))
            if self.rates:
                return self.rates.pop(0)
            return None

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            results = []
            for ccy1, ccy2, asof in queries:
                try:
                    rate = self.query(ccy1, ccy2, asof, strict)
                    results.append(rate)
                except FXRateLookupError:
                    results.append(None)
            return results

    eur = Currency("EUR", 2)
    usd = Currency("USD", 2)
    gbp = Currency("GBP", 2)
    
    today = date.today()
    yesterday = date(today.year, today.month, today.day - 1)
    
    rate1 = FXRate(eur, usd, today, Decimal("1.1"))
    rate2 = FXRate(usd, gbp, yesterday, Decimal("0.8"))
    rate3 = FXRate(gbp, eur, today, Decimal("1.25"))
    
    service = MockFXRateService([rate1, rate2, rate3])
    
    queries = [
        (eur, usd, today),
        (usd, gbp, yesterday),
        (gbp, eur, today),
    ]
    
    results = list(service.queries(queries, strict=False))
    
    assert results == [rate1, rate2, rate3]
    assert len(service.query_calls) == 3
    assert service.query_calls[0] == (eur, usd, today, False)
    assert service.query_calls[1] == (usd, gbp, yesterday, False)
    assert service.query_calls[2] == (gbp, eur, today, False)
    
    service_with_none = MockFXRateService([rate1, None, rate3])
    queries_with_missing = [
        (eur, usd, today),
        (usd, gbp, yesterday),
        (gbp, eur, today),
    ]
    
    results_with_none = list(service_with_none.queries(queries_with_missing, strict=False))
    assert results_with_none == [rate1, None, rate3]
    
    class StrictFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            if strict:
                for ccy1, ccy2, asof in queries:
                    raise FXRateLookupError(ccy1, ccy2, asof)
            return [None] * len(list(queries))

    strict_service = StrictFXRateService()
    queries_single = [(eur, usd, today)]
    
    results_strict_false = list(strict_service.queries(queries_single, strict=False))
    assert results_strict_false == [None]
    
    try:
        list(strict_service.queries(queries_single, strict=True))
        assert False, "Should have raised FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == eur
        assert e.ccy2 == usd
        assert e.asof == today
    
    empty_service = MockFXRateService([])
    empty_results = list(empty_service.queries([], strict=False))
    assert empty_results == []
    assert len(empty_service.query_calls) == 0


# LLM-generated content at query #6
#--------------------------

```python
def test_FXRateService_queries():
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            if ccy1.code == "EUR" and ccy2.code == "USD" and asof == Date(2023, 1, 1):
                return FXRate(ccy1, ccy2, asof, Decimal("1.1"))
            if ccy1.code == "USD" and ccy2.code == "EUR" and asof == Date(2023, 1, 1):
                return FXRate(ccy1, ccy2, asof, Decimal("0.909"))
            if ccy1.code == "GBP" and ccy2.code == "USD" and asof == Date(2023, 1, 2):
                return FXRate(ccy1, ccy2, asof, Decimal("1.25"))
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            results = []
            for ccy1, ccy2, asof in queries:
                try:
                    rate = self.query(ccy1, ccy2, asof, strict)
                    results.append(rate)
                except FXRateLookupError:
                    results.append(None)
            return results

    eur = Currency("EUR", 2)
    usd = Currency("USD", 2)
    gbp = Currency("GBP", 2)
    jpy = Currency("JPY", 0)

    service = MockFXRateService()
    
    queries = [
        (eur, usd, Date(2023, 1, 1)),
        (usd, eur, Date(2023, 1, 1)),
        (gbp, usd, Date(2023, 1, 2)),
        (eur, jpy, Date(2023, 1, 1)),
    ]
    
    results = list(service.queries(queries, strict=False))
    
    assert len(results) == 4
    assert results[0] == FXRate(eur, usd, Date(2023, 1, 1), Decimal("1.1"))
    assert results[1] == FXRate(usd, eur, Date(2023, 1, 1), Decimal("0.909"))
    assert results[2] == FXRate(gbp, usd, Date(2023, 1, 2), Decimal("1.25"))
    assert results[3] is None
    
    results_strict = list(service.queries(queries[:3], strict=True))
    assert len(results_strict) == 3
    assert results_strict[0] == FXRate(eur, usd, Date(2023, 1, 1), Decimal("1.1"))
    assert results_strict[1] == FXRate(usd, eur, Date(2023, 1, 1), Decimal("0.909"))
    assert results_strict[2] == FXRate(gbp, usd, Date(2023, 1, 2), Decimal("1.25"))
    
    empty_results = list(service.queries([], strict=False))
    assert empty_results == []


# LLM-generated content at query #7
#--------------------------

```python
def test_FXRateService_query():
    from decimal import Decimal
    from datetime import date
    from unittest.mock import Mock, patch
    from pypara.currencies import Currencies
    
    # Create a concrete implementation for testing
    class TestFXRateService(FXRateService):
        def __init__(self, rates=None):
            self.rates = rates or {}
            
        def query(self, ccy1, ccy2, asof, strict=False):
            key = (ccy1, ccy2, asof)
            if key in self.rates:
                return self.rates[key]
            elif strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None
            
        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    # Test 1: Query existing rate
    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    test_date = date(2023, 1, 1)
    test_rate = FXRate(eur, usd, test_date, Decimal("1.1"))
    
    service = TestFXRateService({(eur, usd, test_date): test_rate})
    
    result = service.query(eur, usd, test_date)
    assert result == test_rate
    
    # Test 2: Query non-existing rate with strict=False (default)
    result = service.query(usd, eur, test_date)
    assert result is None
    
    # Test 3: Query non-existing rate with strict=True
    try:
        service.query(usd, eur, test_date, strict=True)
        assert False, "Should have raised FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == usd
        assert e.ccy2 == eur
        assert e.asof == test_date
    
    # Test 4: Query with inverted rate
    gbp = Currencies["GBP"]
    inverted_rate = FXRate(usd, gbp, test_date, Decimal("0.8"))
    service.rates[(usd, gbp, test_date)] = inverted_rate
    
    result = service.query(usd, gbp, test_date)
    assert result == inverted_rate
    
    # Test 5: Query with same currency
    same_currency_rate = FXRate(eur, eur, test_date, Decimal("1"))
    service.rates[(eur, eur, test_date)] = same_currency_rate
    
    result = service.query(eur, eur, test_date)
    assert result == same_currency_rate
    assert result.value == Decimal("1")
    
    # Test 6: Test with different date
    other_date = date(2023, 1, 2)
    other_rate = FXRate(eur, usd, other_date, Decimal("1.2"))
    service.rates[(eur, usd, other_date)] = other_rate
    
    result = service.query(eur, usd, other_date)
    assert result == other_rate
    assert result.value == Decimal("1.2")
    
    # Test 7: Test abstract class cannot be instantiated
    try:
        FXRateService()
        assert False, "Should not be able to instantiate abstract class"
    except TypeError:
        pass
    
    # Test 8: Test with multiple currencies
    jpy = Currencies["JPY"]
    jpy_rate = FXRate(usd, jpy, test_date, Decimal("110.5"))
    service.rates[(usd, jpy, test_date)] = jpy_rate
    
    result = service.query(usd, jpy, test_date)
    assert result == jpy_rate
    assert result.value == Decimal("110.5")


# LLM-generated content at query #8
#--------------------------

```python
def test_FXRateService_query():
    # Create a concrete implementation for testing
    class MockFXRateService(FXRateService):
        def __init__(self, rates=None):
            self.rates = rates or {}
            
        def query(self, ccy1, ccy2, asof, strict=False):
            key = (ccy1, ccy2, asof)
            if key in self.rates:
                return self.rates[key]
            elif strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None
            
        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    # Create test currencies and dates
    class MockCurrency:
        def __init__(self, code):
            self.code = code
        def __eq__(self, other):
            return isinstance(other, MockCurrency) and self.code == other.code
        def __hash__(self):
            return hash(self.code)
        def __repr__(self):
            return f"Currency({self.code})"
    
    class MockDate:
        def __init__(self, year, month, day):
            self.year = year
            self.month = month
            self.day = day
        def __eq__(self, other):
            return (isinstance(other, MockDate) and 
                    self.year == other.year and 
                    self.month == other.month and 
                    self.day == other.day)
        def __hash__(self):
            return hash((self.year, self.month, self.day))
        def __repr__(self):
            return f"Date({self.year}-{self.month}-{self.day})"
    
    # Test data
    eur = MockCurrency("EUR")
    usd = MockCurrency("USD")
    gbp = MockCurrency("GBP")
    date1 = MockDate(2023, 1, 1)
    date2 = MockDate(2023, 1, 2)
    
    # Create test FX rates
    rate1 = FXRate(eur, usd, date1, Decimal("1.1"))
    rate2 = FXRate(eur, usd, date2, Decimal("1.2"))
    rate3 = FXRate(usd, gbp, date1, Decimal("0.8"))
    
    # Test 1: Query existing rate (non-strict mode)
    service = MockFXRateService({
        (eur, usd, date1): rate1,
        (eur, usd, date2): rate2,
        (usd, gbp, date1): rate3
    })
    
    result = service.query(eur, usd, date1)
    assert result == rate1
    assert result.ccy1 == eur
    assert result.ccy2 == usd
    assert result.date == date1
    assert result.value == Decimal("1.1")
    
    # Test 2: Query existing rate (strict mode)
    result = service.query(eur, usd, date1, strict=True)
    assert result == rate1
    
    # Test 3: Query non-existing rate (non-strict mode)
    result = service.query(gbp, eur, date1)
    assert result is None
    
    # Test 4: Query non-existing rate (strict mode) - should raise FXRateLookupError
    try:
        service.query(gbp, eur, date1, strict=True)
        assert False, "Should have raised FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == gbp
        assert e.ccy2 == eur
        assert e.asof == date1
    
    # Test 5: Query with different date
    result = service.query(eur, usd, date2)
    assert result == rate2
    assert result.value == Decimal("1.2")
    
    # Test 6: Query different currency pair
    result = service.query(usd, gbp, date1)
    assert result == rate3
    assert result.ccy1 == usd
    assert result.ccy2 == gbp
    
    # Test 7: Test inverted rate using ~ operator
    inverted_rate = ~rate1
    assert inverted_rate.ccy1 == usd
    assert inverted_rate.ccy2 == eur
    assert inverted_rate.date == date1
    assert inverted_rate.value == Decimal("1") / Decimal("1.1")
    
    # Test 8: Empty service query
    empty_service = MockFXRateService()
    result = empty_service.query(eur, usd, date1)
    assert result is None
    
    # Test 9: Empty service query in strict mode
    try:
        empty_service.query(eur, usd, date1, strict=True)
        assert False, "Should have raised FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == eur
        assert e.ccy2 == usd
        assert e.asof == date1
    
    # Test 10: Test FXRate.of method validation
    valid_rate = FXRate.of(eur, usd, date1, Decimal("1.5"))
    assert valid_rate.ccy1 == eur
    assert valid_rate.ccy2 == usd
    assert valid_rate.date == date1
    assert valid_rate.value == Decimal("1.5")
    
    # Test 11: Test same currency rate must be one
    same_ccy_rate = FXRate.of(eur, eur, date1, ONE)
    assert same_ccy_rate.value == ONE
    
    # Test 12: Test queries method
    queries = [(eur, usd, date1), (usd, gbp, date1), (gbp, eur, date1)]
    results = list(service.queries(queries))
    assert results == [rate1, rate3, None]
    
    # Test 13: Test queries method with strict mode
    try:
        list(service.queries(queries, strict=True))
        assert False, "Should have raised FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == gbp
        assert e.ccy2 == eur
        assert e.asof == date1


# LLM-generated content at query #9
#--------------------------

```python
def test_FXRateService_query():
    from decimal import Decimal
    from datetime import date
    from unittest.mock import Mock, patch
    from pypara.currencies import Currency
    from pypara.currencies import Currencies

    # Create a concrete implementation for testing
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            # Simple implementation that returns a fixed rate for EUR/USD
            if ccy1 == Currencies["EUR"] and ccy2 == Currencies["USD"] and asof == date(2023, 1, 1):
                return FXRate(ccy1, ccy2, asof, Decimal("1.1"))
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None

        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = ConcreteFXRateService()

    # Test successful query
    rate = service.query(Currencies["EUR"], Currencies["USD"], date(2023, 1, 1))
    assert rate is not None
    assert rate.ccy1 == Currencies["EUR"]
    assert rate.ccy2 == Currencies["USD"]
    assert rate.date == date(2023, 1, 1)
    assert rate.value == Decimal("1.1")

    # Test non-strict query for non-existent rate
    rate = service.query(Currencies["USD"], Currencies["EUR"], date(2023, 1, 1))
    assert rate is None

    # Test strict query for non-existent rate
    try:
        service.query(Currencies["USD"], Currencies["EUR"], date(2023, 1, 1), strict=True)
        assert False, "Should have raised FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == Currencies["USD"]
        assert e.ccy2 == Currencies["EUR"]
        assert e.asof == date(2023, 1, 1)

    # Test with same currency (should return 1:1 rate if implemented)
    # Note: This depends on the actual implementation
    class SameCurrencyService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if ccy1 == ccy2:
                return FXRate(ccy1, ccy2, asof, Decimal("1"))
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None

        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    same_service = SameCurrencyService()
    rate = same_service.query(Currencies["EUR"], Currencies["EUR"], date(2023, 1, 1))
    assert rate is not None
    assert rate.value == Decimal("1")

    # Test with mocked abstract class
    with patch.multiple(FXRateService, __abstractmethods__=set()):
        mock_service = FXRateService()
        mock_service.query = Mock(return_value=FXRate(
            Currencies["GBP"], Currencies["USD"], date(2023, 1, 1), Decimal("1.3")
        ))
        
        rate = mock_service.query(Currencies["GBP"], Currencies["USD"], date(2023, 1, 1))
        assert rate.ccy1 == Currencies["GBP"]
        assert rate.ccy2 == Currencies["USD"]
        assert rate.value == Decimal("1.3")


# LLM-generated content at query #10
#--------------------------

```python
def test_FXRateService_queries():
    from decimal import Decimal
    from datetime import date
    from unittest.mock import Mock, call
    
    # Create a mock implementation of FXRateService
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            # Simple mock implementation for testing
            if ccy1.code == "EUR" and ccy2.code == "USD" and asof == date(2023, 1, 1):
                return FXRate(ccy1, ccy2, asof, Decimal("1.1"))
            elif ccy1.code == "USD" and ccy2.code == "EUR" and asof == date(2023, 1, 1):
                return FXRate(ccy1, ccy2, asof, Decimal("0.9091"))
            elif strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None
        
        def queries(self, queries, strict=False):
            results = []
            for ccy1, ccy2, asof in queries:
                try:
                    rate = self.query(ccy1, ccy2, asof, strict)
                    results.append(rate)
                except FXRateLookupError:
                    if strict:
                        raise
                    results.append(None)
            return results
    
    # Create currency objects
    eur = Mock(code="EUR")
    usd = Mock(code="USD")
    gbp = Mock(code="GBP")
    
    # Test 1: Basic queries with found rates
    service = MockFXRateService()
    queries = [
        (eur, usd, date(2023, 1, 1)),
        (usd, eur, date(2023, 1, 1)),
    ]
    
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0].ccy1 == eur
    assert results[0].ccy2 == usd
    assert results[0].date == date(2023, 1, 1)
    assert results[0].value == Decimal("1.1")
    assert results[1].value == Decimal("0.9091")
    
    # Test 2: Queries with some not found rates (non-strict mode)
    queries = [
        (eur, usd, date(2023, 1, 1)),
        (gbp, eur, date(2023, 1, 1)),  # This won't be found
        (usd, eur, date(2023, 1, 1)),
    ]
    
    results = list(service.queries(queries, strict=False))
    assert len(results) == 3
    assert results[0] is not None
    assert results[1] is None  # Not found rate
    assert results[2] is not None
    
    # Test 3: Queries with strict mode (should raise exception)
    queries = [
        (eur, usd, date(2023, 1, 1)),
        (gbp, eur, date(2023, 1, 1)),  # This will cause exception
    ]
    
    try:
        list(service.queries(queries, strict=True))
        assert False, "Should have raised FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == gbp
        assert e.ccy2 == eur
        assert e.asof == date(2023, 1, 1)
    
    # Test 4: Empty queries list
    results = list(service.queries([]))
    assert len(results) == 0
    
    # Test 5: Test with actual FXRateService implementation
    class ConcreteFXRateService(FXRateService):
        def __init__(self):
            self.rates = {}
        
        def query(self, ccy1, ccy2, asof, strict=False):
            key = (ccy1.code, ccy2.code, asof)
            if key in self.rates:
                return self.rates[key]
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None
        
        def queries(self, queries, strict=False):
            results = []
            for ccy1, ccy2, asof in queries:
                try:
                    rate = self.query(ccy1, ccy2, asof, strict)
                    results.append(rate)
                except FXRateLookupError:
                    if strict:
                        raise
                    results.append(None)
            return results
    
    concrete_service = ConcreteFXRateService()
    
    # Add some rates
    rate1 = FXRate(eur, usd, date(2023, 1, 1), Decimal("1.1"))
    rate2 = FXRate(usd, gbp, date(2023, 1, 1), Decimal("0.8"))
    concrete_service.rates[("EUR", "USD", date(2023, 1, 1))] = rate1
    concrete_service.rates[("USD", "GBP", date(2023, 1, 1))] = rate2
    
    queries = [
        (eur, usd, date(2023, 1, 1)),
        (usd, gbp, date(2023, 1, 1)),
        (gbp, eur, date(2023, 1, 1)),  # Not found
    ]
    
    results = list(concrete_service.queries(queries, strict=False))
    assert results[0] == rate1
    assert results[1] == rate2
    assert results[2] is None


# LLM-generated content at query #11
#--------------------------

```python
def test_FXRateService_queries():
    from decimal import Decimal
    from datetime import date
    from unittest.mock import Mock, call

    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return Mock()

        def queries(self, queries, strict=False):
            results = []
            for ccy1, ccy2, asof in queries:
                if ccy1.code == "EUR" and ccy2.code == "USD" and asof == date(2023, 1, 1):
                    results.append(FXRate(ccy1, ccy2, asof, Decimal("1.1")))
                elif ccy1.code == "GBP" and ccy2.code == "JPY" and asof == date(2023, 1, 2):
                    results.append(FXRate(ccy1, ccy2, asof, Decimal("150.5")))
                elif strict:
                    raise FXRateLookupError(ccy1, ccy2, asof)
                else:
                    results.append(None)
            return results

    eur = Mock(code="EUR")
    usd = Mock(code="USD")
    gbp = Mock(code="GBP")
    jpy = Mock(code="JPY")
    chf = Mock(code="CHF")

    service = TestFXRateService()
    
    queries = [
        (eur, usd, date(2023, 1, 1)),
        (gbp, jpy, date(2023, 1, 2)),
        (chf, usd, date(2023, 1, 3)),
    ]
    
    results = list(service.queries(queries, strict=False))
    
    assert len(results) == 3
    assert results[0] == FXRate(eur, usd, date(2023, 1, 1), Decimal("1.1"))
    assert results[1] == FXRate(gbp, jpy, date(2023, 1, 2), Decimal("150.5"))
    assert results[2] is None

    queries_strict = [
        (eur, usd, date(2023, 1, 1)),
        (chf, usd, date(2023, 1, 3)),
    ]
    
    try:
        list(service.queries(queries_strict, strict=True))
        assert False, "Should have raised FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == chf
        assert e.ccy2 == usd
        assert e.asof == date(2023, 1, 3)

    empty_results = list(service.queries([], strict=False))
    assert empty_results == []


# LLM-generated content at query #12
#--------------------------

```python
def test_FXRateService_query():
    from decimal import Decimal
    from datetime import date
    from unittest.mock import Mock, patch
    from pypara.currencies import Currency
    from pypara.zeitgeist import Date
    from pypara.fx import FXRate, FXRateService, FXRateLookupError

    # Create mock currencies
    eur = Mock(spec=Currency)
    usd = Mock(spec=Currency)
    gbp = Mock(spec=Currency)
    
    # Create test date
    test_date = date(2023, 1, 1)
    
    # Create a concrete implementation of FXRateService for testing
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            # Simulate finding EUR/USD rate
            if ccy1 == eur and ccy2 == usd and asof == test_date:
                return FXRate(eur, usd, test_date, Decimal("1.1"))
            # Simulate finding USD/EUR rate (inverse)
            elif ccy1 == usd and ccy2 == eur and asof == test_date:
                return FXRate(usd, eur, test_date, Decimal("0.9091"))
            # Simulate not finding GBP/EUR rate
            elif ccy1 == gbp and ccy2 == eur:
                if strict:
                    raise FXRateLookupError(gbp, eur, asof)
                return None
            return None
        
        def queries(self, queries, strict=False):
            return [self.query(q[0], q[1], q[2], strict) for q in queries]
    
    # Create instance of test service
    service = TestFXRateService()
    
    # Test 1: Query existing rate (non-strict mode)
    rate = service.query(eur, usd, test_date, strict=False)
    assert rate is not None
    assert rate.ccy1 == eur
    assert rate.ccy2 == usd
    assert rate.date == test_date
    assert rate.value == Decimal("1.1")
    
    # Test 2: Query existing rate (strict mode)
    rate = service.query(eur, usd, test_date, strict=True)
    assert rate is not None
    assert rate.ccy1 == eur
    assert rate.ccy2 == usd
    assert rate.value == Decimal("1.1")
    
    # Test 3: Query inverse rate
    rate = service.query(usd, eur, test_date, strict=False)
    assert rate is not None
    assert rate.ccy1 == usd
    assert rate.ccy2 == eur
    assert rate.value == Decimal("0.9091")
    
    # Test 4: Query non-existing rate (non-strict mode)
    rate = service.query(gbp, eur, test_date, strict=False)
    assert rate is None
    
    # Test 5: Query non-existing rate (strict mode) - should raise FXRateLookupError
    try:
        service.query(gbp, eur, test_date, strict=True)
        assert False, "Should have raised FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == gbp
        assert e.ccy2 == eur
        assert e.asof == test_date
    
    # Test 6: Query with different date
    other_date = date(2023, 1, 2)
    rate = service.query(eur, usd, other_date, strict=False)
    assert rate is None
    
    # Test 7: Test with None asof
    rate = service.query(eur, usd, None, strict=False)
    assert rate is None


# LLM-generated content at query #13
#--------------------------

```python
def test_FXRateService_queries():
    from datetime import date
    from decimal import Decimal
    from typing import Optional, Iterable
    from unittest.mock import Mock, call

    class MockFXRateService(FXRateService):
        def __init__(self, rates: dict):
            self.rates = rates
            self.query_calls = []

        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            self.query_calls.append((ccy1, ccy2, asof, strict))
            return self.rates.get((ccy1, ccy2, asof))

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    eur = Mock(spec=Currency)
    usd = Mock(spec=Currency)
    gbp = Mock(spec=Currency)
    
    date1 = date(2023, 1, 1)
    date2 = date(2023, 1, 2)
    
    rate1 = FXRate(eur, usd, date1, Decimal("1.1"))
    rate2 = FXRate(usd, gbp, date2, Decimal("0.8"))
    rate3 = FXRate(eur, gbp, date1, Decimal("0.9"))
    
    service = MockFXRateService({
        (eur, usd, date1): rate1,
        (usd, gbp, date2): rate2,
        (eur, gbp, date1): rate3,
    })
    
    queries = [
        (eur, usd, date1),
        (usd, gbp, date2),
        (eur, gbp, date1),
        (gbp, eur, date2),
    ]
    
    results = list(service.queries(queries, strict=False))
    
    assert results == [rate1, rate2, rate3, None]
    assert service.query_calls == [
        (eur, usd, date1, False),
        (usd, gbp, date2, False),
        (eur, gbp, date1, False),
        (gbp, eur, date2, False),
    ]
    
    service_strict = MockFXRateService({
        (eur, usd, date1): rate1,
        (usd, gbp, date2): rate2,
    })
    
    queries_strict = [
        (eur, usd, date1),
        (usd, gbp, date2),
        (eur, gbp, date1),
    ]
    
    try:
        list(service_strict.queries(queries_strict, strict=True))
        assert False, "Should have raised FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == eur
        assert e.ccy2 == gbp
        assert e.asof == date1


# LLM-generated content at query #14
#--------------------------

```python
def test_FXRateService_queries():
    from datetime import date
    from decimal import Decimal
    from typing import Optional, Iterable
    from unittest.mock import Mock, call

    class TestFXRateService(FXRateService):
        def __init__(self, rates: dict):
            self.rates = rates
            self.query_calls = []

        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            self.query_calls.append((ccy1, ccy2, asof, strict))
            return self.rates.get((ccy1, ccy2, asof))

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    eur = Mock(spec=Currency, code="EUR")
    usd = Mock(spec=Currency, code="USD")
    gbp = Mock(spec=Currency, code="GBP")
    
    date1 = date(2023, 1, 1)
    date2 = date(2023, 1, 2)
    
    rate1 = FXRate(eur, usd, date1, Decimal("1.1"))
    rate2 = FXRate(usd, gbp, date2, Decimal("0.8"))
    rate3 = FXRate(eur, gbp, date1, Decimal("0.9"))
    
    service = TestFXRateService({
        (eur, usd, date1): rate1,
        (usd, gbp, date2): rate2,
        (eur, gbp, date1): rate3,
    })
    
    queries = [
        (eur, usd, date1),
        (usd, gbp, date2),
        (eur, gbp, date1),
        (gbp, eur, date2),
    ]
    
    results = list(service.queries(queries, strict=False))
    
    assert results == [rate1, rate2, rate3, None]
    assert service.query_calls == [
        (eur, usd, date1, False),
        (usd, gbp, date2, False),
        (eur, gbp, date1, False),
        (gbp, eur, date2, False),
    ]
    
    service.query_calls.clear()
    
    service_with_missing = TestFXRateService({
        (eur, usd, date1): rate1,
    })
    
    results_strict = list(service_with_missing.queries([(eur, usd, date1), (usd, gbp, date2)], strict=True))
    
    assert results_strict == [rate1, None]
    assert service_with_missing.query_calls == [
        (eur, usd, date1, True),
        (usd, gbp, date2, True),
    ]
    
    empty_service = TestFXRateService({})
    empty_results = list(empty_service.queries([], strict=False))
    
    assert empty_results == []
    assert empty_service.query_calls == []


# LLM-generated content at query #15
#--------------------------

```python
def test_FXRateService_queries():
    from datetime import date
    from decimal import Decimal
    from typing import List, Optional
    from unittest.mock import Mock, create_autospec

    from pypara.currencies import Currency
    from pypara.fx import FXRate, FXRateService

    # Create mock currencies
    eur = create_autospec(Currency, spec_set=True)
    usd = create_autospec(Currency, spec_set=True)
    gbp = create_autospec(Currency, spec_set=True)

    # Create mock FXRate instances
    date1 = date(2023, 1, 1)
    date2 = date(2023, 1, 2)
    rate1 = FXRate(eur, usd, date1, Decimal("1.1"))
    rate2 = FXRate(usd, gbp, date1, Decimal("0.8"))
    rate3 = FXRate(eur, gbp, date2, Decimal("0.9"))

    # Create a concrete implementation of FXRateService
    class TestFXRateService(FXRateService):
        def __init__(self, rates: List[Optional[FXRate]]):
            self.rates = rates
            self.query_calls = []

        def query(self, ccy1: Currency, ccy2: Currency, asof: date, strict: bool = False) -> Optional[FXRate]:
            self.query_calls.append((ccy1, ccy2, asof, strict))
            # For testing, return rates in sequence
            if self.rates:
                return self.rates.pop(0)
            return None

        def queries(self, queries, strict=False):
            results = []
            for ccy1, ccy2, asof in queries:
                rate = self.query(ccy1, ccy2, asof, strict)
                results.append(rate)
            return results

    # Test 1: Normal operation with multiple queries
    service = TestFXRateService([rate1, rate2, rate3])
    queries = [
        (eur, usd, date1),
        (usd, gbp, date1),
        (eur, gbp, date2),
    ]
    
    results = list(service.queries(queries))
    
    assert results == [rate1, rate2, rate3]
    assert len(service.query_calls) == 3
    assert service.query_calls[0] == (eur, usd, date1, False)
    assert service.query_calls[1] == (usd, gbp, date1, False)
    assert service.query_calls[2] == (eur, gbp, date2, False)

    # Test 2: With strict mode
    service = TestFXRateService([rate1, None, rate3])
    queries = [
        (eur, usd, date1),
        (usd, gbp, date1),
        (eur, gbp, date2),
    ]
    
    # In strict mode, None results should raise FXRateLookupError
    # We'll test this by mocking the query method to raise the error
    class StrictFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if strict and (ccy2 == gbp and asof == date1):
                from pypara.fx import FXRateLookupError
                raise FXRateLookupError(ccy1, ccy2, asof)
            return rate1 if ccy1 == eur else None

        def queries(self, queries, strict=False):
            results = []
            for ccy1, ccy2, asof in queries:
                try:
                    rate = self.query(ccy1, ccy2, asof, strict)
                    results.append(rate)
                except FXRateLookupError:
                    results.append(None)
            return results

    strict_service = StrictFXRateService()
    results = list(strict_service.queries(queries, strict=True))
    # The second query should return None due to FXRateLookupError being caught
    assert results[0] == rate1
    assert results[1] is None
    assert results[2] == rate1

    # Test 3: Empty queries
    service = TestFXRateService([])
    results = list(service.queries([]))
    assert results == []
    assert len(service.query_calls) == 0

    # Test 4: Single query
    service = TestFXRateService([rate1])
    queries = [(eur, usd, date1)]
    results = list(service.queries(queries))
    assert results == [rate1]
    assert len(service.query_calls) == 1

    # Test 5: Verify query method is called with correct strict parameter
    service = TestFXRateService([rate1, rate2])
    queries = [(eur, usd, date1), (usd, gbp, date1)]
    results = list(service.queries(queries, strict=True))
    assert service.query_calls[0][3] == True  # strict parameter
    assert service.query_calls[1][3] == True  # strict parameter


# LLM-generated content at query #16
#--------------------------

```python
def test_FXRateService_query():
    from datetime import date
    from decimal import Decimal
    from unittest.mock import Mock, patch
    import pytest
    
    from pypara.currencies import Currency
    from pypara.fx import FXRate, FXRateLookupError, FXRateService
    
    # Create mock currencies
    eur = Mock(spec=Currency)
    usd = Mock(spec=Currency)
    gbp = Mock(spec=Currency)
    
    # Create a concrete implementation of FXRateService for testing
    class TestFXRateService(FXRateService):
        def __init__(self, rates=None):
            self.rates = rates or {}
            
        def query(self, ccy1, ccy2, asof, strict=False):
            key = (ccy1, ccy2, asof)
            if key in self.rates:
                return self.rates[key]
            elif strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None
            
        def queries(self, queries, strict=False):
            results = []
            for ccy1, ccy2, asof in queries:
                try:
                    results.append(self.query(ccy1, ccy2, asof, strict))
                except FXRateLookupError:
                    if strict:
                        raise
                    results.append(None)
            return results
    
    # Test 1: Query existing rate
    test_date = date(2023, 1, 1)
    expected_rate = FXRate(eur, usd, test_date, Decimal("1.1"))
    service = TestFXRateService({
        (eur, usd, test_date): expected_rate
    })
    
    result = service.query(eur, usd, test_date)
    assert result == expected_rate
    
    # Test 2: Query non-existing rate (non-strict mode)
    result = service.query(eur, gbp, test_date)
    assert result is None
    
    # Test 3: Query non-existing rate (strict mode)
    with pytest.raises(FXRateLookupError) as exc_info:
        service.query(eur, gbp, test_date, strict=True)
    assert exc_info.value.ccy1 == eur
    assert exc_info.value.ccy2 == gbp
    assert exc_info.value.asof == test_date
    
    # Test 4: Query with inverted rate available
    service.rates[(usd, eur, test_date)] = FXRate(usd, eur, test_date, Decimal("0.9091"))
    result = service.query(usd, eur, test_date)
    assert result.value == Decimal("0.9091")
    
    # Test 5: Query with different date
    other_date = date(2023, 1, 2)
    result = service.query(eur, usd, other_date)
    assert result is None
    
    # Test 6: Query same currency (should return None or raise)
    result = service.query(eur, eur, test_date)
    assert result is None
    
    # Test 7: Test with strict=True for same currency
    with pytest.raises(FXRateLookupError):
        service.query(eur, eur, test_date, strict=True)
    
    # Test 8: Test with multiple rates
    service.rates[(eur, gbp, test_date)] = FXRate(eur, gbp, test_date, Decimal("0.85"))
    service.rates[(gbp, usd, test_date)] = FXRate(gbp, usd, test_date, Decimal("1.3"))
    
    result1 = service.query(eur, gbp, test_date)
    result2 = service.query(gbp, usd, test_date)
    
    assert result1.value == Decimal("0.85")
    assert result2.value == Decimal("1.3")


# LLM-generated content at query #17
#--------------------------

```python
def test_FXRateService_query():
    # Create a concrete implementation for testing
    class TestFXRateService(FXRateService):
        def __init__(self, rates):
            self.rates = rates
            
        def query(self, ccy1, ccy2, asof, strict=False):
            key = (ccy1, ccy2, asof)
            if key in self.rates:
                return self.rates[key]
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None
            
        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    # Create test currencies
    class TestCurrency:
        def __init__(self, code):
            self.code = code
            
        def __eq__(self, other):
            return isinstance(other, TestCurrency) and self.code == other.code
            
        def __hash__(self):
            return hash(self.code)
            
        def __repr__(self):
            return f"Currency({self.code})"
    
    EUR = TestCurrency("EUR")
    USD = TestCurrency("USD")
    GBP = TestCurrency("GBP")
    
    # Create test dates
    date1 = Date(2023, 1, 1)
    date2 = Date(2023, 1, 2)
    
    # Create test rates
    rate1 = FXRate(EUR, USD, date1, Decimal("1.10"))
    rate2 = FXRate(EUR, USD, date2, Decimal("1.12"))
    rate3 = FXRate(EUR, GBP, date1, Decimal("0.88"))
    
    # Setup service with rates
    service = TestFXRateService({
        (EUR, USD, date1): rate1,
        (EUR, USD, date2): rate2,
        (EUR, GBP, date1): rate3,
    })
    
    # Test 1: Query existing rate
    result = service.query(EUR, USD, date1)
    assert result == rate1
    assert result.ccy1 == EUR
    assert result.ccy2 == USD
    assert result.date == date1
    assert result.value == Decimal("1.10")
    
    # Test 2: Query different date
    result = service.query(EUR, USD, date2)
    assert result == rate2
    assert result.value == Decimal("1.12")
    
    # Test 3: Query different currency pair
    result = service.query(EUR, GBP, date1)
    assert result == rate3
    assert result.value == Decimal("0.88")
    
    # Test 4: Query non-existent rate (non-strict)
    result = service.query(USD, GBP, date1)
    assert result is None
    
    # Test 5: Query non-existent rate (strict) - should raise FXRateLookupError
    try:
        service.query(USD, GBP, date1, strict=True)
        assert False, "Should have raised FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == USD
        assert e.ccy2 == GBP
        assert e.asof == date1
        assert str(e) == f"Foreign exchange rate for {USD}/{GBP} not found as of {date1}"
    
    # Test 6: Query inverted rate (should not find unless explicitly stored)
    result = service.query(USD, EUR, date1)
    assert result is None
    
    # Test 7: Query with same currency (should return None since not in stored rates)
    result = service.query(EUR, EUR, date1)
    assert result is None
    
    # Test 8: Test strict=False with non-existent rate doesn't raise error
    result = service.query(TestCurrency("JPY"), TestCurrency("CAD"), date1, strict=False)
    assert result is None


# LLM-generated content at query #18
#--------------------------

```python
def test_FXRateService_query():
    # Test 1: Test with a concrete implementation that returns a valid FXRate
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            from decimal import Decimal
            from datetime import date
            from pypara.currencies import Currencies
            
            if ccy1 == Currencies["EUR"] and ccy2 == Currencies["USD"] and asof == date(2023, 1, 1):
                return FXRate(ccy1, ccy2, asof, Decimal("1.1"))
            return None
        
        def queries(self, queries, strict=False):
            return []
    
    service = MockFXRateService()
    from datetime import date
    from pypara.currencies import Currencies
    
    # Test successful query
    rate = service.query(Currencies["EUR"], Currencies["USD"], date(2023, 1, 1))
    assert rate is not None
    assert rate.ccy1 == Currencies["EUR"]
    assert rate.ccy2 == Currencies["USD"]
    assert rate.date == date(2023, 1, 1)
    assert rate.value == Decimal("1.1")
    
    # Test query with no result (non-strict mode)
    rate = service.query(Currencies["USD"], Currencies["EUR"], date(2023, 1, 1))
    assert rate is None
    
    # Test 2: Test with strict mode raising FXRateLookupError
    class StrictMockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            from decimal import Decimal
            from datetime import date
            from pypara.currencies import Currencies
            
            if ccy1 == Currencies["EUR"] and ccy2 == Currencies["USD"] and asof == date(2023, 1, 1):
                return FXRate(ccy1, ccy2, asof, Decimal("1.1"))
            
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None
        
        def queries(self, queries, strict=False):
            return []
    
    strict_service = StrictMockFXRateService()
    
    # Test strict mode with found rate
    rate = strict_service.query(Currencies["EUR"], Currencies["USD"], date(2023, 1, 1), strict=True)
    assert rate is not None
    
    # Test strict mode with not found rate
    try:
        strict_service.query(Currencies["USD"], Currencies["EUR"], date(2023, 1, 1), strict=True)
        assert False, "Should have raised FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == Currencies["USD"]
        assert e.ccy2 == Currencies["EUR"]
        assert e.asof == date(2023, 1, 1)
    
    # Test 3: Test with same currency (should return rate with value 1)
    class SameCurrencyService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            from decimal import Decimal
            if ccy1 == ccy2:
                return FXRate(ccy1, ccy2, asof, Decimal("1"))
            return None
        
        def queries(self, queries, strict=False):
            return []
    
    same_service = SameCurrencyService()
    rate = same_service.query(Currencies["EUR"], Currencies["EUR"], date(2023, 1, 1))
    assert rate is not None
    assert rate.value == Decimal("1")


# LLM-generated content at query #19
#--------------------------

```python
def test_FXRateService_query():
    from decimal import Decimal
    from datetime import date
    from unittest.mock import Mock, patch
    import pytest
    
    # Mock Currency class
    class MockCurrency:
        def __init__(self, code):
            self.code = code
        
        def __eq__(self, other):
            return isinstance(other, MockCurrency) and self.code == other.code
    
    # Create mock currencies
    EUR = MockCurrency("EUR")
    USD = MockCurrency("USD")
    GBP = MockCurrency("GBP")
    
    # Create a concrete implementation of FXRateService for testing
    class TestFXRateService(FXRateService):
        def __init__(self, rates=None):
            self.rates = rates or {}
        
        def query(self, ccy1, ccy2, asof, strict=False):
            key = (ccy1, ccy2, asof)
            if key in self.rates:
                return self.rates[key]
            elif strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None
        
        def queries(self, queries, strict=False):
            results = []
            for query in queries:
                try:
                    results.append(self.query(*query, strict=strict))
                except FXRateLookupError:
                    if strict:
                        raise
                    results.append(None)
            return results
    
    # Test 1: Query existing rate
    test_date = date(2023, 1, 1)
    expected_rate = FXRate(EUR, USD, test_date, Decimal("1.1"))
    service = TestFXRateService({
        (EUR, USD, test_date): expected_rate
    })
    
    result = service.query(EUR, USD, test_date)
    assert result == expected_rate
    
    # Test 2: Query non-existing rate with strict=False (default)
    result = service.query(EUR, GBP, test_date)
    assert result is None
    
    # Test 3: Query non-existing rate with strict=True
    with pytest.raises(FXRateLookupError) as exc_info:
        service.query(EUR, GBP, test_date, strict=True)
    assert exc_info.value.ccy1 == EUR
    assert exc_info.value.ccy2 == GBP
    assert exc_info.value.asof == test_date
    
    # Test 4: Query inverted rate (should not find it since not in storage)
    result = service.query(USD, EUR, test_date)
    assert result is None
    
    # Test 5: Query with same currency (should return None if not explicitly stored)
    result = service.query(EUR, EUR, test_date)
    assert result is None
    
    # Test 6: Test with multiple rates
    another_date = date(2023, 1, 2)
    another_rate = FXRate(EUR, USD, another_date, Decimal("1.2"))
    service.rates[(EUR, USD, another_date)] = another_rate
    
    result1 = service.query(EUR, USD, test_date)
    result2 = service.query(EUR, USD, another_date)
    assert result1 == expected_rate
    assert result2 == another_rate
    
    # Test 7: Test with different currency pair
    gbp_rate = FXRate(EUR, GBP, test_date, Decimal("0.85"))
    service.rates[(EUR, GBP, test_date)] = gbp_rate
    
    result = service.query(EUR, GBP, test_date)
    assert result == gbp_rate
    
    # Test 8: Test with strict=False for non-existent rate
    result = service.query(USD, GBP, test_date)
    assert result is None


# LLM-generated content at query #20
#--------------------------

```python
def test_FXRateService_queries():
    from datetime import date
    from decimal import Decimal
    from typing import Optional, Iterable, Tuple
    from pypara.currencies import Currency
    from pypara.currencies import Currencies
    from pypara.fx import FXRate, FXRateService, FXRateLookupError

    class MockFXRateService(FXRateService):
        def __init__(self, rates: dict):
            self.rates = rates

        def query(self, ccy1: Currency, ccy2: Currency, asof: date, strict: bool = False) -> Optional[FXRate]:
            key = (ccy1, ccy2, asof)
            if key in self.rates:
                return self.rates[key]
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None

        def queries(self, queries: Iterable[Tuple[Currency, Currency, date]], strict: bool = False) -> Iterable[Optional[FXRate]]:
            results = []
            for ccy1, ccy2, asof in queries:
                try:
                    results.append(self.query(ccy1, ccy2, asof, strict))
                except FXRateLookupError:
                    if strict:
                        raise
                    results.append(None)
            return results

    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    gbp = Currencies["GBP"]
    d1 = date(2023, 1, 1)
    d2 = date(2023, 1, 2)

    rate1 = FXRate(eur, usd, d1, Decimal("1.1"))
    rate2 = FXRate(eur, gbp, d1, Decimal("0.9"))
    rate3 = FXRate(usd, gbp, d2, Decimal("0.8"))

    service = MockFXRateService({
        (eur, usd, d1): rate1,
        (eur, gbp, d1): rate2,
        (usd, gbp, d2): rate3,
    })

    queries = [
        (eur, usd, d1),
        (eur, gbp, d1),
        (usd, gbp, d2),
        (gbp, eur, d1),
    ]

    results = list(service.queries(queries, strict=False))
    assert results[0] == rate1
    assert results[1] == rate2
    assert results[2] == rate3
    assert results[3] is None

    results_strict = list(service.queries(queries[:3], strict=True))
    assert results_strict[0] == rate1
    assert results_strict[1] == rate2
    assert results_strict[2] == rate3

    try:
        list(service.queries(queries, strict=True))
        assert False, "Should have raised FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == gbp
        assert e.ccy2 == eur
        assert e.asof == d1

    empty_results = list(service.queries([], strict=False))
    assert empty_results == []


# LLM-generated content at query #21
#--------------------------

```python
def test_FXRateService_query():
    from decimal import Decimal
    from datetime import date
    from unittest.mock import Mock, patch
    from pypara.currencies import Currencies
    from pypara.currencies import Currency

    # Create a concrete implementation for testing
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            # Simple test implementation that returns a fixed rate for EUR/USD
            if ccy1 == Currencies["EUR"] and ccy2 == Currencies["USD"] and asof == date(2023, 1, 1):
                return FXRate(ccy1, ccy2, asof, Decimal("1.1"))
            elif strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None
        
        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    # Test 1: Successful query
    service = TestFXRateService()
    rate = service.query(Currencies["EUR"], Currencies["USD"], date(2023, 1, 1))
    assert rate is not None
    assert rate.ccy1 == Currencies["EUR"]
    assert rate.ccy2 == Currencies["USD"]
    assert rate.date == date(2023, 1, 1)
    assert rate.value == Decimal("1.1")

    # Test 2: Non-strict query returns None for missing rate
    rate = service.query(Currencies["USD"], Currencies["EUR"], date(2023, 1, 1))
    assert rate is None

    # Test 3: Strict query raises FXRateLookupError for missing rate
    try:
        service.query(Currencies["USD"], Currencies["EUR"], date(2023, 1, 1), strict=True)
        assert False, "Should have raised FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == Currencies["USD"]
        assert e.ccy2 == Currencies["EUR"]
        assert e.asof == date(2023, 1, 1)

    # Test 4: Test with same currency (should return 1:1 rate)
    # First create a mock service that handles same currency
    class SameCurrencyService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if ccy1 == ccy2:
                return FXRate(ccy1, ccy2, asof, Decimal("1"))
            return None
        
        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    same_service = SameCurrencyService()
    rate = same_service.query(Currencies["EUR"], Currencies["EUR"], date(2023, 1, 1))
    assert rate is not None
    assert rate.value == Decimal("1")

    # Test 5: Test inverted rate using ~ operator
    rate = service.query(Currencies["EUR"], Currencies["USD"], date(2023, 1, 1))
    inverted = ~rate
    assert inverted.ccy1 == Currencies["USD"]
    assert inverted.ccy2 == Currencies["EUR"]
    assert inverted.value == Decimal("1") / Decimal("1.1")

    # Test 6: Test with different date
    rate = service.query(Currencies["EUR"], Currencies["USD"], date(2023, 1, 2))
    assert rate is None

    # Test 7: Test queries method integration
    queries = [
        (Currencies["EUR"], Currencies["USD"], date(2023, 1, 1)),
        (Currencies["USD"], Currencies["EUR"], date(2023, 1, 1)),
    ]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] is not None
    assert results[1] is None


# LLM-generated content at query #22
#--------------------------

```python
def test_FXRateService_queries():
    from decimal import Decimal
    from datetime import date
    from unittest.mock import Mock, call

    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None

        def queries(self, queries, strict=False):
            results = []
            for ccy1, ccy2, asof in queries:
                if ccy1.code == "EUR" and ccy2.code == "USD" and asof == date(2023, 1, 1):
                    results.append(FXRate(ccy1, ccy2, asof, Decimal("1.1")))
                elif ccy1.code == "USD" and ccy2.code == "EUR" and asof == date(2023, 1, 1):
                    results.append(FXRate(ccy1, ccy2, asof, Decimal("0.9")))
                elif strict:
                    raise FXRateLookupError(ccy1, ccy2, asof)
                else:
                    results.append(None)
            return results

    eur = Mock(code="EUR")
    usd = Mock(code="USD")
    gbp = Mock(code="GBP")

    service = ConcreteFXRateService()

    queries = [
        (eur, usd, date(2023, 1, 1)),
        (usd, eur, date(2023, 1, 1)),
        (gbp, eur, date(2023, 1, 1)),
    ]

    results = list(service.queries(queries, strict=False))
    assert len(results) == 3
    assert results[0] == FXRate(eur, usd, date(2023, 1, 1), Decimal("1.1"))
    assert results[1] == FXRate(usd, eur, date(2023, 1, 1), Decimal("0.9"))
    assert results[2] is None

    try:
        list(service.queries(queries, strict=True))
        assert False, "Should have raised FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == gbp
        assert e.ccy2 == eur
        assert e.asof == date(2023, 1, 1)

    empty_results = list(service.queries([], strict=False))
    assert empty_results == []


# LLM-generated content at query #23
#--------------------------

```python
def test_FXRateService_query():
    from unittest.mock import Mock, patch
    from decimal import Decimal
    import datetime
    from pypara.currencies import Currencies

    # Create a concrete implementation for testing
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            # Simple implementation that returns a fixed rate for EUR/USD
            if ccy1 == Currencies["EUR"] and ccy2 == Currencies["USD"] and asof == datetime.date(2023, 1, 1):
                return FXRate(ccy1, ccy2, asof, Decimal("1.1"))
            elif strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None

        def queries(self, queries, strict=False):
            return [self.query(q[0], q[1], q[2], strict) for q in queries]

    # Test 1: Successful query
    service = ConcreteFXRateService()
    rate = service.query(Currencies["EUR"], Currencies["USD"], datetime.date(2023, 1, 1))
    assert rate is not None
    assert rate.ccy1 == Currencies["EUR"]
    assert rate.ccy2 == Currencies["USD"]
    assert rate.date == datetime.date(2023, 1, 1)
    assert rate.value == Decimal("1.1")

    # Test 2: Non-strict query returns None for missing rate
    rate = service.query(Currencies["USD"], Currencies["EUR"], datetime.date(2023, 1, 1))
    assert rate is None

    # Test 3: Strict query raises FXRateLookupError for missing rate
    try:
        service.query(Currencies["USD"], Currencies["EUR"], datetime.date(2023, 1, 1), strict=True)
        assert False, "Should have raised FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == Currencies["USD"]
        assert e.ccy2 == Currencies["EUR"]
        assert e.asof == datetime.date(2023, 1, 1)

    # Test 4: Test with same currency (should return 1:1 rate if implemented)
    # Note: This depends on the actual implementation

    # Test 5: Test with mocked abstract method
    with patch.object(FXRateService, '__abstractmethods__', set()):
        mock_service = Mock(spec=FXRateService)
        mock_rate = FXRate(Currencies["GBP"], Currencies["USD"], datetime.date(2023, 1, 1), Decimal("1.3"))
        mock_service.query.return_value = mock_rate
        
        result = mock_service.query(Currencies["GBP"], Currencies["USD"], datetime.date(2023, 1, 1))
        assert result == mock_rate
        mock_service.query.assert_called_once_with(Currencies["GBP"], Currencies["USD"], datetime.date(2023, 1, 1), False)


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_FXRateService_queries():
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            if ccy1.code == "EUR" and ccy2.code == "USD" and asof == Date(2023, 1, 1):
                return FXRate(ccy1, ccy2, asof, Decimal("1.1"))
            if ccy1.code == "USD" and ccy2.code == "EUR" and asof == Date(2023, 1, 1):
                return FXRate(ccy1, ccy2, asof, Decimal("0.9091"))
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    from pypara.currencies import Currencies
    service = MockFXRateService()
    
    queries = [
        (Currencies["EUR"], Currencies["USD"], Date(2023, 1, 1)),
        (Currencies["USD"], Currencies["EUR"], Date(2023, 1, 1)),
        (Currencies["GBP"], Currencies["JPY"], Date(2023, 1, 1)),
    ]
    
    results = list(service.queries(queries, strict=False))
    assert len(results) == 3
    assert results[0] == FXRate(Currencies["EUR"], Currencies["USD"], Date(2023, 1, 1), Decimal("1.1"))
    assert results[1] == FXRate(Currencies["USD"], Currencies["EUR"], Date(2023, 1, 1), Decimal("0.9091"))
    assert results[2] is None
    
    with pytest.raises(FXRateLookupError) as exc_info:
        list(service.queries(queries, strict=True))
    assert exc_info.value.ccy1 == Currencies["GBP"]
    assert exc_info.value.ccy2 == Currencies["JPY"]
    assert exc_info.value.asof == Date(2023, 1, 1)
    
    empty_results = list(service.queries([], strict=False))
    assert empty_results == []


# LLM-generated content at query #2
#--------------------------

```python
def test_FXRateService_query():
    # Mock implementation for testing
    class MockFXRateService(FXRateService):
        def __init__(self, rates):
            self.rates = rates
            
        def query(self, ccy1, ccy2, asof, strict=False):
            key = (ccy1, ccy2, asof)
            if key in self.rates:
                return self.rates[key]
            elif strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None
            
        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    # Test data
    from decimal import Decimal
    import datetime
    
    class MockCurrency:
        def __init__(self, code):
            self.code = code
        
        def __eq__(self, other):
            return isinstance(other, MockCurrency) and self.code == other.code
        
        def __hash__(self):
            return hash(self.code)
    
    eur = MockCurrency("EUR")
    usd = MockCurrency("USD")
    gbp = MockCurrency("GBP")
    date1 = datetime.date(2023, 1, 1)
    date2 = datetime.date(2023, 1, 2)
    
    # Create test rates
    rate1 = FXRate(eur, usd, date1, Decimal("1.1"))
    rate2 = FXRate(eur, gbp, date1, Decimal("0.9"))
    rate3 = FXRate(usd, gbp, date2, Decimal("0.8"))
    
    rates = {
        (eur, usd, date1): rate1,
        (eur, gbp, date1): rate2,
        (usd, gbp, date2): rate3,
    }
    
    service = MockFXRateService(rates)
    
    # Test 1: Query existing rate
    result = service.query(eur, usd, date1)
    assert result == rate1
    assert result.ccy1 == eur
    assert result.ccy2 == usd
    assert result.date == date1
    assert result.value == Decimal("1.1")
    
    # Test 2: Query non-existing rate with strict=False (default)
    result = service.query(eur, usd, date2)
    assert result is None
    
    # Test 3: Query non-existing rate with strict=True
    try:
        service.query(eur, usd, date2, strict=True)
        assert False, "Should have raised FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == eur
        assert e.ccy2 == usd
        assert e.asof == date2
    
    # Test 4: Query inverted rate (should not find it since not in dictionary)
    result = service.query(usd, eur, date1)
    assert result is None
    
    # Test 5: Query with same currency (should not be in rates, returns None)
    result = service.query(eur, eur, date1)
    assert result is None
    
    # Test 6: Query multiple different rates
    result1 = service.query(eur, usd, date1)
    result2 = service.query(eur, gbp, date1)
    result3 = service.query(usd, gbp, date2)
    
    assert result1 == rate1
    assert result2 == rate2
    assert result3 == rate3
    
    # Test 7: Test queries method integration
    queries = [(eur, usd, date1), (eur, gbp, date1), (usd, gbp, date2), (eur, usd, date2)]
    results = list(service.queries(queries))
    assert results == [rate1, rate2, rate3, None]
    
    # Test 8: Test queries method with strict=True
    try:
        list(service.queries(queries, strict=True))
        assert False, "Should have raised FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == eur
        assert e.ccy2 == usd
        assert e.asof == date2


# LLM-generated content at query #3
#--------------------------

```python
def test_FXRate___invert__():
    from datetime import date
    from decimal import Decimal
    from pypara.currencies import Currencies
    
    # Test basic inversion
    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    test_date = date(2023, 1, 1)
    rate_value = Decimal("1.1")
    
    original = FXRate(eur, usd, test_date, rate_value)
    inverted = ~original
    
    assert inverted.ccy1 == usd
    assert inverted.ccy2 == eur
    assert inverted.date == test_date
    assert inverted.value == Decimal("1") / rate_value
    
    # Test inversion of inversion returns original
    assert ~~original == original
    
    # Test with different values
    rate2 = FXRate(Currencies["GBP"], Currencies["JPY"], date(2023, 2, 1), Decimal("150.5"))
    inverted2 = ~rate2
    assert inverted2.ccy1 == Currencies["JPY"]
    assert inverted2.ccy2 == Currencies["GBP"]
    assert inverted2.value == Decimal("1") / Decimal("150.5")
    
    # Test that inversion works correctly with NamedTuple indexing
    assert inverted[0] == usd
    assert inverted[1] == eur
    assert inverted[2] == test_date
    assert inverted[3] == Decimal("1") / rate_value


# LLM-generated content at query #4
#--------------------------

```python
def test_FXRateService_query():
    from decimal import Decimal
    from datetime import date
    from unittest.mock import Mock, patch
    import pytest
    
    # Create mock currencies
    ccy1 = Mock(spec=Currency)
    ccy2 = Mock(spec=Currency)
    asof_date = date(2023, 1, 1)
    
    # Create a concrete implementation of FXRateService for testing
    class TestFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            # Return a mock FXRate for testing
            if ccy1 == ccy2:
                return FXRate(ccy1, ccy2, asof, ONE)
            return FXRate(ccy1, ccy2, asof, Decimal("1.5"))
        
        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(q[0], q[1], q[2], strict) for q in queries]
    
    # Test 1: Query with different currencies
    service = TestFXRateService()
    result = service.query(ccy1, ccy2, asof_date)
    
    assert result is not None
    assert result.ccy1 == ccy1
    assert result.ccy2 == ccy2
    assert result.date == asof_date
    assert result.value == Decimal("1.5")
    
    # Test 2: Query with same currencies (should return rate of 1)
    result_same = service.query(ccy1, ccy1, asof_date)
    assert result_same is not None
    assert result_same.ccy1 == ccy1
    assert result_same.ccy2 == ccy1
    assert result_same.value == ONE
    
    # Test 3: Test with strict=False (should not raise error even if rate not found)
    class FailingFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None
        
        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(q[0], q[1], q[2], strict) for q in queries]
    
    failing_service = FailingFXRateService()
    result_none = failing_service.query(ccy1, ccy2, asof_date, strict=False)
    assert result_none is None
    
    # Test 4: Test with strict=True (should raise FXRateLookupError)
    with pytest.raises(FXRateLookupError) as exc_info:
        failing_service.query(ccy1, ccy2, asof_date, strict=True)
    
    assert exc_info.value.ccy1 == ccy1
    assert exc_info.value.ccy2 == ccy2
    assert exc_info.value.asof == asof_date
    assert str(exc_info.value) == f"Foreign exchange rate for {ccy1}/{ccy2} not found as of {asof_date}"
    
    # Test 5: Test query method signature compliance
    assert hasattr(service, 'query')
    assert callable(service.query)
    
    # Test 6: Test with inverted rate using __invert__ method
    if result is not None:
        inverted = ~result
        assert inverted.ccy1 == ccy2
        assert inverted.ccy2 == ccy1
        assert inverted.date == asof_date
        assert inverted.value == Decimal("1") / Decimal("1.5")


# LLM-generated content at query #5
#--------------------------

```python
def test_FXRateService_query():
    # Create a concrete implementation for testing
    class MockFXRateService(FXRateService):
        def __init__(self, rates):
            self.rates = rates
            
        def query(self, ccy1, ccy2, asof, strict=False):
            key = (ccy1, ccy2, asof)
            if key in self.rates:
                return self.rates[key]
            elif strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None
            
        def queries(self, queries, strict=False):
            results = []
            for query in queries:
                try:
                    results.append(self.query(*query, strict=strict))
                except FXRateLookupError:
                    if strict:
                        raise
                    results.append(None)
            return results

    # Create test currencies
    class TestCurrency:
        def __init__(self, code):
            self.code = code
            
        def __eq__(self, other):
            return isinstance(other, TestCurrency) and self.code == other.code
            
        def __hash__(self):
            return hash(self.code)
            
        def __repr__(self):
            return f"Currency({self.code})"

    EUR = TestCurrency("EUR")
    USD = TestCurrency("USD")
    GBP = TestCurrency("GBP")
    
    # Create test dates
    date1 = Date(2023, 1, 1)
    date2 = Date(2023, 1, 2)
    
    # Create test FX rates
    rate1 = FXRate(EUR, USD, date1, Decimal("1.1"))
    rate2 = FXRate(EUR, USD, date2, Decimal("1.2"))
    rate3 = FXRate(USD, GBP, date1, Decimal("0.8"))
    
    # Setup mock service with rates
    rates_dict = {
        (EUR, USD, date1): rate1,
        (EUR, USD, date2): rate2,
        (USD, GBP, date1): rate3,
    }
    service = MockFXRateService(rates_dict)
    
    # Test 1: Query existing rate (non-strict mode)
    result = service.query(EUR, USD, date1)
    assert result == rate1
    assert result.ccy1 == EUR
    assert result.ccy2 == USD
    assert result.date == date1
    assert result.value == Decimal("1.1")
    
    # Test 2: Query existing rate (strict mode)
    result = service.query(EUR, USD, date2, strict=True)
    assert result == rate2
    
    # Test 3: Query non-existing rate (non-strict mode)
    result = service.query(GBP, EUR, date1)
    assert result is None
    
    # Test 4: Query non-existing rate (strict mode) - should raise
    try:
        service.query(GBP, EUR, date1, strict=True)
        assert False, "Should have raised FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == GBP
        assert e.ccy2 == EUR
        assert e.asof == date1
    
    # Test 5: Query with inverted currency pair
    result = service.query(USD, EUR, date1)
    assert result is None  # Not in our mock data
    
    # Test 6: Query with same currency (should return None if not in data)
    result = service.query(EUR, EUR, date1)
    assert result is None
    
    # Test 7: Verify FXRate inversion works
    inverted_rate = ~rate1
    assert inverted_rate.ccy1 == USD
    assert inverted_rate.ccy2 == EUR
    assert inverted_rate.date == date1
    assert inverted_rate.value == Decimal("1") / Decimal("1.1")
    
    # Test 8: Multiple queries with queries method
    queries = [(EUR, USD, date1), (USD, GBP, date1), (GBP, EUR, date1)]
    results = list(service.queries(queries))
    assert results == [rate1, rate3, None]
    
    # Test 9: Multiple queries with strict mode
    try:
        list(service.queries(queries, strict=True))
        assert False, "Should have raised FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == GBP
        assert e.ccy2 == EUR
    
    # Test 10: Test FXRate.of validation
    valid_rate = FXRate.of(EUR, USD, date1, Decimal("1.5"))
    assert valid_rate.ccy1 == EUR
    assert valid_rate.ccy2 == USD
    assert valid_rate.date == date1
    assert valid_rate.value == Decimal("1.5")
    
    # Test 11: Test FXRate.of with same currency
    same_ccy_rate = FXRate.of(EUR, EUR, date1, ONE)
    assert same_ccy_rate.value == ONE
    
    # Test 12: Test FXRate.of with invalid value (zero)
    try:
        FXRate.of(EUR, USD, date1, ZERO)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "FX rate value can not be equal to or less than `zero`" in str(e)
    
    # Test 13: Test FXRate.of with invalid value (negative)
    try:
        FXRate.of(EUR, USD, date1, Decimal("-1.0"))
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "FX rate value can not be equal to or less than `zero`" in str(e)
    
    # Test 14: Test FXRate.of with same currency but not one
    try:
        FXRate.of(EUR, EUR, date1, Decimal("2.0"))
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "FX rate to the same currency must be `one`" in str(e)


# LLM-generated content at query #6
#--------------------------

```python
def test_FXRateService_queries():
    class MockFXRateService(FXRateService):
        def __init__(self, rates):
            self.rates = rates
        
        def query(self, ccy1, ccy2, asof, strict=False):
            key = (ccy1, ccy2, asof)
            if key in self.rates:
                return self.rates[key]
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None
        
        def queries(self, queries, strict=False):
            results = []
            for ccy1, ccy2, asof in queries:
                try:
                    rate = self.query(ccy1, ccy2, asof, strict)
                    results.append(rate)
                except FXRateLookupError:
                    results.append(None)
            return results
    
    from decimal import Decimal
    from pypara.currencies import Currencies
    from pypara.commons.zeitgeist import Date
    
    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    gbp = Currencies["GBP"]
    date1 = Date(2023, 1, 1)
    date2 = Date(2023, 1, 2)
    
    rate1 = FXRate(eur, usd, date1, Decimal("1.1"))
    rate2 = FXRate(eur, gbp, date1, Decimal("0.9"))
    rate3 = FXRate(usd, gbp, date2, Decimal("0.8"))
    
    rates = {
        (eur, usd, date1): rate1,
        (eur, gbp, date1): rate2,
        (usd, gbp, date2): rate3,
    }
    
    service = MockFXRateService(rates)
    
    queries = [
        (eur, usd, date1),
        (eur, gbp, date1),
        (usd, gbp, date2),
        (usd, eur, date1),
    ]
    
    results = list(service.queries(queries, strict=False))
    assert results[0] == rate1
    assert results[1] == rate2
    assert results[2] == rate3
    assert results[3] is None
    
    results_strict = list(service.queries(queries[:3], strict=True))
    assert results_strict[0] == rate1
    assert results_strict[1] == rate2
    assert results_strict[2] == rate3
    
    try:
        list(service.queries(queries, strict=True))
        assert False, "Should have raised FXRateLookupError"
    except FXRateLookupError:
        pass
    
    empty_results = list(service.queries([], strict=False))
    assert empty_results == []
    
    single_query = [(eur, usd, date1)]
    single_result = list(service.queries(single_query, strict=False))
    assert single_result == [rate1]
    
    inverted_rate = ~rate1
    inverted_query = [(usd, eur, date1)]
    inverted_result = list(service.queries(inverted_query, strict=False))
    assert inverted_result[0] is None


# LLM-generated content at query #7
#--------------------------

```python
def test_FXRateService_query():
    # Create a concrete implementation for testing
    class TestFXRateService(FXRateService):
        def __init__(self, rates):
            self.rates = rates
        
        def query(self, ccy1, ccy2, asof, strict=False):
            key = (ccy1, ccy2, asof)
            if key in self.rates:
                return self.rates[key]
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None
        
        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    # Create test currencies
    class TestCurrency:
        def __init__(self, code):
            self.code = code
        
        def __eq__(self, other):
            return isinstance(other, TestCurrency) and self.code == other.code
        
        def __hash__(self):
            return hash(self.code)
    
    # Create test data
    eur = TestCurrency("EUR")
    usd = TestCurrency("USD")
    gbp = TestCurrency("GBP")
    date1 = Date(2023, 1, 1)
    date2 = Date(2023, 1, 2)
    
    # Create test rates
    rate1 = FXRate(eur, usd, date1, Decimal("1.1"))
    rate2 = FXRate(eur, gbp, date1, Decimal("0.9"))
    rate3 = FXRate(usd, gbp, date2, Decimal("0.8"))
    
    # Create service with test rates
    service = TestFXRateService({
        (eur, usd, date1): rate1,
        (eur, gbp, date1): rate2,
        (usd, gbp, date2): rate3,
    })
    
    # Test 1: Query existing rate
    result = service.query(eur, usd, date1)
    assert result == rate1
    assert result.ccy1 == eur
    assert result.ccy2 == usd
    assert result.date == date1
    assert result.value == Decimal("1.1")
    
    # Test 2: Query non-existing rate without strict mode
    result = service.query(usd, eur, date1)
    assert result is None
    
    # Test 3: Query non-existing rate with strict mode
    try:
        service.query(usd, eur, date1, strict=True)
        assert False, "Should have raised FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == usd
        assert e.ccy2 == eur
        assert e.asof == date1
    
    # Test 4: Query existing rate with different date
    result = service.query(usd, gbp, date2)
    assert result == rate3
    
    # Test 5: Query non-existing date
    result = service.query(eur, usd, date2)
    assert result is None
    
    # Test 6: Test rate inversion
    inverted = ~rate1
    assert inverted.ccy1 == usd
    assert inverted.ccy2 == eur
    assert inverted.date == date1
    assert inverted.value == Decimal("1") / Decimal("1.1")
    
    # Test 7: Test FXRate.of validation
    valid_rate = FXRate.of(eur, usd, date1, Decimal("1.5"))
    assert valid_rate.ccy1 == eur
    assert valid_rate.ccy2 == usd
    assert valid_rate.date == date1
    assert valid_rate.value == Decimal("1.5")
    
    # Test 8: Test same currency rate
    same_ccy_rate = FXRate.of(eur, eur, date1, Decimal("1"))
    assert same_ccy_rate.ccy1 == eur
    assert same_ccy_rate.ccy2 == eur
    assert same_ccy_rate.value == Decimal("1")


# LLM-generated content at query #8
#--------------------------

```python
def test_FXRateService_queries():
    from decimal import Decimal
    from datetime import date
    from typing import List, Optional
    from unittest.mock import Mock, call

    class MockFXRateService(FXRateService):
        def __init__(self, rates: List[Optional[FXRate]]):
            self.rates = rates
            self.query_calls = []

        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            self.query_calls.append((ccy1, ccy2, asof, strict))
            if self.rates:
                return self.rates.pop(0)
            return None

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            results = []
            for ccy1, ccy2, asof in queries:
                try:
                    rate = self.query(ccy1, ccy2, asof, strict)
                    results.append(rate)
                except FXRateLookupError:
                    results.append(None)
            return results

    eur = Mock(spec=Currency)
    usd = Mock(spec=Currency)
    gbp = Mock(spec=Currency)
    date1 = date(2023, 1, 1)
    date2 = date(2023, 1, 2)

    rate1 = FXRate(eur, usd, date1, Decimal("1.1"))
    rate2 = FXRate(usd, gbp, date2, Decimal("0.8"))
    rate3 = None

    service = MockFXRateService([rate1, rate2, rate3])
    queries = [(eur, usd, date1), (usd, gbp, date2), (gbp, eur, date1)]
    
    results = list(service.queries(queries, strict=False))
    
    assert results == [rate1, rate2, rate3]
    assert service.query_calls == [
        (eur, usd, date1, False),
        (usd, gbp, date2, False),
        (gbp, eur, date1, False)
    ]

    service_with_error = MockFXRateService([rate1])
    service_with_error.query = Mock(side_effect=FXRateLookupError(eur, usd, date1))
    
    results_with_error = list(service_with_error.queries([(eur, usd, date1)], strict=False))
    assert results_with_error == [None]

    empty_service = MockFXRateService([])
    empty_results = list(empty_service.queries([], strict=True))
    assert empty_results == []


# LLM-generated content at query #9
#--------------------------

```python
def test_FXRateService_query():
    from decimal import Decimal
    from datetime import date
    from unittest.mock import Mock, patch
    from pypara.currencies import Currency
    
    # Create mock currencies
    eur = Mock(spec=Currency)
    usd = Mock(spec=Currency)
    gbp = Mock(spec=Currency)
    
    # Create test date
    test_date = date(2023, 1, 1)
    
    # Create a concrete implementation of FXRateService for testing
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            # Simple implementation for testing
            if ccy1 == eur and ccy2 == usd and asof == test_date:
                return FXRate(eur, usd, test_date, Decimal("1.1"))
            elif ccy1 == usd and ccy2 == eur and asof == test_date:
                return FXRate(usd, eur, test_date, Decimal("0.9091"))
            elif strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None
        
        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    # Test 1: Normal query returns FXRate
    service = TestFXRateService()
    rate = service.query(eur, usd, test_date)
    assert rate is not None
    assert rate.ccy1 == eur
    assert rate.ccy2 == usd
    assert rate.date == test_date
    assert rate.value == Decimal("1.1")
    
    # Test 2: Inverted query returns correct FXRate
    inverted_rate = service.query(usd, eur, test_date)
    assert inverted_rate is not None
    assert inverted_rate.ccy1 == usd
    assert inverted_rate.ccy2 == eur
    assert inverted_rate.value == Decimal("0.9091")
    
    # Test 3: Non-strict query for non-existent rate returns None
    non_existent_rate = service.query(eur, gbp, test_date)
    assert non_existent_rate is None
    
    # Test 4: Strict query for non-existent rate raises FXRateLookupError
    try:
        service.query(eur, gbp, test_date, strict=True)
        assert False, "Should have raised FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == eur
        assert e.ccy2 == gbp
        assert e.asof == test_date
    
    # Test 5: Query with same currency returns None (not implemented in test service)
    # This tests that the service handles same-currency queries
    same_currency_rate = service.query(eur, eur, test_date)
    assert same_currency_rate is None
    
    # Test 6: Query with different date returns None
    different_date = date(2023, 1, 2)
    different_date_rate = service.query(eur, usd, different_date)
    assert different_date_rate is None
    
    # Test 7: Verify FXRate inversion works with service results
    rate = service.query(eur, usd, test_date)
    inverted = ~rate
    assert inverted.ccy1 == usd
    assert inverted.ccy2 == eur
    assert inverted.date == test_date
    assert inverted.value == Decimal("1") / Decimal("1.1")


# LLM-generated content at query #10
#--------------------------

```python
def test_FXRateService_queries():
    from decimal import Decimal
    from datetime import date
    from unittest.mock import Mock, call

    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None

        def queries(self, queries, strict=False):
            results = []
            for ccy1, ccy2, asof in queries:
                if ccy1.code == "EUR" and ccy2.code == "USD" and asof == date(2023, 1, 1):
                    results.append(FXRate(ccy1, ccy2, asof, Decimal("1.1")))
                elif ccy1.code == "USD" and ccy2.code == "EUR" and asof == date(2023, 1, 1):
                    results.append(FXRate(ccy1, ccy2, asof, Decimal("0.9")))
                elif strict:
                    raise FXRateLookupError(ccy1, ccy2, asof)
                else:
                    results.append(None)
            return results

    eur = Mock(spec=Currency, code="EUR")
    usd = Mock(spec=Currency, code="USD")
    gbp = Mock(spec=Currency, code="GBP")

    service = TestFXRateService()

    queries = [
        (eur, usd, date(2023, 1, 1)),
        (usd, eur, date(2023, 1, 1)),
        (eur, gbp, date(2023, 1, 1)),
    ]

    results = list(service.queries(queries, strict=False))
    assert len(results) == 3
    assert results[0] == FXRate(eur, usd, date(2023, 1, 1), Decimal("1.1"))
    assert results[1] == FXRate(usd, eur, date(2023, 1, 1), Decimal("0.9"))
    assert results[2] is None

    with pytest.raises(FXRateLookupError) as exc_info:
        list(service.queries(queries, strict=True))
    assert exc_info.value.ccy1 == eur
    assert exc_info.value.ccy2 == gbp
    assert exc_info.value.asof == date(2023, 1, 1)

    empty_results = list(service.queries([], strict=False))
    assert empty_results == []


# LLM-generated content at query #11
#--------------------------

```python
def test_FXRateService_queries():
    from datetime import date
    from decimal import Decimal
    from typing import List, Optional
    from unittest.mock import Mock, call

    class ConcreteFXRateService(FXRateService):
        def __init__(self, rates: List[Optional[FXRate]]):
            self.rates = rates
            self.query_calls = []

        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            self.query_calls.append((ccy1, ccy2, asof, strict))
            return self.rates.pop(0) if self.rates else None

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    eur = Mock(spec=Currency)
    usd = Mock(spec=Currency)
    gbp = Mock(spec=Currency)
    date1 = date(2023, 1, 1)
    date2 = date(2023, 1, 2)

    rate1 = FXRate(eur, usd, date1, Decimal("1.1"))
    rate2 = FXRate(usd, gbp, date2, Decimal("0.8"))
    rate3 = None

    service = ConcreteFXRateService([rate1, rate2, rate3])
    queries = [(eur, usd, date1), (usd, gbp, date2), (gbp, eur, date1)]

    results = list(service.queries(queries, strict=False))

    assert results == [rate1, rate2, rate3]
    assert service.query_calls == [
        (eur, usd, date1, False),
        (usd, gbp, date2, False),
        (gbp, eur, date1, False),
    ]

    service = ConcreteFXRateService([rate1, rate2])
    results = list(service.queries(queries[:2], strict=True))
    assert results == [rate1, rate2]

    class ErrorFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            for ccy1, ccy2, asof in queries:
                try:
                    yield self.query(ccy1, ccy2, asof, strict)
                except FXRateLookupError:
                    yield None

    error_service = ErrorFXRateService()
    results = list(error_service.queries([(eur, usd, date1)], strict=True))
    assert results == [None]

    class EmptyService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            return None

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return []

    empty_service = EmptyService()
    results = list(empty_service.queries([]))
    assert results == []


# LLM-generated content at query #12
#--------------------------

```python
def test_FXRateService_queries():
    from datetime import date
    from decimal import Decimal
    from typing import Optional, Iterable
    from unittest.mock import Mock, call

    class MockFXRateService(FXRateService):
        def __init__(self, rates: dict):
            self.rates = rates
            self.query_calls = []

        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            self.query_calls.append((ccy1, ccy2, asof, strict))
            return self.rates.get((ccy1, ccy2, asof))

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    class MockCurrency:
        def __init__(self, code: str):
            self.code = code

        def __eq__(self, other):
            return isinstance(other, MockCurrency) and self.code == other.code

        def __hash__(self):
            return hash(self.code)

        def __repr__(self):
            return f"Currency({self.code})"

    EUR = MockCurrency("EUR")
    USD = MockCurrency("USD")
    GBP = MockCurrency("GBP")
    JPY = MockCurrency("JPY")

    date1 = date(2023, 1, 1)
    date2 = date(2023, 1, 2)
    date3 = date(2023, 1, 3)

    rate1 = FXRate(EUR, USD, date1, Decimal("1.1"))
    rate2 = FXRate(USD, GBP, date2, Decimal("0.8"))
    rate3 = FXRate(EUR, JPY, date3, Decimal("130.5"))

    service = MockFXRateService({
        (EUR, USD, date1): rate1,
        (USD, GBP, date2): rate2,
        (EUR, JPY, date3): rate3,
    })

    queries = [
        (EUR, USD, date1),
        (USD, GBP, date2),
        (EUR, JPY, date3),
        (GBP, JPY, date1),
    ]

    results = list(service.queries(queries, strict=False))

    assert results == [rate1, rate2, rate3, None]
    assert service.query_calls == [
        (EUR, USD, date1, False),
        (USD, GBP, date2, False),
        (EUR, JPY, date3, False),
        (GBP, JPY, date1, False),
    ]

    service_strict = MockFXRateService({
        (EUR, USD, date1): rate1,
        (USD, GBP, date2): rate2,
    })

    queries_strict = [
        (EUR, USD, date1),
        (USD, GBP, date2),
        (EUR, JPY, date3),
    ]

    try:
        list(service_strict.queries(queries_strict, strict=True))
        assert False, "Should have raised FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == EUR
        assert e.ccy2 == JPY
        assert e.asof == date3

    empty_service = MockFXRateService({})
    empty_results = list(empty_service.queries([], strict=False))
    assert empty_results == []
    assert empty_service.query_calls == []

    single_query = [(EUR, USD, date1)]
    single_results = list(service.queries(single_query, strict=False))
    assert single_results == [rate1]
    assert len(service.query_calls) == 5

    duplicate_queries = [
        (EUR, USD, date1),
        (EUR, USD, date1),
        (USD, GBP, date2),
    ]
    duplicate_results = list(service.queries(duplicate_queries, strict=False))
    assert duplicate_results == [rate1, rate1, rate2]
    assert len(service.query_calls) == 8


# LLM-generated content at query #13
#--------------------------

```python
def test_FXRateService_query():
    # Test 1: Test with a concrete implementation that returns a valid FXRate
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            from decimal import Decimal
            from datetime import date
            from pypara.currencies import Currency
            
            # Return a mock FXRate for EUR/USD on 2023-01-01
            if (str(ccy1) == "EUR" and str(ccy2) == "USD" and 
                asof == date(2023, 1, 1)):
                return FXRate(ccy1, ccy2, asof, Decimal("1.1"))
            return None
        
        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) 
                    for ccy1, ccy2, asof in queries]
    
    service = MockFXRateService()
    from datetime import date
    from pypara.currencies import Currencies
    
    # Test valid query
    rate = service.query(Currencies["EUR"], Currencies["USD"], date(2023, 1, 1))
    assert rate is not None
    assert rate.ccy1 == Currencies["EUR"]
    assert rate.ccy2 == Currencies["USD"]
    assert rate.date == date(2023, 1, 1)
    assert rate.value == Decimal("1.1")
    
    # Test 2: Test with strict=False and non-existent rate
    rate = service.query(Currencies["USD"], Currencies["EUR"], date(2023, 1, 1))
    assert rate is None
    
    # Test 3: Test with strict=True and non-existent rate
    class StrictMockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None
        
        def queries(self, queries, strict=False):
            if strict:
                raise FXRateLookupError(
                    queries[0][0], queries[0][1], queries[0][2]
                )
            return [None] * len(list(queries))
    
    strict_service = StrictMockFXRateService()
    
    # Should return None when strict=False
    rate = strict_service.query(Currencies["EUR"], Currencies["USD"], date(2023, 1, 1))
    assert rate is None
    
    # Should raise FXRateLookupError when strict=True
    try:
        strict_service.query(Currencies["EUR"], Currencies["USD"], date(2023, 1, 1), strict=True)
        assert False, "Should have raised FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == Currencies["EUR"]
        assert e.ccy2 == Currencies["USD"]
        assert e.asof == date(2023, 1, 1)
    
    # Test 4: Test with same currency (should return 1:1 rate if implemented)
    class SameCurrencyService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            from decimal import Decimal
            if ccy1 == ccy2:
                return FXRate(ccy1, ccy2, asof, Decimal("1"))
            return None
        
        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) 
                    for ccy1, ccy2, asof in queries]
    
    same_service = SameCurrencyService()
    rate = same_service.query(Currencies["EUR"], Currencies["EUR"], date(2023, 1, 1))
    assert rate is not None
    assert rate.ccy1 == Currencies["EUR"]
    assert rate.ccy2 == Currencies["EUR"]
    assert rate.value == Decimal("1")


# LLM-generated content at query #14
#--------------------------

```python
def test_FXRateService_query():
    from decimal import Decimal
    from datetime import date
    from unittest.mock import Mock, patch
    import pytest
    
    # Mock dependencies
    mock_ccy1 = Mock()
    mock_ccy2 = Mock()
    mock_asof = date(2023, 1, 1)
    
    # Test 1: Query returns valid FXRate
    mock_service = Mock(spec=FXRateService)
    expected_rate = FXRate(mock_ccy1, mock_ccy2, mock_asof, Decimal("1.5"))
    mock_service.query.return_value = expected_rate
    
    result = mock_service.query(mock_ccy1, mock_ccy2, mock_asof)
    assert result == expected_rate
    mock_service.query.assert_called_once_with(mock_ccy1, mock_ccy2, mock_asof, strict=False)
    
    # Test 2: Query returns None when rate not found (non-strict mode)
    mock_service2 = Mock(spec=FXRateService)
    mock_service2.query.return_value = None
    
    result = mock_service2.query(mock_ccy1, mock_ccy2, mock_asof)
    assert result is None
    
    # Test 3: Query raises FXRateLookupError in strict mode when rate not found
    mock_service3 = Mock(spec=FXRateService)
    mock_service3.query.side_effect = FXRateLookupError(mock_ccy1, mock_ccy2, mock_asof)
    
    with pytest.raises(FXRateLookupError) as exc_info:
        mock_service3.query(mock_ccy1, mock_ccy2, mock_asof, strict=True)
    
    assert exc_info.value.ccy1 == mock_ccy1
    assert exc_info.value.ccy2 == mock_ccy2
    assert exc_info.value.asof == mock_asof
    
    # Test 4: Query with same currency returns rate with value 1
    mock_service4 = Mock(spec=FXRateService)
    same_currency_rate = FXRate(mock_ccy1, mock_ccy1, mock_asof, Decimal("1"))
    mock_service4.query.return_value = same_currency_rate
    
    result = mock_service4.query(mock_ccy1, mock_ccy1, mock_asof)
    assert result.value == Decimal("1")
    
    # Test 5: Query with inverted currency pair
    mock_service5 = Mock(spec=FXRateService)
    original_rate = FXRate(mock_ccy1, mock_ccy2, mock_asof, Decimal("2.0"))
    inverted_rate = ~original_rate
    mock_service5.query.return_value = inverted_rate
    
    result = mock_service5.query(mock_ccy2, mock_ccy1, mock_asof)
    assert result == inverted_rate
    assert result.value == Decimal("0.5")


# LLM-generated content at query #15
#--------------------------

```python
def test_FXRateService_queries():
    from datetime import date
    from decimal import Decimal
    from typing import List, Optional
    from unittest.mock import Mock, create_autospec

    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            pass

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            results = []
            for ccy1, ccy2, asof in queries:
                try:
                    rate = self.query(ccy1, ccy2, asof, strict)
                    results.append(rate)
                except FXRateLookupError:
                    if strict:
                        raise
                    results.append(None)
            return results

    service = ConcreteFXRateService()
    mock_query = Mock(side_effect=[
        FXRate(Currency("USD"), Currency("EUR"), date(2023, 1, 1), Decimal("0.85")),
        FXRate(Currency("EUR"), Currency("GBP"), date(2023, 1, 1), Decimal("0.88")),
        None,
        FXRate(Currency("GBP"), Currency("JPY"), date(2023, 1, 2), Decimal("150.0"))
    ])
    service.query = mock_query

    queries = [
        (Currency("USD"), Currency("EUR"), date(2023, 1, 1)),
        (Currency("EUR"), Currency("GBP"), date(2023, 1, 1)),
        (Currency("JPY"), Currency("USD"), date(2023, 1, 1)),
        (Currency("GBP"), Currency("JPY"), date(2023, 1, 2))
    ]

    results = list(service.queries(queries, strict=False))
    assert len(results) == 4
    assert results[0] == FXRate(Currency("USD"), Currency("EUR"), date(2023, 1, 1), Decimal("0.85"))
    assert results[1] == FXRate(Currency("EUR"), Currency("GBP"), date(2023, 1, 1), Decimal("0.88"))
    assert results[2] is None
    assert results[3] == FXRate(Currency("GBP"), Currency("JPY"), date(2023, 1, 2), Decimal("150.0"))

    mock_query.reset_mock(side_effect=True)
    mock_query.side_effect = [
        FXRate(Currency("USD"), Currency("EUR"), date(2023, 1, 1), Decimal("0.85")),
        FXRateLookupError(Currency("EUR"), Currency("GBP"), date(2023, 1, 1))
    ]

    queries_strict = [
        (Currency("USD"), Currency("EUR"), date(2023, 1, 1)),
        (Currency("EUR"), Currency("GBP"), date(2023, 1, 1))
    ]

    try:
        list(service.queries(queries_strict, strict=True))
        assert False, "Should have raised FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == Currency("EUR")
        assert e.ccy2 == Currency("GBP")
        assert e.asof == date(2023, 1, 1)

    empty_results = list(service.queries([], strict=False))
    assert empty_results == []


# LLM-generated content at query #16
#--------------------------

```python
def test_FXRateService_query():
    from unittest.mock import Mock, patch
    from decimal import Decimal
    import datetime
    from pypara.currencies import Currencies

    # Create a concrete implementation of FXRateService for testing
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            # Mock implementation that returns a fixed rate for EUR/USD
            if ccy1 == Currencies["EUR"] and ccy2 == Currencies["USD"] and asof == datetime.date(2023, 1, 1):
                return FXRate(ccy1, ccy2, asof, Decimal("1.1"))
            elif strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None

        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = TestFXRateService()

    # Test successful query
    rate = service.query(Currencies["EUR"], Currencies["USD"], datetime.date(2023, 1, 1))
    assert rate is not None
    assert rate.ccy1 == Currencies["EUR"]
    assert rate.ccy2 == Currencies["USD"]
    assert rate.date == datetime.date(2023, 1, 1)
    assert rate.value == Decimal("1.1")

    # Test non-strict query for non-existent rate
    rate = service.query(Currencies["USD"], Currencies["EUR"], datetime.date(2023, 1, 1))
    assert rate is None

    # Test strict query for non-existent rate
    try:
        service.query(Currencies["USD"], Currencies["EUR"], datetime.date(2023, 1, 1), strict=True)
        assert False, "Should have raised FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == Currencies["USD"]
        assert e.ccy2 == Currencies["EUR"]
        assert e.asof == datetime.date(2023, 1, 1)

    # Test with same currency pair
    rate = service.query(Currencies["EUR"], Currencies["EUR"], datetime.date(2023, 1, 1))
    assert rate is None

    # Test with different date
    rate = service.query(Currencies["EUR"], Currencies["USD"], datetime.date(2023, 1, 2))
    assert rate is None


# LLM-generated content at query #17
#--------------------------

```python
def test_FXRateService_query():
    from unittest.mock import Mock, patch
    from decimal import Decimal
    import datetime
    from pypara.currencies import Currencies
    
    # Create a concrete implementation for testing
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            # Simple implementation that returns a fixed rate for EUR/USD
            if ccy1 == Currencies["EUR"] and ccy2 == Currencies["USD"] and asof == datetime.date(2023, 1, 1):
                return FXRate(ccy1, ccy2, asof, Decimal("1.1"))
            elif strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None
        
        def queries(self, queries, strict=False):
            return [self.query(q[0], q[1], q[2], strict) for q in queries]
    
    service = TestFXRateService()
    
    # Test successful query
    rate = service.query(Currencies["EUR"], Currencies["USD"], datetime.date(2023, 1, 1))
    assert rate is not None
    assert rate.ccy1 == Currencies["EUR"]
    assert rate.ccy2 == Currencies["USD"]
    assert rate.date == datetime.date(2023, 1, 1)
    assert rate.value == Decimal("1.1")
    
    # Test non-strict query for non-existent rate
    rate = service.query(Currencies["USD"], Currencies["EUR"], datetime.date(2023, 1, 1))
    assert rate is None
    
    # Test strict query for non-existent rate
    try:
        service.query(Currencies["USD"], Currencies["EUR"], datetime.date(2023, 1, 1), strict=True)
        assert False, "Should have raised FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == Currencies["USD"]
        assert e.ccy2 == Currencies["EUR"]
        assert e.asof == datetime.date(2023, 1, 1)
    
    # Test with same currency (should return 1:1 rate if implemented)
    # Note: This depends on the actual implementation
    
    # Test with invalid date
    rate = service.query(Currencies["EUR"], Currencies["USD"], datetime.date(2023, 1, 2))
    assert rate is None


# LLM-generated content at query #18
#--------------------------

```python
def test_FXRateService_queries():
    from datetime import date
    from decimal import Decimal
    from typing import Optional, Iterable
    from unittest.mock import Mock, call

    class TestFXRateService(FXRateService):
        def __init__(self, rates_map):
            self.rates_map = rates_map
            self.query_calls = []

        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            self.query_calls.append((ccy1, ccy2, asof, strict))
            return self.rates_map.get((ccy1, ccy2, asof))

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    eur = Mock(spec=Currency, code="EUR")
    usd = Mock(spec=Currency, code="USD")
    gbp = Mock(spec=Currency, code="GBP")
    
    date1 = date(2023, 1, 1)
    date2 = date(2023, 1, 2)
    
    rate1 = FXRate(eur, usd, date1, Decimal("1.1"))
    rate2 = FXRate(eur, gbp, date2, Decimal("0.9"))
    rate3 = FXRate(usd, gbp, date1, Decimal("0.8"))
    
    service = TestFXRateService({
        (eur, usd, date1): rate1,
        (eur, gbp, date2): rate2,
        (usd, gbp, date1): rate3,
    })
    
    queries = [
        (eur, usd, date1),
        (eur, gbp, date2),
        (usd, gbp, date1),
        (gbp, eur, date1),
    ]
    
    results = list(service.queries(queries, strict=False))
    
    assert results == [rate1, rate2, rate3, None]
    assert service.query_calls == [
        (eur, usd, date1, False),
        (eur, gbp, date2, False),
        (usd, gbp, date1, False),
        (gbp, eur, date1, False),
    ]
    
    service.query_calls.clear()
    
    service_with_strict = TestFXRateService({
        (eur, usd, date1): rate1,
        (eur, gbp, date2): rate2,
    })
    
    try:
        list(service_with_strict.queries([(usd, gbp, date1)], strict=True))
        assert False, "Should have raised FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == usd
        assert e.ccy2 == gbp
        assert e.asof == date1
    
    empty_service = TestFXRateService({})
    empty_results = list(empty_service.queries([], strict=False))
    assert empty_results == []
    assert empty_service.query_calls == []


# LLM-generated content at query #19
#--------------------------

```python
def test_FXRateService_queries():
    from datetime import date
    from decimal import Decimal
    from typing import Optional, Iterable
    from unittest.mock import Mock, call

    class MockFXRateService(FXRateService):
        def __init__(self, rates: dict):
            self.rates = rates
            self.query_calls = []

        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            self.query_calls.append((ccy1, ccy2, asof, strict))
            return self.rates.get((ccy1, ccy2, asof))

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    class MockCurrency:
        def __init__(self, code):
            self.code = code

        def __eq__(self, other):
            return isinstance(other, MockCurrency) and self.code == other.code

        def __hash__(self):
            return hash(self.code)

        def __repr__(self):
            return f"Currency({self.code})"

    EUR = MockCurrency("EUR")
    USD = MockCurrency("USD")
    GBP = MockCurrency("GBP")
    JPY = MockCurrency("JPY")

    date1 = date(2023, 1, 1)
    date2 = date(2023, 1, 2)

    rate1 = FXRate(EUR, USD, date1, Decimal("1.1"))
    rate2 = FXRate(GBP, JPY, date2, Decimal("150.5"))
    rate3 = FXRate(USD, EUR, date1, Decimal("0.9091"))

    service = MockFXRateService({
        (EUR, USD, date1): rate1,
        (GBP, JPY, date2): rate2,
        (USD, EUR, date1): rate3,
    })

    queries = [
        (EUR, USD, date1),
        (GBP, JPY, date2),
        (USD, EUR, date1),
        (JPY, GBP, date2),
    ]

    results = list(service.queries(queries, strict=False))

    assert results == [rate1, rate2, rate3, None]
    assert service.query_calls == [
        (EUR, USD, date1, False),
        (GBP, JPY, date2, False),
        (USD, EUR, date1, False),
        (JPY, GBP, date2, False),
    ]

    service.query_calls.clear()
    service_without_rate = MockFXRateService({})
    empty_queries = [(EUR, USD, date1)]
    empty_results = list(service_without_rate.queries(empty_queries, strict=False))
    assert empty_results == [None]
    assert service_without_rate.query_calls == [(EUR, USD, date1, False)]

    class StrictFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            if strict:
                raise FXRateLookupError(EUR, USD, date1)
            return []

    strict_service = StrictFXRateService()
    try:
        list(strict_service.queries(queries, strict=True))
        assert False, "Should have raised FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == EUR
        assert e.ccy2 == USD
        assert e.asof == date1

    empty_query_list = []
    results_empty = list(service.queries(empty_query_list, strict=False))
    assert results_empty == []
    assert service.query_calls == []


# LLM-generated content at query #20
#--------------------------

```python
def test_FXRateService_queries():
    class TestFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            if ccy1.code == "EUR" and ccy2.code == "USD" and asof == Date(2023, 1, 1):
                return FXRate(ccy1, ccy2, asof, Decimal("1.1"))
            if ccy1.code == "USD" and ccy2.code == "EUR" and asof == Date(2023, 1, 1):
                return FXRate(ccy1, ccy2, asof, Decimal("0.9091"))
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    eur = Currency("EUR", 2)
    usd = Currency("USD", 2)
    date1 = Date(2023, 1, 1)
    date2 = Date(2023, 1, 2)

    service = TestFXRateService()
    queries_list = [(eur, usd, date1), (usd, eur, date1), (eur, usd, date2)]
    results = list(service.queries(queries_list, strict=False))

    assert len(results) == 3
    assert results[0] == FXRate(eur, usd, date1, Decimal("1.1"))
    assert results[1] == FXRate(usd, eur, date1, Decimal("0.9091"))
    assert results[2] is None

    with pytest.raises(FXRateLookupError):
        list(service.queries([(eur, usd, date2)], strict=True))


# LLM-generated content at query #21
#--------------------------

```python
def test_FXRateService_query():
    from decimal import Decimal
    from datetime import date
    from unittest.mock import Mock, patch
    from pypara.currencies import Currencies

    # Create a concrete implementation for testing
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            # Simple implementation that returns a fixed rate for EUR/USD
            if ccy1 == Currencies["EUR"] and ccy2 == Currencies["USD"] and asof == date(2023, 1, 1):
                return FXRate(ccy1, ccy2, asof, Decimal("1.1"))
            elif strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None

        def queries(self, queries, strict=False):
            return [self.query(q[0], q[1], q[2], strict) for q in queries]

    service = TestFXRateService()
    
    # Test successful query
    rate = service.query(Currencies["EUR"], Currencies["USD"], date(2023, 1, 1))
    assert rate is not None
    assert rate.ccy1 == Currencies["EUR"]
    assert rate.ccy2 == Currencies["USD"]
    assert rate.date == date(2023, 1, 1)
    assert rate.value == Decimal("1.1")
    
    # Test non-strict query for non-existent rate
    rate = service.query(Currencies["USD"], Currencies["EUR"], date(2023, 1, 1))
    assert rate is None
    
    # Test strict query for non-existent rate (should raise error)
    try:
        service.query(Currencies["USD"], Currencies["EUR"], date(2023, 1, 1), strict=True)
        assert False, "Should have raised FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == Currencies["USD"]
        assert e.ccy2 == Currencies["EUR"]
        assert e.asof == date(2023, 1, 1)
    
    # Test with same currency (should return None or raise depending on implementation)
    # Our test implementation returns None for same currency
    rate = service.query(Currencies["EUR"], Currencies["EUR"], date(2023, 1, 1))
    assert rate is None
    
    # Test with different date
    rate = service.query(Currencies["EUR"], Currencies["USD"], date(2023, 1, 2))
    assert rate is None


# LLM-generated content at query #22
#--------------------------

```python
def test_FXRateService_queries():
    from datetime import date
    from decimal import Decimal
    from typing import Optional, Iterable, Tuple
    from pypara.currencies import Currency
    from pypara.currencies import Currencies
    
    class MockFXRateService(FXRateService):
        def __init__(self, rates: dict):
            self.rates = rates
            
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            key = (ccy1, ccy2, asof)
            if key in self.rates:
                return self.rates[key]
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None
            
        def queries(self, queries: Iterable[Tuple[Currency, Currency, Date]], strict: bool = False) -> Iterable[Optional[FXRate]]:
            results = []
            for ccy1, ccy2, asof in queries:
                try:
                    results.append(self.query(ccy1, ccy2, asof, strict))
                except FXRateLookupError:
                    if strict:
                        raise
                    results.append(None)
            return results
    
    EUR = Currencies["EUR"]
    USD = Currencies["USD"]
    GBP = Currencies["GBP"]
    date1 = date(2023, 1, 1)
    date2 = date(2023, 1, 2)
    
    rate1 = FXRate(EUR, USD, date1, Decimal("1.1"))
    rate2 = FXRate(EUR, GBP, date1, Decimal("0.9"))
    rate3 = FXRate(USD, EUR, date2, Decimal("0.9"))
    
    service = MockFXRateService({
        (EUR, USD, date1): rate1,
        (EUR, GBP, date1): rate2,
        (USD, EUR, date2): rate3,
    })
    
    queries = [
        (EUR, USD, date1),
        (EUR, GBP, date1),
        (USD, EUR, date2),
        (USD, GBP, date1),
    ]
    
    results = list(service.queries(queries, strict=False))
    assert results[0] == rate1
    assert results[1] == rate2
    assert results[2] == rate3
    assert results[3] is None
    
    results_strict = list(service.queries(queries[:3], strict=False))
    assert results_strict == [rate1, rate2, rate3]
    
    try:
        list(service.queries(queries, strict=True))
        assert False, "Should have raised FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == USD
        assert e.ccy2 == GBP
        assert e.asof == date1
    
    empty_results = list(service.queries([], strict=False))
    assert empty_results == []
    
    empty_results_strict = list(service.queries([], strict=True))
    assert empty_results_strict == []


# LLM-generated content at query #23
#--------------------------

```python
def test_FXRateService_queries():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if ccy1.code == "EUR" and ccy2.code == "USD" and asof.year == 2023:
                return FXRate(ccy1, ccy2, asof, Decimal("1.1"))
            if ccy1.code == "USD" and ccy2.code == "EUR" and asof.year == 2023:
                return FXRate(ccy1, ccy2, asof, Decimal("0.9"))
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None

        def queries(self, queries, strict=False):
            results = []
            for ccy1, ccy2, asof in queries:
                try:
                    rate = self.query(ccy1, ccy2, asof, strict)
                    results.append(rate)
                except FXRateLookupError:
                    results.append(None)
            return results

    service = MockFXRateService()
    eur = Currency("EUR", 2)
    usd = Currency("USD", 2)
    date1 = Date(2023, 1, 1)
    date2 = Date(2023, 2, 1)
    date3 = Date(2022, 1, 1)

    queries = [
        (eur, usd, date1),
        (usd, eur, date2),
        (eur, usd, date3),
    ]

    results = list(service.queries(queries, strict=False))
    assert len(results) == 3
    assert results[0] == FXRate(eur, usd, date1, Decimal("1.1"))
    assert results[1] == FXRate(usd, eur, date2, Decimal("0.9"))
    assert results[2] is None

    results_strict = list(service.queries(queries[:2], strict=True))
    assert len(results_strict) == 2
    assert results_strict[0] == FXRate(eur, usd, date1, Decimal("1.1"))
    assert results_strict[1] == FXRate(usd, eur, date2, Decimal("0.9"))

    empty_results = list(service.queries([], strict=False))
    assert empty_results == []


# LLM-generated content at query #24
#--------------------------

```python
def test_FXRateService_queries():
    from decimal import Decimal
    from datetime import date
    from unittest.mock import Mock, call
    
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        
        def queries(self, queries, strict=False):
            results = []
            for ccy1, ccy2, asof in queries:
                if ccy1.code == "EUR" and ccy2.code == "USD" and asof == date(2023, 1, 1):
                    results.append(FXRate(ccy1, ccy2, asof, Decimal("1.1")))
                elif ccy1.code == "USD" and ccy2.code == "EUR" and asof == date(2023, 1, 2):
                    results.append(FXRate(ccy1, ccy2, asof, Decimal("0.9")))
                elif strict:
                    raise FXRateLookupError(ccy1, ccy2, asof)
                else:
                    results.append(None)
            return results
    
    service = TestFXRateService()
    
    eur = Mock(code="EUR")
    usd = Mock(code="USD")
    
    queries = [
        (eur, usd, date(2023, 1, 1)),
        (usd, eur, date(2023, 1, 2)),
        (eur, usd, date(2023, 1, 3)),
    ]
    
    results = list(service.queries(queries, strict=False))
    
    assert len(results) == 3
    assert results[0] is not None
    assert results[0].ccy1 == eur
    assert results[0].ccy2 == usd
    assert results[0].date == date(2023, 1, 1)
    assert results[0].value == Decimal("1.1")
    
    assert results[1] is not None
    assert results[1].ccy1 == usd
    assert results[1].ccy2 == eur
    assert results[1].date == date(2023, 1, 2)
    assert results[1].value == Decimal("0.9")
    
    assert results[2] is None
    
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))


# LLM-generated content at query #25
#--------------------------

```python
def test_FXRateService_queries():
    from datetime import date
    from decimal import Decimal
    from typing import List, Optional
    from unittest.mock import Mock, call

    class MockFXRateService(FXRateService):
        def __init__(self, rates: List[Optional[FXRate]]):
            self.rates = rates
            self.query_calls = []

        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            self.query_calls.append((ccy1, ccy2, asof, strict))
            if self.query_calls:
                return self.rates.pop(0) if self.rates else None
            return None

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    eur = Mock(spec=Currency)
    usd = Mock(spec=Currency)
    gbp = Mock(spec=Currency)
    date1 = date(2023, 1, 1)
    date2 = date(2023, 1, 2)

    rate1 = FXRate(eur, usd, date1, Decimal("1.1"))
    rate2 = FXRate(usd, gbp, date2, Decimal("0.8"))
    rate3 = None

    service = MockFXRateService([rate1, rate2, rate3])
    queries = [(eur, usd, date1), (usd, gbp, date2), (gbp, eur, date1)]

    results = list(service.queries(queries, strict=False))

    assert results == [rate1, rate2, rate3]
    assert service.query_calls == [
        (eur, usd, date1, False),
        (usd, gbp, date2, False),
        (gbp, eur, date1, False),
    ]

    service2 = MockFXRateService([rate1, rate2])
    queries2 = [(eur, usd, date1), (usd, gbp, date2)]

    results2 = list(service2.queries(queries2, strict=True))

    assert results2 == [rate1, rate2]
    assert service2.query_calls == [
        (eur, usd, date1, True),
        (usd, gbp, date2, True),
    ]

    service3 = MockFXRateService([])
    queries3 = []

    results3 = list(service3.queries(queries3))

    assert results3 == []
    assert service3.query_calls == []


# LLM-generated content at query #26
#--------------------------

```python
def test_FXRateService_query():
    from unittest.mock import Mock, patch
    from decimal import Decimal
    import datetime
    from pypara.currencies import Currencies
    
    # Create a concrete implementation for testing
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            # Simple implementation that returns a fixed rate for EUR/USD
            if ccy1 == Currencies["EUR"] and ccy2 == Currencies["USD"] and asof == datetime.date(2023, 1, 1):
                return FXRate(ccy1, ccy2, asof, Decimal("1.1"))
            elif strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None
        
        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    service = TestFXRateService()
    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    gbp = Currencies["GBP"]
    test_date = datetime.date(2023, 1, 1)
    
    # Test successful query
    rate = service.query(eur, usd, test_date)
    assert rate is not None
    assert rate.ccy1 == eur
    assert rate.ccy2 == usd
    assert rate.date == test_date
    assert rate.value == Decimal("1.1")
    
    # Test non-strict query for non-existent rate
    rate = service.query(eur, gbp, test_date)
    assert rate is None
    
    # Test strict query for non-existent rate
    try:
        service.query(eur, gbp, test_date, strict=True)
        assert False, "Should have raised FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == eur
        assert e.ccy2 == gbp
        assert e.asof == test_date
    
    # Test inverted rate using __invert__ method
    rate = service.query(eur, usd, test_date)
    inverted = ~rate
    assert inverted.ccy1 == usd
    assert inverted.ccy2 == eur
    assert inverted.date == test_date
    assert inverted.value == Decimal("1") / Decimal("1.1")
    
    # Test FXRate.of validation
    valid_rate = FXRate.of(eur, usd, test_date, Decimal("1.1"))
    assert valid_rate.ccy1 == eur
    assert valid_rate.ccy2 == usd
    
    # Test FXRate.of with invalid value
    try:
        FXRate.of(eur, usd, test_date, Decimal("0"))
        assert False, "Should have raised ValueError for zero value"
    except ValueError:
        pass
    
    # Test FXRate.of with same currency but not one
    try:
        FXRate.of(eur, eur, test_date, Decimal("2"))
        assert False, "Should have raised ValueError for same currency with non-one value"
    except ValueError:
        pass
    
    # Test FXRate.of with same currency and one (should work)
    same_currency_rate = FXRate.of(eur, eur, test_date, Decimal("1"))
    assert same_currency_rate.ccy1 == eur
    assert same_currency_rate.ccy2 == eur
    assert same_currency_rate.value == Decimal("1")


