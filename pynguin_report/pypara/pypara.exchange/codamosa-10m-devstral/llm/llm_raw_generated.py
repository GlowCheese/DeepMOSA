####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_FXRateService_queries():
    # Create a concrete implementation of FXRateService for testing
    class TestFXRateService(FXRateService):
        def __init__(self, rates):
            self.rates = rates

        def query(self, ccy1, ccy2, asof, strict=False):
            for rate in self.rates:
                if rate.ccy1 == ccy1 and rate.ccy2 == ccy2 and rate.date == asof:
                    return rate
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None

        def queries(self, queries, strict=False):
            results = []
            for ccy1, ccy2, asof in queries:
                rate = self.query(ccy1, ccy2, asof, strict)
                results.append(rate)
            return results

    # Create test data
    eur = Currency("EUR")
    usd = Currency("USD")
    today = Date.today()
    yesterday = today - datetime.timedelta(days=1)

    rate1 = FXRate(eur, usd, today, Decimal("1.2"))
    rate2 = FXRate(usd, eur, today, Decimal("0.8333"))
    rate3 = FXRate(eur, usd, yesterday, Decimal("1.1"))

    test_rates = [rate1, rate2, rate3]
    service = TestFXRateService(test_rates)

    # Test case 1: Normal queries
    queries = [
        (eur, usd, today),
        (usd, eur, today),
        (eur, usd, yesterday)
    ]
    results = list(service.queries(queries))

    assert len(results) == 3
    assert results[0] == rate1
    assert results[1] == rate2
    assert results[2] == rate3

    # Test case 2: Query with missing rate (non-strict)
    queries = [
        (eur, usd, today),
        (eur, usd, Date(2020, 1, 1))  # This rate doesn't exist
    ]
    results = list(service.queries(queries))

    assert len(results) == 2
    assert results[0] == rate1
    assert results[1] is None

    # Test case 3: Query with missing rate (strict)
    queries = [
        (eur, usd, today),
        (eur, usd, Date(2020, 1, 1))  # This rate doesn't exist
    ]

    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))

    # Test case 4: Empty queries
    assert list(service.queries([])) == []


# LLM-generated content at query #2
#--------------------------

```python
def test_FXRate___invert__():
    # Test basic inversion
    rate = FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("2"))
    inverted = ~rate
    assert inverted.ccy1 == Currency("USD")
    assert inverted.ccy2 == Currency("EUR")
    assert inverted.date == Date(2023, 1, 1)
    assert inverted.value == Decimal("0.5")

    # Test inversion of already inverted rate
    double_inverted = ~~rate
    assert double_inverted == rate

    # Test inversion with different value
    rate2 = FXRate(Currency("GBP"), Currency("JPY"), Date(2023, 1, 1), Decimal("150.25"))
    inverted2 = ~rate2
    assert inverted2.value == Decimal("0.006655")  # 1/150.25 ≈ 0.006655

    # Test inversion with value of 1 (same currency)
    rate3 = FXRate(Currency("USD"), Currency("USD"), Date(2023, 1, 1), Decimal("1"))
    inverted3 = ~rate3
    assert inverted3.value == Decimal("1")


# LLM-generated content at query #3
#--------------------------

```python
def test_FXRateService_query():
    # Create a mock FXRateService implementation for testing
    class MockFXRateService(FXRateService):
        def __init__(self):
            self.rates = {
                (Currency("EUR"), Currency("USD"), Date(2023, 1, 1)): FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1")),
                (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)): FXRate(Currency("USD"), Currency("EUR"), Date(2023, 1, 1), Decimal("0.909")),
            }

        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            key = (ccy1, ccy2, asof)
            if key in self.rates:
                return self.rates[key]
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None

        def queries(self, queries: Iterable[Tuple[Currency, Currency, Date]], strict: bool = False) -> Iterable[Optional[FXRate]]:
            for ccy1, ccy2, asof in queries:
                yield self.query(ccy1, ccy2, asof, strict)

    # Initialize the mock service
    service = MockFXRateService()

    # Test successful query
    rate = service.query(Currency("EUR"), Currency("USD"), Date(2023, 1, 1))
    assert rate is not None
    assert rate.ccy1 == Currency("EUR")
    assert rate.ccy2 == Currency("USD")
    assert rate.date == Date(2023, 1, 1)
    assert rate.value == Decimal("1.1")

    # Test inverted rate
    rate = service.query(Currency("USD"), Currency("EUR"), Date(2023, 1, 1))
    assert rate is not None
    assert rate.ccy1 == Currency("USD")
    assert rate.ccy2 == Currency("EUR")
    assert rate.date == Date(2023, 1, 1)
    assert rate.value == Decimal("0.909")

    # Test non-existent rate without strict
    rate = service.query(Currency("GBP"), Currency("JPY"), Date(2023, 1, 1))
    assert rate is None

    # Test non-existent rate with strict
    try:
        service.query(Currency("GBP"), Currency("JPY"), Date(2023, 1, 1), strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == Currency("GBP")
        assert e.ccy2 == Currency("JPY")
        assert e.asof == Date(2023, 1, 1)


# LLM-generated content at query #4
#--------------------------

```python
def test_FXRateService_queries():
    # Create a mock FXRateService implementation for testing
    class MockFXRateService(FXRateService):
        def __init__(self, rates):
            self.rates = rates

        def query(self, ccy1, ccy2, asof, strict=False):
            key = (ccy1, ccy2, asof)
            if strict and key not in self.rates:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return self.rates.get(key)

        def queries(self, queries, strict=False):
            results = []
            for ccy1, ccy2, asof in queries:
                key = (ccy1, ccy2, asof)
                if strict and key not in self.rates:
                    raise FXRateLookupError(ccy1, ccy2, asof)
                results.append(self.rates.get(key))
            return results

    # Test data
    eur = Currency("EUR", "Euro", 2)
    usd = Currency("USD", "US Dollar", 2)
    date1 = Date(2023, 1, 1)
    date2 = Date(2023, 1, 2)

    rate1 = FXRate(eur, usd, date1, Decimal("1.1"))
    rate2 = FXRate(eur, usd, date2, Decimal("1.2"))
    rate3 = FXRate(usd, eur, date1, Decimal("0.9"))

    rates = {
        (eur, usd, date1): rate1,
        (eur, usd, date2): rate2,
        (usd, eur, date1): rate3,
    }

    service = MockFXRateService(rates)

    # Test 1: Normal query with existing rates
    queries = [(eur, usd, date1), (eur, usd, date2), (usd, eur, date1)]
    results = list(service.queries(queries))
    assert results == [rate1, rate2, rate3]

    # Test 2: Query with non-existing rate (non-strict mode)
    queries = [(eur, usd, date1), (eur, usd, Date(2023, 1, 3))]
    results = list(service.queries(queries))
    assert results == [rate1, None]

    # Test 3: Query with non-existing rate (strict mode)
    queries = [(eur, usd, date1), (eur, usd, Date(2023, 1, 3))]
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))

    # Test 4: Empty query
    queries = []
    results = list(service.queries(queries))
    assert results == []

    # Test 5: Query with inverted rates
    queries = [(usd, eur, date1)]
    results = list(service.queries(queries))
    assert results == [rate3]
    assert results[0] == ~rate1


# LLM-generated content at query #5
#--------------------------

```python
def test_FXRate___invert__():
    # Test basic inversion
    rate = FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("2"))
    inverted = ~rate
    assert inverted.ccy1 == Currency("USD")
    assert inverted.ccy2 == Currency("EUR")
    assert inverted.date == Date(2023, 1, 1)
    assert inverted.value == Decimal("0.5")

    # Test inversion of already inverted rate
    double_inverted = ~~rate
    assert double_inverted == rate

    # Test inversion with different currencies
    rate2 = FXRate(Currency("GBP"), Currency("JPY"), Date(2023, 1, 1), Decimal("150.5"))
    inverted2 = ~rate2
    assert inverted2.ccy1 == Currency("JPY")
    assert inverted2.ccy2 == Currency("GBP")
    assert inverted2.value == Decimal("1") / Decimal("150.5")

    # Test inversion with same currency (should be 1)
    rate3 = FXRate(Currency("EUR"), Currency("EUR"), Date(2023, 1, 1), Decimal("1"))
    inverted3 = ~rate3
    assert inverted3 == rate3


# LLM-generated content at query #6
#--------------------------

```python
def test_FXRateService_query():
    # Create a mock FXRateService implementation for testing
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
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    # Test data
    from datetime import date
    from pypara.currencies import Currencies
    from decimal import Decimal

    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    today = date.today()
    rate = FXRate(eur, usd, today, Decimal("1.2"))

    # Test case 1: Successful query
    service = MockFXRateService({(eur, usd, today): rate})
    result = service.query(eur, usd, today)
    assert result == rate

    # Test case 2: Query with non-existent rate, non-strict
    result = service.query(eur, usd, date(2020, 1, 1))
    assert result is None

    # Test case 3: Query with non-existent rate, strict
    try:
        service.query(eur, usd, date(2020, 1, 1), strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == eur
        assert e.ccy2 == usd
        assert e.asof == date(2020, 1, 1)

    # Test case 4: Query with inverted currencies
    inverted_rate = ~rate
    service = MockFXRateService({(usd, eur, today): inverted_rate})
    result = service.query(usd, eur, today)
    assert result == inverted_rate


# LLM-generated content at query #7
#--------------------------

```python
def test_FXRateService_queries():
    # Create a concrete implementation of FXRateService for testing
    class TestFXRateService(FXRateService):
        def __init__(self, rates):
            self.rates = rates

        def query(self, ccy1, ccy2, asof, strict=False):
            key = (ccy1, ccy2, asof)
            rate = self.rates.get(key)
            if rate is None and strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return rate

        def queries(self, queries, strict=False):
            results = []
            for ccy1, ccy2, asof in queries:
                rate = self.query(ccy1, ccy2, asof, strict)
                results.append(rate)
            return results

    # Setup test data
    from datetime import date
    from pypara.currencies import Currencies

    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    today = date.today()

    rates = {
        (eur, usd, today): FXRate(eur, usd, today, Decimal("1.2")),
        (usd, eur, today): FXRate(usd, eur, today, Decimal("0.8")),
    }

    service = TestFXRateService(rates)

    # Test basic query
    queries = [(eur, usd, today), (usd, eur, today)]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == FXRate(eur, usd, today, Decimal("1.2"))
    assert results[1] == FXRate(usd, eur, today, Decimal("0.8"))

    # Test with missing rate (non-strict)
    queries = [(eur, usd, today), (eur, usd, date(2020, 1, 1))]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == FXRate(eur, usd, today, Decimal("1.2"))
    assert results[1] is None

    # Test with missing rate (strict)
    queries = [(eur, usd, date(2020, 1, 1))]
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))

    # Test empty queries
    assert list(service.queries([])) == []


# LLM-generated content at query #8
#--------------------------

```python
def test_FXRate___invert__():
    # Test inversion of FXRate
    ccy1 = Currency("EUR")
    ccy2 = Currency("USD")
    date = Date(2023, 1, 1)
    value = Decimal("2")
    rate = FXRate(ccy1, ccy2, date, value)
    inverted_rate = ~rate

    assert inverted_rate.ccy1 == ccy2
    assert inverted_rate.ccy2 == ccy1
    assert inverted_rate.date == date
    assert inverted_rate.value == Decimal("0.5")

    # Test inversion of inverted FXRate
    double_inverted_rate = ~inverted_rate
    assert double_inverted_rate == rate


# LLM-generated content at query #9
#--------------------------

```python
def test_FXRateService_query():
    # Setup
    class MockFXRateService(FXRateService):
        def __init__(self, rates):
            self.rates = rates

        def query(self, ccy1, ccy2, asof, strict=False):
            key = (ccy1, ccy2, asof)
            rate = self.rates.get(key)
            if rate is None and strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return rate

        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    # Test data
    eur = Currency("EUR", "Euro", 2)
    usd = Currency("USD", "US Dollar", 2)
    today = Date.today()
    rate = FXRate(eur, usd, today, Decimal("1.20"))

    # Test successful query
    service = MockFXRateService({(eur, usd, today): rate})
    result = service.query(eur, usd, today)
    assert result == rate

    # Test query with no result and non-strict mode
    service = MockFXRateService({})
    result = service.query(eur, usd, today)
    assert result is None

    # Test query with no result and strict mode
    service = MockFXRateService({})
    with pytest.raises(FXRateLookupError):
        service.query(eur, usd, today, strict=True)


# LLM-generated content at query #10
#--------------------------

```python
def test_FXRateService_queries():
    # Create a mock FXRateService implementation for testing
    class MockFXRateService(FXRateService):
        def __init__(self):
            self.queries_called = False
            self.query_called = False

        def query(self, ccy1, ccy2, asof, strict=False):
            self.query_called = True
            if ccy1.code == "EUR" and ccy2.code == "USD":
                return FXRate(Currencies["EUR"], Currencies["USD"], asof, Decimal("1.2"))
            elif ccy1.code == "USD" and ccy2.code == "EUR":
                return FXRate(Currencies["USD"], Currencies["EUR"], asof, Decimal("0.8333"))
            elif strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None

        def queries(self, queries, strict=False):
            self.queries_called = True
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    # Create mock currencies for testing
    class MockCurrency:
        def __init__(self, code):
            self.code = code

    Currencies = {"EUR": MockCurrency("EUR"), "USD": MockCurrency("USD")}

    # Initialize the mock service
    service = MockFXRateService()

    # Test case 1: Normal queries with existing rates
    queries = [
        (Currencies["EUR"], Currencies["USD"], Date(2023, 1, 1)),
        (Currencies["USD"], Currencies["EUR"], Date(2023, 1, 1)),
    ]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == FXRate(Currencies["EUR"], Currencies["USD"], Date(2023, 1, 1), Decimal("1.2"))
    assert results[1] == FXRate(Currencies["USD"], Currencies["EUR"], Date(2023, 1, 1), Decimal("0.8333"))
    assert service.queries_called
    assert service.query_called

    # Reset flags
    service.queries_called = False
    service.query_called = False

    # Test case 2: Queries with non-existing rates (non-strict)
    queries = [
        (Currencies["EUR"], Currencies["EUR"], Date(2023, 1, 1)),
        (Currencies["USD"], Currencies["USD"], Date(2023, 1, 1)),
    ]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] is None
    assert results[1] is None
    assert service.queries_called
    assert service.query_called

    # Reset flags
    service.queries_called = False
    service.query_called = False

    # Test case 3: Queries with non-existing rates (strict)
    queries = [
        (Currencies["EUR"], Currencies["EUR"], Date(2023, 1, 1)),
    ]
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))
    assert service.queries_called
    assert service.query_called


# LLM-generated content at query #11
#--------------------------

```python
def test_FXRateService_queries():
    # Create a mock FXRateService implementation
    class MockFXRateService(FXRateService):
        def __init__(self):
            self.rates = {
                (Currency("EUR"), Currency("USD"), Date(2023, 1, 1)): FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1")),
                (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)): FXRate(Currency("USD"), Currency("EUR"), Date(2023, 1, 1), Decimal("0.9091")),
            }

        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            return self.rates.get((ccy1, ccy2, asof))

        def queries(self, queries: Iterable[Tuple[Currency, Currency, Date]], strict: bool = False) -> Iterable[Optional[FXRate]]:
            for ccy1, ccy2, asof in queries:
                rate = self.query(ccy1, ccy2, asof)
                if rate is None and strict:
                    raise FXRateLookupError(ccy1, ccy2, asof)
                yield rate

    # Test with valid queries
    service = MockFXRateService()
    queries = [
        (Currency("EUR"), Currency("USD"), Date(2023, 1, 1)),
        (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)),
    ]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1"))
    assert results[1] == FXRate(Currency("USD"), Currency("EUR"), Date(2023, 1, 1), Decimal("0.9091"))

    # Test with invalid query (non-strict)
    queries_with_invalid = [
        (Currency("EUR"), Currency("USD"), Date(2023, 1, 1)),
        (Currency("GBP"), Currency("JPY"), Date(2023, 1, 1)),
    ]
    results = list(service.queries(queries_with_invalid))
    assert len(results) == 2
    assert results[0] == FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1"))
    assert results[1] is None

    # Test with invalid query (strict)
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries_with_invalid, strict=True))


# LLM-generated content at query #12
#--------------------------

```python
def test_FXRateService_query():
    # Create a concrete implementation of FXRateService for testing
    class TestFXRateService(FXRateService):
        def __init__(self, rates):
            self.rates = rates

        def query(self, ccy1, ccy2, asof, strict=False):
            key = (ccy1, ccy2, asof)
            rate = self.rates.get(key)
            if rate is None and strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return rate

        def queries(self, queries, strict=False):
            for query in queries:
                yield self.query(*query, strict=strict)

    # Setup test data
    from datetime import date
    from pypara.currencies import Currencies
    from decimal import Decimal

    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    today = date.today()
    yesterday = today - timedelta(days=1)

    test_rates = {
        (eur, usd, today): FXRate(eur, usd, today, Decimal("1.2")),
        (usd, eur, today): FXRate(usd, eur, today, Decimal("0.8333")),
        (eur, usd, yesterday): FXRate(eur, usd, yesterday, Decimal("1.15")),
    }

    service = TestFXRateService(test_rates)

    # Test successful query
    result = service.query(eur, usd, today)
    assert result == FXRate(eur, usd, today, Decimal("1.2"))

    # Test inverted rate
    result = service.query(usd, eur, today)
    assert result == FXRate(usd, eur, today, Decimal("0.8333"))

    # Test different date
    result = service.query(eur, usd, yesterday)
    assert result == FXRate(eur, usd, yesterday, Decimal("1.15"))

    # Test non-existent rate without strict
    result = service.query(eur, usd, date(2000, 1, 1))
    assert result is None

    # Test non-existent rate with strict
    with pytest.raises(FXRateLookupError):
        service.query(eur, usd, date(2000, 1, 1), strict=True)

    # Test same currency
    result = service.query(eur, eur, today)
    assert result is None  # Assuming no rate is stored for same currency

    # Test with strict for same currency
    with pytest.raises(FXRateLookupError):
        service.query(eur, eur, today, strict=True)


# LLM-generated content at query #13
#--------------------------

```python
def test_FXRateService_query():
    # Create a mock FXRateService implementation for testing
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
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    # Test data
    test_date = Date(2023, 1, 1)
    eur = Currency("EUR")
    usd = Currency("USD")
    gbp = Currency("GBP")

    # Create test rates
    rates = {
        (eur, usd, test_date): FXRate(eur, usd, test_date, Decimal("1.10")),
        (usd, gbp, test_date): FXRate(usd, gbp, test_date, Decimal("0.80")),
    }

    # Initialize service
    service = MockFXRateService(rates)

    # Test successful query
    result = service.query(eur, usd, test_date)
    assert result == FXRate(eur, usd, test_date, Decimal("1.10"))

    # Test query with None result (non-strict)
    result = service.query(eur, gbp, test_date)
    assert result is None

    # Test query with exception (strict)
    with pytest.raises(FXRateLookupError):
        service.query(eur, gbp, test_date, strict=True)

    # Test inverted query
    result = service.query(usd, eur, test_date)
    assert result is None  # Should be None as we don't have this rate in our mock

    # Test query with different date
    different_date = Date(2023, 1, 2)
    result = service.query(eur, usd, different_date)
    assert result is None


# LLM-generated content at query #14
#--------------------------

```python
def test_FXRateService_queries():
    # Create a mock FXRateService implementation for testing
    class MockFXRateService(FXRateService):
        def __init__(self, rates):
            self.rates = rates

        def query(self, ccy1, ccy2, asof, strict=False):
            for rate in self.rates:
                if rate.ccy1 == ccy1 and rate.ccy2 == ccy2 and rate.date == asof:
                    return rate
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None

        def queries(self, queries, strict=False):
            results = []
            for ccy1, ccy2, asof in queries:
                rate = self.query(ccy1, ccy2, asof, strict=strict)
                results.append(rate)
            return results

    # Create test data
    eur = Currency("EUR", "Euro", 2)
    usd = Currency("USD", "US Dollar", 2)
    date1 = Date(2023, 1, 1)
    date2 = Date(2023, 1, 2)

    rate1 = FXRate(eur, usd, date1, Decimal("1.10"))
    rate2 = FXRate(eur, usd, date2, Decimal("1.15"))
    rate3 = FXRate(usd, eur, date1, Decimal("0.90"))

    rates = [rate1, rate2, rate3]
    service = MockFXRateService(rates)

    # Test case 1: Multiple queries with existing rates
    queries = [(eur, usd, date1), (eur, usd, date2), (usd, eur, date1)]
    results = list(service.queries(queries))
    assert len(results) == 3
    assert results[0] == rate1
    assert results[1] == rate2
    assert results[2] == rate3

    # Test case 2: Query with non-existing rate (non-strict)
    queries = [(eur, usd, Date(2023, 1, 3))]
    results = list(service.queries(queries))
    assert len(results) == 1
    assert results[0] is None

    # Test case 3: Query with non-existing rate (strict)
    queries = [(eur, usd, Date(2023, 1, 3))]
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))

    # Test case 4: Empty queries
    queries = []
    results = list(service.queries(queries))
    assert len(results) == 0


# LLM-generated content at query #15
#--------------------------

```python
def test_FXRateService_queries():
    # Mock implementation of FXRateService for testing
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
                key = (ccy1, ccy2, asof)
                if key in self.rates:
                    results.append(self.rates[key])
                else:
                    if strict:
                        raise FXRateLookupError(ccy1, ccy2, asof)
                    results.append(None)
            return results

    # Create test data
    from datetime import date
    from decimal import Decimal
    from pypara.currencies import Currencies

    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    gbp = Currencies["GBP"]
    today = date.today()

    rates = {
        (eur, usd, today): FXRate(eur, usd, today, Decimal("1.2")),
        (usd, gbp, today): FXRate(usd, gbp, today, Decimal("0.8")),
    }

    # Test case 1: All rates found
    service = MockFXRateService(rates)
    queries = [(eur, usd, today), (usd, gbp, today)]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == rates[(eur, usd, today)]
    assert results[1] == rates[(usd, gbp, today)]

    # Test case 2: Some rates not found, non-strict mode
    queries = [(eur, usd, today), (eur, gbp, today)]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == rates[(eur, usd, today)]
    assert results[1] is None

    # Test case 3: Some rates not found, strict mode
    queries = [(eur, usd, today), (eur, gbp, today)]
    try:
        list(service.queries(queries, strict=True))
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == eur
        assert e.ccy2 == gbp
        assert e.asof == today

    # Test case 4: Empty queries
    queries = []
    results = list(service.queries(queries))
    assert len(results) == 0


# LLM-generated content at query #16
#--------------------------

```python
def test_FXRateService_queries():
    # Create a mock FXRateService implementation for testing
    class MockFXRateService(FXRateService):
        def __init__(self, rates):
            self.rates = rates

        def query(self, ccy1, ccy2, asof, strict=False):
            key = (ccy1, ccy2, asof)
            if strict and key not in self.rates:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return self.rates.get(key)

        def queries(self, queries, strict=False):
            results = []
            for ccy1, ccy2, asof in queries:
                key = (ccy1, ccy2, asof)
                if strict and key not in self.rates:
                    raise FXRateLookupError(ccy1, ccy2, asof)
                results.append(self.rates.get(key))
            return results

    # Setup test data
    from datetime import date
    from pypara.currencies import Currencies
    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    gbp = Currencies["GBP"]
    today = date.today()

    rates = {
        (eur, usd, today): FXRate(eur, usd, today, Decimal("1.2")),
        (usd, gbp, today): FXRate(usd, gbp, today, Decimal("0.8")),
        (eur, gbp, today): FXRate(eur, gbp, today, Decimal("0.9")),
    }

    service = MockFXRateService(rates)

    # Test 1: Normal query with existing rates
    queries = [(eur, usd, today), (usd, gbp, today)]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == rates[(eur, usd, today)]
    assert results[1] == rates[(usd, gbp, today)]

    # Test 2: Query with non-existing rate (non-strict mode)
    queries = [(eur, usd, today), (gbp, usd, today)]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == rates[(eur, usd, today)]
    assert results[1] is None

    # Test 3: Query with non-existing rate (strict mode)
    queries = [(eur, usd, today), (gbp, usd, today)]
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))

    # Test 4: Empty queries
    assert list(service.queries([])) == []

    # Test 5: Query with inverted rates
    queries = [(usd, eur, today)]
    results = list(service.queries(queries))
    assert len(results) == 1
    assert results[0] is None  # Since we don't have USD/EUR in our rates

    # Test 6: Multiple queries with some missing rates
    queries = [(eur, usd, today), (usd, gbp, today), (gbp, eur, today)]
    results = list(service.queries(queries))
    assert len(results) == 3
    assert results[0] == rates[(eur, usd, today)]
    assert results[1] == rates[(usd, gbp, today)]
    assert results[2] is None


# LLM-generated content at query #17
#--------------------------

```python
def test_FXRateService_queries():
    # Create a concrete implementation of FXRateService for testing
    class TestFXRateService(FXRateService):
        def __init__(self, rates):
            self.rates = rates

        def query(self, ccy1, ccy2, asof, strict=False):
            key = (ccy1, ccy2, asof)
            rate = self.rates.get(key)
            if rate is None and strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return rate

        def queries(self, queries, strict=False):
            results = []
            for query in queries:
                ccy1, ccy2, asof = query
                rate = self.query(ccy1, ccy2, asof, strict)
                results.append(rate)
            return results

    # Setup test data
    from datetime import date
    from pypara.currencies import Currencies

    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    gbp = Currencies["GBP"]

    test_rates = {
        (eur, usd, date(2023, 1, 1)): FXRate(eur, usd, date(2023, 1, 1), Decimal("1.1")),
        (usd, gbp, date(2023, 1, 1)): FXRate(usd, gbp, date(2023, 1, 1), Decimal("0.8")),
        (eur, gbp, date(2023, 1, 2)): FXRate(eur, gbp, date(2023, 1, 2), Decimal("0.88")),
    }

    service = TestFXRateService(test_rates)

    # Test 1: Basic query with existing rates
    queries = [
        (eur, usd, date(2023, 1, 1)),
        (usd, gbp, date(2023, 1, 1)),
    ]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == test_rates[(eur, usd, date(2023, 1, 1))]
    assert results[1] == test_rates[(usd, gbp, date(2023, 1, 1))]

    # Test 2: Query with non-existing rate (non-strict mode)
    queries = [
        (eur, usd, date(2023, 1, 1)),
        (eur, gbp, date(2023, 1, 1)),  # This rate doesn't exist
    ]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == test_rates[(eur, usd, date(2023, 1, 1))]
    assert results[1] is None

    # Test 3: Query with non-existing rate (strict mode)
    queries = [
        (eur, gbp, date(2023, 1, 1)),  # This rate doesn't exist
    ]
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))

    # Test 4: Empty queries
    assert list(service.queries([])) == []

    # Test 5: Query with inverted rate
    queries = [
        (usd, eur, date(2023, 1, 1)),
    ]
    results = list(service.queries(queries))
    assert len(results) == 1
    assert results[0] is None  # Inverted rate not in test data

    # Test 6: Multiple queries with mixed results
    queries = [
        (eur, usd, date(2023, 1, 1)),
        (eur, gbp, date(2023, 1, 1)),  # Doesn't exist
        (eur, gbp, date(2023, 1, 2)),  # Exists
    ]
    results = list(service.queries(queries))
    assert len(results) == 3
    assert results[0] == test_rates[(eur, usd, date(2023, 1, 1))]
    assert results[1] is None
    assert results[2] == test_rates[(eur, gbp, date(2023, 1, 2))]


# LLM-generated content at query #18
#--------------------------

```python
def test_FXRateService_queries():
    # Create a concrete implementation of FXRateService for testing
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
            for ccy1, ccy2, asof in queries:
                yield self.query(ccy1, ccy2, asof, strict)

    # Setup test data
    from datetime import date
    from pypara.currencies import Currencies
    EUR = Currencies["EUR"]
    USD = Currencies["USD"]
    GBP = Currencies["GBP"]

    rates = {
        (EUR, USD, date(2023, 1, 1)): FXRate(EUR, USD, date(2023, 1, 1), Decimal("1.1")),
        (USD, GBP, date(2023, 1, 1)): FXRate(USD, GBP, date(2023, 1, 1), Decimal("0.8")),
        (EUR, GBP, date(2023, 1, 2)): FXRate(EUR, GBP, date(2023, 1, 2), Decimal("0.88")),
    }

    service = TestFXRateService(rates)

    # Test 1: Normal queries
    queries = [
        (EUR, USD, date(2023, 1, 1)),
        (USD, GBP, date(2023, 1, 1)),
        (EUR, GBP, date(2023, 1, 2)),
    ]
    results = list(service.queries(queries))
    assert len(results) == 3
    assert results[0] == FXRate(EUR, USD, date(2023, 1, 1), Decimal("1.1"))
    assert results[1] == FXRate(USD, GBP, date(2023, 1, 1), Decimal("0.8"))
    assert results[2] == FXRate(EUR, GBP, date(2023, 1, 2), Decimal("0.88"))

    # Test 2: Query with missing rate (non-strict)
    queries = [
        (EUR, USD, date(2023, 1, 1)),
        (EUR, GBP, date(2023, 1, 3)),  # Missing
    ]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == FXRate(EUR, USD, date(2023, 1, 1), Decimal("1.1"))
    assert results[1] is None

    # Test 3: Query with missing rate (strict)
    queries = [
        (EUR, GBP, date(2023, 1, 3)),  # Missing
    ]
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))

    # Test 4: Empty queries
    assert list(service.queries([])) == []


# LLM-generated content at query #19
#--------------------------

```python
def test_FXRateService_queries():
    # Create a mock FXRateService implementation
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            if ccy1.code == "EUR" and ccy2.code == "USD" and asof == Date(2023, 1, 1):
                return FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1"))
            elif ccy1.code == "USD" and ccy2.code == "JPY" and asof == Date(2023, 1, 1):
                return FXRate(Currency("USD"), Currency("JPY"), Date(2023, 1, 1), Decimal("130.5"))
            else:
                return None

        def queries(self, queries: Iterable[Tuple[Currency, Currency, Date]], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return (self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries)

    # Initialize the mock service
    service = MockFXRateService()

    # Test case 1: Normal queries
    queries = [
        (Currency("EUR"), Currency("USD"), Date(2023, 1, 1)),
        (Currency("USD"), Currency("JPY"), Date(2023, 1, 1)),
        (Currency("GBP"), Currency("EUR"), Date(2023, 1, 1))
    ]
    results = list(service.queries(queries))
    assert len(results) == 3
    assert results[0] == FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1"))
    assert results[1] == FXRate(Currency("USD"), Currency("JPY"), Date(2023, 1, 1), Decimal("130.5"))
    assert results[2] is None

    # Test case 2: Strict mode with missing rate
    with pytest.raises(FXRateLookupError):
        list(service.queries([(Currency("GBP"), Currency("EUR"), Date(2023, 1, 1))], strict=True))

    # Test case 3: Empty queries
    assert list(service.queries([])) == []

    # Test case 4: Inverted rates
    queries = [
        (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)),
    ]
    results = list(service.queries(queries))
    assert results[0] is None  # Since the mock service doesn't have this rate


# LLM-generated content at query #20
#--------------------------

```python
def test_FXRateService_query():
    # Create a mock FXRateService implementation for testing
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
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    # Test data
    from datetime import date
    from pypara.currencies import Currencies
    from decimal import Decimal

    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    today = date.today()
    rate = FXRate(eur, usd, today, Decimal("1.2"))

    # Test with existing rate
    service = MockFXRateService({(eur, usd, today): rate})
    result = service.query(eur, usd, today)
    assert result == rate

    # Test with non-existing rate, non-strict
    result = service.query(eur, usd, date(2020, 1, 1))
    assert result is None

    # Test with non-existing rate, strict
    try:
        service.query(eur, usd, date(2020, 1, 1), strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == eur
        assert e.ccy2 == usd
        assert e.asof == date(2020, 1, 1)


# LLM-generated content at query #21
#--------------------------

```python
def test_FXRateService_queries():
    # Create a mock FXRateService implementation for testing
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            # Mock implementation that returns a fixed rate for EUR/USD
            if ccy1.code == "EUR" and ccy2.code == "USD":
                return FXRate(ccy1, ccy2, asof, Decimal("1.2"))
            return None

        def queries(self, queries: Iterable[Tuple[Currency, Currency, Date]], strict: bool = False) -> Iterable[Optional[FXRate]]:
            # Mock implementation that processes each query
            results = []
            for ccy1, ccy2, asof in queries:
                result = self.query(ccy1, ccy2, asof, strict)
                if result is None and strict:
                    raise FXRateLookupError(ccy1, ccy2, asof)
                results.append(result)
            return results

    # Initialize the mock service
    service = MockFXRateService()

    # Test case 1: Normal queries with valid rates
    queries = [
        (Currency("EUR"), Currency("USD"), Date(2023, 1, 1)),
        (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)),
    ]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.2"))
    assert results[1] is None  # USD/EUR not in mock data

    # Test case 2: Strict mode with missing rate
    with pytest.raises(FXRateLookupError):
        list(service.queries([(Currency("USD"), Currency("EUR"), Date(2023, 1, 1))], strict=True))

    # Test case 3: Empty queries
    assert list(service.queries([])) == []

    # Test case 4: Multiple queries with some missing rates
    queries = [
        (Currency("EUR"), Currency("USD"), Date(2023, 1, 1)),
        (Currency("GBP"), Currency("JPY"), Date(2023, 1, 1)),
        (Currency("EUR"), Currency("USD"), Date(2023, 1, 2)),
    ]
    results = list(service.queries(queries))
    assert len(results) == 3
    assert results[0] == FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.2"))
    assert results[1] is None  # GBP/JPY not in mock data
    assert results[2] == FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 2), Decimal("1.2"))


# LLM-generated content at query #22
#--------------------------

```python
def test_FXRateService_query():
    # Create a concrete implementation of FXRateService for testing
    class TestFXRateService(FXRateService):
        def __init__(self, rates):
            self.rates = rates

        def query(self, ccy1, ccy2, asof, strict=False):
            key = (ccy1, ccy2, asof)
            rate = self.rates.get(key)
            if rate is None and strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return rate

        def queries(self, queries, strict=False):
            for query in queries:
                yield self.query(*query, strict=strict)

    # Setup test data
    from datetime import date
    from pypara.currencies import Currencies
    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    gbp = Currencies["GBP"]
    today = date.today()

    rates = {
        (eur, usd, today): FXRate(eur, usd, today, Decimal("1.2")),
        (usd, gbp, today): FXRate(usd, gbp, today, Decimal("0.8")),
    }

    service = TestFXRateService(rates)

    # Test successful query
    result = service.query(eur, usd, today)
    assert result == FXRate(eur, usd, today, Decimal("1.2"))

    # Test query with None result (non-strict)
    result = service.query(eur, gbp, today)
    assert result is None

    # Test query with None result (strict)
    with pytest.raises(FXRateLookupError):
        service.query(eur, gbp, today, strict=True)

    # Test inverted rate query
    result = service.query(usd, eur, today)
    assert result == FXRate(usd, eur, today, Decimal("0.833333333333333333333333333"))

    # Test same currency query
    result = service.query(eur, eur, today)
    assert result == FXRate(eur, eur, today, Decimal("1"))


# LLM-generated content at query #23
#--------------------------

```python
def test_FXRateService_queries():
    # Setup
    class TestFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            if ccy1.code == "EUR" and ccy2.code == "USD" and asof == Date(2023, 1, 1):
                return FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1"))
            elif ccy1.code == "USD" and ccy2.code == "EUR" and asof == Date(2023, 1, 1):
                return FXRate(Currency("USD"), Currency("EUR"), Date(2023, 1, 1), Decimal("0.9"))
            elif strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            else:
                return None

        def queries(self, queries: Iterable[Tuple[Currency, Currency, Date]], strict: bool = False) -> Iterable[Optional[FXRate]]:
            for ccy1, ccy2, asof in queries:
                yield self.query(ccy1, ccy2, asof, strict)

    service = TestFXRateService()

    # Test with valid queries
    queries = [
        (Currency("EUR"), Currency("USD"), Date(2023, 1, 1)),
        (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)),
        (Currency("GBP"), Currency("USD"), Date(2023, 1, 1))
    ]
    results = list(service.queries(queries))
    assert len(results) == 3
    assert results[0] == FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1"))
    assert results[1] == FXRate(Currency("USD"), Currency("EUR"), Date(2023, 1, 1), Decimal("0.9"))
    assert results[2] is None

    # Test with strict mode
    queries = [
        (Currency("EUR"), Currency("USD"), Date(2023, 1, 1)),
        (Currency("GBP"), Currency("USD"), Date(2023, 1, 1))
    ]
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))

    # Test with empty queries
    assert list(service.queries([])) == []


# LLM-generated content at query #24
#--------------------------

```python
def test_FXRateService_query():
    # Create a concrete implementation of FXRateService for testing
    class TestFXRateService(FXRateService):
        def __init__(self, rates):
            self.rates = rates

        def query(self, ccy1, ccy2, asof, strict=False):
            key = (ccy1, ccy2, asof)
            rate = self.rates.get(key)
            if rate is None and strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return rate

        def queries(self, queries, strict=False):
            for ccy1, ccy2, asof in queries:
                yield self.query(ccy1, ccy2, asof, strict)

    # Setup test data
    from datetime import date
    from pypara.currencies import Currencies
    test_rates = {
        (Currencies["EUR"], Currencies["USD"], date(2023, 1, 1)): FXRate(Currencies["EUR"], Currencies["USD"], date(2023, 1, 1), Decimal("1.1")),
        (Currencies["USD"], Currencies["EUR"], date(2023, 1, 1)): FXRate(Currencies["USD"], Currencies["EUR"], date(2023, 1, 1), Decimal("0.909")),
    }
    service = TestFXRateService(test_rates)

    # Test successful query
    result = service.query(Currencies["EUR"], Currencies["USD"], date(2023, 1, 1))
    assert result == FXRate(Currencies["EUR"], Currencies["USD"], date(2023, 1, 1), Decimal("1.1"))

    # Test query with non-existent rate (non-strict)
    result = service.query(Currencies["GBP"], Currencies["USD"], date(2023, 1, 1))
    assert result is None

    # Test query with non-existent rate (strict)
    try:
        service.query(Currencies["GBP"], Currencies["USD"], date(2023, 1, 1), strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == Currencies["GBP"]
        assert e.ccy2 == Currencies["USD"]
        assert e.asof == date(2023, 1, 1)


# LLM-generated content at query #25
#--------------------------

```python
def test_FXRateService_query():
    # Create a concrete implementation of FXRateService for testing
    class TestFXRateService(FXRateService):
        def __init__(self):
            self.rates = {
                (Currency("EUR"), Currency("USD"), Date(2023, 1, 1)): FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1")),
                (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)): FXRate(Currency("USD"), Currency("EUR"), Date(2023, 1, 1), Decimal("0.9091")),
            }

        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            key = (ccy1, ccy2, asof)
            if key in self.rates:
                return self.rates[key]
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None

        def queries(self, queries: Iterable[Tuple[Currency, Currency, Date]], strict: bool = False) -> Iterable[Optional[FXRate]]:
            for query in queries:
                yield self.query(*query, strict=strict)

    # Initialize the test service
    service = TestFXRateService()

    # Test successful query
    result = service.query(Currency("EUR"), Currency("USD"), Date(2023, 1, 1))
    assert result == FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1"))

    # Test inverted rate query
    result = service.query(Currency("USD"), Currency("EUR"), Date(2023, 1, 1))
    assert result == FXRate(Currency("USD"), Currency("EUR"), Date(2023, 1, 1), Decimal("0.9091"))

    # Test non-existent rate query without strict
    result = service.query(Currency("GBP"), Currency("USD"), Date(2023, 1, 1))
    assert result is None

    # Test non-existent rate query with strict
    with pytest.raises(FXRateLookupError):
        service.query(Currency("GBP"), Currency("USD"), Date(2023, 1, 1), strict=True)


# LLM-generated content at query #26
#--------------------------

```python
def test_FXRateService_queries():
    # Create a mock FXRateService implementation for testing
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
                rate = self.query(ccy1, ccy2, asof, strict)
                results.append(rate)
            return results

    # Test data
    from datetime import date
    from pypara.currencies import Currencies

    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    today = date.today()

    rates = {
        (eur, usd, today): FXRate(eur, usd, today, Decimal("1.2")),
        (usd, eur, today): FXRate(usd, eur, today, Decimal("0.8333")),
    }

    service = MockFXRateService(rates)

    # Test successful queries
    queries = [(eur, usd, today), (usd, eur, today)]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == rates[(eur, usd, today)]
    assert results[1] == rates[(usd, eur, today)]

    # Test with missing rate (non-strict)
    queries = [(eur, usd, today), (eur, usd, date(2020, 1, 1))]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == rates[(eur, usd, today)]
    assert results[1] is None

    # Test with missing rate (strict)
    queries = [(eur, usd, today), (eur, usd, date(2020, 1, 1))]
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))

    # Test empty queries
    assert list(service.queries([])) == []


# LLM-generated content at query #27
#--------------------------

```python
def test_FXRateService_queries():
    # Create a mock FXRateService implementation for testing
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
                rate = self.query(ccy1, ccy2, asof, strict)
                results.append(rate)
            return results

    # Test data
    eur = Currency("EUR", "Euro", 2)
    usd = Currency("USD", "US Dollar", 2)
    date1 = Date(2023, 1, 1)
    date2 = Date(2023, 1, 2)

    rates = {
        (eur, usd, date1): FXRate(eur, usd, date1, Decimal("1.10")),
        (usd, eur, date1): FXRate(usd, eur, date1, Decimal("0.91")),
        (eur, usd, date2): FXRate(eur, usd, date2, Decimal("1.12")),
    }

    service = MockFXRateService(rates)

    # Test 1: Normal queries
    queries = [(eur, usd, date1), (usd, eur, date1), (eur, usd, date2)]
    results = list(service.queries(queries))
    assert len(results) == 3
    assert results[0] == FXRate(eur, usd, date1, Decimal("1.10"))
    assert results[1] == FXRate(usd, eur, date1, Decimal("0.91"))
    assert results[2] == FXRate(eur, usd, date2, Decimal("1.12"))

    # Test 2: Query with missing rate (non-strict)
    queries = [(eur, usd, date1), (eur, usd, Date(2023, 1, 3))]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == FXRate(eur, usd, date1, Decimal("1.10"))
    assert results[1] is None

    # Test 3: Query with missing rate (strict)
    queries = [(eur, usd, Date(2023, 1, 3))]
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))

    # Test 4: Empty queries
    assert list(service.queries([])) == []


# LLM-generated content at query #28
#--------------------------

```python
def test_FXRateService_queries():
    # Create a mock FXRateService implementation
    class MockFXRateService(FXRateService):
        def __init__(self):
            self.query_calls = []
            self.queries_calls = []

        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            self.query_calls.append((ccy1, ccy2, asof, strict))
            if ccy1.code == "EUR" and ccy2.code == "USD":
                return FXRate(Currencies["EUR"], Currencies["USD"], asof, Decimal("1.2"))
            return None

        def queries(self, queries: Iterable[TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            self.queries_calls.append((queries, strict))
            results = []
            for ccy1, ccy2, asof in queries:
                if ccy1.code == "EUR" and ccy2.code == "USD":
                    results.append(FXRate(Currencies["EUR"], Currencies["USD"], asof, Decimal("1.2")))
                else:
                    results.append(None)
            return results

    # Initialize the mock service
    service = MockFXRateService()

    # Test case 1: Basic queries
    queries = [
        (Currencies["EUR"], Currencies["USD"], Date(2023, 1, 1)),
        (Currencies["USD"], Currencies["EUR"], Date(2023, 1, 1)),
    ]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == FXRate(Currencies["EUR"], Currencies["USD"], Date(2023, 1, 1), Decimal("1.2"))
    assert results[1] is None

    # Test case 2: Empty queries
    assert list(service.queries([])) == []

    # Test case 3: Strict mode
    queries = [(Currencies["EUR"], Currencies["USD"], Date(2023, 1, 1))]
    results = list(service.queries(queries, strict=True))
    assert results[0] == FXRate(Currencies["EUR"], Currencies["USD"], Date(2023, 1, 1), Decimal("1.2"))

    # Verify that queries method was called correctly
    assert len(service.queries_calls) == 3
    assert service.queries_calls[0][0] == queries
    assert service.queries_calls[0][1] == False
    assert service.queries_calls[1][0] == []
    assert service.queries_calls[1][1] == False
    assert service.queries_calls[2][0] == queries
    assert service.queries_calls[2][1] == True


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_FXRateService_queries():
    # Create a concrete implementation of FXRateService for testing
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
            results = []
            for ccy1, ccy2, asof in queries:
                rate = self.query(ccy1, ccy2, asof, strict)
                results.append(rate)
            return results

    # Setup test data
    from datetime import date
    from pypara.currencies import Currencies

    EUR = Currencies["EUR"]
    USD = Currencies["USD"]
    GBP = Currencies["GBP"]

    test_rates = {
        (EUR, USD, date(2023, 1, 1)): FXRate(EUR, USD, date(2023, 1, 1), Decimal("1.1")),
        (USD, GBP, date(2023, 1, 1)): FXRate(USD, GBP, date(2023, 1, 1), Decimal("0.8")),
        (EUR, GBP, date(2023, 1, 2)): FXRate(EUR, GBP, date(2023, 1, 2), Decimal("0.88")),
    }

    service = TestFXRateService(test_rates)

    # Test case 1: All rates found
    queries1 = [
        (EUR, USD, date(2023, 1, 1)),
        (USD, GBP, date(2023, 1, 1)),
        (EUR, GBP, date(2023, 1, 2)),
    ]
    results1 = list(service.queries(queries1))
    assert len(results1) == 3
    assert results1[0] == test_rates[(EUR, USD, date(2023, 1, 1))]
    assert results1[1] == test_rates[(USD, GBP, date(2023, 1, 1))]
    assert results1[2] == test_rates[(EUR, GBP, date(2023, 1, 2))]

    # Test case 2: Some rates not found, strict=False
    queries2 = [
        (EUR, USD, date(2023, 1, 1)),
        (USD, GBP, date(2023, 1, 1)),
        (EUR, GBP, date(2023, 1, 3)),  # Not in test_rates
    ]
    results2 = list(service.queries(queries2))
    assert len(results2) == 3
    assert results2[0] == test_rates[(EUR, USD, date(2023, 1, 1))]
    assert results2[1] == test_rates[(USD, GBP, date(2023, 1, 1))]
    assert results2[2] is None

    # Test case 3: Some rates not found, strict=True
    queries3 = [
        (EUR, USD, date(2023, 1, 1)),
        (EUR, GBP, date(2023, 1, 3)),  # Not in test_rates
    ]
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries3, strict=True))

    # Test case 4: Empty queries
    assert list(service.queries([])) == []

    # Test case 5: Single query
    queries5 = [(EUR, USD, date(2023, 1, 1))]
    results5 = list(service.queries(queries5))
    assert len(results5) == 1
    assert results5[0] == test_rates[(EUR, USD, date(2023, 1, 1))]


# LLM-generated content at query #2
#--------------------------

```python
def test_FXRateService_queries():
    # Create a mock FXRateService subclass for testing
    class MockFXRateService(FXRateService):
        def __init__(self, rates):
            self.rates = rates

        def query(self, ccy1, ccy2, asof, strict=False):
            key = (ccy1, ccy2, asof)
            rate = self.rates.get(key)
            if rate is None and strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return rate

        def queries(self, queries, strict=False):
            results = []
            for ccy1, ccy2, asof in queries:
                rate = self.query(ccy1, ccy2, asof, strict)
                results.append(rate)
            return results

    # Create test data
    from datetime import date
    from decimal import Decimal
    from pypara.currencies import Currencies

    test_rates = {
        (Currencies["EUR"], Currencies["USD"], date(2023, 1, 1)): FXRate(Currencies["EUR"], Currencies["USD"], date(2023, 1, 1), Decimal("1.1")),
        (Currencies["USD"], Currencies["EUR"], date(2023, 1, 1)): FXRate(Currencies["USD"], Currencies["EUR"], date(2023, 1, 1), Decimal("0.9")),
        (Currencies["EUR"], Currencies["GBP"], date(2023, 1, 1)): FXRate(Currencies["EUR"], Currencies["GBP"], date(2023, 1, 1), Decimal("0.85")),
    }

    # Initialize the mock service
    service = MockFXRateService(test_rates)

    # Test case 1: Normal queries with existing rates
    queries = [
        (Currencies["EUR"], Currencies["USD"], date(2023, 1, 1)),
        (Currencies["USD"], Currencies["EUR"], date(2023, 1, 1)),
    ]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == test_rates[(Currencies["EUR"], Currencies["USD"], date(2023, 1, 1))]
    assert results[1] == test_rates[(Currencies["USD"], Currencies["EUR"], date(2023, 1, 1))]

    # Test case 2: Query with non-existent rate (non-strict mode)
    queries = [
        (Currencies["EUR"], Currencies["JPY"], date(2023, 1, 1)),
    ]
    results = list(service.queries(queries))
    assert len(results) == 1
    assert results[0] is None

    # Test case 3: Query with non-existent rate (strict mode)
    queries = [
        (Currencies["EUR"], Currencies["JPY"], date(2023, 1, 1)),
    ]
    try:
        list(service.queries(queries, strict=True))
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == Currencies["EUR"]
        assert e.ccy2 == Currencies["JPY"]
        assert e.asof == date(2023, 1, 1)

    # Test case 4: Empty queries
    results = list(service.queries([]))
    assert len(results) == 0

    # Test case 5: Mixed queries with some existing and some non-existing rates
    queries = [
        (Currencies["EUR"], Currencies["USD"], date(2023, 1, 1)),
        (Currencies["EUR"], Currencies["JPY"], date(2023, 1, 1)),
        (Currencies["EUR"], Currencies["GBP"], date(2023, 1, 1)),
    ]
    results = list(service.queries(queries))
    assert len(results) == 3
    assert results[0] == test_rates[(Currencies["EUR"], Currencies["USD"], date(2023, 1, 1))]
    assert results[1] is None
    assert results[2] == test_rates[(Currencies["EUR"], Currencies["GBP"], date(2023, 1, 1))]


# LLM-generated content at query #3
#--------------------------

```python
def test_FXRateService_queries():
    # Create a mock FXRateService
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            # Mock implementation
            if ccy1.code == "EUR" and ccy2.code == "USD" and asof == Date(2023, 1, 1):
                return FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1"))
            elif ccy1.code == "USD" and ccy2.code == "EUR" and asof == Date(2023, 1, 1):
                return FXRate(Currency("USD"), Currency("EUR"), Date(2023, 1, 1), Decimal("0.9"))
            elif strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            else:
                return None

        def queries(self, queries: Iterable[Tuple[Currency, Currency, Date]], strict: bool = False) -> Iterable[Optional[FXRate]]:
            # Mock implementation
            for ccy1, ccy2, asof in queries:
                yield self.query(ccy1, ccy2, asof, strict)

    # Initialize the mock service
    service = MockFXRateService()

    # Test with valid queries
    queries = [
        (Currency("EUR"), Currency("USD"), Date(2023, 1, 1)),
        (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)),
    ]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1"))
    assert results[1] == FXRate(Currency("USD"), Currency("EUR"), Date(2023, 1, 1), Decimal("0.9"))

    # Test with invalid queries (non-strict)
    queries = [
        (Currency("EUR"), Currency("GBP"), Date(2023, 1, 1)),
        (Currency("USD"), Currency("JPY"), Date(2023, 1, 1)),
    ]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] is None
    assert results[1] is None

    # Test with invalid queries (strict)
    queries = [
        (Currency("EUR"), Currency("GBP"), Date(2023, 1, 1)),
    ]
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))


# LLM-generated content at query #4
#--------------------------

```python
def test_FXRateService_query():
    # Create a concrete implementation of FXRateService for testing
    class TestFXRateService(FXRateService):
        def __init__(self):
            self.rates = {
                (Currency("EUR"), Currency("USD"), Date(2023, 1, 1)): FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1")),
                (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)): FXRate(Currency("USD"), Currency("EUR"), Date(2023, 1, 1), Decimal("0.9")),
            }

        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            key = (ccy1, ccy2, asof)
            if key in self.rates:
                return self.rates[key]
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None

        def queries(self, queries: Iterable[Tuple[Currency, Currency, Date]], strict: bool = False) -> Iterable[Optional[FXRate]]:
            for query in queries:
                yield self.query(*query, strict=strict)

    # Initialize the test service
    service = TestFXRateService()

    # Test successful query
    result = service.query(Currency("EUR"), Currency("USD"), Date(2023, 1, 1))
    assert result == FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1"))

    # Test inverted rate query
    result = service.query(Currency("USD"), Currency("EUR"), Date(2023, 1, 1))
    assert result == FXRate(Currency("USD"), Currency("EUR"), Date(2023, 1, 1), Decimal("0.9"))

    # Test non-existent rate query (non-strict)
    result = service.query(Currency("GBP"), Currency("USD"), Date(2023, 1, 1))
    assert result is None

    # Test non-existent rate query (strict)
    with pytest.raises(FXRateLookupError):
        service.query(Currency("GBP"), Currency("USD"), Date(2023, 1, 1), strict=True)


# LLM-generated content at query #5
#--------------------------

```python
def test_FXRateService_query():
    # Create a concrete implementation of FXRateService for testing
    class TestFXRateService(FXRateService):
        def __init__(self, rates):
            self.rates = rates

        def query(self, ccy1, ccy2, asof, strict=False):
            key = (ccy1, ccy2, asof)
            rate = self.rates.get(key)
            if rate is None and strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return rate

        def queries(self, queries, strict=False):
            results = []
            for query in queries:
                rate = self.query(*query, strict=strict)
                results.append(rate)
            return results

    # Setup test data
    from decimal import Decimal
    from pypara.currencies import Currencies
    import datetime

    test_date = datetime.date.today()
    eur = Currencies["EUR"]
    usd = Currencies["USD"]

    # Create test service with some rates
    test_rates = {
        (eur, usd, test_date): FXRate(eur, usd, test_date, Decimal("1.2")),
        (usd, eur, test_date): FXRate(usd, eur, test_date, Decimal("0.8333"))
    }
    service = TestFXRateService(test_rates)

    # Test 1: Successful query
    result = service.query(eur, usd, test_date)
    assert result == FXRate(eur, usd, test_date, Decimal("1.2"))

    # Test 2: Query with inverted currencies
    result = service.query(usd, eur, test_date)
    assert result == FXRate(usd, eur, test_date, Decimal("0.8333"))

    # Test 3: Query with non-existent rate (non-strict)
    result = service.query(eur, usd, test_date + datetime.timedelta(days=1))
    assert result is None

    # Test 4: Query with non-existent rate (strict)
    try:
        service.query(eur, usd, test_date + datetime.timedelta(days=1), strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == eur
        assert e.ccy2 == usd
        assert e.asof == test_date + datetime.timedelta(days=1)


# LLM-generated content at query #6
#--------------------------

```python
def test_FXRateService_query():
    # Create a mock FXRateService subclass for testing
    class MockFXRateService(FXRateService):
        def __init__(self, rates):
            self.rates = rates

        def query(self, ccy1, ccy2, asof, strict=False):
            key = (ccy1, ccy2, asof)
            rate = self.rates.get(key)
            if rate is None and strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return rate

        def queries(self, queries, strict=False):
            results = []
            for ccy1, ccy2, asof in queries:
                key = (ccy1, ccy2, asof)
                rate = self.rates.get(key)
                if rate is None and strict:
                    raise FXRateLookupError(ccy1, ccy2, asof)
                results.append(rate)
            return results

    # Setup test data
    from datetime import date
    from pypara.currencies import Currencies

    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    today = date.today()
    rate = FXRate(eur, usd, today, Decimal("1.2"))

    # Test case 1: Successful query
    service = MockFXRateService({(eur, usd, today): rate})
    result = service.query(eur, usd, today)
    assert result == rate

    # Test case 2: Query returns None when rate not found
    service = MockFXRateService({})
    result = service.query(eur, usd, today)
    assert result is None

    # Test case 3: Query raises FXRateLookupError when strict=True and rate not found
    service = MockFXRateService({})
    try:
        service.query(eur, usd, today, strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == eur
        assert e.ccy2 == usd
        assert e.asof == today

    # Test case 4: Query returns None when strict=False and rate not found
    service = MockFXRateService({})
    result = service.query(eur, usd, today, strict=False)
    assert result is None


# LLM-generated content at query #7
#--------------------------

```python
def test_FXRateService_query():
    # Create a concrete implementation of FXRateService for testing
    class TestFXRateService(FXRateService):
        def __init__(self, rates):
            self.rates = rates

        def query(self, ccy1, ccy2, asof, strict=False):
            key = (ccy1, ccy2, asof)
            rate = self.rates.get(key)
            if rate is None and strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return rate

        def queries(self, queries, strict=False):
            results = []
            for ccy1, ccy2, asof in queries:
                rate = self.query(ccy1, ccy2, asof, strict)
                results.append(rate)
            return results

    # Setup test data
    from datetime import date
    from pypara.currencies import Currencies
    EUR = Currencies["EUR"]
    USD = Currencies["USD"]
    GBP = Currencies["GBP"]
    today = date.today()

    rates = {
        (EUR, USD, today): FXRate(EUR, USD, today, Decimal("1.2")),
        (USD, GBP, today): FXRate(USD, GBP, today, Decimal("0.8")),
    }

    service = TestFXRateService(rates)

    # Test successful query
    result = service.query(EUR, USD, today)
    assert result == FXRate(EUR, USD, today, Decimal("1.2"))

    # Test query with inversion
    result = service.query(USD, EUR, today)
    assert result is None  # Not in our test data

    # Test query with strict=True raises exception
    with pytest.raises(FXRateLookupError):
        service.query(USD, EUR, today, strict=True)

    # Test query for non-existent rate
    result = service.query(EUR, GBP, today)
    assert result is None

    # Test query with strict=True for non-existent rate
    with pytest.raises(FXRateLookupError):
        service.query(EUR, GBP, today, strict=True)


# LLM-generated content at query #8
#--------------------------

```python
def test_FXRateService_queries():
    # Create a mock FXRateService subclass
    class MockFXRateService(FXRateService):
        def __init__(self, rates):
            self.rates = rates

        def query(self, ccy1, ccy2, asof, strict=False):
            for rate in self.rates:
                if rate.ccy1 == ccy1 and rate.ccy2 == ccy2 and rate.date == asof:
                    return rate
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None

        def queries(self, queries, strict=False):
            results = []
            for ccy1, ccy2, asof in queries:
                rate = self.query(ccy1, ccy2, asof, strict)
                results.append(rate)
            return results

    # Create test data
    eur = Currency("EUR", "Euro", 2)
    usd = Currency("USD", "US Dollar", 2)
    date1 = Date(2023, 1, 1)
    date2 = Date(2023, 1, 2)

    rate1 = FXRate(eur, usd, date1, Decimal("1.1"))
    rate2 = FXRate(eur, usd, date2, Decimal("1.2"))

    # Initialize service with test rates
    service = MockFXRateService([rate1, rate2])

    # Test queries with existing rates
    queries = [(eur, usd, date1), (eur, usd, date2)]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == rate1
    assert results[1] == rate2

    # Test queries with non-existing rate (non-strict)
    queries = [(eur, usd, Date(2023, 1, 3))]
    results = list(service.queries(queries))
    assert len(results) == 1
    assert results[0] is None

    # Test queries with non-existing rate (strict)
    queries = [(eur, usd, Date(2023, 1, 3))]
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))


# LLM-generated content at query #9
#--------------------------

```python
def test_FXRateService_queries():
    # Create a mock FXRateService implementation for testing
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
                key = (ccy1, ccy2, asof)
                if key in self.rates:
                    results.append(self.rates[key])
                else:
                    if strict:
                        raise FXRateLookupError(ccy1, ccy2, asof)
                    results.append(None)
            return results

    # Create test data
    from datetime import date
    from pypara.currencies import Currencies
    from decimal import Decimal

    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    gbp = Currencies["GBP"]
    today = date.today()

    rates = {
        (eur, usd, today): FXRate(eur, usd, today, Decimal("1.2")),
        (usd, gbp, today): FXRate(usd, gbp, today, Decimal("0.8")),
    }

    service = MockFXRateService(rates)

    # Test with existing rates
    queries = [
        (eur, usd, today),
        (usd, gbp, today),
    ]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == rates[(eur, usd, today)]
    assert results[1] == rates[(usd, gbp, today)]

    # Test with non-existing rate (non-strict)
    queries = [
        (eur, usd, today),
        (gbp, usd, today),
    ]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == rates[(eur, usd, today)]
    assert results[1] is None

    # Test with non-existing rate (strict)
    queries = [
        (eur, usd, today),
        (gbp, usd, today),
    ]
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))

    # Test empty queries
    assert list(service.queries([])) == []


# LLM-generated content at query #10
#--------------------------

```python
def test_FXRateService_query():
    # Create a concrete implementation of FXRateService for testing
    class TestFXRateService(FXRateService):
        def __init__(self, rates):
            self.rates = rates

        def query(self, ccy1, ccy2, asof, strict=False):
            key = (ccy1, ccy2, asof)
            rate = self.rates.get(key)
            if rate is None and strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return rate

        def queries(self, queries, strict=False):
            for query in queries:
                yield self.query(*query, strict=strict)

    # Setup test data
    from datetime import date
    from pypara.currencies import Currencies

    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    today = date.today()

    test_rates = {
        (eur, usd, today): FXRate(eur, usd, today, Decimal("1.2")),
        (usd, eur, today): FXRate(usd, eur, today, Decimal("0.8333")),
    }

    service = TestFXRateService(test_rates)

    # Test successful query
    result = service.query(eur, usd, today)
    assert result == test_rates[(eur, usd, today)]

    # Test inverted rate
    result = service.query(usd, eur, today)
    assert result == test_rates[(usd, eur, today)]

    # Test non-existent rate without strict
    result = service.query(eur, usd, date(2020, 1, 1))
    assert result is None

    # Test non-existent rate with strict
    try:
        service.query(eur, usd, date(2020, 1, 1), strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == eur
        assert e.ccy2 == usd
        assert e.asof == date(2020, 1, 1)


# LLM-generated content at query #11
#--------------------------

```python
def test_FXRateService_query():
    # Create a concrete implementation of FXRateService for testing
    class TestFXRateService(FXRateService):
        def __init__(self):
            self.rates = {
                (Currency("EUR"), Currency("USD"), Date(2023, 1, 1)): FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1")),
                (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)): FXRate(Currency("USD"), Currency("EUR"), Date(2023, 1, 1), Decimal("0.909")),
            }

        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            key = (ccy1, ccy2, asof)
            if key in self.rates:
                return self.rates[key]
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None

        def queries(self, queries: Iterable[Tuple[Currency, Currency, Date]], strict: bool = False) -> Iterable[Optional[FXRate]]:
            for query in queries:
                yield self.query(*query, strict=strict)

    service = TestFXRateService()

    # Test successful query
    result = service.query(Currency("EUR"), Currency("USD"), Date(2023, 1, 1))
    assert result == FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1"))

    # Test query with inverted currencies
    result = service.query(Currency("USD"), Currency("EUR"), Date(2023, 1, 1))
    assert result == FXRate(Currency("USD"), Currency("EUR"), Date(2023, 1, 1), Decimal("0.909"))

    # Test query with non-existent rate (non-strict)
    result = service.query(Currency("GBP"), Currency("USD"), Date(2023, 1, 1))
    assert result is None

    # Test query with non-existent rate (strict)
    with pytest.raises(FXRateLookupError):
        service.query(Currency("GBP"), Currency("USD"), Date(2023, 1, 1), strict=True)


# LLM-generated content at query #12
#--------------------------

```python
def test_FXRateService_queries():
    # Mock implementation of FXRateService for testing
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            if ccy1.code == "EUR" and ccy2.code == "USD" and asof == Date(2023, 1, 1):
                return FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1"))
            elif ccy1.code == "USD" and ccy2.code == "EUR" and asof == Date(2023, 1, 1):
                return FXRate(Currency("USD"), Currency("EUR"), Date(2023, 1, 1), Decimal("0.9"))
            elif strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None

        def queries(self, queries: Iterable[Tuple[Currency, Currency, Date]], strict: bool = False) -> Iterable[Optional[FXRate]]:
            for ccy1, ccy2, asof in queries:
                yield self.query(ccy1, ccy2, asof, strict)

    # Create mock service instance
    service = MockFXRateService()

    # Test with valid queries
    queries = [
        (Currency("EUR"), Currency("USD"), Date(2023, 1, 1)),
        (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)),
    ]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1"))
    assert results[1] == FXRate(Currency("USD"), Currency("EUR"), Date(2023, 1, 1), Decimal("0.9"))

    # Test with invalid query (non-strict mode)
    invalid_queries = [
        (Currency("GBP"), Currency("JPY"), Date(2023, 1, 1)),
    ]
    results = list(service.queries(invalid_queries))
    assert len(results) == 1
    assert results[0] is None

    # Test with invalid query (strict mode)
    try:
        list(service.queries(invalid_queries, strict=True))
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1.code == "GBP"
        assert e.ccy2.code == "JPY"
        assert e.asof == Date(2023, 1, 1)


# LLM-generated content at query #13
#--------------------------

```python
def test_FXRateService_query():
    # Create a concrete implementation of FXRateService for testing
    class TestFXRateService(FXRateService):
        def __init__(self):
            self.rates = {
                (Currency("EUR"), Currency("USD"), Date(2023, 1, 1)): FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.10")),
                (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)): FXRate(Currency("USD"), Currency("EUR"), Date(2023, 1, 1), Decimal("0.9091")),
            }

        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            key = (ccy1, ccy2, asof)
            if key in self.rates:
                return self.rates[key]
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None

        def queries(self, queries: Iterable[Tuple[Currency, Currency, Date]], strict: bool = False) -> Iterable[Optional[FXRate]]:
            for query in queries:
                yield self.query(*query, strict=strict)

    # Initialize the test service
    service = TestFXRateService()

    # Test successful query
    rate = service.query(Currency("EUR"), Currency("USD"), Date(2023, 1, 1))
    assert rate is not None
    assert rate.ccy1 == Currency("EUR")
    assert rate.ccy2 == Currency("USD")
    assert rate.date == Date(2023, 1, 1)
    assert rate.value == Decimal("1.10")

    # Test query with non-existent rate (non-strict)
    rate = service.query(Currency("GBP"), Currency("USD"), Date(2023, 1, 1))
    assert rate is None

    # Test query with non-existent rate (strict)
    try:
        service.query(Currency("GBP"), Currency("USD"), Date(2023, 1, 1), strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == Currency("GBP")
        assert e.ccy2 == Currency("USD")
        assert e.asof == Date(2023, 1, 1)


# LLM-generated content at query #14
#--------------------------

```python
def test_FXRateService_queries():
    # Create a mock FXRateService implementation
    class MockFXRateService(FXRateService):
        def __init__(self):
            self.rates = {
                (Currency("EUR"), Currency("USD"), Date(2023, 1, 1)): FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1")),
                (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)): FXRate(Currency("USD"), Currency("EUR"), Date(2023, 1, 1), Decimal("0.909")),
            }

        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            return self.rates.get((ccy1, ccy2, asof))

        def queries(self, queries: Iterable[Tuple[Currency, Currency, Date]], strict: bool = False) -> Iterable[Optional[FXRate]]:
            for ccy1, ccy2, asof in queries:
                rate = self.query(ccy1, ccy2, asof)
                if rate is None and strict:
                    raise FXRateLookupError(ccy1, ccy2, asof)
                yield rate

    # Initialize the mock service
    service = MockFXRateService()

    # Test case 1: Valid queries
    queries = [
        (Currency("EUR"), Currency("USD"), Date(2023, 1, 1)),
        (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)),
    ]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1"))
    assert results[1] == FXRate(Currency("USD"), Currency("EUR"), Date(2023, 1, 1), Decimal("0.909"))

    # Test case 2: Query with missing rate (non-strict)
    queries = [
        (Currency("EUR"), Currency("USD"), Date(2023, 1, 1)),
        (Currency("GBP"), Currency("JPY"), Date(2023, 1, 1)),
    ]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1"))
    assert results[1] is None

    # Test case 3: Query with missing rate (strict)
    queries = [
        (Currency("EUR"), Currency("USD"), Date(2023, 1, 1)),
        (Currency("GBP"), Currency("JPY"), Date(2023, 1, 1)),
    ]
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))

    # Test case 4: Empty queries
    results = list(service.queries([]))
    assert len(results) == 0


# LLM-generated content at query #15
#--------------------------

```python
def test_FXRateService_queries():
    # Create a mock FXRateService subclass for testing
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            # Mock implementation for testing
            if ccy1.code == "EUR" and ccy2.code == "USD" and asof == Date(2023, 1, 1):
                return FXRate(Currencies["EUR"], Currencies["USD"], Date(2023, 1, 1), Decimal("1.1"))
            elif ccy1.code == "USD" and ccy2.code == "GBP" and asof == Date(2023, 1, 1):
                return FXRate(Currencies["USD"], Currencies["GBP"], Date(2023, 1, 1), Decimal("0.8"))
            elif strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            else:
                return None

        def queries(self, queries: Iterable[Tuple[Currency, Currency, Date]], strict: bool = False) -> Iterable[Optional[FXRate]]:
            # Mock implementation for testing
            results = []
            for ccy1, ccy2, asof in queries:
                result = self.query(ccy1, ccy2, asof, strict)
                results.append(result)
            return results

    # Initialize the mock service
    service = MockFXRateService()

    # Test case 1: Successful queries
    queries = [
        (Currencies["EUR"], Currencies["USD"], Date(2023, 1, 1)),
        (Currencies["USD"], Currencies["GBP"], Date(2023, 1, 1))
    ]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == FXRate(Currencies["EUR"], Currencies["USD"], Date(2023, 1, 1), Decimal("1.1"))
    assert results[1] == FXRate(Currencies["USD"], Currencies["GBP"], Date(2023, 1, 1), Decimal("0.8"))

    # Test case 2: Non-strict query with missing rate
    queries = [
        (Currencies["EUR"], Currencies["USD"], Date(2023, 1, 1)),
        (Currencies["EUR"], Currencies["JPY"], Date(2023, 1, 1))
    ]
    results = list(service.queries(queries, strict=False))
    assert len(results) == 2
    assert results[0] == FXRate(Currencies["EUR"], Currencies["USD"], Date(2023, 1, 1), Decimal("1.1"))
    assert results[1] is None

    # Test case 3: Strict query with missing rate
    queries = [
        (Currencies["EUR"], Currencies["USD"], Date(2023, 1, 1)),
        (Currencies["EUR"], Currencies["JPY"], Date(2023, 1, 1))
    ]
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))

    # Test case 4: Empty queries
    results = list(service.queries([]))
    assert len(results) == 0


# LLM-generated content at query #16
#--------------------------

```python
def test_FXRateService_queries():
    # Create a mock FXRateService subclass for testing
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
                key = (ccy1, ccy2, asof)
                if key in self.rates:
                    results.append(self.rates[key])
                else:
                    if strict:
                        raise FXRateLookupError(ccy1, ccy2, asof)
                    results.append(None)
            return results

    # Setup test data
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currencies

    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    gbp = Currencies["GBP"]
    today = date.today()
    yesterday = today - timedelta(days=1)

    rate1 = FXRate(eur, usd, today, Decimal("1.2"))
    rate2 = FXRate(eur, gbp, today, Decimal("0.8"))
    rate3 = FXRate(usd, gbp, yesterday, Decimal("0.7"))

    rates = {
        (eur, usd, today): rate1,
        (eur, gbp, today): rate2,
        (usd, gbp, yesterday): rate3,
    }

    service = MockFXRateService(rates)

    # Test 1: Normal queries with existing rates
    queries = [
        (eur, usd, today),
        (eur, gbp, today),
        (usd, gbp, yesterday),
    ]
    results = service.queries(queries)
    assert list(results) == [rate1, rate2, rate3]

    # Test 2: Queries with non-existing rates (non-strict)
    queries = [
        (eur, usd, today),
        (eur, gbp, yesterday),  # Non-existing
        (usd, gbp, yesterday),
    ]
    results = service.queries(queries)
    assert list(results) == [rate1, None, rate3]

    # Test 3: Queries with non-existing rates (strict)
    queries = [
        (eur, usd, today),
        (eur, gbp, yesterday),  # Non-existing
        (usd, gbp, yesterday),
    ]
    with pytest.raises(FXRateLookupError):
        service.queries(queries, strict=True)

    # Test 4: Empty queries
    assert list(service.queries([])) == []

    # Test 5: Inverted rates
    queries = [
        (usd, eur, today),  # Inverted of rate1
    ]
    results = service.queries(queries)
    assert list(results) == [None]  # Should return None as we don't have the inverted rate

    # Test 6: Same currency queries
    queries = [
        (eur, eur, today),  # Same currency
    ]
    results = service.queries(queries)
    assert list(results) == [None]  # Should return None as we don't have this specific rate


# LLM-generated content at query #17
#--------------------------

```python
def test_FXRateService_query():
    # Create a mock FXRateService implementation for testing
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
            for ccy1, ccy2, asof in queries:
                yield self.query(ccy1, ccy2, asof, strict)

    # Test data
    eur = Currency("EUR", "Euro", 2)
    usd = Currency("USD", "US Dollar", 2)
    today = Date.today()
    rate = FXRate(eur, usd, today, Decimal("1.2"))

    # Test case 1: Rate exists
    service = MockFXRateService({(eur, usd, today): rate})
    result = service.query(eur, usd, today)
    assert result == rate

    # Test case 2: Rate does not exist, strict=False
    service = MockFXRateService({})
    result = service.query(eur, usd, today, strict=False)
    assert result is None

    # Test case 3: Rate does not exist, strict=True
    service = MockFXRateService({})
    with pytest.raises(FXRateLookupError):
        service.query(eur, usd, today, strict=True)


# LLM-generated content at query #18
#--------------------------

```python
def test_FXRateService_queries():
    # Create a mock FXRateService subclass for testing
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
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    # Setup test data
    from pypara.currencies import Currencies
    from decimal import Decimal
    import datetime

    EUR = Currencies["EUR"]
    USD = Currencies["USD"]
    GBP = Currencies["GBP"]
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)

    rates = {
        (EUR, USD, today): FXRate(EUR, USD, today, Decimal("1.2")),
        (USD, GBP, today): FXRate(USD, GBP, today, Decimal("0.8")),
        (EUR, GBP, yesterday): FXRate(EUR, GBP, yesterday, Decimal("0.9")),
    }

    service = MockFXRateService(rates)

    # Test 1: Basic query with existing rates
    queries = [(EUR, USD, today), (USD, GBP, today)]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == rates[(EUR, USD, today)]
    assert results[1] == rates[(USD, GBP, today)]

    # Test 2: Query with non-existing rate (non-strict)
    queries = [(EUR, USD, today), (EUR, GBP, today)]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == rates[(EUR, USD, today)]
    assert results[1] is None

    # Test 3: Query with non-existing rate (strict)
    queries = [(EUR, USD, today), (EUR, GBP, today)]
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))

    # Test 4: Empty queries
    assert list(service.queries([])) == []

    # Test 5: Query with all non-existing rates (non-strict)
    queries = [(GBP, USD, today), (EUR, USD, yesterday)]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert all(r is None for r in results)

    # Test 6: Query with all non-existing rates (strict)
    queries = [(GBP, USD, today), (EUR, USD, yesterday)]
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))


# LLM-generated content at query #19
#--------------------------

```python
def test_FXRateService_queries():
    # Create a concrete implementation of FXRateService for testing
    class TestFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            # Mock implementation for testing
            if ccy1.code == "EUR" and ccy2.code == "USD" and asof == Date(2023, 1, 1):
                return FXRate(ccy1, ccy2, asof, Decimal("1.1"))
            elif ccy1.code == "USD" and ccy2.code == "EUR" and asof == Date(2023, 1, 1):
                return FXRate(ccy1, ccy2, asof, Decimal("0.909"))
            elif strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            else:
                return None

        def queries(self, queries: Iterable[Tuple[Currency, Currency, Date]], strict: bool = False) -> Iterable[Optional[FXRate]]:
            # Mock implementation for testing
            for ccy1, ccy2, asof in queries:
                yield self.query(ccy1, ccy2, asof, strict)

    # Create test data
    eur = Currency("EUR", "Euro", 2)
    usd = Currency("USD", "US Dollar", 2)
    date1 = Date(2023, 1, 1)
    date2 = Date(2023, 1, 2)

    # Test case 1: Successful queries
    queries = [(eur, usd, date1), (usd, eur, date1)]
    service = TestFXRateService()
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == FXRate(eur, usd, date1, Decimal("1.1"))
    assert results[1] == FXRate(usd, eur, date1, Decimal("0.909"))

    # Test case 2: Failed queries (non-strict)
    queries = [(eur, usd, date2), (usd, eur, date2)]
    results = list(service.queries(queries, strict=False))
    assert len(results) == 2
    assert results[0] is None
    assert results[1] is None

    # Test case 3: Failed queries (strict)
    queries = [(eur, usd, date2), (usd, eur, date2)]
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))

    # Test case 4: Empty queries
    queries = []
    results = list(service.queries(queries))
    assert len(results) == 0


# LLM-generated content at query #20
#--------------------------

```python
def test_FXRateService_queries():
    # Create a mock FXRateService implementation for testing
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            # Mock implementation that returns a fixed rate for EUR/USD
            if ccy1.code == "EUR" and ccy2.code == "USD":
                return FXRate(ccy1, ccy2, asof, Decimal("1.2"))
            return None

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            # Mock implementation that processes each query
            results = []
            for ccy1, ccy2, asof in queries:
                result = self.query(ccy1, ccy2, asof, strict)
                if result is None and strict:
                    raise FXRateLookupError(ccy1, ccy2, asof)
                results.append(result)
            return results

    # Create mock currencies and dates
    eur = Currency("EUR", "Euro", 2)
    usd = Currency("USD", "US Dollar", 2)
    today = Date.today()

    # Create the service instance
    service = MockFXRateService()

    # Test case 1: Single query
    queries = [(eur, usd, today)]
    results = list(service.queries(queries))
    assert len(results) == 1
    assert results[0] == FXRate(eur, usd, today, Decimal("1.2"))

    # Test case 2: Multiple queries with same pair
    queries = [(eur, usd, today), (eur, usd, today)]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert all(r == FXRate(eur, usd, today, Decimal("1.2")) for r in results)

    # Test case 3: Query with non-existent rate
    gbp = Currency("GBP", "British Pound", 2)
    queries = [(eur, gbp, today)]
    results = list(service.queries(queries))
    assert len(results) == 1
    assert results[0] is None

    # Test case 4: Strict mode with non-existent rate
    queries = [(eur, gbp, today)]
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))

    # Test case 5: Mixed queries
    queries = [(eur, usd, today), (eur, gbp, today), (usd, eur, today)]
    results = list(service.queries(queries))
    assert len(results) == 3
    assert results[0] == FXRate(eur, usd, today, Decimal("1.2"))
    assert results[1] is None
    assert results[2] == FXRate(usd, eur, today, Decimal("1.2") ** -1)


# LLM-generated content at query #21
#--------------------------

```python
def test_FXRateService_query():
    # Setup
    class MockFXRateService(FXRateService):
        def __init__(self, rates):
            self.rates = rates

        def query(self, ccy1, ccy2, asof, strict=False):
            key = (ccy1, ccy2, asof)
            rate = self.rates.get(key)
            if rate is None and strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return rate

        def queries(self, queries, strict=False):
            return (self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries)

    from datetime import date
    from pypara.currencies import Currencies

    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    today = date.today()
    rate = FXRate(eur, usd, today, Decimal("1.2"))

    service = MockFXRateService({(eur, usd, today): rate})

    # Test successful query
    result = service.query(eur, usd, today)
    assert result == rate

    # Test non-strict query with missing rate
    result = service.query(usd, eur, today)
    assert result is None

    # Test strict query with missing rate
    with pytest.raises(FXRateLookupError):
        service.query(usd, eur, today, strict=True)

    # Test inverted rate
    inverted_rate = ~rate
    service_with_inverted = MockFXRateService({(usd, eur, today): inverted_rate})
    result = service_with_inverted.query(usd, eur, today)
    assert result == inverted_rate


# LLM-generated content at query #22
#--------------------------

```python
def test_FXRateService_query():
    # Create a concrete implementation of FXRateService for testing
    class TestFXRateService(FXRateService):
        def __init__(self):
            self.rates = {
                (Currency("EUR"), Currency("USD"), Date(2023, 1, 1)): FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.10")),
                (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)): FXRate(Currency("USD"), Currency("EUR"), Date(2023, 1, 1), Decimal("0.90")),
            }

        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            return self.rates.get((ccy1, ccy2, asof))

        def queries(self, queries: Iterable[Tuple[Currency, Currency, Date]], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return (self.query(ccy1, ccy2, asof) for ccy1, ccy2, asof in queries)

    # Initialize the test service
    service = TestFXRateService()

    # Test successful query
    rate = service.query(Currency("EUR"), Currency("USD"), Date(2023, 1, 1))
    assert rate == FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.10"))

    # Test inverted query
    rate = service.query(Currency("USD"), Currency("EUR"), Date(2023, 1, 1))
    assert rate == FXRate(Currency("USD"), Currency("EUR"), Date(2023, 1, 1), Decimal("0.90"))

    # Test non-existent rate query with strict=False
    rate = service.query(Currency("GBP"), Currency("JPY"), Date(2023, 1, 1), strict=False)
    assert rate is None

    # Test non-existent rate query with strict=True
    try:
        service.query(Currency("GBP"), Currency("JPY"), Date(2023, 1, 1), strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == Currency("GBP")
        assert e.ccy2 == Currency("JPY")
        assert e.asof == Date(2023, 1, 1)


# LLM-generated content at query #23
#--------------------------

```python
def test_FXRateService_queries():
    # Create a concrete implementation of FXRateService for testing
    class TestFXRateService(FXRateService):
        def __init__(self, rates):
            self.rates = rates

        def query(self, ccy1, ccy2, asof, strict=False):
            for rate in self.rates:
                if rate.ccy1 == ccy1 and rate.ccy2 == ccy2 and rate.date == asof:
                    return rate
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None

        def queries(self, queries, strict=False):
            results = []
            for ccy1, ccy2, asof in queries:
                rate = self.query(ccy1, ccy2, asof, strict)
                results.append(rate)
            return results

    # Test data
    eur = Currency("EUR", "Euro", 2)
    usd = Currency("USD", "US Dollar", 2)
    today = Date.today()
    yesterday = today - Date.resolution

    rate1 = FXRate(eur, usd, today, Decimal("1.2"))
    rate2 = FXRate(usd, eur, yesterday, Decimal("0.8"))

    # Initialize service with test rates
    service = TestFXRateService([rate1, rate2])

    # Test successful queries
    queries = [
        (eur, usd, today),
        (usd, eur, yesterday)
    ]
    results = service.queries(queries)
    assert len(results) == 2
    assert results[0] == rate1
    assert results[1] == rate2

    # Test with non-existent rate (non-strict)
    queries = [
        (eur, usd, yesterday),
        (usd, eur, today)
    ]
    results = service.queries(queries)
    assert len(results) == 2
    assert results[0] is None
    assert results[1] is None

    # Test with non-existent rate (strict)
    queries = [
        (eur, usd, yesterday)
    ]
    with pytest.raises(FXRateLookupError):
        service.queries(queries, strict=True)

    # Test empty queries
    assert list(service.queries([])) == []


# LLM-generated content at query #24
#--------------------------

```python
def test_FXRateService_query():
    # Create a mock FXRateService implementation for testing
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
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    # Test data
    test_currency1 = Currency("EUR", "Euro", "€")
    test_currency2 = Currency("USD", "US Dollar", "$")
    test_date = Date(2023, 1, 1)
    test_rate = FXRate(test_currency1, test_currency2, test_date, Decimal("1.1"))

    # Initialize mock service with test data
    service = MockFXRateService({(test_currency1, test_currency2, test_date): test_rate})

    # Test successful query
    result = service.query(test_currency1, test_currency2, test_date)
    assert result == test_rate

    # Test query with non-existent rate (non-strict)
    non_existent_date = Date(2023, 1, 2)
    result = service.query(test_currency1, test_currency2, non_existent_date)
    assert result is None

    # Test query with non-existent rate (strict)
    with pytest.raises(FXRateLookupError):
        service.query(test_currency1, test_currency2, non_existent_date, strict=True)

    # Test query with inverted currencies
    inverted_rate = ~test_rate
    service_with_inverted = MockFXRateService({(test_currency2, test_currency1, test_date): inverted_rate})
    result = service_with_inverted.query(test_currency2, test_currency1, test_date)
    assert result == inverted_rate


# LLM-generated content at query #25
#--------------------------

```python
def test_FXRateService_query():
    # Create a mock FXRateService subclass for testing
    class MockFXRateService(FXRateService):
        def __init__(self, rates):
            self.rates = rates

        def query(self, ccy1, ccy2, asof, strict=False):
            key = (ccy1, ccy2, asof)
            rate = self.rates.get(key)
            if rate is None and strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return rate

        def queries(self, queries, strict=False):
            for query in queries:
                yield self.query(*query, strict=strict)

    # Test data
    eur = Currency("EUR", "Euro", 2)
    usd = Currency("USD", "US Dollar", 2)
    today = Date.today()
    rate = FXRate(eur, usd, today, Decimal("1.2"))

    # Test with existing rate
    service = MockFXRateService({(eur, usd, today): rate})
    result = service.query(eur, usd, today)
    assert result == rate

    # Test with non-existing rate (non-strict)
    result = service.query(usd, eur, today)
    assert result is None

    # Test with non-existing rate (strict)
    try:
        service.query(usd, eur, today, strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == usd
        assert e.ccy2 == eur
        assert e.asof == today


# LLM-generated content at query #26
#--------------------------

```python
def test_FXRateService_queries():
    # Create a mock FXRateService implementation for testing
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
                key = (ccy1, ccy2, asof)
                if key in self.rates:
                    results.append(self.rates[key])
                else:
                    if strict:
                        raise FXRateLookupError(ccy1, ccy2, asof)
                    results.append(None)
            return results

    # Setup test data
    from datetime import date
    from pypara.currencies import Currencies

    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    today = date.today()

    rates = {
        (eur, usd, today): FXRate(eur, usd, today, Decimal("1.2")),
        (usd, eur, today): FXRate(usd, eur, today, Decimal("0.8333")),
    }

    service = MockFXRateService(rates)

    # Test 1: Basic query with existing rates
    queries = [(eur, usd, today), (usd, eur, today)]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == rates[(eur, usd, today)]
    assert results[1] == rates[(usd, eur, today)]

    # Test 2: Query with non-existing rate (non-strict mode)
    queries = [(eur, usd, today), (eur, usd, date(2020, 1, 1))]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == rates[(eur, usd, today)]
    assert results[1] is None

    # Test 3: Query with non-existing rate (strict mode)
    queries = [(eur, usd, date(2020, 1, 1))]
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))

    # Test 4: Empty query list
    assert list(service.queries([])) == []


# LLM-generated content at query #27
#--------------------------

```python
def test_FXRateService_queries():
    # Create a concrete implementation of FXRateService for testing
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
            results = []
            for ccy1, ccy2, asof in queries:
                rate = self.query(ccy1, ccy2, asof, strict)
                results.append(rate)
            return results

    # Setup test data
    from datetime import date
    from pypara.currencies import Currencies
    from decimal import Decimal

    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    gbp = Currencies["GBP"]
    today = date.today()

    rates = {
        (eur, usd, today): FXRate(eur, usd, today, Decimal("1.20")),
        (usd, gbp, today): FXRate(usd, gbp, today, Decimal("0.80")),
    }

    service = TestFXRateService(rates)

    # Test case 1: Query existing rates
    queries = [(eur, usd, today), (usd, gbp, today)]
    results = service.queries(queries)
    assert len(results) == 2
    assert results[0] == FXRate(eur, usd, today, Decimal("1.20"))
    assert results[1] == FXRate(usd, gbp, today, Decimal("0.80"))

    # Test case 2: Query with non-existing rate (non-strict)
    queries = [(eur, usd, today), (gbp, usd, today)]
    results = service.queries(queries)
    assert len(results) == 2
    assert results[0] == FXRate(eur, usd, today, Decimal("1.20"))
    assert results[1] is None

    # Test case 3: Query with non-existing rate (strict)
    queries = [(eur, usd, today), (gbp, usd, today)]
    with pytest.raises(FXRateLookupError):
        service.queries(queries, strict=True)

    # Test case 4: Empty queries
    assert list(service.queries([])) == []


