####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_FXRate___invert__():
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currencies

    # Test basic inversion
    rate = FXRate(Currencies["EUR"], Currencies["USD"], date(2023, 1, 1), Decimal("2"))
    inverted = ~rate
    assert inverted.ccy1 == Currencies["USD"]
    assert inverted.ccy2 == Currencies["EUR"]
    assert inverted.date == date(2023, 1, 1)
    assert inverted.value == Decimal("0.5")

    # Test inversion of inverted rate
    double_inverted = ~inverted
    assert double_inverted == rate

    # Test inversion with different values
    rate2 = FXRate(Currencies["GBP"], Currencies["JPY"], date(2023, 1, 1), Decimal("150.25"))
    inverted2 = ~rate2
    assert inverted2.value == Decimal("1") / Decimal("150.25")

    # Test inversion with same currency (should be 1)
    rate3 = FXRate(Currencies["EUR"], Currencies["EUR"], date(2023, 1, 1), Decimal("1"))
    inverted3 = ~rate3
    assert inverted3.value == Decimal("1")


# LLM-generated content at query #2
#--------------------------

```python
def test_FXRate___invert__():
    # Test inversion of FXRate
    rate = FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("2"))
    inverted_rate = ~rate
    assert inverted_rate.ccy1 == Currency("USD")
    assert inverted_rate.ccy2 == Currency("EUR")
    assert inverted_rate.date == Date(2023, 1, 1)
    assert inverted_rate.value == Decimal("0.5")

    # Test inversion of inverted rate
    double_inverted_rate = ~inverted_rate
    assert double_inverted_rate == rate

    # Test inversion with different values
    rate2 = FXRate(Currency("GBP"), Currency("JPY"), Date(2023, 1, 1), Decimal("150.25"))
    inverted_rate2 = ~rate2
    assert inverted_rate2.ccy1 == Currency("JPY")
    assert inverted_rate2.ccy2 == Currency("GBP")
    assert inverted_rate2.date == Date(2023, 1, 1)
    assert inverted_rate2.value == Decimal("0.006655102040816326")  # 1 / 150.25

    # Test inversion of same currency (should remain 1)
    rate3 = FXRate(Currency("EUR"), Currency("EUR"), Date(2023, 1, 1), Decimal("1"))
    inverted_rate3 = ~rate3
    assert inverted_rate3 == rate3


# LLM-generated content at query #3
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
    test_rate = FXRate(eur, usd, today, Decimal("1.2"))

    # Test successful query
    service = TestFXRateService({(eur, usd, today): test_rate})
    result = service.query(eur, usd, today)
    assert result == test_rate

    # Test query with non-existent rate (non-strict)
    result = service.query(usd, eur, today)
    assert result is None

    # Test query with non-existent rate (strict)
    with pytest.raises(FXRateLookupError):
        service.query(usd, eur, today, strict=True)

    # Test query with same currency
    same_currency_rate = FXRate(eur, eur, today, Decimal("1"))
    service = TestFXRateService({(eur, eur, today): same_currency_rate})
    result = service.query(eur, eur, today)
    assert result == same_currency_rate


# LLM-generated content at query #4
#--------------------------

```python
def test_FXRateService_queries():
    # Create a concrete implementation of FXRateService for testing
    class TestFXRateService(FXRateService):
        def __init__(self):
            self.rates = {
                (Currency("EUR"), Currency("USD"), Date(2023, 1, 1)): FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1")),
                (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)): FXRate(Currency("USD"), Currency("EUR"), Date(2023, 1, 1), Decimal("0.9")),
                (Currency("EUR"), Currency("GBP"), Date(2023, 1, 1)): FXRate(Currency("EUR"), Currency("GBP"), Date(2023, 1, 1), Decimal("0.8")),
            }

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

    # Initialize the test service
    service = TestFXRateService()

    # Test with existing rates
    queries = [
        (Currency("EUR"), Currency("USD"), Date(2023, 1, 1)),
        (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)),
        (Currency("EUR"), Currency("GBP"), Date(2023, 1, 1)),
    ]
    results = service.queries(queries)
    assert len(results) == 3
    assert results[0] == FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1"))
    assert results[1] == FXRate(Currency("USD"), Currency("EUR"), Date(2023, 1, 1), Decimal("0.9"))
    assert results[2] == FXRate(Currency("EUR"), Currency("GBP"), Date(2023, 1, 1), Decimal("0.8"))

    # Test with non-existing rates (non-strict mode)
    queries = [
        (Currency("EUR"), Currency("JPY"), Date(2023, 1, 1)),
    ]
    results = service.queries(queries)
    assert len(results) == 1
    assert results[0] is None

    # Test with non-existing rates (strict mode)
    queries = [
        (Currency("EUR"), Currency("JPY"), Date(2023, 1, 1)),
    ]
    with pytest.raises(FXRateLookupError):
        service.queries(queries, strict=True)


# LLM-generated content at query #5
#--------------------------

```python
def test_FXRateService_queries():
    # Create a mock FXRateService subclass for testing
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

    # Test data
    date1 = Date(2023, 1, 1)
    date2 = Date(2023, 1, 2)
    eur = Currency("EUR")
    usd = Currency("USD")
    gbp = Currency("GBP")

    rates = [
        FXRate(eur, usd, date1, Decimal("1.10")),
        FXRate(eur, gbp, date1, Decimal("0.90")),
        FXRate(usd, gbp, date2, Decimal("0.80")),
    ]

    service = MockFXRateService(rates)

    # Test successful queries
    queries = [
        (eur, usd, date1),
        (eur, gbp, date1),
        (usd, gbp, date2),
    ]
    results = service.queries(queries)
    assert len(results) == 3
    assert results[0] == rates[0]
    assert results[1] == rates[1]
    assert results[2] == rates[2]

    # Test with missing rate
    queries_with_missing = [
        (eur, usd, date1),
        (gbp, usd, date1),  # Missing
    ]
    results = service.queries(queries_with_missing)
    assert len(results) == 2
    assert results[0] == rates[0]
    assert results[1] is None

    # Test strict mode with missing rate
    try:
        service.queries(queries_with_missing, strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == gbp
        assert e.ccy2 == usd
        assert e.asof == date1

    # Test empty queries
    assert list(service.queries([])) == []


# LLM-generated content at query #6
#--------------------------

```python
def test_FXRateService_query():
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
    test_date = date(2023, 1, 1)
    test_rates = {
        (EUR, USD, test_date): FXRate(EUR, USD, test_date, Decimal("1.1")),
        (USD, GBP, test_date): FXRate(USD, GBP, test_date, Decimal("0.8")),
    }

    # Initialize service
    service = TestFXRateService(test_rates)

    # Test successful query
    result = service.query(EUR, USD, test_date)
    assert result == test_rates[(EUR, USD, test_date)]

    # Test query with non-existent rate (non-strict)
    result = service.query(EUR, GBP, test_date)
    assert result is None

    # Test query with non-existent rate (strict)
    try:
        service.query(EUR, GBP, test_date, strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == EUR
        assert e.ccy2 == GBP
        assert e.asof == test_date

    # Test inverted rate query
    inverted_rate = ~test_rates[(EUR, USD, test_date)]
    result = service.query(USD, EUR, test_date)
    assert result is None  # Not in our test data
    # But if we add it:
    service.rates[(USD, EUR, test_date)] = inverted_rate
    result = service.query(USD, EUR, test_date)
    assert result == inverted_rate


# LLM-generated content at query #7
#--------------------------

```python
def test_FXRateService_queries():
    # Create a mock FXRateService implementation for testing
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            # Mock implementation that returns a fixed rate for EUR/USD
            if ccy1.code == "EUR" and ccy2.code == "USD" and asof == Date(2023, 1, 1):
                return FXRate.of(ccy1, ccy2, asof, Decimal("1.10"))
            elif strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None

        def queries(self, queries: Iterable[Tuple[Currency, Currency, Date]], strict: bool = False) -> Iterable[Optional[FXRate]]:
            # Mock implementation that processes each query
            results = []
            for ccy1, ccy2, asof in queries:
                try:
                    result = self.query(ccy1, ccy2, asof, strict)
                    results.append(result)
                except FXRateLookupError:
                    if strict:
                        raise
                    results.append(None)
            return results

    # Initialize the mock service
    service = MockFXRateService()

    # Test case 1: Successful queries
    queries = [
        (Currency("EUR"), Currency("USD"), Date(2023, 1, 1)),
        (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)),
    ]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == FXRate.of(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.10"))
    assert results[1] is None  # USD/EUR not in mock data

    # Test case 2: Strict mode with missing rate
    queries = [(Currency("GBP"), Currency("JPY"), Date(2023, 1, 1))]
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))

    # Test case 3: Empty queries
    assert list(service.queries([])) == []

    # Test case 4: Multiple queries with some missing
    queries = [
        (Currency("EUR"), Currency("USD"), Date(2023, 1, 1)),
        (Currency("GBP"), Currency("JPY"), Date(2023, 1, 1)),
        (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)),
    ]
    results = list(service.queries(queries))
    assert len(results) == 3
    assert results[0] == FXRate.of(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.10"))
    assert results[1] is None
    assert results[2] is None


# LLM-generated content at query #8
#--------------------------

```python
def test_FXRateService_query():
    # Create a mock FXRateService implementation for testing
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            # Mock data: EUR/USD rate is 1.2 on 2023-01-01
            if ccy1.code == "EUR" and ccy2.code == "USD" and asof == Date(2023, 1, 1):
                return FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.2"))
            # Mock data: USD/EUR rate is 0.8333 on 2023-01-01
            elif ccy1.code == "USD" and ccy2.code == "EUR" and asof == Date(2023, 1, 1):
                return FXRate(Currency("USD"), Currency("EUR"), Date(2023, 1, 1), Decimal("0.8333"))
            # Return None for other cases
            return None

        def queries(self, queries: Iterable[Tuple[Currency, Currency, Date]], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    # Initialize the mock service
    service = MockFXRateService()

    # Test case 1: Query existing rate (EUR/USD)
    rate = service.query(Currency("EUR"), Currency("USD"), Date(2023, 1, 1))
    assert rate is not None
    assert rate.ccy1.code == "EUR"
    assert rate.ccy2.code == "USD"
    assert rate.date == Date(2023, 1, 1)
    assert rate.value == Decimal("1.2")

    # Test case 2: Query existing rate (USD/EUR)
    rate = service.query(Currency("USD"), Currency("EUR"), Date(2023, 1, 1))
    assert rate is not None
    assert rate.ccy1.code == "USD"
    assert rate.ccy2.code == "EUR"
    assert rate.date == Date(2023, 1, 1)
    assert rate.value == Decimal("0.8333")

    # Test case 3: Query non-existing rate (should return None)
    rate = service.query(Currency("GBP"), Currency("JPY"), Date(2023, 1, 1))
    assert rate is None

    # Test case 4: Query with strict=True for non-existing rate (should raise FXRateLookupError)
    with pytest.raises(FXRateLookupError):
        service.query(Currency("GBP"), Currency("JPY"), Date(2023, 1, 1), strict=True)

    # Test case 5: Query with strict=True for existing rate (should return rate)
    rate = service.query(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), strict=True)
    assert rate is not None
    assert rate.ccy1.code == "EUR"
    assert rate.ccy2.code == "USD"
    assert rate.date == Date(2023, 1, 1)
    assert rate.value == Decimal("1.2")


# LLM-generated content at query #9
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
    from decimal import Decimal
    from pypara.currencies import Currencies

    EUR = Currencies["EUR"]
    USD = Currencies["USD"]
    GBP = Currencies["GBP"]

    test_rates = {
        (EUR, USD, date(2023, 1, 1)): FXRate(EUR, USD, date(2023, 1, 1), Decimal("1.1")),
        (EUR, GBP, date(2023, 1, 1)): FXRate(EUR, GBP, date(2023, 1, 1), Decimal("0.85")),
        (USD, GBP, date(2023, 1, 2)): FXRate(USD, GBP, date(2023, 1, 2), Decimal("0.77")),
    }

    service = TestFXRateService(test_rates)

    # Test 1: Successful queries
    queries = [
        (EUR, USD, date(2023, 1, 1)),
        (EUR, GBP, date(2023, 1, 1)),
        (USD, GBP, date(2023, 1, 2)),
    ]
    results = list(service.queries(queries))
    assert len(results) == 3
    assert results[0] == test_rates[(EUR, USD, date(2023, 1, 1))]
    assert results[1] == test_rates[(EUR, GBP, date(2023, 1, 1))]
    assert results[2] == test_rates[(USD, GBP, date(2023, 1, 2))]

    # Test 2: Query with missing rate (non-strict)
    queries = [
        (EUR, USD, date(2023, 1, 1)),
        (GBP, USD, date(2023, 1, 1)),  # Missing
    ]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == test_rates[(EUR, USD, date(2023, 1, 1))]
    assert results[1] is None

    # Test 3: Query with missing rate (strict)
    queries = [
        (GBP, USD, date(2023, 1, 1)),  # Missing
    ]
    try:
        list(service.queries(queries, strict=True))
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == GBP
        assert e.ccy2 == USD
        assert e.asof == date(2023, 1, 1)

    # Test 4: Empty queries
    assert list(service.queries([])) == []


# LLM-generated content at query #10
#--------------------------

```python
def test_FXRateService_queries():
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            if ccy1.code == "EUR" and ccy2.code == "USD" and asof == Date(2023, 1, 1):
                return FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1"))
            elif ccy1.code == "USD" and ccy2.code == "GBP" and asof == Date(2023, 1, 2):
                return FXRate(Currency("USD"), Currency("GBP"), Date(2023, 1, 2), Decimal("0.8"))
            elif strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None

        def queries(self, queries: Iterable[Tuple[Currency, Currency, Date]], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = MockFXRateService()

    # Test with valid queries
    queries = [
        (Currency("EUR"), Currency("USD"), Date(2023, 1, 1)),
        (Currency("USD"), Currency("GBP"), Date(2023, 1, 2)),
    ]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1"))
    assert results[1] == FXRate(Currency("USD"), Currency("GBP"), Date(2023, 1, 2), Decimal("0.8"))

    # Test with invalid query (non-strict)
    queries_with_invalid = [
        (Currency("EUR"), Currency("USD"), Date(2023, 1, 1)),
        (Currency("XYZ"), Currency("ABC"), Date(2023, 1, 3)),
    ]
    results = list(service.queries(queries_with_invalid))
    assert len(results) == 2
    assert results[0] == FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1"))
    assert results[1] is None

    # Test with invalid query (strict)
    queries_with_invalid_strict = [
        (Currency("EUR"), Currency("USD"), Date(2023, 1, 1)),
        (Currency("XYZ"), Currency("ABC"), Date(2023, 1, 3)),
    ]
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries_with_invalid_strict, strict=True))


# LLM-generated content at query #11
#--------------------------

```python
def test_FXRateService_queries():
    # Create a mock FXRateService implementation for testing
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            # Simple mock implementation that returns a fixed rate for EUR/USD
            if ccy1.code == "EUR" and ccy2.code == "USD" and asof == Date(2023, 1, 1):
                return FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1"))
            return None

        def queries(self, queries: Iterable[Tuple[Currency, Currency, Date]], strict: bool = False) -> Iterable[Optional[FXRate]]:
            # Implementation that uses the query method for each query
            for ccy1, ccy2, asof in queries:
                rate = self.query(ccy1, ccy2, asof, strict)
                if strict and rate is None:
                    raise FXRateLookupError(ccy1, ccy2, asof)
                yield rate

    # Initialize the mock service
    service = MockFXRateService()

    # Test case 1: Normal queries
    queries = [
        (Currency("EUR"), Currency("USD"), Date(2023, 1, 1)),
        (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)),
    ]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1"))
    assert results[1] is None

    # Test case 2: Strict queries with missing rate
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))

    # Test case 3: Empty queries
    assert list(service.queries([])) == []

    # Test case 4: Single query
    single_query = [(Currency("EUR"), Currency("USD"), Date(2023, 1, 1))]
    single_result = list(service.queries(single_query))
    assert len(single_result) == 1
    assert single_result[0] == FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1"))


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
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currencies

    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    today = date.today()
    test_rate = FXRate(eur, usd, today, Decimal("1.2"))

    # Initialize service with test data
    service = TestFXRateService({(eur, usd, today): test_rate})

    # Test successful query
    result = service.query(eur, usd, today)
    assert result == test_rate

    # Test query with non-existent rate (non-strict)
    result = service.query(usd, eur, today)
    assert result is None

    # Test query with non-existent rate (strict)
    with pytest.raises(FXRateLookupError):
        service.query(usd, eur, today, strict=True)

    # Test query with same currency
    same_currency_rate = FXRate(eur, eur, today, Decimal("1"))
    service_with_same = TestFXRateService({(eur, eur, today): same_currency_rate})
    result = service_with_same.query(eur, eur, today)
    assert result == same_currency_rate


# LLM-generated content at query #13
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
                key = (ccy1, ccy2, asof)
                rate = self.rates.get(key)
                if rate is None and strict:
                    raise FXRateLookupError(ccy1, ccy2, asof)
                results.append(rate)
            return results

    # Test setup
    from datetime import date
    from pypara.currencies import Currencies
    from decimal import Decimal

    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    today = date.today()
    rate = FXRate(eur, usd, today, Decimal("1.2"))

    # Test case 1: Successful query
    service = TestFXRateService({(eur, usd, today): rate})
    result = service.query(eur, usd, today)
    assert result == rate

    # Test case 2: Query with no result, non-strict mode
    service = TestFXRateService({})
    result = service.query(eur, usd, today)
    assert result is None

    # Test case 3: Query with no result, strict mode
    service = TestFXRateService({})
    with pytest.raises(FXRateLookupError):
        service.query(eur, usd, today, strict=True)

    # Test case 4: Query with inverted currencies
    inverted_rate = ~rate
    service = TestFXRateService({(usd, eur, today): inverted_rate})
    result = service.query(usd, eur, today)
    assert result == inverted_rate


# LLM-generated content at query #14
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
            for ccy1, ccy2, asof in queries:
                yield self.query(ccy1, ccy2, asof, strict)

    # Initialize the test service
    service = TestFXRateService()

    # Test successful query
    result = service.query(Currency("EUR"), Currency("USD"), Date(2023, 1, 1))
    assert result == FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1"))

    # Test inverted query
    result = service.query(Currency("USD"), Currency("EUR"), Date(2023, 1, 1))
    assert result == FXRate(Currency("USD"), Currency("EUR"), Date(2023, 1, 1), Decimal("0.909"))

    # Test non-existent rate without strict
    result = service.query(Currency("GBP"), Currency("USD"), Date(2023, 1, 1))
    assert result is None

    # Test non-existent rate with strict
    try:
        service.query(Currency("GBP"), Currency("USD"), Date(2023, 1, 1), strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == Currency("GBP")
        assert e.ccy2 == Currency("USD")
        assert e.asof == Date(2023, 1, 1)


# LLM-generated content at query #15
#--------------------------

```python
def test_FXRateService_queries():
    # Create a mock FXRateService implementation
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            if ccy1.code == "EUR" and ccy2.code == "USD" and asof == Date(2023, 1, 1):
                return FXRate(Currency("EUR"), Currency("USD"), asof, Decimal("1.1"))
            elif ccy1.code == "USD" and ccy2.code == "EUR" and asof == Date(2023, 1, 1):
                return FXRate(Currency("USD"), Currency("EUR"), asof, Decimal("0.9"))
            elif strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            else:
                return None

        def queries(self, queries: Iterable[Tuple[Currency, Currency, Date]], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return (self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries)

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
    assert results[1] == FXRate(Currency("USD"), Currency("EUR"), Date(2023, 1, 1), Decimal("0.9"))

    # Test case 2: Invalid queries with strict=False
    queries = [
        (Currency("EUR"), Currency("GBP"), Date(2023, 1, 1)),
        (Currency("JPY"), Currency("USD"), Date(2023, 1, 1)),
    ]
    results = list(service.queries(queries, strict=False))
    assert len(results) == 2
    assert results[0] is None
    assert results[1] is None

    # Test case 3: Invalid queries with strict=True
    queries = [
        (Currency("EUR"), Currency("GBP"), Date(2023, 1, 1)),
    ]
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))


# LLM-generated content at query #16
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

    currency1 = Currency("EUR")
    currency2 = Currency("USD")
    date = Date(2023, 1, 1)
    rate = FXRate(currency1, currency2, date, Decimal("1.10"))
    service = MockFXRateService({(currency1, currency2, date): rate})

    # Test successful query
    result = service.query(currency1, currency2, date)
    assert result == rate

    # Test query with strict=True and missing rate
    with pytest.raises(FXRateLookupError):
        service.query(currency1, currency2, Date(2023, 1, 2), strict=True)

    # Test query with strict=False and missing rate
    result = service.query(currency1, currency2, Date(2023, 1, 2), strict=False)
    assert result is None

    # Test inverted currency pair
    inverted_rate = ~rate
    service_with_inverted = MockFXRateService({(currency2, currency1, date): inverted_rate})
    result = service_with_inverted.query(currency2, currency1, date)
    assert result == inverted_rate


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
            rate = self.rates.get(key)
            if rate is None and strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return rate

        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    # Setup test data
    from pypara.currencies import Currencies
    import datetime
    from decimal import Decimal

    EUR = Currencies["EUR"]
    USD = Currencies["USD"]
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)

    rates = {
        (EUR, USD, today): FXRate(EUR, USD, today, Decimal("1.20")),
        (USD, EUR, today): FXRate(USD, EUR, today, Decimal("0.83")),
        (EUR, USD, yesterday): FXRate(EUR, USD, yesterday, Decimal("1.18")),
    }

    service = MockFXRateService(rates)

    # Test successful query
    result = service.query(EUR, USD, today)
    assert result == FXRate(EUR, USD, today, Decimal("1.20"))

    # Test inverted rate query
    result = service.query(USD, EUR, today)
    assert result == FXRate(USD, EUR, today, Decimal("0.83"))

    # Test query with different date
    result = service.query(EUR, USD, yesterday)
    assert result == FXRate(EUR, USD, yesterday, Decimal("1.18"))

    # Test query with non-existent rate (non-strict)
    result = service.query(EUR, USD, datetime.date(2020, 1, 1))
    assert result is None

    # Test query with non-existent rate (strict)
    with pytest.raises(FXRateLookupError):
        service.query(EUR, USD, datetime.date(2020, 1, 1), strict=True)


# LLM-generated content at query #18
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
    eur = Currency("EUR", "Euro", 2)
    usd = Currency("USD", "US Dollar", 2)
    date1 = Date(2023, 1, 1)
    date2 = Date(2023, 1, 2)

    rate1 = FXRate(eur, usd, date1, Decimal("1.1"))
    rate2 = FXRate(eur, usd, date2, Decimal("1.2"))
    rate3 = FXRate(usd, eur, date1, Decimal("0.9"))

    # Initialize the test service
    service = TestFXRateService([rate1, rate2, rate3])

    # Test case 1: Normal query
    queries = [(eur, usd, date1), (eur, usd, date2)]
    results = service.queries(queries)
    assert len(results) == 2
    assert results[0] == rate1
    assert results[1] == rate2

    # Test case 2: Query with non-existent rate
    queries = [(eur, usd, date1), (eur, usd, Date(2023, 1, 3))]
    results = service.queries(queries)
    assert len(results) == 2
    assert results[0] == rate1
    assert results[1] is None

    # Test case 3: Strict query with non-existent rate
    queries = [(eur, usd, date1), (eur, usd, Date(2023, 1, 3))]
    with pytest.raises(FXRateLookupError):
        service.queries(queries, strict=True)

    # Test case 4: Empty queries
    results = service.queries([])
    assert len(results) == 0

    # Test case 5: Inverted rate query
    queries = [(usd, eur, date1)]
    results = service.queries(queries)
    assert len(results) == 1
    assert results[0] == rate3


# LLM-generated content at query #19
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

    # Test data setup
    from pypara.currencies import Currencies
    import datetime
    from decimal import Decimal

    EUR = Currencies["EUR"]
    USD = Currencies["USD"]
    GBP = Currencies["GBP"]
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)

    rates = {
        (EUR, USD, today): FXRate(EUR, USD, today, Decimal("1.2")),
        (USD, GBP, yesterday): FXRate(USD, GBP, yesterday, Decimal("0.8")),
    }

    service = MockFXRateService(rates)

    # Test 1: Basic queries
    queries = [
        (EUR, USD, today),
        (USD, GBP, yesterday),
        (EUR, GBP, today)
    ]
    results = list(service.queries(queries))
    assert len(results) == 3
    assert results[0] == FXRate(EUR, USD, today, Decimal("1.2"))
    assert results[1] == FXRate(USD, GBP, yesterday, Decimal("0.8"))
    assert results[2] is None

    # Test 2: Strict mode with missing rate
    queries_strict = [
        (EUR, USD, today),
        (EUR, GBP, today)
    ]
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries_strict, strict=True))

    # Test 3: Empty queries
    assert list(service.queries([])) == []

    # Test 4: All missing rates
    queries_missing = [
        (EUR, GBP, today),
        (GBP, USD, today)
    ]
    results_missing = list(service.queries(queries_missing))
    assert all(r is None for r in results_missing)

    # Test 5: All found rates
    queries_found = [
        (EUR, USD, today),
        (USD, GBP, yesterday)
    ]
    results_found = list(service.queries(queries_found))
    assert all(r is not None for r in results_found)
    assert results_found[0] == FXRate(EUR, USD, today, Decimal("1.2"))
    assert results_found[1] == FXRate(USD, GBP, yesterday, Decimal("0.8"))


# LLM-generated content at query #20
#--------------------------

```python
def test_FXRateService_queries():
    # Create a mock FXRateService implementation
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
    from datetime import date
    from pypara.currencies import Currencies
    from decimal import Decimal

    EUR = Currencies["EUR"]
    USD = Currencies["USD"]
    GBP = Currencies["GBP"]

    rates = {
        (EUR, USD, date(2023, 1, 1)): FXRate(EUR, USD, date(2023, 1, 1), Decimal("1.1")),
        (USD, GBP, date(2023, 1, 1)): FXRate(USD, GBP, date(2023, 1, 1), Decimal("0.8")),
        (EUR, GBP, date(2023, 1, 2)): FXRate(EUR, GBP, date(2023, 1, 2), Decimal("0.85")),
    }

    service = MockFXRateService(rates)

    # Test successful queries
    queries = [
        (EUR, USD, date(2023, 1, 1)),
        (USD, GBP, date(2023, 1, 1)),
        (EUR, GBP, date(2023, 1, 2)),
    ]
    results = list(service.queries(queries))
    assert len(results) == 3
    assert results[0] == rates[(EUR, USD, date(2023, 1, 1))]
    assert results[1] == rates[(USD, GBP, date(2023, 1, 1))]
    assert results[2] == rates[(EUR, GBP, date(2023, 1, 2))]

    # Test with missing rate (non-strict)
    queries_with_missing = [
        (EUR, USD, date(2023, 1, 1)),
        (EUR, USD, date(2023, 1, 3)),  # Missing
    ]
    results = list(service.queries(queries_with_missing))
    assert len(results) == 2
    assert results[0] == rates[(EUR, USD, date(2023, 1, 1))]
    assert results[1] is None

    # Test with missing rate (strict)
    try:
        list(service.queries(queries_with_missing, strict=True))
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == EUR
        assert e.ccy2 == USD
        assert e.asof == date(2023, 1, 3)

    # Test empty queries
    assert list(service.queries([])) == []


# LLM-generated content at query #21
#--------------------------

```python
def test_FXRateService_queries():
    # Create a mock FXRateService subclass for testing
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

    # Test data
    eur = Currency("EUR", "Euro", 2)
    usd = Currency("USD", "US Dollar", 2)
    today = Date.today()
    yesterday = today - datetime.timedelta(days=1)

    rate1 = FXRate(eur, usd, today, Decimal("1.2"))
    rate2 = FXRate(usd, eur, yesterday, Decimal("0.8"))

    # Initialize service with test rates
    service = MockFXRateService([rate1, rate2])

    # Test normal queries
    queries = [(eur, usd, today), (usd, eur, yesterday)]
    results = list(service.queries(queries))
    assert results[0] == rate1
    assert results[1] == rate2

    # Test query with missing rate (non-strict)
    queries = [(eur, usd, yesterday)]
    results = list(service.queries(queries))
    assert results[0] is None

    # Test query with missing rate (strict)
    queries = [(eur, usd, yesterday)]
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))

    # Test empty queries
    assert list(service.queries([])) == []

    # Test inverted rates
    queries = [(usd, eur, today)]
    results = list(service.queries(queries))
    assert results[0] is None  # Should not find the inverted rate unless explicitly added


# LLM-generated content at query #22
#--------------------------

```python
def test_FXRateService_query():
    # Setup
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            if ccy1.code == "EUR" and ccy2.code == "USD" and asof == Date(2023, 1, 1):
                return FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1"))
            return None

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(*q, strict=strict) for q in queries]

    service = MockFXRateService()
    eur = Currency("EUR")
    usd = Currency("USD")
    test_date = Date(2023, 1, 1)

    # Test successful query
    result = service.query(eur, usd, test_date)
    assert result is not None
    assert result.ccy1 == eur
    assert result.ccy2 == usd
    assert result.date == test_date
    assert result.value == Decimal("1.1")

    # Test non-strict query with missing rate
    result = service.query(usd, eur, test_date)
    assert result is None

    # Test strict query with missing rate
    try:
        service.query(usd, eur, test_date, strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == usd
        assert e.ccy2 == eur
        assert e.asof == test_date


# LLM-generated content at query #23
#--------------------------

```python
def test_FXRateService_queries():
    # Create a mock FXRateService for testing
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            # Mock implementation for testing
            if ccy1.code == "EUR" and ccy2.code == "USD" and asof == Date(2023, 1, 1):
                return FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1"))
            elif ccy1.code == "USD" and ccy2.code == "EUR" and asof == Date(2023, 1, 1):
                return FXRate(Currency("USD"), Currency("EUR"), Date(2023, 1, 1), Decimal("0.9"))
            elif strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None

        def queries(self, queries: Iterable[Tuple[Currency, Currency, Date]], strict: bool = False) -> Iterable[Optional[FXRate]]:
            # Mock implementation for testing
            for ccy1, ccy2, asof in queries:
                yield self.query(ccy1, ccy2, asof, strict)

    # Initialize the mock service
    service = MockFXRateService()

    # Test case 1: Successful queries
    queries = [
        (Currency("EUR"), Currency("USD"), Date(2023, 1, 1)),
        (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)),
    ]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1"))
    assert results[1] == FXRate(Currency("USD"), Currency("EUR"), Date(2023, 1, 1), Decimal("0.9"))

    # Test case 2: Query with missing rate (non-strict)
    queries = [
        (Currency("EUR"), Currency("GBP"), Date(2023, 1, 1)),
    ]
    results = list(service.queries(queries))
    assert len(results) == 1
    assert results[0] is None

    # Test case 3: Query with missing rate (strict)
    queries = [
        (Currency("EUR"), Currency("GBP"), Date(2023, 1, 1)),
    ]
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))


# LLM-generated content at query #24
#--------------------------

```python
def test_FXRateService_queries():
    # Create a mock FXRateService subclass for testing
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

    # Test data
    eur = Currency("EUR", "Euro", 2)
    usd = Currency("USD", "US Dollar", 2)
    today = Date.today()
    yesterday = today - Date.resolution

    rate1 = FXRate(eur, usd, today, Decimal("1.20"))
    rate2 = FXRate(usd, eur, yesterday, Decimal("0.80"))

    # Initialize service with test rates
    service = MockFXRateService([rate1, rate2])

    # Test queries with existing rates
    queries = [(eur, usd, today), (usd, eur, yesterday)]
    results = service.queries(queries)
    assert len(results) == 2
    assert results[0] == rate1
    assert results[1] == rate2

    # Test queries with non-existing rate (non-strict)
    queries = [(eur, usd, yesterday)]
    results = service.queries(queries)
    assert len(results) == 1
    assert results[0] is None

    # Test queries with non-existing rate (strict)
    queries = [(eur, usd, yesterday)]
    with pytest.raises(FXRateLookupError):
        service.queries(queries, strict=True)

    # Test empty queries
    assert list(service.queries([])) == []


# LLM-generated content at query #25
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
    test_rates = {
        (Currencies["EUR"], Currencies["USD"], date(2023, 1, 1)): FXRate(Currencies["EUR"], Currencies["USD"], date(2023, 1, 1), Decimal("1.1")),
        (Currencies["USD"], Currencies["EUR"], date(2023, 1, 1)): FXRate(Currencies["USD"], Currencies["EUR"], date(2023, 1, 1), Decimal("0.909")),
    }
    service = TestFXRateService(test_rates)

    # Test successful query
    result = service.query(Currencies["EUR"], Currencies["USD"], date(2023, 1, 1))
    assert result == FXRate(Currencies["EUR"], Currencies["USD"], date(2023, 1, 1), Decimal("1.1"))

    # Test query with None result (non-strict)
    result = service.query(Currencies["GBP"], Currencies["USD"], date(2023, 1, 1))
    assert result is None

    # Test query with lookup error (strict)
    with pytest.raises(FXRateLookupError):
        service.query(Currencies["GBP"], Currencies["USD"], date(2023, 1, 1), strict=True)

    # Test inverted rate query
    result = service.query(Currencies["USD"], Currencies["EUR"], date(2023, 1, 1))
    assert result == FXRate(Currencies["USD"], Currencies["EUR"], date(2023, 1, 1), Decimal("0.909"))


# LLM-generated content at query #26
#--------------------------

```python
def test_FXRateService_queries():
    # Create a mock FXRateService instance
    class MockFXRateService(FXRateService):
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

    # Create an instance of the mock service
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

    # Test with invalid queries (non-strict mode)
    queries = [
        (Currency("EUR"), Currency("USD"), Date(2023, 1, 1)),
        (Currency("GBP"), Currency("JPY"), Date(2023, 1, 1)),
    ]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1"))
    assert results[1] is None

    # Test with invalid queries (strict mode)
    queries = [
        (Currency("EUR"), Currency("USD"), Date(2023, 1, 1)),
        (Currency("GBP"), Currency("JPY"), Date(2023, 1, 1)),
    ]
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))


# LLM-generated content at query #27
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
    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    today = date.today()

    # Test successful query
    service = TestFXRateService({
        (eur, usd, today): FXRate(eur, usd, today, Decimal("1.2"))
    })
    result = service.query(eur, usd, today)
    assert result == FXRate(eur, usd, today, Decimal("1.2"))

    # Test query with None result
    result = service.query(usd, eur, today)
    assert result is None

    # Test strict query with missing rate
    with pytest.raises(FXRateLookupError):
        service.query(usd, eur, today, strict=True)

    # Test query with different date
    yesterday = date.today() - timedelta(days=1)
    result = service.query(eur, usd, yesterday)
    assert result is None

    # Test query with inverted currency pair
    service = TestFXRateService({
        (usd, eur, today): FXRate(usd, eur, today, Decimal("0.8"))
    })
    result = service.query(eur, usd, today)
    assert result is None  # Should not automatically invert


# LLM-generated content at query #28
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

    # Prepare test data
    from decimal import Decimal
    from pypara.currencies import Currencies
    import datetime

    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)

    rates = {
        (eur, usd, today): FXRate(eur, usd, today, Decimal("1.2")),
        (usd, eur, yesterday): FXRate(usd, eur, yesterday, Decimal("0.8")),
    }

    service = MockFXRateService(rates)

    # Test case 1: All rates found
    queries1 = [(eur, usd, today), (usd, eur, yesterday)]
    results1 = list(service.queries(queries1))
    assert len(results1) == 2
    assert results1[0] == rates[(eur, usd, today)]
    assert results1[1] == rates[(usd, eur, yesterday)]

    # Test case 2: Some rates not found, strict=False
    queries2 = [(eur, usd, today), (eur, usd, yesterday)]
    results2 = list(service.queries(queries2))
    assert len(results2) == 2
    assert results2[0] == rates[(eur, usd, today)]
    assert results2[1] is None

    # Test case 3: Some rates not found, strict=True
    queries3 = [(eur, usd, today), (eur, usd, yesterday)]
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries3, strict=True))

    # Test case 4: Empty queries
    queries4 = []
    results4 = list(service.queries(queries4))
    assert len(results4) == 0


# LLM-generated content at query #29
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
            rate = self.rates.get((ccy1, ccy2, asof))
            if rate is None and strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return rate

        def queries(self, queries: Iterable[Tuple[Currency, Currency, Date]], strict: bool = False) -> Iterable[Optional[FXRate]]:
            for ccy1, ccy2, asof in queries:
                yield self.query(ccy1, ccy2, asof, strict)

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


# LLM-generated content at query #30
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
    eur = Currency("EUR", "Euro", 2)
    usd = Currency("USD", "US Dollar", 2)
    today = Date.today()
    rate = FXRate(eur, usd, today, Decimal("1.2"))

    # Test successful query
    service = MockFXRateService({(eur, usd, today): rate})
    result = service.query(eur, usd, today)
    assert result == rate

    # Test query with strict=True raises exception
    with pytest.raises(FXRateLookupError):
        service.query(eur, usd, today, strict=True)

    # Test query with non-existent rate returns None
    result = service.query(usd, eur, today)
    assert result is None

    # Test query with strict=True for non-existent rate raises exception
    with pytest.raises(FXRateLookupError):
        service.query(usd, eur, today, strict=True)


# LLM-generated content at query #31
#--------------------------

```python
def test_FXRateService_queries():
    # Create a mock FXRateService instance
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            if ccy1.code == "EUR" and ccy2.code == "USD" and asof == Date(2023, 1, 1):
                return FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1"))
            elif ccy1.code == "USD" and ccy2.code == "GBP" and asof == Date(2023, 1, 2):
                return FXRate(Currency("USD"), Currency("GBP"), Date(2023, 1, 2), Decimal("0.8"))
            elif strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            else:
                return None

        def queries(self, queries: Iterable[Tuple[Currency, Currency, Date]], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return (self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries)

    service = MockFXRateService()

    # Test with valid queries
    queries = [
        (Currency("EUR"), Currency("USD"), Date(2023, 1, 1)),
        (Currency("USD"), Currency("GBP"), Date(2023, 1, 2)),
        (Currency("EUR"), Currency("GBP"), Date(2023, 1, 3)),
    ]
    results = list(service.queries(queries))
    assert len(results) == 3
    assert results[0] == FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1"))
    assert results[1] == FXRate(Currency("USD"), Currency("GBP"), Date(2023, 1, 2), Decimal("0.8"))
    assert results[2] is None

    # Test with strict mode
    queries = [
        (Currency("EUR"), Currency("USD"), Date(2023, 1, 1)),
        (Currency("EUR"), Currency("GBP"), Date(2023, 1, 3)),
    ]
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))

    # Test with empty queries
    assert list(service.queries([])) == []


# LLM-generated content at query #32
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
            for ccy1, ccy2, asof in queries:
                yield self.query(ccy1, ccy2, asof, strict)

    # Setup test data
    from pypara.currencies import Currencies
    import datetime
    from decimal import Decimal

    EUR = Currencies["EUR"]
    USD = Currencies["USD"]
    GBP = Currencies["GBP"]
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)

    rates = {
        (EUR, USD, today): FXRate(EUR, USD, today, Decimal("1.2")),
        (USD, GBP, yesterday): FXRate(USD, GBP, yesterday, Decimal("0.8")),
    }

    service = MockFXRateService(rates)

    # Test 1: Basic query with existing rates
    queries = [(EUR, USD, today), (USD, GBP, yesterday)]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == rates[(EUR, USD, today)]
    assert results[1] == rates[(USD, GBP, yesterday)]

    # Test 2: Query with non-existing rate (non-strict mode)
    queries = [(EUR, USD, today), (EUR, GBP, today)]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == rates[(EUR, USD, today)]
    assert results[1] is None

    # Test 3: Query with non-existing rate (strict mode)
    queries = [(EUR, USD, today), (EUR, GBP, today)]
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))

    # Test 4: Empty query
    results = list(service.queries([]))
    assert len(results) == 0


# LLM-generated content at query #33
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
    from decimal import Decimal

    EUR = Currencies["EUR"]
    USD = Currencies["USD"]
    today = date.today()
    rate = FXRate(EUR, USD, today, Decimal("1.2"))

    # Test successful query
    service = TestFXRateService({(EUR, USD, today): rate})
    result = service.query(EUR, USD, today)
    assert result == rate

    # Test query with non-existent rate (non-strict)
    result = service.query(USD, EUR, today)
    assert result is None

    # Test query with non-existent rate (strict)
    try:
        service.query(USD, EUR, today, strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == USD
        assert e.ccy2 == EUR
        assert e.asof == today


# LLM-generated content at query #34
#--------------------------

```python
def test_FXRateService_query():
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
            for query in queries:
                yield self.query(*query, strict=strict)

    # Setup test data
    from datetime import date
    from pypara.currencies import Currencies
    test_rates = {
        (Currencies["EUR"], Currencies["USD"], date(2023, 1, 1)): FXRate(
            Currencies["EUR"], Currencies["USD"], date(2023, 1, 1), Decimal("1.1")
        ),
        (Currencies["USD"], Currencies["JPY"], date(2023, 1, 1)): FXRate(
            Currencies["USD"], Currencies["JPY"], date(2023, 1, 1), Decimal("130.5")
        ),
    }
    service = TestFXRateService(test_rates)

    # Test successful query
    result = service.query(Currencies["EUR"], Currencies["USD"], date(2023, 1, 1))
    assert result == test_rates[(Currencies["EUR"], Currencies["USD"], date(2023, 1, 1))]

    # Test non-strict query with missing rate
    result = service.query(Currencies["GBP"], Currencies["USD"], date(2023, 1, 1))
    assert result is None

    # Test strict query with missing rate
    with pytest.raises(FXRateLookupError):
        service.query(Currencies["GBP"], Currencies["USD"], date(2023, 1, 1), strict=True)


# LLM-generated content at query #35
#--------------------------

```python
def test_FXRateService_queries():
    # Create a mock FXRateService implementation
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
            return (self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries)

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
    assert results[1] == FXRate(Currency("USD"), Currency("EUR"), Date(2023, 1, 1), Decimal("0.9"))

    # Test case 2: Query with no result
    queries = [
        (Currency("GBP"), Currency("JPY"), Date(2023, 1, 1)),
    ]
    results = list(service.queries(queries))
    assert len(results) == 1
    assert results[0] is None

    # Test case 3: Strict mode with missing rate
    queries = [
        (Currency("GBP"), Currency("JPY"), Date(2023, 1, 1)),
    ]
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))

    # Test case 4: Empty queries
    results = list(service.queries([]))
    assert len(results) == 0


# LLM-generated content at query #36
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

    service = TestFXRateService(rates)

    # Test successful queries
    queries = [
        (eur, usd, today),
        (usd, eur, today),
    ]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == rates[(eur, usd, today)]
    assert results[1] == rates[(usd, eur, today)]

    # Test with missing rate (non-strict)
    queries_with_missing = [
        (eur, usd, today),
        (eur, usd, date(2020, 1, 1)),  # Missing
    ]
    results = list(service.queries(queries_with_missing))
    assert len(results) == 2
    assert results[0] == rates[(eur, usd, today)]
    assert results[1] is None

    # Test with missing rate (strict)
    try:
        list(service.queries(queries_with_missing, strict=True))
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == eur
        assert e.ccy2 == usd
        assert e.asof == date(2020, 1, 1)

    # Test empty queries
    assert list(service.queries([])) == []


# LLM-generated content at query #37
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
    eur = Currency("EUR", "Euro", "€")
    usd = Currency("USD", "US Dollar", "$")
    today = Date(2023, 1, 1)
    yesterday = Date(2022, 12, 31)

    rate1 = FXRate(eur, usd, today, Decimal("1.10"))
    rate2 = FXRate(usd, eur, today, Decimal("0.90"))
    rate3 = FXRate(eur, usd, yesterday, Decimal("1.05"))

    rates = {
        (eur, usd, today): rate1,
        (usd, eur, today): rate2,
        (eur, usd, yesterday): rate3,
    }

    service = MockFXRateService(rates)

    # Test basic queries
    queries = [(eur, usd, today), (usd, eur, today)]
    results = list(service.queries(queries))
    assert results == [rate1, rate2]

    # Test with missing rate (non-strict)
    queries = [(eur, usd, today), (eur, usd, Date(2020, 1, 1))]
    results = list(service.queries(queries))
    assert results == [rate1, None]

    # Test with missing rate (strict)
    queries = [(eur, usd, today), (eur, usd, Date(2020, 1, 1))]
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))

    # Test empty queries
    assert list(service.queries([])) == []

    # Test with inverted rates
    queries = [(eur, usd, today), (usd, eur, today)]
    results = list(service.queries(queries))
    assert results[0] == rate1
    assert results[1] == rate2
    assert results[1] == ~results[0]


# LLM-generated content at query #38
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

    # Test successful query
    service = TestFXRateService({
        (eur, usd, today): FXRate(eur, usd, today, Decimal("1.2"))
    })
    result = service.query(eur, usd, today)
    assert result == FXRate(eur, usd, today, Decimal("1.2"))

    # Test query with None result (non-strict)
    result = service.query(usd, eur, today)
    assert result is None

    # Test query with strict=True raising exception
    with pytest.raises(FXRateLookupError):
        service.query(usd, eur, today, strict=True)

    # Test query with different date
    yesterday = date.today() - timedelta(days=1)
    result = service.query(eur, usd, yesterday)
    assert result is None

    # Test query with inverted currencies
    service = TestFXRateService({
        (eur, usd, today): FXRate(eur, usd, today, Decimal("1.2")),
        (usd, eur, today): FXRate(usd, eur, today, Decimal("0.8333"))
    })
    result = service.query(usd, eur, today)
    assert result == FXRate(usd, eur, today, Decimal("0.8333"))


# LLM-generated content at query #39
#--------------------------

```python
def test_FXRateService_queries():
    # Create a mock FXRateService implementation for testing
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            # Mock implementation that returns a fixed rate for EUR/USD
            if ccy1.code == "EUR" and ccy2.code == "USD" and asof == Date(2023, 1, 1):
                return FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1"))
            return None

        def queries(self, queries: Iterable[Tuple[Currency, Currency, Date]], strict: bool = False) -> Iterable[Optional[FXRate]]:
            # Mock implementation that processes queries
            results = []
            for ccy1, ccy2, asof in queries:
                rate = self.query(ccy1, ccy2, asof, strict)
                if rate is None and strict:
                    raise FXRateLookupError(ccy1, ccy2, asof)
                results.append(rate)
            return results

    # Initialize the mock service
    service = MockFXRateService()

    # Test case 1: Single query
    queries = [(Currency("EUR"), Currency("USD"), Date(2023, 1, 1))]
    results = list(service.queries(queries))
    assert len(results) == 1
    assert results[0] == FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1"))

    # Test case 2: Multiple queries with some missing rates
    queries = [
        (Currency("EUR"), Currency("USD"), Date(2023, 1, 1)),
        (Currency("GBP"), Currency("USD"), Date(2023, 1, 1)),
    ]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1"))
    assert results[1] is None

    # Test case 3: Strict mode with missing rate should raise exception
    with pytest.raises(FXRateLookupError):
        queries = [(Currency("GBP"), Currency("USD"), Date(2023, 1, 1))]
        list(service.queries(queries, strict=True))


# LLM-generated content at query #40
#--------------------------

```python
def test_FXRateService_query():
    # Create a mock FXRateService implementation for testing
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

    # Create test data
    from datetime import date
    from decimal import Decimal
    from pypara.currencies import Currencies

    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    today = date.today()
    yesterday = today - timedelta(days=1)

    rates = {
        (eur, usd, today): FXRate(eur, usd, today, Decimal("1.2")),
        (usd, eur, today): FXRate(usd, eur, today, Decimal("0.8333")),
        (eur, usd, yesterday): FXRate(eur, usd, yesterday, Decimal("1.1")),
    }

    service = MockFXRateService(rates)

    # Test successful query
    result = service.query(eur, usd, today)
    assert result == FXRate(eur, usd, today, Decimal("1.2"))

    # Test query with None result (non-strict)
    result = service.query(eur, usd, date(2020, 1, 1))
    assert result is None

    # Test query with strict=True raising exception
    with pytest.raises(FXRateLookupError):
        service.query(eur, usd, date(2020, 1, 1), strict=True)

    # Test inverted rate
    result = service.query(usd, eur, today)
    assert result == FXRate(usd, eur, today, Decimal("0.8333"))

    # Test same currency
    with pytest.raises(ValueError):
        FXRate.of(eur, eur, today, Decimal("2.0"))


# LLM-generated content at query #41
#--------------------------

```python
def test_FXRateService_query():
    # Create a mock FXRateService implementation for testing
    class MockFXRateService(FXRateService):
        def __init__(self):
            self.rates = {
                (Currency("EUR"), Currency("USD"), Date(2023, 1, 1)): FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.10")),
                (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)): FXRate(Currency("USD"), Currency("EUR"), Date(2023, 1, 1), Decimal("0.9091")),
            }

        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            return self.rates.get((ccy1, ccy2, asof))

        def queries(self, queries: Iterable[Tuple[Currency, Currency, Date]], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return (self.query(ccy1, ccy2, asof) for ccy1, ccy2, asof in queries)

    # Initialize the mock service
    service = MockFXRateService()

    # Test successful query
    rate = service.query(Currency("EUR"), Currency("USD"), Date(2023, 1, 1))
    assert rate is not None
    assert rate.ccy1 == Currency("EUR")
    assert rate.ccy2 == Currency("USD")
    assert rate.date == Date(2023, 1, 1)
    assert rate.value == Decimal("1.10")

    # Test inverted rate
    rate = service.query(Currency("USD"), Currency("EUR"), Date(2023, 1, 1))
    assert rate is not None
    assert rate.ccy1 == Currency("USD")
    assert rate.ccy2 == Currency("EUR")
    assert rate.date == Date(2023, 1, 1)
    assert rate.value == Decimal("0.9091")

    # Test non-existent rate with strict=False
    rate = service.query(Currency("EUR"), Currency("GBP"), Date(2023, 1, 1), strict=False)
    assert rate is None

    # Test non-existent rate with strict=True
    try:
        service.query(Currency("EUR"), Currency("GBP"), Date(2023, 1, 1), strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == Currency("EUR")
        assert e.ccy2 == Currency("GBP")
        assert e.asof == Date(2023, 1, 1)


# LLM-generated content at query #42
#--------------------------

```python
def test_FXRateService_query():
    # Mock FXRateService implementation for testing
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            # Mock data
            mock_rate = FXRate(ccy1, ccy2, asof, Decimal("1.2"))
            if ccy1.code == "EUR" and ccy2.code == "USD" and asof == Date(2023, 1, 1):
                return mock_rate
            elif strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None

        def queries(self, queries: Iterable[Tuple[Currency, Currency, Date]], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    # Test setup
    service = MockFXRateService()
    eur = Currency("EUR", "Euro", 2)
    usd = Currency("USD", "US Dollar", 2)
    test_date = Date(2023, 1, 1)

    # Test successful query
    result = service.query(eur, usd, test_date)
    assert result is not None
    assert result.ccy1 == eur
    assert result.ccy2 == usd
    assert result.date == test_date
    assert result.value == Decimal("1.2")

    # Test query with no result (non-strict)
    result = service.query(usd, eur, test_date)
    assert result is None

    # Test query with no result (strict)
    with pytest.raises(FXRateLookupError):
        service.query(usd, eur, test_date, strict=True)


# LLM-generated content at query #43
#--------------------------

```python
def test_FXRateService_queries():
    # Create a mock FXRateService implementation for testing
    class MockFXRateService(FXRateService):
        def __init__(self):
            self.rates = {
                (Currency("EUR"), Currency("USD"), Date(2023, 1, 1)): FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1")),
                (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)): FXRate(Currency("USD"), Currency("EUR"), Date(2023, 1, 1), Decimal("0.9")),
                (Currency("EUR"), Currency("GBP"), Date(2023, 1, 1)): FXRate(Currency("EUR"), Currency("GBP"), Date(2023, 1, 1), Decimal("0.8")),
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

    # Test case 1: Query existing rates
    queries = [
        (Currency("EUR"), Currency("USD"), Date(2023, 1, 1)),
        (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)),
        (Currency("EUR"), Currency("GBP"), Date(2023, 1, 1)),
    ]
    results = list(service.queries(queries))
    assert len(results) == 3
    assert results[0] == FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1"))
    assert results[1] == FXRate(Currency("USD"), Currency("EUR"), Date(2023, 1, 1), Decimal("0.9"))
    assert results[2] == FXRate(Currency("EUR"), Currency("GBP"), Date(2023, 1, 1), Decimal("0.8"))

    # Test case 2: Query non-existing rates without strict mode
    queries = [
        (Currency("EUR"), Currency("JPY"), Date(2023, 1, 1)),
        (Currency("USD"), Currency("JPY"), Date(2023, 1, 1)),
    ]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] is None
    assert results[1] is None

    # Test case 3: Query non-existing rates with strict mode
    queries = [
        (Currency("EUR"), Currency("JPY"), Date(2023, 1, 1)),
    ]
    try:
        list(service.queries(queries, strict=True))
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == Currency("EUR")
        assert e.ccy2 == Currency("JPY")
        assert e.asof == Date(2023, 1, 1)

    # Test case 4: Empty queries
    results = list(service.queries([]))
    assert len(results) == 0


# LLM-generated content at query #44
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
                rate = self.query(ccy1, ccy2, asof, strict)
                results.append(rate)
            return results

    # Create test data
    eur = Currency("EUR", "Euro", "€")
    usd = Currency("USD", "US Dollar", "$")
    today = Date.today()
    yesterday = today - datetime.timedelta(days=1)

    rate1 = FXRate(eur, usd, today, Decimal("1.2"))
    rate2 = FXRate(usd, eur, yesterday, Decimal("0.8"))

    # Initialize the mock service
    service = MockFXRateService([rate1, rate2])

    # Test case 1: Normal queries
    queries = [
        (eur, usd, today),
        (usd, eur, yesterday),
    ]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == rate1
    assert results[1] == rate2

    # Test case 2: Query with missing rate (non-strict)
    queries = [
        (eur, usd, today),
        (eur, usd, yesterday),  # Missing
    ]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == rate1
    assert results[1] is None

    # Test case 3: Query with missing rate (strict)
    queries = [
        (eur, usd, today),
        (eur, usd, yesterday),  # Missing
    ]
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))

    # Test case 4: Empty queries
    results = list(service.queries([]))
    assert len(results) == 0


# LLM-generated content at query #45
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
    today = Date.today()
    yesterday = today - datetime.timedelta(days=1)

    rate1 = FXRate(eur, usd, today, Decimal("1.2"))
    rate2 = FXRate(usd, eur, yesterday, Decimal("0.8"))

    # Initialize service with test rates
    service = MockFXRateService([rate1, rate2])

    # Test basic query
    queries = [(eur, usd, today), (usd, eur, yesterday)]
    results = list(service.queries(queries))
    assert results == [rate1, rate2]

    # Test with non-existent rate (non-strict)
    queries = [(eur, usd, yesterday)]
    results = list(service.queries(queries))
    assert results == [None]

    # Test with non-existent rate (strict)
    queries = [(eur, usd, yesterday)]
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))

    # Test empty queries
    assert list(service.queries([])) == []


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# LLM-generated content at query #1
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
    assert inverted2.value == Decimal("0.00664451")  # 1/150.5 with proper precision

    # Test inversion with same currency (should be 1)
    rate3 = FXRate(Currency("EUR"), Currency("EUR"), Date(2023, 1, 1), Decimal("1"))
    inverted3 = ~rate3
    assert inverted3 == rate3


# LLM-generated content at query #2
#--------------------------

```python
def test_FXRate___invert__():
    # Test inversion of FXRate
    rate = FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("2"))
    inverted_rate = ~rate
    assert inverted_rate.ccy1 == Currency("USD")
    assert inverted_rate.ccy2 == Currency("EUR")
    assert inverted_rate.date == Date(2023, 1, 1)
    assert inverted_rate.value == Decimal("0.5")

    # Test inversion of inverted FXRate
    double_inverted_rate = ~inverted_rate
    assert double_inverted_rate == rate

    # Test inversion with different values
    rate2 = FXRate(Currency("GBP"), Currency("JPY"), Date(2023, 1, 1), Decimal("150.5"))
    inverted_rate2 = ~rate2
    assert inverted_rate2.ccy1 == Currency("JPY")
    assert inverted_rate2.ccy2 == Currency("GBP")
    assert inverted_rate2.date == Date(2023, 1, 1)
    assert inverted_rate2.value == Decimal("0.00664451")  # 1 / 150.5 ≈ 0.00664451


# LLM-generated content at query #3
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

    # Test data setup
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currencies

    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    today = date.today()

    # Create test service with some rates
    rates = {
        (eur, usd, today): FXRate(eur, usd, today, Decimal("1.2")),
        (usd, eur, today): FXRate(usd, eur, today, Decimal("0.8333")),
    }
    service = TestFXRateService(rates)

    # Test successful query
    result = service.query(eur, usd, today)
    assert result == FXRate(eur, usd, today, Decimal("1.2"))

    # Test query with None result (non-strict)
    result = service.query(eur, usd, date(2020, 1, 1))
    assert result is None

    # Test query with strict=True raises exception
    with pytest.raises(FXRateLookupError):
        service.query(eur, usd, date(2020, 1, 1), strict=True)

    # Test inverted rate query
    result = service.query(usd, eur, today)
    assert result == FXRate(usd, eur, today, Decimal("0.8333"))


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
    from pypara.currencies import Currencies
    import datetime
    from decimal import Decimal

    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)

    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    gbp = Currencies["GBP"]

    rates = {
        (eur, usd, today): FXRate(eur, usd, today, Decimal("1.20")),
        (usd, gbp, yesterday): FXRate(usd, gbp, yesterday, Decimal("0.80")),
        (gbp, eur, today): FXRate(gbp, eur, today, Decimal("1.10")),
    }

    service = MockFXRateService(rates)

    # Test case 1: All rates found
    queries1 = [(eur, usd, today), (usd, gbp, yesterday), (gbp, eur, today)]
    results1 = list(service.queries(queries1))
    assert len(results1) == 3
    assert results1[0] == rates[(eur, usd, today)]
    assert results1[1] == rates[(usd, gbp, yesterday)]
    assert results1[2] == rates[(gbp, eur, today)]

    # Test case 2: Some rates not found, non-strict
    queries2 = [(eur, usd, today), (eur, gbp, yesterday), (usd, eur, today)]
    results2 = list(service.queries(queries2))
    assert len(results2) == 3
    assert results2[0] == rates[(eur, usd, today)]
    assert results2[1] is None
    assert results2[2] is None

    # Test case 3: Some rates not found, strict
    queries3 = [(eur, usd, today), (eur, gbp, yesterday)]
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries3, strict=True))

    # Test case 4: Empty queries
    assert list(service.queries([])) == []

    # Test case 5: Single query
    queries5 = [(gbp, eur, today)]
    results5 = list(service.queries(queries5))
    assert len(results5) == 1
    assert results5[0] == rates[(gbp, eur, today)]


# LLM-generated content at query #5
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

    # Setup test data
    from datetime import date
    from pypara.currencies import Currencies
    from decimal import Decimal

    EUR = Currencies["EUR"]
    USD = Currencies["USD"]
    GBP = Currencies["GBP"]
    today = date.today()

    rates = {
        (EUR, USD, today): FXRate(EUR, USD, today, Decimal("1.2")),
        (EUR, GBP, today): FXRate(EUR, GBP, today, Decimal("0.8")),
    }

    service = MockFXRateService(rates)

    # Test 1: Normal queries with existing rates
    queries = [(EUR, USD, today), (EUR, GBP, today)]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == rates[(EUR, USD, today)]
    assert results[1] == rates[(EUR, GBP, today)]

    # Test 2: Query with non-existing rate (non-strict mode)
    queries = [(EUR, USD, today), (GBP, USD, today)]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == rates[(EUR, USD, today)]
    assert results[1] is None

    # Test 3: Query with non-existing rate (strict mode)
    queries = [(EUR, USD, today), (GBP, USD, today)]
    try:
        list(service.queries(queries, strict=True))
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == GBP
        assert e.ccy2 == USD
        assert e.asof == today

    # Test 4: Empty queries
    queries = []
    results = list(service.queries(queries))
    assert len(results) == 0

    # Test 5: Multiple queries with some existing and some non-existing rates
    queries = [(EUR, USD, today), (GBP, USD, today), (EUR, GBP, today)]
    results = list(service.queries(queries))
    assert len(results) == 3
    assert results[0] == rates[(EUR, USD, today)]
    assert results[1] is None
    assert results[2] == rates[(EUR, GBP, today)]


# LLM-generated content at query #6
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
            for ccy1, ccy2, asof in queries:
                rate = self.query(ccy1, ccy2, asof, strict)
                yield rate

    # Test data
    from datetime import date
    from pypara.currencies import Currencies
    EUR = Currencies["EUR"]
    USD = Currencies["USD"]
    GBP = Currencies["GBP"]

    rates = {
        (EUR, USD, date(2023, 1, 1)): FXRate(EUR, USD, date(2023, 1, 1), Decimal("1.1")),
        (EUR, GBP, date(2023, 1, 1)): FXRate(EUR, GBP, date(2023, 1, 1), Decimal("0.9")),
        (USD, GBP, date(2023, 1, 2)): FXRate(USD, GBP, date(2023, 1, 2), Decimal("0.8")),
    }

    service = TestFXRateService(rates)

    # Test 1: Query existing rates
    queries = [
        (EUR, USD, date(2023, 1, 1)),
        (EUR, GBP, date(2023, 1, 1)),
        (USD, GBP, date(2023, 1, 2)),
    ]
    results = list(service.queries(queries))
    assert len(results) == 3
    assert results[0] == FXRate(EUR, USD, date(2023, 1, 1), Decimal("1.1"))
    assert results[1] == FXRate(EUR, GBP, date(2023, 1, 1), Decimal("0.9"))
    assert results[2] == FXRate(USD, GBP, date(2023, 1, 2), Decimal("0.8"))

    # Test 2: Query with non-existing rate (non-strict mode)
    queries = [
        (EUR, USD, date(2023, 1, 1)),
        (EUR, GBP, date(2023, 1, 3)),  # Non-existing
    ]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == FXRate(EUR, USD, date(2023, 1, 1), Decimal("1.1"))
    assert results[1] is None

    # Test 3: Query with non-existing rate (strict mode)
    queries = [
        (EUR, USD, date(2023, 1, 1)),
        (EUR, GBP, date(2023, 1, 3)),  # Non-existing
    ]
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))

    # Test 4: Empty queries
    assert list(service.queries([])) == []


# LLM-generated content at query #7
#--------------------------

```python
def test_FXRateService_queries():
    # Setup
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if ccy1.code == "EUR" and ccy2.code == "USD" and asof == Date(2023, 1, 1):
                return FXRate(ccy1, ccy2, asof, Decimal("1.1"))
            elif ccy1.code == "USD" and ccy2.code == "GBP" and asof == Date(2023, 1, 2):
                return FXRate(ccy1, ccy2, asof, Decimal("0.8"))
            return None

        def queries(self, queries, strict=False):
            results = []
            for ccy1, ccy2, asof in queries:
                result = self.query(ccy1, ccy2, asof)
                if result is None and strict:
                    raise FXRateLookupError(ccy1, ccy2, asof)
                results.append(result)
            return results

    service = MockFXRateService()
    eur = Currency("EUR", "Euro", 2)
    usd = Currency("USD", "US Dollar", 2)
    gbp = Currency("GBP", "British Pound", 2)

    # Test with valid queries
    queries = [
        (eur, usd, Date(2023, 1, 1)),
        (usd, gbp, Date(2023, 1, 2))
    ]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == FXRate(eur, usd, Date(2023, 1, 1), Decimal("1.1"))
    assert results[1] == FXRate(usd, gbp, Date(2023, 1, 2), Decimal("0.8"))

    # Test with invalid query (non-strict)
    queries = [
        (eur, usd, Date(2023, 1, 1)),
        (eur, gbp, Date(2023, 1, 3))  # Invalid
    ]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == FXRate(eur, usd, Date(2023, 1, 1), Decimal("1.1"))
    assert results[1] is None

    # Test with invalid query (strict)
    queries = [
        (eur, gbp, Date(2023, 1, 3))  # Invalid
    ]
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))


# LLM-generated content at query #8
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
                rate = self.query(ccy1, ccy2, asof, strict)
                results.append(rate)
            return results

    # Create test data
    eur = Currency("EUR", "Euro", "€")
    usd = Currency("USD", "US Dollar", "$")
    date1 = Date(2023, 1, 1)
    date2 = Date(2023, 1, 2)

    rate1 = FXRate(eur, usd, date1, Decimal("1.1"))
    rate2 = FXRate(eur, usd, date2, Decimal("1.2"))
    rate3 = FXRate(usd, eur, date1, Decimal("0.9"))

    # Initialize mock service with test rates
    service = MockFXRateService([rate1, rate2, rate3])

    # Test queries with existing rates
    queries = [(eur, usd, date1), (eur, usd, date2), (usd, eur, date1)]
    results = list(service.queries(queries))
    assert len(results) == 3
    assert results[0] == rate1
    assert results[1] == rate2
    assert results[2] == rate3

    # Test queries with non-existing rate (non-strict mode)
    queries = [(eur, usd, Date(2023, 1, 3))]
    results = list(service.queries(queries))
    assert len(results) == 1
    assert results[0] is None

    # Test queries with non-existing rate (strict mode)
    queries = [(eur, usd, Date(2023, 1, 3))]
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))

    # Test empty queries
    assert list(service.queries([])) == []


# LLM-generated content at query #9
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
                key = (ccy1, ccy2, asof)
                if key in self.rates:
                    results.append(self.rates[key])
                else:
                    if strict:
                        raise FXRateLookupError(ccy1, ccy2, asof)
                    results.append(None)
            return results

    # Create test data
    test_date = Date(2023, 1, 1)
    eur = Currency("EUR", "Euro", 978)
    usd = Currency("USD", "US Dollar", 840)
    gbp = Currency("GBP", "British Pound", 826)

    rates = {
        (eur, usd, test_date): FXRate(eur, usd, test_date, Decimal("1.10")),
        (usd, gbp, test_date): FXRate(usd, gbp, test_date, Decimal("0.80")),
    }

    service = TestFXRateService(rates)

    # Test case 1: All rates found
    queries1 = [(eur, usd, test_date), (usd, gbp, test_date)]
    results1 = list(service.queries(queries1))
    assert len(results1) == 2
    assert results1[0] == rates[(eur, usd, test_date)]
    assert results1[1] == rates[(usd, gbp, test_date)]

    # Test case 2: Some rates not found, non-strict mode
    queries2 = [(eur, usd, test_date), (gbp, eur, test_date)]
    results2 = list(service.queries(queries2))
    assert len(results2) == 2
    assert results2[0] == rates[(eur, usd, test_date)]
    assert results2[1] is None

    # Test case 3: Some rates not found, strict mode
    queries3 = [(eur, usd, test_date), (gbp, eur, test_date)]
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries3, strict=True))

    # Test case 4: Empty queries
    queries4 = []
    results4 = list(service.queries(queries4))
    assert len(results4) == 0

    # Test case 5: Multiple queries with same currency pair but different dates
    test_date2 = Date(2023, 1, 2)
    rates2 = {
        (eur, usd, test_date): FXRate(eur, usd, test_date, Decimal("1.10")),
        (eur, usd, test_date2): FXRate(eur, usd, test_date2, Decimal("1.15")),
    }
    service2 = TestFXRateService(rates2)
    queries5 = [(eur, usd, test_date), (eur, usd, test_date2)]
    results5 = list(service2.queries(queries5))
    assert len(results5) == 2
    assert results5[0] == rates2[(eur, usd, test_date)]
    assert results5[1] == rates2[(eur, usd, test_date2)]


# LLM-generated content at query #10
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

    # Test 1: Basic query with existing rates
    queries = [
        (EUR, USD, date(2023, 1, 1)),
        (USD, GBP, date(2023, 1, 1)),
    ]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == test_rates[(EUR, USD, date(2023, 1, 1))]
    assert results[1] == test_rates[(USD, GBP, date(2023, 1, 1))]

    # Test 2: Query with non-existing rate (non-strict mode)
    queries = [
        (EUR, USD, date(2023, 1, 1)),
        (EUR, GBP, date(2023, 1, 1)),  # This rate doesn't exist
    ]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == test_rates[(EUR, USD, date(2023, 1, 1))]
    assert results[1] is None

    # Test 3: Query with non-existing rate (strict mode)
    queries = [
        (EUR, GBP, date(2023, 1, 1)),  # This rate doesn't exist
    ]
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))

    # Test 4: Empty queries
    results = list(service.queries([]))
    assert len(results) == 0

    # Test 5: Multiple queries with mixed results
    queries = [
        (EUR, USD, date(2023, 1, 1)),
        (EUR, GBP, date(2023, 1, 1)),  # Doesn't exist
        (EUR, GBP, date(2023, 1, 2)),
        (USD, GBP, date(2023, 1, 1)),
    ]
    results = list(service.queries(queries))
    assert len(results) == 4
    assert results[0] == test_rates[(EUR, USD, date(2023, 1, 1))]
    assert results[1] is None
    assert results[2] == test_rates[(EUR, GBP, date(2023, 1, 2))]
    assert results[3] == test_rates[(USD, GBP, date(2023, 1, 1))]


# LLM-generated content at query #11
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

    EUR = Currencies["EUR"]
    USD = Currencies["USD"]
    GBP = Currencies["GBP"]

    test_rates = {
        (EUR, USD, date(2023, 1, 1)): FXRate(EUR, USD, date(2023, 1, 1), Decimal("1.1")),
        (USD, GBP, date(2023, 1, 1)): FXRate(USD, GBP, date(2023, 1, 1), Decimal("0.8")),
        (EUR, GBP, date(2023, 1, 2)): FXRate(EUR, GBP, date(2023, 1, 2), Decimal("0.88")),
    }

    service = TestFXRateService(test_rates)

    # Test 1: Basic query with existing rates
    queries = [
        (EUR, USD, date(2023, 1, 1)),
        (USD, GBP, date(2023, 1, 1)),
    ]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == test_rates[(EUR, USD, date(2023, 1, 1))]
    assert results[1] == test_rates[(USD, GBP, date(2023, 1, 1))]

    # Test 2: Query with non-existing rate (non-strict mode)
    queries = [
        (EUR, USD, date(2023, 1, 1)),
        (GBP, USD, date(2023, 1, 1)),  # This rate doesn't exist
    ]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == test_rates[(EUR, USD, date(2023, 1, 1))]
    assert results[1] is None

    # Test 3: Query with non-existing rate (strict mode)
    queries = [
        (EUR, USD, date(2023, 1, 1)),
        (GBP, USD, date(2023, 1, 1)),  # This rate doesn't exist
    ]
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))

    # Test 4: Empty queries
    assert list(service.queries([])) == []

    # Test 5: Query with different dates
    queries = [
        (EUR, USD, date(2023, 1, 1)),
        (EUR, GBP, date(2023, 1, 2)),
    ]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == test_rates[(EUR, USD, date(2023, 1, 1))]
    assert results[1] == test_rates[(EUR, GBP, date(2023, 1, 2))]


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
            if key in self.rates:
                return self.rates[key]
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None

        def queries(self, queries, strict=False):
            for query in queries:
                yield self.query(*query, strict=strict)

    # Setup test data
    from pypara.currencies import Currencies
    import datetime
    from decimal import Decimal

    EUR = Currencies["EUR"]
    USD = Currencies["USD"]
    today = datetime.date.today()
    rate = FXRate(EUR, USD, today, Decimal("1.2"))

    # Initialize service with test data
    service = TestFXRateService({(EUR, USD, today): rate})

    # Test successful query
    result = service.query(EUR, USD, today)
    assert result == rate

    # Test query with non-existent rate (non-strict)
    result = service.query(USD, EUR, today)
    assert result is None

    # Test query with non-existent rate (strict)
    with pytest.raises(FXRateLookupError):
        service.query(USD, EUR, today, strict=True)

    # Test query with same currency
    same_currency_rate = FXRate(EUR, EUR, today, Decimal("1"))
    service_with_same_currency = TestFXRateService({(EUR, EUR, today): same_currency_rate})
    result = service_with_same_currency.query(EUR, EUR, today)
    assert result == same_currency_rate


# LLM-generated content at query #13
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

    # Test data
    eur = Currency("EUR", "Euro", 2)
    usd = Currency("USD", "US Dollar", 2)
    today = Date(2023, 1, 1)
    tomorrow = Date(2023, 1, 2)

    rate1 = FXRate(eur, usd, today, Decimal("1.1"))
    rate2 = FXRate(usd, eur, tomorrow, Decimal("0.9"))

    # Initialize service with test rates
    service = MockFXRateService([rate1, rate2])

    # Test queries with existing rates
    queries = [(eur, usd, today), (usd, eur, tomorrow)]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == rate1
    assert results[1] == rate2

    # Test queries with non-existing rate (non-strict)
    queries = [(eur, usd, tomorrow)]
    results = list(service.queries(queries))
    assert len(results) == 1
    assert results[0] is None

    # Test queries with non-existing rate (strict)
    queries = [(eur, usd, tomorrow)]
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))

    # Test empty queries
    assert list(service.queries([])) == []


# LLM-generated content at query #14
#--------------------------

```python
def test_FXRateService_queries():
    # Create a mock FXRateService implementation
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
    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    today = date.today()
    yesterday = today - timedelta(days=1)

    rates = {
        (eur, usd, today): FXRate(eur, usd, today, Decimal("1.2")),
        (usd, eur, today): FXRate(usd, eur, today, Decimal("0.8333")),
        (eur, usd, yesterday): FXRate(eur, usd, yesterday, Decimal("1.15")),
    }

    # Test normal queries
    service = MockFXRateService(rates)
    queries = [
        (eur, usd, today),
        (usd, eur, today),
        (eur, usd, yesterday),
        (eur, usd, today - timedelta(days=2))  # Not found
    ]
    results = list(service.queries(queries))

    assert len(results) == 4
    assert results[0] == rates[(eur, usd, today)]
    assert results[1] == rates[(usd, eur, today)]
    assert results[2] == rates[(eur, usd, yesterday)]
    assert results[3] is None

    # Test strict queries
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))

    # Test empty queries
    assert list(service.queries([])) == []


# LLM-generated content at query #15
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

    # Create test data
    from datetime import date
    from pypara.currencies import Currencies
    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    today = date.today()
    rate = FXRate(eur, usd, today, Decimal("1.2"))

    # Test successful query
    service = TestFXRateService({(eur, usd, today): rate})
    result = service.query(eur, usd, today)
    assert result == rate

    # Test query with non-existent rate (non-strict)
    result = service.query(usd, eur, today)
    assert result is None

    # Test query with non-existent rate (strict)
    try:
        service.query(usd, eur, today, strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == usd
        assert e.ccy2 == eur
        assert e.asof == today


# LLM-generated content at query #16
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
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currencies

    test_rates = {
        (Currencies["EUR"], Currencies["USD"], date(2023, 1, 1)): FXRate(Currencies["EUR"], Currencies["USD"], date(2023, 1, 1), Decimal("1.1")),
        (Currencies["USD"], Currencies["EUR"], date(2023, 1, 1)): FXRate(Currencies["USD"], Currencies["EUR"], date(2023, 1, 1), Decimal("0.909")),
        (Currencies["EUR"], Currencies["GBP"], date(2023, 1, 1)): FXRate(Currencies["EUR"], Currencies["GBP"], date(2023, 1, 1), Decimal("0.85")),
    }

    service = TestFXRateService(test_rates)

    # Test existing rate
    result = service.query(Currencies["EUR"], Currencies["USD"], date(2023, 1, 1))
    assert result == FXRate(Currencies["EUR"], Currencies["USD"], date(2023, 1, 1), Decimal("1.1"))

    # Test inverted rate
    result = service.query(Currencies["USD"], Currencies["EUR"], date(2023, 1, 1))
    assert result == FXRate(Currencies["USD"], Currencies["EUR"], date(2023, 1, 1), Decimal("0.909"))

    # Test non-existing rate without strict
    result = service.query(Currencies["EUR"], Currencies["JPY"], date(2023, 1, 1))
    assert result is None

    # Test non-existing rate with strict
    try:
        service.query(Currencies["EUR"], Currencies["JPY"], date(2023, 1, 1), strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == Currencies["EUR"]
        assert e.ccy2 == Currencies["JPY"]
        assert e.asof == date(2023, 1, 1)

    # Test same currency
    result = service.query(Currencies["EUR"], Currencies["EUR"], date(2023, 1, 1))
    assert result is None


# LLM-generated content at query #17
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
                rate = self.query(ccy1, ccy2, asof)
                if rate is None and strict:
                    raise FXRateLookupError(ccy1, ccy2, asof)
                results.append(rate)
            return results

    # Create test data
    from datetime import date
    from pypara.currencies import Currencies
    rate1 = FXRate(Currencies["EUR"], Currencies["USD"], date(2023, 1, 1), Decimal("1.1"))
    rate2 = FXRate(Currencies["USD"], Currencies["JPY"], date(2023, 1, 1), Decimal("130.5"))
    rate3 = FXRate(Currencies["EUR"], Currencies["GBP"], date(2023, 1, 2), Decimal("0.85"))

    service = TestFXRateService([rate1, rate2, rate3])

    # Test successful queries
    queries = [
        (Currencies["EUR"], Currencies["USD"], date(2023, 1, 1)),
        (Currencies["USD"], Currencies["JPY"], date(2023, 1, 1)),
        (Currencies["EUR"], Currencies["GBP"], date(2023, 1, 2))
    ]
    results = service.queries(queries)
    assert results == [rate1, rate2, rate3]

    # Test with missing rate (non-strict)
    queries_with_missing = [
        (Currencies["EUR"], Currencies["USD"], date(2023, 1, 1)),
        (Currencies["USD"], Currencies["CAD"], date(2023, 1, 1))  # Missing
    ]
    results = service.queries(queries_with_missing)
    assert results == [rate1, None]

    # Test with missing rate (strict)
    try:
        service.queries(queries_with_missing, strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == Currencies["USD"]
        assert e.ccy2 == Currencies["CAD"]
        assert e.asof == date(2023, 1, 1)

    # Test empty queries
    assert list(service.queries([])) == []


# LLM-generated content at query #18
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

    # Test data
    eur = Currency("EUR", "Euro", 2)
    usd = Currency("USD", "US Dollar", 2)
    today = Date.today()
    yesterday = today - Date.resolution

    rate1 = FXRate(eur, usd, today, Decimal("1.2"))
    rate2 = FXRate(usd, eur, yesterday, Decimal("0.8"))

    # Create the mock service with some rates
    service = MockFXRateService({
        (eur, usd, today): rate1,
        (usd, eur, yesterday): rate2
    })

    # Test successful queries
    queries = [(eur, usd, today), (usd, eur, yesterday)]
    results = list(service.queries(queries))
    assert results == [rate1, rate2]

    # Test with missing rate (non-strict)
    queries_with_missing = [(eur, usd, today), (eur, usd, yesterday)]
    results = list(service.queries(queries_with_missing))
    assert results == [rate1, None]

    # Test with missing rate (strict)
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries_with_missing, strict=True))

    # Test empty queries
    assert list(service.queries([])) == []


# LLM-generated content at query #19
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
    yesterday = today - timedelta(days=1)

    rates = {
        (eur, usd, today): FXRate(eur, usd, today, Decimal("1.2")),
        (usd, gbp, today): FXRate(usd, gbp, today, Decimal("0.8")),
        (eur, gbp, yesterday): FXRate(eur, gbp, yesterday, Decimal("0.9")),
    }

    service = TestFXRateService(rates)

    # Test with existing rates
    queries = [
        (eur, usd, today),
        (usd, gbp, today),
        (eur, gbp, yesterday),
    ]
    results = list(service.queries(queries))
    assert len(results) == 3
    assert results[0] == FXRate(eur, usd, today, Decimal("1.2"))
    assert results[1] == FXRate(usd, gbp, today, Decimal("0.8"))
    assert results[2] == FXRate(eur, gbp, yesterday, Decimal("0.9"))

    # Test with non-existing rates (non-strict)
    queries = [
        (eur, usd, yesterday),
        (gbp, usd, today),
    ]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] is None
    assert results[1] is None

    # Test with non-existing rates (strict)
    queries = [
        (eur, usd, yesterday),
    ]
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))


# LLM-generated content at query #20
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

    # Test data setup
    from datetime import date
    from pypara.currencies import Currencies

    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    today = date.today()

    rates = {
        (eur, usd, today): FXRate(eur, usd, today, Decimal("1.2")),
        (usd, eur, today): FXRate(usd, eur, today, Decimal("0.8333")),
    }

    service = TestFXRateService(rates)

    # Test successful query
    result = service.query(eur, usd, today)
    assert result == FXRate(eur, usd, today, Decimal("1.2"))

    # Test query with inversion
    result = service.query(usd, eur, today)
    assert result == FXRate(usd, eur, today, Decimal("0.8333"))

    # Test query with non-existent rate (non-strict)
    result = service.query(eur, usd, date(2020, 1, 1))
    assert result is None

    # Test query with non-existent rate (strict)
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
def test_FXRateService_query():
    # Setup
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            if ccy1.code == "EUR" and ccy2.code == "USD" and asof == Date(2023, 1, 1):
                return FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1"))
            return None

        def queries(self, queries: Iterable[Tuple[Currency, Currency, Date]], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = MockFXRateService()

    # Test successful query
    result = service.query(Currency("EUR"), Currency("USD"), Date(2023, 1, 1))
    assert result == FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1"))

    # Test non-existent rate without strict
    result = service.query(Currency("USD"), Currency("JPY"), Date(2023, 1, 1))
    assert result is None

    # Test non-existent rate with strict (should raise FXRateLookupError)
    try:
        service.query(Currency("USD"), Currency("JPY"), Date(2023, 1, 1), strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == Currency("USD")
        assert e.ccy2 == Currency("JPY")
        assert e.asof == Date(2023, 1, 1)


# LLM-generated content at query #22
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

    # Initialize the test service
    service = TestFXRateService()

    # Test successful query
    result = service.query(Currency("EUR"), Currency("USD"), Date(2023, 1, 1))
    assert result == FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1"))

    # Test inverted query
    result = service.query(Currency("USD"), Currency("EUR"), Date(2023, 1, 1))
    assert result == FXRate(Currency("USD"), Currency("EUR"), Date(2023, 1, 1), Decimal("0.909"))

    # Test non-existent rate without strict
    result = service.query(Currency("GBP"), Currency("USD"), Date(2023, 1, 1))
    assert result is None

    # Test non-existent rate with strict
    with pytest.raises(FXRateLookupError):
        service.query(Currency("GBP"), Currency("USD"), Date(2023, 1, 1), strict=True)


# LLM-generated content at query #23
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
            return (self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries)

    # Initialize the test service
    service = TestFXRateService()

    # Test successful query
    rate = service.query(Currency("EUR"), Currency("USD"), Date(2023, 1, 1))
    assert rate is not None
    assert rate.ccy1 == Currency("EUR")
    assert rate.ccy2 == Currency("USD")
    assert rate.date == Date(2023, 1, 1)
    assert rate.value == Decimal("1.10")

    # Test query with no result (non-strict)
    rate = service.query(Currency("EUR"), Currency("GBP"), Date(2023, 1, 1))
    assert rate is None

    # Test query with no result (strict)
    try:
        service.query(Currency("EUR"), Currency("GBP"), Date(2023, 1, 1), strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == Currency("EUR")
        assert e.ccy2 == Currency("GBP")
        assert e.asof == Date(2023, 1, 1)

    # Test inverted rate
    rate = service.query(Currency("USD"), Currency("EUR"), Date(2023, 1, 1))
    assert rate is not None
    assert rate.ccy1 == Currency("USD")
    assert rate.ccy2 == Currency("EUR")
    assert rate.date == Date(2023, 1, 1)
    assert rate.value == Decimal("0.90")


# LLM-generated content at query #24
#--------------------------

```python
def test_FXRateService_query():
    # Create a mock FXRateService implementation for testing
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            if ccy1.code == "EUR" and ccy2.code == "USD" and asof == Date(2023, 1, 1):
                return FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1"))
            elif strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    # Test successful query
    service = MockFXRateService()
    result = service.query(Currency("EUR"), Currency("USD"), Date(2023, 1, 1))
    assert result == FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1"))

    # Test query with no result (non-strict)
    result = service.query(Currency("GBP"), Currency("JPY"), Date(2023, 1, 1))
    assert result is None

    # Test query with no result (strict)
    try:
        service.query(Currency("GBP"), Currency("JPY"), Date(2023, 1, 1), strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1.code == "GBP"
        assert e.ccy2.code == "JPY"
        assert e.asof == Date(2023, 1, 1)


# LLM-generated content at query #25
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

    # Setup test data
    eur = Currency("EUR", "Euro", 2)
    usd = Currency("USD", "US Dollar", 2)
    today = Date.today()
    yesterday = today - datetime.timedelta(days=1)

    rate1 = FXRate(eur, usd, today, Decimal("1.2"))
    rate2 = FXRate(usd, eur, yesterday, Decimal("0.8"))

    service = TestFXRateService([rate1, rate2])

    # Test successful queries
    queries = [(eur, usd, today), (usd, eur, yesterday)]
    results = service.queries(queries)
    assert list(results) == [rate1, rate2]

    # Test with non-existent rate (non-strict)
    queries = [(eur, usd, yesterday)]
    results = service.queries(queries)
    assert list(results) == [None]

    # Test with non-existent rate (strict)
    queries = [(eur, usd, yesterday)]
    with pytest.raises(FXRateLookupError):
        service.queries(queries, strict=True)

    # Test empty queries
    assert list(service.queries([])) == []


# LLM-generated content at query #26
#--------------------------

```python
def test_FXRateService_query():
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
    test_rates = {
        (Currencies["EUR"], Currencies["USD"], date(2023, 1, 1)): FXRate(Currencies["EUR"], Currencies["USD"], date(2023, 1, 1), Decimal("1.1")),
        (Currencies["USD"], Currencies["EUR"], date(2023, 1, 1)): FXRate(Currencies["USD"], Currencies["EUR"], date(2023, 1, 1), Decimal("0.909")),
    }
    service = TestFXRateService(test_rates)

    # Test successful query
    result = service.query(Currencies["EUR"], Currencies["USD"], date(2023, 1, 1))
    assert result == test_rates[(Currencies["EUR"], Currencies["USD"], date(2023, 1, 1))]

    # Test query with non-existent rate (non-strict)
    result = service.query(Currencies["GBP"], Currencies["USD"], date(2023, 1, 1))
    assert result is None

    # Test query with non-existent rate (strict)
    with pytest.raises(FXRateLookupError):
        service.query(Currencies["GBP"], Currencies["USD"], date(2023, 1, 1), strict=True)

    # Test inverted rate
    result = service.query(Currencies["USD"], Currencies["EUR"], date(2023, 1, 1))
    assert result == test_rates[(Currencies["USD"], Currencies["EUR"], date(2023, 1, 1))]


# LLM-generated content at query #27
#--------------------------

```python
def test_FXRateService_query():
    # Create a mock FXRateService implementation for testing
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
            for ccy1, ccy2, asof in queries:
                yield self.query(ccy1, ccy2, asof, strict)

    # Test data
    eur = Currency("EUR", "Euro", 2)
    usd = Currency("USD", "US Dollar", 2)
    today = Date.today()
    yesterday = today - Date.resolution

    # Test cases
    rates = {
        (eur, usd, today): FXRate(eur, usd, today, Decimal("1.2")),
        (usd, eur, yesterday): FXRate(usd, eur, yesterday, Decimal("0.8")),
    }

    service = MockFXRateService(rates)

    # Test successful query
    result = service.query(eur, usd, today)
    assert result == FXRate(eur, usd, today, Decimal("1.2"))

    # Test query with None result (non-strict)
    result = service.query(eur, usd, yesterday)
    assert result is None

    # Test query with lookup error (strict)
    with pytest.raises(FXRateLookupError):
        service.query(eur, usd, yesterday, strict=True)

    # Test inverted rate
    result = service.query(usd, eur, yesterday)
    assert result == FXRate(usd, eur, yesterday, Decimal("0.8"))
    assert ~result == FXRate(eur, usd, yesterday, Decimal("1.25"))


# LLM-generated content at query #28
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

    EUR = Currencies["EUR"]
    USD = Currencies["USD"]
    today = date.today()

    test_rates = {
        (EUR, USD, today): FXRate(EUR, USD, today, Decimal("1.2")),
        (USD, EUR, today): FXRate(USD, EUR, today, Decimal("0.8333")),
    }

    service = TestFXRateService(test_rates)

    # Test successful query
    rate = service.query(EUR, USD, today)
    assert rate == FXRate(EUR, USD, today, Decimal("1.2"))

    # Test inverted rate
    rate = service.query(USD, EUR, today)
    assert rate == FXRate(USD, EUR, today, Decimal("0.8333"))

    # Test non-existent rate without strict
    rate = service.query(EUR, USD, date(2020, 1, 1))
    assert rate is None

    # Test non-existent rate with strict
    try:
        service.query(EUR, USD, date(2020, 1, 1), strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == EUR
        assert e.ccy2 == USD
        assert e.asof == date(2020, 1, 1)

    # Test same currency
    rate = service.query(EUR, EUR, today)
    assert rate is None  # Assuming not in test_rates

    # Test with strict for same currency not in rates
    try:
        service.query(EUR, EUR, today, strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == EUR
        assert e.ccy2 == EUR
        assert e.asof == today


# LLM-generated content at query #29
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

    # Test query with non-existent rate (non-strict)
    result = service.query(usd, eur, today)
    assert result is None

    # Test query with non-existent rate (strict)
    with pytest.raises(FXRateLookupError):
        service.query(usd, eur, today, strict=True)


# LLM-generated content at query #30
#--------------------------

```python
def test_FXRateService_query():
    # Create a mock FXRateService instance
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            if ccy1.code == "EUR" and ccy2.code == "USD" and asof == Date(2023, 1, 1):
                return FXRate(Currency("EUR"), Currency("USD"), asof, Decimal("1.1"))
            elif strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            else:
                return None

        def queries(self, queries: Iterable[Tuple[Currency, Currency, Date]], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return (self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries)

    service = MockFXRateService()

    # Test successful query
    result = service.query(Currency("EUR"), Currency("USD"), Date(2023, 1, 1))
    assert result == FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1"))

    # Test query with non-existent rate (non-strict)
    result = service.query(Currency("GBP"), Currency("JPY"), Date(2023, 1, 1))
    assert result is None

    # Test query with non-existent rate (strict)
    with pytest.raises(FXRateLookupError):
        service.query(Currency("GBP"), Currency("JPY"), Date(2023, 1, 1), strict=True)


# LLM-generated content at query #31
#--------------------------

```python
def test_FXRateService_query():
    # Create a mock FXRateService implementation for testing
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            if ccy1.code == "EUR" and ccy2.code == "USD" and asof == Date(2023, 1, 1):
                return FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.10"))
            elif strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    # Initialize the mock service
    service = MockFXRateService()

    # Test successful query
    result = service.query(Currency("EUR"), Currency("USD"), Date(2023, 1, 1))
    assert result == FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.10"))

    # Test non-strict query with missing rate
    result = service.query(Currency("USD"), Currency("JPY"), Date(2023, 1, 1))
    assert result is None

    # Test strict query with missing rate
    with pytest.raises(FXRateLookupError):
        service.query(Currency("USD"), Currency("JPY"), Date(2023, 1, 1), strict=True)


# LLM-generated content at query #32
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
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    # Setup test data
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currencies

    EUR = Currencies["EUR"]
    USD = Currencies["USD"]
    today = date.today()

    test_rates = {
        (EUR, USD, today): FXRate(EUR, USD, today, Decimal("1.2")),
        (USD, EUR, today): FXRate(USD, EUR, today, Decimal("0.8333")),
    }

    service = TestFXRateService(test_rates)

    # Test successful query
    result = service.query(EUR, USD, today)
    assert result == FXRate(EUR, USD, today, Decimal("1.2"))

    # Test query with None result (non-strict)
    result = service.query(EUR, USD, date(2020, 1, 1))
    assert result is None

    # Test query with lookup error (strict)
    with pytest.raises(FXRateLookupError):
        service.query(EUR, USD, date(2020, 1, 1), strict=True)

    # Test inverted rate
    result = service.query(USD, EUR, today)
    assert result == ~FXRate(EUR, USD, today, Decimal("1.2"))


# LLM-generated content at query #33
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
                result = self.query(ccy1, ccy2, asof, strict)
                results.append(result)
            return results

    # Setup test data
    from datetime import date
    from pypara.currencies import Currencies
    from decimal import Decimal

    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    today = date.today()

    rates = {
        (eur, usd, today): FXRate(eur, usd, today, Decimal("1.2")),
        (usd, eur, today): FXRate(usd, eur, today, Decimal("0.8333")),
    }

    service = TestFXRateService(rates)

    # Test 1: Basic query with existing rates
    queries = [(eur, usd, today), (usd, eur, today)]
    results = service.queries(queries)
    assert len(results) == 2
    assert results[0] == rates[(eur, usd, today)]
    assert results[1] == rates[(usd, eur, today)]

    # Test 2: Query with non-existing rate (non-strict)
    queries = [(eur, usd, today), (eur, usd, date(2020, 1, 1))]
    results = service.queries(queries)
    assert len(results) == 2
    assert results[0] == rates[(eur, usd, today)]
    assert results[1] is None

    # Test 3: Query with non-existing rate (strict)
    queries = [(eur, usd, today), (eur, usd, date(2020, 1, 1))]
    with pytest.raises(FXRateLookupError):
        service.queries(queries, strict=True)

    # Test 4: Empty queries
    assert list(service.queries([])) == []

    # Test 5: Multiple queries with some missing
    queries = [
        (eur, usd, today),
        (usd, eur, today),
        (eur, usd, date(2020, 1, 1)),
        (usd, eur, date(2020, 1, 1)),
    ]
    results = service.queries(queries)
    assert len(results) == 4
    assert results[0] == rates[(eur, usd, today)]
    assert results[1] == rates[(usd, eur, today)]
    assert results[2] is None
    assert results[3] is None


# LLM-generated content at query #34
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
            return (self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries)

    # Setup test data
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currencies

    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    today = date.today()

    rates = {
        (eur, usd, today): FXRate(eur, usd, today, Decimal("1.2")),
        (usd, eur, today): FXRate(usd, eur, today, Decimal("0.8333")),
    }

    service = TestFXRateService(rates)

    # Test successful query
    result = service.query(eur, usd, today)
    assert result == FXRate(eur, usd, today, Decimal("1.2"))

    # Test inverted rate
    result = service.query(usd, eur, today)
    assert result == FXRate(usd, eur, today, Decimal("0.8333"))

    # Test non-existent rate without strict
    result = service.query(eur, usd, date(2020, 1, 1))
    assert result is None

    # Test non-existent rate with strict
    with pytest.raises(FXRateLookupError):
        service.query(eur, usd, date(2020, 1, 1), strict=True)

    # Test same currency
    result = service.query(eur, eur, today)
    assert result is None  # Assuming no rate is stored for same currency


# LLM-generated content at query #35
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
    from pypara.currencies import Currencies
    import datetime
    from decimal import Decimal

    EUR = Currencies["EUR"]
    USD = Currencies["USD"]
    GBP = Currencies["GBP"]
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)

    rate1 = FXRate(EUR, USD, today, Decimal("1.2"))
    rate2 = FXRate(GBP, USD, today, Decimal("1.4"))
    rate3 = FXRate(EUR, GBP, yesterday, Decimal("0.8"))

    rates = {
        (EUR, USD, today): rate1,
        (GBP, USD, today): rate2,
        (EUR, GBP, yesterday): rate3
    }

    service = MockFXRateService(rates)

    # Test basic queries
    queries = [
        (EUR, USD, today),
        (GBP, USD, today),
        (EUR, GBP, yesterday)
    ]
    results = list(service.queries(queries))
    assert results == [rate1, rate2, rate3]

    # Test with missing rate (non-strict)
    queries_with_missing = [
        (EUR, USD, today),
        (EUR, USD, yesterday)  # Missing
    ]
    results = list(service.queries(queries_with_missing))
    assert results == [rate1, None]

    # Test with missing rate (strict)
    try:
        list(service.queries(queries_with_missing, strict=True))
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == EUR
        assert e.ccy2 == USD
        assert e.asof == yesterday

    # Test empty queries
    assert list(service.queries([])) == []

    # Test with inverted rates
    queries_inverted = [
        (USD, EUR, today),
        (USD, GBP, today),
        (GBP, EUR, yesterday)
    ]
    results = list(service.queries(queries_inverted))
    assert results == [~rate1, ~rate2, ~rate3]


# LLM-generated content at query #36
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

    # Create test rates
    test_rates = {
        (eur, usd, today): FXRate(eur, usd, today, Decimal("1.2")),
        (usd, eur, today): FXRate(usd, eur, today, Decimal("0.8333")),
    }

    # Initialize test service
    service = TestFXRateService(test_rates)

    # Test successful query
    result = service.query(eur, usd, today)
    assert result == FXRate(eur, usd, today, Decimal("1.2"))

    # Test query with inversion
    result = service.query(usd, eur, today)
    assert result == FXRate(usd, eur, today, Decimal("0.8333"))

    # Test query with non-existent rate (non-strict)
    result = service.query(eur, usd, date(2020, 1, 1))
    assert result is None

    # Test query with non-existent rate (strict)
    with pytest.raises(FXRateLookupError):
        service.query(eur, usd, date(2020, 1, 1), strict=True)

    # Test query with same currency
    result = service.query(eur, eur, today)
    assert result == FXRate(eur, eur, today, Decimal("1"))


# LLM-generated content at query #37
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

    # Create test data
    test_date = Date(2023, 1, 1)
    eur = Currency("EUR", "Euro", 978)
    usd = Currency("USD", "US Dollar", 840)

    # Test successful query
    rates = {
        (eur, usd, test_date): FXRate(eur, usd, test_date, Decimal("1.10")),
        (usd, eur, test_date): FXRate(usd, eur, test_date, Decimal("0.91"))
    }
    service = MockFXRateService(rates)

    result = service.query(eur, usd, test_date)
    assert result == FXRate(eur, usd, test_date, Decimal("1.10"))

    # Test query with None result (non-strict)
    result = service.query(eur, usd, Date(2023, 1, 2))
    assert result is None

    # Test query with strict=True raising exception
    with pytest.raises(FXRateLookupError):
        service.query(eur, usd, Date(2023, 1, 2), strict=True)

    # Test inverted rate query
    result = service.query(usd, eur, test_date)
    assert result == FXRate(usd, eur, test_date, Decimal("0.91"))


# LLM-generated content at query #38
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

    # Test data
    from datetime import date
    from pypara.currencies import Currencies
    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    today = date.today()
    test_rate = FXRate(eur, usd, today, Decimal("1.2"))

    # Initialize service with test data
    service = TestFXRateService({(eur, usd, today): test_rate})

    # Test successful query
    result = service.query(eur, usd, today)
    assert result == test_rate

    # Test query with non-existent rate (non-strict)
    result = service.query(usd, eur, today)
    assert result is None

    # Test query with non-existent rate (strict)
    with pytest.raises(FXRateLookupError):
        service.query(usd, eur, today, strict=True)

    # Test query with same currency
    same_currency_rate = FXRate(eur, eur, today, ONE)
    service_with_same_currency = TestFXRateService({(eur, eur, today): same_currency_rate})
    result = service_with_same_currency.query(eur, eur, today)
    assert result == same_currency_rate


# LLM-generated content at query #39
#--------------------------

```python
def test_FXRateService_queries():
    # Create a mock FXRateService implementation
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            # Mock implementation for testing
            if ccy1.code == "EUR" and ccy2.code == "USD" and asof == Date(2023, 1, 1):
                return FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1"))
            elif ccy1.code == "USD" and ccy2.code == "EUR" and asof == Date(2023, 1, 1):
                return FXRate(Currency("USD"), Currency("EUR"), Date(2023, 1, 1), Decimal("0.9"))
            elif strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            else:
                return None

        def queries(self, queries: Iterable[Tuple[Currency, Currency, Date]], strict: bool = False) -> Iterable[Optional[FXRate]]:
            # Mock implementation for testing
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

    # Initialize the mock service
    service = MockFXRateService()

    # Test case 1: Valid queries
    queries = [
        (Currency("EUR"), Currency("USD"), Date(2023, 1, 1)),
        (Currency("USD"), Currency("EUR"), Date(2023, 1, 1))
    ]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1"))
    assert results[1] == FXRate(Currency("USD"), Currency("EUR"), Date(2023, 1, 1), Decimal("0.9"))

    # Test case 2: Invalid queries (non-strict)
    queries = [
        (Currency("EUR"), Currency("GBP"), Date(2023, 1, 1)),
        (Currency("USD"), Currency("EUR"), Date(2023, 1, 1))
    ]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] is None
    assert results[1] == FXRate(Currency("USD"), Currency("EUR"), Date(2023, 1, 1), Decimal("0.9"))

    # Test case 3: Invalid queries (strict)
    queries = [
        (Currency("EUR"), Currency("GBP"), Date(2023, 1, 1)),
        (Currency("USD"), Currency("EUR"), Date(2023, 1, 1))
    ]
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))


# LLM-generated content at query #40
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
                elif strict:
                    raise FXRateLookupError(ccy1, ccy2, asof)
                else:
                    results.append(None)
            return results

    # Test data
    currency1 = Currency("EUR")
    currency2 = Currency("USD")
    date1 = Date(2023, 1, 1)
    date2 = Date(2023, 1, 2)
    rate1 = FXRate(currency1, currency2, date1, Decimal("1.1"))
    rate2 = FXRate(currency1, currency2, date2, Decimal("1.2"))

    # Initialize service with test rates
    service = MockFXRateService({(currency1, currency2, date1): rate1, (currency1, currency2, date2): rate2})

    # Test queries with existing rates
    queries = [(currency1, currency2, date1), (currency1, currency2, date2)]
    results = list(service.queries(queries))
    assert results == [rate1, rate2]

    # Test queries with non-existing rate (non-strict)
    queries = [(currency1, currency2, date1), (currency2, currency1, date1)]
    results = list(service.queries(queries))
    assert results == [rate1, None]

    # Test queries with non-existing rate (strict)
    queries = [(currency1, currency2, date1), (currency2, currency1, date1)]
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))


# LLM-generated content at query #41
#--------------------------

```python
def test_FXRateService_query():
    # Create a mock FXRateService class for testing
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

    # Test successful query
    result = service.query(eur, usd, today)
    assert result == FXRate(eur, usd, today, Decimal("1.2"))

    # Test query with None result (non-strict)
    result = service.query(eur, usd, date(2020, 1, 1))
    assert result is None

    # Test query with FXRateLookupError (strict)
    try:
        service.query(eur, usd, date(2020, 1, 1), strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == eur
        assert e.ccy2 == usd
        assert e.asof == date(2020, 1, 1)


# LLM-generated content at query #42
#--------------------------

```python
def test_FXRateService_query():
    # Mock implementation of FXRateService for testing
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
    eur = Currency("EUR", "Euro", 978)
    usd = Currency("USD", "US Dollar", 840)
    today = Date.today()
    rate_eur_usd = FXRate(eur, usd, today, Decimal("1.20"))
    rate_usd_eur = FXRate(usd, eur, today, Decimal("0.8333"))

    # Initialize mock service with test data
    rates = {
        (eur, usd, today): rate_eur_usd,
        (usd, eur, today): rate_usd_eur
    }
    service = MockFXRateService(rates)

    # Test successful query
    result = service.query(eur, usd, today)
    assert result == rate_eur_usd

    # Test inverted rate query
    result = service.query(usd, eur, today)
    assert result == rate_usd_eur

    # Test non-existent rate query (non-strict)
    result = service.query(eur, usd, Date(2020, 1, 1))
    assert result is None

    # Test non-existent rate query (strict)
    with pytest.raises(FXRateLookupError):
        service.query(eur, usd, Date(2020, 1, 1), strict=True)


# LLM-generated content at query #43
#--------------------------

```python
def test_FXRateService_query():
    # Create a mock FXRateService implementation for testing
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

    # Setup test data
    from datetime import date
    from pypara.currencies import Currencies
    from decimal import Decimal

    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    today = date.today()
    rate = FXRate(eur, usd, today, Decimal("1.2"))

    # Initialize service with test data
    service = MockFXRateService({(eur, usd, today): rate})

    # Test successful query
    result = service.query(eur, usd, today)
    assert result == rate

    # Test query with non-existent rate (non-strict)
    result = service.query(usd, eur, today)
    assert result is None

    # Test query with non-existent rate (strict)
    with pytest.raises(FXRateLookupError):
        service.query(usd, eur, today, strict=True)

    # Test query with same currency
    same_currency_rate = FXRate(eur, eur, today, Decimal("1"))
    service_with_same_currency = MockFXRateService({(eur, eur, today): same_currency_rate})
    result = service_with_same_currency.query(eur, eur, today)
    assert result == same_currency_rate


# LLM-generated content at query #44
#--------------------------

```python
def test_FXRateService_query():
    # Mock FXRateService implementation for testing
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
    from datetime import date
    from pypara.currencies import Currencies
    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    today = date.today()
    rate = FXRate(eur, usd, today, Decimal("1.2"))

    # Test successful query
    service = MockFXRateService({(eur, usd, today): rate})
    result = service.query(eur, usd, today)
    assert result == rate

    # Test non-strict query with missing rate
    result = service.query(eur, usd, date(2020, 1, 1))
    assert result is None

    # Test strict query with missing rate
    with pytest.raises(FXRateLookupError):
        service.query(eur, usd, date(2020, 1, 1), strict=True)

    # Test query with inverted currencies
    inverted_rate = ~rate
    service = MockFXRateService({(usd, eur, today): inverted_rate})
    result = service.query(usd, eur, today)
    assert result == inverted_rate


# LLM-generated content at query #45
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
            return self.rates.get((ccy1, ccy2, asof))

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return (self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries)

    # Initialize the test service
    service = TestFXRateService()

    # Test successful query
    rate = service.query(Currency("EUR"), Currency("USD"), Date(2023, 1, 1))
    assert rate is not None
    assert rate.ccy1 == Currency("EUR")
    assert rate.ccy2 == Currency("USD")
    assert rate.date == Date(2023, 1, 1)
    assert rate.value == Decimal("1.1")

    # Test inverted query
    rate = service.query(Currency("USD"), Currency("EUR"), Date(2023, 1, 1))
    assert rate is not None
    assert rate.ccy1 == Currency("USD")
    assert rate.ccy2 == Currency("EUR")
    assert rate.date == Date(2023, 1, 1)
    assert rate.value == Decimal("0.909")

    # Test non-existent query with strict=False (should return None)
    rate = service.query(Currency("GBP"), Currency("JPY"), Date(2023, 1, 1), strict=False)
    assert rate is None

    # Test non-existent query with strict=True (should raise FXRateLookupError)
    with pytest.raises(FXRateLookupError):
        service.query(Currency("GBP"), Currency("JPY"), Date(2023, 1, 1), strict=True)


# LLM-generated content at query #46
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
    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    today = date.today()

    # Test successful query
    rates = {
        (eur, usd, today): FXRate(eur, usd, today, Decimal("1.2")),
    }
    service = TestFXRateService(rates)
    result = service.query(eur, usd, today)
    assert result == FXRate(eur, usd, today, Decimal("1.2"))

    # Test query with None result (non-strict)
    result = service.query(usd, eur, today)
    assert result is None

    # Test query with LookupError (strict)
    try:
        service.query(usd, eur, today, strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == usd
        assert e.ccy2 == eur
        assert e.asof == today


# LLM-generated content at query #47
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
        (usd, eur, today): FXRate(usd, eur, today, Decimal("0.8333")),
    }

    service = TestFXRateService(rates)

    # Test case 1: Normal query
    queries = [(eur, usd, today), (usd, eur, today)]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == FXRate(eur, usd, today, Decimal("1.2"))
    assert results[1] == FXRate(usd, eur, today, Decimal("0.8333"))

    # Test case 2: Query with missing rate (non-strict)
    queries = [(eur, usd, today), (eur, usd, date(2020, 1, 1))]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == FXRate(eur, usd, today, Decimal("1.2"))
    assert results[1] is None

    # Test case 3: Query with missing rate (strict)
    queries = [(eur, usd, date(2020, 1, 1))]
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))

    # Test case 4: Empty query
    assert list(service.queries([])) == []


# LLM-generated content at query #48
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
                rate = self.query(ccy1, ccy2, asof, strict)
                results.append(rate)
            return results

    # Test data
    from datetime import date
    from decimal import Decimal
    from pypara.currencies import Currencies

    EUR = Currencies["EUR"]
    USD = Currencies["USD"]
    GBP = Currencies["GBP"]

    rates = {
        (EUR, USD, date(2023, 1, 1)): FXRate(EUR, USD, date(2023, 1, 1), Decimal("1.1")),
        (USD, GBP, date(2023, 1, 1)): FXRate(USD, GBP, date(2023, 1, 1), Decimal("0.8")),
        (EUR, GBP, date(2023, 1, 2)): FXRate(EUR, GBP, date(2023, 1, 2), Decimal("0.88")),
    }

    service = MockFXRateService(rates)

    # Test case 1: All rates found
    queries1 = [
        (EUR, USD, date(2023, 1, 1)),
        (USD, GBP, date(2023, 1, 1)),
        (EUR, GBP, date(2023, 1, 2)),
    ]
    results1 = list(service.queries(queries1))
    assert len(results1) == 3
    assert results1[0] == rates[(EUR, USD, date(2023, 1, 1))]
    assert results1[1] == rates[(USD, GBP, date(2023, 1, 1))]
    assert results1[2] == rates[(EUR, GBP, date(2023, 1, 2))]

    # Test case 2: Some rates not found, strict=False
    queries2 = [
        (EUR, USD, date(2023, 1, 1)),
        (GBP, USD, date(2023, 1, 1)),  # Not in rates
        (EUR, GBP, date(2023, 1, 2)),
    ]
    results2 = list(service.queries(queries2, strict=False))
    assert len(results2) == 3
    assert results2[0] == rates[(EUR, USD, date(2023, 1, 1))]
    assert results2[1] is None
    assert results2[2] == rates[(EUR, GBP, date(2023, 1, 2))]

    # Test case 3: Some rates not found, strict=True
    queries3 = [
        (EUR, USD, date(2023, 1, 1)),
        (GBP, USD, date(2023, 1, 1)),  # Not in rates
    ]
    try:
        list(service.queries(queries3, strict=True))
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == GBP
        assert e.ccy2 == USD
        assert e.asof == date(2023, 1, 1)

    # Test case 4: Empty queries
    assert list(service.queries([])) == []


# LLM-generated content at query #49
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
    from datetime import date
    from pypara.currencies import Currencies
    from decimal import Decimal

    EUR = Currencies["EUR"]
    USD = Currencies["USD"]
    GBP = Currencies["GBP"]
    today = date.today()

    rates = {
        (EUR, USD, today): FXRate(EUR, USD, today, Decimal("1.2")),
        (USD, GBP, today): FXRate(USD, GBP, today, Decimal("0.8")),
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

    # Test 5: Multiple queries with same currency pair but different dates
    yesterday = date.today() - timedelta(days=1)
    rates[(EUR, USD, yesterday)] = FXRate(EUR, USD, yesterday, Decimal("1.1"))
    queries = [(EUR, USD, today), (EUR, USD, yesterday)]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == rates[(EUR, USD, today)]
    assert results[1] == rates[(EUR, USD, yesterday)]


# LLM-generated content at query #50
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
    test_rates = {
        (Currencies["EUR"], Currencies["USD"], date(2023, 1, 1)): FXRate(
            Currencies["EUR"], Currencies["USD"], date(2023, 1, 1), Decimal("1.1")
        ),
        (Currencies["USD"], Currencies["EUR"], date(2023, 1, 1)): FXRate(
            Currencies["USD"], Currencies["EUR"], date(2023, 1, 1), Decimal("0.909")
        ),
    }
    service = TestFXRateService(test_rates)

    # Test successful query
    result = service.query(Currencies["EUR"], Currencies["USD"], date(2023, 1, 1))
    assert result == test_rates[(Currencies["EUR"], Currencies["USD"], date(2023, 1, 1))]

    # Test query with None result (non-strict)
    result = service.query(Currencies["GBP"], Currencies["JPY"], date(2023, 1, 1))
    assert result is None

    # Test query with lookup error (strict)
    with pytest.raises(FXRateLookupError):
        service.query(Currencies["GBP"], Currencies["JPY"], date(2023, 1, 1), strict=True)

    # Test inverted rate
    eur_usd = service.query(Currencies["EUR"], Currencies["USD"], date(2023, 1, 1))
    usd_eur = service.query(Currencies["USD"], Currencies["EUR"], date(2023, 1, 1))
    assert ~eur_usd == usd_eur


# LLM-generated content at query #51
#--------------------------

```python
def test_FXRateService_query():
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

    # Test data
    from decimal import Decimal
    from pypara.currencies import Currencies
    from pypara.commons.zeitgeist import Date
    import datetime

    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    today = Date(datetime.date.today())
    yesterday = Date(datetime.date.today() - datetime.timedelta(days=1))

    rate1 = FXRate(eur, usd, today, Decimal("1.2"))
    rate2 = FXRate(usd, eur, today, Decimal("0.8333"))
    rate3 = FXRate(eur, usd, yesterday, Decimal("1.1"))

    rates = {
        (eur, usd, today): rate1,
        (usd, eur, today): rate2,
        (eur, usd, yesterday): rate3
    }

    service = MockFXRateService(rates)

    # Test successful query
    result = service.query(eur, usd, today)
    assert result == rate1

    # Test successful query with inverted currencies
    result = service.query(usd, eur, today)
    assert result == rate2

    # Test query with different date
    result = service.query(eur, usd, yesterday)
    assert result == rate3

    # Test query with non-existent rate (non-strict)
    result = service.query(eur, usd, Date(datetime.date(2020, 1, 1)))
    assert result is None

    # Test query with non-existent rate (strict)
    try:
        service.query(eur, usd, Date(datetime.date(2020, 1, 1)), strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == eur
        assert e.ccy2 == usd
        assert e.asof == Date(datetime.date(2020, 1, 1))

    # Test query with same currency (should return 1)
    result = service.query(eur, eur, today)
    assert result.value == Decimal("1")


# LLM-generated content at query #52
#--------------------------

```python
def test_FXRateService_query():
    # Create a mock FXRateService implementation for testing
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

    # Setup test data
    from datetime import date
    from pypara.currencies import Currencies
    from decimal import Decimal

    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    today = date.today()

    test_rates = {
        (eur, usd, today): FXRate(eur, usd, today, Decimal("1.20")),
        (usd, eur, today): FXRate(usd, eur, today, Decimal("0.83")),
    }

    service = MockFXRateService(test_rates)

    # Test successful query
    result = service.query(eur, usd, today)
    assert result == FXRate(eur, usd, today, Decimal("1.20"))

    # Test query with None result (non-strict)
    result = service.query(eur, usd, date(2020, 1, 1))
    assert result is None

    # Test query with strict=True raises exception
    with pytest.raises(FXRateLookupError):
        service.query(eur, usd, date(2020, 1, 1), strict=True)

    # Test inverted rate
    result = service.query(usd, eur, today)
    assert result == FXRate(usd, eur, today, Decimal("0.83"))


# LLM-generated content at query #53
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
        (Currencies["USD"], Currencies["JPY"], date(2023, 1, 1)): FXRate(Currencies["USD"], Currencies["JPY"], date(2023, 1, 1), Decimal("130.5")),
    }
    service = TestFXRateService(test_rates)

    # Test successful query
    result = service.query(Currencies["EUR"], Currencies["USD"], date(2023, 1, 1))
    assert result == FXRate(Currencies["EUR"], Currencies["USD"], date(2023, 1, 1), Decimal("1.1"))

    # Test query with non-existent rate (non-strict)
    result = service.query(Currencies["GBP"], Currencies["USD"], date(2023, 1, 1))
    assert result is None

    # Test query with non-existent rate (strict)
    with pytest.raises(FXRateLookupError):
        service.query(Currencies["GBP"], Currencies["USD"], date(2023, 1, 1), strict=True)

    # Test inverted rate query
    result = service.query(Currencies["USD"], Currencies["EUR"], date(2023, 1, 1))
    assert result is None  # Since we don't have this specific rate in our test data

    # Test query with different date
    result = service.query(Currencies["EUR"], Currencies["USD"], date(2022, 1, 1))
    assert result is None


# LLM-generated content at query #54
#--------------------------

```python
def test_FXRateService_queries():
    # Setup
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            if ccy1 == Currency("EUR") and ccy2 == Currency("USD") and asof == Date(2023, 1, 1):
                return FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1"))
            elif ccy1 == Currency("USD") and ccy2 == Currency("EUR") and asof == Date(2023, 1, 1):
                return FXRate(Currency("USD"), Currency("EUR"), Date(2023, 1, 1), Decimal("0.9"))
            elif strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None

        def queries(self, queries: Iterable[Tuple[Currency, Currency, Date]], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = MockFXRateService()
    queries = [
        (Currency("EUR"), Currency("USD"), Date(2023, 1, 1)),
        (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)),
        (Currency("GBP"), Currency("JPY"), Date(2023, 1, 1))
    ]

    # Test non-strict mode
    results = list(service.queries(queries, strict=False))
    assert len(results) == 3
    assert results[0] == FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1"))
    assert results[1] == FXRate(Currency("USD"), Currency("EUR"), Date(2023, 1, 1), Decimal("0.9"))
    assert results[2] is None

    # Test strict mode
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))


# LLM-generated content at query #55
#--------------------------

```python
def test_FXRateService_query():
    class TestFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            if ccy1 == Currency("EUR") and ccy2 == Currency("USD") and asof == Date(2023, 1, 1):
                return FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1"))
            return None

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = TestFXRateService()

    # Test successful query
    result = service.query(Currency("EUR"), Currency("USD"), Date(2023, 1, 1))
    assert result == FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1"))

    # Test non-existent rate with strict=False
    result = service.query(Currency("USD"), Currency("JPY"), Date(2023, 1, 1), strict=False)
    assert result is None

    # Test non-existent rate with strict=True
    with pytest.raises(FXRateLookupError):
        service.query(Currency("USD"), Currency("JPY"), Date(2023, 1, 1), strict=True)


# LLM-generated content at query #56
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
                rate = self.query(ccy1, ccy2, asof, strict)
                results.append(rate)
            return results

    # Create test data
    eur = Currency("EUR", "Euro", 2)
    usd = Currency("USD", "US Dollar", 2)
    today = Date.today()
    yesterday = today - Date.resolution

    rate1 = FXRate(eur, usd, today, Decimal("1.2"))
    rate2 = FXRate(usd, eur, yesterday, Decimal("0.8"))

    # Initialize the mock service
    mock_service = MockFXRateService([rate1, rate2])

    # Test case 1: Normal queries
    queries = [(eur, usd, today), (usd, eur, yesterday)]
    results = list(mock_service.queries(queries))
    assert len(results) == 2
    assert results[0] == rate1
    assert results[1] == rate2

    # Test case 2: Query with missing rate (non-strict)
    queries = [(eur, usd, today), (eur, usd, yesterday)]
    results = list(mock_service.queries(queries))
    assert len(results) == 2
    assert results[0] == rate1
    assert results[1] is None

    # Test case 3: Query with missing rate (strict)
    queries = [(eur, usd, today), (eur, usd, yesterday)]
    with pytest.raises(FXRateLookupError):
        list(mock_service.queries(queries, strict=True))

    # Test case 4: Empty queries
    results = list(mock_service.queries([]))
    assert len(results) == 0


# LLM-generated content at query #57
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
    today = Date.today()
    yesterday = today - datetime.timedelta(days=1)

    rate1 = FXRate(eur, usd, today, Decimal("1.20"))
    rate2 = FXRate(usd, eur, yesterday, Decimal("0.80"))

    rates = {
        (eur, usd, today): rate1,
        (usd, eur, yesterday): rate2
    }

    service = MockFXRateService(rates)

    # Test 1: Basic query
    queries = [(eur, usd, today), (usd, eur, yesterday)]
    results = service.queries(queries)
    assert list(results) == [rate1, rate2]

    # Test 2: Non-existent rate (non-strict)
    queries = [(eur, usd, today), (eur, usd, yesterday)]
    results = service.queries(queries)
    assert list(results) == [rate1, None]

    # Test 3: Non-existent rate (strict)
    queries = [(eur, usd, today), (eur, usd, yesterday)]
    with pytest.raises(FXRateLookupError):
        service.queries(queries, strict=True)

    # Test 4: Empty queries
    assert list(service.queries([])) == []

    # Test 5: Inverted rates
    queries = [(usd, eur, today)]
    results = service.queries(queries)
    assert list(results) == [None]  # Not in our mock data


# LLM-generated content at query #58
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

    rates = {
        (eur, usd, today): FXRate(eur, usd, today, Decimal("1.2")),
        (usd, eur, today): FXRate(usd, eur, today, Decimal("0.8333")),
    }

    service = TestFXRateService(rates)

    # Test successful query
    result = service.query(eur, usd, today)
    assert result == FXRate(eur, usd, today, Decimal("1.2"))

    # Test inverted rate
    result = service.query(usd, eur, today)
    assert result == FXRate(usd, eur, today, Decimal("0.8333"))

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

    # Test same currency
    result = service.query(eur, eur, today)
    assert result == FXRate(eur, eur, today, Decimal("1"))


# LLM-generated content at query #59
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

    # Test cases
    test_rates = {
        (eur, usd, today): FXRate(eur, usd, today, Decimal("1.2")),
        (usd, eur, today): FXRate(usd, eur, today, Decimal("0.8")),
    }

    service = TestFXRateService(test_rates)

    # Test successful query
    result = service.query(eur, usd, today)
    assert result == FXRate(eur, usd, today, Decimal("1.2"))

    # Test inverted rate
    result = service.query(usd, eur, today)
    assert result == FXRate(usd, eur, today, Decimal("0.8"))

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


# LLM-generated content at query #60
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
    test_rates = {
        (Currencies["EUR"], Currencies["USD"], date(2023, 1, 1)): FXRate(
            Currencies["EUR"], Currencies["USD"], date(2023, 1, 1), Decimal("1.1")
        ),
        (Currencies["USD"], Currencies["EUR"], date(2023, 1, 1)): FXRate(
            Currencies["USD"], Currencies["EUR"], date(2023, 1, 1), Decimal("0.909")
        ),
    }
    service = TestFXRateService(test_rates)

    # Test successful query
    result = service.query(Currencies["EUR"], Currencies["USD"], date(2023, 1, 1))
    assert result == FXRate(Currencies["EUR"], Currencies["USD"], date(2023, 1, 1), Decimal("1.1"))

    # Test query with None result (non-strict)
    result = service.query(Currencies["GBP"], Currencies["USD"], date(2023, 1, 1))
    assert result is None

    # Test query with lookup error (strict)
    with pytest.raises(FXRateLookupError):
        service.query(Currencies["GBP"], Currencies["USD"], date(2023, 1, 1), strict=True)

    # Test inverted rate
    result = service.query(Currencies["USD"], Currencies["EUR"], date(2023, 1, 1))
    assert result == ~test_rates[(Currencies["EUR"], Currencies["USD"], date(2023, 1, 1))]


# LLM-generated content at query #61
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
            results = []
            for query in queries:
                result = self.query(*query, strict=strict)
                results.append(result)
            return results

    # Test data setup
    from pypara.currencies import Currencies
    import datetime
    from decimal import Decimal

    EUR = Currencies["EUR"]
    USD = Currencies["USD"]
    today = datetime.date.today()
    rate = FXRate(EUR, USD, today, Decimal("1.2"))

    # Test case 1: Successful query
    service = MockFXRateService({(EUR, USD, today): rate})
    result = service.query(EUR, USD, today)
    assert result == rate

    # Test case 2: Query returns None when rate not found and strict=False
    result = service.query(USD, EUR, today)
    assert result is None

    # Test case 3: Query raises FXRateLookupError when rate not found and strict=True
    with pytest.raises(FXRateLookupError):
        service.query(USD, EUR, today, strict=True)

    # Test case 4: Query with different date
    yesterday = today - datetime.timedelta(days=1)
    result = service.query(EUR, USD, yesterday)
    assert result is None


# LLM-generated content at query #62
#--------------------------

```python
def test_FXRateService_queries():
    # Create a mock FXRateService implementation
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            # Mock implementation for testing
            if ccy1.code == "EUR" and ccy2.code == "USD" and asof == Date(2023, 1, 1):
                return FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1"))
            elif ccy1.code == "USD" and ccy2.code == "EUR" and asof == Date(2023, 1, 1):
                return FXRate(Currency("USD"), Currency("EUR"), Date(2023, 1, 1), Decimal("0.909"))
            elif strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            else:
                return None

        def queries(self, queries: Iterable[Tuple[Currency, Currency, Date]], strict: bool = False) -> Iterable[Optional[FXRate]]:
            # Mock implementation for testing
            results = []
            for ccy1, ccy2, asof in queries:
                try:
                    rate = self.query(ccy1, ccy2, asof, strict)
                    results.append(rate)
                except FXRateLookupError:
                    results.append(None)
            return results

    # Initialize the mock service
    service = MockFXRateService()

    # Test case 1: Valid queries
    queries = [
        (Currency("EUR"), Currency("USD"), Date(2023, 1, 1)),
        (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)),
        (Currency("GBP"), Currency("USD"), Date(2023, 1, 1))
    ]
    results = list(service.queries(queries))
    assert len(results) == 3
    assert results[0] == FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1"))
    assert results[1] == FXRate(Currency("USD"), Currency("EUR"), Date(2023, 1, 1), Decimal("0.909"))
    assert results[2] is None

    # Test case 2: Strict mode with invalid query
    queries = [(Currency("EUR"), Currency("JPY"), Date(2023, 1, 1))]
    results = list(service.queries(queries, strict=True))
    assert len(results) == 1
    assert results[0] is None

    # Test case 3: Empty queries
    queries = []
    results = list(service.queries(queries))
    assert len(results) == 0


# LLM-generated content at query #63
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

    # Create test data
    from datetime import date
    from decimal import Decimal
    from pypara.currencies import Currencies

    EUR = Currencies["EUR"]
    USD = Currencies["USD"]
    GBP = Currencies["GBP"]

    test_rates = {
        (EUR, USD, date(2023, 1, 1)): FXRate(EUR, USD, date(2023, 1, 1), Decimal("1.10")),
        (USD, GBP, date(2023, 1, 1)): FXRate(USD, GBP, date(2023, 1, 1), Decimal("0.80")),
        (EUR, GBP, date(2023, 1, 2)): FXRate(EUR, GBP, date(2023, 1, 2), Decimal("0.88")),
    }

    service = MockFXRateService(test_rates)

    # Test 1: Basic queries
    queries = [
        (EUR, USD, date(2023, 1, 1)),
        (USD, GBP, date(2023, 1, 1)),
        (EUR, GBP, date(2023, 1, 2)),
    ]
    results = list(service.queries(queries))
    assert len(results) == 3
    assert results[0] == test_rates[(EUR, USD, date(2023, 1, 1))]
    assert results[1] == test_rates[(USD, GBP, date(2023, 1, 1))]
    assert results[2] == test_rates[(EUR, GBP, date(2023, 1, 2))]

    # Test 2: Query with missing rate (non-strict)
    queries_with_missing = [
        (EUR, USD, date(2023, 1, 1)),
        (GBP, USD, date(2023, 1, 1)),  # Missing
    ]
    results = list(service.queries(queries_with_missing))
    assert len(results) == 2
    assert results[0] == test_rates[(EUR, USD, date(2023, 1, 1))]
    assert results[1] is None

    # Test 3: Query with missing rate (strict)
    queries_with_missing_strict = [
        (EUR, USD, date(2023, 1, 1)),
        (GBP, USD, date(2023, 1, 1)),  # Missing
    ]
    try:
        list(service.queries(queries_with_missing_strict, strict=True))
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == GBP
        assert e.ccy2 == USD
        assert e.asof == date(2023, 1, 1)

    # Test 4: Empty queries
    assert list(service.queries([])) == []


# LLM-generated content at query #64
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

    # Test setup
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currencies

    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    today = date.today()
    rate = FXRate(eur, usd, today, Decimal("1.2"))

    service = TestFXRateService({(eur, usd, today): rate})

    # Test successful query
    result = service.query(eur, usd, today)
    assert result == rate

    # Test query with non-existent rate (non-strict)
    result = service.query(usd, eur, today)
    assert result is None

    # Test query with non-existent rate (strict)
    with pytest.raises(FXRateLookupError):
        service.query(usd, eur, today, strict=True)

    # Test query with inverted currencies
    inverted_rate = ~rate
    service_with_inverted = TestFXRateService({(usd, eur, today): inverted_rate})
    result = service_with_inverted.query(usd, eur, today)
    assert result == inverted_rate


# LLM-generated content at query #65
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
        (gbp, eur, today): FXRate(gbp, eur, today, Decimal("1.1"))
    }

    service = MockFXRateService(rates)

    # Test successful queries
    queries = [
        (eur, usd, today),
        (usd, gbp, today),
        (gbp, eur, today)
    ]
    results = list(service.queries(queries))
    assert len(results) == 3
    assert results[0] == rates[(eur, usd, today)]
    assert results[1] == rates[(usd, gbp, today)]
    assert results[2] == rates[(gbp, eur, today)]

    # Test with missing rate (non-strict)
    queries_with_missing = [
        (eur, usd, today),
        (eur, gbp, today),  # This one is missing
        (gbp, eur, today)
    ]
    results = list(service.queries(queries_with_missing))
    assert len(results) == 3
    assert results[0] == rates[(eur, usd, today)]
    assert results[1] is None
    assert results[2] == rates[(gbp, eur, today)]

    # Test with missing rate (strict)
    try:
        list(service.queries(queries_with_missing, strict=True))
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == eur
        assert e.ccy2 == gbp
        assert e.asof == today

    # Test empty queries
    assert list(service.queries([])) == []


# LLM-generated content at query #66
#--------------------------

```python
def test_FXRateService_queries():
    # Mock FXRateService implementation for testing
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
            return (self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries)

    # Test data
    eur = Currency("EUR")
    usd = Currency("USD")
    date1 = Date(2023, 1, 1)
    date2 = Date(2023, 1, 2)

    # Test successful queries
    queries = [
        (eur, usd, date1),
        (usd, eur, date1),
    ]
    service = MockFXRateService()
    results = list(service.queries(queries))

    assert len(results) == 2
    assert results[0] == FXRate(eur, usd, date1, Decimal("1.1"))
    assert results[1] == FXRate(usd, eur, date1, Decimal("0.9"))

    # Test with missing rate (non-strict)
    queries_with_missing = [
        (eur, usd, date1),
        (eur, usd, date2),
    ]
    results_with_missing = list(service.queries(queries_with_missing, strict=False))

    assert len(results_with_missing) == 2
    assert results_with_missing[0] == FXRate(eur, usd, date1, Decimal("1.1"))
    assert results_with_missing[1] is None

    # Test with missing rate (strict)
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries_with_missing, strict=True))


# LLM-generated content at query #67
#--------------------------

```python
def test_FXRateService_query():
    # Create a mock FXRateService implementation for testing
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
    from datetime import date
    from pypara.currencies import Currencies
    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    today = date.today()

    # Create mock service with some rates
    rates = {
        (eur, usd, today): FXRate(eur, usd, today, Decimal("1.2")),
        (usd, eur, today): FXRate(usd, eur, today, Decimal("0.8333")),
    }
    service = MockFXRateService(rates)

    # Test successful query
    result = service.query(eur, usd, today)
    assert result == FXRate(eur, usd, today, Decimal("1.2"))

    # Test query with None result (non-strict)
    result = service.query(eur, usd, date(2000, 1, 1))
    assert result is None

    # Test query with strict=True raising exception
    with pytest.raises(FXRateLookupError):
        service.query(eur, usd, date(2000, 1, 1), strict=True)

    # Test inverted rate
    result = service.query(usd, eur, today)
    assert result == FXRate(usd, eur, today, Decimal("0.8333"))


# LLM-generated content at query #68
#--------------------------

```python
def test_FXRateService_query():
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
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    # Setup test data
    from datetime import date
    from pypara.currencies import Currencies
    test_rates = {
        (Currencies["EUR"], Currencies["USD"], date(2023, 1, 1)): FXRate(Currencies["EUR"], Currencies["USD"], date(2023, 1, 1), Decimal("1.10")),
        (Currencies["USD"], Currencies["JPY"], date(2023, 1, 1)): FXRate(Currencies["USD"], Currencies["JPY"], date(2023, 1, 1), Decimal("130.50")),
    }
    service = TestFXRateService(test_rates)

    # Test successful query
    result = service.query(Currencies["EUR"], Currencies["USD"], date(2023, 1, 1))
    assert result == FXRate(Currencies["EUR"], Currencies["USD"], date(2023, 1, 1), Decimal("1.10"))

    # Test query with non-existent rate (non-strict)
    result = service.query(Currencies["GBP"], Currencies["USD"], date(2023, 1, 1))
    assert result is None

    # Test query with non-existent rate (strict)
    with pytest.raises(FXRateLookupError):
        service.query(Currencies["GBP"], Currencies["USD"], date(2023, 1, 1), strict=True)

    # Test query with inverted rate
    result = service.query(Currencies["USD"], Currencies["EUR"], date(2023, 1, 1))
    assert result is None  # Since we don't have this specific rate in our test data


# LLM-generated content at query #69
#--------------------------

```python
def test_FXRateService_queries():
    # Create a mock FXRateService subclass for testing
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            # Mock implementation that returns a fixed rate for EUR/USD
            if ccy1.code == "EUR" and ccy2.code == "USD":
                return FXRate(ccy1, ccy2, asof, Decimal("1.2"))
            return None

        def queries(self, queries: Iterable[Tuple[Currency, Currency, Date]], strict: bool = False) -> Iterable[Optional[FXRate]]:
            # Mock implementation that processes queries
            results = []
            for ccy1, ccy2, asof in queries:
                result = self.query(ccy1, ccy2, asof)
                if result is None and strict:
                    raise FXRateLookupError(ccy1, ccy2, asof)
                results.append(result)
            return results

    # Create an instance of the mock service
    service = MockFXRateService()

    # Create test currencies and date
    eur = Currency("EUR", "Euro", 2)
    usd = Currency("USD", "US Dollar", 2)
    today = Date.today()

    # Test case 1: Single query
    queries = [(eur, usd, today)]
    results = list(service.queries(queries))
    assert len(results) == 1
    assert results[0] == FXRate(eur, usd, today, Decimal("1.2"))

    # Test case 2: Multiple queries
    queries = [(eur, usd, today), (usd, eur, today)]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == FXRate(eur, usd, today, Decimal("1.2"))
    assert results[1] is None

    # Test case 3: Strict mode with missing rate
    queries = [(eur, usd, today), (usd, eur, today)]
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))

    # Test case 4: Empty queries
    queries = []
    results = list(service.queries(queries))
    assert len(results) == 0


# LLM-generated content at query #70
#--------------------------

```python
def test_FXRateService_query():
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

    # Create mock currencies and dates
    ccy1 = Currency("EUR")
    ccy2 = Currency("USD")
    date1 = Date(2023, 1, 1)
    date2 = Date(2023, 1, 2)

    # Create mock rates
    rate1 = FXRate(ccy1, ccy2, date1, Decimal("1.1"))
    rate2 = FXRate(ccy2, ccy1, date2, Decimal("0.9"))

    # Initialize service with mock rates
    service = MockFXRateService({(ccy1, ccy2, date1): rate1, (ccy2, ccy1, date2): rate2})

    # Test successful query
    result = service.query(ccy1, ccy2, date1)
    assert result == rate1

    # Test query with inverted currencies
    result = service.query(ccy2, ccy1, date2)
    assert result == rate2

    # Test query with non-existent rate (non-strict)
    result = service.query(ccy1, ccy2, date2)
    assert result is None

    # Test query with non-existent rate (strict)
    with pytest.raises(FXRateLookupError):
        service.query(ccy1, ccy2, date2, strict=True)


# LLM-generated content at query #71
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

    # Test data
    from datetime import date
    from pypara.currencies import Currencies
    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    today = date.today()

    # Create test service with some rates
    test_rates = {
        (eur, usd, today): FXRate(eur, usd, today, Decimal("1.2")),
        (usd, eur, today): FXRate(usd, eur, today, Decimal("0.8333")),
    }
    service = TestFXRateService(test_rates)

    # Test successful query
    result = service.query(eur, usd, today)
    assert result == FXRate(eur, usd, today, Decimal("1.2"))

    # Test query with None result (non-strict)
    result = service.query(usd, eur, date(2020, 1, 1))
    assert result is None

    # Test query with lookup error (strict)
    with pytest.raises(FXRateLookupError):
        service.query(usd, eur, date(2020, 1, 1), strict=True)

    # Test inverted rate
    result = service.query(usd, eur, today)
    assert result == ~FXRate(eur, usd, today, Decimal("1.2"))


# LLM-generated content at query #72
#--------------------------

```python
def test_FXRateService_query():
    # Create a mock FXRateService implementation for testing
    class MockFXRateService(FXRateService):
        def __init__(self):
            self.rates = {
                (Currency("EUR"), Currency("USD"), Date(2023, 1, 1)): FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.10")),
                (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)): FXRate(Currency("USD"), Currency("EUR"), Date(2023, 1, 1), Decimal("0.91")),
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

    # Initialize the mock service
    service = MockFXRateService()

    # Test successful query
    result = service.query(Currency("EUR"), Currency("USD"), Date(2023, 1, 1))
    assert result is not None
    assert result.ccy1 == Currency("EUR")
    assert result.ccy2 == Currency("USD")
    assert result.date == Date(2023, 1, 1)
    assert result.value == Decimal("1.10")

    # Test successful query with inverted currencies
    result = service.query(Currency("USD"), Currency("EUR"), Date(2023, 1, 1))
    assert result is not None
    assert result.ccy1 == Currency("USD")
    assert result.ccy2 == Currency("EUR")
    assert result.date == Date(2023, 1, 1)
    assert result.value == Decimal("0.91")

    # Test query with non-existent rate (non-strict)
    result = service.query(Currency("GBP"), Currency("JPY"), Date(2023, 1, 1))
    assert result is None

    # Test query with non-existent rate (strict)
    with pytest.raises(FXRateLookupError):
        service.query(Currency("GBP"), Currency("JPY"), Date(2023, 1, 1), strict=True)


# LLM-generated content at query #73
#--------------------------

```python
def test_FXRateService_query():
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
            for query in queries:
                yield self.query(*query, strict=strict)

    # Setup test data
    from datetime import date
    from pypara.currencies import Currencies
    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    today = date.today()
    rate = FXRate(eur, usd, today, Decimal("1.2"))

    # Test successful query
    service = TestFXRateService({(eur, usd, today): rate})
    result = service.query(eur, usd, today)
    assert result == rate

    # Test query with strict=True raises exception
    with pytest.raises(FXRateLookupError):
        service.query(usd, eur, today, strict=True)

    # Test query with strict=False returns None
    result = service.query(usd, eur, today, strict=False)
    assert result is None

    # Test inverted rate query
    inverted_rate = ~rate
    service_with_inverted = TestFXRateService({(usd, eur, today): inverted_rate})
    result = service_with_inverted.query(usd, eur, today)
    assert result == inverted_rate


# LLM-generated content at query #74
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
                rate = self.query(ccy1, ccy2, asof, strict)
                results.append(rate)
            return results

    # Create test data
    eur = Currency("EUR", "Euro", 2)
    usd = Currency("USD", "US Dollar", 2)
    today = Date.today()
    yesterday = today - Date.resolution

    rate1 = FXRate(eur, usd, today, Decimal("1.2"))
    rate2 = FXRate(usd, eur, yesterday, Decimal("0.8"))

    # Initialize the mock service with test rates
    service = MockFXRateService([rate1, rate2])

    # Test case 1: Normal queries
    queries = [(eur, usd, today), (usd, eur, yesterday)]
    results = service.queries(queries)
    assert len(results) == 2
    assert results[0] == rate1
    assert results[1] == rate2

    # Test case 2: Query with missing rate (non-strict)
    queries = [(eur, usd, today), (eur, usd, yesterday)]
    results = service.queries(queries)
    assert len(results) == 2
    assert results[0] == rate1
    assert results[1] is None

    # Test case 3: Query with missing rate (strict)
    queries = [(eur, usd, today), (eur, usd, yesterday)]
    with pytest.raises(FXRateLookupError):
        service.queries(queries, strict=True)

    # Test case 4: Empty queries
    results = service.queries([])
    assert len(results) == 0


# LLM-generated content at query #75
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
    eur = Currency("EUR", "Euro", 2)
    usd = Currency("USD", "US Dollar", 2)
    gbp = Currency("GBP", "British Pound", 2)
    date1 = Date(2023, 1, 1)
    date2 = Date(2023, 1, 2)

    rate1 = FXRate(eur, usd, date1, Decimal("1.1"))
    rate2 = FXRate(eur, gbp, date1, Decimal("0.9"))
    rate3 = FXRate(eur, usd, date2, Decimal("1.2"))

    # Initialize the test service
    service = TestFXRateService([rate1, rate2, rate3])

    # Test case 1: Multiple queries with existing rates
    queries = [(eur, usd, date1), (eur, gbp, date1), (eur, usd, date2)]
    results = list(service.queries(queries))
    assert len(results) == 3
    assert results[0] == rate1
    assert results[1] == rate2
    assert results[2] == rate3

    # Test case 2: Query with non-existing rate (non-strict mode)
    queries = [(eur, gbp, date2)]
    results = list(service.queries(queries))
    assert len(results) == 1
    assert results[0] is None

    # Test case 3: Query with non-existing rate (strict mode)
    queries = [(eur, gbp, date2)]
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))

    # Test case 4: Empty queries
    queries = []
    results = list(service.queries(queries))
    assert len(results) == 0

    # Test case 5: Mixed existing and non-existing rates (non-strict mode)
    queries = [(eur, usd, date1), (eur, gbp, date2), (eur, usd, date2)]
    results = list(service.queries(queries))
    assert len(results) == 3
    assert results[0] == rate1
    assert results[1] is None
    assert results[2] == rate3


# LLM-generated content at query #76
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
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currencies

    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    today = date.today()
    rate = FXRate(eur, usd, today, Decimal("1.2"))

    # Initialize mock service
    service = MockFXRateService({(eur, usd, today): rate})

    # Test successful query
    result = service.query(eur, usd, today)
    assert result == rate

    # Test non-strict query with missing rate
    result = service.query(usd, eur, today)
    assert result is None

    # Test strict query with missing rate
    try:
        service.query(usd, eur, today, strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == usd
        assert e.ccy2 == eur
        assert e.asof == today


# LLM-generated content at query #77
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
    from decimal import Decimal
    from pypara.currencies import Currencies
    import datetime

    date1 = datetime.date(2023, 1, 1)
    date2 = datetime.date(2023, 1, 2)
    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    gbp = Currencies["GBP"]

    rates = {
        (eur, usd, date1): FXRate(eur, usd, date1, Decimal("1.1")),
        (usd, gbp, date1): FXRate(usd, gbp, date1, Decimal("0.8")),
        (eur, gbp, date2): FXRate(eur, gbp, date2, Decimal("0.85")),
    }

    service = TestFXRateService(rates)

    # Test 1: Successful queries
    queries = [
        (eur, usd, date1),
        (usd, gbp, date1),
        (eur, gbp, date2),
    ]
    results = service.queries(queries)
    assert results[0] == FXRate(eur, usd, date1, Decimal("1.1"))
    assert results[1] == FXRate(usd, gbp, date1, Decimal("0.8"))
    assert results[2] == FXRate(eur, gbp, date2, Decimal("0.85"))

    # Test 2: Query with missing rate (non-strict)
    queries = [
        (eur, usd, date1),
        (gbp, usd, date1),  # Missing
    ]
    results = service.queries(queries)
    assert results[0] == FXRate(eur, usd, date1, Decimal("1.1"))
    assert results[1] is None

    # Test 3: Query with missing rate (strict)
    queries = [
        (eur, usd, date1),
        (gbp, usd, date1),  # Missing
    ]
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))

    # Test 4: Empty queries
    assert list(service.queries([])) == []


# LLM-generated content at query #78
#--------------------------

```python
def test_FXRateService_query():
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            if ccy1.code == "EUR" and ccy2.code == "USD" and asof == Date(2023, 1, 1):
                return FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1"))
            return None

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = MockFXRateService()

    # Test successful query
    rate = service.query(Currency("EUR"), Currency("USD"), Date(2023, 1, 1))
    assert rate is not None
    assert rate.ccy1.code == "EUR"
    assert rate.ccy2.code == "USD"
    assert rate.date == Date(2023, 1, 1)
    assert rate.value == Decimal("1.1")

    # Test non-strict query with missing rate
    rate = service.query(Currency("USD"), Currency("JPY"), Date(2023, 1, 1))
    assert rate is None

    # Test strict query with missing rate
    try:
        service.query(Currency("USD"), Currency("JPY"), Date(2023, 1, 1), strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1.code == "USD"
        assert e.ccy2.code == "JPY"
        assert e.asof == Date(2023, 1, 1)


# LLM-generated content at query #79
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
    from pypara.currencies import Currencies
    import datetime
    from decimal import Decimal

    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)

    test_rates = {
        (eur, usd, today): FXRate(eur, usd, today, Decimal("1.2")),
        (usd, eur, today): FXRate(usd, eur, today, Decimal("0.8333")),
        (eur, usd, yesterday): FXRate(eur, usd, yesterday, Decimal("1.18")),
    }

    service = TestFXRateService(test_rates)

    # Test successful query
    result = service.query(eur, usd, today)
    assert result == test_rates[(eur, usd, today)]

    # Test query with inversion
    result = service.query(usd, eur, today)
    assert result == test_rates[(usd, eur, today)]

    # Test query for different date
    result = service.query(eur, usd, yesterday)
    assert result == test_rates[(eur, usd, yesterday)]

    # Test non-existent rate without strict
    result = service.query(eur, usd, datetime.date(2000, 1, 1))
    assert result is None

    # Test non-existent rate with strict
    try:
        service.query(eur, usd, datetime.date(2000, 1, 1), strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == eur
        assert e.ccy2 == usd
        assert e.asof == datetime.date(2000, 1, 1)


# LLM-generated content at query #80
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
    from decimal import Decimal
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
    assert result == FXRate(eur, usd, today, Decimal("1.2"))

    # Test query with None result (non-strict)
    result = service.query(usd, eur, date(2020, 1, 1))
    assert result is None

    # Test query with lookup error (strict)
    with pytest.raises(FXRateLookupError):
        service.query(usd, eur, date(2020, 1, 1), strict=True)

    # Test inverted rate
    result = service.query(usd, eur, today)
    assert result == ~service.query(eur, usd, today)


# LLM-generated content at query #81
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
                result = self.query(ccy1, ccy2, asof, strict)
                results.append(result)
            return results

    # Create test data
    eur = Currency("EUR", "Euro", 978)
    usd = Currency("USD", "US Dollar", 840)
    date1 = Date(2023, 1, 1)
    date2 = Date(2023, 1, 2)

    rate1 = FXRate(eur, usd, date1, Decimal("1.10"))
    rate2 = FXRate(usd, eur, date2, Decimal("0.90"))

    # Initialize the test service
    service = TestFXRateService([rate1, rate2])

    # Test case 1: Normal queries
    queries = [
        (eur, usd, date1),
        (usd, eur, date2),
    ]
    results = service.queries(queries)
    assert len(results) == 2
    assert results[0] == rate1
    assert results[1] == rate2

    # Test case 2: Query with missing rate (non-strict)
    queries = [
        (eur, usd, date1),
        (eur, usd, date2),  # This rate is not in the service
    ]
    results = service.queries(queries)
    assert len(results) == 2
    assert results[0] == rate1
    assert results[1] is None

    # Test case 3: Query with missing rate (strict)
    queries = [
        (eur, usd, date1),
        (eur, usd, date2),  # This rate is not in the service
    ]
    try:
        service.queries(queries, strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == eur
        assert e.ccy2 == usd
        assert e.asof == date2

    # Test case 4: Empty queries
    results = service.queries([])
    assert len(results) == 0


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# LLM-generated content at query #1
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
                rate = self.query(ccy1, ccy2, asof, strict)
                results.append(rate)
            return results

    # Setup test data
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currencies

    EUR = Currencies["EUR"]
    USD = Currencies["USD"]
    GBP = Currencies["GBP"]

    today = date.today()
    yesterday = today - timedelta(days=1)

    rates = {
        (EUR, USD, today): FXRate(EUR, USD, today, Decimal("1.2")),
        (USD, GBP, today): FXRate(USD, GBP, today, Decimal("0.8")),
        (EUR, GBP, yesterday): FXRate(EUR, GBP, yesterday, Decimal("0.9")),
    }

    service = MockFXRateService(rates)

    # Test 1: Normal queries with existing rates
    queries = [
        (EUR, USD, today),
        (USD, GBP, today),
        (EUR, GBP, yesterday),
    ]
    results = service.queries(queries)
    assert len(results) == 3
    assert results[0] == FXRate(EUR, USD, today, Decimal("1.2"))
    assert results[1] == FXRate(USD, GBP, today, Decimal("0.8"))
    assert results[2] == FXRate(EUR, GBP, yesterday, Decimal("0.9"))

    # Test 2: Query with non-existing rate (non-strict mode)
    queries = [
        (EUR, USD, today),
        (GBP, USD, today),  # This one doesn't exist
    ]
    results = service.queries(queries)
    assert len(results) == 2
    assert results[0] == FXRate(EUR, USD, today, Decimal("1.2"))
    assert results[1] is None

    # Test 3: Query with non-existing rate (strict mode)
    queries = [
        (EUR, USD, today),
        (GBP, USD, today),  # This one doesn't exist
    ]
    with pytest.raises(FXRateLookupError):
        service.queries(queries, strict=True)

    # Test 4: Empty queries
    assert list(service.queries([])) == []


# LLM-generated content at query #2
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
    test_rates = {
        (Currencies["EUR"], Currencies["USD"], date(2023, 1, 1)): FXRate(Currencies["EUR"], Currencies["USD"], date(2023, 1, 1), Decimal("1.1")),
        (Currencies["USD"], Currencies["EUR"], date(2023, 1, 1)): FXRate(Currencies["USD"], Currencies["EUR"], date(2023, 1, 1), Decimal("0.909")),
    }
    service = TestFXRateService(test_rates)

    # Test successful query
    result = service.query(Currencies["EUR"], Currencies["USD"], date(2023, 1, 1))
    assert result == FXRate(Currencies["EUR"], Currencies["USD"], date(2023, 1, 1), Decimal("1.1"))

    # Test query with None result
    result = service.query(Currencies["GBP"], Currencies["USD"], date(2023, 1, 1))
    assert result is None

    # Test strict query with existing rate
    result = service.query(Currencies["USD"], Currencies["EUR"], date(2023, 1, 1), strict=True)
    assert result == FXRate(Currencies["USD"], Currencies["EUR"], date(2023, 1, 1), Decimal("0.909"))

    # Test strict query with non-existing rate
    try:
        service.query(Currencies["GBP"], Currencies["USD"], date(2023, 1, 1), strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == Currencies["GBP"]
        assert e.ccy2 == Currencies["USD"]
        assert e.asof == date(2023, 1, 1)


# LLM-generated content at query #3
#--------------------------

```python
def test_FXRate___invert__():
    from decimal import Decimal
    from pypara.currencies import Currencies
    import datetime

    # Test inversion of a normal FX rate
    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("2"))
    inverted_rate = ~rate
    assert inverted_rate == FXRate(Currencies["USD"], Currencies["EUR"], datetime.date.today(), Decimal("0.5"))

    # Test inversion of an already inverted FX rate
    rate = FXRate(Currencies["USD"], Currencies["EUR"], datetime.date.today(), Decimal("0.5"))
    inverted_rate = ~rate
    assert inverted_rate == FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("2"))

    # Test inversion of a rate with value 1 (same currency)
    rate = FXRate(Currencies["EUR"], Currencies["EUR"], datetime.date.today(), Decimal("1"))
    inverted_rate = ~rate
    assert inverted_rate == FXRate(Currencies["EUR"], Currencies["EUR"], datetime.date.today(), Decimal("1"))


# LLM-generated content at query #4
#--------------------------

```python
def test_FXRateService_query():
    # Setup
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
    eur = Currency("EUR", "Euro", 978)
    usd = Currency("USD", "US Dollar", 840)
    today = Date.today()
    rate = FXRate(eur, usd, today, Decimal("1.2"))
    rates = {(eur, usd, today): rate}

    # Test cases
    service = MockFXRateService(rates)

    # Test successful query
    result = service.query(eur, usd, today)
    assert result == rate

    # Test query with non-existent rate (non-strict)
    result = service.query(usd, eur, today)
    assert result is None

    # Test query with non-existent rate (strict)
    with pytest.raises(FXRateLookupError):
        service.query(usd, eur, today, strict=True)

    # Test query with inverted rate
    inverted_rate = ~rate
    rates[(usd, eur, today)] = inverted_rate
    result = service.query(usd, eur, today)
    assert result == inverted_rate


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
            for query in queries:
                yield self.query(*query, strict=strict)

    # Create test data
    from datetime import date
    from pypara.currencies import Currencies
    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    gbp = Currencies["GBP"]
    today = date.today()
    yesterday = today - timedelta(days=1)

    rates = {
        (eur, usd, today): FXRate(eur, usd, today, Decimal("1.2")),
        (usd, gbp, today): FXRate(usd, gbp, today, Decimal("0.8")),
        (eur, gbp, yesterday): FXRate(eur, gbp, yesterday, Decimal("0.9")),
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

    # Test query with inverted currency pair
    result = service.query(usd, eur, today)
    assert result is None

    # Test query with different date
    result = service.query(eur, gbp, yesterday)
    assert result == FXRate(eur, gbp, yesterday, Decimal("0.9"))


# LLM-generated content at query #6
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

    EUR = Currencies["EUR"]
    USD = Currencies["USD"]
    GBP = Currencies["GBP"]

    test_date = date(2023, 1, 1)
    test_rates = {
        (EUR, USD, test_date): FXRate(EUR, USD, test_date, Decimal("1.10")),
        (USD, EUR, test_date): FXRate(USD, EUR, test_date, Decimal("0.91")),
        (EUR, GBP, test_date): FXRate(EUR, GBP, test_date, Decimal("0.85")),
    }

    service = TestFXRateService(test_rates)

    # Test successful query
    result = service.query(EUR, USD, test_date)
    assert result == FXRate(EUR, USD, test_date, Decimal("1.10"))

    # Test inverted rate query
    result = service.query(USD, EUR, test_date)
    assert result == FXRate(USD, EUR, test_date, Decimal("0.91"))

    # Test non-existent rate with strict=False
    result = service.query(GBP, USD, test_date, strict=False)
    assert result is None

    # Test non-existent rate with strict=True
    try:
        service.query(GBP, USD, test_date, strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == GBP
        assert e.ccy2 == USD
        assert e.asof == test_date

    # Test different date
    different_date = date(2023, 1, 2)
    result = service.query(EUR, USD, different_date, strict=False)
    assert result is None

    try:
        service.query(EUR, USD, different_date, strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == EUR
        assert e.ccy2 == USD
        assert e.asof == different_date


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
    test_rates = {
        (Currencies["EUR"], Currencies["USD"], date(2023, 1, 1)): FXRate(Currencies["EUR"], Currencies["USD"], date(2023, 1, 1), Decimal("1.2")),
        (Currencies["USD"], Currencies["EUR"], date(2023, 1, 1)): FXRate(Currencies["USD"], Currencies["EUR"], date(2023, 1, 1), Decimal("0.8333")),
    }
    service = TestFXRateService(test_rates)

    # Test successful query
    result = service.query(Currencies["EUR"], Currencies["USD"], date(2023, 1, 1))
    assert result == FXRate(Currencies["EUR"], Currencies["USD"], date(2023, 1, 1), Decimal("1.2"))

    # Test query with None result (non-strict)
    result = service.query(Currencies["GBP"], Currencies["USD"], date(2023, 1, 1))
    assert result is None

    # Test query with strict=True raising exception
    with pytest.raises(FXRateLookupError):
        service.query(Currencies["GBP"], Currencies["USD"], date(2023, 1, 1), strict=True)

    # Test inverted rate
    result = service.query(Currencies["USD"], Currencies["EUR"], date(2023, 1, 1))
    assert result == FXRate(Currencies["USD"], Currencies["EUR"], date(2023, 1, 1), Decimal("0.8333"))


# LLM-generated content at query #8
#--------------------------

```python
def test_FXRateService_queries():
    # Create a mock FXRateService subclass for testing
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
    eur = Currency("EUR", "Euro", "€")
    usd = Currency("USD", "US Dollar", "$")
    today = Date.today()
    yesterday = today - Date.resolution()

    rate1 = FXRate(eur, usd, today, Decimal("1.20"))
    rate2 = FXRate(usd, eur, today, Decimal("0.83"))
    rate3 = FXRate(eur, usd, yesterday, Decimal("1.18"))

    # Initialize the mock service
    service = MockFXRateService([rate1, rate2, rate3])

    # Test normal queries
    queries = [
        (eur, usd, today),
        (usd, eur, today),
        (eur, usd, yesterday),
    ]
    results = list(service.queries(queries))
    assert results == [rate1, rate2, rate3]

    # Test with non-existent rate (non-strict)
    queries = [
        (eur, usd, today),
        (eur, usd, Date(2020, 1, 1)),  # Non-existent
    ]
    results = list(service.queries(queries))
    assert results == [rate1, None]

    # Test with non-existent rate (strict)
    queries = [
        (eur, usd, today),
        (eur, usd, Date(2020, 1, 1)),  # Non-existent
    ]
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))

    # Test empty queries
    assert list(service.queries([])) == []


# LLM-generated content at query #9
#--------------------------

```python
def test_FXRateService_queries():
    # Create a mock FXRateService implementation for testing
    class MockFXRateService(FXRateService):
        def __init__(self):
            self.rates = {
                (Currency("EUR"), Currency("USD"), Date(2023, 1, 1)): FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1")),
                (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)): FXRate(Currency("USD"), Currency("EUR"), Date(2023, 1, 1), Decimal("0.9")),
                (Currency("GBP"), Currency("USD"), Date(2023, 1, 1)): FXRate(Currency("GBP"), Currency("USD"), Date(2023, 1, 1), Decimal("1.3")),
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

    # Test case 1: Query existing rates
    queries = [
        (Currency("EUR"), Currency("USD"), Date(2023, 1, 1)),
        (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)),
        (Currency("GBP"), Currency("USD"), Date(2023, 1, 1)),
    ]
    results = list(service.queries(queries))
    assert len(results) == 3
    assert results[0] == FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1"))
    assert results[1] == FXRate(Currency("USD"), Currency("EUR"), Date(2023, 1, 1), Decimal("0.9"))
    assert results[2] == FXRate(Currency("GBP"), Currency("USD"), Date(2023, 1, 1), Decimal("1.3"))

    # Test case 2: Query non-existing rates without strict mode
    queries = [
        (Currency("EUR"), Currency("JPY"), Date(2023, 1, 1)),
        (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)),
    ]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] is None
    assert results[1] == FXRate(Currency("USD"), Currency("EUR"), Date(2023, 1, 1), Decimal("0.9"))

    # Test case 3: Query non-existing rates with strict mode
    queries = [
        (Currency("EUR"), Currency("JPY"), Date(2023, 1, 1)),
    ]
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))


# LLM-generated content at query #10
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

    # Setup test data
    from datetime import date
    from pypara.currencies import Currencies
    from decimal import Decimal

    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    today = date.today()
    rate = FXRate(eur, usd, today, Decimal("1.2"))

    # Test successful query
    service = MockFXRateService({(eur, usd, today): rate})
    result = service.query(eur, usd, today)
    assert result == rate

    # Test query with non-existent rate (non-strict)
    result = service.query(usd, eur, today)
    assert result is None

    # Test query with non-existent rate (strict)
    try:
        service.query(usd, eur, today, strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == usd
        assert e.ccy2 == eur
        assert e.asof == today

    # Test query with different date
    yesterday = date.today() - timedelta(days=1)
    result = service.query(eur, usd, yesterday)
    assert result is None


# LLM-generated content at query #11
#--------------------------

```python
def test_FXRateService_query():
    # Create a mock FXRateService implementation for testing
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            if ccy1.code == "EUR" and ccy2.code == "USD" and asof == Date(2023, 1, 1):
                return FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.2"))
            elif strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    # Initialize the mock service
    service = MockFXRateService()

    # Test successful query
    rate = service.query(Currency("EUR"), Currency("USD"), Date(2023, 1, 1))
    assert rate is not None
    assert rate.ccy1.code == "EUR"
    assert rate.ccy2.code == "USD"
    assert rate.date == Date(2023, 1, 1)
    assert rate.value == Decimal("1.2")

    # Test query with no result (non-strict)
    rate = service.query(Currency("USD"), Currency("EUR"), Date(2023, 1, 1))
    assert rate is None

    # Test query with no result (strict)
    with pytest.raises(FXRateLookupError):
        service.query(Currency("USD"), Currency("EUR"), Date(2023, 1, 1), strict=True)


# LLM-generated content at query #12
#--------------------------

```python
def test_FXRateService_query():
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

    # Test data
    eur = Currency("EUR", "Euro", 2)
    usd = Currency("USD", "US Dollar", 2)
    today = Date(2023, 1, 1)
    rate = FXRate(eur, usd, today, Decimal("1.2"))

    # Test successful query
    service = MockFXRateService({(eur, usd, today): rate})
    result = service.query(eur, usd, today)
    assert result == rate

    # Test query with None result (non-strict)
    service = MockFXRateService({})
    result = service.query(eur, usd, today)
    assert result is None

    # Test query with LookupError (strict)
    service = MockFXRateService({})
    with pytest.raises(FXRateLookupError):
        service.query(eur, usd, today, strict=True)

    # Test inverted currency query
    inverted_rate = FXRate(usd, eur, today, Decimal("0.833333333333333333333333333"))
    service = MockFXRateService({(usd, eur, today): inverted_rate})
    result = service.query(usd, eur, today)
    assert result == inverted_rate


# LLM-generated content at query #13
#--------------------------

```python
def test_FXRateService_query():
    # Create a mock FXRateService implementation for testing
    class MockFXRateService(FXRateService):
        def __init__(self):
            self.rates = {
                (Currency("EUR"), Currency("USD"), Date(2023, 1, 1)): FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.10")),
                (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)): FXRate(Currency("USD"), Currency("EUR"), Date(2023, 1, 1), Decimal("0.9091")),
            }

        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            return self.rates.get((ccy1, ccy2, asof))

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(*q) for q in queries]

    service = MockFXRateService()

    # Test successful query
    rate = service.query(Currency("EUR"), Currency("USD"), Date(2023, 1, 1))
    assert rate is not None
    assert rate.ccy1 == Currency("EUR")
    assert rate.ccy2 == Currency("USD")
    assert rate.date == Date(2023, 1, 1)
    assert rate.value == Decimal("1.10")

    # Test query with non-existent rate (non-strict mode)
    rate = service.query(Currency("GBP"), Currency("USD"), Date(2023, 1, 1))
    assert rate is None

    # Test query with non-existent rate (strict mode)
    try:
        service.query(Currency("GBP"), Currency("USD"), Date(2023, 1, 1), strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == Currency("GBP")
        assert e.ccy2 == Currency("USD")
        assert e.asof == Date(2023, 1, 1)

    # Test inverted rate
    rate = service.query(Currency("USD"), Currency("EUR"), Date(2023, 1, 1))
    assert rate is not None
    assert rate.ccy1 == Currency("USD")
    assert rate.ccy2 == Currency("EUR")
    assert rate.date == Date(2023, 1, 1)
    assert rate.value == Decimal("0.9091")


# LLM-generated content at query #14
#--------------------------

```python
def test_FXRateService_query():
    # Create a mock FXRateService implementation for testing
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
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currencies

    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    today = date.today()

    # Create a mock service with some rates
    rates = {
        (eur, usd, today): FXRate(eur, usd, today, Decimal("1.2")),
        (usd, eur, today): FXRate(usd, eur, today, Decimal("0.8333")),
    }
    service = MockFXRateService(rates)

    # Test successful query
    result = service.query(eur, usd, today)
    assert result == FXRate(eur, usd, today, Decimal("1.2"))

    # Test query with inversion
    result = service.query(usd, eur, today)
    assert result == FXRate(usd, eur, today, Decimal("0.8333"))

    # Test query with non-existent rate (non-strict)
    result = service.query(eur, usd, date(2020, 1, 1))
    assert result is None

    # Test query with non-existent rate (strict)
    try:
        service.query(eur, usd, date(2020, 1, 1), strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == eur
        assert e.ccy2 == usd
        assert e.asof == date(2020, 1, 1)


# LLM-generated content at query #15
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

    # Create test data
    from datetime import date
    from decimal import Decimal
    from pypara.currencies import Currencies

    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    today = date.today()

    rates = {
        (eur, usd, today): FXRate(eur, usd, today, Decimal("1.2")),
        (usd, eur, today): FXRate(usd, eur, today, Decimal("0.8333")),
    }

    # Initialize the mock service
    service = MockFXRateService(rates)

    # Test 1: Basic query
    queries = [(eur, usd, today), (usd, eur, today)]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == FXRate(eur, usd, today, Decimal("1.2"))
    assert results[1] == FXRate(usd, eur, today, Decimal("0.8333"))

    # Test 2: Query with missing rate (non-strict)
    queries = [(eur, usd, today), (eur, eur, today)]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == FXRate(eur, usd, today, Decimal("1.2"))
    assert results[1] is None

    # Test 3: Query with missing rate (strict)
    queries = [(eur, usd, today), (eur, eur, today)]
    try:
        list(service.queries(queries, strict=True))
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == eur
        assert e.ccy2 == eur
        assert e.asof == today

    # Test 4: Empty queries
    results = list(service.queries([]))
    assert len(results) == 0


# LLM-generated content at query #16
#--------------------------

```python
def test_FXRateService_queries():
    # Create a mock FXRateService implementation
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            if ccy1.code == "EUR" and ccy2.code == "USD" and asof == Date(2023, 1, 1):
                return FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1"))
            elif ccy1.code == "USD" and ccy2.code == "EUR" and asof == Date(2023, 1, 1):
                return FXRate(Currency("USD"), Currency("EUR"), Date(2023, 1, 1), Decimal("0.909"))
            elif strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None

        def queries(self, queries: Iterable[Tuple[Currency, Currency, Date]], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return (self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries)

    # Initialize the mock service
    service = MockFXRateService()

    # Test with valid queries
    queries = [
        (Currency("EUR"), Currency("USD"), Date(2023, 1, 1)),
        (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)),
        (Currency("GBP"), Currency("USD"), Date(2023, 1, 1))
    ]
    results = list(service.queries(queries))
    assert len(results) == 3
    assert results[0] == FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1"))
    assert results[1] == FXRate(Currency("USD"), Currency("EUR"), Date(2023, 1, 1), Decimal("0.909"))
    assert results[2] is None

    # Test with strict mode
    with pytest.raises(FXRateLookupError):
        list(service.queries([(Currency("GBP"), Currency("USD"), Date(2023, 1, 1))], strict=True))


# LLM-generated content at query #17
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

    # Test data setup
    from datetime import date
    from pypara.currencies import Currencies
    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    today = date.today()

    # Create test service with some rates
    test_rates = {
        (eur, usd, today): FXRate(eur, usd, today, Decimal("1.2")),
        (usd, eur, today): FXRate(usd, eur, today, Decimal("0.8333")),
    }
    service = TestFXRateService(test_rates)

    # Test successful query
    result = service.query(eur, usd, today)
    assert result == FXRate(eur, usd, today, Decimal("1.2"))

    # Test inverted rate query
    result = service.query(usd, eur, today)
    assert result == FXRate(usd, eur, today, Decimal("0.8333"))

    # Test non-strict query with missing rate
    result = service.query(eur, usd, date(2020, 1, 1))
    assert result is None

    # Test strict query with missing rate
    with pytest.raises(FXRateLookupError):
        service.query(eur, usd, date(2020, 1, 1), strict=True)

    # Test same currency query
    result = service.query(eur, eur, today)
    assert result == FXRate(eur, eur, today, Decimal("1"))


# LLM-generated content at query #18
#--------------------------

```python
def test_FXRateService_queries():
    # Create a mock FXRateService instance
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            if ccy1.code == "EUR" and ccy2.code == "USD" and asof == Date(2023, 1, 1):
                return FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1"))
            elif ccy1.code == "USD" and ccy2.code == "JPY" and asof == Date(2023, 1, 2):
                return FXRate(Currency("USD"), Currency("JPY"), Date(2023, 1, 2), Decimal("130.5"))
            else:
                return None

        def queries(self, queries: Iterable[Tuple[Currency, Currency, Date]], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    # Initialize the mock service
    service = MockFXRateService()

    # Test with valid queries
    queries = [
        (Currency("EUR"), Currency("USD"), Date(2023, 1, 1)),
        (Currency("USD"), Currency("JPY"), Date(2023, 1, 2)),
        (Currency("GBP"), Currency("USD"), Date(2023, 1, 3))
    ]

    results = list(service.queries(queries))
    assert len(results) == 3
    assert results[0] == FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1"))
    assert results[1] == FXRate(Currency("USD"), Currency("JPY"), Date(2023, 1, 2), Decimal("130.5"))
    assert results[2] is None

    # Test with strict mode
    try:
        list(service.queries(queries, strict=True))
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError:
        pass

    # Test with empty queries
    assert list(service.queries([])) == []


# LLM-generated content at query #19
#--------------------------

```python
def test_FXRateService_query():
    # Create a concrete implementation of FXRateService for testing
    class TestFXRateService(FXRateService):
        def __init__(self):
            self.rates = {
                (Currency("EUR"), Currency("USD"), Date(2023, 1, 1)): FXRate(
                    Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1")
                ),
                (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)): FXRate(
                    Currency("USD"), Currency("EUR"), Date(2023, 1, 1), Decimal("0.909")
                ),
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
    rate = service.query(Currency("GBP"), Currency("USD"), Date(2023, 1, 1))
    assert rate is None

    # Test non-existent rate with strict
    with pytest.raises(FXRateLookupError):
        service.query(Currency("GBP"), Currency("USD"), Date(2023, 1, 1), strict=True)


# LLM-generated content at query #20
#--------------------------

```python
def test_FXRateService_queries():
    # Create a mock FXRateService subclass for testing
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

    # Test data
    eur = Currency("EUR")
    usd = Currency("USD")
    today = Date.today()
    rate1 = FXRate(eur, usd, today, Decimal("1.2"))
    rate2 = FXRate(usd, eur, today, Decimal("0.8333"))

    # Test case 1: Single query with existing rate
    service = MockFXRateService([rate1])
    queries = [(eur, usd, today)]
    results = service.queries(queries)
    assert list(results) == [rate1]

    # Test case 2: Multiple queries with existing rates
    service = MockFXRateService([rate1, rate2])
    queries = [(eur, usd, today), (usd, eur, today)]
    results = service.queries(queries)
    assert list(results) == [rate1, rate2]

    # Test case 3: Query with non-existing rate (non-strict mode)
    service = MockFXRateService([rate1])
    queries = [(eur, usd, today), (usd, eur, today)]
    results = service.queries(queries)
    assert list(results) == [rate1, None]

    # Test case 4: Query with non-existing rate (strict mode)
    service = MockFXRateService([rate1])
    queries = [(eur, usd, today), (usd, eur, today)]
    with pytest.raises(FXRateLookupError):
        service.queries(queries, strict=True)

    # Test case 5: Empty queries
    service = MockFXRateService([rate1])
    queries = []
    results = service.queries(queries)
    assert list(results) == []


# LLM-generated content at query #21
#--------------------------

```python
def test_FXRateService_query():
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
            for query in queries:
                yield self.query(*query, strict=strict)

    # Test setup
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currencies

    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    today = date.today()

    # Create test rates
    test_rates = {
        (eur, usd, today): FXRate(eur, usd, today, Decimal("1.2")),
        (usd, eur, today): FXRate(usd, eur, today, Decimal("0.8")),
    }

    service = TestFXRateService(test_rates)

    # Test successful query
    result = service.query(eur, usd, today)
    assert result == FXRate(eur, usd, today, Decimal("1.2"))

    # Test query with None result (non-strict)
    result = service.query(eur, usd, date(2000, 1, 1))
    assert result is None

    # Test query with exception (strict)
    with pytest.raises(FXRateLookupError):
        service.query(eur, usd, date(2000, 1, 1), strict=True)

    # Test inverted rate
    result = service.query(usd, eur, today)
    assert result == FXRate(usd, eur, today, Decimal("0.8"))


# LLM-generated content at query #22
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
                rate = self.query(ccy1, ccy2, asof, strict)
                results.append(rate)
            return results

    # Create some test data
    eur = Currency("EUR", "Euro", 978)
    usd = Currency("USD", "US Dollar", 840)
    date1 = Date(2023, 1, 1)
    date2 = Date(2023, 1, 2)

    rates = [
        FXRate(eur, usd, date1, Decimal("1.10")),
        FXRate(eur, usd, date2, Decimal("1.12")),
        FXRate(usd, eur, date1, Decimal("0.9091")),
    ]

    # Create the mock service
    service = MockFXRateService(rates)

    # Test 1: Basic query
    queries = [(eur, usd, date1), (eur, usd, date2)]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == rates[0]
    assert results[1] == rates[1]

    # Test 2: Query with missing rate (non-strict)
    queries = [(eur, usd, date1), (usd, eur, date2)]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == rates[0]
    assert results[1] is None

    # Test 3: Query with missing rate (strict)
    queries = [(eur, usd, date1), (usd, eur, date2)]
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))

    # Test 4: Empty queries
    results = list(service.queries([]))
    assert len(results) == 0

    # Test 5: Query with inverted rate
    queries = [(usd, eur, date1)]
    results = list(service.queries(queries))
    assert len(results) == 1
    assert results[0] == rates[2]


# LLM-generated content at query #23
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

    # Initialize service with test rate
    service = MockFXRateService({(eur, usd, today): rate})

    # Test successful query
    result = service.query(eur, usd, today)
    assert result == rate

    # Test query with non-existent rate (non-strict)
    result = service.query(usd, eur, today)
    assert result is None

    # Test query with non-existent rate (strict)
    with pytest.raises(FXRateLookupError):
        service.query(usd, eur, today, strict=True)

    # Test query with different date
    yesterday = date.today() - timedelta(days=1)
    result = service.query(eur, usd, yesterday)
    assert result is None

    # Test query with inverted currencies
    inverted_rate = ~rate
    service_with_inverted = MockFXRateService({(usd, eur, today): inverted_rate})
    result = service_with_inverted.query(usd, eur, today)
    assert result == inverted_rate


# LLM-generated content at query #24
#--------------------------

```python
def test_FXRateService_query():
    # Create a mock FXRateService implementation for testing
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            # Mock implementation that returns a fixed rate
            if ccy1.code == "EUR" and ccy2.code == "USD":
                return FXRate(Currency("EUR"), Currency("USD"), asof, Decimal("1.2"))
            elif strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None

        def queries(self, queries: Iterable[Tuple[Currency, Currency, Date]], strict: bool = False) -> Iterable[Optional[FXRate]]:
            # Mock implementation for queries
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    # Initialize the service
    service = MockFXRateService()

    # Test successful query
    rate = service.query(Currency("EUR"), Currency("USD"), Date.today())
    assert rate is not None
    assert rate.ccy1.code == "EUR"
    assert rate.ccy2.code == "USD"
    assert rate.value == Decimal("1.2")

    # Test query with no result and no strict
    rate = service.query(Currency("USD"), Currency("JPY"), Date.today())
    assert rate is None

    # Test query with no result and strict
    with pytest.raises(FXRateLookupError):
        service.query(Currency("USD"), Currency("JPY"), Date.today(), strict=True)


# LLM-generated content at query #25
#--------------------------

```python
def test_FXRateService_query():
    # Create a mock FXRateService implementation for testing
    class MockFXRateService(FXRateService):
        def __init__(self):
            self.rates = {
                (Currency("EUR"), Currency("USD"), Date(2023, 1, 1)): FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.10")),
                (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)): FXRate(Currency("USD"), Currency("EUR"), Date(2023, 1, 1), Decimal("0.9091")),
            }

        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            return self.rates.get((ccy1, ccy2, asof))

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof) for ccy1, ccy2, asof in queries]

    # Test successful query
    service = MockFXRateService()
    rate = service.query(Currency("EUR"), Currency("USD"), Date(2023, 1, 1))
    assert rate is not None
    assert rate.ccy1 == Currency("EUR")
    assert rate.ccy2 == Currency("USD")
    assert rate.date == Date(2023, 1, 1)
    assert rate.value == Decimal("1.10")

    # Test query with non-existent rate (non-strict)
    rate = service.query(Currency("GBP"), Currency("JPY"), Date(2023, 1, 1))
    assert rate is None

    # Test query with non-existent rate (strict)
    try:
        service.query(Currency("GBP"), Currency("JPY"), Date(2023, 1, 1), strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == Currency("GBP")
        assert e.ccy2 == Currency("JPY")
        assert e.asof == Date(2023, 1, 1)


# LLM-generated content at query #26
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


# LLM-generated content at query #27
#--------------------------

```python
def test_FXRateService_queries():
    # Create a mock FXRateService implementation for testing
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            if ccy1.code == "EUR" and ccy2.code == "USD" and asof == Date(2023, 1, 1):
                return FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1"))
            elif ccy1.code == "USD" and ccy2.code == "GBP" and asof == Date(2023, 1, 2):
                return FXRate(Currency("USD"), Currency("GBP"), Date(2023, 1, 2), Decimal("0.8"))
            elif strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            else:
                return None

        def queries(self, queries: Iterable[Tuple[Currency, Currency, Date]], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return (self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries)

    # Create an instance of the mock service
    service = MockFXRateService()

    # Test case 1: Valid queries
    queries = [
        (Currency("EUR"), Currency("USD"), Date(2023, 1, 1)),
        (Currency("USD"), Currency("GBP"), Date(2023, 1, 2)),
    ]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1"))
    assert results[1] == FXRate(Currency("USD"), Currency("GBP"), Date(2023, 1, 2), Decimal("0.8"))

    # Test case 2: Query with no result (non-strict)
    queries = [
        (Currency("EUR"), Currency("JPY"), Date(2023, 1, 1)),
    ]
    results = list(service.queries(queries))
    assert len(results) == 1
    assert results[0] is None

    # Test case 3: Query with no result (strict)
    queries = [
        (Currency("EUR"), Currency("JPY"), Date(2023, 1, 1)),
    ]
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))

    # Test case 4: Mixed queries (some valid, some not)
    queries = [
        (Currency("EUR"), Currency("USD"), Date(2023, 1, 1)),
        (Currency("EUR"), Currency("JPY"), Date(2023, 1, 1)),
        (Currency("USD"), Currency("GBP"), Date(2023, 1, 2)),
    ]
    results = list(service.queries(queries))
    assert len(results) == 3
    assert results[0] == FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1"))
    assert results[1] is None
    assert results[2] == FXRate(Currency("USD"), Currency("GBP"), Date(2023, 1, 2), Decimal("0.8"))

    # Test case 5: Empty queries
    queries = []
    results = list(service.queries(queries))
    assert len(results) == 0


# LLM-generated content at query #28
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
    yesterday = today - timedelta(days=1)

    rates = {
        (eur, usd, today): FXRate(eur, usd, today, Decimal("1.2")),
        (usd, gbp, today): FXRate(usd, gbp, today, Decimal("0.8")),
        (eur, gbp, yesterday): FXRate(eur, gbp, yesterday, Decimal("0.9")),
    }

    service = TestFXRateService(rates)

    # Test successful queries
    queries = [
        (eur, usd, today),
        (usd, gbp, today),
        (eur, gbp, yesterday),
    ]
    results = list(service.queries(queries))
    assert len(results) == 3
    assert results[0] == rates[(eur, usd, today)]
    assert results[1] == rates[(usd, gbp, today)]
    assert results[2] == rates[(eur, gbp, yesterday)]

    # Test with missing rate (non-strict)
    queries_with_missing = [
        (eur, usd, today),
        (eur, gbp, today),  # This one is missing
    ]
    results = list(service.queries(queries_with_missing))
    assert len(results) == 2
    assert results[0] == rates[(eur, usd, today)]
    assert results[1] is None

    # Test with missing rate (strict)
    try:
        list(service.queries(queries_with_missing, strict=True))
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == eur
        assert e.ccy2 == gbp
        assert e.asof == today

    # Test empty queries
    assert list(service.queries([])) == []


# LLM-generated content at query #29
#--------------------------

```python
def test_FXRateService_queries():
    # Create a mock FXRateService
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            # Mock implementation for testing
            if ccy1.code == "EUR" and ccy2.code == "USD" and asof == Date(2023, 1, 1):
                return FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1"))
            elif ccy1.code == "USD" and ccy2.code == "EUR" and asof == Date(2023, 1, 1):
                return FXRate(Currency("USD"), Currency("EUR"), Date(2023, 1, 1), Decimal("0.9"))
            elif strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            else:
                return None

        def queries(self, queries: Iterable[Tuple[Currency, Currency, Date]], strict: bool = False) -> Iterable[Optional[FXRate]]:
            # Mock implementation for testing
            for query in queries:
                ccy1, ccy2, asof = query
                yield self.query(ccy1, ccy2, asof, strict)

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
    assert results[1] == FXRate(Currency("USD"), Currency("EUR"), Date(2023, 1, 1), Decimal("0.9"))

    # Test case 2: Invalid queries with strict=False
    queries = [
        (Currency("EUR"), Currency("GBP"), Date(2023, 1, 1)),
        (Currency("JPY"), Currency("USD"), Date(2023, 1, 1)),
    ]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] is None
    assert results[1] is None

    # Test case 3: Invalid queries with strict=True
    queries = [
        (Currency("EUR"), Currency("GBP"), Date(2023, 1, 1)),
        (Currency("JPY"), Currency("USD"), Date(2023, 1, 1)),
    ]
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))


# LLM-generated content at query #30
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
    from pypara.currencies import Currencies
    import datetime
    from decimal import Decimal

    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)

    test_rates = {
        (eur, usd, today): FXRate(eur, usd, today, Decimal("1.2")),
        (usd, eur, today): FXRate(usd, eur, today, Decimal("0.8333")),
        (eur, usd, yesterday): FXRate(eur, usd, yesterday, Decimal("1.19")),
    }

    service = TestFXRateService(test_rates)

    # Test successful query
    result = service.query(eur, usd, today)
    assert result == FXRate(eur, usd, today, Decimal("1.2"))

    # Test query with None result (non-strict)
    result = service.query(eur, usd, datetime.date(2000, 1, 1))
    assert result is None

    # Test query with lookup error (strict)
    with pytest.raises(FXRateLookupError):
        service.query(eur, usd, datetime.date(2000, 1, 1), strict=True)

    # Test inverted rate
    result = service.query(usd, eur, today)
    assert result == FXRate(usd, eur, today, Decimal("0.8333"))


# LLM-generated content at query #31
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
    rate = FXRate(eur, usd, today, Decimal("1.2"))

    # Test successful query
    service = TestFXRateService({(eur, usd, today): rate})
    result = service.query(eur, usd, today)
    assert result == rate

    # Test query with non-existent rate (non-strict)
    result = service.query(eur, usd, date(2020, 1, 1))
    assert result is None

    # Test query with non-existent rate (strict)
    with pytest.raises(FXRateLookupError):
        service.query(eur, usd, date(2020, 1, 1), strict=True)

    # Test query with inverted currencies
    inverted_rate = ~rate
    service = TestFXRateService({(usd, eur, today): inverted_rate})
    result = service.query(usd, eur, today)
    assert result == inverted_rate


# LLM-generated content at query #32
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
    from pypara.currencies import Currencies
    import datetime
    from decimal import Decimal

    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    today = datetime.date.today()
    rate = FXRate(eur, usd, today, Decimal("1.2"))

    # Test successful query
    service = TestFXRateService({(eur, usd, today): rate})
    result = service.query(eur, usd, today)
    assert result == rate

    # Test query with None result
    result = service.query(usd, eur, today)
    assert result is None

    # Test strict query with missing rate
    with pytest.raises(FXRateLookupError):
        service.query(usd, eur, today, strict=True)

    # Test inverted rate query
    inverted_rate = ~rate
    service_with_inverted = TestFXRateService({(usd, eur, today): inverted_rate})
    result = service_with_inverted.query(usd, eur, today)
    assert result == inverted_rate


# LLM-generated content at query #33
#--------------------------

```python
def test_FXRateService_query():
    # Mock implementation of FXRateService for testing
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

    # Setup test data
    eur = Currency("EUR", "Euro", "€")
    usd = Currency("USD", "US Dollar", "$")
    today = Date.today()
    yesterday = today - datetime.timedelta(days=1)

    rate_eur_usd_today = FXRate(eur, usd, today, Decimal("1.2"))
    rate_usd_eur_today = FXRate(usd, eur, today, Decimal("0.8333"))
    rate_eur_usd_yesterday = FXRate(eur, usd, yesterday, Decimal("1.1"))

    rates = {
        (eur, usd, today): rate_eur_usd_today,
        (usd, eur, today): rate_usd_eur_today,
        (eur, usd, yesterday): rate_eur_usd_yesterday,
    }

    service = MockFXRateService(rates)

    # Test successful query
    result = service.query(eur, usd, today)
    assert result == rate_eur_usd_today

    # Test inverted rate
    result = service.query(usd, eur, today)
    assert result == rate_usd_eur_today

    # Test different date
    result = service.query(eur, usd, yesterday)
    assert result == rate_eur_usd_yesterday

    # Test non-existent rate without strict
    result = service.query(eur, usd, Date(2020, 1, 1))
    assert result is None

    # Test non-existent rate with strict
    with pytest.raises(FXRateLookupError):
        service.query(eur, usd, Date(2020, 1, 1), strict=True)

    # Test same currency
    result = service.query(eur, eur, today)
    assert result.value == ONE


# LLM-generated content at query #34
#--------------------------

```python
def test_FXRateService_query():
    # Create a mock FXRateService implementation for testing
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

    # Test data setup
    from pypara.currencies import Currencies
    import datetime
    from decimal import Decimal

    EUR = Currencies["EUR"]
    USD = Currencies["USD"]
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)

    rates = {
        (EUR, USD, today): FXRate(EUR, USD, today, Decimal("1.2")),
        (USD, EUR, today): FXRate(USD, EUR, today, Decimal("0.8333")),
        (EUR, USD, yesterday): FXRate(EUR, USD, yesterday, Decimal("1.15")),
    }

    service = MockFXRateService(rates)

    # Test 1: Successful query
    result = service.query(EUR, USD, today)
    assert result == FXRate(EUR, USD, today, Decimal("1.2"))

    # Test 2: Query with inversion
    result = service.query(USD, EUR, today)
    assert result == FXRate(USD, EUR, today, Decimal("0.8333"))

    # Test 3: Query for different date
    result = service.query(EUR, USD, yesterday)
    assert result == FXRate(EUR, USD, yesterday, Decimal("1.15"))

    # Test 4: Non-existent rate without strict
    result = service.query(EUR, USD, datetime.date(2000, 1, 1))
    assert result is None

    # Test 5: Non-existent rate with strict
    try:
        service.query(EUR, USD, datetime.date(2000, 1, 1), strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == EUR
        assert e.ccy2 == USD
        assert e.asof == datetime.date(2000, 1, 1)

    # Test 6: Query with same currency
    try:
        service.query(EUR, EUR, today)
        assert False, "Expected ValueError for same currency"
    except ValueError:
        pass


# LLM-generated content at query #35
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
    from pypara.currencies import Currencies
    import datetime
    from decimal import Decimal

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

    service = TestFXRateService(rates)

    # Test 1: Normal queries with existing rates
    queries = [
        (EUR, USD, today),
        (USD, GBP, today),
        (EUR, GBP, yesterday),
    ]
    results = list(service.queries(queries))
    assert len(results) == 3
    assert results[0] == FXRate(EUR, USD, today, Decimal("1.2"))
    assert results[1] == FXRate(USD, GBP, today, Decimal("0.8"))
    assert results[2] == FXRate(EUR, GBP, yesterday, Decimal("0.9"))

    # Test 2: Query with non-existing rate (non-strict mode)
    queries = [
        (EUR, USD, today),
        (EUR, USD, yesterday),  # This one doesn't exist
    ]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == FXRate(EUR, USD, today, Decimal("1.2"))
    assert results[1] is None

    # Test 3: Query with non-existing rate (strict mode)
    queries = [
        (EUR, USD, today),
        (EUR, USD, yesterday),  # This one doesn't exist
    ]
    try:
        list(service.queries(queries, strict=True))
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == EUR
        assert e.ccy2 == USD
        assert e.asof == yesterday

    # Test 4: Empty queries
    assert list(service.queries([])) == []


# LLM-generated content at query #36
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
    eur = Currency("EUR", "Euro", 978)
    usd = Currency("USD", "US Dollar", 840)
    today = Date.today()
    yesterday = today - Date.resolution()

    rate1 = FXRate(eur, usd, today, Decimal("1.20"))
    rate2 = FXRate(usd, eur, yesterday, Decimal("0.80"))

    # Initialize the test service
    service = TestFXRateService([rate1, rate2])

    # Test case 1: Single query that exists
    queries = [(eur, usd, today)]
    results = list(service.queries(queries))
    assert len(results) == 1
    assert results[0] == rate1

    # Test case 2: Multiple queries that exist
    queries = [(eur, usd, today), (usd, eur, yesterday)]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == rate1
    assert results[1] == rate2

    # Test case 3: Query that does not exist (non-strict)
    queries = [(eur, usd, yesterday)]
    results = list(service.queries(queries))
    assert len(results) == 1
    assert results[0] is None

    # Test case 4: Query that does not exist (strict)
    queries = [(eur, usd, yesterday)]
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))

    # Test case 5: Empty queries
    queries = []
    results = list(service.queries(queries))
    assert len(results) == 0


# LLM-generated content at query #37
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
    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    today = date.today()

    # Test cases
    rates = {
        (eur, usd, today): FXRate(eur, usd, today, Decimal("1.2")),
        (usd, eur, today): FXRate(usd, eur, today, Decimal("0.8333")),
    }

    service = TestFXRateService(rates)

    # Test successful query
    result = service.query(eur, usd, today)
    assert result == FXRate(eur, usd, today, Decimal("1.2"))

    # Test successful inverted query
    result = service.query(usd, eur, today)
    assert result == FXRate(usd, eur, today, Decimal("0.8333"))

    # Test non-strict query with missing rate
    result = service.query(eur, usd, date(2020, 1, 1))
    assert result is None

    # Test strict query with missing rate
    with pytest.raises(FXRateLookupError):
        service.query(eur, usd, date(2020, 1, 1), strict=True)


# LLM-generated content at query #38
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
    with pytest.raises(FXRateLookupError):
        service.query(eur, usd, date(2020, 1, 1), strict=True)

    # Test same currency
    result = service.query(eur, eur, today)
    assert result is None  # Assuming no rate is stored for same currency

    # Test strict mode with same currency
    with pytest.raises(FXRateLookupError):
        service.query(eur, eur, today, strict=True)


# LLM-generated content at query #39
#--------------------------

```python
def test_FXRateService_query():
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
    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    today = date.today()
    rate = FXRate(eur, usd, today, Decimal("1.2"))

    # Test successful query
    service = TestFXRateService({(eur, usd, today): rate})
    result = service.query(eur, usd, today)
    assert result == rate

    # Test query with non-existent rate (non-strict)
    result = service.query(usd, eur, today)
    assert result is None

    # Test query with non-existent rate (strict)
    try:
        service.query(usd, eur, today, strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == usd
        assert e.ccy2 == eur
        assert e.asof == today


# LLM-generated content at query #40
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
    EUR = Currencies["EUR"]
    USD = Currencies["USD"]
    GBP = Currencies["GBP"]

    test_rates = {
        (EUR, USD, date(2023, 1, 1)): FXRate(EUR, USD, date(2023, 1, 1), Decimal("1.10")),
        (USD, EUR, date(2023, 1, 1)): FXRate(USD, EUR, date(2023, 1, 1), Decimal("0.9091")),
        (EUR, GBP, date(2023, 1, 1)): FXRate(EUR, GBP, date(2023, 1, 1), Decimal("0.85")),
    }

    service = TestFXRateService(test_rates)

    # Test 1: Successful query
    result = service.query(EUR, USD, date(2023, 1, 1))
    assert result == FXRate(EUR, USD, date(2023, 1, 1), Decimal("1.10"))

    # Test 2: Query with inversion
    result = service.query(USD, EUR, date(2023, 1, 1))
    assert result == FXRate(USD, EUR, date(2023, 1, 1), Decimal("0.9091"))

    # Test 3: Query with different currency pair
    result = service.query(EUR, GBP, date(2023, 1, 1))
    assert result == FXRate(EUR, GBP, date(2023, 1, 1), Decimal("0.85"))

    # Test 4: Query with non-existent rate (non-strict)
    result = service.query(EUR, GBP, date(2023, 1, 2))
    assert result is None

    # Test 5: Query with non-existent rate (strict)
    try:
        service.query(EUR, GBP, date(2023, 1, 2), strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == EUR
        assert e.ccy2 == GBP
        assert e.asof == date(2023, 1, 2)


# LLM-generated content at query #41
#--------------------------

```python
def test_FXRateService_queries():
    # Create a mock FXRateService implementation
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
    from datetime import date
    from pypara.currencies import Currencies
    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    today = date.today()
    rate1 = FXRate(eur, usd, today, Decimal("1.2"))
    rate2 = FXRate(usd, eur, today, Decimal("0.8333"))
    rates = [rate1, rate2]

    # Initialize service
    service = MockFXRateService(rates)

    # Test with existing rates
    queries = [(eur, usd, today), (usd, eur, today)]
    results = service.queries(queries)
    assert len(results) == 2
    assert results[0] == rate1
    assert results[1] == rate2

    # Test with non-existing rate (non-strict)
    queries = [(eur, usd, today), (eur, usd, date(2020, 1, 1))]
    results = service.queries(queries)
    assert len(results) == 2
    assert results[0] == rate1
    assert results[1] is None

    # Test with non-existing rate (strict)
    queries = [(eur, usd, date(2020, 1, 1))]
    with pytest.raises(FXRateLookupError):
        service.queries(queries, strict=True)

    # Test empty queries
    assert list(service.queries([])) == []


# LLM-generated content at query #42
#--------------------------

```python
def test_FXRateService_query():
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
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    # Setup test data
    from pypara.currencies import Currencies
    import datetime
    from decimal import Decimal

    EUR = Currencies["EUR"]
    USD = Currencies["USD"]
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)

    rates = {
        (EUR, USD, today): FXRate(EUR, USD, today, Decimal("1.2")),
        (USD, EUR, today): FXRate(USD, EUR, today, Decimal("0.8333")),
        (EUR, USD, yesterday): FXRate(EUR, USD, yesterday, Decimal("1.15")),
    }

    service = TestFXRateService(rates)

    # Test successful query
    result = service.query(EUR, USD, today)
    assert result == rates[(EUR, USD, today)]

    # Test inverted rate
    result = service.query(USD, EUR, today)
    assert result == rates[(USD, EUR, today)]

    # Test non-strict query with missing rate
    result = service.query(EUR, USD, datetime.date(2020, 1, 1))
    assert result is None

    # Test strict query with missing rate
    with pytest.raises(FXRateLookupError):
        service.query(EUR, USD, datetime.date(2020, 1, 1), strict=True)

    # Test query with same currency
    result = service.query(EUR, EUR, today)
    assert result is None  # Assuming same currency rates aren't in our test data


# LLM-generated content at query #43
#--------------------------

```python
def test_FXRateService_query():
    # Create a mock FXRateService subclass for testing
    class MockFXRateService(FXRateService):
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

    # Initialize the mock service
    service = MockFXRateService()

    # Test successful query
    rate = service.query(Currency("EUR"), Currency("USD"), Date(2023, 1, 1))
    assert rate is not None
    assert rate.ccy1 == Currency("EUR")
    assert rate.ccy2 == Currency("USD")
    assert rate.date == Date(2023, 1, 1)
    assert rate.value == Decimal("1.1")

    # Test inverted query
    rate = service.query(Currency("USD"), Currency("EUR"), Date(2023, 1, 1))
    assert rate is not None
    assert rate.ccy1 == Currency("USD")
    assert rate.ccy2 == Currency("EUR")
    assert rate.date == Date(2023, 1, 1)
    assert rate.value == Decimal("0.9091")

    # Test non-existent rate without strict
    rate = service.query(Currency("GBP"), Currency("USD"), Date(2023, 1, 1))
    assert rate is None

    # Test non-existent rate with strict
    with pytest.raises(FXRateLookupError):
        service.query(Currency("GBP"), Currency("USD"), Date(2023, 1, 1), strict=True)


# LLM-generated content at query #44
#--------------------------

```python
def test_FXRateService_queries():
    # Create a mock FXRateService implementation for testing
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            # Mock implementation that returns a fixed rate for specific inputs
            if ccy1.code == "EUR" and ccy2.code == "USD" and asof == Date(2023, 1, 1):
                return FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1"))
            elif ccy1.code == "USD" and ccy2.code == "EUR" and asof == Date(2023, 1, 1):
                return FXRate(Currency("USD"), Currency("EUR"), Date(2023, 1, 1), Decimal("0.9091"))
            elif strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            else:
                return None

        def queries(self, queries: Iterable[Tuple[Currency, Currency, Date]], strict: bool = False) -> Iterable[Optional[FXRate]]:
            # Mock implementation that processes each query
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

    # Initialize the mock service
    service = MockFXRateService()

    # Test case 1: Normal queries with existing rates
    queries = [
        (Currency("EUR"), Currency("USD"), Date(2023, 1, 1)),
        (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)),
    ]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1"))
    assert results[1] == FXRate(Currency("USD"), Currency("EUR"), Date(2023, 1, 1), Decimal("0.9091"))

    # Test case 2: Queries with non-existing rates (non-strict mode)
    queries = [
        (Currency("EUR"), Currency("GBP"), Date(2023, 1, 1)),
        (Currency("GBP"), Currency("EUR"), Date(2023, 1, 1)),
    ]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] is None
    assert results[1] is None

    # Test case 3: Queries with non-existing rates (strict mode)
    queries = [
        (Currency("EUR"), Currency("GBP"), Date(2023, 1, 1)),
    ]
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))

    # Test case 4: Mixed queries with some existing and some non-existing rates (non-strict mode)
    queries = [
        (Currency("EUR"), Currency("USD"), Date(2023, 1, 1)),
        (Currency("EUR"), Currency("GBP"), Date(2023, 1, 1)),
        (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)),
    ]
    results = list(service.queries(queries))
    assert len(results) == 3
    assert results[0] == FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1"))
    assert results[1] is None
    assert results[2] == FXRate(Currency("USD"), Currency("EUR"), Date(2023, 1, 1), Decimal("0.9091"))

    # Test case 5: Empty queries
    queries = []
    results = list(service.queries(queries))
    assert len(results) == 0


# LLM-generated content at query #45
#--------------------------

```python
def test_FXRateService_query():
    # Create a mock FXRateService implementation for testing
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
            for ccy1, ccy2, asof in queries:
                yield self.query(ccy1, ccy2, asof, strict)

    # Test data
    eur = Currency("EUR", "Euro", 2)
    usd = Currency("USD", "US Dollar", 2)
    today = Date.today()
    yesterday = today - Date.resolution
    rate_eur_usd = FXRate(eur, usd, today, Decimal("1.20"))
    rate_usd_eur = FXRate(usd, eur, today, Decimal("0.8333"))

    # Initialize service with test data
    service = MockFXRateService({
        (eur, usd, today): rate_eur_usd,
        (usd, eur, today): rate_usd_eur,
    })

    # Test successful query
    result = service.query(eur, usd, today)
    assert result == rate_eur_usd

    # Test successful inverted query
    result = service.query(usd, eur, today)
    assert result == rate_usd_eur

    # Test non-strict query with missing rate returns None
    result = service.query(eur, usd, yesterday)
    assert result is None

    # Test strict query with missing rate raises exception
    with pytest.raises(FXRateLookupError):
        service.query(eur, usd, yesterday, strict=True)


# LLM-generated content at query #46
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

    # Test successful queries
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


# LLM-generated content at query #47
#--------------------------

```python
def test_FXRateService_queries():
    # Create a mock FXRateService implementation
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            if ccy1.code == "EUR" and ccy2.code == "USD" and asof == Date(2023, 1, 1):
                return FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.10"))
            elif ccy1.code == "USD" and ccy2.code == "EUR" and asof == Date(2023, 1, 1):
                return FXRate(Currency("USD"), Currency("EUR"), Date(2023, 1, 1), Decimal("0.91"))
            elif strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            else:
                return None

        def queries(self, queries: Iterable[Tuple[Currency, Currency, Date]], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return (self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries)

    # Initialize the mock service
    service = MockFXRateService()

    # Test case 1: Valid queries
    queries = [
        (Currency("EUR"), Currency("USD"), Date(2023, 1, 1)),
        (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)),
    ]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.10"))
    assert results[1] == FXRate(Currency("USD"), Currency("EUR"), Date(2023, 1, 1), Decimal("0.91"))

    # Test case 2: Invalid queries (non-strict)
    queries = [
        (Currency("EUR"), Currency("GBP"), Date(2023, 1, 1)),
        (Currency("JPY"), Currency("USD"), Date(2023, 1, 1)),
    ]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] is None
    assert results[1] is None

    # Test case 3: Invalid queries (strict)
    queries = [
        (Currency("EUR"), Currency("GBP"), Date(2023, 1, 1)),
    ]
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))

    # Test case 4: Mixed valid and invalid queries (non-strict)
    queries = [
        (Currency("EUR"), Currency("USD"), Date(2023, 1, 1)),
        (Currency("EUR"), Currency("GBP"), Date(2023, 1, 1)),
    ]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.10"))
    assert results[1] is None

    # Test case 5: Empty queries
    queries = []
    results = list(service.queries(queries))
    assert len(results) == 0


# LLM-generated content at query #48
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

    # Setup test data
    from pypara.currencies import Currencies
    import datetime

    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)

    rate1 = FXRate(eur, usd, today, Decimal("1.2"))
    rate2 = FXRate(usd, eur, yesterday, Decimal("0.8"))

    service = MockFXRateService({
        (eur, usd, today): rate1,
        (usd, eur, yesterday): rate2
    })

    # Test successful query
    result = service.query(eur, usd, today)
    assert result == rate1

    # Test successful query with inverted currencies
    result = service.query(usd, eur, yesterday)
    assert result == rate2

    # Test query with non-existent rate (non-strict)
    result = service.query(eur, usd, yesterday)
    assert result is None

    # Test query with non-existent rate (strict)
    try:
        service.query(eur, usd, yesterday, strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == eur
        assert e.ccy2 == usd
        assert e.asof == yesterday

    # Test query with same currency (should return rate of 1)
    same_currency_rate = FXRate(eur, eur, today, Decimal("1"))
    result = service.query(eur, eur, today)
    assert result == same_currency_rate


# LLM-generated content at query #49
#--------------------------

```python
def test_FXRateService_queries():
    # Create a mock FXRateService implementation for testing
    class MockFXRateService(FXRateService):
        def __init__(self):
            self.rates = {
                (Currency("EUR"), Currency("USD"), Date(2023, 1, 1)): FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.10")),
                (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)): FXRate(Currency("USD"), Currency("EUR"), Date(2023, 1, 1), Decimal("0.90")),
                (Currency("GBP"), Currency("USD"), Date(2023, 1, 1)): FXRate(Currency("GBP"), Currency("USD"), Date(2023, 1, 1), Decimal("1.30")),
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

    # Test with valid queries
    valid_queries = [
        (Currency("EUR"), Currency("USD"), Date(2023, 1, 1)),
        (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)),
        (Currency("GBP"), Currency("USD"), Date(2023, 1, 1)),
    ]

    rates = list(service.queries(valid_queries))
    assert len(rates) == 3
    assert rates[0] == FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.10"))
    assert rates[1] == FXRate(Currency("USD"), Currency("EUR"), Date(2023, 1, 1), Decimal("0.90"))
    assert rates[2] == FXRate(Currency("GBP"), Currency("USD"), Date(2023, 1, 1), Decimal("1.30"))

    # Test with invalid queries (strict=False)
    invalid_queries = [
        (Currency("EUR"), Currency("JPY"), Date(2023, 1, 1)),
        (Currency("AUD"), Currency("USD"), Date(2023, 1, 1)),
    ]

    rates = list(service.queries(invalid_queries))
    assert len(rates) == 2
    assert rates[0] is None
    assert rates[1] is None

    # Test with invalid queries (strict=True)
    try:
        list(service.queries(invalid_queries, strict=True))
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError:
        pass

    # Test with mixed valid and invalid queries (strict=True)
    mixed_queries = [
        (Currency("EUR"), Currency("USD"), Date(2023, 1, 1)),
        (Currency("AUD"), Currency("USD"), Date(2023, 1, 1)),
    ]

    try:
        list(service.queries(mixed_queries, strict=True))
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError:
        pass


# LLM-generated content at query #50
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
    eur = Currency("EUR", "Euro", 2)
    usd = Currency("USD", "US Dollar", 2)
    today = Date.today()
    yesterday = today - Date.resolution

    rate1 = FXRate(eur, usd, today, Decimal("1.2"))
    rate2 = FXRate(usd, eur, yesterday, Decimal("0.8"))

    # Initialize the test service
    service = TestFXRateService([rate1, rate2])

    # Test case 1: Query existing rates
    queries = [(eur, usd, today), (usd, eur, yesterday)]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == rate1
    assert results[1] == rate2

    # Test case 2: Query non-existing rate without strict
    queries = [(eur, usd, yesterday)]
    results = list(service.queries(queries))
    assert len(results) == 1
    assert results[0] is None

    # Test case 3: Query non-existing rate with strict
    queries = [(eur, usd, yesterday)]
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))

    # Test case 4: Empty queries
    queries = []
    results = list(service.queries(queries))
    assert len(results) == 0


# LLM-generated content at query #51
#--------------------------

```python
def test_FXRateService_query():
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

    # Test data
    from datetime import date
    from pypara.currencies import Currencies
    test_date = date(2023, 1, 1)
    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    test_rate = FXRate(eur, usd, test_date, Decimal("1.2"))

    # Initialize service with test data
    service = MockFXRateService({(eur, usd, test_date): test_rate})

    # Test successful query
    result = service.query(eur, usd, test_date)
    assert result == test_rate

    # Test query with non-existent rate (non-strict)
    result = service.query(usd, eur, test_date)
    assert result is None

    # Test query with non-existent rate (strict)
    with pytest.raises(FXRateLookupError):
        service.query(usd, eur, test_date, strict=True)

    # Test query with different date
    result = service.query(eur, usd, date(2023, 1, 2))
    assert result is None


# LLM-generated content at query #52
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

    # Test data
    from datetime import date
    from pypara.currencies import Currencies
    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    today = date.today()

    # Setup test service with some rates
    test_rates = {
        (eur, usd, today): FXRate(eur, usd, today, Decimal("1.2")),
        (usd, eur, today): FXRate(usd, eur, today, Decimal("0.8333")),
    }
    service = TestFXRateService(test_rates)

    # Test successful query
    result = service.query(eur, usd, today)
    assert result == FXRate(eur, usd, today, Decimal("1.2"))

    # Test query with None result (non-strict)
    result = service.query(eur, usd, date(2020, 1, 1))
    assert result is None

    # Test query with lookup error (strict)
    with pytest.raises(FXRateLookupError):
        service.query(eur, usd, date(2020, 1, 1), strict=True)

    # Test inverted rate
    result = service.query(usd, eur, today)
    assert result == FXRate(usd, eur, today, Decimal("0.8333"))


# LLM-generated content at query #53
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
    rate3 = FXRate(usd, eur, date1, Decimal("0.9"))

    # Test case 1: Normal queries
    service = MockFXRateService([rate1, rate2, rate3])
    queries = [(eur, usd, date1), (eur, usd, date2), (usd, eur, date1)]
    results = list(service.queries(queries))

    assert len(results) == 3
    assert results[0] == rate1
    assert results[1] == rate2
    assert results[2] == rate3

    # Test case 2: Query with missing rate (non-strict)
    queries = [(eur, usd, date1), (eur, usd, Date(2023, 1, 3))]
    results = list(service.queries(queries))

    assert len(results) == 2
    assert results[0] == rate1
    assert results[1] is None

    # Test case 3: Query with missing rate (strict)
    queries = [(eur, usd, date1), (eur, usd, Date(2023, 1, 3))]
    try:
        list(service.queries(queries, strict=True))
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == eur
        assert e.ccy2 == usd
        assert e.asof == Date(2023, 1, 3)

    # Test case 4: Empty queries
    results = list(service.queries([]))
    assert len(results) == 0


# LLM-generated content at query #54
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

    # Test non-existent rate without strict
    rate = service.query(Currency("GBP"), Currency("USD"), Date(2023, 1, 1))
    assert rate is None

    # Test non-existent rate with strict
    try:
        service.query(Currency("GBP"), Currency("USD"), Date(2023, 1, 1), strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == Currency("GBP")
        assert e.ccy2 == Currency("USD")
        assert e.asof == Date(2023, 1, 1)


# LLM-generated content at query #55
#--------------------------

```python
def test_FXRateService_queries():
    # Create a mock FXRateService implementation
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            # Mock implementation that returns a fixed rate for EUR/USD
            if ccy1.code == "EUR" and ccy2.code == "USD":
                return FXRate(ccy1, ccy2, asof, Decimal("1.2"))
            return None

        def queries(self, queries: Iterable[Tuple[Currency, Currency, Date]], strict: bool = False) -> Iterable[Optional[FXRate]]:
            # Mock implementation that processes queries
            results = []
            for ccy1, ccy2, asof in queries:
                rate = self.query(ccy1, ccy2, asof)
                if rate is None and strict:
                    raise FXRateLookupError(ccy1, ccy2, asof)
                results.append(rate)
            return results

    # Create mock currencies and dates
    class MockCurrency:
        def __init__(self, code):
            self.code = code

    eur = MockCurrency("EUR")
    usd = MockCurrency("USD")
    gbp = MockCurrency("GBP")
    today = Date.today()

    # Initialize the mock service
    service = MockFXRateService()

    # Test case 1: Single query with existing rate
    queries = [(eur, usd, today)]
    results = list(service.queries(queries))
    assert len(results) == 1
    assert results[0] == FXRate(eur, usd, today, Decimal("1.2"))

    # Test case 2: Multiple queries with existing rates
    queries = [(eur, usd, today), (eur, usd, today)]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert all(r == FXRate(eur, usd, today, Decimal("1.2")) for r in results)

    # Test case 3: Query with non-existing rate (non-strict mode)
    queries = [(eur, gbp, today)]
    results = list(service.queries(queries))
    assert len(results) == 1
    assert results[0] is None

    # Test case 4: Query with non-existing rate (strict mode)
    queries = [(eur, gbp, today)]
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))

    # Test case 5: Mixed queries (existing and non-existing rates)
    queries = [(eur, usd, today), (eur, gbp, today)]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == FXRate(eur, usd, today, Decimal("1.2"))
    assert results[1] is None

    # Test case 6: Mixed queries in strict mode
    queries = [(eur, usd, today), (eur, gbp, today)]
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))


# LLM-generated content at query #56
#--------------------------

```python
def test_FXRateService_queries():
    # Create a mock FXRateService implementation for testing
    class MockFXRateService(FXRateService):
        def __init__(self):
            self.rates = {
                (Currency("EUR"), Currency("USD"), Date(2023, 1, 1)): FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1")),
                (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)): FXRate(Currency("USD"), Currency("EUR"), Date(2023, 1, 1), Decimal("0.909")),
                (Currency("EUR"), Currency("GBP"), Date(2023, 1, 1)): FXRate(Currency("EUR"), Currency("GBP"), Date(2023, 1, 1), Decimal("0.85")),
            }

        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            return self.rates.get((ccy1, ccy2, asof), None)

        def queries(self, queries: Iterable[Tuple[Currency, Currency, Date]], strict: bool = False) -> Iterable[Optional[FXRate]]:
            for ccy1, ccy2, asof in queries:
                rate = self.query(ccy1, ccy2, asof)
                if rate is None and strict:
                    raise FXRateLookupError(ccy1, ccy2, asof)
                yield rate

    # Initialize the mock service
    service = MockFXRateService()

    # Test case 1: Multiple queries with existing rates
    queries = [
        (Currency("EUR"), Currency("USD"), Date(2023, 1, 1)),
        (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)),
        (Currency("EUR"), Currency("GBP"), Date(2023, 1, 1)),
    ]
    results = list(service.queries(queries))
    assert len(results) == 3
    assert results[0] == FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1"))
    assert results[1] == FXRate(Currency("USD"), Currency("EUR"), Date(2023, 1, 1), Decimal("0.909"))
    assert results[2] == FXRate(Currency("EUR"), Currency("GBP"), Date(2023, 1, 1), Decimal("0.85"))

    # Test case 2: Query with non-existing rate (non-strict mode)
    queries = [
        (Currency("EUR"), Currency("JPY"), Date(2023, 1, 1)),
    ]
    results = list(service.queries(queries))
    assert len(results) == 1
    assert results[0] is None

    # Test case 3: Query with non-existing rate (strict mode)
    queries = [
        (Currency("EUR"), Currency("JPY"), Date(2023, 1, 1)),
    ]
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))

    # Test case 4: Mixed queries with some existing and some non-existing rates (non-strict mode)
    queries = [
        (Currency("EUR"), Currency("USD"), Date(2023, 1, 1)),
        (Currency("EUR"), Currency("JPY"), Date(2023, 1, 1)),
        (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)),
    ]
    results = list(service.queries(queries))
    assert len(results) == 3
    assert results[0] == FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1"))
    assert results[1] is None
    assert results[2] == FXRate(Currency("USD"), Currency("EUR"), Date(2023, 1, 1), Decimal("0.909"))

    # Test case 5: Mixed queries with some existing and some non-existing rates (strict mode)
    queries = [
        (Currency("EUR"), Currency("USD"), Date(2023, 1, 1)),
        (Currency("EUR"), Currency("JPY"), Date(2023, 1, 1)),
        (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)),
    ]
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))


# LLM-generated content at query #57
#--------------------------

```python
def test_FXRateService_queries():
    # Create a mock FXRateService subclass for testing
    class MockFXRateService(FXRateService):
        def __init__(self):
            self.query_calls = []

        def query(self, ccy1, ccy2, asof, strict=False):
            self.query_calls.append((ccy1, ccy2, asof, strict))
            if ccy1 == ccy2:
                return FXRate.of(ccy1, ccy2, asof, ONE)
            return FXRate.of(ccy1, ccy2, asof, Decimal("2"))

        def queries(self, queries, strict=False):
            results = []
            for ccy1, ccy2, asof in queries:
                result = self.query(ccy1, ccy2, asof, strict)
                if result is None and strict:
                    raise FXRateLookupError(ccy1, ccy2, asof)
                results.append(result)
            return results

    # Initialize the mock service
    service = MockFXRateService()

    # Test with valid queries
    queries = [
        (Currency("EUR"), Currency("USD"), Date(2023, 1, 1)),
        (Currency("USD"), Currency("EUR"), Date(2023, 1, 2)),
        (Currency("EUR"), Currency("EUR"), Date(2023, 1, 3)),
    ]
    results = list(service.queries(queries))

    assert len(results) == 3
    assert results[0] == FXRate.of(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("2"))
    assert results[1] == FXRate.of(Currency("USD"), Currency("EUR"), Date(2023, 1, 2), Decimal("2"))
    assert results[2] == FXRate.of(Currency("EUR"), Currency("EUR"), Date(2023, 1, 3), ONE)

    # Test with strict mode and missing rate
    class StrictMockFXRateService(MockFXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if ccy1 == Currency("GBP") and ccy2 == Currency("JPY"):
                return None
            return super().query(ccy1, ccy2, asof, strict)

    strict_service = StrictMockFXRateService()
    strict_queries = [
        (Currency("EUR"), Currency("USD"), Date(2023, 1, 1)),
        (Currency("GBP"), Currency("JPY"), Date(2023, 1, 2)),
    ]

    with pytest.raises(FXRateLookupError):
        list(strict_service.queries(strict_queries, strict=True))

    # Test with non-strict mode and missing rate
    non_strict_results = list(strict_service.queries(strict_queries, strict=False))
    assert len(non_strict_results) == 2
    assert non_strict_results[0] == FXRate.of(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("2"))
    assert non_strict_results[1] is None


# LLM-generated content at query #58
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

    # Test setup
    from decimal import Decimal
    from datetime import date
    from pypara.currencies import Currencies

    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    today = date.today()
    rate = FXRate(eur, usd, today, Decimal("1.2"))

    # Initialize service with test data
    service = TestFXRateService({(eur, usd, today): rate})

    # Test successful query
    result = service.query(eur, usd, today)
    assert result == rate

    # Test query with non-existent rate (non-strict)
    result = service.query(eur, usd, date(2020, 1, 1))
    assert result is None

    # Test query with non-existent rate (strict)
    with pytest.raises(FXRateLookupError):
        service.query(eur, usd, date(2020, 1, 1), strict=True)

    # Test query with inverted currencies
    inverted_rate = ~rate
    service_with_inverted = TestFXRateService({(usd, eur, today): inverted_rate})
    result = service_with_inverted.query(usd, eur, today)
    assert result == inverted_rate


# LLM-generated content at query #59
#--------------------------

```python
def test_FXRateService_queries():
    # Create a mock FXRateService subclass for testing
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

    # Test data
    eur = Currency("EUR")
    usd = Currency("USD")
    today = Date.today()
    yesterday = today - datetime.timedelta(days=1)

    rate1 = FXRate(eur, usd, today, Decimal("1.2"))
    rate2 = FXRate(usd, eur, yesterday, Decimal("0.8"))

    # Initialize mock service
    service = MockFXRateService([rate1, rate2])

    # Test successful queries
    queries = [(eur, usd, today), (usd, eur, yesterday)]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == rate1
    assert results[1] == rate2

    # Test with missing rate (non-strict)
    queries = [(eur, usd, today), (eur, usd, yesterday)]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == rate1
    assert results[1] is None

    # Test with missing rate (strict)
    queries = [(eur, usd, today), (eur, usd, yesterday)]
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))

    # Test empty queries
    assert list(service.queries([])) == []


# LLM-generated content at query #60
#--------------------------

```python
def test_FXRateService_query():
    # Create a mock FXRateService implementation for testing
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
            for ccy1, ccy2, asof in queries:
                yield self.query(ccy1, ccy2, asof, strict)

    # Create test data
    from datetime import date
    from decimal import Decimal
    from pypara.currencies import Currencies

    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    today = date.today()
    rate = FXRate(eur, usd, today, Decimal("1.2"))

    # Initialize mock service
    service = MockFXRateService({(eur, usd, today): rate})

    # Test successful query
    result = service.query(eur, usd, today)
    assert result == rate

    # Test query with non-existent rate (non-strict)
    result = service.query(usd, eur, today)
    assert result is None

    # Test query with non-existent rate (strict)
    with pytest.raises(FXRateLookupError):
        service.query(usd, eur, today, strict=True)

    # Test query with different date
    result = service.query(eur, usd, date(2020, 1, 1))
    assert result is None

    # Test query with same currency
    same_currency_rate = FXRate(eur, eur, today, Decimal("1"))
    service_with_same_currency = MockFXRateService({(eur, eur, today): same_currency_rate})
    result = service_with_same_currency.query(eur, eur, today)
    assert result == same_currency_rate


# LLM-generated content at query #61
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

    # Test data
    from datetime import date
    from pypara.currencies import Currencies
    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    today = date.today()

    # Create test service with some rates
    rates = {
        (eur, usd, today): FXRate(eur, usd, today, Decimal("1.2")),
        (usd, eur, today): FXRate(usd, eur, today, Decimal("0.8333")),
    }
    service = TestFXRateService(rates)

    # Test successful query
    result = service.query(eur, usd, today)
    assert result == FXRate(eur, usd, today, Decimal("1.2"))

    # Test query with None result (non-strict)
    result = service.query(eur, usd, date(2000, 1, 1))
    assert result is None

    # Test query with LookupError (strict)
    with pytest.raises(FXRateLookupError):
        service.query(eur, usd, date(2000, 1, 1), strict=True)

    # Test inverted rate
    result = service.query(usd, eur, today)
    assert result == FXRate(usd, eur, today, Decimal("0.8333"))


# LLM-generated content at query #62
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
    from pypara.currencies import Currencies
    import datetime
    from decimal import Decimal

    EUR = Currencies["EUR"]
    USD = Currencies["USD"]
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)

    rates = {
        (EUR, USD, today): FXRate(EUR, USD, today, Decimal("1.2")),
        (USD, EUR, today): FXRate(USD, EUR, today, Decimal("0.8333")),
        (EUR, USD, yesterday): FXRate(EUR, USD, yesterday, Decimal("1.18")),
    }

    service = TestFXRateService(rates)

    # Test successful query
    result = service.query(EUR, USD, today)
    assert result == FXRate(EUR, USD, today, Decimal("1.2"))

    # Test inverted rate
    result = service.query(USD, EUR, today)
    assert result == FXRate(USD, EUR, today, Decimal("0.8333"))

    # Test historical rate
    result = service.query(EUR, USD, yesterday)
    assert result == FXRate(EUR, USD, yesterday, Decimal("1.18"))

    # Test non-existent rate without strict
    result = service.query(EUR, USD, datetime.date(2000, 1, 1))
    assert result is None

    # Test non-existent rate with strict
    try:
        service.query(EUR, USD, datetime.date(2000, 1, 1), strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == EUR
        assert e.ccy2 == USD
        assert e.asof == datetime.date(2000, 1, 1)

    # Test same currency rate
    result = service.query(EUR, EUR, today)
    assert result is None  # Assuming no explicit same-currency rate is stored

    # Test with strict=True for same currency (should raise error if not found)
    try:
        service.query(EUR, EUR, today, strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == EUR
        assert e.ccy2 == EUR
        assert e.asof == today


# LLM-generated content at query #63
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

    # Create test currencies and dates
    from pypara.currencies import Currencies
    import datetime
    EUR = Currencies["EUR"]
    USD = Currencies["USD"]
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)

    # Create test rates
    test_rates = {
        (EUR, USD, today): FXRate(EUR, USD, today, Decimal("1.2")),
        (USD, EUR, today): FXRate(USD, EUR, today, Decimal("0.8333")),
        (EUR, USD, yesterday): FXRate(EUR, USD, yesterday, Decimal("1.15")),
    }

    # Initialize the test service
    service = TestFXRateService(test_rates)

    # Test 1: Successful query
    result = service.query(EUR, USD, today)
    assert result == test_rates[(EUR, USD, today)]

    # Test 2: Query with inversion
    result = service.query(USD, EUR, today)
    assert result == test_rates[(USD, EUR, today)]

    # Test 3: Query for different date
    result = service.query(EUR, USD, yesterday)
    assert result == test_rates[(EUR, USD, yesterday)]

    # Test 4: Non-existent rate without strict
    result = service.query(EUR, USD, datetime.date(2000, 1, 1))
    assert result is None

    # Test 5: Non-existent rate with strict (should raise exception)
    import pytest
    with pytest.raises(FXRateLookupError):
        service.query(EUR, USD, datetime.date(2000, 1, 1), strict=True)

    # Test 6: Verify exception message
    try:
        service.query(EUR, USD, datetime.date(2000, 1, 1), strict=True)
    except FXRateLookupError as e:
        assert str(e) == f"Foreign exchange rate for {EUR}/{USD} not found as of {datetime.date(2000, 1, 1)}"
        assert e.ccy1 == EUR
        assert e.ccy2 == USD
        assert e.asof == datetime.date(2000, 1, 1)


# LLM-generated content at query #64
#--------------------------

```python
def test_FXRateService_query():
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
            for query in queries:
                yield self.query(*query, strict=strict)

    # Setup test data
    from pypara.currencies import Currencies
    import datetime
    from decimal import Decimal

    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)

    rates = {
        (eur, usd, today): FXRate(eur, usd, today, Decimal("1.2")),
        (usd, eur, today): FXRate(usd, eur, today, Decimal("0.8333")),
        (eur, usd, yesterday): FXRate(eur, usd, yesterday, Decimal("1.15")),
    }

    service = TestFXRateService(rates)

    # Test 1: Successful query
    result = service.query(eur, usd, today)
    assert result == FXRate(eur, usd, today, Decimal("1.2"))

    # Test 2: Query with inverted currencies
    result = service.query(usd, eur, today)
    assert result == FXRate(usd, eur, today, Decimal("0.8333"))

    # Test 3: Query for different date
    result = service.query(eur, usd, yesterday)
    assert result == FXRate(eur, usd, yesterday, Decimal("1.15"))

    # Test 4: Non-existent rate without strict
    result = service.query(eur, usd, datetime.date(2000, 1, 1))
    assert result is None

    # Test 5: Non-existent rate with strict
    try:
        service.query(eur, usd, datetime.date(2000, 1, 1), strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == eur
        assert e.ccy2 == usd
        assert e.asof == datetime.date(2000, 1, 1)


# LLM-generated content at query #65
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
    eur = Currency("EUR", "Euro", 978)
    usd = Currency("USD", "US Dollar", 840)
    today = Date.today()
    yesterday = today - Date.resolution

    rate1 = FXRate(eur, usd, today, Decimal("1.2"))
    rate2 = FXRate(usd, eur, today, Decimal("0.8"))
    rate3 = FXRate(eur, usd, yesterday, Decimal("1.1"))

    rates = [rate1, rate2, rate3]

    # Initialize the test service
    service = TestFXRateService(rates)

    # Test case 1: Multiple queries with existing rates
    queries1 = [
        (eur, usd, today),
        (usd, eur, today),
        (eur, usd, yesterday)
    ]
    results1 = list(service.queries(queries1))
    assert results1 == [rate1, rate2, rate3]

    # Test case 2: Query with non-existing rate (non-strict)
    queries2 = [
        (eur, usd, today),
        (eur, usd, Date(2020, 1, 1))  # Non-existing date
    ]
    results2 = list(service.queries(queries2, strict=False))
    assert results2 == [rate1, None]

    # Test case 3: Query with non-existing rate (strict)
    queries3 = [
        (eur, usd, today),
        (eur, usd, Date(2020, 1, 1))  # Non-existing date
    ]
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries3, strict=True))

    # Test case 4: Empty queries
    queries4 = []
    results4 = list(service.queries(queries4))
    assert results4 == []

    # Test case 5: Query with inverted rate
    queries5 = [
        (usd, eur, today)
    ]
    results5 = list(service.queries(queries5))
    assert results5 == [rate2]
    assert results5[0] == ~rate1


# LLM-generated content at query #66
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
    today = Date(2023, 1, 1)
    tomorrow = Date(2023, 1, 2)
    rate1 = FXRate(eur, usd, today, Decimal("1.1"))
    rate2 = FXRate(usd, eur, tomorrow, Decimal("0.9"))
    rates = [rate1, rate2]

    # Create service instance
    service = TestFXRateService(rates)

    # Test queries with existing rates
    queries = [(eur, usd, today), (usd, eur, tomorrow)]
    results = service.queries(queries)
    assert list(results) == [rate1, rate2]

    # Test queries with non-existing rates (non-strict)
    queries = [(eur, usd, tomorrow), (usd, eur, today)]
    results = service.queries(queries)
    assert list(results) == [None, None]

    # Test queries with non-existing rates (strict)
    queries = [(eur, usd, tomorrow)]
    with pytest.raises(FXRateLookupError):
        service.queries(queries, strict=True)

    # Test empty queries
    assert list(service.queries([])) == []


# LLM-generated content at query #67
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

    # Test setup
    from datetime import date
    from pypara.currencies import Currencies

    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    today = date.today()

    rates = {
        (eur, usd, today): FXRate(eur, usd, today, Decimal("1.2")),
        (usd, eur, today): FXRate(usd, eur, today, Decimal("0.8333")),
    }

    service = TestFXRateService(rates)

    # Test queries with existing rates
    queries = [(eur, usd, today), (usd, eur, today)]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == FXRate(eur, usd, today, Decimal("1.2"))
    assert results[1] == FXRate(usd, eur, today, Decimal("0.8333"))

    # Test queries with non-existing rates (non-strict)
    queries = [(eur, usd, today), (usd, eur, today), (eur, usd, date(2020, 1, 1))]
    results = list(service.queries(queries))
    assert len(results) == 3
    assert results[0] == FXRate(eur, usd, today, Decimal("1.2"))
    assert results[1] == FXRate(usd, eur, today, Decimal("0.8333"))
    assert results[2] is None

    # Test queries with non-existing rates (strict)
    queries = [(eur, usd, today), (eur, usd, date(2020, 1, 1))]
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))


# LLM-generated content at query #68
#--------------------------

```python
def test_FXRateService_query():
    # Setup
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
            return (self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries)

    from datetime import date
    from pypara.currencies import Currencies
    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    today = date.today()
    rate = FXRate(eur, usd, today, Decimal("1.2"))
    service = TestFXRateService({(eur, usd, today): rate})

    # Test successful query
    assert service.query(eur, usd, today) == rate

    # Test query with strict=True and existing rate
    assert service.query(eur, usd, today, strict=True) == rate

    # Test query with non-existing rate and strict=False
    assert service.query(usd, eur, today, strict=False) is None

    # Test query with non-existing rate and strict=True
    with pytest.raises(FXRateLookupError):
        service.query(usd, eur, today, strict=True)


# LLM-generated content at query #69
#--------------------------

```python
def test_FXRateService_queries():
    # Create a mock FXRateService implementation for testing
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

    # Setup test data
    eur = Currency("EUR", "Euro", 978)
    usd = Currency("USD", "US Dollar", 840)
    today = Date.today()
    yesterday = today - datetime.timedelta(days=1)

    rate1 = FXRate(eur, usd, today, Decimal("1.20"))
    rate2 = FXRate(usd, eur, yesterday, Decimal("0.83"))

    # Create service instance with test rates
    service = MockFXRateService({
        (eur, usd, today): rate1,
        (usd, eur, yesterday): rate2
    })

    # Test 1: Basic queries
    queries = [
        (eur, usd, today),
        (usd, eur, yesterday)
    ]
    results = service.queries(queries)
    assert list(results) == [rate1, rate2]

    # Test 2: Query with missing rate (non-strict)
    queries = [
        (eur, usd, today),
        (eur, usd, yesterday)  # This rate doesn't exist
    ]
    results = service.queries(queries)
    assert list(results) == [rate1, None]

    # Test 3: Query with missing rate (strict)
    queries = [
        (eur, usd, today),
        (eur, usd, yesterday)  # This rate doesn't exist
    ]
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))

    # Test 4: Empty queries
    assert list(service.queries([])) == []

    # Test 5: Inverted rate query
    queries = [
        (usd, eur, today)  # Should return None as we only have EUR/USD
    ]
    results = service.queries(queries)
    assert list(results) == [None]


# LLM-generated content at query #70
#--------------------------

```python
def test_FXRateService_query():
    # Create a mock FXRateService implementation for testing
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

    # Create test data
    from datetime import date
    from pypara.currencies import Currencies
    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    today = date.today()

    # Test successful query
    service = MockFXRateService({
        (eur, usd, today): FXRate(eur, usd, today, Decimal("1.2"))
    })
    result = service.query(eur, usd, today)
    assert result == FXRate(eur, usd, today, Decimal("1.2"))

    # Test query with None result
    result = service.query(usd, eur, today)
    assert result is None

    # Test strict query with missing rate
    with pytest.raises(FXRateLookupError):
        service.query(usd, eur, today, strict=True)

    # Test query with different date
    yesterday = date.today() - timedelta(days=1)
    result = service.query(eur, usd, yesterday)
    assert result is None


# LLM-generated content at query #71
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

    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    today = date.today()

    rate1 = FXRate(eur, usd, today, Decimal("1.2"))
    rate2 = FXRate(usd, eur, today, Decimal("0.8333"))
    rate3 = FXRate(eur, eur, today, Decimal("1"))

    rates = {
        (eur, usd, today): rate1,
        (usd, eur, today): rate2,
        (eur, eur, today): rate3,
    }

    service = MockFXRateService(rates)

    # Test successful queries
    queries = [
        (eur, usd, today),
        (usd, eur, today),
        (eur, eur, today),
    ]
    results = list(service.queries(queries))
    assert results == [rate1, rate2, rate3]

    # Test with missing rate (non-strict)
    queries_with_missing = [
        (eur, usd, today),
        (usd, eur, today),
        (eur, Currencies["GBP"], today),  # Missing
    ]
    results = list(service.queries(queries_with_missing))
    assert results == [rate1, rate2, None]

    # Test with missing rate (strict)
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries_with_missing, strict=True))

    # Test empty queries
    assert list(service.queries([])) == []


# LLM-generated content at query #72
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
    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    today = date.today()
    test_rate = FXRate(eur, usd, today, Decimal("1.2"))

    # Test case 1: Successful query
    service = TestFXRateService({(eur, usd, today): test_rate})
    result = service.query(eur, usd, today)
    assert result == test_rate

    # Test case 2: Query returns None when rate not found
    result = service.query(usd, eur, today)
    assert result is None

    # Test case 3: Query raises FXRateLookupError when strict=True and rate not found
    with pytest.raises(FXRateLookupError):
        service.query(usd, eur, today, strict=True)

    # Test case 4: Query returns None when strict=False and rate not found
    result = service.query(usd, eur, today, strict=False)
    assert result is None

    # Test case 5: Query with different date
    yesterday = date.today() - timedelta(days=1)
    result = service.query(eur, usd, yesterday)
    assert result is None


# LLM-generated content at query #73
#--------------------------

```python
def test_FXRateService_query():
    # Create a mock FXRateService implementation for testing
    class MockFXRateService(FXRateService):
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


# LLM-generated content at query #74
#--------------------------

```python
def test_FXRateService_query():
    # Create a mock FXRateService implementation for testing
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

    # Test data setup
    from datetime import date
    from pypara.currencies import Currencies
    EUR = Currencies["EUR"]
    USD = Currencies["USD"]
    test_date = date(2023, 1, 1)
    test_rate = FXRate(EUR, USD, test_date, Decimal("1.10"))

    # Initialize service with test data
    service = MockFXRateService({(EUR, USD, test_date): test_rate})

    # Test successful query
    result = service.query(EUR, USD, test_date)
    assert result == test_rate

    # Test query with non-existent rate (non-strict)
    result = service.query(USD, EUR, test_date)
    assert result is None

    # Test query with non-existent rate (strict)
    with pytest.raises(FXRateLookupError):
        service.query(USD, EUR, test_date, strict=True)

    # Test query with different date
    result = service.query(EUR, USD, date(2022, 1, 1))
    assert result is None

    # Test query with inverted currencies
    inverted_rate = ~test_rate
    service.rates[(USD, EUR, test_date)] = inverted_rate
    result = service.query(USD, EUR, test_date)
    assert result == inverted_rate


# LLM-generated content at query #75
#--------------------------

```python
def test_FXRateService_query():
    # Setup
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

    from pypara.currencies import Currencies
    import datetime

    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    today = datetime.date.today()
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


# LLM-generated content at query #76
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
    from pypara.currencies import Currencies
    import datetime
    from decimal import Decimal

    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)

    test_rates = {
        (eur, usd, today): FXRate(eur, usd, today, Decimal("1.2")),
        (usd, eur, today): FXRate(usd, eur, today, Decimal("0.8333")),
        (eur, usd, yesterday): FXRate(eur, usd, yesterday, Decimal("1.15")),
    }

    service = TestFXRateService(test_rates)

    # Test successful query
    result = service.query(eur, usd, today)
    assert result == FXRate(eur, usd, today, Decimal("1.2"))

    # Test query with None result (non-strict)
    result = service.query(eur, usd, datetime.date(2020, 1, 1))
    assert result is None

    # Test query with None result (strict)
    with pytest.raises(FXRateLookupError):
        service.query(eur, usd, datetime.date(2020, 1, 1), strict=True)

    # Test inverted rate
    result = service.query(usd, eur, today)
    assert result == FXRate(usd, eur, today, Decimal("0.8333"))

    # Test different date
    result = service.query(eur, usd, yesterday)
    assert result == FXRate(eur, usd, yesterday, Decimal("1.15"))


# LLM-generated content at query #77
#--------------------------

```python
def test_FXRateService_queries():
    # Mock implementation of FXRateService for testing
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            if ccy1.code == "EUR" and ccy2.code == "USD" and asof == Date(2023, 1, 1):
                return FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1"))
            elif ccy1.code == "USD" and ccy2.code == "JPY" and asof == Date(2023, 1, 1):
                return FXRate(Currency("USD"), Currency("JPY"), Date(2023, 1, 1), Decimal("110.5"))
            elif strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None

        def queries(self, queries: Iterable[Tuple[Currency, Currency, Date]], strict: bool = False) -> Iterable[Optional[FXRate]]:
            for ccy1, ccy2, asof in queries:
                yield self.query(ccy1, ccy2, asof, strict)

    # Test data
    eur = Currency("EUR")
    usd = Currency("USD")
    jpy = Currency("JPY")
    date1 = Date(2023, 1, 1)
    date2 = Date(2023, 1, 2)

    # Test case 1: Valid queries
    queries = [
        (eur, usd, date1),
        (usd, jpy, date1),
    ]
    service = MockFXRateService()
    results = list(service.queries(queries))

    assert len(results) == 2
    assert results[0] == FXRate(eur, usd, date1, Decimal("1.1"))
    assert results[1] == FXRate(usd, jpy, date1, Decimal("110.5"))

    # Test case 2: Query with no result
    queries = [
        (eur, jpy, date2),
    ]
    results = list(service.queries(queries))

    assert len(results) == 1
    assert results[0] is None

    # Test case 3: Strict mode with missing rate
    queries = [
        (eur, jpy, date2),
    ]
    service = MockFXRateService()
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))


# LLM-generated content at query #78
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
    today = Date.today()
    yesterday = today - Date.resolution

    rate1 = FXRate(eur, usd, today, Decimal("1.2"))
    rate2 = FXRate(usd, eur, today, Decimal("0.8333"))
    rate3 = FXRate(eur, usd, yesterday, Decimal("1.1"))

    rates = [rate1, rate2, rate3]
    service = MockFXRateService(rates)

    # Test case 1: Normal queries
    queries = [(eur, usd, today), (usd, eur, today), (eur, usd, yesterday)]
    results = list(service.queries(queries))
    assert results == [rate1, rate2, rate3]

    # Test case 2: Query with missing rate (non-strict)
    queries = [(eur, usd, today), (eur, usd, Date(2020, 1, 1))]
    results = list(service.queries(queries))
    assert results == [rate1, None]

    # Test case 3: Query with missing rate (strict)
    queries = [(eur, usd, today), (eur, usd, Date(2020, 1, 1))]
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))

    # Test case 4: Empty queries
    assert list(service.queries([])) == []

    # Test case 5: Query with inverted rate
    queries = [(usd, eur, today)]
    results = list(service.queries(queries))
    assert results == [rate2]
    assert results[0] == ~rate1


# LLM-generated content at query #79
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
                rate = self.query(ccy1, ccy2, asof, strict)
                results.append(rate)
            return results

    # Test data
    eur = Currency("EUR", "Euro", 2)
    usd = Currency("USD", "US Dollar", 2)
    today = Date.today()
    yesterday = today - datetime.timedelta(days=1)

    rates = [
        FXRate(eur, usd, today, Decimal("1.2")),
        FXRate(usd, eur, today, Decimal("0.8333")),
        FXRate(eur, usd, yesterday, Decimal("1.15")),
    ]

    service = MockFXRateService(rates)

    # Test case 1: Normal queries
    queries = [
        (eur, usd, today),
        (usd, eur, today),
        (eur, usd, yesterday),
    ]
    results = service.queries(queries)
    assert len(results) == 3
    assert results[0] == rates[0]
    assert results[1] == rates[1]
    assert results[2] == rates[2]

    # Test case 2: Query with missing rate (non-strict)
    queries = [
        (eur, usd, today),
        (usd, eur, yesterday),  # Missing
    ]
    results = service.queries(queries)
    assert len(results) == 2
    assert results[0] == rates[0]
    assert results[1] is None

    # Test case 3: Query with missing rate (strict)
    queries = [
        (eur, usd, today),
        (usd, eur, yesterday),  # Missing
    ]
    with pytest.raises(FXRateLookupError):
        service.queries(queries, strict=True)

    # Test case 4: Empty queries
    results = service.queries([])
    assert len(results) == 0


# LLM-generated content at query #80
#--------------------------

```python
def test_FXRateService_queries():
    # Setup
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if ccy1.code == "EUR" and ccy2.code == "USD" and asof == Date(2023, 1, 1):
                return FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1"))
            elif ccy1.code == "USD" and ccy2.code == "GBP" and asof == Date(2023, 1, 2):
                return FXRate(Currency("USD"), Currency("GBP"), Date(2023, 1, 2), Decimal("0.8"))
            elif strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None

        def queries(self, queries, strict=False):
            results = []
            for ccy1, ccy2, asof in queries:
                result = self.query(ccy1, ccy2, asof, strict)
                if result is None and strict:
                    raise FXRateLookupError(ccy1, ccy2, asof)
                results.append(result)
            return results

    service = MockFXRateService()
    queries = [
        (Currency("EUR"), Currency("USD"), Date(2023, 1, 1)),
        (Currency("USD"), Currency("GBP"), Date(2023, 1, 2)),
        (Currency("JPY"), Currency("CAD"), Date(2023, 1, 3))
    ]

    # Test non-strict mode
    results = list(service.queries(queries, strict=False))
    assert len(results) == 3
    assert results[0] == FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1"))
    assert results[1] == FXRate(Currency("USD"), Currency("GBP"), Date(2023, 1, 2), Decimal("0.8"))
    assert results[2] is None

    # Test strict mode
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))


