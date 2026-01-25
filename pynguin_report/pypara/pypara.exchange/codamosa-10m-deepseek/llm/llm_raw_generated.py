####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

Here's a unit test for the `queries` method of the `FXRateService` class:


# LLM-generated content at query #2
#--------------------------

```python
def test_FXRate___invert__():
    from datetime import date
    from decimal import Decimal
    from pypara.currencies import Currency

    ccy1 = Currency(code="EUR", numeric="978", name="Euro", exponent=2)
    ccy2 = Currency(code="USD", numeric="840", name="United States Dollar", exponent=2)
    asof = date(2023, 10, 1)
    value = Decimal("1.18")

    fx_rate = FXRate(ccy1, ccy2, asof, value)
    inverted_fx_rate = fx_rate.__invert__()

    assert inverted_fx_rate.ccy1 == ccy2
    assert inverted_fx_rate.ccy2 == ccy1
    assert inverted_fx_rate.date == asof
    assert inverted_fx_rate.value == Decimal("1") / value


# LLM-generated content at query #3
#--------------------------

def test_FXRateService_query():
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            if ccy1 == Currency("EUR") and ccy2 == Currency("USD") and asof == Date(2023, 1, 1):
                return FXRate(ccy1, ccy2, asof, Decimal("1.2"))
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            pass

    service = MockFXRateService()

    # Test successful query
    rate = service.query(Currency("EUR"), Currency("USD"), Date(2023, 1, 1))
    assert rate is not None
    assert rate.ccy1 == Currency("EUR")
    assert rate.ccy2 == Currency("USD")
    assert rate.date == Date(2023, 1, 1)
    assert rate.value == Decimal("1.2")

    # Test non-strict query for non-existent rate
    rate = service.query(Currency("USD"), Currency("EUR"), Date(2023, 1, 1))
    assert rate is None

    # Test strict query for non-existent rate
    try:
        service.query(Currency("USD"), Currency("EUR"), Date(2023, 1, 1), strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == Currency("USD")
        assert e.ccy2 == Currency("EUR")
        assert e.asof == Date(2023, 1, 1)


# LLM-generated content at query #4
#--------------------------

```python
def test_FXRateService_queries():
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            if ccy1 == Currency("EUR") and ccy2 == Currency("USD") and asof == Date(2023, 10, 1):
                return FXRate(ccy1, ccy2, asof, Decimal("1.2"))
            return None

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = MockFXRateService()
    queries = [
        (Currency("EUR"), Currency("USD"), Date(2023, 10, 1)),
        (Currency("GBP"), Currency("USD"), Date(2023, 10, 1)),
    ]
    results = list(service.queries(queries))
    assert results[0] == FXRate(Currency("EUR"), Currency("USD"), Date(2023, 10, 1), Decimal("1.2"))
    assert results[1] is None


# LLM-generated content at query #5
#--------------------------

def test_FXRate___invert__():
    from datetime import date
    from decimal import Decimal
    from pypara.currencies import Currencies

    # Test with EUR/USD rate
    eur_usd_rate = FXRate(Currencies["EUR"], Currencies["USD"], date(2023, 1, 1), Decimal("1.2"))
    inverted_rate = ~eur_usd_rate
    assert inverted_rate.ccy1 == Currencies["USD"]
    assert inverted_rate.ccy2 == Currencies["EUR"]
    assert inverted_rate.date == date(2023, 1, 1)
    assert inverted_rate.value == Decimal("1") / Decimal("1.2")

    # Test with USD/JPY rate
    usd_jpy_rate = FXRate(Currencies["USD"], Currencies["JPY"], date(2023, 1, 1), Decimal("110.5"))
    inverted_rate = ~usd_jpy_rate
    assert inverted_rate.ccy1 == Currencies["JPY"]
    assert inverted_rate.ccy2 == Currencies["USD"]
    assert inverted_rate.date == date(2023, 1, 1)
    assert inverted_rate.value == Decimal("1") / Decimal("110.5")

    # Test with same currency (should be 1)
    eur_eur_rate = FXRate(Currencies["EUR"], Currencies["EUR"], date(2023, 1, 1), Decimal("1"))
    inverted_rate = ~eur_eur_rate
    assert inverted_rate.ccy1 == Currencies["EUR"]
    assert inverted_rate.ccy2 == Currencies["EUR"]
    assert inverted_rate.date == date(2023, 1, 1)
    assert inverted_rate.value == Decimal("1")


# LLM-generated content at query #6
#--------------------------

```python
def test_FXRateService_query():
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            if ccy1.code == "EUR" and ccy2.code == "USD" and asof == Date(2023, 10, 1):
                return FXRate(ccy1, ccy2, asof, Decimal("1.2"))
            return None

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            pass

    service = MockFXRateService()
    eur = Currency("EUR", "Euro", 2)
    usd = Currency("USD", "US Dollar", 2)
    date = Date(2023, 10, 1)

    # Test successful query
    rate = service.query(eur, usd, date)
    assert rate is not None
    assert rate.ccy1 == eur
    assert rate.ccy2 == usd
    assert rate.date == date
    assert rate.value == Decimal("1.2")

    # Test query with non-existent rate
    gbp = Currency("GBP", "British Pound", 2)
    rate = service.query(eur, gbp, date)
    assert rate is None

    # Test strict mode with non-existent rate
    try:
        service.query(eur, gbp, date, strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == eur
        assert e.ccy2 == gbp
        assert e.asof == date


# LLM-generated content at query #7
#--------------------------

```python
def test_FXRate___invert__():
    from datetime import date
    from decimal import Decimal
    from pypara.currencies import Currencies

    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    asof = date.today()
    value = Decimal("2")
    fx_rate = FXRate(ccy1, ccy2, asof, value)

    inverted_fx_rate = fx_rate.__invert__()

    assert inverted_fx_rate.ccy1 == ccy2
    assert inverted_fx_rate.ccy2 == ccy1
    assert inverted_fx_rate.date == asof
    assert inverted_fx_rate.value == Decimal("0.5")


# LLM-generated content at query #8
#--------------------------

```python
def test_FXRateService_query():
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            if ccy1 == Currency("EUR") and ccy2 == Currency("USD") and asof == Date(2023, 10, 1):
                return FXRate(ccy1, ccy2, asof, Decimal("1.2"))
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            pass

    service = MockFXRateService()

    # Test successful query
    rate = service.query(Currency("EUR"), Currency("USD"), Date(2023, 10, 1))
    assert rate == FXRate(Currency("EUR"), Currency("USD"), Date(2023, 10, 1), Decimal("1.2"))

    # Test query with strict=False and no rate found
    rate = service.query(Currency("GBP"), Currency("USD"), Date(2023, 10, 1))
    assert rate is None

    # Test query with strict=True and no rate found
    try:
        service.query(Currency("GBP"), Currency("USD"), Date(2023, 10, 1), strict=True)
    except FXRateLookupError as e:
        assert e.ccy1 == Currency("GBP")
        assert e.ccy2 == Currency("USD")
        assert e.asof == Date(2023, 10, 1)


# LLM-generated content at query #9
#--------------------------

def test_FXRateService_queries():
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            if ccy1.code == "EUR" and ccy2.code == "USD" and asof == Date(2023, 1, 1):
                return FXRate(ccy1, ccy2, asof, Decimal("1.1"))
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
        (Currencies["EUR"], Currencies["GBP"], Date(2023, 1, 1)),
    ]

    # Test non-strict mode
    results = list(service.queries(queries))
    assert len(results) == 3
    assert results[0] == FXRate(Currencies["EUR"], Currencies["USD"], Date(2023, 1, 1), Decimal("1.1"))
    assert results[1] is None
    assert results[2] is None

    # Test strict mode
    try:
        list(service.queries(queries, strict=True))
        assert False, "Should have raised FXRateLookupError"
    except FXRateLookupError:
        pass

    # Test empty queries
    assert list(service.queries([])) == []


# LLM-generated content at query #10
#--------------------------

def test_FXRateService_query():
    # Mock the abstract methods to create a concrete test class
    class TestFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            if ccy1 == Currency("EUR") and ccy2 == Currency("USD") and asof == Date(2023, 1, 1):
                return FXRate(ccy1, ccy2, asof, Decimal("1.1"))
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            pass  # Not needed for this test

    # Create test instance
    service = TestFXRateService()

    # Test successful query
    rate = service.query(Currency("EUR"), Currency("USD"), Date(2023, 1, 1))
    assert rate is not None
    assert rate.ccy1 == Currency("EUR")
    assert rate.ccy2 == Currency("USD")
    assert rate.date == Date(2023, 1, 1)
    assert rate.value == Decimal("1.1")

    # Test non-strict failed query
    rate = service.query(Currency("USD"), Currency("EUR"), Date(2023, 1, 1))
    assert rate is None

    # Test strict failed query
    try:
        service.query(Currency("USD"), Currency("EUR"), Date(2023, 1, 1), strict=True)
        assert False, "Should have raised FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == Currency("USD")
        assert e.ccy2 == Currency("EUR")
        assert e.asof == Date(2023, 1, 1)


# LLM-generated content at query #11
#--------------------------

def test_FXRateService_query():
    # Mock the abstract class to test the query method
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            if ccy1 == Currency("EUR") and ccy2 == Currency("USD") and asof == Date("2023-01-01"):
                return FXRate(ccy1, ccy2, asof, Decimal("1.1"))
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            pass

    service = MockFXRateService()

    # Test successful query
    rate = service.query(Currency("EUR"), Currency("USD"), Date("2023-01-01"))
    assert rate is not None
    assert rate.ccy1 == Currency("EUR")
    assert rate.ccy2 == Currency("USD")
    assert rate.date == Date("2023-01-01")
    assert rate.value == Decimal("1.1")

    # Test non-strict query with no rate
    rate = service.query(Currency("USD"), Currency("EUR"), Date("2023-01-01"))
    assert rate is None

    # Test strict query with no rate
    try:
        service.query(Currency("USD"), Currency("EUR"), Date("2023-01-01"), strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == Currency("USD")
        assert e.ccy2 == Currency("EUR")
        assert e.asof == Date("2023-01-01")


# LLM-generated content at query #12
#--------------------------

def test_FXRateService_query():
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            if ccy1 == Currency("EUR") and ccy2 == Currency("USD") and asof == Date(2023, 1, 1):
                return FXRate(ccy1, ccy2, asof, Decimal("1.1"))
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            pass

    service = MockFXRateService()
    
    # Test successful query
    rate = service.query(Currency("EUR"), Currency("USD"), Date(2023, 1, 1))
    assert rate is not None
    assert rate.ccy1 == Currency("EUR")
    assert rate.ccy2 == Currency("USD")
    assert rate.date == Date(2023, 1, 1)
    assert rate.value == Decimal("1.1")

    # Test non-strict query with no result
    rate = service.query(Currency("USD"), Currency("EUR"), Date(2023, 1, 1))
    assert rate is None

    # Test strict query with no result (should raise)
    try:
        service.query(Currency("USD"), Currency("EUR"), Date(2023, 1, 1), strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == Currency("USD")
        assert e.ccy2 == Currency("EUR")
        assert e.asof == Date(2023, 1, 1)


# LLM-generated content at query #13
#--------------------------

def test_FXRateService_query():
    # Create a mock implementation of FXRateService for testing
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
            return [self.query(q[0], q[1], q[2], strict) for q in queries]

    # Create test currencies
    ccy1 = Currency("EUR", 978, 2)
    ccy2 = Currency("USD", 840, 2)
    ccy3 = Currency("GBP", 826, 2)

    # Create test date
    test_date = Date(2023, 1, 1)

    # Create test FXRate
    test_rate = FXRate(ccy1, ccy2, test_date, Decimal("1.2"))

    # Initialize the mock service with test data
    service = MockFXRateService({
        (ccy1, ccy2, test_date): test_rate
    })

    # Test successful query
    result = service.query(ccy1, ccy2, test_date)
    assert result == test_rate

    # Test query with inverted currencies
    inverted_result = service.query(ccy2, ccy1, test_date)
    assert inverted_result == ~test_rate

    # Test non-existent rate with strict=False
    non_existent = service.query(ccy1, ccy3, test_date)
    assert non_existent is None

    # Test non-existent rate with strict=True
    try:
        service.query(ccy1, ccy3, test_date, strict=True)
        assert False, "Should have raised FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == ccy1
        assert e.ccy2 == ccy3
        assert e.asof == test_date

    # Test same currency query
    same_ccy = service.query(ccy1, ccy1, test_date)
    assert same_ccy == FXRate(ccy1, ccy1, test_date, ONE)


# LLM-generated content at query #14
#--------------------------

```python
def test_FXRateService_queries():
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            return FXRate(ccy1, ccy2, asof, Decimal("1.0"))

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = MockFXRateService()
    queries = [
        (Currency("USD"), Currency("EUR"), Date(2023, 10, 1)),
        (Currency("GBP"), Currency("JPY"), Date(2023, 10, 2))
    ]
    rates = list(service.queries(queries))

    assert len(rates) == 2
    assert rates[0] == FXRate(Currency("USD"), Currency("EUR"), Date(2023, 10, 1), Decimal("1.0"))
    assert rates[1] == FXRate(Currency("GBP"), Currency("JPY"), Date(2023, 10, 2), Decimal("1.0"))


# LLM-generated content at query #15
#--------------------------

def test_FXRateService_queries():
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            if ccy1 == Currency("EUR") and ccy2 == Currency("USD") and asof == Date(2023, 1, 1):
                return FXRate(ccy1, ccy2, asof, Decimal("1.1"))
            if ccy1 == Currency("USD") and ccy2 == Currency("EUR") and asof == Date(2023, 1, 1):
                return FXRate(ccy1, ccy2, asof, Decimal("0.9"))
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

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

    # Test with one valid and one invalid query (non-strict)
    queries = [
        (Currency("EUR"), Currency("USD"), Date(2023, 1, 1)),
        (Currency("GBP"), Currency("USD"), Date(2023, 1, 1)),
    ]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1"))
    assert results[1] is None

    # Test with one valid and one invalid query (strict)
    queries = [
        (Currency("EUR"), Currency("USD"), Date(2023, 1, 1)),
        (Currency("GBP"), Currency("USD"), Date(2023, 1, 1)),
    ]
    with pytest.raises(FXRateLookupError):
        list(service.queries(queries, strict=True))


# LLM-generated content at query #16
#--------------------------

def test_FXRateService_query():
    # Create a mock implementation of FXRateService
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            if ccy1 == Currency("EUR") and ccy2 == Currency("USD") and asof == Date(2023, 1, 1):
                return FXRate(ccy1, ccy2, asof, Decimal("1.1"))
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            pass  # Not needed for this test

    # Set the mock as default service
    FXRateService.default = MockFXRateService()

    # Test successful query
    rate = FXRateService.default.query(Currency("EUR"), Currency("USD"), Date(2023, 1, 1))
    assert rate is not None
    assert rate.ccy1 == Currency("EUR")
    assert rate.ccy2 == Currency("USD")
    assert rate.date == Date(2023, 1, 1)
    assert rate.value == Decimal("1.1")

    # Test non-strict query with no result
    rate = FXRateService.default.query(Currency("USD"), Currency("EUR"), Date(2023, 1, 1))
    assert rate is None

    # Test strict query with no result (should raise)
    try:
        FXRateService.default.query(Currency("USD"), Currency("EUR"), Date(2023, 1, 1), strict=True)
        assert False, "Should have raised FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == Currency("USD")
        assert e.ccy2 == Currency("EUR")
        assert e.asof == Date(2023, 1, 1)

    # Clean up
    FXRateService.default = None


# LLM-generated content at query #17
#--------------------------

```python
def test_FXRateService_query():
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            if ccy1 == Currency("USD") and ccy2 == Currency("EUR") and asof == Date(2023, 10, 1):
                return FXRate(ccy1, ccy2, asof, Decimal("0.85"))
            elif strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            else:
                return None

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            pass

    service = MockFXRateService()

    # Test successful query
    rate = service.query(Currency("USD"), Currency("EUR"), Date(2023, 10, 1))
    assert rate is not None
    assert rate.ccy1 == Currency("USD")
    assert rate.ccy2 == Currency("EUR")
    assert rate.date == Date(2023, 10, 1)
    assert rate.value == Decimal("0.85")

    # Test query returning None in non-strict mode
    rate = service.query(Currency("GBP"), Currency("JPY"), Date(2023, 10, 1))
    assert rate is None

    # Test query raising FXRateLookupError in strict mode
    try:
        service.query(Currency("GBP"), Currency("JPY"), Date(2023, 10, 1), strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == Currency("GBP")
        assert e.ccy2 == Currency("JPY")
        assert e.asof == Date(2023, 10, 1)


# LLM-generated content at query #18
#--------------------------

def test_FXRateService_query():
    from datetime import date
    from decimal import Decimal
    from unittest.mock import Mock, patch

    # Create mock currencies
    ccy1 = Mock(spec=Currency)
    ccy2 = Mock(spec=Currency)
    test_date = date(2023, 1, 1)
    test_rate = Decimal("1.5")

    # Create a mock FXRateService
    mock_service = Mock(spec=FXRateService)
    mock_service.query.return_value = FXRate(ccy1, ccy2, test_date, test_rate)

    # Test successful query
    result = mock_service.query(ccy1, ccy2, test_date)
    assert isinstance(result, FXRate)
    assert result.ccy1 == ccy1
    assert result.ccy2 == ccy2
    assert result.date == test_date
    assert result.value == test_rate

    # Test strict mode with lookup error
    mock_service.query.side_effect = FXRateLookupError(ccy1, ccy2, test_date)
    try:
        mock_service.query(ccy1, ccy2, test_date, strict=True)
        assert False, "Should have raised FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == ccy1
        assert e.ccy2 == ccy2
        assert e.asof == test_date

    # Test non-strict mode returns None
    mock_service.query.side_effect = FXRateLookupError(ccy1, ccy2, test_date)
    result = mock_service.query(ccy1, ccy2, test_date, strict=False)
    assert result is None

    # Test with None service
    with patch.object(FXRateService, 'default', None):
        try:
            FXRateService.default.query(ccy1, ccy2, test_date)
            assert False, "Should have raised AttributeError"
        except AttributeError:
            pass


# LLM-generated content at query #19
#--------------------------

def test_FXRateService_query():
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            if ccy1 == Currency("EUR") and ccy2 == Currency("USD") and asof == Date(2023, 1, 1):
                return FXRate(ccy1, ccy2, asof, Decimal("1.1"))
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            pass

    service = MockFXRateService()

    # Test successful query
    rate = service.query(Currency("EUR"), Currency("USD"), Date(2023, 1, 1))
    assert rate is not None
    assert rate.ccy1 == Currency("EUR")
    assert rate.ccy2 == Currency("USD")
    assert rate.date == Date(2023, 1, 1)
    assert rate.value == Decimal("1.1")

    # Test non-strict query with no result
    assert service.query(Currency("USD"), Currency("EUR"), Date(2023, 1, 1)) is None

    # Test strict query with no result (should raise)
    try:
        service.query(Currency("USD"), Currency("EUR"), Date(2023, 1, 1), strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == Currency("USD")
        assert e.ccy2 == Currency("EUR")
        assert e.asof == Date(2023, 1, 1)


# LLM-generated content at query #20
#--------------------------

```python
def test_FXRateService_queries():
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            return FXRate.of(ccy1, ccy2, asof, Decimal("1.0"))

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = MockFXRateService()
    ccy1 = Currency(code="USD")
    ccy2 = Currency(code="EUR")
    asof = Date(year=2023, month=10, day=1)
    queries = [(ccy1, ccy2, asof), (ccy2, ccy1, asof)]

    results = list(service.queries(queries))

    assert len(results) == 2
    assert isinstance(results[0], FXRate)
    assert isinstance(results[1], FXRate)
    assert results[0].ccy1 == ccy1
    assert results[0].ccy2 == ccy2
    assert results[0].date == asof
    assert results[0].value == Decimal("1.0")
    assert results[1].ccy1 == ccy2
    assert results[1].ccy2 == ccy1
    assert results[1].date == asof
    assert results[1].value == Decimal("1.0")


# LLM-generated content at query #21
#--------------------------

```python
def test_FXRateService_query():
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            if ccy1.code == "EUR" and ccy2.code == "USD" and asof == Date(2023, 10, 1):
                return FXRate(ccy1, ccy2, asof, Decimal("1.2"))
            elif strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            else:
                return None

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            pass

    service = MockFXRateService()
    eur = Currency("EUR", 2)
    usd = Currency("USD", 2)
    date = Date(2023, 10, 1)

    # Test successful query
    rate = service.query(eur, usd, date)
    assert rate == FXRate(eur, usd, date, Decimal("1.2"))

    # Test query with strict=True and rate found
    rate = service.query(eur, usd, date, strict=True)
    assert rate == FXRate(eur, usd, date, Decimal("1.2"))

    # Test query with strict=True and rate not found
    try:
        service.query(usd, eur, date, strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError:
        pass

    # Test query with strict=False and rate not found
    rate = service.query(usd, eur, date)
    assert rate is None


# LLM-generated content at query #22
#--------------------------

def test_FXRateService_query():
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            if ccy1.code == "EUR" and ccy2.code == "USD" and asof == Date(2023, 10, 1):
                return FXRate(ccy1, ccy2, asof, Decimal("1.1"))
            elif strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            pass

    service = MockFXRateService()
    eur = Currency(code="EUR")
    usd = Currency(code="USD")
    date = Date(2023, 10, 1)

    rate = service.query(eur, usd, date)
    assert rate == FXRate(eur, usd, date, Decimal("1.1"))

    rate = service.query(usd, eur, date, strict=True)
    assert rate is None

    try:
        service.query(usd, eur, date, strict=True)
    except FXRateLookupError as e:
        assert e.ccy1 == usd
        assert e.ccy2 == eur
        assert e.asof == date
    else:
        assert False, "Expected FXRateLookupError"


# LLM-generated content at query #23
#--------------------------

```python
def test_FXRateService_query():
    # Mock implementation of FXRateService for testing
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            if ccy1 == Currency("USD") and ccy2 == Currency("EUR") and asof == Date(2023, 10, 1):
                return FXRate(ccy1, ccy2, asof, Decimal("0.85"))
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return []

    service = MockFXRateService()

    # Test successful query
    rate = service.query(Currency("USD"), Currency("EUR"), Date(2023, 10, 1))
    assert rate is not None
    assert rate.ccy1 == Currency("USD")
    assert rate.ccy2 == Currency("EUR")
    assert rate.date == Date(2023, 10, 1)
    assert rate.value == Decimal("0.85")

    # Test query with strict=True and rate not found
    try:
        service.query(Currency("GBP"), Currency("JPY"), Date(2023, 10, 1), strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == Currency("GBP")
        assert e.ccy2 == Currency("JPY")
        assert e.asof == Date(2023, 10, 1)

    # Test query with strict=False and rate not found
    rate = service.query(Currency("GBP"), Currency("JPY"), Date(2023, 10, 1))
    assert rate is None


# LLM-generated content at query #24
#--------------------------

Here's a unit test for the `queries` method of the `FXRateService` class:


# LLM-generated content at query #25
#--------------------------

Here's a unit test for the `queries` method of the `FXRateService` class:


# LLM-generated content at query #26
#--------------------------

```python
def test_FXRateService_queries():
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            return FXRate(ccy1, ccy2, asof, Decimal("1.0"))

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for (ccy1, ccy2, asof) in queries]

    service = MockFXRateService()
    ccy1 = Currency("USD", "US Dollar", "USD")
    ccy2 = Currency("EUR", "Euro", "EUR")
    date = Date(2023, 10, 1)
    queries = [(ccy1, ccy2, date)]
    result = list(service.queries(queries))
    
    assert len(result) == 1
    assert result[0] == FXRate(ccy1, ccy2, date, Decimal("1.0"))


# LLM-generated content at query #27
#--------------------------

```python
def test_FXRateService_query():
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            if ccy1 == Currency("USD") and ccy2 == Currency("EUR") and asof == Date(2023, 10, 1):
                return FXRate(ccy1, ccy2, asof, Decimal("0.85"))
            return None

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            pass

    service = MockFXRateService()
    rate = service.query(Currency("USD"), Currency("EUR"), Date(2023, 10, 1))
    assert rate is not None
    assert rate.ccy1 == Currency("USD")
    assert rate.ccy2 == Currency("EUR")
    assert rate.date == Date(2023, 10, 1)
    assert rate.value == Decimal("0.85")

    no_rate = service.query(Currency("USD"), Currency("GBP"), Date(2023, 10, 1))
    assert no_rate is None

    with pytest.raises(FXRateLookupError):
        service.query(Currency("USD"), Currency("GBP"), Date(2023, 10, 1), strict=True)


# LLM-generated content at query #28
#--------------------------

def test_FXRateService_query():
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            if ccy1 == Currency("EUR") and ccy2 == Currency("USD") and asof == Date(2023, 1, 1):
                return FXRate(ccy1, ccy2, asof, Decimal("1.1"))
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            pass

    service = MockFXRateService()

    # Test successful query
    rate = service.query(Currency("EUR"), Currency("USD"), Date(2023, 1, 1))
    assert rate is not None
    assert rate.ccy1 == Currency("EUR")
    assert rate.ccy2 == Currency("USD")
    assert rate.date == Date(2023, 1, 1)
    assert rate.value == Decimal("1.1")

    # Test non-strict query for non-existent rate
    rate = service.query(Currency("USD"), Currency("EUR"), Date(2023, 1, 1))
    assert rate is None

    # Test strict query for non-existent rate
    try:
        service.query(Currency("USD"), Currency("EUR"), Date(2023, 1, 1), strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == Currency("USD")
        assert e.ccy2 == Currency("EUR")
        assert e.asof == Date(2023, 1, 1)


# LLM-generated content at query #29
#--------------------------

```python
def test_FXRateService_queries():
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            return FXRate(ccy1, ccy2, asof, Decimal("1.2"))

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = MockFXRateService()
    date = Date.today()
    ccy1 = Currency("USD")
    ccy2 = Currency("EUR")
    queries = [(ccy1, ccy2, date)]
    results = list(service.queries(queries))

    assert len(results) == 1
    assert results[0] == FXRate(ccy1, ccy2, date, Decimal("1.2"))


# LLM-generated content at query #30
#--------------------------

Here's a unit test for the `query` method of the `FXRateService` class using pytest:


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_FXRateService_queries():
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            return FXRate(ccy1, ccy2, asof, Decimal("1.2"))

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for (ccy1, ccy2, asof) in queries]

    service = MockFXRateService()
    ccy1 = Currency("USD", "US Dollar", "USD")
    ccy2 = Currency("EUR", "Euro", "EUR")
    date = Date(2023, 10, 1)
    queries = [(ccy1, ccy2, date), (ccy2, ccy1, date)]
    results = list(service.queries(queries))

    assert len(results) == 2
    assert results[0] == FXRate(ccy1, ccy2, date, Decimal("1.2"))
    assert results[1] == FXRate(ccy2, ccy1, date, Decimal("1.2"))


# LLM-generated content at query #2
#--------------------------

```python
class MockFXRateService(FXRateService):
    def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
        return FXRate(ccy1, ccy2, asof, Decimal("1.0")) if ccy1 != ccy2 else FXRate(ccy1, ccy2, asof, ONE)

    def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
        return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

def test_FXRateService_queries():
    service = MockFXRateService()
    ccy1 = Currency("USD", "US Dollar", "USD")
    ccy2 = Currency("EUR", "Euro", "EUR")
    date = Date(2023, 10, 1)
    queries = [(ccy1, ccy2, date), (ccy1, ccy1, date)]
    
    results = list(service.queries(queries))
    
    assert len(results) == 2
    assert results[0] == FXRate(ccy1, ccy2, date, Decimal("1.0"))
    assert results[1] == FXRate(ccy1, ccy1, date, ONE)


# LLM-generated content at query #3
#--------------------------

def test_FXRateService_queries():
    from datetime import date
    from decimal import Decimal
    from typing import List, Optional, Tuple
    from unittest.mock import MagicMock, patch

    # Create mock currencies
    ccy1 = MagicMock(spec=Currency)
    ccy2 = MagicMock(spec=Currency)
    ccy3 = MagicMock(spec=Currency)
    ccy4 = MagicMock(spec=Currency)

    # Create test dates
    date1 = date(2023, 1, 1)
    date2 = date(2023, 1, 2)

    # Create test queries
    query1 = (ccy1, ccy2, date1)
    query2 = (ccy3, ccy4, date2)
    queries = [query1, query2]

    # Create expected FXRate objects
    fx_rate1 = FXRate(ccy1, ccy2, date1, Decimal("1.5"))
    fx_rate2 = FXRate(ccy3, ccy4, date2, Decimal("2.0"))

    # Create a mock FXRateService
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            if (ccy1, ccy2, asof) == query1:
                return fx_rate1
            elif (ccy1, ccy2, asof) == query2:
                return fx_rate2
            return None

        def queries(self, queries: Iterable[Tuple[Currency, Currency, Date]], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(q[0], q[1], q[2], strict) for q in queries]

    service = MockFXRateService()

    # Test non-strict mode
    results = list(service.queries(queries))
    assert results == [fx_rate1, fx_rate2]

    # Test strict mode with all queries found
    results = list(service.queries(queries, strict=True))
    assert results == [fx_rate1, fx_rate2]

    # Test with one missing query in strict mode
    missing_query = (ccy1, ccy3, date1)
    with pytest.raises(FXRateLookupError):
        list(service.queries([query1, missing_query, query2], strict=True))

    # Test with one missing query in non-strict mode
    results = list(service.queries([query1, missing_query, query2]))
    assert results == [fx_rate1, None, fx_rate2]

    # Test with empty queries
    results = list(service.queries([]))
    assert results == []


# LLM-generated content at query #4
#--------------------------

```python
def test_FXRateService_query():
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            if ccy1 == Currency("USD") and ccy2 == Currency("EUR") and asof == Date(2023, 10, 1):
                return FXRate(ccy1, ccy2, asof, Decimal("0.85"))
            elif strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            else:
                return None

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            pass  # Not needed for this test

    service = MockFXRateService()
    
    # Test successful query
    rate = service.query(Currency("USD"), Currency("EUR"), Date(2023, 10, 1))
    assert rate == FXRate(Currency("USD"), Currency("EUR"), Date(2023, 10, 1), Decimal("0.85"))

    # Test query with strict=True and rate not found
    try:
        service.query(Currency("USD"), Currency("GBP"), Date(2023, 10, 1), strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == Currency("USD")
        assert e.ccy2 == Currency("GBP")
        assert e.asof == Date(2023, 10, 1)

    # Test query with strict=False and rate not found
    rate = service.query(Currency("USD"), Currency("GBP"), Date(2023, 10, 1), strict=False)
    assert rate is None


# LLM-generated content at query #5
#--------------------------

```python
def test_FXRateService_query():
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            if ccy1 == Currency("USD") and ccy2 == Currency("EUR") and asof == Date(2023, 10, 1):
                return FXRate(ccy1, ccy2, asof, Decimal("0.85"))
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            pass

    service = MockFXRateService()

    # Test successful query
    rate = service.query(Currency("USD"), Currency("EUR"), Date(2023, 10, 1))
    assert rate == FXRate(Currency("USD"), Currency("EUR"), Date(2023, 10, 1), Decimal("0.85"))

    # Test query with strict mode and rate not found
    try:
        service.query(Currency("GBP"), Currency("JPY"), Date(2023, 10, 1), strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError:
        pass

    # Test query without strict mode and rate not found
    rate = service.query(Currency("GBP"), Currency("JPY"), Date(2023, 10, 1))
    assert rate is None


# LLM-generated content at query #6
#--------------------------

def test_FXRateService_queries():
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            if ccy1 == Currency("EUR") and ccy2 == Currency("USD") and asof == Date(2023, 1, 1):
                return FXRate(ccy1, ccy2, asof, Decimal("1.1"))
            if ccy1 == Currency("USD") and ccy2 == Currency("EUR") and asof == Date(2023, 1, 1):
                return FXRate(ccy1, ccy2, asof, Decimal("0.9"))
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

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

    # Test with one valid and one invalid query (non-strict)
    queries = [
        (Currency("EUR"), Currency("USD"), Date(2023, 1, 1)),
        (Currency("GBP"), Currency("USD"), Date(2023, 1, 1)),
    ]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == FXRate(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), Decimal("1.1"))
    assert results[1] is None

    # Test with one valid and one invalid query (strict)
    queries = [
        (Currency("EUR"), Currency("USD"), Date(2023, 1, 1)),
        (Currency("GBP"), Currency("USD"), Date(2023, 1, 1)),
    ]
    try:
        list(service.queries(queries, strict=True))
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == Currency("GBP")
        assert e.ccy2 == Currency("USD")
        assert e.asof == Date(2023, 1, 1)


# LLM-generated content at query #7
#--------------------------

def test_FXRateService_query():
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            if ccy1 == Currency("EUR") and ccy2 == Currency("USD") and asof == Date(2023, 1, 1):
                return FXRate(ccy1, ccy2, asof, Decimal("1.1"))
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            pass

    service = MockFXRateService()

    # Test successful query
    rate = service.query(Currency("EUR"), Currency("USD"), Date(2023, 1, 1))
    assert rate is not None
    assert rate.ccy1 == Currency("EUR")
    assert rate.ccy2 == Currency("USD")
    assert rate.date == Date(2023, 1, 1)
    assert rate.value == Decimal("1.1")

    # Test non-strict query with no result
    rate = service.query(Currency("USD"), Currency("EUR"), Date(2023, 1, 1))
    assert rate is None

    # Test strict query with no result
    try:
        service.query(Currency("USD"), Currency("EUR"), Date(2023, 1, 1), strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == Currency("USD")
        assert e.ccy2 == Currency("EUR")
        assert e.asof == Date(2023, 1, 1)


# LLM-generated content at query #8
#--------------------------

```python
def test_FXRateService_queries():
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            return FXRate(ccy1, ccy2, asof, Decimal("1.0")) if ccy1 != ccy2 else None

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = MockFXRateService()
    date = Date(2023, 10, 1)
    ccy1 = Currency("USD")
    ccy2 = Currency("EUR")
    ccy3 = Currency("GBP")

    queries = [(ccy1, ccy2, date), (ccy2, ccy3, date), (ccy1, ccy1, date)]
    results = list(service.queries(queries))

    assert len(results) == 3
    assert results[0] == FXRate(ccy1, ccy2, date, Decimal("1.0"))
    assert results[1] == FXRate(ccy2, ccy3, date, Decimal("1.0"))
    assert results[2] is None


# LLM-generated content at query #9
#--------------------------

```python
def test_FXRateService_query():
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            if ccy1 == Currency("EUR") and ccy2 == Currency("USD") and asof == Date(2023, 10, 1):
                return FXRate(ccy1, ccy2, asof, Decimal("1.2"))
            elif strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            else:
                return None

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            raise NotImplementedError

    service = MockFXRateService()

    # Test successful query
    rate = service.query(Currency("EUR"), Currency("USD"), Date(2023, 10, 1))
    assert rate == FXRate(Currency("EUR"), Currency("USD"), Date(2023, 10, 1), Decimal("1.2"))

    # Test query with strict=True and rate found
    rate = service.query(Currency("EUR"), Currency("USD"), Date(2023, 10, 1), strict=True)
    assert rate == FXRate(Currency("EUR"), Currency("USD"), Date(2023, 10, 1), Decimal("1.2"))

    # Test query with strict=True and rate not found
    try:
        service.query(Currency("GBP"), Currency("USD"), Date(2023, 10, 1), strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError:
        pass

    # Test query with strict=False and rate not found
    rate = service.query(Currency("GBP"), Currency("USD"), Date(2023, 10, 1))
    assert rate is None


# LLM-generated content at query #10
#--------------------------

```python
def test_FXRateService_queries():
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            return FXRate.of(ccy1, ccy2, asof, Decimal("1.2"))

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = MockFXRateService()
    ccy1 = Currency("USD")
    ccy2 = Currency("EUR")
    asof = Date.today()
    queries = [(ccy1, ccy2, asof)]
    result = list(service.queries(queries))
    assert len(result) == 1
    assert result[0] == FXRate.of(ccy1, ccy2, asof, Decimal("1.2"))


# LLM-generated content at query #11
#--------------------------

```python
def test_FXRateService_query():
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            if ccy1 == Currency("EUR") and ccy2 == Currency("USD") and asof == Date(2023, 10, 1):
                return FXRate(ccy1, ccy2, asof, Decimal("1.2"))
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            raise NotImplementedError

    service = MockFXRateService()

    # Test successful query
    rate = service.query(Currency("EUR"), Currency("USD"), Date(2023, 10, 1))
    assert rate == FXRate(Currency("EUR"), Currency("USD"), Date(2023, 10, 1), Decimal("1.2"))

    # Test query with strict mode and rate not found
    try:
        service.query(Currency("USD"), Currency("EUR"), Date(2023, 10, 1), strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == Currency("USD")
        assert e.ccy2 == Currency("EUR")
        assert e.asof == Date(2023, 10, 1)

    # Test query without strict mode and rate not found
    rate = service.query(Currency("USD"), Currency("EUR"), Date(2023, 10, 1))
    assert rate is None


# LLM-generated content at query #12
#--------------------------

```python
def test_FXRateService_queries():
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            return FXRate(ccy1, ccy2, asof, Decimal("1.0")) if ccy1 != ccy2 else None

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = MockFXRateService()
    ccy1 = Currency("USD", "US Dollar", "USD", 2)
    ccy2 = Currency("EUR", "Euro", "EUR", 2)
    date = Date(2023, 10, 1)
    queries = [(ccy1, ccy2, date), (ccy1, ccy1, date)]
    results = list(service.queries(queries))
    
    assert len(results) == 2
    assert results[0] == FXRate(ccy1, ccy2, date, Decimal("1.0"))
    assert results[1] is None


# LLM-generated content at query #13
#--------------------------

```python
def test_FXRateService_queries():
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            return FXRate(ccy1, ccy2, asof, Decimal("1.0"))

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(*query) for query in queries]

    service = MockFXRateService()
    queries = [(Currency("USD"), Currency("EUR"), Date(2023, 10, 1))]
    results = list(service.queries(queries))
    
    assert len(results) == 1
    assert isinstance(results[0], FXRate)
    assert results[0].ccy1 == Currency("USD")
    assert results[0].ccy2 == Currency("EUR")
    assert results[0].date == Date(2023, 10, 1)
    assert results[0].value == Decimal("1.0")


# LLM-generated content at query #14
#--------------------------

```python
def test_FXRateService_query():
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            if ccy1 == Currency("USD") and ccy2 == Currency("EUR") and asof == Date(2023, 10, 1):
                return FXRate(ccy1, ccy2, asof, Decimal("0.85"))
            elif strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            else:
                return None

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            pass

    service = MockFXRateService()

    # Test successful query
    result = service.query(Currency("USD"), Currency("EUR"), Date(2023, 10, 1))
    assert result == FXRate(Currency("USD"), Currency("EUR"), Date(2023, 10, 1), Decimal("0.85"))

    # Test query with no result and strict=False
    result = service.query(Currency("GBP"), Currency("JPY"), Date(2023, 10, 1))
    assert result is None

    # Test query with no result and strict=True
    try:
        service.query(Currency("GBP"), Currency("JPY"), Date(2023, 10, 1), strict=True)
    except FXRateLookupError as e:
        assert e.ccy1 == Currency("GBP")
        assert e.ccy2 == Currency("JPY")
        assert e.asof == Date(2023, 10, 1)


# LLM-generated content at query #15
#--------------------------

```python
def test_FXRateService_queries():
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            if ccy1.code == "EUR" and ccy2.code == "USD" and asof == Date(2023, 10, 1):
                return FXRate(ccy1, ccy2, asof, Decimal("1.2"))
            return None

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = MockFXRateService()
    queries = [
        (Currency("EUR"), Currency("USD"), Date(2023, 10, 1)),
        (Currency("USD"), Currency("EUR"), Date(2023, 10, 1)),
    ]
    results = list(service.queries(queries))

    assert results[0] == FXRate(Currency("EUR"), Currency("USD"), Date(2023, 10, 1), Decimal("1.2"))
    assert results[1] is None


# LLM-generated content at query #16
#--------------------------

def test_FXRateService_query():
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            if ccy1 == Currency("EUR") and ccy2 == Currency("USD") and asof == Date(2023, 1, 1):
                return FXRate(ccy1, ccy2, asof, Decimal("1.1"))
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            pass

    service = MockFXRateService()

    # Test successful query
    rate = service.query(Currency("EUR"), Currency("USD"), Date(2023, 1, 1))
    assert rate is not None
    assert rate.ccy1 == Currency("EUR")
    assert rate.ccy2 == Currency("USD")
    assert rate.date == Date(2023, 1, 1)
    assert rate.value == Decimal("1.1")

    # Test non-strict query with no rate
    rate = service.query(Currency("USD"), Currency("EUR"), Date(2023, 1, 1))
    assert rate is None

    # Test strict query with no rate
    try:
        service.query(Currency("USD"), Currency("EUR"), Date(2023, 1, 1), strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == Currency("USD")
        assert e.ccy2 == Currency("EUR")
        assert e.asof == Date(2023, 1, 1)


# LLM-generated content at query #17
#--------------------------

def test_FXRateService_query():
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            if ccy1 == Currency("EUR") and ccy2 == Currency("USD") and asof == Date(2023, 1, 1):
                return FXRate(ccy1, ccy2, asof, Decimal("1.1"))
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            pass

    service = MockFXRateService()

    # Test successful query
    rate = service.query(Currency("EUR"), Currency("USD"), Date(2023, 1, 1))
    assert rate is not None
    assert rate.ccy1 == Currency("EUR")
    assert rate.ccy2 == Currency("USD")
    assert rate.date == Date(2023, 1, 1)
    assert rate.value == Decimal("1.1")

    # Test non-strict query for non-existent rate
    rate = service.query(Currency("USD"), Currency("EUR"), Date(2023, 1, 1))
    assert rate is None

    # Test strict query for non-existent rate
    try:
        service.query(Currency("USD"), Currency("EUR"), Date(2023, 1, 1), strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == Currency("USD")
        assert e.ccy2 == Currency("EUR")
        assert e.asof == Date(2023, 1, 1)


# LLM-generated content at query #18
#--------------------------

```python
def test_FXRateService_queries():
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            return FXRate.of(ccy1, ccy2, asof, Decimal("1.0"))

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = MockFXRateService()
    queries = [(Currency("USD"), Currency("EUR"), Date(2023, 10, 1))]
    results = list(service.queries(queries))

    assert len(results) == 1
    assert results[0].ccy1 == Currency("USD")
    assert results[0].ccy2 == Currency("EUR")
    assert results[0].date == Date(2023, 10, 1)
    assert results[0].value == Decimal("1.0")


# LLM-generated content at query #19
#--------------------------

def test_FXRateService_queries():
    from datetime import date
    from decimal import Decimal
    from unittest.mock import MagicMock, patch

    # Create mock currencies
    ccy1 = MagicMock(spec=Currency)
    ccy2 = MagicMock(spec=Currency)
    ccy3 = MagicMock(spec=Currency)
    ccy4 = MagicMock(spec=Currency)

    # Create test date
    test_date = date(2023, 1, 1)

    # Create test queries
    queries = [
        (ccy1, ccy2, test_date),
        (ccy3, ccy4, test_date),
    ]

    # Create expected FXRate objects
    rate1 = FXRate(ccy1, ccy2, test_date, Decimal("1.5"))
    rate2 = FXRate(ccy3, ccy4, test_date, Decimal("2.0"))

    # Create a mock FXRateService
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if (ccy1, ccy2, asof) == (ccy1, ccy2, test_date):
                return rate1
            elif (ccy1, ccy2, asof) == (ccy3, ccy4, test_date):
                return rate2
            return None

        def queries(self, queries, strict=False):
            return [self.query(q[0], q[1], q[2], strict) for q in queries]

    service = MockFXRateService()

    # Test non-strict mode
    results = list(service.queries(queries))
    assert results == [rate1, rate2]

    # Test strict mode with all queries found
    results = list(service.queries(queries, strict=True))
    assert results == [rate1, rate2]

    # Test with one missing query in strict mode
    missing_query = (ccy1, ccy3, test_date)
    with patch.object(service, 'query', side_effect=[rate1, FXRateLookupError(ccy1, ccy3, test_date)]):
        with pytest.raises(FXRateLookupError):
            list(service.queries([queries[0], missing_query], strict=True))

    # Test with one missing query in non-strict mode
    with patch.object(service, 'query', side_effect=[rate1, None]):
        results = list(service.queries([queries[0], missing_query]))
        assert results == [rate1, None]

    # Test empty queries
    assert list(service.queries([])) == []


# LLM-generated content at query #20
#--------------------------

def test_FXRateService_queries():
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            if ccy1 == Currency("EUR") and ccy2 == Currency("USD") and asof == Date(2023, 1, 1):
                return FXRate(ccy1, ccy2, asof, Decimal("1.1"))
            elif ccy1 == Currency("USD") and ccy2 == Currency("EUR") and asof == Date(2023, 1, 1):
                return FXRate(ccy1, ccy2, asof, Decimal("0.9"))
            elif strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

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

    # Test with non-strict missing rate
    queries = [
        (Currency("EUR"), Currency("GBP"), Date(2023, 1, 1)),
    ]
    results = list(service.queries(queries))
    assert len(results) == 1
    assert results[0] is None

    # Test with strict missing rate
    queries = [
        (Currency("EUR"), Currency("GBP"), Date(2023, 1, 1)),
    ]
    try:
        list(service.queries(queries, strict=True))
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError:
        pass


# LLM-generated content at query #21
#--------------------------

def test_FXRateService_query():
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            if ccy1.code == "EUR" and ccy2.code == "USD" and asof == Date(2023, 1, 1):
                return FXRate(ccy1, ccy2, asof, Decimal("1.1"))
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            pass

    eur = Currency("EUR", "Euro", 2)
    usd = Currency("USD", "US Dollar", 2)
    date = Date(2023, 1, 1)

    # Test successful query
    service = MockFXRateService()
    rate = service.query(eur, usd, date)
    assert rate is not None
    assert rate.ccy1 == eur
    assert rate.ccy2 == usd
    assert rate.date == date
    assert rate.value == Decimal("1.1")

    # Test query with strict=True and rate exists
    rate = service.query(eur, usd, date, strict=True)
    assert rate is not None

    # Test query with strict=False and rate doesn't exist
    rate = service.query(usd, eur, date)
    assert rate is None

    # Test query with strict=True and rate doesn't exist
    try:
        service.query(usd, eur, date, strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == usd
        assert e.ccy2 == eur
        assert e.asof == date


# LLM-generated content at query #22
#--------------------------

```python
def test_FXRateService_queries():
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            if ccy1 == Currency("USD") and ccy2 == Currency("EUR") and asof == Date(2023, 10, 1):
                return FXRate(ccy1, ccy2, asof, Decimal("0.85"))
            return None

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = MockFXRateService()
    queries = [
        (Currency("USD"), Currency("EUR"), Date(2023, 10, 1)),
        (Currency("EUR"), Currency("USD"), Date(2023, 10, 1)),
    ]
    results = list(service.queries(queries))
    assert results[0] == FXRate(Currency("USD"), Currency("EUR"), Date(2023, 10, 1), Decimal("0.85"))
    assert results[1] is None


# LLM-generated content at query #23
#--------------------------

```python
def test_FXRateService_queries():
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            if ccy1 == Currency("EUR") and ccy2 == Currency("USD") and asof == Date(2023, 10, 1):
                return FXRate(ccy1, ccy2, asof, Decimal("1.2"))
            if ccy1 == Currency("GBP") and ccy2 == Currency("USD") and asof == Date(2023, 10, 1):
                return FXRate(ccy1, ccy2, asof, Decimal("1.5"))
            return None

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(*query, strict) for query in queries]

    service = MockFXRateService()
    queries = [
        (Currency("EUR"), Currency("USD"), Date(2023, 10, 1)),
        (Currency("GBP"), Currency("USD"), Date(2023, 10, 1)),
        (Currency("JPY"), Currency("USD"), Date(2023, 10, 1)),
    ]
    rates = list(service.queries(queries))
    assert rates[0] == FXRate(Currency("EUR"), Currency("USD"), Date(2023, 10, 1), Decimal("1.2"))
    assert rates[1] == FXRate(Currency("GBP"), Currency("USD"), Date(2023, 10, 1), Decimal("1.5"))
    assert rates[2] is None


# LLM-generated content at query #24
#--------------------------

```python
def test_FXRateService_query():
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            if ccy1.code == "USD" and ccy2.code == "EUR" and asof == Date(2023, 10, 1):
                return FXRate(ccy1, ccy2, asof, Decimal("0.85"))
            elif strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            else:
                return None

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            pass

    service = MockFXRateService()
    
    # Test successful query
    usd = Currency(code="USD", numeric_code="840", name="US Dollar")
    eur = Currency(code="EUR", numeric_code="978", name="Euro")
    date = Date(2023, 10, 1)
    rate = service.query(usd, eur, date)
    assert rate == FXRate(usd, eur, date, Decimal("0.85"))

    # Test query with strict=True and rate not found
    gbp = Currency(code="GBP", numeric_code="826", name="British Pound")
    try:
        service.query(usd, gbp, date, strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == usd
        assert e.ccy2 == gbp
        assert e.asof == date

    # Test query with strict=False and rate not found
    rate = service.query(usd, gbp, date)
    assert rate is None


# LLM-generated content at query #25
#--------------------------

def test_FXRateService_query():
    from datetime import date
    from decimal import Decimal
    from unittest.mock import Mock, patch

    # Create mock currencies
    ccy1 = Mock(spec=Currency)
    ccy2 = Mock(spec=Currency)
    test_date = date(2023, 1, 1)
    test_rate = Decimal("1.5")

    # Create a mock FXRateService
    mock_service = Mock(spec=FXRateService)
    mock_service.query.return_value = FXRate(ccy1, ccy2, test_date, test_rate)

    # Test successful query
    result = mock_service.query(ccy1, ccy2, test_date)
    assert isinstance(result, FXRate)
    assert result.ccy1 == ccy1
    assert result.ccy2 == ccy2
    assert result.date == test_date
    assert result.value == test_rate

    # Test strict mode with FXRateLookupError
    mock_service.query.side_effect = FXRateLookupError(ccy1, ccy2, test_date)
    try:
        mock_service.query(ccy1, ccy2, test_date, strict=True)
        assert False, "Should have raised FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == ccy1
        assert e.ccy2 == ccy2
        assert e.asof == test_date

    # Test non-strict mode returns None
    mock_service.query.side_effect = FXRateLookupError(ccy1, ccy2, test_date)
    result = mock_service.query(ccy1, ccy2, test_date, strict=False)
    assert result is None

    # Test with concrete implementation (using patch)
    with patch.object(FXRateService, '__abstractmethods__', set()):
        class TestFXRateService(FXRateService):
            def query(self, ccy1, ccy2, asof, strict=False):
                return FXRate(ccy1, ccy2, asof, test_rate)
            
            def queries(self, queries, strict=False):
                return [FXRate(q[0], q[1], q[2], test_rate) for q in queries]

        service = TestFXRateService()
        result = service.query(ccy1, ccy2, test_date)
        assert isinstance(result, FXRate)
        assert result.value == test_rate


# LLM-generated content at query #26
#--------------------------

```python
def test_FXRateService_queries():
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            return FXRate(ccy1, ccy2, asof, Decimal("1.5")) if ccy1 != ccy2 else FXRate(ccy1, ccy2, asof, ONE)

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = MockFXRateService()
    ccy1 = Currency("USD")
    ccy2 = Currency("EUR")
    date = Date(2023, 10, 1)
    queries = [(ccy1, ccy2, date), (ccy1, ccy1, date)]
    results = list(service.queries(queries))

    assert len(results) == 2
    assert results[0] == FXRate(ccy1, ccy2, date, Decimal("1.5"))
    assert results[1] == FXRate(ccy1, ccy1, date, ONE)


# LLM-generated content at query #27
#--------------------------

```python
def test_FXRateService_queries():
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            if ccy1 == Currency("EUR") and ccy2 == Currency("USD") and asof == Date(2023, 10, 1):
                return FXRate(ccy1, ccy2, asof, Decimal("1.2"))
            return None

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = MockFXRateService()
    queries = [(Currency("EUR"), Currency("USD"), Date(2023, 10, 1)),
               (Currency("USD"), Currency("EUR"), Date(2023, 10, 1))]

    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == FXRate(Currency("EUR"), Currency("USD"), Date(2023, 10, 1), Decimal("1.2"))
    assert results[1] is None


# LLM-generated content at query #28
#--------------------------

```python
def test_FXRateService_query():
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            if ccy1 == Currency("USD") and ccy2 == Currency("EUR") and asof == Date(2023, 10, 1):
                return FXRate(ccy1, ccy2, asof, Decimal("0.85"))
            if strict:
                raise FXRateLookupError(ccy1, ccy2, asof)
            return None

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = MockFXRateService()

    # Test successful query
    rate = service.query(Currency("USD"), Currency("EUR"), Date(2023, 10, 1))
    assert rate is not None
    assert rate.ccy1 == Currency("USD")
    assert rate.ccy2 == Currency("EUR")
    assert rate.date == Date(2023, 10, 1)
    assert rate.value == Decimal("0.85")

    # Test failed query with strict=False
    rate = service.query(Currency("USD"), Currency("GBP"), Date(2023, 10, 1))
    assert rate is None

    # Test failed query with strict=True
    try:
        service.query(Currency("USD"), Currency("GBP"), Date(2023, 10, 1), strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError as e:
        assert e.ccy1 == Currency("USD")
        assert e.ccy2 == Currency("GBP")
        assert e.asof == Date(2023, 10, 1)


