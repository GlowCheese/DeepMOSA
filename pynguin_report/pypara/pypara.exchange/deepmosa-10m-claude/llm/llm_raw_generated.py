####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_invert_fx_rate():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    
    nrate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("2"))
    inverted_rate = ~nrate
    
    assert inverted_rate.ccy1 == Currencies["USD"]
    assert inverted_rate.ccy2 == Currencies["EUR"]
    assert inverted_rate.date == datetime.date.today()
    assert inverted_rate.value == Decimal("0.5")


def test_invert_fx_rate_double_invert():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    
    original_rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("2"))
    double_inverted = ~(~original_rate)
    
    assert double_inverted.ccy1 == original_rate.ccy1
    assert double_inverted.ccy2 == original_rate.ccy2
    assert double_inverted.date == original_rate.date
    assert double_inverted.value == original_rate.value


def test_invert_fx_rate_with_decimal_value():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    
    rate = FXRate(Currencies["GBP"], Currencies["JPY"], datetime.date.today(), Decimal("150.5"))
    inverted = ~rate
    
    assert inverted.ccy1 == Currencies["JPY"]
    assert inverted.ccy2 == Currencies["GBP"]
    assert inverted.value == Decimal("1") / Decimal("150.5")


# LLM-generated content at query #2
#--------------------------

```python
def test_fxrateservice_query_is_abstract():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    # Verify that FXRateService cannot be instantiated directly
    try:
        instance = FXRateService()
        assert False, "FXRateService should not be instantiable"
    except TypeError:
        pass


def test_fxrateservice_query_signature():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    from inspect import signature
    
    sig = signature(FXRateService.query)
    params = list(sig.parameters.keys())
    assert params == ['self', 'ccy1', 'ccy2', 'asof', 'strict']
    assert sig.parameters['strict'].default == False


def test_fxrateservice_query_with_mock_implementation():
    from pypara.exchange import FXRateService, FXRate
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    from decimal import Decimal
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return FXRate(ccy1, ccy2, asof, Decimal("1.5"))
        
        def queries(self, queries, strict=False):
            return []
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    test_date = date(2023, 1, 1)
    
    service = MockFXRateService()
    result = service.query(usd, eur, test_date)
    
    assert result is not None
    assert result.ccy1 == usd
    assert result.ccy2 == eur
    assert result.asof == test_date
    assert result.rate == Decimal("1.5")


def test_fxrateservice_query_with_strict_parameter():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if strict:
                return None
            return None
        
        def queries(self, queries, strict=False):
            return []
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    test_date = date(2023, 1, 1)
    
    service = MockFXRateService()
    result_non_strict = service.query(usd, eur, test_date, strict=False)
    result_strict = service.query(usd, eur, test_date, strict=True)
    
    assert result_non_strict is None
    assert result_strict is None


# LLM-generated content at query #3
#--------------------------

```python
def test_queries_returns_iterable_of_fx_rates():
    from decimal import Decimal
    from datetime import date
    from unittest.mock import Mock, MagicMock
    
    mock_service = Mock(spec=FXRateService)
    mock_service.queries = MagicMock()
    
    ccy1 = Mock()
    ccy2 = Mock()
    asof = date(2023, 1, 1)
    
    query1 = (ccy1, ccy2, asof)
    query2 = (ccy2, ccy1, asof)
    queries = [query1, query2]
    
    expected_rate1 = Decimal("1.5")
    expected_rate2 = Decimal("0.67")
    expected_rates = [expected_rate1, expected_rate2]
    
    mock_service.queries.return_value = expected_rates
    
    result = mock_service.queries(queries, strict=False)
    
    assert list(result) == expected_rates
    mock_service.queries.assert_called_once_with(queries, strict=False)


def test_queries_with_strict_mode_enabled():
    from decimal import Decimal
    from datetime import date
    from unittest.mock import Mock, MagicMock
    
    mock_service = Mock(spec=FXRateService)
    mock_service.queries = MagicMock()
    
    ccy1 = Mock()
    ccy2 = Mock()
    asof = date(2023, 1, 1)
    
    queries = [(ccy1, ccy2, asof)]
    expected_rates = [Decimal("1.5")]
    
    mock_service.queries.return_value = expected_rates
    
    result = mock_service.queries(queries, strict=True)
    
    assert list(result) == expected_rates
    mock_service.queries.assert_called_once_with(queries, strict=True)


def test_queries_returns_none_for_missing_rates():
    from datetime import date
    from unittest.mock import Mock, MagicMock
    
    mock_service = Mock(spec=FXRateService)
    mock_service.queries = MagicMock()
    
    ccy1 = Mock()
    ccy2 = Mock()
    asof = date(2023, 1, 1)
    
    queries = [(ccy1, ccy2, asof), (ccy2, ccy1, asof)]
    expected_rates = [None, None]
    
    mock_service.queries.return_value = expected_rates
    
    result = mock_service.queries(queries, strict=False)
    
    assert list(result) == expected_rates


def test_queries_with_empty_iterable():
    from unittest.mock import Mock, MagicMock
    
    mock_service = Mock(spec=FXRateService)
    mock_service.queries = MagicMock()
    
    queries = []
    expected_rates = []
    
    mock_service.queries.return_value = expected_rates
    
    result = mock_service.queries(queries, strict=False)
    
    assert list(result) == expected_rates


def test_queries_with_mixed_results():
    from decimal import Decimal
    from datetime import date
    from unittest.mock import Mock, MagicMock
    
    mock_service = Mock(spec=FXRateService)
    mock_service.queries = MagicMock()
    
    ccy1 = Mock()
    ccy2 = Mock()
    asof = date(2023, 1, 1)
    
    queries = [(ccy1, ccy2, asof), (ccy2, ccy1, asof), (ccy1, ccy2, asof)]
    expected_rates = [Decimal("1.5"), None, Decimal("1.45")]
    
    mock_service.queries.return_value = expected_rates
    
    result = mock_service.queries(queries, strict=False)
    
    assert list(result) == expected_rates
    mock_service.queries.assert_called_once_with(queries, strict=False)


# LLM-generated content at query #4
#--------------------------

```python
def test_queries_with_empty_iterable():
    from abc import ABCMeta
    from decimal import Decimal
    from datetime import date
    
    class MockCurrency:
        def __init__(self, code):
            self.code = code
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return Decimal("1.5")
        
        def queries(self, queries, strict=False):
            return [Decimal("1.5") if q else None for q in queries]
    
    service = MockFXRateService()
    result = list(service.queries([]))
    assert result == []


def test_queries_with_single_query():
    from decimal import Decimal
    from datetime import date
    
    class MockCurrency:
        def __init__(self, code):
            self.code = code
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return Decimal("1.5")
        
        def queries(self, queries, strict=False):
            return [Decimal("1.5") for _ in queries]
    
    service = MockFXRateService()
    ccy1 = MockCurrency("USD")
    ccy2 = MockCurrency("EUR")
    asof = date(2023, 1, 1)
    
    result = list(service.queries([(ccy1, ccy2, asof)]))
    assert len(result) == 1
    assert result[0] == Decimal("1.5")


def test_queries_with_multiple_queries():
    from decimal import Decimal
    from datetime import date
    
    class MockCurrency:
        def __init__(self, code):
            self.code = code
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return Decimal("1.5")
        
        def queries(self, queries, strict=False):
            return [Decimal("1.5"), Decimal("2.0"), Decimal("0.9")]
    
    service = MockFXRateService()
    ccy1 = MockCurrency("USD")
    ccy2 = MockCurrency("EUR")
    asof = date(2023, 1, 1)
    
    queries = [
        (ccy1, ccy2, asof),
        (ccy2, ccy1, asof),
        (ccy1, MockCurrency("GBP"), asof)
    ]
    
    result = list(service.queries(queries))
    assert len(result) == 3
    assert result[0] == Decimal("1.5")
    assert result[1] == Decimal("2.0")
    assert result[2] == Decimal("0.9")


def test_queries_with_strict_false():
    from decimal import Decimal
    from datetime import date
    
    class MockCurrency:
        def __init__(self, code):
            self.code = code
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None if strict else Decimal("1.5")
        
        def queries(self, queries, strict=False):
            return [None, Decimal("1.5"), None] if not strict else [Decimal("1.5")]
    
    service = MockFXRateService()
    ccy1 = MockCurrency("USD")
    ccy2 = MockCurrency("EUR")
    asof = date(2023, 1, 1)
    
    queries = [(ccy1, ccy2, asof), (ccy2, ccy1, asof), (ccy1, ccy2, asof)]
    result = list(service.queries(queries, strict=False))
    assert len(result) == 3


def test_queries_with_strict_true():
    from decimal import Decimal
    from datetime import date
    
    class MockCurrency:
        def __init__(self, code):
            self.code = code
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return Decimal("1.5")
        
        def queries(self, queries, strict=False):
            if strict:
                return [Decimal("1.5") for _ in queries]
            return [None for _ in queries]
    
    service = MockFXRateService()
    ccy1 = MockCurrency("USD")
    ccy2 = MockCurrency("EUR")
    asof = date(2023, 1, 1)
    
    queries = [(ccy1, ccy2, asof)]
    result = list(service.queries(queries, strict=True))
    assert result[0] == Decimal("1.5")


# LLM-generated content at query #5
#--------------------------

```python
def test_fxrateservice_query_is_abstract():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    test_date = date(2023, 1, 1)
    
    try:
        service = FXRateService()
        service.query(usd, eur, test_date)
        assert False, "Should not be able to instantiate abstract class"
    except TypeError:
        assert True


# LLM-generated content at query #6
#--------------------------

```python
def test_fxrateservice_query_is_abstract():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        
        def queries(self, queries, strict=False):
            return []
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    test_date = date(2023, 1, 1)
    
    service = ConcreteFXRateService()
    result = service.query(usd, eur, test_date)
    
    assert result is None


def test_fxrateservice_query_with_strict_false():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        
        def queries(self, queries, strict=False):
            return []
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    gbp = Currency.of("GBP", "British Pounds", 2, CurrencyType.MONEY)
    test_date = date(2023, 6, 15)
    
    service = MockFXRateService()
    result = service.query(usd, gbp, test_date, strict=False)
    
    assert result is None


def test_fxrateservice_query_accepts_currencies_and_date():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    class TestFXRateService(FXRateService):
        def __init__(self):
            self.last_query = None
        
        def query(self, ccy1, ccy2, asof, strict=False):
            self.last_query = (ccy1, ccy2, asof, strict)
            return None
        
        def queries(self, queries, strict=False):
            return []
    
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    aud = Currency.of("AUD", "Australian Dollars", 2, CurrencyType.MONEY)
    test_date = date(2022, 12, 25)
    
    service = TestFXRateService()
    service.query(jpy, aud, test_date, strict=True)
    
    assert service.last_query == (jpy, aud, test_date, True)


# LLM-generated content at query #7
#--------------------------

```python
def test_fxrateservice_query_is_abstract():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        
        def queries(self, queries, strict=False):
            return []
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    test_date = date(2023, 1, 1)
    
    service = ConcreteFXRateService()
    result = service.query(usd, eur, test_date, strict=False)
    
    assert result is None


def test_fxrateservice_query_with_strict_false():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        
        def queries(self, queries, strict=False):
            return []
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    gbp = Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY)
    test_date = date(2023, 6, 15)
    
    service = MockFXRateService()
    result = service.query(usd, gbp, test_date, strict=False)
    
    assert result is None


def test_fxrateservice_query_accepts_currency_pair_and_date():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    class TestFXRateService(FXRateService):
        def __init__(self):
            self.last_query = None
        
        def query(self, ccy1, ccy2, asof, strict=False):
            self.last_query = (ccy1, ccy2, asof, strict)
            return None
        
        def queries(self, queries, strict=False):
            return []
    
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    chf = Currency.of("CHF", "Swiss Franc", 2, CurrencyType.MONEY)
    test_date = date(2024, 3, 20)
    
    service = TestFXRateService()
    service.query(jpy, chf, test_date, strict=True)
    
    assert service.last_query == (jpy, chf, test_date, True)


def test_fxrateservice_query_default_strict_parameter():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    class TestFXRateService(FXRateService):
        def __init__(self):
            self.strict_value = None
        
        def query(self, ccy1, ccy2, asof, strict=False):
            self.strict_value = strict
            return None
        
        def queries(self, queries, strict=False):
            return []
    
    aud = Currency.of("AUD", "Australian Dollar", 2, CurrencyType.MONEY)
    cad = Currency.of("CAD", "Canadian Dollar", 2, CurrencyType.MONEY)
    test_date = date(2024, 1, 1)
    
    service = TestFXRateService()
    service.query(aud, cad, test_date)
    
    assert service.strict_value is False


# LLM-generated content at query #8
#--------------------------

```python
def test_fxrate_constructor():
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


# LLM-generated content at query #9
#--------------------------

```python
def test_fxrate_constructor():
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


def test_fxrate_constructor_unpacking():
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


# LLM-generated content at query #10
#--------------------------

```python
def test_queries_with_empty_iterable():
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            return None
        
        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [None for _ in queries]
    
    service = MockFXRateService()
    result = list(service.queries([]))
    assert result == []


def test_queries_with_single_query():
    from datetime import date
    from decimal import Decimal
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            return Decimal("1.5")
        
        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [Decimal("1.5") for _ in queries]
    
    service = MockFXRateService()
    mock_currency1 = object()
    mock_currency2 = object()
    mock_date = date.today()
    
    result = list(service.queries([(mock_currency1, mock_currency2, mock_date)]))
    assert len(result) == 1
    assert result[0] == Decimal("1.5")


def test_queries_with_multiple_queries():
    from datetime import date
    from decimal import Decimal
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            return None
        
        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [Decimal("1.5"), Decimal("2.0"), Decimal("0.9")]
    
    service = MockFXRateService()
    mock_queries = [
        (object(), object(), date.today()),
        (object(), object(), date.today()),
        (object(), object(), date.today())
    ]
    
    result = list(service.queries(mock_queries))
    assert len(result) == 3
    assert result[0] == Decimal("1.5")
    assert result[1] == Decimal("2.0")
    assert result[2] == Decimal("0.9")


def test_queries_with_strict_false():
    from datetime import date
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            return None
        
        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [None, None]
    
    service = MockFXRateService()
    mock_queries = [(object(), object(), date.today()), (object(), object(), date.today())]
    
    result = list(service.queries(mock_queries, strict=False))
    assert len(result) == 2
    assert result[0] is None
    assert result[1] is None


def test_queries_with_strict_true():
    from datetime import date
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            return None
        
        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            if strict:
                raise LookupError("Rate not found")
            return []
    
    service = MockFXRateService()
    mock_queries = [(object(), object(), date.today())]
    
    try:
        list(service.queries(mock_queries, strict=True))
        assert False, "Expected LookupError"
    except LookupError:
        assert True


# LLM-generated content at query #11
#--------------------------

```python
def test_queries_with_empty_iterable():
    from abc import ABCMeta
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        
        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    service = MockFXRateService()
    result = list(service.queries([]))
    assert result == []


def test_queries_with_single_query():
    from abc import ABCMeta
    from decimal import Decimal
    from datetime import date
    
    class Currency:
        def __init__(self, code):
            self.code = code
    
    class FXRate:
        def __init__(self, rate):
            self.rate = rate
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return FXRate(Decimal("1.5"))
        
        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    service = MockFXRateService()
    usd = Currency("USD")
    eur = Currency("EUR")
    test_date = date(2023, 1, 1)
    
    result = list(service.queries([(usd, eur, test_date)]))
    assert len(result) == 1
    assert result[0].rate == Decimal("1.5")


def test_queries_with_multiple_queries():
    from abc import ABCMeta
    from decimal import Decimal
    from datetime import date
    
    class Currency:
        def __init__(self, code):
            self.code = code
    
    class FXRate:
        def __init__(self, rate):
            self.rate = rate
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return FXRate(Decimal("1.5"))
        
        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    service = MockFXRateService()
    usd = Currency("USD")
    eur = Currency("EUR")
    gbp = Currency("GBP")
    test_date = date(2023, 1, 1)
    
    result = list(service.queries([(usd, eur, test_date), (eur, gbp, test_date), (usd, gbp, test_date)]))
    assert len(result) == 3


def test_queries_with_strict_false():
    from abc import ABCMeta
    from datetime import date
    
    class Currency:
        def __init__(self, code):
            self.code = code
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        
        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    service = MockFXRateService()
    usd = Currency("USD")
    eur = Currency("EUR")
    test_date = date(2023, 1, 1)
    
    result = list(service.queries([(usd, eur, test_date)], strict=False))
    assert len(result) == 1
    assert result[0] is None


def test_queries_returns_iterable():
    from abc import ABCMeta
    from datetime import date
    
    class Currency:
        def __init__(self, code):
            self.code = code
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        
        def queries(self, queries, strict=False):
            return iter([self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries])
    
    service = MockFXRateService()
    usd = Currency("USD")
    eur = Currency("EUR")
    test_date = date(2023, 1, 1)
    
    result = service.queries([(usd, eur, test_date)])
    assert hasattr(result, '__iter__')


# LLM-generated content at query #12
#--------------------------

```python
def test_fxrate_constructor():
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


def test_fxrate_constructor_with_different_currencies():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    
    ccy1 = Currencies["GBP"]
    ccy2 = Currencies["JPY"]
    date = datetime.date(2023, 1, 15)
    value = Decimal("150.5")
    
    rate = FXRate(ccy1, ccy2, date, value)
    
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value


# LLM-generated content at query #13
#--------------------------

```python
def test_fxrate_constructor_basic():
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


def test_fxrate_constructor_immutability():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("2")
    
    rate = FXRate(ccy1, ccy2, date, value)
    
    try:
        rate.value = Decimal("3")
        assert False, "Should not be able to modify immutable FXRate"
    except AttributeError:
        pass


# LLM-generated content at query #14
#--------------------------

```python
def test_fxrate_constructor():
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


# LLM-generated content at query #15
#--------------------------

```python
def test_fxrate_constructor():
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


# LLM-generated content at query #16
#--------------------------

```python
def test_fxrate_constructor():
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


def test_fxrate_constructor_unpacking():
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


def test_fxrate_constructor_with_different_values():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    
    ccy1 = Currencies["GBP"]
    ccy2 = Currencies["JPY"]
    date = datetime.date(2023, 6, 15)
    value = Decimal("150.5")
    
    rate = FXRate(ccy1, ccy2, date, value)
    
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value


# LLM-generated content at query #17
#--------------------------

```python
def test_fxrate_constructor():
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


# LLM-generated content at query #18
#--------------------------

```python
def test_fxrateservice_query_is_abstract():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        
        def queries(self, queries, strict=False):
            return []
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    test_date = date(2023, 1, 1)
    
    service = ConcreteFXRateService()
    result = service.query(usd, eur, test_date, strict=False)
    
    assert result is None


def test_fxrateservice_query_with_strict_false():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        
        def queries(self, queries, strict=False):
            return []
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    gbp = Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY)
    test_date = date(2023, 6, 15)
    
    service = TestFXRateService()
    result = service.query(usd, gbp, test_date, strict=False)
    
    assert result is None


def test_fxrateservice_query_accepts_currencies_and_date():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return (ccy1, ccy2, asof)
        
        def queries(self, queries, strict=False):
            return []
    
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    chf = Currency.of("CHF", "Swiss Franc", 2, CurrencyType.MONEY)
    test_date = date(2024, 12, 31)
    
    service = MockFXRateService()
    result = service.query(jpy, chf, test_date)
    
    assert result == (jpy, chf, test_date)


def test_fxrateservice_query_default_strict_parameter():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    class TrackingFXRateService(FXRateService):
        def __init__(self):
            self.strict_value = None
        
        def query(self, ccy1, ccy2, asof, strict=False):
            self.strict_value = strict
            return None
        
        def queries(self, queries, strict=False):
            return []
    
    cad = Currency.of("CAD", "Canadian Dollar", 2, CurrencyType.MONEY)
    aud = Currency.of("AUD", "Australian Dollar", 2, CurrencyType.MONEY)
    test_date = date(2023, 3, 20)
    
    service = TrackingFXRateService()
    service.query(cad, aud, test_date)
    
    assert service.strict_value is False


def test_fxrateservice_query_with_strict_true():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    class StrictTrackingService(FXRateService):
        def __init__(self):
            self.strict_value = None
        
        def query(self, ccy1, ccy2, asof, strict=False):
            self.strict_value = strict
            return None
        
        def queries(self, queries, strict=False):
            return []
    
    nzd = Currency.of("NZD", "New Zealand Dollar", 2, CurrencyType.MONEY)
    sgd = Currency.of("SGD", "Singapore Dollar", 2, CurrencyType.MONEY)
    test_date = date(2023, 7, 10)
    
    service = StrictTrackingService()
    service.query(nzd, sgd, test_date, strict=True)
    
    assert service.strict_value is True


# LLM-generated content at query #19
#--------------------------

```python
def test_fxrate_constructor():
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


# LLM-generated content at query #20
#--------------------------

```python
def test_fxrate_constructor():
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


def test_fxrate_constructor_with_different_currencies():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    
    ccy1 = Currencies["GBP"]
    ccy2 = Currencies["JPY"]
    date = datetime.date(2023, 1, 15)
    value = Decimal("150.5")
    
    rate = FXRate(ccy1, ccy2, date, value)
    
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value


# LLM-generated content at query #21
#--------------------------

```python
def test_queries_with_single_query():
    from decimal import Decimal
    from datetime import date
    
    class MockCurrency:
        def __init__(self, code):
            self.code = code
    
    class MockFXRate:
        def __init__(self, rate):
            self.rate = rate
    
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return MockFXRate(Decimal("1.5"))
        
        def queries(self, queries, strict=False):
            return [MockFXRate(Decimal("1.5")) for _ in queries]
    
    service = ConcreteFXRateService()
    ccy1 = MockCurrency("USD")
    ccy2 = MockCurrency("EUR")
    test_date = date(2023, 1, 1)
    
    query_list = [(ccy1, ccy2, test_date)]
    results = list(service.queries(query_list))
    
    assert len(results) == 1
    assert results[0].rate == Decimal("1.5")


def test_queries_with_multiple_queries():
    from decimal import Decimal
    from datetime import date
    
    class MockCurrency:
        def __init__(self, code):
            self.code = code
    
    class MockFXRate:
        def __init__(self, rate):
            self.rate = rate
    
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return MockFXRate(Decimal("1.5"))
        
        def queries(self, queries, strict=False):
            rates = [Decimal("1.5"), Decimal("1.2"), Decimal("0.9")]
            return [MockFXRate(rate) for rate in rates[:len(list(queries))]]
    
    service = ConcreteFXRateService()
    ccy1 = MockCurrency("USD")
    ccy2 = MockCurrency("EUR")
    ccy3 = MockCurrency("GBP")
    test_date = date(2023, 1, 1)
    
    query_list = [(ccy1, ccy2, test_date), (ccy1, ccy3, test_date), (ccy2, ccy3, test_date)]
    results = list(service.queries(query_list))
    
    assert len(results) == 3
    assert results[0].rate == Decimal("1.5")
    assert results[1].rate == Decimal("1.2")
    assert results[2].rate == Decimal("0.9")


def test_queries_with_strict_mode():
    from decimal import Decimal
    from datetime import date
    
    class MockCurrency:
        def __init__(self, code):
            self.code = code
    
    class MockFXRate:
        def __init__(self, rate):
            self.rate = rate
    
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return MockFXRate(Decimal("1.5"))
        
        def queries(self, queries, strict=False):
            return [MockFXRate(Decimal("1.5")) for _ in queries]
    
    service = ConcreteFXRateService()
    ccy1 = MockCurrency("USD")
    ccy2 = MockCurrency("EUR")
    test_date = date(2023, 1, 1)
    
    query_list = [(ccy1, ccy2, test_date)]
    results = list(service.queries(query_list, strict=True))
    
    assert len(results) == 1
    assert results[0].rate == Decimal("1.5")


def test_queries_with_empty_list():
    from datetime import date
    
    class MockCurrency:
        def __init__(self, code):
            self.code = code
    
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        
        def queries(self, queries, strict=False):
            return [None for _ in queries]
    
    service = ConcreteFXRateService()
    
    query_list = []
    results = list(service.queries(query_list))
    
    assert len(results) == 0


def test_queries_returns_none_for_missing_rate():
    from datetime import date
    
    class MockCurrency:
        def __init__(self, code):
            self.code = code
    
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        
        def queries(self, queries, strict=False):
            return [None for _ in queries]
    
    service = ConcreteFXRateService()
    ccy1 = MockCurrency("USD")
    ccy2 = MockCurrency("EUR")
    test_date = date(2023, 1, 1)
    
    query_list = [(ccy1, ccy2, test_date)]
    results = list(service.queries(query_list))
    
    assert len(results) == 1
    assert results[0] is None


# LLM-generated content at query #22
#--------------------------

```python
def test_queries_with_empty_iterable():
    from abc import ABCMeta, abstractmethod
    from typing import Tuple, Iterable, Optional
    from datetime import date
    
    class Currency:
        def __init__(self, code: str):
            self.code = code
    
    class FXRate:
        def __init__(self, rate: float):
            self.rate = rate
    
    class MockFXRateService(metaclass=ABCMeta):
        TQuery = Tuple[Currency, Currency, date]
        
        @abstractmethod
        def query(self, ccy1: Currency, ccy2: Currency, asof: date, strict: bool = False) -> Optional[FXRate]:
            pass
        
        @abstractmethod
        def queries(self, queries: Iterable[TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            pass
    
    class TestFXRateService(MockFXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: date, strict: bool = False) -> Optional[FXRate]:
            return FXRate(1.0)
        
        def queries(self, queries: Iterable[TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [FXRate(1.0) if q else None for q in queries]
    
    service = TestFXRateService()
    result = list(service.queries([], strict=False))
    assert result == []


def test_queries_with_multiple_currency_pairs():
    from abc import ABCMeta, abstractmethod
    from typing import Tuple, Iterable, Optional
    from datetime import date
    
    class Currency:
        def __init__(self, code: str):
            self.code = code
    
    class FXRate:
        def __init__(self, rate: float):
            self.rate = rate
    
    class MockFXRateService(metaclass=ABCMeta):
        TQuery = Tuple[Currency, Currency, date]
        
        @abstractmethod
        def query(self, ccy1: Currency, ccy2: Currency, asof: date, strict: bool = False) -> Optional[FXRate]:
            pass
        
        @abstractmethod
        def queries(self, queries: Iterable[TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            pass
    
    class TestFXRateService(MockFXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: date, strict: bool = False) -> Optional[FXRate]:
            return FXRate(1.5)
        
        def queries(self, queries: Iterable[TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [FXRate(1.5) for _ in queries]
    
    service = TestFXRateService()
    usd = Currency("USD")
    eur = Currency("EUR")
    gbp = Currency("GBP")
    test_date = date(2023, 1, 1)
    
    query_list = [(usd, eur, test_date), (eur, gbp, test_date), (usd, gbp, test_date)]
    result = list(service.queries(query_list, strict=False))
    
    assert len(result) == 3
    assert all(isinstance(r, FXRate) for r in result)
    assert all(r.rate == 1.5 for r in result)


def test_queries_with_strict_mode():
    from abc import ABCMeta, abstractmethod
    from typing import Tuple, Iterable, Optional
    from datetime import date
    
    class Currency:
        def __init__(self, code: str):
            self.code = code
    
    class FXRate:
        def __init__(self, rate: float):
            self.rate = rate
    
    class MockFXRateService(metaclass=ABCMeta):
        TQuery = Tuple[Currency, Currency, date]
        
        @abstractmethod
        def query(self, ccy1: Currency, ccy2: Currency, asof: date, strict: bool = False) -> Optional[FXRate]:
            pass
        
        @abstractmethod
        def queries(self, queries: Iterable[TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            pass
    
    class TestFXRateService(MockFXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: date, strict: bool = False) -> Optional[FXRate]:
            return FXRate(1.2)
        
        def queries(self, queries: Iterable[TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [FXRate(1.2) for _ in queries]
    
    service = TestFXRateService()
    usd = Currency("USD")
    eur = Currency("EUR")
    test_date = date(2023, 1, 1)
    
    query_list = [(usd, eur, test_date)]
    result = list(service.queries(query_list, strict=True))
    
    assert len(result) == 1
    assert result[0].rate == 1.2


def test_queries_returns_iterable():
    from abc import ABCMeta, abstractmethod
    from typing import Tuple, Iterable, Optional
    from datetime import date
    
    class Currency:
        def __init__(self, code: str):
            self.code = code
    
    class FXRate:
        def __init__(self, rate: float):
            self.rate = rate
    
    class MockFXRateService(metaclass=ABCMeta):
        TQuery = Tuple[Currency, Currency, date]
        
        @abstractmethod
        def query(self, ccy1: Currency, ccy2: Currency, asof: date, strict: bool = False) -> Optional[FXRate]:
            pass
        
        @abstractmethod
        def queries(self, queries: Iterable[TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            pass
    
    class TestFXRateService(MockFXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: date, strict: bool = False) -> Optional[FXRate]:
            return None
        
        def queries(self, queries: Iterable[TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return (None for _ in queries)
    
    service = TestFXRateService()
    usd = Currency("USD")
    eur = Currency("EUR")
    test_date = date(2023, 1, 1)
    
    query_list = [(usd, eur, test_date), (eur, usd, test_date)]
    result = service.queries(query_list, strict=False)
    
    assert hasattr(result, '__iter__')
    result_list = list(result)
    assert len(result_list) == 2
    assert all(r is None for r in result_list)


# LLM-generated content at query #23
#--------------------------

```python
def test_fxrate_constructor():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    test_date = datetime.date.today()
    value = Decimal("2")
    
    rate = FXRate(ccy1, ccy2, test_date, value)
    
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == test_date
    assert rate.value == value


def test_fxrate_constructor_tuple_unpacking():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    test_date = datetime.date.today()
    value = Decimal("2")
    
    rate = FXRate(ccy1, ccy2, test_date, value)
    unpacked_ccy1, unpacked_ccy2, unpacked_date, unpacked_value = rate
    
    assert unpacked_ccy1 == ccy1
    assert unpacked_ccy2 == ccy2
    assert unpacked_date == test_date
    assert unpacked_value == value


def test_fxrate_constructor_indexed_access():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    test_date = datetime.date.today()
    value = Decimal("2")
    
    rate = FXRate(ccy1, ccy2, test_date, value)
    
    assert rate[0] == ccy1
    assert rate[1] == ccy2
    assert rate[2] == test_date
    assert rate[3] == value


# LLM-generated content at query #24
#--------------------------

```python
def test_fxrateservice_query_is_abstract():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    test_date = date(2023, 1, 1)
    
    try:
        service = FXRateService()
        service.query(usd, eur, test_date)
        assert False, "Should not be able to instantiate abstract class"
    except TypeError:
        assert True


# LLM-generated content at query #25
#--------------------------

```python
def test_fxrate_constructor():
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


# LLM-generated content at query #26
#--------------------------

```python
def test_queries_with_multiple_currency_pairs():
    from abc import ABC, abstractmethod
    from typing import Tuple, Iterable, Optional
    from decimal import Decimal
    from datetime import date
    
    class Currency:
        def __init__(self, code: str):
            self.code = code
        
        def __eq__(self, other):
            return isinstance(other, Currency) and self.code == other.code
        
        def __hash__(self):
            return hash(self.code)
    
    class FXRate(Decimal):
        pass
    
    class ConcreteFXRateService(ABC):
        TQuery = Tuple[Currency, Currency, date]
        
        def queries(self, queries: Iterable[TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            results = []
            rate_map = {
                (Currency("USD"), Currency("EUR"), date(2023, 1, 1)): FXRate("0.92"),
                (Currency("EUR"), Currency("GBP"), date(2023, 1, 1)): FXRate("0.87"),
                (Currency("USD"), Currency("JPY"), date(2023, 1, 1)): FXRate("130.50"),
            }
            
            for ccy1, ccy2, asof in queries:
                key = (ccy1, ccy2, asof)
                if key in rate_map:
                    results.append(rate_map[key])
                elif strict:
                    raise LookupError(f"Rate not found for {key}")
                else:
                    results.append(None)
            
            return results
    
    service = ConcreteFXRateService()
    
    usd = Currency("USD")
    eur = Currency("EUR")
    gbp = Currency("GBP")
    jpy = Currency("JPY")
    test_date = date(2023, 1, 1)
    
    queries = [
        (usd, eur, test_date),
        (eur, gbp, test_date),
        (usd, jpy, test_date),
    ]
    
    results = list(service.queries(queries))
    
    assert len(results) == 3
    assert results[0] == FXRate("0.92")
    assert results[1] == FXRate("0.87")
    assert results[2] == FXRate("130.50")


def test_queries_with_missing_rates_non_strict():
    from abc import ABC
    from typing import Tuple, Iterable, Optional
    from decimal import Decimal
    from datetime import date
    
    class Currency:
        def __init__(self, code: str):
            self.code = code
        
        def __eq__(self, other):
            return isinstance(other, Currency) and self.code == other.code
        
        def __hash__(self):
            return hash(self.code)
    
    class FXRate(Decimal):
        pass
    
    class ConcreteFXRateService(ABC):
        TQuery = Tuple[Currency, Currency, date]
        
        def queries(self, queries: Iterable[TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            results = []
            rate_map = {
                (Currency("USD"), Currency("EUR"), date(2023, 1, 1)): FXRate("0.92"),
            }
            
            for ccy1, ccy2, asof in queries:
                key = (ccy1, ccy2, asof)
                if key in rate_map:
                    results.append(rate_map[key])
                elif strict:
                    raise LookupError(f"Rate not found for {key}")
                else:
                    results.append(None)
            
            return results
    
    service = ConcreteFXRateService()
    
    usd = Currency("USD")
    eur = Currency("EUR")
    gbp = Currency("GBP")
    test_date = date(2023, 1, 1)
    
    queries = [
        (usd, eur, test_date),
        (eur, gbp, test_date),
    ]
    
    results = list(service.queries(queries, strict=False))
    
    assert len(results) == 2
    assert results[0] == FXRate("0.92")
    assert results[1] is None


def test_queries_with_missing_rates_strict():
    from abc import ABC
    from typing import Tuple, Iterable, Optional
    from decimal import Decimal
    from datetime import date
    
    class Currency:
        def __init__(self, code: str):
            self.code = code
        
        def __eq__(self, other):
            return isinstance(other, Currency) and self.code == other.code
        
        def __hash__(self):
            return hash(self.code)
    
    class FXRate(Decimal):
        pass
    
    class ConcreteFXRateService(ABC):
        TQuery = Tuple[Currency, Currency, date]
        
        def queries(self, queries: Iterable[TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            results = []
            rate_map = {
                (Currency("USD"), Currency("EUR"), date(2023, 1, 1)): FXRate("0.92"),
            }
            
            for ccy1, ccy2, asof in queries:
                key = (ccy1, ccy2, asof)
                if key in rate_map:
                    results.append(rate_map[key])
                elif strict:
                    raise LookupError(f"Rate not found for {key}")
                else:
                    results.append(None)
            
            return results
    
    service = ConcreteFXRateService()
    
    usd = Currency("USD")
    eur = Currency("EUR")
    gbp = Currency("GBP")
    test_date = date(2023, 1, 1)
    
    queries = [
        (usd, eur, test_date),
        (eur, gbp, test_date),
    ]
    
    exception_raised = False
    try:
        list(service.queries(queries, strict=True))
    except LookupError:
        exception_raised = True
    
    assert exception_raised is True


def test_queries_with_empty_input():
    from abc import ABC
    from typing import Tuple, Iterable, Optional
    from decimal import Decimal
    from datetime import date
    
    class Currency:
        def __init__(self, code: str):
            self.code = code
    
    class FXRate(Decimal):
        pass
    
    class ConcreteFXRateService(ABC):
        TQuery = Tuple[Currency, Currency, date]
        
        def queries(self, queries: Iterable[TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return []
    
    service = ConcreteFXRateService()
    
    results = list(service.queries([]))
    
    assert len(results) == 0


# LLM-generated content at query #27
#--------------------------

```python
def test_fxrateservice_query_is_abstract():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        
        def queries(self, queries, strict=False):
            return []
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    test_date = date(2024, 1, 1)
    
    service = ConcreteFXRateService()
    result = service.query(usd, eur, test_date, strict=False)
    
    assert result is None


def test_fxrateservice_query_with_strict_false():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        
        def queries(self, queries, strict=False):
            return []
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    gbp = Currency.of("GBP", "British Pounds", 2, CurrencyType.MONEY)
    test_date = date(2024, 6, 15)
    
    service = MockFXRateService()
    result = service.query(usd, gbp, test_date, strict=False)
    
    assert result is None


def test_fxrateservice_query_different_currencies():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if ccy1.code == "JPY" and ccy2.code == "USD":
                return None
            return None
        
        def queries(self, queries, strict=False):
            return []
    
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    test_date = date(2024, 3, 20)
    
    service = TestFXRateService()
    result = service.query(jpy, usd, test_date)
    
    assert result is None


def test_fxrateservice_query_multiple_dates():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    class DateAwareFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        
        def queries(self, queries, strict=False):
            return []
    
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    chf = Currency.of("CHF", "Swiss Francs", 2, CurrencyType.MONEY)
    
    service = DateAwareFXRateService()
    result1 = service.query(eur, chf, date(2024, 1, 1))
    result2 = service.query(eur, chf, date(2024, 12, 31))
    
    assert result1 is None
    assert result2 is None


# LLM-generated content at query #28
#--------------------------

```python
def test_fxrate_constructor():
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


def test_fxrate_constructor_unpacking():
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


def test_fxrate_constructor_different_currencies():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    
    rate = FXRate(Currencies["GBP"], Currencies["JPY"], datetime.date.today(), Decimal("150.5"))
    
    assert rate.ccy1 == Currencies["GBP"]
    assert rate.ccy2 == Currencies["JPY"]
    assert rate.value == Decimal("150.5")


def test_fxrate_constructor_decimal_precision():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    
    precise_value = Decimal("1.23456789")
    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), precise_value)
    
    assert rate.value == precise_value


# LLM-generated content at query #29
#--------------------------

```python
def test_fxrate_constructor():
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


# LLM-generated content at query #30
#--------------------------

```python
def test_queries_with_empty_iterable():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        
        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    service = MockFXRateService()
    result = list(service.queries([]))
    assert result == []


def test_queries_with_single_query():
    from datetime import date
    from decimal import Decimal
    
    class Currency:
        def __init__(self, code):
            self.code = code
    
    class FXRate:
        def __init__(self, rate):
            self.rate = rate
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if ccy1.code == "USD" and ccy2.code == "EUR":
                return FXRate(Decimal("0.85"))
            return None
        
        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    service = MockFXRateService()
    usd = Currency("USD")
    eur = Currency("EUR")
    test_date = date(2023, 1, 1)
    
    result = list(service.queries([(usd, eur, test_date)]))
    assert len(result) == 1
    assert result[0].rate == Decimal("0.85")


def test_queries_with_multiple_queries():
    from datetime import date
    from decimal import Decimal
    
    class Currency:
        def __init__(self, code):
            self.code = code
    
    class FXRate:
        def __init__(self, rate):
            self.rate = rate
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            rates = {
                ("USD", "EUR"): Decimal("0.85"),
                ("USD", "GBP"): Decimal("0.73"),
            }
            key = (ccy1.code, ccy2.code)
            if key in rates:
                return FXRate(rates[key])
            return None
        
        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    service = MockFXRateService()
    usd = Currency("USD")
    eur = Currency("EUR")
    gbp = Currency("GBP")
    test_date = date(2023, 1, 1)
    
    result = list(service.queries([(usd, eur, test_date), (usd, gbp, test_date)]))
    assert len(result) == 2
    assert result[0].rate == Decimal("0.85")
    assert result[1].rate == Decimal("0.73")


def test_queries_with_not_found_rate():
    from datetime import date
    
    class Currency:
        def __init__(self, code):
            self.code = code
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        
        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    service = MockFXRateService()
    usd = Currency("USD")
    jpy = Currency("JPY")
    test_date = date(2023, 1, 1)
    
    result = list(service.queries([(usd, jpy, test_date)]))
    assert len(result) == 1
    assert result[0] is None


def test_queries_with_strict_mode():
    from datetime import date
    
    class Currency:
        def __init__(self, code):
            self.code = code
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if strict and ccy1.code == "XXX":
                raise LookupError(f"Rate not found for {ccy1.code}/{ccy2.code}")
            return None
        
        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    service = MockFXRateService()
    xxx = Currency("XXX")
    yyy = Currency("YYY")
    test_date = date(2023, 1, 1)
    
    try:
        list(service.queries([(xxx, yyy, test_date)], strict=True))
        assert False, "Expected LookupError"
    except LookupError:
        assert True


# LLM-generated content at query #31
#--------------------------

```python
def test_fxrate_constructor():
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


def test_fxrate_constructor_unpacking():
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


# LLM-generated content at query #32
#--------------------------

```python
def test_fxrateservice_query_is_abstract():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        
        def queries(self, queries, strict=False):
            return []
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof_date = date(2024, 1, 1)
    
    service = ConcreteFXRateService()
    result = service.query(usd, eur, asof_date, strict=False)
    
    assert result is None


def test_fxrateservice_query_with_strict_false():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        
        def queries(self, queries, strict=False):
            return []
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    gbp = Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY)
    asof_date = date(2024, 6, 15)
    
    service = ConcreteFXRateService()
    result = service.query(usd, gbp, asof_date, strict=False)
    
    assert result is None


def test_fxrateservice_query_accepts_currencies_and_date():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    class TestFXRateService(FXRateService):
        def __init__(self):
            self.last_query = None
        
        def query(self, ccy1, ccy2, asof, strict=False):
            self.last_query = (ccy1, ccy2, asof, strict)
            return None
        
        def queries(self, queries, strict=False):
            return []
    
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    chf = Currency.of("CHF", "Swiss Franc", 2, CurrencyType.MONEY)
    test_date = date(2023, 12, 25)
    
    service = TestFXRateService()
    service.query(jpy, chf, test_date, strict=True)
    
    assert service.last_query == (jpy, chf, test_date, True)


def test_fxrateservice_query_default_strict_parameter():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    class TestFXRateService(FXRateService):
        def __init__(self):
            self.strict_value = None
        
        def query(self, ccy1, ccy2, asof, strict=False):
            self.strict_value = strict
            return None
        
        def queries(self, queries, strict=False):
            return []
    
    aud = Currency.of("AUD", "Australian Dollar", 2, CurrencyType.MONEY)
    nzd = Currency.of("NZD", "New Zealand Dollar", 2, CurrencyType.MONEY)
    
    service = TestFXRateService()
    service.query(aud, nzd, date(2024, 1, 1))
    
    assert service.strict_value is False


# LLM-generated content at query #33
#--------------------------

```python
def test_fxrate_constructor():
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


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_fxrateservice_query_is_abstract():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        
        def queries(self, queries, strict=False):
            return []
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    test_date = date(2023, 1, 1)
    
    service = ConcreteFXRateService()
    result = service.query(usd, eur, test_date, strict=False)
    
    assert result is None


def test_fxrateservice_query_with_strict_false():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        
        def queries(self, queries, strict=False):
            return []
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    gbp = Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY)
    test_date = date(2023, 6, 15)
    
    service = TestFXRateService()
    result = service.query(usd, gbp, test_date, strict=False)
    
    assert result is None


def test_fxrateservice_query_accepts_strict_parameter():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    class TestFXRateService(FXRateService):
        def __init__(self):
            self.strict_value_received = None
        
        def query(self, ccy1, ccy2, asof, strict=False):
            self.strict_value_received = strict
            return None
        
        def queries(self, queries, strict=False):
            return []
    
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    cad = Currency.of("CAD", "Canadian Dollar", 2, CurrencyType.MONEY)
    test_date = date(2023, 12, 25)
    
    service = TestFXRateService()
    service.query(jpy, cad, test_date, strict=True)
    
    assert service.strict_value_received is True


def test_fxrateservice_query_with_different_currencies():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    class TestFXRateService(FXRateService):
        def __init__(self):
            self.last_query = None
        
        def query(self, ccy1, ccy2, asof, strict=False):
            self.last_query = (ccy1, ccy2, asof)
            return None
        
        def queries(self, queries, strict=False):
            return []
    
    chf = Currency.of("CHF", "Swiss Franc", 2, CurrencyType.MONEY)
    aud = Currency.of("AUD", "Australian Dollar", 2, CurrencyType.MONEY)
    test_date = date(2024, 1, 1)
    
    service = TestFXRateService()
    service.query(chf, aud, test_date)
    
    assert service.last_query[0] == chf
    assert service.last_query[1] == aud
    assert service.last_query[2] == test_date


# LLM-generated content at query #2
#--------------------------

```python
def test_queries_with_empty_iterable():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        
        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    service = MockFXRateService()
    result = list(service.queries([]))
    assert result == []


def test_queries_with_single_query():
    from datetime import date
    from decimal import Decimal
    
    class MockCurrency:
        def __init__(self, code):
            self.code = code
    
    class MockFXRate:
        def __init__(self, rate):
            self.rate = rate
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return MockFXRate(Decimal("1.5"))
        
        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    service = MockFXRateService()
    ccy1 = MockCurrency("USD")
    ccy2 = MockCurrency("EUR")
    asof = date(2023, 1, 1)
    
    result = list(service.queries([(ccy1, ccy2, asof)]))
    assert len(result) == 1
    assert result[0].rate == Decimal("1.5")


def test_queries_with_multiple_queries():
    from datetime import date
    from decimal import Decimal
    
    class MockCurrency:
        def __init__(self, code):
            self.code = code
    
    class MockFXRate:
        def __init__(self, rate):
            self.rate = rate
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return MockFXRate(Decimal("1.5"))
        
        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    service = MockFXRateService()
    ccy1 = MockCurrency("USD")
    ccy2 = MockCurrency("EUR")
    ccy3 = MockCurrency("GBP")
    asof1 = date(2023, 1, 1)
    asof2 = date(2023, 1, 2)
    
    queries = [(ccy1, ccy2, asof1), (ccy1, ccy3, asof2)]
    result = list(service.queries(queries))
    assert len(result) == 2


def test_queries_with_strict_mode():
    from datetime import date
    
    class MockCurrency:
        def __init__(self, code):
            self.code = code
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if strict:
                raise LookupError("Rate not found")
            return None
        
        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    service = MockFXRateService()
    ccy1 = MockCurrency("USD")
    ccy2 = MockCurrency("EUR")
    asof = date(2023, 1, 1)
    
    try:
        list(service.queries([(ccy1, ccy2, asof)], strict=True))
        assert False, "Expected LookupError"
    except LookupError:
        assert True


def test_queries_returns_iterable():
    from datetime import date
    
    class MockCurrency:
        def __init__(self, code):
            self.code = code
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        
        def queries(self, queries, strict=False):
            for ccy1, ccy2, asof in queries:
                yield self.query(ccy1, ccy2, asof, strict)
    
    service = MockFXRateService()
    ccy1 = MockCurrency("USD")
    ccy2 = MockCurrency("EUR")
    asof = date(2023, 1, 1)
    
    result = service.queries([(ccy1, ccy2, asof)])
    assert hasattr(result, '__iter__')


# LLM-generated content at query #3
#--------------------------

```python
def test_fxrateservice_query_is_abstract():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        
        def queries(self, queries, strict=False):
            return []
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    service = ConcreteFXRateService()
    result = service.query(usd, eur, date(2023, 1, 1))
    
    assert result is None


def test_fxrateservice_query_with_strict_false():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        
        def queries(self, queries, strict=False):
            return []
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    service = ConcreteFXRateService()
    result = service.query(usd, jpy, date(2023, 6, 15), strict=False)
    
    assert result is None


def test_fxrateservice_query_with_strict_true():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if strict:
                raise LookupError("Rate not found")
            return None
        
        def queries(self, queries, strict=False):
            return []
    
    gbp = Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY)
    chf = Currency.of("CHF", "Swiss Franc", 2, CurrencyType.MONEY)
    service = ConcreteFXRateService()
    
    try:
        service.query(gbp, chf, date(2023, 12, 25), strict=True)
        assert False, "Should have raised LookupError"
    except LookupError:
        assert True


def test_fxrateservice_query_returns_fxrate():
    from pypara.exchange import FXRateService, FXRate
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    from decimal import Decimal
    
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return FXRate(ccy1, ccy2, asof, Decimal("1.25"))
        
        def queries(self, queries, strict=False):
            return []
    
    aud = Currency.of("AUD", "Australian Dollar", 2, CurrencyType.MONEY)
    nzd = Currency.of("NZD", "New Zealand Dollar", 2, CurrencyType.MONEY)
    service = ConcreteFXRateService()
    result = service.query(aud, nzd, date(2023, 3, 10))
    
    assert result is not None
    assert result.ccy1 == aud
    assert result.ccy2 == nzd
    assert result.asof == date(2023, 3, 10)
    assert result.rate == Decimal("1.25")


def test_fxrateservice_query_default_strict_parameter():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        
        def queries(self, queries, strict=False):
            return []
    
    cad = Currency.of("CAD", "Canadian Dollar", 2, CurrencyType.MONEY)
    mxn = Currency.of("MXN", "Mexican Peso", 2, CurrencyType.MONEY)
    service = ConcreteFXRateService()
    result = service.query(cad, mxn, date(2023, 9, 1))
    
    assert result is None


# LLM-generated content at query #4
#--------------------------

```python
def test_fxrate_constructor():
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


def test_fxrate_constructor_unpacking():
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


def test_fxrate_constructor_immutability():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("2")
    
    rate = FXRate(ccy1, ccy2, date, value)
    
    try:
        rate.value = Decimal("3")
        assert False, "Expected AttributeError"
    except AttributeError:
        pass


# LLM-generated content at query #5
#--------------------------

```python
def test_fxrateservice_query_is_abstract():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    try:
        FXRateService()
        assert False, "Should not be able to instantiate abstract class"
    except TypeError:
        pass


def test_fxrateservice_query_method_exists():
    from pypara.exchange import FXRateService
    from inspect import isabstract
    
    assert hasattr(FXRateService, 'query')
    assert isabstract(FXRateService)


def test_fxrateservice_query_signature():
    from pypara.exchange import FXRateService
    from inspect import signature
    
    sig = signature(FXRateService.query)
    params = list(sig.parameters.keys())
    
    assert 'self' in params
    assert 'ccy1' in params
    assert 'ccy2' in params
    assert 'asof' in params
    assert 'strict' in params


def test_fxrateservice_query_default_strict_parameter():
    from pypara.exchange import FXRateService
    from inspect import signature
    
    sig = signature(FXRateService.query)
    strict_param = sig.parameters['strict']
    
    assert strict_param.default is False


# LLM-generated content at query #6
#--------------------------

```python
def test_queries_with_empty_iterable():
    from abc import ABCMeta
    from collections.abc import Iterable
    from datetime import date
    from decimal import Decimal
    
    class MockCurrency:
        def __init__(self, code: str):
            self.code = code
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        
        def queries(self, queries, strict=False):
            return [None for _ in queries]
    
    service = MockFXRateService()
    result = list(service.queries([]))
    assert result == []


def test_queries_with_single_query():
    from abc import ABCMeta
    from datetime import date
    from decimal import Decimal
    
    class MockCurrency:
        def __init__(self, code: str):
            self.code = code
    
    class MockFXRate:
        def __init__(self, rate: Decimal):
            self.rate = rate
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return MockFXRate(Decimal("1.5"))
        
        def queries(self, queries, strict=False):
            return [MockFXRate(Decimal("1.5")) for _ in queries]
    
    ccy1 = MockCurrency("USD")
    ccy2 = MockCurrency("EUR")
    test_date = date(2023, 1, 1)
    
    service = MockFXRateService()
    result = list(service.queries([(ccy1, ccy2, test_date)]))
    assert len(result) == 1
    assert result[0].rate == Decimal("1.5")


def test_queries_with_multiple_queries():
    from abc import ABCMeta
    from datetime import date
    from decimal import Decimal
    
    class MockCurrency:
        def __init__(self, code: str):
            self.code = code
    
    class MockFXRate:
        def __init__(self, rate: Decimal):
            self.rate = rate
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        
        def queries(self, queries, strict=False):
            return [MockFXRate(Decimal("1.5")), MockFXRate(Decimal("0.85")), None]
    
    ccy1 = MockCurrency("USD")
    ccy2 = MockCurrency("EUR")
    ccy3 = MockCurrency("GBP")
    test_date = date(2023, 1, 1)
    
    service = MockFXRateService()
    result = list(service.queries([
        (ccy1, ccy2, test_date),
        (ccy1, ccy3, test_date),
        (ccy2, ccy3, test_date)
    ]))
    assert len(result) == 3
    assert result[0].rate == Decimal("1.5")
    assert result[1].rate == Decimal("0.85")
    assert result[2] is None


def test_queries_with_strict_mode_false():
    from abc import ABCMeta
    from datetime import date
    
    class MockCurrency:
        def __init__(self, code: str):
            self.code = code
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        
        def queries(self, queries, strict=False):
            return [None for _ in queries]
    
    ccy1 = MockCurrency("USD")
    ccy2 = MockCurrency("EUR")
    test_date = date(2023, 1, 1)
    
    service = MockFXRateService()
    result = list(service.queries([(ccy1, ccy2, test_date)], strict=False))
    assert result == [None]


def test_queries_returns_iterable():
    from abc import ABCMeta
    from datetime import date
    from collections.abc import Iterable
    
    class MockCurrency:
        def __init__(self, code: str):
            self.code = code
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        
        def queries(self, queries, strict=False):
            return iter([None for _ in queries])
    
    ccy1 = MockCurrency("USD")
    ccy2 = MockCurrency("EUR")
    test_date = date(2023, 1, 1)
    
    service = MockFXRateService()
    result = service.queries([(ccy1, ccy2, test_date)])
    assert isinstance(result, Iterable)


# LLM-generated content at query #7
#--------------------------

```python
def test_fxrate_constructor():
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


def test_fxrate_constructor_unpacking():
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


# LLM-generated content at query #8
#--------------------------

```python
def test_fxrateservice_query_is_abstract():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    # FXRateService is abstract and cannot be instantiated directly
    try:
        service = FXRateService()
        assert False, "FXRateService should not be instantiable"
    except TypeError:
        pass


def test_fxrateservice_query_with_concrete_implementation():
    from pypara.exchange import FXRateService, FXRate
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    from decimal import Decimal
    from typing import Iterable, Optional, Tuple
    
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: date, strict: bool = False) -> Optional[FXRate]:
            if ccy1.code == "USD" and ccy2.code == "EUR":
                return FXRate(ccy1, ccy2, asof, Decimal("0.85"))
            elif strict:
                raise LookupError(f"Rate not found for {ccy1.code}/{ccy2.code}")
            return None
        
        def queries(self, queries: Iterable[Tuple[Currency, Currency, date]], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    gbp = Currency.of("GBP", "British Pounds", 2, CurrencyType.MONEY)
    test_date = date(2023, 1, 1)
    
    service = ConcreteFXRateService()
    
    rate = service.query(usd, eur, test_date, strict=False)
    assert rate is not None
    assert rate.ccy1 == usd
    assert rate.ccy2 == eur
    assert rate.asof == test_date


def test_fxrateservice_query_not_found():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    from typing import Iterable, Optional, Tuple
    
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: date, strict: bool = False) -> Optional[FXRate]:
            return None
        
        def queries(self, queries: Iterable[Tuple[Currency, Currency, date]], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    test_date = date(2023, 1, 1)
    
    service = ConcreteFXRateService()
    
    rate = service.query(usd, eur, test_date, strict=False)
    assert rate is None


def test_fxrateservice_query_strict_mode():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    from typing import Iterable, Optional, Tuple
    
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: date, strict: bool = False) -> Optional[FXRate]:
            if strict:
                raise LookupError(f"Rate not found for {ccy1.code}/{ccy2.code}")
            return None
        
        def queries(self, queries: Iterable[Tuple[Currency, Currency, date]], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    test_date = date(2023, 1, 1)
    
    service = ConcreteFXRateService()
    
    try:
        service.query(usd, eur, test_date, strict=True)
        assert False, "Should raise LookupError"
    except LookupError:
        pass


# LLM-generated content at query #9
#--------------------------

```python
def test_queries_with_single_query():
    from decimal import Decimal
    from datetime import date
    
    class MockCurrency:
        def __init__(self, code):
            self.code = code
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if ccy1.code == "USD" and ccy2.code == "EUR":
                return Decimal("0.85")
            return None
        
        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    service = MockFXRateService()
    ccy1 = MockCurrency("USD")
    ccy2 = MockCurrency("EUR")
    test_date = date(2023, 1, 1)
    
    result = list(service.queries([(ccy1, ccy2, test_date)]))
    assert len(result) == 1
    assert result[0] == Decimal("0.85")


def test_queries_with_multiple_queries():
    from decimal import Decimal
    from datetime import date
    
    class MockCurrency:
        def __init__(self, code):
            self.code = code
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            rates = {
                ("USD", "EUR"): Decimal("0.85"),
                ("USD", "GBP"): Decimal("0.73"),
                ("EUR", "GBP"): Decimal("0.86")
            }
            return rates.get((ccy1.code, ccy2.code))
        
        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    service = MockFXRateService()
    ccy_usd = MockCurrency("USD")
    ccy_eur = MockCurrency("EUR")
    ccy_gbp = MockCurrency("GBP")
    test_date = date(2023, 1, 1)
    
    queries = [(ccy_usd, ccy_eur, test_date), (ccy_usd, ccy_gbp, test_date), (ccy_eur, ccy_gbp, test_date)]
    result = list(service.queries(queries))
    
    assert len(result) == 3
    assert result[0] == Decimal("0.85")
    assert result[1] == Decimal("0.73")
    assert result[2] == Decimal("0.86")


def test_queries_with_none_results():
    from datetime import date
    
    class MockCurrency:
        def __init__(self, code):
            self.code = code
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        
        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    service = MockFXRateService()
    ccy1 = MockCurrency("USD")
    ccy2 = MockCurrency("XYZ")
    test_date = date(2023, 1, 1)
    
    result = list(service.queries([(ccy1, ccy2, test_date)]))
    assert len(result) == 1
    assert result[0] is None


def test_queries_with_strict_mode():
    from datetime import date
    
    class MockCurrency:
        def __init__(self, code):
            self.code = code
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if strict and ccy1.code == "INVALID":
                raise LookupError("Rate not found")
            return None
        
        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    service = MockFXRateService()
    ccy_invalid = MockCurrency("INVALID")
    ccy_usd = MockCurrency("USD")
    test_date = date(2023, 1, 1)
    
    try:
        list(service.queries([(ccy_invalid, ccy_usd, test_date)], strict=True))
        assert False, "Expected LookupError"
    except LookupError:
        assert True


def test_queries_returns_iterable():
    from datetime import date
    
    class MockCurrency:
        def __init__(self, code):
            self.code = code
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        
        def queries(self, queries, strict=False):
            return (self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries)
    
    service = MockFXRateService()
    ccy1 = MockCurrency("USD")
    ccy2 = MockCurrency("EUR")
    test_date = date(2023, 1, 1)
    
    result = service.queries([(ccy1, ccy2, test_date)])
    assert hasattr(result, '__iter__')


# LLM-generated content at query #10
#--------------------------

```python
def test_queries_with_single_query():
    from decimal import Decimal
    from datetime import date
    from unittest.mock import Mock, MagicMock
    
    mock_service = Mock(spec=FXRateService)
    mock_service.queries = MagicMock(return_value=[Decimal('1.5')])
    
    ccy1 = Mock()
    ccy2 = Mock()
    asof = date(2023, 1, 1)
    query = (ccy1, ccy2, asof)
    
    result = list(mock_service.queries([query], strict=False))
    
    assert len(result) == 1
    assert result[0] == Decimal('1.5')
    mock_service.queries.assert_called_once_with([query], strict=False)


def test_queries_with_multiple_queries():
    from decimal import Decimal
    from datetime import date
    from unittest.mock import Mock, MagicMock
    
    mock_service = Mock(spec=FXRateService)
    mock_service.queries = MagicMock(return_value=[Decimal('1.5'), Decimal('2.0'), Decimal('0.8')])
    
    ccy1 = Mock()
    ccy2 = Mock()
    ccy3 = Mock()
    asof1 = date(2023, 1, 1)
    asof2 = date(2023, 1, 2)
    asof3 = date(2023, 1, 3)
    
    queries = [(ccy1, ccy2, asof1), (ccy2, ccy3, asof2), (ccy3, ccy1, asof3)]
    
    result = list(mock_service.queries(queries, strict=False))
    
    assert len(result) == 3
    assert result[0] == Decimal('1.5')
    assert result[1] == Decimal('2.0')
    assert result[2] == Decimal('0.8')
    mock_service.queries.assert_called_once_with(queries, strict=False)


def test_queries_with_none_values():
    from datetime import date
    from unittest.mock import Mock, MagicMock
    
    mock_service = Mock(spec=FXRateService)
    mock_service.queries = MagicMock(return_value=[None, None, None])
    
    ccy1 = Mock()
    ccy2 = Mock()
    queries = [(ccy1, ccy2, date(2023, 1, 1)), (ccy1, ccy2, date(2023, 1, 2)), (ccy1, ccy2, date(2023, 1, 3))]
    
    result = list(mock_service.queries(queries, strict=False))
    
    assert len(result) == 3
    assert all(r is None for r in result)


def test_queries_strict_mode():
    from datetime import date
    from unittest.mock import Mock, MagicMock
    
    mock_service = Mock(spec=FXRateService)
    mock_service.queries = MagicMock(side_effect=LookupError("Rate not found"))
    
    ccy1 = Mock()
    ccy2 = Mock()
    queries = [(ccy1, ccy2, date(2023, 1, 1))]
    
    try:
        list(mock_service.queries(queries, strict=True))
        assert False, "Expected LookupError to be raised"
    except LookupError as e:
        assert str(e) == "Rate not found"


def test_queries_empty_iterable():
    from unittest.mock import Mock, MagicMock
    
    mock_service = Mock(spec=FXRateService)
    mock_service.queries = MagicMock(return_value=[])
    
    result = list(mock_service.queries([], strict=False))
    
    assert len(result) == 0
    mock_service.queries.assert_called_once_with([], strict=False)


def test_queries_default_strict_parameter():
    from decimal import Decimal
    from datetime import date
    from unittest.mock import Mock, MagicMock
    
    mock_service = Mock(spec=FXRateService)
    mock_service.queries = MagicMock(return_value=[Decimal('1.5')])
    
    ccy1 = Mock()
    ccy2 = Mock()
    queries = [(ccy1, ccy2, date(2023, 1, 1))]
    
    result = list(mock_service.queries(queries))
    
    assert len(result) == 1
    mock_service.queries.assert_called_once_with(queries, strict=False)


# LLM-generated content at query #11
#--------------------------

```python
def test_fxrate_constructor():
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


# LLM-generated content at query #12
#--------------------------

```python
def test_fxrate_constructor():
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


# LLM-generated content at query #13
#--------------------------

```python
def test_fxrate_constructor():
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


def test_fxrate_constructor_unpacking():
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


def test_fxrate_constructor_with_different_values():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    
    ccy1 = Currencies["GBP"]
    ccy2 = Currencies["JPY"]
    date = datetime.date(2023, 6, 15)
    value = Decimal("150.5")
    
    rate = FXRate(ccy1, ccy2, date, value)
    
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value


# LLM-generated content at query #14
#--------------------------

```python
def test_fxrate_constructor():
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


# LLM-generated content at query #15
#--------------------------

```python
def test_fxrate_constructor():
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


def test_fxrate_constructor_unpacking():
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


# LLM-generated content at query #16
#--------------------------

```python
def test_fxrate_constructor():
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


# LLM-generated content at query #17
#--------------------------

```python
def test_fxrate_constructor():
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


# LLM-generated content at query #18
#--------------------------

```python
def test_fxrateservice_query_is_abstract():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        
        def queries(self, queries, strict=False):
            return []
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    test_date = date(2023, 1, 1)
    
    service = ConcreteFXRateService()
    result = service.query(usd, eur, test_date, strict=False)
    
    assert result is None


def test_fxrateservice_query_with_strict_false():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        
        def queries(self, queries, strict=False):
            return []
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    gbp = Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY)
    test_date = date(2023, 6, 15)
    
    service = MockFXRateService()
    result = service.query(usd, gbp, test_date, strict=False)
    
    assert result is None


def test_fxrateservice_query_accepts_different_currencies():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if ccy1.code == "JPY" and ccy2.code == "USD":
                return 0.0075
            return None
        
        def queries(self, queries, strict=False):
            return []
    
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    test_date = date(2023, 1, 1)
    
    service = TestFXRateService()
    result = service.query(jpy, usd, test_date)
    
    assert result == 0.0075


def test_fxrateservice_query_with_strict_true():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    class StrictFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if strict:
                raise LookupError("Rate not found")
            return None
        
        def queries(self, queries, strict=False):
            return []
    
    chf = Currency.of("CHF", "Swiss Franc", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    test_date = date(2023, 3, 10)
    
    service = StrictFXRateService()
    
    try:
        service.query(chf, eur, test_date, strict=True)
        assert False, "Expected LookupError"
    except LookupError:
        assert True


def test_fxrateservice_query_default_parameter():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    class DefaultParamFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return {"rate": 1.5, "strict": strict}
        
        def queries(self, queries, strict=False):
            return []
    
    aud = Currency.of("AUD", "Australian Dollar", 2, CurrencyType.MONEY)
    nzd = Currency.of("NZD", "New Zealand Dollar", 2, CurrencyType.MONEY)
    test_date = date(2023, 12, 1)
    
    service = DefaultParamFXRateService()
    result = service.query(aud, nzd, test_date)
    
    assert result["strict"] is False
    assert result["rate"] == 1.5


# LLM-generated content at query #19
#--------------------------

```python
def test_queries_returns_iterable_of_rates():
    from datetime import date
    from decimal import Decimal
    from unittest.mock import Mock, MagicMock
    
    # Create a mock implementation of FXRateService
    mock_service = Mock(spec=FXRateService)
    
    # Create test data
    ccy1 = Mock()
    ccy2 = Mock()
    test_date = date(2024, 1, 15)
    
    query1 = (ccy1, ccy2, test_date)
    query2 = (ccy2, ccy1, test_date)
    queries = [query1, query2]
    
    rate1 = Decimal("1.2345")
    rate2 = Decimal("0.8103")
    expected_rates = [rate1, rate2]
    
    mock_service.queries.return_value = iter(expected_rates)
    
    # Call the method
    result = mock_service.queries(queries, strict=False)
    result_list = list(result)
    
    # Assertions
    assert result_list == expected_rates
    mock_service.queries.assert_called_once_with(queries, strict=False)


def test_queries_with_strict_mode():
    from datetime import date
    from decimal import Decimal
    from unittest.mock import Mock
    
    mock_service = Mock(spec=FXRateService)
    
    ccy1 = Mock()
    ccy2 = Mock()
    test_date = date(2024, 1, 15)
    
    query = (ccy1, ccy2, test_date)
    queries = [query]
    
    rate = Decimal("1.5000")
    mock_service.queries.return_value = iter([rate])
    
    result = mock_service.queries(queries, strict=True)
    result_list = list(result)
    
    assert result_list == [rate]
    mock_service.queries.assert_called_once_with(queries, strict=True)


def test_queries_with_none_rates():
    from datetime import date
    from unittest.mock import Mock
    
    mock_service = Mock(spec=FXRateService)
    
    ccy1 = Mock()
    ccy2 = Mock()
    test_date = date(2024, 1, 15)
    
    query1 = (ccy1, ccy2, test_date)
    query2 = (ccy2, ccy1, test_date)
    queries = [query1, query2]
    
    expected_rates = [None, None]
    mock_service.queries.return_value = iter(expected_rates)
    
    result = mock_service.queries(queries, strict=False)
    result_list = list(result)
    
    assert result_list == expected_rates
    assert all(rate is None for rate in result_list)


def test_queries_with_mixed_rates_and_none():
    from datetime import date
    from decimal import Decimal
    from unittest.mock import Mock
    
    mock_service = Mock(spec=FXRateService)
    
    ccy1 = Mock()
    ccy2 = Mock()
    test_date = date(2024, 1, 15)
    
    queries = [(ccy1, ccy2, test_date), (ccy2, ccy1, test_date), (ccy1, ccy2, test_date)]
    
    expected_rates = [Decimal("1.2345"), None, Decimal("1.5000")]
    mock_service.queries.return_value = iter(expected_rates)
    
    result = mock_service.queries(queries, strict=False)
    result_list = list(result)
    
    assert result_list == expected_rates
    assert result_list[0] == Decimal("1.2345")
    assert result_list[1] is None
    assert result_list[2] == Decimal("1.5000")


def test_queries_with_empty_queries():
    from unittest.mock import Mock
    
    mock_service = Mock(spec=FXRateService)
    
    queries = []
    expected_rates = []
    mock_service.queries.return_value = iter(expected_rates)
    
    result = mock_service.queries(queries, strict=False)
    result_list = list(result)
    
    assert result_list == []
    mock_service.queries.assert_called_once_with(queries, strict=False)


# LLM-generated content at query #20
#--------------------------

```python
def test_fxrate_constructor_basic():
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


def test_fxrate_constructor_immutability():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("2")
    
    rate = FXRate(ccy1, ccy2, date, value)
    
    try:
        rate.value = Decimal("3")
        assert False, "Should not be able to modify immutable FXRate"
    except AttributeError:
        pass


def test_fxrate_constructor_with_different_currencies():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    
    ccy1 = Currencies["GBP"]
    ccy2 = Currencies["JPY"]
    date = datetime.date(2023, 1, 1)
    value = Decimal("150.5")
    
    rate = FXRate(ccy1, ccy2, date, value)
    
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value


def test_fxrate_constructor_with_decimal_precision():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("1.123456789")
    
    rate = FXRate(ccy1, ccy2, date, value)
    
    assert rate.value == Decimal("1.123456789")


# LLM-generated content at query #21
#--------------------------

```python
def test_queries_with_single_query():
    from decimal import Decimal
    from datetime import date
    
    class MockCurrency:
        def __init__(self, code):
            self.code = code
    
    class MockFXRate:
        def __init__(self, rate):
            self.rate = rate
    
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if ccy1.code == "USD" and ccy2.code == "EUR":
                return MockFXRate(Decimal("0.85"))
            return None
        
        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    service = ConcreteFXRateService()
    ccy1 = MockCurrency("USD")
    ccy2 = MockCurrency("EUR")
    asof = date(2023, 1, 1)
    
    result = list(service.queries([(ccy1, ccy2, asof)]))
    
    assert len(result) == 1
    assert result[0] is not None
    assert result[0].rate == Decimal("0.85")


def test_queries_with_multiple_queries():
    from decimal import Decimal
    from datetime import date
    
    class MockCurrency:
        def __init__(self, code):
            self.code = code
    
    class MockFXRate:
        def __init__(self, rate):
            self.rate = rate
    
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            rates = {
                ("USD", "EUR"): Decimal("0.85"),
                ("USD", "GBP"): Decimal("0.73"),
                ("EUR", "GBP"): Decimal("0.86")
            }
            key = (ccy1.code, ccy2.code)
            if key in rates:
                return MockFXRate(rates[key])
            return None
        
        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    service = ConcreteFXRateService()
    ccy_usd = MockCurrency("USD")
    ccy_eur = MockCurrency("EUR")
    ccy_gbp = MockCurrency("GBP")
    asof = date(2023, 1, 1)
    
    query_list = [(ccy_usd, ccy_eur, asof), (ccy_usd, ccy_gbp, asof), (ccy_eur, ccy_gbp, asof)]
    result = list(service.queries(query_list))
    
    assert len(result) == 3
    assert result[0].rate == Decimal("0.85")
    assert result[1].rate == Decimal("0.73")
    assert result[2].rate == Decimal("0.86")


def test_queries_with_missing_rates():
    from decimal import Decimal
    from datetime import date
    
    class MockCurrency:
        def __init__(self, code):
            self.code = code
    
    class MockFXRate:
        def __init__(self, rate):
            self.rate = rate
    
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if ccy1.code == "USD" and ccy2.code == "EUR":
                return MockFXRate(Decimal("0.85"))
            return None
        
        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    service = ConcreteFXRateService()
    ccy_usd = MockCurrency("USD")
    ccy_eur = MockCurrency("EUR")
    ccy_jpy = MockCurrency("JPY")
    asof = date(2023, 1, 1)
    
    query_list = [(ccy_usd, ccy_eur, asof), (ccy_usd, ccy_jpy, asof)]
    result = list(service.queries(query_list))
    
    assert len(result) == 2
    assert result[0] is not None
    assert result[0].rate == Decimal("0.85")
    assert result[1] is None


def test_queries_with_empty_input():
    from datetime import date
    
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        
        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    service = ConcreteFXRateService()
    result = list(service.queries([]))
    
    assert len(result) == 0


def test_queries_with_strict_mode():
    from datetime import date
    
    class MockCurrency:
        def __init__(self, code):
            self.code = code
    
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if strict and ccy1.code == "UNKNOWN":
                raise LookupError("Rate not found")
            return None
        
        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    service = ConcreteFXRateService()
    ccy_unknown = MockCurrency("UNKNOWN")
    ccy_usd = MockCurrency("USD")
    asof = date(2023, 1, 1)
    
    try:
        list(service.queries([(ccy_unknown, ccy_usd, asof)], strict=True))
        assert False, "Expected LookupError"
    except LookupError:
        assert True


# LLM-generated content at query #22
#--------------------------

```python
def test_fxrateservice_query_is_abstract():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    test_date = date(2023, 1, 1)
    
    try:
        service = FXRateService()
        result = service.query(usd, eur, test_date)
        assert False, "Should not be able to instantiate abstract class"
    except TypeError:
        assert True


def test_fxrateservice_query_with_strict_false():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        
        def queries(self, queries, strict=False):
            return []
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    test_date = date(2023, 1, 1)
    
    service = MockFXRateService()
    result = service.query(usd, eur, test_date, strict=False)
    assert result is None


def test_fxrateservice_query_with_strict_true():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if strict:
                raise LookupError(f"Rate not found for {ccy1.code}/{ccy2.code} as of {asof}")
            return None
        
        def queries(self, queries, strict=False):
            return []
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    test_date = date(2023, 1, 1)
    
    service = MockFXRateService()
    try:
        service.query(usd, eur, test_date, strict=True)
        assert False, "Should raise LookupError"
    except LookupError:
        assert True


def test_fxrateservice_query_accepts_currency_pair_and_date():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            assert isinstance(ccy1, Currency)
            assert isinstance(ccy2, Currency)
            assert isinstance(asof, date)
            assert isinstance(strict, bool)
            return None
        
        def queries(self, queries, strict=False):
            return []
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    gbp = Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY)
    test_date = date(2023, 6, 15)
    
    service = MockFXRateService()
    result = service.query(usd, gbp, test_date)
    assert result is None


# LLM-generated content at query #23
#--------------------------

```python
def test_fxrate_constructor():
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


# LLM-generated content at query #24
#--------------------------

```python
def test_fxrate_constructor():
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


# LLM-generated content at query #25
#--------------------------

```python
def test_queries_with_empty_iterable():
    from abc import ABCMeta
    from decimal import Decimal
    from datetime import date
    
    class MockCurrency:
        def __init__(self, code):
            self.code = code
    
    class MockFXRateService(metaclass=ABCMeta):
        default = None
        TQuery = tuple
        
        def query(self, ccy1, ccy2, asof, strict=False):
            pass
        
        def queries(self, queries, strict=False):
            return [None for _ in queries]
    
    service = MockFXRateService()
    result = list(service.queries([]))
    assert result == []


def test_queries_with_single_query():
    from abc import ABCMeta
    from decimal import Decimal
    from datetime import date
    
    class MockCurrency:
        def __init__(self, code):
            self.code = code
    
    class MockFXRateService(metaclass=ABCMeta):
        default = None
        TQuery = tuple
        
        def query(self, ccy1, ccy2, asof, strict=False):
            return Decimal("1.5")
        
        def queries(self, queries, strict=False):
            return [Decimal("1.5") for _ in queries]
    
    service = MockFXRateService()
    ccy1 = MockCurrency("USD")
    ccy2 = MockCurrency("EUR")
    test_date = date(2023, 1, 1)
    queries_list = [(ccy1, ccy2, test_date)]
    
    result = list(service.queries(queries_list))
    assert len(result) == 1
    assert result[0] == Decimal("1.5")


def test_queries_with_multiple_queries():
    from abc import ABCMeta
    from decimal import Decimal
    from datetime import date
    
    class MockCurrency:
        def __init__(self, code):
            self.code = code
    
    class MockFXRateService(metaclass=ABCMeta):
        default = None
        TQuery = tuple
        
        def query(self, ccy1, ccy2, asof, strict=False):
            pass
        
        def queries(self, queries, strict=False):
            return [Decimal("1.5"), Decimal("1.2"), Decimal("0.9")]
    
    service = MockFXRateService()
    ccy1 = MockCurrency("USD")
    ccy2 = MockCurrency("EUR")
    ccy3 = MockCurrency("GBP")
    test_date = date(2023, 1, 1)
    queries_list = [(ccy1, ccy2, test_date), (ccy1, ccy3, test_date), (ccy2, ccy3, test_date)]
    
    result = list(service.queries(queries_list))
    assert len(result) == 3
    assert result[0] == Decimal("1.5")
    assert result[1] == Decimal("1.2")
    assert result[2] == Decimal("0.9")


def test_queries_with_strict_false():
    from abc import ABCMeta
    from datetime import date
    
    class MockCurrency:
        def __init__(self, code):
            self.code = code
    
    class MockFXRateService(metaclass=ABCMeta):
        default = None
        TQuery = tuple
        
        def query(self, ccy1, ccy2, asof, strict=False):
            pass
        
        def queries(self, queries, strict=False):
            return [None, None]
    
    service = MockFXRateService()
    ccy1 = MockCurrency("USD")
    ccy2 = MockCurrency("EUR")
    test_date = date(2023, 1, 1)
    queries_list = [(ccy1, ccy2, test_date), (ccy1, ccy2, test_date)]
    
    result = list(service.queries(queries_list, strict=False))
    assert len(result) == 2
    assert all(r is None for r in result)


def test_queries_returns_iterable():
    from abc import ABCMeta
    from decimal import Decimal
    from datetime import date
    
    class MockCurrency:
        def __init__(self, code):
            self.code = code
    
    class MockFXRateService(metaclass=ABCMeta):
        default = None
        TQuery = tuple
        
        def query(self, ccy1, ccy2, asof, strict=False):
            pass
        
        def queries(self, queries, strict=False):
            for q in queries:
                yield Decimal("1.5")
    
    service = MockFXRateService()
    ccy1 = MockCurrency("USD")
    ccy2 = MockCurrency("EUR")
    test_date = date(2023, 1, 1)
    queries_list = [(ccy1, ccy2, test_date), (ccy1, ccy2, test_date)]
    
    result = service.queries(queries_list)
    rates = [r for r in result]
    assert len(rates) == 2
    assert all(r == Decimal("1.5") for r in rates)


# LLM-generated content at query #26
#--------------------------

```python
def test_fxrateservice_query_is_abstract():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        
        def queries(self, queries, strict=False):
            return []
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    test_date = date(2023, 1, 1)
    
    service = ConcreteFXRateService()
    result = service.query(usd, eur, test_date)
    
    assert result is None


def test_fxrateservice_query_with_strict_false():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        
        def queries(self, queries, strict=False):
            return []
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    test_date = date(2023, 6, 15)
    
    service = TestFXRateService()
    result = service.query(usd, jpy, test_date, strict=False)
    
    assert result is None


def test_fxrateservice_query_accepts_different_currencies():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return (ccy1, ccy2, asof, strict)
        
        def queries(self, queries, strict=False):
            return []
    
    gbp = Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY)
    cad = Currency.of("CAD", "Canadian Dollar", 2, CurrencyType.MONEY)
    test_date = date(2023, 12, 25)
    
    service = TestFXRateService()
    result = service.query(gbp, cad, test_date)
    
    assert result == (gbp, cad, test_date, False)


def test_fxrateservice_query_with_strict_true():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return strict
        
        def queries(self, queries, strict=False):
            return []
    
    aud = Currency.of("AUD", "Australian Dollar", 2, CurrencyType.MONEY)
    nzd = Currency.of("NZD", "New Zealand Dollar", 2, CurrencyType.MONEY)
    test_date = date(2024, 1, 1)
    
    service = TestFXRateService()
    result = service.query(aud, nzd, test_date, strict=True)
    
    assert result is True


# LLM-generated content at query #27
#--------------------------

```python
def test_fxrate_constructor():
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


# LLM-generated content at query #28
#--------------------------

```python
def test_fxrate_constructor():
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


def test_fxrate_constructor_unpacking():
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


def test_fxrate_constructor_with_different_currencies():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    
    ccy1 = Currencies["GBP"]
    ccy2 = Currencies["JPY"]
    date = datetime.date(2023, 1, 15)
    value = Decimal("150.5")
    
    rate = FXRate(ccy1, ccy2, date, value)
    
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value


# LLM-generated content at query #29
#--------------------------

```python
def test_queries_with_empty_iterable():
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            return None
        
        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    service = MockFXRateService()
    result = list(service.queries([]))
    assert result == []


def test_queries_with_single_query():
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            if ccy1.code == "USD" and ccy2.code == "EUR":
                return FXRate(Decimal("0.85"))
            return None
        
        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    service = MockFXRateService()
    ccy1 = Currency(code="USD")
    ccy2 = Currency(code="EUR")
    asof = Date(2023, 1, 1)
    result = list(service.queries([(ccy1, ccy2, asof)]))
    assert len(result) == 1


def test_queries_with_multiple_queries():
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            return FXRate(Decimal("1.0"))
        
        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    service = MockFXRateService()
    ccy1 = Currency(code="USD")
    ccy2 = Currency(code="EUR")
    asof = Date(2023, 1, 1)
    queries_list = [(ccy1, ccy2, asof), (ccy1, ccy2, asof), (ccy1, ccy2, asof)]
    result = list(service.queries(queries_list))
    assert len(result) == 3


def test_queries_with_strict_false():
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            return None
        
        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    service = MockFXRateService()
    ccy1 = Currency(code="USD")
    ccy2 = Currency(code="EUR")
    asof = Date(2023, 1, 1)
    result = list(service.queries([(ccy1, ccy2, asof)], strict=False))
    assert result == [None]


def test_queries_returns_iterable():
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            return FXRate(Decimal("1.0"))
        
        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return (self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries)
    
    service = MockFXRateService()
    ccy1 = Currency(code="USD")
    ccy2 = Currency(code="EUR")
    asof = Date(2023, 1, 1)
    result = service.queries([(ccy1, ccy2, asof)])
    assert hasattr(result, '__iter__')


# LLM-generated content at query #30
#--------------------------

```python
def test_fxrateservice_query_is_abstract():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        
        def queries(self, queries, strict=False):
            return []
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    test_date = date(2023, 1, 1)
    
    service = ConcreteFXRateService()
    result = service.query(usd, eur, test_date, strict=False)
    
    assert result is None


def test_fxrateservice_query_with_strict_false():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        
        def queries(self, queries, strict=False):
            return []
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    gbp = Currency.of("GBP", "British Pounds", 2, CurrencyType.MONEY)
    test_date = date(2023, 6, 15)
    
    service = TestFXRateService()
    result = service.query(usd, gbp, test_date, strict=False)
    
    assert result is None


def test_fxrateservice_query_accepts_different_currencies():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return (ccy1, ccy2, asof)
        
        def queries(self, queries, strict=False):
            return []
    
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    chf = Currency.of("CHF", "Swiss Francs", 2, CurrencyType.MONEY)
    test_date = date(2023, 12, 25)
    
    service = TestFXRateService()
    result = service.query(jpy, chf, test_date)
    
    assert result == (jpy, chf, test_date)


def test_fxrateservice_query_with_strict_true():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if strict:
                return "strict_mode_enabled"
            return None
        
        def queries(self, queries, strict=False):
            return []
    
    aud = Currency.of("AUD", "Australian Dollars", 2, CurrencyType.MONEY)
    cad = Currency.of("CAD", "Canadian Dollars", 2, CurrencyType.MONEY)
    test_date = date(2023, 3, 10)
    
    service = TestFXRateService()
    result = service.query(aud, cad, test_date, strict=True)
    
    assert result == "strict_mode_enabled"


# LLM-generated content at query #31
#--------------------------

```python
def test_fxrate_constructor():
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


def test_fxrate_constructor_immutability():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("2")
    
    rate = FXRate(ccy1, ccy2, date, value)
    
    try:
        rate.value = Decimal("3")
        assert False, "Should not be able to modify immutable FXRate"
    except AttributeError:
        pass


# LLM-generated content at query #32
#--------------------------

```python
def test_fxrate_constructor():
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


