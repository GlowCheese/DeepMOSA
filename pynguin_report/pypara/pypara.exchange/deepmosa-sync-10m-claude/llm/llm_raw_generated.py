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
    double_inverted_rate = ~(~original_rate)
    
    assert double_inverted_rate.ccy1 == original_rate.ccy1
    assert double_inverted_rate.ccy2 == original_rate.ccy2
    assert double_inverted_rate.date == original_rate.date
    assert double_inverted_rate.value == original_rate.value


def test_invert_fx_rate_with_decimal_one():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    
    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("1"))
    inverted_rate = ~rate
    
    assert inverted_rate.value == Decimal("1")


def test_invert_fx_rate_with_small_decimal():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    
    rate = FXRate(Currencies["GBP"], Currencies["JPY"], datetime.date.today(), Decimal("0.1"))
    inverted_rate = ~rate
    
    assert inverted_rate.ccy1 == Currencies["JPY"]
    assert inverted_rate.ccy2 == Currencies["GBP"]
    assert inverted_rate.value == Decimal("10")


# LLM-generated content at query #2
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


# LLM-generated content at query #3
#--------------------------

```python
def test_queries_with_multiple_currency_pairs():
    from abc import ABCMeta
    from datetime import date
    from decimal import Decimal
    
    class MockCurrency:
        def __init__(self, code):
            self.code = code
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if ccy1.code == "USD" and ccy2.code == "EUR":
                return Decimal("0.85")
            elif ccy1.code == "GBP" and ccy2.code == "USD":
                return Decimal("1.25")
            return None
        
        def queries(self, queries, strict=False):
            results = []
            for ccy1, ccy2, asof in queries:
                result = self.query(ccy1, ccy2, asof, strict)
                results.append(result)
            return results
    
    service = MockFXRateService()
    usd = MockCurrency("USD")
    eur = MockCurrency("EUR")
    gbp = MockCurrency("GBP")
    test_date = date(2023, 1, 1)
    
    query_list = [
        (usd, eur, test_date),
        (gbp, usd, test_date),
    ]
    
    results = list(service.queries(query_list))
    
    assert len(results) == 2
    assert results[0] == Decimal("0.85")
    assert results[1] == Decimal("1.25")


def test_queries_with_empty_collection():
    from abc import ABCMeta
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
    query_list = []
    
    results = list(service.queries(query_list))
    
    assert len(results) == 0


def test_queries_with_strict_mode():
    from abc import ABCMeta
    from datetime import date
    from decimal import Decimal
    
    class MockCurrency:
        def __init__(self, code):
            self.code = code
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if ccy1.code == "USD" and ccy2.code == "EUR":
                return Decimal("0.85")
            elif strict:
                raise LookupError(f"Rate not found for {ccy1.code}/{ccy2.code}")
            return None
        
        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    service = MockFXRateService()
    usd = MockCurrency("USD")
    eur = MockCurrency("EUR")
    jpy = MockCurrency("JPY")
    test_date = date(2023, 1, 1)
    
    query_list = [(usd, eur, test_date)]
    results = list(service.queries(query_list, strict=False))
    
    assert len(results) == 1
    assert results[0] == Decimal("0.85")


def test_queries_returns_none_for_missing_rates():
    from abc import ABCMeta
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
    usd = MockCurrency("USD")
    eur = MockCurrency("EUR")
    test_date = date(2023, 1, 1)
    
    query_list = [(usd, eur, test_date)]
    results = list(service.queries(query_list))
    
    assert len(results) == 1
    assert results[0] is None


# LLM-generated content at query #4
#--------------------------

```python
def test_invert_fxrate():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    
    nrate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("2"))
    inverted_rate = ~nrate
    
    assert inverted_rate.ccy1 == Currencies["USD"]
    assert inverted_rate.ccy2 == Currencies["EUR"]
    assert inverted_rate.date == datetime.date.today()
    assert inverted_rate.value == Decimal("0.5")


def test_invert_fxrate_reciprocal():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    
    nrate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("2"))
    rrate = FXRate(Currencies["USD"], Currencies["EUR"], datetime.date.today(), Decimal("0.5"))
    
    assert ~nrate == rrate


def test_invert_fxrate_double_invert():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    
    original_rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("2.5"))
    double_inverted = ~(~original_rate)
    
    assert double_inverted.ccy1 == original_rate.ccy1
    assert double_inverted.ccy2 == original_rate.ccy2
    assert double_inverted.date == original_rate.date
    assert double_inverted.value == original_rate.value


def test_invert_fxrate_with_decimal_one():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    
    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("1"))
    inverted = ~rate
    
    assert inverted.value == Decimal("1")
    assert inverted.ccy1 == Currencies["USD"]
    assert inverted.ccy2 == Currencies["EUR"]


def test_invert_fxrate_with_small_decimal():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    
    rate = FXRate(Currencies["GBP"], Currencies["JPY"], datetime.date.today(), Decimal("0.01"))
    inverted = ~rate
    
    assert inverted.value == Decimal("100")
    assert inverted.ccy1 == Currencies["JPY"]
    assert inverted.ccy2 == Currencies["GBP"]


# LLM-generated content at query #5
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


def test_invert_fx_rate_double_inversion():
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


# LLM-generated content at query #6
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
    
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: date, strict: bool = False) -> Optional[FXRate]:
            return None
        
        def queries(self, queries: Iterable[Tuple[Currency, Currency, date]], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return []
    
    service = ConcreteFXRateService()
    result = list(service.queries([]))
    assert result == []


def test_queries_with_multiple_queries():
    from abc import ABCMeta, abstractmethod
    from typing import Tuple, Iterable, Optional
    from datetime import date
    
    class Currency:
        def __init__(self, code: str):
            self.code = code
    
    class FXRate:
        def __init__(self, rate: float):
            self.rate = rate
    
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: date, strict: bool = False) -> Optional[FXRate]:
            return None
        
        def queries(self, queries: Iterable[Tuple[Currency, Currency, date]], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [FXRate(1.5), FXRate(2.0), None]
    
    service = ConcreteFXRateService()
    usd = Currency("USD")
    eur = Currency("EUR")
    gbp = Currency("GBP")
    test_date = date(2023, 1, 1)
    
    query_list = [(usd, eur, test_date), (eur, gbp, test_date), (gbp, usd, test_date)]
    result = list(service.queries(query_list))
    
    assert len(result) == 3
    assert result[0].rate == 1.5
    assert result[1].rate == 2.0
    assert result[2] is None


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
    
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: date, strict: bool = False) -> Optional[FXRate]:
            return None
        
        def queries(self, queries: Iterable[Tuple[Currency, Currency, date]], strict: bool = False) -> Iterable[Optional[FXRate]]:
            if strict:
                return [FXRate(1.5)]
            return [None]
    
    service = ConcreteFXRateService()
    usd = Currency("USD")
    eur = Currency("EUR")
    test_date = date(2023, 1, 1)
    
    result_strict = list(service.queries([(usd, eur, test_date)], strict=True))
    result_non_strict = list(service.queries([(usd, eur, test_date)], strict=False))
    
    assert result_strict[0].rate == 1.5
    assert result_non_strict[0] is None


# LLM-generated content at query #7
#--------------------------

```python
def test_fxrateservice_query_is_abstract():
    from abc import ABC, ABCMeta
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    # Verify FXRateService is abstract and cannot be instantiated
    assert isinstance(FXRateService, ABCMeta)
    
    # Create a concrete implementation for testing
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        
        def queries(self, queries, strict=False):
            return []
    
    # Test instantiation of concrete implementation
    service = ConcreteFXRateService()
    assert service is not None
    
    # Test query method with currencies
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    test_date = date(2023, 1, 1)
    
    result = service.query(usd, eur, test_date)
    assert result is None
    
    # Test query method with strict=False (default)
    result = service.query(usd, eur, test_date, strict=False)
    assert result is None
    
    # Test query method with strict=True
    result = service.query(usd, eur, test_date, strict=True)
    assert result is None


# LLM-generated content at query #8
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
    
    result = list(service.queries([(ccy1, ccy2, asof1), (ccy1, ccy3, asof2)]))
    assert len(result) == 2
    assert all(r.rate == Decimal("1.5") for r in result)


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
    ccy2 = MockCurrency("EUR")
    asof = date(2023, 1, 1)
    
    result = list(service.queries([(ccy1, ccy2, asof)]))
    assert len(result) == 1
    assert result[0] is None


def test_queries_with_strict_false():
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
    ccy2 = MockCurrency("EUR")
    asof = date(2023, 1, 1)
    
    result = list(service.queries([(ccy1, ccy2, asof)], strict=False))
    assert len(result) == 1


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


# LLM-generated content at query #9
#--------------------------

```python
def test_invert_fxrate():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    
    nrate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("2"))
    inverted_rate = ~nrate
    
    assert inverted_rate.ccy1 == Currencies["USD"]
    assert inverted_rate.ccy2 == Currencies["EUR"]
    assert inverted_rate.date == datetime.date.today()
    assert inverted_rate.value == Decimal("0.5")


def test_invert_fxrate_double_invert():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    
    original_rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("2"))
    double_inverted_rate = ~(~original_rate)
    
    assert double_inverted_rate.ccy1 == original_rate.ccy1
    assert double_inverted_rate.ccy2 == original_rate.ccy2
    assert double_inverted_rate.date == original_rate.date
    assert double_inverted_rate.value == original_rate.value


def test_invert_fxrate_decimal_one():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    
    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("1"))
    inverted_rate = ~rate
    
    assert inverted_rate.value == Decimal("1")


def test_invert_fxrate_small_decimal():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    
    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("0.5"))
    inverted_rate = ~rate
    
    assert inverted_rate.value == Decimal("2")


# LLM-generated content at query #10
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
        def __init__(self, value):
            self.value = value
    
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
    assert result[0].value == Decimal("1.5")


def test_queries_with_multiple_queries():
    from datetime import date
    from decimal import Decimal
    
    class MockCurrency:
        def __init__(self, code):
            self.code = code
    
    class MockFXRate:
        def __init__(self, value):
            self.value = value
    
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
    
    queries_list = [(ccy1, ccy2, asof1), (ccy2, ccy3, asof2)]
    result = list(service.queries(queries_list))
    assert len(result) == 2
    assert all(r.value == Decimal("1.5") for r in result)


def test_queries_with_strict_mode():
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
    ccy2 = MockCurrency("EUR")
    asof = date(2023, 1, 1)
    
    result = list(service.queries([(ccy1, ccy2, asof)], strict=True))
    assert len(result) == 1
    assert result[0] is None


def test_queries_returns_iterable():
    from datetime import date
    
    class MockCurrency:
        def __init__(self, code):
            self.code = code
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        
        def queries(self, queries, strict=False):
            return iter([self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries])
    
    service = MockFXRateService()
    ccy1 = MockCurrency("USD")
    ccy2 = MockCurrency("EUR")
    asof = date(2023, 1, 1)
    
    result = service.queries([(ccy1, ccy2, asof)])
    assert hasattr(result, '__iter__')


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
    
    rate = FXRate(Currencies["GBP"], Currencies["JPY"], datetime.date.today(), Decimal("150.5"))
    
    assert rate.ccy1 == Currencies["GBP"]
    assert rate.ccy2 == Currencies["JPY"]
    assert rate.value == Decimal("150.5")


def test_fxrate_constructor_with_decimal_value():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    
    value = Decimal("1.23456789")
    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), value)
    
    assert rate.value == value


# LLM-generated content at query #12
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


# LLM-generated content at query #13
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
    date = datetime.date(2023, 1, 15)
    value = Decimal("150.5")
    
    rate = FXRate(ccy1, ccy2, date, value)
    
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value


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


# LLM-generated content at query #17
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


# LLM-generated content at query #18
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
    date = datetime.date(2023, 1, 15)
    value = Decimal("150.5")
    
    rate = FXRate(ccy1, ccy2, date, value)
    
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value


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
    
    ccy1 = Currencies["GBP"]
    ccy2 = Currencies["JPY"]
    date = datetime.date(2023, 1, 1)
    value = Decimal("150.5")
    
    rate = FXRate(ccy1, ccy2, date, value)
    
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value


def test_fxrate_constructor_small_decimal_value():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    
    ccy1 = Currencies["USD"]
    ccy2 = Currencies["EUR"]
    date = datetime.date.today()
    value = Decimal("0.001")
    
    rate = FXRate(ccy1, ccy2, date, value)
    
    assert rate.value == value


def test_fxrate_constructor_large_decimal_value():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    
    ccy1 = Currencies["USD"]
    ccy2 = Currencies["JPY"]
    date = datetime.date.today()
    value = Decimal("999999.999999")
    
    rate = FXRate(ccy1, ccy2, date, value)
    
    assert rate.value == value


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


# LLM-generated content at query #21
#--------------------------

```python
def test_fxrate_service_query_is_abstract():
    from pypara.currencies import Currency, CurrencyType
    from pypara.exchange import FXRateService
    from datetime import date
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    asof = date(2023, 1, 1)
    
    try:
        FXRateService().query(usd, eur, asof)
        assert False, "Should not be able to instantiate abstract class"
    except TypeError:
        pass


# LLM-generated content at query #22
#--------------------------

```python
def test_queries_with_empty_iterable():
    from abc import ABCMeta, abstractmethod
    from typing import Tuple, Iterable, Optional
    from datetime import date
    from decimal import Decimal

    class Currency:
        def __init__(self, code: str):
            self.code = code

    class FXRate:
        def __init__(self, rate: Decimal):
            self.rate = rate

    class ConcreteFXRateService(metaclass=ABCMeta):
        default: Optional["ConcreteFXRateService"] = None
        TQuery = Tuple[Currency, Currency, date]

        @abstractmethod
        def query(self, ccy1: Currency, ccy2: Currency, asof: date, strict: bool = False) -> Optional[FXRate]:
            pass

        @abstractmethod
        def queries(self, queries: Iterable[TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            pass

    class TestFXRateService(ConcreteFXRateService):
        def __init__(self):
            self.rates = {}

        def query(self, ccy1: Currency, ccy2: Currency, asof: date, strict: bool = False) -> Optional[FXRate]:
            key = (ccy1.code, ccy2.code, asof)
            if key in self.rates:
                return self.rates[key]
            if strict:
                raise LookupError(f"Rate not found for {ccy1.code}/{ccy2.code} on {asof}")
            return None

        def queries(self, queries: Iterable[TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = TestFXRateService()
    ccy1 = Currency("USD")
    ccy2 = Currency("EUR")
    test_date = date(2024, 1, 1)
    
    query_list = [(ccy1, ccy2, test_date)]
    result = list(service.queries(query_list, strict=False))
    
    assert len(result) == 1
    assert result[0] is None


def test_queries_with_multiple_rates():
    from abc import ABCMeta, abstractmethod
    from typing import Tuple, Iterable, Optional
    from datetime import date
    from decimal import Decimal

    class Currency:
        def __init__(self, code: str):
            self.code = code

    class FXRate:
        def __init__(self, rate: Decimal):
            self.rate = rate

    class ConcreteFXRateService(metaclass=ABCMeta):
        default: Optional["ConcreteFXRateService"] = None
        TQuery = Tuple[Currency, Currency, date]

        @abstractmethod
        def query(self, ccy1: Currency, ccy2: Currency, asof: date, strict: bool = False) -> Optional[FXRate]:
            pass

        @abstractmethod
        def queries(self, queries: Iterable[TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            pass

    class TestFXRateService(ConcreteFXRateService):
        def __init__(self):
            self.rates = {}

        def query(self, ccy1: Currency, ccy2: Currency, asof: date, strict: bool = False) -> Optional[FXRate]:
            key = (ccy1.code, ccy2.code, asof)
            if key in self.rates:
                return self.rates[key]
            if strict:
                raise LookupError(f"Rate not found for {ccy1.code}/{ccy2.code} on {asof}")
            return None

        def queries(self, queries: Iterable[TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = TestFXRateService()
    ccy1 = Currency("USD")
    ccy2 = Currency("EUR")
    ccy3 = Currency("GBP")
    test_date = date(2024, 1, 1)
    
    rate1 = FXRate(Decimal("1.05"))
    rate2 = FXRate(Decimal("1.20"))
    service.rates[("USD", "EUR", test_date)] = rate1
    service.rates[("USD", "GBP", test_date)] = rate2
    
    query_list = [(ccy1, ccy2, test_date), (ccy1, ccy3, test_date)]
    result = list(service.queries(query_list, strict=False))
    
    assert len(result) == 2
    assert result[0].rate == Decimal("1.05")
    assert result[1].rate == Decimal("1.20")


def test_queries_strict_mode_raises_error():
    from abc import ABCMeta, abstractmethod
    from typing import Tuple, Iterable, Optional
    from datetime import date
    from decimal import Decimal

    class Currency:
        def __init__(self, code: str):
            self.code = code

    class FXRate:
        def __init__(self, rate: Decimal):
            self.rate = rate

    class ConcreteFXRateService(metaclass=ABCMeta):
        default: Optional["ConcreteFXRateService"] = None
        TQuery = Tuple[Currency, Currency, date]

        @abstractmethod
        def query(self, ccy1: Currency, ccy2: Currency, asof: date, strict: bool = False) -> Optional[FXRate]:
            pass

        @abstractmethod
        def queries(self, queries: Iterable[TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            pass

    class TestFXRateService(ConcreteFXRateService):
        def __init__(self):
            self.rates = {}

        def query(self, ccy1: Currency, ccy2: Currency, asof: date, strict: bool = False) -> Optional[FXRate]:
            key = (ccy1.code, ccy2.code, asof)
            if key in self.rates:
                return self.rates[key]
            if strict:
                raise LookupError(f"Rate not found for {ccy1.code}/{ccy2.code} on {asof}")
            return None

        def queries(self, queries: Iterable[TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = TestFXRateService()
    ccy1 = Currency("USD")
    ccy2 = Currency("EUR")
    test_date = date(2024, 1, 1)
    
    query_list = [(ccy1, ccy2, test_date)]
    
    try:
        list(service.queries(query_list, strict=True))
        assert False, "Expected LookupError to be raised"
    except LookupError:
        assert True


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
    date = datetime.date(2023, 1, 15)
    value = Decimal("150.5")
    
    rate = FXRate(ccy1, ccy2, date, value)
    
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value


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
    
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        
        def queries(self, queries, strict=False):
            return []
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    gbp = Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY)
    test_date = date(2023, 6, 15)
    
    service = ConcreteFXRateService()
    result = service.query(usd, gbp, test_date, strict=False)
    
    assert result is None


def test_fxrateservice_query_accepts_currencies_and_date():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    class ConcreteFXRateService(FXRateService):
        def __init__(self):
            self.last_query = None
        
        def query(self, ccy1, ccy2, asof, strict=False):
            self.last_query = (ccy1, ccy2, asof, strict)
            return None
        
        def queries(self, queries, strict=False):
            return []
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    test_date = date(2023, 12, 25)
    
    service = ConcreteFXRateService()
    service.query(usd, jpy, test_date, strict=True)
    
    assert service.last_query[0] == usd
    assert service.last_query[1] == jpy
    assert service.last_query[2] == test_date
    assert service.last_query[3] is True


# LLM-generated content at query #27
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
    
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: date, strict: bool = False) -> Optional[FXRate]:
            return FXRate(1.2)
        
        def queries(self, queries: Iterable[Tuple[Currency, Currency, date]], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    service = ConcreteFXRateService()
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
    
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: date, strict: bool = False) -> Optional[FXRate]:
            return FXRate(1.5)
        
        def queries(self, queries: Iterable[Tuple[Currency, Currency, date]], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    service = ConcreteFXRateService()
    usd = Currency("USD")
    eur = Currency("EUR")
    gbp = Currency("GBP")
    test_date = date(2023, 1, 1)
    
    queries = [(usd, eur, test_date), (eur, gbp, test_date), (gbp, usd, test_date)]
    result = list(service.queries(queries, strict=False))
    
    assert len(result) == 3
    assert all(isinstance(rate, FXRate) for rate in result)
    assert all(rate.rate == 1.5 for rate in result)


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
    
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: date, strict: bool = False) -> Optional[FXRate]:
            if strict and ccy1.code == "INVALID":
                raise LookupError(f"Rate not found for {ccy1.code}/{ccy2.code}")
            return FXRate(1.2) if ccy1.code != "INVALID" else None
        
        def queries(self, queries: Iterable[Tuple[Currency, Currency, date]], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    service = ConcreteFXRateService()
    usd = Currency("USD")
    eur = Currency("EUR")
    test_date = date(2023, 1, 1)
    
    queries = [(usd, eur, test_date)]
    result = list(service.queries(queries, strict=True))
    
    assert len(result) == 1
    assert isinstance(result[0], FXRate)


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
    
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: date, strict: bool = False) -> Optional[FXRate]:
            return FXRate(1.3)
        
        def queries(self, queries: Iterable[Tuple[Currency, Currency, date]], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    service = ConcreteFXRateService()
    usd = Currency("USD")
    eur = Currency("EUR")
    test_date = date(2023, 1, 1)
    
    queries = [(usd, eur, test_date)]
    result = service.queries(queries)
    
    assert hasattr(result, '__iter__')


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


# LLM-generated content at query #29
#--------------------------

```python
def test_queries_with_single_query():
    from decimal import Decimal
    from datetime import date
    
    class MockCurrency:
        def __init__(self, code):
            self.code = code
    
    class MockFXRate:
        def __init__(self, value):
            self.value = value
    
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return MockFXRate(Decimal("1.5"))
        
        def queries(self, queries, strict=False):
            return [MockFXRate(Decimal("1.5")) for _ in queries]
    
    service = ConcreteFXRateService()
    ccy1 = MockCurrency("USD")
    ccy2 = MockCurrency("EUR")
    asof = date(2023, 1, 1)
    
    query_list = [(ccy1, ccy2, asof)]
    results = list(service.queries(query_list))
    
    assert len(results) == 1
    assert results[0].value == Decimal("1.5")


def test_queries_with_multiple_queries():
    from decimal import Decimal
    from datetime import date
    
    class MockCurrency:
        def __init__(self, code):
            self.code = code
    
    class MockFXRate:
        def __init__(self, value):
            self.value = value
    
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return MockFXRate(Decimal("1.5"))
        
        def queries(self, queries, strict=False):
            return [MockFXRate(Decimal("1.5")) for _ in queries]
    
    service = ConcreteFXRateService()
    ccy1 = MockCurrency("USD")
    ccy2 = MockCurrency("EUR")
    ccy3 = MockCurrency("GBP")
    asof = date(2023, 1, 1)
    
    query_list = [(ccy1, ccy2, asof), (ccy1, ccy3, asof), (ccy2, ccy3, asof)]
    results = list(service.queries(query_list))
    
    assert len(results) == 3
    assert all(r.value == Decimal("1.5") for r in results)


def test_queries_with_strict_mode():
    from decimal import Decimal
    from datetime import date
    
    class MockCurrency:
        def __init__(self, code):
            self.code = code
    
    class MockFXRate:
        def __init__(self, value):
            self.value = value
    
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return MockFXRate(Decimal("1.5"))
        
        def queries(self, queries, strict=False):
            return [MockFXRate(Decimal("1.5")) for _ in queries]
    
    service = ConcreteFXRateService()
    ccy1 = MockCurrency("USD")
    ccy2 = MockCurrency("EUR")
    asof = date(2023, 1, 1)
    
    query_list = [(ccy1, ccy2, asof)]
    results = list(service.queries(query_list, strict=True))
    
    assert len(results) == 1
    assert results[0].value == Decimal("1.5")


def test_queries_returns_iterable():
    from decimal import Decimal
    from datetime import date
    
    class MockCurrency:
        def __init__(self, code):
            self.code = code
    
    class MockFXRate:
        def __init__(self, value):
            self.value = value
    
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return MockFXRate(Decimal("1.5"))
        
        def queries(self, queries, strict=False):
            for q in queries:
                yield MockFXRate(Decimal("1.5"))
    
    service = ConcreteFXRateService()
    ccy1 = MockCurrency("USD")
    ccy2 = MockCurrency("EUR")
    asof = date(2023, 1, 1)
    
    query_list = [(ccy1, ccy2, asof)]
    results = service.queries(query_list)
    
    assert hasattr(results, '__iter__')
    result_list = list(results)
    assert len(result_list) == 1


def test_queries_with_none_results():
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
    asof = date(2023, 1, 1)
    
    query_list = [(ccy1, ccy2, asof)]
    results = list(service.queries(query_list))
    
    assert len(results) == 1
    assert results[0] is None


# LLM-generated content at query #30
#--------------------------

```python
def test_fxrateservice_query_is_abstract():
    from abc import ABCMeta
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    from decimal import Decimal
    
    # Verify that FXRateService is abstract and cannot be instantiated
    try:
        FXRateService()
        assert False, "FXRateService should not be instantiable"
    except TypeError:
        pass
    
    # Create a concrete implementation to test the query method signature
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        
        def queries(self, queries, strict=False):
            return []
    
    # Create test currencies
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    
    # Create service instance
    service = ConcreteFXRateService()
    
    # Test that query method can be called with proper arguments
    result = service.query(usd, eur, date(2023, 1, 1))
    assert result is None
    
    # Test that query method can be called with strict parameter
    result = service.query(usd, eur, date(2023, 1, 1), strict=True)
    assert result is None
    
    # Test that query method can be called with strict=False explicitly
    result = service.query(usd, eur, date(2023, 1, 1), strict=False)
    assert result is None


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


def test_fxrate_constructor_with_decimal_string():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("1.5")
    
    rate = FXRate(ccy1, ccy2, date, value)
    
    assert rate.value == Decimal("1.5")


def test_fxrate_constructor_with_different_currencies():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    
    ccy1 = Currencies["GBP"]
    ccy2 = Currencies["JPY"]
    date = datetime.date.today()
    value = Decimal("150.25")
    
    rate = FXRate(ccy1, ccy2, date, value)
    
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.value == value


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
        assert False, "Should not be able to modify immutable FXRate"
    except AttributeError:
        pass


# LLM-generated content at query #33
#--------------------------

```python
def test_queries_with_empty_iterable():
    from collections.abc import Iterable
    from decimal import Decimal
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
    result = list(service.queries([]))
    assert result == []


def test_queries_with_single_query():
    from collections.abc import Iterable
    from decimal import Decimal
    from datetime import date
    
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
    test_date = date(2023, 1, 1)
    
    result = list(service.queries([(ccy1, ccy2, test_date)]))
    assert len(result) == 1
    assert result[0].rate == Decimal("1.5")


def test_queries_with_multiple_queries():
    from collections.abc import Iterable
    from decimal import Decimal
    from datetime import date
    
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
    test_date = date(2023, 1, 1)
    
    result = list(service.queries([
        (ccy1, ccy2, test_date),
        (ccy2, ccy3, test_date),
        (ccy1, ccy3, test_date)
    ]))
    assert len(result) == 3
    assert all(r.rate == Decimal("1.5") for r in result)


def test_queries_with_strict_mode():
    from collections.abc import Iterable
    from decimal import Decimal
    from datetime import date
    
    class MockCurrency:
        def __init__(self, code):
            self.code = code
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if strict:
                raise LookupError("FX rate not found")
            return None
        
        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    service = MockFXRateService()
    ccy1 = MockCurrency("USD")
    ccy2 = MockCurrency("EUR")
    test_date = date(2023, 1, 1)
    
    try:
        list(service.queries([(ccy1, ccy2, test_date)], strict=True))
        assert False, "Expected LookupError"
    except LookupError:
        assert True


def test_queries_returns_iterable():
    from collections.abc import Iterable
    from decimal import Decimal
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


# LLM-generated content at query #34
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
    
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        
        def queries(self, queries, strict=False):
            return []
    
    gbp = Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    test_date = date(2023, 6, 15)
    
    service = ConcreteFXRateService()
    result = service.query(gbp, jpy, test_date, strict=False)
    
    assert result is None


def test_fxrateservice_query_accepts_different_currencies():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    class ConcreteFXRateService(FXRateService):
        def __init__(self):
            self.last_query = None
        
        def query(self, ccy1, ccy2, asof, strict=False):
            self.last_query = (ccy1, ccy2, asof, strict)
            return None
        
        def queries(self, queries, strict=False):
            return []
    
    cad = Currency.of("CAD", "Canadian Dollar", 2, CurrencyType.MONEY)
    chf = Currency.of("CHF", "Swiss Franc", 2, CurrencyType.MONEY)
    test_date = date(2023, 12, 25)
    
    service = ConcreteFXRateService()
    service.query(cad, chf, test_date)
    
    assert service.last_query[0] == cad
    assert service.last_query[1] == chf
    assert service.last_query[2] == test_date
    assert service.last_query[3] is False


def test_fxrateservice_query_accepts_strict_parameter():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    class ConcreteFXRateService(FXRateService):
        def __init__(self):
            self.strict_flag = None
        
        def query(self, ccy1, ccy2, asof, strict=False):
            self.strict_flag = strict
            return None
        
        def queries(self, queries, strict=False):
            return []
    
    aud = Currency.of("AUD", "Australian Dollar", 2, CurrencyType.MONEY)
    nzd = Currency.of("NZD", "New Zealand Dollar", 2, CurrencyType.MONEY)
    test_date = date(2023, 3, 1)
    
    service = ConcreteFXRateService()
    service.query(aud, nzd, test_date, strict=True)
    
    assert service.strict_flag is True


# LLM-generated content at query #35
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


# LLM-generated content at query #36
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
    double_inverted_rate = ~(~original_rate)
    
    assert double_inverted_rate.ccy1 == original_rate.ccy1
    assert double_inverted_rate.ccy2 == original_rate.ccy2
    assert double_inverted_rate.date == original_rate.date
    assert double_inverted_rate.value == original_rate.value


def test_invert_fx_rate_with_decimal_one():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    
    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("1"))
    inverted_rate = ~rate
    
    assert inverted_rate.value == Decimal("1")
    assert inverted_rate.ccy1 == Currencies["USD"]
    assert inverted_rate.ccy2 == Currencies["EUR"]


def test_invert_fx_rate_with_fractional_value():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    
    rate = FXRate(Currencies["GBP"], Currencies["JPY"], datetime.date.today(), Decimal("150.5"))
    inverted_rate = ~rate
    
    assert inverted_rate.ccy1 == Currencies["JPY"]
    assert inverted_rate.ccy2 == Currencies["GBP"]
    assert inverted_rate.value == Decimal("1") / Decimal("150.5")


# LLM-generated content at query #2
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
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return MockFXRate(Decimal("1.5"))
        
        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    service = MockFXRateService()
    ccy1 = MockCurrency("USD")
    ccy2 = MockCurrency("EUR")
    test_date = date(2023, 1, 1)
    
    result = list(service.queries([(ccy1, ccy2, test_date)]))
    
    assert len(result) == 1
    assert result[0].rate == Decimal("1.5")


def test_queries_with_multiple_queries():
    from decimal import Decimal
    from datetime import date
    
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
    test_date = date(2023, 1, 1)
    
    result = list(service.queries([(ccy1, ccy2, test_date), (ccy2, ccy3, test_date)]))
    
    assert len(result) == 2
    assert all(r.rate == Decimal("1.5") for r in result)


def test_queries_returns_none_when_not_found():
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
    ccy2 = MockCurrency("EUR")
    test_date = date(2023, 1, 1)
    
    result = list(service.queries([(ccy1, ccy2, test_date)]))
    
    assert len(result) == 1
    assert result[0] is None


def test_queries_respects_strict_parameter():
    from datetime import date
    
    class MockCurrency:
        def __init__(self, code):
            self.code = code
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if strict:
                raise LookupError("FX rate not found")
            return None
        
        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    service = MockFXRateService()
    ccy1 = MockCurrency("USD")
    ccy2 = MockCurrency("EUR")
    test_date = date(2023, 1, 1)
    
    result = list(service.queries([(ccy1, ccy2, test_date)], strict=False))
    assert result[0] is None


# LLM-generated content at query #3
#--------------------------

```python
def test_queries_with_empty_iterable():
    from abc import ABCMeta
    from typing import Iterable, Optional, Tuple
    from datetime import date
    
    class MockCurrency:
        def __init__(self, code: str):
            self.code = code
    
    class MockFXRate:
        def __init__(self, rate: float):
            self.rate = rate
    
    class ConcreteFXRateService(object):
        def __init__(self):
            self._rates = {}
        
        def query(self, ccy1, ccy2, asof, strict=False):
            key = (ccy1.code, ccy2.code, asof)
            return self._rates.get(key)
        
        def queries(self, queries: Iterable, strict: bool = False) -> Iterable[Optional[MockFXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    service = ConcreteFXRateService()
    result = list(service.queries([]))
    assert result == []


def test_queries_with_single_query():
    from datetime import date
    
    class MockCurrency:
        def __init__(self, code: str):
            self.code = code
    
    class MockFXRate:
        def __init__(self, rate: float):
            self.rate = rate
    
    class ConcreteFXRateService(object):
        def __init__(self):
            self._rates = {}
        
        def query(self, ccy1, ccy2, asof, strict=False):
            key = (ccy1.code, ccy2.code, asof)
            return self._rates.get(key)
        
        def queries(self, queries: Iterable, strict: bool = False) -> Iterable[Optional[MockFXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    service = ConcreteFXRateService()
    usd = MockCurrency("USD")
    eur = MockCurrency("EUR")
    test_date = date(2023, 1, 1)
    
    service._rates[(usd.code, eur.code, test_date)] = MockFXRate(0.85)
    
    result = list(service.queries([(usd, eur, test_date)]))
    assert len(result) == 1
    assert result[0].rate == 0.85


def test_queries_with_multiple_queries():
    from datetime import date
    
    class MockCurrency:
        def __init__(self, code: str):
            self.code = code
    
    class MockFXRate:
        def __init__(self, rate: float):
            self.rate = rate
    
    class ConcreteFXRateService(object):
        def __init__(self):
            self._rates = {}
        
        def query(self, ccy1, ccy2, asof, strict=False):
            key = (ccy1.code, ccy2.code, asof)
            return self._rates.get(key)
        
        def queries(self, queries: Iterable, strict: bool = False) -> Iterable[Optional[MockFXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    service = ConcreteFXRateService()
    usd = MockCurrency("USD")
    eur = MockCurrency("EUR")
    gbp = MockCurrency("GBP")
    date1 = date(2023, 1, 1)
    date2 = date(2023, 1, 2)
    
    service._rates[(usd.code, eur.code, date1)] = MockFXRate(0.85)
    service._rates[(usd.code, gbp.code, date2)] = MockFXRate(0.73)
    
    result = list(service.queries([(usd, eur, date1), (usd, gbp, date2)]))
    assert len(result) == 2
    assert result[0].rate == 0.85
    assert result[1].rate == 0.73


def test_queries_with_missing_rates():
    from datetime import date
    
    class MockCurrency:
        def __init__(self, code: str):
            self.code = code
    
    class MockFXRate:
        def __init__(self, rate: float):
            self.rate = rate
    
    class ConcreteFXRateService(object):
        def __init__(self):
            self._rates = {}
        
        def query(self, ccy1, ccy2, asof, strict=False):
            key = (ccy1.code, ccy2.code, asof)
            return self._rates.get(key)
        
        def queries(self, queries: Iterable, strict: bool = False) -> Iterable[Optional[MockFXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    service = ConcreteFXRateService()
    usd = MockCurrency("USD")
    eur = MockCurrency("EUR")
    jpy = MockCurrency("JPY")
    date1 = date(2023, 1, 1)
    date2 = date(2023, 1, 2)
    
    service._rates[(usd.code, eur.code, date1)] = MockFXRate(0.85)
    
    result = list(service.queries([(usd, eur, date1), (usd, jpy, date2)]))
    assert len(result) == 2
    assert result[0].rate == 0.85
    assert result[1] is None


def test_queries_strict_mode():
    from datetime import date
    
    class MockCurrency:
        def __init__(self, code: str):
            self.code = code
    
    class MockFXRate:
        def __init__(self, rate: float):
            self.rate = rate
    
    class ConcreteFXRateService(object):
        def __init__(self):
            self._rates = {}
        
        def query(self, ccy1, ccy2, asof, strict=False):
            key = (ccy1.code, ccy2.code, asof)
            if strict and key not in self._rates:
                raise LookupError(f"Rate not found for {key}")
            return self._rates.get(key)
        
        def queries(self, queries: Iterable, strict: bool = False) -> Iterable[Optional[MockFXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    service = ConcreteFXRateService()
    usd = MockCurrency("USD")
    jpy = MockCurrency("JPY")
    date1 = date(2023, 1, 1)
    
    try:
        list(service.queries([(usd, jpy, date1)], strict=True))
        assert False, "Expected LookupError"
    except LookupError:
        assert True


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


# LLM-generated content at query #5
#--------------------------

```python
def test_fxrateservice_query_is_abstract():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    # Verify that FXRateService cannot be instantiated directly
    try:
        service = FXRateService()
        assert False, "Should not be able to instantiate abstract class"
    except TypeError:
        pass


def test_fxrateservice_query_with_concrete_implementation():
    from pypara.exchange import FXRateService, FXRate
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    from decimal import Decimal
    
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if ccy1.code == "USD" and ccy2.code == "EUR":
                return FXRate(ccy1, ccy2, asof, Decimal("0.85"))
            return None
        
        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    service = ConcreteFXRateService()
    usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    test_date = date(2023, 1, 1)
    
    result = service.query(usd, eur, test_date)
    assert result is not None
    assert result.ccy1 == usd
    assert result.ccy2 == eur
    assert result.asof == test_date
    assert result.rate == Decimal("0.85")


def test_fxrateservice_query_returns_none_when_not_found():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        
        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    service = ConcreteFXRateService()
    usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    gbp = Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY)
    test_date = date(2023, 1, 1)
    
    result = service.query(usd, gbp, test_date, strict=False)
    assert result is None


def test_fxrateservice_query_signature():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    import inspect
    
    sig = inspect.signature(FXRateService.query)
    params = list(sig.parameters.keys())
    assert params == ['self', 'ccy1', 'ccy2', 'asof', 'strict']
    assert sig.parameters['strict'].default == False


# LLM-generated content at query #6
#--------------------------

```python
def test_fxrate_constructor_with_valid_arguments():
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
    value = Decimal("1.23456789")
    
    rate = FXRate(ccy1, ccy2, date, value)
    
    assert rate.value == Decimal("1.23456789")


# LLM-generated content at query #7
#--------------------------

```python
def test_queries_with_multiple_valid_queries():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if ccy1.code == "USD" and ccy2.code == "EUR":
                return FXRate(ccy1, ccy2, asof, Decimal("0.85"))
            elif ccy1.code == "GBP" and ccy2.code == "USD":
                return FXRate(ccy1, ccy2, asof, Decimal("1.27"))
            return None

        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    from datetime import date
    from decimal import Decimal
    
    usd = Currency("USD")
    eur = Currency("EUR")
    gbp = Currency("GBP")
    test_date = date(2023, 1, 1)
    
    service = MockFXRateService()
    query_list = [
        (usd, eur, test_date),
        (gbp, usd, test_date),
    ]
    
    results = list(service.queries(query_list))
    
    assert len(results) == 2
    assert results[0] is not None
    assert results[0].rate == Decimal("0.85")
    assert results[1] is not None
    assert results[1].rate == Decimal("1.27")


def test_queries_with_empty_queries():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None

        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = MockFXRateService()
    query_list = []
    
    results = list(service.queries(query_list))
    
    assert len(results) == 0


def test_queries_with_none_results():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None

        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    from datetime import date
    
    usd = Currency("USD")
    eur = Currency("EUR")
    test_date = date(2023, 1, 1)
    
    service = MockFXRateService()
    query_list = [(usd, eur, test_date)]
    
    results = list(service.queries(query_list))
    
    assert len(results) == 1
    assert results[0] is None


def test_queries_with_strict_mode():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if strict and ccy1.code == "XXX":
                raise LookupError("Rate not found")
            return FXRate(ccy1, ccy2, asof, Decimal("1.0")) if ccy1.code != "XXX" else None

        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    from datetime import date
    
    usd = Currency("USD")
    xxx = Currency("XXX")
    test_date = date(2023, 1, 1)
    
    service = MockFXRateService()
    query_list = [(usd, xxx, test_date)]
    
    results = list(service.queries(query_list, strict=False))
    
    assert len(results) == 1
    assert results[0] is None


# LLM-generated content at query #8
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
    
    queries_list = [(ccy1, ccy2, asof1), (ccy1, ccy3, asof2), (ccy2, ccy3, asof1)]
    result = list(service.queries(queries_list))
    assert len(result) == 3
    assert all(rate.rate == Decimal("1.5") for rate in result)


def test_queries_with_strict_mode():
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
            if strict:
                raise LookupError("FX rate not found")
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
            return (self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries)
    
    service = MockFXRateService()
    ccy1 = MockCurrency("USD")
    ccy2 = MockCurrency("EUR")
    asof = date(2023, 1, 1)
    
    result = service.queries([(ccy1, ccy2, asof)])
    assert hasattr(result, '__iter__')


# LLM-generated content at query #9
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
            if strict:
                raise ValueError("Rate not found")
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
        def query(self, ccy1, ccy2, asof, strict=False):
            return (ccy1, ccy2, asof)
        
        def queries(self, queries, strict=False):
            return []
    
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    chf = Currency.of("CHF", "Swiss Franc", 2, CurrencyType.MONEY)
    test_date = date(2023, 12, 31)
    
    service = TestFXRateService()
    result = service.query(jpy, chf, test_date)
    
    assert result == (jpy, chf, test_date)


def test_fxrateservice_query_default_strict_is_false():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    class StrictCheckFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return strict
        
        def queries(self, queries, strict=False):
            return []
    
    aud = Currency.of("AUD", "Australian Dollar", 2, CurrencyType.MONEY)
    cad = Currency.of("CAD", "Canadian Dollar", 2, CurrencyType.MONEY)
    test_date = date(2023, 3, 20)
    
    service = StrictCheckFXRateService()
    result = service.query(aud, cad, test_date)
    
    assert result is False


# LLM-generated content at query #10
#--------------------------

```python
def test_fxrateservice_query_is_abstract():
    from abc import ABC, abstractmethod
    from pypara.currencies import Currency, CurrencyType
    from pypara.exchange import FXRateService
    from datetime import date
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    test_date = date(2023, 1, 1)
    
    try:
        service = FXRateService()
        assert False, "FXRateService should not be instantiable"
    except TypeError:
        pass


def test_fxrateservice_query_signature():
    from inspect import signature
    from pypara.exchange import FXRateService
    
    sig = signature(FXRateService.query)
    params = list(sig.parameters.keys())
    
    assert 'self' in params
    assert 'ccy1' in params
    assert 'ccy2' in params
    assert 'asof' in params
    assert 'strict' in params
    assert sig.parameters['strict'].default is False


def test_fxrateservice_query_has_abstractmethod_decorator():
    from pypara.exchange import FXRateService
    
    assert hasattr(FXRateService.query, '__isabstractmethod__')
    assert FXRateService.query.__isabstractmethod__ is True


def test_fxrateservice_query_return_type_annotation():
    from inspect import signature
    from pypara.exchange import FXRateService
    
    sig = signature(FXRateService.query)
    return_annotation = sig.return_annotation
    
    assert return_annotation is not signature.empty


# LLM-generated content at query #11
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
    asof_date = date(2023, 1, 1)
    
    service = ConcreteFXRateService()
    result = service.query(usd, eur, asof_date)
    
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


def test_fxrateservice_query_with_strict_true():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if strict:
                raise LookupError("FX rate not found")
            return None
        
        def queries(self, queries, strict=False):
            return []
    
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    cad = Currency.of("CAD", "Canadian Dollar", 2, CurrencyType.MONEY)
    test_date = date(2023, 12, 31)
    
    service = MockFXRateService()
    
    try:
        service.query(jpy, cad, test_date, strict=True)
        assert False, "Expected LookupError"
    except LookupError:
        assert True


def test_fxrateservice_query_same_currency():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        
        def queries(self, queries, strict=False):
            return []
    
    aud = Currency.of("AUD", "Australian Dollar", 2, CurrencyType.MONEY)
    test_date = date(2023, 3, 20)
    
    service = MockFXRateService()
    result = service.query(aud, aud, test_date)
    
    assert result is None


def test_fxrateservice_query_different_dates():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        
        def queries(self, queries, strict=False):
            return []
    
    chf = Currency.of("CHF", "Swiss Franc", 2, CurrencyType.MONEY)
    sek = Currency.of("SEK", "Swedish Krona", 2, CurrencyType.MONEY)
    
    service = MockFXRateService()
    result1 = service.query(chf, sek, date(2023, 1, 1))
    result2 = service.query(chf, sek, date(2023, 12, 31))
    
    assert result1 is None
    assert result2 is None


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


def test_fxrate_constructor_tuple_unpacking():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("2")
    
    rate = FXRate(ccy1, ccy2, date, value)
    c1, c2, d, v = rate
    
    assert c1 == ccy1
    assert c2 == ccy2
    assert d == date
    assert v == value


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
    date = datetime.date(2023, 1, 15)
    value = Decimal("150.5")
    
    rate = FXRate(ccy1, ccy2, date, value)
    
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value


# LLM-generated content at query #15
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


def test_fxrate_constructor_unpacking():
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


def test_fxrate_constructor_unpacking():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("2")
    
    rate = FXRate(ccy1, ccy2, date, value)
    c1, c2, d, v = rate
    
    assert c1 == ccy1
    assert c2 == ccy2
    assert d == date
    assert v == value


def test_fxrate_constructor_with_different_values():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    
    ccy1 = Currencies["GBP"]
    ccy2 = Currencies["JPY"]
    date = datetime.date(2023, 6, 15)
    value = Decimal("150.75")
    
    rate = FXRate(ccy1, ccy2, date, value)
    
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value


# LLM-generated content at query #18
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
    from decimal import Decimal
    from datetime import date
    
    class MockCurrency:
        def __init__(self, code):
            self.code = code
    
    class MockFXRate:
        def __init__(self, value):
            self.value = value
    
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
    assert result[0].value == Decimal("1.5")


def test_queries_with_multiple_queries():
    from decimal import Decimal
    from datetime import date
    
    class MockCurrency:
        def __init__(self, code):
            self.code = code
    
    class MockFXRate:
        def __init__(self, value):
            self.value = value
    
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
    
    result = list(service.queries([(ccy1, ccy2, asof1), (ccy1, ccy3, asof2)]))
    assert len(result) == 2


def test_queries_with_strict_false():
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
    ccy2 = MockCurrency("EUR")
    asof = date(2023, 1, 1)
    
    result = list(service.queries([(ccy1, ccy2, asof)], strict=False))
    assert len(result) == 1
    assert result[0] is None


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


# LLM-generated content at query #19
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
        def query(self, ccy1, ccy2, asof, strict=False):
            return (ccy1, ccy2, asof)
        
        def queries(self, queries, strict=False):
            return []
    
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    chf = Currency.of("CHF", "Swiss Franc", 2, CurrencyType.MONEY)
    test_date = date(2023, 12, 25)
    
    service = TestFXRateService()
    result = service.query(jpy, chf, test_date)
    
    assert result == (jpy, chf, test_date)


def test_fxrateservice_query_default_strict_parameter():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    class RecordingFXRateService(FXRateService):
        def __init__(self):
            self.strict_value = None
        
        def query(self, ccy1, ccy2, asof, strict=False):
            self.strict_value = strict
            return None
        
        def queries(self, queries, strict=False):
            return []
    
    aud = Currency.of("AUD", "Australian Dollar", 2, CurrencyType.MONEY)
    cad = Currency.of("CAD", "Canadian Dollar", 2, CurrencyType.MONEY)
    test_date = date(2023, 3, 10)
    
    service = RecordingFXRateService()
    service.query(aud, cad, test_date)
    
    assert service.strict_value is False


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
    
    rate = FXRate(Currencies["GBP"], Currencies["JPY"], datetime.date.today(), Decimal("150.5"))
    
    assert rate.ccy1 == Currencies["GBP"]
    assert rate.ccy2 == Currencies["JPY"]
    assert rate.value == Decimal("150.5")


def test_fxrate_constructor_with_small_decimal_value():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    
    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("0.001"))
    
    assert rate.value == Decimal("0.001")


def test_fxrate_constructor_with_large_decimal_value():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    
    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("9999.99"))
    
    assert rate.value == Decimal("9999.99")


# LLM-generated content at query #21
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


# LLM-generated content at query #22
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


def test_fxrateservice_query_accepts_different_currencies():
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
    test_date = date(2023, 3, 20)
    
    service = TestFXRateService()
    result = service.query(jpy, usd, test_date)
    
    assert result is None


def test_fxrateservice_query_accepts_date_parameter():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    class DateAwareFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return asof
        
        def queries(self, queries, strict=False):
            return []
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    test_date = date(2023, 12, 25)
    
    service = DateAwareFXRateService()
    result = service.query(usd, eur, test_date)
    
    assert result == test_date


# LLM-generated content at query #23
#--------------------------

```python
def test_queries_returns_iterable_of_rates():
    from abc import ABC, abstractmethod
    from typing import Tuple, Iterable, Optional
    from datetime import date
    from decimal import Decimal
    
    class MockCurrency:
        def __init__(self, code: str):
            self.code = code
    
    class MockFXRate:
        def __init__(self, rate: Decimal):
            self.rate = rate
    
    class ConcreteFXRateService(ABC):
        TQuery = Tuple[MockCurrency, MockCurrency, date]
        
        @abstractmethod
        def query(self, ccy1: MockCurrency, ccy2: MockCurrency, asof: date, strict: bool = False) -> Optional[MockFXRate]:
            pass
        
        @abstractmethod
        def queries(self, queries: Iterable[TQuery], strict: bool = False) -> Iterable[Optional[MockFXRate]]:
            pass
    
    class TestFXRateService(ConcreteFXRateService):
        def query(self, ccy1: MockCurrency, ccy2: MockCurrency, asof: date, strict: bool = False) -> Optional[MockFXRate]:
            return MockFXRate(Decimal("1.5"))
        
        def queries(self, queries: Iterable[TQuery], strict: bool = False) -> Iterable[Optional[MockFXRate]]:
            return [MockFXRate(Decimal("1.5")), MockFXRate(Decimal("2.0")), None]
    
    service = TestFXRateService()
    ccy1 = MockCurrency("USD")
    ccy2 = MockCurrency("EUR")
    test_date = date(2023, 1, 1)
    
    query_list = [(ccy1, ccy2, test_date), (ccy1, ccy2, test_date), (ccy1, ccy2, test_date)]
    results = list(service.queries(query_list, strict=False))
    
    assert len(results) == 3
    assert results[0] is not None
    assert results[0].rate == Decimal("1.5")
    assert results[1] is not None
    assert results[1].rate == Decimal("2.0")
    assert results[2] is None


def test_queries_with_strict_mode():
    from abc import ABC, abstractmethod
    from typing import Tuple, Iterable, Optional
    from datetime import date
    from decimal import Decimal
    
    class MockCurrency:
        def __init__(self, code: str):
            self.code = code
    
    class MockFXRate:
        def __init__(self, rate: Decimal):
            self.rate = rate
    
    class ConcreteFXRateService(ABC):
        TQuery = Tuple[MockCurrency, MockCurrency, date]
        
        @abstractmethod
        def query(self, ccy1: MockCurrency, ccy2: MockCurrency, asof: date, strict: bool = False) -> Optional[MockFXRate]:
            pass
        
        @abstractmethod
        def queries(self, queries: Iterable[TQuery], strict: bool = False) -> Iterable[Optional[MockFXRate]]:
            pass
    
    class TestFXRateService(ConcreteFXRateService):
        def query(self, ccy1: MockCurrency, ccy2: MockCurrency, asof: date, strict: bool = False) -> Optional[MockFXRate]:
            return MockFXRate(Decimal("1.5"))
        
        def queries(self, queries: Iterable[TQuery], strict: bool = False) -> Iterable[Optional[MockFXRate]]:
            if strict:
                return [MockFXRate(Decimal("1.5")) for _ in queries]
            return [MockFXRate(Decimal("1.5")), None]
    
    service = TestFXRateService()
    ccy1 = MockCurrency("USD")
    ccy2 = MockCurrency("EUR")
    test_date = date(2023, 1, 1)
    
    query_list = [(ccy1, ccy2, test_date), (ccy1, ccy2, test_date)]
    results = list(service.queries(query_list, strict=True))
    
    assert len(results) == 2
    assert all(r is not None for r in results)


def test_queries_accepts_iterable_input():
    from abc import ABC, abstractmethod
    from typing import Tuple, Iterable, Optional
    from datetime import date
    from decimal import Decimal
    
    class MockCurrency:
        def __init__(self, code: str):
            self.code = code
    
    class MockFXRate:
        def __init__(self, rate: Decimal):
            self.rate = rate
    
    class ConcreteFXRateService(ABC):
        TQuery = Tuple[MockCurrency, MockCurrency, date]
        
        @abstractmethod
        def query(self, ccy1: MockCurrency, ccy2: MockCurrency, asof: date, strict: bool = False) -> Optional[MockFXRate]:
            pass
        
        @abstractmethod
        def queries(self, queries: Iterable[TQuery], strict: bool = False) -> Iterable[Optional[MockFXRate]]:
            pass
    
    class TestFXRateService(ConcreteFXRateService):
        def query(self, ccy1: MockCurrency, ccy2: MockCurrency, asof: date, strict: bool = False) -> Optional[MockFXRate]:
            return MockFXRate(Decimal("1.5"))
        
        def queries(self, queries: Iterable[TQuery], strict: bool = False) -> Iterable[Optional[MockFXRate]]:
            return [MockFXRate(Decimal("1.5")) for _ in queries]
    
    service = TestFXRateService()
    ccy1 = MockCurrency("USD")
    ccy2 = MockCurrency("EUR")
    test_date = date(2023, 1, 1)
    
    query_generator = ((ccy1, ccy2, test_date) for _ in range(3))
    results = list(service.queries(query_generator))
    
    assert len(results) == 3
    assert all(r is not None for r in results)


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
            if strict:
                raise LookupError("Rate not found")
            return None
        
        def queries(self, queries, strict=False):
            return []
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    gbp = Currency.of("GBP", "British Pounds", 2, CurrencyType.MONEY)
    test_date = date(2023, 6, 15)
    
    service = TestFXRateService()
    result = service.query(usd, gbp, test_date, strict=False)
    
    assert result is None


def test_fxrateservice_query_with_strict_true():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if strict:
                raise LookupError("FX rate not found")
            return None
        
        def queries(self, queries, strict=False):
            return []
    
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    chf = Currency.of("CHF", "Swiss Francs", 2, CurrencyType.MONEY)
    test_date = date(2023, 12, 31)
    
    service = TestFXRateService()
    
    try:
        service.query(jpy, chf, test_date, strict=True)
        assert False, "Expected LookupError"
    except LookupError:
        assert True


def test_fxrateservice_query_default_parameter():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return {"ccy1": ccy1, "ccy2": ccy2, "asof": asof, "strict": strict}
        
        def queries(self, queries, strict=False):
            return []
    
    cad = Currency.of("CAD", "Canadian Dollars", 2, CurrencyType.MONEY)
    aud = Currency.of("AUD", "Australian Dollars", 2, CurrencyType.MONEY)
    test_date = date(2023, 3, 20)
    
    service = TestFXRateService()
    result = service.query(cad, aud, test_date)
    
    assert result["ccy1"] == cad
    assert result["ccy2"] == aud
    assert result["asof"] == test_date
    assert result["strict"] is False


# LLM-generated content at query #27
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
    
    class ConcreteFXRateService:
        def __init__(self):
            self.rates = {}
        
        def queries(self, queries: Iterable[Tuple[Currency, Currency, date]], strict: bool = False) -> Iterable[Optional[FXRate]]:
            results = []
            for ccy1, ccy2, asof in queries:
                key = (ccy1.code, ccy2.code, asof)
                if key in self.rates:
                    results.append(self.rates[key])
                elif strict:
                    raise LookupError(f"Rate not found for {ccy1.code}/{ccy2.code} on {asof}")
                else:
                    results.append(None)
            return results
    
    service = ConcreteFXRateService()
    result = list(service.queries([]))
    assert result == []


def test_queries_with_single_rate_found():
    from datetime import date
    
    class Currency:
        def __init__(self, code: str):
            self.code = code
    
    class FXRate:
        def __init__(self, rate: float):
            self.rate = rate
    
    class ConcreteFXRateService:
        def __init__(self):
            self.rates = {}
        
        def queries(self, queries, strict: bool = False):
            results = []
            for ccy1, ccy2, asof in queries:
                key = (ccy1.code, ccy2.code, asof)
                if key in self.rates:
                    results.append(self.rates[key])
                elif strict:
                    raise LookupError(f"Rate not found for {ccy1.code}/{ccy2.code} on {asof}")
                else:
                    results.append(None)
            return results
    
    service = ConcreteFXRateService()
    usd = Currency("USD")
    eur = Currency("EUR")
    test_date = date(2023, 1, 1)
    fx_rate = FXRate(1.1)
    service.rates[(usd.code, eur.code, test_date)] = fx_rate
    
    result = list(service.queries([(usd, eur, test_date)]))
    assert len(result) == 1
    assert result[0] == fx_rate


def test_queries_with_multiple_rates_mixed():
    from datetime import date
    
    class Currency:
        def __init__(self, code: str):
            self.code = code
    
    class FXRate:
        def __init__(self, rate: float):
            self.rate = rate
    
    class ConcreteFXRateService:
        def __init__(self):
            self.rates = {}
        
        def queries(self, queries, strict: bool = False):
            results = []
            for ccy1, ccy2, asof in queries:
                key = (ccy1.code, ccy2.code, asof)
                if key in self.rates:
                    results.append(self.rates[key])
                elif strict:
                    raise LookupError(f"Rate not found for {ccy1.code}/{ccy2.code} on {asof}")
                else:
                    results.append(None)
            return results
    
    service = ConcreteFXRateService()
    usd = Currency("USD")
    eur = Currency("EUR")
    gbp = Currency("GBP")
    date1 = date(2023, 1, 1)
    date2 = date(2023, 1, 2)
    
    fx_rate1 = FXRate(1.1)
    service.rates[(usd.code, eur.code, date1)] = fx_rate1
    
    queries_list = [(usd, eur, date1), (usd, gbp, date2), (eur, gbp, date1)]
    result = list(service.queries(queries_list))
    
    assert len(result) == 3
    assert result[0] == fx_rate1
    assert result[1] is None
    assert result[2] is None


def test_queries_with_strict_mode_raises_error():
    from datetime import date
    
    class Currency:
        def __init__(self, code: str):
            self.code = code
    
    class FXRate:
        def __init__(self, rate: float):
            self.rate = rate
    
    class ConcreteFXRateService:
        def __init__(self):
            self.rates = {}
        
        def queries(self, queries, strict: bool = False):
            results = []
            for ccy1, ccy2, asof in queries:
                key = (ccy1.code, ccy2.code, asof)
                if key in self.rates:
                    results.append(self.rates[key])
                elif strict:
                    raise LookupError(f"Rate not found for {ccy1.code}/{ccy2.code} on {asof}")
                else:
                    results.append(None)
            return results
    
    service = ConcreteFXRateService()
    usd = Currency("USD")
    eur = Currency("EUR")
    test_date = date(2023, 1, 1)
    
    try:
        list(service.queries([(usd, eur, test_date)], strict=True))
        assert False, "Expected LookupError to be raised"
    except LookupError:
        assert True


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


def test_fxrate_constructor_different_currencies():
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


def test_fxrate_constructor_small_value():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("0.0001")
    
    rate = FXRate(ccy1, ccy2, date, value)
    
    assert rate.value == value


def test_fxrate_constructor_large_value():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("99999.999")
    
    rate = FXRate(ccy1, ccy2, date, value)
    
    assert rate.value == value


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


def test_fxrateservice_query_with_strict_true():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if strict:
                return None
            return None
        
        def queries(self, queries, strict=False):
            return []
    
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    chf = Currency.of("CHF", "Swiss Francs", 2, CurrencyType.MONEY)
    test_date = date(2023, 12, 31)
    
    service = TestFXRateService()
    result = service.query(jpy, chf, test_date, strict=True)
    
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
    
    cad = Currency.of("CAD", "Canadian Dollars", 2, CurrencyType.MONEY)
    aud = Currency.of("AUD", "Australian Dollars", 2, CurrencyType.MONEY)
    test_date = date(2023, 3, 10)
    
    service = TestFXRateService()
    service.query(cad, aud, test_date, strict=False)
    
    assert service.last_query == (cad, aud, test_date, False)


# LLM-generated content at query #31
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
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return MockFXRate(Decimal("1.5"))
        
        def queries(self, queries, strict=False):
            return [MockFXRate(Decimal("1.5")) for _ in queries]
    
    service = MockFXRateService()
    ccy1 = MockCurrency("USD")
    ccy2 = MockCurrency("EUR")
    test_date = date(2023, 1, 1)
    
    query_list = [(ccy1, ccy2, test_date)]
    results = list(service.queries(query_list, strict=False))
    
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
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return MockFXRate(Decimal("1.5"))
        
        def queries(self, queries, strict=False):
            return [MockFXRate(Decimal("1.5")) for _ in queries]
    
    service = MockFXRateService()
    ccy1 = MockCurrency("USD")
    ccy2 = MockCurrency("EUR")
    ccy3 = MockCurrency("GBP")
    test_date = date(2023, 1, 1)
    
    query_list = [(ccy1, ccy2, test_date), (ccy1, ccy3, test_date), (ccy2, ccy3, test_date)]
    results = list(service.queries(query_list, strict=False))
    
    assert len(results) == 3
    assert all(result.rate == Decimal("1.5") for result in results)


def test_queries_with_strict_mode():
    from decimal import Decimal
    from datetime import date
    
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
            return [MockFXRate(Decimal("1.5")) for _ in queries]
    
    service = MockFXRateService()
    ccy1 = MockCurrency("USD")
    ccy2 = MockCurrency("EUR")
    test_date = date(2023, 1, 1)
    
    query_list = [(ccy1, ccy2, test_date)]
    results = list(service.queries(query_list, strict=True))
    
    assert len(results) == 1
    assert results[0].rate == Decimal("1.5")


def test_queries_returns_iterable():
    from datetime import date
    
    class MockCurrency:
        def __init__(self, code):
            self.code = code
    
    class MockFXRate:
        def __init__(self, rate):
            self.rate = rate
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        
        def queries(self, queries, strict=False):
            return iter([None for _ in queries])
    
    service = MockFXRateService()
    ccy1 = MockCurrency("USD")
    ccy2 = MockCurrency("EUR")
    test_date = date(2023, 1, 1)
    
    query_list = [(ccy1, ccy2, test_date)]
    results = service.queries(query_list, strict=False)
    
    assert hasattr(results, '__iter__')


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


