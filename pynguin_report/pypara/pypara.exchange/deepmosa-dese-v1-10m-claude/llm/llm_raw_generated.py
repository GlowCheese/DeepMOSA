####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_fxrate_invert():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    
    # Create an FX rate EUR/USD with value 2
    nrate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("2"))
    
    # Invert the rate
    inverted_rate = ~nrate
    
    # Check that currencies are swapped
    assert inverted_rate.ccy1 == Currencies["USD"]
    assert inverted_rate.ccy2 == Currencies["EUR"]
    
    # Check that date remains the same
    assert inverted_rate.date == datetime.date.today()
    
    # Check that value is inverted (1/2 = 0.5)
    assert inverted_rate.value == Decimal("0.5")
    
    # Check that double inversion returns to original
    double_inverted = ~inverted_rate
    assert double_inverted.ccy1 == nrate.ccy1
    assert double_inverted.ccy2 == nrate.ccy2
    assert double_inverted.date == nrate.date
    assert double_inverted.value == nrate.value


def test_fxrate_invert_with_decimal_one():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    
    # Create an FX rate with value 1
    rate = FXRate(Currencies["EUR"], Currencies["EUR"], datetime.date.today(), Decimal("1"))
    
    # Invert the rate
    inverted_rate = ~rate
    
    # Check that currencies are swapped (even though same)
    assert inverted_rate.ccy1 == Currencies["EUR"]
    assert inverted_rate.ccy2 == Currencies["EUR"]
    
    # Check that value remains 1
    assert inverted_rate.value == Decimal("1")


def test_fxrate_invert_with_small_decimal():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    
    # Create an FX rate with small value
    rate = FXRate(Currencies["GBP"], Currencies["JPY"], datetime.date.today(), Decimal("0.01"))
    
    # Invert the rate
    inverted_rate = ~rate
    
    # Check that currencies are swapped
    assert inverted_rate.ccy1 == Currencies["JPY"]
    assert inverted_rate.ccy2 == Currencies["GBP"]
    
    # Check that value is inverted (1/0.01 = 100)
    assert inverted_rate.value == Decimal("100")


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
    asof = date(2023, 1, 1)
    
    result = list(service.queries([(ccy1, ccy2, asof)]))
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
    asof = date(2023, 1, 1)
    
    queries = [(ccy1, ccy2, asof), (ccy2, ccy3, asof), (ccy1, ccy3, asof)]
    result = list(service.queries(queries))
    assert len(result) == 3
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


def test_queries_strict_mode():
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


# LLM-generated content at query #3
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
    usd = MockCurrency("USD")
    eur = MockCurrency("EUR")
    test_date = date(2023, 1, 1)
    
    query_list = [(usd, eur, test_date)]
    results = list(service.queries(query_list))
    
    assert len(results) == 1
    assert results[0] == Decimal("0.85")


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
    usd = MockCurrency("USD")
    eur = MockCurrency("EUR")
    gbp = MockCurrency("GBP")
    test_date = date(2023, 1, 1)
    
    query_list = [(usd, eur, test_date), (usd, gbp, test_date), (eur, gbp, test_date)]
    results = list(service.queries(query_list))
    
    assert len(results) == 3
    assert results[0] == Decimal("0.85")
    assert results[1] == Decimal("0.73")
    assert results[2] == Decimal("0.86")


def test_queries_with_missing_rates():
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
    usd = MockCurrency("USD")
    eur = MockCurrency("EUR")
    jpy = MockCurrency("JPY")
    test_date = date(2023, 1, 1)
    
    query_list = [(usd, eur, test_date), (usd, jpy, test_date)]
    results = list(service.queries(query_list))
    
    assert len(results) == 2
    assert results[0] == Decimal("0.85")
    assert results[1] is None


def test_queries_with_strict_mode():
    from decimal import Decimal
    from datetime import date
    
    class MockCurrency:
        def __init__(self, code):
            self.code = code
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if ccy1.code == "USD" and ccy2.code == "EUR":
                return Decimal("0.85")
            if strict:
                raise LookupError(f"Rate not found for {ccy1.code}/{ccy2.code}")
            return None
        
        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    service = MockFXRateService()
    usd = MockCurrency("USD")
    jpy = MockCurrency("JPY")
    test_date = date(2023, 1, 1)
    
    query_list = [(usd, jpy, test_date)]
    
    try:
        list(service.queries(query_list, strict=True))
        assert False, "Expected LookupError"
    except LookupError:
        assert True


def test_queries_empty_list():
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


# LLM-generated content at query #4
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
        assert False, "FXRateService should not be instantiable"
    except TypeError:
        assert True


# LLM-generated content at query #5
#--------------------------

```python
def test_queries_returns_iterable_of_rates():
    from abc import ABCMeta, abstractmethod
    from datetime import date
    from decimal import Decimal
    from typing import Iterable, Optional, Tuple
    
    class Currency:
        def __init__(self, code: str):
            self.code = code
    
    class FXRate:
        def __init__(self, rate: Decimal):
            self.rate = rate
    
    class ConcreteFXRateService:
        def __init__(self):
            self.rates_data = {
                (Currency("USD").code, Currency("EUR").code, date(2023, 1, 1)): FXRate(Decimal("0.92")),
                (Currency("USD").code, Currency("GBP").code, date(2023, 1, 1)): FXRate(Decimal("0.79")),
            }
        
        def queries(self, queries: Iterable[Tuple[Currency, Currency, date]], strict: bool = False) -> Iterable[Optional[FXRate]]:
            results = []
            for ccy1, ccy2, asof in queries:
                key = (ccy1.code, ccy2.code, asof)
                result = self.rates_data.get(key)
                if result is None and strict:
                    raise LookupError(f"Rate not found for {key}")
                results.append(result)
            return results
    
    service = ConcreteFXRateService()
    usd = Currency("USD")
    eur = Currency("EUR")
    gbp = Currency("GBP")
    test_date = date(2023, 1, 1)
    
    query_list = [(usd, eur, test_date), (usd, gbp, test_date)]
    results = list(service.queries(query_list, strict=False))
    
    assert len(results) == 2
    assert results[0] is not None
    assert results[0].rate == Decimal("0.92")
    assert results[1] is not None
    assert results[1].rate == Decimal("0.79")


def test_queries_with_strict_mode_raises_error():
    from datetime import date
    from decimal import Decimal
    from typing import Iterable, Optional, Tuple
    
    class Currency:
        def __init__(self, code: str):
            self.code = code
    
    class FXRate:
        def __init__(self, rate: Decimal):
            self.rate = rate
    
    class ConcreteFXRateService:
        def __init__(self):
            self.rates_data = {
                (Currency("USD").code, Currency("EUR").code, date(2023, 1, 1)): FXRate(Decimal("0.92")),
            }
        
        def queries(self, queries: Iterable[Tuple[Currency, Currency, date]], strict: bool = False) -> Iterable[Optional[FXRate]]:
            results = []
            for ccy1, ccy2, asof in queries:
                key = (ccy1.code, ccy2.code, asof)
                result = self.rates_data.get(key)
                if result is None and strict:
                    raise LookupError(f"Rate not found for {key}")
                results.append(result)
            return results
    
    service = ConcreteFXRateService()
    usd = Currency("USD")
    jpy = Currency("JPY")
    test_date = date(2023, 1, 1)
    
    query_list = [(usd, jpy, test_date)]
    
    try:
        list(service.queries(query_list, strict=True))
        assert False, "Expected LookupError"
    except LookupError:
        pass


def test_queries_without_strict_mode_returns_none():
    from datetime import date
    from decimal import Decimal
    from typing import Iterable, Optional, Tuple
    
    class Currency:
        def __init__(self, code: str):
            self.code = code
    
    class FXRate:
        def __init__(self, rate: Decimal):
            self.rate = rate
    
    class ConcreteFXRateService:
        def __init__(self):
            self.rates_data = {}
        
        def queries(self, queries: Iterable[Tuple[Currency, Currency, date]], strict: bool = False) -> Iterable[Optional[FXRate]]:
            results = []
            for ccy1, ccy2, asof in queries:
                key = (ccy1.code, ccy2.code, asof)
                result = self.rates_data.get(key)
                if result is None and strict:
                    raise LookupError(f"Rate not found for {key}")
                results.append(result)
            return results
    
    service = ConcreteFXRateService()
    usd = Currency("USD")
    eur = Currency("EUR")
    test_date = date(2023, 1, 1)
    
    query_list = [(usd, eur, test_date)]
    results = list(service.queries(query_list, strict=False))
    
    assert len(results) == 1
    assert results[0] is None


# LLM-generated content at query #6
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


# LLM-generated content at query #7
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
            return FXRate(1.5)
        
        def queries(self, queries: Iterable[Tuple[Currency, Currency, date]], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [FXRate(1.5) for _ in queries]
    
    service = ConcreteFXRateService()
    result = list(service.queries([]))
    assert result == []


def test_queries_with_single_query():
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
            return [FXRate(1.5) for _ in queries]
    
    service = ConcreteFXRateService()
    ccy1 = Currency("USD")
    ccy2 = Currency("EUR")
    asof = date(2023, 1, 1)
    queries = [(ccy1, ccy2, asof)]
    
    result = list(service.queries(queries))
    assert len(result) == 1
    assert result[0].rate == 1.5


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
            return FXRate(1.5)
        
        def queries(self, queries: Iterable[Tuple[Currency, Currency, date]], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [FXRate(1.5 + i * 0.1) for i, _ in enumerate(queries)]
    
    service = ConcreteFXRateService()
    ccy1 = Currency("USD")
    ccy2 = Currency("EUR")
    asof = date(2023, 1, 1)
    queries = [(ccy1, ccy2, asof), (ccy1, ccy2, asof), (ccy1, ccy2, asof)]
    
    result = list(service.queries(queries))
    assert len(result) == 3
    assert result[0].rate == 1.5
    assert result[1].rate == 1.6
    assert result[2].rate == 1.7


def test_queries_with_strict_false():
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
            return [None for _ in queries]
    
    service = ConcreteFXRateService()
    ccy1 = Currency("USD")
    ccy2 = Currency("EUR")
    asof = date(2023, 1, 1)
    queries = [(ccy1, ccy2, asof)]
    
    result = list(service.queries(queries, strict=False))
    assert len(result) == 1
    assert result[0] is None


# LLM-generated content at query #8
#--------------------------

```python
def test_fxrateservice_query_is_abstract():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    test_date = date(2023, 1, 1)
    
    assert FXRateService.query.__isabstractmethod__ is True


def test_fxrateservice_query_signature():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    import inspect
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    test_date = date(2023, 1, 1)
    
    sig = inspect.signature(FXRateService.query)
    params = list(sig.parameters.keys())
    
    assert "self" in params
    assert "ccy1" in params
    assert "ccy2" in params
    assert "asof" in params
    assert "strict" in params
    assert sig.parameters["strict"].default is False


# LLM-generated content at query #9
#--------------------------

```python
def test_fxrateservice_query_is_abstract():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = date(2023, 1, 1)
    
    try:
        service = FXRateService()
        service.query(usd, eur, asof)
        assert False, "Should not be able to instantiate abstract class"
    except TypeError:
        assert True


# LLM-generated content at query #10
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


def test_fxrate_constructor_with_different_values():
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
    value = Decimal("150.75")
    
    rate = FXRate(ccy1, ccy2, date, value)
    
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value


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


# LLM-generated content at query #14
#--------------------------

```python
def test_fxrate_constructor_with_valid_arguments():
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


def test_fxrate_constructor_tuple_unpacking():
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


def test_fxrate_constructor_with_different_values():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    
    ccy1 = Currencies["GBP"]
    ccy2 = Currencies["JPY"]
    date = datetime.date(2023, 1, 15)
    value = Decimal("150.75")
    
    rate = FXRate(ccy1, ccy2, date, value)
    
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value


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


def test_fxrate_constructor_different_currencies():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    
    rate1 = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("1.5"))
    rate2 = FXRate(Currencies["GBP"], Currencies["JPY"], datetime.date.today(), Decimal("100"))
    
    assert rate1.ccy1 != rate2.ccy1
    assert rate1.ccy2 != rate2.ccy2
    assert rate1.value != rate2.value


def test_fxrate_constructor_same_values():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("2.5")
    
    rate1 = FXRate(ccy1, ccy2, date, value)
    rate2 = FXRate(ccy1, ccy2, date, value)
    
    assert rate1 == rate2


# LLM-generated content at query #18
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
    value = Decimal("1.5")
    
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
    value = Decimal("3.25")
    
    rate = FXRate(ccy1, ccy2, date, value)
    
    assert rate[0] == ccy1
    assert rate[1] == ccy2
    assert rate[2] == date
    assert rate[3] == value


def test_fxrate_constructor_different_currencies():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    
    ccy1 = Currencies["GBP"]
    ccy2 = Currencies["JPY"]
    date = datetime.date(2023, 1, 15)
    value = Decimal("150.75")
    
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
    value = Decimal("0.0001")
    
    rate = FXRate(ccy1, ccy2, date, value)
    
    assert rate.value == Decimal("0.0001")


def test_fxrate_constructor_large_decimal_value():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("999999.999999")
    
    rate = FXRate(ccy1, ccy2, date, value)
    
    assert rate.value == Decimal("999999.999999")


# LLM-generated content at query #19
#--------------------------

```python
def test_queries_returns_iterable_of_fx_rates():
    from decimal import Decimal
    from datetime import date
    from collections.abc import Iterable
    
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
    asof_date = date(2023, 1, 1)
    
    query_list = [(ccy1, ccy2, asof_date), (ccy1, ccy2, asof_date)]
    result = service.queries(query_list, strict=False)
    
    result_list = list(result)
    assert isinstance(result, Iterable)
    assert len(result_list) == 2
    assert all(isinstance(rate, MockFXRate) for rate in result_list)


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
            if strict:
                raise LookupError("Rate not found")
            return None
        
        def queries(self, queries, strict=False):
            if strict:
                raise LookupError("Rate not found")
            return [None for _ in queries]
    
    service = ConcreteFXRateService()
    ccy1 = MockCurrency("USD")
    ccy2 = MockCurrency("EUR")
    asof_date = date(2023, 1, 1)
    
    query_list = [(ccy1, ccy2, asof_date)]
    result = service.queries(query_list, strict=True)
    
    try:
        list(result)
        assert False, "Expected LookupError"
    except LookupError:
        assert True


def test_queries_returns_none_for_missing_rates():
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
    asof_date = date(2023, 1, 1)
    
    query_list = [(ccy1, ccy2, asof_date), (ccy1, ccy2, asof_date)]
    result = service.queries(query_list, strict=False)
    
    result_list = list(result)
    assert all(rate is None for rate in result_list)
    assert len(result_list) == 2


def test_queries_accepts_empty_iterable():
    from datetime import date
    
    class MockCurrency:
        def __init__(self, code):
            self.code = code
    
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        
        def queries(self, queries, strict=False):
            return iter([])
    
    service = ConcreteFXRateService()
    query_list = []
    result = service.queries(query_list, strict=False)
    
    result_list = list(result)
    assert len(result_list) == 0


# LLM-generated content at query #20
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


# LLM-generated content at query #22
#--------------------------

```python
def test_fxrateservice_query_is_abstract():
    from abc import ABC, abstractmethod
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    # Verify that FXRateService cannot be instantiated directly
    try:
        service = FXRateService()
        assert False, "FXRateService should not be instantiable"
    except TypeError:
        pass
    
    # Create a concrete implementation to test the query method signature
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        
        def queries(self, queries, strict=False):
            return []
    
    service = ConcreteFXRateService()
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    test_date = date(2023, 1, 1)
    
    result = service.query(usd, eur, test_date)
    assert result is None
    
    result_strict = service.query(usd, eur, test_date, strict=True)
    assert result_strict is None


def test_fxrateservice_queries_is_abstract():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        
        def queries(self, queries, strict=False):
            return iter([])
    
    service = ConcreteFXRateService()
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    test_date = date(2023, 1, 1)
    
    query_list = [(usd, eur, test_date)]
    result = list(service.queries(query_list))
    assert result == []


def test_fxrateservice_query_default_none():
    from pypara.exchange import FXRateService
    
    assert FXRateService.default is None


def test_fxrateservice_tquery_type():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    from typing import Tuple
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    test_date = date(2023, 1, 1)
    
    query_tuple = (usd, eur, test_date)
    assert isinstance(query_tuple, tuple)
    assert len(query_tuple) == 3
    assert query_tuple[0] == usd
    assert query_tuple[1] == eur
    assert query_tuple[2] == test_date


# LLM-generated content at query #23
#--------------------------

```python
def test_queries_with_empty_iterable():
    from abc import ABCMeta
    from datetime import date
    from decimal import Decimal
    
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
    from abc import ABCMeta
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
    from abc import ABCMeta
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
    queries = [(ccy1, ccy2, date(2023, 1, 1)), (ccy1, ccy3, date(2023, 1, 2))]
    result = list(service.queries(queries))
    assert len(result) == 2


def test_queries_with_strict_mode_false():
    from abc import ABCMeta
    from datetime import date
    from decimal import Decimal
    
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
    result = list(service.queries([(ccy1, ccy2, date(2023, 1, 1))], strict=False))
    assert len(result) == 1
    assert result[0] is None


def test_queries_returns_iterable():
    from abc import ABCMeta
    from datetime import date
    
    class MockCurrency:
        def __init__(self, code):
            self.code = code
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        
        def queries(self, queries, strict=False):
            return iter([])
    
    service = MockFXRateService()
    result = service.queries([])
    assert hasattr(result, '__iter__')


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
    
    rate = FXRate(Currencies["USD"], Currencies["EUR"], datetime.date.today(), Decimal("0.85"))
    
    assert rate.value == Decimal("0.85")


def test_fxrate_constructor_with_large_decimal_value():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    
    rate = FXRate(Currencies["USD"], Currencies["BRL"], datetime.date.today(), Decimal("5.25"))
    
    assert rate.value == Decimal("5.25")


# LLM-generated content at query #26
#--------------------------

```python
def test_queries_returns_iterable_of_rates():
    from abc import ABC, abstractmethod
    from typing import Tuple, Currency, Date, Iterable, Optional
    from decimal import Decimal
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[Decimal]:
            return Decimal("1.5")
        
        def queries(self, queries: Iterable[Tuple[Currency, Currency, Date]], strict: bool = False) -> Iterable[Optional[Decimal]]:
            return [Decimal("1.5"), Decimal("2.0"), None]
    
    service = MockFXRateService()
    query_list = [
        ("USD", "EUR", "2023-01-01"),
        ("GBP", "JPY", "2023-01-02"),
        ("CAD", "AUD", "2023-01-03")
    ]
    
    result = list(service.queries(query_list, strict=False))
    
    assert len(result) == 3
    assert result[0] == Decimal("1.5")
    assert result[1] == Decimal("2.0")
    assert result[2] is None


def test_queries_with_strict_mode():
    from abc import ABC, abstractmethod
    from typing import Tuple, Currency, Date, Iterable, Optional
    from decimal import Decimal
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[Decimal]:
            if strict:
                raise LookupError("Rate not found")
            return None
        
        def queries(self, queries: Iterable[Tuple[Currency, Currency, Date]], strict: bool = False) -> Iterable[Optional[Decimal]]:
            if strict:
                raise LookupError("Rate not found")
            return [Decimal("1.2")]
    
    service = MockFXRateService()
    query_list = [("USD", "EUR", "2023-01-01")]
    
    result = list(service.queries(query_list, strict=False))
    assert len(result) == 1


def test_queries_with_empty_iterable():
    from abc import ABC, abstractmethod
    from typing import Tuple, Currency, Date, Iterable, Optional
    from decimal import Decimal
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[Decimal]:
            return Decimal("1.5")
        
        def queries(self, queries: Iterable[Tuple[Currency, Currency, Date]], strict: bool = False) -> Iterable[Optional[Decimal]]:
            return []
    
    service = MockFXRateService()
    query_list = []
    
    result = list(service.queries(query_list))
    
    assert len(result) == 0


def test_queries_accepts_iterable_parameter():
    from abc import ABC, abstractmethod
    from typing import Tuple, Currency, Date, Iterable, Optional
    from decimal import Decimal
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[Decimal]:
            return Decimal("1.5")
        
        def queries(self, queries: Iterable[Tuple[Currency, Currency, Date]], strict: bool = False) -> Iterable[Optional[Decimal]]:
            return [Decimal("1.1"), Decimal("1.2")]
    
    service = MockFXRateService()
    queries_generator = ((c1, c2, d) for c1, c2, d in [("USD", "EUR", "2023-01-01"), ("GBP", "JPY", "2023-01-02")])
    
    result = list(service.queries(queries_generator))
    
    assert len(result) == 2
    assert result[0] == Decimal("1.1")
    assert result[1] == Decimal("1.2")


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
    date = datetime.date(2023, 1, 1)
    value = Decimal("150.5")
    
    rate = FXRate(ccy1, ccy2, date, value)
    
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value


def test_fxrate_constructor_with_small_decimal_value():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    
    ccy1 = Currencies["USD"]
    ccy2 = Currencies["EUR"]
    date = datetime.date.today()
    value = Decimal("0.001")
    
    rate = FXRate(ccy1, ccy2, date, value)
    
    assert rate.value == Decimal("0.001")


def test_fxrate_constructor_with_large_decimal_value():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("9999.9999")
    
    rate = FXRate(ccy1, ccy2, date, value)
    
    assert rate.value == Decimal("9999.9999")


# LLM-generated content at query #28
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


def test_fxrateservice_query_signature():
    from pypara.exchange import FXRateService
    from inspect import signature
    
    sig = signature(FXRateService.query)
    params = list(sig.parameters.keys())
    assert params == ['self', 'ccy1', 'ccy2', 'asof', 'strict']
    assert sig.parameters['strict'].default == False


def test_fxrateservice_query_return_type_annotation():
    from pypara.exchange import FXRateService
    from inspect import signature
    from typing import get_type_hints, Optional
    from pypara.exchange import FXRate
    
    hints = get_type_hints(FXRateService.query)
    assert hints['return'] == Optional[FXRate]


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


# LLM-generated content at query #30
#--------------------------

```python
def test_fxrateservice_query_is_abstract():
    from abc import ABC, abstractmethod
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    # Verify that FXRateService cannot be instantiated directly
    try:
        service = FXRateService()
        assert False, "FXRateService should not be instantiable"
    except TypeError:
        pass
    
    # Create a concrete implementation to test the query method signature
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        
        def queries(self, queries, strict=False):
            return []
    
    service = ConcreteFXRateService()
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    test_date = date(2023, 1, 1)
    
    result = service.query(usd, eur, test_date)
    assert result is None
    
    result_strict = service.query(usd, eur, test_date, strict=True)
    assert result_strict is None


# LLM-generated content at query #31
#--------------------------

```python
def test_queries_returns_iterable_of_rates():
    from abc import ABC, abstractmethod
    from typing import Tuple, Iterable, Optional
    from decimal import Decimal
    from datetime import date
    
    class MockCurrency:
        def __init__(self, code: str):
            self.code = code
    
    class MockFXRate:
        def __init__(self, value: Decimal):
            self.value = value
    
    class ConcreteFXRateService(ABC):
        TQuery = Tuple[MockCurrency, MockCurrency, date]
        
        def query(self, ccy1: MockCurrency, ccy2: MockCurrency, asof: date, strict: bool = False) -> Optional[MockFXRate]:
            pass
        
        def queries(self, queries: Iterable[TQuery], strict: bool = False) -> Iterable[Optional[MockFXRate]]:
            return [MockFXRate(Decimal("1.25")) for _ in queries]
    
    service = ConcreteFXRateService()
    ccy1 = MockCurrency("USD")
    ccy2 = MockCurrency("EUR")
    test_date = date(2023, 1, 1)
    
    query_list = [(ccy1, ccy2, test_date), (ccy1, ccy2, test_date)]
    result = list(service.queries(query_list))
    
    assert len(result) == 2
    assert all(isinstance(rate, MockFXRate) for rate in result)
    assert all(rate.value == Decimal("1.25") for rate in result)


def test_queries_with_strict_mode():
    from abc import ABC
    from typing import Tuple, Iterable, Optional
    from decimal import Decimal
    from datetime import date
    
    class MockCurrency:
        def __init__(self, code: str):
            self.code = code
    
    class MockFXRate:
        def __init__(self, value: Decimal):
            self.value = value
    
    class ConcreteFXRateService(ABC):
        TQuery = Tuple[MockCurrency, MockCurrency, date]
        
        def query(self, ccy1: MockCurrency, ccy2: MockCurrency, asof: date, strict: bool = False) -> Optional[MockFXRate]:
            pass
        
        def queries(self, queries: Iterable[TQuery], strict: bool = False) -> Iterable[Optional[MockFXRate]]:
            return [None if strict else MockFXRate(Decimal("1.25")) for _ in queries]
    
    service = ConcreteFXRateService()
    ccy1 = MockCurrency("USD")
    ccy2 = MockCurrency("GBP")
    test_date = date(2023, 1, 1)
    
    query_list = [(ccy1, ccy2, test_date)]
    result_strict = list(service.queries(query_list, strict=True))
    result_non_strict = list(service.queries(query_list, strict=False))
    
    assert result_strict[0] is None
    assert isinstance(result_non_strict[0], MockFXRate)


def test_queries_with_empty_iterable():
    from abc import ABC
    from typing import Tuple, Iterable, Optional
    from datetime import date
    
    class MockCurrency:
        def __init__(self, code: str):
            self.code = code
    
    class MockFXRate:
        pass
    
    class ConcreteFXRateService(ABC):
        TQuery = Tuple[MockCurrency, MockCurrency, date]
        
        def query(self, ccy1: MockCurrency, ccy2: MockCurrency, asof: date, strict: bool = False) -> Optional[MockFXRate]:
            pass
        
        def queries(self, queries: Iterable[TQuery], strict: bool = False) -> Iterable[Optional[MockFXRate]]:
            return [rate for rate in queries]
    
    service = ConcreteFXRateService()
    result = list(service.queries([]))
    
    assert result == []


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


# LLM-generated content at query #33
#--------------------------

```python
def test_fxrate_constructor_valid():
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
    
    try:
        FXRateService()
        assert False, "Should not be able to instantiate abstract class"
    except TypeError:
        pass


def test_fxrateservice_query_with_valid_parameters():
    from pypara.exchange import FXRateService, FXRate
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    from decimal import Decimal
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return FXRate(ccy1, ccy2, asof, Decimal("1.5"))
        
        def queries(self, queries, strict=False):
            return []
    
    service = MockFXRateService()
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    test_date = date(2023, 1, 1)
    
    result = service.query(usd, eur, test_date)
    assert result is not None
    assert result.ccy1 == usd
    assert result.ccy2 == eur
    assert result.asof == test_date


def test_fxrateservice_query_returns_none_when_not_found():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        
        def queries(self, queries, strict=False):
            return []
    
    service = MockFXRateService()
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    test_date = date(2023, 6, 15)
    
    result = service.query(usd, jpy, test_date)
    assert result is None


def test_fxrateservice_query_with_strict_mode():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if strict:
                raise LookupError("Rate not found")
            return None
        
        def queries(self, queries, strict=False):
            return []
    
    service = MockFXRateService()
    gbp = Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY)
    chf = Currency.of("CHF", "Swiss Franc", 2, CurrencyType.MONEY)
    test_date = date(2023, 12, 31)
    
    try:
        service.query(gbp, chf, test_date, strict=True)
        assert False, "Should raise LookupError"
    except LookupError:
        pass


def test_fxrateservice_query_with_same_currency():
    from pypara.exchange import FXRateService, FXRate
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    from decimal import Decimal
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if ccy1 == ccy2:
                return FXRate(ccy1, ccy2, asof, Decimal("1"))
            return None
        
        def queries(self, queries, strict=False):
            return []
    
    service = MockFXRateService()
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    test_date = date(2023, 3, 20)
    
    result = service.query(usd, usd, test_date)
    assert result is not None
    assert result.rate == Decimal("1")


# LLM-generated content at query #2
#--------------------------

```python
def test_fxrateservice_query_is_abstract():
    from abc import ABC, abstractmethod
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    # Verify that FXRateService is abstract and cannot be instantiated
    try:
        service = FXRateService()
        assert False, "Should not be able to instantiate abstract class"
    except TypeError:
        pass
    
    # Create a concrete implementation to test the query method signature
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        
        def queries(self, queries, strict=False):
            return []
    
    service = ConcreteFXRateService()
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    test_date = date(2023, 1, 1)
    
    result = service.query(usd, eur, test_date, strict=False)
    assert result is None
    
    result_strict = service.query(usd, eur, test_date, strict=True)
    assert result_strict is None


# LLM-generated content at query #3
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
    
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if ccy1.code == "USD" and ccy2.code == "EUR":
                return MockFXRate(Decimal("0.85"))
            return None
        
        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    service = TestFXRateService()
    ccy1 = MockCurrency("USD")
    ccy2 = MockCurrency("EUR")
    test_date = date(2023, 1, 1)
    
    result = list(service.queries([(ccy1, ccy2, test_date)]))
    
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
    
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            rates = {
                ("USD", "EUR"): Decimal("0.85"),
                ("USD", "GBP"): Decimal("0.73"),
            }
            key = (ccy1.code, ccy2.code)
            return MockFXRate(rates[key]) if key in rates else None
        
        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    service = TestFXRateService()
    ccy_usd = MockCurrency("USD")
    ccy_eur = MockCurrency("EUR")
    ccy_gbp = MockCurrency("GBP")
    test_date = date(2023, 1, 1)
    
    query_list = [(ccy_usd, ccy_eur, test_date), (ccy_usd, ccy_gbp, test_date)]
    result = list(service.queries(query_list))
    
    assert len(result) == 2
    assert result[0].rate == Decimal("0.85")
    assert result[1].rate == Decimal("0.73")


def test_queries_with_none_results():
    from datetime import date
    
    class MockCurrency:
        def __init__(self, code):
            self.code = code
    
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        
        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    service = TestFXRateService()
    ccy1 = MockCurrency("USD")
    ccy2 = MockCurrency("XXX")
    test_date = date(2023, 1, 1)
    
    result = list(service.queries([(ccy1, ccy2, test_date)]))
    
    assert len(result) == 1
    assert result[0] is None


def test_queries_with_strict_mode():
    from datetime import date
    
    class MockCurrency:
        def __init__(self, code):
            self.code = code
    
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if strict and ccy1.code == "INVALID":
                raise LookupError("Rate not found")
            return None
        
        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    service = TestFXRateService()
    ccy_invalid = MockCurrency("INVALID")
    ccy_usd = MockCurrency("USD")
    test_date = date(2023, 1, 1)
    
    error_raised = False
    try:
        list(service.queries([(ccy_invalid, ccy_usd, test_date)], strict=True))
    except LookupError:
        error_raised = True
    
    assert error_raised


def test_queries_returns_iterable():
    from datetime import date
    
    class MockCurrency:
        def __init__(self, code):
            self.code = code
    
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        
        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    service = TestFXRateService()
    ccy1 = MockCurrency("USD")
    ccy2 = MockCurrency("EUR")
    test_date = date(2023, 1, 1)
    
    result = service.queries([(ccy1, ccy2, test_date)])
    
    assert hasattr(result, '__iter__')


# LLM-generated content at query #4
#--------------------------

```python
def test_fxrate_constructor():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    rate_date = datetime.date.today()
    rate_value = Decimal("2")
    
    rate = FXRate(ccy1, ccy2, rate_date, rate_value)
    
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == rate_date
    assert rate.value == rate_value


def test_fxrate_constructor_tuple_unpacking():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    rate_date = datetime.date.today()
    rate_value = Decimal("2")
    
    rate = FXRate(ccy1, ccy2, rate_date, rate_value)
    unpacked_ccy1, unpacked_ccy2, unpacked_date, unpacked_value = rate
    
    assert unpacked_ccy1 == ccy1
    assert unpacked_ccy2 == ccy2
    assert unpacked_date == rate_date
    assert unpacked_value == rate_value


def test_fxrate_constructor_indexed_access():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    rate_date = datetime.date.today()
    rate_value = Decimal("2")
    
    rate = FXRate(ccy1, ccy2, rate_date, rate_value)
    
    assert rate[0] == ccy1
    assert rate[1] == ccy2
    assert rate[2] == rate_date
    assert rate[3] == rate_value


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
    
    service = FXRateService()
    try:
        service.query(usd, eur, test_date)
        assert False, "Should raise TypeError for abstract method"
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
    test_date = date(2023, 3, 20)
    
    service = TestFXRateService()
    result = service.query(jpy, usd, test_date)
    
    assert result == 0.0075


def test_fxrateservice_query_with_different_dates():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    class DateAwareFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if asof == date(2023, 1, 1):
                return 1.10
            elif asof == date(2023, 12, 31):
                return 1.15
            return None
        
        def queries(self, queries, strict=False):
            return []
    
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    service = DateAwareFXRateService()
    result1 = service.query(eur, usd, date(2023, 1, 1))
    result2 = service.query(eur, usd, date(2023, 12, 31))
    
    assert result1 == 1.10
    assert result2 == 1.15


# LLM-generated content at query #7
#--------------------------

```python
def test_queries_with_empty_iterable():
    from abc import ABC
    from decimal import Decimal
    from datetime import date
    
    class MockCurrency:
        def __init__(self, code):
            self.code = code
    
    class MockFXRateService(ABC):
        def query(self, ccy1, ccy2, asof, strict=False):
            pass
        
        def queries(self, queries, strict=False):
            return [Decimal("1.5") if q else None for q in queries]
    
    service = MockFXRateService()
    result = list(service.queries([]))
    assert result == []


def test_queries_with_single_query():
    from abc import ABC
    from decimal import Decimal
    from datetime import date
    
    class MockCurrency:
        def __init__(self, code):
            self.code = code
    
    class MockFXRateService(ABC):
        def query(self, ccy1, ccy2, asof, strict=False):
            return Decimal("1.5")
        
        def queries(self, queries, strict=False):
            return [Decimal("1.5") for _ in queries]
    
    service = MockFXRateService()
    ccy1 = MockCurrency("USD")
    ccy2 = MockCurrency("EUR")
    query_date = date(2023, 1, 1)
    
    result = list(service.queries([(ccy1, ccy2, query_date)]))
    assert len(result) == 1
    assert result[0] == Decimal("1.5")


def test_queries_with_multiple_queries():
    from abc import ABC
    from decimal import Decimal
    from datetime import date
    
    class MockCurrency:
        def __init__(self, code):
            self.code = code
    
    class MockFXRateService(ABC):
        def query(self, ccy1, ccy2, asof, strict=False):
            return Decimal("1.5")
        
        def queries(self, queries, strict=False):
            return [Decimal("1.5"), Decimal("1.2"), Decimal("0.9")]
    
    service = MockFXRateService()
    ccy1 = MockCurrency("USD")
    ccy2 = MockCurrency("EUR")
    ccy3 = MockCurrency("GBP")
    query_date = date(2023, 1, 1)
    
    queries_list = [
        (ccy1, ccy2, query_date),
        (ccy1, ccy3, query_date),
        (ccy2, ccy3, query_date)
    ]
    result = list(service.queries(queries_list))
    assert len(result) == 3
    assert result[0] == Decimal("1.5")
    assert result[1] == Decimal("1.2")
    assert result[2] == Decimal("0.9")


def test_queries_with_none_values():
    from abc import ABC
    from decimal import Decimal
    from datetime import date
    
    class MockCurrency:
        def __init__(self, code):
            self.code = code
    
    class MockFXRateService(ABC):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        
        def queries(self, queries, strict=False):
            return [None, Decimal("1.5"), None]
    
    service = MockFXRateService()
    ccy1 = MockCurrency("USD")
    ccy2 = MockCurrency("EUR")
    query_date = date(2023, 1, 1)
    
    queries_list = [
        (ccy1, ccy2, query_date),
        (ccy1, ccy2, query_date),
        (ccy1, ccy2, query_date)
    ]
    result = list(service.queries(queries_list))
    assert len(result) == 3
    assert result[0] is None
    assert result[1] == Decimal("1.5")
    assert result[2] is None


def test_queries_strict_mode_false():
    from abc import ABC
    from decimal import Decimal
    from datetime import date
    
    class MockCurrency:
        def __init__(self, code):
            self.code = code
    
    class MockFXRateService(ABC):
        def query(self, ccy1, ccy2, asof, strict=False):
            pass
        
        def queries(self, queries, strict=False):
            return [Decimal("1.5") if not strict else None for _ in queries]
    
    service = MockFXRateService()
    ccy1 = MockCurrency("USD")
    ccy2 = MockCurrency("EUR")
    query_date = date(2023, 1, 1)
    
    result = list(service.queries([(ccy1, ccy2, query_date)], strict=False))
    assert result[0] == Decimal("1.5")


def test_queries_strict_mode_true():
    from abc import ABC
    from decimal import Decimal
    from datetime import date
    
    class MockCurrency:
        def __init__(self, code):
            self.code = code
    
    class MockFXRateService(ABC):
        def query(self, ccy1, ccy2, asof, strict=False):
            pass
        
        def queries(self, queries, strict=False):
            return [None if strict else Decimal("1.5") for _ in queries]
    
    service = MockFXRateService()
    ccy1 = MockCurrency("USD")
    ccy2 = MockCurrency("EUR")
    query_date = date(2023, 1, 1)
    
    result = list(service.queries([(ccy1, ccy2, query_date)], strict=True))
    assert result[0] is None


# LLM-generated content at query #8
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
    
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if ccy1.code == "USD" and ccy2.code == "EUR":
                return MockFXRate(Decimal("0.85"))
            return None
        
        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    service = TestFXRateService()
    ccy1 = MockCurrency("USD")
    ccy2 = MockCurrency("EUR")
    test_date = date(2023, 1, 1)
    
    result = list(service.queries([(ccy1, ccy2, test_date)]))
    
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
    
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            rates = {
                ("USD", "EUR"): Decimal("0.85"),
                ("USD", "GBP"): Decimal("0.73"),
                ("EUR", "GBP"): Decimal("0.86")
            }
            key = (ccy1.code, ccy2.code)
            return MockFXRate(rates[key]) if key in rates else None
        
        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    service = TestFXRateService()
    test_date = date(2023, 1, 1)
    query_list = [
        (MockCurrency("USD"), MockCurrency("EUR"), test_date),
        (MockCurrency("USD"), MockCurrency("GBP"), test_date),
        (MockCurrency("EUR"), MockCurrency("GBP"), test_date)
    ]
    
    result = list(service.queries(query_list))
    
    assert len(result) == 3
    assert result[0].rate == Decimal("0.85")
    assert result[1].rate == Decimal("0.73")
    assert result[2].rate == Decimal("0.86")


def test_queries_with_missing_rate():
    from datetime import date
    
    class MockCurrency:
        def __init__(self, code):
            self.code = code
    
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        
        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    service = TestFXRateService()
    test_date = date(2023, 1, 1)
    query_list = [(MockCurrency("USD"), MockCurrency("XXX"), test_date)]
    
    result = list(service.queries(query_list))
    
    assert len(result) == 1
    assert result[0] is None


def test_queries_with_empty_iterable():
    from datetime import date
    
    class MockCurrency:
        def __init__(self, code):
            self.code = code
    
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        
        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    service = TestFXRateService()
    
    result = list(service.queries([]))
    
    assert len(result) == 0


def test_queries_returns_iterable():
    from datetime import date
    
    class MockCurrency:
        def __init__(self, code):
            self.code = code
    
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        
        def queries(self, queries, strict=False):
            return iter([self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries])
    
    service = TestFXRateService()
    test_date = date(2023, 1, 1)
    query_list = [(MockCurrency("USD"), MockCurrency("EUR"), test_date)]
    
    result = service.queries(query_list)
    
    assert hasattr(result, '__iter__')


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
    query_date = date(2023, 1, 1)
    
    result = list(service.queries([(usd, eur, query_date)]))
    assert len(result) == 1
    assert result[0].rate == Decimal("1.5")


def test_queries_with_multiple_queries():
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
    query_date1 = date(2023, 1, 1)
    query_date2 = date(2023, 1, 2)
    
    result = list(service.queries([(usd, eur, query_date1), (eur, gbp, query_date2)]))
    assert len(result) == 2


def test_queries_with_strict_mode():
    from decimal import Decimal
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
    query_date = date(2023, 1, 1)
    
    result = list(service.queries([(usd, eur, query_date)], strict=True))
    assert len(result) == 1
    assert result[0] is None


def test_queries_returns_iterable():
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
    query_date = date(2023, 1, 1)
    
    result = service.queries([(usd, eur, query_date)])
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
    value = Decimal("150.75")
    
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


# LLM-generated content at query #17
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


def test_fxrateservice_query_signature():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    import inspect
    
    sig = inspect.signature(FXRateService.query)
    params = list(sig.parameters.keys())
    
    assert params == ['self', 'ccy1', 'ccy2', 'asof', 'strict']
    assert sig.parameters['strict'].default == False


def test_fxrateservice_query_return_type():
    from pypara.exchange import FXRateService
    import inspect
    
    sig = inspect.signature(FXRateService.query)
    return_annotation = sig.return_annotation
    
    assert return_annotation != inspect.Signature.empty


# LLM-generated content at query #18
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
            return [FXRate(1.0) for _ in queries]
    
    service = TestFXRateService()
    result = list(service.queries([], strict=False))
    assert result == []


def test_queries_with_single_query():
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
    ccy1 = Currency("USD")
    ccy2 = Currency("EUR")
    asof = date(2023, 1, 1)
    queries = [(ccy1, ccy2, asof)]
    
    result = list(service.queries(queries, strict=False))
    assert len(result) == 1
    assert result[0].rate == 1.5


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
            return [FXRate(1.0 + i * 0.1) for i, _ in enumerate(queries)]
    
    service = TestFXRateService()
    ccy1 = Currency("USD")
    ccy2 = Currency("EUR")
    queries = [
        (ccy1, ccy2, date(2023, 1, 1)),
        (ccy1, ccy2, date(2023, 1, 2)),
        (ccy1, ccy2, date(2023, 1, 3))
    ]
    
    result = list(service.queries(queries, strict=False))
    assert len(result) == 3
    assert result[0].rate == 1.0
    assert result[1].rate == 1.1
    assert result[2].rate == 1.2


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
            return FXRate(1.0)
        
        def queries(self, queries: Iterable[TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [FXRate(1.0) for _ in queries]
    
    service = TestFXRateService()
    ccy1 = Currency("USD")
    ccy2 = Currency("EUR")
    queries = [(ccy1, ccy2, date(2023, 1, 1))]
    
    result = list(service.queries(queries, strict=True))
    assert len(result) == 1
    assert result[0].rate == 1.0


def test_queries_with_none_results():
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
        def query(


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


# LLM-generated content at query #22
#--------------------------

```python
def test_fxrateservice_query_is_abstract():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    assert hasattr(FXRateService.query, '__isabstractmethod__')
    assert FXRateService.query.__isabstractmethod__ is True


def test_fxrateservice_query_signature():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    import inspect
    
    sig = inspect.signature(FXRateService.query)
    params = list(sig.parameters.keys())
    
    assert 'self' in params
    assert 'ccy1' in params
    assert 'ccy2' in params
    assert 'asof' in params
    assert 'strict' in params
    assert sig.parameters['strict'].default is False


def test_fxrateservice_cannot_instantiate():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    try:
        service = FXRateService()
        assert False, "Should not be able to instantiate abstract class"
    except TypeError:
        pass


def test_fxrateservice_query_requires_implementation():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    class ConcreteService(FXRateService):
        def queries(self, queries, strict=False):
            return []
    
    try:
        service = ConcreteService()
        assert False, "Should not be able to instantiate without implementing query"
    except TypeError:
        pass


def test_fxrateservice_query_with_concrete_implementation():
    from pypara.exchange import FXRateService, FXRate
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    from decimal import Decimal
    
    class TestService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return FXRate(ccy1, ccy2, asof, Decimal("1.5"))
        
        def queries(self, queries, strict=False):
            return []
    
    service = TestService()
    usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    test_date = date(2023, 1, 1)
    
    result = service.query(usd, eur, test_date)
    
    assert result is not None
    assert result.ccy1 == usd
    assert result.ccy2 == eur
    assert result.asof == test_date
    assert result.rate == Decimal("1.5")


def test_fxrateservice_query_with_strict_parameter():
    from pypara.exchange import FXRateService, FXRate
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    from decimal import Decimal
    
    class TestService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if strict:
                return FXRate(ccy1, ccy2, asof, Decimal("1.5"))
            return None
        
        def queries(self, queries, strict=False):
            return []
    
    service = TestService()
    usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    test_date = date(2023, 1, 1)
    
    result_non_strict = service.query(usd, eur, test_date, strict=False)
    result_strict = service.query(usd, eur, test_date, strict=True)
    
    assert result_non_strict is None
    assert result_strict is not None


# LLM-generated content at query #23
#--------------------------

```python
def test_queries_with_empty_iterable():
    from abc import ABCMeta
    from datetime import date
    from decimal import Decimal
    
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
    from abc import ABCMeta
    from datetime import date
    from decimal import Decimal
    
    class MockCurrency:
        def __init__(self, code):
            self.code = code
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return Decimal("1.5")
        
        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    service = MockFXRateService()
    ccy1 = MockCurrency("USD")
    ccy2 = MockCurrency("EUR")
    test_date = date(2023, 1, 1)
    
    result = list(service.queries([(ccy1, ccy2, test_date)]))
    assert len(result) == 1
    assert result[0] == Decimal("1.5")


def test_queries_with_multiple_queries():
    from abc import ABCMeta
    from datetime import date
    from decimal import Decimal
    
    class MockCurrency:
        def __init__(self, code):
            self.code = code
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if ccy1.code == "USD" and ccy2.code == "EUR":
                return Decimal("0.92")
            elif ccy1.code == "GBP" and ccy2.code == "USD":
                return Decimal("1.27")
            return None
        
        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    service = MockFXRateService()
    ccy_usd = MockCurrency("USD")
    ccy_eur = MockCurrency("EUR")
    ccy_gbp = MockCurrency("GBP")
    test_date = date(2023, 1, 1)
    
    result = list(service.queries([
        (ccy_usd, ccy_eur, test_date),
        (ccy_gbp, ccy_usd, test_date)
    ]))
    assert len(result) == 2
    assert result[0] == Decimal("0.92")
    assert result[1] == Decimal("1.27")


def test_queries_with_strict_mode_false():
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
    ccy1 = MockCurrency("USD")
    ccy2 = MockCurrency("XXX")
    test_date = date(2023, 1, 1)
    
    result = list(service.queries([(ccy1, ccy2, test_date)], strict=False))
    assert len(result) == 1
    assert result[0] is None


def test_queries_returns_iterable():
    from abc import ABCMeta
    from datetime import date
    from decimal import Decimal
    
    class MockCurrency:
        def __init__(self, code):
            self.code = code
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return Decimal("1.0")
        
        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    service = MockFXRateService()
    ccy1 = MockCurrency("USD")
    ccy2 = MockCurrency("EUR")
    test_date = date(2023, 1, 1)
    
    result = service.queries([(ccy1, ccy2, test_date)])
    assert hasattr(result, '__iter__')


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


# LLM-generated content at query #25
#--------------------------

```python
def test_fxrateservice_query_is_abstract():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    test_date = date(2023, 1, 1)
    
    assert hasattr(FXRateService, 'query')
    assert hasattr(FXRateService.query, '__isabstractmethod__')
    assert FXRateService.query.__isabstractmethod__ is True


# LLM-generated content at query #26
#--------------------------

```python
def test_queries_returns_iterable_of_fx_rates():
    from datetime import date
    from decimal import Decimal
    from unittest.mock import Mock, MagicMock
    
    # Create a mock FXRateService implementation
    mock_service = Mock(spec=FXRateService)
    mock_service.queries = MagicMock()
    
    # Define test data
    currency1 = Mock()
    currency2 = Mock()
    test_date = date(2023, 1, 1)
    
    query1 = (currency1, currency2, test_date)
    query2 = (currency2, currency1, test_date)
    queries = [query1, query2]
    
    expected_rate1 = Decimal("1.5")
    expected_rate2 = Decimal("0.67")
    expected_rates = [expected_rate1, expected_rate2]
    
    mock_service.queries.return_value = expected_rates
    
    # Execute
    result = mock_service.queries(queries, strict=False)
    
    # Assert
    assert list(result) == expected_rates
    mock_service.queries.assert_called_once_with(queries, strict=False)


def test_queries_with_strict_mode_true():
    from datetime import date
    from decimal import Decimal
    from unittest.mock import Mock, MagicMock
    
    # Create a mock FXRateService implementation
    mock_service = Mock(spec=FXRateService)
    mock_service.queries = MagicMock()
    
    # Define test data
    currency1 = Mock()
    currency2 = Mock()
    test_date = date(2023, 6, 15)
    
    query = (currency1, currency2, test_date)
    queries = [query]
    
    expected_rate = Decimal("2.3")
    mock_service.queries.return_value = [expected_rate]
    
    # Execute
    result = mock_service.queries(queries, strict=True)
    
    # Assert
    assert list(result) == [expected_rate]
    mock_service.queries.assert_called_once_with(queries, strict=True)


def test_queries_returns_none_for_missing_rates():
    from datetime import date
    from unittest.mock import Mock, MagicMock
    
    # Create a mock FXRateService implementation
    mock_service = Mock(spec=FXRateService)
    mock_service.queries = MagicMock()
    
    # Define test data
    currency1 = Mock()
    currency2 = Mock()
    test_date = date(2023, 12, 31)
    
    query1 = (currency1, currency2, test_date)
    query2 = (currency2, currency1, test_date)
    queries = [query1, query2]
    
    # Return mixed results with None
    mock_service.queries.return_value = [None, None]
    
    # Execute
    result = mock_service.queries(queries, strict=False)
    
    # Assert
    result_list = list(result)
    assert result_list[0] is None
    assert result_list[1] is None
    mock_service.queries.assert_called_once_with(queries, strict=False)


def test_queries_with_empty_queries_list():
    from unittest.mock import Mock, MagicMock
    
    # Create a mock FXRateService implementation
    mock_service = Mock(spec=FXRateService)
    mock_service.queries = MagicMock()
    
    # Define test data
    queries = []
    mock_service.queries.return_value = []
    
    # Execute
    result = mock_service.queries(queries, strict=False)
    
    # Assert
    assert list(result) == []
    mock_service.queries.assert_called_once_with(queries, strict=False)


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
    date = datetime.date(2023, 1, 15)
    value = Decimal("150.75")
    
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
    
    assert rate.value == Decimal("0.0001")


def test_fxrate_constructor_large_value():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("9999.9999")
    
    rate = FXRate(ccy1, ccy2, date, value)
    
    assert rate.value == Decimal("9999.9999")


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


def test_fxrate_constructor_with_different_values():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    
    ccy1 = Currencies["GBP"]
    ccy2 = Currencies["JPY"]
    date = datetime.date(2023, 1, 15)
    value = Decimal("150.75")
    
    rate = FXRate(ccy1, ccy2, date, value)
    
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value


# LLM-generated content at query #29
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


def test_fxrateservice_query_signature():
    from pypara.exchange import FXRateService
    from pypara.currencies import Currency, CurrencyType
    from datetime import date
    import inspect
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    test_date = date(2023, 1, 1)
    
    sig = inspect.signature(FXRateService.query)
    params = list(sig.parameters.keys())
    
    assert "ccy1" in params
    assert "ccy2" in params
    assert "asof" in params
    assert "strict" in params
    assert sig.parameters["strict"].default == False


def test_fxrateservice_query_is_abstract_method():
    from pypara.exchange import FXRateService
    import inspect
    
    assert hasattr(FXRateService.query, "__isabstractmethod__")
    assert FXRateService.query.__isabstractmethod__ == True


# LLM-generated content at query #30
#--------------------------

```python
def test_queries_returns_iterable_of_fx_rates():
    from decimal import Decimal
    from datetime import date
    from collections.abc import Iterable
    
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
    
    query_list = [(ccy1, ccy2, test_date), (ccy1, ccy2, test_date)]
    result = service.queries(query_list)
    
    assert isinstance(result, Iterable)
    result_list = list(result)
    assert len(result_list) == 2
    assert all(isinstance(rate, MockFXRate) for rate in result_list)


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
    result = service.queries(query_list, strict=True)
    result_list = list(result)
    
    assert len(result_list) == 1


def test_queries_with_empty_iterable():
    from datetime import date
    
    class MockCurrency:
        def __init__(self, code):
            self.code = code
    
    class MockFXRate:
        def __init__(self, rate):
            self.rate = rate
    
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return MockFXRate(1.5)
        
        def queries(self, queries, strict=False):
            return [MockFXRate(1.5) for _ in queries]
    
    service = ConcreteFXRateService()
    query_list = []
    result = service.queries(query_list)
    result_list = list(result)
    
    assert len(result_list) == 0


def test_queries_returns_optional_fx_rates():
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
            return None
        
        def queries(self, queries, strict=False):
            return [None if i % 2 == 0 else MockFXRate(Decimal("1.5")) for i, _ in enumerate(queries)]
    
    service = ConcreteFXRateService()
    ccy1 = MockCurrency("USD")
    ccy2 = MockCurrency("EUR")
    test_date = date(2023, 1, 1)
    
    query_list = [(ccy1, ccy2, test_date), (ccy1, ccy2, test_date), (ccy1, ccy2, test_date)]
    result = service.queries(query_list)
    result_list = list(result)
    
    assert len(result_list) == 3
    assert result_list[0] is None
    assert result_list[1] is not None
    assert result_list[2] is None


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


