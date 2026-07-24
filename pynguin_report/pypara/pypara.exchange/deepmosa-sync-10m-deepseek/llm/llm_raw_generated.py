####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_invert_fx_rate():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    nrate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("2"))
    rrate = FXRate(Currencies["USD"], Currencies["EUR"], datetime.date.today(), Decimal("0.5"))
    assert ~nrate == rrate


# LLM-generated content at query #2
#--------------------------

```
def test_invert_fx_rate():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    nrate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("2"))
    rrate = FXRate(Currencies["USD"], Currencies["EUR"], datetime.date.today(), Decimal("0.5"))
    assert ~nrate == rrate


# LLM-generated content at query #3
#--------------------------

```python
def test_queries_with_empty_input():
    service = FXRateService()
    result = list(service.queries([]))
    assert result == []

def test_queries_with_strict_mode():
    service = FXRateService()
    queries = [(Currency('USD'), Currency('EUR'), Date('2023-10-01'))]
    with pytest.raises(LookupError):
        list(service.queries(queries, strict=True))

def test_queries_with_non_strict_mode():
    service = FXRateService()
    queries = [(Currency('USD'), Currency('EUR'), Date('2023-10-01'))]
    result = list(service.queries(queries, strict=False))
    assert result == [None]

def test_queries_with_multiple_queries():
    service = FXRateService()
    queries = [
        (Currency('USD'), Currency('EUR'), Date('2023-10-01')),
        (Currency('GBP'), Currency('JPY'), Date('2023-10-02'))
    ]
    result = list(service.queries(queries, strict=False))
    assert result == [None, None]


# LLM-generated content at query #4
#--------------------------

def test_query_returns_fx_rate_for_currency_pair():
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    EUR = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    date = Date(2023, 1, 1)
    expected_rate = FXRate(Decimal("0.85"))
    
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return expected_rate if ccy1 == USD and ccy2 == EUR and asof == date else None
        
        def queries(self, queries, strict=False):
            pass
    
    service = TestFXRateService()
    result = service.query(USD, EUR, date)
    assert result == expected_rate

def test_query_returns_none_when_rate_not_found():
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    EUR = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    date = Date(2023, 1, 1)
    
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        
        def queries(self, queries, strict=False):
            pass
    
    service = TestFXRateService()
    result = service.query(USD, EUR, date)
    assert result is None

def test_query_raises_error_in_strict_mode_when_rate_not_found():
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    EUR = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    date = Date(2023, 1, 1)
    
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if strict:
                raise LookupError("Rate not found")
            return None
        
        def queries(self, queries, strict=False):
            pass
    
    service = TestFXRateService()
    try:
        service.query(USD, EUR, date, strict=True)
        assert False, "Expected LookupError"
    except LookupError:
        pass


# LLM-generated content at query #5
#--------------------------

```python
def test_query_method_valid_input():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    asof = Date(2023, 10, 1)
    service = FXRateService()
    rate = service.query(ccy1, ccy2, asof)
    assert isinstance(rate, FXRate) or rate is None

def test_query_method_strict_mode():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    asof = Date(2023, 10, 1)
    service = FXRateService()
    try:
        service.query(ccy1, ccy2, asof, strict=True)
    except LookupError:
        pass
    else:
        assert False, "Expected LookupError in strict mode"

def test_query_method_invalid_currency():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = "INVALID_CURRENCY"
    asof = Date(2023, 10, 1)
    service = FXRateService()
    try:
        service.query(ccy1, ccy2, asof)
    except TypeError:
        pass
    else:
        assert False, "Expected TypeError for invalid currency"

def test_query_method_none_asof():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    service = FXRateService()
    try:
        service.query(ccy1, ccy2, None)
    except TypeError:
        pass
    else:
        assert False, "Expected TypeError for None asof"


# LLM-generated content at query #6
#--------------------------

```python
def test_query_returns_fx_rate_for_valid_currency_pair():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 10, 1)
    fx_rate = FXRate(ccy1, ccy2, Decimal("0.85"), asof)
    service = FXRateService()
    service.query = lambda c1, c2, a, strict: fx_rate
    result = service.query(ccy1, ccy2, asof)
    assert result == fx_rate

def test_query_returns_none_for_invalid_currency_pair():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    asof = Date(2023, 10, 1)
    service = FXRateService()
    service.query = lambda c1, c2, a, strict: None
    result = service.query(ccy1, ccy2, asof)
    assert result is None

def test_query_raises_error_for_strict_mode_and_invalid_pair():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    asof = Date(2023, 10, 1)
    service = FXRateService()
    service.query = lambda c1, c2, a, strict: None if not strict else LookupError("FX rate not found")
    try:
        service.query(ccy1, ccy2, asof, strict=True)
        assert False, "Expected LookupError"
    except LookupError:
        assert True


# LLM-generated content at query #7
#--------------------------

```python
def test_query_method():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    date = Date(2023, 10, 1)
    fx_rate_service = FXRateService()
    
    fx_rate = fx_rate_service.query(usd, eur, date)
    assert fx_rate is None
    
    fx_rate = fx_rate_service.query(usd, eur, date, strict=True)
    assert fx_rate is None


# LLM-generated content at query #8
#--------------------------

```
def test_queries_with_empty_iterable():
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            return None
        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return super().queries(queries, strict)
    
    mock_service = MockFXRateService()
    result = mock_service.queries([])
    assert list(result) == []

def test_queries_with_multiple_queries():
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            return FXRate(Decimal('1.0'))
        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    mock_service = MockFXRateService()
    queries = [(Currency('USD'), Currency('EUR'), Date(2023, 10, 1)), (Currency('GBP'), Currency('JPY'), Date(2023, 10, 1))]
    result = mock_service.queries(queries)
    assert all(rate == FXRate(Decimal('1.0')) for rate in result)

def test_queries_with_strict_mode_and_missing_rate():
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            if strict:
                raise LookupError("Rate not found")
            return None
        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    mock_service = MockFXRateService()
    queries = [(Currency('USD'), Currency('EUR'), Date(2023, 10, 1))]
    try:
        mock_service.queries(queries, strict=True)
        assert False
    except LookupError:
        assert True


# LLM-generated content at query #9
#--------------------------

```python
def test_queries_returns_empty_iterable_when_no_queries_provided():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None

        def queries(self, queries, strict=False):
            return super().queries(queries, strict)

    service = MockFXRateService()
    result = service.queries([])
    assert list(result) == []

def test_queries_returns_iterable_with_none_when_no_matching_rate_found():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None

        def queries(self, queries, strict=False):
            return super().queries(queries, strict)

    service = MockFXRateService()
    query = (Currency("USD"), Currency("EUR"), Date(2023, 1, 1))
    result = service.queries([query])
    assert list(result) == [None]

def test_queries_returns_iterable_with_fx_rate_when_matching_rate_found():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return FXRate(Decimal("1.2"))

        def queries(self, queries, strict=False):
            return super().queries(queries, strict)

    service = MockFXRateService()
    query = (Currency("USD"), Currency("EUR"), Date(2023, 1, 1))
    result = service.queries([query])
    assert list(result) == [FXRate(Decimal("1.2"))]

def test_queries_raises_error_when_strict_mode_and_no_matching_rate_found():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if strict:
                raise LookupError("Rate not found")
            return None

        def queries(self, queries, strict=False):
            return super().queries(queries, strict)

    service = MockFXRateService()
    query = (Currency("USD"), Currency("EUR"), Date(2023, 1, 1))
    try:
        service.queries([query], strict=True)
        assert False, "Expected LookupError"
    except LookupError:
        pass

def test_queries_handles_multiple_queries_correctly():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if ccy1 == Currency("USD") and ccy2 == Currency("EUR"):
                return FXRate(Decimal("1.2"))
            return None

        def queries(self, queries, strict=False):
            return super().queries(queries, strict)

    service = MockFXRateService()
    query1 = (Currency("USD"), Currency("EUR"), Date(2023, 1, 1))
    query2 = (Currency("GBP"), Currency("JPY"), Date(2023, 1, 1))
    result = service.queries([query1, query2])
    assert list(result) == [FXRate(Decimal("1.2")), None]


# LLM-generated content at query #10
#--------------------------

```
def test_queries_returns_empty_iterable_when_no_queries():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        def queries(self, queries, strict=False):
            return super().queries(queries, strict)
    service = MockFXRateService()
    result = service.queries([])
    assert list(result) == []

def test_queries_returns_rates_for_valid_queries():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return Decimal('1.5') if ccy1 == 'USD' and ccy2 == 'EUR' else None
        def queries(self, queries, strict=False):
            return [self.query(*q) for q in queries]
    service = MockFXRateService()
    queries = [('USD', 'EUR', '2023-01-01'), ('EUR', 'USD', '2023-01-01')]
    result = service.queries(queries)
    assert list(result) == [Decimal('1.5'), None]

def test_queries_raises_error_in_strict_mode_when_rate_not_found():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if strict and ccy1 != 'USD' and ccy2 != 'EUR':
                raise LookupError()
            return None
        def queries(self, queries, strict=False):
            return super().queries(queries, strict)
    service = MockFXRateService()
    queries = [('GBP', 'JPY', '2023-01-01')]
    try:
        list(service.queries(queries, strict=True))
        assert False
    except LookupError:
        assert True

def test_queries_handles_multiple_queries_correctly():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return Decimal('1.2') if ccy1 == 'USD' and ccy2 == 'GBP' else Decimal('0.8') if ccy1 == 'EUR' and ccy2 == 'USD' else None
        def queries(self, queries, strict=False):
            return [self.query(*q) for q in queries]
    service = MockFXRateService()
    queries = [('USD', 'GBP', '2023-01-01'), ('EUR', 'USD', '2023-01-01'), ('JPY', 'EUR', '2023-01-01')]
    result = service.queries(queries)
    assert list(result) == [Decimal('1.2'), Decimal('0.8'), None]


# LLM-generated content at query #11
#--------------------------

```
def test_constructor_creates_valid_fxrate():
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

def test_constructor_allows_tuple_unpacking():
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

def test_constructor_creates_immutable_object():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("2")
    rate = FXRate(ccy1, ccy2, date, value)
    try:
        rate.ccy1 = Currencies["GBP"]
        assert False, "Should not be able to modify ccy1"
    except AttributeError:
        pass
    try:
        rate.ccy2 = Currencies["GBP"]
        assert False, "Should not be able to modify ccy2"
    except AttributeError:
        pass
    try:
        rate.date = datetime.date(2020, 1, 1)
        assert False, "Should not be able to modify date"
    except AttributeError:
        pass
    try:
        rate.value = Decimal("3")
        assert False, "Should not be able to modify value"
    except AttributeError:
        pass


# LLM-generated content at query #12
#--------------------------

```python
def test_fxrate_constructor():
    from datetime import date
    from decimal import Decimal
    from pypara.currencies import Currencies

    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    fx_date = date.today()
    value = Decimal("2")

    fx_rate = FXRate(ccy1, ccy2, fx_date, value)

    assert fx_rate.ccy1 == ccy1
    assert fx_rate.ccy2 == ccy2
    assert fx_rate.date == fx_date
    assert fx_rate.value == value


# LLM-generated content at query #13
#--------------------------

```
def test_FXRate_constructor_with_valid_arguments():
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

def test_FXRate_constructor_with_same_currencies_and_value_one():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["EUR"]
    date = datetime.date.today()
    value = Decimal("1")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value

def test_FXRate_constructor_with_same_currencies_and_value_not_one():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["EUR"]
    date = datetime.date.today()
    value = Decimal("2")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value

def test_FXRate_constructor_with_zero_value():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("0")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value

def test_FXRate_constructor_with_negative_value():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("-1")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value


# LLM-generated content at query #14
#--------------------------

```python
def test_fxrate_constructor_with_valid_input():
    ccy1 = Currency("EUR", 2)
    ccy2 = Currency("USD", 2)
    date = Date(2023, 10, 1)
    value = Decimal("1.2")
    fx_rate = FXRate(ccy1, ccy2, date, value)
    assert fx_rate.ccy1 == ccy1
    assert fx_rate.ccy2 == ccy2
    assert fx_rate.date == date
    assert fx_rate.value == value

def test_fxrate_constructor_with_invalid_ccy1_type():
    ccy1 = "EUR"
    ccy2 = Currency("USD", 2)
    date = Date(2023, 10, 1)
    value = Decimal("1.2")
    try:
        FXRate(ccy1, ccy2, date, value)
    except TypeError:
        pass
    else:
        assert False, "Expected TypeError"

def test_fxrate_constructor_with_invalid_ccy2_type():
    ccy1 = Currency("EUR", 2)
    ccy2 = "USD"
    date = Date(2023, 10, 1)
    value = Decimal("1.2")
    try:
        FXRate(ccy1, ccy2, date, value)
    except TypeError:
        pass
    else:
        assert False, "Expected TypeError"

def test_fxrate_constructor_with_invalid_date_type():
    ccy1 = Currency("EUR", 2)
    ccy2 = Currency("USD", 2)
    date = "2023-10-01"
    value = Decimal("1.2")
    try:
        FXRate(ccy1, ccy2, date, value)
    except TypeError:
        pass
    else:
        assert False, "Expected TypeError"

def test_fxrate_constructor_with_invalid_value_type():
    ccy1 = Currency("EUR", 2)
    ccy2 = Currency("USD", 2)
    date = Date(2023, 10, 1)
    value = "1.2"
    try:
        FXRate(ccy1, ccy2, date, value)
    except TypeError:
        pass
    else:
        assert False, "Expected TypeError"

def test_fxrate_constructor_with_non_positive_value():
    ccy1 = Currency("EUR", 2)
    ccy2 = Currency("USD", 2)
    date = Date(2023, 10, 1)
    value = Decimal("0")
    try:
        FXRate(ccy1, ccy2, date, value)
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError"

def test_fxrate_constructor_with_same_ccy_and_non_unit_value():
    ccy1 = Currency("EUR", 2)
    ccy2 = Currency("EUR", 2)
    date = Date(2023, 10, 1)
    value = Decimal("1.2")
    try:
        FXRate(ccy1, ccy2, date, value)
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError"


# LLM-generated content at query #15
#--------------------------

```python
def test_fxrate_constructor():
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


# LLM-generated content at query #16
#--------------------------

```python
def test_fxrate_constructor():
    from datetime import date
    from decimal import Decimal
    from pypara.currencies import Currencies

    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    fx_date = date.today()
    value = Decimal("2")

    rate = FXRate(ccy1, ccy2, fx_date, value)

    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == fx_date
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


# LLM-generated content at query #19
#--------------------------

```python
def test_query_method():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    date = Date(2023, 10, 1)
    fx_rate = Decimal("1.05")
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            return fx_rate if ccy1 == usd and ccy2 == eur and asof == date else None
        
        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    service = MockFXRateService()
    result = service.query(usd, eur, date)
    assert result == fx_rate
    assert service.query(eur, usd, date) is None


# LLM-generated content at query #20
#--------------------------

```python
def test_queries_with_empty_input():
    service = FXRateService()
    result = service.queries([])
    assert list(result) == []

def test_queries_with_valid_input():
    service = FXRateService()
    queries = [(Currency("USD"), Currency("EUR"), Date(2023, 10, 1))]
    result = service.queries(queries)
    assert len(list(result)) == 1

def test_queries_with_strict_mode():
    service = FXRateService()
    queries = [(Currency("USD"), Currency("EUR"), Date(2023, 10, 1))]
    result = service.queries(queries, strict=True)
    assert len(list(result)) == 1

def test_queries_with_multiple_queries():
    service = FXRateService()
    queries = [
        (Currency("USD"), Currency("EUR"), Date(2023, 10, 1)),
        (Currency("GBP"), Currency("JPY"), Date(2023, 10, 1))
    ]
    result = service.queries(queries)
    assert len(list(result)) == 2

def test_queries_with_invalid_query():
    service = FXRateService()
    queries = [(Currency("USD"), Currency("XYZ"), Date(2023, 10, 1))]
    result = service.queries(queries)
    assert next(result) is None

def test_queries_with_invalid_query_strict_mode():
    service = FXRateService()
    queries = [(Currency("USD"), Currency("XYZ"), Date(2023, 10, 1))]
    try:
        service.queries(queries, strict=True)
        assert False
    except LookupError:
        assert True


# LLM-generated content at query #21
#--------------------------

```python
def test_FXRate_constructor_with_valid_arguments():
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

def test_FXRate_constructor_with_same_currency_and_value_one():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["EUR"]
    date = datetime.date.today()
    value = Decimal("1")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value


# LLM-generated content at query #22
#--------------------------

```python
def test_FXRate_constructor_creates_valid_instance():
    from datetime import date
    from decimal import Decimal
    from pypara.currencies import Currencies
    fx_rate = FXRate(Currencies["EUR"], Currencies["USD"], date.today(), Decimal("2"))
    assert fx_rate.ccy1 == Currencies["EUR"]
    assert fx_rate.ccy2 == Currencies["USD"]
    assert fx_rate.date == date.today()
    assert fx_rate.value == Decimal("2")


# LLM-generated content at query #23
#--------------------------

```python
def test_query_returns_fx_rate():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    date = Date(2023, 10, 1)
    fx_rate = FXRate.of(usd, eur, date, Decimal("0.85"))

    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return fx_rate

        def queries(self, queries, strict=False):
            return [fx_rate] * len(queries)

    service = MockFXRateService()
    result = service.query(usd, eur, date)
    assert result == fx_rate

def test_query_returns_none_when_rate_not_found():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    date = Date(2023, 10, 1)

    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None

        def queries(self, queries, strict=False):
            return [None] * len(queries)

    service = MockFXRateService()
    result = service.query(usd, eur, date)
    assert result is None

def test_query_raises_error_in_strict_mode():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    date = Date(2023, 10, 1)

    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if strict:
                raise LookupError("FX rate not found")
            return None

        def queries(self, queries, strict=False):
            return [None] * len(queries)

    service = MockFXRateService()
    try:
        service.query(usd, eur, date, strict=True)
        assert False, "Expected LookupError"
    except LookupError:
        assert True


# LLM-generated content at query #24
#--------------------------

```python
def test_queries_returns_correct_fx_rates():
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            return FXRate(Decimal('1.0')) if ccy1 == Currency.USD and ccy2 == Currency.EUR else None

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = MockFXRateService()
    queries = [(Currency.USD, Currency.EUR, Date.today()), (Currency.EUR, Currency.USD, Date.today())]
    results = list(service.queries(queries))
    assert results == [FXRate(Decimal('1.0')), None]

def test_queries_raises_lookup_error_when_strict_is_true():
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            if strict and ccy1 == Currency.EUR and ccy2 == Currency.USD:
                raise LookupError("FX rate not found")
            return FXRate(Decimal('1.0')) if ccy1 == Currency.USD and ccy2 == Currency.EUR else None

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = MockFXRateService()
    queries = [(Currency.USD, Currency.EUR, Date.today()), (Currency.EUR, Currency.USD, Date.today())]
    try:
        list(service.queries(queries, strict=True))
    except LookupError as e:
        assert str(e) == "FX rate not found"
    else:
        assert False, "Expected LookupError to be raised"


# LLM-generated content at query #25
#--------------------------

```python
def test_constructor_valid_input():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("2"))
    assert rate.ccy1 == Currencies["EUR"]
    assert rate.ccy2 == Currencies["USD"]
    assert rate.date == datetime.date.today()
    assert rate.value == Decimal("2")

def test_constructor_invalid_ccy1():
    import datetime
    from decimal import Decimal
    try:
        FXRate("invalid_ccy", Currencies["USD"], datetime.date.today(), Decimal("2"))
    except ValueError as e:
        assert str(e) == "CCY/1 must be of type `Currency`."

def test_constructor_invalid_ccy2():
    import datetime
    from decimal import Decimal
    try:
        FXRate(Currencies["EUR"], "invalid_ccy", datetime.date.today(), Decimal("2"))
    except ValueError as e:
        assert str(e) == "CCY/2 must be of type `Currency`."

def test_constructor_invalid_date():
    from decimal import Decimal
    try:
        FXRate(Currencies["EUR"], Currencies["USD"], "invalid_date", Decimal("2"))
    except ValueError as e:
        assert str(e) == "FX rate date must be of type `date`."

def test_constructor_invalid_value():
    import datetime
    try:
        FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), "invalid_value")
    except ValueError as e:
        assert str(e) == "FX rate value must be of type `Decimal`."

def test_constructor_zero_value():
    import datetime
    from decimal import Decimal
    try:
        FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("0"))
    except ValueError as e:
        assert str(e) == "FX rate value can not be equal to or less than `zero`."

def test_constructor_negative_value():
    import datetime
    from decimal import Decimal
    try:
        FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("-1"))
    except ValueError as e:
        assert str(e) == "FX rate value can not be equal to or less than `zero`."

def test_constructor_same_currency_invalid_value():
    import datetime
    from decimal import Decimal
    try:
        FXRate(Currencies["EUR"], Currencies["EUR"], datetime.date.today(), Decimal("2"))
    except ValueError as e:
        assert str(e) == "FX rate to the same currency must be `one`."

def test_constructor_same_currency_valid_value():
    import datetime
    from decimal import Decimal
    rate = FXRate(Currencies["EUR"], Currencies["EUR"], datetime.date.today(), Decimal("1"))
    assert rate.ccy1 == Currencies["EUR"]
    assert rate.ccy2 == Currencies["EUR"]
    assert rate.date == datetime.date.today()
    assert rate.value == Decimal("1")


# LLM-generated content at query #26
#--------------------------

```python
def test_fxrate_constructor_valid_input():
    ccy1 = Currency("EUR")
    ccy2 = Currency("USD")
    date = datetime.date.today()
    value = Decimal("2")
    fxrate = FXRate(ccy1, ccy2, date, value)
    assert fxrate.ccy1 == ccy1
    assert fxrate.ccy2 == ccy2
    assert fxrate.date == date
    assert fxrate.value == value

def test_fxrate_constructor_invalid_ccy1_type():
    ccy1 = "EUR"
    ccy2 = Currency("USD")
    date = datetime.date.today()
    value = Decimal("2")
    try:
        FXRate(ccy1, ccy2, date, value)
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_fxrate_constructor_invalid_ccy2_type():
    ccy1 = Currency("EUR")
    ccy2 = "USD"
    date = datetime.date.today()
    value = Decimal("2")
    try:
        FXRate(ccy1, ccy2, date, value)
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_fxrate_constructor_invalid_date_type():
    ccy1 = Currency("EUR")
    ccy2 = Currency("USD")
    date = "2023-10-01"
    value = Decimal("2")
    try:
        FXRate(ccy1, ccy2, date, value)
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_fxrate_constructor_invalid_value_type():
    ccy1 = Currency("EUR")
    ccy2 = Currency("USD")
    date = datetime.date.today()
    value = "2"
    try:
        FXRate(ccy1, ccy2, date, value)
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_fxrate_constructor_zero_value():
    ccy1 = Currency("EUR")
    ccy2 = Currency("USD")
    date = datetime.date.today()
    value = Decimal("0")
    try:
        FXRate(ccy1, ccy2, date, value)
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_fxrate_constructor_negative_value():
    ccy1 = Currency("EUR")
    ccy2 = Currency("USD")
    date = datetime.date.today()
    value = Decimal("-1")
    try:
        FXRate(ccy1, ccy2, date, value)
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_fxrate_constructor_same_currency_valid_value():
    ccy1 = Currency("EUR")
    ccy2 = Currency("EUR")
    date = datetime.date.today()
    value = Decimal("1")
    fxrate = FXRate(ccy1, ccy2, date, value)
    assert fxrate.ccy1 == ccy1
    assert fxrate.ccy2 == ccy2
    assert fxrate.date == date
    assert fxrate.value == value

def test_fxrate_constructor_same_currency_invalid_value():
    ccy1 = Currency("EUR")
    ccy2 = Currency("EUR")
    date = datetime.date.today()
    value = Decimal("2")
    try:
        FXRate(ccy1, ccy2, date, value)
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #27
#--------------------------

```python
def test_queries_returns_correct_rates():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return FXRate(Decimal('1.0')) if ccy1 == Currency('USD') and ccy2 == Currency('EUR') else None
        
        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    service = MockFXRateService()
    queries = [(Currency('USD'), Currency('EUR'), Date(2023, 1, 1)), (Currency('GBP'), Currency('JPY'), Date(2023, 1, 1))]
    results = service.queries(queries)
    assert results == [FXRate(Decimal('1.0')), None]

def test_queries_strict_mode_raises_error():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if strict and ccy1 != Currency('USD'):
                raise LookupError("FX rate not found")
            return FXRate(Decimal('1.0')) if ccy1 == Currency('USD') and ccy2 == Currency('EUR') else None
        
        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    service = MockFXRateService()
    queries = [(Currency('USD'), Currency('EUR'), Date(2023, 1, 1)), (Currency('GBP'), Currency('JPY'), Date(2023, 1, 1))]
    try:
        service.queries(queries, strict=True)
    except LookupError:
        pass
    else:
        assert False, "Expected LookupError"

def test_queries_empty_queries_returns_empty_list():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return FXRate(Decimal('1.0')) if ccy1 == Currency('USD') and ccy2 == Currency('EUR') else None
        
        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    service = MockFXRateService()
    queries = []
    results = service.queries(queries)
    assert results == []


# LLM-generated content at query #28
#--------------------------

def test_query_returns_fx_rate_for_currency_pair():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return FXRate.of(ccy1, ccy2, asof, Decimal("1.5"))

        def queries(self, queries, strict=False):
            return [FXRate.of(q[0], q[1], q[2], Decimal("1.5")) for q in queries]

    usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date.today()
    service = MockFXRateService()
    rate = service.query(usd, eur, asof)
    assert rate.ccy1 == usd
    assert rate.ccy2 == eur
    assert rate.date == asof
    assert rate.value == Decimal("1.5")

def test_query_returns_none_when_rate_not_found():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None

        def queries(self, queries, strict=False):
            return [None for _ in queries]

    usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date.today()
    service = MockFXRateService()
    assert service.query(usd, eur, asof) is None

def test_query_raises_error_in_strict_mode_when_rate_not_found():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if strict:
                raise LookupError("Rate not found")
            return None

        def queries(self, queries, strict=False):
            if strict:
                raise LookupError("Rate not found")
            return [None for _ in queries]

    usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date.today()
    service = MockFXRateService()
    try:
        service.query(usd, eur, asof, strict=True)
        assert False, "Expected LookupError"
    except LookupError:
        pass


# LLM-generated content at query #29
#--------------------------

```python
def test_FXRate_constructor_with_valid_arguments():
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

def test_FXRate_constructor_with_same_currency_and_value_one():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["EUR"]
    date = datetime.date.today()
    value = Decimal("1")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value

def test_FXRate_constructor_with_same_currency_and_value_not_one():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["EUR"]
    date = datetime.date.today()
    value = Decimal("2")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value

def test_FXRate_constructor_with_zero_value():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("0")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value

def test_FXRate_constructor_with_negative_value():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("-1")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value


# LLM-generated content at query #30
#--------------------------

```
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

def test_fxrate_constructor_with_same_currency_and_value_one():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["EUR"]
    date = datetime.date.today()
    value = Decimal("1")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value

def test_fxrate_constructor_with_same_currency_and_value_not_one():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["EUR"]
    date = datetime.date.today()
    value = Decimal("2")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value

def test_fxrate_constructor_with_zero_value():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("0")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value

def test_fxrate_constructor_with_negative_value():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("-1")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value


# LLM-generated content at query #31
#--------------------------

def test_query_returns_fx_rate_for_currency_pair():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return FXRate.of(ccy1, ccy2, asof, Decimal("1.5"))

        def queries(self, queries, strict=False):
            return [FXRate.of(q[0], q[1], q[2], Decimal("1.5")) for q in queries]

    ccy1 = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date.today()
    service = MockFXRateService()
    rate = service.query(ccy1, ccy2, asof)
    assert rate == FXRate.of(ccy1, ccy2, asof, Decimal("1.5"))

def test_query_returns_none_when_rate_not_found():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None

        def queries(self, queries, strict=False):
            return [None for _ in queries]

    ccy1 = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date.today()
    service = MockFXRateService()
    rate = service.query(ccy1, ccy2, asof)
    assert rate is None

def test_query_raises_error_in_strict_mode_when_rate_not_found():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if strict:
                raise LookupError("FX rate not found")
            return None

        def queries(self, queries, strict=False):
            if strict:
                raise LookupError("FX rate not found")
            return [None for _ in queries]

    ccy1 = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date.today()
    service = MockFXRateService()
    try:
        service.query(ccy1, ccy2, asof, strict=True)
        assert False, "Expected LookupError"
    except LookupError:
        pass


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```
def test_invert_fx_rate():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    nrate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("2"))
    rrate = FXRate(Currencies["USD"], Currencies["EUR"], datetime.date.today(), Decimal("0.5"))
    assert ~nrate == rrate


# LLM-generated content at query #2
#--------------------------

```python
def test_invert_fx_rate():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    nrate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("2"))
    rrate = FXRate(Currencies["USD"], Currencies["EUR"], datetime.date.today(), Decimal("0.5"))
    assert ~nrate == rrate


# LLM-generated content at query #3
#--------------------------

```python
def test_queries_returns_correct_fx_rates():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if ccy1 == "USD" and ccy2 == "EUR" and asof == "2023-10-01":
                return FXRate("USD", "EUR", "2023-10-01", Decimal("0.85"))
            if ccy1 == "GBP" and ccy2 == "USD" and asof == "2023-10-01":
                return FXRate("GBP", "USD", "2023-10-01", Decimal("1.25"))
            return None

        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = MockFXRateService()
    queries = [("USD", "EUR", "2023-10-01"), ("GBP", "USD", "2023-10-01")]
    results = list(service.queries(queries))
    assert results[0] == FXRate("USD", "EUR", "2023-10-01", Decimal("0.85"))
    assert results[1] == FXRate("GBP", "USD", "2023-10-01", Decimal("1.25"))

def test_queries_returns_none_for_missing_fx_rates():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None

        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = MockFXRateService()
    queries = [("USD", "EUR", "2023-10-01"), ("GBP", "USD", "2023-10-01")]
    results = list(service.queries(queries))
    assert results[0] is None
    assert results[1] is None

def test_queries_raises_error_in_strict_mode():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if strict:
                raise LookupError("FX rate not found")
            return None

        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = MockFXRateService()
    queries = [("USD", "EUR", "2023-10-01"), ("GBP", "USD", "2023-10-01")]
    try:
        list(service.queries(queries, strict=True))
        assert False, "Expected LookupError"
    except LookupError:
        pass


# LLM-generated content at query #4
#--------------------------

```
def test_invert_fx_rate():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    nrate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("2"))
    rrate = FXRate(Currencies["USD"], Currencies["EUR"], datetime.date.today(), Decimal("0.5"))
    assert ~nrate == rrate


# LLM-generated content at query #5
#--------------------------

```python
def test_fxrateservice_query():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    date = Date(2023, 10, 1)
    fx_rate = FXRate(usd, eur, date, Decimal("0.85"))

    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            return fx_rate if ccy1 == usd and ccy2 == eur and asof == date else None

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [fx_rate if ccy1 == usd and ccy2 == eur and asof == date else None for ccy1, ccy2, asof in queries]

    service = MockFXRateService()
    result = service.query(usd, eur, date)
    assert result == fx_rate
    assert result.ccy1 == usd
    assert result.ccy2 == eur
    assert result.date == date
    assert result.rate == Decimal("0.85")


# LLM-generated content at query #6
#--------------------------

```python
def test_query_method_with_valid_currencies_and_date():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 10, 1)
    service = FXRateService()
    rate = service.query(ccy1, ccy2, asof)
    assert rate is not None

def test_query_method_with_invalid_currencies():
    ccy1 = Currency.of("XYZ", "Unknown Currency", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("ABC", "Another Unknown Currency", 2, CurrencyType.MONEY)
    asof = Date(2023, 10, 1)
    service = FXRateService()
    rate = service.query(ccy1, ccy2, asof, strict=True)
    assert rate is None

def test_query_method_with_strict_flag_raises_error():
    ccy1 = Currency.of("XYZ", "Unknown Currency", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("ABC", "Another Unknown Currency", 2, CurrencyType.MONEY)
    asof = Date(2023, 10, 1)
    service = FXRateService()
    try:
        service.query(ccy1, ccy2, asof, strict=True)
    except LookupError:
        assert True
    else:
        assert False

def test_query_method_with_same_currency_returns_one():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    asof = Date(2023, 10, 1)
    service = FXRateService()
    rate = service.query(ccy1, ccy2, asof)
    assert rate == FXRate(Decimal("1.0"))

def test_query_method_with_null_asof_date():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = None
    service = FXRateService()
    rate = service.query(ccy1, ccy2, asof)
    assert rate is None


# LLM-generated content at query #7
#--------------------------

```python
def test_queries_returns_rates_for_valid_queries():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return FXRate(Decimal("1.5")) if ccy1 == Currency("USD") and ccy2 == Currency("EUR") else None

        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = MockFXRateService()
    queries = [(Currency("USD"), Currency("EUR"), Date("2023-10-01")), (Currency("GBP"), Currency("JPY"), Date("2023-10-01"))]
    result = list(service.queries(queries))
    assert result == [FXRate(Decimal("1.5")), None]

def test_queries_raises_error_in_strict_mode():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if ccy1 == Currency("USD") and ccy2 == Currency("EUR"):
                return FXRate(Decimal("1.5"))
            elif strict:
                raise LookupError("Rate not found")
            return None

        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = MockFXRateService()
    queries = [(Currency("USD"), Currency("EUR"), Date("2023-10-01")), (Currency("GBP"), Currency("JPY"), Date("2023-10-01"))]
    try:
        list(service.queries(queries, strict=True))
    except LookupError:
        pass
    else:
        raise AssertionError("Expected LookupError not raised")


# LLM-generated content at query #8
#--------------------------

```python
def test_queries_returns_correct_fx_rates():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return FXRate(Decimal('1.5')) if ccy1 == Currency('USD') and ccy2 == Currency('EUR') else None

        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = MockFXRateService()
    queries = [(Currency('USD'), Currency('EUR'), Date('2023-10-01')), (Currency('GBP'), Currency('JPY'), Date('2023-10-01'))]
    result = list(service.queries(queries))
    assert len(result) == 2
    assert result[0] == FXRate(Decimal('1.5'))
    assert result[1] is None

def test_queries_raises_error_in_strict_mode():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if strict and ccy1 == Currency('USD') and ccy2 == Currency('EUR'):
                raise LookupError('FX rate not found')
            return None

        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = MockFXRateService()
    queries = [(Currency('USD'), Currency('EUR'), Date('2023-10-01'))]
    try:
        list(service.queries(queries, strict=True))
    except LookupError as e:
        assert str(e) == 'FX rate not found'


# LLM-generated content at query #9
#--------------------------

```python
def test_queries_returns_correct_rates():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return Decimal('1.5') if ccy1 == Currency('USD') and ccy2 == Currency('EUR') else None

        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = MockFXRateService()
    queries = [
        (Currency('USD'), Currency('EUR'), Date('2023-01-01')),
        (Currency('EUR'), Currency('USD'), Date('2023-01-01'))
    ]
    results = list(service.queries(queries))
    assert results == [Decimal('1.5'), None]

def test_queries_strict_mode_raises_error():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if strict and ccy1 == Currency('JPY'):
                raise LookupError('Rate not found')
            return None

        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = MockFXRateService()
    queries = [
        (Currency('JPY'), Currency('USD'), Date('2023-01-01')),
        (Currency('USD'), Currency('JPY'), Date('2023-01-01'))
    ]
    try:
        list(service.queries(queries, strict=True))
        assert False, 'Expected LookupError not raised'
    except LookupError:
        pass

def test_queries_empty_input_returns_empty_list():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None

        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = MockFXRateService()
    results = list(service.queries([]))
    assert results == []


# LLM-generated content at query #10
#--------------------------

```python
def test_query_with_valid_currencies_and_date():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date.today()
    service = FXRateService()
    result = service.query(ccy1, ccy2, asof)
    assert result is not None

def test_query_with_same_currencies_returns_one():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    asof = Date.today()
    service = FXRateService()
    result = service.query(ccy1, ccy1, asof)
    assert result == FXRate.one(ccy1, ccy1, asof)

def test_query_with_none_currencies_raises_error():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    asof = Date.today()
    service = FXRateService()
    with pytest.raises(ValueError):
        service.query(ccy1, None, asof)

def test_query_with_invalid_date_raises_error():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    service = FXRateService()
    with pytest.raises(ValueError):
        service.query(ccy1, ccy2, None)

def test_query_with_non_existent_rate_returns_none():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("XYZ", "Non-existent", 2, CurrencyType.MONEY)
    asof = Date.today()
    service = FXRateService()
    result = service.query(ccy1, ccy2, asof)
    assert result is None

def test_query_with_strict_mode_raises_error_for_non_existent_rate():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("XYZ", "Non-existent", 2, CurrencyType.MONEY)
    asof = Date.today()
    service = FXRateService()
    with pytest.raises(LookupError):
        service.query(ccy1, ccy2, asof, strict=True)


# LLM-generated content at query #11
#--------------------------

def test_query_with_valid_currencies_and_date():
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return FXRate.of(ccy1, ccy2, asof, Decimal("1.5"))

        def queries(self, queries, strict=False):
            return [FXRate.of(q[0], q[1], q[2], Decimal("1.5")) for q in queries]

    usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    date = Date.today()
    service = TestFXRateService()
    result = service.query(usd, eur, date)
    assert result.ccy1 == usd
    assert result.ccy2 == eur
    assert result.date == date
    assert result.value == Decimal("1.5")

def test_query_with_none_result():
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None

        def queries(self, queries, strict=False):
            return [None for _ in queries]

    usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    date = Date.today()
    service = TestFXRateService()
    result = service.query(usd, eur, date)
    assert result is None

def test_query_with_strict_mode_raises_error():
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if strict:
                raise LookupError("Rate not found")
            return None

        def queries(self, queries, strict=False):
            if strict:
                raise LookupError("Rate not found")
            return [None for _ in queries]

    usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    date = Date.today()
    service = TestFXRateService()
    try:
        service.query(usd, eur, date, strict=True)
        assert False, "Expected LookupError"
    except LookupError:
        pass


# LLM-generated content at query #12
#--------------------------

```python
def test_fxrate_constructor_valid_input():
    from datetime import date
    from decimal import Decimal
    from pypara.currencies import Currencies
    fx_rate = FXRate(Currencies["EUR"], Currencies["USD"], date.today(), Decimal("2"))
    assert fx_rate.ccy1 == Currencies["EUR"]
    assert fx_rate.ccy2 == Currencies["USD"]
    assert fx_rate.date == date.today()
    assert fx_rate.value == Decimal("2")

def test_fxrate_constructor_same_currency_valid_input():
    from datetime import date
    from decimal import Decimal
    from pypara.currencies import Currencies
    fx_rate = FXRate(Currencies["EUR"], Currencies["EUR"], date.today(), Decimal("1"))
    assert fx_rate.ccy1 == Currencies["EUR"]
    assert fx_rate.ccy2 == Currencies["EUR"]
    assert fx_rate.date == date.today()
    assert fx_rate.value == Decimal("1")


# LLM-generated content at query #13
#--------------------------

```python
def test_fxrate_constructor():
    from pypara.currencies import Currencies
    import datetime
    from decimal import Decimal
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("2")
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


# LLM-generated content at query #15
#--------------------------

```python
def test_fxrate_constructor():
    from pypara.currencies import Currencies
    import datetime
    from decimal import Decimal
    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("2"))
    ccy1, ccy2, date, value = rate
    assert ccy1 == Currencies["EUR"]
    assert ccy2 == Currencies["USD"]
    assert date == datetime.date.today()
    assert value == Decimal("2")


# LLM-generated content at query #16
#--------------------------

```
def test_fxrate_constructor_creates_valid_instance():
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

def test_fxrate_constructor_allows_indexed_access():
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

def test_fxrate_constructor_allows_unpacking():
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
    date = datetime.date.today()
    value = Decimal("2")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value


# LLM-generated content at query #18
#--------------------------

```
def test_fxrate_constructor_with_valid_arguments():
    ccy1 = Currency("EUR", 2)
    ccy2 = Currency("USD", 2)
    date = Date(2023, 1, 1)
    value = Decimal("1.5")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value

def test_fxrate_constructor_with_same_currency_and_value_one():
    ccy1 = Currency("EUR", 2)
    ccy2 = Currency("EUR", 2)
    date = Date(2023, 1, 1)
    value = Decimal("1")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value

def test_fxrate_constructor_with_same_currency_and_value_not_one():
    ccy1 = Currency("EUR", 2)
    ccy2 = Currency("EUR", 2)
    date = Date(2023, 1, 1)
    value = Decimal("2")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value

def test_fxrate_constructor_with_zero_value():
    ccy1 = Currency("EUR", 2)
    ccy2 = Currency("USD", 2)
    date = Date(2023, 1, 1)
    value = Decimal("0")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value

def test_fxrate_constructor_with_negative_value():
    ccy1 = Currency("EUR", 2)
    ccy2 = Currency("USD", 2)
    date = Date(2023, 1, 1)
    value = Decimal("-1.5")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value


# LLM-generated content at query #19
#--------------------------

```python
def test_FXRate_constructor_with_valid_arguments():
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

def test_FXRate_constructor_with_same_currency_and_value_one():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["EUR"]
    date = datetime.date.today()
    value = Decimal("1")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value

def test_FXRate_constructor_with_same_currency_and_value_not_one():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["EUR"]
    date = datetime.date.today()
    value = Decimal("2")
    try:
        FXRate(ccy1, ccy2, date, value)
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_FXRate_constructor_with_zero_value():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("0")
    try:
        FXRate(ccy1, ccy2, date, value)
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_FXRate_constructor_with_negative_value():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("-1")
    try:
        FXRate(ccy1, ccy2, date, value)
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #20
#--------------------------

```python
def test_query_method_returns_fx_rate_for_valid_currency_pair_and_date():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return FXRate(ccy1, ccy2, asof, Decimal("1.0"))
        
        def queries(self, queries, strict=False):
            return [FXRate(ccy1, ccy2, asof, Decimal("1.0")) for ccy1, ccy2, asof in queries]
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    asof = Date(2023, 10, 1)
    
    service = MockFXRateService()
    rate = service.query(usd, eur, asof)
    
    assert rate.ccy1 == usd
    assert rate.ccy2 == eur
    assert rate.asof == asof
    assert rate.value == Decimal("1.0")

def test_query_method_raises_lookup_error_for_invalid_currency_pair_when_strict():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if strict:
                raise LookupError("FX rate not found")
            return None
        
        def queries(self, queries, strict=False):
            return [None for _ in queries]
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    asof = Date(2023, 10, 1)
    
    service = MockFXRateService()
    try:
        service.query(usd, eur, asof, strict=True)
        assert False, "Expected LookupError"
    except LookupError:
        pass

def test_query_method_returns_none_for_invalid_currency_pair_when_not_strict():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        
        def queries(self, queries, strict=False):
            return [None for _ in queries]
    
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    asof = Date(2023, 10, 1)
    
    service = MockFXRateService()
    rate = service.query(usd, eur, asof)
    assert rate is None


# LLM-generated content at query #21
#--------------------------

```python
def test_queries_returns_iterable_of_rates():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return Decimal('1.5') if ccy1 == Currency('USD') and ccy2 == Currency('EUR') else None

        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = MockFXRateService()
    queries = [
        (Currency('USD'), Currency('EUR'), Date('2023-01-01')),
        (Currency('GBP'), Currency('JPY'), Date('2023-01-01'))
    ]
    results = list(service.queries(queries))
    assert results == [Decimal('1.5'), None]

def test_queries_strict_mode_raises_error():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if strict and ccy1 == Currency('GBP'):
                raise LookupError('Rate not found')
            return Decimal('1.5') if ccy1 == Currency('USD') else None

        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = MockFXRateService()
    queries = [
        (Currency('USD'), Currency('EUR'), Date('2023-01-01')),
        (Currency('GBP'), Currency('JPY'), Date('2023-01-01'))
    ]
    try:
        list(service.queries(queries, strict=True))
        assert False, 'Expected LookupError'
    except LookupError:
        pass

def test_queries_empty_input_returns_empty_iterable():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None

        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = MockFXRateService()
    results = list(service.queries([]))
    assert results == []


# LLM-generated content at query #22
#--------------------------

```python
def test_constructor_creates_valid_fxrate():
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


# LLM-generated content at query #23
#--------------------------

```
def test_fxrate_constructor_with_valid_arguments():
    ccy1 = Currency("EUR", 2)
    ccy2 = Currency("USD", 2)
    date = Date(2023, 1, 1)
    value = Decimal("1.5")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value

def test_fxrate_constructor_with_same_currency_and_value_one():
    ccy1 = Currency("EUR", 2)
    ccy2 = Currency("EUR", 2)
    date = Date(2023, 1, 1)
    value = Decimal("1")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value

def test_fxrate_constructor_with_same_currency_and_value_not_one():
    ccy1 = Currency("EUR", 2)
    ccy2 = Currency("EUR", 2)
    date = Date(2023, 1, 1)
    value = Decimal("1.5")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value

def test_fxrate_constructor_with_zero_value():
    ccy1 = Currency("EUR", 2)
    ccy2 = Currency("USD", 2)
    date = Date(2023, 1, 1)
    value = Decimal("0")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value

def test_fxrate_constructor_with_negative_value():
    ccy1 = Currency("EUR", 2)
    ccy2 = Currency("USD", 2)
    date = Date(2023, 1, 1)
    value = Decimal("-1.5")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value


# LLM-generated content at query #24
#--------------------------

```python
def test_query_method():
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            return Decimal("1.25") if ccy1.code == "USD" and ccy2.code == "EUR" else None

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            pass

    service = MockFXRateService()
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    date = Date(2023, 10, 1)

    assert service.query(usd, eur, date) == Decimal("1.25")
    assert service.query(eur, usd, date) is None


# LLM-generated content at query #25
#--------------------------

```python
def test_queries_returns_empty_iterable_for_empty_input():
    service = FXRateService()
    result = service.queries([])
    assert list(result) == []

def test_queries_returns_none_for_invalid_query():
    service = FXRateService()
    queries = [(Currency("USD"), Currency("EUR"), Date("2023-01-01"))]
    result = service.queries(queries)
    assert list(result) == [None]

def test_queries_returns_fxrate_for_valid_query():
    service = FXRateService()
    queries = [(Currency("USD"), Currency("EUR"), Date("2023-01-01"))]
    expected_rate = FXRate(Decimal("1.2"))
    service.query = lambda ccy1, ccy2, asof, strict: expected_rate
    result = service.queries(queries)
    assert list(result) == [expected_rate]

def test_queries_raises_error_for_invalid_query_in_strict_mode():
    service = FXRateService()
    queries = [(Currency("USD"), Currency("EUR"), Date("2023-01-01"))]
    service.query = lambda ccy1, ccy2, asof, strict: None
    try:
        result = service.queries(queries, strict=True)
        assert False, "Expected an exception to be raised"
    except LookupError:
        pass

def test_queries_returns_mixed_results_for_multiple_queries():
    service = FXRateService()
    queries = [
        (Currency("USD"), Currency("EUR"), Date("2023-01-01")),
        (Currency("GBP"), Currency("JPY"), Date("2023-01-01"))
    ]
    expected_rates = [FXRate(Decimal("1.2")), None]
    service.query = lambda ccy1, ccy2, asof, strict: expected_rates.pop(0)
    result = service.queries(queries)
    assert list(result) == expected_rates


# LLM-generated content at query #26
#--------------------------

```python
def test_fxrate_constructor():
    ccy1 = Currency("EUR")
    ccy2 = Currency("USD")
    date = Date(2023, 10, 1)
    value = Decimal("1.2")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value


# LLM-generated content at query #27
#--------------------------

```
def test_FXRate_constructor_with_valid_input():
    ccy1 = Currency("EUR", 978)
    ccy2 = Currency("USD", 840)
    date = Date(2023, 10, 1)
    value = Decimal("1.2")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value

def test_FXRate_constructor_with_same_currency_and_value_one():
    ccy1 = Currency("EUR", 978)
    date = Date(2023, 10, 1)
    value = Decimal("1")
    rate = FXRate(ccy1, ccy1, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy1
    assert rate.date == date
    assert rate.value == value

def test_FXRate_constructor_with_same_currency_and_value_not_one():
    ccy1 = Currency("EUR", 978)
    date = Date(2023, 10, 1)
    value = Decimal("1.2")
    rate = FXRate(ccy1, ccy1, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy1
    assert rate.date == date
    assert rate.value == value

def test_FXRate_constructor_with_zero_value():
    ccy1 = Currency("EUR", 978)
    ccy2 = Currency("USD", 840)
    date = Date(2023, 10, 1)
    value = Decimal("0")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value

def test_FXRate_constructor_with_negative_value():
    ccy1 = Currency("EUR", 978)
    ccy2 = Currency("USD", 840)
    date = Date(2023, 10, 1)
    value = Decimal("-1.2")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value


# LLM-generated content at query #28
#--------------------------

def test_fx_rate_service_query_with_strict_mode():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    date = Date(2023, 1, 1)
    service = FXRateService()
    result = service.query(usd, eur, date, strict=True)
    assert result is None

def test_fx_rate_service_query_with_non_strict_mode():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    date = Date(2023, 1, 1)
    service = FXRateService()
    result = service.query(usd, eur, date, strict=False)
    assert result is None

def test_fx_rate_service_query_with_same_currency():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    date = Date(2023, 1, 1)
    service = FXRateService()
    result = service.query(usd, usd, date)
    assert result == FXRate(Decimal("1"))

def test_fx_rate_service_query_with_inverse_currencies():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    date = Date(2023, 1, 1)
    service = FXRateService()
    service.default = MockFXRateService({(usd, eur, date): FXRate(Decimal("0.85"))})
    result = service.query(eur, usd, date)
    assert result == FXRate(Decimal("1.1764705882352941176470588235"))


# LLM-generated content at query #29
#--------------------------

```python
def test_queries_returns_iterable_of_fxrates():
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            return FXRate(Decimal('1.0')) if ccy1 == Currency('USD') and ccy2 == Currency('EUR') else None

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = MockFXRateService()
    queries = [(Currency('USD'), Currency('EUR'), Date.today()), (Currency('GBP'), Currency('JPY'), Date.today())]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == FXRate(Decimal('1.0'))
    assert results[1] is None


# LLM-generated content at query #30
#--------------------------

```
def test_constructor_creates_valid_fxrate():
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


# LLM-generated content at query #31
#--------------------------

```python
def test_FXRate_constructor_with_valid_arguments():
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

def test_FXRate_constructor_with_same_currency_and_value_one():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy = Currencies["EUR"]
    date = datetime.date.today()
    value = Decimal("1")
    rate = FXRate(ccy, ccy, date, value)
    assert rate.ccy1 == ccy
    assert rate.ccy2 == ccy
    assert rate.date == date
    assert rate.value == value

def test_FXRate_constructor_with_same_currency_and_value_not_one():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy = Currencies["EUR"]
    date = datetime.date.today()
    value = Decimal("2")
    try:
        FXRate(ccy, ccy, date, value)
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_FXRate_constructor_with_zero_value():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("0")
    try:
        FXRate(ccy1, ccy2, date, value)
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_FXRate_constructor_with_negative_value():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("-1")
    try:
        FXRate(ccy1, ccy2, date, value)
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #32
#--------------------------

```python
def test_query_method_returns_fx_rate_for_given_currency_pair_and_date():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    date = Date(2023, 10, 1)
    fx_rate = FXRate.of(usd, eur, date, Decimal("0.85"))
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            return fx_rate if ccy1 == usd and ccy2 == eur and asof == date else None
        
        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [fx_rate if ccy1 == usd and ccy2 == eur and asof == date else None for (ccy1, ccy2, asof) in queries]
    
    service = MockFXRateService()
    result = service.query(usd, eur, date)
    assert result == fx_rate


# LLM-generated content at query #33
#--------------------------

```python
def test_queries_returns_correct_rates():
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            return FXRate(Decimal('1.0')) if ccy1 == Currency('USD') and ccy2 == Currency('EUR') else None

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = MockFXRateService()
    queries = [(Currency('USD'), Currency('EUR'), Date('2023-01-01')), (Currency('GBP'), Currency('JPY'), Date('2023-01-01'))]
    result = list(service.queries(queries))
    assert result == [FXRate(Decimal('1.0')), None]

def test_queries_raises_error_when_strict():
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            if strict and ccy1 != Currency('USD') and ccy2 != Currency('EUR'):
                raise LookupError('Rate not found')
            return FXRate(Decimal('1.0')) if ccy1 == Currency('USD') and ccy2 == Currency('EUR') else None

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = MockFXRateService()
    queries = [(Currency('USD'), Currency('EUR'), Date('2023-01-01')), (Currency('GBP'), Currency('JPY'), Date('2023-01-01'))]
    try:
        list(service.queries(queries, strict=True))
    except LookupError:
        pass
    else:
        assert False, 'Expected LookupError to be raised'


# LLM-generated content at query #34
#--------------------------

```
def test_constructor_creates_valid_fxrate():
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

def test_constructor_allows_indexed_access():
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

def test_constructor_allows_unpacking():
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


# LLM-generated content at query #35
#--------------------------

```python
def test_constructor_valid_input():
    ccy1 = Currency("EUR", 978, 2)
    ccy2 = Currency("USD", 840, 2)
    date = Date(2023, 10, 1)
    value = Decimal("1.2")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value

def test_constructor_invalid_ccy1_type():
    ccy1 = "EUR"
    ccy2 = Currency("USD", 840, 2)
    date = Date(2023, 10, 1)
    value = Decimal("1.2")
    try:
        FXRate(ccy1, ccy2, date, value)
        assert False
    except TypeError:
        assert True

def test_constructor_invalid_ccy2_type():
    ccy1 = Currency("EUR", 978, 2)
    ccy2 = "USD"
    date = Date(2023, 10, 1)
    value = Decimal("1.2")
    try:
        FXRate(ccy1, ccy2, date, value)
        assert False
    except TypeError:
        assert True

def test_constructor_invalid_date_type():
    ccy1 = Currency("EUR", 978, 2)
    ccy2 = Currency("USD", 840, 2)
    date = "2023-10-01"
    value = Decimal("1.2")
    try:
        FXRate(ccy1, ccy2, date, value)
        assert False
    except TypeError:
        assert True

def test_constructor_invalid_value_type():
    ccy1 = Currency("EUR", 978, 2)
    ccy2 = Currency("USD", 840, 2)
    date = Date(2023, 10, 1)
    value = "1.2"
    try:
        FXRate(ccy1, ccy2, date, value)
        assert False
    except TypeError:
        assert True

def test_constructor_value_less_than_zero():
    ccy1 = Currency("EUR", 978, 2)
    ccy2 = Currency("USD", 840, 2)
    date = Date(2023, 10, 1)
    value = Decimal("-1.2")
    try:
        FXRate(ccy1, ccy2, date, value)
        assert False
    except ValueError:
        assert True

def test_constructor_same_currency_invalid_value():
    ccy1 = Currency("EUR", 978, 2)
    ccy2 = Currency("EUR", 978, 2)
    date = Date(2023, 10, 1)
    value = Decimal("1.2")
    try:
        FXRate(ccy1, ccy2, date, value)
        assert False
    except ValueError:
        assert True

def test_constructor_same_currency_valid_value():
    ccy1 = Currency("EUR", 978, 2)
    ccy2 = Currency("EUR", 978, 2)
    date = Date(2023, 10, 1)
    value = Decimal("1.0")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value


