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
    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("2"))
    inverted_rate = ~rate
    expected_rate = FXRate(Currencies["USD"], Currencies["EUR"], datetime.date.today(), Decimal("0.5"))
    assert inverted_rate == expected_rate


# LLM-generated content at query #2
#--------------------------

```python
def test_FXRateService_query():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    date = Date.today()
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return FXRate.of(ccy1, ccy2, date, Decimal("1.2"))
        
        def queries(self, queries, strict=False):
            pass
    
    service = MockFXRateService()
    rate = service.query(usd, eur, date)
    
    assert rate.ccy1 == usd
    assert rate.ccy2 == eur
    assert rate.date == date
    assert rate.value == Decimal("1.2")


# LLM-generated content at query #3
#--------------------------

def test_query_with_same_currency_returns_one():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    service = FXRateService()
    result = service.query(usd, usd, Date.today())
    assert result == FXRate.one(usd, usd)

def test_query_with_different_currencies_returns_rate():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    service = FXRateService()
    service.default = MockFXRateService({(usd, eur, Date.today()): Decimal("0.85")})
    result = service.query(usd, eur, Date.today())
    assert result == FXRate(usd, eur, Decimal("0.85"))

def test_query_with_strict_raises_error_when_rate_not_found():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    service = FXRateService()
    service.default = MockFXRateService({})
    try:
        service.query(usd, eur, Date.today(), strict=True)
        assert False
    except LookupError:
        assert True

def test_query_with_non_strict_returns_none_when_rate_not_found():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    service = FXRateService()
    service.default = MockFXRateService({})
    result = service.query(usd, eur, Date.today(), strict=False)
    assert result is None


# LLM-generated content at query #4
#--------------------------

```python
def test_FXRate_constructor_initializes_fields_correctly():
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

def test_FXRate_constructor_allows_indexed_access():
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

def test_FXRate_constructor_allows_unpacking():
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


# LLM-generated content at query #5
#--------------------------

Here are the test cases for the `query` method of `FXRateService`:


# LLM-generated content at query #6
#--------------------------

```python
def test_constructor_initializes_fields_correctly():
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


# LLM-generated content at query #7
#--------------------------

```python
def test_queries_with_valid_input():
    currency1 = Currency("USD")
    currency2 = Currency("EUR")
    asof_date = Date(2023, 10, 1)
    query = [(currency1, currency2, asof_date)]
    fx_rate_service = FXRateService()
    result = fx_rate_service.queries(query, strict=False)
    assert result == [FXRate(currency1, currency2, asof_date, Decimal('0.85'))]

def test_queries_with_invalid_input_strict_mode():
    currency1 = Currency("USD")
    currency2 = Currency("XYZ")
    asof_date = Date(2023, 10, 1)
    query = [(currency1, currency2, asof_date)]
    fx_rate_service = FXRateService()
    result = fx_rate_service.queries(query, strict=True)
    assert result == [None]

def test_queries_with_multiple_queries():
    currency1 = Currency("USD")
    currency2 = Currency("EUR")
    currency3 = Currency("GBP")
    asof_date = Date(2023, 10, 1)
    query = [(currency1, currency2, asof_date), (currency1, currency3, asof_date)]
    fx_rate_service = FXRateService()
    result = fx_rate_service.queries(query, strict=False)
    assert result == [FXRate(currency1, currency2, asof_date, Decimal('0.85')), FXRate(currency1, currency3, asof_date, Decimal('0.75'))]


# LLM-generated content at query #8
#--------------------------

```python
def test_FXRate_constructor_with_valid_arguments():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("2"))

def test_FXRate_constructor_with_invalid_ccy1_type():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    try:
        FXRate("EUR", Currencies["USD"], datetime.date.today(), Decimal("2"))
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_FXRate_constructor_with_invalid_ccy2_type():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    try:
        FXRate(Currencies["EUR"], "USD", datetime.date.today(), Decimal("2"))
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_FXRate_constructor_with_invalid_date_type():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    try:
        FXRate(Currencies["EUR"], Currencies["USD"], "2023-01-01", Decimal("2"))
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_FXRate_constructor_with_invalid_value_type():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    try:
        FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), "2")
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_FXRate_constructor_with_zero_value():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    try:
        FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("0"))
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_FXRate_constructor_with_negative_value():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    try:
        FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("-1"))
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_FXRate_constructor_with_same_currency_and_non_one_value():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    try:
        FXRate(Currencies["EUR"], Currencies["EUR"], datetime.date.today(), Decimal("2"))
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_FXRate_constructor_with_same_currency_and_one_value():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    FXRate(Currencies["EUR"], Currencies["EUR"], datetime.date.today(), Decimal("1"))


# LLM-generated content at query #9
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


# LLM-generated content at query #10
#--------------------------

def test_query_with_valid_currencies_and_date():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    date = Date(2023, 1, 1)
    service = FXRateService()
    result = service.query(usd, eur, date)
    assert result is None or isinstance(result, FXRate)

def test_query_with_same_currencies_returns_one():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    date = Date(2023, 1, 1)
    service = FXRateService()
    result = service.query(usd, usd, date)
    assert result == FXRate.one(usd, usd, date)

def test_query_with_strict_mode_and_missing_rate_raises_error():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    date = Date(2023, 1, 1)
    service = FXRateService()
    try:
        service.query(usd, eur, date, strict=True)
        assert False, "Expected LookupError"
    except LookupError:
        pass

def test_query_with_none_currency_raises_error():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    date = Date(2023, 1, 1)
    service = FXRateService()
    try:
        service.query(None, usd, date)
        assert False, "Expected TypeError"
    except TypeError:
        pass

def test_query_with_invalid_date_raises_error():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    service = FXRateService()
    try:
        service.query(usd, eur, "invalid-date")
        assert False, "Expected TypeError"
    except TypeError:
        pass


# LLM-generated content at query #11
#--------------------------

```python
def test_queries_returns_correct_rates():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return FXRate(Decimal('1.0')) if ccy1 == Currency('USD') and ccy2 == Currency('EUR') else None

        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = MockFXRateService()
    queries = [(Currency('USD'), Currency('EUR'), Date(2023, 10, 1)), (Currency('GBP'), Currency('JPY'), Date(2023, 10, 1))]
    result = service.queries(queries)
    assert result == [FXRate(Decimal('1.0')), None]

def test_queries_raises_error_in_strict_mode():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if ccy1 == Currency('USD') and ccy2 == Currency('EUR'):
                return FXRate(Decimal('1.0'))
            if strict:
                raise LookupError("FX rate not found")
            return None

        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = MockFXRateService()
    queries = [(Currency('USD'), Currency('EUR'), Date(2023, 10, 1)), (Currency('GBP'), Currency('JPY'), Date(2023, 10, 1))]
    try:
        service.queries(queries, strict=True)
        assert False, "Expected LookupError"
    except LookupError:
        pass


# LLM-generated content at query #12
#--------------------------

```python
def test_queries_returns_correct_rates():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return FXRate(Decimal('1.2')) if ccy1 == 'USD' and ccy2 == 'EUR' else None

        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = MockFXRateService()
    queries = [('USD', 'EUR', '2023-01-01'), ('GBP', 'JPY', '2023-01-01')]
    results = list(service.queries(queries))
    assert results[0] == FXRate(Decimal('1.2'))
    assert results[1] is None

def test_queries_strict_mode_raises_error():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if strict and ccy1 != 'USD':
                raise LookupError("Rate not found")
            return FXRate(Decimal('1.2')) if ccy1 == 'USD' else None

        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = MockFXRateService()
    queries = [('USD', 'EUR', '2023-01-01'), ('GBP', 'JPY', '2023-01-01')]
    try:
        list(service.queries(queries, strict=True))
        assert False, "Expected LookupError"
    except LookupError:
        pass

def test_queries_handles_empty_input():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return FXRate(Decimal('1.2'))

        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = MockFXRateService()
    results = list(service.queries([]))
    assert len(results) == 0


# LLM-generated content at query #13
#--------------------------

```python
def test_queries_returns_iterable_of_rates():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return FXRate(Decimal('1.0'), ccy1, ccy2, asof)

        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = MockFXRateService()
    queries = [(Currency('USD'), Currency('EUR'), Date(2023, 1, 1)), (Currency('GBP'), Currency('JPY'), Date(2023, 1, 1))]
    result = list(service.queries(queries))
    assert len(result) == 2
    assert isinstance(result[0], FXRate)
    assert isinstance(result[1], FXRate)
    assert result[0].ccy1 == Currency('USD')
    assert result[0].ccy2 == Currency('EUR')
    assert result[1].ccy1 == Currency('GBP')
    assert result[1].ccy2 == Currency('JPY')


# LLM-generated content at query #14
#--------------------------

```python
def test_fxrate_constructor():
    from datetime import date
    from decimal import Decimal
    from pypara.currencies import Currencies

    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    d = date.today()
    value = Decimal("2")

    rate = FXRate(ccy1, ccy2, d, value)

    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == d
    assert rate.value == value


# LLM-generated content at query #15
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
        assert False, "Expected ValueError but no exception was raised"
    except ValueError:
        pass

def test_FXRate_constructor_with_non_positive_value():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("0")
    try:
        FXRate(ccy1, ccy2, date, value)
        assert False, "Expected ValueError but no exception was raised"
    except ValueError:
        pass


# LLM-generated content at query #16
#--------------------------

```python
def test_fxrate_constructor():
    ccy1 = Currency("EUR", 2)
    ccy2 = Currency("USD", 2)
    date = datetime.date(2023, 10, 1)
    value = Decimal("1.2")
    fx_rate = FXRate(ccy1, ccy2, date, value)
    assert fx_rate.ccy1 == ccy1
    assert fx_rate.ccy2 == ccy2
    assert fx_rate.date == date
    assert fx_rate.value == value

def test_fxrate_constructor_with_same_currency():
    ccy1 = Currency("EUR", 2)
    ccy2 = Currency("EUR", 2)
    date = datetime.date(2023, 10, 1)
    value = Decimal("1.0")
    fx_rate = FXRate(ccy1, ccy2, date, value)
    assert fx_rate.ccy1 == ccy1
    assert fx_rate.ccy2 == ccy2
    assert fx_rate.date == date
    assert fx_rate.value == value

def test_fxrate_constructor_with_zero_value():
    ccy1 = Currency("EUR", 2)
    ccy2 = Currency("USD", 2)
    date = datetime.date(2023, 10, 1)
    value = Decimal("0.0")
    fx_rate = FXRate(ccy1, ccy2, date, value)
    assert fx_rate.ccy1 == ccy1
    assert fx_rate.ccy2 == ccy2
    assert fx_rate.date == date
    assert fx_rate.value == value


# LLM-generated content at query #17
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


# LLM-generated content at query #18
#--------------------------

```python
def test_queries_returns_correct_rates():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return Decimal('1.5') if (ccy1, ccy2, asof) == ('USD', 'EUR', '2023-01-01') else None
        
        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = MockFXRateService()
    test_queries = [('USD', 'EUR', '2023-01-01'), ('EUR', 'USD', '2023-01-01')]
    results = list(service.queries(test_queries))
    assert results == [Decimal('1.5'), None]

def test_queries_strict_mode_raises_error():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if strict and (ccy1, ccy2, asof) == ('EUR', 'USD', '2023-01-01'):
                raise LookupError("Rate not found")
            return None
        
        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = MockFXRateService()
    test_queries = [('EUR', 'USD', '2023-01-01')]
    try:
        list(service.queries(test_queries, strict=True))
        assert False, "Expected LookupError not raised"
    except LookupError:
        assert True

def test_queries_empty_input_returns_empty_list():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        
        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = MockFXRateService()
    results = list(service.queries([]))
    assert results == []


# LLM-generated content at query #19
#--------------------------

```python
def test_query_method_returns_correct_fx_rate():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 10, 1)
    fx_rate = Decimal("0.85")
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            return fx_rate if ccy1 == Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY) and ccy2 == Currency.of("EUR", "Euro", 2, CurrencyType.MONEY) and asof == Date(2023, 10, 1) else None
        
        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    service = MockFXRateService()
    result = service.query(ccy1, ccy2, asof)
    
    assert result == fx_rate

def test_query_method_returns_none_for_non_existent_rate():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    asof = Date(2023, 10, 1)
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            return None
        
        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    service = MockFXRateService()
    result = service.query(ccy1, ccy2, asof)
    
    assert result is None

def test_query_method_raises_error_in_strict_mode():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    asof = Date(2023, 10, 1)
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            if strict:
                raise LookupError("FX rate not found")
            return None
        
        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    service = MockFXRateService()
    
    try:
        service.query(ccy1, ccy2, asof, strict=True)
        assert False, "Expected LookupError"
    except LookupError:
        pass


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


# LLM-generated content at query #22
#--------------------------

```python
def test_query_method_returns_fx_rate():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    date = Date(2023, 10, 1)
    fx_rate = FXRate(usd, eur, date, Decimal("0.85"))
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return fx_rate
        
        def queries(self, queries, strict=False):
            return [fx_rate]
    
    service = MockFXRateService()
    result = service.query(usd, eur, date)
    assert result == fx_rate

def test_query_method_returns_none_when_rate_not_found():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    date = Date(2023, 10, 1)
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        
        def queries(self, queries, strict=False):
            return [None]
    
    service = MockFXRateService()
    result = service.query(usd, eur, date)
    assert result is None

def test_query_method_raises_error_when_strict_and_rate_not_found():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    date = Date(2023, 10, 1)
    
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if strict:
                raise ValueError("FX rate not found")
            return None
        
        def queries(self, queries, strict=False):
            return [None]
    
    service = MockFXRateService()
    try:
        service.query(usd, eur, date, strict=True)
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert str(e) == "FX rate not found"


# LLM-generated content at query #23
#--------------------------

```python
def test_queries_with_empty_input():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        def queries(self, queries, strict=False):
            return super().queries(queries, strict)
    
    service = MockFXRateService()
    result = service.queries([])
    assert list(result) == []

def test_queries_with_non_empty_input():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        def queries(self, queries, strict=False):
            return [Decimal('1.0')] * len(queries)
    
    service = MockFXRateService()
    queries = [('USD', 'EUR', '2023-01-01'), ('GBP', 'JPY', '2023-01-01')]
    result = service.queries(queries)
    assert list(result) == [Decimal('1.0'), Decimal('1.0')]

def test_queries_with_strict_mode():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if strict:
                raise LookupError("FX rate not found")
            return None
        def queries(self, queries, strict=False):
            return super().queries(queries, strict)
    
    service = MockFXRateService()
    queries = [('USD', 'EUR', '2023-01-01'), ('GBP', 'JPY', '2023-01-01')]
    try:
        service.queries(queries, strict=True)
        assert False
    except LookupError:
        assert True


# LLM-generated content at query #24
#--------------------------

```python
def test_fxrate_constructor():
    from datetime import date
    from decimal import Decimal
    from pypara.currencies import Currencies

    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    today = date.today()
    value = Decimal("2")

    fx_rate = FXRate(ccy1, ccy2, today, value)

    assert fx_rate.ccy1 == ccy1
    assert fx_rate.ccy2 == ccy2
    assert fx_rate.date == today
    assert fx_rate.value == value


# LLM-generated content at query #25
#--------------------------

```
def test_fxrate_constructor_with_valid_arguments():
    ccy1 = Currency("EUR", 978, 2)
    ccy2 = Currency("USD", 840, 2)
    date = Date(2023, 1, 1)
    value = Decimal("1.5")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value

def test_fxrate_constructor_with_same_currency_and_value_one():
    ccy1 = Currency("EUR", 978, 2)
    ccy2 = Currency("EUR", 978, 2)
    date = Date(2023, 1, 1)
    value = Decimal("1")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value

def test_fxrate_constructor_with_same_currency_and_value_not_one():
    ccy1 = Currency("EUR", 978, 2)
    ccy2 = Currency("EUR", 978, 2)
    date = Date(2023, 1, 1)
    value = Decimal("2")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value

def test_fxrate_constructor_with_zero_value():
    ccy1 = Currency("EUR", 978, 2)
    ccy2 = Currency("USD", 840, 2)
    date = Date(2023, 1, 1)
    value = Decimal("0")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value

def test_fxrate_constructor_with_negative_value():
    ccy1 = Currency("EUR", 978, 2)
    ccy2 = Currency("USD", 840, 2)
    date = Date(2023, 1, 1)
    value = Decimal("-1.5")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value


# LLM-generated content at query #26
#--------------------------

def test_fxrateservice_query_with_valid_currencies_and_date():
    usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    date = Date(2023, 1, 1)
    service = FXRateService()
    result = service.query(usd, eur, date)
    assert result is None or isinstance(result, FXRate)

def test_fxrateservice_query_with_same_currencies():
    usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    date = Date(2023, 1, 1)
    service = FXRateService()
    result = service.query(usd, usd, date)
    assert result == FXRate.identity(usd)

def test_fxrateservice_query_with_strict_mode_and_missing_rate():
    usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    date = Date(2023, 1, 1)
    service = FXRateService()
    try:
        service.query(usd, eur, date, strict=True)
        assert False, "Expected LookupError"
    except LookupError:
        pass

def test_fxrateservice_query_with_invalid_currency():
    usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    date = Date(2023, 1, 1)
    service = FXRateService()
    try:
        service.query(usd, "INVALID", date)
        assert False, "Expected TypeError"
    except TypeError:
        pass

def test_fxrateservice_query_with_invalid_date():
    usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    service = FXRateService()
    try:
        service.query(usd, eur, "INVALID")
        assert False, "Expected TypeError"
    except TypeError:
        pass


# LLM-generated content at query #27
#--------------------------

```python
def test_queries_returns_iterable_of_rates():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return Decimal('1.5') if ccy1 != ccy2 else None

        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof) for ccy1, ccy2, asof in queries]

    service = MockFXRateService()
    queries = [("USD", "EUR", "2023-01-01"), ("GBP", "JPY", "2023-01-02"), ("USD", "USD", "2023-01-03")]
    result = list(service.queries(queries))
    assert len(result) == 3
    assert result[0] == Decimal('1.5')
    assert result[1] == Decimal('1.5')
    assert result[2] is None

def test_queries_with_strict_raises_error():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if ccy1 == ccy2 and strict:
                raise LookupError("Rate not found")
            return Decimal('1.5') if ccy1 != ccy2 else None

        def queries(self, queries, strict=False):
            if strict:
                for ccy1, ccy2, asof in queries:
                    if ccy1 == ccy2:
                        raise LookupError("Rate not found")
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = MockFXRateService()
    queries = [("USD", "USD", "2023-01-01")]
    try:
        list(service.queries(queries, strict=True))
        assert False, "Expected LookupError"
    except LookupError:
        pass

def test_queries_with_empty_input():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None

        def queries(self, queries, strict=False):
            return []

    service = MockFXRateService()
    result = list(service.queries([]))
    assert len(result) == 0


# LLM-generated content at query #28
#--------------------------

```python
def test_fxrate_constructor():
    ccy1 = Currency("EUR")
    ccy2 = Currency("USD")
    date = Date(2023, 10, 1)
    value = Decimal("1.2")
    fx_rate = FXRate(ccy1, ccy2, date, value)
    assert fx_rate.ccy1 == ccy1
    assert fx_rate.ccy2 == ccy2
    assert fx_rate.date == date
    assert fx_rate.value == value


# LLM-generated content at query #29
#--------------------------

```python
def test_constructor_valid_input():
    from datetime import date
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    rate_date = date.today()
    value = Decimal("2")
    rate = FXRate(ccy1, ccy2, rate_date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == rate_date
    assert rate.value == value

def test_constructor_invalid_ccy1():
    from datetime import date
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = "EUR"
    ccy2 = Currencies["USD"]
    rate_date = date.today()
    value = Decimal("2")
    try:
        FXRate(ccy1, ccy2, rate_date, value)
    except ValueError as e:
        assert str(e) == "CCY/1 must be of type `Currency`."

def test_constructor_invalid_ccy2():
    from datetime import date
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = "USD"
    rate_date = date.today()
    value = Decimal("2")
    try:
        FXRate(ccy1, ccy2, rate_date, value)
    except ValueError as e:
        assert str(e) == "CCY/2 must be of type `Currency`."

def test_constructor_invalid_date():
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    rate_date = "2023-10-01"
    value = Decimal("2")
    try:
        FXRate(ccy1, ccy2, rate_date, value)
    except ValueError as e:
        assert str(e) == "FX rate date must be of type `date`."

def test_constructor_invalid_value():
    from datetime import date
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    rate_date = date.today()
    value = "2"
    try:
        FXRate(ccy1, ccy2, rate_date, value)
    except ValueError as e:
        assert str(e) == "FX rate value must be of type `Decimal`."

def test_constructor_value_less_than_zero():
    from datetime import date
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    rate_date = date.today()
    value = Decimal("-1")
    try:
        FXRate(ccy1, ccy2, rate_date, value)
    except ValueError as e:
        assert str(e) == "FX rate value can not be equal to or less than `zero`."

def test_constructor_same_currency_invalid_value():
    from datetime import date
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["EUR"]
    rate_date = date.today()
    value = Decimal("2")
    try:
        FXRate(ccy1, ccy2, rate_date, value)
    except ValueError as e:
        assert str(e) == "FX rate to the same currency must be `one`."


# LLM-generated content at query #30
#--------------------------

def test_query_with_valid_currencies_and_date():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    service = FXRateService()
    result = service.query(ccy1, ccy2, asof)
    assert result is None or isinstance(result, FXRate)

def test_query_with_same_currencies_returns_one():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    service = FXRateService()
    result = service.query(ccy1, ccy1, asof)
    assert result == FXRate.one(ccy1, ccy1, asof)

def test_query_with_strict_mode_raises_error_when_rate_not_found():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    service = FXRateService()
    try:
        service.query(ccy1, ccy2, asof, strict=True)
        assert False, "Expected LookupError"
    except LookupError:
        pass

def test_query_with_invalid_currency_raises_error():
    ccy1 = "USD"
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    service = FXRateService()
    try:
        service.query(ccy1, ccy2, asof)
        assert False, "Expected TypeError"
    except TypeError:
        pass

def test_query_with_invalid_date_raises_error():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = "2023-01-01"
    service = FXRateService()
    try:
        service.query(ccy1, ccy2, asof)
        assert False, "Expected TypeError"
    except TypeError:
        pass


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


# LLM-generated content at query #32
#--------------------------

```python
def test_queries_returns_correct_fx_rates():
    ccy1 = Currency("USD")
    ccy2 = Currency("EUR")
    asof = Date(2023, 10, 1)
    queries = [(ccy1, ccy2, asof)]
    expected_rates = [Decimal("0.85")]
    service = FXRateService()
    rates = service.queries(queries)
    assert list(rates) == expected_rates

def test_queries_handles_empty_queries():
    queries = []
    service = FXRateService()
    rates = service.queries(queries)
    assert list(rates) == []

def test_queries_strict_mode_raises_error_on_missing_rate():
    ccy1 = Currency("USD")
    ccy2 = Currency("EUR")
    asof = Date(2023, 10, 1)
    queries = [(ccy1, ccy2, asof)]
    service = FXRateService()
    with pytest.raises(LookupError):
        service.queries(queries, strict=True)

def test_queries_non_strict_mode_returns_none_for_missing_rate():
    ccy1 = Currency("USD")
    ccy2 = Currency("EUR")
    asof = Date(2023, 10, 1)
    queries = [(ccy1, ccy2, asof)]
    service = FXRateService()
    rates = service.queries(queries)
    assert list(rates) == [None]


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


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_query_method():
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            return FXRate(Decimal("1.5"), ccy1, ccy2, asof)

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            pass

    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 10, 1)

    service = MockFXRateService()
    rate = service.query(usd, eur, asof)

    assert rate.value == Decimal("1.5")
    assert rate.ccy1 == usd
    assert rate.ccy2 == eur
    assert rate.asof == asof


# LLM-generated content at query #2
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


# LLM-generated content at query #3
#--------------------------

```python
def test_queries_returns_correct_rates():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if ccy1 == Currency("USD") and ccy2 == Currency("EUR") and asof == Date("2023-01-01"):
                return FXRate(Decimal("0.85"))
            if ccy1 == Currency("GBP") and ccy2 == Currency("JPY") and asof == Date("2023-01-01"):
                return FXRate(Decimal("150.0"))
            return None

        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = MockFXRateService()
    queries = [
        (Currency("USD"), Currency("EUR"), Date("2023-01-01")),
        (Currency("GBP"), Currency("JPY"), Date("2023-01-01")),
        (Currency("USD"), Currency("JPY"), Date("2023-01-01"))
    ]
    results = service.queries(queries)
    assert results == [FXRate(Decimal("0.85")), FXRate(Decimal("150.0")), None]

def test_queries_raises_error_in_strict_mode():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if strict:
                raise LookupError("FX rate not found")
            return None

        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = MockFXRateService()
    queries = [
        (Currency("USD"), Currency("EUR"), Date("2023-01-01")),
        (Currency("GBP"), Currency("JPY"), Date("2023-01-01"))
    ]
    try:
        service.queries(queries, strict=True)
        assert False, "Expected LookupError"
    except LookupError:
        pass


# LLM-generated content at query #4
#--------------------------

def test_query_returns_fx_rate_when_found():
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
    assert result == FXRate.of(usd, eur, date, Decimal("1.5"))


def test_query_returns_none_when_not_found_and_not_strict():
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


def test_query_raises_error_when_not_found_and_strict():
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if strict:
                raise LookupError("FX rate not found")
            return None
        
        def queries(self, queries, strict=False):
            if strict:
                raise LookupError("FX rate not found")
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


# LLM-generated content at query #5
#--------------------------

```python
def test_queries_returns_iterable_of_rates():
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            return FXRate(Decimal('1.5')) if ccy1 == Currency('USD') and ccy2 == Currency('EUR') else None
        
        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    service = MockFXRateService()
    queries = [(Currency('USD'), Currency('EUR'), Date(2023, 1, 1)), (Currency('GBP'), Currency('JPY'), Date(2023, 1, 1))]
    results = list(service.queries(queries))
    assert results == [FXRate(Decimal('1.5')), None]

def test_queries_raises_lookup_error_in_strict_mode():
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            if strict and ccy1 == Currency('GBP') and ccy2 == Currency('JPY'):
                raise LookupError("FX rate not found")
            return FXRate(Decimal('1.5')) if ccy1 == Currency('USD') and ccy2 == Currency('EUR') else None
        
        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    service = MockFXRateService()
    queries = [(Currency('USD'), Currency('EUR'), Date(2023, 1, 1)), (Currency('GBP'), Currency('JPY'), Date(2023, 1, 1))]
    try:
        list(service.queries(queries, strict=True))
    except LookupError as e:
        assert str(e) == "FX rate not found"


# LLM-generated content at query #6
#--------------------------

```python
def test_fxrate_constructor():
    ccy1 = Currency(code="EUR")
    ccy2 = Currency(code="USD")
    date = datetime.date(2023, 10, 1)
    value = Decimal("1.5")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value


# LLM-generated content at query #7
#--------------------------

```python
def test_fxrateservice_query_with_valid_currencies():
    ccy1 = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date.today()
    service = FXRateService()
    result = service.query(ccy1, ccy2, asof)
    assert result is None or isinstance(result, FXRate)

def test_fxrateservice_query_with_same_currencies():
    ccy1 = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    asof = Date.today()
    service = FXRateService()
    result = service.query(ccy1, ccy1, asof)
    assert result == FXRate.one(ccy1)

def test_fxrateservice_query_with_invalid_currencies():
    ccy1 = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("XXX", "Invalid", 2, CurrencyType.MONEY)
    asof = Date.today()
    service = FXRateService()
    result = service.query(ccy1, ccy2, asof, strict=False)
    assert result is None

def test_fxrateservice_query_strict_mode_with_invalid_currencies():
    ccy1 = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("XXX", "Invalid", 2, CurrencyType.MONEY)
    asof = Date.today()
    service = FXRateService()
    try:
        service.query(ccy1, ccy2, asof, strict=True)
        assert False, "Should raise LookupError"
    except LookupError:
        pass

def test_fxrateservice_query_with_none_parameters():
    service = FXRateService()
    try:
        service.query(None, None, None)
        assert False, "Should raise TypeError"
    except TypeError:
        pass


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


# LLM-generated content at query #9
#--------------------------

```python
def test_fxrateservice_query_with_valid_currencies_and_date():
    ccy1 = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    service = FXRateService()
    result = service.query(ccy1, ccy2, asof)
    assert result is None or isinstance(result, Decimal)

def test_fxrateservice_query_with_same_currencies():
    ccy1 = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    service = FXRateService()
    result = service.query(ccy1, ccy1, asof)
    assert result == Decimal(1)

def test_fxrateservice_query_with_invalid_currencies():
    ccy1 = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("XYZ", "Invalid Currency", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    service = FXRateService()
    result = service.query(ccy1, ccy2, asof)
    assert result is None

def test_fxrateservice_query_with_strict_mode():
    ccy1 = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("XYZ", "Invalid Currency", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    service = FXRateService()
    try:
        service.query(ccy1, ccy2, asof, strict=True)
        assert False
    except LookupError:
        assert True

def test_fxrateservice_query_with_none_currency():
    ccy1 = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    service = FXRateService()
    try:
        service.query(ccy1, None, asof)
        assert False
    except TypeError:
        assert True

def test_fxrateservice_query_with_none_date():
    ccy1 = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    service = FXRateService()
    try:
        service.query(ccy1, ccy2, None)
        assert False
    except TypeError:
        assert True


# LLM-generated content at query #10
#--------------------------

```python
def test_queries_returns_correct_rates():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return Decimal("1.5") if ccy1 == Currency("USD") and ccy2 == Currency("EUR") else None

        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = MockFXRateService()
    queries = [(Currency("USD"), Currency("EUR"), Date("2023-01-01")), (Currency("GBP"), Currency("JPY"), Date("2023-01-01"))]
    results = service.queries(queries)
    assert results == [Decimal("1.5"), None]

def test_queries_strict_mode_raises_error():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if ccy1 == Currency("USD") and ccy2 == Currency("EUR"):
                return Decimal("1.5")
            if strict:
                raise LookupError("Rate not found")
            return None

        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = MockFXRateService()
    queries = [(Currency("USD"), Currency("EUR"), Date("2023-01-01")), (Currency("GBP"), Currency("JPY"), Date("2023-01-01"))]
    try:
        service.queries(queries, strict=True)
        assert False, "Expected LookupError"
    except LookupError:
        pass


# LLM-generated content at query #11
#--------------------------

```python
def test_queries_returns_correct_rates():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return Decimal("1.2") if asof == Date(2023, 10, 1) else None

        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = MockFXRateService()
    queries = [(Currency("USD"), Currency("EUR"), Date(2023, 10, 1)), (Currency("GBP"), Currency("JPY"), Date(2023, 10, 2))]
    rates = service.queries(queries)
    assert rates == [Decimal("1.2"), None]

def test_queries_raises_lookup_error_in_strict_mode():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if asof == Date(2023, 10, 1):
                return Decimal("1.2")
            if strict:
                raise LookupError("Rate not found")
            return None

        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = MockFXRateService()
    queries = [(Currency("USD"), Currency("EUR"), Date(2023, 10, 1)), (Currency("GBP"), Currency("JPY"), Date(2023, 10, 2))]
    try:
        service.queries(queries, strict=True)
    except LookupError:
        pass
    else:
        assert False, "Expected LookupError to be raised"

def test_queries_returns_empty_list_for_empty_input():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return Decimal("1.2")

        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = MockFXRateService()
    rates = service.queries([])
    assert rates == []


# LLM-generated content at query #12
#--------------------------

```python
def test_fxrate_constructor():
    ccy1 = Currency("EUR", 2)
    ccy2 = Currency("USD", 2)
    date = datetime.date(2023, 10, 1)
    value = Decimal("2")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value

def test_fxrate_constructor_with_invalid_ccy1():
    ccy1 = "EUR"
    ccy2 = Currency("USD", 2)
    date = datetime.date(2023, 10, 1)
    value = Decimal("2")
    try:
        FXRate(ccy1, ccy2, date, value)
        assert False
    except TypeError:
        assert True

def test_fxrate_constructor_with_invalid_ccy2():
    ccy1 = Currency("EUR", 2)
    ccy2 = "USD"
    date = datetime.date(2023, 10, 1)
    value = Decimal("2")
    try:
        FXRate(ccy1, ccy2, date, value)
        assert False
    except TypeError:
        assert True

def test_fxrate_constructor_with_invalid_date():
    ccy1 = Currency("EUR", 2)
    ccy2 = Currency("USD", 2)
    date = "2023-10-01"
    value = Decimal("2")
    try:
        FXRate(ccy1, ccy2, date, value)
        assert False
    except TypeError:
        assert True

def test_fxrate_constructor_with_invalid_value():
    ccy1 = Currency("EUR", 2)
    ccy2 = Currency("USD", 2)
    date = datetime.date(2023, 10, 1)
    value = "2"
    try:
        FXRate(ccy1, ccy2, date, value)
        assert False
    except TypeError:
        assert True

def test_fxrate_constructor_with_zero_value():
    ccy1 = Currency("EUR", 2)
    ccy2 = Currency("USD", 2)
    date = datetime.date(2023, 10, 1)
    value = Decimal("0")
    try:
        FXRate(ccy1, ccy2, date, value)
        assert False
    except ValueError:
        assert True

def test_fxrate_constructor_with_negative_value():
    ccy1 = Currency("EUR", 2)
    ccy2 = Currency("USD", 2)
    date = datetime.date(2023, 10, 1)
    value = Decimal("-2")
    try:
        FXRate(ccy1, ccy2, date, value)
        assert False
    except ValueError:
        assert True

def test_fxrate_constructor_with_same_currency_and_invalid_value():
    ccy1 = Currency("EUR", 2)
    ccy2 = Currency("EUR", 2)
    date = datetime.date(2023, 10, 1)
    value = Decimal("2")
    try:
        FXRate(ccy1, ccy2, date, value)
        assert False
    except ValueError:
        assert True


# LLM-generated content at query #13
#--------------------------

```python
def test_fxrate_constructor():
    ccy1 = Currency("EUR")
    ccy2 = Currency("USD")
    date = Date(2023, 10, 1)
    value = Decimal("1.2")
    fxrate = FXRate(ccy1, ccy2, date, value)
    assert fxrate.ccy1 == ccy1
    assert fxrate.ccy2 == ccy2
    assert fxrate.date == date
    assert fxrate.value == value


# LLM-generated content at query #14
#--------------------------

```
def test_FXRate_constructor_with_valid_arguments():
    ccy1 = Currency("EUR", 978, 2)
    ccy2 = Currency("USD", 840, 2)
    date = Date(2023, 1, 1)
    value = Decimal("1.5")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value

def test_FXRate_constructor_with_same_currency_and_value_one():
    ccy1 = Currency("EUR", 978, 2)
    ccy2 = Currency("EUR", 978, 2)
    date = Date(2023, 1, 1)
    value = Decimal("1")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value

def test_FXRate_constructor_with_same_currency_and_value_not_one():
    ccy1 = Currency("EUR", 978, 2)
    ccy2 = Currency("EUR", 978, 2)
    date = Date(2023, 1, 1)
    value = Decimal("2")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value

def test_FXRate_constructor_with_zero_value():
    ccy1 = Currency("EUR", 978, 2)
    ccy2 = Currency("USD", 840, 2)
    date = Date(2023, 1, 1)
    value = Decimal("0")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value

def test_FXRate_constructor_with_negative_value():
    ccy1 = Currency("EUR", 978, 2)
    ccy2 = Currency("USD", 840, 2)
    date = Date(2023, 1, 1)
    value = Decimal("-1.5")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value


# LLM-generated content at query #15
#--------------------------

```python
def test_fxrate_constructor():
    ccy1 = Currency("EUR")
    ccy2 = Currency("USD")
    date = Date(2023, 10, 1)
    value = Decimal("1.2")
    fx_rate = FXRate(ccy1, ccy2, date, value)
    assert fx_rate.ccy1 == ccy1
    assert fx_rate.ccy2 == ccy2
    assert fx_rate.date == date
    assert fx_rate.value == value


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


# LLM-generated content at query #17
#--------------------------

```python
def test_queries_returns_correct_rates():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return FXRate(Decimal('1.5')) if (ccy1, ccy2, asof) == ('USD', 'EUR', '2023-10-01') else None

        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = MockFXRateService()
    queries = [('USD', 'EUR', '2023-10-01'), ('GBP', 'JPY', '2023-10-02')]
    result = service.queries(queries)
    assert result == [FXRate(Decimal('1.5')), None]

def test_queries_raises_error_when_strict():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if (ccy1, ccy2, asof) != ('USD', 'EUR', '2023-10-01'):
                raise LookupError("Rate not found")
            return FXRate(Decimal('1.5'))

        def queries(self, queries, strict=False):
            if strict:
                for ccy1, ccy2, asof in queries:
                    if self.query(ccy1, ccy2, asof, strict) is None:
                        raise LookupError("Rate not found")
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = MockFXRateService()
    queries = [('USD', 'EUR', '2023-10-01'), ('GBP', 'JPY', '2023-10-02')]
    try:
        service.queries(queries, strict=True)
        assert False, "Expected LookupError"
    except LookupError:
        pass


# LLM-generated content at query #18
#--------------------------

```python
def test_query_method_returns_fxrate_or_none():
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            return FXRate(ccy1, ccy2, asof, Decimal("1.0")) if ccy1.code == "USD" and ccy2.code == "EUR" else None

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            raise NotImplementedError

    service = MockFXRateService()
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    date = Date(2023, 10, 1)

    fx_rate = service.query(usd, eur, date)
    assert fx_rate is not None
    assert fx_rate.ccy1 == usd
    assert fx_rate.ccy2 == eur
    assert fx_rate.date == date
    assert fx_rate.rate == Decimal("1.0")

    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    fx_rate = service.query(usd, jpy, date)
    assert fx_rate is None


# LLM-generated content at query #19
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


# LLM-generated content at query #20
#--------------------------

```
def test_FXRate_constructor_creates_instance_with_correct_attributes():
    ccy1 = Currency("EUR", 2)
    ccy2 = Currency("USD", 2)
    date = Date(2023, 1, 1)
    value = Decimal("1.5")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value

def test_FXRate_constructor_allows_tuple_unpacking():
    ccy1 = Currency("EUR", 2)
    ccy2 = Currency("USD", 2)
    date = Date(2023, 1, 1)
    value = Decimal("1.5")
    rate = FXRate(ccy1, ccy2, date, value)
    unpacked_ccy1, unpacked_ccy2, unpacked_date, unpacked_value = rate
    assert unpacked_ccy1 == ccy1
    assert unpacked_ccy2 == ccy2
    assert unpacked_date == date
    assert unpacked_value == value

def test_FXRate_constructor_creates_invertible_instance():
    ccy1 = Currency("EUR", 2)
    ccy2 = Currency("USD", 2)
    date = Date(2023, 1, 1)
    value = Decimal("1.5")
    rate = FXRate(ccy1, ccy2, date, value)
    inverted = ~rate
    assert inverted.ccy1 == ccy2
    assert inverted.ccy2 == ccy1
    assert inverted.date == date
    assert inverted.value == value ** -1


# LLM-generated content at query #21
#--------------------------

```python
def test_queries_returns_iterable_of_fxrates():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return Decimal('1.5') if ccy1 == Currency('USD') and ccy2 == Currency('EUR') else None

        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = MockFXRateService()
    queries = [(Currency('USD'), Currency('EUR'), Date('2023-01-01')), (Currency('GBP'), Currency('JPY'), Date('2023-01-01'))]
    results = list(service.queries(queries))
    assert len(results) == 2
    assert results[0] == Decimal('1.5')
    assert results[1] is None

def test_queries_strict_mode_raises_error():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if strict and ccy1 == Currency('GBP'):
                raise LookupError('Rate not found')
            return Decimal('1.5') if ccy1 == Currency('USD') else None

        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = MockFXRateService()
    queries = [(Currency('USD'), Currency('EUR'), Date('2023-01-01')), (Currency('GBP'), Currency('JPY'), Date('2023-01-01'))]
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
    assert len(results) == 0


# LLM-generated content at query #22
#--------------------------

```python
def test_query_returns_fx_rate():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 10, 1)
    fx_rate = FXRate(ccy1, ccy2, asof, Decimal("0.85"))
    service = FXRateService()
    service.query = lambda c1, c2, a, strict: fx_rate
    result = service.query(ccy1, ccy2, asof)
    assert result == fx_rate

def test_query_returns_none_when_rate_not_found():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 10, 1)
    service = FXRateService()
    service.query = lambda c1, c2, a, strict: None
    result = service.query(ccy1, ccy2, asof)
    assert result is None

def test_query_raises_error_in_strict_mode_when_rate_not_found():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 10, 1)
    service = FXRateService()
    service.query = lambda c1, c2, a, strict: LookupError("FX rate not found") if strict else None
    raised_error = False
    try:
        service.query(ccy1, ccy2, asof, strict=True)
    except LookupError:
        raised_error = True
    assert raised_error


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


# LLM-generated content at query #24
#--------------------------

```python
def test_FXRate_constructor():
    from datetime import date
    from decimal import Decimal
    from pypara.currencies import Currencies

    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date_val = date.today()
    value = Decimal("2")

    rate = FXRate(ccy1, ccy2, date_val, value)

    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date_val
    assert rate.value == value


# LLM-generated content at query #25
#--------------------------

```python
def test_queries_returns_correct_rates():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return Decimal("1.0") if ccy1 == Currency.USD and ccy2 == Currency.EUR else None
        
        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    service = MockFXRateService()
    queries = [(Currency.USD, Currency.EUR, Date.today()), (Currency.EUR, Currency.USD, Date.today())]
    rates = service.queries(queries)
    assert rates == [Decimal("1.0"), None]

def test_queries_raises_error_when_strict():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if strict and ccy1 != Currency.USD:
                raise LookupError("Rate not found")
            return Decimal("1.0") if ccy1 == Currency.USD and ccy2 == Currency.EUR else None
        
        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    
    service = MockFXRateService()
    queries = [(Currency.EUR, Currency.USD, Date.today()), (Currency.USD, Currency.EUR, Date.today())]
    try:
        service.queries(queries, strict=True)
        assert False, "Expected LookupError"
    except LookupError:
        assert True


# LLM-generated content at query #26
#--------------------------

```python
def test_query_method_of_fxrateservice():
    class MockFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            return FXRate.of(ccy1, ccy2, asof, Decimal("1.0")) if ccy1.code == "USD" and ccy2.code == "EUR" else None

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            raise NotImplementedError

    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    asof = Date.today()

    service = MockFXRateService()
    rate = service.query(usd, eur, asof)

    assert rate is not None
    assert rate.ccy1 == usd
    assert rate.ccy2 == eur
    assert rate.asof == asof
    assert rate.value == Decimal("1.0")

    assert service.query(usd, jpy, asof) is None


# LLM-generated content at query #27
#--------------------------

```
def test_FXRate_constructor_creates_valid_instance():
    ccy1 = Currency("EUR", 2)
    ccy2 = Currency("USD", 2)
    date = Date(2023, 1, 1)
    value = Decimal("1.5")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value

def test_FXRate_constructor_allows_tuple_unpacking():
    ccy1 = Currency("EUR", 2)
    ccy2 = Currency("USD", 2)
    date = Date(2023, 1, 1)
    value = Decimal("1.5")
    rate = FXRate(ccy1, ccy2, date, value)
    unpacked_ccy1, unpacked_ccy2, unpacked_date, unpacked_value = rate
    assert unpacked_ccy1 == ccy1
    assert unpacked_ccy2 == ccy2
    assert unpacked_date == date
    assert unpacked_value == value

def test_FXRate_constructor_creates_invertible_instance():
    ccy1 = Currency("EUR", 2)
    ccy2 = Currency("USD", 2)
    date = Date(2023, 1, 1)
    value = Decimal("1.5")
    rate = FXRate(ccy1, ccy2, date, value)
    inverted = ~rate
    assert inverted.ccy1 == ccy2
    assert inverted.ccy2 == ccy1
    assert inverted.date == date
    assert inverted.value == Decimal("1") / value


# LLM-generated content at query #28
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


# LLM-generated content at query #29
#--------------------------

```python
def test_queries_returns_correct_rates():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return FXRate(Decimal('1.0'))

        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = MockFXRateService()
    queries = [(Currency('USD'), Currency('EUR'), Date(2023, 10, 1))]
    result = service.queries(queries)
    assert result == [FXRate(Decimal('1.0'))]

def test_queries_returns_none_for_invalid_query():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None

        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = MockFXRateService()
    queries = [(Currency('USD'), Currency('EUR'), Date(2023, 10, 1))]
    result = service.queries(queries)
    assert result == [None]

def test_queries_raises_error_when_strict_is_true():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if strict:
                raise LookupError("Rate not found")
            return None

        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = MockFXRateService()
    queries = [(Currency('USD'), Currency('EUR'), Date(2023, 10, 1))]
    try:
        service.queries(queries, strict=True)
    except LookupError:
        pass
    else:
        assert False, "Expected LookupError to be raised"


# LLM-generated content at query #30
#--------------------------

def test_query_with_valid_currencies_and_date():
    ccy1 = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    service = FXRateService()
    result = service.query(ccy1, ccy2, asof)
    assert result is None or isinstance(result, FXRate)

def test_query_with_same_currencies_returns_one():
    ccy1 = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    service = FXRateService()
    result = service.query(ccy1, ccy1, asof)
    assert result == FXRate.one(ccy1, ccy1)

def test_query_with_strict_mode_raises_error_when_rate_not_found():
    ccy1 = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    service = FXRateService()
    try:
        service.query(ccy1, ccy2, asof, strict=True)
        assert False, "Expected LookupError"
    except LookupError:
        pass

def test_query_with_invalid_currency_raises_error():
    ccy1 = "USD"
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    service = FXRateService()
    try:
        service.query(ccy1, ccy2, asof)
        assert False, "Expected TypeError"
    except TypeError:
        pass

def test_query_with_invalid_date_raises_error():
    ccy1 = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = "2023-01-01"
    service = FXRateService()
    try:
        service.query(ccy1, ccy2, asof)
        assert False, "Expected TypeError"
    except TypeError:
        pass


# LLM-generated content at query #31
#--------------------------

```python
def test_fxrate_constructor():
    ccy1 = Currency("EUR")
    ccy2 = Currency("USD")
    date = Date(2023, 10, 1)
    value = Decimal("2.0")
    fx_rate = FXRate(ccy1, ccy2, date, value)
    assert fx_rate.ccy1 == ccy1
    assert fx_rate.ccy2 == ccy2
    assert fx_rate.date == date
    assert fx_rate.value == value

def test_fxrate_constructor_with_same_currency():
    ccy1 = Currency("EUR")
    ccy2 = Currency("EUR")
    date = Date(2023, 10, 1)
    value = Decimal("1.0")
    fx_rate = FXRate(ccy1, ccy2, date, value)
    assert fx_rate.ccy1 == ccy1
    assert fx_rate.ccy2 == ccy2
    assert fx_rate.date == date
    assert fx_rate.value == value


# LLM-generated content at query #32
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


