####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
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
    assert inverted_rate == FXRate(Currencies["USD"], Currencies["EUR"], datetime.date.today(), Decimal("0.5"))


# LLM-generated content at query #2
#--------------------------

```python
def test_fxrate_invert():
    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date(2023, 1, 1), Decimal("2"))
    inverted_rate = ~rate
    assert inverted_rate.ccy1 == Currencies["USD"]
    assert inverted_rate.ccy2 == Currencies["EUR"]
    assert inverted_rate.date == datetime.date(2023, 1, 1)
    assert inverted_rate.value == Decimal("0.5")


# LLM-generated content at query #3
#--------------------------

```python
def test_query_with_valid_currencies_and_date():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    result = FXRateService.default.query(ccy1, ccy2, asof)
    assert isinstance(result, FXRate) or result is None

def test_query_with_invalid_currencies():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = "EUR"
    asof = Date(2023, 1, 1)
    with pytest.raises(TypeError):
        FXRateService.default.query(ccy1, ccy2, asof)

def test_query_with_strict_true_and_missing_rate():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("XYZ", "Unknown", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    with pytest.raises(LookupError):
        FXRateService.default.query(ccy1, ccy2, asof, strict=True)

def test_query_with_strict_false_and_missing_rate():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("XYZ", "Unknown", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    result = FXRateService.default.query(ccy1, ccy2, asof, strict=False)
    assert result is None


# LLM-generated content at query #4
#--------------------------

```python
def test_invert_fx_rate():
    from decimal import Decimal
    import datetime
    from pypara.currencies import Currencies
    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date(2023, 1, 1), Decimal("2"))
    inverted_rate = ~rate
    assert inverted_rate == FXRate(Currencies["USD"], Currencies["EUR"], datetime.date(2023, 1, 1), Decimal("0.5"))


# LLM-generated content at query #5
#--------------------------

```python
def test_query_returns_fxrate_for_valid_currency_pair_and_date():
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return FXRate(ccy1, ccy2, Decimal("1.5"), asof)

        def queries(self, queries, strict=False):
            pass

    service = TestFXRateService()
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    today = Date.today()
    result = service.query(usd, eur, today)
    assert result == FXRate(usd, eur, Decimal("1.5"), today)

def test_query_returns_none_for_invalid_currency_pair():
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None

        def queries(self, queries, strict=False):
            pass

    service = TestFXRateService()
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    xyz = Currency.of("XYZ", "Unknown Currency", 2, CurrencyType.MONEY)
    today = Date.today()
    result = service.query(usd, xyz, today)
    assert result is None

def test_query_raises_error_in_strict_mode_for_invalid_currency_pair():
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if strict:
                raise LookupError("FX rate not found")
            return None

        def queries(self, queries, strict=False):
            pass

    service = TestFXRateService()
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    xyz = Currency.of("XYZ", "Unknown Currency", 2, CurrencyType.MONEY)
    today = Date.today()

    with pytest.raises(LookupError):
        service.query(usd, xyz, today, strict=True)


# LLM-generated content at query #6
#--------------------------

```python
def test_queries_empty_input():
    service = FXRateService.default
    assert list(service.queries([])) == []

def test_queries_single_query():
    service = FXRateService.default
    queries = [(Currency("USD"), Currency("EUR"), Date(2023, 1, 1))]
    rates = list(service.queries(queries))
    assert len(rates) == 1
    assert isinstance(rates[0], (FXRate, type(None)))

def test_queries_multiple_queries():
    service = FXRateService.default
    queries = [
        (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)),
        (Currency("GBP"), Currency("JPY"), Date(2023, 1, 2)),
    ]
    rates = list(service.queries(queries))
    assert len(rates) == 2
    assert all(isinstance(r, (FXRate, type(None))) for r in rates)

def test_queries_strict_mode_raises_error():
    service = FXRateService.default
    queries = [(Currency("XYZ"), Currency("ABC"), Date(2023, 1, 1))]
    try:
        list(service.queries(queries, strict=True))
        assert False, "Expected a lookup error"
    except LookupError:
        pass


# LLM-generated content at query #7
#--------------------------

```python
def test_query_with_valid_currencies_and_date():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    fx_rate = Decimal("0.92")
    mock_service = create_autospec(FXRateService)
    mock_service.query.return_value = FXRate(ccy1, ccy2, asof, fx_rate)
    result = mock_service.query(ccy1, ccy2, asof)
    assert result == FXRate(ccy1, ccy2, asof, fx_rate)

def test_query_with_invalid_currencies():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    mock_service = create_autospec(FXRateService)
    mock_service.query.return_value = None
    result = mock_service.query(ccy1, ccy2, asof)
    assert result is None

def test_query_with_strict_flag_raises_error():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    mock_service = create_autospec(FXRateService)
    mock_service.query.side_effect = LookupError("FX rate not found")
    with pytest.raises(LookupError):
        mock_service.query(ccy1, ccy2, asof, strict=True)


# LLM-generated content at query #8
#--------------------------

```python
def test_queries_with_valid_inputs():
    service = FXRateService.default
    queries = [(Currency("USD"), Currency("EUR"), Date(2023, 1, 1))]
    result = service.queries(queries)
    assert isinstance(result, Iterable)
    assert len(list(result)) == 1

def test_queries_with_empty_input():
    service = FXRateService.default
    queries = []
    result = service.queries(queries)
    assert list(result) == []

def test_queries_with_strict_flag():
    service = FXRateService.default
    queries = [(Currency("USD"), Currency("EUR"), Date(2023, 1, 1))]
    result = service.queries(queries, strict=True)
    assert isinstance(result, Iterable)
    assert len(list(result)) == 1


# LLM-generated content at query #9
#--------------------------

```python
def test_fxrate_constructor_creates_instance_with_given_values():
    from decimal import Decimal
    import datetime
    from pypara.currencies import Currencies

    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date(2023, 1, 1)
    value = Decimal("1.2345")

    rate = FXRate(ccy1, ccy2, date, value)

    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value


# LLM-generated content at query #10
#--------------------------

```python
def test_queries_returns_iterable_of_rates():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return 1.2345 if (ccy1, ccy2, asof) == ("USD", "EUR", "2023-01-01") else None

        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = MockFXRateService()
    queries = [("USD", "EUR", "2023-01-01"), ("GBP", "JPY", "2023-01-01")]
    result = service.queries(queries)
    assert list(result) == [1.2345, None]

def test_queries_with_strict_raises_error():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if strict and (ccy1, ccy2, asof) != ("USD", "EUR", "2023-01-01"):
                raise LookupError("FX rate not found")
            return 1.2345 if (ccy1, ccy2, asof) == ("USD", "EUR", "2023-01-01") else None

        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = MockFXRateService()
    queries = [("USD", "EUR", "2023-01-01"), ("GBP", "JPY", "2023-01-01")]
    try:
        service.queries(queries, strict=True)
        assert False, "Expected LookupError"
    except LookupError:
        pass

def test_queries_empty_input():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None

        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = MockFXRateService()
    result = service.queries([])
    assert list(result) == []


# LLM-generated content at query #11
#--------------------------

```python
def test_fxrate_constructor_creates_instance_with_given_values():
    from pypara.currencies import Currencies
    import datetime
    from decimal import Decimal

    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date(2023, 1, 1), Decimal("1.2345"))

    assert rate.ccy1 == Currencies["EUR"]
    assert rate.ccy2 == Currencies["USD"]
    assert rate.date == datetime.date(2023, 1, 1)
    assert rate.value == Decimal("1.2345")


# LLM-generated content at query #12
#--------------------------

```python
def test_query_with_valid_currencies_and_date():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    result = FXRateService.default.query(ccy1, ccy2, asof)
    assert isinstance(result, (FXRate, type(None)))

def test_query_with_invalid_currencies():
    ccy1 = "USD"
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    with pytest.raises(TypeError):
        FXRateService.default.query(ccy1, ccy2, asof)

def test_query_with_strict_true_and_missing_rate():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("XYZ", "Unknown", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    with pytest.raises(LookupError):
        FXRateService.default.query(ccy1, ccy2, asof, strict=True)

def test_query_with_strict_false_and_missing_rate():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("XYZ", "Unknown", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    result = FXRateService.default.query(ccy1, ccy2, asof, strict=False)
    assert result is None


# LLM-generated content at query #13
#--------------------------

```python
def test_queries_empty_input():
    service = FXRateService.default
    assert list(service.queries([])) == []

def test_queries_single_query():
    service = FXRateService.default
    query = (Currency("USD"), Currency("EUR"), Date(2023, 1, 1))
    result = list(service.queries([query]))
    assert len(result) == 1
    assert isinstance(result[0], (FXRate, type(None)))

def test_queries_multiple_queries():
    service = FXRateService.default
    queries = [
        (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)),
        (Currency("GBP"), Currency("JPY"), Date(2023, 1, 2)),
    ]
    result = list(service.queries(queries))
    assert len(result) == 2
    assert all(isinstance(r, (FXRate, type(None))) for r in result)

def test_queries_strict_mode_raises_error():
    service = FXRateService.default
    queries = [(Currency("XYZ"), Currency("ABC"), Date(2023, 1, 1))]
    try:
        list(service.queries(queries, strict=True))
        assert False, "Expected LookupError"
    except LookupError:
        pass


# LLM-generated content at query #14
#--------------------------

```python
def test_fxrate_constructor_creates_instance_with_given_values():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies

    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("2"))

    assert rate[0] == Currencies["EUR"]
    assert rate[1] == Currencies["USD"]
    assert rate[2] == datetime.date.today()
    assert rate[3] == Decimal("2")


# LLM-generated content at query #15
#--------------------------

```python
def test_fxrate_constructor():
    from decimal import Decimal
    import datetime
    from pypara.currencies import Currencies
    from pypara.fx import FXRate

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
def test_fxrate_constructor_creates_instance_with_correct_properties():
    from decimal import Decimal
    import datetime
    from pypara.currencies import Currencies

    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date(2023, 1, 1), Decimal("1.5"))

    assert rate.ccy1 == Currencies["EUR"]
    assert rate.ccy2 == Currencies["USD"]
    assert rate.date == datetime.date(2023, 1, 1)
    assert rate.value == Decimal("1.5")


# LLM-generated content at query #17
#--------------------------

```python
def test_fxrate_constructor():
    from decimal import Decimal
    import datetime
    from pypara.currencies import Currencies
    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("2"))
    assert rate.ccy1 == Currencies["EUR"]
    assert rate.ccy2 == Currencies["USD"]
    assert rate.date == datetime.date.today()
    assert rate.value == Decimal("2")


# LLM-generated content at query #18
#--------------------------

```python
def test_fxrate_constructor_creates_valid_instance():
    from decimal import Decimal
    import datetime
    from pypara.currencies import Currencies

    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date(2023, 1, 1), Decimal("1.2"))
    assert rate.ccy1 == Currencies["EUR"]
    assert rate.ccy2 == Currencies["USD"]
    assert rate.date == datetime.date(2023, 1, 1)
    assert rate.value == Decimal("1.2")

def test_fxrate_constructor_allows_same_currency_with_one():
    from decimal import Decimal
    import datetime
    from pypara.currencies import Currencies

    rate = FXRate(Currencies["EUR"], Currencies["EUR"], datetime.date(2023, 1, 1), Decimal("1"))
    assert rate.ccy1 == Currencies["EUR"]
    assert rate.ccy2 == Currencies["EUR"]
    assert rate.date == datetime.date(2023, 1, 1)
    assert rate.value == Decimal("1")

def test_fxrate_constructor_allows_indexed_access():
    from decimal import Decimal
    import datetime
    from pypara.currencies import Currencies

    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date(2023, 1, 1), Decimal("1.2"))
    assert rate[0] == Currencies["EUR"]
    assert rate[1] == Currencies["USD"]
    assert rate[2] == datetime.date(2023, 1, 1)
    assert rate[3] == Decimal("1.2")


# LLM-generated content at query #19
#--------------------------

```python
def test_fxrate_constructor_creates_instance_with_correct_properties():
    from datetime import date
    from decimal import Decimal
    from pypara.currencies import Currencies
    from pypara.fx import FXRate

    rate = FXRate(Currencies["EUR"], Currencies["USD"], date(2023, 1, 1), Decimal("1.2345"))
    assert rate.ccy1 == Currencies["EUR"]
    assert rate.ccy2 == Currencies["USD"]
    assert rate.date == date(2023, 1, 1)
    assert rate.value == Decimal("1.2345")


# LLM-generated content at query #20
#--------------------------

```python
def test_query_with_valid_currencies_and_date():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    result = FXRateService.default.query(ccy1, ccy2, asof)
    assert isinstance(result, FXRate) or result is None

def test_query_with_invalid_currencies():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = "EUR"
    asof = Date(2023, 1, 1)
    with pytest.raises(TypeError):
        FXRateService.default.query(ccy1, ccy2, asof)

def test_query_with_invalid_date():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    asof = "2023-01-01"
    with pytest.raises(TypeError):
        FXRateService.default.query(ccy1, ccy2, asof)

def test_query_with_strict_true_and_missing_rate():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("XYZ", "Unknown", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    with pytest.raises(LookupError):
        FXRateService.default.query(ccy1, ccy2, asof, strict=True)

def test_query_with_strict_false_and_missing_rate():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("XYZ", "Unknown", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    result = FXRateService.default.query(ccy1, ccy2, asof, strict=False)
    assert result is None


# LLM-generated content at query #21
#--------------------------

```python
def test_queries_empty_input():
    service = FXRateService.default
    assert list(service.queries([])) == []

def test_queries_single_rate():
    service = FXRateService.default
    queries = [(Currency("USD"), Currency("EUR"), Date(2023, 1, 1))]
    rates = list(service.queries(queries))
    assert len(rates) == 1
    assert isinstance(rates[0], (FXRate, type(None)))

def test_queries_multiple_rates():
    service = FXRateService.default
    queries = [
        (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)),
        (Currency("GBP"), Currency("JPY"), Date(2023, 1, 2))
    ]
    rates = list(service.queries(queries))
    assert len(rates) == 2
    for rate in rates:
        assert isinstance(rate, (FXRate, type(None)))

def test_queries_strict_mode_raises_error():
    service = FXRateService.default
    queries = [(Currency("XYZ"), Currency("ABC"), Date(2023, 1, 1))]
    try:
        list(service.queries(queries, strict=True))
        assert False, "Expected lookup error not raised"
    except LookupError:
        pass


# LLM-generated content at query #22
#--------------------------

```python
def test_fxrate_constructor():
    from decimal import Decimal
    import datetime
    from pypara.currencies import Currencies

    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("2"))
    assert rate.ccy1 == Currencies["EUR"]
    assert rate.ccy2 == Currencies["USD"]
    assert rate.date == datetime.date.today()
    assert rate.value == Decimal("2")


# LLM-generated content at query #23
#--------------------------

```python
def test_fxrate_constructor_creates_instance_with_correct_properties():
    from pypara.currencies import Currencies
    import datetime
    from decimal import Decimal

    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date(2023, 1, 1), Decimal("1.2"))

    assert rate.ccy1 == Currencies["EUR"]
    assert rate.ccy2 == Currencies["USD"]
    assert rate.date == datetime.date(2023, 1, 1)
    assert rate.value == Decimal("1.2")


# LLM-generated content at query #24
#--------------------------

```python
def test_queries_empty_input():
    service = FXRateService.default
    assert list(service.queries([])) == []

def test_queries_single_rate():
    service = FXRateService.default
    query = (Currency("USD"), Currency("EUR"), Date(2023, 1, 1))
    result = list(service.queries([query]))
    assert len(result) == 1
    assert isinstance(result[0], (FXRate, type(None)))

def test_queries_multiple_rates():
    service = FXRateService.default
    queries = [
        (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)),
        (Currency("GBP"), Currency("JPY"), Date(2023, 1, 2))
    ]
    result = list(service.queries(queries))
    assert len(result) == 2
    assert all(isinstance(r, (FXRate, type(None))) for r in result)

def test_queries_strict_mode_raises_error():
    service = FXRateService.default
    queries = [(Currency("XYZ"), Currency("ABC"), Date(2023, 1, 1))]
    try:
        list(service.queries(queries, strict=True))
        assert False, "Expected a lookup error to be raised"
    except LookupError:
        pass


# LLM-generated content at query #25
#--------------------------

```python
def test_query_abstract_method():
    with pytest.raises(TypeError):
        FXRateService().query(Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY), Currency.of("EUR", "Euros", 2, CurrencyType.MONEY), Date(2023, 1, 1))


# LLM-generated content at query #26
#--------------------------

```python
def test_fxrate_constructor_creates_instance_with_correct_properties():
    from decimal import Decimal
    import datetime
    from pypara.currencies import Currencies

    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date(2023, 1, 1), Decimal("1.2"))

    assert rate.ccy1 == Currencies["EUR"]
    assert rate.ccy2 == Currencies["USD"]
    assert rate.date == datetime.date(2023, 1, 1)
    assert rate.value == Decimal("1.2")


# LLM-generated content at query #27
#--------------------------

```python
def test_fxrate_constructor_creates_instance_with_given_values():
    from decimal import Decimal
    import datetime
    from pypara.currencies import Currencies

    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date(2023, 1, 1), Decimal("1.2345"))

    assert rate.ccy1 == Currencies["EUR"]
    assert rate.ccy2 == Currencies["USD"]
    assert rate.date == datetime.date(2023, 1, 1)
    assert rate.value == Decimal("1.2345")


# LLM-generated content at query #28
#--------------------------

```python
def test_query_returns_fxrate_for_valid_currency_pair_and_date():
    # Mock setup (assuming appropriate mocking is done elsewhere)
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    expected_rate = Decimal("0.92")

    # Assuming a concrete implementation of FXRateService exists
    fx_service = ConcreteFXRateService()  # This would be a real implementation in practice
    result = fx_service.query(ccy1, ccy2, asof)

    assert result == expected_rate

def test_query_returns_none_for_invalid_currency_pair():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("XYZ", "Invalid Currency", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)

    fx_service = ConcreteFXRateService()
    result = fx_service.query(ccy1, ccy2, asof)

    assert result is None

def test_query_raises_error_in_strict_mode_for_missing_rate():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("XYZ", "Invalid Currency", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)

    fx_service = ConcreteFXRateService()

    with pytest.raises(FXRateNotFoundError):
        fx_service.query(ccy1, ccy2, asof, strict=True)


# LLM-generated content at query #29
#--------------------------

```python
def test_queries_with_empty_input():
    service = FXRateService.default
    assert list(service.queries([])) == []

def test_queries_with_single_query():
    service = FXRateService.default
    query = (Currency("USD"), Currency("EUR"), Date(2023, 1, 1))
    result = list(service.queries([query]))
    assert len(result) == 1
    assert isinstance(result[0], (FXRate, type(None)))

def test_queries_with_multiple_queries():
    service = FXRateService.default
    queries = [
        (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)),
        (Currency("GBP"), Currency("JPY"), Date(2023, 1, 2))
    ]
    result = list(service.queries(queries))
    assert len(result) == 2
    assert all(isinstance(r, (FXRate, type(None))) for r in result)

def test_queries_with_strict_false():
    service = FXRateService.default
    query = (Currency("XYZ"), Currency("ABC"), Date(2023, 1, 1))
    result = list(service.queries([query], strict=False))
    assert len(result) == 1
    assert result[0] is None

def test_queries_with_strict_true_raises_error():
    service = FXRateService.default
    query = (Currency("XYZ"), Currency("ABC"), Date(2023, 1, 1))
    try:
        list(service.queries([query], strict=True))
        assert False, "Expected an exception to be raised"
    except LookupError:
        pass


# LLM-generated content at query #30
#--------------------------

```python
def test_fxrate_constructor_creates_instance_with_correct_properties():
    from decimal import Decimal
    import datetime
    from pypara.currencies import Currencies

    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("2"))

    assert rate.ccy1 == Currencies["EUR"]
    assert rate.ccy2 == Currencies["USD"]
    assert rate.date == datetime.date.today()
    assert rate.value == Decimal("2")


# LLM-generated content at query #31
#--------------------------

```python
def test_fxrate_constructor_creates_instance_with_correct_properties():
    from decimal import Decimal
    import datetime
    from pypara.currencies import Currencies

    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("2"))

    assert rate.ccy1 == Currencies["EUR"]
    assert rate.ccy2 == Currencies["USD"]
    assert rate.date == datetime.date.today()
    assert rate.value == Decimal("2")


# LLM-generated content at query #32
#--------------------------

```python
def test_queries_empty_input():
    service = FXRateService.default
    result = service.queries([])
    assert list(result) == []

def test_queries_single_query():
    service = FXRateService.default
    query = (Currency("USD"), Currency("EUR"), Date(2023, 1, 1))
    result = service.queries([query])
    assert len(list(result)) == 1

def test_queries_multiple_queries():
    service = FXRateService.default
    queries = [
        (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)),
        (Currency("GBP"), Currency("JPY"), Date(2023, 1, 2))
    ]
    result = service.queries(queries)
    assert len(list(result)) == 2

def test_queries_strict_mode():
    service = FXRateService.default
    queries = [
        (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)),
        (Currency("XYZ"), Currency("ABC"), Date(2023, 1, 1))
    ]
    try:
        service.queries(queries, strict=True)
        assert False, "Expected lookup error not raised"
    except LookupError:
        pass

def test_queries_non_strict_mode():
    service = FXRateService.default
    queries = [
        (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)),
        (Currency("XYZ"), Currency("ABC"), Date(2023, 1, 1))
    ]
    result = service.queries(queries, strict=False)
    assert len(list(result)) == 2


# LLM-generated content at query #33
#--------------------------

```python
def test_query_method_with_valid_input():
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return Decimal("1.2345")

        def queries(self, queries, strict=False):
            return [Decimal("1.2345")]

    service = ConcreteFXRateService()
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    today = Date.today()
    result = service.query(usd, eur, today)
    assert result == Decimal("1.2345")

def test_query_method_with_strict_false_and_missing_rate():
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None

        def queries(self, queries, strict=False):
            return [None]

    service = ConcreteFXRateService()
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    today = Date.today()
    result = service.query(usd, eur, today, strict=False)
    assert result is None

def test_query_method_with_strict_true_and_missing_rate():
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if strict:
                raise LookupError("FX rate not found")
            return None

        def queries(self, queries, strict=False):
            if strict:
                raise LookupError("FX rate not found")
            return [None]

    service = ConcreteFXRateService()
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    today = Date.today()
    try:
        service.query(usd, eur, today, strict=True)
        assert False, "Expected LookupError"
    except LookupError:
        pass


# LLM-generated content at query #34
#--------------------------

```python
def test_fxrate_constructor_creates_instance_with_correct_properties():
    from decimal import Decimal
    import datetime
    from pypara.currencies import Currencies

    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("2"))

    assert rate.ccy1 == Currencies["EUR"]
    assert rate.ccy2 == Currencies["USD"]
    assert rate.date == datetime.date.today()
    assert rate.value == Decimal("2")


# LLM-generated content at query #35
#--------------------------

```python
def test_fxrate_constructor_creates_instance_with_correct_properties():
    from decimal import Decimal
    from pypara.currencies import Currencies
    import datetime

    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("2"))

    assert rate.ccy1 == Currencies["EUR"]
    assert rate.ccy2 == Currencies["USD"]
    assert rate.date == datetime.date.today()
    assert rate.value == Decimal("2")
    assert rate[0] == Currencies["EUR"]
    assert rate[1] == Currencies["USD"]
    assert rate[2] == datetime.date.today()
    assert rate[3] == Decimal("2")


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_invert_fx_rate():
    from decimal import Decimal
    from pypara.currencies import Currencies
    import datetime

    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("2"))
    inverted_rate = ~rate

    assert inverted_rate.ccy1 == Currencies["USD"]
    assert inverted_rate.ccy2 == Currencies["EUR"]
    assert inverted_rate.date == datetime.date.today()
    assert inverted_rate.value == Decimal("0.5")


# LLM-generated content at query #2
#--------------------------

```python
def test_invert_fx_rate():
    from decimal import Decimal
    from pypara.currencies import Currencies
    import datetime

    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("2"))
    inverted_rate = ~rate

    assert inverted_rate.ccy1 == Currencies["USD"]
    assert inverted_rate.ccy2 == Currencies["EUR"]
    assert inverted_rate.date == datetime.date.today()
    assert inverted_rate.value == Decimal("0.5")


# LLM-generated content at query #3
#--------------------------

```python
def test_queries_with_empty_input():
    service = FXRateService.default
    assert list(service.queries([])) == []

def test_queries_with_single_query():
    service = FXRateService.default
    query = (Currency("USD"), Currency("EUR"), Date(2023, 1, 1))
    result = list(service.queries([query]))
    assert len(result) == 1
    assert isinstance(result[0], (FXRate, type(None)))

def test_queries_with_multiple_queries():
    service = FXRateService.default
    queries = [
        (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)),
        (Currency("GBP"), Currency("JPY"), Date(2023, 1, 2)),
    ]
    result = list(service.queries(queries))
    assert len(result) == 2
    assert all(isinstance(r, (FXRate, type(None))) for r in result)

def test_queries_with_strict_false():
    service = FXRateService.default
    queries = [
        (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)),
        (Currency("XYZ"), Currency("ABC"), Date(2023, 1, 1)),
    ]
    result = list(service.queries(queries, strict=False))
    assert len(result) == 2
    assert all(isinstance(r, (FXRate, type(None))) for r in result)

def test_queries_with_strict_true():
    service = FXRateService.default
    queries = [
        (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)),
        (Currency("XYZ"), Currency("ABC"), Date(2023, 1, 1)),
    ]
    try:
        list(service.queries(queries, strict=True))
        assert False, "Expected a lookup error to be raised"
    except LookupError:
        pass


# LLM-generated content at query #4
#--------------------------

```python
def test_fxrate_constructor():
    from decimal import Decimal
    import datetime
    from pypara.currencies import Currencies

    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("2"))
    assert rate[0] == Currencies["EUR"]
    assert rate[1] == Currencies["USD"]
    assert rate[2] == datetime.date.today()
    assert rate[3] == Decimal("2")


# LLM-generated content at query #5
#--------------------------

```python
def test_invert_fx_rate():
    from decimal import Decimal
    import datetime
    from pypara.currencies import Currencies
    from pypara.fx import FXRate

    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date(2023, 1, 1), Decimal("2"))
    inverted_rate = ~rate

    assert inverted_rate.ccy1 == Currencies["USD"]
    assert inverted_rate.ccy2 == Currencies["EUR"]
    assert inverted_rate.date == datetime.date(2023, 1, 1)
    assert inverted_rate.value == Decimal("0.5")


# LLM-generated content at query #6
#--------------------------

```python
def test_queries_empty_input():
    service = FXRateService.default
    assert list(service.queries([])) == []

def test_queries_single_valid_query():
    service = FXRateService.default
    queries = [(Currency("USD"), Currency("EUR"), Date(2023, 1, 1))]
    rates = list(service.queries(queries))
    assert len(rates) == 1
    assert isinstance(rates[0], (FXRate, type(None)))

def test_queries_multiple_valid_queries():
    service = FXRateService.default
    queries = [
        (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)),
        (Currency("GBP"), Currency("JPY"), Date(2023, 1, 2))
    ]
    rates = list(service.queries(queries))
    assert len(rates) == 2
    assert all(isinstance(r, (FXRate, type(None))) for r in rates)

def test_queries_invalid_query_non_strict():
    service = FXRateService.default
    queries = [(Currency("XYZ"), Currency("ABC"), Date(2023, 1, 1))]
    rates = list(service.queries(queries, strict=False))
    assert len(rates) == 1
    assert rates[0] is None

def test_queries_mixed_valid_invalid_non_strict():
    service = FXRateService.default
    queries = [
        (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)),
        (Currency("XYZ"), Currency("ABC"), Date(2023, 1, 1))
    ]
    rates = list(service.queries(queries, strict=False))
    assert len(rates) == 2
    assert isinstance(rates[0], (FXRate, type(None)))
    assert rates[1] is None


# LLM-generated content at query #7
#--------------------------

```python
def test_query_with_valid_currencies_and_date():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    fx_rate = Decimal("0.92")
    mock_service = Mock()
    mock_service.query.return_value = FXRate(ccy1, ccy2, asof, fx_rate)
    result = mock_service.query(ccy1, ccy2, asof)
    assert result == FXRate(ccy1, ccy2, asof, fx_rate)

def test_query_with_invalid_currencies_and_date():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    mock_service = Mock()
    mock_service.query.return_value = None
    result = mock_service.query(ccy1, ccy2, asof)
    assert result is None

def test_query_with_strict_flag_and_missing_rate():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    mock_service = Mock()
    mock_service.query.side_effect = LookupError("Rate not found")
    with pytest.raises(LookupError):
        mock_service.query(ccy1, ccy2, asof, strict=True)


# LLM-generated content at query #8
#--------------------------

```python
def test_queries_empty_iterable():
    service = FXRateService.default
    assert list(service.queries([])) == []

def test_queries_single_valid_query():
    service = FXRateService.default
    queries = [(Currency("USD"), Currency("EUR"), Date(2023, 1, 1))]
    rates = list(service.queries(queries))
    assert len(rates) == 1
    assert isinstance(rates[0], (FXRate, type(None)))

def test_queries_multiple_valid_queries():
    service = FXRateService.default
    queries = [
        (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)),
        (Currency("GBP"), Currency("JPY"), Date(2023, 1, 2)),
    ]
    rates = list(service.queries(queries))
    assert len(rates) == 2
    assert all(isinstance(r, (FXRate, type(None))) for r in rates)

def test_queries_invalid_query_non_strict():
    service = FXRateService.default
    queries = [(Currency("XYZ"), Currency("ABC"), Date(2023, 1, 1))]
    rates = list(service.queries(queries, strict=False))
    assert len(rates) == 1
    assert rates[0] is None

def test_queries_invalid_query_strict():
    service = FXRateService.default
    queries = [(Currency("XYZ"), Currency("ABC"), Date(2023, 1, 1))]
    with pytest.raises(LookupError):
        list(service.queries(queries, strict=True))


# LLM-generated content at query #9
#--------------------------

```python
def test_queries_empty_input():
    service = FXRateService.default
    queries = []
    result = service.queries(queries)
    assert list(result) == []

def test_queries_single_query():
    service = FXRateService.default
    queries = [(Currency("USD"), Currency("EUR"), Date(2023, 1, 1))]
    result = service.queries(queries)
    assert len(list(result)) == 1

def test_queries_multiple_queries():
    service = FXRateService.default
    queries = [
        (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)),
        (Currency("GBP"), Currency("JPY"), Date(2023, 1, 2))
    ]
    result = service.queries(queries)
    assert len(list(result)) == 2

def test_queries_strict_mode_raises_error():
    service = FXRateService.default
    queries = [(Currency("USD"), Currency("EUR"), Date(2023, 1, 1))]
    try:
        service.queries(queries, strict=True)
        assert False, "Expected an error to be raised"
    except LookupError:
        pass

def test_queries_non_strict_mode_returns_none():
    service = FXRateService.default
    queries = [(Currency("USD"), Currency("EUR"), Date(2023, 1, 1))]
    result = service.queries(queries, strict=False)
    assert next(result) is None


# LLM-generated content at query #10
#--------------------------

```python
def test_invert_fx_rate():
    from decimal import Decimal
    import datetime
    from pypara.currencies import Currencies
    from pypara.fx import FXRate

    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date(2023, 1, 1), Decimal("2"))
    inverted_rate = ~rate
    assert inverted_rate.ccy1 == Currencies["USD"]
    assert inverted_rate.ccy2 == Currencies["EUR"]
    assert inverted_rate.date == datetime.date(2023, 1, 1)
    assert inverted_rate.value == Decimal("0.5")


# LLM-generated content at query #11
#--------------------------

```python
def test_query_returns_fxrate_for_valid_currency_pair_and_date():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    fxrate = Decimal("0.92")
    mock_service = MockFXRateService({(ccy1, ccy2, asof): fxrate})
    assert mock_service.query(ccy1, ccy2, asof) == fxrate

def test_query_returns_none_for_invalid_currency_pair():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    mock_service = MockFXRateService({})
    assert mock_service.query(ccy1, ccy2, asof) is None

def test_query_returns_none_for_invalid_date():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    mock_service = MockFXRateService({(ccy1, ccy2, Date(2022, 1, 1)): Decimal("0.92")})
    assert mock_service.query(ccy1, ccy2, asof) is None

def test_query_raises_error_in_strict_mode_for_invalid_currency_pair():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    mock_service = MockFXRateService({})
    with pytest.raises(FXRateLookupError):
        mock_service.query(ccy1, ccy2, asof, strict=True)

def test_query_raises_error_in_strict_mode_for_invalid_date():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    mock_service = MockFXRateService({(ccy1, ccy2, Date(2022, 1, 1)): Decimal("0.92")})
    with pytest.raises(FXRateLookupError):
        mock_service.query(ccy1, ccy2, asof, strict=True)


# LLM-generated content at query #12
#--------------------------

```python
def test_query_method_returns_fxrate():
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return FXRate(ccy1, ccy2, Decimal("1.5"), asof)

        def queries(self, queries, strict=False):
            return [self.query(*q) for q in queries]

    service = TestFXRateService()
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    result = service.query(usd, eur, asof)
    assert isinstance(result, FXRate)
    assert result.ccy1 == usd
    assert result.ccy2 == eur
    assert result.rate == Decimal("1.5")
    assert result.asof == asof

def test_query_method_returns_none_when_not_found():
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None

        def queries(self, queries, strict=False):
            return [self.query(*q) for q in queries]

    service = TestFXRateService()
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    result = service.query(usd, eur, asof)
    assert result is None

def test_query_method_raises_error_when_strict_and_not_found():
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if strict:
                raise LookupError("FX rate not found")
            return None

        def queries(self, queries, strict=False):
            return [self.query(*q, strict=strict) for q in queries]

    service = TestFXRateService()
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    try:
        service.query(usd, eur, asof, strict=True)
        assert False, "Expected LookupError"
    except LookupError:
        assert True


# LLM-generated content at query #13
#--------------------------

```python
def test_query_returns_fxrate_for_valid_currency_pair_and_date():
    fx_service = FXRateService.default
    ccy1 = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    result = fx_service.query(ccy1, ccy2, asof)
    assert isinstance(result, FXRate)

def test_query_returns_none_for_invalid_currency_pair():
    fx_service = FXRateService.default
    ccy1 = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("XYZ", "Invalid Currency", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    result = fx_service.query(ccy1, ccy2, asof)
    assert result is None

def test_query_raises_error_for_invalid_currency_pair_with_strict_true():
    fx_service = FXRateService.default
    ccy1 = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("XYZ", "Invalid Currency", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    with pytest.raises(LookupError):
        fx_service.query(ccy1, ccy2, asof, strict=True)

def test_query_returns_none_for_invalid_date():
    fx_service = FXRateService.default
    ccy1 = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(1900, 1, 1)
    result = fx_service.query(ccy1, ccy2, asof)
    assert result is None

def test_query_raises_error_for_invalid_date_with_strict_true():
    fx_service = FXRateService.default
    ccy1 = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(1900, 1, 1)
    with pytest.raises(LookupError):
        fx_service.query(ccy1, ccy2, asof, strict=True)


# LLM-generated content at query #14
#--------------------------

```python
def test_fxrate_constructor():
    from decimal import Decimal
    from pypara.currencies import Currencies
    import datetime

    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("1.2345"))
    assert rate.ccy1 == Currencies["EUR"]
    assert rate.ccy2 == Currencies["USD"]
    assert rate.date == datetime.date.today()
    assert rate.value == Decimal("1.2345")


# LLM-generated content at query #15
#--------------------------

```python
def test_fxrate_constructor_creates_instance_with_correct_attributes():
    from decimal import Decimal
    from pypara.currencies import Currencies
    import datetime

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
    from decimal import Decimal
    from pypara.currencies import Currencies
    import datetime

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
    from decimal import Decimal
    import datetime
    from pypara.currencies import Currencies

    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("2"))
    assert rate.ccy1 == Currencies["EUR"]
    assert rate.ccy2 == Currencies["USD"]
    assert rate.date == datetime.date.today()
    assert rate.value == Decimal("2")


# LLM-generated content at query #17
#--------------------------

```python
def test_fxrate_constructor_creates_instance_with_given_values():
    from decimal import Decimal
    import datetime
    from pypara.currencies import Currencies
    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("2"))
    assert rate.ccy1 == Currencies["EUR"]
    assert rate.ccy2 == Currencies["USD"]
    assert rate.date == datetime.date.today()
    assert rate.value == Decimal("2")


# LLM-generated content at query #18
#--------------------------

```python
def test_fxrate_constructor_creates_instance_with_given_values():
    from decimal import Decimal
    import datetime
    from pypara.currencies import Currencies

    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date(2023, 1, 1)
    value = Decimal("1.2345")

    rate = FXRate(ccy1, ccy2, date, value)

    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value


# LLM-generated content at query #19
#--------------------------

```python
def test_fxrate_constructor_creates_instance_with_correct_properties():
    from pypara.currencies import Currencies
    import datetime
    from decimal import Decimal

    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date(2023, 1, 1), Decimal("1.2"))

    assert rate.ccy1 == Currencies["EUR"]
    assert rate.ccy2 == Currencies["USD"]
    assert rate.date == datetime.date(2023, 1, 1)
    assert rate.value == Decimal("1.2")


# LLM-generated content at query #20
#--------------------------

```python
def test_fxrate_constructor():
    from decimal import Decimal
    import datetime
    from pypara.currencies import Currencies

    # Test normal construction
    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("2"))
    assert rate.ccy1 == Currencies["EUR"]
    assert rate.ccy2 == Currencies["USD"]
    assert rate.date == datetime.date.today()
    assert rate.value == Decimal("2")

    # Test tuple unpacking
    ccy1, ccy2, date, value = rate
    assert ccy1 == Currencies["EUR"]
    assert ccy2 == Currencies["USD"]
    assert date == datetime.date.today()
    assert value == Decimal("2")


# LLM-generated content at query #21
#--------------------------

```python
def test_queries_empty_input():
    service = FXRateService.default
    assert list(service.queries([])) == []

def test_queries_single_valid_query():
    service = FXRateService.default
    queries = [(Currency("USD"), Currency("EUR"), Date(2023, 1, 1))]
    rates = list(service.queries(queries))
    assert len(rates) == 1
    assert isinstance(rates[0], (FXRate, type(None)))

def test_queries_multiple_valid_queries():
    service = FXRateService.default
    queries = [
        (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)),
        (Currency("GBP"), Currency("JPY"), Date(2023, 1, 2))
    ]
    rates = list(service.queries(queries))
    assert len(rates) == 2
    assert all(isinstance(r, (FXRate, type(None))) for r in rates)

def test_queries_with_invalid_pair():
    service = FXRateService.default
    queries = [(Currency("XYZ"), Currency("ABC"), Date(2023, 1, 1))]
    rates = list(service.queries(queries, strict=False))
    assert len(rates) == 1
    assert rates[0] is None

def test_queries_strict_mode_raises_error():
    service = FXRateService.default
    queries = [(Currency("XYZ"), Currency("ABC"), Date(2023, 1, 1))]
    try:
        list(service.queries(queries, strict=True))
        assert False, "Expected an error to be raised"
    except LookupError:
        pass


# LLM-generated content at query #22
#--------------------------

```python
def test_query_with_valid_currency_pair_and_date():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    fx_rate = Decimal("0.92")
    mock_fx_service = create_autospec(FXRateService)
    mock_fx_service.query.return_value = FXRate(ccy1, ccy2, asof, fx_rate)
    result = mock_fx_service.query(ccy1, ccy2, asof)
    assert result == FXRate(ccy1, ccy2, asof, fx_rate)

def test_query_with_invalid_currency_pair():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    mock_fx_service = create_autospec(FXRateService)
    mock_fx_service.query.return_value = None
    result = mock_fx_service.query(ccy1, ccy2, asof)
    assert result is None

def test_query_with_strict_flag_raises_error():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    mock_fx_service = create_autospec(FXRateService)
    mock_fx_service.query.side_effect = LookupError("FX rate not found")
    with pytest.raises(LookupError):
        mock_fx_service.query(ccy1, ccy2, asof, strict=True)


# LLM-generated content at query #23
#--------------------------

```python
def test_fxrate_constructor_creates_instance_with_given_values():
    from decimal import Decimal
    from pypara.currencies import Currencies
    import datetime

    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date(2023, 1, 1)
    value = Decimal("1.2345")

    rate = FXRate(ccy1, ccy2, date, value)

    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value


# LLM-generated content at query #24
#--------------------------

```python
def test_fxrate_constructor_creates_instance_with_given_values():
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

def test_fxrate_constructor_allows_indexed_access():
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


# LLM-generated content at query #25
#--------------------------

```python
def test_query_returns_fxrate_for_valid_currency_pair_and_date():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    service = FXRateService()
    result = service.query(ccy1, ccy2, asof)
    assert isinstance(result, FXRate) or result is None

def test_query_returns_none_for_invalid_currency_pair():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("XYZ", "Invalid Currency", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    service = FXRateService()
    result = service.query(ccy1, ccy2, asof)
    assert result is None

def test_query_raises_error_when_strict_and_rate_not_found():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    service = FXRateService()
    with pytest.raises(LookupError):
        service.query(ccy1, ccy2, asof, strict=True)


# LLM-generated content at query #26
#--------------------------

```python
def test_queries_empty_input():
    service = FXRateService.default
    assert list(service.queries([])) == []

def test_queries_single_valid_query():
    service = FXRateService.default
    queries = [(Currency("USD"), Currency("EUR"), Date(2023, 1, 1))]
    result = list(service.queries(queries))
    assert len(result) == 1
    assert isinstance(result[0], FXRate) or result[0] is None

def test_queries_multiple_valid_queries():
    service = FXRateService.default
    queries = [
        (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)),
        (Currency("GBP"), Currency("JPY"), Date(2023, 1, 2)),
    ]
    result = list(service.queries(queries))
    assert len(result) == 2
    for rate in result:
        assert isinstance(rate, FXRate) or rate is None

def test_queries_with_invalid_query_strict_false():
    service = FXRateService.default
    queries = [(Currency("XYZ"), Currency("ABC"), Date(2023, 1, 1))]
    result = list(service.queries(queries, strict=False))
    assert len(result) == 1
    assert result[0] is None

def test_queries_with_invalid_query_strict_true():
    service = FXRateService.default
    queries = [(Currency("XYZ"), Currency("ABC"), Date(2023, 1, 1))]
    try:
        list(service.queries(queries, strict=True))
        assert False, "Expected a lookup error to be raised"
    except LookupError:
        pass


# LLM-generated content at query #27
#--------------------------

```python
def test_fxrate_constructor_creates_instance_with_correct_properties():
    from decimal import Decimal
    import datetime
    from pypara.currencies import Currencies

    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date(2023, 1, 1), Decimal("1.2345"))

    assert rate.ccy1 == Currencies["EUR"]
    assert rate.ccy2 == Currencies["USD"]
    assert rate.date == datetime.date(2023, 1, 1)
    assert rate.value == Decimal("1.2345")


# LLM-generated content at query #28
#--------------------------

```python
def test_fxrate_constructor():
    from decimal import Decimal
    import datetime
    from pypara.currencies import Currencies

    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("2"))
    assert rate.ccy1 == Currencies["EUR"]
    assert rate.ccy2 == Currencies["USD"]
    assert rate.date == datetime.date.today()
    assert rate.value == Decimal("2")


# LLM-generated content at query #29
#--------------------------

```python
def test_query_with_valid_input():
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return Decimal("1.5")

        def queries(self, queries, strict=False):
            return [Decimal("1.5") for _ in queries]

    service = TestFXRateService()
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    result = service.query(usd, eur, Date(2023, 1, 1))
    assert result == Decimal("1.5")

def test_query_with_strict_false_and_missing_rate():
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None

        def queries(self, queries, strict=False):
            return [None for _ in queries]

    service = TestFXRateService()
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    result = service.query(usd, eur, Date(2023, 1, 1), strict=False)
    assert result is None

def test_query_with_strict_true_and_missing_rate():
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if strict:
                raise LookupError("FX rate not found")
            return None

        def queries(self, queries, strict=False):
            if strict:
                raise LookupError("FX rate not found")
            return [None for _ in queries]

    service = TestFXRateService()
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    try:
        service.query(usd, eur, Date(2023, 1, 1), strict=True)
        assert False, "Expected LookupError"
    except LookupError:
        pass


# LLM-generated content at query #30
#--------------------------

```python
def test_queries_empty_input():
    service = FXRateService.default
    assert list(service.queries([])) == []

def test_queries_single_rate():
    service = FXRateService.default
    query = (Currency("USD"), Currency("EUR"), Date(2023, 1, 1))
    result = list(service.queries([query]))
    assert len(result) == 1
    assert result[0] is not None or isinstance(result[0], FXRate)

def test_queries_multiple_rates():
    service = FXRateService.default
    queries = [
        (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)),
        (Currency("GBP"), Currency("JPY"), Date(2023, 1, 2)),
    ]
    result = list(service.queries(queries))
    assert len(result) == 2
    for rate in result:
        assert rate is not None or isinstance(rate, FXRate)

def test_queries_strict_mode_raises_error():
    service = FXRateService.default
    queries = [
        (Currency("USD"), Currency("XYZ"), Date(2023, 1, 1)),  # Invalid currency
    ]
    try:
        list(service.queries(queries, strict=True))
        assert False, "Expected a lookup error to be raised"
    except LookupError:
        pass

def test_queries_non_strict_mode_returns_none():
    service = FXRateService.default
    queries = [
        (Currency("USD"), Currency("XYZ"), Date(2023, 1, 1)),  # Invalid currency
    ]
    result = list(service.queries(queries, strict=False))
    assert len(result) == 1
    assert result[0] is None


# LLM-generated content at query #31
#--------------------------

```python
def test_fxrate_constructor_creates_instance_with_correct_properties():
    from pypara.currencies import Currencies
    import datetime
    from decimal import Decimal
    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date(2023, 1, 1), Decimal("1.2"))
    assert rate.ccy1 == Currencies["EUR"]
    assert rate.ccy2 == Currencies["USD"]
    assert rate.date == datetime.date(2023, 1, 1)
    assert rate.value == Decimal("1.2")


# LLM-generated content at query #32
#--------------------------

```python
def test_fxrate_constructor_creates_instance_with_given_values():
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


# LLM-generated content at query #33
#--------------------------

```python
def test_query_returns_fxrate_for_valid_currency_pair_and_date():
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return FXRate(ccy1, ccy2, Decimal("1.5"), asof)

        def queries(self, queries, strict=False):
            return [self.query(*q, strict=strict) for q in queries]

    service = TestFXRateService()
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    today = Date.today()
    result = service.query(usd, eur, today)
    assert result == FXRate(usd, eur, Decimal("1.5"), today)

def test_query_returns_none_for_invalid_currency_pair_and_date():
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None

        def queries(self, queries, strict=False):
            return [self.query(*q, strict=strict) for q in queries]

    service = TestFXRateService()
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    today = Date.today()
    result = service.query(usd, eur, today)
    assert result is None

def test_query_raises_error_for_invalid_currency_pair_and_date_with_strict_true():
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if strict:
                raise LookupError("FX rate not found")
            return None

        def queries(self, queries, strict=False):
            return [self.query(*q, strict=strict) for q in queries]

    service = TestFXRateService()
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    today = Date.today()
    try:
        service.query(usd, eur, today, strict=True)
        assert False, "Expected LookupError"
    except LookupError:
        pass


# LLM-generated content at query #34
#--------------------------

```python
def test_fxrate_constructor_creates_instance_with_correct_properties():
    from decimal import Decimal
    import datetime
    from pypara.currencies import Currencies

    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("2"))

    assert rate.ccy1 == Currencies["EUR"]
    assert rate.ccy2 == Currencies["USD"]
    assert rate.date == datetime.date.today()
    assert rate.value == Decimal("2")


# LLM-generated content at query #35
#--------------------------

```python
def test_queries_empty_input():
    service = FXRateService.default
    assert list(service.queries([])) == []

def test_queries_single_valid_query():
    service = FXRateService.default
    queries = [(Currency("USD"), Currency("EUR"), Date(2023, 1, 1))]
    rates = service.queries(queries)
    assert next(rates) is not None

def test_queries_multiple_valid_queries():
    service = FXRateService.default
    queries = [
        (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)),
        (Currency("GBP"), Currency("JPY"), Date(2023, 1, 2))
    ]
    rates = list(service.queries(queries))
    assert len(rates) == 2
    assert all(rate is not None for rate in rates)

def test_queries_invalid_query_non_strict():
    service = FXRateService.default
    queries = [(Currency("XYZ"), Currency("ABC"), Date(2023, 1, 1))]
    rates = list(service.queries(queries, strict=False))
    assert rates == [None]

def test_queries_invalid_query_strict():
    service = FXRateService.default
    queries = [(Currency("XYZ"), Currency("ABC"), Date(2023, 1, 1))]
    try:
        list(service.queries(queries, strict=True))
        assert False, "Expected a lookup error to be raised"
    except LookupError:
        pass

def test_queries_mixed_valid_invalid_non_strict():
    service = FXRateService.default
    queries = [
        (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)),
        (Currency("XYZ"), Currency("ABC"), Date(2023, 1, 1))
    ]
    rates = list(service.queries(queries, strict=False))
    assert len(rates) == 2
    assert rates[0] is not None
    assert rates[1] is None


# LLM-generated content at query #36
#--------------------------

```python
def test_fxrate_constructor_creates_instance_with_correct_properties():
    from pypara.currencies import Currencies
    import datetime
    from decimal import Decimal

    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("2"))

    assert rate.ccy1 == Currencies["EUR"]
    assert rate.ccy2 == Currencies["USD"]
    assert rate.date == datetime.date.today()
    assert rate.value == Decimal("2")


