####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_invert_fx_rate():
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
    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date(2023, 1, 1), Decimal("2"))
    inverted_rate = ~rate
    assert inverted_rate == FXRate(Currencies["USD"], Currencies["EUR"], datetime.date(2023, 1, 1), Decimal("0.5"))


# LLM-generated content at query #3
#--------------------------

```python
def test_queries_empty_input():
    service = FXRateService.default
    result = list(service.queries([]))
    assert result == []

def test_queries_single_valid_query():
    service = FXRateService.default
    query = (Currency("USD"), Currency("EUR"), Date(2023, 1, 1))
    result = list(service.queries([query]))
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
    assert all(isinstance(r, FXRate) or r is None for r in result)

def test_queries_strict_mode_raises_error():
    service = FXRateService.default
    query = (Currency("XYZ"), Currency("ABC"), Date(2023, 1, 1))
    try:
        list(service.queries([query], strict=True))
        assert False, "Expected a lookup error to be raised"
    except LookupError:
        pass

def test_queries_non_strict_mode_returns_none():
    service = FXRateService.default
    query = (Currency("XYZ"), Currency("ABC"), Date(2023, 1, 1))
    result = list(service.queries([query], strict=False))
    assert len(result) == 1
    assert result[0] is None


# LLM-generated content at query #4
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
    assert isinstance(rates[0], FXRate) or rates[0] is None

def test_queries_multiple_valid_queries():
    service = FXRateService.default
    queries = [
        (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)),
        (Currency("GBP"), Currency("JPY"), Date(2023, 1, 2))
    ]
    rates = list(service.queries(queries))
    assert len(rates) == 2
    for rate in rates:
        assert isinstance(rate, FXRate) or rate is None

def test_queries_with_invalid_currency_pair():
    service = FXRateService.default
    queries = [(Currency("XYZ"), Currency("ABC"), Date(2023, 1, 1))]
    rates = list(service.queries(queries))
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


# LLM-generated content at query #5
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


# LLM-generated content at query #6
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

def test_query_with_same_currency():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    result = FXRateService.default.query(ccy1, ccy2, asof)
    assert result == FXRate(Decimal("1.0"), ccy1, ccy2, asof)


# LLM-generated content at query #7
#--------------------------

```python
def test_query_returns_fxrate_for_valid_currency_pair_and_date():
    fx_rate_service = FXRateService.default
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    result = fx_rate_service.query(ccy1, ccy2, asof)
    assert isinstance(result, FXRate) or result is None

def test_query_returns_none_for_invalid_currency_pair():
    fx_rate_service = FXRateService.default
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("XYZ", "Unknown Currency", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    result = fx_rate_service.query(ccy1, ccy2, asof)
    assert result is None

def test_query_raises_error_when_strict_and_rate_not_found():
    fx_rate_service = FXRateService.default
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("XYZ", "Unknown Currency", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    try:
        fx_rate_service.query(ccy1, ccy2, asof, strict=True)
    except LookupError:
        pass
    else:
        assert False, "Expected LookupError when strict=True and rate not found"


# LLM-generated content at query #8
#--------------------------

```python
def test_query_with_valid_currency_pair_and_date():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    fx_rate_service = FXRateService.default
    result = fx_rate_service.query(ccy1, ccy2, asof)
    assert isinstance(result, (FXRate, type(None)))

def test_query_with_invalid_currency_pair():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    fx_rate_service = FXRateService.default
    result = fx_rate_service.query(ccy1, ccy2, asof, strict=True)
    assert isinstance(result, (FXRate, type(None)))

def test_query_with_strict_flag_false():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    fx_rate_service = FXRateService.default
    result = fx_rate_service.query(ccy1, ccy2, asof, strict=False)
    assert isinstance(result, (FXRate, type(None)))


# LLM-generated content at query #9
#--------------------------

```python
def test_queries_with_empty_input():
    service = FXRateService.default
    result = service.queries([], strict=False)
    assert list(result) == []

def test_queries_with_single_query():
    service = FXRateService.default
    query = (Currency("USD"), Currency("EUR"), Date(2023, 1, 1))
    result = service.queries([query], strict=False)
    assert len(list(result)) == 1

def test_queries_with_multiple_queries():
    service = FXRateService.default
    queries = [
        (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)),
        (Currency("GBP"), Currency("JPY"), Date(2023, 1, 2))
    ]
    result = service.queries(queries, strict=False)
    assert len(list(result)) == 2

def test_queries_with_strict_false():
    service = FXRateService.default
    query = (Currency("USD"), Currency("EUR"), Date(2023, 1, 1))
    result = service.queries([query], strict=False)
    assert all(r is None or isinstance(r, FXRate) for r in result)

def test_queries_with_strict_true():
    service = FXRateService.default
    query = (Currency("USD"), Currency("EUR"), Date(2023, 1, 1))
    try:
        service.queries([query], strict=True)
    except LookupError:
        pass


# LLM-generated content at query #10
#--------------------------

```python
def test_fxrate_constructor():
    from pypara.currencies import Currencies
    import datetime
    from decimal import Decimal

    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("2"))

    assert rate.ccy1 == Currencies["EUR"]
    assert rate.ccy2 == Currencies["USD"]
    assert rate.date == datetime.date.today()
    assert rate.value == Decimal("2")


# LLM-generated content at query #11
#--------------------------

```python
def test_queries_returns_iterable_of_rates():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return 1.5 if (ccy1, ccy2, asof) == ("USD", "EUR", "2023-01-01") else None

        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = MockFXRateService()
    queries = [("USD", "EUR", "2023-01-01"), ("GBP", "JPY", "2023-01-01")]
    rates = service.queries(queries)
    assert list(rates) == [1.5, None]

def test_queries_raises_error_when_strict_and_rate_not_found():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if strict and (ccy1, ccy2, asof) != ("USD", "EUR", "2023-01-01"):
                raise LookupError("Rate not found")
            return 1.5 if (ccy1, ccy2, asof) == ("USD", "EUR", "2023-01-01") else None

        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = MockFXRateService()
    queries = [("USD", "EUR", "2023-01-01"), ("GBP", "JPY", "2023-01-01")]
    try:
        service.queries(queries, strict=True)
        assert False, "Expected LookupError"
    except LookupError:
        pass


# LLM-generated content at query #12
#--------------------------

```python
def test_query_with_valid_currencies_and_date():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    result = FXRateService.default.query(ccy1, ccy2, asof)
    assert isinstance(result, (Decimal, type(None)))

def test_query_with_invalid_currencies():
    ccy1 = "USD"
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    with pytest.raises(TypeError):
        FXRateService.default.query(ccy1, ccy2, asof)

def test_query_with_strict_flag_true_and_missing_rate():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("XYZ", "Unknown Currency", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    with pytest.raises(LookupError):
        FXRateService.default.query(ccy1, ccy2, asof, strict=True)

def test_query_with_strict_flag_false_and_missing_rate():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("XYZ", "Unknown Currency", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    result = FXRateService.default.query(ccy1, ccy2, asof, strict=False)
    assert result is None


# LLM-generated content at query #13
#--------------------------

```python
def test_fxrate_constructor_creates_instance_with_correct_properties():
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


# LLM-generated content at query #14
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


# LLM-generated content at query #15
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


# LLM-generated content at query #16
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

def test_fxrate_constructor_allows_indexed_access():
    from decimal import Decimal
    import datetime
    from pypara.currencies import Currencies
    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("2"))
    ccy1, ccy2, date, value = rate
    assert ccy1 == Currencies["EUR"]
    assert ccy2 == Currencies["USD"]
    assert date == datetime.date.today()
    assert value == Decimal("2")


# LLM-generated content at query #17
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


# LLM-generated content at query #18
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


# LLM-generated content at query #19
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

def test_queries_with_invalid_query_strict_false():
    service = FXRateService.default
    queries = [(Currency("USD"), Currency("XYZ"), Date(2023, 1, 1))]
    rates = list(service.queries(queries, strict=False))
    assert len(rates) == 1
    assert rates[0] is None

def test_queries_with_invalid_query_strict_true():
    service = FXRateService.default
    queries = [(Currency("USD"), Currency("XYZ"), Date(2023, 1, 1))]
    try:
        list(service.queries(queries, strict=True))
        assert False, "Expected a lookup error to be raised"
    except LookupError:
        pass


# LLM-generated content at query #20
#--------------------------

```python
def test_query_returns_fxrate_for_valid_currency_pair_and_date():
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return FXRate(ccy1, ccy2, Decimal("1.5"), asof)

        def queries(self, queries, strict=False):
            return [self.query(*q, strict) for q in queries]

    service = TestFXRateService()
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    date = Date(2023, 1, 1)
    result = service.query(usd, eur, date)
    assert result == FXRate(usd, eur, Decimal("1.5"), date)

def test_query_returns_none_for_invalid_currency_pair():
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None

        def queries(self, queries, strict=False):
            return [self.query(*q, strict) for q in queries]

    service = TestFXRateService()
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    xyz = Currency.of("XYZ", "Unknown", 2, CurrencyType.MONEY)
    date = Date(2023, 1, 1)
    result = service.query(usd, xyz, date)
    assert result is None

def test_query_raises_error_in_strict_mode_for_invalid_currency_pair():
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if strict:
                raise LookupError("FX rate not found")
            return None

        def queries(self, queries, strict=False):
            return [self.query(*q, strict) for q in queries]

    service = TestFXRateService()
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    xyz = Currency.of("XYZ", "Unknown", 2, CurrencyType.MONEY)
    date = Date(2023, 1, 1)
    try:
        service.query(usd, xyz, date, strict=True)
        assert False, "Expected LookupError"
    except LookupError:
        pass


# LLM-generated content at query #21
#--------------------------

```python
def test_fxrate_constructor():
    from pypara.currencies import Currencies
    import datetime
    from decimal import Decimal

    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("2"))
    assert rate.ccy1 == Currencies["EUR"]
    assert rate.ccy2 == Currencies["USD"]
    assert rate.date == datetime.date.today()
    assert rate.value == Decimal("2")


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
def test_fxrate_constructor():
    from pypara.currencies import Currencies
    import datetime
    from decimal import Decimal

    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("2"))
    assert rate.ccy1 == Currencies["EUR"]
    assert rate.ccy2 == Currencies["USD"]
    assert rate.date == datetime.date.today()
    assert rate.value == Decimal("2")


# LLM-generated content at query #24
#--------------------------

```python
def test_queries_empty_input():
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None

        def queries(self, queries, strict=False):
            return []

    service = TestFXRateService()
    assert list(service.queries([])) == []

def test_queries_single_query():
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return FXRate(Decimal("1.2345"))

        def queries(self, queries, strict=False):
            return [self.query(*q) for q in queries]

    service = TestFXRateService()
    result = list(service.queries([(Currency("USD"), Currency("EUR"), Date(2023, 1, 1))]))
    assert len(result) == 1
    assert result[0] == FXRate(Decimal("1.2345"))

def test_queries_multiple_queries():
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return FXRate(Decimal("1.2345"))

        def queries(self, queries, strict=False):
            return [self.query(*q) for q in queries]

    service = TestFXRateService()
    queries = [
        (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)),
        (Currency("GBP"), Currency("JPY"), Date(2023, 1, 2))
    ]
    result = list(service.queries(queries))
    assert len(result) == 2
    assert all(r == FXRate(Decimal("1.2345")) for r in result)

def test_queries_with_none_result():
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None

        def queries(self, queries, strict=False):
            return [self.query(*q) for q in queries]

    service = TestFXRateService()
    queries = [
        (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)),
        (Currency("GBP"), Currency("JPY"), Date(2023, 1, 2))
    ]
    result = list(service.queries(queries))
    assert len(result) == 2
    assert all(r is None for r in result)

def test_queries_strict_mode_raises_error():
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if strict:
                raise LookupError("Rate not found")
            return None

        def queries(self, queries, strict=False):
            return [self.query(*q, strict=strict) for q in queries]

    service = TestFXRateService()
    queries = [(Currency("USD"), Currency("EUR"), Date(2023, 1, 1))]
    try:
        list(service.queries(queries, strict=True))
        assert False, "Expected LookupError"
    except LookupError:
        pass


# LLM-generated content at query #25
#--------------------------

```python
def test_query_returns_fxrate_for_valid_currency_pair_and_date():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    fx_rate = Decimal("0.92")
    fx_rate_service = FXRateService()
    assert fx_rate_service.query(ccy1, ccy2, asof) == FXRate(ccy1, ccy2, asof, fx_rate)

def test_query_returns_none_for_invalid_currency_pair():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("XYZ", "Invalid Currency", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    fx_rate_service = FXRateService()
    assert fx_rate_service.query(ccy1, ccy2, asof) is None

def test_query_raises_error_for_invalid_date():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    fx_rate_service = FXRateService()
    with pytest.raises(LookupError):
        fx_rate_service.query(ccy1, ccy2, asof, strict=True)


# LLM-generated content at query #26
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


# LLM-generated content at query #27
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
        (Currency("GBP"), Currency("JPY"), Date(2023, 1, 2))
    ]
    result = list(service.queries(queries))
    assert len(result) == 2
    assert all(isinstance(r, (FXRate, type(None))) for r in result)

def test_queries_strict_mode_raises_error():
    service = FXRateService.default
    query = (Currency("USD"), Currency("EUR"), Date(2023, 1, 1))
    try:
        service.queries([query], strict=True)
        assert False, "Expected a lookup error to be raised"
    except LookupError:
        pass


# LLM-generated content at query #28
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
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    date = Date(2023, 1, 1)
    result = service.query(usd, eur, date)
    assert result == FXRate(usd, eur, Decimal("1.5"), date)

def test_query_returns_none_for_invalid_currency_pair():
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None

        def queries(self, queries, strict=False):
            return [self.query(*q, strict=strict) for q in queries]

    service = TestFXRateService()
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    xyz = Currency.of("XYZ", "Unknown Currency", 2, CurrencyType.MONEY)
    date = Date(2023, 1, 1)
    result = service.query(usd, xyz, date)
    assert result is None

def test_query_raises_error_in_strict_mode_for_invalid_currency_pair():
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if strict:
                raise LookupError("FX rate not found")
            return None

        def queries(self, queries, strict=False):
            return [self.query(*q, strict=strict) for q in queries]

    service = TestFXRateService()
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    xyz = Currency.of("XYZ", "Unknown Currency", 2, CurrencyType.MONEY)
    date = Date(2023, 1, 1)
    try:
        service.query(usd, xyz, date, strict=True)
        assert False, "Expected LookupError"
    except LookupError:
        pass


# LLM-generated content at query #29
#--------------------------

```python
def test_fxrate_constructor_creates_instance_with_given_values():
    from pypara.currencies import Currencies
    import datetime
    from decimal import Decimal

    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date(2023, 1, 1)
    value = Decimal("1.2345")

    rate = FXRate(ccy1, ccy2, date, value)

    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value


# LLM-generated content at query #30
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


# LLM-generated content at query #31
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

def test_fxrate_constructor_allows_indexed_access():
    from decimal import Decimal
    import datetime
    from pypara.currencies import Currencies

    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date(2023, 1, 1), Decimal("1.5"))

    assert rate[0] == Currencies["EUR"]
    assert rate[1] == Currencies["USD"]
    assert rate[2] == datetime.date(2023, 1, 1)
    assert rate[3] == Decimal("1.5")


# LLM-generated content at query #32
#--------------------------

```python
def test_query_returns_fx_rate_for_valid_currency_pair_and_date():
    ccy1 = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    fx_rate = Decimal("0.92")
    fx_rate_service = FXRateService()
    fx_rate_service.query(ccy1, ccy2, asof) == fx_rate

def test_query_returns_none_for_invalid_currency_pair():
    ccy1 = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("XYZ", "Invalid Currency", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    fx_rate_service = FXRateService()
    fx_rate_service.query(ccy1, ccy2, asof) is None

def test_query_returns_none_for_invalid_date():
    ccy1 = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(1900, 1, 1)
    fx_rate_service = FXRateService()
    fx_rate_service.query(ccy1, ccy2, asof) is None

def test_query_raises_error_when_strict_and_rate_not_found():
    ccy1 = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("XYZ", "Invalid Currency", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    fx_rate_service = FXRateService()
    try:
        fx_rate_service.query(ccy1, ccy2, asof, strict=True)
    except LookupError:
        pass
    else:
        assert False, "Expected LookupError to be raised"


# LLM-generated content at query #33
#--------------------------

```python
def test_queries_returns_iterable_of_rates():
    queries = [(Currency("USD"), Currency("EUR"), Date(2023, 1, 1))]
    result = FXRateService.default.queries(queries)
    assert isinstance(result, Iterable)
    assert all(isinstance(rate, (FXRate, type(None))) for rate in result)

def test_queries_with_strict_false_returns_none_for_missing_rates():
    queries = [(Currency("USD"), Currency("XYZ"), Date(2023, 1, 1))]
    result = FXRateService.default.queries(queries, strict=False)
    assert next(result) is None

def test_queries_with_strict_true_raises_error_for_missing_rates():
    queries = [(Currency("USD"), Currency("XYZ"), Date(2023, 1, 1))]
    with pytest.raises(LookupError):
        list(FXRateService.default.queries(queries, strict=True))

def test_queries_returns_correct_rates_for_valid_pairs():
    queries = [(Currency("USD"), Currency("EUR"), Date(2023, 1, 1))]
    result = FXRateService.default.queries(queries)
    assert next(result) == FXRate(Decimal("0.92"))

def test_queries_handles_empty_input():
    queries = []
    result = FXRateService.default.queries(queries)
    assert list(result) == []


# LLM-generated content at query #34
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

def test_fxrate_constructor_allows_indexed_access():
    from pypara.currencies import Currencies
    import datetime
    from decimal import Decimal

    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("2"))
    ccy1, ccy2, date, value = rate
    assert ccy1 == Currencies["EUR"]
    assert ccy2 == Currencies["USD"]
    assert date == datetime.date.today()
    assert value == Decimal("2")


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_query_with_valid_inputs():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    result = FXRateService.default.query(ccy1, ccy2, asof)
    assert isinstance(result, FXRate) or result is None

def test_query_with_strict_false():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    result = FXRateService.default.query(ccy1, ccy2, asof, strict=False)
    assert isinstance(result, FXRate) or result is None

def test_query_with_strict_true():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    try:
        FXRateService.default.query(ccy1, ccy2, asof, strict=True)
    except LookupError:
        pass


# LLM-generated content at query #2
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


# LLM-generated content at query #3
#--------------------------

```python
def test_query_with_valid_input():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    result = FXRateService.default.query(ccy1, ccy2, asof)
    assert result is not None or result is None

def test_query_with_strict_true_raises_error():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    try:
        FXRateService.default.query(ccy1, ccy2, asof, strict=True)
    except LookupError:
        pass
    else:
        assert False, "Expected LookupError to be raised"


# LLM-generated content at query #4
#--------------------------

```python
def test_queries_empty_input():
    service = FXRateService.default
    assert list(service.queries([])) == []

def test_queries_single_valid_query():
    service = FXRateService.default
    query = (Currency("USD"), Currency("EUR"), Date(2023, 1, 1))
    result = list(service.queries([query]))
    assert len(result) == 1
    assert isinstance(result[0], (FXRate, type(None)))

def test_queries_multiple_valid_queries():
    service = FXRateService.default
    queries = [
        (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)),
        (Currency("GBP"), Currency("JPY"), Date(2023, 1, 2))
    ]
    result = list(service.queries(queries))
    assert len(result) == 2
    for rate in result:
        assert isinstance(rate, (FXRate, type(None)))

def test_queries_invalid_query_non_strict():
    service = FXRateService.default
    query = (Currency("XYZ"), Currency("ABC"), Date(2023, 1, 1))
    result = list(service.queries([query], strict=False))
    assert len(result) == 1
    assert result[0] is None

def test_queries_invalid_query_strict():
    service = FXRateService.default
    query = (Currency("XYZ"), Currency("ABC"), Date(2023, 1, 1))
    try:
        list(service.queries([query], strict=True))
        assert False, "Expected a lookup error to be raised"
    except LookupError:
        pass

def test_queries_mixed_valid_invalid_non_strict():
    service = FXRateService.default
    queries = [
        (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)),
        (Currency("XYZ"), Currency("ABC"), Date(2023, 1, 1))
    ]
    result = list(service.queries(queries, strict=False))
    assert len(result) == 2
    assert isinstance(result[0], (FXRate, type(None)))
    assert result[1] is None

def test_queries_mixed_valid_invalid_strict():
    service = FXRateService.default
    queries = [
        (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)),
        (Currency("XYZ"), Currency("ABC"), Date(2023, 1, 1))
    ]
    try:
        list(service.queries(queries, strict=True))
        assert False, "Expected a lookup error to be raised"
    except LookupError:
        pass


# LLM-generated content at query #5
#--------------------------

```python
def test_queries_returns_empty_iterable_for_empty_input():
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            pass

        def queries(self, queries, strict=False):
            return iter([])

    service = TestFXRateService()
    result = list(service.queries([]))
    assert result == []

def test_queries_returns_correct_rates_for_valid_queries():
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if ccy1 == "USD" and ccy2 == "EUR" and asof == "2023-01-01":
                return 0.85
            elif ccy1 == "EUR" and ccy2 == "GBP" and asof == "2023-01-01":
                return 0.90
            return None

        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = TestFXRateService()
    queries = [("USD", "EUR", "2023-01-01"), ("EUR", "GBP", "2023-01-01")]
    result = list(service.queries(queries))
    assert result == [0.85, 0.90]

def test_queries_returns_none_for_invalid_queries():
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None

        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = TestFXRateService()
    queries = [("USD", "EUR", "2023-01-01"), ("EUR", "GBP", "2023-01-01")]
    result = list(service.queries(queries))
    assert result == [None, None]

def test_queries_raises_error_for_strict_mode_with_invalid_queries():
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if strict:
                raise LookupError("Rate not found")
            return None

        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = TestFXRateService()
    queries = [("USD", "EUR", "2023-01-01")]
    try:
        list(service.queries(queries, strict=True))
        assert False, "Expected LookupError"
    except LookupError:
        pass


# LLM-generated content at query #6
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
    assert all(isinstance(r, (FXRate, type(None))) for r in rates)

def test_queries_strict_mode_raises_error():
    service = FXRateService.default
    queries = [(Currency("XYZ"), Currency("ABC"), Date(2023, 1, 1))]
    try:
        list(service.queries(queries, strict=True))
        assert False, "Expected a lookup error to be raised"
    except LookupError:
        pass


# LLM-generated content at query #7
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
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    date = Date(2023, 1, 1)
    result = service.query(usd, eur, date)
    assert result == FXRate(usd, eur, Decimal("1.5"), date)

def test_query_returns_none_for_invalid_currency_pair_and_date():
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None

        def queries(self, queries, strict=False):
            pass

    service = TestFXRateService()
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    xyz = Currency.of("XYZ", "Test Currency", 2, CurrencyType.MONEY)
    date = Date(2023, 1, 1)
    result = service.query(usd, xyz, date)
    assert result is None

def test_query_raises_error_for_strict_mode_with_invalid_currency_pair():
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if strict:
                raise LookupError("FX rate not found")
            return None

        def queries(self, queries, strict=False):
            pass

    service = TestFXRateService()
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    xyz = Currency.of("XYZ", "Test Currency", 2, CurrencyType.MONEY)
    date = Date(2023, 1, 1)
    try:
        service.query(usd, xyz, date, strict=True)
        assert False, "Expected LookupError"
    except LookupError:
        pass


# LLM-generated content at query #8
#--------------------------

```python
def test_fxrate_constructor_with_valid_inputs():
    from decimal import Decimal
    import datetime
    from pypara.currencies import Currencies
    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("2"))
    assert rate.ccy1 == Currencies["EUR"]
    assert rate.ccy2 == Currencies["USD"]
    assert rate.date == datetime.date.today()
    assert rate.value == Decimal("2")


# LLM-generated content at query #9
#--------------------------

```python
def test_query_with_valid_currencies_and_date():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    fx_rate = Decimal("0.92")
    mock_service = FXRateService()
    mock_service.query = lambda c1, c2, a, s: fx_rate if (c1, c2, a) == (ccy1, ccy2, asof) else None
    assert mock_service.query(ccy1, ccy2, asof) == fx_rate

def test_query_with_invalid_currency_pair():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    mock_service = FXRateService()
    mock_service.query = lambda c1, c2, a, s: None
    assert mock_service.query(ccy1, ccy2, asof) is None

def test_query_with_strict_flag_raises_error():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    mock_service = FXRateService()
    mock_service.query = lambda c1, c2, a, s: None
    try:
        mock_service.query(ccy1, ccy2, asof, strict=True)
        assert False, "Expected an error to be raised"
    except Exception:
        pass


# LLM-generated content at query #10
#--------------------------

```python
def test_fxrate_constructor_creates_instance_with_given_values():
    from decimal import Decimal
    import datetime
    from pypara.currencies import Currencies
    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date(2023, 1, 1), Decimal("1.2"))
    assert rate.ccy1 == Currencies["EUR"]
    assert rate.ccy2 == Currencies["USD"]
    assert rate.date == datetime.date(2023, 1, 1)
    assert rate.value == Decimal("1.2")


# LLM-generated content at query #11
#--------------------------

```python
def test_fxrate_constructor():
    from decimal import Decimal
    from pypara.currencies import Currencies
    import datetime

    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("2"))
    assert rate.ccy1 == Currencies["EUR"]
    assert rate.ccy2 == Currencies["USD"]
    assert rate.date == datetime.date.today()
    assert rate.value == Decimal("2")


# LLM-generated content at query #12
#--------------------------

```python
def test_fxrate_constructor():
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


# LLM-generated content at query #13
#--------------------------

```python
def test_fxrate_constructor_creates_instance_with_correct_properties():
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


# LLM-generated content at query #14
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

def test_fxrate_constructor_allows_indexed_access():
    from pypara.currencies import Currencies
    import datetime
    from decimal import Decimal

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
    from decimal import Decimal
    from pypara.currencies import Currencies
    import datetime

    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("2"))
    assert rate.ccy1 == Currencies["EUR"]
    assert rate.ccy2 == Currencies["USD"]
    assert rate.date == datetime.date.today()
    assert rate.value == Decimal("2")


# LLM-generated content at query #16
#--------------------------

```python
def test_fxrate_constructor():
    from decimal import Decimal
    from pypara.currencies import Currencies
    import datetime

    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("2"))
    assert rate[0] == Currencies["EUR"]
    assert rate[1] == Currencies["USD"]
    assert rate[2] == datetime.date.today()
    assert rate[3] == Decimal("2")


# LLM-generated content at query #17
#--------------------------

```python
def test_queries_with_empty_input():
    service = FXRateService.default
    result = list(service.queries([]))
    assert result == []

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

def test_queries_with_strict_mode_enabled():
    service = FXRateService.default
    queries = [(Currency("USD"), Currency("XYZ"), Date(2023, 1, 1))]
    try:
        list(service.queries(queries, strict=True))
        assert False, "Expected a lookup error to be raised"
    except LookupError:
        pass

def test_queries_with_strict_mode_disabled():
    service = FXRateService.default
    queries = [(Currency("USD"), Currency("XYZ"), Date(2023, 1, 1))]
    result = list(service.queries(queries, strict=False))
    assert len(result) == 1
    assert result[0] is None


# LLM-generated content at query #18
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


# LLM-generated content at query #19
#--------------------------

```python
def test_query_returns_correct_fxrate():
    class TestFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            return FXRate(ccy1, ccy2, Decimal("1.5"), asof)

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            pass

    service = TestFXRateService()
    usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    today = Date.today()
    result = service.query(usd, eur, today)
    assert result == FXRate(usd, eur, Decimal("1.5"), today)

def test_query_returns_none_when_not_found():
    class TestFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            return None

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            pass

    service = TestFXRateService()
    usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    today = Date.today()
    result = service.query(usd, eur, today)
    assert result is None

def test_query_raises_error_when_strict_and_not_found():
    class TestFXRateService(FXRateService):
        def query(self, ccy1: Currency, ccy2: Currency, asof: Date, strict: bool = False) -> Optional[FXRate]:
            if strict:
                raise LookupError("FX rate not found")
            return None

        def queries(self, queries: Iterable[FXRateService.TQuery], strict: bool = False) -> Iterable[Optional[FXRate]]:
            pass

    service = TestFXRateService()
    usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    today = Date.today()
    try:
        service.query(usd, eur, today, strict=True)
        assert False, "Expected LookupError"
    except LookupError:
        pass


# LLM-generated content at query #20
#--------------------------

```python
def test_fxrate_constructor_creates_instance_with_given_values():
    from decimal import Decimal
    from pypara.currencies import Currencies
    import datetime

    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("2"))

    assert rate[0] == Currencies["EUR"]
    assert rate[1] == Currencies["USD"]
    assert rate[2] == datetime.date.today()
    assert rate[3] == Decimal("2")


# LLM-generated content at query #21
#--------------------------

```python
def test_query_with_valid_inputs():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    result = FXRateService.default.query(ccy1, ccy2, asof)
    assert isinstance(result, (FXRate, type(None)))

def test_query_with_strict_true_and_missing_rate():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("XYZ", "Unknown", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    try:
        FXRateService.default.query(ccy1, ccy2, asof, strict=True)
    except LookupError:
        pass
    else:
        assert False, "Expected LookupError"

def test_query_with_strict_false_and_missing_rate():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("XYZ", "Unknown", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    result = FXRateService.default.query(ccy1, ccy2, asof, strict=False)
    assert result is None


# LLM-generated content at query #22
#--------------------------

```python
def test_queries_with_valid_input():
    queries = [("USD", "EUR", "2023-01-01"), ("GBP", "JPY", "2023-01-01")]
    expected_rates = [Decimal("0.92"), Decimal("140.50")]
    mock_service = FXRateService()
    mock_service.queries = MagicMock(return_value=expected_rates)
    result = mock_service.queries(queries)
    assert list(result) == expected_rates

def test_queries_with_empty_input():
    queries = []
    expected_rates = []
    mock_service = FXRateService()
    mock_service.queries = MagicMock(return_value=expected_rates)
    result = mock_service.queries(queries)
    assert list(result) == expected_rates

def test_queries_with_strict_flag():
    queries = [("USD", "EUR", "2023-01-01"), ("GBP", "JPY", "2023-01-01")]
    expected_rates = [Decimal("0.92"), Decimal("140.50")]
    mock_service = FXRateService()
    mock_service.queries = MagicMock(return_value=expected_rates)
    result = mock_service.queries(queries, strict=True)
    assert list(result) == expected_rates

def test_queries_with_none_rate():
    queries = [("USD", "EUR", "2023-01-01"), ("GBP", "JPY", "2023-01-01")]
    expected_rates = [None, Decimal("140.50")]
    mock_service = FXRateService()
    mock_service.queries = MagicMock(return_value=expected_rates)
    result = mock_service.queries(queries)
    assert list(result) == expected_rates


# LLM-generated content at query #23
#--------------------------

```python
def test_fxrate_constructor_creates_instance_with_given_values():
    from decimal import Decimal
    import datetime
    from pypara.currencies import Currencies

    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date(2023, 1, 1), Decimal("1.2"))
    assert rate.ccy1 == Currencies["EUR"]
    assert rate.ccy2 == Currencies["USD"]
    assert rate.date == datetime.date(2023, 1, 1)
    assert rate.value == Decimal("1.2")

def test_fxrate_constructor_allows_indexed_access():
    from decimal import Decimal
    import datetime
    from pypara.currencies import Currencies

    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date(2023, 1, 1), Decimal("1.2"))
    assert rate[0] == Currencies["EUR"]
    assert rate[1] == Currencies["USD"]
    assert rate[2] == datetime.date(2023, 1, 1)
    assert rate[3] == Decimal("1.2")


# LLM-generated content at query #24
#--------------------------

```python
def test_fxrate_constructor_creates_instance():
    from decimal import Decimal
    import datetime
    from pypara.currencies import Currencies
    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("2"))
    assert rate.ccy1 == Currencies["EUR"]
    assert rate.ccy2 == Currencies["USD"]
    assert rate.date == datetime.date.today()
    assert rate.value == Decimal("2")


# LLM-generated content at query #25
#--------------------------

```python
def test_queries_empty_input():
    fx_service = FXRateService.default
    result = fx_service.queries([])
    assert list(result) == []

def test_queries_single_valid_query():
    fx_service = FXRateService.default
    query = (Currency("USD"), Currency("EUR"), Date(2023, 1, 1))
    result = fx_service.queries([query])
    assert len(list(result)) == 1
    assert isinstance(list(result)[0], FXRate)

def test_queries_multiple_valid_queries():
    fx_service = FXRateService.default
    queries = [
        (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)),
        (Currency("GBP"), Currency("JPY"), Date(2023, 1, 2))
    ]
    result = fx_service.queries(queries)
    assert len(list(result)) == 2
    assert all(isinstance(r, FXRate) for r in result)

def test_queries_with_invalid_query_non_strict():
    fx_service = FXRateService.default
    queries = [
        (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)),
        (Currency("XYZ"), Currency("ABC"), Date(2023, 1, 2))
    ]
    result = fx_service.queries(queries, strict=False)
    assert len(list(result)) == 2
    assert isinstance(list(result)[0], FXRate)
    assert list(result)[1] is None

def test_queries_with_invalid_query_strict():
    fx_service = FXRateService.default
    queries = [
        (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)),
        (Currency("XYZ"), Currency("ABC"), Date(2023, 1, 2))
    ]
    try:
        fx_service.queries(queries, strict=True)
        assert False, "Expected lookup error not raised"
    except LookupError:
        pass


# LLM-generated content at query #26
#--------------------------

```python
def test_query_with_valid_currencies_and_date():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    fx_rate = Decimal("0.92")
    mock_service = MockFXRateService({(ccy1, ccy2, asof): fx_rate})
    assert mock_service.query(ccy1, ccy2, asof) == fx_rate

def test_query_with_invalid_currency_pair():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    mock_service = MockFXRateService({})
    assert mock_service.query(ccy1, ccy2, asof) is None

def test_query_with_strict_mode_enabled():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    mock_service = MockFXRateService({})
    with pytest.raises(LookupError):
        mock_service.query(ccy1, ccy2, asof, strict=True)


# LLM-generated content at query #27
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


# LLM-generated content at query #28
#--------------------------

```python
def test_fxrate_constructor_creates_instance_with_correct_properties():
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


# LLM-generated content at query #29
#--------------------------

```python
def test_queries_empty_input():
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None

        def queries(self, queries, strict=False):
            return []

    service = TestFXRateService()
    assert list(service.queries([])) == []

def test_queries_single_query():
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return Decimal("1.2345")

        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = TestFXRateService()
    result = list(service.queries([(Currency("USD"), Currency("EUR"), Date(2023, 1, 1))]))
    assert result == [Decimal("1.2345")]

def test_queries_multiple_queries():
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return Decimal("1.2345") if ccy1 == Currency("USD") else None

        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = TestFXRateService()
    queries = [
        (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)),
        (Currency("GBP"), Currency("JPY"), Date(2023, 1, 2))
    ]
    result = list(service.queries(queries))
    assert result == [Decimal("1.2345"), None]

def test_queries_strict_mode_raises_error():
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if strict and ccy1 == Currency("GBP"):
                raise LookupError("Rate not found")
            return None

        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = TestFXRateService()
    queries = [(Currency("GBP"), Currency("JPY"), Date(2023, 1, 1))]
    try:
        list(service.queries(queries, strict=True))
        assert False, "Expected LookupError"
    except LookupError:
        pass


# LLM-generated content at query #30
#--------------------------

```python
def test_query_returns_fxrate_for_valid_currency_pair_and_date():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    fxrate = Decimal("0.92")
    fx_rate_service = FXRateService()
    assert fx_rate_service.query(ccy1, ccy2, asof) == fxrate

def test_query_returns_none_for_invalid_currency_pair():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("XYZ", "Invalid Currency", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    fx_rate_service = FXRateService()
    assert fx_rate_service.query(ccy1, ccy2, asof) is None

def test_query_raises_error_for_invalid_date():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2099, 1, 1)
    fx_rate_service = FXRateService()
    with pytest.raises(LookupError):
        fx_rate_service.query(ccy1, ccy2, asof, strict=True)

def test_query_returns_none_for_invalid_date_without_strict():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2099, 1, 1)
    fx_rate_service = FXRateService()
    assert fx_rate_service.query(ccy1, ccy2, asof, strict=False) is None


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

def test_fxrate_constructor_allows_indexed_access():
    from pypara.currencies import Currencies
    import datetime
    from decimal import Decimal

    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date(2023, 1, 1), Decimal("1.2"))
    assert rate[0] == Currencies["EUR"]
    assert rate[1] == Currencies["USD"]
    assert rate[2] == datetime.date(2023, 1, 1)
    assert rate[3] == Decimal("1.2")


# LLM-generated content at query #32
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

def test_fxrate_constructor_allows_indexed_access():
    from decimal import Decimal
    import datetime
    from pypara.currencies import Currencies
    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date(2023, 1, 1), Decimal("1.2"))
    assert rate[0] == Currencies["EUR"]
    assert rate[1] == Currencies["USD"]
    assert rate[2] == datetime.date(2023, 1, 1)
    assert rate[3] == Decimal("1.2")


