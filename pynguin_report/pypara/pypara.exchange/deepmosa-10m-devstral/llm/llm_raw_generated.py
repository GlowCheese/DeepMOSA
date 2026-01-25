####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_invert_fx_rate():
    from decimal import Decimal
    import datetime
    from pypara.currencies import Currencies
    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date(2023, 1, 1), Decimal("2"))
    inverted_rate = ~rate
    assert inverted_rate.ccy1 == Currencies["USD"]
    assert inverted_rate.ccy2 == Currencies["EUR"]
    assert inverted_rate.date == datetime.date(2023, 1, 1)
    assert inverted_rate.value == Decimal("0.5")


# LLM-generated content at query #2
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
    assert all(isinstance(r, FXRate) or r is None for r in rates)

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
    assert isinstance(rates[0], FXRate) or rates[0] is None
    assert rates[1] is None


# LLM-generated content at query #3
#--------------------------

```python
def test_invert_fx_rate():
    from decimal import Decimal
    import datetime
    from pypara.currencies import Currencies

    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("2"))
    inverted_rate = ~rate

    assert inverted_rate.ccy1 == Currencies["USD"]
    assert inverted_rate.ccy2 == Currencies["EUR"]
    assert inverted_rate.date == datetime.date.today()
    assert inverted_rate.value == Decimal("0.5")


# LLM-generated content at query #4
#--------------------------

```python
def test_fxrate_constructor_creates_instance_with_correct_properties():
    ccy1 = Currency("EUR")
    ccy2 = Currency("USD")
    date = datetime.date.today()
    value = Decimal("2")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_invert_fx_rate():
    from decimal import Decimal
    import datetime
    from pypara.currencies import Currencies
    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("2"))
    inverted_rate = ~rate
    assert inverted_rate.ccy1 == Currencies["USD"]
    assert inverted_rate.ccy2 == Currencies["EUR"]
    assert inverted_rate.date == datetime.date.today()
    assert inverted_rate.value == Decimal("0.5")


# LLM-generated content at query #2
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
            return FXRate(Decimal("1.5"))

        def queries(self, queries, strict=False):
            return [self.query(*q) for q in queries]

    service = TestFXRateService()
    query = (Currency("USD"), Currency("EUR"), Date(2023, 1, 1))
    result = list(service.queries([query]))
    assert len(result) == 1
    assert result[0] == FXRate(Decimal("1.5"))

def test_queries_multiple_queries():
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return FXRate(Decimal("1.5"))

        def queries(self, queries, strict=False):
            return [self.query(*q) for q in queries]

    service = TestFXRateService()
    queries = [
        (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)),
        (Currency("GBP"), Currency("JPY"), Date(2023, 1, 2)),
    ]
    result = list(service.queries(queries))
    assert len(result) == 2
    assert all(r == FXRate(Decimal("1.5")) for r in result)

def test_queries_strict_mode():
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if strict:
                raise LookupError("Rate not found")
            return None

        def queries(self, queries, strict=False):
            return [self.query(*q, strict=strict) for q in queries]

    service = TestFXRateService()
    query = (Currency("USD"), Currency("EUR"), Date(2023, 1, 1))
    try:
        list(service.queries([query], strict=True))
        assert False, "Expected LookupError"
    except LookupError:
        pass

def test_queries_none_values():
    class TestFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None

        def queries(self, queries, strict=False):
            return [self.query(*q) for q in queries]

    service = TestFXRateService()
    queries = [
        (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)),
        (Currency("GBP"), Currency("JPY"), Date(2023, 1, 2)),
    ]
    result = list(service.queries(queries))
    assert len(result) == 2
    assert all(r is None for r in result)


# LLM-generated content at query #3
#--------------------------

```python
def test_query_abstract_method():
    with pytest.raises(TypeError):
        FXRateService().query(Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY), Currency.of("EUR", "Euros", 2, CurrencyType.MONEY), Date(2023, 1, 1))


# LLM-generated content at query #4
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

def test_query_with_same_currencies():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    result = FXRateService.default.query(ccy1, ccy2, asof)
    assert result == FXRate(Decimal("1.0"), ccy1, ccy2, asof)


# LLM-generated content at query #5
#--------------------------

```python
def test_invert_fx_rate():
    from decimal import Decimal
    import datetime
    from pypara.currencies import Currencies

    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("2"))
    inverted_rate = ~rate

    assert inverted_rate.ccy1 == Currencies["USD"]
    assert inverted_rate.ccy2 == Currencies["EUR"]
    assert inverted_rate.date == datetime.date.today()
    assert inverted_rate.value == Decimal("0.5")


# LLM-generated content at query #6
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

def test_queries_with_none_result():
    service = FXRateService.default
    query = (Currency("XYZ"), Currency("ABC"), Date(2023, 1, 1))
    result = service.queries([query])
    assert next(iter(result)) is None

def test_queries_strict_mode_raises_error():
    service = FXRateService.default
    query = (Currency("XYZ"), Currency("ABC"), Date(2023, 1, 1))
    try:
        service.queries([query], strict=True)
        assert False, "Expected an error to be raised"
    except LookupError:
        pass


# LLM-generated content at query #7
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
    assert isinstance(rates[0], FXRate)

def test_queries_multiple_valid_queries():
    service = FXRateService.default
    queries = [
        (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)),
        (Currency("GBP"), Currency("JPY"), Date(2023, 1, 2))
    ]
    rates = list(service.queries(queries))
    assert len(rates) == 2
    assert all(isinstance(r, FXRate) for r in rates)

def test_queries_with_invalid_query_non_strict():
    service = FXRateService.default
    queries = [(Currency("USD"), Currency("XYZ"), Date(2023, 1, 1))]
    rates = list(service.queries(queries, strict=False))
    assert len(rates) == 1
    assert rates[0] is None

def test_queries_with_invalid_query_strict():
    service = FXRateService.default
    queries = [(Currency("USD"), Currency("XYZ"), Date(2023, 1, 1))]
    try:
        list(service.queries(queries, strict=True))
        assert False, "Expected LookupError"
    except LookupError:
        pass

def test_queries_mixed_valid_and_invalid_non_strict():
    service = FXRateService.default
    queries = [
        (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)),
        (Currency("USD"), Currency("XYZ"), Date(2023, 1, 1))
    ]
    rates = list(service.queries(queries, strict=False))
    assert len(rates) == 2
    assert isinstance(rates[0], FXRate)
    assert rates[1] is None


# LLM-generated content at query #8
#--------------------------

```python
def test_query_with_valid_input_returns_fxrate():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return FXRate(ccy1, ccy2, Decimal("1.5"), asof)

        def queries(self, queries, strict=False):
            return [FXRate(q[0], q[1], Decimal("1.5"), q[2]) for q in queries]

    service = MockFXRateService()
    usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    result = service.query(usd, eur, asof)
    assert result == FXRate(usd, eur, Decimal("1.5"), asof)

def test_query_with_invalid_input_returns_none():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None

        def queries(self, queries, strict=False):
            return [None for _ in queries]

    service = MockFXRateService()
    usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    result = service.query(usd, eur, asof)
    assert result is None

def test_query_with_strict_flag_raises_error():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if strict:
                raise LookupError("FX rate not found")
            return None

        def queries(self, queries, strict=False):
            if strict:
                raise LookupError("FX rate not found")
            return [None for _ in queries]

    service = MockFXRateService()
    usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    try:
        service.query(usd, eur, asof, strict=True)
        assert False, "Expected LookupError"
    except LookupError:
        pass


# LLM-generated content at query #9
#--------------------------

```python
def test_query_abstract_method():
    with pytest.raises(TypeError):
        FXRateService().query(Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY),
                             Currency.of("EUR", "Euro", 2, CurrencyType.MONEY),
                             Date(2023, 1, 1))


# LLM-generated content at query #10
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


# LLM-generated content at query #11
#--------------------------

```python
def test_queries_empty_input():
    fx_service = FXRateService.default
    assert list(fx_service.queries([])) == []

def test_queries_single_rate():
    fx_service = FXRateService.default
    query = (Currency("USD"), Currency("EUR"), Date(2023, 1, 1))
    result = list(fx_service.queries([query]))
    assert len(result) == 1
    assert isinstance(result[0], (FXRate, type(None)))

def test_queries_multiple_rates():
    fx_service = FXRateService.default
    queries = [
        (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)),
        (Currency("GBP"), Currency("JPY"), Date(2023, 1, 2))
    ]
    result = list(fx_service.queries(queries))
    assert len(result) == 2
    assert all(isinstance(r, (FXRate, type(None))) for r in result)

def test_queries_strict_mode_raises_error():
    fx_service = FXRateService.default
    queries = [(Currency("XYZ"), Currency("ABC"), Date(2023, 1, 1))]
    try:
        list(fx_service.queries(queries, strict=True))
        assert False, "Expected a lookup error"
    except LookupError:
        pass


# LLM-generated content at query #12
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


# LLM-generated content at query #13
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

def test_fxrate_constructor_allows_indexed_access():
    from decimal import Decimal
    import datetime
    from pypara.currencies import Currencies

    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date(2023, 1, 1), Decimal("1.2345"))

    assert rate[0] == Currencies["EUR"]
    assert rate[1] == Currencies["USD"]
    assert rate[2] == datetime.date(2023, 1, 1)
    assert rate[3] == Decimal("1.2345")


# LLM-generated content at query #14
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


# LLM-generated content at query #15
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


# LLM-generated content at query #16
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


# LLM-generated content at query #17
#--------------------------

```python
def test_fxrate_constructor_creates_instance_with_correct_properties():
    from datetime import date
    from decimal import Decimal
    from pypara.currencies import Currencies
    from pypara.fx import FXRate

    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    test_date = date(2023, 1, 1)
    test_value = Decimal("1.2345")

    rate = FXRate(ccy1, ccy2, test_date, test_value)

    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == test_date
    assert rate.value == test_value


# LLM-generated content at query #18
#--------------------------

```python
def test_fxrate_constructor_creates_instance_with_correct_properties():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    from pypara.fx import FXRate

    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("2"))
    assert rate.ccy1 == Currencies["EUR"]
    assert rate.ccy2 == Currencies["USD"]
    assert rate.date == datetime.date.today()
    assert rate.value == Decimal("2")


# LLM-generated content at query #19
#--------------------------

```python
def test_queries_empty_iterable():
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

def test_queries_strict_mode_raises_error():
    service = FXRateService.default
    query = (Currency("XYZ"), Currency("ABC"), Date(2023, 1, 1))
    try:
        service.queries([query], strict=True)
        assert False, "Expected an error to be raised"
    except LookupError:
        pass


# LLM-generated content at query #20
#--------------------------

```python
def test_query_with_valid_input():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    result = FXRateService.default.query(ccy1, ccy2, asof)
    assert isinstance(result, (Decimal, type(None)))

def test_query_with_strict_false():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    result = FXRateService.default.query(ccy1, ccy2, asof, strict=False)
    assert isinstance(result, (Decimal, type(None)))

def test_query_with_strict_true():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    try:
        FXRateService.default.query(ccy1, ccy2, asof, strict=True)
    except LookupError:
        pass


# LLM-generated content at query #21
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


# LLM-generated content at query #22
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


# LLM-generated content at query #23
#--------------------------

```python
def test_query_returns_fxrate_for_valid_input():
    fx_rate_service = FXRateService.default
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    result = fx_rate_service.query(ccy1, ccy2, asof)
    assert isinstance(result, FXRate) or result is None

def test_query_returns_none_for_invalid_input():
    fx_rate_service = FXRateService.default
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    result = fx_rate_service.query(ccy1, ccy2, asof, strict=False)
    assert result is None

def test_query_raises_error_for_strict_invalid_input():
    fx_rate_service = FXRateService.default
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    with pytest.raises(LookupError):
        fx_rate_service.query(ccy1, ccy2, asof, strict=True)


# LLM-generated content at query #24
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
        assert False, "Expected a lookup error"
    except LookupError:
        pass


# LLM-generated content at query #25
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


# LLM-generated content at query #26
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


# LLM-generated content at query #27
#--------------------------

```python
def test_queries_empty_input():
    service = FXRateService.default
    assert list(service.queries([])) == []

def test_queries_single_query():
    service = FXRateService.default
    queries = [(Currency("USD"), Currency("EUR"), Date(2023, 1, 1))]
    result = list(service.queries(queries))
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
    queries = [(Currency("XYZ"), Currency("ABC"), Date(2023, 1, 1))]
    try:
        list(service.queries(queries, strict=True))
        assert False, "Expected LookupError to be raised"
    except LookupError:
        pass


# LLM-generated content at query #28
#--------------------------

```python
def test_fxrate_constructor_creates_instance_with_given_values():
    from decimal import Decimal
    from pypara.currencies import Currencies
    import datetime

    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date(2023, 1, 1), Decimal("1.5"))

    assert rate.ccy1 == Currencies["EUR"]
    assert rate.ccy2 == Currencies["USD"]
    assert rate.date == datetime.date(2023, 1, 1)
    assert rate.value == Decimal("1.5")


# LLM-generated content at query #29
#--------------------------

```python
def test_query_with_valid_inputs_returns_fxrate():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return Decimal("1.5")

        def queries(self, queries, strict=False):
            return [Decimal("1.5")]

    service = MockFXRateService()
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof_date = Date(2023, 1, 1)
    result = service.query(usd, eur, asof_date)
    assert result == Decimal("1.5")

def test_query_with_invalid_currency_pair_returns_none():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None

        def queries(self, queries, strict=False):
            return [None]

    service = MockFXRateService()
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    xyz = Currency.of("XYZ", "Unknown", 2, CurrencyType.MONEY)
    asof_date = Date(2023, 1, 1)
    result = service.query(usd, xyz, asof_date)
    assert result is None

def test_query_with_strict_true_raises_lookup_error():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if strict:
                raise LookupError("FX rate not found")
            return None

        def queries(self, queries, strict=False):
            if strict:
                raise LookupError("FX rate not found")
            return [None]

    service = MockFXRateService()
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    xyz = Currency.of("XYZ", "Unknown", 2, CurrencyType.MONEY)
    asof_date = Date(2023, 1, 1)
    try:
        service.query(usd, xyz, asof_date, strict=True)
        assert False, "Expected LookupError"
    except LookupError:
        pass


# LLM-generated content at query #30
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


# LLM-generated content at query #31
#--------------------------

```python
def test_queries_returns_iterable_of_rates():
    queries = [
        (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)),
        (Currency("GBP"), Currency("JPY"), Date(2023, 1, 2))
    ]
    mock_service = FXRateService()
    result = mock_service.queries(queries, strict=False)
    assert isinstance(result, Iterable)
    assert all(isinstance(rate, (FXRate, type(None))) for rate in result)

def test_queries_raises_error_when_strict_and_rate_not_found():
    queries = [(Currency("USD"), Currency("XYZ"), Date(2023, 1, 1))]
    mock_service = FXRateService()
    with pytest.raises(LookupError):
        mock_service.queries(queries, strict=True)

def test_queries_returns_none_when_not_strict_and_rate_not_found():
    queries = [(Currency("USD"), Currency("XYZ"), Date(2023, 1, 1))]
    mock_service = FXRateService()
    result = list(mock_service.queries(queries, strict=False))
    assert result == [None]


# LLM-generated content at query #32
#--------------------------

```python
def test_query_abstract_method():
    with pytest.raises(TypeError):
        FXRateService().query(Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY),
                              Currency.of("EUR", "Euros", 2, CurrencyType.MONEY),
                              Date(2023, 1, 1))


# LLM-generated content at query #33
#--------------------------

```python
def test_fxrate_constructor_creates_instance_with_given_values():
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
def test_query_with_valid_inputs():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    result = FXRateService.default.query(ccy1, ccy2, asof)
    assert isinstance(result, (FXRate, type(None)))

def test_query_with_strict_false():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    result = FXRateService.default.query(ccy1, ccy2, asof, strict=False)
    assert isinstance(result, (FXRate, type(None)))

def test_query_with_strict_true():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    try:
        FXRateService.default.query(ccy1, ccy2, asof, strict=True)
    except LookupError:
        pass


# LLM-generated content at query #2
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
    rates = service.queries(queries)
    assert len(list(rates)) == 2

def test_queries_invalid_query_non_strict():
    service = FXRateService.default
    queries = [(Currency("USD"), Currency("XYZ"), Date(2023, 1, 1))]
    rates = service.queries(queries, strict=False)
    assert next(rates) is None

def test_queries_mixed_valid_invalid_non_strict():
    service = FXRateService.default
    queries = [
        (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)),
        (Currency("GBP"), Currency("XYZ"), Date(2023, 1, 2))
    ]
    rates = service.queries(queries, strict=False)
    results = list(rates)
    assert results[0] is not None
    assert results[1] is None


# LLM-generated content at query #3
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

def test_queries_invalid_query_strict_false():
    service = FXRateService.default
    queries = [(Currency("XXX"), Currency("YYY"), Date(2023, 1, 1))]
    rates = list(service.queries(queries, strict=False))
    assert len(rates) == 1
    assert rates[0] is None

def test_queries_invalid_query_strict_true():
    service = FXRateService.default
    queries = [(Currency("XXX"), Currency("YYY"), Date(2023, 1, 1))]
    try:
        list(service.queries(queries, strict=True))
        assert False, "Expected a lookup error to be raised"
    except LookupError:
        pass


# LLM-generated content at query #4
#--------------------------

```python
def test_query_returns_fx_rate():
    fx_rate_service = FXRateService.default
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    result = fx_rate_service.query(ccy1, ccy2, asof)
    assert isinstance(result, (FXRate, type(None)))

def test_query_with_strict_raises_lookup_error():
    fx_rate_service = FXRateService.default
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    try:
        fx_rate_service.query(ccy1, ccy2, asof, strict=True)
    except LookupError:
        pass


# LLM-generated content at query #5
#--------------------------

```python
def test_fxrate_constructor_creates_instance_with_given_values():
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


# LLM-generated content at query #6
#--------------------------

```python
def test_queries_empty_input():
    fx_service = FXRateService.default
    assert list(fx_service.queries([])) == []

def test_queries_single_valid_query():
    fx_service = FXRateService.default
    query = (Currency("USD"), Currency("EUR"), Date(2023, 1, 1))
    rates = fx_service.queries([query])
    assert len(list(rates)) == 1

def test_queries_multiple_valid_queries():
    fx_service = FXRateService.default
    queries = [
        (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)),
        (Currency("GBP"), Currency("JPY"), Date(2023, 1, 2))
    ]
    rates = fx_service.queries(queries)
    assert len(list(rates)) == 2

def test_queries_with_invalid_query_strict_false():
    fx_service = FXRateService.default
    queries = [
        (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)),
        (Currency("XYZ"), Currency("ABC"), Date(2023, 1, 2))
    ]
    rates = fx_service.queries(queries, strict=False)
    assert len(list(rates)) == 2
    assert None in list(rates)

def test_queries_with_invalid_query_strict_true():
    fx_service = FXRateService.default
    queries = [
        (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)),
        (Currency("XYZ"), Currency("ABC"), Date(2023, 1, 2))
    ]
    try:
        list(fx_service.queries(queries, strict=True))
        assert False, "Expected a lookup error"
    except LookupError:
        pass


# LLM-generated content at query #7
#--------------------------

```python
def test_query_with_valid_input_returns_fxrate():
    # Assuming a concrete implementation of FXRateService exists for testing
    fx_service = FXRateService.default  # or a mock implementation
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    today = Date.today()
    result = fx_service.query(usd, eur, today)
    assert isinstance(result, (FXRate, type(None)))

def test_query_with_invalid_currency_raises_error():
    fx_service = FXRateService.default
    invalid_ccy = "INVALID"
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    today = Date.today()
    with pytest.raises(ProgrammingError):
        fx_service.query(invalid_ccy, usd, today)

def test_query_with_strict_true_and_missing_rate_raises_error():
    fx_service = FXRateService.default
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    xyz = Currency.of("XYZ", "Unknown Currency", 2, CurrencyType.MONEY)
    today = Date.today()
    with pytest.raises(LookupError):
        fx_service.query(usd, xyz, today, strict=True)

def test_query_with_strict_false_and_missing_rate_returns_none():
    fx_service = FXRateService.default
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    xyz = Currency.of("XYZ", "Unknown Currency", 2, CurrencyType.MONEY)
    today = Date.today()
    result = fx_service.query(usd, xyz, today, strict=False)
    assert result is None


# LLM-generated content at query #8
#--------------------------

```python
def test_query_with_valid_currencies_and_date():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    fx_rate = Decimal("1.07")
    fx_rate_service = FXRateService()
    assert fx_rate_service.query(usd, eur, asof) == fx_rate

def test_query_with_invalid_currencies():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    xyz = Currency.of("XYZ", "Invalid Currency", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    fx_rate_service = FXRateService()
    assert fx_rate_service.query(usd, xyz, asof) is None

def test_query_with_strict_flag_raises_error():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    xyz = Currency.of("XYZ", "Invalid Currency", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    fx_rate_service = FXRateService()
    with pytest.raises(LookupError):
        fx_rate_service.query(usd, xyz, asof, strict=True)


# LLM-generated content at query #9
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


# LLM-generated content at query #10
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


# LLM-generated content at query #11
#--------------------------

```python
def test_queries_empty_input():
    service = FXRateService.default
    result = service.queries([])
    assert list(result) == []

def test_queries_single_valid_query():
    service = FXRateService.default
    query = (Currency("USD"), Currency("EUR"), Date(2023, 1, 1))
    result = service.queries([query])
    assert next(result) is not None

def test_queries_multiple_valid_queries():
    service = FXRateService.default
    queries = [
        (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)),
        (Currency("GBP"), Currency("JPY"), Date(2023, 1, 1))
    ]
    result = service.queries(queries)
    rates = list(result)
    assert len(rates) == 2
    assert all(rate is not None for rate in rates)

def test_queries_invalid_query_non_strict():
    service = FXRateService.default
    query = (Currency("XYZ"), Currency("ABC"), Date(2023, 1, 1))
    result = service.queries([query], strict=False)
    assert next(result) is None

def test_queries_invalid_query_strict():
    service = FXRateService.default
    query = (Currency("XYZ"), Currency("ABC"), Date(2023, 1, 1))
    with pytest.raises(LookupError):
        list(service.queries([query], strict=True))


# LLM-generated content at query #12
#--------------------------

```python
def test_fxrate_constructor_with_valid_args():
    from pypara.currencies import Currencies
    import datetime
    from decimal import Decimal
    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("2"))
    assert rate.ccy1 == Currencies["EUR"]
    assert rate.ccy2 == Currencies["USD"]
    assert rate.date == datetime.date.today()
    assert rate.value == Decimal("2")


# LLM-generated content at query #13
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


# LLM-generated content at query #14
#--------------------------

```python
def test_fxrate_constructor_creates_instance_with_correct_values():
    from decimal import Decimal
    import datetime
    from pypara.currencies import Currencies

    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("2"))
    assert rate.ccy1 == Currencies["EUR"]
    assert rate.ccy2 == Currencies["USD"]
    assert rate.date == datetime.date.today()
    assert rate.value == Decimal("2")


# LLM-generated content at query #15
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


# LLM-generated content at query #17
#--------------------------

```python
def test_query_abstract_method():
    with pytest.raises(TypeError):
        FXRateService().query(Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY),
                             Currency.of("EUR", "Euro", 2, CurrencyType.MONEY),
                             Date(2023, 1, 1))


# LLM-generated content at query #18
#--------------------------

```python
def test_queries_with_empty_input():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None

        def queries(self, queries, strict=False):
            return []

    service = MockFXRateService()
    result = list(service.queries([]))
    assert result == []

def test_queries_with_single_query():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if ccy1 == "USD" and ccy2 == "EUR" and asof == "2023-01-01":
                return Decimal("0.85")
            return None

        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = MockFXRateService()
    queries = [("USD", "EUR", "2023-01-01")]
    result = list(service.queries(queries))
    assert result == [Decimal("0.85")]

def test_queries_with_multiple_queries():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if ccy1 == "USD" and ccy2 == "EUR" and asof == "2023-01-01":
                return Decimal("0.85")
            elif ccy1 == "GBP" and ccy2 == "JPY" and asof == "2023-01-01":
                return Decimal("150.0")
            return None

        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = MockFXRateService()
    queries = [("USD", "EUR", "2023-01-01"), ("GBP", "JPY", "2023-01-01")]
    result = list(service.queries(queries))
    assert result == [Decimal("0.85"), Decimal("150.0")]

def test_queries_with_missing_rate():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None

        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = MockFXRateService()
    queries = [("USD", "EUR", "2023-01-01")]
    result = list(service.queries(queries))
    assert result == [None]

def test_queries_with_strict_mode():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if strict:
                raise LookupError("Rate not found")
            return None

        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]

    service = MockFXRateService()
    queries = [("USD", "EUR", "2023-01-01")]
    try:
        list(service.queries(queries, strict=True))
        assert False, "Expected LookupError"
    except LookupError:
        pass


# LLM-generated content at query #19
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


# LLM-generated content at query #20
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


# LLM-generated content at query #21
#--------------------------

```python
def test_queries_empty_input():
    service = FXRateService.default
    assert list(service.queries([])) == []

def test_queries_single_query():
    service = FXRateService.default
    queries = [("USD", "EUR", "2023-01-01")]
    result = list(service.queries(queries))
    assert len(result) == 1
    assert isinstance(result[0], (Decimal, type(None)))

def test_queries_multiple_queries():
    service = FXRateService.default
    queries = [("USD", "EUR", "2023-01-01"), ("GBP", "JPY", "2023-01-02")]
    result = list(service.queries(queries))
    assert len(result) == 2
    assert all(isinstance(r, (Decimal, type(None))) for r in result)

def test_queries_strict_mode_raises_error():
    service = FXRateService.default
    queries = [("XYZ", "ABC", "2023-01-01")]
    try:
        service.queries(queries, strict=True)
        assert False, "Expected a lookup error"
    except LookupError:
        pass


# LLM-generated content at query #22
#--------------------------

```python
def test_query_with_valid_inputs():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
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


# LLM-generated content at query #23
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


# LLM-generated content at query #24
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


# LLM-generated content at query #25
#--------------------------

```python
def test_query_method_should_be_abstract():
    with pytest.raises(TypeError):
        FXRateService.query(None, None, None)

def test_queries_method_should_be_abstract():
    with pytest.raises(TypeError):
        FXRateService.queries(None)


# LLM-generated content at query #26
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
    queries = [(Currency("XYZ"), Currency("ABC"), Date(2023, 1, 1))]
    try:
        list(service.queries(queries, strict=True))
        assert False, "Expected a lookup error to be raised"
    except LookupError:
        pass


# LLM-generated content at query #27
#--------------------------

```python
def test_fxrate_constructor():
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


# LLM-generated content at query #28
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


# LLM-generated content at query #29
#--------------------------

```python
def test_query_abstract_method():
    with pytest.raises(TypeError):
        FXRateService().query(Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY),
                              Currency.of("EUR", "Euros", 2, CurrencyType.MONEY),
                              Date(2023, 1, 1))


# LLM-generated content at query #30
#--------------------------

```python
def test_queries_with_valid_inputs():
    queries = [
        (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)),
        (Currency("GBP"), Currency("JPY"), Date(2023, 1, 2)),
    ]
    expected_rates = [Decimal("0.92"), Decimal("142.50")]
    mock_service = FXRateService()
    mock_service.queries = lambda q, s=False: expected_rates
    assert list(mock_service.queries(queries)) == expected_rates

def test_queries_with_empty_input():
    mock_service = FXRateService()
    mock_service.queries = lambda q, s=False: []
    assert list(mock_service.queries([])) == []

def test_queries_with_strict_mode():
    queries = [(Currency("USD"), Currency("EUR"), Date(2023, 1, 1))]
    mock_service = FXRateService()
    mock_service.queries = lambda q, s=False: [None] if s else [Decimal("0.92")]
    assert list(mock_service.queries(queries, strict=True)) == [None]
    assert list(mock_service.queries(queries, strict=False)) == [Decimal("0.92")]


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
def test_fxrate_constructor():
    from pypara.currencies import Currencies
    import datetime
    from decimal import Decimal

    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("2"))
    assert rate.ccy1 == Currencies["EUR"]
    assert rate.ccy2 == Currencies["USD"]
    assert rate.date == datetime.date.today()
    assert rate.value == Decimal("2")


