####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_invert_swaps_currencies_and_inverts_value():
    from pypara.currencies import Currencies
    import datetime
    from decimal import Decimal
    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date(2023, 1, 1), Decimal("2"))
    inverted = ~rate
    assert inverted.ccy1 == Currencies["USD"]
    assert inverted.ccy2 == Currencies["EUR"]
    assert inverted.date == datetime.date(2023, 1, 1)
    assert inverted.value == Decimal("0.5")

def test_invert_twice_returns_original():
    from pypara.currencies import Currencies
    import datetime
    from decimal import Decimal
    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date(2023, 1, 1), Decimal("2"))
    inverted = ~rate
    original = ~inverted
    assert original == rate

def test_invert_with_fractional_value():
    from pypara.currencies import Currencies
    import datetime
    from decimal import Decimal
    rate = FXRate(Currencies["USD"], Currencies["EUR"], datetime.date(2023, 1, 1), Decimal("0.75"))
    inverted = ~rate
    assert inverted.ccy1 == Currencies["EUR"]
    assert inverted.ccy2 == Currencies["USD"]
    assert inverted.value == Decimal("1") / Decimal("0.75")

def test_invert_with_one():
    from pypara.currencies import Currencies
    import datetime
    from decimal import Decimal
    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date(2023, 1, 1), Decimal("1"))
    inverted = ~rate
    assert inverted.ccy1 == Currencies["USD"]
    assert inverted.ccy2 == Currencies["EUR"]
    assert inverted.value == Decimal("1")

def test_invert_uses_indexed_access():
    from pypara.currencies import Currencies
    import datetime
    from decimal import Decimal
    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date(2023, 1, 1), Decimal("2"))
    inverted = rate.__invert__()
    assert inverted[0] == rate[1]
    assert inverted[1] == rate[0]
    assert inverted[2] == rate[2]
    assert inverted[3] == rate[3] ** -1


# LLM-generated content at query #2
#--------------------------

def test_queries_returns_correct_rates_for_multiple_queries():
    mock_service = create_autospec(FXRateService)
    mock_service.queries.return_value = [Decimal('1.1'), Decimal('1.2'), None]
    queries = [(Currency('USD'), Currency('EUR'), Date(2023, 1, 1)), (Currency('GBP'), Currency('JPY'), Date(2023, 1, 2)), (Currency('AUD'), Currency('CAD'), Date(2023, 1, 3))]
    result = list(mock_service.queries(queries, strict=False))
    assert result == [Decimal('1.1'), Decimal('1.2'), None]

def test_queries_with_strict_true_raises_lookup_error_on_missing_rate():
    mock_service = create_autospec(FXRateService)
    mock_service.queries.side_effect = LookupError("FX rate not found")
    queries = [(Currency('USD'), Currency('EUR'), Date(2023, 1, 1))]
    try:
        list(mock_service.queries(queries, strict=True))
        assert False, "Expected LookupError"
    except LookupError:
        pass

def test_queries_with_empty_iterable_returns_empty_iterable():
    mock_service = create_autospec(FXRateService)
    mock_service.queries.return_value = []
    result = list(mock_service.queries([], strict=False))
    assert result == []

def test_queries_calls_with_correct_parameters():
    mock_service = create_autospec(FXRateService)
    mock_service.queries.return_value = []
    queries = [(Currency('USD'), Currency('EUR'), Date(2023, 1, 1))]
    list(mock_service.queries(queries, strict=True))
    mock_service.queries.assert_called_once_with(queries, strict=True)

def test_queries_returns_iterable_of_same_length_as_input():
    mock_service = create_autospec(FXRateService)
    mock_service.queries.return_value = [Decimal('1.0'), Decimal('2.0'), Decimal('3.0')]
    queries = [(Currency('USD'), Currency('EUR'), Date(2023, 1, 1)), (Currency('GBP'), Currency('JPY'), Date(2023, 1, 2)), (Currency('AUD'), Currency('CAD'), Date(2023, 1, 3))]
    result = list(mock_service.queries(queries, strict=False))
    assert len(result) == 3


# LLM-generated content at query #3
#--------------------------

def test_query_returns_fxrate_for_valid_currency_pair_and_date():
    service = FXRateService()
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    result = service.query(ccy1, ccy2, asof, strict=False)
    assert result is not None
    assert isinstance(result, FXRate)

def test_query_returns_none_for_nonexistent_pair_when_strict_false():
    service = FXRateService()
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("XYZ", "Unknown", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    result = service.query(ccy1, ccy2, asof, strict=False)
    assert result is None

def test_query_raises_error_for_nonexistent_pair_when_strict_true():
    service = FXRateService()
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("XYZ", "Unknown", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    try:
        service.query(ccy1, ccy2, asof, strict=True)
        assert False
    except LookupError:
        assert True

def test_query_handles_same_currency_pair():
    service = FXRateService()
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    result = service.query(ccy1, ccy2, asof, strict=False)
    assert result is not None
    assert result.rate == Decimal("1")

def test_query_returns_correct_rate_for_inverse_pair():
    service = FXRateService()
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    rate_usd_eur = service.query(ccy1, ccy2, asof, strict=False)
    rate_eur_usd = service.query(ccy2, ccy1, asof, strict=False)
    assert rate_usd_eur is not None
    assert rate_eur_usd is not None
    assert rate_eur_usd.rate == Decimal("1") / rate_usd_eur.rate

def test_query_handles_different_currency_types():
    service = FXRateService()
    ccy1 = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    ccy2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    result = service.query(ccy1, ccy2, asof, strict=False)
    assert result is not None
    assert isinstance(result, FXRate)

def test_query_with_future_date_returns_none():
    service = FXRateService()
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2050, 1, 1)
    result = service.query(ccy1, ccy2, asof, strict=False)
    assert result is None

def test_query_with_past_date_returns_rate():
    service = FXRateService()
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2000, 1, 1)
    result = service.query(ccy1, ccy2, asof, strict=False)
    assert result is not None
    assert isinstance(result, FXRate)

def test_query_handles_currency_with_negative_decimals():
    service = FXRateService()
    ccy1 = Currency.of("ZZZ", "Weird", -1, CurrencyType.CRYPTO)
    ccy2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    result = service.query(ccy1, ccy2, asof, strict=False)
    assert result is not None
    assert isinstance(result, FXRate)

def test_query_handles_currency_with_zero_decimals():
    service = FXRateService()
    ccy1 = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    ccy2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    result = service.query(ccy1, ccy2, asof, strict=False)
    assert result is not None
    assert isinstance(result, FXRate)


# LLM-generated content at query #4
#--------------------------

def test_queries_returns_correct_rates_for_multiple_queries():
    service = FXRateService.default
    queries = [(Currency("USD"), Currency("EUR"), Date(2023, 1, 1)), (Currency("GBP"), Currency("JPY"), Date(2023, 1, 2))]
    results = list(service.queries(queries, strict=False))
    assert len(results) == 2
    assert isinstance(results[0], FXRate) or results[0] is None
    assert isinstance(results[1], FXRate) or results[1] is None

def test_queries_raises_lookup_error_in_strict_mode_when_rate_missing():
    service = FXRateService.default
    queries = [(Currency("USD"), Currency("XYZ"), Date(2023, 1, 1))]
    try:
        list(service.queries(queries, strict=True))
        assert False, "Expected LookupError"
    except LookupError:
        pass

def test_queries_returns_none_for_missing_rate_in_non_strict_mode():
    service = FXRateService.default
    queries = [(Currency("USD"), Currency("XYZ"), Date(2023, 1, 1))]
    results = list(service.queries(queries, strict=False))
    assert results[0] is None

def test_queries_handles_empty_queries_iterable():
    service = FXRateService.default
    results = list(service.queries([], strict=False))
    assert len(results) == 0

def test_queries_preserves_order_of_input_queries():
    service = FXRateService.default
    queries = [(Currency("USD"), Currency("EUR"), Date(2023, 1, 1)), (Currency("EUR"), Currency("USD"), Date(2023, 1, 1))]
    results = list(service.queries(queries, strict=False))
    assert len(results) == 2
    rate1 = service.query(Currency("USD"), Currency("EUR"), Date(2023, 1, 1), strict=False)
    rate2 = service.query(Currency("EUR"), Currency("USD"), Date(2023, 1, 1), strict=False)
    assert results[0] == rate1
    assert results[1] == rate2


# LLM-generated content at query #5
#--------------------------

def test_fxrate_constructor_creates_instance_with_correct_attributes():
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

def test_fxrate_constructor_allows_unpacking():
    from pypara.currencies import Currencies
    import datetime
    from decimal import Decimal
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

def test_fxrate_constructor_accepts_same_currency_with_value_one():
    from pypara.currencies import Currencies
    import datetime
    from decimal import Decimal
    ccy = Currencies["EUR"]
    date = datetime.date.today()
    value = Decimal("1")
    rate = FXRate(ccy, ccy, date, value)
    assert rate.ccy1 == ccy
    assert rate.ccy2 == ccy
    assert rate.value == value

def test_fxrate_constructor_accepts_positive_decimal_value():
    from pypara.currencies import Currencies
    import datetime
    from decimal import Decimal
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("0.5")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.value == value

def test_fxrate_constructor_does_not_validate_input():
    from pypara.currencies import Currencies
    import datetime
    from decimal import Decimal
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    invalid_value = Decimal("0")
    rate = FXRate(ccy1, ccy2, date, invalid_value)
    assert rate.value == invalid_value

def test_fxrate_constructor_does_not_enforce_same_currency_rule():
    from pypara.currencies import Currencies
    import datetime
    from decimal import Decimal
    ccy = Currencies["EUR"]
    date = datetime.date.today()
    invalid_value = Decimal("2")
    rate = FXRate(ccy, ccy, date, invalid_value)
    assert rate.value == invalid_value


# LLM-generated content at query #6
#--------------------------

def test_queries_returns_correct_rates_for_multiple_queries():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if (ccy1, ccy2, asof) == (Currency("USD"), Currency("EUR"), Date(2023, 1, 1)):
                return FXRate(Decimal("0.85"))
            if (ccy1, ccy2, asof) == (Currency("GBP"), Currency("JPY"), Date(2023, 1, 2)):
                return FXRate(Decimal("150.0"))
            return None
        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    service = MockFXRateService()
    queries = [(Currency("USD"), Currency("EUR"), Date(2023, 1, 1)), (Currency("GBP"), Currency("JPY"), Date(2023, 1, 2))]
    results = list(service.queries(queries))
    assert results == [FXRate(Decimal("0.85")), FXRate(Decimal("150.0"))]

def test_queries_returns_none_for_missing_rates_when_strict_false():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    service = MockFXRateService()
    queries = [(Currency("USD"), Currency("EUR"), Date(2023, 1, 1))]
    results = list(service.queries(queries, strict=False))
    assert results == [None]

def test_queries_raises_error_for_missing_rates_when_strict_true():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if strict:
                raise LookupError("Rate not found")
            return None
        def queries(self, queries, strict=False):
            results = []
            for ccy1, ccy2, asof in queries:
                try:
                    results.append(self.query(ccy1, ccy2, asof, strict))
                except LookupError:
                    raise LookupError("Rate not found")
            return results
    service = MockFXRateService()
    queries = [(Currency("USD"), Currency("EUR"), Date(2023, 1, 1))]
    try:
        list(service.queries(queries, strict=True))
        assert False
    except LookupError:
        assert True

def test_queries_handles_empty_queries_list():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return FXRate(Decimal("1.0"))
        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    service = MockFXRateService()
    queries = []
    results = list(service.queries(queries))
    assert results == []

def test_queries_preserves_order_of_input_queries():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            rates = {
                (Currency("A"), Currency("B"), Date(2023, 1, 1)): FXRate(Decimal("1.1")),
                (Currency("C"), Currency("D"), Date(2023, 1, 2)): FXRate(Decimal("2.2")),
                (Currency("E"), Currency("F"), Date(2023, 1, 3)): FXRate(Decimal("3.3"))
            }
            return rates.get((ccy1, ccy2, asof))
        def queries(self, queries, strict=False):
            return [self.query(ccy1, ccy2, asof, strict) for ccy1, ccy2, asof in queries]
    service = MockFXRateService()
    queries = [
        (Currency("C"), Currency("D"), Date(2023, 1, 2)),
        (Currency("A"), Currency("B"), Date(2023, 1, 1)),
        (Currency("E"), Currency("F"), Date(2023, 1, 3))
    ]
    results = list(service.queries(queries))
    assert results == [FXRate(Decimal("2.2")), FXRate(Decimal("1.1")), FXRate(Decimal("3.3"))]


# LLM-generated content at query #7
#--------------------------

def test_queries_returns_correct_rates():
    service = FXRateService.default
    queries = [(Currency("USD"), Currency("EUR"), Date(2023, 1, 1)), (Currency("GBP"), Currency("JPY"), Date(2023, 1, 2))]
    results = list(service.queries(queries, strict=False))
    assert len(results) == 2
    assert results[0] is not None
    assert results[1] is not None

def test_queries_with_strict_mode_raises_error():
    service = FXRateService.default
    queries = [(Currency("USD"), Currency("XYZ"), Date(2023, 1, 1))]
    try:
        list(service.queries(queries, strict=True))
        assert False
    except LookupError:
        assert True

def test_queries_with_strict_mode_false_returns_none():
    service = FXRateService.default
    queries = [(Currency("USD"), Currency("XYZ"), Date(2023, 1, 1))]
    results = list(service.queries(queries, strict=False))
    assert len(results) == 1
    assert results[0] is None

def test_queries_empty_list():
    service = FXRateService.default
    results = list(service.queries([], strict=False))
    assert len(results) == 0

def test_queries_consistent_with_single_query():
    service = FXRateService.default
    single_rate = service.query(Currency("USD"), Currency("EUR"), Date(2023, 1, 1), strict=False)
    queries = [(Currency("USD"), Currency("EUR"), Date(2023, 1, 1))]
    batch_results = list(service.queries(queries, strict=False))
    assert batch_results[0] == single_rate


# LLM-generated content at query #8
#--------------------------

def test_query_returns_fxrate_for_valid_currency_pair_and_date():
    service = FXRateService()
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    result = service.query(usd, eur, asof, strict=False)
    assert isinstance(result, FXRate) or result is None

def test_query_raises_lookup_error_in_strict_mode_when_rate_not_found():
    service = FXRateService()
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    try:
        service.query(usd, eur, asof, strict=True)
        assert False, "Expected LookupError"
    except LookupError:
        pass

def test_query_returns_none_in_non_strict_mode_when_rate_not_found():
    service = FXRateService()
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    result = service.query(usd, eur, asof, strict=False)
    assert result is None

def test_query_handles_same_currency_pair():
    service = FXRateService()
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    result = service.query(usd, usd, asof, strict=False)
    assert result == FXRate.one()

def test_query_uses_default_service_when_available():
    FXRateService.default = FXRateService()
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    result = FXRateService.default.query(usd, eur, asof, strict=False)
    assert isinstance(result, FXRate) or result is None

def test_query_handles_different_currency_types():
    service = FXRateService()
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    btc = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    asof = Date(2023, 1, 1)
    result = service.query(usd, btc, asof, strict=False)
    assert isinstance(result, FXRate) or result is None

def test_query_handles_historical_date():
    service = FXRateService()
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2000, 1, 1)
    result = service.query(usd, eur, asof, strict=False)
    assert isinstance(result, FXRate) or result is None

def test_query_handles_future_date():
    service = FXRateService()
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2100, 1, 1)
    result = service.query(usd, eur, asof, strict=False)
    assert isinstance(result, FXRate) or result is None


# LLM-generated content at query #9
#--------------------------

def test_query_returns_fxrate_for_valid_currency_pair_and_date():
    service = FXRateService()
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    result = service.query(usd, eur, asof, strict=False)
    assert result is not None
    assert isinstance(result, FXRate)

def test_query_returns_none_for_nonexistent_pair_when_not_strict():
    service = FXRateService()
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    result = service.query(usd, jpy, asof, strict=False)
    assert result is None

def test_query_raises_error_for_nonexistent_pair_when_strict():
    service = FXRateService()
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    try:
        service.query(usd, jpy, asof, strict=True)
        assert False
    except LookupError:
        assert True

def test_query_handles_same_currency_pair():
    service = FXRateService()
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    result = service.query(usd, usd, asof, strict=False)
    assert result is not None
    assert result.rate == Decimal("1")

def test_query_returns_consistent_fxrate_for_same_inputs():
    service = FXRateService()
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    result1 = service.query(usd, eur, asof, strict=False)
    result2 = service.query(usd, eur, asof, strict=False)
    assert result1 == result2

def test_query_handles_different_dates():
    service = FXRateService()
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof1 = Date(2023, 1, 1)
    asof2 = Date(2023, 1, 2)
    result1 = service.query(usd, eur, asof1, strict=False)
    result2 = service.query(usd, eur, asof2, strict=False)
    assert result1 != result2


# LLM-generated content at query #10
#--------------------------

def test_query_returns_fx_rate_for_currency_pair_and_date():
    service = FXRateService()
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    result = service.query(ccy1, ccy2, asof, strict=False)
    assert result is None or isinstance(result, FXRate)

def test_query_returns_none_when_rate_not_found_and_strict_false():
    service = FXRateService()
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    result = service.query(ccy1, ccy2, asof, strict=False)
    assert result is None

def test_query_raises_error_when_rate_not_found_and_strict_true():
    service = FXRateService()
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    try:
        service.query(ccy1, ccy2, asof, strict=True)
        assert False
    except LookupError:
        assert True

def test_query_handles_same_currency_pair():
    service = FXRateService()
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    result = service.query(ccy1, ccy1, asof, strict=False)
    assert result == FXRate.one()

def test_query_handles_different_currency_types():
    service = FXRateService()
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    asof = Date(2023, 1, 1)
    result = service.query(ccy1, ccy2, asof, strict=False)
    assert result is None or isinstance(result, FXRate)

def test_query_handles_currency_with_negative_decimals():
    service = FXRateService()
    ccy1 = Currency.of("ZZZ", "Weird Currency", -1, CurrencyType.CRYPTO)
    ccy2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    result = service.query(ccy1, ccy2, asof, strict=False)
    assert result is None or isinstance(result, FXRate)

def test_query_handles_currency_with_zero_decimals():
    service = FXRateService()
    ccy1 = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    ccy2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    result = service.query(ccy1, ccy2, asof, strict=False)
    assert result is None or isinstance(result, FXRate)

def test_query_handles_different_dates():
    service = FXRateService()
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof1 = Date(2023, 1, 1)
    asof2 = Date(2023, 12, 31)
    result1 = service.query(ccy1, ccy2, asof1, strict=False)
    result2 = service.query(ccy1, ccy2, asof2, strict=False)
    assert result1 is None or isinstance(result1, FXRate)
    assert result2 is None or isinstance(result2, FXRate)

def test_query_handles_reverse_currency_pair():
    service = FXRateService()
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    result1 = service.query(ccy1, ccy2, asof, strict=False)
    result2 = service.query(ccy2, ccy1, asof, strict=False)
    assert result1 is None or isinstance(result1, FXRate)
    assert result2 is None or isinstance(result2, FXRate)
    if result1 is not None and result2 is not None:
        assert result1 == result2.inverse()


# LLM-generated content at query #11
#--------------------------

def test_constructor_creates_valid_fxrate():
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
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("2")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate[0] == ccy1
    assert rate[1] == ccy2
    assert rate[2] == date
    assert rate[3] == value

def test_constructor_creates_fxrate_with_same_currency():
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["EUR"]
    date = datetime.date.today()
    value = Decimal("1")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value

def test_constructor_creates_fxrate_with_decimal_value():
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("1.2345")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.value == value

def test_constructor_creates_fxrate_with_negative_value():
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("-1")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.value == value

def test_constructor_creates_fxrate_with_zero_value():
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("0")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.value == value

def test_constructor_creates_fxrate_with_future_date():
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today() + datetime.timedelta(days=1)
    value = Decimal("2")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.date == date

def test_constructor_creates_fxrate_with_past_date():
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today() - datetime.timedelta(days=1)
    value = Decimal("2")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.date == date


# LLM-generated content at query #12
#--------------------------

def test_constructor_creates_fxrate_with_correct_attributes():
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

def test_constructor_creates_fxrate_with_same_currency_and_value_one():
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

def test_constructor_creates_fxrate_with_positive_decimal_value():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("0.5")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.value == Decimal("0.5")

def test_constructor_creates_fxrate_with_large_decimal_value():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("12345.6789")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.value == Decimal("12345.6789")

def test_constructor_creates_fxrate_with_future_date():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date(2100, 12, 31)
    value = Decimal("1.2")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.date == datetime.date(2100, 12, 31)

def test_constructor_creates_fxrate_with_past_date():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date(2000, 1, 1)
    value = Decimal("0.8")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.date == datetime.date(2000, 1, 1)

def test_constructor_creates_fxrate_with_different_currency_pair():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["GBP"]
    ccy2 = Currencies["JPY"]
    date = datetime.date.today()
    value = Decimal("150.75")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == Currencies["GBP"]
    assert rate.ccy2 == Currencies["JPY"]

def test_constructor_creates_fxrate_with_value_one_for_same_currency():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy = Currencies["CAD"]
    date = datetime.date.today()
    value = Decimal("1")
    rate = FXRate(ccy, ccy, date, value)
    assert rate.value == Decimal("1")

def test_constructor_creates_fxrate_with_indexed_access():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("1.1")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate[0] == ccy1
    assert rate[1] == ccy2
    assert rate[2] == date
    assert rate[3] == value


# LLM-generated content at query #13
#--------------------------

def test_constructor_creates_fxrate_with_correct_attributes():
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("2")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value

def test_constructor_allows_unpacking():
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

def test_constructor_accepts_same_currency_with_value_one():
    ccy = Currencies["EUR"]
    date = datetime.date.today()
    value = Decimal("1")
    rate = FXRate(ccy, ccy, date, value)
    assert rate.ccy1 == ccy
    assert rate.ccy2 == ccy
    assert rate.value == value

def test_constructor_accepts_same_currency_with_value_one_decimal():
    ccy = Currencies["EUR"]
    date = datetime.date.today()
    value = Decimal("1.0")
    rate = FXRate(ccy, ccy, date, value)
    assert rate.ccy1 == ccy
    assert rate.ccy2 == ccy
    assert rate.value == value

def test_constructor_accepts_positive_decimal_value():
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("0.5")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.value == value

def test_constructor_accepts_large_positive_decimal_value():
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("123456.789")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.value == value


# LLM-generated content at query #14
#--------------------------

def test_constructor_creates_valid_instance():
    from pypara.currencies import Currencies
    import datetime
    from decimal import Decimal
    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("2"))
    assert rate.ccy1 == Currencies["EUR"]
    assert rate.ccy2 == Currencies["USD"]
    assert rate.date == datetime.date.today()
    assert rate.value == Decimal("2")

def test_constructor_allows_indexed_access():
    from pypara.currencies import Currencies
    import datetime
    from decimal import Decimal
    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("2"))
    ccy1, ccy2, date, value = rate
    assert ccy1 == Currencies["EUR"]
    assert ccy2 == Currencies["USD"]
    assert date == datetime.date.today()
    assert value == Decimal("2")

def test_constructor_creates_same_currency_rate():
    from pypara.currencies import Currencies
    import datetime
    from decimal import Decimal
    rate = FXRate(Currencies["EUR"], Currencies["EUR"], datetime.date.today(), Decimal("1"))
    assert rate.ccy1 == Currencies["EUR"]
    assert rate.ccy2 == Currencies["EUR"]
    assert rate.value == Decimal("1")

def test_constructor_accepts_decimal_value():
    from pypara.currencies import Currencies
    import datetime
    from decimal import Decimal
    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date.today(), Decimal("1.2345"))
    assert rate.value == Decimal("1.2345")

def test_constructor_accepts_future_date():
    from pypara.currencies import Currencies
    import datetime
    from decimal import Decimal
    future_date = datetime.date.today() + datetime.timedelta(days=10)
    rate = FXRate(Currencies["EUR"], Currencies["USD"], future_date, Decimal("1.5"))
    assert rate.date == future_date


# LLM-generated content at query #15
#--------------------------

def test_fxrate_constructor_creates_instance_with_correct_attributes():
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

def test_fxrate_constructor_allows_tuple_unpacking():
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

def test_fxrate_constructor_creates_instance_with_indexed_access():
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

def test_fxrate_constructor_accepts_same_currency_with_value_one():
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

def test_fxrate_constructor_accepts_same_currency_with_value_not_one():
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

def test_fxrate_constructor_accepts_zero_value():
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

def test_fxrate_constructor_accepts_negative_value():
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


# LLM-generated content at query #16
#--------------------------

def test_constructor_creates_valid_fxrate():
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
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("2")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate[0] == ccy1
    assert rate[1] == ccy2
    assert rate[2] == date
    assert rate[3] == value

def test_constructor_accepts_same_currency_with_one():
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["EUR"]
    date = datetime.date.today()
    value = Decimal("1")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value

def test_constructor_accepts_same_currency_with_non_one():
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["EUR"]
    date = datetime.date.today()
    value = Decimal("2")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value

def test_constructor_accepts_zero_value():
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("0")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value

def test_constructor_accepts_negative_value():
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("-1")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value

def test_constructor_accepts_non_currency_for_ccy1():
    ccy1 = "EUR"
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("2")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value

def test_constructor_accepts_non_currency_for_ccy2():
    ccy1 = Currencies["EUR"]
    ccy2 = "USD"
    date = datetime.date.today()
    value = Decimal("2")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value

def test_constructor_accepts_non_date_for_date():
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = "2023-01-01"
    value = Decimal("2")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value

def test_constructor_accepts_non_decimal_for_value():
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = 2.0
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value


# LLM-generated content at query #17
#--------------------------

def test_constructor_creates_fxrate_with_correct_attributes():
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

def test_constructor_creates_fxrate_with_same_currency_and_value_one():
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

def test_constructor_creates_fxrate_with_positive_decimal_value():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("0.5")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.value == value

def test_constructor_creates_fxrate_with_different_currencies_and_value_not_one():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("1.5")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.value == value


# LLM-generated content at query #18
#--------------------------

def test_query_returns_fxrate_for_valid_currency_pair_and_date():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return FXRate(Decimal("1.5"), ccy1, ccy2, asof)
        def queries(self, queries, strict=False):
            return []
    service = MockFXRateService()
    ccy1 = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date.today()
    result = service.query(ccy1, ccy2, asof)
    assert result.ccy1 == ccy1
    assert result.ccy2 == ccy2
    assert result.asof == asof
    assert result.value == Decimal("1.5")

def test_query_returns_none_for_missing_fxrate_when_strict_false():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        def queries(self, queries, strict=False):
            return []
    service = MockFXRateService()
    ccy1 = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date.today()
    result = service.query(ccy1, ccy2, asof, strict=False)
    assert result is None

def test_query_raises_error_for_missing_fxrate_when_strict_true():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if strict:
                raise LookupError("FX rate not found")
            return None
        def queries(self, queries, strict=False):
            return []
    service = MockFXRateService()
    ccy1 = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date.today()
    try:
        service.query(ccy1, ccy2, asof, strict=True)
        assert False, "Expected LookupError"
    except LookupError:
        pass

def test_query_handles_same_currency_pair():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if ccy1 == ccy2:
                return FXRate(Decimal("1.0"), ccy1, ccy2, asof)
            return FXRate(Decimal("1.5"), ccy1, ccy2, asof)
        def queries(self, queries, strict=False):
            return []
    service = MockFXRateService()
    ccy = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    asof = Date.today()
    result = service.query(ccy, ccy, asof)
    assert result.value == Decimal("1.0")

def test_query_handles_different_currency_types():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return FXRate(Decimal("2.0"), ccy1, ccy2, asof)
        def queries(self, queries, strict=False):
            return []
    service = MockFXRateService()
    ccy1 = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    asof = Date.today()
    result = service.query(ccy1, ccy2, asof)
    assert result.ccy1 == ccy1
    assert result.ccy2 == ccy2
    assert result.value == Decimal("2.0")

def test_query_handles_historical_date():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return FXRate(Decimal("0.9"), ccy1, ccy2, asof)
        def queries(self, queries, strict=False):
            return []
    service = MockFXRateService()
    ccy1 = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2020, 1, 1)
    result = service.query(ccy1, ccy2, asof)
    assert result.asof == asof
    assert result.value == Decimal("0.9")

def test_query_handles_future_date():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return FXRate(Decimal("1.1"), ccy1, ccy2, asof)
        def queries(self, queries, strict=False):
            return []
    service = MockFXRateService()
    ccy1 = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2030, 1, 1)
    result = service.query(ccy1, ccy2, asof)
    assert result.asof == asof
    assert result.value == Decimal("1.1")

def test_query_handles_currency_with_zero_decimals():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return FXRate(Decimal("100.0"), ccy1, ccy2, asof)
        def queries(self, queries, strict=False):
            return []
    service = MockFXRateService()
    ccy1 = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    ccy2 = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    asof = Date.today()
    result = service.query(ccy1, ccy2, asof)
    assert result.ccy1 == ccy1
    assert result.ccy2 == ccy2
    assert result.value == Decimal("100.0")

def test_query_handles_currency_with_negative_decimals():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return FXRate(Decimal("0.000001"), ccy1, ccy2, asof)
        def queries(self, queries, strict=False):
            return []
    service = MockFXRateService()
    ccy1 = Currency.of("ZZZ", "Weird Crypto", -1, CurrencyType.CRYPTO)
    ccy2 = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    asof = Date.today()
    result = service.query(ccy1, ccy2, asof)
    assert result.ccy1 == ccy1
    assert result.ccy2 == ccy2
    assert result.value == Decimal("0.000001")

def test_query_handles_reverse_currency_pair():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if ccy1.code == "USD" and ccy2.code == "EUR":
                return FXRate(Decimal("0.85"), ccy1, ccy2, asof)
            elif ccy1.code == "EUR" and ccy2.code == "USD":
                return FXRate(Decimal("1.176470588235294"), ccy1, ccy2, asof)
            return None
        def queries(self, queries, strict=False):
            return []
    service = MockFXRateService()
    usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date.today()
    result1 = service.query(usd, eur, asof)
    result2 = service.query(eur, usd, asof)
    assert result1.value == Decimal("0.85")
    assert result2.value == Decimal("1.176470588235294")


# LLM-generated content at query #19
#--------------------------

def test_queries_returns_correct_rates():
    from unittest.mock import Mock, call
    mock_service = Mock(spec=FXRateService)
    mock_service.query.side_effect = [Decimal('1.1'), Decimal('1.2'), None]
    queries = [(Currency('USD'), Currency('EUR'), Date(2023, 1, 1)), (Currency('GBP'), Currency('JPY'), Date(2023, 1, 2)), (Currency('AUD'), Currency('CAD'), Date(2023, 1, 3))]
    result = list(mock_service.queries(queries))
    expected_calls = [call(Currency('USD'), Currency('EUR'), Date(2023, 1, 1), strict=False), call(Currency('GBP'), Currency('JPY'), Date(2023, 1, 2), strict=False), call(Currency('AUD'), Currency('CAD'), Date(2023, 1, 3), strict=False)]
    assert mock_service.query.call_args_list == expected_calls
    assert result == [Decimal('1.1'), Decimal('1.2'), None]

def test_queries_with_strict_raises_error():
    from unittest.mock import Mock
    mock_service = Mock(spec=FXRateService)
    mock_service.query.side_effect = [Decimal('1.1'), LookupError('Rate not found'), Decimal('1.3')]
    queries = [(Currency('USD'), Currency('EUR'), Date(2023, 1, 1)), (Currency('GBP'), Currency('JPY'), Date(2023, 1, 2)), (Currency('AUD'), Currency('CAD'), Date(2023, 1, 3))]
    try:
        list(mock_service.queries(queries, strict=True))
    except LookupError as e:
        assert str(e) == 'Rate not found'

def test_queries_empty_iterable():
    from unittest.mock import Mock
    mock_service = Mock(spec=FXRateService)
    result = list(mock_service.queries([]))
    assert result == []
    mock_service.query.assert_not_called()

def test_queries_handles_single_query():
    from unittest.mock import Mock, call
    mock_service = Mock(spec=FXRateService)
    mock_service.query.return_value = Decimal('0.85')
    queries = [(Currency('EUR'), Currency('USD'), Date(2023, 5, 10))]
    result = list(mock_service.queries(queries))
    mock_service.query.assert_called_once_with(Currency('EUR'), Currency('USD'), Date(2023, 5, 10), strict=False)
    assert result == [Decimal('0.85')]

def test_queries_strict_false_returns_none_for_missing():
    from unittest.mock import Mock, call
    mock_service = Mock(spec=FXRateService)
    mock_service.query.side_effect = [Decimal('1.5'), None, Decimal('2.0')]
    queries = [(Currency('USD'), Currency('GBP'), Date(2023, 2, 1)), (Currency('JPY'), Currency('EUR'), Date(2023, 2, 2)), (Currency('CAD'), Currency('AUD'), Date(2023, 2, 3))]
    result = list(mock_service.queries(queries, strict=False))
    expected_calls = [call(Currency('USD'), Currency('GBP'), Date(2023, 2, 1), strict=False), call(Currency('JPY'), Currency('EUR'), Date(2023, 2, 2), strict=False), call(Currency('CAD'), Currency('AUD'), Date(2023, 2, 3), strict=False)]
    assert mock_service.query.call_args_list == expected_calls
    assert result == [Decimal('1.5'), None, Decimal('2.0')]


# LLM-generated content at query #20
#--------------------------

def test_fxrate_constructor_creates_instance_with_correct_attributes():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date(2023, 1, 1)
    value = Decimal("1.2")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value

def test_fxrate_constructor_allows_unpacking():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date(2023, 1, 1)
    value = Decimal("1.2")
    rate = FXRate(ccy1, ccy2, date, value)
    unpacked_ccy1, unpacked_ccy2, unpacked_date, unpacked_value = rate
    assert unpacked_ccy1 == ccy1
    assert unpacked_ccy2 == ccy2
    assert unpacked_date == date
    assert unpacked_value == value

def test_fxrate_constructor_accepts_same_currency_with_value_one():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy = Currencies["EUR"]
    date = datetime.date(2023, 1, 1)
    value = Decimal("1")
    rate = FXRate(ccy, ccy, date, value)
    assert rate.ccy1 == ccy
    assert rate.ccy2 == ccy
    assert rate.date == date
    assert rate.value == value

def test_fxrate_constructor_accepts_positive_decimal_value():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date(2023, 1, 1)
    value = Decimal("0.5")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.value == value

def test_fxrate_constructor_accepts_large_positive_decimal_value():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date(2023, 1, 1)
    value = Decimal("1000000.123456")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.value == value


# LLM-generated content at query #21
#--------------------------

def test_constructor_creates_fxrate_with_correct_attributes():
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

def test_constructor_allows_indexed_access():
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

def test_constructor_creates_fxrate_with_same_currency_and_value_one():
    from pypara.currencies import Currencies
    import datetime
    from decimal import Decimal
    ccy = Currencies["EUR"]
    date = datetime.date.today()
    value = Decimal("1")
    rate = FXRate(ccy, ccy, date, value)
    assert rate.ccy1 == ccy
    assert rate.ccy2 == ccy
    assert rate.value == value

def test_constructor_creates_fxrate_with_positive_decimal_value():
    from pypara.currencies import Currencies
    import datetime
    from decimal import Decimal
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("0.5")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.value == value

def test_constructor_creates_fxrate_with_large_decimal_value():
    from pypara.currencies import Currencies
    import datetime
    from decimal import Decimal
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("12345.6789")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.value == value


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_query_returns_fxrate_for_valid_currency_pair_and_date():
    service = FXRateService()
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    result = service.query(usd, eur, asof, strict=False)
    assert isinstance(result, FXRate) or result is None


def test_query_returns_none_for_nonexistent_pair_when_strict_false():
    service = FXRateService()
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    result = service.query(usd, jpy, asof, strict=False)
    assert result is None


def test_query_raises_error_for_nonexistent_pair_when_strict_true():
    service = FXRateService()
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    try:
        service.query(usd, jpy, asof, strict=True)
        assert False
    except LookupError:
        assert True


def test_query_returns_same_rate_for_inverse_currency_pair():
    service = FXRateService()
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    rate1 = service.query(usd, eur, asof, strict=False)
    rate2 = service.query(eur, usd, asof, strict=False)
    if rate1 is not None and rate2 is not None:
        assert rate1 == 1 / rate2
    else:
        assert rate1 is None and rate2 is None


def test_query_returns_none_for_same_currency_pair():
    service = FXRateService()
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    result = service.query(usd, usd, asof, strict=False)
    assert result == FXRate.one(usd, usd, asof)


# LLM-generated content at query #2
#--------------------------

def test_constructor_creates_valid_fxrate():
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
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("2")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate[0] == ccy1
    assert rate[1] == ccy2
    assert rate[2] == date
    assert rate[3] == value

def test_constructor_creates_invertible_fxrate():
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("2")
    rate = FXRate(ccy1, ccy2, date, value)
    inverted = ~rate
    assert inverted.ccy1 == ccy2
    assert inverted.ccy2 == ccy1
    assert inverted.date == date
    assert inverted.value == Decimal("0.5")

def test_constructor_creates_fxrate_with_same_currency():
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["EUR"]
    date = datetime.date.today()
    value = Decimal("1")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value

def test_constructor_creates_fxrate_with_decimal_value():
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("1.2345")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.value == value

def test_constructor_creates_fxrate_with_future_date():
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today() + datetime.timedelta(days=1)
    value = Decimal("2")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.date == date

def test_constructor_creates_fxrate_with_past_date():
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today() - datetime.timedelta(days=1)
    value = Decimal("2")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.date == date

def test_constructor_creates_fxrate_with_high_value():
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("1000000")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.value == value

def test_constructor_creates_fxrate_with_small_value():
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("0.000001")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.value == value

def test_constructor_creates_fxrate_with_negative_value():
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("-2")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.value == value


# LLM-generated content at query #3
#--------------------------

def test_query_returns_fx_rate_for_valid_currency_pair_and_date():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return FXRate.of(Decimal("1.5"), ccy1, ccy2, asof)
        def queries(self, queries, strict=False):
            return []
    usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date.today()
    service = MockFXRateService()
    result = service.query(usd, eur, asof)
    expected = FXRate.of(Decimal("1.5"), usd, eur, asof)
    assert result == expected

def test_query_returns_none_for_missing_rate_when_strict_false():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        def queries(self, queries, strict=False):
            return []
    usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date.today()
    service = MockFXRateService()
    result = service.query(usd, eur, asof, strict=False)
    assert result is None

def test_query_raises_error_for_missing_rate_when_strict_true():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if strict:
                raise LookupError("Rate not found")
            return None
        def queries(self, queries, strict=False):
            return []
    usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date.today()
    service = MockFXRateService()
    try:
        service.query(usd, eur, asof, strict=True)
        assert False
    except LookupError:
        assert True

def test_query_handles_same_currency_pair():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if ccy1 == ccy2:
                return FXRate.of(Decimal("1"), ccy1, ccy2, asof)
            return None
        def queries(self, queries, strict=False):
            return []
    usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    asof = Date.today()
    service = MockFXRateService()
    result = service.query(usd, usd, asof)
    expected = FXRate.of(Decimal("1"), usd, usd, asof)
    assert result == expected

def test_query_uses_correct_currency_order():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return FXRate.of(Decimal("0.5"), ccy1, ccy2, asof)
        def queries(self, queries, strict=False):
            return []
    usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date.today()
    service = MockFXRateService()
    result = service.query(eur, usd, asof)
    expected = FXRate.of(Decimal("0.5"), eur, usd, asof)
    assert result == expected


# LLM-generated content at query #4
#--------------------------

def test_queries_returns_correct_rates_for_multiple_queries():
    mock_service = Mock(spec=FXRateService)
    mock_service.query.side_effect = [Decimal('1.1'), Decimal('1.2'), None]
    queries = [(Currency('USD'), Currency('EUR'), Date(2023, 1, 1)), (Currency('GBP'), Currency('USD'), Date(2023, 1, 2)), (Currency('JPY'), Currency('EUR'), Date(2023, 1, 3))]
    result = list(mock_service.queries(queries, strict=False))
    assert result == [Decimal('1.1'), Decimal('1.2'), None]

def test_queries_raises_error_in_strict_mode_when_rate_missing():
    mock_service = Mock(spec=FXRateService)
    mock_service.query.side_effect = [Decimal('1.1'), LookupError('Rate not found')]
    queries = [(Currency('USD'), Currency('EUR'), Date(2023, 1, 1)), (Currency('GBP'), Currency('USD'), Date(2023, 1, 2))]
    with pytest.raises(LookupError, match='Rate not found'):
        list(mock_service.queries(queries, strict=True))

def test_queries_handles_empty_queries_list():
    mock_service = Mock(spec=FXRateService)
    queries = []
    result = list(mock_service.queries(queries, strict=False))
    assert result == []

def test_queries_calls_query_with_correct_parameters():
    mock_service = Mock(spec=FXRateService)
    mock_service.query.return_value = Decimal('1.5')
    queries = [(Currency('AUD'), Currency('CAD'), Date(2023, 5, 10))]
    list(mock_service.queries(queries, strict=False))
    mock_service.query.assert_called_once_with(Currency('AUD'), Currency('CAD'), Date(2023, 5, 10), strict=False)

def test_queries_passes_strict_flag_to_query_calls():
    mock_service = Mock(spec=FXRateService)
    mock_service.query.return_value = Decimal('0.9')
    queries = [(Currency('CHF'), Currency('USD'), Date(2023, 7, 20))]
    list(mock_service.queries(queries, strict=True))
    mock_service.query.assert_called_once_with(Currency('CHF'), Currency('USD'), Date(2023, 7, 20), strict=True)


# LLM-generated content at query #5
#--------------------------

def test_query_returns_fxrate_for_valid_currency_pair_and_date():
    service = FXRateService()
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    rate = service.query(usd, eur, asof, strict=False)
    assert rate is not None
    assert isinstance(rate, FXRate)
    assert rate.ccy1 == usd
    assert rate.ccy2 == eur
    assert rate.date == asof


def test_query_returns_none_for_nonexistent_pair_when_strict_false():
    service = FXRateService()
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    rate = service.query(usd, jpy, asof, strict=False)
    assert rate is None


def test_query_raises_error_for_nonexistent_pair_when_strict_true():
    service = FXRateService()
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    try:
        service.query(usd, jpy, asof, strict=True)
        assert False
    except LookupError:
        assert True


def test_query_returns_same_rate_for_inverse_currency_pair():
    service = FXRateService()
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    rate1 = service.query(usd, eur, asof, strict=False)
    rate2 = service.query(eur, usd, asof, strict=False)
    if rate1 is not None and rate2 is not None:
        assert rate1.rate * rate2.rate == Decimal("1")


def test_query_handles_same_currency_pair():
    service = FXRateService()
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    rate = service.query(usd, usd, asof, strict=False)
    assert rate is not None
    assert rate.rate == Decimal("1")


# LLM-generated content at query #6
#--------------------------

def test_queries_returns_correct_rates():
    service = FXRateService.default
    queries = [(Currency("USD"), Currency("EUR"), Date(2023, 1, 1)), (Currency("GBP"), Currency("JPY"), Date(2023, 1, 2))]
    results = list(service.queries(queries, strict=False))
    assert len(results) == 2
    assert isinstance(results[0], FXRate) or results[0] is None
    assert isinstance(results[1], FXRate) or results[1] is None

def test_queries_strict_mode_raises_error():
    service = FXRateService.default
    queries = [(Currency("USD"), Currency("XYZ"), Date(2023, 1, 1))]
    try:
        list(service.queries(queries, strict=True))
        assert False
    except LookupError:
        assert True

def test_queries_empty_input():
    service = FXRateService.default
    results = list(service.queries([], strict=False))
    assert len(results) == 0

def test_queries_single_query():
    service = FXRateService.default
    queries = [(Currency("USD"), Currency("EUR"), Date(2023, 1, 1))]
    results = list(service.queries(queries, strict=False))
    assert len(results) == 1
    assert results[0] == service.query(Currency("USD"), Currency("EUR"), Date(2023, 1, 1), strict=False)

def test_queries_handles_none_rates():
    service = FXRateService.default
    queries = [(Currency("AAA"), Currency("BBB"), Date(2023, 1, 1))]
    results = list(service.queries(queries, strict=False))
    assert results[0] is None


# LLM-generated content at query #7
#--------------------------

def test_query_returns_fxrate_for_valid_currency_pair_and_date():
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return FXRate.of(ccy1, ccy2, asof, Decimal("1.5"))
        def queries(self, queries, strict=False):
            return []
    service = ConcreteFXRateService()
    usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date.today()
    result = service.query(usd, eur, asof)
    assert result.ccy1 == usd
    assert result.ccy2 == eur
    assert result.date == asof
    assert result.value == Decimal("1.5")

def test_query_returns_none_for_missing_rate_when_strict_false():
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        def queries(self, queries, strict=False):
            return []
    service = ConcreteFXRateService()
    usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date.today()
    result = service.query(usd, eur, asof, strict=False)
    assert result is None

def test_query_raises_error_for_missing_rate_when_strict_true():
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if strict:
                raise FXRateLookupError("Rate not found")
            return None
        def queries(self, queries, strict=False):
            return []
    service = ConcreteFXRateService()
    usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date.today()
    try:
        service.query(usd, eur, asof, strict=True)
        assert False, "Expected FXRateLookupError"
    except FXRateLookupError:
        pass

def test_query_handles_same_currency_pair():
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if ccy1 == ccy2:
                return FXRate.of(ccy1, ccy2, asof, Decimal("1"))
            return FXRate.of(ccy1, ccy2, asof, Decimal("2"))
        def queries(self, queries, strict=False):
            return []
    service = ConcreteFXRateService()
    usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    asof = Date.today()
    result = service.query(usd, usd, asof)
    assert result.value == Decimal("1")

def test_query_uses_correct_date():
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return FXRate.of(ccy1, ccy2, asof, Decimal("0.5"))
        def queries(self, queries, strict=False):
            return []
    service = ConcreteFXRateService()
    usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    specific_date = Date(2023, 1, 1)
    result = service.query(usd, eur, specific_date)
    assert result.date == specific_date

def test_query_handles_different_currency_types():
    class ConcreteFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return FXRate.of(ccy1, ccy2, asof, Decimal("3"))
        def queries(self, queries, strict=False):
            return []
    service = ConcreteFXRateService()
    money_ccy = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    crypto_ccy = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    asof = Date.today()
    result = service.query(money_ccy, crypto_ccy, asof)
    assert result.ccy1 == money_ccy
    assert result.ccy2 == crypto_ccy


# LLM-generated content at query #8
#--------------------------

def test_queries_returns_correct_rates():
    mock_service = Mock(spec=FXRateService)
    mock_service.query.side_effect = [Decimal('1.2'), Decimal('0.85'), None]
    queries = [(Currency('USD'), Currency('EUR'), Date(2023, 1, 1)), (Currency('GBP'), Currency('USD'), Date(2023, 1, 1)), (Currency('JPY'), Currency('EUR'), Date(2023, 1, 1))]
    result = list(mock_service.queries(queries, strict=False))
    assert result == [Decimal('1.2'), Decimal('0.85'), None]

def test_queries_strict_mode_raises_error():
    mock_service = Mock(spec=FXRateService)
    mock_service.query.side_effect = [Decimal('1.2'), LookupError('Rate not found'), Decimal('0.85')]
    queries = [(Currency('USD'), Currency('EUR'), Date(2023, 1, 1)), (Currency('GBP'), Currency('USD'), Date(2023, 1, 1)), (Currency('JPY'), Currency('EUR'), Date(2023, 1, 1))]
    with pytest.raises(LookupError):
        list(mock_service.queries(queries, strict=True))

def test_queries_empty_input():
    mock_service = Mock(spec=FXRateService)
    result = list(mock_service.queries([], strict=False))
    assert result == []

def test_queries_single_query():
    mock_service = Mock(spec=FXRateService)
    mock_service.query.return_value = Decimal('1.1')
    queries = [(Currency('USD'), Currency('CAD'), Date(2023, 1, 1))]
    result = list(mock_service.queries(queries, strict=False))
    assert result == [Decimal('1.1')]

def test_queries_all_none_when_strict_false():
    mock_service = Mock(spec=FXRateService)
    mock_service.query.return_value = None
    queries = [(Currency('USD'), Currency('EUR'), Date(2023, 1, 1)), (Currency('GBP'), Currency('USD'), Date(2023, 1, 1))]
    result = list(mock_service.queries(queries, strict=False))
    assert result == [None, None]


# LLM-generated content at query #9
#--------------------------

def test_queries_returns_correct_rates():
    mock_service = Mock(spec=FXRateService)
    mock_service.query.side_effect = lambda ccy1, ccy2, asof, strict: Decimal('1.5') if (ccy1, ccy2, asof) == (Currency('USD'), Currency('EUR'), Date(2023, 1, 1)) else Decimal('0.8')
    queries = [(Currency('USD'), Currency('EUR'), Date(2023, 1, 1)), (Currency('GBP'), Currency('USD'), Date(2023, 1, 1))]
    result = list(mock_service.queries(queries, strict=False))
    assert result == [Decimal('1.5'), Decimal('0.8')]

def test_queries_with_strict_raises_error():
    mock_service = Mock(spec=FXRateService)
    mock_service.query.side_effect = FXRateLookupError("Rate not found")
    queries = [(Currency('USD'), Currency('EUR'), Date(2023, 1, 1))]
    with pytest.raises(FXRateLookupError):
        list(mock_service.queries(queries, strict=True))

def test_queries_with_strict_false_returns_none():
    mock_service = Mock(spec=FXRateService)
    mock_service.query.return_value = None
    queries = [(Currency('USD'), Currency('EUR'), Date(2023, 1, 1))]
    result = list(mock_service.queries(queries, strict=False))
    assert result == [None]

def test_queries_empty_iterable():
    mock_service = Mock(spec=FXRateService)
    queries = []
    result = list(mock_service.queries(queries, strict=False))
    assert result == []

def test_queries_calls_query_for_each_input():
    mock_service = Mock(spec=FXRateService)
    mock_service.query.return_value = Decimal('1.0')
    queries = [(Currency('USD'), Currency('EUR'), Date(2023, 1, 1)), (Currency('GBP'), Currency('USD'), Date(2023, 1, 1))]
    list(mock_service.queries(queries, strict=False))
    assert mock_service.query.call_count == 2
    mock_service.query.assert_any_call(Currency('USD'), Currency('EUR'), Date(2023, 1, 1), False)
    mock_service.query.assert_any_call(Currency('GBP'), Currency('USD'), Date(2023, 1, 1), False)


# LLM-generated content at query #10
#--------------------------

def test_constructor_creates_fxrate_with_correct_attributes():
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

def test_constructor_creates_fxrate_with_same_currency_and_value_one():
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

def test_constructor_creates_fxrate_with_positive_decimal_value():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("0.5")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.value == value

def test_constructor_creates_fxrate_with_large_decimal_value():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("12345.6789")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.value == value

def test_constructor_creates_fxrate_with_future_date():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date(2100, 12, 31)
    value = Decimal("1.2")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.date == date

def test_constructor_creates_fxrate_with_past_date():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date(2000, 1, 1)
    value = Decimal("0.9")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.date == date

def test_constructor_creates_fxrate_with_different_currency_objects():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["JPY"]
    ccy2 = Currencies["GBP"]
    date = datetime.date.today()
    value = Decimal("0.007")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2


# LLM-generated content at query #11
#--------------------------

def test_constructor_creates_fxrate_with_correct_attributes():
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

def test_constructor_supports_tuple_unpacking():
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

def test_constructor_allows_same_currency_with_value_one():
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

def test_constructor_allows_same_currency_with_value_not_one():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy = Currencies["EUR"]
    date = datetime.date.today()
    value = Decimal("2")
    rate = FXRate(ccy, ccy, date, value)
    assert rate.ccy1 == ccy
    assert rate.ccy2 == ccy
    assert rate.date == date
    assert rate.value == value

def test_constructor_allows_negative_value():
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

def test_constructor_allows_zero_value():
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


# LLM-generated content at query #12
#--------------------------

def test_constructor_creates_fxrate_with_correct_attributes():
    ccy1 = Currency("EUR")
    ccy2 = Currency("USD")
    date = Date(2023, 1, 1)
    value = Decimal("1.2")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value

def test_constructor_allows_tuple_unpacking():
    ccy1 = Currency("EUR")
    ccy2 = Currency("USD")
    date = Date(2023, 1, 1)
    value = Decimal("1.2")
    rate = FXRate(ccy1, ccy2, date, value)
    unpacked_ccy1, unpacked_ccy2, unpacked_date, unpacked_value = rate
    assert unpacked_ccy1 == ccy1
    assert unpacked_ccy2 == ccy2
    assert unpacked_date == date
    assert unpacked_value == value

def test_constructor_creates_fxrate_with_same_currency_and_one_value():
    ccy = Currency("EUR")
    date = Date(2023, 1, 1)
    value = Decimal("1")
    rate = FXRate(ccy, ccy, date, value)
    assert rate.ccy1 == ccy
    assert rate.ccy2 == ccy
    assert rate.date == date
    assert rate.value == value

def test_constructor_creates_fxrate_with_positive_decimal_value():
    ccy1 = Currency("EUR")
    ccy2 = Currency("USD")
    date = Date(2023, 1, 1)
    value = Decimal("0.5")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.value == value

def test_constructor_creates_fxrate_with_large_decimal_value():
    ccy1 = Currency("EUR")
    ccy2 = Currency("USD")
    date = Date(2023, 1, 1)
    value = Decimal("1000.123456")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.value == value


# LLM-generated content at query #13
#--------------------------

def test_fxrate_constructor_creates_instance_with_correct_attributes():
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

def test_fxrate_constructor_allows_tuple_unpacking():
    from pypara.currencies import Currencies
    import datetime
    from decimal import Decimal
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

def test_fxrate_constructor_accepts_same_currency_with_value_one():
    from pypara.currencies import Currencies
    import datetime
    from decimal import Decimal
    ccy = Currencies["EUR"]
    date = datetime.date.today()
    value = Decimal("1")
    rate = FXRate(ccy, ccy, date, value)
    assert rate.ccy1 == ccy
    assert rate.ccy2 == ccy
    assert rate.value == value

def test_fxrate_constructor_accepts_same_currency_with_value_not_one():
    from pypara.currencies import Currencies
    import datetime
    from decimal import Decimal
    ccy = Currencies["EUR"]
    date = datetime.date.today()
    value = Decimal("2")
    rate = FXRate(ccy, ccy, date, value)
    assert rate.value == value

def test_fxrate_constructor_accepts_value_less_than_zero():
    from pypara.currencies import Currencies
    import datetime
    from decimal import Decimal
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("-0.5")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.value == value

def test_fxrate_constructor_accepts_value_equal_to_zero():
    from pypara.currencies import Currencies
    import datetime
    from decimal import Decimal
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("0")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.value == value


# LLM-generated content at query #14
#--------------------------

def test_constructor_creates_fxrate_with_correct_attributes():
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

def test_constructor_supports_tuple_unpacking():
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

def test_constructor_allows_same_currency_with_value_one():
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

def test_constructor_allows_same_currency_with_value_not_one():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy = Currencies["EUR"]
    date = datetime.date.today()
    value = Decimal("2")
    rate = FXRate(ccy, ccy, date, value)
    assert rate.ccy1 == ccy
    assert rate.ccy2 == ccy
    assert rate.date == date
    assert rate.value == value

def test_constructor_allows_zero_value():
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

def test_constructor_allows_negative_value():
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


# LLM-generated content at query #15
#--------------------------

def test_constructor_creates_fxrate_with_correct_attributes():
    ccy1 = Currency("EUR")
    ccy2 = Currency("USD")
    date = Date(2023, 1, 1)
    value = Decimal("1.2")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value

def test_constructor_allows_tuple_unpacking():
    ccy1 = Currency("EUR")
    ccy2 = Currency("USD")
    date = Date(2023, 1, 1)
    value = Decimal("1.2")
    rate = FXRate(ccy1, ccy2, date, value)
    unpacked_ccy1, unpacked_ccy2, unpacked_date, unpacked_value = rate
    assert unpacked_ccy1 == ccy1
    assert unpacked_ccy2 == ccy2
    assert unpacked_date == date
    assert unpacked_value == value

def test_constructor_creates_fxrate_with_same_currency_and_value_one():
    ccy = Currency("EUR")
    date = Date(2023, 1, 1)
    value = Decimal("1")
    rate = FXRate(ccy, ccy, date, value)
    assert rate.ccy1 == ccy
    assert rate.ccy2 == ccy
    assert rate.date == date
    assert rate.value == value

def test_constructor_creates_fxrate_with_positive_decimal_value():
    ccy1 = Currency("EUR")
    ccy2 = Currency("USD")
    date = Date(2023, 1, 1)
    value = Decimal("0.5")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.value == value

def test_constructor_creates_fxrate_with_large_decimal_value():
    ccy1 = Currency("EUR")
    ccy2 = Currency("USD")
    date = Date(2023, 1, 1)
    value = Decimal("1000000.123456")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.value == value

def test_constructor_creates_fxrate_with_future_date():
    ccy1 = Currency("EUR")
    ccy2 = Currency("USD")
    date = Date(2050, 12, 31)
    value = Decimal("1.2")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.date == date

def test_constructor_creates_fxrate_with_past_date():
    ccy1 = Currency("EUR")
    ccy2 = Currency("USD")
    date = Date(2000, 1, 1)
    value = Decimal("1.2")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.date == date

def test_constructor_creates_fxrate_with_different_currency_objects():
    ccy1 = Currency("EUR", 978)
    ccy2 = Currency("USD", 840)
    date = Date(2023, 1, 1)
    value = Decimal("1.2")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1.code == "EUR"
    assert rate.ccy2.code == "USD"

def test_constructor_creates_fxrate_where_indexed_access_matches_properties():
    ccy1 = Currency("EUR")
    ccy2 = Currency("USD")
    date = Date(2023, 1, 1)
    value = Decimal("1.2")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate[0] == rate.ccy1
    assert rate[1] == rate.ccy2
    assert rate[2] == rate.date
    assert rate[3] == rate.value


# LLM-generated content at query #16
#--------------------------

def test_constructor_creates_fxrate_with_correct_attributes():
    from pypara.currencies import Currencies
    import datetime
    from decimal import Decimal
    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date(2023, 1, 1), Decimal("1.1"))
    assert rate.ccy1 == Currencies["EUR"]
    assert rate.ccy2 == Currencies["USD"]
    assert rate.date == datetime.date(2023, 1, 1)
    assert rate.value == Decimal("1.1")

def test_constructor_allows_indexed_access():
    from pypara.currencies import Currencies
    import datetime
    from decimal import Decimal
    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date(2023, 1, 1), Decimal("1.1"))
    assert rate[0] == Currencies["EUR"]
    assert rate[1] == Currencies["USD"]
    assert rate[2] == datetime.date(2023, 1, 1)
    assert rate[3] == Decimal("1.1")

def test_constructor_allows_unpacking():
    from pypara.currencies import Currencies
    import datetime
    from decimal import Decimal
    ccy1, ccy2, date, value = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date(2023, 1, 1), Decimal("1.1"))
    assert ccy1 == Currencies["EUR"]
    assert ccy2 == Currencies["USD"]
    assert date == datetime.date(2023, 1, 1)
    assert value == Decimal("1.1")

def test_constructor_creates_fxrate_with_same_currency_and_one_value():
    from pypara.currencies import Currencies
    import datetime
    from decimal import Decimal
    rate = FXRate(Currencies["EUR"], Currencies["EUR"], datetime.date(2023, 1, 1), Decimal("1"))
    assert rate.ccy1 == Currencies["EUR"]
    assert rate.ccy2 == Currencies["EUR"]
    assert rate.value == Decimal("1")

def test_constructor_creates_fxrate_with_positive_value():
    from pypara.currencies import Currencies
    import datetime
    from decimal import Decimal
    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date(2023, 1, 1), Decimal("0.0001"))
    assert rate.value > Decimal("0")

def test_constructor_creates_fxrate_with_currency_objects():
    from pypara.currencies import Currencies
    import datetime
    from decimal import Decimal
    eur = Currencies["EUR"]
    usd = Currencies["USD"]
    rate = FXRate(eur, usd, datetime.date(2023, 1, 1), Decimal("1.1"))
    assert rate.ccy1 is eur
    assert rate.ccy2 is usd

def test_constructor_creates_fxrate_with_date_object():
    from pypara.currencies import Currencies
    import datetime
    from decimal import Decimal
    date = datetime.date(2023, 1, 1)
    rate = FXRate(Currencies["EUR"], Currencies["USD"], date, Decimal("1.1"))
    assert rate.date is date

def test_constructor_creates_fxrate_with_decimal_value():
    from pypara.currencies import Currencies
    import datetime
    from decimal import Decimal
    value = Decimal("1.1")
    rate = FXRate(Currencies["EUR"], Currencies["USD"], datetime.date(2023, 1, 1), value)
    assert rate.value is value


# LLM-generated content at query #17
#--------------------------

def test_query_returns_fxrate_for_valid_currency_pair_and_date():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return FXRate(Decimal("1.5"), ccy1, ccy2, asof)
        def queries(self, queries, strict=False):
            return []
    service = MockFXRateService()
    ccy1 = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    result = service.query(ccy1, ccy2, asof)
    assert result == FXRate(Decimal("1.5"), ccy1, ccy2, asof)

def test_query_returns_none_for_missing_fxrate_when_strict_false():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return None
        def queries(self, queries, strict=False):
            return []
    service = MockFXRateService()
    ccy1 = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    result = service.query(ccy1, ccy2, asof, strict=False)
    assert result is None

def test_query_raises_error_for_missing_fxrate_when_strict_true():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if strict:
                raise LookupError("FX rate not found")
            return None
        def queries(self, queries, strict=False):
            return []
    service = MockFXRateService()
    ccy1 = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    try:
        service.query(ccy1, ccy2, asof, strict=True)
        assert False
    except LookupError as e:
        assert str(e) == "FX rate not found"

def test_query_handles_same_currency_pair():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            if ccy1 == ccy2:
                return FXRate(Decimal("1"), ccy1, ccy2, asof)
            return None
        def queries(self, queries, strict=False):
            return []
    service = MockFXRateService()
    ccy = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    result = service.query(ccy, ccy, asof)
    assert result == FXRate(Decimal("1"), ccy, ccy, asof)

def test_query_uses_correct_asof_date():
    class MockFXRateService(FXRateService):
        def query(self, ccy1, ccy2, asof, strict=False):
            return FXRate(Decimal("2.0"), ccy1, ccy2, asof)
        def queries(self, queries, strict=False):
            return []
    service = MockFXRateService()
    ccy1 = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 12, 31)
    result = service.query(ccy1, ccy2, asof)
    assert result.asof == asof


# LLM-generated content at query #18
#--------------------------

def test_constructor_creates_valid_instance():
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

def test_constructor_creates_invertible_instance():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("2")
    rate = FXRate(ccy1, ccy2, date, value)
    inverted = ~rate
    assert inverted.ccy1 == ccy2
    assert inverted.ccy2 == ccy1
    assert inverted.date == date
    assert inverted.value == Decimal("0.5")

def test_constructor_creates_instance_with_same_currency():
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

def test_constructor_creates_instance_with_positive_decimal():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("0.0001")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.value == value


# LLM-generated content at query #19
#--------------------------

def test_constructor_creates_fxrate_with_correct_attributes():
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("2")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.date == date
    assert rate.value == value

def test_constructor_allows_unpacking():
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

def test_constructor_accepts_negative_value():
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("-1")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.value == value

def test_constructor_accepts_zero_value():
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("0")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.value == value

def test_constructor_accepts_same_currency_with_non_one_value():
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["EUR"]
    date = datetime.date.today()
    value = Decimal("5")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.value == value

def test_constructor_accepts_non_currency_types_for_ccy1():
    ccy1 = "EUR"
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("2")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1

def test_constructor_accepts_non_currency_types_for_ccy2():
    ccy1 = Currencies["EUR"]
    ccy2 = "USD"
    date = datetime.date.today()
    value = Decimal("2")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy2 == ccy2

def test_constructor_accepts_non_date_types_for_date():
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = "2023-01-01"
    value = Decimal("2")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.date == date

def test_constructor_accepts_non_decimal_types_for_value():
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = 2.5
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.value == value


# LLM-generated content at query #20
#--------------------------

def test_queries_returns_correct_rates():
    mock_service = Mock(spec=FXRateService)
    mock_service.query.side_effect = lambda ccy1, ccy2, asof, strict: Decimal('1.5') if (ccy1, ccy2, asof) == (Currency('USD'), Currency('EUR'), Date(2023, 1, 1)) else Decimal('0.8')
    queries = [(Currency('USD'), Currency('EUR'), Date(2023, 1, 1)), (Currency('GBP'), Currency('USD'), Date(2023, 1, 1))]
    result = list(mock_service.queries(queries, strict=False))
    assert result == [Decimal('1.5'), Decimal('0.8')]

def test_queries_with_strict_true_raises_error():
    mock_service = Mock(spec=FXRateService)
    mock_service.query.side_effect = LookupError("Rate not found")
    queries = [(Currency('USD'), Currency('EUR'), Date(2023, 1, 1))]
    with pytest.raises(LookupError):
        list(mock_service.queries(queries, strict=True))

def test_queries_with_strict_false_returns_none():
    mock_service = Mock(spec=FXRateService)
    mock_service.query.return_value = None
    queries = [(Currency('USD'), Currency('EUR'), Date(2023, 1, 1))]
    result = list(mock_service.queries(queries, strict=False))
    assert result == [None]

def test_queries_empty_iterable():
    mock_service = Mock(spec=FXRateService)
    queries = []
    result = list(mock_service.queries(queries, strict=False))
    assert result == []

def test_queries_calls_query_for_each_input():
    mock_service = Mock(spec=FXRateService)
    mock_service.query.return_value = Decimal('1.0')
    queries = [(Currency('USD'), Currency('EUR'), Date(2023, 1, 1)), (Currency('GBP'), Currency('USD'), Date(2023, 1, 1))]
    list(mock_service.queries(queries, strict=False))
    assert mock_service.query.call_count == 2

def test_queries_passes_strict_parameter():
    mock_service = Mock(spec=FXRateService)
    mock_service.query.return_value = Decimal('1.0')
    queries = [(Currency('USD'), Currency('EUR'), Date(2023, 1, 1))]
    list(mock_service.queries(queries, strict=True))
    mock_service.query.assert_called_with(Currency('USD'), Currency('EUR'), Date(2023, 1, 1), strict=True)


# LLM-generated content at query #21
#--------------------------

def test_queries_returns_correct_rates_for_multiple_queries():
    mock_service = Mock(spec=FXRateService)
    mock_service.query.side_effect = [Decimal('1.1'), Decimal('1.2'), None]
    queries = [(Currency('USD'), Currency('EUR'), Date(2023, 1, 1)), (Currency('GBP'), Currency('USD'), Date(2023, 1, 2)), (Currency('JPY'), Currency('EUR'), Date(2023, 1, 3))]
    mock_service.queries.return_value = [Decimal('1.1'), Decimal('1.2'), None]
    result = list(mock_service.queries(queries, strict=False))
    assert result == [Decimal('1.1'), Decimal('1.2'), None]

def test_queries_raises_lookup_error_in_strict_mode():
    mock_service = Mock(spec=FXRateService)
    mock_service.query.side_effect = [Decimal('1.1'), LookupError('Rate not found'), Decimal('1.3')]
    queries = [(Currency('USD'), Currency('EUR'), Date(2023, 1, 1)), (Currency('GBP'), Currency('USD'), Date(2023, 1, 2)), (Currency('JPY'), Currency('EUR'), Date(2023, 1, 3))]
    mock_service.queries.side_effect = LookupError('Rate not found')
    try:
        list(mock_service.queries(queries, strict=True))
        assert False
    except LookupError:
        assert True

def test_queries_handles_empty_queries_list():
    mock_service = Mock(spec=FXRateService)
    mock_service.queries.return_value = []
    result = list(mock_service.queries([], strict=False))
    assert result == []

def test_queries_returns_none_for_missing_rates_in_non_strict_mode():
    mock_service = Mock(spec=FXRateService)
    mock_service.query.side_effect = [None, None, None]
    queries = [(Currency('USD'), Currency('EUR'), Date(2023, 1, 1)), (Currency('GBP'), Currency('USD'), Date(2023, 1, 2)), (Currency('JPY'), Currency('EUR'), Date(2023, 1, 3))]
    mock_service.queries.return_value = [None, None, None]
    result = list(mock_service.queries(queries, strict=False))
    assert result == [None, None, None]

def test_queries_passes_strict_flag_to_query_method():
    mock_service = Mock(spec=FXRateService)
    queries = [(Currency('USD'), Currency('EUR'), Date(2023, 1, 1))]
    mock_service.queries.return_value = [Decimal('1.1')]
    result = list(mock_service.queries(queries, strict=True))
    mock_service.queries.assert_called_with(queries, strict=True)
    assert result == [Decimal('1.1')]


# LLM-generated content at query #22
#--------------------------

def test_query_returns_fxrate_for_valid_currency_pair_and_date():
    service = FXRateService()
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    result = service.query(usd, eur, asof, strict=False)
    assert isinstance(result, FXRate) or result is None

def test_query_returns_none_for_nonexistent_pair_when_strict_false():
    service = FXRateService()
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    result = service.query(usd, jpy, asof, strict=False)
    assert result is None

def test_query_raises_error_for_nonexistent_pair_when_strict_true():
    service = FXRateService()
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    try:
        service.query(usd, jpy, asof, strict=True)
        assert False
    except LookupError:
        assert True

def test_query_returns_same_rate_for_inverse_pair():
    service = FXRateService()
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    rate1 = service.query(usd, eur, asof, strict=False)
    rate2 = service.query(eur, usd, asof, strict=False)
    if rate1 is not None and rate2 is not None:
        assert rate1 == 1 / rate2

def test_query_returns_rate_for_same_currency():
    service = FXRateService()
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    result = service.query(usd, usd, asof, strict=False)
    assert result == FXRate.one()

def test_query_handles_different_currency_types():
    service = FXRateService()
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    btc = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    asof = Date(2023, 1, 1)
    result = service.query(usd, btc, asof, strict=False)
    assert isinstance(result, FXRate) or result is None

def test_query_with_currency_having_negative_decimals():
    service = FXRateService()
    zzz = Currency.of("ZZZ", "Weird Crypto", -1, CurrencyType.CRYPTO)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    result = service.query(zzz, usd, asof, strict=False)
    assert isinstance(result, FXRate) or result is None

def test_query_with_currency_having_zero_decimals():
    service = FXRateService()
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    asof = Date(2023, 1, 1)
    result = service.query(jpy, usd, asof, strict=False)
    assert isinstance(result, FXRate) or result is None


# LLM-generated content at query #23
#--------------------------

def test_constructor_creates_fxrate_with_correct_attributes():
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

def test_constructor_creates_fxrate_with_same_currency_and_value_one():
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

def test_constructor_creates_fxrate_with_positive_decimal_value():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("0.5")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.value == value

def test_constructor_creates_fxrate_with_fractional_value():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["USD"]
    ccy2 = Currencies["EUR"]
    date = datetime.date.today()
    value = Decimal("0.75")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.value == value


# LLM-generated content at query #24
#--------------------------

def test_constructor_creates_fxrate_with_correct_attributes():
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

def test_constructor_does_not_validate_input():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["USD"]
    date = datetime.date.today()
    value = Decimal("-1")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.value == value

def test_constructor_accepts_same_currency_with_non_one_value():
    import datetime
    from decimal import Decimal
    from pypara.currencies import Currencies
    ccy1 = Currencies["EUR"]
    ccy2 = Currencies["EUR"]
    date = datetime.date.today()
    value = Decimal("5")
    rate = FXRate(ccy1, ccy2, date, value)
    assert rate.ccy1 == ccy1
    assert rate.ccy2 == ccy2
    assert rate.value == value


