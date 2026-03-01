####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_Currency___eq__():
    # Test equality with same currency
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2

    # Test inequality with different currency code
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    assert not (usd == eur)

    # Test inequality with different currency name
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 == usd2)

    # Test inequality with different decimals
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 0, CurrencyType.MONEY)
    assert not (usd1 == usd2)

    # Test inequality with different currency type
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert not (usd1 == usd2)

    # Test inequality with non-Currency object
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd == "USD")
    assert not (usd == 123)
    assert not (usd == None)


# LLM-generated content at query #2
#--------------------------

```python
def test_Currency___gt__():
    # Test with different codes
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    assert usd > eur

    # Test with same code but different names
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars 2", 2, CurrencyType.MONEY)
    assert usd1 > usd2

    # Test with same code and name but different decimals
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert usd1 > usd2

    # Test with same code, name, decimals but different types
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert usd1 > usd2

    # Test with same attributes (should be False)
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 > usd2)


# LLM-generated content at query #3
#--------------------------

```python
def test_CurrencyRegistry___len__():
    registry = CurrencyRegistry()
    assert len(registry) == 0

    with registry as register:
        register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        register(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))

    assert len(registry) == 2


# LLM-generated content at query #4
#--------------------------

```python
def test_Currency():
    # Test valid currency creation
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.code == "USD"
    assert usd.name == "US Dollars"
    assert usd.decimals == 2
    assert usd.type == CurrencyType.MONEY
    assert usd.quantize(Decimal("1.005")) == Decimal("1.00")
    assert usd.quantize(Decimal("1.015")) == Decimal("1.02")

    # Test JPY with 0 decimals
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy.code == "JPY"
    assert jpy.name == "Japanese Yen"
    assert jpy.decimals == 0
    assert jpy.type == CurrencyType.MONEY
    assert jpy.quantize(Decimal("0.5")) == Decimal("0")
    assert jpy.quantize(Decimal("1.5")) == Decimal("2")

    # Test currency with -1 decimals
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert zzz.code == "ZZZ"
    assert zzz.name == "Some weird currency"
    assert zzz.decimals == -1
    assert zzz.type == CurrencyType.CRYPTO
    assert zzz.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert zzz.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")

    # Test equality and hash
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2
    assert not (usd1 == usdx)
    assert hash(usd1) == hash(usd2)
    assert not (hash(usd1) == hash(usdx))

    # Test invalid currency code
    with pytest.raises(ProgrammingError):
        Currency.of("123", "Invalid Code", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("usd", "Lowercase Code", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD$", "Special Char Code", 2, CurrencyType.MONEY)

    # Test invalid currency name
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "  Trim Me  ", 2, CurrencyType.MONEY)

    # Test invalid decimals
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", -2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", "2", CurrencyType.MONEY)

    # Test invalid currency type
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", 2, "MONEY")


# LLM-generated content at query #5
#--------------------------

```python
def test_Currency___repr__():
    currency = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert repr(currency) == "Currency(code='USD', name='US Dollars', decimals=2, type=CurrencyType.MONEY, quantizer=Decimal('0.01'), hashcache={})".format(hash(('USD', 'US Dollars', 2, CurrencyType.MONEY, Decimal('0.01'))))


# LLM-generated content at query #6
#--------------------------

```python
def test_CurrencyRegistry___getitem__():
    # Setup
    registry = CurrencyRegistry()
    with registry as register:
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
        register(usd)
        register(eur)

    # Test existing currency
    assert registry["USD"] == usd
    assert registry["EUR"] == eur

    # Test non-existing currency
    with pytest.raises(CurrencyLookupError) as excinfo:
        _ = registry["NON-EXISTING"]
    assert str(excinfo.value) == "Currency identified by code 'NON-EXISTING' does not exist"


# LLM-generated content at query #7
#--------------------------

```python
def test_Currency():
    # Test valid currency creation
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.code == "USD"
    assert usd.name == "US Dollars"
    assert usd.decimals == 2
    assert usd.type == CurrencyType.MONEY
    assert usd.quantize(Decimal("1.005")) == Decimal("1.00")
    assert usd.quantize(Decimal("1.015")) == Decimal("1.02")

    # Test currency with zero decimals
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy.code == "JPY"
    assert jpy.name == "Japanese Yen"
    assert jpy.decimals == 0
    assert jpy.type == CurrencyType.MONEY
    assert jpy.quantize(Decimal("0.5")) == Decimal("0")
    assert jpy.quantize(Decimal("1.5")) == Decimal("2")

    # Test currency with negative decimals
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert zzz.code == "ZZZ"
    assert zzz.name == "Some weird currency"
    assert zzz.decimals == -1
    assert zzz.type == CurrencyType.CRYPTO
    assert zzz.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert zzz.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")

    # Test equality and hash
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2
    assert not (usd1 == usdx)
    assert hash(usd1) == hash(usd2)
    assert not (hash(usd1) == hash(usdx))

    # Test invalid currency code
    with pytest.raises(ProgrammingError):
        Currency.of("123", "Invalid Code", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("usd", "Lowercase Code", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("U$D", "Non-Alpha Code", 2, CurrencyType.MONEY)

    # Test invalid currency name
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "  Trimmed Name  ", 2, CurrencyType.MONEY)

    # Test invalid decimals
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", -2, CurrencyType.MONEY)

    # Test invalid currency type
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", 2, "Invalid Type")


# LLM-generated content at query #8
#--------------------------

```python
def test_Currency___repr__():
    currency = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert repr(currency) == "Currency(code='USD', name='US Dollars', decimals=2, type=CurrencyType.MONEY, quantizer=Decimal('0.01'), hashcache=-123456789)"


# LLM-generated content at query #9
#--------------------------

```python
def test_Currency___ge__():
    # Test equality case
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 >= usd2
    assert usd2 >= usd1

    # Test greater than case
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert usd >= jpy
    assert not (jpy >= usd)

    # Test less than case
    assert not (jpy >= usd)

    # Test with different currency types
    gold = Currency.of("XAU", "Gold", 2, CurrencyType.METAL)
    btc = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    assert gold >= btc or btc >= gold  # One should be greater
    assert not (gold >= btc and btc >= gold)  # They shouldn't be equal

    # Test with different decimals
    weird1 = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    weird2 = Currency.of("YYY", "Another weird currency", 0, CurrencyType.CRYPTO)
    assert weird1 >= weird2 or weird2 >= weird1
    assert not (weird1 >= weird2 and weird2 >= weird1)


# LLM-generated content at query #10
#--------------------------

```python
def test_CurrencyRegistry___new__():
    # Test that CurrencyRegistry is a singleton
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2

    # Test that the singleton instance is created only once
    assert CurrencyRegistry._CurrencyRegistry__instance is not None
    assert CurrencyRegistry._CurrencyRegistry__instance is registry1
    assert CurrencyRegistry._CurrencyRegistry__instance is registry2


# LLM-generated content at query #11
#--------------------------

```python
def test_Currency___delattr__():
    with pytest.raises(AttributeError):
        currency = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
        del currency.code


# LLM-generated content at query #12
#--------------------------

```python
def test_Currency():
    # Test valid currency creation
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.code == "USD"
    assert usd.name == "US Dollars"
    assert usd.decimals == 2
    assert usd.type == CurrencyType.MONEY
    assert usd.quantize(Decimal("1.005")) == Decimal("1.00")
    assert usd.quantize(Decimal("1.015")) == Decimal("1.02")

    # Test currency with zero decimals
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy.code == "JPY"
    assert jpy.name == "Japanese Yen"
    assert jpy.decimals == 0
    assert jpy.type == CurrencyType.MONEY
    assert jpy.quantize(Decimal("0.5")) == Decimal("0")
    assert jpy.quantize(Decimal("1.5")) == Decimal("2")

    # Test currency with variable precision
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert zzz.code == "ZZZ"
    assert zzz.name == "Some weird currency"
    assert zzz.decimals == -1
    assert zzz.type == CurrencyType.CRYPTO
    assert zzz.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert zzz.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")

    # Test equality and hash
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2
    assert not (usd1 == usdx)
    assert hash(usd1) == hash(usd2)
    assert not (hash(usd1) == hash(usdx))

    # Test invalid currency code
    with pytest.raises(ProgrammingError):
        Currency.of("usd", "US Dollars", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD1", "US Dollars", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of(123, "US Dollars", 2, CurrencyType.MONEY)

    # Test invalid currency name
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", " US Dollars", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars ", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", 123, 2, CurrencyType.MONEY)

    # Test invalid decimals
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", -2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", "2", CurrencyType.MONEY)

    # Test invalid currency type
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", 2, "MONEY")


# LLM-generated content at query #13
#--------------------------

```python
def test_Currency___eq__():
    # Test equality with same currency
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2

    # Test inequality with different currency codes
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert not (usd == eur)

    # Test inequality with different currency names
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 == usd2)

    # Test inequality with different decimals
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 0, CurrencyType.MONEY)
    assert not (usd1 == usd2)

    # Test inequality with different currency types
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert not (usd1 == usd2)

    # Test inequality with non-Currency object
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd == "USD")


# LLM-generated content at query #14
#--------------------------

```python
def test_CurrencyRegistry___enter__():
    # Test that __enter__ returns the __register method
    registry = CurrencyRegistry()
    with registry as register:
        assert callable(register)
        assert register.__name__ == "_CurrencyRegistry__register"

    # Test that __enter__ sets __ctx_open to True
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open is False
    with registry:
        assert registry._CurrencyRegistry__ctx_open is True
    assert registry._CurrencyRegistry__ctx_open is False

    # Test that __enter__ allows adding currencies
    registry = CurrencyRegistry()
    test_currency = Currency.of("TEST", "Test Currency", 2, CurrencyType.MONEY)
    with registry as register:
        register(test_currency)
    assert "TEST" in registry
    assert registry["TEST"] == test_currency


# LLM-generated content at query #15
#--------------------------

```python
def test_CurrencyRegistry_get():
    # Test with existing currency code
    assert CurrencyRegistry().get("USD") == CurrencyRegistry()["USD"]

    # Test with non-existing currency code and no default
    assert CurrencyRegistry().get("NON-EXISTING") is None

    # Test with non-existing currency code and a default
    default_currency = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    assert CurrencyRegistry().get("NON-EXISTING", default=default_currency) == default_currency

    # Test with existing currency code and a default (default should be ignored)
    assert CurrencyRegistry().get("USD", default=default_currency) == CurrencyRegistry()["USD"]


# LLM-generated content at query #16
#--------------------------

```python
def test_CurrencyLookupError():
    code = "XYZ"
    error = CurrencyLookupError(code)
    assert error.code == code
    assert str(error) == f"Currency identified by code '{code}' does not exist"


# LLM-generated content at query #17
#--------------------------

```python
def test_CurrencyRegistry___contains__():
    # Create a CurrencyRegistry instance
    registry = CurrencyRegistry()

    # Enter the registry population context
    with registry as register:
        # Add some currencies
        register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        register(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))
        register(Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY))

    # Test __contains__ with existing currencies
    assert "USD" in registry
    assert "EUR" in registry
    assert "JPY" in registry

    # Test __contains__ with non-existing currencies
    assert "GBP" not in registry
    assert "XYZ" not in registry


# LLM-generated content at query #18
#--------------------------

```python
def test_Currency___hash__():
    # Test with same currencies
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert hash(usd1) == hash(usd2)

    # Test with different currencies
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert hash(usd1) != hash(usdx)

    # Test with different currency codes
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert hash(usd1) != hash(jpy)

    # Test with different currency types
    btc = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    assert hash(usd1) != hash(btc)

    # Test with different decimals
    weird = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert hash(usd1) != hash(weird)


# LLM-generated content at query #19
#--------------------------

```python
def test_CurrencyRegistry():
    # Test singleton behavior
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2

    # Test initial state
    assert len(registry1) == 0
    assert registry1.all == []
    assert registry1.codes == []
    assert registry1.codenames == []

    # Test context manager
    with registry1 as register:
        currency = Currency.of("TEST", "Test Currency", 2, CurrencyType.MONEY)
        register(currency)

    assert len(registry1) == 1
    assert registry1.has("TEST")
    assert registry1["TEST"] == currency
    assert registry1.all == [currency]
    assert registry1.codes == ["TEST"]
    assert registry1.codenames == [("TEST", "Test Currency")]

    # Test duplicate registration
    with pytest.raises(ValueError):
        with registry1 as register:
            register(currency)

    # Test context manager error
    with pytest.raises(ProgrammingError):
        registry1.__register(currency)

    # Test get method
    assert registry1.get("TEST") == currency
    assert registry1.get("NONEXISTENT") is None
    assert registry1.get("NONEXISTENT", default=currency) == currency

    # Test lookup error
    with pytest.raises(CurrencyLookupError):
        registry1["NONEXISTENT"]


# LLM-generated content at query #20
#--------------------------

```python
def test_CurrencyRegistry___enter__():
    # Test that __enter__ returns the __register method
    registry = CurrencyRegistry()
    with registry as register:
        assert callable(register)
        assert register.__name__ == "__register"

    # Test that __enter__ sets __ctx_open to True
    registry = CurrencyRegistry()
    with registry:
        assert registry._CurrencyRegistry__ctx_open is True

    # Test that __exit__ sets __ctx_open to False
    registry = CurrencyRegistry()
    with registry:
        pass
    assert registry._CurrencyRegistry__ctx_open is False


# LLM-generated content at query #21
#--------------------------

```python
def test_CurrencyRegistry():
    # Test singleton behavior
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2

    # Test initial state
    assert len(registry1) == 0
    assert registry1.all == []
    assert registry1.codes == []
    assert registry1.codenames == []

    # Test context manager
    with registry1 as register:
        # Test adding currencies
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
        register(usd)
        register(eur)

        # Test duplicate currency
        with pytest.raises(ValueError):
            register(usd)

    # Test post-context state
    assert len(registry1) == 2
    assert registry1.all == [eur, usd]  # Sorted by code
    assert registry1.codes == ["EUR", "USD"]
    assert registry1.codenames == [("EUR", "Euro"), ("USD", "US Dollar")]

    # Test currency lookup
    assert registry1["USD"] == usd
    assert registry1["EUR"] == eur
    assert registry1.has("USD")
    assert registry1.has("EUR")
    assert not registry1.has("JPY")
    assert registry1.get("USD") == usd
    assert registry1.get("JPY") is None
    assert registry1.get("JPY", default=usd) == usd

    # Test lookup error
    with pytest.raises(CurrencyLookupError):
        registry1["JPY"]

    # Test adding outside context
    with pytest.raises(ProgrammingError):
        registry1._CurrencyRegistry__register(usd)


# LLM-generated content at query #22
#--------------------------

```python
def test_Currency___lt__():
    # Test with different codes
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert (usd < eur) == ("USD" < "EUR")

    # Test with same code but different names
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 < usd2)

    # Test with same code and name but different decimals
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 0, CurrencyType.MONEY)
    assert (usd1 < usd2) == (2 < 0)

    # Test with same code, name, and decimals but different types
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert (usd1 < usd2) == (CurrencyType.MONEY < CurrencyType.CRYPTO)

    # Test with different quantizers
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert (usd < jpy) == (make_quantizer(2) < ZERO)

    # Test with non-Currency object
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd < "not a currency")


# LLM-generated content at query #23
#--------------------------

```python
def test_Currency___lt__():
    # Test with different codes
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert (usd < eur) == ("USD" < "EUR")

    # Test with same code but different names
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars 2", 2, CurrencyType.MONEY)
    assert (usd1 < usd2) == (("USD", "US Dollars", 2, CurrencyType.MONEY, usd1.quantizer) <
                            ("USD", "US Dollars 2", 2, CurrencyType.MONEY, usd2.quantizer))

    # Test with same code and name but different decimals
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert (usd1 < usd2) == (("USD", "US Dollars", 2, CurrencyType.MONEY, usd1.quantizer) <
                            ("USD", "US Dollars", 3, CurrencyType.MONEY, usd2.quantizer))

    # Test with same code, name, and decimals but different types
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert (usd1 < usd2) == (("USD", "US Dollars", 2, CurrencyType.MONEY, usd1.quantizer) <
                            ("USD", "US Dollars", 2, CurrencyType.CRYPTO, usd2.quantizer))

    # Test with same code, name, decimals, and type but different quantizers
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", -1, CurrencyType.MONEY)
    assert (usd1 < usd2) == (("USD", "US Dollars", 2, CurrencyType.MONEY, usd1.quantizer) <
                            ("USD", "US Dollars", -1, CurrencyType.MONEY, usd2.quantizer))

    # Test with same currency
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd < usd)


# LLM-generated content at query #24
#--------------------------

```python
def test_CurrencyLookupError():
    # Test initialization and message format
    error = CurrencyLookupError("XYZ")
    assert error.code == "XYZ"
    assert str(error) == "Currency identified by code 'XYZ' does not exist"

    # Test inheritance from LookupError
    assert isinstance(error, LookupError)


# LLM-generated content at query #25
#--------------------------

```python
def test_Currency___ge__():
    # Test equality (should be True)
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 >= usd2

    # Test greater than (should be True based on order)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert eur >= usd

    # Test less than (should be False)
    assert not (usd >= eur)

    # Test with different attributes (should be False)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert not (usd >= usdx)
    assert not (usdx >= usd)


# LLM-generated content at query #26
#--------------------------

```python
def test_Currency___lt__():
    # Test with different codes
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert usd < eur or eur < usd  # One must be less than the other

    # Test with same code but different names
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars 2", 2, CurrencyType.MONEY)
    assert not (usd1 < usd2) and not (usd2 < usd1)  # Should be equal, not less

    # Test with same code and name but different decimals
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert usd1 < usd2 or usd2 < usd1  # One must be less than the other

    # Test with same code, name, decimals but different types
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert usd1 < usd2 or usd2 < usd1  # One must be less than the other

    # Test with same code, name, decimals, type but different quantizers (shouldn't happen in practice)
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal("0.01"), hash(("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal("0.01"))))
    assert not (usd1 < usd2) and not (usd2 < usd1)  # Should be equal, not less

    # Test with non-Currency object
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    try:
        usd < "not a currency"
        assert False, "Expected TypeError"
    except TypeError:
        pass


# LLM-generated content at query #27
#--------------------------

```python
def test_CurrencyRegistry___len__():
    registry = CurrencyRegistry()
    assert len(registry) == 0

    with registry as register:
        register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        register(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))
        register(Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY))

    assert len(registry) == 3


# LLM-generated content at query #28
#--------------------------

```python
def test_CurrencyRegistry___exit__():
    # Setup
    registry = CurrencyRegistry()
    test_currency1 = Currency.of("TEST1", "Test Currency 1", 2, CurrencyType.MONEY)
    test_currency2 = Currency.of("TEST2", "Test Currency 2", 0, CurrencyType.MONEY)

    # Test context manager behavior
    with registry as register:
        register(test_currency1)
        register(test_currency2)

    # Verify the registry is properly sorted after exit
    assert registry.codes == ["TEST1", "TEST2"]
    assert registry.all == [test_currency1, test_currency2]
    assert registry.codenames == [("TEST1", "Test Currency 1"), ("TEST2", "Test Currency 2")]

    # Verify the context is closed
    assert not registry._CurrencyRegistry__ctx_open

    # Verify attempting to register outside context raises error
    with pytest.raises(ProgrammingError):
        registry._CurrencyRegistry__register(test_currency1)

    # Verify attempting to register duplicate currency raises error
    with CurrencyRegistry() as register:
        register(test_currency1)
        with pytest.raises(ValueError):
            register(test_currency1)


# LLM-generated content at query #29
#--------------------------

```python
def test_Currency___ge__():
    # Test with equal currencies
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 >= usd2
    assert usd2 >= usd1

    # Test with different currencies (same code but different name)
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 >= usdx)
    assert not (usdx >= usd1)

    # Test with different currencies (different code)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert usd >= jpy or jpy >= usd  # One should be greater than the other
    assert not (usd >= jpy and jpy >= usd)  # They should not be equal

    # Test with different currency types
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    btc = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    assert usd >= btc or btc >= usd  # One should be greater than the other
    assert not (usd >= btc and btc >= usd)  # They should not be equal


# LLM-generated content at query #30
#--------------------------

```python
def test_Currency___ge__():
    # Test equality cases
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 >= usd2
    assert usd2 >= usd1

    # Test greater than cases (based on hash comparison)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert usd >= eur or eur >= usd  # One should be greater than the other

    # Test with different attributes
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert usd >= jpy or jpy >= usd  # One should be greater than the other

    # Test with non-Currency object
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd >= "not a currency")


# LLM-generated content at query #31
#--------------------------

```python
def test_Currency___delattr__():
    # Create a currency instance
    currency = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)

    # Attempt to delete an attribute
    with pytest.raises(AttributeError):
        del currency.code

    # Verify the attribute still exists
    assert hasattr(currency, 'code')
    assert currency.code == "USD"


# LLM-generated content at query #32
#--------------------------

```python
def test_CurrencyRegistry_has():
    # Test with existing currency
    assert Currencies.has("USD") is True

    # Test with non-existing currency
    assert Currencies.has("NONEXISTENT") is False

    # Test case sensitivity
    assert Currencies.has("usd") is False


# LLM-generated content at query #33
#--------------------------

```python
def test_CurrencyRegistry___len__():
    # Create a new instance of CurrencyRegistry
    registry = CurrencyRegistry()

    # Initially, the registry should be empty
    assert len(registry) == 0

    # Populate the registry with some currencies
    with registry as register:
        register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        register(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))
        register(Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY))

    # After adding 3 currencies, the length should be 3
    assert len(registry) == 3


# LLM-generated content at query #34
#--------------------------

```python
def test_Currency():
    # Test valid currency creation
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.code == "USD"
    assert usd.name == "US Dollars"
    assert usd.decimals == 2
    assert usd.type == CurrencyType.MONEY
    assert usd.quantize(Decimal("1.005")) == Decimal("1.00")
    assert usd.quantize(Decimal("1.015")) == Decimal("1.02")

    # Test currency with zero decimals
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy.code == "JPY"
    assert jpy.name == "Japanese Yen"
    assert jpy.decimals == 0
    assert jpy.type == CurrencyType.MONEY
    assert jpy.quantize(Decimal("0.5")) == Decimal("0")
    assert jpy.quantize(Decimal("1.5")) == Decimal("2")

    # Test currency with negative decimals
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert zzz.code == "ZZZ"
    assert zzz.name == "Some weird currency"
    assert zzz.decimals == -1
    assert zzz.type == CurrencyType.CRYPTO
    assert zzz.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert zzz.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")

    # Test equality and hash
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2
    assert not (usd1 == usdx)
    assert hash(usd1) == hash(usd2)
    assert not (hash(usd1) == hash(usdx))

    # Test invalid currency code
    try:
        Currency.of("123", "Invalid Code", 2, CurrencyType.MONEY)
        assert False, "Expected ProgrammingError for invalid code"
    except ProgrammingError:
        pass

    try:
        Currency.of("abc", "Invalid Code", 2, CurrencyType.MONEY)
        assert False, "Expected ProgrammingError for lowercase code"
    except ProgrammingError:
        pass

    # Test invalid currency name
    try:
        Currency.of("USD", "", 2, CurrencyType.MONEY)
        assert False, "Expected ProgrammingError for empty name"
    except ProgrammingError:
        pass

    try:
        Currency.of("USD", "  Trimmed  ", 2, CurrencyType.MONEY)
        assert False, "Expected ProgrammingError for untrimmed name"
    except ProgrammingError:
        pass

    # Test invalid decimals
    try:
        Currency.of("USD", "US Dollars", -2, CurrencyType.MONEY)
        assert False, "Expected ProgrammingError for invalid decimals"
    except ProgrammingError:
        pass

    # Test invalid currency type
    try:
        Currency.of("USD", "US Dollars", 2, "Invalid Type")
        assert False, "Expected ProgrammingError for invalid type"
    except ProgrammingError:
        pass


# LLM-generated content at query #35
#--------------------------

```python
def test_Currency___ge__():
    # Test equal currencies
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 >= usd2
    assert usd2 >= usd1

    # Test different currencies (same code but different name)
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 >= usd2)
    assert not (usd2 >= usd1)

    # Test different currencies (different code)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert not (usd >= jpy)
    assert not (jpy >= usd)

    # Test with non-Currency object
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd >= "USD")
    assert not (usd >= None)


# LLM-generated content at query #36
#--------------------------

```python
def test_CurrencyRegistry___new__():
    # Test that CurrencyRegistry is a singleton
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2

    # Test that the singleton instance is properly initialized
    assert hasattr(registry1, '_CurrencyRegistry__registry')
    assert hasattr(registry1, '_CurrencyRegistry__currencies')
    assert hasattr(registry1, '_CurrencyRegistry__codes')
    assert hasattr(registry1, '_CurrencyRegistry__codenames')
    assert hasattr(registry1, '_CurrencyRegistry__ctx_open')
    assert registry1._CurrencyRegistry__ctx_open is False


# LLM-generated content at query #37
#--------------------------

```python
def test_CurrencyRegistry___contains__():
    # Test with existing currency
    assert "USD" in Currencies

    # Test with non-existing currency
    assert "NONEXISTENT" not in Currencies

    # Test with empty string
    assert "" not in Currencies

    # Test with non-string type
    assert 123 not in Currencies


# LLM-generated content at query #38
#--------------------------

```python
def test_CurrencyRegistry___exit__():
    registry = CurrencyRegistry()
    registry._CurrencyRegistry__ctx_open = True
    registry._CurrencyRegistry__registry = OrderedDict([
        ("USD", Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)),
        ("EUR", Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)),
        ("JPY", Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY))
    ])

    registry.__exit__(None, None, None)

    assert registry._CurrencyRegistry__ctx_open is False
    assert registry._CurrencyRegistry__codes == ["EUR", "JPY", "USD"]
    assert registry._CurrencyRegistry__codenames == [
        ("EUR", "Euro"),
        ("JPY", "Japanese Yen"),
        ("USD", "US Dollar")
    ]
    assert registry._CurrencyRegistry__currencies == [
        Currency.of("EUR", "Euro", 2, CurrencyType.MONEY),
        Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY),
        Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    ]


# LLM-generated content at query #39
#--------------------------

```python
def test_CurrencyRegistry_get():
    # Setup
    registry = CurrencyRegistry()
    with registry as register:
        register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        register(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))

    # Test existing currency with no default
    assert registry.get("USD").code == "USD"
    assert registry.get("EUR").code == "EUR"

    # Test non-existing currency with no default
    assert registry.get("XYZ") is None

    # Test non-existing currency with default
    default_currency = Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY)
    assert registry.get("XYZ", default=default_currency) == default_currency

    # Test that default is not used when currency exists
    assert registry.get("USD", default=default_currency).code == "USD"


# LLM-generated content at query #40
#--------------------------

```python
def test_Currency___delattr__():
    # Create a currency instance
    currency = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)

    # Attempt to delete an attribute and verify it raises an error
    with pytest.raises(AttributeError):
        del currency.code

    with pytest.raises(AttributeError):
        del currency.name

    with pytest.raises(AttributeError):
        del currency.decimals

    with pytest.raises(AttributeError):
        del currency.type

    with pytest.raises(AttributeError):
        del currency.quantizer

    with pytest.raises(AttributeError):
        del currency.hashcache


# LLM-generated content at query #41
#--------------------------

```python
def test_CurrencyRegistry_has():
    # Test with existing currency
    assert Currencies.has("USD") is True

    # Test with non-existing currency
    assert Currencies.has("NONEXISTENT") is False

    # Test with empty string
    assert Currencies.has("") is False

    # Test with case sensitivity
    assert Currencies.has("usd") is False


# LLM-generated content at query #42
#--------------------------

```python
def test_Currency___setattr__():
    # Test that setting attributes on a frozen Currency instance raises an AttributeError
    currency = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)

    with pytest.raises(AttributeError):
        currency.code = "EUR"

    with pytest.raises(AttributeError):
        currency.name = "Euros"

    with pytest.raises(AttributeError):
        currency.decimals = 3

    with pytest.raises(AttributeError):
        currency.type = CurrencyType.CRYPTO

    with pytest.raises(AttributeError):
        currency.quantizer = Decimal("0.001")

    with pytest.raises(AttributeError):
        currency.hashcache = 12345


# LLM-generated content at query #43
#--------------------------

```python
def test_CurrencyRegistry():
    # Test singleton behavior
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2

    # Test initial state
    assert len(registry1) == 0
    assert registry1.all == []
    assert registry1.codes == []
    assert registry1.codenames == []

    # Test context manager behavior
    with registry1 as register:
        # Test adding a currency
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        register(usd)
        assert len(registry1) == 1
        assert "USD" in registry1
        assert registry1["USD"] == usd

        # Test adding duplicate currency raises error
        with pytest.raises(ValueError, match="Currency USD is already registered."):
            register(usd)

    # Test that registry is sorted after context exit
    with registry1 as register:
        jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
        eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
        register(jpy)
        register(eur)

    assert registry1.codes == ["EUR", "JPY", "USD"]
    assert registry1.codenames == [
        ("EUR", "Euro"),
        ("JPY", "Japanese Yen"),
        ("USD", "US Dollar")
    ]

    # Test get method
    assert registry1.get("USD") == usd
    assert registry1.get("NONEXISTENT") is None
    assert registry1.get("NONEXISTENT", default=usd) == usd

    # Test has method
    assert registry1.has("USD")
    assert not registry1.has("NONEXISTENT")

    # Test lookup error
    with pytest.raises(CurrencyLookupError, match="Currency identified by code 'NONEXISTENT' does not exist"):
        registry1["NONEXISTENT"]

    # Test adding outside context raises error
    with pytest.raises(ProgrammingError, match="Can not create currencies outside registry context."):
        registry1._CurrencyRegistry__register(usd)


# LLM-generated content at query #44
#--------------------------

```python
def test_CurrencyRegistry___len__():
    # Create a new instance of CurrencyRegistry
    registry = CurrencyRegistry()

    # Initially, the registry should be empty
    assert len(registry) == 0

    # Add some currencies to the registry
    with registry as register:
        register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        register(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))
        register(Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY))

    # After adding 3 currencies, the length should be 3
    assert len(registry) == 3


# LLM-generated content at query #45
#--------------------------

```python
def test_CurrencyLookupError():
    # Test initialization and message
    error = CurrencyLookupError("XYZ")
    assert error.code == "XYZ"
    assert str(error) == "Currency identified by code 'XYZ' does not exist"

    # Test inheritance
    assert isinstance(error, LookupError)


# LLM-generated content at query #46
#--------------------------

```python
def test_Currency___le__():
    # Test equality cases
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 <= usd2
    assert usd2 <= usd1

    # Test less than cases
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert usd <= jpy or jpy <= usd  # One should be less than or equal to the other

    # Test with different attributes
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    crypto = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    assert usd <= crypto or crypto <= usd  # One should be less than or equal to the other

    # Test with same code but different other attributes
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert usd1 <= usd2 or usd2 <= usd1  # One should be less than or equal to the other


# LLM-generated content at query #47
#--------------------------

```python
def test_Currency___setattr__():
    # Create a currency instance
    currency = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)

    # Test that setting an attribute raises an AttributeError
    with pytest.raises(AttributeError):
        currency.code = "EUR"

    with pytest.raises(AttributeError):
        currency.name = "Euros"

    with pytest.raises(AttributeError):
        currency.decimals = 0

    with pytest.raises(AttributeError):
        currency.type = CurrencyType.CRYPTO

    with pytest.raises(AttributeError):
        currency.quantizer = Decimal("0.01")

    with pytest.raises(AttributeError):
        currency.hashcache = 123


# LLM-generated content at query #48
#--------------------------

```python
def test_CurrencyRegistry___contains__():
    # Setup
    registry = CurrencyRegistry()
    with registry as register:
        register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        register(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))

    # Test existing currency
    assert "USD" in registry
    assert "EUR" in registry

    # Test non-existing currency
    assert "XYZ" not in registry
    assert "NONEXISTENT" not in registry


# LLM-generated content at query #49
#--------------------------

```python
def test_Currency():
    # Test valid currency creation
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.code == "USD"
    assert usd.name == "US Dollars"
    assert usd.decimals == 2
    assert usd.type == CurrencyType.MONEY
    assert usd.quantize(Decimal("1.005")) == Decimal("1.00")
    assert usd.quantize(Decimal("1.015")) == Decimal("1.02")

    # Test currency with zero decimals
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy.code == "JPY"
    assert jpy.name == "Japanese Yen"
    assert jpy.decimals == 0
    assert jpy.type == CurrencyType.MONEY
    assert jpy.quantize(Decimal("0.5")) == Decimal("0")
    assert jpy.quantize(Decimal("1.5")) == Decimal("2")

    # Test currency with variable precision
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert zzz.code == "ZZZ"
    assert zzz.name == "Some weird currency"
    assert zzz.decimals == -1
    assert zzz.type == CurrencyType.CRYPTO
    assert zzz.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert zzz.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")

    # Test equality and hash
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2
    assert usd1 != usdx
    assert hash(usd1) == hash(usd2)
    assert hash(usd1) != hash(usdx)

    # Test invalid currency code
    with pytest.raises(ProgrammingError):
        Currency.of("123", "Invalid Code", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("usd", "Lowercase Code", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD$", "Special Char Code", 2, CurrencyType.MONEY)

    # Test invalid currency name
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "  Trimmed  ", 2, CurrencyType.MONEY)

    # Test invalid decimals
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", -2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", "2", CurrencyType.MONEY)

    # Test invalid currency type
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", 2, "MONEY")


# LLM-generated content at query #50
#--------------------------

```python
def test_Currency___ge__():
    # Test equality
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 >= usd2
    assert usd2 >= usd1

    # Test greater than
    usd3 = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert usd3 >= usd1
    assert not (usd1 >= usd3)

    # Test less than
    usd0 = Currency.of("USD", "US Dollars", 0, CurrencyType.MONEY)
    assert usd1 >= usd0
    assert not (usd0 >= usd1)

    # Test different currencies
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert usd1 >= jpy or jpy >= usd1  # Should be deterministic based on hash


# LLM-generated content at query #51
#--------------------------

```python
def test_CurrencyRegistry():
    # Test singleton behavior
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2

    # Test initial state
    assert len(registry1) == 0
    assert registry1.all == []
    assert registry1.codes == []
    assert registry1.codenames == []

    # Test context manager
    with registry1 as register:
        currency = Currency.of("TEST", "Test Currency", 2, CurrencyType.MONEY)
        register(currency)

    assert len(registry1) == 1
    assert registry1["TEST"] == currency
    assert registry1.all == [currency]
    assert registry1.codes == ["TEST"]
    assert registry1.codenames == [("TEST", "Test Currency")]

    # Test duplicate registration
    with pytest.raises(ValueError):
        with registry1 as register:
            register(currency)

    # Test context manager error
    with pytest.raises(ProgrammingError):
        registry1._CurrencyRegistry__register(currency)

    # Test has and get methods
    assert registry1.has("TEST")
    assert not registry1.has("NONEXISTENT")
    assert registry1.get("TEST") == currency
    assert registry1.get("NONEXISTENT") is None
    assert registry1.get("NONEXISTENT", default=currency) == currency

    # Test lookup error
    with pytest.raises(CurrencyLookupError):
        registry1["NONEXISTENT"]


# LLM-generated content at query #52
#--------------------------

```python
def test_CurrencyRegistry___enter__():
    registry = CurrencyRegistry()
    with registry as register:
        assert callable(register)
        assert register.__name__ == "_CurrencyRegistry__register"
    assert not registry._CurrencyRegistry__ctx_open


# LLM-generated content at query #53
#--------------------------

```python
def test_Currency___eq__():
    # Test equality with same currency
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2

    # Test inequality with different currency
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert not (usd == eur)

    # Test inequality with different name
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 == usd2)

    # Test inequality with different decimals
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 0, CurrencyType.MONEY)
    assert not (usd1 == usd2)

    # Test inequality with different type
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert not (usd1 == usd2)

    # Test inequality with non-Currency object
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd == "USD")

    # Test inequality with None
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd == None)


# LLM-generated content at query #54
#--------------------------

```python
def test_CurrencyRegistry___getitem__():
    registry = CurrencyRegistry()
    with registry:
        registry(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        registry(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))

    assert registry["USD"].code == "USD"
    assert registry["EUR"].code == "EUR"

    with pytest.raises(CurrencyLookupError):
        _ = registry["NON-EXISTING"]


# LLM-generated content at query #55
#--------------------------

```python
def test_Currency___eq__():
    # Test equality with same currency
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2

    # Test inequality with different currency
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 == usdx)

    # Test inequality with non-Currency object
    assert not (usd1 == "USD")

    # Test inequality with different attributes
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert not (usd1 == jpy)

    # Test with different decimals
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert not (usd1 == zzz)

    # Test with different type
    gold = Currency.of("XAU", "Gold", 2, CurrencyType.METAL)
    assert not (usd1 == gold)


# LLM-generated content at query #56
#--------------------------

```python
def test_Currency___repr__():
    currency = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert repr(currency) == "Currency(code='USD', name='US Dollars', decimals=2, type=CurrencyType.MONEY, quantizer=Decimal('0.01'), hashcache={})".format(hash(("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal('0.01'))))


# LLM-generated content at query #57
#--------------------------

```python
def test_CurrencyRegistry___contains__():
    # Create a new CurrencyRegistry instance
    registry = CurrencyRegistry()

    # Create a test currency
    test_currency = Currency.of("TEST", "Test Currency", 2, CurrencyType.MONEY)

    # Use context manager to add the currency
    with registry as register:
        register(test_currency)

    # Test that the currency is in the registry
    assert "TEST" in registry

    # Test that a non-existent currency is not in the registry
    assert "NONEXISTENT" not in registry


# LLM-generated content at query #58
#--------------------------

```python
def test_CurrencyRegistry___enter__():
    # Test that __enter__ returns the __register method
    registry = CurrencyRegistry()
    with registry as register:
        assert callable(register)
        assert register.__name__ == "__register"

    # Test that __enter__ sets the context flag
    assert not registry._CurrencyRegistry__ctx_open
    with registry:
        assert registry._CurrencyRegistry__ctx_open
    assert not registry._CurrencyRegistry__ctx_open


# LLM-generated content at query #59
#--------------------------

```python
def test_CurrencyRegistry___getitem__():
    registry = CurrencyRegistry()

    with registry as register:
        register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        register(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))

    assert registry["USD"].code == "USD"
    assert registry["EUR"].code == "EUR"

    with pytest.raises(CurrencyLookupError):
        _ = registry["NON-EXISTING"]


# LLM-generated content at query #60
#--------------------------

```python
def test_CurrencyRegistry___getitem__():
    # Setup
    registry = CurrencyRegistry()
    with registry as register:
        register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        register(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))

    # Test existing currency
    usd = registry["USD"]
    assert usd.code == "USD"
    assert usd.name == "US Dollar"
    assert usd.decimals == 2
    assert usd.type == CurrencyType.MONEY

    # Test non-existing currency
    with pytest.raises(CurrencyLookupError) as excinfo:
        _ = registry["XYZ"]
    assert str(excinfo.value) == "Currency identified by code 'XYZ' does not exist"


# LLM-generated content at query #61
#--------------------------

```python
def test_Currency___delattr__():
    with pytest.raises(AttributeError):
        currency = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
        del currency.code


# LLM-generated content at query #62
#--------------------------

```python
def test_CurrencyRegistry___contains__():
    # Test with existing currency
    assert "USD" in Currencies

    # Test with non-existing currency
    assert "NONEXISTENT" not in Currencies

    # Test with empty string
    assert "" not in Currencies

    # Test with invalid type (should not raise exception, just return False)
    assert 123 not in Currencies


# LLM-generated content at query #63
#--------------------------

```python
def test_Currency___eq__():
    # Test equality with same currency
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2

    # Test inequality with different currency
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert not (usd == eur)

    # Test inequality with non-Currency object
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd == "USD")

    # Test inequality with different attributes
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert not (usd1 == usd2)

    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert not (usd1 == usd2)

    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    assert not (usd1 == usd2)


# LLM-generated content at query #64
#--------------------------

```python
def test_Currency___lt__():
    # Test with same currency
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 < usd2)
    assert not (usd2 < usd1)

    # Test with different currencies
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert (jpy < usd) or (usd < jpy)  # One should be less than the other

    # Test with different decimals
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    crypto = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert (crypto < usd) or (usd < crypto)  # One should be less than the other

    # Test with different types
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    gold = Currency.of("XAU", "Gold", 2, CurrencyType.METAL)
    assert (gold < usd) or (usd < gold)  # One should be less than the other


# LLM-generated content at query #65
#--------------------------

```python
def test_Currency___ge__():
    # Test equality cases
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 >= usd2
    assert usd2 >= usd1

    # Test greater than cases
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert usd >= jpy  # Assuming USD > JPY based on hash comparison

    # Test less than cases
    assert jpy >= usd is False

    # Test with different attributes
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert usd >= usdx is False
    assert usdx >= usd is False


# LLM-generated content at query #66
#--------------------------

```python
def test_Currency_quantize():
    # Test with USD (2 decimals)
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert USD.quantize(Decimal("1.005")) == Decimal("1.00")
    assert USD.quantize(Decimal("1.015")) == Decimal("1.02")
    assert USD.quantize(Decimal("1.000")) == Decimal("1.00")
    assert USD.quantize(Decimal("1.999")) == Decimal("2.00")

    # Test with JPY (0 decimals)
    JPY = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert JPY.quantize(Decimal("0.5")) == Decimal("0")
    assert JPY.quantize(Decimal("1.5")) == Decimal("2")
    assert JPY.quantize(Decimal("1.0")) == Decimal("1")
    assert JPY.quantize(Decimal("2.9")) == Decimal("3")

    # Test with a weird currency (-1 decimals)
    ZZZ = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert ZZZ.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert ZZZ.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")
    assert ZZZ.quantize(Decimal("1.0000000000000")) == Decimal("1.000000000000")
    assert ZZZ.quantize(Decimal("1.0000000000009")) == Decimal("1.000000000001")

    # Test with negative values
    assert USD.quantize(Decimal("-1.005")) == Decimal("-1.00")
    assert USD.quantize(Decimal("-1.015")) == Decimal("-1.02")
    assert JPY.quantize(Decimal("-0.5")) == Decimal("0")
    assert JPY.quantize(Decimal("-1.5")) == Decimal("-2")
    assert ZZZ.quantize(Decimal("-1.0000000000005")) == Decimal("-1.000000000000")
    assert ZZZ.quantize(Decimal("-1.0000000000015")) == Decimal("-1.000000000002")


# LLM-generated content at query #67
#--------------------------

```python
def test_Currency___lt__():
    # Test with different codes
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert (usd < eur) == ("USD" < "EUR")

    # Test with same code but different names
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars 2", 2, CurrencyType.MONEY)
    assert (usd1 < usd2) == (("USD", "US Dollars", 2, CurrencyType.MONEY, usd1.quantizer) <
                            ("USD", "US Dollars 2", 2, CurrencyType.MONEY, usd2.quantizer))

    # Test with same code and name but different decimals
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert (usd1 < usd2) == (("USD", "US Dollars", 2, CurrencyType.MONEY, usd1.quantizer) <
                            ("USD", "US Dollars", 3, CurrencyType.MONEY, usd2.quantizer))

    # Test with same code, name, and decimals but different types
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert (usd1 < usd2) == (("USD", "US Dollars", 2, CurrencyType.MONEY, usd1.quantizer) <
                            ("USD", "US Dollars", 2, CurrencyType.CRYPTO, usd2.quantizer))

    # Test with same code, name, decimals, and type but different quantizers
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", -1, CurrencyType.MONEY)
    assert (usd1 < usd2) == (("USD", "US Dollars", 2, CurrencyType.MONEY, usd1.quantizer) <
                            ("USD", "US Dollars", -1, CurrencyType.MONEY, usd2.quantizer))

    # Test with same currency
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd < usd)


# LLM-generated content at query #68
#--------------------------

```python
def test_CurrencyLookupError():
    code = "XYZ"
    error = CurrencyLookupError(code)
    assert error.code == code
    assert str(error) == f"Currency identified by code '{code}' does not exist"


# LLM-generated content at query #69
#--------------------------

```python
def test_CurrencyRegistry___contains__():
    # Test with existing currency
    assert "USD" in Currencies

    # Test with non-existing currency
    assert "NONEXISTENT" not in Currencies

    # Test with empty string
    assert "" not in Currencies

    # Test with non-string type (should not raise exception, just return False)
    assert 123 not in Currencies


# LLM-generated content at query #70
#--------------------------

```python
def test_Currency():
    # Test valid currency creation
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.code == "USD"
    assert usd.name == "US Dollars"
    assert usd.decimals == 2
    assert usd.type == CurrencyType.MONEY
    assert usd.quantize(Decimal("1.005")) == Decimal("1.00")
    assert usd.quantize(Decimal("1.015")) == Decimal("1.02")

    # Test currency with zero decimals
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy.code == "JPY"
    assert jpy.name == "Japanese Yen"
    assert jpy.decimals == 0
    assert jpy.type == CurrencyType.MONEY
    assert jpy.quantize(Decimal("0.5")) == Decimal("0")
    assert jpy.quantize(Decimal("1.5")) == Decimal("2")

    # Test currency with negative decimals
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert zzz.code == "ZZZ"
    assert zzz.name == "Some weird currency"
    assert zzz.decimals == -1
    assert zzz.type == CurrencyType.CRYPTO
    assert zzz.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert zzz.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")

    # Test equality and hash
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2
    assert usd1 != usdx
    assert hash(usd1) == hash(usd2)
    assert hash(usd1) != hash(usdx)

    # Test invalid currency code
    with pytest.raises(ProgrammingError):
        Currency.of("123", "Invalid Code", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("abc", "Invalid Code", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("Abc", "Invalid Code", 2, CurrencyType.MONEY)

    # Test invalid currency name
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", " ", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", " US Dollars", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars ", 2, CurrencyType.MONEY)

    # Test invalid decimals
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", -2, CurrencyType.MONEY)

    # Test invalid currency type
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", 2, "Invalid Type")


# LLM-generated content at query #71
#--------------------------

```python
def test_Currency___eq__():
    # Test equality with same currency
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2

    # Test inequality with different currency
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 == usdx)

    # Test inequality with non-Currency object
    assert not (usd1 == "USD")

    # Test inequality with different attributes
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert not (usd1 == jpy)


# LLM-generated content at query #72
#--------------------------

```python
def test_CurrencyLookupError():
    code = "XYZ"
    error = CurrencyLookupError(code)
    assert error.code == code
    assert str(error) == f"Currency identified by code '{code}' does not exist"


# LLM-generated content at query #73
#--------------------------

```python
def test_Currency___hash__():
    # Test basic hash functionality
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert hash(usd1) == hash(usd2)

    # Test hash with different currencies
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert hash(usd1) != hash(jpy)

    # Test hash with different attributes
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert hash(usd1) != hash(usdx)

    # Test hash with different decimals
    usd3 = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert hash(usd1) != hash(usd3)

    # Test hash with different types
    usd4 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert hash(usd1) != hash(usd4)

    # Test hash with negative decimals
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert hash(usd1) != hash(zzz)


# LLM-generated content at query #74
#--------------------------

```python
def test_Currency___delattr__():
    # Create a currency instance
    currency = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)

    # Attempt to delete an attribute and verify it raises an AttributeError
    with pytest.raises(AttributeError):
        del currency.code


# LLM-generated content at query #75
#--------------------------

```python
def test_CurrencyRegistry():
    # Test singleton behavior
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2

    # Test initial state
    assert len(registry1) == 0
    assert registry1.all == []
    assert registry1.codes == []
    assert registry1.codenames == []

    # Test context manager
    with registry1 as register:
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        register(usd)

    assert len(registry1) == 1
    assert "USD" in registry1
    assert registry1["USD"] == usd
    assert registry1.has("USD") is True
    assert registry1.get("USD") == usd
    assert registry1.get("XYZ") is None
    assert registry1.get("XYZ", default=usd) == usd
    assert registry1.all == [usd]
    assert registry1.codes == ["USD"]
    assert registry1.codenames == [("USD", "US Dollar")]

    # Test duplicate registration
    with pytest.raises(ValueError):
        with registry1 as register:
            register(usd)

    # Test registration outside context
    with pytest.raises(ProgrammingError):
        registry1._CurrencyRegistry__register(usd)


# LLM-generated content at query #76
#--------------------------

```python
def test_Currency___ge__():
    # Test equality cases
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 >= usd2
    assert usd2 >= usd1

    # Test greater than cases
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert usd >= jpy
    assert not (jpy >= usd)

    # Test less than cases
    assert jpy >= usd is False

    # Test with different attributes
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert usd >= usdx is False
    assert usdx >= usd is False

    # Test with different types
    gold = Currency.of("XAU", "Gold", 2, CurrencyType.METAL)
    assert usd >= gold is False
    assert gold >= usd is False


# LLM-generated content at query #77
#--------------------------

```python
def test_Currency___ge__():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)

    assert usd1 >= usd2
    assert usd2 >= usd1
    assert not (usd1 >= usdx)
    assert not (usdx >= usd1)
    assert not (usd1 >= jpy)
    assert not (jpy >= usd1)


# LLM-generated content at query #78
#--------------------------

```python
def test_Currency___hash__():
    # Test basic hash functionality
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert hash(usd1) == hash(usd2)

    # Test hash with different currency codes
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert hash(usd1) != hash(eur)

    # Test hash with different names
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert hash(usd1) != hash(usdx)

    # Test hash with different decimals
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert hash(usd1) != hash(jpy)

    # Test hash with different currency types
    btc = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    assert hash(usd1) != hash(btc)

    # Test hash with different quantizers
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert hash(usd1) != hash(zzz)

    # Test hash consistency
    assert hash(usd1) == hash(usd1)


# LLM-generated content at query #79
#--------------------------

```python
def test_Currency():
    # Test valid currency creation
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.code == "USD"
    assert usd.name == "US Dollars"
    assert usd.decimals == 2
    assert usd.type == CurrencyType.MONEY
    assert usd.quantize(Decimal("1.005")) == Decimal("1.00")
    assert usd.quantize(Decimal("1.015")) == Decimal("1.02")

    # Test equality and hash
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd == usd2
    assert hash(usd) == hash(usd2)

    # Test inequality
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert usd != usdx
    assert hash(usd) != hash(usdx)

    # Test JPY with 0 decimals
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy.quantize(Decimal("0.5")) == Decimal("0")
    assert jpy.quantize(Decimal("1.5")) == Decimal("2")

    # Test currency with -1 decimals
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert zzz.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert zzz.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")

    # Test invalid currency code
    with pytest.raises(ProgrammingError):
        Currency.of("123", "Invalid Code", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("usd", "Lowercase Code", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD ", "Code with Space", 2, CurrencyType.MONEY)

    # Test invalid currency name
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", " ", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", " Leading Space", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "Trailing Space ", 2, CurrencyType.MONEY)

    # Test invalid decimals
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", -2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", "2", CurrencyType.MONEY)

    # Test invalid currency type
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", 2, "MONEY")


# LLM-generated content at query #80
#--------------------------

```python
def test_Currency___repr__():
    # Create a Currency instance
    currency = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)

    # Test the __repr__ method
    repr_str = repr(currency)

    # Check that the repr string contains the expected attributes
    assert "Currency" in repr_str
    assert "code='USD'" in repr_str
    assert "name='US Dollars'" in repr_str
    assert "decimals=2" in repr_str
    assert "type=CurrencyType.MONEY" in repr_str

    # Check that the repr string can be used to recreate the object
    # (This is a more advanced test and might not be possible depending on the implementation)
    # new_currency = eval(repr_str)
    # assert new_currency == currency


# LLM-generated content at query #81
#--------------------------

```python
def test_CurrencyRegistry_get():
    # Setup
    registry = CurrencyRegistry()
    with registry as register:
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
        register(usd)
        register(eur)

    # Test existing currency with no default
    assert registry.get("USD") == usd

    # Test existing currency with default
    assert registry.get("EUR", default=usd) == eur

    # Test non-existing currency with no default
    assert registry.get("XYZ") is None

    # Test non-existing currency with default
    assert registry.get("XYZ", default=usd) == usd


# LLM-generated content at query #82
#--------------------------

```python
def test_CurrencyRegistry___exit__():
    # Create a new instance of CurrencyRegistry
    registry = CurrencyRegistry()

    # Enter the context
    register = registry.__enter__()

    # Add some currencies
    usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)

    register(usd)
    register(eur)
    register(jpy)

    # Exit the context
    registry.__exit__(None, None, None)

    # Check if the registry is sorted by code
    assert registry.codes == ["EUR", "JPY", "USD"]

    # Check if the currencies list is sorted by code
    assert [c.code for c in registry.all] == ["EUR", "JPY", "USD"]

    # Check if the codenames list is sorted by code
    assert registry.codenames == [("EUR", "Euro"), ("JPY", "Japanese Yen"), ("USD", "US Dollar")]

    # Check if the context is closed
    assert registry._CurrencyRegistry__ctx_open is False

    # Try to add a currency after exiting the context
    try:
        register(Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY))
        assert False, "Expected ProgrammingError"
    except ProgrammingError:
        pass


# LLM-generated content at query #83
#--------------------------

```python
def test_CurrencyRegistry___getitem__():
    # Test successful retrieval of a currency
    usd = CurrencyRegistry()[0]
    assert usd.code == "USD"
    assert usd.name == "US Dollar"
    assert usd.type == CurrencyType.MONEY

    # Test CurrencyLookupError for non-existent currency
    with pytest.raises(CurrencyLookupError) as exc_info:
        CurrencyRegistry()["NON-EXISTING"]
    assert str(exc_info.value) == "Currency identified by code 'NON-EXISTING' does not exist"


# LLM-generated content at query #84
#--------------------------

```python
def test_CurrencyRegistry___len__():
    registry = CurrencyRegistry()
    assert len(registry) == 0

    with registry as register:
        register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        register(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))
        register(Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY))

    assert len(registry) == 3


# LLM-generated content at query #85
#--------------------------

```python
def test_Currency___repr__():
    # Test with standard currency
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert repr(usd) == "Currency(code='USD', name='US Dollars', decimals=2, type=CurrencyType.MONEY, quantizer=Decimal('0.01'), hashcache={})".format(hash(usd))

    # Test with zero decimals
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert repr(jpy) == "Currency(code='JPY', name='Japanese Yen', decimals=0, type=CurrencyType.MONEY, quantizer=Decimal('0'), hashcache={})".format(hash(jpy))

    # Test with alternative currency type
    alt = Currency.of("ALT", "Alternative Currency", 3, CurrencyType.ALTERNATIVE)
    assert repr(alt) == "Currency(code='ALT', name='Alternative Currency', decimals=3, type=CurrencyType.ALTERNATIVE, quantizer=Decimal('0.001'), hashcache={})".format(hash(alt))

    # Test with crypto currency
    btc = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    assert repr(btc) == "Currency(code='BTC', name='Bitcoin', decimals=8, type=CurrencyType.CRYPTO, quantizer=Decimal('0.00000001'), hashcache={})".format(hash(btc))


# LLM-generated content at query #86
#--------------------------

```python
def test_Currency___lt__():
    # Test with same currency (should be False)
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 < usd2)

    # Test with different codes (should be True)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert usd < eur

    # Test with different names (should be True)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert usd < usdx

    # Test with different decimals (should be True)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("USD", "US Dollars", 0, CurrencyType.MONEY)
    assert jpy < usd

    # Test with different types (should be True)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    btc = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert usd < btc

    # Test with different quantizers (should be True)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    zzz = Currency.of("USD", "US Dollars", -1, CurrencyType.CRYPTO)
    assert zzz < usd


# LLM-generated content at query #87
#--------------------------

```python
def test_CurrencyRegistry___getitem__():
    # Setup
    registry = CurrencyRegistry()
    with registry as register:
        register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        register(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))

    # Test existing currency
    usd = registry["USD"]
    assert usd.code == "USD"
    assert usd.name == "US Dollar"
    assert usd.decimals == 2
    assert usd.type == CurrencyType.MONEY

    # Test non-existing currency
    with pytest.raises(CurrencyLookupError) as excinfo:
        _ = registry["NON-EXISTING"]
    assert str(excinfo.value) == "Currency identified by code 'NON-EXISTING' does not exist"


# LLM-generated content at query #88
#--------------------------

```python
def test_CurrencyRegistry___len__():
    registry = CurrencyRegistry()
    with registry as register:
        register(Currency.of("AED", "UAE Dirham", 2, CurrencyType.MONEY))
        register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
    assert len(registry) == 2


# LLM-generated content at query #89
#--------------------------

```python
def test_Currency___hash__():
    # Test basic hash equality for same currency
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert hash(usd1) == hash(usd2)

    # Test hash inequality for different currencies
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert hash(usd) != hash(jpy)

    # Test hash inequality for currencies with same code but different attributes
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert hash(usd1) != hash(usd2)

    # Test hash inequality for currencies with same code and decimals but different names
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert hash(usd1) != hash(usd2)

    # Test hash inequality for currencies with same code, name, and decimals but different types
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert hash(usd1) != hash(usd2)


# LLM-generated content at query #90
#--------------------------

```python
def test_Currency___setattr__():
    # Test that Currency is immutable and __setattr__ raises an AttributeError
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)

    with pytest.raises(AttributeError):
        usd.code = "EUR"

    with pytest.raises(AttributeError):
        usd.name = "Euros"

    with pytest.raises(AttributeError):
        usd.decimals = 0

    with pytest.raises(AttributeError):
        usd.type = CurrencyType.CRYPTO

    with pytest.raises(AttributeError):
        usd.quantizer = Decimal("0.0001")

    with pytest.raises(AttributeError):
        usd.hashcache = 12345


# LLM-generated content at query #91
#--------------------------

```python
def test_CurrencyRegistry___getitem__():
    # Test getting an existing currency
    assert Currencies["USD"].code == "USD"
    assert Currencies["USD"].name == "US Dollar"
    assert Currencies["USD"].type == CurrencyType.MONEY

    # Test getting a non-existing currency raises CurrencyLookupError
    with pytest.raises(CurrencyLookupError) as excinfo:
        _ = Currencies["NON-EXISTING"]
    assert str(excinfo.value) == "Currency identified by code 'NON-EXISTING' does not exist"


# LLM-generated content at query #92
#--------------------------

```python
def test_Currency():
    # Test valid currency creation
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.code == "USD"
    assert usd.name == "US Dollars"
    assert usd.decimals == 2
    assert usd.type == CurrencyType.MONEY
    assert usd.quantize(Decimal("1.005")) == Decimal("1.00")
    assert usd.quantize(Decimal("1.015")) == Decimal("1.02")

    # Test currency with zero decimals
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy.code == "JPY"
    assert jpy.name == "Japanese Yen"
    assert jpy.decimals == 0
    assert jpy.type == CurrencyType.MONEY
    assert jpy.quantize(Decimal("0.5")) == Decimal("0")
    assert jpy.quantize(Decimal("1.5")) == Decimal("2")

    # Test currency with variable precision
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert zzz.code == "ZZZ"
    assert zzz.name == "Some weird currency"
    assert zzz.decimals == -1
    assert zzz.type == CurrencyType.CRYPTO
    assert zzz.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert zzz.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")

    # Test equality and hash
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2
    assert usd1 != usdx
    assert hash(usd1) == hash(usd2)
    assert hash(usd1) != hash(usdx)

    # Test invalid currency code
    with pytest.raises(ProgrammingError):
        Currency.of("123", "Invalid Code", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("usd", "Lowercase Code", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD ", "Code with Space", 2, CurrencyType.MONEY)

    # Test invalid currency name
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", " ", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", " US Dollars", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars ", 2, CurrencyType.MONEY)

    # Test invalid decimals
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", -2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", "2", CurrencyType.MONEY)

    # Test invalid currency type
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", 2, "MONEY")


# LLM-generated content at query #93
#--------------------------

```python
def test_Currency___delattr__():
    with pytest.raises(AttributeError):
        currency = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
        del currency.code


# LLM-generated content at query #94
#--------------------------

```python
def test_Currency_quantize():
    # Test with USD (2 decimals)
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert USD.quantize(Decimal("1.005")) == Decimal("1.00")
    assert USD.quantize(Decimal("1.015")) == Decimal("1.02")
    assert USD.quantize(Decimal("1.000")) == Decimal("1.00")
    assert USD.quantize(Decimal("1.999")) == Decimal("2.00")

    # Test with JPY (0 decimals)
    JPY = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert JPY.quantize(Decimal("0.5")) == Decimal("0")
    assert JPY.quantize(Decimal("1.5")) == Decimal("2")
    assert JPY.quantize(Decimal("0.0")) == Decimal("0")
    assert JPY.quantize(Decimal("999.9")) == Decimal("1000")

    # Test with a currency with no fixed precision (e.g., -1 decimals)
    ZZZ = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert ZZZ.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert ZZZ.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")
    assert ZZZ.quantize(Decimal("1.0000000000000")) == Decimal("1.000000000000")
    assert ZZZ.quantize(Decimal("1.0000000000001")) == Decimal("1.000000000000")

    # Test edge cases
    EUR = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert EUR.quantize(Decimal("0.001")) == Decimal("0.00")
    assert EUR.quantize(Decimal("0.009")) == Decimal("0.01")
    assert EUR.quantize(Decimal("-1.005")) == Decimal("-1.00")
    assert EUR.quantize(Decimal("-1.015")) == Decimal("-1.02")


# LLM-generated content at query #95
#--------------------------

```python
def test_CurrencyRegistry_has():
    # Test with existing currency
    assert Currencies.has("USD") is True

    # Test with non-existing currency
    assert Currencies.has("NONEXISTENT") is False

    # Test case sensitivity
    assert Currencies.has("usd") is False


# LLM-generated content at query #96
#--------------------------

```python
def test_Currency___repr__():
    # Create a currency instance
    currency = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)

    # Check the __repr__ output
    repr_str = repr(currency)
    assert repr_str == "Currency(code='USD', name='US Dollars', decimals=2, type=CurrencyType.MONEY, quantizer=Decimal('0.01'), hashcache=" + str(hash(('USD', 'US Dollars', 2, CurrencyType.MONEY, Decimal('0.01')))) + ")"


# LLM-generated content at query #97
#--------------------------

```python
def test_CurrencyLookupError():
    with pytest.raises(CurrencyLookupError) as excinfo:
        raise CurrencyLookupError("XYZ")
    assert str(excinfo.value) == "Currency identified by code 'XYZ' does not exist"
    assert excinfo.value.code == "XYZ"


# LLM-generated content at query #98
#--------------------------

```python
def test_CurrencyRegistry___new__():
    # Test that CurrencyRegistry is a singleton
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2

    # Test that the singleton instance is created only once
    assert CurrencyRegistry._CurrencyRegistry__instance is not None
    assert CurrencyRegistry._CurrencyRegistry__instance is registry1
    assert CurrencyRegistry._CurrencyRegistry__instance is registry2

    # Test that the singleton instance is the same across multiple calls
    registry3 = CurrencyRegistry()
    assert registry3 is registry1
    assert registry3 is registry2


# LLM-generated content at query #99
#--------------------------

```python
def test_CurrencyRegistry___enter__():
    registry = CurrencyRegistry()
    with registry as register:
        assert callable(register)
        assert register.__name__ == "_CurrencyRegistry__register"
        assert registry._CurrencyRegistry__ctx_open is True
    assert registry._CurrencyRegistry__ctx_open is False


# LLM-generated content at query #100
#--------------------------

```python
def test_CurrencyRegistry___getitem__():
    # Setup
    registry = CurrencyRegistry()
    with registry as register:
        register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        register(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))

    # Test existing currency
    assert registry["USD"].code == "USD"
    assert registry["EUR"].code == "EUR"

    # Test non-existing currency
    with pytest.raises(CurrencyLookupError):
        registry["NON-EXISTING"]


# LLM-generated content at query #101
#--------------------------

```python
def test_CurrencyRegistry_has():
    # Test with existing currency
    assert Currencies.has("USD") is True

    # Test with non-existing currency
    assert Currencies.has("NONEXISTING") is False

    # Test with empty string
    assert Currencies.has("") is False

    # Test with None (should raise TypeError)
    with pytest.raises(TypeError):
        Currencies.has(None)


# LLM-generated content at query #102
#--------------------------

```python
def test_Currency___repr__():
    # Test with USD
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert repr(usd) == "Currency(code='USD', name='US Dollars', decimals=2, type=<CurrencyType.MONEY: 'Money'>, quantizer=Decimal('0.01'), hashcache=" + str(hash(usd)) + ")"

    # Test with JPY
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert repr(jpy) == "Currency(code='JPY', name='Japanese Yen', decimals=0, type=<CurrencyType.MONEY: 'Money'>, quantizer=Decimal('0'), hashcache=" + str(hash(jpy)) + ")"

    # Test with a weird currency
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert repr(zzz) == "Currency(code='ZZZ', name='Some weird currency', decimals=-1, type=<CurrencyType.CRYPTO: 'Crypto Currency'>, quantizer=Decimal('1E-28'), hashcache=" + str(hash(zzz)) + ")"


# LLM-generated content at query #103
#--------------------------

```python
def test_CurrencyRegistry___contains__():
    # Create a new registry instance
    registry = CurrencyRegistry()

    # Test with empty registry
    assert "USD" not in registry
    assert "EUR" not in registry

    # Populate the registry with some currencies
    with registry as register:
        register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        register(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))

    # Test with populated registry
    assert "USD" in registry
    assert "EUR" in registry
    assert "JPY" not in registry


# LLM-generated content at query #104
#--------------------------

```python
def test_Currency___delattr__():
    currency = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    with pytest.raises(AttributeError):
        del currency.code


# LLM-generated content at query #105
#--------------------------

```python
def test_CurrencyLookupError():
    code = "XYZ"
    error = CurrencyLookupError(code)
    assert error.code == code
    assert str(error) == f"Currency identified by code '{code}' does not exist"


# LLM-generated content at query #106
#--------------------------

```python
def test_Currency___delattr__():
    # Create a Currency instance
    currency = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)

    # Attempt to delete an attribute and verify it raises an AttributeError
    with pytest.raises(AttributeError):
        del currency.code


# LLM-generated content at query #107
#--------------------------

```python
def test_CurrencyRegistry___enter__():
    registry = CurrencyRegistry()
    with registry as register:
        assert callable(register)
        assert register.__name__ == "_CurrencyRegistry__register"
    assert not registry._CurrencyRegistry__ctx_open


# LLM-generated content at query #108
#--------------------------

```python
def test_CurrencyRegistry():
    # Test singleton behavior
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2

    # Test initial state
    assert len(registry1) == 0
    assert registry1.all == []
    assert registry1.codes == []
    assert registry1.codenames == []

    # Test context manager behavior
    with registry1 as register:
        # Test adding currencies
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        register(usd)
        assert len(registry1) == 1
        assert "USD" in registry1
        assert registry1["USD"] == usd

        # Test duplicate currency
        with pytest.raises(ValueError, match="Currency USD is already registered."):
            register(usd)

    # Test post-context state
    assert len(registry1) == 1
    assert registry1.all == [usd]
    assert registry1.codes == ["USD"]
    assert registry1.codenames == [("USD", "US Dollar")]

    # Test adding outside context
    with pytest.raises(ProgrammingError, match="Can not create currencies outside registry context."):
        registry1._CurrencyRegistry__register(usd)

    # Test lookup methods
    assert registry1.has("USD") is True
    assert registry1.has("XYZ") is False
    assert registry1.get("USD") == usd
    assert registry1.get("XYZ") is None
    assert registry1.get("XYZ", default=usd) == usd

    # Test lookup error
    with pytest.raises(CurrencyLookupError, match="Currency identified by code 'XYZ' does not exist"):
        registry1["XYZ"]


# LLM-generated content at query #109
#--------------------------

```python
def test_CurrencyRegistry___contains__():
    # Setup
    registry = CurrencyRegistry()
    with registry:
        registry(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        registry(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))

    # Test existing currency
    assert "USD" in registry
    assert "EUR" in registry

    # Test non-existing currency
    assert "XYZ" not in registry
    assert "NONEXISTENT" not in registry


# LLM-generated content at query #110
#--------------------------

```python
def test_CurrencyRegistry___enter__():
    registry = CurrencyRegistry()
    with registry as register:
        assert callable(register)
        assert register.__name__ == "_CurrencyRegistry__register"
    assert not registry._CurrencyRegistry__ctx_open


# LLM-generated content at query #111
#--------------------------

```python
def test_Currency___le__():
    # Test with same currency
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 <= usd2
    assert usd2 <= usd1

    # Test with different currencies
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert usd <= jpy or jpy <= usd  # One should be less than or equal to the other

    # Test with different attributes
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert usd <= usdx or usdx <= usd  # One should be less than or equal to the other

    # Test with different types
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    btc = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    assert usd <= btc or btc <= usd  # One should be less than or equal to the other


# LLM-generated content at query #112
#--------------------------

```python
def test_CurrencyRegistry___new__():
    # Test that CurrencyRegistry is a singleton
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2

    # Test that the singleton instance is created only once
    assert CurrencyRegistry.__instance is not None
    assert CurrencyRegistry.__instance is registry1
    assert CurrencyRegistry.__instance is registry2


# LLM-generated content at query #113
#--------------------------

```python
def test_Currency___lt__():
    # Test with equal currencies
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 < usd2)
    assert not (usd2 < usd1)

    # Test with different currencies (same type, different code)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    assert (eur < usd) or (usd < eur)  # One should be less than the other
    assert not ((eur < usd) and (usd < eur))  # They can't both be less than each other

    # Test with different types
    gold = Currency.of("XAU", "Gold", 2, CurrencyType.METAL)
    btc = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    assert (gold < btc) or (btc < gold)  # One should be less than the other
    assert not ((gold < btc) and (btc < gold))  # They can't both be less than each other

    # Test with different decimals
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert (jpy < usd) or (usd < jpy)  # One should be less than the other
    assert not ((jpy < usd) and (usd < jpy))  # They can't both be less than each other

    # Test with different names (same code, type, decimals)
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars (Alternative)", 2, CurrencyType.MONEY)
    assert (usd1 < usd2) or (usd2 < usd1)  # One should be less than the other
    assert not ((usd1 < usd2) and (usd2 < usd1))  # They can't both be less than each other


# LLM-generated content at query #114
#--------------------------

```python
def test_CurrencyRegistry___getitem__():
    # Setup
    registry = CurrencyRegistry()
    with registry as register:
        register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        register(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))

    # Test existing currency
    usd = registry["USD"]
    assert usd.code == "USD"
    assert usd.name == "US Dollar"
    assert usd.decimals == 2
    assert usd.type == CurrencyType.MONEY

    # Test non-existing currency
    with pytest.raises(CurrencyLookupError) as excinfo:
        _ = registry["NON-EXISTING"]
    assert str(excinfo.value) == "Currency identified by code 'NON-EXISTING' does not exist"


# LLM-generated content at query #115
#--------------------------

```python
def test_Currency___gt__():
    # Test greater than with different codes
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert (eur > usd) == ("EUR" > "USD")

    # Test greater than with same code but different names
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars 2", 2, CurrencyType.MONEY)
    assert (usd2 > usd1) == (("USD", "US Dollars 2", 2, CurrencyType.MONEY, usd2.quantizer) >
                             ("USD", "US Dollars", 2, CurrencyType.MONEY, usd1.quantizer))

    # Test greater than with same code and name but different decimals
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert (usd2 > usd1) == (("USD", "US Dollars", 3, CurrencyType.MONEY, usd2.quantizer) >
                             ("USD", "US Dollars", 2, CurrencyType.MONEY, usd1.quantizer))

    # Test greater than with same code, name, and decimals but different types
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert (usd2 > usd1) == (("USD", "US Dollars", 2, CurrencyType.CRYPTO, usd2.quantizer) >
                             ("USD", "US Dollars", 2, CurrencyType.MONEY, usd1.quantizer))

    # Test greater than with same code, name, decimals, and type but different quantizers
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", -1, CurrencyType.MONEY)
    assert (usd2 > usd1) == (("USD", "US Dollars", -1, CurrencyType.MONEY, usd2.quantizer) >
                             ("USD", "US Dollars", 2, CurrencyType.MONEY, usd1.quantizer))

    # Test greater than with same currency
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd > usd)


# LLM-generated content at query #116
#--------------------------

```python
def test_CurrencyRegistry___exit__():
    # Setup
    registry = CurrencyRegistry()
    test_currency = Currency.of("TEST", "Test Currency", 2, CurrencyType.MONEY)

    # Test context manager behavior
    with registry as register:
        register(test_currency)

    # Verify that the currency was added and sorted
    assert "TEST" in registry
    assert registry["TEST"] == test_currency
    assert registry.codes == ["TEST"]
    assert registry.codenames == [("TEST", "Test Currency")]

    # Verify that the context is closed
    try:
        register(test_currency)
        assert False, "Should have raised ProgrammingError"
    except ProgrammingError:
        pass

    # Verify that adding a duplicate currency raises ValueError
    with registry as register:
        try:
            register(test_currency)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert str(e) == "Currency TEST is already registered."


# LLM-generated content at query #117
#--------------------------

```python
def test_Currency___setattr__():
    # Test that __setattr__ raises an AttributeError for frozen dataclass
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)

    with pytest.raises(AttributeError):
        usd.code = "EUR"

    with pytest.raises(AttributeError):
        usd.name = "Euros"

    with pytest.raises(AttributeError):
        usd.decimals = 0

    with pytest.raises(AttributeError):
        usd.type = CurrencyType.CRYPTO

    with pytest.raises(AttributeError):
        usd.quantizer = Decimal("0.0001")

    with pytest.raises(AttributeError):
        usd.hashcache = 12345


# LLM-generated content at query #118
#--------------------------

```python
def test_Currency___eq__():
    # Test equality with same currency
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2

    # Test inequality with different currency code
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert not (usd == eur)

    # Test inequality with different currency name
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 == usd2)

    # Test inequality with different decimals
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 0, CurrencyType.MONEY)
    assert not (usd1 == usd2)

    # Test inequality with different currency type
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert not (usd1 == usd2)

    # Test inequality with non-Currency object
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd == "USD")

    # Test inequality with None
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd == None)


# LLM-generated content at query #119
#--------------------------

```python
def test_CurrencyRegistry___len__():
    registry = CurrencyRegistry()
    with registry as register:
        register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        register(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))
        register(Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY))

    assert len(registry) == 3


# LLM-generated content at query #120
#--------------------------

```python
def test_CurrencyRegistry():
    # Test singleton behavior
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2

    # Test initial state
    assert len(registry1) == 0
    assert registry1.all == []
    assert registry1.codes == []
    assert registry1.codenames == []

    # Test context manager
    with registry1 as register:
        # Test adding currencies
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        register(usd)
        assert len(registry1) == 1
        assert "USD" in registry1
        assert registry1["USD"] == usd

        # Test adding duplicate currency raises error
        with pytest.raises(ValueError, match="Currency USD is already registered."):
            register(usd)

    # Test final state after context
    assert len(registry1) == 1
    assert registry1.all == [usd]
    assert registry1.codes == ["USD"]
    assert registry1.codenames == [("USD", "US Dollar")]

    # Test adding outside context raises error
    with pytest.raises(ProgrammingError, match="Can not create currencies outside registry context."):
        registry1._CurrencyRegistry__register(usd)

    # Test has method
    assert registry1.has("USD") is True
    assert registry1.has("XYZ") is False

    # Test get method
    assert registry1.get("USD") == usd
    assert registry1.get("XYZ") is None
    assert registry1.get("XYZ", default=usd) == usd

    # Test lookup error
    with pytest.raises(CurrencyLookupError, match="Currency identified by code 'XYZ' does not exist"):
        registry1["XYZ"]


# LLM-generated content at query #121
#--------------------------

```python
def test_CurrencyRegistry___contains__():
    registry = CurrencyRegistry()
    with registry as register:
        register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        register(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))

    assert "USD" in registry
    assert "EUR" in registry
    assert "XYZ" not in registry


# LLM-generated content at query #122
#--------------------------

```python
def test_CurrencyLookupError():
    code = "XYZ"
    error = CurrencyLookupError(code)
    assert error.code == code
    assert str(error) == f"Currency identified by code '{code}' does not exist"


# LLM-generated content at query #123
#--------------------------

```python
def test_CurrencyRegistry___new__():
    # Test that CurrencyRegistry is a singleton
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2

    # Test that the singleton instance is created only once
    assert CurrencyRegistry._CurrencyRegistry__instance is not None
    assert CurrencyRegistry._CurrencyRegistry__instance is registry1
    assert CurrencyRegistry._CurrencyRegistry__instance is registry2


# LLM-generated content at query #124
#--------------------------

```python
def test_Currency___eq__():
    # Test equality with same currency
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2

    # Test inequality with different currency code
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert usd != eur

    # Test inequality with different currency name
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars Different", 2, CurrencyType.MONEY)
    assert usd1 != usd2

    # Test inequality with different decimals
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert usd1 != usd2

    # Test inequality with different currency type
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert usd1 != usd2

    # Test inequality with non-Currency object
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd != "USD"
    assert usd != 123
    assert usd != None


# LLM-generated content at query #125
#--------------------------

```python
def test_Currency___le__():
    # Test equality cases
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 <= usd2
    assert usd2 <= usd1

    # Test less than cases
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert usd <= jpy or jpy <= usd  # One should be less than or equal to the other

    # Test with different attributes
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert usd1 <= usd2 or usd2 <= usd1

    # Test with different types
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    btc = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    assert usd <= btc or btc <= usd

    # Test with non-Currency object
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd <= "not a currency")


# LLM-generated content at query #126
#--------------------------

```python
def test_Currency___delattr__():
    currency = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    with pytest.raises(AttributeError):
        del currency.code
    with pytest.raises(AttributeError):
        del currency.name
    with pytest.raises(AttributeError):
        del currency.decimals
    with pytest.raises(AttributeError):
        del currency.type
    with pytest.raises(AttributeError):
        del currency.quantizer
    with pytest.raises(AttributeError):
        del currency.hashcache


# LLM-generated content at query #127
#--------------------------

```python
def test_Currency_quantize():
    # Test with USD (2 decimals)
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert USD.quantize(Decimal("1.005")) == Decimal("1.00")
    assert USD.quantize(Decimal("1.015")) == Decimal("1.02")

    # Test with JPY (0 decimals)
    JPY = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert JPY.quantize(Decimal("0.5")) == Decimal("0")
    assert JPY.quantize(Decimal("1.5")) == Decimal("2")

    # Test with a currency with no fixed precision (-1 decimals)
    ZZZ = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert ZZZ.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert ZZZ.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")


# LLM-generated content at query #128
#--------------------------

```python
def test_CurrencyRegistry___getitem__():
    # Setup
    registry = CurrencyRegistry()
    with registry as register:
        register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        register(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))

    # Test existing currency
    usd = registry["USD"]
    assert usd.code == "USD"
    assert usd.name == "US Dollar"
    assert usd.type == CurrencyType.MONEY

    # Test non-existing currency
    with pytest.raises(CurrencyLookupError) as excinfo:
        _ = registry["NON-EXISTING"]
    assert str(excinfo.value) == "Currency identified by code 'NON-EXISTING' does not exist"


# LLM-generated content at query #129
#--------------------------

```python
def test_CurrencyRegistry___enter__():
    # Setup
    registry = CurrencyRegistry()

    # Test that __enter__ returns the __register method
    with registry as register:
        assert callable(register)
        assert register.__name__ == "_CurrencyRegistry__register"

    # Test that context is properly closed after exit
    assert not registry._CurrencyRegistry__ctx_open

    # Test that currencies can be added within the context
    with registry as register:
        test_currency = Currency.of("TEST", "Test Currency", 2, CurrencyType.MONEY)
        register(test_currency)
        assert "TEST" in registry

    # Test that context is properly closed after exit
    assert not registry._CurrencyRegistry__ctx_open

    # Test that adding outside context raises error
    with pytest.raises(ProgrammingError):
        registry._CurrencyRegistry__register(test_currency)

    # Test that duplicate currency raises error
    with registry as register:
        with pytest.raises(ValueError):
            register(test_currency)


# LLM-generated content at query #130
#--------------------------

```python
def test_Currency___gt__():
    # Test with different codes
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert usd > eur
    assert not (eur > usd)

    # Test with same code but different names
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 > usd2)

    # Test with different decimals
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert usd > jpy
    assert not (jpy > usd)

    # Test with different types
    btc = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    assert usd > btc
    assert not (btc > usd)

    # Test with different quantizers
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert usd > zzz
    assert not (zzz > usd)


# LLM-generated content at query #131
#--------------------------

```python
def test_CurrencyRegistry_get():
    # Setup
    registry = CurrencyRegistry()
    with registry as register:
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
        register(usd)
        register(eur)

    # Test existing currency with no default
    assert registry.get("USD") == usd
    assert registry.get("EUR") == eur

    # Test non-existing currency with no default
    assert registry.get("XYZ") is None

    # Test non-existing currency with default
    assert registry.get("XYZ", default=usd) == usd
    assert registry.get("ABC", default=eur) == eur

    # Test existing currency with default (should return the currency, not default)
    assert registry.get("USD", default=eur) == usd
    assert registry.get("EUR", default=usd) == eur


# LLM-generated content at query #132
#--------------------------

```python
def test_Currency_quantize():
    # Test with USD (2 decimals)
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert USD.quantize(Decimal("1.005")) == Decimal("1.00")
    assert USD.quantize(Decimal("1.015")) == Decimal("1.02")
    assert USD.quantize(Decimal("1.000")) == Decimal("1.00")
    assert USD.quantize(Decimal("1.999")) == Decimal("2.00")

    # Test with JPY (0 decimals)
    JPY = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert JPY.quantize(Decimal("0.5")) == Decimal("0")
    assert JPY.quantize(Decimal("1.5")) == Decimal("2")
    assert JPY.quantize(Decimal("1.0")) == Decimal("1")
    assert JPY.quantize(Decimal("9.9")) == Decimal("10")

    # Test with a currency that has no fixed precision (decimals = -1)
    ZZZ = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert ZZZ.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert ZZZ.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")
    assert ZZZ.quantize(Decimal("1.0000000000000")) == Decimal("1.000000000000")
    assert ZZZ.quantize(Decimal("1.0000000000001")) == Decimal("1.000000000000")

    # Test with negative values
    assert USD.quantize(Decimal("-1.005")) == Decimal("-1.00")
    assert USD.quantize(Decimal("-1.015")) == Decimal("-1.02")
    assert JPY.quantize(Decimal("-0.5")) == Decimal("0")
    assert JPY.quantize(Decimal("-1.5")) == Decimal("-2")
    assert ZZZ.quantize(Decimal("-1.0000000000005")) == Decimal("-1.000000000000")
    assert ZZZ.quantize(Decimal("-1.0000000000015")) == Decimal("-1.000000000002")


# LLM-generated content at query #133
#--------------------------

```python
def test_CurrencyRegistry_get():
    # Setup
    registry = CurrencyRegistry()
    with registry as register:
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
        register(usd)
        register(eur)

    # Test existing currency with no default
    assert registry.get("USD") == usd

    # Test existing currency with default (should ignore default)
    assert registry.get("EUR", default=usd) == eur

    # Test non-existing currency with no default
    assert registry.get("XYZ") is None

    # Test non-existing currency with default
    assert registry.get("ABC", default=usd) == usd


# LLM-generated content at query #134
#--------------------------

```python
def test_CurrencyRegistry___len__():
    registry = CurrencyRegistry()
    assert len(registry) == 0

    with registry as register:
        register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        register(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))

    assert len(registry) == 2


# LLM-generated content at query #135
#--------------------------

```python
def test_CurrencyRegistry___len__():
    registry = CurrencyRegistry()
    assert len(registry) == 0

    with registry:
        registry(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        registry(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))

    assert len(registry) == 2


# LLM-generated content at query #136
#--------------------------

```python
def test_CurrencyRegistry___len__():
    # Create a new CurrencyRegistry instance
    registry = CurrencyRegistry()

    # Initially, the registry should be empty
    assert len(registry) == 0

    # Add some currencies to the registry
    with registry as register:
        register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        register(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))
        register(Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY))

    # Now, the registry should have 3 currencies
    assert len(registry) == 3


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_Currency___eq__():
    # Test equality with same currency
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2

    # Test inequality with different currency code
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    assert not (usd == eur)

    # Test inequality with different currency name
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 == usd2)

    # Test inequality with different currency decimals
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 0, CurrencyType.MONEY)
    assert not (usd1 == usd2)

    # Test inequality with different currency type
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert not (usd1 == usd2)

    # Test inequality with non-Currency object
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd == "USD")


# LLM-generated content at query #2
#--------------------------

```python
def test_Currency___gt__():
    # Test with different codes
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert (usd > eur) == ("USD" > "EUR")

    # Test with same code but different names
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars 2", 2, CurrencyType.MONEY)
    assert (usd1 > usd2) == (("USD", "US Dollars", 2, CurrencyType.MONEY, usd1.quantizer) >
                            ("USD", "US Dollars 2", 2, CurrencyType.MONEY, usd2.quantizer))

    # Test with same code and name but different decimals
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert (usd1 > usd2) == (("USD", "US Dollars", 2, CurrencyType.MONEY, usd1.quantizer) >
                            ("USD", "US Dollars", 3, CurrencyType.MONEY, usd2.quantizer))

    # Test with same code, name, and decimals but different types
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert (usd1 > usd2) == (("USD", "US Dollars", 2, CurrencyType.MONEY, usd1.quantizer) >
                            ("USD", "US Dollars", 2, CurrencyType.CRYPTO, usd2.quantizer))

    # Test with same code, name, decimals, and type but different quantizers
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", -1, CurrencyType.MONEY)
    assert (usd1 > usd2) == (("USD", "US Dollars", 2, CurrencyType.MONEY, usd1.quantizer) >
                            ("USD", "US Dollars", -1, CurrencyType.MONEY, usd2.quantizer))

    # Test with equal currencies
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 > usd2)


# LLM-generated content at query #3
#--------------------------

```python
def test_CurrencyRegistry___len__():
    registry = CurrencyRegistry()
    assert len(registry) == 0

    with registry as register:
        register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        register(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))

    assert len(registry) == 2


# LLM-generated content at query #4
#--------------------------

```python
def test_Currency():
    # Test valid currency creation
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.code == "USD"
    assert usd.name == "US Dollars"
    assert usd.decimals == 2
    assert usd.type == CurrencyType.MONEY
    assert usd.quantize(Decimal("1.005")) == Decimal("1.00")
    assert usd.quantize(Decimal("1.015")) == Decimal("1.02")

    # Test currency with zero decimals
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy.code == "JPY"
    assert jpy.name == "Japanese Yen"
    assert jpy.decimals == 0
    assert jpy.type == CurrencyType.MONEY
    assert jpy.quantize(Decimal("0.5")) == Decimal("0")
    assert jpy.quantize(Decimal("1.5")) == Decimal("2")

    # Test currency with negative decimals
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert zzz.code == "ZZZ"
    assert zzz.name == "Some weird currency"
    assert zzz.decimals == -1
    assert zzz.type == CurrencyType.CRYPTO
    assert zzz.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert zzz.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")

    # Test equality and hash
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2
    assert not (usd1 == usdx)
    assert hash(usd1) == hash(usd2)
    assert not (hash(usd1) == hash(usdx))

    # Test invalid currency code
    with pytest.raises(ProgrammingError):
        Currency.of("123", "Invalid Code", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("usd", "Lowercase Code", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD ", "Code with Space", 2, CurrencyType.MONEY)

    # Test invalid currency name
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", " ", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", " US Dollars", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars ", 2, CurrencyType.MONEY)

    # Test invalid decimals
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", -2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", "2", CurrencyType.MONEY)

    # Test invalid currency type
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", 2, "MONEY")


# LLM-generated content at query #5
#--------------------------

```python
def test_Currency___repr__():
    # Test with USD
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert repr(usd) == "Currency(code='USD', name='US Dollars', decimals=2, type=<CurrencyType.MONEY: 'Money'>, quantizer=Decimal('0.01'), hashcache={})".format(hash(usd))

    # Test with JPY
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert repr(jpy) == "Currency(code='JPY', name='Japanese Yen', decimals=0, type=<CurrencyType.MONEY: 'Money'>, quantizer=Decimal('0'), hashcache={})".format(hash(jpy))

    # Test with a weird currency
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert repr(zzz) == "Currency(code='ZZZ', name='Some weird currency', decimals=-1, type=<CurrencyType.CRYPTO: 'Crypto Currency'>, quantizer=Decimal('0.000000000000000000000000000001'), hashcache={})".format(hash(zzz))


# LLM-generated content at query #6
#--------------------------

```python
def test_CurrencyRegistry___len__():
    registry = CurrencyRegistry()
    with registry as register:
        register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        register(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))
        register(Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY))

    assert len(registry) == 3


# LLM-generated content at query #7
#--------------------------

```python
def test_CurrencyRegistry___len__():
    registry = CurrencyRegistry()
    assert len(registry) == 0

    with registry as register:
        register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        register(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))
        register(Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY))

    assert len(registry) == 3


# LLM-generated content at query #8
#--------------------------

```python
def test_CurrencyRegistry_has():
    # Test with existing currency
    assert Currencies.has("USD") is True

    # Test with non-existing currency
    assert Currencies.has("NONEXISTING") is False

    # Test with empty string
    assert Currencies.has("") is False

    # Test with None (should not raise exception)
    assert Currencies.has(None) is False  # type: ignore


# LLM-generated content at query #9
#--------------------------

```python
def test_CurrencyRegistry___exit__():
    registry = CurrencyRegistry()
    with registry as register:
        register(Currency.of("AED", "UAE Dirham", 2, CurrencyType.MONEY))
        register(Currency.of("BHD", "Bahraini Dinar", 3, CurrencyType.MONEY))
        register(Currency.of("CUC", "Cuban Convertible Peso", 2, CurrencyType.MONEY))

    assert len(registry) == 3
    assert registry.codes == ["AED", "BHD", "CUC"]
    assert registry.codenames == [
        ("AED", "UAE Dirham"),
        ("BHD", "Bahraini Dinar"),
        ("CUC", "Cuban Convertible Peso")
    ]
    assert registry["AED"].code == "AED"
    assert registry["BHD"].code == "BHD"
    assert registry["CUC"].code == "CUC"


# LLM-generated content at query #10
#--------------------------

```python
def test_Currency___hash__():
    # Test basic hash functionality
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert hash(usd1) == hash(usd2)

    # Test hash with different attributes
    usd3 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert hash(usd1) != hash(usd3)

    # Test hash with different code
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert hash(usd1) != hash(eur)

    # Test hash with different name
    usd4 = Currency.of("USD", "US Dollars Different", 2, CurrencyType.MONEY)
    assert hash(usd1) != hash(usd4)

    # Test hash with different decimals
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert hash(usd1) != hash(jpy)

    # Test hash with negative decimals
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert hash(usd1) != hash(zzz)

    # Test hash consistency
    assert hash(usd1) == hash(usd1)


# LLM-generated content at query #11
#--------------------------

```python
def test_Currency___setattr__():
    # Test that Currency is immutable and __setattr__ raises an AttributeError
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)

    with pytest.raises(AttributeError):
        usd.code = "EUR"

    with pytest.raises(AttributeError):
        usd.name = "Euros"

    with pytest.raises(AttributeError):
        usd.decimals = 0

    with pytest.raises(AttributeError):
        usd.type = CurrencyType.CRYPTO

    with pytest.raises(AttributeError):
        usd.quantizer = Decimal("0.0001")

    with pytest.raises(AttributeError):
        usd.hashcache = 12345


# LLM-generated content at query #12
#--------------------------

```python
def test_Currency___repr__():
    currency = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert repr(currency) == "Currency(code='USD', name='US Dollars', decimals=2, type=CurrencyType.MONEY, quantizer=Decimal('0.01'), hashcache=" + str(hash(("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal('0.01')))) + ")"


# LLM-generated content at query #13
#--------------------------

```python
def test_Currency_quantize():
    # Test with USD (2 decimals)
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert USD.quantize(Decimal("1.005")) == Decimal("1.00")
    assert USD.quantize(Decimal("1.015")) == Decimal("1.02")
    assert USD.quantize(Decimal("1.000")) == Decimal("1.00")
    assert USD.quantize(Decimal("1.999")) == Decimal("2.00")

    # Test with JPY (0 decimals)
    JPY = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert JPY.quantize(Decimal("0.5")) == Decimal("0")
    assert JPY.quantize(Decimal("1.5")) == Decimal("2")
    assert JPY.quantize(Decimal("1.0")) == Decimal("1")
    assert JPY.quantize(Decimal("9.9")) == Decimal("10")

    # Test with a currency with no fixed precision (-1 decimals)
    ZZZ = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert ZZZ.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert ZZZ.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")
    assert ZZZ.quantize(Decimal("1.0000000000000")) == Decimal("1.000000000000")
    assert ZZZ.quantize(Decimal("1.0000000000001")) == Decimal("1.000000000000")

    # Test with negative values
    assert USD.quantize(Decimal("-1.005")) == Decimal("-1.00")
    assert JPY.quantize(Decimal("-1.5")) == Decimal("-2")
    assert ZZZ.quantize(Decimal("-1.0000000000005")) == Decimal("-1.000000000000")

    # Test with very large values
    assert USD.quantize(Decimal("999999999999.995")) == Decimal("999999999999.99")
    assert JPY.quantize(Decimal("999999999999.5")) == Decimal("999999999999")


# LLM-generated content at query #14
#--------------------------

```python
def test_Currency___lt__():
    # Test with different codes
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert (usd < eur) == ("USD" < "EUR")

    # Test with same code but different names
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars 2", 2, CurrencyType.MONEY)
    assert (usd1 < usd2) == (("USD", "US Dollars", 2, CurrencyType.MONEY, usd1.quantizer) <
                             ("USD", "US Dollars 2", 2, CurrencyType.MONEY, usd2.quantizer))

    # Test with same code and name but different decimals
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert (usd1 < usd2) == (("USD", "US Dollars", 2, CurrencyType.MONEY, usd1.quantizer) <
                             ("USD", "US Dollars", 3, CurrencyType.MONEY, usd2.quantizer))

    # Test with same code, name, and decimals but different types
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert (usd1 < usd2) == (("USD", "US Dollars", 2, CurrencyType.MONEY, usd1.quantizer) <
                             ("USD", "US Dollars", 2, CurrencyType.CRYPTO, usd2.quantizer))

    # Test with same code, name, decimals, and type but different quantizers
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", -1, CurrencyType.MONEY)
    assert (usd1 < usd2) == (("USD", "US Dollars", 2, CurrencyType.MONEY, usd1.quantizer) <
                             ("USD", "US Dollars", -1, CurrencyType.MONEY, usd2.quantizer))

    # Test with same object
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd < usd)


# LLM-generated content at query #15
#--------------------------

```python
def test_CurrencyRegistry___enter__():
    registry = CurrencyRegistry()
    with registry as register:
        assert callable(register)
        assert register.__name__ == "_CurrencyRegistry__register"


# LLM-generated content at query #16
#--------------------------

```python
def test_CurrencyRegistry___exit__():
    # Setup
    registry = CurrencyRegistry()
    test_currency1 = Currency.of("TST", "Test Currency 1", 2, CurrencyType.MONEY)
    test_currency2 = Currency.of("TST2", "Test Currency 2", 0, CurrencyType.CRYPTO)

    # Test context manager behavior
    with registry as register:
        register(test_currency1)
        register(test_currency2)

    # Verify post-exit state
    assert registry.has("TST")
    assert registry.has("TST2")
    assert registry["TST"] == test_currency1
    assert registry["TST2"] == test_currency2
    assert len(registry) == 2
    assert registry.codes == ["TST", "TST2"]
    assert registry.codenames == [("TST", "Test Currency 1"), ("TST2", "Test Currency 2")]

    # Test that registry is properly sorted
    test_currency3 = Currency.of("AAA", "Test Currency 3", 1, CurrencyType.METAL)
    with registry as register:
        register(test_currency3)

    assert registry.codes == ["AAA", "TST", "TST2"]

    # Test that context is properly closed
    try:
        registry._CurrencyRegistry__register(test_currency1)
        assert False, "Should have raised ProgrammingError"
    except ProgrammingError:
        pass


# LLM-generated content at query #17
#--------------------------

```python
def test_CurrencyRegistry():
    # Test singleton behavior
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2

    # Test initial state
    assert len(registry1) == 0
    assert registry1.all == []
    assert registry1.codes == []
    assert registry1.codenames == []

    # Test context manager
    with registry1 as register:
        # Test adding currencies
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
        jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)

        register(usd)
        register(eur)
        register(jpy)

        # Test that currencies are added in the context
        assert len(registry1) == 3
        assert registry1.all == [eur, jpy, usd]  # Sorted by code
        assert registry1.codes == ["EUR", "JPY", "USD"]
        assert registry1.codenames == [("EUR", "Euro"), ("JPY", "Japanese Yen"), ("USD", "US Dollar")]

    # Test that currencies are still there after context exit
    assert len(registry1) == 3
    assert registry1.all == [eur, jpy, usd]
    assert registry1.codes == ["EUR", "JPY", "USD"]
    assert registry1.codenames == [("EUR", "Euro"), ("JPY", "Japanese Yen"), ("USD", "US Dollar")]

    # Test duplicate currency registration
    with pytest.raises(ValueError):
        with registry1 as register:
            register(usd)

    # Test adding currency outside context
    with pytest.raises(ProgrammingError):
        registry1._CurrencyRegistry__register(usd)

    # Test currency lookup
    assert registry1.has("USD")
    assert not registry1.has("XYZ")
    assert registry1["USD"] == usd
    assert registry1.get("USD") == usd
    assert registry1.get("XYZ") is None
    assert registry1.get("XYZ", default=eur) == eur

    # Test CurrencyLookupError
    with pytest.raises(CurrencyLookupError):
        registry1["XYZ"]

    # Test 'in' operator
    assert "USD" in registry1
    assert "XYZ" not in registry1


# LLM-generated content at query #18
#--------------------------

```python
def test_CurrencyRegistry___getitem__():
    # Test existing currency
    assert Currencies["USD"].code == "USD"
    assert Currencies["USD"].name == "US Dollar"

    # Test non-existing currency
    with pytest.raises(CurrencyLookupError):
        Currencies["NON-EXISTING"]


# LLM-generated content at query #19
#--------------------------

```python
def test_Currency():
    # Test valid currency creation
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.code == "USD"
    assert usd.name == "US Dollars"
    assert usd.decimals == 2
    assert usd.type == CurrencyType.MONEY
    assert usd.quantizer == make_quantizer(2)

    # Test currency with zero decimals
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy.code == "JPY"
    assert jpy.name == "Japanese Yen"
    assert jpy.decimals == 0
    assert jpy.type == CurrencyType.MONEY
    assert jpy.quantizer == ZERO

    # Test currency with negative decimals
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert zzz.code == "ZZZ"
    assert zzz.name == "Some weird currency"
    assert zzz.decimals == -1
    assert zzz.type == CurrencyType.CRYPTO
    assert zzz.quantizer == MaxPrecisionQuantizer

    # Test currency equality and hash
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2
    assert hash(usd1) == hash(usd2)

    # Test currency inequality
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert usd1 != usdx
    assert hash(usd1) != hash(usdx)

    # Test quantize method
    assert usd.quantize(Decimal("1.005")) == Decimal("1.00")
    assert usd.quantize(Decimal("1.015")) == Decimal("1.02")
    assert jpy.quantize(Decimal("0.5")) == Decimal("0")
    assert jpy.quantize(Decimal("1.5")) == Decimal("2")
    assert zzz.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert zzz.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")

    # Test invalid currency code
    with pytest.raises(ProgrammingError):
        Currency.of("123", "Invalid Code", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("usd", "Lowercase Code", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD ", "Code with space", 2, CurrencyType.MONEY)

    # Test invalid currency name
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", " ", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", " US Dollars", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars ", 2, CurrencyType.MONEY)

    # Test invalid decimals
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", -2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", "2", CurrencyType.MONEY)

    # Test invalid currency type
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", 2, "MONEY")


# LLM-generated content at query #20
#--------------------------

```python
def test_CurrencyRegistry():
    # Test singleton behavior
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2

    # Test initial state
    assert len(registry1) == 0
    assert registry1.all == []
    assert registry1.codes == []
    assert registry1.codenames == []

    # Test context manager behavior
    with registry1 as register:
        # Test adding currencies
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        register(usd)
        assert len(registry1) == 1
        assert "USD" in registry1
        assert registry1["USD"] == usd

        # Test adding duplicate currency raises error
        with pytest.raises(ValueError, match="Currency USD is already registered."):
            register(usd)

    # Test post-context state
    assert len(registry1) == 1
    assert registry1.all == [usd]
    assert registry1.codes == ["USD"]
    assert registry1.codenames == [("USD", "US Dollar")]

    # Test adding outside context raises error
    with pytest.raises(ProgrammingError, match="Can not create currencies outside registry context."):
        registry1._CurrencyRegistry__register(usd)

    # Test lookup methods
    assert registry1.has("USD") is True
    assert registry1.has("XYZ") is False
    assert registry1.get("USD") == usd
    assert registry1.get("XYZ") is None
    assert registry1.get("XYZ", default=usd) == usd

    # Test lookup error
    with pytest.raises(CurrencyLookupError, match="Currency identified by code 'XYZ' does not exist"):
        registry1["XYZ"]


# LLM-generated content at query #21
#--------------------------

```python
def test_CurrencyRegistry():
    # Test singleton behavior
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2

    # Test initial state
    assert len(registry1) == 0
    assert registry1.all == []
    assert registry1.codes == []
    assert registry1.codenames == []

    # Test context manager
    with registry1 as register:
        # Test adding currencies
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        register(usd)
        assert len(registry1) == 1
        assert "USD" in registry1
        assert registry1["USD"] == usd

        # Test duplicate currency
        with pytest.raises(ValueError):
            register(usd)

    # Test final state after context
    assert len(registry1) == 1
    assert registry1.all == [usd]
    assert registry1.codes == ["USD"]
    assert registry1.codenames == [("USD", "US Dollar")]

    # Test adding outside context
    with pytest.raises(ProgrammingError):
        registry1._CurrencyRegistry__register(usd)

    # Test get methods
    assert registry1.has("USD") is True
    assert registry1.has("XYZ") is False
    assert registry1.get("USD") == usd
    assert registry1.get("XYZ") is None
    assert registry1.get("XYZ", default=usd) == usd

    # Test lookup error
    with pytest.raises(CurrencyLookupError):
        registry1["XYZ"]


# LLM-generated content at query #22
#--------------------------

```python
def test_Currency___hash__():
    # Test basic hash functionality
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert hash(usd1) == hash(usd2)

    # Test hash differs for different currencies
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert hash(usd1) != hash(jpy)

    # Test hash differs when any attribute changes
    usd_different_name = Currency.of("USD", "Different Name", 2, CurrencyType.MONEY)
    assert hash(usd1) != hash(usd_different_name)

    usd_different_decimals = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert hash(usd1) != hash(usd_different_decimals)

    usd_different_type = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert hash(usd1) != hash(usd_different_type)

    # Test hash is consistent with equality
    assert usd1 == usd2
    assert hash(usd1) == hash(usd2)

    # Test hash is cached (same object returns same hash)
    assert hash(usd1) == usd1.hashcache


# LLM-generated content at query #23
#--------------------------

```python
def test_Currency___hash__():
    # Test basic hash consistency
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert hash(usd1) == hash(usd2)

    # Test hash changes with different attributes
    usd3 = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert hash(usd1) != hash(usd3)

    # Test hash changes with different currency type
    usd4 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert hash(usd1) != hash(usd4)

    # Test hash changes with different name
    usd5 = Currency.of("USD", "US Dollars 2", 2, CurrencyType.MONEY)
    assert hash(usd1) != hash(usd5)

    # Test hash changes with different code
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert hash(usd1) != hash(eur)

    # Test hash changes with different decimals
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert hash(usd1) != hash(jpy)

    # Test hash changes with negative decimals
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert hash(usd1) != hash(zzz)


# LLM-generated content at query #24
#--------------------------

```python
def test_CurrencyRegistry_get():
    # Setup
    registry = CurrencyRegistry()
    with registry as register:
        register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        register(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))

    # Test existing currency
    assert registry.get("USD").code == "USD"
    assert registry.get("EUR").name == "Euro"

    # Test non-existing currency with default
    assert registry.get("XYZ") is None
    assert registry.get("XYZ", default=registry["USD"]) == registry["USD"]

    # Test non-existing currency without default
    assert registry.get("NON-EXISTING") is None


# LLM-generated content at query #25
#--------------------------

```python
def test_CurrencyRegistry_get():
    # Setup
    registry = CurrencyRegistry()
    with registry:
        registry(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        registry(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))

    # Test existing currency
    assert registry.get("USD").code == "USD"
    assert registry.get("EUR").name == "Euro"

    # Test non-existing currency with default
    assert registry.get("XYZ") is None
    assert registry.get("XYZ", default=registry["USD"]) == registry["USD"]

    # Test non-existing currency without default
    assert registry.get("NON-EXISTING") is None


# LLM-generated content at query #26
#--------------------------

```python
def test_Currency___le__():
    # Test with equal currencies
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 <= usd2
    assert usd2 <= usd1

    # Test with different currencies
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert usd <= jpy or jpy <= usd  # One should be less than or equal to the other

    # Test with different attributes
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert usd1 <= usd2 or usd2 <= usd1  # One should be less than or equal to the other

    # Test with non-Currency object
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd <= "not a currency")


# LLM-generated content at query #27
#--------------------------

```python
def test_Currency___delattr__():
    currency = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)

    with pytest.raises(AttributeError):
        del currency.code

    with pytest.raises(AttributeError):
        del currency.name

    with pytest.raises(AttributeError):
        del currency.decimals

    with pytest.raises(AttributeError):
        del currency.type

    with pytest.raises(AttributeError):
        del currency.quantizer

    with pytest.raises(AttributeError):
        del currency.hashcache


# LLM-generated content at query #28
#--------------------------

```python
def test_CurrencyRegistry___new__():
    # Test that CurrencyRegistry is a singleton
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2

    # Test that the singleton instance is created only once
    assert CurrencyRegistry._CurrencyRegistry__instance is not None
    assert CurrencyRegistry._CurrencyRegistry__instance is registry1
    assert CurrencyRegistry._CurrencyRegistry__instance is registry2


# LLM-generated content at query #29
#--------------------------

```python
def test_CurrencyLookupError():
    code = "XYZ"
    error = CurrencyLookupError(code)
    assert error.code == code
    assert str(error) == f"Currency identified by code '{code}' does not exist"


# LLM-generated content at query #30
#--------------------------

```python
def test_CurrencyRegistry___enter__():
    # Test that __enter__ returns the __register method
    registry = CurrencyRegistry()
    with registry as register:
        assert callable(register)
        assert register.__name__ == "_CurrencyRegistry__register"

    # Test that __enter__ sets __ctx_open to True
    registry = CurrencyRegistry()
    assert not registry._CurrencyRegistry__ctx_open
    with registry:
        assert registry._CurrencyRegistry__ctx_open
    assert not registry._CurrencyRegistry__ctx_open

    # Test that __enter__ can be used to register currencies
    registry = CurrencyRegistry()
    test_currency = Currency.of("TEST", "Test Currency", 2, CurrencyType.MONEY)
    with registry as register:
        register(test_currency)
    assert "TEST" in registry
    assert registry["TEST"] == test_currency


# LLM-generated content at query #31
#--------------------------

```python
def test_CurrencyRegistry():
    # Test singleton behavior
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2

    # Test initial state
    assert len(registry1) == 0
    assert registry1.all == []
    assert registry1.codes == []
    assert registry1.codenames == []

    # Test context manager behavior
    with registry1 as register:
        # Test adding currencies in context
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        register(usd)
        assert len(registry1) == 1
        assert "USD" in registry1
        assert registry1["USD"] == usd

        # Test adding duplicate currency raises error
        with pytest.raises(ValueError, match="Currency USD is already registered."):
            register(usd)

    # Test post-context state
    assert len(registry1) == 1
    assert registry1.all == [usd]
    assert registry1.codes == ["USD"]
    assert registry1.codenames == [("USD", "US Dollar")]

    # Test adding outside context raises error
    with pytest.raises(ProgrammingError, match="Can not create currencies outside registry context."):
        registry1._CurrencyRegistry__register(usd)

    # Test lookup methods
    assert registry1.has("USD") is True
    assert registry1.has("XYZ") is False
    assert registry1.get("USD") == usd
    assert registry1.get("XYZ") is None
    assert registry1.get("XYZ", default=usd) == usd

    # Test lookup error
    with pytest.raises(CurrencyLookupError, match="Currency identified by code 'XYZ' does not exist"):
        registry1["XYZ"]


# LLM-generated content at query #32
#--------------------------

```python
def test_Currency___eq__():
    # Test equality with same currency
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2

    # Test inequality with different currency
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert not (usd == jpy)

    # Test inequality with different name
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 == usd2)

    # Test inequality with different decimals
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert not (usd1 == usd2)

    # Test inequality with different type
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert not (usd1 == usd2)

    # Test inequality with non-Currency object
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd == "USD")


# LLM-generated content at query #33
#--------------------------

```python
def test_Currency___lt__():
    # Test with same currency
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 < usd2)
    assert not (usd2 < usd1)

    # Test with different currencies
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert (jpy < usd) or (usd < jpy)  # One should be less than the other

    # Test with different attributes
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert (usdx < usd) or (usd < usdx)  # One should be less than the other

    # Test with different decimals
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert (zzz < usd) or (usd < zzz)  # One should be less than the other

    # Test with different types
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    gold = Currency.of("XAU", "Gold", 2, CurrencyType.METAL)
    assert (gold < usd) or (usd < gold)  # One should be less than the other


# LLM-generated content at query #34
#--------------------------

```python
def test_Currency___ge__():
    # Test with equal currencies
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 >= usd2
    assert usd2 >= usd1

    # Test with different currencies (same code but different name)
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 >= usdx)
    assert not (usdx >= usd1)

    # Test with different currencies (different code)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert not (usd >= jpy)
    assert not (jpy >= usd)

    # Test with different currencies (different decimals)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert not (usd >= zzz)
    assert not (zzz >= usd)

    # Test with different currencies (different type)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    gold = Currency.of("XAU", "Gold", 2, CurrencyType.METAL)
    assert not (usd >= gold)
    assert not (gold >= usd)


# LLM-generated content at query #35
#--------------------------

```python
def test_Currency___hash__():
    # Test with same currency instances
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert hash(usd1) == hash(usd2)

    # Test with different currency instances
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert hash(usd1) != hash(usdx)

    # Test with different currency codes
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert hash(usd1) != hash(eur)

    # Test with different currency decimals
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert hash(usd1) != hash(jpy)

    # Test with different currency types
    btc = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    assert hash(usd1) != hash(btc)


# LLM-generated content at query #36
#--------------------------

```python
def test_Currency___eq__():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)

    assert usd1 == usd2
    assert usd1 != usdx
    assert usd1 != jpy
    assert not (usd1 == "USD")
    assert not (usd1 == 123)


# LLM-generated content at query #37
#--------------------------

```python
def test_CurrencyRegistry___enter__():
    # Test that __enter__ returns the __register method
    registry = CurrencyRegistry()
    with registry as register:
        assert register == registry._CurrencyRegistry__register
        assert registry._CurrencyRegistry__ctx_open is True

    # Test that context is closed after exiting
    assert registry._CurrencyRegistry__ctx_open is False

    # Test that adding currency outside context raises error
    with pytest.raises(ProgrammingError, match="Can not create currencies outside registry context."):
        registry._CurrencyRegistry__register(Currency.of("TEST", "Test Currency", 2, CurrencyType.MONEY))


# LLM-generated content at query #38
#--------------------------

```python
def test_Currency___delattr__():
    with pytest.raises(AttributeError):
        usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
        del usd.code


# LLM-generated content at query #39
#--------------------------

```python
def test_CurrencyRegistry___enter__():
    # Setup
    registry = CurrencyRegistry()
    test_currency = Currency.of("TEST", "Test Currency", 2, CurrencyType.MONEY)

    # Test that __enter__ returns the __register method
    with registry as register:
        assert callable(register)
        register(test_currency)

    # Verify the currency was added
    assert "TEST" in registry
    assert registry["TEST"] == test_currency

    # Test context manager behavior
    with pytest.raises(ProgrammingError):
        registry.__register(test_currency)  # Should fail outside context

    # Verify context was properly closed
    assert registry._CurrencyRegistry__ctx_open is False


# LLM-generated content at query #40
#--------------------------

```python
def test_CurrencyRegistry():
    # Test singleton behavior
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2

    # Test initial state
    assert len(registry1) == 0
    assert registry1.all == []
    assert registry1.codes == []
    assert registry1.codenames == []

    # Test context manager behavior
    with registry1 as register:
        # Test adding a currency
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        register(usd)
        assert len(registry1) == 1
        assert "USD" in registry1
        assert registry1["USD"] == usd
        assert registry1.all == [usd]
        assert registry1.codes == ["USD"]
        assert registry1.codenames == [("USD", "US Dollar")]

        # Test adding another currency
        eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
        register(eur)
        assert len(registry1) == 2
        assert "EUR" in registry1
        assert registry1["EUR"] == eur
        assert registry1.all == [eur, usd]  # Should be sorted
        assert registry1.codes == ["EUR", "USD"]
        assert registry1.codenames == [("EUR", "Euro"), ("USD", "US Dollar")]

        # Test adding duplicate currency
        try:
            register(usd)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert str(e) == "Currency USD is already registered."

    # Test adding outside context
    try:
        registry1._CurrencyRegistry__register(usd)
        assert False, "Should have raised ProgrammingError"
    except ProgrammingError as e:
        assert str(e) == "Can not create currencies outside registry context."

    # Test has method
    assert registry1.has("USD")
    assert registry1.has("EUR")
    assert not registry1.has("XYZ")

    # Test get method
    assert registry1.get("USD") == usd
    assert registry1.get("EUR") == eur
    assert registry1.get("XYZ") is None
    assert registry1.get("XYZ", default=usd) == usd

    # Test lookup error
    try:
        registry1["XYZ"]
        assert False, "Should have raised CurrencyLookupError"
    except CurrencyLookupError as e:
        assert str(e) == "Currency identified by code 'XYZ' does not exist"


# LLM-generated content at query #41
#--------------------------

```python
def test_Currency___setattr__():
    # Given that Currency is a frozen dataclass, __setattr__ should raise an AttributeError
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)

    # Test that setting an attribute raises AttributeError
    with pytest.raises(AttributeError):
        usd.code = "EUR"

    with pytest.raises(AttributeError):
        usd.name = "Euros"

    with pytest.raises(AttributeError):
        usd.decimals = 0

    with pytest.raises(AttributeError):
        usd.type = CurrencyType.CRYPTO

    with pytest.raises(AttributeError):
        usd.quantizer = Decimal("0.01")

    with pytest.raises(AttributeError):
        usd.hashcache = 12345


# LLM-generated content at query #42
#--------------------------

```python
def test_CurrencyRegistry___contains__():
    registry = CurrencyRegistry()
    with registry as register:
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
        register(usd)
        register(eur)

    assert "USD" in registry
    assert "EUR" in registry
    assert "JPY" not in registry


# LLM-generated content at query #43
#--------------------------

```python
def test_Currency___hash__():
    # Test basic hash functionality
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert hash(usd1) == hash(usd2)

    # Test hash with different currency attributes
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert hash(usd1) != hash(usdx)

    # Test hash with different currency codes
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert hash(usd1) != hash(jpy)

    # Test hash with different currency types
    usd_crypto = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert hash(usd1) != hash(usd_crypto)

    # Test hash with different decimals
    usd_no_decimals = Currency.of("USD", "US Dollars", 0, CurrencyType.MONEY)
    assert hash(usd1) != hash(usd_no_decimals)

    # Test hash with negative decimals
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert hash(usd1) != hash(zzz)


# LLM-generated content at query #44
#--------------------------

```python
def test_Currency_quantize():
    # Test with USD (2 decimals)
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert USD.quantize(Decimal("1.005")) == Decimal("1.00")
    assert USD.quantize(Decimal("1.015")) == Decimal("1.02")
    assert USD.quantize(Decimal("1.000")) == Decimal("1.00")
    assert USD.quantize(Decimal("1.999")) == Decimal("2.00")

    # Test with JPY (0 decimals)
    JPY = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert JPY.quantize(Decimal("0.5")) == Decimal("0")
    assert JPY.quantize(Decimal("1.5")) == Decimal("2")
    assert JPY.quantize(Decimal("1.0")) == Decimal("1")
    assert JPY.quantize(Decimal("2.9")) == Decimal("3")

    # Test with a currency with no fixed precision (CRYPTO)
    ZZZ = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert ZZZ.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert ZZZ.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")
    assert ZZZ.quantize(Decimal("1.0000000000000")) == Decimal("1.000000000000")
    assert ZZZ.quantize(Decimal("1.0000000000009")) == Decimal("1.000000000001")

    # Test with negative values
    assert USD.quantize(Decimal("-1.005")) == Decimal("-1.00")
    assert USD.quantize(Decimal("-1.015")) == Decimal("-1.02")
    assert JPY.quantize(Decimal("-0.5")) == Decimal("0")
    assert JPY.quantize(Decimal("-1.5")) == Decimal("-2")
    assert ZZZ.quantize(Decimal("-1.0000000000005")) == Decimal("-1.000000000000")
    assert ZZZ.quantize(Decimal("-1.0000000000015")) == Decimal("-1.000000000002")


# LLM-generated content at query #45
#--------------------------

```python
def test_Currency___hash__():
    # Test with same currency
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert hash(usd1) == hash(usd2)

    # Test with different currencies
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert hash(usd) != hash(jpy)

    # Test with different attributes
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert hash(usd1) != hash(usd2)

    # Test with different decimals
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert hash(usd1) != hash(usd2)

    # Test with different names
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars ", 2, CurrencyType.MONEY)
    assert hash(usd1) != hash(usd2)


# LLM-generated content at query #46
#--------------------------

```python
def test_CurrencyRegistry():
    # Test singleton behavior
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2

    # Test initial state
    assert len(registry1) == 0
    assert registry1.all == []
    assert registry1.codes == []
    assert registry1.codenames == []

    # Test context manager
    with registry1 as register:
        # Test adding currencies
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
        register(usd)
        register(eur)

    # Test after context exit
    assert len(registry1) == 2
    assert registry1.all == [eur, usd]  # Should be sorted by code
    assert registry1.codes == ["EUR", "USD"]
    assert registry1.codenames == [("EUR", "Euro"), ("USD", "US Dollar")]

    # Test currency lookup
    assert registry1.has("USD")
    assert registry1["USD"] == usd
    assert registry1.get("EUR") == eur
    assert registry1.get("NONEXISTENT") is None
    assert registry1.get("NONEXISTENT", default=usd) == usd

    # Test contains
    assert "USD" in registry1
    assert "XXX" not in registry1

    # Test duplicate registration
    with pytest.raises(ValueError):
        with registry1 as register:
            register(usd)

    # Test registration outside context
    with pytest.raises(ProgrammingError):
        registry1._CurrencyRegistry__register(usd)

    # Test lookup error
    with pytest.raises(CurrencyLookupError):
        registry1["NONEXISTENT"]


# LLM-generated content at query #47
#--------------------------

```python
def test_Currency___eq__():
    # Test equality with same currency
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2

    # Test inequality with different currency
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert usd1 != usdx

    # Test inequality with non-Currency object
    assert not usd1.__eq__("not a currency")


# LLM-generated content at query #48
#--------------------------

```python
def test_Currency___gt__():
    # Test with same currency
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 > usd2)

    # Test with different currency codes
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert (eur > usd) or (usd > eur)  # One should be greater than the other

    # Test with different names
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars Different", 2, CurrencyType.MONEY)
    assert (usd1 > usd2) or (usd2 > usd1)  # One should be greater than the other

    # Test with different decimals
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert (usd > jpy) or (jpy > usd)  # One should be greater than the other

    # Test with different types
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    btc = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    assert (usd > btc) or (btc > usd)  # One should be greater than the other


# LLM-generated content at query #49
#--------------------------

```python
def test_Currency___delattr__():
    # Create a Currency instance
    currency = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)

    # Attempt to delete an attribute and verify it raises an AttributeError
    with pytest.raises(AttributeError):
        del currency.code

    with pytest.raises(AttributeError):
        del currency.name

    with pytest.raises(AttributeError):
        del currency.decimals

    with pytest.raises(AttributeError):
        del currency.type

    with pytest.raises(AttributeError):
        del currency.quantizer

    with pytest.raises(AttributeError):
        del currency.hashcache


# LLM-generated content at query #50
#--------------------------

```python
def test_Currency___hash__():
    # Test with same currency instances
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert hash(usd1) == hash(usd2)

    # Test with different currency instances
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert hash(usd1) != hash(usdx)

    # Test with different currency codes
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert hash(usd1) != hash(jpy)

    # Test with different currency types
    usd_crypto = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert hash(usd1) != hash(usd_crypto)

    # Test with different decimals
    usd_no_decimals = Currency.of("USD", "US Dollars", 0, CurrencyType.MONEY)
    assert hash(usd1) != hash(usd_no_decimals)


# LLM-generated content at query #51
#--------------------------

```python
def test_Currency___repr__():
    # Test with a standard currency
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert repr(usd) == "Currency(code='USD', name='US Dollars', decimals=2, type=CurrencyType.MONEY, quantizer=Decimal('0.01'), hashcache=" + str(hash(usd)) + ")"

    # Test with a currency with no decimals
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert repr(jpy) == "Currency(code='JPY', name='Japanese Yen', decimals=0, type=CurrencyType.MONEY, quantizer=Decimal('0'), hashcache=" + str(hash(jpy)) + ")"

    # Test with a currency with variable precision
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert repr(zzz) == "Currency(code='ZZZ', name='Some weird currency', decimals=-1, type=CurrencyType.CRYPTO, quantizer=MaxPrecisionQuantizer, hashcache=" + str(hash(zzz)) + ")"


# LLM-generated content at query #52
#--------------------------

```python
def test_Currency___lt__():
    # Test with different codes
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert usd < eur

    # Test with different names (same code)
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars 2", 2, CurrencyType.MONEY)
    assert usd1 < usd2

    # Test with different decimals
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy < usd

    # Test with different types
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    btc = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    assert usd < btc

    # Test with same attributes (should be False)
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 < usd2)


# LLM-generated content at query #53
#--------------------------

```python
def test_Currency___ge__():
    # Test equal currencies
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 >= usd2
    assert usd2 >= usd1

    # Test different currencies
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert usd >= jpy or jpy >= usd  # One should be greater than the other

    # Test with different attributes
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert usd1 >= usd2 or usd2 >= usd1

    # Test with different types
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    btc = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    assert usd >= btc or btc >= usd

    # Test with non-Currency object
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd >= "not a currency")


# LLM-generated content at query #54
#--------------------------

```python
def test_Currency___delattr__():
    with pytest.raises(AttributeError):
        currency = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
        del currency.code


# LLM-generated content at query #55
#--------------------------

```python
def test_CurrencyRegistry_get():
    # Setup
    registry = CurrencyRegistry()
    with registry as register:
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
        register(usd)
        register(eur)

    # Test existing currency with no default
    assert registry.get("USD") == usd

    # Test existing currency with default
    assert registry.get("EUR", default=usd) == eur

    # Test non-existing currency with no default
    assert registry.get("XYZ") is None

    # Test non-existing currency with default
    assert registry.get("XYZ", default=usd) == usd


# LLM-generated content at query #56
#--------------------------

```python
def test_CurrencyLookupError():
    # Test initialization and message
    error = CurrencyLookupError("XYZ")
    assert error.code == "XYZ"
    assert str(error) == "Currency identified by code 'XYZ' does not exist"

    # Test inheritance from LookupError
    assert isinstance(error, LookupError)

    # Test with different code
    error2 = CurrencyLookupError("ABC")
    assert error2.code == "ABC"
    assert str(error2) == "Currency identified by code 'ABC' does not exist"


# LLM-generated content at query #57
#--------------------------

```python
def test_CurrencyRegistry_has():
    # Test with existing currency
    assert Currencies.has("USD") is True

    # Test with non-existing currency
    assert Currencies.has("NONEXISTENT") is False

    # Test with empty string
    assert Currencies.has("") is False

    # Test with non-string input (should return False)
    assert Currencies.has(123) is False


# LLM-generated content at query #58
#--------------------------

```python
def test_CurrencyRegistry___exit__():
    # Setup
    registry = CurrencyRegistry()
    test_currency = Currency.of("TEST", "Test Currency", 2, CurrencyType.MONEY)

    # Test context manager behavior
    with registry as register:
        register(test_currency)

    # Verify the currency was added and sorted
    assert "TEST" in registry
    assert registry["TEST"] == test_currency
    assert registry.codes == ["TEST"]
    assert registry.codenames == [("TEST", "Test Currency")]

    # Verify context is closed
    try:
        registry._CurrencyRegistry__register(test_currency)
        assert False, "Should have raised ProgrammingError"
    except ProgrammingError:
        pass

    # Verify re-entering context works
    with registry as register:
        pass

    # Verify empty registry behavior
    empty_registry = CurrencyRegistry()
    with empty_registry as register:
        pass

    assert len(empty_registry) == 0
    assert empty_registry.codes == []
    assert empty_registry.codenames == []


# LLM-generated content at query #59
#--------------------------

```python
def test_CurrencyRegistry___contains__():
    # Setup
    registry = CurrencyRegistry()
    with registry as register:
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
        register(usd)
        register(eur)

    # Test existing currency
    assert "USD" in registry
    assert "EUR" in registry

    # Test non-existing currency
    assert "JPY" not in registry
    assert "XYZ" not in registry


# LLM-generated content at query #60
#--------------------------

```python
def test_CurrencyRegistry():
    # Test singleton behavior
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2

    # Test initial state
    assert len(registry1) == 0
    assert registry1.all == []
    assert registry1.codes == []
    assert registry1.codenames == []

    # Test context manager behavior
    with registry1 as register:
        # Test adding currencies
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
        register(usd)
        register(eur)

        # Test that currencies are added
        assert len(registry1) == 2
        assert registry1["USD"] == usd
        assert registry1["EUR"] == eur

        # Test that duplicate currencies raise error
        with pytest.raises(ValueError):
            register(usd)

    # Test that context is closed
    with pytest.raises(ProgrammingError):
        register(usd)

    # Test that currencies are sorted after context exit
    assert registry1.codes == ["EUR", "USD"]
    assert registry1.codenames == [("EUR", "Euro"), ("USD", "US Dollar")]

    # Test lookup methods
    assert registry1.has("USD")
    assert not registry1.has("XYZ")
    assert registry1.get("USD") == usd
    assert registry1.get("XYZ") is None
    assert registry1.get("XYZ", default=eur) == eur

    # Test lookup error
    with pytest.raises(CurrencyLookupError):
        registry1["XYZ"]


# LLM-generated content at query #61
#--------------------------

```python
def test_Currency___le__():
    # Test equality case
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 <= usd2

    # Test less than case
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert usd <= eur or eur <= usd

    # Test with different attributes
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert usd <= jpy or jpy <= usd

    # Test with different types
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    btc = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    assert usd <= btc or btc <= usd

    # Test with non-Currency object
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd <= "not a currency")


# LLM-generated content at query #62
#--------------------------

```python
def test_CurrencyRegistry_has():
    # Setup
    registry = CurrencyRegistry()
    with registry as register:
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
        register(usd)
        register(eur)

    # Test existing currency
    assert registry.has("USD") is True
    assert registry.has("EUR") is True

    # Test non-existing currency
    assert registry.has("XYZ") is False
    assert registry.has("NONEXISTENT") is False

    # Test case sensitivity
    assert registry.has("usd") is False
    assert registry.has("eur") is False


# LLM-generated content at query #63
#--------------------------

```python
def test_CurrencyRegistry_has():
    # Setup
    registry = CurrencyRegistry()

    # Test with non-existent currency
    assert not registry.has("NON_EXISTENT")

    # Test with existing currency (assuming USD is registered)
    with registry as register:
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        register(usd)

    assert registry.has("USD")
    assert not registry.has("EUR")


# LLM-generated content at query #64
#--------------------------

```python
def test_Currency___gt__():
    # Test with different codes
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert usd > eur
    assert not (eur > usd)

    # Test with same code but different names
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 > usd2)

    # Test with same code and name but different decimals
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 0, CurrencyType.MONEY)
    assert usd1 > usd2
    assert not (usd2 > usd1)

    # Test with same code, name, and decimals but different types
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert usd1 > usd2
    assert not (usd2 > usd1)


# LLM-generated content at query #65
#--------------------------

```python
def test_CurrencyRegistry_get():
    # Setup
    registry = CurrencyRegistry()
    with registry as register:
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
        register(usd)
        register(eur)

    # Test getting existing currency
    assert registry.get("USD") == usd
    assert registry.get("EUR") == eur

    # Test getting non-existing currency without default
    assert registry.get("XYZ") is None

    # Test getting non-existing currency with default
    default_currency = Currency.of("DEF", "Default", 2, CurrencyType.MONEY)
    assert registry.get("XYZ", default=default_currency) == default_currency

    # Test getting existing currency with default (default should be ignored)
    assert registry.get("USD", default=default_currency) == usd


# LLM-generated content at query #66
#--------------------------

```python
def test_Currency___repr__():
    # Test with a standard currency
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert repr(usd) == "Currency(code='USD', name='US Dollars', decimals=2, type=CurrencyType.MONEY, quantizer=Decimal('0.01'), hashcache={})".format(hash(usd))

    # Test with a currency that has 0 decimals
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert repr(jpy) == "Currency(code='JPY', name='Japanese Yen', decimals=0, type=CurrencyType.MONEY, quantizer=Decimal('0'), hashcache={})".format(hash(jpy))

    # Test with a currency that has variable precision
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert repr(zzz) == "Currency(code='ZZZ', name='Some weird currency', decimals=-1, type=CurrencyType.CRYPTO, quantizer=MaxPrecisionQuantizer, hashcache={})".format(hash(zzz))


# LLM-generated content at query #67
#--------------------------

```python
def test_Currency___ge__():
    # Test with equal currencies
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 >= usd2
    assert usd2 >= usd1

    # Test with different currencies (usd1 should be greater than usd3)
    usd3 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert usd1 >= usd3
    assert not (usd3 >= usd1)

    # Test with different currencies (usd1 should be greater than jpy)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert usd1 >= jpy
    assert not (jpy >= usd1)

    # Test with different currencies (usd1 should be greater than zzz)
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert usd1 >= zzz
    assert not (zzz >= usd1)


# LLM-generated content at query #68
#--------------------------

```python
def test_Currency_quantize():
    # Test USD currency with 2 decimals
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.quantize(Decimal("1.005")) == Decimal("1.00")
    assert usd.quantize(Decimal("1.015")) == Decimal("1.02")
    assert usd.quantize(Decimal("1.000")) == Decimal("1.00")
    assert usd.quantize(Decimal("1.999")) == Decimal("2.00")

    # Test JPY currency with 0 decimals
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy.quantize(Decimal("0.5")) == Decimal("0")
    assert jpy.quantize(Decimal("1.5")) == Decimal("2")
    assert jpy.quantize(Decimal("1.0")) == Decimal("1")
    assert jpy.quantize(Decimal("1.9")) == Decimal("2")

    # Test ZZZ currency with -1 decimals (no fixed precision)
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert zzz.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert zzz.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")
    assert zzz.quantize(Decimal("1.0000000000000")) == Decimal("1.000000000000")
    assert zzz.quantize(Decimal("1.0000000000009")) == Decimal("1.000000000001")

    # Test negative values
    assert usd.quantize(Decimal("-1.005")) == Decimal("-1.00")
    assert jpy.quantize(Decimal("-1.5")) == Decimal("-2")
    assert zzz.quantize(Decimal("-1.0000000000005")) == Decimal("-1.000000000000")

    # Test large values
    assert usd.quantize(Decimal("999999999.995")) == Decimal("999999999.99")
    assert jpy.quantize(Decimal("999999999.5")) == Decimal("999999999")
    assert zzz.quantize(Decimal("999999999.0000000000005")) == Decimal("999999999.000000000000")


# LLM-generated content at query #69
#--------------------------

```python
def test_Currency___le__():
    # Test equality case
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 <= usd2

    # Test less than case
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert usd <= eur or eur <= usd

    # Test with different attributes
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert usd <= jpy or jpy <= usd

    # Test with same code but different name
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars Different", 2, CurrencyType.MONEY)
    assert usd1 <= usd2 or usd2 <= usd1

    # Test with same code and name but different decimals
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert usd1 <= usd2 or usd2 <= usd1

    # Test with same code, name, decimals but different type
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert usd1 <= usd2 or usd2 <= usd1


# LLM-generated content at query #70
#--------------------------

```python
def test_CurrencyRegistry___len__():
    registry = CurrencyRegistry()
    with registry as register:
        register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        register(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))
    assert len(registry) == 2


# LLM-generated content at query #71
#--------------------------

```python
def test_CurrencyRegistry___getitem__():
    # Test getting an existing currency
    assert Currencies["USD"].code == "USD"
    assert Currencies["USD"].name == "US Dollar"

    # Test getting a non-existing currency raises CurrencyLookupError
    with pytest.raises(CurrencyLookupError):
        Currencies["NON-EXISTING"]


# LLM-generated content at query #72
#--------------------------

```python
def test_CurrencyRegistry___enter__():
    # Test that __enter__ returns the __register method
    registry = CurrencyRegistry()
    with registry as register:
        assert callable(register)
        assert register.__name__ == "_CurrencyRegistry__register"

    # Test that __enter__ sets __ctx_open to True
    registry = CurrencyRegistry()
    with registry as register:
        assert registry._CurrencyRegistry__ctx_open is True

    # Test that __exit__ sets __ctx_open to False
    registry = CurrencyRegistry()
    with registry as register:
        pass
    assert registry._CurrencyRegistry__ctx_open is False

    # Test that __enter__ can be called multiple times in nested contexts
    registry = CurrencyRegistry()
    with registry as register1:
        assert registry._CurrencyRegistry__ctx_open is True
        with registry as register2:
            assert registry._CurrencyRegistry__ctx_open is True
            assert callable(register2)
        assert registry._CurrencyRegistry__ctx_open is True
    assert registry._CurrencyRegistry__ctx_open is False


# LLM-generated content at query #73
#--------------------------

```python
def test_Currency___eq__():
    # Test equality with same currency
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2

    # Test inequality with different currency code
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert not (usd == eur)

    # Test inequality with different currency name
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 == usd2)

    # Test inequality with different decimals
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 0, CurrencyType.MONEY)
    assert not (usd1 == usd2)

    # Test inequality with different currency type
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert not (usd1 == usd2)

    # Test inequality with non-Currency object
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd == "USD")

    # Test inequality with None
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd == None)


# LLM-generated content at query #74
#--------------------------

```python
def test_Currency():
    # Test valid currency creation
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.code == "USD"
    assert usd.name == "US Dollars"
    assert usd.decimals == 2
    assert usd.type == CurrencyType.MONEY
    assert usd.quantize(Decimal("1.005")) == Decimal("1.00")
    assert usd.quantize(Decimal("1.015")) == Decimal("1.02")

    # Test equality and hash
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd == usd2
    assert hash(usd) == hash(usd2)

    # Test inequality
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert usd != usdx
    assert hash(usd) != hash(usdx)

    # Test JPY with different precision
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy.quantize(Decimal("0.5")) == Decimal("0")
    assert jpy.quantize(Decimal("1.5")) == Decimal("2")

    # Test weird currency with no fixed precision
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert zzz.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert zzz.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")

    # Test invalid currency code
    with pytest.raises(ProgrammingError):
        Currency.of("123", "Invalid Code", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("usd", "Lowercase Code", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD ", "Code with Space", 2, CurrencyType.MONEY)

    # Test invalid currency name
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", " ", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", " US Dollars", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars ", 2, CurrencyType.MONEY)

    # Test invalid decimals
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", -2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", "2", CurrencyType.MONEY)

    # Test invalid currency type
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", 2, "MONEY")


# LLM-generated content at query #75
#--------------------------

```python
def test_Currency___ge__():
    # Test equality cases
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 >= usd2
    assert usd2 >= usd1

    # Test greater than cases
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert usd >= jpy

    # Test less than cases (should return False)
    assert not (jpy >= usd)

    # Test with different attributes
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert not (usd >= usdx)
    assert not (usdx >= usd)

    # Test with different types
    gold = Currency.of("XAU", "Gold", 2, CurrencyType.METAL)
    assert usd >= gold or gold >= usd  # Should be deterministic based on hash


# LLM-generated content at query #76
#--------------------------

```python
def test_Currency___repr__():
    currency = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert repr(currency) == "Currency(code='USD', name='US Dollars', decimals=2, type=CurrencyType.MONEY, quantizer=Decimal('0.01'), hashcache={})".format(hash(('USD', 'US Dollars', 2, CurrencyType.MONEY, Decimal('0.01'))))


# LLM-generated content at query #77
#--------------------------

```python
def test_Currency___hash__():
    # Test basic hash functionality
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert hash(usd1) == hash(usd2)

    # Test hash with different attributes
    usd3 = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert hash(usd1) != hash(usd3)

    # Test hash with different code
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert hash(usd1) != hash(eur)

    # Test hash with different name
    usd4 = Currency.of("USD", "US Dollars Different", 2, CurrencyType.MONEY)
    assert hash(usd1) != hash(usd4)

    # Test hash with different type
    usd5 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert hash(usd1) != hash(usd5)

    # Test hash with different decimals
    usd6 = Currency.of("USD", "US Dollars", 0, CurrencyType.MONEY)
    assert hash(usd1) != hash(usd6)


# LLM-generated content at query #78
#--------------------------

```python
def test_CurrencyRegistry_get():
    # Setup
    registry = CurrencyRegistry()
    with registry as register:
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
        register(usd)
        register(eur)

    # Test existing currency with no default
    assert registry.get("USD") == usd
    assert registry.get("EUR") == eur

    # Test non-existing currency with no default
    assert registry.get("XYZ") is None

    # Test non-existing currency with default
    assert registry.get("XYZ", default=usd) == usd
    assert registry.get("ABC", default=eur) == eur

    # Test existing currency with default (should ignore default)
    assert registry.get("USD", default=eur) == usd
    assert registry.get("EUR", default=usd) == eur


# LLM-generated content at query #79
#--------------------------

```python
def test_CurrencyLookupError():
    # Test initialization and message
    error = CurrencyLookupError("XYZ")
    assert error.code == "XYZ"
    assert str(error) == "Currency identified by code 'XYZ' does not exist"

    # Test inheritance
    assert isinstance(error, LookupError)


# LLM-generated content at query #80
#--------------------------

```python
def test_CurrencyRegistry___getitem__():
    # Test getting an existing currency
    assert Currencies["USD"].code == "USD"
    assert Currencies["USD"].name == "US Dollar"
    assert Currencies["USD"].type == CurrencyType.MONEY

    # Test getting a non-existing currency raises CurrencyLookupError
    with pytest.raises(CurrencyLookupError) as excinfo:
        _ = Currencies["NON-EXISTING"]
    assert str(excinfo.value) == "Currency identified by code 'NON-EXISTING' does not exist"


# LLM-generated content at query #81
#--------------------------

```python
def test_CurrencyRegistry___exit__():
    # Create a new instance of CurrencyRegistry
    registry = CurrencyRegistry()

    # Enter the context and add some currencies
    with registry as register:
        register(Currency.of("AED", "UAE Dirham", 2, CurrencyType.MONEY))
        register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        register(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))

    # Verify that the registry is sorted by code
    assert registry.codes == ["AED", "EUR", "USD"]

    # Verify that the currencies list is sorted by code
    assert [c.code for c in registry.all] == ["AED", "EUR", "USD"]

    # Verify that the codenames list is sorted by code
    assert [(c[0], c[1]) for c in registry.codenames] == [
        ("AED", "UAE Dirham"),
        ("EUR", "Euro"),
        ("USD", "US Dollar")
    ]

    # Verify that the context is closed
    assert not registry._CurrencyRegistry__ctx_open


# LLM-generated content at query #82
#--------------------------

```python
def test_Currency___le__():
    # Test equality (should return True)
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 <= usd2

    # Test less than (should return True)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy <= usd

    # Test greater than (should return False)
    assert not (usd <= jpy)

    # Test with different attributes
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 <= usdx)
    assert usdx <= usd1

    # Test with different types
    crypto = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    assert crypto <= usd
    assert not (usd <= crypto)


# LLM-generated content at query #83
#--------------------------

```python
def test_CurrencyRegistry___getitem__():
    # Test getting an existing currency
    usd = CurrencyRegistry()["USD"]
    assert usd.code == "USD"
    assert usd.name == "US Dollar"
    assert usd.type == CurrencyType.MONEY

    # Test getting a non-existing currency raises CurrencyLookupError
    with pytest.raises(CurrencyLookupError) as excinfo:
        CurrencyRegistry()["NON-EXISTING"]
    assert str(excinfo.value) == "Currency identified by code 'NON-EXISTING' does not exist"


# LLM-generated content at query #84
#--------------------------

```python
def test_CurrencyRegistry___new__():
    # Test that CurrencyRegistry is a singleton
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2

    # Test that the singleton instance is created only once
    assert CurrencyRegistry._CurrencyRegistry__instance is not None
    assert CurrencyRegistry._CurrencyRegistry__instance is registry1
    assert CurrencyRegistry._CurrencyRegistry__instance is registry2


# LLM-generated content at query #85
#--------------------------

```python
def test_Currency___delattr__():
    with pytest.raises(AttributeError):
        usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
        del usd.code


# LLM-generated content at query #86
#--------------------------

```python
def test_Currency___eq__():
    # Test equality with same currency
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2

    # Test inequality with different currency code
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert not (usd == eur)

    # Test inequality with different currency name
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 == usd2)

    # Test inequality with different decimals
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 0, CurrencyType.MONEY)
    assert not (usd1 == usd2)

    # Test inequality with different currency type
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert not (usd1 == usd2)

    # Test inequality with non-Currency object
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd == "USD")
    assert not (usd == 123)
    assert not (usd == None)


# LLM-generated content at query #87
#--------------------------

```python
def test_Currency___repr__():
    # Test with standard currency (USD)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert repr(usd) == "Currency(code='USD', name='US Dollars', decimals=2, type=CurrencyType.MONEY, quantizer=Decimal('0.01'), hashcache={})".format(hash(usd))

    # Test with zero-decimal currency (JPY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert repr(jpy) == "Currency(code='JPY', name='Japanese Yen', decimals=0, type=CurrencyType.MONEY, quantizer=Decimal('0'), hashcache={})".format(hash(jpy))

    # Test with variable precision currency (CRYPTO)
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert repr(zzz) == "Currency(code='ZZZ', name='Some weird currency', decimals=-1, type=CurrencyType.CRYPTO, quantizer=MaxPrecisionQuantizer, hashcache={})".format(hash(zzz))


# LLM-generated content at query #88
#--------------------------

```python
def test_CurrencyRegistry_get():
    # Setup
    registry = CurrencyRegistry()
    with registry as register:
        register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        register(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))

    # Test existing currency with no default
    assert registry.get("USD") == registry["USD"]

    # Test non-existing currency with no default
    assert registry.get("XYZ") is None

    # Test non-existing currency with default
    default_currency = Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY)
    assert registry.get("XYZ", default=default_currency) == default_currency

    # Test existing currency with default (should ignore default)
    assert registry.get("EUR", default=default_currency) == registry["EUR"]


# LLM-generated content at query #89
#--------------------------

```python
def test_Currency___lt__():
    # Test with different codes
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert (usd < eur) == ("USD" < "EUR")

    # Test with same code but different names
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars 2", 2, CurrencyType.MONEY)
    assert (usd1 < usd2) == (("USD", "US Dollars", 2, CurrencyType.MONEY, usd1.quantizer) <
                             ("USD", "US Dollars 2", 2, CurrencyType.MONEY, usd2.quantizer))

    # Test with same code and name but different decimals
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert (usd1 < usd2) == (("USD", "US Dollars", 2, CurrencyType.MONEY, usd1.quantizer) <
                             ("USD", "US Dollars", 3, CurrencyType.MONEY, usd2.quantizer))

    # Test with same code, name, and decimals but different types
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert (usd1 < usd2) == (("USD", "US Dollars", 2, CurrencyType.MONEY, usd1.quantizer) <
                             ("USD", "US Dollars", 2, CurrencyType.CRYPTO, usd2.quantizer))

    # Test with same code, name, decimals, and type but different quantizers
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", -1, CurrencyType.MONEY)
    assert (usd1 < usd2) == (("USD", "US Dollars", 2, CurrencyType.MONEY, usd1.quantizer) <
                             ("USD", "US Dollars", -1, CurrencyType.MONEY, usd2.quantizer))

    # Test with same currency
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd < usd)


# LLM-generated content at query #90
#--------------------------

```python
def test_CurrencyRegistry___enter__():
    registry = CurrencyRegistry()
    with registry as register:
        assert callable(register)
        assert register.__name__ == "_CurrencyRegistry__register"
    assert not registry._CurrencyRegistry__ctx_open


# LLM-generated content at query #91
#--------------------------

```python
def test_CurrencyRegistry___getitem__():
    # Test getting an existing currency
    usd = CurrencyRegistry()["USD"]
    assert usd.code == "USD"
    assert usd.name == "US Dollar"
    assert usd.type == CurrencyType.MONEY

    # Test getting a non-existing currency raises CurrencyLookupError
    with pytest.raises(CurrencyLookupError) as excinfo:
        CurrencyRegistry()["NON-EXISTING"]
    assert str(excinfo.value) == "Currency identified by code 'NON-EXISTING' does not exist"


# LLM-generated content at query #92
#--------------------------

```python
def test_CurrencyRegistry_get():
    # Test getting an existing currency
    usd = Currencies.get("USD")
    assert usd is not None
    assert usd.code == "USD"
    assert usd.name == "US Dollar"

    # Test getting a non-existing currency without default
    assert Currencies.get("NONEXISTENT") is None

    # Test getting a non-existing currency with default
    default_currency = Currencies["EUR"]
    result = Currencies.get("NONEXISTENT", default=default_currency)
    assert result == default_currency

    # Test that getting a non-existing currency with no default returns None
    assert Currencies.get("ANOTHERNONEXISTENT") is None


# LLM-generated content at query #93
#--------------------------

```python
def test_CurrencyRegistry_get():
    # Setup
    registry = CurrencyRegistry()
    with registry as register:
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
        register(usd)
        register(eur)

    # Test existing currency with no default
    assert registry.get("USD") == usd
    assert registry.get("EUR") == eur

    # Test non-existing currency with no default
    assert registry.get("XYZ") is None

    # Test non-existing currency with default
    assert registry.get("XYZ", default=usd) == usd
    assert registry.get("ABC", default=eur) == eur

    # Test existing currency with default (should ignore default)
    assert registry.get("USD", default=eur) == usd
    assert registry.get("EUR", default=usd) == eur


# LLM-generated content at query #94
#--------------------------

```python
def test_Currency___eq__():
    # Test equality with same currency
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2

    # Test inequality with different currency code
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert not (usd == eur)

    # Test inequality with different currency name
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 == usd2)

    # Test inequality with different decimals
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 0, CurrencyType.MONEY)
    assert not (usd1 == usd2)

    # Test inequality with different currency type
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert not (usd1 == usd2)

    # Test inequality with non-Currency object
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd == "USD")
    assert not (usd == 123)
    assert not (usd == None)


# LLM-generated content at query #95
#--------------------------

```python
def test_Currency___hash__():
    # Test basic hash functionality
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)

    # Same currencies should have same hash
    assert hash(usd1) == hash(usd2)

    # Different currencies should have different hash
    assert hash(usd1) != hash(usdx)
    assert hash(usd1) != hash(jpy)

    # Hash should be consistent
    assert hash(usd1) == usd1.__hash__()
    assert hash(usd2) == usd2.__hash__()

    # Hash should be same as cached hash
    assert hash(usd1) == usd1.hashcache
    assert hash(usd2) == usd2.hashcache


# LLM-generated content at query #96
#--------------------------

```python
def test_CurrencyRegistry___contains__():
    # Test that existing currency code returns True
    assert "USD" in Currencies

    # Test that non-existing currency code returns False
    assert "NONEXISTENT" not in Currencies

    # Test case sensitivity
    assert "usd" not in Currencies
    assert "Usd" not in Currencies

    # Test with empty string
    assert "" not in Currencies

    # Test with special characters
    assert "USD!" not in Currencies
    assert "USD 123" not in Currencies


# LLM-generated content at query #97
#--------------------------

```python
def test_Currency_quantize():
    # Test USD with 2 decimals
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert USD.quantize(Decimal("1.005")) == Decimal("1.00")
    assert USD.quantize(Decimal("1.015")) == Decimal("1.02")

    # Test JPY with 0 decimals
    JPY = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert JPY.quantize(Decimal("0.5")) == Decimal("0")
    assert JPY.quantize(Decimal("1.5")) == Decimal("2")

    # Test ZZZ with -1 decimals (no fixed precision)
    ZZZ = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert ZZZ.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert ZZZ.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")

    # Test with negative values
    assert USD.quantize(Decimal("-1.005")) == Decimal("-1.00")
    assert JPY.quantize(Decimal("-1.5")) == Decimal("-2")

    # Test with zero
    assert USD.quantize(Decimal("0")) == Decimal("0.00")
    assert JPY.quantize(Decimal("0")) == Decimal("0")


# LLM-generated content at query #98
#--------------------------

```python
def test_CurrencyRegistry():
    # Test singleton behavior
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2

    # Test initial state
    assert len(registry1) == 0
    assert registry1.all == []
    assert registry1.codes == []
    assert registry1.codenames == []

    # Test context manager
    with registry1 as register:
        # Test adding currencies
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)

        register(usd)
        assert len(registry1) == 1
        assert registry1["USD"] == usd

        register(eur)
        assert len(registry1) == 2
        assert registry1["EUR"] == eur

        # Test duplicate currency
        try:
            register(usd)
            assert False, "Should have raised ValueError for duplicate currency"
        except ValueError as e:
            assert str(e) == "Currency USD is already registered."

    # Test final state after context exit
    assert len(registry1) == 2
    assert registry1.all == [eur, usd]  # Should be sorted by code
    assert registry1.codes == ["EUR", "USD"]
    assert registry1.codenames == [("EUR", "Euro"), ("USD", "US Dollar")]

    # Test outside context
    try:
        registry1._CurrencyRegistry__register(usd)
        assert False, "Should have raised ProgrammingError for outside context"
    except ProgrammingError as e:
        assert str(e) == "Can not create currencies outside registry context."

    # Test lookup methods
    assert "USD" in registry1
    assert registry1.has("USD")
    assert registry1.get("USD") == usd
    assert registry1.get("XXX") is None
    assert registry1.get("XXX", default=usd) == usd

    # Test invalid lookup
    try:
        registry1["XXX"]
        assert False, "Should have raised CurrencyLookupError"
    except CurrencyLookupError as e:
        assert str(e) == "Currency identified by code 'XXX' does not exist"


# LLM-generated content at query #99
#--------------------------

```python
def test_Currency_quantize():
    # Test with USD (2 decimals)
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert USD.quantize(Decimal("1.005")) == Decimal("1.00")
    assert USD.quantize(Decimal("1.015")) == Decimal("1.02")
    assert USD.quantize(Decimal("1.000")) == Decimal("1.00")
    assert USD.quantize(Decimal("1.999")) == Decimal("2.00")

    # Test with JPY (0 decimals)
    JPY = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert JPY.quantize(Decimal("0.5")) == Decimal("0")
    assert JPY.quantize(Decimal("1.5")) == Decimal("2")
    assert JPY.quantize(Decimal("1.0")) == Decimal("1")
    assert JPY.quantize(Decimal("9.9")) == Decimal("10")

    # Test with a currency with no fixed precision (-1 decimals)
    ZZZ = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert ZZZ.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert ZZZ.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")
    assert ZZZ.quantize(Decimal("1.0000000000000")) == Decimal("1.000000000000")
    assert ZZZ.quantize(Decimal("1.0000000000009")) == Decimal("1.000000000001")

    # Test with negative values
    assert USD.quantize(Decimal("-1.005")) == Decimal("-1.00")
    assert USD.quantize(Decimal("-1.015")) == Decimal("-1.02")
    assert JPY.quantize(Decimal("-0.5")) == Decimal("0")
    assert JPY.quantize(Decimal("-1.5")) == Decimal("-2")
    assert ZZZ.quantize(Decimal("-1.0000000000005")) == Decimal("-1.000000000000")
    assert ZZZ.quantize(Decimal("-1.0000000000015")) == Decimal("-1.000000000002")


# LLM-generated content at query #100
#--------------------------

```python
def test_CurrencyLookupError():
    # Test initialization and message
    error = CurrencyLookupError("XYZ")
    assert error.code == "XYZ"
    assert str(error) == "Currency identified by code 'XYZ' does not exist"

    # Test inheritance
    assert isinstance(error, LookupError)


# LLM-generated content at query #101
#--------------------------

```python
def test_CurrencyRegistry___getitem__():
    # Setup
    registry = CurrencyRegistry()
    with registry as register:
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
        register(usd)
        register(eur)

    # Test existing currency
    assert registry["USD"] == usd
    assert registry["EUR"] == eur

    # Test non-existing currency raises CurrencyLookupError
    with pytest.raises(CurrencyLookupError):
        registry["NONEXISTENT"]


# LLM-generated content at query #102
#--------------------------

```python
def test_CurrencyRegistry_has():
    # Test with existing currency
    assert CurrencyRegistry().has("USD") is True

    # Test with non-existing currency
    assert CurrencyRegistry().has("NONEXISTENT") is False

    # Test with empty string
    assert CurrencyRegistry().has("") is False

    # Test with None (should not raise error)
    assert CurrencyRegistry().has(None) is False


# LLM-generated content at query #103
#--------------------------

```python
def test_CurrencyRegistry___len__():
    registry = CurrencyRegistry()
    assert len(registry) == 0

    with registry as register:
        register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        register(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))

    assert len(registry) == 2


# LLM-generated content at query #104
#--------------------------

```python
def test_Currency___hash__():
    # Test that hash is consistent for the same currency
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert hash(usd1) == hash(usd2)

    # Test that hash is different for different currencies
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert hash(usd1) != hash(jpy)

    # Test that hash is different for currencies with different attributes
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert hash(usd1) != hash(usdx)

    # Test that hash is consistent with equality
    assert usd1 == usd2
    assert hash(usd1) == hash(usd2)

    # Test that hash is different for currencies with different types
    usd_crypto = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert hash(usd1) != hash(usd_crypto)


# LLM-generated content at query #105
#--------------------------

```python
def test_CurrencyRegistry___enter__():
    # Test that __enter__ returns the __register method
    registry = CurrencyRegistry()
    with registry as register:
        assert callable(register)
        assert register.__name__ == "_CurrencyRegistry__register"

    # Test that __enter__ sets __ctx_open to True
    registry = CurrencyRegistry()
    assert not registry._CurrencyRegistry__ctx_open
    with registry as register:
        assert registry._CurrencyRegistry__ctx_open
    assert not registry._CurrencyRegistry__ctx_open


# LLM-generated content at query #106
#--------------------------

```python
def test_Currency___hash__():
    # Test that the hash is computed correctly and cached
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)

    # Test that the hash is computed correctly
    assert hash(usd1) == hash(usd2)
    assert hash(usd1) != hash(usdx)
    assert hash(usd1) != hash(jpy)

    # Test that the hash is cached
    assert usd1.hashcache == hash(usd1)
    assert usd2.hashcache == hash(usd2)
    assert usdx.hashcache == hash(usdx)
    assert jpy.hashcache == hash(jpy)


# LLM-generated content at query #107
#--------------------------

```python
def test_Currency___hash__():
    # Test that hash returns the pre-computed hash
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.__hash__() == hash(("USD", "US Dollars", 2, CurrencyType.MONEY, make_quantizer(2)))

    # Test that two currencies with same attributes have same hash
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1.__hash__() == usd2.__hash__()

    # Test that two currencies with different attributes have different hash
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert usd.__hash__() != jpy.__hash__()

    # Test that hash is consistent with equality
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert (usd1 == usd2) and (usd1.__hash__() == usd2.__hash__())


# LLM-generated content at query #108
#--------------------------

```python
def test_Currency___repr__():
    # Test basic currency representation
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert repr(usd) == "Currency(code='USD', name='US Dollars', decimals=2, type=CurrencyType.MONEY, quantizer=Decimal('0.01'), hashcache=" + str(hash(("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal('0.01')))) + ")"

    # Test currency with no decimals
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert repr(jpy) == "Currency(code='JPY', name='Japanese Yen', decimals=0, type=CurrencyType.MONEY, quantizer=Decimal('0'), hashcache=" + str(hash(("JPY", "Japanese Yen", 0, CurrencyType.MONEY, Decimal('0')))) + ")"

    # Test currency with variable precision
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert repr(zzz) == "Currency(code='ZZZ', name='Some weird currency', decimals=-1, type=CurrencyType.CRYPTO, quantizer=MaxPrecisionQuantizer, hashcache=" + str(hash(("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO, MaxPrecisionQuantizer))) + ")"


# LLM-generated content at query #109
#--------------------------

```python
def test_Currency___le__():
    # Test equality (should return True)
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 <= usd2

    # Test less than (should return True)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert usd <= eur or eur <= usd  # At least one should be True

    # Test greater than (should return False)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert not (eur <= jpy) or not (jpy <= eur)  # At least one should be False

    # Test with different attributes
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert usd <= usdx or usdx <= usd  # At least one should be True


# LLM-generated content at query #110
#--------------------------

```python
def test_Currency___lt__():
    # Test with different codes
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert (usd < eur) == ("USD" < "EUR")

    # Test with same code but different names
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars 2", 2, CurrencyType.MONEY)
    assert not (usd1 < usd2) and not (usd2 < usd1)

    # Test with same code and name but different decimals
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert (usd1 < usd2) == (2 < 3)

    # Test with same code, name, decimals but different types
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert (usd1 < usd2) == (CurrencyType.MONEY < CurrencyType.CRYPTO)

    # Test with same code, name, decimals, type but different quantizers (should not happen in practice)
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal("0.01"), hash(("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal("0.01"))))
    assert not (usd1 < usd2) and not (usd2 < usd1)

    # Test with non-Currency object
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd < "not a currency")


# LLM-generated content at query #111
#--------------------------

```python
def test_Currency_quantize():
    # Test USD with 2 decimals
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert USD.quantize(Decimal("1.005")) == Decimal("1.00")
    assert USD.quantize(Decimal("1.015")) == Decimal("1.02")
    assert USD.quantize(Decimal("1.000")) == Decimal("1.00")
    assert USD.quantize(Decimal("1.999")) == Decimal("2.00")

    # Test JPY with 0 decimals
    JPY = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert JPY.quantize(Decimal("0.5")) == Decimal("0")
    assert JPY.quantize(Decimal("1.5")) == Decimal("2")
    assert JPY.quantize(Decimal("1.0")) == Decimal("1")
    assert JPY.quantize(Decimal("2.9")) == Decimal("3")

    # Test ZZZ with -1 decimals (no fixed precision)
    ZZZ = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert ZZZ.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert ZZZ.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")
    assert ZZZ.quantize(Decimal("1.0000000000000")) == Decimal("1.000000000000")
    assert ZZZ.quantize(Decimal("1.0000000000009")) == Decimal("1.000000000001")


# LLM-generated content at query #112
#--------------------------

```python
def test_Currency___le__():
    # Test equality case
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 <= usd2

    # Test less than case
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert usd <= eur or eur <= usd  # At least one should be true

    # Test with different attributes
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert usd <= jpy or jpy <= usd  # At least one should be true

    # Test with different types
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    btc = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    assert usd <= btc or btc <= usd  # At least one should be true

    # Test with non-Currency object
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd <= "not a currency")


# LLM-generated content at query #113
#--------------------------

```python
def test_CurrencyRegistry___enter__():
    registry = CurrencyRegistry()
    with registry as register:
        assert callable(register)
        assert register.__name__ == "_CurrencyRegistry__register"
        assert registry._CurrencyRegistry__ctx_open is True
    assert registry._CurrencyRegistry__ctx_open is False


# LLM-generated content at query #114
#--------------------------

```python
def test_CurrencyRegistry___len__():
    registry = CurrencyRegistry()
    with registry as register:
        register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        register(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))
        register(Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY))

    assert len(registry) == 3


# LLM-generated content at query #115
#--------------------------

```python
def test_Currency___le__():
    # Test equality (should return True)
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 <= usd2
    assert usd2 <= usd1

    # Test less than (should return True)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert usd <= eur or eur <= usd  # At least one should be True

    # Test greater than (should return False)
    if usd <= eur:
        assert not (eur <= usd)
    else:
        assert not (usd <= eur)

    # Test with different attributes
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 <= usdx) or not (usdx <= usd1)  # Should not be equal


# LLM-generated content at query #116
#--------------------------

```python
def test_CurrencyRegistry():
    # Test singleton behavior
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2

    # Test initial state
    assert len(registry1) == 0
    assert registry1.codes == []
    assert registry1.codenames == []
    assert registry1.all == []

    # Test context manager behavior
    with registry1 as register:
        # Test adding a currency
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        register(usd)

        # Test adding duplicate currency raises error
        with pytest.raises(ValueError, match="Currency USD is already registered."):
            register(usd)

    # Test post-context state
    assert len(registry1) == 1
    assert registry1.codes == ["USD"]
    assert registry1.codenames == [("USD", "US Dollar")]
    assert registry1.all == [usd]

    # Test currency lookup
    assert registry1.has("USD")
    assert "USD" in registry1
    assert registry1["USD"] == usd
    assert registry1.get("USD") == usd
    assert registry1.get("NONEXISTENT") is None
    assert registry1.get("NONEXISTENT", default=usd) == usd

    # Test non-existent currency raises error
    with pytest.raises(CurrencyLookupError, match="Currency identified by code 'NONEXISTENT' does not exist"):
        registry1["NONEXISTENT"]

    # Test adding outside context raises error
    with pytest.raises(ProgrammingError, match="Can not create currencies outside registry context."):
        registry1._CurrencyRegistry__register(usd)


# LLM-generated content at query #117
#--------------------------

```python
def test_Currency___setattr__():
    # Test that Currency is immutable by attempting to set an attribute
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)

    # Attempt to modify an attribute
    with pytest.raises(AttributeError):
        USD.code = "EUR"

    with pytest.raises(AttributeError):
        USD.name = "Euro"

    with pytest.raises(AttributeError):
        USD.decimals = 3

    with pytest.raises(AttributeError):
        USD.type = CurrencyType.CRYPTO

    with pytest.raises(AttributeError):
        USD.quantizer = Decimal("0.001")

    with pytest.raises(AttributeError):
        USD.hashcache = 12345


# LLM-generated content at query #118
#--------------------------

```python
def test_Currency___gt__():
    # Test with different codes
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert ccy1 > ccy2
    assert not ccy2 > ccy1

    # Test with same code but different names
    ccy3 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy4 = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert ccy3 > ccy4
    assert not ccy4 > ccy3

    # Test with same code and name but different decimals
    ccy5 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy6 = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert ccy5 > ccy6
    assert not ccy6 > ccy5

    # Test with same code, name, and decimals but different types
    ccy7 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy8 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert ccy7 > ccy8
    assert not ccy8 > ccy7

    # Test with same code, name, decimals, and type but different quantizers
    ccy9 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy10 = Currency.of("USD", "US Dollars", -1, CurrencyType.MONEY)
    assert ccy9 > ccy10
    assert not ccy10 > ccy9

    # Test with same currency
    ccy11 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy12 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not ccy11 > ccy12
    assert not ccy12 > ccy11


# LLM-generated content at query #119
#--------------------------

```python
def test_CurrencyRegistry___enter__():
    # Test that __enter__ returns the __register method
    registry = CurrencyRegistry()
    with registry as register:
        assert callable(register)
        assert register.__name__ == "__register"

    # Test that __enter__ sets __ctx_open to True
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open is False
    with registry as register:
        assert registry._CurrencyRegistry__ctx_open is True
    assert registry._CurrencyRegistry__ctx_open is False

    # Test that __enter__ can be called multiple times
    registry = CurrencyRegistry()
    with registry as register1:
        with registry as register2:
            assert registry._CurrencyRegistry__ctx_open is True
            assert register1 == register2
        assert registry._CurrencyRegistry__ctx_open is True
    assert registry._CurrencyRegistry__ctx_open is False


# LLM-generated content at query #120
#--------------------------

```python
def test_Currency___hash__():
    # Test basic hash consistency
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert hash(usd1) == hash(usd2)

    # Test hash uniqueness for different currencies
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert hash(usd1) != hash(jpy)

    # Test hash uniqueness for same code but different attributes
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert hash(usd1) != hash(usdx)

    # Test hash consistency with different decimal values
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert hash(usd1) != hash(zzz)

    # Test hash consistency with different currency types
    gold = Currency.of("XAU", "Gold", 2, CurrencyType.METAL)
    assert hash(usd1) != hash(gold)


# LLM-generated content at query #121
#--------------------------

```python
def test_CurrencyRegistry___contains__():
    # Test with existing currency code
    assert "USD" in Currencies

    # Test with non-existing currency code
    assert "NONEXISTENT" not in Currencies

    # Test with empty string
    assert "" not in Currencies

    # Test with code that has special characters
    assert "USD$" not in Currencies

    # Test with lowercase code (assuming registry is case-sensitive)
    assert "usd" not in Currencies


# LLM-generated content at query #122
#--------------------------

```python
def test_CurrencyRegistry___exit__():
    # Test that __exit__ properly finalizes the registry
    registry = CurrencyRegistry()

    # Enter context and add some currencies
    with registry as register:
        c1 = Currency.of("ABC", "Test Currency 1", 2, CurrencyType.MONEY)
        c2 = Currency.of("DEF", "Test Currency 2", 0, CurrencyType.CRYPTO)
        c3 = Currency.of("GHI", "Test Currency 3", 4, CurrencyType.METAL)
        register(c1)
        register(c2)
        register(c3)

    # Verify the registry was properly sorted and finalized
    assert registry.codes == ["ABC", "DEF", "GHI"]
    assert registry.all == [c1, c2, c3]
    assert registry.codenames == [("ABC", "Test Currency 1"), ("DEF", "Test Currency 2"), ("GHI", "Test Currency 3")]
    assert registry["ABC"] == c1
    assert registry["DEF"] == c2
    assert registry["GHI"] == c3

    # Test that context is properly closed
    assert not registry._CurrencyRegistry__ctx_open

    # Test that adding outside context raises error
    try:
        registry._CurrencyRegistry__register(Currency.of("JKL", "Test Currency 4", 2, CurrencyType.ALTERNATIVE))
        assert False, "Should have raised ProgrammingError"
    except ProgrammingError:
        pass


# LLM-generated content at query #123
#--------------------------

```python
def test_Currency___eq__():
    # Test equality with same currency
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2

    # Test equality with different currency (different name)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 == usdx)

    # Test equality with different currency (different decimals)
    usd3 = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert not (usd1 == usd3)

    # Test equality with different currency (different type)
    usd4 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert not (usd1 == usd4)

    # Test equality with non-Currency object
    assert not (usd1 == "USD")
    assert not (usd1 == 123)
    assert not (usd1 == None)


# LLM-generated content at query #124
#--------------------------

```python
def test_CurrencyRegistry___contains__():
    # Setup
    registry = CurrencyRegistry()
    with registry as register:
        register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        register(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))

    # Test existing currency
    assert "USD" in registry
    assert "EUR" in registry

    # Test non-existing currency
    assert "XYZ" not in registry
    assert "NONEXISTENT" not in registry


# LLM-generated content at query #125
#--------------------------

```python
def test_CurrencyRegistry___len__():
    registry = CurrencyRegistry()
    with registry:
        registry(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        registry(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))
        registry(Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY))

    assert len(registry) == 3


# LLM-generated content at query #126
#--------------------------

```python
def test_CurrencyLookupError():
    # Test initialization and attributes
    error = CurrencyLookupError("XYZ")
    assert error.code == "XYZ"
    assert str(error) == "Currency identified by code 'XYZ' does not exist"

    # Test with different code
    error2 = CurrencyLookupError("ABC")
    assert error2.code == "ABC"
    assert str(error2) == "Currency identified by code 'ABC' does not exist"


# LLM-generated content at query #127
#--------------------------

```python
def test_Currency():
    # Test valid currency creation
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.code == "USD"
    assert usd.name == "US Dollars"
    assert usd.decimals == 2
    assert usd.type == CurrencyType.MONEY
    assert usd.quantize(Decimal("1.005")) == Decimal("1.00")
    assert usd.quantize(Decimal("1.015")) == Decimal("1.02")

    # Test JPY with 0 decimals
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy.code == "JPY"
    assert jpy.name == "Japanese Yen"
    assert jpy.decimals == 0
    assert jpy.type == CurrencyType.MONEY
    assert jpy.quantize(Decimal("0.5")) == Decimal("0")
    assert jpy.quantize(Decimal("1.5")) == Decimal("2")

    # Test currency with -1 decimals
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert zzz.code == "ZZZ"
    assert zzz.name == "Some weird currency"
    assert zzz.decimals == -1
    assert zzz.type == CurrencyType.CRYPTO
    assert zzz.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert zzz.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")

    # Test equality and hash
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2
    assert not (usd1 == usdx)
    assert hash(usd1) == hash(usd2)
    assert not (hash(usd1) == hash(usdx))

    # Test invalid currency code
    with pytest.raises(ProgrammingError):
        Currency.of("123", "Invalid Code", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("usd", "Lowercase Code", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD!", "Special Char Code", 2, CurrencyType.MONEY)

    # Test invalid currency name
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "  Trimmed Name  ", 2, CurrencyType.MONEY)

    # Test invalid decimals
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", -2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", "2", CurrencyType.MONEY)

    # Test invalid currency type
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", 2, "MONEY")


# LLM-generated content at query #128
#--------------------------

```python
def test_Currency_quantize():
    # Test with USD (2 decimals)
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert USD.quantize(Decimal("1.005")) == Decimal("1.00")
    assert USD.quantize(Decimal("1.015")) == Decimal("1.02")
    assert USD.quantize(Decimal("1.000")) == Decimal("1.00")
    assert USD.quantize(Decimal("1.999")) == Decimal("2.00")

    # Test with JPY (0 decimals)
    JPY = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert JPY.quantize(Decimal("0.5")) == Decimal("0")
    assert JPY.quantize(Decimal("1.5")) == Decimal("2")
    assert JPY.quantize(Decimal("1.0")) == Decimal("1")
    assert JPY.quantize(Decimal("999.9")) == Decimal("1000")

    # Test with a currency with no fixed precision (CRYPTO)
    ZZZ = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert ZZZ.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert ZZZ.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")
    assert ZZZ.quantize(Decimal("1.0000000000000")) == Decimal("1.000000000000")
    assert ZZZ.quantize(Decimal("1.0000000000001")) == Decimal("1.000000000000")

    # Test edge cases
    EUR = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert EUR.quantize(Decimal("0")) == Decimal("0.00")
    assert EUR.quantize(Decimal("0.001")) == Decimal("0.00")
    assert EUR.quantize(Decimal("0.009")) == Decimal("0.01")
    assert EUR.quantize(Decimal("999999999999.999")) == Decimal("1000000000000.00")


# LLM-generated content at query #129
#--------------------------

```python
def test_Currency():
    # Test valid currency creation
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.code == "USD"
    assert usd.name == "US Dollars"
    assert usd.decimals == 2
    assert usd.type == CurrencyType.MONEY
    assert usd.quantize(Decimal("1.005")) == Decimal("1.00")
    assert usd.quantize(Decimal("1.015")) == Decimal("1.02")

    # Test currency with zero decimals
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy.code == "JPY"
    assert jpy.name == "Japanese Yen"
    assert jpy.decimals == 0
    assert jpy.type == CurrencyType.MONEY
    assert jpy.quantize(Decimal("0.5")) == Decimal("0")
    assert jpy.quantize(Decimal("1.5")) == Decimal("2")

    # Test currency with variable precision
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert zzz.code == "ZZZ"
    assert zzz.name == "Some weird currency"
    assert zzz.decimals == -1
    assert zzz.type == CurrencyType.CRYPTO
    assert zzz.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert zzz.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")

    # Test equality and hash
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2
    assert usd1 != usdx
    assert hash(usd1) == hash(usd2)
    assert hash(usd1) != hash(usdx)

    # Test invalid currency code
    with pytest.raises(ProgrammingError):
        Currency.of("123", "Invalid Code", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("usd", "Lowercase Code", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD$", "Special Char Code", 2, CurrencyType.MONEY)

    # Test invalid currency name
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "  Trimmed  ", 2, CurrencyType.MONEY)

    # Test invalid decimals
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", -2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", "2", CurrencyType.MONEY)

    # Test invalid currency type
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", 2, "MONEY")


# LLM-generated content at query #130
#--------------------------

```python
def test_CurrencyLookupError():
    # Test initialization and attributes
    error = CurrencyLookupError("XYZ")
    assert error.code == "XYZ"
    assert str(error) == "Currency identified by code 'XYZ' does not exist"

    # Test with different code
    error2 = CurrencyLookupError("ABC")
    assert error2.code == "ABC"
    assert str(error2) == "Currency identified by code 'ABC' does not exist"


# LLM-generated content at query #131
#--------------------------

```python
def test_CurrencyLookupError():
    code = "XYZ"
    error = CurrencyLookupError(code)
    assert error.code == code
    assert str(error) == f"Currency identified by code '{code}' does not exist"


# LLM-generated content at query #132
#--------------------------

```python
def test_CurrencyRegistry():
    # Test singleton behavior
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2

    # Test initial state
    assert len(registry1) == 0
    assert registry1.all == []
    assert registry1.codes == []
    assert registry1.codenames == []

    # Test context manager
    with registry1 as register:
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        register(usd)

    assert len(registry1) == 1
    assert registry1["USD"] == usd
    assert registry1.has("USD") is True
    assert "USD" in registry1 is True
    assert registry1.all == [usd]
    assert registry1.codes == ["USD"]
    assert registry1.codenames == [("USD", "US Dollar")]

    # Test adding another currency
    with registry1 as register:
        eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
        register(eur)

    assert len(registry1) == 2
    assert registry1["EUR"] == eur
    assert registry1.has("EUR") is True
    assert "EUR" in registry1 is True
    assert registry1.all == [eur, usd]  # Should be sorted by code
    assert registry1.codes == ["EUR", "USD"]
    assert registry1.codenames == [("EUR", "Euro"), ("USD", "US Dollar")]

    # Test get with default
    assert registry1.get("USD") == usd
    assert registry1.get("XYZ") is None
    assert registry1.get("XYZ", default=usd) == usd

    # Test error cases
    with pytest.raises(CurrencyLookupError):
        registry1["XYZ"]

    with pytest.raises(ProgrammingError):
        registry1._CurrencyRegistry__register(usd)  # Try to add outside context

    with pytest.raises(ValueError):
        with registry1 as register:
            register(usd)  # Try to add duplicate


# LLM-generated content at query #133
#--------------------------

```python
def test_CurrencyRegistry_has():
    # Setup
    registry = CurrencyRegistry()
    with registry as register:
        register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        register(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))

    # Test existing currency
    assert registry.has("USD") is True
    assert registry.has("EUR") is True

    # Test non-existing currency
    assert registry.has("XYZ") is False
    assert registry.has("") is False


# LLM-generated content at query #134
#--------------------------

```python
def test_Currency___ge__():
    # Test equality (should return True)
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 >= usd2

    # Test greater than (should return True based on order)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert eur >= usd  # Assuming 'EUR' > 'USD' in string comparison

    # Test less than (should return False)
    assert not (usd >= eur)

    # Test with different attributes (should return False)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 >= usdx)


# LLM-generated content at query #135
#--------------------------

```python
def test_Currency():
    # Test valid currency creation
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.code == "USD"
    assert usd.name == "US Dollars"
    assert usd.decimals == 2
    assert usd.type == CurrencyType.MONEY
    assert usd.quantizer == make_quantizer(2)

    # Test currency with zero decimals
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy.quantizer == ZERO

    # Test currency with negative decimals
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert zzz.quantizer == MaxPrecisionQuantizer

    # Test equality and hash
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2
    assert hash(usd1) == hash(usd2)

    # Test inequality
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert usd1 != usdx
    assert hash(usd1) != hash(usdx)

    # Test quantize method
    assert usd.quantize(Decimal("1.005")) == Decimal("1.00")
    assert usd.quantize(Decimal("1.015")) == Decimal("1.02")
    assert jpy.quantize(Decimal("0.5")) == Decimal("0")
    assert jpy.quantize(Decimal("1.5")) == Decimal("2")
    assert zzz.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert zzz.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")

    # Test invalid currency code
    with pytest.raises(ProgrammingError):
        Currency.of("123", "Invalid Code", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("usd", "Lowercase Code", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD 1", "Code with space", 2, CurrencyType.MONEY)

    # Test invalid currency name
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "  Trimmed Name  ", 2, CurrencyType.MONEY)

    # Test invalid decimals
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", -2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", "2", CurrencyType.MONEY)

    # Test invalid currency type
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", 2, "MONEY")


# LLM-generated content at query #136
#--------------------------

```python
def test_Currency():
    # Test valid currency creation
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.code == "USD"
    assert usd.name == "US Dollars"
    assert usd.decimals == 2
    assert usd.type == CurrencyType.MONEY
    assert usd.quantize(Decimal("1.005")) == Decimal("1.00")
    assert usd.quantize(Decimal("1.015")) == Decimal("1.02")

    # Test currency with zero decimals
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy.code == "JPY"
    assert jpy.name == "Japanese Yen"
    assert jpy.decimals == 0
    assert jpy.type == CurrencyType.MONEY
    assert jpy.quantize(Decimal("0.5")) == Decimal("0")
    assert jpy.quantize(Decimal("1.5")) == Decimal("2")

    # Test currency with variable precision
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert zzz.code == "ZZZ"
    assert zzz.name == "Some weird currency"
    assert zzz.decimals == -1
    assert zzz.type == CurrencyType.CRYPTO
    assert zzz.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert zzz.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")

    # Test equality and hash
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2
    assert usd1 != usdx
    assert hash(usd1) == hash(usd2)
    assert hash(usd1) != hash(usdx)

    # Test invalid currency code
    with pytest.raises(ProgrammingError):
        Currency.of("123", "Invalid Code", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("usd", "Lowercase Code", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD ", "Code with space", 2, CurrencyType.MONEY)

    # Test invalid currency name
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", " ", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", " US Dollars", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars ", 2, CurrencyType.MONEY)

    # Test invalid decimals
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", -2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", "2", CurrencyType.MONEY)

    # Test invalid currency type
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", 2, "MONEY")


# LLM-generated content at query #137
#--------------------------

```python
def test_Currency():
    # Test valid currency creation
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.code == "USD"
    assert usd.name == "US Dollars"
    assert usd.decimals == 2
    assert usd.type == CurrencyType.MONEY
    assert usd.quantize(Decimal("1.005")) == Decimal("1.00")
    assert usd.quantize(Decimal("1.015")) == Decimal("1.02")

    # Test currency with zero decimals
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy.code == "JPY"
    assert jpy.name == "Japanese Yen"
    assert jpy.decimals == 0
    assert jpy.type == CurrencyType.MONEY
    assert jpy.quantize(Decimal("0.5")) == Decimal("0")
    assert jpy.quantize(Decimal("1.5")) == Decimal("2")

    # Test currency with negative decimals
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert zzz.code == "ZZZ"
    assert zzz.name == "Some weird currency"
    assert zzz.decimals == -1
    assert zzz.type == CurrencyType.CRYPTO
    assert zzz.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert zzz.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")

    # Test equality and hash
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2
    assert not (usd1 == usdx)
    assert hash(usd1) == hash(usd2)
    assert not (hash(usd1) == hash(usdx))

    # Test invalid currency code
    with pytest.raises(ProgrammingError):
        Currency.of("123", "Invalid Code", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("usd", "Lowercase Code", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD$", "Special Char Code", 2, CurrencyType.MONEY)

    # Test invalid currency name
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "  Trim Me  ", 2, CurrencyType.MONEY)

    # Test invalid decimals
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", -2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", "2", CurrencyType.MONEY)

    # Test invalid currency type
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", 2, "MONEY")


# LLM-generated content at query #138
#--------------------------

```python
def test_Currency___delattr__():
    with pytest.raises(AttributeError):
        currency = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
        del currency.code


# LLM-generated content at query #139
#--------------------------

```python
def test_CurrencyRegistry___len__():
    registry = CurrencyRegistry()
    with registry:
        registry(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        registry(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))
        registry(Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY))

    assert len(registry) == 3


# LLM-generated content at query #140
#--------------------------

```python
def test_Currency___le__():
    # Test equality (should return True)
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 <= usd2

    # Test less than (should return True)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy <= usd

    # Test greater than (should return False)
    assert not (usd <= jpy)

    # Test with different attributes
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 <= usdx)
    assert usdx <= usd1

    # Test with different types
    crypto = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    assert crypto <= usd
    assert not (usd <= crypto)


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_Currency___eq__():
    # Test equality with same currency
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2

    # Test inequality with different currency code
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert not (usd == eur)

    # Test inequality with different currency name
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 == usd2)

    # Test inequality with different decimals
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 0, CurrencyType.MONEY)
    assert not (usd1 == usd2)

    # Test inequality with different currency type
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert not (usd1 == usd2)

    # Test inequality with non-Currency object
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd == "USD")
    assert not (usd == 123)
    assert not (usd == None)


# LLM-generated content at query #2
#--------------------------

```python
def test_Currency___gt__():
    # Test with same currency
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 > usd2)

    # Test with different currency codes
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    assert (eur > usd) or (usd > eur)  # One should be greater

    # Test with different names
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert (usd1 > usd2) or (usd2 > usd1)  # One should be greater

    # Test with different decimals
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert (usd > jpy) or (jpy > usd)  # One should be greater

    # Test with different types
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    btc = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    assert (usd > btc) or (btc > usd)  # One should be greater

    # Test with non-Currency object
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    with pytest.raises(TypeError):
        usd > "not a currency"


# LLM-generated content at query #3
#--------------------------

```python
def test_CurrencyRegistry___len__():
    registry = CurrencyRegistry()
    assert len(registry) == 0

    with registry as register:
        register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        register(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))
        register(Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY))

    assert len(registry) == 3


# LLM-generated content at query #4
#--------------------------

```python
def test_Currency():
    # Test valid currency creation
    currency = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert currency.code == "USD"
    assert currency.name == "US Dollars"
    assert currency.decimals == 2
    assert currency.type == CurrencyType.MONEY
    assert currency.quantizer == make_quantizer(2)
    assert hash(currency) == hash(("USD", "US Dollars", 2, CurrencyType.MONEY, make_quantizer(2)))

    # Test currency with zero decimals
    currency = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert currency.code == "JPY"
    assert currency.name == "Japanese Yen"
    assert currency.decimals == 0
    assert currency.type == CurrencyType.MONEY
    assert currency.quantizer == ZERO
    assert hash(currency) == hash(("JPY", "Japanese Yen", 0, CurrencyType.MONEY, ZERO))

    # Test currency with negative decimals
    currency = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert currency.code == "ZZZ"
    assert currency.name == "Some weird currency"
    assert currency.decimals == -1
    assert currency.type == CurrencyType.CRYPTO
    assert currency.quantizer == MaxPrecisionQuantizer
    assert hash(currency) == hash(("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO, MaxPrecisionQuantizer))

    # Test equality
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2
    assert not (usd1 == usdx)
    assert hash(usd1) == hash(usd2)
    assert not (hash(usd1) == hash(usdx))

    # Test quantize method
    assert usd1.quantize(Decimal("1.005")) == Decimal("1.00")
    assert usd1.quantize(Decimal("1.015")) == Decimal("1.02")
    assert usd1.quantize(Decimal("1.005")) == Decimal("1.00")
    assert usd1.quantize(Decimal("1.015")) == Decimal("1.02")

    # Test invalid currency code
    try:
        Currency.of("usd", "US Dollars", 2, CurrencyType.MONEY)
        assert False, "Expected ProgrammingError"
    except ProgrammingError:
        pass

    try:
        Currency.of("123", "US Dollars", 2, CurrencyType.MONEY)
        assert False, "Expected ProgrammingError"
    except ProgrammingError:
        pass

    try:
        Currency.of("USD ", "US Dollars", 2, CurrencyType.MONEY)
        assert False, "Expected ProgrammingError"
    except ProgrammingError:
        pass

    # Test invalid currency name
    try:
        Currency.of("USD", "", 2, CurrencyType.MONEY)
        assert False, "Expected ProgrammingError"
    except ProgrammingError:
        pass

    try:
        Currency.of("USD", " US Dollars", 2, CurrencyType.MONEY)
        assert False, "Expected ProgrammingError"
    except ProgrammingError:
        pass

    try:
        Currency.of("USD", "US Dollars ", 2, CurrencyType.MONEY)
        assert False, "Expected ProgrammingError"
    except ProgrammingError:
        pass

    # Test invalid decimals
    try:
        Currency.of("USD", "US Dollars", -2, CurrencyType.MONEY)
        assert False, "Expected ProgrammingError"
    except ProgrammingError:
        pass

    try:
        Currency.of("USD", "US Dollars", "2", CurrencyType.MONEY)
        assert False, "Expected ProgrammingError"
    except ProgrammingError:
        pass

    # Test invalid currency type
    try:
        Currency.of("USD", "US Dollars", 2, "MONEY")
        assert False, "Expected ProgrammingError"
    except ProgrammingError:
        pass


# LLM-generated content at query #5
#--------------------------

```python
def test_Currency___repr__():
    # Test with USD
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert repr(usd) == "Currency(code='USD', name='US Dollars', decimals=2, type=CurrencyType.MONEY, quantizer=Decimal('0.01'), hashcache=" + str(hash(('USD', 'US Dollars', 2, CurrencyType.MONEY, Decimal('0.01')))) + ")"

    # Test with JPY
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert repr(jpy) == "Currency(code='JPY', name='Japanese Yen', decimals=0, type=CurrencyType.MONEY, quantizer=Decimal('0'), hashcache=" + str(hash(('JPY', 'Japanese Yen', 0, CurrencyType.MONEY, Decimal('0')))) + ")"

    # Test with a crypto currency
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert repr(zzz) == "Currency(code='ZZZ', name='Some weird currency', decimals=-1, type=CurrencyType.CRYPTO, quantizer=" + repr(MaxPrecisionQuantizer) + ", hashcache=" + str(hash(('ZZZ', 'Some weird currency', -1, CurrencyType.CRYPTO, MaxPrecisionQuantizer))) + ")"


# LLM-generated content at query #6
#--------------------------

```python
def test_Currency___lt__():
    # Test with same currency
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 < usd2)
    assert not (usd2 < usd1)

    # Test with different codes
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    assert (eur < usd) or (usd < eur)  # One must be less than the other
    assert not ((eur < usd) and (usd < eur))  # They can't both be less than each other

    # Test with different names
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars 2", 2, CurrencyType.MONEY)
    assert (usd1 < usd2) or (usd2 < usd1)
    assert not ((usd1 < usd2) and (usd2 < usd1))

    # Test with different decimals
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert (jpy < usd) or (usd < jpy)
    assert not ((jpy < usd) and (usd < jpy))

    # Test with different types
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    btc = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    assert (btc < usd) or (usd < btc)
    assert not ((btc < usd) and (usd < btc))

    # Test with different quantizers
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    weird = Currency.of("ZZZ", "Weird", -1, CurrencyType.CRYPTO)
    assert (weird < usd) or (usd < weird)
    assert not ((weird < usd) and (usd < weird))


# LLM-generated content at query #7
#--------------------------

```python
def test_Currency___gt__():
    # Test with different currencies
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    JPY = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    ZZZ = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)

    # Test greater than
    assert USD > JPY
    assert ZZZ > USD
    assert ZZZ > JPY

    # Test not greater than
    assert not (JPY > USD)
    assert not (USD > ZZZ)
    assert not (JPY > ZZZ)

    # Test with same currency
    USD2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (USD > USD2)
    assert not (USD2 > USD)


# LLM-generated content at query #8
#--------------------------

```python
def test_Currency___eq__():
    # Test equality with same currency
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2

    # Test inequality with different currency
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert not (usd == eur)

    # Test inequality with different name
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 == usd2)

    # Test inequality with different decimals
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 0, CurrencyType.MONEY)
    assert not (usd1 == usd2)

    # Test inequality with different type
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert not (usd1 == usd2)

    # Test inequality with non-Currency object
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd == "USD")


# LLM-generated content at query #9
#--------------------------

```python
def test_CurrencyRegistry___enter__():
    # Test that __enter__ returns the __register method
    registry = CurrencyRegistry()
    with registry as register:
        assert callable(register)
        assert register.__name__ == "_CurrencyRegistry__register"

    # Test that __enter__ sets __ctx_open to True
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open is False
    with registry:
        assert registry._CurrencyRegistry__ctx_open is True
    assert registry._CurrencyRegistry__ctx_open is False

    # Test that __enter__ can be used to register currencies
    registry = CurrencyRegistry()
    test_currency = Currency.of("TEST", "Test Currency", 2, CurrencyType.MONEY)
    with registry as register:
        register(test_currency)
    assert "TEST" in registry
    assert registry["TEST"] == test_currency


# LLM-generated content at query #10
#--------------------------

```python
def test_CurrencyRegistry___exit__():
    # Setup
    registry = CurrencyRegistry()
    test_currency = Currency.of("TEST", "Test Currency", 2, CurrencyType.MONEY)

    # Test context manager behavior
    with registry as register:
        register(test_currency)

    # Verify post-context behavior
    assert registry.has("TEST")
    assert registry["TEST"] == test_currency
    assert "TEST" in registry.codes
    assert ("TEST", "Test Currency") in registry.codenames
    assert registry["TEST"] in registry.all

    # Verify sorting
    assert registry.codes == sorted(registry.codes)
    assert registry.codenames == sorted(registry.codenames, key=lambda x: x[0])
    assert registry.all == sorted(registry.all, key=lambda x: x.code)

    # Verify context is closed
    try:
        registry._CurrencyRegistry__register(test_currency)
        assert False, "Should have raised ProgrammingError"
    except ProgrammingError:
        pass


# LLM-generated content at query #11
#--------------------------

```python
def test_CurrencyRegistry___enter__():
    # Test that __enter__ returns the __register method
    registry = CurrencyRegistry()
    with registry as register:
        assert callable(register)
        assert register.__name__ == "_CurrencyRegistry__register"

    # Test that __enter__ sets __ctx_open to True
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open is False
    with registry as register:
        assert registry._CurrencyRegistry__ctx_open is True

    # Test that __exit__ sets __ctx_open back to False
    registry = CurrencyRegistry()
    with registry as register:
        pass
    assert registry._CurrencyRegistry__ctx_open is False


# LLM-generated content at query #12
#--------------------------

```python
def test_CurrencyRegistry___exit__():
    registry = CurrencyRegistry()
    with registry as register:
        register(Currency.of("AED", "UAE Dirham", 2, CurrencyType.MONEY))
        register(Currency.of("BHD", "Bahraini Dinar", 3, CurrencyType.MONEY))
        register(Currency.of("CUC", "Cuban Convertible Peso", 2, CurrencyType.MONEY))

    assert len(registry) == 3
    assert registry.codes == ["AED", "BHD", "CUC"]
    assert registry.codenames == [
        ("AED", "UAE Dirham"),
        ("BHD", "Bahraini Dinar"),
        ("CUC", "Cuban Convertible Peso")
    ]
    assert registry.all == [
        Currency.of("AED", "UAE Dirham", 2, CurrencyType.MONEY),
        Currency.of("BHD", "Bahraini Dinar", 3, CurrencyType.MONEY),
        Currency.of("CUC", "Cuban Convertible Peso", 2, CurrencyType.MONEY)
    ]


# LLM-generated content at query #13
#--------------------------

```python
def test_Currency___eq__():
    # Test equality with same currency
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2

    # Test inequality with different currency
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 == usdx)

    # Test inequality with different type
    assert not (usd1 == "USD")

    # Test inequality with different code
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert not (usd1 == jpy)

    # Test inequality with different decimals
    zzz = Currency.of("USD", "US Dollars", 0, CurrencyType.MONEY)
    assert not (usd1 == zzz)

    # Test inequality with different type
    crypto_usd = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert not (usd1 == crypto_usd)


# LLM-generated content at query #14
#--------------------------

```python
def test_CurrencyRegistry___exit__():
    # Test that __exit__ properly finalizes the registry
    registry = CurrencyRegistry()
    with registry as register:
        register(Currency.of("TEST1", "Test Currency 1", 2, CurrencyType.MONEY))
        register(Currency.of("TEST2", "Test Currency 2", 0, CurrencyType.MONEY))
        register(Currency.of("TEST3", "Test Currency 3", -1, CurrencyType.CRYPTO))

    # Verify the registry is properly sorted
    assert registry.codes == ["TEST1", "TEST2", "TEST3"]
    assert len(registry) == 3
    assert registry["TEST1"].name == "Test Currency 1"
    assert registry["TEST2"].name == "Test Currency 2"
    assert registry["TEST3"].name == "Test Currency 3"

    # Verify the currencies buffer is properly sorted
    assert [c.code for c in registry.all] == ["TEST1", "TEST2", "TEST3"]

    # Verify the codenames buffer is properly sorted
    assert registry.codenames == [
        ("TEST1", "Test Currency 1"),
        ("TEST2", "Test Currency 2"),
        ("TEST3", "Test Currency 3")
    ]

    # Test that the context is properly closed
    try:
        registry.__register(Currency.of("TEST4", "Test Currency 4", 2, CurrencyType.MONEY))
        assert False, "Should have raised ProgrammingError"
    except ProgrammingError:
        pass


# LLM-generated content at query #15
#--------------------------

```python
def test_Currency___le__():
    # Test equality case
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 <= usd2

    # Test less than case
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert usd <= eur or eur <= usd  # At least one should be True

    # Test with different attributes
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert usd <= jpy or jpy <= usd  # At least one should be True

    # Test with different types
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    btc = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    assert usd <= btc or btc <= usd  # At least one should be True

    # Test with same code but different name
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars (Different)", 2, CurrencyType.MONEY)
    assert usd1 <= usd2 or usd2 <= usd1  # At least one should be True


# LLM-generated content at query #16
#--------------------------

```python
def test_CurrencyRegistry___new__():
    # Test that CurrencyRegistry is a singleton
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2

    # Test that the singleton instance is created only once
    assert CurrencyRegistry._CurrencyRegistry__instance is not None
    assert CurrencyRegistry._CurrencyRegistry__instance is registry1
    assert CurrencyRegistry._CurrencyRegistry__instance is registry2


# LLM-generated content at query #17
#--------------------------

```python
def test_Currency___gt__():
    # Test with different codes
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert usd > eur

    # Test with same code but different names
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars 2", 2, CurrencyType.MONEY)
    assert usd1 > usd2

    # Test with same code and name but different decimals
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert usd1 > usd2

    # Test with same code, name, and decimals but different types
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert usd1 > usd2

    # Test with same attributes
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 > usd2)


# LLM-generated content at query #18
#--------------------------

```python
def test_Currency___delattr__():
    # Create a currency instance
    currency = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)

    # Attempt to delete an attribute and expect an AttributeError
    with pytest.raises(AttributeError):
        del currency.code


# LLM-generated content at query #19
#--------------------------

```python
def test_CurrencyRegistry___exit__():
    # Setup
    registry = CurrencyRegistry()
    test_currency1 = Currency.of("TST", "Test Currency 1", 2, CurrencyType.MONEY)
    test_currency2 = Currency.of("TST2", "Test Currency 2", 0, CurrencyType.CRYPTO)

    # Test context manager behavior
    with registry as register:
        register(test_currency1)
        register(test_currency2)

    # Verify post-exit state
    assert len(registry) == 2
    assert registry.codes == ["TST", "TST2"]
    assert registry["TST"] == test_currency1
    assert registry["TST2"] == test_currency2
    assert registry.codenames == [("TST", "Test Currency 1"), ("TST2", "Test Currency 2")]

    # Test that context is closed after exit
    try:
        registry._CurrencyRegistry__register(test_currency1)
        assert False, "Should have raised ProgrammingError"
    except ProgrammingError:
        pass

    # Test re-sorting functionality
    test_currency3 = Currency.of("AAA", "Test Currency 3", 1, CurrencyType.METAL)
    with registry as register:
        register(test_currency3)

    assert registry.codes == ["AAA", "TST", "TST2"]


# LLM-generated content at query #20
#--------------------------

```python
def test_Currency___ge__():
    # Test equality cases
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 >= usd2
    assert usd2 >= usd1

    # Test inequality cases
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 >= usdx)
    assert not (usdx >= usd1)

    # Test with different currencies
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert not (usd1 >= jpy)
    assert not (jpy >= usd1)

    # Test with different currency types
    btc = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    assert not (usd1 >= btc)
    assert not (btc >= usd1)


# LLM-generated content at query #21
#--------------------------

```python
def test_CurrencyRegistry___new__():
    # Test singleton behavior
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2

    # Test that the instance is of the correct type
    assert isinstance(registry1, CurrencyRegistry)


# LLM-generated content at query #22
#--------------------------

```python
def test_Currency_quantize():
    # Test with USD (2 decimals)
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert USD.quantize(Decimal("1.005")) == Decimal("1.00")
    assert USD.quantize(Decimal("1.015")) == Decimal("1.02")
    assert USD.quantize(Decimal("1.000")) == Decimal("1.00")
    assert USD.quantize(Decimal("1.999")) == Decimal("2.00")

    # Test with JPY (0 decimals)
    JPY = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert JPY.quantize(Decimal("0.5")) == Decimal("0")
    assert JPY.quantize(Decimal("1.5")) == Decimal("2")
    assert JPY.quantize(Decimal("0.0")) == Decimal("0")
    assert JPY.quantize(Decimal("999.9")) == Decimal("1000")

    # Test with a currency that has no fixed precision (CRYPTO)
    ZZZ = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert ZZZ.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert ZZZ.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")
    assert ZZZ.quantize(Decimal("1.0000000000000")) == Decimal("1.000000000000")
    assert ZZZ.quantize(Decimal("1.0000000000001")) == Decimal("1.000000000000")

    # Test with negative values
    assert USD.quantize(Decimal("-1.005")) == Decimal("-1.00")
    assert JPY.quantize(Decimal("-1.5")) == Decimal("-2")
    assert ZZZ.quantize(Decimal("-1.0000000000005")) == Decimal("-1.000000000000")


# LLM-generated content at query #23
#--------------------------

```python
def test_Currency___gt__():
    # Test with different codes
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert usd > eur
    assert not (eur > usd)

    # Test with same code but different names
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars 2", 2, CurrencyType.MONEY)
    assert not (usd1 > usd2)
    assert not (usd2 > usd1)

    # Test with same code and name but different decimals
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert usd1 > usd2
    assert not (usd2 > usd1)

    # Test with same code, name, decimals but different types
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert usd1 > usd2
    assert not (usd2 > usd1)

    # Test with all same attributes
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 > usd2)
    assert not (usd2 > usd1)


# LLM-generated content at query #24
#--------------------------

```python
def test_Currency___repr__():
    # Test with a standard currency
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert repr(usd) == "Currency(code='USD', name='US Dollars', decimals=2, type=CurrencyType.MONEY, quantizer=Decimal('0.01'), hashcache={})".format(hash(usd))

    # Test with a currency with no decimals
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert repr(jpy) == "Currency(code='JPY', name='Japanese Yen', decimals=0, type=CurrencyType.MONEY, quantizer=Decimal('0'), hashcache={})".format(hash(jpy))

    # Test with a cryptocurrency with variable precision
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert repr(zzz) == "Currency(code='ZZZ', name='Some weird currency', decimals=-1, type=CurrencyType.CRYPTO, quantizer=MaxPrecisionQuantizer, hashcache={})".format(hash(zzz))


# LLM-generated content at query #25
#--------------------------

```python
def test_CurrencyRegistry___exit__():
    # Setup
    registry = CurrencyRegistry()
    test_currency1 = Currency.of("TST", "Test Currency 1", 2, CurrencyType.MONEY)
    test_currency2 = Currency.of("TST2", "Test Currency 2", 0, CurrencyType.METAL)

    # Test context manager behavior
    with registry as register:
        register(test_currency1)
        register(test_currency2)

    # Verify post-exit state
    assert len(registry) == 2
    assert registry.codes == ["TST", "TST2"]
    assert registry["TST"] == test_currency1
    assert registry["TST2"] == test_currency2
    assert registry.codenames == [("TST", "Test Currency 1"), ("TST2", "Test Currency 2")]

    # Test that currencies are sorted after exit
    test_currency3 = Currency.of("AAA", "Test Currency 3", 2, CurrencyType.CRYPTO)
    with registry as register:
        register(test_currency3)

    assert registry.codes == ["AAA", "TST", "TST2"]


# LLM-generated content at query #26
#--------------------------

```python
def test_Currency():
    # Test valid currency creation
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.code == "USD"
    assert usd.name == "US Dollars"
    assert usd.decimals == 2
    assert usd.type == CurrencyType.MONEY
    assert usd.quantizer == make_quantizer(2)

    # Test currency with zero decimals
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy.code == "JPY"
    assert jpy.name == "Japanese Yen"
    assert jpy.decimals == 0
    assert jpy.type == CurrencyType.MONEY
    assert jpy.quantizer == ZERO

    # Test currency with variable precision
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert zzz.code == "ZZZ"
    assert zzz.name == "Some weird currency"
    assert zzz.decimals == -1
    assert zzz.type == CurrencyType.CRYPTO
    assert zzz.quantizer == MaxPrecisionQuantizer

    # Test equality
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2
    assert hash(usd1) == hash(usd2)

    # Test inequality
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert usd1 != usdx
    assert hash(usd1) != hash(usdx)

    # Test quantize method
    assert usd.quantize(Decimal("1.005")) == Decimal("1.00")
    assert usd.quantize(Decimal("1.015")) == Decimal("1.02")
    assert jpy.quantize(Decimal("0.5")) == Decimal("0")
    assert jpy.quantize(Decimal("1.5")) == Decimal("2")
    assert zzz.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert zzz.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")

    # Test invalid currency creation
    with pytest.raises(ProgrammingError):
        Currency.of(123, "US Dollars", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("usd", "US Dollars", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD1", "US Dollars", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "  US Dollars", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars  ", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", "2", CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", -2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", 2, "MONEY")


# LLM-generated content at query #27
#--------------------------

```python
def test_CurrencyRegistry___len__():
    registry = CurrencyRegistry()
    assert len(registry) == 0

    with registry as register:
        register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        register(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))

    assert len(registry) == 2


# LLM-generated content at query #28
#--------------------------

```python
def test_Currency___eq__():
    # Test equality with same currency
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2

    # Test inequality with different currency
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 == usdx)

    # Test inequality with non-Currency object
    assert not (usd1 == "USD")

    # Test inequality with different attributes
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert not (usd1 == jpy)


# LLM-generated content at query #29
#--------------------------

```python
def test_Currency___ge__():
    # Test equality (should return True)
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 >= usd2

    # Test greater than (should return True)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert usd >= jpy

    # Test less than (should return False)
    assert jpy >= usd == False

    # Test with different types (should return False)
    crypto = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    assert usd >= crypto == False


# LLM-generated content at query #30
#--------------------------

```python
def test_CurrencyRegistry():
    # Test singleton behavior
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2

    # Test initial state
    assert len(registry1) == 0
    assert registry1.all == []
    assert registry1.codes == []
    assert registry1.codenames == []

    # Test context manager behavior
    with registry1 as register:
        # Test adding a currency
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        register(usd)

        # Test that currency is added
        assert len(registry1) == 1
        assert "USD" in registry1
        assert registry1["USD"] == usd

        # Test adding duplicate currency raises error
        with pytest.raises(ValueError):
            register(usd)

    # Test final state after context
    assert len(registry1) == 1
    assert registry1.all == [usd]
    assert registry1.codes == ["USD"]
    assert registry1.codenames == [("USD", "US Dollar")]

    # Test adding outside context raises error
    with pytest.raises(ProgrammingError):
        registry1._CurrencyRegistry__register(usd)

    # Test has method
    assert registry1.has("USD")
    assert not registry1.has("EUR")

    # Test get method
    assert registry1.get("USD") == usd
    assert registry1.get("EUR") is None
    assert registry1.get("EUR", default=usd) == usd

    # Test lookup error
    with pytest.raises(CurrencyLookupError):
        registry1["EUR"]


# LLM-generated content at query #31
#--------------------------

```python
def test_Currency___setattr__():
    # Test that setting attributes on a frozen Currency instance raises an AttributeError
    currency = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    with pytest.raises(AttributeError):
        currency.code = "EUR"
    with pytest.raises(AttributeError):
        currency.name = "Euros"
    with pytest.raises(AttributeError):
        currency.decimals = 0
    with pytest.raises(AttributeError):
        currency.type = CurrencyType.CRYPTO
    with pytest.raises(AttributeError):
        currency.quantizer = Decimal("0.01")
    with pytest.raises(AttributeError):
        currency.hashcache = 12345


# LLM-generated content at query #32
#--------------------------

```python
def test_Currency___le__():
    # Test equality case
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 <= usd2

    # Test less than case
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert usd <= eur or eur <= usd  # One should be less than the other

    # Test with different attributes
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert (usd <= jpy) == (usd.code <= jpy.code)  # Should follow code comparison

    # Test with different types
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    btc = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    assert (usd <= btc) == (usd.code <= btc.code)  # Should follow code comparison

    # Test with non-Currency object
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd <= "not a currency")


# LLM-generated content at query #33
#--------------------------

```python
def test_CurrencyLookupError():
    # Test initialization and message
    error = CurrencyLookupError("XYZ")
    assert error.code == "XYZ"
    assert str(error) == "Currency identified by code 'XYZ' does not exist"

    # Test inheritance from LookupError
    assert isinstance(error, LookupError)

    # Test with empty string
    empty_error = CurrencyLookupError("")
    assert empty_error.code == ""
    assert str(empty_error) == "Currency identified by code '' does not exist"

    # Test with special characters
    special_error = CurrencyLookupError("A$C")
    assert special_error.code == "A$C"
    assert str(special_error) == "Currency identified by code 'A$C' does not exist"


# LLM-generated content at query #34
#--------------------------

```python
def test_Currency___lt__():
    # Test with different codes
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert (usd < eur) == ("USD" < "EUR")

    # Test with same code but different names
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars 2", 2, CurrencyType.MONEY)
    assert not (usd1 < usd2) and not (usd2 < usd1)  # Should be equal

    # Test with same code and name but different decimals
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert (usd1 < usd2) == (2 < 3)

    # Test with same code, name, decimals but different types
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert (usd1 < usd2) == (CurrencyType.MONEY < CurrencyType.CRYPTO)

    # Test with same code, name, decimals, type but different quantizers (should be equal)
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 < usd2) and not (usd2 < usd1)  # Should be equal

    # Test with non-Currency object
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd < "not a currency")


# LLM-generated content at query #35
#--------------------------

```python
def test_Currency___le__():
    # Test equality (should return True)
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 <= usd2
    assert usd2 <= usd1

    # Test less than (should return True)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert usd <= eur or eur <= usd  # At least one should be True

    # Test greater than (should return False)
    if usd <= eur:
        assert not (eur <= usd)
    else:
        assert not (usd <= eur)

    # Test with different attributes
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 <= usdx) or not (usdx <= usd1)  # Should not be equal

    # Test with different types
    btc = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    assert usd <= btc or btc <= usd  # At least one should be True


# LLM-generated content at query #36
#--------------------------

```python
def test_CurrencyRegistry_has():
    # Setup
    registry = CurrencyRegistry()
    with registry:
        registry(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        registry(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))

    # Test existing currency
    assert registry.has("USD") is True
    assert registry.has("EUR") is True

    # Test non-existing currency
    assert registry.has("XYZ") is False
    assert registry.has("") is False

    # Test case sensitivity
    assert registry.has("usd") is False
    assert registry.has("eur") is False


# LLM-generated content at query #37
#--------------------------

```python
def test_CurrencyRegistry___getitem__():
    # Setup
    registry = CurrencyRegistry()
    with registry as register:
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        register(usd)

    # Test existing currency
    assert registry["USD"] == usd

    # Test non-existing currency
    with pytest.raises(CurrencyLookupError):
        registry["NON-EXISTING"]


# LLM-generated content at query #38
#--------------------------

```python
def test_Currency___le__():
    # Test equality (should return True)
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 <= usd2

    # Test less than (should return True)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert usd <= eur or eur <= usd  # At least one should be True

    # Test greater than (should return False)
    assert not (eur <= usd) or not (usd <= eur)  # At least one should be False

    # Test with different attributes
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 <= usdx)  # Different name, should be False

    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert not (usd1 <= jpy)  # Different decimals, should be False

    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert not (usd1 <= zzz)  # Different type, should be False


# LLM-generated content at query #39
#--------------------------

```python
def test_CurrencyRegistry___enter__():
    # Setup
    registry = CurrencyRegistry()
    test_currency = Currency.of("TEST", "Test Currency", 2, CurrencyType.MONEY)

    # Test that __enter__ returns the register method
    with registry as register:
        assert callable(register)
        register(test_currency)

    # Verify the currency was added
    assert "TEST" in registry
    assert registry["TEST"] == test_currency

    # Test context manager behavior
    with pytest.raises(ProgrammingError):
        registry.__register__(test_currency)  # Should fail outside context

    with pytest.raises(ValueError):
        with registry as register:
            register(test_currency)  # Should fail when adding duplicate


# LLM-generated content at query #40
#--------------------------

```python
def test_CurrencyRegistry():
    # Test singleton behavior
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2

    # Test initial state
    assert len(registry1) == 0
    assert registry1.all == []
    assert registry1.codes == []
    assert registry1.codenames == []

    # Test context manager behavior
    with registry1 as register:
        # Test adding currencies
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)

        register(usd)
        register(eur)

        # Test that currencies are not yet available
        assert len(registry1) == 0
        assert "USD" not in registry1
        assert "EUR" not in registry1

    # Test that currencies are now available after context exit
    assert len(registry1) == 2
    assert "USD" in registry1
    assert "EUR" in registry1
    assert registry1["USD"] == usd
    assert registry1["EUR"] == eur
    assert registry1.all == [eur, usd]  # Sorted by code
    assert registry1.codes == ["EUR", "USD"]
    assert registry1.codenames == [("EUR", "Euro"), ("USD", "US Dollar")]

    # Test duplicate currency registration
    with pytest.raises(ValueError):
        with registry1 as register:
            register(usd)

    # Test adding currencies outside context
    with pytest.raises(ProgrammingError):
        registry1._CurrencyRegistry__register(usd)

    # Test get method
    assert registry1.get("USD") == usd
    assert registry1.get("NONEXISTENT") is None
    assert registry1.get("NONEXISTENT", default=eur) == eur

    # Test lookup error
    with pytest.raises(CurrencyLookupError):
        registry1["NONEXISTENT"]


# LLM-generated content at query #41
#--------------------------

```python
def test_Currency():
    # Test valid currency creation
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.code == "USD"
    assert usd.name == "US Dollars"
    assert usd.decimals == 2
    assert usd.type == CurrencyType.MONEY
    assert usd.quantize(Decimal("1.005")) == Decimal("1.00")
    assert usd.quantize(Decimal("1.015")) == Decimal("1.02")

    # Test JPY with 0 decimals
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy.code == "JPY"
    assert jpy.name == "Japanese Yen"
    assert jpy.decimals == 0
    assert jpy.type == CurrencyType.MONEY
    assert jpy.quantize(Decimal("0.5")) == Decimal("0")
    assert jpy.quantize(Decimal("1.5")) == Decimal("2")

    # Test currency with -1 decimals
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert zzz.code == "ZZZ"
    assert zzz.name == "Some weird currency"
    assert zzz.decimals == -1
    assert zzz.type == CurrencyType.CRYPTO
    assert zzz.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert zzz.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")

    # Test equality and hash
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2
    assert not (usd1 == usdx)
    assert hash(usd1) == hash(usd2)
    assert not (hash(usd1) == hash(usdx))

    # Test invalid currency code
    with pytest.raises(ProgrammingError):
        Currency.of("usd", "US Dollars", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD1", "US Dollars", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("U$D", "US Dollars", 2, CurrencyType.MONEY)

    # Test invalid currency name
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "  US Dollars", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars  ", 2, CurrencyType.MONEY)

    # Test invalid decimals
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", -2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", "2", CurrencyType.MONEY)

    # Test invalid currency type
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", 2, "Money")


# LLM-generated content at query #42
#--------------------------

```python
def test_CurrencyRegistry():
    # Test singleton behavior
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2

    # Test initial state
    assert len(registry1) == 0
    assert registry1.all == []
    assert registry1.codes == []
    assert registry1.codenames == []

    # Test context manager behavior
    with registry1 as register:
        # Test adding currencies in context
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        register(usd)
        assert len(registry1) == 1
        assert "USD" in registry1
        assert registry1["USD"] == usd

        # Test duplicate currency error
        with pytest.raises(ValueError, match="Currency USD is already registered."):
            register(usd)

    # Test final state after context
    assert len(registry1) == 1
    assert registry1.all == [usd]
    assert registry1.codes == ["USD"]
    assert registry1.codenames == [("USD", "US Dollar")]

    # Test adding outside context raises error
    with pytest.raises(ProgrammingError, match="Can not create currencies outside registry context."):
        registry1._CurrencyRegistry__register(usd)

    # Test get methods
    assert registry1.has("USD") is True
    assert registry1.has("XYZ") is False
    assert registry1.get("USD") == usd
    assert registry1.get("XYZ") is None
    assert registry1.get("XYZ", default=usd) == usd

    # Test lookup error
    with pytest.raises(CurrencyLookupError, match="Currency identified by code 'XYZ' does not exist"):
        registry1["XYZ"]


# LLM-generated content at query #43
#--------------------------

```python
def test_Currency___setattr__():
    # Test that setting attributes on a Currency instance raises an AttributeError
    # since Currency is a frozen dataclass
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)

    with pytest.raises(AttributeError):
        USD.code = "EUR"

    with pytest.raises(AttributeError):
        USD.name = "Euros"

    with pytest.raises(AttributeError):
        USD.decimals = 0

    with pytest.raises(AttributeError):
        USD.type = CurrencyType.CRYPTO

    with pytest.raises(AttributeError):
        USD.quantizer = Decimal("0.0001")

    with pytest.raises(AttributeError):
        USD.hashcache = 12345


# LLM-generated content at query #44
#--------------------------

```python
def test_Currency___setattr__():
    # Test that Currency is immutable and __setattr__ raises AttributeError
    currency = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)

    with pytest.raises(AttributeError):
        currency.code = "EUR"

    with pytest.raises(AttributeError):
        currency.name = "Euros"

    with pytest.raises(AttributeError):
        currency.decimals = 0

    with pytest.raises(AttributeError):
        currency.type = CurrencyType.CRYPTO

    with pytest.raises(AttributeError):
        currency.quantizer = Decimal("0.0001")

    with pytest.raises(AttributeError):
        currency.hashcache = 12345


# LLM-generated content at query #45
#--------------------------

```python
def test_CurrencyRegistry___enter__():
    registry = CurrencyRegistry()
    with registry as register:
        assert callable(register)
        assert register.__name__ == "_CurrencyRegistry__register"
    assert not registry._CurrencyRegistry__ctx_open


# LLM-generated content at query #46
#--------------------------

```python
def test_CurrencyRegistry():
    # Test singleton behavior
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2

    # Test initial state
    assert len(registry1) == 0
    assert registry1.all == []
    assert registry1.codes == []
    assert registry1.codenames == []

    # Test context manager
    with registry1 as register:
        # Test adding a currency
        currency = Currency.of("TEST", "Test Currency", 2, CurrencyType.MONEY)
        register(currency)

    # Test after context
    assert len(registry1) == 1
    assert "TEST" in registry1
    assert registry1["TEST"] == currency
    assert registry1.all == [currency]
    assert registry1.codes == ["TEST"]
    assert registry1.codenames == [("TEST", "Test Currency")]

    # Test adding duplicate currency
    with pytest.raises(ValueError):
        with registry1 as register:
            register(currency)

    # Test adding outside context
    with pytest.raises(ProgrammingError):
        registry1._CurrencyRegistry__register(currency)

    # Test has method
    assert registry1.has("TEST")
    assert not registry1.has("NONEXISTENT")

    # Test get method
    assert registry1.get("TEST") == currency
    assert registry1.get("NONEXISTENT") is None
    assert registry1.get("NONEXISTENT", default=currency) == currency

    # Test lookup error
    with pytest.raises(CurrencyLookupError):
        registry1["NONEXISTENT"]


# LLM-generated content at query #47
#--------------------------

```python
def test_Currency___setattr__():
    # Create a currency instance
    currency = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)

    # Attempt to set an attribute and verify it raises an AttributeError
    with pytest.raises(AttributeError):
        currency.code = "EUR"

    with pytest.raises(AttributeError):
        currency.name = "Euros"

    with pytest.raises(AttributeError):
        currency.decimals = 3

    with pytest.raises(AttributeError):
        currency.type = CurrencyType.CRYPTO

    with pytest.raises(AttributeError):
        currency.quantizer = Decimal("0.001")

    with pytest.raises(AttributeError):
        currency.hashcache = 12345


# LLM-generated content at query #48
#--------------------------

```python
def test_Currency():
    # Test valid currency creation
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.code == "USD"
    assert usd.name == "US Dollars"
    assert usd.decimals == 2
    assert usd.type == CurrencyType.MONEY
    assert usd.quantize(Decimal("1.005")) == Decimal("1.00")
    assert usd.quantize(Decimal("1.015")) == Decimal("1.02")

    # Test currency with zero decimals
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy.quantize(Decimal("0.5")) == Decimal("0")
    assert jpy.quantize(Decimal("1.5")) == Decimal("2")

    # Test currency with variable precision
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert zzz.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert zzz.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")

    # Test equality and hash
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2
    assert usd1 != usdx
    assert hash(usd1) == hash(usd2)
    assert hash(usd1) != hash(usdx)

    # Test invalid currency code
    with pytest.raises(ProgrammingError):
        Currency.of("123", "Invalid", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("abc", "Invalid", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("Abc", "Invalid", 2, CurrencyType.MONEY)

    # Test invalid currency name
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", " ", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", " Invalid", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "Invalid ", 2, CurrencyType.MONEY)

    # Test invalid decimals
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", -2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", "2", CurrencyType.MONEY)

    # Test invalid currency type
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", 2, "MONEY")


# LLM-generated content at query #49
#--------------------------

```python
def test_Currency___setattr__():
    # Test that Currency is immutable and __setattr__ raises an error
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)

    with pytest.raises(AttributeError):
        usd.code = "EUR"

    with pytest.raises(AttributeError):
        usd.name = "Euros"

    with pytest.raises(AttributeError):
        usd.decimals = 0

    with pytest.raises(AttributeError):
        usd.type = CurrencyType.CRYPTO

    with pytest.raises(AttributeError):
        usd.quantizer = Decimal("0.01")

    with pytest.raises(AttributeError):
        usd.hashcache = 12345


# LLM-generated content at query #50
#--------------------------

```python
def test_CurrencyRegistry___len__():
    registry = CurrencyRegistry()
    assert len(registry) == 0

    with registry as register:
        register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        register(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))
        register(Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY))

    assert len(registry) == 3


# LLM-generated content at query #51
#--------------------------

```python
def test_CurrencyRegistry():
    # Test singleton behavior
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2

    # Test initial state
    assert len(registry1) == 0
    assert registry1.all == []
    assert registry1.codes == []
    assert registry1.codenames == []

    # Test context manager
    with registry1 as register:
        currency = Currency.of("TEST", "Test Currency", 2, CurrencyType.MONEY)
        register(currency)

    assert len(registry1) == 1
    assert registry1["TEST"] == currency
    assert registry1.all == [currency]
    assert registry1.codes == ["TEST"]
    assert registry1.codenames == [("TEST", "Test Currency")]

    # Test duplicate registration
    with pytest.raises(ValueError):
        with registry1 as register:
            register(currency)

    # Test outside context registration
    with pytest.raises(ProgrammingError):
        registry1._CurrencyRegistry__register(currency)

    # Test has and get methods
    assert registry1.has("TEST") is True
    assert registry1.has("NONEXISTENT") is False
    assert registry1.get("TEST") == currency
    assert registry1.get("NONEXISTENT") is None
    assert registry1.get("NONEXISTENT", default=currency) == currency

    # Test lookup error
    with pytest.raises(CurrencyLookupError):
        registry1["NONEXISTENT"]


# LLM-generated content at query #52
#--------------------------

```python
def test_Currency___hash__():
    # Test basic hash functionality
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert hash(usd1) == hash(usd2)

    # Test hash with different attributes
    usd_different_name = Currency.of("USD", "Different Name", 2, CurrencyType.MONEY)
    assert hash(usd1) != hash(usd_different_name)

    # Test hash with different decimals
    usd_different_decimals = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert hash(usd1) != hash(usd_different_decimals)

    # Test hash with different currency type
    usd_different_type = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert hash(usd1) != hash(usd_different_type)

    # Test hash with different code
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert hash(usd1) != hash(eur)

    # Test hash consistency
    assert hash(usd1) == hash(usd1)


# LLM-generated content at query #53
#--------------------------

```python
def test_Currency___repr__():
    # Test with USD
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert repr(usd) == "Currency(code='USD', name='US Dollars', decimals=2, type=CurrencyType.MONEY, quantizer=Decimal('0.01'), hashcache={})".format(hash(("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal('0.01'))))

    # Test with JPY
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert repr(jpy) == "Currency(code='JPY', name='Japanese Yen', decimals=0, type=CurrencyType.MONEY, quantizer=Decimal('0'), hashcache={})".format(hash(("JPY", "Japanese Yen", 0, CurrencyType.MONEY, Decimal('0'))))

    # Test with a weird currency
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert repr(zzz) == "Currency(code='ZZZ', name='Some weird currency', decimals=-1, type=CurrencyType.CRYPTO, quantizer=MaxPrecisionQuantizer, hashcache={})".format(hash(("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO, MaxPrecisionQuantizer)))


# LLM-generated content at query #54
#--------------------------

```python
def test_CurrencyRegistry___getitem__():
    # Setup
    registry = CurrencyRegistry()
    with registry as register:
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
        register(usd)
        register(eur)

    # Test existing currency
    assert registry["USD"] == usd
    assert registry["EUR"] == eur

    # Test non-existing currency
    with pytest.raises(CurrencyLookupError) as excinfo:
        _ = registry["NON-EXISTING"]
    assert str(excinfo.value) == "Currency identified by code 'NON-EXISTING' does not exist"


# LLM-generated content at query #55
#--------------------------

```python
def test_CurrencyRegistry___getitem__():
    # Setup
    registry = CurrencyRegistry()
    with registry as register:
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
        register(usd)
        register(eur)

    # Test existing currency
    assert registry["USD"] == usd
    assert registry["EUR"] == eur

    # Test non-existing currency
    with pytest.raises(CurrencyLookupError) as excinfo:
        _ = registry["NON-EXISTING"]
    assert str(excinfo.value) == "Currency identified by code 'NON-EXISTING' does not exist"


# LLM-generated content at query #56
#--------------------------

```python
def test_Currency___gt__():
    # Test with same currency
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 > usd2)
    assert not (usd2 > usd1)

    # Test with different currencies
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert (eur > usd) or (usd > eur)  # One should be greater than the other

    # Test with different decimals
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert (jpy > usd) or (usd > jpy)  # One should be greater than the other

    # Test with different types
    btc = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    assert (btc > usd) or (usd > btc)  # One should be greater than the other


# LLM-generated content at query #57
#--------------------------

```python
def test_Currency___setattr__():
    # Create a Currency instance
    currency = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)

    # Attempt to set an attribute and verify it raises an AttributeError
    with pytest.raises(AttributeError):
        currency.code = "EUR"

    with pytest.raises(AttributeError):
        currency.name = "Euros"

    with pytest.raises(AttributeError):
        currency.decimals = 0

    with pytest.raises(AttributeError):
        currency.type = CurrencyType.CRYPTO

    with pytest.raises(AttributeError):
        currency.quantizer = Decimal("0.0001")

    with pytest.raises(AttributeError):
        currency.hashcache = 12345


# LLM-generated content at query #58
#--------------------------

```python
def test_CurrencyRegistry___getitem__():
    # Test getting an existing currency
    usd = CurrencyRegistry()["USD"]
    assert usd.code == "USD"
    assert usd.name == "US Dollar"
    assert usd.type == CurrencyType.MONEY

    # Test getting a non-existing currency raises CurrencyLookupError
    with pytest.raises(CurrencyLookupError) as excinfo:
        CurrencyRegistry()["NON-EXISTING"]
    assert str(excinfo.value) == "Currency identified by code 'NON-EXISTING' does not exist"


# LLM-generated content at query #59
#--------------------------

```python
def test_Currency___delattr__():
    # Create a currency instance
    currency = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)

    # Attempt to delete an attribute and expect an AttributeError since Currency is frozen
    with pytest.raises(AttributeError):
        del currency.code

    with pytest.raises(AttributeError):
        del currency.name

    with pytest.raises(AttributeError):
        del currency.decimals

    with pytest.raises(AttributeError):
        del currency.type

    with pytest.raises(AttributeError):
        del currency.quantizer

    with pytest.raises(AttributeError):
        del currency.hashcache


# LLM-generated content at query #60
#--------------------------

```python
def test_Currency():
    # Test valid currency creation
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.code == "USD"
    assert usd.name == "US Dollars"
    assert usd.decimals == 2
    assert usd.type == CurrencyType.MONEY
    assert usd.quantize(Decimal("1.005")) == Decimal("1.00")
    assert usd.quantize(Decimal("1.015")) == Decimal("1.02")

    # Test currency with zero decimals
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy.code == "JPY"
    assert jpy.name == "Japanese Yen"
    assert jpy.decimals == 0
    assert jpy.type == CurrencyType.MONEY
    assert jpy.quantize(Decimal("0.5")) == Decimal("0")
    assert jpy.quantize(Decimal("1.5")) == Decimal("2")

    # Test currency with negative decimals
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert zzz.code == "ZZZ"
    assert zzz.name == "Some weird currency"
    assert zzz.decimals == -1
    assert zzz.type == CurrencyType.CRYPTO
    assert zzz.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert zzz.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")

    # Test equality and hash
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2
    assert not (usd1 == usdx)
    assert hash(usd1) == hash(usd2)
    assert not (hash(usd1) == hash(usdx))

    # Test invalid currency code
    with pytest.raises(ProgrammingError):
        Currency.of("usd", "US Dollars", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD1", "US Dollars", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("U$D", "US Dollars", 2, CurrencyType.MONEY)

    # Test invalid currency name
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "  US Dollars", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars  ", 2, CurrencyType.MONEY)

    # Test invalid decimals
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", -2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", "2", CurrencyType.MONEY)

    # Test invalid currency type
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", 2, "MONEY")


# LLM-generated content at query #61
#--------------------------

```python
def test_CurrencyRegistry___getitem__():
    # Setup
    registry = CurrencyRegistry()
    with registry as register:
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
        register(usd)
        register(eur)

    # Test existing currency
    assert registry["USD"] == usd
    assert registry["EUR"] == eur

    # Test non-existing currency
    with pytest.raises(CurrencyLookupError) as excinfo:
        _ = registry["NON-EXISTING"]
    assert str(excinfo.value) == "Currency identified by code 'NON-EXISTING' does not exist"


# LLM-generated content at query #62
#--------------------------

```python
def test_Currency___gt__():
    # Test with different codes
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert usd > eur
    assert not (eur > usd)

    # Test with same code but different names
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 > usd2)

    # Test with different decimals
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert jpy > usd
    assert not (usd > jpy)

    # Test with different types
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    btc = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    assert usd > btc
    assert not (btc > usd)

    # Test with different quantizers
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert zzz > usd
    assert not (usd > zzz)


# LLM-generated content at query #63
#--------------------------

```python
def test_CurrencyRegistry___enter__():
    # Test that __enter__ returns the __register method
    registry = CurrencyRegistry()
    with registry as register:
        assert callable(register)
        assert register.__name__ == "_CurrencyRegistry__register"

    # Test that __enter__ opens the context
    assert registry._CurrencyRegistry__ctx_open is False  # Context should be closed after exit

    # Test that __enter__ can be called multiple times
    with registry as register:
        assert registry._CurrencyRegistry__ctx_open is True
    assert registry._CurrencyRegistry__ctx_open is False

    # Test that __enter__ works with nested contexts
    with registry as register1:
        assert registry._CurrencyRegistry__ctx_open is True
        with registry as register2:
            assert registry._CurrencyRegistry__ctx_open is True
            assert register1 == register2
        assert registry._CurrencyRegistry__ctx_open is True
    assert registry._CurrencyRegistry__ctx_open is False


# LLM-generated content at query #64
#--------------------------

```python
def test_Currency___setattr__():
    # Test that setting attributes on a frozen Currency instance raises an AttributeError
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)

    with pytest.raises(AttributeError):
        usd.code = "EUR"

    with pytest.raises(AttributeError):
        usd.name = "Euros"

    with pytest.raises(AttributeError):
        usd.decimals = 0

    with pytest.raises(AttributeError):
        usd.type = CurrencyType.CRYPTO

    with pytest.raises(AttributeError):
        usd.quantizer = Decimal("0.0001")

    with pytest.raises(AttributeError):
        usd.hashcache = 12345


# LLM-generated content at query #65
#--------------------------

```python
def test_CurrencyRegistry___contains__():
    # Setup
    registry = CurrencyRegistry()
    with registry as register:
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
        register(usd)
        register(eur)

    # Test existing currency
    assert "USD" in registry
    assert "EUR" in registry

    # Test non-existing currency
    assert "XYZ" not in registry
    assert "NONEXISTENT" not in registry


# LLM-generated content at query #66
#--------------------------

```python
def test_CurrencyRegistry___len__():
    # Create a new CurrencyRegistry instance
    registry = CurrencyRegistry()

    # Initially, the registry should be empty
    assert len(registry) == 0

    # Add some currencies to the registry
    with registry as register:
        register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        register(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))
        register(Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY))

    # Now, the registry should have 3 currencies
    assert len(registry) == 3


# LLM-generated content at query #67
#--------------------------

```python
def test_Currency_quantize():
    # Test with USD (2 decimals)
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert USD.quantize(Decimal("1.005")) == Decimal("1.00")
    assert USD.quantize(Decimal("1.015")) == Decimal("1.02")
    assert USD.quantize(Decimal("1.00")) == Decimal("1.00")
    assert USD.quantize(Decimal("1.001")) == Decimal("1.00")
    assert USD.quantize(Decimal("1.009")) == Decimal("1.01")

    # Test with JPY (0 decimals)
    JPY = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert JPY.quantize(Decimal("0.5")) == Decimal("0")
    assert JPY.quantize(Decimal("1.5")) == Decimal("2")
    assert JPY.quantize(Decimal("1.0")) == Decimal("1")
    assert JPY.quantize(Decimal("1.4")) == Decimal("1")
    assert JPY.quantize(Decimal("1.6")) == Decimal("2")

    # Test with a currency with no fixed precision (e.g., -1 decimals)
    ZZZ = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert ZZZ.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert ZZZ.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")
    assert ZZZ.quantize(Decimal("1.000000000000")) == Decimal("1.000000000000")
    assert ZZZ.quantize(Decimal("1.0000000000001")) == Decimal("1.000000000000")
    assert ZZZ.quantize(Decimal("1.0000000000009")) == Decimal("1.000000000001")

    # Test with a currency with 3 decimals
    TND = Currency.of("TND", "Tunisian Dinar", 3, CurrencyType.MONEY)
    assert TND.quantize(Decimal("1.0005")) == Decimal("1.000")
    assert TND.quantize(Decimal("1.0015")) == Decimal("1.002")
    assert TND.quantize(Decimal("1.000")) == Decimal("1.000")
    assert TND.quantize(Decimal("1.0001")) == Decimal("1.000")
    assert TND.quantize(Decimal("1.0009")) == Decimal("1.001")


# LLM-generated content at query #68
#--------------------------

```python
def test_CurrencyRegistry___len__():
    # Create a new instance of CurrencyRegistry
    registry = CurrencyRegistry()

    # Check that the initial length is 0
    assert len(registry) == 0

    # Enter the registry population context
    with registry as register:
        # Add a currency
        register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))

    # Check that the length is now 1
    assert len(registry) == 1

    # Enter the registry population context again
    with registry as register:
        # Add another currency
        register(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))

    # Check that the length is now 2
    assert len(registry) == 2


# LLM-generated content at query #69
#--------------------------

```python
def test_CurrencyRegistry___len__():
    registry = CurrencyRegistry()
    assert len(registry) == 0

    with registry as register:
        register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        register(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))

    assert len(registry) == 2


# LLM-generated content at query #70
#--------------------------

```python
def test_CurrencyRegistry___new__():
    # Test that CurrencyRegistry is a singleton
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2

    # Test that the singleton instance is created only once
    assert CurrencyRegistry._CurrencyRegistry__instance is not None
    assert CurrencyRegistry._CurrencyRegistry__instance is registry1
    assert CurrencyRegistry._CurrencyRegistry__instance is registry2


# LLM-generated content at query #71
#--------------------------

```python
def test_CurrencyRegistry___exit__():
    # Test normal exit
    registry = CurrencyRegistry()
    with registry as register:
        register(Currency.of("TEST1", "Test Currency 1", 2, CurrencyType.MONEY))
        register(Currency.of("TEST2", "Test Currency 2", 0, CurrencyType.MONEY))
    assert "TEST1" in registry
    assert "TEST2" in registry
    assert registry.codes == ["TEST1", "TEST2"]
    assert len(registry) == 2

    # Test that registry is sorted after exit
    with registry as register:
        register(Currency.of("AAA", "Test Currency AAA", 2, CurrencyType.MONEY))
        register(Currency.of("ZZZ", "Test Currency ZZZ", 0, CurrencyType.MONEY))
    assert registry.codes == ["AAA", "TEST1", "TEST2", "ZZZ"]

    # Test that context is closed after exit
    with registry as register:
        pass
    try:
        registry._CurrencyRegistry__register(Currency.of("FAIL", "Should Fail", 2, CurrencyType.MONEY))
        assert False, "Should have raised ProgrammingError"
    except ProgrammingError:
        pass

    # Test with exception during context
    try:
        with registry as register:
            register(Currency.of("EXC", "Exception Test", 2, CurrencyType.MONEY))
            raise ValueError("Test exception")
    except ValueError:
        pass
    assert "EXC" not in registry  # Should not be registered due to exception


# LLM-generated content at query #72
#--------------------------

```python
def test_CurrencyRegistry___getitem__():
    # Setup
    registry = CurrencyRegistry()
    with registry as register:
        register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        register(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))

    # Test existing currency
    usd = registry["USD"]
    assert usd.code == "USD"
    assert usd.name == "US Dollar"
    assert usd.type == CurrencyType.MONEY

    # Test non-existing currency
    with pytest.raises(CurrencyLookupError):
        registry["XYZ"]


# LLM-generated content at query #73
#--------------------------

```python
def test_Currency___gt__():
    # Test with same currency
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 > usd2)

    # Test with different currency codes
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert (eur > usd) or (usd > eur)  # One must be greater than the other

    # Test with different names
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars 2", 2, CurrencyType.MONEY)
    assert (usd1 > usd2) or (usd2 > usd1)  # One must be greater than the other

    # Test with different decimals
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert (usd > jpy) or (jpy > usd)  # One must be greater than the other

    # Test with different types
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    btc = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    assert (usd > btc) or (btc > usd)  # One must be greater than the other

    # Test with different quantizers (indirectly through decimals)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    weird = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert (usd > weird) or (weird > usd)  # One must be greater than the other


# LLM-generated content at query #74
#--------------------------

```python
def test_CurrencyRegistry___len__():
    # Test the length of the currency registry
    assert len(Currencies) == len(Currencies.all)
    assert len(Currencies) == len(Currencies.codes)
    assert len(Currencies) == len(Currencies.codenames)


# LLM-generated content at query #75
#--------------------------

```python
def test_CurrencyRegistry___len__():
    # Create a new instance of CurrencyRegistry
    registry = CurrencyRegistry()

    # Initially, the registry should be empty
    assert len(registry) == 0

    # Add a currency to the registry
    with registry as register:
        register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))

    # Now, the registry should have one currency
    assert len(registry) == 1

    # Add another currency to the registry
    with registry as register:
        register(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))

    # Now, the registry should have two currencies
    assert len(registry) == 2


# LLM-generated content at query #76
#--------------------------

```python
def test_Currency___delattr__():
    # Create a currency instance
    currency = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)

    # Attempt to delete an attribute and verify it raises an AttributeError
    with pytest.raises(AttributeError):
        del currency.code

    with pytest.raises(AttributeError):
        del currency.name

    with pytest.raises(AttributeError):
        del currency.decimals

    with pytest.raises(AttributeError):
        del currency.type

    with pytest.raises(AttributeError):
        del currency.quantizer

    with pytest.raises(AttributeError):
        del currency.hashcache


# LLM-generated content at query #77
#--------------------------

```python
def test_CurrencyRegistry___new__():
    # Test that CurrencyRegistry is a singleton
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2

    # Test that the singleton instance is created only once
    assert CurrencyRegistry._CurrencyRegistry__instance is not None
    assert CurrencyRegistry._CurrencyRegistry__instance is registry1
    assert CurrencyRegistry._CurrencyRegistry__instance is registry2


# LLM-generated content at query #78
#--------------------------

```python
def test_Currency___setattr__():
    # Test that Currency is immutable and __setattr__ raises AttributeError
    currency = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)

    with pytest.raises(AttributeError):
        currency.code = "EUR"

    with pytest.raises(AttributeError):
        currency.name = "Euros"

    with pytest.raises(AttributeError):
        currency.decimals = 0

    with pytest.raises(AttributeError):
        currency.type = CurrencyType.CRYPTO

    with pytest.raises(AttributeError):
        currency.quantizer = Decimal("0.0001")

    with pytest.raises(AttributeError):
        currency.hashcache = 12345


# LLM-generated content at query #79
#--------------------------

```python
def test_Currency___eq__():
    # Test equality with same currency
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2

    # Test inequality with different currency code
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    assert not (usd == eur)

    # Test inequality with different currency name
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 == usd2)

    # Test inequality with different decimals
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 0, CurrencyType.MONEY)
    assert not (usd1 == usd2)

    # Test inequality with different currency type
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert not (usd1 == usd2)

    # Test inequality with non-Currency object
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd == "USD")
    assert not (usd == 123)
    assert not (usd == None)


# LLM-generated content at query #80
#--------------------------

```python
def test_CurrencyRegistry():
    # Test singleton behavior
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2

    # Test initial state
    assert len(registry1) == 0
    assert registry1.all == []
    assert registry1.codes == []
    assert registry1.codenames == []

    # Test context manager
    with registry1 as register:
        # Test adding currencies in context
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        register(usd)
        assert len(registry1) == 1
        assert "USD" in registry1
        assert registry1["USD"] == usd

        # Test adding duplicate currency raises error
        with pytest.raises(ValueError, match="Currency USD is already registered."):
            register(usd)

    # Test final state after context
    assert len(registry1) == 1
    assert registry1.all == [usd]
    assert registry1.codes == ["USD"]
    assert registry1.codenames == [("USD", "US Dollar")]

    # Test adding outside context raises error
    with pytest.raises(ProgrammingError, match="Can not create currencies outside registry context."):
        registry1._CurrencyRegistry__register(usd)

    # Test lookup methods
    assert registry1.has("USD") is True
    assert registry1.has("XYZ") is False
    assert registry1.get("USD") == usd
    assert registry1.get("XYZ") is None
    assert registry1.get("XYZ", default=usd) == usd

    # Test lookup error
    with pytest.raises(CurrencyLookupError, match="Currency identified by code 'XYZ' does not exist"):
        registry1["XYZ"]


# LLM-generated content at query #81
#--------------------------

```python
def test_Currency():
    # Test valid currency creation
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.code == "USD"
    assert usd.name == "US Dollars"
    assert usd.decimals == 2
    assert usd.type == CurrencyType.MONEY
    assert usd.quantize(Decimal("1.005")) == Decimal("1.00")
    assert usd.quantize(Decimal("1.015")) == Decimal("1.02")

    # Test currency with zero decimals
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy.code == "JPY"
    assert jpy.name == "Japanese Yen"
    assert jpy.decimals == 0
    assert jpy.type == CurrencyType.MONEY
    assert jpy.quantize(Decimal("0.5")) == Decimal("0")
    assert jpy.quantize(Decimal("1.5")) == Decimal("2")

    # Test currency with variable precision
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert zzz.code == "ZZZ"
    assert zzz.name == "Some weird currency"
    assert zzz.decimals == -1
    assert zzz.type == CurrencyType.CRYPTO
    assert zzz.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert zzz.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")

    # Test equality and hash
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2
    assert usd1 != usdx
    assert hash(usd1) == hash(usd2)
    assert hash(usd1) != hash(usdx)

    # Test invalid currency code
    with pytest.raises(ProgrammingError):
        Currency.of(123, "US Dollars", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("usd", "US Dollars", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD1", "US Dollars", 2, CurrencyType.MONEY)

    # Test invalid currency name
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", " US Dollars", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars ", 2, CurrencyType.MONEY)

    # Test invalid decimals
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", -2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", "2", CurrencyType.MONEY)

    # Test invalid currency type
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", 2, "MONEY")


# LLM-generated content at query #82
#--------------------------

```python
def test_CurrencyRegistry___contains__():
    registry = CurrencyRegistry()
    with registry as register:
        register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        register(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))

    assert "USD" in registry
    assert "EUR" in registry
    assert "JPY" not in registry


# LLM-generated content at query #83
#--------------------------

```python
def test_CurrencyRegistry___exit__():
    # Setup
    registry = CurrencyRegistry()
    test_currency1 = Currency.of("TEST1", "Test Currency 1", 2, CurrencyType.MONEY)
    test_currency2 = Currency.of("TEST2", "Test Currency 2", 0, CurrencyType.CRYPTO)
    test_currency3 = Currency.of("TEST3", "Test Currency 3", -1, CurrencyType.ALTERNATIVE)

    # Test context manager behavior
    with registry as register:
        register(test_currency1)
        register(test_currency2)
        register(test_currency3)

    # Verify post-exit state
    assert len(registry) == 3
    assert registry.codes == ["TEST1", "TEST2", "TEST3"]
    assert registry.codenames == [
        ("TEST1", "Test Currency 1"),
        ("TEST2", "Test Currency 2"),
        ("TEST3", "Test Currency 3")
    ]
    assert registry["TEST1"] == test_currency1
    assert registry["TEST2"] == test_currency2
    assert registry["TEST3"] == test_currency3

    # Verify sorting occurred
    unsorted_currency = Currency.of("AAA", "Unsorted Currency", 2, CurrencyType.MONEY)
    with registry as register:
        register(unsorted_currency)

    assert registry.codes == ["AAA", "TEST1", "TEST2", "TEST3"]

    # Test that context is properly closed
    try:
        registry._CurrencyRegistry__register(test_currency1)
        assert False, "Should have raised ProgrammingError"
    except ProgrammingError:
        pass


# LLM-generated content at query #84
#--------------------------

```python
def test_Currency___hash__():
    # Test basic hash functionality
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert hash(usd1) == hash(usd2)

    # Test different currencies have different hashes
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert hash(usd1) != hash(jpy)

    # Test hash is consistent with equality
    assert usd1 == usd2
    assert hash(usd1) == hash(usd2)

    # Test hash is different for currencies with different attributes
    usd_different_name = Currency.of("USD", "Different Name", 2, CurrencyType.MONEY)
    assert hash(usd1) != hash(usd_different_name)

    usd_different_decimals = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert hash(usd1) != hash(usd_different_decimals)

    usd_different_type = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert hash(usd1) != hash(usd_different_type)

    # Test hash is computed from all attributes
    weird = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert hash(weird) != hash(usd1)


# LLM-generated content at query #85
#--------------------------

```python
def test_CurrencyRegistry___len__():
    registry = CurrencyRegistry()
    assert len(registry) == 0

    with registry as register:
        register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        register(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))

    assert len(registry) == 2


# LLM-generated content at query #86
#--------------------------

```python
def test_CurrencyRegistry___new__():
    # Test that CurrencyRegistry is a singleton
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2

    # Test that the singleton instance is created only once
    assert CurrencyRegistry._CurrencyRegistry__instance is not None
    assert CurrencyRegistry._CurrencyRegistry__instance is registry1
    assert CurrencyRegistry._CurrencyRegistry__instance is registry2


# LLM-generated content at query #87
#--------------------------

```python
def test_CurrencyRegistry___enter__():
    # Test that __enter__ returns the __register method
    registry = CurrencyRegistry()
    with registry as register:
        assert callable(register)
        assert register.__name__ == "_CurrencyRegistry__register"

    # Test that context is properly managed
    assert not registry._CurrencyRegistry__ctx_open

    # Test that currencies can be added within the context
    test_currency = Currency.of("TEST", "Test Currency", 2, CurrencyType.MONEY)
    with registry as register:
        register(test_currency)
        assert "TEST" in registry
        assert registry["TEST"] == test_currency

    # Test that context is closed after exit
    assert not registry._CurrencyRegistry__ctx_open


# LLM-generated content at query #88
#--------------------------

```python
def test_Currency():
    # Test valid currency creation
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.code == "USD"
    assert usd.name == "US Dollars"
    assert usd.decimals == 2
    assert usd.type == CurrencyType.MONEY
    assert usd.quantizer == make_quantizer(2)

    # Test currency with zero decimals
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy.code == "JPY"
    assert jpy.name == "Japanese Yen"
    assert jpy.decimals == 0
    assert jpy.type == CurrencyType.MONEY
    assert jpy.quantizer == ZERO

    # Test currency with negative decimals
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert zzz.code == "ZZZ"
    assert zzz.name == "Some weird currency"
    assert zzz.decimals == -1
    assert zzz.type == CurrencyType.CRYPTO
    assert zzz.quantizer == MaxPrecisionQuantizer

    # Test equality
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2
    assert hash(usd1) == hash(usd2)

    # Test inequality
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert usd1 != usdx
    assert hash(usd1) != hash(usdx)

    # Test quantize method
    assert usd.quantize(Decimal("1.005")) == Decimal("1.00")
    assert usd.quantize(Decimal("1.015")) == Decimal("1.02")
    assert jpy.quantize(Decimal("0.5")) == Decimal("0")
    assert jpy.quantize(Decimal("1.5")) == Decimal("2")
    assert zzz.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert zzz.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")

    # Test invalid currency code
    with pytest.raises(ProgrammingError):
        Currency.of("123", "Invalid Code", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("abc", "Invalid Code", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("Abc", "Invalid Code", 2, CurrencyType.MONEY)

    # Test invalid currency name
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "  Invalid Name  ", 2, CurrencyType.MONEY)

    # Test invalid decimals
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", -2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", "2", CurrencyType.MONEY)

    # Test invalid currency type
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", 2, "MONEY")


# LLM-generated content at query #89
#--------------------------

```python
def test_Currency___gt__():
    # Test greater than with same currency type and different codes
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert usd > eur

    # Test greater than with different currency types
    gold = Currency.of("XAU", "Gold", 2, CurrencyType.METAL)
    btc = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    assert gold > btc

    # Test greater than with same code but different attributes
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 0, CurrencyType.MONEY)
    assert usd1 > usd2

    # Test greater than with same attributes (should be False)
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 > usd2)

    # Test greater than with different decimals
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert jpy > usd


# LLM-generated content at query #90
#--------------------------

```python
def test_Currency___hash__():
    # Test with same currency instances
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert hash(usd1) == hash(usd2)

    # Test with different currency instances
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert hash(usd1) != hash(usdx)

    # Test with different currency codes
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert hash(usd1) != hash(jpy)

    # Test with different currency types
    btc = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    assert hash(usd1) != hash(btc)

    # Test with different decimals
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert hash(usd1) != hash(zzz)


# LLM-generated content at query #91
#--------------------------

```python
def test_Currency___setattr__():
    # Test that setting attributes on a frozen Currency instance raises an AttributeError
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)

    with pytest.raises(AttributeError):
        usd.code = "EUR"

    with pytest.raises(AttributeError):
        usd.name = "Euro"

    with pytest.raises(AttributeError):
        usd.decimals = 0

    with pytest.raises(AttributeError):
        usd.type = CurrencyType.CRYPTO

    with pytest.raises(AttributeError):
        usd.quantizer = Decimal("0.0001")

    with pytest.raises(AttributeError):
        usd.hashcache = 12345


# LLM-generated content at query #92
#--------------------------

```python
def test_Currency___delattr__():
    currency = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    with pytest.raises(AttributeError):
        del currency.code
    with pytest.raises(AttributeError):
        del currency.name
    with pytest.raises(AttributeError):
        del currency.decimals
    with pytest.raises(AttributeError):
        del currency.type
    with pytest.raises(AttributeError):
        del currency.quantizer
    with pytest.raises(AttributeError):
        del currency.hashcache


# LLM-generated content at query #93
#--------------------------

```python
def test_Currency_quantize():
    # Test USD with 2 decimals
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert USD.quantize(Decimal("1.005")) == Decimal("1.00")
    assert USD.quantize(Decimal("1.015")) == Decimal("1.02")
    assert USD.quantize(Decimal("1.234567")) == Decimal("1.23")

    # Test JPY with 0 decimals
    JPY = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert JPY.quantize(Decimal("0.5")) == Decimal("0")
    assert JPY.quantize(Decimal("1.5")) == Decimal("2")
    assert JPY.quantize(Decimal("123.456")) == Decimal("123")

    # Test ZZZ with -1 decimals (max precision)
    ZZZ = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert ZZZ.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert ZZZ.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")
    assert ZZZ.quantize(Decimal("1.12345678901234567890123456789")) == Decimal("1.12345678901234567890123456789")

    # Test negative values
    assert USD.quantize(Decimal("-1.005")) == Decimal("-1.00")
    assert JPY.quantize(Decimal("-1.5")) == Decimal("-2")
    assert ZZZ.quantize(Decimal("-1.0000000000005")) == Decimal("-1.000000000000")

    # Test zero
    assert USD.quantize(Decimal("0")) == Decimal("0.00")
    assert JPY.quantize(Decimal("0")) == Decimal("0")
    assert ZZZ.quantize(Decimal("0")) == Decimal("0")


# LLM-generated content at query #94
#--------------------------

```python
def test_CurrencyRegistry_get():
    # Setup
    registry = CurrencyRegistry()
    with registry as register:
        register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        register(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))

    # Test existing currency with no default
    assert registry.get("USD").code == "USD"
    assert registry.get("USD").name == "US Dollar"

    # Test non-existing currency with no default
    assert registry.get("XYZ") is None

    # Test non-existing currency with default
    default_currency = Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY)
    assert registry.get("XYZ", default=default_currency) == default_currency

    # Test existing currency with default (should return the existing one, not default)
    assert registry.get("EUR", default=default_currency).code == "EUR"


# LLM-generated content at query #95
#--------------------------

```python
def test_CurrencyRegistry_get():
    # Setup: Create a test currency and register it
    test_currency = Currency.of("TEST", "Test Currency", 2, CurrencyType.MONEY)
    with Currencies as register:
        register(test_currency)

    # Test 1: Get existing currency
    result = Currencies.get("TEST")
    assert result == test_currency

    # Test 2: Get non-existing currency without default
    result = Currencies.get("NONEXISTENT")
    assert result is None

    # Test 3: Get non-existing currency with default
    default_currency = Currency.of("DEFAULT", "Default Currency", 0, CurrencyType.MONEY)
    result = Currencies.get("NONEXISTENT", default=default_currency)
    assert result == default_currency

    # Test 4: Get existing currency with default (should return the existing one, not default)
    result = Currencies.get("TEST", default=default_currency)
    assert result == test_currency


# LLM-generated content at query #96
#--------------------------

```python
def test_CurrencyRegistry():
    # Test singleton behavior
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2

    # Test initial state
    assert len(registry1) == 0
    assert registry1.all == []
    assert registry1.codes == []
    assert registry1.codenames == []

    # Test context manager behavior
    with registry1 as register:
        # Test adding currencies in context
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        register(usd)

        # Test duplicate currency raises error
        with pytest.raises(ValueError, match="Currency USD is already registered."):
            register(usd)

    # Test post-context state
    assert len(registry1) == 1
    assert registry1.all == [usd]
    assert registry1.codes == ["USD"]
    assert registry1.codenames == [("USD", "US Dollar")]

    # Test adding outside context raises error
    with pytest.raises(ProgrammingError, match="Can not create currencies outside registry context."):
        registry1._CurrencyRegistry__register(usd)

    # Test currency access methods
    assert "USD" in registry1
    assert registry1.has("USD")
    assert registry1["USD"] == usd
    assert registry1.get("USD") == usd
    assert registry1.get("XXX") is None
    assert registry1.get("XXX", default=usd) == usd

    # Test non-existent currency raises lookup error
    with pytest.raises(CurrencyLookupError, match="Currency identified by code 'XXX' does not exist"):
        registry1["XXX"]


# LLM-generated content at query #97
#--------------------------

```python
def test_CurrencyRegistry___exit__():
    # Create a new CurrencyRegistry instance
    registry = CurrencyRegistry()

    # Enter the context
    register = registry.__enter__()

    # Add some currencies
    usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    register(usd)
    register(eur)

    # Exit the context
    registry.__exit__(None, None, None)

    # Check that the registry is sorted
    assert registry.codes == ["EUR", "USD"]
    assert registry.codenames == [("EUR", "Euro"), ("USD", "US Dollar")]
    assert registry.all == [eur, usd]

    # Check that the context is closed
    assert registry._CurrencyRegistry__ctx_open is False

    # Check that adding a currency outside the context raises an error
    with pytest.raises(ProgrammingError):
        register(usd)


# LLM-generated content at query #98
#--------------------------

```python
def test_CurrencyRegistry___exit__():
    # Setup
    registry = CurrencyRegistry()
    test_currency1 = Currency.of("TST", "Test Currency 1", 2, CurrencyType.MONEY)
    test_currency2 = Currency.of("TST2", "Test Currency 2", 0, CurrencyType.METAL)

    # Test context manager behavior
    with registry as register:
        register(test_currency1)
        register(test_currency2)

    # Verify post-exit state
    assert len(registry) == 2
    assert registry.codes == ["TST", "TST2"]
    assert registry["TST"] == test_currency1
    assert registry["TST2"] == test_currency2
    assert registry.codenames == [("TST", "Test Currency 1"), ("TST2", "Test Currency 2")]

    # Test that context is closed after exit
    try:
        registry._CurrencyRegistry__register(test_currency1)
        assert False, "Should have raised ProgrammingError"
    except ProgrammingError:
        pass

    # Test that registry maintains order after multiple additions
    with registry as register:
        test_currency3 = Currency.of("TST3", "Test Currency 3", 3, CurrencyType.CRYPTO)
        register(test_currency3)

    assert registry.codes == ["TST", "TST2", "TST3"]


# LLM-generated content at query #99
#--------------------------

```python
def test_CurrencyRegistry___len__():
    registry = CurrencyRegistry()
    assert len(registry) == 0

    with registry as register:
        register(Currency.of("TEST1", "Test Currency 1", 2, CurrencyType.MONEY))
        register(Currency.of("TEST2", "Test Currency 2", 0, CurrencyType.MONEY))

    assert len(registry) == 2


# LLM-generated content at query #100
#--------------------------

```python
def test_CurrencyRegistry___enter__():
    # Test that __enter__ returns the __register method
    registry = CurrencyRegistry()
    with registry as register:
        assert callable(register)
        assert register.__name__ == "_CurrencyRegistry__register"

    # Test that __enter__ sets __ctx_open to True
    registry = CurrencyRegistry()
    assert not registry._CurrencyRegistry__ctx_open
    with registry:
        assert registry._CurrencyRegistry__ctx_open
    assert not registry._CurrencyRegistry__ctx_open


# LLM-generated content at query #101
#--------------------------

```python
def test_CurrencyRegistry_get():
    # Setup
    registry = CurrencyRegistry()
    with registry as register:
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
        register(usd)
        register(eur)

    # Test existing currency with no default
    assert registry.get("USD") == usd
    assert registry.get("EUR") == eur

    # Test non-existing currency with no default
    assert registry.get("XYZ") is None

    # Test non-existing currency with default
    assert registry.get("XYZ", default=usd) == usd
    assert registry.get("ABC", default=eur) == eur

    # Test that default is not used when currency exists
    assert registry.get("USD", default=eur) == usd
    assert registry.get("EUR", default=usd) == eur


# LLM-generated content at query #102
#--------------------------

```python
def test_CurrencyRegistry___enter__():
    # Test that __enter__ returns the __register method
    registry = CurrencyRegistry()
    with registry as register:
        assert callable(register)
        assert register.__name__ == "__register"

    # Test that __enter__ sets __ctx_open to True
    registry = CurrencyRegistry()
    with registry:
        assert registry._CurrencyRegistry__ctx_open is True

    # Test that __exit__ sets __ctx_open back to False
    registry = CurrencyRegistry()
    with registry:
        pass
    assert registry._CurrencyRegistry__ctx_open is False


# LLM-generated content at query #103
#--------------------------

```python
def test_CurrencyRegistry___enter__():
    # Test that __enter__ returns the __register method
    registry = CurrencyRegistry()
    with registry as register:
        assert callable(register)
        assert register.__name__ == "_CurrencyRegistry__register"

    # Test that __enter__ sets __ctx_open to True
    registry = CurrencyRegistry()
    with registry:
        assert registry._CurrencyRegistry__ctx_open is True

    # Test that __exit__ sets __ctx_open back to False
    registry = CurrencyRegistry()
    with registry:
        pass
    assert registry._CurrencyRegistry__ctx_open is False


# LLM-generated content at query #104
#--------------------------

```python
def test_CurrencyRegistry_get():
    # Setup
    registry = CurrencyRegistry()
    with registry as register:
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
        register(usd)
        register(eur)

    # Test existing currency with no default
    assert registry.get("USD") == usd
    assert registry.get("EUR") == eur

    # Test non-existing currency with no default
    assert registry.get("XYZ") is None

    # Test non-existing currency with default
    assert registry.get("XYZ", default=usd) == usd
    assert registry.get("ABC", default=eur) == eur

    # Test existing currency with default (should return the currency, not the default)
    assert registry.get("USD", default=eur) == usd
    assert registry.get("EUR", default=usd) == eur


# LLM-generated content at query #105
#--------------------------

```python
def test_CurrencyLookupError():
    # Test initialization and message
    error = CurrencyLookupError("XYZ")
    assert error.code == "XYZ"
    assert str(error) == "Currency identified by code 'XYZ' does not exist"

    # Test inheritance
    assert isinstance(error, LookupError)


# LLM-generated content at query #106
#--------------------------

```python
def test_Currency_quantize():
    # Test with USD (2 decimals)
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert USD.quantize(Decimal("1.005")) == Decimal("1.00")
    assert USD.quantize(Decimal("1.015")) == Decimal("1.02")
    assert USD.quantize(Decimal("1.000")) == Decimal("1.00")
    assert USD.quantize(Decimal("1.999")) == Decimal("2.00")

    # Test with JPY (0 decimals)
    JPY = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert JPY.quantize(Decimal("0.5")) == Decimal("0")
    assert JPY.quantize(Decimal("1.5")) == Decimal("2")
    assert JPY.quantize(Decimal("1.0")) == Decimal("1")
    assert JPY.quantize(Decimal("9.9")) == Decimal("10")

    # Test with a currency with no fixed precision (ZZZ)
    ZZZ = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert ZZZ.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert ZZZ.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")
    assert ZZZ.quantize(Decimal("1.0000000000000")) == Decimal("1.000000000000")
    assert ZZZ.quantize(Decimal("1.0000000000009")) == Decimal("1.000000000001")

    # Test with negative values
    assert USD.quantize(Decimal("-1.005")) == Decimal("-1.00")
    assert USD.quantize(Decimal("-1.015")) == Decimal("-1.02")
    assert JPY.quantize(Decimal("-0.5")) == Decimal("0")
    assert JPY.quantize(Decimal("-1.5")) == Decimal("-2")
    assert ZZZ.quantize(Decimal("-1.0000000000005")) == Decimal("-1.000000000000")
    assert ZZZ.quantize(Decimal("-1.0000000000015")) == Decimal("-1.000000000002")


# LLM-generated content at query #107
#--------------------------

```python
def test_Currency___eq__():
    # Test equality with same currency
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2

    # Test inequality with different currency code
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert not (usd == eur)

    # Test inequality with different currency name
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 == usd2)

    # Test inequality with different decimals
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 0, CurrencyType.MONEY)
    assert not (usd1 == usd2)

    # Test inequality with different currency type
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert not (usd1 == usd2)

    # Test inequality with non-Currency object
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd == "USD")
    assert not (usd == 123)
    assert not (usd == None)


# LLM-generated content at query #108
#--------------------------

```python
def test_Currency___delattr__():
    # Create a currency instance
    currency = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)

    # Attempt to delete an attribute and verify it raises an AttributeError
    with pytest.raises(AttributeError):
        del currency.code


# LLM-generated content at query #109
#--------------------------

```python
def test_CurrencyLookupError():
    # Test initialization and message
    error = CurrencyLookupError("XYZ")
    assert error.code == "XYZ"
    assert str(error) == "Currency identified by code 'XYZ' does not exist"

    # Test inheritance
    assert isinstance(error, LookupError)


# LLM-generated content at query #110
#--------------------------

```python
def test_CurrencyRegistry_has():
    # Test with existing currency
    assert Currencies.has("USD") is True

    # Test with non-existing currency
    assert Currencies.has("NONEXISTENT") is False

    # Test with empty string
    assert Currencies.has("") is False

    # Test with non-string input (should return False)
    assert Currencies.has(123) is False


# LLM-generated content at query #111
#--------------------------

```python
def test_CurrencyRegistry___exit__():
    # Setup
    registry = CurrencyRegistry()
    test_currency = Currency.of("TEST", "Test Currency", 2, CurrencyType.MONEY)

    # Test context management
    with registry as register:
        register(test_currency)

    # Verify the currency was added
    assert "TEST" in registry
    assert registry["TEST"] == test_currency

    # Verify the internal structures are sorted
    assert registry.codes == sorted(registry.codes)
    assert registry.all == sorted(registry.all, key=lambda x: x.code)
    assert registry.codenames == sorted(registry.codenames, key=lambda x: x[0])

    # Verify context is closed
    try:
        registry._CurrencyRegistry__register(test_currency)
        assert False, "Should not be able to register outside context"
    except ProgrammingError:
        pass

    # Verify no exception handling issues
    with pytest.raises(ValueError):
        with registry as register:
            register(test_currency)  # Should fail as already registered


# LLM-generated content at query #112
#--------------------------

```python
def test_Currency___le__():
    # Test equality case
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 <= usd2
    assert usd2 <= usd1

    # Test less than case (by code)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert eur <= usd
    assert not (usd <= eur)

    # Test less than case (by name)
    usd1 = Currency.of("USD", "AAA", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "BBB", 2, CurrencyType.MONEY)
    assert usd1 <= usd2
    assert not (usd2 <= usd1)

    # Test less than case (by decimals)
    usd1 = Currency.of("USD", "US Dollars", 1, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 <= usd2
    assert not (usd2 <= usd1)

    # Test less than case (by type)
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert usd1 <= usd2
    assert not (usd2 <= usd1)

    # Test with different quantizers (indirectly through decimals)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy <= usd
    assert not (usd <= jpy)


# LLM-generated content at query #113
#--------------------------

```python
def test_Currency___repr__():
    # Test with USD
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert repr(usd) == "Currency(code='USD', name='US Dollars', decimals=2, type=CurrencyType.MONEY, quantizer=Decimal('0.01'), hashcache={})".format(hash(('USD', 'US Dollars', 2, CurrencyType.MONEY, Decimal('0.01'))))

    # Test with JPY
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert repr(jpy) == "Currency(code='JPY', name='Japanese Yen', decimals=0, type=CurrencyType.MONEY, quantizer=Decimal('0'), hashcache={})".format(hash(('JPY', 'Japanese Yen', 0, CurrencyType.MONEY, Decimal('0'))))

    # Test with a weird currency
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert repr(zzz) == "Currency(code='ZZZ', name='Some weird currency', decimals=-1, type=CurrencyType.CRYPTO, quantizer=MaxPrecisionQuantizer, hashcache={})".format(hash(('ZZZ', 'Some weird currency', -1, CurrencyType.CRYPTO, MaxPrecisionQuantizer)))


# LLM-generated content at query #114
#--------------------------

```python
def test_CurrencyLookupError():
    # Test initialization and message
    error = CurrencyLookupError("XYZ")
    assert error.code == "XYZ"
    assert str(error) == "Currency identified by code 'XYZ' does not exist"

    # Test inheritance from LookupError
    assert isinstance(error, LookupError)

    # Test that the error can be raised
    with pytest.raises(CurrencyLookupError) as exc_info:
        raise CurrencyLookupError("ABC")
    assert exc_info.value.code == "ABC"
    assert str(exc_info.value) == "Currency identified by code 'ABC' does not exist"


# LLM-generated content at query #115
#--------------------------

```python
def test_CurrencyRegistry_get():
    # Setup
    registry = CurrencyRegistry()
    with registry as register:
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
        register(usd)
        register(eur)

    # Test existing currency with no default
    assert registry.get("USD") == usd
    assert registry.get("EUR") == eur

    # Test non-existing currency with no default
    assert registry.get("XXX") is None

    # Test non-existing currency with default
    assert registry.get("XXX", default=usd) == usd
    assert registry.get("YYY", default=eur) == eur

    # Test that default is not used when currency exists
    assert registry.get("USD", default=eur) == usd
    assert registry.get("EUR", default=usd) == eur


# LLM-generated content at query #116
#--------------------------

```python
def test_CurrencyRegistry_get():
    # Setup
    registry = CurrencyRegistry()
    with registry as register:
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
        register(usd)
        register(eur)

    # Test existing currency with no default
    assert registry.get("USD") == usd

    # Test existing currency with default (should ignore default)
    assert registry.get("EUR", default=usd) == eur

    # Test non-existing currency with no default
    assert registry.get("XYZ") is None

    # Test non-existing currency with default
    assert registry.get("XYZ", default=usd) == usd


# LLM-generated content at query #117
#--------------------------

```python
def test_CurrencyRegistry_get():
    # Setup
    registry = CurrencyRegistry()
    with registry as register:
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
        register(usd)
        register(eur)

    # Test existing currency
    assert registry.get("USD") == usd
    assert registry.get("EUR") == eur

    # Test non-existing currency with default
    assert registry.get("XXX") is None
    assert registry.get("XXX", default=eur) == eur

    # Test non-existing currency without default
    assert registry.get("NON-EXISTING") is None


# LLM-generated content at query #118
#--------------------------

```python
def test_Currency___setattr__():
    with pytest.raises(AttributeError):
        currency = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
        currency.code = "EUR"


# LLM-generated content at query #119
#--------------------------

```python
def test_CurrencyLookupError():
    # Test initialization and message
    error = CurrencyLookupError("XYZ")
    assert error.code == "XYZ"
    assert str(error) == "Currency identified by code 'XYZ' does not exist"

    # Test inheritance
    assert isinstance(error, LookupError)


# LLM-generated content at query #120
#--------------------------

```python
def test_CurrencyRegistry___contains__():
    # Setup
    registry = CurrencyRegistry()
    with registry as register:
        register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        register(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))

    # Test existing currency
    assert "USD" in registry
    assert "EUR" in registry

    # Test non-existing currency
    assert "XYZ" not in registry
    assert "NONEXISTENT" not in registry


# LLM-generated content at query #121
#--------------------------

```python
def test_Currency___eq__():
    # Test equality with same currency
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2

    # Test inequality with different currency
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert usd1 != usdx

    # Test inequality with different type
    assert usd1 != "USD"

    # Test inequality with different code
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert usd1 != jpy

    # Test inequality with different name
    usd3 = Currency.of("USD", "US Dollars Different", 2, CurrencyType.MONEY)
    assert usd1 != usd3

    # Test inequality with different decimals
    usd4 = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert usd1 != usd4

    # Test inequality with different type
    usd5 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert usd1 != usd5


# LLM-generated content at query #122
#--------------------------

```python
def test_CurrencyRegistry___exit__():
    # Setup
    registry = CurrencyRegistry()
    test_currency = Currency.of("TEST", "Test Currency", 2, CurrencyType.MONEY)

    # Test context manager behavior
    with registry as register:
        register(test_currency)
        # Verify currency is added during context
        assert "TEST" in registry
        assert registry["TEST"] == test_currency

    # Verify post-exit behavior
    assert registry["TEST"] == test_currency
    assert registry.codes == ["TEST"]
    assert registry.codenames == [("TEST", "Test Currency")]
    assert registry.all == [test_currency]

    # Test that context is closed after exit
    try:
        registry._CurrencyRegistry__register(test_currency)
        assert False, "Should have raised ProgrammingError"
    except ProgrammingError:
        pass

    # Test re-sorting behavior
    with registry as register:
        register(Currency.of("AAA", "First Currency", 2, CurrencyType.MONEY))
        register(Currency.of("ZZZ", "Last Currency", 2, CurrencyType.MONEY))

    assert registry.codes == ["AAA", "TEST", "ZZZ"]
    assert [c.code for c in registry.all] == ["AAA", "TEST", "ZZZ"]


# LLM-generated content at query #123
#--------------------------

```python
def test_CurrencyRegistry___new__():
    # Test that CurrencyRegistry is a singleton
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2

    # Test that the singleton instance is created only once
    assert CurrencyRegistry._CurrencyRegistry__instance is not None
    assert CurrencyRegistry._CurrencyRegistry__instance is registry1
    assert CurrencyRegistry._CurrencyRegistry__instance is registry2


# LLM-generated content at query #124
#--------------------------

```python
def test_CurrencyRegistry_has():
    # Test with existing currency
    assert Currencies.has("USD") is True

    # Test with non-existing currency
    assert Currencies.has("NONEXISTENT") is False

    # Test with empty string
    assert Currencies.has("") is False

    # Test with case sensitivity
    assert Currencies.has("usd") is False


# LLM-generated content at query #125
#--------------------------

```python
def test_CurrencyRegistry_has():
    # Setup
    registry = CurrencyRegistry()
    with registry as register:
        register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        register(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))

    # Test existing currency
    assert registry.has("USD") is True
    assert registry.has("EUR") is True

    # Test non-existing currency
    assert registry.has("XYZ") is False
    assert registry.has("NONEXISTENT") is False

    # Test case sensitivity
    assert registry.has("usd") is False
    assert registry.has("eur") is False


# LLM-generated content at query #126
#--------------------------

```python
def test_CurrencyRegistry___enter__():
    # Test that __enter__ returns the __register method
    registry = CurrencyRegistry()
    with registry as register:
        assert callable(register)
        assert register.__name__ == "_CurrencyRegistry__register"

    # Test that __enter__ sets __ctx_open to True
    registry = CurrencyRegistry()
    with registry as register:
        assert registry._CurrencyRegistry__ctx_open is True

    # Test that __exit__ sets __ctx_open to False after context exits
    registry = CurrencyRegistry()
    with registry as register:
        pass
    assert registry._CurrencyRegistry__ctx_open is False

    # Test that __enter__ can be called multiple times in nested contexts
    registry = CurrencyRegistry()
    with registry as register1:
        assert registry._CurrencyRegistry__ctx_open is True
        with registry as register2:
            assert registry._CurrencyRegistry__ctx_open is True
            assert callable(register2)
        assert registry._CurrencyRegistry__ctx_open is True
    assert registry._CurrencyRegistry__ctx_open is False


# LLM-generated content at query #127
#--------------------------

```python
def test_CurrencyRegistry___contains__():
    registry = CurrencyRegistry()
    with registry:
        registry(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        registry(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))

    assert "USD" in registry
    assert "EUR" in registry
    assert "JPY" not in registry


# LLM-generated content at query #128
#--------------------------

```python
def test_CurrencyRegistry():
    # Test singleton behavior
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2

    # Test initial state
    assert len(registry1) == 0
    assert registry1.all == []
    assert registry1.codes == []
    assert registry1.codenames == []

    # Test context manager behavior
    with registry1 as register:
        test_currency = Currency.of("TEST", "Test Currency", 2, CurrencyType.MONEY)
        register(test_currency)

    assert len(registry1) == 1
    assert "TEST" in registry1
    assert registry1["TEST"] == test_currency
    assert registry1.all == [test_currency]
    assert registry1.codes == ["TEST"]
    assert registry1.codenames == [("TEST", "Test Currency")]

    # Test duplicate registration
    with pytest.raises(ValueError):
        with registry1 as register:
            register(test_currency)

    # Test registration outside context
    with pytest.raises(ProgrammingError):
        registry1._CurrencyRegistry__register(test_currency)


# LLM-generated content at query #129
#--------------------------

```python
def test_CurrencyRegistry():
    # Test singleton behavior
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2

    # Test initial state
    assert len(registry1) == 0
    assert registry1.all == []
    assert registry1.codes == []
    assert registry1.codenames == []

    # Test context manager
    with registry1 as register:
        # Test adding a currency
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        register(usd)

        # Test adding duplicate currency raises error
        with pytest.raises(ValueError, match="Currency USD is already registered."):
            register(usd)

        # Test adding currency outside context raises error
        with pytest.raises(ProgrammingError, match="Can not create currencies outside registry context."):
            registry1._CurrencyRegistry__register(usd)

    # Test post-context state
    assert len(registry1) == 1
    assert registry1.all == [usd]
    assert registry1.codes == ["USD"]
    assert registry1.codenames == [("USD", "US Dollar")]

    # Test currency lookup
    assert registry1.has("USD") is True
    assert registry1["USD"] == usd
    assert registry1.get("USD") == usd
    assert registry1.get("XXX") is None
    assert registry1.get("XXX", default=usd) == usd

    # Test non-existing currency raises error
    with pytest.raises(CurrencyLookupError, match="Currency identified by code 'XXX' does not exist"):
        registry1["XXX"]

    # Test containment
    assert "USD" in registry1
    assert "XXX" not in registry1


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_Currency___eq__():
    # Test equality with same currency
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2

    # Test inequality with different currency
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert not (usd == eur)

    # Test inequality with different name
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 == usd2)

    # Test inequality with different decimals
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 0, CurrencyType.MONEY)
    assert not (usd1 == usd2)

    # Test inequality with different type
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert not (usd1 == usd2)

    # Test inequality with non-Currency object
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd == "USD")


# LLM-generated content at query #2
#--------------------------

```python
def test_Currency___eq__():
    # Test equality with same currency
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2

    # Test inequality with different currency
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert not (usd == eur)

    # Test inequality with different name
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 == usd2)

    # Test inequality with different decimals
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 0, CurrencyType.MONEY)
    assert not (usd1 == usd2)

    # Test inequality with different type
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert not (usd1 == usd2)

    # Test inequality with non-Currency object
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd == "USD")


# LLM-generated content at query #3
#--------------------------

```python
def test_CurrencyRegistry___getitem__():
    # Test getting an existing currency
    usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    with Currencies as register:
        register(usd)
    assert Currencies["USD"] == usd

    # Test getting a non-existing currency raises CurrencyLookupError
    with pytest.raises(CurrencyLookupError):
        Currencies["NON-EXISTING"]


# LLM-generated content at query #4
#--------------------------

```python
def test_CurrencyRegistry___getitem__():
    # Setup: Create a CurrencyRegistry instance and populate it with test currencies
    registry = CurrencyRegistry()
    with registry as register:
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
        register(usd)
        register(eur)

    # Test: Retrieve existing currencies
    assert registry["USD"] == usd
    assert registry["EUR"] == eur

    # Test: Attempt to retrieve non-existing currency
    with pytest.raises(CurrencyLookupError) as excinfo:
        _ = registry["NON-EXISTING"]
    assert str(excinfo.value) == "Currency identified by code 'NON-EXISTING' does not exist"


# LLM-generated content at query #5
#--------------------------

```python
def test_Currency___eq__():
    # Test equality with same currency
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2

    # Test inequality with different currency code
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert not (usd == eur)

    # Test inequality with different currency name
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 == usd2)

    # Test inequality with different currency decimals
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 0, CurrencyType.MONEY)
    assert not (usd1 == usd2)

    # Test inequality with different currency type
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert not (usd1 == usd2)

    # Test inequality with non-Currency object
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd == "USD")
    assert not (usd == 123)
    assert not (usd == None)


# LLM-generated content at query #6
#--------------------------

```python
def test_Currency___eq__():
    # Test equality with same currency
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2

    # Test inequality with different currency
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 == usdx)

    # Test inequality with different type
    assert not (usd1 == "USD")

    # Test inequality with different attributes
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert not (usd1 == jpy)


# LLM-generated content at query #7
#--------------------------

```python
def test_CurrencyRegistry___getitem__():
    # Setup a test currency
    test_currency = Currency.of("TEST", "Test Currency", 2, CurrencyType.MONEY)

    # Use the registry context to add the test currency
    with CurrencyRegistry() as register:
        register(test_currency)

    # Test successful lookup
    assert CurrencyRegistry()["TEST"] == test_currency

    # Test lookup error for non-existing currency
    with pytest.raises(CurrencyLookupError):
        CurrencyRegistry()["NONEXISTENT"]


# LLM-generated content at query #8
#--------------------------

```python
def test_CurrencyRegistry___getitem__():
    # Setup
    registry = CurrencyRegistry()
    with registry as register:
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
        register(usd)
        register(eur)

    # Test existing currency
    assert registry["USD"] == usd
    assert registry["EUR"] == eur

    # Test non-existing currency
    with pytest.raises(CurrencyLookupError) as excinfo:
        _ = registry["NON-EXISTING"]
    assert str(excinfo.value) == "Currency identified by code 'NON-EXISTING' does not exist"


# LLM-generated content at query #9
#--------------------------

```python
def test_Currency___eq__():
    # Test equality with same currency
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2

    # Test inequality with different currency
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert not (usd == eur)

    # Test inequality with different name
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 == usd2)

    # Test inequality with different decimals
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 0, CurrencyType.MONEY)
    assert not (usd1 == usd2)

    # Test inequality with different type
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert not (usd1 == usd2)

    # Test inequality with non-Currency object
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd == "USD")


# LLM-generated content at query #10
#--------------------------

```python
def test_Currency___eq__():
    # Test equality with same currency
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2

    # Test inequality with different currency code
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert not (usd == eur)

    # Test inequality with different currency name
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 == usd2)

    # Test inequality with different decimals
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 0, CurrencyType.MONEY)
    assert not (usd1 == usd2)

    # Test inequality with different currency type
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert not (usd1 == usd2)

    # Test inequality with non-Currency object
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd == "USD")
    assert not (usd == 123)
    assert not (usd == None)


# LLM-generated content at query #11
#--------------------------

```python
def test_Currency___repr__():
    # Test with USD
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert repr(usd) == "Currency(code='USD', name='US Dollars', decimals=2, type=CurrencyType.MONEY, quantizer=Decimal('0.01'), hashcache={})".format(hash(('USD', 'US Dollars', 2, CurrencyType.MONEY, Decimal('0.01'))))

    # Test with JPY
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert repr(jpy) == "Currency(code='JPY', name='Japanese Yen', decimals=0, type=CurrencyType.MONEY, quantizer=Decimal('0'), hashcache={})".format(hash(('JPY', 'Japanese Yen', 0, CurrencyType.MONEY, Decimal('0'))))

    # Test with a weird currency
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert repr(zzz) == "Currency(code='ZZZ', name='Some weird currency', decimals=-1, type=CurrencyType.CRYPTO, quantizer=MaxPrecisionQuantizer, hashcache={})".format(hash(('ZZZ', 'Some weird currency', -1, CurrencyType.CRYPTO, MaxPrecisionQuantizer)))


# LLM-generated content at query #12
#--------------------------

```python
def test_Currency___hash__():
    # Test that the hash is correctly computed and cached
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)

    # Check that the hash is the same for the same currency
    assert hash(usd1) == hash(usd2)

    # Check that the hash is different for different currencies
    assert hash(usd1) != hash(usdx)

    # Check that the hash is computed correctly
    assert hash(usd1) == hash(("USD", "US Dollars", 2, CurrencyType.MONEY, make_quantizer(2)))


# LLM-generated content at query #13
#--------------------------

```python
def test_Currency___gt__():
    # Test with same currency
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 > usd2)

    # Test with different currencies
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert usd > jpy or jpy > usd  # One should be greater than the other

    # Test with different attributes
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert usd > usdx or usdx > usd  # One should be greater than the other

    # Test with different decimals
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert usd > zzz or zzz > usd  # One should be greater than the other

    # Test with different types
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    gold = Currency.of("XAU", "Gold", 2, CurrencyType.METAL)
    assert usd > gold or gold > usd  # One should be greater than the other


# LLM-generated content at query #14
#--------------------------

```python
def test_Currency():
    # Test valid currency creation
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.code == "USD"
    assert usd.name == "US Dollars"
    assert usd.decimals == 2
    assert usd.type == CurrencyType.MONEY
    assert usd.quantize(Decimal("1.005")) == Decimal("1.00")
    assert usd.quantize(Decimal("1.015")) == Decimal("1.02")

    # Test currency with zero decimals
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy.code == "JPY"
    assert jpy.name == "Japanese Yen"
    assert jpy.decimals == 0
    assert jpy.type == CurrencyType.MONEY
    assert jpy.quantize(Decimal("0.5")) == Decimal("0")
    assert jpy.quantize(Decimal("1.5")) == Decimal("2")

    # Test currency with variable precision
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert zzz.code == "ZZZ"
    assert zzz.name == "Some weird currency"
    assert zzz.decimals == -1
    assert zzz.type == CurrencyType.CRYPTO
    assert zzz.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert zzz.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")

    # Test equality and hash
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2
    assert usd1 != usdx
    assert hash(usd1) == hash(usd2)
    assert hash(usd1) != hash(usdx)

    # Test invalid currency code
    with pytest.raises(ProgrammingError):
        Currency.of("123", "Invalid Code", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("abc", "Invalid Code", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("Abc", "Invalid Code", 2, CurrencyType.MONEY)

    # Test invalid currency name
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", " ", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", " US Dollars", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars ", 2, CurrencyType.MONEY)

    # Test invalid decimals
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", -2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", "2", CurrencyType.MONEY)

    # Test invalid currency type
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", 2, "MONEY")


# LLM-generated content at query #15
#--------------------------

```python
def test_CurrencyRegistry_has():
    # Test with existing currency
    assert Currencies.has("USD") is True

    # Test with non-existing currency
    assert Currencies.has("NONEXIST") is False

    # Test with empty string
    assert Currencies.has("") is False

    # Test with code that has special characters
    assert Currencies.has("USD!") is False

    # Test with code that has spaces
    assert Currencies.has("USD ") is False


# LLM-generated content at query #16
#--------------------------

```python
def test_CurrencyRegistry___new__():
    # Test singleton behavior
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2

    # Test that the instance is of the correct type
    assert isinstance(registry1, CurrencyRegistry)


# LLM-generated content at query #17
#--------------------------

```python
def test_Currency___eq__():
    # Test equality with same currency
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2

    # Test inequality with different currency
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert not (usd == eur)

    # Test inequality with different name
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 == usd2)

    # Test inequality with different decimals
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 0, CurrencyType.MONEY)
    assert not (usd1 == usd2)

    # Test inequality with different type
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert not (usd1 == usd2)

    # Test inequality with non-Currency object
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd == "USD")


# LLM-generated content at query #18
#--------------------------

```python
def test_Currency___hash__():
    # Test that hash returns the pre-computed hashcache
    currency = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert currency.__hash__() == currency.hashcache

    # Test that equal currencies have the same hash
    currency1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    currency2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert currency1.__hash__() == currency2.__hash__()

    # Test that different currencies have different hashes
    currency3 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert currency1.__hash__() != currency3.__hash__()

    # Test that currencies with different attributes have different hashes
    currency4 = Currency.of("USD", "US Dollars", 0, CurrencyType.MONEY)
    assert currency1.__hash__() != currency4.__hash__()


# LLM-generated content at query #19
#--------------------------

```python
def test_Currency___gt__():
    # Test with different codes
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert (usd > eur) == ("USD" > "EUR")

    # Test with same code but different names
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars 2", 2, CurrencyType.MONEY)
    assert (usd1 > usd2) == (("USD", "US Dollars", 2, CurrencyType.MONEY, usd1.quantizer) >
                            ("USD", "US Dollars 2", 2, CurrencyType.MONEY, usd2.quantizer))

    # Test with same code and name but different decimals
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert (usd1 > usd2) == (("USD", "US Dollars", 2, CurrencyType.MONEY, usd1.quantizer) >
                            ("USD", "US Dollars", 3, CurrencyType.MONEY, usd2.quantizer))

    # Test with same code, name, and decimals but different types
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert (usd1 > usd2) == (("USD", "US Dollars", 2, CurrencyType.MONEY, usd1.quantizer) >
                            ("USD", "US Dollars", 2, CurrencyType.CRYPTO, usd2.quantizer))

    # Test with same code, name, decimals, and type but different quantizers
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", -1, CurrencyType.MONEY)
    assert (usd1 > usd2) == (("USD", "US Dollars", 2, CurrencyType.MONEY, usd1.quantizer) >
                            ("USD", "US Dollars", -1, CurrencyType.MONEY, usd2.quantizer))

    # Test with same currency
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 > usd2)


# LLM-generated content at query #20
#--------------------------

```python
def test_Currency():
    # Test valid currency creation
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.code == "USD"
    assert usd.name == "US Dollars"
    assert usd.decimals == 2
    assert usd.type == CurrencyType.MONEY
    assert usd.quantize(Decimal("1.005")) == Decimal("1.00")
    assert usd.quantize(Decimal("1.015")) == Decimal("1.02")

    # Test equality and hash
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd == usd2
    assert hash(usd) == hash(usd2)

    # Test inequality
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert usd != usdx
    assert hash(usd) != hash(usdx)

    # Test JPY with 0 decimals
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy.quantize(Decimal("0.5")) == Decimal("0")
    assert jpy.quantize(Decimal("1.5")) == Decimal("2")

    # Test currency with -1 decimals
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert zzz.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert zzz.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")

    # Test invalid currency code
    with pytest.raises(ProgrammingError):
        Currency.of("123", "Invalid Code", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("usd", "Lowercase Code", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD ", "Code with Space", 2, CurrencyType.MONEY)

    # Test invalid currency name
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "  Trim Me  ", 2, CurrencyType.MONEY)

    # Test invalid decimals
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", -2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", "2", CurrencyType.MONEY)

    # Test invalid currency type
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", 2, "MONEY")


# LLM-generated content at query #21
#--------------------------

```python
def test_Currency___hash__():
    # Test basic hash functionality
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert hash(usd1) == hash(usd2)

    # Test hash with different currencies
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert hash(usd1) != hash(jpy)

    # Test hash with different parameters
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert hash(usd1) != hash(usdx)

    # Test hash with different decimals
    usd3 = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert hash(usd1) != hash(usd3)

    # Test hash with different types
    usd4 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert hash(usd1) != hash(usd4)

    # Test hash with negative decimals
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert hash(usd1) != hash(zzz)


# LLM-generated content at query #22
#--------------------------

```python
def test_Currency___repr__():
    currency = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert repr(currency) == "Currency(code='USD', name='US Dollars', decimals=2, type=CurrencyType.MONEY, quantizer=Decimal('0.01'), hashcache=...)"


# LLM-generated content at query #23
#--------------------------

```python
def test_Currency():
    # Test valid currency creation
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.code == "USD"
    assert usd.name == "US Dollars"
    assert usd.decimals == 2
    assert usd.type == CurrencyType.MONEY
    assert usd.quantizer == make_quantizer(2)

    # Test currency with zero decimals
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy.code == "JPY"
    assert jpy.name == "Japanese Yen"
    assert jpy.decimals == 0
    assert jpy.type == CurrencyType.MONEY
    assert jpy.quantizer == ZERO

    # Test currency with negative decimals
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert zzz.code == "ZZZ"
    assert zzz.name == "Some weird currency"
    assert zzz.decimals == -1
    assert zzz.type == CurrencyType.CRYPTO
    assert zzz.quantizer == MaxPrecisionQuantizer

    # Test currency equality and hash
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2
    assert hash(usd1) == hash(usd2)

    # Test currency inequality
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert usd1 != usdx
    assert hash(usd1) != hash(usdx)

    # Test quantize method
    assert usd.quantize(Decimal("1.005")) == Decimal("1.00")
    assert usd.quantize(Decimal("1.015")) == Decimal("1.02")
    assert jpy.quantize(Decimal("0.5")) == Decimal("0")
    assert jpy.quantize(Decimal("1.5")) == Decimal("2")
    assert zzz.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert zzz.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")

    # Test invalid currency code
    with pytest.raises(ProgrammingError):
        Currency.of("123", "Invalid Code", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("usd", "Lowercase Code", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD ", "Code with Space", 2, CurrencyType.MONEY)

    # Test invalid currency name
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", " ", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", " Leading Space", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "Trailing Space ", 2, CurrencyType.MONEY)

    # Test invalid decimals
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", -2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", "2", CurrencyType.MONEY)

    # Test invalid currency type
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", 2, "MONEY")


# LLM-generated content at query #24
#--------------------------

```python
def test_Currency___setattr__():
    # Test that Currency is immutable and __setattr__ raises an error
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)

    with pytest.raises(AttributeError):
        usd.code = "EUR"

    with pytest.raises(AttributeError):
        usd.name = "Euros"

    with pytest.raises(AttributeError):
        usd.decimals = 0

    with pytest.raises(AttributeError):
        usd.type = CurrencyType.CRYPTO

    with pytest.raises(AttributeError):
        usd.quantizer = Decimal("0.0001")

    with pytest.raises(AttributeError):
        usd.hashcache = 12345


# LLM-generated content at query #25
#--------------------------

```python
def test_Currency___ge__():
    # Test equality
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 >= usd2
    assert usd2 >= usd1

    # Test greater than
    usd3 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd4 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd3 >= usd4
    assert usd4 >= usd3

    # Test less than
    usd5 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd6 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd5 >= usd6
    assert usd6 >= usd5

    # Test different currencies
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert usd >= jpy or jpy >= usd


# LLM-generated content at query #26
#--------------------------

```python
def test_CurrencyRegistry___contains__():
    # Setup
    registry = CurrencyRegistry()
    with registry as register:
        register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        register(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))

    # Test existing currency
    assert "USD" in registry
    assert "EUR" in registry

    # Test non-existing currency
    assert "XYZ" not in registry
    assert "NONEXISTENT" not in registry


# LLM-generated content at query #27
#--------------------------

```python
def test_Currency___eq__():
    # Test equality with same currency
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2

    # Test inequality with different currency
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert not (usd == eur)

    # Test inequality with non-Currency object
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd == "USD")

    # Test inequality with different attributes
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert not (usd1 == usd2)

    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert not (usd1 == usd2)


# LLM-generated content at query #28
#--------------------------

```python
def test_CurrencyRegistry___getitem__():
    # Setup
    registry = CurrencyRegistry()
    with registry as register:
        register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        register(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))

    # Test existing currency
    usd = registry["USD"]
    assert usd.code == "USD"
    assert usd.name == "US Dollar"
    assert usd.decimals == 2
    assert usd.type == CurrencyType.MONEY

    # Test non-existing currency
    with pytest.raises(CurrencyLookupError) as excinfo:
        _ = registry["NON-EXISTING"]
    assert str(excinfo.value) == "Currency identified by code 'NON-EXISTING' does not exist"


# LLM-generated content at query #29
#--------------------------

```python
def test_CurrencyRegistry_has():
    # Setup
    registry = CurrencyRegistry()
    with registry as register:
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
        register(usd)
        register(eur)

    # Test existing currency
    assert registry.has("USD") is True
    assert registry.has("EUR") is True

    # Test non-existing currency
    assert registry.has("XYZ") is False
    assert registry.has("") is False


# LLM-generated content at query #30
#--------------------------

```python
def test_CurrencyRegistry___enter__():
    registry = CurrencyRegistry()
    with registry as register:
        assert callable(register)
        assert register.__name__ == "_CurrencyRegistry__register"
    assert not registry._CurrencyRegistry__ctx_open


# LLM-generated content at query #31
#--------------------------

```python
def test_Currency___gt__():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)

    assert (usd > jpy) == (usd.hashcache > jpy.hashcache)
    assert (jpy > zzz) == (jpy.hashcache > zzz.hashcache)
    assert (zzz > usd) == (zzz.hashcache > usd.hashcache)
    assert (usd > usd) == False


# LLM-generated content at query #32
#--------------------------

```python
def test_Currency___hash__():
    # Test basic hash functionality
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert hash(usd1) == hash(usd2)

    # Test hash with different currencies
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert hash(usd1) != hash(jpy)

    # Test hash with different attributes
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert hash(usd1) != hash(usdx)

    # Test hash with different decimals
    usd3 = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert hash(usd1) != hash(usd3)

    # Test hash with different types
    usd4 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert hash(usd1) != hash(usd4)


# LLM-generated content at query #33
#--------------------------

```python
def test_CurrencyRegistry___contains__():
    # Setup
    registry = CurrencyRegistry()
    with registry as register:
        register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        register(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))

    # Test existing currency
    assert "USD" in registry
    assert "EUR" in registry

    # Test non-existing currency
    assert "XYZ" not in registry
    assert "NONEXISTENT" not in registry


# LLM-generated content at query #34
#--------------------------

```python
def test_Currency___eq__():
    # Test equality with same currency
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2

    # Test inequality with different currency
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert not (usd1 == eur)

    # Test inequality with non-Currency object
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd == "USD")

    # Test inequality with same code but different name
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 == usd2)

    # Test inequality with same code but different decimals
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 0, CurrencyType.MONEY)
    assert not (usd1 == usd2)

    # Test inequality with same code but different type
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert not (usd1 == usd2)


# LLM-generated content at query #35
#--------------------------

```python
def test_Currency():
    # Test valid currency creation
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.code == "USD"
    assert usd.name == "US Dollars"
    assert usd.decimals == 2
    assert usd.type == CurrencyType.MONEY
    assert usd.quantize(Decimal("1.005")) == Decimal("1.00")
    assert usd.quantize(Decimal("1.015")) == Decimal("1.02")

    # Test currency with zero decimals
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy.code == "JPY"
    assert jpy.name == "Japanese Yen"
    assert jpy.decimals == 0
    assert jpy.type == CurrencyType.MONEY
    assert jpy.quantize(Decimal("0.5")) == Decimal("0")
    assert jpy.quantize(Decimal("1.5")) == Decimal("2")

    # Test currency with negative decimals
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert zzz.code == "ZZZ"
    assert zzz.name == "Some weird currency"
    assert zzz.decimals == -1
    assert zzz.type == CurrencyType.CRYPTO
    assert zzz.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert zzz.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")

    # Test equality and hash
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2
    assert usd1 != usdx
    assert hash(usd1) == hash(usd2)
    assert hash(usd1) != hash(usdx)

    # Test invalid currency code
    with pytest.raises(ProgrammingError):
        Currency.of("usd", "US Dollars", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD1", "US Dollars", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of(123, "US Dollars", 2, CurrencyType.MONEY)

    # Test invalid currency name
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "  US Dollars", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars  ", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", 123, 2, CurrencyType.MONEY)

    # Test invalid decimals
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", -2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", "2", CurrencyType.MONEY)

    # Test invalid currency type
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", 2, "Money")


# LLM-generated content at query #36
#--------------------------

```python
def test_Currency___setattr__():
    # Create a currency instance
    ccy = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)

    # Test that setting an attribute raises an AttributeError
    with pytest.raises(AttributeError):
        ccy.code = "EUR"

    with pytest.raises(AttributeError):
        ccy.name = "Euros"

    with pytest.raises(AttributeError):
        ccy.decimals = 3

    with pytest.raises(AttributeError):
        ccy.type = CurrencyType.CRYPTO

    with pytest.raises(AttributeError):
        ccy.quantizer = Decimal("0.001")

    with pytest.raises(AttributeError):
        ccy.hashcache = 123


# LLM-generated content at query #37
#--------------------------

```python
def test_Currency___le__():
    # Test equality case
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 <= usd2
    assert usd2 <= usd1

    # Test less than case (by code)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert eur <= usd
    assert not (usd <= eur)

    # Test less than case (by name)
    usd1 = Currency.of("USD", "AAA", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "BBB", 2, CurrencyType.MONEY)
    assert usd1 <= usd2
    assert not (usd2 <= usd1)

    # Test less than case (by decimals)
    usd1 = Currency.of("USD", "US Dollars", 1, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 <= usd2
    assert not (usd2 <= usd1)

    # Test less than case (by type)
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert usd1 <= usd2
    assert not (usd2 <= usd1)

    # Test with different quantizers
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert usd1 <= usd2
    assert not (usd2 <= usd1)

    # Test with negative decimals
    weird1 = Currency.of("ZZZ", "Weird", -1, CurrencyType.CRYPTO)
    weird2 = Currency.of("ZZZ", "Weird", 0, CurrencyType.CRYPTO)
    assert weird1 <= weird2
    assert not (weird2 <= weird1)


# LLM-generated content at query #38
#--------------------------

```python
def test_Currency_quantize():
    # Test with USD (2 decimals)
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert USD.quantize(Decimal("1.005")) == Decimal("1.00")
    assert USD.quantize(Decimal("1.015")) == Decimal("1.02")
    assert USD.quantize(Decimal("1.000")) == Decimal("1.00")
    assert USD.quantize(Decimal("1.999")) == Decimal("2.00")

    # Test with JPY (0 decimals)
    JPY = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert JPY.quantize(Decimal("0.5")) == Decimal("0")
    assert JPY.quantize(Decimal("1.5")) == Decimal("2")
    assert JPY.quantize(Decimal("1.0")) == Decimal("1")
    assert JPY.quantize(Decimal("9.9")) == Decimal("10")

    # Test with a currency with no fixed precision
    ZZZ = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert ZZZ.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert ZZZ.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")
    assert ZZZ.quantize(Decimal("1.0000000000000")) == Decimal("1.000000000000")
    assert ZZZ.quantize(Decimal("1.0000000000009")) == Decimal("1.000000000001")

    # Test with negative values
    assert USD.quantize(Decimal("-1.005")) == Decimal("-1.00")
    assert USD.quantize(Decimal("-1.015")) == Decimal("-1.02")
    assert JPY.quantize(Decimal("-0.5")) == Decimal("0")
    assert JPY.quantize(Decimal("-1.5")) == Decimal("-2")
    assert ZZZ.quantize(Decimal("-1.0000000000005")) == Decimal("-1.000000000000")
    assert ZZZ.quantize(Decimal("-1.0000000000015")) == Decimal("-1.000000000002")


# LLM-generated content at query #39
#--------------------------

```python
def test_CurrencyRegistry___len__():
    registry = CurrencyRegistry()
    assert len(registry) == 0

    with registry:
        registry(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        registry(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))

    assert len(registry) == 2


# LLM-generated content at query #40
#--------------------------

```python
def test_Currency___hash__():
    # Test with USD
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert hash(usd1) == hash(usd2)

    # Test with different currencies
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert hash(usd1) != hash(jpy)

    # Test with different attributes
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert hash(usd1) != hash(usdx)

    # Test with different decimals
    usd3 = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert hash(usd1) != hash(usd3)

    # Test with different types
    usd_crypto = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert hash(usd1) != hash(usd_crypto)


# LLM-generated content at query #41
#--------------------------

```python
def test_CurrencyRegistry___new__():
    # Test singleton behavior
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2

    # Test that the instance is of the correct type
    assert isinstance(registry1, CurrencyRegistry)


# LLM-generated content at query #42
#--------------------------

```python
def test_CurrencyRegistry___contains__():
    # Test with existing currency code
    assert "USD" in Currencies

    # Test with non-existing currency code
    assert "NONEXISTENT" not in Currencies

    # Test with empty string
    assert "" not in Currencies

    # Test with code that has special characters
    assert "ABC123" not in Currencies


# LLM-generated content at query #43
#--------------------------

```python
def test_CurrencyRegistry___contains__():
    # Test with existing currency
    assert "USD" in Currencies

    # Test with non-existing currency
    assert "NONEXISTENT" not in Currencies

    # Test case sensitivity
    assert "usd" not in Currencies
    assert "Usd" not in Currencies

    # Test with empty string
    assert "" not in Currencies

    # Test with non-string type (should not raise exception)
    assert 123 not in Currencies


# LLM-generated content at query #44
#--------------------------

```python
def test_CurrencyRegistry___exit__():
    # Setup
    registry = CurrencyRegistry()
    test_currency = Currency.of("TEST", "Test Currency", 2, CurrencyType.MONEY)

    # Test context manager behavior
    with registry as register:
        register(test_currency)
        # Verify currency was added during context
        assert "TEST" in registry
        assert registry["TEST"] == test_currency

    # Verify post-context behavior
    assert "TEST" in registry  # Should still be registered
    assert registry["TEST"] == test_currency
    assert registry.codes == ["TEST"]  # Should be sorted
    assert registry.codenames == [("TEST", "Test Currency")]  # Should be sorted

    # Test with multiple currencies to verify sorting
    registry2 = CurrencyRegistry()
    with registry2 as register:
        register(Currency.of("ZAR", "South African Rand", 2, CurrencyType.MONEY))
        register(Currency.of("AED", "UAE Dirham", 2, CurrencyType.MONEY))
        register(Currency.of("MAD", "Moroccan Dirham", 2, CurrencyType.MONEY))

    assert registry2.codes == ["AED", "MAD", "ZAR"]
    assert [c.code for c in registry2.all] == ["AED", "MAD", "ZAR"]

    # Test that context is properly closed
    assert not registry._CurrencyRegistry__ctx_open
    assert not registry2._CurrencyRegistry__ctx_open


# LLM-generated content at query #45
#--------------------------

```python
def test_Currency___delattr__():
    with pytest.raises(AttributeError):
        currency = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
        del currency.code


# LLM-generated content at query #46
#--------------------------

```python
def test_CurrencyRegistry___len__():
    registry = CurrencyRegistry()
    assert len(registry) == 0

    with registry as register:
        register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        register(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))

    assert len(registry) == 2


# LLM-generated content at query #47
#--------------------------

```python
def test_CurrencyRegistry_get():
    # Setup
    registry = CurrencyRegistry()
    with registry as register:
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
        register(usd)
        register(eur)

    # Test existing currency
    assert registry.get("USD") == usd
    assert registry.get("EUR") == eur

    # Test non-existing currency with default
    assert registry.get("XYZ") is None
    assert registry.get("XYZ", default=usd) == usd

    # Test non-existing currency without default
    assert registry.get("NON-EXISTING") is None


# LLM-generated content at query #48
#--------------------------

```python
def test_CurrencyLookupError():
    code = "XYZ"
    error = CurrencyLookupError(code)
    assert error.code == code
    assert str(error) == f"Currency identified by code '{code}' does not exist"


# LLM-generated content at query #49
#--------------------------

```python
def test_CurrencyRegistry():
    # Test singleton behavior
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2

    # Test initial state
    assert len(registry1) == 0
    assert registry1.all == []
    assert registry1.codes == []
    assert registry1.codenames == []

    # Test context manager behavior
    with registry1 as register:
        # Test adding a currency
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        register(usd)
        assert len(registry1) == 1
        assert "USD" in registry1
        assert registry1["USD"] == usd

        # Test adding duplicate currency raises error
        with pytest.raises(ValueError, match="Currency USD is already registered."):
            register(usd)

    # Test final state after context
    assert len(registry1) == 1
    assert registry1.all == [usd]
    assert registry1.codes == ["USD"]
    assert registry1.codenames == [("USD", "US Dollar")]

    # Test adding outside context raises error
    with pytest.raises(ProgrammingError, match="Can not create currencies outside registry context."):
        registry1._CurrencyRegistry__register(usd)

    # Test has method
    assert registry1.has("USD")
    assert not registry1.has("EUR")

    # Test get method
    assert registry1.get("USD") == usd
    assert registry1.get("EUR") is None
    assert registry1.get("EUR", default=usd) == usd

    # Test lookup error
    with pytest.raises(CurrencyLookupError, match="Currency identified by code 'EUR' does not exist"):
        registry1["EUR"]


# LLM-generated content at query #50
#--------------------------

```python
def test_Currency___le__():
    # Test equality (should return True)
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 <= usd2

    # Test less than (should return True)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy <= usd

    # Test greater than (should return False)
    assert not (usd <= jpy)

    # Test with different attributes
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 <= usdx)
    assert usdx <= usd1

    # Test with different types
    crypto = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    assert crypto <= usd or usd <= crypto  # Either order is acceptable


# LLM-generated content at query #51
#--------------------------

```python
def test_CurrencyRegistry():
    # Test singleton behavior
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2

    # Test initial state
    assert len(registry1) == 0
    assert registry1.all == []
    assert registry1.codes == []
    assert registry1.codenames == []

    # Test context manager behavior
    with registry1 as register:
        # Test adding currencies
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
        register(usd)
        register(eur)

        # Test that currencies are added
        assert len(registry1) == 2
        assert registry1.all == [usd, eur]
        assert registry1.codes == ["EUR", "USD"]
        assert registry1.codenames == [("EUR", "Euro"), ("USD", "US Dollar")]

    # Test that context is closed
    try:
        register(usd)
        assert False, "Should not be able to register outside context"
    except ProgrammingError:
        pass

    # Test that duplicate currencies are not allowed
    with registry1 as register:
        try:
            register(usd)
            assert False, "Should not be able to register duplicate currency"
        except ValueError:
            pass

    # Test currency lookup
    assert registry1.has("USD")
    assert not registry1.has("XYZ")
    assert registry1["USD"] == usd

    # Test get method
    assert registry1.get("USD") == usd
    assert registry1.get("XYZ") is None
    assert registry1.get("XYZ", default=eur) == eur

    # Test lookup error
    try:
        registry1["XYZ"]
        assert False, "Should raise CurrencyLookupError"
    except CurrencyLookupError as e:
        assert e.code == "XYZ"


# LLM-generated content at query #52
#--------------------------

```python
def test_Currency_quantize():
    # Test with USD (2 decimals)
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert USD.quantize(Decimal("1.005")) == Decimal("1.00")
    assert USD.quantize(Decimal("1.015")) == Decimal("1.02")
    assert USD.quantize(Decimal("1.000")) == Decimal("1.00")
    assert USD.quantize(Decimal("1.001")) == Decimal("1.00")
    assert USD.quantize(Decimal("1.009")) == Decimal("1.01")

    # Test with JPY (0 decimals)
    JPY = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert JPY.quantize(Decimal("0.5")) == Decimal("0")
    assert JPY.quantize(Decimal("1.5")) == Decimal("2")
    assert JPY.quantize(Decimal("1.0")) == Decimal("1")
    assert JPY.quantize(Decimal("1.9")) == Decimal("2")

    # Test with ZZZ (-1 decimals, max precision)
    ZZZ = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert ZZZ.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert ZZZ.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")
    assert ZZZ.quantize(Decimal("1.0000000000000")) == Decimal("1.000000000000")
    assert ZZZ.quantize(Decimal("1.0000000000001")) == Decimal("1.000000000000")

    # Test with negative values
    assert USD.quantize(Decimal("-1.005")) == Decimal("-1.00")
    assert USD.quantize(Decimal("-1.015")) == Decimal("-1.02")
    assert JPY.quantize(Decimal("-0.5")) == Decimal("0")
    assert JPY.quantize(Decimal("-1.5")) == Decimal("-2")
    assert ZZZ.quantize(Decimal("-1.0000000000005")) == Decimal("-1.000000000000")


# LLM-generated content at query #53
#--------------------------

```python
def test_CurrencyRegistry_has():
    # Setup
    registry = CurrencyRegistry()
    with registry as register:
        register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        register(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))

    # Test existing currency
    assert registry.has("USD") is True
    assert registry.has("EUR") is True

    # Test non-existing currency
    assert registry.has("XYZ") is False
    assert registry.has("") is False


# LLM-generated content at query #54
#--------------------------

```python
def test_Currency___delattr__():
    # Create a Currency instance
    currency = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)

    # Attempt to delete an attribute and expect AttributeError
    with pytest.raises(AttributeError):
        del currency.code


# LLM-generated content at query #55
#--------------------------

```python
def test_Currency___eq__():
    # Test equality with same currency
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2

    # Test inequality with different currency code
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert not (usd == eur)

    # Test inequality with different currency name
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars Different", 2, CurrencyType.MONEY)
    assert not (usd1 == usd2)

    # Test inequality with different decimals
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert not (usd1 == usd2)

    # Test inequality with different currency type
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert not (usd1 == usd2)

    # Test inequality with non-Currency object
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd == "USD")
    assert not (usd == 123)
    assert not (usd == None)


# LLM-generated content at query #56
#--------------------------

```python
def test_CurrencyRegistry():
    # Test singleton behavior
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2

    # Test initial state
    assert len(registry1) == 0
    assert registry1.all == []
    assert registry1.codes == []
    assert registry1.codenames == []

    # Test context manager behavior
    with registry1 as register:
        test_currency = Currency.of("TEST", "Test Currency", 2, CurrencyType.MONEY)
        register(test_currency)

    assert len(registry1) == 1
    assert "TEST" in registry1
    assert registry1["TEST"] == test_currency
    assert registry1.all == [test_currency]
    assert registry1.codes == ["TEST"]
    assert registry1.codenames == [("TEST", "Test Currency")]

    # Test duplicate registration
    with pytest.raises(ValueError):
        with registry1 as register:
            register(test_currency)

    # Test registration outside context
    with pytest.raises(ProgrammingError):
        registry1._CurrencyRegistry__register(test_currency)


# LLM-generated content at query #57
#--------------------------

```python
def test_CurrencyRegistry_get():
    # Setup
    registry = CurrencyRegistry()
    with registry as register:
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
        register(usd)
        register(eur)

    # Test existing currency with no default
    assert registry.get("USD") == usd
    assert registry.get("EUR") == eur

    # Test non-existing currency with no default
    assert registry.get("XYZ") is None

    # Test non-existing currency with default
    assert registry.get("XYZ", default=usd) == usd
    assert registry.get("ABC", default=eur) == eur

    # Test existing currency with default (should ignore default)
    assert registry.get("USD", default=eur) == usd
    assert registry.get("EUR", default=usd) == eur


# LLM-generated content at query #58
#--------------------------

```python
def test_CurrencyRegistry___exit__():
    # Setup
    registry = CurrencyRegistry()
    test_currency1 = Currency.of("TEST1", "Test Currency 1", 2, CurrencyType.MONEY)
    test_currency2 = Currency.of("TEST2", "Test Currency 2", 0, CurrencyType.CRYPTO)

    # Test context manager behavior
    with registry as register:
        register(test_currency1)
        register(test_currency2)

    # Verify post-exit state
    assert registry.has("TEST1")
    assert registry.has("TEST2")
    assert len(registry) == 2
    assert registry.codes == ["TEST1", "TEST2"]
    assert registry.codenames == [("TEST1", "Test Currency 1"), ("TEST2", "Test Currency 2")]
    assert registry.all == [test_currency1, test_currency2]

    # Test that registry is properly sorted
    test_currency3 = Currency.of("AAA", "First Currency", 2, CurrencyType.MONEY)
    test_currency4 = Currency.of("ZZZ", "Last Currency", 3, CurrencyType.ALTERNATIVE)

    with registry as register:
        register(test_currency3)
        register(test_currency4)

    assert registry.codes == ["AAA", "TEST1", "TEST2", "ZZZ"]


# LLM-generated content at query #59
#--------------------------

```python
def test_CurrencyRegistry___len__():
    registry = CurrencyRegistry()
    assert len(registry) == 0

    with registry:
        registry(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        registry(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))
        registry(Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY))

    assert len(registry) == 3


# LLM-generated content at query #60
#--------------------------

```python
def test_Currency___lt__():
    # Test with same currency
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 < usd2)
    assert not (usd2 < usd1)

    # Test with different currencies
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert (jpy < usd) or (usd < jpy)  # One must be less than the other

    # Test with different decimals
    crypto1 = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    crypto2 = Currency.of("YYY", "Another weird currency", 0, CurrencyType.CRYPTO)
    assert (crypto1 < crypto2) or (crypto2 < crypto1)  # One must be less than the other

    # Test with different types
    metal = Currency.of("XAU", "Gold", 2, CurrencyType.METAL)
    alt = Currency.of("ABC", "Alternative Currency", 2, CurrencyType.ALTERNATIVE)
    assert (metal < alt) or (alt < metal)  # One must be less than the other


# LLM-generated content at query #61
#--------------------------

```python
def test_CurrencyRegistry___enter__():
    registry = CurrencyRegistry()
    with registry as register:
        assert callable(register)
        assert register.__name__ == "_CurrencyRegistry__register"
    assert not registry._CurrencyRegistry__ctx_open


# LLM-generated content at query #62
#--------------------------

```python
def test_Currency___repr__():
    # Test with USD
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert repr(usd) == "Currency(code='USD', name='US Dollars', decimals=2, type=CurrencyType.MONEY, quantizer=Decimal('0.01'), hashcache=" + str(hash(usd)) + ")"

    # Test with JPY
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert repr(jpy) == "Currency(code='JPY', name='Japanese Yen', decimals=0, type=CurrencyType.MONEY, quantizer=Decimal('0'), hashcache=" + str(hash(jpy)) + ")"

    # Test with a crypto currency
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert repr(zzz) == "Currency(code='ZZZ', name='Some weird currency', decimals=-1, type=CurrencyType.CRYPTO, quantizer=MaxPrecisionQuantizer, hashcache=" + str(hash(zzz)) + ")"


# LLM-generated content at query #63
#--------------------------

```python
def test_Currency___setattr__():
    # Create a currency instance
    currency = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)

    # Test that setting an attribute raises an AttributeError since Currency is frozen
    with pytest.raises(AttributeError):
        currency.code = "EUR"

    with pytest.raises(AttributeError):
        currency.name = "Euros"

    with pytest.raises(AttributeError):
        currency.decimals = 3

    with pytest.raises(AttributeError):
        currency.type = CurrencyType.CRYPTO

    with pytest.raises(AttributeError):
        currency.quantizer = Decimal("0.001")

    with pytest.raises(AttributeError):
        currency.hashcache = 12345


# LLM-generated content at query #64
#--------------------------

```python
def test_Currency___gt__():
    # Test with different codes
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert usd > eur

    # Test with same code but different names
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert usd1 > usd2

    # Test with same code and name but different decimals
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 0, CurrencyType.MONEY)
    assert usd1 > usd2

    # Test with same code, name, and decimals but different types
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert usd1 > usd2

    # Test with same attributes
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 > usd2)


# LLM-generated content at query #65
#--------------------------

```python
def test_Currency___ge__():
    # Test equality cases
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 >= usd2
    assert usd2 >= usd1

    # Test greater than cases (based on code comparison)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert eur >= usd  # 'EUR' < 'USD' in string comparison
    assert not (usd >= eur)

    # Test with different attributes
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert usd >= jpy  # 'USD' > 'JPY' in string comparison

    # Test with different types
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    btc = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    assert btc >= usd  # 'BTC' < 'USD' in string comparison

    # Test with non-Currency object
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd >= "not a currency")
    assert not (usd >= 123)
    assert not (usd >= None)


# LLM-generated content at query #66
#--------------------------

```python
def test_Currency___ge__():
    # Test equality
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 >= usd2
    assert usd2 >= usd1

    # Test greater than
    usd3 = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert usd3 >= usd1
    assert not (usd1 >= usd3)

    # Test different currencies
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert usd1 >= jpy or jpy >= usd1  # Either should be true
    assert not (usd1 >= jpy and jpy >= usd1)  # But not both

    # Test with different types
    btc = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    assert usd1 >= btc or btc >= usd1  # Either should be true
    assert not (usd1 >= btc and btc >= usd1)  # But not both


# LLM-generated content at query #67
#--------------------------

```python
def test_Currency_quantize():
    # Test USD with 2 decimals
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert USD.quantize(Decimal("1.005")) == Decimal("1.00")
    assert USD.quantize(Decimal("1.015")) == Decimal("1.02")
    assert USD.quantize(Decimal("1.000")) == Decimal("1.00")
    assert USD.quantize(Decimal("1.999")) == Decimal("2.00")

    # Test JPY with 0 decimals
    JPY = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert JPY.quantize(Decimal("0.5")) == Decimal("0")
    assert JPY.quantize(Decimal("1.5")) == Decimal("2")
    assert JPY.quantize(Decimal("1.0")) == Decimal("1")
    assert JPY.quantize(Decimal("2.9")) == Decimal("3")

    # Test ZZZ with -1 decimals (no fixed precision)
    ZZZ = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert ZZZ.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert ZZZ.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")
    assert ZZZ.quantize(Decimal("1.0000000000000")) == Decimal("1.000000000000")
    assert ZZZ.quantize(Decimal("1.0000000000009")) == Decimal("1.000000000001")

    # Test negative values
    assert USD.quantize(Decimal("-1.005")) == Decimal("-1.00")
    assert JPY.quantize(Decimal("-1.5")) == Decimal("-2")
    assert ZZZ.quantize(Decimal("-1.0000000000005")) == Decimal("-1.000000000000")

    # Test zero
    assert USD.quantize(Decimal("0")) == Decimal("0.00")
    assert JPY.quantize(Decimal("0")) == Decimal("0")
    assert ZZZ.quantize(Decimal("0")) == Decimal("0")


# LLM-generated content at query #68
#--------------------------

```python
def test_CurrencyRegistry_get():
    # Setup
    registry = CurrencyRegistry()
    with registry:
        registry(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        registry(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))

    # Test existing currency with no default
    assert registry.get("USD").code == "USD"
    assert registry.get("USD").name == "US Dollar"

    # Test non-existing currency with no default
    assert registry.get("XYZ") is None

    # Test non-existing currency with default
    default_currency = Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY)
    assert registry.get("XYZ", default=default_currency) == default_currency

    # Test existing currency with default (should ignore default)
    assert registry.get("EUR", default=default_currency).code == "EUR"


# LLM-generated content at query #69
#--------------------------

```python
def test_Currency___gt__():
    # Test with different codes
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert usd > eur

    # Test with same code but different names
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars 2", 2, CurrencyType.MONEY)
    assert usd1 > usd2

    # Test with same code and name but different decimals
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert usd1 > usd2

    # Test with same code, name, and decimals but different types
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert usd1 > usd2

    # Test with same attributes (should be False)
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 > usd2)


# LLM-generated content at query #70
#--------------------------

```python
def test_CurrencyRegistry___contains__():
    # Test with existing currency
    assert "USD" in Currencies

    # Test with non-existing currency
    assert "NONEXISTENT" not in Currencies

    # Test case sensitivity
    assert "usd" not in Currencies
    assert "Usd" not in Currencies

    # Test with empty string
    assert "" not in Currencies

    # Test with special characters
    assert "USD!" not in Currencies
    assert "USD123" not in Currencies


# LLM-generated content at query #71
#--------------------------

```python
def test_CurrencyRegistry___getitem__():
    # Setup
    registry = CurrencyRegistry()
    with registry as register:
        register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        register(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))

    # Test existing currency
    usd = registry["USD"]
    assert usd.code == "USD"
    assert usd.name == "US Dollar"
    assert usd.decimals == 2
    assert usd.type == CurrencyType.MONEY

    # Test non-existing currency
    with pytest.raises(CurrencyLookupError) as excinfo:
        _ = registry["NON-EXISTING"]
    assert str(excinfo.value) == "Currency identified by code 'NON-EXISTING' does not exist"


# LLM-generated content at query #72
#--------------------------

```python
def test_CurrencyRegistry_has():
    # Test with existing currency
    assert Currencies.has("USD") is True

    # Test with non-existing currency
    assert Currencies.has("NONEXISTING") is False

    # Test with empty string
    assert Currencies.has("") is False

    # Test with non-string input (should return False)
    assert Currencies.has(123) is False


# LLM-generated content at query #73
#--------------------------

```python
def test_CurrencyRegistry___new__():
    # Test singleton behavior
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2

    # Test that the registry is empty initially
    assert len(registry1) == 0
    assert registry1.all == []
    assert registry1.codes == []
    assert registry1.codenames == []

    # Test that the context is closed initially
    with pytest.raises(ProgrammingError, match="Can not create currencies outside registry context."):
        registry1._CurrencyRegistry__register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))


# LLM-generated content at query #74
#--------------------------

```python
def test_CurrencyRegistry_has():
    # Setup
    registry = CurrencyRegistry()
    with registry as register:
        register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        register(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))

    # Test existing currency
    assert registry.has("USD") is True
    assert registry.has("EUR") is True

    # Test non-existing currency
    assert registry.has("XYZ") is False
    assert registry.has("NONEXISTENT") is False

    # Test case sensitivity
    assert registry.has("usd") is False
    assert registry.has("eur") is False


# LLM-generated content at query #75
#--------------------------

```python
def test_CurrencyRegistry___len__():
    # Create a new instance of CurrencyRegistry
    registry = CurrencyRegistry()

    # Initially, the registry should be empty
    assert len(registry) == 0

    # Populate the registry with some currencies
    with registry as register:
        register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        register(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))
        register(Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY))

    # After adding 3 currencies, the length should be 3
    assert len(registry) == 3

    # Create a new instance to ensure singleton behavior
    another_registry = CurrencyRegistry()

    # The new instance should have the same length as the previous one
    assert len(another_registry) == 3


# LLM-generated content at query #76
#--------------------------

```python
def test_Currency___ge__():
    # Test equality
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 >= usd2
    assert usd2 >= usd1

    # Test greater than
    usd3 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd4 = Currency.of("USDA", "US Dollars A", 2, CurrencyType.MONEY)
    assert usd4 >= usd3
    assert not (usd3 >= usd4)

    # Test with different types
    usd5 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy >= usd5
    assert not (usd5 >= jpy)

    # Test with different decimals
    usd6 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    crypto = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert crypto >= usd6
    assert not (usd6 >= crypto)


# LLM-generated content at query #77
#--------------------------

```python
def test_Currency():
    # Test valid currency creation
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.code == "USD"
    assert usd.name == "US Dollars"
    assert usd.decimals == 2
    assert usd.type == CurrencyType.MONEY
    assert usd.quantize(Decimal("1.005")) == Decimal("1.00")
    assert usd.quantize(Decimal("1.015")) == Decimal("1.02")

    # Test currency with zero decimals
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy.code == "JPY"
    assert jpy.name == "Japanese Yen"
    assert jpy.decimals == 0
    assert jpy.type == CurrencyType.MONEY
    assert jpy.quantize(Decimal("0.5")) == Decimal("0")
    assert jpy.quantize(Decimal("1.5")) == Decimal("2")

    # Test currency with variable precision
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert zzz.code == "ZZZ"
    assert zzz.name == "Some weird currency"
    assert zzz.decimals == -1
    assert zzz.type == CurrencyType.CRYPTO
    assert zzz.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert zzz.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")

    # Test equality and hash
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2
    assert not (usd1 == usdx)
    assert hash(usd1) == hash(usd2)
    assert not (hash(usd1) == hash(usdx))

    # Test invalid currency code
    with pytest.raises(ProgrammingError):
        Currency.of("usd", "US Dollars", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD1", "US Dollars", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of(123, "US Dollars", 2, CurrencyType.MONEY)

    # Test invalid currency name
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "  US Dollars", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars  ", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", 123, 2, CurrencyType.MONEY)

    # Test invalid decimals
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", -2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", "2", CurrencyType.MONEY)

    # Test invalid currency type
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", 2, "MONEY")


# LLM-generated content at query #78
#--------------------------

```python
def test_CurrencyRegistry___exit__():
    # Test that __exit__ correctly finalizes the registry after population
    registry = CurrencyRegistry()
    assert registry.__ctx_open is False
    assert len(registry) == 0

    # Enter context and add a currency
    with registry as register:
        test_currency = Currency.of("TEST", "Test Currency", 2, CurrencyType.MONEY)
        register(test_currency)
        assert registry.__ctx_open is True
        assert len(registry) == 1

    # After exiting context, verify finalization
    assert registry.__ctx_open is False
    assert len(registry) == 1
    assert registry["TEST"] == test_currency
    assert registry.codes == ["TEST"]
    assert registry.codenames == [("TEST", "Test Currency")]

    # Test that adding multiple currencies maintains sorting
    with registry as register:
        currency_a = Currency.of("AED", "UAE Dirham", 2, CurrencyType.MONEY)
        currency_b = Currency.of("BHD", "Bahraini Dinar", 3, CurrencyType.MONEY)
        register(currency_a)
        register(currency_b)

    assert registry.codes == ["AED", "BHD", "TEST"]
    assert registry.codenames == [
        ("AED", "UAE Dirham"),
        ("BHD", "Bahraini Dinar"),
        ("TEST", "Test Currency")
    ]

    # Test that __exit__ handles exceptions properly
    with pytest.raises(ValueError):
        with registry as register:
            duplicate_currency = Currency.of("AED", "Duplicate", 2, CurrencyType.MONEY)
            register(duplicate_currency)

    # Verify registry remains consistent after exception
    assert registry.codes == ["AED", "BHD", "TEST"]


# LLM-generated content at query #79
#--------------------------

```python
def test_Currency___le__():
    # Test less than
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    assert usd <= eur
    assert not (eur <= usd)

    # Test equal
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 <= usd2
    assert usd2 <= usd1

    # Test greater than
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert not (usd <= jpy)
    assert jpy <= usd

    # Test with different types
    btc = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    assert usd <= btc
    assert not (btc <= usd)


# LLM-generated content at query #80
#--------------------------

```python
def test_Currency___hash__():
    # Test that hash returns the pre-computed hash
    currency1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    currency2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    currency3 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)

    assert hash(currency1) == currency1.hashcache
    assert hash(currency2) == currency2.hashcache
    assert hash(currency3) == currency3.hashcache

    # Test that equal currencies have the same hash
    assert hash(currency1) == hash(currency2)

    # Test that different currencies have different hashes
    assert hash(currency1) != hash(currency3)


# LLM-generated content at query #81
#--------------------------

```python
def test_CurrencyRegistry___getitem__():
    # Test getting an existing currency
    currency = CurrencyRegistry()["USD"]
    assert currency.code == "USD"
    assert currency.name == "US Dollar"
    assert currency.type == CurrencyType.MONEY

    # Test getting a non-existing currency
    with pytest.raises(CurrencyLookupError) as excinfo:
        CurrencyRegistry()["NON-EXISTING"]
    assert str(excinfo.value) == "Currency identified by code 'NON-EXISTING' does not exist"


# LLM-generated content at query #82
#--------------------------

```python
def test_Currency___gt__():
    # Test with same currency
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 > usd2)

    # Test with different currency codes
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert usd > eur or eur > usd  # One should be greater than the other

    # Test with different names
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert usd1 > usd2 or usd2 > usd1  # One should be greater than the other

    # Test with different decimals
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert usd > jpy or jpy > usd  # One should be greater than the other

    # Test with different types
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    btc = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    assert usd > btc or btc > usd  # One should be greater than the other


# LLM-generated content at query #83
#--------------------------

```python
def test_Currency___eq__():
    # Test equality with same currency
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2

    # Test inequality with different currency code
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    assert not (usd == eur)

    # Test inequality with different currency name
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 == usd2)

    # Test inequality with different decimals
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 0, CurrencyType.MONEY)
    assert not (usd1 == usd2)

    # Test inequality with different currency type
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert not (usd1 == usd2)

    # Test inequality with non-Currency object
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd == "USD")
    assert not (usd == 123)
    assert not (usd == None)


# LLM-generated content at query #84
#--------------------------

```python
def test_Currency():
    # Test valid currency creation
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.code == "USD"
    assert usd.name == "US Dollars"
    assert usd.decimals == 2
    assert usd.type == CurrencyType.MONEY
    assert usd.quantize(Decimal("1.005")) == Decimal("1.00")
    assert usd.quantize(Decimal("1.015")) == Decimal("1.02")

    # Test currency with zero decimals
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy.code == "JPY"
    assert jpy.name == "Japanese Yen"
    assert jpy.decimals == 0
    assert jpy.type == CurrencyType.MONEY
    assert jpy.quantize(Decimal("0.5")) == Decimal("0")
    assert jpy.quantize(Decimal("1.5")) == Decimal("2")

    # Test currency with variable precision
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert zzz.code == "ZZZ"
    assert zzz.name == "Some weird currency"
    assert zzz.decimals == -1
    assert zzz.type == CurrencyType.CRYPTO
    assert zzz.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert zzz.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")

    # Test equality and hash
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2
    assert not (usd1 == usdx)
    assert hash(usd1) == hash(usd2)
    assert not (hash(usd1) == hash(usdx))

    # Test invalid currency code
    with pytest.raises(ProgrammingError):
        Currency.of("123", "Invalid Code", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("usd", "Lowercase Code", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD$", "Special Char Code", 2, CurrencyType.MONEY)

    # Test invalid currency name
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "  Trimmed  ", 2, CurrencyType.MONEY)

    # Test invalid decimals
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", -2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", "2", CurrencyType.MONEY)

    # Test invalid currency type
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", 2, "MONEY")


# LLM-generated content at query #85
#--------------------------

```python
def test_Currency___le__():
    # Test equality (should return True)
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 <= usd2
    assert usd2 <= usd1

    # Test less than (should return True)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert usd <= eur or eur <= usd  # At least one should be True

    # Test greater than (should return False)
    if usd <= eur:
        assert not (eur <= usd)
    else:
        assert not (usd <= eur)

    # Test with different attributes
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert not (usd <= usdx) or not (usdx <= usd)  # Should not be equal


# LLM-generated content at query #86
#--------------------------

```python
def test_CurrencyRegistry___len__():
    registry = CurrencyRegistry()
    assert len(registry) == 0

    with registry as register:
        register(Currency.of("AED", "UAE Dirham", 2, CurrencyType.MONEY))
        register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))

    assert len(registry) == 2


# LLM-generated content at query #87
#--------------------------

```python
def test_Currency___repr__():
    currency = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert repr(currency) == "Currency(code='USD', name='US Dollars', decimals=2, type=CurrencyType.MONEY, quantizer=Decimal('0.01'), hashcache={})".format(hash(('USD', 'US Dollars', 2, CurrencyType.MONEY, Decimal('0.01'))))


# LLM-generated content at query #88
#--------------------------

```python
def test_CurrencyRegistry():
    # Test singleton behavior
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2

    # Test initial state
    assert len(registry1) == 0
    assert registry1.all == []
    assert registry1.codes == []
    assert registry1.codenames == []

    # Test context manager behavior
    with registry1 as register:
        # Test adding currencies in context
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        register(usd)
        assert len(registry1) == 1
        assert "USD" in registry1
        assert registry1["USD"] == usd

        # Test adding duplicate currency raises error
        with pytest.raises(ValueError, match="Currency USD is already registered."):
            register(usd)

    # Test final state after context
    assert len(registry1) == 1
    assert registry1.all == [usd]
    assert registry1.codes == ["USD"]
    assert registry1.codenames == [("USD", "US Dollar")]

    # Test adding outside context raises error
    with pytest.raises(ProgrammingError, match="Can not create currencies outside registry context."):
        registry1._CurrencyRegistry__register(usd)

    # Test lookup methods
    assert registry1.has("USD") is True
    assert registry1.has("XYZ") is False
    assert registry1.get("USD") == usd
    assert registry1.get("XYZ") is None
    assert registry1.get("XYZ", default=usd) == usd

    # Test lookup error
    with pytest.raises(CurrencyLookupError, match="Currency identified by code 'XYZ' does not exist"):
        registry1["XYZ"]


# LLM-generated content at query #89
#--------------------------

```python
def test_CurrencyRegistry_get():
    # Setup
    registry = CurrencyRegistry()
    with registry as register:
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
        register(usd)
        register(eur)

    # Test existing currency with no default
    assert registry.get("USD") == usd

    # Test existing currency with default
    assert registry.get("EUR", default=usd) == eur

    # Test non-existing currency with no default
    assert registry.get("XYZ") is None

    # Test non-existing currency with default
    assert registry.get("XYZ", default=usd) == usd


# LLM-generated content at query #90
#--------------------------

```python
def test_Currency___le__():
    # Test equality cases
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 <= usd2
    assert usd2 <= usd1

    # Test less than cases
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy <= usd
    assert not (usd <= jpy)

    # Test with different attributes
    usd_high = Currency.of("USD", "US Dollars High", 3, CurrencyType.MONEY)
    usd_low = Currency.of("USD", "US Dollars Low", 1, CurrencyType.MONEY)
    assert usd_low <= usd_high
    assert not (usd_high <= usd_low)

    # Test with different types
    usd_money = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd_crypto = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert usd_crypto <= usd_money
    assert not (usd_money <= usd_crypto)

    # Test with non-Currency object
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd <= "not a currency")


# LLM-generated content at query #91
#--------------------------

```python
def test_Currency___delattr__():
    # Create a currency instance
    currency = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)

    # Since Currency is a frozen dataclass, attempting to delete an attribute should raise an AttributeError
    with pytest.raises(AttributeError):
        del currency.code

    with pytest.raises(AttributeError):
        del currency.name

    with pytest.raises(AttributeError):
        del currency.decimals

    with pytest.raises(AttributeError):
        del currency.type

    with pytest.raises(AttributeError):
        del currency.quantizer

    with pytest.raises(AttributeError):
        del currency.hashcache


# LLM-generated content at query #92
#--------------------------

```python
def test_CurrencyRegistry_has():
    registry = CurrencyRegistry()

    # Test with a non-existing currency
    assert not registry.has("NONEXISTENT")

    # Test with an existing currency (assuming USD is registered)
    with registry as register:
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        register(usd)

    assert registry.has("USD")
    assert not registry.has("EUR")


# LLM-generated content at query #93
#--------------------------

```python
def test_CurrencyRegistry_get():
    # Setup
    registry = CurrencyRegistry()
    with registry as register:
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
        register(usd)
        register(eur)

    # Test existing currency with no default
    assert registry.get("USD") == usd
    assert registry.get("EUR") == eur

    # Test non-existing currency with no default
    assert registry.get("XYZ") is None

    # Test non-existing currency with default
    assert registry.get("XYZ", default=usd) == usd
    assert registry.get("ABC", default=eur) == eur

    # Test with CurrencyLookupError when no default is provided (implicitly tested above)
    # The method should return None instead of raising an error when default is not provided


# LLM-generated content at query #94
#--------------------------

```python
def test_Currency___gt__():
    # Test with different codes
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert usd > eur
    assert not (eur > usd)

    # Test with same code but different names
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars 2", 2, CurrencyType.MONEY)
    assert not (usd1 > usd2)
    assert not (usd2 > usd1)

    # Test with same code and name but different decimals
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert usd1 > usd2
    assert not (usd2 > usd1)

    # Test with same code, name, and decimals but different types
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert usd1 > usd2
    assert not (usd2 > usd1)

    # Test with same code, name, decimals, and type
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 > usd2)
    assert not (usd2 > usd1)


# LLM-generated content at query #95
#--------------------------

```python
def test_Currency___gt__():
    # Test with different codes
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert usd > eur
    assert not (eur > usd)

    # Test with same code but different names
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    assert not (usd1 > usd2)
    assert not (usd2 > usd1)

    # Test with same code and name but different decimals
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert usd1 > usd2
    assert not (usd2 > usd1)

    # Test with same code, name, and decimals but different types
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert usd1 > usd2
    assert not (usd2 > usd1)

    # Test with all attributes same
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 > usd2)
    assert not (usd2 > usd1)


# LLM-generated content at query #96
#--------------------------

```python
def test_CurrencyRegistry___len__():
    # Create a new instance of CurrencyRegistry
    registry = CurrencyRegistry()

    # Enter the context to add currencies
    with registry:
        registry(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        registry(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))
        registry(Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY))

    # Test the length
    assert len(registry) == 3


# LLM-generated content at query #97
#--------------------------

```python
def test_Currency___gt__():
    # Test with different codes
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert usd > eur

    # Test with same code but different names
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars 2", 2, CurrencyType.MONEY)
    assert usd1 > usd2

    # Test with same code and name but different decimals
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert usd1 > usd2

    # Test with same code, name, and decimals but different types
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert usd1 > usd2

    # Test with same attributes (should be False)
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 > usd2)

    # Test with non-Currency object (should raise TypeError)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    try:
        usd > "not a currency"
        assert False, "Expected TypeError"
    except TypeError:
        pass


# LLM-generated content at query #98
#--------------------------

```python
def test_CurrencyRegistry___enter__():
    # Test that __enter__ returns the __register method
    registry = CurrencyRegistry()
    with registry as register:
        assert callable(register)
        assert register.__name__ == "_CurrencyRegistry__register"

    # Test that __enter__ sets __ctx_open to True
    registry = CurrencyRegistry()
    assert not registry._CurrencyRegistry__ctx_open
    with registry:
        assert registry._CurrencyRegistry__ctx_open
    assert not registry._CurrencyRegistry__ctx_open


# LLM-generated content at query #99
#--------------------------

```python
def test_Currency_quantize():
    # Test with USD (2 decimals)
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert USD.quantize(Decimal("1.005")) == Decimal("1.00")
    assert USD.quantize(Decimal("1.015")) == Decimal("1.02")
    assert USD.quantize(Decimal("1.000")) == Decimal("1.00")
    assert USD.quantize(Decimal("1.001")) == Decimal("1.00")
    assert USD.quantize(Decimal("1.009")) == Decimal("1.01")

    # Test with JPY (0 decimals)
    JPY = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert JPY.quantize(Decimal("0.5")) == Decimal("0")
    assert JPY.quantize(Decimal("1.5")) == Decimal("2")
    assert JPY.quantize(Decimal("1.0")) == Decimal("1")
    assert JPY.quantize(Decimal("1.9")) == Decimal("2")
    assert JPY.quantize(Decimal("0.0")) == Decimal("0")

    # Test with a currency with no fixed precision (CRYPTO)
    ZZZ = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert ZZZ.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert ZZZ.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")
    assert ZZZ.quantize(Decimal("1.0000000000000")) == Decimal("1.000000000000")
    assert ZZZ.quantize(Decimal("1.0000000000001")) == Decimal("1.000000000000")
    assert ZZZ.quantize(Decimal("1.0000000000009")) == Decimal("1.000000000001")

    # Test edge cases
    EUR = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert EUR.quantize(Decimal("0.001")) == Decimal("0.00")
    assert EUR.quantize(Decimal("0.009")) == Decimal("0.01")
    assert EUR.quantize(Decimal("999.999")) == Decimal("1000.00")
    assert EUR.quantize(Decimal("-1.001")) == Decimal("-1.00")
    assert EUR.quantize(Decimal("-1.009")) == Decimal("-1.01")


# LLM-generated content at query #100
#--------------------------

```python
def test_Currency_quantize():
    # Test USD with 2 decimals
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert USD.quantize(Decimal("1.005")) == Decimal("1.00")
    assert USD.quantize(Decimal("1.015")) == Decimal("1.02")
    assert USD.quantize(Decimal("1.000")) == Decimal("1.00")
    assert USD.quantize(Decimal("1.999")) == Decimal("2.00")

    # Test JPY with 0 decimals
    JPY = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert JPY.quantize(Decimal("0.5")) == Decimal("0")
    assert JPY.quantize(Decimal("1.5")) == Decimal("2")
    assert JPY.quantize(Decimal("1.0")) == Decimal("1")
    assert JPY.quantize(Decimal("9.9")) == Decimal("10")

    # Test ZZZ with -1 decimals (no fixed precision)
    ZZZ = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert ZZZ.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert ZZZ.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")
    assert ZZZ.quantize(Decimal("1.0000000000000")) == Decimal("1.000000000000")
    assert ZZZ.quantize(Decimal("1.0000000000001")) == Decimal("1.000000000000")

    # Test negative values
    assert USD.quantize(Decimal("-1.005")) == Decimal("-1.00")
    assert USD.quantize(Decimal("-1.015")) == Decimal("-1.02")
    assert JPY.quantize(Decimal("-0.5")) == Decimal("0")
    assert JPY.quantize(Decimal("-1.5")) == Decimal("-2")
    assert ZZZ.quantize(Decimal("-1.0000000000005")) == Decimal("-1.000000000000")
    assert ZZZ.quantize(Decimal("-1.0000000000015")) == Decimal("-1.000000000002")

    # Test zero
    assert USD.quantize(Decimal("0")) == Decimal("0.00")
    assert JPY.quantize(Decimal("0")) == Decimal("0")
    assert ZZZ.quantize(Decimal("0")) == Decimal("0")


# LLM-generated content at query #101
#--------------------------

```python
def test_CurrencyRegistry_has():
    # Setup
    registry = CurrencyRegistry()
    with registry as register:
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
        register(usd)
        register(eur)

    # Test existing currency
    assert registry.has("USD") is True
    assert registry.has("EUR") is True

    # Test non-existing currency
    assert registry.has("XYZ") is False
    assert registry.has("") is False


# LLM-generated content at query #102
#--------------------------

```python
def test_CurrencyRegistry___new__():
    # Test that CurrencyRegistry is a singleton
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2

    # Test that the singleton instance is created only once
    assert CurrencyRegistry._CurrencyRegistry__instance is not None
    assert CurrencyRegistry._CurrencyRegistry__instance is registry1
    assert CurrencyRegistry._CurrencyRegistry__instance is registry2


# LLM-generated content at query #103
#--------------------------

```python
def test_Currency():
    # Test valid currency creation
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.code == "USD"
    assert usd.name == "US Dollars"
    assert usd.decimals == 2
    assert usd.type == CurrencyType.MONEY
    assert usd.quantize(Decimal("1.005")) == Decimal("1.00")
    assert usd.quantize(Decimal("1.015")) == Decimal("1.02")

    # Test currency with zero decimals
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy.code == "JPY"
    assert jpy.name == "Japanese Yen"
    assert jpy.decimals == 0
    assert jpy.type == CurrencyType.MONEY
    assert jpy.quantize(Decimal("0.5")) == Decimal("0")
    assert jpy.quantize(Decimal("1.5")) == Decimal("2")

    # Test currency with variable precision
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert zzz.code == "ZZZ"
    assert zzz.name == "Some weird currency"
    assert zzz.decimals == -1
    assert zzz.type == CurrencyType.CRYPTO
    assert zzz.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert zzz.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")

    # Test equality and hash
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2
    assert usd1 != usdx
    assert hash(usd1) == hash(usd2)
    assert hash(usd1) != hash(usdx)

    # Test invalid currency code
    with pytest.raises(ProgrammingError):
        Currency.of("123", "Invalid Code", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("usd", "Lowercase Code", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD!", "Special Char Code", 2, CurrencyType.MONEY)

    # Test invalid currency name
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "  Trimmed  ", 2, CurrencyType.MONEY)

    # Test invalid decimals
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", -2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", "2", CurrencyType.MONEY)

    # Test invalid currency type
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", 2, "MONEY")


# LLM-generated content at query #104
#--------------------------

```python
def test_Currency___le__():
    # Test less than
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert usd <= jpy or jpy <= usd

    # Test equal
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 <= usd2
    assert usd2 <= usd1

    # Test greater than
    crypto = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert usd <= crypto or crypto <= usd


# LLM-generated content at query #105
#--------------------------

```python
def test_Currency___ge__():
    # Test equality case
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 >= usd2
    assert usd2 >= usd1

    # Test greater than case
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert usd >= jpy
    assert not (jpy >= usd)

    # Test less than case
    assert not (jpy >= usd)

    # Test with different currency types
    gold = Currency.of("XAU", "Gold", 2, CurrencyType.METAL)
    btc = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    assert gold >= btc or btc >= gold  # One should be greater
    assert not (gold >= btc and btc >= gold)  # They shouldn't be equal

    # Test with different decimals
    weird1 = Currency.of("ZZZ", "Weird Currency 1", -1, CurrencyType.CRYPTO)
    weird2 = Currency.of("YYY", "Weird Currency 2", 5, CurrencyType.CRYPTO)
    assert weird1 >= weird2 or weird2 >= weird1
    assert not (weird1 >= weird2 and weird2 >= weird1)


# LLM-generated content at query #106
#--------------------------

```python
def test_CurrencyRegistry_get():
    registry = CurrencyRegistry()
    with registry as register:
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
        register(usd)
        register(eur)

    # Test getting existing currency
    assert registry.get("USD") == usd
    assert registry.get("EUR") == eur

    # Test getting non-existing currency without default
    assert registry.get("XYZ") is None

    # Test getting non-existing currency with default
    assert registry.get("XYZ", default=eur) == eur


# LLM-generated content at query #107
#--------------------------

```python
def test_Currency___gt__():
    # Test with same currency
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 > usd2)

    # Test with different currencies
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert usd > jpy or jpy > usd  # One should be greater than the other

    # Test with different decimals
    crypto1 = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    crypto2 = Currency.of("YYY", "Some other weird currency", -1, CurrencyType.CRYPTO)
    assert crypto1 > crypto2 or crypto2 > crypto1  # One should be greater than the other

    # Test with different types
    metal = Currency.of("XAU", "Gold", 2, CurrencyType.METAL)
    alt = Currency.of("ALT", "Alternative Currency", 2, CurrencyType.ALTERNATIVE)
    assert metal > alt or alt > metal  # One should be greater than the other


# LLM-generated content at query #108
#--------------------------

```python
def test_CurrencyRegistry___len__():
    registry = CurrencyRegistry()
    assert len(registry) == 0

    with registry as register:
        register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        register(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))

    assert len(registry) == 2


# LLM-generated content at query #109
#--------------------------

```python
def test_Currency___gt__():
    # Test with different codes
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert usd > eur

    # Test with same code but different names
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars 2", 2, CurrencyType.MONEY)
    assert usd1 > usd2

    # Test with same code and name but different decimals
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert usd1 > usd2

    # Test with same code, name, and decimals but different types
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert usd1 > usd2

    # Test with same attributes (should be False)
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 > usd2)


# LLM-generated content at query #110
#--------------------------

```python
def test_CurrencyRegistry_has():
    # Test with existing currency
    assert Currencies.has("USD") is True

    # Test with non-existing currency
    assert Currencies.has("NONEXISTENT") is False

    # Test with empty string
    assert Currencies.has("") is False

    # Test with code that has special characters
    assert Currencies.has("USD!") is False

    # Test with code that has spaces
    assert Currencies.has("USD ") is False


# LLM-generated content at query #111
#--------------------------

```python
def test_Currency___lt__():
    # Test with different codes
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert (usd < eur) == ("USD" < "EUR")

    # Test with same code but different names
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars 2", 2, CurrencyType.MONEY)
    assert not (usd1 < usd2) and not (usd2 < usd1)

    # Test with same code and name but different decimals
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert (usd1 < usd2) == (2 < 3)

    # Test with same code, name, decimals but different types
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert (usd1 < usd2) == (CurrencyType.MONEY < CurrencyType.CRYPTO)

    # Test with same attributes (should be False)
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 < usd2)

    # Test with different quantizers (indirectly through decimals)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert (jpy < usd) == (0 < 2)


# LLM-generated content at query #112
#--------------------------

```python
def test_CurrencyRegistry_get():
    # Setup
    registry = CurrencyRegistry()
    with registry as register:
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
        register(usd)
        register(eur)

    # Test with existing currency code
    assert registry.get("USD") == usd
    assert registry.get("EUR") == eur

    # Test with non-existing currency code and no default
    assert registry.get("XYZ") is None

    # Test with non-existing currency code and default
    assert registry.get("XYZ", default=usd) == usd


# LLM-generated content at query #113
#--------------------------

```python
def test_Currency___lt__():
    # Test with same currency
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 < usd2)
    assert not (usd2 < usd1)

    # Test with different currencies
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert (jpy < usd) or (usd < jpy)  # One must be less than the other

    # Test with different attributes
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert (usdx < usd) or (usd < usdx)  # One must be less than the other

    # Test with different decimal precision
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert (zzz < usd) or (usd < zzz)  # One must be less than the other

    # Test with different currency types
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    gold = Currency.of("XAU", "Gold", 2, CurrencyType.METAL)
    assert (gold < usd) or (usd < gold)  # One must be less than the other


# LLM-generated content at query #114
#--------------------------

```python
def test_CurrencyLookupError():
    # Test initialization and message
    error = CurrencyLookupError("XYZ")
    assert error.code == "XYZ"
    assert str(error) == "Currency identified by code 'XYZ' does not exist"

    # Test inheritance
    assert isinstance(error, LookupError)

    # Test with different code
    error2 = CurrencyLookupError("ABC")
    assert error2.code == "ABC"
    assert str(error2) == "Currency identified by code 'ABC' does not exist"


# LLM-generated content at query #115
#--------------------------

```python
def test_CurrencyLookupError():
    code = "XYZ"
    error = CurrencyLookupError(code)
    assert error.code == code
    assert str(error) == f"Currency identified by code '{code}' does not exist"


# LLM-generated content at query #116
#--------------------------

```python
def test_Currency___gt__():
    # Test with different codes
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    assert (usd > eur) == ("USD" > "EUR")

    # Test with same code but different names
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars 2", 2, CurrencyType.MONEY)
    assert (usd1 > usd2) == (("USD", "US Dollars", 2, CurrencyType.MONEY, usd1.quantizer) >
                            ("USD", "US Dollars 2", 2, CurrencyType.MONEY, usd2.quantizer))

    # Test with same code and name but different decimals
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert (usd1 > usd2) == (("USD", "US Dollars", 2, CurrencyType.MONEY, usd1.quantizer) >
                            ("USD", "US Dollars", 3, CurrencyType.MONEY, usd2.quantizer))

    # Test with same code, name, and decimals but different types
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert (usd1 > usd2) == (("USD", "US Dollars", 2, CurrencyType.MONEY, usd1.quantizer) >
                            ("USD", "US Dollars", 2, CurrencyType.CRYPTO, usd2.quantizer))

    # Test with same code, name, decimals, and type but different quantizers
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", -1, CurrencyType.MONEY)
    assert (usd1 > usd2) == (("USD", "US Dollars", 2, CurrencyType.MONEY, usd1.quantizer) >
                            ("USD", "US Dollars", -1, CurrencyType.MONEY, usd2.quantizer))

    # Test with equal currencies
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 > usd2)


# LLM-generated content at query #117
#--------------------------

```python
def test_CurrencyRegistry___contains__():
    # Setup
    registry = CurrencyRegistry()
    with registry as register:
        register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        register(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))

    # Test existing currency
    assert "USD" in registry
    assert "EUR" in registry

    # Test non-existing currency
    assert "XYZ" not in registry
    assert "NONEXISTENT" not in registry


# LLM-generated content at query #118
#--------------------------

```python
def test_CurrencyRegistry_get():
    # Setup
    registry = CurrencyRegistry()
    with registry:
        registry(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        registry(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))

    # Test existing currency
    usd = registry.get("USD")
    assert usd is not None
    assert usd.code == "USD"
    assert usd.name == "US Dollar"

    # Test non-existing currency without default
    assert registry.get("XYZ") is None

    # Test non-existing currency with default
    default_currency = registry.get("XYZ", default=registry["USD"])
    assert default_currency == registry["USD"]


# LLM-generated content at query #119
#--------------------------

```python
def test_CurrencyRegistry():
    # Test singleton behavior
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2

    # Test initial state
    assert len(registry1) == 0
    assert registry1.all == []
    assert registry1.codes == []
    assert registry1.codenames == []

    # Test context manager behavior
    with registry1 as register:
        # Test adding a currency
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        register(usd)

        # Test that currency is not yet available (context not closed)
        assert "USD" not in registry1
        assert not registry1.has("USD")

    # After context closes, test that currency is now available
    assert "USD" in registry1
    assert registry1.has("USD")
    assert registry1["USD"] == usd
    assert len(registry1) == 1
    assert registry1.all == [usd]
    assert registry1.codes == ["USD"]
    assert registry1.codenames == [("USD", "US Dollar")]

    # Test adding duplicate currency raises error
    with pytest.raises(ValueError):
        with registry1 as register:
            register(usd)

    # Test adding currency outside context raises error
    with pytest.raises(ProgrammingError):
        registry1._CurrencyRegistry__register(usd)

    # Test get method
    assert registry1.get("USD") == usd
    assert registry1.get("XYZ") is None
    assert registry1.get("XYZ", default=usd) == usd

    # Test lookup error
    with pytest.raises(CurrencyLookupError):
        registry1["XYZ"]


# LLM-generated content at query #120
#--------------------------

```python
def test_Currency():
    # Test valid currency creation
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.code == "USD"
    assert usd.name == "US Dollars"
    assert usd.decimals == 2
    assert usd.type == CurrencyType.MONEY
    assert usd.quantize(Decimal("1.005")) == Decimal("1.00")
    assert usd.quantize(Decimal("1.015")) == Decimal("1.02")

    # Test currency with zero decimals
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy.code == "JPY"
    assert jpy.name == "Japanese Yen"
    assert jpy.decimals == 0
    assert jpy.type == CurrencyType.MONEY
    assert jpy.quantize(Decimal("0.5")) == Decimal("0")
    assert jpy.quantize(Decimal("1.5")) == Decimal("2")

    # Test currency with variable precision
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert zzz.code == "ZZZ"
    assert zzz.name == "Some weird currency"
    assert zzz.decimals == -1
    assert zzz.type == CurrencyType.CRYPTO
    assert zzz.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert zzz.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")

    # Test equality
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2
    assert not (usd1 == usdx)
    assert hash(usd1) == hash(usd2)
    assert not (hash(usd1) == hash(usdx))

    # Test invalid currency code
    with pytest.raises(ProgrammingError):
        Currency.of("123", "Invalid Code", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("usd", "Lowercase Code", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD ", "Code with space", 2, CurrencyType.MONEY)

    # Test invalid currency name
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", " ", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", " US Dollars", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars ", 2, CurrencyType.MONEY)

    # Test invalid decimals
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", -2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", "2", CurrencyType.MONEY)

    # Test invalid currency type
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", 2, "MONEY")


# LLM-generated content at query #121
#--------------------------

```python
def test_Currency___repr__():
    currency = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert repr(currency) == "Currency(code='USD', name='US Dollars', decimals=2, type=CurrencyType.MONEY, quantizer=Decimal('0.01'), hashcache={})".format(hash(('USD', 'US Dollars', 2, CurrencyType.MONEY, Decimal('0.01'))))


# LLM-generated content at query #122
#--------------------------

```python
def test_CurrencyRegistry___enter__():
    registry = CurrencyRegistry()
    with registry as register:
        assert callable(register)
        assert register.__name__ == "_CurrencyRegistry__register"
    assert not registry._CurrencyRegistry__ctx_open


# LLM-generated content at query #123
#--------------------------

```python
def test_Currency___ge__():
    # Test equality case
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 >= usd2
    assert usd2 >= usd1

    # Test greater than case (different codes)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert eur >= usd  # 'EUR' > 'USD' lexicographically

    # Test less than case (different codes)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert not (jpy >= usd)  # 'JPY' < 'USD' lexicographically

    # Test different decimals
    crypto1 = Currency.of("ZZZ", "Crypto1", -1, CurrencyType.CRYPTO)
    crypto2 = Currency.of("ZZZ", "Crypto2", 0, CurrencyType.CRYPTO)
    assert crypto2 >= crypto1  # 0 > -1

    # Test different types
    metal = Currency.of("XAU", "Gold", 2, CurrencyType.METAL)
    alt = Currency.of("XAU", "Gold Alt", 2, CurrencyType.ALTERNATIVE)
    assert alt >= metal  # ALTERNATIVE > METAL in enum ordering

    # Test with completely different currencies
    gbp = Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY)
    chf = Currency.of("CHF", "Swiss Franc", 2, CurrencyType.MONEY)
    assert chf >= gbp  # 'CHF' > 'GBP' lexicographically


# LLM-generated content at query #124
#--------------------------

```python
def test_CurrencyRegistry_has():
    # Test with existing currency
    assert Currencies.has("USD") is True

    # Test with non-existing currency
    assert Currencies.has("NONEXISTENT") is False

    # Test with empty string
    assert Currencies.has("") is False

    # Test with None (should raise TypeError)
    with pytest.raises(TypeError):
        Currencies.has(None)

    # Test with non-string type (should raise TypeError)
    with pytest.raises(TypeError):
        Currencies.has(123)


# LLM-generated content at query #125
#--------------------------

```python
def test_CurrencyRegistry_get():
    # Test with existing currency
    assert Currencies.get("USD") == Currencies["USD"]

    # Test with non-existing currency and no default
    assert Currencies.get("NON-EXISTING") is None

    # Test with non-existing currency and a default
    default_currency = Currencies["USD"]
    assert Currencies.get("NON-EXISTING", default=default_currency) == default_currency

    # Test with existing currency and a default (should return the existing currency, not the default)
    assert Currencies.get("USD", default=default_currency) == Currencies["USD"]


# LLM-generated content at query #126
#--------------------------

```python
def test_Currency_quantize():
    # Test with USD (2 decimals)
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert USD.quantize(Decimal("1.005")) == Decimal("1.00")
    assert USD.quantize(Decimal("1.015")) == Decimal("1.02")
    assert USD.quantize(Decimal("1.000")) == Decimal("1.00")
    assert USD.quantize(Decimal("1.999")) == Decimal("2.00")

    # Test with JPY (0 decimals)
    JPY = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert JPY.quantize(Decimal("0.5")) == Decimal("0")
    assert JPY.quantize(Decimal("1.5")) == Decimal("2")
    assert JPY.quantize(Decimal("0.0")) == Decimal("0")
    assert JPY.quantize(Decimal("999.9")) == Decimal("1000")

    # Test with a currency that has no fixed precision (CRYPTO)
    ZZZ = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert ZZZ.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert ZZZ.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")
    assert ZZZ.quantize(Decimal("1.0000000000000")) == Decimal("1.000000000000")
    assert ZZZ.quantize(Decimal("1.0000000000001")) == Decimal("1.000000000000")

    # Test edge cases
    EUR = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert EUR.quantize(Decimal("0.001")) == Decimal("0.00")
    assert EUR.quantize(Decimal("0.009")) == Decimal("0.01")
    assert EUR.quantize(Decimal("-1.005")) == Decimal("-1.00")
    assert EUR.quantize(Decimal("-1.015")) == Decimal("-1.02")


# LLM-generated content at query #127
#--------------------------

```python
def test_Currency():
    # Test valid currency creation
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.code == "USD"
    assert usd.name == "US Dollars"
    assert usd.decimals == 2
    assert usd.type == CurrencyType.MONEY
    assert usd.quantizer == make_quantizer(2)

    # Test currency with zero decimals
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy.quantizer == ZERO

    # Test currency with variable precision
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert zzz.quantizer == MaxPrecisionQuantizer

    # Test equality and hash
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2
    assert hash(usd1) == hash(usd2)

    # Test inequality
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert usd1 != usdx
    assert hash(usd1) != hash(usdx)

    # Test quantize method
    assert usd.quantize(Decimal("1.005")) == Decimal("1.00")
    assert usd.quantize(Decimal("1.015")) == Decimal("1.02")
    assert jpy.quantize(Decimal("0.5")) == Decimal("0")
    assert jpy.quantize(Decimal("1.5")) == Decimal("2")
    assert zzz.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert zzz.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")

    # Test error cases
    with pytest.raises(ProgrammingError):
        Currency.of(123, "US Dollars", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("usd", "US Dollars", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", " US Dollars", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars ", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", "2", CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", -2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", 2, "MONEY")


# LLM-generated content at query #128
#--------------------------

```python
def test_Currency___eq__():
    # Test equality with same currency
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2

    # Test inequality with different currency
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 == usdx)

    # Test inequality with different type
    assert not (usd1 == "USD")

    # Test inequality with different code
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert not (usd1 == jpy)

    # Test inequality with different name
    usd3 = Currency.of("USD", "US Dollars Different", 2, CurrencyType.MONEY)
    assert not (usd1 == usd3)

    # Test inequality with different decimals
    usd4 = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert not (usd1 == usd4)

    # Test inequality with different type
    usd5 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert not (usd1 == usd5)


# LLM-generated content at query #129
#--------------------------

```python
def test_CurrencyRegistry___len__():
    registry = CurrencyRegistry()
    assert len(registry) == 0

    with registry as register:
        register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        register(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))
        register(Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY))

    assert len(registry) == 3


# LLM-generated content at query #130
#--------------------------

```python
def test_Currency___le__():
    # Test equality case
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 <= usd2

    # Test less than case (by code)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    gbp = Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY)
    assert eur <= gbp

    # Test less than case (by name when codes are equal)
    usd_alt = Currency.of("USD", "US Dollars Alternative", 2, CurrencyType.MONEY)
    assert usd1 <= usd_alt

    # Test less than case (by decimals when code and name are equal)
    usd_3dec = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert usd1 <= usd_3dec

    # Test less than case (by type when code, name, and decimals are equal)
    usd_metal = Currency.of("USD", "US Dollars", 2, CurrencyType.METAL)
    assert usd1 <= usd_metal

    # Test greater than case (should return False)
    assert not gbp <= eur
    assert not usd_alt <= usd1
    assert not usd_3dec <= usd1
    assert not usd_metal <= usd1


# LLM-generated content at query #131
#--------------------------

```python
def test_CurrencyLookupError():
    code = "XYZ"
    error = CurrencyLookupError(code)
    assert error.code == code
    assert str(error) == f"Currency identified by code '{code}' does not exist"


# LLM-generated content at query #132
#--------------------------

```python
def test_CurrencyRegistry___getitem__():
    # Test successful retrieval of a currency
    usd = CurrencyRegistry()[0]  # Assuming USD is the first currency in the registry
    assert CurrencyRegistry()["USD"] == usd

    # Test retrieval of a non-existent currency
    with pytest.raises(CurrencyLookupError):
        CurrencyRegistry()["NON-EXISTING"]


# LLM-generated content at query #133
#--------------------------

```python
def test_Currency_quantize():
    # Test with USD (2 decimals)
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert USD.quantize(Decimal("1.005")) == Decimal("1.00")
    assert USD.quantize(Decimal("1.015")) == Decimal("1.02")
    assert USD.quantize(Decimal("1.000")) == Decimal("1.00")
    assert USD.quantize(Decimal("1.999")) == Decimal("2.00")

    # Test with JPY (0 decimals)
    JPY = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert JPY.quantize(Decimal("0.5")) == Decimal("0")
    assert JPY.quantize(Decimal("1.5")) == Decimal("2")
    assert JPY.quantize(Decimal("1.0")) == Decimal("1")
    assert JPY.quantize(Decimal("9.9")) == Decimal("10")

    # Test with a currency with no fixed precision (CRYPTO)
    ZZZ = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert ZZZ.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert ZZZ.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")
    assert ZZZ.quantize(Decimal("1.0000000000000")) == Decimal("1.000000000000")
    assert ZZZ.quantize(Decimal("1.0000000000009")) == Decimal("1.000000000001")

    # Test with negative values
    assert USD.quantize(Decimal("-1.005")) == Decimal("-1.00")
    assert JPY.quantize(Decimal("-1.5")) == Decimal("-2")
    assert ZZZ.quantize(Decimal("-1.0000000000005")) == Decimal("-1.000000000000")


# LLM-generated content at query #134
#--------------------------

```python
def test_Currency_quantize():
    # Test USD with 2 decimals
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert USD.quantize(Decimal("1.005")) == Decimal("1.00")
    assert USD.quantize(Decimal("1.015")) == Decimal("1.02")
    assert USD.quantize(Decimal("1.000")) == Decimal("1.00")
    assert USD.quantize(Decimal("1.999")) == Decimal("2.00")

    # Test JPY with 0 decimals
    JPY = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert JPY.quantize(Decimal("0.5")) == Decimal("0")
    assert JPY.quantize(Decimal("1.5")) == Decimal("2")
    assert JPY.quantize(Decimal("1.0")) == Decimal("1")
    assert JPY.quantize(Decimal("9.9")) == Decimal("10")

    # Test ZZZ with -1 decimals (no fixed precision)
    ZZZ = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert ZZZ.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert ZZZ.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")
    assert ZZZ.quantize(Decimal("1.0000000000000")) == Decimal("1.000000000000")
    assert ZZZ.quantize(Decimal("1.0000000000001")) == Decimal("1.000000000000")

    # Test negative values
    assert USD.quantize(Decimal("-1.005")) == Decimal("-1.00")
    assert JPY.quantize(Decimal("-1.5")) == Decimal("-2")
    assert ZZZ.quantize(Decimal("-1.0000000000005")) == Decimal("-1.000000000000")

    # Test zero
    assert USD.quantize(Decimal("0")) == Decimal("0.00")
    assert JPY.quantize(Decimal("0")) == Decimal("0")
    assert ZZZ.quantize(Decimal("0")) == Decimal("0")


# LLM-generated content at query #135
#--------------------------

```python
def test_CurrencyRegistry___len__():
    registry = CurrencyRegistry()
    with registry:
        registry(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        registry(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))
    assert len(registry) == 2


# LLM-generated content at query #136
#--------------------------

```python
def test_Currency_quantize():
    # Test with USD (2 decimals)
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert USD.quantize(Decimal("1.005")) == Decimal("1.00")
    assert USD.quantize(Decimal("1.015")) == Decimal("1.02")
    assert USD.quantize(Decimal("1.23456")) == Decimal("1.23")
    assert USD.quantize(Decimal("1.23556")) == Decimal("1.24")

    # Test with JPY (0 decimals)
    JPY = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert JPY.quantize(Decimal("0.5")) == Decimal("0")
    assert JPY.quantize(Decimal("1.5")) == Decimal("2")
    assert JPY.quantize(Decimal("123.456")) == Decimal("123")

    # Test with a currency with -1 decimals (max precision)
    ZZZ = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert ZZZ.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert ZZZ.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")
    assert ZZZ.quantize(Decimal("1.123456789012345678901234567890")) == Decimal("1.123456789012345678901234567890")

    # Test with negative numbers
    assert USD.quantize(Decimal("-1.005")) == Decimal("-1.00")
    assert USD.quantize(Decimal("-1.015")) == Decimal("-1.02")
    assert JPY.quantize(Decimal("-0.5")) == Decimal("0")
    assert JPY.quantize(Decimal("-1.5")) == Decimal("-2")

    # Test with zero
    assert USD.quantize(Decimal("0")) == Decimal("0")
    assert JPY.quantize(Decimal("0")) == Decimal("0")
    assert ZZZ.quantize(Decimal("0")) == Decimal("0")


# LLM-generated content at query #137
#--------------------------

```python
def test_Currency___setattr__():
    # Test that Currency instances are immutable (frozen)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)

    # Attempt to modify attributes should raise AttributeError
    with pytest.raises(AttributeError):
        usd.code = "EUR"

    with pytest.raises(AttributeError):
        usd.name = "Euro"

    with pytest.raises(AttributeError):
        usd.decimals = 3

    with pytest.raises(AttributeError):
        usd.type = CurrencyType.CRYPTO

    with pytest.raises(AttributeError):
        usd.quantizer = Decimal("0.001")

    with pytest.raises(AttributeError):
        usd.hashcache = 123


# LLM-generated content at query #138
#--------------------------

```python
def test_Currency___gt__():
    # Test with different codes
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert usd > eur

    # Test with same code but different names
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars 2", 2, CurrencyType.MONEY)
    assert usd1 > usd2

    # Test with same code and name but different decimals
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert usd1 > usd2

    # Test with same code, name, and decimals but different types
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert usd1 > usd2

    # Test with same code, name, decimals, and type but different quantizers
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", -1, CurrencyType.MONEY)
    assert usd1 > usd2

    # Test with same currency
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 > usd2)


# LLM-generated content at query #139
#--------------------------

```python
def test_Currency___hash__():
    # Test basic hash functionality
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert hash(usd1) == hash(usd2)

    # Test hash with different currencies
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert hash(usd1) != hash(jpy)

    # Test hash with different attributes
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert hash(usd1) != hash(usdx)

    # Test hash with different decimals
    usd_high_precision = Currency.of("USD", "US Dollars", 4, CurrencyType.MONEY)
    assert hash(usd1) != hash(usd_high_precision)

    # Test hash with different types
    usd_crypto = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert hash(usd1) != hash(usd_crypto)

    # Test hash with negative decimals
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert isinstance(hash(zzz), int)


# LLM-generated content at query #140
#--------------------------

```python
def test_CurrencyLookupError():
    # Test initialization and message
    error = CurrencyLookupError("XYZ")
    assert error.code == "XYZ"
    assert str(error) == "Currency identified by code 'XYZ' does not exist"

    # Test inheritance
    assert isinstance(error, LookupError)

    # Test with different code
    error2 = CurrencyLookupError("ABC")
    assert error2.code == "ABC"
    assert str(error2) == "Currency identified by code 'ABC' does not exist"


# LLM-generated content at query #141
#--------------------------

```python
def test_CurrencyRegistry___getitem__():
    # Test existing currency
    assert Currencies["USD"].code == "USD"

    # Test non-existing currency
    with pytest.raises(CurrencyLookupError):
        Currencies["NON-EXISTING"]


# LLM-generated content at query #142
#--------------------------

```python
def test_CurrencyRegistry_get():
    # Test getting an existing currency
    usd = CurrencyRegistry().get("USD")
    assert usd is not None
    assert usd.code == "USD"
    assert usd.name == "US Dollar"
    assert usd.type == CurrencyType.MONEY

    # Test getting a non-existing currency without default
    assert CurrencyRegistry().get("NONEXISTENT") is None

    # Test getting a non-existing currency with default
    default_currency = Currency.of("XYZ", "Test Currency", 2, CurrencyType.MONEY)
    result = CurrencyRegistry().get("NONEXISTENT", default=default_currency)
    assert result == default_currency

    # Test that getting with default doesn't affect the registry
    assert "NONEXISTENT" not in CurrencyRegistry()
    assert len(CurrencyRegistry()) == len(CurrencyRegistry().all)


# LLM-generated content at query #143
#--------------------------

```python
def test_Currency___setattr__():
    # Create a currency instance
    currency = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)

    # Attempt to set an attribute and expect an AttributeError since Currency is frozen
    with pytest.raises(AttributeError):
        currency.code = "EUR"

    with pytest.raises(AttributeError):
        currency.name = "Euros"

    with pytest.raises(AttributeError):
        currency.decimals = 3

    with pytest.raises(AttributeError):
        currency.type = CurrencyType.CRYPTO

    with pytest.raises(AttributeError):
        currency.quantizer = Decimal("0.001")

    with pytest.raises(AttributeError):
        currency.hashcache = 12345


# LLM-generated content at query #144
#--------------------------

```python
def test_CurrencyRegistry___enter__():
    registry = CurrencyRegistry()
    with registry as register:
        assert callable(register)
        assert register.__name__ == "_CurrencyRegistry__register"
    assert not registry._CurrencyRegistry__ctx_open


# LLM-generated content at query #145
#--------------------------

```python
def test_CurrencyRegistry_has():
    # Test with existing currency
    assert Currencies.has("USD") is True

    # Test with non-existing currency
    assert Currencies.has("NONEXISTENT") is False

    # Test with empty string
    assert Currencies.has("") is False

    # Test with code that has special characters
    assert Currencies.has("USD!") is False

    # Test with code that has lowercase letters
    assert Currencies.has("usd") is False


# LLM-generated content at query #146
#--------------------------

```python
def test_Currency___ge__():
    # Test equality cases
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 >= usd2
    assert usd2 >= usd1

    # Test inequality cases
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert usd1 >= jpy
    assert not (jpy >= usd1)

    # Test with different attributes
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert usd1 >= usdx
    assert usdx >= usd1

    # Test with different decimals
    crypto = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert usd1 >= crypto
    assert not (crypto >= usd1)

    # Test with different types
    metal = Currency.of("XAU", "Gold", 2, CurrencyType.METAL)
    assert usd1 >= metal
    assert not (metal >= usd1)


# LLM-generated content at query #147
#--------------------------

```python
def test_Currency___repr__():
    # Test basic currency representation
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert repr(usd) == "Currency(code='USD', name='US Dollars', decimals=2, type=CurrencyType.MONEY, quantizer=Decimal('0.01'), hashcache={})".format(hash(usd))

    # Test currency with different decimals
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert repr(jpy) == "Currency(code='JPY', name='Japanese Yen', decimals=0, type=CurrencyType.MONEY, quantizer=Decimal('0'), hashcache={})".format(hash(jpy))

    # Test crypto currency with negative decimals
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert repr(zzz) == "Currency(code='ZZZ', name='Some weird currency', decimals=-1, type=CurrencyType.CRYPTO, quantizer=MaxPrecisionQuantizer, hashcache={})".format(hash(zzz))

    # Test metal currency
    xau = Currency.of("XAU", "Gold", 4, CurrencyType.METAL)
    assert repr(xau) == "Currency(code='XAU', name='Gold', decimals=4, type=CurrencyType.METAL, quantizer=Decimal('0.0001'), hashcache={})".format(hash(xau))

    # Test alternative currency
    alt = Currency.of("ALT", "Alternative Currency", 3, CurrencyType.ALTERNATIVE)
    assert repr(alt) == "Currency(code='ALT', name='Alternative Currency', decimals=3, type=CurrencyType.ALTERNATIVE, quantizer=Decimal('0.001'), hashcache={})".format(hash(alt))


# LLM-generated content at query #148
#--------------------------

```python
def test_CurrencyRegistry___enter__():
    registry = CurrencyRegistry()
    with registry as register:
        assert callable(register)
        assert register.__name__ == "_CurrencyRegistry__register"
    assert not registry._CurrencyRegistry__ctx_open


# LLM-generated content at query #149
#--------------------------

```python
def test_CurrencyRegistry():
    # Test singleton behavior
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2

    # Test initial state
    assert len(registry1) == 0
    assert registry1.all == []
    assert registry1.codes == []
    assert registry1.codenames == []

    # Test context manager
    with registry1 as register:
        # Test adding currency
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        register(usd)

        # Test adding duplicate currency
        with pytest.raises(ValueError):
            register(usd)

    # Test post-context state
    assert len(registry1) == 1
    assert registry1.all == [usd]
    assert registry1.codes == ["USD"]
    assert registry1.codenames == [("USD", "US Dollar")]

    # Test currency lookup
    assert registry1.has("USD")
    assert "USD" in registry1
    assert registry1["USD"] == usd
    assert registry1.get("USD") == usd
    assert registry1.get("XXX") is None
    assert registry1.get("XXX", default=usd) == usd

    # Test non-existing currency lookup
    with pytest.raises(CurrencyLookupError):
        registry1["XXX"]

    # Test adding currency outside context
    with pytest.raises(ProgrammingError):
        registry1._CurrencyRegistry__register(usd)


# LLM-generated content at query #150
#--------------------------

```python
def test_Currency___le__():
    # Test less than
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert usd <= jpy or jpy <= usd  # Should always be true for different currencies

    # Test equal
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 <= usd2
    assert usd2 <= usd1

    # Test with same code but different attributes
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 <= usdx) or not (usdx <= usd1)  # Should not be equal

    # Test with different types
    gold = Currency.of("XAU", "Gold", 2, CurrencyType.METAL)
    btc = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    assert gold <= btc or btc <= gold  # Should always be true for different currencies


