####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_CurrencyRegistry___enter__():
    registry = CurrencyRegistry()
    with registry as register:
        assert callable(register)
        assert register.__name__ == "_CurrencyRegistry__register"


# LLM-generated content at query #2
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


# LLM-generated content at query #3
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

    # Test case sensitivity (should be case-sensitive)
    assert Currencies.has("usd") is False


# LLM-generated content at query #4
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


# LLM-generated content at query #5
#--------------------------

```python
def test_Currency___gt__():
    # Create currencies for testing
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)

    # Test greater than
    assert usd > jpy
    assert zzz > usd
    assert zzz > jpy

    # Test not greater than
    assert not (jpy > usd)
    assert not (usd > zzz)
    assert not (jpy > zzz)

    # Test equality (should not be greater than)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd > usd2)
    assert not (usd2 > usd)


# LLM-generated content at query #6
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


# LLM-generated content at query #7
#--------------------------

```python
def test_CurrencyRegistry___getitem__():
    # Setup
    registry = CurrencyRegistry()
    with registry:
        registry(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        registry(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))

    # Test existing currency
    usd = registry["USD"]
    assert usd.code == "USD"
    assert usd.name == "US Dollar"
    assert usd.decimals == 2
    assert usd.type == CurrencyType.MONEY

    # Test non-existing currency
    with pytest.raises(CurrencyLookupError) as excinfo:
        registry["NON-EXISTING"]
    assert str(excinfo.value) == "Currency identified by code 'NON-EXISTING' does not exist"


# LLM-generated content at query #8
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
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert not (usd1 >= eur)
    assert not (eur >= usd1)

    # Test different types
    usd_metal = Currency.of("USD", "US Dollars", 2, CurrencyType.METAL)
    assert not (usd1 >= usd_metal)
    assert not (usd_metal >= usd1)


# LLM-generated content at query #9
#--------------------------

```python
def test_Currency___le__():
    # Test equality
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 <= usd2

    # Test less than
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert usd <= eur or eur <= usd

    # Test with different attributes
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert (usd <= jpy) == (usd.code <= jpy.code)

    # Test with non-Currency object
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd <= "not a currency")


# LLM-generated content at query #10
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
    result = Currencies.get("NONEXIST")
    assert result is None

    # Test 3: Get non-existing currency with default
    default_currency = Currency.of("DEFAULT", "Default Currency", 0, CurrencyType.MONEY)
    result = Currencies.get("NONEXIST", default=default_currency)
    assert result == default_currency

    # Test 4: Get existing currency with default (should return the existing one, not default)
    result = Currencies.get("TEST", default=default_currency)
    assert result == test_currency


# LLM-generated content at query #11
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
        Currency.of("abc", "Invalid Code", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("Abc", "Invalid Code", 2, CurrencyType.MONEY)

    # Test invalid currency name
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "  ", 2, CurrencyType.MONEY)
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


# LLM-generated content at query #12
#--------------------------

```python
def test_CurrencyRegistry___len__():
    registry = CurrencyRegistry()
    with registry as register:
        register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        register(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))
        register(Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY))

    assert len(registry) == 3


# LLM-generated content at query #13
#--------------------------

```python
def test_Currency___hash__():
    # Test basic hash functionality
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert hash(usd1) == hash(usd2)

    # Test that different currencies have different hashes
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert hash(usd) != hash(jpy)

    # Test that same attributes produce same hash
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert hash(usd1) == hash(usd2)

    # Test that different attributes produce different hashes
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 0, CurrencyType.MONEY)
    assert hash(usd1) != hash(usd2)

    # Test that hash is consistent with equality
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2
    assert hash(usd1) == hash(usd2)

    # Test that hash is different for different currency types
    usd_money = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd_crypto = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert hash(usd_money) != hash(usd_crypto)


# LLM-generated content at query #14
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
    assert (usd < usdx) or (usdx < usd)  # One must be less than the other

    # Test with different types
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    btc = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    assert (usd < btc) or (btc < usd)  # One must be less than the other


# LLM-generated content at query #15
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


# LLM-generated content at query #16
#--------------------------

```python
def test_CurrencyRegistry___new__():
    # Test that CurrencyRegistry is a singleton
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2

    # Test that the instance is of type CurrencyRegistry
    assert isinstance(registry1, CurrencyRegistry)


# LLM-generated content at query #17
#--------------------------

```python
def test_CurrencyRegistry___enter__():
    registry = CurrencyRegistry()
    with registry as register:
        assert callable(register)
        assert register.__name__ == "_CurrencyRegistry__register"
    assert not registry._CurrencyRegistry__ctx_open


# LLM-generated content at query #18
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


# LLM-generated content at query #19
#--------------------------

```python
def test_CurrencyRegistry___getitem__():
    # Test getting an existing currency
    registry = CurrencyRegistry()
    with registry as register:
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        register(usd)

    assert registry["USD"] == usd
    assert registry["USD"].code == "USD"
    assert registry["USD"].name == "US Dollar"

    # Test getting a non-existing currency
    with pytest.raises(CurrencyLookupError) as excinfo:
        _ = registry["NON-EXISTING"]
    assert str(excinfo.value) == "Currency identified by code 'NON-EXISTING' does not exist"


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

    # Test currency with zero decimals
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


# LLM-generated content at query #21
#--------------------------

```python
def test_CurrencyRegistry___enter__():
    registry = CurrencyRegistry()
    with registry as register:
        assert callable(register)
        assert register.__name__ == "_CurrencyRegistry__register"
    assert not registry._CurrencyRegistry__ctx_open


# LLM-generated content at query #22
#--------------------------

```python
def test_Currency___repr__():
    # Test with a standard currency
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert repr(usd) == "Currency(code='USD', name='US Dollars', decimals=2, type=CurrencyType.MONEY, quantizer=Decimal('0.01'), hashcache=" + str(hash(usd)) + ")"

    # Test with a currency that has no decimals
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert repr(jpy) == "Currency(code='JPY', name='Japanese Yen', decimals=0, type=CurrencyType.MONEY, quantizer=Decimal('0'), hashcache=" + str(hash(jpy)) + ")"

    # Test with a crypto currency
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert repr(zzz) == "Currency(code='ZZZ', name='Some weird currency', decimals=-1, type=CurrencyType.CRYPTO, quantizer=Decimal('1E-28'), hashcache=" + str(hash(zzz)) + ")"


# LLM-generated content at query #23
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
    assert usd <= eur

    # Test greater than case (by code)
    assert eur <= usd is False

    # Test different decimals
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy <= usd

    # Test different types
    btc = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    assert btc <= usd is False

    # Test with different names (same code, decimals, type)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert usd <= usdx is False


# LLM-generated content at query #24
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


# LLM-generated content at query #25
#--------------------------

```python
def test_Currency___delattr__():
    # Test that __delattr__ raises an AttributeError since Currency is frozen
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    with pytest.raises(AttributeError):
        del usd.code
    with pytest.raises(AttributeError):
        del usd.name
    with pytest.raises(AttributeError):
        del usd.decimals
    with pytest.raises(AttributeError):
        del usd.type
    with pytest.raises(AttributeError):
        del usd.quantizer
    with pytest.raises(AttributeError):
        del usd.hashcache


# LLM-generated content at query #26
#--------------------------

```python
def test_Currency___hash__():
    # Test basic hash functionality
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)

    # Test that equal currencies have equal hashes
    assert hash(usd1) == hash(usd2)

    # Test that different currencies have different hashes
    assert hash(usd1) != hash(usdx)
    assert hash(usd1) != hash(jpy)

    # Test that hash is consistent with equality
    assert usd1 == usd2
    assert hash(usd1) == hash(usd2)
    assert usd1 != usdx
    assert hash(usd1) != hash(usdx)

    # Test that hash is pre-computed and cached
    assert usd1.hashcache == hash(usd1)
    assert usd2.hashcache == hash(usd2)


# LLM-generated content at query #27
#--------------------------

```python
def test_Currency___setattr__():
    # Test that Currency is immutable (frozen dataclass)
    currency = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)

    # Attempt to modify attributes
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


# LLM-generated content at query #28
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

    # Check the length after adding currencies
    assert len(registry) == 3


# LLM-generated content at query #29
#--------------------------

```python
def test_Currency___delattr__():
    # Create a Currency instance
    currency = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)

    # Verify that the Currency instance is frozen and should raise an AttributeError when trying to delete an attribute
    with pytest.raises(AttributeError):
        delattr(currency, 'code')

    with pytest.raises(AttributeError):
        delattr(currency, 'name')

    with pytest.raises(AttributeError):
        delattr(currency, 'decimals')

    with pytest.raises(AttributeError):
        delattr(currency, 'type')

    with pytest.raises(AttributeError):
        delattr(currency, 'quantizer')

    with pytest.raises(AttributeError):
        delattr(currency, 'hashcache')


# LLM-generated content at query #30
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
    usd_different = Currency.of("USD", "Different Name", 2, CurrencyType.CRYPTO)
    assert hash(usd1) != hash(usd_different)

    # Test hash with negative decimals
    weird = Currency.of("ZZZ", "Weird Currency", -1, CurrencyType.CRYPTO)
    assert isinstance(hash(weird), int)

    # Test hash with different currency types
    metal = Currency.of("XAU", "Gold", 2, CurrencyType.METAL)
    assert hash(usd1) != hash(metal)


# LLM-generated content at query #31
#--------------------------

```python
def test_CurrencyRegistry___getitem__():
    # Test getting an existing currency
    usd = CurrencyRegistry()[0]["USD"]
    assert usd.code == "USD"
    assert usd.name == "US Dollar"
    assert usd.type == CurrencyType.MONEY

    # Test getting a non-existing currency
    with pytest.raises(CurrencyLookupError) as excinfo:
        CurrencyRegistry()[0]["NON-EXISTING"]
    assert str(excinfo.value) == "Currency identified by code 'NON-EXISTING' does not exist"


# LLM-generated content at query #32
#--------------------------

```python
def test_Currency___hash__():
    # Test basic hash consistency
    currency1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    currency2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert hash(currency1) == hash(currency2)

    # Test hash uniqueness for different currencies
    currency3 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert hash(currency1) != hash(currency3)

    # Test hash uniqueness for same code but different attributes
    currency4 = Currency.of("USD", "US Dollars", 0, CurrencyType.MONEY)
    assert hash(currency1) != hash(currency4)

    # Test hash uniqueness for different currency types
    currency5 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert hash(currency1) != hash(currency5)

    # Test hash uniqueness for different quantizers
    currency6 = Currency.of("BTC", "Bitcoin", -1, CurrencyType.CRYPTO)
    assert hash(currency1) != hash(currency6)


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
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
    with pytest.raises(CurrencyLookupError):
        registry["NON-EXISTING"]


# LLM-generated content at query #2
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


# LLM-generated content at query #3
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


# LLM-generated content at query #4
#--------------------------

```python
def test_Currency___eq__():
    # Test equality with same currency
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2

    # Test equality with different currency
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 == usdx)

    # Test equality with different type
    assert not (usd1 == "USD")

    # Test equality with different code
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert not (usd1 == jpy)

    # Test equality with different name
    usd3 = Currency.of("USD", "US Dollars Different", 2, CurrencyType.MONEY)
    assert not (usd1 == usd3)

    # Test equality with different decimals
    usd4 = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert not (usd1 == usd4)

    # Test equality with different type
    usd5 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert not (usd1 == usd5)


# LLM-generated content at query #5
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

    # Test inequality with different object type
    assert usd1 != "USD"

    # Test inequality with None
    assert usd1 != None


# LLM-generated content at query #6
#--------------------------

```python
def test_CurrencyRegistry___getitem__():
    # Setup
    registry = CurrencyRegistry()
    with registry as register:
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        register(usd)

    # Test valid code
    assert registry["USD"] == usd
    assert registry["USD"].code == "USD"
    assert registry["USD"].name == "US Dollar"

    # Test invalid code
    with pytest.raises(CurrencyLookupError) as exc_info:
        _ = registry["NON-EXISTING"]
    assert str(exc_info.value) == "Currency identified by code 'NON-EXISTING' does not exist"


# LLM-generated content at query #7
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

    # Test valid currency code
    assert registry["USD"] == usd
    assert registry["EUR"] == eur

    # Test invalid currency code
    with pytest.raises(CurrencyLookupError) as excinfo:
        _ = registry["NON-EXISTING"]
    assert str(excinfo.value) == "Currency identified by code 'NON-EXISTING' does not exist"


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
def test_CurrencyRegistry___getitem__():
    # Test getting an existing currency
    assert Currencies["USD"].code == "USD"
    assert Currencies["USD"].name == "US Dollar"
    assert Currencies["USD"].type == CurrencyType.MONEY

    # Test getting a non-existing currency raises CurrencyLookupError
    with pytest.raises(CurrencyLookupError) as excinfo:
        _ = Currencies["NON-EXISTING"]
    assert str(excinfo.value) == "Currency identified by code 'NON-EXISTING' does not exist"


# LLM-generated content at query #10
#--------------------------

```python
def test_CurrencyRegistry___getitem__():
    # Test getting an existing currency
    usd = CurrencyRegistry()[0]["USD"]
    assert usd.code == "USD"
    assert usd.name == "US Dollar"
    assert usd.type == CurrencyType.MONEY

    # Test getting a non-existing currency
    with pytest.raises(CurrencyLookupError) as excinfo:
        CurrencyRegistry()[0]["NON-EXISTING"]
    assert str(excinfo.value) == "Currency identified by code 'NON-EXISTING' does not exist"


# LLM-generated content at query #11
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
        test_currency = Currency.of("TEST", "Test Currency", 2, CurrencyType.MONEY)
        register(test_currency)

    assert len(registry1) == 1
    assert "TEST" in registry1
    assert registry1["TEST"] == test_currency
    assert registry1.all == [test_currency]
    assert registry1.codes == ["TEST"]
    assert registry1.codenames == [("TEST", "Test Currency")]

    # Test adding duplicate currency
    with pytest.raises(ValueError):
        with registry1 as register:
            register(test_currency)

    # Test adding outside context
    with pytest.raises(ProgrammingError):
        registry1._CurrencyRegistry__register(test_currency)


# LLM-generated content at query #12
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


# LLM-generated content at query #13
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
    assert jpy <= usd or usd <= jpy  # One of these should be true based on ordering

    # Test with different attributes
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    crypto = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    assert usd <= crypto or crypto <= usd  # One should be true based on ordering

    # Test with same code but different other attributes
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert usd1 <= usd2 or usd2 <= usd1  # One should be true based on ordering


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


# LLM-generated content at query #15
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


# LLM-generated content at query #16
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


# LLM-generated content at query #17
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
    assert repr(zzz) == "Currency(code='ZZZ', name='Some weird currency', decimals=-1, type=CurrencyType.CRYPTO, quantizer=Decimal('1E-28'), hashcache=" + str(hash(zzz)) + ")"


# LLM-generated content at query #18
#--------------------------

```python
def test_Currency___setattr__():
    # Test that Currency is immutable and __setattr__ raises AttributeError
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


# LLM-generated content at query #19
#--------------------------

```python
def test_Currency___repr__():
    currency = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert repr(currency) == "Currency(code='USD', name='US Dollars', decimals=2, type=CurrencyType.MONEY, quantizer=Decimal('0.01'), hashcache=" + str(hash(('USD', 'US Dollars', 2, CurrencyType.MONEY, Decimal('0.01')))) + ")"


# LLM-generated content at query #20
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

    # Test hash differs for same code but different attributes
    usd_different = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert hash(usd1) != hash(usd_different)

    # Test hash consistency
    assert hash(usd1) == hash(usd1)

    # Test hash is cached
    assert usd1.__hash__() == usd1.hashcache


# LLM-generated content at query #21
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

    # Test with a currency with no fixed precision (-1 decimals)
    ZZZ = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert ZZZ.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert ZZZ.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")
    assert ZZZ.quantize(Decimal("1.0000000000000")) == Decimal("1.000000000000")
    assert ZZZ.quantize(Decimal("1.0000000000009")) == Decimal("1.000000000001")

    # Test edge cases
    EUR = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert EUR.quantize(Decimal("0.001")) == Decimal("0.00")
    assert EUR.quantize(Decimal("0.009")) == Decimal("0.01")
    assert EUR.quantize(Decimal("-1.005")) == Decimal("-1.00")
    assert EUR.quantize(Decimal("-1.015")) == Decimal("-1.02")


# LLM-generated content at query #22
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

    # Test same code but different attributes have different hashes
    usd_different = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert hash(usd1) != hash(usd_different)

    # Test hash consistency
    assert hash(usd1) == hash(usd1)

    # Test hash is cached
    assert usd1.__hash__() == usd1.hashcache


# LLM-generated content at query #23
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


# LLM-generated content at query #24
#--------------------------

```python
def test_CurrencyRegistry_get():
    # Test getting an existing currency
    usd = CurrencyRegistry().get("USD")
    assert usd is not None
    assert usd.code == "USD"
    assert usd.name == "US Dollar"

    # Test getting a non-existing currency without default
    assert CurrencyRegistry().get("NONEXISTENT") is None

    # Test getting a non-existing currency with default
    default_currency = Currency.of("XYZ", "Test Currency", 2, CurrencyType.MONEY)
    result = CurrencyRegistry().get("NONEXISTENT", default=default_currency)
    assert result == default_currency

    # Test that getting with default doesn't add the currency to registry
    assert "NONEXISTENT" not in CurrencyRegistry()
    assert len(CurrencyRegistry()) == len(CurrencyRegistry().all)


# LLM-generated content at query #25
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

    # Test hash differs when attributes differ
    usd_different_name = Currency.of("USD", "Different Name", 2, CurrencyType.MONEY)
    assert hash(usd1) != hash(usd_different_name)

    usd_different_decimals = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert hash(usd1) != hash(usd_different_decimals)

    usd_different_type = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert hash(usd1) != hash(usd_different_type)

    # Test hash is consistent with equality
    assert usd1 == usd2
    assert hash(usd1) == hash(usd2)

    # Test hash is cached
    assert usd1.__hash__() == usd1.hashcache


# LLM-generated content at query #26
#--------------------------

```python
def test_Currency___delattr__():
    # Create a currency instance
    currency = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)

    # Attempt to delete an attribute and verify it raises an AttributeError
    with pytest.raises(AttributeError):
        del currency.code


# LLM-generated content at query #27
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
    assert hash(usd) == hash(("USD", "US Dollars", 2, CurrencyType.MONEY, make_quantizer(2)))

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
        Currency.of("USD", "US Dollars", -2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", 2, "MONEY")


# LLM-generated content at query #28
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


# LLM-generated content at query #29
#--------------------------

```python
def test_Currency_quantize():
    # Test USD with 2 decimals
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert USD.quantize(Decimal("1.005")) == Decimal("1.00")
    assert USD.quantize(Decimal("1.015")) == Decimal("1.02")
    assert USD.quantize(Decimal("1.000")) == Decimal("1.00")
    assert USD.quantize(Decimal("1.001")) == Decimal("1.00")
    assert USD.quantize(Decimal("1.009")) == Decimal("1.01")

    # Test JPY with 0 decimals
    JPY = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert JPY.quantize(Decimal("0.5")) == Decimal("0")
    assert JPY.quantize(Decimal("1.5")) == Decimal("2")
    assert JPY.quantize(Decimal("1.0")) == Decimal("1")
    assert JPY.quantize(Decimal("1.9")) == Decimal("2")

    # Test ZZZ with -1 decimals (max precision)
    ZZZ = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert ZZZ.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert ZZZ.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")
    assert ZZZ.quantize(Decimal("1.00000000000000000000000000001")) == Decimal("1.00000000000000000000000000001")

    # Test negative values
    assert USD.quantize(Decimal("-1.005")) == Decimal("-1.00")
    assert USD.quantize(Decimal("-1.015")) == Decimal("-1.02")
    assert JPY.quantize(Decimal("-1.5")) == Decimal("-2")
    assert ZZZ.quantize(Decimal("-1.0000000000005")) == Decimal("-1.000000000000")

    # Test zero
    assert USD.quantize(Decimal("0")) == Decimal("0.00")
    assert JPY.quantize(Decimal("0")) == Decimal("0")
    assert ZZZ.quantize(Decimal("0")) == Decimal("0")


# LLM-generated content at query #30
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


# LLM-generated content at query #31
#--------------------------

```python
def test_CurrencyLookupError():
    # Test initialization and message
    error = CurrencyLookupError("XYZ")
    assert error.code == "XYZ"
    assert str(error) == "Currency identified by code 'XYZ' does not exist"

    # Test with different code
    error2 = CurrencyLookupError("ABC")
    assert error2.code == "ABC"
    assert str(error2) == "Currency identified by code 'ABC' does not exist"


# LLM-generated content at query #32
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

    # Test hash is consistent with equality
    usd3 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd3
    assert hash(usd1) == hash(usd3)

    # Test hash differs when any attribute differs
    usd4 = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert usd1 != usd4
    assert hash(usd1) != hash(usd4)

    usd5 = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert usd1 != usd5
    assert hash(usd1) != hash(usd5)

    usd6 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert usd1 != usd6
    assert hash(usd1) != hash(usd6)

    # Test hash is deterministic
    assert hash(usd1) == hash(usd1)


# LLM-generated content at query #33
#--------------------------

```python
def test_Currency___hash__():
    # Test with same currency attributes
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert hash(usd1) == hash(usd2)

    # Test with different currency attributes
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert hash(usd1) != hash(usdx)

    # Test with different currency code
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert hash(usd1) != hash(jpy)

    # Test with different currency type
    crypto_usd = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert hash(usd1) != hash(crypto_usd)

    # Test with different decimals
    weird_currency = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert hash(usd1) != hash(weird_currency)


