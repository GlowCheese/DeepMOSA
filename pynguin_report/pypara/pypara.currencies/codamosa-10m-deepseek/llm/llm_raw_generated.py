####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_CurrencyRegistry___enter__():
    registry = CurrencyRegistry()
    
    with registry as register:
        assert registry._CurrencyRegistry__ctx_open
        register(Currency.of("TEST", "Test Currency", 2, CurrencyType.MONEY))
    
    assert not registry._CurrencyRegistry__ctx_open
    assert "TEST" in registry
    assert registry["TEST"].code == "TEST"
    assert registry["TEST"].name == "Test Currency"
    assert registry["TEST"].decimals == 2
    assert registry["TEST"].type == CurrencyType.MONEY

    with pytest.raises(ProgrammingError):
        register(Currency.of("FAIL", "Fail Currency", 2, CurrencyType.MONEY))


# LLM-generated content at query #2
#--------------------------

```python
def test_Currency___eq__():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    
    assert usd1 == usd2
    assert not (usd1 == usdx)
    assert usd1 != "not a currency"


# LLM-generated content at query #3
#--------------------------

```python
def test_CurrencyRegistry_has():
    registry = CurrencyRegistry()

    # Test with a known currency code
    assert registry.has("USD") == True

    # Test with a non-existent currency code
    assert registry.has("XXX") == False

    # Test with a lowercase currency code (should return False as codes are uppercase)
    assert registry.has("usd") == False

    # Test with an empty string
    assert registry.has("") == False

    # Test with None (should raise TypeError)
    try:
        registry.has(None)
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test with a numeric string
    assert registry.has("123") == False

    # Test with a special character string
    assert registry.has("@#!") == False

    # Test with a currency code that is registered during runtime
    with registry as register:
        register(Currency.of("TEST", "Test Currency", 2, CurrencyType.MONEY))
    assert registry.has("TEST") == True


# LLM-generated content at query #4
#--------------------------

```python
def test_CurrencyRegistry___contains__():
    registry = CurrencyRegistry()
    
    # Test with a currency that exists
    assert "USD" in registry
    
    # Test with a currency that does not exist
    assert "XYZ" not in registry
    
    # Test with a currency that is added within the context
    with registry as register:
        register(Currency.of("ABC", "Test Currency", 2, CurrencyType.MONEY))
    
    assert "ABC" in registry


# LLM-generated content at query #5
#--------------------------

def test_Currency___gt__():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USDX", "UX Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)

    assert not (usd1 > usd2)
    assert not (usd1 > usdx)
    assert usdx > usd1
    assert jpy > usd1


# LLM-generated content at query #6
#--------------------------

def test_CurrencyRegistry_has():
    # Test with existing currency code
    assert Currencies.has("USD") == True
    
    # Test with non-existing currency code
    assert Currencies.has("XXX") == False
    
    # Test with empty string
    assert Currencies.has("") == False
    
    # Test with lowercase code (should be case sensitive)
    assert Currencies.has("usd") == False
    
    # Test with numeric string
    assert Currencies.has("123") == False
    
    # Test with None
    assert Currencies.has(None) == False


# LLM-generated content at query #7
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

    # Test context management and registration
    with CurrencyRegistry() as register:
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        register(usd)

    assert len(registry1) == 1
    assert registry1.codes == ["USD"]
    assert registry1.codenames == [("USD", "US Dollar")]
    assert registry1.all == [usd]

    # Test containment and retrieval
    assert "USD" in registry1
    assert registry1.has("USD")
    assert registry1["USD"] == usd
    assert registry1.get("USD") == usd

    # Test lookup error
    try:
        registry1["NON_EXISTING"]
        assert False, "Expected CurrencyLookupError"
    except CurrencyLookupError:
        pass

    # Test default value in get
    assert registry1.get("NON_EXISTING") is None
    assert registry1.get("NON_EXISTING", default=usd) == usd

    # Test duplicate registration
    try:
        with CurrencyRegistry() as register:
            register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #8
#--------------------------

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
    assert jpy.quantize(Decimal("0.5")) == Decimal("0")
    assert jpy.quantize(Decimal("1.5")) == Decimal("2")

    # Test crypto currency with no fixed precision
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

    # Test invalid code raises ProgrammingError
    with pytest.raises(ProgrammingError):
        Currency.of("usd", "US Dollars", 2, CurrencyType.MONEY)  # lowercase
    with pytest.raises(ProgrammingError):
        Currency.of("US1", "US Dollars", 2, CurrencyType.MONEY)  # contains number
    with pytest.raises(ProgrammingError):
        Currency.of("", "US Dollars", 2, CurrencyType.MONEY)  # empty

    # Test invalid name raises ProgrammingError
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "", 2, CurrencyType.MONEY)  # empty
    with pytest.raises(ProgrammingError):
        Currency.of("USD", " US Dollars", 2, CurrencyType.MONEY)  # leading space
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars ", 2, CurrencyType.MONEY)  # trailing space

    # Test invalid decimals raises ProgrammingError
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", -2, CurrencyType.MONEY)  # less than -1
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", "2", CurrencyType.MONEY)  # not int

    # Test invalid type raises ProgrammingError
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", 2, "MONEY")  # not CurrencyType


# LLM-generated content at query #9
#--------------------------

```python
def test_CurrencyRegistry___enter__():
    registry = CurrencyRegistry()
    
    # Test that __enter__ returns the __register method
    with registry as register:
        assert callable(register)
        assert register.__name__ == "__register"
    
    # Test that the context is marked as open after __enter__
    assert not registry._CurrencyRegistry__ctx_open


# LLM-generated content at query #10
#--------------------------

Here's the unit test for the `__le__` method of the `Currency` class:


# LLM-generated content at query #11
#--------------------------

def test_Currency___eq__():
    # Test equality with same currency
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2

    # Test inequality with different currency (different name)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 == usdx)

    # Test inequality with different currency (different code)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert not (usd1 == eur)

    # Test inequality with different decimals
    usd_dec3 = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert not (usd1 == usd_dec3)

    # Test inequality with different type
    usd_crypto = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert not (usd1 == usd_crypto)

    # Test inequality with non-Currency object
    assert not (usd1 == "USD")

    # Test inequality with None
    assert not (usd1 == None)


# LLM-generated content at query #12
#--------------------------

```python
def test_CurrencyRegistry_has():
    registry = CurrencyRegistry()
    
    # Test with a known currency code
    assert registry.has("USD") is True
    
    # Test with a non-existent currency code
    assert registry.has("XXX") is False
    
    # Test with an empty string
    assert registry.has("") is False
    
    # Test with None (should raise TypeError)
    try:
        registry.has(None)
        assert False, "Expected TypeError"
    except TypeError:
        pass
    
    # Test with a lowercase currency code (assuming registry is case-sensitive)
    assert registry.has("usd") is False


# LLM-generated content at query #13
#--------------------------

def test_CurrencyRegistry___len__():
    # Create a new registry instance
    registry = CurrencyRegistry()
    
    # Initially, the registry should be empty
    assert len(registry) == 0

    # Add some currencies to the registry
    with registry as register:
        register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        register(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))
        register(Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY))

    # After adding currencies, the length should reflect the count
    assert len(registry) == 3

    # Add another currency
    with registry as register:
        register(Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY))

    # Length should now be 4
    assert len(registry) == 4


# LLM-generated content at query #14
#--------------------------

def test_Currency___setattr__():
    # Test that attempting to set an attribute raises AttributeError
    ccy = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    with pytest.raises(AttributeError):
        ccy.code = "EUR"
    
    with pytest.raises(AttributeError):
        ccy.name = "Euro"
    
    with pytest.raises(AttributeError):
        ccy.decimals = 3
    
    with pytest.raises(AttributeError):
        ccy.type = CurrencyType.CRYPTO
    
    with pytest.raises(AttributeError):
        ccy.quantizer = ZERO
    
    with pytest.raises(AttributeError):
        ccy.hashcache = hash(("EUR", "Euro", 2, CurrencyType.MONEY, ZERO))


# LLM-generated content at query #15
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

    # Test context manager and registration
    with registry1 as register:
        currency = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        register(currency)

    assert len(registry1) == 1
    assert registry1.all == [currency]
    assert registry1.codes == ["USD"]
    assert registry1.codenames == [("USD", "US Dollar")]

    # Test __contains__ and has method
    assert "USD" in registry1
    assert registry1.has("USD") is True
    assert "XYZ" not in registry1
    assert registry1.has("XYZ") is False

    # Test __getitem__ and get method
    assert registry1["USD"].code == "USD"
    assert registry1.get("USD") == currency
    assert registry1.get("XYZ") is None
    assert registry1.get("XYZ", currency) == currency

    # Test CurrencyLookupError
    try:
        registry1["XYZ"]
        assert False, "Expected CurrencyLookupError"
    except CurrencyLookupError:
        pass

    # Test duplicate registration
    try:
        with registry1 as register:
            register(currency)
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #16
#--------------------------

```python
def test_CurrencyRegistry___new__():
    # Create first instance
    instance1 = CurrencyRegistry()
    
    # Create second instance
    instance2 = CurrencyRegistry()
    
    # Check that both instances are the same (singleton behavior)
    assert instance1 is instance2
    
    # Check that the instance is of type CurrencyRegistry
    assert isinstance(instance1, CurrencyRegistry)
    
    # Check that the instance is the singleton instance
    assert CurrencyRegistry._CurrencyRegistry__instance is instance1


# LLM-generated content at query #17
#--------------------------

def test_Currency___setattr__():
    # Test that attributes cannot be modified after creation (frozen dataclass)
    currency = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    with pytest.raises(dataclasses.FrozenInstanceError):
        currency.code = "EUR"
    
    with pytest.raises(dataclasses.FrozenInstanceError):
        currency.name = "Euro"
    
    with pytest.raises(dataclasses.FrozenInstanceError):
        currency.decimals = 3
    
    with pytest.raises(dataclasses.FrozenInstanceError):
        currency.type = CurrencyType.CRYPTO
    
    with pytest.raises(dataclasses.FrozenInstanceError):
        currency.quantizer = Decimal("0.001")
    
    with pytest.raises(dataclasses.FrozenInstanceError):
        currency.hashcache = 12345


# LLM-generated content at query #18
#--------------------------

def test_CurrencyLookupError():
    # Test initialization with a code
    error = CurrencyLookupError("XYZ")
    assert error.code == "XYZ"
    assert str(error) == "Currency identified by code 'XYZ' does not exist"

    # Test initialization with a different code
    error = CurrencyLookupError("USD")
    assert error.code == "USD"
    assert str(error) == "Currency identified by code 'USD' does not exist"


# LLM-generated content at query #19
#--------------------------

Here's a unit test for the `__new__` method of the `CurrencyRegistry` class:


# LLM-generated content at query #20
#--------------------------

```python
def test_Currency___ge__():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)

    assert usd1 >= usd2
    assert not (usdx >= usd1)
    assert usd1 >= usdx


# LLM-generated content at query #21
#--------------------------

def test_CurrencyLookupError():
    error = CurrencyLookupError("XYZ")
    assert error.code == "XYZ"
    assert str(error) == "Currency identified by code 'XYZ' does not exist"


# LLM-generated content at query #22
#--------------------------

```python
def test_CurrencyRegistry___exit__():
    registry = CurrencyRegistry()
    
    # Create a few currencies
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    
    # Add currencies to the registry
    with registry as register:
        register(usd)
        register(eur)
        register(jpy)
    
    # Test that currencies are sorted correctly
    assert registry.codes == ["EUR", "JPY", "USD"]
    assert registry.all == [eur, jpy, usd]
    assert registry.codenames == [("EUR", "Euro"), ("JPY", "Japanese Yen"), ("USD", "US Dollars")]
    
    # Test that context is closed after exiting
    assert registry._CurrencyRegistry__ctx_open == False
    
    # Test that attempting to register outside the context raises an error
    try:
        registry._CurrencyRegistry__register(Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY))
    except ProgrammingError as e:
        assert str(e) == "Can not create currencies outside registry context."
    else:
        assert False, "Expected ProgrammingError"


# LLM-generated content at query #23
#--------------------------

```python
def test_CurrencyRegistry___exit__():
    # Create a CurrencyRegistry instance
    registry = CurrencyRegistry()

    # Create some currencies
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)

    # Enter the registry population context and register currencies
    with registry as register:
        register(eur)
        register(jpy)
        register(usd)

    # Check if the registry is sorted by code
    assert registry.codes == ["EUR", "JPY", "USD"]
    assert registry.codenames == [("EUR", "Euro"), ("JPY", "Japanese Yen"), ("USD", "US Dollars")]
    assert registry.all == [eur, jpy, usd]

    # Check if the context is closed after exiting
    assert registry._CurrencyRegistry__ctx_open is False


# LLM-generated content at query #24
#--------------------------

```python
def test_Currency___ge__():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    
    assert usd1 >= usd2
    assert usd1 >= usdx
    assert usdx >= usd1
    assert jpy >= usd1
    assert usd1 >= jpy
    assert not (usd1 > usd2)
    assert not (usd1 > usdx)
    assert not (usdx > usd1)
    assert not (jpy > usd1)
    assert not (usd1 > jpy)


# LLM-generated content at query #25
#--------------------------

```python
def test_CurrencyRegistry___len__():
    # Initialize the registry
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


# LLM-generated content at query #26
#--------------------------

```python
def test_Currency_quantize():
    # Test with USD (2 decimal places)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.quantize(Decimal("1.005")) == Decimal("1.00")
    assert usd.quantize(Decimal("1.015")) == Decimal("1.02")

    # Test with JPY (0 decimal places)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy.quantize(Decimal("0.5")) == Decimal("0")
    assert jpy.quantize(Decimal("1.5")) == Decimal("2")

    # Test with ZZZ (no fixed precision)
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert zzz.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert zzz.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")


# LLM-generated content at query #27
#--------------------------

def test_CurrencyRegistry___len__():
    # Create a new registry instance
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

    # Add another currency and check again
    with registry as register:
        register(Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY))

    assert len(registry) == 4


# LLM-generated content at query #28
#--------------------------

def test_CurrencyRegistry___contains__():
    # Test with existing currency code
    assert "USD" in Currencies

    # Test with non-existing currency code
    assert "XXX" not in Currencies

    # Test with empty string code
    assert "" not in Currencies

    # Test with lowercase code (should be case-sensitive)
    assert "usd" not in Currencies

    # Test with numeric string (should not be considered as valid)
    assert "123" not in Currencies

    # Test with special characters (should not be considered as valid)
    assert "U$D" not in Currencies

    # Test with None (should not raise exception but return False)
    assert None not in Currencies  # type: ignore


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_CurrencyRegistry___enter__():
    registry = CurrencyRegistry()
    
    # Ensure the context is closed initially
    assert not registry._CurrencyRegistry__ctx_open
    
    # Enter the context and verify the context is open
    add_method = registry.__enter__()
    assert registry._CurrencyRegistry__ctx_open
    
    # Verify the returned method is the __register method
    assert add_method == registry._CurrencyRegistry__register
    
    # Exit the context and verify the context is closed
    registry.__exit__(None, None, None)
    assert not registry._CurrencyRegistry__ctx_open


# LLM-generated content at query #2
#--------------------------

```python
def test_Currency___eq__():
    # Test equality with same currency instances
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2

    # Test inequality with different currencies
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 == usdx)

    # Test inequality with non-Currency object
    assert not (usd1 == "USD")

    # Test inequality with different currency types
    crypto_usd = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert not (usd1 == crypto_usd)

    # Test inequality with different decimals
    usd_zero_decimals = Currency.of("USD", "US Dollars", 0, CurrencyType.MONEY)
    assert not (usd1 == usd_zero_decimals)


# LLM-generated content at query #3
#--------------------------

def test_CurrencyRegistry_has():
    # Test with existing currency code
    assert Currencies.has("USD") == True
    
    # Test with non-existing currency code
    assert Currencies.has("XXX") == False
    
    # Test with empty string
    assert Currencies.has("") == False
    
    # Test with None (should raise TypeError)
    try:
        Currencies.has(None)
        assert False, "Expected TypeError"
    except TypeError:
        pass
    
    # Test with non-string type (should raise TypeError)
    try:
        Currencies.has(123)
        assert False, "Expected TypeError"
    except TypeError:
        pass


# LLM-generated content at query #4
#--------------------------

```python
def test_CurrencyRegistry___contains__():
    registry = CurrencyRegistry()

    # Test that the registry contains known currency codes
    assert "USD" in registry
    assert "EUR" in registry

    # Test that the registry does not contain unknown currency codes
    assert "XXX" not in registry
    assert "YYY" not in registry

    # Test with an empty registry (should return False for any code)
    empty_registry = CurrencyRegistry()
    assert "USD" not in empty_registry
    assert "EUR" not in empty_registry


# LLM-generated content at query #5
#--------------------------

def test_Currency___gt__():
    # Create some currencies for comparison
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    EUR = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    JPY = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    BTC = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)

    # Test same currency (should be equal, not greater)
    assert not (USD > USD)

    # Test different currencies with same type and decimals
    # Order is based on code, name, decimals, type, quantizer
    assert (EUR > USD) == (EUR.code > USD.code)

    # Test different decimals
    assert (JPY > USD) == (JPY.decimals > USD.decimals)

    # Test different types
    assert (BTC > USD) == (BTC.type.value > USD.type.value)

    # Test with non-Currency object
    assert not (USD > "not a currency")


# LLM-generated content at query #6
#--------------------------

```python
def test_CurrencyLookupError():
    code = "XYZ"
    error = CurrencyLookupError(code)
    
    assert error.code == code
    assert str(error) == f"Currency identified by code '{code}' does not exist"


# LLM-generated content at query #7
#--------------------------

```python
def test_CurrencyLookupError():
    code = "XYZ"
    error = CurrencyLookupError(code)
    
    assert error.code == code
    assert str(error) == f"Currency identified by code '{code}' does not exist"
    assert isinstance(error, LookupError)


# LLM-generated content at query #8
#--------------------------

```python
def test_Currency___delattr__():
    # Test that attributes cannot be deleted from Currency instances
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    # Attempt to delete an attribute and expect an AttributeError
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


# LLM-generated content at query #9
#--------------------------

```python
def test_Currency___hash__():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy3 = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    
    assert hash(ccy1) == hash(ccy2)
    assert hash(ccy1) != hash(ccy3)
    assert isinstance(hash(ccy1), int)


# LLM-generated content at query #10
#--------------------------

def test_Currency___setattr__():
    # Test that Currency is immutable and cannot have attributes set
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
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
    assert usd.quantizer == make_quantizer(2)
    assert usd.hashcache == hash(("USD", "US Dollars", 2, CurrencyType.MONEY, make_quantizer(2)))

    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy.code == "JPY"
    assert jpy.name == "Japanese Yen"
    assert jpy.decimals == 0
    assert jpy.type == CurrencyType.MONEY
    assert jpy.quantizer == ZERO
    assert jpy.hashcache == hash(("JPY", "Japanese Yen", 0, CurrencyType.MONEY, ZERO))

    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert zzz.code == "ZZZ"
    assert zzz.name == "Some weird currency"
    assert zzz.decimals == -1
    assert zzz.type == CurrencyType.CRYPTO
    assert zzz.quantizer == MaxPrecisionQuantizer
    assert zzz.hashcache == hash(("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO, MaxPrecisionQuantizer))

    # Test equality and hash
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2
    assert hash(usd1) == hash(usd2)

    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert usd1 != usdx
    assert hash(usd1) != hash(usdx)

    # Test invalid code
    with pytest.raises(ProgrammingError):
        Currency.of("123", "Invalid Code", 2, CurrencyType.MONEY)

    with pytest.raises(ProgrammingError):
        Currency.of("usd", "Invalid Code", 2, CurrencyType.MONEY)

    # Test invalid name
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "", 2, CurrencyType.MONEY)

    with pytest.raises(ProgrammingError):
        Currency.of("USD", " US Dollars ", 2, CurrencyType.MONEY)

    # Test invalid decimals
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", -2, CurrencyType.MONEY)

    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", "two", CurrencyType.MONEY)

    # Test invalid type
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", 2, "Invalid Type")


# LLM-generated content at query #12
#--------------------------

```python
def test_CurrencyLookupError():
    # Test initialization with a specific code
    code = "XYZ"
    error = CurrencyLookupError(code)
    
    # Check that the code attribute is set correctly
    assert error.code == code
    
    # Check that the error message is formatted correctly
    assert str(error) == f"Currency identified by code '{code}' does not exist"


# LLM-generated content at query #13
#--------------------------

Here's a unit test for the `__hash__` method of the `Currency` class:


# LLM-generated content at query #14
#--------------------------

def test_CurrencyRegistry_has():
    # Test with existing currency code
    assert Currencies.has("USD") == True
    
    # Test with non-existing currency code
    assert Currencies.has("XXX") == False
    
    # Test with empty string
    assert Currencies.has("") == False
    
    # Test with lowercase code (should be case sensitive)
    assert Currencies.has("usd") == False
    
    # Test with numeric string
    assert Currencies.has("123") == False
    
    # Test with special characters
    assert Currencies.has("U$D") == False


# LLM-generated content at query #15
#--------------------------

def test_CurrencyRegistry___contains__():
    # Test with a known currency code
    assert "USD" in Currencies
    
    # Test with a non-existent currency code
    assert "XXX" not in Currencies
    
    # Test with empty string
    assert "" not in Currencies
    
    # Test with None (should raise TypeError)
    try:
        None in Currencies
        assert False, "Expected TypeError"
    except TypeError:
        pass
    
    # Test with non-string type (int)
    try:
        123 in Currencies
        assert False, "Expected TypeError"
    except TypeError:
        pass


# LLM-generated content at query #16
#--------------------------

```python
def test_Currency___gt__():
    # Create some currencies for comparison
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    EUR = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    JPY = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    BTC = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)

    # Test same currency (should be False since __gt__ is not implemented and falls back to __eq__)
    assert not (USD > USD)

    # Test different currencies with same type and decimals
    assert (EUR > USD) == (EUR.code > USD.code)  # Lexicographical comparison of codes

    # Test different currencies with same type but different decimals
    assert (JPY > USD) == (JPY.code > USD.code)  # Lexicographical comparison of codes

    # Test different currency types
    assert (BTC > USD) == (BTC.code > USD.code)  # Lexicographical comparison of codes

    # Test with non-Currency object (should raise TypeError)
    try:
        _ = USD > "not_a_currency"
        assert False, "Should have raised TypeError"
    except TypeError:
        pass


# LLM-generated content at query #17
#--------------------------

def test_Currency___delattr__():
    # Create a currency instance
    currency = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    # Test that deleting attributes raises AttributeError
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
    
    # Test that deleting non-existent attribute raises AttributeError
    with pytest.raises(AttributeError):
        del currency.non_existent_attr


# LLM-generated content at query #18
#--------------------------

```python
def test_Currency___repr__():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert repr(usd) == "Currency(code='USD', name='US Dollars', decimals=2, type=<CurrencyType.MONEY: 'Money'>, quantizer=Decimal('1E-2'), hashcache={})".format(hash(usd))

    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert repr(jpy) == "Currency(code='JPY', name='Japanese Yen', decimals=0, type=<CurrencyType.MONEY: 'Money'>, quantizer=Decimal('1E-0'), hashcache={})".format(hash(jpy))

    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert repr(zzz) == "Currency(code='ZZZ', name='Some weird currency', decimals=-1, type=<CurrencyType.CRYPTO: 'Crypto Currency'>, quantizer=Decimal('1E-12'), hashcache={})".format(hash(zzz))


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
    assert usd.quantize(Decimal("1.005")) == Decimal("1.00")
    assert usd.quantize(Decimal("1.015")) == Decimal("1.02")

    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy.code == "JPY"
    assert jpy.name == "Japanese Yen"
    assert jpy.decimals == 0
    assert jpy.type == CurrencyType.MONEY
    assert jpy.quantize(Decimal("0.5")) == Decimal("0")
    assert jpy.quantize(Decimal("1.5")) == Decimal("2")

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

    # Test invalid currency name
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", " US Dollars ", 2, CurrencyType.MONEY)

    # Test invalid decimals
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", -2, CurrencyType.MONEY)

    # Test invalid currency type
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", 2, "InvalidType")


# LLM-generated content at query #20
#--------------------------

def test_Currency_quantize():
    # Test with USD (2 decimals)
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert USD.quantize(Decimal("1.005")) == Decimal("1.00")
    assert USD.quantize(Decimal("1.015")) == Decimal("1.02")
    assert USD.quantize(Decimal("1.025")) == Decimal("1.02")
    assert USD.quantize(Decimal("1.035")) == Decimal("1.04")

    # Test with JPY (0 decimals)
    JPY = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert JPY.quantize(Decimal("0.5")) == Decimal("0")
    assert JPY.quantize(Decimal("1.5")) == Decimal("2")
    assert JPY.quantize(Decimal("2.5")) == Decimal("2")
    assert JPY.quantize(Decimal("3.5")) == Decimal("4")

    # Test with crypto currency (no fixed precision)
    ZZZ = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert ZZZ.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert ZZZ.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")
    assert ZZZ.quantize(Decimal("1.0000000000025")) == Decimal("1.000000000002")
    assert ZZZ.quantize(Decimal("1.0000000000035")) == Decimal("1.000000000004")

    # Test with exact values (should remain unchanged)
    assert USD.quantize(Decimal("1.00")) == Decimal("1.00")
    assert JPY.quantize(Decimal("1")) == Decimal("1")
    assert ZZZ.quantize(Decimal("1.000000000000")) == Decimal("1.000000000000")


# LLM-generated content at query #21
#--------------------------

```python
def test_Currency___ge__():
    # Create two currencies with the same attributes
    currency1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    currency2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    # Assert that currency1 is greater than or equal to currency2
    assert currency1 >= currency2

    # Create a currency with a higher order (different name)
    currency3 = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    
    # Assert that currency1 is not greater than or equal to currency3
    assert not (currency1 >= currency3)

    # Create a currency with a higher order (different decimals)
    currency4 = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    
    # Assert that currency1 is not greater than or equal to currency4
    assert not (currency1 >= currency4)

    # Create a currency with a higher order (different type)
    currency5 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    
    # Assert that currency1 is not greater than or equal to currency5
    assert not (currency1 >= currency5)

    # Create a currency with a higher order (different quantizer)
    currency6 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    currency6.quantizer = Decimal('0.01')
    
    # Assert that currency1 is not greater than or equal to currency6
    assert not (currency1 >= currency6)


# LLM-generated content at query #22
#--------------------------

def test_CurrencyRegistry_has():
    # Test with existing currency code
    assert Currencies.has("USD") is True
    
    # Test with non-existing currency code
    assert Currencies.has("XXX") is False
    
    # Test with empty string
    assert Currencies.has("") is False
    
    # Test with None (should raise TypeError)
    try:
        Currencies.has(None)  # type: ignore
        assert False, "Expected TypeError"
    except TypeError:
        pass
    
    # Test with non-string type
    try:
        Currencies.has(123)  # type: ignore
        assert False, "Expected TypeError"
    except TypeError:
        pass


# LLM-generated content at query #23
#--------------------------

```python
def test_Currency___ge__():
    # Create Currency instances for comparison
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)

    # Test equality
    assert (usd1 >= usd2)
    assert (usd2 >= usd1)

    # Test inequality (ordering based on hashcache)
    assert (usd1 >= jpy) == (hash(usd1) >= hash(jpy))
    assert (jpy >= usd1) == (hash(jpy) >= hash(usd1))

    # Test with different currency types
    assert (usd1 >= zzz) == (hash(usd1) >= hash(zzz))
    assert (zzz >= usd1) == (hash(zzz) >= hash(usd1))

    # Test with same code but different names
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert (usd1 >= usdx) == (hash(usd1) >= hash(usdx))
    assert (usdx >= usd1) == (hash(usdx) >= hash(usd1))


# LLM-generated content at query #24
#--------------------------

```python
def test_CurrencyRegistry___enter__():
    registry = CurrencyRegistry()
    with registry as register:
        assert callable(register)
        assert registry._CurrencyRegistry__ctx_open is True

    assert registry._CurrencyRegistry__ctx_open is False


# LLM-generated content at query #25
#--------------------------

```python
def test_CurrencyRegistry___getitem__():
    registry = CurrencyRegistry()
    
    # Mocking the registry with a known currency
    usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    registry._CurrencyRegistry__registry = {"USD": usd}
    
    # Test retrieving a valid currency
    assert registry["USD"] == usd
    
    # Test retrieving a non-existent currency
    with pytest.raises(CurrencyLookupError):
        registry["XYZ"]


# LLM-generated content at query #26
#--------------------------

```python
def test_CurrencyRegistry_get():
    # Setup: Create a CurrencyRegistry instance and register a currency
    registry = CurrencyRegistry()
    usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    
    # Enter the context and register the currency
    with registry as register:
        register(usd)
    
    # Test case 1: Retrieve an existing currency
    result = registry.get("USD")
    assert result == usd
    
    # Test case 2: Try to retrieve a non-existing currency without a default
    result = registry.get("XXX")
    assert result is None
    
    # Test case 3: Try to retrieve a non-existing currency with a default
    default_currency = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    result = registry.get("XXX", default=default_currency)
    assert result == default_currency
    
    # Test case 4: Retrieve an existing currency with a default (should ignore default)
    result = registry.get("USD", default=default_currency)
    assert result == usd


# LLM-generated content at query #27
#--------------------------

```python
def test_CurrencyRegistry___len__():
    registry = CurrencyRegistry()
    
    # Initially, the registry should be empty
    assert len(registry) == 0

    # Add a currency and check the length
    with registry as register:
        register(Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY))
    
    assert len(registry) == 1

    # Add another currency and check the length
    with registry as register:
        register(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))
    
    assert len(registry) == 2

    # Remove currencies and check the length
    registry = CurrencyRegistry()
    assert len(registry) == 0


# LLM-generated content at query #28
#--------------------------

```python
def test_Currency_quantize():
    # Test with USD (precision 2)
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert USD.quantize(Decimal("1.005")) == Decimal("1.00")
    assert USD.quantize(Decimal("1.015")) == Decimal("1.02")

    # Test with JPY (precision 0)
    JPY = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert JPY.quantize(Decimal("0.5")) == Decimal("0")
    assert JPY.quantize(Decimal("1.5")) == Decimal("2")

    # Test with ZZZ (no fixed precision)
    ZZZ = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert ZZZ.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert ZZZ.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")


# LLM-generated content at query #29
#--------------------------

```python
def test_CurrencyLookupError():
    # Test initialization with a specific code
    code = "XYZ"
    error = CurrencyLookupError(code)
    
    # Assert the code attribute is set correctly
    assert error.code == code
    
    # Assert the error message is formatted correctly
    assert str(error) == f"Currency identified by code '{code}' does not exist"


# LLM-generated content at query #30
#--------------------------

def test_Currency___ge__():
    # Create some currencies for testing
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    JPY = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    ZZZ = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    
    # Test same currency
    assert USD >= USD
    assert JPY >= JPY
    assert ZZZ >= ZZZ
    
    # Test different currencies (order is based on the dataclass fields)
    # USD should be greater than JPY based on field comparison order
    assert USD >= JPY
    assert not JPY >= USD
    
    # ZZZ should be greater than USD and JPY based on field comparison order
    assert ZZZ >= USD
    assert ZZZ >= JPY
    assert not USD >= ZZZ
    assert not JPY >= ZZZ
    
    # Test with non-Currency object
    assert USD >= "USD" is NotImplemented
    assert JPY >= 123 is NotImplemented


# LLM-generated content at query #31
#--------------------------

```python
def test_Currency___gt__():
    # Create some currencies for comparison
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    btc = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)

    # Test ordering based on code (since Currency is ordered and uses code for comparison)
    assert eur > usd  # 'EUR' > 'USD' alphabetically
    assert jpy > eur  # 'JPY' > 'EUR' alphabetically
    assert btc > jpy  # 'BTC' > 'JPY' alphabetically

    # Test against non-Currency objects
    assert usd > "USD"  # Currency should be greater than string
    assert usd > 1  # Currency should be greater than int

    # Test equal currencies
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not usd > usd2  # Equal currencies should not be greater
    assert not usd2 > usd  # Equal currencies should not be greater


# LLM-generated content at query #32
#--------------------------

def test_CurrencyRegistry___exit__():
    # Create a test registry
    registry = CurrencyRegistry()

    # Create some test currencies
    usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)

    # Enter the context and register currencies in non-alphabetical order
    with registry as register:
        register(jpy)
        register(usd)
        register(eur)

    # Test that currencies are sorted alphabetically by code
    assert registry.codes == ["EUR", "JPY", "USD"]
    assert registry.all[0].code == "EUR"
    assert registry.all[1].code == "JPY"
    assert registry.all[2].code == "USD"

    # Test that codenames are also sorted
    assert registry.codenames == [
        ("EUR", "Euro"),
        ("JPY", "Japanese Yen"),
        ("USD", "US Dollar")
    ]

    # Test that context is closed after exit
    assert registry._CurrencyRegistry__ctx_open is False

    # Test that trying to register outside context raises error
    try:
        registry._CurrencyRegistry__register(usd)
        assert False, "Should have raised ProgrammingError"
    except ProgrammingError:
        pass

    # Test that registry is properly ordered
    assert list(registry._CurrencyRegistry__registry.keys()) == ["EUR", "JPY", "USD"]


# LLM-generated content at query #33
#--------------------------

```python
def test_Currency_quantize():
    # Test with USD (precision 2)
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert USD.quantize(Decimal("1.005")) == Decimal("1.00")
    assert USD.quantize(Decimal("1.015")) == Decimal("1.02")

    # Test with JPY (precision 0)
    JPY = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert JPY.quantize(Decimal("0.5")) == Decimal("0")
    assert JPY.quantize(Decimal("1.5")) == Decimal("2")

    # Test with ZZZ (precision -1, max precision)
    ZZZ = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert ZZZ.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert ZZZ.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")


# LLM-generated content at query #34
#--------------------------

```python
def test_Currency___eq__():
    # Test equality with identical currencies
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2

    # Test equality with different currencies
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 == usdx)

    # Test equality with non-Currency objects
    assert not (usd1 == "USD")

    # Test equality with different currency types
    crypto = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    assert not (usd1 == crypto)

    # Test equality with different decimal places
    usd_diff_decimals = Currency.of("USD", "US Dollars", 0, CurrencyType.MONEY)
    assert not (usd1 == usd_diff_decimals)

    # Test equality with different names
    usd_diff_name = Currency.of("USD", "American Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 == usd_diff_name)


# LLM-generated content at query #35
#--------------------------

```python
def test_Currency___lt__():
    # Create some currencies for comparison
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    
    # Test that USD is less than EUR (based on code order)
    assert usd < eur
    
    # Test that JPY is less than USD (based on code order)
    assert jpy < usd
    
    # Test that EUR is not less than JPY (based on code order)
    assert not (eur < jpy)
    
    # Test that a currency is not less than itself
    assert not (usd < usd)


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_CurrencyRegistry___getitem__():
    # Test with existing currency
    usd = Currencies["USD"]
    assert usd.code == "USD"
    assert usd.name == "US Dollar"
    assert usd.type == CurrencyType.MONEY

    # Test with non-existing currency
    try:
        Currencies["NON-EXISTING"]
        assert False, "Expected CurrencyLookupError"
    except CurrencyLookupError as e:
        assert e.code == "NON-EXISTING"
        assert str(e) == "Currency identified by code 'NON-EXISTING' does not exist"

    # Test case sensitivity
    try:
        Currencies["usd"]
        assert False, "Expected CurrencyLookupError"
    except CurrencyLookupError:
        pass

    # Test empty string
    try:
        Currencies[""]
        assert False, "Expected CurrencyLookupError"
    except CurrencyLookupError:
        pass


# LLM-generated content at query #2
#--------------------------

```python
def test_Currency___eq__():
    # Create two identical Currency instances
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)

    # Create a different Currency instance
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)

    # Test equality between identical instances
    assert usd1 == usd2

    # Test equality between different instances
    assert not (usd1 == usdx)

    # Test equality with a non-Currency object
    assert not (usd1 == "USD")

    # Test equality with None
    assert not (usd1 == None)


# LLM-generated content at query #3
#--------------------------

def test_Currency___eq__():
    # Create two identical currencies
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    # Create a different currency with same code but different name
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    
    # Create a completely different currency
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    
    # Test equality between identical currencies
    assert usd1 == usd2
    
    # Test inequality with same code but different name
    assert not (usd1 == usdx)
    
    # Test inequality with completely different currency
    assert not (usd1 == jpy)
    
    # Test inequality with non-Currency object
    assert not (usd1 == "USD")
    
    # Test inequality with None
    assert not (usd1 == None)


# LLM-generated content at query #4
#--------------------------

```python
def test_Currency___eq__():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    
    assert usd1 == usd2
    assert not (usd1 == usdx)
    assert not (usd1 == jpy)
    assert not (usd1 == "not a currency")


# LLM-generated content at query #5
#--------------------------

```python
def test_Currency___eq__():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)

    # Test equality with same currency
    assert usd1 == usd2

    # Test inequality with different currency
    assert not (usd1 == usdx)

    # Test inequality with different currency code
    assert not (usd1 == jpy)

    # Test inequality with non-Currency object
    assert not (usd1 == "USD")


# LLM-generated content at query #6
#--------------------------

```python
def test_Currency___eq__():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)

    assert usd1 == usd2
    assert not (usd1 == usdx)
    assert not (usd1 == jpy)
    assert not (usd1 == "not a Currency object")


# LLM-generated content at query #7
#--------------------------

```python
def test_Currency___eq__():
    currency1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    currency2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    currency3 = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    currency4 = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)

    assert currency1 == currency2
    assert not (currency1 == currency3)
    assert not (currency1 == currency4)


# LLM-generated content at query #8
#--------------------------

```python
def test_Currency___eq__():
    # Test equality with same currency attributes
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2

    # Test inequality with different currency attributes
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert usd1 != usdx

    # Test equality with same hash cache
    usd3 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd3

    # Test inequality with different hash cache
    usdy = Currency.of("USD", "US Dollars", 0, CurrencyType.MONEY)
    assert usd1 != usdy

    # Test inequality with non-Currency object
    assert usd1 != "Not a Currency object"


# LLM-generated content at query #9
#--------------------------

```python
def test_Currency___eq__():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    
    assert usd1 == usd2
    assert not (usd1 == usdx)
    assert not (usd1 == "not a currency")


# LLM-generated content at query #10
#--------------------------

```python
def test_Currency___eq__():
    # Create two identical Currency instances
    currency1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    currency2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)

    # Create a different Currency instance
    currency3 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)

    # Test equality with identical instances
    assert currency1 == currency2

    # Test equality with different instances
    assert not (currency1 == currency3)

    # Test equality with non-Currency object
    assert not (currency1 == "USD")

    # Test equality with None
    assert not (currency1 == None)


# LLM-generated content at query #11
#--------------------------

def test_CurrencyLookupError():
    error = CurrencyLookupError("XYZ")
    assert error.code == "XYZ"
    assert str(error) == "Currency identified by code 'XYZ' does not exist"


# LLM-generated content at query #12
#--------------------------

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
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        register(usd)
    
    # Verify registration worked
    assert len(registry1) == 1
    assert registry1.all == [usd]
    assert registry1.codes == ["USD"]
    assert registry1.codenames == [("USD", "US Dollar")]

    # Test contains and has methods
    assert "USD" in registry1
    assert registry1.has("USD")
    assert "EUR" not in registry1
    assert not registry1.has("EUR")

    # Test getitem and get methods
    assert registry1["USD"] == usd
    assert registry1.get("USD") == usd
    assert registry1.get("EUR") is None
    assert registry1.get("EUR", default=usd) == usd

    # Test duplicate registration
    with pytest.raises(ValueError):
        with registry1 as register:
            register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))

    # Test registration outside context
    with pytest.raises(ProgrammingError):
        registry1.__register(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))

    # Test lookup error
    with pytest.raises(CurrencyLookupError):
        registry1["XYZ"]


# LLM-generated content at query #13
#--------------------------

```python
def test_Currency___delattr__():
    # Create a Currency instance
    currency = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)

    # Attempt to delete an attribute
    with pytest.raises(AttributeError):
        del currency.code

    # Verify that the attribute still exists
    assert currency.code == "USD"

    # Attempt to delete a non-existing attribute
    with pytest.raises(AttributeError):
        del currency.non_existing_attribute


# LLM-generated content at query #14
#--------------------------

```python
def test_Currency___le__():
    # Create two currencies with the same attributes
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)

    # Create a different currency
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)

    # Test that usd1 <= usd2 since they are equal
    assert usd1 <= usd2

    # Test that usd1 is not less than or equal to jpy
    assert not (usd1 <= jpy)

    # Test that jpy is not less than or equal to usd1
    assert not (jpy <= usd1)

    # Test with a different currency type
    btc = Currency.of("BTC", "Bitcoin", -1, CurrencyType.CRYPTO)
    assert not (usd1 <= btc)
    assert not (btc <= usd1)


# LLM-generated content at query #15
#--------------------------

def test_CurrencyRegistry___exit__():
    # Create a new registry instance
    registry = CurrencyRegistry()
    
    # Create some test currencies
    usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    
    # Enter the context and register currencies in unsorted order
    with registry as register:
        register(jpy)
        register(usd)
        register(eur)
    
    # After exiting context, verify everything is properly sorted
    assert registry.codes == ["EUR", "JPY", "USD"]
    assert registry.all == [eur, jpy, usd]
    assert registry.codenames == [("EUR", "Euro"), ("JPY", "Japanese Yen"), ("USD", "US Dollar")]
    assert registry["EUR"] == eur
    assert registry["JPY"] == jpy
    assert registry["USD"] == usd
    
    # Verify the registry is properly closed after context exit
    assert registry._CurrencyRegistry__ctx_open is False
    
    # Verify trying to register outside context raises error
    try:
        registry._CurrencyRegistry__register(usd)
        assert False, "Should have raised ProgrammingError"
    except ProgrammingError:
        pass


# LLM-generated content at query #16
#--------------------------

def test_Currency___ge__():
    # Create some currencies for testing
    USD1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    USD2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    JPY = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    ZZZ = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)

    # Test equal currencies
    assert USD1 >= USD2
    assert USD2 >= USD1

    # Test different currencies (should use the frozen dataclass ordering)
    assert USD1 >= JPY or JPY >= USD1  # One of these must be true
    assert USD1 >= ZZZ or ZZZ >= USD1  # One of these must be true

    # Test with non-Currency objects
    class FakeCurrency:
        pass

    fake = FakeCurrency()
    assert not (USD1 >= fake)  # Should return NotImplemented which evaluates to False in comparison


# LLM-generated content at query #17
#--------------------------

def test_CurrencyRegistry___new__():
    # Test that __new__ returns the same instance (singleton pattern)
    instance1 = CurrencyRegistry()
    instance2 = CurrencyRegistry()
    assert instance1 is instance2

    # Test that the instance is of type CurrencyRegistry
    assert isinstance(instance1, CurrencyRegistry)

    # Test that the instance has the expected attributes
    assert hasattr(instance1, "_CurrencyRegistry__registry")
    assert hasattr(instance1, "_CurrencyRegistry__currencies")
    assert hasattr(instance1, "_CurrencyRegistry__codes")
    assert hasattr(instance1, "_CurrencyRegistry__codenames")
    assert hasattr(instance1, "_CurrencyRegistry__ctx_open")

    # Test that the singleton instance is properly set
    assert CurrencyRegistry._CurrencyRegistry__instance is instance1


# LLM-generated content at query #18
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
    assert currency.hashcache == hash(("USD", "US Dollars", 2, CurrencyType.MONEY, make_quantizer(2)))

    # Test currency with zero decimals
    currency_zero_decimals = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert currency_zero_decimals.quantizer == ZERO

    # Test currency with negative decimals
    currency_neg_decimals = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert currency_neg_decimals.quantizer == MaxPrecisionQuantizer

    # Test equality and hash
    currency1 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    currency2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    currency3 = Currency.of("EUR", "Euro", 1, CurrencyType.MONEY)
    assert currency1 == currency2
    assert hash(currency1) == hash(currency2)
    assert currency1 != currency3
    assert hash(currency1) != hash(currency3)

    # Test quantize method
    assert currency.quantize(Decimal("1.005")) == Decimal("1.00")
    assert currency.quantize(Decimal("1.015")) == Decimal("1.02")

    # Test invalid code raises ProgrammingError
    with pytest.raises(ProgrammingError):
        Currency.of("usd", "US Dollars", 2, CurrencyType.MONEY)  # lowercase code

    with pytest.raises(ProgrammingError):
        Currency.of("US1", "US Dollars", 2, CurrencyType.MONEY)  # numeric code

    # Test invalid name raises ProgrammingError
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "", 2, CurrencyType.MONEY)  # empty name

    with pytest.raises(ProgrammingError):
        Currency.of("USD", " US Dollars ", 2, CurrencyType.MONEY)  # name with spaces

    # Test invalid decimals raises ProgrammingError
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", -2, CurrencyType.MONEY)  # decimals < -1

    # Test invalid type raises ProgrammingError
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", 2, "InvalidType")  # invalid type


# LLM-generated content at query #19
#--------------------------

def test_Currency___delattr__():
    # Test that deleting attributes is not allowed (frozen dataclass)
    currency = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    # Try deleting code attribute
    with pytest.raises(AttributeError):
        del currency.code
        
    # Try deleting name attribute
    with pytest.raises(AttributeError):
        del currency.name
        
    # Try deleting decimals attribute
    with pytest.raises(AttributeError):
        del currency.decimals
        
    # Try deleting type attribute
    with pytest.raises(AttributeError):
        del currency.type
        
    # Try deleting quantizer attribute
    with pytest.raises(AttributeError):
        del currency.quantizer
        
    # Try deleting hashcache attribute
    with pytest.raises(AttributeError):
        del currency.hashcache
        
    # Try deleting non-existent attribute
    with pytest.raises(AttributeError):
        del currency.non_existent


# LLM-generated content at query #20
#--------------------------

def test_CurrencyRegistry___enter__():
    registry = CurrencyRegistry()
    
    # Test that __enter__ returns the __register method
    with registry as register:
        assert callable(register)
        assert register.__name__ == "__register"
        
    # Test that context manager properly sets __ctx_open flag
    assert not registry._CurrencyRegistry__ctx_open  # pylint: disable=protected-access
    
    # Verify that __enter__ sets the context flag correctly
    with registry as register:
        assert registry._CurrencyRegistry__ctx_open  # pylint: disable=protected-access
        
    # Verify flag is reset after context
    assert not registry._CurrencyRegistry__ctx_open  # pylint: disable=protected-access


# LLM-generated content at query #21
#--------------------------

def test_Currency___hash__():
    # Test that two identical currencies have the same hash
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert hash(usd1) == hash(usd2)

    # Test that different currencies have different hashes
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert hash(usd1) != hash(jpy)

    # Test that currencies with different codes have different hashes
    usdx = Currency.of("USX", "US Dollars", 2, CurrencyType.MONEY)
    assert hash(usd1) != hash(usdx)

    # Test that currencies with different names have different hashes
    usd_alt_name = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert hash(usd1) != hash(usd_alt_name)

    # Test that currencies with different decimals have different hashes
    usd_diff_decimals = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert hash(usd1) != hash(usd_diff_decimals)

    # Test that currencies with different types have different hashes
    usd_diff_type = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert hash(usd1) != hash(usd_diff_type)


# LLM-generated content at query #22
#--------------------------

def test_Currency___hash__():
    # Create two identical currencies
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    # Create a different currency with same code but different name
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    
    # Create a completely different currency
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    
    # Test that identical currencies have same hash
    assert hash(usd1) == hash(usd2)
    
    # Test that similar but not identical currencies have different hashes
    assert hash(usd1) != hash(usdx)
    
    # Test that completely different currencies have different hashes
    assert hash(usd1) != hash(jpy)
    
    # Test that hash is consistent with equality
    assert (usd1 == usd2) == (hash(usd1) == hash(usd2))
    assert (usd1 == usdx) == (hash(usd1) == hash(usdx))
    
    # Test that hash returns the precomputed hashcache
    assert hash(usd1) == usd1.hashcache
    assert hash(usd2) == usd2.hashcache
    assert hash(usdx) == usdx.hashcache


# LLM-generated content at query #23
#--------------------------

def test_CurrencyRegistry___contains__():
    # Test with existing currency code
    assert "USD" in Currencies

    # Test with non-existing currency code
    assert "XXX" not in Currencies

    # Test with empty string
    assert "" not in Currencies

    # Test with lowercase code (should be case-sensitive)
    assert "usd" not in Currencies

    # Test with numeric string
    assert "123" not in Currencies

    # Test with special characters
    assert "U$D" not in Currencies

    # Test with None (should raise TypeError)
    try:
        None in Currencies
        assert False, "Should have raised TypeError"
    except TypeError:
        pass

    # Test with non-string type
    try:
        123 in Currencies
        assert False, "Should have raised TypeError"
    except TypeError:
        pass


# LLM-generated content at query #24
#--------------------------

Here's the unit test for the `__lt__` method of the `Currency` class:


# LLM-generated content at query #25
#--------------------------

def test_CurrencyRegistry___enter__():
    registry = CurrencyRegistry()
    with registry as register:
        assert callable(register)
        assert registry._CurrencyRegistry__ctx_open is True

    assert registry._CurrencyRegistry__ctx_open is False


# LLM-generated content at query #26
#--------------------------

```python
def test_CurrencyRegistry___contains__():
    # Create a CurrencyRegistry instance
    registry = CurrencyRegistry()

    # Add a currency to the registry
    with registry as register:
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        register(usd)

    # Test that the currency is in the registry
    assert "USD" in registry

    # Test that a non-existent currency is not in the registry
    assert "XYZ" not in registry


# LLM-generated content at query #27
#--------------------------

```python
def test_Currency___ge__():
    # Create two Currency instances
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)

    # Test equality
    assert usd1 >= usd2

    # Test inequality (usd1 and jpy are different)
    assert not (usd1 >= jpy)

    # Test with a non-Currency object
    non_currency = "USD"
    assert not (usd1 >= non_currency)


# LLM-generated content at query #28
#--------------------------

```python
def test_Currency___repr__():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert repr(usd) == "Currency(code='USD', name='US Dollars', decimals=2, type=<CurrencyType.MONEY: 'Money'>, quantizer=Decimal('0.01'), hashcache=hash(('USD', 'US Dollars', 2, CurrencyType.MONEY, Decimal('0.01'))))"

    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert repr(jpy) == "Currency(code='JPY', name='Japanese Yen', decimals=0, type=<CurrencyType.MONEY: 'Money'>, quantizer=Decimal('0'), hashcache=hash(('JPY', 'Japanese Yen', 0, CurrencyType.MONEY, Decimal('0'))))"

    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert repr(zzz) == "Currency(code='ZZZ', name='Some weird currency', decimals=-1, type=<CurrencyType.CRYPTO: 'Crypto Currency'>, quantizer=MaxPrecisionQuantizer, hashcache=hash(('ZZZ', 'Some weird currency', -1, CurrencyType.CRYPTO, MaxPrecisionQuantizer)))"


# LLM-generated content at query #29
#--------------------------

def test_CurrencyRegistry_has():
    # Test with existing currency code
    assert Currencies.has("USD") == True
    
    # Test with non-existing currency code
    assert Currencies.has("XXX") == False
    
    # Test with empty string
    assert Currencies.has("") == False
    
    # Test with lowercase code (should fail since registry expects uppercase)
    assert Currencies.has("usd") == False
    
    # Test with numeric string
    assert Currencies.has("123") == False
    
    # Test with special characters
    assert Currencies.has("US$") == False


# LLM-generated content at query #30
#--------------------------

Here's a unit test for the `__hash__` method of the `Currency` class:


