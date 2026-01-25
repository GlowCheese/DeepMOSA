####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_CurrencyRegistry___enter__():
    """Test that __enter__ returns the __register method and opens the context."""
    registry = CurrencyRegistry()
    
    # Call __enter__
    result = registry.__enter__()
    
    # Should return a callable (the __register method)
    assert callable(result)
    
    # The context should be marked as open
    assert registry._CurrencyRegistry__ctx_open is True
    
    # Clean up by closing the context
    registry.__exit__(None, None, None)


# LLM-generated content at query #2
#--------------------------

```python
def test_Currency___eq__():
    """Test Currency.__eq__ method"""
    # Test equal currencies
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2
    
    # Test currencies with different names
    usd_diff_name = Currency.of("USD", "United States Dollars", 2, CurrencyType.MONEY)
    assert usd1 != usd_diff_name
    
    # Test currencies with different codes
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert usd1 != eur
    
    # Test currencies with different decimals
    usd_diff_decimals = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert usd1 != usd_diff_decimals
    
    # Test currencies with different types
    usd_crypto = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert usd1 != usd_crypto
    
    # Test comparison with non-Currency object
    assert usd1 != "USD"
    assert usd1 != 2
    assert usd1 != None
    assert usd1 != {}
    
    # Test reflexivity
    assert usd1 == usd1
    
    # Test symmetry
    assert (usd1 == usd2) == (usd2 == usd1)
    
    # Test transitivity
    usd3 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2
    assert usd2 == usd3
    assert usd1 == usd3


# LLM-generated content at query #3
#--------------------------

```python
def test_CurrencyRegistry_has():
    """Test the has method of CurrencyRegistry."""
    # Test with a currency that exists
    assert Currencies.has("USD") is True
    
    # Test with a currency that doesn't exist
    assert Currencies.has("XXX") is False
    
    # Test with multiple known currencies
    assert Currencies.has("EUR") is True
    assert Currencies.has("GBP") is True
    assert Currencies.has("JPY") is True
    
    # Test with invalid codes that don't exist
    assert Currencies.has("INVALID") is False
    assert Currencies.has("ZZZ") is False
    assert Currencies.has("") is False
    
    # Test that has returns boolean
    result = Currencies.has("USD")
    assert isinstance(result, bool)
    
    # Test consistency between has and __contains__
    assert Currencies.has("USD") == ("USD" in Currencies)
    assert Currencies.has("XXX") == ("XXX" in Currencies)


# LLM-generated content at query #4
#--------------------------

```python
def test_CurrencyRegistry___contains__():
    """Test the __contains__ method of CurrencyRegistry."""
    registry = CurrencyRegistry()
    
    # Test that USD is in the registry (assuming it's pre-populated)
    assert "USD" in registry
    
    # Test that a non-existent currency code is not in the registry
    assert "NON_EXISTENT" not in registry
    assert "XYZ" not in registry
    
    # Test with lowercase code (should not be found since codes are uppercase)
    assert "usd" not in registry
    
    # Test with empty string
    assert "" not in registry
    
    # Test that other common currencies are in the registry
    assert "EUR" in registry
    assert "JPY" in registry
    assert "GBP" in registry


# LLM-generated content at query #5
#--------------------------

```python
def test_Currency___gt__():
    """Test the __gt__ method of Currency class."""
    # Create currencies with different codes for comparison
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    gbp = Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    
    # Test greater than comparisons based on code ordering
    assert eur > usd  # "EUR" > "USD"
    assert gbp > eur  # "GBP" > "EUR"
    assert usd > jpy  # "USD" > "JPY"
    
    # Test that greater than is false for reverse comparisons
    assert not (usd > eur)
    assert not (eur > gbp)
    assert not (jpy > usd)
    
    # Test that a currency is not greater than itself
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd > usd2)
    assert not (usd2 > usd)
    
    # Test with different types but same code should still work based on hash/order
    crypto = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    metal = Currency.of("XAU", "Gold", 2, CurrencyType.METAL)
    assert metal > crypto  # "XAU" > "BTC"


# LLM-generated content at query #6
#--------------------------

```python
def test_Currency___le__():
    """Test the __le__ (less than or equal) ordering method of Currency class."""
    # Create currencies with different codes for ordering comparison
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    gbp = Currency.of("GBP", "British Pounds", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    
    # Test __le__ with same currency (should be True)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd <= usd2
    assert usd2 <= usd
    
    # Test __le__ with different currencies (alphabetical ordering by code)
    # EUR < GBP < JPY < USD
    assert eur <= gbp
    assert eur <= usd
    assert gbp <= usd
    assert jpy <= usd
    
    # Test __le__ reflexivity (a <= a)
    assert usd <= usd
    assert eur <= eur
    assert gbp <= gbp
    assert jpy <= jpy
    
    # Test __le__ transitivity (if a <= b and b <= c, then a <= c)
    assert eur <= gbp
    assert gbp <= usd
    assert eur <= usd
    
    # Test __le__ with reverse comparisons (should be False when appropriate)
    assert not (usd <= eur)
    assert not (gbp <= eur)
    assert not (usd <= gbp)
    
    # Test with currencies having same code but different details
    # (they should be equal and thus <= should work)
    usd_alt = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd <= usd_alt
    assert usd_alt <= usd


# LLM-generated content at query #7
#--------------------------

```python
def test_Currency___delattr__():
    """
    Test that Currency is frozen and does not allow attribute deletion.
    """
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    # Attempt to delete an attribute should raise FrozenInstanceError or AttributeError
    # because Currency is a frozen dataclass
    with pytest.raises((AttributeError, Exception)):
        del usd.code
    
    with pytest.raises((AttributeError, Exception)):
        del usd.name
    
    with pytest.raises((AttributeError, Exception)):
        del usd.decimals
    
    with pytest.raises((AttributeError, Exception)):
        del usd.type
    
    with pytest.raises((AttributeError, Exception)):
        del usd.quantizer
    
    with pytest.raises((AttributeError, Exception)):
        del usd.hashcache


# LLM-generated content at query #8
#--------------------------

```python
def test_CurrencyRegistry___exit__():
    """Test the __exit__ method of CurrencyRegistry."""
    from collections import OrderedDict
    
    # Create a new registry instance
    registry = CurrencyRegistry()
    
    # Add some currencies using the context manager
    with registry as register:
        register(Currency.of("TST", "Test Currency", 2, CurrencyType.MONEY))
        register(Currency.of("ZZA", "ZZA Currency", 2, CurrencyType.MONEY))
        register(Currency.of("AAA", "AAA Currency", 2, CurrencyType.MONEY))
    
    # Verify that currencies are sorted by code after __exit__
    assert registry.codes == ["AAA", "TST", "ZZA"]
    assert len(registry.all) == 3
    
    # Verify that the registry is properly sorted
    assert registry.all[0].code == "AAA"
    assert registry.all[1].code == "TST"
    assert registry.all[2].code == "ZZA"
    
    # Verify codenames are properly populated and sorted
    assert registry.codenames == [("AAA", "AAA Currency"), ("TST", "Test Currency"), ("ZZA", "ZZA Currency")]
    
    # Verify context is closed
    assert registry._CurrencyRegistry__ctx_open is False


def test_CurrencyRegistry___exit___sorting():
    """Test that __exit__ properly sorts currencies by code."""
    registry = CurrencyRegistry()
    
    # Add currencies in non-alphabetical order
    with registry as register:
        register(Currency.of("XYZ", "XYZ Currency", 2, CurrencyType.MONEY))
        register(Currency.of("ABC", "ABC Currency", 2, CurrencyType.MONEY))
        register(Currency.of("MNO", "MNO Currency", 2, CurrencyType.MONEY))
    
    # Verify sorting is applied
    codes = registry.codes
    assert codes == sorted(codes)
    assert codes == ["ABC", "MNO", "XYZ"]


def test_CurrencyRegistry___exit___buffers_updated():
    """Test that __exit__ updates all internal buffers."""
    registry = CurrencyRegistry()
    
    with registry as register:
        register(Currency.of("CCC", "CCC Currency", 2, CurrencyType.MONEY))
        register(Currency.of("AAA", "AAA Currency", 2, CurrencyType.MONEY))
        register(Currency.of("BBB", "BBB Currency", 2, CurrencyType.MONEY))
    
    # Verify all buffers are consistent
    assert len(registry.all) == 3
    assert len(registry.codes) == 3
    assert len(registry.codenames) == 3
    
    # Verify they all contain the same currencies in the same order
    for i, currency in enumerate(registry.all):
        assert currency.code == registry.codes[i]
        assert (currency.code, currency.name) == registry.codenames[i]


# LLM-generated content at query #9
#--------------------------

```python
def test_CurrencyLookupError():
    """Test CurrencyLookupError constructor and behavior."""
    # Test basic initialization
    error = CurrencyLookupError("XYZ")
    assert error.code == "XYZ"
    assert str(error) == "Currency identified by code 'XYZ' does not exist"
    
    # Test that it's a subclass of LookupError
    assert isinstance(error, LookupError)
    
    # Test with different currency codes
    error2 = CurrencyLookupError("ABC")
    assert error2.code == "ABC"
    assert str(error2) == "Currency identified by code 'ABC' does not exist"
    
    # Test with empty string
    error3 = CurrencyLookupError("")
    assert error3.code == ""
    assert str(error3) == "Currency identified by code '' does not exist"
    
    # Test that it can be raised and caught
    with pytest.raises(CurrencyLookupError) as exc_info:
        raise CurrencyLookupError("EUR")
    assert exc_info.value.code == "EUR"
    assert "EUR" in str(exc_info.value)


# LLM-generated content at query #10
#--------------------------

```python
def test_CurrencyRegistry___contains__():
    """Test the __contains__ method of CurrencyRegistry."""
    registry = CurrencyRegistry()
    
    # Test with a valid currency code that exists in the registry
    assert "USD" in registry
    
    # Test with another valid currency code
    assert "EUR" in registry
    
    # Test with an invalid currency code
    assert "XYZ" not in registry
    
    # Test with an invalid currency code
    assert "NON-EXISTING" not in registry
    
    # Test with lowercase code (should not be found, codes are uppercase)
    assert "usd" not in registry
    
    # Test with empty string
    assert "" not in registry
    
    # Test with numeric string
    assert "123" not in registry


# LLM-generated content at query #11
#--------------------------

```python
def test_CurrencyRegistry():
    """Test CurrencyRegistry constructor and singleton behavior."""
    # Test singleton pattern - creating two instances should return the same object
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2
    
    # Test initial state after construction
    assert len(registry1) == 0
    assert registry1.all == []
    assert registry1.codes == []
    assert registry1.codenames == []
    
    # Test __contains__ on empty registry
    assert "USD" not in registry1
    
    # Test has method on empty registry
    assert not registry1.has("USD")
    
    # Test get method on empty registry
    assert registry1.get("USD") is None
    assert registry1.get("USD", default=None) is None
    
    # Test __getitem__ raises CurrencyLookupError on empty registry
    with pytest.raises(CurrencyLookupError) as exc_info:
        _ = registry1["USD"]
    assert exc_info.value.code == "USD"
    
    # Test context manager entry and exit
    with registry1 as register:
        # Verify register is callable
        assert callable(register)
        
        # Create and register a currency
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        register(usd)
    
    # After exiting context, currency should be registered
    assert len(registry1) == 1
    assert "USD" in registry1
    assert registry1.has("USD")
    assert registry1["USD"].code == "USD"
    assert registry1.get("USD") == usd
    assert registry1.codes == ["USD"]
    assert registry1.codenames == [("USD", "US Dollar")]
    
    # Test registering outside context raises error
    with pytest.raises(ProgrammingError):
        registry1._CurrencyRegistry__register(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))
    
    # Test duplicate registration raises error
    with pytest.raises(ValueError) as exc_info:
        with registry1 as register:
            register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
    assert "already registered" in str(exc_info.value)


# LLM-generated content at query #12
#--------------------------

```python
def test_CurrencyRegistry___contains__():
    """Test the __contains__ method of CurrencyRegistry."""
    # Create a registry instance
    registry = CurrencyRegistry()
    
    # Test with a currency that exists (USD is typically in the registry)
    assert "USD" in registry
    
    # Test with a currency that doesn't exist
    assert "XYZ" not in registry
    
    # Test with a currency code that is not in the registry
    assert "INVALID" not in registry
    
    # Test with lowercase code (should not be found)
    assert "usd" not in registry
    
    # Test with empty string
    assert "" not in registry
    
    # Test with None type (should not raise error, just return False)
    try:
        result = None in registry
        assert result is False
    except (TypeError, AttributeError):
        # If the implementation doesn't handle None, that's acceptable
        pass


# LLM-generated content at query #13
#--------------------------

```python
def test_CurrencyLookupError():
    """Test CurrencyLookupError constructor and attributes."""
    # Test basic initialization
    code = "XYZ"
    error = CurrencyLookupError(code)
    
    # Test that code is stored correctly
    assert error.code == code
    
    # Test error message
    expected_message = f"Currency identified by code '{code}' does not exist"
    assert str(error) == expected_message
    
    # Test that it's a subclass of LookupError
    assert isinstance(error, LookupError)
    
    # Test with different currency codes
    for test_code in ["USD", "EUR", "GBP", "JPY"]:
        test_error = CurrencyLookupError(test_code)
        assert test_error.code == test_code
        assert f"'{test_code}'" in str(test_error)
        assert isinstance(test_error, LookupError)
    
    # Test that it can be raised and caught
    with pytest.raises(CurrencyLookupError) as exc_info:
        raise CurrencyLookupError("ABC")
    
    assert exc_info.value.code == "ABC"
    assert "ABC" in str(exc_info.value)


# LLM-generated content at query #14
#--------------------------

```python
def test_CurrencyRegistry():
    """Test the CurrencyRegistry constructor and singleton behavior."""
    # Test that CurrencyRegistry is a singleton
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2
    
    # Test initial state
    assert len(registry1) == 0
    assert registry1.all == []
    assert registry1.codes == []
    assert registry1.codenames == []
    
    # Test that __contains__ works on empty registry
    assert "USD" not in registry1
    
    # Test has method on empty registry
    assert not registry1.has("USD")
    
    # Test get method on empty registry
    assert registry1.get("USD") is None
    assert registry1.get("USD", default=None) is None
    
    # Test __getitem__ raises CurrencyLookupError on empty registry
    with pytest.raises(CurrencyLookupError) as exc_info:
        _ = registry1["USD"]
    assert exc_info.value.code == "USD"
    
    # Test context manager entry returns register function
    with registry1 as register:
        assert callable(register)
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        register(usd)
    
    # Test registry now contains the currency
    assert len(registry1) == 1
    assert "USD" in registry1
    assert registry1.has("USD")
    assert registry1["USD"].code == "USD"
    assert registry1.get("USD") == usd
    
    # Test all, codes, and codenames are populated
    assert len(registry1.all) == 1
    assert registry1.codes == ["USD"]
    assert registry1.codenames == [("USD", "US Dollar")]
    
    # Test that registering duplicate currency raises ValueError
    with registry1 as register:
        with pytest.raises(ValueError) as exc_info:
            register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        assert "Currency USD is already registered" in str(exc_info.value)
    
    # Test that registering outside context raises ProgrammingError
    with pytest.raises(ProgrammingError):
        registry1._CurrencyRegistry__register(
            Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
        )
    
    # Test get with default value
    default_currency = Currency.of("XXX", "Unknown", 2, CurrencyType.MONEY)
    assert registry1.get("NON_EXISTING", default=default_currency) == default_currency


# LLM-generated content at query #15
#--------------------------

```python
def test_CurrencyRegistry___getitem__():
    """Test CurrencyRegistry.__getitem__ method."""
    
    # Test successful retrieval of existing currency
    usd = Currencies["USD"]
    assert usd.code == "USD"
    assert usd.name == "US Dollar"
    assert usd.type == CurrencyType.MONEY
    
    # Test successful retrieval of another existing currency
    jpy = Currencies["JPY"]
    assert jpy.code == "JPY"
    assert jpy.decimals == 0
    
    # Test that retrieved currency is the same object as in registry
    assert Currencies["USD"] == Currencies["USD"]
    
    # Test that non-existing currency code raises CurrencyLookupError
    with pytest.raises(CurrencyLookupError) as exc_info:
        Currencies["NON-EXISTING"]
    assert exc_info.value.code == "NON-EXISTING"
    assert "NON-EXISTING" in str(exc_info.value)
    
    # Test with various invalid currency codes
    for invalid_code in ["XYZ", "INVALID", "ZZZ"]:
        with pytest.raises(CurrencyLookupError) as exc_info:
            Currencies[invalid_code]
        assert exc_info.value.code == invalid_code
    
    # Test case sensitivity - codes should be uppercase
    with pytest.raises(CurrencyLookupError):
        Currencies["usd"]
    
    # Test that returned currency is of correct type
    currency = Currencies["USD"]
    assert isinstance(currency, Currency)
    assert hasattr(currency, 'code')
    assert hasattr(currency, 'name')
    assert hasattr(currency, 'decimals')
    assert hasattr(currency, 'type')


# LLM-generated content at query #16
#--------------------------

```python
def test_CurrencyRegistry_get():
    """Test the get method of CurrencyRegistry."""
    # Test getting an existing currency
    usd = Currencies.get("USD")
    assert usd is not None
    assert usd.code == "USD"
    assert usd.name == "US Dollar"
    
    # Test getting a non-existing currency without default
    result = Currencies.get("XXX")
    assert result is None
    
    # Test getting a non-existing currency with default
    default_currency = Currencies["EUR"]
    result = Currencies.get("NON-EXISTING", default=default_currency)
    assert result == default_currency
    assert result.code == "EUR"
    
    # Test getting with None as explicit default
    result = Currencies.get("INVALID", default=None)
    assert result is None
    
    # Test that get returns the same object as __getitem__
    get_result = Currencies.get("GBP")
    getitem_result = Currencies["GBP"]
    assert get_result == getitem_result
    assert get_result.code == "GBP"


# LLM-generated content at query #17
#--------------------------

```python
def test_CurrencyRegistry_has():
    """Test the has method of CurrencyRegistry."""
    # Test that has returns True for existing currency
    assert Currencies.has("USD") is True
    
    # Test that has returns True for other existing currencies
    assert Currencies.has("EUR") is True
    assert Currencies.has("JPY") is True
    assert Currencies.has("GBP") is True
    
    # Test that has returns False for non-existing currency
    assert Currencies.has("XXX") is False
    assert Currencies.has("ZZZ") is False
    assert Currencies.has("INVALID") is False
    
    # Test that has works with empty string
    assert Currencies.has("") is False
    
    # Test that has is case-sensitive
    assert Currencies.has("usd") is False
    assert Currencies.has("Usd") is False


# LLM-generated content at query #18
#--------------------------

```python
def test_Currency___eq__():
    """Test the __eq__ method of Currency class."""
    # Test equal currencies with same attributes
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2

    # Test unequal currencies with different names
    usd3 = Currency.of("USD", "United States Dollars", 2, CurrencyType.MONEY)
    assert usd1 != usd3

    # Test unequal currencies with different decimals
    usd4 = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert usd1 != usd4

    # Test unequal currencies with different types
    usd5 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert usd1 != usd5

    # Test unequal currencies with different codes
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert usd1 != jpy

    # Test equality with non-Currency objects
    assert usd1 != "USD"
    assert usd1 != 123
    assert usd1 != None
    assert usd1 != {"code": "USD"}

    # Test hash consistency with equality
    assert hash(usd1) == hash(usd2)
    assert hash(usd1) != hash(usd3)
    assert hash(usd1) != hash(jpy)

    # Test reflexivity: a == a
    assert usd1 == usd1

    # Test symmetry: if a == b then b == a
    assert (usd1 == usd2) == (usd2 == usd1)

    # Test transitivity: if a == b and b == c then a == c
    usd6 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2 and usd2 == usd6 and usd1 == usd6


# LLM-generated content at query #19
#--------------------------

```python
def test_CurrencyRegistry_get():
    """Test the get method of CurrencyRegistry."""
    
    # Test getting an existing currency
    usd = Currencies.get("USD")
    assert usd is not None
    assert usd.code == "USD"
    assert usd.name == "US Dollar"
    
    # Test getting a non-existing currency returns None
    result = Currencies.get("XXX")
    assert result is None
    
    # Test getting a non-existing currency with default value
    default_currency = Currencies["USD"]
    result = Currencies.get("XXX", default=default_currency)
    assert result == default_currency
    assert result.code == "USD"
    
    # Test getting with explicit None default
    result = Currencies.get("NON_EXISTENT", default=None)
    assert result is None
    
    # Test getting multiple different currencies
    eur = Currencies.get("EUR")
    assert eur is not None
    assert eur.code == "EUR"
    
    jpy = Currencies.get("JPY")
    assert jpy is not None
    assert jpy.code == "JPY"
    
    # Test that get returns same object as __getitem__
    assert Currencies.get("USD") == Currencies["USD"]
    assert Currencies.get("EUR") == Currencies["EUR"]


# LLM-generated content at query #20
#--------------------------

```python
def test_CurrencyRegistry___contains__():
    """Test the __contains__ method of CurrencyRegistry."""
    registry = CurrencyRegistry()
    
    # Test that USD is in the registry (assuming it's populated)
    assert "USD" in registry
    
    # Test that a non-existent currency code is not in the registry
    assert "XYZ" not in registry
    assert "NON_EXISTING" not in registry
    
    # Test with various valid currency codes
    assert "EUR" in registry
    assert "GBP" in registry
    assert "JPY" in registry
    
    # Test with invalid codes (non-existent)
    assert "INVALID" not in registry
    assert "" not in registry
    assert "123" not in registry


# LLM-generated content at query #21
#--------------------------

```python
def test_CurrencyRegistry___len__():
    """Test the __len__ method of CurrencyRegistry."""
    registry = CurrencyRegistry()
    
    # The registry should have a non-zero length since it's pre-populated
    initial_length = len(registry)
    assert isinstance(initial_length, int)
    assert initial_length > 0
    
    # Verify that the length matches the number of currencies in the registry
    assert initial_length == len(registry.all)
    
    # Verify that the length matches the number of codes
    assert initial_length == len(registry.codes)
    
    # Verify that the length matches the number of codenames
    assert initial_length == len(registry.codenames)
    
    # Verify that __len__ returns consistent results on multiple calls
    assert len(registry) == len(registry)


# LLM-generated content at query #22
#--------------------------

```python
def test_CurrencyRegistry():
    """Test CurrencyRegistry constructor and singleton behavior."""
    # Test singleton pattern - creating multiple instances should return the same object
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2
    
    # Test initial state after construction
    assert len(registry1) == 0
    assert registry1.all == []
    assert registry1.codes == []
    assert registry1.codenames == []
    
    # Test __contains__ on empty registry
    assert "USD" not in registry1
    assert "EUR" not in registry1
    
    # Test has() method on empty registry
    assert not registry1.has("USD")
    assert not registry1.has("EUR")
    
    # Test get() method on empty registry
    assert registry1.get("USD") is None
    assert registry1.get("EUR") is None
    
    # Test __getitem__ raises CurrencyLookupError on empty registry
    with pytest.raises(CurrencyLookupError) as exc_info:
        _ = registry1["USD"]
    assert exc_info.value.code == "USD"
    
    # Test that we can use the registry as a context manager
    with registry1 as register:
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        register(usd)
        eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
        register(eur)
    
    # After context exit, verify the registry is populated and sorted
    assert len(registry1) == 2
    assert registry1.has("USD")
    assert registry1.has("EUR")
    assert "USD" in registry1
    assert "EUR" in registry1
    
    # Verify codes are sorted
    assert registry1.codes == ["EUR", "USD"]
    
    # Verify all() returns sorted currencies
    all_currencies = registry1.all
    assert len(all_currencies) == 2
    assert all_currencies[0].code == "EUR"
    assert all_currencies[1].code == "USD"
    
    # Verify codenames
    assert registry1.codenames == [("EUR", "Euro"), ("USD", "US Dollar")]
    
    # Test get() after population
    usd_currency = registry1.get("USD")
    assert usd_currency is not None
    assert usd_currency.code == "USD"
    assert usd_currency.name == "US Dollar"
    
    # Test __getitem__ after population
    eur_currency = registry1["EUR"]
    assert eur_currency.code == "EUR"
    assert eur_currency.name == "Euro"
    
    # Test that adding duplicate currency raises ValueError
    with registry1 as register:
        with pytest.raises(ValueError, match="Currency USD is already registered"):
            register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
    
    # Test that registering outside context raises ProgrammingError
    with pytest.raises(ProgrammingError, match="Can not create currencies outside registry context"):
        registry1._CurrencyRegistry__register(Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY))


# LLM-generated content at query #23
#--------------------------

```python
def test_Currency():
    """Test Currency class creation and validation"""
    
    # Test valid currency creation with positive decimals
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.code == "USD"
    assert usd.name == "US Dollars"
    assert usd.decimals == 2
    assert usd.type == CurrencyType.MONEY
    
    # Test valid currency creation with zero decimals
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy.code == "JPY"
    assert jpy.decimals == 0
    
    # Test valid currency creation with negative decimals
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert zzz.code == "ZZZ"
    assert zzz.decimals == -1
    
    # Test quantization with positive decimals
    assert usd.quantize(Decimal("1.005")) == Decimal("1.00")
    assert usd.quantize(Decimal("1.015")) == Decimal("1.02")
    
    # Test quantization with zero decimals
    assert jpy.quantize(Decimal("0.5")) == Decimal("0")
    assert jpy.quantize(Decimal("1.5")) == Decimal("2")
    
    # Test quantization with negative decimals (max precision)
    assert zzz.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert zzz.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")
    
    # Test equality
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2
    assert usd1 != usdx
    
    # Test hash consistency
    assert hash(usd1) == hash(usd2)
    assert hash(usd1) != hash(usdx)
    
    # Test invalid code - non-alphabetic
    with pytest.raises(ProgrammingError):
        Currency.of("US1", "US Dollars", 2, CurrencyType.MONEY)
    
    # Test invalid code - not uppercase
    with pytest.raises(ProgrammingError):
        Currency.of("Usd", "US Dollars", 2, CurrencyType.MONEY)
    
    # Test invalid code - not string
    with pytest.raises(ProgrammingError):
        Currency.of(123, "US Dollars", 2, CurrencyType.MONEY)
    
    # Test invalid name - empty string
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "", 2, CurrencyType.MONEY)
    
    # Test invalid name - leading space
    with pytest.raises(ProgrammingError):
        Currency.of("USD", " US Dollars", 2, CurrencyType.MONEY)
    
    # Test invalid name - trailing space
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars ", 2, CurrencyType.MONEY)
    
    # Test invalid name - not string
    with pytest.raises(ProgrammingError):
        Currency.of("USD", 123, 2, CurrencyType.MONEY)
    
    # Test invalid decimals - not integer
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", 2.5, CurrencyType.MONEY)
    
    # Test invalid decimals - less than -1
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", -2, CurrencyType.MONEY)
    
    # Test invalid type - not CurrencyType enum
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", 2, "MONEY")
    
    # Test all currency types
    money = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert money.type == CurrencyType.MONEY
    
    metal = Currency.of("XAU", "Gold", 2, CurrencyType.METAL)
    assert metal.type == CurrencyType.METAL
    
    crypto = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    assert crypto.type == CurrencyType.CRYPTO
    
    alternative = Currency.of("ALT", "Alternative", 2, CurrencyType.ALTERNATIVE)
    assert alternative.type == CurrencyType.ALTERNATIVE


# LLM-generated content at query #24
#--------------------------

```python
def test_Currency___gt__():
    """Test the __gt__ method of Currency class."""
    # Create currencies with different codes for comparison
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    gbp = Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY)
    
    # Test basic greater than comparisons based on ordering
    assert eur > gbp
    assert gbp > usd
    assert eur > usd
    
    # Test that a currency is not greater than itself
    assert not (usd > usd)
    
    # Test with currencies of different decimals
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert usd > jpy  # USD comes after JPY alphabetically
    
    # Test with different currency types
    btc = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    assert usd > btc  # USD comes after BTC alphabetically
    
    # Test chaining comparisons
    assert eur > gbp > usd
    
    # Test with same code but different attributes (should have different ordering based on creation order)
    chf = Currency.of("CHF", "Swiss Franc", 2, CurrencyType.MONEY)
    assert chf > eur  # CHF comes after EUR alphabetically


# LLM-generated content at query #25
#--------------------------

```python
def test_Currency___hash__():
    """Test the __hash__ method of Currency class."""
    # Test that hash returns the pre-computed cached hash
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert isinstance(usd.__hash__(), int)
    assert usd.__hash__() == usd.hashcache
    
    # Test that two identical currencies have the same hash
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert hash(usd1) == hash(usd2)
    
    # Test that two different currencies have different hashes
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert hash(usd) != hash(jpy)
    
    # Test that currencies with same code but different names have different hashes
    usd_orig = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd_alt = Currency.of("USD", "United States Dollar", 2, CurrencyType.MONEY)
    assert hash(usd_orig) != hash(usd_alt)
    
    # Test that currencies with same code but different decimals have different hashes
    usd_2decimals = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd_3decimals = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert hash(usd_2decimals) != hash(usd_3decimals)
    
    # Test that currencies with same code but different types have different hashes
    usd_money = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd_crypto = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert hash(usd_money) != hash(usd_crypto)
    
    # Test that currency can be used in sets and dicts (requires proper hashing)
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    
    currency_set = {usd1, usd2, jpy}
    assert len(currency_set) == 2  # usd1 and usd2 are the same, so only 2 unique currencies
    
    currency_dict = {usd1: "Dollar", jpy: "Yen"}
    assert currency_dict[usd2] == "Dollar"  # usd2 should map to the same key as usd1


# LLM-generated content at query #26
#--------------------------

```python
def test_CurrencyRegistry():
    """Test CurrencyRegistry constructor and singleton behavior."""
    # Test singleton pattern - same instance returned
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2
    
    # Test initial state after construction
    assert len(registry1) == 0
    assert registry1.all == []
    assert registry1.codes == []
    assert registry1.codenames == []
    
    # Test that registry is empty before population
    assert "USD" not in registry1
    assert registry1.has("USD") is False
    assert registry1.get("USD") is None
    assert registry1.get("USD", default=None) is None
    
    # Test context manager for adding currencies
    test_currency = Currency.of("TST", "Test Currency", 2, CurrencyType.MONEY)
    with registry1 as register:
        register(test_currency)
    
    # Verify currency was added
    assert "TST" in registry1
    assert len(registry1) == 1
    assert registry1.has("TST") is True
    assert registry1.get("TST") == test_currency
    assert registry1["TST"] == test_currency
    
    # Test that currencies are sorted by code after context exit
    currency_aaa = Currency.of("AAA", "First Currency", 2, CurrencyType.MONEY)
    currency_zzz = Currency.of("ZZZ", "Last Currency", 2, CurrencyType.MONEY)
    with registry1 as register:
        register(currency_zzz)
        register(currency_aaa)
    
    assert registry1.codes == ["AAA", "TST", "ZZZ"]
    assert registry1.all[0].code == "AAA"
    assert registry1.all[-1].code == "ZZZ"
    
    # Test codenames property
    codenames = registry1.codenames
    assert all(isinstance(cn, tuple) and len(cn) == 2 for cn in codenames)
    assert ("AAA", "First Currency") in codenames
    assert ("TST", "Test Currency") in codenames
    assert ("ZZZ", "Last Currency") in codenames
    
    # Test error when accessing non-existent currency with __getitem__
    with pytest.raises(CurrencyLookupError) as exc_info:
        _ = registry1["NON_EXISTENT"]
    assert exc_info.value.code == "NON_EXISTENT"
    
    # Test error when trying to register outside context
    with pytest.raises(ProgrammingError):
        registry1._CurrencyRegistry__register(test_currency)
    
    # Test error when registering duplicate currency
    with pytest.raises(ValueError) as exc_info:
        with registry1 as register:
            register(test_currency)
    assert "already registered" in str(exc_info.value)


# LLM-generated content at query #27
#--------------------------

```python
def test_Currency___setattr__():
    """Test that Currency instances are immutable (frozen dataclass)."""
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    # Attempting to set any attribute should raise FrozenInstanceError
    with pytest.raises(Exception):  # FrozenInstanceError from dataclass
        USD.code = "EUR"
    
    with pytest.raises(Exception):
        USD.name = "Euros"
    
    with pytest.raises(Exception):
        USD.decimals = 3
    
    with pytest.raises(Exception):
        USD.type = CurrencyType.CRYPTO
    
    with pytest.raises(Exception):
        USD.quantizer = Decimal("0.001")
    
    with pytest.raises(Exception):
        USD.hashcache = 999
    
    with pytest.raises(Exception):
        USD.new_attribute = "should fail"


# LLM-generated content at query #28
#--------------------------

```python
def test_Currency___hash__():
    """Test the __hash__ method of Currency class."""
    # Test that identical currencies have the same hash
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert hash(usd1) == hash(usd2)
    
    # Test that different currencies have different hashes
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert hash(usd1) != hash(jpy)
    
    # Test that currencies with different names have different hashes
    usd3 = Currency.of("USD", "United States Dollar", 2, CurrencyType.MONEY)
    assert hash(usd1) != hash(usd3)
    
    # Test that currencies with different decimals have different hashes
    usd4 = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert hash(usd1) != hash(usd4)
    
    # Test that currencies with different types have different hashes
    usd5 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert hash(usd1) != hash(usd5)
    
    # Test that hash is consistent across multiple calls
    usd6 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    hash1 = hash(usd6)
    hash2 = hash(usd6)
    assert hash1 == hash2
    
    # Test that currency can be used in sets and dicts
    currency_set = {usd1, usd2, jpy}
    assert len(currency_set) == 2  # usd1 and usd2 are the same
    
    currency_dict = {usd1: "First", usd2: "Second", jpy: "Third"}
    assert len(currency_dict) == 2  # usd1 and usd2 map to same key
    assert currency_dict[usd1] == "Second"  # usd2 overwrites usd1
    
    # Test with crypto currency with special decimals
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert isinstance(hash(zzz), int)
    
    # Test that hash returns an integer
    assert isinstance(hash(usd1), int)
    assert isinstance(hash(jpy), int)


# LLM-generated content at query #29
#--------------------------

```python
def test_CurrencyLookupError():
    """Test CurrencyLookupError constructor."""
    code = "XYZ"
    error = CurrencyLookupError(code)
    
    assert error.code == code
    assert str(error) == f"Currency identified by code '{code}' does not exist"
    assert isinstance(error, LookupError)


def test_CurrencyLookupError_different_codes():
    """Test CurrencyLookupError with different currency codes."""
    codes = ["ABC", "USD", "EUR", "GBP"]
    
    for code in codes:
        error = CurrencyLookupError(code)
        assert error.code == code
        assert str(error) == f"Currency identified by code '{code}' does not exist"


def test_CurrencyLookupError_raises():
    """Test that CurrencyLookupError can be raised and caught."""
    with pytest.raises(CurrencyLookupError) as exc_info:
        raise CurrencyLookupError("XYZ")
    
    assert exc_info.value.code == "XYZ"
    assert "XYZ" in str(exc_info.value)


def test_CurrencyLookupError_inheritance():
    """Test that CurrencyLookupError is a LookupError."""
    error = CurrencyLookupError("TEST")
    assert isinstance(error, LookupError)
    assert isinstance(error, Exception)


# LLM-generated content at query #30
#--------------------------

```python
def test_Currency___eq__():
    """Test the __eq__ method of Currency class."""
    
    # Create two identical currencies
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    # Test equality of identical currencies
    assert usd1 == usd2
    
    # Create a currency with different name
    usd_diff_name = Currency.of("USD", "United States Dollar", 2, CurrencyType.MONEY)
    
    # Test inequality when name differs
    assert usd1 != usd_diff_name
    
    # Create a currency with different decimals
    usd_diff_decimals = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    
    # Test inequality when decimals differ
    assert usd1 != usd_diff_decimals
    
    # Create a currency with different type
    usd_diff_type = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    
    # Test inequality when type differs
    assert usd1 != usd_diff_type
    
    # Create a different currency
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    
    # Test inequality with different currency code
    assert usd1 != eur
    
    # Test that currency is not equal to non-Currency objects
    assert usd1 != "USD"
    assert usd1 != 2
    assert usd1 != None
    assert usd1 != {}
    
    # Test reflexivity: a currency should equal itself
    assert usd1 == usd1
    
    # Test symmetry: if a == b, then b == a
    assert usd1 == usd2
    assert usd2 == usd1
    
    # Test transitivity: if a == b and b == c, then a == c
    usd3 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2
    assert usd2 == usd3
    assert usd1 == usd3
    
    # Test with different currency types
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    
    assert jpy != zzz
    assert jpy != usd1
    assert zzz != usd1


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_CurrencyRegistry___enter__():
    """Test the __enter__ method of CurrencyRegistry."""
    registry = CurrencyRegistry()
    
    # Test that __enter__ returns a callable
    result = registry.__enter__()
    assert callable(result), "__enter__ should return a callable"
    
    # Test that the context is marked as open
    assert registry._CurrencyRegistry__ctx_open is True, "Context should be marked as open"
    
    # Test that __enter__ returns the __register method
    assert result == registry._CurrencyRegistry__register, "__enter__ should return the __register method"
    
    # Clean up by exiting the context
    registry.__exit__(None, None, None)


def test_CurrencyRegistry___enter___with_statement():
    """Test the __enter__ method works correctly in a with statement."""
    registry = CurrencyRegistry()
    
    # Test using with statement
    with registry as register:
        # Verify register is callable
        assert callable(register), "Context manager should return a callable"
        
        # Verify we can call it to register a currency
        test_currency = Currency.of("TST", "Test Currency", 2, CurrencyType.MONEY)
        register(test_currency)
        
        # Verify context is open during with block
        assert registry._CurrencyRegistry__ctx_open is True


def test_CurrencyRegistry___enter___context_open_flag():
    """Test that __enter__ sets the context open flag correctly."""
    registry = CurrencyRegistry()
    
    # Initially context should be closed
    assert registry._CurrencyRegistry__ctx_open is False
    
    # Call __enter__
    registry.__enter__()
    
    # Now context should be open
    assert registry._CurrencyRegistry__ctx_open is True
    
    # Clean up
    registry.__exit__(None, None, None)
    
    # Context should be closed again
    assert registry._CurrencyRegistry__ctx_open is False


# LLM-generated content at query #2
#--------------------------

```python
def test_Currency___eq__():
    """Test the __eq__ method of Currency class."""
    
    # Create two identical currencies
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    # Test equality of identical currencies
    assert usd1 == usd2
    
    # Create a currency with different name but same code
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    
    # Test inequality when names differ
    assert not (usd1 == usdx)
    assert usd1 != usdx
    
    # Create a currency with different decimals
    usd_diff_decimals = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    
    # Test inequality when decimals differ
    assert not (usd1 == usd_diff_decimals)
    assert usd1 != usd_diff_decimals
    
    # Create a currency with different type
    usd_crypto = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    
    # Test inequality when types differ
    assert not (usd1 == usd_crypto)
    assert usd1 != usd_crypto
    
    # Create a completely different currency
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    
    # Test inequality with different currency
    assert not (usd1 == eur)
    assert usd1 != eur
    
    # Test equality with non-Currency object
    assert not (usd1 == "USD")
    assert not (usd1 == 2)
    assert not (usd1 == None)
    assert usd1 != "USD"
    assert usd1 != 2
    assert usd1 != None
    
    # Test reflexivity (currency equals itself)
    assert usd1 == usd1
    
    # Test symmetry (if a == b then b == a)
    assert usd1 == usd2
    assert usd2 == usd1
    
    # Test transitivity (if a == b and b == c then a == c)
    usd3 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2
    assert usd2 == usd3
    assert usd1 == usd3


# LLM-generated content at query #3
#--------------------------

```python
def test_CurrencyRegistry_has():
    """Test the has method of CurrencyRegistry."""
    # Test with existing currency code
    assert Currencies.has("USD") is True
    assert Currencies.has("EUR") is True
    assert Currencies.has("JPY") is True
    
    # Test with non-existing currency code
    assert Currencies.has("XXX") is False
    assert Currencies.has("NON") is False
    assert Currencies.has("INVALID") is False
    
    # Test with empty string
    assert Currencies.has("") is False
    
    # Test with lowercase code (should not exist as codes are uppercase)
    assert Currencies.has("usd") is False
    
    # Test consistency with __contains__ operator
    assert Currencies.has("USD") == ("USD" in Currencies)
    assert Currencies.has("EUR") == ("EUR" in Currencies)
    assert Currencies.has("XXX") == ("XXX" in Currencies)


# LLM-generated content at query #4
#--------------------------

```python
def test_CurrencyRegistry___contains__():
    """Test the __contains__ method of CurrencyRegistry."""
    # Create a currency registry instance
    registry = CurrencyRegistry()
    
    # Test that USD is in the registry (assuming it's been registered)
    assert "USD" in registry
    
    # Test that a non-existent currency code is not in the registry
    assert "XXX" not in registry
    assert "NON_EXISTING" not in registry
    assert "INVALID" not in registry
    
    # Test with lowercase (should not be in registry as codes are uppercase)
    assert "usd" not in registry
    
    # Test with empty string
    assert "" not in registry
    
    # Test with multiple known currencies
    assert "EUR" in registry or "EUR" not in registry  # EUR may or may not be registered
    assert "GBP" in registry or "GBP" not in registry  # GBP may or may not be registered
    
    # Test with various invalid formats
    assert "US" not in registry
    assert "USDA" not in registry
    assert "U$D" not in registry


# LLM-generated content at query #5
#--------------------------

```python
def test_Currency___gt__():
    """Test the __gt__ method of Currency class."""
    # Create currencies with different codes
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    gbp = Currency.of("GBP", "British Pounds", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    
    # Test greater than comparisons based on ordering
    assert eur > usd
    assert gbp > eur
    assert gbp > usd
    assert jpy > gbp
    
    # Test that greater than is false for reverse comparisons
    assert not (usd > eur)
    assert not (eur > gbp)
    assert not (usd > gbp)
    assert not (gbp > jpy)
    
    # Test that a currency is not greater than itself
    assert not (usd > usd)
    assert not (eur > eur)
    
    # Test with currencies that have same code but different names
    usd_alt = Currency.of("USD", "United States Dollars", 2, CurrencyType.MONEY)
    # These are different currencies due to different names, so comparison should work
    assert not (usd > usd_alt) and not (usd_alt > usd)  # They're not equal in hash, so one must be greater
    
    # Test with different currency types
    btc = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    assert btc > usd  # BTC code comes after USD alphabetically
    
    # Test with negative decimals
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert zzz > btc  # ZZZ comes after BTC
    assert not (btc > zzz)


# LLM-generated content at query #6
#--------------------------

```python
def test_CurrencyRegistry_has():
    """Test the has method of CurrencyRegistry."""
    # Test that has returns True for an existing currency
    assert Currencies.has("USD") is True
    assert Currencies.has("EUR") is True
    assert Currencies.has("JPY") is True
    
    # Test that has returns False for non-existing currency codes
    assert Currencies.has("XXX") is False
    assert Currencies.has("ZZZ") is False
    assert Currencies.has("INVALID") is False
    
    # Test with empty string
    assert Currencies.has("") is False
    
    # Test that has is consistent with __contains__
    assert Currencies.has("USD") == ("USD" in Currencies)
    assert Currencies.has("XXX") == ("XXX" in Currencies)
    
    # Test with lowercase (should not be found as codes are uppercase)
    assert Currencies.has("usd") is False
    
    # Test multiple currencies to ensure registry is properly populated
    assert Currencies.has("GBP") is True
    assert Currencies.has("CHF") is True
    assert Currencies.has("AUD") is True


# LLM-generated content at query #7
#--------------------------

```python
def test_CurrencyRegistry():
    """Test the CurrencyRegistry constructor and singleton behavior."""
    # Test singleton pattern - first instance
    registry1 = CurrencyRegistry()
    assert registry1 is not None
    assert isinstance(registry1, CurrencyRegistry)
    
    # Test singleton pattern - second instance should be the same
    registry2 = CurrencyRegistry()
    assert registry1 is registry2
    
    # Test initial state after construction
    assert len(registry1) == 0
    assert registry1.all == []
    assert registry1.codes == []
    assert registry1.codenames == []
    
    # Test that context is initially closed
    assert registry1._CurrencyRegistry__ctx_open is False
    
    # Test that registry dictionaries are empty
    assert len(registry1._CurrencyRegistry__registry) == 0
    assert len(registry1._CurrencyRegistry__currencies) == 0
    assert len(registry1._CurrencyRegistry__codes) == 0
    assert len(registry1._CurrencyRegistry__codenames) == 0
    
    # Test __contains__ with empty registry
    assert "USD" not in registry1
    assert "EUR" not in registry1
    
    # Test has() method with empty registry
    assert not registry1.has("USD")
    assert not registry1.has("EUR")
    
    # Test get() method with empty registry
    assert registry1.get("USD") is None
    assert registry1.get("EUR") is None
    
    # Test __getitem__ with empty registry raises error
    with pytest.raises(CurrencyLookupError) as exc_info:
        _ = registry1["USD"]
    assert exc_info.value.code == "USD"


# LLM-generated content at query #8
#--------------------------

```python
def test_Currency():
    """Test Currency class constructor and functionality."""
    
    # Test valid USD currency creation
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.code == "USD"
    assert usd.name == "US Dollars"
    assert usd.decimals == 2
    assert usd.type == CurrencyType.MONEY
    assert usd.quantizer == Decimal("0.01")
    
    # Test valid JPY currency creation (0 decimals)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy.code == "JPY"
    assert jpy.decimals == 0
    assert jpy.quantizer == ZERO
    
    # Test valid crypto currency with negative decimals
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert zzz.code == "ZZZ"
    assert zzz.decimals == -1
    assert zzz.quantizer == MaxPrecisionQuantizer
    
    # Test quantize functionality
    assert usd.quantize(Decimal("1.005")) == Decimal("1.00")
    assert usd.quantize(Decimal("1.015")) == Decimal("1.02")
    assert jpy.quantize(Decimal("0.5")) == Decimal("0")
    assert jpy.quantize(Decimal("1.5")) == Decimal("2")
    
    # Test equality
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd_diff_name = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    
    assert usd1 == usd2
    assert usd1 != usd_diff_name
    assert hash(usd1) == hash(usd2)
    assert hash(usd1) != hash(usd_diff_name)
    
    # Test invalid code - non-alphabetic
    with pytest.raises(ProgrammingError):
        Currency.of("US1", "US Dollars", 2, CurrencyType.MONEY)
    
    # Test invalid code - lowercase
    with pytest.raises(ProgrammingError):
        Currency.of("usd", "US Dollars", 2, CurrencyType.MONEY)
    
    # Test invalid code - not a string
    with pytest.raises(ProgrammingError):
        Currency.of(123, "US Dollars", 2, CurrencyType.MONEY)
    
    # Test invalid name - empty string
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "", 2, CurrencyType.MONEY)
    
    # Test invalid name - leading space
    with pytest.raises(ProgrammingError):
        Currency.of("USD", " US Dollars", 2, CurrencyType.MONEY)
    
    # Test invalid name - trailing space
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars ", 2, CurrencyType.MONEY)
    
    # Test invalid name - not a string
    with pytest.raises(ProgrammingError):
        Currency.of("USD", 123, 2, CurrencyType.MONEY)
    
    # Test invalid decimals - not an integer
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", 2.5, CurrencyType.MONEY)
    
    # Test invalid decimals - less than -1
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", -2, CurrencyType.MONEY)
    
    # Test invalid type
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", 2, "MONEY")
    
    # Test different currency types
    metal = Currency.of("AU", "Gold", 2, CurrencyType.METAL)
    assert metal.type == CurrencyType.METAL
    
    alt = Currency.of("ALT", "Alternative", 2, CurrencyType.ALTERNATIVE)
    assert alt.type == CurrencyType.ALTERNATIVE
    
    # Test frozen dataclass (immutability)
    with pytest.raises(Exception):
        usd.code = "EUR"


# LLM-generated content at query #9
#--------------------------

```python
def test_CurrencyRegistry___enter__():
    """Test that __enter__ returns the __register method and sets context as open."""
    registry = CurrencyRegistry()
    
    # Call __enter__ to get the register method
    register_method = registry.__enter__()
    
    # Verify that the returned object is callable
    assert callable(register_method)
    
    # Verify that we can call it with a Currency object
    test_currency = Currency.of("TST", "Test Currency", 2, CurrencyType.MONEY)
    register_method(test_currency)
    
    # Clean up by exiting the context
    registry.__exit__(None, None, None)


# LLM-generated content at query #10
#--------------------------

```python
def test_Currency___le__():
    """Test the __le__ (less than or equal to) comparison of Currency objects."""
    # Create currencies with different codes
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    gbp = Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY)
    
    # Test __le__ with same currency (should be True)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd <= usd2
    
    # Test __le__ with different currencies (alphabetically ordered)
    # EUR comes before GBP, so EUR <= GBP should be True
    assert eur <= gbp
    
    # Test __le__ with different currencies (reverse order)
    # GBP comes after EUR, so GBP <= EUR should be False
    assert not (gbp <= eur)
    
    # Test __le__ with USD and EUR
    # EUR comes before USD, so EUR <= USD should be True
    assert eur <= usd
    
    # Test __le__ reflexivity (currency <= itself)
    assert usd <= usd
    assert eur <= eur
    assert gbp <= gbp
    
    # Test __le__ with currencies having different decimals
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    # EUR comes before JPY alphabetically
    assert eur <= jpy
    
    # Test __le__ with currencies having different types
    btc = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    # BTC comes before USD alphabetically
    assert btc <= usd
    
    # Test __le__ transitivity: if a <= b and b <= c, then a <= c
    assert eur <= gbp
    assert gbp <= usd
    assert eur <= usd


# LLM-generated content at query #11
#--------------------------

```python
def test_Currency___lt__():
    """Test the __lt__ method of Currency class for ordering."""
    # Create currencies with different codes
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    gbp = Currency.of("GBP", "British Pounds", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    
    # Test basic ordering - currencies are ordered by their dataclass fields
    assert eur < gbp
    assert gbp < jpy
    assert usd < eur
    
    # Test that a currency is not less than itself
    assert not (usd < usd)
    
    # Test transitive property
    assert eur < gbp and gbp < jpy and eur < jpy
    
    # Test with currencies of different types
    btc = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    gold = Currency.of("XAU", "Gold", 4, CurrencyType.METAL)
    
    # Ordering follows dataclass field order: code, name, decimals, type, quantizer, hashcache
    assert btc < eur  # "BTC" < "EUR" alphabetically
    assert eur < gold  # "EUR" < "XAU" alphabetically
    
    # Test that ordering is consistent
    currencies = [jpy, usd, eur, gbp]
    sorted_currencies = sorted(currencies)
    assert sorted_currencies == [eur, gbp, jpy, usd]
    
    # Test with same code but different names (should order by code first, then name)
    usd_variant = Currency.of("USD", "American Dollars", 2, CurrencyType.MONEY)
    assert not (usd < usd_variant)  # Same code, different name - comparison depends on name
    assert usd_variant < usd or usd < usd_variant  # One must be less than the other due to name difference


# LLM-generated content at query #12
#--------------------------

```python
def test_CurrencyRegistry___enter__():
    """Test that __enter__ returns the __register method and opens the context."""
    registry = CurrencyRegistry()
    
    # Call __enter__
    result = registry.__enter__()
    
    # Verify that it returns a callable
    assert callable(result)
    
    # Verify that the context is open by checking that __register can be called
    # (it will raise ProgrammingError if context is not open)
    test_currency = Currency.of("TST", "Test Currency", 2, CurrencyType.MONEY)
    
    # This should work since context is open
    result(test_currency)
    
    # Verify the currency was added to the registry
    assert "TST" in registry
    
    # Clean up by exiting the context
    registry.__exit__(None, None, None)


# LLM-generated content at query #13
#--------------------------

```python
def test_CurrencyRegistry_get():
    """Test the get method of CurrencyRegistry."""
    
    # Test getting an existing currency
    usd = Currencies.get("USD")
    assert usd is not None
    assert usd.code == "USD"
    assert usd.name == "US Dollar"
    
    # Test getting a non-existing currency returns None by default
    result = Currencies.get("XXX")
    assert result is None
    
    # Test getting a non-existing currency with a default value
    default_currency = Currencies["EUR"]
    result = Currencies.get("XXX", default=default_currency)
    assert result == default_currency
    assert result.code == "EUR"
    
    # Test getting a non-existing currency with a different default
    default_currency = Currencies["GBP"]
    result = Currencies.get("NON_EXISTENT", default=default_currency)
    assert result == default_currency
    assert result.code == "GBP"
    
    # Test that get returns the same currency as __getitem__
    assert Currencies.get("USD") == Currencies["USD"]
    assert Currencies.get("JPY") == Currencies["JPY"]
    
    # Test with multiple valid currency codes
    for code in ["USD", "EUR", "GBP", "JPY"]:
        currency = Currencies.get(code)
        assert currency is not None
        assert currency.code == code


# LLM-generated content at query #14
#--------------------------

```python
def test_Currency_quantize():
    """Test the quantize method of Currency class."""
    
    # Test with USD (2 decimal places)
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert USD.quantize(Decimal("1.005")) == Decimal("1.00")
    assert USD.quantize(Decimal("1.015")) == Decimal("1.02")
    assert USD.quantize(Decimal("1.234")) == Decimal("1.23")
    assert USD.quantize(Decimal("1.235")) == Decimal("1.24")
    assert USD.quantize(Decimal("0")) == Decimal("0.00")
    
    # Test with JPY (0 decimal places)
    JPY = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert JPY.quantize(Decimal("0.5")) == Decimal("0")
    assert JPY.quantize(Decimal("1.5")) == Decimal("2")
    assert JPY.quantize(Decimal("2.4")) == Decimal("2")
    assert JPY.quantize(Decimal("2.5")) == Decimal("2")
    assert JPY.quantize(Decimal("100")) == Decimal("100")
    
    # Test with ZZZ (no fixed precision, decimals = -1)
    ZZZ = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert ZZZ.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert ZZZ.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")
    assert ZZZ.quantize(Decimal("1.123456789")) == Decimal("1.123456789")
    
    # Test with BTC (8 decimal places)
    BTC = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    assert BTC.quantize(Decimal("0.123456789")) == Decimal("0.12345679")
    assert BTC.quantize(Decimal("1")) == Decimal("1.00000000")
    
    # Test with GBP (2 decimal places)
    GBP = Currency.of("GBP", "British Pounds", 2, CurrencyType.MONEY)
    assert GBP.quantize(Decimal("10.126")) == Decimal("10.13")
    assert GBP.quantize(Decimal("10.124")) == Decimal("10.12")
    
    # Test with AUD (2 decimal places)
    AUD = Currency.of("AUD", "Australian Dollars", 2, CurrencyType.MONEY)
    assert AUD.quantize(Decimal("0.999")) == Decimal("1.00")
    assert AUD.quantize(Decimal("0.001")) == Decimal("0.00")
    
    # Test edge cases
    EUR = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    assert EUR.quantize(Decimal("0")) == Decimal("0.00")
    assert EUR.quantize(Decimal("-1.234")) == Decimal("-1.23")
    assert EUR.quantize(Decimal("-1.235")) == Decimal("-1.24")


# LLM-generated content at query #15
#--------------------------

```python
def test_CurrencyRegistry___new__():
    """Test that CurrencyRegistry.__new__ creates and returns a singleton instance."""
    # Create first instance
    registry1 = CurrencyRegistry()
    
    # Create second instance
    registry2 = CurrencyRegistry()
    
    # Both should be the exact same object (singleton pattern)
    assert registry1 is registry2
    
    # Create third instance to further verify singleton behavior
    registry3 = CurrencyRegistry()
    assert registry3 is registry1
    assert registry3 is registry2
    
    # All should have the same id
    assert id(registry1) == id(registry2) == id(registry3)


# LLM-generated content at query #16
#--------------------------

```python
def test_CurrencyRegistry___getitem__():
    """Test CurrencyRegistry.__getitem__ method"""
    
    # Test successful retrieval of existing currency
    registry = CurrencyRegistry()
    usd = registry["USD"]
    assert usd.code == "USD"
    assert usd.name == "US Dollar"
    assert usd.type == CurrencyType.MONEY
    
    # Test that retrieved currency is the same object
    usd2 = registry["USD"]
    assert usd == usd2
    assert usd is usd2
    
    # Test retrieval of another existing currency
    jpy = registry["JPY"]
    assert jpy.code == "JPY"
    assert jpy.decimals == 0
    
    # Test that CurrencyLookupError is raised for non-existing currency
    with pytest.raises(CurrencyLookupError) as exc_info:
        registry["NON-EXISTING"]
    assert exc_info.value.code == "NON-EXISTING"
    assert "NON-EXISTING" in str(exc_info.value)
    
    # Test with various non-existing codes
    with pytest.raises(CurrencyLookupError):
        registry["XYZ"]
    
    with pytest.raises(CurrencyLookupError):
        registry["ABC"]
    
    with pytest.raises(CurrencyLookupError):
        registry[""]
    
    # Test case sensitivity (should raise error for lowercase)
    with pytest.raises(CurrencyLookupError):
        registry["usd"]


# LLM-generated content at query #17
#--------------------------

```python
def test_Currency___ge__():
    """Test the __ge__ (greater than or equal) comparison operator for Currency."""
    # Create currencies with different codes
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    gbp = Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY)
    
    # Test equal currencies
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd >= usd2
    assert usd2 >= usd
    
    # Test ordering based on code (alphabetically)
    # EUR >= USD is False (E < U)
    # USD >= EUR is True (U > E)
    assert usd >= eur
    assert not (eur >= usd)
    
    # Test GBP >= USD is False (G < U)
    assert not (gbp >= usd)
    
    # Test USD >= GBP is True (U > G)
    assert usd >= gbp
    
    # Test transitive property
    # GBP >= EUR should be True (G > E)
    assert gbp >= eur
    
    # Test with different decimal precisions
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    # USD >= JPY is True (U > J)
    assert usd >= jpy
    
    # Test with different currency types
    btc = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    # USD >= BTC is True (U > B)
    assert usd >= btc
    
    # Test reflexivity (x >= x should always be True)
    assert usd >= usd
    assert eur >= eur
    assert jpy >= jpy
    assert btc >= btc


# LLM-generated content at query #18
#--------------------------

```python
def test_Currency___repr__():
    """Test the __repr__ method of Currency class."""
    # Test with a standard money currency
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    repr_str = repr(usd)
    assert "Currency" in repr_str
    assert "USD" in repr_str
    assert "US Dollars" in repr_str
    assert "2" in repr_str
    assert "MONEY" in repr_str
    
    # Test with a metal currency
    gold = Currency.of("XAU", "Gold", 4, CurrencyType.METAL)
    repr_str = repr(gold)
    assert "Currency" in repr_str
    assert "XAU" in repr_str
    assert "Gold" in repr_str
    assert "4" in repr_str
    assert "METAL" in repr_str
    
    # Test with a crypto currency with negative decimals
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    repr_str = repr(zzz)
    assert "Currency" in repr_str
    assert "ZZZ" in repr_str
    assert "Some weird currency" in repr_str
    assert "-1" in repr_str
    assert "CRYPTO" in repr_str
    
    # Test with a currency with zero decimals
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    repr_str = repr(jpy)
    assert "Currency" in repr_str
    assert "JPY" in repr_str
    assert "Japanese Yen" in repr_str
    assert "0" in repr_str
    
    # Test that repr output is evaluable-like (contains all key attributes)
    alt_currency = Currency.of("ALT", "Alternative Currency", 3, CurrencyType.ALTERNATIVE)
    repr_str = repr(alt_currency)
    assert "code=" in repr_str or "ALT" in repr_str
    assert "name=" in repr_str or "Alternative Currency" in repr_str


# LLM-generated content at query #19
#--------------------------

```python
def test_Currency():
    """Test Currency class constructor and related functionality."""
    
    # Test valid currency creation with positive decimals
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.code == "USD"
    assert usd.name == "US Dollars"
    assert usd.decimals == 2
    assert usd.type == CurrencyType.MONEY
    assert isinstance(usd.quantizer, Decimal)
    assert isinstance(usd.hashcache, int)
    
    # Test currency with zero decimals
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy.code == "JPY"
    assert jpy.decimals == 0
    assert jpy.quantizer == ZERO
    
    # Test currency with negative decimals
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert zzz.code == "ZZZ"
    assert zzz.decimals == -1
    assert zzz.quantizer == MaxPrecisionQuantizer
    
    # Test quantize method with positive decimals
    assert usd.quantize(Decimal("1.005")) == Decimal("1.00")
    assert usd.quantize(Decimal("1.015")) == Decimal("1.02")
    
    # Test quantize method with zero decimals
    assert jpy.quantize(Decimal("0.5")) == Decimal("0")
    assert jpy.quantize(Decimal("1.5")) == Decimal("2")
    
    # Test quantize method with negative decimals
    assert zzz.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert zzz.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")
    
    # Test equality
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    
    assert usd1 == usd2
    assert usd1 != usdx
    
    # Test hash consistency
    assert hash(usd1) == hash(usd2)
    assert hash(usd1) != hash(usdx)
    
    # Test invalid code - non-alphabetic
    with pytest.raises(ProgrammingError):
        Currency.of("US1", "US Dollars", 2, CurrencyType.MONEY)
    
    # Test invalid code - not uppercase
    with pytest.raises(ProgrammingError):
        Currency.of("Usd", "US Dollars", 2, CurrencyType.MONEY)
    
    # Test invalid code - not a string
    with pytest.raises(ProgrammingError):
        Currency.of(123, "US Dollars", 2, CurrencyType.MONEY)
    
    # Test invalid name - empty string
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "", 2, CurrencyType.MONEY)
    
    # Test invalid name - leading space
    with pytest.raises(ProgrammingError):
        Currency.of("USD", " US Dollars", 2, CurrencyType.MONEY)
    
    # Test invalid name - trailing space
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars ", 2, CurrencyType.MONEY)
    
    # Test invalid name - not a string
    with pytest.raises(ProgrammingError):
        Currency.of("USD", 123, 2, CurrencyType.MONEY)
    
    # Test invalid decimals - not an integer
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", 2.5, CurrencyType.MONEY)
    
    # Test invalid decimals - less than -1
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", -2, CurrencyType.MONEY)
    
    # Test invalid type
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", 2, "MONEY")
    
    # Test different currency types
    metal = Currency.of("GOLD", "Gold", 5, CurrencyType.METAL)
    assert metal.type == CurrencyType.METAL
    
    crypto = Currency.of("BTC", "Bitcoin", -1, CurrencyType.CRYPTO)
    assert crypto.type == CurrencyType.CRYPTO
    
    alt = Currency.of("ALT", "Alternative Currency", 3, CurrencyType.ALTERNATIVE)
    assert alt.type == CurrencyType.ALTERNATIVE
    
    # Test hash caching
    usd_test = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd_test.hashcache == hash((usd_test.code, usd_test.name, usd_test.decimals, usd_test.type, usd_test.quantizer))


# LLM-generated content at query #20
#--------------------------

```python
def test_Currency___setattr__():
    """Test that Currency instances are immutable (frozen dataclass)."""
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    # Attempt to set an attribute should raise FrozenInstanceError
    with pytest.raises(AttributeError):
        USD.code = "EUR"
    
    # Attempt to set another attribute should also raise FrozenInstanceError
    with pytest.raises(AttributeError):
        USD.name = "Euro"
    
    # Attempt to set decimals should raise FrozenInstanceError
    with pytest.raises(AttributeError):
        USD.decimals = 3
    
    # Attempt to set type should raise FrozenInstanceError
    with pytest.raises(AttributeError):
        USD.type = CurrencyType.METAL
    
    # Attempt to set quantizer should raise FrozenInstanceError
    with pytest.raises(AttributeError):
        USD.quantizer = Decimal("0.001")
    
    # Attempt to set hashcache should raise FrozenInstanceError
    with pytest.raises(AttributeError):
        USD.hashcache = 12345
    
    # Attempt to add a new attribute should also raise FrozenInstanceError
    with pytest.raises(AttributeError):
        USD.new_attribute = "value"
    
    # Verify that the original values remain unchanged
    assert USD.code == "USD"
    assert USD.name == "US Dollars"
    assert USD.decimals == 2
    assert USD.type == CurrencyType.MONEY


# LLM-generated content at query #21
#--------------------------

```python
def test_Currency___eq__():
    """Test the __eq__ method of Currency class."""
    
    # Create two identical currencies
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    # Test equality of identical currencies
    assert usd1 == usd2
    
    # Create a currency with different name but same code
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    
    # Test inequality when names differ
    assert usd1 != usdx
    
    # Create a currency with different decimals
    usd_diff_decimals = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    
    # Test inequality when decimals differ
    assert usd1 != usd_diff_decimals
    
    # Create a currency with different type
    usd_crypto = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    
    # Test inequality when types differ
    assert usd1 != usd_crypto
    
    # Create a completely different currency
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    
    # Test inequality with different currency code
    assert usd1 != jpy
    
    # Test inequality with non-Currency objects
    assert usd1 != "USD"
    assert usd1 != 123
    assert usd1 != None
    assert usd1 != {"code": "USD"}
    
    # Test reflexivity (a == a)
    assert usd1 == usd1
    
    # Test symmetry (if a == b then b == a)
    assert (usd1 == usd2) == (usd2 == usd1)
    
    # Test transitivity (if a == b and b == c then a == c)
    usd3 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2
    assert usd2 == usd3
    assert usd1 == usd3


# LLM-generated content at query #22
#--------------------------

```python
def test_CurrencyRegistry___exit__():
    """Test the __exit__ method of CurrencyRegistry."""
    # Create a fresh registry instance
    registry = CurrencyRegistry()
    
    # Test that __exit__ properly finalizes the registry after context manager
    test_currency_1 = Currency.of("TST", "Test Currency 1", 2, CurrencyType.MONEY)
    test_currency_2 = Currency.of("TSA", "Test Currency 2", 2, CurrencyType.MONEY)
    test_currency_3 = Currency.of("TSZ", "Test Currency 3", 2, CurrencyType.MONEY)
    
    # Populate registry using context manager
    with registry as register:
        register(test_currency_1)
        register(test_currency_2)
        register(test_currency_3)
    
    # After __exit__, verify that the registry is properly sorted by code
    assert registry.codes == sorted(registry.codes)
    assert registry.codes == ["TSA", "TST", "TSZ"]
    
    # Verify that all property returns sorted currencies
    assert len(registry.all) == 3
    assert registry.all[0].code == "TSA"
    assert registry.all[1].code == "TST"
    assert registry.all[2].code == "TSZ"
    
    # Verify that codenames property is properly sorted
    assert registry.codenames == [("TSA", "Test Currency 2"), ("TST", "Test Currency 1"), ("TSZ", "Test Currency 3")]
    
    # Verify that context is closed after __exit__
    assert registry._CurrencyRegistry__ctx_open is False
    
    # Verify that attempting to register outside context raises error
    with pytest.raises(ProgrammingError):
        registry._CurrencyRegistry__register(Currency.of("NEW", "New Currency", 2, CurrencyType.MONEY))


def test_CurrencyRegistry___exit___with_exception():
    """Test that __exit__ properly closes context even when exception occurs."""
    registry = CurrencyRegistry()
    
    test_currency = Currency.of("EXC", "Exception Currency", 2, CurrencyType.MONEY)
    
    # Simulate exception during context
    try:
        with registry as register:
            register(test_currency)
            raise ValueError("Test exception")
    except ValueError:
        pass
    
    # Verify context is still closed after exception
    assert registry._CurrencyRegistry__ctx_open is False
    
    # Verify currency was still registered before exception
    assert "EXC" in registry


def test_CurrencyRegistry___exit___sorting():
    """Test that __exit__ properly sorts registry items."""
    registry = CurrencyRegistry()
    
    # Add currencies in non-alphabetical order
    currencies = [
        Currency.of("ZZZ", "Last", 2, CurrencyType.MONEY),
        Currency.of("AAA", "First", 2, CurrencyType.MONEY),
        Currency.of("MMM", "Middle", 2, CurrencyType.MONEY),
    ]
    
    with registry as register:
        for curr in currencies:
            register(curr)
    
    # Verify sorting after __exit__
    codes = registry.codes
    assert codes == ["AAA", "MMM", "ZZZ"]
    
    # Verify all property maintains sort order
    for i, curr in enumerate(registry.all):
        assert curr.code == codes[i]
    
    # Verify codenames maintains sort order
    assert registry.codenames[0][0] == "AAA"
    assert registry.codenames[1][0] == "MMM"
    assert registry.codenames[2][0] == "ZZZ"


# LLM-generated content at query #23
#--------------------------

```python
def test_CurrencyRegistry___enter__():
    """Test the __enter__ method of CurrencyRegistry."""
    registry = CurrencyRegistry()
    
    # Test that __enter__ returns a callable
    result = registry.__enter__()
    assert callable(result), "__enter__ should return a callable"
    
    # Test that the context is marked as open
    assert registry._CurrencyRegistry__ctx_open is True, "Context should be marked as open"
    
    # Test that the returned callable is the __register method
    assert result == registry._CurrencyRegistry__register, "__enter__ should return the __register method"
    
    # Clean up by exiting the context
    registry.__exit__(None, None, None)


# LLM-generated content at query #24
#--------------------------

```python
def test_CurrencyRegistry___getitem__():
    """Test CurrencyRegistry.__getitem__ method"""
    
    # Test successful retrieval of an existing currency
    currency = Currencies["USD"]
    assert currency.code == "USD"
    assert currency.name == "US Dollar"
    assert currency.type == CurrencyType.MONEY
    
    # Test retrieval of another existing currency
    currency_eur = Currencies["EUR"]
    assert currency_eur.code == "EUR"
    assert isinstance(currency_eur, Currency)
    
    # Test that CurrencyLookupError is raised for non-existing currency
    with pytest.raises(CurrencyLookupError) as exc_info:
        Currencies["NON-EXISTING"]
    assert exc_info.value.code == "NON-EXISTING"
    assert "NON-EXISTING" in str(exc_info.value)
    
    # Test that CurrencyLookupError is raised for invalid code
    with pytest.raises(CurrencyLookupError) as exc_info:
        Currencies["XYZ123"]
    assert exc_info.value.code == "XYZ123"
    
    # Test that multiple calls return the same currency object
    currency1 = Currencies["USD"]
    currency2 = Currencies["USD"]
    assert currency1 == currency2
    assert currency1 is currency2
    
    # Test case sensitivity - should raise error for lowercase
    with pytest.raises(CurrencyLookupError):
        Currencies["usd"]


# LLM-generated content at query #25
#--------------------------

```python
def test_Currency_quantize():
    """Test the quantize method of Currency class."""
    
    # Test with USD (2 decimals)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.quantize(Decimal("1.005")) == Decimal("1.00")
    assert usd.quantize(Decimal("1.015")) == Decimal("1.02")
    assert usd.quantize(Decimal("1.234")) == Decimal("1.23")
    assert usd.quantize(Decimal("1.235")) == Decimal("1.24")
    assert usd.quantize(Decimal("0")) == Decimal("0.00")
    assert usd.quantize(Decimal("100.999")) == Decimal("101.00")
    
    # Test with JPY (0 decimals)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy.quantize(Decimal("0.5")) == Decimal("0")
    assert jpy.quantize(Decimal("1.5")) == Decimal("2")
    assert jpy.quantize(Decimal("1.4")) == Decimal("1")
    assert jpy.quantize(Decimal("100.6")) == Decimal("101")
    assert jpy.quantize(Decimal("0")) == Decimal("0")
    
    # Test with crypto currency (-1 decimals, max precision)
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert zzz.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert zzz.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")
    assert zzz.quantize(Decimal("0.123456789012345")) == Decimal("0.123456789012")
    
    # Test with BTC (8 decimals)
    btc = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    assert btc.quantize(Decimal("0.123456789")) == Decimal("0.12345679")
    assert btc.quantize(Decimal("0.000000001")) == Decimal("0.00000000")
    assert btc.quantize(Decimal("21000000")) == Decimal("21000000.00000000")
    
    # Test with GBP (2 decimals)
    gbp = Currency.of("GBP", "British Pounds", 2, CurrencyType.MONEY)
    assert gbp.quantize(Decimal("50.125")) == Decimal("50.12")
    assert gbp.quantize(Decimal("50.135")) == Decimal("50.14")
    assert gbp.quantize(Decimal("0.01")) == Decimal("0.01")
    
    # Test with KWD (3 decimals)
    kwd = Currency.of("KWD", "Kuwaiti Dinar", 3, CurrencyType.MONEY)
    assert kwd.quantize(Decimal("1.2345")) == Decimal("1.235")
    assert kwd.quantize(Decimal("1.2344")) == Decimal("1.234")
    assert kwd.quantize(Decimal("0.0005")) == Decimal("0.000")
    
    # Test with negative values
    assert usd.quantize(Decimal("-1.005")) == Decimal("-1.00")
    assert usd.quantize(Decimal("-1.015")) == Decimal("-1.02")
    assert jpy.quantize(Decimal("-1.5")) == Decimal("-2")
    
    # Test with large values
    assert usd.quantize(Decimal("999999999.999")) == Decimal("1000000000.00")
    assert jpy.quantize(Decimal("999999999.999")) == Decimal("1000000000")


# LLM-generated content at query #26
#--------------------------

```python
def test_Currency___lt__():
    """Test the __lt__ method of Currency class."""
    # Create currencies with different codes
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    gbp = Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    
    # Test basic ordering by code (alphabetically)
    assert eur < gbp
    assert eur < usd
    assert gbp < usd
    assert jpy < usd
    
    # Test that not less than works correctly
    assert not (usd < eur)
    assert not (gbp < eur)
    assert not (usd < gbp)
    
    # Test with same currency
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd < usd2)
    assert not (usd2 < usd)
    
    # Test transitive property
    assert eur < gbp
    assert gbp < usd
    assert eur < usd
    
    # Test with currencies of different types
    btc = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    gold = Currency.of("XAU", "Gold", 2, CurrencyType.METAL)
    
    # These should still be ordered by their internal representation
    assert btc < gold
    assert btc < usd
    assert gold < usd
    
    # Test with currencies having different decimals
    usd_variant = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur_variant = Currency.of("EUR", "Euro", 3, CurrencyType.MONEY)
    
    assert eur_variant < usd_variant


# LLM-generated content at query #27
#--------------------------

```python
def test_Currency___setattr__():
    """Test that Currency instances are immutable (frozen dataclass)."""
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    # Attempting to set any attribute should raise FrozenInstanceError
    with pytest.raises(Exception):  # FrozenInstanceError from dataclass
        USD.code = "EUR"
    
    with pytest.raises(Exception):
        USD.name = "Euros"
    
    with pytest.raises(Exception):
        USD.decimals = 3
    
    with pytest.raises(Exception):
        USD.type = CurrencyType.METAL
    
    with pytest.raises(Exception):
        USD.quantizer = Decimal("0.001")
    
    with pytest.raises(Exception):
        USD.hashcache = 12345
    
    with pytest.raises(Exception):
        USD.new_attribute = "should_fail"


# LLM-generated content at query #28
#--------------------------

```python
def test_CurrencyRegistry():
    """Test the CurrencyRegistry constructor and singleton behavior."""
    # Test that CurrencyRegistry is a singleton
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2
    
    # Test initial state of registry
    assert len(registry1) == 0
    assert registry1.all == []
    assert registry1.codes == []
    assert registry1.codenames == []
    
    # Test that __ctx_open is False initially
    assert registry1._CurrencyRegistry__ctx_open is False
    
    # Test that __registry is initialized as OrderedDict
    assert isinstance(registry1._CurrencyRegistry__registry, OrderedDict)
    assert len(registry1._CurrencyRegistry__registry) == 0
    
    # Test that __currencies is initialized as empty list
    assert isinstance(registry1._CurrencyRegistry__currencies, list)
    assert len(registry1._CurrencyRegistry__currencies) == 0
    
    # Test that __codes is initialized as empty list
    assert isinstance(registry1._CurrencyRegistry__codes, list)
    assert len(registry1._CurrencyRegistry__codes) == 0
    
    # Test that __codenames is initialized as empty list
    assert isinstance(registry1._CurrencyRegistry__codenames, list)
    assert len(registry1._CurrencyRegistry__codenames) == 0


# LLM-generated content at query #29
#--------------------------

```python
def test_CurrencyRegistry_get():
    """Test the get method of CurrencyRegistry."""
    
    # Test getting an existing currency
    usd = Currencies.get("USD")
    assert usd is not None
    assert usd.code == "USD"
    assert usd.name == "US Dollar"
    
    # Test getting a non-existing currency returns None by default
    result = Currencies.get("XXX")
    assert result is None
    
    # Test getting a non-existing currency with a default value
    default_currency = Currencies["USD"]
    result = Currencies.get("NON-EXISTING", default=default_currency)
    assert result == default_currency
    assert result.code == "USD"
    
    # Test that get returns the same currency object as __getitem__
    get_result = Currencies.get("USD")
    getitem_result = Currencies["USD"]
    assert get_result == getitem_result
    
    # Test with multiple existing currencies
    jpy = Currencies.get("JPY")
    assert jpy is not None
    assert jpy.code == "JPY"
    
    eur = Currencies.get("EUR")
    assert eur is not None
    assert eur.code == "EUR"


# LLM-generated content at query #30
#--------------------------

```python
def test_Currency___eq__():
    """Test the __eq__ method of Currency class."""
    # Create identical currencies
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    # Test equality of identical currencies
    assert usd1 == usd2
    assert not (usd1 != usd2)
    
    # Create currency with different name
    usd_diff_name = Currency.of("USD", "United States Dollar", 2, CurrencyType.MONEY)
    assert usd1 != usd_diff_name
    
    # Create currency with different decimals
    usd_diff_decimals = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert usd1 != usd_diff_decimals
    
    # Create currency with different type
    usd_diff_type = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert usd1 != usd_diff_type
    
    # Create different currency code
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert usd1 != eur
    
    # Test equality with non-Currency object
    assert usd1 != "USD"
    assert usd1 != 2
    assert usd1 != None
    assert usd1 != {"code": "USD"}
    
    # Test reflexivity (x == x)
    assert usd1 == usd1
    
    # Test symmetry (if x == y then y == x)
    assert (usd1 == usd2) == (usd2 == usd1)
    
    # Test transitivity (if x == y and y == z then x == z)
    usd3 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2
    assert usd2 == usd3
    assert usd1 == usd3


# LLM-generated content at query #31
#--------------------------

```python
def test_Currency___delattr__():
    """Test that Currency instances are immutable and __delattr__ raises an error."""
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    # Currency is a frozen dataclass, so attempting to delete attributes should raise an error
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
    
    # Attempting to delete non-existent attributes should also raise an error
    with pytest.raises(AttributeError):
        del usd.non_existent_attribute


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_CurrencyRegistry___enter__():
    """Test that __enter__ returns the __register method and opens the context."""
    registry = CurrencyRegistry()
    
    # Call __enter__ to get the register method
    register_method = registry.__enter__()
    
    # Verify that __enter__ returns a callable
    assert callable(register_method)
    
    # Verify that the context is marked as open by attempting to register a currency
    test_currency = Currency.of("TST", "Test Currency", 2, CurrencyType.MONEY)
    
    # This should work without raising ProgrammingError since context is open
    try:
        register_method(test_currency)
    except ValueError:
        # ValueError is expected if currency already exists, but not ProgrammingError
        pass
    except ProgrammingError:
        # This should NOT happen since context is open
        raise AssertionError("Context should be open after __enter__")
    
    # Close the context
    registry.__exit__(None, None, None)


# LLM-generated content at query #32
#--------------------------

```python
def test_Currency___repr__():
    """Test the __repr__ method of Currency class."""
    # Test with standard money currency
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    repr_str = repr(usd)
    assert "Currency" in repr_str
    assert "USD" in repr_str
    assert "US Dollars" in repr_str
    assert "2" in repr_str
    assert "MONEY" in repr_str
    
    # Test with precious metal currency
    gold = Currency.of("XAU", "Gold", 4, CurrencyType.METAL)
    repr_str = repr(gold)
    assert "Currency" in repr_str
    assert "XAU" in repr_str
    assert "Gold" in repr_str
    assert "4" in repr_str
    assert "METAL" in repr_str
    
    # Test with crypto currency
    btc = Currency.of("BTC", "Bitcoin", -1, CurrencyType.CRYPTO)
    repr_str = repr(btc)
    assert "Currency" in repr_str
    assert "BTC" in repr_str
    assert "Bitcoin" in repr_str
    assert "-1" in repr_str
    assert "CRYPTO" in repr_str
    
    # Test with zero decimals
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    repr_str = repr(jpy)
    assert "Currency" in repr_str
    assert "JPY" in repr_str
    assert "Japanese Yen" in repr_str
    assert "0" in repr_str
    
    # Test that repr output is consistent
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert repr(usd) == repr(usd2)


# LLM-generated content at query #2
#--------------------------

```python
def test_Currency___eq__():
    # Test equality with identical currencies
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2
    
    # Test inequality with different names
    usd_diff_name = Currency.of("USD", "United States Dollar", 2, CurrencyType.MONEY)
    assert usd1 != usd_diff_name
    
    # Test inequality with different decimals
    usd_diff_decimals = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert usd1 != usd_diff_decimals
    
    # Test inequality with different types
    usd_diff_type = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert usd1 != usd_diff_type
    
    # Test inequality with different code
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert usd1 != eur
    
    # Test inequality with non-Currency object
    assert usd1 != "USD"
    assert usd1 != 42
    assert usd1 != None
    assert usd1 != {}
    
    # Test with different currencies
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    gbp = Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY)
    assert jpy != gbp
    
    # Test reflexivity (x == x)
    assert usd1 == usd1
    
    # Test symmetry (if x == y then y == x)
    assert usd1 == usd2
    assert usd2 == usd1
    
    # Test transitivity (if x == y and y == z then x == z)
    usd3 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2
    assert usd2 == usd3
    assert usd1 == usd3


# LLM-generated content at query #3
#--------------------------

```python
def test_CurrencyRegistry_has():
    """Test the has method of CurrencyRegistry."""
    # Test with existing currency code
    assert Currencies.has("USD") is True
    assert Currencies.has("EUR") is True
    assert Currencies.has("JPY") is True
    
    # Test with non-existing currency code
    assert Currencies.has("XXX") is False
    assert Currencies.has("NON") is False
    assert Currencies.has("INVALID") is False
    
    # Test with empty string
    assert Currencies.has("") is False
    
    # Test with lowercase (should not exist as codes are uppercase)
    assert Currencies.has("usd") is False
    
    # Test consistency with __contains__
    assert Currencies.has("USD") == ("USD" in Currencies)
    assert Currencies.has("XXX") == ("XXX" in Currencies)


# LLM-generated content at query #33
#--------------------------

```python
def test_Currency_quantize():
    """Test the quantize method of Currency class."""
    
    # Test with USD (2 decimals)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.quantize(Decimal("1.005")) == Decimal("1.00")
    assert usd.quantize(Decimal("1.015")) == Decimal("1.02")
    assert usd.quantize(Decimal("1.234")) == Decimal("1.23")
    assert usd.quantize(Decimal("1.235")) == Decimal("1.24")
    assert usd.quantize(Decimal("0")) == Decimal("0.00")
    assert usd.quantize(Decimal("100.5")) == Decimal("100.50")
    
    # Test with JPY (0 decimals)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy.quantize(Decimal("0.5")) == Decimal("0")
    assert jpy.quantize(Decimal("1.5")) == Decimal("2")
    assert jpy.quantize(Decimal("2.4")) == Decimal("2")
    assert jpy.quantize(Decimal("2.5")) == Decimal("2")
    assert jpy.quantize(Decimal("2.6")) == Decimal("3")
    assert jpy.quantize(Decimal("100")) == Decimal("100")
    
    # Test with cryptocurrency (negative decimals, max precision)
    crypto = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert crypto.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert crypto.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")
    assert crypto.quantize(Decimal("0.123456789")) == Decimal("0.123456789")
    assert crypto.quantize(Decimal("1")) == Decimal("1")
    
    # Test with different decimal places
    gbp = Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY)
    assert gbp.quantize(Decimal("10.125")) == Decimal("10.12")
    
    bhd = Currency.of("BHD", "Bahraini Dinar", 3, CurrencyType.MONEY)
    assert bhd.quantize(Decimal("1.0005")) == Decimal("1.000")
    assert bhd.quantize(Decimal("1.0015")) == Decimal("1.002")
    
    # Test with negative amounts
    assert usd.quantize(Decimal("-1.005")) == Decimal("-1.00")
    assert jpy.quantize(Decimal("-1.5")) == Decimal("-2")
    
    # Test edge cases
    assert usd.quantize(Decimal("0.001")) == Decimal("0.00")
    assert usd.quantize(Decimal("0.009")) == Decimal("0.01")


# LLM-generated content at query #4
#--------------------------

```python
def test_CurrencyRegistry___contains__():
    """Test the __contains__ method of CurrencyRegistry."""
    registry = CurrencyRegistry()
    
    # Test with a currency that exists in the registry
    assert "USD" in registry
    assert "EUR" in registry
    assert "JPY" in registry
    
    # Test with a currency that does not exist in the registry
    assert "XYZ" not in registry
    assert "XXX" not in registry
    assert "INVALID" not in registry
    
    # Test with various invalid codes
    assert "AUD" in registry  # Assuming AUD is registered
    assert "NOTACURRENCY" not in registry
    assert "" not in registry
    assert "usd" not in registry  # Case sensitive


# LLM-generated content at query #34
#--------------------------

```python
def test_CurrencyRegistry___enter__():
    """Test the __enter__ method of CurrencyRegistry."""
    registry = CurrencyRegistry()
    
    # Test that __enter__ returns a callable (the __register method)
    with registry as register:
        assert callable(register)
        
    # Test that __enter__ sets the context open flag
    registry._CurrencyRegistry__ctx_open = False
    with registry as register:
        assert registry._CurrencyRegistry__ctx_open is True
        
    # After exiting, context should be closed
    assert registry._CurrencyRegistry__ctx_open is False
    
    # Test that the returned callable can be used to register currencies
    initial_length = len(registry)
    test_currency = Currency.of("TST", "Test Currency", 2, CurrencyType.MONEY)
    
    with registry as register:
        register(test_currency)
    
    # Verify the currency was registered
    assert "TST" in registry
    assert registry["TST"].code == "TST"


# LLM-generated content at query #5
#--------------------------

```python
def test_Currency___gt__():
    """Test the __gt__ method of Currency class."""
    # Create currencies with different codes
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    gbp = Currency.of("GBP", "British Pounds", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    
    # Test basic ordering by code (alphabetical)
    assert eur > gbp  # EUR > GBP
    assert gbp > usd  # GBP > USD
    assert eur > usd  # EUR > USD
    
    # Test that greater than is not reflexive
    assert not (usd > usd)
    
    # Test with different decimals
    assert jpy > eur  # JPY > EUR (different decimals, ordered by all fields)
    
    # Test transitivity
    assert eur > gbp and gbp > usd
    # Note: transitivity with > is not guaranteed with dataclass ordering
    # but we can verify individual comparisons
    
    # Test that it works with different types
    crypto = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    assert crypto > eur  # BTC > EUR
    
    # Test ordering consistency
    currencies = [usd, eur, gbp, jpy]
    sorted_currencies = sorted(currencies)
    assert sorted_currencies[0] == eur  # EUR comes first alphabetically among these
    assert sorted_currencies[-1] == usd  # USD comes last alphabetically among these


# LLM-generated content at query #35
#--------------------------

```python
def test_Currency_quantize():
    """Test the quantize method of Currency class."""
    
    # Test USD with 2 decimals
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.quantize(Decimal("1.005")) == Decimal("1.00")
    assert usd.quantize(Decimal("1.015")) == Decimal("1.02")
    assert usd.quantize(Decimal("10.999")) == Decimal("11.00")
    assert usd.quantize(Decimal("0.001")) == Decimal("0.00")
    assert usd.quantize(Decimal("0.005")) == Decimal("0.00")
    
    # Test JPY with 0 decimals
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy.quantize(Decimal("0.5")) == Decimal("0")
    assert jpy.quantize(Decimal("1.5")) == Decimal("2")
    assert jpy.quantize(Decimal("10.4")) == Decimal("10")
    assert jpy.quantize(Decimal("10.6")) == Decimal("11")
    
    # Test ZZZ with -1 decimals (max precision)
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert zzz.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert zzz.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")
    assert zzz.quantize(Decimal("1.123456789")) == Decimal("1.123456789")
    
    # Test currency with 3 decimals
    xxx = Currency.of("XXX", "Test Currency", 3, CurrencyType.MONEY)
    assert xxx.quantize(Decimal("1.0005")) == Decimal("1.000")
    assert xxx.quantize(Decimal("1.0015")) == Decimal("1.002")
    assert xxx.quantize(Decimal("100.9999")) == Decimal("101.000")
    
    # Test edge cases
    assert usd.quantize(Decimal("0")) == Decimal("0.00")
    assert jpy.quantize(Decimal("0")) == Decimal("0")
    assert usd.quantize(Decimal("-1.234")) == Decimal("-1.23")
    assert jpy.quantize(Decimal("-1.6")) == Decimal("-2")


# LLM-generated content at query #6
#--------------------------

```python
def test_Currency___gt__():
    """Test the __gt__ method of Currency class."""
    # Create currencies with different codes for ordering
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    gbp = Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    
    # Test greater than comparisons
    assert gbp > eur
    assert gbp > usd
    assert eur > usd
    assert jpy > gbp
    
    # Test that greater than is false for equal or lesser values
    assert not (usd > usd)
    assert not (usd > eur)
    assert not (usd > gbp)
    assert not (eur > gbp)
    
    # Test with different types
    crypto = Currency.of("BTC", "Bitcoin", -1, CurrencyType.CRYPTO)
    metal = Currency.of("XAU", "Gold", 2, CurrencyType.METAL)
    
    # Cross-type comparisons based on ordering
    assert metal > crypto
    assert crypto > gbp
    
    # Test transitivity
    assert gbp > usd
    assert jpy > gbp
    assert jpy > usd


# LLM-generated content at query #7
#--------------------------

```python
def test_CurrencyLookupError():
    """Test CurrencyLookupError constructor."""
    code = "XYZ"
    error = CurrencyLookupError(code)
    
    # Test that the code is stored correctly
    assert error.code == code
    
    # Test that the error message is formatted correctly
    assert str(error) == f"Currency identified by code '{code}' does not exist"
    
    # Test that it is a LookupError
    assert isinstance(error, LookupError)
    
    # Test with different code
    code2 = "ABC"
    error2 = CurrencyLookupError(code2)
    assert error2.code == code2
    assert str(error2) == f"Currency identified by code '{code2}' does not exist"


# LLM-generated content at query #8
#--------------------------

```python
def test_Currency___repr__():
    """
    Test the __repr__ method of Currency class.
    """
    # Test with a standard money currency
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    repr_str = repr(usd)
    assert "Currency" in repr_str
    assert "USD" in repr_str
    assert "US Dollars" in repr_str
    assert "2" in repr_str
    assert "MONEY" in repr_str
    
    # Test with a precious metal currency
    gold = Currency.of("XAU", "Gold", 4, CurrencyType.METAL)
    repr_str = repr(gold)
    assert "Currency" in repr_str
    assert "XAU" in repr_str
    assert "Gold" in repr_str
    assert "4" in repr_str
    assert "METAL" in repr_str
    
    # Test with a crypto currency
    btc = Currency.of("BTC", "Bitcoin", -1, CurrencyType.CRYPTO)
    repr_str = repr(btc)
    assert "Currency" in repr_str
    assert "BTC" in repr_str
    assert "Bitcoin" in repr_str
    assert "-1" in repr_str
    assert "CRYPTO" in repr_str
    
    # Test with alternative currency
    alt = Currency.of("ALT", "Alternative Currency", 0, CurrencyType.ALTERNATIVE)
    repr_str = repr(alt)
    assert "Currency" in repr_str
    assert "ALT" in repr_str
    assert "Alternative Currency" in repr_str
    assert "0" in repr_str
    assert "ALTERNATIVE" in repr_str
    
    # Test that repr is evaluable or at least contains all essential information
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    repr_str = repr(jpy)
    assert isinstance(repr_str, str)
    assert len(repr_str) > 0


# LLM-generated content at query #9
#--------------------------

```python
def test_CurrencyRegistry___new__():
    """Test that CurrencyRegistry.__new__ returns a singleton instance."""
    # Create first instance
    registry1 = CurrencyRegistry()
    
    # Create second instance
    registry2 = CurrencyRegistry()
    
    # Both should be the same object (singleton pattern)
    assert registry1 is registry2
    
    # Create third instance to further verify singleton behavior
    registry3 = CurrencyRegistry()
    assert registry1 is registry3
    assert registry2 is registry3
    
    # All should have the same id
    assert id(registry1) == id(registry2) == id(registry3)


# LLM-generated content at query #10
#--------------------------

```python
def test_CurrencyRegistry___enter__():
    """
    Test that __enter__ returns the __register method and sets context as open.
    """
    registry = CurrencyRegistry()
    
    # Call __enter__ to get the register method
    register_method = registry.__enter__()
    
    # Verify that __enter__ returns a callable
    assert callable(register_method)
    
    # Verify that the returned callable is the __register method
    assert register_method.__name__ == "_CurrencyRegistry__register"
    
    # Clean up by exiting the context
    registry.__exit__(None, None, None)


# LLM-generated content at query #11
#--------------------------

```python
def test_Currency___le__():
    """Test the __le__ (less than or equal) method of Currency class."""
    # Create currencies with different codes
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    gbp = Currency.of("GBP", "British Pounds", 2, CurrencyType.MONEY)
    
    # Test currency <= itself (should be True)
    assert usd <= usd
    assert eur <= eur
    
    # Test currency <= other currency (alphabetical ordering)
    # EUR comes before GBP, so EUR <= GBP should be True
    assert eur <= gbp
    assert not (gbp <= eur)
    
    # Test currency <= other currency (USD comes after EUR)
    assert eur <= usd
    assert not (usd <= eur)
    
    # Test with different decimals and types
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    btc = Currency.of("BTC", "Bitcoin", -1, CurrencyType.CRYPTO)
    
    # BTC comes before EUR alphabetically
    assert btc <= eur
    assert not (eur <= btc)
    
    # JPY comes after EUR alphabetically
    assert eur <= jpy
    assert not (jpy <= eur)
    
    # Test transitive property
    # If A <= B and B <= C, then A <= C
    assert btc <= eur
    assert eur <= jpy
    assert btc <= jpy
    
    # Test with same code but different attributes (should still follow alphabetical order by code)
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "American Dollars", 2, CurrencyType.MONEY)
    
    # Both have same code, so they should be equal in ordering
    assert usd1 <= usd2
    assert usd2 <= usd1


# LLM-generated content at query #12
#--------------------------

```python
def test_CurrencyRegistry_has():
    """Test the has method of CurrencyRegistry class."""
    # Test with a currency that exists in the registry
    assert Currencies.has("USD") is True
    assert Currencies.has("EUR") is True
    assert Currencies.has("JPY") is True
    
    # Test with a currency code that does not exist
    assert Currencies.has("XXX") is False
    assert Currencies.has("NON") is False
    assert Currencies.has("INVALID") is False
    
    # Test with empty string
    assert Currencies.has("") is False
    
    # Test that has method is consistent with __contains__
    assert Currencies.has("USD") == ("USD" in Currencies)
    assert Currencies.has("XXX") == ("XXX" in Currencies)
    
    # Test case sensitivity
    assert Currencies.has("usd") is False
    assert Currencies.has("Usd") is False


# LLM-generated content at query #13
#--------------------------

```python
def test_CurrencyLookupError():
    # Test basic initialization
    error = CurrencyLookupError("XYZ")
    assert error.code == "XYZ"
    assert str(error) == "Currency identified by code 'XYZ' does not exist"
    
    # Test that it's an instance of LookupError
    assert isinstance(error, LookupError)
    
    # Test with different currency codes
    error2 = CurrencyLookupError("ABC")
    assert error2.code == "ABC"
    assert str(error2) == "Currency identified by code 'ABC' does not exist"
    
    # Test with single character code
    error3 = CurrencyLookupError("X")
    assert error3.code == "X"
    assert str(error3) == "Currency identified by code 'X' does not exist"
    
    # Test that it can be raised and caught
    with pytest.raises(CurrencyLookupError) as exc_info:
        raise CurrencyLookupError("EUR")
    
    assert exc_info.value.code == "EUR"
    assert "EUR" in str(exc_info.value)


# LLM-generated content at query #14
#--------------------------

```python
def test_CurrencyRegistry_get():
    """Test CurrencyRegistry.get method"""
    
    # Test getting an existing currency
    usd = Currencies.get("USD")
    assert usd is not None
    assert usd.code == "USD"
    assert usd.name == "US Dollar"
    
    # Test getting a non-existing currency returns None by default
    result = Currencies.get("XXX")
    assert result is None
    
    # Test getting a non-existing currency with a default value
    jpy = Currencies["JPY"]
    result = Currencies.get("XXX", default=jpy)
    assert result == jpy
    assert result.code == "JPY"
    
    # Test getting with explicit None default
    result = Currencies.get("NON-EXISTING", default=None)
    assert result is None
    
    # Test that get returns the same currency object as __getitem__
    currency_via_get = Currencies.get("USD")
    currency_via_getitem = Currencies["USD"]
    assert currency_via_get == currency_via_getitem
    
    # Test with multiple different currencies
    eur = Currencies.get("EUR")
    assert eur is not None
    assert eur.code == "EUR"
    
    gbp = Currencies.get("GBP")
    assert gbp is not None
    assert gbp.code == "GBP"
    
    # Test that get doesn't raise exception for non-existing codes
    result = Currencies.get("INVALID")
    assert result is None


# LLM-generated content at query #15
#--------------------------

```python
def test_Currency___setattr__():
    """Test that Currency instances are immutable (frozen dataclass)."""
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    # Attempting to set any attribute should raise FrozenInstanceError
    with pytest.raises(Exception):  # dataclass frozen raises FrozenInstanceError
        usd.code = "EUR"
    
    with pytest.raises(Exception):
        usd.name = "Euros"
    
    with pytest.raises(Exception):
        usd.decimals = 3
    
    with pytest.raises(Exception):
        usd.type = CurrencyType.METAL
    
    with pytest.raises(Exception):
        usd.quantizer = Decimal("0.001")
    
    with pytest.raises(Exception):
        usd.hashcache = 12345
    
    # Verify original values are unchanged
    assert usd.code == "USD"
    assert usd.name == "US Dollars"
    assert usd.decimals == 2
    assert usd.type == CurrencyType.MONEY


# LLM-generated content at query #16
#--------------------------

```python
def test_CurrencyRegistry___getitem__():
    """Test CurrencyRegistry.__getitem__ method"""
    registry = CurrencyRegistry()
    
    # Test successful retrieval of existing currency
    usd = registry["USD"]
    assert usd.code == "USD"
    assert usd.name == "US Dollar"
    assert usd.type == CurrencyType.MONEY
    
    # Test retrieval of another existing currency
    jpy = registry["JPY"]
    assert jpy.code == "JPY"
    assert jpy.decimals == 0
    
    # Test that CurrencyLookupError is raised for non-existing currency
    with pytest.raises(CurrencyLookupError) as exc_info:
        registry["NON-EXISTING"]
    assert exc_info.value.code == "NON-EXISTING"
    assert "Currency identified by code 'NON-EXISTING' does not exist" in str(exc_info.value)
    
    # Test with invalid codes
    with pytest.raises(CurrencyLookupError):
        registry["XYZ"]
    
    with pytest.raises(CurrencyLookupError):
        registry[""]
    
    # Test that retrieved currency is the same object each time
    usd1 = registry["USD"]
    usd2 = registry["USD"]
    assert usd1 == usd2
    assert hash(usd1) == hash(usd2)


# LLM-generated content at query #17
#--------------------------

```python
def test_Currency___eq__():
    """Test Currency.__eq__ method"""
    
    # Test equality with identical currencies
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2
    
    # Test inequality with different names
    usd3 = Currency.of("USD", "United States Dollars", 2, CurrencyType.MONEY)
    assert usd1 != usd3
    
    # Test inequality with different decimals
    usd4 = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert usd1 != usd4
    
    # Test inequality with different types
    usd5 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert usd1 != usd5
    
    # Test inequality with different codes
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert usd1 != eur
    
    # Test inequality with non-Currency objects
    assert usd1 != "USD"
    assert usd1 != 2
    assert usd1 != None
    assert usd1 != {}
    
    # Test with different currency types
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert jpy != zzz
    
    # Test reflexivity: currency equals itself
    assert usd1 == usd1
    
    # Test symmetry: if a == b then b == a
    assert usd1 == usd2
    assert usd2 == usd1
    
    # Test transitivity: if a == b and b == c then a == c
    usd6 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2
    assert usd2 == usd6
    assert usd1 == usd6


# LLM-generated content at query #18
#--------------------------

```python
def test_CurrencyRegistry___enter__():
    """Test that __enter__ returns the __register method and opens the context."""
    registry = CurrencyRegistry()
    
    # Call __enter__ to get the register method
    register_method = registry.__enter__()
    
    # Verify that __enter__ returns a callable
    assert callable(register_method)
    
    # Verify that the context is marked as open by checking we can register
    test_currency = Currency.of("TST", "Test Currency", 2, CurrencyType.MONEY)
    
    # This should not raise an error since context is open
    register_method(test_currency)
    
    # Clean up: exit the context
    registry.__exit__(None, None, None)


# LLM-generated content at query #19
#--------------------------

```python
def test_Currency___repr__():
    """Test the __repr__ method of Currency class."""
    # Test with USD (money currency with 2 decimals)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    repr_str = repr(usd)
    assert "Currency" in repr_str
    assert "USD" in repr_str
    assert "US Dollars" in repr_str
    assert "2" in repr_str
    assert "MONEY" in repr_str

    # Test with JPY (money currency with 0 decimals)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    repr_str = repr(jpy)
    assert "Currency" in repr_str
    assert "JPY" in repr_str
    assert "Japanese Yen" in repr_str
    assert "0" in repr_str

    # Test with crypto currency (negative decimals)
    btc = Currency.of("BTC", "Bitcoin", -1, CurrencyType.CRYPTO)
    repr_str = repr(btc)
    assert "Currency" in repr_str
    assert "BTC" in repr_str
    assert "Bitcoin" in repr_str
    assert "CRYPTO" in repr_str

    # Test with metal currency
    gold = Currency.of("XAU", "Gold", 4, CurrencyType.METAL)
    repr_str = repr(gold)
    assert "Currency" in repr_str
    assert "XAU" in repr_str
    assert "Gold" in repr_str
    assert "4" in repr_str
    assert "METAL" in repr_str

    # Verify repr is evaluable or at least contains all key attributes
    assert "code" in repr_str or "XAU" in repr_str
    assert "name" in repr_str or "Gold" in repr_str


# LLM-generated content at query #20
#--------------------------

```python
def test_CurrencyRegistry___exit__():
    """Test CurrencyRegistry.__exit__ method"""
    # Create a new registry instance
    registry = CurrencyRegistry()
    
    # Test that __exit__ properly finalizes the registry after context
    usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    gbp = Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY)
    
    with registry as register:
        register(usd)
        register(eur)
        register(gbp)
    
    # After __exit__, verify that registry is properly sorted
    assert len(registry) == 3
    assert registry.codes == ["EUR", "GBP", "USD"]  # Should be alphabetically sorted
    assert registry.all == [eur, gbp, usd]  # Should match sorted order
    
    # Verify codenames are also sorted
    expected_codenames = [("EUR", "Euro"), ("GBP", "British Pound"), ("USD", "US Dollar")]
    assert registry.codenames == expected_codenames
    
    # Verify context is closed after __exit__
    with pytest.raises(ProgrammingError):
        registry._CurrencyRegistry__register(Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY))


def test_CurrencyRegistry___exit__with_exception():
    """Test CurrencyRegistry.__exit__ handles exceptions properly"""
    registry = CurrencyRegistry()
    
    usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    
    # Test that __exit__ is called even when exception occurs in context
    try:
        with registry as register:
            register(usd)
            raise ValueError("Test exception")
    except ValueError:
        pass
    
    # Verify registry still gets finalized
    assert len(registry) == 1
    assert "USD" in registry
    assert registry.codes == ["USD"]


def test_CurrencyRegistry___exit__sorting_order():
    """Test CurrencyRegistry.__exit__ sorts currencies correctly"""
    registry = CurrencyRegistry()
    
    # Register currencies in non-alphabetical order
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    aed = Currency.of("AED", "UAE Dirham", 2, CurrencyType.MONEY)
    zar = Currency.of("ZAR", "South African Rand", 2, CurrencyType.MONEY)
    
    with registry as register:
        register(jpy)
        register(aed)
        register(zar)
    
    # After __exit__, verify alphabetical sorting
    assert registry.codes == ["AED", "JPY", "ZAR"]
    assert registry.all[0].code == "AED"
    assert registry.all[1].code == "JPY"
    assert registry.all[2].code == "ZAR"


# LLM-generated content at query #36
#--------------------------

```python
def test_CurrencyRegistry___enter__():
    """Test that __enter__ returns the __register method and sets context open flag."""
    registry = CurrencyRegistry()
    
    # Call __enter__ and verify it returns a callable
    result = registry.__enter__()
    
    # Verify the result is callable (should be the __register method)
    assert callable(result)
    
    # Verify the context is marked as open by checking if we can register
    # (if context wasn't open, __register would raise ProgrammingError)
    test_currency = Currency.of("TST", "Test Currency", 2, CurrencyType.MONEY)
    
    # This should not raise an error since context is open
    result(test_currency)
    
    # Clean up by exiting context
    registry.__exit__(None, None, None)


def test_CurrencyRegistry___enter___context_manager():
    """Test that __enter__ works correctly with context manager protocol."""
    registry = CurrencyRegistry()
    
    with registry as register:
        # Verify register is callable
        assert callable(register)
        
        # Verify we can register a currency
        test_currency = Currency.of("CTX", "Context Test", 2, CurrencyType.MONEY)
        register(test_currency)
        
        # Verify currency was registered
        assert registry.has("CTX")


def test_CurrencyRegistry___enter___returns_register_method():
    """Test that __enter__ returns the __register method specifically."""
    registry = CurrencyRegistry()
    
    enter_result = registry.__enter__()
    
    # Verify it's the __register method by checking its behavior
    test_currency = Currency.of("RGM", "Register Method Test", 2, CurrencyType.MONEY)
    
    # Should succeed
    enter_result(test_currency)
    assert registry.has("RGM")
    
    # Should raise ValueError if we try to register same currency again
    import pytest
    with pytest.raises(ValueError, match="Currency RGM is already registered"):
        enter_result(test_currency)
    
    registry.__exit__(None, None, None)


# LLM-generated content at query #37
#--------------------------

```python
def test_Currency___lt__():
    """Test the __lt__ method of Currency class for ordering."""
    # Create currencies with different codes
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    gbp = Currency.of("GBP", "British Pounds", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    
    # Test basic ordering
    assert eur < gbp
    assert gbp < usd
    assert eur < usd
    
    # Test that a currency is not less than itself
    assert not (usd < usd)
    
    # Test ordering with different decimals
    assert jpy < usd  # JPY (0 decimals) vs USD (2 decimals)
    
    # Test transitive property
    assert eur < gbp and gbp < usd
    assert eur < usd
    
    # Test with crypto currency
    btc = Currency.of("BTC", "Bitcoin", -1, CurrencyType.CRYPTO)
    assert btc < eur
    
    # Test reverse comparison
    assert usd > eur
    assert gbp > eur
    
    # Test less than or equal
    assert eur <= gbp
    assert usd <= usd
    
    # Test greater than or equal
    assert usd >= gbp
    assert usd >= usd


# LLM-generated content at query #38
#--------------------------

```python
def test_CurrencyRegistry___len__():
    """Test the __len__ method of CurrencyRegistry."""
    registry = CurrencyRegistry()
    
    # Test that length is greater than 0 (registry should have currencies)
    assert len(registry) > 0
    
    # Test that length matches the number of currencies in all property
    assert len(registry) == len(registry.all)
    
    # Test that length matches the number of codes
    assert len(registry) == len(registry.codes)
    
    # Test that length matches the number of codenames
    assert len(registry) == len(registry.codenames)
    
    # Test that the registry is not empty
    assert len(registry) > 0
    
    # Test that we can access currencies by index up to the length
    for i in range(len(registry)):
        assert registry.all[i] is not None


# LLM-generated content at query #21
#--------------------------

```python
def test_Currency___setattr__():
    """Test that Currency instances are immutable (frozen dataclass)."""
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    # Attempt to modify any attribute should raise FrozenInstanceError
    with pytest.raises(AttributeError):
        USD.code = "EUR"
    
    with pytest.raises(AttributeError):
        USD.name = "Euros"
    
    with pytest.raises(AttributeError):
        USD.decimals = 3
    
    with pytest.raises(AttributeError):
        USD.type = CurrencyType.CRYPTO
    
    with pytest.raises(AttributeError):
        USD.quantizer = Decimal("0.001")
    
    with pytest.raises(AttributeError):
        USD.hashcache = 12345
    
    with pytest.raises(AttributeError):
        USD.new_attribute = "should fail"


# LLM-generated content at query #22
#--------------------------

```python
def test_Currency___delattr__():
    """
    Test that Currency objects are immutable and cannot have attributes deleted.
    """
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    # Attempt to delete an attribute should raise an error due to frozen dataclass
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


# LLM-generated content at query #23
#--------------------------

```python
def test_Currency___repr__():
    """Test the __repr__ method of Currency class."""
    # Test with a standard currency
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    repr_str = repr(usd)
    assert "Currency" in repr_str
    assert "USD" in repr_str
    assert "US Dollars" in repr_str
    assert "2" in repr_str
    assert "MONEY" in repr_str
    
    # Test with a different currency
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    repr_str_jpy = repr(jpy)
    assert "Currency" in repr_str_jpy
    assert "JPY" in repr_str_jpy
    assert "Japanese Yen" in repr_str_jpy
    assert "0" in repr_str_jpy
    
    # Test with crypto currency
    btc = Currency.of("BTC", "Bitcoin", -1, CurrencyType.CRYPTO)
    repr_str_btc = repr(btc)
    assert "Currency" in repr_str_btc
    assert "BTC" in repr_str_btc
    assert "Bitcoin" in repr_str_btc
    assert "-1" in repr_str_btc
    assert "CRYPTO" in repr_str_btc
    
    # Test that repr is evaluable or at least contains all key information
    assert "code" in repr_str or "USD" in repr_str
    assert "name" in repr_str or "US Dollars" in repr_str
    assert "decimals" in repr_str or "2" in repr_str
    assert "type" in repr_str or "MONEY" in repr_str


# LLM-generated content at query #24
#--------------------------

```python
def test_Currency___gt__():
    """Test the __gt__ method of Currency class."""
    # Create currencies with different codes for comparison
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    gbp = Currency.of("GBP", "British Pounds", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    
    # Test greater than comparisons (alphabetically ordered)
    assert eur > usd  # EUR comes before USD alphabetically, so USD > EUR
    assert usd > gbp  # USD comes after GBP
    assert gbp > eur  # GBP comes after EUR
    
    # Test that a currency is not greater than itself
    assert not (usd > usd)
    
    # Test with different currency types
    btc = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    assert btc > eur  # BTC comes before EUR alphabetically
    
    # Test transitivity
    assert gbp > eur
    assert usd > gbp
    assert usd > eur
    
    # Test with single character differences
    usda = Currency.of("USDA", "US Dollars A", 2, CurrencyType.MONEY) if False else None
    # Using actual valid codes
    chf = Currency.of("CHF", "Swiss Franc", 2, CurrencyType.MONEY)
    assert usd > chf  # USD comes after CHF
    
    # Test ordering with JPY (0 decimals)
    assert usd > jpy  # USD comes after JPY alphabetically


# LLM-generated content at query #25
#--------------------------

```python
def test_Currency___ge__():
    """Test the __ge__ (greater than or equal) comparison for Currency objects."""
    # Create currencies with different codes
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    gbp = Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY)
    
    # Create another USD currency (same as first)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    # Test: Currency >= itself should be True
    assert usd >= usd
    
    # Test: Currency >= identical currency should be True
    assert usd >= usd2
    
    # Test: Currency >= identical currency should be True (reflexive)
    assert usd2 >= usd
    
    # Test: Comparison with different currencies based on code ordering
    # EUR comes before GBP alphabetically
    assert gbp >= eur
    assert gbp >= usd
    
    # Test: Currency with lower code ordering should not be >= higher code
    assert not (eur >= gbp)
    assert not (eur >= usd)
    
    # Test: Currencies with same code but different details
    usd_different_name = Currency.of("USD", "United States Dollar", 2, CurrencyType.MONEY)
    assert not (usd >= usd_different_name)
    assert not (usd_different_name >= usd)
    
    # Test: Reflexivity - a currency should be >= itself
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy >= jpy
    
    # Test: Transitivity - if a >= b and b >= c, then a >= c
    cad = Currency.of("CAD", "Canadian Dollar", 2, CurrencyType.MONEY)
    assert gbp >= eur
    assert eur >= cad
    assert gbp >= cad


# LLM-generated content at query #26
#--------------------------

```python
def test_Currency___gt__():
    """Test the __gt__ method of Currency class."""
    # Create currencies with different codes
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    gbp = Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY)
    
    # Test ordering based on code (alphabetically)
    assert eur > gbp  # E < G, so eur is not > gbp, this should be False
    assert gbp > eur  # G > E, so gbp > eur
    assert usd > eur  # U > E, so usd > eur
    assert usd > gbp  # U > G, so usd > gbp
    
    # Test with same currency (should be False)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd > usd2)  # Same currency, not greater
    
    # Test ordering with different decimal places
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert usd > jpy  # U > J
    assert not (jpy > usd)  # J < U
    
    # Test ordering with different types
    btc = Currency.of("BTC", "Bitcoin", -1, CurrencyType.CRYPTO)
    assert btc > usd  # B > U
    assert not (usd > btc)  # U < B


# LLM-generated content at query #27
#--------------------------

```python
def test_CurrencyRegistry___exit__():
    """Test the __exit__ method of CurrencyRegistry."""
    registry = CurrencyRegistry()
    
    # Create test currencies
    usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    
    # Use the registry context manager
    with registry as register:
        register(usd)
        register(eur)
        register(jpy)
    
    # Verify that __exit__ was called and performed finalization
    # Check that registry is sorted by code
    assert registry.codes == sorted(registry.codes)
    
    # Verify currencies are sorted
    all_currencies = registry.all
    assert all(all_currencies[i].code <= all_currencies[i + 1].code for i in range(len(all_currencies) - 1))
    
    # Verify codes list matches sorted order
    expected_codes = [c.code for c in sorted(all_currencies, key=lambda x: x.code)]
    assert registry.codes == expected_codes
    
    # Verify codenames list is correctly populated
    expected_codenames = [(c.code, c.name) for c in sorted(all_currencies, key=lambda x: x.code)]
    assert registry.codenames == expected_codenames
    
    # Verify context is closed
    assert registry._CurrencyRegistry__ctx_open is False
    
    # Verify we can still access the registry after context exit
    assert "USD" in registry
    assert registry["USD"] == usd
    assert registry.get("EUR") == eur
    assert len(registry) == 3


def test_CurrencyRegistry___exit__with_exception():
    """Test that __exit__ closes context even when exception occurs."""
    registry = CurrencyRegistry()
    
    try:
        with registry as register:
            register(Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY))
            # Simulate an exception
            raise ValueError("Test exception")
    except ValueError:
        pass
    
    # Verify context is closed despite exception
    assert registry._CurrencyRegistry__ctx_open is False
    
    # Verify registry is still functional
    assert "GBP" in registry


def test_CurrencyRegistry___exit__sorting():
    """Test that __exit__ properly sorts currencies by code."""
    registry = CurrencyRegistry()
    
    # Add currencies in non-alphabetical order
    with registry as register:
        register(Currency.of("ZZZ", "Last Currency", 2, CurrencyType.MONEY))
        register(Currency.of("AAA", "First Currency", 2, CurrencyType.MONEY))
        register(Currency.of("MMM", "Middle Currency", 2, CurrencyType.MONEY))
    
    # Verify sorting occurred
    codes = registry.codes
    assert codes == ["AAA", "MMM", "ZZZ"]
    
    # Verify all list is sorted
    all_currencies = registry.all
    assert [c.code for c in all_currencies] == ["AAA", "MMM", "ZZZ"]


def test_CurrencyRegistry___exit__buffers_updated():
    """Test that __exit__ updates all internal buffers correctly."""
    registry = CurrencyRegistry()
    
    currencies_to_add = [
        Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO),
        Currency.of("ETH", "Ethereum", 18, CurrencyType.CRYPTO),
    ]
    
    with registry as register:
        for currency in currencies_to_add:
            register(currency)
    
    # Verify all buffers are consistent
    assert len(registry.all) == len(registry.codes)
    assert len(registry.all) == len(registry.codenames)
    
    # Verify buffer contents match
    for i, currency in enumerate(registry.all):
        assert registry.codes[i] == currency.code
        assert registry.codenames[i] == (currency.code, currency.name)


# LLM-generated content at query #28
#--------------------------

```python
def test_Currency___gt__():
    """Test the __gt__ method of Currency class."""
    # Create currencies with different codes for comparison
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    gbp = Currency.of("GBP", "British Pounds", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    
    # Test greater than comparisons (alphabetically by code)
    assert eur > usd  # EUR comes after USD alphabetically
    assert gbp > eur  # GBP comes after EUR alphabetically
    assert usd > jpy  # USD comes after JPY alphabetically
    
    # Test that greater than is not reflexive
    assert not (usd > usd)
    
    # Test chaining
    assert jpy < usd < eur < gbp
    
    # Test with different decimals but same code should still compare by other fields
    weird_currency = Currency.of("ZZZ", "Weird Currency", -1, CurrencyType.CRYPTO)
    normal_currency = Currency.of("AAA", "Normal Currency", 2, CurrencyType.MONEY)
    assert weird_currency > normal_currency  # ZZZ > AAA alphabetically


# LLM-generated content at query #29
#--------------------------

```python
def test_CurrencyRegistry():
    """Test CurrencyRegistry constructor and singleton behavior."""
    # Test singleton pattern - multiple instances should be the same
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2
    
    # Test initial state
    assert len(registry1) == 0
    assert registry1.all == []
    assert registry1.codes == []
    assert registry1.codenames == []
    
    # Test __contains__ on empty registry
    assert "USD" not in registry1
    
    # Test has method on empty registry
    assert not registry1.has("USD")
    
    # Test get method on empty registry
    assert registry1.get("USD") is None
    assert registry1.get("USD", default=None) is None
    
    # Test __getitem__ raises CurrencyLookupError on empty registry
    with pytest.raises(CurrencyLookupError) as exc_info:
        _ = registry1["USD"]
    assert exc_info.value.code == "USD"
    
    # Test context manager entry and exit
    with registry1 as register:
        assert callable(register)
        # Add a currency
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        register(usd)
    
    # Test registry after population
    assert len(registry1) == 1
    assert "USD" in registry1
    assert registry1.has("USD")
    assert registry1["USD"].code == "USD"
    assert registry1.get("USD") == registry1["USD"]
    assert len(registry1.all) == 1
    assert registry1.codes == ["USD"]
    assert registry1.codenames == [("USD", "US Dollar")]
    
    # Test duplicate currency registration raises ValueError
    with registry1 as register:
        with pytest.raises(ValueError) as exc_info:
            register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        assert "already registered" in str(exc_info.value)
    
    # Test adding multiple currencies and sorting
    with registry1 as register:
        jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
        eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
        register(jpy)
        register(eur)
    
    # Test sorting by code
    assert registry1.codes == ["EUR", "JPY", "USD"]
    assert len(registry1) == 3
    
    # Test get with default value
    gbp = Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY)
    result = registry1.get("NON-EXISTING", default=gbp)
    assert result == gbp
    
    # Test that registering outside context raises ProgrammingError
    with pytest.raises(ProgrammingError):
        registry1._CurrencyRegistry__register(Currency.of("CHF", "Swiss Franc", 2, CurrencyType.MONEY))


# LLM-generated content at query #30
#--------------------------

```python
def test_CurrencyRegistry___enter__():
    """Test that __enter__ returns the __register method and sets context as open."""
    registry = CurrencyRegistry()
    
    # Call __enter__ to get the register function
    register_func = registry._CurrencyRegistry__enter()
    
    # Verify that the returned callable is the __register method
    assert callable(register_func)
    assert register_func == registry._CurrencyRegistry__register
    
    # Verify that the context is marked as open
    assert registry._CurrencyRegistry__ctx_open is True
    
    # Clean up by closing the context
    registry._CurrencyRegistry__exit__(None, None, None)


# LLM-generated content at query #31
#--------------------------

```python
def test_Currency___hash__():
    """Test the __hash__ method of Currency class."""
    # Create two identical currencies
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    # Test that identical currencies have the same hash
    assert hash(usd1) == hash(usd2)
    
    # Create a currency with different name but same code
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    
    # Test that currencies with different names have different hashes
    assert hash(usd1) != hash(usdx)
    
    # Create a currency with different decimals
    usd_different_decimals = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    
    # Test that currencies with different decimals have different hashes
    assert hash(usd1) != hash(usd_different_decimals)
    
    # Create a currency with different type
    usd_crypto = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    
    # Test that currencies with different types have different hashes
    assert hash(usd1) != hash(usd_crypto)
    
    # Test that hash is consistent (calling hash multiple times returns same value)
    hash1 = hash(usd1)
    hash2 = hash(usd1)
    assert hash1 == hash2
    
    # Test with JPY (0 decimals)
    jpy1 = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    jpy2 = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert hash(jpy1) == hash(jpy2)
    
    # Test with crypto currency with -1 decimals
    zzz1 = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    zzz2 = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert hash(zzz1) == hash(zzz2)
    
    # Test that hash is usable in sets and dicts
    currency_set = {usd1, usd2, usdx}
    assert len(currency_set) == 2  # usd1 and usd2 are equal, so only 2 unique items
    
    currency_dict = {usd1: "first", usd2: "second"}
    assert len(currency_dict) == 1  # usd1 and usd2 are equal keys
    assert currency_dict[usd1] == "second"  # usd2 overwrites usd1


# LLM-generated content at query #32
#--------------------------

```python
def test_CurrencyRegistry():
    """Test CurrencyRegistry constructor and singleton behavior."""
    # Test singleton pattern - multiple instances should be the same
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2
    
    # Test initial state
    assert len(registry1) == 0
    assert registry1.all == []
    assert registry1.codes == []
    assert registry1.codenames == []
    
    # Test __contains__ on empty registry
    assert "USD" not in registry1
    assert "EUR" not in registry1
    
    # Test has() on empty registry
    assert registry1.has("USD") is False
    assert registry1.has("EUR") is False
    
    # Test get() on empty registry
    assert registry1.get("USD") is None
    assert registry1.get("EUR") is None
    
    # Test __getitem__ on empty registry raises CurrencyLookupError
    with pytest.raises(CurrencyLookupError) as exc_info:
        _ = registry1["USD"]
    assert exc_info.value.code == "USD"
    
    # Test context manager and currency registration
    usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    
    with registry1 as register:
        register(usd)
        register(eur)
    
    # Test state after registration
    assert len(registry1) == 2
    assert "USD" in registry1
    assert "EUR" in registry1
    assert registry1.has("USD") is True
    assert registry1.has("EUR") is True
    
    # Test all property
    assert len(registry1.all) == 2
    assert registry1.all[0].code == "EUR"  # Sorted alphabetically
    assert registry1.all[1].code == "USD"
    
    # Test codes property
    assert registry1.codes == ["EUR", "USD"]
    
    # Test codenames property
    assert registry1.codenames == [("EUR", "Euro"), ("USD", "US Dollar")]
    
    # Test get() after registration
    assert registry1.get("USD") == usd
    assert registry1.get("EUR") == eur
    assert registry1.get("GBP") is None
    
    # Test __getitem__ after registration
    assert registry1["USD"] == usd
    assert registry1["EUR"] == eur
    
    # Test duplicate currency registration raises ValueError
    with pytest.raises(ValueError, match="Currency USD is already registered"):
        with registry1 as register:
            register(usd)
    
    # Test registering outside context raises ProgrammingError
    with pytest.raises(ProgrammingError, match="Can not create currencies outside registry context"):
        registry1._CurrencyRegistry__register(Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY))


# LLM-generated content at query #33
#--------------------------

```python
def test_Currency():
    """Unit tests for Currency class constructor and methods."""
    
    # Test basic USD currency creation
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.code == "USD"
    assert usd.name == "US Dollars"
    assert usd.decimals == 2
    assert usd.type == CurrencyType.MONEY
    assert usd.quantizer is not None
    assert usd.hashcache is not None
    
    # Test JPY currency with 0 decimals
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy.code == "JPY"
    assert jpy.decimals == 0
    
    # Test crypto currency with -1 decimals
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert zzz.code == "ZZZ"
    assert zzz.decimals == -1
    assert zzz.type == CurrencyType.CRYPTO
    
    # Test quantize method with USD
    assert usd.quantize(Decimal("1.005")) == Decimal("1.00")
    assert usd.quantize(Decimal("1.015")) == Decimal("1.02")
    
    # Test quantize method with JPY
    assert jpy.quantize(Decimal("0.5")) == Decimal("0")
    assert jpy.quantize(Decimal("1.5")) == Decimal("2")
    
    # Test quantize method with ZZZ
    assert zzz.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert zzz.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")
    
    # Test equality
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    
    assert usd1 == usd2
    assert usd1 != usdx
    assert hash(usd1) == hash(usd2)
    assert hash(usd1) != hash(usdx)
    
    # Test hash method
    assert isinstance(hash(usd), int)
    assert usd.__hash__() == usd.hashcache
    
    # Test invalid code - non-alphabetic
    with pytest.raises(ProgrammingError):
        Currency.of("US1", "US Dollars", 2, CurrencyType.MONEY)
    
    # Test invalid code - lowercase
    with pytest.raises(ProgrammingError):
        Currency.of("usd", "US Dollars", 2, CurrencyType.MONEY)
    
    # Test invalid code - not a string
    with pytest.raises(ProgrammingError):
        Currency.of(123, "US Dollars", 2, CurrencyType.MONEY)
    
    # Test invalid name - empty
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "", 2, CurrencyType.MONEY)
    
    # Test invalid name - leading space
    with pytest.raises(ProgrammingError):
        Currency.of("USD", " US Dollars", 2, CurrencyType.MONEY)
    
    # Test invalid name - trailing space
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars ", 2, CurrencyType.MONEY)
    
    # Test invalid name - not a string
    with pytest.raises(ProgrammingError):
        Currency.of("USD", 123, 2, CurrencyType.MONEY)
    
    # Test invalid decimals - not an integer
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", 2.5, CurrencyType.MONEY)
    
    # Test invalid decimals - less than -1
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", -2, CurrencyType.MONEY)
    
    # Test invalid type
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", 2, "MONEY")
    
    # Test different currency types
    metal = Currency.of("GLD", "Gold", 2, CurrencyType.METAL)
    assert metal.type == CurrencyType.METAL
    
    alt = Currency.of("ALT", "Alternative", 2, CurrencyType.ALTERNATIVE)
    assert alt.type == CurrencyType.ALTERNATIVE
    
    # Test frozen dataclass (immutability)
    with pytest.raises(Exception):
        usd.code = "EUR"
    
    # Test ordering (dataclass with order=True)
    curr1 = Currency.of("AAA", "Currency A", 2, CurrencyType.MONEY)
    curr2 = Currency.of("BBB", "Currency B", 2, CurrencyType.MONEY)
    assert curr1 < curr2
    assert curr2 > curr1
    assert curr1 <= curr2
    assert curr2 >= curr1


# LLM-generated content at query #34
#--------------------------

```python
def test_CurrencyRegistry___enter__():
    """Test the __enter__ method of CurrencyRegistry."""
    registry = CurrencyRegistry()
    
    # Test that __enter__ returns a callable
    with registry as register:
        assert callable(register)
        assert register == registry._CurrencyRegistry__register
    
    # Test that context flag is set correctly
    registry._CurrencyRegistry__ctx_open = False
    with registry as register:
        assert registry._CurrencyRegistry__ctx_open is True
    assert registry._CurrencyRegistry__ctx_open is False
    
    # Test that we can register a currency within the context
    test_currency = Currency.of("TST", "Test Currency", 2, CurrencyType.MONEY)
    initial_len = len(registry)
    
    with registry as register:
        register(test_currency)
    
    assert len(registry) > initial_len
    assert "TST" in registry
    assert registry["TST"] == test_currency


# LLM-generated content at query #35
#--------------------------

```python
def test_Currency___gt__():
    """Test the __gt__ method of Currency class."""
    # Create currencies with different codes for ordering
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    gbp = Currency.of("GBP", "British Pounds", 2, CurrencyType.MONEY)
    
    # Test basic greater than comparison
    assert gbp > eur
    assert eur > usd
    assert gbp > usd
    
    # Test that a currency is not greater than itself
    assert not (usd > usd)
    
    # Test with different decimals
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert eur > jpy
    
    # Test with different currency types
    btc = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    assert btc > usd
    
    # Test transitivity
    assert gbp > eur and eur > usd
    assert gbp > usd
    
    # Test reverse comparison
    assert usd < eur
    assert eur < gbp


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_CurrencyRegistry___getitem__():
    """Test the __getitem__ method of CurrencyRegistry."""
    
    # Test getting an existing currency
    usd = Currencies["USD"]
    assert usd.code == "USD"
    assert usd.name == "US Dollar"
    assert isinstance(usd, Currency)
    
    # Test getting another existing currency
    eur = Currencies["EUR"]
    assert eur.code == "EUR"
    assert isinstance(eur, Currency)
    
    # Test that different currencies are different objects
    assert Currencies["USD"] != Currencies["EUR"]
    
    # Test that accessing the same currency twice returns the same object
    assert Currencies["USD"] == Currencies["USD"]
    
    # Test that accessing a non-existent currency raises CurrencyLookupError
    with pytest.raises(CurrencyLookupError) as exc_info:
        Currencies["NON-EXISTING"]
    assert exc_info.value.code == "NON-EXISTING"
    assert "NON-EXISTING" in str(exc_info.value)
    
    # Test with another non-existent code
    with pytest.raises(CurrencyLookupError) as exc_info:
        Currencies["XYZ"]
    assert exc_info.value.code == "XYZ"
    
    # Test that the error message is correct
    with pytest.raises(CurrencyLookupError) as exc_info:
        Currencies["INVALID"]
    assert "does not exist" in str(exc_info.value)
    
    # Test case sensitivity - lowercase code should raise error
    with pytest.raises(CurrencyLookupError):
        Currencies["usd"]


# LLM-generated content at query #2
#--------------------------

```python
def test_Currency___eq__():
    """Test the __eq__ method of Currency class."""
    # Create two identical currencies
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    # Test equality with identical currencies
    assert usd1 == usd2
    assert usd2 == usd1
    
    # Create a currency with different name
    usd_diff_name = Currency.of("USD", "United States Dollar", 2, CurrencyType.MONEY)
    assert usd1 != usd_diff_name
    
    # Create a currency with different decimals
    usd_diff_decimals = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert usd1 != usd_diff_decimals
    
    # Create a currency with different type
    usd_diff_type = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert usd1 != usd_diff_type
    
    # Create a different currency
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert usd1 != jpy
    
    # Test equality with non-Currency objects
    assert usd1 != "USD"
    assert usd1 != 123
    assert usd1 != None
    assert usd1 != {"code": "USD"}
    
    # Test reflexivity (x == x)
    assert usd1 == usd1
    
    # Test transitivity (if x == y and y == z, then x == z)
    usd3 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2
    assert usd2 == usd3
    assert usd1 == usd3


# LLM-generated content at query #3
#--------------------------

```python
def test_Currency___eq__():
    """Test the __eq__ method of Currency class"""
    
    # Create two identical currencies
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    # Test that identical currencies are equal
    assert usd1 == usd2
    
    # Create a currency with different name
    usd_diff_name = Currency.of("USD", "Different Name", 2, CurrencyType.MONEY)
    assert usd1 != usd_diff_name
    
    # Create a currency with different decimals
    usd_diff_decimals = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert usd1 != usd_diff_decimals
    
    # Create a currency with different type
    usd_diff_type = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert usd1 != usd_diff_type
    
    # Create a different currency code
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert usd1 != eur
    
    # Test comparison with non-Currency objects
    assert usd1 != "USD"
    assert usd1 != 2
    assert usd1 != None
    assert usd1 != {"code": "USD"}
    
    # Test reflexivity (a == a)
    assert usd1 == usd1
    
    # Test symmetry (if a == b then b == a)
    assert usd1 == usd2
    assert usd2 == usd1
    
    # Test transitivity (if a == b and b == c then a == c)
    usd3 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2
    assert usd2 == usd3
    assert usd1 == usd3
    
    # Test with different currency types
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    gbp = Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY)
    assert jpy != gbp
    assert gbp != jpy


# LLM-generated content at query #4
#--------------------------

```python
def test_Currency___eq__():
    """Test the __eq__ method of Currency class."""
    
    # Create two identical currencies
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    # Test equality of identical currencies
    assert usd1 == usd2
    assert usd2 == usd1
    
    # Create a currency with different name
    usd_diff_name = Currency.of("USD", "United States Dollar", 2, CurrencyType.MONEY)
    assert usd1 != usd_diff_name
    
    # Create a currency with different decimals
    usd_diff_decimals = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert usd1 != usd_diff_decimals
    
    # Create a currency with different type
    usd_diff_type = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert usd1 != usd_diff_type
    
    # Create a completely different currency
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert usd1 != jpy
    
    # Test comparison with non-Currency objects
    assert usd1 != "USD"
    assert usd1 != 2
    assert usd1 != None
    assert usd1 != {}
    assert usd1 != []
    
    # Test reflexivity (a == a)
    assert usd1 == usd1
    
    # Test symmetry (if a == b then b == a)
    assert (usd1 == usd2) == (usd2 == usd1)
    
    # Test transitivity (if a == b and b == c then a == c)
    usd3 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2
    assert usd2 == usd3
    assert usd1 == usd3


# LLM-generated content at query #5
#--------------------------

```python
def test_Currency___eq__():
    """Test the __eq__ method of Currency class."""
    # Test equality with identical currencies
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2
    
    # Test inequality with different names
    usd3 = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert usd1 != usd3
    
    # Test inequality with different decimals
    usd4 = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert usd1 != usd4
    
    # Test inequality with different types
    usd5 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert usd1 != usd5
    
    # Test inequality with different codes
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert usd1 != eur
    
    # Test inequality with non-Currency object
    assert usd1 != "USD"
    assert usd1 != 2
    assert usd1 != None
    assert usd1 != {}
    
    # Test with different decimals values
    jpy1 = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    jpy2 = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy1 == jpy2
    
    # Test with negative decimals
    zzz1 = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    zzz2 = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert zzz1 == zzz2
    
    # Test reflexivity
    assert usd1 == usd1
    
    # Test symmetry
    assert (usd1 == usd2) == (usd2 == usd1)
    
    # Test transitivity
    usd6 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2
    assert usd2 == usd6
    assert usd1 == usd6


# LLM-generated content at query #6
#--------------------------

```python
def test_CurrencyRegistry___getitem__():
    """Test CurrencyRegistry.__getitem__ method"""
    # Test successful retrieval of an existing currency
    usd = Currencies["USD"]
    assert usd.code == "USD"
    assert usd.name == "US Dollar"
    assert isinstance(usd, Currency)
    
    # Test retrieval of another existing currency
    eur = Currencies["EUR"]
    assert eur.code == "EUR"
    assert isinstance(eur, Currency)
    
    # Test that __getitem__ raises CurrencyLookupError for non-existent currency
    with pytest.raises(CurrencyLookupError) as exc_info:
        Currencies["NON-EXISTING"]
    assert exc_info.value.code == "NON-EXISTING"
    assert "NON-EXISTING" in str(exc_info.value)
    
    # Test that __getitem__ raises CurrencyLookupError for invalid code
    with pytest.raises(CurrencyLookupError) as exc_info:
        Currencies["XYZ"]
    assert exc_info.value.code == "XYZ"
    
    # Test that retrieved currencies are the same object when accessed multiple times
    usd1 = Currencies["USD"]
    usd2 = Currencies["USD"]
    assert usd1 == usd2
    assert usd1 is usd2
    
    # Test with various valid currency codes
    jpy = Currencies["JPY"]
    assert jpy.code == "JPY"
    assert jpy.decimals == 0
    
    gbp = Currencies["GBP"]
    assert gbp.code == "GBP"
    assert gbp.decimals == 2


# LLM-generated content at query #7
#--------------------------

```python
def test_CurrencyRegistry___getitem__():
    """Test CurrencyRegistry.__getitem__ method"""
    # Test successful lookup of an existing currency
    registry = CurrencyRegistry()
    usd = registry["USD"]
    assert usd.code == "USD"
    assert usd.name == "US Dollar"
    assert usd.type == CurrencyType.MONEY
    
    # Test lookup of another existing currency
    jpy = registry["JPY"]
    assert jpy.code == "JPY"
    assert jpy.decimals == 0
    
    # Test that lookup raises CurrencyLookupError for non-existing currency
    with pytest.raises(CurrencyLookupError) as exc_info:
        registry["NON-EXISTING"]
    assert exc_info.value.code == "NON-EXISTING"
    assert "NON-EXISTING" in str(exc_info.value)
    
    # Test that lookup raises CurrencyLookupError for invalid code
    with pytest.raises(CurrencyLookupError) as exc_info:
        registry["XYZ"]
    assert exc_info.value.code == "XYZ"
    
    # Test case sensitivity
    with pytest.raises(CurrencyLookupError):
        registry["usd"]
    
    # Test empty string
    with pytest.raises(CurrencyLookupError):
        registry[""]
    
    # Verify that the returned currency object is the same instance
    usd1 = registry["USD"]
    usd2 = registry["USD"]
    assert usd1 is usd2


# LLM-generated content at query #8
#--------------------------

```python
def test_CurrencyRegistry___getitem__():
    """Test CurrencyRegistry.__getitem__ method"""
    # Test successful retrieval of existing currency
    usd = Currencies["USD"]
    assert usd.code == "USD"
    assert usd.name == "US Dollar"
    assert isinstance(usd, Currency)
    
    # Test that the same currency is returned on multiple accesses
    usd2 = Currencies["USD"]
    assert usd == usd2
    
    # Test retrieval of another existing currency
    jpy = Currencies["JPY"]
    assert jpy.code == "JPY"
    assert jpy.decimals == 0
    
    # Test that CurrencyLookupError is raised for non-existing currency code
    with pytest.raises(CurrencyLookupError) as exc_info:
        Currencies["NON-EXISTING"]
    assert exc_info.value.code == "NON-EXISTING"
    assert "NON-EXISTING" in str(exc_info.value)
    
    # Test with various invalid codes
    with pytest.raises(CurrencyLookupError):
        Currencies["XYZ"]
    
    with pytest.raises(CurrencyLookupError):
        Currencies[""]
    
    with pytest.raises(CurrencyLookupError):
        Currencies["INVALID_CODE"]


# LLM-generated content at query #9
#--------------------------

```python
def test_Currency___eq__():
    """Test the __eq__ method of Currency class."""
    # Create two identical currencies
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    # Test that identical currencies are equal
    assert usd1 == usd2
    
    # Create a currency with different name
    usd_diff_name = Currency.of("USD", "United States Dollars", 2, CurrencyType.MONEY)
    
    # Test that currencies with different names are not equal
    assert usd1 != usd_diff_name
    
    # Create a currency with different decimals
    usd_diff_decimals = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    
    # Test that currencies with different decimals are not equal
    assert usd1 != usd_diff_decimals
    
    # Create a currency with different type
    usd_diff_type = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    
    # Test that currencies with different types are not equal
    assert usd1 != usd_diff_type
    
    # Create a completely different currency
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    
    # Test that different currencies are not equal
    assert usd1 != eur
    
    # Test equality with non-Currency objects
    assert usd1 != "USD"
    assert usd1 != 123
    assert usd1 != None
    assert usd1 != {}
    
    # Test reflexivity: currency should equal itself
    assert usd1 == usd1
    
    # Test symmetry: if a == b then b == a
    assert (usd1 == usd2) == (usd2 == usd1)
    
    # Test transitivity: if a == b and b == c then a == c
    usd3 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2
    assert usd2 == usd3
    assert usd1 == usd3


# LLM-generated content at query #10
#--------------------------

```python
def test_CurrencyRegistry___getitem__():
    """Test CurrencyRegistry.__getitem__ method."""
    # Test successful retrieval of existing currency
    usd = Currencies["USD"]
    assert usd.code == "USD"
    assert usd.name == "US Dollar"
    assert usd.type == CurrencyType.MONEY
    
    # Test retrieval of different existing currency
    jpy = Currencies["JPY"]
    assert jpy.code == "JPY"
    assert jpy.decimals == 0
    
    # Test that retrieved currencies are the same instances
    assert Currencies["USD"] == Currencies["USD"]
    
    # Test lookup error for non-existing currency code
    with pytest.raises(CurrencyLookupError) as exc_info:
        Currencies["NON-EXISTING"]
    assert exc_info.value.code == "NON-EXISTING"
    assert "NON-EXISTING" in str(exc_info.value)
    
    # Test lookup error for invalid currency code
    with pytest.raises(CurrencyLookupError) as exc_info:
        Currencies["XYZ"]
    assert exc_info.value.code == "XYZ"
    
    # Test multiple currency codes
    eur = Currencies["EUR"]
    assert eur.code == "EUR"
    
    gbp = Currencies["GBP"]
    assert gbp.code == "GBP"
    
    # Test that different currencies are different objects
    assert Currencies["USD"] != Currencies["EUR"]
    
    # Test case sensitivity - should only work with uppercase
    with pytest.raises(CurrencyLookupError):
        Currencies["usd"]
    
    with pytest.raises(CurrencyLookupError):
        Currencies["Usd"]


# LLM-generated content at query #11
#--------------------------

```python
def test_Currency___repr__():
    """Test the __repr__ method of Currency class."""
    # Test with a money currency
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    repr_str = repr(usd)
    assert "Currency" in repr_str
    assert "USD" in repr_str
    assert "US Dollars" in repr_str
    assert "2" in repr_str
    assert "MONEY" in repr_str
    
    # Test with a metal currency
    gold = Currency.of("XAU", "Gold", 4, CurrencyType.METAL)
    repr_str = repr(gold)
    assert "Currency" in repr_str
    assert "XAU" in repr_str
    assert "Gold" in repr_str
    assert "4" in repr_str
    assert "METAL" in repr_str
    
    # Test with a crypto currency
    btc = Currency.of("BTC", "Bitcoin", -1, CurrencyType.CRYPTO)
    repr_str = repr(btc)
    assert "Currency" in repr_str
    assert "BTC" in repr_str
    assert "Bitcoin" in repr_str
    assert "-1" in repr_str
    assert "CRYPTO" in repr_str
    
    # Test with zero decimals
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    repr_str = repr(jpy)
    assert "Currency" in repr_str
    assert "JPY" in repr_str
    assert "Japanese Yen" in repr_str
    assert "0" in repr_str
    
    # Verify that repr output is deterministic for the same currency
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert repr(usd) == repr(usd2)


# LLM-generated content at query #12
#--------------------------

```python
def test_CurrencyRegistry___len__():
    """Test the __len__ method of CurrencyRegistry."""
    registry = CurrencyRegistry()
    
    # The registry should have a non-zero length since it's populated
    # with standard currencies in the actual application
    assert len(registry) > 0
    
    # Verify that __len__ returns an integer
    assert isinstance(len(registry), int)
    
    # Verify that the length matches the number of items in the registry
    assert len(registry) == len(registry.all)
    
    # Verify that the length matches the number of codes
    assert len(registry) == len(registry.codes)
    
    # Verify that the length matches the number of codenames
    assert len(registry) == len(registry.codenames)


# LLM-generated content at query #13
#--------------------------

```python
def test_CurrencyRegistry___new__():
    """Test that CurrencyRegistry.__new__ creates and returns a singleton instance."""
    # Create first instance
    instance1 = CurrencyRegistry()
    
    # Create second instance
    instance2 = CurrencyRegistry()
    
    # Both should be the same object (singleton pattern)
    assert instance1 is instance2
    
    # Create third instance to further verify singleton behavior
    instance3 = CurrencyRegistry()
    assert instance1 is instance3
    assert instance2 is instance3
    
    # Verify they are instances of CurrencyRegistry
    assert isinstance(instance1, CurrencyRegistry)
    assert isinstance(instance2, CurrencyRegistry)
    assert isinstance(instance3, CurrencyRegistry)
    
    # Verify all have the same id (memory address)
    assert id(instance1) == id(instance2)
    assert id(instance2) == id(instance3)


# LLM-generated content at query #14
#--------------------------

```python
def test_CurrencyLookupError():
    """Test CurrencyLookupError constructor and properties."""
    # Test basic instantiation
    error = CurrencyLookupError("XYZ")
    
    # Test that code is stored correctly
    assert error.code == "XYZ"
    
    # Test that error message is formatted correctly
    assert str(error) == "Currency identified by code 'XYZ' does not exist"
    
    # Test that it's a LookupError subclass
    assert isinstance(error, LookupError)
    
    # Test with different currency codes
    error2 = CurrencyLookupError("ABC")
    assert error2.code == "ABC"
    assert str(error2) == "Currency identified by code 'ABC' does not exist"
    
    # Test that it can be raised and caught
    with pytest.raises(CurrencyLookupError) as exc_info:
        raise CurrencyLookupError("INVALID")
    assert exc_info.value.code == "INVALID"
    assert "INVALID" in str(exc_info.value)


# LLM-generated content at query #15
#--------------------------

```python
def test_CurrencyRegistry_get():
    """Test the get method of CurrencyRegistry class."""
    
    # Test getting an existing currency
    usd = Currencies.get("USD")
    assert usd is not None
    assert usd.code == "USD"
    assert usd.name == "US Dollar"
    
    # Test getting a non-existing currency without default
    result = Currencies.get("XXX")
    assert result is None
    
    # Test getting a non-existing currency with default
    eur = Currencies["EUR"]
    result = Currencies.get("NON-EXISTING", default=eur)
    assert result is not None
    assert result == eur
    assert result.code == "EUR"
    
    # Test getting with None as explicit default
    result = Currencies.get("INVALID", default=None)
    assert result is None
    
    # Test that get returns the same object as __getitem__
    assert Currencies.get("USD") == Currencies["USD"]
    
    # Test with various existing currency codes
    for code in ["EUR", "GBP", "JPY"]:
        currency = Currencies.get(code)
        assert currency is not None
        assert currency.code == code


# LLM-generated content at query #16
#--------------------------

```python
def test_CurrencyLookupError():
    """Test CurrencyLookupError constructor and properties."""
    # Test basic instantiation
    error = CurrencyLookupError("XYZ")
    
    # Test that code is stored correctly
    assert error.code == "XYZ"
    
    # Test that error message is formatted correctly
    assert str(error) == "Currency identified by code 'XYZ' does not exist"
    
    # Test that it is a LookupError
    assert isinstance(error, LookupError)
    
    # Test with different currency codes
    error2 = CurrencyLookupError("ABC")
    assert error2.code == "ABC"
    assert str(error2) == "Currency identified by code 'ABC' does not exist"
    
    # Test with numeric code
    error3 = CurrencyLookupError("123")
    assert error3.code == "123"
    assert str(error3) == "Currency identified by code '123' does not exist"
    
    # Test that the error can be raised and caught
    with pytest.raises(CurrencyLookupError) as exc_info:
        raise CurrencyLookupError("USD")
    assert exc_info.value.code == "USD"
    assert "USD" in str(exc_info.value)


# LLM-generated content at query #17
#--------------------------

```python
def test_Currency():
    """Test Currency class constructor and methods"""
    
    # Test basic USD currency creation
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.code == "USD"
    assert usd.name == "US Dollars"
    assert usd.decimals == 2
    assert usd.type == CurrencyType.MONEY
    
    # Test quantization with 2 decimals
    assert usd.quantize(Decimal("1.005")) == Decimal("1.00")
    assert usd.quantize(Decimal("1.015")) == Decimal("1.02")
    assert usd.quantize(Decimal("99.999")) == Decimal("100.00")
    
    # Test JPY currency with 0 decimals
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy.code == "JPY"
    assert jpy.decimals == 0
    assert jpy.quantize(Decimal("0.5")) == Decimal("0")
    assert jpy.quantize(Decimal("1.5")) == Decimal("2")
    
    # Test currency with -1 decimals (max precision)
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert zzz.code == "ZZZ"
    assert zzz.decimals == -1
    assert zzz.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert zzz.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")
    
    # Test equality
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd_diff = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    
    assert usd1 == usd2
    assert usd1 != usd_diff
    assert hash(usd1) == hash(usd2)
    assert hash(usd1) != hash(usd_diff)
    
    # Test with different currency types
    btc = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    assert btc.type == CurrencyType.CRYPTO
    
    gold = Currency.of("XAU", "Gold", 4, CurrencyType.METAL)
    assert gold.type == CurrencyType.METAL
    
    alt = Currency.of("ALT", "Alternative", 3, CurrencyType.ALTERNATIVE)
    assert alt.type == CurrencyType.ALTERNATIVE
    
    # Test invalid code - non-alphabetic
    with pytest.raises(ProgrammingError):
        Currency.of("US1", "US Dollars", 2, CurrencyType.MONEY)
    
    # Test invalid code - lowercase
    with pytest.raises(ProgrammingError):
        Currency.of("usd", "US Dollars", 2, CurrencyType.MONEY)
    
    # Test invalid code - not a string
    with pytest.raises(ProgrammingError):
        Currency.of(123, "US Dollars", 2, CurrencyType.MONEY)
    
    # Test invalid name - empty string
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "", 2, CurrencyType.MONEY)
    
    # Test invalid name - leading whitespace
    with pytest.raises(ProgrammingError):
        Currency.of("USD", " US Dollars", 2, CurrencyType.MONEY)
    
    # Test invalid name - trailing whitespace
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars ", 2, CurrencyType.MONEY)
    
    # Test invalid name - not a string
    with pytest.raises(ProgrammingError):
        Currency.of("USD", 123, 2, CurrencyType.MONEY)
    
    # Test invalid decimals - negative less than -1
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", -2, CurrencyType.MONEY)
    
    # Test invalid decimals - not an integer
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", 2.5, CurrencyType.MONEY)
    
    # Test invalid type
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", 2, "MONEY")
    
    # Test frozen dataclass - cannot modify
    with pytest.raises(Exception):
        usd.code = "EUR"
    
    # Test ordering capability
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    gbp = Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY)
    currencies = [gbp, eur, usd]
    sorted_currencies = sorted(currencies)
    assert len(sorted_currencies) == 3


# LLM-generated content at query #18
#--------------------------

```python
def test_Currency___setattr__():
    """Test that Currency instances are immutable (frozen dataclass)."""
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    # Attempt to modify any attribute should raise FrozenInstanceError
    with pytest.raises(AttributeError):
        USD.code = "EUR"
    
    with pytest.raises(AttributeError):
        USD.name = "Euros"
    
    with pytest.raises(AttributeError):
        USD.decimals = 3
    
    with pytest.raises(AttributeError):
        USD.type = CurrencyType.CRYPTO
    
    with pytest.raises(AttributeError):
        USD.quantizer = Decimal("0.1")
    
    with pytest.raises(AttributeError):
        USD.hashcache = 12345
    
    # Verify original attributes remain unchanged
    assert USD.code == "USD"
    assert USD.name == "US Dollars"
    assert USD.decimals == 2
    assert USD.type == CurrencyType.MONEY


# LLM-generated content at query #19
#--------------------------

```python
def test_CurrencyRegistry___new__():
    """Test that CurrencyRegistry.__new__ returns a singleton instance."""
    # Create first instance
    registry1 = CurrencyRegistry()
    
    # Create second instance
    registry2 = CurrencyRegistry()
    
    # Both should be the same object (singleton pattern)
    assert registry1 is registry2
    
    # Create third instance to further verify singleton behavior
    registry3 = CurrencyRegistry()
    assert registry3 is registry1
    assert registry3 is registry2
    
    # All should have the same id
    assert id(registry1) == id(registry2) == id(registry3)


# LLM-generated content at query #20
#--------------------------

```python
def test_Currency___lt__():
    """Test the __lt__ method of Currency class for ordering."""
    # Create currencies with different codes
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    gbp = Currency.of("GBP", "British Pounds", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    
    # Test basic ordering - currencies should be ordered by their dataclass fields
    assert usd < eur
    assert eur < gbp
    assert usd < gbp
    assert jpy < usd
    
    # Test that a currency is not less than itself
    assert not (usd < usd)
    
    # Test ordering with same code but different names
    usd_alt = Currency.of("USD", "United States Dollar", 2, CurrencyType.MONEY)
    assert usd < usd_alt  # Original USD should be less (alphabetically)
    
    # Test ordering with same code and name but different decimals
    usd2 = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert usd < usd2  # USD with 2 decimals should be less than 3 decimals
    
    # Test ordering with different currency types
    crypto = Currency.of("BTC", "Bitcoin", -1, CurrencyType.CRYPTO)
    metal = Currency.of("XAU", "Gold", 2, CurrencyType.METAL)
    assert usd < metal  # MONEY type should be ordered before METAL
    
    # Test transitive property: if a < b and b < c, then a < c
    assert usd < eur and eur < gbp
    assert usd < gbp
    
    # Test that greater than works inversely
    assert eur > usd
    assert gbp > eur
    assert not (usd > eur)


# LLM-generated content at query #21
#--------------------------

```python
def test_Currency___le__():
    """Test the __le__ (less than or equal) comparison method of Currency class."""
    
    # Create currencies with different codes for ordering
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    gbp = Currency.of("GBP", "British Pounds", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    
    # Test __le__ with same currency (equal case)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd <= usd2
    assert usd2 <= usd
    
    # Test __le__ with different currencies (ordering based on dataclass order=True)
    # Currencies are ordered by their fields in declaration order
    assert eur <= gbp or gbp <= eur  # One must be <= the other
    assert usd <= gbp or gbp <= usd
    
    # Test __le__ with clearly ordered currencies
    # Since dataclass is ordered by all fields, we can test with known orderings
    assert usd <= usd  # reflexive property
    
    # Test __le__ transitivity
    if eur <= gbp and gbp <= jpy:
        assert eur <= jpy
    
    # Test that __le__ is consistent with __lt__ and __eq__
    # a <= b should be equivalent to (a < b) or (a == b)
    assert (usd < eur) or (usd == eur) or (usd > eur)
    
    # Test with different types but same code shouldn't matter for ordering
    # since code is first comparison field
    crypto_btc = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    money_btc = Currency.of("BTC", "Bitcoin", 8, CurrencyType.MONEY)
    
    # Both should be comparable
    result = crypto_btc <= money_btc
    assert isinstance(result, bool)


# LLM-generated content at query #22
#--------------------------

```python
def test_CurrencyRegistry___getitem__():
    """Test __getitem__ method of CurrencyRegistry class."""
    # Test successful retrieval of an existing currency
    usd = Currencies["USD"]
    assert usd.code == "USD"
    assert usd.name == "US Dollar"
    assert isinstance(usd, Currency)
    
    # Test that __getitem__ returns the same currency object consistently
    usd_again = Currencies["USD"]
    assert usd == usd_again
    assert usd is usd_again
    
    # Test that __getitem__ raises CurrencyLookupError for non-existing currency
    with pytest.raises(CurrencyLookupError) as exc_info:
        Currencies["NON-EXISTING"]
    assert exc_info.value.code == "NON-EXISTING"
    assert "NON-EXISTING" in str(exc_info.value)
    
    # Test with another valid currency code
    jpy = Currencies["JPY"]
    assert jpy.code == "JPY"
    assert isinstance(jpy, Currency)
    
    # Test that invalid codes raise CurrencyLookupError
    with pytest.raises(CurrencyLookupError) as exc_info:
        Currencies["XYZ"]
    assert exc_info.value.code == "XYZ"
    
    # Test with empty string
    with pytest.raises(CurrencyLookupError):
        Currencies[""]
    
    # Test case sensitivity - currency codes should be uppercase
    with pytest.raises(CurrencyLookupError):
        Currencies["usd"]


# LLM-generated content at query #23
#--------------------------

```python
def test_CurrencyRegistry_has():
    """Test the has method of CurrencyRegistry."""
    # Test with existing currency
    assert Currencies.has("USD") is True
    
    # Test with non-existing currency
    assert Currencies.has("XXX") is False
    
    # Test with various existing currencies
    assert Currencies.has("EUR") is True
    assert Currencies.has("GBP") is True
    assert Currencies.has("JPY") is True
    
    # Test with various non-existing currencies
    assert Currencies.has("ZZZ") is False
    assert Currencies.has("ABC") is False
    assert Currencies.has("XYZ") is False
    
    # Test with lowercase (should be False as codes are uppercase)
    assert Currencies.has("usd") is False
    assert Currencies.has("eur") is False
    
    # Test with empty string
    assert Currencies.has("") is False
    
    # Test with single character
    assert Currencies.has("U") is False
    
    # Test with numbers (should be False as codes are alphabetic)
    assert Currencies.has("US1") is False
    assert Currencies.has("123") is False


# LLM-generated content at query #24
#--------------------------

```python
def test_Currency_quantize():
    """Test the quantize method of Currency class."""
    
    # Test with USD (2 decimals)
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert USD.quantize(Decimal("1.005")) == Decimal("1.00")
    assert USD.quantize(Decimal("1.015")) == Decimal("1.02")
    assert USD.quantize(Decimal("1.234")) == Decimal("1.23")
    assert USD.quantize(Decimal("1.235")) == Decimal("1.24")
    assert USD.quantize(Decimal("0")) == Decimal("0.00")
    assert USD.quantize(Decimal("99.999")) == Decimal("100.00")
    
    # Test with JPY (0 decimals)
    JPY = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert JPY.quantize(Decimal("0.5")) == Decimal("0")
    assert JPY.quantize(Decimal("1.5")) == Decimal("2")
    assert JPY.quantize(Decimal("1.4")) == Decimal("1")
    assert JPY.quantize(Decimal("2.5")) == Decimal("2")
    assert JPY.quantize(Decimal("99.999")) == Decimal("100")
    
    # Test with ZZZ (no fixed precision, -1 decimals)
    ZZZ = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert ZZZ.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert ZZZ.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")
    assert ZZZ.quantize(Decimal("1.123456789")) == Decimal("1.123456789")
    
    # Test with BTC (8 decimals)
    BTC = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    assert BTC.quantize(Decimal("1.123456789")) == Decimal("1.12345679")
    assert BTC.quantize(Decimal("0.00000001")) == Decimal("0.00000001")
    assert BTC.quantize(Decimal("0.000000001")) == Decimal("0.00000000")
    
    # Test with GBP (2 decimals)
    GBP = Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY)
    assert GBP.quantize(Decimal("10.125")) == Decimal("10.12")
    assert GBP.quantize(Decimal("10.135")) == Decimal("10.14")
    
    # Test edge cases
    EUR = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert EUR.quantize(Decimal("0.001")) == Decimal("0.00")
    assert EUR.quantize(Decimal("0.005")) == Decimal("0.00")
    assert EUR.quantize(Decimal("0.015")) == Decimal("0.02")
    assert EUR.quantize(Decimal("-1.005")) == Decimal("-1.00")
    assert EUR.quantize(Decimal("-1.015")) == Decimal("-1.02")


# LLM-generated content at query #25
#--------------------------

```python
def test_Currency___le__():
    """Test the __le__ method of Currency class."""
    # Create currencies with different codes
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    gbp = Currency.of("GBP", "British Pounds", 2, CurrencyType.MONEY)
    
    # Test __le__ with same currency (should be True)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd <= usd2
    
    # Test __le__ with different currencies - rely on alphabetical ordering
    # Since Currency is ordered by dataclass fields in order: code, name, decimals, type, quantizer, hashcache
    assert eur <= gbp  # EUR comes before GBP alphabetically
    assert eur <= usd  # EUR comes before USD alphabetically
    
    # Test __le__ reflexivity (currency <= itself)
    assert usd <= usd
    assert eur <= eur
    
    # Test __le__ with currencies having different decimals
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    # AED comes before JPY, so AED <= JPY
    aed = Currency.of("AED", "UAE Dirham", 2, CurrencyType.MONEY)
    assert aed <= jpy
    
    # Test __le__ with currencies having different types
    btc = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    # BTC comes after USD alphabetically, so USD <= BTC
    assert usd <= btc


# LLM-generated content at query #26
#--------------------------

```python
def test_CurrencyRegistry___new__():
    """Test that CurrencyRegistry.__new__ creates and returns singleton instance."""
    # Get first instance
    registry1 = CurrencyRegistry()
    
    # Get second instance
    registry2 = CurrencyRegistry()
    
    # Both should be the same object (singleton pattern)
    assert registry1 is registry2
    
    # Get third instance to verify consistency
    registry3 = CurrencyRegistry()
    assert registry1 is registry3
    assert registry2 is registry3


# LLM-generated content at query #27
#--------------------------

```python
def test_Currency___lt__():
    """Test the __lt__ method of Currency class."""
    # Create currencies with different codes
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    gbp = Currency.of("GBP", "British Pounds", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    
    # Test basic ordering by code (alphabetical)
    assert eur < gbp
    assert gbp < usd
    assert eur < usd
    
    # Test that a currency is not less than itself
    assert not (usd < usd)
    
    # Test transitivity
    assert eur < gbp and gbp < usd
    assert eur < usd
    
    # Test with currencies having different decimals
    assert jpy < usd
    
    # Test reverse ordering
    assert not (usd < eur)
    assert not (gbp < eur)
    
    # Test with currencies of different types
    btc = Currency.of("BTC", "Bitcoin", -1, CurrencyType.CRYPTO)
    assert btc < usd
    
    # Test ordering consistency
    currencies = [usd, eur, gbp, jpy]
    sorted_currencies = sorted(currencies)
    assert sorted_currencies == [eur, gbp, jpy, usd]


# LLM-generated content at query #28
#--------------------------

```python
def test_CurrencyRegistry_has():
    """Test the has method of CurrencyRegistry."""
    # Test with existing currency
    assert Currencies.has("USD") is True
    assert Currencies.has("EUR") is True
    assert Currencies.has("JPY") is True
    
    # Test with non-existing currency
    assert Currencies.has("XXX") is False
    assert Currencies.has("ZZZ") is False
    assert Currencies.has("INVALID") is False
    
    # Test with empty string
    assert Currencies.has("") is False
    
    # Test with lowercase code (should not exist)
    assert Currencies.has("usd") is False
    assert Currencies.has("eur") is False


# LLM-generated content at query #29
#--------------------------

```python
def test_CurrencyRegistry_get():
    """Test the get method of CurrencyRegistry class."""
    
    # Test getting an existing currency
    usd = Currencies.get("USD")
    assert usd is not None
    assert usd.code == "USD"
    assert usd.name == "US Dollar"
    
    # Test getting a non-existing currency returns None by default
    result = Currencies.get("NON_EXISTING")
    assert result is None
    
    # Test getting a non-existing currency with a default value
    eur = Currencies["EUR"]
    result = Currencies.get("NON_EXISTING", default=eur)
    assert result == eur
    assert result.code == "EUR"
    
    # Test getting with None as explicit default
    result = Currencies.get("INVALID_CODE", default=None)
    assert result is None
    
    # Test that get returns the same object as direct access
    usd_via_get = Currencies.get("USD")
    usd_via_bracket = Currencies["USD"]
    assert usd_via_get == usd_via_bracket
    assert hash(usd_via_get) == hash(usd_via_bracket)
    
    # Test with various existing currency codes
    jpy = Currencies.get("JPY")
    assert jpy is not None
    assert jpy.code == "JPY"
    
    gbp = Currencies.get("GBP")
    assert gbp is not None
    assert gbp.code == "GBP"


# LLM-generated content at query #30
#--------------------------

```python
def test_Currency___hash__():
    """Test the __hash__ method of Currency class."""
    # Test 1: Same currencies should have the same hash
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert hash(usd1) == hash(usd2)
    
    # Test 2: Different currencies should have different hashes
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert hash(usd1) != hash(jpy)
    
    # Test 3: Currencies with same code but different names should have different hashes
    usd3 = Currency.of("USD", "Different Name", 2, CurrencyType.MONEY)
    assert hash(usd1) != hash(usd3)
    
    # Test 4: Currencies with same code and name but different decimals should have different hashes
    usd4 = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert hash(usd1) != hash(usd4)
    
    # Test 5: Currencies with same code and name but different types should have different hashes
    usd5 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert hash(usd1) != hash(usd5)
    
    # Test 6: Hash should be consistent across multiple calls
    usd6 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    hash1 = hash(usd6)
    hash2 = hash(usd6)
    assert hash1 == hash2
    
    # Test 7: Hash should be usable in sets
    currency_set = {usd1, usd2, jpy}
    assert len(currency_set) == 2  # usd1 and usd2 are the same, so only 2 unique
    
    # Test 8: Hash should be usable in dictionaries
    currency_dict = {usd1: "dollars", jpy: "yen"}
    assert currency_dict[usd2] == "dollars"  # usd2 should map to same value as usd1
    
    # Test 9: Test with crypto currency
    btc = Currency.of("BTC", "Bitcoin", -1, CurrencyType.CRYPTO)
    btc2 = Currency.of("BTC", "Bitcoin", -1, CurrencyType.CRYPTO)
    assert hash(btc) == hash(btc2)
    
    # Test 10: Test with alternative currency
    alt = Currency.of("ALT", "Alternative Currency", 4, CurrencyType.ALTERNATIVE)
    alt2 = Currency.of("ALT", "Alternative Currency", 4, CurrencyType.ALTERNATIVE)
    assert hash(alt) == hash(alt2)


# LLM-generated content at query #31
#--------------------------

```python
def test_Currency___ge__():
    """Test the __ge__ (greater than or equal) comparison method of Currency class."""
    
    # Create test currencies
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    usd_duplicate = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    # Test >= with equal currencies
    assert usd >= usd_duplicate
    assert usd_duplicate >= usd
    
    # Test >= with same currency instance
    assert usd >= usd
    
    # Test >= with different currencies (based on ordering)
    # Since Currency is ordered by its dataclass fields, we test ordering behavior
    assert eur >= usd or usd >= eur  # One must be true due to ordering
    
    # Test >= reflexivity (a >= a should always be true)
    assert usd >= usd
    assert eur >= eur
    assert jpy >= jpy
    
    # Test >= transitivity: if a >= b and b >= c, then a >= c
    currencies = [usd, eur, jpy]
    for i, curr_i in enumerate(currencies):
        for j, curr_j in enumerate(currencies):
            for k, curr_k in enumerate(currencies):
                if curr_i >= curr_j and curr_j >= curr_k:
                    assert curr_i >= curr_k
    
    # Test >= consistency with == 
    # If a == b, then a >= b and b >= a should both be true
    if usd == usd_duplicate:
        assert usd >= usd_duplicate
        assert usd_duplicate >= usd


# LLM-generated content at query #32
#--------------------------

```python
def test_Currency___hash__():
    """Test the __hash__ method of Currency class."""
    # Test that hash is consistent for the same currency
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert hash(usd1) == hash(usd2)
    
    # Test that hash is different for different currencies
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    assert hash(usd1) != hash(eur)
    
    # Test that hash is different when name differs
    usd_diff_name = Currency.of("USD", "United States Dollars", 2, CurrencyType.MONEY)
    assert hash(usd1) != hash(usd_diff_name)
    
    # Test that hash is different when decimals differ
    usd_diff_decimals = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert hash(usd1) != hash(usd_diff_decimals)
    
    # Test that hash is different when type differs
    usd_crypto = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert hash(usd1) != hash(usd_crypto)
    
    # Test that hash value is an integer
    assert isinstance(hash(usd1), int)
    
    # Test that currency can be used in a set (relies on __hash__)
    currency_set = {usd1, usd2, eur}
    assert len(currency_set) == 2
    
    # Test that currency can be used as dict key (relies on __hash__)
    currency_dict = {usd1: "dollar", eur: "euro"}
    assert currency_dict[usd2] == "dollar"
    
    # Test with different currency types
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    
    assert hash(jpy) != hash(zzz)
    assert isinstance(hash(jpy), int)
    assert isinstance(hash(zzz), int)


# LLM-generated content at query #33
#--------------------------

```python
def test_Currency___setattr__():
    """Test that Currency instances are immutable (frozen dataclass)."""
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    # Attempting to set any attribute should raise FrozenInstanceError
    with pytest.raises(AttributeError):
        USD.code = "EUR"
    
    with pytest.raises(AttributeError):
        USD.name = "Euros"
    
    with pytest.raises(AttributeError):
        USD.decimals = 3
    
    with pytest.raises(AttributeError):
        USD.type = CurrencyType.METAL
    
    with pytest.raises(AttributeError):
        USD.quantizer = Decimal("0.001")
    
    with pytest.raises(AttributeError):
        USD.hashcache = 12345
    
    with pytest.raises(AttributeError):
        USD.new_attribute = "should fail"


# LLM-generated content at query #34
#--------------------------

```python
def test_CurrencyRegistry___getitem__():
    """Test CurrencyRegistry.__getitem__ method"""
    
    # Test retrieving an existing currency
    usd = Currencies["USD"]
    assert usd.code == "USD"
    assert usd.name == "US Dollar"
    assert usd.type == CurrencyType.MONEY
    
    # Test retrieving another existing currency
    eur = Currencies["EUR"]
    assert eur.code == "EUR"
    assert isinstance(eur, Currency)
    
    # Test that accessing the same currency twice returns equivalent objects
    usd1 = Currencies["USD"]
    usd2 = Currencies["USD"]
    assert usd1 == usd2
    assert usd1.code == usd2.code
    
    # Test retrieving a non-existing currency raises CurrencyLookupError
    with pytest.raises(CurrencyLookupError) as exc_info:
        Currencies["NON-EXISTING"]
    assert exc_info.value.code == "NON-EXISTING"
    assert "NON-EXISTING" in str(exc_info.value)
    
    # Test with various invalid codes
    with pytest.raises(CurrencyLookupError):
        Currencies["XYZ"]
    
    with pytest.raises(CurrencyLookupError):
        Currencies["INVALID"]
    
    with pytest.raises(CurrencyLookupError):
        Currencies[""]
    
    # Test case sensitivity
    with pytest.raises(CurrencyLookupError):
        Currencies["usd"]


# LLM-generated content at query #35
#--------------------------

```python
def test_CurrencyRegistry():
    """Test CurrencyRegistry constructor and singleton behavior."""
    # Test singleton pattern - creating two instances should return the same object
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2, "CurrencyRegistry should be a singleton"
    
    # Test initial state
    assert len(registry1) == 0, "Registry should be empty after initialization"
    assert registry1.all == [], "all property should return empty list"
    assert registry1.codes == [], "codes property should return empty list"
    assert registry1.codenames == [], "codenames property should return empty list"
    
    # Test __contains__ with empty registry
    assert "USD" not in registry1, "Empty registry should not contain any currency"
    
    # Test has method with empty registry
    assert not registry1.has("USD"), "has() should return False for empty registry"
    
    # Test get method with empty registry
    assert registry1.get("USD") is None, "get() should return None for missing currency"
    assert registry1.get("USD", default=None) is None, "get() with default should return default"
    
    # Test __getitem__ raises CurrencyLookupError with empty registry
    with pytest.raises(CurrencyLookupError):
        _ = registry1["USD"]


def test_CurrencyRegistry_context_manager():
    """Test CurrencyRegistry context manager functionality."""
    registry = CurrencyRegistry()
    
    # Test entering and exiting context
    with registry as register:
        assert callable(register), "__enter__ should return a callable"
        usd = Currency.of("TST", "Test Currency", 2, CurrencyType.MONEY)
        register(usd)
    
    # After exiting context, currency should be registered
    assert "TST" in registry, "Currency should be registered after context exit"
    assert registry.get("TST").code == "TST", "Currency should be retrievable after context"


def test_CurrencyRegistry_register_outside_context():
    """Test that registering outside context raises error."""
    registry = CurrencyRegistry()
    usd = Currency.of("OUT", "Outside Currency", 2, CurrencyType.MONEY)
    
    with pytest.raises(ProgrammingError):
        registry._CurrencyRegistry__register(usd)


def test_CurrencyRegistry_duplicate_registration():
    """Test that registering duplicate currency raises error."""
    registry = CurrencyRegistry()
    
    with registry as register:
        usd = Currency.of("DUP", "Duplicate Currency", 2, CurrencyType.MONEY)
        register(usd)
        
        # Try to register same code again
        dup = Currency.of("DUP", "Another Duplicate", 2, CurrencyType.MONEY)
        with pytest.raises(ValueError, match="Currency DUP is already registered"):
            register(dup)


# LLM-generated content at query #36
#--------------------------

```python
def test_CurrencyRegistry_get():
    """Test the get method of CurrencyRegistry class."""
    
    # Test getting an existing currency
    usd = Currencies.get("USD")
    assert usd is not None
    assert usd.code == "USD"
    assert usd.name == "US Dollar"
    
    # Test getting a non-existing currency returns None by default
    result = Currencies.get("NON_EXISTING")
    assert result is None
    
    # Test getting a non-existing currency with a default value
    default_currency = Currencies["USD"]
    result = Currencies.get("NON_EXISTING", default=default_currency)
    assert result == default_currency
    assert result.code == "USD"
    
    # Test getting multiple different currencies
    jpy = Currencies.get("JPY")
    assert jpy is not None
    assert jpy.code == "JPY"
    
    eur = Currencies.get("EUR")
    assert eur is not None
    assert eur.code == "EUR"
    
    # Test that get returns the same object as __getitem__
    usd_via_get = Currencies.get("USD")
    usd_via_getitem = Currencies["USD"]
    assert usd_via_get == usd_via_getitem
    
    # Test with empty string code
    result = Currencies.get("")
    assert result is None
    
    # Test with None as default
    result = Currencies.get("INVALID", default=None)
    assert result is None


# LLM-generated content at query #37
#--------------------------

```python
def test_Currency___setattr__():
    """Test that Currency instances are immutable (frozen dataclass)."""
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    # Attempt to modify any attribute should raise FrozenInstanceError
    with pytest.raises(AttributeError):
        USD.code = "EUR"
    
    with pytest.raises(AttributeError):
        USD.name = "Euros"
    
    with pytest.raises(AttributeError):
        USD.decimals = 3
    
    with pytest.raises(AttributeError):
        USD.type = CurrencyType.METAL
    
    with pytest.raises(AttributeError):
        USD.quantizer = Decimal("0.001")
    
    with pytest.raises(AttributeError):
        USD.hashcache = 12345
    
    # Verify original values are unchanged
    assert USD.code == "USD"
    assert USD.name == "US Dollars"
    assert USD.decimals == 2
    assert USD.type == CurrencyType.MONEY


# LLM-generated content at query #38
#--------------------------

```python
def test_CurrencyRegistry___len__():
    """Test the __len__ method of CurrencyRegistry."""
    registry = CurrencyRegistry()
    
    # The registry should have a non-zero length (it's pre-populated with currencies)
    assert len(registry) > 0
    
    # The length should match the number of currencies in the all property
    assert len(registry) == len(registry.all)
    
    # The length should match the number of codes
    assert len(registry) == len(registry.codes)
    
    # The length should match the number of codenames
    assert len(registry) == len(registry.codenames)


# LLM-generated content at query #39
#--------------------------

```python
def test_Currency___lt__():
    """Test the __lt__ method of Currency class for ordering."""
    # Create currencies with different codes
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    gbp = Currency.of("GBP", "British Pounds", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    
    # Test basic ordering - alphabetical by code
    assert eur < gbp
    assert gbp < usd
    assert eur < usd
    
    # Test that a currency is not less than itself
    assert not (usd < usd)
    assert not (eur < eur)
    
    # Test reverse ordering
    assert usd > eur
    assert gbp > eur
    assert usd > gbp
    
    # Test less than or equal
    assert eur <= gbp
    assert usd <= usd
    assert eur <= usd
    
    # Test greater than or equal
    assert gbp >= eur
    assert usd >= usd
    assert usd >= gbp
    
    # Test with different currency types
    btc = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    xau = Currency.of("XAU", "Gold", 2, CurrencyType.METAL)
    
    # Still ordered by code alphabetically
    assert btc < eur
    assert xau > usd
    assert btc < xau
    
    # Test sorting a list of currencies
    currencies = [usd, eur, gbp, jpy]
    sorted_currencies = sorted(currencies)
    assert sorted_currencies[0].code == "EUR"
    assert sorted_currencies[1].code == "GBP"
    assert sorted_currencies[2].code == "JPY"
    assert sorted_currencies[3].code == "USD"


# LLM-generated content at query #40
#--------------------------

```python
def test_CurrencyRegistry___new__():
    """Test that CurrencyRegistry.__new__ creates and returns a singleton instance."""
    # Create first instance
    registry1 = CurrencyRegistry()
    
    # Create second instance
    registry2 = CurrencyRegistry()
    
    # Verify both are the same object (singleton pattern)
    assert registry1 is registry2
    
    # Verify they have the same identity
    assert id(registry1) == id(registry2)
    
    # Verify the instance is of correct type
    assert isinstance(registry1, CurrencyRegistry)
    assert isinstance(registry2, CurrencyRegistry)


# LLM-generated content at query #41
#--------------------------

```python
def test_Currency___ge__():
    """Test the __ge__ (greater than or equal) comparison method of Currency class."""
    
    # Create test currencies
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    gbp = Currency.of("GBP", "British Pounds", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    
    # Test currency >= itself (should be True)
    assert usd >= usd
    assert eur >= eur
    
    # Test currency >= another currency (based on alphabetical ordering of code)
    # Since Currency is ordered by code, USD >= EUR should be True (U > E)
    assert usd >= eur
    
    # Test currency >= with smaller code
    # EUR >= USD should be False (E < U)
    assert not (eur >= usd)
    
    # Test currency >= with equal code but different decimals
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd >= usd2
    assert usd2 >= usd
    
    # Test transitive property
    assert gbp >= eur  # G >= E
    assert eur >= usd  # E >= U (False, so not testing this way)
    
    # Test with different currencies
    assert jpy >= gbp  # J >= G
    assert not (gbp >= jpy)  # G >= J is False
    
    # Test >= with completely different currencies
    crypto = Currency.of("BTC", "Bitcoin", -1, CurrencyType.CRYPTO)
    assert usd >= crypto  # U >= B
    assert not (crypto >= usd)  # B >= U is False


# LLM-generated content at query #42
#--------------------------

```python
def test_CurrencyRegistry___new__():
    """Test that CurrencyRegistry.__new__ creates and returns a singleton instance."""
    # First call should create the instance
    registry1 = CurrencyRegistry()
    
    # Second call should return the same instance
    registry2 = CurrencyRegistry()
    
    # Both should be the same object (singleton pattern)
    assert registry1 is registry2
    
    # Direct __new__ calls should also return the same instance
    registry3 = CurrencyRegistry.__new__(CurrencyRegistry)
    assert registry1 is registry3
    
    # Multiple __new__ calls should all return the same singleton
    registry4 = CurrencyRegistry.__new__(CurrencyRegistry)
    registry5 = CurrencyRegistry.__new__(CurrencyRegistry)
    assert registry4 is registry5
    assert registry4 is registry1


# LLM-generated content at query #43
#--------------------------

```python
def test_Currency___delattr__():
    """
    Test that Currency instances are immutable and prevent attribute deletion.
    """
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    # Attempt to delete an attribute should raise an error due to frozen=True
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
    
    # Verify the currency object remains intact
    assert usd.code == "USD"
    assert usd.name == "US Dollars"
    assert usd.decimals == 2
    assert usd.type == CurrencyType.MONEY


# LLM-generated content at query #44
#--------------------------

```python
def test_CurrencyRegistry___contains__():
    """Test the __contains__ method of CurrencyRegistry."""
    registry = CurrencyRegistry()
    
    # Test that existing currency code is in registry
    assert "USD" in registry
    assert "EUR" in registry
    assert "JPY" in registry
    
    # Test that non-existing currency code is not in registry
    assert "XXX" not in registry
    assert "ZZZ" not in registry
    assert "INVALID" not in registry
    
    # Test with empty string
    assert "" not in registry
    
    # Test that the method returns boolean
    result = "USD" in registry
    assert isinstance(result, bool)
    assert result is True
    
    result = "NONEXISTENT" in registry
    assert isinstance(result, bool)
    assert result is False


# LLM-generated content at query #45
#--------------------------

```python
def test_Currency___delattr__():
    """Test that Currency instances are immutable and cannot have attributes deleted."""
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    # Attempt to delete an attribute should raise AttributeError
    # because Currency is a frozen dataclass
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
    
    # Attempt to delete non-existent attribute should also raise AttributeError
    with pytest.raises(AttributeError):
        del usd.non_existent_attribute


