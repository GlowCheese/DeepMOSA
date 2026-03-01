####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_Currency___eq__():
    # Test equality with same currency
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2
    
    # Test equality with different currency (different code)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert not (usd1 == eur)
    
    # Test equality with different currency (different name)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 == usdx)
    
    # Test equality with different currency (different decimals)
    usd_dec3 = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert not (usd1 == usd_dec3)
    
    # Test equality with different currency (different type)
    usd_crypto = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert not (usd1 == usd_crypto)
    
    # Test equality with non-Currency object
    assert not (usd1 == "USD")
    assert not (usd1 == 123)
    assert not (usd1 == None)
    
    # Test equality with same hash but different attributes (should not happen but testing edge case)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert not (usd1 == jpy)
    
    # Test equality with subclass (should return False)
    class FakeCurrency:
        def __init__(self):
            self.hashcache = usd1.hashcache
    
    fake = FakeCurrency()
    assert not (usd1 == fake)


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_Currency___eq__():
    # Test equality with same currency objects
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2
    
    # Test equality with different currency objects (same code but different name)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 == usdx)
    
    # Test equality with different currency objects (different code)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert not (usd1 == eur)
    
    # Test equality with different currency objects (different decimals)
    usd_dec3 = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert not (usd1 == usd_dec3)
    
    # Test equality with different currency objects (different type)
    usd_crypto = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert not (usd1 == usd_crypto)
    
    # Test equality with non-Currency object
    assert not (usd1 == "USD")
    assert not (usd1 == 123)
    assert not (usd1 == None)
    
    # Test equality with same hash but different attributes (should not happen but testing)
    # Create two currencies with same hash by manipulating hashcache
    ccy1 = Currency.of("AAA", "Currency A", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("BBB", "Currency B", 2, CurrencyType.MONEY)
    
    # Manually set same hash (simulating hash collision)
    object.__setattr__(ccy1, 'hashcache', 999)
    object.__setattr__(ccy2, 'hashcache', 999)
    
    # With same hashcache, they should be considered equal
    assert ccy1 == ccy2
    
    # Test reflexive property
    assert usd1 == usd1
    
    # Test symmetric property
    assert (usd1 == usd2) == (usd2 == usd1)


# LLM-generated content at query #2
#--------------------------

```python
def test_Currency___gt__():
    # Test basic ordering by code
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    
    assert eur > usd  # EUR comes after USD alphabetically
    assert jpy > eur  # JPY comes after EUR alphabetically
    assert jpy > usd  # Transitive property
    
    # Test ordering with different names but same code (shouldn't happen in practice)
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    
    # Since they have different names, they should be ordered by name
    assert usd2 > usd1  # "UX Dollars" > "US Dollars" alphabetically
    
    # Test ordering with different decimals
    btc = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    eth = Currency.of("ETH", "Ethereum", 8, CurrencyType.CRYPTO)
    
    assert eth > btc  # ETH comes after BTC alphabetically
    
    # Test ordering with different currency types
    gold = Currency.of("XAU", "Gold", 4, CurrencyType.METAL)
    silver = Currency.of("XAG", "Silver", 4, CurrencyType.METAL)
    
    assert gold > silver  # XAU comes after XAG alphabetically
    
    # Test that ordering is consistent with __lt__
    assert not (usd > eur) == (usd < eur)
    assert not (eur > jpy) == (eur < jpy)
    
    # Test that equal currencies are not greater than each other
    usd_copy = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not usd > usd_copy
    assert not usd_copy > usd
    
    # Test ordering with edge cases
    aaa = Currency.of("AAA", "Currency A", 2, CurrencyType.MONEY)
    zzz = Currency.of("ZZZ", "Currency Z", 2, CurrencyType.MONEY)
    
    assert zzz > aaa
    assert not aaa > zzz
    
    # Test that ordering works with mixed currency types
    # The ordering should consider all fields: code, name, decimals, type, quantizer
    weird1 = Currency.of("ABC", "Test Currency", -1, CurrencyType.CRYPTO)
    weird2 = Currency.of("ABC", "Test Currency", 2, CurrencyType.CRYPTO)
    
    # Different decimals should affect ordering
    assert weird2 > weird1 or weird1 > weird2
    
    # Verify the ordering is transitive
    currencies = [
        Currency.of("AAA", "First", 0, CurrencyType.MONEY),
        Currency.of("BBB", "Second", 1, CurrencyType.MONEY),
        Currency.of("CCC", "Third", 2, CurrencyType.MONEY),
    ]
    
    assert currencies[2] > currencies[1]
    assert currencies[1] > currencies[0]
    assert currencies[2] > currencies[0]  # Transitive


# LLM-generated content at query #3
#--------------------------

```python
def test_CurrencyRegistry___len__():
    # Create a fresh registry instance
    registry = CurrencyRegistry()
    
    # Initially should be empty
    assert len(registry) == 0
    
    # Add some currencies
    with registry as register:
        register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        register(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))
    
    # Should have 2 currencies now
    assert len(registry) == 2
    
    # Add more currencies
    with registry as register:
        register(Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY))
        register(Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO))
    
    # Should have 4 currencies now
    assert len(registry) == 4
    
    # Test that adding duplicate currency doesn't change length
    try:
        with registry as register:
            register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
    except ValueError:
        pass  # Expected to fail
    
    # Length should still be 4
    assert len(registry) == 4


# LLM-generated content at query #4
#--------------------------

```python
def test_Currency():
    # Test basic currency creation
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.code == "USD"
    assert usd.name == "US Dollars"
    assert usd.decimals == 2
    assert usd.type == CurrencyType.MONEY
    assert usd.quantizer == make_quantizer(2)
    
    # Test currency with 0 decimals
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy.code == "JPY"
    assert jpy.name == "Japanese Yen"
    assert jpy.decimals == 0
    assert jpy.type == CurrencyType.MONEY
    assert jpy.quantizer == ZERO
    
    # Test currency with -1 decimals (no fixed precision)
    crypto = Currency.of("BTC", "Bitcoin", -1, CurrencyType.CRYPTO)
    assert crypto.code == "BTC"
    assert crypto.name == "Bitcoin"
    assert crypto.decimals == -1
    assert crypto.type == CurrencyType.CRYPTO
    assert crypto.quantizer == MaxPrecisionQuantizer
    
    # Test equality and hash
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2
    assert hash(usd1) == hash(usd2)
    
    # Test inequality with different name
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert usd1 != usdx
    assert hash(usd1) != hash(usdx)
    
    # Test inequality with different decimals
    usd_diff_dec = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert usd1 != usd_diff_dec
    
    # Test inequality with different type
    usd_diff_type = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert usd1 != usd_diff_type
    
    # Test quantize method
    assert usd.quantize(Decimal("1.005")) == Decimal("1.00")
    assert usd.quantize(Decimal("1.015")) == Decimal("1.02")
    assert jpy.quantize(Decimal("0.5")) == Decimal("0")
    assert jpy.quantize(Decimal("1.5")) == Decimal("2")
    assert crypto.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert crypto.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")
    
    # Test that currency is immutable (frozen dataclass)
    with pytest.raises(dataclasses.FrozenInstanceError):
        usd.code = "EUR"
    
    # Test ordering (from order=True in dataclass)
    currencies = [
        Currency.of("AAA", "Currency A", 2, CurrencyType.MONEY),
        Currency.of("BBB", "Currency B", 2, CurrencyType.MONEY),
        Currency.of("CCC", "Currency C", 2, CurrencyType.MONEY),
    ]
    sorted_currencies = sorted(currencies)
    assert [c.code for c in sorted_currencies] == ["AAA", "BBB", "CCC"]


# LLM-generated content at query #5
#--------------------------

```python
def test_Currency___repr__():
    # Test basic repr for money currency
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert repr(usd) == "Currency(code='USD', name='US Dollars', decimals=2, type=<CurrencyType.MONEY: 'Money'>, quantizer=Decimal('0.01'), hashcache=" + str(hash(usd)) + ")"
    
    # Test repr for currency with 0 decimals
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert repr(jpy) == "Currency(code='JPY', name='Japanese Yen', decimals=0, type=<CurrencyType.MONEY: 'Money'>, quantizer=Decimal('0'), hashcache=" + str(hash(jpy)) + ")"
    
    # Test repr for crypto currency with -1 decimals (max precision)
    btc = Currency.of("BTC", "Bitcoin", -1, CurrencyType.CRYPTO)
    assert repr(btc) == "Currency(code='BTC', name='Bitcoin', decimals=-1, type=<CurrencyType.CRYPTO: 'Crypto Currency'>, quantizer=Decimal('0.000000000000000000000000000000000000000000000000000000000000'), hashcache=" + str(hash(btc)) + ")"
    
    # Test repr for metal currency
    xau = Currency.of("XAU", "Gold", 4, CurrencyType.METAL)
    assert repr(xau) == "Currency(code='XAU', name='Gold', decimals=4, type=<CurrencyType.METAL: 'Precious Metal'>, quantizer=Decimal('0.0001'), hashcache=" + str(hash(xau)) + ")"
    
    # Test repr for alternative currency
    alt = Currency.of("ALT", "Alternative Currency", 3, CurrencyType.ALTERNATIVE)
    assert repr(alt) == "Currency(code='ALT', name='Alternative Currency', decimals=3, type=<CurrencyType.ALTERNATIVE: 'Alternative'>, quantizer=Decimal('0.001'), hashcache=" + str(hash(alt)) + ")"


# LLM-generated content at query #6
#--------------------------

```python
def test_Currency___repr__():
    # Test with standard currency (USD)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert repr(usd) == "Currency(code='USD', name='US Dollars', decimals=2, type=<CurrencyType.MONEY: 'Money'>, quantizer=Decimal('0.01'), hashcache=" + str(hash(usd)) + ")"
    
    # Test with zero decimal currency (JPY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert repr(jpy) == "Currency(code='JPY', name='Japanese Yen', decimals=0, type=<CurrencyType.MONEY: 'Money'>, quantizer=Decimal('0'), hashcache=" + str(hash(jpy)) + ")"
    
    # Test with negative decimal currency (crypto)
    btc = Currency.of("BTC", "Bitcoin", -1, CurrencyType.CRYPTO)
    assert repr(btc) == "Currency(code='BTC', name='Bitcoin', decimals=-1, type=<CurrencyType.CRYPTO: 'Crypto Currency'>, quantizer=MaxPrecisionQuantizer, hashcache=" + str(hash(btc)) + ")"
    
    # Test with metal currency
    xau = Currency.of("XAU", "Gold", 4, CurrencyType.METAL)
    assert repr(xau) == "Currency(code='XAU', name='Gold', decimals=4, type=<CurrencyType.METAL: 'Precious Metal'>, quantizer=Decimal('0.0001'), hashcache=" + str(hash(xau)) + ")"
    
    # Test with alternative currency
    lts = Currency.of("LTS", "Local Trade System", 2, CurrencyType.ALTERNATIVE)
    assert repr(lts) == "Currency(code='LTS', name='Local Trade System', decimals=2, type=<CurrencyType.ALTERNATIVE: 'Alternative'>, quantizer=Decimal('0.01'), hashcache=" + str(hash(lts)) + ")"


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
    assert registry1.all == []
    assert registry1.codes == []
    assert registry1.codenames == []
    
    # Test context manager registration
    with registry1 as register:
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        register(usd)
    
    # Verify registration worked
    assert len(registry1) == 1
    assert registry1.has("USD") is True
    assert "USD" in registry1
    assert registry1["USD"] == usd
    assert registry1.get("USD") == usd
    assert registry1.all == [usd]
    assert registry1.codes == ["USD"]
    assert registry1.codenames == [("USD", "US Dollar")]
    
    # Test duplicate registration
    with registry1 as register:
        try:
            register(usd)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Currency USD is already registered" in str(e)
    
    # Test registration outside context
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    try:
        registry1._CurrencyRegistry__register(eur)
        assert False, "Should have raised ProgrammingError"
    except ProgrammingError as e:
        assert "Can not create currencies outside registry context" in str(e)
    
    # Test get with default
    assert registry1.get("EUR") is None
    assert registry1.get("EUR", default=usd) == usd
    
    # Test contains and has
    assert "EUR" not in registry1
    assert registry1.has("EUR") is False
    
    # Test lookup error
    try:
        _ = registry1["EUR"]
        assert False, "Should have raised CurrencyLookupError"
    except CurrencyLookupError as e:
        assert e.code == "EUR"
        assert "Currency identified by code 'EUR' does not exist" in str(e)
    
    # Test multiple registrations maintain order
    with registry1 as register:
        eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
        jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
        register(eur)
        register(jpy)
    
    # Verify alphabetical order
    assert registry1.codes == ["EUR", "JPY", "USD"]
    assert [c.code for c in registry1.all] == ["EUR", "JPY", "USD"]
    assert registry1.codenames == [
        ("EUR", "Euro"),
        ("JPY", "Japanese Yen"),
        ("USD", "US Dollar")
    ]


# LLM-generated content at query #8
#--------------------------

```python
def test_Currency___lt__():
    # Test basic ordering by code
    usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    
    assert usd < eur  # U < E
    assert eur > usd  # E > U
    assert jpy < usd  # J < U
    
    # Test ordering with same code but different name
    usd1 = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    assert usd1 < usd2  # "US Dollar" < "US Dollars"
    assert not usd2 < usd1
    
    # Test ordering with same code and name but different decimals
    btc1 = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    btc2 = Currency.of("BTC", "Bitcoin", 10, CurrencyType.CRYPTO)
    
    assert btc1 < btc2  # 8 < 10
    assert not btc2 < btc1
    
    # Test ordering with same code, name, decimals but different type
    xau1 = Currency.of("XAU", "Gold", -1, CurrencyType.METAL)
    xau2 = Currency.of("XAU", "Gold", -1, CurrencyType.MONEY)
    
    assert xau1 < xau2  # METAL < MONEY (alphabetical: M vs P)
    assert not xau2 < xau1
    
    # Test ordering with different currency types
    gold = Currency.of("XAU", "Gold", -1, CurrencyType.METAL)
    btc = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    lts = Currency.of("LTS", "Local Time", 0, CurrencyType.ALTERNATIVE)
    
    assert btc < gold  # BTC < XAU (code comparison)
    assert lts < btc   # LTS < BTC (code comparison)
    
    # Test that equal currencies are not less than each other
    identical1 = Currency.of("XYZ", "Test Currency", 2, CurrencyType.MONEY)
    identical2 = Currency.of("XYZ", "Test Currency", 2, CurrencyType.MONEY)
    
    assert not identical1 < identical2
    assert not identical2 < identical1
    
    # Test ordering with negative decimals
    weird1 = Currency.of("ZZZ", "Weird Currency", -1, CurrencyType.CRYPTO)
    weird2 = Currency.of("ZZZ", "Weird Currency", 0, CurrencyType.CRYPTO)
    
    assert weird1 < weird2  # -1 < 0
    assert not weird2 < weird1


# LLM-generated content at query #9
#--------------------------

```python
def test_CurrencyRegistry_get():
    # Test getting existing currency
    usd = Currencies.get("USD")
    assert usd is not None
    assert usd.code == "USD"
    assert usd.name == "US Dollar"
    assert usd.type == CurrencyType.MONEY
    
    # Test getting non-existing currency without default
    non_existing = Currencies.get("NONEXISTENT")
    assert non_existing is None
    
    # Test getting non-existing currency with default
    default_currency = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    result = Currencies.get("NONEXISTENT", default=default_currency)
    assert result == default_currency
    
    # Test that get returns same as __getitem__ for existing currency
    assert Currencies.get("USD") == Currencies["USD"]
    
    # Test that get with None default returns None for non-existing
    assert Currencies.get("NONEXISTENT", default=None) is None
    
    # Test that get returns correct currency instance
    jpy = Currencies.get("JPY")
    assert jpy is not None
    assert jpy.code == "JPY"
    assert jpy.decimals == 0
    
    # Test case sensitivity - should work with uppercase
    assert Currencies.get("usd") is None
    assert Currencies.get("Usd") is None
    assert Currencies.get("USD") is not None


# LLM-generated content at query #10
#--------------------------

```python
def test_Currency___gt__():
    # Create currencies with different codes for comparison
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    
    # Test ordering by code (alphabetical)
    assert eur > usd  # EUR > USD alphabetically
    assert jpy > eur  # JPY > EUR alphabetically
    assert jpy > usd  # JPY > USD alphabetically
    
    # Test that ordering is not based on other attributes
    # Create currencies with same code but different names (shouldn't happen in practice but testing)
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    # Equal currencies should not satisfy greater than
    assert not (usd1 > usd2)
    assert not (usd2 > usd1)
    
    # Test with different currency types but same code ordering
    btc = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    xau = Currency.of("XAU", "Gold", -1, CurrencyType.METAL)
    
    # BTC > XAU alphabetically
    assert btc > xau
    
    # Test that comparison with non-Currency objects returns NotImplemented
    result = usd.__gt__("not a currency")
    assert result is NotImplemented
    
    # Test transitive property
    aud = Currency.of("AUD", "Australian Dollar", 2, CurrencyType.MONEY)
    cad = Currency.of("CAD", "Canadian Dollar", 2, CurrencyType.MONEY)
    gbp = Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY)
    
    # AUD < CAD < GBP
    assert cad > aud
    assert gbp > cad
    assert gbp > aud  # Transitive
    
    # Test reverse ordering
    assert not (aud > cad)
    assert not (aud > gbp)
    assert not (cad > gbp)


# LLM-generated content at query #11
#--------------------------

```python
def test_CurrencyLookupError():
    # Test basic instantiation with a code
    error = CurrencyLookupError("XYZ")
    assert error.code == "XYZ"
    assert str(error) == "Currency identified by code 'XYZ' does not exist"
    
    # Test that it's a subclass of LookupError
    assert isinstance(error, LookupError)
    assert isinstance(error, CurrencyLookupError)
    
    # Test with different currency code
    error2 = CurrencyLookupError("ABC")
    assert error2.code == "ABC"
    assert str(error2) == "Currency identified by code 'ABC' does not exist"
    
    # Test that the error can be raised and caught
    try:
        raise CurrencyLookupError("TEST")
    except CurrencyLookupError as e:
        assert e.code == "TEST"
        assert str(e) == "Currency identified by code 'TEST' does not exist"
    
    # Test that it can be caught as a generic LookupError
    try:
        raise CurrencyLookupError("ERR")
    except LookupError as e:
        assert isinstance(e, CurrencyLookupError)
        assert e.code == "ERR"


# LLM-generated content at query #12
#--------------------------

```python
def test_CurrencyRegistry_get():
    # Test getting an existing currency
    usd = Currencies.get("USD")
    assert usd is not None
    assert usd.code == "USD"
    assert usd.name == "US Dollar"
    assert usd.type == CurrencyType.MONEY
    
    # Test getting a non-existing currency without default
    non_existing = Currencies.get("NONEXIST")
    assert non_existing is None
    
    # Test getting a non-existing currency with default
    default_currency = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    result = Currencies.get("NONEXIST", default=default_currency)
    assert result == default_currency
    
    # Test that get returns the same as __getitem__ for existing currencies
    assert Currencies.get("USD") == Currencies["USD"]
    
    # Test that get doesn't raise for non-existing currencies (unlike __getitem__)
    try:
        Currencies.get("NONEXIST")
    except Exception:
        assert False, "get() should not raise for non-existing currencies"
    
    # Test with empty string code
    assert Currencies.get("") is None
    
    # Test with None as code (should raise TypeError when checking __contains__)
    try:
        Currencies.get(None)  # type: ignore
        assert False, "Should have raised TypeError"
    except TypeError:
        pass
    
    # Test that default parameter works with None value
    result = Currencies.get("NONEXIST", default=None)
    assert result is None


# LLM-generated content at query #13
#--------------------------

```python
def test_Currency___ge__():
    # Create currencies with different codes for comparison
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    
    # Test ordering by code (alphabetical)
    # USD > EUR because 'U' > 'E' alphabetically
    assert usd >= eur
    assert not (eur >= usd)
    
    # Test equality case
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd >= usd2
    assert usd2 >= usd
    
    # Test with different names but same code (should be equal due to same hash)
    usd_diff_name = Currency.of("USD", "United States Dollar", 2, CurrencyType.MONEY)
    # These are not equal due to different names, but ordering is by code
    assert usd >= usd_diff_name
    assert usd_diff_name >= usd
    
    # Test with different decimals but same code
    usd_diff_dec = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert usd >= usd_diff_dec
    assert usd_diff_dec >= usd
    
    # Test with different currency types but same code
    usd_crypto = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert usd >= usd_crypto
    assert usd_crypto >= usd
    
    # Test ordering with JPY (J < U alphabetically)
    assert usd >= jpy
    assert not (jpy >= usd)
    
    # Test ordering with EUR and JPY (E < J alphabetically)
    assert jpy >= eur
    assert not (eur >= jpy)
    
    # Test with same code but different attributes (should be equal due to hash comparison)
    # The __ge__ uses hash comparison, so different attributes with same hash would be equal
    # But in practice, Currency.of ensures same attributes produce same hash
    
    # Test that Currency objects are comparable with themselves
    assert usd >= usd
    
    # Test with non-Currency object (should return NotImplemented)
    result = usd.__ge__("not a currency")
    assert result is NotImplemented


# LLM-generated content at query #14
#--------------------------

```python
def test_CurrencyRegistry___new__():
    # Test that __new__ returns a singleton instance
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    
    # Both should be the same instance
    assert registry1 is registry2
    
    # Test that the instance is of correct type
    assert isinstance(registry1, CurrencyRegistry)
    
    # Test that the singleton instance is stored in the class
    assert CurrencyRegistry._CurrencyRegistry__instance is registry1
    
    # Test that the registry is properly initialized
    assert hasattr(registry1, '_CurrencyRegistry__registry')
    assert isinstance(registry1._CurrencyRegistry__registry, OrderedDict)
    assert len(registry1._CurrencyRegistry__registry) == 0
    
    # Test that other attributes are initialized
    assert hasattr(registry1, '_CurrencyRegistry__currencies')
    assert hasattr(registry1, '_CurrencyRegistry__codes')
    assert hasattr(registry1, '_CurrencyRegistry__codenames')
    assert hasattr(registry1, '_CurrencyRegistry__ctx_open')
    assert registry1._CurrencyRegistry__ctx_open is False


# LLM-generated content at query #15
#--------------------------

```python
def test_CurrencyRegistry___contains__():
    # Test with existing currency
    assert "USD" in Currencies
    
    # Test with non-existing currency
    assert "XXX" not in Currencies
    
    # Test with empty string
    assert "" not in Currencies
    
    # Test with lowercase currency code
    assert "usd" not in Currencies
    
    # Test with numeric string
    assert "123" not in Currencies
    
    # Test with special characters
    assert "USD-EUR" not in Currencies
    
    # Test that __contains__ returns bool
    result = "USD" in Currencies
    assert isinstance(result, bool)
    
    # Test that __contains__ works with getitem
    if "USD" in Currencies:
        currency = Currencies["USD"]
        assert currency.code == "USD"
    
    # Test that __contains__ matches has() method
    assert ("USD" in Currencies) == Currencies.has("USD")
    assert ("XXX" in Currencies) == Currencies.has("XXX")


# LLM-generated content at query #16
#--------------------------

```python
def test_CurrencyRegistry___new__():
    # Test that __new__ returns a singleton instance
    instance1 = CurrencyRegistry()
    instance2 = CurrencyRegistry()
    assert instance1 is instance2
    
    # Test that the instance is of correct type
    assert isinstance(instance1, CurrencyRegistry)
    
    # Test that the singleton instance is stored in the class
    assert CurrencyRegistry._CurrencyRegistry__instance is instance1
    
    # Test that the instance is properly initialized
    assert hasattr(instance1, '_CurrencyRegistry__registry')
    assert isinstance(instance1._CurrencyRegistry__registry, OrderedDict)
    assert len(instance1._CurrencyRegistry__registry) == 0
    
    # Test that the instance has all expected attributes
    assert hasattr(instance1, '_CurrencyRegistry__currencies')
    assert hasattr(instance1, '_CurrencyRegistry__codes')
    assert hasattr(instance1, '_CurrencyRegistry__codenames')
    assert hasattr(instance1, '_CurrencyRegistry__ctx_open')
    
    # Test that the instance is the same even after multiple calls
    instance3 = CurrencyRegistry()
    assert instance1 is instance3
    assert instance2 is instance3


# LLM-generated content at query #17
#--------------------------

```python
def test_CurrencyRegistry___enter__():
    registry = CurrencyRegistry()
    
    with registry as register:
        assert callable(register)
        assert registry._CurrencyRegistry__ctx_open is True
        
        currency = Currency.of("TEST", "Test Currency", 2, CurrencyType.MONEY)
        register(currency)
        
        assert "TEST" in registry
        assert registry["TEST"] == currency
    
    assert registry._CurrencyRegistry__ctx_open is False
    assert "TEST" in registry._CurrencyRegistry__registry
    assert registry._CurrencyRegistry__currencies[-1].code == "TEST"
    assert registry._CurrencyRegistry__codes[-1] == "TEST"
    assert registry._CurrencyRegistry__codenames[-1] == ("TEST", "Test Currency")


# LLM-generated content at query #18
#--------------------------

```python
def test_Currency___delattr__():
    # Create a currency instance
    currency = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    # Test that __delattr__ raises AttributeError for all attributes
    # This is expected since Currency is a frozen dataclass
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
    
    # Test that __delattr__ raises AttributeError for non-existent attributes
    with pytest.raises(AttributeError):
        del currency.non_existent_attribute
    
    # Verify that attributes are still accessible after failed deletion attempts
    assert currency.code == "USD"
    assert currency.name == "US Dollars"
    assert currency.decimals == 2
    assert currency.type == CurrencyType.MONEY
    assert isinstance(currency.quantizer, Decimal)
    assert isinstance(currency.hashcache, int)


# LLM-generated content at query #19
#--------------------------

```python
def test_Currency___delattr__():
    # Create a currency instance
    currency = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    # Test that attempting to delete any attribute raises AttributeError
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
    
    # Test that attempting to delete a non-existent attribute also raises AttributeError
    with pytest.raises(AttributeError):
        del currency.nonexistent_attribute


# LLM-generated content at query #20
#--------------------------

```python
def test_CurrencyRegistry___enter__():
    registry = CurrencyRegistry()
    
    with registry as register:
        assert callable(register)
        assert register.__name__ == "__register"
        
        currency = Currency.of("TEST", "Test Currency", 2, CurrencyType.MONEY)
        register(currency)
        
        assert "TEST" in registry
        assert registry["TEST"] == currency
    
    assert registry["TEST"].code == "TEST"
    assert registry["TEST"].name == "Test Currency"
    assert registry["TEST"].decimals == 2
    assert registry["TEST"].type == CurrencyType.MONEY
    
    with pytest.raises(ProgrammingError) as exc_info:
        currency2 = Currency.of("TEST2", "Test Currency 2", 2, CurrencyType.MONEY)
        registry.__register(currency2)
    assert "Can not create currencies outside registry context" in str(exc_info.value)
    
    with registry as register2:
        currency3 = Currency.of("AAA", "AAA Currency", 0, CurrencyType.MONEY)
        currency4 = Currency.of("ZZZ", "ZZZ Currency", 2, CurrencyType.CRYPTO)
        register2(currency4)
        register2(currency3)
    
    assert registry.codes == ["AAA", "TEST", "ZZZ"]
    assert registry.all[0].code == "AAA"
    assert registry.all[-1].code == "ZZZ"


# LLM-generated content at query #21
#--------------------------

```python
def test_Currency___setattr__():
    # Create a currency instance
    currency = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    # Test that setting attributes raises AttributeError (frozen dataclass)
    with pytest.raises(AttributeError):
        currency.code = "EUR"
    
    with pytest.raises(AttributeError):
        currency.name = "Euro"
    
    with pytest.raises(AttributeError):
        currency.decimals = 3
    
    with pytest.raises(AttributeError):
        currency.type = CurrencyType.CRYPTO
    
    with pytest.raises(AttributeError):
        currency.quantizer = Decimal("0.001")
    
    with pytest.raises(AttributeError):
        currency.hashcache = 12345
    
    # Test that setting a new attribute also raises AttributeError
    with pytest.raises(AttributeError):
        currency.new_attribute = "test"
    
    # Verify original attributes are unchanged
    assert currency.code == "USD"
    assert currency.name == "US Dollars"
    assert currency.decimals == 2
    assert currency.type == CurrencyType.MONEY
    assert currency.quantize(Decimal("1.005")) == Decimal("1.00")
    assert currency.quantize(Decimal("1.015")) == Decimal("1.02")


# LLM-generated content at query #22
#--------------------------

```python
def test_Currency___ge__():
    # Create currencies for comparison
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd_same = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd_diff_name = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    
    # Test equality (__ge__ should return True when equal)
    assert usd >= usd_same
    assert usd_same >= usd
    
    # Test with different name but same code
    assert not (usd >= usd_diff_name)
    assert not (usd_diff_name >= usd)
    
    # Test with different currencies - ordering is based on dataclass order=True
    # which orders by fields in declaration order: code, name, decimals, type, quantizer, hashcache
    # USD < EUR alphabetically by code
    assert not (usd >= eur)
    assert eur >= usd
    
    # JPY < USD (J < U alphabetically)
    assert not (jpy >= usd)
    assert usd >= jpy
    
    # Test with non-Currency object
    assert not (usd >= "USD")
    assert not (usd >= 123)
    assert not (usd >= None)
    
    # Test reflexivity
    assert usd >= usd
    
    # Test transitivity
    # Create currencies that have clear alphabetical ordering
    aaa = Currency.of("AAA", "Currency A", 2, CurrencyType.MONEY)
    bbb = Currency.of("BBB", "Currency B", 2, CurrencyType.MONEY)
    ccc = Currency.of("CCC", "Currency C", 2, CurrencyType.MONEY)
    
    if aaa >= bbb and bbb >= ccc:
        assert aaa >= ccc
    
    # Test with different currency types
    crypto = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    metal = Currency.of("XAU", "Gold", 4, CurrencyType.METAL)
    
    # BTC < XAU alphabetically (B < X)
    assert not (crypto >= metal)
    assert metal >= crypto


# LLM-generated content at query #23
#--------------------------

```python
def test_CurrencyRegistry___enter__():
    registry = CurrencyRegistry()
    
    # Test that __enter__ returns a callable
    with registry as register_func:
        assert callable(register_func)
        
        # Test that the returned callable can register currencies
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        register_func(usd)
        
        # Verify currency was registered
        assert "USD" in registry
        assert registry["USD"] == usd
        
        # Test that multiple currencies can be registered
        eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
        register_func(eur)
        
        assert "EUR" in registry
        assert registry["EUR"] == eur
    
    # Test that currencies are sorted after context exit
    assert registry.codes == ["EUR", "USD"]
    
    # Test that __enter__ can be called multiple times
    with registry as register_func:
        assert callable(register_func)
        
        # Test that existing currencies cannot be re-registered
        try:
            register_func(usd)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Currency USD is already registered" in str(e)
    
    # Test that currencies remain after multiple context entries
    assert "USD" in registry
    assert "EUR" in registry
    assert len(registry) == 2


# LLM-generated content at query #24
#--------------------------

```python
def test_Currency___le__():
    # Create currencies for comparison
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd_same = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd_different_name = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    
    # Test equality (__le__ should return True for equal objects)
    assert usd <= usd_same
    assert usd_same <= usd
    
    # Test with different currencies (should use hash comparison)
    # Since USD and EUR have different codes, USD <= EUR depends on hash comparison
    # We can't predict the exact result, but both comparisons should be consistent
    usd_le_eur = usd <= eur
    eur_le_usd = eur <= usd
    assert (usd_le_eur and not eur_le_usd) or (eur_le_usd and not usd_le_eur) or (usd_le_eur and eur_le_usd)
    
    # Test with same code but different name
    usd_le_diff_name = usd <= usd_different_name
    diff_name_le_usd = usd_different_name <= usd
    assert (usd_le_diff_name and not diff_name_le_usd) or (diff_name_le_usd and not usd_le_diff_name) or (usd_le_diff_name and diff_name_le_usd)
    
    # Test reflexivity
    assert usd <= usd
    assert eur <= eur
    assert jpy <= jpy
    
    # Test transitivity (if a <= b and b <= c then a <= c)
    # This depends on hash values, so we need to check the actual comparisons
    currencies = [usd, eur, jpy]
    for i in range(len(currencies)):
        for j in range(len(currencies)):
            for k in range(len(currencies)):
                if currencies[i] <= currencies[j] and currencies[j] <= currencies[k]:
                    assert currencies[i] <= currencies[k]
    
    # Test with non-Currency object (should return NotImplemented)
    result = usd.__le__("not a currency")
    assert result is NotImplemented
    
    # Test that __le__ uses hash comparison as defined in __eq__ and __hash__
    assert (usd <= usd_same) == (usd == usd_same)
    
    # Test with different currency types
    crypto = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    metal = Currency.of("XAU", "Gold", -1, CurrencyType.METAL)
    
    # These comparisons should work consistently
    btc_le_xau = crypto <= metal
    xau_le_btc = metal <= crypto
    assert (btc_le_xau and not xau_le_btc) or (xau_le_btc and not btc_le_xau) or (btc_le_xau and xau_le_btc)


# LLM-generated content at query #25
#--------------------------

```python
def test_CurrencyRegistry___len__():
    # Create a fresh registry instance
    registry = CurrencyRegistry()
    
    # Initially should be empty
    assert len(registry) == 0
    
    # Add currencies and check length
    with registry as register:
        register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        assert len(registry) == 1
        
        register(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))
        assert len(registry) == 2
        
        register(Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY))
        assert len(registry) == 3
    
    # After context exit, length should remain the same
    assert len(registry) == 3
    
    # Test with duplicate currency (should raise error)
    try:
        with registry as register:
            register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
    except ValueError:
        pass  # Expected behavior
    
    # Length should not change after failed duplicate addition
    assert len(registry) == 3
    
    # Test with different currency types
    with registry as register:
        register(Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO))
        assert len(registry) == 4
        
        register(Currency.of("XAU", "Gold", -1, CurrencyType.METAL))
        assert len(registry) == 5
    
    # Final length check
    assert len(registry) == 5


# LLM-generated content at query #26
#--------------------------

```python
def test_CurrencyRegistry_has():
    # Test with existing currency
    assert Currencies.has("USD") is True
    
    # Test with non-existing currency
    assert Currencies.has("NONEXISTENT") is False
    
    # Test with empty string
    assert Currencies.has("") is False
    
    # Test with lowercase currency code
    assert Currencies.has("usd") is False
    
    # Test with numeric string
    assert Currencies.has("123") is False
    
    # Test that has() returns same result as 'in' operator
    test_code = "EUR"
    assert Currencies.has(test_code) == (test_code in Currencies)
    
    # Test with special characters
    assert Currencies.has("USD-") is False
    assert Currencies.has("USD/EUR") is False
    
    # Test with whitespace
    assert Currencies.has(" USD ") is False
    assert Currencies.has("\tUSD") is False
    
    # Test that has() doesn't modify the registry
    original_length = len(Currencies)
    Currencies.has("TEST")
    assert len(Currencies) == original_length


# LLM-generated content at query #27
#--------------------------

```python
def test_CurrencyRegistry___exit__():
    # Create a fresh registry instance
    registry = CurrencyRegistry()
    
    # Create some test currencies
    currency1 = Currency.of("ZZZ", "Test Currency Z", 2, CurrencyType.MONEY)
    currency2 = Currency.of("AAA", "Test Currency A", 0, CurrencyType.CRYPTO)
    currency3 = Currency.of("MMM", "Test Currency M", -1, CurrencyType.METAL)
    
    # Enter context and register currencies in unsorted order
    with registry as register:
        register(currency3)  # MMM
        register(currency1)  # ZZZ
        register(currency2)  # AAA
    
    # After __exit__ is called, verify everything is sorted by code
    assert registry.codes == ["AAA", "MMM", "ZZZ"]
    assert [c.code for c in registry.all] == ["AAA", "MMM", "ZZZ"]
    assert [(code, name) for code, name in registry.codenames] == [
        ("AAA", "Test Currency A"),
        ("MMM", "Test Currency M"),
        ("ZZZ", "Test Currency Z")
    ]
    
    # Verify the currencies are in the registry and accessible
    assert registry["AAA"] == currency2
    assert registry["MMM"] == currency3
    assert registry["ZZZ"] == currency1
    
    # Verify context is closed after __exit__
    try:
        registry._CurrencyRegistry__register(currency1)
        assert False, "Should have raised ProgrammingError"
    except ProgrammingError as e:
        assert "Can not create currencies outside registry context" in str(e)
    
    # Test with empty registry
    empty_registry = CurrencyRegistry()
    with empty_registry:
        pass  # No currencies registered
    
    assert len(empty_registry) == 0
    assert empty_registry.codes == []
    assert empty_registry.all == []
    assert empty_registry.codenames == []


# LLM-generated content at query #28
#--------------------------

```python
def test_Currency():
    # Test basic currency creation
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert USD.code == "USD"
    assert USD.name == "US Dollars"
    assert USD.decimals == 2
    assert USD.type == CurrencyType.MONEY
    assert USD.quantizer == Decimal('0.01')
    
    # Test currency with 0 decimals
    JPY = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert JPY.code == "JPY"
    assert JPY.name == "Japanese Yen"
    assert JPY.decimals == 0
    assert JPY.type == CurrencyType.MONEY
    assert JPY.quantizer == ZERO
    
    # Test currency with negative decimals (no fixed precision)
    ZZZ = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert ZZZ.code == "ZZZ"
    assert ZZZ.name == "Some weird currency"
    assert ZZZ.decimals == -1
    assert ZZZ.type == CurrencyType.CRYPTO
    assert ZZZ.quantizer == MaxPrecisionQuantizer
    
    # Test equality and hash
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    
    assert usd1 == usd2
    assert usd1 != usdx
    assert hash(usd1) == hash(usd2)
    assert hash(usd1) != hash(usdx)
    
    # Test quantize method
    assert USD.quantize(Decimal("1.005")) == Decimal("1.00")
    assert USD.quantize(Decimal("1.015")) == Decimal("1.02")
    assert JPY.quantize(Decimal("0.5")) == Decimal("0")
    assert JPY.quantize(Decimal("1.5")) == Decimal("2")
    assert ZZZ.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert ZZZ.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")
    
    # Test different currency types
    metal = Currency.of("XAU", "Gold", 4, CurrencyType.METAL)
    assert metal.type == CurrencyType.METAL
    
    alt = Currency.of("LOC", "Local Currency", 2, CurrencyType.ALTERNATIVE)
    assert alt.type == CurrencyType.ALTERNATIVE
    
    # Test hashcache is properly set
    assert usd1.hashcache == hash(usd1)
    
    # Test that Currency is immutable (frozen dataclass)
    with pytest.raises(dataclasses.FrozenInstanceError):
        usd1.code = "EUR"


# LLM-generated content at query #29
#--------------------------

```python
def test_Currency___hash__():
    # Test that equal currencies have equal hash codes
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert hash(usd1) == hash(usd2)
    
    # Test that different currencies have different hash codes
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert hash(usd1) != hash(jpy)
    
    # Test hash consistency - same currency should return same hash on multiple calls
    assert hash(usd1) == hash(usd1)
    
    # Test that hash is pre-computed and cached
    assert usd1.hashcache == hash(usd1)
    assert usd2.hashcache == hash(usd2)
    
    # Test that different codes produce different hashes
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert hash(usd1) != hash(eur)
    
    # Test that different names produce different hashes
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert hash(usd1) != hash(usdx)
    
    # Test that different decimals produce different hashes
    usd_dec3 = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert hash(usd1) != hash(usd_dec3)
    
    # Test that different types produce different hashes
    crypto = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    metal = Currency.of("XAU", "Gold", 4, CurrencyType.METAL)
    assert hash(crypto) != hash(metal)
    
    # Test that currencies with negative decimals work correctly
    weird = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert hash(weird) == hash(weird)
    
    # Test that hash is integer
    assert isinstance(hash(usd1), int)


# LLM-generated content at query #30
#--------------------------

```python
def test_Currency___setattr__():
    # Create a currency instance
    currency = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    # Test that setting attributes raises AttributeError (frozen dataclass)
    with pytest.raises(AttributeError):
        currency.code = "EUR"
    
    with pytest.raises(AttributeError):
        currency.name = "Euro"
    
    with pytest.raises(AttributeError):
        currency.decimals = 3
    
    with pytest.raises(AttributeError):
        currency.type = CurrencyType.CRYPTO
    
    with pytest.raises(AttributeError):
        currency.quantizer = Decimal("0.001")
    
    with pytest.raises(AttributeError):
        currency.hashcache = 12345
    
    # Test that setting a new attribute raises AttributeError
    with pytest.raises(AttributeError):
        currency.new_attribute = "test"
    
    # Verify original attributes remain unchanged
    assert currency.code == "USD"
    assert currency.name == "US Dollars"
    assert currency.decimals == 2
    assert currency.type == CurrencyType.MONEY
    assert currency.quantize(Decimal("1.005")) == Decimal("1.00")


# LLM-generated content at query #31
#--------------------------

```python
def test_CurrencyRegistry___contains__():
    # Test that USD is in the registry
    assert "USD" in Currencies
    
    # Test that a non-existent currency is not in the registry
    assert "XXX" not in Currencies
    
    # Test that EUR is in the registry
    assert "EUR" in Currencies
    
    # Test that GBP is in the registry
    assert "GBP" in Currencies
    
    # Test case sensitivity - should be uppercase
    assert "usd" not in Currencies
    
    # Test empty string
    assert "" not in Currencies
    
    # Test with special characters
    assert "US$" not in Currencies
    
    # Test with numbers
    assert "123" not in Currencies
    
    # Test that JPY is in the registry
    assert "JPY" in Currencies
    
    # Test that multiple valid currencies are in the registry
    assert all(ccy in Currencies for ccy in ["USD", "EUR", "GBP", "JPY"])
    
    # Test that invalid currencies are not in the registry
    assert not any(ccy in Currencies for ccy in ["XYZ", "ABC", "123", "usd"])


# LLM-generated content at query #32
#--------------------------

```python
def test_CurrencyRegistry___contains__():
    # Test that existing currency codes return True
    assert "USD" in Currencies
    assert "EUR" in Currencies
    assert "JPY" in Currencies
    
    # Test that non-existing currency codes return False
    assert "XXX" not in Currencies
    assert "NONEXISTENT" not in Currencies
    assert "" not in Currencies
    
    # Test case sensitivity - currency codes should be uppercase
    assert "usd" not in Currencies
    assert "Eur" not in Currencies
    
    # Test with empty string
    assert "" not in Currencies
    
    # Test with special characters
    assert "USD$" not in Currencies
    assert "US-D" not in Currencies
    
    # Test that the method works with the has() method consistently
    registry = CurrencyRegistry()
    assert ("USD" in registry) == registry.has("USD")
    assert ("XXX" in registry) == registry.has("XXX")


# LLM-generated content at query #33
#--------------------------

```python
def test_CurrencyRegistry_has():
    # Test with existing currency code
    assert Currencies.has("USD") is True
    
    # Test with non-existing currency code
    assert Currencies.has("NONEXISTENT") is False
    
    # Test with empty string
    assert Currencies.has("") is False
    
    # Test with lowercase currency code (should be case-sensitive)
    assert Currencies.has("usd") is False
    
    # Test that has() returns same result as 'in' operator
    test_code = "EUR"
    assert Currencies.has(test_code) == (test_code in Currencies)
    
    # Test with special characters
    assert Currencies.has("USD$") is False
    assert Currencies.has("123") is False
    
    # Test with None (should handle gracefully or raise TypeError)
    try:
        Currencies.has(None)
        # If no exception, verify it returns False
        assert Currencies.has(None) is False
    except TypeError:
        # TypeError is also acceptable behavior
        pass


# LLM-generated content at query #34
#--------------------------

```python
def test_Currency___repr__():
    # Test USD currency repr
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert repr(USD) == "Currency(code='USD', name='US Dollars', decimals=2, type=<CurrencyType.MONEY: 'Money'>, quantizer=Decimal('0.01'), hashcache=hash((USD.code, USD.name, USD.decimals, USD.type, USD.quantizer)))"
    
    # Test JPY currency repr with 0 decimals
    JPY = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert repr(JPY) == "Currency(code='JPY', name='Japanese Yen', decimals=0, type=<CurrencyType.MONEY: 'Money'>, quantizer=Decimal('0'), hashcache=hash((JPY.code, JPY.name, JPY.decimals, JPY.type, JPY.quantizer)))"
    
    # Test crypto currency repr with -1 decimals
    ZZZ = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert repr(ZZZ) == "Currency(code='ZZZ', name='Some weird currency', decimals=-1, type=<CurrencyType.CRYPTO: 'Crypto Currency'>, quantizer=MaxPrecisionQuantizer, hashcache=hash((ZZZ.code, ZZZ.name, ZZZ.decimals, ZZZ.type, ZZZ.quantizer)))"
    
    # Test metal currency repr
    XAU = Currency.of("XAU", "Gold", 4, CurrencyType.METAL)
    assert repr(XAU) == "Currency(code='XAU', name='Gold', decimals=4, type=<CurrencyType.METAL: 'Precious Metal'>, quantizer=Decimal('0.0001'), hashcache=hash((XAU.code, XAU.name, XAU.decimals, XAU.type, XAU.quantizer)))"
    
    # Test alternative currency repr
    ALT = Currency.of("ALT", "Alternative Currency", 3, CurrencyType.ALTERNATIVE)
    assert repr(ALT) == "Currency(code='ALT', name='Alternative Currency', decimals=3, type=<CurrencyType.ALTERNATIVE: 'Alternative'>, quantizer=Decimal('0.001'), hashcache=hash((ALT.code, ALT.name, ALT.decimals, ALT.type, ALT.quantizer)))"


# LLM-generated content at query #35
#--------------------------

```python
def test_Currency_quantize():
    # Test USD with 2 decimals
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert USD.quantize(Decimal("1.005")) == Decimal("1.00")
    assert USD.quantize(Decimal("1.015")) == Decimal("1.02")
    assert USD.quantize(Decimal("0.000")) == Decimal("0.00")
    assert USD.quantize(Decimal("123.456")) == Decimal("123.46")
    assert USD.quantize(Decimal("-1.005")) == Decimal("-1.00")
    assert USD.quantize(Decimal("-1.015")) == Decimal("-1.02")
    
    # Test JPY with 0 decimals
    JPY = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert JPY.quantize(Decimal("0.5")) == Decimal("0")
    assert JPY.quantize(Decimal("1.5")) == Decimal("2")
    assert JPY.quantize(Decimal("123.456")) == Decimal("123")
    assert JPY.quantize(Decimal("-0.5")) == Decimal("0")
    assert JPY.quantize(Decimal("-1.5")) == Decimal("-2")
    
    # Test currency with -1 decimals (no fixed precision)
    ZZZ = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert ZZZ.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert ZZZ.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")
    assert ZZZ.quantize(Decimal("0.0000000000005")) == Decimal("0.000000000000")
    assert ZZZ.quantize(Decimal("-1.0000000000005")) == Decimal("-1.000000000000")
    
    # Test currency with 1 decimal
    ONE_DEC = Currency.of("ABC", "One Decimal", 1, CurrencyType.MONEY)
    assert ONE_DEC.quantize(Decimal("1.05")) == Decimal("1.0")
    assert ONE_DEC.quantize(Decimal("1.15")) == Decimal("1.2")
    assert ONE_DEC.quantize(Decimal("0.05")) == Decimal("0.0")
    
    # Test currency with 3 decimals
    THREE_DEC = Currency.of("XYZ", "Three Decimals", 3, CurrencyType.MONEY)
    assert THREE_DEC.quantize(Decimal("1.0005")) == Decimal("1.000")
    assert THREE_DEC.quantize(Decimal("1.0015")) == Decimal("1.002")
    assert THREE_DEC.quantize(Decimal("0.0005")) == Decimal("0.000")
    
    # Test exact values (no rounding needed)
    assert USD.quantize(Decimal("100.00")) == Decimal("100.00")
    assert JPY.quantize(Decimal("100")) == Decimal("100")
    assert ZZZ.quantize(Decimal("1.000000000000")) == Decimal("1.000000000000")
    
    # Test zero
    assert USD.quantize(Decimal("0")) == Decimal("0.00")
    assert JPY.quantize(Decimal("0")) == Decimal("0")
    assert ZZZ.quantize(Decimal("0")) == Decimal("0.000000000000")


# LLM-generated content at query #36
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
    
    # Test context manager registration
    with registry1 as register:
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        register(usd)
    
    # Verify registration worked
    assert len(registry1) == 1
    assert "USD" in registry1
    assert registry1.has("USD") is True
    assert registry1["USD"] == usd
    assert registry1.get("USD") == usd
    assert registry1.all == [usd]
    assert registry1.codes == ["USD"]
    assert registry1.codenames == [("USD", "US Dollar")]
    
    # Test adding multiple currencies
    with registry1 as register:
        eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
        jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
        register(eur)
        register(jpy)
    
    # Verify sorting and multiple currencies
    assert len(registry1) == 3
    assert registry1.codes == ["EUR", "JPY", "USD"]
    assert registry1.all == [eur, jpy, usd]
    
    # Test duplicate registration
    with registry1 as register:
        try:
            register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Currency USD is already registered" in str(e)
    
    # Test registration outside context
    try:
        registry1._CurrencyRegistry__register(Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY))
        assert False, "Should have raised ProgrammingError"
    except ProgrammingError as e:
        assert "Can not create currencies outside registry context" in str(e)
    
    # Test contains and has methods
    assert "EUR" in registry1
    assert registry1.has("EUR") is True
    assert "XYZ" not in registry1
    assert registry1.has("XYZ") is False
    
    # Test get with default
    assert registry1.get("EUR") == eur
    assert registry1.get("XYZ") is None
    assert registry1.get("XYZ", default=usd) == usd
    
    # Test __getitem__ with non-existent currency
    try:
        registry1["XYZ"]
        assert False, "Should have raised CurrencyLookupError"
    except CurrencyLookupError as e:
        assert e.code == "XYZ"
        assert "Currency identified by code 'XYZ' does not exist" in str(e)
    
    # Test that singleton maintains state
    assert len(registry2) == 3
    assert registry2["USD"] == usd


# LLM-generated content at query #37
#--------------------------

```python
def test_CurrencyRegistry_has():
    # Test with existing currency code
    assert Currencies.has("USD") is True
    
    # Test with non-existing currency code
    assert Currencies.has("NONEXISTENT") is False
    
    # Test with empty string
    assert Currencies.has("") is False
    
    # Test with lowercase currency code
    assert Currencies.has("usd") is False
    
    # Test with numeric string
    assert Currencies.has("123") is False
    
    # Test with special characters
    assert Currencies.has("USD$") is False
    
    # Test that has() and __contains__ give same result
    assert Currencies.has("USD") == ("USD" in Currencies)
    assert Currencies.has("NONEXISTENT") == ("NONEXISTENT" in Currencies)
    
    # Test with multiple existing currencies
    assert Currencies.has("EUR") is True
    assert Currencies.has("GBP") is True
    assert Currencies.has("JPY") is True


# LLM-generated content at query #38
#--------------------------

```python
def test_CurrencyRegistry___enter__():
    registry = CurrencyRegistry()
    
    # Test that __enter__ returns a callable
    with registry as register_func:
        assert callable(register_func)
        
        # Test that the returned callable can register currencies
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        register_func(usd)
        
        # Verify currency was registered
        assert "USD" in registry
        assert registry["USD"] == usd
    
    # Test that context is closed after exit
    assert not registry._CurrencyRegistry__ctx_open
    
    # Test that currencies are sorted after context exit
    registry2 = CurrencyRegistry()
    with registry2 as register:
        eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
        register(eur)
        gbp = Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY)
        register(gbp)
        aud = Currency.of("AUD", "Australian Dollar", 2, CurrencyType.MONEY)
        register(aud)
    
    # Verify currencies are sorted by code
    assert registry2.codes == ["AUD", "EUR", "GBP"]
    assert registry2.all[0].code == "AUD"
    assert registry2.all[1].code == "EUR"
    assert registry2.all[2].code == "GBP"


# LLM-generated content at query #39
#--------------------------

```python
def test_Currency___le__():
    # Create currencies for comparison
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd_same = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd_diff_name = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    
    # Test equality (__le__ should return True for equal objects)
    assert usd <= usd_same
    assert usd_same <= usd
    
    # Test with different name but same code
    assert not (usd <= usd_diff_name)
    assert not (usd_diff_name <= usd)
    
    # Test with different currencies - should compare based on order
    # Since Currency is ordered (order=True in dataclass), it will compare
    # fields in the order they're defined in the dataclass
    assert (usd <= jpy) == (usd.code <= jpy.code)
    assert (jpy <= eur) == (jpy.code <= eur.code)
    
    # Test with same type but different decimals
    btc = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    eth = Currency.of("ETH", "Ethereum", 18, CurrencyType.CRYPTO)
    assert (btc <= eth) == (btc.code <= eth.code)
    
    # Test with different currency types
    xau = Currency.of("XAU", "Gold", -1, CurrencyType.METAL)
    assert (usd <= xau) == (usd.code <= xau.code)
    
    # Test reflexivity
    assert usd <= usd
    assert jpy <= jpy
    assert eur <= eur
    
    # Test transitivity
    if usd <= jpy and jpy <= eur:
        assert usd <= eur
    
    # Test with non-Currency object
    assert not (usd <= "USD")
    assert not (usd <= 123)
    assert not (usd <= None)


# LLM-generated content at query #40
#--------------------------

```python
def test_Currency___delattr__():
    # Create a currency instance
    currency = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    # Test that attempting to delete any attribute raises AttributeError
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
    
    # Test that attempting to delete a non-existent attribute also raises AttributeError
    with pytest.raises(AttributeError):
        del currency.nonexistent_attribute
    
    # Verify that the currency instance is still intact after the attempts
    assert currency.code == "USD"
    assert currency.name == "US Dollars"
    assert currency.decimals == 2
    assert currency.type == CurrencyType.MONEY
    assert isinstance(currency.quantizer, Decimal)
    assert isinstance(currency.hashcache, int)


# LLM-generated content at query #41
#--------------------------

```python
def test_CurrencyRegistry___new__():
    # Test that __new__ returns a singleton instance
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2
    
    # Test that the instance is of correct type
    assert isinstance(registry1, CurrencyRegistry)
    
    # Test that the singleton instance is stored in class variable
    assert CurrencyRegistry._CurrencyRegistry__instance is registry1
    
    # Test that the registry is properly initialized
    assert hasattr(registry1, '_CurrencyRegistry__registry')
    assert isinstance(registry1._CurrencyRegistry__registry, OrderedDict)
    assert len(registry1._CurrencyRegistry__registry) == 0
    
    # Test that the instance can be used as context manager
    with registry1 as register:
        assert callable(register)
    
    # Test that the instance has expected methods
    assert hasattr(registry1, 'has')
    assert hasattr(registry1, 'get')
    assert hasattr(registry1, '__contains__')
    assert hasattr(registry1, '__getitem__')
    assert hasattr(registry1, '__len__')
    
    # Test property access
    assert isinstance(registry1.all, list)
    assert isinstance(registry1.codes, list)
    assert isinstance(registry1.codenames, list)


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_Currency___eq__():
    # Test equality with same currency
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2
    
    # Test inequality with different code
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert not (usd1 == eur)
    
    # Test inequality with different name
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 == usdx)
    
    # Test inequality with different decimals
    usd_decimals = Currency.of("USD", "US Dollars", 0, CurrencyType.MONEY)
    assert not (usd1 == usd_decimals)
    
    # Test inequality with different type
    usd_crypto = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert not (usd1 == usd_crypto)
    
    # Test equality with different instance but same attributes
    jpy1 = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    jpy2 = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy1 == jpy2
    
    # Test inequality with completely different currency
    btc = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    assert not (usd1 == btc)
    
    # Test comparison with non-Currency object
    assert not (usd1 == "USD")
    assert not (usd1 == 123)
    assert not (usd1 == None)
    
    # Test that hash equality implies object equality
    assert hash(usd1) == hash(usd2)
    assert usd1 == usd2


# LLM-generated content at query #2
#--------------------------

```python
def test_Currency___eq__():
    # Test equality with same currency
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2
    
    # Test equality with different currency objects but same attributes
    eur1 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    eur2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert eur1 == eur2
    
    # Test inequality with different code
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
    btc1 = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    btc2 = Currency.of("BTC", "Bitcoin", 8, CurrencyType.MONEY)
    assert not (btc1 == btc2)
    
    # Test equality with different currency types but same attributes
    gold1 = Currency.of("XAU", "Gold", -1, CurrencyType.METAL)
    gold2 = Currency.of("XAU", "Gold", -1, CurrencyType.METAL)
    assert gold1 == gold2
    
    # Test inequality with non-Currency object
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd == "USD")
    assert not (usd == 123)
    assert not (usd == None)
    
    # Test inequality with different quantizer due to different decimals
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (jpy == usd)
    
    # Test equality with alternative currency type
    alt1 = Currency.of("LOC", "Local Currency", 2, CurrencyType.ALTERNATIVE)
    alt2 = Currency.of("LOC", "Local Currency", 2, CurrencyType.ALTERNATIVE)
    assert alt1 == alt2


# LLM-generated content at query #3
#--------------------------

```python
def test_CurrencyRegistry___getitem__():
    # Test getting an existing currency
    registry = CurrencyRegistry()
    
    # Create a test currency and add it to registry
    with registry as register:
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        register(usd)
    
    # Test that we can retrieve the currency by code
    assert registry["USD"] == usd
    assert registry["USD"].code == "USD"
    assert registry["USD"].name == "US Dollar"
    assert registry["USD"].decimals == 2
    assert registry["USD"].type == CurrencyType.MONEY
    
    # Test that getting non-existent currency raises CurrencyLookupError
    try:
        registry["NONEXISTENT"]
        assert False, "Should have raised CurrencyLookupError"
    except CurrencyLookupError as e:
        assert e.code == "NONEXISTENT"
        assert str(e) == "Currency identified by code 'NONEXISTENT' does not exist"
    
    # Test case sensitivity - currency codes should be uppercase
    try:
        registry["usd"]
        assert False, "Should have raised CurrencyLookupError for lowercase code"
    except CurrencyLookupError as e:
        assert e.code == "usd"
    
    # Test with multiple currencies
    registry2 = CurrencyRegistry()
    with registry2 as register:
        eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
        jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
        register(eur)
        register(jpy)
    
    assert registry2["EUR"] == eur
    assert registry2["JPY"] == jpy
    assert registry2["EUR"].code == "EUR"
    assert registry2["JPY"].code == "JPY"


# LLM-generated content at query #4
#--------------------------

```python
def test_CurrencyRegistry___getitem__():
    # Test getting an existing currency
    assert Currencies["USD"].code == "USD"
    assert Currencies["USD"].name == "US Dollar"
    assert Currencies["USD"].type.name == "MONEY"
    
    # Test getting another existing currency
    assert Currencies["EUR"].code == "EUR"
    
    # Test that lookup is case-sensitive
    try:
        Currencies["usd"]
        assert False, "Should have raised CurrencyLookupError"
    except CurrencyLookupError as e:
        assert e.code == "usd"
    
    # Test getting non-existing currency raises CurrencyLookupError
    try:
        Currencies["NON-EXISTING"]
        assert False, "Should have raised CurrencyLookupError"
    except CurrencyLookupError as e:
        assert e.code == "NON-EXISTING"
        assert str(e) == "Currency identified by code 'NON-EXISTING' does not exist"
    
    # Test getting another non-existing currency
    try:
        Currencies["XYZ"]
        assert False, "Should have raised CurrencyLookupError"
    except CurrencyLookupError as e:
        assert e.code == "XYZ"
    
    # Test that returned currency is the same instance as from get() method
    assert Currencies["USD"] == Currencies.get("USD")
    
    # Test that different currencies are different objects
    assert Currencies["USD"] != Currencies["EUR"]
    
    # Test that currency has correct attributes
    usd = Currencies["USD"]
    assert isinstance(usd, Currency)
    assert usd.code == "USD"
    assert usd.name == "US Dollar"
    assert usd.decimals == 2
    assert usd.type == CurrencyType.MONEY


# LLM-generated content at query #5
#--------------------------

```python
def test_Currency___eq__():
    # Test equality with same currency objects
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2
    
    # Test inequality with different currency codes
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert not (usd1 == eur)
    
    # Test inequality with different names
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 == usdx)
    
    # Test inequality with different decimals
    usd_decimals = Currency.of("USD", "US Dollars", 0, CurrencyType.MONEY)
    assert not (usd1 == usd_decimals)
    
    # Test inequality with different currency types
    usd_crypto = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert not (usd1 == usd_crypto)
    
    # Test equality with same hashcache
    jpy1 = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    jpy2 = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy1 == jpy2
    
    # Test inequality with non-Currency object
    assert not (usd1 == "USD")
    assert not (usd1 == 123)
    assert not (usd1 == None)
    
    # Test inequality with different quantizer due to decimals
    weird1 = Currency.of("ZZZ", "Weird Currency", -1, CurrencyType.CRYPTO)
    weird2 = Currency.of("ZZZ", "Weird Currency", 2, CurrencyType.CRYPTO)
    assert not (weird1 == weird2)


# LLM-generated content at query #6
#--------------------------

```python
def test_Currency___ge__():
    # Create currencies for comparison
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd_same = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd_different_name = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    
    # Test equality (__ge__ should return True for equal objects)
    assert usd >= usd_same
    assert usd_same >= usd
    
    # Test with different name but same code - should be different
    # Since currencies are ordered by the tuple (code, name, decimals, type, quantizer)
    # USD with "US Dollars" comes before USD with "UX Dollars" alphabetically
    assert not (usd >= usd_different_name)
    assert usd_different_name >= usd
    
    # Test ordering by code
    assert eur >= usd  # "EUR" > "USD" alphabetically
    assert not (usd >= eur)
    
    # Test with JPY which has different decimals
    assert jpy >= usd  # "JPY" > "USD" alphabetically
    assert not (usd >= jpy)
    
    # Test reflexivity
    assert usd >= usd
    assert jpy >= jpy
    assert eur >= eur
    
    # Test transitivity
    # If A >= B and B >= C, then A >= C
    # Create currencies with sequential codes
    abc = Currency.of("ABC", "Test A", 2, CurrencyType.MONEY)
    def_ = Currency.of("DEF", "Test B", 2, CurrencyType.MONEY)
    ghi = Currency.of("GHI", "Test C", 2, CurrencyType.MONEY)
    
    assert ghi >= def_
    assert def_ >= abc
    assert ghi >= abc
    
    # Test with different currency types
    crypto = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    metal = Currency.of("XAU", "Gold", 4, CurrencyType.METAL)
    
    # Crypto vs Metal - ordering based on full tuple
    assert crypto >= metal or metal >= crypto
    
    # Test that comparison with non-Currency objects raises TypeError
    try:
        _ = usd >= "USD"
        assert False, "Should have raised TypeError"
    except TypeError:
        pass
    
    # Test with currencies having negative decimals
    weird = Currency.of("ZZZ", "Weird Currency", -1, CurrencyType.CRYPTO)
    assert weird >= usd  # "ZZZ" > "USD" alphabetically
    assert not (usd >= weird)


# LLM-generated content at query #7
#--------------------------

```python
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
    
    # Test that deleting non-existent attribute also raises AttributeError
    with pytest.raises(AttributeError):
        del currency.non_existent_attribute
    
    # Test that the currency object remains unchanged after failed deletions
    assert currency.code == "USD"
    assert currency.name == "US Dollars"
    assert currency.decimals == 2
    assert currency.type == CurrencyType.MONEY
    assert isinstance(currency.quantizer, Decimal)
    assert isinstance(currency.hashcache, int)


# LLM-generated content at query #8
#--------------------------

```python
def test_CurrencyRegistry_has():
    # Test with existing currency
    assert Currencies.has("USD") == True
    
    # Test with non-existing currency
    assert Currencies.has("NONEXISTENT") == False
    
    # Test with empty string
    assert Currencies.has("") == False
    
    # Test with lowercase currency code
    assert Currencies.has("usd") == False
    
    # Test with special characters
    assert Currencies.has("USD$") == False
    
    # Test that has() returns same result as 'in' operator
    test_code = "EUR"
    assert Currencies.has(test_code) == (test_code in Currencies)
    
    # Test with numeric string
    assert Currencies.has("123") == False
    
    # Test with very long string
    assert Currencies.has("A" * 100) == False


# LLM-generated content at query #9
#--------------------------

```python
def test_CurrencyRegistry___enter__():
    # Test that __enter__ returns a callable
    registry = CurrencyRegistry()
    with registry as register_func:
        assert callable(register_func)
        
        # Test that the returned callable can register currencies
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        register_func(usd)
        
        # Verify the currency was registered
        assert "USD" in registry
        assert registry["USD"] == usd
    
    # Test that context is closed after exit
    assert not registry._CurrencyRegistry__ctx_open
    
    # Test that currencies are properly sorted after context exit
    registry2 = CurrencyRegistry()
    with registry2 as register:
        eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
        gbp = Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY)
        aud = Currency.of("AUD", "Australian Dollar", 2, CurrencyType.MONEY)
        
        # Register in non-alphabetical order
        register(gbp)
        register(eur)
        register(aud)
    
    # Check alphabetical sorting
    assert registry2.codes == ["AUD", "EUR", "GBP"]
    assert [c.code for c in registry2.all] == ["AUD", "EUR", "GBP"]
    
    # Test that duplicate registration raises ValueError
    registry3 = CurrencyRegistry()
    with registry3 as register:
        currency1 = Currency.of("TEST", "Test Currency", 2, CurrencyType.MONEY)
        currency2 = Currency.of("TEST", "Test Currency Duplicate", 2, CurrencyType.MONEY)
        register(currency1)
        
        try:
            register(currency2)
            assert False, "Should have raised ValueError for duplicate currency"
        except ValueError as e:
            assert "Currency TEST is already registered" in str(e)
    
    # Test that registering outside context raises ProgrammingError
    registry4 = CurrencyRegistry()
    try:
        registry4._CurrencyRegistry__register(Currency.of("XYZ", "Test", 2, CurrencyType.MONEY))
        assert False, "Should have raised ProgrammingError"
    except ProgrammingError as e:
        assert "Can not create currencies outside registry context" in str(e)


# LLM-generated content at query #10
#--------------------------

```python
def test_CurrencyRegistry___getitem__():
    # Test getting an existing currency
    usd = Currencies["USD"]
    assert usd.code == "USD"
    assert usd.name == "US Dollar"
    assert usd.type == CurrencyType.MONEY
    
    # Test getting another existing currency
    eur = Currencies["EUR"]
    assert eur.code == "EUR"
    assert eur.name == "Euro"
    assert eur.type == CurrencyType.MONEY
    
    # Test that lookup is case-sensitive
    try:
        Currencies["usd"]
        assert False, "Should have raised CurrencyLookupError"
    except CurrencyLookupError as e:
        assert e.code == "usd"
    
    # Test getting non-existent currency raises CurrencyLookupError
    try:
        Currencies["NONEXISTENT"]
        assert False, "Should have raised CurrencyLookupError"
    except CurrencyLookupError as e:
        assert e.code == "NONEXISTENT"
        assert str(e) == "Currency identified by code 'NONEXISTENT' does not exist"
    
    # Test getting another non-existent currency
    try:
        Currencies["XYZ"]
        assert False, "Should have raised CurrencyLookupError"
    except CurrencyLookupError as e:
        assert e.code == "XYZ"
    
    # Test that returned currency is the same instance as from get() method
    assert Currencies["USD"] == Currencies.get("USD")
    
    # Test that different currencies are not equal
    assert Currencies["USD"] != Currencies["EUR"]
    
    # Test that currency has proper attributes
    jpy = Currencies["JPY"]
    assert jpy.code == "JPY"
    assert jpy.decimals == 0
    assert jpy.type == CurrencyType.MONEY
    
    # Test crypto currency if exists in registry
    try:
        btc = Currencies["BTC"]
        assert btc.type == CurrencyType.CRYPTO
    except CurrencyLookupError:
        pass  # BTC might not be in default registry


# LLM-generated content at query #11
#--------------------------

```python
def test_Currency___le__():
    # Test same currency comparison
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    # Test different currency comparison (based on ordering defined by @dataclass(frozen=True, order=True))
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    
    # Test with different attributes that affect ordering
    usd_lower_name = Currency.of("USD", "AA Dollars", 2, CurrencyType.MONEY)
    usd_higher_name = Currency.of("USD", "ZZ Dollars", 2, CurrencyType.MONEY)
    
    # Test with different decimals
    usd_dec_1 = Currency.of("USD", "US Dollars", 1, CurrencyType.MONEY)
    usd_dec_3 = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    
    # Test with different currency types
    usd_money = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd_crypto = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    
    # Test less than or equal comparisons
    assert usd1 <= usd2  # Same currency
    assert usd1 <= usd2 and usd2 <= usd1  # Both directions for equality
    
    # Test ordering based on dataclass field order (code, name, decimals, type, quantizer, hashcache)
    assert jpy <= eur  # "JPY" < "EUR" alphabetically
    assert not (eur <= jpy)  # "EUR" is not <= "JPY"
    
    # Test name affects ordering when codes are same
    assert usd_lower_name <= usd1  # "AA Dollars" <= "US Dollars"
    assert usd1 <= usd_higher_name  # "US Dollars" <= "ZZ Dollars"
    
    # Test decimals affects ordering when codes and names are same
    assert usd_dec_1 <= usd1  # decimals 1 <= 2
    assert usd1 <= usd_dec_3  # decimals 2 <= 3
    
    # Test type affects ordering when codes, names and decimals are same
    assert usd_money <= usd_crypto  # MONEY <= CRYPTO (alphabetically by enum value)
    
    # Test with non-Currency objects
    assert not (usd1 <= "USD")
    assert not (usd1 <= 123)
    assert not (usd1 <= None)
    
    # Test transitive property
    a = Currency.of("AAA", "Currency A", 0, CurrencyType.MONEY)
    b = Currency.of("BBB", "Currency B", 1, CurrencyType.MONEY)
    c = Currency.of("CCC", "Currency C", 2, CurrencyType.MONEY)
    
    assert a <= b
    assert b <= c
    assert a <= c  # Transitive: a <= b and b <= c implies a <= c


# LLM-generated content at query #12
#--------------------------

```python
def test_Currency___eq__():
    # Test equality with same currency
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2
    
    # Test inequality with different code
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert not (usd1 == eur)
    
    # Test inequality with different name
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 == usdx)
    
    # Test inequality with different decimals
    usd_decimals = Currency.of("USD", "US Dollars", 0, CurrencyType.MONEY)
    assert not (usd1 == usd_decimals)
    
    # Test inequality with different type
    usd_crypto = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert not (usd1 == usd_crypto)
    
    # Test equality with same hashcache
    jpy1 = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    jpy2 = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy1.hashcache == jpy2.hashcache
    assert jpy1 == jpy2
    
    # Test inequality with non-Currency object
    assert not (usd1 == "USD")
    assert not (usd1 == 123)
    assert not (usd1 == None)
    
    # Test inequality with Currency subclass (if existed)
    class FakeCurrency:
        def __init__(self):
            self.hashcache = usd1.hashcache
    
    fake = FakeCurrency()
    assert not (usd1 == fake)


# LLM-generated content at query #13
#--------------------------

```python
def test_Currency_quantize():
    # Test USD with 2 decimals (standard currency)
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    # Test rounding behavior with HALF-TO-EVEN
    assert USD.quantize(Decimal("1.005")) == Decimal("1.00")  # 1.005 rounds down to 1.00
    assert USD.quantize(Decimal("1.015")) == Decimal("1.02")  # 1.015 rounds up to 1.02
    assert USD.quantize(Decimal("1.025")) == Decimal("1.02")  # 1.025 rounds down to 1.02
    assert USD.quantize(Decimal("1.035")) == Decimal("1.04")  # 1.035 rounds up to 1.04
    
    # Test exact values
    assert USD.quantize(Decimal("1.00")) == Decimal("1.00")
    assert USD.quantize(Decimal("1.01")) == Decimal("1.01")
    assert USD.quantize(Decimal("1.99")) == Decimal("1.99")
    
    # Test JPY with 0 decimals
    JPY = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    
    assert JPY.quantize(Decimal("0.5")) == Decimal("0")   # 0.5 rounds down to 0
    assert JPY.quantize(Decimal("1.5")) == Decimal("2")   # 1.5 rounds up to 2
    assert JPY.quantize(Decimal("2.5")) == Decimal("2")   # 2.5 rounds down to 2
    assert JPY.quantize(Decimal("3.5")) == Decimal("4")   # 3.5 rounds up to 4
    
    # Test exact values
    assert JPY.quantize(Decimal("100")) == Decimal("100")
    assert JPY.quantize(Decimal("0")) == Decimal("0")
    
    # Test crypto currency with -1 decimals (max precision)
    ZZZ = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    
    # Test with very high precision
    assert ZZZ.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert ZZZ.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")
    assert ZZZ.quantize(Decimal("1.0000000000025")) == Decimal("1.000000000002")
    assert ZZZ.quantize(Decimal("1.0000000000035")) == Decimal("1.000000000004")
    
    # Test with different decimal values
    assert ZZZ.quantize(Decimal("0.0000000000005")) == Decimal("0.000000000000")
    assert ZZZ.quantize(Decimal("0.0000000000015")) == Decimal("0.000000000002")
    
    # Test currency with 1 decimal
    ONE_DEC = Currency.of("TST", "Test Currency", 1, CurrencyType.MONEY)
    
    assert ONE_DEC.quantize(Decimal("1.05")) == Decimal("1.0")   # 1.05 rounds down to 1.0
    assert ONE_DEC.quantize(Decimal("1.15")) == Decimal("1.2")   # 1.15 rounds up to 1.2
    assert ONE_DEC.quantize(Decimal("1.25")) == Decimal("1.2")   # 1.25 rounds down to 1.2
    assert ONE_DEC.quantize(Decimal("1.35")) == Decimal("1.4")   # 1.35 rounds up to 1.4
    
    # Test currency with 3 decimals
    THREE_DEC = Currency.of("THR", "Three Decimal", 3, CurrencyType.MONEY)
    
    assert THREE_DEC.quantize(Decimal("1.0005")) == Decimal("1.000")
    assert THREE_DEC.quantize(Decimal("1.0015")) == Decimal("1.002")
    assert THREE_DEC.quantize(Decimal("1.0025")) == Decimal("1.002")
    assert THREE_DEC.quantize(Decimal("1.0035")) == Decimal("1.004")
    
    # Test negative values
    assert USD.quantize(Decimal("-1.005")) == Decimal("-1.00")
    assert USD.quantize(Decimal("-1.015")) == Decimal("-1.02")
    assert JPY.quantize(Decimal("-0.5")) == Decimal("0")
    assert JPY.quantize(Decimal("-1.5")) == Decimal("-2")
    
    # Test zero
    assert USD.quantize(Decimal("0")) == Decimal("0.00")
    assert JPY.quantize(Decimal("0")) == Decimal("0")
    assert ZZZ.quantize(Decimal("0")) == Decimal("0.000000000000")


# LLM-generated content at query #14
#--------------------------

```python
def test_CurrencyRegistry():
    # Test singleton pattern
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2
    
    # Test initial state
    assert len(registry1) == 0
    assert registry1.all == []
    assert registry1.codes == []
    assert registry1.codenames == []
    
    # Test context manager registration
    with registry1 as register:
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        register(usd)
    
    # Verify registration worked
    assert len(registry1) == 1
    assert registry1.has("USD") is True
    assert "USD" in registry1
    assert registry1["USD"] == usd
    assert registry1.get("USD") == usd
    assert registry1.all == [usd]
    assert registry1.codes == ["USD"]
    assert registry1.codenames == [("USD", "US Dollar")]
    
    # Test duplicate registration
    with registry1 as register:
        try:
            register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Currency USD is already registered" in str(e)
    
    # Test registration outside context
    try:
        registry1._CurrencyRegistry__register(usd)
        assert False, "Should have raised ProgrammingError"
    except ProgrammingError as e:
        assert "Can not create currencies outside registry context" in str(e)
    
    # Test multiple registrations
    with registry1 as register:
        eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
        jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
        register(eur)
        register(jpy)
    
    # Verify sorting
    assert registry1.codes == ["EUR", "JPY", "USD"]
    assert registry1.all == [eur, jpy, usd]
    assert registry1.codenames == [
        ("EUR", "Euro"),
        ("JPY", "Japanese Yen"),
        ("USD", "US Dollar")
    ]
    
    # Test __getitem__ with non-existent currency
    try:
        registry1["XYZ"]
        assert False, "Should have raised CurrencyLookupError"
    except CurrencyLookupError as e:
        assert e.code == "XYZ"
        assert "Currency identified by code 'XYZ' does not exist" in str(e)
    
    # Test get() with default
    assert registry1.get("XYZ") is None
    assert registry1.get("XYZ", default=usd) == usd
    
    # Test has() method
    assert registry1.has("EUR") is True
    assert registry1.has("XYZ") is False
    
    # Test __contains__
    assert "EUR" in registry1
    assert "XYZ" not in registry1


# LLM-generated content at query #15
#--------------------------

```python
def test_Currency___le__():
    # Test same currency comparison
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    # Test different currency comparison (based on ordering)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    
    # Test with different attributes
    usd_diff_name = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    usd_diff_decimals = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    usd_diff_type = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    
    # Test __le__ with same currencies
    assert usd1 <= usd2
    assert usd2 <= usd1
    
    # Test __le__ with different currencies (ordering based on dataclass fields)
    # Since Currency is a dataclass with order=True, comparison uses field order
    # Fields order: code, name, decimals, type, quantizer, hashcache
    # So comparison primarily depends on code, then name, etc.
    
    # JPY (code "JPY") should be <= EUR (code "EUR")? Actually "EUR" < "JPY" alphabetically
    # So EUR <= JPY should be True
    assert eur <= jpy
    
    # Test __le__ with different name but same code
    # Since code is same, comparison moves to name
    # "US Dollars" < "UX Dollars" alphabetically
    assert usd1 <= usd_diff_name
    
    # Test __le__ with different decimals but same code and name
    # Since code and name are same, comparison moves to decimals
    # 2 <= 3
    assert usd1 <= usd_diff_decimals
    
    # Test __le__ with different type but same code, name, decimals
    # MONEY (value "Money") vs CRYPTO (value "Crypto Currency")
    # "Money" < "Crypto Currency" alphabetically
    assert usd1 <= usd_diff_type
    
    # Test __le__ with non-Currency object
    # Should return NotImplemented to allow Python to try reverse operation
    result = usd1.__le__("not a currency")
    assert result is NotImplemented


# LLM-generated content at query #16
#--------------------------

```python
def test_Currency___hash__():
    # Test that same currency objects have same hash
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert hash(usd1) == hash(usd2)
    
    # Test that different currency codes produce different hashes
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert hash(usd) != hash(eur)
    
    # Test that different names produce different hashes
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "United States Dollars", 2, CurrencyType.MONEY)
    assert hash(usd1) != hash(usd2)
    
    # Test that different decimals produce different hashes
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 0, CurrencyType.MONEY)
    assert hash(usd1) != hash(usd2)
    
    # Test that different currency types produce different hashes
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert hash(usd1) != hash(usd2)
    
    # Test that hash is consistent with equality
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2
    assert hash(usd1) == hash(usd2)
    
    # Test hash for currency with negative decimals
    crypto = Currency.of("BTC", "Bitcoin", -1, CurrencyType.CRYPTO)
    assert isinstance(hash(crypto), int)
    
    # Test hash for currency with zero decimals
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert isinstance(hash(jpy), int)
    
    # Test that hash is cached and returns same value multiple times
    currency = Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY)
    hash1 = hash(currency)
    hash2 = hash(currency)
    hash3 = hash(currency)
    assert hash1 == hash2 == hash3


# LLM-generated content at query #17
#--------------------------

```python
def test_CurrencyRegistry___enter__():
    # Test that __enter__ returns a callable
    registry = CurrencyRegistry()
    with registry as register_func:
        assert callable(register_func)
        
        # Test that the returned callable can register currencies
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        register_func(usd)
        
        # Verify the currency was registered
        assert "USD" in registry
        assert registry["USD"] == usd
    
    # Test that context is properly closed after exit
    assert not registry._CurrencyRegistry__ctx_open
    
    # Test that currencies are sorted after context exit
    registry2 = CurrencyRegistry()
    with registry2 as register:
        eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
        register(eur)
        gbp = Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY)
        register(gbp)
        aud = Currency.of("AUD", "Australian Dollar", 2, CurrencyType.MONEY)
        register(aud)
    
    # Check sorting by code
    assert registry2.codes == ["AUD", "EUR", "GBP"]
    
    # Test that registering outside context raises ProgrammingError
    registry3 = CurrencyRegistry()
    try:
        registry3._CurrencyRegistry__register(Currency.of("TEST", "Test", 2, CurrencyType.MONEY))
        assert False, "Should have raised ProgrammingError"
    except ProgrammingError:
        pass
    
    # Test duplicate registration within context raises ValueError
    registry4 = CurrencyRegistry()
    with registry4 as register:
        test_ccy = Currency.of("TEST", "Test", 2, CurrencyType.MONEY)
        register(test_ccy)
        try:
            register(test_ccy)
            assert False, "Should have raised ValueError for duplicate"
        except ValueError as e:
            assert "TEST" in str(e)


# LLM-generated content at query #18
#--------------------------

```python
def test_Currency___ge__():
    # Create currencies for comparison
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd_same = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    
    # Test equality (__ge__ should return True for equal objects)
    assert usd >= usd_same
    
    # Test with different currencies - should compare based on dataclass order
    # Since currencies are ordered by (code, name, decimals, type, quantizer, hashcache)
    # USD should be greater than JPY because "USD" > "JPY" alphabetically
    assert usd >= jpy
    
    # Test reflexivity
    assert usd >= usd
    
    # Test transitivity: if USD >= JPY and JPY >= EUR, then USD >= EUR
    # Note: This depends on actual ordering which is based on all fields
    usd_tuple = (usd.code, usd.name, usd.decimals, usd.type, usd.quantizer, usd.hashcache)
    eur_tuple = (eur.code, eur.name, eur.decimals, eur.type, eur.quantizer, eur.hashcache)
    
    # If USD tuple is greater than or equal to EUR tuple, then USD >= EUR should be True
    if usd_tuple >= eur_tuple:
        assert usd >= eur
    else:
        assert not (usd >= eur)
    
    # Test with non-Currency object
    # __ge__ should return NotImplemented for non-Currency comparisons
    result = usd.__ge__("not a currency")
    assert result is NotImplemented


# LLM-generated content at query #19
#--------------------------

```python
def test_CurrencyRegistry___len__():
    # Create a fresh registry instance
    registry = CurrencyRegistry()
    
    # Initially should be empty
    assert len(registry) == 0
    
    # Add some currencies and verify length updates
    with registry as register:
        register(Currency.of("TEST1", "Test Currency 1", 2, CurrencyType.MONEY))
        assert len(registry) == 1
        
        register(Currency.of("TEST2", "Test Currency 2", 0, CurrencyType.MONEY))
        assert len(registry) == 2
        
        register(Currency.of("TEST3", "Test Currency 3", -1, CurrencyType.CRYPTO))
        assert len(registry) == 3
    
    # After context exit, length should remain the same
    assert len(registry) == 3
    
    # Verify length matches the number of registered currencies
    assert len(registry) == len(registry.all)
    assert len(registry) == len(registry.codes)
    assert len(registry) == len(registry.codenames)


# LLM-generated content at query #20
#--------------------------

```python
def test_Currency():
    # Test basic currency creation
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert USD.code == "USD"
    assert USD.name == "US Dollars"
    assert USD.decimals == 2
    assert USD.type == CurrencyType.MONEY
    assert USD.quantizer == Decimal('0.01')
    
    # Test currency with 0 decimals
    JPY = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert JPY.code == "JPY"
    assert JPY.name == "Japanese Yen"
    assert JPY.decimals == 0
    assert JPY.quantizer == ZERO
    
    # Test currency with negative decimals (no fixed precision)
    ZZZ = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert ZZZ.code == "ZZZ"
    assert ZZZ.name == "Some weird currency"
    assert ZZZ.decimals == -1
    assert ZZZ.type == CurrencyType.CRYPTO
    assert ZZZ.quantizer == MaxPrecisionQuantizer
    
    # Test equality and hash
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    
    assert usd1 == usd2
    assert usd1 != usdx
    assert hash(usd1) == hash(usd2)
    assert hash(usd1) != hash(usdx)
    
    # Test quantize method
    assert USD.quantize(Decimal("1.005")) == Decimal('1.00')
    assert USD.quantize(Decimal("1.015")) == Decimal('1.02')
    assert JPY.quantize(Decimal("0.5")) == Decimal('0')
    assert JPY.quantize(Decimal("1.5")) == Decimal('2')
    assert ZZZ.quantize(Decimal("1.0000000000005")) == Decimal('1.000000000000')
    assert ZZZ.quantize(Decimal("1.0000000000015")) == Decimal('1.000000000002')
    
    # Test different currency types
    metal = Currency.of("XAU", "Gold", 4, CurrencyType.METAL)
    assert metal.type == CurrencyType.METAL
    
    alt = Currency.of("LOC", "Local Currency", 2, CurrencyType.ALTERNATIVE)
    assert alt.type == CurrencyType.ALTERNATIVE
    
    # Test hashcache is properly set
    assert USD.hashcache == hash(("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal('0.01')))


# LLM-generated content at query #21
#--------------------------

```python
def test_Currency():
    # Test valid currency creation
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert USD.code == "USD"
    assert USD.name == "US Dollars"
    assert USD.decimals == 2
    assert USD.type == CurrencyType.MONEY
    assert USD.quantizer == Decimal('0.01')
    
    # Test currency with 0 decimals
    JPY = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert JPY.code == "JPY"
    assert JPY.name == "Japanese Yen"
    assert JPY.decimals == 0
    assert JPY.type == CurrencyType.MONEY
    assert JPY.quantizer == ZERO
    
    # Test currency with -1 decimals (no fixed precision)
    ZZZ = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert ZZZ.code == "ZZZ"
    assert ZZZ.name == "Some weird currency"
    assert ZZZ.decimals == -1
    assert ZZZ.type == CurrencyType.CRYPTO
    assert ZZZ.quantizer == MaxPrecisionQuantizer
    
    # Test equality and hash
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2
    assert hash(usd1) == hash(usd2)
    
    # Test inequality with different name
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert usd1 != usdx
    assert hash(usd1) != hash(usdx)
    
    # Test quantize method
    assert USD.quantize(Decimal("1.005")) == Decimal("1.00")
    assert USD.quantize(Decimal("1.015")) == Decimal("1.02")
    assert JPY.quantize(Decimal("0.5")) == Decimal("0")
    assert JPY.quantize(Decimal("1.5")) == Decimal("2")
    
    # Test different currency types
    metal = Currency.of("XAU", "Gold", 4, CurrencyType.METAL)
    assert metal.type == CurrencyType.METAL
    
    alt = Currency.of("ALT", "Alternative", 2, CurrencyType.ALTERNATIVE)
    assert alt.type == CurrencyType.ALTERNATIVE
    
    # Test hash cache
    assert usd1.hashcache == hash(usd1)
    assert usd2.hashcache == hash(usd2)
    
    # Test that currency is immutable (frozen dataclass)
    with pytest.raises(dataclasses.FrozenInstanceError):
        USD.code = "EUR"


# LLM-generated content at query #22
#--------------------------

```python
def test_CurrencyRegistry___len__():
    # Create a fresh registry instance
    registry = CurrencyRegistry()
    
    # Initially should be empty
    assert len(registry) == 0
    
    # Add some currencies and check length
    with registry as register:
        register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        assert len(registry) == 1
        
        register(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))
        assert len(registry) == 2
        
        register(Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY))
        assert len(registry) == 3
    
    # After context exit, length should remain the same
    assert len(registry) == 3
    
    # Try to add outside context (should fail)
    try:
        registry._CurrencyRegistry__register(Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY))
    except ProgrammingError:
        pass  # Expected behavior
    
    # Length should still be 3
    assert len(registry) == 3
    
    # Test with duplicate currency (should fail)
    try:
        with registry as register:
            register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
    except ValueError:
        pass  # Expected behavior
    
    # Length should still be 3 (duplicate not added)
    assert len(registry) == 3


# LLM-generated content at query #23
#--------------------------

```python
def test_CurrencyRegistry___exit__():
    # Create a fresh registry instance
    registry = CurrencyRegistry()
    
    # Add some currencies in unsorted order
    with registry as register:
        register(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))
        register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
        register(Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY))
        register(Currency.of("AED", "UAE Dirham", 2, CurrencyType.MONEY))
    
    # Verify that currencies are sorted by code after __exit__
    assert registry.codes == ["AED", "EUR", "JPY", "USD"]
    
    # Verify that currencies list is sorted
    assert [c.code for c in registry.all] == ["AED", "EUR", "JPY", "USD"]
    
    # Verify that codenames are sorted
    assert registry.codenames == [
        ("AED", "UAE Dirham"),
        ("EUR", "Euro"),
        ("JPY", "Japanese Yen"),
        ("USD", "US Dollar")
    ]
    
    # Verify that registry is accessible by code in sorted order
    assert registry["AED"].code == "AED"
    assert registry["EUR"].code == "EUR"
    assert registry["JPY"].code == "JPY"
    assert registry["USD"].code == "USD"
    
    # Verify that __ctx_open is False after __exit__
    assert registry._CurrencyRegistry__ctx_open == False
    
    # Test with empty registry
    empty_registry = CurrencyRegistry()
    with empty_registry as register:
        pass  # No currencies added
    
    assert empty_registry.codes == []
    assert empty_registry.all == []
    assert empty_registry.codenames == []
    assert empty_registry._CurrencyRegistry__ctx_open == False


# LLM-generated content at query #24
#--------------------------

```python
def test_CurrencyRegistry___exit__():
    # Create a fresh registry instance
    registry = CurrencyRegistry()
    
    # Add some currencies in unsorted order
    with registry as register:
        register(Currency.of("ZZZ", "Z Currency", 2, CurrencyType.MONEY))
        register(Currency.of("AAA", "A Currency", 2, CurrencyType.MONEY))
        register(Currency.of("MMM", "M Currency", 2, CurrencyType.MONEY))
    
    # Verify that currencies are sorted by code after __exit__
    assert registry.codes == ["AAA", "MMM", "ZZZ"]
    assert registry.all[0].code == "AAA"
    assert registry.all[1].code == "MMM"
    assert registry.all[2].code == "ZZZ"
    
    # Verify that codenames are also sorted
    assert registry.codenames == [
        ("AAA", "A Currency"),
        ("MMM", "M Currency"),
        ("ZZZ", "Z Currency")
    ]
    
    # Verify that __registry is sorted
    assert list(registry._CurrencyRegistry__registry.keys()) == ["AAA", "MMM", "ZZZ"]
    
    # Verify that context is closed after __exit__
    assert registry._CurrencyRegistry__ctx_open == False
    
    # Test that currencies can be accessed properly after __exit__
    assert registry["AAA"].code == "AAA"
    assert registry["AAA"].name == "A Currency"
    assert registry["MMM"].code == "MMM"
    assert registry["ZZZ"].code == "ZZZ"
    
    # Test that __len__ works correctly
    assert len(registry) == 3
    
    # Test that __contains__ works correctly
    assert "AAA" in registry
    assert "MMM" in registry
    assert "ZZZ" in registry
    assert "BBB" not in registry
    
    # Test that has() method works correctly
    assert registry.has("AAA") == True
    assert registry.has("BBB") == False
    
    # Test that get() method works correctly
    assert registry.get("AAA").code == "AAA"
    assert registry.get("BBB") is None
    assert registry.get("BBB", default=registry["AAA"]).code == "AAA"


# LLM-generated content at query #25
#--------------------------

```python
def test_CurrencyRegistry___enter__():
    # Test that __enter__ returns a callable
    registry = CurrencyRegistry()
    with registry as register_func:
        assert callable(register_func)
        assert register_func.__name__ == "__register"
        
        # Test that the context is open
        assert registry._CurrencyRegistry__ctx_open is True
        
        # Test that we can register a currency within the context
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        register_func(usd)
        
        # Verify the currency was registered
        assert "USD" in registry._CurrencyRegistry__registry
        
    # Test that context is closed after exit
    assert registry._CurrencyRegistry__ctx_open is False
    
    # Test that we cannot register outside context
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    try:
        registry._CurrencyRegistry__register(eur)
        assert False, "Should have raised ProgrammingError"
    except ProgrammingError as e:
        assert "Can not create currencies outside registry context" in str(e)


# LLM-generated content at query #26
#--------------------------

```python
def test_Currency___delattr__():
    # Create a currency instance
    currency = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    # Test that __delattr__ raises AttributeError for all attributes
    # This is expected since Currency is a frozen dataclass
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
    
    # Test that __delattr__ raises AttributeError for non-existent attributes
    with pytest.raises(AttributeError):
        del currency.non_existent_attribute
    
    # Test that __delattr__ doesn't affect the equality of currencies
    currency1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    currency2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    # Verify they're equal before attempting deletion
    assert currency1 == currency2
    
    # Attempt deletion on one instance (should fail)
    with pytest.raises(AttributeError):
        del currency1.code
    
    # Verify they're still equal after attempted deletion
    assert currency1 == currency2
    
    # Test that the currency remains usable after attempted deletion
    assert currency1.code == "USD"
    assert currency1.name == "US Dollars"
    assert currency1.decimals == 2
    assert currency1.type == CurrencyType.MONEY
    assert currency1.quantize(Decimal("1.005")) == Decimal("1.00")


# LLM-generated content at query #27
#--------------------------

```python
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
    
    # Test that deleting non-existent attribute also raises AttributeError
    with pytest.raises(AttributeError):
        del currency.non_existent_attribute
    
    # Verify the currency instance is still intact after failed deletions
    assert currency.code == "USD"
    assert currency.name == "US Dollars"
    assert currency.decimals == 2
    assert currency.type == CurrencyType.MONEY
    assert isinstance(currency.quantizer, Decimal)
    assert isinstance(currency.hashcache, int)


# LLM-generated content at query #28
#--------------------------

```python
def test_Currency___le__():
    # Test equal currencies
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 <= usd2
    assert usd2 <= usd1
    
    # Test different currencies (ordering based on dataclass order=True)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    
    # Since dataclass has order=True, comparison uses field order:
    # code, name, decimals, type, quantizer, hashcache
    # So comparison is primarily based on code
    assert jpy <= eur  # "JPY" < "EUR" alphabetically? Actually "E" < "J", so EUR < JPY
    # Let's verify the actual ordering
    currencies = sorted([jpy, eur])
    assert currencies[0].code <= currencies[1].code
    
    # Test with same code but different name
    usd_a = Currency.of("USD", "A Dollars", 2, CurrencyType.MONEY)
    usd_b = Currency.of("USD", "B Dollars", 2, CurrencyType.MONEY)
    assert usd_a <= usd_b  # "A Dollars" < "B Dollars"
    
    # Test with same code and name but different decimals
    usd_dec2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd_dec3 = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert usd_dec2 <= usd_dec3  # 2 < 3
    
    # Test with same code, name, decimals but different type
    usd_money = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd_crypto = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    # CurrencyType.MONEY < CurrencyType.CRYPTO? Based on enum definition order
    assert usd_money <= usd_crypto
    
    # Test reflexivity
    assert usd1 <= usd1
    
    # Test transitivity
    btc = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    eth = Currency.of("ETH", "Ethereum", 8, CurrencyType.CRYPTO)
    xrp = Currency.of("XRP", "Ripple", 6, CurrencyType.CRYPTO)
    
    sorted_currencies = sorted([btc, eth, xrp])
    for i in range(len(sorted_currencies) - 1):
        assert sorted_currencies[i] <= sorted_currencies[i + 1]
    
    # Test with non-Currency object (should return NotImplemented)
    result = usd1.__le__("not a currency")
    assert result is NotImplemented


# LLM-generated content at query #29
#--------------------------

```python
def test_Currency___hash__():
    # Test that identical currencies have same hash
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert hash(usd1) == hash(usd2)
    
    # Test that different currencies have different hashes
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert hash(usd) != hash(eur)
    
    # Test that hash changes with different code
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("EUR", "US Dollars", 2, CurrencyType.MONEY)
    assert hash(usd1) != hash(usd2)
    
    # Test that hash changes with different name
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    assert hash(usd1) != hash(usd2)
    
    # Test that hash changes with different decimals
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 0, CurrencyType.MONEY)
    assert hash(usd1) != hash(usd2)
    
    # Test that hash changes with different type
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert hash(usd1) != hash(usd2)
    
    # Test that hash is consistent across multiple calls
    currency = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    hash1 = hash(currency)
    hash2 = hash(currency)
    hash3 = hash(currency)
    assert hash1 == hash2 == hash3
    
    # Test hash with negative decimals (crypto-like currency)
    btc = Currency.of("BTC", "Bitcoin", -1, CurrencyType.CRYPTO)
    eth = Currency.of("ETH", "Ethereum", -1, CurrencyType.CRYPTO)
    assert hash(btc) != hash(eth)
    
    # Test that hashcache is used correctly
    currency = Currency.of("CAD", "Canadian Dollar", 2, CurrencyType.MONEY)
    assert currency.hashcache == hash(currency)
    
    # Test hash equality with same attributes but different instances
    ccy1 = Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY)
    assert hash(ccy1) == hash(ccy2)
    assert ccy1 == ccy2


# LLM-generated content at query #30
#--------------------------

```python
def test_Currency___hash__():
    # Test that equal currencies have equal hash values
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1.__hash__() == usd2.__hash__()
    
    # Test that different currencies have different hash values
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert usd1.__hash__() != jpy.__hash__()
    
    # Test that hash is consistent with equality
    assert (usd1 == usd2) == (usd1.__hash__() == usd2.__hash__())
    
    # Test that different code affects hash
    usdx = Currency.of("EUR", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1.__hash__() != usdx.__hash__()
    
    # Test that different name affects hash
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert usd1.__hash__() != usdx.__hash__()
    
    # Test that different decimals affects hash
    usdx = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert usd1.__hash__() != usdx.__hash__()
    
    # Test that different type affects hash
    usdx = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert usd1.__hash__() != usdx.__hash__()
    
    # Test that hash is cached and returns same value on multiple calls
    hash1 = usd1.__hash__()
    hash2 = usd1.__hash__()
    assert hash1 == hash2
    
    # Test hash with negative decimals (crypto-like currency)
    btc = Currency.of("BTC", "Bitcoin", -1, CurrencyType.CRYPTO)
    assert btc.__hash__() is not None
    
    # Test hash with zero decimals
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy.__hash__() is not None


# LLM-generated content at query #31
#--------------------------

```python
def test_Currency___ge__():
    # Create currencies for comparison
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd_same = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd_different_name = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    
    # Test equality (__ge__ should return True for equal objects)
    assert usd >= usd_same
    assert usd_same >= usd
    
    # Test with different currencies (ordering is based on dataclass order=True)
    # Since currencies are ordered by all fields, we can test ordering
    assert usd >= usd  # Reflexive
    assert not (usd >= usd_different_name) or not (usd_different_name >= usd)  # Not both true
    
    # Test with non-Currency objects
    assert not (usd >= "USD")
    assert not (usd >= 123)
    assert not (usd >= None)
    
    # Test transitive property if applicable
    # Note: Since currencies have multiple fields, transitive property might not hold
    # for all comparisons, but we can test specific cases
    
    # Test that __ge__ works with __le__ and __eq__
    assert (usd >= usd_same) == (usd_same <= usd)
    assert (usd >= usd_same) == (usd == usd_same)
    
    # Test with different currency types
    crypto = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    metal = Currency.of("XAU", "Gold", -1, CurrencyType.METAL)
    
    # These should not raise errors
    _ = crypto >= metal
    _ = metal >= crypto


# LLM-generated content at query #32
#--------------------------

```python
def test_CurrencyRegistry___enter__():
    registry = CurrencyRegistry()
    
    # Test that __enter__ returns a callable
    with registry as register_func:
        assert callable(register_func)
        
        # Test that the returned function can register a currency
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        register_func(usd)
        
        # Verify the currency was registered
        assert "USD" in registry
        assert registry["USD"] == usd
    
    # Test that context is properly closed after exit
    assert not registry._CurrencyRegistry__ctx_open
    
    # Test that currencies are sorted after context exit
    registry2 = CurrencyRegistry()
    with registry2 as register:
        eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
        gbp = Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY)
        register(gbp)
        register(eur)
    
    # Verify currencies are sorted by code
    assert registry2.codes == ["EUR", "GBP"]
    
    # Test that duplicate registration raises ValueError
    registry3 = CurrencyRegistry()
    with registry3 as register:
        currency1 = Currency.of("TEST", "Test Currency", 2, CurrencyType.MONEY)
        currency2 = Currency.of("TEST", "Another Test", 2, CurrencyType.MONEY)
        register(currency1)
        
        import pytest
        with pytest.raises(ValueError, match="Currency TEST is already registered"):
            register(currency2)
    
    # Test that registering outside context raises ProgrammingError
    registry4 = CurrencyRegistry()
    with pytest.raises(ProgrammingError, match="Can not create currencies outside registry context"):
        registry4._CurrencyRegistry__register(usd)


# LLM-generated content at query #33
#--------------------------

```python
def test_Currency___repr__():
    # Test basic repr for money currency
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert repr(usd) == "Currency(code='USD', name='US Dollars', decimals=2, type=<CurrencyType.MONEY: 'Money'>, quantizer=Decimal('0.01'), hashcache=" + str(hash(usd)) + ")"
    
    # Test repr for currency with 0 decimals
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert repr(jpy) == "Currency(code='JPY', name='Japanese Yen', decimals=0, type=<CurrencyType.MONEY: 'Money'>, quantizer=Decimal('0'), hashcache=" + str(hash(jpy)) + ")"
    
    # Test repr for crypto currency with -1 decimals (max precision)
    btc = Currency.of("BTC", "Bitcoin", -1, CurrencyType.CRYPTO)
    assert repr(btc) == "Currency(code='BTC', name='Bitcoin', decimals=-1, type=<CurrencyType.CRYPTO: 'Crypto Currency'>, quantizer=MaxPrecisionQuantizer, hashcache=" + str(hash(btc)) + ")"
    
    # Test repr for metal currency
    xau = Currency.of("XAU", "Gold", 4, CurrencyType.METAL)
    assert repr(xau) == "Currency(code='XAU', name='Gold', decimals=4, type=<CurrencyType.METAL: 'Precious Metal'>, quantizer=Decimal('0.0001'), hashcache=" + str(hash(xau)) + ")"
    
    # Test repr for alternative currency
    lvc = Currency.of("LVC", "Local Currency", 2, CurrencyType.ALTERNATIVE)
    assert repr(lvc) == "Currency(code='LVC', name='Local Currency', decimals=2, type=<CurrencyType.ALTERNATIVE: 'Alternative'>, quantizer=Decimal('0.01'), hashcache=" + str(hash(lvc)) + ")"


