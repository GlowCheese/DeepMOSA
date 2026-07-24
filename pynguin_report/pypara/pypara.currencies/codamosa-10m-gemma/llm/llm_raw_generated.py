####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_CurrencyRegistry___enter__():
    """
    Tests the __enter__ method of CurrencyRegistry.
    The method should set the internal context flag to True and return the register method.
    """
    # Reset singleton instance for clean test environment
    CurrencyRegistry._CurrencyRegistry__instance = None
    registry = CurrencyRegistry()
    
    # Verify initial state: context should be closed (False)
    # We access private attribute __ctx_open for verification
    assert registry._CurrencyRegistry__ctx_open is False
    
    # Execute __enter__
    register_method = registry.__enter__()
    
    # Verify context is now open
    assert registry._CurrencyRegistry__ctx_open is True
    
    # Verify that the returned object is the internal __register method
    assert register_method == registry._CurrencyRegistry__register
    
    # Verify that calling the returned method works (as part of the context)
    # Create a valid currency
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    # Since we are manually calling __enter__, we must manually call __exit__ 
    # to maintain registry integrity, or just test the enter logic in isolation.
    # Here we test if the register method (returned by __enter__) can be called.
    register_method(usd)
    assert registry.has("USD")
    
    # Clean up: close the context manually
    registry.__exit__(None, None, None)
    assert registry._CurrencyRegistry__ctx_open is False
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from decimal import Decimal

def test_Currency___eq__():
    # Setup valid currencies
    usd_base = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd_same = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    # Setup different currencies (different name)
    usd_diff_name = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    
    # Setup different currencies (different code)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    
    # Setup different currencies (different decimals)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    
    # Setup different currencies (different type)
    usd_crypto = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)

    # Test equality
    assert usd_base == usd_same
    
    # Test inequality with different name
    assert usd_base != usd_diff_name
    
    # Test inequality with different code
    assert usd_base != eur
    
    # Test inequality with different decimals
    assert usd_base != jpy
    
    # Test inequality with different type
    assert usd_base != usd_crypto
    
    # Test equality with different object identity but same values
    assert usd_base == Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)

    # Test equality with non-Currency objects
    assert usd_base != "USD"
    assert usd_base != 123
    assert usd_base != None

    # Test hash consistency for equal objects
    assert hash(usd_base) == hash(usd_same)
    assert hash(usd_base) != hash(usd_diff_name)
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest

def test_CurrencyRegistry_has():
    """
    Tests the 'has' method of the CurrencyRegistry class.
    """
    # Reset Singleton instance for clean testing environment
    CurrencyRegistry._instance = None
    registry = CurrencyRegistry()
    
    # Create sample currencies
    usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONelli)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    crypto = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)

    # Test 1: 'has' should return False when registry is empty
    assert registry.has("USD") is False
    assert "USD" not in registry

    # Test 2: 'has' should return True after registration via context manager
    with registry as register:
        register(usd)
        register(eur)
        register(jpy)
        register(crypto)
    
    # After context exit, registry is sorted and finalized
    assert registry.has("USD") is True
    assert registry.has("EUR") is True
    assert registry.has("JPY") is True
    assert registry.has("BTC") is True

    # Test 3: 'has' should return False for non-existent codes
    assert registry.has("GBP") is False
    assert registry.has("XYZ") is False
    assert registry.has("") is False

    # Test 4: 'has' should return False for case-sensitive mismatches
    # (Since Currency.of enforces uppercase, 'usd' is not a valid key)
    assert registry.has("usd") is False
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest

def test_CurrencyRegistry___contains__():
    """
    Tests the __contains__ method of CurrencyRegistry.
    """
    # Reset singleton instance for clean test environment
    CurrencyRegistry._CurrencyRegistry__instance = None
    registry = CurrencyRegistry()
    
    # Create test currencies
    usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    
    # Test empty registry
    assert "USD" not in registry
    assert "JPY" not in registry
    
    # Populate registry using context manager
    with registry as register:
        register(usd)
        register(jpy)
        
    # Test presence of registered currencies
    assert "USD" in registry
    assert "JPY" in registry
    
    # Test absence of non-registered currency
    assert "EUR" not in registry
    
    # Test with an empty string
    assert "" not in registry
```


# LLM-generated content at query #5
#--------------------------

```python
def test_Currency___gt__():
    """
    Tests the __gt__ (greater than) magic method of the Currency class.
    Note: Since the class is decorated with @dataclass(order=True), 
    __gt__ is automatically implemented based on the order of fields 
    defined in the class (code, name, decimals, type, quantizer, hashcache).
    """
    # Setup currencies with identical attributes except for 'code' to test comparison
    # We use .of() to ensure valid construction
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    abc = Currency.of("ABC", "Alpha Currency", 2, CurrencyType.MONEY)
    
    # Since 'code' is the first field in the dataclass, 
    # the comparison starts with 'code'.
    # 'USD' > 'ABC' should be True
    assert usd > abc
    
    # 'ABC' > 'USD' should be False
    assert not (abc > usd)

    # Test with different decimals (the next field in order after name/code)
    # We need to ensure the 'code' and 'name' are the same to reach 'decimals' comparison
    usd_low_precision = Currency.of("USD", "US Dollars", 0, CurrencyType.MONEY)
    usd_high_precision = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    # In the dataclass definition: code, name, decimals...
    # For usd_high_precision vs usd_low_precision:
    # code is same, name is same, 2 > 0 is True
    assert usd_high_precision > usd_low_precision
    assert not (usd_low_precision > usd_high_precision)

    # Test equality case (not gt)
    usd_duplicate = Currency.of("USD", "US Dollars", 2, CurrencyType.MONTY) # This would fail due to name/code logic
    # Let's just use a direct copy logic if we were to bypass .of, 
    # but using .of with same params:
    usd_copy = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd > usd_copy)
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest
from decimal import Decimal

def test_Currency___setattr__():
    """
    Tests that the Currency class, being a frozen dataclass, 
    raises FrozenInstanceError (or AttributeError) when attempting 
    to set an attribute after instantiation.
    """
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    # Since the class is decorated with @dataclass(frozen=True),
    # any attempt to modify attributes after creation should raise an error.
    with pytest.raises(AttributeError):
        usd.code = "EUR"

    with pytest.raises(AttributeError):
        usd.decimals = 3

    with pytest.raises(AttributeError):
        usd.name = "New Name"
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
from decimal import Decimal

def test_Currency():
    # Test valid creation via the .of() factory method
    usd_code = "USD"
    usd_name = "US Dollars"
    usd_decimals = 2
    usd_type = CurrencyType.MONEY
    
    usd = Currency.of(usd_code, usd_name, usd_decimals, usd_type)
    
    assert usd.code == usd_code
    assert usd.name == usd_name
    assert usd.decimals == usd_decimals
    assert usd.type == usd_type
    assert isinstance(usd.quantizer, Decimal)
    
    # Test Quantization logic for different decimal settings
    # 2 decimals (USD)
    assert usd.quantize(Decimal("1.005")) == Decimal("1.00")
    assert usd.quantize(Decimal("1.015")) == Decimal("1.02")
    
    # 0 decimals (JPY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy.quantize(Decimal("0.5")) == Decimal("0")
    assert jpy.quantize(Decimal("1.5")) == Decimal("2")
    
    # -1 decimals (Crypto/Special)
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert zzz.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert zzz.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")

    # Test Equality and Hashing
    usd_duplicate = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd_diff_name = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    
    assert usd == usd_duplicate
    assert usd != usd_diff_name
    assert hash(usd) == hash(usd_duplicate)
    assert hash(usd) != hash(usd_diff_name)

    # Test Validation Errors (ProgrammingError)
    # Invalid code (not uppercase)
    with pytest.raises(ProgrammingError):
        Currency.of("usd", "US Dollars", 2, CurrencyType.MONEY)
    
    # Invalid code (not alphabetic)
    with pytest.raises(ProgrammingError):
        Currency.of("US1", "US Dollars", 2, CurrencyType.MONEY)
        
    # Invalid name (empty)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "", 2, CurrencyType.MONEY)
        
    # Invalid name (leading/trailing space)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", " US Dollars", 2, CurrencyType.MONEY)
        
    # Invalid decimals (less than -1)
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", -2, CurrencyType.MONEY)
        
    # Invalid type
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", 2, "NotAType")

def test_CurrencyLookupError():
    error = CurrencyLookupError("XYZ")
    assert error.code == "XYZ"
    assert "XYZ" in str(error)
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest

def test_CurrencyRegistry___contains__():
    """
    Tests the __contains__ method of CurrencyRegistry.
    """
    # Reset singleton instance for a clean test environment
    CurrencyRegistry._CurrencyRegistry__instance = None
    registry = CurrencyRegistry()
    
    # Create a test currency
    usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    
    # Test before registry population
    assert "USD" not in registry
    assert "EUR" not in registry
    
    # Populate registry using the context manager
    with registry as register:
        register(usd)
        register(eur)
    
    # Test presence of registered currencies
    assert "USD" in registry
    assert "EUR" in registry
    
    # Test absence of unregistered currencies
    assert "JPY" not in registry
    assert "GBP" not in registry
    assert "" not in registry
    assert None not in registry
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest

def test_CurrencyRegistry___contains__():
    """
    Tests the __contains__ method of CurrencyRegistry.
    """
    # Reset singleton instance for clean testing environment
    CurrencyRegistry._instance = None
    registry = CurrencyRegistry()
    
    # Create sample currencies
    usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    crypto = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)

    # Test outside of context (registry is empty)
    assert "USD" not in registry
    assert "JPY" not in registry
    assert "BTC" not in registry

    # Test inside registry context
    with registry as register:
        register(usd)
        register(jpy)
        register(crypto)
    
    # Verify existence of registered codes
    assert "USD" in registry
    assert "JPY" in registry
    assert "BTC" in registry
    
    # Verify non-existent codes are not in registry
    assert "EUR" not in registry
    assert "ABC" not in registry
    assert "" not in registry
    
    # Verify type of membership check (should handle string lookup)
    assert isinstance("USD" in registry, bool)
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_CurrencyRegistry___enter__():
    """
    Tests the __enter__ method of CurrencyRegistry.
    It should set the internal context flag to True and return the __register method.
    """
    # Reset the singleton instance for a clean test environment
    CurrencyRegistry._CurrencyRegistry__instance = None
    registry = CurrencyRegistry()
    
    # Verify initial state: context should be closed (not open)
    # Since __ctx_open is private, we check the behavior of __register
    with pytest.raises(ProgrammingError, match="Can not create currencies outside registry context."):
        dummy_currency = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        registry.__register(dummy_currency)

    # Use the context manager
    with registry as register_func:
        # 1. Verify that the returned object is the __register method
        assert callable(register_func)
        assert register_func == registry._CurrencyRegistry__register
        
        # 2. Verify that we can now register a currency without error
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        register_func(usd)
        assert "USD" in registry

    # 3. Verify that after exiting the context, the context flag is reset (closed)
    # and adding a duplicate/new currency outside context raises ProgrammingError
    with pytest.raises(ProgrammingError, match="Can not create currencies outside registry context."):
        eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
        registry.__register(eur)
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
from decimal import Decimal

def test_Currency___le__():
    """
    Tests the __le__ (less than or equal to) magic method of the Currency class.
    Since the class is decorated with @dataclass(order=True), the comparison
    is based on the order of fields defined in the class:
    code, name, decimals, type, quantizer, hashcache.
    """
    # Setup currencies for comparison
    # USD is "less than" USD_ALT because 'USD' == 'USD' but 'US Dollars' < 'USD Alternative'
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd_alt = Currency.of("USD", "USD Alternative", 2, CurrencyType.MONEY)
    
    # AUD is "less than" USD because 'AUD' < 'USD'
    aud = Currency.of("AUD", "Australian Dollars", 2, CurrencyType.MONEY)
    
    # USD is "greater than" AUD
    
    # Test equality (le should be true)
    assert usd <= usd
    
    # Test less than (le should be true)
    assert aud <= usd
    assert aud <= usd_alt
    
    # Test equal to (le should be true)
    assert usd <= usd_alt # 'US Dollars' comes before 'USD Alternative' alphabetically
    
    # Test greater than (le should be false)
    assert usd <= aud is False
    assert usd_alt <= aud is False
    assert usd_alt <= usd is False

    # Test with different decimals (affects order if code and name are same)
    usd_zero_dec = Currency.of("USD", "US Dollars", 0, CurrencyType.MONEY)
    # Comparing (USD, US Dollars, 2, ...) vs (USD, US Dollars, 0, ...)
    # Since 0 < 2, the 0-decimal version is "less than" the 2-decimal version
    assert usd_zero_dec <= usd
    assert usd > usd_zero_dec
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
from decimal import Decimal

def test_Currency___lt__():
    """
    Tests the __lt__ method of the Currency class.
    Since Currency is a dataclass with order=True, it implements __lt__ 
    based on the order of fields defined in the class: 
    (code, name, decimals, type, quantizer, hashcache).
    """
    # Setup currencies with different codes to test primary sorting attribute
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    abc = Currency.of("ABC", "Alpha Currency", 2, CurrencyType.MONEY)
    
    # Setup currencies with same code but different names to test secondary attribute
    usd_alt_name = Currency.of("USD", "United States Dollars", 2, CurrencyType.MONEY)
    
    # Setup currencies with same code and name but different decimals
    usd_zero_dec = Currency.of("USD", "US Dollars", 0, CurrencyType.MONEY)

    # Test 1: Comparison based on 'code' (primary attribute)
    # 'ABC' < 'USD' alphabetically
    assert abc < usd
    assert usd > abc

    # Test 2: Comparison based on 'name' (secondary attribute)
    # 'US Dollars' < 'United States Dollars'
    assert usd < usd_alt_name
    assert usd_alt_name > usd

    # Test 3: Comparison based on 'decimals' (tertiary attribute)
    # 'USD' (2) vs 'USD' (0) -> Since code and name are same, it checks decimals
    # Note: In the class definition, 'decimals' comes before 'type'
    # We need to ensure 'code' and 'name' are identical for this test
    usd_2_dec = Currency.of("USD", "US Dollars", 2, Currency    CurrencyType.MONEY)
    usd_0_dec = Currency.of("USD", "US Dollars", 0, CurrencyType.MONEY)
    
    # Since 0 < 2, the one with 0 decimals should be "less than" the one with 2 decimals
    # if the preceding fields (code, name) are equal.
    assert usd_0_dec < usd_2_dec
    assert usd_2_dec > usd_0_dec

    # Test 4: Equality check (not strictly __lt__, but part of ordering logic)
    # Two currencies with same properties should not be less than each other
    assert not (usd < usd)
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest
from decimal import Decimal

def test_Currency___hash__():
    """
    Tests the __hash__ method of the Currency class to ensure it returns
    the pre-computed hashcache and maintains consistency for identical objects.
    """
    # Setup identical currency objects
    code = "USD"
    name = "US Dollars"
    decimals = 2
    ctype = CurrencyType.MONEY
    
    usd1 = Currency.of(code, name, decimals, ctype)
    usd2 = Currency.of(code, name, decimals, ctype)
    
    # Setup a different currency object
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)

    # Test that hash is consistent for the same object
    assert hash(usd1) == usd1.hashcache
    
    # Test that hash is consistent for different instances with the same values
    assert hash(usd1) == hash(usd2)
    assert usd1 == usd2
    
    # Test that different currencies produce different hashes
    assert hash(usd1) != hash(eur)
    
    # Test that the hash is used correctly in a set
    currency_set = {usd1, usd2, eur}
    assert len(currency_set) == 2
    assert usd1 in currency_set
    assert eur in currency_set
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest
from decimal import Decimal

def test_Currency___ge__():
    """
    Tests the __ge__ (greater than or equal) magic method of the Currency class.
    Since the class is decorated with @dataclass(order=True), __ge__ is 
    automatically implemented based on the order of fields in the class definition.
    The order of fields is: code, name, decimals, type, quantizer, hashcache.
    """
    # Setup currencies with different values for the first field (code)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    gbp = Currency.of("GBP", "British Pounds", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    
    # Test equality (ge should be True when equal)
    usd_duplicate = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd >= usd_duplicate
    assert usd >= usd

    # Test greater than (based on alphabetical order of 'code')
    # 'USD' comes after 'GBP' and 'JPY'
    assert usd >= gbp
    assert usd >= jpy
    
    # Test less than (gbp is alphabetically less than usd)
    assert gbp <= usd
    assert jpy <= usd

    # Test field precedence: if codes are equal, check 'name'
    usd_alt_name = Currency.of("USD", "United States Dollars", 2, CurrencyType.MONEV) # This would fail .of() validation
    # Because .of() has strict validation, we must use the constructor for specific edge cases 
    # to test the internal logic of __ge__ without triggering ProgrammingError.
    
    # Manual creation to bypass .of() validation for testing order logic
    # Comparing USD (code="USD", name="US Dollars") vs USD_Z (code="USD", name="US Dollars Z")
    usd_z = Currency("USD", "US Dollars Z", 2, CurrencyType.MONEY, Decimal("0.01"), hash("USD_Z"))
    assert usd_z >= usd
    assert usd >= usd_z # False, because 'US Dollars Z' > 'US Dollars'
    
    # Comparing decimals: USD (2) vs USD_0 (0)
    usd_0 = Currency("USD", "US Dollars", 0, CurrencyType.MONEY, Decimal("1"), hash("USD_0"))
    # Note: 'code' and 'name' are identical, so it checks 'decimals'
    # In @dataclass(order=True), higher decimals value means 'greater'
    assert usd >= usd_0
    assert usd_0 <= usd

    # Test inequality via __ge__
    # 'GBP' is alphabetically smaller than 'USD'
    assert not (gbp >= usd)
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest
from decimal import Decimal

def test_Currency___ge__():
    """
    Tests the __ge__ (greater than or equal) magic method of the Currency class.
    Since Currency is a dataclass with order=True, it implements __ge__ 
    based on the order of fields in the class definition.
    The field order is: code, name, decimals, type, quantizer, hashcache.
    """
    # Setup currencies with different attributes to test comparison logic
    # Currency 1: Base currency
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    # Currency 2: Same as USD but with a 'greater' code alphabetically ('USDT')
    usdt = Currency.of("USDT", "Tether", 2, CurrencyType.CRYPTO)
    
    # Currency 3: Same code as USD but 'greater' name ('US Dollars Plus')
    usd_plus = Currency.of("USD", "US Dollars Plus", 2, CurrencyType.MONEY)
    
    # Currency 4: Same code and name as USD but 'greater' decimals (3 instead of 2)
    usd_3dec = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    
    # Currency 5: Identical to USD
    usd_identical = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)

    # Test 1: Equality (Greater than or equal should be True)
    assert usd >= usd_identical
    assert usd_identical >= usd

    # Test 2: Alphabetical comparison on 'code' (USDT > USD)
    assert usdt > usd
    assert usdt >= usd
    assert usd < usdt
    assert not usd >= usdt

    # Test 3: Comparison on 'name' when 'code' is equal (US Dollars Plus > US Dollars)
    assert usd_plus > usd
    assert usd_plus >= usd
    assert usd < usd_plus
    assert not usd >= usd_plus

    # Test 4: Comparison on 'decimals' when 'code' and 'name' are equal (3 > 2)
    assert usd_3dec > usd
    assert usd_3dec >= usd
    assert usd < usd_3dec
    assert not usd >= usd_3dec

    # Test 5: Comparison on 'type' (Comparing Enum values)
    # Note: CurrencyType.MONEY is "Money", CurrencyType.CRYPTO is "Crypto Currency"
    # In Python Enums, comparison depends on the order of definition or value.
    # However, for dataclass order=True, it uses the Enum members.
    gold = Currency.of("XAU", "Gold", 2, CurrencyType.METAL)
    # XAU comes after USD alphabetically
    assert gold > usd
```


# LLM-generated content at query #16
#--------------------------

```python
def test_CurrencyLookupError():
    code = "XYZ"
    error = CurrencyLookupError(code)
    
    assert error.code == code
    assert str(error) == f"Currency identified by code '{code}' does not exist"
    assert isinstance(error, LookupError)
```


# LLM-generated content at query #17
#--------------------------

```python
import pytest
from unittest.mock import patch

def test_CurrencyRegistry():
    """
    Tests the constructor and singleton behavior of CurrencyRegistry.
    """
    # Since CurrencyRegistry is a singleton, we must reset the internal 
    # __instance to ensure a clean state for the test.
    with patch("pypara.currencies.CurrencyRegistry._CurrencyRegistry__instance", None):
        registry = CurrencyRegistry()
        
        # Test initialization of attributes
        assert registry._CurrencyRegistry__registry == {}
        assert registry._CurrencyRegistry__currencies == []
        assert registry._CurrencyRegistry__codes == []
        assert registry._CurrencyRegistry__codenames == []
        assert registry._CurrencyRegistry__ctx_open is False

        # Test singleton behavior: second instantiation returns the same object
        registry_second = CurrencyRegistry()
        assert registry is registry_second

        # Test that the registry is an instance of CurrencyRegistry
        assert isinstance(registry, CurrencyRegistry)

# Note: The above assumes the module structure allows patching the private __instance.
# In a real environment, one might need to reset the singleton via a dedicated 
# test utility if the class does not provide a reset mechanism.
```


# LLM-generated content at query #18
#--------------------------

```python
import pytest
from decimal import Decimal

def test_CurrencyRegistry_get():
    """
    Tests the 'get' method of CurrencyRegistry for existing,
    non-existing, and default value scenarios.
    """
    # Reset singleton instance for clean testing environment
    CurrencyRegistry.__instance = None
    registry = CurrencyRegistry()
    
    # Define test currencies
    usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEX) if hasattr(CurrencyType, 'MONEX') else Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    fallback = Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY)

    # Use the registry context to register currencies
    with registry as register:
        register(usd)
        register(eur)

    # 1. Test retrieving an existing currency
    retrieved_usd = registry.get("USD")
    assert retrieved_usd == usd
    assert retrieved_usd.code == "USD"

    # 2. Test retrieving a non-existing currency without a default
    # The method signature implies it returns None if not found and no default is provided
    assert registry.get("JPY") is None

    # 3. Test retrieving a non-existing currency with a provided default
    retrieved_fallback = registry.get("JPY", default=fallback)
    assert retrieved_fallback == fallback
    assert retrieved_fallback.code == "GBP"

    # 4. Test retrieving an existing currency with a default (should still return the existing one)
    retrieved_with_default = registry.get("EUR", default=fallback)
    assert retrieved_with_default == eur
```


# LLM-generated content at query #19
#--------------------------

```python
import pytest
from decimal import Decimal

def test_Currency___le__():
    """
    Tests the __le__ (less than or equal) method of the Currency class.
    Since the class is decorated with @dataclass(order=True), 
    the comparison follows the order of fields defined in the class:
    code, name, decimals, type, quantizer, hashcache.
    """
    # Setup base currency
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    # 1. Test Equality (le should be True when objects are equal)
    usd_duplicate = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd <= usd_duplicate
    assert usd_duplicate <= usd

    # 2. Test Less Than (based on 'code' field)
    # 'EUR' comes before 'USD' alphabetically
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert eur < usd
    assert eur <= usd
    assert not usd <= eur

    # 3. Test Less Than (based on 'name' field when codes are equal)
    # 'US Dollars' vs 'US Dollars (Alt)'
    usd_alt = Currency.of("USD", "US Dollars (Alt)", 2, CurrencyType.MONEY)
    assert usd < usd_alt
    assert usd_alt > usd

    # 4. Test Less Than (based on 'decimals' field when code and name are equal)
    # 0 decimals < 2 decimals
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyaryType.MONEY) # Note: using existing logic
    # We need to be careful with the 'of' constraints, but 'JPY' is valid.
    # If 'JPY' has the same code/name as a hypothetical 'JPY' with 2 decimals:
    jpy_2_dec = Currency.of("JPY", "Japanese Yen", 2, CurrencyType.MONEY)
    # Note: In the provided class, 'code' is the first field. 
    # To test decimals, we must have identical code and name.
    # Since 'of' validates code is uppercase and alpha, we can use 'JPY'
    # but we must ensure the name is identical to trigger the decimals check.
    
    # Re-creating a scenario where code and name are identical but decimals differ:
    # This is tricky because the 'of' method might not allow different decimals for same code/name 
    # if we assume code/name uniqueness, but the class allows it.
    
    # Let's use a custom field comparison logic check:
    # We'll use 'ABC' as a code.
    c1 = Currency.of("ABC", "Alpha", 1, CurrencyType.MONEY)
    c2 = Currency.of("ABC", "Alpha", 2, CurrencyType.MONEY)
    
    assert c1 < c2
    assert c2 > c1
    assert c1 <= c2

    # 5. Test Type Error
    # Comparing Currency with a non-Currency object should ideally raise TypeError 
    # or return False depending on implementation. Since it's a dataclass(order=True),
    # comparing with incompatible types usually raises TypeError.
    with pytest.raises(TypeError):
        assert usd <= "Not a currency"
```


# LLM-generated content at query #20
#--------------------------

```python
import pytest
from decimal import Decimal

def test_Currency___ge__():
    """
    Tests the __ge__ (greater than or equal) magic method of the Currency class.
    Since the class is decorated with @dataclass(order=True), the order is 
    determined by the order of fields in the class definition.
    """
    # Define base currency for comparison
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    # 1. Test Equality (GE should be true if objects are equal)
    usd_duplicate = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd >= usd_duplicate
    
    # 2. Test Greater Than (Based on field order: code, name, decimals, type, quantizer, hashcache)
    # 'USD' is the same, but 'US Dollars' vs 'US Dollars Premium'
    # 'US Dollars' < 'US Dollars Premium' alphabetically
    usd_premium = Currency.of("USD", "US Dollars Premium", 2, Currencyty.MONEY)
    assert usd_premium >= usd
    assert not usd >= usd_premium

    # 3. Test Comparison via code (The first field in the dataclass)
    # 'EUR' > 'USD' is False ('E' < 'U')
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert usd >= eur
    assert not eur >= usd

    # 4. Test comparison with different decimals (The third field)
    # If codes and names are identical, check decimals
    jpy = Currency.of("USD", "US Dollars", 0, CurrencyType.MONEY)
    assert usd > jpy
    assert jpy < usd

    # 5. Test type safety (Comparing with non-Currency object)
    # dataclass order=True uses the same logic as __gt__, __lt__, etc.
    # Comparing Currency with a string should raise TypeError
    with pytest.raises(TypeError):
        _ = usd >= "Not a currency"
```


# LLM-generated content at query #21
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_CurrencyRegistry___enter__():
    """
    Tests the __enter__ method of CurrencyRegistry.
    It should set the context as open and return the __register method.
    """
    # Reset singleton instance for a clean test environment
    CurrencyRegistry._CurrencyRegistry__instance = None
    registry = CurrencyRegistry()
    
    # Initially, the context should be closed (internal flag check)
    # Since __ctx_open is private, we check the side effect: 
    # __register should raise ProgrammingError if called outside context.
    dummy_currency = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    with pytest.raises(ProgrammingError, match="Can not create currencies outside registry context."):
        registry.__register(dummy_currency)

    # Enter the context
    register_func = registry.__enter__()

    # 1. Check if the returned object is the __register method
    assert callable(register_func)
    assert register_func == registry.__register

    # 2. Check if the context is now open by attempting to register a currency
    # This should NOT raise a ProgrammingError now
    try:
        register_func(dummy_currency)
    except ProgrammingError:
        pytest.fail("__enter__ did not open the registry context.")

    # 3. Verify that the registration actually occurred
    assert dummy_currency.code in registry.codes
```


####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from decimal import Decimal

def test_CurrencyRegistry___getitem__():
    """
    Tests the __getitem__ method of CurrencyRegistry for both
    successful retrieval and raising CurrencyLookupError.
    """
    # Reset singleton instance for clean test environment
    CurrencyRegistry.__instance = None
    registry = CurrencyRegistry()
    
    # Create a test currency
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONode)
    
    # We need to use the context manager to register the currency
    # as per the class implementation requirements
    with registry as register:
        register(usd)
    
    # Test successful retrieval
    assert registry["USD"] == usd
    assert registry["USD"].code == "USD"
    
    # Test retrieval of non-existent code raises CurrencyLookupError
    with pytest.raises(CurrencyLookupError) as excinfo:
        _ = registry["NON_EXISTENT"]
    
    assert "Currency identified by code 'NON_EXISTENT' does not exist" in str(excinfo.value)
    assert excinfo.value.code == "NON_EXISTENT"
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from decimal import Decimal

def test_Currency___eq__():
    """
    Tests the __eq__ implementation of the Currency class.
    Ensures that equality is based on the hashcache (which represents the object's identity 
    in this implementation) and handles comparisons with non-Currency objects.
    """
    # Setup valid currencies
    usd_params = ("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd1 = Currency.of(*usd_params)
    usd2 = Currency.of(*usd_params)
    
    # Different name, same code/decimals/type (should be unequal because hash includes name)
    usd_diff_name = Currency.of("USD", "US Dollars Modified", 2, CurrencyType.MONEY)
    
    # Different decimals (should be unequal)
    usd_diff_decimals = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    
    # Different type (should be unequal)
    usd_diff_type = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)

    # 1. Test equality of identical currency definitions
    assert usd1 == usd2
    
    # 2. Test inequality of different currency definitions
    assert usd1 != usd_diff_name
    assert usd1 != usd_diff_decimals
    assert usd1 != usd_diff_type
    
    # 3. Test equality with different objects that have the same hash/content
    # Since Currency.of computes hash based on the tuple, two separate instances 
    # with same params must be equal.
    assert hash(usd1) == hash(usd2)
    
    # 4. Test equality with non-Currency objects (should return False, not raise error)
    assert usd1 != "USD"
    assert usd1 != 123
    assert usd1 != None
    
    # 5. Test inequality with a different class that might have same attributes
    @dataclass(frozen=True)
    class MockCurrency:
        code: str
        name: str
        decimals: int
        type: CurrencyType
        quantizer: Decimal
        hashcache: int

    mock_currency = MockCurrency(
        code="USD", 
        name="US Dollars", 
        decimals=2, 
        type=CurrencyType.MONEY, 
        quantizer=Decimal("0.01"), 
        hashcache=hash(usd1)
    )
    # __eq__ checks isinstance(other, Currency), so this should be False
    assert usd1 != mock_currency
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from decimal import Decimal

def test_Currency___eq__():
    # Setup valid currencies
    usd_params = ("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd1 = Currency.of(*usd_params)
    usd2 = Currency.of(*usd_params)
    
    # Different name, same code/decimals/type (should be False due to hashcache calculation)
    usd_alt_name = Currency.of("USD", "US Dollars Alt", 2, CurrencyType.MONEY)
    
    # Different decimals
    usd_zero_dec = Currency.of("USD", "US Dollars", 0, CurrencyType.MONEY)
    
    # Different type
    usd_crypto = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    
    # Different code
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    
    # Test Equality
    assert usd1 == usd2, "Identical currencies should be equal"
    
    # Test Inequality with different attributes
    assert usd1 != usd_alt_name, "Currencies with different names should not be equal"
    assert usd1 != usd_zero_dec, "Currencies with different decimals should not be equal"
    assert usd1 != usd_crypto, "Currencies with different types should not be equal"
    assert usd1 != eur, "Currencies with different codes should not be equal"
    
    # Test Equality with different object types
    assert usd1 != "USD", "Currency should not be equal to a string"
    assert usd1 != 123, "Currency should not be equal to an integer"
    assert usd1 != None, "Currency should not be equal to None"
    
    # Test Hash Consistency
    assert hash(usd1) == hash(usd2), "Hash must be identical for equal currencies"
    assert hash(usd1) != hash(eur), "Hash must differ for different currencies"
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from decimal import Decimal

def test_Currency___eq__():
    # Setup valid currencies
    usd_1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd_2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    # Different name, same code/decimals/type (should be False due to hash mismatch)
    usd_alt_name = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    
    # Different decimals
    usd_zero_dec = Currency.of("USD", "US Dollars", 0, CurrencyType.MONEY)
    
    # Different type
    usd_crypto = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    
    # Different code
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    
    # Test equality
    assert usd_1 == usd_2
    
    # Test inequality with different attributes
    assert usd_1 != usd_alt_name
    assert usd_1 != usd_zero_dec
    assert usd_1 != usd_crypto
    assert usd_1 != eur
    
    # Test equality with non-Currency types
    assert usd_1 != "USD"
    assert usd_1 != 123
    assert usd_1 != None
    
    # Test hash consistency
    assert hash(usd_1) == hash(usd_2)
    assert hash(usd_1) != hash(usd_alt_name)
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from decimal import Decimal

def test_Currency___eq__():
    """
    Tests the __eq__ method of the Currency class to ensure it correctly
    compares currency objects based on their hashcache (representing identity).
    """
    # Setup common parameters
    code = "USD"
    name = "US Dollars"
    decimals = 2
    ctype = CurrencyType.MONEY

    # Create identical currency objects
    usd1 = Currency.of(code, name, decimals, ctype)
    usd2 = Currency.of(code, name, decimals, ctype)
    
    # Create a currency object with same code/decimals but different name
    # Note: According to the implementation, __eq__ relies on hashcache.
    # hashcache is computed from (code, name, decimals, ctype, quantizer).
    # Therefore, if name differs, hashcache differs.
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    
    # Create a currency object with different type
    usdy = Currency.of("USD", "US Dollars", 2, CurrencyType.ALTERNATIVE)

    # Test equality of identical objects
    assert usd1 == usd2
    
    # Test inequality with different names
    assert usd1 != usdx
    
    # Test inequality with different types
    assert usd1 != usdy
    
    # Test equality with different object references but same values (via hashcache)
    assert usd1 == usd2
    
    # Test inequality with non-Currency types
    assert usd1 != "USD"
    assert usd1 != 123
    assert usd1 != None

    # Test hash consistency
    assert hash(usd1) == hash(usd2)
    assert hash(usd1) != hash(usdx)
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest

def test_CurrencyRegistry_has():
    """
    Tests the 'has' method of the CurrencyRegistry class.
    """
    # Reset Singleton instance for clean testing environment
    CurrencyRegistry._instance = None
    registry = CurrencyRegistry()
    
    # Create test currencies
    usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    
    # Test 'has' when registry is empty
    assert registry.has("USD") is False
    
    # Populate registry using the context manager
    with registry as register:
        register(usd)
        register(jpy)
    
    # Test 'has' for registered currencies
    assert registry.has("USD") is True
    assert registry.has("JPY") is True
    
    # Test 'has' for non-registered currency
    assert registry.has("EUR") is False
    
    # Test 'has' for invalid types/formats (should still return False if not in registry)
    assert registry.has("") is False
    assert registry.has(None) is False
    assert registry.has(123) is False
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
from decimal import Decimal

def test_CurrencyRegistry___exit__():
    """
    Tests the __exit__ method of CurrencyRegistry to ensure it correctly 
    finalizes the registry by sorting and updating internal buffers 
    after a context manager block completes.
    """
    # Reset the singleton instance for a clean test environment
    CurrencyRegistry._CurrencyRegistry__instance = None
    registry = CurrencyRegistry()

    # Define some currencies out of order
    ccy_usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    ccy_aed = Currency.of("AED", "UAE Dirham", 2, CurrencyType.MONEY)
    ccoi_jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)

    # Use the context manager to register currencies
    with registry as register:
        # Registering in non-alphabetical order
        register(ccy_usd)
        register(ccoi_jpy)
        register(ccy_aed)
        
        # While inside the context, the internal buffers might not be finalized/sorted
        # depending on the implementation of __register, but we primarily test __exit__
    
    # After __exit__, the registry should be sorted by code: AED, JPY, USD
    
    # 1. Test that the 'all' list is sorted by code
    assert registry.all == [ccy_aed, ccoi_jpy, ccy_usd]
    
    # 2. Test that 'codes' property is sorted
    assert registry.codes == ["AED", "JPY", "USD"]
    
    # 3. Test that 'codenames' property is sorted and contains correct tuples
    expected_codenames = [("AED", "UAE Dirham"), ("JPY", "Japanese Yen"), ("USD", "US Dollar")]
    assert registry.codenames == expected_codenames
    
    # 4. Verify the internal registry order via __getitem__
    assert registry["AED"].code == "AED"
    assert registry["JPY"].code == "JPY"
    assert registry["USD"].code == "USD"

    # 5. Verify that the context flag is closed (cannot register without context)
    with pytest.raises(ProgrammingError, match="Can not create currencies outside registry context."):
        registry.__register(Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY))
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
from decimal import Decimal

def test_Currency_quantize():
    # Test Case 1: Standard Money Currency (USD) - 2 decimals
    # Uses ROUND_HALF_EVEN (default context behavior)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.quantize(Decimal("1.005")) == Decimal("1.00")
    assert usd.quantize(Decimal("1.015")) == Decimal("1.02")
    assert usd.quantize(Decimal("1.1234")) == Decimal("1.12")

    # Test Case 2: Zero decimal currency (JPY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy.quantize(Decimal("0.5")) == Decimal("0")
    assert jpy.quantize(Decimal("1.5")) == Decimal("2")
    assert jpy.quantize(Decimal("1.9")) == Decimal("2")

    # Test Case 3: Negative decimal currency (Crypto/Special)
    # Uses MaxPrecisionQuantizer logic
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    # Testing very small increments to verify precision handling
    assert zzz.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert zzz.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")

    # Test Case 4: High precision decimals
    gbp = Currency.of("GBP", "British Pound", 4, CurrencyType.MONEY)
    assert gbp.quantize(Decimal("1.123456")) == Decimal("1.1235")
    assert gbp.quantize(Decimal("1.123444")) == Decimal("1.1234")
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from decimal import Decimal

def test_Currency___le__():
    """
    Tests the __le__ (less than or equal) magic method of the Currency class.
    Since Currency is a dataclass with order=True, it implements comparison 
    operators based on the order of fields in the class definition.
    The field order is: code, name, decimals, type, quantizer, hashcache.
    """
    # Base currency for comparison
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    # 1. Test equality (less than or equal should be True)
    # We create an identical instance via .of()
    usd_duplicate = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd <= usd_duplicate
    assert usd_duplicate <= usd

    # 2. Test less than (based on 'code' field)
    # 'ABC' comes before 'USD' alphabetically
    abc = Currency.of("ABC", "Alpha Currency", 2, CurrencyType.MONEY)
    assert abc < usd
    assert usd > abc

    # 3. Test less than (based on 'name' field)
    # Same code, but name 'AAA' comes before 'US Dollars'
    aaa = Currency.of("USD", "AAA Currency", 2, CurrencyType.MONEY)
    assert aaa < usd
    assert usd > aaa

    # 4. Test less than (based on 'decimals' field)
    # Same code and name, but 0 decimals comes before 2
    jpy = Currency.of("USD", "US Dollars", 0, CurrencyType.MONEY)
    assert jpy < usd
    assert usd > jpy

    # 5. Test less than (based on 'type' field)
    # CurrencyType is an Enum. Comparison depends on Enum member order.
    # Order: MONEY, METAL, CRYPTO, ALTERNATIVE
    crypto_usd = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert usd < crypto_usd
    assert crypto_usd > usd

    # 6. Test error handling for incompatible types
    with pytest.raises(TypeError):
        _ = usd <= "Not a Currency object"
```


# LLM-generated content at query #10
#--------------------------

```python
def test_CurrencyLookupError():
    code = "XYZ"
    error = CurrencyLookupError(code)
    
    assert error.code == code
    assert str(error) == f"Currency identified by code '{code}' does not exist"
    assert isinstance(error, LookupError)
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest

def test_CurrencyRegistry___new__():
    """
    Tests that the __new__ method correctly implements the Singleton pattern
    for the CurrencyRegistry class.
    """
    # First instance creation
    registry1 = CurrencyRegistry()
    
    # Second instance creation
    registry2 = CurrencyRegistry()
    
    # Check that both references point to the exact same object in memory
    assert registry1 is registry2
    
    # Check that a third instance also points to the same object
    registry3 = CurrencyRegistry()
    assert registry1 is registry3
    
    # Verify that the instance attribute is indeed stored in the class
    assert CurrencyRegistry._CurrencyRegistry__instance is registry1
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
from decimal import Decimal

def test_Currency():
    # Test valid creation of various Currency types
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.code == "USD"
    assert usd.name == "US Dollars"
    assert usd.decimals == 2
    assert usd.type == CurrencyType.MONEY
    assert usd.quantizer == Decimal("0.01")

    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy.decimals == 0
    assert jpy.quantizer == Decimal("1")

    crypto = Currency.of("BTC", "Bitcoin", -1, CurrencyType.CRYPTO)
    assert crypto.decimals == -1
    # MaxPrecisionQuantizer is used for decimals < 0

    # Test equality and hashing
    usd_clone = Currency.of("USD", "US Dollars", 2, CurrencylyType.MONEY)
    assert usd == usd_clone
    assert hash(usd) == hash(usd_clone)

    # Test inequality (different name)
    usd_diff_name = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    assert usd != usd_diff_name

    # Test quantization logic
    assert usd.quantize(Decimal("1.005")) == Decimal("1.00")
    assert usd.quantize(Decimal("1.015")) == Decimal("1.02")
    assert jpy.quantize(Decimal("1.5")) == Decimal("2")
    assert jpy.quantize(Decimal("0.5")) == Decimal("0")

    # Test Validation: Code errors
    with pytest.raises(ProgrammingError):
        Currency.of("usd", "US Dollars", 2, CurrencyType.MONEY)  # Not uppercase
    with pytest.raises(ProgrammingError):
        Currency.of("U1D", "US Dollars", 2, CurrencyType.MONEY)  # Not alpha
    with pytest.raises(ProgrammingError):
        Currency.of(123, "US Dollars", 2, CurrencyType.MONEY)    # Not string

    # Test Validation: Name errors
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "", 2, CurrencyType.MONEY)            # Empty name
    with pytest.raises(ProgrammingError):
        Currency.of("USD", " US Dollars", 2, CurrencyType.MONEY) # Leading space
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars ", 2, CurrencyType.MONEY) # Trailing space

    # Test Validation: Decimals errors
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", -2, CurrencyType.MONEY) # Less than -1
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", "2", CurrencyType.MONEY) # Not int

    # Test Validation: Type errors
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", 2, "MONEY")             # Not CurrencyType enum
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest
from decimal import Decimal

def test_Currency___hash__():
    """
    Tests that the __hash__ method of the Currency class returns the pre-computed hashcache
    and that identical currency definitions result in the same hash.
    """
    # Define attributes for two identical currencies
    code = "USD"
    name = "US Dollars"
    decimals = 2
    ctype = CurrencyType.MONEY

    # Create two separate instances that are logically identical
    usd1 = Currency.of(code, name, decimals, ctype)
    usd2 = Currency.of(code, name, decimals, ctype)
    
    # Create an instance that is different
    gbp = Currency.of("GBP", "British Pounds", 2, CurrencyType.MONEY)

    # 1. Test that hash of identical objects is the same
    assert hash(usd1) == hash(usd2)
    
    # 2. Test that hash is consistent with the internal hashcache attribute
    assert hash(usd1) == usd1.hashcache
    
    # 3. Test that different objects have different hashes
    # (Note: while hash collisions are theoretically possible, in this context 
    # the components used for the hash are unique enough to expect different values)
    assert hash(usd1) != hash(gbp)

    # 4. Test that hash works correctly in a set (demonstrates property of __hash__)
    currency_set = {usd1, usd2, gbp}
    assert len(currency_set) == 2
    assert usd1 in currency_set
    assert gbp in currency_set
```


# LLM-generated content at query #14
#--------------------------

```python
def test_Currency___repr__():
    """
    Tests the __repr__ method of the Currency class.
    Note: Since the provided code does not explicitly implement __repr__, 
    this test verifies the default behavior of a dataclass __repr__.
    """
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    # The default dataclass __repr__ includes the class name and all field values
    repr_str = repr(usd)
    
    assert repr_str.startswith("Currency(")
    assert "code='USD'" in repr_str
    assert "name='US Dollars'" in repr_str
    assert "decimals=2" in repr_str
    assert "type=CurrencyType.MONEY" in repr_str
    assert "quantizer=" in repr_str
    assert "hashcache=" in repr_str
    assert repr_str.endswith(")")
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest
from decimal import Decimal

def test_CurrencyRegistry_get():
    """
    Tests the 'get' method of CurrencyRegistry for:
    1. Retrieving an existing currency.
    2. Returning the default value when a currency is not found.
    3. Returning None when a currency is not found and no default is provided.
    """
    # Reset singleton instance for clean testing environment
    CurrencyRegistry._instance = None
    registry = CurrencyRegistry()
    
    # Create test currencies
    usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEX)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    
    # Use the registry context to populate
    with registry as register:
        register(usd)
        register(eur)
        
    # 1. Test retrieving an existing currency
    assert registry.get("USD") == usd
    assert registry.get("EUR") == eur
    
    # 2. Test returning the default value when a currency is not found
    default_currency = Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY)
    assert registry.get("GBP", default=default_currency) == default_currency
    
    # 3. Test returning None when a currency is not found and no default is provided
    assert registry.get("JPY") is None
    assert registry.get("NON_EXISTENT") is None
```


# LLM-generated content at query #16
#--------------------------

```python
def test_Currency___hash__():
    """
    Tests the __hash__ method of the Currency class to ensure it returns
    the pre-computed hashcache and maintains consistency.
    """
    # Setup identical currencies
    usd_params = ("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd1 = Currency.of(*usd_params)
    usd2 = Currency.of(*usd_params)
    
    # Setup different currency
    jpy_params = ("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    jpy = Currency.of(*jpy_params)

    # Test that hash returns the internal hashcache
    assert hash(usd1) == usd1.hashcache
    assert hash(usd2) == usd2.hashcache
    
    # Test that identical objects produce the same hash
    assert hash(usd1) == hash(usd2)
    
    # Test that different objects produce different hashes
    assert hash(usd1) != hash(jpy)
    
    # Test that the hash is consistent with the equality logic used in the class
    # (The __eq__ implementation relies on hashcache)
    assert hash(usd1) == hash(Currency(
        code=usd1.code,
        name=usd1.name,
        decimals=usd1.decimals,
        type=usd1.type,
        quantizer=usd1.quantizer,
        hashcache=usd1.hashcache
    ))
```


# LLM-generated content at query #17
#--------------------------

```python
def test_Currency___repr__():
    """
    Tests the __repr__ method of the Currency class.
    Note: Since the provided code does not explicitly implement __repr__, 
    it relies on the default dataclass implementation.
    """
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    # The default dataclass __repr__ follows the pattern: 
    # ClassName(attr1=val1, attr2=val2, ...)
    # We check if the string representation contains the key attributes.
    repr_str = repr(usd)
    
    assert "Currency" in repr_str
    assert "code='USD'" in repr_str
    assert "name='US Dollars'" in repr_str
    assert "decimals=2" in repr_str
    assert "type=CurrencyType.MONEY" in repr_str
```


# LLM-generated content at query #18
#--------------------------

```python
import pytest

def test_CurrencyRegistry___contains__():
    # Reset the singleton instance for a clean test environment
    CurrencyRegistry._instance = None
    registry = CurrencyRegistry()
    
    # Create a dummy currency for testing
    usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    gbp = Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY)

    # Test: __contains__ should return False when registry is empty
    assert "USD" not in registry
    
    # Test: __contains__ should return True after registration via context manager
    with registry as register:
        register(usd)
        register(gbp)
    
    # Test: __contains__ returns True for registered codes
    assert "USD" in registry
    assert "GBP" in registry
    
    # Test: __contains__ returns False for unregistered codes
    assert "JPY" not in registry
    assert "XYZ" not in registry
    
    # Test: __contains__ with non-existent code
    assert "NON-EXISTENT" not in registry
```


# LLM-generated content at query #19
#--------------------------

```python
def test_CurrencyRegistry___new__():
    """
    Tests the singleton behavior of CurrencyRegistry.__new__.
    """
    instance1 = CurrencyRegistry()
    instance2 = CurrencyRegistry()
    
    # Verify that both calls return the exact same instance (singleton pattern)
    assert instance1 is instance2
    
    # Verify that the __instance class attribute is indeed the instance created
    assert CurrencyRegistry.__instance is instance1
```


# LLM-generated content at query #20
#--------------------------

```python
import pytest
from decimal import Decimal

def test_Currency_quantize():
    # Test Case 1: Standard Money Currency (USD) - 2 decimals
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.quantize(Decimal("1.005")) == Decimal("1.00")
    assert usd.quantize(Decimal("1.015")) == Decimal("1.02")
    assert usd.quantize(Decimal("1.5")) == Decimal("1.50")
    assert usd.quantize(Decimal("1")) == Decimal("1.00")

    # Test Case 2: Zero decimal currency (JPY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy.quantize(Decimal("0.5")) == Decimal("0")
    assert jpy.quantize(Decimal("1.5")) == Decimal("2")
    assert jpy.quantize(Decimal("1.1")) == Decimal("1")

    # Test Case 3: Negative decimal currency (Crypto/Special)
    # Using -1 to trigger MaxPrecisionQuantizer behavior as per docstring
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert zzz.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert zzz.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")

    # Test Case 4: High precision decimals
    gbp = Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY)
    assert gbp.quantize(Decimal("1.23456")) == Decimal("1.23")
    assert gbp.quantize(Decimal("1.23556")) == Decimal("1.24")

    # Test Case 5: Edge case with zero value
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert eur.quantize(Decimal("0")) == Decimal("0.00")
```


# LLM-generated content at query #21
#--------------------------

```python
import pytest
from decimal import Decimal

def test_Currency___gt__():
    """
    Tests the __gt__ (greater than) implementation for the Currency class.
    Note: Since the class uses @dataclass(order=True), the comparison 
    is based on the order of fields defined in the class.
    """
    # Define common parameters
    usd_params = ("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur_params = ("EUR", "Euro", 2, CurrencyType.MONEY)
    jpy_params = ("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    
    # Create currency instances
    # The order of fields in Currency is: code, name, decimals, type, quantizer, hashcache
    usd = Currency.of(*usd_params)
    eur = Currency.of(*eur_params)
    jpy = Currency.of(*jpy_params)
    
    # Test field-by-field comparison (Primary key is 'code')
    # 'USD' > 'EUR' is True because 'U' > 'E'
    assert usd > eur
    assert eur < usd
    
    # 'JPY' < 'USD' is True because 'J' < 'U'
    assert jpy < usd
    
    # Test equality (should not be greater than)
    usd_duplicate = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd > usd_duplicate)
    
    # Test with different names but same code (if possible, but code is the first field)
    # Since 'code' is the first field in the dataclass, it dominates the comparison.
    # We'll create a scenario where code is the same to test the second field 'name'
    # Note: Currency.of validates code.isalpha() and isupper().
    
    # We must bypass .of() to test the internal comparison logic of the dataclass 
    # if we want to compare same codes with different names, 
    # because .of() is used for creation and we want to see if 'name' is the tie-breaker.
    
    # Creating instances manually to control the fields for tie-breaking tests
    # We use the same quantizer and hashcache from a valid object
    common_quantizer = usd.quantizer
    common_hash = usd.hashcache
    
    c1 = Currency("USD", "Alpha", 2, CurrencyType.MONESS, common_quantizer, common_hash)
    c2 = Currency("USD", "Zeta", 2, CurrencyType.MONESS, common_quantizer, common_hash)
    
    # Since 'code' is same, 'name' "Zeta" > "Alpha"
    assert c2 > c1
    assert c1 < c2

    # Test failure with incomparable types (standard python behavior for dataclass order=True)
    with pytest.raises(TypeError):
        assert usd > "Not a Currency object"
```


# LLM-generated content at query #22
#--------------------------

```python
import pytest

def test_CurrencyRegistry():
    """
    Tests the constructor and singleton behavior of CurrencyRegistry.
    """
    # Reset singleton instance for a clean test environment
    CurrencyRegistry._CurrencyRegistry__instance = None
    
    # Test first instantiation
    registry1 = CurrencyRegistry()
    assert isinstance(registry1, CurrencyRegistry)
    
    # Test singleton property: second instantiation should return the same object
    registry2 = CurrencyRegistry()
    assert registry1 is registry2
    
    # Test initial state of the registry
    assert len(registry1) == 0
    assert registry1.all == []
    assert registry1.codes == []
    assert registry1.codenames == []
    
    # Test internal buffers are initialized as empty containers
    assert registry1._CurrencyRegistry__registry == {}
    assert registry1._CurrencyRegistry__currencies == []
    assert registry1._CurrencyRegistry__codes == []
    assert registry1._CurrencyRegistry__codenames == []
    assert registry1._CurrencyRegistry__ctx_open is False
```


# LLM-generated content at query #23
#--------------------------

```python
import pytest

def test_CurrencyRegistry():
    """
    Tests the constructor and initialization logic of the CurrencyRegistry singleton.
    """
    # Reset the singleton instance for a clean test environment
    # Since __instance is a class attribute, we clear it to ensure __init__ runs fresh
    CurrencyRegistry._CurrencyRegistry__instance = None
    
    registry = CurrencyRegistry()
    
    # Test Singleton property: creating a new instance returns the same object
    registry_second = CurrencyRegistry()
    assert registry is registry_second

    # Test initial state of the registry containers via internal access or public properties
    # Note: accessing private members is used here to verify the constructor's state
    assert len(registry) == 0
    assert registry.all == []
    assert registry.codes == []
    assert registry.codenames == []
    
    # Test that the context flag is initialized to False
    assert registry._CurrencyRegistry__ctx_open is False

    # Verify that the registry is empty and doesn't contain any random codes
    assert "USD" not in registry
    assert registry.has("USD") is False

    # Verify that the registry handles the singleton instance correctly across multiple calls
    # by checking if the internal dictionary is initialized as an OrderedDict
    assert isinstance(registry._CurrencyRegistry__registry, type(pytest.importorskip("collections").OrderedDict)())
```


# LLM-generated content at query #24
#--------------------------

```python
import pytest
from decimal import Decimal

def test_Currency___hash__():
    """
    Tests the __hash__ method of the Currency class to ensure it returns 
    the pre-computed hashcache and maintains consistency for identical objects.
    """
    # Setup identical currency objects
    usd_code = "USD"
    usd_name = "US Dollars"
    usd_decimals = 2
    usd_type = CurrencyType.MONEY
    
    usd1 = Currency.of(usd_code, usd_name, usd_decimals, usd_type)
    usd2 = Currency.of(usd_code, usd_name, usd_decimals, usd_type)
    
    # Test that hash is equal for identical objects
    assert hash(usd1) == hash(usd2)
    
    # Test that hash matches the internal hashcache attribute
    assert hash(usd1) == usd1.hashcache
    
    # Test that hash is different for different objects (different name)
    usd_alt = Currency.of("USD", "UX Dollars", usd_decimals, usd_type)
    assert hash(usd1) != hash(usd_alt)
    
    # Test that the hash is consistent across multiple calls
    first_call = hash(usd1)
    second_call = hash(usd1)
    assert first_call == second_call
```


