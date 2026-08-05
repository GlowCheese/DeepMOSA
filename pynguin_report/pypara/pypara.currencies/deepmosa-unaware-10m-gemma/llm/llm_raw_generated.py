####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from decimal import Decimal

def test_Currency___eq__():
    """
    Tests the equality logic of the Currency class, ensuring that two currencies 
    are considered equal only if they share the same hash (derived from their attributes)
    and that different currencies or non-currency types return False.
    """
    # Setup identical currencies
    usd_1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd_2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    # Setup different currency (different name/code/etc will result in different hash)
    usd_alt_name = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    gbp = Currency.of("GBP", "British Pounds", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    
    # Test equality of identical objects
    assert usd_1 == usd_2
    
    # Test inequality with different name but same code/decimals (if hash allows)
    # Based on the implementation: self.hashcache == other.hashcache
    # Since hash is computed via (code, name, decimals, ctype, quantizer), 
    # changing 'name' changes the hash.
    assert usd_1 != usd_alt_name
    
    # Test inequality with different currency codes
    assert usd_1 != gbp
    
    # Test inequality with different decimal precision
    assert usd_1 != jpy
    
    # Test equality with non-Currency types
    assert usd_1 != "USD"
    assert usd_1 != 123
    assert usd_1 != None

    # Test hash consistency for equality
    assert hash(usd_1) == hash(usd_2)
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from decimal import Decimal

def test_Currency___gt__():
    """
    Tests the __gt__ (greater than) implementation for the Currency class.
    Since the class is decorated with @dataclass(order=True), 
    it uses the order of fields defined in the dataclass: 
    code, name, decimals, type, quantizer, hashcache.
    """
    # Setup currencies with different attributes to test ordering
    # We use Currency.of to ensure valid object creation
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    gbp = Currency.of("GBP", "British Pounds", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    
    # Testing based on the first field 'code' (alphabetical order)
    # USD > GBP is False because 'U' > 'G' is True, but we check __gt__ logic
    assert usd > gbp  # 'USD' > 'GBP' is True
    assert gbp > jpy  # 'GBP' > 'JPY' is False (alphabetical)
    assert jpy < usd  # 'JPY' < 'USD' is True

    # Testing based on name if codes were identical (hypothetically)
    # Since code is the first field, we must manipulate logic or use direct comparison
    # In a dataclass(order=True), it compares tuple(fields)
    
    usd_same_code = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd_diff_name = Currency.of("USD", "United States Dollars", 2, CurrencyType.MONEY)
    
    # 'US Dollars' < 'United States Dollars' (alphabetical order of name field)
    assert usd_same_code < usd_diff_name
    assert usd_diff_name > usd_same_code

    # Testing based on decimals if code and name were identical
    usd_low_dec = Currency.of("USD", "US Dollars", 0, CurrencyType.MONEY)
    usd_high_dec = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    # 'USD', 'US Dollars', 2 > 'USD', 'US Dollars', 0
    assert usd_high_dec > usd_low_dec
    assert usd_low_dec < usd_high_dec

    # Testing with different types if code, name, and decimals are identical
    usd_alt_type = Currency.of("USD", "US Dollars", 2, CurrencyType.ALTERNATIVE)
    # Enum comparison: MONEY (value 'Money') vs ALTERNATIVE (value 'Alternative')
    # Note: The order depends on the string value of the enum or definition order.
    # In Currency.of, ctype is part of the hash/tuple. 
    # Since it's an Enum, comparison follows the order of members in the class.
    assert usd_alt_type < usd # ALTERNATIVE comes after MONEY in some contexts, 
                               # but here we test the logic of the tuple comparison.
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest

def test_CurrencyRegistry___len__():
    """
    Tests the __len__ method of CurrencyRegistry.
    Ensures it correctly reports the number of registered currencies 
    within a registry context.
    """
    # Reset singleton instance for clean testing environment
    CurrencyRegistry._CurrencyRegistry__instance = None
    registry = CurrencyRegistry()

    # Test length is 0 initially
    assert len(registry) == 0

    # Register some currencies within the context
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)

    with registry as register:
        register(usd)
        assert len(registry) == 1
        register(eur)
        assert len(registry) == 2
        register(jpy)
        assert len(registry) == 3

    # Verify length remains consistent after context exit (sorting/finalization)
    assert len(registry) == 3
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from decimal import Decimal

def test_Currency():
    # Test valid creation using factory method .of()
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.code == "USD"
    assert usd.name == "US Dollars"
    assert usd.decimals == 2
    assert usd.type == CurrencyType.MONEY
    assert isinstance(usd.quantizer, Decimal)

    # Test different decimal precisions (Quantization logic)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy.quantize(Decimal("1.5")) == Decimal("2")
    assert jpy.quantize(Decimal("1.4")) == Decimal("1")

    # Test negative decimals (MaxPrecisionQuantizer behavior via .of)
    zzz = Currency.of("ZZZ", "Crypto", -1, CurrencyType.CRYPTO)
    assert zzz.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")

    # Test equality and hashing
    usd_duplicate = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd_different_name = Currency.of("USD", "Different Name", 2, CurrencyType.MONEY)
    
    assert usd == usd_duplicate
    assert usd != usd_different_name
    assert hash(usd) == hash(usd_duplicate)

    # Test validation errors (ProgrammingError via passert)
    with pytest.raises(Exception): # Specifically looking for the error raised by ProgrammingError.passert
        Currency.of("usd", "Lower Case", 2, CurrencyType.MONEY)
    
    with pytest.raises(Exception):
        Currency.of("USD", " Leading Space", 2, CurrencyType.MONEY)

    with pytest.raises(Exception):
        Currency.of("USD", "Trailing Space ", 2, CurrencyType.MONEY)

    with pytest.raises(Exception):
        Currency.of("USD1", "Has Numbers", 2, CurrencyType.MONEY)

    with pytest.raises(Exception):
        Currency.of("USD", "", 2, CurrencyType.MONEY)

    with pytest.raises(Exception):
        Currency.of("USD", "US Dollars", -2, CurrencyType.MONEY)

    with pytest.raises(Exception):
        Currency.of("USD", "US Dollars", 2, "NotAType")

def test_CurrencyLookupError():
    error = CurrencyLookupError("XYZ")
    assert error.code == "XYZ"
    assert "XYZ" in str(error)

def test_CurrencyTypeEnum():
    assert CurrencyType.MONEY.value == "Money"
    assert CurrencyType.CRYPTO.value == "Crypto Currency"
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from decimal import Decimal

def test_Currency___repr__():
    """
    Tests the __repr__ method of the Currency class.
    Note: Since @dataclass(frozen=True) is used, a default __repr__ is generated 
    by the dataclass decorator unless explicitly overridden.
    """
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    # The default dataclass repr includes all fields
    expected_repr = (
        "Currency("
        f"code={usd.code!r}, "
        f"name={usd.name!r}, "
        f"decimals={usd.decimals!r}, "
        f"type={usd.type!r}, "
        f"quantizer={usd.quantizer!r}, "
        f"hashcache={usd.hashcache!r})"
    )
    
    assert repr(usd) == expected_repr
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest
from decimal import Decimal

def test_Currency___lt__():
    """
    Tests the __lt__ (less than) implementation of the Currency class.
    Since the class is decorated with @dataclass(order=True), 
    the comparison follows the order of fields defined in the dataclass:
    code, name, decimals, type, quantizer, hashcache.
    """
    # Base currency for comparisons
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    # 1. Test primary field: code (alphabetical order)
    # 'ABC' < 'USD'
    abc_ccy = Currency.of("ABC", "Alpha Currency", 2, CurrencyType.MONEY)
    assert abc_ccy < usd
    assert not usd < abc_ccy

    # 2. Test secondary field: name (alphabetical order)
    # Same code 'USD', but different name 'US Dollars' vs 'US X Dollars'
    usdx = Currency.of("USD", "US X Dollars", 2, CurrencyType.MONEY)
    assert usdx < usd  # 'US X...' comes before 'US D...' is False, actually 'US D' < 'US X'
    # Correction: 'US Dollars' < 'US X Dollars'
    assert usd < usdx
    
    # 3. Test tertiary field: decimals (numerical order)
    # Same code and name, but different decimals
    usd_zero_dec = Currency.of("USD", "US Dollars", 0, CurrencyType.MONEY)
    assert usd_zero_dec < usd

    # 4. Test quaternary field: type (Enum order)
    # Note: Enum comparison depends on definition order in CurrencyType
    # MONEY is first, METAL is second...
    usd_metal = Currency.of("USD", "US Dollars", 2, CurrencyType.METAL)
    assert usd < usd_metal

    # 5. Test edge case: Identity
    # A currency cannot be less than itself
    assert not (usd < usd)

    # 6. Verification of the underlying tuple comparison logic used by @dataclass(order=True)
    # The order is (code, name, decimals, type, quantizer, hashcache)
    # We test a very specific sequence
    c1 = Currency.of("A", "Name", 2, CurrencyType.MONEY)
    c2 = Currency.of("B", "Name", 2, CurrencyType.MONEY)
    assert c1 < c2

    c3 = Currency.of("A", "Z Name", 2, CurrencyType.MONEY)
    assert c1 < c3

    c4 = Currency.of("A", "Name", 1, CurrencyType.MONEY)
    assert c4 < c1
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
from decimal import Decimal

def test_Currency___delattr__():
    """
    Tests that attempting to delete an attribute from a frozen Currency instance
    raises a FrozenInstanceError (via AttributeError).
    """
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    # Since the dataclass is decorated with @dataclass(frozen=True),
    # any attempt to delete an attribute should raise AttributeError.
    with pytest.raises(AttributeError):
        del usd.code

    with pytest.raises(AttributeError):
        del usd.name

    with pytest.raises(AttributeError):
        del usd.decimals

    with pytest.raises(AttributeError):
        del usd.type
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
from decimal import Decimal

def test_Currency___lt__():
    """
    Tests the __lt__ (less than) implementation of the Currency class.
    Since Currency is a frozen dataclass with order=True, it implements 
    comparison based on the order of fields defined in the dataclass.
    The fields are: code, name, decimals, type, quantizer, hashcache.
    """
    # Setup currencies with varying attributes to test comparison logic
    # Note: We use .of() to ensure all internal properties like quantizer and hashcache are correctly set
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    gbp = Currency.of("GBP", "British Pounds", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    crypto = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    
    # Test: Code comparison (Primary sort key is 'code')
    # 'BTC' < 'GBP' < 'JPY' < 'USD'
    assert crypto < gbp
    assert gbp < jpy
    assert jpy < usd
    assert not usd < crypto

    # Test: Name comparison (If codes were identical, would check name)
    # We can't easily use .of() to create same code but different name because 
    # Currency.of validates the code is upper and alpha.
    # However, we can manually instantiate for edge case testing of order=True logic
    # specifically targeting the 'name' field if codes match.
    currency_a = Currency("AAA", "Alpha", 2, CurrencyType.MONEY, Decimal("0.01"), 1)
    currency_b = Currency("AAA", "Beta", 2, CurrencyTYpe.MONEY, Decimal("0.01"), 2)
    assert currency_a < currency_b

    # Test: Decimals comparison (If codes and names are identical)
    currency_low_dec = Currency("USD", "US Dollars", 0, CurrencyType.MONEY, Decimal("1"), 3)
    currency_high_dec = Currency("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal("0.01"), 4)
    assert currency_low_dec < currency_high_dec

    # Test: Type comparison (If code, name, and decimals are identical)
    # Enum comparison is based on member order/value
    currency_money = Currency("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal("0.01"), 5)
    currency_crypto = Currency("USD", "US Dollars", 2, CurrencyType.CRYPTO, Decimal("0.01"), 6)
    assert currency_money < currency_crypto

    # Test: Equality (not less than)
    assert not usd < usd
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from decimal import Decimal

def test_CurrencyRegistry___getitem__():
    """
    Tests the __getitem__ method of CurrencyRegistry.
    Verifies that it returns the correct Currency object for existing codes
    and raises CurrencyLookupError for non-existent codes.
    """
    # Reset singleton instance for clean testing environment
    CurrencyRegistry._instance = None
    registry = CurrencyRegistry()
    
    # Setup: Create and register currencies using the context manager
    usd_code = "USD"
    usd_currency = Currency.of(usd_code, "US Dollars", 2, CurrencyType.MONEY)
    eur_code = "EUR"
    eur_currency = Currency.of(eur_code, "Euro", 2, CurrencyType.MONEY)
    
    with registry as register:
        register(usd_currency)
        register(eur_currency)

    # Test case 1: Retrieve existing currency (USD)
    assert registry[usd_code] == usd_currency
    assert registry[usd_code].code == "USD"

    # Test case 2: Retrieve another existing currency (EUR)
    assert registry[eur_code] == eur_currency
    assert registry[eur_code].name == "Euro"

    # Test case 3: Retrieve non-existent currency (raises CurrencyLookupError)
    non_existent_code = "XYZ"
    with pytest.raises(CurrencyLookupError) as excinfo:
        _ = registry[non_existent_code]
    
    assert str(excinfo.value) == f"Currency identified by code '{non_existent_code}' does not exist"
    assert excinfo.value.code == non_existent_code
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
from decimal import Decimal

def test_Currency___gt__():
    """
    Tests the __gt__ (greater than) implementation for the Currency class.
    Note: Since the class is decorated with @dataclass(order=True), 
    the comparison order is determined by the order of fields in the dataclass definition:
    code, name, decimals, type, quantizer, hashcache.
    """
    # Create base currency
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    # Case 1: Test equality (not greater than)
    usd_copy = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd > usd_copy)

    # Case 2: Test greater than based on 'code' field (lexicographical)
    # 'USD' is alphabetically less than 'ZAR'
    zar = Currency.of("ZAR", "South African Rand", 2, CurrencyType.MONEY)
    assert zar > usd

    # Case 3: Test greater than based on 'name' field (when codes are identical)
    # 'US Dollars' vs 'US Dollars Premium'
    usd_premium = Currency.of("USD", "US Dollars Premium", 2, CurrencyType.MONEY)
    assert usd_premium > usd

    # Case 4: Test greater than based on 'decimals' field (when codes and names are identical)
    # 2 decimals vs 0 decimals
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencylyType.MONEY) # Note: We use same name for testing logic if needed
    # To strictly test the __gt__ logic provided by dataclass(order=True), we must manipulate fields in order.
    # Let's create a controlled comparison where code and name are same but decimals differ.
    usd_zero_dec = Currency.of("USD", "US Dollars", 0, CurrencyType.MONEY)
    assert usd > usd_zero_dec  # Because '2' (from USD) is actually compared via the fields... 
                               # Wait, in dataclass(order=True), it compares field by field.
                               # For USD: code="USD", name="US Dollars", decimals=2
                               # For USD_ZERO: code="USD", name="US Dollars", decimals=0
                               # Since 2 > 0, usd > usd_zero_dec is True.
    assert usd > usd_zero_dec

    # Case 5: Test less than (the inverse)
    assert not (usd > zar)
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest

def test_CurrencyRegistry___new__():
    """
    Tests that the __new__ method implements a singleton pattern, 
    ensuring that multiple instantiations return the exact same instance.
    """
    # First instantiation
    instance1 = CurrencyRegistry()
    
    # Second instantiation
    instance2 = CurrencyRegistry()
    
    # Verify both variables point to the same memory address (Singleton behavior)
    assert instance1 is instance2
    
    # Verify that calling __new__ via class directly also returns the singleton
    instance3 = CurrencyRegistry.__new__(CurrencyRegistry)
    assert instance1 is instance3

def test_CurrencyRegistry_singleton_persistence():
    """
    Tests that the singleton instance persists across different variable assignments.
    """
    registry_a = CurrencyRegistry()
    registry_b = CurrencyRegistry()
    
    # Verify identity
    assert id(registry_a) == id(registry_b)
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
from decimal import Decimal

def test_Currency___ge__():
    """
    Tests the greater-than-or-equal operator (__ge__) for the Currency class.
    Since Currency is a dataclass with order=True, __ge__ uses the 
    comparison of fields in order: code, name, decimals, type, quantizer, hashcache.
    """
    # Setup currencies for comparison
    # Base currency
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    # Case 1: Equality (self == other)
    # Since order=True, __ge__ should return True if items are equal
    usd_duplicate = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd >= usd_duplicate

    # Case 2: Greater than (self > other)
    # 'USD' is greater than 'ABC' alphabetically in the first field 'code'
    abc = Currency.of("ABC", "Alpha Currency", 2, CurrencyType.MONEY)
    assert usd >= abc
    
    # Case 3: Less than (self < other)
    # 'USD' is less than 'XYZ' alphabetically
    xyz = Currency.of("XYZ", "X Currency", 2, CurrencyType.MONEY)
    assert not (usd >= xyz)

    # Case 4: Testing field precedence (code -> name)
    # Same code, but different name ('US Dollars' > 'US')
    usd_alt_name = Currency.of("USD", "US", 2, CurrencyType.MONEY)
    assert usd >= usd_alt_name
    assert usd_alt_name < usd

    # Case 5: Testing field precedence (name -> decimals)
    # Same code and name, but different decimals (2 > 1)
    usd_low_decimal = Currency.of("USD", "US Dollars", 1, CurrencyType.MONEY)
    assert usd >= usd_low_decimal
    assert usd_low_decimal < usd

    # Case 6: Testing field precedence (decimals -> type)
    # Same code, name, and decimals, but different type
    # Note: Enum comparison depends on definition order; MONEY is first in CurrencyType
    usd_crypto = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert usd >= usd_crypto
    assert usd_crypto > usd

    # Case 7: Testing with different types of objects (should raise TypeError for incompatible types)
    with pytest.raises(TypeError):
        _ = usd >= "Not a currency"
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest
from decimal import Decimal

def test_Currency___le__():
    """
    Tests the less-than-or-equal comparison (__le__) for the Currency class.
    Since the Currency class is decorated with @dataclass(order=True), 
    it implements __le__ based on the order of fields in the dataclass:
    code, name, decimals, type, quantizer, hashcache.
    """
    # Base currency for comparison
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    # Case 1: Equality (should be True)
    # Recreating same attributes results in the same object/value
    usd_equal = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd <= usd_equal
    
    # Case 2: Less than based on 'code' (alphabetical order)
    # 'AUD' comes before 'USD'
    aud = Currency.of("AUD", "Australian Dollars", 2, CurrencyType.MONEY)
    assert aud <= usd
    assert not usd <= aud

    # Case 3: Less than based on 'name' (when codes are identical)
    # 'US Dollars' vs 'US Dollar' (alphabetical order)
    usd_short_name = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    assert usd_short_name <= usd
    assert not usd <= usd_short_name

    # Case 4: Less than based on 'decimals' (when code and name are identical)
    # 0 decimals < 2 decimals
    jpy = Currency.of("JPY", "Japanese Yen", 0, Currency    rrencyType.MONEY)
    usd_zero_decimals = Currency.of("USD", "US Dollars", 0, CurrencyType.MONEY)
    # Note: For the comparison to reach 'decimals', code and name must match.
    # Let's create a scenario where code/name are same but decimals differ.
    val_low_dec = Currency.of("USD", "US Dollars", 0, CurrencyType.MONEY)
    val_high_dec = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert val_low_dec <= val_high_dec

    # Case 5: Testing the boundary of decimals (decimals >= -1)
    crypto_weird = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    # Since 'USD' < 'ZZZ' alphabetically
    assert usd <= crypto_weird
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest
from decimal import Decimal

def test_Currency():
    # Test successful creation of different currency types
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.code == "USD"
    assert usd.name == "US Dollars"
    assert usd.decimals == 2
    assert usd.type == CurrencyType.MONEY
    assert usd.quantize(Decimal("1.005")) == Decimal("1.00")
    assert usd.quantize(Decimal("1.015")) == Decimal("1.02")

    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy.decimals == 0
    assert jpy.quantize(Decimal("0.5")) == Decimal("0")
    assert jpy.quantize(Decimal("1.5")) == Decimal("2")

    crypto = Currency.of("BTC", "Bitcoin", -1, CurrencyType.CRYPTO)
    assert crypto.decimals == -1
    assert crypto.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")

    # Test Equality and Hash
    usd_duplicate = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd_different_name = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert usd == usd_duplicate
    assert usd != usd_different_name
    assert hash(usd) == hash(usd_duplicate)
    assert hash(usd) != hash(usd_different_name)

    # Test Validation: Code constraints
    with pytest.raises(ProgrammingError):
        Currency.of("usd", "US Dollars", 2, CurrencyType.MONEY)  # Not uppercase
    with pytest.raises(ProgrammingError):
        Currency.of("US1", "US Dollars", 2, CurrencyType.MONEY) # Non-alpha
    with pytest.raises(ProgrammingError):
        Currency.of(123, "US Dollars", 2, CurrencyType.MONEY)   # Not a string

    # Test Validation: Name constraints
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "", 2, CurrencyType.MONEY)           # Empty name
    with pytest.raises(ProgrammingError):
        Currency.of("USD", " US Dollars ", 2, CurrencyType.MONEY) # Untrimmed

    # Test Validation: Decimals constraints
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", -2, CurrencyType.MONEY) # Less than -1
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", "2", CurrencyType.MONEY) # Not an int

    # Test Validation: Type constraints
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", 2, "MONEY")            # Not a CurrencyType enum
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest
from decimal import Decimal

def test_Currency___delattr__():
    """
    Tests that attempting to delete an attribute from a frozen dataclass 
    raises a FrozenInstanceError (or AttributeError).
    """
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    with pytest.raises(AttributeError):
        del usd.code

    with pytest.raises(AttributeError):
        del usd.name

    with pytest.raises(AttributeError):
        del usd.decimals

    with pytest.raises(AttributeError):
        del usd.type
```


# LLM-generated content at query #16
#--------------------------

```python
def test_CurrencyRegistry___new__():
    """
    Tests that CurrencyRegistry implements the Singleton pattern via __new__.
    Ensures that multiple instantiations return the exact same object instance.
    """
    registry_instance_1 = CurrencyRegistry()
    registry_instance_2 = CurrencyRegistry()

    # Check that both variables point to the same memory address (Singleton)
    assert registry_instance_1 is registry_instance_2
    # Check that they are of the correct type
    assert isinstance(registry_instance_1, CurrencyRegistry)
    assert isinstance(registry_instance_2, CurrencyRegistry)
```


# LLM-generated content at query #17
#--------------------------

```python
def test_CurrencyLookupError():
    code = "XYZ"
    error = CurrencyLookupError(code)
    
    assert error.code == code
    assert str(error) == f"Currency identified by code '{code}' does not exist"
    assert isinstance(error, LookupError)
```


# LLM-generated content at query #18
#--------------------------

```python
import pytest

def test_CurrencyRegistry___getitem__():
    # Reset singleton instance for clean testing environment
    CurrencyRegistry._instance = None
    registry = CurrencyRegistry()
    
    usd_code = "USD"
    usd_name = "US Dollar"
    usd_currency = Currency.of(usd_code, usd_name, 2, CurrencyType.MONEY)
    non_existent_code = "XYZ"

    # Test context for registration
    with registry as register:
        register(usd_currency)

    # Test successful retrieval via __getitem__
    retrieved_currency = registry[usd_code]
    assert retrieved_currency.code == usd_code
    assert retrieved_currency.name == usd_name
    assert retrieved_currency == usd_currency

    # Test failure via __getitem__ raises CurrencyLookupError
    with pytest.raises(CurrencyLookupError) as excinfo:
        _ = registry[non_existent_code]
    
    assert non_existent_code in str(excinfo.value)
    assert f"Currency identified by code '{non_existent_code}' does not exist" in str(excinfo.value)

    # Test edge case: accessing an empty registry
    CurrencyRegistry._instance = None
    empty_registry = CurrencyRegistry()
    with pytest.raises(CurrencyLookupError):
        _ = empty_registry["USD"]
```


# LLM-generated content at query #19
#--------------------------

```python
import pytest
from decimal import Decimal

def test_CurrencyRegistry___getitem__():
    # Reset singleton instance for clean testing environment
    CurrencyRegistry.__instance = None
    registry = CurrencyRegistry()
    
    usd_code = "USD"
    usd_name = "US Dollars"
    usd_currency = Currency.of(usd_code, usd_name, 2, CurrencyType.MONEY)
    
    # Test case 1: Successful retrieval of an existing currency
    with registry as register:
        register(usd_currency)
    
    assert registry[usd_code] == usd_currency
    assert registry[usd_code].code == usd_code
    assert registry[usd_code].name == usd_name

    # Test case 2: Retrieval of a non-existent currency raises CurrencyLookupError
    non_existent_code = "XYZ"
    with pytest.raises(CurrencyLookupError) as excinfo:
        _ = registry[non_existent_code]
    
    assert str(excinfo.value) == f"Currency identified by code '{non_existent_code}' does not exist"
    assert excinfo.value.code == non_existent_code

    # Test case 3: Ensure the error is specifically a CurrencyLookupError (subclass of LookupError)
    with pytest.raises(LookupError):
        _ = registry["NON-EXISTENT"]
```


# LLM-generated content at query #20
#--------------------------

```python
def test_CurrencyLookupError():
    invalid_code = "XYZ"
    error = CurrencyLookupError(invalid_code)
    
    assert error.code == invalid_code
    assert str(error) == f"Currency identified by code '{invalid_code}' does not exist"
    assert isinstance(error, LookupError)
```


# LLM-generated content at query #21
#--------------------------

```python
import pytest
from decimal import Decimal

def test_CurrencyRegistry___getitem__():
    # Reset singleton instance for clean testing environment
    CurrencyRegistry._instance = None
    registry = CurrencyRegistry()
    
    # Define a test currency
    usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    gbp = Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY)

    # Test Case 1: Successful retrieval within registry context
    with registry as register:
        register(usd)
        register(gbp)
    
    assert registry["USD"] == usd
    assert registry["GBP"] == gbp
    assert registry["USD"].code == "USD"

    # Test Case 2: Raising CurrencyLookupError for non-existent code
    with pytest.raises(CurrencyLookupError) as excinfo:
        _ = registry["XYZ"]
    
    assert "Currency identified by code 'XYZ' does not exist" in str(excinfo.value)
    assert excinfo.value.code == "XYZ"

    # Test Case 3: Verify error type is specifically CurrencyLookupError
    with pytest.raises(CurrencyLookupError):
        _ = registry["NON-EXISTENT"]
```


# LLM-generated content at query #22
#--------------------------

```python
import pytest
from decimal import Decimal

def test_CurrencyRegistry___getitem__():
    # Reset singleton instance for a clean test environment
    CurrencyRegistry._instance = None
    registry = CurrencyRegistry()
    
    # Create dummy currencies
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)

    # Test context-based registration to populate the registry
    with registry as register:
        register(usd)
        register(jpy)

    # Test case 1: Successful lookup for an existing code
    assert registry["USD"] == usd
    assert registry["JPY"].code == "JPY"

    # Test case 2: Lookup raises CurrencyLookupError for non-existent code
    with pytest.raises(CurrencyLookupError) as excinfo:
        _ = registry["XYZ"]
    
    assert "Currency identified by code 'XYZ' does not exist" in str(excinfo.value)
    assert excinfo.value.code == "XYZ"

    # Test case 3: Ensure it specifically raises the custom error, not a standard KeyError
    with pytest.raises(CurrencyLookupError):
        _ = registry["NON_EXISTENT"]
```


# LLM-generated content at query #23
#--------------------------

```python
import pytest
from decimal import Decimal

def test_Currency_quantize():
    # Test Case 1: Standard Money Currency (2 decimals) - Round Half to Even
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.quantize(Decimal("1.005")) == Decimal("1.00")
    assert usd.quantize(Decimal("1.015")) == Decimal("1.02")
    assert usd.quantize(Decimal("1.006")) == Decimal("1.01")
    assert usd.quantize(Decimal("1.004")) == Decimal("1.00")

    # Test Case 2: Zero decimals (e.g., JPY) - Rounding to integer
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy.quantize(Decimal("0.5")) == Decimal("0")
    assert jpy.quantize(Decimal("1.5")) == Decimal("2")
    assert jpy.quantize(Decimal("0.4")) == Decimal("0")
    assert jpy.quantize(Decimal("1.4")) == Decimal("1")

    # Test Case 3: Negative decimals (e.g., Crypto/High precision) - Using MaxPrecisionQuantizer behavior
    # Note: Based on the docstring, decimals < 0 uses MaxPrecisionQuantizer which handles large scale
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert zzz.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert zzz.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")

    # Test Case 4: Edge case with exactly zero
    zero_val = Decimal("0.000")
    assert usd.quantize(zero_val) == Decimal("0.00")
```


# LLM-generated content at query #24
#--------------------------

```python
import pytest
from decimal import Decimal

def test_Currency___hash__():
    """
    Tests that the __hash__ method of the Currency class returns the pre-computed 
    hashcache and ensures consistency for identical currency objects.
    """
    # Setup valid currency parameters
    code = "USD"
    name = "US Dollars"
    decimals = 2
    ctype = CurrencyType.MONEY

    # Create two different instances that represent the same currency
    usd1 = Currency.of(code, name, decimals, ctype)
    usd2 = Currency.of(code, name, decimals, ctype)
    
    # Create an instance that is different (different name)
    usd_diff_name = Currency.of("USD", "US Dollars Modified", decimals, ctype)

    # 1. Test that __hash__ returns the cached value directly
    assert hasattr(usd1, 'hashcache')
    assert hash(usd1) == usd1.hashcache

    # 2. Test that identical currencies produce the same hash
    assert hash(usd1) == hash(usd2)
    assert hash(usd1) == hash(usd2)

    # 3. Test that different currencies produce different hashes
    # Note: While collisions are theoretically possible, for these distinct objects they should differ
    assert hash(usd1) != hash(usd_diff_name)

    # 4. Verify that the hash is consistent across multiple calls
    first_call = hash(usd1)
    second_call = hash(usd1)
    assert first_call == second_call

    # 5. Test that the hash is compatible with set/dict operations (standard requirement for __hash__)
    currency_set = {usd1, usd2, usd_diff_name}
    assert len(currency_set) == 2  # usd1 and usd2 are considered same due to equality and hash
    assert usd1 in currency_set
```


# LLM-generated content at query #25
#--------------------------

```python
import pytest
from decimal import Decimal

def test_CurrencyRegistry___getitem__():
    # Reset singleton instance for clean testing environment
    CurrencyRegistry._instance = None
    registry = CurrencyRegistry()
    
    # Setup: Define a dummy currency
    usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    
    # We must use the context manager to register currencies as per implementation requirements
    with registry as register:
        register(usd)
        
    # Test Case 1: Successful retrieval of an existing currency
    assert registry["USD"] == usd
    assert registry["USD"].code == "USD"
    assert registry["USD"].name == "US Dollar"

    # Test Case 2: Retrieval of a non-existing currency raises CurrencyLookupError
    with pytest.raises(CurrencyLookupError) as excinfo:
        _ = registry["NON_EXISTENT"]
    
    assert "Currency identified by code 'NON_EXISTENT' does not exist" in str(excinfo.value)
    assert excinfo.value.code == "NON_EXISTENT"

    # Test Case 3: Ensure it behaves correctly with different types of unregistered keys
    with pytest.raises(CurrencyLookupError):
        _ = registry[""]
        
    with pytest.raises(CurrencyLookupError):
        _ = registry["123"]
```


# LLM-generated content at query #26
#--------------------------

```python
import pytest
from decimal import Decimal

def test_Currency___setattr__():
    """
    Tests that the Currency class, being a frozen dataclass, 
    raises FrozenInstanceError (or TypeError) when attempting to modify attributes.
    """
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    # Since the class is decorated with @dataclass(frozen=True),
    # any attempt to set an attribute after initialization should raise a TypeError.
    with pytest.raises(TypeError):
        usd.code = "EUR"

    with pytest.raises(TypeError):
        usd.decimals = 3

    with pytest.raises(TypeError):
        usd.name = "New Name"

    # Verify that the original values remain unchanged despite failed attempts
    assert usd.code == "USD"
    assert usd.decimals == 2
    assert usd.name == "US Dollars"
```


# LLM-generated content at query #27
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_CurrencyRegistry___enter__():
    """
    Tests that the __enter__ method of CurrencyRegistry returns the 
    internal __register method and sets the context flag to True.
    """
    # Reset singleton instance for clean testing environment
    CurrencyRegistry.__instance = None
    registry = CurrencyRegistry()

    # Verify initial state: context should not be open
    assert registry._CurrencyRegistry__ctx_open is False

    # Use the context manager via __enter__
    register_func = registry.__enter__()

    # 1. Check if the returned object is the internal register method
    # Since __register is name-mangled, we access it via the mangled name
    assert register_func == registry._CurrencyRegistry__register
    
    # 2. Check if the context flag was set to True
    assert registry._CurrencyRegistry__ctx_open is True

    # Clean up: exit the context manually since we called __enter__ directly
    registry.__exit__(None, None, None)
    assert registry._CurrencyRegistry__ctx_open is False
```


# LLM-generated content at query #28
#--------------------------

```python
import pytest
from decimal import Decimal

def test_Currency___eq__():
    # Setup valid currencies
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    # Different name but same core identifying attributes for the purpose of hash/equality logic in this implementation
    # Note: The provided __eq__ uses self.hashcache == other.hashcache.
    # Based on the .of() method, hashcode is derived from (code, name, decimals, ctype, quantizer)
    usd_diff_name = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY) # Identical to usd1
    
    # Create a truly different currency
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    
    # Case 1: Equality with identical object
    assert usd1 == usd1
    
    # Case 2: Equality with different instance but same attributes (same hashcache)
    assert usd1 == usd2
    
    # Case 3: Inequality with different code/name/decimals/type
    assert usd1 != eur
    assert usd1 != jpy
    
    # Case 4: Inequality with different type
    usd_crypto = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert usd1 != usd_crypto

    # Case 5: Equality check against non-Currency types (should return False via isinstance check)
    assert usd1 != "USD"
    assert usd1 != 123
    assert usd1 != None

    # Case 6: Verify hash consistency for equality
    assert hash(usd1) == hash(usd2)
```


# LLM-generated content at query #29
#--------------------------

```python
import pytest
from decimal import Decimal

def test_Currency___lt__():
    """
    Tests the __lt__ (less than) implementation of the Currency class.
    Since Currency is a dataclass with order=True, it uses the 
    order of fields defined in the class for comparison.
    The field order is: code, name, decimals, type, quantizer, hashcache.
    """
    # Setup currencies with different values to test ordering
    # Case 1: Different codes (code is the first field)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    # 'E' comes before 'U', so EUR < USD
    assert eur < usd
    assert not usd < eur

    # Case 2: Same code, different names (name is the second field)
    usd_alt = Currency.of("USD", "US Dollars Alt", 2, CurrencyType.MONEY)
    # "US Dollars" comes before "US Dollars Alt" lexicographically
    assert usd < usd_alt
    assert not usd_alt < usd

    # Case 3: Same code and name, different decimals (decimals is the third field)
    usd_zero_dec = Currency.of("USD", "US Dollars", 0, Currencyty.MONEY) # This would fail validation in .of()
    # Since we cannot use .of() for invalid decimal logic easily without refactoring, 
    # we rely on the fact that decimals is compared as an integer.
    # Let's create valid ones with different decimals via a manual bypass or valid range.
    usd_2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    # Note: .of() enforces code.isalpha(), etc. We use valid increments.
    # Let's assume we compare two currencies where the first two fields are identical.
    # We need to be careful because 'decimals' is part of the identity in the provided class.
    
    # Creating a scenario for decimals comparison:
    # Since .of() doesn't allow us to easily change only one field while keeping others same 
    # without violating the validation logic (like name or code), we test the logic directly.
    
    # We can use the dataclass feature: if all preceding fields are equal, it compares the next.
    # We will construct objects manually bypassing .of() to test the __lt__ logic specifically 
    # for the 'decimals' field, as .of() validates and creates them.
    
    def create_raw_currency(code, name, decimals, ctype):
        # Replicating the internal structure of Currency.of for testing purposes
        quantizer = Decimal("0.01") if decimals > 0 else (Decimal("0") if decimals == 0 else Decimal("1"))
        h = hash((code, name, decimals, ctype, quantizer))
        return Currency(code, name, decimals, ctype, quantizer, h)

    c_low_dec = create_raw_currency("USD", "US Dollars", 1, CurrencyType.MONEY)
    c_high_dec = create_raw_currency("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    assert c_low_dec < c_high_dec
    assert not c_high_dec < c_low_dec

    # Case 4: Same code, name, and decimals, different type (type is the fourth field)
    usd_money = create_raw_currency("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd_crypto = create_raw_currency("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    # Enum comparison: MONEY ("Money") vs CRYPTO ("Crypto Currency")
    # In Python Enums, comparison depends on the order of definition if they are comparable,
    # but for standard Enums, it's not supported unless using @total_ordering or IntEnum.
    # However, dataclass order=True uses the value/order of the field. 
    # For CurrencyType (Enum), it compares the members themselves.
    
    # Testing the logic that if all previous fields are equal, it moves to 'type'.
    # Note: Standard Enums don't support < unless they are IntEnums or we compare their values/names.
    # But in the context of dataclass(order=True), it will attempt to compare CurrencyType members.
    # Since CurrencyType is a standard Enum, comparison (<) between members raises TypeError 
    # UNLESS we are comparing the underlying values or if they were IntEnums.
    # Given the provided code uses 'Enum', __lt__ on Currency will raise TypeError if it reaches 'type'.
    # However, assuming the requirement is to test the functional capability of the dataclass order:
    
    with pytest.raises(TypeError):
        _ = usd_money < usd_crypto
```


# LLM-generated content at query #30
#--------------------------

```python
import pytest
from decimal import Decimal

def test_Currency___gt__():
    """
    Tests the __gt__ (greater than) magic method of the Currency class.
    Since the class is decorated with @dataclass(order=True), 
    the comparison follows the order of fields defined in the dataclass:
    code, name, decimals, type, quantizer, hashcache.
    """
    # Setup currencies for comparison
    # We will manipulate fields to trigger different comparison results
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    gbp = Currency.of("GBP", "British Pounds", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    
    # Test 1: Comparison based on 'code' (first field)
    # 'USD' > 'GBP' is True alphabetically
    assert usd > gbp
    assert gbp < usd
    
    # Test 2: Comparison where codes are equal, check 'name'
    usd_alt_name = Currency.of("USD", "United States Dollars", 2, CurrencyType.MONEY)
    # 'US Dollars' < 'United States Dollars' (alphabetical order)
    assert usd_alt_name > usd
    assert usd > usd_alt_name

    # Test 3: Comparison where code and name are equal, check 'decimals'
    # JPY has 0 decimals, USD has 2. In the dataclass order, decimals is 3rd.
    # Note: To reach 'decimals', 'code' and 'name' must match.
    # We create a dummy currency with same code/name as USD but different decimals.
    usd_high_precision = Currency.of("USD", "US Dollars", 4, CurrencyType.MONEY)
    # In dataclass order: code(same), name(same), decimals(4 > 2)
    assert usd_high_precision > usd

    # Test 4: Negative test for __gt__
    # Ensure that a 'smaller' currency does not return True for __gt__
    assert not (gbp > usd)
    
    # Test 5: Comparison with different types via the order of attributes
    # Creating a complex comparison case
    # Currency(code, name, decimals, type, ...)
    # 'A' < 'B'
    curr_a = Currency.of("AAA", "Alpha", 2, CurrencyType.MONARY if hasattr(CurrencyType, 'MONARY') else CurrencyType.MONEY)
    curr_b = Currency.of("BBB", "Beta", 2, CurrencyType.MONEY)
    assert curr_b > curr_a

    # Test 6: Verify that __gt__ is not used for non-Currency objects (should raise TypeError or behave via standard logic)
    with pytest.raises(TypeError):
        # Comparing Currency to a string should fail if the implementation relies on object type comparison
        # Since @dataclass(order=True) compares fields, comparing with a different type 
        # usually results in an error when it tries to access attributes of the other object.
        _ = usd > "USD"
```


# LLM-generated content at query #31
#--------------------------

```python
import pytest
from decimal import Decimal

def test_Currency___eq__():
    # Setup valid currencies
    usd_1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd_2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    # Different name but same hashable components (code, name, decimals, type, quantizer)
    # Note: The implementation of __eq__ uses self.hashcache == other.hashcache.
    # Since hashcache is computed from (code, name, decimals, ctype, quantizer), 
    # changing the name changes the hashcache.
    usd_diff_name = Currency.of("USD", "US Dollars Updated", 2, CurrencyType.MONEY)
    
    # Different decimals
    usd_zero_dec = Currency.of("USD", "US Dollars", 0, CurrencyType.MONEY)
    
    # Different type
    usd_crypto = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    
    # Different code
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    
    # Test Equality
    assert usd_1 == usd_2
    
    # Test Inequality with different name
    assert usd_1 != usd_diff_name
    
    # Test Inequality with different decimals
    assert usd_1 != usd_zero_dec
    
    # Test Inequality with different type
    assert usd_1 != usd_crypto
    
    # Test Inequality with different code
    assert usd_1 != eur
    
    # Test Inequality with different types (e.g., comparing to a string)
    assert usd_1 != "USD"
    
    # Test Equality for hash consistency (as mentioned in docstring)
    assert hash(usd_1) == hash(usd_2)
```


# LLM-generated content at query #32
#--------------------------

```python
import pytest
from decimal import Decimal

def test_Currency_quantize():
    """
    Tests the quantize method of the Currency class with various scenarios including
    standard money, zero-decimal currencies (JPY), and negative-decimal (crypto) currencies.
    """
    # Test case 1: Standard Money Currency (USD - 2 decimals)
    # Should use ROUND_HALF_EVEN (default decimal context behavior in the provided code)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.quantize(Decimal("1.005")) == Decimal("1.00")
    assert usd.quantize(Decimal("1.015")) == Decimal("1.02")
    assert usd.quantize(Decimal("1.000")) == Decimal("1.00")

    # Test case 2: Zero decimal currency (JPY - 0 decimals)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy.quantize(Decimal("0.5")) == Decimal("0")
    assert jpy.quantize(Decimal("1.5")) == Decimal("2")
    assert jpy.quantize(Decimal("1.9")) == Decimal("2")

    # Test case 3: Negative decimal currency (ZZZ - -1 decimals)
    # Tests the MaxPrecisionQuantizer logic via the 'decimals < 0' branch
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert zzz.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert zzz.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")

    # Test case 4: Edge case with high precision decimals
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert eur.quantize(Decimal("10.555")) == Decimal("10.56")
    assert eur.quantize(Decimal("10.554")) == Decimal("10.55")

    # Test case 5: Testing precision preservation
    gbp = Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY)
    input_val = Decimal("100.123456")
    assert gbp.quantize(input_val) == Decimal("100.12")
```


# LLM-generated content at query #33
#--------------------------

```python
import pytest

def test_CurrencyRegistry___getitem__():
    # Reset singleton instance for clean testing environment
    CurrencyRegistry._instance = None
    registry = CurrencyRegistry()
    
    usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    gbp = Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY)

    # Setup registry context
    with registry as register:
        register(usd)
        register(gbp)

    # Test successful retrieval
    assert registry["USD"] == usd
    assert registry["GBP"] == gbp

    # Test retrieval of existing currency via code attribute
    assert registry["USD"].code == "USD"

    # Test retrieval of non-existent currency raises CurrencyLookupError
    with pytest.raises(CurrencyLookupError) as excinfo:
        _ = registry["XYZ"]
    
    assert "Currency identified by code 'XYZ' does not exist" in str(excinfo.value)
    assert excinfo.value.code == "XYZ"

    # Test retrieval of non-existent currency raises error for different code
    with pytest.raises(CurrencyLookupError):
        _ = registry["NON-EXISTING"]
```


# LLM-generated content at query #34
#--------------------------

```python
import pytest

def test_CurrencyRegistry():
    # Reset Singleton instance for clean testing environment
    CurrencyRegistry._instance = None
    
    registry = CurrencyRegistry()
    
    # Test initial state of the registry
    assert len(registry) == 0
    assert registry.all == []
    assert registry.codes == []
    assert registry.codenames == []
    
    # Test Singleton behavior: a new instance should be the same object
    another_registry = CurrencyRegistry()
    assert registry is another_registry

    # Test population via context manager
    usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)

    with registry as register:
        register(usd)
        register(jpy)
        register(eur)
        # Inside context, ctx_open should be True (via __enter__)
        assert registry._CurrencyRegistry__ctx_open is True

    # Test post-context state (sorting and finalization)
    # The registry should be sorted by code: EUR, JPY, USD
    assert registry.codes == ["EUR", "JPY", "USD"]
    assert registry.codenames == [("EUR", "Euro"), ("JPY", "Japanese Yen"), ("USD", "US Dollar")]
    assert len(registry) == 3
    assert registry.all == [eur, jpy, usd]
    assert registry._CurrencyRegistry__ctx_open is False

    # Test lookup functionality after initialization
    assert "USD" in registry
    assert registry["USD"].code == "USD"
    assert registry.has("EUR") is True
    assert registry.has("GBP") is False
    assert registry.get("JPY") == jpy
    assert registry.get("NON_EXISTENT", default=usd) == usd

    # Test error handling for duplicate registration
    with pytest de(ValueError, "Currency USD is already registered."):
        with registry as register:
            register(usd)

    # Test error handling for registration outside context
    with pytest.raises(ProgrammingError, match="Can not create currencies outside registry context."):
        registry.__register(usd)
```


# LLM-generated content at query #35
#--------------------------

```python
import pytest
from decimal import Decimal

def test_CurrencyRegistry___getitem__():
    """
    Tests the __getitem__ method of CurrencyRegistry.
    Verifies that it returns the correct currency when the code exists
    and raises CurrencyLookupError when the code does not exist.
    """
    # Reset singleton instance for clean testing environment
    CurrencyRegistry.__instance = None
    registry = CurrencyRegistry()
    
    usd_code = "USD"
    usd_currency = Currency.of(usd_code, "US Dollars", 2, CurrencyType.MONEY)
    non_existent_code = "XYZ"

    # Test case: Accessing a non-existent key before registry is populated
    with pytest.raises(CurrencyLookupError) as excinfo:
        _ = registry[non_existent_code]
    assert f"Currency identified by code '{non_existent_code}' does not exist" in str(excinfo.value)
    assert excinfo.value.code == non_existent_code

    # Test case: Accessing a registered currency
    with registry as register:
        register(usd_currency)
    
    # Verify retrieval
    retrieved_currency = registry[usd_code]
    assert retrieved_currency == usd_currency
    assert retrieved_currency.code == usd_code
    assert retrieved_currency.name == "US Dollars"

    # Test case: Accessing a non-existent key after registration
    with pytest.raises(CurrencyLookupError) as excinfo:
        _ = registry[non_existent_code]
    assert f"Currency identified by code '{non_existent_code}' does not exist" in str(excinfo.value)
```


# LLM-generated content at query #36
#--------------------------

```python
import pytest

def test_CurrencyRegistry___len__():
    """
    Tests the __len__ method of the CurrencyRegistry class.
    Ensures that it correctly returns the count of registered currencies.
    """
    # Reset singleton instance for a clean test environment if possible, 
    # though in many test runners we just work with the existing one.
    registry = CurrencyRegistry()
    
    # Since Registry is a singleton, we must clear/re-initialize it or use a context 
    # to ensure we are testing from a known state (0).
    # We access the private attribute for the purpose of this unit test.
    registry._CurrencyRegistry__registry = OrderedDict()
    registry._CurrencyRegistry__currencies = []
    registry._CurrencyRegistry__codes = []
    registry._CurrencyRegistry__codenames = []

    # Verify initial length is 0
    assert len(registry) == 0

    # Use the registry context to add currencies
    with registry as register:
        usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
        register(usd)
        register(eur)

    # After context exit, the registry is finalized and sorted
    assert len(registry) == 2
    
    # Verify individual addition via register (if we were to add another)
    # Note: To add more we need a new context since __ctx_open was set to False
    with registry as register:
        gbp = Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY)
        register(gbp)

    assert len(registry) == 3
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from decimal import Decimal

def test_Currency___eq__():
    # Setup identical currencies (same hashcache)
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    # Setup currency with different name but same code/decimals/type (to test hashcache dependency)
    # Since __eq__ uses hashcache, we need to ensure the attributes used in hash calculation are identical
    # In the provided implementation, hashcode is derived from (code, name, decimals, ctype, quantizer)
    # Therefore, if names differ, the hashcode will likely differ.
    usd_alt_name = Currency.of("USD", "US Dollars Updated", 2, CurrencyType.MONEY)
    
    # Setup currency with different type
    usd_crypto = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)

    # Test equality of identical objects
    assert usd1 == usd2
    
    # Test inequality of objects with different names (different hashcache)
    assert usd1 != usd_alt_name
    
    # Test inequality of objects with different types (different hashcache)
    assert usd1 != usd_crypto

    # Test equality against different types (non-Currency objects)
    assert usd1 != "USD"
    assert usd1 != 123
    assert usd1 != None

    # Test symmetry
    assert usd2 == usd1

    # Test hash consistency for equality
    assert hash(usd1) == hash(usd2)
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from decimal import Decimal

def test_Currency___eq__():
    # Setup common components
    code = "USD"
    name = "US Dollars"
    decimals = 2
    ctype = CurrencyType.MONEY
    quantizer = Decimal("0.01")
    hash_val = hash((code, name, decimals, ctype, quantizer))

    # Create base currency instance
    usd1 = Currency(code, name, decimals, ctype, quantizer, hash_val)
    
    # 1. Test equality with identical instance
    usd2 = Currency(code, name, decimals, ctype, quantizer, hash_val)
    assert usd1 == usd2

    # 2. Test equality with different object but same hashcache (as per __eq__ implementation)
    # Note: The implementation specifically checks self.hashcache == other.hashcache
    usd_same_hash = Currency("XYZ", "Different Name", 5, CurrencyType.CRYPTO, Decimal("0.00001"), hash_val)
    assert usd1 == usd_same_hash

    # 3. Test inequality with different hashcache
    different_hash_val = hash((code, name, decimals, ctype, Decimal("1.0")))
    usd3 = Currency(code, name, decimals, ctype, Decimal("1.0"), different_hash_val)
    assert usd1 != usd3

    # 4. Test inequality with different type (not a Currency instance)
    assert usd1 != "Not a currency object"
    assert usd1 != None

    # 5. Test equality via the factory method (Integration check)
    usd_factory = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd_factory
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from decimal import Decimal

def test_CurrencyRegistry___getitem__():
    """
    Tests the __getitem__ method of CurrencyRegistry, verifying that it returns 
    the correct Currency object for a valid code and raises CurrencyLookupError 
    for an invalid code.
    """
    # Reset singleton instance for clean testing environment
    CurrencyRegistry.__instance = None
    registry = CurrencyRegistry()
    
    # Create dummy currencies
    usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    
    # Use the registry context to register currencies
    with registry as register:
        register(usd)
        register(eur)
    
    # Case 1: Valid lookup returns the correct object
    assert registry["USD"] == usd
    assert registry["EUR"] == eur
    
    # Case 2: Lookup for a non-existent code raises CurrencyLookupError
    with pytest.raises(CurrencyLookupError) as excinfo:
        _ = registry["NON_EXISTENT"]
    
    assert "Currency identified by code 'NON_EXISTENT' does not exist" in str(excinfo.value)
    assert excinfo.value.code == "NON_EXISTENT"
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest

def test_CurrencyRegistry___getitem__():
    # Reset singleton instance for a clean test environment
    CurrencyRegistry._instance = None
    registry = CurrencyRegistry()
    
    usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    gbp = Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY)

    # Register currencies using the context manager
    with registry as register:
        register(usd)
        register(gbp)

    # Test successful retrieval
    assert registry["USD"] == usd
    assert registry["GBP"] == gbp
    assert registry["USD"].code == "USD"
    assert registry["USD"].name == "US Dollar"

    # Test retrieval of non-existent code raises CurrencyLookupError
    with pytest.raises(CurrencyLookupError) as excinfo:
        _ = registry["XYZ"]
    
    assert "Currency identified by code 'XYZ' does not exist" in str(excinfo.value)
    assert excinfo.value.code == "XYZ"

    # Test retrieval of non-existent code with invalid type (e.g., integer) 
    # to ensure the error message formats correctly even if input isn't a string
    with pytest.raises(CurrencyLookupError) as excinfo:
        _ = registry[123]
    
    assert "Currency identified by code '123' does not exist" in str(excinfo.value)
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from decimal import Decimal

def test_Currency___eq__():
    # Setup identical currencies
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    # Setup currency with same properties but different name (should be False based on docstring logic/hash mismatch)
    # Note: The provided implementation's __eq__ relies solely on hashcache.
    # Since hashcache is derived from (code, name, decimals, ctype, quantizer), 
    # a different name will result in a different hashcache.
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    
    # Setup currency with different code
    gbp = Currency.of("GBP", "British Pounds", 2, CurrencyType.MONEY)
    
    # Setup currency with different decimals
    usd_zero_dec = Currency.of("USD", "US Dollars", 0, CurrencyType.MONEY)

    # Assertions for equality
    assert usd1 == usd2, "Identical currencies should be equal"
    assert usd1 == Currency(usd1.code, usd1.name, usd1.decimals, usd1.type, usd1.quantizer, usd1.hashcache), "Manually reconstructed identical currency should be equal"

    # Assertions for inequality
    assert usd1 != usdx, "Currencies with different names should not be equal"
    assert usd1 != gbp, "Currencies with different codes should not be equal"
    assert usd1 != usd_zero_dec, "Currencies with different decimals should not be equal"
    assert usd1 != "USD", "Currency should not be equal to a string"
    assert usd1 != None, "Currency should not be equal to None"

    # Assertions for hash consistency (required for __eq__ logic in the provided code)
    assert hash(usd1) == hash(usd2), "Identical currencies must have the same hash"
    assert hash(usd1) != hash(usdx), "Different currencies should have different hashes"
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest
from decimal import Decimal

def test_Currency___le__():
    """
    Tests the __le__ (less than or equal) method of the Currency class.
    Since the class is decorated with @dataclass(order=True), 
    the order is determined by the field order: code, name, decimals, type...
    """
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd_alt = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    gbp = Currency.of("GBP", "British Pounds", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    
    # Test Equality (Less than or equal should be True if items are identical)
    assert usd <= usd_alt
    assert usd_alt <= usd
    
    # Test Less Than based on 'code' attribute (First field in dataclass)
    # 'GBP' comes before 'JPY' alphabetically
    assert gbp <= jpy
    assert gbp < jpy
    
    # 'JPY' comes after 'GBP'
    assert jpy >= gbp
    
    # Test 'code' comparison specifically
    # 'USD' comes after 'GBP'
    assert gbp < usd
    
    # Test based on 'name' (If codes were identical, it would check name)
    usd_different_name = Currency.of("USD", "United States Dollars", 2, CurrencyType.MONEY)
    # 'US Dollars' vs 'United States Dollars' -> 'US' < 'United'
    assert usd <= usd_different_name
    assert usd_different_name >= usd

    # Test with decimals change (if codes/names were same)
    # Note: The provided Currency.of implementation checks code, name, etc.
    # We manually create a currency to bypass .of() validation if necessary, 
    # but using .of() is safer for standard testing.
    usd_zero_decimal = Currency.of("USD", "US Dollars", 0, CurrencyType.MONSD) # This would fail .of() logic
```

*Note: Because the `Currency` class uses `order=True` and the fields are ordered as `code`, `name`, `decimals`, `type`, etc., the `__le__` method effectively performs a lexicographical comparison of those attributes in order.*


# LLM-generated content at query #7
#--------------------------

```python
import pytest

def test_CurrencyRegistry___contains__():
    """
    Tests the __contains__ method of CurrencyRegistry.
    Verifies that it correctly identifies registered and unregistered currency codes.
    """
    # Reset the singleton instance for a clean testing environment
    CurrencyRegistry._CurrencyRegistry__instance = None
    registry = CurrencyRegistry()

    # Define some test currencies
    usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)

    # Test before registration
    assert "USD" not in registry
    assert "EUR" not in registry
    assert "XYZ" not in registry

    # Register currencies using the context manager
    with registry as register:
        register(usd)
        register(eur)
        register(jpy)

    # Test after registration
    assert "USD" in registry
    assert "EUR" in registry
    assert "JPY" in registry
    assert "XYZ" not in registry
    
    # Test with lowercase/different casing (codes are case-sensitive based on implementation)
    assert "usd" not in registry
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest

def test_CurrencyRegistry___contains__():
    """
    Tests the __contains__ method of CurrencyRegistry.
    """
    # Reset singleton instance for clean testing environment
    CurrencyRegistry._CurrencyRegistry__instance = None
    registry = CurrencyRegistry()
    
    # Create sample currencies
    usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    crypto = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)

    # Test registry behavior outside of context (should be empty)
    assert "USD" not in registry
    assert "JPY" not in registry
    assert "BTC" not in registry

    # Register currencies using the context manager
    with registry as register:
        register(usd)
        register(jpy)
        register(crypto)

    # Test successful containment
    assert "USD" in registry
    assert "JPY" in registry
    assert "BTC" in registry

    # Test non-existent codes
    assert "EUR" not in registry
    assert "XYZ" not in registry
    assert "" not in registry
    assert 123 not in registry

    # Test case sensitivity (assuming USD != usd)
    assert "usd" not in registry
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from decimal import Decimal

def test_CurrencyRegistry___exit__():
    """
    Tests that the __exit__ method correctly re-sorts and synchronizes 
    the registry's internal buffers (registry, currencies, codes, codenames) 
    after a population context has closed.
    """
    # Reset singleton instance for clean test environment
    CurrencyRegistry.__instance = None
    registry = CurrencyRegistry()

    # Create some currencies out of alphabetical order
    # ZZZ is added first, then AAA
    ccy_zzz = Currency.of("ZZZ", "Z Currency", 2, CurrencyType.CRYPTO)
    ccy_aaa = Currency.of("AAA", "A Currency", 0, CurrencyType.MONEY)
    ccy_bbb = Currency.of("BBB", "B Currency", 2, CurrencyType.MONEY)

    # Use the context manager to register currencies
    with registry as register:
        register(ccy_zzz)
        register(ccy_aaa)
        register(ccy_bbb)
        
        # Inside the context, check if ctx_open is True (via internal access or behavior)
        # We verify that before exit, the order depends on insertion (OrderedDict default)
        assert len(registry.all) == 3

    # After __exit__, the registry should be sorted alphabetically by code: AAA, BBB, ZZZ
    
    # 1. Check 'all' property is sorted
    assert registry.all[0].code == "AAA"
    assert registry.all[1].code == "BBB"
    assert registry.all[2].code == "ZZZ"

    # 2. Check 'codes' property matches sorted order
    assert registry.codes == ["AAA", "BBB", "ZZZ"]

    # 3. Check 'codenames' property matches sorted order
    expected_codenames = [("AAA", "A Currency"), ("BBB", "B Currency"), ("ZZZ", "Z Currency")]
    assert registry.codenames == expected_codenames

    # 4. Verify that __contains__ and __getitem__ still work correctly with the new order
    assert "AAA" in registry
    assert registry["AAA"].name == "A Currency"
    assert registry["ZZZ"].code == "ZZZ"

    # 5. Verify context is closed (registering outside context should raise error)
    with pytest.raises(ProgrammingError, match="Can not create currencies outside registry context."):
        registry.__register(Currency.of("CCC", "C Currency", 2, CurrencyType.MONEY))
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
from decimal import Decimal

def test_Currency___delattr__():
    """
    Tests that attempting to delete an attribute from a frozen Currency instance 
    raises a FrozenInstanceError (or AttributeError, which is how dataclass 
    frozen=True manifests).
    """
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    # Since the class is decorated with @dataclass(frozen=True),
    # any attempt to delete an attribute should raise an AttributeError.
    with pytest.raises(AttributeError):
        del USD.code

    with pytest.raises(AttributeError):
        del USD.name

    with pytest.raises(AttributeError):
        del USD.decimals

    with pytest.raises(AttributeError):
        del USD.type
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
from decimal import Decimal

def test_Currency_quantize():
    """
    Tests the quantize method of the Currency class with various scenarios 
    including standard money, zero-decimal currency (JPY), and crypto/high precision.
    """
    # Test Case 1: Standard Money (USD) - 2 decimals
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.quantize(Decimal("1.005")) == Decimal("1.00")
    assert usd.quantize(Decimal("1.015")) == Decimal("1.02")
    assert usd.quantize(Decimal("1.555")) == Decimal("1.56")

    # Test Case 2: Zero decimals (JPY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy.quantize(Decimal("0.5")) == Decimal("0")
    assert jpy.quantize(Decimal("1.5")) == Decimal("2")
    assert jpy.quantize(Decimal("1.4")) == Decimal("1")

    # Test Case 3: Negative decimals (Crypto/High Precision)
    # Using -1 as per the docstring example logic for MaxPrecisionQuantizer
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert zzz.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert zzz.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")

    # Test Case 4: Rounding half to even (Banker's rounding)
    # 2.5 rounds to 2, 3.5 rounds to 4 when precision is 0
    gbp = Currency.of("GBP", "British Pounds", 0, CurrencyType.MONEY)
    assert gbp.quantize(Decimal("2.5")) == Decimal("2")
    assert gbp.quantize(Decimal("3.5")) == Decimal("4")

    # Test Case 5: Large decimal input for small precision currency
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert eur.quantize(Decimal("100.123456789")) == Decimal("100.12")
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest

def test_CurrencyRegistry___new__():
    """
    Tests that the __new__ method implements a singleton pattern,
    ensuring that multiple calls to CurrencyRegistry() return the same instance.
    """
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    
    # Check that both variables point to the exact same object in memory
    assert registry1 is registry2
    
    # Verify that a third instantiation also returns the same singleton
    registry3 = object.__new__(CurrencyRegistry) 
    # Note: We use object.__new__ for the third check to avoid triggering 
    # the existing __new__ logic if we wanted to test pure allocation,
    # but since CurrencyRegistry.__new__ is already written, 
    # simply calling the constructor is the standard way to test it.
    
    registry4 = CurrencyRegistry()
    assert registry1 is registry4
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest

def test_CurrencyRegistry___contains__():
    """
    Tests the __contains__ method of CurrencyRegistry.
    Verifies that it correctly identifies existing and non-existing currency codes.
    """
    # Reset singleton instance for testing isolation if possible, 
    # but since we can't modify the class here, we work with the existing singleton.
    registry = CurrencyRegistry()
    
    # We need to use a context manager to register currencies as per implementation requirements
    usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)

    with registry as register:
        register(usd)
        register(eur)
        register(jpy)
    
    # Test existing codes
    assert "USD" in registry
    assert "EUR" in registry
    assert "JPY" in registry
    
    # Test non-existing codes
    assert "GBP" not in registry
    assert "XYZ" not in registry
    assert "" not in registry

    # Test with a code that is part of a name but not a registered code
    assert "Dollar" not in registry
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest

def test_CurrencyRegistry():
    # Reset singleton instance for clean testing environment
    CurrencyRegistry._instance = None
    
    # Initialize registry
    registry = CurrencyRegistry()
    
    # Test 1: Check if it's a singleton
    registry2 = CurrencyRegistry()
    assert registry is registry2

    # Test 2: Check initial state of internal buffers (via public properties)
    assert len(registry) == 0
    assert registry.all == []
    assert registry.codes == []
    assert registry.codenames == []

    # Test 3: Test registration via context manager
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)

    with registry as register:
        register(usd)
        register(jpy) # Register out of alphabetical order to test sorting in __exit__
        register(eur)

    # Test 4: Verify post-context state (Ordering and content)
    assert len(registry) == 3
    assert registry.codes == ["EUR", "JPY", "USD"]  # Sorted alphabetically by code
    assert registry.codenames == [("EUR", "Euro"), ("JPY", "Japanese Yen"), ("USD", "US Dollars")]
    assert usd in registry
    assert "EUR" in registry
    assert "XXX" not in registry

    # Test 5: Verify item access and error handling
    assert registry["USD"] == usd
    assert registry.get("EUR") == eur
    assert registry.has("JPY") is True
    assert registry.has("GBP") is False
    assert registry.get("GBP", default=usd) == usd

    with pytest.raises(CurrencyLookupError) as excinfo:
        _ = registry["NON_EXISTENT"]
    assert "NON_EXISTENT" in str(excinfo.value)

    # Test 6: Verify error when registering outside context
    with pytest.raises(ProgrammingError) as excinfo:
        registry.__register(usd) # Accessing private method for test purposes
    assert "outside registry context" in str(excinfo.value)

    # Test 7: Verify duplicate registration error within context
    with registry as register:
        register(usd)
        with pytest.raises(ValueError) as excinfo:
            register(usd)
        assert "already registered" in str(excinfo.value)
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest
from decimal import Decimal

def test_CurrencyRegistry_get():
    """
    Tests the 'get' method of the CurrencyRegistry class.
    Verifies retrieving an existing currency, returning None for non-existent keys,
    and returning a default value when provided.
    """
    # Reset singleton instance for clean testing environment
    CurrencyRegistry._instance = None
    registry = CurrencyRegistry()
    
    # Define test currencies
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)

    # Populate the registry using a context manager
    with registry as register:
        register(usd)
        register(eur)
        register(jpy)

    # Test case 1: Retrieve an existing currency by code
    assert registry.get("USD") == usd
    assert registry.get("EUR").name == "Euro"
    assert registry.get("JPY").decimals == 0

    # Test case 2: Retrieve a non-existent currency (should return None)
    assert registry.get("GBP") is None
    assert registry.get("XYZ") is None

    # Test case 3: Retrieve with a default value provided
    # If code doesn't exist, return the fallback currency
    default_ccy = Currency.of("ABC", "Fallback", 2, CurrencyType.ALTERNATIVE)
    assert registry.get("GBP", default=default_ccy) == default_ccy
    assert registry.get("XYZ", default=usd) == usd

    # Test case 4: Verify the retrieval is case-sensitive (as per dict implementation)
    assert registry.get("usd") is None
```


# LLM-generated content at query #16
#--------------------------

```python
import pytest
from decimal import Decimal

def test_Currency___delattr__():
    """
    Tests that attempting to delete an attribute from a frozen dataclass 
    raises AttributeError.
    """
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    # Since Currency is decorated with @dataclass(frozen=True),
    # __delattr__ should raise an AttributeError.
    with pytest.raises(AttributeError) as excinfo:
        del usd.code
    
    assert "can't set attribute" in str(excinfo.value).lower() or "immutable" in str(excinfo.value).lower()

    # Verify that other attributes are still accessible and unchanged
    assert usd.code == "USD"
    assert usd.decimals == 2
```


# LLM-generated content at query #17
#--------------------------

```python
import pytest

def test_CurrencyRegistry___contains__():
    """
    Tests the __contains__ implementation of CurrencyRegistry.
    """
    # Reset Singleton instance for a clean testing environment
    CurrencyRegistry._CurrencyRegistry__instance = None
    registry = CurrencyRegistry()
    
    # Create some dummy currencies
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    crypto = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)

    # Verify initial state (empty registry)
    assert "USD" not in registry
    assert "JPY" not in registry
    assert "BTC" not in registry

    # Use the context manager to register currencies
    with registry as register:
        register(usd)
        register(jpy)
        register(crypto)

    # Verify that registered codes are present
    assert "USD" in registry
    assert "JPY" in registry
    assert "BTC" in registry

    # Verify that non-existent/unregistered codes are not present
    assert "EUR" not in registry
    assert "XYZ" not in registry
    assert "" not in registry
```


# LLM-generated content at query #18
#--------------------------

```python
import pytest
from decimal import Decimal

def test_Currency___ge__():
    """
    Tests the __ge__ (greater than or equal) operator for the Currency class.
    Since Currency is decorated with @dataclass(order=True), 
    it uses the fields in order: code, name, decimals, type, quantizer, hashcache.
    """
    # Setup currencies with different attributes to test ordering logic
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    gbp = Currency.of("GBP", "British Pounds", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    
    # Test equality (should be True for same values)
    usd_copy = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd >= usd_copy
    assert usd == usd_copy

    # Test greater than based on 'code' (alphabetical order: USD > GBP)
    assert usd >= gbp
    assert usd > gbp
    assert gbp < usd

    # Test equality with different name but same code/decimals/type 
    # Note: In @dataclass(order=True), if 'code' is same, it compares 'name'
    usd_alt_name = Currency.of("USD", "US Dollars Alt", 2, CurrencyType.MONIT) # This would fail .of() validation
    # Since we can't easily bypass .of() validation for invalid names, 
    # let's use valid attributes that change the order.
    
    # Test comparison based on 'code' ascending
    assert jpy < usd
    assert jpy >= gbp # JPY (J) is less than GBP (G)? No, G < J.
    
    # Correcting logic for alphabetical: GBP, JPY, USD
    assert gbp < jpy
    assert jpy < usd
    assert usd >= usd
    assert gbp <= jpy
    assert jpy >= gbp

    # Test comparison where code is same, but name differs (alphabetical)
    # We need to bypass .of() to create a currency with same code but different name 
    # because .of() enforces strict rules. However, we can use the constructor directly 
    # if we provide all required args including the pre-computed hashcache.
    # But a simpler way is to compare objects where 'code' is the first field.
    
    # Case: Same code, different name (USD vs US Dollars Alt)
    # We must manually calculate what the quantizer/hash would be or use a mock approach
    # But since we are testing __ge__, and it relies on the dataclass order:
    # 1. code 2. name 3. decimals ...
    
    # Create an object with identicals except name using constructor directly to bypass .of() validation
    usd_same_code_diff_name = Currency(
        code="USD",
        name="USD Different Name",
        decimals=2,
        type=CurrencyType.MONEY,
        quantizer=Decimal("0.01"),
        hashcache=hash(("USD", "USD Different Name", 2, CurrencyType.MONEY, Decimal("0.01")))
    )

    assert usd >= usd_same_code_diff_name # 'US Dollars' < 'USD Different Name' is False? 
    # 'US D' vs 'USD D' -> ' ' comes before 'D'. So 'US Dollars' < 'USD Different Name'
    assert usd_same_code_diff_name > usd
```


# LLM-generated content at query #19
#--------------------------

```python
import pytest
from decimal import Decimal

def test_Currency___hash__():
    """
    Tests that the __hash__ method returns the pre-computed hashcache 
    and maintains consistency for identical currency definitions.
    """
    # Create two identical currencies (different instances, same data)
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    # Create a different currency
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    
    # Verify that hash of identical objects is the same
    assert hash(usd1) == hash(usd2)
    
    # Verify that hash of different objects is different
    assert hash(usd1) != hash(jpy)
    
    # Verify that the returned value is indeed the cached hash attribute
    assert hash(usd1) == usd1.hashcache
```


# LLM-generated content at query #20
#--------------------------

```python
import pytest
from decimal import Decimal

def test_Currency___gt__():
    """
    Tests the __gt__ (greater than) magic method of the Currency class.
    Since the class is decorated with @dataclass(order=True), 
    it implements comparison based on the order of fields in the dataclass definition.
    The field order is: code, name, decimals, type, quantizer, hashcache.
    """
    # Setup currencies with different values for the first field (code)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    gbp = Currency.of("GBP", "British Pounds", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    
    # Test: USD > GBP is False because 'U' > 'G' is True, but we are testing the __gt__ implementation
    # In Python, for dataclasses, __gt__ compares tuples of fields.
    # 'USD' > 'GBP' -> True
    assert usd > gbp
    
    # Test: GBP > JPY -> True ('G' > 'J' is False, wait: 'G' comes before 'J')
    # Alphabetical order: GBP, JPY, USD. 
    # So: USD > JPY (True), USD > GBP (True), JPY > GBP (True)
    assert usd > jpy
    assert jpy > gbp
    assert not gbp > usd

    # Test comparing same code but different name/decimals to trigger subsequent field comparisons
    # Field 1: code, Field 2: name
    usd_alt_name = Currency.of("USD", "Alternative US Dollars", 2, CurrencyType.MONEY)
    assert usd_alt_name > usd  # 'A' < 'U', so USD > USD_alt is True. Wait, let's check:
    # tuple(usd) = ('USD', 'US Dollars', ...)
    # tuple(usd_alt) = ('USD', 'Alternative US Dollars', ...)
    # Comparing ('USD', 'US Dollars') and ('USD', 'Alternative US Dollars'):
    # 'US Dollars' > 'Alternative US Dollars' is True because 'U' > 'A'.
    assert usd > usd_alt_name

    # Test decimals field impact
    # If code and name are identical, compare decimals
    usd_high_decimal = Currency.of("USD", "US Dollars", 4, CurrencyType.MONEY)
    # ('USD', 'US Dollars', 4, ...) > ('USD', 'US Dollars', 2, ...)
    assert usd_high_decimal > usd

    # Test with error handling (comparing incompatible types)
    with pytest.raises(TypeError):
        assert usd > "Not a currency"
```


# LLM-generated content at query #21
#--------------------------

```python
import pytest

def test_CurrencyRegistry___len__():
    """
    Tests the __len__ method of the CurrencyRegistry class.
    Ensures that len() correctly returns the number of registered currencies
    within a registry context.
    """
    # Reset Singleton instance for a clean testing environment
    CurrencyRegistry._CurrencyRegistry__instance = None
    registry = CurrencyRegistry()

    # Initial length should be 0
    assert len(registry) == 0

    # Define some sample currencies
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    btc = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)

    # Test length during registry population context
    with registry as register:
        register(usd)
        assert len(registry) == 1
        register(jpy)
        assert len(registry) == 2
        register(btc)
        assert len(registry) == 3

    # After exiting the context, the length should remain consistent with registered items
    assert len(registry) == 3
```


# LLM-generated content at query #22
#--------------------------

```python
import pytest
from decimal import Decimal

def test_Currency___lt__():
    """
    Tests the __lt__ (less than) magic method of the Currency class.
    Since the class is decorated with @dataclass(order=True), 
    it uses the order of its fields for comparison.
    The field order is: code, name, decimals, type, quantizer, hashcache.
    """
    # Setup currencies with varying attributes to test ordering logic
    # Case 1: Different codes (Primary sort key)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    gbp = Currency.of("GBP", "British Pounds", 2, CurrencyType.MONEY)
    assert usd > gbp  # 'U' comes after 'G'
    assert gbp < usd

    # Case 2: Same code, different names (Secondary sort key)
    usd_alt = Currency.of("USD", "Alternative USD", 2, CurrencyType.MONEY)
    assert usd_alt < usd  # 'A' comes before 'U'

    # Case 3: Same code and name, different decimals (Tertiary sort key)
    usd_high_prec = Currency.of("USD", "US Dollars", 4, CurrencyType.MONEY)
    assert usd < usd_high_prec  # 2 < 4

    # Case 4: Same code, name, and decimals, different type (Quaternary sort key)
    # Note: Enum comparison depends on member order in Enum definition
    usd_crypto = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    # In CurrencyType: MONEY is index 0/first, CRYPTO is index 2/third
    assert usd < usd_crypto

    # Case 5: Testing equality logic within ordering (Identical attributes)
    usd_duplicate = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd < usd_duplicate)
    assert not (usd_duplicate < usd)
    assert usd == usd_duplicate

    # Case 6: Extreme edge case - testing the sorting of decimals (Negative precision)
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    usd_low_dec = Currency.of("USD", "US Dollars", -1, CurrencyType.MONEY)
    # 'U' < 'Z', so USD should be less than ZZZ regardless of decimals
    assert usd_low_dec < zzz
```


# LLM-generated content at query #23
#--------------------------

```python
import pytest

def test_CurrencyRegistry_get():
    # Reset singleton instance for clean testing environment
    CurrencyRegistry._instance = None
    registry = CurrencyRegistry()
    
    # Create dummy currencies
    usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    alt = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)

    # Populate registry using the context manager
    with registry as register:
        register(usd)
        register(jpy)
        register(alt)

    # Test Case 1: Retrieve existing currency by code
    assert registry.get("USD") == usd
    assert registry.get("JPY") == jpy
    assert registry.get("BTC") == alt

    # Test Case 2: Retrieve non-existent currency with default (None)
    assert registry.get("XYZ") is None

    # Test Case 3: Retrieve non-existent currency with specific default value
    default_currency = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert registry.get("NON_EXISTENT", default=default_currency) == default_currency

    # Test Case 4: Ensure it returns the exact same object instance
    retrieved_usd = registry.get("USD")
    assert retrieved_usd is usd
```


# LLM-generated content at query #24
#--------------------------

```python
import pytest

def test_CurrencyRegistry___len__():
    """
    Tests the __len__ method of CurrencyRegistry to ensure it correctly 
    returns the number of registered currencies.
    """
    # Reset the singleton instance for a clean test environment
    CurrencyRegistry._CurrencyRegistry__instance = None
    registry = CurrencyRegistry()

    # Case 1: Initial length should be 0
    assert len(registry) == 0

    # Case 2: Adding currencies and checking length
    usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    jpy = Currency.to_be_defined_or_mocked_if_needed # Using actual constructor logic
    # Since we can't easily mock the whole module, we use valid data:
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)

    with registry as register:
        register(usd)
        assert len(registry) == 1
        register(eur)
        assert len(registry) == 2
        register(jpy)
        assert len(registry) == 3

    # Case 3: Verify length remains consistent after context exit (re-sorting/finalization)
    with registry as register:
        # Adding a duplicate should trigger ValueError, so we test valid addition
        gbp = Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY)
        register(gbp)
    
    assert len(registry) == 4
```


# LLM-generated content at query #25
#--------------------------

```python
import pytest
from decimal import Decimal

def test_CurrencyRegistry___exit__():
    """
    Tests the __exit__ method of CurrencyRegistry to ensure it correctly 
    finalizes, sorts, and populates the registry buffers after a context manager block.
    """
    # Reset Singleton instance for clean testing environment
    CurrencyRegistry._CurrencyRegistry__instance = None
    registry = CurrencyRegistry()

    # Define some currencies in non-alphabetical order to test sorting logic in __exit__
    ccy_usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy_aed = Currency.of("AED", "UAE Dirham", 2, CurrencyType.MONEY)
    ccy_jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)

    # Use the registry context manager
    with registry as register:
        register(ccy_usd)
        register(ccy_jpy)
        register(ccy_aed)
        # At this point, __ctx_open is True and buffers are not yet sorted/finalized

    # After exiting the block, __exit__ should have been called.
    # 1. Check if context is closed
    assert registry._CurrencyRegistry__ctx_open is False

    # 2. Check if currencies are sorted by code (AED, JPY, USD)
    expected_codes = ["AED", "JPY", "USD"]
    assert registry.codes == expected_codes

    # 3. Check if the 'all' property returns the sorted list of Currency objects
    assert registry.all == [ccy_aed, ccy_jpy, ccy_usd]

    # 4. Check if codenames are correctly populated and sorted
    expected_codenames = [("AED", "UAE Dirham"), ("JPY", "Japanese Yen"), ("USD", "US Dollars")]
    assert registry.codenames == expected_codenames

    # 5. Verify the internal registry dictionary is also sorted by key
    # We check this by iterating through the registry keys
    actual_keys = list(registry._CurrencyRegistry__registry.keys())
    assert actual_keys == expected_codes
```


# LLM-generated content at query #26
#--------------------------

```python
import pytest
from decimal import Decimal

def test_Currency___hash__():
    """
    Tests that the __hash__ method of Currency returns the pre-computed hashcache
    and that identical currencies produce the same hash.
    """
    # Setup two identical currency instances
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    # Setup a different currency instance
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    
    # Verify that hashcache is actually used by __hash__
    assert hash(usd1) == usd1.hashcache
    assert hash(usd2) == usd2.hashcache
    
    # Verify equality of hashes for identical objects
    assert hash(usd1) == hash(usd2)
    
    # Verify inequality of hashes for different objects
    assert hash(usd1) != hash(eur)
    
    # Verify that the hash is consistent with Python's built-in hash behavior for the tuple used in creation
    # (The class computes hash based on: (code, name, decimals, ctype, quantizer))
    expected_internal_hash = hash((
        usd1.code, 
        usd1.name, 
        usd1.decimals, 
        usd1.type, 
        usd1.quantizer
    ))
    assert hash(usd1) == expected_internal_hash
```


# LLM-generated content at query #27
#--------------------------

```python
import pytest

def test_CurrencyRegistry_has():
    """
    Tests the 'has' method of the CurrencyRegistry class.
    Verifies that it correctly identifies registered and unregistered currency codes.
    """
    # Reset singleton instance for clean testing environment
    CurrencyRegistry._instance = None
    registry = CurrencyRegistry()

    # Define test currencies
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)

    # Test case 1: 'has' returns False when registry is empty
    assert registry.has("USD") is False
    assert "USD" in registry is False

    # Register currencies using the context manager
    with registry as register:
        register(usd)
        register(eur)

    # Test case 2: 'has' returns True for registered codes
    assert registry.has("USD") is True
    assert registry.has("EUR") is True
    assert "USD" in registry is True
    assert "EUR" in registry is True

    # Test case 3: 'has' returns False for unregistered codes
    assert registry.has("JPY") is False
    assert registry.has("GBP") is False
    assert "JPY" in registry is False

    # Test case 4: Check behavior with non-existent codes (random string)
    assert registry.has("NONEXISTENT") is False
```


# LLM-generated content at query #28
#--------------------------

```python
import pytest
from decimal import Decimal

def test_Currency___gt__():
    """
    Tests the __gt__ (greater than) method of the Currency class.
    Since the class is decorated with @dataclass(order=True), 
    comparison operators are generated based on the order of fields in the class definition.
    The field order is: code, name, decimals, type, quantizer, hashcache.
    """
    # Create base currency
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    
    # Test case 1: Different code (alphabetically greater)
    zwd = Currency.of("ZWD", "Zimbabwean Dollars", 2, CurrencyType.MONEY)
    assert zwd > usd
    assert not usd > zwd

    # Test case 2: Same code, different name (alphabetically greater)
    usd_alt = Currency.of("USD", "US Alt Dollars", 2, CurrencyType.MONEY)
    assert usd_alt > usd
    assert not usd > usd_alt

    # Test case 3: Same code and name, different decimals
    usd_high_prec = Currency.of("USD", "US Dollars", 4, CurrencyType.MONEY)
    # Note: In dataclass order=True, 'decimals' comes after 'name'. 
    # However, we must ensure the comparison logic follows the field sequence.
    # If code and name are equal, it compares decimals.
    assert usd_high_prec > usd
    assert not usd > usd_high_prec

    # Test case 4: Same code, name, and decimals, different type
    # Since CurrencyType is an Enum, comparison depends on enum member order/value
    # In the definition: MONEY (0), METAL (1), CRYPTO (2), ALTERNATIVE (3)
    btc = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    assert btc > usd
    assert not usd > btc

    # Test case 5: Equality check via __gt__ (should be False)
    usd_duplicate = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd > usd_duplicate)
```


# LLM-generated content at query #29
#--------------------------

```python
import pytest

def test_CurrencyRegistry():
    # Reset singleton instance for a clean test environment
    CurrencyRegistry._instance = None
    
    # Test instantiation and singleton behavior
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2
    
    # Test initial state of the newly created registry
    assert len(registry1) == 0
    assert registry1.all == []
    assert registry1.codes == []
    assert registry1.codenames == []

    # Create valid currency objects for testing registration
    usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    jpy = Currency.to_be_defined_elsewhere_but_we_use_logic # Mocking logic:
    # Since we can't use external mocks easily without imports, we rely on the class logic
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)

    # Test registration via context manager (the intended way)
    with registry1 as register:
        register(usd)
        register(eur)
        # Verify intermediate state before __exit__ (context not yet closed/sorted)
        assert "USD" in registry1
        assert "EUR" in registry1
        
        # Test duplicate registration error
        with pytest.raises(ValueError, match="Currency USD is already registered"):
            register(usd)

    # Test state after context manager exit (__exit__ handles sorting and buffers)
    assert len(registry1) == 2
    assert registry1.codes == ["EUR", "USD"]  # Should be sorted alphabetically by code
    assert registry1.codenames == [("EUR", "Euro"), ("USD", "US Dollar")]
    assert registry1.all == [eur, usd]

    # Test lookup and accessors
    assert registry1["USD"] == usd
    assert registry1.has("USD") is True
    assert registry1.has("GBP") is False
    assert registry1.get("EUR") == eur
    assert registry1.get("GBP", default=usd) == usd

    # Test error handling for invalid lookup
    with pytest.raises(CurrencyLookupError) as excinfo:
        _ = registry1["XYZ"]
    assert "XYZ" in str(excinfo.value)

    # Test registration outside of context raises ProgrammingError
    with pytest.raises(ProgrammingError, match="Can not create currencies outside registry context"):
        registry1.__register(jpy)
```


# LLM-generated content at query #30
#--------------------------

```python
import pytest

def test_CurrencyRegistry___len__():
    """
    Tests the __len__ method of CurrencyRegistry.
    Ensures that len() correctly returns the count of registered currencies
    after they are added via the registry context.
    """
    # Reset singleton instance for a clean testing environment
    CurrencyRegistry._CurrencyRegistry__instance = None
    registry = CurrencyRegistry()

    # Initial length should be 0
    assert len(registry) == 0

    # Create some currencies
    usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    btc = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)

    # Test length within context (registry is not yet finalized/sorted)
    with registry as register:
        register(usd)
        assert len(registry) == 1
        register(jpy)
        assert len(registry) == 2
        register(btc)
        assert len(registry) == 3

    # After context exit, length should remain the same (the count of registered items)
    assert len(registry) == 3

    # Test adding a duplicate within a new context should raise ValueError
    # and not increment the length
    with pytest.raises(ValueError, match="Currency USD is already registered"):
        with registry as register:
            register(usd)
    
    assert len(registry) == 3
```


# LLM-generated content at query #31
#--------------------------

```python
import pytest

def test_CurrencyRegistry___enter__():
    """
    Tests that the __enter__ method of CurrencyRegistry returns the 
    internal register method and sets the context open flag.
    """
    # Reset Singleton instance for clean testing environment
    CurrencyRegistry._instance = None
    registry = CurrencyRegistry()
    
    # Before entering context, we should not be able to register via the internal mechanism
    # (Note: __register is private, but we test the logic of ctx_open)
    
    with registry as register:
        # 1. Check that the return value of __enter__ is a callable (the register method)
        assert callable(register)
        
        # 2. Verify we can use the returned function to add a currency
        usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
        register(usd)
        
        # 3. Check that the currency is now accessible in the registry
        assert "USD" in registry
        assert registry["USD"] == usd

    # 4. After exiting context, check if ctx_open was reset to False
    # Since __ctx_open is private, we verify by attempting to register outside context
    # which should raise a ProgrammingError
    with pytest.raises(ProgrammingError, match="Can not create currencies outside registry context."):
        registry.__register(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))

    # 5. Verify that the exit logic (sorting/finalization) occurred by checking order
    # Add another currency out of alphabetical order
    with registry as register_again:
        register_again(Currency.of("ABC", "Alpha", 0, CurrencyType.MONEY))
    
    # The __exit__ method sorts the registry by code. 
    # Since USD was already there and ABC was added, ABC should now be first.
    assert registry.codes == ["ABC", "USD"]
```


# LLM-generated content at query #32
#--------------------------

```python
import pytest
from decimal import Decimal

def test_Currency___gt__():
    """
    Tests the __gt__ (greater than) magic method of the Currency class.
    Note: The provided implementation uses @dataclass(order=True), 
    which implements comparison methods based on field order.
    The fields in Currency are: code, name, decimals, type, quantizer, hashcache.
    """
    # Create basic currencies for testing
    # USD has 'U' in code, JPY has 'J' in code. 
    # Since 'code' is the first field, comparison starts there.
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    gbp = Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY)
    
    # Test USD > JPY (U > J)
    assert usd > jpy
    
    # Test JPY < USD (J < U)
    assert jpy < usd
    
    # Test GBP > JPY (G > J is False, G comes before J)
    assert not (gbp > jpy)
    assert gbp < usd
    
    # Test equality behavior with __gt__
    # Since USD == USD, USD > USD must be False
    assert not (usd > usd)

    # Test tie-breaking with the second field 'name'
    # Same code, different name: 'US Dollars' vs 'US Dollars Plus'
    usd_plus = Currency.of("USD", "US Dollars Plus", 2, CurrencyType.MONEY)
    assert usd_plus > usd
    assert usd < usd_plus

    # Test tie-breaking with the third field 'decimals'
    # Same code and name, different decimals
    usd_zero_dec = Currency.of("USD", "US Dollars", 0, CurrencyType.MONEY)
    # Note: We have to be careful because the @dataclass(order=True) 
    # compares all fields in order. If 'code' and 'name' are same, it checks decimals.
    # For this test, we need a way to bypass the .of() validation if we want to force 
    # identical code/name but different decimals for pure logic testing, 
    # however, '.of' allows same code/name as long as they are valid strings.
    
    # We use a secondary currency with same code and name but higher decimals
    usd_high_dec = Currency.of("USD", "US Dollars Plus", 4, CurrencyType.MONIC) # This will fail validation if we don't follow rules
    # Since we cannot easily create identical code/name/type with different decimals 
    # without violating the logic of .of() (it would technically be a different object),
    # we rely on the lexicographical order of the fields provided.

    # Final check: verifying the comparison follows the field order defined in the dataclass
    # Field 1: code, Field 2: name, Field 3: decimals...
    assert Currency.of("Z", "A", 0, CurrencyType.MONEY) > Currency.of("Y", "A", 0, CurrencyType.MONEY)
```


# LLM-generated content at query #33
#--------------------------

```python
import pytest

def test_CurrencyRegistry___new__():
    """
    Tests that the __new__ method implements a singleton pattern, 
    ensuring that multiple calls return the exact same instance.
    """
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    
    # Verify both variables point to the same object in memory
    assert registry1 is registry2
    
    # Verify that the internal singleton instance is indeed the one returned
    assert CurrencyRegistry._CurrencyRegistry__instance is registry1
```


# LLM-generated content at query #34
#--------------------------

```python
import pytest
from decimal import Decimal

def test_CurrencyRegistry___exit__():
    """
    Tests that the __exit__ method correctly finalizes the registry by:
    1. Re-sorting the registry by currency code.
    2. Updating internal buffers (currencies, codes, codenames).
    3. Closing the population context flag.
    """
    # Reset singleton instance for clean test environment
    CurrencyRegistry._CurrencyRegistry__instance = None
    registry = CurrencyRegistry()

    # Create currencies in non-alphabetical order
    ccy_b = Currency.of("BHD", "Bahraini Dinar", 3, CurrencyType.MONEY)
    ccy_a = Currency.of("AUD", "Australian Dollar", 2, CurrencyType.MONEY)
    ccy_c = Currency.of("CAD", "Canadian Dollar", 2, CurrencyType.MONEY)

    # Use the context manager to register currencies
    with registry as register:
        register(ccy_b)
        register(ccy_a)
        register(ccy_c)
        # Inside context, ctx_open should be True
        assert registry._CurrencyRegistry__ctx_open is True

    # After exiting the context (__exit__ called):
    # 1. Check if context flag is closed
    assert registry._CurrencyRegistry__ctx_open is False

    # 2. Check if currencies are sorted by code (AUD, BHD, CAD)
    expected_codes = ["AUD", "BHD", "CAD"]
    assert registry.codes == expected_codes
    assert registry.all[0].code == "AUD"
    assert registry.all[1].code == "BHD"
    assert registry.all[2].code == "CAD"

    # 3. Check if codenames buffer is updated and sorted correctly
    expected_codenames = [("AUD", "Australian Dollar"), ("BHD", "Bahraini Dinar"), ("CAD", "Canadian Dollar")]
    assert registry.codenames == expected_codenames

    # 4. Verify the internal dictionary order (registry) matches the sorted list
    actual_registry_keys = list(registry._CurrencyRegistry__registry.keys())
    assert actual_registry_keys == expected_codes
```


# LLM-generated content at query #35
#--------------------------

```python
import pytest
from decimal import Decimal

def test_Currency_quantize():
    # Test USD: 2 decimals (Standard rounding)
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.quantize(Decimal("1.005")) == Decimal("1.00")
    assert usd.quantize(Decimal("1.015")) == Decimal("1.02")
    assert usd.quantize(Decimal("1.555")) == Decimal("1.56")

    # Test JPY: 0 decimals (Integer rounding)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy.quantize(Decimal("0.5")) == Decimal("0")
    assert jpy.quantize(Decimal("1.5")) == Decimal("2")
    assert jpy.quantize(Decimal("100.9")) == Decimal("101")

    # Test Crypto/Special: -1 decimals (High precision rounding)
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert zzz.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert zzz.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")

    # Test edge case: Exactly zero
    zero_val = Decimal("0.000")
    assert usd.quantize(zero_val) == Decimal("0.00")
```


# LLM-generated content at query #36
#--------------------------

```python
import pytest

def test_CurrencyRegistry___getitem__():
    """
    Tests the __getitem__ method of CurrencyRegistry.
    Verifies that it returns the correct currency when the code exists,
    and raises CurrencyLookupError when the code does not exist.
    """
    # Reset singleton instance for clean testing environment
    CurrencyRegistry.__instance = None
    registry = CurrencyRegistry()
    
    usd_code = "USD"
    usd_currency = Currency.of(usd_code, "US Dollars", 2, CurrencyType.MONEY)
    non_existent_code = "XYZ"

    # Test case: Accessing an existing currency via __getitem__
    # We must use the registry context to register the currency first
    with registry as register:
        register(usd_currency)

    # Assert that __getitem__ returns the correct Currency object
    assert registry[usd_code] == usd_currency
    assert registry[usd_code].code == usd_code
    assert registry[usd_code].name == "US Dollars"

    # Test case: Accessing a non-existent currency via __getitem__
    # Should raise CurrencyLookupError with the correct message
    with pytest.raises(CurrencyLookupError) as excinfo:
        _ = registry[non_existent_code]
    
    assert str(excinfo.value) == f"Currency identified by code '{non_existent_code}' does not exist"
    assert excinfo.value.code == non_existent_code
```


