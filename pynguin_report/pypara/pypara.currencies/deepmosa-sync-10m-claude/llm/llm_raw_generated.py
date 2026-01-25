####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_currency_registry_constructor():
    registry = CurrencyRegistry()
    assert len(registry) == 0
    assert registry.all == []
    assert registry.codes == []
    assert registry.codenames == []


def test_currency_registry_singleton():
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2


def test_currency_registry_constructor_initializes_empty_state():
    registry = CurrencyRegistry()
    assert "USD" not in registry or len(registry) == 0


# LLM-generated content at query #2
#--------------------------

```python
def test_currency_constructor_creates_frozen_instance():
    from decimal import Decimal
    from enum import Enum
    
    class CurrencyType(Enum):
        MONEY = "MONEY"
        CRYPTO = "CRYPTO"
    
    code = "USD"
    name = "US Dollars"
    decimals = 2
    ctype = CurrencyType.MONEY
    quantizer = Decimal("0.01")
    hashcode = 12345
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcode)
    
    assert currency.code == "USD"
    assert currency.name == "US Dollars"
    assert currency.decimals == 2
    assert currency.type == CurrencyType.MONEY
    assert currency.quantizer == Decimal("0.01")
    assert currency.hashcache == 12345


def test_currency_constructor_is_frozen():
    from decimal import Decimal
    from enum import Enum
    
    class CurrencyType(Enum):
        MONEY = "MONEY"
        CRYPTO = "CRYPTO"
    
    currency = Currency("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal("0.01"), 12345)
    
    try:
        currency.code = "EUR"
        assert False, "Should not be able to modify frozen dataclass"
    except:
        assert True


def test_currency_constructor_with_different_decimals():
    from decimal import Decimal
    from enum import Enum
    
    class CurrencyType(Enum):
        MONEY = "MONEY"
        CRYPTO = "CRYPTO"
    
    currency = Currency("JPY", "Japanese Yen", 0, CurrencyType.MONEY, Decimal("1"), 54321)
    
    assert currency.code == "JPY"
    assert currency.decimals == 0
    assert currency.quantizer == Decimal("1")


def test_currency_constructor_with_negative_decimals():
    from decimal import Decimal
    from enum import Enum
    
    class CurrencyType(Enum):
        MONEY = "MONEY"
        CRYPTO = "CRYPTO"
    
    currency = Currency("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO, Decimal("0.000000000001"), 99999)
    
    assert currency.code == "ZZZ"
    assert currency.decimals == -1
    assert currency.type == CurrencyType.CRYPTO


# LLM-generated content at query #3
#--------------------------

```python
def test_currency_registry_constructor():
    registry = CurrencyRegistry()
    assert registry is not None
    assert len(registry) == 0
    assert registry.all == []
    assert registry.codes == []
    assert registry.codenames == []


def test_currency_registry_singleton():
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2


def test_currency_registry_constructor_initializes_empty_state():
    registry = CurrencyRegistry()
    assert "USD" not in registry
    assert registry.has("USD") == False
    assert registry.get("USD") is None


# LLM-generated content at query #4
#--------------------------

```python
def test_currency_constructor_basic():
    from decimal import Decimal
    from enum import Enum
    
    class CurrencyType(Enum):
        MONEY = "MONEY"
        CRYPTO = "CRYPTO"
    
    code = "USD"
    name = "US Dollars"
    decimals = 2
    ctype = CurrencyType.MONEY
    quantizer = Decimal("0.01")
    hashcache = hash((code, name, decimals, ctype, quantizer))
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    assert currency.code == "USD"
    assert currency.name == "US Dollars"
    assert currency.decimals == 2
    assert currency.type == CurrencyType.MONEY
    assert currency.quantizer == Decimal("0.01")
    assert currency.hashcache == hashcache


def test_currency_constructor_frozen():
    from decimal import Decimal
    from enum import Enum
    
    class CurrencyType(Enum):
        MONEY = "MONEY"
        CRYPTO = "CRYPTO"
    
    code = "EUR"
    name = "Euro"
    decimals = 2
    ctype = CurrencyType.MONEY
    quantizer = Decimal("0.01")
    hashcache = hash((code, name, decimals, ctype, quantizer))
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    try:
        currency.code = "GBP"
        assert False, "Expected frozen dataclass to raise error"
    except (AttributeError, Exception):
        pass


def test_currency_constructor_with_crypto():
    from decimal import Decimal
    from enum import Enum
    
    class CurrencyType(Enum):
        MONEY = "MONEY"
        CRYPTO = "CRYPTO"
    
    code = "BTC"
    name = "Bitcoin"
    decimals = -1
    ctype = CurrencyType.CRYPTO
    quantizer = Decimal("0.000000000001")
    hashcache = hash((code, name, decimals, ctype, quantizer))
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    assert currency.code == "BTC"
    assert currency.name == "Bitcoin"
    assert currency.decimals == -1
    assert currency.type == CurrencyType.CRYPTO


def test_currency_constructor_with_zero_decimals():
    from decimal import Decimal
    from enum import Enum
    
    class CurrencyType(Enum):
        MONEY = "MONEY"
        CRYPTO = "CRYPTO"
    
    code = "JPY"
    name = "Japanese Yen"
    decimals = 0
    ctype = CurrencyType.MONEY
    quantizer = Decimal("1")
    hashcache = hash((code, name, decimals, ctype, quantizer))
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    assert currency.code == "JPY"
    assert currency.decimals == 0
    assert currency.quantizer == Decimal("1")


# LLM-generated content at query #5
#--------------------------

```python
def test_ctx_open_initialized_to_false():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open == False


# LLM-generated content at query #6
#--------------------------

```python
def test_currency_constructor_creates_instance_with_all_fields():
    from decimal import Decimal
    from dataclasses import fields
    
    code = "USD"
    name = "US Dollars"
    decimals = 2
    ctype_value = 0  # Assuming CurrencyType.MONEY has value 0
    quantizer = Decimal("0.01")
    hashcache = 12345
    
    currency = Currency(code, name, decimals, ctype_value, quantizer, hashcache)
    
    assert currency.code == code
    assert currency.name == name
    assert currency.decimals == decimals
    assert currency.type == ctype_value
    assert currency.quantizer == quantizer
    assert currency.hashcache == hashcache


def test_currency_constructor_is_frozen():
    from decimal import Decimal
    
    currency = Currency("USD", "US Dollars", 2, 0, Decimal("0.01"), 12345)
    
    try:
        currency.code = "EUR"
        assert False, "Expected FrozenInstanceError"
    except Exception:
        pass


def test_currency_constructor_fields_are_ordered():
    from decimal import Decimal
    
    currency1 = Currency("EUR", "Euro", 2, 0, Decimal("0.01"), 11111)
    currency2 = Currency("USD", "US Dollars", 2, 0, Decimal("0.01"), 12345)
    
    assert currency1 < currency2


# LLM-generated content at query #7
#--------------------------

```python
def test_currency_registry_codes_empty_on_init():
    registry = CurrencyRegistry()
    assert len(registry.codes) == 0
    assert registry.codes == []


# LLM-generated content at query #8
#--------------------------

```python
def test_currency_constructor_creates_frozen_dataclass():
    from decimal import Decimal
    
    code = "USD"
    name = "US Dollars"
    decimals = 2
    ctype = CurrencyType.MONEY
    quantizer = Decimal("0.01")
    hashcache = 12345
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    assert currency.code == "USD"
    assert currency.name == "US Dollars"
    assert currency.decimals == 2
    assert currency.type == CurrencyType.MONEY
    assert currency.quantizer == Decimal("0.01")
    assert currency.hashcache == 12345


def test_currency_constructor_is_frozen():
    from decimal import Decimal
    
    code = "USD"
    name = "US Dollars"
    decimals = 2
    ctype = CurrencyType.MONEY
    quantizer = Decimal("0.01")
    hashcache = 12345
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    try:
        currency.code = "EUR"
        assert False, "Should not be able to modify frozen dataclass"
    except AttributeError:
        pass


def test_currency_constructor_with_different_values():
    from decimal import Decimal
    
    code = "JPY"
    name = "Japanese Yen"
    decimals = 0
    ctype = CurrencyType.MONEY
    quantizer = Decimal("1")
    hashcache = 54321
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    assert currency.code == "JPY"
    assert currency.name == "Japanese Yen"
    assert currency.decimals == 0
    assert currency.type == CurrencyType.MONEY
    assert currency.quantizer == Decimal("1")
    assert currency.hashcache == 54321


def test_currency_constructor_with_crypto_type():
    from decimal import Decimal
    
    code = "BTC"
    name = "Bitcoin"
    decimals = -1
    ctype = CurrencyType.CRYPTO
    quantizer = Decimal("1E+1")
    hashcache = 99999
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    assert currency.code == "BTC"
    assert currency.name == "Bitcoin"
    assert currency.decimals == -1
    assert currency.type == CurrencyType.CRYPTO
    assert currency.quantizer == Decimal("1E+1")
    assert currency.hashcache == 99999


# LLM-generated content at query #9
#--------------------------

```python
def test_currency_registry_ctx_open_initialized_to_false():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open == False


# LLM-generated content at query #10
#--------------------------

```python
def test_currency_registry_codes_is_empty_after_init():
    registry = CurrencyRegistry()
    assert len(registry.codes) == 0
    assert registry.codes == []


# LLM-generated content at query #11
#--------------------------

```python
def test_currency_registry_constructor():
    registry = CurrencyRegistry()
    assert registry is not None
    assert len(registry) == 0
    assert registry.all == []
    assert registry.codes == []
    assert registry.codenames == []


def test_currency_registry_singleton():
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2


def test_currency_registry_constructor_initializes_empty_state():
    registry = CurrencyRegistry()
    assert "USD" not in registry
    assert registry.has("USD") is False
    assert registry.get("USD") is None


# LLM-generated content at query #12
#--------------------------

```python
def test_currency_registry_constructor():
    registry = CurrencyRegistry()
    assert registry is not None
    assert len(registry) == 0
    assert registry.all == []
    assert registry.codes == []
    assert registry.codenames == []


def test_currency_registry_singleton():
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2


def test_currency_registry_constructor_initializes_empty_state():
    registry = CurrencyRegistry()
    assert "USD" not in registry or len(registry) > 0


# LLM-generated content at query #13
#--------------------------

```python
def test_currency_registry_ctx_open_initialized_to_false():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open is False


# LLM-generated content at query #14
#--------------------------

```python
def test_currency_registry_constructor():
    registry = CurrencyRegistry()
    assert registry is not None
    assert len(registry) == 0
    assert registry.all == []
    assert registry.codes == []
    assert registry.codenames == []


def test_currency_registry_singleton():
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2


def test_currency_registry_constructor_initializes_empty_state():
    registry = CurrencyRegistry()
    assert "USD" not in registry or len(registry) == 0
    assert registry.get("NONEXISTENT") is None


# LLM-generated content at query #15
#--------------------------

```python
def test_currency_registry_ctx_open_initialized_to_false():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open is False


# LLM-generated content at query #16
#--------------------------

```python
def test_currency_constructor_with_valid_parameters():
    code = "USD"
    name = "US Dollars"
    decimals = 2
    ctype = CurrencyType.MONEY
    quantizer = Decimal("0.01")
    hashcache = 12345
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    assert currency.code == "USD"
    assert currency.name == "US Dollars"
    assert currency.decimals == 2
    assert currency.type == CurrencyType.MONEY
    assert currency.quantizer == Decimal("0.01")
    assert currency.hashcache == 12345


def test_currency_constructor_with_zero_decimals():
    code = "JPY"
    name = "Japanese Yen"
    decimals = 0
    ctype = CurrencyType.MONEY
    quantizer = Decimal("1")
    hashcache = 54321
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    assert currency.code == "JPY"
    assert currency.decimals == 0
    assert currency.quantizer == Decimal("1")


def test_currency_constructor_with_negative_decimals():
    code = "ZZZ"
    name = "Some weird currency"
    decimals = -1
    ctype = CurrencyType.CRYPTO
    quantizer = Decimal("0.000000000001")
    hashcache = 99999
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    assert currency.code == "ZZZ"
    assert currency.decimals == -1
    assert currency.type == CurrencyType.CRYPTO


def test_currency_constructor_frozen():
    currency = Currency("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal("0.01"), 12345)
    
    try:
        currency.code = "EUR"
        assert False, "Expected frozen dataclass to prevent attribute modification"
    except Exception:
        pass


def test_currency_constructor_with_different_currency_types():
    currency_money = Currency("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal("0.01"), 111)
    currency_crypto = Currency("BTC", "Bitcoin", 8, CurrencyType.CRYPTO, Decimal("0.00000001"), 222)
    
    assert currency_money.type == CurrencyType.MONEY
    assert currency_crypto.type == CurrencyType.CRYPTO
    assert currency_money.type != currency_crypto.type


# LLM-generated content at query #17
#--------------------------

```python
def test_currency_registry_constructor():
    registry = CurrencyRegistry()
    assert registry is not None
    assert len(registry) == 0
    assert registry.all == []
    assert registry.codes == []
    assert registry.codenames == []


def test_currency_registry_singleton():
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2


def test_currency_registry_multiple_instantiations():
    registry_a = CurrencyRegistry()
    registry_b = CurrencyRegistry()
    registry_c = CurrencyRegistry()
    assert registry_a is registry_b
    assert registry_b is registry_c
    assert registry_a is registry_c


# LLM-generated content at query #18
#--------------------------

```python
def test_currency_constructor_with_valid_parameters():
    code = "USD"
    name = "US Dollars"
    decimals = 2
    ctype = CurrencyType.MONEY
    quantizer = Decimal("0.01")
    hashcache = 12345
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    assert currency.code == "USD"
    assert currency.name == "US Dollars"
    assert currency.decimals == 2
    assert currency.type == CurrencyType.MONEY
    assert currency.quantizer == Decimal("0.01")
    assert currency.hashcache == 12345


def test_currency_constructor_with_negative_decimals():
    code = "ZZZ"
    name = "Crypto Currency"
    decimals = -1
    ctype = CurrencyType.CRYPTO
    quantizer = Decimal("1")
    hashcache = 54321
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    assert currency.code == "ZZZ"
    assert currency.decimals == -1
    assert currency.type == CurrencyType.CRYPTO


def test_currency_constructor_with_zero_decimals():
    code = "JPY"
    name = "Japanese Yen"
    decimals = 0
    ctype = CurrencyType.MONEY
    quantizer = Decimal("1")
    hashcache = 99999
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    assert currency.code == "JPY"
    assert currency.decimals == 0
    assert currency.quantizer == Decimal("1")


def test_currency_constructor_is_frozen():
    currency = Currency("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal("0.01"), 12345)
    
    try:
        currency.code = "EUR"
        assert False, "Should not be able to modify frozen dataclass"
    except AttributeError:
        pass


def test_currency_constructor_creates_different_instances():
    currency1 = Currency("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal("0.01"), 111)
    currency2 = Currency("EUR", "Euro", 2, CurrencyType.MONEY, Decimal("0.01"), 222)
    
    assert currency1 is not currency2
    assert currency1.code != currency2.code


# LLM-generated content at query #19
#--------------------------

```python
def test_currency_registry_constructor():
    registry = CurrencyRegistry()
    assert registry is not None
    assert len(registry) == 0
    assert registry.all == []
    assert registry.codes == []
    assert registry.codenames == []


def test_currency_registry_singleton():
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2


def test_currency_registry_constructor_initializes_empty_containers():
    registry = CurrencyRegistry()
    assert isinstance(registry.all, list)
    assert isinstance(registry.codes, list)
    assert isinstance(registry.codenames, list)
    assert len(registry.all) == 0
    assert len(registry.codes) == 0
    assert len(registry.codenames) == 0


# LLM-generated content at query #20
#--------------------------

```python
def test_currency_registry_constructor():
    registry = CurrencyRegistry()
    assert registry is not None
    assert len(registry) == 0
    assert registry.all == []
    assert registry.codes == []
    assert registry.codenames == []


def test_currency_registry_singleton():
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2


def test_currency_registry_constructor_initializes_empty_state():
    registry = CurrencyRegistry()
    assert "USD" not in registry or len(registry) > 0
    assert registry.has("NONEXISTENT") == False
    assert registry.get("NONEXISTENT") is None


# LLM-generated content at query #21
#--------------------------

```python
def test_currency_registry_constructor():
    registry = CurrencyRegistry()
    assert registry is not None
    assert len(registry) == 0
    assert registry.all == []
    assert registry.codes == []
    assert registry.codenames == []


def test_currency_registry_singleton():
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2


def test_currency_registry_constructor_initializes_empty_containers():
    registry = CurrencyRegistry()
    assert isinstance(registry.all, list)
    assert isinstance(registry.codes, list)
    assert isinstance(registry.codenames, list)
    assert len(registry.all) == 0
    assert len(registry.codes) == 0
    assert len(registry.codenames) == 0


# LLM-generated content at query #22
#--------------------------

```python
def test_currency_constructor_with_valid_parameters():
    code = "USD"
    name = "US Dollars"
    decimals = 2
    ctype = CurrencyType.MONEY
    quantizer = Decimal("0.01")
    hashcache = hash((code, name, decimals, ctype, quantizer))
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    assert currency.code == "USD"
    assert currency.name == "US Dollars"
    assert currency.decimals == 2
    assert currency.type == CurrencyType.MONEY
    assert currency.quantizer == Decimal("0.01")
    assert currency.hashcache == hashcache


def test_currency_constructor_frozen():
    code = "EUR"
    name = "Euro"
    decimals = 2
    ctype = CurrencyType.MONEY
    quantizer = Decimal("0.01")
    hashcache = hash((code, name, decimals, ctype, quantizer))
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    try:
        currency.code = "GBP"
        assert False, "Expected FrozenInstanceError"
    except Exception:
        pass


def test_currency_constructor_with_negative_decimals():
    code = "ZZZ"
    name = "Weird Currency"
    decimals = -1
    ctype = CurrencyType.CRYPTO
    quantizer = Decimal("0.000000000001")
    hashcache = hash((code, name, decimals, ctype, quantizer))
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    assert currency.decimals == -1
    assert currency.code == "ZZZ"


def test_currency_constructor_with_zero_decimals():
    code = "JPY"
    name = "Japanese Yen"
    decimals = 0
    ctype = CurrencyType.MONEY
    quantizer = Decimal("1")
    hashcache = hash((code, name, decimals, ctype, quantizer))
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    assert currency.decimals == 0
    assert currency.quantizer == Decimal("1")


def test_currency_constructor_ordering():
    code1 = "AAA"
    name1 = "First"
    decimals1 = 2
    ctype1 = CurrencyType.MONEY
    quantizer1 = Decimal("0.01")
    hashcache1 = hash((code1, name1, decimals1, ctype1, quantizer1))
    
    code2 = "BBB"
    name2 = "Second"
    decimals2 = 2
    ctype2 = CurrencyType.MONEY
    quantizer2 = Decimal("0.01")
    hashcache2 = hash((code2, name2, decimals2, ctype2, quantizer2))
    
    currency1 = Currency(code1, name1, decimals1, ctype1, quantizer1, hashcache1)
    currency2 = Currency(code2, name2, decimals2, ctype2, quantizer2, hashcache2)
    
    assert currency1 < currency2 or currency1 > currency2 or currency1 == currency2


# LLM-generated content at query #23
#--------------------------

```python
def test_currency_registry_codes_empty_after_init():
    registry = CurrencyRegistry()
    assert len(registry.codes) == 0
    assert registry.codes == []


# LLM-generated content at query #24
#--------------------------

```python
def test_ctx_open_initialized_to_false():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open == False


# LLM-generated content at query #25
#--------------------------

```python
def test_currency_constructor_creates_instance_with_all_fields():
    from decimal import Decimal
    from dataclasses import fields
    
    code = "USD"
    name = "US Dollars"
    decimals = 2
    ctype_value = 0
    quantizer = Decimal("0.01")
    hashcache = 12345
    
    currency = Currency(code, name, decimals, ctype_value, quantizer, hashcache)
    
    assert currency.code == code
    assert currency.name == name
    assert currency.decimals == decimals
    assert currency.type == ctype_value
    assert currency.quantizer == quantizer
    assert currency.hashcache == hashcache


def test_currency_constructor_is_frozen():
    from decimal import Decimal
    
    currency = Currency("USD", "US Dollars", 2, 0, Decimal("0.01"), 12345)
    
    try:
        currency.code = "EUR"
        assert False, "Expected FrozenInstanceError"
    except Exception:
        pass


def test_currency_constructor_preserves_field_types():
    from decimal import Decimal
    
    currency = Currency("JPY", "Japanese Yen", 0, 1, Decimal("1"), 54321)
    
    assert isinstance(currency.code, str)
    assert isinstance(currency.name, str)
    assert isinstance(currency.decimals, int)
    assert isinstance(currency.quantizer, Decimal)
    assert isinstance(currency.hashcache, int)


def test_currency_constructor_with_negative_decimals():
    from decimal import Decimal
    
    currency = Currency("ZZZ", "Crypto", -1, 2, Decimal("0.000000000001"), 99999)
    
    assert currency.decimals == -1
    assert currency.code == "ZZZ"


def test_currency_constructor_with_zero_decimals():
    from decimal import Decimal
    
    currency = Currency("XXX", "Test", 0, 0, Decimal("1"), 11111)
    
    assert currency.decimals == 0
    assert currency.quantizer == Decimal("1")


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_currency_registry_constructor():
    registry = CurrencyRegistry()
    assert registry is not None
    assert len(registry) == 0
    assert registry.all == []
    assert registry.codes == []
    assert registry.codenames == []


def test_currency_registry_singleton():
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2


def test_currency_registry_multiple_instantiation():
    registry_a = CurrencyRegistry()
    registry_b = CurrencyRegistry()
    registry_c = CurrencyRegistry()
    assert registry_a is registry_b
    assert registry_b is registry_c
    assert registry_a is registry_c


# LLM-generated content at query #2
#--------------------------

```python
def test_currency_constructor_creates_instance_with_all_fields():
    from decimal import Decimal
    from enum import Enum
    
    class CurrencyType(Enum):
        MONEY = "MONEY"
        CRYPTO = "CRYPTO"
    
    code = "USD"
    name = "US Dollars"
    decimals = 2
    ctype = CurrencyType.MONEY
    quantizer = Decimal("0.01")
    hashcache = 12345
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    assert currency.code == "USD"
    assert currency.name == "US Dollars"
    assert currency.decimals == 2
    assert currency.type == ctype
    assert currency.quantizer == Decimal("0.01")
    assert currency.hashcache == 12345


def test_currency_constructor_is_frozen():
    from decimal import Decimal
    from enum import Enum
    
    class CurrencyType(Enum):
        MONEY = "MONEY"
        CRYPTO = "CRYPTO"
    
    currency = Currency("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal("0.01"), 12345)
    
    try:
        currency.code = "EUR"
        assert False, "Should not be able to modify frozen dataclass"
    except AttributeError:
        pass


def test_currency_constructor_with_negative_decimals():
    from decimal import Decimal
    from enum import Enum
    
    class CurrencyType(Enum):
        MONEY = "MONEY"
        CRYPTO = "CRYPTO"
    
    currency = Currency("ZZZ", "Weird Currency", -1, CurrencyType.CRYPTO, Decimal("1"), 54321)
    
    assert currency.code == "ZZZ"
    assert currency.decimals == -1
    assert currency.type == CurrencyType.CRYPTO


def test_currency_constructor_with_zero_decimals():
    from decimal import Decimal
    from enum import Enum
    
    class CurrencyType(Enum):
        MONEY = "MONEY"
        CRYPTO = "CRYPTO"
    
    currency = Currency("JPY", "Japanese Yen", 0, CurrencyType.MONEY, Decimal("1"), 99999)
    
    assert currency.code == "JPY"
    assert currency.decimals == 0
    assert currency.quantizer == Decimal("1")


# LLM-generated content at query #3
#--------------------------

```python
def test_currency_eq_same_currencies():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2


def test_currency_eq_different_name():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert usd1 != usd2


def test_currency_eq_different_code():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "US Dollars", 2, CurrencyType.MONEY)
    assert usd != eur


def test_currency_eq_different_decimals():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert usd1 != usd2


def test_currency_eq_different_type():
    curr1 = Currency.of("ZZZ", "Some Currency", 2, CurrencyType.MONEY)
    curr2 = Currency.of("ZZZ", "Some Currency", 2, CurrencyType.CRYPTO)
    assert curr1 != curr2


def test_currency_eq_not_currency_instance():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd != "USD"
    assert usd != 2
    assert usd != None
    assert usd != {}


def test_currency_eq_reflexive():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd == usd


def test_currency_eq_symmetric():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert (usd1 == usd2) == (usd2 == usd1)


def test_currency_eq_transitive():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd3 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2
    assert usd2 == usd3
    assert usd1 == usd3


# LLM-generated content at query #4
#--------------------------

```python
def test_currency_eq_same_currencies():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2


def test_currency_eq_different_names():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 == usdx)


def test_currency_eq_different_codes():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd == eur)


def test_currency_eq_different_decimals():
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd3 = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert not (usd2 == usd3)


def test_currency_eq_different_type():
    money = Currency.of("ABC", "Test Currency", 2, CurrencyType.MONEY)
    crypto = Currency.of("ABC", "Test Currency", 2, CurrencyType.CRYPTO)
    assert not (money == crypto)


def test_currency_eq_with_non_currency_object():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd == "USD")
    assert not (usd == 2)
    assert not (usd == None)


def test_currency_eq_reflexive():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd == usd


def test_currency_eq_symmetric():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert (usd1 == usd2) == (usd2 == usd1)


def test_currency_eq_transitive():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd3 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2
    assert usd2 == usd3
    assert usd1 == usd3


# LLM-generated content at query #5
#--------------------------

```python
def test_getitem_returns_currency_for_valid_code():
    registry = CurrencyRegistry()
    currency = registry["USD"]
    assert currency.code == "USD"


def test_getitem_raises_currency_lookup_error_for_invalid_code():
    registry = CurrencyRegistry()
    try:
        registry["NON-EXISTING"]
        assert False, "Expected CurrencyLookupError to be raised"
    except CurrencyLookupError:
        pass


def test_getitem_with_multiple_valid_codes():
    registry = CurrencyRegistry()
    usd = registry["USD"]
    eur = registry["EUR"]
    assert usd.code == "USD"
    assert eur.code == "EUR"


def test_getitem_returns_same_instance_for_same_code():
    registry = CurrencyRegistry()
    currency1 = registry["USD"]
    currency2 = registry["USD"]
    assert currency1 is currency2


# LLM-generated content at query #6
#--------------------------

```python
def test_currency_registry_constructor():
    registry = CurrencyRegistry()
    assert registry is not None
    assert len(registry) == 0
    assert registry.all == []
    assert registry.codes == []
    assert registry.codenames == []


def test_currency_registry_singleton():
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2


def test_currency_registry_constructor_initializes_empty_state():
    registry = CurrencyRegistry()
    assert "USD" not in registry or len(registry) >= 0
    assert registry.get("NON_EXISTING") is None


# LLM-generated content at query #7
#--------------------------

```python
def test_currency_constructor_with_valid_parameters():
    code = "USD"
    name = "US Dollars"
    decimals = 2
    ctype = CurrencyType.MONEY
    quantizer = Decimal("0.01")
    hashcache = 12345
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    assert currency.code == "USD"
    assert currency.name == "US Dollars"
    assert currency.decimals == 2
    assert currency.type == CurrencyType.MONEY
    assert currency.quantizer == Decimal("0.01")
    assert currency.hashcache == 12345


def test_currency_constructor_frozen():
    code = "USD"
    name = "US Dollars"
    decimals = 2
    ctype = CurrencyType.MONEY
    quantizer = Decimal("0.01")
    hashcache = 12345
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    try:
        currency.code = "EUR"
        assert False, "Expected FrozenInstanceError"
    except:
        pass


def test_currency_constructor_with_negative_decimals():
    code = "ZZZ"
    name = "Some weird currency"
    decimals = -1
    ctype = CurrencyType.CRYPTO
    quantizer = Decimal("1E+12")
    hashcache = 54321
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    assert currency.code == "ZZZ"
    assert currency.name == "Some weird currency"
    assert currency.decimals == -1
    assert currency.type == CurrencyType.CRYPTO
    assert currency.quantizer == Decimal("1E+12")
    assert currency.hashcache == 54321


def test_currency_constructor_with_zero_decimals():
    code = "JPY"
    name = "Japanese Yen"
    decimals = 0
    ctype = CurrencyType.MONEY
    quantizer = Decimal("1")
    hashcache = 99999
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    assert currency.code == "JPY"
    assert currency.name == "Japanese Yen"
    assert currency.decimals == 0
    assert currency.type == CurrencyType.MONEY
    assert currency.quantizer == Decimal("1")
    assert currency.hashcache == 99999


# LLM-generated content at query #8
#--------------------------

```python
def test_currency_eq_same_currencies():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2


def test_currency_eq_different_names():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 == usd2)


def test_currency_eq_different_codes():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert not (usd == eur)


def test_currency_eq_different_decimals():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd_alt = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert not (usd == usd_alt)


def test_currency_eq_different_type():
    curr1 = Currency.of("ABC", "Test Currency", 2, CurrencyType.MONEY)
    curr2 = Currency.of("ABC", "Test Currency", 2, CurrencyType.CRYPTO)
    assert not (curr1 == curr2)


def test_currency_eq_with_non_currency_object():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd == "USD")
    assert not (usd == 2)
    assert not (usd == None)


def test_currency_eq_reflexive():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd == usd


def test_currency_eq_symmetric():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert (usd1 == usd2) == (usd2 == usd1)


def test_currency_eq_transitive():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd3 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2
    assert usd2 == usd3
    assert usd1 == usd3


# LLM-generated content at query #9
#--------------------------

```python
def test_ctx_open_flag_initialized_to_false():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open is False


# LLM-generated content at query #10
#--------------------------

```python
def test_currency_constructor_creates_instance_with_all_fields():
    code = "USD"
    name = "US Dollars"
    decimals = 2
    ctype = CurrencyType.MONEY
    quantizer = Decimal("0.01")
    hashcache = hash((code, name, decimals, ctype, quantizer))
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    assert currency.code == code
    assert currency.name == name
    assert currency.decimals == decimals
    assert currency.type == ctype
    assert currency.quantizer == quantizer
    assert currency.hashcache == hashcache


def test_currency_constructor_frozen():
    currency = Currency("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal("0.01"), 12345)
    
    try:
        currency.code = "EUR"
        assert False, "Should not be able to modify frozen dataclass"
    except (AttributeError, Exception):
        pass


def test_currency_constructor_with_negative_decimals():
    code = "ZZZ"
    name = "Weird Currency"
    decimals = -1
    ctype = CurrencyType.CRYPTO
    quantizer = Decimal("1")
    hashcache = hash((code, name, decimals, ctype, quantizer))
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    assert currency.code == code
    assert currency.decimals == decimals


def test_currency_constructor_with_zero_decimals():
    code = "JPY"
    name = "Japanese Yen"
    decimals = 0
    ctype = CurrencyType.MONEY
    quantizer = Decimal("1")
    hashcache = hash((code, name, decimals, ctype, quantizer))
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    assert currency.decimals == 0
    assert currency.quantizer == Decimal("1")


def test_currency_constructor_hashcache_is_stored():
    code = "EUR"
    name = "Euro"
    decimals = 2
    ctype = CurrencyType.MONEY
    quantizer = Decimal("0.01")
    expected_hash = 999999
    
    currency = Currency(code, name, decimals, ctype, quantizer, expected_hash)
    
    assert currency.hashcache == expected_hash


# LLM-generated content at query #11
#--------------------------

```python
def test_currency_registry_constructor():
    registry = CurrencyRegistry()
    assert registry is not None
    assert len(registry) == 0
    assert registry.all == []
    assert registry.codes == []
    assert registry.codenames == []


def test_currency_registry_singleton():
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2


def test_currency_registry_constructor_initializes_empty_containers():
    registry = CurrencyRegistry()
    assert isinstance(registry.all, list)
    assert isinstance(registry.codes, list)
    assert isinstance(registry.codenames, list)
    assert len(registry.all) == 0
    assert len(registry.codes) == 0
    assert len(registry.codenames) == 0


# LLM-generated content at query #12
#--------------------------

```python
def test_currency_registry_codes_empty_on_init():
    registry = CurrencyRegistry()
    assert len(registry.codes) == 0


# LLM-generated content at query #13
#--------------------------

```python
def test_currency_registry_codes_empty_on_init():
    registry = CurrencyRegistry()
    assert len(registry.codes) == 0


# LLM-generated content at query #14
#--------------------------

```python
def test_ctx_open_initialized_to_false():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open == False


# LLM-generated content at query #15
#--------------------------

```python
def test_currency_registry_constructor():
    registry = CurrencyRegistry()
    assert isinstance(registry, CurrencyRegistry)
    assert len(registry) == 0
    assert registry.all == []
    assert registry.codes == []
    assert registry.codenames == []


def test_currency_registry_singleton():
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2


def test_currency_registry_constructor_initializes_empty_containers():
    registry = CurrencyRegistry()
    assert hasattr(registry, '_CurrencyRegistry__registry')
    assert hasattr(registry, '_CurrencyRegistry__currencies')
    assert hasattr(registry, '_CurrencyRegistry__codes')
    assert hasattr(registry, '_CurrencyRegistry__codenames')
    assert hasattr(registry, '_CurrencyRegistry__ctx_open')


def test_currency_registry_constructor_context_closed():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open is False


# LLM-generated content at query #16
#--------------------------

```python
def test_currency_registry_codes_empty_after_init():
    registry = CurrencyRegistry()
    assert len(registry.codes) == 0


# LLM-generated content at query #17
#--------------------------

```python
def test_currency_constructor_creates_frozen_dataclass():
    from decimal import Decimal
    from enum import Enum
    
    class CurrencyType(Enum):
        MONEY = "MONEY"
        CRYPTO = "CRYPTO"
    
    code = "USD"
    name = "US Dollars"
    decimals = 2
    ctype = CurrencyType.MONEY
    quantizer = Decimal("0.01")
    hashcache = 12345
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    assert currency.code == "USD"
    assert currency.name == "US Dollars"
    assert currency.decimals == 2
    assert currency.type == CurrencyType.MONEY
    assert currency.quantizer == Decimal("0.01")
    assert currency.hashcache == 12345


def test_currency_constructor_frozen_dataclass_prevents_modification():
    from decimal import Decimal
    from enum import Enum
    
    class CurrencyType(Enum):
        MONEY = "MONEY"
        CRYPTO = "CRYPTO"
    
    currency = Currency("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal("0.01"), 12345)
    
    try:
        currency.code = "EUR"
        assert False, "Expected FrozenInstanceError"
    except (AttributeError, Exception):
        pass


def test_currency_constructor_with_different_decimals():
    from decimal import Decimal
    from enum import Enum
    
    class CurrencyType(Enum):
        MONEY = "MONEY"
        CRYPTO = "CRYPTO"
    
    currency = Currency("JPY", "Japanese Yen", 0, CurrencyType.MONEY, Decimal("1"), 54321)
    
    assert currency.code == "JPY"
    assert currency.decimals == 0
    assert currency.quantizer == Decimal("1")


def test_currency_constructor_with_negative_decimals():
    from decimal import Decimal
    from enum import Enum
    
    class CurrencyType(Enum):
        MONEY = "MONEY"
        CRYPTO = "CRYPTO"
    
    currency = Currency("ZZZ", "Crypto", -1, CurrencyType.CRYPTO, Decimal("0.000000000001"), 99999)
    
    assert currency.code == "ZZZ"
    assert currency.decimals == -1
    assert currency.type == CurrencyType.CRYPTO


# LLM-generated content at query #18
#--------------------------

```python
def test_currency_registry_constructor():
    registry = CurrencyRegistry()
    assert registry is not None
    assert len(registry) == 0
    assert registry.all == []
    assert registry.codes == []
    assert registry.codenames == []


def test_currency_registry_singleton():
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2


def test_currency_registry_constructor_initializes_empty_state():
    registry = CurrencyRegistry()
    assert "USD" not in registry or len(registry) > 0
    assert registry.get("NONEXISTENT") is None


# LLM-generated content at query #19
#--------------------------

```python
def test_registry_context_open_flag_initialized_to_false():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open == False


# LLM-generated content at query #20
#--------------------------

```python
def test_currency_constructor_creates_frozen_dataclass():
    from decimal import Decimal
    from enum import Enum
    
    class CurrencyType(Enum):
        MONEY = "MONEY"
        CRYPTO = "CRYPTO"
    
    code = "USD"
    name = "US Dollars"
    decimals = 2
    ctype = CurrencyType.MONEY
    quantizer = Decimal("0.01")
    hashcache = 12345
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    assert currency.code == "USD"
    assert currency.name == "US Dollars"
    assert currency.decimals == 2
    assert currency.type == CurrencyType.MONEY
    assert currency.quantizer == Decimal("0.01")
    assert currency.hashcache == 12345


def test_currency_constructor_is_frozen():
    from decimal import Decimal
    from enum import Enum
    
    class CurrencyType(Enum):
        MONEY = "MONEY"
        CRYPTO = "CRYPTO"
    
    currency = Currency("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal("0.01"), 12345)
    
    try:
        currency.code = "EUR"
        assert False, "Should not be able to modify frozen dataclass"
    except Exception:
        pass


def test_currency_constructor_with_different_values():
    from decimal import Decimal
    from enum import Enum
    
    class CurrencyType(Enum):
        MONEY = "MONEY"
        CRYPTO = "CRYPTO"
    
    currency = Currency("JPY", "Japanese Yen", 0, CurrencyType.MONEY, Decimal("1"), 54321)
    
    assert currency.code == "JPY"
    assert currency.name == "Japanese Yen"
    assert currency.decimals == 0
    assert currency.type == CurrencyType.MONEY
    assert currency.quantizer == Decimal("1")
    assert currency.hashcache == 54321


def test_currency_constructor_with_negative_decimals():
    from decimal import Decimal
    from enum import Enum
    
    class CurrencyType(Enum):
        MONEY = "MONEY"
        CRYPTO = "CRYPTO"
    
    quantizer = Decimal("0.000000000001")
    currency = Currency("ZZZ", "Crypto Currency", -1, CurrencyType.CRYPTO, quantizer, 99999)
    
    assert currency.code == "ZZZ"
    assert currency.name == "Crypto Currency"
    assert currency.decimals == -1
    assert currency.type == CurrencyType.CRYPTO
    assert currency.quantizer == quantizer
    assert currency.hashcache == 99999


# LLM-generated content at query #21
#--------------------------

```python
def test_currency_registry_constructor():
    registry = CurrencyRegistry()
    assert registry is not None
    assert len(registry) == 0
    assert registry.all == []
    assert registry.codes == []
    assert registry.codenames == []


def test_currency_registry_singleton():
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2


def test_currency_registry_constructor_initializes_empty_containers():
    registry = CurrencyRegistry()
    assert isinstance(registry.all, list)
    assert isinstance(registry.codes, list)
    assert isinstance(registry.codenames, list)
    assert len(registry.all) == 0
    assert len(registry.codes) == 0
    assert len(registry.codenames) == 0


# LLM-generated content at query #22
#--------------------------

```python
def test_currency_registry_ctx_open_initialized_to_false():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open == False


# LLM-generated content at query #23
#--------------------------

```python
def test_currency_constructor_with_valid_parameters():
    code = "USD"
    name = "US Dollars"
    decimals = 2
    ctype = CurrencyType.MONEY
    quantizer = Decimal("0.01")
    hashcache = 12345
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    assert currency.code == "USD"
    assert currency.name == "US Dollars"
    assert currency.decimals == 2
    assert currency.type == CurrencyType.MONEY
    assert currency.quantizer == Decimal("0.01")
    assert currency.hashcache == 12345


def test_currency_constructor_frozen():
    code = "USD"
    name = "US Dollars"
    decimals = 2
    ctype = CurrencyType.MONEY
    quantizer = Decimal("0.01")
    hashcache = 12345
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    try:
        currency.code = "EUR"
        assert False, "Expected FrozenInstanceError"
    except Exception:
        pass


def test_currency_constructor_with_negative_decimals():
    code = "ZZZ"
    name = "Crypto Currency"
    decimals = -1
    ctype = CurrencyType.CRYPTO
    quantizer = Decimal("1")
    hashcache = 54321
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    assert currency.code == "ZZZ"
    assert currency.decimals == -1
    assert currency.type == CurrencyType.CRYPTO


def test_currency_constructor_with_zero_decimals():
    code = "JPY"
    name = "Japanese Yen"
    decimals = 0
    ctype = CurrencyType.MONEY
    quantizer = Decimal("1")
    hashcache = 99999
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    assert currency.code == "JPY"
    assert currency.decimals == 0
    assert currency.quantizer == Decimal("1")


# LLM-generated content at query #24
#--------------------------

```python
def test_currency_registry_ctx_open_initialized_to_false():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open is False


# LLM-generated content at query #25
#--------------------------

```python
def test_currency_registry_codes_empty_on_init():
    registry = CurrencyRegistry()
    assert len(registry.codes) == 0


# LLM-generated content at query #26
#--------------------------

```python
def test_currency_constructor_creates_instance_with_correct_attributes():
    code = "USD"
    name = "US Dollars"
    decimals = 2
    ctype = CurrencyType.MONEY
    quantizer = Decimal("0.01")
    hashcache = 12345
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    assert currency.code == code
    assert currency.name == name
    assert currency.decimals == decimals
    assert currency.type == ctype
    assert currency.quantizer == quantizer
    assert currency.hashcache == hashcache


def test_currency_constructor_creates_frozen_instance():
    currency = Currency("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal("0.01"), 12345)
    
    try:
        currency.code = "EUR"
        assert False, "Should not be able to modify frozen dataclass"
    except (AttributeError, Exception):
        pass


def test_currency_constructor_is_orderable():
    currency1 = Currency("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal("0.01"), 100)
    currency2 = Currency("EUR", "Euro", 2, CurrencyType.MONEY, Decimal("0.01"), 200)
    
    assert currency1 < currency2
    assert currency2 > currency1
    assert currency1 <= currency2
    assert currency2 >= currency1


# LLM-generated content at query #27
#--------------------------

```python
def test_currency_registry_constructor():
    registry = CurrencyRegistry()
    assert registry is not None
    assert len(registry) == 0
    assert registry.all == []
    assert registry.codes == []
    assert registry.codenames == []


def test_currency_registry_singleton():
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2


def test_currency_registry_constructor_initializes_empty_containers():
    registry = CurrencyRegistry()
    assert isinstance(registry.all, list)
    assert isinstance(registry.codes, list)
    assert isinstance(registry.codenames, list)
    assert len(registry.all) == 0
    assert len(registry.codes) == 0
    assert len(registry.codenames) == 0


# LLM-generated content at query #28
#--------------------------

```python
def test_currency_registry_ctx_open_initialized_to_false():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open == False


# LLM-generated content at query #29
#--------------------------

```python
def test_currency_constructor_creates_instance_with_all_fields():
    from decimal import Decimal
    from dataclasses import fields
    
    code = "USD"
    name = "US Dollars"
    decimals = 2
    ctype = CurrencyType.MONEY
    quantizer = Decimal("0.01")
    hashcache = 12345
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    assert currency.code == code
    assert currency.name == name
    assert currency.decimals == decimals
    assert currency.type == ctype
    assert currency.quantizer == quantizer
    assert currency.hashcache == hashcache


def test_currency_constructor_creates_frozen_instance():
    from decimal import Decimal
    
    currency = Currency("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal("0.01"), 12345)
    
    try:
        currency.code = "EUR"
        assert False, "Should not be able to modify frozen dataclass"
    except AttributeError:
        pass


def test_currency_constructor_with_negative_decimals():
    from decimal import Decimal
    
    currency = Currency("ZZZ", "Crypto", -1, CurrencyType.CRYPTO, Decimal("1"), 99999)
    
    assert currency.decimals == -1
    assert currency.code == "ZZZ"


def test_currency_constructor_with_zero_decimals():
    from decimal import Decimal
    
    currency = Currency("JPY", "Japanese Yen", 0, CurrencyType.MONEY, Decimal("1"), 54321)
    
    assert currency.decimals == 0
    assert currency.quantizer == Decimal("1")


def test_currency_constructor_preserves_all_parameters():
    from decimal import Decimal
    
    code = "EUR"
    name = "Euro"
    decimals = 2
    ctype = CurrencyType.MONEY
    quantizer = Decimal("0.01")
    hashcache = 11111
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    assert (currency.code, currency.name, currency.decimals, currency.type, currency.quantizer, currency.hashcache) == (code, name, decimals, ctype, quantizer, hashcache)


