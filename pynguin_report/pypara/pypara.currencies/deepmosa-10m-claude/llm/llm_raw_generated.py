####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
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


def test_currency_constructor_with_zero_decimals():
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


def test_currency_constructor_with_negative_decimals():
    code = "ZZZ"
    name = "Some weird currency"
    decimals = -1
    ctype = CurrencyType.CRYPTO
    quantizer = Decimal("1e-12")
    hashcache = hash((code, name, decimals, ctype, quantizer))
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    assert currency.code == "ZZZ"
    assert currency.decimals == -1
    assert currency.type == CurrencyType.CRYPTO


def test_currency_constructor_frozen():
    code = "USD"
    name = "US Dollars"
    decimals = 2
    ctype = CurrencyType.MONEY
    quantizer = Decimal("0.01")
    hashcache = hash((code, name, decimals, ctype, quantizer))
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    try:
        currency.code = "EUR"
        assert False, "Should not be able to modify frozen dataclass"
    except (AttributeError, Exception):
        pass


# LLM-generated content at query #2
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
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    registry3 = CurrencyRegistry()
    assert registry1 is registry2 is registry3


# LLM-generated content at query #3
#--------------------------

```python
def test_currency_constructor_creates_instance_with_all_attributes():
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


def test_currency_constructor_with_zero_decimals():
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
    assert currency.quantizer == Decimal("0.000000000001")


# LLM-generated content at query #4
#--------------------------

```python
def test_registry_context_closed_on_init():
    registry = CurrencyRegistry()
    try:
        registry._CurrencyRegistry__register(None)
    except Exception as e:
        assert isinstance(e, ProgrammingError)
        assert str(e) == "Can not create currencies outside registry context."


# LLM-generated content at query #5
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
    assert "USD" not in registry or len(registry) == 0
    assert registry.get("NONEXISTENT") is None


# LLM-generated content at query #7
#--------------------------

```python
def test_currency_registry_codes_empty_on_init():
    registry = CurrencyRegistry()
    assert len(registry.codes) == 0


# LLM-generated content at query #8
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


def test_currency_registry_constructor_initializes_empty_containers():
    registry = CurrencyRegistry()
    assert isinstance(registry.all, list)
    assert isinstance(registry.codes, list)
    assert isinstance(registry.codenames, list)
    assert len(registry.all) == 0
    assert len(registry.codes) == 0
    assert len(registry.codenames) == 0


# LLM-generated content at query #9
#--------------------------

```python
def test_currency_constructor_creates_instance_with_all_fields():
    from decimal import Decimal
    from dataclasses import fields
    
    code = "USD"
    name = "US Dollars"
    decimals = 2
    ctype_value = 1
    quantizer = Decimal("0.01")
    hashcache = 12345
    
    # Create a mock CurrencyType enum value
    class CurrencyType:
        MONEY = 1
    
    ctype = CurrencyType.MONEY
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    assert currency.code == code
    assert currency.name == name
    assert currency.decimals == decimals
    assert currency.type == ctype
    assert currency.quantizer == quantizer
    assert currency.hashcache == hashcache


def test_currency_constructor_is_frozen():
    from decimal import Decimal
    
    class CurrencyType:
        MONEY = 1
    
    currency = Currency("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal("0.01"), 12345)
    
    try:
        currency.code = "EUR"
        assert False, "Expected FrozenInstanceError"
    except Exception as e:
        assert "frozen" in str(type(e)).lower() or "immutable" in str(e).lower()


def test_currency_constructor_with_negative_decimals():
    from decimal import Decimal
    
    class CurrencyType:
        CRYPTO = 2
    
    quantizer = Decimal("1E+12")
    currency = Currency("ZZZ", "Weird Currency", -1, CurrencyType.CRYPTO, quantizer, 99999)
    
    assert currency.decimals == -1
    assert currency.quantizer == quantizer


def test_currency_constructor_with_zero_decimals():
    from decimal import Decimal
    
    class CurrencyType:
        MONEY = 1
    
    currency = Currency("JPY", "Japanese Yen", 0, CurrencyType.MONEY, Decimal("1"), 54321)
    
    assert currency.decimals == 0
    assert currency.quantizer == Decimal("1")


def test_currency_constructor_preserves_all_attributes_immutably():
    from decimal import Decimal
    
    class CurrencyType:
        MONEY = 1
    
    code = "GBP"
    name = "British Pound"
    decimals = 2
    ctype = CurrencyType.MONEY
    quantizer = Decimal("0.01")
    hashcache = 11111
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    assert currency.code == code
    assert currency.name == name
    assert currency.decimals == decimals
    assert currency.type == ctype
    assert currency.quantizer == quantizer
    assert currency.hashcache == hashcache
    
    # Verify all attributes remain unchanged when accessed multiple times
    assert currency.code == code
    assert currency.name == name


# LLM-generated content at query #10
#--------------------------

```python
def test_currency_constructor_creates_frozen_dataclass():
    from decimal import Decimal
    ccy = Currency(
        code="USD",
        name="US Dollars",
        decimals=2,
        type=CurrencyType.MONEY,
        quantizer=Decimal("0.01"),
        hashcache=12345
    )
    assert ccy.code == "USD"
    assert ccy.name == "US Dollars"
    assert ccy.decimals == 2
    assert ccy.type == CurrencyType.MONEY
    assert ccy.quantizer == Decimal("0.01")
    assert ccy.hashcache == 12345


def test_currency_constructor_is_frozen():
    from decimal import Decimal
    ccy = Currency(
        code="USD",
        name="US Dollars",
        decimals=2,
        type=CurrencyType.MONEY,
        quantizer=Decimal("0.01"),
        hashcache=12345
    )
    try:
        ccy.code = "EUR"
        assert False, "Should not be able to modify frozen dataclass"
    except (AttributeError, Exception):
        pass


def test_currency_constructor_with_different_decimals():
    from decimal import Decimal
    ccy = Currency(
        code="JPY",
        name="Japanese Yen",
        decimals=0,
        type=CurrencyType.MONEY,
        quantizer=Decimal("1"),
        hashcache=67890
    )
    assert ccy.code == "JPY"
    assert ccy.decimals == 0
    assert ccy.quantizer == Decimal("1")


def test_currency_constructor_with_negative_decimals():
    from decimal import Decimal
    ccy = Currency(
        code="ZZZ",
        name="Weird Currency",
        decimals=-1,
        type=CurrencyType.CRYPTO,
        quantizer=Decimal("0.000000000001"),
        hashcache=11111
    )
    assert ccy.code == "ZZZ"
    assert ccy.decimals == -1
    assert ccy.type == CurrencyType.CRYPTO


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
    assert "USD" not in registry or len(registry) > 0
    assert registry.has("NONEXISTENT") == False
    assert registry.get("NONEXISTENT") is None


# LLM-generated content at query #12
#--------------------------

```python
def test_currency_registry_codes_empty_on_init():
    registry = CurrencyRegistry()
    assert len(registry.codes) == 0
    assert registry.codes == []


# LLM-generated content at query #13
#--------------------------

```python
def test_currency_registry_codes_empty_on_init():
    registry = CurrencyRegistry()
    assert len(registry.codes) == 0
    assert registry.codes == []


# LLM-generated content at query #14
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


def test_currency_constructor_creates_frozen_object():
    code = "EUR"
    name = "Euro"
    decimals = 2
    ctype = CurrencyType.MONEY
    quantizer = Decimal("0.01")
    hashcache = 67890
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    try:
        currency.code = "GBP"
        assert False, "Expected FrozenInstanceError"
    except Exception:
        pass


def test_currency_constructor_with_zero_decimals():
    code = "JPY"
    name = "Japanese Yen"
    decimals = 0
    ctype = CurrencyType.MONEY
    quantizer = Decimal("1")
    hashcache = 11111
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    assert currency.decimals == 0
    assert currency.quantizer == Decimal("1")


def test_currency_constructor_with_negative_decimals():
    code = "ZZZ"
    name = "Some weird currency"
    decimals = -1
    ctype = CurrencyType.CRYPTO
    quantizer = Decimal("1E+12")
    hashcache = 22222
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    assert currency.decimals == -1
    assert currency.code == "ZZZ"


def test_currency_constructor_with_different_currency_type():
    code = "BTC"
    name = "Bitcoin"
    decimals = 8
    ctype = CurrencyType.CRYPTO
    quantizer = Decimal("0.00000001")
    hashcache = 33333
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    assert currency.type == CurrencyType.CRYPTO
    assert currency.name == "Bitcoin"


# LLM-generated content at query #15
#--------------------------

```python
def test_currency_registry_codes_empty_on_init():
    registry = CurrencyRegistry()
    assert len(registry.codes) == 0


# LLM-generated content at query #16
#--------------------------

```python
def test_registry_context_open_flag_initialized_to_false():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open is False


# LLM-generated content at query #17
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
        assert False, "Should not be able to modify frozen dataclass"
    except Exception:
        pass


def test_currency_constructor_with_crypto_type():
    code = "BTC"
    name = "Bitcoin"
    decimals = -1
    ctype = CurrencyType.CRYPTO
    quantizer = Decimal("1E+1")
    hashcache = hash((code, name, decimals, ctype, quantizer))
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    assert currency.code == "BTC"
    assert currency.name == "Bitcoin"
    assert currency.decimals == -1
    assert currency.type == CurrencyType.CRYPTO
    assert currency.quantizer == Decimal("1E+1")


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


# LLM-generated content at query #18
#--------------------------

```python
def test_currency_registry_ctx_open_initialized_to_false():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open == False


# LLM-generated content at query #19
#--------------------------

```python
def test_currency_registry_codes_empty_on_init():
    registry = CurrencyRegistry()
    assert len(registry.codes) == 0


# LLM-generated content at query #20
#--------------------------

```python
def test_currency_constructor_creates_instance_with_all_fields():
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


def test_currency_constructor_is_frozen():
    currency = Currency("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal("0.01"), 12345)
    
    try:
        currency.code = "EUR"
        assert False, "Expected FrozenInstanceError"
    except Exception:
        pass


def test_currency_constructor_is_orderable():
    currency1 = Currency("EUR", "Euro", 2, CurrencyType.MONEY, Decimal("0.01"), 111)
    currency2 = Currency("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal("0.01"), 222)
    
    result = currency1 < currency2
    assert isinstance(result, bool)


def test_currency_constructor_with_negative_decimals():
    code = "ZZZ"
    name = "Crypto"
    decimals = -1
    ctype = CurrencyType.CRYPTO
    quantizer = Decimal("1")
    hashcache = 99999
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    assert currency.decimals == -1
    assert currency.code == code


def test_currency_constructor_with_zero_decimals():
    code = "JPY"
    name = "Japanese Yen"
    decimals = 0
    ctype = CurrencyType.MONEY
    quantizer = Decimal("1")
    hashcache = 55555
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    assert currency.decimals == 0
    assert currency.quantizer == quantizer


# LLM-generated content at query #21
#--------------------------

```python
def test_registry_context_open_flag_initialized_to_false():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open == False


# LLM-generated content at query #22
#--------------------------

```python
def test_currency_registry_codes_empty_on_init():
    registry = CurrencyRegistry()
    assert len(registry.codes) == 0


# LLM-generated content at query #23
#--------------------------

```python
def test_currency_constructor_valid():
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
        assert False, "Should not be able to modify frozen dataclass"
    except Exception:
        pass


def test_currency_constructor_with_negative_decimals():
    from decimal import Decimal
    from enum import Enum
    
    class CurrencyType(Enum):
        MONEY = "MONEY"
        CRYPTO = "CRYPTO"
    
    code = "ZZZ"
    name = "Weird Currency"
    decimals = -1
    ctype = CurrencyType.CRYPTO
    quantizer = Decimal("1")
    hashcache = hash((code, name, decimals, ctype, quantizer))
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    assert currency.decimals == -1
    assert currency.code == "ZZZ"


def test_currency_constructor_zero_decimals():
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
    
    assert currency.decimals == 0
    assert currency.quantizer == Decimal("1")


# LLM-generated content at query #24
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
    assert "USD" not in registry or len(registry) > 0


# LLM-generated content at query #25
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


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_currency_eq_same_currencies():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2


def test_currency_eq_different_names():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert usd1 != usd2


def test_currency_eq_different_codes():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "US Dollars", 2, CurrencyType.MONEY)
    assert usd != eur


def test_currency_eq_different_decimals():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert usd1 != usd2


def test_currency_eq_different_types():
    usd_money = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd_crypto = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert usd_money != usd_crypto


def test_currency_eq_with_non_currency_object():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd != "USD"
    assert usd != 2
    assert usd != None


def test_currency_eq_reflexive():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd == usd


def test_currency_eq_with_negative_decimals():
    zzz1 = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    zzz2 = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert zzz1 == zzz2


def test_currency_eq_zero_decimals():
    jpy1 = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    jpy2 = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy1 == jpy2


# LLM-generated content at query #2
#--------------------------

```python
def test_getitem_returns_currency_by_code():
    registry = CurrencyRegistry()
    currency = registry["USD"]
    assert currency.code == "USD"
    assert currency.name == "US Dollar"


def test_getitem_raises_currency_lookup_error_for_invalid_code():
    registry = CurrencyRegistry()
    try:
        registry["NON-EXISTING"]
        assert False, "Expected CurrencyLookupError to be raised"
    except CurrencyLookupError:
        pass


def test_getitem_with_existing_currency():
    registry = CurrencyRegistry()
    usd = registry["USD"]
    eur = registry["EUR"]
    assert usd.code == "USD"
    assert eur.code == "EUR"
    assert usd != eur


def test_getitem_returns_same_instance():
    registry = CurrencyRegistry()
    currency1 = registry["USD"]
    currency2 = registry["USD"]
    assert currency1 == currency2


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
    assert "USD" not in registry or len(registry) > 0
    assert len(registry.all) == len(registry)
    assert len(registry.codes) == len(registry)
    assert len(registry.codenames) == len(registry)


# LLM-generated content at query #4
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
    code = "USD"
    name = "US Dollars"
    decimals = 2
    ctype = CurrencyType.MONEY
    quantizer = Decimal("0.01")
    hashcache = hash((code, name, decimals, ctype, quantizer))
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    try:
        currency.code = "EUR"
        assert False, "Should not be able to modify frozen dataclass"
    except Exception:
        pass


def test_currency_constructor_with_zero_decimals():
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


def test_currency_constructor_with_negative_decimals():
    code = "ZZZ"
    name = "Some weird currency"
    decimals = -1
    ctype = CurrencyType.CRYPTO
    quantizer = Decimal("1e-12")
    hashcache = hash((code, name, decimals, ctype, quantizer))
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    assert currency.code == "ZZZ"
    assert currency.decimals == -1
    assert currency.type == CurrencyType.CRYPTO


def test_currency_constructor_hashcache():
    code = "EUR"
    name = "Euro"
    decimals = 2
    ctype = CurrencyType.MONEY
    quantizer = Decimal("0.01")
    hashcache = 12345
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    assert currency.hashcache == 12345


# LLM-generated content at query #5
#--------------------------

```python
def test_currency_registry_ctx_open_initialized_to_false():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open is False


# LLM-generated content at query #6
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


def test_currency_constructor_with_negative_decimals():
    from decimal import Decimal
    
    code = "ZZZ"
    name = "Crypto"
    decimals = -1
    quantizer = Decimal("1E+1")
    hashcache = 54321
    
    currency = Currency(code, name, decimals, 1, quantizer, hashcache)
    
    assert currency.decimals == -1
    assert currency.quantizer == quantizer


def test_currency_constructor_with_zero_decimals():
    from decimal import Decimal
    
    code = "JPY"
    name = "Japanese Yen"
    decimals = 0
    quantizer = Decimal("1")
    hashcache = 99999
    
    currency = Currency(code, name, decimals, 0, quantizer, hashcache)
    
    assert currency.decimals == 0
    assert currency.quantizer == quantizer


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
    except Exception:
        pass


def test_currency_constructor_with_negative_decimals():
    code = "ZZZ"
    name = "Weird Currency"
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


# LLM-generated content at query #8
#--------------------------

```python
def test_currency_registry_codes_empty_on_init():
    registry = CurrencyRegistry()
    assert len(registry.codes) == 0
    assert registry.codes == []


# LLM-generated content at query #9
#--------------------------

```python
def test_currency_constructor_creates_instance_with_all_fields():
    from decimal import Decimal
    
    code = "USD"
    name = "US Dollars"
    decimals = 2
    ctype_value = "MONEY"
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
    
    currency = Currency("USD", "US Dollars", 2, "MONEY", Decimal("0.01"), 12345)
    
    try:
        currency.code = "EUR"
        assert False, "Should not be able to modify frozen dataclass"
    except AttributeError:
        pass


def test_currency_constructor_with_zero_decimals():
    from decimal import Decimal
    
    currency = Currency("JPY", "Japanese Yen", 0, "MONEY", Decimal("1"), 54321)
    
    assert currency.code == "JPY"
    assert currency.decimals == 0
    assert currency.quantizer == Decimal("1")


def test_currency_constructor_with_negative_decimals():
    from decimal import Decimal
    
    currency = Currency("ZZZ", "Crypto", -1, "CRYPTO", Decimal("0.000000000001"), 99999)
    
    assert currency.code == "ZZZ"
    assert currency.decimals == -1
    assert currency.hashcache == 99999


def test_currency_constructor_ordering():
    from decimal import Decimal
    
    currency1 = Currency("AAA", "First", 2, "MONEY", Decimal("0.01"), 111)
    currency2 = Currency("BBB", "Second", 2, "MONEY", Decimal("0.01"), 222)
    
    assert currency1 < currency2
    assert currency2 > currency1
    assert not (currency1 > currency2)
    assert not (currency2 < currency1)


# LLM-generated content at query #10
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


# LLM-generated content at query #11
#--------------------------

```python
def test_currency_registry_codes_empty_on_init():
    registry = CurrencyRegistry()
    assert len(registry.codes) == 0


# LLM-generated content at query #12
#--------------------------

```python
def test_currency_registry_codes_empty_on_init():
    registry = CurrencyRegistry()
    assert len(registry.codes) == 0
    assert registry.codes == []


# LLM-generated content at query #13
#--------------------------

```python
def test_currency_registry_ctx_open_initialized_to_false():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open == False


# LLM-generated content at query #14
#--------------------------

```python
def test_currency_registry_codes_empty_on_init():
    registry = CurrencyRegistry()
    assert len(registry.codes) == 0
    assert registry.codes == []


# LLM-generated content at query #15
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


# LLM-generated content at query #16
#--------------------------

```python
def test_currency_constructor_creates_frozen_instance():
    from decimal import Decimal
    ccy = Currency(
        code="USD",
        name="US Dollars",
        decimals=2,
        type=CurrencyType.MONEY,
        quantizer=Decimal("0.01"),
        hashcache=12345
    )
    assert ccy.code == "USD"
    assert ccy.name == "US Dollars"
    assert ccy.decimals == 2
    assert ccy.type == CurrencyType.MONEY
    assert ccy.quantizer == Decimal("0.01")
    assert ccy.hashcache == 12345


def test_currency_constructor_is_frozen():
    from decimal import Decimal
    ccy = Currency(
        code="USD",
        name="US Dollars",
        decimals=2,
        type=CurrencyType.MONEY,
        quantizer=Decimal("0.01"),
        hashcache=12345
    )
    try:
        ccy.code = "EUR"
        assert False, "Expected FrozenInstanceError"
    except Exception:
        pass


def test_currency_constructor_with_all_attributes():
    from decimal import Decimal
    ccy = Currency(
        code="JPY",
        name="Japanese Yen",
        decimals=0,
        type=CurrencyType.MONEY,
        quantizer=Decimal("1"),
        hashcache=54321
    )
    assert ccy.code == "JPY"
    assert ccy.name == "Japanese Yen"
    assert ccy.decimals == 0
    assert ccy.type == CurrencyType.MONEY
    assert ccy.quantizer == Decimal("1")
    assert ccy.hashcache == 54321


def test_currency_constructor_with_negative_decimals():
    from decimal import Decimal
    ccy = Currency(
        code="ZZZ",
        name="Crypto",
        decimals=-1,
        type=CurrencyType.CRYPTO,
        quantizer=Decimal("0.000000000001"),
        hashcache=99999
    )
    assert ccy.decimals == -1
    assert ccy.type == CurrencyType.CRYPTO


# LLM-generated content at query #17
#--------------------------

```python
def test_currency_registry_codes_empty_on_init():
    registry = CurrencyRegistry()
    assert len(registry.codes) == 0
    assert registry.codes == []


# LLM-generated content at query #18
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
    assert currency.type == CurrencyType.MONEY
    assert currency.quantizer == Decimal("0.01")
    assert currency.hashcache == 12345


def test_currency_constructor_with_negative_decimals():
    from decimal import Decimal
    from enum import Enum
    
    class CurrencyType(Enum):
        MONEY = "MONEY"
        CRYPTO = "CRYPTO"
    
    code = "ZZZ"
    name = "Some weird currency"
    decimals = -1
    ctype = CurrencyType.CRYPTO
    quantizer = Decimal("1")
    hashcache = 54321
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    assert currency.code == "ZZZ"
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
    hashcache = 99999
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    assert currency.code == "JPY"
    assert currency.decimals == 0
    assert currency.quantizer == Decimal("1")


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
    except (AttributeError, TypeError):
        pass


def test_currency_constructor_supports_ordering():
    from decimal import Decimal
    from enum import Enum
    
    class CurrencyType(Enum):
        MONEY = "MONEY"
        CRYPTO = "CRYPTO"
    
    currency1 = Currency("AAA", "Currency A", 2, CurrencyType.MONEY, Decimal("0.01"), 100)
    currency2 = Currency("BBB", "Currency B", 2, CurrencyType.MONEY, Decimal("0.01"), 200)
    
    assert currency1 < currency2
    assert currency2 > currency1
    assert currency1 <= currency2
    assert currency2 >= currency1


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


def test_currency_registry_constructor_initializes_empty_state():
    registry = CurrencyRegistry()
    assert "USD" not in registry or len(registry) == 0
    assert registry.get("NONEXISTENT") is None


# LLM-generated content at query #20
#--------------------------

```python
def test_currency_registry_codes_empty_on_init():
    registry = CurrencyRegistry()
    assert len(registry.codes) == 0


# LLM-generated content at query #21
#--------------------------

```python
def test_ctx_open_initialized_to_false():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open == False


# LLM-generated content at query #22
#--------------------------

```python
def test_currency_registry_codes_empty_on_init():
    registry = CurrencyRegistry()
    assert len(registry.codes) == 0
    assert registry.codes == []


# LLM-generated content at query #23
#--------------------------

```python
def test_currency_constructor_creates_instance_with_all_fields():
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
    code = "USD"
    name = "US Dollars"
    decimals = 2
    ctype = CurrencyType.MONEY
    quantizer = Decimal("0.01")
    hashcode = 12345
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcode)
    
    try:
        currency.code = "EUR"
        assert False, "Expected FrozenInstanceError"
    except Exception:
        pass


def test_currency_constructor_with_different_decimals():
    code = "JPY"
    name = "Japanese Yen"
    decimals = 0
    ctype = CurrencyType.MONEY
    quantizer = Decimal("1")
    hashcode = 54321
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcode)
    
    assert currency.code == "JPY"
    assert currency.decimals == 0
    assert currency.quantizer == Decimal("1")


def test_currency_constructor_with_negative_decimals():
    code = "ZZZ"
    name = "Weird Currency"
    decimals = -1
    ctype = CurrencyType.CRYPTO
    quantizer = Decimal("0.000000000001")
    hashcode = 99999
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcode)
    
    assert currency.code == "ZZZ"
    assert currency.decimals == -1
    assert currency.type == CurrencyType.CRYPTO


# LLM-generated content at query #24
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


# LLM-generated content at query #25
#--------------------------

```python
def test_currency_constructor_creates_instance_with_all_fields():
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


def test_currency_constructor_is_frozen():
    currency = Currency("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal("0.01"), 12345)
    
    try:
        currency.code = "EUR"
        assert False, "Should not be able to modify frozen dataclass"
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
    
    assert currency.code == code
    assert currency.decimals == decimals
    assert currency.type == ctype


def test_currency_constructor_with_zero_decimals():
    code = "JPY"
    name = "Japanese Yen"
    decimals = 0
    ctype = CurrencyType.MONEY
    quantizer = Decimal("1")
    hashcache = 99999
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    assert currency.code == code
    assert currency.decimals == decimals
    assert currency.quantizer == quantizer


# LLM-generated content at query #26
#--------------------------

```python
def test_ctx_open_initialized_to_false():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open is False


