####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
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


def test_currency_registry_constructor_initializes_empty_containers():
    registry = CurrencyRegistry()
    assert isinstance(registry.all, list)
    assert isinstance(registry.codes, list)
    assert isinstance(registry.codenames, list)
    assert len(registry.all) == 0
    assert len(registry.codes) == 0
    assert len(registry.codenames) == 0


# LLM-generated content at query #3
#--------------------------

```python
def test_currency_registry_codes_empty_on_init():
    registry = CurrencyRegistry()
    assert len(registry.codes) == 0


# LLM-generated content at query #4
#--------------------------

```python
def test_currency_registry_codes_empty_on_init():
    registry = CurrencyRegistry()
    assert len(registry.codes) == 0
    assert registry.codes == []


# LLM-generated content at query #5
#--------------------------

```python
def test_currency_registry_ctx_open_initialized_to_false():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open is False


# LLM-generated content at query #6
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


# LLM-generated content at query #7
#--------------------------

```python
def test_currency_constructor():
    from decimal import Decimal
    from dataclasses import fields
    
    code = "USD"
    name = "US Dollars"
    decimals = 2
    ctype_value = "MONEY"
    quantizer = Decimal("0.01")
    hashcache = 12345
    
    currency = Currency(code, name, decimals, ctype_value, quantizer, hashcache)
    
    assert currency.code == "USD"
    assert currency.name == "US Dollars"
    assert currency.decimals == 2
    assert currency.type == "MONEY"
    assert currency.quantizer == Decimal("0.01")
    assert currency.hashcache == 12345


def test_currency_constructor_with_negative_decimals():
    from decimal import Decimal
    
    code = "ZZZ"
    name = "Some weird currency"
    decimals = -1
    ctype_value = "CRYPTO"
    quantizer = Decimal("1E+1")
    hashcache = 54321
    
    currency = Currency(code, name, decimals, ctype_value, quantizer, hashcache)
    
    assert currency.code == "ZZZ"
    assert currency.name == "Some weird currency"
    assert currency.decimals == -1
    assert currency.type == "CRYPTO"
    assert currency.quantizer == Decimal("1E+1")
    assert currency.hashcache == 54321


def test_currency_constructor_with_zero_decimals():
    from decimal import Decimal
    
    code = "JPY"
    name = "Japanese Yen"
    decimals = 0
    ctype_value = "MONEY"
    quantizer = Decimal("1")
    hashcache = 99999
    
    currency = Currency(code, name, decimals, ctype_value, quantizer, hashcache)
    
    assert currency.code == "JPY"
    assert currency.name == "Japanese Yen"
    assert currency.decimals == 0
    assert currency.type == "MONEY"
    assert currency.quantizer == Decimal("1")
    assert currency.hashcache == 99999


def test_currency_constructor_frozen():
    from decimal import Decimal
    
    currency = Currency("USD", "US Dollars", 2, "MONEY", Decimal("0.01"), 12345)
    
    try:
        currency.code = "EUR"
        assert False, "Should not be able to modify frozen dataclass"
    except (AttributeError, Exception):
        pass


# LLM-generated content at query #8
#--------------------------

```python
def test_currency_registry_codes_empty_on_init():
    registry = CurrencyRegistry()
    assert len(registry.codes) == 0


# LLM-generated content at query #9
#--------------------------

```python
def test_ctx_open_initialized_to_false():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open is False


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


def test_currency_registry_constructor_initializes_empty_containers():
    registry = CurrencyRegistry()
    assert isinstance(registry.all, list)
    assert isinstance(registry.codes, list)
    assert isinstance(registry.codenames, list)
    assert len(registry.all) == 0
    assert len(registry.codes) == 0
    assert len(registry.codenames) == 0


# LLM-generated content at query #11
#--------------------------

```python
def test_registry_context_open_flag_initialized_to_false():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open == False


# LLM-generated content at query #12
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
    hashcache = 12345
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    try:
        currency.code = "EUR"
        assert False, "Should not be able to modify frozen dataclass"
    except AttributeError:
        pass


def test_currency_constructor_with_different_decimals():
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
    name = "Weird Currency"
    decimals = -1
    ctype = CurrencyType.CRYPTO
    quantizer = Decimal("0.000000000001")
    hashcache = 99999
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    assert currency.code == "ZZZ"
    assert currency.decimals == -1
    assert currency.type == CurrencyType.CRYPTO


# LLM-generated content at query #13
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
    except AttributeError:
        pass


def test_currency_constructor_is_orderable():
    currency1 = Currency("EUR", "Euro", 2, CurrencyType.MONEY, Decimal("0.01"), 111)
    currency2 = Currency("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal("0.01"), 222)
    
    assert currency1 < currency2


def test_currency_constructor_with_negative_decimals():
    code = "ZZZ"
    name = "Weird Currency"
    decimals = -1
    ctype = CurrencyType.CRYPTO
    quantizer = Decimal("1E-12")
    hashcache = 54321
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    assert currency.decimals == -1
    assert currency.quantizer == quantizer


def test_currency_constructor_with_zero_decimals():
    code = "JPY"
    name = "Japanese Yen"
    decimals = 0
    ctype = CurrencyType.MONEY
    quantizer = Decimal("1")
    hashcache = 99999
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    assert currency.decimals == 0
    assert currency.quantizer == quantizer


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


# LLM-generated content at query #16
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
        assert False, "Expected FrozenInstanceError"
    except:
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
    assert currency.quantizer == Decimal("1")
    assert currency.hashcache == 54321


def test_currency_constructor_with_negative_decimals():
    from decimal import Decimal
    from enum import Enum
    
    class CurrencyType(Enum):
        MONEY = "MONEY"
        CRYPTO = "CRYPTO"
    
    max_precision_quantizer = Decimal("0.000000000001")
    currency = Currency("ZZZ", "Crypto", -1, CurrencyType.CRYPTO, max_precision_quantizer, 99999)
    
    assert currency.code == "ZZZ"
    assert currency.decimals == -1
    assert currency.quantizer == max_precision_quantizer


# LLM-generated content at query #17
#--------------------------

```python
def test_currency_constructor_creates_instance_with_all_attributes():
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


def test_currency_constructor_creates_frozen_instance():
    code = "EUR"
    name = "Euro"
    decimals = 2
    ctype = CurrencyType.MONEY
    quantizer = Decimal("0.01")
    hashcache = 54321
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    try:
        currency.code = "GBP"
        assert False, "Should not be able to modify frozen dataclass"
    except AttributeError:
        pass


def test_currency_constructor_is_orderable():
    currency1 = Currency("AAA", "Currency A", 2, CurrencyType.MONEY, Decimal("0.01"), 100)
    currency2 = Currency("BBB", "Currency B", 2, CurrencyType.MONEY, Decimal("0.01"), 200)
    
    assert currency1 < currency2
    assert currency2 > currency1
    assert currency1 <= currency2
    assert currency2 >= currency1


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


def test_currency_registry_multiple_instantiation():
    first = CurrencyRegistry()
    second = CurrencyRegistry()
    third = CurrencyRegistry()
    assert first is second is third


# LLM-generated content at query #19
#--------------------------

```python
def test_currency_registry_codes_empty_on_init():
    registry = CurrencyRegistry()
    assert len(registry.codes) == 0
    assert registry.codes == []


# LLM-generated content at query #20
#--------------------------

```python
def test_currency_registry_codes_empty_on_init():
    registry = CurrencyRegistry()
    assert len(registry.codes) == 0


# LLM-generated content at query #21
#--------------------------

```python
def test_currency_of_valid_usd():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.code == "USD"
    assert usd.name == "US Dollars"
    assert usd.decimals == 2
    assert usd.type == CurrencyType.MONEY

def test_currency_of_valid_jpy():
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy.code == "JPY"
    assert jpy.name == "Japanese Yen"
    assert jpy.decimals == 0
    assert jpy.type == CurrencyType.MONEY

def test_currency_of_valid_crypto():
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert zzz.code == "ZZZ"
    assert zzz.name == "Some weird currency"
    assert zzz.decimals == -1
    assert zzz.type == CurrencyType.CRYPTO

def test_currency_of_invalid_code_not_string():
    try:
        Currency.of(123, "US Dollars", 2, CurrencyType.MONEY)
        assert False, "Should raise ProgrammingError"
    except ProgrammingError:
        pass

def test_currency_of_invalid_code_not_alpha():
    try:
        Currency.of("US1", "US Dollars", 2, CurrencyType.MONEY)
        assert False, "Should raise ProgrammingError"
    except ProgrammingError:
        pass

def test_currency_of_invalid_code_not_uppercase():
    try:
        Currency.of("usd", "US Dollars", 2, CurrencyType.MONEY)
        assert False, "Should raise ProgrammingError"
    except ProgrammingError:
        pass

def test_currency_of_invalid_name_not_string():
    try:
        Currency.of("USD", 123, 2, CurrencyType.MONEY)
        assert False, "Should raise ProgrammingError"
    except ProgrammingError:
        pass

def test_currency_of_invalid_name_empty():
    try:
        Currency.of("USD", "", 2, CurrencyType.MONEY)
        assert False, "Should raise ProgrammingError"
    except ProgrammingError:
        pass

def test_currency_of_invalid_name_leading_space():
    try:
        Currency.of("USD", " US Dollars", 2, CurrencyType.MONEY)
        assert False, "Should raise ProgrammingError"
    except ProgrammingError:
        pass

def test_currency_of_invalid_name_trailing_space():
    try:
        Currency.of("USD", "US Dollars ", 2, CurrencyType.MONEY)
        assert False, "Should raise ProgrammingError"
    except ProgrammingError:
        pass

def test_currency_of_invalid_decimals_not_int():
    try:
        Currency.of("USD", "US Dollars", 2.5, CurrencyType.MONEY)
        assert False, "Should raise ProgrammingError"
    except ProgrammingError:
        pass

def test_currency_of_invalid_decimals_less_than_minus_one():
    try:
        Currency.of("USD", "US Dollars", -2, CurrencyType.MONEY)
        assert False, "Should raise ProgrammingError"
    except ProgrammingError:
        pass

def test_currency_of_invalid_type_not_currency_type():
    try:
        Currency.of("USD", "US Dollars", 2, "MONEY")
        assert False, "Should raise ProgrammingError"
    except ProgrammingError:
        pass

def test_currency_equality_same_values():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2

def test_currency_equality_different_name():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert usd1 != usdx

def test_currency_hash_same_values():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert hash(usd1) == hash(usd2)

def test_currency_hash_different_name():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert hash(usd1) != hash(usdx)

def test_currency_quantize_usd():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.quantize(Decimal("1.005")) == Decimal("1.00")
    assert usd.quantize(Decimal("1.015")) == Decimal("1.02")

def test_currency_quantize_jpy():
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy.quantize(Decimal("0.5")) == Decimal("0")
    assert jpy.quantize(Decimal("1.5")) == Decimal("2")

def test_currency_quantize_crypto():
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert zzz.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert zzz.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")


# LLM-generated content at query #22
#--------------------------

```python
def test_registry_context_open_flag_initialized_to_false():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open is False


# LLM-generated content at query #23
#--------------------------

```python
def test_currency_registry_codes_empty_on_init():
    registry = CurrencyRegistry()
    assert len(registry.codes) == 0


# LLM-generated content at query #24
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
    hashcache = 54321
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    try:
        currency.code = "GBP"
        assert False, "Should not be able to modify frozen dataclass"
    except (AttributeError, Exception):
        pass


def test_currency_constructor_with_different_decimals():
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


def test_currency_constructor_with_crypto_type():
    code = "BTC"
    name = "Bitcoin"
    decimals = 8
    ctype = CurrencyType.CRYPTO
    quantizer = Decimal("0.00000001")
    hashcache = 77777
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    assert currency.type == CurrencyType.CRYPTO
    assert currency.name == "Bitcoin"


def test_currency_constructor_with_negative_decimals():
    code = "ZZZ"
    name = "Weird Currency"
    decimals = -1
    ctype = CurrencyType.CRYPTO
    quantizer = Decimal("0.000000000001")
    hashcache = 55555
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    assert currency.decimals == -1
    assert currency.quantizer == Decimal("0.000000000001")


# LLM-generated content at query #25
#--------------------------

```python
def test_ctx_open_initialized_to_false():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open == False


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
    assert not (usd1 == usd2)


def test_currency_eq_different_decimals():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 3, CurrencyType.MONEY)
    assert not (usd1 == usd2)


def test_currency_eq_different_types():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    zz = Currency.of("ZZZ", "Weird Currency", 2, CurrencyType.CRYPTO)
    assert not (usd == zz)


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


# LLM-generated content at query #2
#--------------------------

```python
def test_getitem_returns_currency_by_code():
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
    assert usd != eur


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
    assert len(registry) == 0
    assert isinstance(registry.all, list)
    assert isinstance(registry.codes, list)
    assert isinstance(registry.codenames, list)


# LLM-generated content at query #4
#--------------------------

```python
def test_currency_equality_with_same_currencies():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    result = usd1 == usd2
    assert result is True


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


def test_currency_registry_multiple_instantiations():
    first_instance = CurrencyRegistry()
    second_instance = CurrencyRegistry()
    third_instance = CurrencyRegistry()
    assert first_instance is second_instance
    assert second_instance is third_instance
    assert first_instance is third_instance


# LLM-generated content at query #6
#--------------------------

```python
def test_currency_registry_ctx_open_initialized_to_false():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open is False


# LLM-generated content at query #7
#--------------------------

```python
def test_currency_constructor_creates_instance_with_all_attributes():
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
    except (AttributeError, Exception):
        pass


def test_currency_constructor_maintains_order():
    currency1 = Currency("AUD", "Australian Dollar", 2, CurrencyType.MONEY, Decimal("0.01"), 100)
    currency2 = Currency("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal("0.01"), 200)
    
    assert currency1 < currency2


def test_currency_constructor_with_negative_decimals():
    code = "ZZZ"
    name = "Weird Currency"
    decimals = -1
    ctype = CurrencyType.CRYPTO
    quantizer = Decimal("1E+1")
    hashcache = 54321
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    assert currency.code == code
    assert currency.decimals == decimals


def test_currency_constructor_with_zero_decimals():
    code = "JPY"
    name = "Japanese Yen"
    decimals = 0
    ctype = CurrencyType.MONEY
    quantizer = Decimal("1")
    hashcache = 99999
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    assert currency.decimals == decimals
    assert currency.quantizer == quantizer


# LLM-generated content at query #8
#--------------------------

```python
def test_currency_registry_ctx_open_initialized_to_false():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open == False


# LLM-generated content at query #9
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


def test_currency_is_frozen():
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


def test_currency_registry_constructor_initializes_empty_registry():
    registry = CurrencyRegistry()
    assert isinstance(registry.all, list)
    assert isinstance(registry.codes, list)
    assert isinstance(registry.codenames, list)


# LLM-generated content at query #11
#--------------------------

```python
def test_currency_constructor_creates_instance_with_all_attributes():
    code = "USD"
    name = "US Dollars"
    decimals = 2
    ctype = CurrencyType.MONEY
    quantizer = Decimal("0.01")
    hashcode = 12345
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcode)
    
    assert currency.code == code
    assert currency.name == name
    assert currency.decimals == decimals
    assert currency.type == ctype
    assert currency.quantizer == quantizer
    assert currency.hashcache == hashcode


def test_currency_constructor_is_frozen():
    currency = Currency("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal("0.01"), 12345)
    
    try:
        currency.code = "EUR"
        assert False, "Should not be able to modify frozen dataclass"
    except Exception:
        pass


def test_currency_constructor_with_zero_decimals():
    currency = Currency("JPY", "Japanese Yen", 0, CurrencyType.MONEY, Decimal("1"), 54321)
    
    assert currency.code == "JPY"
    assert currency.decimals == 0
    assert currency.quantizer == Decimal("1")


def test_currency_constructor_with_negative_decimals():
    currency = Currency("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO, Decimal("0.000000000001"), 99999)
    
    assert currency.code == "ZZZ"
    assert currency.decimals == -1
    assert currency.type == CurrencyType.CRYPTO


def test_currency_constructor_ordering():
    currency1 = Currency("AAA", "Currency A", 2, CurrencyType.MONEY, Decimal("0.01"), 1000)
    currency2 = Currency("ZZZ", "Currency Z", 2, CurrencyType.MONEY, Decimal("0.01"), 2000)
    
    assert currency1 < currency2


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
    code = "EUR"
    name = "Euro"
    decimals = 2
    ctype = CurrencyType.MONEY
    quantizer = Decimal("0.01")
    hashcache = 54321
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    try:
        currency.code = "GBP"
        assert False, "Should not be able to modify frozen dataclass"
    except Exception:
        pass


def test_currency_constructor_with_negative_decimals():
    from decimal import Decimal
    code = "ZZZ"
    name = "Weird Currency"
    decimals = -1
    ctype = CurrencyType.CRYPTO
    quantizer = Decimal("1")
    hashcache = 99999
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    assert currency.code == "ZZZ"
    assert currency.decimals == -1
    assert currency.type == CurrencyType.CRYPTO


def test_currency_constructor_with_zero_decimals():
    from decimal import Decimal
    code = "JPY"
    name = "Japanese Yen"
    decimals = 0
    ctype = CurrencyType.MONEY
    quantizer = Decimal("1")
    hashcache = 11111
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    assert currency.code == "JPY"
    assert currency.decimals == 0
    assert currency.quantizer == Decimal("1")


def test_currency_constructor_all_attributes_assigned():
    from decimal import Decimal
    code = "GBP"
    name = "British Pound"
    decimals = 2
    ctype = CurrencyType.MONEY
    quantizer = Decimal("0.01")
    hashcache = 77777
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    assert hasattr(currency, "code")
    assert hasattr(currency, "name")
    assert hasattr(currency, "decimals")
    assert hasattr(currency, "type")
    assert hasattr(currency, "quantizer")
    assert hasattr(currency, "hashcache")


# LLM-generated content at query #14
#--------------------------

```python
def test_registry_context_open_flag_initialized_to_false():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open == False


# LLM-generated content at query #15
#--------------------------

```python
def test_currency_registry_ctx_open_initialized_to_false():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open is False


# LLM-generated content at query #16
#--------------------------

```python
def test_currency_registry_codes_empty_on_init():
    registry = CurrencyRegistry()
    assert len(registry.codes) == 0


# LLM-generated content at query #17
#--------------------------

```python
def test_currency_constructor_creates_instance_with_all_attributes():
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


def test_currency_constructor_creates_frozen_instance():
    code = "EUR"
    name = "Euro"
    decimals = 2
    ctype = CurrencyType.MONEY
    quantizer = Decimal("0.01")
    hashcode = 54321
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcode)
    
    try:
        currency.code = "GBP"
        assert False, "Should not be able to modify frozen dataclass"
    except AttributeError:
        pass


def test_currency_constructor_instances_are_orderable():
    currency1 = Currency("AAA", "Currency A", 2, CurrencyType.MONEY, Decimal("0.01"), 111)
    currency2 = Currency("BBB", "Currency B", 2, CurrencyType.MONEY, Decimal("0.01"), 222)
    
    assert currency1 < currency2
    assert currency2 > currency1
    assert currency1 <= currency2
    assert currency2 >= currency1


def test_currency_constructor_with_negative_decimals():
    code = "ZZZ"
    name = "Weird Currency"
    decimals = -1
    ctype = CurrencyType.CRYPTO
    quantizer = Decimal("1")
    hashcode = 99999
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcode)
    
    assert currency.decimals == -1
    assert currency.quantizer == Decimal("1")


def test_currency_constructor_with_zero_decimals():
    code = "JPY"
    name = "Japanese Yen"
    decimals = 0
    ctype = CurrencyType.MONEY
    quantizer = Decimal("1")
    hashcode = 77777
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcode)
    
    assert currency.decimals == 0
    assert currency.quantizer == Decimal("1")


# LLM-generated content at query #18
#--------------------------

```python
def test_currency_registry_codes_empty_on_init():
    registry = CurrencyRegistry()
    assert len(registry.codes) == 0
    assert registry.codes == []


# LLM-generated content at query #19
#--------------------------

```python
def test_currency_registry_ctx_open_initialized_to_false():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open is False


# LLM-generated content at query #20
#--------------------------

```python
def test_currency_constructor_creates_instance_with_all_attributes():
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
    name = "Some weird currency"
    decimals = -1
    ctype = CurrencyType.CRYPTO
    quantizer = Decimal("1E+1")
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


def test_currency_constructor_preserves_hashcache():
    code = "EUR"
    name = "Euro"
    decimals = 2
    ctype = CurrencyType.MONEY
    quantizer = Decimal("0.01")
    hashcache = 777
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    assert currency.hashcache == 777
    assert hash(currency) == 777


# LLM-generated content at query #21
#--------------------------

```python
def test_currency_registry_codes_empty_on_init():
    registry = CurrencyRegistry()
    assert len(registry.codes) == 0


# LLM-generated content at query #22
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


# LLM-generated content at query #23
#--------------------------

```python
def test_currency_constructor_creates_instance_with_all_attributes():
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


def test_currency_constructor_with_different_decimals():
    currency = Currency("JPY", "Japanese Yen", 0, CurrencyType.MONEY, Decimal("1"), 54321)
    
    assert currency.code == "JPY"
    assert currency.decimals == 0
    assert currency.quantizer == Decimal("1")


def test_currency_constructor_with_negative_decimals():
    currency = Currency("ZZZ", "Crypto", -1, CurrencyType.CRYPTO, Decimal("0.000000000001"), 99999)
    
    assert currency.code == "ZZZ"
    assert currency.decimals == -1
    assert currency.type == CurrencyType.CRYPTO


# LLM-generated content at query #24
#--------------------------

```python
def test_registry_context_open_flag_initialized_to_false():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open == False


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


# LLM-generated content at query #26
#--------------------------

```python
def test_currency_registry_ctx_open_initialized_to_false():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open == False


# LLM-generated content at query #27
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
    hashcache = 12345
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    try:
        currency.code = "EUR"
        assert False, "Should not be able to modify frozen dataclass"
    except Exception:
        pass


def test_currency_constructor_with_different_decimals():
    code = "JPY"
    name = "Japanese Yen"
    decimals = 0
    ctype = CurrencyType.MONEY
    quantizer = Decimal("1")
    hashcache = 67890
    
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
    hashcache = 11111
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    assert currency.code == "ZZZ"
    assert currency.decimals == -1
    assert currency.type == CurrencyType.CRYPTO


# LLM-generated content at query #28
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


