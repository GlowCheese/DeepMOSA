####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_currency_constructor_valid_assignment():
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

def test_currency_equality_same_hash():
    currency1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    currency2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert currency1 == currency2
    assert hash(currency1) == hash(currency2)

def test_currency_equality_different_name():
    currency1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    currency2 = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert currency1 != currency2
    assert hash(currency1) != hash(currency2)

def test_currency_quantize_usd():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.quantize(Decimal("1.005")) == Decimal("1.00")
    assert usd.quantize(Decimal("1.015")) == Decimal("US Dollars".replace("US Dollars", "1.02")) # Logic check: Decimal('1.02')
    assert usd.quantize(Decimal("1.015")) == Decimal("1.02")

def test_currency_quantize_jpy():
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy.quantize(Decimal("0.5")) == Decimal("0")
    assert jpy.quantize(Decimal("1.5")) == Decimal("2")

def test_currency_quantize_crypto():
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert zzz.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert zzz.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")
```


# LLM-generated content at query #2
#--------------------------

```python
def test_currency_registry_constructor_initialization():
    registry = CurrencyRegistry()
    assert len(registry) == 0
    assert registry.all == []
    assert registry.codes == []
    assert registry.codenames == []
```


# LLM-generated content at query #3
#--------------------------

```python
def test_registry_context_open_flag_initialization_is_false():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open is False
```


# LLM-generated content at query #4
#--------------------------

```python
def test_codes_buffer_is_empty_on_init():
    registry = CurrencyRegistry()
    assert len(registry.codes) == 0
```


# LLM-generated content at query #5
#--------------------------

```python
def test_currency_registry_constructor_initializes_empty_containers():
    registry = CurrencyRegistry()
    assert len(registry) == 0
    assert registry.all == []
    assert registry.codes == []
    assert registry.codenames == []
```


# LLM-generated content at query #6
#--------------------------

```python
def test_currency_registry_constructor_initializes_empty_state():
    registry = CurrencyRegistry()
    assert len(registry) == 0
    assert registry.all == []
    assert registry.codes == []
    assert registry.codenames == []
```


# LLM-generated content at query #7
#--------------------------

```python
def test_currency_constructor_valid_assignment():
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

def test_currency_equality_same_hashcache():
    code = "USD"
    name = "US Dollars"
    decimals = 2
    ctype = CurrencyType.MONTO
    quantizer = Decimal("0.01")
    hashcache = 999
    currency1 = Currency(code, name, decimals, ctype, quantizer, hashcache)
    currency2 = Currency(code, name, decimals, ctype, quantizer, hashcache)
    assert currency1 == currency2

def test_currency_equality_different_hashcache():
    code = "USD"
    name = "US Dollars"
    decimals = 2
    ctype = CurrencyType.MONEY
    quantizer = Decimal("0.01")
    currency1 = Currency(code, name, decimals, ctype, quantizer, 111)
    currency2 = Currency(code, name, decimals, ctype, quantizer, 222)
    assert currency1 != currency2

def test_currency_hash_returns_hashcache():
    code = "USD"
    name = "US Dollars"
    decimals = 2
    ctype = CurrencyType.MONEY
    quantizer = Decimal("0.01")
    hashcache = 777
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    assert hash(currency) == 777
```


# LLM-generated content at query #8
#--------------------------

```python
def test_currency_registry_constructor_initialization():
    registry = CurrencyRegistry()
    assert len(registry) == 0
    assert registry.all == []
    assert registry.codes == []
    assert registry.codenames == []
```


# LLM-generated content at query #9
#--------------------------

```python
def test_currency_registry_init_ctx_open_is_false():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open is False
```


# LLM-generated content at query #10
#--------------------------

```python
def test_currency_constructor_valid_initialization():
    quantizer = Decimal("0.01")
    hashcode = 12345
    currency = Currency(code="USD", name="US Dollars", decimals=2, type=CurrencyType.MONEY, quantizer=quantizer, hashcache=hashcode)
    assert currency.code == "USD"
    assert currency.name == "US Dollars"
    assert currency.decimals == 2
    assert currency.type == CurrencyType.MONEY
    assert currency.quantizer == quantizer
    assert currency.hashcache == hashcode

def test_currency_equality_same_hashcache():
    currency1 = Currency(code="USD", name="US Dollars", decimals=2, type=CurrencyType.MONEY, quantizer=Decimal("0.01"), hashcache=100)
    currency2 = Currency(code="USD", name="US Dollars", decimals=2, type=CurrencyType.MONEY, quantizer=Decimal("0.01"), hashcache=100)
    assert currency1 == currency2

def test_currency_inequality_different_hashcache():
    currency1 = Currency(code="USD", name="US Dollars", decimals=2, type=CurrencyType.MONEY, quantizer=Decimal("0.01"), hashcache=100)
    currency2 = Currency(code="USD", name="US Dollars", decimals=2, type=CurrencyType.MONEY, quantizer=Decimal("0.01"), hashcache=200)
    assert currency1 != currency2

def test_currency_hash_returns_hashcache():
    hashcode = 999
    currency = Currency(code="EUR", name="Euro", decimals=2, type=CurrencyType.MONEY, quantizer=Decimal("0.01"), hashcache=hashcode)
    assert hash(currency) == hashcode
```


# LLM-generated content at query #11
#--------------------------

```python
def test_codes_buffer_is_not_empty_after_registration():
    registry = CurrencyRegistry()
    currency_aed = Currency.of("AED", "UAE Dirham", 2, CurrencyType.MONEY)
    currency_usd = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    
    with registry as register:
        register(currency_aed)
        register(currency_usd)
    
    assert len(registry.codes) != 0
```


# LLM-generated content at query #12
#--------------------------

```python
def test_currency_constructor_valid_assignment():
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

def test_currency_constructor_equality_and_hash():
    code = "EUR"
    name = "Euro"
    decimals = 2
    ctype = CurrencyType.MONTO
    quantizer = Decimal("0.01")
    hashcache = 999
    
    currency1 = Currency(code, name, decimals, ctype, quantizer, hashcache)
    currency2 = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    assert currency1 == currency2
    assert hash(currency1) == 999
    assert hash(currency2) == 999

def test_currency_constructor_inequality():
    code1 = "USD"
    name1 = "US Dollars"
    decimals1 = 2
    ctype1 = CurrencyType.MONEY
    quantizer1 = Decimal("0.01")
    hashcache1 = 111
    
    code2 = "USD"
    name2 = "UX Dollars"
    decimals2 = 2
    ctype2 = CurrencyType.MONEY
    quantizer2 = Decimal("0.01")
    hashcache2 = 222
    
    currency1 = Currency(code1, name1, decimals1, ctype1, quantizer1, hashcache1)
    currency2 = Currency(code2, name2, decimals2, ctype2, quantizer2, hashcache2)
    
    assert currency1 != currency2
    assert hash(currency1) != hash(currency2)
```


# LLM-generated content at query #13
#--------------------------

```python
def test_currency_constructor_valid_assignment():
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

def test_currency_equality_same_hashcache():
    code = "USD"
    name = "US Dollars"
    decimals = 2
    ctype = CurrencyType.MONEY
    quantizer = Decimal("0.01")
    hashcache = 999
    currency1 = Currency(code, name, decimals, ctype, quantizer, hashcache)
    currency2 = Currency(code, name, decimals, ctype, quantizer, hashcache)
    assert currency1 == currency2

def test_currency_inequality_different_hashcache():
    code = "USD"
    name = "US Dollars"
    decimals = 2
    ctype = CurrencyType.MONEY
    quantizer = Decimal("0.01")
    currency1 = Currency(code, name, decimals, ctype, quantizer, 111)
    currency2 = Currency(code, name, decimals, ctype, quantizer, 222)
    assert currency1 != currency2

def test_currency_hash_return_value():
    code = "USD"
    name = "US Dollars"
    decimals = 2
    ctype = CurrencyType.MONEY
    quantizer = Decimal("0.01")
    hashcache = 888
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    assert hash(currency) == 888
```


# LLM-generated content at query #14
#--------------------------

```python
def test_codes_buffer_is_empty_after_init():
    registry = CurrencyRegistry()
    assert len(registry.codes) == 0
```


# LLM-generated content at query #15
#--------------------------

```python
def test_registry_initialization_context_flag_is_false():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open is False
```


# LLM-generated content at query #16
#--------------------------

```python
def test_currency_constructor_valid_parameters():
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

def test_currency_equality_same_hashcache():
    code = "USD"
    name = "US Dollars"
    decimals = 2
    ctype = CurrencyType.MONEY
    quantizer = Decimal("0.01")
    hashcache = 999
    currency1 = Currency(code, name, decimals, ctype, quantizer, hashcache)
    currency2 = Currency(code, name, decimals, ctype, quantizer, hashcache)
    assert currency1 == currency2

def test_currency_inequality_different_hashcache():
    code = "USD"
    name = "US Dollars"
    decimals = 2
    ctype = CurrencyType.MONEY
    quantizer = Decimal("0.01")
    currency1 = Currency(code, name, decimals, ctype, quantizer, 111)
    currency2 = Currency(code, name, decimals, ctype, quantizer, 222)
    assert currency1 != currency2

def test_currency_hash_return_value():
    code = "USD"
    name = "US Dollars"
    decimals = 2
    ctype = CurrencyType.MONEY
    quantizer = Decimal("0.01")
    hashcache = 777
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    assert hash(currency) == 777
```


# LLM-generated content at query #17
#--------------------------

```python
def test_init_ctx_open_is_false():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open is False
```


# LLM-generated content at query #18
#--------------------------

```python
def test_codes_buffer_is_empty_after_initialization():
    registry = CurrencyRegistry()
    assert len(registry.codes) == 0
```


# LLM-generated content at query #19
#--------------------------

```python
def test_currency_constructor_valid_initialization():
    quantizer = Decimal("0.01")
    hashcode = 12345
    currency = Currency(
        code="USD",
        name="US Dollars",
        decimals=2,
        type=CurrencyType.MONEY,
        quantizer=quantizer,
        hashcache=hashcode
    )
    assert currency.code == "USD"
    assert currency.name == "US Dollars"
    assert currency.decimals == 2
    assert currency.type == CurrencyType.MONEY
    assert currency.quantizer == quantizer
    assert currency.hashcache == hashcode

def test_currency_equality_same_hashcache():
    currency1 = Currency("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal("0.01"), 999)
    currency2 = Currency("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal("0.01"), 999)
    assert currency1 == currency2

def test_currency_equality_different_hashcache():
    currency1 = Currency("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal("0.01"), 999)
    currency2 = Currency("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal("0.01"), 888)
    assert currency1 != currency2

def test_currency_hash_returns_hashcache():
    hashcode = 12345
    currency = Currency("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal("0.01"), hashcode)
    assert hash(currency) == hashcode

def test_currency_order_comparison():
    currency1 = Currency("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal("0.01"), 1)
    currency2 = Currency("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal("0.01"), 2)
    assert currency1 < currency2
```


# LLM-generated content at query #20
#--------------------------

```python
def test_currency_codes_buffer_is_empty_after_init():
    registry = CurrencyRegistry()
    assert len(registry.codes) == 0
```


# LLM-generated content at query #21
#--------------------------

```python
def test_registry_context_open_flag_initialization_is_false():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open is False
```


# LLM-generated content at query #22
#--------------------------

```python
def test_currency_constructor_valid_assignment():
    quantizer = Decimal("0.01")
    hashcode = 12345
    ctype = CurrencyType.MONEY
    currency = Currency(code="USD", name="US Dollars", decimals=2, type=ctype, quantizer=quantizer, hashcache=hashcode)
    assert currency.code == "USD"
    assert currency.name == "US Dollars"
    assert currency.decimals == 2
    assert currency.type == ctype
    assert currency.quantizer == quantizer
    assert currency.hashcache == hashcode

def test_currency_equality_logic():
    ctype = CurrencyType.MONEY
    usd1 = Currency.of("USD", "US Dollars", 2, ctype)
    usd2 = Currency.of("USD", "US Dollars", 2, ctype)
    usdx = Currency.of("USD", "UX Dollars", 2, ctype)
    assert usd1 == usd1
    assert usd1 == usd2
    assert usd1 != usdx

def test_currency_hash_consistency():
    ctype = CurrencyType.MONEY
    usd1 = Currency.of("USD", "US Dollars", 2, ctype)
    usd2 = Currency.of("USD", "US Dollars", 2, ctype)
    assert hash(usd1) == hash(usd2)

def test_currency_of_factory_creates_correct_quantizer_for_positive_decimals():
    ctype = CurrencyType.MONEY
    usd = Currency.of("USD", "US Dollars", 2, ctype)
    assert usd.decimals == 2
    assert usd.quantize(Decimal("1.005")) == Decimal("1.00")
    assert usd.quantize(Decimal("1.015")) == Decimal("1.02")

def test_currency_of_factory_creates_correct_quantizer_for_zero_decimals():
    ctype = CurrencyType.MONEY
    jpy = Currency.of("JPY", "Japanese Yen", 0, ctype)
    assert jpy.decimals == 0
    assert jpy.quantize(Decimal("0.5")) == Decimal("0")
    assert jpy.quantize(Decimal("1.5")) == Decimal("2")

def test_currency_of_factory_creates_correct_quantizer_for_negative_decimals():
    ctype = CurrencyType.CRYPTO
    zzz = Currency.of("ZZZ", "Some weird currency", -1, ctype)
    assert zzz.decimals == -1
    assert zzz.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert zzz.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")
```


# LLM-generated content at query #23
#--------------------------

```python
def test_codes_is_not_empty_after_initialization_is_false():
    registry = CurrencyRegistry()
    assert len(registry.codes) == 0
```


# LLM-generated content at query #24
#--------------------------

```python
def test_currency_registry_constructor_initialization():
    registry = CurrencyRegistry()
    assert len(registry) == 0
    assert registry.all == []
    assert registry.codes == []
    assert registry.codenames == []
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_currency_eq_equality_same_attributes():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2

def test_currency_eq_inequality_different_name():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert usd1 != usdx

def test_currency_eq_inequality_different_code():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    gbp = Currency.of("GBP", "US Dollars", 2, CurrencyType.MONEY)
    assert usd != gbp

def test_currency_eq_inequality_different_decimals():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "US Dollars", 0, CurrencyType.MONEY)
    assert usd != jpy

def test_currency_eq_inequality_different_type():
    usd_money = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd_crypto = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert usd_money != usd_crypto

def test_currency_eq_with_different_type_object():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd != "USD"
```


# LLM-generated content at query #2
#--------------------------

```python
def test_currency_registry_constructor_initialization():
    registry = CurrencyRegistry()
    assert len(registry) == 0
    assert registry.all == []
    assert registry.codes == []
    assert registry.codenames == []
```


# LLM-generated content at query #3
#--------------------------

```python
def test_currency_registry_initial_codes_is_empty():
    registry = CurrencyRegistry()
    assert len(registry.codes) == 0
```


# LLM-generated content at query #4
#--------------------------

```python
def test_currency_constructor_valid_initialization():
    quantizer = Decimal("0.01")
    hashcache = 12345
    currency = Currency(code="USD", name="US Dollars", decimals=2, type=CurrencyType.MONEY, quantizer=quantizer, hashcache=hashcache)
    assert currency.code == "USD"
    assert currency.name == "US Dollars"
    assert currency.decimals == 2
    assert currency.type == CurrencyType.MONEY
    assert currency.quantizer == quantizer
    assert currency.hashcache == hashcache

def test_currency_equality_same_hashcache():
    currency1 = Currency(code="USD", name="US Dollars", decimals=2, type=CurrencyType.MONEY, quantizer=Decimal("0.01"), hashcache=100)
    currency2 = Currency(code="USD", name="US Dollars", decimals=2, type=CurrencyType.MONEY, quantizer=Decimal("0.01"), hashcache=100)
    assert currency1 == currency2

def test_currency_inequality_different_hashcache():
    currency1 = Currency(code="USD", name="US Dollars", decimals=2, type=CurrencyType.MONEY, quantizer=Decimal("0.01"), hashcache=100)
    currency2 = Currency(code="USD", name="UX Dollars", decimals=2, type=CurrencyType.MONEY, quantizer=Decimal("0.01"), hashcache=200)
    assert currency1 != currency2

def test_currency_hash_returns_hashcache():
    hash_val = 999
    currency = Currency(code="USD", name="US Dollars", decimals=2, type=CurrencyType.MONEY, quantizer=Decimal("0.01"), hashcache=hash_val)
    assert hash(currency) == hash_val
```


# LLM-generated content at query #5
#--------------------------

```python
def test_currency_registry_constructor_initialization():
    registry = CurrencyRegistry()
    assert len(registry) == 0
    assert registry.all == []
    assert registry.codes == []
    assert registry.codenames == []
```


# LLM-generated content at query #6
#--------------------------

```python
def test_currency_registry_constructor_initialization():
    registry = CurrencyRegistry()
    assert len(registry) == 0
    assert registry.all == []
    assert registry.codes == []
    assert registry.codenames == []
```


# LLM-generated content at query #7
#--------------------------

```python
def test_currency_constructor_success():
    quantizer = Decimal("0.01")
    hashcode = 12345
    currency = Currency(
        code="USD",
        name="US Dollars",
        decimals=2,
        type=CurrencyType.MONEY,
        quantizer=quantizer,
        hashcache=hashcode
    )
    assert currency.code == "USD"
    assert currency.name == "US Dollars"
    assert currency.decimals == 2
    assert currency.type == CurrencyType.MONEY
    assert currency.quantizer == quantizer
    assert currency.hashcache == hashcode

def test_currency_equality_same_attributes():
    quantizer = Decimal("0.01")
    currency1 = Currency("USD", "US Dollars", 2, CurrencyType.MONEY, quantizer, 100)
    currency2 = Currency("USD", "US Dollars", 2, CurrencyType.MONEY, quantizer, 100)
    assert currency1 == currency2

def test_currency_inequality_different_hashcache():
    quantizer = Decimal("0.01")
    currency1 = Currency("USD", "US Dollars", 2, CurrencyType.MONEY, quantizer, 100)
    currency2 = Currency("USD", "US Dollars", 2, CurrencyType.MONEY, quantizer, 200)
    assert currency1 != currency2

def test_currency_hash():
    quantizer = Decimal("0.01")
    hashcode = 999
    currency = Currency("USD", "US Dollars", 2, CurrencyType.MONEY, quantizer, hashcode)
    assert hash(currency) == hashcode
```


# LLM-generated content at query #8
#--------------------------

```python
def test_currency_constructor_valid_assignment():
    quantizer = Decimal("0.01")
    hashcode = 12345
    currency = Currency(
        code="USD",
        name="US Dollars",
        decimals=2,
        type=CurrencyType.MONEY,
        quantizer=quantizer,
        hashcache=hashcode
    )
    assert currency.code == "USD"
    assert currency.name == "US Dollars"
    assert currency.decimals == 2
    assert currency.type == CurrencyType.MONEY
    assert currency.quantizer == quantizer
    assert currency.hashcache == hashcode

def test_currency_equality_same_hash():
    usd1 = Currency("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal("0.01"), 100)
    usd2 = Currency("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal("0.01"), 100)
    assert usd1 == usd2

def test_currency_inequality_different_hash():
    usd1 = Currency("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal("0.01"), 100)
    usd2 = Currency("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal("0.01"), 200)
    assert usd1 != usd2

def test_currency_hash_consistency():
    hashcode = 12345
    currency = Currency("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal("0.01"), hashcode)
    assert hash(currency) == hashcode
```


# LLM-generated content at query #9
#--------------------------

```python
def test_currency_registry_constructor_initialization():
    registry = CurrencyRegistry()
    assert len(registry) == 0
    assert registry.all == []
    assert registry.codes == []
    assert registry.codenames == []
```


# LLM-generated content at query #10
#--------------------------

```python
def test_init_ctx_open_is_false():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open is False
```


# LLM-generated content at query #11
#--------------------------

```python
def test_currency_constructor_valid_initialization():
    quantizer = Decimal("0.01")
    hashcache = 12345
    currency = Currency(code="USD", name="US Dollars", decimals=2, type=CurrencyType.MONEY, quantizer=quantizer, hashcache=hashcache)
    assert currency.code == "USD"
    assert currency.name == "US Dollars"
    assert currency.decimals == 2
    assert currency.type == CurrencyType.MONEY
    assert currency.quantizer == quantizer
    assert currency.hashcache == hashcache

def test_currency_equality_with_same_hashcache():
    currency1 = Currency(code="USD", name="US Dollars", decimals=2, type=CurrencyType.MONEY, quantizer=Decimal("0.01"), hashcache=100)
    currency2 = Currency(code="USD", name="Different Name", decimals=2, type=CurrencyType.MONEY, quantizer=Decimal("0.01"), hashcache=100)
    assert currency1 == currency2

def test_currency_inequality_with_different_hashcache():
    currency1 = Currency(code="USD", name="US Dollars", decimals=2, type=CurrencyType.MONEY, quantizer=Decimal("0.01"), hashcache=100)
    currency2 = Currency(code="USD", name="US Dollars", decimals=2, type=CurrencyType.MONEY, quantizer=Decimal("0.01"), hashcache=200)
    assert currency1 != currency2

def test_currency_hash_returns_hashcache():
    hash_val = 999
    currency = Currency(code="EUR", name="Euro", decimals=2, type=CurrencyType.MONEY, quantizer=Decimal("0.01"), hashcache=hash_val)
    assert hash(currency) == hash_val
```


# LLM-generated content at query #12
#--------------------------

```python
def test_currency_registry_constructor_initialization():
    registry = CurrencyRegistry()
    assert len(registry) == 0
    assert registry.all == []
    assert registry.codes == []
    assert registry.codenames == []
```


# LLM-generated content at query #13
#--------------------------

```python
def test_currency_registry_constructor_initializes_empty_state():
    registry = CurrencyRegistry()
    assert len(registry) == 0
    assert registry.all == []
    assert registry.codes == []
    assert registry.codenames == []
```


# LLM-generated content at query #14
#--------------------------

```python
def test_currency_registry_constructor_initialization():
    registry = CurrencyRegistry()
    assert len(registry) == 0
    assert registry.all == []
    assert registry.codes == []
    assert registry.codenames == []
```


# LLM-generated content at query #15
#--------------------------

```python
def test_currency_constructor_success():
    quantizer = Decimal("0.01")
    hashcode = 12345
    currency = Currency("USD", "US Dollars", 2, CurrencyType.MONEY, quantizer, hashcode)
    assert currency.code == "USD"
    assert currency.name == "US Dollars"
    assert currency.decimals == 2
    assert currency.type == CurrencyType.MONEY
    assert currency.quantizer == quantizer
    assert currency.hashcache == hashcode

def test_currency_equality_same_hashcache():
    currency1 = Currency("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal("0.01"), 123)
    currency2 = Currency("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal("0.01"), 123)
    assert currency1 == currency2

def test_currency_inequality_different_hashcache():
    currency1 = Currency("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal("0.01"), 123)
    currency2 = Currency("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal("0.01"), 456)
    assert currency1 != currency2

def test_currency_hash_consistency():
    hashcode = 12345
    currency = Currency("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal("0.01"), hashcode)
    assert hash(currency) == hashcode
```


# LLM-generated content at query #16
#--------------------------

```python
def test_currency_registry_constructor_initialization_state():
    registry = CurrencyRegistry()
    assert len(registry) == 0
    assert registry.all == []
    assert registry.codes == []
    assert registry.codenames == []
    assert "USD" not in registry
    assert registry.has("USD") is False
```


# LLM-generated content at query #17
#--------------------------

```python
def test_currency_constructor_valid_assignment():
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

def test_currency_equality_same_hashcache():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2

def test_currency_inequality_different_name():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert usd1 != usdx

def test_currency_hash_consistency():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert hash(usd1) == hash(usd2)

def test_currency_of_factory_creation_usd():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.code == "USD"
    assert usd.decimals == 2
    assert usd.quantizer == Decimal("0.01")

def test_currency_of_factory_creation_jpy():
    jpy = Currency.of("JPY", "Japanese Yen", 0, Currencyty.MONEY)
    assert jpy.decimals == 0
    assert jpy.quantizer == Decimal("1")

def test_currency_of_factory_creation_crypto():
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert zzz.decimals == -1
```


# LLM-generated content at query #18
#--------------------------

```python
def test_currency_registry_constructor_initializes_empty_state():
    registry = CurrencyRegistry()
    assert len(registry) == 0
    assert registry.all == []
    assert registry.codes == []
    assert registry.codenames == []
```


# LLM-generated content at query #19
#--------------------------

```python
def test_codes_buffer_is_empty_on_init():
    registry = CurrencyRegistry()
    assert len(registry.codes) == 0
```


# LLM-generated content at query #20
#--------------------------

```python
def test_currency_constructor_valid_assignment():
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

def test_currency_equality_same_hashcache():
    code = "USD"
    name = "US Dollars"
    decimals = 2
    ctype = CurrencyType.MONTO
    quantizer = Decimal("0.01")
    hashcache = 999
    currency1 = Currency(code, name, decimals, ctype, quantizer, hashcache)
    currency2 = Currency(code, name, decimals, ctype, quantizer, hashcache)
    assert currency1 == currency2

def test_currency_inequality_different_hashcache():
    code = "USD"
    name = "US Dollars"
    decimals = 2
    ctype = CurrencyType.MONEY
    quantizer = Decimal("0.01")
    currency1 = Currency(code, name, decimals, ctype, quantizer, 111)
    currency2 = Currency(code, name, decimals, ctype, quantizer, 222)
    assert currency1 != currency2

def test_currency_hash_returns_hashcache():
    code = "USD"
    name = "US Dollars"
    decimals = 2
    ctype = CurrencyType.MONEY
    quantizer = Decimal("0.01")
    hashcache = 888
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    assert hash(currency) == 888
```


# LLM-generated content at query #21
#--------------------------

```python
def test_init_ctx_open_is_false():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open is False
```


# LLM-generated content at query #22
#--------------------------

```python
def test_currency_registry_constructor_initialization():
    registry = CurrencyRegistry()
    assert len(registry) == 0
    assert registry.all == []
    assert registry.codes == []
    assert registry.codenames == []
```


# LLM-generated content at query #23
#--------------------------

```python
def test_registry_initialization_ctx_open_is_false():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open is False
```


# LLM-generated content at query #24
#--------------------------

```python
def test_currency_registry_constructor_initialization():
    registry = CurrencyRegistry()
    assert len(registry) == 0
    assert registry.all == []
    assert registry.codes == []
    assert registry.codenames == []
```


# LLM-generated content at query #25
#--------------------------

```python
def test_currency_constructor_valid_assignment():
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

def test_currency_equality_same_hashcache():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2

def test_currency_inequality_different_name():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert usd1 != usdx

def test_currency_hash_consistency():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert hash(usd1) == hash(usd2)

def test_currency_hash_inconsistency():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert hash(usd1) != hash(usdx)
```


