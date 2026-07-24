####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_CurrencyRegistry_constructor():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__registry == {}
    assert registry._CurrencyRegistry__currencies == []
    assert registry._CurrencyRegistry__codes == []
    assert registry._CurrencyRegistry__codenames == []
    assert registry._CurrencyRegistry__ctx_open == False


# LLM-generated content at query #2
#--------------------------

```python
def test_currency_registry_constructor():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__registry == OrderedDict([])
    assert registry._CurrencyRegistry__currencies == []
    assert registry._CurrencyRegistry__codes == []
    assert registry._CurrencyRegistry__codenames == []
    assert registry._CurrencyRegistry__ctx_open == False


# LLM-generated content at query #3
#--------------------------

```python
def test_currency_registry_initialization():
    registry = CurrencyRegistry()
    assert not registry._CurrencyRegistry__codes


# LLM-generated content at query #4
#--------------------------

```python
def test_initial_context_open_flag_is_false():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open is False


# LLM-generated content at query #5
#--------------------------

```python
def test_currency_registry_constructor():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__registry == {}
    assert registry._CurrencyRegistry__currencies == []
    assert registry._CurrencyRegistry__codes == []
    assert registry._CurrencyRegistry__codenames == []
    assert registry._CurrencyRegistry__ctx_open is False


# LLM-generated content at query #6
#--------------------------

```python
def test_ctx_open_initialized_to_false():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open is False


# LLM-generated content at query #7
#--------------------------

```python
def test_currency_constructor():
    currency = Currency("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal('0.01'), hash(("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal('0.01'))))
    assert currency.code == "USD"
    assert currency.name == "US Dollars"
    assert currency.decimals == 2
    assert currency.type == CurrencyType.MONEY
    assert currency.quantizer == Decimal('0.01')
    assert currency.hashcache == hash(("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal('0.01')))


# LLM-generated content at query #8
#--------------------------

```python
def test_currency_constructor_with_valid_arguments():
    currency = Currency("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal('0.01'), hash(("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal('0.01'))))
    assert currency.code == "USD"
    assert currency.name == "US Dollars"
    assert currency.decimals == 2
    assert currency.type == CurrencyType.MONEY
    assert currency.quantizer == Decimal('0.01')
    assert currency.hashcache == hash(("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal('0.01')))

def test_currency_constructor_with_zero_decimals():
    currency = Currency("JPY", "Japanese Yen", 0, CurrencyType.MONEY, Decimal('1'), hash(("JPY", "Japanese Yen", 0, CurrencyType.MONEY, Decimal('1'))))
    assert currency.decimals == 0
    assert currency.quantizer == Decimal('1')

def test_currency_constructor_with_negative_decimals():
    currency = Currency("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO, MaxPrecisionQuantizer, hash(("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO, MaxPrecisionQuantizer)))
    assert currency.decimals == -1
    assert currency.quantizer == MaxPrecisionQuantizer

def test_currency_constructor_with_cached_hash():
    hash_value = hash(("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal('0.01')))
    currency = Currency("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal('0.01'), hash_value)
    assert hash(currency) == hash_value

def test_currency_equality_with_same_hash():
    currency1 = Currency("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal('0.01'), hash(("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal('0.01'))))
    currency2 = Currency("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal('0.01'), hash(("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal('0.01'))))
    assert currency1 == currency2

def test_currency_equality_with_different_hash():
    currency1 = Currency("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal('0.01'), hash(("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal('0.01'))))
    currency2 = Currency("USD", "UX Dollars", 2, CurrencyType.MONEY, Decimal('0.01'), hash(("USD", "UX Dollars", 2, CurrencyType.MONEY, Decimal('0.01'))))
    assert currency1 != currency2


# LLM-generated content at query #9
#--------------------------

```
def test_currency_registry_constructor_singleton():
    instance1 = CurrencyRegistry()
    instance2 = CurrencyRegistry()
    assert instance1 is instance2

def test_currency_registry_constructor_initial_state():
    registry = CurrencyRegistry()
    assert len(registry) == 0
    assert registry.all == []
    assert registry.codes == []
    assert registry.codenames == []

def test_currency_registry_constructor_private_attributes():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__registry == OrderedDict([])
    assert registry._CurrencyRegistry__currencies == []
    assert registry._CurrencyRegistry__codes == []
    assert registry._CurrencyRegistry__codenames == []
    assert registry._CurrencyRegistry__ctx_open is False


# LLM-generated content at query #10
#--------------------------

```python
def test_currency_codes_buffer_is_empty_after_initialization():
    registry = CurrencyRegistry()
    assert not registry.codes


# LLM-generated content at query #11
#--------------------------

```python
def test_currency_constructor():
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


# LLM-generated content at query #12
#--------------------------

```python
def test_currency_registry_constructor():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__registry == {}
    assert registry._CurrencyRegistry__currencies == []
    assert registry._CurrencyRegistry__codes == []
    assert registry._CurrencyRegistry__codenames == []
    assert registry._CurrencyRegistry__ctx_open is False


# LLM-generated content at query #13
#--------------------------

```python
def test_currency_constructor_valid_input():
    currency = Currency("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal("0.01"), hash(("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal("0.01"))))
    assert currency.code == "USD"
    assert currency.name == "US Dollars"
    assert currency.decimals == 2
    assert currency.type == CurrencyType.MONEY
    assert currency.quantizer == Decimal("0.01")
    assert currency.hashcache == hash(("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal("0.01")))

def test_currency_constructor_equality_check():
    currency1 = Currency("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal("0.01"), hash(("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal("0.01"))))
    currency2 = Currency("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal("0.01"), hash(("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal("0.01"))))
    assert currency1 == currency2
    assert hash(currency1) == hash(currency2)

def test_currency_constructor_inequality_check():
    currency1 = Currency("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal("0.01"), hash(("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal("0.01"))))
    currency2 = Currency("USD", "UX Dollars", 2, CurrencyType.MONEY, Decimal("0.01"), hash(("USD", "UX Dollars", 2, CurrencyType.MONEY, Decimal("0.01"))))
    assert currency1 != currency2
    assert hash(currency1) != hash(currency2)


# LLM-generated content at query #14
#--------------------------

```python
def test_registry_population_context_flag_initialized_to_false():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open == False


# LLM-generated content at query #15
#--------------------------

```
def test_registry_population_context_flag_initialized_to_false():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open is False


# LLM-generated content at query #16
#--------------------------

```python
def test_currency_constructor_with_valid_inputs():
    currency = Currency(code="USD", name="US Dollars", decimals=2, type=CurrencyType.MONEY, quantizer=Decimal('0.01'), hashcache=hash(("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal('0.01'))))
    assert currency.code == "USD"
    assert currency.name == "US Dollars"
    assert currency.decimals == 2
    assert currency.type == CurrencyType.MONEY
    assert currency.quantizer == Decimal('0.01')
    assert currency.hashcache == hash(("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal('0.01')))

def test_currency_constructor_with_invalid_code():
    try:
        Currency(code="usd", name="US Dollars", decimals=2, type=CurrencyType.MONEY, quantizer=Decimal('0.01'), hashcache=hash(("usd", "US Dollars", 2, CurrencyType.MONEY, Decimal('0.01'))))
        assert False
    except ProgrammingError:
        assert True

def test_currency_constructor_with_invalid_name():
    try:
        Currency(code="USD", name="", decimals=2, type=CurrencyType.MONEY, quantizer=Decimal('0.01'), hashcache=hash(("USD", "", 2, CurrencyType.MONEY, Decimal('0.01'))))
        assert False
    except ProgrammingError:
        assert True

def test_currency_constructor_with_invalid_decimals():
    try:
        Currency(code="USD", name="US Dollars", decimals=-2, type=CurrencyType.MONEY, quantizer=Decimal('0.01'), hashcache=hash(("USD", "US Dollars", -2, CurrencyType.MONEY, Decimal('0.01'))))
        assert False
    except ProgrammingError:
        assert True

def test_currency_constructor_with_invalid_type():
    try:
        Currency(code="USD", name="US Dollars", decimals=2, type="INVALID_TYPE", quantizer=Decimal('0.01'), hashcache=hash(("USD", "US Dollars", 2, "INVALID_TYPE", Decimal('0.01'))))
        assert False
    except ProgrammingError:
        assert True


# LLM-generated content at query #17
#--------------------------

```python
def test_ctx_open_flag_initialized_as_false():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open is False


# LLM-generated content at query #18
#--------------------------

```python
def test_currency_registry_init_codes_empty():
    registry = CurrencyRegistry()
    assert not registry._CurrencyRegistry__codes


# LLM-generated content at query #19
#--------------------------

```python
def test_CurrencyRegistry_constructor():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__registry == {}
    assert registry._CurrencyRegistry__currencies == []
    assert registry._CurrencyRegistry__codes == []
    assert registry._CurrencyRegistry__codenames == []
    assert registry._CurrencyRegistry__ctx_open == False


# LLM-generated content at query #20
#--------------------------

```python
def test_currency_equality():
    currency1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    currency2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert currency1 == currency2

def test_currency_inequality():
    currency1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    currency2 = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert currency1 != currency2

def test_currency_hash_equality():
    currency1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    currency2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert hash(currency1) == hash(currency2)

def test_currency_hash_inequality():
    currency1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    currency2 = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert hash(currency1) != hash(currency2)

def test_currency_quantize():
    currency = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert currency.quantize(Decimal("1.005")) == Decimal("1.00")
    assert currency.quantize(Decimal("1.015")) == Decimal("1.02")

def test_currency_of_invalid_code():
    try:
        Currency.of("usd", "US Dollars", 2, CurrencyType.MONEY)
        assert False, "Expected ProgrammingError"
    except ProgrammingError:
        pass

def test_currency_of_empty_name():
    try:
        Currency.of("USD", "", 2, CurrencyType.MONEY)
        assert False, "Expected ProgrammingError"
    except ProgrammingError:
        pass

def test_currency_of_invalid_decimals():
    try:
        Currency.of("USD", "US Dollars", -2, CurrencyType.MONEY)
        assert False, "Expected ProgrammingError"
    except ProgrammingError:
        pass

def test_currency_of_invalid_type():
    try:
        Currency.of("USD", "US Dollars", 2, "INVALID_TYPE")
        assert False, "Expected ProgrammingError"
    except ProgrammingError:
        pass


# LLM-generated content at query #21
#--------------------------

```python
def test_registry_initialization():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__registry == OrderedDict()
    assert registry._CurrencyRegistry__currencies == []
    assert registry._CurrencyRegistry__codes == []
    assert registry._CurrencyRegistry__codenames == []
    assert registry._CurrencyRegistry__ctx_open is False


# LLM-generated content at query #22
#--------------------------

```
def test_constructor_initializes_registry():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__registry == OrderedDict([])
    assert registry._CurrencyRegistry__currencies == []
    assert registry._CurrencyRegistry__codes == []
    assert registry._CurrencyRegistry__codenames == []
    assert registry._CurrencyRegistry__ctx_open == False

def test_constructor_returns_singleton_instance():
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2


# LLM-generated content at query #23
#--------------------------

```python
def test_currency_constructor_with_valid_inputs():
    currency = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert currency.code == "USD"
    assert currency.name == "US Dollars"
    assert currency.decimals == 2
    assert currency.type == CurrencyType.MONEY
    assert currency.quantizer == make_quantizer(2)
    assert currency.hashcache == hash(("USD", "US Dollars", 2, CurrencyType.MONEY, make_quantizer(2)))

def test_currency_constructor_with_zero_decimals():
    currency = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert currency.code == "JPY"
    assert currency.name == "Japanese Yen"
    assert currency.decimals == 0
    assert currency.type == CurrencyType.MONEY
    assert currency.quantizer == ZERO
    assert currency.hashcache == hash(("JPY", "Japanese Yen", 0, CurrencyType.MONEY, ZERO))

def test_currency_constructor_with_negative_decimals():
    currency = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert currency.code == "ZZZ"
    assert currency.name == "Some weird currency"
    assert currency.decimals == -1
    assert currency.type == CurrencyType.CRYPTO
    assert currency.quantizer == MaxPrecisionQuantizer
    assert currency.hashcache == hash(("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO, MaxPrecisionQuantizer))

def test_currency_constructor_raises_error_for_invalid_code():
    try:
        Currency.of("usd", "US Dollars", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        pass

def test_currency_constructor_raises_error_for_empty_name():
    try:
        Currency.of("USD", "", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        pass

def test_currency_constructor_raises_error_for_invalid_decimals():
    try:
        Currency.of("USD", "US Dollars", -2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        pass

def test_currency_constructor_raises_error_for_invalid_type():
    try:
        Currency.of("USD", "US Dollars", 2, "MONEY")
        assert False
    except ProgrammingError:
        pass


# LLM-generated content at query #24
#--------------------------

```python
def test_currency_codes_buffer_is_empty_at_initialization():
    currencies = CurrencyRegistry()
    assert len(currencies.codes) == 0


# LLM-generated content at query #25
#--------------------------

```python
def test_currency_constructor_with_valid_arguments():
    currency = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert currency.code == "USD"
    assert currency.name == "US Dollars"
    assert currency.decimals == 2
    assert currency.type == CurrencyType.MONEY
    assert currency.quantizer == make_quantizer(2)
    assert currency.hashcache == hash(("USD", "US Dollars", 2, CurrencyType.MONEY, make_quantizer(2)))

def test_currency_constructor_with_zero_decimals():
    currency = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert currency.decimals == 0
    assert currency.quantizer == ZERO

def test_currency_constructor_with_negative_decimals():
    currency = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert currency.decimals == -1
    assert currency.quantizer == MaxPrecisionQuantizer

def test_currency_constructor_with_trimmed_name():
    currency = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert currency.name == "Euro"

def test_currency_constructor_equality():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2
    assert hash(usd1) == hash(usd2)

def test_currency_constructor_inequality():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert usd1 != usdx
    assert hash(usd1) != hash(usdx)


# LLM-generated content at query #26
#--------------------------

```python
def test_constructor_creates_singleton_instance():
    instance1 = CurrencyRegistry()
    instance2 = CurrencyRegistry()
    assert instance1 is instance2

def test_constructor_initializes_registry_as_empty():
    registry = CurrencyRegistry()
    assert len(registry) == 0

def test_constructor_initializes_currencies_as_empty():
    registry = CurrencyRegistry()
    assert registry.all == []

def test_constructor_initializes_codes_as_empty():
    registry = CurrencyRegistry()
    assert registry.codes == []

def test_constructor_initializes_codenames_as_empty():
    registry = CurrencyRegistry()
    assert registry.codenames == []

def test_constructor_initializes_context_as_closed():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open is False


# LLM-generated content at query #27
#--------------------------

```
def test_ctx_open_initialized_to_false():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open is False


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```
def test_currency_registry_constructor_singleton():
    instance1 = CurrencyRegistry()
    instance2 = CurrencyRegistry()
    assert instance1 is instance2

def test_currency_registry_constructor_initial_state():
    registry = CurrencyRegistry()
    assert len(registry) == 0
    assert registry.all == []
    assert registry.codes == []
    assert registry.codenames == []


# LLM-generated content at query #2
#--------------------------

```python
def test_CurrencyRegistry_constructor():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__registry == OrderedDict([])
    assert registry._CurrencyRegistry__currencies == []
    assert registry._CurrencyRegistry__codes == []
    assert registry._CurrencyRegistry__codenames == []
    assert registry._CurrencyRegistry__ctx_open == False


# LLM-generated content at query #3
#--------------------------

```python
def test_registry_population_context_flag_initialized_to_false():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open is False


# LLM-generated content at query #4
#--------------------------

```python
def test_currency_constructor_with_valid_arguments():
    currency = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    assert currency.code == "USD"
    assert currency.name == "US Dollar"
    assert currency.decimals == 2
    assert currency.type == CurrencyType.MONEY
    assert currency.quantizer == Decimal('0.01')

def test_currency_constructor_with_zero_decimals():
    currency = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert currency.code == "JPY"
    assert currency.name == "Japanese Yen"
    assert currency.decimals == 0
    assert currency.type == CurrencyType.MONEY
    assert currency.quantizer == Decimal('1')

def test_currency_constructor_with_negative_decimals():
    currency = Currency.of("BTC", "Bitcoin", -1, CurrencyType.CRYPTO)
    assert currency.code == "BTC"
    assert currency.name == "Bitcoin"
    assert currency.decimals == -1
    assert currency.type == CurrencyType.CRYPTO
    assert currency.quantizer == Decimal('0.000000000000')

def test_currency_equality():
    currency1 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    currency2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    currency3 = Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY)
    assert currency1 == currency2
    assert currency1 != currency3

def test_currency_hash():
    currency1 = Currency.of("CAD", "Canadian Dollar", 2, CurrencyType.MONEY)
    currency2 = Currency.of("CAD", "Canadian Dollar", 2, CurrencyType.MONEY)
    currency3 = Currency.of("AUD", "Australian Dollar", 2, CurrencyType.MONEY)
    assert hash(currency1) == hash(currency2)
    assert hash(currency1) != hash(currency3)

def test_currency_quantize_with_positive_decimals():
    currency = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    assert currency.quantize(Decimal('1.005')) == Decimal('1.00')
    assert currency.quantize(Decimal('1.015')) == Decimal('1.02')

def test_currency_quantize_with_zero_decimals():
    currency = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert currency.quantize(Decimal('0.5')) == Decimal('0')
    assert currency.quantize(Decimal('1.5')) == Decimal('2')

def test_currency_quantize_with_negative_decimals():
    currency = Currency.of("ETH", "Ethereum", -1, CurrencyType.CRYPTO)
    assert currency.quantize(Decimal('1.0000000000005')) == Decimal('1.000000000000')
    assert currency.quantize(Decimal('1.0000000000015')) == Decimal('1.000000000002')


# LLM-generated content at query #5
#--------------------------

```python
def test_currency_constructor_creates_valid_instance():
    code = "USD"
    name = "US Dollars"
    decimals = 2
    ctype = CurrencyType.MONEY
    quantizer = make_quantizer(decimals)
    hashcache = hash((code, name, decimals, ctype, quantizer))
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    assert currency.code == code
    assert currency.name == name
    assert currency.decimals == decimals
    assert currency.type == ctype
    assert currency.quantizer == quantizer
    assert currency.hashcache == hashcache

def test_currency_constructor_with_zero_decimals():
    code = "JPY"
    name = "Japanese Yen"
    decimals = 0
    ctype = CurrencyType.MONEY
    quantizer = ZERO
    hashcache = hash((code, name, decimals, ctype, quantizer))
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    assert currency.decimals == decimals
    assert currency.quantizer == quantizer

def test_currency_constructor_with_negative_decimals():
    code = "ZZZ"
    name = "Some weird currency"
    decimals = -1
    ctype = CurrencyType.CRYPTO
    quantizer = MaxPrecisionQuantizer
    hashcache = hash((code, name, decimals, ctype, quantizer))
    
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    
    assert currency.decimals == decimals
    assert currency.quantizer == quantizer


# LLM-generated content at query #6
#--------------------------

```python
def test___getitem___returns_currency_for_valid_code():
    registry = CurrencyRegistry()
    currency = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    registry.__register(currency)
    assert registry["USD"] == currency

def test___getitem___raises_lookup_error_for_invalid_code():
    registry = CurrencyRegistry()
    try:
        registry["NON-EXISTING"]
        assert False, "Expected CurrencyLookupError"
    except CurrencyLookupError:
        assert True


# LLM-generated content at query #7
#--------------------------

```python
def test_currency_codes_buffer_is_empty_after_init():
    registry = CurrencyRegistry()
    assert not registry._CurrencyRegistry__codes


# LLM-generated content at query #8
#--------------------------

```python
def test_ctx_open_initialized_to_false():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open is False


# LLM-generated content at query #9
#--------------------------

```python
def test_CurrencyRegistry_constructor_initializes_attributes():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__registry == {}
    assert registry._CurrencyRegistry__currencies == []
    assert registry._CurrencyRegistry__codes == []
    assert registry._CurrencyRegistry__codenames == []
    assert registry._CurrencyRegistry__ctx_open is False

def test_CurrencyRegistry_constructor_returns_singleton_instance():
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2


# LLM-generated content at query #10
#--------------------------

```python
def test_eq_returns_true_when_currencies_are_identical():
    c1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    c2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert c1 == c2

def test_eq_returns_false_when_currencies_have_different_names():
    c1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    c2 = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert not (c1 == c2)

def test_eq_returns_false_when_currencies_have_different_codes():
    c1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    c2 = Currency.of("EUR", "US Dollars", 2, CurrencyType.MONEY)
    assert not (c1 == c2)

def test_eq_returns_false_when_currencies_have_different_decimals():
    c1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    c2 = Currency.of("USD", "US Dollars", 0, CurrencyType.MONEY)
    assert not (c1 == c2)

def test_eq_returns_false_when_currencies_have_different_types():
    c1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    c2 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert not (c1 == c2)

def test_eq_returns_false_when_comparing_with_non_currency_object():
    c1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (c1 == "USD")


# LLM-generated content at query #11
#--------------------------

```
def test___eq___with_same_currency_objects():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2

def test___eq___with_different_currency_objects():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert not (usd == jpy)

def test___eq___with_non_currency_object():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd == "USD")

def test___eq___with_same_code_but_different_name():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 == usd2)

def test___eq___with_same_code_and_name_but_different_decimals():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 0, CurrencyType.MONEY)
    assert not (usd1 == usd2)

def test___eq___with_same_code_name_decimals_but_different_type():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert not (usd1 == usd2)


# LLM-generated content at query #12
#--------------------------

```python
def test_currency_equality_predicate_evaluates_to_true():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert isinstance(usd2, Currency) and usd1.hashcache == usd2.hashcache


# LLM-generated content at query #13
#--------------------------

```python
def test_currency_eq_same_instance():
    currency1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    currency2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert currency1 == currency2

def test_currency_eq_different_instance():
    currency1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    currency2 = Currency.of("EUR", "Euros", 2, CurrencyType.MONEY)
    assert not (currency1 == currency2)

def test_currency_eq_same_code_different_name():
    currency1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    currency2 = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert not (currency1 == currency2)

def test_currency_eq_same_code_same_name_different_decimals():
    currency1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    currency2 = Currency.of("USD", "US Dollars", 0, CurrencyType.MONEY)
    assert not (currency1 == currency2)

def test_currency_eq_same_code_same_name_same_decimals_different_type():
    currency1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    currency2 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert not (currency1 == currency2)

def test_currency_eq_with_non_currency_object():
    currency = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    non_currency = "USD"
    assert not (currency == non_currency)


# LLM-generated content at query #14
#--------------------------

```python
def test_currency_registry_initialization():
    registry = CurrencyRegistry()
    assert len(registry._CurrencyRegistry__codes) == 0


# LLM-generated content at query #15
#--------------------------

```python
def test_registry_population_context_flag_initialized_to_false():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open is False


# LLM-generated content at query #16
#--------------------------

```python
def test_currency_constructor():
    currency = Currency("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal("0.01"), hash(("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal("0.01"))))
    assert currency.code == "USD"
    assert currency.name == "US Dollars"
    assert currency.decimals == 2
    assert currency.type == CurrencyType.MONEY
    assert currency.quantizer == Decimal("0.01")
    assert currency.hashcache == hash(("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal("0.01")))


# LLM-generated content at query #17
#--------------------------

```python
def test_currency_constructor_with_valid_arguments():
    ctype = CurrencyType.MONEY
    currency = Currency.of("USD", "US Dollars", 2, ctype)
    assert currency.code == "USD"
    assert currency.name == "US Dollars"
    assert currency.decimals == 2
    assert currency.type == ctype
    assert currency.quantizer == Decimal('0.01')
    assert currency.hashcache == hash(("USD", "US Dollars", 2, ctype, Decimal('0.01')))

def test_currency_constructor_with_zero_decimals():
    ctype = CurrencyType.MONEY
    currency = Currency.of("JPY", "Japanese Yen", 0, ctype)
    assert currency.code == "JPY"
    assert currency.name == "Japanese Yen"
    assert currency.decimals == 0
    assert currency.type == ctype
    assert currency.quantizer == Decimal('0')
    assert currency.hashcache == hash(("JPY", "Japanese Yen", 0, ctype, Decimal('0')))

def test_currency_constructor_with_negative_decimals():
    ctype = CurrencyType.CRYPTO
    currency = Currency.of("ZZZ", "Some weird currency", -1, ctype)
    assert currency.code == "ZZZ"
    assert currency.name == "Some weird currency"
    assert currency.decimals == -1
    assert currency.type == ctype
    assert currency.quantizer == Decimal('0.000000000001')
    assert currency.hashcache == hash(("ZZZ", "Some weird currency", -1, ctype, Decimal('0.000000000001')))

def test_currency_constructor_with_different_currency_type():
    ctype = CurrencyType.CRYPTO
    currency = Currency.of("BTC", "Bitcoin", 8, ctype)
    assert currency.code == "BTC"
    assert currency.name == "Bitcoin"
    assert currency.decimals == 8
    assert currency.type == ctype
    assert currency.quantizer == Decimal('0.00000001')
    assert currency.hashcache == hash(("BTC", "Bitcoin", 8, ctype, Decimal('0.00000001')))


# LLM-generated content at query #18
#--------------------------

```python
def test_initial_codes_empty():
    registry = CurrencyRegistry()
    assert not registry.codes


# LLM-generated content at query #19
#--------------------------

```python
def test_currency_registry_init_codes_empty():
    registry = CurrencyRegistry()
    assert not registry.codes


# LLM-generated content at query #20
#--------------------------

```python
def test_registry_context_flag_initialized_to_false():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open is False


# LLM-generated content at query #21
#--------------------------

```python
def test_currency_constructor():
    currency = Currency(code="USD", name="US Dollars", decimals=2, type=CurrencyType.MONEY, quantizer=Decimal('0.01'), hashcache=hash(("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal('0.01'))))
    assert currency.code == "USD"
    assert currency.name == "US Dollars"
    assert currency.decimals == 2
    assert currency.type == CurrencyType.MONEY
    assert currency.quantizer == Decimal('0.01')
    assert currency.hashcache == hash(("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal('0.01')))


# LLM-generated content at query #22
#--------------------------

```python
def test_initial_context_is_closed():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open is False


# LLM-generated content at query #23
#--------------------------

```python
def test_currency_registry_constructor():
    registry = CurrencyRegistry()
    assert registry is CurrencyRegistry()
    assert registry._CurrencyRegistry__registry == {}
    assert registry._CurrencyRegistry__currencies == []
    assert registry._CurrencyRegistry__codes == []
    assert registry._CurrencyRegistry__codenames == []
    assert registry._CurrencyRegistry__ctx_open is False


# LLM-generated content at query #24
#--------------------------

```python
def test_currency_constructor():
    currency = Currency("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal("0.01"), hash(("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal("0.01"))))
    assert currency.code == "USD"
    assert currency.name == "US Dollars"
    assert currency.decimals == 2
    assert currency.type == CurrencyType.MONEY
    assert currency.quantizer == Decimal("0.01")
    assert currency.hashcache == hash(("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal("0.01")))


# LLM-generated content at query #25
#--------------------------

```python
def test_constructor_initializes_registry():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__registry == OrderedDict([])
    assert registry._CurrencyRegistry__currencies == []
    assert registry._CurrencyRegistry__codes == []
    assert registry._CurrencyRegistry__codenames == []
    assert registry._CurrencyRegistry__ctx_open is False


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_constructor_creates_singleton():
    instance1 = CurrencyRegistry()
    instance2 = CurrencyRegistry()
    assert instance1 is instance2

def test_constructor_initializes_registry():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__registry == {}
    assert registry._CurrencyRegistry__currencies == []
    assert registry._CurrencyRegistry__codes == []
    assert registry._CurrencyRegistry__codenames == []
    assert registry._CurrencyRegistry__ctx_open == False


# LLM-generated content at query #2
#--------------------------

def test_ctx_open_initialized_to_false():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open == False


# LLM-generated content at query #3
#--------------------------

def test_currency_of_creates_valid_instance():
    ccy = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert ccy.code == "USD"
    assert ccy.name == "US Dollars"
    assert ccy.decimals == 2
    assert ccy.type == CurrencyType.MONEY
    assert ccy.quantizer == make_quantizer(2)
    assert ccy.hashcache == hash(("USD", "US Dollars", 2, CurrencyType.MONEY, make_quantizer(2)))

def test_currency_of_raises_on_non_string_code():
    try:
        Currency.of(123, "US Dollars", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_on_non_alphabetic_code():
    try:
        Currency.of("US1", "US Dollars", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_on_non_uppercase_code():
    try:
        Currency.of("usd", "US Dollars", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_on_non_string_name():
    try:
        Currency.of("USD", 123, 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_on_empty_name():
    try:
        Currency.of("USD", "", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_on_name_with_leading_space():
    try:
        Currency.of("USD", " US Dollars", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_on_name_with_trailing_space():
    try:
        Currency.of("USD", "US Dollars ", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_on_non_integer_decimals():
    try:
        Currency.of("USD", "US Dollars", "2", CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_on_decimals_less_than_minus_one():
    try:
        Currency.of("USD", "US Dollars", -2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_on_non_currency_type():
    try:
        Currency.of("USD", "US Dollars", 2, "MONEY")
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_with_zero_decimals():
    ccy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert ccy.decimals == 0
    assert ccy.quantizer == ZERO

def test_currency_of_with_negative_decimals():
    ccy = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert ccy.decimals == -1
    assert ccy.quantizer == MaxPrecisionQuantizer

def test_currency_equality():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2
    assert not (usd1 == usdx)

def test_currency_hash_equality():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert hash(usd1) == hash(usd2)
    assert hash(usd1) != hash(usdx)

def test_currency_quantize_positive_decimals():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.quantize(Decimal("1.005")) == Decimal("1.00")
    assert usd.quantize(Decimal("1.015")) == Decimal("1.02")

def test_currency_quantize_zero_decimals():
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy.quantize(Decimal("0.5")) == Decimal("0")
    assert jpy.quantize(Decimal("1.5")) == Decimal("2")

def test_currency_quantize_negative_decimals():
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert zzz.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert zzz.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")


# LLM-generated content at query #4
#--------------------------

def test_constructor_creates_singleton():
    instance1 = CurrencyRegistry()
    instance2 = CurrencyRegistry()
    assert instance1 is instance2

def test_constructor_initializes_registry():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__registry == {}
    assert registry._CurrencyRegistry__currencies == []
    assert registry._CurrencyRegistry__codes == []
    assert registry._CurrencyRegistry__codenames == []
    assert registry._CurrencyRegistry__ctx_open == False

def test_constructor_singleton_preserves_state():
    instance1 = CurrencyRegistry()
    instance1._CurrencyRegistry__ctx_open = True
    instance2 = CurrencyRegistry()
    assert instance2._CurrencyRegistry__ctx_open == True


# LLM-generated content at query #5
#--------------------------

def test_currency_equality_same_instance():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2

def test_currency_equality_different_code():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert not (usd == eur)

def test_currency_equality_different_name():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 == usdx)

def test_currency_equality_different_decimals():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert not (usd == jpy)

def test_currency_equality_different_type():
    money = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    crypto = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    assert not (money == crypto)

def test_currency_equality_with_non_currency():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd == "USD")

def test_currency_equality_same_hash():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1.hashcache == usd2.hashcache
    assert usd1 == usd2

def test_currency_equality_different_hash():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert usd1.hashcache != usdx.hashcache
    assert not (usd1 == usdx)


# LLM-generated content at query #6
#--------------------------

def test_ctx_open_initialized_to_false():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open == False


# LLM-generated content at query #7
#--------------------------

def test___getitem___returns_currency_for_valid_code():
    registry = CurrencyRegistry()
    with registry as register:
        currency = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        register(currency)
    result = registry["USD"]
    assert result.code == "USD"
    assert result.name == "US Dollar"
    assert result.type == CurrencyType.MONEY

def test___getitem___raises_CurrencyLookupError_for_invalid_code():
    registry = CurrencyRegistry()
    try:
        registry["NON-EXISTING"]
        assert False
    except CurrencyLookupError as e:
        assert str(e) == "Currency identified by code 'NON-EXISTING' does not exist"

def test___getitem___accesses_same_instance_as_registry():
    registry = CurrencyRegistry()
    with registry as register:
        currency = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
        register(currency)
    retrieved = registry["EUR"]
    assert retrieved is currency

def test___getitem___after_context_exit_still_works():
    registry = CurrencyRegistry()
    with registry as register:
        currency = Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY)
        register(currency)
    result = registry["GBP"]
    assert result.code == "GBP"

def test___getitem___case_sensitive():
    registry = CurrencyRegistry()
    with registry as register:
        currency = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
        register(currency)
    try:
        registry["usd"]
        assert False
    except CurrencyLookupError:
        assert True


# LLM-generated content at query #8
#--------------------------

def test_constructor_creates_singleton():
    instance1 = CurrencyRegistry()
    instance2 = CurrencyRegistry()
    assert instance1 is instance2

def test_constructor_initializes_registry():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__registry == {}
    assert registry._CurrencyRegistry__currencies == []
    assert registry._CurrencyRegistry__codes == []
    assert registry._CurrencyRegistry__codenames == []
    assert registry._CurrencyRegistry__ctx_open == False

def test_constructor_singleton_preserves_state():
    registry1 = CurrencyRegistry()
    registry1._CurrencyRegistry__ctx_open = True
    registry2 = CurrencyRegistry()
    assert registry2._CurrencyRegistry__ctx_open == True


# LLM-generated content at query #9
#--------------------------

def test_constructor_creates_singleton():
    instance1 = CurrencyRegistry()
    instance2 = CurrencyRegistry()
    assert instance1 is instance2

def test_constructor_initializes_registry():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__registry == {}
    assert registry._CurrencyRegistry__currencies == []
    assert registry._CurrencyRegistry__codes == []
    assert registry._CurrencyRegistry__codenames == []
    assert registry._CurrencyRegistry__ctx_open == False

def test_constructor_singleton_preserves_state():
    instance1 = CurrencyRegistry()
    instance1._CurrencyRegistry__ctx_open = True
    instance2 = CurrencyRegistry()
    assert instance2._CurrencyRegistry__ctx_open == True


# LLM-generated content at query #10
#--------------------------

def test_constructor_creates_singleton():
    instance1 = CurrencyRegistry()
    instance2 = CurrencyRegistry()
    assert instance1 is instance2

def test_constructor_initializes_registry():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__registry == {}
    assert registry._CurrencyRegistry__currencies == []
    assert registry._CurrencyRegistry__codes == []
    assert registry._CurrencyRegistry__codenames == []
    assert registry._CurrencyRegistry__ctx_open is False

def test_constructor_singleton_preserves_state():
    instance1 = CurrencyRegistry()
    instance1._CurrencyRegistry__ctx_open = True
    instance2 = CurrencyRegistry()
    assert instance2._CurrencyRegistry__ctx_open is True


# LLM-generated content at query #11
#--------------------------

def test_constructor_creates_singleton():
    instance1 = CurrencyRegistry()
    instance2 = CurrencyRegistry()
    assert instance1 is instance2

def test_constructor_initializes_registry():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__registry == {}
    assert registry._CurrencyRegistry__currencies == []
    assert registry._CurrencyRegistry__codes == []
    assert registry._CurrencyRegistry__codenames == []
    assert registry._CurrencyRegistry__ctx_open == False

def test_constructor_singleton_persistence():
    CurrencyRegistry._CurrencyRegistry__instance = None
    first_instance = CurrencyRegistry()
    second_instance = CurrencyRegistry()
    assert first_instance is second_instance
    assert CurrencyRegistry._CurrencyRegistry__instance is first_instance


# LLM-generated content at query #12
#--------------------------

def test_constructor_creates_singleton():
    instance1 = CurrencyRegistry()
    instance2 = CurrencyRegistry()
    assert instance1 is instance2

def test_constructor_initializes_registry():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__registry == {}
    assert registry._CurrencyRegistry__currencies == []
    assert registry._CurrencyRegistry__codes == []
    assert registry._CurrencyRegistry__codenames == []
    assert registry._CurrencyRegistry__ctx_open == False

def test_constructor_singleton_preserves_state():
    registry1 = CurrencyRegistry()
    registry1._CurrencyRegistry__ctx_open = True
    registry2 = CurrencyRegistry()
    assert registry2._CurrencyRegistry__ctx_open == True


# LLM-generated content at query #13
#--------------------------

def test___getitem___returns_currency_for_valid_code():
    with Currencies as register:
        register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
    result = Currencies["USD"]
    assert result.code == "USD"

def test___getitem___raises_currencylookuperror_for_invalid_code():
    with Currencies as register:
        register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
    try:
        _ = Currencies["XXX"]
        assert False
    except CurrencyLookupError:
        assert True

def test___getitem___accesses_same_instance_as_get():
    with Currencies as register:
        register(Currency.of("EUR", "Euro", 2, CurrencyType.MONEY))
    assert Currencies["EUR"] == Currencies.get("EUR")

def test___getitem___after_context_exit_still_works():
    with Currencies as register:
        register(Currency.of("GBP", "British Pound", 2, CurrencyType.MONEY))
    result = Currencies["GBP"]
    assert result.code == "GBP"

def test___getitem___is_case_sensitive():
    with Currencies as register:
        register(Currency.of("usd", "US Dollar Lower", 2, CurrencyType.MONEY))
    result_lower = Currencies["usd"]
    assert result_lower.code == "usd"
    try:
        _ = Currencies["USD"]
        assert False
    except CurrencyLookupError:
        assert True


# LLM-generated content at query #14
#--------------------------

def test_ctx_open_initialized_to_false():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open == False


# LLM-generated content at query #15
#--------------------------

def test_currency_of_creates_valid_instance():
    currency = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert currency.code == "USD"
    assert currency.name == "US Dollars"
    assert currency.decimals == 2
    assert currency.type == CurrencyType.MONEY
    assert currency.quantizer == make_quantizer(2)
    assert currency.hashcache == hash(("USD", "US Dollars", 2, CurrencyType.MONEY, make_quantizer(2)))

def test_currency_of_with_zero_decimals():
    currency = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert currency.code == "JPY"
    assert currency.name == "Japanese Yen"
    assert currency.decimals == 0
    assert currency.type == CurrencyType.MONEY
    assert currency.quantizer == ZERO

def test_currency_of_with_negative_decimals():
    currency = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert currency.code == "ZZZ"
    assert currency.name == "Some weird currency"
    assert currency.decimals == -1
    assert currency.type == CurrencyType.CRYPTO
    assert currency.quantizer == MaxPrecisionQuantizer

def test_currency_of_raises_on_non_string_code():
    try:
        Currency.of(123, "US Dollars", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_on_non_alphabetic_code():
    try:
        Currency.of("US1", "US Dollars", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_on_non_uppercase_code():
    try:
        Currency.of("usd", "US Dollars", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_on_non_string_name():
    try:
        Currency.of("USD", 123, 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_on_empty_name():
    try:
        Currency.of("USD", "", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_on_name_with_leading_space():
    try:
        Currency.of("USD", " US Dollars", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_on_name_with_trailing_space():
    try:
        Currency.of("USD", "US Dollars ", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_on_non_integer_decimals():
    try:
        Currency.of("USD", "US Dollars", "2", CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_on_decimals_less_than_minus_one():
    try:
        Currency.of("USD", "US Dollars", -2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_on_non_currency_type():
    try:
        Currency.of("USD", "US Dollars", 2, "MONEY")
        assert False
    except ProgrammingError:
        assert True

def test_currency_equality():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2
    assert not (usd1 == usdx)

def test_currency_hash():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert hash(usd1) == hash(usd2)
    assert hash(usd1) != hash(usdx)

def test_currency_quantize_positive_decimals():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.quantize(Decimal("1.005")) == Decimal("1.00")
    assert usd.quantize(Decimal("1.015")) == Decimal("1.02")

def test_currency_quantize_zero_decimals():
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy.quantize(Decimal("0.5")) == Decimal("0")
    assert jpy.quantize(Decimal("1.5")) == Decimal("2")

def test_currency_quantize_negative_decimals():
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert zzz.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert zzz.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")


# LLM-generated content at query #16
#--------------------------

def test_currency_of_creates_valid_instance():
    currency = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert currency.code == "USD"
    assert currency.name == "US Dollars"
    assert currency.decimals == 2
    assert currency.type == CurrencyType.MONEY
    assert currency.quantizer == make_quantizer(2)
    assert currency.hashcache == hash(("USD", "US Dollars", 2, CurrencyType.MONEY, make_quantizer(2)))

def test_currency_of_with_zero_decimals():
    currency = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert currency.code == "JPY"
    assert currency.name == "Japanese Yen"
    assert currency.decimals == 0
    assert currency.type == CurrencyType.MONEY
    assert currency.quantizer == ZERO

def test_currency_of_with_negative_decimals():
    currency = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert currency.code == "ZZZ"
    assert currency.name == "Some weird currency"
    assert currency.decimals == -1
    assert currency.type == CurrencyType.CRYPTO
    assert currency.quantizer == MaxPrecisionQuantizer

def test_currency_of_raises_on_invalid_code_type():
    try:
        Currency.of(123, "US Dollars", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_on_non_alpha_code():
    try:
        Currency.of("US1", "US Dollars", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_on_non_uppercase_code():
    try:
        Currency.of("usd", "US Dollars", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_on_invalid_name_type():
    try:
        Currency.of("USD", 123, 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_on_empty_name():
    try:
        Currency.of("USD", "", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_on_name_with_leading_space():
    try:
        Currency.of("USD", " US Dollars", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_on_name_with_trailing_space():
    try:
        Currency.of("USD", "US Dollars ", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_on_invalid_decimals_type():
    try:
        Currency.of("USD", "US Dollars", "2", CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_on_decimals_less_than_minus_one():
    try:
        Currency.of("USD", "US Dollars", -2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_on_invalid_currency_type():
    try:
        Currency.of("USD", "US Dollars", 2, "MONEY")
        assert False
    except ProgrammingError:
        assert True

def test_currency_equality():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2
    assert not (usd1 == usdx)

def test_currency_hash():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert hash(usd1) == hash(usd2)
    assert hash(usd1) != hash(usdx)

def test_currency_quantize_with_positive_decimals():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.quantize(Decimal("1.005")) == Decimal("1.00")
    assert usd.quantize(Decimal("1.015")) == Decimal("1.02")

def test_currency_quantize_with_zero_decimals():
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy.quantize(Decimal("0.5")) == Decimal("0")
    assert jpy.quantize(Decimal("1.5")) == Decimal("2")

def test_currency_quantize_with_negative_decimals():
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert zzz.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert zzz.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")


# LLM-generated content at query #17
#--------------------------

def test_currency_of_creates_valid_currency():
    ccy = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert ccy.code == "USD"
    assert ccy.name == "US Dollars"
    assert ccy.decimals == 2
    assert ccy.type == CurrencyType.MONEY
    assert ccy.quantizer == make_quantizer(2)
    assert ccy.hashcache == hash(("USD", "US Dollars", 2, CurrencyType.MONEY, make_quantizer(2)))

def test_currency_of_creates_currency_with_zero_decimals():
    ccy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert ccy.code == "JPY"
    assert ccy.name == "Japanese Yen"
    assert ccy.decimals == 0
    assert ccy.type == CurrencyType.MONEY
    assert ccy.quantizer == ZERO

def test_currency_of_creates_currency_with_negative_decimals():
    ccy = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert ccy.code == "ZZZ"
    assert ccy.name == "Some weird currency"
    assert ccy.decimals == -1
    assert ccy.type == CurrencyType.CRYPTO
    assert ccy.quantizer == MaxPrecisionQuantizer

def test_currency_of_raises_error_for_non_string_code():
    try:
        Currency.of(123, "US Dollars", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_error_for_non_alphabetic_code():
    try:
        Currency.of("US1", "US Dollars", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_error_for_non_uppercase_code():
    try:
        Currency.of("usd", "US Dollars", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_error_for_non_string_name():
    try:
        Currency.of("USD", 123, 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_error_for_empty_name():
    try:
        Currency.of("USD", "", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_error_for_name_with_leading_space():
    try:
        Currency.of("USD", " US Dollars", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_error_for_name_with_trailing_space():
    try:
        Currency.of("USD", "US Dollars ", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_error_for_non_integer_decimals():
    try:
        Currency.of("USD", "US Dollars", "2", CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_error_for_decimals_less_than_minus_one():
    try:
        Currency.of("USD", "US Dollars", -2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_error_for_non_currencytype_type():
    try:
        Currency.of("USD", "US Dollars", 2, "MONEY")
        assert False
    except ProgrammingError:
        assert True

def test_currency_equality_based_on_hashcache():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2
    assert not (usd1 == usdx)

def test_currency_hash_based_on_hashcache():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert hash(usd1) == hash(usd2)
    assert hash(usd1) != hash(usdx)

def test_currency_quantize_with_positive_decimals():
    ccy = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    result = ccy.quantize(Decimal("1.005"))
    assert result == Decimal("1.00")
    result = ccy.quantize(Decimal("1.015"))
    assert result == Decimal("1.02")

def test_currency_quantize_with_zero_decimals():
    ccy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    result = ccy.quantize(Decimal("0.5"))
    assert result == Decimal("0")
    result = ccy.quantize(Decimal("1.5"))
    assert result == Decimal("2")

def test_currency_quantize_with_negative_decimals():
    ccy = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    result = ccy.quantize(Decimal("1.0000000000005"))
    assert result == Decimal("1.000000000000")
    result = ccy.quantize(Decimal("1.0000000000015"))
    assert result == Decimal("1.000000000002")


# LLM-generated content at query #18
#--------------------------

def test_ctx_open_initialized_to_false():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open == False


# LLM-generated content at query #19
#--------------------------

def test_constructor_creates_singleton():
    instance1 = CurrencyRegistry()
    instance2 = CurrencyRegistry()
    assert instance1 is instance2

def test_constructor_initializes_registry():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__registry == {}
    assert registry._CurrencyRegistry__currencies == []
    assert registry._CurrencyRegistry__codes == []
    assert registry._CurrencyRegistry__codenames == []
    assert registry._CurrencyRegistry__ctx_open == False

def test_constructor_singleton_preserves_state():
    instance1 = CurrencyRegistry()
    instance1._CurrencyRegistry__ctx_open = True
    instance2 = CurrencyRegistry()
    assert instance2._CurrencyRegistry__ctx_open == True


# LLM-generated content at query #20
#--------------------------

def test_constructor_creates_singleton():
    instance1 = CurrencyRegistry()
    instance2 = CurrencyRegistry()
    assert instance1 is instance2

def test_constructor_initializes_registry():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__registry == {}
    assert registry._CurrencyRegistry__currencies == []
    assert registry._CurrencyRegistry__codes == []
    assert registry._CurrencyRegistry__codenames == []
    assert registry._CurrencyRegistry__ctx_open == False

def test_constructor_singleton_preserves_state():
    registry1 = CurrencyRegistry()
    registry1._CurrencyRegistry__ctx_open = True
    registry2 = CurrencyRegistry()
    assert registry2._CurrencyRegistry__ctx_open == True


# LLM-generated content at query #21
#--------------------------

def test_constructor_creates_singleton():
    instance1 = CurrencyRegistry()
    instance2 = CurrencyRegistry()
    assert instance1 is instance2

def test_constructor_initializes_registry():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__registry == {}
    assert registry._CurrencyRegistry__currencies == []
    assert registry._CurrencyRegistry__codes == []
    assert registry._CurrencyRegistry__codenames == []
    assert registry._CurrencyRegistry__ctx_open == False

def test_constructor_singleton_preserves_state():
    instance1 = CurrencyRegistry()
    instance1._CurrencyRegistry__ctx_open = True
    instance2 = CurrencyRegistry()
    assert instance2._CurrencyRegistry__ctx_open == True


# LLM-generated content at query #22
#--------------------------

def test_constructor_creates_singleton():
    instance1 = CurrencyRegistry()
    instance2 = CurrencyRegistry()
    assert instance1 is instance2

def test_constructor_initializes_registry():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__registry == {}
    assert registry._CurrencyRegistry__currencies == []
    assert registry._CurrencyRegistry__codes == []
    assert registry._CurrencyRegistry__codenames == []
    assert registry._CurrencyRegistry__ctx_open == False


# LLM-generated content at query #23
#--------------------------

def test_constructor_creates_singleton():
    instance1 = CurrencyRegistry()
    instance2 = CurrencyRegistry()
    assert instance1 is instance2

def test_constructor_initializes_registry():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__registry == {}
    assert registry._CurrencyRegistry__currencies == []
    assert registry._CurrencyRegistry__codes == []
    assert registry._CurrencyRegistry__codenames == []
    assert registry._CurrencyRegistry__ctx_open == False


# LLM-generated content at query #24
#--------------------------

def test_constructor_creates_singleton():
    instance1 = CurrencyRegistry()
    instance2 = CurrencyRegistry()
    assert instance1 is instance2

def test_constructor_initializes_registry():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__registry == {}
    assert registry._CurrencyRegistry__currencies == []
    assert registry._CurrencyRegistry__codes == []
    assert registry._CurrencyRegistry__codenames == []
    assert registry._CurrencyRegistry__ctx_open == False

def test_constructor_singleton_persistence():
    instance1 = CurrencyRegistry()
    instance1._CurrencyRegistry__ctx_open = True
    instance2 = CurrencyRegistry()
    assert instance2._CurrencyRegistry__ctx_open == True


# LLM-generated content at query #25
#--------------------------

def test_constructor_creates_singleton():
    instance1 = CurrencyRegistry()
    instance2 = CurrencyRegistry()
    assert instance1 is instance2

def test_constructor_initializes_registry():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__registry == {}
    assert registry._CurrencyRegistry__currencies == []
    assert registry._CurrencyRegistry__codes == []
    assert registry._CurrencyRegistry__codenames == []
    assert registry._CurrencyRegistry__ctx_open == False

def test_constructor_singleton_preserves_state():
    instance1 = CurrencyRegistry()
    instance1._CurrencyRegistry__ctx_open = True
    instance2 = CurrencyRegistry()
    assert instance2._CurrencyRegistry__ctx_open == True


# LLM-generated content at query #26
#--------------------------

def test_constructor_creates_singleton():
    instance1 = CurrencyRegistry()
    instance2 = CurrencyRegistry()
    assert instance1 is instance2

def test_constructor_initializes_registry():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__registry == {}
    assert registry._CurrencyRegistry__currencies == []
    assert registry._CurrencyRegistry__codes == []
    assert registry._CurrencyRegistry__codenames == []
    assert registry._CurrencyRegistry__ctx_open == False

def test_constructor_singleton_preserves_state():
    instance1 = CurrencyRegistry()
    instance1._CurrencyRegistry__ctx_open = True
    instance2 = CurrencyRegistry()
    assert instance2._CurrencyRegistry__ctx_open == True


# LLM-generated content at query #27
#--------------------------

def test_constructor_creates_singleton():
    instance1 = CurrencyRegistry()
    instance2 = CurrencyRegistry()
    assert instance1 is instance2

def test_constructor_initializes_registry():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__registry == {}
    assert registry._CurrencyRegistry__currencies == []
    assert registry._CurrencyRegistry__codes == []
    assert registry._CurrencyRegistry__codenames == []
    assert registry._CurrencyRegistry__ctx_open == False

def test_constructor_singleton_preserves_state():
    registry1 = CurrencyRegistry()
    registry1._CurrencyRegistry__ctx_open = True
    registry2 = CurrencyRegistry()
    assert registry2._CurrencyRegistry__ctx_open == True


# LLM-generated content at query #28
#--------------------------

def test_constructor_creates_singleton():
    instance1 = CurrencyRegistry()
    instance2 = CurrencyRegistry()
    assert instance1 is instance2

def test_constructor_initializes_registry():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__registry == {}
    assert registry._CurrencyRegistry__currencies == []
    assert registry._CurrencyRegistry__codes == []
    assert registry._CurrencyRegistry__codenames == []
    assert registry._CurrencyRegistry__ctx_open == False

def test_constructor_singleton_preserves_state():
    registry1 = CurrencyRegistry()
    registry1._CurrencyRegistry__ctx_open = True
    registry2 = CurrencyRegistry()
    assert registry2._CurrencyRegistry__ctx_open == True


# LLM-generated content at query #29
#--------------------------

def test_constructor_creates_singleton():
    instance1 = CurrencyRegistry()
    instance2 = CurrencyRegistry()
    assert instance1 is instance2

def test_constructor_initializes_registry():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__registry == {}
    assert registry._CurrencyRegistry__currencies == []
    assert registry._CurrencyRegistry__codes == []
    assert registry._CurrencyRegistry__codenames == []
    assert registry._CurrencyRegistry__ctx_open == False


# LLM-generated content at query #30
#--------------------------

def test_constructor_creates_singleton_instance():
    instance1 = CurrencyRegistry()
    instance2 = CurrencyRegistry()
    assert instance1 is instance2

def test_constructor_initializes_registry_as_ordered_dict():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__registry is not None

def test_constructor_initializes_currencies_as_empty_list():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__currencies == []

def test_constructor_initializes_codes_as_empty_list():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__codes == []

def test_constructor_initializes_codenames_as_empty_list():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__codenames == []

def test_constructor_initializes_ctx_open_as_false():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open is False

def test_constructor_sets_same_registry_for_multiple_calls():
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1._CurrencyRegistry__registry is registry2._CurrencyRegistry__registry

def test_constructor_sets_same_currencies_for_multiple_calls():
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1._CurrencyRegistry__currencies is registry2._CurrencyRegistry__currencies

def test_constructor_sets_same_codes_for_multiple_calls():
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1._CurrencyRegistry__codes is registry2._CurrencyRegistry__codes

def test_constructor_sets_same_codenames_for_multiple_calls():
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1._CurrencyRegistry__codenames is registry2._CurrencyRegistry__codenames

def test_constructor_sets_same_ctx_open_for_multiple_calls():
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1._CurrencyRegistry__ctx_open is registry2._CurrencyRegistry__ctx_open


# LLM-generated content at query #31
#--------------------------

def test_currency_of_valid_arguments():
    ctype = CurrencyType.MONEY
    currency = Currency.of("USD", "US Dollars", 2, ctype)
    assert currency.code == "USD"
    assert currency.name == "US Dollars"
    assert currency.decimals == 2
    assert currency.type == ctype
    assert currency.quantizer == make_quantizer(2)
    assert currency.hashcache == hash(("USD", "US Dollars", 2, ctype, make_quantizer(2)))

def test_currency_of_code_not_string():
    try:
        Currency.of(123, "US Dollars", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_code_not_alphabetic():
    try:
        Currency.of("US1", "US Dollars", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_code_not_uppercase():
    try:
        Currency.of("usd", "US Dollars", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_name_not_string():
    try:
        Currency.of("USD", 123, 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_name_empty():
    try:
        Currency.of("USD", "", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_name_has_leading_space():
    try:
        Currency.of("USD", " US Dollars", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_name_has_trailing_space():
    try:
        Currency.of("USD", "US Dollars ", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_decimals_not_integer():
    try:
        Currency.of("USD", "US Dollars", 2.5, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_decimals_less_than_minus_one():
    try:
        Currency.of("USD", "US Dollars", -2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_type_not_currency_type():
    try:
        Currency.of("USD", "US Dollars", 2, "MONEY")
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_decimals_zero_quantizer_zero():
    currency = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert currency.quantizer == ZERO

def test_currency_of_decimals_negative_quantizer_max_precision():
    currency = Currency.of("ZZZ", "Weird Crypto", -1, CurrencyType.CRYPTO)
    assert currency.quantizer == MaxPrecisionQuantizer

def test_currency_of_decimals_positive_quantizer_custom():
    currency = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert currency.quantizer == make_quantizer(2)

def test_currency_equality_same():
    ctype = CurrencyType.MONEY
    usd1 = Currency.of("USD", "US Dollars", 2, ctype)
    usd2 = Currency.of("USD", "US Dollars", 2, ctype)
    assert usd1 == usd2
    assert hash(usd1) == hash(usd2)

def test_currency_equality_different_name():
    ctype = CurrencyType.MONEY
    usd1 = Currency.of("USD", "US Dollars", 2, ctype)
    usdx = Currency.of("USD", "UX Dollars", 2, ctype)
    assert usd1 != usdx
    assert hash(usd1) != hash(usdx)

def test_currency_quantize_positive_decimals():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.quantize(Decimal("1.005")) == Decimal("1.00")
    assert usd.quantize(Decimal("1.015")) == Decimal("1.02")

def test_currency_quantize_zero_decimals():
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy.quantize(Decimal("0.5")) == Decimal("0")
    assert jpy.quantize(Decimal("1.5")) == Decimal("2")

def test_currency_quantize_negative_decimals():
    zzz = Currency.of("ZZZ", "Weird Crypto", -1, CurrencyType.CRYPTO)
    assert zzz.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert zzz.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")


# LLM-generated content at query #32
#--------------------------

def test_constructor_creates_singleton_instance():
    instance1 = CurrencyRegistry()
    instance2 = CurrencyRegistry()
    assert instance1 is instance2

def test_constructor_initializes_registry_as_ordered_dict():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__registry is not None

def test_constructor_initializes_currencies_as_empty_list():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__currencies == []

def test_constructor_initializes_codes_as_empty_list():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__codes == []

def test_constructor_initializes_codenames_as_empty_list():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__codenames == []

def test_constructor_initializes_ctx_open_as_false():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open is False


# LLM-generated content at query #33
#--------------------------

def test_constructor_creates_singleton():
    instance1 = CurrencyRegistry()
    instance2 = CurrencyRegistry()
    assert instance1 is instance2

def test_constructor_initializes_registry():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__registry == {}
    assert registry._CurrencyRegistry__currencies == []
    assert registry._CurrencyRegistry__codes == []
    assert registry._CurrencyRegistry__codenames == []
    assert registry._CurrencyRegistry__ctx_open == False

def test_constructor_singleton_preserves_state():
    instance1 = CurrencyRegistry()
    instance1._CurrencyRegistry__ctx_open = True
    instance2 = CurrencyRegistry()
    assert instance2._CurrencyRegistry__ctx_open == True


# LLM-generated content at query #34
#--------------------------

def test_currency_registry_constructor_singleton():
    instance1 = CurrencyRegistry()
    instance2 = CurrencyRegistry()
    assert instance1 is instance2

def test_currency_registry_constructor_initial_state():
    registry = CurrencyRegistry()
    assert len(registry) == 0
    assert registry.all == []
    assert registry.codes == []
    assert registry.codenames == []

def test_currency_registry_constructor_context_closed():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open == False

def test_currency_registry_constructor_registry_empty():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__registry == {}

def test_currency_registry_constructor_currencies_empty():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__currencies == []

def test_currency_registry_constructor_codes_empty():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__codes == []

def test_currency_registry_constructor_codenames_empty():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__codenames == []


# LLM-generated content at query #35
#--------------------------

def test_currency_of_creates_valid_instance():
    currency = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert currency.code == "USD"
    assert currency.name == "US Dollars"
    assert currency.decimals == 2
    assert currency.type == CurrencyType.MONEY
    assert currency.quantizer == make_quantizer(2)
    assert currency.hashcache == hash(("USD", "US Dollars", 2, CurrencyType.MONEY, make_quantizer(2)))

def test_currency_of_creates_instance_with_zero_decimals():
    currency = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert currency.code == "JPY"
    assert currency.name == "Japanese Yen"
    assert currency.decimals == 0
    assert currency.type == CurrencyType.MONEY
    assert currency.quantizer == ZERO

def test_currency_of_creates_instance_with_negative_decimals():
    currency = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert currency.code == "ZZZ"
    assert currency.name == "Some weird currency"
    assert currency.decimals == -1
    assert currency.type == CurrencyType.CRYPTO
    assert currency.quantizer == MaxPrecisionQuantizer

def test_currency_of_raises_error_for_non_string_code():
    try:
        Currency.of(123, "US Dollars", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_error_for_non_alphabetic_code():
    try:
        Currency.of("US1", "US Dollars", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_error_for_non_uppercase_code():
    try:
        Currency.of("usd", "US Dollars", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_error_for_non_string_name():
    try:
        Currency.of("USD", 123, 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_error_for_empty_name():
    try:
        Currency.of("USD", "", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_error_for_name_with_leading_space():
    try:
        Currency.of("USD", " US Dollars", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_error_for_name_with_trailing_space():
    try:
        Currency.of("USD", "US Dollars ", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_error_for_non_integer_decimals():
    try:
        Currency.of("USD", "US Dollars", "2", CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_error_for_decimals_less_than_minus_one():
    try:
        Currency.of("USD", "US Dollars", -2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_error_for_non_currencytype_type():
    try:
        Currency.of("USD", "US Dollars", 2, "MONEY")
        assert False
    except ProgrammingError:
        assert True

def test_currency_equality():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2
    assert not (usd1 == usdx)

def test_currency_hash_equality():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert hash(usd1) == hash(usd2)
    assert hash(usd1) != hash(usdx)

def test_currency_quantize_with_positive_decimals():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.quantize(Decimal("1.005")) == Decimal("1.00")
    assert usd.quantize(Decimal("1.015")) == Decimal("1.02")

def test_currency_quantize_with_zero_decimals():
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy.quantize(Decimal("0.5")) == Decimal("0")
    assert jpy.quantize(Decimal("1.5")) == Decimal("2")

def test_currency_quantize_with_negative_decimals():
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert zzz.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert zzz.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")


# LLM-generated content at query #36
#--------------------------

def test_ctx_open_initialized_to_false():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open == False


# LLM-generated content at query #37
#--------------------------

def test_constructor_creates_singleton():
    instance1 = CurrencyRegistry()
    instance2 = CurrencyRegistry()
    assert instance1 is instance2

def test_constructor_initializes_registry():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__registry == {}
    assert registry._CurrencyRegistry__currencies == []
    assert registry._CurrencyRegistry__codes == []
    assert registry._CurrencyRegistry__codenames == []
    assert registry._CurrencyRegistry__ctx_open == False

def test_constructor_singleton_preserves_state():
    instance1 = CurrencyRegistry()
    instance1._CurrencyRegistry__ctx_open = True
    instance2 = CurrencyRegistry()
    assert instance2._CurrencyRegistry__ctx_open == True


# LLM-generated content at query #38
#--------------------------

def test_currency_registry_initialization():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__codes == []


# LLM-generated content at query #39
#--------------------------

def test_ctx_open_initialized_to_false():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open == False


# LLM-generated content at query #40
#--------------------------

def test_currency_of_creates_valid_instance():
    currency = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert currency.code == "USD"
    assert currency.name == "US Dollars"
    assert currency.decimals == 2
    assert currency.type == CurrencyType.MONEY
    assert currency.quantizer == make_quantizer(2)
    assert currency.hashcache == hash(("USD", "US Dollars", 2, CurrencyType.MONEY, make_quantizer(2)))

def test_currency_of_with_zero_decimals():
    currency = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert currency.code == "JPY"
    assert currency.name == "Japanese Yen"
    assert currency.decimals == 0
    assert currency.type == CurrencyType.MONEY
    assert currency.quantizer == ZERO

def test_currency_of_with_negative_decimals():
    currency = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert currency.code == "ZZZ"
    assert currency.name == "Some weird currency"
    assert currency.decimals == -1
    assert currency.type == CurrencyType.CRYPTO
    assert currency.quantizer == MaxPrecisionQuantizer

def test_currency_of_raises_error_for_non_string_code():
    try:
        Currency.of(123, "US Dollars", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_error_for_non_alphabetic_code():
    try:
        Currency.of("US1", "US Dollars", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_error_for_non_uppercase_code():
    try:
        Currency.of("usd", "US Dollars", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_error_for_non_string_name():
    try:
        Currency.of("USD", 123, 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_error_for_empty_name():
    try:
        Currency.of("USD", "", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_error_for_name_with_leading_space():
    try:
        Currency.of("USD", " US Dollars", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_error_for_name_with_trailing_space():
    try:
        Currency.of("USD", "US Dollars ", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_error_for_non_integer_decimals():
    try:
        Currency.of("USD", "US Dollars", "2", CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_error_for_decimals_less_than_minus_one():
    try:
        Currency.of("USD", "US Dollars", -2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_error_for_non_currencytype_type():
    try:
        Currency.of("USD", "US Dollars", 2, "MONEY")
        assert False
    except ProgrammingError:
        assert True

def test_currency_equality():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2
    assert not (usd1 == usdx)

def test_currency_hash_equality():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert hash(usd1) == hash(usd2)
    assert hash(usd1) != hash(usdx)

def test_currency_quantize_with_positive_decimals():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.quantize(Decimal("1.005")) == Decimal("1.00")
    assert usd.quantize(Decimal("1.015")) == Decimal("1.02")

def test_currency_quantize_with_zero_decimals():
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy.quantize(Decimal("0.5")) == Decimal("0")
    assert jpy.quantize(Decimal("1.5")) == Decimal("2")

def test_currency_quantize_with_negative_decimals():
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert zzz.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert zzz.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")


# LLM-generated content at query #41
#--------------------------

def test_constructor_creates_singleton():
    instance1 = CurrencyRegistry()
    instance2 = CurrencyRegistry()
    assert instance1 is instance2

def test_constructor_initializes_registry():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__registry == {}
    assert registry._CurrencyRegistry__currencies == []
    assert registry._CurrencyRegistry__codes == []
    assert registry._CurrencyRegistry__codenames == []
    assert registry._CurrencyRegistry__ctx_open == False

def test_constructor_singleton_persistence():
    CurrencyRegistry._CurrencyRegistry__instance = None
    instance1 = CurrencyRegistry()
    instance2 = CurrencyRegistry()
    assert instance1 is instance2
    assert CurrencyRegistry._CurrencyRegistry__instance is instance1

def test_constructor_reinitialization_does_not_override():
    CurrencyRegistry._CurrencyRegistry__instance = None
    instance = CurrencyRegistry()
    instance._CurrencyRegistry__ctx_open = True
    instance._CurrencyRegistry__registry = {"TEST": "dummy"}
    new_instance = CurrencyRegistry()
    assert new_instance._CurrencyRegistry__ctx_open == True
    assert new_instance._CurrencyRegistry__registry == {"TEST": "dummy"}


# LLM-generated content at query #42
#--------------------------

def test_ctx_open_initialized_to_false():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open == False


# LLM-generated content at query #43
#--------------------------

def test_currency_of_creates_valid_currency():
    c = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert c.code == "USD"
    assert c.name == "US Dollars"
    assert c.decimals == 2
    assert c.type == CurrencyType.MONEY
    assert c.quantizer == make_quantizer(2)
    assert c.hashcache == hash(("USD", "US Dollars", 2, CurrencyType.MONEY, make_quantizer(2)))

def test_currency_of_with_zero_decimals():
    c = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert c.code == "JPY"
    assert c.name == "Japanese Yen"
    assert c.decimals == 0
    assert c.type == CurrencyType.MONEY
    assert c.quantizer == ZERO

def test_currency_of_with_negative_decimals():
    c = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert c.code == "ZZZ"
    assert c.name == "Some weird currency"
    assert c.decimals == -1
    assert c.type == CurrencyType.CRYPTO
    assert c.quantizer == MaxPrecisionQuantizer

def test_currency_of_raises_on_non_string_code():
    try:
        Currency.of(123, "US Dollars", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_on_non_alpha_code():
    try:
        Currency.of("US1", "US Dollars", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_on_non_uppercase_code():
    try:
        Currency.of("usd", "US Dollars", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_on_non_string_name():
    try:
        Currency.of("USD", 123, 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_on_empty_name():
    try:
        Currency.of("USD", "", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_on_name_with_leading_space():
    try:
        Currency.of("USD", " US Dollars", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_on_name_with_trailing_space():
    try:
        Currency.of("USD", "US Dollars ", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_on_non_integer_decimals():
    try:
        Currency.of("USD", "US Dollars", "2", CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_on_decimals_less_than_minus_one():
    try:
        Currency.of("USD", "US Dollars", -2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_on_non_currencytype_type():
    try:
        Currency.of("USD", "US Dollars", 2, "MONEY")
        assert False
    except ProgrammingError:
        assert True

def test_currency_equality():
    c1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    c2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert c1 == c2
    assert hash(c1) == hash(c2)

def test_currency_inequality_different_name():
    c1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    c2 = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert c1 != c2
    assert hash(c1) != hash(c2)

def test_currency_quantize_positive_decimals():
    c = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert c.quantize(Decimal("1.005")) == Decimal("1.00")
    assert c.quantize(Decimal("1.015")) == Decimal("1.02")

def test_currency_quantize_zero_decimals():
    c = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert c.quantize(Decimal("0.5")) == Decimal("0")
    assert c.quantize(Decimal("1.5")) == Decimal("2")

def test_currency_quantize_negative_decimals():
    c = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert c.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert c.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")


