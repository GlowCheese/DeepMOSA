####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_currency_constructor_with_valid_inputs():
    code = "USD"
    name = "US Dollars"
    decimals = 2
    ctype = CurrencyType.MONEY
    quantizer = make_quantizer(2)
    hashcode = hash((code, name, decimals, ctype, quantizer))
    currency = Currency(code, name, decimals, ctype, quantizer, hashcode)
    assert currency.code == code
    assert currency.name == name
    assert currency.decimals == decimals
    assert currency.type == ctype
    assert currency.quantizer == quantizer
    assert currency.hashcache == hashcode

def test_currency_constructor_with_zero_decimals():
    code = "JPY"
    name = "Japanese Yen"
    decimals = 0
    ctype = CurrencyType.MONEY
    quantizer = ZERO
    hashcode = hash((code, name, decimals, ctype, quantizer))
    currency = Currency(code, name, decimals, ctype, quantizer, hashcode)
    assert currency.code == code
    assert currency.name == name
    assert currency.decimals == decimals
    assert currency.type == ctype
    assert currency.quantizer == quantizer
    assert currency.hashcache == hashcode

def test_currency_constructor_with_negative_decimals():
    code = "ZZZ"
    name = "Some weird currency"
    decimals = -1
    ctype = CurrencyType.CRYPTO
    quantizer = MaxPrecisionQuantizer
    hashcode = hash((code, name, decimals, ctype, quantizer))
    currency = Currency(code, name, decimals, ctype, quantizer, hashcode)
    assert currency.code == code
    assert currency.name == name
    assert currency.decimals == decimals
    assert currency.type == ctype
    assert currency.quantizer == quantizer
    assert currency.hashcache == hashcode


# LLM-generated content at query #2
#--------------------------

```python
def test_currency_registry_constructor():
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2
    assert len(registry1) == 0
    assert registry1.all == []
    assert registry1.codes == []
    assert registry1.codenames == []


# LLM-generated content at query #3
#--------------------------

```python
def test_currency_constructor_creates_frozen_instance():
    currency = Currency("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal("0.01"), hash(("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal("0.01"))))
    with pytest.raises(FrozenInstanceError):
        currency.code = "EUR"

def test_currency_constructor_sets_attributes_correctly():
    currency = Currency("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal("0.01"), hash(("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal("0.01"))))
    assert currency.code == "USD"
    assert currency.name == "US Dollars"
    assert currency.decimals == 2
    assert currency.type == CurrencyType.MONEY
    assert currency.quantizer == Decimal("0.01")
    assert currency.hashcache == hash(("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal("0.01")))

def test_currency_constructor_enables_ordering():
    currency1 = Currency("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal("0.01"), hash(("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal("0.01"))))
    currency2 = Currency("EUR", "Euro", 2, CurrencyType.MONEY, Decimal("0.01"), hash(("EUR", "Euro", 2, CurrencyType.MONEY, Decimal("0.01"))))
    assert currency1 < currency2 or currency1 > currency2


# LLM-generated content at query #4
#--------------------------

```python
def test_currency_registry_constructor_creates_singleton_instance():
    instance1 = CurrencyRegistry()
    instance2 = CurrencyRegistry()
    assert instance1 is instance2


# LLM-generated content at query #5
#--------------------------

```python
def test_currency_registry_constructor():
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2
    assert len(registry1) == 0
    assert not registry1.has("USD")
    assert registry1.get("USD") is None


# LLM-generated content at query #6
#--------------------------

```python
def test_currency_constructor_with_valid_inputs():
    code = "USD"
    name = "US Dollars"
    decimals = 2
    ctype = CurrencyType.MONEY
    quantizer = make_quantizer(2)
    hashcache = hash((code, name, decimals, ctype, quantizer))
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    assert currency.code == code
    assert currency.name == name
    assert currency.decimals == decimals
    assert currency.type == ctype
    assert currency.quantizer == quantizer
    assert currency.hashcache == hashcache


# LLM-generated content at query #7
#--------------------------

```python
def test_currency_registry_initialization():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__codes == []


# LLM-generated content at query #8
#--------------------------

```python
def test_currency_registry_initialization():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open is False


# LLM-generated content at query #9
#--------------------------

```python
def test_currency_registry_initialization():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open is False


# LLM-generated content at query #10
#--------------------------

```python
def test_currency_registry_constructor():
    registry = CurrencyRegistry()
    assert isinstance(registry, CurrencyRegistry)
    assert registry.__instance is not None
    assert registry.__registry == OrderedDict([])
    assert registry.__currencies == []
    assert registry.__codes == []
    assert registry.__codenames == []
    assert registry.__ctx_open is False


# LLM-generated content at query #11
#--------------------------

```python
def test_currency_registry_initialization():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__codes == []


# LLM-generated content at query #12
#--------------------------

```python
def test_currency_constructor_with_valid_parameters():
    code = "USD"
    name = "US Dollars"
    decimals = 2
    ctype = CurrencyType.MONEY
    quantizer = make_quantizer(2)
    hashcode = hash((code, name, decimals, ctype, quantizer))
    currency = Currency(code, name, decimals, ctype, quantizer, hashcode)
    assert currency.code == code
    assert currency.name == name
    assert currency.decimals == decimals
    assert currency.type == ctype
    assert currency.quantizer == quantizer
    assert currency.hashcache == hashcode


# LLM-generated content at query #13
#--------------------------

```python
def test_currency_registry_constructor():
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2
    assert len(registry1) == 0
    assert "USD" not in registry1
    assert registry1.has("USD") is False
    assert registry1.get("USD") is None
    assert registry1.all == []
    assert registry1.codes == []
    assert registry1.codenames == []


# LLM-generated content at query #14
#--------------------------

```python
def test_currency_registry_constructor():
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2
    assert len(registry1) == 0
    assert not registry1.__ctx_open
    assert registry1.__registry == OrderedDict([])
    assert registry1.__currencies == []
    assert registry1.__codes == []
    assert registry1.__codenames == []


# LLM-generated content at query #15
#--------------------------

```python
def test_currency_constructor_creates_frozen_instance():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    with pytest.raises(FrozenInstanceError):
        usd.code = "EUR"

def test_currency_constructor_with_valid_inputs():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.code == "USD"
    assert usd.name == "US Dollars"
    assert usd.decimals == 2
    assert usd.type == CurrencyType.MONEY
    assert usd.quantizer == make_quantizer(2)

def test_currency_constructor_with_zero_decimals():
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy.quantizer == ZERO

def test_currency_constructor_with_negative_decimals():
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert zzz.quantizer == MaxPrecisionQuantizer

def test_currency_constructor_with_invalid_code_type():
    with pytest.raises(ProgrammingError):
        Currency.of(123, "US Dollars", 2, CurrencyType.MONEY)

def test_currency_constructor_with_invalid_code_case():
    with pytest.raises(ProgrammingError):
        Currency.of("usd", "US Dollars", 2, CurrencyType.MONEY)

def test_currency_constructor_with_invalid_code_chars():
    with pytest.raises(ProgrammingError):
        Currency.of("USD1", "US Dollars", 2, CurrencyType.MONEY)

def test_currency_constructor_with_empty_name():
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "", 2, CurrencyType.MONEY)

def test_currency_constructor_with_whitespace_name():
    with pytest.raises(ProgrammingError):
        Currency.of("USD", " US Dollars", 2, CurrencyType.MONEY)

def test_currency_constructor_with_invalid_decimals_type():
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", "2", CurrencyType.MONEY)

def test_currency_constructor_with_invalid_decimals_value():
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", -2, CurrencyType.MONEY)

def test_currency_constructor_with_invalid_type():
    with pytest.raises(ProgrammingError):
        Currency.of("USD", "US Dollars", 2, "MONEY")


# LLM-generated content at query #16
#--------------------------

```python
def test_currency_registry_singleton():
    instance1 = CurrencyRegistry()
    instance2 = CurrencyRegistry()
    assert instance1 is instance2


# LLM-generated content at query #17
#--------------------------

```python
def test_codes_initialization():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__codes == []


# LLM-generated content at query #18
#--------------------------

```python
def test_init_initializes_ctx_open_to_false():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open is False


# LLM-generated content at query #19
#--------------------------

```python
def test_currency_constructor_with_valid_parameters():
    code = "USD"
    name = "US Dollars"
    decimals = 2
    ctype = CurrencyType.MONEY
    quantizer = make_quantizer(2)
    hashcache = hash((code, name, decimals, ctype, quantizer))

    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)

    assert currency.code == code
    assert currency.name == name
    assert currency.decimals == decimals
    assert currency.type == ctype
    assert currency.quantizer == quantizer
    assert currency.hashcache == hashcache


# LLM-generated content at query #20
#--------------------------

```python
def test_currency_registry_initialization():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open is False


# LLM-generated content at query #21
#--------------------------

```python
def test_currency_constructor_creates_immutable_instance():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    with pytest.raises(FrozenInstanceError):
        usd.code = "EUR"

def test_currency_constructor_stores_code():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.code == "USD"

def test_currency_constructor_stores_name():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.name == "US Dollars"

def test_currency_constructor_stores_decimals():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.decimals == 2

def test_currency_constructor_stores_type():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.type == CurrencyType.MONEY

def test_currency_constructor_stores_quantizer():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.quantizer == make_quantizer(2)

def test_currency_constructor_stores_hashcache():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.hashcache == hash(("USD", "US Dollars", 2, CurrencyType.MONEY, make_quantizer(2)))


# LLM-generated content at query #22
#--------------------------

```python
def test_codes_initialization():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__codes == []


# LLM-generated content at query #23
#--------------------------

```python
def test_codes_initialization():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__codes == []


# LLM-generated content at query #24
#--------------------------

```python
def test_currency_constructor_with_valid_inputs():
    code = "USD"
    name = "US Dollars"
    decimals = 2
    ctype = CurrencyType.MONEY
    quantizer = make_quantizer(2)
    hashcode = hash((code, name, decimals, ctype, quantizer))
    currency = Currency(code, name, decimals, ctype, quantizer, hashcode)
    assert currency.code == "USD"
    assert currency.name == "US Dollars"
    assert currency.decimals == 2
    assert currency.type == CurrencyType.MONEY
    assert currency.quantizer == make_quantizer(2)
    assert currency.hashcache == hash((code, name, decimals, ctype, quantizer))

def test_currency_constructor_with_zero_decimals():
    code = "JPY"
    name = "Japanese Yen"
    decimals = 0
    ctype = CurrencyType.MONEY
    quantizer = ZERO
    hashcode = hash((code, name, decimals, ctype, quantizer))
    currency = Currency(code, name, decimals, ctype, quantizer, hashcode)
    assert currency.code == "JPY"
    assert currency.name == "Japanese Yen"
    assert currency.decimals == 0
    assert currency.type == CurrencyType.MONEY
    assert currency.quantizer == ZERO
    assert currency.hashcache == hash((code, name, decimals, ctype, quantizer))

def test_currency_constructor_with_negative_decimals():
    code = "ZZZ"
    name = "Some weird currency"
    decimals = -1
    ctype = CurrencyType.CRYPTO
    quantizer = MaxPrecisionQuantizer
    hashcode = hash((code, name, decimals, ctype, quantizer))
    currency = Currency(code, name, decimals, ctype, quantizer, hashcode)
    assert currency.code == "ZZZ"
    assert currency.name == "Some weird currency"
    assert currency.decimals == -1
    assert currency.type == CurrencyType.CRYPTO
    assert currency.quantizer == MaxPrecisionQuantizer
    assert currency.hashcache == hash((code, name, decimals, ctype, quantizer))


# LLM-generated content at query #25
#--------------------------

```python
def test_init_ctx_open_flag():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open is False


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_currency_equality_same_attributes():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2

def test_currency_equality_different_name():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 == usdx)

def test_currency_equality_different_type():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert not (usd1 == usd2)

def test_currency_equality_different_decimals():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 0, CurrencyType.MONEY)
    assert not (usd1 == usd2)

def test_currency_equality_different_code():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert not (usd == eur)

def test_currency_equality_non_currency_object():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd == "USD")


# LLM-generated content at query #2
#--------------------------

```python
def test_currency_registry_constructor():
    registry = CurrencyRegistry()
    assert isinstance(registry, CurrencyRegistry)
    assert registry is CurrencyRegistry.__instance
    assert CurrencyRegistry.__instance is not None


# LLM-generated content at query #3
#--------------------------

```python
def test_currency_registry_initialization():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open is False


# LLM-generated content at query #4
#--------------------------

```python
def test_currency_constructor_with_valid_parameters():
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


# LLM-generated content at query #5
#--------------------------

```python
def test_currency_registry_constructor():
    registry = CurrencyRegistry()
    assert isinstance(registry, CurrencyRegistry)
    assert len(registry) == 0
    assert registry.all == []
    assert registry.codes == []
    assert registry.codenames == []


# LLM-generated content at query #6
#--------------------------

```python
def test_currency_registry_initialization():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__codes == []


# LLM-generated content at query #7
#--------------------------

```python
def test_currency_constructor_with_valid_inputs():
    code = "USD"
    name = "US Dollars"
    decimals = 2
    ctype = CurrencyType.MONEY
    quantizer = make_quantizer(decimals)
    hashcode = hash((code, name, decimals, ctype, quantizer))
    currency = Currency(code, name, decimals, ctype, quantizer, hashcode)
    assert currency.code == code
    assert currency.name == name
    assert currency.decimals == decimals
    assert currency.type == ctype
    assert currency.quantizer == quantizer
    assert currency.hashcache == hashcode

def test_currency_constructor_with_zero_decimals():
    code = "JPY"
    name = "Japanese Yen"
    decimals = 0
    ctype = CurrencyType.MONEY
    quantizer = ZERO
    hashcode = hash((code, name, decimals, ctype, quantizer))
    currency = Currency(code, name, decimals, ctype, quantizer, hashcode)
    assert currency.code == code
    assert currency.name == name
    assert currency.decimals == decimals
    assert currency.type == ctype
    assert currency.quantizer == quantizer
    assert currency.hashcache == hashcode

def test_currency_constructor_with_negative_decimals():
    code = "ZZZ"
    name = "Some weird currency"
    decimals = -1
    ctype = CurrencyType.CRYPTO
    quantizer = MaxPrecisionQuantizer
    hashcode = hash((code, name, decimals, ctype, quantizer))
    currency = Currency(code, name, decimals, ctype, quantizer, hashcode)
    assert currency.code == code
    assert currency.name == name
    assert currency.decimals == decimals
    assert currency.type == ctype
    assert currency.quantizer == quantizer
    assert currency.hashcache == hashcode


# LLM-generated content at query #8
#--------------------------

```python
def test_currency_registry_initialization():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open is False


# LLM-generated content at query #9
#--------------------------

```python
def test_currency_constructor_with_valid_parameters():
    code = "USD"
    name = "US Dollars"
    decimals = 2
    ctype = CurrencyType.MONEY
    quantizer = make_quantizer(decimals)
    hashcode = hash((code, name, decimals, ctype, quantizer))
    currency = Currency(code, name, decimals, ctype, quantizer, hashcode)
    assert currency.code == code
    assert currency.name == name
    assert currency.decimals == decimals
    assert currency.type == ctype
    assert currency.quantizer == quantizer
    assert currency.hashcache == hashcode


# LLM-generated content at query #10
#--------------------------

```python
def test_currency_registry_init_codes_empty():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__codes == []


# LLM-generated content at query #11
#--------------------------

```python
def test_currency_registry_constructor():
    registry = CurrencyRegistry()
    assert isinstance(registry, CurrencyRegistry)
    assert len(registry) == 0
    assert registry.codes == []
    assert registry.codenames == []
    assert registry.all == []


# LLM-generated content at query #12
#--------------------------

```python
def test_registry_population_context_flag_initialization():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open is False


# LLM-generated content at query #13
#--------------------------

```python
def test_currency_constructor_with_valid_parameters():
    code = "USD"
    name = "US Dollars"
    decimals = 2
    ctype = CurrencyType.MONEY
    quantizer = make_quantizer(2)
    hashcache = hash((code, name, decimals, ctype, quantizer))
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)

    assert currency.code == code
    assert currency.name == name
    assert currency.decimals == decimals
    assert currency.type == ctype
    assert currency.quantizer == quantizer
    assert currency.hashcache == hashcache


# LLM-generated content at query #14
#--------------------------

```python
def test_currency_registry_initialization():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open == False


# LLM-generated content at query #15
#--------------------------

```python
def test_currency_registry_constructor():
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2
    assert len(registry1) == 0
    assert not registry1.has("USD")
    assert registry1.get("USD") is None
    assert registry1.all == []
    assert registry1.codes == []
    assert registry1.codenames == []


# LLM-generated content at query #16
#--------------------------

```python
def test_currency_constructor_with_valid_parameters():
    code = "USD"
    name = "US Dollars"
    decimals = 2
    ctype = CurrencyType.MONEY
    quantizer = make_quantizer(2)
    hashcache = hash((code, name, decimals, ctype, quantizer))

    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)

    assert currency.code == code
    assert currency.name == name
    assert currency.decimals == decimals
    assert currency.type == ctype
    assert currency.quantizer == quantizer
    assert currency.hashcache == hashcache


# LLM-generated content at query #17
#--------------------------

```python
def test_currency_registry_constructor():
    registry = CurrencyRegistry()
    assert isinstance(registry, CurrencyRegistry)
    assert len(registry) == 0
    assert "USD" not in registry
    assert not registry.has("USD")
    assert registry.get("USD") is None
    assert registry.all == []
    assert registry.codes == []
    assert registry.codenames == []


# LLM-generated content at query #18
#--------------------------

```python
def test_currency_registry_constructor_creates_singleton_instance():
    instance1 = CurrencyRegistry()
    instance2 = CurrencyRegistry()
    assert instance1 is instance2


# LLM-generated content at query #19
#--------------------------

```python
def test_currency_constructor_with_valid_inputs():
    currency = Currency("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal("0.01"), hash(("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal("0.01"))))
    assert currency.code == "USD"
    assert currency.name == "US Dollars"
    assert currency.decimals == 2
    assert currency.type == CurrencyType.MONEY
    assert currency.quantizer == Decimal("0.01")
    assert currency.hashcache == hash(("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal("0.01")))

def test_currency_constructor_with_zero_decimals():
    currency = Currency("JPY", "Japanese Yen", 0, CurrencyType.MONEY, Decimal("1"), hash(("JPY", "Japanese Yen", 0, CurrencyType.MONEY, Decimal("1"))))
    assert currency.code == "JPY"
    assert currency.name == "Japanese Yen"
    assert currency.decimals == 0
    assert currency.type == CurrencyType.MONEY
    assert currency.quantizer == Decimal("1")
    assert currency.hashcache == hash(("JPY", "Japanese Yen", 0, CurrencyType.MONEY, Decimal("1")))

def test_currency_constructor_with_negative_decimals():
    currency = Currency("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO, MaxPrecisionQuantizer, hash(("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO, MaxPrecisionQuantizer)))
    assert currency.code == "ZZZ"
    assert currency.name == "Some weird currency"
    assert currency.decimals == -1
    assert currency.type == CurrencyType.CRYPTO
    assert currency.quantizer == MaxPrecisionQuantizer
    assert currency.hashcache == hash(("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO, MaxPrecisionQuantizer))


# LLM-generated content at query #20
#--------------------------

```python
def test_currency_registry_initialization():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open is False


# LLM-generated content at query #21
#--------------------------

```python
def test_currency_registry_constructor_creates_singleton_instance():
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2


# LLM-generated content at query #22
#--------------------------

```python
def test_currency_registry_initialization():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open is False


# LLM-generated content at query #23
#--------------------------

```python
def test_currency_constructor_creates_immutable_instance():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    with pytest.raises(FrozenInstanceError):
        usd.code = "EUR"
    with pytest.raises(FrozenInstanceError):
        usd.name = "Euro"
    with pytest.raises(FrozenInstanceError):
        usd.decimals = 0
    with pytest.raises(FrozenInstanceError):
        usd.type = CurrencyType.CRYPTO
    with pytest.raises(FrozenInstanceError):
        usd.quantizer = Decimal("0.0001")
    with pytest.raises(FrozenInstanceError):
        usd.hashcache = 12345


# LLM-generated content at query #24
#--------------------------

```python
def test_currency_registry_initialization():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open == False


# LLM-generated content at query #25
#--------------------------

```python
def test_currency_registry_initialization():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__codes == []


# LLM-generated content at query #26
#--------------------------

```python
def test_currency_constructor_with_valid_inputs():
    code = "USD"
    name = "US Dollars"
    decimals = 2
    ctype = CurrencyType.MONEY
    quantizer = make_quantizer(2)
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
    assert currency.code == code
    assert currency.name == name
    assert currency.decimals == decimals
    assert currency.type == ctype
    assert currency.quantizer == quantizer
    assert currency.hashcache == hashcache

def test_currency_constructor_with_negative_decimals():
    code = "ZZZ"
    name = "Some weird currency"
    decimals = -1
    ctype = CurrencyType.CRYPTO
    quantizer = MaxPrecisionQuantizer
    hashcache = hash((code, name, decimals, ctype, quantizer))
    currency = Currency(code, name, decimals, ctype, quantizer, hashcache)
    assert currency.code == code
    assert currency.name == name
    assert currency.decimals == decimals
    assert currency.type == ctype
    assert currency.quantizer == quantizer
    assert currency.hashcache == hashcache


