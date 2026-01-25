####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_currency_registry_constructor_creates_singleton_instance():
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2


# LLM-generated content at query #2
#--------------------------

```python
def test_currency_registry_initialization():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__codes == []


# LLM-generated content at query #3
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


# LLM-generated content at query #4
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


# LLM-generated content at query #5
#--------------------------

```python
def test_currency_registry_constructor_creates_singleton_instance():
    instance1 = CurrencyRegistry()
    instance2 = CurrencyRegistry()
    assert instance1 is instance2


# LLM-generated content at query #6
#--------------------------

```python
def test_currency_constructor_with_valid_inputs():
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

def test_currency_constructor_with_invalid_code_type():
    code = 123
    name = "US Dollars"
    decimals = 2
    ctype = CurrencyType.MONEY
    quantizer = make_quantizer(decimals)
    hashcache = hash((code, name, decimals, ctype, quantizer))
    try:
        Currency(code, name, decimals, ctype, quantizer, hashcache)
        assert False, "Expected ProgrammingError"
    except ProgrammingError:
        pass

def test_currency_constructor_with_invalid_code_characters():
    code = "USD1"
    name = "US Dollars"
    decimals = 2
    ctype = CurrencyType.MONEY
    quantizer = make_quantizer(decimals)
    hashcache = hash((code, name, decimals, ctype, quantizer))
    try:
        Currency(code, name, decimals, ctype, quantizer, hashcache)
        assert False, "Expected ProgrammingError"
    except ProgrammingError:
        pass

def test_currency_constructor_with_invalid_code_case():
    code = "usd"
    name = "US Dollars"
    decimals = 2
    ctype = CurrencyType.MONEY
    quantizer = make_quantizer(decimals)
    hashcache = hash((code, name, decimals, ctype, quantizer))
    try:
        Currency(code, name, decimals, ctype, quantizer, hashcache)
        assert False, "Expected ProgrammingError"
    except ProgrammingError:
        pass

def test_currency_constructor_with_invalid_name_type():
    code = "USD"
    name = 123
    decimals = 2
    ctype = CurrencyType.MONEY
    quantizer = make_quantizer(decimals)
    hashcache = hash((code, name, decimals, ctype, quantizer))
    try:
        Currency(code, name, decimals, ctype, quantizer, hashcache)
        assert False, "Expected ProgrammingError"
    except ProgrammingError:
        pass

def test_currency_constructor_with_empty_name():
    code = "USD"
    name = ""
    decimals = 2
    ctype = CurrencyType.MONEY
    quantizer = make_quantizer(decimals)
    hashcache = hash((code, name, decimals, ctype, quantizer))
    try:
        Currency(code, name, decimals, ctype, quantizer, hashcache)
        assert False, "Expected ProgrammingError"
    except ProgrammingError:
        pass

def test_currency_constructor_with_name_with_leading_or_trailing_spaces():
    code = "USD"
    name = " US Dollars "
    decimals = 2
    ctype = CurrencyType.MONEY
    quantizer = make_quantizer(decimals)
    hashcache = hash((code, name, decimals, ctype, quantizer))
    try:
        Currency(code, name, decimals, ctype, quantizer, hashcache)
        assert False, "Expected ProgrammingError"
    except ProgrammingError:
        pass

def test_currency_constructor_with_invalid_decimals_type():
    code = "USD"
    name = "US Dollars"
    decimals = "2"
    ctype = CurrencyType.MONEY
    quantizer = make_quantizer(2)
    hashcache = hash((code, name, decimals, ctype, quantizer))
    try:
        Currency(code, name, decimals, ctype, quantizer, hashcache)
        assert False, "Expected ProgrammingError"
    except ProgrammingError:
        pass

def test_currency_constructor_with_invalid_decimals_value():
    code = "USD"
    name = "US Dollars"
    decimals = -2
    ctype = CurrencyType.MONEY
    quantizer = make_quantizer(2)
    hashcache = hash((code, name, decimals, ctype, quantizer))
    try:
        Currency(code, name, decimals, ctype, quantizer, hashcache)
        assert False, "Expected ProgrammingError"
    except ProgrammingError:
        pass

def test_currency_constructor_with_invalid_type():
    code = "USD"
    name = "US Dollars"
    decimals = 2
    ctype = "MONEY"
    quantizer = make_quantizer(decimals)
    hashcache = hash((code, name, decimals, ctype, quantizer))
    try:
        Currency(code, name, decimals, ctype, quantizer, hashcache)
        assert False, "Expected ProgrammingError"
    except ProgrammingError:
        pass


# LLM-generated content at query #7
#--------------------------

```python
def test_init_codes_empty():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__codes == []


# LLM-generated content at query #8
#--------------------------

```python
def test_currency_registry_constructor():
    registry = CurrencyRegistry()
    assert isinstance(registry, CurrencyRegistry)
    assert registry is CurrencyRegistry.__instance
    assert len(registry) == 0
    assert registry.codes == []
    assert registry.codenames == []


# LLM-generated content at query #9
#--------------------------

```python
def test_empty_codes_list():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__codes == []


# LLM-generated content at query #10
#--------------------------

```python
def test_currency_registry_constructor_singleton():
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2


# LLM-generated content at query #11
#--------------------------

```python
def test_initialization_sets_ctx_open_to_false():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open is False


# LLM-generated content at query #12
#--------------------------

```python
def test_currency_registry_init_empty_codes():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__codes == []


# LLM-generated content at query #13
#--------------------------

```python
def test_currency_registry_constructor_creates_singleton_instance():
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2


# LLM-generated content at query #14
#--------------------------

```python
def test_initialization_sets_ctx_open_to_false():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open is False


# LLM-generated content at query #15
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


# LLM-generated content at query #16
#--------------------------

```python
def test_currency_registry_constructor():
    registry = CurrencyRegistry()
    assert registry.__instance is not None
    assert isinstance(registry.__registry, OrderedDict)
    assert len(registry.__registry) == 0
    assert isinstance(registry.__currencies, list)
    assert len(registry.__currencies) == 0
    assert isinstance(registry.__codes, list)
    assert len(registry.__codes) == 0
    assert isinstance(registry.__codenames, list)
    assert len(registry.__codenames) == 0
    assert registry.__ctx_open is False


# LLM-generated content at query #17
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


# LLM-generated content at query #18
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


# LLM-generated content at query #19
#--------------------------

```python
def test_currency_constructor():
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


# LLM-generated content at query #20
#--------------------------

```python
def test_currency_registry_initialization():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open == False


# LLM-generated content at query #21
#--------------------------

```python
def test_currency_registry_constructor():
    registry = CurrencyRegistry()
    assert isinstance(registry, CurrencyRegistry)
    assert registry is CurrencyRegistry.__instance


# LLM-generated content at query #22
#--------------------------

```python
def test_currency_registry_initialization():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open == False


# LLM-generated content at query #23
#--------------------------

```python
def test_currency_constructor_with_valid_inputs():
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


# LLM-generated content at query #24
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


# LLM-generated content at query #25
#--------------------------

```python
def test_init_codes_empty():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__codes == []


# LLM-generated content at query #26
#--------------------------

```python
def test_registry_population_context_flag_initialization():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open == False


# LLM-generated content at query #27
#--------------------------

```python
def test_currency_constructor():
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


# LLM-generated content at query #28
#--------------------------

```python
def test_currency_registry_initialization():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__codes == []


# LLM-generated content at query #29
#--------------------------

```python
def test_currency_registry_initialization():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open == False


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
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


# LLM-generated content at query #2
#--------------------------

```python
def test_currency_constructor_with_valid_parameters():
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


# LLM-generated content at query #3
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


# LLM-generated content at query #4
#--------------------------

```python
def test_currency_registry_initialization_codes_empty():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__codes == []


# LLM-generated content at query #5
#--------------------------

```python
def test_registry_population_context_flag_initialization():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open == False


# LLM-generated content at query #6
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

def test_currency_equality_non_currency_object():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd == "USD")


# LLM-generated content at query #7
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


# LLM-generated content at query #8
#--------------------------

```python
def test_currency_constructor():
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


# LLM-generated content at query #9
#--------------------------

```python
def test_currency_equality():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2


# LLM-generated content at query #10
#--------------------------

```python
def test_currency_registry_getitem_existing_currency():
    assert Currencies["USD"].code == 'USD'
    assert Currencies["USD"].name == 'US Dollar'
    assert Currencies["USD"].type.name == 'MONEY'

def test_currency_registry_getitem_non_existing_currency():
    try:
        Currencies["NON-EXISTING"]
        assert False, "Expected CurrencyLookupError"
    except CurrencyLookupError as e:
        assert str(e) == "Currency identified by code 'NON-EXISTING' does not exist"


# LLM-generated content at query #11
#--------------------------

```python
def test_currency_registry_init():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open is False


# LLM-generated content at query #12
#--------------------------

```python
def test_currency_equality():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2


# LLM-generated content at query #13
#--------------------------

```python
def test_currency_registry_constructor_creates_singleton_instance():
    instance1 = CurrencyRegistry()
    instance2 = CurrencyRegistry()
    assert instance1 is instance2


# LLM-generated content at query #14
#--------------------------

```python
def test_currency_constructor():
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


# LLM-generated content at query #15
#--------------------------

```python
def test_currency_lookup_error_raised_for_invalid_code():
    assert "NON-EXISTING" not in Currencies
    try:
        Currencies["NON-EXISTING"]
        assert False, "Expected CurrencyLookupError to be raised"
    except CurrencyLookupError:
        pass


# LLM-generated content at query #16
#--------------------------

```python
def test_currency_registry_constructor():
    registry = CurrencyRegistry()
    assert isinstance(registry, CurrencyRegistry)
    assert registry is CurrencyRegistry.__instance


# LLM-generated content at query #17
#--------------------------

```python
def test_currency_registry_init_codes_empty():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__codes == []


# LLM-generated content at query #18
#--------------------------

```python
def test_currency_registry_initialization():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__codes == []


# LLM-generated content at query #19
#--------------------------

```python
def test_currency_registry_singleton():
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    assert registry1 is registry2


# LLM-generated content at query #20
#--------------------------

```python
def test_currency_registry_init_codes_empty():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__codes == []


# LLM-generated content at query #21
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


# LLM-generated content at query #22
#--------------------------

```python
def test_currency_registry_initialization():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open == False


# LLM-generated content at query #23
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


# LLM-generated content at query #24
#--------------------------

```python
def test_currency_constructor_creates_immutable_instance():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    with pytest.raises(Exception):
        usd.code = "EUR"

def test_currency_constructor_sets_code():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.code == "USD"

def test_currency_constructor_sets_name():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.name == "US Dollars"

def test_currency_constructor_sets_decimals():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.decimals == 2

def test_currency_constructor_sets_type():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.type == CurrencyType.MONEY

def test_currency_constructor_sets_quantizer_for_positive_decimals():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.quantizer == make_quantizer(2)

def test_currency_constructor_sets_quantizer_for_zero_decimals():
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy.quantizer == ZERO

def test_currency_constructor_sets_quantizer_for_negative_decimals():
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert zzz.quantizer == MaxPrecisionQuantizer

def test_currency_constructor_computes_hash():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    expected_hash = hash(("USD", "US Dollars", 2, CurrencyType.MONEY, make_quantizer(2)))
    assert usd.hashcache == expected_hash


# LLM-generated content at query #25
#--------------------------

```python
def test_codes_initialization():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__codes == []


# LLM-generated content at query #26
#--------------------------

```python
def test_currency_registry_initialization():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open == False


# LLM-generated content at query #27
#--------------------------

```python
def test_currency_constructor_creates_immutable_instance():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    with pytest.raises(FrozenInstanceError):
        usd.code = "EUR"

def test_currency_constructor_sets_attributes_correctly():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.code == "USD"
    assert usd.name == "US Dollars"
    assert usd.decimals == 2
    assert usd.type == CurrencyType.MONEY
    assert usd.quantizer == make_quantizer(2)
    assert usd.hashcache == hash(("USD", "US Dollars", 2, CurrencyType.MONEY, make_quantizer(2)))

def test_currency_constructor_with_zero_decimals():
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy.quantizer == ZERO

def test_currency_constructor_with_negative_decimals():
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert zzz.quantizer == MaxPrecisionQuantizer


# LLM-generated content at query #28
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


# LLM-generated content at query #29
#--------------------------

```python
def test_currency_registry_initialization():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open == False


# LLM-generated content at query #30
#--------------------------

```python
def test_currency_constructor_with_valid_inputs():
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


