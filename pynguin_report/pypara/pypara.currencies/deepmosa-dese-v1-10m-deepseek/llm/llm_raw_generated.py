####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_currency_of_creates_valid_currency():
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert USD.code == "USD"
    assert USD.name == "US Dollars"
    assert USD.decimals == 2
    assert USD.type == CurrencyType.MONEY
    assert USD.quantizer == make_quantizer(2)

def test_currency_of_creates_valid_currency_with_zero_decimals():
    JPY = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert JPY.code == "JPY"
    assert JPY.name == "Japanese Yen"
    assert JPY.decimals == 0
    assert JPY.type == CurrencyType.MONEY
    assert JPY.quantizer == ZERO

def test_currency_of_creates_valid_currency_with_negative_decimals():
    ZZZ = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert ZZZ.code == "ZZZ"
    assert ZZZ.name == "Some weird currency"
    assert ZZZ.decimals == -1
    assert ZZZ.type == CurrencyType.CRYPTO
    assert ZZZ.quantizer == MaxPrecisionQuantizer

def test_currency_of_raises_error_for_non_string_code():
    try:
        Currency.of(123, "US Dollars", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError as e:
        assert "Currency code must be a string" in str(e)

def test_currency_of_raises_error_for_non_alphabetic_code():
    try:
        Currency.of("US1", "US Dollars", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError as e:
        assert "Currency code must contain only alphabetic characters" in str(e)

def test_currency_of_raises_error_for_non_uppercase_code():
    try:
        Currency.of("usd", "US Dollars", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError as e:
        assert "Currency code must be all uppercase" in str(e)

def test_currency_of_raises_error_for_non_string_name():
    try:
        Currency.of("USD", 123, 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError as e:
        assert "Currency name must be a string" in str(e)

def test_currency_of_raises_error_for_empty_name():
    try:
        Currency.of("USD", "", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError as e:
        assert "Currency name can not be empty" in str(e)

def test_currency_of_raises_error_for_name_with_leading_space():
    try:
        Currency.of("USD", " US Dollars", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError as e:
        assert "Trim the currency name" in str(e)

def test_currency_of_raises_error_for_name_with_trailing_space():
    try:
        Currency.of("USD", "US Dollars ", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError as e:
        assert "Trim the currency name" in str(e)

def test_currency_of_raises_error_for_non_integer_decimals():
    try:
        Currency.of("USD", "US Dollars", "2", CurrencyType.MONEY)
        assert False
    except ProgrammingError as e:
        assert "Number of decimals must be an integer" in str(e)

def test_currency_of_raises_error_for_decimals_less_than_minus_one():
    try:
        Currency.of("USD", "US Dollars", -2, CurrencyType.MONEY)
        assert False
    except ProgrammingError as e:
        assert "Number of decimals can not be less than -1" in str(e)

def test_currency_of_raises_error_for_non_currencytype_type():
    try:
        Currency.of("USD", "US Dollars", 2, "MONEY")
        assert False
    except ProgrammingError as e:
        assert "Currency Type must be of type `CurrencyType`" in str(e)

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
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert USD.quantize(Decimal("1.005")) == Decimal("1.00")
    assert USD.quantize(Decimal("1.015")) == Decimal("1.02")

def test_currency_quantize_with_zero_decimals():
    JPY = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert JPY.quantize(Decimal("0.5")) == Decimal("0")
    assert JPY.quantize(Decimal("1.5")) == Decimal("2")

def test_currency_quantize_with_negative_decimals():
    ZZZ = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert ZZZ.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert ZZZ.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")


# LLM-generated content at query #2
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

def test_constructor_initializes_empty_containers():
    registry = CurrencyRegistry()
    assert len(registry._CurrencyRegistry__registry) == 0
    assert len(registry._CurrencyRegistry__currencies) == 0
    assert len(registry._CurrencyRegistry__codes) == 0
    assert len(registry._CurrencyRegistry__codenames) == 0

def test_constructor_returns_same_instance_multiple_calls():
    registry1 = CurrencyRegistry()
    registry2 = CurrencyRegistry()
    registry3 = CurrencyRegistry()
    assert registry1 is registry2
    assert registry2 is registry3
    assert registry1 is registry3


# LLM-generated content at query #3
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


# LLM-generated content at query #5
#--------------------------

def test_currency_registry_initialization():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__codes == []


# LLM-generated content at query #6
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


# LLM-generated content at query #7
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
    instance1 = CurrencyRegistry()
    instance1._CurrencyRegistry__ctx_open = True
    instance2 = CurrencyRegistry()
    assert instance2._CurrencyRegistry__ctx_open == True

def test_constructor_returns_same_instance_multiple_calls():
    instances = [CurrencyRegistry() for _ in range(5)]
    first_instance = instances[0]
    for instance in instances:
        assert instance is first_instance

def test_constructor_initial_has_method_returns_false():
    registry = CurrencyRegistry()
    assert registry.has("USD") == False

def test_constructor_initial_len_returns_zero():
    registry = CurrencyRegistry()
    assert len(registry) == 0

def test_constructor_initial_contains_returns_false():
    registry = CurrencyRegistry()
    assert ("USD" in registry) == False

def test_constructor_initial_all_property_empty():
    registry = CurrencyRegistry()
    assert registry.all == []

def test_constructor_initial_codes_property_empty():
    registry = CurrencyRegistry()
    assert registry.codes == []

def test_constructor_initial_codenames_property_empty():
    registry = CurrencyRegistry()
    assert registry.codenames == []

def test_constructor_getitem_raises_error_initially():
    registry = CurrencyRegistry()
    try:
        registry["USD"]
        assert False
    except CurrencyLookupError:
        assert True

def test_constructor_get_returns_none_initially():
    registry = CurrencyRegistry()
    assert registry.get("USD") is None

def test_constructor_get_with_default_returns_default():
    registry = CurrencyRegistry()
    default_currency = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert registry.get("USD", default=default_currency) is default_currency


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
    assert registry._CurrencyRegistry__ctx_open == False

def test_constructor_singleton_preserves_state():
    registry1 = CurrencyRegistry()
    registry1._CurrencyRegistry__ctx_open = True
    registry2 = CurrencyRegistry()
    assert registry2._CurrencyRegistry__ctx_open == True


# LLM-generated content at query #11
#--------------------------

def test_currency_registry_singleton():
    instance1 = CurrencyRegistry()
    instance2 = CurrencyRegistry()
    assert instance1 is instance2

def test_currency_registry_initialization():
    registry = CurrencyRegistry()
    assert len(registry) == 0
    assert registry.all == []
    assert registry.codes == []
    assert registry.codenames == []

def test_currency_registry_context_manager():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open == False
    with registry as register:
        assert registry._CurrencyRegistry__ctx_open == True
        assert callable(register)
    assert registry._CurrencyRegistry__ctx_open == False

def test_currency_registry_contains():
    registry = CurrencyRegistry()
    assert "USD" not in registry
    assert registry.has("USD") == False

def test_currency_registry_getitem():
    registry = CurrencyRegistry()
    try:
        registry["USD"]
        assert False
    except CurrencyLookupError:
        assert True

def test_currency_registry_get():
    registry = CurrencyRegistry()
    assert registry.get("USD") is None
    default_currency = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert registry.get("USD", default=default_currency) is default_currency


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

def test_constructor_singleton_persistence():
    CurrencyRegistry._CurrencyRegistry__instance = None
    first = CurrencyRegistry()
    first._CurrencyRegistry__registry = {"test": "dummy"}
    second = CurrencyRegistry()
    assert second._CurrencyRegistry__registry == {"test": "dummy"}


# LLM-generated content at query #13
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


# LLM-generated content at query #14
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


# LLM-generated content at query #15
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

def test_currency_registry_constructor_context_closed_initially():
    registry = CurrencyRegistry()
    try:
        registry._CurrencyRegistry__register(Currency.of("TEST", "Test Currency", 2, CurrencyType.MONEY))
        assert False
    except ProgrammingError:
        assert True

def test_currency_registry_constructor_private_attributes():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__registry == OrderedDict([])
    assert registry._CurrencyRegistry__currencies == []
    assert registry._CurrencyRegistry__codes == []
    assert registry._CurrencyRegistry__codenames == []
    assert registry._CurrencyRegistry__ctx_open == False


# LLM-generated content at query #16
#--------------------------

def test_currency_of_creates_valid_currency():
    ctype = CurrencyType.MONEY
    currency = Currency.of("USD", "US Dollars", 2, ctype)
    assert currency.code == "USD"
    assert currency.name == "US Dollars"
    assert currency.decimals == 2
    assert currency.type == ctype
    assert currency.quantizer == make_quantizer(2)
    assert currency.hashcache == hash(("USD", "US Dollars", 2, ctype, make_quantizer(2)))

def test_currency_of_with_zero_decimals():
    ctype = CurrencyType.MONEY
    currency = Currency.of("JPY", "Japanese Yen", 0, ctype)
    assert currency.code == "JPY"
    assert currency.name == "Japanese Yen"
    assert currency.decimals == 0
    assert currency.type == ctype
    assert currency.quantizer == ZERO

def test_currency_of_with_negative_decimals():
    ctype = CurrencyType.CRYPTO
    currency = Currency.of("ZZZ", "Some weird currency", -1, ctype)
    assert currency.code == "ZZZ"
    assert currency.name == "Some weird currency"
    assert currency.decimals == -1
    assert currency.type == ctype
    assert currency.quantizer == MaxPrecisionQuantizer

def test_currency_of_raises_on_non_string_code():
    try:
        Currency.of(123, "US Dollars", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError as e:
        assert "Currency code must be a string" in str(e)

def test_currency_of_raises_on_non_alpha_code():
    try:
        Currency.of("US1", "US Dollars", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError as e:
        assert "Currency code must contain only alphabetic characters" in str(e)

def test_currency_of_raises_on_non_uppercase_code():
    try:
        Currency.of("usd", "US Dollars", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError as e:
        assert "Currency code must be all uppercase" in str(e)

def test_currency_of_raises_on_non_string_name():
    try:
        Currency.of("USD", 123, 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError as e:
        assert "Currency name must be a string" in str(e)

def test_currency_of_raises_on_empty_name():
    try:
        Currency.of("USD", "", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError as e:
        assert "Currency name can not be empty" in str(e)

def test_currency_of_raises_on_name_with_leading_space():
    try:
        Currency.of("USD", " US Dollars", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError as e:
        assert "Trim the currency name" in str(e)

def test_currency_of_raises_on_name_with_trailing_space():
    try:
        Currency.of("USD", "US Dollars ", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError as e:
        assert "Trim the currency name" in str(e)

def test_currency_of_raises_on_non_integer_decimals():
    try:
        Currency.of("USD", "US Dollars", "2", CurrencyType.MONEY)
        assert False
    except ProgrammingError as e:
        assert "Number of decimals must be an integer" in str(e)

def test_currency_of_raises_on_decimals_less_than_minus_one():
    try:
        Currency.of("USD", "US Dollars", -2, CurrencyType.MONEY)
        assert False
    except ProgrammingError as e:
        assert "Number of decimals can not be less than -1" in str(e)

def test_currency_of_raises_on_non_currencytype_type():
    try:
        Currency.of("USD", "US Dollars", 2, "MONEY")
        assert False
    except ProgrammingError as e:
        assert "Currency Type must be of type `CurrencyType`" in str(e)

def test_currency_equality():
    ctype = CurrencyType.MONEY
    usd1 = Currency.of("USD", "US Dollars", 2, ctype)
    usd2 = Currency.of("USD", "US Dollars", 2, ctype)
    assert usd1 == usd2
    assert hash(usd1) == hash(usd2)

def test_currency_inequality_due_to_name():
    ctype = CurrencyType.MONEY
    usd1 = Currency.of("USD", "US Dollars", 2, ctype)
    usdx = Currency.of("USD", "UX Dollars", 2, ctype)
    assert usd1 != usdx
    assert hash(usd1) != hash(usdx)

def test_currency_quantize_with_positive_decimals():
    ctype = CurrencyType.MONEY
    usd = Currency.of("USD", "US Dollars", 2, ctype)
    assert usd.quantize(Decimal("1.005")) == Decimal("1.00")
    assert usd.quantize(Decimal("1.015")) == Decimal("1.02")

def test_currency_quantize_with_zero_decimals():
    ctype = CurrencyType.MONEY
    jpy = Currency.of("JPY", "Japanese Yen", 0, ctype)
    assert jpy.quantize(Decimal("0.5")) == Decimal("0")
    assert jpy.quantize(Decimal("1.5")) == Decimal("2")

def test_currency_quantize_with_negative_decimals():
    ctype = CurrencyType.CRYPTO
    zzz = Currency.of("ZZZ", "Some weird currency", -1, ctype)
    assert zzz.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert zzz.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")


# LLM-generated content at query #17
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

def test_constructor_singleton_persistence():
    instance1 = CurrencyRegistry()
    instance1._CurrencyRegistry__ctx_open = True
    instance2 = CurrencyRegistry()
    assert instance2._CurrencyRegistry__ctx_open == True


# LLM-generated content at query #20
#--------------------------

def test_currency_registry_init_codes_empty():
    registry = CurrencyRegistry()
    result = bool(registry._CurrencyRegistry__codes)
    assert result == False


# LLM-generated content at query #21
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

def test_currency_registry_constructor_context_closed_initially():
    registry = CurrencyRegistry()
    try:
        registry._CurrencyRegistry__register(Currency.of("TEST", "Test Currency", 2, CurrencyType.MONEY))
        assert False, "Should have raised ProgrammingError"
    except ProgrammingError:
        pass

def test_currency_registry_constructor_private_attributes_exist():
    registry = CurrencyRegistry()
    assert hasattr(registry, '_CurrencyRegistry__registry')
    assert hasattr(registry, '_CurrencyRegistry__currencies')
    assert hasattr(registry, '_CurrencyRegistry__codes')
    assert hasattr(registry, '_CurrencyRegistry__codenames')
    assert hasattr(registry, '_CurrencyRegistry__ctx_open')
    assert registry._CurrencyRegistry__ctx_open is False


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

def test_constructor_singleton_preserves_state():
    registry1 = CurrencyRegistry()
    registry1._CurrencyRegistry__ctx_open = True
    registry2 = CurrencyRegistry()
    assert registry2._CurrencyRegistry__ctx_open == True


# LLM-generated content at query #23
#--------------------------

def test_currency_of_creates_valid_instance():
    ccy = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert ccy.code == "USD"
    assert ccy.name == "US Dollars"
    assert ccy.decimals == 2
    assert ccy.type == CurrencyType.MONEY
    assert ccy.quantizer == make_quantizer(2)
    assert ccy.hashcache == hash(("USD", "US Dollars", 2, CurrencyType.MONEY, make_quantizer(2)))

def test_currency_of_with_zero_decimals():
    ccy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert ccy.code == "JPY"
    assert ccy.name == "Japanese Yen"
    assert ccy.decimals == 0
    assert ccy.type == CurrencyType.MONEY
    assert ccy.quantizer == ZERO

def test_currency_of_with_negative_decimals():
    ccy = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert ccy.code == "ZZZ"
    assert ccy.name == "Some weird currency"
    assert ccy.decimals == -1
    assert ccy.type == CurrencyType.CRYPTO
    assert ccy.quantizer == MaxPrecisionQuantizer

def test_currency_of_raises_on_invalid_code_type():
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


# LLM-generated content at query #24
#--------------------------

def test_currency_of_creates_valid_currency():
    ctype = CurrencyType.MONEY
    currency = Currency.of("USD", "US Dollars", 2, ctype)
    assert currency.code == "USD"
    assert currency.name == "US Dollars"
    assert currency.decimals == 2
    assert currency.type == ctype
    assert currency.quantizer == make_quantizer(2)
    assert currency.hashcache == hash(("USD", "US Dollars", 2, ctype, make_quantizer(2)))

def test_currency_of_with_zero_decimals():
    ctype = CurrencyType.MONEY
    currency = Currency.of("JPY", "Japanese Yen", 0, ctype)
    assert currency.code == "JPY"
    assert currency.name == "Japanese Yen"
    assert currency.decimals == 0
    assert currency.type == ctype
    assert currency.quantizer == ZERO
    assert currency.hashcache == hash(("JPY", "Japanese Yen", 0, ctype, ZERO))

def test_currency_of_with_negative_decimals():
    ctype = CurrencyType.CRYPTO
    currency = Currency.of("ZZZ", "Some weird currency", -1, ctype)
    assert currency.code == "ZZZ"
    assert currency.name == "Some weird currency"
    assert currency.decimals == -1
    assert currency.type == ctype
    assert currency.quantizer == MaxPrecisionQuantizer
    assert currency.hashcache == hash(("ZZZ", "Some weird currency", -1, ctype, MaxPrecisionQuantizer))

def test_currency_of_raises_on_non_string_code():
    try:
        Currency.of(123, "US Dollars", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError as e:
        assert "Currency code must be a string" in str(e)

def test_currency_of_raises_on_non_alphabetic_code():
    try:
        Currency.of("US1", "US Dollars", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError as e:
        assert "Currency code must contain only alphabetic characters" in str(e)

def test_currency_of_raises_on_non_uppercase_code():
    try:
        Currency.of("usd", "US Dollars", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError as e:
        assert "Currency code must be all uppercase" in str(e)

def test_currency_of_raises_on_non_string_name():
    try:
        Currency.of("USD", 123, 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError as e:
        assert "Currency name must be a string" in str(e)

def test_currency_of_raises_on_empty_name():
    try:
        Currency.of("USD", "", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError as e:
        assert "Currency name can not be empty" in str(e)

def test_currency_of_raises_on_name_with_leading_space():
    try:
        Currency.of("USD", " US Dollars", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError as e:
        assert "Trim the currency name" in str(e)

def test_currency_of_raises_on_name_with_trailing_space():
    try:
        Currency.of("USD", "US Dollars ", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError as e:
        assert "Trim the currency name" in str(e)

def test_currency_of_raises_on_non_integer_decimals():
    try:
        Currency.of("USD", "US Dollars", "2", CurrencyType.MONEY)
        assert False
    except ProgrammingError as e:
        assert "Number of decimals must be an integer" in str(e)

def test_currency_of_raises_on_decimals_less_than_minus_one():
    try:
        Currency.of("USD", "US Dollars", -2, CurrencyType.MONEY)
        assert False
    except ProgrammingError as e:
        assert "Number of decimals can not be less than -1" in str(e)

def test_currency_of_raises_on_non_currencytype_type():
    try:
        Currency.of("USD", "US Dollars", 2, "MONEY")
        assert False
    except ProgrammingError as e:
        assert "Currency Type must be of type `CurrencyType`" in str(e)

def test_currency_equality():
    ctype = CurrencyType.MONEY
    usd1 = Currency.of("USD", "US Dollars", 2, ctype)
    usd2 = Currency.of("USD", "US Dollars", 2, ctype)
    assert usd1 == usd2
    assert hash(usd1) == hash(usd2)

def test_currency_inequality_due_to_name():
    ctype = CurrencyType.MONEY
    usd1 = Currency.of("USD", "US Dollars", 2, ctype)
    usdx = Currency.of("USD", "UX Dollars", 2, ctype)
    assert usd1 != usdx
    assert hash(usd1) != hash(usdx)

def test_currency_inequality_due_to_code():
    ctype = CurrencyType.MONEY
    usd = Currency.of("USD", "US Dollars", 2, ctype)
    eur = Currency.of("EUR", "US Dollars", 2, ctype)
    assert usd != eur
    assert hash(usd) != hash(eur)

def test_currency_inequality_due_to_decimals():
    ctype = CurrencyType.MONEY
    usd2 = Currency.of("USD", "US Dollars", 2, ctype)
    usd3 = Currency.of("USD", "US Dollars", 3, ctype)
    assert usd2 != usd3
    assert hash(usd2) != hash(usd3)

def test_currency_inequality_due_to_type():
    usd_money = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd_crypto = Currency.of("USD", "US Dollars", 2, CurrencyType.CRYPTO)
    assert usd_money != usd_crypto
    assert hash(usd_money) != hash(usd_crypto)

def test_currency_quantize_positive_decimals():
    currency = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert currency.quantize(Decimal("1.005")) == Decimal("1.00")
    assert currency.quantize(Decimal("1.015")) == Decimal("1.02")

def test_currency_quantize_zero_decimals():
    currency = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert currency.quantize(Decimal("0.5")) == Decimal("0")
    assert currency.quantize(Decimal("1.5")) == Decimal("2")

def test_currency_quantize_negative_decimals():
    currency = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert currency.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert currency.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")


# LLM-generated content at query #25
#--------------------------

def test_currency_registry_initialization():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__codes == []


# LLM-generated content at query #26
#--------------------------

def test_currency_of_creates_valid_instance():
    ccy = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert ccy.code == "USD"
    assert ccy.name == "US Dollars"
    assert ccy.decimals == 2
    assert ccy.type == CurrencyType.MONEY
    assert ccy.quantizer == make_quantizer(2)
    assert ccy.hashcache == hash(("USD", "US Dollars", 2, CurrencyType.MONEY, make_quantizer(2)))

def test_currency_of_creates_instance_with_zero_decimals():
    ccy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert ccy.code == "JPY"
    assert ccy.name == "Japanese Yen"
    assert ccy.decimals == 0
    assert ccy.type == CurrencyType.MONEY
    assert ccy.quantizer == ZERO

def test_currency_of_creates_instance_with_negative_decimals():
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
    except ProgrammingError as e:
        assert str(e) == "Currency code must be a string"

def test_currency_of_raises_error_for_non_alphabetic_code():
    try:
        Currency.of("US1", "US Dollars", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError as e:
        assert str(e) == "Currency code must contain only alphabetic characters"

def test_currency_of_raises_error_for_non_uppercase_code():
    try:
        Currency.of("usd", "US Dollars", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError as e:
        assert str(e) == "Currency code must be all uppercase"

def test_currency_of_raises_error_for_non_string_name():
    try:
        Currency.of("USD", 123, 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError as e:
        assert str(e) == "Currency name must be a string"

def test_currency_of_raises_error_for_empty_name():
    try:
        Currency.of("USD", "", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError as e:
        assert str(e) == "Currency name can not be empty"

def test_currency_of_raises_error_for_name_with_leading_or_trailing_spaces():
    try:
        Currency.of("USD", " US Dollars ", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError as e:
        assert str(e) == "Trim the currency name"

def test_currency_of_raises_error_for_non_integer_decimals():
    try:
        Currency.of("USD", "US Dollars", "2", CurrencyType.MONEY)
        assert False
    except ProgrammingError as e:
        assert str(e) == "Number of decimals must be an integer"

def test_currency_of_raises_error_for_decimals_less_than_minus_one():
    try:
        Currency.of("USD", "US Dollars", -2, CurrencyType.MONEY)
        assert False
    except ProgrammingError as e:
        assert str(e) == "Number of decimals can not be less than -1"

def test_currency_of_raises_error_for_non_currencytype_type():
    try:
        Currency.of("USD", "US Dollars", 2, "MONEY")
        assert False
    except ProgrammingError as e:
        assert str(e) == "Currency Type must be of type `CurrencyType`"

def test_currency_equality_based_on_hashcache():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2
    assert usd1 != usdx
    assert hash(usd1) == hash(usd2)
    assert hash(usd1) != hash(usdx)

def test_currency_quantize_with_positive_decimals():
    ccy = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert ccy.quantize(Decimal("1.005")) == Decimal("1.00")
    assert ccy.quantize(Decimal("1.015")) == Decimal("1.02")

def test_currency_quantize_with_zero_decimals():
    ccy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert ccy.quantize(Decimal("0.5")) == Decimal("0")
    assert ccy.quantize(Decimal("1.5")) == Decimal("2")

def test_currency_quantize_with_negative_decimals():
    ccy = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert ccy.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert ccy.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")


# LLM-generated content at query #27
#--------------------------

def test_ctx_open_initialized_to_false():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open == False


# LLM-generated content at query #28
#--------------------------

def test_currency_registry_constructor_singleton():
    instance1 = CurrencyRegistry()
    instance2 = CurrencyRegistry()
    assert instance1 is instance2

def test_currency_registry_constructor_initial_private_attributes():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__registry == {}
    assert registry._CurrencyRegistry__currencies == []
    assert registry._CurrencyRegistry__codes == []
    assert registry._CurrencyRegistry__codenames == []
    assert registry._CurrencyRegistry__ctx_open == False

def test_currency_registry_constructor_initial_public_methods():
    registry = CurrencyRegistry()
    assert callable(registry.__enter__)
    assert callable(registry.__exit__)
    assert callable(registry.__register)
    assert callable(registry.__len__)
    assert callable(registry.__contains__)
    assert callable(registry.__getitem__)
    assert callable(registry.has)
    assert callable(registry.get)

def test_currency_registry_constructor_initial_properties():
    registry = CurrencyRegistry()
    assert registry.all == []
    assert registry.codes == []
    assert registry.codenames == []


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

def test_constructor_singleton_persistence():
    instance1 = CurrencyRegistry()
    instance1._CurrencyRegistry__registry["TEST"] = "dummy"
    instance2 = CurrencyRegistry()
    assert instance2._CurrencyRegistry__registry["TEST"] == "dummy"


# LLM-generated content at query #30
#--------------------------

def test_constructor_creates_singleton():
    instance1 = CurrencyRegistry()
    instance2 = CurrencyRegistry()
    assert instance1 is instance2

def test_constructor_initializes_registry():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__registry == {}

def test_constructor_initializes_currencies_buffer():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__currencies == []

def test_constructor_initializes_codes_buffer():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__codes == []

def test_constructor_initializes_codenames_buffer():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__codenames == []

def test_constructor_initializes_context_flag_as_false():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open == False


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_eq_same_currency_objects():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2

def test_eq_different_currency_codes():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    eur = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert not (usd == eur)

def test_eq_different_names():
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert not (usd1 == usdx)

def test_eq_different_decimals():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert not (usd == jpy)

def test_eq_different_types():
    money = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    crypto = Currency.of("BTC", "Bitcoin", 8, CurrencyType.CRYPTO)
    assert not (money == crypto)

def test_eq_with_non_currency_object():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd == "USD")

def test_eq_with_none():
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert not (usd == None)

def test_eq_same_hash_different_fields():
    c1 = Currency.of("AAA", "Currency A", 2, CurrencyType.MONEY)
    c2 = Currency("AAA", "Currency A", 2, CurrencyType.MONEY, make_quantizer(2), hash(("AAA", "Currency A", 2, CurrencyType.MONEY, make_quantizer(2))))
    assert c1 == c2


# LLM-generated content at query #2
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
    registry1 = CurrencyRegistry()
    registry1._CurrencyRegistry__ctx_open = True
    registry2 = CurrencyRegistry()
    assert registry2._CurrencyRegistry__ctx_open is True


# LLM-generated content at query #3
#--------------------------

def test_ctx_open_initialized_to_false():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open == False


# LLM-generated content at query #4
#--------------------------

def test_constructor_creates_singleton():
    instance1 = CurrencyRegistry()
    instance2 = CurrencyRegistry()
    assert instance1 is instance2

def test_constructor_initializes_registry():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__registry == OrderedDict([])

def test_constructor_initializes_currencies_buffer():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__currencies == []

def test_constructor_initializes_codes_buffer():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__codes == []

def test_constructor_initializes_codenames_buffer():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__codenames == []

def test_constructor_initializes_context_flag_as_false():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open == False


# LLM-generated content at query #5
#--------------------------

def test_currency_registry_constructor_singleton():
    instance1 = CurrencyRegistry()
    instance2 = CurrencyRegistry()
    assert instance1 is instance2

def test_currency_registry_constructor_initial_attributes():
    registry = CurrencyRegistry()
    assert len(registry) == 0
    assert registry.all == []
    assert registry.codes == []
    assert registry.codenames == []

def test_currency_registry_constructor_private_attributes():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open == False
    assert registry._CurrencyRegistry__registry == {}
    assert registry._CurrencyRegistry__currencies == []
    assert registry._CurrencyRegistry__codes == []
    assert registry._CurrencyRegistry__codenames == []


# LLM-generated content at query #6
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


# LLM-generated content at query #7
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
    assert registry._CurrencyRegistry__registry == OrderedDict([])


# LLM-generated content at query #8
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


# LLM-generated content at query #9
#--------------------------

def test_currency_registry_constructor_singleton():
    instance1 = CurrencyRegistry()
    instance2 = CurrencyRegistry()
    assert instance1 is instance2

def test_currency_registry_constructor_initial_containers():
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


# LLM-generated content at query #10
#--------------------------

def test_currency_registry_singleton():
    instance1 = CurrencyRegistry()
    instance2 = CurrencyRegistry()
    assert instance1 is instance2

def test_currency_registry_initial_state():
    registry = CurrencyRegistry()
    assert len(registry) == 0
    assert registry.all == []
    assert registry.codes == []
    assert registry.codenames == []

def test_currency_registry_enter_exit():
    registry = CurrencyRegistry()
    with registry as register:
        assert callable(register)
    assert registry._CurrencyRegistry__ctx_open == False

def test_currency_registry_register_outside_context():
    registry = CurrencyRegistry()
    try:
        registry._CurrencyRegistry__register(None)
        assert False
    except ProgrammingError:
        assert True

def test_currency_registry_contains_empty():
    registry = CurrencyRegistry()
    assert "USD" not in registry
    assert registry.has("USD") == False

def test_currency_registry_getitem_empty():
    registry = CurrencyRegistry()
    try:
        registry["USD"]
        assert False
    except CurrencyLookupError:
        assert True

def test_currency_registry_get_empty():
    registry = CurrencyRegistry()
    assert registry.get("USD") is None
    default_currency = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert registry.get("USD", default=default_currency) is default_currency


# LLM-generated content at query #11
#--------------------------

def test_currency_of_creates_valid_currency():
    from decimal import Decimal
    from your_module import Currency, CurrencyType
    ccy = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    assert ccy.code == "USD"
    assert ccy.name == "US Dollar"
    assert ccy.decimals == 2
    assert ccy.type == CurrencyType.MONEY
    assert ccy.quantizer == Decimal('0.01')
    assert ccy.hashcache == hash(("USD", "US Dollar", 2, CurrencyType.MONEY, Decimal('0.01')))

def test_currency_of_with_zero_decimals():
    from decimal import Decimal
    from your_module import Currency, CurrencyType
    ccy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert ccy.code == "JPY"
    assert ccy.name == "Japanese Yen"
    assert ccy.decimals == 0
    assert ccy.type == CurrencyType.MONEY
    assert ccy.quantizer == Decimal('0')
    assert ccy.hashcache == hash(("JPY", "Japanese Yen", 0, CurrencyType.MONEY, Decimal('0')))

def test_currency_of_with_negative_decimals():
    from decimal import Decimal
    from your_module import Currency, CurrencyType, MaxPrecisionQuantizer
    ccy = Currency.of("ZZZ", "Weird Crypto", -1, CurrencyType.CRYPTO)
    assert ccy.code == "ZZZ"
    assert ccy.name == "Weird Crypto"
    assert ccy.decimals == -1
    assert ccy.type == CurrencyType.CRYPTO
    assert ccy.quantizer == MaxPrecisionQuantizer
    assert ccy.hashcache == hash(("ZZZ", "Weird Crypto", -1, CurrencyType.CRYPTO, MaxPrecisionQuantizer))

def test_currency_of_raises_on_non_string_code():
    from your_module import Currency, CurrencyType, ProgrammingError
    try:
        Currency.of(123, "US Dollar", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_on_non_alphabetic_code():
    from your_module import Currency, CurrencyType, ProgrammingError
    try:
        Currency.of("US1", "US Dollar", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_on_non_uppercase_code():
    from your_module import Currency, CurrencyType, ProgrammingError
    try:
        Currency.of("usd", "US Dollar", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_on_non_string_name():
    from your_module import Currency, CurrencyType, ProgrammingError
    try:
        Currency.of("USD", 123, 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_on_empty_name():
    from your_module import Currency, CurrencyType, ProgrammingError
    try:
        Currency.of("USD", "", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_on_name_with_leading_space():
    from your_module import Currency, CurrencyType, ProgrammingError
    try:
        Currency.of("USD", " US Dollar", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_on_name_with_trailing_space():
    from your_module import Currency, CurrencyType, ProgrammingError
    try:
        Currency.of("USD", "US Dollar ", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_on_non_integer_decimals():
    from your_module import Currency, CurrencyType, ProgrammingError
    try:
        Currency.of("USD", "US Dollar", "2", CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_on_decimals_less_than_minus_one():
    from your_module import Currency, CurrencyType, ProgrammingError
    try:
        Currency.of("USD", "US Dollar", -2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_on_non_currencytype_type():
    from your_module import Currency, CurrencyType, ProgrammingError
    try:
        Currency.of("USD", "US Dollar", 2, "MONEY")
        assert False
    except ProgrammingError:
        assert True

def test_currency_equality():
    from your_module import Currency, CurrencyType
    ccy1 = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    assert ccy1 == ccy2
    assert hash(ccy1) == hash(ccy2)

def test_currency_inequality_due_to_name():
    from your_module import Currency, CurrencyType
    ccy1 = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("USD", "UX Dollar", 2, CurrencyType.MONEY)
    assert ccy1 != ccy2
    assert hash(ccy1) != hash(ccy2)

def test_currency_quantize_with_positive_decimals():
    from decimal import Decimal
    from your_module import Currency, CurrencyType
    ccy = Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY)
    assert ccy.quantize(Decimal("1.005")) == Decimal("1.00")
    assert ccy.quantize(Decimal("1.015")) == Decimal("1.02")

def test_currency_quantize_with_zero_decimals():
    from decimal import Decimal
    from your_module import Currency, CurrencyType
    ccy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert ccy.quantize(Decimal("0.5")) == Decimal("0")
    assert ccy.quantize(Decimal("1.5")) == Decimal("2")

def test_currency_quantize_with_negative_decimals():
    from decimal import Decimal
    from your_module import Currency, CurrencyType, MaxPrecisionQuantizer
    ccy = Currency.of("ZZZ", "Weird Crypto", -1, CurrencyType.CRYPTO)
    assert ccy.quantizer == MaxPrecisionQuantizer
    result1 = ccy.quantize(Decimal("1.0000000000005"))
    expected1 = Decimal("1.000000000000")
    assert result1 == expected1
    result2 = ccy.quantize(Decimal("1.0000000000015"))
    expected2 = Decimal("1.000000000002")
    assert result2 == expected2


# LLM-generated content at query #12
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

def test_currency_registry_constructor_context_closed_initially():
    registry = CurrencyRegistry()
    try:
        registry._CurrencyRegistry__register(Currency.of("TEST", "Test Currency", 2, CurrencyType.MONEY))
        assert False
    except ProgrammingError:
        assert True

def test_currency_registry_constructor_singleton_preserves_state():
    registry1 = CurrencyRegistry()
    with registry1 as register:
        register(Currency.of("USD", "US Dollar", 2, CurrencyType.MONEY))
    registry2 = CurrencyRegistry()
    assert len(registry2) == 1
    assert "USD" in registry2
    assert registry2["USD"].code == "USD"


# LLM-generated content at query #13
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


# LLM-generated content at query #14
#--------------------------

def test_ctx_open_initialized_to_false():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open == False


# LLM-generated content at query #15
#--------------------------

def test_currency_registry_constructor_singleton():
    instance1 = CurrencyRegistry()
    instance2 = CurrencyRegistry()
    assert instance1 is instance2

def test_currency_registry_constructor_initial_attributes():
    registry = CurrencyRegistry()
    assert len(registry) == 0
    assert registry.all == []
    assert registry.codes == []
    assert registry.codenames == []

def test_currency_registry_constructor_context_closed_initially():
    registry = CurrencyRegistry()
    with pytest.raises(ProgrammingError):
        registry._CurrencyRegistry__register(Currency.of("TEST", "Test Currency", 2, CurrencyType.MONEY))

def test_currency_registry_constructor_reinitialization_preserves_singleton():
    instance1 = CurrencyRegistry()
    instance1._CurrencyRegistry__registry["DUMMY"] = "dummy"
    instance2 = CurrencyRegistry()
    assert instance2._CurrencyRegistry__registry["DUMMY"] == "dummy"
    del instance1._CurrencyRegistry__registry["DUMMY"]


# LLM-generated content at query #16
#--------------------------

def test_currency_of_creates_valid_currency():
    from decimal import Decimal
    from your_module import Currency, CurrencyType
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.code == "USD"
    assert usd.name == "US Dollars"
    assert usd.decimals == 2
    assert usd.type == CurrencyType.MONEY
    assert usd.quantizer == Decimal('0.01')
    assert usd.hashcache == hash(("USD", "US Dollars", 2, CurrencyType.MONEY, Decimal('0.01')))

def test_currency_of_with_zero_decimals():
    from decimal import Decimal
    from your_module import Currency, CurrencyType
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy.code == "JPY"
    assert jpy.name == "Japanese Yen"
    assert jpy.decimals == 0
    assert jpy.type == CurrencyType.MONEY
    assert jpy.quantizer == Decimal('0')
    assert jpy.hashcache == hash(("JPY", "Japanese Yen", 0, CurrencyType.MONEY, Decimal('0')))

def test_currency_of_with_negative_decimals():
    from decimal import Decimal
    from your_module import Currency, CurrencyType, MaxPrecisionQuantizer
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert zzz.code == "ZZZ"
    assert zzz.name == "Some weird currency"
    assert zzz.decimals == -1
    assert zzz.type == CurrencyType.CRYPTO
    assert zzz.quantizer == MaxPrecisionQuantizer
    assert zzz.hashcache == hash(("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO, MaxPrecisionQuantizer))

def test_currency_of_raises_error_for_non_string_code():
    from your_module import Currency, CurrencyType, ProgrammingError
    try:
        Currency.of(123, "US Dollars", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_error_for_non_alpha_code():
    from your_module import Currency, CurrencyType, ProgrammingError
    try:
        Currency.of("US1", "US Dollars", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_error_for_non_uppercase_code():
    from your_module import Currency, CurrencyType, ProgrammingError
    try:
        Currency.of("usd", "US Dollars", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_error_for_non_string_name():
    from your_module import Currency, CurrencyType, ProgrammingError
    try:
        Currency.of("USD", 123, 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_error_for_empty_name():
    from your_module import Currency, CurrencyType, ProgrammingError
    try:
        Currency.of("USD", "", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_error_for_name_with_leading_space():
    from your_module import Currency, CurrencyType, ProgrammingError
    try:
        Currency.of("USD", " US Dollars", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_error_for_name_with_trailing_space():
    from your_module import Currency, CurrencyType, ProgrammingError
    try:
        Currency.of("USD", "US Dollars ", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_error_for_non_integer_decimals():
    from your_module import Currency, CurrencyType, ProgrammingError
    try:
        Currency.of("USD", "US Dollars", "2", CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_error_for_decimals_less_than_minus_one():
    from your_module import Currency, CurrencyType, ProgrammingError
    try:
        Currency.of("USD", "US Dollars", -2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_error_for_non_currencytype_type():
    from your_module import Currency, CurrencyType, ProgrammingError
    try:
        Currency.of("USD", "US Dollars", 2, "MONEY")
        assert False
    except ProgrammingError:
        assert True

def test_currency_equality():
    from your_module import Currency, CurrencyType
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert usd1 == usd2
    assert not (usd1 == usdx)

def test_currency_hash_equality():
    from your_module import Currency, CurrencyType
    usd1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usd2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    usdx = Currency.of("USD", "UX Dollars", 2, CurrencyType.MONEY)
    assert hash(usd1) == hash(usd2)
    assert hash(usd1) != hash(usdx)

def test_currency_quantize_with_positive_decimals():
    from decimal import Decimal
    from your_module import Currency, CurrencyType
    usd = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert usd.quantize(Decimal("1.005")) == Decimal('1.00')
    assert usd.quantize(Decimal("1.015")) == Decimal('1.02')

def test_currency_quantize_with_zero_decimals():
    from decimal import Decimal
    from your_module import Currency, CurrencyType
    jpy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert jpy.quantize(Decimal("0.5")) == Decimal('0')
    assert jpy.quantize(Decimal("1.5")) == Decimal('2')

def test_currency_quantize_with_negative_decimals():
    from decimal import Decimal
    from your_module import Currency, CurrencyType
    zzz = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert zzz.quantize(Decimal("1.0000000000005")) == Decimal('1.000000000000')
    assert zzz.quantize(Decimal("1.0000000000015")) == Decimal('1.000000000002')


# LLM-generated content at query #17
#--------------------------

def test_currency_of_creates_valid_currency():
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert USD.code == "USD"
    assert USD.name == "US Dollars"
    assert USD.decimals == 2
    assert USD.type == CurrencyType.MONEY
    assert USD.quantizer == make_quantizer(2)
    assert USD.hashcache == hash(("USD", "US Dollars", 2, CurrencyType.MONEY, make_quantizer(2)))

def test_currency_of_creates_currency_with_zero_decimals():
    JPY = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert JPY.code == "JPY"
    assert JPY.name == "Japanese Yen"
    assert JPY.decimals == 0
    assert JPY.type == CurrencyType.MONEY
    assert JPY.quantizer == ZERO

def test_currency_of_creates_currency_with_negative_decimals():
    ZZZ = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert ZZZ.code == "ZZZ"
    assert ZZZ.name == "Some weird currency"
    assert ZZZ.decimals == -1
    assert ZZZ.type == CurrencyType.CRYPTO
    assert ZZZ.quantizer == MaxPrecisionQuantizer

def test_currency_of_raises_error_for_non_string_code():
    try:
        Currency.of(123, "US Dollars", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError as e:
        assert True

def test_currency_of_raises_error_for_non_alphabetic_code():
    try:
        Currency.of("US1", "US Dollars", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError as e:
        assert True

def test_currency_of_raises_error_for_non_uppercase_code():
    try:
        Currency.of("usd", "US Dollars", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError as e:
        assert True

def test_currency_of_raises_error_for_non_string_name():
    try:
        Currency.of("USD", 123, 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError as e:
        assert True

def test_currency_of_raises_error_for_empty_name():
    try:
        Currency.of("USD", "", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError as e:
        assert True

def test_currency_of_raises_error_for_name_with_leading_space():
    try:
        Currency.of("USD", " US Dollars", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError as e:
        assert True

def test_currency_of_raises_error_for_name_with_trailing_space():
    try:
        Currency.of("USD", "US Dollars ", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError as e:
        assert True

def test_currency_of_raises_error_for_non_integer_decimals():
    try:
        Currency.of("USD", "US Dollars", "2", CurrencyType.MONEY)
        assert False
    except ProgrammingError as e:
        assert True

def test_currency_of_raises_error_for_decimals_less_than_minus_one():
    try:
        Currency.of("USD", "US Dollars", -2, CurrencyType.MONEY)
        assert False
    except ProgrammingError as e:
        assert True

def test_currency_of_raises_error_for_non_currencytype_type():
    try:
        Currency.of("USD", "US Dollars", 2, "MONEY")
        assert False
    except ProgrammingError as e:
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

def test_currency_quantize_positive_decimals():
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert USD.quantize(Decimal("1.005")) == Decimal("1.00")
    assert USD.quantize(Decimal("1.015")) == Decimal("1.02")

def test_currency_quantize_zero_decimals():
    JPY = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert JPY.quantize(Decimal("0.5")) == Decimal("0")
    assert JPY.quantize(Decimal("1.5")) == Decimal("2")

def test_currency_quantize_negative_decimals():
    ZZZ = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert ZZZ.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert ZZZ.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")


# LLM-generated content at query #18
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


# LLM-generated content at query #19
#--------------------------

def test_ctx_open_initialized_to_false():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open == False


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
    instance1 = CurrencyRegistry()
    instance1._CurrencyRegistry__ctx_open = True
    instance2 = CurrencyRegistry()
    assert instance2._CurrencyRegistry__ctx_open == True


# LLM-generated content at query #21
#--------------------------

def test_currency_of_creates_valid_currency():
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert USD.code == "USD"
    assert USD.name == "US Dollars"
    assert USD.decimals == 2
    assert USD.type == CurrencyType.MONEY
    assert USD.quantizer == make_quantizer(2)
    assert USD.hashcache == hash(("USD", "US Dollars", 2, CurrencyType.MONEY, make_quantizer(2)))

def test_currency_of_with_zero_decimals():
    JPY = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert JPY.code == "JPY"
    assert JPY.name == "Japanese Yen"
    assert JPY.decimals == 0
    assert JPY.type == CurrencyType.MONEY
    assert JPY.quantizer == ZERO

def test_currency_of_with_negative_decimals():
    ZZZ = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert ZZZ.code == "ZZZ"
    assert ZZZ.name == "Some weird currency"
    assert ZZZ.decimals == -1
    assert ZZZ.type == CurrencyType.CRYPTO
    assert ZZZ.quantizer == MaxPrecisionQuantizer

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
    USD = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert USD.quantize(Decimal("1.005")) == Decimal("1.00")
    assert USD.quantize(Decimal("1.015")) == Decimal("1.02")

def test_currency_quantize_with_zero_decimals():
    JPY = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert JPY.quantize(Decimal("0.5")) == Decimal("0")
    assert JPY.quantize(Decimal("1.5")) == Decimal("2")

def test_currency_quantize_with_negative_decimals():
    ZZZ = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert ZZZ.quantize(Decimal("1.0000000000005")) == Decimal("1.000000000000")
    assert ZZZ.quantize(Decimal("1.0000000000015")) == Decimal("1.000000000002")


# LLM-generated content at query #22
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
    assert registry._CurrencyRegistry__ctx_open is False

def test_constructor_singleton_preserves_state():
    instance1 = CurrencyRegistry()
    instance1._CurrencyRegistry__ctx_open = True
    instance2 = CurrencyRegistry()
    assert instance2._CurrencyRegistry__ctx_open is True


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
    instance1._CurrencyRegistry__registry = {"test": "dummy"}
    instance2 = CurrencyRegistry()
    assert instance2._CurrencyRegistry__registry == {"test": "dummy"}


# LLM-generated content at query #25
#--------------------------

def test_constructor_creates_singleton():
    instance1 = CurrencyRegistry()
    instance2 = CurrencyRegistry()
    assert instance1 is instance2

def test_constructor_initializes_registry():
    registry = CurrencyRegistry()
    assert len(registry) == 0
    assert registry.all == []
    assert registry.codes == []
    assert registry.codenames == []

def test_constructor_initializes_context_flag():
    registry = CurrencyRegistry()
    with registry as register:
        pass
    assert registry._CurrencyRegistry__ctx_open == False


# LLM-generated content at query #26
#--------------------------

def test_ctx_open_initialized_to_false():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open == False


# LLM-generated content at query #27
#--------------------------

def test_currency_registry_singleton():
    instance1 = CurrencyRegistry()
    instance2 = CurrencyRegistry()
    assert instance1 is instance2

def test_currency_registry_initial_registry_empty():
    registry = CurrencyRegistry()
    assert len(registry) == 0

def test_currency_registry_initial_all_empty():
    registry = CurrencyRegistry()
    assert registry.all == []

def test_currency_registry_initial_codes_empty():
    registry = CurrencyRegistry()
    assert registry.codes == []

def test_currency_registry_initial_codenames_empty():
    registry = CurrencyRegistry()
    assert registry.codenames == []

def test_currency_registry_initial_ctx_open_false():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open is False

def test_currency_registry_initial_contains_false():
    registry = CurrencyRegistry()
    assert ("USD" in registry) is False

def test_currency_registry_initial_has_false():
    registry = CurrencyRegistry()
    assert registry.has("USD") is False

def test_currency_registry_initial_getitem_raises():
    registry = CurrencyRegistry()
    try:
        _ = registry["USD"]
        assert False
    except CurrencyLookupError:
        assert True

def test_currency_registry_initial_get_returns_none():
    registry = CurrencyRegistry()
    assert registry.get("USD") is None

def test_currency_registry_initial_get_with_default():
    registry = CurrencyRegistry()
    default_currency = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    result = registry.get("USD", default=default_currency)
    assert result is default_currency


# LLM-generated content at query #28
#--------------------------

def test_currency_of_creates_valid_instance():
    ccy = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert ccy.code == "USD"
    assert ccy.name == "US Dollars"
    assert ccy.decimals == 2
    assert ccy.type == CurrencyType.MONEY
    assert ccy.quantizer == make_quantizer(2)
    assert ccy.hashcache == hash(("USD", "US Dollars", 2, CurrencyType.MONEY, make_quantizer(2)))

def test_currency_of_raises_on_invalid_code_type():
    try:
        Currency.of(123, "US Dollars", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_on_code_not_alpha():
    try:
        Currency.of("US1", "US Dollars", 2, CurrencyType.MONEY)
        assert False
    except ProgrammingError:
        assert True

def test_currency_of_raises_on_code_not_upper():
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

def test_currency_of_with_zero_decimals():
    ccy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    assert ccy.decimals == 0
    assert ccy.quantizer == ZERO

def test_currency_of_with_negative_decimals():
    ccy = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    assert ccy.decimals == -1
    assert ccy.quantizer == MaxPrecisionQuantizer

def test_currency_of_positive_decimals_quantizer():
    ccy = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert ccy.quantizer == make_quantizer(2)

def test_currency_equality():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    assert ccy1 == ccy2
    assert hash(ccy1) == hash(ccy2)

def test_currency_inequality():
    ccy1 = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    ccy2 = Currency.of("EUR", "Euro", 2, CurrencyType.MONEY)
    assert ccy1 != ccy2
    assert hash(ccy1) != hash(ccy2)

def test_currency_quantize_positive_decimals():
    ccy = Currency.of("USD", "US Dollars", 2, CurrencyType.MONEY)
    result = ccy.quantize(Decimal("1.005"))
    assert result == Decimal("1.00")
    result = ccy.quantize(Decimal("1.015"))
    assert result == Decimal("1.02")

def test_currency_quantize_zero_decimals():
    ccy = Currency.of("JPY", "Japanese Yen", 0, CurrencyType.MONEY)
    result = ccy.quantize(Decimal("0.5"))
    assert result == Decimal("0")
    result = ccy.quantize(Decimal("1.5"))
    assert result == Decimal("2")

def test_currency_quantize_negative_decimals():
    ccy = Currency.of("ZZZ", "Some weird currency", -1, CurrencyType.CRYPTO)
    result = ccy.quantize(Decimal("1.0000000000005"))
    assert result == Decimal("1.000000000000")
    result = ccy.quantize(Decimal("1.0000000000015"))
    assert result == Decimal("1.000000000002")


# LLM-generated content at query #29
#--------------------------

def test_currency_registry_constructor_singleton():
    instance1 = CurrencyRegistry()
    instance2 = CurrencyRegistry()
    assert instance1 is instance2

def test_currency_registry_constructor_initialization():
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


# LLM-generated content at query #30
#--------------------------

def test_currency_registry_initialization():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__codes == []


# LLM-generated content at query #31
#--------------------------

def test_currency_registry_constructor_singleton():
    instance1 = CurrencyRegistry()
    instance2 = CurrencyRegistry()
    assert instance1 is instance2

def test_currency_registry_constructor_initial_attributes():
    registry = CurrencyRegistry()
    assert len(registry) == 0
    assert registry.all == []
    assert registry.codes == []
    assert registry.codenames == []

def test_currency_registry_constructor_context_closed_initially():
    registry = CurrencyRegistry()
    try:
        registry._CurrencyRegistry__register(Currency.of("TEST", "Test Currency", 2, CurrencyType.MONEY))
        assert False
    except ProgrammingError:
        pass

def test_currency_registry_constructor_private_registry_empty():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__registry == OrderedDict([])

def test_currency_registry_constructor_private_currencies_empty():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__currencies == []

def test_currency_registry_constructor_private_codes_empty():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__codes == []

def test_currency_registry_constructor_private_codenames_empty():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__codenames == []

def test_currency_registry_constructor_private_ctx_open_false():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open == False


# LLM-generated content at query #32
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

def test_currency_registry_constructor_context_closed_initially():
    registry = CurrencyRegistry()
    try:
        registry._CurrencyRegistry__register(Currency.of("TEST", "Test Currency", 2, CurrencyType.MONEY))
        assert False
    except ProgrammingError:
        assert True

def test_currency_registry_constructor_private_attributes_exist():
    registry = CurrencyRegistry()
    assert hasattr(registry, '_CurrencyRegistry__registry')
    assert hasattr(registry, '_CurrencyRegistry__currencies')
    assert hasattr(registry, '_CurrencyRegistry__codes')
    assert hasattr(registry, '_CurrencyRegistry__codenames')
    assert hasattr(registry, '_CurrencyRegistry__ctx_open')
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

def test_constructor_initializes_currencies_buffer():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__currencies == []

def test_constructor_initializes_codes_buffer():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__codes == []

def test_constructor_initializes_codenames_buffer():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__codenames == []

def test_constructor_initializes_context_flag_as_false():
    registry = CurrencyRegistry()
    assert registry._CurrencyRegistry__ctx_open == False


# LLM-generated content at query #34
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


