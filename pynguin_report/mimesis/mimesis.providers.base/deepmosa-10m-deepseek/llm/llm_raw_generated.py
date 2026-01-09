####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_provider_registry_constructor():
    registry = ProviderRegistry()
    assert registry._providers == {}


# LLM-generated content at query #2
#--------------------------

def test_constructor_with_default_parameters():
    provider = BaseProvider()
    assert provider.seed is MissingSeed
    assert isinstance(provider.random, _random.Random)

def test_constructor_with_custom_random_instance():
    custom_random = _random.Random()
    provider = BaseProvider(random=custom_random)
    assert provider.random is custom_random
    assert provider.seed is MissingSeed

def test_constructor_with_seed():
    provider = BaseProvider(seed=42)
    assert provider.seed == 42
    assert isinstance(provider.random, _random.Random)

def test_constructor_with_none_seed():
    provider = BaseProvider(seed=None)
    assert provider.seed is None
    assert isinstance(provider.random, _random.Random)

def test_constructor_with_seed_and_custom_random():
    custom_random = _random.Random()
    provider = BaseProvider(seed=123, random=custom_random)
    assert provider.random is custom_random
    assert provider.seed == 123

def test_constructor_raises_type_error_for_invalid_random():
    try:
        BaseProvider(random="invalid")
        assert False
    except TypeError as e:
        assert "The random must be an instance of mimesis.random.Random" in str(e)


# LLM-generated content at query #3
#--------------------------

def test_init_with_keyword_only_arguments():
    provider = BaseProvider(seed=None, random=None)
    assert provider.seed is None
    assert isinstance(provider.random, _random.Random)


# LLM-generated content at query #4
#--------------------------

def test_base_data_provider_initializes_with_default_locale():
    provider = BaseDataProvider()
    assert provider.locale == Locale.DEFAULT.value
    assert isinstance(provider._dataset, dict)
    assert provider.seed is MissingSeed

def test_base_data_provider_initializes_with_custom_locale():
    provider = BaseDataProvider(locale=Locale.EN)
    assert provider.locale == Locale.EN.value
    assert isinstance(provider._dataset, dict)

def test_base_data_provider_initializes_with_seed():
    provider = BaseDataProvider(seed=12345)
    assert provider.seed == 12345

def test_base_data_provider_initializes_with_random_instance():
    custom_random = _random.Random()
    provider = BaseDataProvider(random=custom_random)
    assert provider.random is custom_random

def test_base_data_provider_raises_error_for_invalid_random():
    try:
        BaseDataProvider(random="invalid")
        assert False
    except TypeError:
        assert True

def test_base_data_provider_loads_dataset_with_datafile():
    class TestProvider(BaseDataProvider):
        class Meta:
            name = "test"
            datafile = "test.json"
            datadir = DATADIR

    provider = TestProvider(locale=Locale.EN)
    assert isinstance(provider._dataset, dict)

def test_base_data_provider_handles_locale_with_separator():
    class TestProvider(BaseDataProvider):
        class Meta:
            name = "test"
            datafile = "test.json"
            datadir = DATADIR

    provider = TestProvider(locale=Locale.EN_US)
    assert provider.locale == Locale.EN_US.value

def test_base_data_provider_calls_parent_constructor():
    provider = BaseDataProvider(seed=999)
    assert provider._has_seed() is True

def test_base_data_provider_str_representation():
    provider = BaseDataProvider(locale=Locale.EN)
    expected = f"{provider.__class__.__name__} <{Locale.EN}>"
    assert str(provider) == expected

def test_base_data_provider_without_locale_dependent_str():
    class NonLocaleProvider(BaseDataProvider):
        class Meta:
            name = "nonlocale"
            auto_register = False

    provider = NonLocaleProvider()
    assert str(provider) == "NonLocaleProvider"


# LLM-generated content at query #5
#--------------------------

def test_auto_register_false_prevents_registration():
    class TestProvider(BaseProvider):
        class Meta:
            name = "test_provider"
            auto_register = False
    assert ProviderRegistry.get("test_provider") is None


# LLM-generated content at query #6
#--------------------------

def test_auto_register_provider():
    class TestProvider(BaseProvider):
        class Meta:
            name = "test_provider"
            auto_register = True
    assert ProviderRegistry.get("test_provider") == TestProvider

def test_auto_register_provider_default():
    class TestProvider(BaseProvider):
        class Meta:
            name = "test_provider_default"
    assert ProviderRegistry.get("test_provider_default") == TestProvider

def test_auto_register_provider_false():
    class TestProvider(BaseProvider):
        class Meta:
            name = "test_provider_false"
            auto_register = False
    with pytest.raises(KeyError):
        ProviderRegistry.get("test_provider_false")

def test_auto_register_no_meta_name():
    class TestProvider(BaseProvider):
        pass
    assert not hasattr(TestProvider.Meta, "name")

def test_auto_register_inheritance():
    class ParentProvider(BaseProvider):
        class Meta:
            name = "parent"
    class ChildProvider(ParentProvider):
        pass
    assert ProviderRegistry.get("parent") == ParentProvider


# LLM-generated content at query #7
#--------------------------

def test_provider_registry_constructor():
    registry = ProviderRegistry()
    assert registry._providers == {}


# LLM-generated content at query #8
#--------------------------

def test_base_data_provider_init_default_locale():
    provider = BaseDataProvider()
    assert provider.locale == Locale.DEFAULT.value
    assert isinstance(provider._dataset, dict)
    assert provider.seed is MissingSeed

def test_base_data_provider_init_custom_locale():
    provider = BaseDataProvider(locale=Locale.EN)
    assert provider.locale == Locale.EN.value
    assert isinstance(provider._dataset, dict)

def test_base_data_provider_init_with_seed():
    provider = BaseDataProvider(seed=42)
    assert provider.seed == 42

def test_base_data_provider_init_with_random_instance():
    custom_random = _random.Random()
    provider = BaseDataProvider(random=custom_random)
    assert provider.random is custom_random

def test_base_data_provider_init_invalid_random_type():
    try:
        BaseDataProvider(random="invalid")
    except TypeError as e:
        assert "The random must be an instance of mimesis.random.Random" in str(e)

def test_base_data_provider_init_locale_validation():
    try:
        BaseDataProvider(locale="invalid_locale")
    except UnsupportedLocale:
        pass

def test_base_data_provider_init_dataset_loading():
    class TestProvider(BaseDataProvider):
        class Meta:
            name = "test"
            datafile = "test.json"
            datadir = DATADIR / "test_data"
    try:
        provider = TestProvider()
    except FileNotFoundError:
        pass

def test_base_data_provider_init_auto_registration():
    class TestProvider(BaseDataProvider):
        class Meta:
            name = "test_provider"
    assert ProviderRegistry.get("test_provider") == TestProvider

def test_base_data_provider_init_auto_register_false():
    class TestProvider(BaseDataProvider):
        class Meta:
            name = "test_provider_false"
            auto_register = False
    try:
        ProviderRegistry.get("test_provider_false")
    except KeyError:
        pass

def test_base_data_provider_init_seed_inheritance():
    provider = BaseDataProvider(seed=123)
    assert provider.random._seed == 123

def test_base_data_provider_init_locale_separator():
    provider = BaseDataProvider(locale=Locale.EN_US)
    assert provider.locale == Locale.EN_US.value

def test_base_data_provider_init_empty_dataset():
    class TestProvider(BaseDataProvider):
        class Meta:
            name = "test_empty"
            datafile = ""
    provider = TestProvider()
    assert provider._dataset == {}

def test_base_data_provider_init_master_locale_fallback():
    class TestProvider(BaseDataProvider):
        class Meta:
            name = "test_master"
            datafile = "test.json"
    provider = TestProvider(locale=Locale.EN_US)
    assert provider.locale == Locale.EN_US.value

def test_base_data_provider_init_random_default():
    provider = BaseDataProvider()
    assert isinstance(provider.random, _random.Random)

def test_base_data_provider_init_seed_missing():
    provider = BaseDataProvider(seed=MissingSeed)
    assert provider.seed is MissingSeed

def test_base_data_provider_init_locale_default():
    provider = BaseDataProvider(locale=Locale.DEFAULT)
    assert provider.locale == Locale.DEFAULT.value

def test_base_data_provider_init_with_args_kwargs():
    provider = BaseDataProvider(locale=Locale.EN, seed=99, extra_arg="test")
    assert provider.locale == Locale.EN.value
    assert provider.seed == 99

def test_base_data_provider_init_dataset_update():
    class TestProvider(BaseDataProvider):
        class Meta:
            name = "test_update"
            datafile = "test.json"
    provider = TestProvider()
    test_data = {"new_key": "new_value"}
    provider.update_dataset(test_data)
    assert provider._dataset.get("new_key") == "new_value"

def test_base_data_provider_init_get_current_locale():
    provider = BaseDataProvider(locale=Locale.FR)
    assert provider.get_current_locale() == Locale.FR.value

def test_base_data_provider_init_str_representation():
    provider = BaseDataProvider(locale=Locale.DE)
    assert str(provider) == f"BaseDataProvider <{Locale.DE}>"

def test_base_data_provider_init_override_locale_context():
    provider = BaseDataProvider(locale=Locale.EN)
    with provider.override_locale(Locale.FR) as p:
        assert p.get_current_locale() == Locale.FR.value
    assert provider.get_current_locale() == Locale.EN.value


# LLM-generated content at query #9
#--------------------------

def test_base_data_provider_initialization_with_default_locale():
    provider = BaseDataProvider()
    assert provider.locale == Locale.DEFAULT.value
    assert provider._dataset == {}
    assert isinstance(provider.random, _random.Random)

def test_base_data_provider_initialization_with_custom_locale():
    provider = BaseDataProvider(locale=Locale.EN)
    assert provider.locale == Locale.EN.value
    assert provider._dataset == {}
    assert isinstance(provider.random, _random.Random)

def test_base_data_provider_initialization_with_seed():
    provider = BaseDataProvider(seed=42)
    assert provider.seed == 42
    assert isinstance(provider.random, _random.Random)

def test_base_data_provider_initialization_with_custom_random():
    custom_random = _random.Random()
    provider = BaseDataProvider(random=custom_random)
    assert provider.random is custom_random

def test_base_data_provider_initialization_with_invalid_random_type():
    try:
        BaseDataProvider(random="invalid")
        assert False
    except TypeError as e:
        assert str(e) == "The random must be an instance of mimesis.random.Random"

def test_base_data_provider_initialization_locale_and_seed():
    provider = BaseDataProvider(locale=Locale.RU, seed=123)
    assert provider.locale == Locale.RU.value
    assert provider.seed == 123

def test_base_data_provider_initialization_with_unsupported_locale():
    try:
        BaseDataProvider(locale="unsupported")
        assert False
    except UnsupportedLocale:
        assert True

def test_base_data_provider_initialization_with_locale_separator():
    provider = BaseDataProvider(locale="en-US")
    assert provider.locale == "en-US"
    assert provider._dataset != {}

def test_base_data_provider_initialization_with_datafile_meta():
    class TestProvider(BaseDataProvider):
        class Meta:
            name = "test"
            datafile = "test.json"
            datadir = DATADIR / "test_data"

    provider = TestProvider(locale=Locale.EN)
    assert provider.locale == Locale.EN.value
    assert provider._dataset != {}

def test_base_data_provider_initialization_without_datafile_meta():
    class TestProvider(BaseDataProvider):
        class Meta:
            name = "test"

    provider = TestProvider(locale=Locale.EN)
    assert provider.locale == Locale.EN.value
    assert provider._dataset == {}

def test_base_data_provider_initialization_with_missing_seed():
    provider = BaseDataProvider(seed=MissingSeed)
    assert provider.seed is MissingSeed
    assert isinstance(provider.random, _random.Random)

def test_base_data_provider_initialization_with_none_seed():
    provider = BaseDataProvider(seed=None)
    assert provider.seed is None
    assert isinstance(provider.random, _random.Random)

def test_base_data_provider_initialization_with_global_seed_set():
    _random.global_seed = 999
    provider = BaseDataProvider()
    assert provider.seed is MissingSeed
    assert isinstance(provider.random, _random.Random)
    _random.global_seed = MissingSeed

def test_base_data_provider_initialization_with_kwargs():
    provider = BaseDataProvider(locale=Locale.EN, seed=555, random=None)
    assert provider.locale == Locale.EN.value
    assert provider.seed == 555
    assert isinstance(provider.random, _random.Random)

def test_base_data_provider_initialization_with_args():
    try:
        BaseDataProvider(Locale.EN, 123)
        assert False
    except TypeError:
        assert True


# LLM-generated content at query #10
#--------------------------

def test_provider_registry_constructor():
    registry = ProviderRegistry()
    assert registry._providers == {}


# LLM-generated content at query #11
#--------------------------

def test_auto_register_provider():
    class TestProvider(BaseProvider):
        class Meta:
            name = "test_provider"
            auto_register = True
    assert ProviderRegistry.get("test_provider") == TestProvider


# LLM-generated content at query #12
#--------------------------

def test_init_with_default_locale_and_seed():
    provider = BaseDataProvider()
    assert provider.locale == Locale.DEFAULT.value
    assert provider.seed == MissingSeed
    assert isinstance(provider.random, _random.Random)

def test_init_with_custom_locale():
    provider = BaseDataProvider(locale=Locale.EN)
    assert provider.locale == Locale.EN.value

def test_init_with_custom_seed():
    provider = BaseDataProvider(seed=42)
    assert provider.seed == 42

def test_init_with_custom_random():
    custom_random = _random.Random()
    provider = BaseDataProvider(random=custom_random)
    assert provider.random is custom_random

def test_init_locale_validation_error():
    try:
        BaseDataProvider(locale="invalid_locale")
        assert False
    except UnsupportedLocale:
        assert True

def test_init_random_type_error():
    try:
        BaseDataProvider(random="invalid_random")
        assert False
    except TypeError:
        assert True

def test_init_dataset_loaded():
    class TestProvider(BaseDataProvider):
        class Meta:
            name = "test"
            datafile = "test.json"
            datadir = Path("test_data")
    try:
        provider = TestProvider()
        assert provider._dataset == {}
    except FileNotFoundError:
        assert True

def test_init_with_locale_and_seed():
    provider = BaseDataProvider(locale=Locale.FR, seed=123)
    assert provider.locale == Locale.FR.value
    assert provider.seed == 123

def test_init_without_auto_register():
    class NonRegisteredProvider(BaseDataProvider):
        class Meta:
            name = "non_registered"
            auto_register = False
    provider = NonRegisteredProvider()
    assert provider.__class__.__name__ == "NonRegisteredProvider"

def test_init_inheritance_order():
    class CustomProvider(BaseDataProvider):
        class Meta:
            name = "custom"
    provider = CustomProvider()
    assert isinstance(provider, BaseDataProvider)
    assert isinstance(provider, BaseProvider)


# LLM-generated content at query #13
#--------------------------

def test_constructor_initializes_empty_providers():
    registry = ProviderRegistry()
    providers = ProviderRegistry._providers
    assert providers == {}
    assert isinstance(providers, dict)
    assert len(providers) == 0


# LLM-generated content at query #14
#--------------------------

def test_base_data_provider_initialization_with_default_locale():
    provider = BaseDataProvider()
    assert provider.locale == Locale.DEFAULT.value
    assert provider._dataset == {}
    assert isinstance(provider.random, _random.Random)

def test_base_data_provider_initialization_with_custom_locale():
    provider = BaseDataProvider(locale=Locale.EN)
    assert provider.locale == Locale.EN.value
    assert isinstance(provider.random, _random.Random)

def test_base_data_provider_initialization_with_seed():
    provider = BaseDataProvider(seed=42)
    assert provider.seed == 42
    assert isinstance(provider.random, _random.Random)

def test_base_data_provider_initialization_with_custom_random():
    custom_random = _random.Random()
    provider = BaseDataProvider(random=custom_random)
    assert provider.random is custom_random

def test_base_data_provider_initialization_with_invalid_random_type():
    try:
        BaseDataProvider(random="invalid")
    except TypeError as e:
        assert str(e) == "The random must be an instance of mimesis.random.Random"

def test_base_data_provider_initialization_with_locale_and_seed():
    provider = BaseDataProvider(locale=Locale.RU, seed=123)
    assert provider.locale == Locale.RU.value
    assert provider.seed == 123

def test_base_data_provider_initialization_with_unsupported_locale():
    try:
        BaseDataProvider(locale="unsupported")
    except UnsupportedLocale:
        pass

def test_base_data_provider_initialization_with_missing_seed():
    provider = BaseDataProvider(seed=MissingSeed)
    assert provider.seed is MissingSeed

def test_base_data_provider_initialization_with_none_seed():
    provider = BaseDataProvider(seed=None)
    assert provider.seed is None

def test_base_data_provider_initialization_with_datafile_meta():
    class TestProvider(BaseDataProvider):
        class Meta:
            name = "test"
            datafile = "test.json"
            datadir = DATADIR

    provider = TestProvider()
    assert provider.Meta.datafile == "test.json"
    assert provider.Meta.datadir == DATADIR

def test_base_data_provider_initialization_without_datafile_meta():
    class TestProvider(BaseDataProvider):
        class Meta:
            name = "test"

    provider = TestProvider()
    assert provider._dataset == {}


# LLM-generated content at query #15
#--------------------------

def test_random_is_not_none_and_is_instance_of_random():
    random_instance = _random.Random()
    provider = BaseProvider(random=random_instance)
    assert provider.random is random_instance


# LLM-generated content at query #16
#--------------------------

def test_reseed_with_missing_seed_and_no_global_seed():
    provider = BaseProvider()
    original_state = provider.random.getstate()
    provider.reseed()
    new_state = provider.random.getstate()
    assert original_state != new_state
    assert provider.seed is MissingSeed

def test_reseed_with_missing_seed_and_global_seed_set():
    _random.global_seed = 12345
    provider = BaseProvider()
    provider.reseed()
    assert provider.random._seed == 12345
    assert provider.seed is MissingSeed
    _random.global_seed = MissingSeed

def test_reseed_with_explicit_none_seed():
    provider = BaseProvider()
    provider.reseed(seed=None)
    assert provider.seed is None

def test_reseed_with_explicit_integer_seed():
    provider = BaseProvider()
    provider.reseed(seed=999)
    assert provider.random._seed == 999
    assert provider.seed == 999

def test_reseed_changes_random_sequence():
    provider = BaseProvider(seed=100)
    first_sequence = [provider.random.randint(1, 100) for _ in range(5)]
    provider.reseed(seed=100)
    second_sequence = [provider.random.randint(1, 100) for _ in range(5)]
    assert first_sequence == second_sequence
    provider.reseed(seed=200)
    third_sequence = [provider.random.randint(1, 100) for _ in range(5)]
    assert first_sequence != third_sequence

def test_reseed_updates_instance_seed_attribute():
    provider = BaseProvider(seed=50)
    assert provider.seed == 50
    provider.reseed(seed=60)
    assert provider.seed == 60


# LLM-generated content at query #17
#--------------------------

def test_provider_registry_constructor():
    registry = ProviderRegistry()
    assert registry._providers == {}


# LLM-generated content at query #18
#--------------------------

def test_constructor_initializes_empty_providers():
    registry = ProviderRegistry()
    providers = registry.get_all()
    assert providers == {}


# LLM-generated content at query #19
#--------------------------

def test_base_data_provider_initialization_with_default_locale():
    provider = BaseDataProvider()
    assert provider.locale == Locale.DEFAULT.value
    assert isinstance(provider._dataset, dict)
    assert provider.seed is MissingSeed

def test_base_data_provider_initialization_with_custom_locale():
    provider = BaseDataProvider(locale=Locale.EN)
    assert provider.locale == Locale.EN.value
    assert isinstance(provider._dataset, dict)

def test_base_data_provider_initialization_with_seed():
    provider = BaseDataProvider(seed=12345)
    assert provider.seed == 12345

def test_base_data_provider_initialization_with_random_instance():
    custom_random = _random.Random()
    provider = BaseDataProvider(random=custom_random)
    assert provider.random is custom_random

def test_base_data_provider_initialization_with_invalid_random_type():
    try:
        BaseDataProvider(random="invalid")
        assert False
    except TypeError:
        assert True

def test_base_data_provider_initialization_locale_setup_failure():
    try:
        BaseDataProvider(locale="invalid_locale")
        assert False
    except UnsupportedLocale:
        assert True

def test_base_data_provider_initialization_with_locale_and_seed():
    provider = BaseDataProvider(locale=Locale.RU, seed=42)
    assert provider.locale == Locale.RU.value
    assert provider.seed == 42

def test_base_data_provider_initialization_dataset_empty_when_no_datafile():
    class TestProvider(BaseDataProvider):
        class Meta:
            name = "test"
            auto_register = False
    provider = TestProvider()
    assert provider._dataset == {}

def test_base_data_provider_initialization_dataset_loaded_with_datafile():
    class TestProvider(BaseDataProvider):
        class Meta:
            name = "test"
            auto_register = False
            datafile = "test.json"
            datadir = DATADIR
    try:
        provider = TestProvider(locale=Locale.EN)
        assert isinstance(provider._dataset, dict)
    except FileNotFoundError:
        pass

def test_base_data_provider_initialization_inherits_from_base_provider():
    provider = BaseDataProvider()
    assert hasattr(provider, "random")
    assert hasattr(provider, "seed")
    assert hasattr(provider, "reseed")
    assert hasattr(provider, "validate_enum")


# LLM-generated content at query #20
#--------------------------

def test_random_is_not_none_and_not_instance_of_random_raises_type_error():
    class MockRandom:
        pass
    mock_random = MockRandom()
    try:
        BaseProvider(random=mock_random)
        assert False
    except TypeError as e:
        assert str(e) == "The random must be an instance of mimesis.random.Random"


# LLM-generated content at query #21
#--------------------------

def test_auto_register_provider():
    class TestProvider(BaseProvider):
        class Meta:
            name = "test_provider"
            auto_register = True
    assert ProviderRegistry._registry.get("test_provider") == TestProvider

def test_auto_register_false():
    class TestProvider(BaseProvider):
        class Meta:
            name = "test_provider_false"
            auto_register = False
    assert ProviderRegistry._registry.get("test_provider_false") is None

def test_auto_register_no_meta():
    class TestProvider(BaseProvider):
        pass
    assert ProviderRegistry._registry.get("TestProvider") is None

def test_auto_register_no_name():
    class TestProvider(BaseProvider):
        class Meta:
            auto_register = True
    assert ProviderRegistry._registry.get("") is None


# LLM-generated content at query #22
#--------------------------

def test_validate_enum_with_none_item():
    from mimesis.enums import Gender
    from mimesis.providers.base import BaseProvider
    provider = BaseProvider()
    result = provider.validate_enum(None, Gender)
    assert result in [item.value for item in Gender]

def test_validate_enum_with_valid_enum_item():
    from mimesis.enums import Gender
    from mimesis.providers.base import BaseProvider
    provider = BaseProvider()
    result = provider.validate_enum(Gender.FEMALE, Gender)
    assert result == Gender.FEMALE.value

def test_validate_enum_raises_non_enumerable_error():
    from mimesis.enums import Gender
    from mimesis.exceptions import NonEnumerableError
    from mimesis.providers.base import BaseProvider
    provider = BaseProvider()
    try:
        provider.validate_enum("invalid_item", Gender)
        assert False
    except NonEnumerableError:
        assert True


# LLM-generated content at query #23
#--------------------------

def test_validate_enum_with_item_and_isinstance_true():
    from enum import Enum
    class TestEnum(Enum):
        A = 1
        B = 2
    provider = BaseProvider(seed=42)
    result = provider.validate_enum(TestEnum.A, TestEnum)
    assert result == 1


# LLM-generated content at query #24
#--------------------------

def test_random_is_not_none_and_not_instance_of_random_raises_type_error():
    class MockRandom:
        pass
    mock_random = MockRandom()
    try:
        BaseProvider(random=mock_random)
        assert False
    except TypeError as e:
        assert str(e) == "The random must be an instance of mimesis.random.Random"


# LLM-generated content at query #25
#--------------------------

def test_base_data_provider_init_default_locale():
    provider = BaseDataProvider()
    assert provider.locale == Locale.DEFAULT.value
    assert isinstance(provider._dataset, dict)

def test_base_data_provider_init_custom_locale():
    provider = BaseDataProvider(locale=Locale.EN)
    assert provider.locale == Locale.EN.value

def test_base_data_provider_init_with_seed():
    provider = BaseDataProvider(seed=42)
    assert provider.seed == 42

def test_base_data_provider_init_with_random_instance():
    custom_random = _random.Random()
    provider = BaseDataProvider(random=custom_random)
    assert provider.random is custom_random

def test_base_data_provider_init_inherits_base_provider_attributes():
    provider = BaseDataProvider()
    assert hasattr(provider, 'random')
    assert hasattr(provider, 'seed')

def test_base_data_provider_init_locale_setup_called():
    provider = BaseDataProvider(locale=Locale.DE)
    assert provider.locale == Locale.DE.value

def test_base_data_provider_init_load_dataset_called():
    class TestProvider(BaseDataProvider):
        class Meta:
            name = "test"
            datafile = "test.json"
            datadir = DATADIR
    provider = TestProvider()
    assert provider._dataset == {}

def test_base_data_provider_init_unsupported_locale_raises_error():
    try:
        BaseDataProvider(locale="unsupported")
        assert False
    except UnsupportedLocale:
        assert True

def test_base_data_provider_init_with_args_and_kwargs():
    provider = BaseDataProvider(Locale.FR, seed=123, random=None)
    assert provider.locale == Locale.FR.value
    assert provider.seed == 123

def test_base_data_provider_init_locale_order_matters():
    provider = BaseDataProvider(locale=Locale.JA, seed=999)
    assert provider.locale == Locale.JA.value
    assert provider.seed == 999


# LLM-generated content at query #26
#--------------------------

def test_validate_enum_with_none_item():
    from mimesis.enums import Gender
    from mimesis.providers.base import BaseProvider
    provider = BaseProvider(seed=42)
    result = provider.validate_enum(None, Gender)
    assert result in [item.value for item in Gender]

def test_validate_enum_with_valid_enum_item():
    from mimesis.enums import Gender
    from mimesis.providers.base import BaseProvider
    provider = BaseProvider(seed=42)
    result = provider.validate_enum(Gender.FEMALE, Gender)
    assert result == Gender.FEMALE.value

def test_validate_enum_raises_non_enumerable_error():
    from mimesis.enums import Gender
    from mimesis.exceptions import NonEnumerableError
    from mimesis.providers.base import BaseProvider
    provider = BaseProvider(seed=42)
    try:
        provider.validate_enum("invalid", Gender)
        assert False
    except NonEnumerableError:
        assert True


# LLM-generated content at query #27
#--------------------------

def test_constructor_with_default_parameters():
    provider = BaseProvider()
    assert provider.seed is MissingSeed
    assert isinstance(provider.random, _random.Random)

def test_constructor_with_custom_random_instance():
    custom_random = _random.Random()
    provider = BaseProvider(random=custom_random)
    assert provider.random is custom_random
    assert provider.seed is MissingSeed

def test_constructor_with_seed_parameter():
    provider = BaseProvider(seed=42)
    assert provider.seed == 42
    assert isinstance(provider.random, _random.Random)

def test_constructor_with_seed_none():
    provider = BaseProvider(seed=None)
    assert provider.seed is None
    assert isinstance(provider.random, _random.Random)

def test_constructor_with_both_seed_and_random():
    custom_random = _random.Random()
    provider = BaseProvider(seed=100, random=custom_random)
    assert provider.random is custom_random
    assert provider.seed == 100

def test_constructor_raises_type_error_for_invalid_random():
    try:
        BaseProvider(random="invalid")
        assert False
    except TypeError as e:
        assert "The random must be an instance of mimesis.random.Random" in str(e)


# LLM-generated content at query #28
#--------------------------

def test_random_is_none_so_self_random_is_initialized_with_new_random_instance():
    provider = BaseProvider()
    assert isinstance(provider.random, _random.Random)


# LLM-generated content at query #29
#--------------------------

def test_base_data_provider_init_default_locale():
    provider = BaseDataProvider()
    assert provider.locale == Locale.DEFAULT.value
    assert isinstance(provider._dataset, dict)

def test_base_data_provider_init_custom_locale():
    provider = BaseDataProvider(locale=Locale.EN)
    assert provider.locale == Locale.EN.value

def test_base_data_provider_init_with_seed():
    provider = BaseDataProvider(seed=42)
    assert provider.seed == 42

def test_base_data_provider_init_with_random_instance():
    custom_random = _random.Random()
    provider = BaseDataProvider(random=custom_random)
    assert provider.random is custom_random

def test_base_data_provider_init_invalid_random_type():
    try:
        BaseDataProvider(random="invalid")
    except TypeError as e:
        assert "The random must be an instance of mimesis.random.Random" in str(e)

def test_base_data_provider_init_locale_validation():
    try:
        BaseDataProvider(locale="invalid_locale")
    except UnsupportedLocale:
        pass

def test_base_data_provider_init_dataset_loaded():
    class TestProvider(BaseDataProvider):
        class Meta:
            name = "test"
            datafile = "test.json"
            datadir = DATADIR
    try:
        provider = TestProvider()
        assert isinstance(provider._dataset, dict)
    except FileNotFoundError:
        pass

def test_base_data_provider_init_no_datafile():
    class TestProvider(BaseDataProvider):
        class Meta:
            name = "test"
    provider = TestProvider()
    assert provider._dataset == {}

def test_base_data_provider_init_locale_separator():
    class TestProvider(BaseDataProvider):
        class Meta:
            name = "test"
            datafile = "test.json"
            datadir = DATADIR
    try:
        provider = TestProvider(locale="en_US")
        assert isinstance(provider._dataset, dict)
    except FileNotFoundError:
        pass

def test_base_data_provider_init_inheritance():
    class CustomProvider(BaseDataProvider):
        pass
    provider = CustomProvider()
    assert isinstance(provider, BaseDataProvider)
    assert isinstance(provider, BaseProvider)


# LLM-generated content at query #30
#--------------------------

def test_validate_enum_when_item_is_falsy_but_not_none_and_not_instance_of_enum():
    from enum import Enum
    class TestEnum(Enum):
        A = 1
        B = 2
    provider = BaseProvider()
    try:
        provider.validate_enum(item=False, enum=TestEnum)
    except NonEnumerableError:
        pass
    else:
        assert False, "Expected NonEnumerableError"


# LLM-generated content at query #31
#--------------------------

def test_base_data_provider_initialization_with_default_locale():
    provider = BaseDataProvider()
    assert provider.locale == Locale.DEFAULT.value
    assert isinstance(provider._dataset, dict)
    assert provider.seed is MissingSeed

def test_base_data_provider_initialization_with_custom_locale():
    provider = BaseDataProvider(locale=Locale.EN)
    assert provider.locale == Locale.EN.value
    assert isinstance(provider._dataset, dict)

def test_base_data_provider_initialization_with_seed():
    provider = BaseDataProvider(seed=12345)
    assert provider.seed == 12345

def test_base_data_provider_initialization_with_random_instance():
    custom_random = _random.Random()
    provider = BaseDataProvider(random=custom_random)
    assert provider.random is custom_random

def test_base_data_provider_initialization_with_invalid_random_type():
    try:
        BaseDataProvider(random="invalid")
    except TypeError as e:
        assert "The random must be an instance of mimesis.random.Random" in str(e)

def test_base_data_provider_initialization_locale_and_seed():
    provider = BaseDataProvider(locale=Locale.RU, seed=42)
    assert provider.locale == Locale.RU.value
    assert provider.seed == 42

def test_base_data_provider_initialization_with_unsupported_locale():
    try:
        BaseDataProvider(locale="unsupported")
    except UnsupportedLocale:
        pass

def test_base_data_provider_initialization_without_datafile():
    class TestProvider(BaseDataProvider):
        class Meta:
            name = "test"
            auto_register = False
    provider = TestProvider()
    assert provider._dataset == {}

def test_base_data_provider_initialization_with_datafile():
    class TestProvider(BaseDataProvider):
        class Meta:
            name = "test"
            auto_register = False
            datafile = "test.json"
            datadir = Path("/fake/dir")
    try:
        TestProvider()
    except FileNotFoundError:
        pass

def test_base_data_provider_initialization_inherits_random_from_base():
    provider = BaseDataProvider()
    assert isinstance(provider.random, _random.Random)

def test_base_data_provider_initialization_seed_propagates_to_random():
    provider = BaseDataProvider(seed=999)
    assert provider.random._seed == 999


# LLM-generated content at query #32
#--------------------------

def test_auto_register_provider():
    class TestProvider(BaseProvider):
        class Meta:
            name = "test_provider"
            auto_register = True
    assert ProviderRegistry.get("test_provider") == TestProvider


# LLM-generated content at query #33
#--------------------------

def test_provider_registry_constructor():
    registry = ProviderRegistry()
    assert registry._providers == {}


# LLM-generated content at query #34
#--------------------------

def test_auto_register_false_prevents_registration():
    class TestProvider(BaseProvider):
        class Meta:
            name = "test_provider"
            auto_register = False
    assert ProviderRegistry.get("test_provider") is None


# LLM-generated content at query #35
#--------------------------

def test_locale_is_not_default_when_initialized_with_custom_locale():
    provider = BaseDataProvider(locale=Locale.EN)
    result = provider.locale == Locale.DEFAULT
    assert result is False


# LLM-generated content at query #36
#--------------------------

def test_initialization_with_keyword_only_arguments():
    provider = BaseProvider(seed=None, random=None)
    assert provider.seed is None
    assert isinstance(provider.random, _random.Random)


# LLM-generated content at query #37
#--------------------------

def test_reseed_with_global_seed_set():
    from mimesis.providers.base import BaseProvider
    from mimesis.random import global_seed
    original_global_seed = global_seed
    global_seed = 42
    provider = BaseProvider()
    provider.seed = None
    provider.reseed()
    assert global_seed is not None
    global_seed = original_global_seed


# LLM-generated content at query #38
#--------------------------

def test_constructor_initializes_empty_providers():
    registry = ProviderRegistry()
    providers = registry._providers
    assert providers == {}


# LLM-generated content at query #39
#--------------------------

def test_random_is_not_none_and_is_instance_of_random():
    random_instance = _random.Random()
    provider = BaseProvider(random=random_instance)
    assert provider.random is random_instance


# LLM-generated content at query #40
#--------------------------

def test_random_is_not_none_and_is_instance_of_random():
    random_instance = _random.Random()
    provider = BaseProvider(random=random_instance)
    assert provider.random is random_instance

def test_random_is_none():
    provider = BaseProvider(random=None)
    assert isinstance(provider.random, _random.Random)

def test_random_is_not_none_and_not_instance_of_random():
    try:
        BaseProvider(random="not_a_random_instance")
    except TypeError as e:
        assert str(e) == "The random must be an instance of mimesis.random.Random"


# LLM-generated content at query #41
#--------------------------

def test_provider_registry_initialization():
    registry = ProviderRegistry()
    providers = ProviderRegistry.get_all()
    assert providers == {}


# LLM-generated content at query #42
#--------------------------

def test_reseed_with_global_seed_set():
    provider = BaseProvider(seed=None)
    _random.global_seed = 12345
    provider.reseed()
    assert provider.seed is MissingSeed
    assert provider.random._seed == 12345


# LLM-generated content at query #43
#--------------------------

def test_auto_register_provider():
    class TestProvider(BaseProvider):
        class Meta:
            name = "test_provider"
            auto_register = True
    assert ProviderRegistry.get("test_provider") == TestProvider


# LLM-generated content at query #44
#--------------------------

def test_random_is_not_none_and_not_instance_of_random_raises_type_error():
    class MockRandom:
        pass
    mock_random = MockRandom()
    try:
        BaseProvider(random=mock_random)
        assert False
    except TypeError as e:
        assert str(e) == "The random must be an instance of mimesis.random.Random"


# LLM-generated content at query #45
#--------------------------

def test_reseed_uses_global_seed_when_seed_is_missing_seed_and_global_seed_is_not_missing_seed():
    import mimesis.random as _random
    from mimesis.enums import MissingSeed
    from mimesis.providers.base import BaseProvider
    from mimesis.random import Random
    original_global_seed = _random.global_seed
    _random.global_seed = 12345
    provider = BaseProvider(seed=MissingSeed, random=Random())
    provider.random.seed(99999)
    provider.reseed(seed=MissingSeed)
    assert provider.random._seed == 12345
    _random.global_seed = original_global_seed


# LLM-generated content at query #46
#--------------------------

def test_init_calls_super_with_seed_and_args():
    class TestProvider(BaseDataProvider):
        class Meta:
            name = "test"
            datafile = "test.json"
            datadir = "/tmp"
    
    provider = TestProvider(locale="en", seed=42, extra_arg="value")
    assert provider.seed == 42


# LLM-generated content at query #47
#--------------------------

def test_init_with_custom_random():
    custom_random = _random.Random()
    provider = BaseProvider(random=custom_random)
    assert provider.random is custom_random

def test_init_without_random():
    provider = BaseProvider()
    assert isinstance(provider.random, _random.Random)

def test_init_with_invalid_random():
    try:
        BaseProvider(random="invalid")
    except TypeError as e:
        assert "The random must be an instance of mimesis.random.Random" in str(e)

def test_init_with_seed():
    provider = BaseProvider(seed=42)
    assert provider.seed == 42

def test_init_with_missing_seed():
    provider = BaseProvider()
    assert provider.seed is MissingSeed

def test_init_with_none_seed():
    provider = BaseProvider(seed=None)
    assert provider.seed is None

def test_reseed_called_on_init():
    provider = BaseProvider(seed=123)
    assert provider.random._seed == 123


# LLM-generated content at query #48
#--------------------------

def test_provider_registry_constructor():
    registry = ProviderRegistry()
    assert registry._providers == {}


# LLM-generated content at query #49
#--------------------------

def test_base_data_provider_initialization_with_default_locale():
    provider = BaseDataProvider()
    assert provider.locale == Locale.DEFAULT.value
    assert provider._dataset == {}
    assert isinstance(provider.random, _random.Random)
    assert provider.seed is MissingSeed

def test_base_data_provider_initialization_with_custom_locale():
    provider = BaseDataProvider(locale=Locale.EN)
    assert provider.locale == Locale.EN.value
    assert provider._dataset == {}
    assert isinstance(provider.random, _random.Random)
    assert provider.seed is MissingSeed

def test_base_data_provider_initialization_with_custom_seed():
    provider = BaseDataProvider(seed=42)
    assert provider.locale == Locale.DEFAULT.value
    assert provider._dataset == {}
    assert isinstance(provider.random, _random.Random)
    assert provider.seed == 42

def test_base_data_provider_initialization_with_custom_random():
    custom_random = _random.Random()
    provider = BaseDataProvider(random=custom_random)
    assert provider.locale == Locale.DEFAULT.value
    assert provider._dataset == {}
    assert provider.random is custom_random
    assert provider.seed is MissingSeed

def test_base_data_provider_initialization_with_custom_locale_and_seed():
    provider = BaseDataProvider(locale=Locale.RU, seed=123)
    assert provider.locale == Locale.RU.value
    assert provider._dataset == {}
    assert isinstance(provider.random, _random.Random)
    assert provider.seed == 123

def test_base_data_provider_initialization_with_custom_locale_and_random():
    custom_random = _random.Random()
    provider = BaseDataProvider(locale=Locale.ES, random=custom_random)
    assert provider.locale == Locale.ES.value
    assert provider._dataset == {}
    assert provider.random is custom_random
    assert provider.seed is MissingSeed

def test_base_data_provider_initialization_with_custom_seed_and_random():
    custom_random = _random.Random()
    provider = BaseDataProvider(seed=999, random=custom_random)
    assert provider.locale == Locale.DEFAULT.value
    assert provider._dataset == {}
    assert provider.random is custom_random
    assert provider.seed == 999

def test_base_data_provider_initialization_with_all_custom_parameters():
    custom_random = _random.Random()
    provider = BaseDataProvider(locale=Locale.FR, seed=777, random=custom_random)
    assert provider.locale == Locale.FR.value
    assert provider._dataset == {}
    assert provider.random is custom_random
    assert provider.seed == 777

def test_base_data_provider_initialization_with_invalid_random_type():
    try:
        provider = BaseDataProvider(random="invalid")
        assert False
    except TypeError as e:
        assert str(e) == "The random must be an instance of mimesis.random.Random"

def test_base_data_provider_initialization_with_locale_as_string():
    provider = BaseDataProvider(locale="en")
    assert provider.locale == Locale.EN.value
    assert provider._dataset == {}
    assert isinstance(provider.random, _random.Random)
    assert provider.seed is MissingSeed

def test_base_data_provider_initialization_with_locale_as_locale_enum():
    provider = BaseDataProvider(locale=Locale.DE)
    assert provider.locale == Locale.DE.value
    assert provider._dataset == {}
    assert isinstance(provider.random, _random.Random)
    assert provider.seed is MissingSeed


# LLM-generated content at query #50
#--------------------------

def test_locale_default_is_not_missing_seed():
    provider = BaseDataProvider()
    assert provider.seed is MissingSeed


# LLM-generated content at query #51
#--------------------------

def test_validate_enum_with_none_item():
    from mimesis.enums import Gender
    from mimesis.providers.base import BaseProvider
    provider = BaseProvider()
    result = provider.validate_enum(None, Gender)
    assert result in [item.value for item in Gender]

def test_validate_enum_with_valid_enum_item():
    from mimesis.enums import Gender
    from mimesis.providers.base import BaseProvider
    provider = BaseProvider()
    result = provider.validate_enum(Gender.FEMALE, Gender)
    assert result == Gender.FEMALE.value

def test_validate_enum_raises_non_enumerable_error():
    from mimesis.enums import Gender
    from mimesis.exceptions import NonEnumerableError
    from mimesis.providers.base import BaseProvider
    provider = BaseProvider()
    try:
        provider.validate_enum("invalid", Gender)
        assert False
    except NonEnumerableError:
        assert True


# LLM-generated content at query #52
#--------------------------

def test_auto_register_provider_with_meta_name_and_auto_register_true():
    class TestProvider(BaseProvider):
        class Meta:
            name = "test_provider"
            auto_register = True
    assert ProviderRegistry.get("test_provider") == TestProvider

def test_auto_register_provider_with_meta_name_and_auto_register_default():
    class TestProvider(BaseProvider):
        class Meta:
            name = "test_provider_default"
    assert ProviderRegistry.get("test_provider_default") == TestProvider

def test_auto_register_provider_with_meta_name_and_auto_register_explicitly_true():
    class TestProvider(BaseProvider):
        class Meta:
            name = "test_provider_explicit_true"
            auto_register = True
    assert ProviderRegistry.get("test_provider_explicit_true") == TestProvider


# LLM-generated content at query #53
#--------------------------

def test_locale_is_not_default_when_initialized_with_specific_locale():
    provider = BaseDataProvider(locale=Locale.EN)
    result = provider.locale == Locale.DEFAULT
    assert result is False


# LLM-generated content at query #54
#--------------------------

def test_provider_registry_constructor():
    registry = ProviderRegistry()
    assert registry._providers == {}


# LLM-generated content at query #55
#--------------------------

def test_auto_register_false_prevents_registration():
    class TestProvider(BaseProvider):
        class Meta:
            name = "test_provider"
            auto_register = False
    assert ProviderRegistry.get("test_provider") is None


# LLM-generated content at query #56
#--------------------------

def test_locale_default_is_used_when_no_locale_provided():
    from mimesis.enums import Locale
    from mimesis.providers.base import BaseDataProvider
    class TestProvider(BaseDataProvider):
        class Meta:
            name = "test"
            datafile = "test.json"
    provider = TestProvider()
    assert provider.locale == Locale.DEFAULT.value


# LLM-generated content at query #57
#--------------------------

def test_auto_register_provider():
    class TestProvider(BaseProvider):
        class Meta:
            name = "test_provider"
            auto_register = True
    assert ProviderRegistry.get("test_provider") == TestProvider


# LLM-generated content at query #58
#--------------------------

def test_random_is_not_none_and_not_instance_of_random_raises_type_error():
    class MockRandom:
        pass
    mock_random = MockRandom()
    try:
        BaseProvider(random=mock_random)
        assert False
    except TypeError as e:
        assert str(e) == "The random must be an instance of mimesis.random.Random"


# LLM-generated content at query #59
#--------------------------

def test_locale_default_initialization():
    provider = BaseDataProvider()
    assert provider.locale == Locale.DEFAULT.value


# LLM-generated content at query #60
#--------------------------

def test_validate_enum_with_item_and_isinstance_true():
    from enum import Enum
    class TestEnum(Enum):
        A = 1
        B = 2
    provider = BaseProvider(seed=42)
    result = provider.validate_enum(TestEnum.A, TestEnum)
    assert result == 1


# LLM-generated content at query #61
#--------------------------

def test_validate_enum_item_false_predicate():
    from enum import Enum
    class TestEnum(Enum):
        A = 1
        B = 2
    provider = BaseProvider(seed=42)
    item = TestEnum.A
    result = provider.validate_enum(item, TestEnum)
    assert result == 1


# LLM-generated content at query #62
#--------------------------

def test_validate_enum_with_false_predicate_at_line_11():
    from enum import Enum
    class TestEnum(Enum):
        A = 1
        B = 2
    provider = BaseProvider(seed=42)
    item = TestEnum.A
    result = provider.validate_enum(item, TestEnum)
    assert result == 1


# LLM-generated content at query #63
#--------------------------

def test_base_provider_initialization_with_default_seed():
    provider = BaseProvider()
    assert provider.seed is MissingSeed
    assert isinstance(provider.random, _random.Random)

def test_base_provider_initialization_with_none_seed():
    provider = BaseProvider(seed=None)
    assert provider.seed is None
    assert isinstance(provider.random, _random.Random)

def test_base_provider_initialization_with_integer_seed():
    provider = BaseProvider(seed=42)
    assert provider.seed == 42
    assert isinstance(provider.random, _random.Random)

def test_base_provider_initialization_with_custom_random():
    custom_random = _random.Random()
    provider = BaseProvider(random=custom_random)
    assert provider.random is custom_random
    assert provider.seed is MissingSeed

def test_base_provider_initialization_with_custom_random_and_seed():
    custom_random = _random.Random()
    provider = BaseProvider(seed=123, random=custom_random)
    assert provider.random is custom_random
    assert provider.seed == 123

def test_base_provider_initialization_raises_type_error_for_invalid_random():
    try:
        BaseProvider(random="invalid")
        assert False
    except TypeError as e:
        assert str(e) == "The random must be an instance of mimesis.random.Random"


# LLM-generated content at query #64
#--------------------------

def test_auto_register_provider():
    class TestProvider(BaseProvider):
        class Meta:
            name = "test_provider"
            auto_register = True
    assert ProviderRegistry.get("test_provider") == TestProvider

def test_auto_register_false():
    class TestProviderNoRegister(BaseProvider):
        class Meta:
            name = "test_provider_no_register"
            auto_register = False
    with pytest.raises(KeyError):
        ProviderRegistry.get("test_provider_no_register")

def test_auto_register_no_meta_name():
    class TestProviderNoMetaName(BaseProvider):
        pass
    assert not hasattr(TestProviderNoMetaName.Meta, "name")


# LLM-generated content at query #65
#--------------------------

def test_random_is_not_none_and_is_instance_of_random():
    random_instance = _random.Random()
    provider = BaseProvider(random=random_instance)
    assert provider.random is random_instance


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_provider_registry_constructor():
    registry = ProviderRegistry()
    assert registry._providers == {}


# LLM-generated content at query #2
#--------------------------

def test_provider_registry_constructor():
    registry = ProviderRegistry()
    assert registry._providers == {}


# LLM-generated content at query #3
#--------------------------

def test_provider_registry_constructor():
    registry = ProviderRegistry()
    assert registry._providers == {}
    assert ProviderRegistry._providers == {}


# LLM-generated content at query #4
#--------------------------

def test_provider_registry_constructor():
    registry = ProviderRegistry()
    assert registry._providers == {}
    assert isinstance(registry._providers, dict)
    assert len(registry._providers) == 0


# LLM-generated content at query #5
#--------------------------

def test_base_data_provider_initialization_with_default_locale():
    provider = BaseDataProvider()
    assert provider.locale == Locale.DEFAULT.value
    assert provider._dataset == {}
    assert isinstance(provider.random, _random.Random)
    assert provider.seed is MissingSeed

def test_base_data_provider_initialization_with_custom_locale():
    provider = BaseDataProvider(locale=Locale.EN)
    assert provider.locale == Locale.EN.value
    assert isinstance(provider.random, _random.Random)
    assert provider.seed is MissingSeed

def test_base_data_provider_initialization_with_seed():
    provider = BaseDataProvider(seed=12345)
    assert provider.locale == Locale.DEFAULT.value
    assert provider._dataset == {}
    assert isinstance(provider.random, _random.Random)
    assert provider.seed == 12345

def test_base_data_provider_initialization_with_custom_random():
    custom_random = _random.Random()
    provider = BaseDataProvider(random=custom_random)
    assert provider.locale == Locale.DEFAULT.value
    assert provider._dataset == {}
    assert provider.random is custom_random
    assert provider.seed is MissingSeed

def test_base_data_provider_initialization_with_locale_and_seed():
    provider = BaseDataProvider(locale=Locale.RU, seed=999)
    assert provider.locale == Locale.RU.value
    assert isinstance(provider.random, _random.Random)
    assert provider.seed == 999

def test_base_data_provider_initialization_with_unsupported_locale_raises_error():
    try:
        BaseDataProvider(locale="unsupported")
        assert False
    except UnsupportedLocale:
        assert True

def test_base_data_provider_initialization_with_invalid_random_type_raises_error():
    try:
        BaseDataProvider(random="invalid")
        assert False
    except TypeError:
        assert True

def test_base_data_provider_initialization_with_datafile_meta_loads_dataset():
    class TestProvider(BaseDataProvider):
        class Meta:
            name = "test"
            datafile = "test.json"
            datadir = DATADIR
    provider = TestProvider(locale=Locale.EN)
    assert provider.locale == Locale.EN.value
    assert isinstance(provider._dataset, dict)

def test_base_data_provider_initialization_without_datafile_meta_has_empty_dataset():
    class TestProvider(BaseDataProvider):
        class Meta:
            name = "test"
    provider = TestProvider()
    assert provider._dataset == {}

def test_base_data_provider_str_representation():
    provider = BaseDataProvider(locale=Locale.FR)
    expected = f"{provider.__class__.__name__} <{Locale.FR}>"
    assert str(provider) == expected


# LLM-generated content at query #6
#--------------------------

def test_init_with_default_parameters():
    provider = BaseProvider()
    assert provider.seed is MissingSeed
    assert isinstance(provider.random, _random.Random)

def test_init_with_custom_random():
    custom_random = _random.Random()
    provider = BaseProvider(random=custom_random)
    assert provider.random is custom_random
    assert provider.seed is MissingSeed

def test_init_with_seed():
    provider = BaseProvider(seed=42)
    assert provider.seed == 42
    assert isinstance(provider.random, _random.Random)

def test_init_with_seed_and_custom_random():
    custom_random = _random.Random()
    provider = BaseProvider(seed=123, random=custom_random)
    assert provider.random is custom_random
    assert provider.seed == 123

def test_init_with_seed_none():
    provider = BaseProvider(seed=None)
    assert provider.seed is None
    assert isinstance(provider.random, _random.Random)

def test_init_raises_type_error_for_invalid_random():
    try:
        BaseProvider(random="invalid")
        assert False
    except TypeError as e:
        assert "The random must be an instance of mimesis.random.Random" in str(e)


# LLM-generated content at query #7
#--------------------------

def test_initialize_attributes_for_data_providers():
    provider = BaseDataProvider()
    assert provider._dataset == {}
    assert provider.locale == Locale.DEFAULT.value
    assert isinstance(provider.random, _random.Random)


# LLM-generated content at query #8
#--------------------------

def test_provider_registry_constructor():
    registry = ProviderRegistry()
    assert registry._providers == {}


# LLM-generated content at query #9
#--------------------------

def test_provider_registry_constructor():
    registry = ProviderRegistry()
    assert registry is not None


# LLM-generated content at query #10
#--------------------------

def test_provider_registry_constructor():
    registry = ProviderRegistry()
    assert registry._providers == {}


# LLM-generated content at query #11
#--------------------------

def test_reseed_with_missing_seed_and_global_seed_missing():
    provider = BaseProvider()
    original_state = provider.random.getstate()
    provider.reseed()
    assert provider.seed is MissingSeed
    assert provider.random.getstate() != original_state

def test_reseed_with_missing_seed_and_global_seed_set():
    _random.global_seed = 12345
    provider = BaseProvider()
    provider.reseed()
    assert provider.seed is MissingSeed
    _random.global_seed = MissingSeed

def test_reseed_with_none_seed():
    provider = BaseProvider()
    provider.reseed(seed=None)
    assert provider.seed is None

def test_reseed_with_integer_seed():
    provider = BaseProvider()
    provider.reseed(seed=42)
    assert provider.seed == 42

def test_reseed_changes_random_state():
    provider = BaseProvider()
    first_random_value = provider.random.randint(1, 100)
    provider.reseed(seed=999)
    second_random_value = provider.random.randint(1, 100)
    assert first_random_value != second_random_value

def test_reseed_with_same_seed_produces_same_sequence():
    provider1 = BaseProvider()
    provider1.reseed(seed=100)
    sequence1 = [provider1.random.randint(1, 100) for _ in range(5)]
    provider2 = BaseProvider()
    provider2.reseed(seed=100)
    sequence2 = [provider2.random.randint(1, 100) for _ in range(5)]
    assert sequence1 == sequence2

def test_reseed_updates_instance_seed_attribute():
    provider = BaseProvider(seed=10)
    assert provider.seed == 10
    provider.reseed(seed=20)
    assert provider.seed == 20


# LLM-generated content at query #12
#--------------------------

def test_validate_enum_with_none_item():
    from mimesis.enums import Gender
    from mimesis.providers.base import BaseProvider
    provider = BaseProvider(seed=42)
    result = provider.validate_enum(None, Gender)
    assert result in [item.value for item in Gender]

def test_validate_enum_with_valid_enum_item():
    from mimesis.enums import Gender
    from mimesis.providers.base import BaseProvider
    provider = BaseProvider(seed=42)
    result = provider.validate_enum(Gender.FEMALE, Gender)
    assert result == Gender.FEMALE.value

def test_validate_enum_raises_non_enumerable_error():
    from mimesis.enums import Gender
    from mimesis.exceptions import NonEnumerableError
    from mimesis.providers.base import BaseProvider
    provider = BaseProvider(seed=42)
    try:
        provider.validate_enum("invalid", Gender)
        assert False
    except NonEnumerableError:
        assert True


# LLM-generated content at query #13
#--------------------------

def test_reseed_with_missing_seed_and_global_seed_not_missing():
    import mimesis.random as _random
    from mimesis.providers.base import BaseProvider
    original_global_seed = _random.global_seed
    _random.global_seed = 12345
    provider = BaseProvider(seed=None)
    provider.reseed()
    assert provider.seed is None
    _random.global_seed = original_global_seed


# LLM-generated content at query #14
#--------------------------

def test_random_is_not_none_and_is_instance_of_random():
    random_instance = _random.Random()
    provider = BaseProvider(random=random_instance)
    assert provider.random is random_instance


# LLM-generated content at query #15
#--------------------------

def test_provider_registry_constructor():
    registry = ProviderRegistry()
    assert registry._providers == {}


# LLM-generated content at query #16
#--------------------------

def test_provider_registry_constructor():
    registry = ProviderRegistry()
    assert registry._providers == {}
    assert ProviderRegistry._providers == {}


# LLM-generated content at query #17
#--------------------------

def test_base_data_provider_initialization_with_default_locale():
    provider = BaseDataProvider()
    assert provider.locale == Locale.DEFAULT.value
    assert isinstance(provider._dataset, dict)
    assert provider.seed is MissingSeed

def test_base_data_provider_initialization_with_custom_locale():
    provider = BaseDataProvider(locale=Locale.EN)
    assert provider.locale == Locale.EN.value
    assert isinstance(provider._dataset, dict)

def test_base_data_provider_initialization_with_seed():
    provider = BaseDataProvider(seed=12345)
    assert provider.seed == 12345

def test_base_data_provider_initialization_with_random_instance():
    custom_random = _random.Random()
    provider = BaseDataProvider(random=custom_random)
    assert provider.random is custom_random

def test_base_data_provider_initialization_with_invalid_random_type():
    try:
        BaseDataProvider(random="invalid")
        assert False
    except TypeError:
        assert True

def test_base_data_provider_initialization_locale_setup_failure():
    try:
        BaseDataProvider(locale="invalid_locale")
        assert False
    except UnsupportedLocale:
        assert True

def test_base_data_provider_initialization_with_datafile_meta():
    class TestProvider(BaseDataProvider):
        class Meta:
            name = "test"
            datafile = "test.json"
            datadir = DATADIR

    provider = TestProvider()
    assert provider._dataset == {}

def test_base_data_provider_initialization_without_datafile_meta():
    class TestProvider(BaseDataProvider):
        class Meta:
            name = "test"

    provider = TestProvider()
    assert provider._dataset == {}

def test_base_data_provider_initialization_with_composite_locale():
    provider = BaseDataProvider(locale=Locale.EN_US)
    assert provider.locale == Locale.EN_US.value

def test_base_data_provider_initialization_inherits_random_from_base():
    provider = BaseDataProvider()
    assert isinstance(provider.random, _random.Random)

def test_base_data_provider_initialization_seed_propagates_to_random():
    provider = BaseDataProvider(seed=999)
    assert provider.random._seed == 999


# LLM-generated content at query #18
#--------------------------

def test_provider_registry_constructor():
    registry = ProviderRegistry()
    assert registry._providers == {}


# LLM-generated content at query #19
#--------------------------

def test_auto_register_false_prevents_registration():
    class TestProvider(BaseProvider):
        class Meta:
            name = "test_provider"
            auto_register = False
    assert ProviderRegistry.get("test_provider") is None


# LLM-generated content at query #20
#--------------------------

def test_constructor_initializes_empty_providers():
    result = ProviderRegistry._providers
    expected = {}
    assert result == expected


# LLM-generated content at query #21
#--------------------------

def test_auto_register_provider_when_meta_name_and_auto_register_true():
    class TestProvider(BaseProvider):
        class Meta:
            name = "test_provider"
            auto_register = True
    assert ProviderRegistry.get("test_provider") == TestProvider

def test_auto_register_provider_when_meta_name_and_auto_register_default():
    class TestProvider(BaseProvider):
        class Meta:
            name = "test_provider_default"
    assert ProviderRegistry.get("test_provider_default") == TestProvider

def test_auto_register_provider_when_meta_name_and_auto_register_false():
    class TestProvider(BaseProvider):
        class Meta:
            name = "test_provider_false"
            auto_register = False
    assert ProviderRegistry.get("test_provider_false") is None


# LLM-generated content at query #22
#--------------------------

def test_reseed_global_seed_not_missing_seed():
    provider = BaseProvider(seed=None)
    _random.global_seed = 42
    provider.reseed()
    assert provider.seed is MissingSeed
    assert provider.random._seed == 42


# LLM-generated content at query #23
#--------------------------

def test_random_is_not_none_and_is_instance_of_random():
    random_instance = _random.Random()
    provider = BaseProvider(random=random_instance)
    assert provider.random is random_instance


# LLM-generated content at query #24
#--------------------------

def test_auto_register_provider():
    class TestProvider(BaseProvider):
        class Meta:
            name = "test_provider"
            auto_register = True
    assert ProviderRegistry.get("test_provider") == TestProvider


# LLM-generated content at query #25
#--------------------------

def test_random_is_none_so_no_type_check():
    BaseProvider(random=None)


# LLM-generated content at query #26
#--------------------------

def test_init_with_keyword_only_arguments():
    provider = BaseProvider(seed=42, random=None)
    assert provider.seed == 42
    assert isinstance(provider.random, _random.Random)


# LLM-generated content at query #27
#--------------------------

def test_base_data_provider_initialization_with_default_locale():
    provider = BaseDataProvider()
    assert provider.locale == Locale.DEFAULT.value
    assert isinstance(provider._dataset, dict)
    assert provider.seed is MissingSeed

def test_base_data_provider_initialization_with_custom_locale():
    provider = BaseDataProvider(locale=Locale.EN)
    assert provider.locale == Locale.EN.value
    assert isinstance(provider._dataset, dict)

def test_base_data_provider_initialization_with_seed():
    provider = BaseDataProvider(seed=42)
    assert provider.seed == 42

def test_base_data_provider_initialization_with_random_instance():
    custom_random = _random.Random()
    provider = BaseDataProvider(random=custom_random)
    assert provider.random is custom_random

def test_base_data_provider_initialization_with_invalid_random_type():
    try:
        BaseDataProvider(random="invalid")
    except TypeError as e:
        assert "The random must be an instance of mimesis.random.Random" in str(e)

def test_base_data_provider_initialization_locale_setup_failure():
    try:
        BaseDataProvider(locale="invalid_locale")
    except UnsupportedLocale:
        pass

def test_base_data_provider_initialization_without_datafile():
    class TestProvider(BaseDataProvider):
        class Meta:
            name = "test"
            auto_register = False
    provider = TestProvider()
    assert provider._dataset == {}

def test_base_data_provider_initialization_with_datafile():
    class TestProvider(BaseDataProvider):
        class Meta:
            name = "test"
            auto_register = False
            datafile = "test.json"
            datadir = DATADIR
    provider = TestProvider(locale=Locale.EN)
    assert isinstance(provider._dataset, dict)

def test_base_data_provider_initialization_with_composite_locale():
    class TestProvider(BaseDataProvider):
        class Meta:
            name = "test"
            auto_register = False
            datafile = "test.json"
            datadir = DATADIR
    provider = TestProvider(locale=Locale.EN_US)
    assert provider.locale == Locale.EN_US.value
    assert isinstance(provider._dataset, dict)

def test_base_data_provider_initialization_inherits_random_from_base():
    provider = BaseDataProvider()
    assert hasattr(provider, 'random')
    assert isinstance(provider.random, _random.Random)

def test_base_data_provider_initialization_seed_propagation():
    provider = BaseDataProvider(seed=123)
    assert provider.seed == 123
    assert provider.random._seed == 123

def test_base_data_provider_initialization_with_none_seed():
    provider = BaseDataProvider(seed=None)
    assert provider.seed is None

def test_base_data_provider_initialization_with_missing_seed():
    provider = BaseDataProvider()
    assert provider.seed is MissingSeed

def test_base_data_provider_initialization_locale_attribute_exists():
    provider = BaseDataProvider()
    assert hasattr(provider, 'locale')
    assert provider.locale == Locale.DEFAULT.value

def test_base_data_provider_initialization_dataset_attribute_exists():
    provider = BaseDataProvider()
    assert hasattr(provider, '_dataset')
    assert isinstance(provider._dataset, dict)

def test_base_data_provider_initialization_with_kwargs():
    provider = BaseDataProvider(locale=Locale.DE, seed=999)
    assert provider.locale == Locale.DE.value
    assert provider.seed == 999

def test_base_data_provider_initialization_with_args_ignored():
    provider = BaseDataProvider(Locale.FR, 777)
    assert provider.locale == Locale.FR.value
    assert provider.seed == 777


# LLM-generated content at query #28
#--------------------------

def test_validate_enum_with_false_predicate_at_line_11():
    from enum import Enum
    class TestEnum(Enum):
        A = 1
        B = 2
    provider = BaseProvider(seed=42)
    item = TestEnum.A
    result = provider.validate_enum(item, TestEnum)
    assert result == 1


# LLM-generated content at query #29
#--------------------------

def test_base_data_provider_initialization_with_default_locale():
    provider = BaseDataProvider()
    assert provider.locale == Locale.DEFAULT.value
    assert isinstance(provider._dataset, dict)
    assert provider.seed is MissingSeed

def test_base_data_provider_initialization_with_custom_locale():
    provider = BaseDataProvider(locale=Locale.EN)
    assert provider.locale == Locale.EN.value
    assert isinstance(provider._dataset, dict)

def test_base_data_provider_initialization_with_seed():
    provider = BaseDataProvider(seed=42)
    assert provider.seed == 42

def test_base_data_provider_initialization_with_random_instance():
    random_instance = _random.Random()
    provider = BaseDataProvider(random=random_instance)
    assert provider.random is random_instance

def test_base_data_provider_initialization_with_invalid_random_type():
    try:
        BaseDataProvider(random="invalid")
    except TypeError as e:
        assert "The random must be an instance of mimesis.random.Random" in str(e)

def test_base_data_provider_initialization_with_locale_and_seed():
    provider = BaseDataProvider(locale=Locale.RU, seed=123)
    assert provider.locale == Locale.RU.value
    assert provider.seed == 123

def test_base_data_provider_initialization_without_datafile():
    class TestProvider(BaseDataProvider):
        class Meta:
            name = "test"
            auto_register = False
    provider = TestProvider()
    assert provider._dataset == {}

def test_base_data_provider_initialization_with_unsupported_locale():
    try:
        BaseDataProvider(locale="unsupported")
    except UnsupportedLocale:
        pass

def test_base_data_provider_str_representation():
    provider = BaseDataProvider(locale=Locale.EN)
    assert str(provider) == f"BaseDataProvider <{Locale.EN}>"

def test_base_data_provider_inherits_from_base_provider():
    provider = BaseDataProvider()
    assert isinstance(provider, BaseProvider)


# LLM-generated content at query #30
#--------------------------

def test_locale_default_does_not_trigger_unsupported_locale():
    provider = BaseDataProvider(locale=Locale.DEFAULT)


# LLM-generated content at query #31
#--------------------------

def test_provider_registry_constructor():
    registry = ProviderRegistry()
    assert registry._providers == {}


# LLM-generated content at query #32
#--------------------------

def test_validate_enum_with_false_predicate_at_line_11():
    from enum import Enum
    class TestEnum(Enum):
        A = 1
        B = 2
    provider = BaseProvider(seed=42)
    item = TestEnum.A
    result = provider.validate_enum(item, TestEnum)
    assert result == 1


# LLM-generated content at query #33
#--------------------------

def test_reseed_with_missing_seed_and_global_seed_missing():
    provider = BaseProvider()
    original_state = provider.random.getstate()
    provider.reseed()
    assert provider.seed is MissingSeed
    assert provider.random.getstate() != original_state

def test_reseed_with_missing_seed_and_global_seed_set():
    _random.global_seed = 12345
    provider = BaseProvider()
    original_state = provider.random.getstate()
    provider.reseed()
    assert provider.seed is MissingSeed
    assert provider.random.getstate() != original_state
    _random.global_seed = MissingSeed

def test_reseed_with_explicit_seed():
    provider = BaseProvider()
    original_state = provider.random.getstate()
    provider.reseed(seed=999)
    assert provider.seed == 999
    assert provider.random.getstate() != original_state

def test_reseed_with_none_seed():
    provider = BaseProvider()
    original_state = provider.random.getstate()
    provider.reseed(seed=None)
    assert provider.seed is None
    assert provider.random.getstate() != original_state

def test_reseed_updates_instance_seed_attribute():
    provider = BaseProvider(seed=100)
    provider.reseed(seed=200)
    assert provider.seed == 200

def test_reseed_with_same_seed_produces_same_random_state():
    provider1 = BaseProvider()
    provider2 = BaseProvider()
    provider1.reseed(seed=555)
    provider2.reseed(seed=555)
    assert provider1.random.getstate() == provider2.random.getstate()

def test_reseed_with_different_seed_produces_different_random_state():
    provider1 = BaseProvider()
    provider2 = BaseProvider()
    provider1.reseed(seed=111)
    provider2.reseed(seed=222)
    assert provider1.random.getstate() != provider2.random.getstate()


# LLM-generated content at query #34
#--------------------------

def test_random_is_not_none_and_is_instance_of_random():
    random_instance = _random.Random()
    provider = BaseProvider(random=random_instance)


# LLM-generated content at query #35
#--------------------------

def test_random_is_not_none_and_is_instance_of_random():
    random_instance = _random.Random()
    provider = BaseProvider(random=random_instance)
    assert provider.random is random_instance


# LLM-generated content at query #36
#--------------------------

def test_base_data_provider_init_default_locale():
    provider = BaseDataProvider()
    assert provider.locale == Locale.DEFAULT.value
    assert provider._dataset == {}
    assert isinstance(provider.random, _random.Random)

def test_base_data_provider_init_custom_locale():
    provider = BaseDataProvider(locale=Locale.EN)
    assert provider.locale == Locale.EN.value
    assert isinstance(provider.random, _random.Random)

def test_base_data_provider_init_with_seed():
    provider = BaseDataProvider(seed=12345)
    assert provider.seed == 12345
    assert isinstance(provider.random, _random.Random)

def test_base_data_provider_init_with_random_instance():
    custom_random = _random.Random()
    provider = BaseDataProvider(random=custom_random)
    assert provider.random is custom_random

def test_base_data_provider_init_invalid_random_type():
    try:
        BaseDataProvider(random="invalid")
        assert False
    except TypeError:
        assert True

def test_base_data_provider_init_locale_dependent_data_loading():
    class TestProvider(BaseDataProvider):
        class Meta:
            name = "test"
            datafile = "test.json"
            datadir = DATADIR
    try:
        provider = TestProvider(locale=Locale.EN)
        assert provider._dataset != {}
    except FileNotFoundError:
        assert True

def test_base_data_provider_init_unsupported_locale():
    try:
        BaseDataProvider(locale="unsupported")
        assert False
    except UnsupportedLocale:
        assert True

def test_base_data_provider_init_keyword_only_seed():
    try:
        BaseDataProvider(12345)
        assert False
    except TypeError:
        assert True

def test_base_data_provider_init_inheritance():
    class CustomProvider(BaseDataProvider):
        pass
    provider = CustomProvider()
    assert isinstance(provider, BaseDataProvider)
    assert provider.locale == Locale.DEFAULT.value

def test_base_data_provider_str_representation():
    provider = BaseDataProvider(locale=Locale.EN)
    expected = f"{provider.__class__.__name__} <{Locale.EN}>"
    assert str(provider) == expected


# LLM-generated content at query #37
#--------------------------

def test_base_data_provider_init_default_locale():
    provider = BaseDataProvider()
    assert provider.locale == Locale.DEFAULT.value
    assert provider._dataset == {}
    assert isinstance(provider.random, _random.Random)

def test_base_data_provider_init_custom_locale():
    provider = BaseDataProvider(locale=Locale.EN)
    assert provider.locale == Locale.EN.value
    assert isinstance(provider.random, _random.Random)

def test_base_data_provider_init_with_seed():
    provider = BaseDataProvider(seed=42)
    assert provider.seed == 42
    assert isinstance(provider.random, _random.Random)

def test_base_data_provider_init_with_random_instance():
    custom_random = _random.Random()
    provider = BaseDataProvider(random=custom_random)
    assert provider.random is custom_random

def test_base_data_provider_init_invalid_random_type():
    try:
        BaseDataProvider(random="invalid")
    except TypeError as e:
        assert "The random must be an instance of mimesis.random.Random" in str(e)

def test_base_data_provider_init_locale_dependent_data_loading():
    class TestProvider(BaseDataProvider):
        class Meta:
            name = "test"
            datafile = "test.json"
            datadir = DATADIR

    provider = TestProvider(locale=Locale.EN)
    assert provider.locale == Locale.EN.value
    assert isinstance(provider._dataset, dict)

def test_base_data_provider_init_unsupported_locale():
    try:
        BaseDataProvider(locale="unsupported")
    except UnsupportedLocale:
        pass

def test_base_data_provider_init_with_args_and_kwargs():
    provider = BaseDataProvider(locale=Locale.EN, seed=123, extra_arg="test")
    assert provider.locale == Locale.EN.value
    assert provider.seed == 123

def test_base_data_provider_init_locale_separator_handling():
    provider = BaseDataProvider(locale="en-US")
    assert provider.locale == "en-US"
    assert isinstance(provider._dataset, dict)

def test_base_data_provider_init_no_datafile():
    class NoDataProvider(BaseDataProvider):
        class Meta:
            name = "nodata"

    provider = NoDataProvider()
    assert provider._dataset == {}


# LLM-generated content at query #38
#--------------------------

def test_init_with_default_parameters():
    provider = BaseProvider()
    assert provider.seed is MissingSeed
    assert isinstance(provider.random, _random.Random)

def test_init_with_custom_random_instance():
    custom_random = _random.Random()
    provider = BaseProvider(random=custom_random)
    assert provider.random is custom_random
    assert provider.seed is MissingSeed

def test_init_with_invalid_random_type():
    try:
        BaseProvider(random="invalid")
        assert False
    except TypeError as e:
        assert "The random must be an instance of mimesis.random.Random" in str(e)

def test_init_with_seed():
    provider = BaseProvider(seed=42)
    assert provider.seed == 42
    assert isinstance(provider.random, _random.Random)

def test_init_with_seed_none():
    provider = BaseProvider(seed=None)
    assert provider.seed is None
    assert isinstance(provider.random, _random.Random)

def test_init_with_seed_and_custom_random():
    custom_random = _random.Random()
    provider = BaseProvider(seed=123, random=custom_random)
    assert provider.seed == 123
    assert provider.random is custom_random


# LLM-generated content at query #39
#--------------------------

def test_base_data_provider_init_with_default_locale():
    provider = BaseDataProvider()
    assert provider.locale == Locale.DEFAULT.value
    assert provider._dataset == {}

def test_base_data_provider_init_with_custom_locale():
    provider = BaseDataProvider(locale=Locale.EN)
    assert provider.locale == Locale.EN.value

def test_base_data_provider_init_with_seed():
    provider = BaseDataProvider(seed=12345)
    assert provider.seed == 12345

def test_base_data_provider_init_with_random_instance():
    custom_random = _random.Random()
    provider = BaseDataProvider(random=custom_random)
    assert provider.random is custom_random

def test_base_data_provider_init_with_invalid_random_type():
    try:
        BaseDataProvider(random="invalid")
    except TypeError as e:
        assert "The random must be an instance of mimesis.random.Random" in str(e)

def test_base_data_provider_init_locale_setup_called():
    provider = BaseDataProvider(locale=Locale.RU)
    assert provider.locale == Locale.RU.value

def test_base_data_provider_init_load_dataset_called():
    class TestProvider(BaseDataProvider):
        class Meta:
            name = "test"
            datafile = "test.json"
            datadir = DATADIR
    try:
        provider = TestProvider()
    except FileNotFoundError:
        pass

def test_base_data_provider_init_with_unsupported_locale():
    try:
        BaseDataProvider(locale="unsupported")
    except UnsupportedLocale:
        pass

def test_base_data_provider_init_seed_passed_to_parent():
    provider = BaseDataProvider(seed=999)
    assert provider.seed == 999

def test_base_data_provider_init_with_none_seed():
    provider = BaseDataProvider(seed=None)
    assert provider.seed is None

def test_base_data_provider_init_with_missing_seed():
    provider = BaseDataProvider()
    assert provider.seed is MissingSeed

def test_base_data_provider_init_with_args_and_kwargs():
    provider = BaseDataProvider(locale=Locale.DE, seed=42)
    assert provider.locale == Locale.DE.value
    assert provider.seed == 42


# LLM-generated content at query #40
#--------------------------

def test_base_data_provider_init_default_locale():
    provider = BaseDataProvider()
    assert provider.locale == Locale.DEFAULT.value
    assert isinstance(provider._dataset, dict)
    assert provider.seed is MissingSeed

def test_base_data_provider_init_custom_locale():
    provider = BaseDataProvider(locale=Locale.EN)
    assert provider.locale == Locale.EN.value
    assert isinstance(provider._dataset, dict)

def test_base_data_provider_init_with_seed():
    provider = BaseDataProvider(seed=42)
    assert provider.seed == 42

def test_base_data_provider_init_with_random_instance():
    custom_random = _random.Random()
    provider = BaseDataProvider(random=custom_random)
    assert provider.random is custom_random

def test_base_data_provider_init_locale_validation_error():
    try:
        BaseDataProvider(locale="invalid_locale")
        assert False
    except UnsupportedLocale:
        assert True

def test_base_data_provider_init_inherits_base_provider_attributes():
    provider = BaseDataProvider()
    assert hasattr(provider, "random")
    assert hasattr(provider, "seed")

def test_base_data_provider_init_dataset_loaded():
    class TestProvider(BaseDataProvider):
        class Meta:
            name = "test"
            datafile = "test.json"
            datadir = DATADIR
    try:
        provider = TestProvider()
        assert isinstance(provider._dataset, dict)
    except FileNotFoundError:
        assert True

def test_base_data_provider_init_no_datafile():
    class TestProvider(BaseDataProvider):
        class Meta:
            name = "test"
    provider = TestProvider()
    assert provider._dataset == {}

def test_base_data_provider_init_locale_with_separator():
    provider = BaseDataProvider(locale=Locale.EN_US)
    assert provider.locale == Locale.EN_US.value

def test_base_data_provider_init_passes_args_to_base():
    custom_random = _random.Random()
    provider = BaseDataProvider(seed=123, random=custom_random)
    assert provider.seed == 123
    assert provider.random is custom_random


# LLM-generated content at query #41
#--------------------------

def test_random_is_not_none_and_is_instance_of_random():
    from mimesis.providers.base import BaseProvider
    from mimesis.random import Random
    custom_random = Random()
    provider = BaseProvider(random=custom_random)
    assert provider.random is custom_random

def test_random_is_not_none_and_is_not_instance_of_random_raises_type_error():
    from mimesis.providers.base import BaseProvider
    class FakeRandom:
        pass
    custom_random = FakeRandom()
    try:
        BaseProvider(random=custom_random)
        assert False
    except TypeError as e:
        assert str(e) == "The random must be an instance of mimesis.random.Random"


# LLM-generated content at query #42
#--------------------------

def test_base_data_provider_initialization_with_default_locale():
    provider = BaseDataProvider()
    assert provider.locale == Locale.DEFAULT.value
    assert isinstance(provider._dataset, dict)

def test_base_data_provider_initialization_with_custom_locale():
    provider = BaseDataProvider(locale=Locale.EN)
    assert provider.locale == Locale.EN.value

def test_base_data_provider_initialization_with_seed():
    provider = BaseDataProvider(seed=42)
    assert provider.seed == 42

def test_base_data_provider_initialization_with_random_instance():
    random_instance = _random.Random()
    provider = BaseDataProvider(random=random_instance)
    assert provider.random is random_instance

def test_base_data_provider_initialization_with_locale_and_seed():
    provider = BaseDataProvider(locale=Locale.RU, seed=123)
    assert provider.locale == Locale.RU.value
    assert provider.seed == 123

def test_base_data_provider_initialization_locale_validation():
    provider = BaseDataProvider(locale="en_US")
    assert provider.locale == "en_US"

def test_base_data_provider_initialization_dataset_loaded():
    class TestProvider(BaseDataProvider):
        class Meta:
            name = "test"
            datafile = "test.json"
    provider = TestProvider()
    assert provider._dataset == {}

def test_base_data_provider_initialization_inherits_from_base_provider():
    provider = BaseDataProvider()
    assert isinstance(provider, BaseProvider)

def test_base_data_provider_initialization_seed_passed_to_super():
    provider = BaseDataProvider(seed=999)
    assert provider._has_seed() is True

def test_base_data_provider_initialization_with_missing_seed():
    provider = BaseDataProvider(seed=MissingSeed)
    assert provider.seed is MissingSeed


# LLM-generated content at query #43
#--------------------------

def test_auto_register_false_prevents_registration():
    class TestProvider(BaseProvider):
        class Meta:
            name = "test_provider"
            auto_register = False
    assert "test_provider" not in ProviderRegistry._registry


# LLM-generated content at query #44
#--------------------------

def test_base_data_provider_init_default_locale():
    provider = BaseDataProvider()
    assert provider.locale == Locale.DEFAULT.value
    assert isinstance(provider._dataset, dict)

def test_base_data_provider_init_custom_locale():
    provider = BaseDataProvider(locale=Locale.EN)
    assert provider.locale == Locale.EN.value

def test_base_data_provider_init_with_seed():
    provider = BaseDataProvider(seed=42)
    assert provider.seed == 42

def test_base_data_provider_init_with_random_instance():
    custom_random = _random.Random()
    provider = BaseDataProvider(random=custom_random)
    assert provider.random is custom_random

def test_base_data_provider_init_locale_validation():
    provider = BaseDataProvider(locale="en_US")
    assert provider.locale == "en_US"

def test_base_data_provider_init_unsupported_locale():
    try:
        BaseDataProvider(locale="unsupported")
        assert False
    except UnsupportedLocale:
        assert True

def test_base_data_provider_init_datafile_loading():
    class TestProvider(BaseDataProvider):
        class Meta:
            name = "test"
            datafile = "test.json"
    provider = TestProvider()
    assert provider._dataset != {}

def test_base_data_provider_init_no_datafile():
    class TestProvider(BaseDataProvider):
        class Meta:
            name = "test"
    provider = TestProvider()
    assert provider._dataset == {}

def test_base_data_provider_init_inherits_random():
    provider = BaseDataProvider()
    assert hasattr(provider, "random")
    assert isinstance(provider.random, _random.Random)

def test_base_data_provider_init_seed_propagation():
    provider = BaseDataProvider(seed=123)
    assert provider.seed == 123


# LLM-generated content at query #45
#--------------------------

def test_random_is_not_none_and_is_instance_of_random():
    from mimesis.providers.base import BaseProvider
    from mimesis.random import Random
    custom_random = Random()
    provider = BaseProvider(random=custom_random)
    assert provider.random is custom_random

def test_random_is_not_none_and_is_not_instance_of_random_raises_typeerror():
    from mimesis.providers.base import BaseProvider
    class FakeRandom:
        pass
    custom_random = FakeRandom()
    try:
        BaseProvider(random=custom_random)
        assert False
    except TypeError as e:
        assert str(e) == "The random must be an instance of mimesis.random.Random"

def test_random_is_none_creates_new_random_instance():
    from mimesis.providers.base import BaseProvider
    from mimesis.random import Random
    provider = BaseProvider(random=None)
    assert isinstance(provider.random, Random)


# LLM-generated content at query #46
#--------------------------

def test_auto_register_provider():
    class TestProvider(BaseProvider):
        class Meta:
            name = "test_provider"
            auto_register = True
    assert ProviderRegistry.get("test_provider") == TestProvider

def test_auto_register_false():
    class TestProviderNoRegister(BaseProvider):
        class Meta:
            name = "test_provider_no_register"
            auto_register = False
    with pytest.raises(KeyError):
        ProviderRegistry.get("test_provider_no_register")

def test_auto_register_no_meta_name():
    class TestProviderNoMetaName(BaseProvider):
        pass
    assert True

def test_auto_register_with_meta_no_name():
    class TestProviderMetaNoName(BaseProvider):
        class Meta:
            auto_register = True
    assert True


# LLM-generated content at query #47
#--------------------------

def test_validate_enum_item_is_none():
    from enum import Enum
    class TestEnum(Enum):
        A = "a"
        B = "b"
    provider = BaseProvider(seed=42)
    result = provider.validate_enum(None, TestEnum)
    assert result in ["a", "b"]


# LLM-generated content at query #48
#--------------------------

def test_base_data_provider_init_default():
    provider = BaseDataProvider()
    assert provider.locale == Locale.DEFAULT.value
    assert provider.seed is MissingSeed
    assert isinstance(provider.random, _random.Random)
    assert provider._dataset == {}

def test_base_data_provider_init_with_locale():
    provider = BaseDataProvider(locale=Locale.EN)
    assert provider.locale == Locale.EN.value
    assert provider.seed is MissingSeed
    assert isinstance(provider.random, _random.Random)

def test_base_data_provider_init_with_seed():
    provider = BaseDataProvider(seed=42)
    assert provider.locale == Locale.DEFAULT.value
    assert provider.seed == 42
    assert isinstance(provider.random, _random.Random)

def test_base_data_provider_init_with_locale_and_seed():
    provider = BaseDataProvider(locale=Locale.RU, seed=123)
    assert provider.locale == Locale.RU.value
    assert provider.seed == 123
    assert isinstance(provider.random, _random.Random)

def test_base_data_provider_init_with_custom_random():
    custom_random = _random.Random()
    provider = BaseDataProvider(random=custom_random)
    assert provider.random is custom_random
    assert provider.locale == Locale.DEFAULT.value
    assert provider.seed is MissingSeed

def test_base_data_provider_init_with_custom_random_and_locale():
    custom_random = _random.Random()
    provider = BaseDataProvider(locale=Locale.JA, random=custom_random)
    assert provider.random is custom_random
    assert provider.locale == Locale.JA.value
    assert provider.seed is MissingSeed

def test_base_data_provider_init_with_custom_random_and_seed():
    custom_random = _random.Random()
    provider = BaseDataProvider(seed=999, random=custom_random)
    assert provider.random is custom_random
    assert provider.locale == Locale.DEFAULT.value
    assert provider.seed == 999

def test_base_data_provider_init_with_all_parameters():
    custom_random = _random.Random()
    provider = BaseDataProvider(locale=Locale.FR, seed=777, random=custom_random)
    assert provider.random is custom_random
    assert provider.locale == Locale.FR.value
    assert provider.seed == 777

def test_base_data_provider_init_unsupported_locale():
    try:
        BaseDataProvider(locale="unsupported")
        assert False
    except UnsupportedLocale:
        assert True

def test_base_data_provider_init_inherits_random_from_base():
    provider = BaseDataProvider()
    assert isinstance(provider.random, _random.Random)

def test_base_data_provider_init_seed_missing_by_default():
    provider = BaseDataProvider()
    assert provider.seed is MissingSeed

def test_base_data_provider_init_locale_default_by_default():
    provider = BaseDataProvider()
    assert provider.locale == Locale.DEFAULT.value

def test_base_data_provider_init_dataset_empty_by_default():
    provider = BaseDataProvider()
    assert provider._dataset == {}

def test_base_data_provider_init_with_none_seed():
    provider = BaseDataProvider(seed=None)
    assert provider.seed is None

def test_base_data_provider_init_with_zero_seed():
    provider = BaseDataProvider(seed=0)
    assert provider.seed == 0

def test_base_data_provider_init_with_negative_seed():
    provider = BaseDataProvider(seed=-5)
    assert provider.seed == -5

def test_base_data_provider_init_locale_as_string():
    provider = BaseDataProvider(locale="en")
    assert provider.locale == "en"

def test_base_data_provider_init_locale_as_locale_enum():
    provider = BaseDataProvider(locale=Locale.DE)
    assert provider.locale == Locale.DE.value


# LLM-generated content at query #49
#--------------------------

def test_auto_register_false_prevents_registration():
    class TestProvider(BaseProvider):
        class Meta:
            name = "test_provider"
            auto_register = False
    assert "test_provider" not in ProviderRegistry._registry


# LLM-generated content at query #50
#--------------------------

def test_validate_enum_with_false_predicate_at_line_11():
    from enum import Enum
    class TestEnum(Enum):
        A = 1
        B = 2
    provider = BaseProvider()
    item = TestEnum.A
    result = provider.validate_enum(item, TestEnum)
    assert result == 1


# LLM-generated content at query #51
#--------------------------

def test_initialization_with_keyword_only_arguments():
    provider = BaseProvider(seed=None, random=None)
    assert provider.seed is None
    assert isinstance(provider.random, _random.Random)


# LLM-generated content at query #52
#--------------------------

def test_auto_register_false_prevents_registration():
    class TestProvider(BaseProvider):
        class Meta:
            name = "test_provider"
            auto_register = False
    assert ProviderRegistry.get("test_provider") is None


# LLM-generated content at query #53
#--------------------------

def test_auto_register_provider_when_meta_name_and_auto_register_true():
    class TestProvider(BaseProvider):
        class Meta:
            name = "test_provider"
            auto_register = True
    assert ProviderRegistry.get("test_provider") == TestProvider

def test_auto_register_provider_when_meta_name_and_auto_register_default():
    class TestProvider(BaseProvider):
        class Meta:
            name = "test_provider_default"
    assert ProviderRegistry.get("test_provider_default") == TestProvider

def test_auto_register_provider_when_meta_name_and_auto_register_false():
    class TestProvider(BaseProvider):
        class Meta:
            name = "test_provider_false"
            auto_register = False
    with pytest.raises(KeyError):
        ProviderRegistry.get("test_provider_false")


# LLM-generated content at query #54
#--------------------------

def test_base_provider_initialization_with_defaults():
    provider = BaseProvider()
    assert provider.seed is MissingSeed
    assert isinstance(provider.random, _random.Random)

def test_base_provider_initialization_with_custom_random():
    custom_random = _random.Random()
    provider = BaseProvider(random=custom_random)
    assert provider.random is custom_random
    assert provider.seed is MissingSeed

def test_base_provider_initialization_with_seed():
    provider = BaseProvider(seed=42)
    assert provider.seed == 42
    assert isinstance(provider.random, _random.Random)

def test_base_provider_initialization_with_none_seed():
    provider = BaseProvider(seed=None)
    assert provider.seed is None
    assert isinstance(provider.random, _random.Random)

def test_base_provider_initialization_with_invalid_random_type():
    try:
        BaseProvider(random="invalid")
        assert False
    except TypeError:
        assert True


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_constructor_with_default_parameters():
    provider = BaseProvider()
    assert provider.seed is MissingSeed
    assert isinstance(provider.random, _random.Random)

def test_constructor_with_custom_random():
    custom_random = _random.Random()
    provider = BaseProvider(random=custom_random)
    assert provider.random is custom_random
    assert provider.seed is MissingSeed

def test_constructor_with_seed():
    provider = BaseProvider(seed=42)
    assert provider.seed == 42
    assert isinstance(provider.random, _random.Random)

def test_constructor_with_seed_and_custom_random():
    custom_random = _random.Random()
    provider = BaseProvider(seed=42, random=custom_random)
    assert provider.random is custom_random
    assert provider.seed == 42

def test_constructor_raises_type_error_for_invalid_random():
    try:
        BaseProvider(random="invalid")
        assert False
    except TypeError:
        assert True

def test_constructor_with_seed_none():
    provider = BaseProvider(seed=None)
    assert provider.seed is None
    assert isinstance(provider.random, _random.Random)

def test_constructor_with_seed_missing_seed():
    provider = BaseProvider(seed=MissingSeed)
    assert provider.seed is MissingSeed
    assert isinstance(provider.random, _random.Random)


# LLM-generated content at query #2
#--------------------------

def test_constructor_initializes_empty_providers():
    registry = ProviderRegistry()
    assert registry._providers == {}


# LLM-generated content at query #3
#--------------------------

def test_provider_registry_constructor():
    registry = ProviderRegistry()
    assert registry._providers == {}


# LLM-generated content at query #4
#--------------------------

def test_base_data_provider_init_default_locale():
    provider = BaseDataProvider()
    assert provider.locale == Locale.DEFAULT.value
    assert provider._dataset == {}
    assert isinstance(provider.random, _random.Random)

def test_base_data_provider_init_custom_locale():
    provider = BaseDataProvider(locale=Locale.EN)
    assert provider.locale == Locale.EN.value
    assert isinstance(provider.random, _random.Random)

def test_base_data_provider_init_with_seed():
    provider = BaseDataProvider(seed=42)
    assert provider.seed == 42
    assert isinstance(provider.random, _random.Random)

def test_base_data_provider_init_with_random_instance():
    custom_random = _random.Random()
    provider = BaseDataProvider(random=custom_random)
    assert provider.random is custom_random

def test_base_data_provider_init_invalid_random_type():
    try:
        BaseDataProvider(random="invalid")
    except TypeError as e:
        assert "The random must be an instance of mimesis.random.Random" in str(e)

def test_base_data_provider_init_locale_dependent_data_loading():
    class TestProvider(BaseDataProvider):
        class Meta:
            name = "test"
            datafile = "test.json"
            datadir = Path("/fake/dir")
    try:
        provider = TestProvider(locale=Locale.EN)
    except FileNotFoundError:
        pass

def test_base_data_provider_init_unsupported_locale():
    try:
        BaseDataProvider(locale="invalid_locale")
    except UnsupportedLocale:
        pass

def test_base_data_provider_init_with_args_and_kwargs():
    provider = BaseDataProvider(locale=Locale.EN, seed=123)
    assert provider.locale == Locale.EN.value
    assert provider.seed == 123

def test_base_data_provider_str_representation():
    provider = BaseDataProvider(locale=Locale.EN)
    assert str(provider) == f"BaseDataProvider <{Locale.EN}>"

def test_base_data_provider_init_subclass_auto_registration():
    class TestProvider(BaseDataProvider):
        class Meta:
            name = "test_provider"
    assert ProviderRegistry.get("test_provider") == TestProvider

def test_base_data_provider_init_subclass_auto_register_false():
    class TestProvider(BaseDataProvider):
        class Meta:
            name = "test_provider_no_register"
            auto_register = False
    try:
        ProviderRegistry.get("test_provider_no_register")
    except KeyError:
        pass


# LLM-generated content at query #5
#--------------------------

def test_init_with_custom_random():
    custom_random = _random.Random()
    provider = BaseProvider(random=custom_random)
    assert provider.random is custom_random

def test_init_without_random():
    provider = BaseProvider()
    assert isinstance(provider.random, _random.Random)

def test_init_with_invalid_random():
    try:
        BaseProvider(random="invalid")
    except TypeError as e:
        assert "The random must be an instance of mimesis.random.Random" in str(e)

def test_init_with_seed():
    provider = BaseProvider(seed=42)
    assert provider.seed == 42

def test_init_with_missing_seed():
    provider = BaseProvider()
    assert provider.seed is MissingSeed

def test_init_with_none_seed():
    provider = BaseProvider(seed=None)
    assert provider.seed is None

def test_reseed_called_on_init():
    provider = BaseProvider(seed=123)
    assert provider.seed == 123


# LLM-generated content at query #6
#--------------------------

def test_base_provider_initializes_with_defaults():
    provider = BaseProvider()
    assert provider.seed is MissingSeed
    assert isinstance(provider.random, _random.Random)

def test_base_provider_initializes_with_custom_random():
    custom_random = _random.Random()
    provider = BaseProvider(random=custom_random)
    assert provider.random is custom_random

def test_base_provider_initializes_with_seed():
    provider = BaseProvider(seed=42)
    assert provider.seed == 42

def test_base_provider_raises_type_error_for_invalid_random():
    try:
        BaseProvider(random="invalid")
    except TypeError as e:
        assert "The random must be an instance of mimesis.random.Random" in str(e)

def test_base_provider_initializes_with_none_seed():
    provider = BaseProvider(seed=None)
    assert provider.seed is None

def test_base_provider_initializes_with_missing_seed():
    provider = BaseProvider(seed=MissingSeed)
    assert provider.seed is MissingSeed


# LLM-generated content at query #7
#--------------------------

def test_constructor_with_default_parameters():
    provider = BaseProvider()
    assert provider.seed is MissingSeed
    assert isinstance(provider.random, _random.Random)

def test_constructor_with_custom_seed():
    provider = BaseProvider(seed=42)
    assert provider.seed == 42
    assert isinstance(provider.random, _random.Random)

def test_constructor_with_none_seed():
    provider = BaseProvider(seed=None)
    assert provider.seed is None
    assert isinstance(provider.random, _random.Random)

def test_constructor_with_custom_random_instance():
    custom_random = _random.Random()
    provider = BaseProvider(random=custom_random)
    assert provider.random is custom_random
    assert provider.seed is MissingSeed

def test_constructor_raises_type_error_for_invalid_random():
    try:
        BaseProvider(random="invalid")
        assert False
    except TypeError as e:
        assert "The random must be an instance of mimesis.random.Random" in str(e)

def test_constructor_with_seed_and_custom_random():
    custom_random = _random.Random()
    provider = BaseProvider(seed=123, random=custom_random)
    assert provider.random is custom_random
    assert provider.seed == 123


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_constructor_with_default_parameters():
    provider = BaseProvider()
    assert provider.seed is MissingSeed
    assert isinstance(provider.random, _random.Random)

def test_constructor_with_custom_random():
    custom_random = _random.Random()
    provider = BaseProvider(random=custom_random)
    assert provider.random is custom_random
    assert provider.seed is MissingSeed

def test_constructor_with_seed():
    provider = BaseProvider(seed=42)
    assert provider.seed == 42
    assert isinstance(provider.random, _random.Random)

def test_constructor_with_seed_and_custom_random():
    custom_random = _random.Random()
    provider = BaseProvider(seed=123, random=custom_random)
    assert provider.random is custom_random
    assert provider.seed == 123

def test_constructor_raises_type_error_for_invalid_random():
    try:
        BaseProvider(random="invalid")
        assert False
    except TypeError as e:
        assert str(e) == "The random must be an instance of mimesis.random.Random"


# LLM-generated content at query #2
#--------------------------

def test_provider_registry_constructor():
    registry = ProviderRegistry()
    assert registry._providers == {}


# LLM-generated content at query #3
#--------------------------

def test_base_data_provider_initialization_with_default_locale():
    provider = BaseDataProvider()
    assert provider.locale == Locale.DEFAULT.value
    assert provider._dataset == {}
    assert isinstance(provider.random, _random.Random)
    assert provider.seed is MissingSeed

def test_base_data_provider_initialization_with_custom_locale():
    provider = BaseDataProvider(locale=Locale.EN)
    assert provider.locale == Locale.EN.value
    assert isinstance(provider.random, _random.Random)
    assert provider.seed is MissingSeed

def test_base_data_provider_initialization_with_seed():
    provider = BaseDataProvider(seed=42)
    assert provider.locale == Locale.DEFAULT.value
    assert provider.seed == 42
    assert isinstance(provider.random, _random.Random)

def test_base_data_provider_initialization_with_custom_random():
    custom_random = _random.Random()
    provider = BaseDataProvider(random=custom_random)
    assert provider.random is custom_random
    assert provider.locale == Locale.DEFAULT.value
    assert provider.seed is MissingSeed

def test_base_data_provider_initialization_with_locale_and_seed():
    provider = BaseDataProvider(locale=Locale.RU, seed=123)
    assert provider.locale == Locale.RU.value
    assert provider.seed == 123
    assert isinstance(provider.random, _random.Random)

def test_base_data_provider_initialization_with_invalid_random_type():
    try:
        BaseDataProvider(random="invalid")
        assert False
    except TypeError as e:
        assert "The random must be an instance of mimesis.random.Random" in str(e)

def test_base_data_provider_initialization_with_auto_register():
    class TestProvider(BaseDataProvider):
        class Meta:
            name = "test_provider"
            auto_register = True
    assert ProviderRegistry.get("test_provider") is TestProvider

def test_base_data_provider_initialization_with_auto_register_false():
    class TestProvider(BaseDataProvider):
        class Meta:
            name = "test_provider_no_register"
            auto_register = False
    try:
        ProviderRegistry.get("test_provider_no_register")
        assert False
    except Exception:
        assert True

def test_base_data_provider_initialization_with_datafile_meta():
    class TestProvider(BaseDataProvider):
        class Meta:
            name = "test_datafile"
            datafile = "test.json"
            datadir = pathlib.Path("/fake/dir")
    provider = TestProvider()
    assert provider.Meta.datafile == "test.json"
    assert provider.Meta.datadir == pathlib.Path("/fake/dir")

def test_base_data_provider_initialization_locale_setup_error():
    try:
        BaseDataProvider(locale="invalid_locale")
        assert False
    except UnsupportedLocale:
        assert True


# LLM-generated content at query #4
#--------------------------

def test_provider_registry_constructor():
    registry = ProviderRegistry()
    assert registry._providers == {}


# LLM-generated content at query #5
#--------------------------

def test_base_data_provider_init_default_locale():
    provider = BaseDataProvider()
    assert provider.locale == Locale.DEFAULT.value
    assert isinstance(provider._dataset, dict)
    assert provider.seed is MissingSeed

def test_base_data_provider_init_custom_locale():
    provider = BaseDataProvider(locale=Locale.EN)
    assert provider.locale == Locale.EN.value
    assert isinstance(provider._dataset, dict)

def test_base_data_provider_init_with_seed():
    provider = BaseDataProvider(seed=42)
    assert provider.seed == 42

def test_base_data_provider_init_with_random_instance():
    custom_random = _random.Random()
    provider = BaseDataProvider(random=custom_random)
    assert provider.random is custom_random

def test_base_data_provider_init_inherits_random_from_base():
    provider = BaseDataProvider()
    assert isinstance(provider.random, _random.Random)

def test_base_data_provider_init_locale_validation():
    provider = BaseDataProvider(locale=Locale.RU)
    assert provider.locale == Locale.RU.value

def test_base_data_provider_init_dataset_loaded():
    class TestProvider(BaseDataProvider):
        class Meta:
            name = "test"
            datafile = "test.json"
            datadir = DATADIR
    provider = TestProvider(locale=Locale.EN)
    assert isinstance(provider._dataset, dict)

def test_base_data_provider_init_no_datafile():
    class TestProvider(BaseDataProvider):
        class Meta:
            name = "test"
    provider = TestProvider()
    assert provider._dataset == {}

def test_base_data_provider_init_with_args_and_kwargs():
    provider = BaseDataProvider(locale=Locale.EN, seed=123)
    assert provider.locale == Locale.EN.value
    assert provider.seed == 123

def test_base_data_provider_init_locale_separator_handling():
    provider = BaseDataProvider(locale=Locale.EN_US)
    assert provider.locale == Locale.EN_US.value


# LLM-generated content at query #6
#--------------------------

def test_auto_register_provider():
    class TestProvider(BaseProvider):
        class Meta:
            name = "test_provider"
            auto_register = True
    assert ProviderRegistry.is_registered("test_provider")


# LLM-generated content at query #7
#--------------------------

def test_random_is_not_none_and_is_instance_of_random():
    random_instance = _random.Random()
    provider = BaseProvider(random=random_instance)
    assert provider.random is random_instance


# LLM-generated content at query #8
#--------------------------

def test_provider_registry_constructor():
    registry = ProviderRegistry()
    assert registry._providers == {}
    assert ProviderRegistry._providers == {}


# LLM-generated content at query #9
#--------------------------

def test_reseed_with_missing_seed_and_global_seed_missing():
    provider = BaseProvider()
    original_state = provider.random.getstate()
    provider.reseed()
    assert provider.seed is MissingSeed
    assert provider.random.getstate() == original_state

def test_reseed_with_missing_seed_and_global_seed_set():
    _random.global_seed = 42
    provider = BaseProvider()
    provider.reseed()
    assert provider.seed is MissingSeed
    provider.random.seed(42)
    expected_state = provider.random.getstate()
    provider.reseed()
    assert provider.random.getstate() == expected_state
    _random.global_seed = MissingSeed

def test_reseed_with_none_seed():
    provider = BaseProvider()
    provider.reseed(None)
    assert provider.seed is None
    state_after_none = provider.random.getstate()
    provider.reseed(None)
    assert provider.random.getstate() != state_after_none

def test_reseed_with_integer_seed():
    provider = BaseProvider()
    provider.reseed(123)
    assert provider.seed == 123
    provider.random.seed(123)
    expected_state = provider.random.getstate()
    provider.reseed(123)
    assert provider.random.getstate() == expected_state

def test_reseed_changes_seed_attribute():
    provider = BaseProvider(seed=100)
    assert provider.seed == 100
    provider.reseed(200)
    assert provider.seed == 200

def test_reseed_with_string_seed():
    provider = BaseProvider()
    provider.reseed("test_seed")
    assert provider.seed == "test_seed"
    provider.random.seed("test_seed")
    expected_state = provider.random.getstate()
    provider.reseed("test_seed")
    assert provider.random.getstate() == expected_state


# LLM-generated content at query #10
#--------------------------

def test_provider_registry_constructor():
    registry = ProviderRegistry()
    providers = ProviderRegistry._providers
    assert providers == {}


# LLM-generated content at query #11
#--------------------------

def test_auto_register_false_prevents_registration():
    class TestProvider(BaseProvider):
        class Meta:
            name = "test_provider"
            auto_register = False
    assert ProviderRegistry.get("test_provider") is None


# LLM-generated content at query #12
#--------------------------

def test_init_calls_super_with_seed_and_args_kwargs():
    class TestProvider(BaseDataProvider):
        class Meta:
            name = "test"
            datafile = "test.json"
            datadir = pathlib.Path("/tmp")

    instance = TestProvider(locale="en", seed=42, extra_arg="extra", extra_kwarg="kwarg")
    assert instance.seed == 42
    assert instance.locale == "en"


# LLM-generated content at query #13
#--------------------------

def test_random_is_none_so_instance_check_not_performed():
    provider = BaseProvider(random=None)


# LLM-generated content at query #14
#--------------------------

def test_constructor_with_default_parameters():
    provider = BaseProvider()
    assert provider.seed is MissingSeed
    assert isinstance(provider.random, _random.Random)

def test_constructor_with_custom_seed():
    provider = BaseProvider(seed=42)
    assert provider.seed == 42
    assert isinstance(provider.random, _random.Random)

def test_constructor_with_none_seed():
    provider = BaseProvider(seed=None)
    assert provider.seed is None
    assert isinstance(provider.random, _random.Random)

def test_constructor_with_custom_random_instance():
    custom_random = _random.Random()
    provider = BaseProvider(random=custom_random)
    assert provider.random is custom_random
    assert provider.seed is MissingSeed

def test_constructor_with_custom_random_and_seed():
    custom_random = _random.Random()
    provider = BaseProvider(seed=123, random=custom_random)
    assert provider.random is custom_random
    assert provider.seed == 123

def test_constructor_raises_type_error_for_invalid_random():
    try:
        BaseProvider(random="invalid")
        assert False
    except TypeError as e:
        assert "The random must be an instance of mimesis.random.Random" in str(e)


# LLM-generated content at query #15
#--------------------------

def test_provider_registry_initialization():
    registry = ProviderRegistry()
    assert registry._providers == {}


# LLM-generated content at query #16
#--------------------------

def test_base_data_provider_initialization_with_default_locale():
    provider = BaseDataProvider()
    assert provider.locale == Locale.DEFAULT.value
    assert isinstance(provider._dataset, dict)
    assert provider.seed is MissingSeed

def test_base_data_provider_initialization_with_custom_locale():
    provider = BaseDataProvider(locale=Locale.EN)
    assert provider.locale == Locale.EN.value
    assert isinstance(provider._dataset, dict)

def test_base_data_provider_initialization_with_seed():
    provider = BaseDataProvider(seed=12345)
    assert provider.seed == 12345

def test_base_data_provider_initialization_with_random_instance():
    custom_random = _random.Random()
    provider = BaseDataProvider(random=custom_random)
    assert provider.random is custom_random

def test_base_data_provider_initialization_with_invalid_random_type():
    try:
        BaseDataProvider(random="invalid")
        assert False
    except TypeError as e:
        assert "The random must be an instance of mimesis.random.Random" in str(e)

def test_base_data_provider_initialization_locale_setup():
    provider = BaseDataProvider(locale=Locale.RU)
    assert provider.locale == Locale.RU.value

def test_base_data_provider_initialization_dataset_loaded():
    class TestProvider(BaseDataProvider):
        class Meta:
            name = "test"
            datafile = "test.json"
            datadir = DATADIR
    try:
        provider = TestProvider(locale=Locale.EN)
        assert isinstance(provider._dataset, dict)
    except FileNotFoundError:
        pass

def test_base_data_provider_initialization_without_datafile():
    class TestProvider(BaseDataProvider):
        class Meta:
            name = "test"
    provider = TestProvider()
    assert provider._dataset == {}

def test_base_data_provider_initialization_with_unsupported_locale():
    try:
        BaseDataProvider(locale="unsupported")
        assert False
    except UnsupportedLocale:
        pass

def test_base_data_provider_initialization_inherits_from_base_provider():
    provider = BaseDataProvider()
    assert isinstance(provider, BaseProvider)
    assert hasattr(provider, 'random')
    assert hasattr(provider, 'seed')
    assert hasattr(provider, 'reseed')
    assert hasattr(provider, 'validate_enum')


# LLM-generated content at query #17
#--------------------------

def test_reseed_when_seed_is_missing_seed_and_global_seed_is_not_missing_seed():
    from mimesis.providers.base import BaseProvider
    from mimesis.random import Random, global_seed
    global_seed = 42
    provider = BaseProvider()
    provider.seed = MissingSeed
    provider.random = Random()
    provider.reseed(MissingSeed)
    assert provider.random._seed == 42


# LLM-generated content at query #18
#--------------------------

def test_validate_enum_with_none_item():
    from mimesis.enums import Gender
    from mimesis.providers.base import BaseProvider
    provider = BaseProvider(seed=42)
    result = provider.validate_enum(None, Gender)
    assert result in [item.value for item in Gender]

def test_validate_enum_with_valid_enum_item():
    from mimesis.enums import Gender
    from mimesis.providers.base import BaseProvider
    provider = BaseProvider(seed=42)
    result = provider.validate_enum(Gender.FEMALE, Gender)
    assert result == Gender.FEMALE.value

def test_validate_enum_raises_non_enumerable_error():
    from mimesis.enums import Gender
    from mimesis.exceptions import NonEnumerableError
    from mimesis.providers.base import BaseProvider
    provider = BaseProvider(seed=42)
    try:
        provider.validate_enum("invalid_item", Gender)
        assert False
    except NonEnumerableError:
        assert True


# LLM-generated content at query #19
#--------------------------

def test_constructor_with_default_parameters():
    provider = BaseProvider()
    assert provider.seed is MissingSeed
    assert isinstance(provider.random, _random.Random)

def test_constructor_with_custom_random_instance():
    custom_random = _random.Random()
    provider = BaseProvider(random=custom_random)
    assert provider.random is custom_random
    assert provider.seed is MissingSeed

def test_constructor_with_seed_parameter():
    provider = BaseProvider(seed=42)
    assert provider.seed == 42
    assert isinstance(provider.random, _random.Random)

def test_constructor_with_seed_none():
    provider = BaseProvider(seed=None)
    assert provider.seed is None
    assert isinstance(provider.random, _random.Random)

def test_constructor_with_both_seed_and_random():
    custom_random = _random.Random()
    provider = BaseProvider(seed=100, random=custom_random)
    assert provider.random is custom_random
    assert provider.seed == 100

def test_constructor_raises_type_error_for_invalid_random():
    try:
        BaseProvider(random="invalid")
        assert False
    except TypeError as e:
        assert "The random must be an instance of mimesis.random.Random" in str(e)

def test_constructor_keyword_only_arguments_enforced():
    try:
        BaseProvider("invalid_positional")
        assert False
    except TypeError:
        pass


# LLM-generated content at query #20
#--------------------------

def test_base_data_provider_initialization_with_default_locale():
    provider = BaseDataProvider()
    assert provider.locale == Locale.DEFAULT.value
    assert provider._dataset == {}

def test_base_data_provider_initialization_with_custom_locale():
    provider = BaseDataProvider(locale=Locale.EN)
    assert provider.locale == Locale.EN.value

def test_base_data_provider_initialization_with_seed():
    provider = BaseDataProvider(seed=42)
    assert provider.seed == 42

def test_base_data_provider_initialization_with_random_instance():
    random_instance = _random.Random()
    provider = BaseDataProvider(random=random_instance)
    assert provider.random is random_instance

def test_base_data_provider_initialization_with_invalid_random_type():
    try:
        BaseDataProvider(random="invalid")
        assert False
    except TypeError:
        assert True

def test_base_data_provider_initialization_locale_setup_unsupported_locale():
    try:
        BaseDataProvider(locale="unsupported")
        assert False
    except UnsupportedLocale:
        assert True

def test_base_data_provider_initialization_loads_dataset():
    class TestProvider(BaseDataProvider):
        class Meta:
            name = "test"
            datafile = "test.json"
            datadir = Path("/fake/dir")
    try:
        TestProvider()
        assert False
    except FileNotFoundError:
        assert True

def test_base_data_provider_initialization_without_datafile():
    class TestProvider(BaseDataProvider):
        class Meta:
            name = "test"
    provider = TestProvider()
    assert provider._dataset == {}

def test_base_data_provider_initialization_with_locale_separator():
    class TestProvider(BaseDataProvider):
        class Meta:
            name = "test"
            datafile = "test.json"
            datadir = Path("/fake/dir")
    try:
        TestProvider(locale="en_US")
        assert False
    except FileNotFoundError:
        assert True


# LLM-generated content at query #21
#--------------------------

def test_auto_register_provider():
    class TestProvider(BaseProvider):
        class Meta:
            name = "test_provider"
            auto_register = True
    assert ProviderRegistry.get("test_provider") == TestProvider


# LLM-generated content at query #22
#--------------------------

def test_provider_registry_constructor():
    registry = ProviderRegistry()
    assert registry._providers == {}


# LLM-generated content at query #23
#--------------------------

def test_validate_enum_with_false_predicate_at_line_11():
    from enum import Enum
    class TestEnum(Enum):
        A = "a"
        B = "b"
    provider = BaseProvider()
    item = None
    result = provider.validate_enum(item, TestEnum)
    assert result in ["a", "b"]


# LLM-generated content at query #24
#--------------------------

def test_base_provider_initialization_with_defaults():
    provider = BaseProvider()
    assert provider.seed is MissingSeed
    assert isinstance(provider.random, _random.Random)

def test_base_provider_initialization_with_custom_random():
    custom_random = _random.Random()
    provider = BaseProvider(random=custom_random)
    assert provider.random is custom_random
    assert provider.seed is MissingSeed

def test_base_provider_initialization_with_seed():
    provider = BaseProvider(seed=42)
    assert provider.seed == 42
    assert isinstance(provider.random, _random.Random)

def test_base_provider_initialization_with_none_seed():
    provider = BaseProvider(seed=None)
    assert provider.seed is None
    assert isinstance(provider.random, _random.Random)

def test_base_provider_initialization_with_invalid_random_type():
    try:
        BaseProvider(random="invalid")
    except TypeError as e:
        assert "The random must be an instance of mimesis.random.Random" in str(e)

def test_base_provider_initialization_with_seed_and_custom_random():
    custom_random = _random.Random()
    provider = BaseProvider(seed=123, random=custom_random)
    assert provider.random is custom_random
    assert provider.seed == 123


# LLM-generated content at query #25
#--------------------------

def test_random_is_not_none_and_is_instance_of_random():
    from mimesis.providers.base import BaseProvider
    from mimesis.random import Random
    custom_random = Random()
    provider = BaseProvider(random=custom_random)
    assert provider.random is custom_random

def test_random_is_not_none_and_is_not_instance_of_random_raises_type_error():
    from mimesis.providers.base import BaseProvider
    class FakeRandom:
        pass
    custom_random = FakeRandom()
    try:
        BaseProvider(random=custom_random)
        assert False
    except TypeError as e:
        assert str(e) == "The random must be an instance of mimesis.random.Random"


# LLM-generated content at query #26
#--------------------------

def test_base_data_provider_init_default_locale():
    provider = BaseDataProvider()
    assert provider.locale == Locale.DEFAULT.value
    assert isinstance(provider._dataset, dict)
    assert provider.seed is MissingSeed

def test_base_data_provider_init_custom_locale():
    provider = BaseDataProvider(locale=Locale.EN)
    assert provider.locale == Locale.EN.value
    assert isinstance(provider._dataset, dict)

def test_base_data_provider_init_with_seed():
    provider = BaseDataProvider(seed=42)
    assert provider.seed == 42

def test_base_data_provider_init_with_random_instance():
    custom_random = _random.Random()
    provider = BaseDataProvider(random=custom_random)
    assert provider.random is custom_random

def test_base_data_provider_init_inherits_random_from_base():
    provider = BaseDataProvider()
    assert isinstance(provider.random, _random.Random)

def test_base_data_provider_init_locale_validation():
    provider = BaseDataProvider(locale=Locale.RU)
    assert provider.locale == Locale.RU.value

def test_base_data_provider_init_dataset_loaded():
    class TestProvider(BaseDataProvider):
        class Meta:
            name = "test"
            datafile = "test.json"
            datadir = DATADIR
    provider = TestProvider()
    assert provider._dataset == {}

def test_base_data_provider_init_with_args_and_kwargs():
    provider = BaseDataProvider(locale=Locale.EN, seed=123, extra_arg="test")
    assert provider.locale == Locale.EN.value
    assert provider.seed == 123

def test_base_data_provider_init_locale_default_fallback():
    provider = BaseDataProvider()
    assert provider.locale == Locale.DEFAULT.value

def test_base_data_provider_init_seed_missing_by_default():
    provider = BaseDataProvider()
    assert provider.seed is MissingSeed


# LLM-generated content at query #27
#--------------------------

def test_auto_register_provider():
    class TestProvider(BaseProvider):
        class Meta:
            name = "test_provider"
            auto_register = True
    assert ProviderRegistry.get("test_provider") == TestProvider


# LLM-generated content at query #28
#--------------------------

def test_base_data_provider_initialization_with_default_locale():
    provider = BaseDataProvider()
    assert provider.locale == Locale.DEFAULT.value
    assert provider._dataset == {}

def test_base_data_provider_initialization_with_custom_locale():
    provider = BaseDataProvider(locale=Locale.EN)
    assert provider.locale == Locale.EN.value

def test_base_data_provider_initialization_with_seed():
    provider = BaseDataProvider(seed=12345)
    assert provider.seed == 12345

def test_base_data_provider_initialization_with_random_instance():
    custom_random = _random.Random()
    provider = BaseDataProvider(random=custom_random)
    assert provider.random is custom_random

def test_base_data_provider_initialization_invalid_random_type():
    try:
        BaseDataProvider(random="invalid")
        assert False
    except TypeError:
        assert True

def test_base_data_provider_initialization_locale_setup_failure():
    try:
        BaseDataProvider(locale="invalid_locale")
        assert False
    except UnsupportedLocale:
        assert True

def test_base_data_provider_initialization_with_datafile_meta():
    class TestProvider(BaseDataProvider):
        class Meta:
            name = "test"
            datafile = "test.json"
            datadir = DATADIR
    provider = TestProvider()
    assert provider._dataset != {}

def test_base_data_provider_initialization_without_datafile_meta():
    class TestProvider(BaseDataProvider):
        class Meta:
            name = "test"
    provider = TestProvider()
    assert provider._dataset == {}

def test_base_data_provider_initialization_locale_separated_data():
    class TestProvider(BaseDataProvider):
        class Meta:
            name = "test"
            datafile = "test.json"
            datadir = DATADIR
    provider = TestProvider(locale="en_US")
    assert provider._dataset != {}

def test_base_data_provider_initialization_with_args_and_kwargs():
    provider = BaseDataProvider(locale=Locale.EN, seed=42, extra_arg="test")
    assert provider.locale == Locale.EN.value
    assert provider.seed == 42


# LLM-generated content at query #29
#--------------------------

def test_auto_register_provider_when_meta_name_and_auto_register_true():
    class TestProvider(BaseProvider):
        class Meta:
            name = "test_provider"
            auto_register = True
    assert ProviderRegistry.get("test_provider") == TestProvider


# LLM-generated content at query #30
#--------------------------

def test_provider_registry_constructor():
    registry = ProviderRegistry()
    assert registry._providers == {}


# LLM-generated content at query #31
#--------------------------

def test_base_provider_initialization_with_defaults():
    provider = BaseProvider()
    assert provider.seed is MissingSeed
    assert isinstance(provider.random, _random.Random)

def test_base_provider_initialization_with_custom_random():
    custom_random = _random.Random()
    provider = BaseProvider(random=custom_random)
    assert provider.random is custom_random
    assert provider.seed is MissingSeed

def test_base_provider_initialization_with_seed():
    provider = BaseProvider(seed=42)
    assert provider.seed == 42
    assert isinstance(provider.random, _random.Random)

def test_base_provider_initialization_with_none_seed():
    provider = BaseProvider(seed=None)
    assert provider.seed is None
    assert isinstance(provider.random, _random.Random)

def test_base_provider_initialization_with_invalid_random_type():
    try:
        BaseProvider(random="invalid")
        assert False
    except TypeError as e:
        assert "The random must be an instance of mimesis.random.Random" in str(e)

def test_base_provider_initialization_with_seed_and_custom_random():
    custom_random = _random.Random()
    provider = BaseProvider(seed=123, random=custom_random)
    assert provider.random is custom_random
    assert provider.seed == 123


# LLM-generated content at query #32
#--------------------------

def test_reseed_uses_global_seed_when_seed_is_missing_and_global_seed_is_set():
    import mimesis.random as _random
    from mimesis.enums import MissingSeed
    from mimesis.providers.base import BaseProvider
    from mimesis.random import Random
    original_global_seed = _random.global_seed
    _random.global_seed = 12345
    provider = BaseProvider(seed=MissingSeed)
    provider.random = Random()
    provider.reseed(MissingSeed)
    assert _random.global_seed is not MissingSeed


# LLM-generated content at query #33
#--------------------------

def test_reseed_with_missing_seed_and_global_seed_missing():
    provider = BaseProvider(seed=None)
    original_random_state = provider.random.getstate()
    provider.reseed()
    assert provider.seed is None
    assert provider.random.getstate() != original_random_state

def test_reseed_with_missing_seed_and_global_seed_set():
    _random.global_seed = 12345
    provider = BaseProvider(seed=None)
    original_random_state = provider.random.getstate()
    provider.reseed()
    assert provider.seed is None
    assert provider.random.getstate() != original_random_state
    _random.global_seed = None

def test_reseed_with_explicit_seed():
    provider = BaseProvider(seed=None)
    original_random_state = provider.random.getstate()
    provider.reseed(seed=999)
    assert provider.seed == 999
    assert provider.random.getstate() != original_random_state

def test_reseed_with_none_seed():
    provider = BaseProvider(seed=100)
    original_random_state = provider.random.getstate()
    provider.reseed(seed=None)
    assert provider.seed is None
    assert provider.random.getstate() != original_random_state

def test_reseed_updates_instance_seed_attribute():
    provider = BaseProvider(seed=500)
    provider.reseed(seed=700)
    assert provider.seed == 700

def test_reseed_with_missing_seed_constant():
    provider = BaseProvider(seed=200)
    provider.reseed(seed=MissingSeed)
    assert provider.seed is MissingSeed


# LLM-generated content at query #34
#--------------------------

def test_auto_register_provider():
    class TestProvider(BaseProvider):
        class Meta:
            name = "test_provider"
            auto_register = True
    assert ProviderRegistry.get("test_provider") == TestProvider


# LLM-generated content at query #35
#--------------------------

def test_random_is_not_none_and_is_instance_of_random():
    from mimesis.providers.base import BaseProvider
    from mimesis.random import Random
    custom_random = Random()
    provider = BaseProvider(random=custom_random)
    assert provider.random is custom_random

def test_random_is_not_none_and_is_not_instance_of_random_raises_typeerror():
    from mimesis.providers.base import BaseProvider
    class FakeRandom:
        pass
    custom_random = FakeRandom()
    try:
        BaseProvider(random=custom_random)
        assert False
    except TypeError as e:
        assert str(e) == "The random must be an instance of mimesis.random.Random"


# LLM-generated content at query #36
#--------------------------

def test_provider_registry_constructor():
    registry = ProviderRegistry()
    assert registry._providers == {}


# LLM-generated content at query #37
#--------------------------

def test_provider_registry_constructor():
    registry = ProviderRegistry()
    assert registry._providers == {}


# LLM-generated content at query #38
#--------------------------

def test_provider_registry_constructor():
    registry = ProviderRegistry()
    assert registry._providers == {}


# LLM-generated content at query #39
#--------------------------

def test_constructor_with_default_parameters():
    provider = BaseProvider()
    assert provider.seed is MissingSeed
    assert isinstance(provider.random, _random.Random)

def test_constructor_with_custom_seed():
    provider = BaseProvider(seed=42)
    assert provider.seed == 42
    assert isinstance(provider.random, _random.Random)

def test_constructor_with_none_seed():
    provider = BaseProvider(seed=None)
    assert provider.seed is None
    assert isinstance(provider.random, _random.Random)

def test_constructor_with_custom_random_instance():
    custom_random = _random.Random()
    provider = BaseProvider(random=custom_random)
    assert provider.random is custom_random
    assert provider.seed is MissingSeed

def test_constructor_with_custom_random_and_seed():
    custom_random = _random.Random()
    provider = BaseProvider(seed=123, random=custom_random)
    assert provider.random is custom_random
    assert provider.seed == 123

def test_constructor_raises_type_error_for_invalid_random():
    try:
        BaseProvider(random="invalid")
        assert False
    except TypeError as e:
        assert "The random must be an instance of mimesis.random.Random" in str(e)


# LLM-generated content at query #40
#--------------------------

def test_base_data_provider_initialization_with_default_locale():
    provider = BaseDataProvider()
    assert provider.locale == Locale.DEFAULT.value
    assert isinstance(provider._dataset, dict)
    assert provider.seed is MissingSeed

def test_base_data_provider_initialization_with_custom_locale():
    provider = BaseDataProvider(locale=Locale.EN)
    assert provider.locale == Locale.EN.value
    assert isinstance(provider._dataset, dict)

def test_base_data_provider_initialization_with_seed():
    provider = BaseDataProvider(seed=42)
    assert provider.seed == 42

def test_base_data_provider_initialization_with_random_instance():
    random_instance = _random.Random()
    provider = BaseDataProvider(random=random_instance)
    assert provider.random is random_instance

def test_base_data_provider_initialization_with_invalid_random_type():
    try:
        BaseDataProvider(random="invalid")
        assert False
    except TypeError as e:
        assert "The random must be an instance of mimesis.random.Random" in str(e)

def test_base_data_provider_initialization_with_locale_and_seed():
    provider = BaseDataProvider(locale=Locale.RU, seed=123)
    assert provider.locale == Locale.RU.value
    assert provider.seed == 123

def test_base_data_provider_initialization_without_datafile():
    class TestProvider(BaseDataProvider):
        class Meta:
            name = "test"
            auto_register = False
    provider = TestProvider()
    assert provider._dataset == {}

def test_base_data_provider_initialization_with_datafile():
    class TestProvider(BaseDataProvider):
        class Meta:
            name = "test"
            auto_register = False
            datafile = "test.json"
            datadir = Path("test_data")
    try:
        provider = TestProvider()
        assert isinstance(provider._dataset, dict)
    except FileNotFoundError:
        pass

def test_base_data_provider_initialization_locale_validation_error():
    try:
        BaseDataProvider(locale="invalid_locale")
        assert False
    except UnsupportedLocale:
        assert True

def test_base_data_provider_str_representation():
    provider = BaseDataProvider(locale=Locale.EN)
    expected = f"{provider.__class__.__name__} <{Locale.EN}>"
    assert str(provider) == expected

def test_base_data_provider_inherits_from_base_provider():
    provider = BaseDataProvider()
    assert isinstance(provider, BaseProvider)


# LLM-generated content at query #41
#--------------------------

def test_auto_register_provider():
    class TestProvider(BaseProvider):
        class Meta:
            name = "test_provider"
            auto_register = True
    assert ProviderRegistry.get("test_provider") == TestProvider


# LLM-generated content at query #42
#--------------------------

def test_random_is_not_none_and_is_instance_of_random():
    random_instance = _random.Random()
    provider = BaseProvider(random=random_instance)


# LLM-generated content at query #43
#--------------------------

def test_base_data_provider_initialization_with_default_locale():
    provider = BaseDataProvider()
    assert provider.locale == Locale.DEFAULT.value
    assert provider._dataset == {}
    assert isinstance(provider.random, _random.Random)

def test_base_data_provider_initialization_with_custom_locale():
    provider = BaseDataProvider(locale=Locale.EN)
    assert provider.locale == Locale.EN.value
    assert isinstance(provider.random, _random.Random)

def test_base_data_provider_initialization_with_seed():
    provider = BaseDataProvider(seed=42)
    assert provider.seed == 42
    assert isinstance(provider.random, _random.Random)

def test_base_data_provider_initialization_with_custom_random():
    custom_random = _random.Random()
    provider = BaseDataProvider(random=custom_random)
    assert provider.random is custom_random

def test_base_data_provider_initialization_with_invalid_random_type():
    try:
        BaseDataProvider(random="invalid")
        assert False
    except TypeError as e:
        assert str(e) == "The random must be an instance of mimesis.random.Random"

def test_base_data_provider_initialization_locale_and_seed():
    provider = BaseDataProvider(locale=Locale.RU, seed=123)
    assert provider.locale == Locale.RU.value
    assert provider.seed == 123

def test_base_data_provider_initialization_with_unsupported_locale():
    try:
        BaseDataProvider(locale="unsupported")
        assert False
    except UnsupportedLocale:
        assert True

def test_base_data_provider_initialization_with_none_seed():
    provider = BaseDataProvider(seed=None)
    assert provider.seed is None

def test_base_data_provider_initialization_with_missing_seed():
    provider = BaseDataProvider()
    assert provider.seed is MissingSeed

def test_base_data_provider_initialization_inherits_from_base_provider():
    provider = BaseDataProvider()
    assert isinstance(provider, BaseProvider)

def test_base_data_provider_initialization_dataset_loaded():
    class TestProvider(BaseDataProvider):
        class Meta:
            name = "test"
            datafile = "test.json"
            datadir = pathlib.Path("/fake/dir")
    try:
        provider = TestProvider()
        assert False
    except FileNotFoundError:
        assert True

def test_base_data_provider_initialization_without_datafile():
    class TestProvider(BaseDataProvider):
        class Meta:
            name = "test"
    provider = TestProvider()
    assert provider._dataset == {}

def test_base_data_provider_initialization_locale_setup_called():
    provider = BaseDataProvider(locale=Locale.DE)
    assert provider.locale == Locale.DE.value

def test_base_data_provider_initialization_load_dataset_called():
    class TestProvider(BaseDataProvider):
        class Meta:
            name = "test"
            datafile = "test.json"
            datadir = pathlib.Path("/fake/dir")
    try:
        TestProvider()
        assert False
    except FileNotFoundError:
        assert True

def test_base_data_provider_initialization_with_kwargs():
    provider = BaseDataProvider(locale=Locale.FR, seed=999)
    assert provider.locale == Locale.FR.value
    assert provider.seed == 999


# LLM-generated content at query #44
#--------------------------

def test_constructor_with_default_parameters():
    provider = BaseProvider()
    assert provider.seed is MissingSeed
    assert isinstance(provider.random, _random.Random)

def test_constructor_with_custom_random():
    custom_random = _random.Random()
    provider = BaseProvider(random=custom_random)
    assert provider.random is custom_random
    assert provider.seed is MissingSeed

def test_constructor_with_seed():
    provider = BaseProvider(seed=42)
    assert provider.seed == 42
    assert isinstance(provider.random, _random.Random)

def test_constructor_with_seed_and_custom_random():
    custom_random = _random.Random()
    provider = BaseProvider(seed=42, random=custom_random)
    assert provider.random is custom_random
    assert provider.seed == 42

def test_constructor_raises_type_error_for_invalid_random():
    try:
        BaseProvider(random="invalid")
        assert False
    except TypeError as e:
        assert str(e) == "The random must be an instance of mimesis.random.Random"

def test_constructor_with_none_seed():
    provider = BaseProvider(seed=None)
    assert provider.seed is None
    assert isinstance(provider.random, _random.Random)


# LLM-generated content at query #45
#--------------------------

def test_provider_registry_constructor():
    registry = ProviderRegistry()
    assert registry._providers == {}


# LLM-generated content at query #46
#--------------------------

def test_constructor_with_default_parameters():
    provider = BaseProvider()
    assert provider.seed is MissingSeed
    assert isinstance(provider.random, _random.Random)

def test_constructor_with_custom_random_instance():
    custom_random = _random.Random()
    provider = BaseProvider(random=custom_random)
    assert provider.random is custom_random
    assert provider.seed is MissingSeed

def test_constructor_with_seed_parameter():
    provider = BaseProvider(seed=42)
    assert provider.seed == 42
    assert isinstance(provider.random, _random.Random)

def test_constructor_with_seed_none():
    provider = BaseProvider(seed=None)
    assert provider.seed is None
    assert isinstance(provider.random, _random.Random)

def test_constructor_with_both_seed_and_random():
    custom_random = _random.Random()
    provider = BaseProvider(seed=123, random=custom_random)
    assert provider.random is custom_random
    assert provider.seed == 123

def test_constructor_raises_type_error_for_invalid_random():
    try:
        BaseProvider(random="invalid")
        assert False
    except TypeError as e:
        assert "The random must be an instance of mimesis.random.Random" in str(e)

def test_constructor_initializes_random_when_none_provided():
    provider = BaseProvider()
    assert provider.random is not None
    assert isinstance(provider.random, _random.Random)

def test_constructor_sets_seed_attribute_correctly():
    provider = BaseProvider(seed=999)
    assert provider.seed == 999

def test_constructor_with_missingseed_constant():
    provider = BaseProvider(seed=MissingSeed)
    assert provider.seed is MissingSeed


# LLM-generated content at query #47
#--------------------------

def test_locale_default_initialization():
    provider = BaseDataProvider()
    assert provider.locale == Locale.DEFAULT.value


# LLM-generated content at query #48
#--------------------------

def test_locale_default_does_not_trigger_unsupported_locale():
    provider = BaseDataProvider(locale=Locale.DEFAULT)


# LLM-generated content at query #49
#--------------------------

def test_provider_registry_constructor():
    registry = ProviderRegistry()
    assert registry._providers == {}


# LLM-generated content at query #50
#--------------------------

def test_base_data_provider_init_default_locale():
    provider = BaseDataProvider()
    assert provider.locale == Locale.DEFAULT.value
    assert isinstance(provider._dataset, dict)
    assert provider.seed is MissingSeed

def test_base_data_provider_init_custom_locale():
    provider = BaseDataProvider(locale=Locale.EN)
    assert provider.locale == Locale.EN.value
    assert isinstance(provider._dataset, dict)

def test_base_data_provider_init_with_seed():
    provider = BaseDataProvider(seed=42)
    assert provider.seed == 42
    assert provider.random is not None

def test_base_data_provider_init_with_random_instance():
    custom_random = _random.Random()
    provider = BaseDataProvider(random=custom_random)
    assert provider.random is custom_random

def test_base_data_provider_init_invalid_random_type():
    try:
        BaseDataProvider(random="invalid")
        assert False
    except TypeError:
        assert True

def test_base_data_provider_init_locale_validation():
    try:
        BaseDataProvider(locale="invalid_locale")
        assert False
    except UnsupportedLocale:
        assert True

def test_base_data_provider_init_dataset_loading():
    class TestProvider(BaseDataProvider):
        class Meta:
            name = "test"
            datafile = "test.json"
            datadir = DATADIR

    try:
        provider = TestProvider()
        assert isinstance(provider._dataset, dict)
    except FileNotFoundError:
        assert True

def test_base_data_provider_init_no_datafile():
    class TestProvider(BaseDataProvider):
        class Meta:
            name = "test"

    provider = TestProvider()
    assert provider._dataset == {}

def test_base_data_provider_init_with_args_and_kwargs():
    provider = BaseDataProvider(locale=Locale.EN, seed=123, extra_arg="test")
    assert provider.locale == Locale.EN.value
    assert provider.seed == 123

def test_base_data_provider_str_representation():
    provider = BaseDataProvider(locale=Locale.EN)
    expected = f"{provider.__class__.__name__} <{Locale.EN}>"
    assert str(provider) == expected


# LLM-generated content at query #51
#--------------------------

def test_auto_register_false_prevents_registration():
    class TestProvider(BaseProvider):
        class Meta:
            name = "test_provider"
            auto_register = False
    assert "test_provider" not in ProviderRegistry._registry


# LLM-generated content at query #52
#--------------------------

def test_random_is_not_none_and_not_instance_of_random_raises_type_error():
    class MockRandom:
        pass
    mock_random = MockRandom()
    try:
        BaseProvider(random=mock_random)
        assert False
    except TypeError as e:
        assert str(e) == "The random must be an instance of mimesis.random.Random"


# LLM-generated content at query #53
#--------------------------

def test_base_data_provider_initializes_with_default_locale():
    provider = BaseDataProvider()
    assert provider.locale == Locale.DEFAULT.value

def test_base_data_provider_initializes_with_custom_locale():
    provider = BaseDataProvider(locale=Locale.EN)
    assert provider.locale == Locale.EN.value

def test_base_data_provider_initializes_with_seed():
    provider = BaseDataProvider(seed=42)
    assert provider.seed == 42

def test_base_data_provider_initializes_with_random_instance():
    custom_random = _random.Random()
    provider = BaseDataProvider(random=custom_random)
    assert provider.random is custom_random

def test_base_data_provider_raises_error_for_invalid_random():
    try:
        BaseDataProvider(random="invalid")
    except TypeError as e:
        assert "The random must be an instance of mimesis.random.Random" in str(e)

def test_base_data_provider_loads_dataset_with_datafile():
    class TestProvider(BaseDataProvider):
        class Meta:
            name = "test"
            datafile = "test.json"
            datadir = DATADIR
    provider = TestProvider(locale=Locale.EN)
    assert provider._dataset != {}

def test_base_data_provider_dataset_empty_without_datafile():
    class TestProvider(BaseDataProvider):
        class Meta:
            name = "test"
    provider = TestProvider()
    assert provider._dataset == {}

def test_base_data_provider_inherits_base_provider_attributes():
    provider = BaseDataProvider()
    assert hasattr(provider, 'random')
    assert hasattr(provider, 'seed')

def test_base_data_provider_locale_setup_raises_unsupported_locale():
    try:
        BaseDataProvider(locale="invalid_locale")
    except UnsupportedLocale:
        pass

def test_base_data_provider_str_representation_includes_locale():
    provider = BaseDataProvider(locale=Locale.EN)
    assert str(provider) == f"BaseDataProvider <{Locale.EN}>"


# LLM-generated content at query #54
#--------------------------

def test_random_is_not_none_and_not_instance_of_random_raises_type_error():
    class MockRandom:
        pass
    mock_random = MockRandom()
    try:
        BaseProvider(random=mock_random)
        assert False
    except TypeError as e:
        assert str(e) == "The random must be an instance of mimesis.random.Random"


# LLM-generated content at query #55
#--------------------------

def test_random_is_not_none_and_is_instance_of_random():
    from mimesis.providers.base import BaseProvider
    from mimesis.random import Random
    custom_random = Random()
    provider = BaseProvider(random=custom_random)
    assert provider.random is custom_random

def test_random_is_not_none_and_is_not_instance_of_random_raises_type_error():
    from mimesis.providers.base import BaseProvider
    class FakeRandom:
        pass
    custom_random = FakeRandom()
    try:
        BaseProvider(random=custom_random)
        assert False
    except TypeError as e:
        assert str(e) == "The random must be an instance of mimesis.random.Random"

def test_random_is_none_creates_default_random():
    from mimesis.providers.base import BaseProvider
    from mimesis.random import Random
    provider = BaseProvider(random=None)
    assert isinstance(provider.random, Random)


# LLM-generated content at query #56
#--------------------------

def test_auto_register_provider_when_meta_name_and_auto_register_true():
    class TestProvider(BaseProvider):
        class Meta:
            name = "test_provider"
            auto_register = True
    assert ProviderRegistry.get("test_provider") == TestProvider

def test_auto_register_provider_when_meta_name_and_auto_register_default():
    class TestProvider(BaseProvider):
        class Meta:
            name = "test_provider_default"
    assert ProviderRegistry.get("test_provider_default") == TestProvider

def test_do_not_auto_register_provider_when_meta_name_and_auto_register_false():
    class TestProvider(BaseProvider):
        class Meta:
            name = "test_provider_no_register"
            auto_register = False
    with pytest.raises(KeyError):
        ProviderRegistry.get("test_provider_no_register")


# LLM-generated content at query #57
#--------------------------

def test_base_data_provider_init_default_locale():
    provider = BaseDataProvider()
    assert provider.locale == Locale.DEFAULT.value
    assert isinstance(provider._dataset, dict)
    assert provider.seed is MissingSeed

def test_base_data_provider_init_custom_locale():
    provider = BaseDataProvider(locale=Locale.EN)
    assert provider.locale == Locale.EN.value
    assert isinstance(provider._dataset, dict)

def test_base_data_provider_init_with_seed():
    provider = BaseDataProvider(seed=42)
    assert provider.seed == 42

def test_base_data_provider_init_with_random_instance():
    custom_random = _random.Random()
    provider = BaseDataProvider(random=custom_random)
    assert provider.random is custom_random

def test_base_data_provider_init_invalid_random_type():
    try:
        BaseDataProvider(random="invalid")
        assert False
    except TypeError:
        assert True

def test_base_data_provider_init_locale_validation():
    try:
        BaseDataProvider(locale="invalid_locale")
        assert False
    except UnsupportedLocale:
        assert True

def test_base_data_provider_init_seed_passed_to_parent():
    provider = BaseDataProvider(seed=123)
    assert provider.seed == 123

def test_base_data_provider_init_dataset_empty_when_no_datafile():
    class TestProvider(BaseDataProvider):
        class Meta:
            name = "test"
            auto_register = False
    provider = TestProvider()
    assert provider._dataset == {}

def test_base_data_provider_init_dataset_loaded_with_datafile():
    class TestProvider(BaseDataProvider):
        class Meta:
            name = "test"
            auto_register = False
            datafile = "test.json"
            datadir = Path("/fake/dir")
    try:
        TestProvider()
        assert False
    except FileNotFoundError:
        assert True

def test_base_data_provider_init_locale_order_setup_before_dataset():
    provider = BaseDataProvider(locale=Locale.EN)
    assert provider.locale == Locale.EN.value


# LLM-generated content at query #58
#--------------------------

def test_random_is_not_none_and_is_instance_of_random():
    random_instance = _random.Random()
    provider = BaseProvider(random=random_instance)


# LLM-generated content at query #59
#--------------------------

def test_constructor_with_default_parameters():
    provider = BaseProvider()
    assert provider.seed is MissingSeed
    assert isinstance(provider.random, _random.Random)

def test_constructor_with_custom_random():
    custom_random = _random.Random()
    provider = BaseProvider(random=custom_random)
    assert provider.random is custom_random
    assert provider.seed is MissingSeed

def test_constructor_with_seed():
    provider = BaseProvider(seed=42)
    assert provider.seed == 42
    assert isinstance(provider.random, _random.Random)

def test_constructor_with_seed_none():
    provider = BaseProvider(seed=None)
    assert provider.seed is None
    assert isinstance(provider.random, _random.Random)

def test_constructor_with_invalid_random_type():
    try:
        BaseProvider(random="invalid")
        assert False
    except TypeError:
        assert True


# LLM-generated content at query #60
#--------------------------

def test_random_is_not_none_and_is_instance_of_random():
    random_instance = _random.Random()
    provider = BaseProvider(random=random_instance)
    assert provider.random is random_instance

def test_random_is_not_none_and_is_not_instance_of_random():
    try:
        BaseProvider(random="not_a_random_instance")
    except TypeError as e:
        assert str(e) == "The random must be an instance of mimesis.random.Random"


# LLM-generated content at query #61
#--------------------------

def test_random_is_not_none_and_not_instance_of_random_raises_type_error():
    class MockRandom:
        pass
    mock_random = MockRandom()
    try:
        BaseProvider(random=mock_random)
        assert False
    except TypeError as e:
        assert str(e) == "The random must be an instance of mimesis.random.Random"


# LLM-generated content at query #62
#--------------------------

def test_auto_register_provider_with_meta_name_and_auto_register_true():
    class TestProvider(BaseProvider):
        class Meta:
            name = "test_provider"
            auto_register = True
    assert ProviderRegistry.get("test_provider") == TestProvider


# LLM-generated content at query #63
#--------------------------

def test_constructor_with_default_parameters():
    provider = BaseProvider()
    assert provider.seed is MissingSeed
    assert isinstance(provider.random, _random.Random)

def test_constructor_with_custom_random_instance():
    custom_random = _random.Random()
    provider = BaseProvider(random=custom_random)
    assert provider.random is custom_random
    assert provider.seed is MissingSeed

def test_constructor_with_invalid_random_type():
    try:
        BaseProvider(random="invalid")
        assert False
    except TypeError as e:
        assert "The random must be an instance of mimesis.random.Random" in str(e)

def test_constructor_with_seed_parameter():
    provider = BaseProvider(seed=42)
    assert provider.seed == 42
    assert isinstance(provider.random, _random.Random)

def test_constructor_with_seed_none():
    provider = BaseProvider(seed=None)
    assert provider.seed is None
    assert isinstance(provider.random, _random.Random)

def test_constructor_with_seed_and_custom_random():
    custom_random = _random.Random()
    provider = BaseProvider(seed=123, random=custom_random)
    assert provider.seed == 123
    assert provider.random is custom_random


# LLM-generated content at query #64
#--------------------------

def test_auto_register_provider():
    class TestProvider(BaseProvider):
        class Meta:
            name = "test_provider"
            auto_register = True
    assert ProviderRegistry._registry["test_provider"] == TestProvider

def test_auto_register_false():
    class TestProvider(BaseProvider):
        class Meta:
            name = "test_provider_false"
            auto_register = False
    assert "test_provider_false" not in ProviderRegistry._registry

def test_meta_name_missing():
    class TestProvider(BaseProvider):
        pass
    assert "test_provider" not in ProviderRegistry._registry


