####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_generic_constructor_default_locale_and_seed. Retrieved 2/4 statements.
# Failed to parse test_generic_constructor_custom_locale.
# Partially parsed test_generic_constructor_lazy_initialization. Retrieved 4/7 statements.


import mimesis.providers.generic as module_0


def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.locale
    var_2 = var_0.seed
    var_3 = var_0.random


def test_case_0():
    var_0 = 12345
    var_1 = module_0.Generic(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 12345


def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'person'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = 'address'
    var_5 = hasattr(var_0, var_4)
    var_6 = bool(var_5)
    assert var_6 is True


def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'generic'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(not var_2)
    assert var_3 is True


def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0._person
    var_2 = var_0.person
    var_3 = var_0.person


def test_case_0():
    var_0 = 999
    var_1 = module_0.Generic(seed=var_0)
    var_2 = var_1.person.seed
    assert var_2 == 999


def test_case_0():
    var_0 = None
    var_1 = module_0.Generic(seed=var_0)
    var_2 = var_1.seed
    assert var_2 is None


def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0._person
    var_2 = callable(var_1)
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = var_0.person
    var_5 = callable(var_4)
    var_6 = bool(not var_5)
    assert var_6 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_generic_constructor_default_locale_and_seed. Retrieved 2/4 statements.
# Failed to parse test_generic_constructor_custom_locale.
# Partially parsed test_generic_constructor_base_providers_instantiated. Retrieved 3/5 statements.
# Partially parsed test_generic_constructor_getattr_lazy_initialization. Retrieved 2/3 statements.
# Failed to parse test_generic_constructor_repr.



def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.locale
    var_2 = var_0.seed
    var_3 = var_0.random


def test_case_0():
    var_0 = 12345
    var_1 = module_0.Generic(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 12345


def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'person'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = 'address'
    var_5 = hasattr(var_0, var_4)
    var_6 = bool(var_5)
    assert var_6 is True
    var_7 = 'text'
    var_8 = hasattr(var_0, var_7)
    var_9 = bool(var_8)
    assert var_9 is True


def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'generic'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(not var_2)
    assert var_3 is True


def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0._person
    var_2 = callable(var_1)
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = var_0._address
    var_5 = callable(var_4)
    var_6 = bool(var_5)
    assert var_6 is True
    var_7 = var_0._text
    var_8 = callable(var_7)
    var_9 = bool(var_8)
    assert var_9 is True


def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.cryptographic
    var_2 = var_0.development


def test_case_0():
    var_0 = 999
    var_1 = module_0.Generic(seed=var_0)
    var_2 = var_1.cryptographic.seed
    assert var_2 == 999
    var_3 = var_1.development.seed
    assert var_3 == 999


def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.__dir__()
    var_2 = 'person'
    var_3 = bool('person' in var_1)
    assert var_3 is True
    var_4 = 'address'
    var_5 = bool('address' in var_1)
    assert var_5 is True
    var_6 = 'text'
    var_7 = bool('text' in var_1)
    assert var_7 is True
    var_8 = 'locale'
    var_9 = bool('locale' not in var_1)
    assert var_9 is True


def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.person
    var_2 = var_1.locale
    var_3 = bool(var_1.locale == var_0.locale)
    assert var_3 is True
    var_4 = var_1.seed
    var_5 = bool(var_1.seed == var_0.seed)
    assert var_5 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_add_provider_adds_custom_provider. Retrieved 5/9 statements.
# Partially parsed test_add_provider_raises_type_error_for_non_baseprovider_subclass. Retrieved 1/5 statements.
# Partially parsed test_add_provider_raises_type_error_for_generic. Retrieved 1/3 statements.
# Partially parsed test_add_provider_uses_meta_name. Retrieved 5/9 statements.
# Partially parsed test_add_provider_falls_back_to_class_name_lowercase. Retrieved 4/8 statements.
# Partially parsed test_add_provider_ignores_seed_kwarg. Retrieved 4/7 statements.
# Partially parsed test_add_provider_passes_extra_kwargs. Retrieved 2/7 statements.



def test_case_0():
    var_0 = 'custom'
    var_1 = module_0.Generic()
    var_2 = 'custom'
    var_3 = hasattr(var_1, var_2)
    var_4 = bool(var_3)
    assert var_4 is True
    var_5 = var_1.custom


def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'not_a_class'
    var_2 = {}
    var_3 = var_0.add_provider(var_1, **var_2)
    var_4 = 'The provider must be a class'


def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'The provider must be a subclass of mimesis.providers.BaseProvider'


def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'Cannot add Generic instance to itself.'


def test_case_0():
    var_0 = 'special'
    var_1 = module_0.Generic()
    var_2 = 'special'
    var_3 = hasattr(var_1, var_2)
    var_4 = bool(var_3)
    assert var_4 is True
    var_5 = var_1.special


def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'customprovider'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = var_0.customprovider


def test_case_0():
    var_0 = 'custom'
    var_1 = 12345
    var_2 = module_0.Generic(seed=var_1)
    var_3 = 99999
    var_4 = var_2.custom.seed
    assert var_4 == 12345


def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'test'
    var_2 = var_0.customprovider.extra
    assert var_2 == 'test'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_reseed_updates_seed_on_attached_providers. Retrieved 4/9 statements.
# Partially parsed test_reseed_after_adding_custom_provider. Retrieved 5/8 statements.



def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.seed
    var_2 = 12345
    var_3 = var_0.reseed(var_2)
    var_4 = var_0.seed
    var_5 = bool(var_0.seed != var_1)
    assert var_5 is True
    var_6 = var_0.seed
    assert var_6 == 12345


def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.__dir__()
    var_2 = 67890
    var_3 = var_0.reseed(var_2)


def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.seed
    var_2 = var_0.reseed()
    var_3 = var_0.seed
    var_4 = bool(var_0.seed != var_1)
    assert var_4 is True


def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 11111
    var_2 = var_0.reseed(var_1)
    var_3 = var_0.seed
    assert var_3 == 11111


def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'custom'
    var_2 = var_0.custom.seed
    var_3 = 22222
    var_4 = var_0.reseed(var_3)
    var_5 = var_0.custom.seed
    assert var_5 == 22222



# Parsed testcases at query #5
#--------------------------

# Failed to parse test_provider_registry_excludes_generic.




# Parsed testcases at query #6
#--------------------------






# Parsed testcases at query #7
#--------------------------

# Partially parsed test___getattr___returns_initialized_provider_for_existing_data_provider. Retrieved 4/5 statements.
# Partially parsed test___getattr___caches_provider_after_first_access. Retrieved 5/10 statements.
# Partially parsed test___getattr___handles_attribute_without_leading_underscore. Retrieved 4/5 statements.
# Partially parsed test___getattr___returns_none_for_non_callable_attribute. Retrieved 3/4 statements.



def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'non_existent'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None


def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'person_provider'
    var_2 = 'person'
    var_3 = var_0.__getattr__(var_2)
    assert var_3 == 'person_provider'


def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 0
    assert var_1 == 1
    var_2 = 'address'
    var_3 = var_0.__getattr__(var_2)
    assert var_3 == 'cached_provider'
    var_4 = var_0.__getattr__(var_2)
    assert var_4 == 'cached_provider'


def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'text_provider'
    var_2 = 'text'
    var_3 = var_0.__getattr__(var_2)
    assert var_3 == 'text_provider'


def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'non_callable'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None



# Parsed testcases at query #8
#--------------------------

# Partially parsed test___getattr___returns_callable_for_existing_provider. Retrieved 7/8 statements.
# Partially parsed test___getattr___caches_provider_instance. Retrieved 5/10 statements.
# Partially parsed test___getattr___handles_underscore_prefix_correctly. Retrieved 6/7 statements.
# Partially parsed test___getattr___returns_none_for_non_callable_attribute. Retrieved 3/4 statements.



def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'non_existent'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None


def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'test_value'
    var_2 = lambda : var_1
    var_3 = 'test_provider'
    var_4 = var_0.__getattr__(var_3)
    var_5 = callable(var_4)
    var_6 = bool(var_5)
    assert var_6 is True
    var_7 = var_4()
    assert var_7 == 'test_value'


def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 0
    assert var_1 == 1
    var_2 = 'test_provider'
    var_3 = var_0.__getattr__(var_2)
    var_4 = var_0.__getattr__(var_2)
    var_5 = bool(var_3 is var_4)
    assert var_5 is True


def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'underscored'
    var_2 = lambda : var_1
    var_3 = 'underscore_provider'
    var_4 = var_0.__getattr__(var_3)
    var_5 = var_4()
    assert var_5 == 'underscored'


def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'non_callable'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_reseed_handles_attribute_error. Retrieved 2/4 statements.



def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.reseed()



# Parsed testcases at query #10
#--------------------------

# Partially parsed test___getattr___returns_callable_attribute. Retrieved 5/11 statements.
# Partially parsed test___getattr___caches_attribute_after_first_call. Retrieved 7/14 statements.
# Partially parsed test___getattr___handles_attribute_error_gracefully. Retrieved 3/4 statements.
# Partially parsed test___getattr___initializes_provider_with_locale_and_seed. Retrieved 5/16 statements.



def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'non_existent'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None


def test_case_0():
    var_0 = 'mock'
    var_1 = module_0.Generic()
    var_2 = var_1.__getattr__(var_0)
    var_3 = callable(var_2)
    var_4 = bool(var_3)
    assert var_4 is True
    var_5 = var_2()
    assert var_5 == 'mocked'


def test_case_0():
    var_0 = 0
    var_1 = 'mock'
    var_2 = module_0.Generic()
    var_3 = var_2.__getattr__(var_1)
    var_4 = var_2.__getattr__(var_1)
    var_5 = bool(var_3 is var_4)
    assert var_5 is True
    var_6 = var_3()
    assert var_6 == 1
    var_7 = var_4()
    assert var_7 == 1


def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'invalid'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = 'mock'
    var_3 = 'en'
    var_4 = 12345
    var_5 = bool(var_1 == var_4)
    assert var_5 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_add_provider_adds_custom_provider. Retrieved 4/12 statements.
# Partially parsed test_add_provider_raises_type_error_for_non_baseprovider_subclass. Retrieved 1/6 statements.
# Partially parsed test_add_provider_raises_type_error_for_generic. Retrieved 1/4 statements.
# Partially parsed test_add_provider_uses_meta_name. Retrieved 4/9 statements.
# Partially parsed test_add_provider_falls_back_to_lowercase_class_name. Retrieved 3/8 statements.
# Partially parsed test_add_provider_ignores_seed_kwarg. Retrieved 3/10 statements.
# Partially parsed test_add_provider_adds_instance_to_generic. Retrieved 3/9 statements.
# Partially parsed test_add_provider_preserves_seed_across_providers. Retrieved 4/14 statements.



def test_case_0():
    var_0 = 'custom'
    var_1 = module_0.Generic()
    var_2 = 'custom'
    var_3 = hasattr(var_1, var_2)
    var_4 = bool(var_3)
    assert var_4 is True


def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'not_a_class'
    var_2 = {}
    var_3 = var_0.add_provider(var_1, **var_2)


def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'subclass of mimesis.providers.BaseProvider'


def test_case_0():
    var_0 = module_0.Generic()


def test_case_0():
    var_0 = 'customname'
    var_1 = module_0.Generic()
    var_2 = 'customname'
    var_3 = hasattr(var_1, var_2)
    var_4 = bool(var_3)
    assert var_4 is True


def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'customprovider'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True


def test_case_0():
    var_0 = 12345
    var_1 = module_0.Generic(seed=var_0)
    var_2 = 'test'
    var_3 = var_1.customprovider.extra
    assert var_3 == 'test'
    var_4 = var_1.customprovider.seed
    assert var_4 == 12345


def test_case_0():
    var_0 = 'custom'
    var_1 = module_0.Generic()
    var_2 = var_1.custom


def test_case_0():
    var_0 = 'custom'
    var_1 = 999
    var_2 = module_0.Generic(seed=var_1)
    var_3 = module_0.Generic(seed=var_1)



# Parsed testcases at query #12
#--------------------------






# Parsed testcases at query #13
#--------------------------

# Partially parsed test_reseed_handles_attribute_error. Retrieved 3/5 statements.



def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 12345
    var_2 = var_0.reseed(var_1)



# Parsed testcases at query #14
#--------------------------

# Failed to parse test_getattr_with_non_callable_attribute.




# Parsed testcases at query #15
#--------------------------

# Partially parsed test_reseed_handles_attribute_error. Retrieved 5/7 statements.



def test_case_0():
    var_0 = module_0.Generic()
    var_1 = None
    var_2 = 'invalid_attr'
    var_3 = [var_2]
    var_4 = var_0.reseed()



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_generic_constructor_default_locale_and_seed. Retrieved 2/4 statements.
# Failed to parse test_generic_constructor_custom_locale.
# Partially parsed test_generic_constructor_provider_registry_initialization. Retrieved 7/9 statements.
# Partially parsed test_generic_constructor_lazy_initialization. Retrieved 2/3 statements.



def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.locale
    var_2 = var_0.seed
    var_3 = var_0.random


def test_case_0():
    var_0 = 12345
    var_1 = module_0.Generic(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 12345


def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'person'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = 'address'
    var_5 = hasattr(var_0, var_4)
    var_6 = bool(var_5)
    assert var_6 is True
    var_7 = var_0.person
    var_8 = var_0.address


def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'generic'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(not var_2)
    assert var_3 is True


def test_case_0():
    var_0 = module_0.Generic()
    var_1 = '_person'
    var_2 = bool('_person' in var_0.__dict__)
    assert var_2 is True
    var_3 = 'person'
    var_4 = bool('person' not in var_0.__dict__)
    assert var_4 is True
    var_5 = var_0.person
    var_6 = 'person'
    var_7 = bool('person' in var_0.__dict__)
    assert var_7 is True


def test_case_0():
    var_0 = 999
    var_1 = module_0.Generic(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 999
    var_3 = var_1.person.seed
    assert var_3 == 999
    var_4 = var_1.address.seed
    assert var_4 == 999


def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.seed
    var_2 = var_0.person.seed

def test_case_0():
    pass


def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.__dir__()
    var_2 = 'person'
    var_3 = bool('person' in var_1)
    assert var_3 is True
    var_4 = 'address'
    var_5 = bool('address' in var_1)
    assert var_5 is True
    var_6 = 'locale'
    var_7 = bool('locale' not in var_1)
    assert var_7 is True
    var_8 = 'seed'
    var_9 = bool('seed' not in var_1)
    assert var_9 is True



# Parsed testcases at query #17
#--------------------------






# Parsed testcases at query #18
#--------------------------

# Partially parsed test_skip_generic_in_registry. Retrieved 7/24 statements.


def test_case_0():
    var_0 = 'mock_data'
    var_1 = True
    var_2 = 'mock_base'
    var_3 = True
    var_4 = 'generic'
    var_5 = '_mock_data'
    var_6 = 'mock_base'



# Parsed testcases at query #19
#--------------------------

# Failed to parse test_provider_registry_does_not_contain_generic.




# Parsed testcases at query #20
#--------------------------

# Partially parsed test___getattr__with_non_callable_attribute. Retrieved 2/3 statements.



def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.test_provider
    assert var_1 is None



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_getattr_with_non_callable_attribute. Retrieved 2/3 statements.



def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.test_attr
    assert var_1 is None



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_condition_at_line_16_evaluates_to_false. Retrieved 8/26 statements.


def test_case_0():
    var_0 = 'mock_data'
    var_1 = True
    var_2 = 'mock_base'
    var_3 = True
    var_4 = 'mock_base'
    var_5 = '_mock_base'
    var_6 = 'mock_data'
    var_7 = '_mock_data'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test___getattr___returns_callable_attribute. Retrieved 7/8 statements.
# Partially parsed test___getattr___caches_attribute_after_first_access. Retrieved 6/11 statements.
# Partially parsed test___getattr___returns_none_for_non_callable_attribute. Retrieved 3/4 statements.
# Partially parsed test___getattr___handles_attribute_with_underscore_correctly. Retrieved 6/7 statements.



def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'non_existent'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None


def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'value'
    var_2 = lambda : var_1
    var_3 = 'test_provider'
    var_4 = var_0.__getattr__(var_3)
    var_5 = callable(var_4)
    var_6 = bool(var_5)
    assert var_6 is True
    var_7 = var_4()
    assert var_7 == 'value'


def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 0
    var_2 = 'test_provider'
    var_3 = var_0.__getattr__(var_2)
    var_4 = var_0.__getattr__(var_2)
    var_5 = bool(var_3 is var_4)
    assert var_5 is True
    var_6 = var_3()
    assert var_6 == 1


def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'test_provider'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None


def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'test'
    var_2 = lambda : var_1
    var_3 = 'test_provider'
    var_4 = var_0.__getattr__(var_3)
    var_5 = var_4()
    assert var_5 == 'test'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test___getattr___returns_provider_instance_for_existing_data_provider. Retrieved 4/5 statements.
# Partially parsed test___getattr___caches_provider_instance_in_dict. Retrieved 5/6 statements.
# Partially parsed test___getattr___handles_attribute_without_leading_underscore. Retrieved 4/5 statements.
# Partially parsed test___getattr___returns_none_for_non_callable_attribute. Retrieved 3/4 statements.



def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'non_existent'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None


def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'PersonProvider'
    var_2 = 'person'
    var_3 = var_0.__getattr__(var_2)
    assert var_3 == 'PersonProvider'


def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'AddressProvider'
    var_2 = 'address'
    var_3 = var_0.__getattr__(var_2)
    assert var_3 == 'AddressProvider'
    var_4 = var_0.__getattr__(var_2)
    assert var_4 == 'AddressProvider'
    var_5 = 'address'
    var_6 = bool('address' in var_0.__dict__)
    assert var_6 is True


def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'TextProvider'
    var_2 = 'text'
    var_3 = var_0.__getattr__(var_2)
    assert var_3 == 'TextProvider'


def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'non_callable'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_add_provider_with_meta_name_does_not_raise_attribute_error. Retrieved 4/9 statements.



def test_case_0():
    var_0 = 'custom'
    var_1 = module_0.Generic()
    var_2 = 'custom'
    var_3 = hasattr(var_1, var_2)
    var_4 = bool(var_3)
    assert var_4 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_generic_initialization_with_default_locale_and_seed. Retrieved 2/4 statements.
# Failed to parse test_generic_initialization_with_custom_locale.
# Partially parsed test_generic_initialization_lazy_loading. Retrieved 7/8 statements.
# Partially parsed test_generic_initialization_base_provider_subclasses. Retrieved 4/6 statements.



def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.locale
    var_2 = var_0.seed
    var_3 = var_0.random


def test_case_0():
    var_0 = 12345
    var_1 = module_0.Generic(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 12345


def test_case_0():
    var_0 = module_0.Generic()
    var_1 = dir(var_0)
    var_2 = 'person'
    var_3 = bool('person' in var_1)
    assert var_3 is True
    var_4 = dir(var_0)
    var_5 = 'address'
    var_6 = bool('address' in var_4)
    assert var_6 is True


def test_case_0():
    var_0 = module_0.Generic()
    var_1 = '_person'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = 'person'
    var_5 = hasattr(var_0, var_4)
    var_6 = bool(not var_5)
    assert var_6 is True
    var_7 = var_0.person
    var_8 = hasattr(var_0, var_4)
    var_9 = bool(var_8)
    assert var_9 is True


def test_case_0():
    var_0 = 'invalid_locale'
    var_1 = module_0.Generic(var_0)
    var_2 = var_1.locale
    assert var_2 == 'invalid_locale'


def test_case_0():
    var_0 = None
    var_1 = module_0.Generic(seed=var_0)
    var_2 = var_1.seed
    assert var_2 is None


def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'generic'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(not var_2)
    assert var_3 is True


def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'random'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = var_0.random


def test_case_0():
    var_0 = module_0.Generic()
    var_1 = dir(var_0)
    var_2 = 'locale'
    var_3 = bool('locale' not in var_1)
    assert var_3 is True
    var_4 = 'seed'
    var_5 = bool('seed' not in var_1)
    assert var_5 is True
    var_6 = 'random'
    var_7 = bool('random' not in var_1)
    assert var_7 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_reseed_handles_attribute_error. Retrieved 3/4 statements.



def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 12345
    var_2 = var_0.reseed(var_1)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test___getattr___returns_callable_attribute. Retrieved 7/8 statements.
# Partially parsed test___getattr___caches_attribute_after_first_access. Retrieved 5/10 statements.
# Partially parsed test___getattr___returns_none_for_non_callable_attribute. Retrieved 3/4 statements.
# Partially parsed test___getattr___handles_attribute_with_underscore. Retrieved 6/7 statements.



def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'non_existent'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None


def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'test_value'
    var_2 = lambda : var_1
    var_3 = 'test_provider'
    var_4 = var_0.__getattr__(var_3)
    var_5 = callable(var_4)
    var_6 = bool(var_5)
    assert var_6 is True
    var_7 = var_4()
    assert var_7 == 'test_value'


def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 0
    assert var_1 == 1
    var_2 = 'test_provider'
    var_3 = var_0.__getattr__(var_2)
    var_4 = var_0.__getattr__(var_2)
    var_5 = bool(var_3 is var_4)
    assert var_5 is True


def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'test_provider'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None


def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'underscore'
    var_2 = lambda : var_1
    var_3 = 'test_provider'
    var_4 = var_0.__getattr__(var_3)
    var_5 = var_4()
    assert var_5 == 'underscore'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_reseed_handles_attribute_error. Retrieved 3/5 statements.



def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 12345
    var_2 = var_0.reseed(var_1)



# Parsed testcases at query #10
#--------------------------






# Parsed testcases at query #11
#--------------------------






# Parsed testcases at query #12
#--------------------------

# Partially parsed test_provider_cls_is_not_generic_and_not_subclass_of_baseprovider. Retrieved 2/14 statements.



def test_case_0():
    var_0 = 'mock'
    var_1 = module_0.Generic()
    var_2 = bool(False)
    assert var_2 is True
    var_3 = bool(True)
    assert var_3 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_getattr_with_non_callable_attribute. Retrieved 2/3 statements.



def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.test_attr
    assert var_1 is None



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_getattr_with_non_callable_attribute. Retrieved 2/3 statements.



def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.test_attr
    assert var_1 is None



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_reseed_updates_seed_on_generic. Retrieved 1/8 statements.
# Partially parsed test_reseed_updates_seed_on_attached_providers. Retrieved 1/10 statements.
# Failed to parse test_reseed_with_missing_seed_generates_new_seed.
# Partially parsed test_reseed_handles_attribute_error_gracefully. Retrieved 3/9 statements.
# Partially parsed test_reseed_propagates_to_all_providers_in_dir. Retrieved 1/12 statements.


def test_case_0():
    var_0 = 999

def test_case_0():
    var_0 = 12345

def test_case_0():
    var_0 = 'fake_attr'
    var_1 = None
    var_2 = 555

def test_case_0():
    var_0 = 777



# Parsed testcases at query #16
#--------------------------






# Parsed testcases at query #17
#--------------------------

# Partially parsed test_generic_constructor_default_locale_and_seed. Retrieved 2/4 statements.
# Failed to parse test_generic_constructor_custom_locale.
# Partially parsed test_generic_constructor_base_provider_attributes_excluded. Retrieved 4/9 statements.
# Partially parsed test_generic_constructor_data_providers_lazily_loaded. Retrieved 4/5 statements.
# Partially parsed test_generic_constructor_non_data_providers_instantiated. Retrieved 2/3 statements.
# Failed to parse test_generic_constructor_locale_passed_to_data_providers.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.locale
    var_2 = var_0.seed
    var_3 = var_0.random

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 12345
    var_1 = module_0.Generic(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 12345

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = dir(var_0)
    var_2 = 'address'
    var_3 = bool('address' in var_1)
    assert var_3 is True
    var_4 = dir(var_0)
    var_5 = 'person'
    var_6 = bool('person' in var_4)
    assert var_6 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = dir(var_0)
    var_2 = 'generic'
    var_3 = bool('generic' not in var_1)
    assert var_3 is True

import mimesis.providers.generic as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = module_1.BaseProvider()
    var_2 = 'locale'
    var_3 = dir(var_0)

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0._address
    var_2 = callable(var_1)
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = var_0.address

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.cryptographic

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 999
    var_1 = module_0.Generic(seed=var_0)
    var_2 = var_1.cryptographic.seed
    assert var_2 == 999



# Parsed testcases at query #18
#--------------------------

# Failed to parse test_skip_generic_in_registry.




# Parsed testcases at query #19
#--------------------------

# Partially parsed test_reseed_handles_attribute_error. Retrieved 3/4 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 12345
    var_2 = var_0.reseed(var_1)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_skip_generic_in_registry. Retrieved 2/19 statements.


def test_case_0():
    var_0 = 'generic'
    var_1 = 'generic'
    var_2 = None



# Parsed testcases at query #21
#--------------------------




import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'non_existent'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'person'
    var_2 = var_0.__getattr__(var_1)
    var_3 = bool(var_2 is not None)
    assert var_3 is True
    var_4 = callable(var_2)
    assert var_4 is False

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'address'
    var_2 = var_0.__getattr__(var_1)
    var_3 = var_0.__getattr__(var_1)
    var_4 = bool(var_2 is var_3)
    assert var_4 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = '_test_provider'
    var_2 = 'test_value'
    var_3 = lambda locale, seed: var_2
    var_4 = setattr(var_0, var_1, var_3)
    var_5 = 'test_provider'
    var_6 = var_0.__getattr__(var_5)
    assert var_6 == 'test_value'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = '_non_callable'
    var_2 = 'some_string'
    var_3 = setattr(var_0, var_1, var_2)
    var_4 = 'non_callable'
    var_5 = var_0.__getattr__(var_4)
    assert var_5 is None



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_add_provider_with_meta_name_attribute. Retrieved 4/9 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom'
    var_1 = module_0.Generic()
    var_2 = 'custom'
    var_3 = hasattr(var_1, var_2)
    var_4 = bool(var_3)
    assert var_4 is True



# Parsed testcases at query #23
#--------------------------






# Parsed testcases at query #24
#--------------------------

# Partially parsed test_getattr_with_non_callable_attribute. Retrieved 2/3 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.test_attr
    assert var_1 is None



# Parsed testcases at query #25
#--------------------------

# Partially parsed test___getattr___returns_initialized_provider_for_valid_attribute. Retrieved 4/5 statements.
# Partially parsed test___getattr___caches_initialized_provider_in_dict. Retrieved 4/5 statements.
# Partially parsed test___getattr___returns_none_for_non_callable_attribute. Retrieved 3/4 statements.
# Partially parsed test___getattr___handles_attribute_with_underscore. Retrieved 4/5 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'non_existing'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'initialized_provider'
    var_2 = 'test_provider'
    var_3 = var_0.__getattr__(var_2)
    assert var_3 == 'initialized_provider'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'cached_provider'
    var_2 = 'test_provider'
    var_3 = var_0.__getattr__(var_2)
    var_4 = var_0.__dict__['test_provider']
    assert var_4 == 'cached_provider'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'test_provider'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'underscore_provider'
    var_2 = 'test_provider'
    var_3 = var_0.__getattr__(var_2)
    assert var_3 == 'underscore_provider'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_generic_initialization_with_default_locale_and_seed. Retrieved 2/4 statements.
# Failed to parse test_generic_initialization_with_custom_locale.
# Partially parsed test_generic_initialization_lazy_loading. Retrieved 6/7 statements.
# Partially parsed test_generic_initialization_base_provider_attributes_excluded. Retrieved 4/8 statements.
# Failed to parse test_generic_initialization_with_missing_seed.
# Partially parsed test_generic_initialization_ensures_random_instance. Retrieved 2/4 statements.
# Partially parsed test_generic_initialization_locale_independent_providers_present. Retrieved 3/4 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.locale
    var_2 = var_0.seed
    var_3 = var_0.random

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 12345
    var_1 = module_0.Generic(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 12345

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = dir(var_0)
    var_2 = 'person'
    var_3 = bool('person' in var_1)
    assert var_3 is True
    var_4 = dir(var_0)
    var_5 = 'address'
    var_6 = bool('address' in var_4)
    assert var_6 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = '_person'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = var_0._person
    var_5 = callable(var_4)
    var_6 = bool(var_5)
    assert var_6 is True
    var_7 = var_0.person

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = dir(var_0)
    var_2 = 'generic'
    var_3 = bool('generic' not in var_1)
    assert var_3 is True

import mimesis.providers.generic as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = module_1.BaseProvider()
    var_2 = 'locale'
    var_3 = var_0.__dir__()

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.Generic(seed=var_0)
    var_2 = var_1.seed
    assert var_2 is None

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.random

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = dir(var_0)
    var_2 = 'cryptographic'
    var_3 = bool('cryptographic' in var_1)
    assert var_3 is True
    var_4 = var_0.cryptographic



# Parsed testcases at query #27
#--------------------------

# Failed to parse test_skip_generic_in_registry.




# Parsed testcases at query #28
#--------------------------

# Partially parsed test_reseed_handles_attribute_error. Retrieved 2/4 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.reseed()



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_getattr_with_non_callable_attribute. Retrieved 2/3 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.test_attr
    assert var_1 is None



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_add_provider_adds_custom_provider. Retrieved 4/12 statements.
# Partially parsed test_add_provider_raises_type_error_for_non_baseprovider_subclass. Retrieved 1/6 statements.
# Partially parsed test_add_provider_raises_type_error_for_generic. Retrieved 1/4 statements.
# Partially parsed test_add_provider_uses_meta_name. Retrieved 6/11 statements.
# Partially parsed test_add_provider_falls_back_to_lowercase_class_name. Retrieved 3/8 statements.
# Partially parsed test_add_provider_passes_kwargs. Retrieved 2/9 statements.
# Partially parsed test_add_provider_ignores_seed_kwarg. Retrieved 3/10 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom'
    var_1 = module_0.Generic()
    var_2 = 'custom'
    var_3 = hasattr(var_1, var_2)
    var_4 = bool(var_3)
    assert var_4 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'not_a_class'
    var_2 = {}
    var_3 = var_0.add_provider(var_1, **var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = bool(False)
    assert var_1 is True
    var_2 = bool(True)
    assert var_2 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = bool(False)
    assert var_1 is True
    var_2 = bool(True)
    assert var_2 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'special'
    var_1 = module_0.Generic()
    var_2 = 'special'
    var_3 = hasattr(var_1, var_2)
    var_4 = bool(var_3)
    assert var_4 is True
    var_5 = 'customprovider'
    var_6 = hasattr(var_1, var_5)
    var_7 = bool(not var_6)
    assert var_7 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'anotherprovider'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'test'
    var_2 = var_0.kwargprovider.extra
    assert var_2 == 'test'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Generic(seed=var_0)
    var_2 = 999
    var_3 = var_1.seedcheckprovider.seed_passed
    assert var_3 is None
    var_4 = var_1.seedcheckprovider.seed
    assert var_4 == 42



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_reseed_handles_attribute_error. Retrieved 11/15 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'MockProvider'
    var_2 = ()
    var_3 = 'reseed'
    var_4 = None
    var_5 = lambda self, seed: var_4
    var_6 = {var_3: var_5}
    var_7 = [var_1, var_2, var_6]
    var_8 = 'attr_without_provider'
    var_9 = [var_8]
    var_10 = 12345
    var_11 = var_0.reseed(var_10)



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_getattr_with_non_callable_attribute. Retrieved 2/3 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.test_attr
    assert var_1 is None



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_generic_constructor_default_locale_and_seed. Retrieved 2/4 statements.
# Failed to parse test_generic_constructor_custom_locale.
# Partially parsed test_generic_constructor_data_providers_lazy_loaded. Retrieved 9/10 statements.
# Partially parsed test_generic_constructor_base_providers_instantiated. Retrieved 4/5 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.locale
    var_2 = var_0.seed
    var_3 = var_0.random

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 12345
    var_1 = module_0.Generic(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 12345

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = dir(var_0)
    var_2 = 'address'
    var_3 = bool('address' in var_1)
    assert var_3 is True
    var_4 = dir(var_0)
    var_5 = 'person'
    var_6 = bool('person' in var_4)
    assert var_6 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = '_address'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = var_0._address
    var_5 = callable(var_4)
    var_6 = bool(var_5)
    assert var_6 is True
    var_7 = 'address'
    var_8 = hasattr(var_0, var_7)
    var_9 = bool(not var_8)
    assert var_9 is True
    var_10 = var_0.address
    var_11 = hasattr(var_0, var_7)
    var_12 = bool(var_11)
    assert var_12 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'cryptographic'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = var_0.cryptographic

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = dir(var_0)
    var_2 = 'generic'
    var_3 = bool('generic' not in var_1)
    assert var_3 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 999
    var_1 = module_0.Generic(seed=var_0)
    var_2 = var_1.cryptographic.seed
    assert var_2 == 999

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = dir(var_0)
    var_2 = 'locale'
    var_3 = bool('locale' not in var_1)
    assert var_3 is True
    var_4 = 'seed'
    var_5 = bool('seed' not in var_1)
    assert var_5 is True
    var_6 = 'random'
    var_7 = bool('random' not in var_1)
    assert var_7 is True
    var_8 = 'reseed'
    var_9 = bool('reseed' not in var_1)
    assert var_9 is True
    var_10 = 'validate_enum'
    var_11 = bool('validate_enum' not in var_1)
    assert var_11 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = dir(var_0)
    var_2 = 'address'
    var_3 = bool('address' in var_1)
    assert var_3 is True
    var_4 = 'person'
    var_5 = bool('person' in var_1)
    assert var_5 is True
    var_6 = 'cryptographic'
    var_7 = bool('cryptographic' in var_1)
    assert var_7 is True



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_add_provider_with_meta_name_attribute. Retrieved 5/13 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom'
    var_1 = module_0.Generic()
    var_2 = 'custom'
    var_3 = hasattr(var_1, var_2)
    var_4 = bool(var_3)
    assert var_4 is True
    var_5 = var_1.custom



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_generic_constructor_default_locale_and_seed. Retrieved 2/4 statements.
# Failed to parse test_generic_constructor_custom_locale.
# Partially parsed test_generic_constructor_lazy_initialization. Retrieved 2/3 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.locale
    var_2 = var_0.seed
    var_3 = var_0.random

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 12345
    var_1 = module_0.Generic(seed=var_0)
    var_2 = var_1.seed
    var_3 = bool(var_1.seed == var_0)
    assert var_3 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'person'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = 'address'
    var_5 = hasattr(var_0, var_4)
    var_6 = bool(var_5)
    assert var_6 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'generic'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(not var_2)
    assert var_3 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = '_person'
    var_2 = bool('_person' in var_0.__dict__)
    assert var_2 is True
    var_3 = 'person'
    var_4 = bool('person' not in var_0.__dict__)
    assert var_4 is True
    var_5 = var_0.person
    var_6 = 'person'
    var_7 = bool('person' in var_0.__dict__)
    assert var_7 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 999
    var_1 = module_0.Generic(seed=var_0)
    var_2 = var_1.person
    var_3 = var_2.seed
    var_4 = bool(var_2.seed == var_0)
    assert var_4 is True



# Parsed testcases at query #36
#--------------------------

# Partially parsed test___getattr___returns_callable_attribute. Retrieved 5/11 statements.
# Partially parsed test___getattr___caches_attribute_after_first_access. Retrieved 7/14 statements.
# Partially parsed test___getattr___handles_attribute_without_leading_underscore. Retrieved 3/4 statements.
# Partially parsed test___getattr___returns_none_for_non_callable_attribute. Retrieved 3/4 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'non_existent'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'mock'
    var_1 = module_0.Generic()
    var_2 = var_1.__getattr__(var_0)
    var_3 = callable(var_2)
    var_4 = bool(var_3)
    assert var_4 is True
    var_5 = var_2()
    assert var_5 == 'mocked'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 0
    var_1 = 'mock'
    var_2 = module_0.Generic()
    var_3 = var_2.__getattr__(var_1)
    var_4 = var_2.__getattr__(var_1)
    var_5 = bool(var_3 is var_4)
    assert var_5 is True
    var_6 = var_3()
    assert var_6 == 1
    var_7 = var_4()
    assert var_7 == 1

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'test_attr'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 == 'value'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'non_callable'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_issubclass_of_baseprovider_but_not_basedataprovider. Retrieved 2/16 statements.


def test_case_0():
    var_0 = 'custom'
    var_1 = 'custom'



# Parsed testcases at query #38
#--------------------------






# Parsed testcases at query #39
#--------------------------

# Failed to parse test_provider_registry_does_not_contain_generic.




# Parsed testcases at query #40
#--------------------------

# Failed to parse test_generic_constructor_custom_locale.
# Partially parsed test_generic_constructor_locale_and_seed. Retrieved 1/3 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.locale
    var_2 = var_0.seed

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 12345
    var_1 = module_0.Generic(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 12345

def test_case_0():
    var_0 = 98765

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'person'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = 'address'
    var_5 = hasattr(var_0, var_4)
    var_6 = bool(var_5)
    assert var_6 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'random'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = 'seed'
    var_5 = hasattr(var_0, var_4)
    var_6 = bool(var_5)
    assert var_6 is True

def test_case_0():
    pass



# Parsed testcases at query #41
#--------------------------






# Parsed testcases at query #42
#--------------------------

# Partially parsed test___getattr__predicate_false. Retrieved 3/4 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'test_provider'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_generic_constructor_default_locale_and_seed. Retrieved 2/4 statements.
# Failed to parse test_generic_constructor_custom_locale.
# Partially parsed test_generic_constructor_custom_locale_and_seed. Retrieved 1/3 statements.
# Partially parsed test_generic_constructor_base_providers_instantiated. Retrieved 3/5 statements.
# Partially parsed test_generic_constructor_base_provider_attributes_excluded. Retrieved 4/8 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.locale
    var_2 = var_0.seed
    var_3 = var_0.random

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 12345
    var_1 = module_0.Generic(seed=var_0)
    var_2 = var_1.locale
    var_3 = var_1.seed
    assert var_3 == 12345

def test_case_0():
    var_0 = 98765

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'person'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = 'address'
    var_5 = hasattr(var_0, var_4)
    var_6 = bool(var_5)
    assert var_6 is True
    var_7 = 'datetime'
    var_8 = hasattr(var_0, var_7)
    var_9 = bool(var_8)
    assert var_9 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'generic'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(not var_2)
    assert var_3 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = '_person'
    var_2 = None
    var_3 = getattr(var_0, var_1, var_2)
    var_4 = callable(var_3)
    var_5 = bool(var_4)
    assert var_5 is True
    var_6 = '_address'
    var_7 = getattr(var_0, var_6, var_2)
    var_8 = callable(var_7)
    var_9 = bool(var_8)
    assert var_9 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.cryptographic
    var_2 = var_0.development

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Generic(seed=var_0)
    var_2 = var_1.cryptographic.seed
    assert var_2 == 42
    var_3 = var_1.development.seed
    assert var_3 == 42

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.__dir__()
    var_2 = 'person'
    var_3 = bool('person' in var_1)
    assert var_3 is True
    var_4 = 'address'
    var_5 = bool('address' in var_1)
    assert var_5 is True
    var_6 = 'cryptographic'
    var_7 = bool('cryptographic' in var_1)
    assert var_7 is True
    var_8 = 'locale'
    var_9 = bool('locale' not in var_1)
    assert var_9 is True

import mimesis.providers.generic as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = module_1.BaseProvider()
    var_2 = 'locale'
    var_3 = var_0.__dir__()



