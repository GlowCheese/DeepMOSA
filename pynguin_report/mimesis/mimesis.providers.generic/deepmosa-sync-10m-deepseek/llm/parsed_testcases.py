####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test___getattr___returns_initialized_provider_for_valid_attribute. Retrieved 4/5 statements.
# Partially parsed test___getattr___caches_initialized_provider_in_dict. Retrieved 4/5 statements.
# Partially parsed test___getattr___returns_none_for_non_callable_attribute. Retrieved 3/4 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'non_existent'
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
    var_1 = '_invalid'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None



# Parsed testcases at query #2
#--------------------------

# Partially parsed test___getattr__returns_none_when_attribute_is_not_callable. Retrieved 2/3 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.test_attr
    assert var_1 is None



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_add_provider_success. Retrieved 3/11 statements.
# Partially parsed test_add_provider_with_kwargs. Retrieved 4/15 statements.
# Partially parsed test_add_provider_removes_seed_kwarg. Retrieved 6/16 statements.
# Partially parsed test_add_provider_raises_type_error_for_non_baseprovider_subclass. Retrieved 3/7 statements.
# Partially parsed test_add_provider_raises_type_error_for_generic_instance. Retrieved 3/5 statements.
# Partially parsed test_add_provider_uses_class_name_lowercase_when_meta_name_missing. Retrieved 2/10 statements.
# Partially parsed test_add_provider_ensures_same_seed_across_providers. Retrieved 7/25 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom'
    var_1 = False
    var_2 = module_0.Generic()

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom'
    var_1 = False
    var_2 = module_0.Generic()
    var_3 = 'extra_value'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom'
    var_1 = False
    var_2 = 12345
    var_3 = module_0.Generic(seed=var_2)
    var_4 = 99999
    var_5 = 'preserved'
    var_6 = var_3.custom.custom_seed
    assert var_6 == 'preserved'
    var_7 = var_3.custom.seed
    assert var_7 == 12345

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = False
    var_2 = 'not_a_class'
    var_3 = {}
    var_4 = var_0.add_provider(var_2, **var_3)
    var_5 = True
    var_6 = bool(var_5)
    assert var_6 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = False
    var_2 = True
    var_3 = bool(var_2)
    assert var_3 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = False
    var_2 = True
    var_3 = bool(var_2)
    assert var_3 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Generic()

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'providera'
    var_1 = False
    var_2 = 'providerb'
    var_3 = False
    var_4 = 42
    var_5 = module_0.Generic(seed=var_4)
    var_6 = module_0.Generic(seed=var_4)



# Parsed testcases at query #4
#--------------------------

# Failed to parse test_generic_constructor_custom_locale.
# Partially parsed test_generic_constructor_locale_and_seed. Retrieved 1/3 statements.
# Partially parsed test_generic_constructor_providers_shared_seed. Retrieved 3/5 statements.


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
    var_7 = 'datetime'
    var_8 = hasattr(var_0, var_7)
    var_9 = bool(var_8)
    assert var_9 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.person.full_name
    var_2 = callable(var_1)
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = var_0.address.address
    var_5 = callable(var_4)
    var_6 = bool(var_5)
    assert var_6 is True
    var_7 = var_0.datetime.datetime
    var_8 = callable(var_7)
    var_9 = bool(var_8)
    assert var_9 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Generic(seed=var_0)
    var_2 = module_0.Generic(seed=var_0)

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
    var_1 = 'random'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = 'seed'
    var_5 = hasattr(var_0, var_4)
    var_6 = bool(var_5)
    assert var_6 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = dir(var_0)
    var_2 = 'person'
    var_3 = bool('person' in var_1)
    assert var_3 is True
    var_4 = 'address'
    var_5 = bool('address' in var_1)
    assert var_5 is True
    var_6 = 'datetime'
    var_7 = bool('datetime' in var_1)
    assert var_7 is True
    var_8 = 'locale'
    var_9 = bool('locale' not in var_1)
    assert var_9 is True



# Parsed testcases at query #5
#--------------------------






# Parsed testcases at query #6
#--------------------------

# Partially parsed test_provider_cls_is_not_subclass_of_baseprovider. Retrieved 3/9 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'dummy'
    var_1 = module_0.Generic()
    var_2 = hasattr(var_1, var_0)
    var_3 = bool(not var_2)
    assert var_3 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_reseed_updates_seed_on_attached_providers. Retrieved 5/6 statements.
# Partially parsed test_reseed_handles_providers_without_reseed_method. Retrieved 3/7 statements.
# Partially parsed test_reseed_propagates_to_all_providers_in_dir. Retrieved 7/9 statements.


import mimesis.providers.generic as module_0

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

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.baseprovider
    var_2 = var_1.seed
    var_3 = 67890
    var_4 = var_0.reseed(var_3)
    var_5 = var_1.seed
    var_6 = bool(var_1.seed != var_2)
    assert var_6 is True
    var_7 = var_1.seed
    assert var_7 == 67890

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.seed
    var_2 = var_0.reseed()
    var_3 = var_0.seed
    var_4 = bool(var_0.seed != var_1)
    assert var_4 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 11111
    var_2 = var_0.reseed(var_1)
    var_3 = var_0.seed
    assert var_3 == 11111

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.baseprovider
    var_2 = var_0.basedataprovider
    var_3 = var_1.seed
    var_4 = var_2.seed
    var_5 = 22222
    var_6 = var_0.reseed(var_5)
    var_7 = var_1.seed
    var_8 = bool(var_1.seed != var_3)
    assert var_8 is True
    var_9 = var_2.seed
    var_10 = bool(var_2.seed != var_4)
    assert var_10 is True
    var_11 = var_1.seed
    assert var_11 == 22222
    var_12 = var_2.seed
    assert var_12 == 22222

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = module_0.Generic(var_0)
    var_2 = var_1.locale
    var_3 = 33333
    var_4 = var_1.reseed(var_3)
    var_5 = var_1.locale
    var_6 = bool(var_1.locale == var_2)
    assert var_6 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 44444
    var_1 = module_0.Generic(seed=var_0)
    var_2 = var_1.baseprovider
    var_3 = var_1.reseed(var_0)
    var_4 = var_1.seed
    assert var_4 == 44444
    var_5 = var_2.seed
    assert var_5 == 44444



# Parsed testcases at query #8
#--------------------------

# Partially parsed test___getattr__with_non_callable_attribute. Retrieved 2/3 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.test_attribute
    assert var_1 is None



# Parsed testcases at query #9
#--------------------------

# Failed to parse test___getattr__predicate_false.




# Parsed testcases at query #10
#--------------------------

# Partially parsed test_skip_generic_in_registry. Retrieved 3/14 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = bool(var_1)
    assert var_2 is True
    var_3 = module_0.Generic()



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_generic_initialization_with_default_locale_and_seed. Retrieved 2/4 statements.
# Failed to parse test_generic_initialization_with_custom_locale.
# Partially parsed test_generic_initialization_sets_base_provider_instances. Retrieved 1/8 statements.
# Partially parsed test_generic_initialization_sets_base_data_provider_attributes. Retrieved 1/9 statements.
# Partially parsed test_generic_initialization_lazy_loading_of_data_providers. Retrieved 1/7 statements.
# Failed to parse test_generic_initialization_with_custom_locale_propagates_to_data_providers.
# Partially parsed test_generic_initialization_with_custom_seed_propagates_to_base_providers. Retrieved 2/7 statements.
# Partially parsed test_generic_initialization_dir_excludes_base_provider_attributes. Retrieved 4/8 statements.


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
    var_0 = None
    var_1 = module_0.Generic(seed=var_0)
    var_2 = var_1.seed
    assert var_2 is None

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

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 999
    var_1 = module_0.Generic(seed=var_0)

import mimesis.providers.generic as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.__dir__()
    var_2 = module_1.BaseProvider()
    var_3 = 'locale'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.__dir__()

def test_case_0():
    pass



# Parsed testcases at query #12
#--------------------------






# Parsed testcases at query #13
#--------------------------

# Partially parsed test___getattr___returns_provider_instance_for_valid_attribute. Retrieved 4/5 statements.
# Partially parsed test___getattr___caches_provider_instance_in_dict. Retrieved 4/5 statements.
# Partially parsed test___getattr___returns_cached_instance_on_subsequent_calls. Retrieved 5/6 statements.
# Partially parsed test___getattr___handles_non_callable_attribute. Retrieved 3/4 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'non_existent'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'person_provider'
    var_2 = 'person'
    var_3 = var_0.__getattr__(var_2)
    assert var_3 == 'person_provider'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'person_provider'
    var_2 = 'person'
    var_3 = var_0.__getattr__(var_2)
    var_4 = var_0.__dict__['person']
    assert var_4 == 'person_provider'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'person_provider'
    var_2 = 'person'
    var_3 = var_0.__getattr__(var_2)
    var_4 = var_0.__getattr__(var_2)
    var_5 = bool(var_3 is var_4)
    assert var_5 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'non_callable'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_reseed_handles_attribute_error. Retrieved 11/16 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'TestProvider'
    var_2 = 'Meta'
    var_3 = ()
    var_4 = 'name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = [var_2, var_3, var_6]
    var_8 = None
    var_9 = setattr(var_0, var_5, var_8)
    var_10 = 12345
    var_11 = var_0.reseed(var_10)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_add_provider_adds_custom_provider. Retrieved 3/11 statements.
# Partially parsed test_add_provider_raises_type_error_for_non_baseprovider_subclass. Retrieved 3/7 statements.
# Partially parsed test_add_provider_raises_type_error_for_generic_instance. Retrieved 3/5 statements.
# Partially parsed test_add_provider_uses_meta_name. Retrieved 3/11 statements.
# Partially parsed test_add_provider_uses_lowercase_class_name_if_no_meta. Retrieved 1/8 statements.
# Partially parsed test_add_provider_ignores_seed_kwarg. Retrieved 4/15 statements.
# Partially parsed test_add_provider_preserves_generic_seed. Retrieved 4/12 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom'
    var_1 = False
    var_2 = module_0.Generic()

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = False
    var_2 = 'not_a_class'
    var_3 = {}
    var_4 = var_0.add_provider(var_2, **var_3)
    var_5 = True
    var_6 = bool(var_5)
    assert var_6 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = False
    var_2 = True
    var_3 = bool(var_2)
    assert var_3 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = False
    var_2 = True
    var_3 = bool(var_2)
    assert var_3 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'special'
    var_1 = False
    var_2 = module_0.Generic()

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom'
    var_1 = False
    var_2 = module_0.Generic()
    var_3 = 'extra_value'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'seedprovider'
    var_1 = False
    var_2 = 42
    var_3 = module_0.Generic(seed=var_2)



# Parsed testcases at query #16
#--------------------------






# Parsed testcases at query #17
#--------------------------






# Parsed testcases at query #18
#--------------------------

# Partially parsed test_add_provider_without_meta_name. Retrieved 3/8 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'customprovider'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_getattr_with_non_callable_attribute. Retrieved 2/3 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.test_attr
    assert var_1 is None



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_add_provider_with_non_generic_instance. Retrieved 4/9 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom'
    var_1 = module_0.Generic()
    var_2 = 'custom'
    var_3 = hasattr(var_1, var_2)
    var_4 = bool(var_3)
    assert var_4 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test___getattr___returns_none_for_non_callable_attribute. Retrieved 3/4 statements.
# Partially parsed test___getattr___initializes_and_caches_callable_provider. Retrieved 5/8 statements.
# Partially parsed test___getattr___passes_locale_and_seed_to_provider. Retrieved 7/10 statements.
# Partially parsed test___getattr___returns_cached_provider_on_subsequent_calls. Retrieved 6/9 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'non_existent'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'non_callable'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'MockProvider'
    var_2 = {}
    var_3 = 'mock'
    var_4 = var_0.__getattr__(var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True
    var_6 = 'mock'
    var_7 = bool('mock' in var_0.__dict__)
    assert var_7 is True
    var_8 = var_0.__dict__['mock']
    var_9 = bool(var_0.__dict__['mock'] is var_4)
    assert var_9 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = 12345
    var_2 = module_0.Generic(var_0, var_1)
    var_3 = 'MockProvider'
    var_4 = {}
    var_5 = 'mock'
    var_6 = var_2.__getattr__(var_5)
    var_7 = var_6.locale
    assert var_7 == 'en'
    var_8 = var_6.seed
    assert var_8 == 12345

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'MockProvider'
    var_2 = {}
    var_3 = 'mock'
    var_4 = var_0.__getattr__(var_3)
    var_5 = var_0.__getattr__(var_3)
    var_6 = bool(var_4 is var_5)
    assert var_6 is True



# Parsed testcases at query #22
#--------------------------






# Parsed testcases at query #23
#--------------------------

# Partially parsed test___getattr___returns_none_for_attribute_without_leading_underscore. Retrieved 3/4 statements.
# Partially parsed test___getattr___returns_initialized_provider_for_valid_attribute. Retrieved 3/9 statements.
# Partially parsed test___getattr___caches_initialized_provider_in_dict. Retrieved 4/9 statements.
# Partially parsed test___getattr___returns_cached_provider_on_subsequent_calls. Retrieved 4/9 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'non_existent'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'test_provider'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'mock'
    var_2 = var_0.__getattr__(var_1)
    var_3 = var_2.locale
    var_4 = bool(var_2.locale == var_0.locale)
    assert var_4 is True
    var_5 = var_2.seed
    var_6 = bool(var_2.seed == var_0.seed)
    assert var_6 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'mock'
    var_2 = var_0.__getattr__(var_1)
    var_3 = var_0.__dict__[var_1]
    var_4 = bool(var_2 is var_3)
    assert var_4 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'mock'
    var_2 = var_0.__getattr__(var_1)
    var_3 = var_0.__getattr__(var_1)
    var_4 = bool(var_2 is var_3)
    assert var_4 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = '_invalid'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_add_provider_adds_custom_provider. Retrieved 4/2 statements.
# Partially parsed test_add_provider_uses_provider_meta_name. Retrieved 7/12 statements.
# Partially parsed test_add_provider_falls_back_to_class_name_lower. Retrieved 3/8 statements.
# Partially parsed test_add_provider_passes_kwargs_to_provider. Retrieved 2/9 statements.
# Partially parsed test_add_provider_removes_seed_from_kwargs. Retrieved 4/11 statements.
# Partially parsed test_add_provider_raises_type_error_for_non_baseprovider_subclass. Retrieved 1/6 statements.
# Partially parsed test_add_provider_raises_type_error_for_generic_instance. Retrieved 1/4 statements.
# Partially parsed test_add_provider_ensures_seed_consistency. Retrieved 6/13 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.Generic()
    var_2 = 'custom'
    var_3 = hasattr(var_1, var_2)
    var_4 = bool(var_3)
    assert var_4 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.Generic()
    var_2 = 'custom'
    var_3 = hasattr(var_1, var_2)
    var_4 = bool(var_3)
    assert var_4 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'special'
    var_1 = False
    var_2 = module_0.Generic()
    var_3 = 'special'
    var_4 = hasattr(var_2, var_3)
    var_5 = bool(var_4)
    assert var_5 is True
    var_6 = 'customprovider'
    var_7 = hasattr(var_2, var_6)
    var_8 = bool(not var_7)
    assert var_8 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'customprovider'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'value'
    var_2 = var_0.customprovider.extra
    assert var_2 == 'value'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 12345
    var_1 = module_0.Generic(seed=var_0)
    var_2 = 99999
    var_3 = 11111
    var_4 = var_1.customprovider.seed
    assert var_4 == 12345
    var_5 = var_1.customprovider.custom_seed
    assert var_5 == 11111

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
    var_0 = 'custom'
    var_1 = False
    var_2 = 42
    var_3 = module_0.Generic(seed=var_2)
    var_4 = var_3.customprovider.seed
    assert var_4 == 42
    var_5 = 1
    var_6 = 100



# Parsed testcases at query #25
#--------------------------






# Parsed testcases at query #26
#--------------------------






# Parsed testcases at query #27
#--------------------------

# Partially parsed test_add_provider_adds_custom_provider. Retrieved 4/2 statements.
# Partially parsed test_add_provider_raises_type_error_for_non_baseprovider_subclass. Retrieved 1/6 statements.
# Partially parsed test_add_provider_raises_type_error_for_generic_instance. Retrieved 1/4 statements.
# Partially parsed test_add_provider_uses_meta_name. Retrieved 4/9 statements.
# Partially parsed test_add_provider_uses_lowercase_class_name_when_meta_name_missing. Retrieved 3/8 statements.
# Partially parsed test_add_provider_ignores_seed_kwarg. Retrieved 4/9 statements.
# Partially parsed test_add_provider_passes_extra_kwargs. Retrieved 3/11 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = module_0.Generic()
    var_2 = 'custom'
    var_3 = hasattr(var_1, var_2)
    var_4 = bool(var_3)
    assert var_4 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'test_value'
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
    var_5 = 'The provider must be a class'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'The provider must be a subclass of mimesis.providers.BaseProvider'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'Cannot add Generic instance to itself.'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'special_name'
    var_1 = module_0.Generic()
    var_2 = 'special_name'
    var_3 = hasattr(var_1, var_2)
    var_4 = bool(var_3)
    assert var_4 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'customprovider'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom'
    var_1 = 12345
    var_2 = module_0.Generic(seed=var_1)
    var_3 = 99999
    var_4 = var_2.custom.seed
    assert var_4 == 12345

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom'
    var_1 = module_0.Generic()
    var_2 = 'extra_value'
    var_3 = var_1.custom.extra_param
    assert var_3 == 'extra_value'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_issubclass_of_baseprovider_evaluates_false. Retrieved 4/22 statements.


def test_case_0():
    var_0 = 'mockdataprovider'
    var_1 = True
    var_2 = 'mockbaseprovider'
    var_3 = True



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

# Partially parsed test_generic_constructor_default_locale_and_seed. Retrieved 2/4 statements.
# Failed to parse test_generic_constructor_custom_locale.
# Partially parsed test_generic_constructor_base_data_providers_lazy. Retrieved 6/8 statements.
# Partially parsed test_generic_constructor_base_providers_immediate. Retrieved 4/5 statements.
# Failed to parse test_generic_constructor_locale_passed_to_lazy_providers.
# Failed to parse test_generic_constructor_str_representation.


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
    var_2 = 'person'
    var_3 = hasattr(var_0, var_2)
    var_4 = bool(not var_3)
    assert var_4 is True
    var_5 = var_0.person
    var_6 = bool(var_5 is not None)
    assert var_6 is True
    var_7 = hasattr(var_0, var_2)
    var_8 = bool(var_7)
    assert var_8 is True

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
    var_0 = 999
    var_1 = module_0.Generic(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 999
    var_3 = var_1.cryptographic.seed
    assert var_3 == 999

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.__dir__()
    var_2 = 'locale'
    var_3 = bool('locale' not in var_1)
    assert var_3 is True
    var_4 = 'seed'
    var_5 = bool('seed' not in var_1)
    assert var_5 is True
    var_6 = 'random'
    var_7 = bool('random' not in var_1)
    assert var_7 is True
    var_8 = 'person'
    var_9 = bool('person' in var_1)
    assert var_9 is True
    var_10 = 'address'
    var_11 = bool('address' in var_1)
    assert var_11 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.__dir__()
    var_2 = 'person'
    var_3 = bool('person' in var_1)
    assert var_3 is True
    var_4 = 'text'
    var_5 = bool('text' in var_1)
    assert var_5 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.__dir__()
    var_2 = 'cryptographic'
    var_3 = bool('cryptographic' in var_1)
    assert var_3 is True
    var_4 = 'development'
    var_5 = bool('development' in var_1)
    assert var_5 is True



# Parsed testcases at query #31
#--------------------------






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

# Failed to parse test_provider_registry_excludes_generic.




# Parsed testcases at query #34
#--------------------------

# Partially parsed test___getattr___returns_callable_attribute. Retrieved 7/8 statements.
# Partially parsed test___getattr___caches_attribute_after_first_call. Retrieved 5/10 statements.
# Partially parsed test___getattr___returns_none_for_non_callable_attribute. Retrieved 3/4 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'non_existent'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None

import mimesis.providers.generic as module_0

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

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 0
    assert var_1 == 1
    var_2 = 'test_provider'
    var_3 = var_0.__getattr__(var_2)
    var_4 = var_0.__getattr__(var_2)
    var_5 = bool(var_3 is var_4)
    assert var_5 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = '_invalid'
    var_2 = var_0.__getattr__(var_1)

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'non_callable'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_condition_at_line_16_evaluates_false. Retrieved 5/36 statements.


def test_case_0():
    var_0 = 'mockdataprovider'
    var_1 = True
    var_2 = 'mockbaseprovider'
    var_3 = True
    var_4 = 12345



# Parsed testcases at query #36
#--------------------------






# Parsed testcases at query #37
#--------------------------

# Partially parsed test_getattr_with_non_callable_attribute. Retrieved 2/3 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.test_attr
    assert var_1 is None



# Parsed testcases at query #38
#--------------------------

# Failed to parse test_provider_registry_excludes_generic.




# Parsed testcases at query #39
#--------------------------

# Failed to parse test_provider_registry_does_not_contain_generic.




# Parsed testcases at query #40
#--------------------------






####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test___getattr___returns_initialized_provider_for_valid_attribute. Retrieved 4/5 statements.
# Partially parsed test___getattr___caches_initialized_provider_in_dict. Retrieved 4/5 statements.
# Partially parsed test___getattr___returns_none_for_non_callable_attribute. Retrieved 3/4 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'non_existent'
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
    var_1 = '_invalid_start'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_add_provider_adds_custom_provider. Retrieved 5/13 statements.
# Partially parsed test_add_provider_uses_provider_meta_name. Retrieved 7/12 statements.
# Partially parsed test_add_provider_falls_back_to_lowercase_class_name. Retrieved 3/8 statements.
# Partially parsed test_add_provider_enforces_same_seed. Retrieved 4/12 statements.
# Partially parsed test_add_provider_raises_type_error_for_non_baseprovider_subclass. Retrieved 1/6 statements.
# Partially parsed test_add_provider_raises_type_error_for_generic_instance. Retrieved 1/4 statements.
# Partially parsed test_add_provider_passes_kwargs_to_provider. Retrieved 4/15 statements.
# Partially parsed test_add_provider_overwrites_existing_provider. Retrieved 5/19 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom'
    var_1 = False
    var_2 = module_0.Generic()
    var_3 = 'custom'
    var_4 = hasattr(var_2, var_3)
    var_5 = bool(var_4)
    assert var_5 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'special'
    var_1 = False
    var_2 = module_0.Generic()
    var_3 = 'special'
    var_4 = hasattr(var_2, var_3)
    var_5 = bool(var_4)
    assert var_5 is True
    var_6 = 'customprovider'
    var_7 = hasattr(var_2, var_6)
    var_8 = bool(not var_7)
    assert var_8 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'anotherprovider'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'seedcheck'
    var_1 = False
    var_2 = 42
    var_3 = module_0.Generic(seed=var_2)

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'not a class'
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
    var_0 = 'kwargs'
    var_1 = False
    var_2 = module_0.Generic()
    var_3 = 'test'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = False
    var_2 = 'test'
    var_3 = False
    var_4 = module_0.Generic()



# Parsed testcases at query #3
#--------------------------






# Parsed testcases at query #4
#--------------------------

# Partially parsed test_generic_constructor_default_locale_and_seed. Retrieved 2/4 statements.
# Failed to parse test_generic_constructor_custom_locale.
# Partially parsed test_generic_constructor_data_providers_lazy_loaded. Retrieved 7/8 statements.
# Partially parsed test_generic_constructor_base_providers_instantiated. Retrieved 4/5 statements.
# Partially parsed test_generic_constructor_respects_global_seed. Retrieved 1/3 statements.
# Partially parsed test_generic_constructor_with_explicit_random_instance. Retrieved 3/4 statements.


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
    var_4 = 'address'
    var_5 = hasattr(var_0, var_4)
    var_6 = bool(not var_5)
    assert var_6 is True
    var_7 = var_0.address
    var_8 = hasattr(var_0, var_4)
    var_9 = bool(var_8)
    assert var_9 is True

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
    var_1 = 'generic'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(not var_2)
    assert var_3 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.random._seed
    assert var_1 == 999

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 777
    var_1 = module_0.Generic(seed=var_0)
    var_2 = module_0.Generic(seed=var_0)
    var_3 = var_1.random
    var_4 = bool(var_1.random is not var_2.random)
    assert var_4 is True



# Parsed testcases at query #5
#--------------------------

# Failed to parse test_provider_registry_does_not_contain_generic_itself.




# Parsed testcases at query #6
#--------------------------

# Partially parsed test_generic_constructor_default_locale_and_seed. Retrieved 2/4 statements.
# Failed to parse test_generic_constructor_custom_locale.
# Partially parsed test_generic_constructor_lazy_initialization. Retrieved 8/10 statements.
# Partially parsed test_generic_constructor_base_provider_initialization. Retrieved 4/6 statements.


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
    var_1 = var_0._person
    var_2 = var_0.__dict__
    var_3 = 'person'
    var_4 = hasattr(var_2, var_3)
    var_5 = bool(not var_4)
    assert var_5 is True
    var_6 = var_0.person
    var_7 = var_0.__dict__
    var_8 = hasattr(var_7, var_3)
    var_9 = bool(var_8)
    assert var_9 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'random'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = var_0.random

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.Generic(seed=var_0)
    var_2 = var_1.seed
    assert var_2 is None

def test_case_0():
    pass



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_reseed_updates_seed_on_generic_and_providers. Retrieved 1/12 statements.
# Partially parsed test_reseed_propagates_to_regular_providers. Retrieved 1/8 statements.
# Partially parsed test_reseed_propagates_to_data_providers. Retrieved 2/11 statements.
# Failed to parse test_reseed_with_missing_seed_generates_new_random_seed.
# Partially parsed test_reseed_handles_attribute_error_gracefully. Retrieved 1/11 statements.


def test_case_0():
    var_0 = 12345

def test_case_0():
    var_0 = 999

def test_case_0():
    var_0 = 777
    var_1 = 'address'

def test_case_0():
    var_0 = 555



# Parsed testcases at query #8
#--------------------------






# Parsed testcases at query #9
#--------------------------

# Partially parsed test_provider_cls_is_not_subclass_of_baseprovider_when_not_baseprovider. Retrieved 2/14 statements.


def test_case_0():
    var_0 = 'not_base'
    var_1 = '_not_base'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_reseed_updates_seed_on_attached_providers. Retrieved 5/10 statements.
# Partially parsed test_reseed_handles_providers_without_reseed_method. Retrieved 5/8 statements.
# Partially parsed test_reseed_does_not_affect_other_attributes. Retrieved 3/4 statements.


import mimesis.providers.generic as module_0

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

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.__dir__()
    var_2 = {}
    var_3 = 67890
    var_4 = var_0.reseed(var_3)

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.seed
    var_2 = var_0.reseed()
    var_3 = var_0.seed
    var_4 = bool(var_0.seed != var_1)
    assert var_4 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'DummyProvider'
    var_2 = {}
    var_3 = 11111
    var_4 = var_0.reseed(var_3)
    var_5 = var_0.seed
    assert var_5 == 11111

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = module_0.Generic(var_0)
    var_2 = 22222
    var_3 = var_1.reseed(var_2)
    var_4 = var_1.locale
    assert var_4 == 'en'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 33333
    var_2 = var_0.reseed(var_1)
    var_3 = var_0.some_attribute
    assert var_3 == 'test_value'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_add_provider_with_non_generic_instance. Retrieved 4/10 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom'
    var_1 = module_0.Generic()
    var_2 = 'custom'
    var_3 = getattr(var_1, var_2)



# Parsed testcases at query #12
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
    var_4 = 'full_name'
    var_5 = hasattr(var_2, var_4)
    var_6 = bool(var_5)
    assert var_6 is True

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
    var_2 = 'mocked'
    var_3 = lambda locale, seed: var_2
    var_4 = setattr(var_0, var_1, var_3)
    var_5 = 'test_provider'
    var_6 = var_0.__getattr__(var_5)
    assert var_6 == 'mocked'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = '_non_callable'
    var_2 = 'not a function'
    var_3 = setattr(var_0, var_1, var_2)
    var_4 = 'non_callable'
    var_5 = var_0.__getattr__(var_4)
    assert var_5 is None



# Parsed testcases at query #13
#--------------------------






# Parsed testcases at query #14
#--------------------------

# Partially parsed test_reseed_updates_seed_on_attached_providers. Retrieved 4/9 statements.


import mimesis.providers.generic as module_0

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

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.__dir__()
    var_2 = 67890
    var_3 = var_0.reseed(var_2)

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.seed
    var_2 = var_0.reseed()
    var_3 = var_0.seed
    var_4 = bool(var_0.seed != var_1)
    assert var_4 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = '_test_provider'
    var_2 = None
    var_3 = setattr(var_0, var_1, var_2)
    var_4 = 11111
    var_5 = var_0.reseed(var_4)
    var_6 = var_0.seed
    assert var_6 == 11111

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 22222
    var_2 = var_0.reseed(var_1)
    var_3 = var_0.seed
    assert var_3 == 22222



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_add_provider_without_meta_name. Retrieved 3/8 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'customprovider'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_add_provider_with_non_generic_instance. Retrieved 3/9 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom'
    var_1 = module_0.Generic()
    var_2 = var_1.custom



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_provider_registry_skips_generic. Retrieved 5/13 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'generic'
    var_1 = 'test_provider'
    var_2 = module_0.Generic()
    var_3 = hasattr(var_2, var_0)
    var_4 = bool(not var_3)
    assert var_4 is True
    var_5 = hasattr(var_2, var_1)
    var_6 = bool(var_5)
    assert var_6 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_generic_constructor_default_locale_and_seed. Retrieved 2/4 statements.
# Failed to parse test_generic_constructor_custom_locale.
# Partially parsed test_generic_constructor_excludes_base_attributes. Retrieved 3/6 statements.
# Partially parsed test_generic_constructor_lazy_loading. Retrieved 2/3 statements.
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
    var_2 = 'person'
    var_3 = bool('person' in var_1)
    assert var_3 is True
    var_4 = dir(var_0)
    var_5 = 'address'
    var_6 = bool('address' in var_4)
    assert var_6 is True

import mimesis.providers.generic as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = dir(var_0)
    var_2 = 'locale'
    var_3 = bool('locale' not in var_1)
    assert var_3 is True
    var_4 = module_1.BaseProvider()

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
    assert var_3 == 999

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = dir(var_0)
    var_2 = 'generic'
    var_3 = bool('generic' not in var_1)
    assert var_3 is True

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
    var_1 = 'person'
    var_2 = bool('person' not in var_0.__dict__)
    assert var_2 is True
    var_3 = '_person'
    var_4 = bool('_person' in var_0.__dict__)
    assert var_4 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_add_provider_does_not_raise_typeerror_for_non_generic_instance. Retrieved 4/9 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom'
    var_1 = module_0.Generic()
    var_2 = 'custom'
    var_3 = hasattr(var_1, var_2)
    var_4 = bool(var_3)
    assert var_4 is True



# Parsed testcases at query #20
#--------------------------

# Failed to parse test_generic_initialization_with_custom_locale.
# Partially parsed test_generic_initialization_with_locale_and_seed. Retrieved 1/3 statements.
# Partially parsed test_generic_initialization_providers_registered. Retrieved 3/5 statements.
# Partially parsed test_generic_initialization_base_providers_instantiated. Retrieved 1/4 statements.
# Partially parsed test_generic_initialization_data_providers_lazy. Retrieved 1/6 statements.
# Partially parsed test_generic_initialization_seed_propagation. Retrieved 2/4 statements.
# Partially parsed test_generic_initialization_excludes_base_attributes. Retrieved 4/8 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.locale
    var_2 = var_0.seed

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
    var_1 = var_0.__dir__()
    var_2 = len(var_1)
    var_3 = bool(var_2 > 0)
    assert var_3 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 55555
    var_1 = module_0.Generic(seed=var_0)

def test_case_0():
    pass

import mimesis.providers.generic as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.__dir__()
    var_2 = module_1.BaseProvider()
    var_3 = 'locale'



# Parsed testcases at query #21
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



# Parsed testcases at query #22
#--------------------------






# Parsed testcases at query #23
#--------------------------

# Partially parsed test_add_provider_with_meta_name_attribute. Retrieved 5/12 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom'
    var_1 = module_0.Generic()
    var_2 = 'custom'
    var_3 = hasattr(var_1, var_2)
    var_4 = bool(var_3)
    assert var_4 is True
    var_5 = var_1.custom



# Parsed testcases at query #24
#--------------------------






# Parsed testcases at query #25
#--------------------------

# Partially parsed test_generic_initialization_with_default_locale_and_seed. Retrieved 2/4 statements.
# Failed to parse test_generic_initialization_with_custom_locale.
# Partially parsed test_generic_initialization_sets_base_provider_instances. Retrieved 1/8 statements.
# Partially parsed test_generic_initialization_sets_base_data_provider_attributes. Retrieved 1/9 statements.
# Partially parsed test_generic_initialization_with_seed_propagates_to_providers. Retrieved 2/7 statements.
# Partially parsed test_generic_initialization_creates_lazy_data_providers. Retrieved 1/7 statements.
# Failed to parse test_generic_initialization_with_custom_locale_propagates_to_lazy_providers.
# Partially parsed test_generic_initialization_with_random_instance. Retrieved 1/5 statements.
# Partially parsed test_generic_initialization_dir_excludes_base_attributes. Retrieved 4/8 statements.
# Failed to parse test_generic_initialization_str_representation.


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
    var_1 = 'generic'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(not var_2)
    assert var_3 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 999
    var_1 = module_0.Generic(seed=var_0)

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.Generic(seed=var_0)
    var_2 = var_1.seed
    assert var_2 is None

def test_case_0():
    var_0 = 777

def test_case_0():
    pass

import mimesis.providers.generic as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.__dir__()
    var_2 = module_1.BaseProvider()
    var_3 = 'locale'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.__dir__()



