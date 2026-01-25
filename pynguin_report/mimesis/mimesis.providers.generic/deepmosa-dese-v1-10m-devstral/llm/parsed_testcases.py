####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_generic_initialization. Retrieved 6/8 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = 42
    var_2 = module_0.Generic(var_0, var_1)
    var_3 = 'random'
    var_4 = hasattr(var_2, var_3)
    var_5 = var_2.random



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_add_provider_with_valid_class. Retrieved 4/6 statements.
# Partially parsed test_add_provider_with_non_baseprovider_class. Retrieved 1/3 statements.
# Partially parsed test_add_provider_with_generic_class. Retrieved 1/3 statements.
# Partially parsed test_add_provider_with_seed_enforcement. Retrieved 2/5 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'customprovider'
    var_2 = hasattr(var_0, var_1)
    var_3 = var_0.customprovider

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'not_a_class'
    var_2 = var_0.add_provider(var_1)

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Generic(seed=var_0)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_add_provider_with_generic_class. Retrieved 1/3 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_getattr_initializes_data_provider_on_first_access. Retrieved 3/5 statements.
# Partially parsed test_getattr_initializes_provider_with_correct_locale_and_seed. Retrieved 1/4 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.personal
    var_2 = var_0.address

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Generic(seed=var_0)
    var_2 = var_1.personal
    var_3 = var_1.personal

def test_case_0():
    var_0 = 42



# Parsed testcases at query #5
#--------------------------




import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'generic'
    var_2 = hasattr(var_0, var_1)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_reseed_updates_seed_and_providers. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 42
    var_1 = 100



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_add_provider_without_meta_name. Retrieved 3/6 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'customprovider'
    var_2 = hasattr(var_0, var_1)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_skip_generic_in_initialization. Retrieved 3/5 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'generic'
    var_2 = hasattr(var_0, var_1)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_getattr_returns_none_when_attribute_is_not_callable. Retrieved 1/2 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_generic_initialization. Retrieved 4/6 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'random'
    var_2 = hasattr(var_0, var_1)
    var_3 = var_0.random



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_add_provider_with_generic_class. Retrieved 1/3 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_generic_initialization_with_default_locale. Retrieved 2/4 statements.
# Failed to parse test_generic_initialization_with_custom_locale.
# Partially parsed test_generic_initialization_with_custom_seed. Retrieved 3/5 statements.
# Partially parsed test_generic_initialization_with_custom_locale_and_seed. Retrieved 1/6 statements.
# Partially parsed test_generic_initialization_adds_providers. Retrieved 1/11 statements.
# Failed to parse test_generic_initialization_with_custom_random.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.random

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Generic(seed=var_0)
    var_2 = var_1.random

def test_case_0():
    var_0 = 42

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_reseed_continues_on_attribute_error. Retrieved 4/7 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = ()
    var_2 = 42
    var_3 = var_0.reseed(var_2)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_generic_initialization_with_default_locale. Retrieved 1/2 statements.
# Failed to parse test_generic_initialization_with_custom_locale.
# Partially parsed test_generic_initialization_with_seed. Retrieved 2/3 statements.
# Partially parsed test_generic_initialization_with_locale_and_seed. Retrieved 1/4 statements.
# Failed to parse test_generic_initialization_with_custom_random.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Generic(seed=var_0)

def test_case_0():
    var_0 = 42

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'personal'
    var_2 = hasattr(var_0, var_1)
    var_3 = 'address'
    var_4 = hasattr(var_0, var_3)
    var_5 = 'food'
    var_6 = hasattr(var_0, var_5)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_add_provider_with_valid_class. Retrieved 4/6 statements.
# Partially parsed test_add_provider_with_invalid_class. Retrieved 1/3 statements.
# Partially parsed test_add_provider_with_generic_class. Retrieved 1/3 statements.
# Partially parsed test_add_provider_with_custom_kwargs. Retrieved 5/7 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'baseprovider'
    var_2 = hasattr(var_0, var_1)
    var_3 = var_0.baseprovider

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'not_a_class'
    var_2 = var_0.add_provider(var_1)

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'test'
    var_2 = 'baseprovider'
    var_3 = hasattr(var_0, var_2)
    var_4 = var_0.baseprovider



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_getattr_returns_none_when_attribute_is_not_callable. Retrieved 1/2 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_getattr_returns_none_when_attribute_is_not_callable. Retrieved 1/2 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_reseed_continues_on_attribute_error. Retrieved 3/4 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 42
    var_2 = var_0.reseed(var_1)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_getattr_lazy_loads_data_provider. Retrieved 2/3 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.person

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.person
    var_2 = var_0.person

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'de'
    var_1 = module_0.Generic(var_0)
    var_2 = var_1.person

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Generic(seed=var_0)
    var_2 = var_1.person



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_add_provider_without_meta_name. Retrieved 3/6 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'customprovider'
    var_2 = hasattr(var_0, var_1)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_getattr_creates_and_returns_provider_instance. Retrieved 2/3 statements.
# Partially parsed test_getattr_with_locale_and_seed. Retrieved 1/5 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.person

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.person

def test_case_0():
    var_0 = 42



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_generic_initialization_with_custom_locale. Retrieved 2/6 statements.
# Partially parsed test_generic_initialization_with_custom_locale_and_seed. Retrieved 3/7 statements.
# Partially parsed test_generic_initialization_creates_provider_instances. Retrieved 3/5 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'personal'
    var_2 = hasattr(var_0, var_1)
    var_3 = 'address'
    var_4 = hasattr(var_0, var_3)

def test_case_0():
    var_0 = 'personal'
    var_1 = 'address'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Generic(seed=var_0)
    var_2 = 'personal'
    var_3 = hasattr(var_1, var_2)
    var_4 = 'address'
    var_5 = hasattr(var_1, var_4)

def test_case_0():
    var_0 = 42
    var_1 = 'personal'
    var_2 = 'address'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.personal
    var_2 = var_0.address

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'generic'
    var_2 = hasattr(var_0, var_1)



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_add_provider_without_meta_name. Retrieved 3/6 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'customprovider'
    var_2 = hasattr(var_0, var_1)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_add_provider_with_valid_class. Retrieved 4/6 statements.
# Partially parsed test_add_provider_with_invalid_class. Retrieved 1/3 statements.
# Partially parsed test_add_provider_with_generic_class. Retrieved 1/3 statements.
# Partially parsed test_add_provider_with_custom_kwargs. Retrieved 4/5 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'person'
    var_2 = hasattr(var_0, var_1)
    var_3 = var_0.person

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'not_a_class'
    var_2 = var_0.add_provider(var_1)

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'female'
    var_2 = 'person'
    var_3 = hasattr(var_0, var_2)



# Parsed testcases at query #25
#--------------------------

# Failed to parse test_issubclass_predicate_evaluates_to_false.




# Parsed testcases at query #26
#--------------------------

# Partially parsed test_add_provider_with_generic_class. Retrieved 1/3 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_getattr_returns_none_for_non_callable_attribute. Retrieved 1/2 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_reseed_continues_on_attribute_error. Retrieved 3/4 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 42
    var_2 = var_0.reseed(var_1)



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_add_provider_without_meta_name. Retrieved 3/6 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'customprovider'
    var_2 = hasattr(var_0, var_1)



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_getattr_returns_provider_instance. Retrieved 2/3 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.person

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.invalid_attribute
    assert var_1 is None

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.person
    var_2 = var_0.person



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_add_provider_without_meta_name. Retrieved 3/6 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'customprovider'
    var_2 = hasattr(var_0, var_1)



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_add_provider_with_valid_provider. Retrieved 2/4 statements.
# Partially parsed test_add_provider_with_non_baseprovider_subclass. Retrieved 1/3 statements.
# Partially parsed test_add_provider_with_generic_instance. Retrieved 1/3 statements.
# Partially parsed test_add_provider_with_custom_seed. Retrieved 2/3 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.baseprovider

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'not a class'
    var_2 = var_0.add_provider(var_1)

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Generic(seed=var_0)



# Parsed testcases at query #33
#--------------------------

# Failed to parse test_generic_provider_skips_itself_in_initialization.




# Parsed testcases at query #34
#--------------------------

# Partially parsed test_reseed_continues_on_attribute_error. Retrieved 3/4 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 42
    var_2 = var_0.reseed(var_1)



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_generic_initialization_with_default_locale. Retrieved 1/2 statements.
# Failed to parse test_generic_initialization_with_custom_locale.
# Partially parsed test_generic_initialization_with_seed. Retrieved 2/3 statements.
# Partially parsed test_generic_initialization_with_custom_random. Retrieved 1/3 statements.
# Partially parsed test_generic_initialization_sets_providers. Retrieved 1/12 statements.
# Failed to parse test_generic_initialization_with_missing_seed.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Generic(seed=var_0)

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_getattr_returns_none_when_attribute_is_not_callable. Retrieved 1/2 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_add_provider_without_meta_name. Retrieved 3/6 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'customprovider'
    var_2 = hasattr(var_0, var_1)



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_getattr_with_non_callable_attribute. Retrieved 1/2 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_getattr_returns_none_when_attribute_is_not_callable. Retrieved 1/2 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_generic_skips_itself_in_registry. Retrieved 3/5 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'generic'
    var_2 = hasattr(var_0, var_1)



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_reseed_continues_on_attribute_error. Retrieved 3/4 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 42
    var_2 = var_0.reseed(var_1)



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_getattr_returns_none_when_attribute_is_not_callable. Retrieved 1/2 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_generic_initialization. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 42
    var_1 = 'random'



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_add_provider_without_meta_name. Retrieved 3/6 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'customprovider'
    var_2 = hasattr(var_0, var_1)



# Parsed testcases at query #45
#--------------------------




import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'generic'
    var_2 = hasattr(var_0, var_1)



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_getattr_with_non_callable_attribute. Retrieved 1/2 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_getattr_initializes_and_returns_provider. Retrieved 2/3 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.person

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.person
    var_2 = var_0.person

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_generic_provider_not_base_data_provider. Retrieved 1/6 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_reseed_continues_on_attribute_error. Retrieved 3/4 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 42
    var_2 = var_0.reseed(var_1)



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_add_provider_without_meta_name. Retrieved 3/6 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'customprovider'
    var_2 = hasattr(var_0, var_1)



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_reseed_updates_seed_for_all_providers. Retrieved 4/7 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.seed
    var_2 = 42
    var_3 = var_0.reseed(var_2)



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_generic_initialization. Retrieved 4/12 statements.


def test_case_0():
    var_0 = 42
    var_1 = 'person'
    var_2 = 'address'
    var_3 = '_generic'



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_generic_skips_itself_in_registry. Retrieved 3/5 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'generic'
    var_2 = hasattr(var_0, var_1)



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_add_provider_with_valid_class. Retrieved 4/6 statements.
# Partially parsed test_add_provider_with_invalid_class. Retrieved 1/3 statements.
# Partially parsed test_add_provider_with_generic_class. Retrieved 1/3 statements.
# Partially parsed test_add_provider_with_custom_kwargs. Retrieved 5/7 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'baseprovider'
    var_2 = hasattr(var_0, var_1)
    var_3 = var_0.baseprovider

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'value'
    var_2 = 'baseprovider'
    var_3 = hasattr(var_0, var_2)
    var_4 = var_0.baseprovider



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_getattr_initializes_and_returns_provider_instance. Retrieved 2/3 statements.
# Partially parsed test_getattr_initializes_provider_with_correct_locale_and_seed. Retrieved 1/4 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.person

def test_case_0():
    var_0 = 42



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_getattr_with_non_callable_attribute. Retrieved 1/2 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test___getattr___lazily_initializes_data_provider. Retrieved 3/4 statements.
# Partially parsed test___getattr___uses_locale_and_seed. Retrieved 2/5 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'nonexistent'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'person'
    var_2 = var_0.__getattr__(var_1)

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'person'
    var_2 = var_0.__getattr__(var_1)
    var_3 = var_0.__getattr__(var_1)

def test_case_0():
    var_0 = 42
    var_1 = 'person'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_generic_initialization. Retrieved 8/10 statements.
# Failed to parse test_generic_initialization_with_custom_locale.
# Partially parsed test_generic_initialization_with_custom_seed. Retrieved 3/4 statements.
# Partially parsed test_generic_initialization_with_custom_random. Retrieved 1/3 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.random
    var_2 = 'locale'
    var_3 = hasattr(var_0, var_2)
    var_4 = 'seed'
    var_5 = hasattr(var_0, var_4)
    var_6 = 'random'
    var_7 = hasattr(var_0, var_6)

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Generic(seed=var_0)
    var_2 = 32

def test_case_0():
    var_0 = 100

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'not_a_random_instance'
    var_1 = module_0.Generic()



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_getattr_with_non_callable_attribute. Retrieved 1/2 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_add_provider_with_valid_provider. Retrieved 4/6 statements.
# Partially parsed test_add_provider_with_non_baseprovider_subclass. Retrieved 1/3 statements.
# Partially parsed test_add_provider_with_generic_instance. Retrieved 1/3 statements.
# Partially parsed test_add_provider_with_custom_kwargs. Retrieved 2/3 statements.
# Partially parsed test_add_provider_with_seed_override. Retrieved 3/4 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'customprovider'
    var_2 = hasattr(var_0, var_1)
    var_3 = var_0.customprovider

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'not_a_class'
    var_2 = var_0.add_provider(var_1)

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'value'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Generic(seed=var_0)
    var_2 = 100



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_generic_initialization_with_default_locale_and_seed. Retrieved 2/4 statements.
# Partially parsed test_generic_initialization_with_custom_locale_and_seed. Retrieved 1/6 statements.
# Partially parsed test_generic_initialization_excludes_itself_from_providers. Retrieved 4/5 statements.
# Failed to parse test_generic_initialization_with_custom_random.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.random

def test_case_0():
    var_0 = 42

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'personal'
    var_2 = hasattr(var_0, var_1)
    var_3 = 'address'
    var_4 = hasattr(var_0, var_3)
    var_5 = 'food'
    var_6 = hasattr(var_0, var_5)
    var_7 = 'text'
    var_8 = hasattr(var_0, var_7)

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'generic'
    var_2 = None
    var_3 = getattr(var_0, var_1, var_2)

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'not_a_random_instance'
    var_1 = module_0.Generic()



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_reseed_updates_seed_and_propagates_to_providers. Retrieved 4/7 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.seed
    var_2 = 42
    var_3 = var_0.reseed(var_2)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_generic_initialization_with_default_locale_and_seed. Retrieved 4/6 statements.
# Partially parsed test_generic_initialization_with_custom_locale_and_seed. Retrieved 2/8 statements.
# Partially parsed test_generic_initialization_sets_all_providers. Retrieved 1/12 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'random'
    var_2 = hasattr(var_0, var_1)
    var_3 = var_0.random

def test_case_0():
    var_0 = 42
    var_1 = 'random'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_getattr_with_non_callable_attribute. Retrieved 1/2 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_getattr_returns_none_when_attribute_is_not_callable. Retrieved 1/2 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_generic_provider_skips_itself. Retrieved 3/5 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'generic'
    var_2 = hasattr(var_0, var_1)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_add_provider_with_valid_class. Retrieved 3/4 statements.
# Partially parsed test_add_provider_with_invalid_class. Retrieved 1/3 statements.
# Partially parsed test_add_provider_with_generic_class. Retrieved 1/3 statements.
# Partially parsed test_add_provider_with_custom_kwargs. Retrieved 4/5 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'baseprovider'
    var_2 = hasattr(var_0, var_1)

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'not a class'
    var_2 = var_0.add_provider(var_1)

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'value'
    var_2 = 'baseprovider'
    var_3 = hasattr(var_0, var_2)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_add_provider_with_valid_class. Retrieved 4/6 statements.
# Partially parsed test_add_provider_with_invalid_class. Retrieved 1/3 statements.
# Partially parsed test_add_provider_with_generic_class. Retrieved 1/3 statements.
# Partially parsed test_add_provider_with_custom_provider. Retrieved 5/9 statements.
# Partially parsed test_add_provider_with_kwargs. Retrieved 6/10 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'baseprovider'
    var_2 = hasattr(var_0, var_1)
    var_3 = var_0.baseprovider

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'not a class'
    var_2 = var_0.add_provider(var_1)

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom'
    var_1 = module_0.Generic()
    var_2 = 'custom'
    var_3 = hasattr(var_1, var_2)
    var_4 = var_1.custom

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom'
    var_1 = module_0.Generic()
    var_2 = 'value'
    var_3 = 'custom'
    var_4 = hasattr(var_1, var_3)
    var_5 = var_1.custom



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_reseed_continues_on_attribute_error. Retrieved 3/4 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 42
    var_2 = var_0.reseed(var_1)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_getattr_initializes_and_returns_provider_on_first_access. Retrieved 2/3 statements.
# Partially parsed test_getattr_initializes_provider_with_correct_locale_and_seed. Retrieved 1/4 statements.
# Partially parsed test_getattr_returns_none_for_non_callable_attribute. Retrieved 1/2 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.person

def test_case_0():
    var_0 = 42

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_add_provider_without_meta_name. Retrieved 3/6 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'customprovider'
    var_2 = hasattr(var_0, var_1)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_add_provider_with_valid_class. Retrieved 4/6 statements.
# Partially parsed test_add_provider_with_invalid_class. Retrieved 1/3 statements.
# Partially parsed test_add_provider_with_generic_class. Retrieved 1/3 statements.
# Partially parsed test_add_provider_with_kwargs. Retrieved 4/5 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'baseprovider'
    var_2 = hasattr(var_0, var_1)
    var_3 = var_0.baseprovider

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'value'
    var_2 = 'baseprovider'
    var_3 = hasattr(var_0, var_2)



# Parsed testcases at query #17
#--------------------------

# Failed to parse test_issubclass_predicate_evaluates_to_false.




# Parsed testcases at query #18
#--------------------------

# Failed to parse test_issubclass_predicate_evaluates_to_false.




# Parsed testcases at query #19
#--------------------------

# Partially parsed test_getattr_with_non_callable_attribute. Retrieved 1/2 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_reseed_updates_all_providers_seed. Retrieved 3/6 statements.
# Partially parsed test_reseed_ignores_non_provider_attributes. Retrieved 3/4 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 42
    var_2 = var_0.reseed(var_1)

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 42
    var_2 = var_0.reseed(var_1)

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 100
    var_1 = module_0.Generic(seed=var_0)
    var_2 = var_1.reseed()

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 42
    var_2 = var_0.reseed(var_1)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_reseed_continues_on_attribute_error. Retrieved 3/4 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 42
    var_2 = var_0.reseed(var_1)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_reseed_continues_on_attribute_error. Retrieved 3/4 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 42
    var_2 = var_0.reseed(var_1)



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_add_provider_without_meta_name. Retrieved 3/6 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'customprovider'
    var_2 = hasattr(var_0, var_1)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_getattr_with_non_callable_attribute. Retrieved 1/2 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()



# Parsed testcases at query #25
#--------------------------

# Failed to parse test_generic_initialization_with_custom_locale.
# Failed to parse test_generic_initialization_with_missing_seed.
# Partially parsed test_generic_initialization_sets_providers. Retrieved 3/7 statements.
# Failed to parse test_generic_initialization_with_custom_random.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'seed'
    var_2 = hasattr(var_0, var_1)
    var_3 = 'random'
    var_4 = hasattr(var_0, var_3)

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Generic(seed=var_0)

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.__dir__()
    var_2 = len(var_1)

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.__dir__()



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_generic_provider_skips_self_registration. Retrieved 3/7 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'generic'
    var_2 = hasattr(var_0, var_1)



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_reseed_updates_all_providers. Retrieved 3/6 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 42
    var_2 = var_0.reseed(var_1)

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.reseed()

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 42
    var_2 = var_0.reseed(var_1)



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_reseed_continues_on_attribute_error. Retrieved 3/4 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 42
    var_2 = var_0.reseed(var_1)



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_reseed_continues_on_attribute_error. Retrieved 4/7 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = ()
    var_2 = 42
    var_3 = var_0.reseed(var_2)



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_getattr_with_non_callable_attribute. Retrieved 1/2 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_add_provider_without_meta_name. Retrieved 3/6 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'customprovider'
    var_2 = hasattr(var_0, var_1)



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_getattr_returns_none_when_attribute_is_not_callable. Retrieved 3/4 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'test_attr'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None



# Parsed testcases at query #33
#--------------------------




import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'nonexistent_attribute'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = '_nonexistent_attribute'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = '_nonexistent_attribute'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = '_nonexistent_attribute'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = '_nonexistent_attribute'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = '_nonexistent_attribute'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = '_nonexistent_attribute'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = '_nonexistent_attribute'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = '_nonexistent_attribute'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = '_nonexistent_attribute'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = '_nonexistent_attribute'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = '_nonexistent_attribute'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = '_nonexistent_attribute'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = '_nonexistent_attribute'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = '_nonexistent_attribute'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = '_nonexistent_attribute'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = '_nonexistent_attribute'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = '_nonexistent_attribute'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = '_nonexistent_attribute'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = '_nonexistent_attribute'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = '_nonexistent_attribute'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = '_nonexistent_attribute'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = '_nonexistent_attribute'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = '_nonexistent_attribute'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = '_nonexistent_attribute'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = '_nonexistent_attribute'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = '_nonexistent_attribute'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = '_nonexistent_attribute'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None



# Parsed testcases at query #34
#--------------------------




import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'generic'
    var_2 = hasattr(var_0, var_1)



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_reseed_continues_on_attribute_error. Retrieved 3/4 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 42
    var_2 = var_0.reseed(var_1)



# Parsed testcases at query #36
#--------------------------




import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'generic'
    var_2 = hasattr(var_0, var_1)



# Parsed testcases at query #37
#--------------------------




import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.reseed()



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_getattr_with_non_callable_attribute. Retrieved 1/2 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_reseed_updates_seed_and_propagates_to_providers. Retrieved 4/6 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.seed
    var_2 = 42
    var_3 = var_0.reseed(var_2)



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_generic_skips_itself_in_registry. Retrieved 7/13 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'generic'
    var_1 = 'mock'
    var_2 = False
    var_3 = 'mock'
    var_4 = module_0.Generic()
    var_5 = hasattr(var_4, var_0)
    var_6 = hasattr(var_4, var_3)



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_getattr_calls_provider_class_with_locale_and_seed. Retrieved 1/5 statements.
# Partially parsed test_getattr_returns_none_for_non_callable_attribute. Retrieved 2/3 statements.
# Partially parsed test_getattr_stores_provider_instance_in_dict. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 42

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.test_attr
    assert var_1 is None

def test_case_0():
    var_0 = 42
    var_1 = 'person'



# Parsed testcases at query #42
#--------------------------

# Failed to parse test_generic_provider_not_base_data_provider.




# Parsed testcases at query #43
#--------------------------

# Partially parsed test_add_provider_without_meta_name. Retrieved 3/7 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'Meta'
    var_2 = 'name'



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_generic_skips_itself_in_registry. Retrieved 3/7 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'generic'
    var_2 = hasattr(var_0, var_1)



# Parsed testcases at query #45
#--------------------------




import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'generic'
    var_2 = hasattr(var_0, var_1)



