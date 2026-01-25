####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_generic_initialization_with_default_locale. Retrieved 2/4 statements.
# Failed to parse test_generic_initialization_with_custom_locale.
# Partially parsed test_generic_initialization_with_custom_locale_and_seed. Retrieved 1/3 statements.
# Partially parsed test_generic_initialization_sets_providers. Retrieved 1/12 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.locale
    var_2 = var_0.seed
    var_3 = var_0.random

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Generic(seed=var_0)
    var_2 = var_1.locale
    var_3 = var_1.seed
    assert var_3 == 42

def test_case_0():
    var_0 = 123

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_getattr_instantiates_and_returns_data_provider. Retrieved 2/3 statements.
# Partially parsed test_getattr_uses_locale_and_seed. Retrieved 1/4 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.non_existent_attr
    assert var_1 is None

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.person
    var_2 = var_0.__dict__['person']
    var_3 = bool(var_0.__dict__['person'] is var_1)
    assert var_3 is True

def test_case_0():
    var_0 = 42



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_reseed_updates_seed_and_calls_reseed_on_providers. Retrieved 4/7 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.seed
    var_2 = 42
    var_3 = var_0.reseed(var_2)
    var_4 = var_0.seed
    var_5 = bool(var_0.seed == var_2)
    assert var_5 is True
    var_6 = var_0.seed
    var_7 = bool(var_0.seed != var_1)
    assert var_7 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_reseed_propagates_to_all_providers. Retrieved 3/6 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.seed
    var_2 = 42
    var_3 = var_0.reseed(var_2)
    var_4 = var_0.seed
    var_5 = bool(var_0.seed == var_2)
    assert var_5 is True
    var_6 = var_0.seed
    var_7 = bool(var_0.seed != var_1)
    assert var_7 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 42
    var_2 = var_0.reseed(var_1)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_generic_init_sets_base_data_provider_as_private_attribute. Retrieved 10/21 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'mock_data'
    var_1 = 'mock'
    var_2 = 'mock_data'
    var_3 = 'mock'
    var_4 = module_0.Generic()
    var_5 = '_mock_data'
    var_6 = hasattr(var_4, var_5)
    var_7 = bool(var_6)
    assert var_7 is True
    var_8 = hasattr(var_4, var_2)
    var_9 = bool(not var_8)
    assert var_9 is True
    var_10 = getattr(var_4, var_5)
    var_11 = getattr(var_4, var_5)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_add_provider_with_valid_provider. Retrieved 4/6 statements.
# Partially parsed test_add_provider_with_non_baseprovider_subclass. Retrieved 1/3 statements.
# Partially parsed test_add_provider_with_generic_instance. Retrieved 1/3 statements.
# Partially parsed test_add_provider_with_custom_kwargs. Retrieved 4/5 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'person'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = var_0.person

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'not_a_class'
    var_2 = {}
    var_3 = var_0.add_provider(var_1, **var_2)

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'female'
    var_2 = 'person'
    var_3 = hasattr(var_0, var_2)
    var_4 = bool(var_3)
    assert var_4 is True
    var_5 = var_0.person.gender
    assert var_5 == 'female'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_add_provider_with_valid_class. Retrieved 4/6 statements.
# Partially parsed test_add_provider_with_non_baseprovider_class. Retrieved 1/3 statements.
# Partially parsed test_add_provider_with_generic_class. Retrieved 1/3 statements.
# Partially parsed test_add_provider_with_kwargs. Retrieved 2/3 statements.
# Partially parsed test_add_provider_with_seed. Retrieved 2/3 statements.
# Partially parsed test_add_provider_without_meta_name. Retrieved 3/4 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'customprovider'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = var_0.customprovider

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'not a class'
    var_2 = {}
    var_3 = var_0.add_provider(var_1, **var_2)

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
    var_2 = var_0.customprovider.custom_arg
    assert var_2 == 'value'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Generic(seed=var_0)
    var_2 = var_1.customprovider.seed
    assert var_2 == 42

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'providerwithoutmeta'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_generic_init_sets_base_data_provider_correctly. Retrieved 1/8 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_add_provider_without_meta_name. Retrieved 3/6 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'customprovider'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_generic_skips_itself_in_registry. Retrieved 3/5 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'generic'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(not var_2)
    assert var_3 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_add_provider_without_meta_name. Retrieved 3/6 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'customprovider'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_generic_initialization. Retrieved 2/4 statements.
# Partially parsed test_generic_initialization_with_locale_and_seed. Retrieved 1/6 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.locale
    var_2 = var_0.seed
    var_3 = var_0.random
    var_4 = bool(var_0.random is not None)
    assert var_4 is True
    var_5 = var_0.random

def test_case_0():
    var_0 = 42



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_getattr_returns_none_when_attribute_is_not_callable. Retrieved 1/2 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.test_attr
    assert var_1 is None



# Parsed testcases at query #14
#--------------------------




import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'generic'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(not var_2)
    assert var_3 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test___getattr___initializes_and_returns_data_provider. Retrieved 1/3 statements.
# Partially parsed test___getattr___initializes_with_correct_locale_and_seed. Retrieved 1/5 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.nonexistent_attr
    assert var_1 is None

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.__dict__['person']

def test_case_0():
    var_0 = 42



# Parsed testcases at query #16
#--------------------------




import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'generic'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(not var_2)
    assert var_3 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_add_provider_with_valid_class. Retrieved 4/6 statements.
# Partially parsed test_add_provider_with_invalid_class. Retrieved 1/3 statements.
# Partially parsed test_add_provider_with_generic_class. Retrieved 1/3 statements.
# Partially parsed test_add_provider_with_kwargs. Retrieved 5/7 statements.
# Partially parsed test_add_provider_with_seed_kwarg. Retrieved 6/8 statements.
# Partially parsed test_add_provider_with_class_without_meta. Retrieved 4/8 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'baseprovider'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = var_0.baseprovider

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'test'
    var_2 = 'baseprovider'
    var_3 = hasattr(var_0, var_2)
    var_4 = bool(var_3)
    assert var_4 is True
    var_5 = var_0.baseprovider

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Generic(seed=var_0)
    var_2 = 100
    var_3 = 'baseprovider'
    var_4 = hasattr(var_1, var_3)
    var_5 = bool(var_4)
    assert var_5 is True
    var_6 = var_1.baseprovider
    var_7 = var_1.baseprovider.seed
    assert var_7 == 42

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'not_a_class'
    var_2 = {}
    var_3 = var_0.add_provider(var_1, **var_2)

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'customprovider'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = var_0.customprovider



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_reseed_updates_all_providers_seed. Retrieved 3/6 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.seed
    var_2 = 42
    var_3 = var_0.reseed(var_2)
    var_4 = var_0.seed
    var_5 = bool(var_0.seed == var_2)
    assert var_5 is True
    var_6 = var_0.seed
    var_7 = bool(var_0.seed != var_1)
    assert var_7 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 42
    var_2 = var_0.reseed(var_1)

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.seed
    var_2 = var_0.reseed()
    var_3 = var_0.seed
    var_4 = bool(var_0.seed != var_1)
    assert var_4 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test___getattr___lazily_initializes_data_provider. Retrieved 2/3 statements.
# Partially parsed test___getattr___passes_locale_and_seed_to_provider. Retrieved 1/4 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.nonexistent_attr
    assert var_1 is None

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.personal

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.personal
    var_2 = var_0.personal
    var_3 = bool(var_1 is var_2)
    assert var_3 is True

def test_case_0():
    var_0 = 42



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_getattr_with_non_callable_attribute. Retrieved 1/2 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.test_attr
    assert var_1 is None



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_generic_skips_itself_in_registry. Retrieved 3/6 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'generic'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(not var_2)
    assert var_3 is True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_reseed_continues_on_attribute_error. Retrieved 3/4 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 42
    var_2 = var_0.reseed(var_1)
    var_3 = bool(True)
    assert var_3 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_getattr_returns_none_when_attribute_is_not_callable. Retrieved 1/2 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.test_attr
    assert var_1 is None



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_ensure_predicate_at_line_19_evaluates_to_false. Retrieved 3/7 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom'
    var_1 = module_0.Generic()
    var_2 = 'Meta'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_getattr_predicate_false. Retrieved 3/4 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'test_attr'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_getattr_initializes_and_returns_data_provider. Retrieved 3/5 statements.
# Partially parsed test_getattr_returns_none_for_non_callable_attribute. Retrieved 3/4 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'nonexistent_attr'
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
    var_1 = 'test_attr'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_generic_initialization_with_default_locale. Retrieved 2/4 statements.
# Failed to parse test_generic_initialization_with_custom_locale.
# Partially parsed test_generic_initialization_with_custom_seed. Retrieved 3/5 statements.
# Partially parsed test_generic_initialization_with_custom_locale_and_seed. Retrieved 1/6 statements.
# Failed to parse test_generic_initialization_with_custom_random.
# Partially parsed test_generic_initialization_with_invalid_random. Retrieved 1/3 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.locale
    var_2 = var_0.random
    var_3 = var_0.seed

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Generic(seed=var_0)
    var_2 = var_1.locale
    var_3 = var_1.random
    var_4 = var_1.seed
    assert var_4 == 42

def test_case_0():
    var_0 = 100

def test_case_0():
    var_0 = 'not a random object'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'personal'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = 'address'
    var_5 = hasattr(var_0, var_4)
    var_6 = bool(var_5)
    assert var_6 is True
    var_7 = 'food'
    var_8 = hasattr(var_0, var_7)
    var_9 = bool(var_8)
    assert var_9 is True



# Parsed testcases at query #28
#--------------------------




import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'generic'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(not var_2)
    assert var_3 is True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_getattr_returns_none_when_attribute_is_not_callable. Retrieved 1/2 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.test_attr
    assert var_1 is None



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_getattr_returns_none_when_attribute_is_not_callable. Retrieved 1/2 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.test_attr
    assert var_1 is None



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_reseed_continues_on_attribute_error. Retrieved 3/4 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 42
    var_2 = var_0.reseed(var_1)
    var_3 = bool(True)
    assert var_3 is True



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_add_provider_without_meta_name. Retrieved 3/6 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'customprovider'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True



# Parsed testcases at query #33
#--------------------------

# Failed to parse test_generic_provider_not_base_data_provider.




# Parsed testcases at query #34
#--------------------------

# Partially parsed test_getattr_initializes_and_returns_provider_instance. Retrieved 2/3 statements.
# Partially parsed test_getattr_returns_none_for_non_callable_attribute. Retrieved 1/2 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.nonexistent_attr
    assert var_1 is None

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.person

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Generic(seed=var_0)
    var_2 = var_1.person
    var_3 = var_2.seed
    assert var_3 == 42

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.person
    var_2 = var_0.person
    var_3 = bool(var_1 is var_2)
    assert var_3 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.test_attr
    assert var_1 is None



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_reseed_continues_on_attribute_error. Retrieved 4/7 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = ()
    var_2 = 42
    var_3 = var_0.reseed(var_2)



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_getattr_with_non_callable_attribute. Retrieved 1/2 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.test_attr
    assert var_1 is None



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_add_provider_without_meta_name. Retrieved 4/8 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'customprovider'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = getattr(var_0, var_1)



# Parsed testcases at query #38
#--------------------------

# Failed to parse test_generic_skips_itself_in_registry.




# Parsed testcases at query #39
#--------------------------

# Partially parsed test_add_provider_with_valid_class. Retrieved 4/6 statements.
# Partially parsed test_add_provider_with_invalid_class. Retrieved 1/3 statements.
# Partially parsed test_add_provider_with_generic_class. Retrieved 1/3 statements.
# Partially parsed test_add_provider_with_kwargs. Retrieved 5/7 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'baseprovider'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = var_0.baseprovider

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
    var_4 = bool(var_3)
    assert var_4 is True
    var_5 = var_0.baseprovider



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_reseed_updates_seed_and_providers. Retrieved 4/7 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.seed
    var_2 = 42
    var_3 = var_0.reseed(var_2)
    var_4 = var_0.seed
    assert var_4 == 42
    var_5 = var_0.seed
    var_6 = bool(var_0.seed != var_1)
    assert var_6 is True



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_getattr_with_non_callable_attribute. Retrieved 1/2 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.test_attr
    assert var_1 is None



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_getattr_calls_provider_with_locale_and_seed. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 42

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
    var_3 = bool(var_1 is var_2)
    assert var_3 is True



# Parsed testcases at query #43
#--------------------------

# Failed to parse test_generic_not_registered_in_registry.




# Parsed testcases at query #44
#--------------------------

# Failed to parse test_generic_provider_not_base_data_provider.




# Parsed testcases at query #45
#--------------------------

# Partially parsed test_generic_skip_itself_in_registry. Retrieved 5/9 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'generic'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(not var_2)
    assert var_3 is True
    var_4 = '_generic'
    var_5 = hasattr(var_0, var_4)
    var_6 = bool(not var_5)
    assert var_6 is True



# Parsed testcases at query #46
#--------------------------




import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'generic'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(not var_2)
    assert var_3 is True



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_reseed_updates_seed_and_all_providers. Retrieved 4/7 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.seed
    var_2 = 42
    var_3 = var_0.reseed(var_2)
    var_4 = var_0.seed
    var_5 = bool(var_0.seed == var_2)
    assert var_5 is True



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_add_provider_without_meta_name. Retrieved 3/6 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'customprovider'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_getattr_returns_none_for_non_callable_attribute. Retrieved 3/4 statements.
# Partially parsed test_getattr_instantiates_and_returns_callable_attribute. Retrieved 4/5 statements.
# Partially parsed test_getattr_caches_instantiated_attribute. Retrieved 5/6 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'nonexistent_attr'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'test_attr'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'callable_result'
    var_2 = 'test_attr'
    var_3 = var_0.__getattr__(var_2)
    assert var_3 == 'callable_result'
    var_4 = var_0.test_attr
    assert var_4 == 'callable_result'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'callable_result'
    var_2 = 'test_attr'
    var_3 = var_0.__getattr__(var_2)
    var_4 = var_0.__getattr__(var_2)
    var_5 = bool(var_3 == var_4)
    assert var_5 is True
    var_6 = var_0.test_attr
    assert var_6 == 'callable_result'



# Parsed testcases at query #50
#--------------------------

# Partially parsed test___getattr___lazily_instantiates_data_provider. Retrieved 2/3 statements.
# Partially parsed test___getattr___passes_locale_and_seed_to_provider. Retrieved 1/4 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.nonexistent_attr
    assert var_1 is None

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.person
    var_2 = var_0.person
    var_3 = bool(var_0.person is var_0.person)
    assert var_3 is True

def test_case_0():
    var_0 = 42



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_generic_initialization_with_default_locale_and_seed. Retrieved 4/6 statements.
# Failed to parse test_generic_initialization_with_custom_locale.
# Partially parsed test_generic_initialization_with_custom_locale_and_seed. Retrieved 1/3 statements.
# Partially parsed test_generic_initialization_sets_all_providers. Retrieved 1/12 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.locale
    var_2 = var_0.seed
    var_3 = 'random'
    var_4 = hasattr(var_0, var_3)
    var_5 = bool(var_4)
    assert var_5 is True
    var_6 = var_0.random

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Generic(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 42

def test_case_0():
    var_0 = 123

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_generic_provider_skips_itself. Retrieved 3/6 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'generic'
    var_1 = module_0.Generic()
    var_2 = hasattr(var_1, var_0)
    var_3 = bool(not var_2)
    assert var_3 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_getattr_initializes_and_returns_data_provider. Retrieved 2/3 statements.
# Partially parsed test_getattr_initializes_with_correct_locale_and_seed. Retrieved 1/4 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.nonexistent_attr
    assert var_1 is None

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.person
    var_2 = var_0.__dict__['person']
    var_3 = bool(var_0.__dict__['person'] is var_1)
    assert var_3 is True

def test_case_0():
    var_0 = 42



# Parsed testcases at query #4
#--------------------------

# Failed to parse test_generic_provider_not_base_data_provider.




# Parsed testcases at query #5
#--------------------------

# Partially parsed test_add_provider_with_valid_class. Retrieved 2/4 statements.
# Partially parsed test_add_provider_with_non_baseprovider_class. Retrieved 1/3 statements.
# Partially parsed test_add_provider_with_generic_class. Retrieved 1/3 statements.
# Partially parsed test_add_provider_with_custom_kwargs. Retrieved 2/3 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.mockprovider

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'not a class'
    var_2 = {}
    var_3 = var_0.add_provider(var_1, **var_2)

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
    var_2 = var_0.mockprovider.custom_arg
    assert var_2 == 'value'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_add_provider_without_meta_name. Retrieved 3/6 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'customprovider'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_getattr_returns_none_when_attribute_is_not_callable. Retrieved 1/2 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.test_attr
    assert var_1 is None



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_generic_skips_itself_in_registry. Retrieved 3/5 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'generic'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(not var_2)
    assert var_3 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_reseed_with_seed. Retrieved 3/5 statements.
# Partially parsed test_reseed_without_seed. Retrieved 2/4 statements.
# Partially parsed test_reseed_updates_all_providers. Retrieved 4/6 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 42
    var_2 = var_0.reseed(var_1)
    var_3 = var_0.seed
    assert var_3 == 42

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.reseed()
    var_2 = var_0.seed

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.seed
    var_2 = 42
    var_3 = var_0.reseed(var_2)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_reseed_propagates_to_all_providers. Retrieved 4/7 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Generic(seed=var_0)
    var_2 = var_1.seed
    var_3 = var_1.person
    var_4 = 100
    var_5 = var_1.reseed(var_4)
    var_6 = var_1.seed
    assert var_6 == 100
    var_7 = var_3.seed
    assert var_7 == 100
    var_8 = bool(var_2 != var_1.seed)
    assert var_8 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Generic(seed=var_0)
    var_2 = var_1.reseed()
    var_3 = var_1.seed

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Generic(seed=var_0)
    var_2 = 200
    var_3 = var_1.reseed(var_2)

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Generic(seed=var_0)
    var_2 = 300
    var_3 = var_1.reseed(var_2)
    var_4 = var_1.locale
    var_5 = var_1.locale
    var_6 = 'seed'
    var_7 = hasattr(var_5, var_6)
    var_8 = bool(not var_7)
    assert var_8 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_generic_initialization. Retrieved 4/6 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.locale
    var_2 = var_0.seed
    var_3 = 'random'
    var_4 = hasattr(var_0, var_3)
    var_5 = bool(var_4)
    assert var_5 is True
    var_6 = var_0.random



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_getattr_with_non_callable_attribute. Retrieved 1/2 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.test_attr
    assert var_1 is None



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_getattr_returns_none_when_attribute_is_not_callable. Retrieved 1/2 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.test_attr
    assert var_1 is None



# Parsed testcases at query #14
#--------------------------

# Partially parsed test___getattr___lazily_initializes_data_provider. Retrieved 2/3 statements.
# Partially parsed test___getattr___uses_correct_locale_and_seed. Retrieved 1/4 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.nonexistent_attr
    assert var_1 is None

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.person
    var_2 = var_0.__dict__['person']
    var_3 = bool(var_0.__dict__['person'] is var_1)
    assert var_3 is True

def test_case_0():
    var_0 = 42

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.locale
    var_2 = bool(var_0.locale is not None)
    assert var_2 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_reseed_updates_seed_and_reseeds_all_providers. Retrieved 4/7 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.seed
    var_2 = 42
    var_3 = var_0.reseed(var_2)
    var_4 = var_0.seed
    var_5 = bool(var_0.seed == var_2)
    assert var_5 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_add_provider_without_meta_name. Retrieved 3/6 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'customprovider'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_getattr_lazily_initializes_provider. Retrieved 2/3 statements.
# Partially parsed test_getattr_returns_none_for_non_callable_attribute. Retrieved 1/2 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.nonexistent_attr
    assert var_1 is None

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.person

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Generic(seed=var_0)
    var_2 = var_1.person
    var_3 = var_2.seed
    assert var_3 == 42

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.person
    var_2 = var_0.person
    var_3 = bool(var_1 is var_2)
    assert var_3 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.test_attr
    assert var_1 is None



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_reseed_continues_on_attribute_error. Retrieved 3/4 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 42
    var_2 = var_0.reseed(var_1)
    var_3 = bool(True)
    assert var_3 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_generic_initialization. Retrieved 1/9 statements.


def test_case_0():
    var_0 = 42
    var_1 = 'person'
    var_2 = 'address'
    var_3 = 'food'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_getattr_with_non_callable_attribute. Retrieved 1/2 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.test_attr
    assert var_1 is None



# Parsed testcases at query #21
#--------------------------

# Failed to parse test_provider_cls_is_not_generic.




# Parsed testcases at query #22
#--------------------------

# Partially parsed test_add_provider_without_meta_name. Retrieved 3/6 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'customprovider'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_add_provider_without_meta_name. Retrieved 3/6 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'customprovider'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test___getattr___lazily_initializes_data_provider. Retrieved 2/3 statements.
# Partially parsed test___getattr___uses_correct_locale_and_seed. Retrieved 1/4 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.nonexistent_attr
    assert var_1 is None

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.person
    var_2 = var_0.person
    var_3 = bool(var_0.person is var_1)
    assert var_3 is True

def test_case_0():
    var_0 = 42



# Parsed testcases at query #25
#--------------------------

# Failed to parse test_issubclass_predicate_evaluates_to_false.




# Parsed testcases at query #26
#--------------------------

# Partially parsed test_generic_provider_not_registered. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'generic'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_getattr_returns_none_when_attribute_is_not_callable. Retrieved 1/2 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.test_attr
    assert var_1 is None



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_add_provider_without_meta_name. Retrieved 3/6 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'customprovider'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True



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

# Partially parsed test_generic_initialization. Retrieved 1/8 statements.


def test_case_0():
    var_0 = 42
    var_1 = 'person'
    var_2 = 'address'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_getattr_initializes_and_returns_provider_instance. Retrieved 2/3 statements.
# Partially parsed test_getattr_initializes_provider_with_correct_locale_and_seed. Retrieved 1/4 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.non_existent_attr
    assert var_1 is None

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.person
    var_2 = var_0.__dict__['person']
    var_3 = bool(var_0.__dict__['person'] is var_1)
    assert var_3 is True

def test_case_0():
    var_0 = 42



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_reseed_continues_on_attribute_error. Retrieved 3/4 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 42
    var_2 = var_0.reseed(var_1)
    var_3 = bool(True)
    assert var_3 is True



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_add_provider_with_valid_class. Retrieved 4/6 statements.
# Partially parsed test_add_provider_with_non_baseprovider_subclass. Retrieved 1/3 statements.
# Partially parsed test_add_provider_with_generic_instance. Retrieved 1/3 statements.
# Partially parsed test_add_provider_with_custom_kwargs. Retrieved 2/3 statements.
# Partially parsed test_add_provider_with_seed_override. Retrieved 3/4 statements.
# Partially parsed test_add_provider_without_meta_name. Retrieved 3/4 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'customprovider'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = var_0.customprovider

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'not a class'
    var_2 = {}
    var_3 = var_0.add_provider(var_1, **var_2)

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
    var_2 = var_0.customprovider.custom_arg
    assert var_2 == 'value'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Generic(seed=var_0)
    var_2 = 100
    var_3 = var_1.customprovider.seed
    assert var_3 == 42

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'customproviderwithoutmeta'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_generic_initialization. Retrieved 6/8 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = 42
    var_2 = module_0.Generic(var_0, var_1)
    var_3 = var_2.locale
    assert var_3 == 'en'
    var_4 = var_2.seed
    assert var_4 == 42
    var_5 = 'random'
    var_6 = hasattr(var_2, var_5)
    var_7 = bool(var_6)
    assert var_7 is True
    var_8 = var_2.random



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_getattr_returns_none_when_attribute_is_not_callable. Retrieved 1/2 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.test_attr
    assert var_1 is None



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_generic_init_sets_base_data_provider_correctly. Retrieved 1/13 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_getattr_returns_none_when_attribute_is_not_callable. Retrieved 1/2 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.test_attr
    assert var_1 is None



# Parsed testcases at query #38
#--------------------------

# Failed to parse test_issubclass_predicate_evaluates_to_false.




# Parsed testcases at query #39
#--------------------------

# Partially parsed test_getattr_initializes_and_returns_data_provider. Retrieved 2/3 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.nonexistent_attr
    assert var_1 is None

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.person
    var_2 = var_1.locale
    var_3 = bool(var_1.locale == var_0.locale)
    assert var_3 is True
    var_4 = var_1.seed
    var_5 = bool(var_1.seed == var_0.seed)
    assert var_5 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.person
    var_2 = var_0.person
    var_3 = bool(var_1 is var_2)
    assert var_3 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = '_test_attr'
    var_2 = 'not_callable'
    var_3 = setattr(var_0, var_1, var_2)
    var_4 = var_0.test_attr
    assert var_4 is None



# Parsed testcases at query #40
#--------------------------




import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 42
    var_2 = var_0.reseed(var_1)
    var_3 = bool(True)
    assert var_3 is True



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_add_provider_with_valid_class. Retrieved 4/6 statements.
# Partially parsed test_add_provider_with_non_baseprovider_class. Retrieved 1/3 statements.
# Partially parsed test_add_provider_with_generic_class. Retrieved 1/3 statements.
# Partially parsed test_add_provider_with_kwargs. Retrieved 2/3 statements.
# Partially parsed test_add_provider_with_seed_override. Retrieved 3/5 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'customprovider'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = var_0.customprovider

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'not a class'
    var_2 = {}
    var_3 = var_0.add_provider(var_1, **var_2)

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
    var_2 = var_0.customprovider.custom_arg
    assert var_2 == 'value'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Generic(seed=var_0)
    var_2 = 100



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_generic_provider_skips_itself_in_initialization. Retrieved 3/6 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'generic'
    var_1 = module_0.Generic()
    var_2 = hasattr(var_1, var_0)
    var_3 = bool(not var_2)
    assert var_3 is True



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_reseed_continues_on_attribute_error. Retrieved 3/4 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 42
    var_2 = var_0.reseed(var_1)
    var_3 = bool(True)
    assert var_3 is True



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_generic_init_sets_base_data_provider_correctly. Retrieved 5/10 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'test_data'
    var_1 = 'test_data'
    var_2 = module_0.Generic()
    var_3 = '_test_data'
    var_4 = hasattr(var_2, var_3)
    var_5 = bool(var_4)
    assert var_5 is True
    var_6 = var_2._test_data



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_getattr_with_non_callable_attribute. Retrieved 1/2 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.test_attr
    assert var_1 is None



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_generic_initialization_with_custom_locale. Retrieved 3/8 statements.
# Partially parsed test_generic_initialization_with_custom_locale_and_seed. Retrieved 4/9 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.locale
    var_2 = var_0.seed
    var_3 = 'personal'
    var_4 = hasattr(var_0, var_3)
    var_5 = bool(var_4)
    assert var_5 is True
    var_6 = 'address'
    var_7 = hasattr(var_0, var_6)
    var_8 = bool(var_7)
    assert var_8 is True
    var_9 = 'food'
    var_10 = hasattr(var_0, var_9)
    var_11 = bool(var_10)
    assert var_11 is True

def test_case_0():
    var_0 = 'personal'
    var_1 = 'address'
    var_2 = 'food'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Generic(seed=var_0)
    var_2 = var_1.locale
    var_3 = var_1.seed
    assert var_3 == 42
    var_4 = 'personal'
    var_5 = hasattr(var_1, var_4)
    var_6 = bool(var_5)
    assert var_6 is True
    var_7 = 'address'
    var_8 = hasattr(var_1, var_7)
    var_9 = bool(var_8)
    assert var_9 is True
    var_10 = 'food'
    var_11 = hasattr(var_1, var_10)
    var_12 = bool(var_11)
    assert var_12 is True

def test_case_0():
    var_0 = 42
    var_1 = 'personal'
    var_2 = 'address'
    var_3 = 'food'



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_add_provider_without_meta_name. Retrieved 3/6 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'customprovider'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True



