####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test___getattr___initializes_and_returns_data_provider. Retrieved 3/4 statements.
# Partially parsed test___getattr___initializes_with_correct_locale_and_seed. Retrieved 2/5 statements.
# Partially parsed test___getattr___returns_none_for_non_callable_attribute. Retrieved 3/4 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'nonexistent_attribute'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'personal'
    var_2 = var_0.__getattr__(var_1)
    var_3 = var_0.__dict__['personal']
    var_4 = bool(var_0.__dict__['personal'] is var_2)
    assert var_4 is True

def test_case_0():
    var_0 = 42
    var_1 = 'personal'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'test_attr'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_generic_initialization_with_default_locale. Retrieved 2/4 statements.
# Failed to parse test_generic_initialization_with_custom_locale.
# Partially parsed test_generic_initialization_with_custom_seed. Retrieved 3/5 statements.
# Partially parsed test_generic_initialization_with_custom_locale_and_seed. Retrieved 1/6 statements.
# Partially parsed test_generic_initialization_sets_data_providers. Retrieved 1/13 statements.


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
    var_2 = var_1.seed
    assert var_2 == 42
    var_3 = var_1.random

def test_case_0():
    var_0 = 100

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_add_provider_with_valid_class. Retrieved 5/9 statements.
# Partially parsed test_add_provider_with_invalid_class. Retrieved 1/5 statements.
# Partially parsed test_add_provider_with_generic_instance. Retrieved 1/3 statements.
# Partially parsed test_add_provider_with_custom_kwargs. Retrieved 5/11 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'custom'
    var_2 = 'custom'
    var_3 = hasattr(var_0, var_2)
    var_4 = bool(var_3)
    assert var_4 is True
    var_5 = var_0.custom

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
    var_2 = {}
    var_3 = var_0.add_provider(var_1, **var_2)

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'custom'
    var_2 = 42
    var_3 = 'custom'
    var_4 = hasattr(var_0, var_3)
    var_5 = bool(var_4)
    assert var_5 is True
    var_6 = var_0.custom.value
    assert var_6 == 42



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_add_provider_with_generic_instance. Retrieved 1/4 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_add_provider_without_meta_name. Retrieved 3/6 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'customprovider'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_getattr_with_non_callable_attribute. Retrieved 1/2 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.test_attr
    assert var_1 is None



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_getattr_initializes_and_returns_provider. Retrieved 2/3 statements.
# Partially parsed test_getattr_initializes_provider_with_locale_and_seed. Retrieved 1/4 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.nonexistent_attr
    assert var_1 is None

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.person
    var_2 = bool(var_1 is var_0.person)
    assert var_2 is True

def test_case_0():
    var_0 = 42



# Parsed testcases at query #8
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



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_getattr_with_non_callable_attribute. Retrieved 1/2 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.test_attr
    assert var_1 is None



# Parsed testcases at query #10
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
    var_1 = 123
    var_2 = var_0.reseed(var_1)

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.reseed()
    var_2 = var_0.seed



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_generic_init_sets_baseprovider_instance. Retrieved 1/9 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_add_provider_with_valid_class. Retrieved 4/6 statements.
# Partially parsed test_add_provider_with_non_baseprovider_class. Retrieved 1/3 statements.
# Partially parsed test_add_provider_with_generic_class. Retrieved 1/3 statements.
# Partially parsed test_add_provider_with_custom_kwargs. Retrieved 2/3 statements.
# Partially parsed test_add_provider_with_seed_override. Retrieved 3/4 statements.


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



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_generic_provider_not_base_data_provider. Retrieved 1/4 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_getattr_returns_none_when_attribute_is_not_callable. Retrieved 1/2 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.test_attr
    assert var_1 is None



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_getattr_returns_none_when_attribute_is_not_callable. Retrieved 1/2 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.test_attr
    assert var_1 is None



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_reseed_continues_on_attribute_error. Retrieved 3/4 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 42
    var_2 = var_0.reseed(var_1)
    var_3 = bool(True)
    assert var_3 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_add_provider_without_meta_name. Retrieved 3/6 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'customprovider'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_getattr_with_non_callable_attribute. Retrieved 1/2 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.test_attr
    assert var_1 is None



# Parsed testcases at query #19
#--------------------------

# Failed to parse test_generic_provider_not_base_data_provider.




# Parsed testcases at query #20
#--------------------------

# Partially parsed test_add_provider_with_valid_class. Retrieved 5/9 statements.
# Partially parsed test_add_provider_with_invalid_class. Retrieved 1/5 statements.
# Partially parsed test_add_provider_with_generic_class. Retrieved 1/3 statements.
# Partially parsed test_add_provider_with_custom_kwargs. Retrieved 3/9 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'custom'
    var_2 = 'custom'
    var_3 = hasattr(var_0, var_2)
    var_4 = bool(var_3)
    assert var_4 is True
    var_5 = var_0.custom

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
    var_2 = {}
    var_3 = var_0.add_provider(var_1, **var_2)

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'custom'
    var_2 = 42
    var_3 = var_0.custom.value
    assert var_3 == 42



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_add_provider_without_meta_name. Retrieved 3/6 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'customprovider'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True



# Parsed testcases at query #22
#--------------------------

# Failed to parse test_generic_provider_skips_itself.




# Parsed testcases at query #23
#--------------------------

# Partially parsed test_generic_initialization_with_default_locale. Retrieved 2/4 statements.
# Failed to parse test_generic_initialization_with_custom_locale.
# Partially parsed test_generic_initialization_with_custom_seed. Retrieved 2/3 statements.
# Failed to parse test_generic_initialization_with_custom_random.
# Partially parsed test_generic_initialization_with_invalid_random_type. Retrieved 1/3 statements.


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
    var_2 = var_1.seed
    assert var_2 == 42

def test_case_0():
    var_0 = 'not_a_random_instance'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_getattr_returns_none_when_attribute_is_not_callable. Retrieved 1/2 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.test_attr
    assert var_1 is None



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_reseed_updates_seed_and_all_providers. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 42
    var_1 = 100



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_generic_provider_skips_itself. Retrieved 3/6 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'generic'
    var_1 = module_0.Generic()
    var_2 = hasattr(var_1, var_0)
    var_3 = bool(not var_2)
    assert var_3 is True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_generic_initialization. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 42
    var_1 = 'random'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_getattr_creates_and_returns_provider_instance. Retrieved 2/3 statements.
# Partially parsed test_getattr_uses_same_seed_for_provider. Retrieved 4/6 statements.
# Failed to parse test_getattr_uses_same_locale_for_provider.


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
    var_3 = var_1.person



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_generic_initialization. Retrieved 4/6 statements.
# Partially parsed test_generic_initialization_with_locale_and_seed. Retrieved 2/8 statements.
# Failed to parse test_generic_initialization_with_custom_random.
# Partially parsed test_generic_initialization_with_invalid_random. Retrieved 1/3 statements.


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

def test_case_0():
    var_0 = 42
    var_1 = 'random'

def test_case_0():
    var_0 = 'invalid_random'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_reseed_handles_attribute_error_gracefully. Retrieved 3/4 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 42
    var_2 = var_0.reseed(var_1)
    var_3 = bool(True)
    assert var_3 is True



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test___getattr___initializes_and_returns_provider_instance. Retrieved 2/3 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.non_existent_attr
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



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_getattr_returns_none_when_attribute_is_not_callable. Retrieved 1/2 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.test_attr
    assert var_1 is None



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_getattr_with_non_callable_attribute. Retrieved 1/2 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.test_attr
    assert var_1 is None



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_getattr_returns_none_for_nonexistent_attribute_with_underscore_and_callable. Retrieved 1/2 statements.
# Partially parsed test_getattr_returns_none_for_nonexistent_attribute_with_underscore_and_callable_with_args. Retrieved 2/3 statements.
# Partially parsed test_getattr_returns_none_for_nonexistent_attribute_with_underscore_and_callable_with_kwargs. Retrieved 2/3 statements.
# Partially parsed test_getattr_returns_none_for_nonexistent_attribute_with_underscore_and_callable_with_args_and_kwargs. Retrieved 2/3 statements.
# Partially parsed test_getattr_returns_none_for_nonexistent_attribute_with_underscore_and_callable_with_args_and_kwargs_and_seed. Retrieved 3/4 statements.
# Partially parsed test_getattr_returns_none_for_nonexistent_attribute_with_underscore_and_callable_with_args_and_kwargs_and_seed_and_locale. Retrieved 4/5 statements.
# Partially parsed test_getattr_returns_none_for_nonexistent_attribute_with_underscore_and_callable_with_args_and_kwargs_and_seed_and_locale_and_nonexistent_attr. Retrieved 5/6 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.nonexistent_attr
    assert var_1 is None

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0._nonexistent_attr
    assert var_1 is None

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'arg'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'arg'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'arg'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Generic(seed=var_0)
    var_2 = 'arg'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = 42
    var_2 = module_0.Generic(var_0, var_1)
    var_3 = 'arg'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = 42
    var_2 = module_0.Generic(var_0, var_1)
    var_3 = 'arg'
    var_4 = 'nonexistent_attr'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_generic_initialization_with_default_locale. Retrieved 2/4 statements.
# Failed to parse test_generic_initialization_with_custom_locale.
# Partially parsed test_generic_initialization_with_custom_seed. Retrieved 3/5 statements.
# Partially parsed test_generic_initialization_with_custom_locale_and_seed. Retrieved 1/6 statements.
# Partially parsed test_generic_initialization_sets_providers. Retrieved 1/12 statements.


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
    var_0 = 42

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()

def test_case_0():
    pass



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_getattr_initializes_and_returns_provider. Retrieved 3/4 statements.
# Partially parsed test_getattr_initializes_provider_with_locale_and_seed. Retrieved 2/5 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'nonexistent_attr'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'personal'
    var_2 = var_0.__getattr__(var_1)
    var_3 = var_0.personal
    var_4 = bool(var_0.personal is var_2)
    assert var_4 is True

def test_case_0():
    var_0 = 42
    var_1 = 'personal'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = '_test_attr'
    var_2 = 'not_callable'
    var_3 = setattr(var_0, var_1, var_2)
    var_4 = 'test_attr'
    var_5 = var_0.__getattr__(var_4)
    assert var_5 is None



# Parsed testcases at query #7
#--------------------------

# Failed to parse test_issubclass_predicate_evaluates_to_false.




# Parsed testcases at query #8
#--------------------------

# Partially parsed test_generic_provider_skips_itself_during_initialization. Retrieved 3/6 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'generic'
    var_1 = module_0.Generic()
    var_2 = hasattr(var_1, var_0)
    var_3 = bool(not var_2)
    assert var_3 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_add_provider_with_valid_class. Retrieved 2/4 statements.
# Partially parsed test_add_provider_with_non_baseprovider_class. Retrieved 1/3 statements.
# Partially parsed test_add_provider_with_generic_instance. Retrieved 1/3 statements.
# Partially parsed test_add_provider_with_custom_seed. Retrieved 2/3 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.customprovider
    var_2 = var_0.customprovider.seed
    var_3 = bool(var_0.customprovider.seed == var_0.seed)
    assert var_3 is True

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
    var_0 = 42
    var_1 = module_0.Generic(seed=var_0)
    var_2 = var_1.customprovider.seed
    assert var_2 == 42



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_reseed_updates_all_providers_seed. Retrieved 3/6 statements.
# Partially parsed test_reseed_with_missing_seed. Retrieved 3/4 statements.


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
    var_1 = 42
    var_2 = var_0.reseed(var_1)

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.reseed()
    var_2 = var_0.seed



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_add_provider_with_valid_class. Retrieved 3/4 statements.
# Partially parsed test_add_provider_with_invalid_class. Retrieved 1/3 statements.
# Partially parsed test_add_provider_with_generic_class. Retrieved 1/3 statements.
# Partially parsed test_add_provider_with_kwargs. Retrieved 4/5 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'baseprovider'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
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
    var_1 = 'value'
    var_2 = 'baseprovider'
    var_3 = hasattr(var_0, var_2)
    var_4 = bool(var_3)
    assert var_4 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_issubclass_predicate_evaluates_to_false. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 16 evaluates to False when provider_cls is not a subclass of BaseProvider.'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_reseed_handles_attribute_error. Retrieved 3/4 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 42
    var_2 = var_0.reseed(var_1)
    var_3 = bool(True)
    assert var_3 is True



# Parsed testcases at query #14
#--------------------------

# Failed to parse test_issubclass_predicate_evaluates_to_false.




# Parsed testcases at query #15
#--------------------------

# Failed to parse test_skip_generic_provider_in_init.




# Parsed testcases at query #16
#--------------------------

# Partially parsed test_reseed_updates_seed_and_all_providers. Retrieved 3/6 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 42
    var_2 = var_0.reseed(var_1)
    var_3 = var_0.seed
    var_4 = bool(var_0.seed == var_1)
    assert var_4 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_add_provider_with_non_baseprovider_subclass. Retrieved 1/5 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #18
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
    var_2 = var_0.seed
    var_3 = var_0.random

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Generic(seed=var_0)
    var_2 = var_1.locale
    var_3 = var_1.seed
    assert var_3 == 42
    var_4 = var_1.random

def test_case_0():
    var_0 = 42

def test_case_0():
    var_0 = 'not a random instance'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_reseed_continues_on_attribute_error. Retrieved 3/4 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 42
    var_2 = var_0.reseed(var_1)
    var_3 = bool(True)
    assert var_3 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_add_provider_with_non_baseprovider_subclass. Retrieved 1/5 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #21
#--------------------------




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



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_getattr_returns_provider_instance. Retrieved 3/5 statements.
# Partially parsed test_getattr_uses_locale_and_seed. Retrieved 1/4 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.person
    var_2 = var_0.address

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.invalid_attr
    assert var_1 is None

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.person
    var_2 = var_0.person
    var_3 = bool(var_1 is var_2)
    assert var_3 is True

def test_case_0():
    var_0 = 42



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

# Partially parsed test_add_provider_with_valid_class. Retrieved 4/6 statements.
# Partially parsed test_add_provider_with_non_baseprovider_class. Retrieved 1/3 statements.
# Partially parsed test_add_provider_with_generic_class. Retrieved 1/3 statements.
# Partially parsed test_add_provider_with_kwargs. Retrieved 2/3 statements.
# Partially parsed test_add_provider_with_seed_override. Retrieved 2/3 statements.


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
    var_1 = 'test'
    var_2 = var_0.customprovider.custom_arg
    assert var_2 == 'test'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Generic(seed=var_0)
    var_2 = var_1.customprovider.seed
    assert var_2 == 42



# Parsed testcases at query #25
#--------------------------

# Failed to parse test_generic_initialization_with_custom_locale.
# Partially parsed test_generic_initialization_with_custom_seed_and_locale. Retrieved 1/3 statements.
# Failed to parse test_generic_initialization_with_missing_seed.
# Partially parsed test_generic_initialization_with_custom_random. Retrieved 1/3 statements.
# Partially parsed test_generic_initialization_with_invalid_random. Retrieved 2/4 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.locale
    var_2 = 'random'
    var_3 = hasattr(var_0, var_2)
    var_4 = bool(var_3)
    assert var_4 is True
    var_5 = 'seed'
    var_6 = hasattr(var_0, var_5)
    var_7 = bool(var_6)
    assert var_7 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Generic(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 42

def test_case_0():
    var_0 = 42

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.Generic(seed=var_0)
    var_2 = var_1.seed
    assert var_2 is None

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

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'generic'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(not var_2)
    assert var_3 is True

def test_case_0():
    var_0 = 42

def test_case_0():
    var_0 = 42
    var_1 = 'invalid_random'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_add_provider_raises_typeerror_when_adding_generic_instance. Retrieved 1/3 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_add_provider_raises_typeerror_when_adding_generic_instance. Retrieved 1/3 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_add_provider_raises_typeerror_when_adding_generic_instance. Retrieved 1/3 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_add_provider_raises_typeerror_when_adding_generic_instance. Retrieved 2/4 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'Test that adding a Generic instance to itself raises TypeError.'
    var_1 = module_0.Generic()



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_add_provider_without_meta_name. Retrieved 3/6 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'customprovider'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_add_provider_without_meta_name. Retrieved 3/6 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'customprovider'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_getattr_returns_none_when_attribute_is_not_callable. Retrieved 1/2 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.test_attr
    assert var_1 is None



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_getattr_with_non_callable_attribute. Retrieved 1/2 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.test_attr
    assert var_1 is None



# Parsed testcases at query #34
#--------------------------




import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'generic'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(not var_2)
    assert var_3 is True



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_generic_init_no_baseprovider_subclass. Retrieved 3/6 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom'
    var_1 = module_0.Generic()
    var_2 = hasattr(var_1, var_0)
    var_3 = bool(not var_2)
    assert var_3 is True



