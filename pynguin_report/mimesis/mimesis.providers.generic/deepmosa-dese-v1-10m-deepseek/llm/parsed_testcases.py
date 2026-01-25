####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_generic_constructor_default_values. Retrieved 2/4 statements.
# Failed to parse test_generic_constructor_custom_locale.
# Partially parsed test_generic_constructor_providers_registered. Retrieved 5/7 statements.
# Partially parsed test_generic_constructor_providers_initialized. Retrieved 1/4 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.random

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Generic(seed=var_0)

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.__dir__()
    var_2 = len(var_1)
    var_3 = var_0.__dir__()
    var_4 = '_'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.__dir__()



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_predicate_at_line_16_evaluates_to_false. Retrieved 1/7 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_reseed_updates_provider_seeds. Retrieved 3/6 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.seed
    var_2 = 12345
    var_3 = var_0.reseed(var_2)

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 12345
    var_2 = var_0.reseed(var_1)

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.seed
    var_2 = var_0.reseed()



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_add_provider_valid_provider. Retrieved 5/8 statements.
# Partially parsed test_add_provider_invalid_provider_not_subclass. Retrieved 1/5 statements.
# Partially parsed test_add_provider_generic_to_itself. Retrieved 1/3 statements.
# Partially parsed test_add_provider_with_kwargs. Retrieved 7/13 statements.
# Partially parsed test_add_provider_seed_propagation. Retrieved 7/10 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'test_provider'
    var_1 = False
    var_2 = module_0.Generic()
    var_3 = 'test_provider'
    var_4 = hasattr(var_2, var_3)

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

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'test_provider'
    var_1 = False
    var_2 = module_0.Generic()
    var_3 = 'test_value'
    var_4 = 'test_provider'
    var_5 = getattr(var_2, var_4)
    var_6 = var_5.custom_arg
    assert var_6 == 'test_value'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'test_provider'
    var_1 = False
    var_2 = 42
    var_3 = module_0.Generic(seed=var_2)
    var_4 = 'test_provider'
    var_5 = getattr(var_3, var_4)
    var_6 = var_5.seed
    assert var_6 == 42



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_add_provider_with_meta_name. Retrieved 4/7 statements.
# Partially parsed test_add_provider_without_meta_name. Retrieved 3/6 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom_provider'
    var_1 = module_0.Generic()
    var_2 = 'custom_provider'
    var_3 = hasattr(var_1, var_2)

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'customprovider'
    var_2 = hasattr(var_0, var_1)



# Parsed testcases at query #6
#--------------------------




import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = '_test_provider'
    var_2 = 'test_value'
    var_3 = lambda locale, seed: var_2
    var_4 = setattr(var_0, var_1, var_3)
    var_5 = var_0.test_provider
    assert var_5 == 'test_value'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = '_test_provider'
    var_2 = 'non_callable_value'
    var_3 = setattr(var_0, var_1, var_2)
    var_4 = var_0.test_provider
    assert var_4 is None



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_reseed_updates_seed_for_all_providers. Retrieved 4/7 statements.
# Partially parsed test_reseed_propagates_to_nested_providers. Retrieved 3/5 statements.
# Partially parsed test_reseed_handles_non_provider_attributes_gracefully. Retrieved 3/4 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.seed
    var_2 = 12345
    var_3 = var_0.reseed(var_2)

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.seed
    var_2 = var_0.reseed()

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 98765
    var_2 = var_0.reseed(var_1)

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 42
    var_2 = var_0.reseed(var_1)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test___getattr___with_non_callable_attribute. Retrieved 3/4 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'test_attr'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_predicate_at_line_16_evaluates_to_false. Retrieved 1/7 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_reseed_updates_seed_for_all_providers. Retrieved 4/7 statements.
# Partially parsed test_reseed_with_missing_seed_uses_random_seed. Retrieved 3/6 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.seed
    var_2 = 12345
    var_3 = var_0.reseed(var_2)

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.seed
    var_2 = var_0.reseed()

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = '_test_provider'
    var_2 = None
    var_3 = setattr(var_0, var_1, var_2)
    var_4 = 12345
    var_5 = var_0.reseed(var_4)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_generic_constructor_default_locale_and_seed. Retrieved 2/4 statements.
# Failed to parse test_generic_constructor_custom_locale.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.random

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 12345
    var_1 = module_0.Generic(seed=var_0)

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'address'
    var_2 = hasattr(var_0, var_1)
    var_3 = 'person'
    var_4 = hasattr(var_0, var_3)
    var_5 = 'datetime'
    var_6 = hasattr(var_0, var_5)

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'generic'
    var_2 = hasattr(var_0, var_1)

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = '_random'
    var_2 = hasattr(var_0, var_1)
    var_3 = '_seed'
    var_4 = hasattr(var_0, var_3)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_generic_constructor_default_locale. Retrieved 2/4 statements.
# Failed to parse test_generic_constructor_custom_locale.
# Partially parsed test_generic_constructor_with_seed. Retrieved 3/5 statements.
# Partially parsed test_generic_constructor_without_seed. Retrieved 2/4 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.random

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 12345
    var_1 = module_0.Generic(seed=var_0)
    var_2 = var_1.random

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.random

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'person'
    var_2 = hasattr(var_0, var_1)
    var_3 = 'address'
    var_4 = hasattr(var_0, var_3)
    var_5 = 'datetime'
    var_6 = hasattr(var_0, var_5)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_add_provider_with_meta_name. Retrieved 4/7 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom_provider'
    var_1 = module_0.Generic()
    var_2 = 'custom_provider'
    var_3 = hasattr(var_1, var_2)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_generic_constructor_default_locale_and_seed. Retrieved 2/4 statements.
# Failed to parse test_generic_constructor_custom_locale.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.random

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 12345
    var_1 = module_0.Generic(seed=var_0)

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'address'
    var_2 = hasattr(var_0, var_1)
    var_3 = 'person'
    var_4 = hasattr(var_0, var_3)
    var_5 = 'datetime'
    var_6 = hasattr(var_0, var_5)

def test_case_0():
    pass



# Parsed testcases at query #15
#--------------------------




import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = module_0.Generic(var_0)

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 12345
    var_1 = module_0.Generic(seed=var_0)

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'fr'
    var_1 = 67890
    var_2 = module_0.Generic(var_0, var_1)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_add_provider_with_valid_provider. Retrieved 3/6 statements.
# Partially parsed test_add_provider_with_invalid_provider. Retrieved 1/3 statements.
# Partially parsed test_add_provider_with_generic_itself. Retrieved 1/3 statements.
# Partially parsed test_add_provider_with_non_provider_class. Retrieved 1/5 statements.
# Partially parsed test_add_provider_preserves_seed. Retrieved 5/8 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'customprovider'
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

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Generic(seed=var_0)
    var_2 = 'customprovider'
    var_3 = getattr(var_1, var_2)
    var_4 = var_3.seed



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_generic_constructor_custom_locale_and_seed. Retrieved 1/3 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()

def test_case_0():
    var_0 = 12345

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'address'
    var_2 = hasattr(var_0, var_1)
    var_3 = 'person'
    var_4 = hasattr(var_0, var_3)
    var_5 = 'datetime'
    var_6 = hasattr(var_0, var_5)

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'generic'
    var_2 = hasattr(var_0, var_1)

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'random'
    var_2 = hasattr(var_0, var_1)
    var_3 = 'reseed'
    var_4 = hasattr(var_0, var_3)
    var_5 = 'validate_enum'
    var_6 = hasattr(var_0, var_5)

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = '_locale'
    var_2 = hasattr(var_0, var_1)
    var_3 = var_0.__dir__()



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_add_provider_with_meta_name. Retrieved 4/7 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom_provider'
    var_1 = module_0.Generic()
    var_2 = 'custom_provider'
    var_3 = hasattr(var_1, var_2)



# Parsed testcases at query #19
#--------------------------

# Failed to parse test_provider_registry_does_not_contain_generic.




# Parsed testcases at query #20
#--------------------------

# Failed to parse test_provider_registry_contains_generic_class.




# Parsed testcases at query #21
#--------------------------

# Partially parsed test_add_provider_with_meta_name_defined. Retrieved 4/7 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom_provider'
    var_1 = module_0.Generic()
    var_2 = 'custom_provider'
    var_3 = hasattr(var_1, var_2)



# Parsed testcases at query #22
#--------------------------

# Failed to parse test_generic_constructor_custom_locale.
# Partially parsed test_generic_constructor_custom_locale_and_seed. Retrieved 1/3 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Generic(seed=var_0)

def test_case_0():
    var_0 = 123



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_provider_cls_is_generic_evaluates_to_true. Retrieved 1/6 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_generic_constructor_default_locale_and_seed. Retrieved 2/4 statements.
# Failed to parse test_generic_constructor_custom_locale.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.random

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 12345
    var_1 = module_0.Generic(seed=var_0)

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'address'
    var_2 = hasattr(var_0, var_1)
    var_3 = 'person'
    var_4 = hasattr(var_0, var_3)
    var_5 = 'datetime'
    var_6 = hasattr(var_0, var_5)

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'generic'
    var_2 = hasattr(var_0, var_1)

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'random'
    var_2 = hasattr(var_0, var_1)
    var_3 = 'seed'
    var_4 = hasattr(var_0, var_3)
    var_5 = 'reseed'
    var_6 = hasattr(var_0, var_5)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_generic_constructor_default_locale. Retrieved 2/4 statements.
# Failed to parse test_generic_constructor_custom_locale.
# Partially parsed test_generic_constructor_with_seed. Retrieved 3/5 statements.
# Partially parsed test_generic_constructor_without_seed. Retrieved 2/4 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.random

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Generic(seed=var_0)
    var_2 = var_1.random

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.random



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test___getattr__. Retrieved 4/5 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'test_value'
    var_2 = 'non_existent'
    var_3 = var_0.__getattr__(var_2)
    assert var_3 is None



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_add_provider. Retrieved 7/14 statements.
# Partially parsed test_add_provider_raises_type_error_for_non_base_provider_subclass. Retrieved 1/5 statements.
# Partially parsed test_add_provider_raises_type_error_for_generic_instance. Retrieved 1/3 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom'
    var_1 = False
    var_2 = module_0.Generic()
    var_3 = 'value'
    var_4 = 'custom'
    var_5 = hasattr(var_2, var_4)
    var_6 = var_2.custom

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



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_reseed_method_propagates_to_providers. Retrieved 4/7 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.seed
    var_2 = 12345
    var_3 = var_0.reseed(var_2)

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.seed
    var_2 = var_0.reseed()

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.seed
    var_2 = 12345
    var_3 = var_0.reseed(var_2)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_getattr_with_non_callable_attribute. Retrieved 3/4 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'test_attr'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None



# Parsed testcases at query #5
#--------------------------

# Partially parsed test___getattr__. Retrieved 3/4 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'test_value'
    var_2 = var_0.test_provider
    assert var_2 == 'test_value'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_getattr_with_non_callable_attribute. Retrieved 2/3 statements.
# Partially parsed test_getattr_with_empty_attribute. Retrieved 2/3 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.test_attr
    assert var_1 is None

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.nonexistent_attr
    assert var_1 is None

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.empty_attr
    assert var_1 is None



# Parsed testcases at query #7
#--------------------------

# Partially parsed test___getattr__predicate_evaluates_to_false. Retrieved 3/6 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'dummy'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_generic_constructor_default_locale. Retrieved 2/4 statements.
# Failed to parse test_generic_constructor_custom_locale.
# Partially parsed test_generic_constructor_custom_seed. Retrieved 3/5 statements.
# Partially parsed test_generic_constructor_custom_locale_and_seed. Retrieved 1/6 statements.


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
    var_0 = 123



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_getattr_predicate_evaluates_to_false. Retrieved 3/4 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'example'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_generic_constructor_default_locale_and_seed. Retrieved 2/4 statements.
# Failed to parse test_generic_constructor_custom_locale.
# Partially parsed test_generic_constructor_providers_initialization. Retrieved 3/5 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.random

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 12345
    var_1 = module_0.Generic(seed=var_0)

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.__dir__()
    var_2 = len(var_1)

def test_case_0():
    pass



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_reseed_handles_attribute_error. Retrieved 2/6 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.reseed()



# Parsed testcases at query #12
#--------------------------

# Failed to parse test_provider_registry_does_not_contain_generic.




# Parsed testcases at query #13
#--------------------------

# Partially parsed test_add_provider_with_non_generic_provider. Retrieved 3/6 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom'
    var_1 = False
    var_2 = module_0.Generic()



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_getattr_non_callable_attribute. Retrieved 3/4 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'non_callable'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_add_provider_not_generic_instance. Retrieved 3/6 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom'
    var_1 = False
    var_2 = module_0.Generic()



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_getattr_with_non_callable_attribute. Retrieved 3/4 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'test_attr'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None



# Parsed testcases at query #17
#--------------------------




import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.seed
    var_2 = 12345
    var_3 = var_0.reseed(var_2)

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.person
    var_2 = var_1.seed
    var_3 = 67890
    var_4 = var_0.reseed(var_3)

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.seed
    var_2 = var_0.reseed()



# Parsed testcases at query #18
#--------------------------

# Failed to parse test_generic_constructor_custom_locale.
# Partially parsed test_generic_constructor_custom_locale_and_seed. Retrieved 1/3 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Generic(seed=var_0)

def test_case_0():
    var_0 = 123



# Parsed testcases at query #19
#--------------------------

# Failed to parse test_provider_registry_contains_generic.




# Parsed testcases at query #20
#--------------------------

# Partially parsed test_add_provider_success. Retrieved 4/8 statements.
# Partially parsed test_add_provider_not_subclass_of_base_provider. Retrieved 1/5 statements.
# Partially parsed test_add_provider_generic_instance. Retrieved 1/3 statements.
# Partially parsed test_add_provider_with_kwargs. Retrieved 6/12 statements.
# Partially parsed test_add_provider_seed_overwrite. Retrieved 7/10 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom'
    var_1 = module_0.Generic()
    var_2 = 'custom'
    var_3 = getattr(var_1, var_2)

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
    var_0 = 'custom'
    var_1 = module_0.Generic()
    var_2 = 'test'
    var_3 = 'custom'
    var_4 = getattr(var_1, var_3)
    var_5 = var_4.custom_arg
    assert var_5 == 'test'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom'
    var_1 = 123
    var_2 = module_0.Generic(seed=var_1)
    var_3 = 456
    var_4 = 'custom'
    var_5 = getattr(var_2, var_4)
    var_6 = var_5.seed
    assert var_6 == 123



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_provider_cls_is_not_generic_or_subclass_of_base_provider. Retrieved 3/6 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom'
    var_1 = module_0.Generic()
    var_2 = hasattr(var_1, var_0)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_reseed_with_existing_provider. Retrieved 3/4 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 123
    var_2 = var_0.reseed(var_1)

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 123
    var_2 = var_0.reseed(var_1)



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_add_provider_without_attribute_error. Retrieved 4/7 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom'
    var_1 = module_0.Generic()
    var_2 = 'custom'
    var_3 = hasattr(var_1, var_2)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_add_provider_without_meta_name. Retrieved 3/6 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'customprovider'
    var_2 = hasattr(var_0, var_1)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_provider_cls_is_not_generic_and_not_subclass_of_base_provider. Retrieved 3/6 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'mock_provider'
    var_1 = module_0.Generic()
    var_2 = hasattr(var_1, var_0)



# Parsed testcases at query #26
#--------------------------

# Failed to parse test_provider_registry_contains_generic_class.




# Parsed testcases at query #27
#--------------------------

# Partially parsed test_add_provider_with_meta_name. Retrieved 4/7 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom'
    var_1 = module_0.Generic()
    var_2 = 'custom'
    var_3 = hasattr(var_1, var_2)



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_add_provider_with_meta_name. Retrieved 4/7 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom'
    var_1 = module_0.Generic()
    var_2 = 'custom'
    var_3 = hasattr(var_1, var_2)



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_provider_registry_contains_generic. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'generic'



# Parsed testcases at query #30
#--------------------------




import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_getattr_with_non_callable_attribute. Retrieved 3/6 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'mock_provider'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_reseed_handles_attribute_error. Retrieved 5/7 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = None
    var_2 = 'non_existent_provider'
    var_3 = [var_2]
    var_4 = var_0.reseed()



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_getattr_with_non_callable_attribute. Retrieved 3/4 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'test_attribute'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_add_provider_valid_provider. Retrieved 5/9 statements.
# Partially parsed test_add_provider_not_subclass_of_baseprovider. Retrieved 1/5 statements.
# Partially parsed test_add_provider_generic_instance. Retrieved 1/3 statements.
# Partially parsed test_add_provider_with_kwargs. Retrieved 3/9 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom'
    var_1 = module_0.Generic()
    var_2 = 'custom'
    var_3 = hasattr(var_1, var_2)
    var_4 = var_1.custom

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
    var_0 = 'custom'
    var_1 = module_0.Generic()
    var_2 = 'test'



# Parsed testcases at query #35
#--------------------------

# Failed to parse test_generic_constructor_custom_locale.
# Partially parsed test_generic_constructor_custom_locale_and_seed. Retrieved 1/3 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 12345
    var_1 = module_0.Generic(seed=var_0)

def test_case_0():
    var_0 = 67890

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.__dir__()



# Parsed testcases at query #36
#--------------------------




import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = None
    var_2 = lambda : var_1
    var_3 = var_0.add_provider(var_2)
    var_4 = 123
    var_5 = var_0.reseed(var_4)



# Parsed testcases at query #37
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
    var_1 = 'address'
    var_2 = var_0.__getattr__(var_1)

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'address'
    var_2 = var_0.__getattr__(var_1)
    var_3 = var_0.__getattr__(var_1)



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_ensure_predicate_at_line_16_evaluates_to_false. Retrieved 5/10 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'test_provider'
    var_1 = True
    var_2 = module_0.Generic()
    var_3 = 'test_provider'
    var_4 = getattr(var_2, var_3)



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_getattr_with_non_callable_attribute. Retrieved 2/3 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.test_attribute
    assert var_1 is None



# Parsed testcases at query #40
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
    var_1 = 'locale'
    var_2 = var_0.__getattr__(var_1)
    var_3 = callable(var_2)

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'locale'
    var_2 = var_0.__getattr__(var_1)



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_predicate_at_line_16_evaluates_to_false. Retrieved 6/12 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'mock_data_provider'
    var_1 = 'mock_base_provider'
    var_2 = module_0.Generic()
    var_3 = '_mock_base_provider'
    var_4 = hasattr(var_2, var_3)
    var_5 = hasattr(var_2, var_1)



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_add_provider_without_meta_name. Retrieved 3/6 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'customprovider'
    var_2 = hasattr(var_0, var_1)



# Parsed testcases at query #43
#--------------------------

# Failed to parse test_provider_registry_contains_generic.




# Parsed testcases at query #44
#--------------------------

# Partially parsed test_reseed_handles_attribute_error_correctly. Retrieved 2/3 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.reseed()



# Parsed testcases at query #45
#--------------------------

# Partially parsed test___getattr___with_non_callable_attribute. Retrieved 3/7 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'test'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_provider_cls_is_generic_evaluates_to_true. Retrieved 1/4 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_add_provider_with_valid_provider. Retrieved 5/8 statements.
# Partially parsed test_add_provider_with_non_baseprovider_subclass. Retrieved 1/5 statements.
# Partially parsed test_add_provider_with_generic_itself. Retrieved 1/3 statements.
# Partially parsed test_add_provider_preserves_seed. Retrieved 4/7 statements.
# Partially parsed test_add_provider_with_kwargs. Retrieved 4/10 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom'
    var_1 = False
    var_2 = module_0.Generic()
    var_3 = 'custom'
    var_4 = hasattr(var_2, var_3)

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
    var_0 = 'custom'
    var_1 = False
    var_2 = 12345
    var_3 = module_0.Generic(seed=var_2)

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom'
    var_1 = False
    var_2 = module_0.Generic()
    var_3 = 'test'



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_generic_provider_initialization_skips_base_data_provider. Retrieved 3/9 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'mock_provider'
    var_2 = hasattr(var_0, var_1)



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_reseed_handles_attribute_error. Retrieved 5/7 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = None
    var_2 = 'invalid_attr'
    var_3 = [var_2]
    var_4 = var_0.reseed



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_getattr_predicate_evaluates_to_false. Retrieved 3/4 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'test_attribute'
    var_2 = var_0.__getattr__(var_1)
    assert var_2 is None



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_getattr_with_non_callable_attribute. Retrieved 2/3 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.test_attr
    assert var_1 is None



