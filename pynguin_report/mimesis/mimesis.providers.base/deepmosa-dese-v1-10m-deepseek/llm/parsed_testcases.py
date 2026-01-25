####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.ProviderRegistry()



# Parsed testcases at query #2
#--------------------------

# Failed to parse test_BaseDataProvider_constructor_custom_locale.
# Partially parsed test_BaseDataProvider_constructor_custom_locale_and_seed. Retrieved 1/3 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseDataProvider()

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.BaseDataProvider(seed=var_0)

def test_case_0():
    var_0 = 456



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_validate_enum_with_none. Retrieved 4/6 statements.
# Partially parsed test_validate_enum_with_valid_enum_item. Retrieved 3/6 statements.
# Partially parsed test_validate_enum_with_invalid_item. Retrieved 4/7 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'A'
    var_1 = 'B'
    var_2 = module_0.BaseProvider()
    var_3 = None

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'A'
    var_1 = 'B'
    var_2 = module_0.BaseProvider()

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'A'
    var_1 = 'B'
    var_2 = module_0.BaseProvider()
    var_3 = 'C'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_validate_enum_with_item_not_in_enum. Retrieved 1/9 statements.


import builtins as module_0

def test_case_0():
    var_0 = module_0.object()



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_BaseProvider_initialization_with_default_seed. Retrieved 2/4 statements.
# Partially parsed test_BaseProvider_initialization_with_custom_seed. Retrieved 3/5 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.random

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseProvider(random=var_0)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'not_a_random_instance'
    var_1 = module_0.BaseProvider(random=var_0)



# Parsed testcases at query #6
#--------------------------

# Failed to parse test_BaseDataProvider_constructor_custom_locale.
# Failed to parse test_BaseDataProvider_constructor_locale_setup.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseDataProvider()

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 12345
    var_1 = module_0.BaseDataProvider(seed=var_0)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseDataProvider()

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseDataProvider()

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0.BaseDataProvider()

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseDataProvider()



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_validate_enum_with_valid_item. Retrieved 3/7 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.BaseProvider()



# Parsed testcases at query #8
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.ProviderRegistry()



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_init_with_random_none. Retrieved 3/5 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.BaseProvider(random=var_0)
    var_2 = var_1.random



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_reseed_with_missing_seed. Retrieved 2/3 statements.
# Partially parsed test_reseed_with_global_seed_set. Retrieved 2/4 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = 100
    var_3 = var_1.reseed(var_2)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = None
    var_3 = var_1.reseed(var_2)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_BaseProvider_initialization_with_default_values. Retrieved 2/4 statements.
# Partially parsed test_BaseProvider_initialization_with_custom_seed. Retrieved 3/5 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.random

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseProvider(random=var_0)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0.BaseProvider(random=var_0)



# Parsed testcases at query #12
#--------------------------




import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 42
    var_2 = module_1.BaseProvider(seed=var_1, random=var_0)



# Parsed testcases at query #13
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = 123
    var_2 = module_0.BaseDataProvider(var_0, var_1)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.BaseDataProvider(seed=var_0)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = module_0.BaseDataProvider(var_0)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'unsupported_locale'
    var_1 = module_0.BaseDataProvider(var_0)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_base_provider_initialization. Retrieved 3/4 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseProvider(random=var_0)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0.BaseProvider(random=var_0)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = 100
    var_3 = var_1.reseed(var_2)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.reseed()



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_reseed_when_seed_is_missing_seed_and_global_seed_is_set. Retrieved 2/7 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.BaseProvider(seed=var_0)



# Parsed testcases at query #16
#--------------------------




import builtins as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = module_1.BaseProvider(random=var_0)



# Parsed testcases at query #17
#--------------------------

# Failed to parse test_base_data_provider_constructor_custom_locale.
# Partially parsed test_base_data_provider_constructor_locale_and_seed. Retrieved 1/3 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseDataProvider()

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseDataProvider(seed=var_0)

def test_case_0():
    var_0 = 42

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'INVALID_LOCALE'
    var_1 = module_0.BaseDataProvider(var_0)



# Parsed testcases at query #18
#--------------------------

# Failed to parse test_BaseDataProvider_constructor_custom_locale.
# Partially parsed test_BaseDataProvider_constructor_dataset_initialization. Retrieved 2/3 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseDataProvider()

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 12345
    var_1 = module_0.BaseDataProvider(seed=var_0)

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseDataProvider()

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'not_a_random_instance'
    var_1 = module_0.BaseDataProvider()

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseDataProvider()
    var_1 = var_0._dataset



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_random_instance_validation. Retrieved 3/5 statements.
# Partially parsed test_default_random_instance. Retrieved 2/4 statements.


import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseProvider(random=var_0)
    var_2 = var_1.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'not_a_random_instance'
    var_1 = module_0.BaseProvider(random=var_0)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.random



# Parsed testcases at query #20
#--------------------------




import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 42
    var_2 = module_1.BaseProvider(seed=var_1, random=var_0)
    var_3 = module_1.BaseProvider(random=var_0)
    var_4 = module_1.BaseProvider(seed=var_1)
    var_5 = module_1.BaseProvider()



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_constructor_does_not_modify_class_level_providers. Retrieved 1/2 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.ProviderRegistry()

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.ProviderRegistry()



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_predicate_at_line_11_evaluates_to_true. Retrieved 2/4 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseDataProvider()
    var_1 = var_0.seed



# Parsed testcases at query #23
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = module_0.BaseDataProvider(var_0)



# Parsed testcases at query #24
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseDataProvider()



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_base_data_provider_init_with_default_locale. Retrieved 2/3 statements.
# Failed to parse test_base_data_provider_init_with_custom_locale.
# Partially parsed test_base_data_provider_init_with_seed. Retrieved 3/4 statements.
# Partially parsed test_base_data_provider_init_with_custom_random. Retrieved 3/4 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseDataProvider()
    var_1 = var_0._dataset

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseDataProvider(seed=var_0)
    var_2 = var_1._dataset

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseDataProvider()
    var_2 = var_1._dataset

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0.BaseDataProvider()

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'invalid_locale'
    var_1 = module_0.BaseDataProvider(var_0)



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_constructor_initializes_empty_providers. Retrieved 1/2 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.ProviderRegistry()



# Parsed testcases at query #27
#--------------------------

# Failed to parse test_BaseDataProvider_constructor_custom_locale.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseDataProvider()

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseDataProvider()

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseDataProvider(seed=var_0)

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseDataProvider()

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0.BaseDataProvider()

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseDataProvider()



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_provider_registry_constructor. Retrieved 2/4 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.ProviderRegistry()
    var_1 = 'non_existent_provider'



# Parsed testcases at query #29
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.ProviderRegistry()



# Parsed testcases at query #30
#--------------------------

# Failed to parse test_BaseDataProvider_constructor_custom_locale.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseDataProvider()

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 12345
    var_1 = module_0.BaseDataProvider(seed=var_0)

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseDataProvider()

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'not_a_random_instance'
    var_1 = module_0.BaseDataProvider()

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseDataProvider()



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_base_data_provider_init_with_default_locale. Retrieved 2/3 statements.
# Partially parsed test_base_data_provider_init_with_custom_locale. Retrieved 3/4 statements.
# Partially parsed test_base_data_provider_init_with_seed. Retrieved 3/4 statements.
# Partially parsed test_base_data_provider_init_with_custom_locale_and_seed. Retrieved 4/5 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseDataProvider()
    var_1 = var_0._dataset

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = module_0.BaseDataProvider(var_0)
    var_2 = var_1._dataset

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseDataProvider(seed=var_0)
    var_2 = var_1._dataset

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'fr'
    var_1 = 123
    var_2 = module_0.BaseDataProvider(var_0, var_1)
    var_3 = var_2._dataset

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0.BaseDataProvider(var_0)



# Parsed testcases at query #32
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.ProviderRegistry()



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_predicate_at_line_11_evaluates_to_true. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'test_provider'



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_predicate_at_line_21_evaluates_to_false. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'test_provider'
    var_1 = False



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_initialization_with_default_seed. Retrieved 2/4 statements.
# Partially parsed test_initialization_with_custom_seed. Retrieved 3/5 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.random

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseProvider(random=var_0)

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 123
    var_2 = module_1.BaseProvider(seed=var_1, random=var_0)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0.BaseProvider(random=var_0)



# Parsed testcases at query #36
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseDataProvider()
    var_1 = var_0.Meta
    var_2 = 'name'
    var_3 = hasattr(var_1, var_2)



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_init_without_auto_register. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = False
    var_2 = 'test'



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_init_with_keyword_only_arguments. Retrieved 2/8 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Random()



# Parsed testcases at query #39
#--------------------------

# Failed to parse test_BaseDataProvider_initialization_with_custom_locale.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseDataProvider()

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 12345
    var_1 = module_0.BaseDataProvider(seed=var_0)

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseDataProvider()

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'invalid_random'
    var_1 = module_0.BaseDataProvider()



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_reseed_with_global_seed_not_missing. Retrieved 3/4 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.reseed()



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_validate_enum_with_valid_enum_item. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_ProviderRegistry_constructor. Retrieved 1/2 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.ProviderRegistry()



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_predicate_at_line_8_evaluates_to_true. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'test_provider'
    var_1 = True



# Parsed testcases at query #44
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)



# Parsed testcases at query #45
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseDataProvider()
    var_1 = 'Meta'
    var_2 = hasattr(var_0, var_1)
    var_3 = var_0.Meta
    var_4 = 'name'
    var_5 = hasattr(var_3, var_4)



# Parsed testcases at query #46
#--------------------------

# Failed to parse test_BaseDataProvider_initialization_with_custom_locale.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseDataProvider()

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseDataProvider(seed=var_0)

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseDataProvider()

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0.BaseDataProvider()
    var_2 = 'Expected TypeError for invalid random instance'
    var_3 = AssertionError(var_2)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseDataProvider()



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_init_with_keyword_only_arguments. Retrieved 2/5 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Random()



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_init_subclass_auto_register_true. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 'name'
    var_1 = 'auto_register'
    var_2 = True



# Parsed testcases at query #49
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.ProviderRegistry()



# Parsed testcases at query #50
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.ProviderRegistry()



# Parsed testcases at query #51
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.BaseProvider(seed=var_0)



# Parsed testcases at query #52
#--------------------------

# Failed to parse test_BaseDataProvider_constructor_custom_locale.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseDataProvider()

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 12345
    var_1 = module_0.BaseDataProvider(seed=var_0)

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseDataProvider()

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'invalid_random'
    var_1 = module_0.BaseDataProvider()



# Parsed testcases at query #53
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.reseed()



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_init_without_keyword_arguments_raises_error. Retrieved 2/4 statements.


import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 42
    var_2 = module_1.BaseProvider(seed=var_1, random=var_0)

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 42



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_validate_enum_with_none. Retrieved 2/3 statements.
# Partially parsed test_validate_enum_with_valid_enum_item. Retrieved 1/3 statements.
# Partially parsed test_validate_enum_with_invalid_enum_item. Retrieved 2/4 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = None

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = 'D'



# Parsed testcases at query #56
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.ProviderRegistry()



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_validate_enum_with_none. Retrieved 5/7 statements.
# Partially parsed test_validate_enum_with_valid_enum_item. Retrieved 4/7 statements.
# Partially parsed test_validate_enum_with_invalid_enum_item. Retrieved 5/10 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.BaseProvider()
    var_4 = None

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.BaseProvider()

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = module_0.BaseProvider()



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_init_with_missing_seed_and_no_random. Retrieved 2/4 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseDataProvider()
    var_1 = var_0.random



# Parsed testcases at query #59
#--------------------------

# Failed to parse test_BaseDataProvider_constructor_custom_locale.
# Partially parsed test_BaseDataProvider_constructor_with_args. Retrieved 1/3 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseDataProvider()

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseDataProvider(seed=var_0)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.BaseDataProvider(seed=var_0)

def test_case_0():
    var_0 = 42



# Parsed testcases at query #60
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.ProviderRegistry()



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_validate_enum_with_non_enum_item. Retrieved 4/8 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.BaseProvider()
    var_3 = 'not_an_enum_value'



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_constructor_with_seed. Retrieved 3/5 statements.
# Partially parsed test_constructor_with_default_values. Retrieved 2/4 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.random

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseProvider(random=var_0)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'not_a_random_instance'
    var_1 = module_0.BaseProvider(random=var_0)



# Parsed testcases at query #63
#--------------------------

# Failed to parse test_random_parameter_must_be_instance_of_random_class.




# Parsed testcases at query #64
#--------------------------

# Partially parsed test_BaseProvider_init_with_default_seed. Retrieved 2/4 statements.
# Partially parsed test_BaseProvider_init_with_custom_seed. Retrieved 3/5 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.random

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseProvider(random=var_0)

import builtins as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = module_1.BaseProvider(random=var_0)



# Parsed testcases at query #65
#--------------------------

# Failed to parse test_constructor_with_missing_seed.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseDataProvider()

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'fr'
    var_1 = module_0.BaseDataProvider(var_0)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseDataProvider(seed=var_0)

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseDataProvider()

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.BaseDataProvider()



# Parsed testcases at query #66
#--------------------------

# Partially parsed test_validate_enum_with_invalid_item. Retrieved 4/7 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.BaseProvider()
    var_3 = 3



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_hasattr_Meta_name_and_auto_register. Retrieved 5/13 statements.


def test_case_0():
    var_0 = 'test_provider'
    var_1 = True
    var_2 = 'Meta'
    var_3 = 'name'
    var_4 = 'auto_register'



# Parsed testcases at query #68
#--------------------------




import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseProvider(random=var_0)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_init_without_datafile. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'test_provider'
    var_1 = False



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_init_without_keyword_args_raises_error. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'seed_value'



# Parsed testcases at query #71
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseDataProvider()



# Parsed testcases at query #72
#--------------------------

# Partially parsed test_BaseDataProvider_initialization. Retrieved 2/3 statements.
# Failed to parse test_BaseDataProvider_initialization_with_custom_locale.
# Partially parsed test_BaseDataProvider_initialization_with_custom_seed. Retrieved 3/4 statements.
# Partially parsed test_BaseDataProvider_initialization_with_custom_locale_and_seed. Retrieved 1/5 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseDataProvider()
    var_1 = var_0._dataset

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseDataProvider(seed=var_0)
    var_2 = var_1._dataset

def test_case_0():
    var_0 = 42



# Parsed testcases at query #73
#--------------------------




import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 42
    var_2 = module_1.BaseProvider(seed=var_1, random=var_0)



# Parsed testcases at query #74
#--------------------------

# Partially parsed test_init_with_locale_and_seed. Retrieved 4/5 statements.
# Partially parsed test_init_with_default_locale_and_seed. Retrieved 3/4 statements.
# Partially parsed test_init_with_locale_and_missing_seed. Retrieved 3/4 statements.
# Partially parsed test_init_with_default_locale_and_missing_seed. Retrieved 2/3 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = 42
    var_2 = module_0.BaseDataProvider(var_0, var_1)
    var_3 = var_2._dataset

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseDataProvider(seed=var_0)
    var_2 = var_1._dataset

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = module_0.BaseDataProvider(var_0)
    var_2 = var_1._dataset

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseDataProvider()
    var_1 = var_0._dataset



# Parsed testcases at query #75
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseDataProvider()



# Parsed testcases at query #76
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseDataProvider()



# Parsed testcases at query #77
#--------------------------

# Partially parsed test_init_subclass_with_valid_meta. Retrieved 6/13 statements.


def test_case_0():
    var_0 = 'test_provider'
    var_1 = True
    var_2 = 'Meta'
    var_3 = 'name'
    var_4 = 'auto_register'
    var_5 = True



# Parsed testcases at query #78
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseDataProvider()



# Parsed testcases at query #79
#--------------------------

# Partially parsed test_init_without_keyword_only_arguments. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 123
    var_1 = None



# Parsed testcases at query #80
#--------------------------

# Partially parsed test_init_with_keyword_only_arguments. Retrieved 3/17 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Random()
    var_2 = module_0.Random()



# Parsed testcases at query #81
#--------------------------

# Partially parsed test_hasattr_Meta_and_name. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'test_provider'
    var_1 = 'Meta'
    var_2 = 'name'



# Parsed testcases at query #82
#--------------------------

# Partially parsed test_init_with_default_seed. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = False



# Parsed testcases at query #83
#--------------------------

# Partially parsed test_has_meta_name_attribute. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'test_provider'
    var_1 = 'Meta'
    var_2 = 'name'



# Parsed testcases at query #84
#--------------------------

# Partially parsed test_init_with_default_random. Retrieved 2/4 statements.


import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseProvider(random=var_0)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.random



# Parsed testcases at query #85
#--------------------------

# Partially parsed test_base_provider_initialization. Retrieved 3/5 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseProvider(random=var_0)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0.BaseProvider(random=var_0)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = 100
    var_3 = var_1.reseed(var_2)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.reseed()

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = str(var_0)
    assert var_1 == 'BaseProvider'



# Parsed testcases at query #86
#--------------------------

# Failed to parse test_hasattr_cls_Meta_and_cls_Meta_name_evaluates_to_false.




# Parsed testcases at query #87
#--------------------------

# Partially parsed test_base_data_provider_init_with_default_locale. Retrieved 2/3 statements.
# Partially parsed test_base_data_provider_init_with_custom_locale. Retrieved 3/4 statements.
# Partially parsed test_base_data_provider_init_with_custom_seed. Retrieved 3/4 statements.
# Partially parsed test_base_data_provider_init_with_custom_locale_and_seed. Retrieved 4/5 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseDataProvider()
    var_1 = var_0._dataset

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = module_0.BaseDataProvider(var_0)
    var_2 = var_1._dataset

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseDataProvider(seed=var_0)
    var_2 = var_1._dataset

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = 42
    var_2 = module_0.BaseDataProvider(var_0, var_1)
    var_3 = var_2._dataset

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0.BaseDataProvider(var_0)



# Parsed testcases at query #88
#--------------------------

# Partially parsed test_init_without_auto_register. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = False
    var_2 = 'test'



# Parsed testcases at query #89
#--------------------------




import mimesis.providers.base as module_0
import mimesis.random as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.BaseProvider(seed=var_0, random=var_0)
    var_2 = 123
    var_3 = module_1.Random()
    var_4 = module_0.BaseProvider(seed=var_2, random=var_3)



# Parsed testcases at query #90
#--------------------------

# Partially parsed test_BaseDataProvider_constructor. Retrieved 4/6 statements.
# Partially parsed test_BaseDataProvider_constructor_default_locale. Retrieved 2/4 statements.
# Partially parsed test_BaseDataProvider_constructor_default_seed. Retrieved 3/5 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = 42
    var_2 = module_0.BaseDataProvider(var_0, var_1)
    var_3 = var_2.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseDataProvider()
    var_1 = var_0.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = module_0.BaseDataProvider(var_0)
    var_2 = var_1.random

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = 'en'
    var_1 = 42
    var_2 = module_0.Random()
    var_3 = module_1.BaseDataProvider(var_0, var_1)



# Parsed testcases at query #91
#--------------------------

# Partially parsed test_validate_enum_with_none_item. Retrieved 5/7 statements.
# Partially parsed test_validate_enum_with_valid_enum_item. Retrieved 3/6 statements.
# Partially parsed test_validate_enum_raises_non_enumerable_error. Retrieved 4/7 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'A'
    var_1 = 'B'
    var_2 = 'C'
    var_3 = module_0.BaseProvider()
    var_4 = None

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'A'
    var_1 = 'B'
    var_2 = module_0.BaseProvider()

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'A'
    var_1 = 'B'
    var_2 = module_0.BaseProvider()
    var_3 = 'invalid'



# Parsed testcases at query #92
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.ProviderRegistry()



# Parsed testcases at query #93
#--------------------------

# Partially parsed test_validate_enum_with_none. Retrieved 4/6 statements.
# Partially parsed test_validate_enum_with_valid_enum_item. Retrieved 3/6 statements.
# Partially parsed test_validate_enum_with_invalid_enum_item. Retrieved 4/7 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = module_0.BaseProvider()
    var_3 = None

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = module_0.BaseProvider()

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = module_0.BaseProvider()
    var_3 = 'invalid'



# Parsed testcases at query #94
#--------------------------

# Failed to parse test_init_with_non_keyword_arguments.




# Parsed testcases at query #95
#--------------------------

# Partially parsed test_base_data_provider_constructor_default_locale. Retrieved 2/3 statements.
# Partially parsed test_base_data_provider_constructor_custom_locale. Retrieved 3/4 statements.
# Partially parsed test_base_data_provider_constructor_with_seed. Retrieved 3/4 statements.
# Partially parsed test_base_data_provider_constructor_with_custom_random. Retrieved 3/4 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseDataProvider()
    var_1 = var_0._dataset

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = module_0.BaseDataProvider(var_0)
    var_2 = var_1._dataset

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseDataProvider(seed=var_0)
    var_2 = var_1._dataset

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseDataProvider()
    var_2 = var_1._dataset

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0.BaseDataProvider()



# Parsed testcases at query #96
#--------------------------

# Partially parsed test_reseed_with_global_seed_set. Retrieved 2/4 statements.
# Partially parsed test_reseed_does_not_use_global_seed_when_local_seed_provided. Retrieved 3/5 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.reseed()

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = None
    var_2 = var_0.reseed(var_1)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = 42
    var_2 = var_0.reseed(var_1)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = 123
    var_2 = var_0.reseed(var_1)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.reseed()

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = 789
    var_2 = var_0.reseed(var_1)



# Parsed testcases at query #97
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.ProviderRegistry()



# Parsed testcases at query #98
#--------------------------

# Partially parsed test_BaseDataProvider_initialization. Retrieved 4/5 statements.
# Partially parsed test_BaseDataProvider_initialization_default_locale. Retrieved 3/4 statements.
# Partially parsed test_BaseDataProvider_initialization_default_seed. Retrieved 3/4 statements.
# Partially parsed test_BaseDataProvider_initialization_default_values. Retrieved 2/3 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = 123
    var_2 = module_0.BaseDataProvider(var_0, var_1)
    var_3 = var_2._dataset

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.BaseDataProvider(seed=var_0)
    var_2 = var_1._dataset

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = module_0.BaseDataProvider(var_0)
    var_2 = var_1._dataset

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseDataProvider()
    var_1 = var_0._dataset



# Parsed testcases at query #99
#--------------------------

# Partially parsed test_constructor_with_default_seed_and_random. Retrieved 2/4 statements.
# Partially parsed test_constructor_with_custom_seed. Retrieved 3/5 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.random

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseProvider(random=var_0)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'invalid_random'
    var_1 = module_0.BaseProvider(random=var_0)



# Parsed testcases at query #100
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.BaseProvider(seed=var_0, random=var_0)



# Parsed testcases at query #101
#--------------------------

# Partially parsed test_initialize_with_keyword_only_arguments. Retrieved 2/8 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Random()



# Parsed testcases at query #102
#--------------------------

# Partially parsed test_random_instance_validation. Retrieved 4/11 statements.


import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseProvider(random=var_0)
    var_2 = module_1.BaseProvider()
    var_3 = var_2.random



# Parsed testcases at query #103
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.ProviderRegistry()



# Parsed testcases at query #104
#--------------------------

# Partially parsed test_base_data_provider_constructor_default_locale. Retrieved 2/3 statements.
# Failed to parse test_base_data_provider_constructor_custom_locale.
# Partially parsed test_base_data_provider_constructor_with_seed. Retrieved 3/4 statements.
# Partially parsed test_base_data_provider_constructor_with_custom_random. Retrieved 3/4 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseDataProvider()
    var_1 = var_0._dataset

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseDataProvider(seed=var_0)
    var_2 = var_1._dataset

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseDataProvider()
    var_2 = var_1._dataset

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0.BaseDataProvider()



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_constructor_with_default_values. Retrieved 2/4 statements.
# Partially parsed test_constructor_with_seed. Retrieved 3/5 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.random

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseProvider(random=var_0)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0.BaseProvider(random=var_0)

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 42
    var_2 = module_1.BaseProvider(seed=var_1, random=var_0)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_provider_registry_initialization. Retrieved 4/5 statements.
# Partially parsed test_provider_registry_register. Retrieved 1/6 statements.
# Partially parsed test_provider_registry_get_all. Retrieved 1/6 statements.
# Partially parsed test_provider_registry_get_existing. Retrieved 1/5 statements.
# Partially parsed test_provider_registry_get_nonexistent. Retrieved 1/2 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.ProviderRegistry()
    var_1 = var_0._providers
    var_2 = var_0._providers
    var_3 = len(var_2)
    assert var_3 == 0

def test_case_0():
    var_0 = 'test'

def test_case_0():
    var_0 = 'test'

def test_case_0():
    var_0 = 'test'

def test_case_0():
    var_0 = 'nonexistent'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_init_with_none_random. Retrieved 3/5 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.BaseProvider(random=var_0)
    var_2 = var_1.random



# Parsed testcases at query #4
#--------------------------




import mimesis.providers.base as module_0
import mimesis.random as module_1

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = module_1.Random()
    var_3 = module_0.BaseProvider(random=var_2)



# Parsed testcases at query #5
#--------------------------




import mimesis.providers.base as module_0
import mimesis.random as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.BaseProvider(seed=var_0, random=var_0)
    var_2 = 42
    var_3 = module_1.Random()
    var_4 = module_0.BaseProvider(seed=var_2, random=var_3)



# Parsed testcases at query #6
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = 100
    var_3 = var_1.reseed(var_2)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.reseed()

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = 50
    var_2 = var_0.reseed(var_1)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = None
    var_3 = var_1.reseed(var_2)



# Parsed testcases at query #7
#--------------------------

# Failed to parse test_BaseDataProvider_constructor_custom_locale.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseDataProvider()

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.BaseDataProvider(seed=var_0)

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseDataProvider()



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_base_data_provider_init_with_default_locale. Retrieved 2/3 statements.
# Failed to parse test_base_data_provider_init_with_custom_locale.
# Partially parsed test_base_data_provider_init_with_seed. Retrieved 3/4 statements.
# Partially parsed test_base_data_provider_init_with_custom_locale_and_seed. Retrieved 1/5 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseDataProvider()
    var_1 = var_0._dataset

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseDataProvider(seed=var_0)
    var_2 = var_1._dataset

def test_case_0():
    var_0 = 123

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'unsupported_locale'
    var_1 = module_0.BaseDataProvider(var_0)



# Parsed testcases at query #9
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseDataProvider()



# Parsed testcases at query #10
#--------------------------

# Failed to parse test_base_data_provider_init_custom_locale.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseDataProvider()

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseDataProvider()

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 12345
    var_1 = module_0.BaseDataProvider(seed=var_0)

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseDataProvider()

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'invalid_random'
    var_1 = module_0.BaseDataProvider()



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_validate_enum_with_none. Retrieved 4/6 statements.
# Partially parsed test_validate_enum_with_valid_enum. Retrieved 3/6 statements.
# Partially parsed test_validate_enum_with_invalid_enum. Retrieved 4/7 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = module_0.BaseProvider()
    var_3 = None

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = module_0.BaseProvider()

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = module_0.BaseProvider()
    var_3 = 'c'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_reseed_with_global_seed. Retrieved 2/3 statements.
# Partially parsed test_reseed_with_missing_seed. Retrieved 1/2 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.random.seed_value
    var_2 = var_0.reseed()

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = 12345
    var_2 = var_0.reseed(var_1)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.reseed()

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = None
    var_2 = var_0.reseed(var_1)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = 12345
    var_2 = var_0.reseed(var_1)
    var_3 = module_0.BaseProvider()
    var_4 = var_3.reseed(var_1)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = 12345
    var_2 = var_0.reseed(var_1)
    var_3 = module_0.BaseProvider()
    var_4 = 67890
    var_5 = var_3.reseed(var_4)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_reseed_with_global_seed. Retrieved 2/3 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.reseed()



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_provider_registry_constructor. Retrieved 1/2 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.ProviderRegistry()



# Parsed testcases at query #15
#--------------------------

# Failed to parse test_BaseDataProvider_constructor_with_custom_locale.
# Partially parsed test_BaseDataProvider_constructor_with_custom_locale_and_seed. Retrieved 1/4 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseDataProvider()
    var_1 = var_0.get_current_locale()

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseDataProvider(seed=var_0)

def test_case_0():
    var_0 = 42



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_validate_enum_with_valid_item. Retrieved 3/7 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.BaseProvider()



# Parsed testcases at query #17
#--------------------------

# Failed to parse test_BaseDataProvider_initialization_with_custom_locale.
# Partially parsed test_BaseDataProvider_initialization_with_all_parameters. Retrieved 2/4 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseDataProvider()

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseDataProvider(seed=var_0)

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseDataProvider()

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 42



# Parsed testcases at query #18
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseProvider(random=var_0)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_base_data_provider_init_with_default_locale. Retrieved 2/3 statements.
# Failed to parse test_base_data_provider_init_with_custom_locale.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseDataProvider()
    var_1 = var_0._dataset

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseDataProvider(seed=var_0)

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseDataProvider()

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0.BaseDataProvider()

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'xx'
    var_1 = module_0.BaseDataProvider(var_0)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_constructor_with_default_parameters. Retrieved 2/4 statements.
# Partially parsed test_constructor_with_seed. Retrieved 3/5 statements.
# Partially parsed test_constructor_with_none_seed. Retrieved 3/5 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.random

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseProvider(random=var_0)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'invalid_random'
    var_1 = module_0.BaseProvider(random=var_0)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_constructor_with_default_values. Retrieved 2/4 statements.
# Partially parsed test_constructor_with_custom_seed. Retrieved 3/5 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.random

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseProvider(random=var_0)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 12345
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'not_a_random_instance'
    var_1 = module_0.BaseProvider(random=var_0)



# Parsed testcases at query #22
#--------------------------

# Failed to parse test_BaseDataProvider_constructor_custom_locale.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseDataProvider()

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseDataProvider(seed=var_0)

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseDataProvider()

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0.BaseDataProvider()



# Parsed testcases at query #23
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseProvider(random=var_0)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.BaseProvider(seed=var_0)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0.BaseProvider(random=var_0)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = 100
    var_3 = var_1.reseed(var_2)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = None
    var_3 = var_1.reseed(var_2)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = str(var_0)
    assert var_1 == 'BaseProvider'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_ProviderRegistry_initial_state. Retrieved 2/4 statements.
# Partially parsed test_ProviderRegistry_register_and_retrieve. Retrieved 1/6 statements.
# Partially parsed test_ProviderRegistry_register_multiple. Retrieved 2/11 statements.
# Partially parsed test_ProviderRegistry_overwrite_registration. Retrieved 1/8 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.ProviderRegistry()
    var_1 = 'nonexistent'

def test_case_0():
    var_0 = 'test_provider'

def test_case_0():
    var_0 = 'provider1'
    var_1 = 'provider2'

def test_case_0():
    var_0 = 'test'



