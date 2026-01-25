####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_provider_registry_initial_state. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'non_existent'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_validate_enum_with_none_item. Retrieved 8/10 statements.
# Partially parsed test_validate_enum_with_valid_item. Retrieved 7/10 statements.
# Partially parsed test_validate_enum_with_invalid_item. Retrieved 8/11 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = 'TestEnum'
    var_2 = 'A'
    var_3 = 'B'
    var_4 = 1
    var_5 = 2
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = None

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = 'TestEnum'
    var_2 = 'A'
    var_3 = 'B'
    var_4 = 1
    var_5 = 2
    var_6 = {var_2: var_4, var_3: var_5}

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = 'TestEnum'
    var_2 = 'A'
    var_3 = 'B'
    var_4 = 1
    var_5 = 2
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 'invalid'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_reseed_with_global_seed_set. Retrieved 2/4 statements.


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
    var_1 = var_0.reseed()

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.seed
    var_2 = 99
    var_3 = var_0.reseed(var_2)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_base_data_provider_constructor. Retrieved 4/6 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = 42
    var_2 = module_0.BaseDataProvider(var_0, var_1)
    var_3 = var_2.random



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_validate_enum_with_none_item. Retrieved 8/10 statements.
# Partially parsed test_validate_enum_with_valid_item. Retrieved 7/10 statements.
# Partially parsed test_validate_enum_with_invalid_item. Retrieved 8/11 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = 'TestEnum'
    var_2 = 'A'
    var_3 = 'B'
    var_4 = 1
    var_5 = 2
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = None

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = 'TestEnum'
    var_2 = 'A'
    var_3 = 'B'
    var_4 = 1
    var_5 = 2
    var_6 = {var_2: var_4, var_3: var_5}

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = 'TestEnum'
    var_2 = 'A'
    var_3 = 'B'
    var_4 = 1
    var_5 = 2
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 'invalid'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_baseprovider_constructor_with_seed. Retrieved 3/5 statements.
# Partially parsed test_baseprovider_constructor_without_seed_or_random. Retrieved 2/4 statements.


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

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.random



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_provider_registry_initialization. Retrieved 4/6 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.ProviderRegistry()
    var_1 = '_providers'
    var_2 = hasattr(var_0, var_1)
    var_3 = var_0._providers



# Parsed testcases at query #8
#--------------------------




import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseProvider(random=var_0)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'invalid_random'
    var_1 = module_0.BaseProvider(random=var_0)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_init_with_keyword_only_arguments. Retrieved 3/5 statements.
# Partially parsed test_init_without_arguments_raises_type_error. Retrieved 1/3 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.random

def test_case_0():
    var_0 = 42



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_validate_enum_with_none_item. Retrieved 8/10 statements.
# Partially parsed test_validate_enum_with_valid_item. Retrieved 7/10 statements.
# Partially parsed test_validate_enum_with_invalid_item. Retrieved 8/11 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = 'TestEnum'
    var_2 = 'A'
    var_3 = 'B'
    var_4 = 1
    var_5 = 2
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = None

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = 'TestEnum'
    var_2 = 'A'
    var_3 = 'B'
    var_4 = 1
    var_5 = 2
    var_6 = {var_2: var_4, var_3: var_5}

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = 'TestEnum'
    var_2 = 'A'
    var_3 = 'B'
    var_4 = 1
    var_5 = 2
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 'invalid'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_validate_enum_with_none_item. Retrieved 9/12 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = 'Enum'
    var_2 = ()
    var_3 = 'A'
    var_4 = 'B'
    var_5 = 1
    var_6 = 2
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = None



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_provider_registry_initialization. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'nonexistent'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_init_with_positional_args_fails. Retrieved 1/3 statements.


def test_case_0():
    var_0 = None



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_base_data_provider_constructor_defaults. Retrieved 2/4 statements.
# Partially parsed test_base_data_provider_constructor_with_locale. Retrieved 3/5 statements.
# Partially parsed test_base_data_provider_constructor_with_seed. Retrieved 3/5 statements.
# Partially parsed test_base_data_provider_constructor_with_locale_and_seed. Retrieved 4/6 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseDataProvider()
    var_1 = var_0.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'de'
    var_1 = module_0.BaseDataProvider(var_0)
    var_2 = var_1.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseDataProvider(seed=var_0)
    var_2 = var_1.random

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseDataProvider()

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'not_a_random_object'
    var_1 = module_0.BaseDataProvider()

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'es'
    var_1 = 100
    var_2 = module_0.BaseDataProvider(var_0, var_1)
    var_3 = var_2.random



# Parsed testcases at query #15
#--------------------------

# Failed to parse test_reseed_with_missing_seed.
# Partially parsed test_reseed_with_global_seed. Retrieved 3/6 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.reseed(var_0)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.reseed(var_0)

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseProvider(random=var_0)
    var_2 = 42
    var_3 = var_1.reseed(var_2)

def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = provider.random.getstate()[var_1][var_0]
    assert var_2 == 100



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_reseed_with_missing_seed_and_global_seed_not_missing. Retrieved 2/3 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.reseed()



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_base_data_provider_constructor_defaults. Retrieved 2/4 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseDataProvider()
    var_1 = var_0.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'de'
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
    var_0 = 'not a random object'
    var_1 = module_0.BaseDataProvider()



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_init_with_positional_args. Retrieved 2/3 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.BaseProvider(seed=var_0)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_validate_enum_with_none_item. Retrieved 9/11 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = 'Enum'
    var_2 = ()
    var_3 = 'A'
    var_4 = 'B'
    var_5 = 'a'
    var_6 = 'b'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = None



# Parsed testcases at query #20
#--------------------------

# Failed to parse test_init_docstring_exists.




# Parsed testcases at query #21
#--------------------------

# Failed to parse test_provider_registry_initialization.




# Parsed testcases at query #22
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)



# Parsed testcases at query #23
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_base_provider_constructor_initializes_random. Retrieved 2/4 statements.


import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseProvider(random=var_0)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'not a random object'
    var_1 = module_0.BaseProvider(random=var_0)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.random



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_base_data_provider_constructor_defaults. Retrieved 2/4 statements.
# Partially parsed test_base_data_provider_constructor_custom_locale. Retrieved 3/5 statements.
# Partially parsed test_base_data_provider_constructor_custom_seed. Retrieved 3/5 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseDataProvider()
    var_1 = var_0.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'de'
    var_1 = module_0.BaseDataProvider(var_0)
    var_2 = var_1.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseDataProvider(seed=var_0)
    var_2 = var_1.random

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = 100
    var_1 = module_0.Random(var_0)
    var_2 = module_1.BaseDataProvider()

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'not_a_random_object'
    var_1 = module_0.BaseDataProvider()



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_validate_enum_with_valid_item. Retrieved 3/8 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'value1'
    var_1 = 'value2'
    var_2 = module_0.BaseProvider()



# Parsed testcases at query #27
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = 42
    var_2 = module_0.BaseDataProvider(var_0, var_1)



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_init_ensures_dataset_is_initialized. Retrieved 4/5 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseDataProvider()
    var_1 = '_dataset'
    var_2 = hasattr(var_0, var_1)
    var_3 = var_0._dataset



# Parsed testcases at query #29
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_base_data_provider_constructor_defaults. Retrieved 2/4 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseDataProvider()
    var_1 = var_0.random



# Parsed testcases at query #31
#--------------------------

# Failed to parse test_provider_registry_initialization.




# Parsed testcases at query #32
#--------------------------

# Failed to parse test_provider_registry_initialization.




# Parsed testcases at query #33
#--------------------------

# Partially parsed test_base_data_provider_constructor_defaults. Retrieved 2/4 statements.
# Partially parsed test_base_data_provider_constructor_with_locale. Retrieved 3/5 statements.
# Partially parsed test_base_data_provider_constructor_with_seed. Retrieved 3/5 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseDataProvider()
    var_1 = var_0.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'de'
    var_1 = module_0.BaseDataProvider(var_0)
    var_2 = var_1.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseDataProvider(seed=var_0)
    var_2 = var_1.random

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseDataProvider()

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'not_a_random_instance'
    var_1 = module_0.BaseDataProvider()



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_provider_registry_initial_state. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'nonexistent'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_init_docstring_predicate. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'Initialize attributes for data providers.'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_base_provider_constructor_with_custom_random. Retrieved 2/3 statements.
# Partially parsed test_base_provider_constructor_with_seed. Retrieved 2/3 statements.
# Partially parsed test_base_provider_constructor_without_seed. Retrieved 1/2 statements.
# Partially parsed test_base_provider_constructor_with_both_seed_and_random. Retrieved 3/4 statements.


import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseProvider(random=var_0)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'not_a_random_instance'
    var_1 = module_0.BaseProvider(random=var_0)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 42
    var_2 = module_1.BaseProvider(seed=var_1, random=var_0)



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_base_provider_constructor_with_invalid_random. Retrieved 3/5 statements.
# Partially parsed test_base_provider_constructor_with_seed. Retrieved 2/3 statements.
# Partially parsed test_base_provider_constructor_without_seed. Retrieved 1/2 statements.
# Partially parsed test_base_provider_constructor_with_missing_seed_and_global_seed. Retrieved 1/4 statements.
# Partially parsed test_base_provider_constructor_initializes_random. Retrieved 2/4 statements.


import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseProvider(random=var_0)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'not_a_random'
    var_1 = module_0.BaseProvider(random=var_0)
    var_2 = str(var_0)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.random

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 42
    var_2 = module_1.BaseProvider(seed=var_1, random=var_0)



# Parsed testcases at query #38
#--------------------------

# Failed to parse test_init_docstring_is_not_empty.




# Parsed testcases at query #39
#--------------------------

# Partially parsed test_validate_enum_with_valid_item. Retrieved 4/11 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 42
    var_3 = module_0.BaseProvider(seed=var_2)



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_validate_enum_with_valid_item. Retrieved 3/8 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.BaseProvider()



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_init_with_keyword_only_arguments. Retrieved 3/5 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.random



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_provider_registry_initialization. Retrieved 6/7 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.ProviderRegistry()
    var_1 = '_providers'
    var_2 = hasattr(var_0, var_1)
    var_3 = var_0._providers
    var_4 = var_0._providers
    var_5 = len(var_4)
    assert var_5 == 0



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_init_docstring_predicate. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'Initialize attributes for data providers.'



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_reseed_with_missing_seed. Retrieved 1/2 statements.
# Partially parsed test_reseed_with_global_seed. Retrieved 4/6 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()

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
    var_3 = 0
    var_4 = 1
    var_5 = provider.random.getstate()[var_4][var_3]

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = 0
    var_2 = 1
    var_3 = provider.random.getstate()[var_2][var_1]
    assert var_3 == 100

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = 123
    var_2 = var_0.reseed(var_1)
    var_3 = 456
    var_4 = var_0.reseed(var_3)



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_base_data_provider_constructor_defaults. Retrieved 2/4 statements.
# Partially parsed test_base_data_provider_constructor_with_locale. Retrieved 4/7 statements.
# Partially parsed test_base_data_provider_constructor_with_seed. Retrieved 3/5 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseDataProvider()
    var_1 = var_0.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'de'
    var_1 = module_0.BaseDataProvider(var_0)
    var_2 = var_1._dataset
    var_3 = var_1.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseDataProvider(seed=var_0)
    var_2 = var_1.random

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseDataProvider()

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'not_a_random_object'
    var_1 = module_0.BaseDataProvider()



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_base_provider_constructor_with_seed. Retrieved 2/3 statements.
# Partially parsed test_base_provider_constructor_without_seed. Retrieved 1/2 statements.
# Partially parsed test_base_provider_constructor_with_missing_seed_and_global_seed. Retrieved 1/4 statements.


import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseProvider(random=var_0)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'invalid_random'
    var_1 = module_0.BaseProvider(random=var_0)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_base_data_provider_constructor_defaults. Retrieved 2/4 statements.
# Partially parsed test_base_data_provider_constructor_with_locale. Retrieved 3/5 statements.
# Partially parsed test_base_data_provider_constructor_with_seed. Retrieved 3/5 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseDataProvider()
    var_1 = var_0.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'de'
    var_1 = module_0.BaseDataProvider(var_0)
    var_2 = var_1.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseDataProvider(seed=var_0)
    var_2 = var_1.random

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseDataProvider()

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'not_a_random_object'
    var_1 = module_0.BaseDataProvider()



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_base_provider_constructor_with_seed. Retrieved 3/5 statements.
# Partially parsed test_base_provider_constructor_without_seed. Retrieved 2/4 statements.


import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseProvider(random=var_0)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'not_a_random_instance'
    var_1 = module_0.BaseProvider(random=var_0)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.random



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_base_data_provider_init_docstring. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'Initialize attributes for data providers.'



# Parsed testcases at query #50
#--------------------------

# Failed to parse test_init_docstring_is_not_empty.




# Parsed testcases at query #51
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseDataProvider()



# Parsed testcases at query #52
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)



# Parsed testcases at query #53
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_base_data_provider_constructor_defaults. Retrieved 2/5 statements.
# Partially parsed test_base_data_provider_constructor_with_locale. Retrieved 4/7 statements.
# Partially parsed test_base_data_provider_constructor_with_seed. Retrieved 2/3 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseDataProvider()
    var_1 = var_0.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'de'
    var_1 = module_0.BaseDataProvider(var_0)
    var_2 = var_1._dataset
    var_3 = var_1.random

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
    var_0 = 'not a random instance'
    var_1 = module_0.BaseDataProvider()



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_provider_registry_initialization. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'nonexistent'



# Parsed testcases at query #56
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #57
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseDataProvider()



# Parsed testcases at query #58
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_base_data_provider_init_docstring. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'Initialize attributes for data providers.'



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_base_data_provider_constructor_defaults. Retrieved 2/4 statements.
# Partially parsed test_base_data_provider_constructor_with_locale. Retrieved 3/5 statements.
# Partially parsed test_base_data_provider_constructor_with_seed. Retrieved 3/5 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseDataProvider()
    var_1 = var_0.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'de'
    var_1 = module_0.BaseDataProvider(var_0)
    var_2 = var_1.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseDataProvider(seed=var_0)
    var_2 = var_1.random

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseDataProvider()

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'not_a_random_instance'
    var_1 = module_0.BaseDataProvider()



# Parsed testcases at query #61
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseDataProvider()



# Parsed testcases at query #62
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_init_with_keyword_only_args. Retrieved 3/5 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.random



# Parsed testcases at query #64
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = 'seed'
    var_3 = hasattr(var_1, var_2)



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_base_data_provider_constructor_with_default_locale_and_seed. Retrieved 2/4 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseDataProvider()
    var_1 = var_0.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'de'
    var_1 = module_0.BaseDataProvider(var_0)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseDataProvider(seed=var_0)

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = 100
    var_1 = module_0.Random(var_0)
    var_2 = module_1.BaseDataProvider()

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'not_a_random_object'
    var_1 = module_0.BaseDataProvider()



# Parsed testcases at query #66
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = module_0.BaseDataProvider(var_0)



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_init_without_keyword_args. Retrieved 2/4 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.random



# Parsed testcases at query #68
#--------------------------

# Partially parsed test_base_provider_constructor_initializes_random. Retrieved 2/4 statements.


import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseProvider(random=var_0)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'invalid_random'
    var_1 = module_0.BaseProvider(random=var_0)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.random



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_keyword_only_arguments. Retrieved 2/4 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.random



