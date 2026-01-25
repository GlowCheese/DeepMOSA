####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_validate_enum_with_none_item. Retrieved 5/7 statements.
# Partially parsed test_validate_enum_with_valid_item. Retrieved 4/7 statements.
# Partially parsed test_validate_enum_with_invalid_item. Retrieved 5/8 statements.


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
    var_2 = 'C'
    var_3 = module_0.BaseProvider()

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'A'
    var_1 = 'B'
    var_2 = 'C'
    var_3 = module_0.BaseProvider()
    var_4 = 'D'
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #2
#--------------------------

# Failed to parse test_constructor_initializes_empty_providers_dict.
# Failed to parse test_constructor_does_not_modify_class_level_providers.




# Parsed testcases at query #3
#--------------------------

# Failed to parse test_provider_registry_constructor.




# Parsed testcases at query #4
#--------------------------

# Partially parsed test_base_provider_initialization_with_seed. Retrieved 3/5 statements.
# Partially parsed test_base_provider_initialization_with_default_values. Retrieved 2/4 statements.


import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseProvider(random=var_0)
    var_2 = var_1.random
    var_3 = bool(var_1.random == var_0)
    assert var_3 is True
    var_4 = var_1.seed

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed
    var_3 = bool(var_1.seed == var_0)
    assert var_3 is True
    var_4 = var_1.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.random
    var_2 = var_0.seed

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0.BaseProvider(random=var_0)
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #5
#--------------------------

# Failed to parse test_init_with_non_random_instance.




# Parsed testcases at query #6
#--------------------------

# Failed to parse test_init_with_non_random_instance.




# Parsed testcases at query #7
#--------------------------

# Partially parsed test_base_data_provider_initialization_with_default_locale. Retrieved 1/3 statements.
# Partially parsed test_base_data_provider_initialization_with_missing_seed. Retrieved 1/2 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = 42
    var_2 = {}
    var_3 = module_0.BaseDataProvider(var_0, var_1, **var_2)
    var_4 = var_3.locale
    assert var_4 == 'en'
    var_5 = var_3.seed
    assert var_5 == 42
    var_6 = var_3._dataset
    var_7 = bool(var_3._dataset == {})
    assert var_7 is True

def test_case_0():
    var_0 = 42

def test_case_0():
    var_0 = 'en'

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = None
    var_2 = {}
    var_3 = module_0.BaseDataProvider(var_0, var_1, **var_2)
    var_4 = var_3.locale
    assert var_4 == 'en'
    var_5 = var_3.seed
    assert var_5 is None
    var_6 = var_3._dataset
    var_7 = bool(var_3._dataset == {})
    assert var_7 is True

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = 42
    var_2 = 'value'
    var_3 = 'custom_arg'
    var_4 = 'custom_kwarg'
    var_5 = {var_3: var_2, var_4: var_2}
    var_6 = module_0.BaseDataProvider(var_0, var_1, **var_5)
    var_7 = var_6.locale
    assert var_7 == 'en'
    var_8 = var_6.seed
    assert var_8 == 42
    var_9 = var_6._dataset
    var_10 = bool(var_6._dataset == {})
    assert var_10 is True



# Parsed testcases at query #8
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 42



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_reseed_with_global_seed. Retrieved 3/4 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = 123
    var_3 = var_1.reseed(var_2)
    var_4 = var_1.seed
    assert var_4 == 123

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.reseed()
    var_3 = var_1.seed

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.reseed()
    var_3 = var_1.seed

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = None
    var_3 = var_1.reseed(var_2)
    var_4 = var_1.seed
    assert var_4 is None



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_init_with_random_none. Retrieved 3/5 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.BaseProvider(random=var_0)
    var_2 = var_1.random



# Parsed testcases at query #11
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.seed
    var_3 = var_1.locale

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = 'fr'
    var_2 = {}
    var_3 = module_0.BaseDataProvider(var_1, var_0, **var_2)
    var_4 = var_3.seed
    var_5 = bool(var_3.seed == var_0)
    assert var_5 is True
    var_6 = var_3.locale
    var_7 = bool(var_3.locale == var_1)
    assert var_7 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_validate_enum_with_none. Retrieved 2/3 statements.
# Partially parsed test_validate_enum_with_valid_item. Retrieved 1/3 statements.
# Partially parsed test_validate_enum_with_invalid_item. Retrieved 2/4 statements.


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
    var_1 = 'C'
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_base_data_provider_initialization_with_default_locale. Retrieved 3/6 statements.
# Failed to parse test_base_data_provider_initialization_with_custom_locale.
# Partially parsed test_base_data_provider_initialization_with_custom_seed. Retrieved 4/7 statements.
# Partially parsed test_base_data_provider_initialization_with_custom_random. Retrieved 3/4 statements.
# Partially parsed test_base_data_provider_initialization_with_custom_locale_and_seed. Retrieved 1/8 statements.
# Partially parsed test_base_data_provider_initialization_with_custom_locale_and_random. Retrieved 1/5 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    var_3 = var_1._dataset
    var_4 = var_1.seed
    var_5 = var_1.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.locale
    var_4 = var_2._dataset
    var_5 = var_2.seed
    assert var_5 == 42
    var_6 = var_2.random

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_1.BaseDataProvider(**var_2)
    var_4 = var_3.locale
    var_5 = var_3._dataset
    var_6 = var_3.seed
    var_7 = var_3.random
    var_8 = bool(var_3.random is var_0)
    assert var_8 is True

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'invalid_random'
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_0.BaseDataProvider(**var_2)
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 42

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_predicate_at_line_11_evaluates_to_true. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'test_provider'
    var_1 = True
    var_2 = None



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_BaseDataProvider_init_with_locale_and_seed. Retrieved 4/5 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = 42
    var_2 = {}
    var_3 = module_0.BaseDataProvider(var_0, var_1, **var_2)
    var_4 = var_3.locale
    assert var_4 == 'en'
    var_5 = var_3.seed
    assert var_5 == 42
    var_6 = var_3._dataset



# Parsed testcases at query #16
#--------------------------

# Failed to parse test_constructor_initializes_empty_providers.




# Parsed testcases at query #17
#--------------------------

# Partially parsed test_validate_enum_with_valid_item. Retrieved 3/7 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.BaseProvider()



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_BaseProvider_constructor_with_defaults. Retrieved 2/4 statements.
# Partially parsed test_BaseProvider_constructor_with_custom_seed. Retrieved 3/5 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.seed
    var_2 = var_0.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 42
    var_3 = var_1.random

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseProvider(random=var_0)
    var_2 = var_1.seed
    var_3 = var_1.random
    var_4 = bool(var_1.random == var_0)
    assert var_4 is True

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0.BaseProvider(random=var_0)
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_init_super_call_is_correct. Retrieved 7/18 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = 'a'
    var_5 = 'b'
    var_6 = {var_4: var_0, var_5: var_1}



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_validate_enum_with_valid_item. Retrieved 2/5 statements.
# Partially parsed test_validate_enum_with_none_item. Retrieved 3/6 statements.
# Partially parsed test_validate_enum_with_invalid_item. Retrieved 3/6 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = module_0.BaseProvider()

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = module_0.BaseProvider()
    var_2 = None

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = module_0.BaseProvider()
    var_2 = 'invalid'
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_BaseDataProvider_constructor_default_locale. Retrieved 2/3 statements.
# Failed to parse test_BaseDataProvider_constructor_custom_locale.
# Failed to parse test_BaseDataProvider_constructor_locale_dependent_dataset.
# Partially parsed test_BaseDataProvider_constructor_locale_independent_dataset. Retrieved 1/4 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    var_3 = var_1._dataset
    var_4 = var_1.seed

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 12345
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.seed
    var_4 = bool(var_2.seed == var_0)
    assert var_4 is True
    var_5 = var_2.random.seed
    var_6 = bool(var_2.random.seed == var_0)
    assert var_6 is True

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_1.BaseDataProvider(**var_2)
    var_4 = var_3.random
    var_5 = bool(var_3.random is var_0)
    assert var_5 is True

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_0.BaseDataProvider(**var_2)
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = None



# Parsed testcases at query #22
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.random
    var_2 = bool(var_0.random is not None)
    assert var_2 is True
    var_3 = var_0.seed

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseProvider(random=var_0)
    var_2 = var_1.random
    var_3 = bool(var_1.random is var_0)
    assert var_3 is True
    var_4 = var_1.seed

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 12345
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed
    var_3 = bool(var_1.seed == var_0)
    assert var_3 is True
    var_4 = var_1.random
    var_5 = bool(var_1.random is not None)
    assert var_5 is True

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'not_a_random_instance'
    var_1 = module_0.BaseProvider(random=var_0)
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #23
#--------------------------

# Failed to parse test_constructor_initializes_empty_providers.




# Parsed testcases at query #24
#--------------------------

# Partially parsed test_validate_enum_with_valid_item. Retrieved 3/7 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.BaseProvider()



# Parsed testcases at query #25
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.seed
    assert var_3 == 42



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_base_data_provider_init_without_meta_name. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'Meta'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_reseed_with_global_seed. Retrieved 3/4 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.reseed()
    var_3 = var_1.seed
    var_4 = var_1.random.seed
    assert var_4 == 456



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_reseed_with_global_seed. Retrieved 2/3 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.reseed()
    var_2 = var_0.random._seed
    assert var_2 == 42



# Parsed testcases at query #29
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #30
#--------------------------

# Failed to parse test_constructor_initializes_empty_providers_dict.




# Parsed testcases at query #31
#--------------------------

# Failed to parse test_ProviderRegistry_constructor.




# Parsed testcases at query #32
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = 'keyword-only arguments'
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_constructor_with_default_random. Retrieved 2/4 statements.
# Failed to parse test_constructor_with_missing_seed.


import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseProvider(random=var_0)
    var_2 = var_1.random
    var_3 = bool(var_1.random is var_0)
    assert var_3 is True

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0.BaseProvider(random=var_0)
    var_2 = bool(False)
    assert var_2 is True

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 42

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed
    assert var_2 is None



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_reseed_with_missing_seed. Retrieved 2/3 statements.
# Partially parsed test_reseed_with_global_seed. Retrieved 2/4 statements.
# Partially parsed test_reseed_with_initial_missing_seed. Retrieved 1/3 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = 123
    var_3 = var_1.reseed(var_2)
    var_4 = var_1.seed
    assert var_4 == 123

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = None
    var_3 = var_1.reseed(var_2)
    var_4 = var_1.seed
    assert var_4 is None

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed

def test_case_0():
    var_0 = 99

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = 50
    var_3 = var_1.reseed(var_2)
    var_4 = var_1.seed
    assert var_4 == 50



# Parsed testcases at query #35
#--------------------------

# Failed to parse test_base_data_provider_constructor_with_custom_locale.
# Partially parsed test_base_data_provider_constructor_with_dataset_loading. Retrieved 2/6 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.seed
    var_4 = bool(var_2.seed == var_0)
    assert var_4 is True

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_1.BaseDataProvider(**var_2)
    var_4 = var_3.random
    var_5 = bool(var_3.random == var_0)
    assert var_5 is True

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'not a random instance'
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_0.BaseDataProvider(**var_2)
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 'test'
    var_1 = 'test.json'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_base_data_provider_initialization. Retrieved 2/7 statements.
# Partially parsed test_base_data_provider_default_initialization. Retrieved 2/4 statements.


def test_case_0():
    var_0 = 'en'
    var_1 = 42

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    var_3 = var_1.seed
    var_4 = var_1.random

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_1.BaseDataProvider(**var_2)
    var_4 = var_3.random
    var_5 = bool(var_3.random == var_0)
    assert var_5 is True

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'invalid_random'
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_0.BaseDataProvider(**var_2)
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_validate_enum_raises_non_enumerable_error_when_item_is_not_instance_of_enum. Retrieved 4/7 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'example_value'
    var_1 = 42
    var_2 = module_0.BaseProvider(seed=var_1)
    var_3 = 'not_an_enum_instance'
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_init_requires_keyword_only_arguments. Retrieved 1/5 statements.


def test_case_0():
    var_0 = None
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #39
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    var_3 = var_1.seed
    var_4 = var_1._dataset
    var_5 = bool(var_1._dataset == {})
    assert var_5 is True



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_init_requires_keyword_only_arguments. Retrieved 1/4 statements.


def test_case_0():
    var_0 = None



# Parsed testcases at query #41
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'en_US'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)
    var_3 = var_2.locale



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_BaseDataProvider_constructor_default_locale. Retrieved 2/3 statements.
# Partially parsed test_BaseDataProvider_constructor_custom_locale. Retrieved 3/4 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    var_3 = var_1._dataset
    var_4 = var_1.seed

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'en-US'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)
    var_3 = var_2.locale
    var_4 = bool(var_2.locale == var_0)
    assert var_4 is True
    var_5 = var_2._dataset

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 12345
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.seed
    var_4 = bool(var_2.seed == var_0)
    assert var_4 is True
    var_5 = var_2.random.seed
    var_6 = bool(var_2.random.seed == var_0)
    assert var_6 is True

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_1.BaseDataProvider(**var_2)
    var_4 = var_3.random
    var_5 = bool(var_3.random is var_0)
    assert var_5 is True

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_0.BaseDataProvider(**var_2)
    var_4 = bool(False)
    assert var_4 is True

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'xx-XX'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #43
#--------------------------




import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseProvider(random=var_0)
    var_2 = var_1.random
    var_3 = bool(var_1.random is var_0)
    assert var_3 is True



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_validate_enum_raises_non_enumerable_error_when_item_is_not_none_and_not_instance_of_enum. Retrieved 1/8 statements.


def test_case_0():
    var_0 = 'invalid_item'
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_predicate_at_line_21_evaluates_to_false. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'mock_name'
    var_1 = False
    var_2 = 'mock_name'



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_init_uses_keyword_only_arguments. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 42



# Parsed testcases at query #47
#--------------------------

# Failed to parse test_constructor.




# Parsed testcases at query #48
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    var_3 = var_1.seed
    var_4 = var_1._dataset
    var_5 = bool(var_1._dataset == {})
    assert var_5 is True



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_provider_registry_constructor. Retrieved 1/4 statements.


def test_case_0():
    var_0 = []
    var_1 = 'non_existent'



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_validate_enum_with_valid_item. Retrieved 3/7 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.BaseProvider()



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_constructor_with_default_values. Retrieved 2/3 statements.
# Failed to parse test_constructor_with_custom_locale.
# Partially parsed test_constructor_with_custom_locale_and_seed. Retrieved 1/3 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    var_3 = var_1.seed
    var_4 = var_1._dataset

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 12345
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.seed
    var_4 = bool(var_2.seed == var_0)
    assert var_4 is True

def test_case_0():
    var_0 = 12345



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_predicate_at_line_21_evaluates_to_true. Retrieved 6/12 statements.


def test_case_0():
    var_0 = 'test_provider'
    var_1 = True
    var_2 = 'test_data.json'
    var_3 = '/path/to/data'
    var_4 = 'Meta'
    var_5 = 'name'



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_BaseProvider_initializes_with_default_values. Retrieved 2/4 statements.
# Partially parsed test_BaseProvider_initializes_with_seed. Retrieved 3/5 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.seed
    var_2 = var_0.random

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseProvider(random=var_0)
    var_2 = var_1.random
    var_3 = bool(var_1.random is var_0)
    assert var_3 is True
    var_4 = var_1.seed

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 42
    var_3 = var_1.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'invalid_random'
    var_1 = module_0.BaseProvider(random=var_0)
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #54
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'en_US'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)
    var_3 = var_2.locale
    var_4 = bool(var_2.locale == var_0)
    assert var_4 is True



# Parsed testcases at query #55
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_init_only_accepts_keyword_arguments. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 123
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_init_without_datafile. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'test_provider'
    var_1 = False



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_init_with_default_random. Retrieved 2/4 statements.
# Failed to parse test_init_with_missing_seed.


import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseProvider(random=var_0)
    var_2 = var_1.random
    var_3 = bool(var_1.random is var_0)
    assert var_3 is True

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 42

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'not_a_random_instance'
    var_1 = module_0.BaseProvider(random=var_0)



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_BaseDataProvider_initialization_with_default_values. Retrieved 2/4 statements.
# Partially parsed test_BaseDataProvider_initialization_with_custom_locale_and_seed. Retrieved 2/7 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    var_3 = var_1.seed
    var_4 = var_1.random

def test_case_0():
    var_0 = 'fr'
    var_1 = 12345

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'invalid_locale'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_1.BaseDataProvider(**var_2)
    var_4 = var_3.random
    var_5 = bool(var_3.random == var_0)
    assert var_5 is True

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'invalid_random'
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_0.BaseDataProvider(**var_2)
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #60
#--------------------------

# Failed to parse test_provider_registry_initialization.
# Partially parsed test_provider_registry_register. Retrieved 1/4 statements.
# Partially parsed test_provider_registry_get_all. Retrieved 1/5 statements.
# Partially parsed test_provider_registry_get_existing. Retrieved 1/5 statements.
# Partially parsed test_provider_registry_get_nonexistent. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'test'

def test_case_0():
    var_0 = 'test'

def test_case_0():
    var_0 = 'test'

def test_case_0():
    var_0 = 'nonexistent'



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_reseed_with_missing_seed. Retrieved 1/2 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = 42
    var_2 = var_0.reseed(var_1)
    var_3 = var_0.seed
    assert var_3 == 42

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.seed

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = None
    var_2 = var_0.reseed(var_1)
    var_3 = var_0.seed
    assert var_3 is None

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = 123
    var_2 = var_0.reseed(var_1)
    var_3 = var_0.random.seed_value
    assert var_3 == 123



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_BaseDataProvider_constructor_default_locale. Retrieved 2/3 statements.
# Failed to parse test_BaseDataProvider_constructor_custom_locale.
# Partially parsed test_BaseDataProvider_constructor_inherits_random. Retrieved 4/6 statements.
# Partially parsed test_BaseDataProvider_constructor_loads_dataset. Retrieved 1/5 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    var_3 = var_1._dataset
    var_4 = var_1.seed

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.seed
    assert var_3 == 42

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = 'random'
    var_3 = hasattr(var_1, var_2)
    var_4 = bool(var_3)
    assert var_4 is True
    var_5 = var_1.random

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_1.BaseDataProvider(**var_2)
    var_4 = var_3.random
    var_5 = bool(var_3.random is var_0)
    assert var_5 is True

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_0.BaseDataProvider(**var_2)
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 'test.json'
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_constructor_with_default_random. Retrieved 2/4 statements.
# Failed to parse test_constructor_with_missing_seed.


import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseProvider(random=var_0)
    var_2 = var_1.random
    var_3 = bool(var_1.random == var_0)
    assert var_3 is True

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 42

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'invalid_random'
    var_1 = module_0.BaseProvider(random=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = bool(True)
    assert var_3 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_init_requires_keyword_only_arguments. Retrieved 3/5 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.BaseProvider(seed=var_0, random=var_0)
    var_2 = var_1.seed
    assert var_2 is None
    var_3 = var_1.random



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_validate_enum_with_none. Retrieved 5/7 statements.
# Partially parsed test_validate_enum_with_valid_enum_item. Retrieved 3/6 statements.
# Partially parsed test_validate_enum_with_invalid_item. Retrieved 4/7 statements.


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
    var_2 = module_0.BaseProvider()

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.BaseProvider()
    var_3 = 'invalid'
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True



# Parsed testcases at query #6
#--------------------------

# Failed to parse test_init_with_non_random_instance_raises_type_error.




# Parsed testcases at query #7
#--------------------------

# Partially parsed test_BaseProvider_initialization_with_default_random. Retrieved 2/4 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.random
    var_2 = bool(var_0.random is not None)
    assert var_2 is True
    var_3 = var_0.random
    var_4 = var_0.seed

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseProvider(random=var_0)
    var_2 = var_1.random
    var_3 = bool(var_1.random == var_0)
    assert var_3 is True
    var_4 = var_1.seed

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed
    var_3 = bool(var_1.seed == var_0)
    assert var_3 is True

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'not_random_instance'
    var_1 = module_0.BaseProvider(random=var_0)
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_reseed_updates_random_seed. Retrieved 4/6 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.reseed()
    var_2 = var_0.seed

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = None
    var_2 = var_0.reseed(var_1)
    var_3 = var_0.seed
    assert var_3 is None

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = 42
    var_2 = var_0.reseed(var_1)
    var_3 = var_0.seed
    assert var_3 == 42

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = 'test'
    var_2 = var_0.reseed(var_1)
    var_3 = var_0.seed
    assert var_3 == 'test'

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = 123
    var_2 = var_0.reseed(var_1)
    var_3 = var_0.reseed(var_1)



# Parsed testcases at query #9
#--------------------------

# Failed to parse test_provider_registry_initialization.
# Partially parsed test_provider_registry_register. Retrieved 1/4 statements.
# Partially parsed test_provider_registry_get_all. Retrieved 2/9 statements.
# Partially parsed test_provider_registry_get. Retrieved 1/5 statements.
# Partially parsed test_provider_registry_get_nonexistent. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'test_provider'

def test_case_0():
    var_0 = 'test_provider1'
    var_1 = 'test_provider2'

def test_case_0():
    var_0 = 'test_provider'

def test_case_0():
    var_0 = 'nonexistent_provider'



# Parsed testcases at query #10
#--------------------------

# Failed to parse test_ProviderRegistry_constructor.




# Parsed testcases at query #11
#--------------------------

# Partially parsed test_base_data_provider_init_with_default_locale. Retrieved 2/4 statements.
# Failed to parse test_base_data_provider_init_with_custom_locale.
# Partially parsed test_base_data_provider_init_with_seed. Retrieved 3/5 statements.
# Partially parsed test_base_data_provider_init_with_locale_and_seed. Retrieved 1/6 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    var_3 = var_1.random
    var_4 = var_1.seed

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 12345
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.locale
    var_4 = var_2.random
    var_5 = var_2.seed
    var_6 = bool(var_2.seed == var_0)
    assert var_6 is True

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_1.BaseDataProvider(**var_2)
    var_4 = var_3.locale
    var_5 = var_3.random
    var_6 = bool(var_3.random is var_0)
    assert var_6 is True
    var_7 = var_3.seed

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_0.BaseDataProvider(**var_2)
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 54321



# Parsed testcases at query #12
#--------------------------

# Failed to parse test_provider_registry_constructor.




# Parsed testcases at query #13
#--------------------------

# Failed to parse test_provider_registry_constructor.




# Parsed testcases at query #14
#--------------------------

# Partially parsed test_base_data_provider_constructor_default_locale. Retrieved 2/3 statements.
# Failed to parse test_base_data_provider_constructor_custom_locale.
# Partially parsed test_base_data_provider_constructor_inherits_random_from_base. Retrieved 4/6 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    var_3 = var_1._dataset
    var_4 = var_1.seed

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.seed
    assert var_3 == 42

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = 'random'
    var_3 = hasattr(var_1, var_2)
    var_4 = bool(var_3)
    assert var_4 is True
    var_5 = var_1.random

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_1.BaseDataProvider(**var_2)
    var_4 = var_3.random
    var_5 = bool(var_3.random is var_0)
    assert var_5 is True

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_0.BaseDataProvider(**var_2)
    var_4 = bool(False)
    assert var_4 is True

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1._dataset
    var_3 = bool(var_1._dataset == {})
    assert var_3 is True



# Parsed testcases at query #15
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.BaseProvider(random=var_0)
    var_2 = var_1.random
    var_3 = bool(var_1.random is not None)
    assert var_3 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_reseed_with_global_seed. Retrieved 2/3 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.reseed()
    var_2 = var_0.random._seed
    assert var_2 == 42



# Parsed testcases at query #17
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'fr'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'fr'

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 123
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.seed
    assert var_3 == 123

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_1.BaseDataProvider(**var_2)
    var_4 = var_3.random
    var_5 = bool(var_3.random == var_0)
    assert var_5 is True

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'invalid_random_instance'
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_0.BaseDataProvider(**var_2)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'es'
    var_1 = 456
    var_2 = 'value'
    var_3 = 'custom_arg'
    var_4 = 'custom_kwarg'
    var_5 = {var_3: var_2, var_4: var_2}
    var_6 = module_0.BaseDataProvider(var_0, var_1, **var_5)
    var_7 = var_6.locale
    assert var_7 == 'es'
    var_8 = var_6.seed
    assert var_8 == 456

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1._dataset
    var_3 = bool(var_1._dataset == {})
    assert var_3 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_predicate_at_line_11_evaluates_to_true. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'test_provider'
    var_1 = True
    var_2 = 123



# Parsed testcases at query #19
#--------------------------

# Failed to parse test_init_raises_type_error_when_random_is_not_instance_of_random.




# Parsed testcases at query #20
#--------------------------

# Partially parsed test_init_with_seed. Retrieved 2/3 statements.
# Partially parsed test_init_without_seed. Retrieved 1/2 statements.
# Partially parsed test_init_with_global_seed. Retrieved 1/4 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.seed
    assert var_3 == 42

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.seed

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.seed



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_provider_registry_constructor. Retrieved 1/4 statements.


def test_case_0():
    var_0 = []
    var_1 = 'non_existent_provider'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_provider_registry_initial_state. Retrieved 1/4 statements.


def test_case_0():
    var_0 = []
    var_1 = 'non_existent_provider'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_predicate_at_line_8_evaluates_to_true. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 'test_provider'
    var_1 = True
    var_2 = 'Meta'
    var_3 = 'name'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_init_requires_keyword_only_arguments. Retrieved 2/4 statements.


def test_case_0():
    var_0 = 123
    var_1 = None
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_validate_enum_non_enumerable_error. Retrieved 2/7 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = 'invalid_item'
    var_2 = bool(False)
    assert var_2 is True
    var_3 = bool(True)
    assert var_3 is True



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_predicate_at_line_11_evaluates_to_true. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 123



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_init_with_keyword_only_arguments. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 42



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_reseed_with_global_seed_not_missing. Retrieved 3/4 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.reseed()
    var_3 = var_1.random._seed
    assert var_3 == 42



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_predicate_at_line_11_evaluates_to_true. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 'test_provider'
    var_1 = True
    var_2 = 'Meta'
    var_3 = 'name'



# Parsed testcases at query #30
#--------------------------

# Failed to parse test_ProviderRegistry_constructor.




# Parsed testcases at query #31
#--------------------------




import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 1234
    var_2 = module_1.BaseProvider(seed=var_1, random=var_0)
    var_3 = var_2.seed
    assert var_3 == 1234
    var_4 = var_2.random
    var_5 = bool(var_2.random == var_0)
    assert var_5 is True



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_init_with_random_none. Retrieved 3/5 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.BaseProvider(random=var_0)
    var_2 = var_1.random

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseProvider(random=var_0)
    var_2 = var_1.random
    var_3 = bool(var_1.random is var_0)
    assert var_3 is True

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0.BaseProvider(random=var_0)



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_init_without_auto_register. Retrieved 2/7 statements.


def test_case_0():
    var_0 = False
    var_1 = 'name'



# Parsed testcases at query #34
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    var_3 = var_1.seed

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'en_US'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'en_US'

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 12345
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.seed
    assert var_3 == 12345

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'fr_FR'
    var_1 = 67890
    var_2 = {}
    var_3 = module_0.BaseDataProvider(var_0, var_1, **var_2)
    var_4 = var_3.locale
    assert var_4 == 'fr_FR'
    var_5 = var_3.seed
    assert var_5 == 67890

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'de_DE'
    var_1 = 54321
    var_2 = 'value'
    var_3 = 'extra_arg'
    var_4 = {var_3: var_2}
    var_5 = module_0.BaseDataProvider(var_0, var_1, **var_4)
    var_6 = var_5.locale
    assert var_6 == 'de_DE'
    var_7 = var_5.seed
    assert var_7 == 54321



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_validate_enum_with_none. Retrieved 5/7 statements.
# Partially parsed test_validate_enum_with_valid_enum_item. Retrieved 4/7 statements.
# Partially parsed test_validate_enum_with_invalid_enum_item. Retrieved 5/10 statements.


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
    var_2 = 'C'
    var_3 = module_0.BaseProvider()

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'A'
    var_1 = 'B'
    var_2 = 'C'
    var_3 = 'D'
    var_4 = module_0.BaseProvider()
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_base_data_provider_constructor_default_values. Retrieved 2/4 statements.
# Partially parsed test_base_data_provider_constructor_custom_locale. Retrieved 3/5 statements.
# Partially parsed test_base_data_provider_constructor_custom_seed. Retrieved 3/5 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    var_3 = var_1.seed
    var_4 = var_1.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'en_US'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'en_US'
    var_4 = var_2.seed
    var_5 = var_2.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.locale
    var_4 = var_2.seed
    assert var_4 == 42
    var_5 = var_2.random

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_1.BaseDataProvider(**var_2)
    var_4 = var_3.locale
    var_5 = var_3.seed
    var_6 = var_3.random
    var_7 = bool(var_3.random == var_0)
    assert var_7 is True

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'invalid_random'
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_0.BaseDataProvider(**var_2)



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_base_data_provider_init_with_default_locale. Retrieved 2/3 statements.
# Failed to parse test_base_data_provider_init_with_custom_locale.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    var_3 = var_1._dataset
    var_4 = var_1.seed

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.seed
    assert var_3 == 42

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'invalid_locale'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_1.BaseDataProvider(**var_2)
    var_4 = var_3.random
    var_5 = bool(var_3.random is var_0)
    assert var_5 is True

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'invalid_random'
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_0.BaseDataProvider(**var_2)
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_base_data_provider_initialization_with_default_values. Retrieved 2/4 statements.
# Partially parsed test_base_data_provider_initialization_with_custom_locale. Retrieved 1/3 statements.
# Partially parsed test_base_data_provider_initialization_with_locale_and_seed. Retrieved 2/4 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    var_3 = var_1.seed
    var_4 = var_1.random

def test_case_0():
    var_0 = 'fr'

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 12345
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.seed
    var_4 = bool(var_2.seed == var_0)
    assert var_4 is True

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_1.BaseDataProvider(**var_2)
    var_4 = var_3.random
    var_5 = bool(var_3.random == var_0)
    assert var_5 is True

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'not_a_random_instance'
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_0.BaseDataProvider(**var_2)
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 'de'
    var_1 = 67890



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_validate_enum_with_invalid_item. Retrieved 4/7 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'valid'
    var_1 = 42
    var_2 = module_0.BaseProvider(seed=var_1)
    var_3 = 'invalid'
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_base_provider_initialization. Retrieved 3/5 statements.
# Partially parsed test_base_provider_default_seed. Retrieved 2/4 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed
    var_3 = bool(var_1.seed == var_0)
    assert var_3 is True
    var_4 = var_1.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.seed
    var_2 = var_0.random

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseProvider(random=var_0)
    var_2 = var_1.random
    var_3 = bool(var_1.random is var_0)
    assert var_3 is True
    var_4 = var_1.seed

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0.BaseProvider(random=var_0)
    var_2 = bool(False)
    assert var_2 is True

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = 123
    var_3 = var_1.reseed(var_2)
    var_4 = var_1.seed
    var_5 = bool(var_1.seed == var_2)
    assert var_5 is True

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.reseed()
    var_3 = var_1.seed

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = str(var_0)
    assert var_1 == 'BaseProvider'



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_base_data_provider_init_with_default_locale. Retrieved 2/3 statements.
# Failed to parse test_base_data_provider_init_with_custom_locale.
# Partially parsed test_base_data_provider_init_with_locale_and_seed. Retrieved 1/3 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    var_3 = var_1._dataset
    var_4 = var_1.seed

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.seed
    assert var_3 == 42

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_1.BaseDataProvider(**var_2)
    var_4 = var_3.random
    var_5 = bool(var_3.random is var_0)
    assert var_5 is True

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_0.BaseDataProvider(**var_2)
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 123



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_validate_enum_predicate_evaluates_to_false. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 'value'
    var_1 = 'invalid'
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_predicate_evaluates_to_true. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'mock_provider'



# Parsed testcases at query #44
#--------------------------

# Failed to parse test_provider_registry_initialization.




# Parsed testcases at query #45
#--------------------------

# Failed to parse test_provider_registry_constructor.




# Parsed testcases at query #46
#--------------------------

# Partially parsed test_BaseDataProvider_initialization_with_default_values. Retrieved 2/4 statements.
# Failed to parse test_BaseDataProvider_initialization_with_custom_locale.
# Partially parsed test_BaseDataProvider_initialization_with_custom_seed. Retrieved 3/5 statements.
# Partially parsed test_BaseDataProvider_initialization_with_custom_locale_and_seed. Retrieved 1/6 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    var_3 = var_1.seed
    var_4 = var_1.random
    var_5 = var_1._dataset
    var_6 = bool(var_1._dataset == {})
    assert var_6 is True

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.locale
    var_4 = var_2.seed
    var_5 = bool(var_2.seed == var_0)
    assert var_5 is True
    var_6 = var_2.random
    var_7 = var_2._dataset
    var_8 = bool(var_2._dataset == {})
    assert var_8 is True

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_1.BaseDataProvider(**var_2)
    var_4 = var_3.locale
    var_5 = var_3.seed
    var_6 = var_3.random
    var_7 = bool(var_3.random == var_0)
    assert var_7 is True
    var_8 = var_3._dataset
    var_9 = bool(var_3._dataset == {})
    assert var_9 is True

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'not a Random instance'
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_0.BaseDataProvider(**var_2)

def test_case_0():
    var_0 = 42



# Parsed testcases at query #47
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



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_validate_enum_with_valid_item. Retrieved 3/7 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.BaseProvider()



# Parsed testcases at query #49
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale



# Parsed testcases at query #50
#--------------------------

# Failed to parse test_provider_registry_initialization.
# Partially parsed test_provider_registry_register. Retrieved 1/4 statements.
# Partially parsed test_provider_registry_get_all. Retrieved 1/5 statements.
# Partially parsed test_provider_registry_get_existing. Retrieved 1/5 statements.
# Partially parsed test_provider_registry_get_nonexistent. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'test'

def test_case_0():
    var_0 = 'test'

def test_case_0():
    var_0 = 'test'

def test_case_0():
    var_0 = 'nonexistent'



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_constructor_with_default_random. Retrieved 2/4 statements.
# Failed to parse test_constructor_with_missing_seed.


import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseProvider(random=var_0)
    var_2 = var_1.random
    var_3 = bool(var_1.random == var_0)
    assert var_3 is True

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 42

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'not_a_random_instance'
    var_1 = module_0.BaseProvider(random=var_0)
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_validate_enum_with_valid_item. Retrieved 2/10 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2



# Parsed testcases at query #53
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'en_US'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'en_US'

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.seed

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 12345
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.seed
    assert var_3 == 12345

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1._dataset
    var_3 = bool(var_1._dataset == {})
    assert var_3 is True

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'en_US'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'en_US'



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_init_without_keyword_only_args. Retrieved 2/4 statements.


def test_case_0():
    var_0 = 123
    var_1 = None
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_validate_enum_with_valid_item. Retrieved 2/7 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider()



# Parsed testcases at query #56
#--------------------------

# Failed to parse test_constructor_initializes_providers.




# Parsed testcases at query #57
#--------------------------

# Failed to parse test_init_with_random_not_instance_of_random.




# Parsed testcases at query #58
#--------------------------

# Partially parsed test_init_without_datafile. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'test_provider'
    var_1 = False



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_constructor_with_default_parameters. Retrieved 2/4 statements.
# Partially parsed test_constructor_with_seed. Retrieved 3/5 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.seed
    var_2 = var_0.random

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseProvider(random=var_0)
    var_2 = var_1.random
    var_3 = bool(var_1.random is var_0)
    assert var_3 is True
    var_4 = var_1.seed

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 42
    var_3 = var_1.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0.BaseProvider(random=var_0)
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #60
#--------------------------

# Failed to parse test_init_with_non_random_instance.




# Parsed testcases at query #61
#--------------------------

# Failed to parse test_provider_registry_constructor.




# Parsed testcases at query #62
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_init_sets_up_locale_and_loads_dataset. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 'test_provider'
    var_1 = 'test_data.json'
    var_2 = '/tmp'
    var_3 = '_dataset'
    var_4 = 'locale'



# Parsed testcases at query #64
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_no_meta_class_defined. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'Meta'



# Parsed testcases at query #66
#--------------------------

# Partially parsed test_init_requires_keyword_only_arguments. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 123
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 123



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_auto_register_false. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'test_provider'
    var_1 = False
    var_2 = 'test_provider'



# Parsed testcases at query #68
#--------------------------

# Partially parsed test_auto_register_is_false. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 'test_provider'
    var_1 = False
    var_2 = 'Meta'
    var_3 = 'auto_register'



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_init_with_non_keyword_arguments_should_raise_error. Retrieved 2/4 statements.


def test_case_0():
    var_0 = 123
    var_1 = None
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_random_is_none_creates_new_random_instance. Retrieved 3/5 statements.


import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseProvider(random=var_0)
    var_2 = var_1.random
    var_3 = bool(var_1.random is var_0)
    assert var_3 is True

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.BaseProvider(random=var_0)
    var_2 = var_1.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'not_a_random_instance'
    var_1 = module_0.BaseProvider(random=var_0)
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #71
#--------------------------

# Partially parsed test_predicate_at_line_8_evaluates_to_true. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 'test_provider'
    var_1 = True
    var_2 = '_dataset'



# Parsed testcases at query #72
#--------------------------

# Partially parsed test_base_data_provider_constructor. Retrieved 4/5 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = 42
    var_2 = {}
    var_3 = module_0.BaseDataProvider(var_0, var_1, **var_2)
    var_4 = var_3.locale
    assert var_4 == 'en'
    var_5 = var_3.seed
    assert var_5 == 42
    var_6 = var_3._dataset

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.locale
    var_4 = var_2.seed
    assert var_4 == 42

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'en'
    var_4 = var_2.seed

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'invalid_locale'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'en'
    var_2 = 'random'
    var_3 = {var_2: var_0}
    var_4 = module_1.BaseDataProvider(var_1, **var_3)
    var_5 = var_4.random
    var_6 = bool(var_4.random is var_0)
    assert var_6 is True



# Parsed testcases at query #73
#--------------------------

# Partially parsed test_base_data_provider_init_with_default_values. Retrieved 2/4 statements.
# Failed to parse test_base_data_provider_init_with_custom_locale.
# Partially parsed test_base_data_provider_init_with_custom_seed. Retrieved 3/5 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    var_3 = var_1.seed
    var_4 = var_1.random
    var_5 = var_1._dataset
    var_6 = bool(var_1._dataset == {})
    assert var_6 is True

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.locale
    var_4 = var_2.seed
    assert var_4 == 42
    var_5 = var_2.random
    var_6 = var_2._dataset
    var_7 = bool(var_2._dataset == {})
    assert var_7 is True

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_1.BaseDataProvider(**var_2)
    var_4 = var_3.locale
    var_5 = var_3.seed
    var_6 = var_3.random
    var_7 = bool(var_3.random is var_0)
    assert var_7 is True
    var_8 = var_3._dataset
    var_9 = bool(var_3._dataset == {})
    assert var_9 is True

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_0.BaseDataProvider(**var_2)
    var_4 = bool(False)
    assert var_4 is True

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 42
    var_2 = 'random'
    var_3 = {var_2: var_0}
    var_4 = module_1.BaseDataProvider(seed=var_1, **var_3)
    var_5 = var_4.locale
    var_6 = var_4.seed
    assert var_6 == 42
    var_7 = var_4.random
    var_8 = bool(var_4.random is var_0)
    assert var_8 is True
    var_9 = var_4._dataset
    var_10 = bool(var_4._dataset == {})
    assert var_10 is True



# Parsed testcases at query #74
#--------------------------

# Partially parsed test_base_data_provider_constructor_with_default_locale. Retrieved 2/3 statements.
# Failed to parse test_base_data_provider_constructor_with_custom_locale.
# Partially parsed test_base_data_provider_constructor_with_seed. Retrieved 3/4 statements.
# Partially parsed test_base_data_provider_constructor_with_locale_and_seed. Retrieved 1/5 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    var_3 = var_1._dataset
    var_4 = var_1.seed

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.locale
    var_4 = var_2._dataset
    var_5 = var_2.seed
    assert var_5 == 42

def test_case_0():
    var_0 = 123



# Parsed testcases at query #75
#--------------------------

# Failed to parse test_init_with_locale_and_seed.




# Parsed testcases at query #76
#--------------------------

# Partially parsed test_constructor_called_with_positional_args. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 123
    var_1 = 'Constructor should raise TypeError when called with positional arguments'
    var_2 = AssertionError(var_1)



# Parsed testcases at query #77
#--------------------------

# Partially parsed test_constructor_with_default_seed_and_random. Retrieved 2/4 statements.
# Partially parsed test_constructor_with_custom_seed. Retrieved 3/5 statements.
# Partially parsed test_constructor_with_none_seed. Retrieved 3/5 statements.
# Partially parsed test_constructor_with_none_random. Retrieved 3/5 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.seed
    var_2 = var_0.random

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseProvider(random=var_0)
    var_2 = var_1.random
    var_3 = bool(var_1.random is var_0)
    assert var_3 is True

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 42
    var_3 = var_1.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'not_a_random_instance'
    var_1 = module_0.BaseProvider(random=var_0)
    var_2 = bool(False)
    assert var_2 is True

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed
    assert var_2 is None
    var_3 = var_1.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.BaseProvider(random=var_0)
    var_2 = var_1.random



# Parsed testcases at query #78
#--------------------------

# Partially parsed test_constructor_with_default_parameters. Retrieved 2/4 statements.
# Partially parsed test_constructor_with_seed. Retrieved 3/5 statements.
# Partially parsed test_constructor_with_none_seed. Retrieved 3/5 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.seed
    var_2 = var_0.random

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseProvider(random=var_0)
    var_2 = var_1.random
    var_3 = bool(var_1.random is var_0)
    assert var_3 is True
    var_4 = var_1.seed

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0.BaseProvider(random=var_0)
    var_2 = bool(False)
    assert var_2 is True

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 42
    var_3 = var_1.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed
    assert var_2 is None
    var_3 = var_1.random



# Parsed testcases at query #79
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = 42
    var_2 = {}
    var_3 = module_0.BaseDataProvider(var_0, var_1, **var_2)
    var_4 = var_3._dataset
    var_5 = bool(var_3._dataset == {})
    assert var_5 is True
    var_6 = var_3.locale
    assert var_6 == 'en'
    var_7 = var_3.seed
    assert var_7 == 42



# Parsed testcases at query #80
#--------------------------




import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Random()
    var_2 = module_1.BaseProvider(seed=var_0, random=var_1)



# Parsed testcases at query #81
#--------------------------

# Partially parsed test_BaseDataProvider_constructor_default_locale. Retrieved 2/3 statements.
# Failed to parse test_BaseDataProvider_constructor_custom_locale.
# Partially parsed test_BaseDataProvider_constructor_with_seed. Retrieved 3/4 statements.
# Partially parsed test_BaseDataProvider_constructor_inherits_random_from_base. Retrieved 3/4 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    var_3 = var_1._dataset
    var_4 = var_1.seed

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.locale
    var_4 = var_2._dataset
    var_5 = var_2.seed
    assert var_5 == 42

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_1.BaseDataProvider(**var_2)
    var_4 = var_3.random
    var_5 = bool(var_3.random is var_0)
    assert var_5 is True
    var_6 = var_3.locale
    var_7 = var_3._dataset



# Parsed testcases at query #82
#--------------------------

# Partially parsed test_base_data_provider_constructor_default. Retrieved 2/4 statements.
# Failed to parse test_base_data_provider_constructor_custom_locale.
# Partially parsed test_base_data_provider_constructor_custom_seed. Retrieved 3/5 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    var_3 = var_1.seed
    var_4 = var_1.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 12345
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.locale
    var_4 = var_2.seed
    assert var_4 == 12345
    var_5 = var_2.random

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_1.BaseDataProvider(**var_2)
    var_4 = var_3.locale
    var_5 = var_3.seed
    var_6 = var_3.random
    var_7 = bool(var_3.random == var_0)
    assert var_7 is True

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'invalid_random'
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_0.BaseDataProvider(**var_2)



# Parsed testcases at query #83
#--------------------------

# Partially parsed test_base_data_provider_constructor_default_locale. Retrieved 2/3 statements.
# Failed to parse test_base_data_provider_constructor_custom_locale.
# Partially parsed test_base_data_provider_constructor_inherits_random. Retrieved 4/6 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    var_3 = var_1._dataset
    var_4 = var_1.seed

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.seed
    assert var_3 == 42

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = 'random'
    var_3 = hasattr(var_1, var_2)
    var_4 = bool(var_3)
    assert var_4 is True
    var_5 = var_1.random

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_1.BaseDataProvider(**var_2)
    var_4 = var_3.random
    var_5 = bool(var_3.random is var_0)
    assert var_5 is True

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_0.BaseDataProvider(**var_2)
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #84
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale



# Parsed testcases at query #85
#--------------------------

# Partially parsed test_base_data_provider_constructor_with_default_locale. Retrieved 2/4 statements.
# Partially parsed test_base_data_provider_constructor_with_custom_locale. Retrieved 3/5 statements.
# Partially parsed test_base_data_provider_constructor_with_custom_seed. Retrieved 3/5 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    var_3 = var_1._dataset
    var_4 = bool(var_1._dataset == {})
    assert var_4 is True
    var_5 = var_1.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'fr'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'fr'
    var_4 = var_2._dataset
    var_5 = bool(var_2._dataset == {})
    assert var_5 is True
    var_6 = var_2.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 123
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.seed
    assert var_3 == 123
    var_4 = var_2.random

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_1.BaseDataProvider(**var_2)
    var_4 = var_3.random
    var_5 = bool(var_3.random == var_0)
    assert var_5 is True

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'invalid_random'
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_0.BaseDataProvider(**var_2)
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #86
#--------------------------

# Partially parsed test_has_seed_evaluates_to_false. Retrieved 2/3 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = None
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)



# Parsed testcases at query #87
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'en_US'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'en_US'



# Parsed testcases at query #88
#--------------------------

# Partially parsed test_base_data_provider_constructor_default_locale. Retrieved 2/3 statements.
# Failed to parse test_base_data_provider_constructor_custom_locale.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    var_3 = var_1._dataset
    var_4 = var_1.seed

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.seed
    var_4 = bool(var_2.seed == var_0)
    assert var_4 is True

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'invalid_locale'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_1.BaseDataProvider(**var_2)
    var_4 = var_3.random
    var_5 = bool(var_3.random is var_0)
    assert var_5 is True

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'invalid_random'
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_0.BaseDataProvider(**var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True



