####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Failed to parse test_provider_registry_constructor.




# Parsed testcases at query #2
#--------------------------

# Partially parsed test_constructor_with_default_parameters. Retrieved 2/4 statements.
# Partially parsed test_constructor_with_seed. Retrieved 3/5 statements.
# Partially parsed test_constructor_with_none_seed. Retrieved 3/5 statements.


import mimesis.providers.base as module_0


def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.seed
    var_2 = var_0.random

import mimesis.providers.base as module_1
import mimesis.random as module_0


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


def test_case_0():
    var_0 = None
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed
    assert var_2 is None
    var_3 = var_1.random

import mimesis.random as module_0


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 123
    var_2 = module_1.BaseProvider(seed=var_1, random=var_0)
    var_3 = var_2.random
    var_4 = bool(var_2.random is var_0)
    assert var_4 is True
    var_5 = var_2.seed
    assert var_5 == 123

import mimesis.providers.base as module_0


def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0.BaseProvider(random=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'The random must be an instance of mimesis.random.Random'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_init_with_keyword_only_arguments. Retrieved 3/5 statements.



def test_case_0():
    var_0 = None
    var_1 = module_0.BaseProvider(seed=var_0, random=var_0)
    var_2 = var_1.seed
    assert var_2 is None
    var_3 = var_1.random



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_base_data_provider_initializes_with_default_locale. Retrieved 2/3 statements.
# Failed to parse test_base_data_provider_initializes_with_custom_locale.
# Partially parsed test_base_data_provider_loads_dataset_with_datafile. Retrieved 2/9 statements.
# Partially parsed test_base_data_provider_handles_locale_with_separator. Retrieved 2/7 statements.
# Partially parsed test_base_data_provider_calls_parent_constructor. Retrieved 2/3 statements.
# Failed to parse test_base_data_provider_str_representation.
# Partially parsed test_base_data_provider_without_locale_dependent_str. Retrieved 2/6 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    var_3 = var_1._dataset
    var_4 = var_1.seed


def test_case_0():
    var_0 = 12345
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.seed
    assert var_3 == 12345

import mimesis.random as module_0


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
    var_5 = bool(True)
    assert var_5 is True

def test_case_0():
    var_0 = 'test'
    var_1 = 'test.json'

def test_case_0():
    var_0 = 'test'
    var_1 = 'test.json'


def test_case_0():
    var_0 = 999
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)

def test_case_0():
    var_0 = 'nonlocale'
    var_1 = False



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_auto_register_false_prevents_registration. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'test_provider'
    var_1 = False
    var_2 = 'test_provider'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_auto_register_provider. Retrieved 3/6 statements.
# Partially parsed test_auto_register_provider_default. Retrieved 2/5 statements.
# Partially parsed test_auto_register_provider_false. Retrieved 3/7 statements.
# Partially parsed test_auto_register_no_meta_name. Retrieved 1/5 statements.
# Partially parsed test_auto_register_inheritance. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'test_provider'
    var_1 = True
    var_2 = 'test_provider'

def test_case_0():
    var_0 = 'test_provider_default'
    var_1 = 'test_provider_default'

def test_case_0():
    var_0 = 'test_provider_false'
    var_1 = False
    var_2 = 'test_provider_false'

def test_case_0():
    var_0 = 'name'

def test_case_0():
    var_0 = 'parent'
    var_1 = 'parent'



# Parsed testcases at query #7
#--------------------------

# Failed to parse test_provider_registry_constructor.




# Parsed testcases at query #8
#--------------------------

# Partially parsed test_base_data_provider_init_default_locale. Retrieved 2/3 statements.
# Failed to parse test_base_data_provider_init_custom_locale.
# Partially parsed test_base_data_provider_init_dataset_loading. Retrieved 2/7 statements.
# Partially parsed test_base_data_provider_init_auto_registration. Retrieved 2/5 statements.
# Partially parsed test_base_data_provider_init_auto_register_false. Retrieved 3/7 statements.
# Failed to parse test_base_data_provider_init_locale_separator.
# Partially parsed test_base_data_provider_init_empty_dataset. Retrieved 2/5 statements.
# Partially parsed test_base_data_provider_init_master_locale_fallback. Retrieved 2/6 statements.
# Partially parsed test_base_data_provider_init_random_default. Retrieved 2/4 statements.
# Failed to parse test_base_data_provider_init_seed_missing.
# Failed to parse test_base_data_provider_init_locale_default.
# Partially parsed test_base_data_provider_init_with_args_kwargs. Retrieved 2/4 statements.
# Partially parsed test_base_data_provider_init_dataset_update. Retrieved 5/10 statements.
# Failed to parse test_base_data_provider_init_get_current_locale.
# Failed to parse test_base_data_provider_init_str_representation.
# Failed to parse test_base_data_provider_init_override_locale_context.



def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    var_3 = var_1._dataset
    var_4 = var_1.seed


def test_case_0():
    var_0 = 42
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.seed
    assert var_3 == 42

import mimesis.random as module_0


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
    var_4 = 'The random must be an instance of mimesis.random.Random'


def test_case_0():
    var_0 = 'invalid_locale'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)

def test_case_0():
    var_0 = 'test'
    var_1 = 'test.json'

def test_case_0():
    var_0 = 'test_provider'
    var_1 = 'test_provider'

def test_case_0():
    var_0 = 'test_provider_false'
    var_1 = False
    var_2 = 'test_provider_false'


def test_case_0():
    var_0 = 123
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.random._seed
    assert var_3 == 123

def test_case_0():
    var_0 = 'test_empty'
    var_1 = ''

def test_case_0():
    var_0 = 'test_master'
    var_1 = 'test.json'


def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.random

def test_case_0():
    var_0 = 99
    var_1 = 'test'

def test_case_0():
    var_0 = 'test_update'
    var_1 = 'test.json'
    var_2 = 'new_key'
    var_3 = 'new_value'
    var_4 = {var_2: var_3}



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_base_data_provider_initialization_with_default_locale. Retrieved 2/4 statements.
# Failed to parse test_base_data_provider_initialization_with_custom_locale.
# Partially parsed test_base_data_provider_initialization_with_seed. Retrieved 3/5 statements.
# Partially parsed test_base_data_provider_initialization_locale_and_seed. Retrieved 1/3 statements.
# Partially parsed test_base_data_provider_initialization_with_datafile_meta. Retrieved 2/7 statements.
# Partially parsed test_base_data_provider_initialization_without_datafile_meta. Retrieved 1/5 statements.
# Failed to parse test_base_data_provider_initialization_with_missing_seed.
# Partially parsed test_base_data_provider_initialization_with_none_seed. Retrieved 3/5 statements.
# Partially parsed test_base_data_provider_initialization_with_global_seed_set. Retrieved 2/6 statements.
# Partially parsed test_base_data_provider_initialization_with_kwargs. Retrieved 2/7 statements.
# Partially parsed test_base_data_provider_initialization_with_args. Retrieved 1/4 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    var_3 = var_1._dataset
    var_4 = bool(var_1._dataset == {})
    assert var_4 is True
    var_5 = var_1.random


def test_case_0():
    var_0 = 42
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.seed
    assert var_3 == 42
    var_4 = var_2.random

import mimesis.random as module_0


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


def test_case_0():
    var_0 = 'unsupported'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True


def test_case_0():
    var_0 = 'en-US'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'en-US'
    var_4 = var_2._dataset
    var_5 = bool(var_2._dataset != {})
    assert var_5 is True

def test_case_0():
    var_0 = 'test'
    var_1 = 'test.json'

def test_case_0():
    var_0 = 'test'


def test_case_0():
    var_0 = None
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.seed
    assert var_3 is None
    var_4 = var_2.random


def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.seed
    var_3 = var_1.random

def test_case_0():
    var_0 = 555
    var_1 = None

def test_case_0():
    var_0 = 123
    var_1 = bool(False)
    assert var_1 is True
    var_2 = bool(True)
    assert var_2 is True



# Parsed testcases at query #10
#--------------------------

# Failed to parse test_provider_registry_constructor.




# Parsed testcases at query #11
#--------------------------

# Partially parsed test_auto_register_provider. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'test_provider'
    var_1 = True
    var_2 = 'test_provider'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_init_with_default_locale_and_seed. Retrieved 2/4 statements.
# Failed to parse test_init_with_custom_locale.
# Partially parsed test_init_dataset_loaded. Retrieved 2/7 statements.
# Partially parsed test_init_with_locale_and_seed. Retrieved 1/3 statements.
# Partially parsed test_init_without_auto_register. Retrieved 2/5 statements.
# Partially parsed test_init_inheritance_order. Retrieved 1/6 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    var_3 = var_1.seed
    var_4 = var_1.random


def test_case_0():
    var_0 = 42
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.seed
    assert var_3 == 42

import mimesis.random as module_0


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
    var_0 = 'invalid_locale'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True


def test_case_0():
    var_0 = 'invalid_random'
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_0.BaseDataProvider(**var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

def test_case_0():
    var_0 = 'test'
    var_1 = 'test.json'
    var_2 = bool(True)
    assert var_2 is True

def test_case_0():
    var_0 = 123

def test_case_0():
    var_0 = 'non_registered'
    var_1 = False

def test_case_0():
    var_0 = 'custom'



# Parsed testcases at query #13
#--------------------------

# Failed to parse test_constructor_initializes_empty_providers.




# Parsed testcases at query #14
#--------------------------

# Partially parsed test_base_data_provider_initialization_with_default_locale. Retrieved 2/4 statements.
# Failed to parse test_base_data_provider_initialization_with_custom_locale.
# Partially parsed test_base_data_provider_initialization_with_seed. Retrieved 3/5 statements.
# Partially parsed test_base_data_provider_initialization_with_locale_and_seed. Retrieved 1/3 statements.
# Failed to parse test_base_data_provider_initialization_with_missing_seed.
# Partially parsed test_base_data_provider_initialization_with_datafile_meta. Retrieved 2/6 statements.
# Partially parsed test_base_data_provider_initialization_without_datafile_meta. Retrieved 1/4 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    var_3 = var_1._dataset
    var_4 = bool(var_1._dataset == {})
    assert var_4 is True
    var_5 = var_1.random


def test_case_0():
    var_0 = 42
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.seed
    assert var_3 == 42
    var_4 = var_2.random

import mimesis.random as module_0


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

def test_case_0():
    var_0 = 123


def test_case_0():
    var_0 = 'unsupported'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)


def test_case_0():
    var_0 = None
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.seed
    assert var_3 is None

def test_case_0():
    var_0 = 'test'
    var_1 = 'test.json'

def test_case_0():
    var_0 = 'test'



# Parsed testcases at query #15
#--------------------------




import mimesis.random as module_0


def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseProvider(random=var_0)
    var_2 = var_1.random
    var_3 = bool(var_1.random is var_0)
    assert var_3 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_reseed_with_missing_seed_and_no_global_seed. Retrieved 2/4 statements.
# Partially parsed test_reseed_with_missing_seed_and_global_seed_set. Retrieved 2/4 statements.


import mimesis.providers.base as module_0


def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.reseed()
    var_2 = var_0.seed


def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.reseed()
    var_2 = var_0.random._seed
    assert var_2 == 12345
    var_3 = var_0.seed


def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = None
    var_2 = var_0.reseed(var_1)
    var_3 = var_0.seed
    assert var_3 is None


def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = 999
    var_2 = var_0.reseed(var_1)
    var_3 = var_0.random._seed
    assert var_3 == 999
    var_4 = var_0.seed
    assert var_4 == 999


def test_case_0():
    var_0 = 100
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = 5
    var_3 = range(var_2)
    var_4 = 1
    var_5 = [var_1.random.randint(var_4, var_0) for _ in var_3]
    var_6 = var_1.reseed(var_0)
    var_7 = range(var_2)
    var_8 = [var_1.random.randint(var_4, var_0) for _ in var_7]
    var_9 = bool(var_5 == var_8)
    assert var_9 is True
    var_10 = 200
    var_11 = var_1.reseed(var_10)
    var_12 = range(var_2)
    var_13 = [var_1.random.randint(var_4, var_0) for _ in var_12]
    var_14 = bool(var_5 != var_13)
    assert var_14 is True


def test_case_0():
    var_0 = 50
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 50
    var_3 = 60
    var_4 = var_1.reseed(var_3)
    var_5 = var_1.seed
    assert var_5 == 60



# Parsed testcases at query #17
#--------------------------

# Failed to parse test_provider_registry_constructor.




# Parsed testcases at query #18
#--------------------------

# Failed to parse test_constructor_initializes_empty_providers.




# Parsed testcases at query #19
#--------------------------

# Partially parsed test_base_data_provider_initialization_with_default_locale. Retrieved 2/3 statements.
# Failed to parse test_base_data_provider_initialization_with_custom_locale.
# Partially parsed test_base_data_provider_initialization_with_locale_and_seed. Retrieved 1/3 statements.
# Partially parsed test_base_data_provider_initialization_dataset_empty_when_no_datafile. Retrieved 2/5 statements.
# Partially parsed test_base_data_provider_initialization_dataset_loaded_with_datafile. Retrieved 3/11 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    var_3 = var_1._dataset
    var_4 = var_1.seed


def test_case_0():
    var_0 = 12345
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.seed
    assert var_3 == 12345

import mimesis.random as module_0


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
    var_5 = bool(True)
    assert var_5 is True


def test_case_0():
    var_0 = 'invalid_locale'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 42

def test_case_0():
    var_0 = 'test'
    var_1 = False

def test_case_0():
    var_0 = 'test'
    var_1 = False
    var_2 = 'test.json'


def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = 'random'
    var_3 = hasattr(var_1, var_2)
    var_4 = bool(var_3)
    assert var_4 is True
    var_5 = 'seed'
    var_6 = hasattr(var_1, var_5)
    var_7 = bool(var_6)
    assert var_7 is True
    var_8 = 'reseed'
    var_9 = hasattr(var_1, var_8)
    var_10 = bool(var_9)
    assert var_10 is True
    var_11 = 'validate_enum'
    var_12 = hasattr(var_1, var_11)
    var_13 = bool(var_12)
    assert var_13 is True



# Parsed testcases at query #20
#--------------------------

# Failed to parse test_random_is_not_none_and_not_instance_of_random_raises_type_error.




# Parsed testcases at query #21
#--------------------------

# Partially parsed test_auto_register_provider. Retrieved 3/6 statements.
# Partially parsed test_auto_register_false. Retrieved 3/6 statements.
# Partially parsed test_auto_register_no_meta. Retrieved 1/4 statements.
# Partially parsed test_auto_register_no_name. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'test_provider'
    var_1 = True
    var_2 = 'test_provider'

def test_case_0():
    var_0 = 'test_provider_false'
    var_1 = False
    var_2 = 'test_provider_false'

def test_case_0():
    var_0 = 'TestProvider'

def test_case_0():
    var_0 = True
    var_1 = ''



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_validate_enum_with_none_item. Retrieved 2/5 statements.
# Partially parsed test_validate_enum_with_valid_enum_item. Retrieved 1/5 statements.
# Partially parsed test_validate_enum_raises_non_enumerable_error. Retrieved 2/7 statements.



def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = None


def test_case_0():
    var_0 = module_0.BaseProvider()


def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = 'invalid_item'
    var_2 = bool(False)
    assert var_2 is True
    var_3 = bool(True)
    assert var_3 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_validate_enum_with_item_and_isinstance_true. Retrieved 4/8 statements.



def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 42
    var_3 = module_0.BaseProvider(seed=var_2)



# Parsed testcases at query #24
#--------------------------

# Failed to parse test_random_is_not_none_and_not_instance_of_random_raises_type_error.




# Parsed testcases at query #25
#--------------------------

# Partially parsed test_base_data_provider_init_default_locale. Retrieved 2/3 statements.
# Failed to parse test_base_data_provider_init_custom_locale.
# Failed to parse test_base_data_provider_init_locale_setup_called.
# Partially parsed test_base_data_provider_init_load_dataset_called. Retrieved 2/6 statements.
# Partially parsed test_base_data_provider_init_with_args_and_kwargs. Retrieved 2/4 statements.
# Partially parsed test_base_data_provider_init_locale_order_matters. Retrieved 1/3 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    var_3 = var_1._dataset


def test_case_0():
    var_0 = 42
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.seed
    assert var_3 == 42

import mimesis.random as module_0


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
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = 'random'
    var_3 = hasattr(var_1, var_2)
    var_4 = bool(var_3)
    assert var_4 is True
    var_5 = 'seed'
    var_6 = hasattr(var_1, var_5)
    var_7 = bool(var_6)
    assert var_7 is True

def test_case_0():
    var_0 = 'test'
    var_1 = 'test.json'


def test_case_0():
    var_0 = 'unsupported'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 123
    var_1 = None

def test_case_0():
    var_0 = 999



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_validate_enum_with_none_item. Retrieved 3/6 statements.
# Partially parsed test_validate_enum_with_valid_enum_item. Retrieved 2/6 statements.
# Partially parsed test_validate_enum_raises_non_enumerable_error. Retrieved 3/8 statements.



def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = None


def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)


def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = 'invalid'
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_constructor_with_default_parameters. Retrieved 2/4 statements.
# Partially parsed test_constructor_with_seed_parameter. Retrieved 3/5 statements.
# Partially parsed test_constructor_with_seed_none. Retrieved 3/5 statements.



def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.seed
    var_2 = var_0.random

import mimesis.random as module_0


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


def test_case_0():
    var_0 = None
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed
    assert var_2 is None
    var_3 = var_1.random

import mimesis.random as module_0


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 100
    var_2 = module_1.BaseProvider(seed=var_1, random=var_0)
    var_3 = var_2.random
    var_4 = bool(var_2.random is var_0)
    assert var_4 is True
    var_5 = var_2.seed
    assert var_5 == 100

import mimesis.providers.base as module_0


def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0.BaseProvider(random=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'The random must be an instance of mimesis.random.Random'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_random_is_none_so_self_random_is_initialized_with_new_random_instance. Retrieved 2/4 statements.



def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.random



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_base_data_provider_init_default_locale. Retrieved 2/3 statements.
# Failed to parse test_base_data_provider_init_custom_locale.
# Partially parsed test_base_data_provider_init_dataset_loaded. Retrieved 2/9 statements.
# Partially parsed test_base_data_provider_init_no_datafile. Retrieved 1/4 statements.
# Partially parsed test_base_data_provider_init_locale_separator. Retrieved 3/10 statements.
# Failed to parse test_base_data_provider_init_inheritance.



def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    var_3 = var_1._dataset


def test_case_0():
    var_0 = 42
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.seed
    assert var_3 == 42

import mimesis.random as module_0


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
    var_4 = 'The random must be an instance of mimesis.random.Random'


def test_case_0():
    var_0 = 'invalid_locale'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)

def test_case_0():
    var_0 = 'test'
    var_1 = 'test.json'

def test_case_0():
    var_0 = 'test'

def test_case_0():
    var_0 = 'test'
    var_1 = 'test.json'
    var_2 = 'en_US'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_validate_enum_when_item_is_falsy_but_not_none_and_not_instance_of_enum. Retrieved 4/8 statements.



def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.BaseProvider()
    var_3 = False
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_base_data_provider_initialization_with_default_locale. Retrieved 2/3 statements.
# Failed to parse test_base_data_provider_initialization_with_custom_locale.
# Partially parsed test_base_data_provider_initialization_locale_and_seed. Retrieved 1/3 statements.
# Partially parsed test_base_data_provider_initialization_without_datafile. Retrieved 2/5 statements.
# Partially parsed test_base_data_provider_initialization_with_datafile. Retrieved 3/8 statements.
# Partially parsed test_base_data_provider_initialization_inherits_random_from_base. Retrieved 2/4 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    var_3 = var_1._dataset
    var_4 = var_1.seed


def test_case_0():
    var_0 = 12345
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.seed
    assert var_3 == 12345

import mimesis.random as module_0


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
    var_4 = 'The random must be an instance of mimesis.random.Random'

def test_case_0():
    var_0 = 42


def test_case_0():
    var_0 = 'unsupported'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)

def test_case_0():
    var_0 = 'test'
    var_1 = False

def test_case_0():
    var_0 = 'test'
    var_1 = False
    var_2 = 'test.json'


def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.random


def test_case_0():
    var_0 = 999
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.random._seed
    assert var_3 == 999



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_auto_register_provider. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'test_provider'
    var_1 = True
    var_2 = 'test_provider'



# Parsed testcases at query #33
#--------------------------

# Failed to parse test_provider_registry_constructor.




# Parsed testcases at query #34
#--------------------------

# Partially parsed test_auto_register_false_prevents_registration. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'test_provider'
    var_1 = False
    var_2 = 'test_provider'



# Parsed testcases at query #35
#--------------------------

# Failed to parse test_locale_is_not_default_when_initialized_with_custom_locale.




# Parsed testcases at query #36
#--------------------------

# Partially parsed test_initialization_with_keyword_only_arguments. Retrieved 3/5 statements.



def test_case_0():
    var_0 = None
    var_1 = module_0.BaseProvider(seed=var_0, random=var_0)
    var_2 = var_1.seed
    assert var_2 is None
    var_3 = var_1.random



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_reseed_with_global_seed_set. Retrieved 3/8 statements.



def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider()
    var_2 = var_1.reseed()
    var_3 = bool(var_0 is not None)
    assert var_3 is True



# Parsed testcases at query #38
#--------------------------

# Failed to parse test_constructor_initializes_empty_providers.




# Parsed testcases at query #39
#--------------------------




import mimesis.random as module_0


def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseProvider(random=var_0)
    var_2 = var_1.random
    var_3 = bool(var_1.random is var_0)
    assert var_3 is True



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_random_is_none. Retrieved 3/5 statements.



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


def test_case_0():
    var_0 = 'not_a_random_instance'
    var_1 = module_0.BaseProvider(random=var_0)



# Parsed testcases at query #41
#--------------------------

# Failed to parse test_provider_registry_initialization.




# Parsed testcases at query #42
#--------------------------

# Partially parsed test_reseed_with_global_seed_set. Retrieved 3/4 statements.



def test_case_0():
    var_0 = None
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.reseed()
    var_3 = var_1.seed
    var_4 = var_1.random._seed
    assert var_4 == 12345



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_auto_register_provider. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'test_provider'
    var_1 = True
    var_2 = 'test_provider'



# Parsed testcases at query #44
#--------------------------

# Failed to parse test_random_is_not_none_and_not_instance_of_random_raises_type_error.




# Parsed testcases at query #45
#--------------------------

# Partially parsed test_reseed_uses_global_seed_when_seed_is_missing_seed_and_global_seed_is_not_missing_seed. Retrieved 2/12 statements.


import mimesis.random as module_0


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 99999



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_init_calls_super_with_seed_and_args. Retrieved 6/9 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = 'test.json'
    var_2 = '/tmp'
    var_3 = 'en'
    var_4 = 42
    var_5 = 'value'



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_init_without_random. Retrieved 2/4 statements.



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


def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0.BaseProvider(random=var_0)
    var_2 = 'The random must be an instance of mimesis.random.Random'


def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 42


def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.seed


def test_case_0():
    var_0 = None
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed
    assert var_2 is None


def test_case_0():
    var_0 = 123
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.random._seed
    assert var_2 == 123



# Parsed testcases at query #48
#--------------------------

# Failed to parse test_provider_registry_constructor.




# Parsed testcases at query #49
#--------------------------

# Partially parsed test_base_data_provider_initialization_with_default_locale. Retrieved 2/4 statements.
# Failed to parse test_base_data_provider_initialization_with_custom_locale.
# Partially parsed test_base_data_provider_initialization_with_custom_seed. Retrieved 3/5 statements.
# Partially parsed test_base_data_provider_initialization_with_custom_locale_and_seed. Retrieved 1/6 statements.
# Partially parsed test_base_data_provider_initialization_with_custom_locale_and_random. Retrieved 1/3 statements.
# Partially parsed test_base_data_provider_initialization_with_all_custom_parameters. Retrieved 2/4 statements.
# Partially parsed test_base_data_provider_initialization_with_locale_as_string. Retrieved 3/5 statements.
# Failed to parse test_base_data_provider_initialization_with_locale_as_locale_enum.



def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    var_3 = var_1._dataset
    var_4 = bool(var_1._dataset == {})
    assert var_4 is True
    var_5 = var_1.random
    var_6 = var_1.seed


def test_case_0():
    var_0 = 42
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.locale
    var_4 = var_2._dataset
    var_5 = bool(var_2._dataset == {})
    assert var_5 is True
    var_6 = var_2.random
    var_7 = var_2.seed
    assert var_7 == 42

import mimesis.random as module_0


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_1.BaseDataProvider(**var_2)
    var_4 = var_3.locale
    var_5 = var_3._dataset
    var_6 = bool(var_3._dataset == {})
    assert var_6 is True
    var_7 = var_3.random
    var_8 = bool(var_3.random is var_0)
    assert var_8 is True
    var_9 = var_3.seed

def test_case_0():
    var_0 = 123


def test_case_0():
    var_0 = module_0.Random()


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 999
    var_2 = 'random'
    var_3 = {var_2: var_0}
    var_4 = module_1.BaseDataProvider(seed=var_1, **var_3)
    var_5 = var_4.locale
    var_6 = var_4._dataset
    var_7 = bool(var_4._dataset == {})
    assert var_7 is True
    var_8 = var_4.random
    var_9 = bool(var_4.random is var_0)
    assert var_9 is True
    var_10 = var_4.seed
    assert var_10 == 999


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 777

import mimesis.providers.base as module_0


def test_case_0():
    var_0 = 'invalid'
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_0.BaseDataProvider(**var_2)
    var_4 = bool(False)
    assert var_4 is True


def test_case_0():
    var_0 = 'en'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)
    var_3 = var_2.locale
    var_4 = var_2._dataset
    var_5 = bool(var_2._dataset == {})
    assert var_5 is True
    var_6 = var_2.random
    var_7 = var_2.seed



# Parsed testcases at query #50
#--------------------------





def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.seed



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_validate_enum_with_none_item. Retrieved 2/5 statements.
# Partially parsed test_validate_enum_with_valid_enum_item. Retrieved 1/5 statements.
# Partially parsed test_validate_enum_raises_non_enumerable_error. Retrieved 2/7 statements.



def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = None


def test_case_0():
    var_0 = module_0.BaseProvider()


def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = 'invalid'
    var_2 = bool(False)
    assert var_2 is True
    var_3 = bool(True)
    assert var_3 is True



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_auto_register_provider_with_meta_name_and_auto_register_true. Retrieved 3/6 statements.
# Partially parsed test_auto_register_provider_with_meta_name_and_auto_register_default. Retrieved 2/5 statements.
# Partially parsed test_auto_register_provider_with_meta_name_and_auto_register_explicitly_true. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'test_provider'
    var_1 = True
    var_2 = 'test_provider'

def test_case_0():
    var_0 = 'test_provider_default'
    var_1 = 'test_provider_default'

def test_case_0():
    var_0 = 'test_provider_explicit_true'
    var_1 = True
    var_2 = 'test_provider_explicit_true'



# Parsed testcases at query #53
#--------------------------

# Failed to parse test_locale_is_not_default_when_initialized_with_specific_locale.




# Parsed testcases at query #54
#--------------------------

# Failed to parse test_provider_registry_constructor.




# Parsed testcases at query #55
#--------------------------

# Partially parsed test_auto_register_false_prevents_registration. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'test_provider'
    var_1 = False
    var_2 = 'test_provider'



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_locale_default_is_used_when_no_locale_provided. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = 'test.json'



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_auto_register_provider. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'test_provider'
    var_1 = True
    var_2 = 'test_provider'



# Parsed testcases at query #58
#--------------------------

# Failed to parse test_random_is_not_none_and_not_instance_of_random_raises_type_error.




# Parsed testcases at query #59
#--------------------------





def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_validate_enum_with_item_and_isinstance_true. Retrieved 4/8 statements.



def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 42
    var_3 = module_0.BaseProvider(seed=var_2)



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_validate_enum_item_false_predicate. Retrieved 4/8 statements.



def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 42
    var_3 = module_0.BaseProvider(seed=var_2)



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_validate_enum_with_false_predicate_at_line_11. Retrieved 4/8 statements.



def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 42
    var_3 = module_0.BaseProvider(seed=var_2)



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_base_provider_initialization_with_default_seed. Retrieved 2/4 statements.
# Partially parsed test_base_provider_initialization_with_none_seed. Retrieved 3/5 statements.
# Partially parsed test_base_provider_initialization_with_integer_seed. Retrieved 3/5 statements.



def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.seed
    var_2 = var_0.random


def test_case_0():
    var_0 = None
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed
    assert var_2 is None
    var_3 = var_1.random


def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 42
    var_3 = var_1.random

import mimesis.random as module_0


def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseProvider(random=var_0)
    var_2 = var_1.random
    var_3 = bool(var_1.random is var_0)
    assert var_3 is True
    var_4 = var_1.seed


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 123
    var_2 = module_1.BaseProvider(seed=var_1, random=var_0)
    var_3 = var_2.random
    var_4 = bool(var_2.random is var_0)
    assert var_4 is True
    var_5 = var_2.seed
    assert var_5 == 123

import mimesis.providers.base as module_0


def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0.BaseProvider(random=var_0)
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_auto_register_provider. Retrieved 3/6 statements.
# Partially parsed test_auto_register_false. Retrieved 3/7 statements.
# Partially parsed test_auto_register_no_meta_name. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'test_provider'
    var_1 = True
    var_2 = 'test_provider'

def test_case_0():
    var_0 = 'test_provider_no_register'
    var_1 = False
    var_2 = 'test_provider_no_register'

def test_case_0():
    var_0 = 'name'



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Failed to parse test_provider_registry_constructor.




# Parsed testcases at query #2
#--------------------------

# Failed to parse test_provider_registry_constructor.




# Parsed testcases at query #3
#--------------------------

# Failed to parse test_provider_registry_constructor.




# Parsed testcases at query #4
#--------------------------

# Failed to parse test_provider_registry_constructor.




# Parsed testcases at query #5
#--------------------------

# Partially parsed test_base_data_provider_initialization_with_default_locale. Retrieved 2/4 statements.
# Failed to parse test_base_data_provider_initialization_with_custom_locale.
# Partially parsed test_base_data_provider_initialization_with_seed. Retrieved 3/5 statements.
# Partially parsed test_base_data_provider_initialization_with_locale_and_seed. Retrieved 1/6 statements.
# Partially parsed test_base_data_provider_initialization_with_datafile_meta_loads_dataset. Retrieved 2/9 statements.
# Partially parsed test_base_data_provider_initialization_without_datafile_meta_has_empty_dataset. Retrieved 1/4 statements.
# Failed to parse test_base_data_provider_str_representation.



def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    var_3 = var_1._dataset
    var_4 = bool(var_1._dataset == {})
    assert var_4 is True
    var_5 = var_1.random
    var_6 = var_1.seed


def test_case_0():
    var_0 = 12345
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.locale
    var_4 = var_2._dataset
    var_5 = bool(var_2._dataset == {})
    assert var_5 is True
    var_6 = var_2.random
    var_7 = var_2.seed
    assert var_7 == 12345

import mimesis.random as module_0


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_1.BaseDataProvider(**var_2)
    var_4 = var_3.locale
    var_5 = var_3._dataset
    var_6 = bool(var_3._dataset == {})
    assert var_6 is True
    var_7 = var_3.random
    var_8 = bool(var_3.random is var_0)
    assert var_8 is True
    var_9 = var_3.seed

def test_case_0():
    var_0 = 999

import mimesis.providers.base as module_0


def test_case_0():
    var_0 = 'unsupported'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True


def test_case_0():
    var_0 = 'invalid'
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_0.BaseDataProvider(**var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

def test_case_0():
    var_0 = 'test'
    var_1 = 'test.json'

def test_case_0():
    var_0 = 'test'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_init_with_default_parameters. Retrieved 2/4 statements.
# Partially parsed test_init_with_seed. Retrieved 3/5 statements.
# Partially parsed test_init_with_seed_none. Retrieved 3/5 statements.



def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.seed
    var_2 = var_0.random

import mimesis.random as module_0


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

import mimesis.random as module_0


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 123
    var_2 = module_1.BaseProvider(seed=var_1, random=var_0)
    var_3 = var_2.random
    var_4 = bool(var_2.random is var_0)
    assert var_4 is True
    var_5 = var_2.seed
    assert var_5 == 123

import mimesis.providers.base as module_0


def test_case_0():
    var_0 = None
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed
    assert var_2 is None
    var_3 = var_1.random


def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0.BaseProvider(random=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'The random must be an instance of mimesis.random.Random'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_initialize_attributes_for_data_providers. Retrieved 2/4 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1._dataset
    var_3 = bool(var_1._dataset == {})
    assert var_3 is True
    var_4 = var_1.locale
    var_5 = var_1.random



# Parsed testcases at query #8
#--------------------------

# Failed to parse test_provider_registry_constructor.




# Parsed testcases at query #9
#--------------------------

# Failed to parse test_provider_registry_constructor.




# Parsed testcases at query #10
#--------------------------

# Failed to parse test_provider_registry_constructor.




# Parsed testcases at query #11
#--------------------------

# Partially parsed test_reseed_with_missing_seed_and_global_seed_missing. Retrieved 2/4 statements.
# Partially parsed test_reseed_with_missing_seed_and_global_seed_set. Retrieved 2/4 statements.
# Partially parsed test_reseed_changes_random_state. Retrieved 5/7 statements.



def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.reseed()
    var_2 = var_0.seed


def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.reseed()
    var_2 = var_0.seed


def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = None
    var_2 = var_0.reseed(var_1)
    var_3 = var_0.seed
    assert var_3 is None


def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = 42
    var_2 = var_0.reseed(var_1)
    var_3 = var_0.seed
    assert var_3 == 42


def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = 1
    var_2 = 100
    var_3 = 999
    var_4 = var_0.reseed(var_3)


def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = 100
    var_2 = var_0.reseed(var_1)
    var_3 = 5
    var_4 = range(var_3)
    var_5 = 1
    var_6 = [var_0.random.randint(var_5, var_1) for _ in var_4]
    var_7 = module_0.BaseProvider()
    var_8 = var_7.reseed(var_1)
    var_9 = range(var_3)
    var_10 = [var_7.random.randint(var_5, var_1) for _ in var_9]
    var_11 = bool(var_6 == var_10)
    assert var_11 is True


def test_case_0():
    var_0 = 10
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 10
    var_3 = 20
    var_4 = var_1.reseed(var_3)
    var_5 = var_1.seed
    assert var_5 == 20



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_validate_enum_with_none_item. Retrieved 3/6 statements.
# Partially parsed test_validate_enum_with_valid_enum_item. Retrieved 2/6 statements.
# Partially parsed test_validate_enum_raises_non_enumerable_error. Retrieved 3/8 statements.



def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = None


def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)


def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = 'invalid'
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_reseed_with_missing_seed_and_global_seed_not_missing. Retrieved 3/9 statements.



def test_case_0():
    var_0 = None
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.reseed()
    var_3 = var_1.seed
    assert var_3 is None



# Parsed testcases at query #14
#--------------------------




import mimesis.random as module_0


def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseProvider(random=var_0)
    var_2 = var_1.random
    var_3 = bool(var_1.random is var_0)
    assert var_3 is True



# Parsed testcases at query #15
#--------------------------

# Failed to parse test_provider_registry_constructor.




# Parsed testcases at query #16
#--------------------------

# Failed to parse test_provider_registry_constructor.




# Parsed testcases at query #17
#--------------------------

# Partially parsed test_base_data_provider_initialization_with_default_locale. Retrieved 2/3 statements.
# Failed to parse test_base_data_provider_initialization_with_custom_locale.
# Partially parsed test_base_data_provider_initialization_with_datafile_meta. Retrieved 2/6 statements.
# Partially parsed test_base_data_provider_initialization_without_datafile_meta. Retrieved 1/4 statements.
# Failed to parse test_base_data_provider_initialization_with_composite_locale.
# Partially parsed test_base_data_provider_initialization_inherits_random_from_base. Retrieved 2/4 statements.


import mimesis.providers.base as module_0


def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    var_3 = var_1._dataset
    var_4 = var_1.seed


def test_case_0():
    var_0 = 12345
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.seed
    assert var_3 == 12345

import mimesis.random as module_0


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
    var_5 = bool(True)
    assert var_5 is True


def test_case_0():
    var_0 = 'invalid_locale'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'test'
    var_1 = 'test.json'

def test_case_0():
    var_0 = 'test'


def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.random


def test_case_0():
    var_0 = 999
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.random._seed
    assert var_3 == 999



# Parsed testcases at query #18
#--------------------------

# Failed to parse test_provider_registry_constructor.




# Parsed testcases at query #19
#--------------------------

# Partially parsed test_auto_register_false_prevents_registration. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'test_provider'
    var_1 = False
    var_2 = 'test_provider'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_constructor_initializes_empty_providers. Retrieved 1/2 statements.


def test_case_0():
    var_0 = {}



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_auto_register_provider_when_meta_name_and_auto_register_true. Retrieved 3/6 statements.
# Partially parsed test_auto_register_provider_when_meta_name_and_auto_register_default. Retrieved 2/5 statements.
# Partially parsed test_auto_register_provider_when_meta_name_and_auto_register_false. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'test_provider'
    var_1 = True
    var_2 = 'test_provider'

def test_case_0():
    var_0 = 'test_provider_default'
    var_1 = 'test_provider_default'

def test_case_0():
    var_0 = 'test_provider_false'
    var_1 = False
    var_2 = 'test_provider_false'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_reseed_global_seed_not_missing_seed. Retrieved 3/4 statements.



def test_case_0():
    var_0 = None
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.reseed()
    var_3 = var_1.seed
    var_4 = var_1.random._seed
    assert var_4 == 42



# Parsed testcases at query #23
#--------------------------




import mimesis.random as module_0


def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseProvider(random=var_0)
    var_2 = var_1.random
    var_3 = bool(var_1.random is var_0)
    assert var_3 is True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_auto_register_provider. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'test_provider'
    var_1 = True
    var_2 = 'test_provider'



# Parsed testcases at query #25
#--------------------------




import mimesis.providers.base as module_0


def test_case_0():
    var_0 = None
    var_1 = module_0.BaseProvider(random=var_0)



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_init_with_keyword_only_arguments. Retrieved 4/6 statements.



def test_case_0():
    var_0 = 42
    var_1 = None
    var_2 = module_0.BaseProvider(seed=var_0, random=var_1)
    var_3 = var_2.seed
    assert var_3 == 42
    var_4 = var_2.random



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_base_data_provider_initialization_with_default_locale. Retrieved 2/3 statements.
# Failed to parse test_base_data_provider_initialization_with_custom_locale.
# Partially parsed test_base_data_provider_initialization_without_datafile. Retrieved 2/5 statements.
# Partially parsed test_base_data_provider_initialization_with_datafile. Retrieved 3/10 statements.
# Partially parsed test_base_data_provider_initialization_with_composite_locale. Retrieved 3/10 statements.
# Partially parsed test_base_data_provider_initialization_inherits_random_from_base. Retrieved 4/6 statements.
# Partially parsed test_base_data_provider_initialization_dataset_attribute_exists. Retrieved 4/5 statements.
# Partially parsed test_base_data_provider_initialization_with_kwargs. Retrieved 1/3 statements.
# Partially parsed test_base_data_provider_initialization_with_args_ignored. Retrieved 1/3 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    var_3 = var_1._dataset
    var_4 = var_1.seed


def test_case_0():
    var_0 = 42
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.seed
    assert var_3 == 42

import mimesis.random as module_0


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
    var_4 = 'The random must be an instance of mimesis.random.Random'


def test_case_0():
    var_0 = 'invalid_locale'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)

def test_case_0():
    var_0 = 'test'
    var_1 = False

def test_case_0():
    var_0 = 'test'
    var_1 = False
    var_2 = 'test.json'

def test_case_0():
    var_0 = 'test'
    var_1 = False
    var_2 = 'test.json'


def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = 'random'
    var_3 = hasattr(var_1, var_2)
    var_4 = bool(var_3)
    assert var_4 is True
    var_5 = var_1.random


def test_case_0():
    var_0 = 123
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.seed
    assert var_3 == 123
    var_4 = var_2.random._seed
    assert var_4 == 123


def test_case_0():
    var_0 = None
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.seed
    assert var_3 is None


def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.seed


def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = 'locale'
    var_3 = hasattr(var_1, var_2)
    var_4 = bool(var_3)
    assert var_4 is True
    var_5 = var_1.locale


def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = '_dataset'
    var_3 = hasattr(var_1, var_2)
    var_4 = bool(var_3)
    assert var_4 is True
    var_5 = var_1._dataset

def test_case_0():
    var_0 = 999

def test_case_0():
    var_0 = 777



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_validate_enum_with_false_predicate_at_line_11. Retrieved 4/8 statements.



def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 42
    var_3 = module_0.BaseProvider(seed=var_2)



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_base_data_provider_initialization_with_default_locale. Retrieved 2/3 statements.
# Failed to parse test_base_data_provider_initialization_with_custom_locale.
# Partially parsed test_base_data_provider_initialization_with_locale_and_seed. Retrieved 1/3 statements.
# Partially parsed test_base_data_provider_initialization_without_datafile. Retrieved 2/5 statements.
# Failed to parse test_base_data_provider_str_representation.
# Partially parsed test_base_data_provider_inherits_from_base_provider. Retrieved 1/2 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    var_3 = var_1._dataset
    var_4 = var_1.seed


def test_case_0():
    var_0 = 42
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.seed
    assert var_3 == 42

import mimesis.random as module_0


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
    var_4 = 'The random must be an instance of mimesis.random.Random'

def test_case_0():
    var_0 = 123

def test_case_0():
    var_0 = 'test'
    var_1 = False


def test_case_0():
    var_0 = 'unsupported'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)


def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)



# Parsed testcases at query #30
#--------------------------

# Failed to parse test_locale_default_does_not_trigger_unsupported_locale.




# Parsed testcases at query #31
#--------------------------

# Failed to parse test_provider_registry_constructor.




# Parsed testcases at query #32
#--------------------------

# Partially parsed test_validate_enum_with_false_predicate_at_line_11. Retrieved 4/8 statements.



def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 42
    var_3 = module_0.BaseProvider(seed=var_2)



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_reseed_with_missing_seed_and_global_seed_missing. Retrieved 2/4 statements.
# Partially parsed test_reseed_with_missing_seed_and_global_seed_set. Retrieved 2/6 statements.
# Partially parsed test_reseed_with_explicit_seed. Retrieved 3/5 statements.
# Partially parsed test_reseed_with_none_seed. Retrieved 3/5 statements.
# Partially parsed test_reseed_with_same_seed_produces_same_random_state. Retrieved 5/7 statements.
# Partially parsed test_reseed_with_different_seed_produces_different_random_state. Retrieved 6/8 statements.



def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.reseed()
    var_2 = var_0.seed


def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.reseed()
    var_2 = var_0.seed


def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = 999
    var_2 = var_0.reseed(var_1)
    var_3 = var_0.seed
    assert var_3 == 999


def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = None
    var_2 = var_0.reseed(var_1)
    var_3 = var_0.seed
    assert var_3 is None


def test_case_0():
    var_0 = 100
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = 200
    var_3 = var_1.reseed(var_2)
    var_4 = var_1.seed
    assert var_4 == 200


def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = module_0.BaseProvider()
    var_2 = 555
    var_3 = var_0.reseed(var_2)
    var_4 = var_1.reseed(var_2)


def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = module_0.BaseProvider()
    var_2 = 111
    var_3 = var_0.reseed(var_2)
    var_4 = 222
    var_5 = var_1.reseed(var_4)



# Parsed testcases at query #34
#--------------------------




import mimesis.random as module_0


def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseProvider(random=var_0)



# Parsed testcases at query #35
#--------------------------





def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseProvider(random=var_0)
    var_2 = var_1.random
    var_3 = bool(var_1.random is var_0)
    assert var_3 is True



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_base_data_provider_init_default_locale. Retrieved 2/4 statements.
# Failed to parse test_base_data_provider_init_custom_locale.
# Partially parsed test_base_data_provider_init_with_seed. Retrieved 3/5 statements.
# Partially parsed test_base_data_provider_init_locale_dependent_data_loading. Retrieved 2/8 statements.
# Failed to parse test_base_data_provider_init_inheritance.
# Failed to parse test_base_data_provider_str_representation.


import mimesis.providers.base as module_0


def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    var_3 = var_1._dataset
    var_4 = bool(var_1._dataset == {})
    assert var_4 is True
    var_5 = var_1.random


def test_case_0():
    var_0 = 12345
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.seed
    assert var_3 == 12345
    var_4 = var_2.random

import mimesis.random as module_0


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
    var_5 = bool(True)
    assert var_5 is True

def test_case_0():
    var_0 = 'test'
    var_1 = 'test.json'
    var_2 = bool(True)
    assert var_2 is True


def test_case_0():
    var_0 = 'unsupported'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True


def test_case_0():
    var_0 = 12345
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_base_data_provider_init_default_locale. Retrieved 2/4 statements.
# Failed to parse test_base_data_provider_init_custom_locale.
# Partially parsed test_base_data_provider_init_with_seed. Retrieved 3/5 statements.
# Partially parsed test_base_data_provider_init_locale_dependent_data_loading. Retrieved 2/9 statements.
# Partially parsed test_base_data_provider_init_with_args_and_kwargs. Retrieved 2/4 statements.
# Partially parsed test_base_data_provider_init_locale_separator_handling. Retrieved 3/4 statements.
# Partially parsed test_base_data_provider_init_no_datafile. Retrieved 1/4 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    var_3 = var_1._dataset
    var_4 = bool(var_1._dataset == {})
    assert var_4 is True
    var_5 = var_1.random


def test_case_0():
    var_0 = 42
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.seed
    assert var_3 == 42
    var_4 = var_2.random

import mimesis.random as module_0


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
    var_4 = 'The random must be an instance of mimesis.random.Random'

def test_case_0():
    var_0 = 'test'
    var_1 = 'test.json'


def test_case_0():
    var_0 = 'unsupported'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)

def test_case_0():
    var_0 = 123
    var_1 = 'test'


def test_case_0():
    var_0 = 'en-US'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'en-US'
    var_4 = var_2._dataset

def test_case_0():
    var_0 = 'nodata'



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_init_with_default_parameters. Retrieved 2/4 statements.
# Partially parsed test_init_with_seed. Retrieved 3/5 statements.
# Partially parsed test_init_with_seed_none. Retrieved 3/5 statements.



def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.seed
    var_2 = var_0.random

import mimesis.random as module_0


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
    var_3 = 'The random must be an instance of mimesis.random.Random'


def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 42
    var_3 = var_1.random


def test_case_0():
    var_0 = None
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed
    assert var_2 is None
    var_3 = var_1.random

import mimesis.random as module_0


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 123
    var_2 = module_1.BaseProvider(seed=var_1, random=var_0)
    var_3 = var_2.seed
    assert var_3 == 123
    var_4 = var_2.random
    var_5 = bool(var_2.random is var_0)
    assert var_5 is True



# Parsed testcases at query #39
#--------------------------

# Failed to parse test_base_data_provider_init_with_custom_locale.
# Failed to parse test_base_data_provider_init_locale_setup_called.
# Partially parsed test_base_data_provider_init_load_dataset_called. Retrieved 2/7 statements.
# Partially parsed test_base_data_provider_init_with_args_and_kwargs. Retrieved 1/3 statements.


import mimesis.providers.base as module_0


def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    var_3 = var_1._dataset
    var_4 = bool(var_1._dataset == {})
    assert var_4 is True


def test_case_0():
    var_0 = 12345
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.seed
    assert var_3 == 12345

import mimesis.random as module_0


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
    var_4 = 'The random must be an instance of mimesis.random.Random'

def test_case_0():
    var_0 = 'test'
    var_1 = 'test.json'


def test_case_0():
    var_0 = 'unsupported'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)


def test_case_0():
    var_0 = 999
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.seed
    assert var_3 == 999


def test_case_0():
    var_0 = None
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.seed
    assert var_3 is None


def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.seed

def test_case_0():
    var_0 = 42



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_base_data_provider_init_default_locale. Retrieved 2/3 statements.
# Failed to parse test_base_data_provider_init_custom_locale.
# Partially parsed test_base_data_provider_init_dataset_loaded. Retrieved 2/9 statements.
# Partially parsed test_base_data_provider_init_no_datafile. Retrieved 1/4 statements.
# Failed to parse test_base_data_provider_init_locale_with_separator.



def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    var_3 = var_1._dataset
    var_4 = var_1.seed


def test_case_0():
    var_0 = 42
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.seed
    assert var_3 == 42

import mimesis.random as module_0


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
    var_0 = 'invalid_locale'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = 'random'
    var_3 = hasattr(var_1, var_2)
    var_4 = bool(var_3)
    assert var_4 is True
    var_5 = 'seed'
    var_6 = hasattr(var_1, var_5)
    var_7 = bool(var_6)
    assert var_7 is True

def test_case_0():
    var_0 = 'test'
    var_1 = 'test.json'
    var_2 = bool(True)
    assert var_2 is True

def test_case_0():
    var_0 = 'test'

import mimesis.random as module_0


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 123
    var_2 = 'random'
    var_3 = {var_2: var_0}
    var_4 = module_1.BaseDataProvider(seed=var_1, **var_3)
    var_5 = var_4.seed
    assert var_5 == 123
    var_6 = var_4.random
    var_7 = bool(var_4.random is var_0)
    assert var_7 is True



# Parsed testcases at query #41
#--------------------------

# Failed to parse test_random_is_not_none_and_is_not_instance_of_random_raises_type_error.



def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseProvider(random=var_0)
    var_2 = var_1.random
    var_3 = bool(var_1.random is var_0)
    assert var_3 is True



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_base_data_provider_initialization_with_default_locale. Retrieved 2/3 statements.
# Failed to parse test_base_data_provider_initialization_with_custom_locale.
# Partially parsed test_base_data_provider_initialization_with_locale_and_seed. Retrieved 1/3 statements.
# Partially parsed test_base_data_provider_initialization_dataset_loaded. Retrieved 2/5 statements.
# Partially parsed test_base_data_provider_initialization_inherits_from_base_provider. Retrieved 1/2 statements.
# Partially parsed test_base_data_provider_initialization_seed_passed_to_super. Retrieved 2/3 statements.
# Failed to parse test_base_data_provider_initialization_with_missing_seed.


import mimesis.providers.base as module_0


def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    var_3 = var_1._dataset


def test_case_0():
    var_0 = 42
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.seed
    assert var_3 == 42

import mimesis.random as module_0


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_1.BaseDataProvider(**var_2)
    var_4 = var_3.random
    var_5 = bool(var_3.random is var_0)
    assert var_5 is True

def test_case_0():
    var_0 = 123

import mimesis.providers.base as module_0


def test_case_0():
    var_0 = 'en_US'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'en_US'

def test_case_0():
    var_0 = 'test'
    var_1 = 'test.json'


def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)


def test_case_0():
    var_0 = 999
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)



# Parsed testcases at query #43
#--------------------------




def test_case_0():
    var_0 = 'test_provider'
    var_1 = False
    var_2 = 'test_provider'



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_base_data_provider_init_default_locale. Retrieved 2/3 statements.
# Failed to parse test_base_data_provider_init_custom_locale.
# Partially parsed test_base_data_provider_init_datafile_loading. Retrieved 2/5 statements.
# Partially parsed test_base_data_provider_init_no_datafile. Retrieved 1/4 statements.
# Partially parsed test_base_data_provider_init_inherits_random. Retrieved 4/6 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    var_3 = var_1._dataset


def test_case_0():
    var_0 = 42
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.seed
    assert var_3 == 42

import mimesis.random as module_0


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
    var_0 = 'en_US'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'en_US'


def test_case_0():
    var_0 = 'unsupported'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'test'
    var_1 = 'test.json'

def test_case_0():
    var_0 = 'test'


def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = 'random'
    var_3 = hasattr(var_1, var_2)
    var_4 = bool(var_3)
    assert var_4 is True
    var_5 = var_1.random


def test_case_0():
    var_0 = 123
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.seed
    assert var_3 == 123



# Parsed testcases at query #45
#--------------------------

# Failed to parse test_random_is_not_none_and_is_not_instance_of_random_raises_typeerror.
# Partially parsed test_random_is_none_creates_new_random_instance. Retrieved 3/6 statements.


import mimesis.random as module_0


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



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_auto_register_provider. Retrieved 3/6 statements.
# Partially parsed test_auto_register_false. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'test_provider'
    var_1 = True
    var_2 = 'test_provider'

def test_case_0():
    var_0 = 'test_provider_no_register'
    var_1 = False
    var_2 = 'test_provider_no_register'

def test_case_0():
    var_0 = bool(True)
    assert var_0 is True

def test_case_0():
    var_0 = True
    var_1 = bool(True)
    assert var_1 is True



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_validate_enum_item_is_none. Retrieved 5/8 statements.



def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 42
    var_3 = module_0.BaseProvider(seed=var_2)
    var_4 = None



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_base_data_provider_init_default. Retrieved 2/4 statements.
# Failed to parse test_base_data_provider_init_with_locale.
# Partially parsed test_base_data_provider_init_with_seed. Retrieved 3/5 statements.
# Partially parsed test_base_data_provider_init_with_locale_and_seed. Retrieved 1/6 statements.
# Partially parsed test_base_data_provider_init_with_custom_random_and_locale. Retrieved 1/3 statements.
# Partially parsed test_base_data_provider_init_with_all_parameters. Retrieved 2/4 statements.
# Partially parsed test_base_data_provider_init_inherits_random_from_base. Retrieved 2/4 statements.
# Failed to parse test_base_data_provider_init_locale_as_locale_enum.



def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    var_3 = var_1.seed
    var_4 = var_1.random
    var_5 = var_1._dataset
    var_6 = bool(var_1._dataset == {})
    assert var_6 is True


def test_case_0():
    var_0 = 42
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.locale
    var_4 = var_2.seed
    assert var_4 == 42
    var_5 = var_2.random

def test_case_0():
    var_0 = 123

import mimesis.random as module_0


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_1.BaseDataProvider(**var_2)
    var_4 = var_3.random
    var_5 = bool(var_3.random is var_0)
    assert var_5 is True
    var_6 = var_3.locale
    var_7 = var_3.seed


def test_case_0():
    var_0 = module_0.Random()


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 999
    var_2 = 'random'
    var_3 = {var_2: var_0}
    var_4 = module_1.BaseDataProvider(seed=var_1, **var_3)
    var_5 = var_4.random
    var_6 = bool(var_4.random is var_0)
    assert var_6 is True
    var_7 = var_4.locale
    var_8 = var_4.seed
    assert var_8 == 999


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 777

import mimesis.providers.base as module_0


def test_case_0():
    var_0 = 'unsupported'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.random


def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.seed


def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale


def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1._dataset
    var_3 = bool(var_1._dataset == {})
    assert var_3 is True


def test_case_0():
    var_0 = None
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.seed
    assert var_3 is None


def test_case_0():
    var_0 = 0
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.seed
    assert var_3 == 0


def test_case_0():
    var_0 = -5
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.seed
    assert var_3 == -5


def test_case_0():
    var_0 = 'en'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'en'



# Parsed testcases at query #49
#--------------------------




def test_case_0():
    var_0 = 'test_provider'
    var_1 = False
    var_2 = 'test_provider'



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_validate_enum_with_false_predicate_at_line_11. Retrieved 3/7 statements.



def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.BaseProvider()



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_initialization_with_keyword_only_arguments. Retrieved 3/5 statements.



def test_case_0():
    var_0 = None
    var_1 = module_0.BaseProvider(seed=var_0, random=var_0)
    var_2 = var_1.seed
    assert var_2 is None
    var_3 = var_1.random



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_auto_register_false_prevents_registration. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'test_provider'
    var_1 = False
    var_2 = 'test_provider'



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_auto_register_provider_when_meta_name_and_auto_register_true. Retrieved 3/6 statements.
# Partially parsed test_auto_register_provider_when_meta_name_and_auto_register_default. Retrieved 2/5 statements.
# Partially parsed test_auto_register_provider_when_meta_name_and_auto_register_false. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'test_provider'
    var_1 = True
    var_2 = 'test_provider'

def test_case_0():
    var_0 = 'test_provider_default'
    var_1 = 'test_provider_default'

def test_case_0():
    var_0 = 'test_provider_false'
    var_1 = False
    var_2 = 'test_provider_false'



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_base_provider_initialization_with_defaults. Retrieved 2/4 statements.
# Partially parsed test_base_provider_initialization_with_seed. Retrieved 3/5 statements.
# Partially parsed test_base_provider_initialization_with_none_seed. Retrieved 3/5 statements.



def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.seed
    var_2 = var_0.random

import mimesis.random as module_0


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


def test_case_0():
    var_0 = None
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed
    assert var_2 is None
    var_3 = var_1.random


def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0.BaseProvider(random=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = bool(True)
    assert var_3 is True



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_constructor_with_default_parameters. Retrieved 2/4 statements.
# Partially parsed test_constructor_with_seed. Retrieved 3/5 statements.
# Partially parsed test_constructor_with_seed_none. Retrieved 3/5 statements.
# Failed to parse test_constructor_with_seed_missing_seed.



def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.seed
    var_2 = var_0.random

import mimesis.random as module_0


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

import mimesis.random as module_0


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 42
    var_2 = module_1.BaseProvider(seed=var_1, random=var_0)
    var_3 = var_2.random
    var_4 = bool(var_2.random is var_0)
    assert var_4 is True
    var_5 = var_2.seed
    assert var_5 == 42

import mimesis.providers.base as module_0


def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0.BaseProvider(random=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = bool(True)
    assert var_3 is True


def test_case_0():
    var_0 = None
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed
    assert var_2 is None
    var_3 = var_1.random



# Parsed testcases at query #2
#--------------------------

# Failed to parse test_constructor_initializes_empty_providers.




# Parsed testcases at query #3
#--------------------------

# Failed to parse test_provider_registry_constructor.




# Parsed testcases at query #4
#--------------------------

# Partially parsed test_base_data_provider_init_default_locale. Retrieved 2/4 statements.
# Failed to parse test_base_data_provider_init_custom_locale.
# Partially parsed test_base_data_provider_init_with_seed. Retrieved 3/5 statements.
# Partially parsed test_base_data_provider_init_locale_dependent_data_loading. Retrieved 2/8 statements.
# Partially parsed test_base_data_provider_init_with_args_and_kwargs. Retrieved 1/3 statements.
# Failed to parse test_base_data_provider_str_representation.
# Partially parsed test_base_data_provider_init_subclass_auto_registration. Retrieved 2/5 statements.
# Partially parsed test_base_data_provider_init_subclass_auto_register_false. Retrieved 3/7 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    var_3 = var_1._dataset
    var_4 = bool(var_1._dataset == {})
    assert var_4 is True
    var_5 = var_1.random


def test_case_0():
    var_0 = 42
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.seed
    assert var_3 == 42
    var_4 = var_2.random

import mimesis.random as module_0


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
    var_4 = 'The random must be an instance of mimesis.random.Random'

def test_case_0():
    var_0 = 'test'
    var_1 = 'test.json'


def test_case_0():
    var_0 = 'invalid_locale'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)

def test_case_0():
    var_0 = 123

def test_case_0():
    var_0 = 'test_provider'
    var_1 = 'test_provider'

def test_case_0():
    var_0 = 'test_provider_no_register'
    var_1 = False
    var_2 = 'test_provider_no_register'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_init_without_random. Retrieved 2/4 statements.


import mimesis.random as module_0


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


def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0.BaseProvider(random=var_0)
    var_2 = 'The random must be an instance of mimesis.random.Random'


def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 42


def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.seed


def test_case_0():
    var_0 = None
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed
    assert var_2 is None


def test_case_0():
    var_0 = 123
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 123



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_base_provider_initializes_with_defaults. Retrieved 2/4 statements.
# Failed to parse test_base_provider_initializes_with_missing_seed.



def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.seed
    var_2 = var_0.random

import mimesis.random as module_0


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


def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0.BaseProvider(random=var_0)
    var_2 = 'The random must be an instance of mimesis.random.Random'


def test_case_0():
    var_0 = None
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed
    assert var_2 is None



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_constructor_with_default_parameters. Retrieved 2/4 statements.
# Partially parsed test_constructor_with_seed. Retrieved 3/5 statements.



def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.seed
    var_2 = var_0.random

import mimesis.random as module_0


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

import mimesis.random as module_0


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 123
    var_2 = module_1.BaseProvider(seed=var_1, random=var_0)
    var_3 = var_2.random
    var_4 = bool(var_2.random is var_0)
    assert var_4 is True
    var_5 = var_2.seed
    assert var_5 == 123

import mimesis.providers.base as module_0


def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0.BaseProvider(random=var_0)
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #2
#--------------------------

# Failed to parse test_provider_registry_constructor.




# Parsed testcases at query #3
#--------------------------

# Partially parsed test_base_data_provider_initialization_with_default_locale. Retrieved 2/4 statements.
# Failed to parse test_base_data_provider_initialization_with_custom_locale.
# Partially parsed test_base_data_provider_initialization_with_seed. Retrieved 3/5 statements.
# Partially parsed test_base_data_provider_initialization_with_locale_and_seed. Retrieved 1/6 statements.
# Partially parsed test_base_data_provider_initialization_with_auto_register. Retrieved 3/6 statements.
# Partially parsed test_base_data_provider_initialization_with_auto_register_false. Retrieved 3/7 statements.
# Partially parsed test_base_data_provider_initialization_with_datafile_meta. Retrieved 3/8 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    var_3 = var_1._dataset
    var_4 = bool(var_1._dataset == {})
    assert var_4 is True
    var_5 = var_1.random
    var_6 = var_1.seed


def test_case_0():
    var_0 = 42
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.locale
    var_4 = var_2.seed
    assert var_4 == 42
    var_5 = var_2.random

import mimesis.random as module_0


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_1.BaseDataProvider(**var_2)
    var_4 = var_3.random
    var_5 = bool(var_3.random is var_0)
    assert var_5 is True
    var_6 = var_3.locale
    var_7 = var_3.seed

def test_case_0():
    var_0 = 123

import mimesis.providers.base as module_0


def test_case_0():
    var_0 = 'invalid'
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_0.BaseDataProvider(**var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'The random must be an instance of mimesis.random.Random'

def test_case_0():
    var_0 = 'test_provider'
    var_1 = True
    var_2 = 'test_provider'

def test_case_0():
    var_0 = 'test_provider_no_register'
    var_1 = False
    var_2 = 'test_provider_no_register'
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'test_datafile'
    var_1 = 'test.json'
    var_2 = '/fake/dir'


def test_case_0():
    var_0 = 'invalid_locale'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True



# Parsed testcases at query #4
#--------------------------

# Failed to parse test_provider_registry_constructor.




# Parsed testcases at query #5
#--------------------------

# Partially parsed test_base_data_provider_init_default_locale. Retrieved 2/3 statements.
# Failed to parse test_base_data_provider_init_custom_locale.
# Partially parsed test_base_data_provider_init_inherits_random_from_base. Retrieved 2/4 statements.
# Failed to parse test_base_data_provider_init_locale_validation.
# Partially parsed test_base_data_provider_init_dataset_loaded. Retrieved 2/9 statements.
# Partially parsed test_base_data_provider_init_no_datafile. Retrieved 1/4 statements.
# Partially parsed test_base_data_provider_init_with_args_and_kwargs. Retrieved 1/3 statements.
# Failed to parse test_base_data_provider_init_locale_separator_handling.



def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    var_3 = var_1._dataset
    var_4 = var_1.seed


def test_case_0():
    var_0 = 42
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.seed
    assert var_3 == 42

import mimesis.random as module_0


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
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.random

def test_case_0():
    var_0 = 'test'
    var_1 = 'test.json'

def test_case_0():
    var_0 = 'test'

def test_case_0():
    var_0 = 123



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_auto_register_provider. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'test_provider'
    var_1 = True
    var_2 = 'test_provider'



# Parsed testcases at query #7
#--------------------------




import mimesis.random as module_0


def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseProvider(random=var_0)
    var_2 = var_1.random
    var_3 = bool(var_1.random is var_0)
    assert var_3 is True



# Parsed testcases at query #8
#--------------------------

# Failed to parse test_provider_registry_constructor.




# Parsed testcases at query #9
#--------------------------

# Partially parsed test_reseed_with_missing_seed_and_global_seed_missing. Retrieved 2/4 statements.
# Partially parsed test_reseed_with_missing_seed_and_global_seed_set. Retrieved 4/9 statements.
# Partially parsed test_reseed_with_none_seed. Retrieved 4/6 statements.
# Partially parsed test_reseed_with_integer_seed. Retrieved 4/7 statements.
# Partially parsed test_reseed_with_string_seed. Retrieved 4/7 statements.


import mimesis.providers.base as module_0


def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.reseed()
    var_2 = var_0.seed


def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.reseed()
    var_2 = var_0.seed
    var_3 = 42
    var_4 = var_0.reseed()


def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = None
    var_2 = var_0.reseed(var_1)
    var_3 = var_0.seed
    assert var_3 is None
    var_4 = var_0.reseed(var_1)


def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = 123
    var_2 = var_0.reseed(var_1)
    var_3 = var_0.seed
    assert var_3 == 123
    var_4 = var_0.reseed(var_1)


def test_case_0():
    var_0 = 100
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 100
    var_3 = 200
    var_4 = var_1.reseed(var_3)
    var_5 = var_1.seed
    assert var_5 == 200


def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = 'test_seed'
    var_2 = var_0.reseed(var_1)
    var_3 = var_0.seed
    assert var_3 == 'test_seed'
    var_4 = var_0.reseed(var_1)



# Parsed testcases at query #10
#--------------------------

# Failed to parse test_provider_registry_constructor.




# Parsed testcases at query #11
#--------------------------

# Partially parsed test_auto_register_false_prevents_registration. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'test_provider'
    var_1 = False
    var_2 = 'test_provider'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_init_calls_super_with_seed_and_args_kwargs. Retrieved 6/10 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = 'test.json'
    var_2 = 'en'
    var_3 = 42
    var_4 = 'extra'
    var_5 = 'kwarg'



# Parsed testcases at query #13
#--------------------------





def test_case_0():
    var_0 = None
    var_1 = module_0.BaseProvider(random=var_0)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_constructor_with_default_parameters. Retrieved 2/4 statements.
# Partially parsed test_constructor_with_custom_seed. Retrieved 3/5 statements.
# Partially parsed test_constructor_with_none_seed. Retrieved 3/5 statements.



def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.seed
    var_2 = var_0.random


def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 42
    var_3 = var_1.random


def test_case_0():
    var_0 = None
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed
    assert var_2 is None
    var_3 = var_1.random

import mimesis.random as module_0


def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseProvider(random=var_0)
    var_2 = var_1.random
    var_3 = bool(var_1.random is var_0)
    assert var_3 is True
    var_4 = var_1.seed


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 123
    var_2 = module_1.BaseProvider(seed=var_1, random=var_0)
    var_3 = var_2.random
    var_4 = bool(var_2.random is var_0)
    assert var_4 is True
    var_5 = var_2.seed
    assert var_5 == 123

import mimesis.providers.base as module_0


def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0.BaseProvider(random=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'The random must be an instance of mimesis.random.Random'



# Parsed testcases at query #15
#--------------------------

# Failed to parse test_provider_registry_initialization.




# Parsed testcases at query #16
#--------------------------

# Partially parsed test_base_data_provider_initialization_with_default_locale. Retrieved 2/3 statements.
# Failed to parse test_base_data_provider_initialization_with_custom_locale.
# Failed to parse test_base_data_provider_initialization_locale_setup.
# Partially parsed test_base_data_provider_initialization_dataset_loaded. Retrieved 2/10 statements.
# Partially parsed test_base_data_provider_initialization_without_datafile. Retrieved 1/4 statements.
# Partially parsed test_base_data_provider_initialization_inherits_from_base_provider. Retrieved 9/10 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    var_3 = var_1._dataset
    var_4 = var_1.seed


def test_case_0():
    var_0 = 12345
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.seed
    assert var_3 == 12345

import mimesis.random as module_0


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
    var_5 = 'The random must be an instance of mimesis.random.Random'

def test_case_0():
    var_0 = 'test'
    var_1 = 'test.json'

def test_case_0():
    var_0 = 'test'


def test_case_0():
    var_0 = 'unsupported'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = 'random'
    var_3 = hasattr(var_1, var_2)
    var_4 = bool(var_3)
    assert var_4 is True
    var_5 = 'seed'
    var_6 = hasattr(var_1, var_5)
    var_7 = bool(var_6)
    assert var_7 is True
    var_8 = 'reseed'
    var_9 = hasattr(var_1, var_8)
    var_10 = bool(var_9)
    assert var_10 is True
    var_11 = 'validate_enum'
    var_12 = hasattr(var_1, var_11)
    var_13 = bool(var_12)
    assert var_13 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_reseed_when_seed_is_missing_seed_and_global_seed_is_not_missing_seed. Retrieved 2/7 statements.



def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider()
    var_2 = var_1.random._seed
    assert var_2 == 42



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_validate_enum_with_none_item. Retrieved 3/6 statements.
# Partially parsed test_validate_enum_with_valid_enum_item. Retrieved 2/6 statements.
# Partially parsed test_validate_enum_raises_non_enumerable_error. Retrieved 3/8 statements.



def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = None


def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)


def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = 'invalid_item'
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_constructor_with_default_parameters. Retrieved 2/4 statements.
# Partially parsed test_constructor_with_seed_parameter. Retrieved 3/5 statements.
# Partially parsed test_constructor_with_seed_none. Retrieved 3/5 statements.
# Partially parsed test_constructor_keyword_only_arguments_enforced. Retrieved 1/3 statements.



def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.seed
    var_2 = var_0.random

import mimesis.random as module_0


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


def test_case_0():
    var_0 = None
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed
    assert var_2 is None
    var_3 = var_1.random

import mimesis.random as module_0


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 100
    var_2 = module_1.BaseProvider(seed=var_1, random=var_0)
    var_3 = var_2.random
    var_4 = bool(var_2.random is var_0)
    assert var_4 is True
    var_5 = var_2.seed
    assert var_5 == 100

import mimesis.providers.base as module_0


def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0.BaseProvider(random=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'The random must be an instance of mimesis.random.Random'

def test_case_0():
    var_0 = 'invalid_positional'
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #20
#--------------------------

# Failed to parse test_base_data_provider_initialization_with_custom_locale.
# Partially parsed test_base_data_provider_initialization_loads_dataset. Retrieved 2/7 statements.
# Partially parsed test_base_data_provider_initialization_without_datafile. Retrieved 1/4 statements.
# Partially parsed test_base_data_provider_initialization_with_locale_separator. Retrieved 3/8 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    var_3 = var_1._dataset
    var_4 = bool(var_1._dataset == {})
    assert var_4 is True


def test_case_0():
    var_0 = 42
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.seed
    assert var_3 == 42

import mimesis.random as module_0


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
    var_5 = bool(True)
    assert var_5 is True


def test_case_0():
    var_0 = 'unsupported'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'test'
    var_1 = 'test.json'
    var_2 = bool(False)
    assert var_2 is True
    var_3 = bool(True)
    assert var_3 is True

def test_case_0():
    var_0 = 'test'

def test_case_0():
    var_0 = 'test'
    var_1 = 'test.json'
    var_2 = 'en_US'
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_auto_register_provider. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'test_provider'
    var_1 = True
    var_2 = 'test_provider'



# Parsed testcases at query #22
#--------------------------

# Failed to parse test_provider_registry_constructor.




# Parsed testcases at query #23
#--------------------------

# Partially parsed test_validate_enum_with_false_predicate_at_line_11. Retrieved 4/7 statements.



def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = module_0.BaseProvider()
    var_3 = None



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_base_provider_initialization_with_defaults. Retrieved 2/4 statements.
# Partially parsed test_base_provider_initialization_with_seed. Retrieved 3/5 statements.
# Partially parsed test_base_provider_initialization_with_none_seed. Retrieved 3/5 statements.



def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.seed
    var_2 = var_0.random

import mimesis.random as module_0


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


def test_case_0():
    var_0 = None
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed
    assert var_2 is None
    var_3 = var_1.random


def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0.BaseProvider(random=var_0)
    var_2 = 'The random must be an instance of mimesis.random.Random'

import mimesis.random as module_0


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 123
    var_2 = module_1.BaseProvider(seed=var_1, random=var_0)
    var_3 = var_2.random
    var_4 = bool(var_2.random is var_0)
    assert var_4 is True
    var_5 = var_2.seed
    assert var_5 == 123



# Parsed testcases at query #25
#--------------------------

# Failed to parse test_random_is_not_none_and_is_not_instance_of_random_raises_type_error.



def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseProvider(random=var_0)
    var_2 = var_1.random
    var_3 = bool(var_1.random is var_0)
    assert var_3 is True



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_base_data_provider_init_default_locale. Retrieved 2/3 statements.
# Failed to parse test_base_data_provider_init_custom_locale.
# Partially parsed test_base_data_provider_init_inherits_random_from_base. Retrieved 2/4 statements.
# Failed to parse test_base_data_provider_init_locale_validation.
# Partially parsed test_base_data_provider_init_dataset_loaded. Retrieved 2/6 statements.
# Partially parsed test_base_data_provider_init_with_args_and_kwargs. Retrieved 2/4 statements.


import mimesis.providers.base as module_0


def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    var_3 = var_1._dataset
    var_4 = var_1.seed


def test_case_0():
    var_0 = 42
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.seed
    assert var_3 == 42

import mimesis.random as module_0


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
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.random

def test_case_0():
    var_0 = 'test'
    var_1 = 'test.json'

def test_case_0():
    var_0 = 123
    var_1 = 'test'


def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale


def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.seed



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_auto_register_provider. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'test_provider'
    var_1 = True
    var_2 = 'test_provider'



# Parsed testcases at query #28
#--------------------------

# Failed to parse test_base_data_provider_initialization_with_custom_locale.
# Partially parsed test_base_data_provider_initialization_with_datafile_meta. Retrieved 2/6 statements.
# Partially parsed test_base_data_provider_initialization_without_datafile_meta. Retrieved 1/4 statements.
# Partially parsed test_base_data_provider_initialization_locale_separated_data. Retrieved 3/7 statements.
# Partially parsed test_base_data_provider_initialization_with_args_and_kwargs. Retrieved 2/4 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    var_3 = var_1._dataset
    var_4 = bool(var_1._dataset == {})
    assert var_4 is True


def test_case_0():
    var_0 = 12345
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.seed
    assert var_3 == 12345

import mimesis.random as module_0


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
    var_5 = bool(True)
    assert var_5 is True


def test_case_0():
    var_0 = 'invalid_locale'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'test'
    var_1 = 'test.json'

def test_case_0():
    var_0 = 'test'

def test_case_0():
    var_0 = 'test'
    var_1 = 'test.json'
    var_2 = 'en_US'

def test_case_0():
    var_0 = 42
    var_1 = 'test'



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_auto_register_provider_when_meta_name_and_auto_register_true. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'test_provider'
    var_1 = True
    var_2 = 'test_provider'



# Parsed testcases at query #30
#--------------------------

# Failed to parse test_provider_registry_constructor.




# Parsed testcases at query #31
#--------------------------

# Partially parsed test_base_provider_initialization_with_defaults. Retrieved 2/4 statements.
# Partially parsed test_base_provider_initialization_with_seed. Retrieved 3/5 statements.
# Partially parsed test_base_provider_initialization_with_none_seed. Retrieved 3/5 statements.



def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.seed
    var_2 = var_0.random

import mimesis.random as module_0


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


def test_case_0():
    var_0 = None
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed
    assert var_2 is None
    var_3 = var_1.random


def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0.BaseProvider(random=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'The random must be an instance of mimesis.random.Random'

import mimesis.random as module_0


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 123
    var_2 = module_1.BaseProvider(seed=var_1, random=var_0)
    var_3 = var_2.random
    var_4 = bool(var_2.random is var_0)
    assert var_4 is True
    var_5 = var_2.seed
    assert var_5 == 123



# Parsed testcases at query #32
#--------------------------

# Failed to parse test_reseed_uses_global_seed_when_seed_is_missing_and_global_seed_is_set.




# Parsed testcases at query #33
#--------------------------

# Partially parsed test_reseed_with_missing_seed_and_global_seed_missing. Retrieved 3/5 statements.
# Partially parsed test_reseed_with_missing_seed_and_global_seed_set. Retrieved 3/7 statements.
# Partially parsed test_reseed_with_explicit_seed. Retrieved 4/6 statements.
# Partially parsed test_reseed_with_none_seed. Retrieved 4/6 statements.
# Partially parsed test_reseed_with_missing_seed_constant. Retrieved 2/3 statements.


import mimesis.providers.base as module_0


def test_case_0():
    var_0 = None
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.reseed()
    var_3 = var_1.seed
    assert var_3 is None


def test_case_0():
    var_0 = None
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.reseed()
    var_3 = var_1.seed
    assert var_3 is None


def test_case_0():
    var_0 = None
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = 999
    var_3 = var_1.reseed(var_2)
    var_4 = var_1.seed
    assert var_4 == 999


def test_case_0():
    var_0 = 100
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = None
    var_3 = var_1.reseed(var_2)
    var_4 = var_1.seed
    assert var_4 is None


def test_case_0():
    var_0 = 500
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = 700
    var_3 = var_1.reseed(var_2)
    var_4 = var_1.seed
    assert var_4 == 700


def test_case_0():
    var_0 = 200
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_auto_register_provider. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'test_provider'
    var_1 = True
    var_2 = 'test_provider'



# Parsed testcases at query #35
#--------------------------

# Failed to parse test_random_is_not_none_and_is_not_instance_of_random_raises_typeerror.


import mimesis.random as module_0


def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseProvider(random=var_0)
    var_2 = var_1.random
    var_3 = bool(var_1.random is var_0)
    assert var_3 is True



# Parsed testcases at query #36
#--------------------------

# Failed to parse test_provider_registry_constructor.




# Parsed testcases at query #37
#--------------------------

# Failed to parse test_provider_registry_constructor.




# Parsed testcases at query #38
#--------------------------

# Failed to parse test_provider_registry_constructor.




# Parsed testcases at query #39
#--------------------------

# Partially parsed test_constructor_with_default_parameters. Retrieved 2/4 statements.
# Partially parsed test_constructor_with_custom_seed. Retrieved 3/5 statements.
# Partially parsed test_constructor_with_none_seed. Retrieved 3/5 statements.


import mimesis.providers.base as module_0


def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.seed
    var_2 = var_0.random


def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 42
    var_3 = var_1.random


def test_case_0():
    var_0 = None
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed
    assert var_2 is None
    var_3 = var_1.random

import mimesis.random as module_0


def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseProvider(random=var_0)
    var_2 = var_1.random
    var_3 = bool(var_1.random is var_0)
    assert var_3 is True
    var_4 = var_1.seed


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 123
    var_2 = module_1.BaseProvider(seed=var_1, random=var_0)
    var_3 = var_2.random
    var_4 = bool(var_2.random is var_0)
    assert var_4 is True
    var_5 = var_2.seed
    assert var_5 == 123

import mimesis.providers.base as module_0


def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0.BaseProvider(random=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'The random must be an instance of mimesis.random.Random'



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_base_data_provider_initialization_with_default_locale. Retrieved 2/3 statements.
# Failed to parse test_base_data_provider_initialization_with_custom_locale.
# Partially parsed test_base_data_provider_initialization_with_locale_and_seed. Retrieved 1/3 statements.
# Partially parsed test_base_data_provider_initialization_without_datafile. Retrieved 2/5 statements.
# Partially parsed test_base_data_provider_initialization_with_datafile. Retrieved 3/10 statements.
# Failed to parse test_base_data_provider_str_representation.
# Partially parsed test_base_data_provider_inherits_from_base_provider. Retrieved 1/2 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    var_3 = var_1._dataset
    var_4 = var_1.seed


def test_case_0():
    var_0 = 42
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.seed
    assert var_3 == 42

import mimesis.random as module_0


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
    var_5 = 'The random must be an instance of mimesis.random.Random'

def test_case_0():
    var_0 = 123

def test_case_0():
    var_0 = 'test'
    var_1 = False

def test_case_0():
    var_0 = 'test'
    var_1 = False
    var_2 = 'test.json'


def test_case_0():
    var_0 = 'invalid_locale'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_auto_register_provider. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'test_provider'
    var_1 = True
    var_2 = 'test_provider'



# Parsed testcases at query #42
#--------------------------




import mimesis.random as module_0


def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseProvider(random=var_0)



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_base_data_provider_initialization_with_default_locale. Retrieved 2/4 statements.
# Failed to parse test_base_data_provider_initialization_with_custom_locale.
# Partially parsed test_base_data_provider_initialization_with_seed. Retrieved 3/5 statements.
# Partially parsed test_base_data_provider_initialization_locale_and_seed. Retrieved 1/3 statements.
# Partially parsed test_base_data_provider_initialization_inherits_from_base_provider. Retrieved 1/2 statements.
# Partially parsed test_base_data_provider_initialization_dataset_loaded. Retrieved 2/7 statements.
# Partially parsed test_base_data_provider_initialization_without_datafile. Retrieved 1/4 statements.
# Failed to parse test_base_data_provider_initialization_locale_setup_called.
# Partially parsed test_base_data_provider_initialization_load_dataset_called. Retrieved 2/7 statements.
# Partially parsed test_base_data_provider_initialization_with_kwargs. Retrieved 1/3 statements.


import mimesis.providers.base as module_0


def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    var_3 = var_1._dataset
    var_4 = bool(var_1._dataset == {})
    assert var_4 is True
    var_5 = var_1.random


def test_case_0():
    var_0 = 42
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.seed
    assert var_3 == 42
    var_4 = var_2.random

import mimesis.random as module_0


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


def test_case_0():
    var_0 = 'unsupported'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True


def test_case_0():
    var_0 = None
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.seed
    assert var_3 is None


def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.seed


def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)

def test_case_0():
    var_0 = 'test'
    var_1 = 'test.json'
    var_2 = bool(False)
    assert var_2 is True
    var_3 = bool(True)
    assert var_3 is True

def test_case_0():
    var_0 = 'test'

def test_case_0():
    var_0 = 'test'
    var_1 = 'test.json'
    var_2 = bool(False)
    assert var_2 is True
    var_3 = bool(True)
    assert var_3 is True

def test_case_0():
    var_0 = 999



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_constructor_with_default_parameters. Retrieved 2/4 statements.
# Partially parsed test_constructor_with_seed. Retrieved 3/5 statements.
# Partially parsed test_constructor_with_none_seed. Retrieved 3/5 statements.



def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.seed
    var_2 = var_0.random

import mimesis.random as module_0


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

import mimesis.random as module_0


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 42
    var_2 = module_1.BaseProvider(seed=var_1, random=var_0)
    var_3 = var_2.random
    var_4 = bool(var_2.random is var_0)
    assert var_4 is True
    var_5 = var_2.seed
    assert var_5 == 42

import mimesis.providers.base as module_0


def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0.BaseProvider(random=var_0)
    var_2 = bool(False)
    assert var_2 is True


def test_case_0():
    var_0 = None
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed
    assert var_2 is None
    var_3 = var_1.random



# Parsed testcases at query #45
#--------------------------

# Failed to parse test_provider_registry_constructor.




# Parsed testcases at query #46
#--------------------------

# Partially parsed test_constructor_with_default_parameters. Retrieved 2/4 statements.
# Partially parsed test_constructor_with_seed_parameter. Retrieved 3/5 statements.
# Partially parsed test_constructor_with_seed_none. Retrieved 3/5 statements.
# Partially parsed test_constructor_initializes_random_when_none_provided. Retrieved 2/4 statements.
# Failed to parse test_constructor_with_missingseed_constant.



def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.seed
    var_2 = var_0.random

import mimesis.random as module_0


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


def test_case_0():
    var_0 = None
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed
    assert var_2 is None
    var_3 = var_1.random

import mimesis.random as module_0


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 123
    var_2 = module_1.BaseProvider(seed=var_1, random=var_0)
    var_3 = var_2.random
    var_4 = bool(var_2.random is var_0)
    assert var_4 is True
    var_5 = var_2.seed
    assert var_5 == 123

import mimesis.providers.base as module_0


def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0.BaseProvider(random=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'The random must be an instance of mimesis.random.Random'


def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.random
    var_2 = bool(var_0.random is not None)
    assert var_2 is True
    var_3 = var_0.random


def test_case_0():
    var_0 = 999
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 999



# Parsed testcases at query #47
#--------------------------





def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale



# Parsed testcases at query #48
#--------------------------

# Failed to parse test_locale_default_does_not_trigger_unsupported_locale.




# Parsed testcases at query #49
#--------------------------

# Failed to parse test_provider_registry_constructor.




# Parsed testcases at query #50
#--------------------------

# Partially parsed test_base_data_provider_init_default_locale. Retrieved 2/3 statements.
# Failed to parse test_base_data_provider_init_custom_locale.
# Partially parsed test_base_data_provider_init_dataset_loading. Retrieved 2/9 statements.
# Partially parsed test_base_data_provider_init_no_datafile. Retrieved 1/4 statements.
# Partially parsed test_base_data_provider_init_with_args_and_kwargs. Retrieved 2/4 statements.
# Failed to parse test_base_data_provider_str_representation.



def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    var_3 = var_1._dataset
    var_4 = var_1.seed


def test_case_0():
    var_0 = 42
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.seed
    assert var_3 == 42
    var_4 = var_2.random
    var_5 = bool(var_2.random is not None)
    assert var_5 is True

import mimesis.random as module_0


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
    var_5 = bool(True)
    assert var_5 is True


def test_case_0():
    var_0 = 'invalid_locale'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'test'
    var_1 = 'test.json'
    var_2 = bool(True)
    assert var_2 is True

def test_case_0():
    var_0 = 'test'

def test_case_0():
    var_0 = 123
    var_1 = 'test'



# Parsed testcases at query #51
#--------------------------




def test_case_0():
    var_0 = 'test_provider'
    var_1 = False
    var_2 = 'test_provider'



# Parsed testcases at query #52
#--------------------------

# Failed to parse test_random_is_not_none_and_not_instance_of_random_raises_type_error.




# Parsed testcases at query #53
#--------------------------

# Failed to parse test_base_data_provider_initializes_with_custom_locale.
# Partially parsed test_base_data_provider_loads_dataset_with_datafile. Retrieved 2/7 statements.
# Partially parsed test_base_data_provider_dataset_empty_without_datafile. Retrieved 1/4 statements.
# Failed to parse test_base_data_provider_str_representation_includes_locale.



def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale


def test_case_0():
    var_0 = 42
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.seed
    assert var_3 == 42

import mimesis.random as module_0


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
    var_4 = 'The random must be an instance of mimesis.random.Random'

def test_case_0():
    var_0 = 'test'
    var_1 = 'test.json'

def test_case_0():
    var_0 = 'test'


def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = 'random'
    var_3 = hasattr(var_1, var_2)
    var_4 = bool(var_3)
    assert var_4 is True
    var_5 = 'seed'
    var_6 = hasattr(var_1, var_5)
    var_7 = bool(var_6)
    assert var_7 is True


def test_case_0():
    var_0 = 'invalid_locale'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)



# Parsed testcases at query #54
#--------------------------

# Failed to parse test_random_is_not_none_and_not_instance_of_random_raises_type_error.




# Parsed testcases at query #55
#--------------------------

# Failed to parse test_random_is_not_none_and_is_not_instance_of_random_raises_type_error.
# Partially parsed test_random_is_none_creates_default_random. Retrieved 3/6 statements.


import mimesis.random as module_0


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



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_auto_register_provider_when_meta_name_and_auto_register_true. Retrieved 3/6 statements.
# Partially parsed test_auto_register_provider_when_meta_name_and_auto_register_default. Retrieved 2/5 statements.
# Partially parsed test_do_not_auto_register_provider_when_meta_name_and_auto_register_false. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'test_provider'
    var_1 = True
    var_2 = 'test_provider'

def test_case_0():
    var_0 = 'test_provider_default'
    var_1 = 'test_provider_default'

def test_case_0():
    var_0 = 'test_provider_no_register'
    var_1 = False
    var_2 = 'test_provider_no_register'



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_base_data_provider_init_default_locale. Retrieved 2/3 statements.
# Failed to parse test_base_data_provider_init_custom_locale.
# Partially parsed test_base_data_provider_init_dataset_empty_when_no_datafile. Retrieved 2/5 statements.
# Partially parsed test_base_data_provider_init_dataset_loaded_with_datafile. Retrieved 3/8 statements.
# Failed to parse test_base_data_provider_init_locale_order_setup_before_dataset.



def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    var_3 = var_1._dataset
    var_4 = var_1.seed


def test_case_0():
    var_0 = 42
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.seed
    assert var_3 == 42

import mimesis.random as module_0


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
    var_5 = bool(True)
    assert var_5 is True


def test_case_0():
    var_0 = 'invalid_locale'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True


def test_case_0():
    var_0 = 123
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.seed
    assert var_3 == 123

def test_case_0():
    var_0 = 'test'
    var_1 = False

def test_case_0():
    var_0 = 'test'
    var_1 = False
    var_2 = 'test.json'
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True



# Parsed testcases at query #58
#--------------------------




import mimesis.random as module_0


def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseProvider(random=var_0)



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_constructor_with_default_parameters. Retrieved 2/4 statements.
# Partially parsed test_constructor_with_seed. Retrieved 3/5 statements.
# Partially parsed test_constructor_with_seed_none. Retrieved 3/5 statements.


import mimesis.providers.base as module_0


def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.seed
    var_2 = var_0.random

import mimesis.random as module_0


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


def test_case_0():
    var_0 = None
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed
    assert var_2 is None
    var_3 = var_1.random


def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0.BaseProvider(random=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = bool(True)
    assert var_3 is True



# Parsed testcases at query #60
#--------------------------




import mimesis.random as module_0


def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_1.BaseProvider(random=var_0)
    var_2 = var_1.random
    var_3 = bool(var_1.random is var_0)
    assert var_3 is True

import mimesis.providers.base as module_0


def test_case_0():
    var_0 = 'not_a_random_instance'
    var_1 = module_0.BaseProvider(random=var_0)



# Parsed testcases at query #61
#--------------------------

# Failed to parse test_random_is_not_none_and_not_instance_of_random_raises_type_error.




# Parsed testcases at query #62
#--------------------------

# Partially parsed test_auto_register_provider_with_meta_name_and_auto_register_true. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'test_provider'
    var_1 = True
    var_2 = 'test_provider'



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_constructor_with_default_parameters. Retrieved 2/4 statements.
# Partially parsed test_constructor_with_seed_parameter. Retrieved 3/5 statements.
# Partially parsed test_constructor_with_seed_none. Retrieved 3/5 statements.



def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.seed
    var_2 = var_0.random

import mimesis.random as module_0


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
    var_3 = 'The random must be an instance of mimesis.random.Random'


def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 42
    var_3 = var_1.random


def test_case_0():
    var_0 = None
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed
    assert var_2 is None
    var_3 = var_1.random

import mimesis.random as module_0


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 123
    var_2 = module_1.BaseProvider(seed=var_1, random=var_0)
    var_3 = var_2.seed
    assert var_3 == 123
    var_4 = var_2.random
    var_5 = bool(var_2.random is var_0)
    assert var_5 is True



# Parsed testcases at query #64
#--------------------------




def test_case_0():
    var_0 = 'test_provider'
    var_1 = True

def test_case_0():
    var_0 = 'test_provider_false'
    var_1 = False
    var_2 = 'test_provider_false'

def test_case_0():
    var_0 = 'test_provider'



