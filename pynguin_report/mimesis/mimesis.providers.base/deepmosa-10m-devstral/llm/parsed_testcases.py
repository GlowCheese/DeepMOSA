####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #2
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_reseed_with_missing_seed_and_global_seed_set. Retrieved 5/6 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.reseed()
    var_2 = var_0.seed
    var_3 = 0
    var_4 = 1
    var_5 = var_0.random.getstate()[var_4][var_3]
    assert var_5 == 42

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = 123
    var_2 = var_0.reseed(var_1)
    var_3 = var_0.seed
    assert var_3 == 123
    var_4 = 0
    var_5 = 1
    var_6 = var_0.random.getstate()[var_5][var_4]
    assert var_6 == 123

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = None
    var_2 = var_0.reseed(var_1)
    var_3 = var_0.seed
    assert var_3 is None
    var_4 = 0
    var_5 = 1
    var_6 = var_0.random.getstate()[var_5][var_4]
    var_7 = bool(var_6 != 0)
    assert var_7 is True



# Parsed testcases at query #4
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_base_provider_constructor_with_seed. Retrieved 2/3 statements.
# Partially parsed test_base_provider_constructor_without_seed. Retrieved 1/2 statements.
# Partially parsed test_base_provider_constructor_initializes_random. Retrieved 2/4 statements.


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
    var_0 = 'invalid_random'
    var_1 = module_0.BaseProvider(random=var_0)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 42

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.seed

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.random



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_provider_registry_initialization. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'nonexistent'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_base_data_provider_constructor_with_defaults. Retrieved 2/4 statements.
# Partially parsed test_base_data_provider_constructor_with_custom_locale. Retrieved 3/5 statements.
# Partially parsed test_base_data_provider_constructor_with_custom_seed. Retrieved 3/5 statements.


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
    var_0 = 'de'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'de'
    var_4 = var_2.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.seed
    var_4 = bool(var_2.seed == var_0)
    assert var_4 is True
    var_5 = var_2.random

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
    var_0 = 'not_a_random_instance'
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_0.BaseDataProvider(**var_2)



# Parsed testcases at query #8
#--------------------------




def test_case_0():
    var_0 = 'Seed to all the random functions.'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_base_data_provider_constructor_defaults. Retrieved 2/4 statements.
# Partially parsed test_base_data_provider_constructor_with_locale. Retrieved 3/5 statements.
# Partially parsed test_base_data_provider_constructor_with_seed. Retrieved 3/5 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    assert var_2 == 'en'
    var_3 = var_1._dataset
    var_4 = bool(var_1._dataset == {})
    assert var_4 is True
    var_5 = var_1.seed
    var_6 = var_1.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'de'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'de'
    var_4 = var_2._dataset
    var_5 = bool(var_2._dataset != {})
    assert var_5 is True
    var_6 = var_2.seed
    var_7 = var_2.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'en'
    var_4 = var_2._dataset
    var_5 = bool(var_2._dataset == {})
    assert var_5 is True
    var_6 = var_2.seed
    assert var_6 == 42
    var_7 = var_2.random

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_1.BaseDataProvider(**var_2)
    var_4 = var_3.locale
    assert var_4 == 'en'
    var_5 = var_3._dataset
    var_6 = bool(var_3._dataset == {})
    assert var_6 is True
    var_7 = var_3.seed
    var_8 = var_3.random
    var_9 = bool(var_3.random is var_0)
    assert var_9 is True

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'not a random object'
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_0.BaseDataProvider(**var_2)



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
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_base_data_provider_constructor_defaults. Retrieved 2/4 statements.
# Partially parsed test_base_data_provider_constructor_with_locale. Retrieved 4/7 statements.
# Partially parsed test_base_data_provider_constructor_with_seed. Retrieved 3/5 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    assert var_2 == 'en'
    var_3 = var_1._dataset
    var_4 = bool(var_1._dataset == {})
    assert var_4 is True
    var_5 = var_1.random
    var_6 = var_1.seed

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'de'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'de'
    var_4 = var_2._dataset
    var_5 = var_2.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.seed
    var_4 = bool(var_2.seed == var_0)
    assert var_4 is True
    var_5 = var_2.random

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
    var_0 = 'not_a_random_object'
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_0.BaseDataProvider(**var_2)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_validate_enum_predicate_false. Retrieved 9/12 statements.


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
    var_8 = [var_1, var_2, var_7]
    var_9 = None



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_base_data_provider_constructor_with_defaults. Retrieved 2/4 statements.
# Partially parsed test_base_data_provider_constructor_with_locale. Retrieved 4/7 statements.
# Partially parsed test_base_data_provider_constructor_with_seed. Retrieved 3/5 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    assert var_2 == 'en'
    var_3 = var_1._dataset
    var_4 = bool(var_1._dataset == {})
    assert var_4 is True
    var_5 = var_1.random
    var_6 = var_1.seed

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'de'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'de'
    var_4 = var_2._dataset
    var_5 = var_2.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.seed
    var_4 = bool(var_2.seed == var_0)
    assert var_4 is True
    var_5 = var_2.random

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
    var_0 = 'not_a_random_instance'
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_0.BaseDataProvider(**var_2)



# Parsed testcases at query #14
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_reseed_with_custom_seed. Retrieved 5/6 statements.
# Partially parsed test_reseed_with_missing_seed_and_global_seed_set. Retrieved 3/5 statements.
# Partially parsed test_reseed_with_missing_seed_and_no_global_seed. Retrieved 3/5 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = 100
    var_3 = var_1.reseed(var_2)
    var_4 = var_1.seed
    assert var_4 == 100
    var_5 = 32

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.reseed()
    var_2 = var_0.seed
    var_3 = 32

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.reseed()
    var_2 = var_0.seed
    var_3 = 32



# Parsed testcases at query #16
#--------------------------




def test_case_0():
    var_0 = 'Seed to all the random functions.'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_init_with_keyword_only_arguments. Retrieved 3/5 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 42
    var_3 = var_1.random



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_base_data_provider_constructor_defaults. Retrieved 2/4 statements.
# Partially parsed test_base_data_provider_constructor_custom_locale. Retrieved 3/5 statements.
# Partially parsed test_base_data_provider_constructor_custom_seed. Retrieved 3/5 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    assert var_2 == 'en'
    var_3 = var_1._dataset
    var_4 = bool(var_1._dataset == {})
    assert var_4 is True
    var_5 = var_1.seed
    var_6 = var_1.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'de'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'de'
    var_4 = var_2._dataset
    var_5 = bool(var_2._dataset != {})
    assert var_5 is True
    var_6 = var_2.seed
    var_7 = var_2.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'en'
    var_4 = var_2._dataset
    var_5 = bool(var_2._dataset == {})
    assert var_5 is True
    var_6 = var_2.seed
    assert var_6 == 42
    var_7 = var_2.random

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_1.BaseDataProvider(**var_2)
    var_4 = var_3.locale
    assert var_4 == 'en'
    var_5 = var_3._dataset
    var_6 = bool(var_3._dataset == {})
    assert var_6 is True
    var_7 = var_3.seed
    var_8 = var_3.random
    var_9 = bool(var_3.random is var_0)
    assert var_9 is True

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'not_a_random_instance'
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_0.BaseDataProvider(**var_2)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_reseed_with_missing_seed. Retrieved 2/3 statements.
# Partially parsed test_reseed_with_none_seed. Retrieved 4/5 statements.
# Partially parsed test_reseed_with_valid_seed. Retrieved 3/4 statements.
# Partially parsed test_reseed_updates_internal_random. Retrieved 3/5 statements.
# Partially parsed test_reseed_with_global_seed. Retrieved 2/5 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.reseed()

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = None
    var_3 = var_1.reseed(var_2)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = 123
    var_2 = var_0.reseed(var_1)
    var_3 = var_0.seed
    assert var_3 == 123

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.reseed(var_0)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.reseed()



# Parsed testcases at query #20
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_base_provider_constructor_with_seed. Retrieved 3/5 statements.
# Partially parsed test_base_provider_constructor_without_seed_or_random. Retrieved 2/4 statements.


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
    var_0 = 'invalid_random'
    var_1 = module_0.BaseProvider(random=var_0)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 42
    var_3 = var_1.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.seed
    var_2 = var_0.random



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_base_provider_constructor_with_seed. Retrieved 2/3 statements.
# Partially parsed test_base_provider_constructor_without_seed. Retrieved 1/2 statements.
# Partially parsed test_base_provider_constructor_initializes_random. Retrieved 2/4 statements.


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
    var_0 = 'invalid_random'
    var_1 = module_0.BaseProvider(random=var_0)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 42

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.seed

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.random



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_keyword_only_args. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'positional_arg'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_base_provider_constructor_with_seed. Retrieved 2/3 statements.
# Partially parsed test_base_provider_constructor_without_seed. Retrieved 1/2 statements.
# Partially parsed test_base_provider_constructor_with_missing_seed_and_global_seed. Retrieved 1/4 statements.
# Partially parsed test_base_provider_constructor_initializes_random. Retrieved 2/4 statements.


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
    var_0 = 'not a random object'
    var_1 = module_0.BaseProvider(random=var_0)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 42

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.seed

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.random



# Parsed testcases at query #25
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_base_data_provider_constructor_defaults. Retrieved 2/4 statements.
# Partially parsed test_base_data_provider_constructor_custom_locale. Retrieved 3/5 statements.
# Partially parsed test_base_data_provider_constructor_custom_seed. Retrieved 3/5 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    assert var_2 == 'en'
    var_3 = var_1._dataset
    var_4 = bool(var_1._dataset == {})
    assert var_4 is True
    var_5 = var_1.random
    var_6 = var_1.seed

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'de'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'de'
    var_4 = var_2._dataset
    var_5 = bool(var_2._dataset != {})
    assert var_5 is True
    var_6 = var_2.random
    var_7 = var_2.seed

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'en'
    var_4 = var_2._dataset
    var_5 = bool(var_2._dataset == {})
    assert var_5 is True
    var_6 = var_2.random
    var_7 = var_2.seed
    assert var_7 == 42

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_1.BaseDataProvider(**var_2)
    var_4 = var_3.locale
    assert var_4 == 'en'
    var_5 = var_3._dataset
    var_6 = bool(var_3._dataset == {})
    assert var_6 is True
    var_7 = var_3.random
    var_8 = bool(var_3.random is var_0)
    assert var_8 is True
    var_9 = var_3.seed

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'not_a_random_object'
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_0.BaseDataProvider(**var_2)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'unsupported_locale'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_validate_enum_with_none_item. Retrieved 5/7 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = None



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_validate_enum_with_none_item. Retrieved 8/10 statements.
# Partially parsed test_validate_enum_with_valid_enum_item. Retrieved 7/10 statements.
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
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_base_data_provider_constructor_defaults. Retrieved 2/4 statements.
# Partially parsed test_base_data_provider_constructor_custom_locale. Retrieved 3/5 statements.
# Partially parsed test_base_data_provider_constructor_custom_seed. Retrieved 3/5 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    assert var_2 == 'en'
    var_3 = var_1._dataset
    var_4 = bool(var_1._dataset == {})
    assert var_4 is True
    var_5 = var_1.seed
    var_6 = var_1.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'de'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'de'
    var_4 = var_2._dataset
    var_5 = bool(var_2._dataset != {})
    assert var_5 is True
    var_6 = var_2.seed
    var_7 = var_2.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'en'
    var_4 = var_2._dataset
    var_5 = bool(var_2._dataset == {})
    assert var_5 is True
    var_6 = var_2.seed
    var_7 = bool(var_2.seed == var_0)
    assert var_7 is True
    var_8 = var_2.random

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = 42
    var_1 = module_0.Random(var_0)
    var_2 = 'random'
    var_3 = {var_2: var_1}
    var_4 = module_1.BaseDataProvider(**var_3)
    var_5 = var_4.locale
    assert var_5 == 'en'
    var_6 = var_4._dataset
    var_7 = bool(var_4._dataset == {})
    assert var_7 is True
    var_8 = var_4.seed
    var_9 = var_4.random
    var_10 = bool(var_4.random == var_1)
    assert var_10 is True

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'not_a_random_instance'
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_0.BaseDataProvider(**var_2)



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_validate_enum_with_valid_item. Retrieved 3/8 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'value1'
    var_1 = 'value2'
    var_2 = module_0.BaseProvider()



# Parsed testcases at query #31
#--------------------------

# Failed to parse test_default_locale_parameter.




# Parsed testcases at query #32
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
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_init_with_positional_argument. Retrieved 1/3 statements.


def test_case_0():
    var_0 = None



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_init_params_docstring. Retrieved 2/6 statements.


def test_case_0():
    var_0 = ':param locale: Current locale.'
    var_1 = ':param seed: Seed to all the random functions.'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_base_data_provider_initialization. Retrieved 2/4 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    assert var_2 == 'en'
    var_3 = var_1._dataset
    var_4 = bool(var_1._dataset == {})
    assert var_4 is True
    var_5 = var_1.seed
    var_6 = var_1.random



# Parsed testcases at query #36
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 42



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_provider_registry_initial_state. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'non_existent'



# Parsed testcases at query #38
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_provider_registry_initial_state. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'nonexistent'



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_init_requires_keyword_only_arguments. Retrieved 4/6 statements.


import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = 42
    var_1 = module_0.Random()
    var_2 = module_1.BaseProvider(seed=var_0, random=var_1)
    var_3 = var_2.seed
    assert var_3 == 42
    var_4 = var_2.random



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_init_with_positional_args. Retrieved 1/3 statements.


def test_case_0():
    var_0 = None



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_base_provider_constructor_with_seed. Retrieved 2/3 statements.
# Partially parsed test_base_provider_constructor_without_seed. Retrieved 1/2 statements.


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
    var_0 = 'invalid_random'
    var_1 = module_0.BaseProvider(random=var_0)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 42

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.seed

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 42
    var_2 = module_1.BaseProvider(seed=var_1, random=var_0)
    var_3 = var_2.seed
    assert var_3 == 42
    var_4 = var_2.random
    var_5 = bool(var_2.random == var_0)
    assert var_5 is True



# Parsed testcases at query #43
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale



# Parsed testcases at query #44
#--------------------------

# Failed to parse test_init_docstring_is_not_empty.




# Parsed testcases at query #45
#--------------------------

# Partially parsed test_base_data_provider_constructor_defaults. Retrieved 2/4 statements.
# Partially parsed test_base_data_provider_constructor_with_locale. Retrieved 3/5 statements.
# Partially parsed test_base_data_provider_constructor_with_seed. Retrieved 3/5 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    assert var_2 == 'en'
    var_3 = var_1._dataset
    var_4 = bool(var_1._dataset == {})
    assert var_4 is True
    var_5 = var_1.random
    var_6 = var_1.seed

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'de'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'de'
    var_4 = var_2._dataset
    var_5 = bool(var_2._dataset != {})
    assert var_5 is True
    var_6 = var_2.random
    var_7 = var_2.seed

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'en'
    var_4 = var_2._dataset
    var_5 = bool(var_2._dataset == {})
    assert var_5 is True
    var_6 = var_2.random
    var_7 = var_2.seed
    assert var_7 == 42

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_1.BaseDataProvider(**var_2)
    var_4 = var_3.locale
    assert var_4 == 'en'
    var_5 = var_3._dataset
    var_6 = bool(var_3._dataset == {})
    assert var_6 is True
    var_7 = var_3.random
    var_8 = bool(var_3.random is var_0)
    assert var_8 is True
    var_9 = var_3.seed

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'not_a_random_instance'
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_0.BaseDataProvider(**var_2)



# Parsed testcases at query #46
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale



# Parsed testcases at query #47
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_init_with_positional_args_fails. Retrieved 1/3 statements.


def test_case_0():
    var_0 = None



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_provider_registry_initial_state. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'non_existent'



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_base_provider_constructor_with_seed. Retrieved 3/5 statements.
# Partially parsed test_base_provider_constructor_without_seed_or_random. Retrieved 2/4 statements.


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
    var_0 = 'invalid_random'
    var_1 = module_0.BaseProvider(random=var_0)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 42
    var_3 = var_1.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.seed
    var_2 = var_0.random



# Parsed testcases at query #51
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_init_without_datafile. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = ''



# Parsed testcases at query #53
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_provider_registry_initial_state. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'nonexistent'



# Parsed testcases at query #55
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #56
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_base_data_provider_constructor_defaults. Retrieved 2/4 statements.
# Partially parsed test_base_data_provider_constructor_custom_seed. Retrieved 3/5 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    assert var_2 == 'en'
    var_3 = var_1._dataset
    var_4 = bool(var_1._dataset == {})
    assert var_4 is True
    var_5 = var_1.seed
    var_6 = var_1.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'de'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'de'
    var_4 = var_2._dataset
    var_5 = bool(var_2._dataset != {})
    assert var_5 is True
    var_6 = var_2.seed

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'en'
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
    var_4 = var_3.random
    var_5 = bool(var_3.random is var_0)
    assert var_5 is True
    var_6 = var_3.locale
    assert var_6 == 'en'
    var_7 = var_3.seed

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'not a random object'
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_0.BaseDataProvider(**var_2)



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_provider_registry_initial_state. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'nonexistent'



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_base_data_provider_constructor_defaults. Retrieved 2/4 statements.
# Partially parsed test_base_data_provider_constructor_custom_locale. Retrieved 3/5 statements.
# Partially parsed test_base_data_provider_constructor_custom_seed. Retrieved 3/5 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    assert var_2 == 'en'
    var_3 = var_1._dataset
    var_4 = bool(var_1._dataset == {})
    assert var_4 is True
    var_5 = var_1.random
    var_6 = var_1.seed

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'de'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'de'
    var_4 = var_2.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.seed
    var_4 = bool(var_2.seed == var_0)
    assert var_4 is True
    var_5 = var_2.random

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
    var_0 = 'not_a_random_instance'
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_0.BaseDataProvider(**var_2)



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_init_docstring. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'Initialize attributes for data providers.'



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_init_docstring_predicate. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'Initialize attributes for data providers.'



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_init_docstring_predicate. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'Initialize attributes for data providers.'



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_base_provider_constructor_with_seed. Retrieved 2/3 statements.
# Partially parsed test_base_provider_constructor_without_seed. Retrieved 1/2 statements.
# Partially parsed test_base_provider_constructor_with_missing_seed_and_global_seed. Retrieved 1/4 statements.


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
    var_0 = 'invalid_random'
    var_1 = module_0.BaseProvider(random=var_0)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 42

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.seed

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.seed



# Parsed testcases at query #64
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 42



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_init_requires_keyword_only_arguments. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 123



# Parsed testcases at query #66
#--------------------------

# Partially parsed test_base_data_provider_constructor_with_default_locale_and_seed. Retrieved 2/4 statements.
# Partially parsed test_base_data_provider_constructor_with_custom_locale_and_seed. Retrieved 4/6 statements.


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
    var_0 = 'de'
    var_1 = 42
    var_2 = {}
    var_3 = module_0.BaseDataProvider(var_0, var_1, **var_2)
    var_4 = var_3.locale
    assert var_4 == 'de'
    var_5 = var_3.seed
    assert var_5 == 42
    var_6 = var_3.random
    var_7 = var_3._dataset
    var_8 = bool(var_3._dataset != {})
    assert var_8 is True

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
    var_6 = var_3.seed
    var_7 = var_3.locale
    var_8 = var_3._dataset
    var_9 = bool(var_3._dataset == {})
    assert var_9 is True

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'not_a_random_instance'
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_0.BaseDataProvider(**var_2)



# Parsed testcases at query #67
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1._dataset
    var_3 = bool(var_1._dataset == {})
    assert var_3 is True



# Parsed testcases at query #68
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)
    var_3 = var_2._dataset
    var_4 = bool(var_2._dataset == {})
    assert var_4 is True



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_init_sets_dataset_to_empty_dict. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = 'test.json'



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_base_data_provider_constructor_defaults. Retrieved 2/4 statements.
# Partially parsed test_base_data_provider_constructor_with_locale. Retrieved 3/5 statements.
# Partially parsed test_base_data_provider_constructor_with_seed. Retrieved 3/5 statements.


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
    var_0 = 'de'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'de'
    var_4 = var_2.seed
    var_5 = var_2.random
    var_6 = var_2._dataset
    var_7 = bool(var_2._dataset != {})
    assert var_7 is True

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
    var_0 = 'not_a_random_object'
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_0.BaseDataProvider(**var_2)



# Parsed testcases at query #71
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_base_provider_constructor_with_seed. Retrieved 2/3 statements.
# Partially parsed test_base_provider_constructor_without_seed. Retrieved 1/2 statements.
# Failed to parse test_base_provider_constructor_with_missing_seed.
# Partially parsed test_base_provider_constructor_initializes_random. Retrieved 2/4 statements.


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
    var_0 = 'not_a_random_object'
    var_1 = module_0.BaseProvider(random=var_0)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 42

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.seed

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.random



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_init_with_keyword_only_arguments. Retrieved 5/7 statements.


import mimesis.providers.base as module_0
import mimesis.random as module_1

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 42
    var_3 = var_1.random
    var_4 = module_1.Random()
    var_5 = module_0.BaseProvider(seed=var_0, random=var_4)
    var_6 = var_5.seed
    assert var_6 == 42
    var_7 = var_5.random
    var_8 = bool(var_5.random is var_4)
    assert var_8 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_base_data_provider_constructor_with_default_locale. Retrieved 2/4 statements.
# Partially parsed test_base_data_provider_constructor_with_custom_locale. Retrieved 3/5 statements.
# Partially parsed test_base_data_provider_constructor_with_seed. Retrieved 3/5 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    assert var_2 == 'en'
    var_3 = var_1._dataset
    var_4 = bool(var_1._dataset == {})
    assert var_4 is True
    var_5 = var_1.random
    var_6 = var_1.seed

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'de'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'de'
    var_4 = var_2._dataset
    var_5 = bool(var_2._dataset != {})
    assert var_5 is True
    var_6 = var_2.random
    var_7 = var_2.seed

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'en'
    var_4 = var_2._dataset
    var_5 = bool(var_2._dataset == {})
    assert var_5 is True
    var_6 = var_2.random
    var_7 = var_2.seed
    assert var_7 == 42

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_1.BaseDataProvider(**var_2)
    var_4 = var_3.locale
    assert var_4 == 'en'
    var_5 = var_3._dataset
    var_6 = bool(var_3._dataset == {})
    assert var_6 is True
    var_7 = var_3.random
    var_8 = bool(var_3.random is var_0)
    assert var_8 is True
    var_9 = var_3.seed

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'not_a_random_instance'
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_0.BaseDataProvider(**var_2)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'unsupported_locale'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_reseed_with_missing_seed_and_global_seed. Retrieved 2/3 statements.
# Partially parsed test_reseed_updates_random_seed. Retrieved 3/5 statements.


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



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_provider_registry_initial_state. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'non_existent'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_base_data_provider_constructor_with_default_locale. Retrieved 2/4 statements.
# Partially parsed test_base_data_provider_constructor_with_custom_locale. Retrieved 3/5 statements.
# Partially parsed test_base_data_provider_constructor_with_seed. Retrieved 3/5 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    var_3 = var_1.random
    var_4 = var_1._dataset
    var_5 = bool(var_1._dataset == {})
    assert var_5 is True
    var_6 = var_1.seed

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'de'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'de'
    var_4 = var_2.random
    var_5 = var_2._dataset
    var_6 = bool(var_2._dataset != {})
    assert var_6 is True
    var_7 = var_2.seed

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.locale
    var_4 = var_2.random
    var_5 = var_2._dataset
    var_6 = bool(var_2._dataset == {})
    assert var_6 is True
    var_7 = var_2.seed
    assert var_7 == 42

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
    var_8 = bool(var_3._dataset == {})
    assert var_8 is True
    var_9 = var_3.seed

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'not_a_random_instance'
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_0.BaseDataProvider(**var_2)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_base_data_provider_constructor_defaults. Retrieved 2/4 statements.
# Partially parsed test_base_data_provider_constructor_with_locale. Retrieved 4/7 statements.
# Partially parsed test_base_data_provider_constructor_with_seed. Retrieved 3/5 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    assert var_2 == 'en'
    var_3 = var_1._dataset
    var_4 = bool(var_1._dataset == {})
    assert var_4 is True
    var_5 = var_1.random
    var_6 = var_1.seed

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'de'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'de'
    var_4 = var_2._dataset
    var_5 = var_2.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.seed
    assert var_3 == 42
    var_4 = var_2.random

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
    var_0 = 'not_a_random_object'
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_0.BaseDataProvider(**var_2)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_provider_registry_initialization. Retrieved 1/4 statements.


def test_case_0():
    var_0 = []
    var_1 = 'nonexistent'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_base_data_provider_constructor_defaults. Retrieved 2/4 statements.
# Partially parsed test_base_data_provider_constructor_with_locale. Retrieved 3/4 statements.
# Partially parsed test_base_data_provider_constructor_with_seed. Retrieved 2/3 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    var_3 = var_1._dataset
    var_4 = bool(var_1._dataset == {})
    assert var_4 is True
    var_5 = var_1.random
    var_6 = var_1.seed

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'de'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'de'
    var_4 = var_2._dataset

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
    var_0 = 100
    var_1 = module_0.Random(var_0)
    var_2 = 'random'
    var_3 = {var_2: var_1}
    var_4 = module_1.BaseDataProvider(**var_3)
    var_5 = var_4.random
    var_6 = bool(var_4.random is var_1)
    assert var_6 is True

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'not_a_random_object'
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_0.BaseDataProvider(**var_2)



# Parsed testcases at query #10
#--------------------------




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



# Parsed testcases at query #11
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_reseed_with_missing_seed_and_global_seed_not_missing. Retrieved 2/3 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.reseed()



# Parsed testcases at query #13
#--------------------------




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
    var_0 = 'invalid_random'
    var_1 = module_0.BaseProvider(random=var_0)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 42
    var_3 = var_1.random
    var_4 = bool(var_1.random is not None)
    assert var_4 is True

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.seed
    var_2 = var_0.random
    var_3 = bool(var_0.random is not None)
    assert var_3 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_base_provider_constructor_with_seed. Retrieved 2/3 statements.
# Partially parsed test_base_provider_constructor_without_seed. Retrieved 1/2 statements.
# Partially parsed test_base_provider_constructor_with_missing_seed_and_global_seed. Retrieved 1/4 statements.
# Partially parsed test_base_provider_constructor_initializes_random. Retrieved 2/4 statements.


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
    var_0 = 'invalid_random'
    var_1 = module_0.BaseProvider(random=var_0)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 42

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.seed

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.seed

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.random



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_base_data_provider_constructor_default_locale. Retrieved 2/4 statements.
# Partially parsed test_base_data_provider_constructor_custom_locale. Retrieved 3/5 statements.
# Partially parsed test_base_data_provider_constructor_custom_seed. Retrieved 3/5 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    assert var_2 == 'en'
    var_3 = var_1._dataset
    var_4 = bool(var_1._dataset == {})
    assert var_4 is True
    var_5 = var_1.random
    var_6 = var_1.seed

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'de'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'de'
    var_4 = var_2._dataset
    var_5 = bool(var_2._dataset != {})
    assert var_5 is True
    var_6 = var_2.random
    var_7 = var_2.seed

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'en'
    var_4 = var_2._dataset
    var_5 = bool(var_2._dataset == {})
    assert var_5 is True
    var_6 = var_2.random
    var_7 = var_2.seed
    assert var_7 == 42

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_1.BaseDataProvider(**var_2)
    var_4 = var_3.locale
    assert var_4 == 'en'
    var_5 = var_3._dataset
    var_6 = bool(var_3._dataset == {})
    assert var_6 is True
    var_7 = var_3.random
    var_8 = bool(var_3.random is var_0)
    assert var_8 is True
    var_9 = var_3.seed

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'not_a_random_instance'
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_0.BaseDataProvider(**var_2)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_init_with_positional_args_raises_type_error. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'some_seed'



# Parsed testcases at query #17
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.reseed()
    var_2 = var_0.seed
    var_3 = 0
    var_4 = 1
    var_5 = var_0.random.getstate()[var_4][var_3]

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = 42
    var_2 = var_0.reseed(var_1)
    var_3 = var_0.seed
    var_4 = bool(var_0.seed == var_1)
    assert var_4 is True
    var_5 = 0
    var_6 = 1
    var_7 = var_0.random.getstate()[var_6][var_5]
    var_8 = bool(var_7 == var_1)
    assert var_8 is True

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = None
    var_2 = var_0.reseed(var_1)
    var_3 = var_0.seed
    assert var_3 is None
    var_4 = 0
    var_5 = 1
    var_6 = var_0.random.getstate()[var_5][var_4]
    var_7 = bool(var_6 is not None)
    assert var_7 is True



# Parsed testcases at query #18
#--------------------------




def test_case_0():
    var_0 = 'Initialize attributes for data providers.'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_base_data_provider_constructor_defaults. Retrieved 2/4 statements.
# Partially parsed test_base_data_provider_constructor_with_locale. Retrieved 3/5 statements.
# Partially parsed test_base_data_provider_constructor_with_seed. Retrieved 3/5 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    assert var_2 == 'en'
    var_3 = var_1._dataset
    var_4 = bool(var_1._dataset == {})
    assert var_4 is True
    var_5 = var_1.seed
    var_6 = var_1.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'de'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'de'
    var_4 = var_2._dataset
    var_5 = bool(var_2._dataset != {})
    assert var_5 is True
    var_6 = var_2.seed
    var_7 = var_2.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'en'
    var_4 = var_2._dataset
    var_5 = bool(var_2._dataset == {})
    assert var_5 is True
    var_6 = var_2.seed
    assert var_6 == 42
    var_7 = var_2.random

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_1.BaseDataProvider(**var_2)
    var_4 = var_3.locale
    assert var_4 == 'en'
    var_5 = var_3._dataset
    var_6 = bool(var_3._dataset == {})
    assert var_6 is True
    var_7 = var_3.seed
    var_8 = var_3.random
    var_9 = bool(var_3.random is var_0)
    assert var_9 is True

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'not_a_random_object'
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_0.BaseDataProvider(**var_2)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_init_with_positional_args. Retrieved 1/3 statements.


def test_case_0():
    var_0 = None



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_base_provider_constructor_with_seed. Retrieved 3/5 statements.
# Partially parsed test_base_provider_constructor_defaults. Retrieved 2/4 statements.


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

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.seed
    var_2 = var_0.random



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_provider_registry_initialization. Retrieved 1/4 statements.


def test_case_0():
    var_0 = []
    var_1 = 'nonexistent'



# Parsed testcases at query #23
#--------------------------




def test_case_0():
    var_0 = 'seed: Seed to all the random functions.'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_base_data_provider_constructor_defaults. Retrieved 2/4 statements.
# Partially parsed test_base_data_provider_constructor_with_locale. Retrieved 3/5 statements.
# Partially parsed test_base_data_provider_constructor_with_seed. Retrieved 3/5 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    assert var_2 == 'en'
    var_3 = var_1._dataset
    var_4 = bool(var_1._dataset == {})
    assert var_4 is True
    var_5 = var_1.seed
    var_6 = var_1.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'de'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'de'
    var_4 = var_2._dataset
    var_5 = bool(var_2._dataset != {})
    assert var_5 is True
    var_6 = var_2.seed
    var_7 = var_2.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'en'
    var_4 = var_2.seed
    var_5 = bool(var_2.seed == var_0)
    assert var_5 is True
    var_6 = var_2.random

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_1.BaseDataProvider(**var_2)
    var_4 = var_3.locale
    assert var_4 == 'en'
    var_5 = var_3.random
    var_6 = bool(var_3.random is var_0)
    assert var_6 is True
    var_7 = var_3.seed

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'not_a_random_instance'
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_0.BaseDataProvider(**var_2)



# Parsed testcases at query #25
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #26
#--------------------------




def test_case_0():
    var_0 = 'seed'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_base_provider_constructor_with_seed. Retrieved 3/5 statements.
# Partially parsed test_base_provider_constructor_without_seed. Retrieved 2/4 statements.


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
    var_0 = 'not_a_random_instance'
    var_1 = module_0.BaseProvider(random=var_0)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 42
    var_3 = var_1.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.seed
    var_2 = var_0.random



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_reseed_with_missing_seed_and_global_seed_not_missing. Retrieved 2/3 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.reseed()
    var_2 = var_0.seed



# Parsed testcases at query #29
#--------------------------




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
    var_0 = 'invalid_random'
    var_1 = module_0.BaseProvider(random=var_0)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 42
    var_3 = var_1.random
    var_4 = bool(var_1.random is not None)
    assert var_4 is True

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.seed
    var_2 = var_0.random
    var_3 = bool(var_0.random is not None)
    assert var_3 is True



# Parsed testcases at query #30
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = 42
    var_2 = {}
    var_3 = module_0.BaseDataProvider(var_0, var_1, **var_2)
    var_4 = var_3.seed
    assert var_4 == 42



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_provider_registry_initialization. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'nonexistent'



# Parsed testcases at query #32
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_provider_registry_initialization. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'non_existent'



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_base_provider_constructor_with_seed. Retrieved 2/3 statements.
# Partially parsed test_base_provider_constructor_without_seed. Retrieved 1/2 statements.


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
    var_0 = 'invalid_random'
    var_1 = module_0.BaseProvider(random=var_0)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 42

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.seed

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 42
    var_2 = module_1.BaseProvider(seed=var_1, random=var_0)
    var_3 = var_2.seed
    assert var_3 == 42
    var_4 = var_2.random
    var_5 = bool(var_2.random is var_0)
    assert var_5 is True



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_init_with_positional_args. Retrieved 2/4 statements.


def test_case_0():
    var_0 = 'Test that __init__ raises TypeError when called with positional args.'
    var_1 = 'positional_arg'



# Parsed testcases at query #36
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)
    var_3 = var_2._dataset
    var_4 = bool(var_2._dataset != {})
    assert var_4 is True



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_init_without_auto_register. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = False
    var_2 = 'test'



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_base_provider_constructor_with_seed. Retrieved 2/3 statements.
# Partially parsed test_base_provider_constructor_without_seed. Retrieved 1/2 statements.


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
    var_0 = 'invalid_random'
    var_1 = module_0.BaseProvider(random=var_0)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 42

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.seed

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 42
    var_2 = module_1.BaseProvider(seed=var_1, random=var_0)
    var_3 = var_2.seed
    assert var_3 == 42
    var_4 = var_2.random
    var_5 = bool(var_2.random is var_0)
    assert var_5 is True



# Parsed testcases at query #39
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_base_provider_constructor_with_seed. Retrieved 2/3 statements.
# Partially parsed test_base_provider_constructor_without_seed. Retrieved 1/2 statements.
# Partially parsed test_base_provider_constructor_with_missing_seed_and_global_seed. Retrieved 1/4 statements.
# Partially parsed test_base_provider_constructor_initializes_random. Retrieved 2/4 statements.


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
    var_0 = 'invalid_random'
    var_1 = module_0.BaseProvider(random=var_0)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 42

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.seed

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.seed

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.random



# Parsed testcases at query #41
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_init_with_positional_argument_raises_type_error. Retrieved 1/3 statements.


def test_case_0():
    var_0 = None



# Parsed testcases at query #43
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #44
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #45
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_init_with_keyword_only_arguments. Retrieved 3/5 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 42
    var_3 = var_1.random



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_init_docstring_predicate. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'Initialize attributes for data providers.'



# Parsed testcases at query #48
#--------------------------

# Failed to parse test_provider_registry_initialization.




# Parsed testcases at query #49
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_base_provider_constructor_with_seed. Retrieved 2/3 statements.
# Partially parsed test_base_provider_constructor_without_seed. Retrieved 1/2 statements.
# Partially parsed test_base_provider_constructor_default_random. Retrieved 2/4 statements.


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
    var_0 = 'invalid_random'
    var_1 = module_0.BaseProvider(random=var_0)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 42

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.seed

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
    var_3 = var_2.seed
    assert var_3 == 42
    var_4 = var_2.random
    var_5 = bool(var_2.random is var_0)
    assert var_5 is True



# Parsed testcases at query #51
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_init_docstring_predicate. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'Initialize attributes for data providers.'



# Parsed testcases at query #53
#--------------------------

# Failed to parse test_init_docstring_is_not_empty.




# Parsed testcases at query #54
#--------------------------

# Partially parsed test_base_provider_constructor_with_seed. Retrieved 3/5 statements.
# Partially parsed test_base_provider_constructor_without_seed. Retrieved 2/4 statements.


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
    var_0 = 'invalid_random'
    var_1 = module_0.BaseProvider(random=var_0)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 42
    var_3 = var_1.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.seed
    var_2 = var_0.random

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 42
    var_2 = module_1.BaseProvider(seed=var_1, random=var_0)
    var_3 = var_2.random
    var_4 = bool(var_2.random == var_0)
    assert var_4 is True
    var_5 = var_2.seed
    assert var_5 == 42



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_base_data_provider_constructor_defaults. Retrieved 2/4 statements.
# Partially parsed test_base_data_provider_constructor_custom_locale. Retrieved 3/5 statements.
# Partially parsed test_base_data_provider_constructor_custom_seed. Retrieved 3/5 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    assert var_2 == 'en'
    var_3 = var_1._dataset
    var_4 = bool(var_1._dataset == {})
    assert var_4 is True
    var_5 = var_1.seed
    var_6 = var_1.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'de'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'de'
    var_4 = var_2._dataset
    var_5 = bool(var_2._dataset != {})
    assert var_5 is True
    var_6 = var_2.seed
    var_7 = var_2.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'en'
    var_4 = var_2._dataset
    var_5 = bool(var_2._dataset == {})
    assert var_5 is True
    var_6 = var_2.seed
    assert var_6 == 42
    var_7 = var_2.random

import mimesis.random as module_0
import mimesis.providers.base as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_1.BaseDataProvider(**var_2)
    var_4 = var_3.locale
    assert var_4 == 'en'
    var_5 = var_3._dataset
    var_6 = bool(var_3._dataset == {})
    assert var_6 is True
    var_7 = var_3.seed
    var_8 = var_3.random
    var_9 = bool(var_3.random is var_0)
    assert var_9 is True

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'not_a_random_object'
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_0.BaseDataProvider(**var_2)



# Parsed testcases at query #56
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale



# Parsed testcases at query #57
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 42



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_base_data_provider_constructor_defaults. Retrieved 2/4 statements.
# Partially parsed test_base_data_provider_constructor_with_locale. Retrieved 3/4 statements.
# Partially parsed test_base_data_provider_constructor_with_seed. Retrieved 2/3 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    assert var_2 == 'en'
    var_3 = var_1._dataset
    var_4 = bool(var_1._dataset == {})
    assert var_4 is True
    var_5 = var_1.random
    var_6 = var_1.seed

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'de'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'de'
    var_4 = var_2._dataset

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
    var_0 = 'not_a_random_instance'
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_0.BaseDataProvider(**var_2)



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_init_with_positional_args_raises_type_error. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'invalid_positional_arg'



# Parsed testcases at query #60
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1._dataset
    var_3 = bool(var_1._dataset == {})
    assert var_3 is True



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_init_docstring_predicate. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'Initialize attributes for data providers.'



# Parsed testcases at query #62
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.get_current_locale()



# Parsed testcases at query #63
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_base_data_provider_constructor_defaults. Retrieved 2/4 statements.
# Failed to parse test_base_data_provider_constructor_custom_locale.


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
    var_5 = bool(var_3.random is var_0)
    assert var_5 is True

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'not_a_random_instance'
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_0.BaseDataProvider(**var_2)

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'unsupported_locale'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_init_without_keyword_args_raises_type_error. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'invalid_seed'



# Parsed testcases at query #66
#--------------------------

# Partially parsed test_init_docstring_predicate. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'Initialize attributes for data providers.'



# Parsed testcases at query #67
#--------------------------




import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    var_3 = bool(var_1.locale is not None)
    assert var_3 is True



# Parsed testcases at query #68
#--------------------------

# Partially parsed test_base_data_provider_constructor_defaults. Retrieved 2/4 statements.
# Partially parsed test_base_data_provider_constructor_custom_locale. Retrieved 4/7 statements.
# Partially parsed test_base_data_provider_constructor_custom_seed. Retrieved 3/5 statements.


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.BaseDataProvider(**var_0)
    var_2 = var_1.locale
    assert var_2 == 'en'
    var_3 = var_1._dataset
    var_4 = bool(var_1._dataset == {})
    assert var_4 is True
    var_5 = var_1.random
    var_6 = var_1.seed

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'de'
    var_1 = {}
    var_2 = module_0.BaseDataProvider(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'de'
    var_4 = var_2._dataset
    var_5 = var_2.random

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = {}
    var_2 = module_0.BaseDataProvider(seed=var_0, **var_1)
    var_3 = var_2.seed
    var_4 = bool(var_2.seed == var_0)
    assert var_4 is True
    var_5 = var_2.random

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
    var_6 = var_3.seed

import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'not_a_random_instance'
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_0.BaseDataProvider(**var_2)



