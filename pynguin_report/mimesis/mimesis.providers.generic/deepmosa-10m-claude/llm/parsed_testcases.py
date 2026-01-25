####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Failed to parse test_generic_constructor_custom_locale.
# Partially parsed test_generic_constructor_with_locale_and_seed. Retrieved 1/5 statements.
# Partially parsed test_generic_constructor_has_random_instance. Retrieved 2/5 statements.
# Partially parsed test_generic_constructor_providers_have_same_seed. Retrieved 4/9 statements.
# Partially parsed test_generic_constructor_locale_attribute_exists. Retrieved 1/6 statements.
# Partially parsed test_generic_constructor_auto_register_is_false. Retrieved 1/4 statements.
# Partially parsed test_generic_constructor_name_is_generic. Retrieved 1/4 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.locale
    var_2 = bool(var_0.seed is not None or var_0.random is not None)
    assert var_2 is True

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
    var_1 = var_0.__dir__()
    var_2 = len(var_1)
    var_3 = bool(var_2 > 0)
    assert var_3 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.random

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Generic(seed=var_0)
    var_2 = var_1.__dir__()
    var_3 = 'seed'

def test_case_0():
    var_0 = 'locale'

def test_case_0():
    var_0 = 'auto_register'

def test_case_0():
    var_0 = 'name'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_generic_init_line_16_predicate_false. Retrieved 6/20 statements.


def test_case_0():
    var_0 = 'Test that line 16 predicate evaluates to False for BaseDataProvider subclasses.'
    var_1 = 'mock_data'
    var_2 = True
    var_3 = 'simple'
    var_4 = True
    var_5 = 'locale'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_generic_init_predicate_line_16_evaluates_to_false. Retrieved 1/13 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 16 evaluates to False for BaseDataProvider subclasses.'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_add_provider_with_valid_provider. Retrieved 5/11 statements.
# Partially parsed test_add_provider_with_kwargs. Retrieved 3/11 statements.
# Partially parsed test_add_provider_uses_generic_seed. Retrieved 3/8 statements.
# Partially parsed test_add_provider_without_meta_name. Retrieved 3/8 statements.
# Partially parsed test_add_provider_with_non_baseprovider_raises_error. Retrieved 1/6 statements.
# Partially parsed test_add_provider_with_generic_raises_error. Retrieved 1/4 statements.
# Partially parsed test_add_provider_seed_kwarg_is_ignored. Retrieved 4/9 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom'
    var_1 = module_0.Generic()
    var_2 = 'custom'
    var_3 = hasattr(var_1, var_2)
    var_4 = bool(var_3)
    assert var_4 is True
    var_5 = var_1.custom

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom'
    var_1 = module_0.Generic()
    var_2 = 'test_value'
    var_3 = var_1.custom.custom_arg
    assert var_3 == 'test_value'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom'
    var_1 = 42
    var_2 = module_0.Generic(seed=var_1)
    var_3 = var_2.custom.seed
    assert var_3 == 42

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'mycustomprovider'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True

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
    var_2 = 'subclass of mimesis.providers.BaseProvider'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'Cannot add Generic instance to itself'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom'
    var_1 = 42
    var_2 = module_0.Generic(seed=var_1)
    var_3 = 999
    var_4 = var_2.custom.seed
    assert var_4 == 42



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_add_provider_with_valid_provider. Retrieved 5/11 statements.
# Partially parsed test_add_provider_with_provider_without_meta_name. Retrieved 4/10 statements.
# Partially parsed test_add_provider_with_kwargs. Retrieved 5/13 statements.
# Partially parsed test_add_provider_seed_is_enforced. Retrieved 4/9 statements.
# Partially parsed test_add_provider_raises_typeerror_not_subclass. Retrieved 1/4 statements.
# Partially parsed test_add_provider_raises_typeerror_generic_instance. Retrieved 3/9 statements.
# Partially parsed test_add_provider_instance_has_same_seed_as_generic. Retrieved 3/8 statements.
# Partially parsed test_add_provider_overwrites_existing_provider. Retrieved 5/15 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom'
    var_1 = module_0.Generic()
    var_2 = 'custom'
    var_3 = hasattr(var_1, var_2)
    var_4 = bool(var_3)
    assert var_4 is True
    var_5 = var_1.custom

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'mycustomprovider'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = var_0.mycustomprovider

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom_kwargs'
    var_1 = module_0.Generic()
    var_2 = 'test_value'
    var_3 = 'custom_kwargs'
    var_4 = hasattr(var_1, var_3)
    var_5 = bool(var_4)
    assert var_5 is True
    var_6 = var_1.custom_kwargs.custom_arg
    assert var_6 == 'test_value'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom_seed'
    var_1 = 42
    var_2 = module_0.Generic(seed=var_1)
    var_3 = 999
    var_4 = var_2.custom_seed.seed
    assert var_4 == 42

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'subclass of mimesis.providers.BaseProvider'

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
    var_0 = 'generic_sub'
    var_1 = False
    var_2 = module_0.Generic()
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'Cannot add Generic instance to itself'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom_seed_test'
    var_1 = 12345
    var_2 = module_0.Generic(seed=var_1)
    var_3 = var_2.custom_seed_test.seed
    assert var_3 == 12345

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'overwrite_test'
    var_1 = 'overwrite_test'
    var_2 = module_0.Generic()
    var_3 = var_2.overwrite_test
    var_4 = var_2.overwrite_test
    var_5 = bool(var_3 is not var_4)
    assert var_5 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_generic_init_skips_generic_provider. Retrieved 6/10 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'Test that Generic.__init__ skips Generic provider in registry loop.'
    var_1 = module_0.Generic()
    var_2 = 'generic'
    var_3 = 'generic'
    var_4 = hasattr(var_1, var_3)
    var_5 = bool(not var_4)
    assert var_5 is True
    var_6 = dir(var_1)
    var_7 = len(var_6)
    var_8 = bool(var_7 > 0)
    assert var_8 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_reseed_propagates_to_all_providers. Retrieved 3/7 statements.
# Partially parsed test_reseed_with_missing_seed. Retrieved 1/4 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.reseed()
    var_2 = var_0.seed
    var_3 = bool(var_0.seed is not None)
    assert var_3 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 12345
    var_2 = var_0.reseed(var_1)
    var_3 = var_0.seed
    var_4 = bool(var_0.seed == var_1)
    assert var_4 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 54321
    var_2 = var_0.reseed(var_1)

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.seed

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 111
    var_2 = var_0.reseed(var_1)
    var_3 = var_0.seed
    assert var_3 == 111
    var_4 = 222
    var_5 = var_0.reseed(var_4)
    var_6 = var_0.seed
    assert var_6 == 222
    var_7 = 333
    var_8 = var_0.reseed(var_7)
    var_9 = var_0.seed
    assert var_9 == 333



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_generic_getattr_returns_provider_instance. Retrieved 2/8 statements.
# Partially parsed test_generic_getattr_caches_provider_instance. Retrieved 1/7 statements.
# Partially parsed test_generic_getattr_initializes_with_correct_locale. Retrieved 1/6 statements.
# Partially parsed test_generic_getattr_initializes_with_correct_seed. Retrieved 2/7 statements.
# Partially parsed test_generic_getattr_nonexistent_attribute_returns_none. Retrieved 1/6 statements.
# Partially parsed test_generic_getattr_multiple_providers. Retrieved 1/7 statements.
# Partially parsed test_generic_getattr_provider_is_callable. Retrieved 1/11 statements.


def test_case_0():
    var_0 = 'Test that __getattr__ returns an instance of a data provider.'
    var_1 = 'full_name'

def test_case_0():
    var_0 = 'Test that __getattr__ caches the provider instance.'

def test_case_0():
    var_0 = 'Test that __getattr__ initializes provider with correct locale.'

def test_case_0():
    var_0 = 'Test that __getattr__ initializes provider with correct seed.'
    var_1 = 12345

def test_case_0():
    var_0 = 'Test that __getattr__ returns None for nonexistent attributes.'

def test_case_0():
    var_0 = 'Test that __getattr__ works for multiple different providers.'

def test_case_0():
    var_0 = 'Test that retrieved provider has callable methods.'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_generic_getattr_returns_provider_instance. Retrieved 2/8 statements.
# Partially parsed test_generic_getattr_caches_provider_instance. Retrieved 1/7 statements.
# Partially parsed test_generic_getattr_with_underscore_prefix. Retrieved 6/13 statements.
# Partially parsed test_generic_getattr_returns_none_for_invalid_attribute. Retrieved 2/7 statements.
# Partially parsed test_generic_getattr_initializes_with_correct_locale. Retrieved 1/6 statements.


def test_case_0():
    var_0 = 'Test that __getattr__ returns an instance of a data provider.'
    var_1 = 'full_name'

def test_case_0():
    var_0 = 'Test that __getattr__ caches the provider instance in __dict__.'

def test_case_0():
    var_0 = 'Test that __getattr__ handles attributes with underscore prefix.'
    var_1 = 'MockProvider'
    var_2 = ()
    var_3 = {}
    var_4 = [var_1, var_2, var_3]
    var_5 = '__dict__'
    var_6 = '_person'

def test_case_0():
    var_0 = 'Test that __getattr__ returns None for invalid attributes.'
    var_1 = 'nonexistent_provider_xyz'

def test_case_0():
    var_0 = 'Test that __getattr__ initializes provider with correct locale.'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'Test that __getattr__ initializes provider with correct seed.'
    var_1 = 42
    var_2 = module_0.Generic(seed=var_1)
    var_3 = var_2.person
    var_4 = var_3.seed
    var_5 = bool(var_3.seed == var_1)
    assert var_5 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_add_provider_with_valid_provider. Retrieved 6/15 statements.
# Partially parsed test_add_provider_with_provider_without_meta_name. Retrieved 4/11 statements.
# Partially parsed test_add_provider_with_non_baseprovider_subclass_raises_type_error. Retrieved 1/6 statements.
# Partially parsed test_add_provider_with_generic_raises_type_error. Retrieved 1/4 statements.
# Partially parsed test_add_provider_preserves_seed. Retrieved 4/11 statements.
# Partially parsed test_add_provider_ignores_seed_kwarg. Retrieved 5/10 statements.
# Partially parsed test_add_provider_with_kwargs. Retrieved 4/12 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom'
    var_1 = False
    var_2 = module_0.Generic()
    var_3 = 'custom'
    var_4 = hasattr(var_2, var_3)
    var_5 = bool(var_4)
    assert var_5 is True
    var_6 = var_2.custom

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'mycustomprovider'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = var_0.mycustomprovider

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
    var_2 = 'subclass of mimesis.providers.BaseProvider'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'Cannot add Generic instance to itself'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom'
    var_1 = False
    var_2 = 42
    var_3 = module_0.Generic(seed=var_2)
    var_4 = var_3.custom.seed
    assert var_4 == 42

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom'
    var_1 = False
    var_2 = 42
    var_3 = module_0.Generic(seed=var_2)
    var_4 = 100
    var_5 = var_3.custom.seed
    assert var_5 == 42

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom'
    var_1 = False
    var_2 = module_0.Generic()
    var_3 = 'test_value'
    var_4 = var_2.custom.custom_arg
    assert var_4 == 'test_value'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_add_provider_with_valid_provider. Retrieved 5/11 statements.
# Partially parsed test_add_provider_with_provider_without_meta_name. Retrieved 4/10 statements.
# Partially parsed test_add_provider_with_kwargs. Retrieved 5/13 statements.
# Partially parsed test_add_provider_with_non_baseprovider_subclass_raises_type_error. Retrieved 1/6 statements.
# Partially parsed test_add_provider_with_generic_raises_type_error. Retrieved 3/8 statements.
# Partially parsed test_add_provider_seed_inheritance. Retrieved 3/8 statements.
# Partially parsed test_add_provider_removes_seed_kwarg. Retrieved 4/12 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom'
    var_1 = module_0.Generic()
    var_2 = 'custom'
    var_3 = hasattr(var_1, var_2)
    var_4 = bool(var_3)
    assert var_4 is True
    var_5 = var_1.custom

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'anotherprovider'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = var_0.anotherprovider

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'parametrized'
    var_1 = module_0.Generic()
    var_2 = 'test_value'
    var_3 = 'parametrized'
    var_4 = hasattr(var_1, var_3)
    var_5 = bool(var_4)
    assert var_5 is True
    var_6 = var_1.parametrized.custom_param
    assert var_6 == 'test_value'

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
    var_2 = 'subclass of mimesis.providers.BaseProvider'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'generic_sub'
    var_1 = False
    var_2 = module_0.Generic()
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'Cannot add Generic instance to itself'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'seedcheck'
    var_1 = 42
    var_2 = module_0.Generic(seed=var_1)
    var_3 = var_2.seedcheck.seed
    assert var_3 == 42

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'kwargs_test'
    var_1 = module_0.Generic()
    var_2 = 999
    var_3 = 'custom'
    var_4 = var_1.kwargs_test.other_param
    assert var_4 == 'custom'
    var_5 = var_1.kwargs_test.seed
    var_6 = bool(var_1.kwargs_test.seed != 999)
    assert var_6 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_getattr_predicate_evaluates_to_false. Retrieved 6/9 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 8 evaluates to False when attribute is not callable.'
    var_1 = module_0.Generic()
    var_2 = '_test_attr'
    var_3 = 'non_callable_string'
    var_4 = 'test_attr'
    var_5 = var_1.__getattr__(var_4)
    assert var_5 is None



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_reseed_handles_attribute_error. Retrieved 5/12 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'Test that reseed catches AttributeError when provider lacks reseed method.'
    var_1 = 'custom_no_reseed'
    var_2 = module_0.Generic()
    var_3 = 42
    var_4 = var_2.reseed(var_3)
    var_5 = var_2.seed
    assert var_5 == 42



# Parsed testcases at query #14
#--------------------------

# Failed to parse test_generic_constructor_custom_locale.
# Partially parsed test_generic_constructor_custom_locale_and_seed. Retrieved 1/5 statements.
# Partially parsed test_generic_constructor_registers_base_providers. Retrieved 3/8 statements.
# Partially parsed test_generic_constructor_seed_propagation. Retrieved 3/8 statements.
# Partially parsed test_generic_constructor_creates_generic_instance. Retrieved 7/9 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.locale
    var_2 = var_0.seed

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Generic(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 42
    var_3 = var_1.locale

def test_case_0():
    var_0 = 123

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.__dir__()
    var_2 = len(var_1)
    var_3 = bool(var_2 > 0)
    assert var_3 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 100
    var_1 = module_0.Generic(seed=var_0)
    var_2 = var_1.__dir__()

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 999
    var_1 = module_0.Generic(seed=var_0)
    var_2 = var_1.__dir__()

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.Generic(seed=var_0)
    var_2 = var_1.seed
    assert var_2 is None

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'random'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = 'locale'
    var_5 = hasattr(var_0, var_4)
    var_6 = bool(var_5)
    assert var_6 is True
    var_7 = 'seed'
    var_8 = hasattr(var_0, var_7)
    var_9 = bool(var_8)
    assert var_9 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.__dir__()
    var_2 = 'generic'
    var_3 = bool('generic' not in var_1)
    assert var_3 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.__dir__()
    var_2 = 'locale'
    var_3 = bool('locale' not in var_1)
    assert var_3 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_getattr_predicate_evaluates_to_false. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 8 evaluates to False when attribute is not callable.'
    var_1 = 'test_attr'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_add_provider_with_valid_provider. Retrieved 5/11 statements.
# Partially parsed test_add_provider_without_meta_name. Retrieved 4/10 statements.
# Partially parsed test_add_provider_with_kwargs. Retrieved 5/13 statements.
# Partially parsed test_add_provider_seed_enforcement. Retrieved 4/9 statements.
# Partially parsed test_add_provider_not_subclass_of_base_provider. Retrieved 1/6 statements.
# Partially parsed test_add_provider_cannot_add_generic. Retrieved 1/4 statements.
# Partially parsed test_add_provider_preserves_generic_seed. Retrieved 3/8 statements.
# Partially parsed test_add_provider_multiple_providers. Retrieved 9/19 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom'
    var_1 = module_0.Generic()
    var_2 = 'custom'
    var_3 = hasattr(var_1, var_2)
    var_4 = bool(var_3)
    assert var_4 is True
    var_5 = var_1.custom

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'mycustomprovider'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = var_0.mycustomprovider

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom'
    var_1 = module_0.Generic()
    var_2 = 'test_value'
    var_3 = 'custom'
    var_4 = hasattr(var_1, var_3)
    var_5 = bool(var_4)
    assert var_5 is True
    var_6 = var_1.custom.custom_arg
    assert var_6 == 'test_value'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom'
    var_1 = 42
    var_2 = module_0.Generic(seed=var_1)
    var_3 = 999
    var_4 = var_2.custom.seed
    assert var_4 == 42

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
    var_2 = 'subclass of mimesis.providers.BaseProvider'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'Cannot add Generic instance to itself'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom'
    var_1 = 123
    var_2 = module_0.Generic(seed=var_1)
    var_3 = var_2.custom.seed
    assert var_3 == 123
    var_4 = var_2.seed
    assert var_4 == 123

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'provider_one'
    var_1 = 'provider_two'
    var_2 = module_0.Generic()
    var_3 = 'provider_one'
    var_4 = hasattr(var_2, var_3)
    var_5 = bool(var_4)
    assert var_5 is True
    var_6 = 'provider_two'
    var_7 = hasattr(var_2, var_6)
    var_8 = bool(var_7)
    assert var_8 is True
    var_9 = var_2.provider_one
    var_10 = var_2.provider_two



# Parsed testcases at query #17
#--------------------------

# Failed to parse test_generic_constructor_with_locale.
# Partially parsed test_generic_constructor_with_locale_and_seed. Retrieved 1/5 statements.
# Partially parsed test_generic_constructor_locale_independent_providers. Retrieved 6/9 statements.
# Partially parsed test_generic_constructor_seed_propagation. Retrieved 2/7 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.locale
    var_2 = var_0.seed
    var_3 = var_0.random
    var_4 = bool(var_0.random is not None)
    assert var_4 is True

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
    var_1 = var_0.__dir__()
    var_2 = len(var_1)
    var_3 = bool(var_2 > 0)
    assert var_3 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.__dir__()
    var_2 = 'person'
    var_3 = 'address'
    var_4 = 'text'
    var_5 = [var_2, var_3, var_4]

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 999
    var_1 = module_0.Generic(seed=var_0)

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Generic(seed=var_0)
    var_2 = module_0.Generic(seed=var_0)
    var_3 = var_1.locale
    var_4 = bool(var_1.locale == var_2.locale)
    assert var_4 is True
    var_5 = var_1.seed
    var_6 = bool(var_1.seed == var_2.seed)
    assert var_6 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_getattr_predicate_false_when_attribute_not_callable. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 8 evaluates to False when attribute is not callable.'



# Parsed testcases at query #19
#--------------------------

# Failed to parse test_generic_constructor_custom_locale.
# Partially parsed test_generic_constructor_with_locale_and_seed. Retrieved 1/5 statements.
# Failed to parse test_generic_constructor_locale_stored.
# Partially parsed test_generic_constructor_dir_method_available. Retrieved 3/5 statements.


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
    var_0 = 54321

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'random'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = var_0.random
    var_5 = bool(var_0.random is not None)
    assert var_5 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'reseed'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = var_0.reseed
    var_5 = callable(var_4)
    var_6 = bool(var_5)
    assert var_6 is True
    var_7 = 'validate_enum'
    var_8 = hasattr(var_0, var_7)
    var_9 = bool(var_8)
    assert var_9 is True
    var_10 = var_0.validate_enum
    var_11 = callable(var_10)
    var_12 = bool(var_11)
    assert var_12 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.Generic(seed=var_0)
    var_2 = var_1.seed
    assert var_2 is None

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Generic(seed=var_0)
    var_2 = module_0.Generic(seed=var_0)
    var_3 = var_1.seed
    var_4 = bool(var_1.seed == var_2.seed)
    assert var_4 is True
    var_5 = bool(var_1 is not var_2)
    assert var_5 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.__dir__()
    var_2 = len(var_1)
    var_3 = bool(var_2 > 0)
    assert var_3 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_generic_init_skips_generic_provider. Retrieved 2/11 statements.


def test_case_0():
    var_0 = 'Test that Generic.__init__ skips Generic provider in registry iteration.'
    var_1 = 'generic'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_reseed_catches_attribute_error. Retrieved 5/13 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = "Test that reseed catches AttributeError when provider doesn't have reseed method."
    var_1 = 'custom_no_reseed'
    var_2 = module_0.Generic()
    var_3 = 42
    var_4 = var_2.reseed(var_3)
    var_5 = var_2.seed
    assert var_5 == 42



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_reseed_catches_attribute_error. Retrieved 5/14 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'Test that reseed method catches AttributeError when reseeding providers.'
    var_1 = 'mock'
    var_2 = module_0.Generic()
    var_3 = 12345
    var_4 = var_2.reseed(var_3)
    var_5 = bool(var_2 is not None)
    assert var_5 is True
    var_6 = var_2.locale
    var_7 = bool(var_2.locale is not None)
    assert var_7 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_getattr_predicate_evaluates_to_false. Retrieved 4/7 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 8 evaluates to False when attribute is not callable.'
    var_1 = module_0.Generic()
    var_2 = 'test_attr'
    var_3 = var_1.__getattr__(var_2)
    assert var_3 is None



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_getattr_predicate_evaluates_to_false. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 8 evaluates to False.'
    var_1 = '_test_attr'
    var_2 = 'non_callable_value'
    var_3 = 'test_attr'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_add_provider_with_meta_name_attribute. Retrieved 6/14 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 19 evaluates to False when Meta.name exists.'
    var_1 = 'custom'
    var_2 = module_0.Generic()
    var_3 = 'custom'
    var_4 = hasattr(var_2, var_3)
    var_5 = bool(var_4)
    assert var_5 is True
    var_6 = var_2.custom



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_getattr_returns_provider_instance. Retrieved 2/8 statements.
# Partially parsed test_getattr_caches_provider_instance. Retrieved 1/7 statements.
# Partially parsed test_getattr_initializes_with_correct_locale. Retrieved 1/6 statements.
# Partially parsed test_getattr_initializes_with_correct_seed. Retrieved 2/7 statements.
# Partially parsed test_getattr_nonexistent_attribute_returns_none. Retrieved 2/7 statements.
# Partially parsed test_getattr_with_underscore_prefix. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'Test that __getattr__ returns a provider instance when called.'
    var_1 = 'first_name'

def test_case_0():
    var_0 = 'Test that __getattr__ caches the provider instance in __dict__.'

def test_case_0():
    var_0 = 'Test that __getattr__ initializes provider with correct locale.'

def test_case_0():
    var_0 = 'Test that __getattr__ initializes provider with correct seed.'
    var_1 = 12345

def test_case_0():
    var_0 = 'Test that __getattr__ returns None for nonexistent attributes.'
    var_1 = '_nonexistent_provider'

def test_case_0():
    var_0 = 'Test that __getattr__ handles attribute names with underscore prefix.'
    var_1 = '_person'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_generic_init_predicate_line_16_evaluates_to_false. Retrieved 5/15 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 16 evaluates to False for BaseDataProvider subclasses.'
    var_1 = module_0.Generic()
    var_2 = False
    var_3 = True
    var_4 = hasattr(var_1, var_0)
    var_5 = bool(var_4)
    assert var_5 is True
    var_6 = bool(var_3)
    assert var_6 is True



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_add_provider_with_valid_provider. Retrieved 5/11 statements.
# Partially parsed test_add_provider_with_provider_without_meta_name. Retrieved 4/10 statements.
# Partially parsed test_add_provider_with_kwargs. Retrieved 5/13 statements.
# Partially parsed test_add_provider_raises_type_error_for_non_baseprovider_subclass. Retrieved 1/6 statements.
# Partially parsed test_add_provider_raises_type_error_for_generic_instance. Retrieved 1/4 statements.
# Partially parsed test_add_provider_enforces_same_seed. Retrieved 4/9 statements.
# Partially parsed test_add_provider_with_multiple_providers. Retrieved 9/19 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom'
    var_1 = module_0.Generic()
    var_2 = 'custom'
    var_3 = hasattr(var_1, var_2)
    var_4 = bool(var_3)
    assert var_4 is True
    var_5 = var_1.custom

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'anotherprovider'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = var_0.anotherprovider

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom_with_kwargs'
    var_1 = module_0.Generic()
    var_2 = 'test_value'
    var_3 = 'custom_with_kwargs'
    var_4 = hasattr(var_1, var_3)
    var_5 = bool(var_4)
    assert var_5 is True
    var_6 = var_1.custom_with_kwargs.custom_param
    assert var_6 == 'test_value'

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
    var_2 = 'subclass of mimesis.providers.BaseProvider'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'Cannot add Generic instance to itself'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'seedtest'
    var_1 = 42
    var_2 = module_0.Generic(seed=var_1)
    var_3 = 999
    var_4 = var_2.seedtest.seed
    assert var_4 == 42

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'provider_one'
    var_1 = 'provider_two'
    var_2 = module_0.Generic()
    var_3 = 'provider_one'
    var_4 = hasattr(var_2, var_3)
    var_5 = bool(var_4)
    assert var_5 is True
    var_6 = 'provider_two'
    var_7 = hasattr(var_2, var_6)
    var_8 = bool(var_7)
    assert var_8 is True
    var_9 = var_2.provider_one
    var_10 = var_2.provider_two



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_reseed_attribute_error_handling. Retrieved 5/13 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'Test that AttributeError is caught during reseed when a provider lacks reseed method.'
    var_1 = 'custom_no_reseed'
    var_2 = module_0.Generic()
    var_3 = 12345
    var_4 = var_2.reseed(var_3)
    var_5 = var_2.locale
    var_6 = bool(var_2.locale is not None)
    assert var_6 is True



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_generic_init_skips_generic_provider. Retrieved 3/13 statements.


def test_case_0():
    var_0 = 'Test that Generic.__init__ skips Generic provider in registry loop.'
    var_1 = 'generic'
    var_2 = None



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_getattr_predicate_evaluates_to_false. Retrieved 3/6 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 8 evaluates to False when attribute is not callable.'
    var_1 = module_0.Generic()
    var_2 = var_1.test_attr
    assert var_2 is None



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_getattr_predicate_evaluates_to_false. Retrieved 5/13 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 8 evaluates to False when attribute is not callable.'
    var_1 = module_0.Generic()
    var_2 = 'mock'
    var_3 = 'testattr'
    var_4 = var_1.__getattr__(var_3)
    assert var_4 is None



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_reseed_handles_attribute_error. Retrieved 5/12 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'Test that reseed gracefully handles AttributeError when reseeding providers.'
    var_1 = module_0.Generic()
    var_2 = 'mock'
    var_3 = 42
    var_4 = var_1.reseed(var_3)
    var_5 = var_1.seed
    assert var_5 == 42



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_generic_init_skips_generic_provider. Retrieved 4/14 statements.


def test_case_0():
    var_0 = 'Test that Generic.__init__ skips Generic class in provider registry iteration.'
    var_1 = 'Generic'
    var_2 = None
    var_3 = 'random'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_generic_constructor_with_locale. Retrieved 1/6 statements.
# Partially parsed test_generic_constructor_with_locale_and_seed. Retrieved 2/7 statements.
# Partially parsed test_generic_constructor_initializes_providers. Retrieved 3/6 statements.
# Partially parsed test_generic_constructor_seed_propagation. Retrieved 2/7 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.locale
    var_2 = bool(var_0.seed is not None or var_0.seed is None)
    assert var_2 is True
    var_3 = 'random'
    var_4 = hasattr(var_0, var_3)
    var_5 = bool(var_4)
    assert var_5 is True

def test_case_0():
    var_0 = 'random'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Generic(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 42
    var_3 = 'random'
    var_4 = hasattr(var_1, var_3)
    var_5 = bool(var_4)
    assert var_5 is True

def test_case_0():
    var_0 = 123
    var_1 = 'random'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = dir(var_0)
    var_2 = len(var_1)
    var_3 = bool(var_2 > 0)
    assert var_3 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Generic(seed=var_0)
    var_2 = 2
    var_3 = module_0.Generic(seed=var_2)
    var_4 = bool(var_1 is not var_3)
    assert var_4 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.__dict__
    var_2 = '_'
    var_3 = '__'
    var_4 = [attr for attr in var_1 if attr.startswith(var_2) and (not attr.startswith(var_3))]
    var_5 = len(var_4)
    var_6 = bool(var_5 > 0)
    assert var_6 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 999
    var_1 = module_0.Generic(seed=var_0)



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_getattr_predicate_evaluates_to_false. Retrieved 5/10 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 8 evaluates to False when attribute is not callable.'
    var_1 = module_0.Generic()
    var_2 = 'mock'
    var_3 = 'testattr'
    var_4 = var_1.__getattr__(var_3)
    assert var_4 is None



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_getattr_with_non_callable_attribute. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'Test that __getattr__ returns None when attribute is not callable.'
    var_1 = 'test_attr'



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_add_provider_with_meta_name_attribute. Retrieved 5/11 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom_provider'
    var_1 = module_0.Generic()
    var_2 = 'custom_provider'
    var_3 = hasattr(var_1, var_2)
    var_4 = bool(var_3)
    assert var_4 is True
    var_5 = var_1.custom_provider



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_generic_init_skips_generic_provider. Retrieved 6/10 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'Test that Generic.__init__ skips Generic provider in registry loop.'
    var_1 = module_0.Generic()
    var_2 = 'generic'
    var_3 = 'generic'
    var_4 = hasattr(var_1, var_3)
    var_5 = bool(not var_4)
    assert var_5 is True
    var_6 = var_1.__dir__()
    var_7 = len(var_6)
    var_8 = bool(var_7 > 0)
    assert var_8 is True



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_reseed_attribute_error_handling. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'Test that AttributeError is caught and handled in reseed method.'
    var_1 = 42



# Parsed testcases at query #41
#--------------------------

# Failed to parse test_generic_constructor_custom_locale.
# Partially parsed test_generic_constructor_with_locale_and_seed. Retrieved 1/5 statements.
# Partially parsed test_generic_constructor_initializes_providers. Retrieved 1/6 statements.


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
    var_3 = var_1.locale

def test_case_0():
    var_0 = 54321

def test_case_0():
    var_0 = 'random'

def test_case_0():
    pass

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.Generic(seed=var_0)
    var_2 = var_1.seed
    assert var_2 is None
    var_3 = var_1.locale



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_generic_getattr_returns_provider_instance. Retrieved 1/7 statements.
# Failed to parse test_generic_getattr_caches_provider_instance.
# Failed to parse test_generic_getattr_initializes_provider_with_locale.
# Partially parsed test_generic_getattr_initializes_provider_with_seed. Retrieved 1/6 statements.
# Failed to parse test_generic_getattr_nonexistent_attribute_returns_none.
# Failed to parse test_generic_getattr_removes_underscore_prefix.


def test_case_0():
    var_0 = 'first_name'

def test_case_0():
    var_0 = 12345



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_add_provider_with_meta_name_attribute. Retrieved 6/14 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'Test that the except AttributeError block (line 19) is not executed when Meta.name exists.'
    var_1 = 'custom_provider'
    var_2 = module_0.Generic()
    var_3 = 'custom_provider'
    var_4 = hasattr(var_2, var_3)
    var_5 = bool(var_4)
    assert var_5 is True
    var_6 = var_2.custom_provider



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_getattr_with_non_callable_attribute. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'Test that __getattr__ returns None when attribute is not callable.'
    var_1 = 'test_attr'



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_generic_constructor_default. Retrieved 5/9 statements.
# Partially parsed test_generic_constructor_with_locale. Retrieved 1/5 statements.
# Partially parsed test_generic_constructor_with_locale_and_seed. Retrieved 2/6 statements.
# Partially parsed test_generic_constructor_has_random_instance. Retrieved 3/7 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'Test Generic constructor with default parameters.'
    var_1 = module_0.Generic()
    var_2 = var_1.locale
    var_3 = var_1.random
    var_4 = 'seed'
    var_5 = hasattr(var_1, var_4)
    var_6 = bool(var_5)
    assert var_6 is True

def test_case_0():
    var_0 = 'Test Generic constructor with specific locale.'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'Test Generic constructor with seed parameter.'
    var_1 = 12345
    var_2 = module_0.Generic(seed=var_1)
    var_3 = var_2.seed
    assert var_3 == 12345

def test_case_0():
    var_0 = 'Test Generic constructor with both locale and seed.'
    var_1 = 54321

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'Test that Generic constructor initializes providers.'
    var_1 = module_0.Generic()
    var_2 = var_1.__dir__()
    var_3 = len(var_2)
    var_4 = bool(var_3 > 0)
    assert var_4 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'Test Generic constructor with None as seed.'
    var_1 = None
    var_2 = module_0.Generic(seed=var_1)
    var_3 = var_2.seed
    assert var_3 is None

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'Test that Generic constructor creates random instance.'
    var_1 = module_0.Generic()
    var_2 = var_1.random

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'Test that multiple Generic instances are independent.'
    var_1 = 100
    var_2 = module_0.Generic(seed=var_1)
    var_3 = module_0.Generic(seed=var_1)
    var_4 = var_2.locale
    var_5 = bool(var_2.locale == var_3.locale)
    assert var_5 is True
    var_6 = var_2.seed
    var_7 = bool(var_2.seed == var_3.seed)
    assert var_7 is True
    var_8 = bool(var_2 is not var_3)
    assert var_8 is True



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_generic_init_line_16_predicate_false. Retrieved 5/19 statements.


def test_case_0():
    var_0 = 'Test that line 16 predicate evaluates to False for BaseDataProvider subclasses.'
    var_1 = 'mock_data'
    var_2 = 'mock_data'
    var_3 = '_mock_data'
    var_4 = None



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_reseed_attribute_error_handling. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'Test that AttributeError is caught and handled in reseed method.'
    var_1 = 42



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_generic_getattr_returns_provider_instance. Retrieved 1/7 statements.
# Failed to parse test_generic_getattr_caches_provider_instance.
# Failed to parse test_generic_getattr_initializes_with_correct_locale.
# Partially parsed test_generic_getattr_initializes_with_correct_seed. Retrieved 1/6 statements.
# Partially parsed test_generic_getattr_returns_none_for_non_callable_attribute. Retrieved 1/7 statements.
# Failed to parse test_generic_getattr_multiple_providers.


def test_case_0():
    var_0 = 'full_name'

def test_case_0():
    var_0 = 12345

def test_case_0():
    var_0 = 'nonexistent'



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_getattr_predicate_evaluates_to_false. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 8 evaluates to False when attribute is not callable.'



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_add_provider_with_meta_name_attribute. Retrieved 6/14 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 19 evaluates to False (no AttributeError).'
    var_1 = 'custom_provider'
    var_2 = module_0.Generic()
    var_3 = 'custom_provider'
    var_4 = hasattr(var_2, var_3)
    var_5 = bool(var_4)
    assert var_5 is True
    var_6 = var_2.custom_provider



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_generic_getattr_returns_provider_instance. Retrieved 2/8 statements.
# Partially parsed test_generic_getattr_caches_provider_instance. Retrieved 1/7 statements.
# Partially parsed test_generic_getattr_with_underscore_attribute. Retrieved 2/7 statements.
# Partially parsed test_generic_getattr_returns_none_for_invalid_attribute. Retrieved 2/7 statements.
# Partially parsed test_generic_getattr_initializes_with_correct_locale. Retrieved 1/6 statements.


def test_case_0():
    var_0 = 'Test that __getattr__ returns a provider instance when called.'
    var_1 = 'first_name'

def test_case_0():
    var_0 = 'Test that __getattr__ caches the provider instance after first access.'

def test_case_0():
    var_0 = 'Test that __getattr__ handles underscore-prefixed attributes correctly.'
    var_1 = '_person'

def test_case_0():
    var_0 = 'Test that __getattr__ returns None for invalid attributes.'
    var_1 = '_nonexistent_provider'

def test_case_0():
    var_0 = 'Test that __getattr__ initializes provider with correct locale.'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'Test that __getattr__ initializes provider with correct seed.'
    var_1 = 42
    var_2 = module_0.Generic(seed=var_1)
    var_3 = var_2.person
    var_4 = var_3.seed
    assert var_4 == 42



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_getattr_with_non_callable_attribute. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'Test that __getattr__ returns None when attribute is not callable.'
    var_1 = 'test_attr'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_generic_constructor_custom_locale. Retrieved 2/7 statements.
# Partially parsed test_generic_constructor_with_locale_and_seed. Retrieved 3/8 statements.
# Partially parsed test_generic_constructor_locale_attribute. Retrieved 1/5 statements.
# Partially parsed test_generic_constructor_seed_propagation. Retrieved 3/8 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'Test Generic constructor with default locale.'
    var_1 = module_0.Generic()
    var_2 = var_1.locale
    assert var_2 == 'en'
    var_3 = bool(var_1.seed is not None or var_1.seed is None)
    assert var_3 is True
    var_4 = 'random'
    var_5 = hasattr(var_1, var_4)
    var_6 = bool(var_5)
    assert var_6 is True

def test_case_0():
    var_0 = 'Test Generic constructor with custom locale.'
    var_1 = 'random'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'Test Generic constructor with custom seed.'
    var_1 = 42
    var_2 = module_0.Generic(seed=var_1)
    var_3 = var_2.seed
    assert var_3 == 42
    var_4 = 'random'
    var_5 = hasattr(var_2, var_4)
    var_6 = bool(var_5)
    assert var_6 is True

def test_case_0():
    var_0 = 'Test Generic constructor with both locale and seed.'
    var_1 = 123
    var_2 = 'random'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'Test that providers are initialized in Generic constructor.'
    var_1 = module_0.Generic()
    var_2 = var_1.__dir__()
    var_3 = len(var_2)
    var_4 = bool(var_3 > 0)
    assert var_4 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'Test that BaseDataProvider subclasses are stored as classes.'
    var_1 = module_0.Generic()
    var_2 = var_1.__dict__
    var_3 = '_'
    var_4 = [attr for attr in var_2 if attr.startswith(var_3)]
    var_5 = len(var_4)
    var_6 = bool(var_5 > 0)
    assert var_6 is True

def test_case_0():
    var_0 = 'Test that locale attribute is set correctly.'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'Test that seed is propagated to non-data providers.'
    var_1 = 999
    var_2 = module_0.Generic(seed=var_1)
    var_3 = var_2.seed
    assert var_3 == 999



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_add_provider_with_valid_provider. Retrieved 6/14 statements.
# Partially parsed test_add_provider_with_provider_without_meta_name. Retrieved 5/11 statements.
# Partially parsed test_add_provider_with_non_baseprovider_subclass. Retrieved 1/6 statements.
# Partially parsed test_add_provider_with_generic_instance. Retrieved 1/4 statements.
# Partially parsed test_add_provider_inherits_seed. Retrieved 4/9 statements.
# Partially parsed test_add_provider_with_kwargs. Retrieved 4/12 statements.
# Partially parsed test_add_provider_seed_kwarg_ignored. Retrieved 5/10 statements.
# Partially parsed test_add_provider_overwrites_existing. Retrieved 7/20 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom'
    var_1 = False
    var_2 = module_0.Generic()
    var_3 = 'custom'
    var_4 = hasattr(var_2, var_3)
    var_5 = bool(var_4)
    assert var_5 is True
    var_6 = var_2.custom

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Generic()
    var_2 = 'mycustomprovider'
    var_3 = hasattr(var_1, var_2)
    var_4 = bool(var_3)
    assert var_4 is True
    var_5 = var_1.mycustomprovider

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
    var_2 = 'subclass of mimesis.providers.BaseProvider'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'Cannot add Generic instance to itself'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom_seeded'
    var_1 = False
    var_2 = 42
    var_3 = module_0.Generic(seed=var_2)
    var_4 = var_3.custom_seeded.seed
    assert var_4 == 42

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom_kwargs'
    var_1 = False
    var_2 = module_0.Generic()
    var_3 = 'test_value'
    var_4 = var_2.custom_kwargs.custom_arg
    assert var_4 == 'test_value'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom_ignore_seed'
    var_1 = False
    var_2 = 42
    var_3 = module_0.Generic(seed=var_2)
    var_4 = 100
    var_5 = var_3.custom_ignore_seed.seed
    assert var_5 == 42

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom_overwrite'
    var_1 = False
    var_2 = 'custom_overwrite'
    var_3 = False
    var_4 = module_0.Generic()
    var_5 = var_4.custom_overwrite
    var_6 = var_4.custom_overwrite
    var_7 = bool(var_5 is not var_6)
    assert var_7 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_generic_init_skips_generic_provider. Retrieved 1/17 statements.


def test_case_0():
    var_0 = 'generic'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_add_provider_with_valid_provider. Retrieved 5/11 statements.
# Partially parsed test_add_provider_with_provider_without_meta_name. Retrieved 4/10 statements.
# Partially parsed test_add_provider_with_kwargs. Retrieved 3/11 statements.
# Partially parsed test_add_provider_with_seed_in_kwargs. Retrieved 4/9 statements.
# Partially parsed test_add_provider_not_subclass_of_base_provider. Retrieved 1/6 statements.
# Partially parsed test_add_provider_generic_instance. Retrieved 1/4 statements.
# Partially parsed test_add_provider_preserves_locale. Retrieved 1/8 statements.
# Partially parsed test_add_provider_multiple_providers. Retrieved 9/19 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom'
    var_1 = module_0.Generic()
    var_2 = 'custom'
    var_3 = hasattr(var_1, var_2)
    var_4 = bool(var_3)
    assert var_4 is True
    var_5 = var_1.custom

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'mycustomprovider'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = var_0.mycustomprovider

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom'
    var_1 = module_0.Generic()
    var_2 = 'test_value'
    var_3 = var_1.custom.custom_arg
    assert var_3 == 'test_value'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom'
    var_1 = 42
    var_2 = module_0.Generic(seed=var_1)
    var_3 = 999
    var_4 = var_2.custom.seed
    assert var_4 == 42

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
    var_2 = 'subclass of mimesis.providers.BaseProvider'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'Cannot add Generic instance to itself'

def test_case_0():
    var_0 = 'custom'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom1'
    var_1 = 'custom2'
    var_2 = module_0.Generic()
    var_3 = 'custom1'
    var_4 = hasattr(var_2, var_3)
    var_5 = bool(var_4)
    assert var_5 is True
    var_6 = 'custom2'
    var_7 = hasattr(var_2, var_6)
    var_8 = bool(var_7)
    assert var_8 is True
    var_9 = var_2.custom1
    var_10 = var_2.custom2



# Parsed testcases at query #7
#--------------------------

# Failed to parse test_generic_constructor_with_locale.
# Partially parsed test_generic_constructor_with_locale_and_seed. Retrieved 1/5 statements.
# Partially parsed test_generic_constructor_has_random_attribute. Retrieved 4/7 statements.
# Partially parsed test_generic_constructor_locale_attribute_exists. Retrieved 1/6 statements.
# Partially parsed test_generic_constructor_meta_attributes. Retrieved 2/6 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.locale
    var_2 = var_0.seed

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
    var_1 = var_0.__dir__()
    var_2 = len(var_1)
    var_3 = bool(var_2 > 0)
    assert var_3 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'random'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = var_0.random

def test_case_0():
    var_0 = 'locale'

def test_case_0():
    var_0 = 'Meta'
    var_1 = 'name'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_getattr_predicate_evaluates_to_false. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 8 evaluates to False when attribute is not callable.'
    var_1 = 'test_attr'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_reseed_with_missing_seed. Retrieved 2/5 statements.
# Partially parsed test_reseed_handles_attribute_errors. Retrieved 4/6 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Generic(seed=var_0)
    var_2 = var_1.seed
    var_3 = 123
    var_4 = var_1.reseed(var_3)
    var_5 = var_1.seed
    assert var_5 == 123
    var_6 = var_1.seed
    var_7 = bool(var_1.seed != var_2)
    assert var_7 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Generic(seed=var_0)
    var_2 = var_1.__dir__()
    var_3 = 'seed'
    var_4 = [getattr(var_1, attr) for attr in var_2 if hasattr(getattr(var_1, attr), var_3)]
    var_5 = 999
    var_6 = var_1.reseed(var_5)
    var_7 = var_1.__dir__()
    var_8 = [getattr(var_1, attr) for attr in var_7 if hasattr(getattr(var_1, attr), var_3)]

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Generic(seed=var_0)
    var_2 = var_1.seed
    var_3 = bool(var_1.seed is not None)
    assert var_3 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Generic(seed=var_0)
    var_2 = 555
    var_3 = var_1.reseed(var_2)
    var_4 = var_1.seed
    assert var_4 == 555

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.Generic(seed=var_0)
    var_2 = 777
    var_3 = var_1.reseed(var_2)
    var_4 = var_1.seed
    assert var_4 == 777



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_reseed_attribute_error_handling. Retrieved 5/13 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'Test that AttributeError is caught during reseed when provider lacks reseed method.'
    var_1 = 'custom_no_reseed'
    var_2 = module_0.Generic()
    var_3 = 12345
    var_4 = var_2.reseed(var_3)
    var_5 = var_2.seed
    assert var_5 == 12345



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_add_provider_with_meta_name_attribute. Retrieved 6/14 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 19 evaluates to False (no AttributeError is raised).'
    var_1 = 'custom_provider'
    var_2 = module_0.Generic()
    var_3 = 'custom_provider'
    var_4 = hasattr(var_2, var_3)
    var_5 = bool(var_4)
    assert var_5 is True
    var_6 = var_2.custom_provider



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_add_provider_with_valid_provider. Retrieved 3/11 statements.
# Partially parsed test_add_provider_without_meta_name. Retrieved 2/10 statements.
# Partially parsed test_add_provider_with_non_baseprovider_raises_type_error. Retrieved 1/6 statements.
# Partially parsed test_add_provider_with_generic_raises_type_error. Retrieved 1/4 statements.
# Partially parsed test_add_provider_with_seed. Retrieved 5/15 statements.
# Partially parsed test_add_provider_with_kwargs. Retrieved 4/15 statements.
# Partially parsed test_add_provider_removes_seed_kwarg. Retrieved 5/13 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom'
    var_1 = False
    var_2 = module_0.Generic()

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Generic()

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'not a class'
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
    var_2 = 'subclass of mimesis.providers.BaseProvider'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'Cannot add Generic instance to itself'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom'
    var_1 = False
    var_2 = 42
    var_3 = module_0.Generic(seed=var_2)
    var_4 = module_0.Generic(seed=var_2)

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom'
    var_1 = False
    var_2 = module_0.Generic()
    var_3 = 'test_value'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom'
    var_1 = False
    var_2 = 123
    var_3 = module_0.Generic(seed=var_2)
    var_4 = 999
    var_5 = var_3.custom.init_seed
    assert var_5 == 123



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_generic_getattr_returns_provider_instance. Retrieved 1/7 statements.
# Failed to parse test_generic_getattr_caches_provider_instance.
# Failed to parse test_generic_getattr_with_locale.
# Partially parsed test_generic_getattr_with_seed. Retrieved 1/6 statements.
# Failed to parse test_generic_getattr_nonexistent_attribute.
# Failed to parse test_generic_getattr_multiple_providers.


def test_case_0():
    var_0 = 'full_name'

def test_case_0():
    var_0 = 42



# Parsed testcases at query #14
#--------------------------

# Failed to parse test_generic_constructor_with_locale.
# Partially parsed test_generic_constructor_with_locale_and_seed. Retrieved 1/5 statements.
# Partially parsed test_generic_constructor_meta_class. Retrieved 1/3 statements.
# Partially parsed test_generic_constructor_locale_attribute_exists. Retrieved 1/6 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.locale
    var_2 = bool(var_0.seed is not None or var_0.seed is None)
    assert var_2 is True

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
    var_1 = 'random'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = var_0.random
    var_5 = bool(var_0.random is not None)
    assert var_5 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.__dir__()
    var_2 = len(var_1)
    var_3 = bool(var_2 > 0)
    assert var_3 is True

def test_case_0():
    var_0 = 'Meta'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Generic(seed=var_0)
    var_2 = module_0.Generic(seed=var_0)
    var_3 = var_1.seed
    var_4 = bool(var_1.seed == var_2.seed)
    assert var_4 is True

def test_case_0():
    var_0 = 'locale'



# Parsed testcases at query #15
#--------------------------




import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.locale
    assert var_1 == 'en'
    var_2 = bool(var_0.seed is not None or var_0.seed is None)
    assert var_2 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'fr'
    var_1 = module_0.Generic(var_0)
    var_2 = var_1.locale
    assert var_2 == 'fr'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 12345
    var_1 = module_0.Generic(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 12345

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'de'
    var_1 = 42
    var_2 = module_0.Generic(var_0, var_1)
    var_3 = var_2.locale
    assert var_3 == 'de'
    var_4 = var_2.seed
    assert var_4 == 42

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
    var_1 = 'random'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = var_0.random
    var_5 = bool(var_0.random is not None)
    assert var_5 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.Generic(seed=var_0)
    var_2 = var_1.seed
    assert var_2 is None

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'es'
    var_1 = module_0.Generic(var_0)
    var_2 = var_1.locale
    var_3 = bool(var_1.locale == var_0)
    assert var_3 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_generic_getattr_returns_provider_instance. Retrieved 1/7 statements.
# Failed to parse test_generic_getattr_caches_provider_instance.
# Failed to parse test_generic_getattr_initializes_with_locale.
# Partially parsed test_generic_getattr_initializes_with_seed. Retrieved 1/6 statements.
# Partially parsed test_generic_getattr_nonexistent_attribute_returns_none. Retrieved 1/6 statements.
# Failed to parse test_generic_getattr_stores_in_dict.


def test_case_0():
    var_0 = 'full_name'

def test_case_0():
    var_0 = 12345

def test_case_0():
    var_0 = 'nonexistent_provider'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_add_provider_with_meta_name_attribute. Retrieved 6/14 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 19 evaluates to False when Meta.name exists.'
    var_1 = 'custom_provider'
    var_2 = module_0.Generic()
    var_3 = 'custom_provider'
    var_4 = hasattr(var_2, var_3)
    var_5 = bool(var_4)
    assert var_5 is True
    var_6 = var_2.custom_provider



# Parsed testcases at query #18
#--------------------------

# Failed to parse test_generic_constructor_with_custom_locale.
# Partially parsed test_generic_constructor_with_locale_and_seed. Retrieved 1/5 statements.
# Partially parsed test_generic_constructor_providers_have_same_seed. Retrieved 3/8 statements.
# Partially parsed test_generic_constructor_excludes_generic_class. Retrieved 1/7 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.locale
    var_2 = var_0.seed
    var_3 = var_0.random
    var_4 = bool(var_0.random is not None)
    assert var_4 is True

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
    var_1 = dir(var_0)
    var_2 = len(var_1)
    var_3 = bool(var_2 > 0)
    assert var_3 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.Generic(seed=var_0)
    var_2 = var_1.seed
    assert var_2 is None

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Generic(seed=var_0)
    var_2 = dir(var_1)

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_generic_init_skips_generic_provider. Retrieved 3/17 statements.


def test_case_0():
    var_0 = 'generic'
    var_1 = None
    var_2 = False



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_getattr_with_non_callable_attribute. Retrieved 4/7 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'Test that __getattr__ returns None when attribute is not callable.'
    var_1 = module_0.Generic()
    var_2 = 'test_attr'
    var_3 = var_1.__getattr__(var_2)
    assert var_3 is None



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_generic_getattr_returns_provider_instance. Retrieved 1/7 statements.
# Failed to parse test_generic_getattr_caches_provider.
# Failed to parse test_generic_getattr_with_valid_locale.
# Partially parsed test_generic_getattr_with_seed. Retrieved 1/6 statements.
# Failed to parse test_generic_getattr_multiple_providers.
# Partially parsed test_generic_getattr_returns_none_for_invalid_attribute. Retrieved 1/6 statements.


def test_case_0():
    var_0 = 'full_name'

def test_case_0():
    var_0 = 42

def test_case_0():
    var_0 = 'nonexistent_provider'



# Parsed testcases at query #22
#--------------------------

# Failed to parse test_generic_constructor_with_locale.
# Partially parsed test_generic_constructor_with_locale_and_seed. Retrieved 1/5 statements.
# Partially parsed test_generic_constructor_has_random_attribute. Retrieved 2/5 statements.
# Partially parsed test_generic_constructor_locale_attribute_set. Retrieved 1/6 statements.
# Partially parsed test_generic_constructor_seed_propagated_to_random. Retrieved 5/8 statements.
# Partially parsed test_generic_constructor_underscore_providers_are_lazy. Retrieved 3/8 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.locale
    var_2 = var_0.seed
    var_3 = var_0.random
    var_4 = bool(var_0.random is not None)
    assert var_4 is True

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
    var_1 = var_0.__dir__()
    var_2 = len(var_1)
    var_3 = bool(var_2 > 0)
    assert var_3 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.__dir__()
    var_2 = 'generic'
    var_3 = bool('generic' not in var_1)
    assert var_3 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.random

def test_case_0():
    var_0 = 'locale'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 999
    var_1 = module_0.Generic(seed=var_0)
    var_2 = module_0.Generic(seed=var_0)
    var_3 = 1
    var_4 = 1000000

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = '_'
    var_2 = '_BaseProvider__dict__'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_generic_getattr_returns_provider_instance. Retrieved 2/8 statements.
# Partially parsed test_generic_getattr_caches_provider_instance. Retrieved 1/7 statements.
# Partially parsed test_generic_getattr_initializes_with_correct_locale. Retrieved 1/6 statements.
# Partially parsed test_generic_getattr_initializes_with_correct_seed. Retrieved 2/7 statements.
# Partially parsed test_generic_getattr_with_nonexistent_attribute. Retrieved 1/6 statements.
# Partially parsed test_generic_getattr_multiple_providers. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 'Test that __getattr__ returns a provider instance when accessed.'
    var_1 = 'first_name'

def test_case_0():
    var_0 = 'Test that __getattr__ caches the provider instance in __dict__.'

def test_case_0():
    var_0 = 'Test that __getattr__ initializes provider with correct locale.'

def test_case_0():
    var_0 = 'Test that __getattr__ initializes provider with correct seed.'
    var_1 = 12345

def test_case_0():
    var_0 = 'Test that __getattr__ returns None for non-existent provider.'

def test_case_0():
    var_0 = 'Test that __getattr__ correctly initializes multiple different providers.'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_reseed_all_accessible_providers. Retrieved 3/8 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.reseed()
    var_2 = var_0.seed
    var_3 = bool(var_0.seed is not None)
    assert var_3 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 12345
    var_2 = var_0.reseed(var_1)
    var_3 = var_0.seed
    var_4 = bool(var_0.seed == var_1)
    assert var_4 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 42
    var_2 = var_0.reseed(var_1)
    var_3 = var_0.person
    var_4 = var_3.seed
    var_5 = bool(var_3.seed == var_1)
    assert var_5 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 100
    var_2 = var_0.reseed(var_1)
    var_3 = var_0.seed
    assert var_3 == 100
    var_4 = 200
    var_5 = var_0.reseed(var_4)
    var_6 = var_0.seed
    assert var_6 == 200
    var_7 = bool(var_3 != var_6)
    assert var_7 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 12345
    var_1 = module_0.Generic(seed=var_0)
    var_2 = None
    var_3 = var_1.reseed(var_2)
    var_4 = var_1.seed
    assert var_4 is None

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 999
    var_2 = var_0.reseed(var_1)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_reseed_attribute_error_handling. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'Test that reseed handles AttributeError gracefully.'
    var_1 = 12345



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_generic_init_skips_generic_provider. Retrieved 2/15 statements.


def test_case_0():
    var_0 = 'generic'
    var_1 = None



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_reseed_with_added_custom_provider. Retrieved 5/10 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.reseed()
    var_2 = var_0.seed
    var_3 = bool(var_0.seed is not None)
    assert var_3 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 12345
    var_2 = var_0.reseed(var_1)
    var_3 = var_0.seed
    var_4 = bool(var_0.seed == var_1)
    assert var_4 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 42
    var_2 = var_0.reseed(var_1)
    var_3 = var_0.person
    var_4 = var_3.seed
    var_5 = bool(var_3.seed == var_1)
    assert var_5 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 999
    var_2 = var_0.reseed(var_1)
    var_3 = var_0.person
    var_4 = var_0.address
    var_5 = var_3.seed
    var_6 = bool(var_3.seed == var_1)
    assert var_6 is True
    var_7 = var_4.seed
    var_8 = bool(var_4.seed == var_1)
    assert var_8 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 100
    var_1 = module_0.Generic(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 100
    var_3 = 200
    var_4 = var_1.reseed(var_3)
    var_5 = var_1.seed
    assert var_5 == 200
    var_6 = 300
    var_7 = var_1.reseed(var_6)
    var_8 = var_1.seed
    assert var_8 == 300

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom'
    var_1 = module_0.Generic()
    var_2 = 555
    var_3 = var_1.reseed(var_2)
    var_4 = var_1.custom
    var_5 = var_4.seed
    var_6 = bool(var_4.seed == var_2)
    assert var_6 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 777
    var_2 = var_0.reseed(var_1)
    var_3 = var_0.seed
    assert var_3 == 777



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_generic_getattr_returns_provider_instance. Retrieved 1/7 statements.
# Failed to parse test_generic_getattr_caches_provider_instance.
# Failed to parse test_generic_getattr_with_locale.
# Partially parsed test_generic_getattr_with_seed. Retrieved 1/10 statements.
# Failed to parse test_generic_getattr_nonexistent_attribute.
# Partially parsed test_generic_getattr_initializes_with_correct_locale_and_seed. Retrieved 1/6 statements.


def test_case_0():
    var_0 = 'first_name'

def test_case_0():
    var_0 = 42

def test_case_0():
    var_0 = 123



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_add_provider_with_meta_name_attribute. Retrieved 5/11 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom_provider'
    var_1 = module_0.Generic()
    var_2 = 'custom_provider'
    var_3 = hasattr(var_1, var_2)
    var_4 = bool(var_3)
    assert var_4 is True
    var_5 = var_1.custom_provider



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_getattr_predicate_false_when_attribute_not_callable. Retrieved 3/6 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 8 evaluates to False when attribute is not callable.'
    var_1 = module_0.Generic()
    var_2 = var_1.test_attr
    assert var_2 is None



# Parsed testcases at query #31
#--------------------------

# Failed to parse test_generic_constructor_with_locale.
# Partially parsed test_generic_constructor_with_locale_and_seed. Retrieved 1/4 statements.
# Partially parsed test_generic_constructor_meta_attributes. Retrieved 1/3 statements.
# Partially parsed test_generic_constructor_creates_instance. Retrieved 3/6 statements.
# Failed to parse test_generic_constructor_locale_attribute_exists.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.locale
    var_2 = bool(var_0.seed is not None or var_0.seed is None)
    assert var_2 is True

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
    var_1 = 'random'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = var_0.random
    var_5 = bool(var_0.random is not None)
    assert var_5 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'locale'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = 'seed'
    var_5 = hasattr(var_0, var_4)
    var_6 = bool(var_5)
    assert var_6 is True
    var_7 = 'random'
    var_8 = hasattr(var_0, var_7)
    var_9 = bool(var_8)
    assert var_9 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 999
    var_1 = module_0.Generic(seed=var_0)
    var_2 = var_1.__dir__()
    var_3 = 0
    var_4 = var_2[var_3]
    var_5 = getattr(var_1, var_4)
    var_6 = var_5.seed
    assert var_6 == 999

def test_case_0():
    var_0 = 'Meta'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = str(var_0)
    var_2 = 'Generic'



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_generic_init_line_16_predicate_false. Retrieved 2/14 statements.


def test_case_0():
    var_0 = 'Test that line 16 predicate evaluates to False when provider_cls is BaseDataProvider.'
    var_1 = 'mock_data'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_reseed_updates_all_providers. Retrieved 3/7 statements.
# Partially parsed test_reseed_with_missing_seed. Retrieved 1/4 statements.
# Partially parsed test_reseed_with_default_locale. Retrieved 1/6 statements.
# Partially parsed test_reseed_preserves_locale. Retrieved 1/6 statements.
# Partially parsed test_reseed_all_provider_instances_have_same_seed. Retrieved 4/10 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 12345
    var_2 = var_0.reseed(var_1)
    var_3 = var_0.seed
    var_4 = bool(var_0.seed == var_1)
    assert var_4 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 42
    var_2 = var_0.reseed(var_1)

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.seed

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 111
    var_2 = var_0.reseed(var_1)
    var_3 = var_0.seed
    assert var_3 == 111
    var_4 = 222
    var_5 = var_0.reseed(var_4)
    var_6 = var_0.seed
    assert var_6 == 222
    var_7 = 333
    var_8 = var_0.reseed(var_7)
    var_9 = var_0.seed
    assert var_9 == 333

def test_case_0():
    var_0 = 999

def test_case_0():
    var_0 = 555

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 777
    var_2 = var_0.reseed(var_1)
    var_3 = []



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_add_provider_with_meta_name_attribute. Retrieved 5/13 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom_provider'
    var_1 = module_0.Generic()
    var_2 = 'custom_provider'
    var_3 = hasattr(var_1, var_2)
    var_4 = bool(var_3)
    assert var_4 is True
    var_5 = var_1.custom_provider



# Parsed testcases at query #35
#--------------------------

# Failed to parse test_generic_constructor_custom_locale.
# Partially parsed test_generic_constructor_with_locale_and_seed. Retrieved 1/5 statements.
# Partially parsed test_generic_constructor_seed_propagation. Retrieved 2/7 statements.
# Partially parsed test_generic_constructor_random_instance. Retrieved 2/5 statements.
# Partially parsed test_generic_constructor_does_not_register. Retrieved 1/4 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.locale
    var_2 = var_0.seed

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
    var_1 = var_0.__dir__()
    var_2 = len(var_1)
    var_3 = bool(var_2 > 0)
    assert var_3 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = '_person'
    var_2 = hasattr(var_0, var_1)
    var_3 = 'person'
    var_4 = hasattr(var_0, var_3)
    var_5 = var_0.__dir__()
    var_6 = len(var_5)
    var_7 = 0
    var_8 = var_6 > var_7
    var_9 = bool(var_2 or var_4 or var_8)
    assert var_9 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 999
    var_1 = module_0.Generic(seed=var_0)

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.random

def test_case_0():
    var_0 = 'generic'

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Generic(seed=var_0)
    var_2 = module_0.Generic(seed=var_0)
    var_3 = var_1.locale
    var_4 = bool(var_1.locale == var_2.locale)
    assert var_4 is True
    var_5 = var_1.seed
    var_6 = bool(var_1.seed == var_2.seed)
    assert var_6 is True



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_generic_getattr_returns_provider_instance. Retrieved 2/8 statements.
# Partially parsed test_generic_getattr_caches_provider_instance. Retrieved 1/7 statements.
# Partially parsed test_generic_getattr_initializes_with_correct_locale. Retrieved 1/6 statements.
# Partially parsed test_generic_getattr_initializes_with_correct_seed. Retrieved 2/7 statements.
# Partially parsed test_generic_getattr_invalid_provider_returns_none. Retrieved 1/6 statements.


def test_case_0():
    var_0 = 'Test that __getattr__ returns a provider instance for valid provider names.'
    var_1 = 'full_name'

def test_case_0():
    var_0 = 'Test that __getattr__ caches the provider instance after first access.'

def test_case_0():
    var_0 = 'Test that __getattr__ initializes provider with the correct locale.'

def test_case_0():
    var_0 = 'Test that __getattr__ initializes provider with the correct seed.'
    var_1 = 12345

def test_case_0():
    var_0 = 'Test that __getattr__ returns None for invalid provider names.'



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_getattr_with_non_callable_attribute. Retrieved 4/7 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'Test that __getattr__ returns None when attribute is not callable.'
    var_1 = module_0.Generic()
    var_2 = 'test_attr'
    var_3 = var_1.__getattr__(var_2)
    assert var_3 is None



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_generic_init_line_16_predicate_false. Retrieved 2/17 statements.


def test_case_0():
    var_0 = 'Test that line 16 predicate evaluates to False for BaseDataProvider subclasses.'
    var_1 = 'mock_data'



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_generic_getattr_returns_provider_instance. Retrieved 1/7 statements.
# Failed to parse test_generic_getattr_caches_provider_instance.
# Failed to parse test_generic_getattr_initializes_with_correct_locale.
# Partially parsed test_generic_getattr_initializes_with_correct_seed. Retrieved 1/6 statements.
# Partially parsed test_generic_getattr_returns_none_for_non_callable_attribute. Retrieved 1/7 statements.
# Partially parsed test_generic_getattr_with_multiple_providers. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 'full_name'

def test_case_0():
    var_0 = 42

def test_case_0():
    var_0 = 'nonexistent'

def test_case_0():
    var_0 = 123



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_add_provider_with_meta_name_attribute. Retrieved 5/13 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom_provider'
    var_1 = module_0.Generic()
    var_2 = 'custom_provider'
    var_3 = hasattr(var_1, var_2)
    var_4 = bool(var_3)
    assert var_4 is True
    var_5 = var_1.custom_provider



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_reseed_propagates_to_all_providers. Retrieved 3/7 statements.
# Partially parsed test_reseed_with_missing_seed. Retrieved 1/4 statements.
# Partially parsed test_reseed_preserves_locale. Retrieved 1/6 statements.
# Partially parsed test_reseed_with_custom_provider. Retrieved 6/11 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.reseed()
    var_2 = var_0.seed
    var_3 = bool(var_0.seed is not None)
    assert var_3 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 12345
    var_2 = var_0.reseed(var_1)
    var_3 = var_0.seed
    var_4 = bool(var_0.seed == var_1)
    assert var_4 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 42
    var_2 = var_0.reseed(var_1)

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 111
    var_2 = 222
    var_3 = var_0.reseed(var_1)
    var_4 = var_0.seed
    var_5 = bool(var_0.seed == var_1)
    assert var_5 is True
    var_6 = var_0.reseed(var_2)
    var_7 = var_0.seed
    var_8 = bool(var_0.seed == var_2)
    assert var_8 is True

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.seed
    var_2 = bool(var_0.seed is not None)
    assert var_2 is True

def test_case_0():
    var_0 = 999

import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom'
    var_1 = module_0.Generic()
    var_2 = 777
    var_3 = var_1.reseed(var_2)
    var_4 = 'custom'
    var_5 = getattr(var_1, var_4)
    var_6 = var_5.seed
    var_7 = bool(var_5.seed == var_2)
    assert var_7 is True



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_generic_init_skips_generic_provider. Retrieved 3/17 statements.


def test_case_0():
    var_0 = 'Test that Generic.__init__ skips Generic provider in registry loop.'
    var_1 = 'generic'
    var_2 = None



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_getattr_predicate_evaluates_to_false. Retrieved 4/7 statements.


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 8 evaluates to False when attribute is not callable.'
    var_1 = module_0.Generic()
    var_2 = 'test_attr'
    var_3 = var_1.__getattr__(var_2)
    assert var_3 is None



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_generic_init_line_16_predicate_false. Retrieved 2/10 statements.


def test_case_0():
    var_0 = 'Test that line 16 predicate evaluates to False when provider is BaseDataProvider.'
    var_1 = 'mock_data'



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_generic_getattr_returns_provider_instance. Retrieved 1/7 statements.
# Failed to parse test_generic_getattr_caches_provider_instance.
# Failed to parse test_generic_getattr_initializes_with_correct_locale.
# Partially parsed test_generic_getattr_initializes_with_correct_seed. Retrieved 1/6 statements.
# Partially parsed test_generic_getattr_returns_none_for_non_callable_attribute. Retrieved 1/6 statements.
# Failed to parse test_generic_getattr_multiple_providers.


def test_case_0():
    var_0 = 'full_name'

def test_case_0():
    var_0 = 12345

def test_case_0():
    var_0 = 'nonexistent_provider'



