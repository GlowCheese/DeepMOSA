####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_basefield_initialization_with_default_locale_and_seed. Retrieved 4/7 statements.
# Failed to parse test_basefield_initialization_with_custom_locale.
# Partially parsed test_basefield_initialization_generic_instance. Retrieved 2/3 statements.
# Failed to parse test_basefield_initialization_generic_locale_matches.


import mimesis.schema as module_0


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0.seed
    var_2 = var_0._generic.locale
    var_3 = var_0._cache
    var_4 = var_0._handlers
    var_5 = var_0.aliases
    var_6 = var_0.aliases
    var_7 = bool(var_0.aliases == {})
    assert var_7 is True


def test_case_0():
    var_0 = 12345
    var_1 = module_0.BaseField(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 12345


def test_case_0():
    var_0 = None
    var_1 = module_0.BaseField(seed=var_0)
    var_2 = var_1.seed
    assert var_2 is None


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0.aliases
    var_2 = bool(var_0.aliases == {})
    assert var_2 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0._cache
    var_2 = bool(var_0._cache == {})
    assert var_2 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0._handlers
    var_2 = bool(var_0._handlers == {})
    assert var_2 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0._generic


def test_case_0():
    var_0 = 999
    var_1 = module_0.BaseField(seed=var_0)
    var_2 = var_1._generic.seed
    assert var_2 == 999



# Parsed testcases at query #2
#--------------------------





def test_case_0():
    var_0 = 'id'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 5
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = var_5.iterations
    assert var_6 == 5
    var_7 = var_5._transformers
    var_8 = bool(var_5._transformers == [])
    assert var_8 is True
    var_9 = var_5._custom_context
    var_10 = bool(var_5._custom_context == {})
    assert var_10 is True


def test_case_0():
    var_0 = {}
    var_1 = lambda : var_0
    var_2 = 0
    var_3 = module_0.Schema(var_1, var_2)
    var_4 = bool(False)
    assert var_4 is True


def test_case_0():
    var_0 = 'not a callable'
    var_1 = module_0.Schema(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = bool(True)
    assert var_3 is True


def test_case_0():
    var_0 = {}
    var_1 = lambda : var_0
    var_2 = module_0.Schema(var_1)
    var_3 = var_2.iterations
    assert var_3 == 10


def test_case_0():
    var_0 = {}
    var_1 = lambda : var_0
    var_2 = 12345
    var_3 = module_0.Schema(var_1, seed=var_2)
    var_4 = var_3._Schema__seed
    var_5 = bool(var_3._Schema__seed == var_2)
    assert var_5 is True


def test_case_0():
    var_0 = {}
    var_1 = lambda : var_0
    var_2 = module_0.Schema(var_1)
    var_3 = var_2._Schema__seed



# Parsed testcases at query #3
#--------------------------





def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'handler1'
    var_2 = lambda random, **kwargs: var_1
    var_3 = 'handler2'
    var_4 = lambda random, **kwargs: var_3
    var_5 = 'field1'
    var_6 = (var_5, var_2)
    var_7 = 'field2'
    var_8 = (var_7, var_4)
    var_9 = [var_6, var_8]
    var_10 = var_0.register_handlers(var_9)
    var_11 = {}
    var_12 = var_0.perform(var_5, **var_11)
    assert var_12 == 'handler1'
    var_13 = {}
    var_14 = var_0.perform(var_7, **var_13)
    assert var_14 == 'handler2'


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'handler1'
    var_2 = lambda random, **kwargs: var_1
    var_3 = 'handler2'
    var_4 = lambda random, **kwargs: var_3
    var_5 = 'field1'
    var_6 = (var_5, var_2)
    var_7 = (var_5, var_4)
    var_8 = [var_6, var_7]
    var_9 = var_0.register_handlers(var_8)
    var_10 = {}
    var_11 = var_0.perform(var_5, **var_10)
    assert var_11 == 'handler1'


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = []
    var_2 = var_0.register_handlers(var_1)
    var_3 = var_0._handlers
    var_4 = bool(var_0._handlers == {})
    assert var_4 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'handler'
    var_2 = lambda random, **kwargs: var_1
    var_3 = 123
    var_4 = (var_3, var_2)
    var_5 = [var_4]
    var_6 = var_0.register_handlers(var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = bool(True)
    assert var_8 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'handler'
    var_2 = lambda random, **kwargs: var_1
    var_3 = '123field'
    var_4 = (var_3, var_2)
    var_5 = [var_4]
    var_6 = var_0.register_handlers(var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = bool(True)
    assert var_8 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'field1'
    var_2 = 'not_callable'
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = var_0.register_handlers(var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = bool(True)
    assert var_7 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'handler'
    var_2 = lambda random: var_1
    var_3 = 'field1'
    var_4 = (var_3, var_2)
    var_5 = [var_4]
    var_6 = var_0.register_handlers(var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = bool(True)
    assert var_8 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'handler1'
    var_2 = lambda random, **kwargs: var_1
    var_3 = 'handler2'
    var_4 = lambda random, **kwargs: var_3
    var_5 = 'field1'
    var_6 = var_0.register_handler(var_5, var_2)
    var_7 = (var_5, var_4)
    var_8 = [var_7]
    var_9 = var_0.register_handlers(var_8)
    var_10 = {}
    var_11 = var_0.perform(var_5, **var_10)
    assert var_11 == 'handler1'


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'valid'
    var_2 = lambda random, **kwargs: var_1
    var_3 = 'invalid'
    var_4 = 'field1'
    var_5 = (var_4, var_2)
    var_6 = 'field2'
    var_7 = (var_6, var_3)
    var_8 = [var_5, var_7]
    var_9 = var_0.register_handlers(var_8)
    var_10 = bool(False)
    assert var_10 is True
    var_11 = bool(True)
    assert var_11 is True
    var_12 = {}
    var_13 = var_0.perform(var_4, **var_12)
    assert var_13 == 'valid'



# Parsed testcases at query #4
#--------------------------






# Parsed testcases at query #5
#--------------------------






# Parsed testcases at query #6
#--------------------------

# Partially parsed test_pick_from_raises_value_error_when_builder_is_none. Retrieved 7/8 statements.
# Partially parsed test_pick_from_calls_builder_pick_from_with_correct_arguments. Retrieved 3/8 statements.
# Partially parsed test_pick_from_calls_builder_pick_from_without_field. Retrieved 5/10 statements.



def test_case_0():
    var_0 = 0
    var_1 = None
    var_2 = module_0.SchemaContext(var_0, builder=var_1)
    var_3 = None
    var_4 = 'some_schema'
    var_5 = var_2.pick_from(var_4)
    var_6 = bool(var_3 is not None)
    assert var_6 is True
    var_7 = str(var_3)
    assert var_7 == 'pick_from() requires SchemaBuilder'

def test_case_0():
    var_0 = 0
    var_1 = 'test_schema'
    var_2 = 'test_field'

def test_case_0():
    var_0 = 'id'
    var_1 = 1
    var_2 = 0
    var_3 = 'test_schema'
    var_4 = None



# Parsed testcases at query #7
#--------------------------






# Parsed testcases at query #8
#--------------------------

# Partially parsed test_basefield_initializes_with_default_locale_and_seed. Retrieved 4/7 statements.
# Failed to parse test_basefield_initializes_with_custom_locale.
# Partially parsed test_basefield_initializes_with_generic_instance. Retrieved 2/3 statements.
# Failed to parse test_basefield_initializes_with_same_locale_in_generic.



def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0.seed
    var_2 = var_0._generic.locale
    var_3 = var_0._cache
    var_4 = var_0._handlers
    var_5 = var_0.aliases


def test_case_0():
    var_0 = 12345
    var_1 = module_0.BaseField(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 12345


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0._cache
    var_2 = bool(var_0._cache == {})
    assert var_2 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0._handlers
    var_2 = bool(var_0._handlers == {})
    assert var_2 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0.aliases
    var_2 = bool(var_0.aliases == {})
    assert var_2 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0._generic


def test_case_0():
    var_0 = 999
    var_1 = module_0.BaseField(seed=var_0)
    var_2 = var_1._generic.seed
    assert var_2 == 999



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_constructor_without_seed. Retrieved 2/3 statements.
# Partially parsed test_constructor_with_seed. Retrieved 3/4 statements.



def test_case_0():
    var_0 = module_0.SchemaBuilder()
    var_1 = var_0._seed
    var_2 = var_0._random
    var_3 = var_0._schemas
    var_4 = bool(var_0._schemas == {})
    assert var_4 is True
    var_5 = var_0._data
    var_6 = bool(var_0._data == {})
    assert var_6 is True


def test_case_0():
    var_0 = 42
    var_1 = module_0.SchemaBuilder(var_0)
    var_2 = var_1._seed
    assert var_2 == 42
    var_3 = var_1._random
    var_4 = var_1._random.seed
    assert var_4 == 42
    var_5 = var_1._schemas
    var_6 = bool(var_1._schemas == {})
    assert var_6 is True
    var_7 = var_1._data
    var_8 = bool(var_1._data == {})
    assert var_8 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_ref_calls_builder_get_data_with_correct_schema_name. Retrieved 11/15 statements.
# Partially parsed test_ref_returns_empty_list_when_builder_returns_empty_list. Retrieved 8/12 statements.
# Partially parsed test_ref_passes_schema_name_to_builder. Retrieved 6/15 statements.
# Partially parsed test_ref_works_with_custom_context_data. Retrieved 15/19 statements.



def test_case_0():
    var_0 = 0
    var_1 = None
    var_2 = module_0.SchemaContext(var_0, builder=var_1)
    var_3 = 'some_schema'
    var_4 = var_2.ref(var_3)
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = 'MockBuilder'
    var_1 = ()
    var_2 = '_get_data'
    var_3 = 'id'
    var_4 = 1
    var_5 = {var_3: var_4}
    var_6 = [var_5]
    var_7 = lambda schema_name: var_6
    var_8 = {var_2: var_7}
    var_9 = [var_0, var_1, var_8]
    var_10 = 0
    var_11 = 'test_schema'

def test_case_0():
    var_0 = 'MockBuilder'
    var_1 = ()
    var_2 = '_get_data'
    var_3 = []
    var_4 = lambda schema_name: var_3
    var_5 = {var_2: var_4}
    var_6 = [var_0, var_1, var_5]
    var_7 = 0
    var_8 = 'empty_schema'

def test_case_0():
    var_0 = None
    assert var_0 == 'my_schema'
    var_1 = 'MockBuilder'
    var_2 = ()
    var_3 = '_get_data'
    var_4 = 0
    var_5 = 'my_schema'

def test_case_0():
    var_0 = 'MockBuilder'
    var_1 = ()
    var_2 = '_get_data'
    var_3 = 'data'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = [var_5]
    var_7 = lambda schema_name: var_6
    var_8 = {var_2: var_7}
    var_9 = [var_0, var_1, var_8]
    var_10 = 5
    var_11 = 123
    var_12 = 'key'
    var_13 = 'val'
    var_14 = {var_12: var_13}
    var_15 = 'schema'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_create_returns_list_of_dicts_from_schema. Retrieved 7/11 statements.
# Partially parsed test_create_applies_transformers_with_context. Retrieved 7/11 statements.
# Partially parsed test_create_skips_none_results. Retrieved 2/9 statements.
# Partially parsed test_create_with_custom_context. Retrieved 8/14 statements.



def test_case_0():
    var_0 = 'id'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 5
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = var_5.create()
    var_7 = len(var_6)
    assert var_7 == 5


def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 3
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = var_5.create()


def test_case_0():
    var_0 = 'value'
    var_1 = 0
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 2
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = 1
    var_7 = lambda item: {var_0: item[var_0] + var_6}
    var_8 = var_5.map(var_7)
    var_9 = var_5.create()
    var_10 = bool(var_9 == [{'value': 1}, {'value': 1}])
    assert var_10 is True


def test_case_0():
    var_0 = 'data'
    var_1 = 'x'
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 3
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = var_5.create()
    var_7 = [item['index'] for item in var_6]
    var_8 = bool([item['index'] for item in var_6] == [0, 1, 2])
    assert var_8 is True

def test_case_0():
    var_0 = 0
    var_1 = 2


def test_case_0():
    var_0 = {}
    var_1 = lambda : var_0
    var_2 = 2
    var_3 = module_0.Schema(var_1, var_2)
    var_4 = 'value'
    var_5 = 'key'
    var_6 = {var_5: var_4}
    var_7 = var_3.with_context(**var_6)
    var_8 = var_3.create()
    var_9 = 'custom'


def test_case_0():
    var_0 = {}
    var_1 = lambda : var_0
    var_2 = 0
    var_3 = module_0.Schema(var_1, var_2)
    var_4 = bool(False)
    assert var_4 is True


def test_case_0():
    var_0 = 'not callable'
    var_1 = module_0.Schema(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = bool(True)
    assert var_3 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_constructor_without_seed. Retrieved 2/3 statements.
# Partially parsed test_constructor_with_seed. Retrieved 3/4 statements.
# Partially parsed test_constructor_with_none_seed. Retrieved 3/4 statements.
# Partially parsed test_constructor_with_string_seed. Retrieved 3/4 statements.



def test_case_0():
    var_0 = module_0.SchemaBuilder()
    var_1 = var_0._seed
    var_2 = var_0._random
    var_3 = var_0._schemas
    var_4 = bool(var_0._schemas == {})
    assert var_4 is True
    var_5 = var_0._data
    var_6 = bool(var_0._data == {})
    assert var_6 is True


def test_case_0():
    var_0 = 12345
    var_1 = module_0.SchemaBuilder(var_0)
    var_2 = var_1._seed
    var_3 = bool(var_1._seed == var_0)
    assert var_3 is True
    var_4 = var_1._random
    var_5 = var_1._random.seed
    var_6 = bool(var_1._random.seed == var_0)
    assert var_6 is True
    var_7 = var_1._schemas
    var_8 = bool(var_1._schemas == {})
    assert var_8 is True
    var_9 = var_1._data
    var_10 = bool(var_1._data == {})
    assert var_10 is True


def test_case_0():
    var_0 = None
    var_1 = module_0.SchemaBuilder(var_0)
    var_2 = var_1._seed
    assert var_2 is None
    var_3 = var_1._random
    var_4 = var_1._random.seed
    assert var_4 is None
    var_5 = var_1._schemas
    var_6 = bool(var_1._schemas == {})
    assert var_6 is True
    var_7 = var_1._data
    var_8 = bool(var_1._data == {})
    assert var_8 is True


def test_case_0():
    var_0 = 'test_seed'
    var_1 = module_0.SchemaBuilder(var_0)
    var_2 = var_1._seed
    var_3 = bool(var_1._seed == var_0)
    assert var_3 is True
    var_4 = var_1._random
    var_5 = var_1._random.seed
    var_6 = bool(var_1._random.seed == var_0)
    assert var_6 is True
    var_7 = var_1._schemas
    var_8 = bool(var_1._schemas == {})
    assert var_8 is True
    var_9 = var_1._data
    var_10 = bool(var_1._data == {})
    assert var_10 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_handle_returns_decorated_function. Retrieved 2/5 statements.
# Partially parsed test_handle_does_not_overwrite_existing_handler. Retrieved 2/7 statements.



def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'custom_handler'
    var_2 = bool('custom_handler' in var_0._handlers)
    assert var_2 is True
    var_3 = var_0._handlers['custom_handler']


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'my_field'
    var_2 = bool('my_field' in var_0._handlers)
    assert var_2 is True
    var_3 = var_0._handlers['my_field']


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0.get_random_instance()


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'custom_handler'
    var_2 = {}
    var_3 = var_0.perform(var_1, **var_2)
    assert var_3 == 'custom_value'


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'custom_handler'
    var_2 = 'test_'
    var_3 = 'prefix'
    var_4 = {var_3: var_2}
    var_5 = var_0.perform(var_1, **var_4)
    assert var_5 == 'test_value'


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = bool(False)
    assert var_1 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = bool(False)
    assert var_1 is True
    var_2 = '123invalid'


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0.handle()
    var_2 = 123
    var_3 = var_1(var_2)
    var_4 = bool(False)
    assert var_4 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = bool(False)
    assert var_1 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'my_field'
    var_2 = var_0._handlers['my_field']



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_fieldset_call_with_default_iterations. Retrieved 1/6 statements.
# Partially parsed test_fieldset_call_with_custom_iterations_at_instance. Retrieved 2/5 statements.
# Partially parsed test_fieldset_call_with_custom_iterations_at_call. Retrieved 2/5 statements.
# Partially parsed test_fieldset_call_with_iterations_override_at_call. Retrieved 3/6 statements.
# Partially parsed test_fieldset_call_with_zero_iterations_raises_error. Retrieved 2/5 statements.
# Partially parsed test_fieldset_call_with_negative_iterations_raises_error. Retrieved 2/5 statements.
# Partially parsed test_fieldset_call_with_custom_iterations_kwarg. Retrieved 3/7 statements.
# Partially parsed test_fieldset_call_with_custom_default_iterations. Retrieved 2/6 statements.
# Partially parsed test_fieldset_call_passes_arguments_to_perform. Retrieved 7/16 statements.
# Partially parsed test_fieldset_call_returns_list_of_perform_results. Retrieved 3/12 statements.


def test_case_0():
    var_0 = []
    var_1 = 'username'

def test_case_0():
    var_0 = 5
    var_1 = []
    var_2 = 'username'

def test_case_0():
    var_0 = []
    var_1 = 'username'
    var_2 = 3

def test_case_0():
    var_0 = 7
    var_1 = []
    var_2 = 'username'
    var_3 = 2

def test_case_0():
    var_0 = []
    var_1 = 'username'
    var_2 = 0
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = []
    var_1 = 'username'
    var_2 = -5
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'iter'
    var_1 = 4
    var_2 = 'username'

def test_case_0():
    var_0 = 6
    var_1 = 'username'

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'arg1'
    var_3 = 'arg2'
    var_4 = 'val1'
    var_5 = 'val2'
    var_6 = 2
    var_7 = len(var_0)
    assert var_7 == 2
    var_8 = var_0[0]
    var_9 = bool(var_0[0] == (('arg1', 'arg2'), {'key1': 'val1', 'key2': 'val2'}))
    assert var_9 is True

def test_case_0():
    var_0 = 0
    var_1 = []
    var_2 = 'test'
    var_3 = 3



# Parsed testcases at query #15
#--------------------------






# Parsed testcases at query #16
#--------------------------

# Partially parsed test_create_returns_correct_number_of_items. Retrieved 1/6 statements.
# Partially parsed test_create_skips_none_results. Retrieved 2/9 statements.
# Partially parsed test_create_with_iterations_one. Retrieved 1/6 statements.
# Partially parsed test_create_with_large_iterations. Retrieved 1/6 statements.
# Partially parsed test_create_with_transformers. Retrieved 1/11 statements.


def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 0
    var_1 = 3
    var_2 = bool(var_0 > 3)
    assert var_2 is True

def test_case_0():
    var_0 = 1

def test_case_0():
    var_0 = 100

def test_case_0():
    var_0 = 3



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_constructor_without_seed. Retrieved 2/3 statements.
# Partially parsed test_constructor_with_seed. Retrieved 3/4 statements.



def test_case_0():
    var_0 = module_0.SchemaBuilder()
    var_1 = var_0._seed
    var_2 = var_0._random
    var_3 = var_0._schemas
    var_4 = bool(var_0._schemas == {})
    assert var_4 is True
    var_5 = var_0._data
    var_6 = bool(var_0._data == {})
    assert var_6 is True


def test_case_0():
    var_0 = 12345
    var_1 = module_0.SchemaBuilder(var_0)
    var_2 = var_1._seed
    assert var_2 == 12345
    var_3 = var_1._random
    var_4 = var_1._schemas
    var_5 = bool(var_1._schemas == {})
    assert var_5 is True
    var_6 = var_1._data
    var_7 = bool(var_1._data == {})
    assert var_7 is True



# Parsed testcases at query #18
#--------------------------






# Parsed testcases at query #19
#--------------------------

# Partially parsed test_create_generates_data_for_specified_schemas. Retrieved 11/15 statements.
# Partially parsed test_create_raises_error_for_undefined_schema. Retrieved 5/8 statements.
# Partially parsed test_create_preserves_original_schema_transformers. Retrieved 5/8 statements.
# Partially parsed test_create_preserves_original_schema_iterations. Retrieved 5/8 statements.
# Partially parsed test_create_stores_data_internal_data. Retrieved 7/9 statements.
# Partially parsed test_create_returns_correct_data_structure. Retrieved 6/10 statements.
# Partially parsed test_create_with_zero_count_generates_empty_list. Retrieved 5/7 statements.



def test_case_0():
    var_0 = 42
    var_1 = module_0.SchemaBuilder(var_0)
    var_2 = 'users'
    var_3 = 'products'
    var_4 = 5
    var_5 = 3
    var_6 = 'users'
    var_7 = 'products'
    var_8 = {var_6: var_4, var_7: var_5}
    var_9 = var_1.create(**var_8)
    var_10 = 'users'
    var_11 = bool('users' in var_9)
    assert var_11 is True
    var_12 = 'products'
    var_13 = bool('products' in var_9)
    assert var_13 is True
    var_14 = var_9[var_2]
    var_15 = len(var_14)
    assert var_15 == 5
    var_16 = var_9[var_3]
    var_17 = len(var_16)
    assert var_17 == 3


def test_case_0():
    var_0 = module_0.SchemaBuilder()
    var_1 = 'users'
    var_2 = 2
    var_3 = 1
    var_4 = 'users'
    var_5 = 'products'
    var_6 = {var_4: var_2, var_5: var_3}
    var_7 = var_0.create(**var_6)


def test_case_0():
    var_0 = 123
    var_1 = module_0.SchemaBuilder(var_0)
    var_2 = 'test'
    var_3 = 2
    var_4 = 'test'
    var_5 = {var_4: var_3}
    var_6 = var_1.create(**var_5)


def test_case_0():
    var_0 = 456
    var_1 = module_0.SchemaBuilder(var_0)
    var_2 = 'test'
    var_3 = 3
    var_4 = 'test'
    var_5 = {var_4: var_3}
    var_6 = var_1.create(**var_5)


def test_case_0():
    var_0 = 789
    var_1 = module_0.SchemaBuilder(var_0)
    var_2 = 'items'
    var_3 = 4
    var_4 = 'items'
    var_5 = {var_4: var_3}
    var_6 = var_1.create(**var_5)
    var_7 = 'items'
    var_8 = bool('items' in var_1._data)
    assert var_8 is True
    var_9 = var_1._data[var_2]
    var_10 = len(var_9)
    assert var_10 == 4


def test_case_0():
    var_0 = 999
    var_1 = module_0.SchemaBuilder(var_0)
    var_2 = 'data'
    var_3 = 1
    var_4 = 'data'
    var_5 = {var_4: var_3}
    var_6 = var_1.create(**var_5)
    var_7 = 'data'
    var_8 = bool('data' in var_6)
    assert var_8 is True
    var_9 = var_6[var_2]


def test_case_0():
    var_0 = 111
    var_1 = module_0.SchemaBuilder(var_0)
    var_2 = 'empty'
    var_3 = 0
    var_4 = 'empty'
    var_5 = {var_4: var_3}
    var_6 = var_1.create(**var_5)
    var_7 = var_6['empty']
    var_8 = bool(var_6['empty'] == [])
    assert var_8 is True



# Parsed testcases at query #20
#--------------------------






# Parsed testcases at query #21
#--------------------------

# Partially parsed test_fieldset_raises_error_when_iterations_less_than_one. Retrieved 2/5 statements.


def test_case_0():
    var_0 = []
    var_1 = 'test'
    var_2 = 0
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_to_json_writes_correct_data_to_file. Retrieved 9/14 statements.
# Partially parsed test_to_json_handles_json_dump_kwargs. Retrieved 10/16 statements.
# Partially parsed test_to_json_with_empty_schema_output. Retrieved 5/10 statements.
# Partially parsed test_to_json_applies_transformers. Retrieved 9/14 statements.
# Partially parsed test_to_json_with_custom_context_in_transformers. Retrieved 7/16 statements.
# Partially parsed test_to_json_filters_none_results. Retrieved 3/13 statements.
# Partially parsed test_to_json_uses_utf8_encoding. Retrieved 7/12 statements.
# Partially parsed test_to_json_with_seed_does_not_affect_output. Retrieved 8/13 statements.



def test_case_0():
    var_0 = 'id'
    var_1 = 'name'
    var_2 = 1
    var_3 = 'test'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = lambda : var_4
    var_6 = 2
    var_7 = module_0.Schema(var_5, var_6)
    var_8 = 'output.json'


def test_case_0():
    var_0 = 'id'
    var_1 = 'name'
    var_2 = 1
    var_3 = 'test'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = lambda : var_4
    var_6 = module_0.Schema(var_5, var_2)
    var_7 = 'output.json'
    var_8 = 4
    var_9 = '[\n    {'


def test_case_0():
    var_0 = {}
    var_1 = lambda : var_0
    var_2 = 3
    var_3 = module_0.Schema(var_1, var_2)
    var_4 = 'output.json'


def test_case_0():
    var_0 = 'value'
    var_1 = 5
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 2
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = lambda item: {var_0: item[var_0] * var_4}
    var_7 = var_5.map(var_6)
    var_8 = 'output.json'


def test_case_0():
    var_0 = 'index'
    var_1 = 0
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 2
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = 'output.json'

def test_case_0():
    var_0 = 0
    var_1 = 3
    var_2 = 'output.json'


def test_case_0():
    var_0 = 'text'
    var_1 = 'café'
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 1
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = 'output.json'


def test_case_0():
    var_0 = 'random'
    var_1 = 42
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 2
    var_5 = 12345
    var_6 = module_0.Schema(var_3, var_4, var_5)
    var_7 = 'output.json'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_basefield_initialization_with_defaults. Retrieved 2/3 statements.
# Failed to parse test_basefield_initialization_with_custom_locale.



def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0.seed
    var_2 = var_0._generic
    var_3 = var_0._generic.locale
    var_4 = var_0._cache
    var_5 = bool(var_0._cache == {})
    assert var_5 is True
    var_6 = var_0._handlers
    var_7 = bool(var_0._handlers == {})
    assert var_7 is True
    var_8 = var_0.aliases
    var_9 = bool(var_0.aliases == {})
    assert var_9 is True


def test_case_0():
    var_0 = 12345
    var_1 = module_0.BaseField(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 12345
    var_3 = var_1._generic.seed
    assert var_3 == 12345


def test_case_0():
    var_0 = None
    var_1 = module_0.BaseField(seed=var_0)
    var_2 = var_1.seed
    assert var_2 is None
    var_3 = var_1._generic.seed
    assert var_3 is None


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0.aliases
    var_2 = bool(var_0.aliases == {})
    assert var_2 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0._cache
    var_2 = bool(var_0._cache == {})
    assert var_2 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0._handlers
    var_2 = bool(var_0._handlers == {})
    assert var_2 is True



# Parsed testcases at query #24
#--------------------------






# Parsed testcases at query #25
#--------------------------

# Partially parsed test_basefield_initialization_with_defaults. Retrieved 4/7 statements.
# Failed to parse test_basefield_initialization_with_custom_locale.
# Partially parsed test_basefield_initialization_generic_instance. Retrieved 2/3 statements.



def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0.seed
    var_2 = var_0._generic.locale
    var_3 = var_0._cache
    var_4 = var_0._cache
    var_5 = bool(var_0._cache == {})
    assert var_5 is True
    var_6 = var_0._handlers
    var_7 = var_0._handlers
    var_8 = bool(var_0._handlers == {})
    assert var_8 is True
    var_9 = var_0.aliases
    var_10 = var_0.aliases
    var_11 = bool(var_0.aliases == {})
    assert var_11 is True


def test_case_0():
    var_0 = 12345
    var_1 = module_0.BaseField(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 12345


def test_case_0():
    var_0 = None
    var_1 = module_0.BaseField(seed=var_0)
    var_2 = var_1.seed
    assert var_2 is None


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0.aliases
    var_2 = bool(var_0.aliases == {})
    assert var_2 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0._cache
    var_2 = bool(var_0._cache == {})
    assert var_2 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0._handlers
    var_2 = bool(var_0._handlers == {})
    assert var_2 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0._generic



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_create_returns_correct_number_of_items. Retrieved 4/11 statements.
# Partially parsed test_create_skips_none_results. Retrieved 2/9 statements.
# Partially parsed test_create_with_transformer. Retrieved 3/14 statements.
# Partially parsed test_create_with_context_in_transformer. Retrieved 1/11 statements.
# Partially parsed test_create_with_custom_context. Retrieved 3/15 statements.
# Partially parsed test_create_iterations_one. Retrieved 1/6 statements.
# Partially parsed test_create_iterations_large. Retrieved 3/10 statements.
# Partially parsed test_create_with_seed. Retrieved 2/9 statements.
# Partially parsed test_create_empty_schema. Retrieved 2/9 statements.
# Partially parsed test_create_with_nested_transformations. Retrieved 3/18 statements.


def test_case_0():
    var_0 = 5
    var_1 = 'id'
    var_2 = 1
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 0
    var_1 = 3

def test_case_0():
    var_0 = 4
    var_1 = 'value'
    var_2 = 11

def test_case_0():
    var_0 = 3

def test_case_0():
    var_0 = 2
    var_1 = 'value'
    var_2 = 'custom'

def test_case_0():
    var_0 = 1

def test_case_0():
    var_0 = 1000
    var_1 = 'x'
    var_2 = 0

def test_case_0():
    var_0 = 5
    var_1 = 42

def test_case_0():
    var_0 = 7
    var_1 = {}

def test_case_0():
    var_0 = 3
    var_1 = 'count'
    var_2 = 2



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_create_returns_list_of_correct_length. Retrieved 7/9 statements.
# Partially parsed test_create_with_custom_iterations. Retrieved 8/10 statements.
# Partially parsed test_create_applies_transformers. Retrieved 7/15 statements.
# Partially parsed test_create_applies_transformers_with_context. Retrieved 6/15 statements.
# Partially parsed test_create_with_custom_context. Retrieved 8/16 statements.
# Partially parsed test_create_skips_none_results. Retrieved 5/14 statements.
# Partially parsed test_create_with_seed. Retrieved 1/8 statements.



def test_case_0():
    var_0 = 'id'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = module_0.Schema(var_3)
    var_5 = var_4.create()
    var_6 = len(var_5)
    assert var_6 == 10


def test_case_0():
    var_0 = 'value'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 5
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = var_5.create()
    var_7 = len(var_6)
    assert var_7 == 5


def test_case_0():
    var_0 = 'id'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = module_0.Schema(var_3)
    var_5 = 'transformed'
    var_6 = True


def test_case_0():
    var_0 = 'id'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = module_0.Schema(var_3)
    var_5 = 'index'


def test_case_0():
    var_0 = 'id'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = module_0.Schema(var_3)
    var_5 = 'value'
    var_6 = 'key'
    var_7 = {var_6: var_5}
    var_8 = var_4.with_context(**var_7)
    var_9 = 'custom'

def test_case_0():
    var_0 = 0
    assert var_0 == 5
    var_1 = 3
    var_2 = 'id'
    var_3 = 2
    var_4 = 1

def test_case_0():
    var_0 = 42


def test_case_0():
    var_0 = {}
    var_1 = lambda : var_0
    var_2 = 0
    var_3 = module_0.Schema(var_1, var_2)
    var_4 = bool(False)
    assert var_4 is True


def test_case_0():
    var_0 = 'not callable'
    var_1 = module_0.Schema(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = bool(True)
    assert var_3 is True



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_random_initialized_with_seed. Retrieved 3/4 statements.



def test_case_0():
    var_0 = 42
    var_1 = module_0.SchemaBuilder(var_0)
    var_2 = var_1._seed
    assert var_2 == 42
    var_3 = var_1._random



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_fieldset_raises_error_when_iterations_less_than_one. Retrieved 2/5 statements.


def test_case_0():
    var_0 = []
    var_1 = 'username'
    var_2 = 0
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_next_returns_items_until_iterations_reached. Retrieved 10/14 statements.
# Partially parsed test_next_skips_none_results. Retrieved 4/12 statements.
# Partially parsed test_next_respects_transformers. Retrieved 11/15 statements.
# Partially parsed test_next_uses_custom_context_in_transformers. Retrieved 8/14 statements.
# Partially parsed test_next_with_seed_produces_deterministic_results. Retrieved 4/10 statements.
# Partially parsed test_next_works_with_empty_custom_context. Retrieved 9/11 statements.
# Partially parsed test_next_with_custom_context. Retrieved 11/13 statements.
# Partially parsed test_next_after_iter_reset. Retrieved 12/14 statements.



def test_case_0():
    var_0 = 'id'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 3
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = []
    var_7 = next(var_5)
    var_8 = len(var_6)
    assert var_8 == 3
    var_9 = {var_7: var_1}


def test_case_0():
    var_0 = 'id'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = module_0.Schema(var_3, var_1)
    var_5 = next(var_4)
    var_6 = next(var_4)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = bool(True)
    assert var_8 is True

def test_case_0():
    var_0 = 0
    var_1 = 2
    var_2 = []
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = var_2[0]
    var_5 = bool(var_2[0] == {'id': 1})
    assert var_5 is True
    var_6 = var_2[1]
    var_7 = bool(var_2[1] == {'id': 3})
    assert var_7 is True


def test_case_0():
    var_0 = 'value'
    var_1 = 5
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 2
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = lambda item: {var_0: item[var_0] * var_4}
    var_7 = var_5.map(var_6)
    var_8 = []
    var_9 = next(var_5)
    var_10 = 10


def test_case_0():
    var_0 = 'data'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 3
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = []
    var_7 = next(var_5)
    var_8 = [item['index'] for item in var_6]
    var_9 = bool([item['index'] for item in var_6] == [0, 1, 2])
    assert var_9 is True

def test_case_0():
    var_0 = 3
    var_1 = 42
    var_2 = range(var_0)
    var_3 = range(var_0)


def test_case_0():
    var_0 = 'counter'
    var_1 = 0
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 5
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = next(var_5)
    var_7 = var_5._Schema__counter
    assert var_7 == 5


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 2
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = range(var_4)
    var_7 = [next(var_5) for _ in var_6]
    var_8 = len(var_7)
    assert var_8 == 2


def test_case_0():
    var_0 = 'data'
    var_1 = 'original'
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 2
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = 'info'
    var_7 = 'extra'
    var_8 = {var_7: var_6}
    var_9 = var_5.with_context(**var_8)
    var_10 = range(var_4)
    var_11 = [next(var_5) for _ in var_10]
    var_12 = len(var_11)
    assert var_12 == 2


def test_case_0():
    var_0 = 'id'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 2
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = next(var_5)
    var_7 = iter(var_5)
    var_8 = range(var_4)
    var_9 = [next(var_5) for _ in var_8]
    var_10 = len(var_9)
    assert var_10 == 2
    var_11 = {var_0: var_1}



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_create_skips_none_results. Retrieved 1/6 statements.
# Partially parsed test_create_includes_non_none_results. Retrieved 4/11 statements.
# Partially parsed test_create_handles_mixed_none_and_valid. Retrieved 5/14 statements.


def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 3
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 0
    var_1 = 4
    var_2 = 'id'
    var_3 = 2
    var_4 = 0
    var_5 = bool(var_0 >= 4)
    assert var_5 is True



# Parsed testcases at query #32
#--------------------------






# Parsed testcases at query #33
#--------------------------

# Partially parsed test_basefield_initialization_with_defaults. Retrieved 4/7 statements.
# Failed to parse test_basefield_initialization_with_custom_locale.
# Partially parsed test_basefield_initialization_with_locale_and_seed. Retrieved 1/3 statements.
# Partially parsed test_basefield_initialization_generic_instance. Retrieved 2/3 statements.



def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0.seed
    var_2 = var_0._generic.locale
    var_3 = var_0._cache
    var_4 = var_0._cache
    var_5 = bool(var_0._cache == {})
    assert var_5 is True
    var_6 = var_0._handlers
    var_7 = var_0._handlers
    var_8 = bool(var_0._handlers == {})
    assert var_8 is True
    var_9 = var_0.aliases
    var_10 = var_0.aliases
    var_11 = bool(var_0.aliases == {})
    assert var_11 is True


def test_case_0():
    var_0 = 12345
    var_1 = module_0.BaseField(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 12345

def test_case_0():
    var_0 = 98765


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0._generic


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0.aliases
    var_2 = bool(var_0.aliases == {})
    assert var_2 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0._handlers
    var_2 = bool(var_0._handlers == {})
    assert var_2 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0._cache
    var_2 = bool(var_0._cache == {})
    assert var_2 is True



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_basefield_initialization_with_default_locale_and_seed. Retrieved 4/7 statements.
# Failed to parse test_basefield_initialization_with_custom_locale.
# Partially parsed test_basefield_initialization_generic_instance. Retrieved 2/3 statements.
# Failed to parse test_basefield_initialization_generic_locale_matches.



def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0.seed
    var_2 = var_0._generic.locale
    var_3 = var_0._cache
    var_4 = var_0._cache
    var_5 = bool(var_0._cache == {})
    assert var_5 is True
    var_6 = var_0._handlers
    var_7 = var_0._handlers
    var_8 = bool(var_0._handlers == {})
    assert var_8 is True
    var_9 = var_0.aliases
    var_10 = var_0.aliases
    var_11 = bool(var_0.aliases == {})
    assert var_11 is True


def test_case_0():
    var_0 = 12345
    var_1 = module_0.BaseField(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 12345


def test_case_0():
    var_0 = None
    var_1 = module_0.BaseField(seed=var_0)
    var_2 = var_1.seed
    assert var_2 is None


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0.aliases
    var_2 = bool(var_0.aliases == {})
    assert var_2 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0._cache
    var_2 = bool(var_0._cache == {})
    assert var_2 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0._handlers
    var_2 = bool(var_0._handlers == {})
    assert var_2 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0._generic


def test_case_0():
    var_0 = 999
    var_1 = module_0.BaseField(seed=var_0)
    var_2 = var_1._generic.seed
    assert var_2 == 999



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_basefield_initialization_with_defaults. Retrieved 4/7 statements.
# Failed to parse test_basefield_initialization_with_custom_locale.
# Partially parsed test_basefield_initialization_with_locale_and_seed. Retrieved 1/3 statements.
# Partially parsed test_basefield_initialization_generic_instance. Retrieved 2/3 statements.
# Partially parsed test_basefield_initialization_random_instance_accessible. Retrieved 2/3 statements.



def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0.seed
    var_2 = var_0._generic.locale
    var_3 = var_0._cache
    var_4 = var_0._cache
    var_5 = bool(var_0._cache == {})
    assert var_5 is True
    var_6 = var_0._handlers
    var_7 = var_0._handlers
    var_8 = bool(var_0._handlers == {})
    assert var_8 is True
    var_9 = var_0.aliases
    var_10 = var_0.aliases
    var_11 = bool(var_0.aliases == {})
    assert var_11 is True


def test_case_0():
    var_0 = 12345
    var_1 = module_0.BaseField(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 12345

def test_case_0():
    var_0 = 98765


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0.aliases
    var_2 = bool(var_0.aliases == {})
    assert var_2 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0._cache
    var_2 = bool(var_0._cache == {})
    assert var_2 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0._handlers
    var_2 = bool(var_0._handlers == {})
    assert var_2 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0._generic


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0.get_random_instance()


def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseField(seed=var_0)
    var_2 = var_1._generic.seed
    assert var_2 == 42



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_to_csv_writes_correct_data. Retrieved 8/19 statements.
# Partially parsed test_to_csv_with_custom_csv_writer_options. Retrieved 12/24 statements.
# Partially parsed test_to_csv_with_empty_schema. Retrieved 4/15 statements.
# Partially parsed test_to_csv_with_transformed_data. Retrieved 8/18 statements.
# Partially parsed test_to_csv_with_custom_context_does_not_affect_output. Retrieved 8/19 statements.



def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = lambda : var_4
    var_6 = 3
    var_7 = module_0.Schema(var_5, var_6)


def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 10
    var_3 = 20
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = lambda : var_4
    var_6 = 2
    var_7 = module_0.Schema(var_5, var_6)
    var_8 = ';'
    var_9 = '"x";"y"'
    var_10 = '"10";"20"'
    var_11 = [var_9, var_10, var_10]


def test_case_0():
    var_0 = {}
    var_1 = lambda : var_0
    var_2 = 1
    var_3 = module_0.Schema(var_1, var_2)


def test_case_0():
    var_0 = 'value'
    var_1 = 5
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 2
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = lambda item: {var_0: item[var_0] * var_4}
    var_7 = var_5.map(var_6)


def test_case_0():
    var_0 = 'id'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 2
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = 'data'
    var_7 = 'extra'
    var_8 = {var_7: var_6}
    var_9 = var_5.with_context(**var_8)



# Parsed testcases at query #4
#--------------------------






# Parsed testcases at query #5
#--------------------------





def test_case_0():
    var_0 = 'id'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 5
    var_5 = 42
    var_6 = module_0.Schema(var_3, var_4, var_5)
    var_7 = var_6.iterations
    assert var_7 == 5
    var_8 = var_6._custom_context
    var_9 = bool(var_6._custom_context == {})
    assert var_9 is True
    var_10 = var_6._transformers
    var_11 = bool(var_6._transformers == [])
    assert var_11 is True


def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = module_0.Schema(var_3)
    var_5 = var_4.iterations
    assert var_5 == 10


def test_case_0():
    var_0 = 'data'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 0
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True


def test_case_0():
    var_0 = 'not a function'
    var_1 = module_0.Schema(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = bool(True)
    assert var_3 is True


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 123
    var_5 = module_0.Schema(var_3, seed=var_4)
    var_6 = var_5.iterations
    assert var_6 == 10



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Failed to parse test_basefield_initialization_with_custom_locale.
# Partially parsed test_basefield_initialization_aliases_is_empty_dict. Retrieved 4/5 statements.
# Partially parsed test_basefield_initialization_handlers_is_empty_dict. Retrieved 4/5 statements.
# Partially parsed test_basefield_initialization_cache_is_empty_dict. Retrieved 4/5 statements.
# Partially parsed test_basefield_initialization_generic_is_instance_of_generic. Retrieved 2/3 statements.
# Failed to parse test_basefield_initialization_generic_has_same_locale.



def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0.seed
    var_2 = var_0._generic.locale
    var_3 = var_0._cache
    var_4 = bool(var_0._cache == {})
    assert var_4 is True
    var_5 = var_0._handlers
    var_6 = bool(var_0._handlers == {})
    assert var_6 is True
    var_7 = var_0.aliases
    var_8 = bool(var_0.aliases == {})
    assert var_8 is True


def test_case_0():
    var_0 = 12345
    var_1 = module_0.BaseField(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 12345


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0.aliases
    var_2 = var_0.aliases
    var_3 = len(var_2)
    assert var_3 == 0


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0._handlers
    var_2 = var_0._handlers
    var_3 = len(var_2)
    assert var_3 == 0


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0._cache
    var_2 = var_0._cache
    var_3 = len(var_2)
    assert var_3 == 0


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0._generic


def test_case_0():
    var_0 = 999
    var_1 = module_0.BaseField(seed=var_0)
    var_2 = var_1._generic.seed
    assert var_2 == 999



# Parsed testcases at query #2
#--------------------------






# Parsed testcases at query #3
#--------------------------

# Partially parsed test_pick_from_raises_value_error_when_builder_is_none. Retrieved 7/8 statements.
# Partially parsed test_pick_from_calls_builder_pick_from_with_correct_arguments. Retrieved 3/8 statements.
# Partially parsed test_pick_from_calls_builder_pick_from_without_field. Retrieved 5/10 statements.



def test_case_0():
    var_0 = 0
    var_1 = None
    var_2 = module_0.SchemaContext(var_0, builder=var_1)
    var_3 = None
    var_4 = 'some_schema'
    var_5 = var_2.pick_from(var_4)
    var_6 = bool(var_3 is not None)
    assert var_6 is True
    var_7 = str(var_3)
    assert var_7 == 'pick_from() requires SchemaBuilder'

def test_case_0():
    var_0 = 0
    var_1 = 'test_schema'
    var_2 = 'field_name'

def test_case_0():
    var_0 = 'id'
    var_1 = 1
    var_2 = 0
    var_3 = 'test_schema'
    var_4 = None



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_basefield_initializes_with_default_locale_and_seed. Retrieved 4/7 statements.
# Failed to parse test_basefield_initializes_with_custom_locale.
# Partially parsed test_basefield_generic_is_instance_of_generic. Retrieved 2/3 statements.



def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0.seed
    var_2 = var_0._generic.locale
    var_3 = var_0._cache
    var_4 = var_0._handlers
    var_5 = var_0.aliases


def test_case_0():
    var_0 = 12345
    var_1 = module_0.BaseField(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 12345


def test_case_0():
    var_0 = None
    var_1 = module_0.BaseField(seed=var_0)
    var_2 = var_1.seed
    assert var_2 is None


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0.aliases
    var_2 = bool(var_0.aliases == {})
    assert var_2 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0._cache
    var_2 = bool(var_0._cache == {})
    assert var_2 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0._handlers
    var_2 = bool(var_0._handlers == {})
    assert var_2 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0._generic



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_basefield_initialization_with_default_locale_and_seed. Retrieved 4/7 statements.
# Failed to parse test_basefield_initialization_with_custom_locale.
# Partially parsed test_basefield_initialization_generic_instance_created. Retrieved 2/3 statements.
# Failed to parse test_basefield_initialization_generic_has_same_locale.



def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0.seed
    var_2 = var_0._generic.locale
    var_3 = var_0._cache
    var_4 = var_0._cache
    var_5 = bool(var_0._cache == {})
    assert var_5 is True
    var_6 = var_0._handlers
    var_7 = var_0._handlers
    var_8 = bool(var_0._handlers == {})
    assert var_8 is True
    var_9 = var_0.aliases
    var_10 = var_0.aliases
    var_11 = bool(var_0.aliases == {})
    assert var_11 is True


def test_case_0():
    var_0 = 12345
    var_1 = module_0.BaseField(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 12345


def test_case_0():
    var_0 = None
    var_1 = module_0.BaseField(seed=var_0)
    var_2 = var_1.seed
    assert var_2 is None


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0.aliases
    var_2 = bool(var_0.aliases == {})
    assert var_2 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0._cache
    var_2 = bool(var_0._cache == {})
    assert var_2 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0._handlers
    var_2 = bool(var_0._handlers == {})
    assert var_2 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0._generic


def test_case_0():
    var_0 = 999
    var_1 = module_0.BaseField(seed=var_0)
    var_2 = var_1._generic.seed
    assert var_2 == 999



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_create_generates_data_for_specified_schemas. Retrieved 10/14 statements.
# Partially parsed test_create_raises_error_for_undefined_schema. Retrieved 5/8 statements.
# Partially parsed test_create_stores_data_internal_data_attribute. Retrieved 6/8 statements.
# Partially parsed test_create_resets_schema_iterations_after_generation. Retrieved 4/7 statements.
# Partially parsed test_create_resets_schema_transformers_after_generation. Retrieved 4/7 statements.
# Partially parsed test_create_with_seed_produces_deterministic_data. Retrieved 7/11 statements.



def test_case_0():
    var_0 = module_0.SchemaBuilder()
    var_1 = 'users'
    var_2 = 'posts'
    var_3 = 5
    var_4 = 3
    var_5 = 'users'
    var_6 = 'posts'
    var_7 = {var_5: var_3, var_6: var_4}
    var_8 = var_0.create(**var_7)
    var_9 = 'users'
    var_10 = bool('users' in var_8)
    assert var_10 is True
    var_11 = 'posts'
    var_12 = bool('posts' in var_8)
    assert var_12 is True
    var_13 = var_8[var_1]
    var_14 = len(var_13)
    assert var_14 == 5
    var_15 = var_8[var_2]
    var_16 = len(var_15)
    assert var_16 == 3


def test_case_0():
    var_0 = module_0.SchemaBuilder()
    var_1 = 'users'
    var_2 = 2
    var_3 = 1
    var_4 = 'users'
    var_5 = 'posts'
    var_6 = {var_4: var_2, var_5: var_3}
    var_7 = var_0.create(**var_6)
    var_8 = bool(False)
    assert var_8 is True


def test_case_0():
    var_0 = module_0.SchemaBuilder()
    var_1 = {}
    var_2 = var_0.create(**var_1)
    var_3 = bool(var_2 == {})
    assert var_3 is True


def test_case_0():
    var_0 = module_0.SchemaBuilder()
    var_1 = 'items'
    var_2 = 4
    var_3 = 'items'
    var_4 = {var_3: var_2}
    var_5 = var_0.create(**var_4)
    var_6 = 'items'
    var_7 = bool('items' in var_0._data)
    assert var_7 is True
    var_8 = var_0._data[var_1]
    var_9 = len(var_8)
    assert var_9 == 4


def test_case_0():
    var_0 = module_0.SchemaBuilder()
    var_1 = 'things'
    var_2 = 2
    var_3 = 'things'
    var_4 = {var_3: var_2}
    var_5 = var_0.create(**var_4)


def test_case_0():
    var_0 = module_0.SchemaBuilder()
    var_1 = 'objects'
    var_2 = 1
    var_3 = 'objects'
    var_4 = {var_3: var_2}
    var_5 = var_0.create(**var_4)


def test_case_0():
    var_0 = 42
    var_1 = module_0.SchemaBuilder(var_0)
    var_2 = 'data'
    var_3 = 3
    var_4 = 'data'
    var_5 = {var_4: var_3}
    var_6 = var_1.create(**var_5)
    var_7 = module_0.SchemaBuilder(var_0)
    var_8 = 'data'
    var_9 = {var_8: var_3}
    var_10 = var_7.create(**var_9)
    var_11 = bool(var_6 == var_10)
    assert var_11 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_fieldset_call_with_default_iterations. Retrieved 1/6 statements.
# Partially parsed test_fieldset_call_with_specified_iterations. Retrieved 2/7 statements.
# Partially parsed test_fieldset_call_with_iterations_from_init. Retrieved 2/7 statements.
# Partially parsed test_fieldset_call_with_override_iterations. Retrieved 3/8 statements.
# Partially parsed test_fieldset_call_with_zero_iterations_raises_error. Retrieved 2/5 statements.
# Partially parsed test_fieldset_call_with_negative_iterations_raises_error. Retrieved 2/5 statements.
# Partially parsed test_fieldset_call_passes_arguments_to_perform. Retrieved 2/6 statements.
# Partially parsed test_fieldset_call_with_custom_iterations_kwarg. Retrieved 3/9 statements.
# Partially parsed test_fieldset_call_with_custom_default_iterations. Retrieved 2/8 statements.


def test_case_0():
    var_0 = []
    var_1 = 'username'

def test_case_0():
    var_0 = []
    var_1 = 'username'
    var_2 = 5

def test_case_0():
    var_0 = 3
    var_1 = []
    var_2 = 'username'

def test_case_0():
    var_0 = 7
    var_1 = []
    var_2 = 'username'
    var_3 = 2

def test_case_0():
    var_0 = []
    var_1 = 'username'
    var_2 = 0
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = []
    var_1 = 'username'
    var_2 = -5
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'test_arg'
    var_1 = 'kwarg'

def test_case_0():
    var_0 = 'count'
    var_1 = 'username'
    var_2 = 4

def test_case_0():
    var_0 = 6
    var_1 = 'username'



# Parsed testcases at query #4
#--------------------------






# Parsed testcases at query #5
#--------------------------

# Partially parsed test___iter___enables_iteration. Retrieved 10/12 statements.
# Partially parsed test___iter___works_with_custom_context. Retrieved 12/14 statements.
# Partially parsed test___iter___handles_none_results. Retrieved 2/10 statements.
# Partially parsed test___iter___with_seed. Retrieved 2/10 statements.



def test_case_0():
    var_0 = 'id'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 5
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = var_5.__iter__()
    var_7 = bool(var_6 is var_5)
    assert var_7 is True


def test_case_0():
    var_0 = 'id'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 3
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = list(var_5)
    var_7 = var_5.__iter__()
    var_8 = var_5._Schema__counter
    assert var_8 == 0


def test_case_0():
    var_0 = 'id'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 3
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = var_5.__iter__()
    var_7 = list(var_5)
    var_8 = len(var_7)
    assert var_8 == 3
    var_9 = {var_0: var_1}


def test_case_0():
    var_0 = 'value'
    var_1 = 0
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 2
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = 1
    var_7 = lambda x: {var_0: x[var_0] + var_6}
    var_8 = var_5.map(var_7)
    var_9 = var_5.__iter__()
    var_10 = list(var_5)
    var_11 = bool(var_10 == [{'value': 1}, {'value': 1}])
    assert var_11 is True


def test_case_0():
    var_0 = 'index'
    var_1 = 0
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 2
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = 'data'
    var_7 = 'extra'
    var_8 = {var_7: var_6}
    var_9 = var_5.with_context(**var_8)
    var_10 = var_5.__iter__()
    var_11 = list(var_5)
    var_12 = len(var_11)
    assert var_12 == 2
    var_13 = {var_0: var_1}

def test_case_0():
    var_0 = 0
    var_1 = 2


def test_case_0():
    var_0 = 'data'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 2
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = var_5.__iter__()
    var_7 = list(var_6)
    var_8 = var_5.__iter__()
    var_9 = list(var_8)
    var_10 = bool(var_7 == var_9)
    assert var_10 is True
    var_11 = len(var_7)
    assert var_11 == 2

def test_case_0():
    var_0 = 3
    var_1 = 42


def test_case_0():
    var_0 = {}
    var_1 = lambda : var_0
    var_2 = 0
    var_3 = module_0.Schema(var_1, var_2)
    var_4 = var_3.__iter__()
    var_5 = list(var_3)
    var_6 = []
    var_7 = bool(var_6 == [])
    assert var_7 is True


def test_case_0():
    var_0 = 'counter'
    var_1 = 0
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 3
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = var_5.__iter__()
    var_7 = next(var_6)
    var_8 = next(var_6)
    var_9 = next(var_6)
    var_10 = bool(var_7 == {'counter': 0})
    assert var_10 is True
    var_11 = bool(var_8 == {'counter': 0})
    assert var_11 is True
    var_12 = bool(var_9 == {'counter': 0})
    assert var_12 is True



# Parsed testcases at query #6
#--------------------------






# Parsed testcases at query #7
#--------------------------

# Partially parsed test_handle_returns_decorated_function. Retrieved 3/6 statements.



def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'custom_handler'
    var_2 = bool('custom_handler' in var_0._handlers)
    assert var_2 is True
    var_3 = 'custom_handler'
    var_4 = {}
    var_5 = var_0.perform(var_3, **var_4)
    assert var_5 == 'custom'


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'my_field'
    var_2 = bool('my_field' in var_0._handlers)
    assert var_2 is True
    var_3 = 'my_field'
    var_4 = {}
    var_5 = var_0.perform(var_3, **var_4)
    assert var_5 == 'custom'


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'test_field'
    var_2 = var_0.handle(var_1)
    var_3 = 'test_field'
    var_4 = bool('test_field' in var_0._handlers)
    assert var_4 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = bool(False)
    assert var_1 is True
    var_2 = bool(True)
    assert var_2 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = bool(False)
    assert var_1 is True
    var_2 = bool(True)
    assert var_2 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'test'
    var_2 = var_0.handle(var_1)
    var_3 = 'not_callable'
    var_4 = var_2(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = bool(False)
    assert var_1 is True
    var_2 = bool(True)
    assert var_2 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'my_field'
    var_2 = {}
    var_3 = var_0.perform(var_1, **var_2)
    assert var_3 == 'first'


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'some_handler'
    var_2 = bool('some_handler' in var_0._handlers)
    assert var_2 is True
    var_3 = 'some_handler'
    var_4 = {}
    var_5 = var_0.perform(var_3, **var_4)
    assert var_5 == 'result'


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'random_test'
    var_2 = {}
    var_3 = var_0.perform(var_1, **var_2)
    var_4 = 1
    var_5 = bool(1 <= var_3)
    assert var_5 is True
    var_6 = bool(var_3 <= 10)
    assert var_6 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_basefield_initialization_with_default_locale_and_seed. Retrieved 4/7 statements.
# Failed to parse test_basefield_initialization_with_custom_locale.
# Partially parsed test_basefield_initialization_generic_instance_created. Retrieved 2/3 statements.
# Failed to parse test_basefield_initialization_generic_locale_matches.
# Failed to parse test_basefield_initialization_with_missingseed_constant.
# Partially parsed test_basefield_initialization_aliases_is_dict. Retrieved 2/3 statements.
# Partially parsed test_basefield_initialization_cache_is_dict. Retrieved 2/3 statements.
# Partially parsed test_basefield_initialization_handlers_is_dict. Retrieved 2/3 statements.



def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0.seed
    var_2 = var_0._generic.locale
    var_3 = var_0._cache
    var_4 = var_0._cache
    var_5 = bool(var_0._cache == {})
    assert var_5 is True
    var_6 = var_0._handlers
    var_7 = var_0._handlers
    var_8 = bool(var_0._handlers == {})
    assert var_8 is True
    var_9 = var_0.aliases
    var_10 = var_0.aliases
    var_11 = bool(var_0.aliases == {})
    assert var_11 is True


def test_case_0():
    var_0 = 12345
    var_1 = module_0.BaseField(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 12345


def test_case_0():
    var_0 = None
    var_1 = module_0.BaseField(seed=var_0)
    var_2 = var_1.seed
    assert var_2 is None


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0.aliases
    var_2 = bool(var_0.aliases == {})
    assert var_2 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0._cache
    var_2 = bool(var_0._cache == {})
    assert var_2 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0._handlers
    var_2 = bool(var_0._handlers == {})
    assert var_2 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0._generic


def test_case_0():
    var_0 = 999
    var_1 = module_0.BaseField(seed=var_0)
    var_2 = var_1._generic.seed
    assert var_2 == 999


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0.aliases


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0._cache


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0._handlers


def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseField(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 42


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0.aliases
    var_2 = len(var_1)
    assert var_2 == 0


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0._cache
    var_2 = len(var_1)
    assert var_2 == 0


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0._handlers
    var_2 = len(var_1)
    assert var_2 == 0



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_create_returns_items_from_schema. Retrieved 8/10 statements.
# Partially parsed test_create_applies_transformers_with_context. Retrieved 7/11 statements.
# Partially parsed test_create_skips_none_results. Retrieved 2/9 statements.
# Partially parsed test_create_uses_custom_context. Retrieved 10/16 statements.



def test_case_0():
    var_0 = 'id'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 5
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = var_5.create()
    var_7 = len(var_6)
    assert var_7 == 5


def test_case_0():
    var_0 = 'value'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 3
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = var_5.create()
    var_7 = {var_0: var_1}


def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 2
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = lambda item: {var_0: item[var_0] * var_4}
    var_7 = var_5.map(var_6)
    var_8 = var_5.create()
    var_9 = bool(var_8 == [{'x': 2}, {'x': 2}])
    assert var_9 is True


def test_case_0():
    var_0 = 'data'
    var_1 = 'a'
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 3
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = var_5.create()
    var_7 = [item['index'] for item in var_6]
    var_8 = bool([item['index'] for item in var_6] == [0, 1, 2])
    assert var_8 is True

def test_case_0():
    var_0 = 0
    var_1 = 3


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 2
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = 'value'
    var_7 = 'key'
    var_8 = {var_7: var_6}
    var_9 = var_5.with_context(**var_8)
    var_10 = var_5.create()
    var_11 = 'custom'


def test_case_0():
    var_0 = {}
    var_1 = lambda : var_0
    var_2 = 0
    var_3 = module_0.Schema(var_1, var_2)
    var_4 = bool(False)
    assert var_4 is True


def test_case_0():
    var_0 = {}
    var_1 = lambda : var_0
    var_2 = -1
    var_3 = module_0.Schema(var_1, var_2)
    var_4 = bool(False)
    assert var_4 is True


def test_case_0():
    var_0 = 'not callable'
    var_1 = 1
    var_2 = module_0.Schema(var_0, var_1)
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #10
#--------------------------






# Parsed testcases at query #11
#--------------------------

# Partially parsed test_basefield_initialization_with_default_locale_and_seed. Retrieved 4/7 statements.
# Failed to parse test_basefield_initialization_with_custom_locale.
# Partially parsed test_basefield_initialization_generic_instance. Retrieved 2/3 statements.
# Failed to parse test_basefield_initialization_generic_locale_matches.



def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0.seed
    var_2 = var_0._generic.locale
    var_3 = var_0._cache
    var_4 = var_0._handlers
    var_5 = var_0.aliases
    var_6 = var_0.aliases
    var_7 = bool(var_0.aliases == {})
    assert var_7 is True


def test_case_0():
    var_0 = 12345
    var_1 = module_0.BaseField(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 12345


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0.aliases
    var_2 = bool(var_0.aliases == {})
    assert var_2 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0._cache
    var_2 = bool(var_0._cache == {})
    assert var_2 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0._handlers
    var_2 = bool(var_0._handlers == {})
    assert var_2 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0._generic


def test_case_0():
    var_0 = 999
    var_1 = module_0.BaseField(seed=var_0)
    var_2 = var_1._generic.seed
    assert var_2 == 999



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_constructor_without_seed. Retrieved 2/3 statements.
# Partially parsed test_constructor_with_seed. Retrieved 3/4 statements.



def test_case_0():
    var_0 = module_0.SchemaBuilder()
    var_1 = var_0._seed
    var_2 = var_0._random
    var_3 = var_0._schemas
    var_4 = bool(var_0._schemas == {})
    assert var_4 is True
    var_5 = var_0._data
    var_6 = bool(var_0._data == {})
    assert var_6 is True


def test_case_0():
    var_0 = 42
    var_1 = module_0.SchemaBuilder(var_0)
    var_2 = var_1._seed
    assert var_2 == 42
    var_3 = var_1._random
    var_4 = var_1._random.seed
    assert var_4 == 42
    var_5 = var_1._schemas
    var_6 = bool(var_1._schemas == {})
    assert var_6 is True
    var_7 = var_1._data
    var_8 = bool(var_1._data == {})
    assert var_8 is True



# Parsed testcases at query #13
#--------------------------






# Parsed testcases at query #14
#--------------------------

# Partially parsed test_to_csv_writes_correct_data. Retrieved 8/20 statements.
# Partially parsed test_to_csv_with_custom_csv_writer_args. Retrieved 12/24 statements.
# Partially parsed test_to_csv_applies_transformers. Retrieved 8/18 statements.
# Partially parsed test_to_csv_handles_empty_schema. Retrieved 4/14 statements.
# Partially parsed test_to_csv_uses_correct_encoding. Retrieved 6/16 statements.



def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = lambda : var_4
    var_6 = 3
    var_7 = module_0.Schema(var_5, var_6)


def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 10
    var_3 = 20
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = lambda : var_4
    var_6 = 2
    var_7 = module_0.Schema(var_5, var_6)
    var_8 = ';'
    var_9 = '"x";"y"'
    var_10 = '"10";"20"'
    var_11 = [var_9, var_10, var_10]


def test_case_0():
    var_0 = 'id'
    var_1 = 0
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 2
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = lambda item, ctx: {var_0: ctx.index}
    var_7 = var_5.map(var_6)


def test_case_0():
    var_0 = {}
    var_1 = lambda : var_0
    var_2 = 1
    var_3 = module_0.Schema(var_1, var_2)


def test_case_0():
    var_0 = 'text'
    var_1 = 'café'
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 1
    var_5 = module_0.Schema(var_3, var_4)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test___next___returns_items_until_iterations. Retrieved 10/14 statements.
# Partially parsed test___next___skips_none_results. Retrieved 3/11 statements.
# Partially parsed test___next___with_transformer_applies_transformation. Retrieved 6/13 statements.
# Partially parsed test___next___with_custom_context. Retrieved 6/11 statements.
# Partially parsed test___next___uses_index_in_context. Retrieved 2/10 statements.
# Partially parsed test___next___with_seed. Retrieved 2/13 statements.



def test_case_0():
    var_0 = 'id'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 3
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = []
    var_7 = next(var_5)
    var_8 = len(var_6)
    assert var_8 == 3
    var_9 = {var_7: var_1}


def test_case_0():
    var_0 = 'id'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = module_0.Schema(var_3, var_1)
    var_5 = next(var_4)
    var_6 = bool(var_5 == {'id': 1})
    assert var_6 is True
    var_7 = next(var_4)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = bool(True)
    assert var_9 is True

def test_case_0():
    var_0 = 0
    var_1 = 2
    var_2 = []
    var_3 = bool(var_2 == [{'id': 1}, {'id': 3}])
    assert var_3 is True


def test_case_0():
    var_0 = 'value'
    var_1 = 5
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 2
    var_5 = module_0.Schema(var_3, var_4)


def test_case_0():
    var_0 = {}
    var_1 = lambda : var_0
    var_2 = 1
    var_3 = module_0.Schema(var_1, var_2)
    var_4 = 'value'
    var_5 = 'key'
    var_6 = {var_5: var_4}
    var_7 = var_3.with_context(**var_6)


def test_case_0():
    var_0 = 'id'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 2
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = list(var_5)
    var_7 = list(var_5)
    var_8 = bool(var_6 == var_7)
    assert var_8 is True
    var_9 = len(var_6)
    assert var_9 == 2

def test_case_0():
    var_0 = []
    var_1 = 3
    var_2 = bool(var_0 == [0, 1, 2])
    assert var_2 is True

def test_case_0():
    var_0 = 2
    var_1 = 42



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_basefield_initialization_with_default_locale_and_seed. Retrieved 4/7 statements.
# Failed to parse test_basefield_initialization_with_custom_locale.
# Partially parsed test_basefield_initialization_generic_instance_created. Retrieved 2/3 statements.
# Failed to parse test_basefield_initialization_generic_has_same_locale.



def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0.seed
    var_2 = var_0._generic.locale
    var_3 = var_0._cache
    var_4 = var_0._cache
    var_5 = bool(var_0._cache == {})
    assert var_5 is True
    var_6 = var_0._handlers
    var_7 = var_0._handlers
    var_8 = bool(var_0._handlers == {})
    assert var_8 is True
    var_9 = var_0.aliases
    var_10 = var_0.aliases
    var_11 = bool(var_0.aliases == {})
    assert var_11 is True


def test_case_0():
    var_0 = 12345
    var_1 = module_0.BaseField(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 12345


def test_case_0():
    var_0 = None
    var_1 = module_0.BaseField(seed=var_0)
    var_2 = var_1.seed
    assert var_2 is None


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0.aliases
    var_2 = bool(var_0.aliases == {})
    assert var_2 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0._cache
    var_2 = bool(var_0._cache == {})
    assert var_2 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0._handlers
    var_2 = bool(var_0._handlers == {})
    assert var_2 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0._generic


def test_case_0():
    var_0 = 999
    var_1 = module_0.BaseField(seed=var_0)
    var_2 = var_1._generic.seed
    assert var_2 == 999



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_perform_with_valid_explicit_field. Retrieved 3/4 statements.
# Partially parsed test_perform_with_valid_fuzzy_field. Retrieved 3/4 statements.
# Partially parsed test_perform_with_aliases. Retrieved 4/6 statements.
# Partially parsed test_perform_with_key_function. Retrieved 4/5 statements.
# Partially parsed test_perform_with_key_function_using_random. Retrieved 2/6 statements.
# Partially parsed test_perform_with_custom_handler. Retrieved 3/6 statements.
# Partially parsed test_perform_with_different_delimiters. Retrieved 9/13 statements.
# Partially parsed test_perform_with_kwargs. Retrieved 4/5 statements.
# Partially parsed test_perform_with_aliases_type_error. Retrieved 5/7 statements.



def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'person.full_name'
    var_2 = {}
    var_3 = var_0.perform(var_1, **var_2)


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'full_name'
    var_2 = {}
    var_3 = var_0.perform(var_1, **var_2)


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'alias_name'
    var_2 = 'person.full_name'
    var_3 = {}
    var_4 = var_0.perform(var_1, **var_3)


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'person.full_name'
    var_2 = lambda x: x.upper()
    var_3 = {}
    var_4 = var_0.perform(var_1, var_2, **var_3)


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'person.full_name'


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'custom_field'
    var_2 = {}
    var_3 = var_0.perform(var_1, **var_2)
    assert var_3 == 'custom'


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'person.full_name'
    var_2 = {}
    var_3 = var_0.perform(var_1, **var_2)
    var_4 = 'person:full_name'
    var_5 = {}
    var_6 = var_0.perform(var_4, **var_5)
    var_7 = 'person/full_name'
    var_8 = {}
    var_9 = var_0.perform(var_7, **var_8)
    var_10 = 'person full_name'
    var_11 = {}
    var_12 = var_0.perform(var_10, **var_11)


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'invalid.provider'
    var_2 = {}
    var_3 = var_0.perform(var_1, **var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = None
    var_2 = {}
    var_3 = var_0.perform(var_1, **var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'person.first_name'
    var_2 = 'F'
    var_3 = 'sex'
    var_4 = {var_3: var_2}
    var_5 = var_0.perform(var_1, **var_4)


def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseField(seed=var_0)
    var_2 = 'person.full_name'
    var_3 = {}
    var_4 = var_1.perform(var_2, **var_3)
    var_5 = var_1.reseed(var_0)
    var_6 = {}
    var_7 = var_1.perform(var_2, **var_6)
    var_8 = bool(var_4 == var_7)
    assert var_8 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'invalid'
    var_2 = 123
    var_3 = 'person.full_name'
    var_4 = {}
    var_5 = var_0.perform(var_3, **var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = bool(True)
    assert var_7 is True



# Parsed testcases at query #18
#--------------------------





def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'handler1'
    var_2 = lambda random, **kwargs: var_1
    var_3 = 'handler2'
    var_4 = lambda random, **kwargs: var_3
    var_5 = 'custom_field1'
    var_6 = (var_5, var_2)
    var_7 = 'custom_field2'
    var_8 = (var_7, var_4)
    var_9 = [var_6, var_8]
    var_10 = var_0.register_handlers(var_9)
    var_11 = {}
    var_12 = var_0.perform(var_5, **var_11)
    assert var_12 == 'handler1'
    var_13 = {}
    var_14 = var_0.perform(var_7, **var_13)
    assert var_14 == 'handler2'


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'handler1'
    var_2 = lambda random, **kwargs: var_1
    var_3 = 'handler2'
    var_4 = lambda random, **kwargs: var_3
    var_5 = 'custom_field'
    var_6 = (var_5, var_2)
    var_7 = (var_5, var_4)
    var_8 = [var_6, var_7]
    var_9 = var_0.register_handlers(var_8)
    var_10 = {}
    var_11 = var_0.perform(var_5, **var_10)
    assert var_11 == 'handler1'


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = []
    var_2 = var_0.register_handlers(var_1)
    var_3 = var_0._handlers
    var_4 = bool(var_0._handlers == {})
    assert var_4 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'handler1'
    var_2 = lambda random, **kwargs: var_1
    var_3 = 'handler2'
    var_4 = lambda random, **kwargs: var_3
    var_5 = 'custom_field'
    var_6 = var_0.register_handler(var_5, var_2)
    var_7 = (var_5, var_4)
    var_8 = [var_7]
    var_9 = var_0.register_handlers(var_8)
    var_10 = {}
    var_11 = var_0.perform(var_5, **var_10)
    assert var_11 == 'handler2'


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'handler1'
    var_2 = lambda random, **kwargs: var_1
    var_3 = 'handler2'
    var_4 = lambda random, **kwargs: var_3
    var_5 = 'handler3'
    var_6 = lambda random, **kwargs: var_5
    var_7 = 'field1'
    var_8 = (var_7, var_2)
    var_9 = 'field2'
    var_10 = (var_9, var_4)
    var_11 = 'field3'
    var_12 = (var_11, var_6)
    var_13 = [var_8, var_10, var_12]
    var_14 = var_0.register_handlers(var_13)
    var_15 = {}
    var_16 = var_0.perform(var_7, **var_15)
    assert var_16 == 'handler1'
    var_17 = {}
    var_18 = var_0.perform(var_9, **var_17)
    assert var_18 == 'handler2'
    var_19 = {}
    var_20 = var_0.perform(var_11, **var_19)
    assert var_20 == 'handler3'


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'handler'
    var_2 = lambda random, **kwargs: var_1
    var_3 = 123
    var_4 = (var_3, var_2)
    var_5 = [var_4]
    var_6 = var_0.register_handlers(var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = bool(True)
    assert var_8 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'handler'
    var_2 = lambda random, **kwargs: var_1
    var_3 = '123invalid'
    var_4 = (var_3, var_2)
    var_5 = [var_4]
    var_6 = var_0.register_handlers(var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = bool(True)
    assert var_8 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'custom_field'
    var_2 = 'not_callable'
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = var_0.register_handlers(var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = bool(True)
    assert var_7 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'handler'
    var_2 = lambda random: var_1
    var_3 = 'custom_field'
    var_4 = (var_3, var_2)
    var_5 = [var_4]
    var_6 = var_0.register_handlers(var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = bool(True)
    assert var_8 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_basefield_initialization_with_default_locale_and_seed. Retrieved 4/7 statements.
# Failed to parse test_basefield_initialization_with_custom_locale.
# Partially parsed test_basefield_initialization_generic_instance_created. Retrieved 2/3 statements.
# Partially parsed test_basefield_initialization_with_locale_and_seed. Retrieved 1/3 statements.



def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0.seed
    var_2 = var_0._generic.locale
    var_3 = var_0._cache
    var_4 = var_0._cache
    var_5 = bool(var_0._cache == {})
    assert var_5 is True
    var_6 = var_0._handlers
    var_7 = var_0._handlers
    var_8 = bool(var_0._handlers == {})
    assert var_8 is True
    var_9 = var_0.aliases
    var_10 = var_0.aliases
    var_11 = bool(var_0.aliases == {})
    assert var_11 is True


def test_case_0():
    var_0 = 12345
    var_1 = module_0.BaseField(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 12345


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0.aliases
    var_2 = bool(var_0.aliases == {})
    assert var_2 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0._cache
    var_2 = bool(var_0._cache == {})
    assert var_2 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0._handlers
    var_2 = bool(var_0._handlers == {})
    assert var_2 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0._generic

def test_case_0():
    var_0 = 999



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_create_returns_list_of_correct_length. Retrieved 10/11 statements.
# Partially parsed test_create_returns_list_of_dicts. Retrieved 9/11 statements.
# Partially parsed test_create_applies_transformers_with_context. Retrieved 7/12 statements.
# Partially parsed test_create_skips_none_results. Retrieved 2/9 statements.
# Partially parsed test_create_uses_custom_context. Retrieved 9/14 statements.
# Partially parsed test_create_with_seed. Retrieved 2/9 statements.



def test_case_0():
    var_0 = 'id'
    var_1 = 'name'
    var_2 = 1
    var_3 = 'test'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = lambda : var_4
    var_6 = 5
    var_7 = module_0.Schema(var_5, var_6)
    var_8 = var_7.create()
    var_9 = len(var_8)
    assert var_9 == 5


def test_case_0():
    var_0 = 'id'
    var_1 = 'name'
    var_2 = 1
    var_3 = 'test'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = lambda : var_4
    var_6 = 3
    var_7 = module_0.Schema(var_5, var_6)
    var_8 = var_7.create()
    var_9 = bool(var_0)
    assert var_9 is True


def test_case_0():
    var_0 = 'value'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 2
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = lambda x: {var_0: x[var_0] * var_4}
    var_7 = var_5.map(var_6)
    var_8 = var_5.create()


def test_case_0():
    var_0 = 'value'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 3
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = var_5.create()

def test_case_0():
    var_0 = 0
    var_1 = 3


def test_case_0():
    var_0 = 'data'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 2
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = '1.0'
    var_7 = 'version'
    var_8 = {var_7: var_6}
    var_9 = var_5.with_context(**var_8)
    var_10 = var_5.create()

def test_case_0():
    var_0 = 2
    var_1 = 42


def test_case_0():
    var_0 = {}
    var_1 = lambda : var_0
    var_2 = 0
    var_3 = module_0.Schema(var_1, var_2)
    var_4 = bool(False)
    assert var_4 is True


def test_case_0():
    var_0 = 'not a callable'
    var_1 = module_0.Schema(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = bool(True)
    assert var_3 is True


def test_case_0():
    var_0 = {}
    var_1 = lambda : var_0
    var_2 = 4
    var_3 = module_0.Schema(var_1, var_2)
    var_4 = var_3.create()
    var_5 = len(var_4)
    assert var_5 == 4



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_fieldset_call_with_default_iterations. Retrieved 1/5 statements.
# Partially parsed test_fieldset_call_with_specified_iterations. Retrieved 2/6 statements.
# Partially parsed test_fieldset_call_with_iterations_from_init. Retrieved 2/6 statements.
# Partially parsed test_fieldset_call_with_iterations_override. Retrieved 3/7 statements.
# Partially parsed test_fieldset_call_with_zero_iterations_raises_error. Retrieved 2/5 statements.
# Partially parsed test_fieldset_call_with_negative_iterations_raises_error. Retrieved 2/5 statements.
# Partially parsed test_fieldset_call_passes_arguments_to_perform. Retrieved 8/17 statements.


def test_case_0():
    var_0 = []
    var_1 = 'username'

def test_case_0():
    var_0 = []
    var_1 = 'username'
    var_2 = 5

def test_case_0():
    var_0 = 3
    var_1 = []
    var_2 = 'username'

def test_case_0():
    var_0 = 7
    var_1 = []
    var_2 = 'username'
    var_3 = 2

def test_case_0():
    var_0 = []
    var_1 = 'username'
    var_2 = 0
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = []
    var_1 = 'username'
    var_2 = -5
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = False
    var_1 = None
    var_2 = None
    var_3 = []
    var_4 = 'arg1'
    var_5 = 'arg2'
    var_6 = 'val1'
    var_7 = 'val2'
    var_8 = 3
    var_9 = bool(var_0)
    assert var_9 is True
    var_10 = bool(var_1 == ('arg1', 'arg2'))
    assert var_10 is True
    var_11 = bool(var_2 == {'key1': 'val1', 'key2': 'val2'})
    assert var_11 is True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_random_initialized_without_seed. Retrieved 2/3 statements.



def test_case_0():
    var_0 = module_0.SchemaBuilder()
    var_1 = var_0._random



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_constructor_without_seed. Retrieved 2/3 statements.
# Partially parsed test_constructor_with_seed. Retrieved 3/4 statements.
# Partially parsed test_constructor_with_seed_zero. Retrieved 3/4 statements.
# Partially parsed test_constructor_with_negative_seed. Retrieved 3/4 statements.
# Partially parsed test_constructor_with_large_seed. Retrieved 3/4 statements.



def test_case_0():
    var_0 = module_0.SchemaBuilder()
    var_1 = var_0._seed
    var_2 = var_0._random
    var_3 = var_0._schemas
    var_4 = bool(var_0._schemas == {})
    assert var_4 is True
    var_5 = var_0._data
    var_6 = bool(var_0._data == {})
    assert var_6 is True


def test_case_0():
    var_0 = 12345
    var_1 = module_0.SchemaBuilder(var_0)
    var_2 = var_1._seed
    assert var_2 == 12345
    var_3 = var_1._random
    var_4 = var_1._schemas
    var_5 = bool(var_1._schemas == {})
    assert var_5 is True
    var_6 = var_1._data
    var_7 = bool(var_1._data == {})
    assert var_7 is True


def test_case_0():
    var_0 = 0
    var_1 = module_0.SchemaBuilder(var_0)
    var_2 = var_1._seed
    assert var_2 == 0
    var_3 = var_1._random
    var_4 = var_1._schemas
    var_5 = bool(var_1._schemas == {})
    assert var_5 is True
    var_6 = var_1._data
    var_7 = bool(var_1._data == {})
    assert var_7 is True


def test_case_0():
    var_0 = -999
    var_1 = module_0.SchemaBuilder(var_0)
    var_2 = var_1._seed
    assert var_2 == -999
    var_3 = var_1._random
    var_4 = var_1._schemas
    var_5 = bool(var_1._schemas == {})
    assert var_5 is True
    var_6 = var_1._data
    var_7 = bool(var_1._data == {})
    assert var_7 is True


def test_case_0():
    var_0 = 999999
    var_1 = module_0.SchemaBuilder(var_0)
    var_2 = var_1._seed
    assert var_2 == 999999
    var_3 = var_1._random
    var_4 = var_1._schemas
    var_5 = bool(var_1._schemas == {})
    assert var_5 is True
    var_6 = var_1._data
    var_7 = bool(var_1._data == {})
    assert var_7 is True



# Parsed testcases at query #24
#--------------------------






# Parsed testcases at query #25
#--------------------------

# Partially parsed test_pick_from_calls_builder_pick_from_with_correct_arguments. Retrieved 3/6 statements.
# Partially parsed test_pick_from_calls_builder_pick_from_without_field. Retrieved 2/5 statements.



def test_case_0():
    var_0 = 0
    var_1 = module_0.SchemaContext(var_0)
    var_2 = False
    var_3 = 'some_schema'
    var_4 = var_1.pick_from(var_3)
    var_5 = True
    var_6 = bool(var_5)
    assert var_6 is True

def test_case_0():
    var_0 = 0
    var_1 = 'test_schema'
    var_2 = 'test_field'

def test_case_0():
    var_0 = 0
    var_1 = 'test_schema'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_constructor_without_seed. Retrieved 2/3 statements.
# Partially parsed test_constructor_with_seed. Retrieved 3/4 statements.



def test_case_0():
    var_0 = module_0.SchemaBuilder()
    var_1 = var_0._seed
    var_2 = var_0._random
    var_3 = var_0._schemas
    var_4 = bool(var_0._schemas == {})
    assert var_4 is True
    var_5 = var_0._data
    var_6 = bool(var_0._data == {})
    assert var_6 is True


def test_case_0():
    var_0 = 42
    var_1 = module_0.SchemaBuilder(var_0)
    var_2 = var_1._seed
    assert var_2 == 42
    var_3 = var_1._random
    var_4 = var_1._schemas
    var_5 = bool(var_1._schemas == {})
    assert var_5 is True
    var_6 = var_1._data
    var_7 = bool(var_1._data == {})
    assert var_7 is True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_perform_key_function_with_one_parameter. Retrieved 6/8 statements.
# Partially parsed test_perform_key_function_with_two_parameters. Retrieved 10/12 statements.
# Partially parsed test_perform_key_is_none. Retrieved 6/8 statements.
# Partially parsed test_perform_key_is_not_callable. Retrieved 6/8 statements.



def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'test_result'
    var_2 = lambda : var_1
    var_3 = 'test'
    var_4 = lambda result: result.upper()
    var_5 = {}
    var_6 = var_0.perform(var_3, var_4, **var_5)
    assert var_6 == 'TEST_RESULT'


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'test_result'
    var_2 = lambda : var_1
    var_3 = 'test'
    var_4 = lambda result, random: result.upper() + str(random)
    var_5 = {}
    var_6 = var_0.perform(var_3, var_4, **var_5)
    var_7 = 'TEST_RESULT'
    var_8 = var_0.get_random_instance()
    var_9 = str(var_8)
    var_10 = var_7 + var_9
    var_11 = bool(var_6 == var_10)
    assert var_11 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'test_result'
    var_2 = lambda : var_1
    var_3 = 'test'
    var_4 = None
    var_5 = {}
    var_6 = var_0.perform(var_3, var_4, **var_5)
    assert var_6 == 'test_result'


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'test_result'
    var_2 = lambda : var_1
    var_3 = 'test'
    var_4 = 'not_callable'
    var_5 = {}
    var_6 = var_0.perform(var_3, var_4, **var_5)
    assert var_6 == 'test_result'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_next_raises_stop_iteration_when_counter_reaches_iterations. Retrieved 7/9 statements.



def test_case_0():
    var_0 = 'id'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 2
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = next(var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = bool(True)
    assert var_8 is True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_to_csv_with_empty_data. Retrieved 6/8 statements.



def test_case_0():
    var_0 = None
    var_1 = lambda : var_0
    var_2 = 0
    var_3 = module_0.Schema(var_1, var_2)
    var_4 = 'test.csv'
    var_5 = {}
    var_6 = var_3.to_csv(var_4, **var_5)



# Parsed testcases at query #30
#--------------------------






# Parsed testcases at query #31
#--------------------------

# Partially parsed test_create_stops_when_results_length_equals_iterations. Retrieved 8/9 statements.



def test_case_0():
    var_0 = 'id'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 3
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = var_5.create()
    var_7 = len(var_6)
    assert var_7 == 3



# Parsed testcases at query #32
#--------------------------






# Parsed testcases at query #33
#--------------------------






# Parsed testcases at query #34
#--------------------------





def test_case_0():
    var_0 = None
    var_1 = lambda : var_0
    var_2 = 1
    var_3 = module_0.Schema(var_1, var_2)
    var_4 = iter(var_3)
    var_5 = next(var_4)
    assert var_5 is None



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_constructor_without_seed. Retrieved 2/3 statements.
# Partially parsed test_constructor_with_seed. Retrieved 3/4 statements.
# Partially parsed test_constructor_with_none_seed. Retrieved 3/4 statements.
# Partially parsed test_constructor_with_string_seed. Retrieved 3/4 statements.



def test_case_0():
    var_0 = module_0.SchemaBuilder()
    var_1 = var_0._seed
    var_2 = var_0._random
    var_3 = var_0._schemas
    var_4 = bool(var_0._schemas == {})
    assert var_4 is True
    var_5 = var_0._data
    var_6 = bool(var_0._data == {})
    assert var_6 is True


def test_case_0():
    var_0 = 42
    var_1 = module_0.SchemaBuilder(var_0)
    var_2 = var_1._seed
    assert var_2 == 42
    var_3 = var_1._random
    var_4 = var_1._random.seed
    assert var_4 == 42
    var_5 = var_1._schemas
    var_6 = bool(var_1._schemas == {})
    assert var_6 is True
    var_7 = var_1._data
    var_8 = bool(var_1._data == {})
    assert var_8 is True


def test_case_0():
    var_0 = None
    var_1 = module_0.SchemaBuilder(var_0)
    var_2 = var_1._seed
    assert var_2 is None
    var_3 = var_1._random
    var_4 = var_1._random.seed
    assert var_4 is None
    var_5 = var_1._schemas
    var_6 = bool(var_1._schemas == {})
    assert var_6 is True
    var_7 = var_1._data
    var_8 = bool(var_1._data == {})
    assert var_8 is True


def test_case_0():
    var_0 = 'test_seed'
    var_1 = module_0.SchemaBuilder(var_0)
    var_2 = var_1._seed
    assert var_2 == 'test_seed'
    var_3 = var_1._random
    var_4 = var_1._random.seed
    assert var_4 == 'test_seed'
    var_5 = var_1._schemas
    var_6 = bool(var_1._schemas == {})
    assert var_6 is True
    var_7 = var_1._data
    var_8 = bool(var_1._data == {})
    assert var_8 is True


def test_case_0():
    var_0 = module_0.SchemaBuilder()
    var_1 = var_0._schemas
    var_2 = len(var_1)
    assert var_2 == 0
    var_3 = var_0._data
    var_4 = len(var_3)
    assert var_4 == 0
    var_5 = var_0._seed
    var_6 = var_0._random
    var_7 = bool(var_0._random is not None)
    assert var_7 is True


def test_case_0():
    var_0 = module_0.SchemaBuilder()
    var_1 = module_0.SchemaBuilder()
    var_2 = var_0._random
    var_3 = bool(var_0._random is not var_1._random)
    assert var_3 is True
    var_4 = 123
    var_5 = module_0.SchemaBuilder(var_4)
    var_6 = module_0.SchemaBuilder(var_4)
    var_7 = var_5._random
    var_8 = bool(var_5._random is not var_6._random)
    assert var_8 is True



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_iterator_skips_none_results. Retrieved 2/8 statements.
# Partially parsed test_iterator_uses_custom_context. Retrieved 8/10 statements.



def test_case_0():
    var_0 = 'id'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 5
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = var_5.iterator()
    var_7 = bool(var_6 is var_5)
    assert var_7 is True


def test_case_0():
    var_0 = 'id'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 3
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = list(var_5)
    var_7 = var_5.__iter__()
    var_8 = var_5._Schema__counter
    assert var_8 == 0


def test_case_0():
    var_0 = 'id'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 4
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = 0
    var_7 = 1
    var_8 = var_6 + var_7
    assert var_8 == 4


def test_case_0():
    var_0 = 'value'
    var_1 = 0
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 2
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = 10
    var_7 = lambda x: {var_0: x[var_0] + var_6}
    var_8 = var_5.map(var_7)
    var_9 = list(var_5)
    var_10 = bool(var_9 == [{'value': 10}, {'value': 10}])
    assert var_10 is True

def test_case_0():
    var_0 = 0
    assert var_0 == 3
    var_1 = 2


def test_case_0():
    var_0 = {}
    var_1 = lambda : var_0
    var_2 = 2
    var_3 = module_0.Schema(var_1, var_2)
    var_4 = 'data'
    var_5 = 'extra'
    var_6 = {var_5: var_4}
    var_7 = var_3.with_context(**var_6)
    var_8 = list(var_3)
    var_9 = 'extra'


def test_case_0():
    var_0 = 'id'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = module_0.Schema(var_3, var_1)
    var_5 = iter(var_4)
    var_6 = next(var_5)
    var_7 = next(var_5)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = bool(True)
    assert var_9 is True


def test_case_0():
    var_0 = 'id'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 2
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = list(var_5)
    var_7 = list(var_5)
    var_8 = bool(var_6 == var_7)
    assert var_8 is True


def test_case_0():
    var_0 = 'id'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 0
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = list(var_5)
    var_7 = bool(var_6 == [])
    assert var_7 is True


def test_case_0():
    var_0 = 'id'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 3
    var_5 = 42
    var_6 = module_0.Schema(var_3, var_4, var_5)
    var_7 = list(var_6)
    var_8 = len(var_7)
    assert var_8 == 3



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_basefield_initialization_with_defaults. Retrieved 2/3 statements.
# Failed to parse test_basefield_initialization_with_custom_locale.
# Partially parsed test_basefield_initialization_generic_instance. Retrieved 2/3 statements.
# Failed to parse test_basefield_initialization_generic_locale_matches.



def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0.seed
    var_2 = var_0._generic
    var_3 = var_0._cache
    var_4 = bool(var_0._cache == {})
    assert var_4 is True
    var_5 = var_0._handlers
    var_6 = bool(var_0._handlers == {})
    assert var_6 is True
    var_7 = var_0.aliases
    var_8 = bool(var_0.aliases == {})
    assert var_8 is True


def test_case_0():
    var_0 = 12345
    var_1 = module_0.BaseField(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 12345


def test_case_0():
    var_0 = None
    var_1 = module_0.BaseField(seed=var_0)
    var_2 = var_1.seed
    assert var_2 is None


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0.aliases
    var_2 = bool(var_0.aliases == {})
    assert var_2 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0._cache
    var_2 = bool(var_0._cache == {})
    assert var_2 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0._handlers
    var_2 = bool(var_0._handlers == {})
    assert var_2 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0._generic


def test_case_0():
    var_0 = 999
    var_1 = module_0.BaseField(seed=var_0)
    var_2 = var_1._generic.seed
    assert var_2 == 999


def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseField(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 42



# Parsed testcases at query #3
#--------------------------






# Parsed testcases at query #4
#--------------------------

# Partially parsed test_ref_calls_builder_get_data_with_correct_schema_name. Retrieved 7/11 statements.
# Partially parsed test_ref_returns_empty_list_when_builder_returns_empty_list. Retrieved 2/6 statements.
# Partially parsed test_ref_returns_list_of_complex_items_from_builder. Retrieved 11/15 statements.
# Partially parsed test_ref_with_custom_context_data_present. Retrieved 7/11 statements.
# Partially parsed test_ref_with_seed_provided. Retrieved 4/8 statements.



def test_case_0():
    var_0 = 0
    var_1 = None
    var_2 = module_0.SchemaContext(var_0, builder=var_1)
    var_3 = 'some_schema'
    var_4 = var_2.ref(var_3)
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = 'id'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 2
    var_4 = {var_0: var_3}
    var_5 = 0
    var_6 = 'test_schema'

def test_case_0():
    var_0 = 0
    var_1 = 'empty_schema'

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'Alice'
    var_3 = 30
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'Bob'
    var_6 = 25
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = [var_4, var_7]
    var_9 = 0
    var_10 = 'people'

def test_case_0():
    var_0 = 'data'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'key'
    var_4 = {var_3: var_1}
    var_5 = 5
    var_6 = 'some_schema'

def test_case_0():
    var_0 = 'item'
    var_1 = 12345
    var_2 = 0
    var_3 = 'seeded_schema'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_constructor_without_seed. Retrieved 2/3 statements.
# Partially parsed test_constructor_with_seed. Retrieved 3/4 statements.
# Partially parsed test_constructor_with_seed_zero. Retrieved 3/4 statements.
# Partially parsed test_constructor_with_negative_seed. Retrieved 3/4 statements.



def test_case_0():
    var_0 = module_0.SchemaBuilder()
    var_1 = var_0._seed
    var_2 = var_0._random
    var_3 = var_0._schemas
    var_4 = bool(var_0._schemas == {})
    assert var_4 is True
    var_5 = var_0._data
    var_6 = bool(var_0._data == {})
    assert var_6 is True


def test_case_0():
    var_0 = 42
    var_1 = module_0.SchemaBuilder(var_0)
    var_2 = var_1._seed
    assert var_2 == 42
    var_3 = var_1._random
    var_4 = var_1._random.seed
    assert var_4 == 42
    var_5 = var_1._schemas
    var_6 = bool(var_1._schemas == {})
    assert var_6 is True
    var_7 = var_1._data
    var_8 = bool(var_1._data == {})
    assert var_8 is True


def test_case_0():
    var_0 = 0
    var_1 = module_0.SchemaBuilder(var_0)
    var_2 = var_1._seed
    assert var_2 == 0
    var_3 = var_1._random
    var_4 = var_1._random.seed
    assert var_4 == 0
    var_5 = var_1._schemas
    var_6 = bool(var_1._schemas == {})
    assert var_6 is True
    var_7 = var_1._data
    var_8 = bool(var_1._data == {})
    assert var_8 is True


def test_case_0():
    var_0 = -123
    var_1 = module_0.SchemaBuilder(var_0)
    var_2 = var_1._seed
    assert var_2 == -123
    var_3 = var_1._random
    var_4 = var_1._random.seed
    assert var_4 == -123
    var_5 = var_1._schemas
    var_6 = bool(var_1._schemas == {})
    assert var_6 is True
    var_7 = var_1._data
    var_8 = bool(var_1._data == {})
    assert var_8 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_random_initialized_with_seed. Retrieved 3/4 statements.



def test_case_0():
    var_0 = 42
    var_1 = module_0.SchemaBuilder(var_0)
    var_2 = var_1._seed
    assert var_2 == 42
    var_3 = var_1._random



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_fieldset_call_with_default_iterations. Retrieved 1/6 statements.
# Partially parsed test_fieldset_call_with_specified_iterations. Retrieved 2/7 statements.
# Partially parsed test_fieldset_call_with_iterations_set_at_init. Retrieved 2/7 statements.
# Partially parsed test_fieldset_call_with_iterations_override_at_call. Retrieved 3/8 statements.
# Partially parsed test_fieldset_call_with_zero_iterations_raises_error. Retrieved 2/5 statements.
# Partially parsed test_fieldset_call_with_negative_iterations_raises_error. Retrieved 2/5 statements.
# Partially parsed test_fieldset_call_with_custom_iterations_kwarg. Retrieved 3/9 statements.
# Partially parsed test_fieldset_call_with_custom_default_iterations. Retrieved 2/8 statements.
# Partially parsed test_fieldset_call_passes_arguments_to_perform. Retrieved 7/16 statements.


def test_case_0():
    var_0 = []
    var_1 = 'username'

def test_case_0():
    var_0 = []
    var_1 = 'username'
    var_2 = 5

def test_case_0():
    var_0 = 3
    var_1 = []
    var_2 = 'username'

def test_case_0():
    var_0 = 7
    var_1 = []
    var_2 = 'username'
    var_3 = 4

def test_case_0():
    var_0 = []
    var_1 = 'username'
    var_2 = 0
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = []
    var_1 = 'username'
    var_2 = -5
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'iter'
    var_1 = 'username'
    var_2 = 6

def test_case_0():
    var_0 = 15
    var_1 = 'username'

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'arg1'
    var_3 = 'arg2'
    var_4 = 'val1'
    var_5 = 'val2'
    var_6 = 2
    var_7 = len(var_0)
    assert var_7 == 2
    var_8 = var_0[0]
    var_9 = bool(var_0[0] == (('arg1', 'arg2'), {'key1': 'val1', 'key2': 'val2'}))
    assert var_9 is True
    var_10 = var_0[1]
    var_11 = bool(var_0[1] == (('arg1', 'arg2'), {'key1': 'val1', 'key2': 'val2'}))
    assert var_11 is True



# Parsed testcases at query #8
#--------------------------






# Parsed testcases at query #9
#--------------------------

# Partially parsed test___next___skips_none_results. Retrieved 2/10 statements.
# Partially parsed test___next___applies_transformers. Retrieved 6/12 statements.
# Partially parsed test___next___uses_custom_context. Retrieved 6/12 statements.
# Partially parsed test___next___works_with_seed_in_context. Retrieved 5/17 statements.



def test_case_0():
    var_0 = 'id'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 3
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = iter(var_5)
    var_7 = next(var_6)
    var_8 = next(var_6)
    var_9 = next(var_6)
    var_10 = bool(var_7 == {'id': 1})
    assert var_10 is True
    var_11 = bool(var_8 == {'id': 1})
    assert var_11 is True
    var_12 = bool(var_9 == {'id': 1})
    assert var_12 is True


def test_case_0():
    var_0 = 'id'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 2
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = iter(var_5)
    var_7 = next(var_6)
    var_8 = next(var_6)
    var_9 = next(var_6)
    var_10 = bool(False)
    assert var_10 is True
    var_11 = bool(True)
    assert var_11 is True

def test_case_0():
    var_0 = 0
    var_1 = 2


def test_case_0():
    var_0 = 'value'
    var_1 = 5
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 2
    var_5 = module_0.Schema(var_3, var_4)


def test_case_0():
    var_0 = {}
    var_1 = lambda : var_0
    var_2 = 1
    var_3 = module_0.Schema(var_1, var_2)
    var_4 = 'value'
    var_5 = 'key'
    var_6 = {var_5: var_4}
    var_7 = var_3.with_context(**var_6)


def test_case_0():
    var_0 = 'id'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 2
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = list(var_5)
    var_7 = list(var_5)
    var_8 = bool(var_6 == [{'id': 1}, {'id': 1}])
    assert var_8 is True
    var_9 = bool(var_7 == [{'id': 1}, {'id': 1}])
    assert var_9 is True

def test_case_0():
    var_0 = 'rand'
    var_1 = 1
    var_2 = 100
    var_3 = 42
    var_4 = 2



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_constructor_without_seed. Retrieved 2/3 statements.
# Partially parsed test_constructor_with_seed. Retrieved 3/4 statements.
# Partially parsed test_constructor_with_none_seed. Retrieved 3/4 statements.
# Partially parsed test_constructor_with_string_seed. Retrieved 3/4 statements.
# Partially parsed test_constructor_seed_type_preserved. Retrieved 3/4 statements.
# Failed to parse test_constructor_empty_seed_object.



def test_case_0():
    var_0 = module_0.SchemaBuilder()
    var_1 = var_0._seed
    var_2 = var_0._random
    var_3 = var_0._schemas
    var_4 = bool(var_0._schemas == {})
    assert var_4 is True
    var_5 = var_0._data
    var_6 = bool(var_0._data == {})
    assert var_6 is True


def test_case_0():
    var_0 = 42
    var_1 = module_0.SchemaBuilder(var_0)
    var_2 = var_1._seed
    assert var_2 == 42
    var_3 = var_1._random
    var_4 = var_1._random.seed
    assert var_4 == 42
    var_5 = var_1._schemas
    var_6 = bool(var_1._schemas == {})
    assert var_6 is True
    var_7 = var_1._data
    var_8 = bool(var_1._data == {})
    assert var_8 is True


def test_case_0():
    var_0 = None
    var_1 = module_0.SchemaBuilder(var_0)
    var_2 = var_1._seed
    assert var_2 is None
    var_3 = var_1._random
    var_4 = var_1._schemas
    var_5 = bool(var_1._schemas == {})
    assert var_5 is True
    var_6 = var_1._data
    var_7 = bool(var_1._data == {})
    assert var_7 is True


def test_case_0():
    var_0 = 'test'
    var_1 = module_0.SchemaBuilder(var_0)
    var_2 = var_1._seed
    assert var_2 == 'test'
    var_3 = var_1._random
    var_4 = var_1._schemas
    var_5 = bool(var_1._schemas == {})
    assert var_5 is True
    var_6 = var_1._data
    var_7 = bool(var_1._data == {})
    assert var_7 is True


def test_case_0():
    var_0 = module_0.SchemaBuilder()
    var_1 = var_0._schemas
    var_2 = len(var_1)
    assert var_2 == 0
    var_3 = var_0._data
    var_4 = len(var_3)
    assert var_4 == 0
    var_5 = var_0._random
    var_6 = bool(var_0._random is not None)
    assert var_6 is True


def test_case_0():
    var_0 = 123.456
    var_1 = module_0.SchemaBuilder(var_0)
    var_2 = var_1._seed
    var_3 = bool(var_1._seed == 123.456)
    assert var_3 is True
    var_4 = var_1._random



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_next_skips_none_results. Retrieved 2/10 statements.
# Partially parsed test_next_respects_custom_context. Retrieved 2/9 statements.



def test_case_0():
    var_0 = 'id'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 3
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = var_5.__iter__()
    var_7 = var_5.__next__()
    var_8 = var_5.__next__()
    var_9 = var_5.__next__()
    var_10 = bool(var_7 == {'id': 1})
    assert var_10 is True
    var_11 = bool(var_8 == {'id': 1})
    assert var_11 is True
    var_12 = bool(var_9 == {'id': 1})
    assert var_12 is True


def test_case_0():
    var_0 = 'id'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = module_0.Schema(var_3, var_1)
    var_5 = var_4.__iter__()
    var_6 = var_4.__next__()
    var_7 = var_4.__next__()
    var_8 = bool(False)
    assert var_8 is True
    var_9 = bool(True)
    assert var_9 is True

def test_case_0():
    var_0 = 0
    assert var_0 == 3
    var_1 = 2


def test_case_0():
    var_0 = 'id'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 2
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = lambda item: {var_0: item[var_0] + var_1}
    var_7 = var_5.map(var_6)
    var_8 = var_5.__iter__()
    var_9 = var_5.__next__()
    var_10 = var_5.__next__()
    var_11 = bool(var_9 == {'id': 2})
    assert var_11 is True
    var_12 = bool(var_10 == {'id': 2})
    assert var_12 is True

def test_case_0():
    var_0 = 2
    var_1 = 'test_value'


def test_case_0():
    var_0 = 'id'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 2
    var_5 = 42
    var_6 = module_0.Schema(var_3, var_4, var_5)
    var_7 = var_6.__iter__()
    var_8 = var_6.__next__()
    var_9 = var_6.__next__()
    var_10 = bool(var_8 == {'id': 1})
    assert var_10 is True
    var_11 = bool(var_9 == {'id': 1})
    assert var_11 is True


def test_case_0():
    var_0 = 'id'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 3
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = var_5.__iter__()
    var_7 = var_5._Schema__counter
    assert var_7 == 0
    var_8 = var_5.__next__()
    var_9 = var_5._Schema__counter
    assert var_9 == 1
    var_10 = var_5.__next__()
    var_11 = var_5._Schema__counter
    assert var_11 == 2
    var_12 = var_5.__next__()
    var_13 = var_5._Schema__counter
    assert var_13 == 3


def test_case_0():
    var_0 = 'id'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 0
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = var_5.__iter__()
    var_7 = var_5.__next__()
    var_8 = bool(False)
    assert var_8 is True
    var_9 = bool(True)
    assert var_9 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_basefield_initialization_with_default_locale_and_seed. Retrieved 4/7 statements.
# Failed to parse test_basefield_initialization_with_custom_locale.
# Partially parsed test_basefield_initialization_generic_instance. Retrieved 2/3 statements.
# Failed to parse test_basefield_initialization_generic_locale_matches.



def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0.seed
    var_2 = var_0._generic.locale
    var_3 = var_0._cache
    var_4 = var_0._cache
    var_5 = bool(var_0._cache == {})
    assert var_5 is True
    var_6 = var_0._handlers
    var_7 = var_0._handlers
    var_8 = bool(var_0._handlers == {})
    assert var_8 is True
    var_9 = var_0.aliases
    var_10 = var_0.aliases
    var_11 = bool(var_0.aliases == {})
    assert var_11 is True


def test_case_0():
    var_0 = 12345
    var_1 = module_0.BaseField(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 12345


def test_case_0():
    var_0 = None
    var_1 = module_0.BaseField(seed=var_0)
    var_2 = var_1.seed
    assert var_2 is None


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0.aliases
    var_2 = bool(var_0.aliases == {})
    assert var_2 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0._cache
    var_2 = bool(var_0._cache == {})
    assert var_2 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0._handlers
    var_2 = bool(var_0._handlers == {})
    assert var_2 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0._generic


def test_case_0():
    var_0 = 999
    var_1 = module_0.BaseField(seed=var_0)
    var_2 = var_1._generic.seed
    assert var_2 == 999



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_register_handler_success. Retrieved 2/5 statements.
# Partially parsed test_register_handler_duplicate. Retrieved 2/8 statements.
# Partially parsed test_register_handler_non_string_name. Retrieved 2/6 statements.
# Partially parsed test_register_handler_invalid_identifier. Retrieved 2/6 statements.
# Partially parsed test_register_handler_insufficient_parameters. Retrieved 2/6 statements.



def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'custom_field'
    var_2 = 'custom_field'
    var_3 = bool('custom_field' in var_0._handlers)
    assert var_3 is True
    var_4 = var_0._handlers['custom_field']


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'duplicate_field'
    var_2 = var_0._handlers['duplicate_field']


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 123
    var_2 = bool(False)
    assert var_2 is True
    var_3 = bool(True)
    assert var_3 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'invalid-field'
    var_2 = bool(False)
    assert var_2 is True
    var_3 = bool(True)
    assert var_3 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'field_name'
    var_2 = 'not_callable'
    var_3 = var_0.register_handler(var_1, var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'insufficient'
    var_2 = bool(False)
    assert var_2 is True
    var_3 = bool(True)
    assert var_3 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_create_returns_items_from_schema. Retrieved 8/10 statements.
# Partially parsed test_create_applies_transformers. Retrieved 10/12 statements.
# Partially parsed test_create_skips_none_results. Retrieved 2/9 statements.
# Partially parsed test_create_with_custom_context. Retrieved 13/15 statements.
# Partially parsed test_create_with_seed. Retrieved 5/15 statements.
# Partially parsed test_create_with_multiple_transformers. Retrieved 12/14 statements.
# Partially parsed test_create_transformer_with_no_parameters. Retrieved 11/13 statements.



def test_case_0():
    var_0 = 'id'
    var_1 = 'name'
    var_2 = 1
    var_3 = 'test'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = lambda : var_4
    var_6 = 5
    var_7 = module_0.Schema(var_5, var_6)
    var_8 = var_7.create()
    var_9 = len(var_8)
    assert var_9 == 5


def test_case_0():
    var_0 = 'id'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 3
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = var_5.create()
    var_7 = {var_0: var_1}


def test_case_0():
    var_0 = 'value'
    var_1 = 5
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 2
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = lambda x: {var_0: x[var_0] * var_4}
    var_7 = var_5.map(var_6)
    var_8 = var_5.create()
    var_9 = 10


def test_case_0():
    var_0 = 'index'
    var_1 = 0
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 2
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = lambda item, ctx: {var_0: ctx.index}
    var_7 = var_5.map(var_6)
    var_8 = var_5.create()
    var_9 = var_8[0]['index']
    assert var_9 == 0
    var_10 = var_8[1]['index']
    assert var_10 == 1

def test_case_0():
    var_0 = 0
    var_1 = 3


def test_case_0():
    var_0 = 'ctx_value'
    var_1 = 0
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 2
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = 'custom_value'
    var_7 = 'custom_key'
    var_8 = {var_7: var_6}
    var_9 = var_5.with_context(**var_8)
    var_10 = 'custom'
    var_11 = 'custom_key'
    var_12 = lambda item, ctx: {var_0: item[var_0], var_10: ctx.custom[var_11]}
    var_13 = var_5.map(var_12)
    var_14 = var_5.create()

def test_case_0():
    var_0 = 42
    var_1 = 'random'
    var_2 = 1
    var_3 = 100
    var_4 = 2


def test_case_0():
    var_0 = {}
    var_1 = lambda : var_0
    var_2 = 0
    var_3 = module_0.Schema(var_1, var_2)
    var_4 = bool(False)
    assert var_4 is True


def test_case_0():
    var_0 = 'not a callable'
    var_1 = 5
    var_2 = module_0.Schema(var_0, var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True


def test_case_0():
    var_0 = {}
    var_1 = lambda : var_0
    var_2 = 1
    var_3 = module_0.Schema(var_1, var_2)
    var_4 = var_3.create()
    var_5 = bool(var_4 == [{}])
    assert var_5 is True


def test_case_0():
    var_0 = 'value'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 2
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = lambda x: {var_0: x[var_0] + var_1}
    var_7 = var_5.map(var_6)
    var_8 = lambda x: {var_0: x[var_0] * var_4}
    var_9 = var_5.map(var_8)
    var_10 = var_5.create()
    var_11 = 4


def test_case_0():
    var_0 = 'value'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 2
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = 999
    var_7 = {var_0: var_6}
    var_8 = lambda : var_7
    var_9 = var_5.map(var_8)
    var_10 = var_5.create()



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_register_handler_success. Retrieved 2/5 statements.
# Partially parsed test_register_handler_duplicate. Retrieved 2/8 statements.
# Partially parsed test_register_handler_non_string_name. Retrieved 2/6 statements.
# Partially parsed test_register_handler_invalid_identifier. Retrieved 2/6 statements.
# Partially parsed test_register_handler_insufficient_parameters. Retrieved 2/6 statements.
# Partially parsed test_register_handler_sufficient_parameters. Retrieved 2/5 statements.



def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'custom_field'
    var_2 = 'custom_field'
    var_3 = bool('custom_field' in var_0._handlers)
    assert var_3 is True
    var_4 = var_0._handlers['custom_field']


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'duplicate_field'
    var_2 = var_0._handlers['duplicate_field']


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 123


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'invalid-field'
    var_2 = 'invalid-field'


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'field_name'
    var_2 = 'not_callable'
    var_3 = var_0.register_handler(var_1, var_2)


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'insufficient'


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'sufficient'
    var_2 = 'sufficient'
    var_3 = bool('sufficient' in var_0._handlers)
    assert var_3 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_fieldset_raises_error_when_iterations_less_than_one. Retrieved 2/5 statements.


def test_case_0():
    var_0 = []
    var_1 = 'username'
    var_2 = 0
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #17
#--------------------------






# Parsed testcases at query #18
#--------------------------






# Parsed testcases at query #19
#--------------------------

# Partially parsed test_perform_with_valid_explicit_field. Retrieved 3/4 statements.
# Partially parsed test_perform_with_valid_fuzzy_field. Retrieved 3/4 statements.
# Partially parsed test_perform_with_aliases. Retrieved 4/6 statements.
# Partially parsed test_perform_with_key_function. Retrieved 4/5 statements.
# Partially parsed test_perform_with_key_function_using_random. Retrieved 2/6 statements.
# Partially parsed test_perform_with_custom_handler. Retrieved 3/6 statements.
# Partially parsed test_perform_with_different_delimiters. Retrieved 7/10 statements.
# Partially parsed test_perform_with_invalid_aliases_type. Retrieved 3/5 statements.
# Partially parsed test_perform_with_kwargs. Retrieved 4/5 statements.
# Partially parsed test_perform_cache_usage. Retrieved 4/6 statements.



def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'person.full_name'
    var_2 = {}
    var_3 = var_0.perform(var_1, **var_2)


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'full_name'
    var_2 = {}
    var_3 = var_0.perform(var_1, **var_2)


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'alias_name'
    var_2 = 'person.full_name'
    var_3 = {}
    var_4 = var_0.perform(var_1, **var_3)


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'person.full_name'
    var_2 = lambda x: x.upper()
    var_3 = {}
    var_4 = var_0.perform(var_1, var_2, **var_3)


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'person.full_name'


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'custom_field'
    var_2 = {}
    var_3 = var_0.perform(var_1, **var_2)
    assert var_3 == 'custom_value'


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'person:full_name'
    var_2 = {}
    var_3 = var_0.perform(var_1, **var_2)
    var_4 = 'person/full_name'
    var_5 = {}
    var_6 = var_0.perform(var_4, **var_5)
    var_7 = 'person full_name'
    var_8 = {}
    var_9 = var_0.perform(var_7, **var_8)


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'invalid_provider.invalid_method'
    var_2 = {}
    var_3 = var_0.perform(var_1, **var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = None
    var_2 = {}
    var_3 = var_0.perform(var_1, **var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'person.full_name'
    var_2 = {}
    var_3 = var_0.perform(var_1, **var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'person.first_name'
    var_2 = 'F'
    var_3 = 'sex'
    var_4 = {var_3: var_2}
    var_5 = var_0.perform(var_1, **var_4)


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'person.full_name'
    var_2 = {}
    var_3 = var_0.perform(var_1, **var_2)
    var_4 = {}
    var_5 = var_0.perform(var_1, **var_4)


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'provider.method.submethod'
    var_2 = {}
    var_3 = var_0.perform(var_1, **var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_pick_from_calls_builder_pick_from. Retrieved 3/8 statements.
# Partially parsed test_pick_from_calls_builder_pick_from_without_field. Retrieved 5/10 statements.



def test_case_0():
    var_0 = 0
    var_1 = module_0.SchemaContext(var_0)
    var_2 = 'some_schema'
    var_3 = var_1.pick_from(var_2)
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 0
    var_1 = 'test_schema'
    var_2 = 'field_name'

def test_case_0():
    var_0 = 'id'
    var_1 = 1
    var_2 = 0
    var_3 = 'test_schema'
    var_4 = None



# Parsed testcases at query #21
#--------------------------





def test_case_0():
    var_0 = 0
    var_1 = None
    var_2 = module_0.SchemaContext(var_0, builder=var_1)
    var_3 = 'some_schema'
    var_4 = var_2.ref(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_perform_key_function_with_two_parameters. Retrieved 4/7 statements.



def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'random_int'
    var_2 = 0
    var_3 = 10



# Parsed testcases at query #23
#--------------------------






# Parsed testcases at query #24
#--------------------------






# Parsed testcases at query #25
#--------------------------





def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'handler1'
    var_2 = lambda random, **kwargs: var_1
    var_3 = 'handler2'
    var_4 = lambda random, **kwargs: var_3
    var_5 = 'custom1'
    var_6 = (var_5, var_2)
    var_7 = 'custom2'
    var_8 = (var_7, var_4)
    var_9 = [var_6, var_8]
    var_10 = var_0.register_handlers(var_9)
    var_11 = {}
    var_12 = var_0.perform(var_5, **var_11)
    assert var_12 == 'handler1'
    var_13 = {}
    var_14 = var_0.perform(var_7, **var_13)
    assert var_14 == 'handler2'


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = []
    var_2 = var_0.register_handlers(var_1)
    var_3 = var_0._handlers
    var_4 = bool(var_0._handlers == {})
    assert var_4 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'old'
    var_2 = lambda random, **kwargs: var_1
    var_3 = 'new'
    var_4 = lambda random, **kwargs: var_3
    var_5 = 'custom'
    var_6 = var_0.register_handler(var_5, var_2)
    var_7 = (var_5, var_4)
    var_8 = [var_7]
    var_9 = var_0.register_handlers(var_8)
    var_10 = {}
    var_11 = var_0.perform(var_5, **var_10)
    assert var_11 == 'new'


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'a'
    var_2 = 'A'
    var_3 = lambda r, **k: var_2
    var_4 = (var_1, var_3)
    var_5 = 'b'
    var_6 = 'B'
    var_7 = lambda r, **k: var_6
    var_8 = (var_5, var_7)
    var_9 = 'c'
    var_10 = 'C'
    var_11 = lambda r, **k: var_10
    var_12 = (var_9, var_11)
    var_13 = [var_4, var_8, var_12]
    var_14 = var_0.register_handlers(var_13)
    var_15 = var_0._handlers
    var_16 = len(var_15)
    assert var_16 == 3
    var_17 = {}
    var_18 = var_0.perform(var_1, **var_17)
    assert var_18 == 'A'
    var_19 = {}
    var_20 = var_0.perform(var_5, **var_19)
    assert var_20 == 'B'
    var_21 = {}
    var_22 = var_0.perform(var_9, **var_21)
    assert var_22 == 'C'


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = None
    var_2 = lambda random, **kwargs: var_1
    var_3 = 123
    var_4 = (var_3, var_2)
    var_5 = [var_4]
    var_6 = var_0.register_handlers(var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = bool(True)
    assert var_8 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'custom'
    var_2 = 'not_callable'
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = var_0.register_handlers(var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = bool(True)
    assert var_7 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = None
    var_2 = lambda random: var_1
    var_3 = 'custom'
    var_4 = (var_3, var_2)
    var_5 = [var_4]
    var_6 = var_0.register_handlers(var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = bool(True)
    assert var_8 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'first'
    var_2 = lambda random, **kwargs: var_1
    var_3 = 'second'
    var_4 = lambda random, **kwargs: var_3
    var_5 = 'dup'
    var_6 = (var_5, var_2)
    var_7 = (var_5, var_4)
    var_8 = [var_6, var_7]
    var_9 = var_0.register_handlers(var_8)
    var_10 = {}
    var_11 = var_0.perform(var_5, **var_10)
    assert var_11 == 'second'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test___next___returns_items_until_iterations. Retrieved 1/8 statements.
# Partially parsed test___next___raises_stop_iteration_after_iterations. Retrieved 1/9 statements.
# Partially parsed test___next___skips_none_results_and_continues. Retrieved 2/12 statements.
# Partially parsed test___next___applies_transformers. Retrieved 1/11 statements.
# Partially parsed test___next___uses_context_in_transformer. Retrieved 1/12 statements.
# Partially parsed test___next___with_custom_context. Retrieved 2/13 statements.
# Partially parsed test___next___resets_counter_on_new_iter. Retrieved 1/11 statements.


def test_case_0():
    var_0 = 3

def test_case_0():
    var_0 = 2
    var_1 = bool(False)
    assert var_1 is True
    var_2 = bool(True)
    assert var_2 is True

def test_case_0():
    var_0 = 0
    assert var_0 == 3
    var_1 = 2

def test_case_0():
    var_0 = 2

def test_case_0():
    var_0 = 3

def test_case_0():
    var_0 = 2
    var_1 = 10

def test_case_0():
    var_0 = 2
    var_1 = bool(False)
    assert var_1 is True
    var_2 = bool(True)
    assert var_2 is True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_create_stops_when_results_length_equals_iterations. Retrieved 8/9 statements.



def test_case_0():
    var_0 = 'id'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 3
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = var_5.create()
    var_7 = len(var_6)
    assert var_7 == 3



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_seed_is_missing_seed. Retrieved 2/3 statements.



def test_case_0():
    var_0 = module_0.SchemaBuilder()
    var_1 = var_0._seed
    var_2 = var_0._random



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_basefield_initialization_with_default_locale_and_seed. Retrieved 4/7 statements.
# Failed to parse test_basefield_initialization_with_custom_locale.
# Partially parsed test_basefield_initialization_generic_instance_created. Retrieved 2/3 statements.



def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0.seed
    var_2 = var_0._generic.locale
    var_3 = var_0._cache
    var_4 = var_0._cache
    var_5 = bool(var_0._cache == {})
    assert var_5 is True
    var_6 = var_0._handlers
    var_7 = var_0._handlers
    var_8 = bool(var_0._handlers == {})
    assert var_8 is True
    var_9 = var_0.aliases
    var_10 = var_0.aliases
    var_11 = bool(var_0.aliases == {})
    assert var_11 is True


def test_case_0():
    var_0 = 12345
    var_1 = module_0.BaseField(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 12345


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0.aliases
    var_2 = bool(var_0.aliases == {})
    assert var_2 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0._cache
    var_2 = bool(var_0._cache == {})
    assert var_2 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0._handlers
    var_2 = bool(var_0._handlers == {})
    assert var_2 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0._generic



# Parsed testcases at query #30
#--------------------------





def test_case_0():
    var_0 = 'id'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 5
    var_5 = 42
    var_6 = module_0.Schema(var_3, var_4, var_5)
    var_7 = var_6.iterations
    assert var_7 == 5
    var_8 = var_6._Schema__schema
    var_9 = bool(var_6._Schema__schema == var_3)
    assert var_9 is True
    var_10 = var_6._Schema__seed
    assert var_10 == 42
    var_11 = var_6._Schema__counter
    assert var_11 == 0
    var_12 = var_6._transformers
    var_13 = bool(var_6._transformers == [])
    assert var_13 is True
    var_14 = var_6._custom_context
    var_15 = bool(var_6._custom_context == {})
    assert var_15 is True


def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = module_0.Schema(var_3)
    var_5 = var_4.iterations
    assert var_5 == 10
    var_6 = var_4._Schema__seed
    var_7 = var_4._Schema__counter
    assert var_7 == 0


def test_case_0():
    var_0 = {}
    var_1 = lambda : var_0
    var_2 = 0
    var_3 = module_0.Schema(var_1, var_2)
    var_4 = bool(False)
    assert var_4 is True


def test_case_0():
    var_0 = 'not a callable'
    var_1 = module_0.Schema(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = bool(True)
    assert var_3 is True



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_seed_is_not_missing_seed. Retrieved 3/4 statements.



def test_case_0():
    var_0 = 42
    var_1 = module_0.SchemaBuilder(var_0)
    var_2 = var_1._seed
    assert var_2 == 42
    var_3 = var_1._random



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_iterator_skips_none_results. Retrieved 5/14 statements.
# Partially parsed test_iterator_uses_custom_context. Retrieved 5/12 statements.
# Partially parsed test_iterator_with_seed. Retrieved 2/9 statements.
# Partially parsed test_iterator_index_in_context. Retrieved 5/12 statements.



def test_case_0():
    var_0 = 'id'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 3
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = var_5.iterator()
    var_7 = bool(var_6 is var_5)
    assert var_7 is True


def test_case_0():
    var_0 = 'id'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 2
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = list(var_5)
    var_7 = var_5.__iter__()
    var_8 = var_5._Schema__counter
    assert var_8 == 0


def test_case_0():
    var_0 = 'value'
    var_1 = 5
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = module_0.Schema(var_3, var_1)
    var_5 = 0
    var_6 = 1
    var_7 = var_5 + var_6
    assert var_7 == 5

def test_case_0():
    var_0 = 0
    var_1 = 3
    var_2 = 'id'
    var_3 = 2
    var_4 = 1


def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 2
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = 10
    var_7 = lambda item: {var_0: item[var_0] + var_6}
    var_8 = var_5.map(var_7)
    var_9 = list(var_5)
    var_10 = bool(var_9 == [{'x': 11}, {'x': 11}])
    assert var_10 is True

def test_case_0():
    var_0 = 1
    var_1 = 'custom_value'
    var_2 = 'ctx'
    var_3 = 'custom_key'
    var_4 = lambda item, ctx: {var_2: ctx.custom[var_3]}

def test_case_0():
    var_0 = 42
    var_1 = 2


def test_case_0():
    var_0 = 'data'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 0
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = iter(var_5)
    var_7 = next(var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = bool(True)
    assert var_9 is True


def test_case_0():
    var_0 = 'i'
    var_1 = 0
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 2
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = list(var_5)
    var_7 = list(var_5)
    var_8 = bool(var_6 == var_7)
    assert var_8 is True
    var_9 = len(var_6)
    assert var_9 == 2

def test_case_0():
    var_0 = []
    var_1 = 3
    var_2 = 'index'
    var_3 = lambda item, ctx: {var_2: ctx.index}
    var_4 = 'index'
    var_5 = bool(var_0 == [0, 1, 2])
    assert var_5 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_pick_from_calls_builder_pick_from_with_correct_arguments. Retrieved 4/9 statements.
# Partially parsed test_pick_from_calls_builder_pick_from_without_field. Retrieved 4/9 statements.



def test_case_0():
    var_0 = 0
    var_1 = None
    var_2 = module_0.SchemaContext(var_0, builder=var_1)
    var_3 = 'test_schema'
    var_4 = var_2.pick_from(var_3)
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = 'picked_item'
    var_1 = 0
    var_2 = 'test_schema'
    var_3 = 'test_field'

def test_case_0():
    var_0 = 'picked_item'
    var_1 = 0
    var_2 = 'test_schema'
    var_3 = None



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_basefield_initialization_with_default_locale_and_seed. Retrieved 4/7 statements.
# Failed to parse test_basefield_initialization_with_custom_locale.
# Partially parsed test_basefield_initialization_generic_instance_created. Retrieved 2/3 statements.
# Failed to parse test_basefield_initialization_generic_locale_matches.
# Failed to parse test_basefield_initialization_with_missingseed.



def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0.seed
    var_2 = var_0._generic.locale
    var_3 = var_0._cache
    var_4 = var_0._cache
    var_5 = bool(var_0._cache == {})
    assert var_5 is True
    var_6 = var_0._handlers
    var_7 = var_0._handlers
    var_8 = bool(var_0._handlers == {})
    assert var_8 is True
    var_9 = var_0.aliases
    var_10 = var_0.aliases
    var_11 = bool(var_0.aliases == {})
    assert var_11 is True


def test_case_0():
    var_0 = 12345
    var_1 = module_0.BaseField(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 12345


def test_case_0():
    var_0 = None
    var_1 = module_0.BaseField(seed=var_0)
    var_2 = var_1.seed
    assert var_2 is None


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0.aliases
    var_2 = bool(var_0.aliases == {})
    assert var_2 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0._cache
    var_2 = bool(var_0._cache == {})
    assert var_2 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0._handlers
    var_2 = bool(var_0._handlers == {})
    assert var_2 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0._generic


def test_case_0():
    var_0 = 999
    var_1 = module_0.BaseField(seed=var_0)
    var_2 = var_1._generic.seed
    assert var_2 == 999


def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseField(seed=var_0)
    var_2 = var_1.seed
    assert var_2 == 42



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_with_context_used_in_create. Retrieved 2/7 statements.
# Partially parsed test_with_context_used_in_iterator. Retrieved 2/7 statements.



def test_case_0():
    var_0 = {}
    var_1 = lambda : var_0
    var_2 = module_0.Schema(var_1)
    var_3 = 'value1'
    var_4 = 42
    var_5 = 'key1'
    var_6 = 'key2'
    var_7 = {var_5: var_3, var_6: var_4}
    var_8 = var_2.with_context(**var_7)
    var_9 = var_2._custom_context
    var_10 = bool(var_2._custom_context == {'key1': 'value1', 'key2': 42})
    assert var_10 is True


def test_case_0():
    var_0 = {}
    var_1 = lambda : var_0
    var_2 = module_0.Schema(var_1)
    var_3 = 1
    var_4 = 'a'
    var_5 = {var_4: var_3}
    var_6 = var_2.with_context(**var_5)
    var_7 = bool(var_6 is var_2)
    assert var_7 is True


def test_case_0():
    var_0 = {}
    var_1 = lambda : var_0
    var_2 = module_0.Schema(var_1)
    var_3 = 10
    var_4 = 'x'
    var_5 = {var_4: var_3}
    var_6 = var_2.with_context(**var_5)
    var_7 = 20
    var_8 = 'y'
    var_9 = {var_8: var_7}
    var_10 = var_6.with_context(**var_9)
    var_11 = var_2._custom_context
    var_12 = bool(var_2._custom_context == {'x': 10, 'y': 20})
    assert var_12 is True


def test_case_0():
    var_0 = {}
    var_1 = lambda : var_0
    var_2 = module_0.Schema(var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 'a'
    var_6 = 'b'
    var_7 = {var_5: var_3, var_6: var_4}
    var_8 = var_2.with_context(**var_7)
    var_9 = 100
    var_10 = 3
    var_11 = 'a'
    var_12 = 'c'
    var_13 = {var_11: var_9, var_12: var_10}
    var_14 = var_2.with_context(**var_13)
    var_15 = var_2._custom_context
    var_16 = bool(var_2._custom_context == {'a': 100, 'b': 2, 'c': 3})
    assert var_16 is True


def test_case_0():
    var_0 = {}
    var_1 = lambda : var_0
    var_2 = module_0.Schema(var_1)
    var_3 = {}
    var_4 = var_2.with_context(**var_3)
    var_5 = var_2._custom_context
    var_6 = bool(var_2._custom_context == {})
    assert var_6 is True

def test_case_0():
    var_0 = 1
    var_1 = 'my_value'

def test_case_0():
    var_0 = 1
    var_1 = 'my_value'



# Parsed testcases at query #5
#--------------------------





def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'test1'
    var_2 = lambda random, **kwargs: var_1
    var_3 = 'test2'
    var_4 = lambda random, **kwargs: var_3
    var_5 = 'custom_field1'
    var_6 = (var_5, var_2)
    var_7 = 'custom_field2'
    var_8 = (var_7, var_4)
    var_9 = [var_6, var_8]
    var_10 = var_0.register_handlers(var_9)
    var_11 = 'custom_field1'
    var_12 = bool('custom_field1' in var_0._handlers)
    assert var_12 is True
    var_13 = 'custom_field2'
    var_14 = bool('custom_field2' in var_0._handlers)
    assert var_14 is True
    var_15 = var_0._handlers['custom_field1']
    var_16 = bool(var_0._handlers['custom_field1'] is var_2)
    assert var_16 is True
    var_17 = var_0._handlers['custom_field2']
    var_18 = bool(var_0._handlers['custom_field2'] is var_4)
    assert var_18 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'test1'
    var_2 = lambda random, **kwargs: var_1
    var_3 = 'test2'
    var_4 = lambda random, **kwargs: var_3
    var_5 = 'custom_field'
    var_6 = var_0.register_handler(var_5, var_2)
    var_7 = (var_5, var_4)
    var_8 = [var_7]
    var_9 = var_0.register_handlers(var_8)
    var_10 = var_0._handlers['custom_field']
    var_11 = bool(var_0._handlers['custom_field'] is var_4)
    assert var_11 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = []
    var_2 = var_0.register_handlers(var_1)
    var_3 = var_0._handlers
    var_4 = bool(var_0._handlers == {})
    assert var_4 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'test'
    var_2 = lambda random, **kwargs: var_1
    var_3 = 'single_field'
    var_4 = (var_3, var_2)
    var_5 = [var_4]
    var_6 = var_0.register_handlers(var_5)
    var_7 = 'single_field'
    var_8 = bool('single_field' in var_0._handlers)
    assert var_8 is True
    var_9 = var_0._handlers['single_field']
    var_10 = bool(var_0._handlers['single_field'] is var_2)
    assert var_10 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 5
    var_2 = range(var_1)
    var_3 = [(f'field{i}', lambda random, **kwargs: f'test{i}') for i in var_2]
    var_4 = var_0.register_handlers(var_3)


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'test'
    var_2 = lambda random, **kwargs: var_1
    var_3 = 123
    var_4 = (var_3, var_2)
    var_5 = [var_4]
    var_6 = var_0.register_handlers(var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = bool(True)
    assert var_8 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'test'
    var_2 = lambda random, **kwargs: var_1
    var_3 = 'invalid-field'
    var_4 = (var_3, var_2)
    var_5 = [var_4]
    var_6 = var_0.register_handlers(var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = bool(True)
    assert var_8 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'custom_field'
    var_2 = 'not_callable'
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = var_0.register_handlers(var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = bool(True)
    assert var_7 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'test'
    var_2 = lambda random: var_1
    var_3 = 'custom_field'
    var_4 = (var_3, var_2)
    var_5 = [var_4]
    var_6 = var_0.register_handlers(var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = bool(True)
    assert var_8 is True


def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'test'
    var_2 = lambda : var_1
    var_3 = 'custom_field'
    var_4 = (var_3, var_2)
    var_5 = [var_4]
    var_6 = var_0.register_handlers(var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = bool(True)
    assert var_8 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_next_skips_none_results_and_continues. Retrieved 2/10 statements.
# Partially parsed test_next_applies_transformers. Retrieved 6/13 statements.
# Partially parsed test_next_uses_context_in_transformer. Retrieved 6/13 statements.
# Partially parsed test_next_with_custom_context. Retrieved 8/15 statements.



def test_case_0():
    var_0 = 'id'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 3
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = var_5.__iter__()
    var_7 = var_5.__next__()
    var_8 = var_5.__next__()
    var_9 = var_5.__next__()
    var_10 = bool(var_7 == {'id': 1})
    assert var_10 is True
    var_11 = bool(var_8 == {'id': 1})
    assert var_11 is True
    var_12 = bool(var_9 == {'id': 1})
    assert var_12 is True


def test_case_0():
    var_0 = 'id'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = module_0.Schema(var_3, var_1)
    var_5 = var_4.__iter__()
    var_6 = var_4.__next__()
    var_7 = var_4.__next__()
    var_8 = bool(False)
    assert var_8 is True
    var_9 = bool(True)
    assert var_9 is True

def test_case_0():
    var_0 = 0
    assert var_0 == 3
    var_1 = 2


def test_case_0():
    var_0 = 'value'
    var_1 = 5
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 2
    var_5 = module_0.Schema(var_3, var_4)


def test_case_0():
    var_0 = 'index'
    var_1 = 0
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 2
    var_5 = module_0.Schema(var_3, var_4)


def test_case_0():
    var_0 = 'id'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 2
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = var_5.__iter__()
    var_7 = var_5.__next__()
    var_8 = var_5.__iter__()
    var_9 = var_5.__next__()
    var_10 = var_5.__next__()
    var_11 = bool(var_7 == {'id': 1})
    assert var_11 is True
    var_12 = bool(var_9 == {'id': 1})
    assert var_12 is True
    var_13 = bool(var_10 == {'id': 1})
    assert var_13 is True


def test_case_0():
    var_0 = 'ctx'
    var_1 = ''
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 2
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = 'custom_value'
    var_7 = 'key'
    var_8 = {var_7: var_6}
    var_9 = var_5.with_context(**var_8)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_create_returns_list_of_correct_length. Retrieved 10/11 statements.
# Partially parsed test_create_returns_list_of_dicts. Retrieved 9/11 statements.
# Partially parsed test_create_applies_transformers. Retrieved 11/13 statements.
# Partially parsed test_create_uses_custom_context. Retrieved 12/14 statements.
# Partially parsed test_create_skips_none_results. Retrieved 5/14 statements.
# Partially parsed test_create_with_seed_produces_deterministic_results. Retrieved 2/9 statements.



def test_case_0():
    var_0 = 'id'
    var_1 = 'name'
    var_2 = 1
    var_3 = 'test'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = lambda : var_4
    var_6 = 5
    var_7 = module_0.Schema(var_5, var_6)
    var_8 = var_7.create()
    var_9 = len(var_8)
    assert var_9 == 5


def test_case_0():
    var_0 = 'id'
    var_1 = 'name'
    var_2 = 1
    var_3 = 'test'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = lambda : var_4
    var_6 = 3
    var_7 = module_0.Schema(var_5, var_6)
    var_8 = var_7.create()


def test_case_0():
    var_0 = 'value'
    var_1 = 5
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 2
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = 'doubled'
    var_7 = lambda x: {var_6: x}
    var_8 = var_5.map(var_7)
    var_9 = var_5.create()
    var_10 = 10


def test_case_0():
    var_0 = 'context_value'
    var_1 = None
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 2
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = 'my_value'
    var_7 = 'my_key'
    var_8 = {var_7: var_6}
    var_9 = var_5.with_context(**var_8)
    var_10 = 'my_key'
    var_11 = lambda item, ctx: {var_0: item}
    var_12 = var_5.map(var_11)
    var_13 = var_5.create()

def test_case_0():
    var_0 = 0
    var_1 = 3
    var_2 = 'id'
    var_3 = 2
    var_4 = 1


def test_case_0():
    var_0 = {}
    var_1 = lambda : var_0
    var_2 = 0
    var_3 = module_0.Schema(var_1, var_2)
    var_4 = bool(False)
    assert var_4 is True


def test_case_0():
    var_0 = {}
    var_1 = lambda : var_0
    var_2 = -5
    var_3 = module_0.Schema(var_1, var_2)
    var_4 = bool(False)
    assert var_4 is True


def test_case_0():
    var_0 = 'not a callable'
    var_1 = 5
    var_2 = module_0.Schema(var_0, var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True


def test_case_0():
    var_0 = 'id'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 2
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = var_5.create()
    var_7 = var_5.create()
    var_8 = len(var_6)
    assert var_8 == 2
    var_9 = len(var_7)
    assert var_9 == 2

def test_case_0():
    var_0 = 3
    var_1 = 42



