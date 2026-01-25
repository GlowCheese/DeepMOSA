####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 30
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'Jane'
    var_6 = {var_0: var_5}
    var_7 = module_0.apply_overwrites_to_context(var_4, var_6)
    var_8 = bool(var_4 == {'name': 'Jane', 'age': 30})
    assert var_8 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'John'
    var_2 = {var_0: var_1}
    var_3 = 'new_var'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_2, var_5)
    var_7 = bool(var_2 == {'name': 'John'})
    assert var_7 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'config'
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'key2'
    var_6 = 'value2'
    var_7 = {var_5: var_6}
    var_8 = {var_0: var_7}
    var_9 = module_0.apply_overwrites_to_context(var_4, var_8)
    var_10 = bool(var_4 == {'config': {'key1': 'value1', 'key2': 'value2'}})
    assert var_10 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'flavor'
    var_1 = 'vanilla'
    var_2 = 'chocolate'
    var_3 = 'strawberry'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = {var_0: var_2}
    var_7 = module_0.apply_overwrites_to_context(var_5, var_6)
    var_8 = bool(var_5 == {'flavor': ['chocolate', 'vanilla', 'strawberry']})
    assert var_8 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'flavor'
    var_1 = 'vanilla'
    var_2 = 'chocolate'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 'pistachio'
    var_6 = {var_0: var_5}
    var_7 = module_0.apply_overwrites_to_context(var_4, var_6)
    var_8 = bool(False)
    assert var_8 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'toppings'
    var_1 = 'pepperoni'
    var_2 = 'mushroom'
    var_3 = 'onion'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = [var_2, var_3]
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_5, var_7)
    var_9 = bool(var_5 == {'toppings': ['mushroom', 'onion']})
    assert var_9 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'toppings'
    var_1 = 'pepperoni'
    var_2 = 'mushroom'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 'pineapple'
    var_6 = [var_1, var_5]
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_4, var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = 'pineapple'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'use_feature'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = 'yes'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(var_2 == {'use_feature': True})
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'use_feature'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'false'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(var_2 == {'use_feature': False})
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'debug'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = 'true'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(var_2 == {'debug': True})
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'flag'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'maybe'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(False)
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'database'
    var_1 = 'host'
    var_2 = 'port'
    var_3 = 'localhost'
    var_4 = 5432
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 3306
    var_8 = {var_2: var_7}
    var_9 = {var_0: var_8}
    var_10 = module_0.apply_overwrites_to_context(var_6, var_9)
    var_11 = bool(var_6 == {'database': {'host': 'localhost', 'port': 3306}})
    assert var_11 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'settings'
    var_1 = 'options'
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'c'
    var_5 = [var_2, var_3, var_4]
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = 'x'
    var_9 = 'y'
    var_10 = [var_8, var_9]
    var_11 = {var_1: var_10}
    var_12 = {var_0: var_11}
    var_13 = module_0.apply_overwrites_to_context(var_7, var_12)
    var_14 = bool(var_7 == {'settings': {'options': ['x', 'y']}})
    assert var_14 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'city'
    var_3 = 'John'
    var_4 = 30
    var_5 = 'NYC'
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 'Jane'
    var_8 = 25
    var_9 = {var_0: var_7, var_1: var_8}
    var_10 = module_0.apply_overwrites_to_context(var_6, var_9)
    var_11 = bool(var_6 == {'name': 'Jane', 'age': 25, 'city': 'NYC'})
    assert var_11 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'John'
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = module_0.apply_overwrites_to_context(var_2, var_3)
    var_5 = bool(var_2 == {'name': 'John'})
    assert var_5 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'enabled'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = '  yes  '
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(var_2 == {'enabled': True})
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'flag'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = '1'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(var_2 == {'flag': True})
    assert var_6 is True



# Parsed testcases at query #2
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'existing'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'new_variable'
    var_4 = 'new_value'
    var_5 = {var_3: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_2, var_5)
    var_7 = bool(var_2 == {'existing': 'value'})
    assert var_7 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'nested'
    var_1 = 'existing'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'new_variable'
    var_6 = 'new_value'
    var_7 = {var_5: var_6}
    var_8 = True
    var_9 = module_0.apply_overwrites_to_context(var_4, var_7, in_dictionary_variable=var_8)
    var_10 = bool(var_4 == {'nested': {'existing': 'value', 'new_variable': 'new_value'}})
    assert var_10 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'variable'
    var_1 = 'old_value'
    var_2 = {var_0: var_1}
    var_3 = 'new_value'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(var_2 == {'variable': 'new_value'})
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'choice_var'
    var_1 = 'option1'
    var_2 = 'option2'
    var_3 = 'option3'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = {var_0: var_2}
    var_7 = module_0.apply_overwrites_to_context(var_5, var_6)
    var_8 = var_5['choice_var']
    var_9 = bool(var_5['choice_var'] == ['option2', 'option1', 'option3'])
    assert var_9 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'choice_var'
    var_1 = 'option1'
    var_2 = 'option2'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 'invalid_option'
    var_6 = {var_0: var_5}
    var_7 = module_0.apply_overwrites_to_context(var_4, var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'invalid_option'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'multichoice_var'
    var_1 = 'option1'
    var_2 = 'option2'
    var_3 = 'option3'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = [var_1, var_3]
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_5, var_7)
    var_9 = bool(var_5 == {'multichoice_var': ['option1', 'option3']})
    assert var_9 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'multichoice_var'
    var_1 = 'option1'
    var_2 = 'option2'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 'invalid_option'
    var_6 = [var_1, var_5]
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_4, var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = 'invalid_option'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'nested'
    var_1 = 'key1'
    var_2 = 'key2'
    var_3 = 'value1'
    var_4 = 'value2'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'new_value1'
    var_8 = {var_1: var_7}
    var_9 = {var_0: var_8}
    var_10 = module_0.apply_overwrites_to_context(var_6, var_9)
    var_11 = bool(var_6 == {'nested': {'key1': 'new_value1', 'key2': 'value2'}})
    assert var_11 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'bool_var'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'yes'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(var_2 == {'bool_var': True})
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'bool_var'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = 'no'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(var_2 == {'bool_var': False})
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'bool_var'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'true'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(var_2 == {'bool_var': True})
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'bool_var'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = 'false'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(var_2 == {'bool_var': False})
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'bool_var'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'invalid'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'invalid'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'nested'
    var_1 = 'list_var'
    var_2 = 'opt1'
    var_3 = 'opt2'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = {var_1: var_3}
    var_8 = {var_0: var_7}
    var_9 = module_0.apply_overwrites_to_context(var_6, var_8)
    var_10 = var_6['nested']['list_var']
    var_11 = bool(var_6['nested']['list_var'] == ['opt2', 'opt1'])
    assert var_11 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'nested'
    var_1 = 'list_var'
    var_2 = 'opt1'
    var_3 = 'opt2'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_2, var_3]
    var_8 = {var_1: var_7}
    var_9 = {var_0: var_8}
    var_10 = True
    var_11 = module_0.apply_overwrites_to_context(var_6, var_9, in_dictionary_variable=var_10)
    var_12 = var_6['nested']['list_var']
    var_13 = bool(var_6['nested']['list_var'] == ['opt1', 'opt2'])
    assert var_13 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'var1'
    var_1 = 'var2'
    var_2 = 'var3'
    var_3 = 'value1'
    var_4 = 'value2'
    var_5 = 'value3'
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 'new_value1'
    var_8 = 'new_value3'
    var_9 = {var_0: var_7, var_2: var_8}
    var_10 = module_0.apply_overwrites_to_context(var_6, var_9)
    var_11 = bool(var_6 == {'var1': 'new_value1', 'var2': 'value2', 'var3': 'new_value3'})
    assert var_11 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'bool_var'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = '  yes  '
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(var_2 == {'bool_var': True})
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'bool_var'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'YES'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(var_2 == {'bool_var': True})
    assert var_6 is True



# Parsed testcases at query #3
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 46 evaluates to False when context_value is not a dict or overwrite is not a dict.'
    var_1 = 'key'
    var_2 = 'nested'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'string_value'
    var_7 = {var_1: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_5, var_7)
    var_9 = var_5['key']
    assert var_9 == 'string_value'
    var_10 = {var_1: var_6}
    var_11 = {var_2: var_3}
    var_12 = {var_1: var_11}
    var_13 = module_0.apply_overwrites_to_context(var_10, var_12)
    var_14 = var_10['key']
    var_15 = bool(var_10['key'] == {'nested': 'value'})
    assert var_15 is True
    var_16 = 'original'
    var_17 = {var_1: var_16}
    var_18 = 'new'
    var_19 = {var_1: var_18}
    var_20 = module_0.apply_overwrites_to_context(var_17, var_19)
    var_21 = var_17['key']
    assert var_21 == 'new'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_render_and_create_dir_with_empty_dirname. Retrieved 3/9 statements.
# Partially parsed test_render_and_create_dir_creates_new_directory. Retrieved 3/8 statements.
# Partially parsed test_render_and_create_dir_with_template_rendering. Retrieved 5/10 statements.
# Partially parsed test_render_and_create_dir_existing_dir_no_overwrite. Retrieved 4/12 statements.
# Partially parsed test_render_and_create_dir_existing_dir_with_overwrite. Retrieved 4/11 statements.
# Partially parsed test_render_and_create_dir_nested_path. Retrieved 3/8 statements.
# Partially parsed test_render_and_create_dir_none_dirname. Retrieved 3/9 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = ''
    var_3 = bool(False)
    assert var_3 is True

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = 'test_dir'

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'project_name'
    var_2 = 'my_project'
    var_3 = {var_1: var_2}
    var_4 = '{{ project_name }}_dir'

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = 'existing_dir'
    var_3 = False
    var_4 = bool(False)
    assert var_4 is True

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = 'existing_dir'
    var_3 = True

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = 'parent/child/grandchild'

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = None
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_render_and_create_dir_with_empty_dirname. Retrieved 3/11 statements.
# Partially parsed test_render_and_create_dir_creates_new_directory. Retrieved 3/10 statements.
# Partially parsed test_render_and_create_dir_with_template_variable. Retrieved 5/12 statements.
# Partially parsed test_render_and_create_dir_existing_dir_without_overwrite. Retrieved 5/16 statements.
# Partially parsed test_render_and_create_dir_existing_dir_with_overwrite. Retrieved 4/14 statements.
# Partially parsed test_render_and_create_dir_nested_path. Retrieved 3/10 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = ''
    var_2 = {}
    var_3 = bool(False)
    assert var_3 is True

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'test_dir'
    var_2 = {}

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'project_name'
    var_2 = 'my_project'
    var_3 = {var_1: var_2}
    var_4 = '{{ project_name }}'

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'existing_dir'
    var_2 = 'existing_dir'
    var_3 = {}
    var_4 = False
    var_5 = bool(False)
    assert var_5 is True

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'existing_dir'
    var_2 = {}
    var_3 = True

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'parent/child/grandchild'
    var_2 = {}



# Parsed testcases at query #6
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test that line 57 predicate evaluates to False when boolean conversion succeeds.'
    var_1 = 'debug'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = 'false'
    var_5 = {var_1: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)
    var_7 = var_3['debug']
    assert var_7 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = "Test that line 57 predicate evaluates to False with 'yes' input."
    var_1 = 'enabled'
    var_2 = False
    var_3 = {var_1: var_2}
    var_4 = 'yes'
    var_5 = {var_1: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)
    var_7 = var_3['enabled']
    assert var_7 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = "Test that line 57 predicate evaluates to False with 'no' input."
    var_1 = 'enabled'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = 'no'
    var_5 = {var_1: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)
    var_7 = var_3['enabled']
    assert var_7 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = "Test that line 57 predicate evaluates to False with '1' input."
    var_1 = 'flag'
    var_2 = False
    var_3 = {var_1: var_2}
    var_4 = '1'
    var_5 = {var_1: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)
    var_7 = var_3['flag']
    assert var_7 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = "Test that line 57 predicate evaluates to False with '0' input."
    var_1 = 'flag'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = '0'
    var_5 = {var_1: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)
    var_7 = var_3['flag']
    assert var_7 is False



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_generate_context_with_valid_json_file. Retrieved 3/7 statements.
# Partially parsed test_generate_context_with_invalid_json_file. Retrieved 4/8 statements.
# Partially parsed test_generate_context_with_default_context. Retrieved 6/10 statements.
# Partially parsed test_generate_context_with_extra_context. Retrieved 6/10 statements.
# Partially parsed test_generate_context_with_choice_variable. Retrieved 6/10 statements.
# Partially parsed test_generate_context_with_invalid_choice. Retrieved 7/11 statements.
# Partially parsed test_generate_context_with_multichoice_variable. Retrieved 8/12 statements.
# Partially parsed test_generate_context_with_invalid_multichoice. Retrieved 9/13 statements.
# Partially parsed test_generate_context_with_boolean_variable. Retrieved 6/10 statements.
# Partially parsed test_generate_context_with_boolean_false. Retrieved 6/10 statements.
# Partially parsed test_generate_context_with_invalid_boolean. Retrieved 7/11 statements.
# Partially parsed test_generate_context_with_nested_dict. Retrieved 8/12 statements.
# Partially parsed test_generate_context_with_simple_overwrite. Retrieved 8/12 statements.
# Partially parsed test_generate_context_with_empty_json. Retrieved 3/7 statements.
# Partially parsed test_generate_context_preserves_original_with_default_context_error. Retrieved 3/5 statements.


def test_case_0():
    var_0 = 'Test generate_context with a valid JSON file.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "my_project", "author": "John"}'
    var_3 = 'cookiecutter'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_context with an invalid JSON file.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"invalid": json}'
    var_3 = module_0.generate_context(var_0)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'JSON decoding error'

def test_case_0():
    var_0 = 'Test generate_context with default_context parameter.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "my_project", "version": "1.0"}'
    var_3 = 'project_name'
    var_4 = 'default_project'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test generate_context with extra_context parameter.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "my_project", "version": "1.0"}'
    var_3 = 'project_name'
    var_4 = 'extra_project'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test generate_context with choice variable.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"license": ["MIT", "Apache", "GPL"]}'
    var_3 = 'license'
    var_4 = 'Apache'
    var_5 = {var_3: var_4}

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_context with invalid choice variable.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"license": ["MIT", "Apache"]}'
    var_3 = 'license'
    var_4 = 'BSD'
    var_5 = {var_3: var_4}
    var_6 = module_0.generate_context(var_0, extra_context=var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'BSD'

def test_case_0():
    var_0 = 'Test generate_context with multichoice variable.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"features": ["feature1", "feature2", "feature3"]}'
    var_3 = 'features'
    var_4 = 'feature2'
    var_5 = 'feature3'
    var_6 = [var_4, var_5]
    var_7 = {var_3: var_6}

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_context with invalid multichoice variable.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"features": ["feature1", "feature2"]}'
    var_3 = 'features'
    var_4 = 'feature1'
    var_5 = 'invalid_feature'
    var_6 = [var_4, var_5]
    var_7 = {var_3: var_6}
    var_8 = module_0.generate_context(var_0, extra_context=var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = 'invalid_feature'

def test_case_0():
    var_0 = 'Test generate_context with boolean variable.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"use_docker": true}'
    var_3 = 'use_docker'
    var_4 = 'yes'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test generate_context with boolean false variable.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"use_docker": true}'
    var_3 = 'use_docker'
    var_4 = 'no'
    var_5 = {var_3: var_4}

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_context with invalid boolean variable.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"use_docker": true}'
    var_3 = 'use_docker'
    var_4 = 'maybe'
    var_5 = {var_3: var_4}
    var_6 = module_0.generate_context(var_0, extra_context=var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'could not be converted to a boolean'

def test_case_0():
    var_0 = 'Test generate_context with nested dictionary.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"config": {"database": "postgres", "port": 5432}}'
    var_3 = 'config'
    var_4 = 'database'
    var_5 = 'mysql'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}

def test_case_0():
    var_0 = 'Test generate_context with simple string overwrite.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "original", "author": "original_author"}'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'new_project'
    var_6 = 'new_author'
    var_7 = {var_3: var_5, var_4: var_6}

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_context with non-existent file.'
    var_1 = '/non/existent/path/cookiecutter.json'
    var_2 = module_0.generate_context(var_1)
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'Test generate_context with empty JSON object.'
    var_1 = 'cookiecutter.json'
    var_2 = '{}'
    var_3 = 'cookiecutter'

def test_case_0():
    var_0 = "Test generate_context with invalid default context doesn't break."
    var_1 = 'cookiecutter.json'
    var_2 = '{"license": ["MIT", "Apache"]}'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_render_and_create_dir_predicate_line_24_false. Retrieved 4/11 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = {}
    var_2 = module_0.Environment()
    var_3 = True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_calls_hooks_run_hook_from_repo_dir. Retrieved 14/21 statements.
# Partially parsed test_run_hook_from_repo_dir_deprecation_warning. Retrieved 9/14 statements.
# Partially parsed test_run_hook_from_repo_dir_passes_all_arguments. Retrieved 14/20 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test that _run_hook_from_repo_dir calls hooks.run_hook_from_repo_dir.'
    var_1 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_2 = 'cookiecutter.generate.warnings.warn'
    var_3 = '/path/to/repo'
    var_4 = 'post_gen_project'
    var_5 = '/path/to/project'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'test'
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = True
    var_12 = module_0._run_hook_from_repo_dir(var_3, var_4, var_5, var_10, var_11)
    var_13 = 0
    var_14 = 'deprecated'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test that _run_hook_from_repo_dir issues a deprecation warning.'
    var_1 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_2 = 'cookiecutter.generate.warnings.warn'
    var_3 = 'repo'
    var_4 = 'hook'
    var_5 = 'project'
    var_6 = {}
    var_7 = False
    var_8 = module_0._run_hook_from_repo_dir(var_3, var_4, var_5, var_6, var_7)
    var_9 = '_run_hook_from_repo_dir'
    var_10 = 'cookiecutter.hooks.run_hook_from_repo_dir'

def test_case_0():
    var_0 = 'Test that _run_hook_from_repo_dir passes all arguments correctly.'
    var_1 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_2 = 'cookiecutter.generate.warnings.warn'
    var_3 = '/template/repo'
    var_4 = 'pre_gen_project'
    var_5 = '/output/project'
    var_6 = [var_5]
    var_7 = 'cookiecutter'
    var_8 = 'name'
    var_9 = 'version'
    var_10 = 'myproject'
    var_11 = '1.0'
    var_12 = {var_8: var_10, var_9: var_11}
    var_13 = {var_7: var_12}
    var_14 = False



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_generate_context_json_decoding_error. Retrieved 2/14 statements.


def test_case_0():
    var_0 = 'Test that ValueError is caught at line 20 and ContextDecodingException is raised.'
    var_1 = '{invalid json content}'
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'JSON decoding error'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_render_and_create_dir_predicate_line_24_true. Retrieved 6/14 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 24 evaluates to True when directory exists.'
    var_1 = 'existing_dir'
    var_2 = True
    var_3 = {}
    var_4 = module_0.Environment()
    var_5 = 'existing_dir'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_generate_context_with_valid_json_file. Retrieved 2/6 statements.
# Partially parsed test_generate_context_with_default_context. Retrieved 5/9 statements.
# Partially parsed test_generate_context_with_extra_context. Retrieved 5/9 statements.
# Partially parsed test_generate_context_with_invalid_json. Retrieved 3/7 statements.
# Partially parsed test_generate_context_with_boolean_variable_and_string_override. Retrieved 5/9 statements.
# Partially parsed test_generate_context_with_choice_variable. Retrieved 5/9 statements.
# Partially parsed test_generate_context_with_multichoice_variable. Retrieved 7/11 statements.
# Partially parsed test_generate_context_with_nested_dict. Retrieved 4/10 statements.
# Partially parsed test_generate_context_with_invalid_choice_override. Retrieved 6/10 statements.
# Partially parsed test_generate_context_with_invalid_boolean_string. Retrieved 6/10 statements.
# Partially parsed test_generate_context_with_custom_context_file_name. Retrieved 2/6 statements.
# Partially parsed test_generate_context_with_invalid_default_context_warns. Retrieved 5/9 statements.
# Partially parsed test_generate_context_preserves_order. Retrieved 3/10 statements.


def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = '{"project_name": "my_project", "author": "John"}'
    var_2 = 'cookiecutter'

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = '{"project_name": "my_project", "author": "John"}'
    var_2 = 'project_name'
    var_3 = 'default_project'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = '{"project_name": "my_project", "author": "John"}'
    var_2 = 'author'
    var_3 = 'Jane'
    var_4 = {var_2: var_3}

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = '{"project_name": invalid json}'
    var_2 = module_0.generate_context(var_0)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'JSON decoding error'

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = '{"use_feature": true}'
    var_2 = 'use_feature'
    var_3 = 'false'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = '{"license": ["MIT", "Apache", "GPL"]}'
    var_2 = 'license'
    var_3 = 'Apache'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = '{"features": ["feature1", "feature2", "feature3"]}'
    var_2 = 'features'
    var_3 = 'feature2'
    var_4 = 'feature3'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = '{"settings": {"debug": true, "timeout": 30}}'
    var_2 = 'settings'
    var_3 = 'debug'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '/nonexistent/path/cookiecutter.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = bool(False)
    assert var_2 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = '{"license": ["MIT", "Apache"]}'
    var_2 = 'license'
    var_3 = 'GPL'
    var_4 = {var_2: var_3}
    var_5 = module_0.generate_context(var_0, extra_context=var_4)
    var_6 = bool(False)
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = '{"use_feature": true}'
    var_2 = 'use_feature'
    var_3 = 'invalid'
    var_4 = {var_2: var_3}
    var_5 = module_0.generate_context(var_0, extra_context=var_4)
    var_6 = bool(False)
    assert var_6 is True

def test_case_0():
    var_0 = 'custom.json'
    var_1 = '{"project_name": "my_project"}'
    var_2 = 'custom'

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = '{"license": ["MIT", "Apache"]}'
    var_2 = 'license'
    var_3 = 'GPL'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = '{"z_field": "z", "a_field": "a", "m_field": "m"}'
    var_2 = 'cookiecutter'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_render_and_create_dir_predicate_line_24_false. Retrieved 4/12 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = {}
    var_2 = module_0.Environment()
    var_3 = True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_generate_context_predicate_line_38_false. Retrieved 8/18 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 38 (if default_context:) evaluates to False.'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'author'
    var_4 = 'test_project'
    var_5 = 'test_author'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = None
    var_8 = 'cookiecutter'



# Parsed testcases at query #15
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test that boolean conversion succeeds with valid yes/no input.'
    var_1 = 'feature_enabled'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = 'no'
    var_5 = {var_1: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)
    var_7 = var_3['feature_enabled']
    assert var_7 is False



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_generate_context_basic. Retrieved 3/7 statements.
# Partially parsed test_generate_context_with_default_context. Retrieved 6/10 statements.
# Partially parsed test_generate_context_with_extra_context. Retrieved 6/10 statements.
# Partially parsed test_generate_context_with_choice_variable. Retrieved 6/10 statements.
# Partially parsed test_generate_context_with_boolean_variable. Retrieved 6/10 statements.
# Partially parsed test_generate_context_with_nested_dict. Retrieved 8/12 statements.
# Partially parsed test_generate_context_invalid_json. Retrieved 4/8 statements.
# Partially parsed test_generate_context_with_multichoice_variable. Retrieved 9/15 statements.
# Partially parsed test_generate_context_invalid_choice_raises_error. Retrieved 7/11 statements.
# Partially parsed test_generate_context_custom_filename. Retrieved 3/7 statements.
# Partially parsed test_generate_context_with_all_parameters. Retrieved 11/15 statements.


def test_case_0():
    var_0 = 'Test generate_context with a basic JSON file.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "my_project", "author": "John"}'
    var_3 = 'cookiecutter'

def test_case_0():
    var_0 = 'Test generate_context with default_context parameter.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "my_project", "version": "1.0"}'
    var_3 = 'project_name'
    var_4 = 'overridden_project'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test generate_context with extra_context parameter.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "my_project"}'
    var_3 = 'project_name'
    var_4 = 'extra_project'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test generate_context with choice variable (list).'
    var_1 = 'cookiecutter.json'
    var_2 = '{"license": ["MIT", "Apache", "GPL"]}'
    var_3 = 'license'
    var_4 = 'Apache'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test generate_context with boolean variable.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"use_docker": true}'
    var_3 = 'use_docker'
    var_4 = 'false'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test generate_context with nested dictionary.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"config": {"debug": false, "timeout": 30}}'
    var_3 = 'config'
    var_4 = 'debug'
    var_5 = 'true'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_context with invalid JSON file.'
    var_1 = 'cookiecutter.json'
    var_2 = '{invalid json}'
    var_3 = module_0.generate_context(var_0)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'JSON decoding error'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_context with non-existent file.'
    var_1 = '/nonexistent/path/cookiecutter.json'
    var_2 = module_0.generate_context(var_1)
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'Test generate_context with multichoice variable.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"features": ["feature1", "feature2", "feature3"]}'
    var_3 = 'features'
    var_4 = 'feature2'
    var_5 = 'feature3'
    var_6 = [var_4, var_5]
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_context with invalid choice raises ValueError.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"license": ["MIT", "Apache"]}'
    var_3 = 'license'
    var_4 = 'InvalidLicense'
    var_5 = {var_3: var_4}
    var_6 = module_0.generate_context(var_0, extra_context=var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'provided for choice variable'

def test_case_0():
    var_0 = 'Test generate_context with custom context file name.'
    var_1 = 'custom_context.json'
    var_2 = '{"name": "test"}'
    var_3 = 'custom_context'

def test_case_0():
    var_0 = 'Test generate_context with all parameters.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project": "default", "version": "1.0", "active": true}'
    var_3 = 'project'
    var_4 = 'from_default'
    var_5 = {var_3: var_4}
    var_6 = 'version'
    var_7 = 'active'
    var_8 = '2.0'
    var_9 = 'false'
    var_10 = {var_6: var_8, var_7: var_9}



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_generate_files_basic. Retrieved 13/35 statements.
# Partially parsed test_generate_files_with_empty_dirname. Retrieved 7/14 statements.
# Partially parsed test_generate_files_directory_exists_no_overwrite. Retrieved 9/18 statements.
# Partially parsed test_generate_files_creates_directory. Retrieved 7/13 statements.
# Partially parsed test_generate_files_with_context_variables. Retrieved 9/15 statements.
# Partially parsed test_generate_files_overwrite_existing. Retrieved 10/20 statements.


def test_case_0():
    var_0 = 'Test generate_files with basic template structure.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'README.md'
    var_4 = '# {{cookiecutter.project_name}}'
    var_5 = 'output'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'my_project'
    var_9 = {var_7: var_8}
    var_10 = (var_6, var_9)
    var_11 = [var_10]
    var_12 = [var_11]
    var_13 = False
    var_14 = bool(var_5)
    assert var_14 is True

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test generate_files raises error on empty directory name.'
    var_1 = module_0.Environment()
    var_2 = 'output'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = ''
    var_7 = bool(False)
    assert var_7 is True

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test generate_files raises error when output directory exists and overwrite is False.'
    var_1 = module_0.Environment()
    var_2 = 'output'
    var_3 = 'existing_project'
    var_4 = 'cookiecutter'
    var_5 = {}
    var_6 = {var_4: var_5}
    var_7 = 'existing_project'
    var_8 = False
    var_9 = bool(False)
    assert var_9 is True

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test generate_files successfully creates output directory.'
    var_1 = module_0.Environment()
    var_2 = 'output'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'new_project'

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test generate_files renders context variables in directory names.'
    var_1 = module_0.Environment()
    var_2 = 'output'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'awesome_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = '{{cookiecutter.project_name}}'

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test generate_files can overwrite existing directory.'
    var_1 = module_0.Environment()
    var_2 = 'output'
    var_3 = 'my_project'
    var_4 = 'test.txt'
    var_5 = 'old content'
    var_6 = 'cookiecutter'
    var_7 = {}
    var_8 = {var_6: var_7}
    var_9 = True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test is_copy_only_path returns True when path matches pattern.'
    var_1 = 'cookiecutter'
    var_2 = '_copy_without_render'
    var_3 = '*.pyc'
    var_4 = 'node_modules/*'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = 'test.pyc'
    var_9 = module_0.is_copy_only_path(var_8, var_7)
    assert var_9 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = "Test is_copy_only_path returns False when path doesn't match pattern."
    var_1 = 'cookiecutter'
    var_2 = '_copy_without_render'
    var_3 = '*.pyc'
    var_4 = 'node_modules/*'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = 'test.py'
    var_9 = module_0.is_copy_only_path(var_8, var_7)
    assert var_9 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test is_copy_only_path returns False when _copy_without_render not configured.'
    var_1 = 'cookiecutter'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = 'test.py'
    var_5 = module_0.is_copy_only_path(var_4, var_3)
    assert var_5 is False



# Parsed testcases at query #18
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = '*.pyc'
    var_3 = '*.so'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'test.pyc'
    var_8 = module_0.is_copy_only_path(var_7, var_6)
    assert var_8 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = '*.pyc'
    var_3 = '*.so'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'test.py'
    var_8 = module_0.is_copy_only_path(var_7, var_6)
    assert var_8 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = 'node_modules/*'
    var_3 = 'venv/*'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'node_modules/package.json'
    var_8 = module_0.is_copy_only_path(var_7, var_6)
    assert var_8 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = 'test.pyc'
    var_4 = module_0.is_copy_only_path(var_3, var_2)
    assert var_4 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'test.pyc'
    var_2 = module_0.is_copy_only_path(var_1, var_0)
    assert var_2 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = []
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test.pyc'
    var_6 = module_0.is_copy_only_path(var_5, var_4)
    assert var_6 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = '*.pyc'
    var_3 = '*.so'
    var_4 = '*.bin'
    var_5 = [var_2, var_3, var_4]
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = 'test.pyc'
    var_9 = module_0.is_copy_only_path(var_8, var_7)
    assert var_9 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = '*.pyc'
    var_3 = '*.so'
    var_4 = '*.bin'
    var_5 = [var_2, var_3, var_4]
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = 'test.bin'
    var_9 = module_0.is_copy_only_path(var_8, var_7)
    assert var_9 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = 'test?.txt'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}
    var_6 = 'test1.txt'
    var_7 = module_0.is_copy_only_path(var_6, var_5)
    assert var_7 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = 'test[0-9].txt'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}
    var_6 = 'test5.txt'
    var_7 = module_0.is_copy_only_path(var_6, var_5)
    assert var_7 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_generate_context_opens_file_with_utf8_encoding. Retrieved 9/17 statements.


import json as module_0

def test_case_0():
    var_0 = 'Test that generate_context opens the context file with utf-8 encoding.'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'author'
    var_4 = 'test_project'
    var_5 = 'Test Author'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = {}
    var_8 = module_0.dumps(var_6, **var_7)
    var_9 = 'utf-8'
    var_10 = 'cookiecutter'



# Parsed testcases at query #20
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test that boolean conversion succeeds when valid yes/no string is provided.'
    var_1 = 'flag'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = 'no'
    var_5 = {var_1: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)
    var_7 = var_3['flag']
    assert var_7 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test that line 57 predicate (except clause) evaluates to False for valid input.'
    var_1 = 'flag'
    var_2 = False
    var_3 = {var_1: var_2}
    var_4 = 'yes'
    var_5 = {var_1: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)
    var_7 = var_3['flag']
    assert var_7 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_predicate_line_62_evaluates_to_false. Retrieved 17/40 statements.


def test_case_0():
    var_0 = "Test that the predicate at line 62 (for root, dirs, files in os.walk('.')) evaluates to False when os.walk returns empty."
    var_1 = 'repo'
    var_2 = var_0 / var_1
    var_3 = '{{cookiecutter.project_name}}'
    var_4 = var_2 / var_3
    var_5 = 'output'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'test_project'
    var_9 = {var_7: var_8}
    var_10 = (var_6, var_9)
    var_11 = [var_10]
    var_12 = [var_11]
    var_13 = []
    var_14 = 'test_project'
    var_15 = str(var_1)
    var_16 = True
    var_17 = False
    var_18 = bool(var_13 == [])
    assert var_18 is True



# Parsed testcases at query #22
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'existing_var'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'new_var'
    var_4 = 'new_value'
    var_5 = {var_3: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_2, var_5)
    var_7 = bool(var_2 == {'existing_var': 'value'})
    assert var_7 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'nested'
    var_1 = 'existing'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'new_var'
    var_6 = 'new_value'
    var_7 = {var_5: var_6}
    var_8 = {var_0: var_7}
    var_9 = True
    var_10 = module_0.apply_overwrites_to_context(var_4, var_8, in_dictionary_variable=var_9)
    var_11 = bool(var_4 == {'nested': {'existing': 'value', 'new_var': 'new_value'}})
    assert var_11 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'choices'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = [var_2, var_3]
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_5, var_7)
    var_9 = bool(var_5 == {'choices': ['b', 'c']})
    assert var_9 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'choices'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 'd'
    var_7 = 'e'
    var_8 = [var_6, var_7]
    var_9 = {var_0: var_8}
    var_10 = module_0.apply_overwrites_to_context(var_5, var_9)
    var_11 = bool(False)
    assert var_11 is True
    var_12 = 'multi-choice variable'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'choice'
    var_1 = 'default'
    var_2 = 'option1'
    var_3 = 'option2'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = {var_0: var_2}
    var_7 = module_0.apply_overwrites_to_context(var_5, var_6)
    var_8 = bool(var_5 == {'choice': ['option1', 'default', 'option2']})
    assert var_8 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'choice'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 'invalid'
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_5, var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = 'choice variable'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'settings'
    var_1 = 'debug'
    var_2 = 'port'
    var_3 = True
    var_4 = 8000
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = False
    var_8 = {var_1: var_7}
    var_9 = {var_0: var_8}
    var_10 = module_0.apply_overwrites_to_context(var_6, var_9)
    var_11 = bool(var_6 == {'settings': {'debug': False, 'port': 8000}})
    assert var_11 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'use_feature'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = 'yes'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(var_2 == {'use_feature': True})
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'use_feature'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'no'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(var_2 == {'use_feature': False})
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'enabled'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = 'true'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(var_2 == {'enabled': True})
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'enabled'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'false'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(var_2 == {'enabled': False})
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'enabled'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'invalid'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'could not be converted to a boolean'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'version'
    var_2 = 'old_name'
    var_3 = '1.0'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'new_name'
    var_6 = {var_0: var_5}
    var_7 = module_0.apply_overwrites_to_context(var_4, var_6)
    var_8 = bool(var_4 == {'name': 'new_name', 'version': '1.0'})
    assert var_8 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'config'
    var_1 = 'items'
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'c'
    var_5 = [var_2, var_3, var_4]
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = 'x'
    var_9 = 'y'
    var_10 = [var_8, var_9]
    var_11 = {var_1: var_10}
    var_12 = {var_0: var_11}
    var_13 = True
    var_14 = module_0.apply_overwrites_to_context(var_7, var_12, in_dictionary_variable=var_13)
    var_15 = bool(var_7 == {'config': {'items': ['x', 'y']}})
    assert var_15 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'var1'
    var_1 = 'var2'
    var_2 = 'var3'
    var_3 = 'val1'
    var_4 = 'val2'
    var_5 = 'val3'
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 'new_val1'
    var_8 = 'new_val3'
    var_9 = {var_0: var_7, var_2: var_8}
    var_10 = module_0.apply_overwrites_to_context(var_6, var_9)
    var_11 = bool(var_6 == {'var1': 'new_val1', 'var2': 'val2', 'var3': 'new_val3'})
    assert var_11 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_render_and_create_dir_raises_empty_dir_name_exception. Retrieved 4/11 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = {}
    var_1 = '.'
    var_2 = [var_1]
    var_3 = module_0.Environment()
    var_4 = ''
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_render_and_create_dir_empty_dirname. Retrieved 3/11 statements.
# Partially parsed test_render_and_create_dir_creates_new_directory. Retrieved 3/10 statements.
# Partially parsed test_render_and_create_dir_with_template_rendering. Retrieved 5/12 statements.
# Partially parsed test_render_and_create_dir_existing_directory_no_overwrite. Retrieved 3/14 statements.
# Partially parsed test_render_and_create_dir_existing_directory_with_overwrite. Retrieved 4/14 statements.
# Partially parsed test_render_and_create_dir_nested_path. Retrieved 3/10 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = ''
    var_3 = bool(False)
    assert var_3 is True

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = 'test_dir'

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'project_name'
    var_2 = 'myproject'
    var_3 = {var_1: var_2}
    var_4 = '{{ project_name }}_dir'

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = 'existing_dir'
    var_3 = bool(False)
    assert var_3 is True

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = 'existing_dir'
    var_3 = True

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = 'nested/path/to/dir'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_render_and_create_dir_raises_on_empty_dirname. Retrieved 5/12 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test that render_and_create_dir raises EmptyDirNameException when dirname is empty string.'
    var_1 = {}
    var_2 = '/tmp'
    var_3 = [var_2]
    var_4 = module_0.Environment()
    var_5 = ''
    var_6 = bool(False)
    assert var_6 is True
    var_7 = bool(True)
    assert var_7 is True



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_delete_project_on_failure_true_when_output_directory_created_and_keep_project_false. Retrieved 2/4 statements.
# Partially parsed test_delete_project_on_failure_false_when_output_directory_not_created. Retrieved 2/4 statements.
# Partially parsed test_delete_project_on_failure_false_when_keep_project_true. Retrieved 2/4 statements.
# Partially parsed test_delete_project_on_failure_false_when_both_conditions_fail. Retrieved 2/4 statements.


def test_case_0():
    var_0 = True
    var_1 = False

def test_case_0():
    var_0 = False
    var_1 = False

def test_case_0():
    var_0 = True
    var_1 = True

def test_case_0():
    var_0 = False
    var_1 = True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_generate_files_basic. Retrieved 16/37 statements.
# Partially parsed test_generate_files_with_overwrite. Retrieved 17/34 statements.
# Partially parsed test_generate_files_empty_context. Retrieved 10/24 statements.
# Partially parsed test_generate_files_skip_if_file_exists. Retrieved 18/39 statements.
# Partially parsed test_generate_files_with_hooks. Retrieved 15/31 statements.
# Partially parsed test_generate_files_output_dir_created. Retrieved 16/33 statements.


def test_case_0():
    var_0 = 'Test basic file generation with a simple template.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'README.md'
    var_4 = '# {{cookiecutter.project_name}}'
    var_5 = 'cookiecutter'
    var_6 = 'project_name'
    var_7 = 'my_project'
    var_8 = {var_6: var_7}
    var_9 = (var_5, var_8)
    var_10 = [var_9]
    var_11 = [var_10]
    var_12 = 'output'
    var_13 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_14 = None
    var_15 = lambda *args, **kwargs: var_14
    var_16 = False

def test_case_0():
    var_0 = 'Test file generation with overwrite_if_exists=True.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.name}}'
    var_3 = 'file.txt'
    var_4 = 'content'
    var_5 = 'output'
    var_6 = 'cookiecutter'
    var_7 = 'name'
    var_8 = 'project'
    var_9 = {var_7: var_8}
    var_10 = (var_6, var_9)
    var_11 = [var_10]
    var_12 = [var_11]
    var_13 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_14 = None
    var_15 = lambda *args, **kwargs: var_14
    var_16 = False
    var_17 = True

def test_case_0():
    var_0 = 'Test file generation with no context provided.'
    var_1 = 'repo'
    var_2 = 'simple_project'
    var_3 = 'file.txt'
    var_4 = 'static content'
    var_5 = 'output'
    var_6 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_7 = None
    var_8 = lambda *args, **kwargs: var_7
    var_9 = False

def test_case_0():
    var_0 = 'Test file generation with skip_if_file_exists=True.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.proj}}'
    var_3 = 'existing.txt'
    var_4 = 'new content'
    var_5 = 'output'
    var_6 = 'cookiecutter'
    var_7 = 'proj'
    var_8 = 'myproj'
    var_9 = {var_7: var_8}
    var_10 = (var_6, var_9)
    var_11 = [var_10]
    var_12 = [var_11]
    var_13 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_14 = None
    var_15 = lambda *args, **kwargs: var_14
    var_16 = False
    var_17 = 'original content'
    var_18 = True

def test_case_0():
    var_0 = 'Test file generation with hooks acceptance.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.name}}'
    var_3 = 'file.txt'
    var_4 = 'content'
    var_5 = 'output'
    var_6 = 'cookiecutter'
    var_7 = 'name'
    var_8 = 'proj'
    var_9 = {var_7: var_8}
    var_10 = (var_6, var_9)
    var_11 = [var_10]
    var_12 = [var_11]
    var_13 = []
    var_14 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_15 = True
    var_16 = 'pre_gen_project'
    var_17 = bool('pre_gen_project' in var_13)
    assert var_17 is True
    var_18 = 'post_gen_project'
    var_19 = bool('post_gen_project' in var_13)
    assert var_19 is True

def test_case_0():
    var_0 = "Test that output directory is created if it doesn't exist."
    var_1 = 'repo'
    var_2 = '{{cookiecutter.name}}'
    var_3 = 'file.txt'
    var_4 = 'test'
    var_5 = 'nonexistent'
    var_6 = 'output'
    var_7 = 'cookiecutter'
    var_8 = 'name'
    var_9 = {var_8: var_4}
    var_10 = (var_7, var_9)
    var_11 = [var_10]
    var_12 = [var_11]
    var_13 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_14 = None
    var_15 = lambda *args, **kwargs: var_14
    var_16 = False



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_render_and_create_dir_raises_empty_dirname_exception. Retrieved 3/10 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = ''
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'directory name is empty'



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_render_and_create_dir_predicate_line_24_true. Retrieved 7/14 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 24 evaluates to True when directory exists.'
    var_1 = 'existing_project'
    var_2 = True
    var_3 = {}
    var_4 = module_0.Environment()
    var_5 = 'existing_project'
    var_6 = True



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_generate_context_basic. Retrieved 3/7 statements.
# Partially parsed test_generate_context_with_default_context. Retrieved 6/10 statements.
# Partially parsed test_generate_context_with_extra_context. Retrieved 6/10 statements.
# Partially parsed test_generate_context_with_boolean_overwrite. Retrieved 6/10 statements.
# Partially parsed test_generate_context_with_choice_variable. Retrieved 6/10 statements.
# Partially parsed test_generate_context_with_dict_variable. Retrieved 8/12 statements.
# Partially parsed test_generate_context_invalid_json. Retrieved 4/8 statements.
# Partially parsed test_generate_context_with_multichoice_valid. Retrieved 9/15 statements.
# Partially parsed test_generate_context_with_invalid_choice. Retrieved 7/11 statements.
# Partially parsed test_generate_context_with_invalid_boolean_string. Retrieved 7/11 statements.
# Partially parsed test_generate_context_default_context_invalid_warning. Retrieved 6/10 statements.
# Partially parsed test_generate_context_both_default_and_extra_context. Retrieved 12/16 statements.


def test_case_0():
    var_0 = 'Test generate_context with a basic JSON file.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "my_project", "author": "John Doe"}'
    var_3 = 'cookiecutter'

def test_case_0():
    var_0 = 'Test generate_context with default_context parameter.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "my_project", "version": "1.0"}'
    var_3 = 'project_name'
    var_4 = 'default_project'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test generate_context with extra_context parameter.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "my_project", "version": "1.0"}'
    var_3 = 'project_name'
    var_4 = 'extra_project'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test generate_context with boolean variable and string overwrite.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"use_feature": true}'
    var_3 = 'use_feature'
    var_4 = 'false'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test generate_context with choice variable.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"license": ["MIT", "Apache", "GPL"]}'
    var_3 = 'license'
    var_4 = 'Apache'
    var_5 = {var_3: var_4}
    var_6 = 'MIT'

def test_case_0():
    var_0 = 'Test generate_context with nested dictionary.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"config": {"debug": false, "port": 8000}}'
    var_3 = 'config'
    var_4 = 'debug'
    var_5 = 'true'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_context with invalid JSON raises ContextDecodingException.'
    var_1 = 'cookiecutter.json'
    var_2 = '{invalid json}'
    var_3 = module_0.generate_context(var_0)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'JSON decoding error'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_context with non-existent file.'
    var_1 = '/nonexistent/path/cookiecutter.json'
    var_2 = module_0.generate_context(var_1)
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'Test generate_context with valid multichoice variable.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"features": ["feature1", "feature2", "feature3"]}'
    var_3 = 'features'
    var_4 = 'feature2'
    var_5 = 'feature3'
    var_6 = [var_4, var_5]
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_context with invalid choice raises ValueError.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"license": ["MIT", "Apache"]}'
    var_3 = 'license'
    var_4 = 'GPL'
    var_5 = {var_3: var_4}
    var_6 = module_0.generate_context(var_0, extra_context=var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'GPL'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_context with invalid boolean string raises ValueError.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"use_feature": true}'
    var_3 = 'use_feature'
    var_4 = 'maybe'
    var_5 = {var_3: var_4}
    var_6 = module_0.generate_context(var_0, extra_context=var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'could not be converted to a boolean'

def test_case_0():
    var_0 = 'Test generate_context with invalid default_context logs warning.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"license": ["MIT", "Apache"]}'
    var_3 = 'license'
    var_4 = 'InvalidLicense'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test generate_context with both default and extra context.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project": "proj", "version": "1.0", "author": "Unknown"}'
    var_3 = 'project'
    var_4 = 'author'
    var_5 = 'default_proj'
    var_6 = 'Default Author'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = 'version'
    var_9 = '2.0'
    var_10 = 'Extra Author'
    var_11 = {var_8: var_9, var_4: var_10}



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_generate_file_renders_text_file. Retrieved 17/34 statements.
# Partially parsed test_generate_file_copies_binary_file. Retrieved 12/26 statements.
# Partially parsed test_generate_file_skips_existing_file. Retrieved 11/22 statements.
# Partially parsed test_generate_file_handles_empty_filename. Retrieved 10/18 statements.
# Partially parsed test_generate_file_renders_filename. Retrieved 17/32 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project'
    var_1 = 'templates'
    var_2 = 'test_{{cookiecutter.name}}.txt'
    var_3 = 'Hello {{cookiecutter.name}}'
    var_4 = 'utf-8'
    var_5 = 'cookiecutter'
    var_6 = 'name'
    var_7 = '_new_lines'
    var_8 = 'world'
    var_9 = False
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = {var_5: var_10}
    var_12 = module_0.Environment()
    var_13 = 'os.getcwd'
    var_14 = 'builtins.open'
    var_15 = 'shutil.copymode'
    var_16 = 'generate_file.is_binary'

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project'
    var_1 = 'templates'
    var_2 = 'binary_file.bin'
    var_3 = b'\x89PNG\r\n\x1a\n'
    var_4 = 'cookiecutter'
    var_5 = {}
    var_6 = {var_4: var_5}
    var_7 = module_0.Environment()
    var_8 = 'generate_file.is_binary'
    var_9 = True
    var_10 = 'shutil.copyfile'
    var_11 = 'shutil.copymode'

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project'
    var_1 = 'existing.txt'
    var_2 = 'w'
    var_3 = 'existing.txt'
    var_4 = 'cookiecutter'
    var_5 = {}
    var_6 = {var_4: var_5}
    var_7 = module_0.Environment()
    var_8 = 'generate_file.is_binary'
    var_9 = False
    var_10 = True

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project'
    var_1 = '{{cookiecutter.name}}'
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = ''
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.Environment()
    var_8 = 'os.path.isdir'
    var_9 = True

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project'
    var_1 = 'templates'
    var_2 = '{{cookiecutter.filename}}.txt'
    var_3 = 'content'
    var_4 = 'utf-8'
    var_5 = 'cookiecutter'
    var_6 = 'filename'
    var_7 = '_new_lines'
    var_8 = 'output'
    var_9 = '\n'
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = {var_5: var_10}
    var_12 = module_0.Environment()
    var_13 = 'generate_file.is_binary'
    var_14 = False
    var_15 = 'shutil.copymode'
    var_16 = 'builtins.open'



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_render_and_create_dir_with_empty_dirname. Retrieved 4/10 statements.
# Partially parsed test_render_and_create_dir_creates_new_directory. Retrieved 4/11 statements.
# Partially parsed test_render_and_create_dir_with_template_rendering. Retrieved 7/14 statements.
# Partially parsed test_render_and_create_dir_existing_dir_without_overwrite. Retrieved 6/15 statements.
# Partially parsed test_render_and_create_dir_existing_dir_with_overwrite. Retrieved 5/14 statements.
# Partially parsed test_render_and_create_dir_nested_directory. Retrieved 4/11 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test that EmptyDirNameException is raised when dirname is empty.'
    var_1 = module_0.Environment()
    var_2 = {}
    var_3 = ''
    var_4 = bool(False)
    assert var_4 is True

import jinja2.environment as module_0

def test_case_0():
    var_0 = "Test that a new directory is created when it doesn't exist."
    var_1 = module_0.Environment()
    var_2 = {}
    var_3 = 'test_dir'

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test that directory name is rendered from template.'
    var_1 = module_0.Environment()
    var_2 = 'project_name'
    var_3 = 'my_project'
    var_4 = {var_2: var_3}
    var_5 = '{{ project_name }}_dir'
    var_6 = 'my_project_dir'

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test that OutputDirExistsException is raised when directory exists and overwrite is False.'
    var_1 = module_0.Environment()
    var_2 = {}
    var_3 = 'existing_dir'
    var_4 = True
    var_5 = False
    var_6 = bool(False)
    assert var_6 is True

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test that existing directory is allowed when overwrite_if_exists is True.'
    var_1 = module_0.Environment()
    var_2 = {}
    var_3 = 'existing_dir'
    var_4 = True

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test that nested directories are created correctly.'
    var_1 = module_0.Environment()
    var_2 = {}
    var_3 = 'parent/child/grandchild'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_generate_context_opens_file_with_utf8_encoding. Retrieved 9/17 statements.


import json as module_0

def test_case_0():
    var_0 = 'Test that generate_context opens the context file with utf-8 encoding.'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'author'
    var_4 = 'test_project'
    var_5 = 'Test Author'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = {}
    var_8 = module_0.dumps(var_6, **var_7)
    var_9 = 'utf-8'
    var_10 = 'cookiecutter'



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_render_and_create_dir_predicate_line_24_false. Retrieved 5/12 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 24 evaluates to False when directory exists.'
    var_1 = 'test_dir'
    var_2 = {}
    var_3 = module_0.Environment()
    var_4 = True



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_generate_context_applies_default_context_when_provided. Retrieved 11/17 statements.


import json as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = 'Test that line 38 predicate evaluates to True when default_context is provided.'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'author'
    var_4 = 'my_project'
    var_5 = 'John Doe'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = {}
    var_8 = module_0.dumps(var_6, **var_7)
    var_9 = 'default_project'
    var_10 = {var_2: var_9}
    var_11 = module_1.generate_context(var_1, var_10)
    var_12 = var_11['cookiecutter']['project_name']
    assert var_12 == 'default_project'
    var_13 = 'cookiecutter'
    var_14 = bool('cookiecutter' in var_11)
    assert var_14 is True



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_generate_files_with_valid_context. Retrieved 18/41 statements.
# Partially parsed test_generate_files_empty_context. Retrieved 8/20 statements.
# Partially parsed test_generate_files_overwrite_if_exists. Retrieved 19/38 statements.
# Partially parsed test_generate_files_skip_if_file_exists. Retrieved 21/38 statements.
# Partially parsed test_generate_files_with_subdirectories. Retrieved 19/44 statements.
# Partially parsed test_generate_files_with_hooks. Retrieved 18/37 statements.


def test_case_0():
    var_0 = 'Test generate_files with valid context and template structure.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'README.md'
    var_4 = '# {{cookiecutter.project_name}}\n'
    var_5 = 'cookiecutter'
    var_6 = 'project_name'
    var_7 = '_jinja2_env_vars'
    var_8 = 'my_project'
    var_9 = {}
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = (var_5, var_10)
    var_12 = [var_11]
    var_13 = [var_12]
    var_14 = 'output'
    var_15 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_16 = None
    var_17 = lambda *args, **kwargs: var_16
    var_18 = False

def test_case_0():
    var_0 = 'Test generate_files with None context.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project}}'
    var_3 = 'output'
    var_4 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_5 = None
    var_6 = lambda *args, **kwargs: var_5
    var_7 = False

def test_case_0():
    var_0 = 'Test generate_files with overwrite_if_exists=True.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'file.txt'
    var_4 = 'content'
    var_5 = 'output'
    var_6 = 'my_project'
    var_7 = 'cookiecutter'
    var_8 = 'project_name'
    var_9 = '_jinja2_env_vars'
    var_10 = {}
    var_11 = {var_8: var_6, var_9: var_10}
    var_12 = (var_7, var_11)
    var_13 = [var_12]
    var_14 = [var_13]
    var_15 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_16 = None
    var_17 = lambda *args, **kwargs: var_16
    var_18 = True
    var_19 = False

def test_case_0():
    var_0 = 'Test generate_files with skip_if_file_exists=True.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'existing.txt'
    var_4 = '{{cookiecutter.content}}'
    var_5 = 'output'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'content'
    var_9 = '_jinja2_env_vars'
    var_10 = 'my_project'
    var_11 = 'new content'
    var_12 = {}
    var_13 = {var_7: var_10, var_8: var_11, var_9: var_12}
    var_14 = (var_6, var_13)
    var_15 = [var_14]
    var_16 = [var_15]
    var_17 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_18 = None
    var_19 = lambda *args, **kwargs: var_18
    var_20 = True
    var_21 = False

def test_case_0():
    var_0 = 'Test generate_files with nested template directories.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'src'
    var_4 = 'main.py'
    var_5 = '# {{cookiecutter.project_name}}'
    var_6 = 'output'
    var_7 = 'cookiecutter'
    var_8 = 'project_name'
    var_9 = '_jinja2_env_vars'
    var_10 = 'my_app'
    var_11 = {}
    var_12 = {var_8: var_10, var_9: var_11}
    var_13 = (var_7, var_12)
    var_14 = [var_13]
    var_15 = [var_14]
    var_16 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_17 = None
    var_18 = lambda *args, **kwargs: var_17
    var_19 = False

def test_case_0():
    var_0 = 'Test generate_files with accept_hooks=True.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'file.txt'
    var_4 = 'content'
    var_5 = 'output'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = '_jinja2_env_vars'
    var_9 = 'my_project'
    var_10 = {}
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = (var_6, var_11)
    var_13 = [var_12]
    var_14 = [var_13]
    var_15 = []
    var_16 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_17 = True
    var_18 = len(var_15)
    var_19 = bool(var_18 >= 2)
    assert var_19 is True



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_file_name_is_empty_predicate_true. Retrieved 13/31 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project'
    var_1 = 'subdir'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = 'cookiecutter.generate'
    var_7 = 'test_file'
    var_8 = 'from_string'
    var_9 = 'obj'
    var_10 = 'render'
    var_11 = lambda **kw: var_1
    var_12 = {var_10: var_11}



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_generate_files_with_valid_context. Retrieved 13/31 statements.
# Partially parsed test_generate_files_empty_directory_name_raises_exception. Retrieved 9/24 statements.
# Partially parsed test_generate_files_output_dir_exists_without_overwrite. Retrieved 11/28 statements.
# Partially parsed test_generate_files_with_overwrite_if_exists. Retrieved 16/40 statements.
# Partially parsed test_generate_files_default_context. Retrieved 6/20 statements.
# Partially parsed test_generate_files_with_nested_directories. Retrieved 18/43 statements.


def test_case_0():
    var_0 = 'Test generate_files with valid context and template directory.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'README.md'
    var_4 = '# {{cookiecutter.project_name}}'
    var_5 = 'output'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'my_project'
    var_9 = {var_7: var_8}
    var_10 = (var_6, var_9)
    var_11 = [var_10]
    var_12 = [var_11]
    var_13 = False
    var_14 = 'my_project'

def test_case_0():
    var_0 = 'Test generate_files raises EmptyDirNameException for empty directory name.'
    var_1 = 'repo'
    var_2 = '{{}}'
    var_3 = 'output'
    var_4 = 'cookiecutter'
    var_5 = {}
    var_6 = (var_4, var_5)
    var_7 = [var_6]
    var_8 = [var_7]
    var_9 = False
    var_10 = bool(False)
    assert var_10 is True

def test_case_0():
    var_0 = 'Test generate_files raises OutputDirExistsException when output exists without overwrite.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'output'
    var_4 = 'my_project'
    var_5 = 'cookiecutter'
    var_6 = 'project_name'
    var_7 = {var_6: var_4}
    var_8 = (var_5, var_7)
    var_9 = [var_8]
    var_10 = [var_9]
    var_11 = False
    var_12 = bool(False)
    assert var_12 is True

def test_case_0():
    var_0 = 'Test generate_files overwrites existing directory when overwrite_if_exists is True.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'file.txt'
    var_4 = 'content'
    var_5 = 'output'
    var_6 = 'my_project'
    var_7 = 'old_file.txt'
    var_8 = 'old content'
    var_9 = 'cookiecutter'
    var_10 = 'project_name'
    var_11 = {var_10: var_6}
    var_12 = (var_9, var_11)
    var_13 = [var_12]
    var_14 = [var_13]
    var_15 = True
    var_16 = False

def test_case_0():
    var_0 = 'Test generate_files works with default empty context.'
    var_1 = 'repo'
    var_2 = 'myproject'
    var_3 = 'output'
    var_4 = None
    var_5 = False

def test_case_0():
    var_0 = 'Test generate_files with nested template directories.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'src'
    var_4 = '{{cookiecutter.module_name}}'
    var_5 = True
    var_6 = 'module.py'
    var_7 = '# {{cookiecutter.module_name}}'
    var_8 = 'output'
    var_9 = 'cookiecutter'
    var_10 = 'project_name'
    var_11 = 'module_name'
    var_12 = 'my_project'
    var_13 = 'my_module'
    var_14 = {var_10: var_12, var_11: var_13}
    var_15 = (var_9, var_14)
    var_16 = [var_15]
    var_17 = [var_16]
    var_18 = False



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_generate_context_basic. Retrieved 3/7 statements.
# Partially parsed test_generate_context_with_default_context. Retrieved 6/10 statements.
# Partially parsed test_generate_context_with_extra_context. Retrieved 6/10 statements.
# Partially parsed test_generate_context_with_list_choice. Retrieved 6/10 statements.
# Partially parsed test_generate_context_with_multichoice_list. Retrieved 9/15 statements.
# Partially parsed test_generate_context_with_boolean_true. Retrieved 6/10 statements.
# Partially parsed test_generate_context_with_boolean_false. Retrieved 6/10 statements.
# Partially parsed test_generate_context_with_nested_dict. Retrieved 5/11 statements.
# Partially parsed test_generate_context_invalid_json. Retrieved 4/8 statements.
# Partially parsed test_generate_context_invalid_choice_overwrite. Retrieved 7/11 statements.
# Partially parsed test_generate_context_with_string_overwrite. Retrieved 6/10 statements.
# Partially parsed test_generate_context_with_numeric_values. Retrieved 6/10 statements.


def test_case_0():
    var_0 = 'Test generate_context with a basic JSON file.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "my_project", "author": "John Doe"}'
    var_3 = 'cookiecutter'

def test_case_0():
    var_0 = 'Test generate_context with default_context overrides.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "my_project", "version": "1.0"}'
    var_3 = 'project_name'
    var_4 = 'overridden_project'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test generate_context with extra_context overrides.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "my_project", "version": "1.0"}'
    var_3 = 'version'
    var_4 = '2.0'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test generate_context with choice variable (list).'
    var_1 = 'cookiecutter.json'
    var_2 = '{"license": ["MIT", "Apache", "GPL"]}'
    var_3 = 'license'
    var_4 = 'Apache'
    var_5 = {var_3: var_4}
    var_6 = 'MIT'

def test_case_0():
    var_0 = 'Test generate_context with multichoice variable.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"features": ["feature1", "feature2", "feature3"]}'
    var_3 = 'features'
    var_4 = 'feature2'
    var_5 = 'feature3'
    var_6 = [var_4, var_5]
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter'

def test_case_0():
    var_0 = 'Test generate_context with boolean variable converted from string.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"use_ci": false}'
    var_3 = 'use_ci'
    var_4 = 'yes'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test generate_context with boolean variable converted to false.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"use_ci": true}'
    var_3 = 'use_ci'
    var_4 = 'no'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test generate_context with nested dictionary.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"config": {"debug": true, "port": 8000}}'
    var_3 = 'config'
    var_4 = 'debug'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_context with invalid JSON file.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"invalid": json}'
    var_3 = module_0.generate_context(var_0)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'ContextDecodingException'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_context with non-existent file.'
    var_1 = '/non/existent/file.json'
    var_2 = module_0.generate_context(var_1)
    var_3 = bool(False)
    assert var_3 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_context with invalid choice in extra_context.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"license": ["MIT", "Apache"]}'
    var_3 = 'license'
    var_4 = 'GPL'
    var_5 = {var_3: var_4}
    var_6 = module_0.generate_context(var_0, extra_context=var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'GPL'

def test_case_0():
    var_0 = 'Test generate_context with simple string overwrite.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"author": "John", "email": "john@example.com"}'
    var_3 = 'author'
    var_4 = 'Jane'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test generate_context with numeric values.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"port": 8000, "timeout": 30}'
    var_3 = 'port'
    var_4 = 9000
    var_5 = {var_3: var_4}



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_generate_context_loads_json_file. Retrieved 3/7 statements.
# Partially parsed test_generate_context_with_default_context. Retrieved 6/10 statements.
# Partially parsed test_generate_context_with_extra_context. Retrieved 6/10 statements.
# Partially parsed test_generate_context_invalid_json_raises_exception. Retrieved 4/8 statements.
# Partially parsed test_generate_context_with_custom_filename. Retrieved 3/7 statements.
# Partially parsed test_generate_context_with_choice_variable. Retrieved 6/10 statements.
# Partially parsed test_generate_context_with_boolean_variable. Retrieved 6/10 statements.
# Partially parsed test_generate_context_with_nested_dict. Retrieved 8/12 statements.
# Partially parsed test_generate_context_with_multichoice_variable. Retrieved 8/12 statements.
# Partially parsed test_generate_context_with_invalid_choice_raises_error. Retrieved 7/11 statements.
# Partially parsed test_generate_context_empty_file. Retrieved 3/7 statements.
# Partially parsed test_generate_context_with_default_and_extra_context. Retrieved 10/14 statements.


def test_case_0():
    var_0 = 'Test that generate_context loads a JSON file and returns it in a dictionary.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "my_project", "author": "John Doe"}'
    var_3 = 'cookiecutter'

def test_case_0():
    var_0 = 'Test that generate_context applies default_context overwrites.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "my_project", "version": "1.0"}'
    var_3 = 'project_name'
    var_4 = 'override_project'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test that generate_context applies extra_context overwrites.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "my_project", "version": "1.0"}'
    var_3 = 'version'
    var_4 = '2.0'
    var_5 = {var_3: var_4}

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test that generate_context raises ContextDecodingException for invalid JSON.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"invalid json"'
    var_3 = module_0.generate_context(var_0)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'ContextDecodingException'
    var_6 = 'JSON decoding error'

def test_case_0():
    var_0 = 'Test that generate_context uses the filename as the key.'
    var_1 = 'custom_config.json'
    var_2 = '{"key": "value"}'
    var_3 = 'custom_config'

def test_case_0():
    var_0 = 'Test that generate_context handles choice variables in extra_context.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"license": ["MIT", "Apache", "GPL"]}'
    var_3 = 'license'
    var_4 = 'Apache'
    var_5 = {var_3: var_4}
    var_6 = 'MIT'
    var_7 = 'GPL'

def test_case_0():
    var_0 = 'Test that generate_context converts string to boolean for boolean variables.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"use_docker": true}'
    var_3 = 'use_docker'
    var_4 = 'false'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test that generate_context handles nested dictionary overwrites.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"config": {"debug": false, "timeout": 30}}'
    var_3 = 'config'
    var_4 = 'debug'
    var_5 = 'true'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}

def test_case_0():
    var_0 = 'Test that generate_context handles multichoice variables.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"features": ["auth", "api", "admin", "logging"]}'
    var_3 = 'features'
    var_4 = 'auth'
    var_5 = 'admin'
    var_6 = [var_4, var_5]
    var_7 = {var_3: var_6}

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test that generate_context raises ValueError for invalid choice.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"license": ["MIT", "Apache"]}'
    var_3 = 'license'
    var_4 = 'BSD'
    var_5 = {var_3: var_4}
    var_6 = module_0.generate_context(var_0, extra_context=var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'BSD'
    var_9 = 'choice variable'

def test_case_0():
    var_0 = 'Test that generate_context handles empty JSON object.'
    var_1 = 'cookiecutter.json'
    var_2 = '{}'
    var_3 = 'cookiecutter'

def test_case_0():
    var_0 = 'Test that generate_context applies both default and extra context.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"name": "default", "version": "1.0", "author": "original"}'
    var_3 = 'name'
    var_4 = 'version'
    var_5 = 'from_default'
    var_6 = '2.0'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = '3.0'
    var_9 = {var_4: var_8}



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_generate_files_with_minimal_context. Retrieved 14/30 statements.
# Partially parsed test_generate_files_skip_if_file_exists. Retrieved 16/34 statements.
# Partially parsed test_generate_files_overwrite_if_exists. Retrieved 13/29 statements.
# Partially parsed test_generate_files_returns_project_dir_path. Retrieved 12/28 statements.
# Partially parsed test_generate_files_with_nested_directories. Retrieved 17/37 statements.
# Partially parsed test_generate_files_default_output_dir. Retrieved 11/21 statements.


def test_case_0():
    var_0 = 'Test generate_files with minimal context creates project directory.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'README.md'
    var_4 = '# {{cookiecutter.project_name}}'
    var_5 = 'output'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'test_project'
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = 'COOKIECUTTER_ACCEPT_HOOKS'
    var_12 = 'False'
    var_13 = False
    var_14 = 'test_project'

def test_case_0():
    var_0 = 'Test generate_files respects skip_if_file_exists flag.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'file.txt'
    var_4 = '{{cookiecutter.content}}'
    var_5 = 'output'
    var_6 = 'test_project'
    var_7 = 'existing content'
    var_8 = 'cookiecutter'
    var_9 = 'project_name'
    var_10 = 'content'
    var_11 = 'new content'
    var_12 = {var_9: var_6, var_10: var_11}
    var_13 = {var_8: var_12}
    var_14 = True
    var_15 = False

def test_case_0():
    var_0 = 'Test generate_files with overwrite_if_exists flag.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'file.txt'
    var_4 = 'content'
    var_5 = 'output'
    var_6 = 'test_project'
    var_7 = 'cookiecutter'
    var_8 = 'project_name'
    var_9 = {var_8: var_6}
    var_10 = {var_7: var_9}
    var_11 = True
    var_12 = False

def test_case_0():
    var_0 = 'Test generate_files returns the absolute path to project directory.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'file.txt'
    var_4 = 'content'
    var_5 = 'output'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'my_project'
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = False

def test_case_0():
    var_0 = 'Test generate_files creates nested directory structure.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'src'
    var_4 = '{{cookiecutter.module_name}}'
    var_5 = True
    var_6 = 'main.py'
    var_7 = '# {{cookiecutter.module_name}}'
    var_8 = 'output'
    var_9 = 'cookiecutter'
    var_10 = 'project_name'
    var_11 = 'module_name'
    var_12 = 'test_proj'
    var_13 = 'mymodule'
    var_14 = {var_10: var_12, var_11: var_13}
    var_15 = {var_9: var_14}
    var_16 = False

def test_case_0():
    var_0 = 'Test generate_files uses current directory as default output_dir.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'file.txt'
    var_4 = 'content'
    var_5 = 'cookiecutter'
    var_6 = 'project_name'
    var_7 = 'test_project'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = False
    var_11 = 'test_project'



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_generate_context_basic. Retrieved 3/7 statements.
# Partially parsed test_generate_context_with_default_context. Retrieved 6/10 statements.
# Partially parsed test_generate_context_with_extra_context. Retrieved 6/10 statements.
# Partially parsed test_generate_context_with_choice_variable. Retrieved 6/10 statements.
# Partially parsed test_generate_context_with_multichoice_variable. Retrieved 8/12 statements.
# Partially parsed test_generate_context_with_boolean_variable. Retrieved 8/12 statements.
# Partially parsed test_generate_context_with_nested_dict. Retrieved 8/12 statements.
# Partially parsed test_generate_context_invalid_json. Retrieved 4/8 statements.
# Partially parsed test_generate_context_invalid_choice_value. Retrieved 7/11 statements.
# Partially parsed test_generate_context_invalid_multichoice_value. Retrieved 9/13 statements.
# Partially parsed test_generate_context_invalid_boolean_conversion. Retrieved 7/11 statements.
# Partially parsed test_generate_context_preserves_other_types. Retrieved 3/7 statements.
# Partially parsed test_generate_context_default_context_with_invalid_value. Retrieved 6/10 statements.


def test_case_0():
    var_0 = 'Test basic context generation from a JSON file.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "my_project", "version": "1.0"}'
    var_3 = 'cookiecutter'

def test_case_0():
    var_0 = 'Test context generation with default context overrides.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "default_name", "version": "1.0"}'
    var_3 = 'project_name'
    var_4 = 'overridden_name'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test context generation with extra context overrides.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "default_name", "version": "1.0"}'
    var_3 = 'version'
    var_4 = '2.0'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test context generation with choice variables.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"license": ["MIT", "Apache", "GPL"]}'
    var_3 = 'license'
    var_4 = 'Apache'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test context generation with multi-choice variables.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"features": ["auth", "api", "admin", "logging"]}'
    var_3 = 'features'
    var_4 = 'api'
    var_5 = 'logging'
    var_6 = [var_4, var_5]
    var_7 = {var_3: var_6}

def test_case_0():
    var_0 = 'Test context generation with boolean variables.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"use_docker": true, "use_ci": false}'
    var_3 = 'use_docker'
    var_4 = 'use_ci'
    var_5 = 'no'
    var_6 = 'yes'
    var_7 = {var_3: var_5, var_4: var_6}

def test_case_0():
    var_0 = 'Test context generation with nested dictionary variables.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"config": {"debug": true, "port": 8000}}'
    var_3 = 'config'
    var_4 = 'debug'
    var_5 = 'no'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test context generation with invalid JSON raises ContextDecodingException.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"invalid": json}'
    var_3 = module_0.generate_context(var_0)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'ContextDecodingException'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test context generation with invalid choice value in extra context.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"license": ["MIT", "Apache"]}'
    var_3 = 'license'
    var_4 = 'GPL'
    var_5 = {var_3: var_4}
    var_6 = module_0.generate_context(var_0, extra_context=var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'GPL'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test context generation with invalid multichoice value in extra context.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"features": ["auth", "api"]}'
    var_3 = 'features'
    var_4 = 'auth'
    var_5 = 'invalid'
    var_6 = [var_4, var_5]
    var_7 = {var_3: var_6}
    var_8 = module_0.generate_context(var_0, extra_context=var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = 'invalid'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test context generation with invalid boolean conversion.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"use_docker": true}'
    var_3 = 'use_docker'
    var_4 = 'invalid_bool'
    var_5 = {var_3: var_4}
    var_6 = module_0.generate_context(var_0, extra_context=var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'could not be converted to a boolean'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test context generation with non-existent file.'
    var_1 = '/nonexistent/path/cookiecutter.json'
    var_2 = module_0.generate_context(var_1)
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'Test that context generation preserves other data types.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"port": 8000, "timeout": 30.5, "name": "test"}'

def test_case_0():
    var_0 = 'Test that invalid default context raises warning but continues.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"license": ["MIT", "Apache"]}'
    var_3 = 'license'
    var_4 = 'InvalidLicense'
    var_5 = {var_3: var_4}



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_generate_files_with_minimal_context. Retrieved 13/27 statements.
# Partially parsed test_generate_files_empty_dirname_raises_exception. Retrieved 11/24 statements.
# Partially parsed test_generate_files_with_overwrite_if_exists. Retrieved 14/31 statements.
# Partially parsed test_generate_files_without_context. Retrieved 8/20 statements.
# Partially parsed test_generate_files_with_skip_if_file_exists. Retrieved 14/28 statements.


def test_case_0():
    var_0 = 'Test generate_files with minimal context.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'test.txt'
    var_4 = 'Hello {{cookiecutter.project_name}}'
    var_5 = 'output'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'my_project'
    var_9 = {var_7: var_8}
    var_10 = (var_6, var_9)
    var_11 = [var_10]
    var_12 = [var_11]
    var_13 = False
    var_14 = 'my_project'

def test_case_0():
    var_0 = 'Test that empty directory name raises EmptyDirNameException.'
    var_1 = 'repo'
    var_2 = ''
    var_3 = 'output'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'my_project'
    var_7 = {var_5: var_6}
    var_8 = (var_4, var_7)
    var_9 = [var_8]
    var_10 = [var_9]
    var_11 = False
    var_12 = bool(False)
    assert var_12 is True

def test_case_0():
    var_0 = 'Test generate_files with overwrite_if_exists flag.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'test.txt'
    var_4 = 'Content'
    var_5 = 'output'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'my_project'
    var_9 = {var_7: var_8}
    var_10 = (var_6, var_9)
    var_11 = [var_10]
    var_12 = [var_11]
    var_13 = False
    var_14 = True

def test_case_0():
    var_0 = 'Test generate_files with None context.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'test.txt'
    var_4 = 'Content'
    var_5 = 'output'
    var_6 = None
    var_7 = False

def test_case_0():
    var_0 = 'Test generate_files with skip_if_file_exists flag.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'test.txt'
    var_4 = 'Original content'
    var_5 = 'output'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'my_project'
    var_9 = {var_7: var_8}
    var_10 = (var_6, var_9)
    var_11 = [var_10]
    var_12 = [var_11]
    var_13 = True
    var_14 = False



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_generate_context_file_open_predicate_line_18. Retrieved 2/10 statements.


def test_case_0():
    var_0 = "Test that the predicate at line 18 (open file operation) evaluates to False when file doesn't exist."
    var_1 = 'non_existent_cookiecutter.json'
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_generate_files_basic. Retrieved 16/38 statements.
# Partially parsed test_generate_files_empty_dirname_raises_exception. Retrieved 13/33 statements.
# Partially parsed test_generate_files_with_overwrite_if_exists. Retrieved 19/43 statements.
# Partially parsed test_generate_files_with_skip_if_file_exists. Retrieved 18/40 statements.
# Partially parsed test_generate_files_with_copy_without_render. Retrieved 19/36 statements.
# Partially parsed test_generate_files_default_context. Retrieved 7/15 statements.


def test_case_0():
    var_0 = 'Test basic generate_files functionality with minimal setup.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'test.txt'
    var_4 = 'Hello {{cookiecutter.project_name}}'
    var_5 = 'output'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'my_project'
    var_9 = {var_7: var_8}
    var_10 = (var_6, var_9)
    var_11 = [var_10]
    var_12 = [var_11]
    var_13 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_14 = None
    var_15 = lambda *args, **kwargs: var_14
    var_16 = False

def test_case_0():
    var_0 = 'Test that empty directory name raises EmptyDirNameException.'
    var_1 = 'repo'
    var_2 = ''
    var_3 = 'output'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'my_project'
    var_7 = {var_5: var_6}
    var_8 = (var_4, var_7)
    var_9 = [var_8]
    var_10 = [var_9]
    var_11 = 'cookiecutter.generate.find_template'
    var_12 = 'cookiecutter.generate.os.path.split'
    var_13 = False
    var_14 = bool(False)
    assert var_14 is True

def test_case_0():
    var_0 = 'Test generate_files with overwrite_if_exists=True.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'test.txt'
    var_4 = 'Hello {{cookiecutter.project_name}}'
    var_5 = 'output'
    var_6 = 'my_project'
    var_7 = 'old_file.txt'
    var_8 = 'old content'
    var_9 = 'cookiecutter'
    var_10 = 'project_name'
    var_11 = {var_10: var_6}
    var_12 = (var_9, var_11)
    var_13 = [var_12]
    var_14 = [var_13]
    var_15 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_16 = None
    var_17 = lambda *args, **kwargs: var_16
    var_18 = True
    var_19 = False

def test_case_0():
    var_0 = 'Test generate_files with skip_if_file_exists=True.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'test.txt'
    var_4 = 'New content'
    var_5 = 'output'
    var_6 = 'my_project'
    var_7 = 'old content'
    var_8 = 'cookiecutter'
    var_9 = 'project_name'
    var_10 = {var_9: var_6}
    var_11 = (var_8, var_10)
    var_12 = [var_11]
    var_13 = [var_12]
    var_14 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_15 = None
    var_16 = lambda *args, **kwargs: var_15
    var_17 = True
    var_18 = False

def test_case_0():
    var_0 = 'Test generate_files respects _copy_without_render setting.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'binary.bin'
    var_4 = b'\x89PNG\r\n\x1a\n'
    var_5 = 'output'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = '_copy_without_render'
    var_9 = 'my_project'
    var_10 = '*.bin'
    var_11 = [var_10]
    var_12 = {var_7: var_9, var_8: var_11}
    var_13 = (var_6, var_12)
    var_14 = [var_13]
    var_15 = [var_14]
    var_16 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_17 = None
    var_18 = lambda *args, **kwargs: var_17
    var_19 = False

def test_case_0():
    var_0 = 'Test generate_files with default None context.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'output'
    var_4 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_5 = None
    var_6 = lambda *args, **kwargs: var_5



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_generate_context_basic. Retrieved 3/7 statements.
# Partially parsed test_generate_context_with_default_context. Retrieved 6/10 statements.
# Partially parsed test_generate_context_with_extra_context. Retrieved 6/10 statements.
# Partially parsed test_generate_context_invalid_json. Retrieved 4/8 statements.
# Partially parsed test_generate_context_with_list_variable. Retrieved 6/10 statements.
# Partially parsed test_generate_context_with_dict_variable. Retrieved 5/11 statements.
# Partially parsed test_generate_context_with_boolean_string. Retrieved 6/10 statements.
# Partially parsed test_generate_context_with_ordered_dict. Retrieved 4/11 statements.
# Partially parsed test_generate_context_nested_file_name. Retrieved 4/10 statements.
# Partially parsed test_generate_context_invalid_default_context. Retrieved 6/10 statements.
# Partially parsed test_generate_context_invalid_extra_context_raises. Retrieved 7/11 statements.
# Partially parsed test_generate_context_with_multichoice. Retrieved 9/15 statements.
# Partially parsed test_generate_context_custom_file_name. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'Test basic context generation from a JSON file.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "my_project", "author": "John"}'
    var_3 = 'cookiecutter'

def test_case_0():
    var_0 = 'Test context generation with default_context overrides.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "my_project", "author": "John"}'
    var_3 = 'author'
    var_4 = 'Jane'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test context generation with extra_context overrides.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "my_project", "author": "John"}'
    var_3 = 'project_name'
    var_4 = 'new_project'
    var_5 = {var_3: var_4}

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test context generation with invalid JSON raises ContextDecodingException.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"invalid": json}'
    var_3 = module_0.generate_context(var_0)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'ContextDecodingException'
    var_6 = 'JSON decoding error'

def test_case_0():
    var_0 = 'Test context generation with list (choice) variables.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"flavor": ["vanilla", "chocolate", "strawberry"]}'
    var_3 = 'flavor'
    var_4 = 'chocolate'
    var_5 = {var_3: var_4}
    var_6 = 'vanilla'
    var_7 = 'strawberry'

def test_case_0():
    var_0 = 'Test context generation with nested dictionary variables.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"options": {"debug": false, "verbose": true}}'
    var_3 = 'options'
    var_4 = 'debug'

def test_case_0():
    var_0 = 'Test context generation with boolean variable and string overwrite.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"use_https": true}'
    var_3 = 'use_https'
    var_4 = 'false'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test context generation preserves order.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"first": "1", "second": "2", "third": "3"}'
    var_3 = 'cookiecutter'

def test_case_0():
    var_0 = 'Test context generation with nested directory structure.'
    var_1 = 'templates'
    var_2 = 'cookiecutter.json'
    var_3 = '{"name": "test"}'
    var_4 = 'cookiecutter'

def test_case_0():
    var_0 = 'Test context generation with invalid default_context issues warning.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"choices": ["a", "b"], "name": "test"}'
    var_3 = 'choices'
    var_4 = 'invalid_choice'
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test context generation with invalid extra_context raises ValueError.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"choices": ["a", "b"]}'
    var_3 = 'choices'
    var_4 = 'invalid_choice'
    var_5 = {var_3: var_4}
    var_6 = module_0.generate_context(var_0, extra_context=var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'invalid_choice'

def test_case_0():
    var_0 = 'Test context generation with multi-choice variables.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"features": ["auth", "api", "admin", "tests"]}'
    var_3 = 'features'
    var_4 = 'api'
    var_5 = 'tests'
    var_6 = [var_4, var_5]
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter'

def test_case_0():
    var_0 = 'Test context generation with custom context file name.'
    var_1 = 'config.json'
    var_2 = '{"value": "data"}'
    var_3 = 'config'



# Parsed testcases at query #2
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 30
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'Jane'
    var_6 = {var_0: var_5}
    var_7 = module_0.apply_overwrites_to_context(var_4, var_6)
    var_8 = bool(var_4 == {'name': 'Jane', 'age': 30})
    assert var_8 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'John'
    var_2 = {var_0: var_1}
    var_3 = 'new_var'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_2, var_5)
    var_7 = bool(var_2 == {'name': 'John'})
    assert var_7 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'settings'
    var_1 = 'theme'
    var_2 = 'dark'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'new_key'
    var_6 = 'new_value'
    var_7 = {var_5: var_6}
    var_8 = {var_0: var_7}
    var_9 = module_0.apply_overwrites_to_context(var_4, var_8)
    var_10 = bool(var_4 == {'settings': {'theme': 'dark', 'new_key': 'new_value'}})
    assert var_10 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'flavor'
    var_1 = 'vanilla'
    var_2 = 'chocolate'
    var_3 = 'strawberry'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = {var_0: var_2}
    var_7 = module_0.apply_overwrites_to_context(var_5, var_6)
    var_8 = bool(var_5 == {'flavor': ['chocolate', 'vanilla', 'strawberry']})
    assert var_8 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'flavor'
    var_1 = 'vanilla'
    var_2 = 'chocolate'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 'invalid'
    var_6 = {var_0: var_5}
    var_7 = module_0.apply_overwrites_to_context(var_4, var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'invalid provided for choice variable'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'options'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = [var_2, var_3]
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_5, var_7)
    var_9 = bool(var_5 == {'options': ['b', 'c']})
    assert var_9 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'options'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 'd'
    var_7 = [var_2, var_6]
    var_8 = {var_0: var_7}
    var_9 = module_0.apply_overwrites_to_context(var_5, var_8)
    var_10 = bool(False)
    assert var_10 is True
    var_11 = 'multi-choice variable'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'enabled'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = 'yes'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(var_2 == {'enabled': True})
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'enabled'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'no'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(var_2 == {'enabled': False})
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'debug'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = 'true'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(var_2 == {'debug': True})
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'debug'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'false'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(var_2 == {'debug': False})
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'enabled'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'invalid'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'could not be converted to a boolean'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'db'
    var_1 = 'host'
    var_2 = 'port'
    var_3 = 'localhost'
    var_4 = 5432
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'remotehost'
    var_8 = {var_1: var_7}
    var_9 = {var_0: var_8}
    var_10 = module_0.apply_overwrites_to_context(var_6, var_9)
    var_11 = bool(var_6 == {'db': {'host': 'remotehost', 'port': 5432}})
    assert var_11 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'settings'
    var_1 = 'theme'
    var_2 = 'dark'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'new_setting'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = {var_0: var_7}
    var_9 = True
    var_10 = module_0.apply_overwrites_to_context(var_4, var_8, in_dictionary_variable=var_9)
    var_11 = bool(var_4 == {'settings': {'theme': 'dark', 'new_setting': 'value'}})
    assert var_11 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'options'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 'c'
    var_6 = [var_5]
    var_7 = {var_0: var_6}
    var_8 = True
    var_9 = module_0.apply_overwrites_to_context(var_4, var_7, in_dictionary_variable=var_8)
    var_10 = bool(var_4 == {'options': ['c']})
    assert var_10 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'city'
    var_3 = 'John'
    var_4 = 30
    var_5 = 'NYC'
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 'Jane'
    var_8 = 25
    var_9 = {var_0: var_7, var_1: var_8}
    var_10 = module_0.apply_overwrites_to_context(var_6, var_9)
    var_11 = bool(var_6 == {'name': 'Jane', 'age': 25, 'city': 'NYC'})
    assert var_11 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 30
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {}
    var_6 = module_0.apply_overwrites_to_context(var_4, var_5)
    var_7 = bool(var_4 == {'name': 'John', 'age': 30})
    assert var_7 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_predicate_at_line_52_evaluates_to_false. Retrieved 3/6 statements.
# Partially parsed test_predicate_at_line_52_evaluates_to_false_when_context_value_not_bool. Retrieved 3/6 statements.
# Partially parsed test_predicate_at_line_52_evaluates_to_false_when_overwrite_not_str. Retrieved 3/6 statements.
# Partially parsed test_predicate_at_line_52_evaluates_to_false_when_both_wrong_type. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 52 evaluates to False for non-matching types.'
    var_1 = 'not a boolean'
    var_2 = 'yes'

def test_case_0():
    var_0 = 'Test predicate is False when context_value is not a boolean.'
    var_1 = 42
    var_2 = 'yes'

def test_case_0():
    var_0 = 'Test predicate is False when overwrite is not a string.'
    var_1 = True
    var_2 = 123

def test_case_0():
    var_0 = 'Test predicate is False when both context_value and overwrite are wrong types.'
    var_1 = []
    var_2 = {}



# Parsed testcases at query #4
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 52 evaluates to False with non-matching types.'
    var_1 = 'key'
    var_2 = 123
    var_3 = {var_1: var_2}
    var_4 = 'value'
    var_5 = {var_1: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)
    var_7 = var_3['key']
    assert var_7 == 'value'



# Parsed testcases at query #5
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 52 evaluates to False when conditions are not met.'
    var_1 = 'my_var'
    var_2 = 'string_value'
    var_3 = {var_1: var_2}
    var_4 = 'new_string'
    var_5 = {var_1: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)
    var_7 = var_3['my_var']
    assert var_7 == 'new_string'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_predicate_at_line_52_evaluates_to_false. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 52 evaluates to False.'
    var_1 = 'not_a_bool'
    var_2 = 123



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_render_and_create_dir_empty_dirname. Retrieved 3/12 statements.
# Partially parsed test_render_and_create_dir_none_dirname. Retrieved 3/12 statements.
# Partially parsed test_render_and_create_dir_creates_new_directory. Retrieved 3/12 statements.
# Partially parsed test_render_and_create_dir_with_template_rendering. Retrieved 6/15 statements.
# Partially parsed test_render_and_create_dir_exists_no_overwrite. Retrieved 4/15 statements.
# Partially parsed test_render_and_create_dir_exists_with_overwrite. Retrieved 4/14 statements.
# Partially parsed test_render_and_create_dir_nested_path. Retrieved 3/12 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = ''
    var_3 = bool(False)
    assert var_3 is True

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = None
    var_3 = bool(False)
    assert var_3 is True

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = 'test_dir'

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'project_name'
    var_2 = 'my_project'
    var_3 = {var_1: var_2}
    var_4 = '{{ project_name }}_dir'
    var_5 = 'my_project_dir'

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = 'existing_dir'
    var_3 = False
    var_4 = bool(False)
    assert var_4 is True

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = 'existing_dir'
    var_3 = True

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = 'parent/child/grandchild'



# Parsed testcases at query #8
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test that valid yes/no responses are converted to boolean without raising InvalidResponse.'
    var_1 = 'debug'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = 'no'
    var_5 = {var_1: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)
    var_7 = var_3['debug']
    assert var_7 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = "Test that 'yes' response is converted to True."
    var_1 = 'enabled'
    var_2 = False
    var_3 = {var_1: var_2}
    var_4 = 'yes'
    var_5 = {var_1: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)
    var_7 = var_3['enabled']
    assert var_7 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = "Test that 'true' string is converted to boolean True."
    var_1 = 'flag'
    var_2 = False
    var_3 = {var_1: var_2}
    var_4 = 'true'
    var_5 = {var_1: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)
    var_7 = var_3['flag']
    assert var_7 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = "Test that 'false' string is converted to boolean False."
    var_1 = 'flag'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = 'false'
    var_5 = {var_1: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)
    var_7 = var_3['flag']
    assert var_7 is False



# Parsed testcases at query #9
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'existing_var'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'new_var'
    var_4 = 'new_value'
    var_5 = {var_3: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_2, var_5)
    var_7 = bool(var_2 == {'existing_var': 'value'})
    assert var_7 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'nested'
    var_1 = 'existing'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'new_key'
    var_6 = 'new_value'
    var_7 = {var_5: var_6}
    var_8 = {var_0: var_7}
    var_9 = module_0.apply_overwrites_to_context(var_4, var_8)
    var_10 = bool(var_4 == {'nested': {'existing': 'value', 'new_key': 'new_value'}})
    assert var_10 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'choices'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = [var_2, var_3]
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_5, var_7)
    var_9 = bool(var_5 == {'choices': ['b', 'c']})
    assert var_9 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'choices'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 'd'
    var_7 = 'e'
    var_8 = [var_6, var_7]
    var_9 = {var_0: var_8}
    var_10 = module_0.apply_overwrites_to_context(var_5, var_9)
    var_11 = bool(False)
    assert var_11 is True
    var_12 = 'provided for multi-choice variable'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'choice'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = {var_0: var_3}
    var_7 = module_0.apply_overwrites_to_context(var_5, var_6)
    var_8 = bool(var_5 == {'choice': ['c', 'a', 'b']})
    assert var_8 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'choice'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 'd'
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_5, var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = 'provided for choice variable'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'config'
    var_1 = 'key1'
    var_2 = 'key2'
    var_3 = 'value1'
    var_4 = 'value2'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'new_value1'
    var_8 = {var_1: var_7}
    var_9 = {var_0: var_8}
    var_10 = module_0.apply_overwrites_to_context(var_6, var_9)
    var_11 = bool(var_6 == {'config': {'key1': 'new_value1', 'key2': 'value2'}})
    assert var_11 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'is_enabled'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'yes'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(var_2 == {'is_enabled': True})
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'is_enabled'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'no'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(var_2 == {'is_enabled': False})
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'is_enabled'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'invalid'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'could not be converted to a boolean'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'var'
    var_1 = 'old_value'
    var_2 = {var_0: var_1}
    var_3 = 'new_value'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(var_2 == {'var': 'new_value'})
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'nested'
    var_1 = 'choices'
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'c'
    var_5 = [var_2, var_3, var_4]
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = 'new_value'
    var_9 = {var_1: var_8}
    var_10 = {var_0: var_9}
    var_11 = module_0.apply_overwrites_to_context(var_7, var_10)
    var_12 = bool(var_7 == {'nested': {'choices': 'new_value'}})
    assert var_12 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'var1'
    var_1 = 'var2'
    var_2 = 'var3'
    var_3 = 'value1'
    var_4 = 'value2'
    var_5 = 'value3'
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 'new1'
    var_8 = 'new3'
    var_9 = {var_0: var_7, var_2: var_8}
    var_10 = module_0.apply_overwrites_to_context(var_6, var_9)
    var_11 = bool(var_6 == {'var1': 'new1', 'var2': 'value2', 'var3': 'new3'})
    assert var_11 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'flag'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = '1'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(var_2 == {'flag': True})
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'flag'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = '0'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(var_2 == {'flag': False})
    assert var_6 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_render_and_create_dir_with_empty_dirname. Retrieved 3/8 statements.
# Partially parsed test_render_and_create_dir_with_none_dirname. Retrieved 3/8 statements.
# Partially parsed test_render_and_create_dir_creates_new_directory. Retrieved 3/9 statements.
# Partially parsed test_render_and_create_dir_with_template_rendering. Retrieved 6/12 statements.
# Partially parsed test_render_and_create_dir_existing_dir_raises_exception. Retrieved 4/12 statements.
# Partially parsed test_render_and_create_dir_existing_dir_with_overwrite. Retrieved 4/11 statements.
# Partially parsed test_render_and_create_dir_creates_nested_directories. Retrieved 3/9 statements.
# Partially parsed test_render_and_create_dir_returns_tuple. Retrieved 5/15 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Environment()
    var_2 = ''
    var_3 = bool(False)
    assert var_3 is True

import jinja2.environment as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Environment()
    var_2 = None
    var_3 = bool(False)
    assert var_3 is True

import jinja2.environment as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Environment()
    var_2 = 'my_new_dir'

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'my_project'
    var_2 = {var_0: var_1}
    var_3 = module_0.Environment()
    var_4 = '{{ project_name }}_dir'
    var_5 = 'my_project_dir'

import jinja2.environment as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Environment()
    var_2 = 'existing_dir'
    var_3 = True
    var_4 = bool(False)
    assert var_4 is True

import jinja2.environment as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Environment()
    var_2 = 'existing_dir'
    var_3 = True

import jinja2.environment as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Environment()
    var_2 = 'parent/child/grandchild'

import jinja2.environment as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Environment()
    var_2 = 'test_dir'
    var_3 = 0
    var_4 = 1



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_render_and_create_dir_predicate_line_24_evaluates_to_true. Retrieved 6/13 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 24 evaluates to True when directory exists.'
    var_1 = 'existing_project'
    var_2 = True
    var_3 = 'project_name'
    var_4 = {var_3: var_1}
    var_5 = module_0.Environment()



# Parsed testcases at query #12
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = '*.bin'
    var_3 = '*.exe'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'file.bin'
    var_8 = module_0.is_copy_only_path(var_7, var_6)
    assert var_8 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = '*.bin'
    var_3 = '*.exe'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'file.txt'
    var_8 = module_0.is_copy_only_path(var_7, var_6)
    assert var_8 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = 'node_modules/*'
    var_3 = 'dist/*'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'node_modules/package'
    var_8 = module_0.is_copy_only_path(var_7, var_6)
    assert var_8 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = 'file.txt'
    var_4 = module_0.is_copy_only_path(var_3, var_2)
    assert var_4 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'file.txt'
    var_2 = module_0.is_copy_only_path(var_1, var_0)
    assert var_2 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = []
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'file.txt'
    var_6 = module_0.is_copy_only_path(var_5, var_4)
    assert var_6 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = '*.bin'
    var_3 = '*.exe'
    var_4 = '*.dll'
    var_5 = [var_2, var_3, var_4]
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = 'app.exe'
    var_9 = module_0.is_copy_only_path(var_8, var_7)
    assert var_9 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = '*.bin'
    var_3 = '*.exe'
    var_4 = '*.dll'
    var_5 = [var_2, var_3, var_4]
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = 'library.dll'
    var_9 = module_0.is_copy_only_path(var_8, var_7)
    assert var_9 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = 'file?.txt'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}
    var_6 = 'file1.txt'
    var_7 = module_0.is_copy_only_path(var_6, var_5)
    assert var_7 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = 'file[0-9].txt'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}
    var_6 = 'file5.txt'
    var_7 = module_0.is_copy_only_path(var_6, var_5)
    assert var_7 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_generate_context_loads_json_file. Retrieved 3/7 statements.
# Partially parsed test_generate_context_with_extra_context. Retrieved 8/12 statements.
# Partially parsed test_generate_context_with_default_context. Retrieved 6/10 statements.
# Partially parsed test_generate_context_with_boolean_conversion. Retrieved 6/10 statements.
# Partially parsed test_generate_context_with_choice_variable. Retrieved 6/10 statements.
# Partially parsed test_generate_context_with_multichoice_variable. Retrieved 8/12 statements.
# Partially parsed test_generate_context_with_nested_dict. Retrieved 8/12 statements.
# Partially parsed test_generate_context_invalid_json_raises_exception. Retrieved 4/8 statements.
# Partially parsed test_generate_context_invalid_choice_raises_error. Retrieved 7/11 statements.
# Partially parsed test_generate_context_invalid_multichoice_raises_error. Retrieved 9/13 statements.
# Partially parsed test_generate_context_invalid_boolean_conversion_raises_error. Retrieved 7/11 statements.
# Partially parsed test_generate_context_custom_file_stem. Retrieved 3/7 statements.
# Partially parsed test_generate_context_boolean_true_conversion. Retrieved 6/10 statements.
# Partially parsed test_generate_context_simple_string_overwrite. Retrieved 6/10 statements.


def test_case_0():
    var_0 = 'Test that generate_context loads a JSON file correctly.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "my_project", "project_slug": "my_project"}'
    var_3 = 'cookiecutter'

def test_case_0():
    var_0 = 'Test that generate_context applies extra_context overrides.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "my_project", "author": "default_author"}'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'overridden_project'
    var_6 = 'new_author'
    var_7 = {var_3: var_5, var_4: var_6}

def test_case_0():
    var_0 = 'Test that generate_context applies default_context overrides.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "my_project"}'
    var_3 = 'project_name'
    var_4 = 'default_overridden'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test that generate_context converts string to boolean in extra_context.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"use_docker": true}'
    var_3 = 'use_docker'
    var_4 = 'false'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test that generate_context handles choice variables.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"license": ["MIT", "Apache", "GPL"]}'
    var_3 = 'license'
    var_4 = 'Apache'
    var_5 = {var_3: var_4}
    var_6 = 'MIT'
    var_7 = 'GPL'

def test_case_0():
    var_0 = 'Test that generate_context handles multichoice variables.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"features": ["feature1", "feature2", "feature3"]}'
    var_3 = 'features'
    var_4 = 'feature1'
    var_5 = 'feature3'
    var_6 = [var_4, var_5]
    var_7 = {var_3: var_6}

def test_case_0():
    var_0 = 'Test that generate_context handles nested dictionary variables.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"config": {"key1": "value1", "key2": "value2"}}'
    var_3 = 'config'
    var_4 = 'key1'
    var_5 = 'overridden'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test that generate_context raises ContextDecodingException for invalid JSON.'
    var_1 = 'cookiecutter.json'
    var_2 = '{invalid json}'
    var_3 = module_0.generate_context(var_0)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'ContextDecodingException'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test that generate_context raises ValueError for invalid choice.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"license": ["MIT", "Apache"]}'
    var_3 = 'license'
    var_4 = 'GPL'
    var_5 = {var_3: var_4}
    var_6 = module_0.generate_context(var_0, extra_context=var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'GPL'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test that generate_context raises ValueError for invalid multichoice.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"features": ["feature1", "feature2"]}'
    var_3 = 'features'
    var_4 = 'feature1'
    var_5 = 'invalid_feature'
    var_6 = [var_4, var_5]
    var_7 = {var_3: var_6}
    var_8 = module_0.generate_context(var_0, extra_context=var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = 'invalid_feature'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test that generate_context raises ValueError for invalid boolean conversion.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"use_docker": true}'
    var_3 = 'use_docker'
    var_4 = 'invalid_bool'
    var_5 = {var_3: var_4}
    var_6 = module_0.generate_context(var_0, extra_context=var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'could not be converted to a boolean'

def test_case_0():
    var_0 = 'Test that generate_context uses custom file stem as context key.'
    var_1 = 'custom_config.json'
    var_2 = '{"project": "test"}'
    var_3 = 'custom_config'

def test_case_0():
    var_0 = "Test that generate_context converts 'true' string to boolean True."
    var_1 = 'cookiecutter.json'
    var_2 = '{"use_feature": false}'
    var_3 = 'use_feature'
    var_4 = 'yes'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test that generate_context overwrites simple string variables.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"author": "John Doe", "email": "john@example.com"}'
    var_3 = 'author'
    var_4 = 'Jane Doe'
    var_5 = {var_3: var_4}



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_deprecated_warning. Retrieved 10/31 statements.
# Partially parsed test_run_hook_from_repo_dir_calls_actual_hook. Retrieved 9/21 statements.
# Partially parsed test_run_hook_from_repo_dir_with_different_hook_names. Retrieved 15/29 statements.


def test_case_0():
    var_0 = 'Test that _run_hook_from_repo_dir issues a deprecation warning.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    assert var_3 == 1
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'always'
    var_7 = 'post_gen_project'
    var_8 = False
    var_9 = var_4.category
    var_10 = 'deprecated'
    var_11 = 'run_hook_from_repo_dir'

def test_case_0():
    var_0 = 'Test that _run_hook_from_repo_dir calls the actual run_hook_from_repo_dir.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = True
    var_7 = 'ignore'
    var_8 = 'pre_prompt'

def test_case_0():
    var_0 = 'Test _run_hook_from_repo_dir with various hook names.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'test'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'pre_prompt'
    var_9 = 'post_prompt'
    var_10 = 'pre_gen_project'
    var_11 = 'post_gen_project'
    var_12 = [var_8, var_9, var_10, var_11]
    var_13 = 'ignore'
    var_14 = False



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_deprecation_warning. Retrieved 13/20 statements.
# Partially parsed test_run_hook_from_repo_dir_calls_actual_function. Retrieved 13/17 statements.
# Partially parsed test_run_hook_from_repo_dir_with_false_delete_flag. Retrieved 12/16 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test that _run_hook_from_repo_dir issues a deprecation warning.'
    var_1 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_2 = '/path/to/repo'
    var_3 = 'post_gen_project'
    var_4 = '/path/to/project'
    var_5 = 'cookiecutter'
    var_6 = 'project_name'
    var_7 = 'test'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = True
    var_11 = module_0._run_hook_from_repo_dir(var_2, var_3, var_4, var_9, var_10)
    var_12 = 0
    var_13 = 'deprecated'
    var_14 = bool('deprecated' in var_7)
    assert var_14 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test that _run_hook_from_repo_dir delegates to run_hook_from_repo_dir.'
    var_1 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_2 = 'warnings.warn'
    var_3 = '/path/to/repo'
    var_4 = 'post_gen_project'
    var_5 = '/path/to/project'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'test'
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = True
    var_12 = module_0._run_hook_from_repo_dir(var_3, var_4, var_5, var_10, var_11)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test _run_hook_from_repo_dir with delete_project_on_failure=False.'
    var_1 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_2 = 'warnings.warn'
    var_3 = '/path/to/repo'
    var_4 = 'pre_gen_project'
    var_5 = '/path/to/project'
    var_6 = 'cookiecutter'
    var_7 = {}
    var_8 = {var_6: var_7}
    var_9 = False
    var_10 = module_0._run_hook_from_repo_dir(var_3, var_4, var_5, var_8, var_9)
    var_11 = False



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_render_and_create_dir_predicate_at_line_24_evaluates_to_true. Retrieved 4/11 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = {}
    var_2 = module_0.Environment()
    var_3 = True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_generate_context_loads_json_file. Retrieved 2/6 statements.
# Partially parsed test_generate_context_with_default_context. Retrieved 5/9 statements.
# Partially parsed test_generate_context_with_extra_context. Retrieved 5/9 statements.
# Partially parsed test_generate_context_with_both_default_and_extra_context. Retrieved 8/12 statements.
# Partially parsed test_generate_context_invalid_json_raises_exception. Retrieved 3/7 statements.
# Partially parsed test_generate_context_file_not_found_raises_exception. Retrieved 1/5 statements.
# Partially parsed test_generate_context_with_choice_variable. Retrieved 5/9 statements.
# Partially parsed test_generate_context_with_multichoice_variable. Retrieved 8/14 statements.
# Partially parsed test_generate_context_with_boolean_variable_yes. Retrieved 5/9 statements.
# Partially parsed test_generate_context_with_boolean_variable_no. Retrieved 5/9 statements.
# Partially parsed test_generate_context_with_nested_dict. Retrieved 7/11 statements.
# Partially parsed test_generate_context_with_invalid_choice_raises_error. Retrieved 6/10 statements.
# Partially parsed test_generate_context_custom_context_file_name. Retrieved 2/6 statements.
# Partially parsed test_generate_context_with_invalid_default_context_warning. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = '{"project_name": "my_project", "version": "1.0"}'
    var_2 = 'cookiecutter'

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = '{"project_name": "my_project", "version": "1.0"}'
    var_2 = 'project_name'
    var_3 = 'default_project'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = '{"project_name": "my_project", "version": "1.0"}'
    var_2 = 'version'
    var_3 = '2.0'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = '{"project_name": "my_project", "version": "1.0"}'
    var_2 = 'project_name'
    var_3 = 'default_project'
    var_4 = {var_2: var_3}
    var_5 = 'version'
    var_6 = '2.0'
    var_7 = {var_5: var_6}

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = '{invalid json}'
    var_2 = module_0.generate_context(var_0)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'JSON decoding error'

def test_case_0():
    var_0 = 'nonexistent.json'
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = '{"license": ["MIT", "Apache", "GPL"]}'
    var_2 = 'license'
    var_3 = 'Apache'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = '{"features": ["feature1", "feature2", "feature3"]}'
    var_2 = 'features'
    var_3 = 'feature2'
    var_4 = 'feature3'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = 'cookiecutter'

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = '{"use_docker": false}'
    var_2 = 'use_docker'
    var_3 = 'yes'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = '{"use_docker": true}'
    var_2 = 'use_docker'
    var_3 = 'no'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = '{"database": {"engine": "postgresql", "port": 5432}}'
    var_2 = 'database'
    var_3 = 'port'
    var_4 = 3306
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = '{"license": ["MIT", "Apache"]}'
    var_2 = 'license'
    var_3 = 'GPL'
    var_4 = {var_2: var_3}
    var_5 = module_0.generate_context(var_0, extra_context=var_4)
    var_6 = bool(False)
    assert var_6 is True

def test_case_0():
    var_0 = 'custom.json'
    var_1 = '{"project_name": "my_project"}'
    var_2 = 'custom'

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = '{"license": ["MIT", "Apache"]}'
    var_2 = 'license'
    var_3 = 'GPL'
    var_4 = {var_2: var_3}



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_render_and_create_dir_raises_on_empty_dirname. Retrieved 5/12 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test that EmptyDirNameException is raised when dirname is empty string.'
    var_1 = {}
    var_2 = '/tmp'
    var_3 = [var_2]
    var_4 = module_0.Environment()
    var_5 = ''
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #19
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test that line 57 predicate evaluates to False when boolean conversion succeeds.'
    var_1 = 'flag'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = 'yes'
    var_5 = {var_1: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)
    var_7 = var_3['flag']
    assert var_7 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test that line 57 predicate evaluates to False when converting to False.'
    var_1 = 'flag'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = 'no'
    var_5 = {var_1: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)
    var_7 = var_3['flag']
    assert var_7 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test that line 57 predicate evaluates to False when converting zero string.'
    var_1 = 'enabled'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = '0'
    var_5 = {var_1: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)
    var_7 = var_3['enabled']
    assert var_7 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = "Test that line 57 predicate evaluates to False when converting 'false'."
    var_1 = 'active'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = 'false'
    var_5 = {var_1: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)
    var_7 = var_3['active']
    assert var_7 is False



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_generate_files_basic. Retrieved 14/35 statements.
# Partially parsed test_generate_files_with_subdirectories. Retrieved 14/34 statements.
# Partially parsed test_generate_files_empty_context. Retrieved 8/23 statements.
# Partially parsed test_generate_files_overwrite_if_exists. Retrieved 16/37 statements.
# Partially parsed test_generate_files_skip_if_file_exists. Retrieved 16/33 statements.
# Partially parsed test_generate_files_binary_file. Retrieved 15/37 statements.
# Partially parsed test_generate_files_copy_without_render. Retrieved 19/43 statements.


def test_case_0():
    var_0 = 'Test generate_files with basic template structure.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'file.txt'
    var_4 = 'Hello {{cookiecutter.project_name}}'
    var_5 = 'output'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'myproject'
    var_9 = {var_7: var_8}
    var_10 = (var_6, var_9)
    var_11 = [var_10]
    var_12 = [var_11]
    var_13 = 'cookiecutter.generate.accept_hooks'
    var_14 = False
    var_15 = 'myproject'

def test_case_0():
    var_0 = 'Test generate_files with subdirectories.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'src'
    var_4 = 'main.py'
    var_5 = '# {{cookiecutter.project_name}}'
    var_6 = 'output'
    var_7 = 'cookiecutter'
    var_8 = 'project_name'
    var_9 = 'myapp'
    var_10 = {var_8: var_9}
    var_11 = (var_7, var_10)
    var_12 = [var_11]
    var_13 = [var_12]
    var_14 = False

def test_case_0():
    var_0 = 'Test generate_files with empty context.'
    var_1 = 'repo'
    var_2 = 'mytemplate'
    var_3 = 'README.md'
    var_4 = 'Static content'
    var_5 = 'output'
    var_6 = None
    var_7 = False

def test_case_0():
    var_0 = 'Test generate_files with overwrite_if_exists flag.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.name}}'
    var_3 = 'file.txt'
    var_4 = 'New content'
    var_5 = 'output'
    var_6 = 'myproject'
    var_7 = 'old_file.txt'
    var_8 = 'Old content'
    var_9 = 'cookiecutter'
    var_10 = 'name'
    var_11 = {var_10: var_6}
    var_12 = (var_9, var_11)
    var_13 = [var_12]
    var_14 = [var_13]
    var_15 = True
    var_16 = False

def test_case_0():
    var_0 = 'Test generate_files with skip_if_file_exists flag.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project}}'
    var_3 = 'config.txt'
    var_4 = 'Config {{cookiecutter.version}}'
    var_5 = 'output'
    var_6 = 'cookiecutter'
    var_7 = 'project'
    var_8 = 'version'
    var_9 = 'app'
    var_10 = '1.0'
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = (var_6, var_11)
    var_13 = [var_12]
    var_14 = [var_13]
    var_15 = True
    var_16 = False

def test_case_0():
    var_0 = 'Test generate_files with binary file.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.name}}'
    var_3 = 'image.bin'
    var_4 = b'\x89PNG\r\n\x1a\n'
    var_5 = 'text.txt'
    var_6 = 'Content'
    var_7 = 'output'
    var_8 = 'cookiecutter'
    var_9 = 'name'
    var_10 = 'project'
    var_11 = {var_9: var_10}
    var_12 = (var_8, var_11)
    var_13 = [var_12]
    var_14 = [var_13]
    var_15 = False

def test_case_0():
    var_0 = 'Test generate_files with _copy_without_render setting.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.name}}'
    var_3 = 'static'
    var_4 = 'style.css'
    var_5 = 'body { color: {{cookiecutter.color}}; }'
    var_6 = 'output'
    var_7 = 'cookiecutter'
    var_8 = 'name'
    var_9 = 'color'
    var_10 = '_copy_without_render'
    var_11 = 'webapp'
    var_12 = 'blue'
    var_13 = 'static/*'
    var_14 = [var_13]
    var_15 = {var_8: var_11, var_9: var_12, var_10: var_14}
    var_16 = (var_7, var_15)
    var_17 = [var_16]
    var_18 = [var_17]
    var_19 = False
    var_20 = '{{cookiecutter.color}}'

def test_case_0():
    var_0 = 'Test generate_files with _new_lines configuration.'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_render_and_create_dir_with_empty_dirname. Retrieved 3/8 statements.
# Partially parsed test_render_and_create_dir_creates_new_directory. Retrieved 3/7 statements.
# Partially parsed test_render_and_create_dir_with_template_rendering. Retrieved 5/9 statements.
# Partially parsed test_render_and_create_dir_exists_no_overwrite. Retrieved 4/11 statements.
# Partially parsed test_render_and_create_dir_exists_with_overwrite. Retrieved 4/10 statements.
# Partially parsed test_render_and_create_dir_creates_nested_directories. Retrieved 3/7 statements.
# Partially parsed test_render_and_create_dir_with_context_variables. Retrieved 7/11 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = ''
    var_3 = bool(False)
    assert var_3 is True

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = 'test_dir'

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'project_name'
    var_2 = 'my_project'
    var_3 = {var_1: var_2}
    var_4 = '{{project_name}}_dir'

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = 'existing_dir'
    var_3 = False
    var_4 = bool(False)
    assert var_4 is True

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = 'existing_dir'
    var_3 = True

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = 'parent/child/grandchild'

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'name'
    var_2 = 'version'
    var_3 = 'test'
    var_4 = '1.0'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = '{{name}}_v{{version}}'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_render_and_create_dir_predicate_line_24_evaluates_to_true. Retrieved 5/12 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'existing_project'
    var_1 = True
    var_2 = {}
    var_3 = module_0.Environment()
    var_4 = 'existing_project'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_generate_context_opens_file_with_utf8_encoding. Retrieved 9/16 statements.


import json as module_0

def test_case_0():
    var_0 = 'Test that generate_context opens the context file with utf-8 encoding.'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'author'
    var_4 = 'test'
    var_5 = 'Test Author'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = {}
    var_8 = module_0.dumps(var_6, **var_7)
    var_9 = 'utf-8'
    var_10 = 'cookiecutter'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_render_and_create_dir_predicate_line_24_true. Retrieved 7/14 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 24 evaluates to True when directory exists.'
    var_1 = 'existing_project'
    var_2 = True
    var_3 = 'project_name'
    var_4 = {var_3: var_1}
    var_5 = module_0.Environment()
    var_6 = '{{ project_name }}'



# Parsed testcases at query #25
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test that boolean conversion succeeds and InvalidResponse is not raised.'
    var_1 = 'flag'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = 'yes'
    var_5 = {var_1: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)
    var_7 = var_3['flag']
    assert var_7 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = "Test that boolean conversion with 'no' succeeds."
    var_1 = 'flag'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = 'no'
    var_5 = {var_1: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)
    var_7 = var_3['flag']
    assert var_7 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = "Test that boolean conversion with '0' succeeds."
    var_1 = 'enabled'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = '0'
    var_5 = {var_1: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)
    var_7 = var_3['enabled']
    assert var_7 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = "Test that boolean conversion with '1' succeeds."
    var_1 = 'enabled'
    var_2 = False
    var_3 = {var_1: var_2}
    var_4 = '1'
    var_5 = {var_1: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)
    var_7 = var_3['enabled']
    assert var_7 is True



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_delete_project_on_failure_predicate_true_when_output_directory_created_and_keep_project_false. Retrieved 2/4 statements.
# Partially parsed test_delete_project_on_failure_predicate_false_when_output_directory_not_created. Retrieved 2/4 statements.
# Partially parsed test_delete_project_on_failure_predicate_false_when_keep_project_true. Retrieved 2/4 statements.
# Partially parsed test_delete_project_on_failure_predicate_false_when_both_conditions_false. Retrieved 2/4 statements.


def test_case_0():
    var_0 = True
    var_1 = False

def test_case_0():
    var_0 = False
    var_1 = False

def test_case_0():
    var_0 = True
    var_1 = True

def test_case_0():
    var_0 = False
    var_1 = True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_generate_context_opens_file_with_utf8_encoding. Retrieved 9/17 statements.


import json as module_0

def test_case_0():
    var_0 = 'Test that generate_context opens the context file with utf-8 encoding.'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'author'
    var_4 = 'test_project'
    var_5 = 'test_author'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = {}
    var_8 = module_0.dumps(var_6, **var_7)
    var_9 = 'utf-8'
    var_10 = 'cookiecutter'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_predicate_at_line_62_evaluates_to_false. Retrieved 3/14 statements.


def test_case_0():
    var_0 = "Test that the predicate at line 62 (for root, dirs, files in os.walk('.')) evaluates to False."
    var_1 = '.'
    var_2 = 0



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_generate_files_basic. Retrieved 14/39 statements.
# Partially parsed test_generate_files_with_hooks. Retrieved 12/35 statements.
# Partially parsed test_generate_files_overwrite_existing. Retrieved 18/49 statements.
# Partially parsed test_generate_files_skip_if_exists. Retrieved 15/40 statements.
# Partially parsed test_generate_files_with_copy_without_render. Retrieved 15/32 statements.


def test_case_0():
    var_0 = 'Test generate_files with basic template structure.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'test.txt'
    var_4 = 'Project: {{cookiecutter.project_name}}'
    var_5 = 'output'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'my_project'
    var_9 = {var_7: var_8}
    var_10 = (var_6, var_9)
    var_11 = [var_10]
    var_12 = [var_11]
    var_13 = None
    var_14 = False
    var_15 = 'my_project'

def test_case_0():
    var_0 = 'Test generate_files calls hooks when accept_hooks is True.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.name}}'
    var_3 = 'output'
    var_4 = 'cookiecutter'
    var_5 = 'name'
    var_6 = 'test_project'
    var_7 = {var_5: var_6}
    var_8 = (var_4, var_7)
    var_9 = [var_8]
    var_10 = [var_9]
    var_11 = []
    var_12 = True
    var_13 = 'pre_gen_project'
    var_14 = bool('pre_gen_project' in var_11)
    assert var_14 is True
    var_15 = 'post_gen_project'
    var_16 = bool('post_gen_project' in var_11)
    assert var_16 is True

def test_case_0():
    var_0 = 'Test generate_files with overwrite_if_exists=True.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.proj}}'
    var_3 = 'file.txt'
    var_4 = 'content'
    var_5 = 'output'
    var_6 = 'my_proj'
    var_7 = 'old_file.txt'
    var_8 = 'old content'
    var_9 = 'cookiecutter'
    var_10 = 'proj'
    var_11 = {var_10: var_6}
    var_12 = (var_9, var_11)
    var_13 = [var_12]
    var_14 = [var_13]
    var_15 = None
    var_16 = True
    var_17 = False
    var_18 = bool(var_3)
    assert var_18 is True
    var_19 = 'file.txt'
    var_20 = bool(var_5)
    assert var_20 is True

def test_case_0():
    var_0 = 'Test generate_files with skip_if_file_exists=True.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.name}}'
    var_3 = 'existing.txt'
    var_4 = 'new content'
    var_5 = 'output'
    var_6 = 'cookiecutter'
    var_7 = 'name'
    var_8 = 'project'
    var_9 = {var_7: var_8}
    var_10 = (var_6, var_9)
    var_11 = [var_10]
    var_12 = [var_11]
    var_13 = None
    var_14 = True
    var_15 = False
    var_16 = bool(var_3)
    assert var_16 is True

def test_case_0():
    var_0 = 'Test generate_files with _copy_without_render context setting.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.name}}'
    var_3 = 'static'
    var_4 = 'file.txt'
    var_5 = '{{not_rendered}}'
    var_6 = 'output'
    var_7 = 'cookiecutter'
    var_8 = 'name'
    var_9 = '_copy_without_render'
    var_10 = 'myproject'
    var_11 = [var_3]
    var_12 = {var_8: var_10, var_9: var_11}
    var_13 = (var_7, var_12)
    var_14 = [var_13]
    var_15 = [var_14]



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_undefined_error_caught_at_line_36. Retrieved 13/30 statements.


def test_case_0():
    var_0 = 'Test that UndefinedError at line 36 is caught and re-raised as UndefinedVariableInTemplate.'
    var_1 = 'repo'
    var_2 = var_0 / var_1
    var_3 = '{{cookiecutter.project_name}}'
    var_4 = var_2 / var_3
    var_5 = 'cookiecutter'
    var_6 = 'project_name'
    var_7 = '{{ undefined_variable }}'
    var_8 = {var_6: var_7}
    var_9 = (var_5, var_8)
    var_10 = [var_9]
    var_11 = [var_10]
    var_12 = 'output'
    var_13 = False
    var_14 = bool(False)
    assert var_14 is True
    var_15 = 'Unable to create project directory'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_generate_context_handles_json_decode_error. Retrieved 2/12 statements.


def test_case_0():
    var_0 = 'Test that ValueError on line 20 is caught and re-raised as ContextDecodingException.'
    var_1 = '{ invalid json content }'
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'JSON decoding error'



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_generate_file_renders_text_file. Retrieved 17/42 statements.
# Partially parsed test_generate_file_skips_existing_file. Retrieved 14/36 statements.
# Partially parsed test_generate_file_copies_binary_file. Retrieved 12/30 statements.
# Partially parsed test_generate_file_returns_on_empty_filename. Retrieved 10/25 statements.
# Partially parsed test_generate_file_uses_configured_newline. Retrieved 12/30 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project'
    assert var_0 == 'Hello world'
    var_1 = True
    var_2 = 'test_{{cookiecutter.name}}.txt'
    var_3 = 'test_{{cookiecutter.name}}.txt'
    var_4 = 'Hello {{cookiecutter.name}}'
    var_5 = 'cookiecutter'
    var_6 = 'name'
    var_7 = '_new_lines'
    var_8 = 'world'
    var_9 = False
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = {var_5: var_10}
    var_12 = module_0.Environment()
    var_13 = 'os.chdir'
    var_14 = None
    var_15 = 'shutil.copymode'
    var_16 = 'test_world.txt'

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project'
    var_1 = True
    var_2 = 'test.txt'
    var_3 = 'test.txt'
    var_4 = 'Original content'
    var_5 = 'Existing content'
    var_6 = 'cookiecutter'
    var_7 = '_new_lines'
    var_8 = False
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = module_0.Environment()
    var_12 = 'shutil.copymode'
    var_13 = True
    assert var_13 == 'Existing content'

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project'
    var_1 = True
    var_2 = 'binary.bin'
    var_3 = 'binary.bin'
    var_4 = b'\x89PNG\r\n\x1a\n'
    var_5 = 'cookiecutter'
    var_6 = {}
    var_7 = {var_5: var_6}
    var_8 = module_0.Environment()
    var_9 = 'cookiecutter.generate.is_binary'
    var_10 = 'shutil.copyfile'
    var_11 = 'shutil.copymode'

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project'
    var_1 = True
    var_2 = '{{cookiecutter.skip_dir}}'
    var_3 = '{{cookiecutter.skip_dir}}'
    var_4 = 'cookiecutter'
    var_5 = 'skip_dir'
    var_6 = ''
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = module_0.Environment()

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project'
    var_1 = True
    var_2 = 'test.txt'
    var_3 = 'test.txt'
    var_4 = 'Line1\nLine2\n'
    var_5 = 'cookiecutter'
    var_6 = '_new_lines'
    var_7 = '\r\n'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = module_0.Environment()
    var_11 = 'shutil.copymode'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_generate_context_basic. Retrieved 3/7 statements.
# Partially parsed test_generate_context_with_default_context. Retrieved 6/10 statements.
# Partially parsed test_generate_context_with_extra_context. Retrieved 6/10 statements.
# Partially parsed test_generate_context_with_both_defaults_and_extra. Retrieved 9/13 statements.
# Partially parsed test_generate_context_invalid_json. Retrieved 4/8 statements.
# Partially parsed test_generate_context_choice_variable. Retrieved 6/10 statements.
# Partially parsed test_generate_context_boolean_variable. Retrieved 6/10 statements.
# Partially parsed test_generate_context_nested_dict. Retrieved 5/11 statements.
# Partially parsed test_generate_context_multichoice_variable. Retrieved 9/15 statements.
# Partially parsed test_generate_context_invalid_choice_value. Retrieved 7/11 statements.
# Partially parsed test_generate_context_invalid_boolean_string. Retrieved 7/11 statements.
# Partially parsed test_generate_context_file_stem. Retrieved 3/7 statements.
# Partially parsed test_generate_context_ordered_dict. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 'Test generate_context loads a basic JSON file.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "my_project", "author": "John"}'
    var_3 = 'cookiecutter'

def test_case_0():
    var_0 = 'Test generate_context applies default_context overwrites.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "my_project", "author": "John"}'
    var_3 = 'author'
    var_4 = 'Jane'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test generate_context applies extra_context overwrites.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "my_project", "author": "John"}'
    var_3 = 'project_name'
    var_4 = 'new_project'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test generate_context applies both default and extra context.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "my_project", "author": "John", "version": "1.0"}'
    var_3 = 'author'
    var_4 = 'Jane'
    var_5 = {var_3: var_4}
    var_6 = 'version'
    var_7 = '2.0'
    var_8 = {var_6: var_7}

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_context raises ContextDecodingException for invalid JSON.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "my_project"'
    var_3 = module_0.generate_context(var_0)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'JSON decoding error'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_context raises error for nonexistent file.'
    var_1 = '/nonexistent/path/cookiecutter.json'
    var_2 = module_0.generate_context(var_1)
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'Test generate_context with choice variable in extra_context.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"flavor": ["vanilla", "chocolate", "strawberry"]}'
    var_3 = 'flavor'
    var_4 = 'chocolate'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test generate_context with boolean variable in extra_context.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"use_docker": true}'
    var_3 = 'use_docker'
    var_4 = 'false'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test generate_context with nested dictionary in extra_context.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"config": {"debug": true, "port": 8000}}'
    var_3 = 'config'
    var_4 = 'debug'

def test_case_0():
    var_0 = 'Test generate_context with multichoice variable in extra_context.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"features": ["feature1", "feature2", "feature3"]}'
    var_3 = 'features'
    var_4 = 'feature2'
    var_5 = 'feature3'
    var_6 = [var_4, var_5]
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_context with invalid choice value raises ValueError.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"flavor": ["vanilla", "chocolate"]}'
    var_3 = 'flavor'
    var_4 = 'mint'
    var_5 = {var_3: var_4}
    var_6 = module_0.generate_context(var_0, extra_context=var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'mint'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_context with invalid boolean string raises ValueError.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"use_docker": true}'
    var_3 = 'use_docker'
    var_4 = 'maybe'
    var_5 = {var_3: var_4}
    var_6 = module_0.generate_context(var_0, extra_context=var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'could not be converted to a boolean'

def test_case_0():
    var_0 = 'Test generate_context uses correct file stem as context key.'
    var_1 = 'custom_template.json'
    var_2 = '{"name": "test"}'
    var_3 = 'custom_template'

def test_case_0():
    var_0 = 'Test generate_context preserves order with OrderedDict.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"z_field": 1, "a_field": 2, "m_field": 3}'
    var_3 = 'cookiecutter'



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_generate_files_with_minimal_context. Retrieved 13/32 statements.
# Partially parsed test_generate_files_empty_dirname_raises_exception. Retrieved 9/23 statements.
# Partially parsed test_generate_files_with_existing_output_dir_no_overwrite. Retrieved 11/27 statements.
# Partially parsed test_generate_files_with_overwrite_if_exists. Retrieved 16/39 statements.
# Partially parsed test_generate_files_skip_if_file_exists. Retrieved 13/31 statements.
# Partially parsed test_generate_files_with_subdirectories. Retrieved 14/34 statements.
# Partially parsed test_generate_files_with_binary_file. Retrieved 6/17 statements.


def test_case_0():
    var_0 = 'Test generate_files with minimal context.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'README.md'
    var_4 = '# {{cookiecutter.project_name}}'
    var_5 = 'output'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'my_project'
    var_9 = {var_7: var_8}
    var_10 = (var_6, var_9)
    var_11 = [var_10]
    var_12 = [var_11]
    var_13 = False
    var_14 = 'my_project'

def test_case_0():
    var_0 = 'Test that empty directory name raises EmptyDirNameException.'
    var_1 = 'repo'
    var_2 = ''
    var_3 = 'output'
    var_4 = 'cookiecutter'
    var_5 = {}
    var_6 = (var_4, var_5)
    var_7 = [var_6]
    var_8 = [var_7]
    var_9 = False
    var_10 = bool(False)
    assert var_10 is True

def test_case_0():
    var_0 = 'Test that existing output directory raises exception when overwrite is False.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'output'
    var_4 = 'my_project'
    var_5 = 'cookiecutter'
    var_6 = 'project_name'
    var_7 = {var_6: var_4}
    var_8 = (var_5, var_7)
    var_9 = [var_8]
    var_10 = [var_9]
    var_11 = False
    var_12 = bool(False)
    assert var_12 is True

def test_case_0():
    var_0 = 'Test generate_files with overwrite_if_exists=True.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'README.md'
    var_4 = '# {{cookiecutter.project_name}}'
    var_5 = 'output'
    var_6 = 'my_project'
    var_7 = 'old.txt'
    var_8 = 'old content'
    var_9 = 'cookiecutter'
    var_10 = 'project_name'
    var_11 = {var_10: var_6}
    var_12 = (var_9, var_11)
    var_13 = [var_12]
    var_14 = [var_13]
    var_15 = True
    var_16 = False
    var_17 = 'my_project'

def test_case_0():
    var_0 = 'Test generate_files with skip_if_file_exists=True.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'existing.txt'
    var_4 = 'new content from template'
    var_5 = 'output'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'my_project'
    var_9 = {var_7: var_8}
    var_10 = (var_6, var_9)
    var_11 = [var_10]
    var_12 = [var_11]
    var_13 = False

def test_case_0():
    var_0 = 'Test generate_files with nested directory structure.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'src'
    var_4 = 'main.py'
    var_5 = '# {{cookiecutter.project_name}}'
    var_6 = 'output'
    var_7 = 'cookiecutter'
    var_8 = 'project_name'
    var_9 = 'my_project'
    var_10 = {var_8: var_9}
    var_11 = (var_7, var_10)
    var_12 = [var_11]
    var_13 = [var_12]
    var_14 = False

def test_case_0():
    var_0 = 'Test generate_files with binary files.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'image.png'
    var_4 = b'\x89PNG\r\n\x1a\n'
    var_5 = 'output'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_generate_files_skips_hooks_when_accept_hooks_is_false. Retrieved 13/27 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test that pre_gen_project hook is not run when accept_hooks is False.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'file.txt'
    var_4 = 'test'
    var_5 = 'cookiecutter'
    var_6 = 'project_name'
    var_7 = 'test_project'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = 'output'
    var_11 = False
    var_12 = module_0.generate_files(var_0, var_9, var_1, accept_hooks=var_11)



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_generate_files_basic. Retrieved 15/40 statements.
# Partially parsed test_generate_files_with_subdirectories. Retrieved 14/38 statements.
# Partially parsed test_generate_files_overwrite_if_exists. Retrieved 14/34 statements.
# Partially parsed test_generate_files_skip_if_file_exists. Retrieved 17/38 statements.
# Partially parsed test_generate_files_empty_context. Retrieved 6/17 statements.
# Partially parsed test_generate_files_with_copy_without_render. Retrieved 16/33 statements.
# Partially parsed test_generate_files_default_output_dir. Retrieved 9/17 statements.


def test_case_0():
    var_0 = 'Test generate_files with basic template structure.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'README.md'
    var_4 = '# {{cookiecutter.project_name}}'
    var_5 = 'output'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'my_project'
    var_9 = {var_7: var_8}
    var_10 = (var_6, var_9)
    var_11 = [var_10]
    var_12 = [var_11]
    var_13 = 'COOKIECUTTER_ACCEPT_HOOKS'
    var_14 = 'false'
    var_15 = False

def test_case_0():
    var_0 = 'Test generate_files with nested directory structure.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'src'
    var_4 = 'main.py'
    var_5 = '# {{cookiecutter.project_name}}'
    var_6 = 'output'
    var_7 = 'cookiecutter'
    var_8 = 'project_name'
    var_9 = 'test_app'
    var_10 = {var_8: var_9}
    var_11 = (var_7, var_10)
    var_12 = [var_11]
    var_13 = [var_12]
    var_14 = False

def test_case_0():
    var_0 = 'Test generate_files with overwrite_if_exists flag.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'output'
    var_4 = 'my_project'
    var_5 = 'old_file.txt'
    var_6 = 'old content'
    var_7 = 'cookiecutter'
    var_8 = 'project_name'
    var_9 = {var_8: var_4}
    var_10 = (var_7, var_9)
    var_11 = [var_10]
    var_12 = [var_11]
    var_13 = True
    var_14 = False

def test_case_0():
    var_0 = 'Test generate_files with skip_if_file_exists flag.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'config.txt'
    var_4 = 'config={{cookiecutter.value}}'
    var_5 = 'output'
    var_6 = 'my_project'
    var_7 = 'existing config'
    var_8 = 'cookiecutter'
    var_9 = 'project_name'
    var_10 = 'value'
    var_11 = 'new'
    var_12 = {var_9: var_6, var_10: var_11}
    var_13 = (var_8, var_12)
    var_14 = [var_13]
    var_15 = [var_14]
    var_16 = True
    var_17 = False

def test_case_0():
    var_0 = 'Test generate_files with empty context.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'output'
    var_4 = None
    var_5 = False

def test_case_0():
    var_0 = 'Test generate_files with _copy_without_render setting.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'image.bin'
    var_4 = b'\x89PNG\r\n\x1a\n'
    var_5 = 'output'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = '_copy_without_render'
    var_9 = 'my_project'
    var_10 = '*.bin'
    var_11 = [var_10]
    var_12 = {var_7: var_9, var_8: var_11}
    var_13 = (var_6, var_12)
    var_14 = [var_13]
    var_15 = [var_14]
    var_16 = False

def test_case_0():
    var_0 = 'Test generate_files with default output directory.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = (var_3, var_6)
    var_8 = [var_7]
    var_9 = [var_8]



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_generate_context_default_context_predicate. Retrieved 7/20 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 38 (if default_context:) evaluates to True.'
    var_1 = 'project_name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = 'project_name'
    var_5 = 'custom_name'
    var_6 = {var_4: var_5}
    var_7 = 'cookiecutter'



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_generate_files_with_valid_context. Retrieved 13/28 statements.
# Partially parsed test_generate_files_with_binary_file. Retrieved 13/28 statements.
# Partially parsed test_generate_files_with_copy_without_render. Retrieved 15/30 statements.
# Partially parsed test_generate_files_skip_if_file_exists. Retrieved 14/32 statements.
# Partially parsed test_generate_files_with_subdirectories. Retrieved 16/33 statements.
# Partially parsed test_generate_files_overwrite_if_exists. Retrieved 14/32 statements.
# Partially parsed test_generate_files_with_newline_configuration. Retrieved 3/11 statements.


def test_case_0():
    var_0 = 'Test generate_files with a valid context and template structure.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'README.md'
    var_4 = '# {{cookiecutter.project_name}}\n'
    var_5 = 'output'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'my_project'
    var_9 = {var_7: var_8}
    var_10 = (var_6, var_9)
    var_11 = [var_10]
    var_12 = [var_11]
    var_13 = False
    var_14 = 'my_project'

def test_case_0():
    var_0 = 'Test generate_files with a binary file in the template.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'image.bin'
    var_4 = b'\x89PNG\r\n\x1a\n'
    var_5 = 'output'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'my_project'
    var_9 = {var_7: var_8}
    var_10 = (var_6, var_9)
    var_11 = [var_10]
    var_12 = [var_11]
    var_13 = False

def test_case_0():
    var_0 = 'Test generate_files with _copy_without_render setting.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'config.txt'
    var_4 = '{{not_rendered}}\n'
    var_5 = 'output'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = '_copy_without_render'
    var_9 = 'my_project'
    var_10 = [var_3]
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = (var_6, var_11)
    var_13 = [var_12]
    var_14 = [var_13]
    var_15 = False

def test_case_0():
    var_0 = 'Test generate_files with skip_if_file_exists option.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'file.txt'
    var_4 = 'Original content'
    var_5 = 'output'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'my_project'
    var_9 = {var_7: var_8}
    var_10 = (var_6, var_9)
    var_11 = [var_10]
    var_12 = [var_11]
    var_13 = False
    var_14 = True

def test_case_0():
    var_0 = 'Test generate_files with nested directory structure.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = '{{cookiecutter.src_dir}}'
    var_4 = 'main.py'
    var_5 = '# {{cookiecutter.project_name}}\n'
    var_6 = 'output'
    var_7 = 'cookiecutter'
    var_8 = 'project_name'
    var_9 = 'src_dir'
    var_10 = 'my_project'
    var_11 = 'src'
    var_12 = {var_8: var_10, var_9: var_11}
    var_13 = (var_7, var_12)
    var_14 = [var_13]
    var_15 = [var_14]
    var_16 = False

def test_case_0():
    var_0 = 'Test generate_files with overwrite_if_exists option.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'file.txt'
    var_4 = 'Content'
    var_5 = 'output'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'my_project'
    var_9 = {var_7: var_8}
    var_10 = (var_6, var_9)
    var_11 = [var_10]
    var_12 = [var_11]
    var_13 = False
    var_14 = True

def test_case_0():
    var_0 = 'Test generate_files with _new_lines configuration.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_generate_files_basic. Retrieved 14/28 statements.
# Partially parsed test_generate_files_with_context. Retrieved 15/31 statements.
# Partially parsed test_generate_files_overwrite_if_exists. Retrieved 13/28 statements.
# Partially parsed test_generate_files_skip_if_file_exists. Retrieved 13/26 statements.
# Partially parsed test_generate_files_with_subdirectories. Retrieved 13/29 statements.
# Partially parsed test_generate_files_empty_context. Retrieved 7/16 statements.
# Partially parsed test_generate_files_default_output_dir. Retrieved 11/20 statements.
# Partially parsed test_generate_files_binary_file. Retrieved 12/28 statements.
# Partially parsed test_generate_files_with_copy_without_render. Retrieved 15/27 statements.


def test_case_0():
    var_0 = 'Test generate_files with basic template structure.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'test.txt'
    var_4 = 'Hello {{cookiecutter.project_name}}'
    var_5 = 'output'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'my_project'
    var_9 = {var_7: var_8}
    var_10 = (var_6, var_9)
    var_11 = [var_10]
    var_12 = [var_11]
    var_13 = 'cookiecutter.generate.accept_hooks'
    var_14 = False
    var_15 = 'my_project'

def test_case_0():
    var_0 = 'Test generate_files renders context variables correctly.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.name}}'
    var_3 = '{{cookiecutter.filename}}.txt'
    var_4 = 'Project: {{cookiecutter.name}}'
    var_5 = 'output'
    var_6 = 'cookiecutter'
    var_7 = 'name'
    var_8 = 'filename'
    var_9 = 'test_project'
    var_10 = 'readme'
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = (var_6, var_11)
    var_13 = [var_12]
    var_14 = [var_13]
    var_15 = 'readme.txt'
    var_16 = 'Project: test_project'

def test_case_0():
    var_0 = 'Test generate_files with overwrite_if_exists flag.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project}}'
    var_3 = 'file.txt'
    var_4 = 'content'
    var_5 = 'output'
    var_6 = 'myproject'
    var_7 = 'cookiecutter'
    var_8 = 'project'
    var_9 = {var_8: var_6}
    var_10 = (var_7, var_9)
    var_11 = [var_10]
    var_12 = [var_11]
    var_13 = True

def test_case_0():
    var_0 = 'Test generate_files with skip_if_file_exists flag.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.proj}}'
    var_3 = 'existing.txt'
    var_4 = 'template content'
    var_5 = 'output'
    var_6 = 'cookiecutter'
    var_7 = 'proj'
    var_8 = 'project'
    var_9 = {var_7: var_8}
    var_10 = (var_6, var_9)
    var_11 = [var_10]
    var_12 = [var_11]
    var_13 = False

def test_case_0():
    var_0 = 'Test generate_files with nested directory structure.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.name}}'
    var_3 = 'src'
    var_4 = 'main.py'
    var_5 = '# {{cookiecutter.name}}'
    var_6 = 'output'
    var_7 = 'cookiecutter'
    var_8 = 'name'
    var_9 = 'myapp'
    var_10 = {var_8: var_9}
    var_11 = (var_7, var_10)
    var_12 = [var_11]
    var_13 = [var_12]

def test_case_0():
    var_0 = 'Test generate_files with empty context.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'file.txt'
    var_4 = 'content'
    var_5 = 'output'
    var_6 = None

def test_case_0():
    var_0 = 'Test generate_files with default output directory.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.name}}'
    var_3 = 'file.txt'
    var_4 = 'test'
    var_5 = 'cookiecutter'
    var_6 = 'name'
    var_7 = 'testproj'
    var_8 = {var_6: var_7}
    var_9 = (var_5, var_8)
    var_10 = [var_9]
    var_11 = [var_10]
    var_12 = 'testproj'

def test_case_0():
    var_0 = 'Test generate_files handles binary files correctly.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.name}}'
    var_3 = 'image.bin'
    var_4 = b'\x89PNG\r\n\x1a\n'
    var_5 = 'output'
    var_6 = 'cookiecutter'
    var_7 = 'name'
    var_8 = 'project'
    var_9 = {var_7: var_8}
    var_10 = (var_6, var_9)
    var_11 = [var_10]
    var_12 = [var_11]

def test_case_0():
    var_0 = 'Test generate_files with _copy_without_render setting.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.name}}'
    var_3 = 'static'
    var_4 = '{{cookiecutter.name}}.txt'
    var_5 = 'output'
    var_6 = 'cookiecutter'
    var_7 = 'name'
    var_8 = '_copy_without_render'
    var_9 = 'project'
    var_10 = 'static/*'
    var_11 = [var_10]
    var_12 = {var_7: var_9, var_8: var_11}
    var_13 = (var_6, var_12)
    var_14 = [var_13]
    var_15 = [var_14]



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_file_name_is_empty_predicate_true. Retrieved 8/22 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project'
    var_1 = 'subdir'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = 'subdir'
    var_7 = 'dummy content'



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_template_syntax_error_translated_false. Retrieved 7/18 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'test_template.txt'
    var_1 = '{{ unclosed_variable'
    var_2 = module_0.Environment()
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = None
    var_7 = bool(var_6 is not None)
    assert var_7 is True
    var_8 = var_6.translated
    assert var_8 is False



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_generate_file_binary_file. Retrieved 11/32 statements.
# Partially parsed test_generate_file_skip_if_exists. Retrieved 13/32 statements.
# Partially parsed test_generate_file_text_file. Retrieved 13/36 statements.
# Partially parsed test_generate_file_empty_filename. Retrieved 9/26 statements.
# Partially parsed test_generate_file_with_custom_newline. Retrieved 14/37 statements.


def test_case_0():
    var_0 = 'Test that binary files are copied without rendering.'
    var_1 = 'project'
    var_2 = True
    var_3 = 'binary_file.bin'
    var_4 = b'\x00\x01\x02\x03'
    var_5 = 'cookiecutter'
    var_6 = {}
    var_7 = {var_5: var_6}
    var_8 = 'generate_file.is_binary'
    var_9 = 'generate_file.shutil.copyfile'
    var_10 = 'generate_file.shutil.copymode'

def test_case_0():
    var_0 = 'Test that file generation is skipped if file exists and skip_if_file_exists is True.'
    var_1 = 'project'
    var_2 = True
    var_3 = 'template.txt'
    var_4 = 'test content'
    var_5 = 'output.txt'
    var_6 = 'existing content'
    var_7 = 'cookiecutter'
    var_8 = {}
    var_9 = {var_7: var_8}
    var_10 = 'generate_file.is_binary'
    var_11 = False
    var_12 = True

def test_case_0():
    var_0 = 'Test that text files are rendered and written.'
    var_1 = 'project'
    var_2 = True
    var_3 = 'template.txt'
    var_4 = 'Hello {{ name }}\n'
    var_5 = 'cookiecutter'
    var_6 = 'name'
    var_7 = 'World'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = 'generate_file.is_binary'
    var_11 = False
    var_12 = 'generate_file.shutil.copymode'

def test_case_0():
    var_0 = 'Test that generation is skipped when resulting filename is empty.'
    var_1 = 'project'
    var_2 = True
    var_3 = 'template.txt'
    var_4 = 'test'
    var_5 = 'cookiecutter'
    var_6 = {}
    var_7 = {var_5: var_6}
    var_8 = 'generate_file.is_binary'

def test_case_0():
    var_0 = 'Test that custom newline character from context is used.'
    var_1 = 'project'
    var_2 = True
    var_3 = 'template.txt'
    var_4 = 'line1\nline2\n'
    var_5 = 'cookiecutter'
    var_6 = '_new_lines'
    var_7 = '\r\n'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = 'generate_file.is_binary'
    var_11 = False
    var_12 = 'generate_file.shutil.copymode'
    var_13 = 'builtins.open'



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_generate_file_renders_text_file. Retrieved 15/34 statements.
# Partially parsed test_generate_file_copies_binary_file. Retrieved 16/35 statements.
# Partially parsed test_generate_file_skips_existing_file. Retrieved 14/30 statements.
# Partially parsed test_generate_file_returns_on_empty_filename. Retrieved 12/27 statements.
# Partially parsed test_generate_file_renders_filename_with_context. Retrieved 16/33 statements.
# Partially parsed test_generate_file_uses_configured_newline. Retrieved 17/39 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test that generate_file renders and writes text file content.'
    var_1 = 'project'
    var_2 = 'template'
    var_3 = 'test.txt'
    var_4 = 'Hello {{ name }}'
    assert var_4 == 'Hello World'
    var_5 = module_0.Environment()
    var_6 = 'name'
    var_7 = 'cookiecutter'
    var_8 = 'World'
    var_9 = {}
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = 'generate_file.is_binary'
    var_12 = False
    var_13 = 'os.path.isdir'
    var_14 = 'os.path.exists'

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test that generate_file copies binary files without rendering.'
    var_1 = 'project'
    var_2 = 'template'
    var_3 = 'test.bin'
    var_4 = b'\x89PNG\r\n\x1a\n'
    var_5 = module_0.Environment()
    var_6 = 'cookiecutter'
    var_7 = {}
    var_8 = {var_6: var_7}
    var_9 = 'generate_file.is_binary'
    var_10 = True
    var_11 = 'os.path.isdir'
    var_12 = False
    var_13 = 'os.path.exists'
    var_14 = 'shutil.copyfile'
    var_15 = 'shutil.copymode'

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test that generate_file skips file if skip_if_file_exists is True.'
    var_1 = 'project'
    var_2 = 'template'
    var_3 = 'test.txt'
    var_4 = 'content'
    var_5 = module_0.Environment()
    var_6 = 'cookiecutter'
    var_7 = {}
    var_8 = {var_6: var_7}
    var_9 = 'os.path.isdir'
    var_10 = False
    var_11 = 'os.path.exists'
    var_12 = True
    var_13 = 'generate_file.is_binary'

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test that generate_file returns early if resulting filename is empty.'
    var_1 = 'project'
    var_2 = 'template'
    var_3 = 'test.txt'
    var_4 = 'content'
    var_5 = module_0.Environment()
    var_6 = 'cookiecutter'
    var_7 = {}
    var_8 = {var_6: var_7}
    var_9 = 'os.path.isdir'
    var_10 = True
    var_11 = 'generate_file.is_binary'

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test that generate_file renders the output filename using context.'
    var_1 = 'project'
    var_2 = 'template'
    var_3 = '{{ filename }}.txt'
    var_4 = 'content'
    var_5 = module_0.Environment()
    var_6 = 'filename'
    var_7 = 'cookiecutter'
    var_8 = 'output'
    var_9 = {}
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = 'generate_file.is_binary'
    var_12 = False
    var_13 = 'os.path.isdir'
    var_14 = 'os.path.exists'
    var_15 = 'output.txt'

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test that generate_file uses configured newline from context.'
    var_1 = 'project'
    var_2 = 'template'
    var_3 = 'test.txt'
    var_4 = 'line1\nline2'
    var_5 = module_0.Environment()
    var_6 = 'cookiecutter'
    var_7 = '_new_lines'
    var_8 = '\r\n'
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = 'generate_file.is_binary'
    var_12 = False
    var_13 = 'os.path.isdir'
    var_14 = 'os.path.exists'
    var_15 = 'builtins.open'
    var_16 = 'w'



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_generate_context_file_not_found. Retrieved 3/9 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = "Test that the predicate at line 18 evaluates to False when file doesn't exist."
    var_1 = '/tmp/this_file_does_not_exist_12345.json'
    var_2 = module_0.generate_context(var_1)
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_predicate_at_line_67_evaluates_to_false. Retrieved 12/25 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'test_template.txt'
    var_1 = 'Hello {{ name }}'
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = 'World'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.Environment()
    var_8 = 'cookiecutter'
    var_9 = var_6[var_8]
    var_10 = '_new_lines'
    var_11 = False



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_template_syntax_error_has_translated_set_to_false. Retrieved 10/24 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = '{{ unclosed_variable'
    var_1 = 'template.txt'
    var_2 = module_0.Environment()
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = False
    var_7 = False
    assert var_7 is True
    var_8 = 'template.txt'
    var_9 = True
    assert var_9 is True



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_is_binary_predicate_evaluates_to_true. Retrieved 16/57 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'binary_file.bin'
    var_1 = b'\x89PNG\r\n\x1a\n'
    var_2 = 'project'
    var_3 = []
    var_4 = []
    var_5 = 'shutil.copyfile'
    var_6 = 'shutil.copymode'
    var_7 = '__main__'
    var_8 = '__main__'
    var_9 = module_0.Environment()
    var_10 = 'cookiecutter'
    var_11 = {}
    var_12 = {var_10: var_11}
    var_13 = 'binary_file.bin'
    var_14 = len(var_3)
    assert var_14 == 1
    var_15 = len(var_4)
    assert var_15 == 1



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_generate_file_renders_text_file. Retrieved 10/23 statements.
# Partially parsed test_generate_file_skips_if_file_exists. Retrieved 10/23 statements.
# Partially parsed test_generate_file_renders_filename. Retrieved 11/24 statements.
# Partially parsed test_generate_file_copies_binary_file. Retrieved 8/21 statements.
# Partially parsed test_generate_file_returns_early_for_empty_filename. Retrieved 8/20 statements.
# Partially parsed test_generate_file_uses_configured_newlines. Retrieved 10/23 statements.
# Partially parsed test_generate_file_detects_newlines. Retrieved 8/21 statements.
# Partially parsed test_generate_file_preserves_file_permissions. Retrieved 9/27 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'templates'
    var_1 = 'project'
    var_2 = 'test.txt'
    var_3 = 'Hello {{ name }}'
    var_4 = 'cookiecutter'
    var_5 = 'name'
    var_6 = 'World'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = module_0.Environment()

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'templates'
    var_1 = 'project'
    var_2 = 'test.txt'
    var_3 = 'Content'
    var_4 = 'Existing content'
    var_5 = 'cookiecutter'
    var_6 = {}
    var_7 = {var_5: var_6}
    var_8 = module_0.Environment()
    var_9 = True

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'templates'
    var_1 = 'project'
    var_2 = '{{name}}.txt'
    var_3 = 'Content'
    var_4 = 'cookiecutter'
    var_5 = 'name'
    var_6 = 'myfile'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = module_0.Environment()
    var_10 = 'myfile.txt'

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'templates'
    var_1 = 'project'
    var_2 = 'image.bin'
    var_3 = b'\x89PNG\r\n\x1a\n'
    var_4 = 'cookiecutter'
    var_5 = {}
    var_6 = {var_4: var_5}
    var_7 = module_0.Environment()

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'templates'
    var_1 = 'project'
    var_2 = 'test.txt'
    var_3 = 'Content'
    var_4 = 'cookiecutter'
    var_5 = {}
    var_6 = {var_4: var_5}
    var_7 = module_0.Environment()

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'templates'
    var_1 = 'project'
    var_2 = 'test.txt'
    var_3 = 'Line 1\nLine 2'
    var_4 = 'cookiecutter'
    var_5 = '_new_lines'
    var_6 = '\r\n'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = module_0.Environment()
    var_10 = b'Line 1\r\nLine 2'

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'templates'
    var_1 = 'project'
    var_2 = 'test.txt'
    var_3 = 'Line 1\nLine 2'
    var_4 = 'cookiecutter'
    var_5 = {}
    var_6 = {var_4: var_5}
    var_7 = module_0.Environment()

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'templates'
    var_1 = 'project'
    var_2 = 'test.txt'
    var_3 = 'Content'
    var_4 = 493
    var_5 = 'cookiecutter'
    var_6 = {}
    var_7 = {var_5: var_6}
    var_8 = module_0.Environment()



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_generate_file_renders_text_file. Retrieved 10/25 statements.
# Partially parsed test_generate_file_copies_binary_file. Retrieved 8/22 statements.
# Partially parsed test_generate_file_skips_existing_file. Retrieved 10/23 statements.
# Partially parsed test_generate_file_renders_filename. Retrieved 11/24 statements.
# Partially parsed test_generate_file_preserves_file_permissions. Retrieved 11/28 statements.
# Partially parsed test_generate_file_returns_early_for_empty_filename. Retrieved 7/20 statements.
# Partially parsed test_generate_file_uses_configured_newline. Retrieved 10/23 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'templates'
    var_1 = 'project'
    var_2 = 'test.txt'
    var_3 = 'Hello {{ name }}!'
    var_4 = module_0.Environment()
    var_5 = 'cookiecutter'
    var_6 = 'name'
    var_7 = 'World'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'templates'
    var_1 = 'project'
    var_2 = 'image.bin'
    var_3 = b'\x89PNG\r\n\x1a\n'
    var_4 = module_0.Environment()
    var_5 = 'cookiecutter'
    var_6 = {}
    var_7 = {var_5: var_6}

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'templates'
    var_1 = 'project'
    var_2 = 'test.txt'
    var_3 = 'Template content'
    var_4 = 'Existing content'
    var_5 = module_0.Environment()
    var_6 = 'cookiecutter'
    var_7 = {}
    var_8 = {var_6: var_7}
    var_9 = True

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'templates'
    var_1 = 'project'
    var_2 = '{{ project_name }}.txt'
    var_3 = 'Content'
    var_4 = module_0.Environment()
    var_5 = 'cookiecutter'
    var_6 = 'project_name'
    var_7 = 'myproject'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = 'myproject.txt'

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'templates'
    var_1 = 'project'
    var_2 = 'script.sh'
    var_3 = '#!/bin/bash\necho {{ message }}'
    var_4 = 493
    var_5 = module_0.Environment()
    var_6 = 'cookiecutter'
    var_7 = 'message'
    var_8 = 'hello'
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'templates'
    var_1 = 'project'
    var_2 = 'subdir'
    var_3 = module_0.Environment()
    var_4 = 'cookiecutter'
    var_5 = {}
    var_6 = {var_4: var_5}

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'templates'
    var_1 = 'project'
    var_2 = 'test.txt'
    var_3 = 'line1\nline2'
    var_4 = module_0.Environment()
    var_5 = 'cookiecutter'
    var_6 = '_new_lines'
    var_7 = '\r\n'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_generate_file_renders_text_file. Retrieved 17/41 statements.
# Partially parsed test_generate_file_copies_binary_file. Retrieved 11/33 statements.
# Partially parsed test_generate_file_skips_existing_file. Retrieved 13/33 statements.
# Partially parsed test_generate_file_returns_on_empty_filename. Retrieved 10/30 statements.
# Partially parsed test_generate_file_renders_template_with_context. Retrieved 22/46 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project'
    var_1 = True
    var_2 = 'test_{{name}}.txt'
    var_3 = '{{name}}'
    var_4 = 'file'
    var_5 = 'Hello {{name}}!'
    var_6 = 'utf-8'
    var_7 = 'cookiecutter'
    var_8 = 'name'
    var_9 = 'world'
    var_10 = {var_8: var_9}
    var_11 = {var_7: var_10}
    var_12 = module_0.Environment()
    var_13 = 'os.getcwd'
    var_14 = 'builtins.open'
    var_15 = 'shutil.copymode'
    var_16 = 'shutil.copyfile'

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project'
    var_1 = True
    var_2 = 'binary_file.bin'
    var_3 = b'\x89PNG\r\n\x1a\n'
    var_4 = 'cookiecutter'
    var_5 = {}
    var_6 = {var_4: var_5}
    var_7 = module_0.Environment()
    var_8 = 'cookiecutter.generate.is_binary'
    var_9 = 'shutil.copyfile'
    var_10 = 'shutil.copymode'

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project'
    var_1 = True
    var_2 = 'existing.txt'
    var_3 = 'existing.txt'
    var_4 = 'existing content'
    var_5 = 'cookiecutter'
    var_6 = {}
    var_7 = {var_5: var_6}
    var_8 = module_0.Environment()
    var_9 = 'os.path.exists'
    var_10 = 'os.path.isdir'
    var_11 = False
    var_12 = True

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project'
    var_1 = True
    var_2 = 'test.txt'
    var_3 = 'content'
    var_4 = 'cookiecutter'
    var_5 = {}
    var_6 = {var_4: var_5}
    var_7 = module_0.Environment()
    var_8 = 'os.path.isdir'
    var_9 = 'shutil.copyfile'

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project'
    var_1 = True
    var_2 = 'template.txt'
    var_3 = 'Hello {{name}}!'
    var_4 = 'utf-8'
    var_5 = 'cookiecutter'
    var_6 = 'name'
    var_7 = '_new_lines'
    var_8 = 'Alice'
    var_9 = '\n'
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = {var_5: var_10}
    var_12 = module_0.Environment()
    var_13 = 'os.path.isdir'
    var_14 = False
    var_15 = 'os.path.exists'
    var_16 = 'cookiecutter.generate.is_binary'
    var_17 = 'shutil.copymode'
    var_18 = 'project'
    var_19 = 'template.txt'
    var_20 = var_1 / var_19
    var_21 = 'utf-8'
    var_22 = 'Hello Alice!'



